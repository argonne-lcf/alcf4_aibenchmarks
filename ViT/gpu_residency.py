#!/usr/bin/env python3
"""Record proof that the benchmark ran on Intel GPUs, not on CPU fallback.

The ALCF-4 figure of merit is a FLOP rate, and a rate alone cannot tell a
vendor which hardware produced it. A CPU fallback does not announce itself:
torch runs the same model and prints the same TFLOPS line. So each production
run emits this block, and the numbers in it are checkable rather than
asserted.

Four independent signals, in increasing order of how hard they are to fake:

  1. torch.xpu reports the device  -- configuration, the weakest evidence
  2. an allocation shows up in per-device memory -- the tensors are resident
  3. a matmul's result differs bitwise from the same matmul on CPU -- the
     arithmetic was done by a different unit, not merely dispatched
  4. the achieved rate exceeds what the node's CPUs can do -- the decisive
     one. Each Aurora node has 2 Xeon Max sockets, ~4 TFLOPS fp32 peak
     combined; the measured ViT rate is ~774 TFLOPS/node. No dispatch bug
     or mislabeled device can move a CPU two orders of magnitude.

Run standalone inside a job (one rank per tile):

    mpiexec -n 12 --ppn 12 python3 ViT/gpu_residency.py
"""
import json
import os
import sys

# Peak fp32 throughput of one Aurora node's two Xeon CPU Max 9470C sockets:
# 52 cores x 2 sockets x 2.4 GHz x 16 flop/cycle (AVX-512 FMA) ~= 4 TFLOPS.
# Published ViT rates are ~774 TFLOPS/node, ~190x this. The threshold is set
# an order of magnitude above CPU peak so it cannot trip on a fast CPU.
CPU_PEAK_TFLOPS_PER_NODE = 4.0
RANKS_PER_NODE = 12


def _rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        for name in ("PALS_RANKID", "PMIX_RANK", "PMI_RANK", "RANK"):
            if name in os.environ:
                return int(os.environ[name])
        return 0


def _local_rank(rank):
    """Which tile this rank owns.

    This script runs before deepspeed.init_distributed(), so torch.xpu's
    current device is still 0 on every rank -- binding happens inside the
    training script, which takes its device from dist.get_local_rank(). Left
    alone, all 12 ranks would benchmark tile 0 at once and report a rate
    deflated by the contention rather than the tile's real throughput.
    """
    for name in ("PALS_LOCAL_RANKID", "MPI_LOCALRANKID", "LOCAL_RANK",
                 "OMPI_COMM_WORLD_LOCAL_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    # PALS does not always export a local id; ranks are assigned to nodes in
    # contiguous blocks of RANKS_PER_NODE, so derive it.
    return rank % RANKS_PER_NODE


def residency_report():
    """Collect the evidence. Returns a dict; raises nothing."""
    import torch

    rep = {"rank": _rank()}
    rep["torch_version"] = torch.__version__
    rep["torch_xpu_build"] = getattr(torch.version, "xpu", None)
    rep["xpu_available"] = bool(torch.xpu.is_available())

    if not rep["xpu_available"]:
        rep["verdict"] = "NO XPU -- this run would be a CPU fallback"
        return rep

    dev = _local_rank(rep["rank"]) % torch.xpu.device_count()
    torch.xpu.set_device(dev)
    props = torch.xpu.get_device_properties(dev)
    rep["device_index"] = dev
    rep["device_count"] = torch.xpu.device_count()
    rep["device_name"] = torch.xpu.get_device_name(dev)
    rep["total_memory_GiB"] = round(props.total_memory / 2**30, 2)

    # (2) Allocation lands in device memory.
    before = torch.xpu.memory_allocated(dev)
    a = torch.randn(4096, 4096, device=f"xpu:{dev}", dtype=torch.float32)
    b = torch.randn(4096, 4096, device=f"xpu:{dev}", dtype=torch.float32)
    torch.xpu.synchronize()
    rep["alloc_delta_MiB"] = round(
        (torch.xpu.memory_allocated(dev) - before) / 2**20, 1)

    # (3) The arithmetic happens on a different unit than the CPU's. Two IEEE
    # fp32 matmuls of the same inputs agree only if the same accumulation
    # order ran; a GPU's differs. An exact bitwise match would mean the work
    # really was done on the CPU.
    c_xpu = (a @ b).cpu()
    c_cpu = a.cpu() @ b.cpu()
    rep["bitwise_identical_to_cpu"] = bool(torch.equal(c_xpu, c_cpu))
    rep["max_abs_diff_vs_cpu"] = float((c_xpu - c_cpu).abs().max())
    rep["numerically_close_to_cpu"] = bool(
        torch.allclose(c_xpu, c_cpu, rtol=1e-3, atol=1e-3))

    # (4) The rate. 2*n^3 flop for an n x n matmul, timed after a warmup.
    n, iters = 8192, 10
    x = torch.randn(n, n, device=f"xpu:{dev}", dtype=torch.float16)
    y = torch.randn(n, n, device=f"xpu:{dev}", dtype=torch.float16)
    for _ in range(3):
        x @ y
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        x @ y
    end.record()
    torch.xpu.synchronize()
    secs = start.elapsed_time(end) / 1000.0
    tflops = (2.0 * n**3 * iters) / secs / 1e12
    rep["matmul_TFLOPS_fp16_per_tile"] = round(tflops, 1)
    rep["implied_node_TFLOPS"] = round(tflops * RANKS_PER_NODE, 1)
    rep["vs_cpu_peak"] = round(
        tflops * RANKS_PER_NODE / CPU_PEAK_TFLOPS_PER_NODE, 1)

    del a, b, x, y
    torch.xpu.empty_cache()

    exceeds_cpu = rep["implied_node_TFLOPS"] > 10 * CPU_PEAK_TFLOPS_PER_NODE
    distinct_unit = not rep["bitwise_identical_to_cpu"]
    rep["verdict"] = (
        "GPU CONFIRMED" if (exceeds_cpu and distinct_unit)
        else "SUSPECT -- see fields above")
    return rep


def main():
    try:
        rep = residency_report()
    except Exception as e:  # never let the proof block kill a benchmark run
        print(f"[gpu-residency rank {_rank()}] ERROR: {type(e).__name__}: {e}",
              flush=True)
        return 0

    rank = rep["rank"]
    if rank == 0:
        print("=" * 72, flush=True)
        print("GPU residency evidence", flush=True)
        print("=" * 72, flush=True)
        for k, v in rep.items():
            print(f"  {k:.<34} {v}", flush=True)
        print("=" * 72, flush=True)
    else:
        # One compact line per rank, so all 12 tiles are on the record.
        print(f"[gpu-residency rank {rank}] {json.dumps(rep)}", flush=True)

    return 0 if rep.get("verdict") == "GPU CONFIRMED" else 1


if __name__ == "__main__":
    sys.exit(main())
