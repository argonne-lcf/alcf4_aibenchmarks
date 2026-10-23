"""Per-rank environment check for the ALCF-4 3D ViT benchmark on Aurora.

Run under mpiexec by smoketest_aurora.sh. Every check prints a PASS/FAIL line
prefixed with the rank so a failure names the dependency that broke instead of
leaving a stack trace to interpret.

Rank 0 prints detail; all ranks participate in the collective checks, since
the point is to exercise XCCL across all 12 tiles of a node.
"""
import os
import sys
import traceback

RANK = int(os.environ.get("PMI_RANK", os.environ.get("RANK", "0")))
WORLD = int(os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", "1")))

_failures = []


def check(name, fn, rank0_only=True):
    """Run fn(); print PASS/FAIL. Returns fn's value, or None on failure."""
    if rank0_only and RANK != 0:
        try:
            return fn()
        except Exception:
            return None
    try:
        val = fn()
        detail = f" -- {val}" if val is not None else ""
        print(f"[rank {RANK}] PASS  {name}{detail}", flush=True)
        return val
    except Exception as e:
        _failures.append(name)
        print(f"[rank {RANK}] FAIL  {name}: {type(e).__name__}: {e}", flush=True)
        if RANK == 0:
            traceback.print_exc()
        return None


def main():
    if RANK == 0:
        print("=" * 72)
        print("ViT environment smoke test")
        print(f"python {sys.version.split()[0]}  world_size={WORLD}")
        print("=" * 72)

    # ---- 1. torch and its XPU build -------------------------------------
    import torch
    check("torch imports", lambda: torch.__version__)
    check("torch XPU runtime present", lambda: torch.version.xpu)
    check("torch.xpu.is_available()",
          lambda: _assert_true(torch.xpu.is_available(), "torch.xpu.is_available() is False"))
    n_tiles = check("torch.xpu.device_count()",
                    lambda: _assert_nonzero(torch.xpu.device_count(), "no XPU devices visible"))

    # The fused RMSNorm path keys off this attribute. It is False without
    # IPEX, which means the model takes the plain rmsnorm path -- a real
    # difference from pre-2026 runs, so record it rather than assert it.
    check("torch.xpu has IpexRmsNorm (informational)",
          lambda: hasattr(torch.xpu, "IpexRmsNorm"))

    # ---- 2. DeepSpeed accelerator selection -----------------------------
    import deepspeed
    check("deepspeed imports", lambda: deepspeed.__version__)
    from deepspeed.accelerator.real_accelerator import get_accelerator
    acc = check("accelerator is xpu",
                lambda: _assert_eq(get_accelerator().device_name(), "xpu"))
    check("comm backend selected",
          lambda: get_accelerator().communication_backend_name())

    # ---- 3. The IPEX guard actually works -------------------------------
    # This is the import that failed before the guard in
    # megatron/model/__init__.py. On the XPU path it is the hard blocker.
    check("megatron.model imports (IPEX guard)",
          lambda: _import_ok("megatron.model"))
    check("megatron.training imports", lambda: _import_ok("megatron.training"))
    check("megatron.model.transformer imports",
          lambda: _import_ok("megatron.model.transformer"))

    # ---- 4. Flash attention builder (FA=1) ------------------------------
    # mds_launch.sh passes --use-flash-attn-builder on Aurora. Importing the
    # builder is not enough -- it JIT-builds on first load, so call load().
    def _fa():
        builder = get_accelerator().get_op_builder("FlashAttentionBuilder")
        mod = builder().load()
        variant = ("v1/flash_attn_func" if hasattr(mod, "flash_attn_func")
                   else "v2/flash_attn_func_v2" if hasattr(mod, "flash_attn_func_v2")
                   else "NEITHER func NOR func_v2")
        if "NEITHER" in variant:
            raise RuntimeError(variant)
        return variant
    check("FlashAttentionBuilder loads", _fa)

    # ---- 5. Ulysses sequence parallelism --------------------------------
    # DistributedAttention is what SP>1 uses; it is the core of this benchmark.
    check("deepspeed Ulysses DistributedAttention",
          lambda: _attr_ok("deepspeed.sequence.layer", "DistributedAttention"))

    # ---- 6. Allocate on device and do real work -------------------------
    def _alloc():
        dev = torch.device("xpu", torch.xpu.current_device())
        a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
        b = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
        c = (a @ b).float().sum().item()
        torch.xpu.synchronize()
        gib = torch.xpu.max_memory_reserved() / 1024 ** 3
        if not (c == c):  # NaN check
            raise RuntimeError("matmul produced NaN")
        return f"fp16 matmul OK, {gib:.2f} GiB reserved"
    check("XPU allocate + fp16 matmul", _alloc, rank0_only=False)

    # ---- 7. Distributed init over XCCL ----------------------------------
    # The real question the login node cannot answer: does a 12-rank
    # process group come up on the native XCCL backend?
    def _dist():
        torch.xpu.set_device(RANK % max(torch.xpu.device_count(), 1))
        deepspeed.init_distributed()
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError("init_distributed() returned but group not initialized")
        return f"backend={dist.get_backend()} rank={dist.get_rank()}/{dist.get_world_size()}"
    check("deepspeed.init_distributed()", _dist, rank0_only=False)

    # ---- 8. Collectives: allreduce and all_to_all -----------------------
    # all_to_all is the Ulysses primitive. If SP is going to work at scale,
    # it has to work here first.
    def _allreduce():
        import torch.distributed as dist
        dev = torch.device("xpu", torch.xpu.current_device())
        t = torch.ones(1024, device=dev) * (RANK + 1)
        dist.all_reduce(t)
        expected = WORLD * (WORLD + 1) / 2
        got = t[0].item()
        if abs(got - expected) > 1e-3:
            raise RuntimeError(f"allreduce wrong: got {got}, expected {expected}")
        return f"sum={got:.0f} correct"
    check("XCCL all_reduce", _allreduce, rank0_only=False)

    def _alltoall():
        import torch.distributed as dist
        dev = torch.device("xpu", torch.xpu.current_device())
        send = torch.full((WORLD, 16), float(RANK), device=dev)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send)
        # row i of recv should hold the value i (sent by rank i)
        for i in range(WORLD):
            if abs(recv[i][0].item() - i) > 1e-3:
                raise RuntimeError(f"all_to_all row {i}: got {recv[i][0].item()}, want {i}")
        return "all_to_all correct (Ulysses primitive)"
    check("XCCL all_to_all_single", _alltoall, rank0_only=False)

    # ---- summary --------------------------------------------------------
    import torch.distributed as dist
    nfail = torch.tensor([len(_failures)], dtype=torch.int32)
    if dist.is_initialized():
        nfail = nfail.to(torch.device("xpu", torch.xpu.current_device()))
        dist.all_reduce(nfail)
    total = int(nfail.item())

    if RANK == 0:
        print("=" * 72)
        if total == 0:
            print("RESULT: all checks PASSED on all ranks")
        else:
            print(f"RESULT: {total} check failure(s) across {WORLD} ranks")
        print("=" * 72)
    return 1 if total else 0


def _assert_true(v, msg):
    if not v:
        raise RuntimeError(msg)
    return v


def _assert_nonzero(v, msg):
    if not v:
        raise RuntimeError(msg)
    return v


def _assert_eq(got, want):
    if got != want:
        raise RuntimeError(f"got {got!r}, expected {want!r}")
    return got


def _import_ok(name):
    __import__(name)
    return "ok"


def _attr_ok(mod, attr):
    m = __import__(mod, fromlist=[attr])
    if not hasattr(m, attr):
        raise RuntimeError(f"{mod} has no {attr}")
    return "present"


if __name__ == "__main__":
    sys.exit(main())
