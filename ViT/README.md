# Vision Transformer

## Overview
A 3D Vision Transformer (3D ViT) is an extension of the Vision Transformer
(ViT) architecture designed to process 3D data, such as videos, and volumet-
ric images (e.g. medical scans like CT or MRI). These kind of vision models
are specifically important to ALCF users targeting APS volumetric data such
as processing BCDI data or spatio-temporal inputs from climate applications.
Specifically, we stress-test the backbone model of VIT which also generalizes to
newer VIT variants. We use a transformer based no-mask encoder to processes
the sequence of 3D patch embeddings using self-attention mechanisms and feed-
forward layers. We can extend the work to include the decoder architecture in the future.

The benchmark's distinguishing stress is **sequence length**, not parameter
count. The model is fixed at 315M parameters while the token count sweeps from
4K to 1.73M. Because attention cost is quadratic in sequence length, the
longest point does roughly 180,000x the attention work of the shortest, and the
two longest points do not fit in one tile's memory at all — they require
sequence parallelism. That is the capability this benchmark is really
measuring.

## Code Access
The benchmark runs on a fork of Megatron-DeepSpeed maintained for ALCF-4:

- Fork: [jtchilders/Megatron-DeepSpeed_ViT](https://github.com/jtchilders/Megatron-DeepSpeed_ViT), branch `aurora-2026`
- Upstream: [Eugene29/Megatron-DeepSpeed_ViT](https://github.com/Eugene29/Megatron-DeepSpeed_ViT)

It is vendored here as the submodule `ViT/Megatron-DeepSpeed_ViT`:

```shell
git submodule update --init ViT/Megatron-DeepSpeed_ViT
```

The fork's Aurora-2026 changes, each explained in its git history. None
changes the model or the work it does; each makes a point run, or run
correctly, that did not before.

| change | why |
| --- | --- |
| IPEX and `torchinfo` imports made optional | IPEX's XPU support is now in `torch.xpu` and the package is gone; `torchinfo` only printed a table |
| Positional encoding built in fp32 | fp16 position indices produced NaN at 262K+ tokens — see [Correctness notes](#correctness-notes) |
| Data generated per rank | the source rank allocated the whole global batch: 52.7 GiB at 8 nodes for factor 120 |
| Only this rank's positional-encoding shard built | the full table was a ~15 GiB transient at 1.73M tokens |
| Volume cut into its slab before patchifying | patchify made a full 13 GiB second copy on every SP rank |
| Toy images seeded per SP group, not broadcast | a 13 GiB off-node broadcast failed in the network stack (`cxil_map: write error`) |
| Per-run state overridable (`DS_TAG`, `--hostfile`, `MASTER_ADDR`) | lets several runs share one PBS allocation — see [ensembles](#2-the-weak-scaling-series-ensembles) |
| `VIT_HEADS` and `LOG_INTERVAL` knobs | 12 heads stays the default; see [the head count is fixed](#the-head-count-is-part-of-the-benchmark) |

## Dataset
Synthetic. The benchmark generates random volumetric data and labels on the
fly (`pretrain_vision_classify.py`), so there is nothing to stage and no
filesystem dependency in the timed region. Image edge length is
`16 x factor` voxels, cut into 16-voxel cubic patches, giving `factor^3`
tokens.

## Figure-of-merit (FOM)
The detailed modeling and projection report is at: [ViT-Performance_Modeling.pdf](https://github.com/argonne-lcf/alcf4_aibenchmarks/blob/main/ViT/ViT_modeling_and_projection.pdf)
 ```math
 FOM = \frac{ b_sL(N^2 d+6 d^2 N)}{T}
```
where
 ```math
N = H.D.W / p^3
```

```bash
where b_s = global batch size
      L = Number of layers
      N = sequence length
      H,D,W = height, width and depth of the input image.
      p = patch size
      d = hidden dimension of the model
      T = Time to solution.
```

The run configuration is fixed by `alcf4_bench_polaris.sh` in the submodule,
which is the canonical ALCF-4 definition: `VIT="315M"` (14 layers, hidden 1536,
FFN 4096, 12 heads), `VIT3D=1`, fp16, ZeRO stage 1, micro-batch 1, FlashAttention
on. Five sequence lengths are swept:

| seq | factor | image | SP (Aurora) | timed iters |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 16 | 256³ | 1 | 40 |
| 13.8K | 24 | 384³ | 1 | 40 |
| 32.8K | 32 | 512³ | 1 | 40 |
| 262K | 64 | 1024³ | 4 | 20 |
| 1.73M | 120 | 1920³ | 12, with activation recompute | 20 |

`factor` is the number of patches along each edge, so tokens = `factor³` and
attention work grows as `factor⁶`. **120 rather than 128** because
`120³ = 2⁹·3³·5³` divides evenly by both 12 and 24 (Aurora has 12 ranks per
node), while `128³ = 2²¹` has no factor of 3 and cannot — the model asserts
that sequence length is divisible by the sequence-parallel width.

Sequence parallelism is DeepSpeed **Ulysses**, which splits one sequence across
SP ranks and exchanges attention heads with `all_to_all`. The global batch
`b_s` is one sample per data-parallel group, `nodes x 12 / SP`, so it grows
with the job — this is weak scaling.

### What a vendor may and may not change
The rule is: **keep the algorithm, change the computer science freely.**

- **Fixed — the math.** The model (14 layers, hidden 1536, FFN 4096,
  **12 attention heads**, 16-voxel patches) and the five sequence lengths.
  These define what is being computed.
- **Free — how it is computed.** Sequence-parallel degree and scheme,
  activation recompute, ZeRO stage, kernels, communication library,
  process layout. The values used here are what worked on Aurora, not
  requirements.

fp16 is the reference precision; bf16 was measured as a cross-check (see
[fp16 vs bf16](#fp16-vs-bf16)).

### The head count is part of the benchmark
Neither the 308,281,344 parameters nor the FOM formula has a head term — the
QKV and MLP projections are the same matrices at any head count — so it is
tempting to treat heads as a free parameter. It is not. Head count sets
`head_dim` (1536 / 12 = 128), which changes the attention computation and its
measured speed. On Aurora, FlashAttention at `head_dim` 64 (24 heads) runs
1.3–1.75x slower than at 128 (12 heads):

| factor | 12 heads, TFLOPS/tile | 24 heads, TFLOPS/tile | 12 heads faster by |
| ---: | ---: | ---: | ---: |
| 16 | 62.9 | 49.4 | 1.27x |
| 24 | 66.9 | 43.1 | 1.55x |
| 32 | 65.1 | 37.6 | 1.73x |
| 64 | 56.3 | 32.2 | 1.75x |

(One node, same FOM formula.) A 24-head run is therefore a different model and
must not be reported as this one. The fork's default is 12; `VIT_HEADS`
exists only for experiments, and the run scripts tag any non-12-head log
`_h<heads>` so `summarize_results.py` never mixes the two.

This is what decides the SP choice at factor 120. Ulysses shards heads across
the SP group and asserts `num_attention_heads % SP == 0`, so SP must divide
12. `alcf4_bench_polaris.sh` asks for SP=24 with 12 heads, which cannot be
built as written. Aurora runs the point at **SP=12** — one node's 12 tiles —
which needs activation recompute to fit: 27.7 GiB peak per tile with it.

### Recompute is not credited
The FOM counts model FLOPs only. Recompute repeats forward work, which costs
time but adds nothing to the numerator, so it lowers the FOM. Megatron prints
two rates and they differ exactly here:

```
 iteration        1/       1 | ... | TFLOPs: 60.93 |              <- per tile, includes recompute
iteration:1  time:1411.02  TFLOPS:548.34  TFLOPS_per_gpu:45.7     <- the FOM: per node, per tile
```

Report the second. `summarize_results.py` reads only that line. At factor 120
the difference is 25%: an implementation that avoided recompute — selective
recompute, a different SP layout, more memory per device — and kept the same
per-FLOP rate would approach ~61 TFLOPS/tile rather than 45.7. That headroom
is the vendor's to take; the 27.7 GiB peak leaves 36 GiB of each 64 GiB tile
unused.

## Environment
Aurora's `frameworks/2026.1.0` module supplies the whole stack; **no venv is
needed**, and the one `mds_launch.sh` used to activate
(`/lus/flare/projects/Aurora_deployment/eku/venv/vit`) is dead — it was built
against `/opt/aurora/24.347.0`, which no longer exists, so `import torch`
inside it fails.

| component | version |
| --- | --- |
| Aurora software stack | `/opt/aurora/26.181.0` |
| GPU compute runtime | 25.18.33578.77 |
| python | 3.12.12 |
| torch | 2.13.0a0+gitcf30153 (Intel-patched, XPU built in) |
| deepspeed | 0.19.3 |
| transformers / einops / mpi4py | 5.14.1 / 0.8.2 / 4.1.2 |

Deliberately absent, and why that is fine:

- `intel_extension_for_pytorch` — upstreamed into `torch.xpu`.
- `oneccl_bindings_for_pytorch` — replaced by torch's native **XCCL** backend;
  DeepSpeed detects the absence and selects `xccl`.
- `apex` — only imported on the CUDA path.
- `wandb` — runs set `WANDB_MODE=disabled`.

`ViT/env_aurora.sh` does the module setup and sets `PYTHONPATH`. There is
nothing to compile: unlike DBSCAN, this benchmark has no build step.

## Scripts

| script | runs on | does |
| --- | --- | --- |
| `smoketest_aurora.sh` | PBS, 1 node | staged environment checks, then 3 training iterations |
| `submit_ensemble_aurora.sh` | login node | plans and submits the weak-scaling series as ensemble jobs |
| `run_ensemble_aurora.sh` | PBS | runs several points at once on disjoint slices of one allocation |
| `run_vit_aurora.sh` | PBS | runs the sweep (or a subset) on all of a job's nodes |
| `submit_scaling_aurora.sh` | login node | one job per point; for a handful of single points |
| `check_logs.sh` | anywhere | per-log triage: trained? clean exit? NaN? GPU fault on which host? |
| `summarize_results.py` | anywhere | steady-state and weak-scaling tables from the logs |
| `test_ensemble_aurora.sh` | login node | tests `run_ensemble_aurora.sh` with a stub launcher, no GPUs |
| `gpu_residency.py` | PBS | evidence the work ran on the GPUs (called by the run scripts) |
| `check_posenc_fp16.py` | anywhere | reproduces the positional-encoding analysis, no GPU |

All submit commands are run from the repository root; jobs locate the repo
through `PBS_O_WORKDIR`. Logs land in `ViT/results/` (ignored by git) as
`vit_n<nodes>_factor<f>_sp<sp>[_h<heads>]_rep<r>.log`.

## Steps to Run

### 1. Smoke test (1 node, debug queue)
Validates the environment before spending allocation. It runs 17 staged checks
— XPU visibility, accelerator and comm-backend selection, both import guards,
FlashAttention JIT build, Ulysses `DistributedAttention`, an fp16 matmul, a
12-rank XCCL group, and `all_reduce`/`all_to_all` correctness — the last three
on all 12 ranks, then the GPU-residency block and a 3-iteration training run.

```shell
qsub -l select=1 -q debug -A <project> ViT/smoketest_aurora.sh
```

Options via `-v`: `SKIP_TRAIN=1`, `NUM_ITERS=N`, `factor=N`, `SP=N`, `bf16=1`.

The `all_to_all` check matters more than it looks: it is the Ulysses
primitive, and it has no upstream XPU CI coverage.

### 2. The weak-scaling series (ensembles)
```shell
./ViT/submit_ensemble_aurora.sh --project <project>             # dry run, prints the plan
./ViT/submit_ensemble_aurora.sh --project <project> --reps 2 --submit
```

**Why ensembles.** Aurora's queues leave a gap that one-point-per-job
submission cannot cross:

| queue | nodes | walltime | queued jobs |
| --- | --- | --- | --- |
| `debug` | 1–2 | 1 h | 1 per user |
| `debug-scaling` | ≤256 | 1 h | 1 per user |
| `prod` (routes to small/medium/large) | ≥256 | up to 12–24 h | 10 per project |

Per-iteration time is set by sequence length and SP, not node count — more
nodes add data-parallel replicas, not speed — so factor 120 needs ~7.8 h for
its 20 iterations at *every* node count, and below 256 nodes no queue accepts
a job that long. The short points fit `debug-scaling`, but one queued job per
user turns ~100 of them into days of drip-feeding.

An ensemble requests one large `prod` allocation and runs many points in it
concurrently, each `mpiexec` on its own disjoint slice of `PBS_NODEFILE`. One
weak-scaling curve, 1+2+4+…+256 nodes, is 511 nodes, so the default plan is a
512-node job per sequence point per repetition:

```
# batch: 1:16:40 2:16:40 ... 256:16:40 1:24:40      (512 of 512 nodes, 00:10:00)
# batch: 2:24:40 ... 256:24:40 1:32:40              (511 of 512 nodes, 00:14:00)
# batch: 2:32:40 ... 256:32:40                      (510 of 512 nodes, 00:14:00)
# batch: 1:64:20 ... 256:64:20                      (511 of 512 nodes, 00:43:00)
# batch: 1:120:20 ... 256:120:20                    (511 of 512 nodes, 09:57:00)
```

Members are `nodes:factor:iters`. Walltime is sized per job from its slowest
member. Points share a job only when their run times are within 1.5x, so no
member's nodes idle for most of the job — factor 120 is always alone. Members
of one ensemble do not share nodes, so they do not contend for GPUs or NICs;
they do share the system network, as any concurrent jobs would.

Options: `--nodes "1 2 ... 256"`, `--factor "16 24 32 64 120"`, `--iters N`
(default: canonical per point), `--reps N`, `--rep-start N`, `--alloc N`,
`--walltime`, `--exclude "host ..."`. Expect long queue waits for the
factor-120 jobs: in October 2026 a 512-node, 10-hour `prod` job waited more
than a day to start.

**Re-running a single member** — e.g. the 256-node points of repetition 2 —
uses the same tool with a 256-node allocation:

```shell
./ViT/submit_ensemble_aurora.sh --project <project> --nodes 256 --factor "32 64" \
    --alloc 256 --rep-start 2 --exclude x4407c5s5b0n0 --submit
```

`--exclude` passes `EXCLUDE_NODES` to the runner, which drops those hosts from
the node file before slicing, and adds one spare node per excluded host to the
request so every member keeps its full size whether or not a bad host was
assigned. See [Known issues](#known-issues-on-aurora).

### 3. Check the logs
```shell
./ViT/check_logs.sh                    # all of ViT/results
./ViT/check_logs.sh ViT/results/vit_n256_*.log
```

```
log                                            iters   trained   exit skip  nan  problem
vit_n256_factor32_sp1_rep1.log                 40/40       yes  clean    0    0
vit_n256_factor64_sp4_rep1.log                 20/20       yes   hung    0    0
vit_n256_factor64_sp4_rep2.log.gpufault-x4407c5s5b0n0    0/?         NO      -    0    0  GPU page fault on x4407c5s5b0n0 rank 578
vit_n256_factor64_sp4_rep2.log.lossscale       10/20        NO      -    5    0  fp16 loss scale collapsed

GPU page faults by host (consider EXCLUDE_NODES):
        1 x4407c5s5b0n0
```

**Judge a run by its log, not by the PBS exit status.** Runs often finish
training and then fail to exit (see [Known issues](#known-issues-on-aurora));
PBS then reports a failure over complete data. A run is good when it shows
`trained yes`, `skip 0` and `nan 0`; `hung` only means the timing data were
complete before the process stalled. `skip` counts optimizer steps skipped
for fp16 overflow after warmup — two in the first five iterations are normal
while the loss scale settles, any later one is not.

### 4. Summarize
```shell
python3 ViT/summarize_results.py [results_dir] [--warmup N] [--csv out.csv]
```

Parses the per-iteration `iteration:N time:… TFLOPS:…` lines (written every 5
iterations), discards the first 5 iterations as warmup — the rate is flat from
about iteration 6 — and averages the rest. Repetitions are averaged together;
`spread` is the worst within-repetition (max − min) / mean, so you can see when
the warmup assumption or the system is noisy. Runs are keyed by
(nodes, factor, SP, heads), and a weak-scaling series is never formed across
different SP or head counts. A log with optimizer steps skipped after warmup
is rejected and listed, never averaged: a skipped step is less work, so such
a run reports a healthy-looking rate while not training (see
[Correctness notes](#correctness-notes)).

### Alternatives to ensembles
A single job running the sweep on all of its nodes:

```shell
qsub -l select=<N> -q <queue> -A <project> ViT/run_vit_aurora.sh
```

It runs each point in turn, skipping (not failing) any whose SP does not fit
the job's ranks. Options via `-v`: `FACTORS="32_64"` (underscores — `qsub -v`
cannot carry spaces), `REP=n`, `NUM_ITERS=n`, `RESULTS_DIR=/path`. It has no
exit watchdog, so a run that hangs at exit holds the job until its walltime.

`./ViT/submit_scaling_aurora.sh [--submit]` submits that script once per
(node count, point, repetition), choosing `debug`, `debug-scaling` or `prod`
by size and walltime, and skipping factor 120 below 256 nodes. Useful for a
few points; for a whole series, use ensembles.

### Testing changes to the ensemble runner
```shell
./ViT/test_ensemble_aurora.sh
```

Runs `run_ensemble_aurora.sh` on a login node against a stub `mds_launch.sh`
with members that exit cleanly, hang after training, ignore SIGTERM, and fail
— plus an excluded host — and checks each is handled. About 45 s, no job.

## Known issues on Aurora

### Runs that finish training but do not exit
In 27 of 70 trained 12-head runs, the log reached its last iteration and
`[after training is done]`, then `mpiexec` did not return for 10–22+ minutes
— against 3–26 s for a clean exit. Within an ensemble, the members that
stalled tended to be those finishing a little after the first member exited,
and the stalled members were released together. The cause is not yet known. All timed iterations are
logged before the stall, so no data are lost, but a stalled run holds its
nodes and can run a job into its walltime (PBS exit -29).

`run_ensemble_aurora.sh` therefore carries an **exit watchdog**. A member
still running `EXIT_GRACE` seconds (default 180) after logging
`after training is done` is sent SIGTERM to its process group — `mpiexec`
forwards it to every rank — and SIGKILL 30 s later if it is still there. The
job summary reports such a member as
`trained; stopped at exit by the watchdog`, not as a failure. A member that
dies *before* the marker is still reported `FAILED`.

### A node with a faulty GPU: x4407c5s5b0n0
This node was in three of our 512-node jobs, and in all three the rank on its
local device 2 died at the first training step with the same GPU page fault:

```
Segmentation fault from GPU at 0xff010009f5bf1000, ctx_id: 1 (CCS)
  type: 0 (NotPresent), level: 2 (PDP), access: 1 (Write), banned: 1, aborting.
Abort was called at 288 line in file:
.../intel-compute-runtime-25.18.33578.77/shared/source/os_interface/linux/drm_neo.cpp
x4407c5s5b0n0.hsn.cm.aurora.alcf.anl.gov: rank 590 died from signal 6 and dumped core
```

Only the member containing that node failed; the other ~510 nodes, and the
same configurations on other nodes in other repetitions, ran normally. It was
still in service on 2026-10-27. Until it is repaired, pass
`--exclude x4407c5s5b0n0` (or `-v EXCLUDE_NODES=x4407c5s5b0n0` to the runner).
`check_logs.sh` lists every host that produced a GPU page fault, so a new bad
node shows up the same way: the same host and local rank across jobs is the
signature of hardware, not code.

### Nodes that break fp16 training
Four runs never trained: from the first steps every gradient overflowed, the
fp16 loss scale halved each step (4096 → 128 by iteration 5, 4 or 8 by
iteration 10, where a healthy run settles at 1024 after two skips) and
DeepSpeed stopped with
`Exception: Current loss scale already at minimum - cannot decrease scale anymore`.
Grad norm read 0.000 and no NaN iterations were logged; only the skipped-step
count shows it. Iteration times were normal throughout.

Model initialisation and the synthetic data use fixed seeds, so every
repetition of a configuration should follow nearly the same loss curve —
and the same configurations on other nodes trained cleanly, with no skipped
steps after warmup. What the failing runs share is nodes:

| runs that collapsed | nodes in common, and in no healthy run |
| --- | --- |
| 32-node factor 24 and factor 32 (24-head, two jobs) | x4318c2s3b0n0, x4614c4s1b0n0, x4614c4s2b0n0, x4713c5s2b0n0–x4713c5s7b0n0, x4713c6s0b0n0–x4713c6s2b0n0, x4713c7s5b0n0, x4713c7s7b0n0 |
| 256-node factor 16 (24-head) and 256-node factor 64 rep 2 (12-head) | x4208c4s6b0n0, x4208c4s7b0n0, x4208c5s0b0n0, x4208c5s1b0n0 |

A rank on a faulty device that produces inf or garbage gradients would do
exactly this: the overflow check runs after the all-reduce, so one bad tile
makes every rank skip, and no host is named. The evidence narrows the
suspects to these 18 nodes but not to one each, so we exclude them all:

```shell
--exclude "x4407c5s5b0n0 x4318c2s3b0n0 x4614c4s1b0n0 x4614c4s2b0n0 \
x4713c5s2b0n0 x4713c5s3b0n0 x4713c5s4b0n0 x4713c5s5b0n0 x4713c5s6b0n0 \
x4713c5s7b0n0 x4713c6s0b0n0 x4713c6s1b0n0 x4713c6s2b0n0 x4713c7s5b0n0 \
x4713c7s7b0n0 x4208c4s6b0n0 x4208c4s7b0n0 x4208c5s0b0n0 x4208c5s1b0n0"
```

With them excluded, the 256-node factor-64 repetition 2 trained cleanly, and
its grad norms matched repetition 1's to the third decimal (26.217 vs 26.216
at iteration 10, 26.948 at iteration 20) — confirming that healthy nodes
reproduce the same trajectory.

To find suspects after a new collapse, intersect the collapsed members' node
files in `results/nodefiles_<job>/` and remove every host that appears in a
healthy member's file.

### Queue mechanics worth knowing
- `small`, `medium` and `large` are `from_route_only`: submit to `prod` and
  let it route. A direct `qsub -q small` is refused.
- PBS copies the job script at `qsub`, so editing `run_*.sh` does not affect
  jobs already queued. `env_aurora.sh` and the submodule's `mds_launch.sh`
  are read when the job *runs* — do not edit them while jobs are queued.
- A rank that dies on a GPU fault can drop a core file of tens of GB into the
  working directory. Check for `core.*` files after a failure; a few of them
  will fill a `/home` quota.

## Verifying it ran on the GPUs
A FLOP rate does not say which hardware produced it — a CPU fallback runs the
same model and prints the same TFLOPS line. Every run therefore records its own
evidence via `ViT/gpu_residency.py`, written to
`results/gpu_residency_n<N>_rep<R>.log`:

1. `torch.xpu` reports the device (configuration only — weakest).
2. An allocation appears in per-device memory (tensors are resident).
3. A matmul differs **bitwise** from the same matmul on CPU — a different
   arithmetic unit ran it, not merely a different dispatch label.
4. The achieved rate against CPU peak. An Aurora node's two Xeon Max sockets
   peak near 4 TFLOPS fp32. The probe's own dense fp16 matmul measures
   182-191 TFLOPS per tile (~2,200 TFLOPS/node, 545x CPU peak), and the ViT
   benchmark itself sustains ~800 TFLOPS/node. No dispatch bug moves a CPU
   two orders of magnitude.

Measured output, one node:

```
  device_name....................... Intel(R) Data Center GPU Max 1550
  device_count...................... 12
  total_memory_GiB.................. 63.98
  bitwise_identical_to_cpu.......... False
  matmul_TFLOPS_fp16_per_tile....... 181.7
  implied_node_TFLOPS............... 2180.6
  vs_cpu_peak....................... 545.2
  verdict........................... GPU CONFIRMED
```

Run it standalone inside a job with
`mpiexec -n 12 --ppn 12 python3 ViT/gpu_residency.py`.

Note the probe binds its own tile from the launcher's local rank. It runs
before `deepspeed.init_distributed()`, and the training script takes its
device from `dist.get_local_rank()` later, so without that binding all 12
ranks would land on tile 0 and report a rate deflated by contention
(~14 TFLOPS/tile instead of ~190).

## Correctness notes

### The positional encoding had to be fixed before any long-sequence number was meaningful
`vit_backbone.py` hardcoded `torch.float16` for the **position index** of the
sinusoidal encoding. fp16 spaces consecutive integers exactly only up to 2048
and has no finite value above 65504, so at this benchmark's sequence lengths
the table was wrong in two escalating ways:

| factor | tokens | distinct positions | consequence |
| ---: | ---: | ---: | --- |
| 16 | 4,096 | 3,073 | 25% of adjacent tokens positionally identical |
| 24 | 13,824 | 4,801 | 65% identical |
| 32 | 32,768 | 6,145 | 81% identical |
| 64 | 262,144 | 7,168 | 196,624 positions overflow to inf → **NaN** |
| 120 | 1,728,000 | 7,168 | 1,662,465 overflow → **NaN** |

So the two longest points — the ones sequence parallelism exists to reach — fed
NaN position vectors into the model. **And they still reported a healthy
62.26 TFLOPS/tile while doing it.** The fp16 loss scale collapsed to its
minimum, grad norm read 0.000, and every optimizer step was skipped; a
benchmark that only times iterations would have recorded that as a good result.
Worth noting the direction of the error: after the fix the same configuration
reports *56.27* TFLOPS/tile. The broken run scored **higher**, because skipping
optimizer steps is less work. This is why `check_logs.sh` reports skipped
and NaN iterations next to the timings, and why `summarize_results.py`
rejects any run with skipped steps after warmup. Watch `skip` rather than
`nan`: a collapsing loss scale can skip every step while logging zero NaN
iterations (see the node issue under
[Known issues](#nodes-that-break-fp16-training)).

The same line also broke `--bf16` outright: the table came back fp16 against a
bf16 model, and `bf16 + fp16` promotes to fp32, so LayerNorm received fp32
activations against bf16 weights (`RuntimeError: expected scalar type Float but
found BFloat16`).

The fix builds the table in fp32 — where `sin`/`cos` should be evaluated
anyway, and where `div_term` avoids underflowing fp16 subnormals — then casts
once to the model dtype. Outputs lie in [-1, 1], which both fp16 and bf16 hold
comfortably. `ViT/check_posenc_fp16.py` reproduces the whole analysis with no
GPU and no job, so the table above is checkable rather than asserted:

```shell
python3 -I ViT/check_posenc_fp16.py
```

### Known differences from pre-2026 published numbers
- **RMSNorm is not fused.** `torch.xpu.IpexRmsNorm` does not exist without
  IPEX, so the model takes the plain RMSNorm path. This is a real performance
  difference, not just a cosmetic one.
- **`CCL_*` tuning variables are carried over untested.** They were tuned for
  the oneCCL bindings, which this stack no longer uses. They are kept so runs
  match `mds_launch.sh`'s intent; their effect under native XCCL is unknown.
- **`--num-workers 12` exceeds the dataloader's own recommendation** of 7 given
  `OMP_NUM_THREADS=7`. Left at the canonical value; the data is synthetic, so
  the loader is not on the critical path.

## Results on Aurora

Measured on `frameworks/2026.1.0`, 12 ranks per node (6 PVC GPUs x 2 tiles),
12 heads, fp16, ZeRO-1, micro-batch 1, FlashAttention on, with the
positional-encoding fix applied. Weak-scaling numbers are the mean of two
repetitions run as ensembles (October 2026), first 5 iterations discarded.

### FOM by node count

Aggregate FOM in TFLOPS (the numerator of the FOM formula over measured time,
summed over all tiles):

| nodes | factor 16 (4K) | factor 24 (13.8K) | factor 32 (32.8K) | factor 64 (262K) | factor 120 (1.73M) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 755.1 | 803.2 | 780.6 | 675.5 | pending |
| 2 | 1,403.8 | 1,582.3 | 1,542.6 | 1,335.7 | pending |
| 4 | 2,719.6 | 3,153.0 | 3,046.2 | 2,676.8 | pending |
| 8 | 5,272.2 | 6,191.2 | 6,126.1 | 5,324.0 | pending |
| 16 | 10,530.5 | 12,379.2 | 12,262.0 | 10,618.9 | pending |
| 32 | 20,370.9 | 24,531.4 | 24,210.7 | 21,109.1 | pending |
| 64 | 38,334.7 | 48,312.1 | 48,053.0 | 41,919.0 | pending |
| 128 | 48,114.9 | 88,597.1 | 94,294.2 | 84,072.2 | pending |
| 256 | 78,071.7 | 166,261.1 | 184,968.5 | 164,382.9 | pending |

The 256-node factor-64 repetition 2 took three attempts: the first landed on
x4407c5s5b0n0, the second collapsed its fp16 loss scale, and the third ran
with the suspect nodes excluded (see
[Known issues](#nodes-that-break-fp16-training)).

### Weak-scaling efficiency

TFLOPS per tile at 1 node, and efficiency relative to it:

| nodes | factor 16 | factor 24 | factor 32 | factor 64 |
| ---: | ---: | ---: | ---: | ---: |
| 1 (TFLOPS/tile) | 62.9 | 66.9 | 65.1 | 56.3 |
| 2 | 93% | 99% | 99% | 99% |
| 4 | 90% | 98% | 98% | 99% |
| 8 | 87% | 96% | 98% | 99% |
| 16 | 87% | 96% | 98% | 98% |
| 32 | 84% | 95% | 97% | 98% |
| 64 | 79% | 94% | 96% | 97% |
| 128 | 50% | 86% | 94% | 97% |
| 256 | 40% | 81% | 93% | 95% |

Iteration time at 1 node is 0.19 s (factor 16), 1.12 s (24), 5.2 s (32) and
81 s (64). Efficiency tracks it: the longer an iteration, the more easily the
data-parallel gradient all-reduce of 308M parameters can hide behind compute.
Factor 16 falls off sharply past 64 nodes. With 0.2 s iterations it is also
the noisiest point — within-run spread of 30–74% at several node counts — so
read its large-scale numbers as communication-bound and approximate. The
long-sequence points, the ones the benchmark exists for, hold 93–95% at 256
nodes.

### Factor 120 (1.73M tokens)

At SP=12 with activation recompute, one node runs one sample in **1411 s per
iteration**, a FOM of **548.3 TFLOPS/node (45.7 TFLOPS/tile)**, with a
27.7 GiB peak per tile (single-iteration measurement). The full 1→256-node
curve, two repetitions, is queued as two 512-node ensembles; this section will
hold it.

For reference only — not the benchmark model — the earlier 24-head
configuration at SP=24 (two nodes per sample, no recompute) ran
1080 s/iteration at 29.7 TFLOPS/tile, flat from 16 to 128 nodes (99%
efficiency). Its lower rate despite no recompute is the `head_dim` 64 penalty
above.

### Single node, other configurations

| factor | tokens | SP | s/iter | TFLOPS/node | TFLOPS/tile | mem/tile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 32,768 | 1 | 5.238 | 773.9 | 64.49 | 22.38 GiB |
| 32 | 32,768 | 4 | 1.46 | 692.0 | 57.66 | 9.71 GiB |
| 64 | 262,144 | 4 | 80.9 | 675.4 | 56.28 | 45.31 GiB |

SP=4 at factor 32 shows the cost of sequence parallelism where it is not
needed: 11% lower rate, for 2.3x less memory per tile.

`factor=64` at SP=1 is a useful negative control: it exits with
`UR_RESULT_ERROR_OUT_OF_RESOURCES` before the first iteration, confirming that
262K tokens genuinely do not fit on one tile unsharded and that the sequence
parallelism is doing real work rather than being decorative.

### fp16 vs bf16

The canonical configuration is fp16. bf16 was measured as a cross-check once
the positional encoding was fixed — both now train cleanly at 262K tokens, and
fp16 is the faster of the two, so there is no reason to depart from the
canonical definition:

| precision | s/iter | TFLOPS/tile | loss | grad norm | skipped | NaN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 80.9 | 56.28 | 3.71 | 24.4 | 0 | 0 |
| bf16 | 86.6 | 52.56 | 3.40 | 24.8 | 0 | 0 |

bf16 remains available via `-v bf16=1` on the smoke test.

## Link to prior experimental results
https://argonnedoe-my.sharepoint.com/:x:/r/personal/vsastry_anl_gov/Documents/3D_ViT_benchmarks_experiments.xlsx?d=w697311c7877e492ca135d882927524a7&csf=1&web=1&e=QpfU9s
