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

- Fork: [argonne-lcf / jtchilders Megatron-DeepSpeed_ViT](https://github.com/jtchilders/Megatron-DeepSpeed_ViT), branch `aurora-2026`
- Upstream: [Eugene29/Megatron-DeepSpeed_ViT](https://github.com/Eugene29/Megatron-DeepSpeed_ViT)

It is vendored here as the submodule `ViT/Megatron-DeepSpeed_ViT`:

```shell
git submodule update --init ViT/Megatron-DeepSpeed_ViT
```

The fork carries three Aurora-2026 changes, all documented in its git history:

1. **IPEX import made optional.** `intel_extension_for_pytorch` was a mandatory
   separate package on older Aurora stacks. Its XPU support is now upstreamed
   into `torch.xpu` and the standalone package is gone, so the import is
   best-effort rather than fatal.
2. **`torchinfo` import made optional.** It printed a parameter-count table and
   is not part of the benchmark; a missing diagnostic should not stop training.
3. **Positional encoding built in fp32.** See
   [Correctness notes](#correctness-notes) — this one changes results, not just
   whether the job starts.

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

| seq | factor | image | SP | timed iters |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 16 | 256³ | 1 | 40 |
| 13.8K | 24 | 384³ | 1 | 40 |
| 32.8K | 32 | 512³ | 1 | 40 |
| 262K | 64 | 1024³ | 4 | 20 |
| 1.73M | 120 | 1920³ | 24 | 20 |

`factor` is the number of patches along each edge, so tokens = `factor³` and
attention work grows as `factor⁶`. **120 rather than 128** because
`120³ = 2⁹·3³·5³` divides evenly by both 12 and 24 (Aurora has 12 ranks per
node), while `128³ = 2²¹` has no factor of 3 and cannot — the model asserts
that sequence length is divisible by the sequence-parallel width.

Sequence parallelism is DeepSpeed **Ulysses**, which splits one sequence across
SP ranks and exchanges attention heads with `all_to_all`. SP=4 keeps the 262K
point within a tile's memory; SP=24 spans exactly two Aurora nodes for the
1.73M point.

## Environment
Aurora's `frameworks/2026.1.0` module supplies the whole stack; **no venv is
needed**, and the one `mds_launch.sh` used to activate
(`/lus/flare/projects/Aurora_deployment/eku/venv/vit`) is dead — it was built
against `/opt/aurora/24.347.0`, which no longer exists, so `import torch`
inside it fails.

| component | version |
| --- | --- |
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

### 2. A single node count
```shell
qsub -l select=<N> -q <queue> -A <project> ViT/run_vit_aurora.sh
```

Runs the full five-point sweep on whatever nodes the job has, **skipping**
(not failing) points that do not fit: SP wider than the world, a world not
divisible by SP, or fewer nodes than the point needs to finish in a sane
walltime. `FORCE_ALL=1` overrides the last. Options via `-v`: `FACTORS="32_64"`
(underscores — `qsub -v` cannot carry spaces), `REP=n`, `NUM_ITERS=n`,
`RESULTS_DIR=/path`.

### 3. The weak-scaling series
```shell
./ViT/submit_scaling_aurora.sh              # dry run, prints the plan
./ViT/submit_scaling_aurora.sh --submit     # queue it
```

One job per (node count, repetition), default 1→2048 doubling x 3 reps, with
queue and walltime sized per node count.

### 4. Summarize
```shell
python3 ViT/summarize_results.py [results_dir] [--warmup N] [--csv out.csv]
```

Parses the per-iteration lines, discards the first 5 iterations as warmup (the
rate is flat from about iteration 6), and prints steady-state and weak-scaling
tables with the spread so you can see when that assumption fails.

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
   peak near 4 TFLOPS fp32; the measured ViT rate is ~774 TFLOPS/node, ~190x.
   No dispatch bug moves a CPU two orders of magnitude.

Run it standalone inside a job with
`mpiexec -n 12 --ppn 12 python3 ViT/gpu_residency.py`.

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
optimizer steps is less work.

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
fp16, ZeRO-1, micro-batch 1, FlashAttention on, with the positional-encoding
fix applied.

### Single node

| factor | tokens | SP | s/iter | spread | TFLOPS/node | TFLOPS/tile | mem/tile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 32,768 | 1 | 5.238 | 0.6% | 773.9 | 64.49 | 22.38 GiB |
| 32 | 32,768 | 4 | 1.46 | — | 692.0 | 57.66 | 9.71 GiB |
| 64 | 262,144 | 4 | 80.9 | 0.2% | 675.4 | 56.28 | 45.31 GiB |

Iteration-to-iteration spread is well under 1%, so few repetitions are needed
for a stable number. Fixed startup overhead is ~27 s per run.

`factor=64` at SP=1 is a useful negative control: it exits with
`UR_RESULT_ERROR_OUT_OF_RESOURCES` before the first iteration, confirming that
262K tokens genuinely do not fit on one tile unsharded and that the sequence
parallelism is doing real work rather than being decorative.

### Weak scaling

_Pending — the 1→2048 node series has not been run yet. This section will hold
the full table and the per-sequence weak-scaling efficiencies produced by
`summarize_results.py`._

## Link to prior experimental results
https://argonnedoe-my.sharepoint.com/:x:/r/personal/vsastry_anl_gov/Documents/3D_ViT_benchmarks_experiments.xlsx?d=w697311c7877e492ca135d882927524a7&csf=1&web=1&e=QpfU9s
