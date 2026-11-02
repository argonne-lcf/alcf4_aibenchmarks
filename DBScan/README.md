# DBSCAN

## Overview
Clustering is a data mining technique that splits a set of objects into disjoint
classes (clusters), each containing similar objects. DBSCAN (Density-Based
Spatial Clustering of Applications with Noise) is a popular clustering algorithm
[1], particularly useful when the number of clusters or their shape is not known a
priori. It is used in a diverse set of applications such as bioinformatics,
noise filtering and outlier detection, cosmology, image segmentation, and
others.

DBSCAN defines local density by the number of points within a given radius.
Dense neighborhoods form clusters, while the rest of the points in low-density
regions are considered outliers (noise).

This benchmark uses the distributed version of the DBSCAN algorithm, with the
local algorithm described in [2]. The benchmark is implemented within
[ArborX](https://github.com/arborx/ArborX) library, a performance portable
geometric search library that uses [Kokkos](https://kokkos.org) for on-node
parallelism. Kokkos supports CPUs and GPUs from Nvidia, AMD, and Intel through
backends such as OpenMP, CUDA, SYCL, and HIP. For Aurora GPUs, we use SYCL backend.

DBSCAN has been implemented in multiple ML related libraries: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html),
[RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/api/#dbscan), [R](https://cran.r-project.org/web/packages/dbscan/index.html).

### Science Motivation
ArborX was originally developed as part of the Software Techonolgies (ST) thrust
within Exascale Computing Project (ECP) with the primary goal to support DOE
applications at scale. One exascale application that ArborX supports is Hardware
Accelerated Cosmology Code (HACC), a cosmoloy framework developed at Argonne.
ArborX is a critical part of of the HACC's analysis tools, which uses DBSCAN [3]
and spherical overdensity algorithms.
In addition to HACC, there are at least eight additional applications using ArborX in
production listed [here](https://github.com/arborx/ArborX/wiki/AppsUsingArborx).

**Halo finding in cosmology:** one of the recurring analysis steps during
cosmological N-body simulations is identifying halos (regions with a high
density of dark matter particles). HACC framework uses ArborX's implementation
of DBSCAN to find halos. Here, the dataset to cluster is composed of the 3D
positions of particles [3].

**Intelligent subsampling of computational fluid dynamics (CFD) datasets:** training
reduced-order machine learned models on data from CFD simulations can be
challenging because of the large number grid points used by the solver to
describe the solution fields. This leads to very large sample sizes, increasing
the computational cost of training the models. Brewer, et al. designed a method
for intelligent subsampling CFD data by clustering the mesh points based on flow
descriptor variables and drawing samples from the clusters that maximize the
entropy of the cluster probability distributions [4].

**Identifying cell populations in single-cell analysis:** single-cell data, such as
data from mass cytometry, can quickly become large. FastPG was designed to
quickly cluster millions of cells in order to identify cell populations [5]. It
was funded by ECP through ExaGraph and is integrated into
[MCMICRO](https://mcmicro.org/), a pipeline that transforms multi-channel
whole-slide images into single-cell data [6].


## Code Access
This benchmark is part of the open source
[ArborX](https://github.com/arborx/ArborX) library. The executable is `/benchmarks/cluster/ArborX_Benchmark_DistributedDBSCAN.exe`.

## Dataset
This benchmark uses three-dimensional synthetic data generated on each rank. The data generation code is in ArborX at [/benchmarks/cluster/distributed_data.hpp](https://github.com/arborx/ArborX/blob/master/benchmarks/cluster/distributed_data.hpp). Here is a sketch of the steps:
1. Factorize `comm_size` (the number of ranks) into three factors so that each rank can be assigned a different region in 3D space. For example, for 12 ranks, `factors = [3, 2, 2]` and for 48 ranks, `factors = [4, 4, 3]`.
2. Assign each rank indices `Is` representing how the ranks are arranged in 3D space. For example, for 12 ranks, rank 0 is assigned `Is = [0, 0, 0]` and rank 11 is assigned `Is = [2, 1, 1]`.
3. For simplicity, update `n`, the requested number of points per rank, to be a cube (`n = nx^3` for some integer `nx`). For example, for `--n 400000000`, `n` is updated to `n = 398688256 = 736^3`.
4. On each rank, generate `n` points with a Kokkos parallel for loop.
   - In each direction (`x, y, z`), there are `num_seq` points in a row followed by user-defined spacing. For example, if `num_seq = 50` and `spacing = 10`, the possible x coordinates are `x = 0, 1, 2, ..., 49, 59, 60, 61, ..., 108, 118, ...` The `pos` function handles adding the spacing. 
   - This means that if `1 < eps < spacing`, there are clear clusters with `num_seq^3` points. For example, for `num_seq = 50`, each cluster has 125000 points.
   - For `num_seq = 50`, the first cluster consists of the 125000 points where `x = 0, 1, 2, ..., 49`, `y = 0, 1, 2, ..., 49`, and `z = 0, 1, 2, ..., 49`.
   - The indices `Is` separate the ranks in 3D space. For example, for rank 0, `Is = [0, 0, 0]`, and for rank 1, `Is = [1, 0, 0]`, so the data generated on ranks 0 and 1 have the same y and z coordinates but separate x coordinates. Specifically, the largest x coordinate on rank 0 is `pos(nx-1)` and the x coordinates on rank 1 start at `pos(nx)`. Depending on the values of `nx`, `num_seq`, and `spacing`, the gap between `pos(nx-1)` and `pos(nx)` could be 1 (meaning that a cluster spans rank 0 and rank 1 in the x dimension), or there could be a gap of `spacing`.
     
## Figure-of-merit (FOM)
 ```math
 FOM = \frac{nd}{T}
```
where $n$ is the number of points to cluster, $d$ is their dimension, and $T$
is the time to solution. FOM is highly dependent on the data distribution and
the DBSCAN parameters $minPts$ and $\varepsilon$. Here we consider $minPts = 2$,  $\varepsilon = 5$, and the data distribution described above.

### Results on Aurora

Measured October 2026 with **ArborX 2.1.99 (commit `826d457`) and Kokkos
5.1.1**, using `--impl fdbscan-densebox`, at $n = 669921876 \approx 875^3$
three-dimensional points per rank, $\varepsilon = 5$, $minPts = 2$,
`num_seq = 50`, `spacing = 10`. One MPI rank per PVC tile, 12 ranks per node.

$T$ is the median of 10 timed repetitions per node count (11 run, the first
discarded; see [Repetitions](#repetitions-and-run-to-run-variation) below).

| Number of Nodes | n (global) | T (seconds) | FOM | Weak-scaling eff. |
| -----------: | -----------: | ---: | ---: | ---: |
| 1 | 8,039,062,500 | 9.722 | 2,480,681,701 | 93% |
| 2 | 16,078,125,000 | 8.805 | 5,478,377,534 | 103% |
| 4 | 32,156,250,000 | 9.082 | 10,621,972,033 | 100% |
| 8 | 64,312,500,000 | 8.871 | 21,748,013,301 | 102% |
| 16 | 128,625,000,000 | 9.130 | 42,264,512,596 | 99% |
| 32 | 257,250,000,000 | 9.056 | 85,219,743,816 | 100% |
| 64 | 514,500,000,000 | 8.575 | 180,000,000,000 | 106% |
| 128 | 1,029,000,000,000 | 8.617 | 358,224,543,081 | 105% |
| 256 | 2,058,000,000,000 | 8.677 | 711,577,248,891 | 105% |
| 512 | 4,116,000,000,000 | 8.957 | 1,378,509,628,803 | 101% |
| 1024 | 8,232,000,000,000 | 8.853 | 2,789,562,860,047 | 103% |
| 2048 | 16,464,000,000,000 | 9.792 | 5,044,117,647,059 | 93% |

Weak scaling holds the work per rank fixed, so perfect scaling means constant
$T$ and the efficiency column is $T_{ref}/T(n)$, not a speedup. $T_{ref}$ is
the median over the 1–4 node runs rather than the single 1-node run, which is
the least representative point: with one node there is no inter-node
communication, and that run was ~10% slower with 3–5× the run-to-run scatter
of every other size. Values slightly above 100% are real, not rounding.

Scaling is flat to 1024 nodes and degrades to 93% at 2048 — the first real
departure, in a single step from 1024. The printed step breakdown localizes
it: from 1024 to 2048 nodes the local DBSCAN kernel is unchanged (6.595 →
6.596 s), while step 3 (convert local to global labels) goes 0.259 → 0.941 s
and step 6 (communicate merge pairs) 0.315 → 0.491 s. Those two account for
0.86 s of the 0.94 s increase. The compute is scaling; the label-reconciliation
phase is not.

**4096 nodes was attempted and is not in the table.** The job (submitted
22 Oct, started 24 Oct after ~54 hours in the queue) completed only its warmup
repetition (15.708 s). One node then stopped responding (`ping failed ... No
reply ... after 97s`), and repetitions 2–11 all failed to launch
(`Couldn't forward RPC launch ... Resource temporarily unavailable`, exit 127).
A lone warmup is the run the statistics exclude, so there is no 4096 result to
report; `summarize_results.py` skips warmup-only files for this reason. The
warmup itself was in line with the trend at 1024 and 2048 nodes (see
[Repetitions](#repetitions-and-run-to-run-variation)), so nothing in it points
to the node that failed afterward.

> **This table is not comparable to results published before October 2026.**
> Those used `--impl fdbscan`, which no longer fits in GPU memory at this
> problem size under ArborX 2.x. See
> [Implementation: why `fdbscan-densebox`](#implementation-why-fdbscan-densebox).

### Implementation: why `fdbscan-densebox`

ArborX provides two implementations of the same algorithm. **They compute
identical clusters** and differ only in how neighbors are found:

- **`fdbscan`** inserts all $n$ points into a bounding-volume hierarchy and
  issues one radius query per point.
- **`fdbscan-densebox`** first overlays a uniform grid of cell size
  $h = \varepsilon/\sqrt{d}$, chosen so that *any two points in the same cell
  are necessarily within $\varepsilon$ of each other*. A cell holding at least
  $minPts$ points is a "dense box": its points are known to be core points and
  mutually connected without any distance test, and the whole cell enters the
  tree as a single box instead of as individual points.

On this benchmark's data the generator produces tight clusters, and the run
output shows **100% of points falling into dense cells** — about 31 million
boxes in the tree instead of 670 million points per rank, a ~22× reduction.
That is why densebox is both faster and far lighter on memory here. On
uniformly scattered data the advantage would shrink, since few cells would be
dense.

This matters because the two no longer perform equivalently at benchmark size:

| Implementation | n (global) | T (seconds) | Result |
| :--- | -----------: | ---: | :--- |
| `fdbscan` (ArborX 1.7.99, as published pre-2026) | 8,039,062,500 | 21.527 | ran |
| `fdbscan` (ArborX 2.1.99) | 8,039,062,500 | — | **out of memory** |
| `fdbscan-densebox` (ArborX 2.1.99) | 8,039,062,500 | 9.722 | ran |

Under ArborX 2.x, `fdbscan` aborts with `UR_RESULT_ERROR_OUT_OF_RESOURCES`
from the Level Zero backend on most ranks. Measured ceiling on one node:
$839^3$ = 590,589,719 points per rank runs; $874^3$ = 667,627,624 does not.
Reaching the full benchmark size on a current build therefore requires
`--impl fdbscan-densebox`.

**Report `fdbscan-densebox`.** A system with more memory per GPU may be able
to run `fdbscan` as well; those numbers are interesting but are not what this
table reports.

### Repetitions and run-to-run variation

Each job runs the benchmark 11 times and the first repetition is discarded.
Repetition 1 pays one-off costs the others do not — paging in a ~1.2 GB
executable, allocator and MPI connection setup — and including it biases both
the median and the spread. It is still recorded, and `summarize_results.py`
reports it as a `run-1` ratio against the mean of the rest.

As measured by `total time`, that ratio was 1.00–1.52× with no clear trend in
node count (1.52× at 256 nodes, 1.17× at 2048). That understates the warmup
cost at scale. `total time` is rank 0's clock (see
[What $T$ measures](#what-t-measures)), and at 512, 2048 and 4096 nodes the
slowest rank was still in step 1 of the warmup after rank 0 had finished.
Step 1 (receive global neighbors) is where the warmup cost lands:

| Nodes | warmup step 1, mean over ranks | warmup step 1, slowest rank | timed reps, slowest rank |
| ---: | ---: | ---: | ---: |
| 1–64 | 1.5–2.0 s | 1.8–4.5 s | 1.5–2.2 s |
| 128 | 3.1 s | 4.9 s | 1.5 s |
| 256 | 2.7 s | 6.0 s | 1.5 s |
| 512 | 2.8 s | 15.0 s | 1.6 s |
| 1024 | 3.6 s | 10.3 s | 1.6 s |
| 2048 | 6.4 s | 18.4 s | 1.6 s |
| 4096 | 11.9 s | 39.7 s | — |

From 1024 nodes up the warmup's step 1 roughly doubles per doubling of node
count, while the timed repetitions stay flat. We have not isolated the cause;
first-use setup of point-to-point connections to remote ranks would fit the
pattern, but we have not tested that. None of this reaches the results table:
in the timed repetitions the slowest rank never outlasted rank 0. It is not
SYCL JIT: the build is
ahead-of-time compiled for PVC (`-fsycl-targets=spir64_gen -device 12.60.7`,
inherited from the Kokkos module), and with `NEO_CACHE_PERSISTENT=1` against
an empty cache directory, 12 ranks × 6 repetitions wrote zero cache files.

Across the timed repetitions the coefficient of variation was 0.2–1.0% at
every node count except 1 node (2.3%), so 10 repetitions are more than enough
to resolve differences of a few percent.

### What $T$ measures

`total time` is the ArborX `DistributedDBSCAN::total` profiling region — the
clustering call itself. It **excludes** MPI and Kokkos initialization, data
generation, binary load, and teardown; data generation completes before the
timed region opens.

It is **rank 0's elapsed time**, not the slowest rank's: the benchmark prints
rank 0's local timer. Most of the algorithm's exchanges are with neighboring
ranks rather than global collectives, so rank 0 can finish while a distant rank
is still working. We checked: in every timed repetition at every node count,
the slowest rank's longest single step was at most 0.79× rank 0's total, so
rank 0 was not finishing ahead of a straggler. It did happen in some warmup
repetitions (see [below](#repetitions-and-run-to-run-variation)). A system with
a slow or unhealthy node can therefore report an optimistic $T$; compare the
`MaxOverProcs` column of the step breakdown against `total time` to catch it.

It **includes** the distributed algorithm's communication and label
reconciliation, which is the point. With `--verbose` the benchmark prints a
per-step breakdown (min / mean / max over ranks); at the benchmark size the
median over timed repetitions is:

| Step | 1 node | 2048 nodes |
| :--- | ---: | ---: |
| 1. receive global neighbors | 1.769 s | 1.325 s |
| 2. local DBSCAN | 6.540 s | 6.596 s |
| 3. convert local to global labels | 0.344 s | 0.941 s |
| 4. communicate labels back to owning ranks | 0.763 s | 0.307 s |
| 5. process multi-labeled indices | 0.051 s | 0.057 s |
| 6. communicate merge pairs | 0.262 s | 0.491 s |
| 7. flatten labels | 0.038 s | 0.069 s |
| **total** | **9.722 s** | **9.792 s** |

The local DBSCAN kernel is 67% of $T$ at both ends, and the explicitly
communicating steps (1, 4, 6) are 29% at 1 node and 22% at 2048. A FOM
computed from step 2 alone would be about 1.5× larger, but it would measure
the kernel rather than the distributed algorithm.

Steps do not sum exactly to the total: each is a mean over ranks while the
total is rank 0's own time, so load imbalance lands in the gap. The imbalance is real
and not investigated here — at 1 node step 1 ranges 1.30–2.37 s across ranks
and step 3 ranges 0.011–1.106 s.

### Problem size

$n$ per rank is fixed at the published value and should not be changed when
reporting a FOM. $T \approx 10$ s is short, which raises the fair question of
whether the problem is large enough to measure properly. A single-node sweep
from 994M to 8.72G points global (12 ranks) says it is:

| n (global) | T (s) | step 2 (local DBSCAN) | T − step 2 | points/s/rank |
| ---: | ---: | ---: | ---: | ---: |
| 994,582,272 | 4.538 | 0.817 | 3.721 | 18.3 M |
| 2,986,298,268 | 5.921 | 2.335 | 3.586 | 42.0 M |
| 5,564,217,888 | 8.325 | 4.470 | 3.855 | 55.7 M |
| 7,984,063,404 | 11.028 | 6.484 | 4.544 | 60.3 M |
| 8,718,872,388 | 11.735 | 7.096 | 4.639 | 61.9 M |

Apparent throughput rises 3.4× across this range, but the mechanism is in the
last two columns: the local DBSCAN kernel scales almost perfectly linearly
with $n$ (0.817 s → 7.096 s for 8.8× the points), while everything else is a
nearly fixed ~3.6–4.6 s floor. Rising points/s is that fixed cost being
amortized, not the GPU working better.

The benchmark size (8.04G global) sits where the curve has flattened — about
94% of the best throughput measured. Going larger would keep inflating the FOM
by diluting the fixed cost, which makes the number look better without the
hardware doing anything better, and would forfeit comparability with the table
above. Run-to-run variation is already 0.2–1.0%, so a longer run buys no
precision either.

## Steps to Run

The benchmark is part of the [ArborX repository](https://github.com/arborx/ArborX).
The scripts in this directory reproduce the table above on Aurora end to end.
They are specific to Aurora — on another system they are a worked reference
for what has to be set, not a drop-in.

| File | Role |
| :--- | :--- |
| [`env_aurora.sh`](env_aurora.sh) | Module environment; sourced by the other scripts |
| [`build_aurora.sh`](build_aurora.sh) | Clones ArborX at the pinned commit and builds the benchmark |
| [`run_dbscan_aurora.sh`](run_dbscan_aurora.sh) | The PBS jobscript: one node count, 11 repetitions |
| [`submit_scaling_aurora.sh`](submit_scaling_aurora.sh) | Submits one job per node count |
| [`summarize_results.py`](summarize_results.py) | Turns the job outputs into the table above |

### 1. Build

```shell
git clone https://github.com/argonne-lcf/alcf4_aibenchmarks.git
cd alcf4_aibenchmarks
./DBScan/build_aurora.sh $HOME/dbscan-build
```

Run this on a login node; no GPU is needed to build. A clean run took about 13
minutes (clone included) on an Aurora login node — most of it compiling the
SYCL device code ahead-of-time for PVC — and produces a ~1.2 GB executable at
`$HOME/dbscan-build/build-arborx/benchmarks/cluster/ArborX_Benchmark_DistributedDBSCAN.exe`.

ArborX is pinned to commit `826d457`, the commit the table above was measured
with. ArborX master moves, and a later commit can change memory behaviour or
performance without warning. To build something else, set
`ARBORX_COMMIT=master` (or any ref) — the benchmark prints its ArborX and
Kokkos versions at startup, so record what you actually built.

The build environment is in [`env_aurora.sh`](env_aurora.sh):

```shell
module restore            # NOT 'module purge' -- see the note in the script
module load oneapi/release/2026.1.0
module load kokkos/5.1.1-sycl
module load boost
module load cmake
```

Kokkos is the **prebuilt Aurora module**, not a Kokkos built alongside ArborX.
See [install.md](install.md) for what the configure line does and what goes
wrong without each piece.

### 2. Run the scaling study

```shell
./DBScan/submit_scaling_aurora.sh -A <project> -b $HOME/dbscan-build
```

This submits one job per node count — powers of two from 1 up to `-m` (2048 by
default). Report whatever node counts your system reaches; there is no reason
to stop where we did. Useful options:

| Option | Meaning |
| :--- | :--- |
| `-A <project>` | PBS account to charge (required) |
| `-b <dir>` | build directory from step 1 (default `./dbscan-build`) |
| `-n "1 2 4"` | explicit node counts instead of powers of two |
| `-m <max>` | largest node count (default 2048) |
| `-o <dir>` | results directory (default `<build_dir>/results`) |
| `-r <reps>` | repetitions per job (default 11) |
| `-s` | submit serially — wait for each job before submitting the next |
| `-d` | dry run: print the `qsub` commands and submit nothing |

Each job writes `dbscan_n<N>.out` into the results directory. Queue routing is
handled for you and reflects the Aurora limits: ≤2 nodes → `debug`, ≤256 →
`debug-scaling`, above that → `prod`. **`prod` is a routing queue**; it
dispatches to `small` / `medium` / `large` by size, and submitting directly to
`small` is refused with "Access to queue is denied".

`debug` and `debug-scaling` allow only one queued job per user, so the script
always waits for those to finish before submitting the next. Jobs above 256
nodes are submitted all at once so they accrue queue priority in parallel —
expect them to wait. Our 2048-node job sat in the queue for 3 hours before a
4½-minute run. Budget wall-clock for queueing, not for computing.

Walltime is requested as 15 minutes at every size; 11 repetitions measured
2.5–4.5 minutes from 1 to 2048 nodes. **Do not inflate this.** Shorter
requests are scheduled sooner — our 2048-node job (15 min requested) was
dispatched ahead of 512- and 1024-node jobs submitted earlier with 30-minute
requests — and `qalter` on an already-queued job is rejected by a site hook,
so an oversized request cannot be corrected after submission.

### 3. Summarize

```shell
python3 DBScan/summarize_results.py $HOME/dbscan-build/results/dbscan_n*.out
python3 DBScan/summarize_results.py --markdown $HOME/dbscan-build/results/dbscan_n*.out
```

Runs on a login node and only reads text files. The plain form prints median,
mean, standard deviation, CV, IQR, FOM, weak-scaling efficiency and the `run-1`
ratio per node count; `--markdown` emits the results table above, ready to
paste. It warns if the files mix implementations or library versions, which
would make the rows incomparable. Written for the login nodes' python3 (3.6),
with no third-party dependencies.

### Running a single configuration by hand

```shell
source DBScan/env_aurora.sh
export CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads
mpiexec --no-transfer -np 12 --ppn 12 --cpu-bind=${CPU_BIND} gpu_tile_compact.sh \
   ./ArborX_Benchmark_DistributedDBSCAN.exe --eps 5 --dimension 3 --n 669921876 \
   --num-seq 50 --spacing 10 --impl fdbscan-densebox --verbose
```

12 ranks per node is one rank per PVC tile (6 GPUs × 2 tiles);
`gpu_tile_compact.sh` is the Aurora wrapper that pins each rank to its tile.
In the output, `n (global)` is $n$ in the FOM and `total time` is $T$.

Four details that are easy to get wrong, all of which cost us a run:

- `--verbose` is **required**: `total time` is only reported when it is set.
  Without it the run completes and prints no timing at all.
- `--n 669921876`, not `669921875`. The generator floors the cube root, and
  `std::pow(669921875, 1./3.)` is `874.9999999999997` in floating point, so the
  exact cube silently yields $874^3$ per rank — a smaller problem than this
  table reports. Exact cubes are the *worst* choice here, not the safe one.
  Always check the reported `n (global)` against what you intended.
- `--impl fdbscan-densebox` is required on ArborX 2.x at this size; see
  [above](#implementation-why-fdbscan-densebox).
- `OMP_NUM_THREADS=8` keeps each rank from spawning one host thread per node
  core (208) onto the 8 cores `--cpu-bind` gives it.

### Aurora specifics worth recording

- **Allocation.** Jobs are charged node-hours. The full study as configured
  (1→2048 nodes, powers of two) is about 4095 node × 15 min ≈ 1024 node-hours
  requested, considerably less consumed.
- **Filesystems.** A job that touches a filesystem it did not request is held
  by the scheduler rather than failing. `submit_scaling_aurora.sh` derives the
  `-l filesystems=` list from where the executable, scripts and results live.
- **Kernel cache.** `NEO_CACHE_PERSISTENT` makes no difference to this build
  and is deliberately not set; the build is ahead-of-time compiled, and a
  controlled test wrote zero cache files across 12 ranks × 6 repetitions.

## References
[1] [Original DBSCAN paper:](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. "A density-based algorithm for discovering clusters in large spatial databases with noise." In KDD, vol. 96, no. 34, pp. 226-231. 1996.

[2] [Paper on DBSCAN in ArborX:](https://dl.acm.org/doi/10.1145/3605573.3605594)
Andrey Prokopenko, Damien Lebrun-Grandié, and Daniel Arndt. 2023. Fast tree-based algorithms for DBSCAN for low-dimensional data on GPUs. In 52nd International Conference on Parallel Processing (ICPP 2023), August 07–10, 2023, Salt Lake City, UT, USA. ACM, New York, NY, USA, 10 pages.

[3] [Paper on improvements to ArborX to support HACC:](https://journals.sagepub.com/doi/abs/10.1177/10943420241298296)
Prokopenko, Andrey, Daniel Arndt, Damien Lebrun-Grandié, Bruno Turcksin, Nicholas Frontiere, J. D. Emberson, and Michael Buehlmann. "Advances in ArborX to support exascale applications." The International Journal of High Performance Computing Applications 39, no. 1 (2025): 167-176.

[4] [Paper on clustering to subsample CFD meshes:](https://dl.acm.org/doi/pdf/10.1145/3624062.3626084)
Brewer, Wesley, Daniel Martinez, Muralikrishnan Gopalakrishnan Meena, Aditya Kashi, Katarzyna Borowiec, Siyan Liu, Christopher Pilmaier, Greg Burgreen, and Shanti Bhushan. "Entropy-driven Optimal Sub-sampling of Fluid Dynamics for Developing Machine-learned Surrogates." In Proceedings of the SC'23 Workshops of the International Conference on High Performance Computing, Network, Storage, and Analysis, pp. 73-80. 2023.

[5] [Paper on FastPG for clustering cells:](https://www.biorxiv.org/content/10.1101/2020.06.19.159749v2.full.pdf)
Bodenheimer, Tom, Mahantesh Halappanavar, Stuart Jefferys, Ryan Gibson, Siyao Liu, Peter J. Mucha, Natalie Stanley, Joel S. Parker, and Sara R. Selitsky. "FastPG: fast clustering of millions of single cells." BioRxiv (2020): 2020-06.

[6] [Paper on MCMICRO:](https://www.nature.com/articles/s41592-021-01308-y.pdf)
Schapiro, Denis, Artem Sokolov, Clarence Yapp, Yu-An Chen, Jeremy L. Muhlich, Joshua Hess, Allison L. Creason et al. "MCMICRO: a scalable, modular image-processing pipeline for multiplexed tissue imaging." Nature methods 19, no. 3 (2022): 311-315.


