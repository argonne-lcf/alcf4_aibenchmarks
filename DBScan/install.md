# Installation instructions

The benchmark is a part of the
[ArborX](https://github.com/arborx/ArborX/) benchmark suite. Here, we provide instructions for building ArborX on Aurora.

**To just build it, run [`build_aurora.sh`](build_aurora.sh):**

```shell
./DBScan/build_aurora.sh $HOME/dbscan-build
```

That script does everything on this page — loads the modules, clones ArborX at
the pinned commit `826d457`, configures and builds the single benchmark target.
The rest of this page explains what each step does and what goes wrong without
it, which is what you need if you are adapting this to another system or
debugging a build that failed.

## Kokkos

Aurora now provides prebuilt Kokkos modules with the SYCL backend and the PVC
architecture already enabled, so a source build is no longer necessary:

```shell
source DBScan/env_aurora.sh
```

which does:

```shell
module restore
module load oneapi/release/2026.1.0
module load kokkos/5.1.1-sycl
module load boost
module load cmake
```

Notes:
- Use `module restore`, **not** `module purge`. `boost` is provided under the
  `mpich` + `intel-oneapi-compilers` hierarchy; purging unloads `mpich` and
  `boost` then becomes unfindable.
- `module load cmake` matters: the default `/usr/bin/cmake` is older than the
  `cmake/3.31.11` the module provides.
- `kokkos/5.1.1-sycl` is available under both `oneapi/release/2025.3.1` and
  `2026.1.0`. An older `kokkos/4.7.02-sycl` exists but only under
  `oneapi/release/2025.3.1`, and it is built as C++17 while current ArborX
  requires C++20 — prefer 5.1.1.

Loading these sets `KOKKOS_ROOT` and `BOOST_ROOT`, which the ArborX configure
step below consumes.

<details>
<summary>Building Kokkos from source (only if you need a specific version)</summary>

```shell
git clone --depth 1 --branch 4.5.01 https://github.com/kokkos/kokkos.git source-kokkos
cmake -S source-kokkos -B build-kokkos \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -D CMAKE_INSTALL_PREFIX=$PWD/install-kokkos \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_CXX_COMPILER=icpx \
    -D CMAKE_CXX_EXTENSIONS=OFF \
    -D CMAKE_CXX_FLAGS="-ffp-model=precise -fsycl-device-code-split=per_kernel" \
    -D CMAKE_EXE_LINKER_FLAGS="-ffp-model=precise" \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_SYCL=ON \
    -D Kokkos_ARCH_INTEL_PVC=ON
cmake --build build-kokkos --parallel 16
cmake --install build-kokkos
```

Then pass `-D Kokkos_ROOT="$PWD/install-kokkos"` below instead of
`"${KOKKOS_ROOT}"`. Do not set `CMAKE_CXX_STANDARD=17`: ArborX 2.x declares
`cxx_std_20`, so Kokkos must be built as C++20 to match.

</details>

## Build ArborX
You can follow the ArborX build instructions
[here](https://github.com/arborx/ArborX/wiki/Build). For Aurora, we built the benchmark using the following steps, which are what [`build_aurora.sh`](build_aurora.sh) automates:
```shell
git clone https://github.com/arborx/ArborX.git source-arborx
git -C source-arborx checkout 826d457
cmake -S source-arborx -B build-arborx \
     -D Kokkos_ROOT="${KOKKOS_ROOT}" \
     -D Boost_ROOT="${BOOST_ROOT}" \
     -D CMAKE_CXX_COMPILER=icpx \
     -D CMAKE_CXX_EXTENSIONS=OFF \
     -D ARBORX_ENABLE_MPI=ON \
     -D ARBORX_ENABLE_GPU_AWARE_MPI=ON \
     -D ARBORX_ENABLE_BENCHMARKS=ON
cmake --build build-arborx --target ArborX_Benchmark_DistributedDBSCAN.exe --parallel 16
```

Building just the `ArborX_Benchmark_DistributedDBSCAN.exe` target avoids
compiling the rest of the benchmark suite. Drop `--target ...` to build
everything.

Two notes on the clone:
- **Not `--depth 1`.** A shallow clone of the default branch cannot check out
  an arbitrary older commit, and `826d457` is not the tip.
- `826d457` is ArborX 2.1.99, the commit the [README.md](README.md) results
  were measured with. Pin it to reproduce those numbers; the benchmark prints
  its ArborX and Kokkos versions at startup, so record what you built if you
  use a different one.

A clean build took about 13 minutes on a login node (clone included, 16-way
parallel) and produces a ~1.2 GB executable. Most of that is compiling the
SYCL device code ahead-of-time for PVC, which is slow here but means there is
no JIT cost at run time.

Nothing above asks for AOT: `-fsycl-targets=spir64_gen -device 12.60.7` comes
from the `kokkos/5.1.1-sycl` module's exported CMake interface and propagates
to the benchmark automatically. Verify with:

```shell
grep -rhoE '\-fsycl-targets=[^;" ]*|\-device [0-9.]+' \
    "${KOKKOS_ROOT}"/lib*/cmake/Kokkos/*.cmake | sort -u
```

A Kokkos built without those flags produces a much smaller executable that
JIT-compiles on first use — the per-run startup cost then becomes real, and
`NEO_CACHE_PERSISTENT=1` starts to matter (with this module it is a no-op;
see [README.md](README.md#repetitions-and-run-to-run-variation)).

# Running the benchmark

The benchmark is SYCL-only, so it must run on a compute node — there are no
GPUs on the Aurora login nodes.

```shell
cd build-arborx/benchmarks/cluster
export CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads
mpiexec --no-transfer -np 12 --ppn 12 --cpu-bind=${CPU_BIND} gpu_tile_compact.sh \
   ./ArborX_Benchmark_DistributedDBSCAN.exe --eps 5 --dimension 3 --n 669921876 --num-seq 50 --spacing 10 --impl fdbscan-densebox --verbose
```

Set `OMP_NUM_THREADS`: the Kokkos build has an OpenMP host backend as well as
SYCL, and without it every rank spawns 208 host threads (one per node core)
onto the 8 cores `--cpu-bind` gives it, and prints a 12-fold
`Kokkos::OpenMP::initialize WARNING: You are likely oversubscribing your CPU
cores`.

[`run_dbscan_aurora.sh`](run_dbscan_aurora.sh) wraps all of the above. Node
count comes from the submission, so one script covers the scaling study:

```shell
qsub -l select=1  -q debug         -A <project> DBScan/run_dbscan_aurora.sh
qsub -l select=8  -q debug-scaling -A <project> DBScan/run_dbscan_aurora.sh
qsub -l select=64 -q prod          -A <project> DBScan/run_dbscan_aurora.sh
```

Defaults are overridable with `-v`, e.g. `-v REPS=5,IMPL=fdbscan`. Submitted by
hand like this, the jobscript looks for `env_aurora.sh` and the executable
relative to `$PBS_O_WORKDIR`, so submit from the top of the repository, or pass
`-v ENV_SH=...,EXE=...` explicitly.

For the full study, use [`submit_scaling_aurora.sh`](submit_scaling_aurora.sh)
instead — it handles queue routing, the one-queued-job limit on the debug
queues, and the `filesystems` request. See
[README.md](README.md#steps-to-run).

`--verbose` is **required to collect the FOM**. The `total time` line — $T$ in
the FOM — is only printed when it is set; without it the run completes but
reports no timing.

Add `--verify` to check correctness. It is far slower than the benchmark
itself, so use it on a small `--n` only, not on a timing run.

The generator floors the cube root of `--n`, so the value given is rounded down
to the nearest cube. Pass `669921876`, **not** `669921875`: although
$875^3 = 669921875$ exactly, `std::pow(669921875, 1./3.)` evaluates to
`874.9999999999997` in floating point, so the floor yields $874^3$ and you
silently get a smaller problem than the results table in [README.md](README.md)
reports. The extra `+1` absorbs the rounding error.

## Problem size limits

With ArborX 2.1.99 the default `fdbscan` implementation **runs out of GPU
memory** at the benchmark size, aborting with
`UR_RESULT_ERROR_OUT_OF_RESOURCES` from the Level Zero backend. Measured on one
node (12 ranks, one per PVC tile), largest working size:

| `--n` per rank | fdbscan |
| ---: | :--- |
| $699^3$ = 341,532,099 | runs |
| $839^3$ = 590,589,719 | runs |
| $874^3$ = 667,627,624 | out of memory |
| $875^3$ = 669,921,875 | out of memory |

Use `--impl fdbscan-densebox` to reach the full benchmark size. It is the
lower-memory implementation and, at this size, also the faster one.
