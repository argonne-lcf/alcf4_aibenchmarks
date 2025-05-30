# Installation instructions

The benchmark is a part of the
[ArborX](https://github.com/arborx/ArborX/) benchmark suite. Here, we provide instructions for building ArborX on Aurora.

## Build Kokkos

Kokkos is available on Aurora. Alternatively, one can build it following ALCF's Kokkos [documentation](https://docs.alcf.anl.gov/aurora/programming-models/kokkos-aurora/#building-a-kokkos-application-using-cmake).
For Aurora, we built Kokkos using the following steps:
```shell
git clone --depth 1 --branch 4.5.01 https://github.com/kokkos/kokkos.git source-kokkos
cmake -S source-kokkos -B build-kokkos \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -D CMAKE_INSTALL_PREFIX=$PWD/install-kokkos \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_CXX_COMPILER=icpx \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_EXTENSIONS=OFF \
    -D CMAKE_CXX_FLAGS="-ffp-model=precise -fsycl-device-code-split=per_kernel" \
    -D CMAKE_EXE_LINKER_FLAGS="-ffp-model=precise" \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_SYCL=ON \
    -D Kokkos_ARCH_INTEL_PVC=ON
cmake --build build-kokkos --parallel 16
cmake --install build-kokkos
```

## Build ArborX
You can follow the ArborX build instructions
[here](https://github.com/arborx/ArborX/wiki/Build). For Aurora, we built the benchmark using the following steps:
```shell
module load boost
module load cmake
git clone --depth 1 https://github.com/arborx/ArborX.git source-arborx
cmake -S source-arborx -B build-arborx
     -D Kokkos_ROOT="$PWD/install-kokkos" \
     -D Boost_ROOT="${BOOST_ROOT}" \
     -D CMAKE_CXX_COMPILER=icpx \
     -D CMAKE_CXX_EXTENSIONS=OFF \
     -D ARBORX_ENABLE_MPI=ON \
     -D ARBORX_ENABLE_GPU_AWARE_MPI=ON \
     -D ARBORX_ENABLE_BENCHMARKS=ON
cmake --build build-arborx --parallel 16
```

# Running the benchmark

```shell
cd build-arborx/benchmarks/cluster
export CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
mpiexec --no-transfer -np 12 --ppn 12 --cpu-bind=${CPU_BIND} gpu_tile_compact.sh \
   ./ArborX_Benchmark_DistributedDBSCAN.exe --eps 5 --dimension 3 --n 669921876 --num-seq 50 --spacing 10
```
