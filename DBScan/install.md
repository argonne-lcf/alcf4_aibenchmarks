# Installation instructions
This benchmark is part of [ArborX](https://github.com/arborx/ArborX/), so ArborX's installation instructions are helpful. However, here are some extra details on installing ArborX on Aurora with the current software environment.

## Build Google Benchmark
```
git clone https://github.com/google/benchmark.git
cd benchmark/
module load cmake
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
```

## Build Kokkos
Kokkos is available on Aurora, or you can build it yourself. (See ALCF's Kokkos documentation [here](https://docs.alcf.anl.gov/aurora/programming-models/kokkos-aurora/#building-a-kokkos-application-using-cmake)).
To test this benchmark on Aurora, we built Kokkos like this:
```
git clone https://github.com/kokkos/kokkos.git
cd kokkos
mkdir build
cd build
cmake\
    -D CMAKE_BUILD_TYPE=RelWithDebInfo\
    -D CMAKE_INSTALL_PREFIX=$HOME/local/opt/aurora/kokkos-4.5.1\
    -D BUILD_SHARED_LIBS=ON\
    -D CMAKE_CXX_COMPILER=icpx\
    -D CMAKE_CXX_STANDARD=17\
    -D CMAKE_CXX_EXTENSIONS=OFF\
    -D CMAKE_CXX_FLAGS="-ffp-model=precise -fsycl-device-code-split=per_kernel"\
    -D CMAKE_EXE_LINKER_FLAGS="-ffp-model=precise"\
    -D Kokkos_ENABLE_SERIAL=ON\
    -D Kokkos_ENABLE_SYCL=ON\
    -D Kokkos_ARCH_INTEL_PVC=ON\
    -D CMAKE_VERBOSE_MAKEFILE=OFF\
    ..
make -j16 -l16 install
```

## Build ArborX
You can follow the ArborX installation instructions [here](https://github.com/arborx/ArborX/wiki/Build). To build this benchmark on Aurora, we more specifically did this:
```
export ARBORX_SOURCE_DIR=$(pwd)
mkdir build; cd build
export ARBORX_INSTALL_DIR=$(pwd)/install
module load boost 
module load cmake
OPTIONS=(\
     -D CMAKE_INSTALL_PREFIX="${ARBORX_INSTALL_DIR}"\
     -D ARBORX_ENABLE_MPI=ON\
     -D Kokkos_ROOT="/path/to/your/install/directory"\
     -D CMAKE_CXX_COMPILER="$(which icpx)"\
     -D CMAKE_CXX_EXTENSIONS=OFF # required by Kokkos\
     -D ARBORX_ENABLE_EXAMPLES=ON\
     -D Boost_ROOT="${BOOST_ROOT}"\
     -D benchmark_ROOT="/path/to/your/install/directory"\
     -D ARBORX_ENABLE_BENCHMARKS=ON\
     )

cmake "${OPTIONS[@]}" "${ARBORX_SOURCE_DIR:-../}"
make -j51
make install
```
