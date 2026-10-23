#!/bin/bash
# Build the ALCF-4 DBSCAN benchmark (ArborX) on Aurora, from nothing.
#
#   ./DBScan/build_aurora.sh [build_dir]
#
# Clones ArborX at the pinned commit, configures it against the prebuilt
# Kokkos module, and builds the single benchmark executable. Run it on a
# login node; no GPU is needed to build. A clean run took ~13 minutes, most
# of it compiling the SYCL device code ahead-of-time for PVC.
#
# Default build_dir is ./dbscan-build relative to where you run this.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-${PWD}/dbscan-build}"

# Pinned to the commit the published results in README.md were measured with.
# ArborX master moves; a later commit can change memory behaviour or
# performance with no warning, so reproducing those numbers means building
# this hash. Override with ARBORX_COMMIT=master (or any ref) to build
# something else -- the benchmark prints the ArborX and Kokkos versions at
# startup, so record what you actually built.
: "${ARBORX_REPO:=https://github.com/arborx/ArborX.git}"
: "${ARBORX_COMMIT:=826d457}"
: "${BUILD_JOBS:=16}"

SRC="${BUILD_DIR}/source-arborx"
BLD="${BUILD_DIR}/build-arborx"
EXE="${BLD}/benchmarks/cluster/ArborX_Benchmark_DistributedDBSCAN.exe"

echo "=== DBSCAN benchmark build ==="
echo "build dir : ${BUILD_DIR}"
echo "ArborX    : ${ARBORX_REPO} @ ${ARBORX_COMMIT}"
echo

mkdir -p "${BUILD_DIR}"

echo "--- 1/4: modules ---"
# shellcheck source=env_aurora.sh
source "${HERE}/env_aurora.sh"
echo "KOKKOS_ROOT = ${KOKKOS_ROOT:-<unset>}"
echo "BOOST_ROOT  = ${BOOST_ROOT:-<unset>}"
if [[ -z "${KOKKOS_ROOT:-}" || -z "${BOOST_ROOT:-}" ]]; then
    echo "ERROR: the Kokkos or Boost module did not set its root variable." >&2
    echo "Check that 'module load kokkos/5.1.1-sycl boost' succeeded." >&2
    exit 1
fi

echo
echo "--- 2/4: source ---"
if [[ -d "${SRC}/.git" ]]; then
    echo "reusing existing clone at ${SRC}"
else
    # Not --depth 1: a shallow clone of the default branch cannot check out an
    # arbitrary older commit.
    git clone "${ARBORX_REPO}" "${SRC}"
fi
git -C "${SRC}" checkout --quiet "${ARBORX_COMMIT}"
echo "ArborX at $(git -C "${SRC}" describe --always)"

echo
echo "--- 3/4: configure ---"
cmake -S "${SRC}" -B "${BLD}" \
    -D Kokkos_ROOT="${KOKKOS_ROOT}" \
    -D Boost_ROOT="${BOOST_ROOT}" \
    -D CMAKE_CXX_COMPILER=icpx \
    -D CMAKE_CXX_EXTENSIONS=OFF \
    -D ARBORX_ENABLE_MPI=ON \
    -D ARBORX_ENABLE_GPU_AWARE_MPI=ON \
    -D ARBORX_ENABLE_BENCHMARKS=ON

echo
echo "--- 4/4: build (this is the slow part) ---"
# Just the one target: the full benchmark suite takes considerably longer.
cmake --build "${BLD}" --target ArborX_Benchmark_DistributedDBSCAN.exe \
      --parallel "${BUILD_JOBS}"

if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: build reported success but ${EXE} is missing" >&2
    exit 1
fi

echo
echo "=== build complete ==="
echo "executable: ${EXE}"
echo "size      : $(du -h "${EXE}" | cut -f1)"
echo
echo "Next: submit the scaling study with"
echo "  ./DBScan/submit_scaling_aurora.sh -A <project> -b ${BUILD_DIR}"
