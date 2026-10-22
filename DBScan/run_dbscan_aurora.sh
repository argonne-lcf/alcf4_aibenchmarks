#!/bin/bash -l
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -N dbscan

# ALCF-4 DBSCAN benchmark on Aurora, 12 ranks per node (one per PVC tile).
#
# Node count comes from the submission, not from this file, so one script
# covers the whole scaling study:
#
#   qsub -l select=1   -q debug         -A <project> DBScan/run_dbscan_aurora.sh
#   qsub -l select=8   -q debug-scaling -A <project> DBScan/run_dbscan_aurora.sh
#   qsub -l select=64  -q prod          -A <project> DBScan/run_dbscan_aurora.sh
#
# Override defaults with -v, e.g.  qsub -v REPS=5,IMPL=fdbscan ...
#
# Three settings below are required to get a valid FOM; see README.md:
#   --verbose        "total time" (the FOM denominator) is printed only with it
#   --n 669921876    NOT ...875: the generator floors the cube root and
#                    pow(669921875, 1./3.) == 874.9999999999997, so the exact
#                    cube silently gives 874^3/rank
#   --impl           fdbscan (as published) runs out of GPU memory at this size
#                    under ArborX 2.x; fdbscan-densebox fits and is faster

cd "${PBS_O_WORKDIR}"
source "${PBS_O_WORKDIR}/DBScan/env_aurora.sh"

# Path to the built benchmark. Override with -v EXE=/path/to/exe
: "${EXE:=${PBS_O_WORKDIR}/build-arborx/benchmarks/cluster/ArborX_Benchmark_DistributedDBSCAN.exe}"
: "${N_PER_RANK:=669921876}"
: "${IMPL:=fdbscan-densebox}"
: "${EPS:=5}"
: "${NUM_SEQ:=50}"
: "${SPACING:=10}"
: "${REPS:=3}"

if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: benchmark not found at ${EXE}" >&2
    echo "Build it per DBScan/install.md, or pass -v EXE=/path/to/exe" >&2
    exit 1
fi

NNODES=$(wc -l < "${PBS_NODEFILE}")
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# One rank per tile, 8 cores each. Without OMP_NUM_THREADS every rank spawns
# one host thread per node core (208) onto those 8.
export CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "=== nodes: ${NNODES}  ranks: ${NRANKS}  impl: ${IMPL}  n/rank: ${N_PER_RANK} ==="
echo "=== host: $(hostname)  date: $(date) ==="

for (( rep=1; rep<=REPS; rep++ )); do
    echo
    echo "########## nodes=${NNODES} rep=${rep}/${REPS} ##########"
    mpiexec --no-transfer -np "${NRANKS}" --ppn "${RANKS_PER_NODE}" \
            --cpu-bind="${CPU_BIND}" gpu_tile_compact.sh \
        "${EXE}" --eps "${EPS}" --dimension 3 --n "${N_PER_RANK}" \
                 --num-seq "${NUM_SEQ}" --spacing "${SPACING}" \
                 --impl "${IMPL}" --verbose
    echo "=== nodes=${NNODES} rep=${rep} exit: $? ==="
done
