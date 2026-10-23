#!/bin/bash
# Submit the ALCF-4 DBSCAN weak-scaling study on Aurora.
#
#   ./DBScan/submit_scaling_aurora.sh -A <project> [-b <build_dir>] [options]
#
# Submits one job per node count. Each job runs the benchmark 11 times and
# writes a PBS output file into the results directory; summarize_results.py
# turns those files into the table in README.md.
#
# Options:
#   -A <project>   PBS account to charge                        (required)
#   -b <dir>       build directory used by build_aurora.sh      (./dbscan-build)
#   -n "1 2 4 .."  node counts to run              (1 2 4 ... up to -m, below)
#   -m <max>       largest node count to run                          (2048)
#   -o <dir>       results directory               (<build_dir>/results)
#   -r <reps>      repetitions per job                                  (11)
#   -s             submit serially: wait for each job before the next
#   -d             dry run: print the qsub commands, submit nothing
#
# Serial (-s) matters for the small end. The debug-scaling queue permits only
# one queued job per user, so submitting 1..256 all at once gets most of them
# rejected. Jobs above 256 go to the 'prod' routing queue, which has no such
# limit, so those are submitted all at once by default -- they wait hours for
# a scheduling window and should accrue that wait in parallel.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACCT=""
BUILD_DIR="${PWD}/dbscan-build"
NODE_LIST=""
MAX_NODES=2048
OUTDIR=""
REPS=11
SERIAL=0
DRYRUN=0

while getopts "A:b:n:m:o:r:sdh" opt; do
    case "${opt}" in
        A) ACCT="${OPTARG}" ;;
        b) BUILD_DIR="${OPTARG}" ;;
        n) NODE_LIST="${OPTARG}" ;;
        m) MAX_NODES="${OPTARG}" ;;
        o) OUTDIR="${OPTARG}" ;;
        r) REPS="${OPTARG}" ;;
        s) SERIAL=1 ;;
        d) DRYRUN=1 ;;
        h) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "try -h" >&2; exit 2 ;;
    esac
done

if [[ -z "${ACCT}" ]]; then
    echo "ERROR: -A <project> is required (PBS account to charge)." >&2
    echo "try -h" >&2
    exit 2
fi

EXE="${BUILD_DIR}/build-arborx/benchmarks/cluster/ArborX_Benchmark_DistributedDBSCAN.exe"
: "${OUTDIR:=${BUILD_DIR}/results}"
JOBSCRIPT="${HERE}/run_dbscan_aurora.sh"

# A job that touches a filesystem it did not request is held by the scheduler,
# so request whichever ones the executable, this repo and the results live on.
fs_of() {
    case "$(readlink -f "$1")" in
        /lus/flare/*) echo flare ;;
        /lus/gila/*)  echo gila  ;;
        *)            echo home  ;;
    esac
}
FILESYSTEMS=$(printf '%s\n' "$(fs_of "${EXE}")" "$(fs_of "${HERE}")" \
                            "$(fs_of "${OUTDIR}")" | sort -u | paste -sd:)

if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: benchmark not found at ${EXE}" >&2
    echo "Build it first:  ./DBScan/build_aurora.sh ${BUILD_DIR}" >&2
    exit 1
fi

# Default node list: powers of two up to -m. Report whatever your system
# reaches; there is no requirement to stop at the sizes we ran.
if [[ -z "${NODE_LIST}" ]]; then
    NODE_LIST=""
    n=1
    while (( n <= MAX_NODES )); do
        NODE_LIST+="${n} "
        n=$(( n * 2 ))
    done
fi

mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/submitted_jobs.txt"

# Queue by node count, per the Aurora queue limits in README.md. 'prod' is a
# ROUTING queue: it dispatches to small/medium/large by size. Submitting
# straight to 'small' is refused ("Access to queue is denied") -- it only
# accepts routed jobs.
queue_for() {
    local n=$1
    if   (( n <= 2 ));   then echo "debug"
    elif (( n <= 256 )); then echo "debug-scaling"
    else                      echo "prod"
    fi
}

# Measured: 11 reps take ~2.5-4.5 min of walltime at every size from 1 to 2048
# node, because the run time per rep is nearly flat and launch overhead grows
# only slowly. 15 minutes is a comfortable margin. Do not inflate this --
# shorter requests are scheduled sooner, and at large node counts an oversized
# request can also be charged against your allocation.
walltime_for() { echo "00:15:00"; }

echo "=== DBSCAN scaling study ==="
echo "account   : ${ACCT}"
echo "executable: ${EXE}"
echo "results   : ${OUTDIR}"
echo "node list : ${NODE_LIST}"
echo "reps/job  : ${REPS}"
echo

for n in ${NODE_LIST}; do
    q=$(queue_for "${n}")
    wt=$(walltime_for "${n}")
    out="${OUTDIR}/dbscan_n${n}.out"

    cmd=(qsub -l "select=${n}" -l "walltime=${wt}" -l "filesystems=${FILESYSTEMS}"
         -q "${q}" -A "${ACCT}" -N "dbscan_n${n}" -o "${out}"
         -v "EXE=${EXE},REPS=${REPS},ENV_SH=${HERE}/env_aurora.sh"
         "${JOBSCRIPT}")

    if (( DRYRUN )); then
        printf '%q ' "${cmd[@]}"; echo
        continue
    fi

    jobid=$("${cmd[@]}" 2>&1)
    if [[ ! "${jobid}" =~ ^[0-9]+\. ]]; then
        echo "nodes=${n}: SUBMIT FAILED: ${jobid}" | tee -a "${LOG}"
        continue
    fi
    short="${jobid%%.*}"
    echo "nodes=${n} queue=${q} job=${short} submitted $(date -Is)" | tee -a "${LOG}"

    # Serial mode, and always for the queues with a one-queued-job limit.
    if (( SERIAL )) || [[ "${q}" == "debug" || "${q}" == "debug-scaling" ]]; then
        while qstat "${short}" 2>/dev/null | grep -qE ' [QRHE] '; do sleep 30; done
        echo "nodes=${n} job=${short} finished  $(date -Is)" | tee -a "${LOG}"
    fi
done

if (( DRYRUN )); then
    echo
    echo "(dry run: nothing submitted)"
    exit 0
fi

echo
echo "=== submitted ==="
echo "Watch with : qstat -u \$USER"
echo "Summarize  : python3 ${HERE}/summarize_results.py ${OUTDIR}/dbscan_n*.out"
echo
echo "Jobs above 256 nodes were submitted without waiting; they may sit in the"
echo "queue for hours waiting for a large enough scheduling window."
