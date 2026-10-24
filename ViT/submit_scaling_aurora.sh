#!/bin/bash
# Submit the ALCF-4 3D ViT weak-scaling series on Aurora.
#
# Run this on a login node (it only calls qsub; it is not itself a PBS job):
#
#   ./ViT/submit_scaling_aurora.sh                  # dry run, prints the plan
#   ./ViT/submit_scaling_aurora.sh --submit         # actually submit
#
# Options:
#   --nodes "1 2 4 ... 2048"   node counts        (default: 1..2048 doubling)
#   --reps N                   repetitions        (default: 3)
#   --factors "32 64"          sequence subset    (default: all five)
#   --project NAME             PBS account        (default: alcf4-benchmarks)
#   --walltime HH:MM:SS        override per job   (default: sized per point)
#   --iters120 N               factor=120 iters   (default: 20, the canonical)
#   --max-queued N             pause while this many jobs sit in Q (default: 0,
#                              meaning do not throttle)
#   --submit                   stop dry-running
#
# ONE JOB PER (node count, factor, repetition). Each job runs a single
# sequence length, so the small points land in debug/debug-scaling and only
# factor=120 needs a production queue.
#
# Why one point per job rather than the whole sweep in one:
#
#   factor   s/iter   iters   wall      queue
#       16     0.08      40    ~2 min   debug / debug-scaling
#       24     0.93      40    ~3 min   debug / debug-scaling
#       32     5.24      40    ~6 min   debug / debug-scaling
#       64    80.90      20   ~29 min   debug / debug-scaling
#      120  1106.00      20   ~6.2 h    small / medium / large
#
# Per-iteration time is set by sequence length and SP, NOT by node count:
# adding nodes adds data-parallel replicas (more samples per iteration), it
# does not make an iteration finish sooner. So the full sweep is ~6.7 h at
# every scale, which never fits the 1 h debug-scaling cap -- but four of its
# five points fit comfortably on their own.
#
# Iterations are NOT split across jobs. Warmup is about 5 iterations (at
# factor=32: 6.09, 5.08, 5.65, 5.20, then flat at ~5.25 s), so a job that ran
# 2 iterations would report warmup numbers only and never reach steady state.
# Each job must run its own warmup plus enough steady iterations to average.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODES="1 2 4 8 16 32 64 128 256 512 1024 2048"
REPS=3
FACTORS=""
PROJECT="alcf4-benchmarks"
WALLTIME=""
ITERS120=20
MAX_QUEUED=0
SUBMIT=0

while (( $# )); do
    case "$1" in
        --nodes)       NODES="$2"; shift 2 ;;
        --reps)        REPS="$2"; shift 2 ;;
        --factors)     FACTORS="$2"; shift 2 ;;
        --project)     PROJECT="$2"; shift 2 ;;
        --walltime)    WALLTIME="$2"; shift 2 ;;
        --iters120)    ITERS120="$2"; shift 2 ;;
        --max-queued)  MAX_QUEUED="$2"; shift 2 ;;
        --submit)      SUBMIT=1; shift ;;
        -h|--help)     sed -n '2,44p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# factor:SP:iters:min_nodes, per the canonical FOM definition.
# factor=120 needs SP=24, i.e. 24 ranks, i.e. 2 Aurora nodes.
points_for() {
    echo "16:1:40:1 24:1:40:1 32:1:40:1 64:4:20:1 120:24:${ITERS120}:2"
}

# Measured seconds per iteration (1 node, fp16, 24 heads). Used only to size
# walltime; the sweep itself measures these properly.
sec_per_iter() {
    case "$1" in
        16)  echo 0.08 ;;
        24)  echo 0.93 ;;
        32)  echo 5.24 ;;
        64)  echo 80.9 ;;
        120) echo 1106 ;;
    esac
}

# Round up to a whole minute and add startup (module load, model build,
# FlashAttention JIT, allocator warmup), then 50% headroom.
walltime_for() {
    local factor=$1 iters=$2
    [[ -n "${WALLTIME}" ]] && { echo "${WALLTIME}"; return; }
    local secs
    secs=$(awk -v s="$(sec_per_iter "${factor}")" -v n="${iters}" \
               'BEGIN{printf "%d", (s*n + 180) * 1.5}')
    # Floor at 10 minutes: the tiny points are dominated by startup, and a
    # too-tight walltime wastes a whole job.
    (( secs < 600 )) && secs=600
    printf '%02d:%02d:00\n' $(( secs / 3600 )) $(( (secs % 3600) / 60 + 1 ))
}

# Aurora queue policy. Two independent constraints have to be satisfied at
# once, and between them they leave a gap:
#
#   debug <=2 nodes, debug-scaling <=256 nodes -- both capped at 1 h walltime
#   prod (routes to small/medium/large) requires >=256 nodes, allows 12-24 h
#
# So a job needing more than an hour on fewer than 256 nodes has no queue to
# go to. That is exactly factor=120 (~6.2 h for 20 iterations at any scale),
# which is why it cannot be run below 256 nodes at the canonical iteration
# count. Returns the queue name, or "none" when nothing accepts the job.
queue_for() {
    local n=$1 wt=$2
    local hours=${wt%%:*}
    if (( 10#${hours} >= 1 )); then
        if   (( n < 256 ));   then echo none
        elif (( n <= 1024 )); then echo small
        elif (( n <= 1999 )); then echo medium
        else                       echo large
        fi
    else
        if   (( n <= 2 ));   then echo debug
        elif (( n <= 256 )); then echo debug-scaling
        elif (( n <= 1024 )); then echo small
        elif (( n <= 1999 )); then echo medium
        else                       echo large
        fi
    fi
}

queued_count() { qstat -u "${USER}" 2>/dev/null | grep -cE ' Q ' || true; }

echo "repo:      ${REPO_ROOT}"
echo "project:   ${PROJECT}"
echo "reps:      ${REPS}"
echo "factors:   ${FACTORS:-all (16 24 32 64 120)}"
echo "nodes:     ${NODES}"
echo "f120 iters: ${ITERS120}"
(( SUBMIT )) || echo $'\n*** DRY RUN -- pass --submit to actually queue these ***'
echo

njobs=0
for n in ${NODES}; do
    for point in $(points_for); do
        IFS=: read -r f sp iters min_nodes <<< "${point}"

        [[ -n "${FACTORS}" ]] && ! grep -qw "${f}" <<< "${FACTORS}" && continue

        nranks=$(( n * 12 ))
        (( sp > nranks ))      && continue   # not enough ranks for this SP
        (( nranks % sp != 0 )) && continue   # world must divide by SP
        (( n < min_nodes ))    && continue

        wt=$(walltime_for "${f}" "${iters}")
        q=$(queue_for "${n}" "${wt}")

        if [[ "${q}" == "none" ]]; then
            echo "# SKIP n=${n} factor=${f}: needs ${wt}, but queues under 256"
            echo "#      nodes cap at 1 h and prod needs >=256 nodes."
            continue
        fi

        for rep in $(seq 1 "${REPS}"); do
            cmd=(qsub -l "select=${n}" -l "walltime=${wt}"
                 -q "${q}" -A "${PROJECT}"
                 -N "vit-n${n}-f${f}-r${rep}"
                 -v "REP=${rep},FACTORS=${f},NUM_ITERS=${iters}"
                 "${REPO_ROOT}/ViT/run_vit_aurora.sh")
            if (( SUBMIT )); then
                while (( MAX_QUEUED > 0 )) && (( $(queued_count) >= MAX_QUEUED )); do
                    sleep 60
                done
                jobid=$("${cmd[@]}" 2>&1) \
                    && echo "submitted ${jobid}  (n=${n} f=${f} rep=${rep} q=${q} wt=${wt})" \
                    || { echo "FAILED n=${n} f=${f} rep=${rep}: ${jobid}" >&2; exit 1; }
            else
                printf '%s\n' "${cmd[*]}"
            fi
            (( njobs++ ))
        done
    done
done

echo
echo "${njobs} job(s) $( (( SUBMIT )) && echo submitted || echo planned )."
(( SUBMIT )) && {
    echo "Watch with:   qstat -u \$USER"
    echo "Summarize:    python3 ${REPO_ROOT}/ViT/summarize_results.py"
}
exit 0
