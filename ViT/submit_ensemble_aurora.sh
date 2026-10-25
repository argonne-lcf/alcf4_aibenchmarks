#!/bin/bash
# Pack ViT benchmark points into ensemble jobs and submit them.
#
#   ./ViT/submit_ensemble_aurora.sh                 # dry run, prints the plan
#   ./ViT/submit_ensemble_aurora.sh --submit        # actually submit
#
# Options:
#   --nodes "2 4 8 ... 128"   node counts to cover   (default: 2..128)
#   --factor "16 24 32 64"    sequence point(s)      (default: 120)
#   --iters N                 iterations per member  (default: 20)
#   --reps N                  repetitions            (default: 1)
#   --alloc N                 nodes per ensemble job (default: 256)
#   --walltime HH:MM:SS       per job                (default: sized to fit)
#   --project NAME            PBS account            (default: alcf4-benchmarks)
#   --submit                  stop dry-running
#
# Aurora's queues leave a gap that single-point jobs cannot cross: debug
# (<=2 nodes) and debug-scaling (<=256 nodes) cap at 1 h walltime, while prod
# routes only jobs of 256+ nodes. The 1.73M-token point needs ~6.2 h for 20
# iterations at ANY node count -- per-iteration time is set by sequence length
# and SP, not by node count -- so on fewer than 256 nodes it has nowhere to
# run.
#
# Packing the whole weak-scaling curve into one 256-node prod allocation
# closes the gap, and costs less than running the points separately would if
# it were possible: members run concurrently, so the job is as long as its
# slowest member, not the sum.
#
#   2 + 4 + 8 + 16 + 32 + 64 + 128 = 254 nodes, one 12 h job

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODES="2 4 8 16 32 64 128"
FACTOR=120
ITERS=20
REPS=1
ALLOC=256
WALLTIME=""
PROJECT="alcf4-benchmarks"
SUBMIT=0

while (( $# )); do
    case "$1" in
        --nodes)    NODES="$2"; shift 2 ;;
        --factor)   FACTOR="$2"; shift 2 ;;
        --iters)    ITERS="$2"; shift 2 ;;
        --reps)     REPS="$2"; shift 2 ;;
        --alloc)    ALLOC="$2"; shift 2 ;;
        --walltime) WALLTIME="$2"; shift 2 ;;
        --project)  PROJECT="$2"; shift 2 ;;
        --submit)   SUBMIT=1; shift ;;
        -h|--help)  sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

sec_per_iter() {
    case "$1" in
        16)  echo 0.08 ;;
        24)  echo 0.93 ;;
        32)  echo 5.24 ;;
        64)  echo 80.9 ;;
        120) echo 1106 ;;
        *)   echo 1106 ;;
    esac
}

# Members run concurrently, so the job lasts as long as its SLOWEST member,
# not the sum. Size to the largest factor requested, add startup and 50%
# headroom, then clamp to the queue ceiling.
if [[ -z "${WALLTIME}" ]]; then
    slowest=0
    for f in ${FACTOR}; do
        s=$(awk -v s="$(sec_per_iter "${f}")" -v n="${ITERS}" \
                'BEGIN{printf "%d", s*n}')
        (( s > slowest )) && slowest="${s}"
    done
    secs=$(awk -v s="${slowest}" 'BEGIN{printf "%d", (s + 300) * 1.5}')
    max=43200                       # small/medium cap at 12 h
    (( secs > max )) && secs="${max}"
    WALLTIME=$(printf '%02d:%02d:00' $(( secs / 3600 )) $(( (secs % 3600) / 60 + 1 )))
fi

queue_for() {
    local n=$1
    if   (( n < 256 ));   then echo "TOO SMALL FOR PROD"
    elif (( n <= 1024 )); then echo small
    elif (( n <= 1999 )); then echo medium
    else                       echo large
    fi
}

echo "repo:     ${REPO_ROOT}"
echo "project:  ${PROJECT}"
echo "factor:   ${FACTOR}   iters: ${ITERS}   reps: ${REPS}"
echo "alloc:    ${ALLOC} nodes per job"
echo "walltime: ${WALLTIME}"
echo "nodes:    ${NODES}"
(( SUBMIT )) || echo $'\n*** DRY RUN -- pass --submit to actually queue these ***'
echo

# Greedily pack node counts into allocations of ALLOC nodes.
njobs=0
for rep in $(seq 1 "${REPS}"); do
    batch=""
    used=0
    flush() {
        [[ -z "${batch}" ]] && return
        local q
        q=$(queue_for "${ALLOC}")
        if [[ "${q}" == "TOO SMALL FOR PROD" ]]; then
            echo "ERROR: --alloc ${ALLOC} is below prod's 256-node minimum" >&2
            exit 1
        fi
        # qsub -v cannot carry spaces; the runner turns _ back into space.
        local members="${batch// /_}"
        local cmd=(qsub -l "select=${ALLOC}" -l "walltime=${WALLTIME}"
                   -q "${q}" -A "${PROJECT}"
                   -N "vit-ens-f${FACTOR// /_}-r${rep}"
                   -v "MEMBERS=${members},REP=${rep}"
                   "${REPO_ROOT}/ViT/run_ensemble_aurora.sh")
        echo "# batch: ${batch}  (${used} of ${ALLOC} nodes used)"
        if (( SUBMIT )); then
            local jobid
            jobid=$("${cmd[@]}" 2>&1) \
                && echo "submitted ${jobid}" \
                || { echo "FAILED: ${jobid}" >&2; exit 1; }
        else
            printf '%s\n' "${cmd[*]}"
        fi
        echo
        njobs=$(( njobs + 1 ))
        batch=""
        used=0
    }

    for f in ${FACTOR}; do
        for n in ${NODES}; do
            # SP must fit in the member's own ranks.
            case "${f}" in
                64)  need_sp=4  ;;
                120) need_sp=24 ;;
                *)   need_sp=1  ;;
            esac
            if (( n * 12 < need_sp )); then
                echo "# SKIP n=${n} factor=${f}: ${n} node(s) cannot host SP=${need_sp}"
                continue
            fi
            if (( n > ALLOC )); then
                echo "# SKIP n=${n}: larger than --alloc ${ALLOC}"
                continue
            fi
            if (( used + n > ALLOC )); then
                flush
            fi
            batch="${batch:+${batch} }${n}:${f}:${ITERS}"
            used=$(( used + n ))
        done
    done
    flush
done

echo "${njobs} ensemble job(s) $( (( SUBMIT )) && echo submitted || echo planned )."
(( SUBMIT )) && {
    echo "Watch with:   qstat -u \$USER"
    echo "Summarize:    python3 ${REPO_ROOT}/ViT/summarize_results.py"
}
exit 0
