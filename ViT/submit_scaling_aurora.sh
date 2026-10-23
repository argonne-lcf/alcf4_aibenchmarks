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
#   --walltime HH:MM:SS        per job            (default: sized per node count)
#   --submit                   stop dry-running
#
# One job per (node count, repetition). Each job runs the full FOM sweep on
# the nodes it has, skipping the sequence lengths whose sequence-parallel
# width does not fit -- so the SP=24 point simply does not appear below 2
# nodes. Jobs are independent; PBS schedules them as capacity appears.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODES="1 2 4 8 16 32 64 128 256 512 1024 2048"
REPS=3
FACTORS=""
PROJECT="alcf4-benchmarks"
WALLTIME=""
SUBMIT=0

while (( $# )); do
    case "$1" in
        --nodes)    NODES="$2"; shift 2 ;;
        --reps)     REPS="$2"; shift 2 ;;
        --factors)  FACTORS="$2"; shift 2 ;;
        --project)  PROJECT="$2"; shift 2 ;;
        --walltime) WALLTIME="$2"; shift 2 ;;
        --submit)   SUBMIT=1; shift ;;
        -h|--help)  sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# Aurora queue policy by node count.
queue_for() {
    local n=$1
    if   (( n <= 2 ));    then echo debug
    elif (( n <= 256 ));  then echo debug-scaling
    elif (( n <= 512 ));  then echo small
    elif (( n <= 1024 )); then echo medium
    else                       echo large
    fi
}

# The sweep is dominated by its largest sequence. run_vit_aurora.sh skips
# factor=120 below 32 nodes, so jobs under that are short; from 32 nodes up
# the 1.73M-token point runs and its walltime falls as nodes rise, because
# Ulysses divides fixed attention work across more ranks.
walltime_for() {
    local n=$1
    [[ -n "${WALLTIME}" ]] && { echo "${WALLTIME}"; return; }
    if   (( n < 32 ));   then echo "01:00:00"   # factor=120 skipped
    elif (( n <= 64 ));  then echo "06:00:00"
    elif (( n <= 256 )); then echo "03:00:00"
    else                      echo "01:30:00"
    fi
}

echo "repo:     ${REPO_ROOT}"
echo "project:  ${PROJECT}"
echo "reps:     ${REPS}"
echo "factors:  ${FACTORS:-all (16 24 32 64 120)}"
echo "nodes:    ${NODES}"
(( SUBMIT )) || echo $'\n*** DRY RUN -- pass --submit to actually queue these ***'
echo

njobs=0
for n in ${NODES}; do
    q=$(queue_for "${n}")
    wt=$(walltime_for "${n}")
    for rep in $(seq 1 "${REPS}"); do
        vars="REP=${rep}"
        [[ -n "${FACTORS}" ]] && vars="${vars},FACTORS=${FACTORS// /_}"
        cmd=(qsub -l "select=${n}" -l "walltime=${wt}"
             -q "${q}" -A "${PROJECT}"
             -N "vit-n${n}-r${rep}"
             -v "${vars}"
             "${REPO_ROOT}/ViT/run_vit_aurora.sh")
        if (( SUBMIT )); then
            jobid=$("${cmd[@]}" 2>&1) && echo "submitted ${jobid}  (n=${n} rep=${rep} q=${q} wt=${wt})" \
                || { echo "FAILED n=${n} rep=${rep}: ${jobid}" >&2; exit 1; }
        else
            printf '%s\n' "${cmd[*]}"
        fi
        (( njobs++ ))
    done
done

echo
echo "${njobs} job(s) $( (( SUBMIT )) && echo submitted || echo planned )."
(( SUBMIT )) && {
    echo "Watch with:   qstat -u \$USER"
    echo "Summarize:    python3 ${REPO_ROOT}/ViT/summarize_results.py"
}
exit 0
