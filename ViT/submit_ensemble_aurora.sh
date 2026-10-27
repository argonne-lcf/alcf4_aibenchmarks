#!/bin/bash
# Pack ViT benchmark points into ensemble jobs and submit them. This is how
# the published Aurora results were collected.
#
#   ./ViT/submit_ensemble_aurora.sh                 # dry run, prints the plan
#   ./ViT/submit_ensemble_aurora.sh --submit        # actually submit
#
# Options:
#   --nodes "1 2 4 ... 256"   node counts to cover   (default: 1..256 doubling)
#   --factor "16 24 32 64"    sequence point(s)      (default: all five)
#   --iters N                 iterations per member  (default: canonical, 40
#                             for factors 16-32 and 20 for 64 and 120)
#   --reps N                  repetitions            (default: 1)
#   --rep-start N             first repetition index (default: 1)
#   --alloc N                 nodes per ensemble job (default: 512)
#   --exclude "host ..."      known-bad hosts to skip; one spare node is
#                             requested per host so members keep full size
#   --walltime HH:MM:SS       per job                (default: sized per job)
#   --project NAME            PBS account            (default: alcf4-benchmarks)
#   --submit                  stop dry-running
#
# Aurora's queues leave a gap that single-point jobs cannot cross: debug
# (<=2 nodes) and debug-scaling (<=256 nodes) cap at 1 h walltime and allow
# one queued job per user, while prod routes only jobs of 256+ nodes. The
# 1.73M-token point needs ~7.8 h for 20 iterations at ANY node count --
# per-iteration time is set by sequence length and SP, not by node count --
# so on fewer than 256 nodes it has nowhere to run.
#
# Packing a weak-scaling curve into one prod allocation closes the gap. One
# curve, 1+2+4+...+256 nodes, is 511 nodes, so the default plan is one
# 512-node job per sequence point and repetition (members spill across
# points where they fit). Members run concurrently, so a job lasts as long as
# its slowest member, not the sum.
#
# Points share a job only when their run times are within 1.5x of each
# other, so no member's nodes sit idle for more than a third of the job:
# factors 16 and 24 pack together, 32 and 64 do not, and 120 is always alone.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODES="1 2 4 8 16 32 64 128 256"
FACTOR="16 24 32 64 120"
ITERS=""
REPS=1
REP_START=1
ALLOC=512
EXCLUDE=""
WALLTIME=""
PROJECT="alcf4-benchmarks"
SUBMIT=0

while (( $# )); do
    case "$1" in
        --nodes)     NODES="$2"; shift 2 ;;
        --factor)    FACTOR="$2"; shift 2 ;;
        --iters)     ITERS="$2"; shift 2 ;;
        --reps)      REPS="$2"; shift 2 ;;
        --rep-start) REP_START="$2"; shift 2 ;;
        --alloc)     ALLOC="$2"; shift 2 ;;
        --exclude)   EXCLUDE="$2"; shift 2 ;;
        --walltime)  WALLTIME="$2"; shift 2 ;;
        --project)   PROJECT="$2"; shift 2 ;;
        --submit)    SUBMIT=1; shift ;;
        -h|--help)   sed -n '2,37p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# Measured seconds per iteration, 12 heads, one node (256-node members run
# up to ~7% slower). Used only to size walltime.
sec_per_iter() {
    case "$1" in
        16)  echo 0.19 ;;
        24)  echo 1.12 ;;
        32)  echo 5.2 ;;
        64)  echo 81 ;;
        *)   echo 1411 ;;
    esac
}

# Canonical iteration counts (alcf4_bench_polaris.sh), unless --iters.
iters_for() {
    if [[ -n "${ITERS}" ]]; then echo "${ITERS}"; return; fi
    case "$1" in
        16|24|32) echo 40 ;;
        *)        echo 20 ;;
    esac
}

# Ulysses SP each point runs at, as run_ensemble_aurora.sh sets it.
sp_for() {
    case "$1" in
        64)  echo 4  ;;
        120) echo 12 ;;
        *)   echo 1  ;;
    esac
}

# Seconds one member needs: its iterations, +25% and 10 minutes for startup
# (module load, FlashAttention JIT, collective setup) and the exit watchdog's
# grace.
member_secs() {
    awk -v s="$(sec_per_iter "$1")" -v n="$2" 'BEGIN{printf "%d", s * n * 1.25 + 600}'
}

# A job's walltime is its slowest member's, clamped to the 12 h prod ceiling.
walltime_for() {
    [[ -n "${WALLTIME}" ]] && { echo "${WALLTIME}"; return; }
    local secs=$1
    (( secs > 43200 )) && secs=43200
    printf '%02d:%02d:00\n' $(( secs / 3600 )) $(( (secs % 3600) / 60 ))
}

nexclude=$(wc -w <<< "${EXCLUDE}")
SELECT=$(( ALLOC + nexclude ))

# small, medium and large are from_route_only: qsub refuses them directly
# ("Access to queue is denied") and they are reachable only through the prod
# router, which picks the execution queue itself.
if (( ALLOC < 256 )); then
    echo "ERROR: --alloc ${ALLOC} is below prod's 256-node minimum" >&2
    exit 1
fi

echo "repo:     ${REPO_ROOT}"
echo "project:  ${PROJECT}"
echo "factors:  ${FACTOR}   iters: ${ITERS:-canonical}   reps: ${REPS} from ${REP_START}"
echo "alloc:    ${ALLOC} nodes per job${EXCLUDE:+ + ${nexclude} spare (excluding ${EXCLUDE})}"
echo "nodes:    ${NODES}"
(( SUBMIT )) || echo $'\n*** DRY RUN -- pass --submit to actually queue these ***'
echo

njobs=0
batch=()
used=0
fastest=0
slowest=0

flush() {
    (( ${#batch[@]} )) || return 0
    local wt factors members env
    wt=$(walltime_for "${slowest}")
    factors=$(printf '%s\n' "${batch[@]}" | cut -d: -f2 | uniq | paste -sd_)
    # qsub -v cannot carry spaces; the runner turns _ back into space.
    members=$(IFS=_; echo "${batch[*]}")
    env="MEMBERS=${members},REP=${rep}"
    [[ -n "${EXCLUDE}" ]] && env="${env},EXCLUDE_NODES=${EXCLUDE// /_}"
    local cmd=(qsub -l "select=${SELECT}" -l "walltime=${wt}"
               -q prod -A "${PROJECT}"
               -N "vit-ens-f${factors}-r${rep}"
               -v "${env}"
               "${REPO_ROOT}/ViT/run_ensemble_aurora.sh")
    echo "# batch: ${batch[*]}  (${used} of ${ALLOC} nodes, ${wt})"
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
    batch=()
    used=0
    fastest=0
    slowest=0
}

# Greedily pack members into allocations of ALLOC nodes.
for rep in $(seq "${REP_START}" $(( REP_START + REPS - 1 ))); do
    for f in ${FACTOR}; do
        sp=$(sp_for "${f}")
        for n in ${NODES}; do
            if (( n * 12 < sp )); then
                echo "# SKIP n=${n} factor=${f}: ${n} node(s) cannot host SP=${sp}"
                continue
            fi
            if (( n > ALLOC )); then
                echo "# SKIP n=${n}: larger than --alloc ${ALLOC}"
                continue
            fi
            it=$(iters_for "${f}")
            secs=$(member_secs "${f}" "${it}")
            if (( used + n > ALLOC )) || (( ${#batch[@]} && \
                    ( secs * 2 > fastest * 3 || slowest * 2 > secs * 3 ) )); then
                flush
            fi
            batch+=("${n}:${f}:${it}")
            used=$(( used + n ))
            (( fastest == 0 || secs < fastest )) && fastest="${secs}"
            (( secs > slowest )) && slowest="${secs}"
        done
    done
    flush
done

echo "${njobs} ensemble job(s) $( (( SUBMIT )) && echo submitted || echo planned )."
(( SUBMIT )) && {
    echo "Watch with:   qstat -u \$USER"
    echo "Check logs:   ${REPO_ROOT}/ViT/check_logs.sh"
    echo "Summarize:    python3 ${REPO_ROOT}/ViT/summarize_results.py"
}
exit 0
