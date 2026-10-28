#!/bin/bash
# Triage ViT benchmark logs: did each run train to the end, how did it exit,
# and if it died, why. Read this, not the PBS exit status.
#
#   ./ViT/check_logs.sh                       # every log in ViT/results
#   ./ViT/check_logs.sh ViT/results/vit_n256_*.log
#
# Why the PBS exit status is not enough: on Aurora a run often finishes
# training, logs "after training is done", and then mpiexec does not return
# for 10-20+ minutes. A single-point job then hits its walltime and PBS
# reports a failure (exit -29) over complete data; in an ensemble the exit
# watchdog stops the member and reports it as trained. Either way the timed
# iterations are already in the log, so the log is the record.
#
# Columns:
#   iters     last iteration logged / iterations requested
#   trained   "after training is done" reached
#   exit      clean  -- mpiexec printed its "Application ... resources:" line
#             hung   -- trained, but mpiexec never reported an exit
#             -      -- did not train, so the exit says nothing new
#   skip      optimizer steps skipped for fp16 overflow after warmup (must be 0)
#   nan       NaN iterations reported after warmup (must be 0; see README)
#   problem   first failure signature, if any
#
# GPU page faults name the host. Hosts listed at the end are candidates for
# EXCLUDE_NODES in run_ensemble_aurora.sh -- a host that faults in more than
# one job, at the same local rank, is very likely bad hardware.
#
# A run whose loss scale collapses (skip > 0, then "loss scale already at
# minimum") names no host: the overflow is detected after the gradient
# all-reduce, so every rank sees it. Intersect the node files of the runs
# that collapsed (results/nodefiles_<job>/) to find the suspect.

: "${WARMUP:=5}"

if (( $# == 0 )); then
    dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
    set -- "${dir}"/vit_n*_rep*.log*
    [[ -e "$1" ]] || { echo "no logs in ${dir}" >&2; exit 1; }
fi

printf '%-44s %-9s %7s %6s %4s %4s  %s\n' log '  iters' trained exit skip nan problem
faults=$(mktemp)
trap 'rm -f "${faults}"' EXIT

for log in "$@"; do
    awk -v warmup="${WARMUP}" -v name="$(basename "${log}")" -v faults="${faults}" '
        /^ *iteration +[0-9]+\/ *[0-9]+ \|/ {
            split($0, a, "|"); split(a[1], it, /[ \/]+/)
            last = it[3]; want = it[4]
            if (match($0, /number of nan iterations: +[0-9]+/) && last > warmup) {
                v = substr($0, RSTART, RLENGTH); sub(/.*: +/, "", v); nan += v
            }
            if (match($0, /number of skipped iterations: +[0-9]+/) && last > warmup) {
                v = substr($0, RSTART, RLENGTH); sub(/.*: +/, "", v); skip += v
            }
        }
        /after training is done/                 { trained = 1 }
        /^Application .* resources:/             { clean = 1 }
        problem == "" && /Segmentation fault from GPU/ { problem = "GPU page fault" }
        problem == "" && /loss scale already at minimum/ { problem = "fp16 loss scale collapsed"; nohost = 1 }
        /died from signal/ && host == "" {
            host = $1; sub(/:$/, "", host); sub(/\..*/, "", host)
            rank = $3
        }
        problem == "" && /OUT_OF_RESOURCES|out of memory/ { problem = "out of device memory" }
        problem == "" && /^Traceback|Error:/     { problem = substr($0, 1, 60) }
        END {
            # After training, "died from signal" is the exit watchdog at work;
            # after a loss-scale collapse it is whichever rank was signalled
            # first. Neither says anything about the host.
            if (host != "" && !trained && !nohost) {
                problem = (problem == "" ? "died" : problem) " on " host " rank " rank
                if (problem ~ /GPU page fault/) print host >> faults
            }
            exit_ = trained ? (clean ? "clean" : "hung") : "-"
            printf "%-44s %4s/%-4s %7s %6s %4d %4d  %s\n", name, (last == "" ? "0" : last),
                (want == "" ? "?" : want), (trained ? "yes" : "NO"), exit_, skip, nan, problem
        }
    ' "${log}"
done

if [[ -s "${faults}" ]]; then
    echo
    echo "GPU page faults by host (consider EXCLUDE_NODES):"
    sort "${faults}" | uniq -c | sort -rn | sed 's/^/  /'
fi
