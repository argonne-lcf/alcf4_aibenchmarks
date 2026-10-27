#!/bin/bash
# Exercise run_ensemble_aurora.sh on a login node, with no PBS job and no
# GPUs: a stub mds_launch.sh stands in for training. Run this after changing
# the ensemble runner.
#
#   ./ViT/test_ensemble_aurora.sh
#
# Five fake hosts, one excluded (EXCLUDE_NODES), four members, one per
# behaviour the runner has to handle:
#   factor 16  trains, exits 0                      -> "exit 0"
#   factor 24  trains, then hangs with a child      -> SIGTERM by the watchdog
#   factor 32  dies before training is done         -> "FAILED", job exits 3
#   factor 64  trains, hangs and ignores SIGTERM    -> SIGKILL by the watchdog
# Takes about 45 s.

set -u
runner="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ensemble_aurora.sh"
tmp=$(mktemp -d)
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/vitroot" "${tmp}/results" "${tmp}/work"
: > "${tmp}/env.sh"
printf 'host%s.example\n' 1 2 3 4 5 > "${tmp}/nodes"

cat > "${tmp}/vitroot/mds_launch.sh" <<'EOF'
#!/bin/bash
echo "stub factor=${factor} pid=$$ pgid=$(ps -o pgid= $$ | tr -d ' ') nodes=$(tr '\n' ' ' < "${PBS_NODEFILE}")"
case "${factor}" in
    16) echo "[after training is done] datetime: x"; exit 0 ;;
    24) echo "[after training is done] datetime: x"; sleep 4242 & wait ;;
    32) echo "Traceback: boom"; exit 3 ;;
    64) trap '' TERM
        echo "[after training is done] datetime: x"
        bash -c "trap '' TERM; sleep 4243" & wait ;;
esac
EOF

env PBS_O_WORKDIR="${tmp}/work" PBS_NODEFILE="${tmp}/nodes" PBS_JOBID=123.test \
    ViT_DIR="${tmp}" ENV_SH="${tmp}/env.sh" ViT_ROOT="${tmp}/vitroot" \
    RESULTS_DIR="${tmp}/results" MEMBERS="1:16:1_1:24:1_1:32:1_1:64:1" \
    EXCLUDE_NODES="host3" EXIT_GRACE=4 EXIT_POLL=1 \
    bash "${runner}" > "${tmp}/out" 2>&1
rc=$?

fails=0
check() {   # check <description> <command...>
    local what=$1; shift
    if "$@"; then echo "PASS  ${what}"; else echo "FAIL  ${what}"; fails=$(( fails + 1 )); fi
}
has() { grep -qE "$1" "${tmp}/out"; }
stubs() { grep -h "^stub" "${tmp}"/results/*.log; }

check "excluded host reported"              has "excluded: 1 host"
check "excluded host given to no member"    bash -c "! grep -h '^stub' ${tmp}/results/*.log | grep -q host3"
check "clean member exits 0"                has "member n1_f16: exit 0"
check "hung member stopped by SIGTERM"      has "member n1_f24: trained; stopped at exit by the watchdog"
check "member that ignores TERM is KILLed"  has "member n1_f64: ignored SIGTERM -- sending SIGKILL"
check "SIGKILLed member counted as trained" has "member n1_f64: trained; stopped at exit by the watchdog"
check "untrained member reported FAILED"    has "member n1_f32: exit 3 +<-- FAILED"
check "job exit is the failure's (3)"       test "${rc}" -eq 3
check "each member leads its own group"     bash -c "grep -h '^stub' ${tmp}/results/*.log | awk '{split(\$3,a,\"=\"); split(\$4,b,\"=\"); if (a[2] != b[2]) exit 1}'"
check "no stub processes left behind"       bash -c "! pgrep -f 'sleep 424[23]' > /dev/null"

if (( fails )); then
    echo
    echo "--- runner output:"
    cat "${tmp}/out"
    echo "--- member logs:"
    stubs
fi
exit "${fails}"
