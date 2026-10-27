#!/bin/bash -l
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -N vit-ensemble

# Run several ViT benchmark points CONCURRENTLY inside one PBS allocation,
# each on its own disjoint slice of the nodes.
#
# Why this exists: Aurora's queues leave a gap. debug (<=2 nodes) and
# debug-scaling (<=256 nodes) both cap at 1 h walltime, while prod routes
# only jobs of 256+ nodes. A run needing more than an hour on fewer than 256
# nodes therefore has no queue at all -- and that is exactly the 1.73M-token
# point, which takes ~7.8 h for 20 iterations at ANY node count (per-iteration
# time is set by sequence length and SP, not by how many nodes you add).
#
# An ensemble closes the gap: ask for 256 nodes in prod, then run the whole
# weak-scaling curve inside it at once.
#
#   members 2+4+8+16+32+64+128 = 254 nodes, all concurrent, one 12 h job
#
# Submit via ViT/submit_ensemble_aurora.sh, or directly:
#
#   qsub -l select=256 -l walltime=12:00:00 -q prod -A <project> \
#        -v MEMBERS="2:120:20 4:120:20 8:120:20" ViT/run_ensemble_aurora.sh
#
# MEMBERS is a space-separated list of nodes:factor:iters. Each member gets
# a disjoint slice of PBS_NODEFILE and its own log. The job ends when every
# member has finished; one member failing does not stop the others.

cd "${PBS_O_WORKDIR}"

: "${ViT_DIR:=${PBS_O_WORKDIR}/ViT}"
: "${ENV_SH:=${ViT_DIR}/env_aurora.sh}"
: "${ViT_ROOT:=${ViT_DIR}/Megatron-DeepSpeed_ViT}"
: "${RESULTS_DIR:=${ViT_DIR}/results}"
: "${REP:=1}"
: "${MEMBERS:=}"

if [[ -z "${MEMBERS}" ]]; then
    echo "ERROR: MEMBERS is empty. Pass -v MEMBERS=\"2:120:20 4:120:20\"" >&2
    exit 1
fi
if [[ ! -r "${ENV_SH}" ]]; then
    echo "ERROR: cannot read ${ENV_SH}" >&2
    exit 1
fi

export ViT_ROOT
source "${ENV_SH}"

RANKS_PER_NODE=12
NNODES_TOTAL=$(wc -l < "${PBS_NODEFILE}")
mkdir -p "${RESULTS_DIR}"

# MEMBERS may arrive underscore-separated: qsub -v cannot carry spaces.
MEMBERS="${MEMBERS//_/ }"

echo "=== ALCF-4 3D ViT ensemble ==="
echo "=== allocation: ${NNODES_TOTAL} nodes  rep: ${REP} ==="
echo "=== host: $(hostname)  date: $(date) ==="
echo "=== members: ${MEMBERS} ==="
echo

# Check the slices fit before launching any of them: a member that silently
# overlapped another would corrupt both runs' timings.
need=0
for m in ${MEMBERS}; do
    IFS=: read -r n _ _ <<< "${m}"
    need=$(( need + n ))
done
if (( need > NNODES_TOTAL )); then
    echo "ERROR: members need ${need} nodes, allocation has ${NNODES_TOTAL}" >&2
    exit 1
fi
echo "node budget: ${need} of ${NNODES_TOTAL} used"
echo

# Binding and CCL settings, matching run_vit_aurora.sh.
export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export NUMEXPR_MAX_THREADS=7
export OMP_NUM_THREADS=7
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export PALS_PMI=pmix
export CCL_ATL_TRANSPORT=mpi
export WANDB_MODE=disabled

# Model configuration, fixed across the sweep (alcf4_bench_polaris.sh).
export VIT="315M"
export VIT3D=1
export fp16=1
export ZERO=1
export MBS=1
export FA=1

SLICE_DIR="${RESULTS_DIR}/nodefiles_${PBS_JOBID%%.*}"
mkdir -p "${SLICE_DIR}"

offset=0
member_idx=0
pids=()
names=()

for m in ${MEMBERS}; do
    IFS=: read -r n f iters <<< "${m}"
    # factor=120 at SP=12 keeps the 12-head model on one node's 12 tiles, and
    # needs activation recompute to fit: 27.7 GB peak per tile with it. The
    # FOM counts model FLOPs only, so recompute costs time, not credit.
    sp=1
    ckpt=0
    case "${f}" in
        64)  sp=4  ;;
        120) sp=12; ckpt=1 ;;
    esac

    # 12 heads is the benchmark model. Ulysses needs heads % SP == 0, so a
    # point whose SP does not divide 12 has to run more heads -- a different,
    # slower model (see mds_launch.sh), tagged _h<heads> in its log name.
    heads=12
    (( 12 % sp != 0 )) && heads="${sp}"
    htag=""
    (( heads != 12 )) && htag="_h${heads}"

    nranks=$(( n * RANKS_PER_NODE ))
    if (( sp > nranks )) || (( nranks % sp != 0 )); then
        echo "--- SKIP member ${m}: ${nranks} ranks cannot host SP=${sp} ---"
        continue
    fi

    # Carve this member's own nodefile out of the allocation.
    slice="${SLICE_DIR}/member${member_idx}_n${n}_f${f}"
    tail -n +$(( offset + 1 )) "${PBS_NODEFILE}" | head -n "${n}" > "${slice}"
    offset=$(( offset + n ))

    log="${RESULTS_DIR}/vit_n${n}_factor${f}_sp${sp}${htag}_rep${REP}.log"
    echo "######## member ${member_idx}: ${n} nodes, factor=${f}, SP=${sp}, heads=${heads}, ${iters} iters"
    echo "         nodes: $(tr '\n' ' ' < "${slice}")"
    echo "         log:   ${log}"

    # Each member runs mds_launch.sh in its own environment:
    #   PBS_NODEFILE  -> its slice, so NHOSTS and --hostfile are its own
    #   PBS_O_WORKDIR -> the submodule, which mds_launch.sh requires
    #   DS_TAG        -> unique DeepSpeed config filename, no collisions
    #   MASTER_ADDR   -> its own first node
    #   MASTER_PORT   -> distinct per member
    (
        export PBS_NODEFILE="${slice}"
        export PBS_O_WORKDIR="${ViT_ROOT}"
        export DS_TAG="m${member_idx}f${f}"
        export MASTER_ADDR="$(head -n 1 "${slice}")"
        export MASTER_PORT=$(( 3442 + member_idx * 7 ))
        export factor="${f}"
        export SP="${sp}"
        export VIT_HEADS="${heads}"
        export ACT_CKPT="${ckpt}"
        export NUM_ITERS="${iters}"
        cd "${ViT_ROOT}"
        bash "${ViT_ROOT}/mds_launch.sh"
    ) > "${log}" 2>&1 &

    pids+=($!)
    names+=("n${n}_f${f}")
    member_idx=$(( member_idx + 1 ))
    echo
done

if (( ${#pids[@]} == 0 )); then
    echo "ERROR: no runnable members" >&2
    exit 1
fi

echo "=== ${#pids[@]} member(s) launched, waiting ==="
echo

overall=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "member ${names[$i]}: exit 0"
    else
        rc=$?
        echo "member ${names[$i]}: exit ${rc}  <-- FAILED"
        overall="${rc}"
    fi
done

echo
echo "=== ensemble complete, exit ${overall} ==="
echo "=== summarize: python3 ${PBS_O_WORKDIR}/ViT/summarize_results.py ==="
exit "${overall}"
