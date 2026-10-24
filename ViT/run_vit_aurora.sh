#!/bin/bash -l
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -N vit-fom

# ALCF-4 3D ViT benchmark -- one PBS job runs the FOM sweep on the nodes it
# was given. Submit one job per node count; submit_scaling_aurora.sh does
# that for a whole weak-scaling series.
#
#   qsub -l select=<N> -q <queue> -A <project> ViT/run_vit_aurora.sh
#
# The sweep follows alcf4_bench_polaris.sh in the Megatron-DeepSpeed_ViT
# submodule, which is the canonical ALCF-4 FOM definition: five sequence
# lengths, 40 timed iterations for the three that fit without sequence
# parallelism and 20 for the two that need it.
#
#   seq      factor   SP   iters
#   4K         16      1     40
#   13.8K      24      1     40
#   32.8K      32      1     40
#   262K       64      4     20
#   1.73M     120     24     20
#
# factor is the number of 16-voxel patches along each edge of the cubic
# volume, so the token count is factor^3 and the attention cost -- which is
# quadratic in token count -- grows as factor^6. 120 rather than 128 because
# 120^3 = 2^9*3^3*5^3 divides evenly by 12 and 24 (Aurora has 12 ranks per
# node) while 128^3 = 2^21 has no factor of 3 and cannot.
#
# Options via -v:
#   FACTORS="32 64"     subset of the sweep (default: all five). Under
#                       `qsub -v`, which cannot carry spaces, write "32_64".
#   REP=2               repetition index; only tags the log name
#   NUM_ITERS=10        override every point's iteration count
#   RESULTS_DIR=/path   where logs land (default: ViT/results)

cd "${PBS_O_WORKDIR}"

# PBS runs a copy of this script from its spool directory, so BASH_SOURCE
# cannot locate the repo. Derive paths from PBS_O_WORKDIR; override with -v.
: "${ViT_DIR:=${PBS_O_WORKDIR}/ViT}"
: "${ENV_SH:=${ViT_DIR}/env_aurora.sh}"
: "${ViT_ROOT:=${ViT_DIR}/Megatron-DeepSpeed_ViT}"
: "${RESULTS_DIR:=${ViT_DIR}/results}"
: "${REP:=1}"
: "${FACTORS:=}"
# NUM_ITERS is the documented override, but mds_launch.sh also reads it, so
# capture it here and unset it -- otherwise it would silently override every
# point's canonical iteration count through the environment.
ITERS_OVERRIDE="${NUM_ITERS:-}"
unset NUM_ITERS

if [[ ! -r "${ENV_SH}" ]]; then
    echo "ERROR: cannot read ${ENV_SH}" >&2
    echo "Submit from the repo root, or pass -v ViT_DIR=/path/to/ViT" >&2
    exit 1
fi
if [[ ! -d "${ViT_ROOT}/megatron" ]]; then
    echo "ERROR: Megatron-DeepSpeed_ViT not found at ${ViT_ROOT}" >&2
    echo "Initialize it:  git submodule update --init ViT/Megatron-DeepSpeed_ViT" >&2
    exit 1
fi

export ViT_ROOT
source "${ENV_SH}"

NNODES=$(wc -l < "${PBS_NODEFILE}")
RANKS_PER_NODE=12            # one rank per PVC tile (6 GPUs x 2 tiles)
NRANKS=$(( NNODES * RANKS_PER_NODE ))

mkdir -p "${RESULTS_DIR}"

# Binding and CCL settings carried over from mds_launch.sh's Aurora setup.
# NOTE: the CCL_* variables were tuned for the oneCCL bindings, which this
# stack no longer uses (native XCCL instead). Kept so runs match the
# launcher's intent; see README for what is known about their effect.
export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export NUMEXPR_MAX_THREADS=7
export OMP_NUM_THREADS=7

export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export PALS_PMI=pmix
export CCL_ATL_TRANSPORT=mpi

# In an ensemble the member sets these to its own slice; keep any value
# already in the environment (see submit_ensemble_aurora.sh).
export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-$(( RANDOM + 1024 ))}"
export WANDB_MODE=disabled

# Model configuration, fixed across the sweep (alcf4_bench_polaris.sh).
export VIT="315M"            # NLAYERS 14, HSIZE 1536, FFN 4096, HEADS 12
export VIT3D=1
export fp16=1
export ZERO=1
export MBS=1
export FA=1

echo "=== ALCF-4 3D ViT FOM sweep ==="
echo "=== nodes: ${NNODES}  ranks: ${NRANKS}  rep: ${REP} ==="
echo "=== host: $(hostname)  date: $(date) ==="
echo "=== results: ${RESULTS_DIR} ==="
echo

# Record that this run is on the GPUs before timing anything. A CPU fallback
# produces the same TFLOPS line, so the FOM alone cannot distinguish them;
# this writes the evidence into every result. It is a few seconds on one node
# and is informational -- a failure here does not stop the sweep.
echo "--- GPU residency ---"
mpiexec --envall -n "${RANKS_PER_NODE}" --ppn "${RANKS_PER_NODE}" \
        --cpu-bind="${CPU_BIND}" \
    python3 "${ViT_DIR}/gpu_residency.py" \
    2>&1 | tee "${RESULTS_DIR}/gpu_residency_n${NNODES}_rep${REP}.log"
echo

# mds_launch.sh writes its DeepSpeed config into PBS_O_WORKDIR and resolves
# the model script relative to it, so it has to run with the submodule as the
# working directory. Save the real one first -- it is how we found the repo.
REPO_WORKDIR="${PBS_O_WORKDIR}"
export PBS_O_WORKDIR="${ViT_ROOT}"
cd "${ViT_ROOT}"

# factor:SP:iters:min_nodes, per the canonical FOM definition.
#
# min_nodes is a hard floor: SP ranks must exist, and SP=24 needs 2 Aurora
# nodes. It is NOT a walltime floor. An earlier version set 32 here on the
# theory that more nodes shorten the run; that was wrong. Ulysses pins
# per-iteration time to the sequence length and SP, so adding nodes adds
# data-parallel replicas (more samples per iteration) at identical wall
# clock. factor=120 takes ~1106 s/iteration at 2 nodes and at 2048.
# Set FORCE_ALL=1 to ignore these floors.
ALL_POINTS="16:1:40:1 24:1:40:1 32:1:40:1 64:4:20:1 120:24:20:2"
# qsub -v cannot carry spaces, so submit_scaling_aurora.sh passes a subset
# underscore-separated ("32_64"). Accept either form.
FACTORS="${FACTORS//_/ }"

overall=0
for point in ${ALL_POINTS}; do
    IFS=: read -r f sp iters min_nodes <<< "${point}"

    # Honor a FACTORS subset if one was given.
    if [[ -n "${FACTORS}" ]] && ! grep -qw "${f}" <<< "${FACTORS}"; then
        continue
    fi
    # Allow an explicit iteration override.
    [[ -n "${ITERS_OVERRIDE}" ]] && iters="${ITERS_OVERRIDE}"

    # Sequence parallelism splits one sequence across SP ranks, so the job
    # must have at least that many. Skip rather than fail: a 1-node job
    # legitimately cannot run the SP=24 point.
    if (( sp > NRANKS )); then
        echo "--- SKIP factor=${f} (SP=${sp} needs ${sp} ranks, have ${NRANKS}) ---"
        continue
    fi
    if (( NRANKS % sp != 0 )); then
        echo "--- SKIP factor=${f} (world ${NRANKS} not divisible by SP=${sp}) ---"
        continue
    fi
    if (( NNODES < min_nodes )) && [[ "${FORCE_ALL:-0}" != "1" ]]; then
        echo "--- SKIP factor=${f} (needs >=${min_nodes} nodes to finish in a"
        echo "    reasonable walltime; set FORCE_ALL=1 to run it anyway) ---"
        continue
    fi

    log="${RESULTS_DIR}/vit_n${NNODES}_factor${f}_sp${sp}_rep${REP}.log"
    echo "######## factor=${f}  seq=$((f*f*f))  SP=${sp}  iters=${iters} ########"
    echo "         log: ${log}"

    env factor="${f}" SP="${sp}" NUM_ITERS="${iters}" \
        bash "${ViT_ROOT}/mds_launch.sh" 2>&1 | tee "${log}"
    rc="${PIPESTATUS[0]}"

    if (( rc == 0 )); then
        echo "         factor=${f} exit 0"
    else
        echo "         factor=${f} exit ${rc}  <-- FAILED"
        overall="${rc}"
    fi
    echo
done

echo "=== sweep complete, nodes=${NNODES} rep=${REP}, exit ${overall} ==="
echo "=== summarize with: python3 ${REPO_WORKDIR}/ViT/summarize_results.py ${RESULTS_DIR} ==="
exit "${overall}"
