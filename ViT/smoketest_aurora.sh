#!/bin/bash -l
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -N vit-smoke

# Environment smoke test for the ALCF-4 3D ViT benchmark on Aurora.
#
# This is NOT the benchmark. It answers the questions a login node cannot:
# the login nodes have no PVCs, so torch reports no XPU and DeepSpeed falls
# back to the CPU accelerator -- every XPU code path is untested until a job
# runs. In order, this checks:
#
#   1. torch's XPU build sees the 12 tiles
#   2. DeepSpeed selects the 'xpu' accelerator and a comm backend
#   3. megatron.model imports            <- the IPEX guard; the known blocker
#   4. FlashAttentionBuilder JIT-builds  <- what FA=1 needs
#   5. Ulysses DistributedAttention is importable
#   6. an fp16 matmul runs on device
#   7. deepspeed.init_distributed() brings up a 12-rank group over XCCL
#   8. all_reduce and all_to_all are correct  <- all_to_all is the Ulysses
#      primitive; if SP is to work at scale it has to work here first
#
# Then, unless SKIP_TRAIN=1, it runs the smallest real training config
# (factor=16 => 4K sequence, SP=1, 3 iterations) to confirm the whole stack
# trains end to end and prints the TFLOPS/sample-rate line the FOM comes from.
#
# Submit (1 node, debug queue):
#   qsub -l select=1 -q debug -A <project> ViT/smoketest_aurora.sh
#
# Options via -v:
#   SKIP_TRAIN=1   environment checks only, no training step
#   NUM_ITERS=5    training iterations (default 3)
#   factor=24      sequence-length factor (default 16 => 4K tokens)

cd "${PBS_O_WORKDIR}"

# PBS runs a copy of this script from its spool directory, so BASH_SOURCE
# cannot locate the repo. Derive paths from PBS_O_WORKDIR; override with -v.
: "${ViT_DIR:=${PBS_O_WORKDIR}/ViT}"
: "${ENV_SH:=${ViT_DIR}/env_aurora.sh}"
: "${ViT_ROOT:=${ViT_DIR}/Megatron-DeepSpeed_ViT}"

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

# Binding and CCL settings carried over from mds_launch.sh's Aurora setup.
# NOTE: the CCL_* variables were tuned for the oneCCL bindings, which this
# stack no longer uses (native XCCL instead). They are kept here so the smoke
# test matches the launcher; whether they still do anything under XCCL is an
# open question and is deliberately one of the things this run informs.
export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export NUMEXPR_MAX_THREADS=7
export OMP_NUM_THREADS=7

export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export PALS_PMI=pmix
export CCL_ATL_TRANSPORT=mpi

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(( RANDOM + 1024 ))
export WANDB_MODE=disabled

echo "=== ViT environment smoke test ==="
echo "=== nodes: ${NNODES}  ranks: ${NRANKS}  host: $(hostname) ==="
echo "=== date: $(date) ==="
echo "=== ViT_ROOT: ${ViT_ROOT} ==="
echo
echo "--- module list ---"
module list 2>&1
echo
echo "--- python ---"
which python3
python3 --version
echo

# ---------------------------------------------------------------------------
# Stage 1: per-rank environment checks
# ---------------------------------------------------------------------------
echo "############ STAGE 1: environment checks (${NRANKS} ranks) ############"
mpiexec --envall -n "${NRANKS}" --ppn "${RANKS_PER_NODE}" \
        --cpu-bind="${CPU_BIND}" \
    python3 "${ViT_DIR}/smoketest_env.py"
STAGE1=$?
echo "=== stage 1 exit: ${STAGE1} ==="

if (( STAGE1 != 0 )); then
    echo
    echo "Environment checks failed. Skipping the training step -- fix the"
    echo "failures above first; a training run would only re-report them."
    exit "${STAGE1}"
fi

echo
echo "############ GPU residency evidence ############"
mpiexec --envall -n "${NRANKS}" --ppn "${RANKS_PER_NODE}" \
        --cpu-bind="${CPU_BIND}" \
    python3 "${ViT_DIR}/gpu_residency.py"
echo "=== gpu residency exit: $? ==="

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "SKIP_TRAIN=1 set; stopping after environment checks."
    exit 0
fi

# ---------------------------------------------------------------------------
# Stage 2: smallest real training config, end to end
# ---------------------------------------------------------------------------
# factor=16 => IMG 256^3, patch 16 => seq len 16^3 = 4096 tokens, SP=1.
# This is the smallest of the five FOM points, and the only one that fits
# comfortably on a single node -- enough to prove the stack trains and emits
# the "iteration: ... TFLOPS: ... Samples/Sec: ..." line the FOM reads.
echo
echo "############ STAGE 2: training, factor=${factor:-16}, ${NUM_ITERS:-3} iters ############"

export VIT="315M"
export VIT3D=1
# mds_launch.sh rejects having both set, so default to fp16 only when bf16
# was not requested (-v bf16=1).
if [[ "${bf16:-0}" == "1" ]]; then
    export bf16=1
    export fp16=0
else
    export fp16=1
    export bf16=0
fi
export ZERO=1
export MBS=1
export FA=1
export SP=${SP:-1}
export factor=${factor:-16}
export NUM_ITERS=${NUM_ITERS:-3}

# mds_launch.sh writes its DeepSpeed config into PBS_O_WORKDIR and resolves
# the model script relative to it, so it has to run with the submodule as
# the working directory.
export PBS_O_WORKDIR="${ViT_ROOT}"
cd "${ViT_ROOT}"

bash "${ViT_ROOT}/mds_launch.sh"
STAGE2=$?
echo "=== stage 2 exit: ${STAGE2} ==="

echo
if (( STAGE2 == 0 )); then
    echo "RESULT: environment checks and a ${NUM_ITERS}-iteration training run both completed."
else
    echo "RESULT: environment checks passed; training exited ${STAGE2} (see above)."
fi
exit "${STAGE2}"
