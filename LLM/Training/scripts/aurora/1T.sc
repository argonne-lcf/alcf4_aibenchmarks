#!/bin/sh
#PBS -l walltime=1:00:00
#PBS -A Aurora_deployment
#PBS -q lustre_scaling
#PBS -l select=64
#PBS -l filesystems=flare:home
#PBS -N 1T-daos
#PBS -ldaos=default

cd ${PBS_O_WORKDIR}
export FILESYSTEM=/flare/
export USE_DAOS=${USE_DAOS:-"1"}
export PPN=12
export CC=icx
export CXX=icpx
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export TRANSFER_PACKAGE=1
export BUILD=2024-09-27
export CHECKPOINT_ONLY=0
TRANSFER_PACKAGE=1 source ${FILESYSTEM}/Aurora_deployment/AuroraGPT/build/${BUILD}/conda.sh
export MD=${PBS_O_WORKDIR}/Megatron-DeepSpeed/
echo "####### Python environment ###########"
which python
echo "--------------------------------------"

echo "####### Megatron-DeepSpeed ###########"
echo "Using Megatron-DeepSpeed code from $MD"
cd $MD
echo "Git commit: `git rev-parse HEAD`"
cd - 

export TORCH_PROFILER_ENABLE=1
export DFTRACER_ENABLE=1
export DFTRACER_DISABLE_IO=1
#export DLIO_PROFILER_DISABLE_IO=1
#export DLIO_PROFILER_DATASET_DIR=$FILESYSTEM/Aurora_deployment/AuroraGPT/datasets/dolma/
IFS='.' read -ra ADDR <<< "$PBS_JOBID"
export JOBID=$ADDR
export PYTHONPATH=$MD:$PYTHONPATH

export DATE_TAG=$(date +"%Y-%m-%d-%H-%M-%S")

echo "BS: $BS - PP:$PP - TP: $TP, PBS_JOBSIZE: $PBS_JOBSIZE"

# Architecture setup
# 7B model configuration
HIDDEN_SIZE=24576
NUM_LAYERS=${NUM_LAYERS:-128}
SEQ_LENGTH=2048
ATTN_HEADS=192
FFN_HIDDEN_SIZE=98304
EMBEDDINGS=$SEQ_LENGTH

# Training setup
export TP=${TP:-12}
export PP=${PP:-64}
export MBS=${MBS:-1}
export OPT=${OPT:-"adamw"}
export ZERO_STAGE=${ZERO_STAGE:-1}
export GRADIENT_ACC=${GAS:-$((8*PP))}
export MICS_SHARD_SIZE=${MICS_SHARD_SIZE:-12}
export SAVE_INTERVAL=${SAVE_INTERVAL:-5}
export BS=$((MBS*PBS_JOBSIZE*PPN/PP/TP*GRADIENT_ACC))
export SP=$((PBS_JOBSIZE*PPN/PP/TP))
export NUM_TOKENS=2000000000000
#export TRAIN_ITERS=$((NUM_TOKENS/BS/SEQ_LENGTH))
export TRAIN_ITERS=30
export TRAIN_SAMPLES=$((TRAIN_ITERS*BS))
#export TRAIN_SAMPLES=$((NUM_TOKENS/SEQ_LENGTH))

MODEL=1T
OUTPUT_PREFIX=${MODEL}_z${ZERO_STAGE}_seqlen${SEQ_LENGTH}_mp${MP}_pp${PP}_sp${SP}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_gb${BS}_mb${MBS}_opt${OPT}

mkdir -p ${PBS_O_WORKDIR}/outputs/${OUTPUT_PREFIX}/$DATE_TAG
mkdir -p ${PBS_O_WORKDIR}/checkpoints/${OUTPUT_PREFIX}/$DATE_TAG


sed -e "s/STAGE/$ZERO_STAGE/g" \
    -e "s/MICS_SHARD_SIZE/${MICS_SHARD_SIZE}/g" \
    -e "s/MICRO_BATCH_SIZE/$MBS/g" \
    -e "s/ALLGATHER_PARTITIONS/true/g" \
    -e "s/REDUCE_SCATTER/false/g" \
    -e "s/ALLGATHER_BUCKET_SIZE/5e8/g" \
    -e "s/REDUCE_BUCKET_SIZE/3e7/g" \
    -e "s/OVERLAP_COMM/false/g" \
    -e "s/GRADIENT_ACC/$GRADIENT_ACC/g" $AGPT_ROOT/soft/DEEPSPEED_CONFIG_TEMPLATE.json > ds_config.$JOBID.json
    
export DATA_FILE_LIST="$FILESYSTEM/Aurora_deployment/AuroraGPT/datasets/dolma/dolma_v1_7_file_list_v2.txt"
export TENSORBOARD_DIR=${PBS_O_WORKDIR}/outputs/${OUTPUT_PREFIX}/$DATE_TAG/tensorboard/ 
export TRACE_DIR=${PBS_O_WORKDIR}/outputs/${OUTPUT_PREFIX}/$DATE_TAG/trace/
export TOKENIZER_MODEL=${AGPT_ROOT}/soft/tokenization/tokenizer.model 

export DS_CONFIG=${PBS_O_WORKDIR}/ds_config.$JOBID.json
#export DS_CONFIG=${PBS_O_WORKDIR}/ds_config.json
export CPU_BIND=list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96

export PATH=${FILESYSTEM}/Aurora_deployment/AuroraGPT/soft/:$PATH
export JOBSIZE=${JOBSIZE:-$PBS_JOBSIZE}
export DATA_CACHE_PATH=${PBS_O_WORKDIR}/data_cache_path_n${PBS_JOBSIZE}_s${SEQ_LENGTH}_ns${TRAIN_SAMPLES}/
# create checkpoint dir 
if [ $USE_DAOS -eq "1" ]; then
    module use /soft/modulefiles
    module load daos
    DAOS_POOL=LLM-GPT-Intel
    DAOS_CONT=LLM-1T-checkpoint-$JOBID
    daos container destroy ${DAOS_POOL} ${DAOS_CONT} 
    daos container create --type POSIX ${DAOS_POOL} ${DAOS_CONT} --properties rd_fac:1
    launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
    ls /tmp/${DAOS_POOL}/${DAOS_CONT}
    export CHECKPOINT_DIR=/tmp/${DAOS_POOL}/${DAOS_CONT}/checkpoints/${OUTPUT_PREFIX}/
    #export DATA_CACHE_PATH=/tmp/${DAOS_POOL}/${DAOS_CONT}/data_cache_path_n${PBS_JOBSIZE}_s${SEQ_LENGTH}_ns${TRAIN_SAMPLES}/
else
    export CHECKPOINT_DIR=$PBS_O_WORKDIR/checkpoints/${OUTPUT_PREFIX}/$DATE_TAG

fi
mkdir -p $CHECKPOINT_DIR
    
runcmd="mpiexec --no-vni --pmi=pmix -np $((PBS_JOBSIZE*PPN)) --ppn $PPN --cpu-bind $CPU_BIND $AGPT_ROOT/soft/interposer.sh $AGPT_ROOT/soft/local_rank.sh python3 ${MD}/pretrain_gpt_alcf.py \
      --tensor-model-parallel-size ${TP} \
      --pipeline-model-parallel-size ${PP} \
      --num-layers ${NUM_LAYERS} \
      --hidden-size ${HIDDEN_SIZE} \
      --num-attention-heads ${ATTN_HEADS} \
      --micro-batch-size ${MBS} \
      --global-batch-size ${BS} \
      --seq-length ${SEQ_LENGTH} \
      --max-position-embeddings ${EMBEDDINGS} \
      --train-samples ${TRAIN_SAMPLES} \
      --save ${CHECKPOINT_DIR} \
      --load ${CHECKPOINT_DIR} \
      --tensorboard-dir ${PBS_O_WORKDIR}/outputs/${OUTPUT_PREFIX}/$DATE_TAG/tensorboard/ \
      --log-timers-to-tensorboard --tensorboard-log-interval 1 \
      --trace-dir ${PBS_O_WORKDIR}/outputs/${OUTPUT_PREFIX}/$DATE_TAG/trace/ \
      --data-file-list ${DATA_FILE_LIST} \
      --vocab-file ${MD}/dataset/gpt2-vocab.json --merge-file ${MD}/dataset/gpt2-merges.txt \
      --zero-stage=${ZERO_STAGE} --deepspeed_config=${DS_CONFIG} \
      --data-cache-path $DATA_CACHE_PATH \
      --tokenizer-model ${AGPT_ROOT}/tokenization/tokenizer.model \
      --tensor-model-parallel-size ${TP} --pipeline-model-parallel-size ${PP} \
      --bf16 --split 100,0,0   --log-interval 1  --no-bias-gelu-fusion \
      --lr-decay-style cosine  --no-bias-dropout-fusion  --no-masked-softmax-fusion \
      --tokenizer-type Llama2Tokenizer  --no-gradient-accumulation-fusion \
      --accumulate-allreduce-grads-in-fp32  --use-checkpoint-opt_param-scheduler  --log-timers-to-tensorboard \
      --log-optimizer-states-to-tensorboard  --lr 0.003 --optimizer ${OPT} --eval-iters 10  --distributed-backend ccl \
      --save-interval ${SAVE_INTERVAL} --eval-interval 50000 --ffn-hidden-size ${FFN_HIDDEN_SIZE} --lr-warmup-fraction 0.05 \
      --no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights \
      --swiglu --normalization rmsnorm --disable-bias-linear --timing-log-level 1 --log-timers-to-tensorboard \
      --log-optimizer-states-to-tensorboard --deepspeed-activation-checkpointing \
      --deepspeed --checkpoint-activations --checkpoint-num-layers 1 \
      --use-flash-attn-builder \
      --attention-dropout 0 --hidden-dropout=0"
export PATH=${FILESYSTEM}/Aurora_deployment/AuroraGPT/soft/checkpoint_restart/:$PATH
check_hang.py --timeout 1000 --check 60 --command python3 --output $PBS_JOBNAME.o$JOBID:$PBS_JOBNAME.e$JOBID:${PBS_O_WORKDIR}/outputs/${OUTPUT_PREFIX}/$DATE_TAG/output.log >& ${PBS_O_WORKDIR}/check_hang.$JOBID.o &
$runcmd | tee -a ${PBS_O_WORKDIR}/outputs/${OUTPUT_PREFIX}/$DATE_TAG/output.log
pkill python
pkill python3

if [ $USE_DAOS -eq "1" ]; then
    clean-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
    fusermount3 -u /tmp/${DAOS_POOL}/${DAOS_CONT}
    daos container destroy ${DAOS_POOL} ${DAOS_CONT}
fi
