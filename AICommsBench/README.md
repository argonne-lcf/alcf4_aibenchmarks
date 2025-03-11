
# AICommsBench (AI collective communications benchmark)

## Overview 

This mini-app captures the performance effects on major collectives (`allreduce`,
`allgather`, `reduce-scatter`) in the presence of proxy compute (for example, 
GEMM calculations). The mini-app is designed to accommodate different 
parallelisms such as tensor parallelism, sequence parallelism and data 
parallelism -- commonly used in LLM and ViT training schemes.

## Code Access

There are two scripts in the repository:

`tensor_parallel_with_gradient_synchronization.py`: The main Python file with 
the PyTorch implementation of the benchmark mini-app.

`qsub_aurora_no_sp_tensor_parallelism_compute.sh`: An example PBS job submission
script to run on Aurora.

## FOM

We do not have a strictly defined FOM, but we measure several important 
characteristic quantities regarding collectives (i.e. Throughout, and timing 
measurement of each collective operations)

## Steps to Run

This mini-app depends only on PyTorch and associated collective communication 
library for backends (i.e. NCCL, oneCCL). At this point of development, we 
tested the app on Nvidia A100 and Intel Data Center Max 1550 machines.

```
git clone https://github.com/argonne-lcf/alcf4_aibenchmarks.git

cd alcf4_aibenchmarks/AICommsBench

python ${WORK_DIR}/tensor_parallel_with_gradient_synchronization.py -dvc "xpu" \
-tp_degree=${TP_DEGREE} --warmup_iterations ${WARMUPS} --iterations=${TIMING_LOOPS} --precision ${PRECISION} -n_layers ${N_LAYERS} \
--logging --log_directory=${WORK_DIR}/run_scripts/outdir_aurora/logs --log_file=${RUN_ID}.log
```


