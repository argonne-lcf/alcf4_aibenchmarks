
# AICommsBench (AI collective communications benchmark)

## Overview 

This mini-app captures the performance effects on major collectives (`allreduce`,
`allgather`, `reduce-scatter`, `alltoall`) in the presence of proxy compute (for example, 
GEMM calculations). The mini-app is designed to accommodate different 
parallelisms such as tensor parallelism, sequence parallelism, data parallelism 
-- commonly used in LLM and ViT training schemes. We also have a proxy
implementation of the communication pattern of the Ulysses sequence 
parallelism.

### Important Features
In this benchmark we tried to capture the dominant communication patterns in LLM
and ViT training while remaining faithful to the full application. We measure 
the timing of the collectives in the presence of proxy compute which presents a
more realistic (although not complete) scenario than measuring them in isolation.

In capturing the communication patterns, we tried to implement different 
communication groups following the patterns in the full application. This 
captures the main essence of the sub-communicators in the training scheme which
plays important role in the estimating the training iteration time.

_Note_: We are working on a code path, where this mini-app can be utilized as a
communication only (no compute) benchmark.

## Code Access

There are two scripts in the repository:

`tensor_parallel_with_gradient_synchronization_a2a_debug.py`: The main Python file with 
the PyTorch implementation of the benchmark mini-app.

`qsub_aurora_1t_analytical.sh`: An example PBS job submission
script to run on Aurora.

### Important Arguments
The most important arguments that the mini-app takes are the following:

- `sequence_length`, `hidden_dimension`
- `tp_degree`: Degree of tensor parallelism, representing the `allreduce` communication pattern.
- `sp_switch`: Allowing sequence parallelism, representing the `allgather`, `reduce-scatter` communication patterns.
- `ulysses_enable`: Allowing Ulysses sequence parallelism, representing the `alltoall` communication pattern

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


