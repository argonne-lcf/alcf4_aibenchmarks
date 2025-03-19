## Overview 

Llama 3.1 405B is one of the largest open-source language models developed by Meta AI, consisting of 405 billion weight parameters. It is built on a Transformer architecture with Group Query Attention at its core. The model is trained on vast datasets and finetuned with human feedback. Llama 3.1 405B supports a context length of 128k tokens and is multilingual, supporting eight languages. It offers various applications such as synthetic data generation, model distillation, and research. The model also includes safety features like Llama Guard 3 and Prompt Guard to mitigate harmful outputs and prompt injection attacks.

Throughput is a key indicator of a hardware’s processing efficiency. It provides insight into the model’s capacity to handle sequences and batches. We define throughput as the total number of tokens (both input and output) processed by the hardware per second. We first calculate the end-to-end latency, the time elapsed between the input prompt provided to LLM, and the generation of the final output token.

## Code Access

## First time Setup
On Aurora, the steps to set up the required software environment are listed below. vLLM is used as the inference serving engine with Ray as the backend to enable multi-node runs.
```bash

module load cuda/12.3.0
source ~/.init_conda_x86.sh
conda create -n vllm python=3.10
conda activate vllm

pip install vllm

```

## Running a test Experiment 

```bash
git clone https://github.com/argonne-lcf/alcf4_aibenchmarks.git
cd alcf4_aibenchmarks/LLM/Inference

python benchmark_fom.py --batch-size=32 --tensor-parallel-size=1 --input-len=32--output-len=32 --model="meta-llama/Llama-2-7b-hf" --dtype="float16" --trust-remote-code
```
## FOM



## Run FOMs

Use `benchmark_fom.py` script provided here to collect throughput measurements in the respective `csv` file. 



### Collect FOM Metric

Use the provided shell script `run-fom.sh` in this directory to run `benchmark_fom.py` for various configurations of input, output lengths and batch sizes. 

```bash
    source run-fom.sh
```

