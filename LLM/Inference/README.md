## Overview 


We use `vLLM` which is an open-source library designed to optimize the inference and serving. Originally developed at UC Berkeley's Sky Computing Lab, it has evolved into a community-driven project. The library is built around the innovative PagedAttention algorithm, which significantly improves memory management by reducing waste in Key-Value (KV) cache memory.

We use Llama 3.1 405B model for FOM calculation which is one of the largest open-source language models developed by Meta AI, consisting of 405 billion weight parameters. It is built on a Transformer architecture with Group Query Attention at its core. The model is trained on vast datasets and finetuned with human feedback. Llama 3.1 405B supports a context length of 128k tokens and is multilingual, supporting eight languages. It offers various applications such as synthetic data generation, model distillation, and research. The model also includes safety features like Llama Guard 3 and Prompt Guard to mitigate harmful outputs and prompt injection attacks.

Throughput is a key indicator of a hardware’s processing efficiency. It provides insight into the model’s capacity to handle sequences and batches. We define throughput as the total number of tokens (both input and output) processed by the hardware per second. We first calculate the end-to-end latency, the time elapsed between the input prompt provided to LLM, and the generation of the final output token.

## Code Access


## Install vLLM on Aurora

1. SSH to an Aurora login node:
```bash linenums="1"
ssh <username>@aurora.alcf.anl.gov
```

2. Setup new virtual environment and install vLLM

```bash linenums="1" title="Install vLLM using pre-built wheels"
module load frameworks
conda create --name vllm python=3.10 -y
conda activate vllm

module unload oneapi/eng-compiler/2024.07.30.002
module use /opt/aurora/24.180.3/spack/unified/0.8.0/install/modulefiles/oneapi/2024.07.30.002
module use /soft/preview/pe/24.347.0-RC2/modulefiles
module add oneapi/release

pip install /flare/datasets/softwares/vllm-install/wheels/*
pip install /flare/datasets/softwares/vllm-install/vllm-0.6.6.post2.dev28+g5dbf8545.d20250129.xpu-py3-none-any.whl
```

## Access Model Weights

Model weights for commonly used open-weight models are downloaded and available in the following directory on Aurora:
```bash linenums="1"
/flare/datascience/model-weights/hub
```

To ensure your workflows utilize the preloaded model weights and datasets, update the following environment variables in your session. Some models hosted on Hugging Face may be gated, requiring additional authentication. 

To access these gated models, you will need a [Hugging Face authentication token](https://huggingface.co/docs/hub/en/security-tokens).
```bash linenums="1"
export HF_HOME="/flare/datascience/model-weights/hub"
export HF_DATASETS_CACHE="/flare/datascience/model-weights/hub"
export HF_TOKEN="YOUR_HF_TOKEN"
export RAY_TMPDIR="/tmp"
export TMPDIR="/tmp"
```

## Running Experiment to collect FOM

The following example serves `meta-llama/Llama-3.1-405B-Instruct` model using 2 nodes with `TP=8` and `PP=10`. Models exceeding 70 billion parameters generally require more than one Aurora node.


Simplly run `run-fom.sh` script. This scripts serves `meta-llama/Llama-3.1-405B-Instruct` model using 10 nodes with Tensor Parallel = 8 and Pipeline Parallel =10. It has following steps:

1. [`setup_ray_cluster.sh`](./setup_ray_cluster.sh) script sets up a Ray cluster across nodes.

    ```bash linenums="1"
    source setup_ray_cluster.sh
    main
    ```

2. Serve model using vLLM.
    ```
    export tp_size=8
    export pp_size=`wc -l < $PBS_NODEFILE`
    export context_length=32768
    start_vllm_serve $model_name $tp_size $pp_size $context_length 0.9
    ```
    Setting `--max-model-len` is important in order to fit this model on 2 nodes. In order to use higher `--max-model-len` values, you will need to use additonal nodes. 

2. Finally `infr_bench.py` script is used to collect throughput metric in the respective `csv` file. 

    ```bash
    python infr-bench.py --input-length 8192 --output-length 8192 --batch-size 1
    python infr-bench.py --input-length 8192 --output-length 1 --batch-size 1
    python infr-bench.py --input-length 512 --output-length 8192 --batch-size 1
    ```
    We collect 3 scenarios: 
    (1) same input, output lenght of `8192` tokens
    (2) Prefill Stage: `8192` input length, `1` output length
    (3) decode stage: `512` input length, `8192` output length
    `max-model-len` for all scneraios is `32768`

## Figure of Merit 
Since inference includes two phases, prefill (compute-bound) and decode (memory bandwidth-bound), we introduce problem complexities (C) separately for two phases.

```math
C_{prefill} = { b N (6d^2 T_{in} + d T_{in}^2)}
```

```math
C_{decode} = { b N T_{out} * 6d^2}
```

```math 
FOM = \frac {C_{prefill} + C_{decode}} {L}
```

```bash
where 
      N = Number of layers
      b = global batch size
      d = hidden dimension of the model
      T_{in} = input token length
      T_{out} = output token length
      L = End-to-end latency 
```


### Figure of Metrit on Aurora


### `meta-llama/Llama-3.1-8B-Instruct`
| Number of Nodes | Tiles per Node | Input Length | Output Length | Time to Solution | FOM |
|---|---|---|---|---|---|
|1|1|8192|8192|3.123|1.97 × 10^13|
|1|1|8192|1|1.183|2.98 × 10^13|
|1|1|512|8192|1.993|1.41 × 10^13|
|1|8|8192|8192|15.64|3.94 × 10^12|
|1|8|8192|1|0.568|6.2 × 10^13|
|1|8|512|8192|6.498|4.32 × 10^12|

### `meta-llama/Llama-3.3-70B-Instruct`
| Number of Nodes | Tiles per Node | Input Length | Output Length | Time to Solution | FOM |
|---|---|---|---|---|---|
|1|8|8192|8192|10.253|5.58 × 10^13|
|1|8|8192|1|2.046|1.5 × 10^14|
|1|8|512|8192|11.574|2.42 × 10^13|
|2|8|8192|8192|14.642|3.9 × 10^13|
|2|8|8192|1|2.31|1.33 × 10^14|
|2|8|512|8192|22.859|1.23 × 10^13|
|4|8|8192|8192|10.726|5.33 × 10^13|
|4|8|8192|1|3.103|9.92 × 10^13|
|4|8|512|8192|6.168|4.55 × 10^13|

### `meta-llama/Llama-3.1-405B-Instruct`
| Number of Nodes | Tiles per Node | Input Length | Output Length | Time to Solution | FOM |
|---|---|---|---|---|---|
|2|8|1024|1024|39.319|1.36 × 10^11|
|2|8|8100|1|8.376|1.77 × 10^13|
|2|8|512|7168|18.13|6.91 × 10^11|
|4|8|8192|8192|105.625|1.55 × 10^12|
|4|8|8192|1|9.385|1.61 × 10^13|
|4|8|512|8192|8.435|1.67 × 10^12|
|8|8|8192|8192|26.528|6.19 × 10^12|
|8|8|8192|1|10.802|1.4 × 10^13|
|8|8|512|8192|7.551|1.87 × 10^12|
|10|8|8192|8192|160.146|1.02 × 10^12|
|10|8|8192|1|11.579|1.31 × 10^13|
|10|8|512|8192|12.688|1.11 × 10^12|
