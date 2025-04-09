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

## Running a sample Experiment 

The following example serves `meta-llama/Llama-3.1-405B-Instruct` model using 2 nodes with `TP=8` and `PP=2`. Models exceeding 70 billion parameters generally require more than one Aurora node.

1. Use [`setup_ray_cluster.sh`](./setup_ray_cluster.sh) script to setup a Ray cluster across nodes
    ```bash
    source setup_ray_cluster.sh
    ```

2. Serve model

    ```bash linenums="1"
    vllm serve meta-llama/Llama-3.1-405B-Instruct --port 8000 --tensor-parallel-size 8 --pipeline-parallel-size 2 --device xpu --dtype float16 --trust-remote-code --max-model-len 4096
    ```
    Setting `--max-model-len` is important in order to fit this model on 2 nodes. In order to use higher `--max-model-len` values, you will need to use additonal nodes. 

3. Use `infr_bench.py` script provided here to collect throughput measurements in the respective `csv` file. 

    ```bash
    python infr_bench.py --input-length 1024 --output-length 1024 --batch-size 1
    ```

## FOM 
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

<!--
1. `FOM1` : Collect throughput with large `input-length` and moderate `output-length` (pre-fill compute intensive)
2. `FOM2` : Colelct throughput with moderate `input-length` and large `output-length` (decode memory intensive)
3. `FOM3` : Collect throughput with moderate `input-length`, `output-length` but large `max-model-len` (need mulltiple nodes i/o intensive)

```python 
FOM = FOM1 + FOM2 + FOM3
```


## FOM for Aurora

-->


## meta-llama/Llama-3.1-8B-Instruct
| Number of Nodes | Tiles per Node | Time to Solution | FOM |
|---|---|---|---|
|1|1|3.123|2.77 × 10^17|
|1|1|1.183|1.83 × 10^17|
|1|1|1.993|1.22 × 10^17|
|1|8|15.64|5.53 × 10^16|
|1|8|0.568|3.81 × 10^17|
|1|8|6.498|3.76 × 10^16|

## meta-llama/Llama-3.3-70B-Instruct
| Number of Nodes | Tiles per Node | Time to Solution | FOM |
|---|---|---|---|
|1|8|10.253|8.43 × 10^17|
|1|8|2.046|1.06 × 10^18|
|1|8|11.574|2.11 × 10^17|
|2|8|14.642|5.91 × 10^17|
|2|8|2.31|9.36 × 10^17|
|2|8|22.859|1.07 × 10^17|
|4|8|10.726|8.06 × 10^17|
|4|8|3.103|6.97 × 10^17|
|4|8|6.168|3.96 × 10^17|

## meta-llama/Llama-3.1-405B-Instruct
| Number of Nodes | Tiles per Node | Time to Solution | FOM |
|---|---|---|---|
|2|8|39.319|1.67 × 10^14|
|2|8|8.376|1.22 × 10^16|
|2|8|18.13|5.08 × 10^15|
|4|8|105.625|3.97 × 10^15|
|4|8|9.385|1.12 × 10^16|
|4|8|8.435|1.4 × 10^16|
|8|8|26.528|1.58 × 10^16|
|8|8|10.802|9.71 × 10^15|
|8|8|7.551|1.57 × 10^16|
|10|8|160.146|2.62 × 10^15|
|10|8|11.579|9.06 × 10^15|
|10|8|12.688|9.32 × 10^15|
