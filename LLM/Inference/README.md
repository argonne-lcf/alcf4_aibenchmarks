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
    python generic-bench.py --input-length 1024 --output-length 1024 --batch-size 1
    ```

## Collect FOM metric 

1. `FOM1` : Collect throughput with large `input-length` and moderate `output-length` (pre-fill compute intensive)
2. `FOM2` : Colelct throughput with moderate `input-length` and large `output-length` (decode memory intensive)
3. `FOM3` : Collect throughput with moderate `input-length`, `output-length` but large `max-model-len` (need mulltiple nodes i/o intensive)

```python 
FOM = FOM1 + FOM2 + FOM3
```


## FOM for Aurora




