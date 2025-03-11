# Run Inference FOM

## First time Setup

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

## Run FOMs

Use `benchmark_fom.py` script provided here to collect throughput measurements in the respective `csv` file. 



### Collect FOM Metric

Use provided shell script `run-fom.sh` in this directory to run `benchmark_fom.py` for various configurations of input, output lengths and batch sizes. 

```bash
    source run-fom.sh
```

