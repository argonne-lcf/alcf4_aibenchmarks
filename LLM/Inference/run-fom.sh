#!/bin/bash -l
#PBS -l select=10
#PBS -l walltime=0:60:00
#PBS -q debug-scaling
#PBS -l filesystems=flare
#PBS -A datascience


start_vllm_serve(){
    echo "Startign vLLM Serve..."
    log_file="nohup.out"
    rm $log_file
    
    vllm serve $1 --port 8000 --tensor-parallel-size $2 --pipeline-parallel-size $3 --device xpu --dtype float16 --trust-remote-code --max-model-len $4 --gpu-memory-utilization $5  &> $log_file &
    
    while true; do
        if [[ -f $log_file ]] && grep -q "INFO:     Application startup complete." "$log_file"; then
            echo "vLLM Serving Successfully model: $1 "
            return 0
        fi
        tail -5 $log_file
        echo "Checking if vLLM has started..."
        sleep 5
    done
    wait
    rm $log_file
}


export model_name="meta-llama/Llama-3.1-405B-Instruct"
export PROJ_DIR=/flare/datascience/sraskar/vllm-2025_1_release/vllm-2025_1/vllm/benchmarks/
cd $PROJ_DIR

export tp_size=8
export pp_size=`wc -l < $PBS_NODEFILE`
export context_length=32768

source $PROJ_DIR/setup_ray_cluster.sh
main 

start_vllm_serve $model_name $tp_size $pp_size $context_length 0.9

# Run the benchmark
python infr-bench.py --input-length 65000 --output-length 65000 --batch-size 1
python infr-bench.py --input-length 65000 --output-length 1 --batch-size 1
python infr-bench.py --input-length 512 --output-length 65000 --batch-size 1
