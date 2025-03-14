# Deep Learning I/O benchmarks

## Overview 
Deep Learning I/O (DLIO) Benchmark is a benchmark suite aiming at emulating the I/O pattern and behavior of deep learning applications. The benchmark is delivered as an executable that can be configured for various deep learning workloads. It uses a modular design to incorporate different data loaders, data formats, dataset organizations, and use training configuration parameters similar to the actual deep learning applications. It is able to represent the I/O process of a broad spectrum of deep leanrning applications.

We are using DLIO. 

## Code Access
The benchmark is publically available 
GitHub: https://github.com/argonne-lcf/dlio_benchmark

## FOM
This set of benchmarks are microbenchmarks. We do not define FOM here, but the benchmark will have pass/fail, and throughput metric. We will add requirements here later. 

## Steps to Run
Install DLIO
```bash
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark
CC=mpicc CXX=mpicxx python setup.py build
CC=mpicc CXX=mpicxx python setup.py install
```

### Read benchmark
We have three workloads (ResNet50, UNet3D, CosmoFlow). Below are commands to run CosmoFlow workload

* Generating the datasets
```bash
mpiexec -np $NP --ppn $PPN --cpu-bind XXX dlio_benchmark workload=cosmoflow_a100 \
    ++workflow.generate_data=True \
    ++workload.workflow.train=False \
    ++workload.dataset.data_folder=$STORAGE_FOLDER \
```
* Running the benchmarks
```bash
mpiexec -np $NP --ppn $PPN --cpu-bind XXX dlio_benchmark workload=cosmoflow_a100 \
    ++workflow.generate_data=False \
    ++workload.workflow.train=True \
    ++workload.dataset.data_folder=$STORAGE_FOLDER \
```
More information will be added here such as number of files, samples that we want the vendor to test. 

### Checkpoint write benchmark
In this ALCF-4 benchmarks, we would like vendor to test the performance for Llama 3 transformer type model. Currently we have 8B, 70B, 405B, 1T configurations available for testing. Below are commands to run 1T model. The total size of checkpoint data each iteration will be about 17TB. 
```bash
mpiexec -np 1024 –ppn 8 –cpu-bind list:xxxxx python3 dlio_benchmark workload=llama_1t
```
