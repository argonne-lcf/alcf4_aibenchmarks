# DBSCAN

## Overview 
DBSCAN (Density-based Spatial Clustering of Applications with Noise) is a popular clustering algorithm. It identifies dense neighborhoods of points as clusters while points in low-density regions are left as outliers. This benchmark uses the distributed DBSCAN algorithm implemented in the ArborX library, a performance portable geometric search library using Kokkos. Kokkos supports CPUs and GPUs from Nvidia, AMD, and Intel through backends such as OpenMP, CUDA, SYCL, and HIP.  

## Code Access
This benchmark is part of the open source [ArborX](https://github.com/arborx/ArborX) library.

## FOM
 ```math
 FOM = \frac{n}{T}
```
where $n$ is the number of points to cluster and $T$ is the time to solution. 

## Steps to Run
This benchmark is part of the [ArborX repository](https://github.com/arborx/ArborX).

### Installation
See the ArborX installation instructions. There are recent installation instructions specific to running this benchmark on Aurora on [this page](install.md). 

### Example command
```
cd /ArborX/build/benchmarks/cluster
mpiexec --no-transfer -np 12 --ppn 12 --cpu-bind=list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 gpu_tile_compact.sh ./ArborX_Benchmark_DistributedDBSCAN.exe --eps 1 --dimension 3 --n 400000000 --num-seq 100
```


