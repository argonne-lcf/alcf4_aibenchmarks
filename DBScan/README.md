# DBScan

## Overview 

## Code Access

## FOM

## Steps to Run
This benchmark is part of the [ArborX repository](https://github.com/arborx/ArborX).

### Installation
See the ArborX installation instructions. There are recent installation instructions specific to running this benchmark on Aurora on [this page](install.md). 

### Example command
```
cd /ArborX/build/benchmarks/cluster
mpiexec --no-transfer -np 12 --ppn 12 --cpu-bind=list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 gpu_tile_compact.sh ./ArborX_Benchmark_DistributedDBSCAN.exe --eps 1 --dimension 3 --n 400000000 --num-seq 100
```


