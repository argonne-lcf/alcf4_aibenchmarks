# DBSCAN

## Overview 
DBSCAN (Density-based Spatial Clustering of Applications with Noise) is a popular clustering algorithm. It identifies dense neighborhoods of points as clusters while points in low-density regions are left as outliers. This benchmark uses the distributed DBSCAN algorithm implemented in the [ArborX](https://github.com/arborx/ArborX) library, a performance portable geometric search library using [Kokkos](https://kokkos.org). Kokkos supports CPUs and GPUs from Nvidia, AMD, and Intel through backends such as OpenMP, CUDA, SYCL, and HIP.  

### Science Motivation
Halo finding in cosmology: one of the recurring analysis steps during cosmological N-body simulations is identifying halos (regions with a high density of dark matter particles). The Hardware Accelerated Cosmology Code (HACC) framework uses ArborX's implementation of DBSCAN to find halos. Here, the dataset to cluster is composed of the 3-D positions of particles.


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

## References
[Original DBSCAN paper:](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. "A density-based algorithm for discovering clusters in large spatial databases with noise." In KDD, vol. 96, no. 34, pp. 226-231. 1996.

[Paper on DBSCAN in ArborX:](https://dl.acm.org/doi/10.1145/3605573.3605594)
Andrey Prokopenko, Damien Lebrun-Grandié, and Daniel Arndt. 2023. Fast tree-based algorithms for DBSCAN for low-dimensional data on GPUs. In 52nd International Conference on Parallel Processing (ICPP 2023), August 07–10, 2023, Salt Lake City, UT, USA. ACM, New York, NY, USA, 10 pages. 

[Paper on improvements to ArborX to support HACC:](https://journals.sagepub.com/doi/abs/10.1177/10943420241298296)
Prokopenko, Andrey, Daniel Arndt, Damien Lebrun-Grandié, Bruno Turcksin, Nicholas Frontiere, J. D. Emberson, and Michael Buehlmann. "Advances in ArborX to support exascale applications." The International Journal of High Performance Computing Applications 39, no. 1 (2025): 167-176.
