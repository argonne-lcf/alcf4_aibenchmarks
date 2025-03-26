# DBSCAN

## Overview 
DBSCAN (Density-based Spatial Clustering of Applications with Noise) is a popular clustering algorithm [1]. It identifies dense neighborhoods of points as clusters while points in low-density regions are left as outliers. This benchmark uses the distributed DBSCAN algorithm implemented in the [ArborX](https://github.com/arborx/ArborX) library, a performance portable geometric search library using [Kokkos](https://kokkos.org) [2]. Kokkos supports CPUs and GPUs from Nvidia, AMD, and Intel through backends such as OpenMP, CUDA, SYCL, and HIP.  

### Science Motivation
ArborX was developed as part of the Exascale Computing Project (ECP) with the goal to support DOE applications at scale. One exascale application that ArborX supports is HACC (see below), which uses DBSCAN [3]. However, there are eight additional applications using ArborX in production listed [here](https://github.com/arborx/ArborX/wiki/AppsUsingArborx).

Halo finding in cosmology: one of the recurring analysis steps during cosmological N-body simulations is identifying halos (regions with a high density of dark matter particles). The Hardware Accelerated Cosmology Code (HACC) framework uses ArborX's implementation of DBSCAN to find halos. Here, the dataset to cluster is composed of the 3-D positions of particles [3]. 

Downsampling data in computational fluid dynamics (CFD): it can be challenging to train reduced-order machine learned models on data from fluid dynamics simulations because large meshes result in many features (high-dimensional datasets). Brewer, et al. designed a method for subsampling meshes using clustering [4]. Although they demonstrated their method on smaller problems, CFD simulations on leadership-class supercomputers often have meshes that do not fit in memory on a single node.

Identifying cell populations in single-cell analysis: single-cell data, such as data from mass cytometry, can quickly become large. FastPG was designed to quickly cluster millions of cells in order to identify cell populations [5]. It was funded by ECP through ExaGraph and is integrated into [MCMICRO](https://mcmicro.org/), a pipeline that transforms multi-channel whole-slide images into single-cell data [6]. 


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
[1] [Original DBSCAN paper:](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. "A density-based algorithm for discovering clusters in large spatial databases with noise." In KDD, vol. 96, no. 34, pp. 226-231. 1996.

[2] [Paper on DBSCAN in ArborX:](https://dl.acm.org/doi/10.1145/3605573.3605594)
Andrey Prokopenko, Damien Lebrun-Grandié, and Daniel Arndt. 2023. Fast tree-based algorithms for DBSCAN for low-dimensional data on GPUs. In 52nd International Conference on Parallel Processing (ICPP 2023), August 07–10, 2023, Salt Lake City, UT, USA. ACM, New York, NY, USA, 10 pages. 

[3] [Paper on improvements to ArborX to support HACC:](https://journals.sagepub.com/doi/abs/10.1177/10943420241298296)
Prokopenko, Andrey, Daniel Arndt, Damien Lebrun-Grandié, Bruno Turcksin, Nicholas Frontiere, J. D. Emberson, and Michael Buehlmann. "Advances in ArborX to support exascale applications." The International Journal of High Performance Computing Applications 39, no. 1 (2025): 167-176.

[4] [Paper on clustering to subsample CFD meshes:](https://dl.acm.org/doi/pdf/10.1145/3624062.3626084)
Brewer, Wesley, Daniel Martinez, Muralikrishnan Gopalakrishnan Meena, Aditya Kashi, Katarzyna Borowiec, Siyan Liu, Christopher Pilmaier, Greg Burgreen, and Shanti Bhushan. "Entropy-driven Optimal Sub-sampling of Fluid Dynamics for Developing Machine-learned Surrogates." In Proceedings of the SC'23 Workshops of the International Conference on High Performance Computing, Network, Storage, and Analysis, pp. 73-80. 2023.

[5] [Paper on FastPG for clustering cells:](https://www.biorxiv.org/content/10.1101/2020.06.19.159749v2.full.pdf)
Bodenheimer, Tom, Mahantesh Halappanavar, Stuart Jefferys, Ryan Gibson, Siyao Liu, Peter J. Mucha, Natalie Stanley, Joel S. Parker, and Sara R. Selitsky. "FastPG: fast clustering of millions of single cells." BioRxiv (2020): 2020-06.

[6] [Paper on MCMICRO:](https://www.nature.com/articles/s41592-021-01308-y.pdf)
Schapiro, Denis, Artem Sokolov, Clarence Yapp, Yu-An Chen, Jeremy L. Muhlich, Joshua Hess, Allison L. Creason et al. "MCMICRO: a scalable, modular image-processing pipeline for multiplexed tissue imaging." Nature methods 19, no. 3 (2022): 311-315.


