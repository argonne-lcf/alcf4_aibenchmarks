# DBSCAN

## Overview
Clustering is a data mining technique that splits a set of objects into disjoint
classes (clusters), each containing similar objects. DBSCAN (Density-Based
Spatial Clustering of Applications with Noise) is a popular clustering algorithm
[1], particularly useful when the number of clusters or their shape is not known a
priori. It is used in a diverse set of applications such as bioinformatics,
noise filtering and outlier detection, cosmology, image segmentation, and
others.

DBSCAN defines local density by the number of points within a given radius.
Dense neighborhoods form clusters, while the rest of the points in low-density
regions are considered outliers (noise).

This benchmark uses the distributed version of the DBSCAN algorithm, with the
local algorithm described in [2]. The benchmark is implemented within
[ArborX](https://github.com/arborx/ArborX) library, a performance portable
geometric search library that uses [Kokkos](https://kokkos.org) for on-node
parallelism. Kokkos supports CPUs and GPUs from Nvidia, AMD, and Intel through
backends such as OpenMP, CUDA, SYCL, and HIP. For Aurora GPUs, we use SYCL backend.

DBSCAN has been implemented in multiple ML related libraries: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html),
[RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/api/#dbscan), [R](https://cran.r-project.org/web/packages/dbscan/index.html).

### Science Motivation
ArborX was originally developed as part of the Software Techonolgies (ST) thrust
within Exascale Computing Project (ECP) with the primary goal to support DOE
applications at scale. One exascale application that ArborX supports is Hardware
Accelerated Cosmology Code (HACC), a cosmoloy framework developed at Argonne.
ArborX is a critical part of of the HACC's analysis tools, which uses DBSCAN [3]
and spherical overdensity algorithms.
In addition to HACC, there are at least eight additional applications using ArborX in
production listed [here](https://github.com/arborx/ArborX/wiki/AppsUsingArborx).

**Halo finding in cosmology:** one of the recurring analysis steps during
cosmological N-body simulations is identifying halos (regions with a high
density of dark matter particles). HACC framework uses ArborX's implementation
of DBSCAN to find halos. Here, the dataset to cluster is composed of the 3D
positions of particles [3].

**Intelligent subsampling of computational fluid dynamics (CFD) datasets:** training
reduced-order machine learned models on data from CFD simulations can be
challenging because of the large number grid points used by the solver to
describe the solution fields. This leads to very large sample sizes, increasing
the computational cost of training the models. Brewer, et al. designed a method
for intelligent subsampling CFD data by clustering the mesh points based on flow
descriptor variables and drawing samples from the clusters that maximize the
entropy of the cluster probability distributions [4].

**Identifying cell populations in single-cell analysis:** single-cell data, such as
data from mass cytometry, can quickly become large. FastPG was designed to
quickly cluster millions of cells in order to identify cell populations [5]. It
was funded by ECP through ExaGraph and is integrated into
[MCMICRO](https://mcmicro.org/), a pipeline that transforms multi-channel
whole-slide images into single-cell data [6].


## Code Access
This benchmark is part of the open source
[ArborX](https://github.com/arborx/ArborX) library.

## Figure-of-merit (FOM)
 ```math
 FOM = \frac{nd}{T}
```
where $n$ is the number of points to cluster, $d$ is their dimension, and $T$
is the time to solution. FOM is highly dependent on the data distribution and
the DBSCAN parameters $minPts$ and $\varepsilon$.

## Steps to Run
This benchmark is part of the [ArborX repository](https://github.com/arborx/ArborX).

### Installation
See the ArborX installation instructions. There are recent installation
instructions specific to running this benchmark on Aurora on [this
page](install.md).

### Example command
```shell
cd /ArborX/build/benchmarks/cluster
mpiexec --no-transfer -np 12 --ppn 12 --cpu-bind=list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 gpu_tile_compact.sh ./ArborX_Benchmark_DistributedDBSCAN.exe --eps 1 --dimension 3 --n 400000000 --num-seq 100
```
In the output, `n (global)` corrensponds to $n$ in FOM, and `total time`
corresponds to $T$.

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


