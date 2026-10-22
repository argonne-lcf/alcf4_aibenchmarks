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
[ArborX](https://github.com/arborx/ArborX) library. The executable is `/benchmarks/cluster/ArborX_Benchmark_DistributedDBSCAN.exe`.

## Dataset
This benchmark uses three-dimensional synthetic data generated on each rank. The data generation code is in ArborX at [/benchmarks/cluster/distributed_data.hpp](https://github.com/arborx/ArborX/blob/master/benchmarks/cluster/distributed_data.hpp). Here is a sketch of the steps:
1. Factorize `comm_size` (the number of ranks) into three factors so that each rank can be assigned a different region in 3D space. For example, for 12 ranks, `factors = [3, 2, 2]` and for 48 ranks, `factors = [4, 4, 3]`.
2. Assign each rank indices `Is` representing how the ranks are arranged in 3D space. For example, for 12 ranks, rank 0 is assigned `Is = [0, 0, 0]` and rank 11 is assigned `Is = [2, 1, 1]`.
3. For simplicity, update `n`, the requested number of points per rank, to be a cube (`n = nx^3` for some integer `nx`). For example, for `--n 400000000`, `n` is updated to `n = 398688256 = 736^3`.
4. On each rank, generate `n` points with a Kokkos parallel for loop.
   - In each direction (`x, y, z`), there are `num_seq` points in a row followed by user-defined spacing. For example, if `num_seq = 50` and `spacing = 10`, the possible x coordinates are `x = 0, 1, 2, ..., 49, 59, 60, 61, ..., 108, 118, ...` The `pos` function handles adding the spacing. 
   - This means that if `1 < eps < spacing`, there are clear clusters with `num_seq^3` points. For example, for `num_seq = 50`, each cluster has 125000 points.
   - For `num_seq = 50`, the first cluster consists of the 125000 points where `x = 0, 1, 2, ..., 49`, `y = 0, 1, 2, ..., 49`, and `z = 0, 1, 2, ..., 49`.
   - The indices `Is` separate the ranks in 3D space. For example, for rank 0, `Is = [0, 0, 0]`, and for rank 1, `Is = [1, 0, 0]`, so the data generated on ranks 0 and 1 have the same y and z coordinates but separate x coordinates. Specifically, the largest x coordinate on rank 0 is `pos(nx-1)` and the x coordinates on rank 1 start at `pos(nx)`. Depending on the values of `nx`, `num_seq`, and `spacing`, the gap between `pos(nx-1)` and `pos(nx)` could be 1 (meaning that a cluster spans rank 0 and rank 1 in the x dimension), or there could be a gap of `spacing`.
     
## Figure-of-merit (FOM)
 ```math
 FOM = \frac{nd}{T}
```
where $n$ is the number of points to cluster, $d$ is their dimension, and $T$
is the time to solution. FOM is highly dependent on the data distribution and
the DBSCAN parameters $minPts$ and $\varepsilon$. Here we consider $minPts = 2$,  $\varepsilon = 5$, and the data distribution described above.

### Results on Aurora
These results are with $n = 669921875 = 875^3$ three-dimensional points per rank, $\varepsilon = 5$, `num_seq = 50`, and `spacing = 10`. 

| Number of Nodes | n (global) | T (seconds)| FOM |
| -----------: | -----------: | ---: | ---: |
| 1 | 8,039,062,500 | 21.527 |    1,120,322,734 |
| 2 |16,078,125,000 | 21.619 |    2,231,110,366 |
| 4 |32,156,250,000 | 21.714 |    4,442,698,259 |
| 8 |64,312,500,000 | 21.794 |    8,852,780,582 |
| 16 | 128,625,000,000 | 22.212 | 17,372,366,288 |
| 32 | 257,250,000,000 | 22.029 | 35,033,365,110 |
| 64 | 514,500,000,000 | 22.193 | 69,548,956,878 |
| 128 | 1,029,000,000,000 | 22.782 | 135,501,711,878 | 
| 256 | 2,058,000,000,000 | 23.574 |    261,898,701,960 | 
| 512 | 4,116,000,000,000 | 25.186 |    490,272,373,541 |
| 1024 | 8,232,000,000,000 | 28.842 |    856,251,300,187 |

The library versions were ArborX 1.7.99 and Kokkos 4.5.1.

### Reproducing these on the current software stack

Neither version is what the current Aurora stack produces: ArborX master is now
2.x and the prebuilt Kokkos modules are 4.7.02 / 5.1.1. Record the ArborX and
Kokkos versions alongside any new numbers — the benchmark prints both at
startup.

With **ArborX 2.1.99 / Kokkos 5.1.1**, the default `fdbscan` implementation
used for the table above **no longer fits in GPU memory** at this problem size:
it aborts with `UR_RESULT_ERROR_OUT_OF_RESOURCES` from the Level Zero backend
on most ranks. On one node it runs up to $839^3$ = 590,589,719 points per rank
and fails at $874^3$. Reaching the benchmark size on a current build requires
`--impl fdbscan-densebox`, which is also the faster of the two here:

| Implementation | n (global) | T (seconds) | FOM |
| :--- | -----------: | ---: | ---: |
| `fdbscan` (as published, ArborX 1.7.99) | 8,039,062,500 | 21.527 | 1,120,322,734 |
| `fdbscan` (ArborX 2.1.99) | 8,039,062,500 | — | out of memory |
| `fdbscan-densebox` (ArborX 2.1.99) | 8,039,062,500 | 9.641 | 2,502,749,502 |

$T$ is the mean of three consecutive runs on one node (9.875, 9.702, 9.347 s;
5.5% spread), all at the full $875^3$ points per rank, so $n$ matches the
published table exactly. A current build is about 2.2× the published 1-node
FOM — but by a different algorithm, so this is not a like-for-like
reproduction of the row above it.

## Steps to Run
This benchmark is part of the [ArborX repository](https://github.com/arborx/ArborX).

### Installation
See the ArborX installation instructions. There are recent installation
instructions specific to running this benchmark on Aurora on [this
page](install.md).

### Example command
```shell
cd /ArborX/build/benchmarks/cluster
export CPU_BIND="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
export OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads
mpiexec --no-transfer -np 12 --ppn 12 --cpu-bind=${CPU_BIND} gpu_tile_compact.sh \
   ./ArborX_Benchmark_DistributedDBSCAN.exe --eps 5 --dimension 3 --n 669921876 --num-seq 50 --spacing 10 --impl fdbscan-densebox --verbose
```
In the output, `n (global)` corrensponds to $n$ in FOM, and `total time`
corresponds to $T$. Three details that are easy to get wrong:

- `--verbose` is **required**: `total time` is only reported when it is set.
  Without it the run completes and prints no timing at all.
- `--n 669921876`, not `669921875`. The generator floors the cube root, and
  `std::pow(669921875, 1./3.)` is `874.9999999999997` in floating point, so the
  exact cube silently yields $874^3$ per rank and a smaller problem than this
  table reports.
- `--impl fdbscan-densebox` is needed on current ArborX; see above.
- `OMP_NUM_THREADS` keeps each rank from spawning one host thread per node core
  (208) onto the 8 cores `--cpu-bind` gives it.

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


