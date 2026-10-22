#!/bin/bash
# Module environment for building and running the ALCF-4 DBSCAN benchmark on
# Aurora. Source this before configuring, and again from the jobscript.
#
# Use `module restore`, NOT `module purge`: boost is provided under the
# mpich + intel-oneapi-compilers hierarchy, and purging unloads mpich, after
# which `module load boost` fails with "cannot be loaded as requested".

module restore >/dev/null 2>&1
module load oneapi/release/2026.1.0
module load kokkos/5.1.1-sycl
module load boost
module load cmake

# Set by the modules above; consumed by the ArborX cmake configure.
#   KOKKOS_ROOT, BOOST_ROOT
