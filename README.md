GPU-Accelerated Matrix-Free Evaluation for the Discontinuous Galerkin Method
============================================

## Installation

The code is based on the generic finite element library [deal.II](https://github.com/dealii/dealii). Using MPI requires the "forest-of-octrees" library [p4est](https://github.com/cburstedde/p4est) which is responsible to efficiently partition the triangulation with respect to distributed memory.

---

### Installing p4est

First clone **deal.II**, here, e.g. cloned to the directory _dealii_:

```bash
git clone https://github.com/dealii/dealii.git dealii
```
or, alternatively, for ssh access:

```bash
git clone git@github.com:dealii/dealii.git dealii
```

Next, we download **p4est** into a previously created directory _p4est_:

```bash
mkdir p4est
cd p4est
wget http://p4est.github.io/release/p4est-2.8.tar.gz
```

We run the _p4est-setup.sh_ script provided by deal.II to compile the debug (DEBUG) as well as release (FAST) build:

```bash
bash path_to_dealii/doc/external-libs/p4est-setup.sh p4est-2.8.tar.gz `pwd`
```

where `pwd` returns the current working directory _path_to_p4est_. After p4est is built and installed we set an environment variable to _path_to_p4est_:

```bash
export P4EST_DIR=path_to_p4est
```

When deal.II searches for external dependencies it evaluates this variable as hint for p4est.

### Installing deal.II

First clone this project **CellSmootherDistortedMesh**, e.g. into the directory _csdm_, and use the _dealii-setup.sh_ script to build deal.II:

```bash
cd path_to_dealii
mkdir build
cd build
cp path_to_csdm/scripts/dealii-setup.sh .
bash dealii-setup.sh
make -j8
```

Installing the build into a separate directory is possible and explained in deal.II's INSTALL README.

### Configuring CellSmootherDistortedMesh

```bash
git clone <csdm-repo> <folder-name>
mkdir <folder-name>/build
cd <folder-name>/build
bash ../scripts/gputps-setup.sh
make poisson
```

#### Running applications

```bash
<go to build folder>
make poisson
./apps/poisson
```

For the applications in `apps`  a few methodological parameters need to be set at compile time. These parameters are available in `include/ct_parameter.h`.

There exists a helper script `scripts/ct_parameter.py` which can be run with a python3 interpreter. Using

```bash
python3 scripts/ct_parameter.py -h
```

presents an overview of possible options. For example, running

```bash
python3 scripts/ct_parameter.py -DIM 2 -DEG 3 -VNUM float
```

resets the spatial dimension to 2, finite element degree to 3 and number type for the multigrid v-cycle to float. Note, that the executables need to be rebuilt.

The Poisson problem can be executed in parallel via MPI.

```bash
mpirun -np 2 ./apps/poisson
```

---
**Warning**

MPI implementation can directly perform MPI operations on data located in GPU device memory without explicitly copying to CPU memory first. This feature is commonly referred to as **CUDA-aware MPI**. 

Configure MPI with
```bash
./configure --with-cuda ...
```

and check with

```bash
ompi_info --parsable -l 9 --all | grep mpi_built_with_cuda_support:value
```

which, in case, everything is fine, ought to give:

```bash
mca:mpi:base:param:mpi_built_with_cuda_support:value:true
```
---