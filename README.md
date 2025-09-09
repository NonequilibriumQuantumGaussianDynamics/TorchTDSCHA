Exact TDSCHA

## Installation

To avoid libopen conflicts, use conda-force

```bash
conda create -n tdscha -c conda-forge python=3.10 numpy scipy spglib=2.2 ase mpi4py gfortran libblas lapack openmpi openmpi-mpicc setuptools=64 -y
conda activate tdscha
pip install --no-deps cellconstructor python-sscha tdscha
conda install -c conda-forge pytorch torchvision torchaudio -y
```
