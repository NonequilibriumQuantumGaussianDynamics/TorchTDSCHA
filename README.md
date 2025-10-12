# 📦 TorchTDSCHA

[![CI](https://github.com/NonequilibriumQuantumGaussianDynamics/TorchTDSCHA/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/NonequilibriumQuantumGaussianDynamics/TorchTDSCHA/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue.svg)](https://www.python.org/downloads/)
[![Backend](https://img.shields.io/badge/Backend-NumPy%20|%20PyTorch-orange.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A Python package for simulating quantum nuclear dynamics of solids under ultrafast excitation.*

---

## ✨ Overview

**TorchTDSCHA** implements the **time-dependent extension** of the self-consistent harmonic approximation (TDSCHA), allowing the **real-time propagation of quantum nuclei** in anharmonic potentials.

It solves the TDSCHA integro-differential equations through exact Gaussian integration of the quantum forces, powered by PyTorch for massive GPU acceleration of the underlying tensor algebra.

---

## ⚙️ Installation

This package is written in **Python** and interfaces with **ASE** and **CellConstructor**
for atomistic structure management and force-constant data.  
It includes a full **PyTorch-accelerated implementation** of the TDSCHA equations of motion,
designed to speed up the computation of **quantum forces** and
**potential energies**.

A minimal conda environment can be created as follows:


```bash
# Create and activate the environment
conda create -n sscha -c conda-forge python=3.10 gfortran libblas lapack \
  openmpi openmpi-mpicc pip numpy scipy spglib pkgconfig -y
conda activate sscha

# Install Python dependencies
pip install ase mpi4py cellconstructor torch torchdiffeq

# Clone and install this repository
git clone https://github.com/NonequilibriumQuantumGaussianDynamics/TorchTDSCHA.git
cd TorchTDSCHA
pip install -e .
```

---

## 📖 References

Details about the numerical methods can be found at
> F. Libbi *et al.*, *Quantum cooling below absolute zero", arXiv:2505.22791, https://arxiv.org/abs/2505.22791 
> F. Libbi *et al.*, *Atomistic simulations of out-of-equilibrium quantum nuclear dynamics*, npj Computational Materials  11, 144 (2025) https://doi.org/10.1038/s41524-025-01588-4



