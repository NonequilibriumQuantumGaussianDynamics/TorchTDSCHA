# 📦 TorchTDSCHA

[![CI](https://github.com/NonequilibriumQuantumGaussianDynamics/exact_tdscha/actions/workflows/python-ci.yml/badge.svg)](https://github.com/NonequilibriumQuantumGaussianDynamics/exact_tdscha/actions/workflows/python-ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue.svg)](https://www.python.org/downloads/)
[![Backend](https://img.shields.io/badge/Backend-NumPy%20|%20PyTorch-orange.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A Python package for simulating quantum nuclear dynamics of solids under ultrafast excitation.*

---

## ✨ Overview

**Exact_TDSCHA** implements the **time-dependent extension** of the stochastic self-consistent harmonic approximation (SSCHA), allowing the **real-time propagation of quantum nuclei** in anharmonic potentials.

The method describes the nuclear quantum state as a **time-evolving Gaussian wavepacket**, whose centroid (`R, P`) and covariance matrices (`A, B, C`) obey deterministic coupled equations of motion:

---

## ⚙️ Installation

This package is written in **Python < 3.11** and interfaces with **ASE** and **CellConstructor**
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
git clone https://github.com/NonequilibriumQuantumGaussianDynamics/exact_tdscha.git
cd exact_tdscha
pip install -e .



