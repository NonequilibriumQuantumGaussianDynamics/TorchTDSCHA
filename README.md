<p align="center">
  <img src="https://img.shields.io/badge/Language-Python%203.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Build-GitHub%20Actions-success?style=for-the-badge&logo=githubactions"/>
  <img src="https://img.shields.io/badge/Backend-NumPy%20%7C%20PyTorch-orange?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<h1 align="center">TDSCHA</h1>

---

## 🧠 Theory

The **Time-Dependent Self-Consistent Harmonic Approximation (TDSCHA)** is a non-perturbative extension of the stochastic self-consistent harmonic approximation (SSCHA) to **real-time dynamics** under **fast drive**.

It describes the **time evolution of quantum nuclei** beyond the harmonic limit by approximating the nuclear density as a **Gaussian wavepacket** evolving under an effective anharmonic potential.  
This allows one to capture:

- Finite-temperature quantum fluctuations of nuclei  
- Anharmonic phonon renormalization  
- Real-time dynamics of Gaussian nuclear states under external perturbations  
- Ultrafast responses to optical or THz fields  

The method propagates both **centroid coordinates and covariance matrices** (`R, P, A, B, C`) according to deterministic equations derived from Ehrenfest-like dynamics within the Gaussian ansatz.  

Mathematically, the TDSCHA evolves the Gaussian state parameters as:

\[
\dot{R} = P, \quad
\dot{P} = F(R, A) - \gamma P + F_\text{ext}(t)
\]

\[
\dot{A} = C + C^T, \quad
\dot{B} = -[\kappa(R, A) C + (\kappa(R, A) C)^T], \quad
\dot{C} = B - A\kappa(R, A)
\]

where the **force** \( F(R, A) \) and **curvature tensor** \( \kappa(R, A) \) include harmonic, cubic, and quartic force constants (φ, χ, ψ).  
Coupling to an **external electric field** is handled through the **Born effective charges** and dielectric tensor.

---

## ⚙️ Installation

A lightweight Python environment (Python < 3.11) can be created using **conda**:

```bash
# Create and activate the environment
conda create -n sscha -c conda-forge python=3.10 gfortran libblas lapack \
  openmpi openmpi-mpicc pip numpy scipy spglib pkgconfig -y
conda activate sscha

# Install required Python packages
pip install ase julia mpi4py pytest
pip install cellconstructor
pip install torch torchdiffeq

# Clone and install this repository
git clone https://github.com/NonequilibriumQuantumGaussianDynamics/exact_tdscha.git
cd exact_tdscha
pip install -e .
