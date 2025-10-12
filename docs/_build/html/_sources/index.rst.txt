.. TorchTDSCHA documentation master file

TorchTDSCHA documentation
=========================

TorchTDSCHA implements the time-dependent extension of the stochastic
self-consistent harmonic approximation (SSCHA), enabling the real-time propagation
of quantum nuclei in anharmonic potentials. It solves the TDSCHA equations
through exact Gaussian integration of the quantum forces, accelerated with
PyTorch for massively parallel tensor algebra on GPUs.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   source/init
   source/averages
   source/dynamics
   source/phonons
   source/diff_2nd
   source/diff_3rd
   source/diff_4th
