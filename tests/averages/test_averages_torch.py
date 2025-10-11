from ase import Atoms
import numpy as np
import cellconstructor as CC, cellconstructor.Phonons
import time
from pathlib import Path

from torch_tdscha.dynamics import td_evolution, torch_evolution
from torch_tdscha.averages import force, kappa, av_V, torch_av_V, ext_for, torch_force, torch_kappa, torch_ext_for
from torch_tdscha.init import load_from_sscha, read_charges

#
import torch


def test_dynamics():

    T = 0

    PATH = Path(__file__).resolve().parent.parent
    dyn = CC.Phonons.Phonons(f"{PATH}/dynamics/final_result")
    om, eigv = dyn.DiagonalizeSupercell()

    path_diff = f"{PATH}/dynamics"
    path = f"{PATH}/dynamics"
    nat, nmod, phi, chi, psi, R, P, masses, A, B, C = load_from_sscha(
        f"{PATH}/dynamics/final_result", path, T, new_format=True, path_diff=path_diff
    )
    R[0] += 0.1

    start = time.perf_counter()
    f_numpy = force(R, A, phi, chi, psi)
    end = time.perf_counter()

    print("Numpy forces:", end - start, "seconds")

    start = time.perf_counter()
    k_numpy = kappa(R, A, phi, chi, psi)
    end = time.perf_counter()

    print("Numpy curvature:", end - start, "seconds")

    start = time.perf_counter()
    V_numpy = av_V(R, A, phi, chi, psi)
    end = time.perf_counter()

    print("Numpy potential energy:", end - start, "seconds")

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    phi = torch.from_numpy(phi).to(device=device, dtype=dtype)
    chi = torch.from_numpy(chi).to(device=device, dtype=dtype)
    psi = torch.from_numpy(psi).to(device=device, dtype=dtype)
    R = torch.from_numpy(R).to(device=device, dtype=dtype)
    P = torch.from_numpy(P).to(device=device, dtype=dtype)
    masses = torch.from_numpy(masses).to(device=device, dtype=dtype)
    A = torch.from_numpy(A).to(device=device, dtype=dtype)
    B = torch.from_numpy(B).to(device=device, dtype=dtype)
    C = torch.from_numpy(C).to(device=device, dtype=dtype)

    start = time.perf_counter()
    f_torch = torch_force(R, A, phi, chi, psi)
    end = time.perf_counter()

    print("Torch forces:", end - start, "seconds")

    start = time.perf_counter()
    k_torch = torch_kappa(R, A, phi, chi, psi)
    end = time.perf_counter()

    print("Torch curvature:", end - start, "seconds")

    start = time.perf_counter()
    V_torch = torch_av_V(R, A, phi, chi, psi)
    end = time.perf_counter()

    print("Torch potential energy:", end - start, "seconds")

    assert np.linalg.norm(f_torch - f_numpy) < 1e-8
    assert np.linalg.norm(k_torch - k_numpy) < 1e-8
    assert np.linalg.norm(V_torch - V_numpy) < 1e-8


test_dynamics()
