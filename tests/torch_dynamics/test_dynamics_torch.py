from ase import Atoms
import numpy as np
from averages import *
from dynamics import *
from init import *
import cellconstructor as CC, cellconstructor.Phonons
import time


#
import torch


def test_dynamics():

    T = 0
    Time = 1000  # 10000
    NS = 1000
    chunks = 1

    Eamp = 1000.0  # kV/cm
    om_L = 200.00
    t0 = 200  # fs
    sig = 150.0  # fs
    pulse = "pulse"
    edir = [0, 0, 1]  # WARNING

    gamma = 0
    label = "test_H2"

    dyn = CC.Phonons.Phonons("../averages/final_result")
    om, eigv = dyn.DiagonalizeSupercell()

    path_diff = "../averages"
    path = "../averages"
    nat, nmod, phi, chi, psi, R, P, masses, A, B, C = load_from_sscha(
        "../averages/final_result", path, T, new_format=True, path_diff=path_diff
    )

    Zeff, eps = read_charges("eps_charges", masses)
    field = {
        "amp": Eamp,
        "freq": om_L,
        "edir": edir,
        "type": pulse,
        "t0": t0,
        "sig": sig,
        "Zeff": Zeff,
        "eps": eps,
    }

    """
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    phi    = torch.from_numpy(phi).to(device=device, dtype=dtype)
    chi    = torch.from_numpy(chi).to(device=device, dtype=dtype)
    psi    = torch.from_numpy(psi).to(device=device, dtype=dtype)
    R      = torch.from_numpy(R).to(device=device, dtype=dtype)
    P      = torch.from_numpy(P).to(device=device, dtype=dtype)
    masses = torch.from_numpy(masses).to(device=device, dtype=dtype)
    A      = torch.from_numpy(A).to(device=device, dtype=dtype)
    B      = torch.from_numpy(B).to(device=device, dtype=dtype)
    C      = torch.from_numpy(C).to(device=device, dtype=dtype)
    """

    sol = torch_evolution(
        R, P, A, B, C, field, gamma, phi, chi, psi, Time, NS, label=label, chunks=chunks
    )

    sys.exit()
    sol = td_evolution(
        R, P, A, B, C, field, gamma, phi, chi, psi, Time, NS, label=label, chunks=chunks
    )
    sol = torch_evolution(
        R, P, A, B, C, field, gamma, phi, chi, psi, Time, NS, label=label, chunks=chunks
    )


test_dynamics()
