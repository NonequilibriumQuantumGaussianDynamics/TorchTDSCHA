from ase import Atoms
import numpy as np
from averages import *
from dynamics import *
from init import *
import cellconstructor as CC, cellconstructor.Phonons
from pathlib import Path

PATH = Path(__file__).resolve().parent
os.chdir(PATH)
print('HERE PATH', PATH)


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
    edir = [1, 0, 0]  # WARNING

    gamma = 0

    label = "test_H2"

    dyn = CC.Phonons.Phonons("final_result")
    om, eigv = dyn.DiagonalizeSupercell()

    path_diff = "."
    path = "."
    nat, nmod, phi, chi, psi, R, P, masses, A, B, C = load_from_sscha(
        "final_result", path, T, new_format=True, path_diff=path_diff
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

    sol = td_evolution(
        R, P, A, B, C, field, gamma, phi, chi, psi, Time, NS, label=label, chunks=chunks
    )

    s0 = np.load("dynamics_H2_0.npz")["arr_0"]
    s1 = np.load("test_H2_0.npz")["arr_0"]

    assert np.linalg.norm(s1 - s0) < 1e-8
