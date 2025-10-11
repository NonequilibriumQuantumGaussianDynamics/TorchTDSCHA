from ase import Atoms
import numpy as np
from ase.calculators.emt import EMT
from torch_tdscha.diff_2nd import diff_2nd
from torch_tdscha.diff_3rd import diff_3rd
from torch_tdscha.diff_4th import diff_4th
from pathlib import Path

PATH = Path(__file__).resolve().parent


def test_d2():
    input_structure = PATH / "final_result"
    calculator = EMT()
    what = "dynamical_matrix"

    diff_2nd(input_structure, calculator, what)

    phi = np.load(PATH / "phi.npy")
    phi_test = np.load(PATH / "phi_test.npy")
    diff = np.linalg.norm(phi - phi_test)

    print("2nd derivatives error", diff)
    assert diff < 1e-8


def test_d3():
    input_structure = PATH / "final_result"
    calculator = EMT()
    what = "dynamical_matrix"

    diff_3rd(input_structure, calculator, what)

    chi = np.load(PATH / "chi.npy")
    chi_test = np.load(PATH / "chi_test.npy")
    diff = np.linalg.norm(chi - chi_test)

    print("3rd derivatives error", diff)
    assert diff < 1e-8


def test_d4():
    input_structure = PATH / "final_result"
    calculator = EMT()
    what = "dynamical_matrix"

    diff_4th(input_structure, calculator, what)

    psi = np.load(PATH / "psi.npy")
    psi_test = np.load(PATH / "psi_test.npy")
    diff = np.linalg.norm(psi - psi_test)

    print("4th derivatives error", diff)
    assert diff < 1e-8
