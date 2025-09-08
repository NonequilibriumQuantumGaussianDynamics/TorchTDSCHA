from ase import Atoms
import numpy as np
from ase.calculators.emt import EMT
from diff_2nd import diff_2nd
from diff_3rd import diff_3rd
from diff_4th import diff_4th

def test_d2():
    input_structure = 'final_result'
    calculator = EMT()
    what = 'dynamical_matrix'

    diff_2nd(input_structure, calculator, what)

    phi = np.load('phi.npy')
    phi_test = np.load('phi_test.npy')
    diff = np.linalg.norm(phi-phi_test)

    print('2nd derivatives error', diff)
    assert diff < 1e-8

def test_d3():
    input_structure = 'final_result'
    calculator = EMT()
    what = 'dynamical_matrix'

    diff_3rd(input_structure, calculator, what)

    chi = np.load('chi.npy')
    chi_test = np.load('chi_test.npy')
    diff = np.linalg.norm(chi-chi_test)

    print('3rd derivatives error', diff)
    assert diff < 1e-8


def test_d4():
    input_structure = 'final_result'
    calculator = EMT()
    what = 'dynamical_matrix'

    diff_4th(input_structure, calculator, what)

    psi = np.load('psi.npy')
    psi_test = np.load('psi_test.npy')
    diff = np.linalg.norm(psi-psi_test)

    print('4th derivatives error', diff)
    assert diff < 1e-8






