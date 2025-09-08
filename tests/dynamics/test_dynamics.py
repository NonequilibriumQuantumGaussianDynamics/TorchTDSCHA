from ase import Atoms
import numpy as np
from model_multi import *
from init import *
import cellconstructor as CC, cellconstructor.Phonons

def test_dynamics():

    T=0
    Time = 6000 # 10000
    NS = 1000
    chunks = 1

    Eamp = 2000.0 #kV/cm
    om_L = 16.00
    t0 = 500 #fs
    sig = 150.0 #fs
    pulse = 'pulse'
    edir = [1,0, 0]  #WARNING

    gamma = 0

    label = 'dynamics_H2'

    dyn = CC.Phonons.Phonons('final_result')
    om, eigv = dyn.DiagonalizeSupercell()

    path_corrected = '.'
    path = '.'
    nat, nmod, phi, chi, psi, R, P,  masses, A, B, C = load_from_sscha('final_result', path,T, read_corrected = True, path_corrected = path_corrected)

    Zeff, eps = read_charges('eps_charges', masses)
    field = {'amp':Eamp, 'freq':om_L, 'edir': edir, 'type':pulse, 't0':t0, 'sig': sig, 'Zeff':Zeff, 'eps':eps}

    sol    = td_evolution(R, P, A, B, C, field, gamma, phi, chi, psi, Time, NS, label=label, chunks=chunks)

test_dynamics()







