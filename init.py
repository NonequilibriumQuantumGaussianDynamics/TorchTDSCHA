import numpy as np
import copy
import h5py
from model_multi import *

def read_phi(file):
    f2 = h5py.File(file, 'r')
    fc2 = f2['fc2']
    print(fc2)
    nat = np.shape(fc2)[0]

    newfc2 = np.zeros((3*nat,3*nat))
    for i in range(nat):
        for j in range(nat):
            newfc2[3*i:3*i+3, 3*j:3*j+3] = fc2[i,j,:,:]
    return newfc2

def read_psi(file):
    f4 = h5py.File(file, 'r')
    fc4 = f4['fc4']
    print(fc2)
    nat = np.shape(fc4)[0]

    newfc4 = np.zeros((3*nat,3*nat,3*nat,3*nat))
    for i in range(nat):
        for j in range(nat):
            for j in range(nat):
                for l in range(nat):
                    newfc4[3*i:3*i+3, 3*j:3*j+3, 3*k+3, 3*l+3] = fc4[i,j,k,l,:,:,:,:]
    return newfc4

def init(path, T):

    nmod = 3*nat
    phi = np.zeros((nmod, nmod))
    psi = np.zeros((nmod, nmod, nmod, nmod))

    R = np.zeros(nmod)
    P = np.zeros(nmod)
    masses = np.zeros(nmod)

    C = np.zeros((nmod, nmod))

    # Assign values
    masses[:] = 911 #m_ry, proton
    
    d = -0.1
    for i in range(nmod):
        phi[i,i] = d  #Ry/B^2
        psi[i,i,i,i] = 0.5  #Ry/B^4
    for i in range(3):
        phi[i,i+3] = -d
        phi[i+3,i] = -d

    """
    psi[0,0,0,1] = 0.1
    psi[1,0,0,0] = 0.1
    psi[0,1,0,0] = 0.1
    psi[0,0,1,0] = 0.1
    """

    # Scale by masses
    phi = np.einsum('i,j,ij->ij', 1/np.sqrt(masses), 1/np.sqrt(masses), phi)
    psi = np.einsum('i,j,k,l,ijkl->ijkl', 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), psi)
    om, eigv = get_phonons(phi)

    A, B = get_AB(om, eigv, T)

    return nat, nmod, phi, psi, R, P,  masses, A, B, C



def init_test(nat, T):

    nmod = 3*nat
    phi = np.zeros((nmod, nmod))
    psi = np.zeros((nmod, nmod, nmod, nmod))

    R = np.zeros(nmod)
    P = np.zeros(nmod)
    masses = np.zeros(nmod)

    C = np.zeros((nmod, nmod))

    # Assign values
    masses[:] = 911 #m_ry, proton
    
    d = -0.1
    for i in range(nmod):
        phi[i,i] = d  #Ry/B^2
        psi[i,i,i,i] = 0.5  #Ry/B^4
    for i in range(3):
        phi[i,i+3] = -d
        phi[i+3,i] = -d

    """
    psi[0,0,0,1] = 0.1
    psi[1,0,0,0] = 0.1
    psi[0,1,0,0] = 0.1
    psi[0,0,1,0] = 0.1
    """

    # Scale by masses
    phi = np.einsum('i,j,ij->ij', 1/np.sqrt(masses), 1/np.sqrt(masses), phi)
    psi = np.einsum('i,j,k,l,ijkl->ijkl', 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), psi)
    om, eigv = get_phonons(phi)

    A, B = get_AB(om, eigv, T)

    return nat, nmod, phi, psi, R, P,  masses, A, B, C




