import numpy as np
import copy
import h5py
from model_multi import *
from ase.io.vasp import read_vasp

def read_phi(path):
    f2 = h5py.File(path+'/fc2.hdf5', 'r')
    fc2 = f2['fc2']
    print(fc2)
    nat = np.shape(fc2)[0]

    """
    newfc2 = np.zeros((3*nat,3*nat))
    for i in range(nat):
        for j in range(nat):
            newfc2[3*i:3*i+3, 3*j:3*j+3] = fc2[i,j,:,:]
    """

    newfc2 = np.reshape(np.transpose(fc2, [0,2,1,3]),(3*nat, 3*nat))
    return newfc2

def read_psi(path):
    f4 = h5py.File(path+'/fc4.hdf5', 'r')
    fc4 = f4['fc4']
    nat = np.shape(fc4)[0]
    
    """
    newfc4 = np.zeros((3*nat,3*nat,3*nat,3*nat))
    for i in range(nat):
        for j in range(nat):
            for k in range(nat):
                for l in range(nat):
                    newfc4[3*i:3*i+3, 3*j:3*j+3, 3*k:3*k+3, 3*l:3*l+3] = fc4[i,j,k,l,:,:,:,:]
    """

    newfc4 = np.transpose(fc4, [0,4,1,5,2,6,3,7])
    print("reshaping")
    newfc4 = np.reshape(newfc4, (3*nat,3*nat,3*nat,3*nat))
    print(np.shape(newfc4))
    return newfc4

def init_test(path, T):

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



def init(nat, T):

    nmod = 3*nat
    phi = np.zeros((nmod, nmod))
    psi = np.zeros((nmod, nmod, nmod, nmod))

    R = np.zeros(nmod)
    P = np.zeros(nmod)
    masses = np.zeros(nmod)

    C = np.zeros((nmod, nmod))

    # Assign values
    masses[:] = 911 #m_ry, proton
    
    d = 0.1
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
    om, eigv = get_phonons_r(phi)

    A, B = get_AB(om, eigv, T)

    return nat, nmod, phi, psi, R, P,  masses, A, B, C

def init_read(path,T):
    
    Ry_to_eV = 13.60570397
    uma_to_Ry = 911.444175
    A_to_B = 1.889725988

    atoms = read_vasp(path + "/SPOSCAR")
    
    masses = atoms.get_masses()
    masses = np.repeat(masses,3) * uma_to_Ry
    
    nat = len(atoms.positions)
    nmod = 3*nat

    R = np.reshape(atoms.positions, nmod) * A_to_B
    R = R*np.sqrt(masses)
  
    phi = read_phi(path)
    phi = phi/Ry_to_eV/A_to_B**2
    phi = np.einsum('i,j,ij->ij', 1/np.sqrt(masses), 1/np.sqrt(masses), phi)
    print("Check phonons")
    print_phonons_mat(phi)

    psi = read_psi(path)
    psi = psi/Ry_to_eV/A_to_B**4
    psi = np.einsum('i,j,k,l,ijkl->ijkl', 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), psi)
        
    P = np.zeros(nmod)
    C = np.zeros((nmod, nmod))
  
    om, eigv = get_phonons_r(phi)
    A, B = get_AB(om, eigv, T)

    return nat, nmod, phi, psi, R, P,  masses, A, B, C

