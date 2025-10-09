import numpy as np
import copy
import torch
#import h5py
#from model_multi import *
from phonons import *

from ase.io.vasp import read_vasp
import cellconstructor as CC, cellconstructor.Phonons


def load_from_sscha(dyn_file, path, T, new_format = False, path_diff = ''):

    Ry_to_eV = 13.60570397
    uma_to_Ry = 911.444175
    A_to_B = 1.889725988

    dyn = CC.Phonons.Phonons(dyn_file)
    
    masses = dyn.structure.get_masses_array()
    masses = np.repeat(masses,3)

    positions = dyn.structure.coords
    atoms = read_vasp(path + "/SPOSCAR")
    

    nat = len(positions)
    nmod = 3*nat
    R = np.reshape(positions-atoms.positions, nmod) * A_to_B
    sqrtm = np.sqrt(masses)
    sqrtm_inv = 1/sqrtm
    R *= sqrtm

    if not new_format:
        phi = read_phi(path)
        phi = phi/Ry_to_eV/A_to_B**2
        phi *= sqrtm_inv[:, None]
        phi *= sqrtm_inv[None, :]

        chi = read_chi(path)
        chi = chi/Ry_to_eV/A_to_B**3
        chi *= sqrtm_inv[:,  None, None]  
        chi *= sqrtm_inv[ None,:,  None]  
        chi *= sqrtm_inv[ None,None,:]  

        psi = read_psi(path)
        psi = psi/Ry_to_eV/A_to_B**4
        psi *= sqrtm_inv[:,  None, None, None]
        psi *= sqrtm_inv[ None,:,  None, None]
        psi *= sqrtm_inv[ None,None,:,  None]
        psi *= sqrtm_inv[ None,None,None,:]

    else:
        phi = np.load(path_diff + '/phi.npy')
        phi *= sqrtm_inv[:, None]
        phi *= sqrtm_inv[None, :]

        chi = np.load(path_diff + '/chi.npy')
        chi *= sqrtm_inv[:,  None, None]
        chi *= sqrtm_inv[ None,:,  None]
        chi *= sqrtm_inv[ None,None,:]

        psi = np.load(path_diff + '/psi.npy')
        psi *= sqrtm_inv[:,  None, None, None]
        psi *= sqrtm_inv[ None,:,  None, None]
        psi *= sqrtm_inv[ None,None,:,  None]
        psi *= sqrtm_inv[ None,None,None,:]

    P = np.zeros(nmod)
    C = np.zeros((nmod, nmod))

    om, eigv =  dyn.DiagonalizeSupercell()

    A, B = get_AB(om, eigv, T)

    return nat, nmod, phi, chi, psi, R, P,  masses, A, B, C

def read_phi(path):
    f2 = h5py.File(path+'/fc2.hdf5', 'r')
    fc2 = f2['fc2']
    nat = np.shape(fc2)[0]

    newfc2 = np.reshape(np.transpose(fc2, [0,2,1,3]),(3*nat, 3*nat))
    return newfc2

def read_chi(path):
    f3 = h5py.File(path+'/fc3.hdf5', 'r')
    fc3 = f3['fc3']
    nat = np.shape(fc3)[0]

    newfc3 = np.reshape(np.transpose(fc3, [0,3,1,4,2,5]),(3*nat, 3*nat, 3*nat))
    return newfc3

def read_psi(path):
    f4 = h5py.File(path+'/fc4.hdf5', 'r')
    fc4 = f4['fc4']
    nat = np.shape(fc4)[0]
    
    newfc4 = np.transpose(fc4, [0,4,1,5,2,6,3,7])
    newfc4 = np.reshape(newfc4, (3*nat,3*nat,3*nat,3*nat))
    return newfc4


def reduce_model(modes, nmod, phi, chi, psi, R, P, A, B, C, eigv, Zeff):

    print("Reducing model to ", modes)
    modes = [m-1 for m in modes]
    
    phi_mu = np.einsum('ij,im,jn->mn', phi, eigv, eigv)
    chi_mu = np.einsum('ijk,im,jn,kp->mnp', chi, eigv, eigv, eigv, optimize = "optimal")
    psi_mu = np.einsum('ijkl,im,jn,kp,lq->mnpq', psi, eigv, eigv, eigv, eigv, optimize = "optimal")

    phi_mu = phi_mu[np.ix_(modes, modes)]
    chi_mu = chi_mu[np.ix_(modes, modes, modes)]
    psi_mu = psi_mu[np.ix_(modes, modes, modes, modes)]

    R_mu = np.einsum('i,is->s', R, eigv)
    P_mu = np.einsum('i,is->s', P, eigv)
    A_mu = np.einsum('ij,is,jt->st', A, eigv, eigv)
    B_mu = np.einsum('ij,is,jt->st', B, eigv, eigv)
    C_mu = np.einsum('ij,is,jt->st', C, eigv, eigv)

    R_mu = R_mu[np.ix_(modes)]
    P_mu = P_mu[np.ix_(modes)]
    A_mu = A_mu[np.ix_(modes, modes)]
    B_mu = B_mu[np.ix_(modes, modes)]
    C_mu = C_mu[np.ix_(modes, modes)]

    Zeff_mu = np.einsum('is, ij->sj', eigv, Zeff)
    Zeff_mu = Zeff_mu[modes,:]

    nmod = len(modes)

    return nmod, phi_mu, chi_mu, psi_mu, R_mu, P_mu, A_mu, B_mu, C_mu, Zeff_mu 

def isolate_couplings(modes, phi, chi, psi, eigv, exclude_diag = []):

    print("Reducing to ", modes)
    modes = [m-1 for m in modes]
    
    phi_mu = np.einsum('ij,im,jn->mn', phi, eigv, eigv)
    chi_mu = np.einsum('ijk,im,jn,kp->mnp', chi, eigv, eigv, eigv, optimize = "optimal")
    psi_mu = np.einsum('ijkl,im,jn,kp,lq->mnpq', psi, eigv, eigv, eigv, eigv, optimize = "optimal")

    shape = phi_mu.shape
    mask = np.zeros(shape, dtype=bool)
    idx = np.ix_(modes, modes)
    mask[idx] = True
    phi_mu[~mask] = 0

    shape = chi_mu.shape
    mask = np.zeros(shape, dtype=bool)
    idx = np.ix_(modes, modes, modes)
    mask[idx] = True
    chi_mu[~mask] = 0

    shape = psi_mu.shape
    mask = np.zeros(shape, dtype=bool)
    idx = np.ix_(modes, modes, modes, modes)
    mask[idx] = True
    psi_mu[~mask] = 0

    for mod in exclude_diag:
        s = mod-1
        print(phi_mu[s,s])
        phi_mu[s,s] = 0
        print(chi_mu[s,s,s])
        chi_mu[s,s,s] = 0
        print(psi_mu[s,s,s,s])
        psi_mu[s,s,s,s] = 0

    phi = np.einsum('mn,im,jn->ij', phi_mu, eigv, eigv, optimize = "optimal")
    chi = np.einsum('mnp,im,jn,kp->ijk', chi_mu, eigv, eigv, eigv, optimize = "optimal")
    psi = np.einsum('mnpq,im,jn,kp,lq->ijkl', psi_mu, eigv, eigv, eigv, eigv, optimize = "optimal")

    return phi, chi, psi


def continue_evolution(fil):
    sol = np.load(fil)['arr_0']
    tfin = sol[-1,0]*(4.8377687*1e-2)

    sol =  sol[:,1:]
 
    N, n1 = np.shape(sol)
    nmod = int((-2+np.sqrt(4+12*n1))/6)

    R = sol[-1,:nmod]
    P = sol[-1,nmod:2*nmod]
    A = sol[-1,2*nmod:2*nmod+nmod**2]
    A = np.reshape(A,(nmod,nmod))
    B = sol[-1,2*nmod+nmod**2:2*nmod+2*nmod**2]
    B = np.reshape(B,(nmod,nmod))
    C = sol[-1,2*nmod+2*nmod**2:2*nmod+3*nmod**2]
    C = np.reshape(C,(nmod,nmod))

    newlabel = fil[:-5]+'cont'
    return R, P, A, B, C, newlabel, tfin

def merge_evolutions(fil,fil1,fil2=''):
    sol = np.load(fil)['arr_0']
    sol1 = np.load(fil1)['arr_0']
    N1, x = np.shape(sol)
    N2, x = np.shape(sol1)
    merged = np.zeros((N1+N2-1,x))

    if fil2!='':
        sol2 = np.load(fil2)['arr_0']
        N3, x = np.shape(sol2)
        merged = np.zeros((N1+N2+N3-2,x))

    merged[:N1,:] = sol

    if fil2!='':
        merged[N1:N1+N2-1,:] = sol1[1:,:]
        merged[N1+N2-1:,:] = sol2[1:,:]
    else:
        merged[N1:,:] = sol1[1:,:]

    return merged


def read_charges(path, masses):

    ff = open(path)
    lines = ff.readlines()
    ff.close()

    for i in range(len(lines)):
        if 'number of atoms/cell' in lines[i]:
            nat = int(lines[i].split()[-1])
        if 'Effective charges (d Force / dE) in cartesian axis with asr applied' in lines[i]:
            Zeff = []
            for j in range(nat):
                charge = []
                for l in range(3):
                    k = i +4*j + 3 + l
                    line = lines[k].split()
                    charge.append( [float(line[2]), float(line[3]), float(line[4])])
                Zeff.append(charge)

            Zeff = np.array(Zeff)
            for i in range(3):
                print("check ", i+1, np.sum(Zeff[:,i]))
        if 'Dielectric constant in cartesian axis' in lines[i]:
            eps = float(lines[i+2].split()[1])

    NewZeff = np.zeros((3*nat, 3))
    for i in range(nat):
        NewZeff[3*i:3*i+3] = Zeff[i,:,:]
    NewZeff = np.einsum('ij,i->ij', NewZeff, 1/np.sqrt(masses))
    return NewZeff, eps


def read_solution(label, chunks):
    for i in range(chunks):
        fil = np.load(label + '_%d.npy' %i)
        sh = np.shape(fil)
        N0 = sh[0]
        if i==0:
            y = np.zeros((N0*chunks, sh[1]))
        y[N0*i:N0*(i+1),:] = fil
    t = y[:,0]
    sol = y[:,1:]
    return t, sol



def torch_init(R, P, A, B, C, masses, phi, chi, psi, dtype = torch.float64, grad_enabled = False):
    
    torch.set_grad_enabled(grad_enabled)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phi    = torch.from_numpy(phi).to(device=device, dtype=dtype)
    chi    = torch.from_numpy(chi).to(device=device, dtype=dtype)
    psi    = torch.from_numpy(psi).to(device=device, dtype=dtype)
    R      = torch.from_numpy(R).to(device=device, dtype=dtype)
    P      = torch.from_numpy(P).to(device=device, dtype=dtype)
    masses = torch.from_numpy(masses).to(device=device, dtype=dtype)
    A      = torch.from_numpy(A).to(device=device, dtype=dtype)
    B      = torch.from_numpy(B).to(device=device, dtype=dtype)
    C      = torch.from_numpy(C).to(device=device, dtype=dtype)

    return R, P, A, B, C, masses, phi, chi, psi




