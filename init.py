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

def read_chi(path):
    f3 = h5py.File(path+'/fc3.hdf5', 'r')
    fc3 = f3['fc3']
    nat = np.shape(fc3)[0]

    """
    newfc2 = np.zeros((3*nat,3*nat))
    for i in range(nat):
        for j in range(nat):
            newfc2[3*i:3*i+3, 3*j:3*j+3] = fc2[i,j,:,:]
    """

    newfc3 = np.reshape(np.transpose(fc3, [0,3,1,4,2,5]),(3*nat, 3*nat, 3*nat))
    return newfc3

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


def load_from_sscha(dyn_file, path, T, read_corrected = False, path_corrected = ''):

    Ry_to_eV = 13.60570397
    uma_to_Ry = 911.444175
    A_to_B = 1.889725988

    import cellconstructor as CC, cellconstructor.Phonons

    print("WARNING: the fine dyn has tho have the same atomic ordering as the SPOSCAR")

    dyn = CC.Phonons.Phonons(dyn_file)
    
    masses = dyn.structure.get_masses_array()
    masses = np.repeat(masses,3)

    positions = dyn.structure.coords
    atoms = read_vasp(path + "/SPOSCAR")
    

    nat = len(positions)
    nmod = 3*nat
    R = np.reshape(positions-atoms.positions, nmod) * A_to_B
    R = np.einsum('i,i->i', R, np.sqrt(masses))

    if not read_corrected:
        phi = read_phi(path)
        phi = phi/Ry_to_eV/A_to_B**2
        phi = np.einsum('i,j,ij->ij', 1/np.sqrt(masses), 1/np.sqrt(masses), phi)

        chi = read_chi(path)
        chi = chi/Ry_to_eV/A_to_B**3
        chi = np.einsum('i,j,k,ijk->ijk', 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), chi)

        psi = read_psi(path)
        psi = psi/Ry_to_eV/A_to_B**4
        psi = np.einsum('i,j,k,l,ijkl->ijkl', 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), psi)
    else:
        phi = np.load(path_corrected + '/phi.npy')
        #phi = phi/Ry_to_eV/A_to_B**2
        phi = np.einsum('i,j,ij->ij', 1/np.sqrt(masses), 1/np.sqrt(masses), phi)

        chi = np.load(path_corrected + '/chi.npy')
        #chi = chi/Ry_to_eV/A_to_B**3
        chi = np.einsum('i,j,k,ijk->ijk', 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), chi)

        psi = np.load(path_corrected + '/psi.npy')
        #psi = psi/Ry_to_eV/A_to_B**4
        psi = np.einsum('i,j,k,l,ijkl->ijkl', 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), 1/np.sqrt(masses), psi)

    P = np.zeros(nmod)
    C = np.zeros((nmod, nmod))

    om, eigv =  dyn.DiagonalizeSupercell()

    A, B = get_AB(om, eigv, T)

    return nat, nmod, phi, chi, psi, R, P,  masses, A, B, C

def reduce_model(modes, nat, nmod, phi, chi, psi, R, P,  masses, A, B, C, eigv):
    
    phi_mu = np.einsum('ij,im,jn->mn', phi, eigv, eigv)
    chi_mu = np.einsum('ijk,im,jn,kp->mnp', chi, eigv, eigv, eigv)
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

    nmod = len(modes)


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
    Zeff = np.einsum('ij,i->ij', NewZeff, 1/np.sqrt(masses))
    return Zeff, eps


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



            
