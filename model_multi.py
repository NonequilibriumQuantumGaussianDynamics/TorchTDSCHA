import numpy as np
import sys
import matplotlib
import scipy
from scipy.optimize import minimize
from scipy.integrate import odeint, solve_ivp
import copy

#matplotlib.use('agg')
from matplotlib import pyplot as pl
import os
from matplotlib.animation import FuncAnimation
matplotlib.rc('axes',lw=1.5)
matplotlib.rcParams['axes.labelsize'] = 17
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13




RyToeV = 13.605703976

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
    
    for i in range(nmod):
        phi[i,i] = -0.1  #Ry/B^2
        psi[i,i,i,i] = 0.5  #Ry/B^4
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

def get_phonons_THz(phi):
    eig, eigv = np.linalg.eig(phi)
    eig = np.sqrt(eig)*13.6*241.8
    return eig, eigv

def get_phonons(phi):
    eig, eigv = np.linalg.eig(phi)
    eig = np.sqrt(eig)
    return eig, eigv

def get_phonons_r(phi):
    eig, eigv = np.linalg.eig(phi)
    eig = np.abs(eig)
    eig = np.sqrt(eig)
    return eig, eigv

def print_phonons(om):
    print("phonons")
    for i in range(len(om)):
        print("Mode %d" %(i+1), om[i]*241.8*13.6)
 
def get_AB(om, eigv, T):
    K_to_Ry=6.336857346553283e-06
    arg = om/(T*K_to_Ry)/2.0

    lambd = 1/np.tanh(arg)/(2*om)
    A= np.einsum('s,is,js->ij', lambd, eigv, eigv)

    lambd = om/np.tanh(arg)/(2)
    B= np.einsum('s,is,js->ij', lambd, eigv, eigv)
    return A,B
    
def force(R,A,phi,psi):
    f1 = np.einsum('ij,j->i', phi, R)  
    f3 = 1/6*np.einsum('ijkl,j,k,l->i', psi, R, R, R)
    fq = 1/2*np.einsum('ijkl,j,kl->i', psi, R, A)

    return -f1-f3-fq

def f_classic(R,phi,psi):
    f1 = np.einsum('ij,j->i', phi, R)  
    f3 = 1/6*np.einsum('ijkl,j,k,l->i', psi, R, R, R)

    return -f1-f3

def ext_for(t, Eamp, om_L, nmod):
    Eeff = Eamp * 2.7502067*1e-7  * 2 /(1+np.sqrt(6))
    freq = om_L/(2.0670687*1e4)

    Zeff = np.ones(nmod)
    return Zeff*Eeff*np.sin(2*np.pi*freq*t)

def kappa(R, A, phi, psi):
    k1 = 1/2*np.einsum('ijkl, k,l->ij', psi, R, R)
    k2 = 1/2*np.einsum('ijkl, kl->ij', psi, A)
    
    return phi + k1 +   k2

def d2V(R, phi, psi):
    k1 = 1/2*np.einsum('ijkl, k,l->ij', psi, R, R)
    return phi+k1

def av_V(R, A, phi, psi):
  
    V0 = 1/2*np.einsum('i,j,ij', R, R, phi)
    V1 = 1/2*np.einsum('ij,ij', A, phi)
    V2 = 1/24*np.einsum('ijkl,i,j,k,l', psi, R ,R ,R ,R)
    V3 = 1/4*np.einsum('ijkl,i,j,kl', psi, R ,R ,A)
    V4 = 1/8*np.einsum('ijkl,ij,kl', psi, A ,A)
    
    lamb, vect = np.linalg.eig(A)
    Q = np.einsum('s, is,js,ks,ls -> ijkl', lamb**2, vect, vect, vect, vect)
    V5 = 1/8*np.einsum('ijkl,ijkl', psi, Q)

    return V0 + V1 + V2 + V3 + V4 

def Lambda(om,eigv,T):
    
    nmod = len(om)
    F0 = np.zeros((nmod,nmod))

    K_to_Ry=6.336857346553283e-06
    def nb(om,T):
        if T>1e-3:
            beta = 1/(T*K_to_Ry)
            return 1.0/(np.exp(om*beta)-1)
        else:
            return 0*om
    
    for i in range(nmod):
        n_mu = nb(om[i],T)
        for j in range(i,nmod):
            n_nu = nb(om[j],T)
            dn_domega_nu = -1/(T*K_to_Ry)*n_nu*(1+n_nu)
            if j==i:
                F0[i,i] = 2*((2*n_nu+1)/(2*omega_nu) - dn_domega_nu)
                F0[i,i] /= om[i]**2
            else:
                F0[i,j] = 2*( (n_mu + n_nu + 1)/(om[i] + om[j]) - (n_mu-n_nu)/(om[i]-om[j]) ) / (om[i]*om[j])
                F0[j,i] = F0[i,j]

    lamb = np.einsum('ij,aj,bi,cj,di->abcd', F0, eigv, eigv, eigv, eigv)  # masses
    return -8*lamb
 

def get_y0(R,P,A,B,C):
    
    nmod = len(R)
    y0 = np.zeros(2*nmod+3*nmod**2)
    y0[:nmod] = R
    y0[nmod:2*nmod] = P
    
    A_lin = np.reshape(A, nmod**2)
    B_lin = np.reshape(B, nmod**2)
    C_lin = np.reshape(C, nmod**2)
    
    y0[2*nmod:2*nmod+nmod**2] = A_lin
    y0[2*nmod+nmod**2:2*nmod+2*nmod**2] = B_lin
    y0[2*nmod+2*nmod**2:2*nmod+3*nmod**2] = C_lin
    
    return(y0)

def func(y,t, phi, psi, Eamp, om_L, gamma):

    nmod = int((-2 + np.sqrt(4+12*len(y)))/6)

    R = y[:nmod]    
    P = y[nmod:2*nmod] 

    A_lin = y[2*nmod:2*nmod+nmod**2] 
    B_lin = y[2*nmod+nmod**2:2*nmod+2*nmod**2] 
    C_lin = y[2*nmod+2*nmod**2:2*nmod+3*nmod**2] 

    A = np.reshape(A_lin, (nmod, nmod))
    B = np.reshape(B_lin, (nmod, nmod))
    C = np.reshape(C_lin, (nmod, nmod))

    f = force(R,A,phi,psi)
    #f = f_classic(R,phi,psi)
    curv = kappa(R,A,phi,psi)

    ydot = np.zeros(len(y))

    ydot[:nmod] = P
    ydot[nmod:2*nmod] = f + ext_for(t, Eamp, om_L, nmod)

    Adot = C + np.transpose(C)
    Bdot = -np.dot(curv, C)
    Bdot = Bdot + np.transpose(Bdot)
    Cdot = B - np.dot(A,curv)
    
    ydot[2*nmod:2*nmod+nmod**2] = np.reshape(Adot, nmod**2)
    ydot[2*nmod+nmod**2:2*nmod+2*nmod**2] = np.reshape(Bdot, nmod**2)
    ydot[2*nmod+2*nmod**2:2*nmod+3*nmod**2] = np.reshape(Cdot, nmod**2)

    return ydot

def td_evolution(R, P, A, B, C,  Eamp, om_L, gamma, phi, psi, Time, NS):
    # om_L in THz, Time in fs

    Time = Time/(4.8377687*1e-2)

    t = np.linspace(0,Time,NS)
    y0 = get_y0(R,P,A,B,C)
    sol = odeint(func, y0, t, args=(phi, psi, Eamp, om_L, gamma))
    return t, sol

def get_x0(R,Phi):

    nmod = len(R)
    Phi_lin = np.zeros(int((nmod**2+nmod)/2))
    sqPhi = scipy.linalg.sqrtm(Phi)

    count=0
    for i in range(nmod):
        for j in range(i,nmod):
            Phi_lin[count] = sqPhi[i,j]
            count += 1

    x0 = np.zeros(int((nmod**2+3*nmod)/2))
    x0[:nmod] = R
    x0[nmod:] = Phi_lin

    return x0

def get_R_Phi(x):

    nmod = int((-3+np.sqrt(9+8*len(x)))/2)
    R = x[:nmod]
    sqPhi_lin = x[nmod:]
    sqPhi = np.zeros((nmod,nmod))
    
    count=0
    for i in range(nmod):
        for j in range(i,nmod):
            sqPhi[i,j] = sqPhi_lin[count]
            if i!=j:
                sqPhi[j,i] = sqPhi[i,j]
            count += 1
    return R,np.dot(sqPhi,sqPhi)

def get_R_sqPhi(x):

    nmod = int((-3+np.sqrt(9+8*len(x)))/2)
    R = x[:nmod]
    sqPhi_lin = x[nmod:]
    sqPhi = np.zeros((nmod,nmod))
    
    count=0
    for i in range(nmod):
        for j in range(i,nmod):
            sqPhi[i,j] = sqPhi_lin[count]
            if i!=j:
                sqPhi[j,i] = sqPhi[i,j]
            count += 1
    return R,sqPhi

def dAdPhi(x, *args):
    
    T, phi, psi  = args

    R, Phi = get_R_Phi(x)
    om, eigv = get_phonons(Phi)
    A, B = get_AB(om, eigv, T)

    lambd = Lambda(om, eigv, T)
    Tmatrix = np.einsum()
    

def F(x,*args):

    R, sqPhi = get_R_sqPhi(x)
    nmod = len(R)

    om, eigv = np.linalg.eig(sqPhi)
    om = np.abs(om)
    Phi = np.dot(sqPhi,sqPhi)

    #print_phonons(om)

    T, phi, psi  = args

    K_to_Ry=6.336857346553283e-06
    def nb(om,T):
        if T>1e-3:
            beta = 1/(T*K_to_Ry)
            return 1.0/(np.exp(om*beta)-1)
        else:
            return 0*om

    F_harm = om/2.0 - T*K_to_Ry*np.log(1+nb(om,T))
    F_harm = np.sum(F_harm)

    A, B = get_AB(om, eigv, T)

    avg_V = av_V(R,A,phi,psi)
    V_harm = 1/2*np.einsum('ij,ij', A, Phi)
    avg_V -= V_harm

    return F_harm + avg_V

def minimize_free_energy(T,phi,psi, R0):

    nmod = len(R0)

    Phi0 = d2V(R0,phi,psi)
    om, eigv = get_phonons_r(Phi0) # Absolute value at first trial!

    A, B = get_AB(om, eigv, T)
    Phi0 = kappa(R0,A,phi,psi)
    om, eigv = get_phonons_r(Phi0) # Same here
    Phi0 = np.einsum('k,ik,jk->ij', om, eigv, eigv) # Now regularize Phi0 
    print('R', R0,'om',  om, 'Ry',  om*13.605*8065.5401, ' cmm1', om*13.605*241.798, 'THz')

    x0 = get_x0(R0,Phi0)

    res = minimize(F, x0, args = (T,phi,psi))
    R, Phi = get_R_Phi(res['x'])
    om, eigv = get_phonons(Phi)
    om = np.abs(om)
    print('R', R,'om',  om, 'Ry',  om*13.605*8065.5401, ' cmm1', om*13.605*241.798, 'THz')
    print(res)
    print(F(res['x'], T,phi,psi))
    A, B = get_AB(om, eigv, T)
     
    kap = kappa(R,A, phi,psi )
    forc = force(R,A, phi,psi)
    print('check f', np.linalg.norm(forc))
    print('check A', np.linalg.norm(B-np.dot( A, kap)))
    sys.exit()
    return R, om, A, B


def F_pos(x,*args):

    om = x
    om = np.abs(om)
    R, m, T, k, k3 = args

    K_to_Ry=6.336857346553283e-06
    def nb(om,T):
        if T>1e-3:
            beta = 1/(T*K_to_Ry)
            return 1.0/(np.exp(om*beta)-1)
        else:
            return 0


    arg = om/(T*K_to_Ry)/2.0
    A = 1/np.tanh(arg)/(2*om)/m

    F_harm = om/2.0 - T*K_to_Ry*np.log(1+nb(om,T))

    avg_V = k3/24.0*R**4 + k3/8.0*A**2 + 1/2*k*A + (1/2*k+k3/4.0*A)*R**2 - 1/2.0*m*om**2*A

    return F_harm + avg_V



def func_stress1(t, y,  ppsi, pphi, Eamp, om_L, gamma, m, M, a0):

    #k, k3, Eamp, om_L = args
    R,V,A,AP,AS,a,pa  = y

    strain = a/a0 - 1

    k, k3 = get_coeff_from_strain_new(strain, ppsi, pphi)

    ydot = []
    ydot.append(m*V + pa/M/a*R)
    ydot.append(f(R,A,k,k3)/m + Eamp*np.sin(2*np.pi*om_L*t)/m -gamma*V -pa/M/a*V )
    ydot.append(AP)
    ydot.append(AS)
    ydot.append(-4*kappa(R,A,k,k3)*AP/m -k3*(AP + 2*R*V)*A/m)
    ydot.append(pa/M)
    ydot.append(+m**2*V**2/a - R/a*f(R,A,k,k3)/m)
    #print(t, ydot[-1])

    return ydot

def func_stress2(t, y,  ppsi, pphi, Eamp, om_L, gamma, m, M, a0):

    #k, k3, Eamp, om_L = args
    R,V,A,AP,AS,a,pa  = y

    strain = a/a0 - 1

    k, k3 = get_coeff_from_strain_new(strain, ppsi, pphi)

    press = pressure(R, A, m, a0**3, strain, ppsi, pphi)

    ydot = []
    ydot.append(m*V + pa/M/a*R)
    ydot.append(f(R,A,k,k3)/m + Eamp*np.sin(2*np.pi*om_L*t)/m -gamma*V -pa/M/a*V )
    ydot.append(AP)
    ydot.append(AS)
    ydot.append(-4*kappa(R,A,k,k3)*AP/m -k3*(AP + 2*R*V)*A/m)
    ydot.append(pa/M)
    ydot.append(press/M*a0**2)
    #print(t, ydot[-1])

    return ydot


def func_stress(t, y,  ppsi, pphi, Eamp, om_L, gamma, m, M, a0, ef):

    #k, k3, Eamp, om_L = args
    R,V,A,AP,AS,a,pa  = y

    strain = a/a0 - 1

    k, k3 = get_coeff_from_strain_new(strain, ppsi, pphi)

    press = pressure(R, A, m, a0**3, strain, ppsi, pphi)
    """
    Efield = Eamp*np.sin(2*np.pi*om_L*t)
    t0 = 2000/(4.8377687*1e-2)
    sig = 2000/(4.8377687*1e-2)
    Efield = Eamp*np.sin(2*np.pi*om_L*t)*np.exp(-(t-t0)**2/(sig**2))
    """
    Efield = ef(t)

    ydot = []
    ydot.append(m*V )
    ydot.append(f(R,A,k,k3)/m + Efield/m -gamma*V )
    ydot.append(AP)
    ydot.append(AS)
    ydot.append(-4*kappa(R,A,k,k3)*AP/m -k3*(AP + 2*R*V)*A/m)
    ydot.append(pa)
    ydot.append(press/M*a0**2 - gamma/10.*pa)

    return ydot

def npt_evolution(R, A, Eamp, om_L, gamma, ppsi, pphi, Time, NS, m, M, a0, ef):
    # om_L in THz, Time in fs

    om_L = om_L/(2.0670687*1e4)
    Time = Time/(4.8377687*1e-2)

    t = np.linspace(0,Time,NS)
    y0 = [R, 0, A, 0, 0, a0, 0]
    sol = odeint(func_stress, y0, t, args=(ppsi, pphi, Eamp, om_L, gamma, m, M, a0, ef), tfirst = True)
    #sol = solve_ivp(func_stress, (t[0],t[-1]), y0, args=(px, pv, Eamp, om_L, gamma, m, M, a0))
    #return sol.t, sol.y.T
    return t, sol

def classic(y,t, k, k3, Eamp, om_L, gamma, m):

    R,V = y
    ydot = []
    ydot.append(V)
    ydot.append(f_classic(R,k,k3)/m + Eamp*np.sin(2*np.pi*om_L*t)/m -gamma*V)    
    return ydot

def td_classic(R, Eamp, om_L, gamma, k, k3, Time, NS, m):

    om_L = om_L/(2.0670687*1e4)
    Time = Time/(4.8377687*1e-2)

    t = np.linspace(0,Time,NS)
    y0 = [R, 0]
    sol = odeint(classic, y0, t, args=(k, k3, Eamp, om_L, gamma, m))

    return t, sol

def gen_stochastic_u(om, T, m, NC):
    A = corr(om, T, m)
    B = pcorr(om, T, m)
    u = np.sqrt(A)*np.random.randn(NC)
    p = np.sqrt(B)*np.random.randn(NC)
    return u, p


def stochastic(y_,t,k, k3, Eamp, om_L, gamma,m):

    Nd = int(len(y_)/2)
    y = np.reshape(copy.copy(y_), (Nd, 2) )
    R = y[0,0]
    V = y[0,1]

    ydot = np.zeros((Nd,2))
    forces = np.zeros(Nd-1)
    antiforces = np.zeros(Nd-1)

    for i in range(1,Nd):
        forces[i-1] = f_classic(R+y[i,0],k,k3)
        antiforces[i-1] = f_classic(R-y[i,0],k,k3)

    f_av = np.sum(forces)/(float(Nd-1)) + np.sum(antiforces)/(float(Nd-1))
    
    ydot[0,0] = V
    ydot[0,1] = f_av/m + Eamp*np.sin(2*np.pi*om_L*t)/m -gamma*V

    for i in range(1,Nd):
        ydot[i,0] = y[i,1]
        ydot[i,1] = forces[i-1]/m

    ydot = np.reshape(ydot, 2*Nd)

    v = y_[2::2]
    #print("A", np.dot(v,v)/len(v))

    """
    R,V = y[:2]
    ydot = []
    ydot.append(V)
    ydot.append(f_classic(R,k,k3)/m + Eamp*np.sin(2*np.pi*om_L*t)/m -gamma*V)    

    for j in range(2,len(y)):
        if j%2 == 0:
            ydot.append(y[j+1])
        else:
            ydot.append(f_classic(R+y[j-1],k,k3)/m)
    """
    return ydot


def td_stochastic(R0,V0, u0, p0, Eamp, om_L, gamma, k, k3, Time, NS, m):
 
    y0 = np.zeros((len(u0)+1, 2))
    y0[0,0] = R0
    y0[0,1] = V0
    y0[1:,0] = u0
    y0[1:,1]= p0/m
    y0 = np.reshape(y0, 2*len(y0[:,0]))
    
    v = y0[2::2]
    #print("A", np.dot(v,v)/len(v))

    Time = Time/(4.8377687*1e-2)
    t = np.linspace(0,Time,NS)

    sol= odeint(stochastic, y0, t, args=(k, k3, Eamp, om_L, gamma, m))
    return t, sol
