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


def get_phonons_THz(phi):
    om, eigv = np.linalg.eigh(phi)
    om = np.sqrt(om)*13.6*241.8
    return om, eigv

def get_phonons(phi):
    om, eigv = np.linalg.eigh(phi)
    om = np.sqrt(om)
    return om, eigv

def get_phonons_r(phi):
    om, eigv = scipy.linalg.eigh(phi)
    om = np.abs(om)
    om = np.sqrt(om)
    return om, eigv

def regularize(phi):
    om, eigv = scipy.linalg.eigh(phi)
    om = np.abs(om)

    phi = np.einsum('s, is, js -> ij', om, eigv, eigv)
    return phi

def print_phonons(om):
    print("phonons")
    for i in range(len(om)):
        print("Mode %d" %(i+1), om[i]*241.8*13.6)

def print_phonons_mat(phi):
    om, eigv = np.linalg.eigh(phi)
    mask = np.where(om<0)
    om = np.abs(om)
    om = np.sqrt(om)
    om[mask] *= -1
    print("phonons")
    for i in range(len(om)):
        print("Mode %d" %(i+1), om[i]*241.8*13.6, 'THz', om[i]*8065.54429*13.6, 'cmm1')

def remove_translations(om, eigv, thr=1e-6):
    # thr of 1e-6 corresponds to around 0.01 THz and 0.3 cmm1
    nmod = len(om)
    nom = copy.deepcopy(om)
    neigv = copy.deepcopy(eigv)
    mask = np.where(np.abs(nom)>thr)
    nacoustic = nmod-len(mask[0])
    if nacoustic!=3:
        print("WARNING, n acoustic modes = ", nacoustic)
    nom = nom[mask]
    neigv = neigv[:,mask]
    neigv = neigv[:,0,:]
    return nom, neigv

def remove_translations_from_mat(phi, thr=1e-6):
    # thr of 1e-6 corresponds to around 0.01 THz and 0.3 cmm1

    om, eigv = np.linalg.eigh(phi)
    mask = np.where(om<0)
    om = np.abs(om)
    om = np.sqrt(om)
    om[mask] *= -1

    nmod = len(om)
    nom = copy.deepcopy(om)
    neigv = copy.deepcopy(eigv)
    mask = np.where(np.abs(nom)>thr)
    nacoustic = nmod-len(mask[0])
    if nacoustic!=3:
        print("WARNING, n acoustic modes = ", nacoustic)
    nom = nom[mask]
    neigv = neigv[:,mask]
    neigv = neigv[:,0,:]
    return nom, neigv
    
 
def get_AB(fom, feigv, T):
    K_to_Ry=6.336857346553283e-06
 
    om, eigv = remove_translations(fom, feigv)
    if T<0.001:
        tanh = np.ones(len(om))
    else:
        arg = om/(T*K_to_Ry)/2.0
        tanh = np.tanh(arg)

    lambd = 1/tanh/(2*om)
    A= np.einsum('s,is,js->ij', lambd, eigv, eigv)

    lambd = om/tanh/(2)
    B= np.einsum('s,is,js->ij', lambd, eigv, eigv)
    return A,B

def get_alpha(om, eigv, T):
    K_to_Ry=6.336857346553283e-06
 
    if T<0.001:
        tanh = np.ones(len(om))
    else:
        arg = om/(T*K_to_Ry)/2.0
        tanh = np.tanh(arg)

    lambd = tanh*(2*om)
    alpha = np.einsum('s,is,js->ij', lambd, eigv, eigv)

    return alpha


def inv_Phi(fom, feigv):
    om, eigv = remove_translations(fom, feigv)
    return np.einsum('k, ik, jk ->ij', 1/om**2, eigv, eigv)
    
def force(R,A,phi,chi,psi):
    f1 = np.einsum('ij,j->i', phi, R)  
    f3 = 1/6*np.einsum('ijkl,j,k,l->i', psi, R, R, R)
    fq3 = 1/2*np.einsum('ijkl,j,kl->i', psi, R, A)

    f2 = 1/2*np.einsum('ijk,j,k->i', chi, R, R)
    fq2 = 1/2*np.einsum('ijk,jk->i', chi, A)

    return -f1-f3-fq3-f2-fq2

def force_t(R,A,phi,chi,psi):
    # now R and A are function of time, first index
    print("f1")
    f1 = np.einsum('ij,tj->ti', phi, R)  
    print("f3")
    f3 = 1/6*np.einsum('ijkl,tj,tk,tl->ti', psi, R, R, R, optimize = "optimal")
    print("fq")
    fq3 = 1/2*np.einsum('ijkl,tj,tkl->ti', psi, R, A, optimize = "optimal")

    print("f2")
    f2 = 1/2*np.einsum('ijk,tj,tk->ti', chi, R, R, optimize = "optimal")
    print("fq2")
    fq2 = 1/2*np.einsum('ijk,tjk->ti', chi, A)

    return -f1-f3-fq3-f2-fq2

def f_classic(R,phi,chi,psi):
    f1 = np.einsum('ij,j->i', phi, R)  
    f3 = 1/6*np.einsum('ijkl,j,k,l->i', psi, R, R, R)
    f2 = 1/2*np.einsum('ijk,j,k->i', chi, R, R)

    return -f1-f3-f2

def ext_for(t, field):

    Eamp = field['amp']
    om_L = field['freq']
    edir = field['edir']
    t0 = field['t0']
    sig = field['sig']
    case = field['type']
    Zeff = field['Zeff']
    eps = field['eps']
    
    if np.abs(np.linalg.norm(edir) - 1) > 1e-7:
        sys.exit("Direction not normalized")

    Eeff = Eamp * 2.7502067*1e-7  * 2 /(1+np.sqrt(eps))
    freq = om_L/(2.0670687*1e4)

    nmod = len(Zeff)
    nat = int(nmod/3)

    force = []
    for i in range(nat):
        force.append(np.dot(Zeff[3*i:3*i+3,:], edir)*Eeff*np.sqrt(2))
    force = np.array(force)
    force = np.reshape(force, nmod)
    #force = force / np.sqrt(masses)  EFFECTIVE CHARGES ARE ALREADY RESCALED FOR MASSES


    if case=='sine':
        return force*np.sin(2*np.pi*freq*t)
    elif case=='gaussian1':
        t0 = t0/(4.8377687*1e-2)
        sig = 1/(2*np.pi*freq)
        return -force * (t-t0)/sig * np.exp(-0.5*(t-t0)**2/sig**2 + 0.5)
    elif case=='pulse':
        t0 = t0/(4.8377687*1e-2)
        sig = sig/(4.8377687*1e-2)
        return -force * np.cos(2*np.pi*freq*(t-t0)) * np.exp(-0.5*(t-t0)**2/sig**2)
    elif case=='gaussian2':
        t0 = t0/(4.8377687*1e-2)
        sig = 1/(np.sqrt(2)*np.pi*freq)
        return -force * (1 - (t-t0)**2/sig**2) * np.exp(-0.5*(t-t0)**2/sig**2)
    else:
        sys.exit("Field not implemented")

def kappa(R, A, phi, chi, psi):
    k1 = 1/2*np.einsum('ijkl, k,l->ij', psi, R, R)
    k2 = 1/2*np.einsum('ijkl, kl->ij', psi, A)

    k3 = np.einsum('ijk,k->ij', chi, R)
    #print_phonons_mat(phi + k1 + k2 + 0*k3)
    #sys.exit()
 
    return phi + k1 + k2 + k3

def d2V(R, phi, chi,  psi):
    k1 = 1/2*np.einsum('ijkl, k,l->ij', psi, R, R)
    k2 = np.einsum('ijk,k->ij', chi, R)
    return phi+k1+k2

def av_V(R, A, phi, chi, psi):
  
    V0 = 1/2*np.einsum('i,j,ij', R, R, phi)
    V1 = 1/2*np.einsum('ij,ij', A, phi)
    V2 = 1/24*np.einsum('ijkl,i,j,k,l', psi, R ,R ,R ,R, optimize=True)
    V3 = 1/4*np.einsum('ijkl,i,j,kl', psi, R ,R ,A)
    V4 = 1/8*np.einsum('ijkl,ij,kl', psi, A ,A)
 
    lamb, vect = np.linalg.eigh(A)
    #Q = np.einsum('s, is,js,ks,ls -> ijkl', lamb**2, vect, vect, vect, vect)
    #V5 = 1/8*np.einsum('ijkl,ijkl', psi, Q)
    #V5 = 1/8*np.einsum('ijkl,im,jm,km,lm,m', psi, vect, vect, vect, vect, lamb**2, optimize= 'optimal')

    V6 = 1/6*np.einsum('ijk,i,j,k', chi, R, R, R)
    V7 = 1/2*np.einsum('ijk,i,jk', chi, R, A)

    return V0 + V1 + V2 + V3 + V4 + V6 + V7

def av_V_t(R, A, phi, chi, psi):
  
    V0 = 1/2*np.einsum('ti,tj,ij->t', R, R, phi, optimize = 'optimal')
    V1 = 1/2*np.einsum('tij,ij->t', A, phi, optimize = 'optimal')
    V2 = 1/24*np.einsum('ijkl,ti,tj,tk,tl->t', psi, R ,R ,R ,R, optimize='optimal')
    V3 = 1/4*np.einsum('ijkl,ti,tj,tkl->t', psi, R ,R ,A, optimize = 'optimal')
    V4 = 1/8*np.einsum('ijkl,tij,tkl->t', psi, A ,A , optimize = 'optimal')
 
    #Q = np.einsum('s, is,js,ks,ls -> ijkl', lamb**2, vect, vect, vect, vect)
    #V5 = 1/8*np.einsum('ijkl,ijkl', psi, Q)
    #V5 = 1/8*np.einsum('ijkl,im,jm,km,lm,m', psi, vect, vect, vect, vect, lamb**2, optimize= 'optimal')

    V6 = 1/6*np.einsum('ijk,ti,tj,tk->t', chi, R, R, R, optimize = 'optimal')
    V7 = 1/2*np.einsum('ijk,ti,tjk->t', chi, R, A, optimize = 'optimal')

    return V0 + V1 + V2 + V3 + V4 + V6 + V7
 
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

def func(t,y, phi, chi, psi, field, gamma):

    print("time ", t*4.8377687*1e-2)
    nmod = int((-2 + np.sqrt(4+12*len(y)))/6)

    R = y[:nmod]    
    P = y[nmod:2*nmod] 

    A_lin = y[2*nmod:2*nmod+nmod**2] 
    B_lin = y[2*nmod+nmod**2:2*nmod+2*nmod**2] 
    C_lin = y[2*nmod+2*nmod**2:2*nmod+3*nmod**2] 

    A = np.reshape(A_lin, (nmod, nmod))
    B = np.reshape(B_lin, (nmod, nmod))
    C = np.reshape(C_lin, (nmod, nmod))

    f = force(R,A,phi,chi,psi)
    #f = f_classic(R,phi,psi)
    curv = kappa(R,A,phi,chi,psi)

    ydot = np.zeros(len(y))

    ydot[:nmod] = P
    ydot[nmod:2*nmod] = f + ext_for(t, field)# -0.001*P #

    Adot = C + np.transpose(C)

    Bdot = -np.dot(curv, C) 
    Bdot = Bdot + np.transpose(Bdot)
    Cdot = B - np.dot(A,curv) # -0.001*C
    
    ydot[2*nmod:2*nmod+nmod**2] = np.reshape(Adot, nmod**2)
    ydot[2*nmod+nmod**2:2*nmod+2*nmod**2] = np.reshape(Bdot, nmod**2)
    ydot[2*nmod+2*nmod**2:2*nmod+3*nmod**2] = np.reshape(Cdot, nmod**2)

    return ydot

def td_evolution(R, P, A, B, C,  field, gamma, phi, chi, psi, Time, NS, y0=None, init_t=0, chunks=1, label="solution"):
    # om_L in THz, Time in fs

    init_t = init_t/(4.8377687*1e-2)
    Time = Time/(4.8377687*1e-2)
    Time = Time/chunks
    NS = int(NS/chunks)

    if y0 is None:
        y0 = get_y0(R,P,A,B,C)

    for i in range(chunks):
        print("Chunk", i)
 
        #sol = odeint(func, y0, tfirst=True,  t, args=(phi, psi, field, gamma, masses))
        t_eval = np.linspace(init_t,init_t+Time,NS)
        tspan = [init_t,init_t+Time]
        sol = solve_ivp(func, tspan, y0, t_eval=t_eval, args=(phi, chi,  psi, field, gamma))
        save(label+'_%d' %i, sol.t, sol.y)

        y0 = sol.y[:,-1]
        init_t+=Time

    return sol

def save(label, t, sol):
    sol = np.transpose(sol)
    sh = np.shape(sol)
    print(sh)
    sh = [sh[0],sh[1]+1]
    y = np.zeros(sh)
    y[:,0] = t
    y[:,1:] = sol
    #np.save(label,y)
    np.savez_compressed(label,y)

def displace_along_mode(mod, eigv, eta):
    eta = eta * 1.889725988*np.sqrt(911.444175)
    return eigv[:,mod]*eta


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

def get_gradx0(R,Phi):

    nmod = len(R)
    Phi_lin = np.zeros(int((nmod**2+nmod)/2))

    count=0
    for i in range(nmod):
        for j in range(i,nmod):
            Phi_lin[count] = Phi[i,j]
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

    

def nb(om,T, thr=1e-6):
    K_to_Ry=6.336857346553283e-06
    
    if np.any(om<0):
        sys.exit("Error, negative frequency in the Bose-Einstein")
    if T>1e-3:
        beta = 1/(T*K_to_Ry)
        bos = np.zeros(len(om))
        mask = np.where(om>thr)
        bos[mask] = 1.0/(np.exp(om[mask]*beta)-1)
        #print('here')
        #print(om)
        #print(bos)
        return bos
    else:
        return 0*om

def Lambda(om,eigv,T, thr=1e-6):
    
    nmod = len(om)
    F0 = np.zeros((nmod,nmod))

    K_to_Ry=6.336857346553283e-06
    
    for i in range(nmod):
        if np.abs(om[i])<thr:
            F[i,:] = 0
            F[:,i] = 0
        else:
            n_mu = nb(om[i],T)
            for j in range(i,nmod):
                if np.abs(om[i])>=thr:
                    n_nu = nb(om[j],T)
                    dn_domega_nu = -1/(T*K_to_Ry)*n_nu*(1+n_nu)
                    if j==i:
                        F0[i,i] = 2*((2*n_nu+1)/(2*omega[j]) - dn_domega_nu)
                        F0[i,i] /= om[i]**2
                    else:
                        F0[i,j] = 2*( (n_mu + n_nu + 1)/(om[i] + om[j]) - (n_mu-n_nu)/(om[i]-om[j]) ) / (om[i]*om[j])
                        F0[j,i] = F0[i,j]

    lamb = np.einsum('ij,aj,bi,cj,di->abcd', F0, eigv, eigv, eigv, eigv)  # masses
    return -8*lamb

def dAdPhi(x, *args):
    
    T, phi, psi  = args

    R, Phi = get_R_sqPhi(x)
    om, eigv = np.linalg.eigv(sqPhi)
    om = np.abs(om)
    #A, B = get_AB(om, eigv, T)

    lambd = Lambda(om, eigv, T)

def grad(x, *args):
    
    T, phi, chi, psi  = args

    R, sqPhi = get_R_sqPhi(x)
    om, eigv = np.linalg.eigh(sqPhi)
    om = np.abs(om)
    A, B = get_AB(om, eigv, T)
    Phi = np.dot(sqPhi,sqPhi)

    gradPhi = kappa(R, A, phi, chi, psi) - Phi
    gradPhi = np.dot(sqPhi,gradPhi) + np.dot(gradPhi,sqPhi)

    forc = force(R,A,phi,chi,psi)

    Phi_inv = inv_Phi(om, eigv)
    gradR = np.dot(Phi_inv, forc) 

    gradx = get_gradx0(gradR, gradPhi)
    print("Position gradient (A\sqrt{u})")
    print(gradR/1.89/np.sqrt(911))
    print("FC gradient (meV/A^2u)")
    print(gradPhi*1.89**2*911*13605)
    #print("Gradient", np.linalg.norm(gradx))
    #print("Condition", np.linalg.norm(Phi-kappa(R,A,phi, chi, psi)))
    #print("Condition", np.linalg.norm(B-np.dot(A,kappa(R,A,phi,chi,psi))))

    return gradx
    

def F(x,*args):

    R, sqPhi = get_R_sqPhi(x)
    nmod = len(R)

    om, eigv = np.linalg.eigh(sqPhi)
    om = np.abs(om)
    Phi = np.dot(sqPhi,sqPhi)

    #print_phonons(om)

    T, phi, chi, psi  = args


    K_to_Ry=6.336857346553283e-06
    F_harm = om/2.0 - T*K_to_Ry*np.log(1+nb(om,T))
    F_harm = np.sum(F_harm)

    A, B = get_AB(om, eigv, T)

    avg_V = av_V(R,A,phi,chi,psi)
    V_harm = 1/2*np.einsum('ij,ij', A, Phi)
    avg_V -= V_harm
     
    print("Iter ", F_harm *13605, avg_V *13605, (F_harm+avg_V)*13605 )

    return (F_harm + avg_V)

def minimize_free_energy(T,phi,chi,psi, R0):

    nmod = len(R0)

    Phi0 = d2V(R0,phi,chi,psi)
    om, eigv = get_phonons_r(Phi0) # Absolute value at first trial!
    print("Initial phonons")
    print_phonons_mat(Phi0)

    A, B = get_AB(om, eigv, T)
    Phi0 = kappa(R0,A,phi,chi,psi)
    om, eigv = get_phonons_r(Phi0) # Same here
    Phi0 = np.einsum('k,ik,jk->ij', om**2, eigv, eigv) # Now regularize Phi0 

    print("Curvature phonons")
    print_phonons_mat(Phi0)

    x0 = get_x0(R0,Phi0)
    print("Initial gradient") 
    print(grad(x0, T, phi, chi, psi))

    #res = minimize(F, x0, args = (T,phi,chi,psi))
    res = minimize(F, x0, args = (T,phi,chi,psi), jac = grad, method = 'BFGS', options={'gtol':1e-8})
    #res = minimize(F, x0, args = (T,phi,chi,psi), method = 'CG', jac = grad, options={'gtol':1e-8})
    #res = minimize(F, x0, args = (T,phi,psi), method = 'CG') #options={'gtol':1e-30})
    R, Phi = get_R_Phi(res['x'])
    om, eigv = get_phonons(Phi)
    om = np.abs(om)

    print("Final phonons")
    print_phonons(om)

    #print('R', R,'om',  om, 'Ry',  om*13.605*8065.5401, ' cmm1', om*13.605*241.798, 'THz')
    print(res)
    print("Free energy")
    print(F(res['x'], T,phi,chi,psi)*13605)
    print("Final grad")
    print(grad(res['x'], T, phi, chi, psi))
    A, B = get_AB(om, eigv, T)
     
    kap = kappa(R,A, phi,chi,psi )
    forc = force(R,A, phi,chi,psi)
    print('check f', np.linalg.norm(forc))
    print('check A', np.linalg.norm(B-np.dot( A, kap)))
    return R, om, A, B


def iter_minimiz(T,phi,psi, R0, maxiter=10):

    K_to_Ry=6.336857346553283e-06
    nmod = len(R0)

    Phi0 = d2V(R0,phi,psi)

    """
    om, eigv = get_phonons(Phi0)
    om, eigv = remove_translations(om, eigv)
    nmod = len(om)

    phimm = np.einsum('ij,im,jm->m', phi, eigv, eigv)
    psimm = np.einsum('ijkl, im ,jm, km, lm-> m', psi, eigv, eigv, eigv, eigv, optimize=True)
    
    def obj(x, *args):

        ph, ps, T = args
        if T<0.001:
            tanh = x
        else:
            arg = x/(T*K_to_Ry)/2.0
            tanh = np.tanh(arg)

        lambd = 1/tanh/(2*x)
        #print(lambd, T*K_to_Ry/x**2)
        return x**2 - ph - 1/2*ps*lambd

    newom = []
    for i in range(nmod):
        phi0 = (phimm[i]+np.sqrt(phimm[i]**2+2*psimm[i]*T*K_to_Ry))/2
        res = scipy.optimize.fsolve(obj, phi0, args = (phimm[i], psimm[i], T))
        ph = phimm[i]
        ps = psimm[i]
        print("phi ", phimm[i])
        print("psi ", psimm[i])
        arg = res[0]/(T*K_to_Ry)/2.0
        tanh = np.tanh(arg)
        lambd = 1/tanh/(2*res[0])
        #print("fact", K_to_Ry*T/res[0], lambd)
        #print(obj(np.sqrt(phi0), ph, ps, T), obj(res[0], ph, ps, T))
        print(om[i]*13.6*241.8, res[0]*13.6*241.8,np.sqrt(phi0)*13.6*241.8)
        newom.append(res[0])
    newom =np.array(newom)
    Phi0 = np.einsum('s, is, js-> ij', newom**2, eigv, eigv)
    """

    for i in range(maxiter):
        om, eigv = get_phonons_r(Phi0) # Absolute value at first trial!
        A, B = get_AB(om, eigv, T)
        kap = kappa(R0,A,phi,psi)
        check = np.linalg.norm(B-np.dot( A, kap))
        print("Iter", i+1, check)
        if check < 1e-7:
            print("Optimization successfull")
            return A
        Phi0 = kap
        print_phonons_mat(Phi0)
    sys.exit("Optimization not succesful")

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
