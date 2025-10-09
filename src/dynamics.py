import numpy as np
import sys
import scipy
from scipy.integrate import odeint, solve_ivp
import copy

import os
from averages import force, kappa, ext_for
from averages import torch_force, torch_kappa, torch_ext_for
from phonons import *

from torchdiffeq import odeint
import torch
import math

matplotlib.rc('axes',lw=1.5)
matplotlib.rcParams['axes.labelsize'] = 17
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13


 
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

def get_y0_torch(R,P,A,B,C):

    nmod = len(R)
    y0 = torch.zeros(2*nmod + 3*nmod ** 2, dtype=R.dtype, device=R.device)

    y0[:nmod] = R
    y0[nmod:2 * nmod] = P

    A_lin = A.reshape(-1)
    B_lin = B.reshape(-1)
    C_lin = C.reshape(-1)

    y0[2*nmod : 2*nmod + nmod**2] = A_lin
    y0[2*nmod + nmod**2 : 2*nmod + 2*nmod**2] = B_lin
    y0[2*nmod + 2*nmod**2 : 2*nmod + 3*nmod**2] = C_lin

    return y0
    

def tdscha(t,y, phi, chi, psi, field, gamma):

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
    ydot[nmod:2*nmod] = f + ext_for(t, field) - gamma*P #

    Adot = C + np.transpose(C)

    Bdot = -np.dot(curv, C) 
    Bdot = Bdot + np.transpose(Bdot)
    Cdot = B - np.dot(A,curv) # -0.001*C
    
    ydot[2*nmod:2*nmod+nmod**2] = np.reshape(Adot, nmod**2)
    ydot[2*nmod+nmod**2:2*nmod+2*nmod**2] = np.reshape(Bdot, nmod**2)
    ydot[2*nmod+2*nmod**2:2*nmod+3*nmod**2] = np.reshape(Cdot, nmod**2)

    return ydot


def tdscha_torch(t,y, phi, chi, psi, field, gamma):

    print("time ", t*4.8377687*1e-2)

    L = y.numel()
    nmod = int((-2 + math.sqrt(4 + 12 * L)) / 6)

    R = y[:nmod]
    P = y[nmod:2*nmod]

    A_lin = y[2*nmod:2*nmod + nmod**2]
    B_lin = y[2*nmod + nmod**2:2*nmod + 2*nmod**2]
    C_lin = y[2*nmod + 2*nmod**2:2*nmod + 3*nmod**2]

    A = A_lin.reshape(nmod, nmod)
    B = B_lin.reshape(nmod, nmod)
    C = C_lin.reshape(nmod, nmod)

    f = torch_force(R, A, phi, chi, psi)          
    curv = torch_kappa(R, A, phi, chi, psi)

    # Allocate output derivative
    ydot = torch.zeros_like(y)

    # ydot components
    ydot[:nmod] = P
    ydot[nmod:2*nmod] = f #+ ext_for(t, field) - gamma * P     WARNING

    Adot = C + C.t()

    Bdot = -torch.matmul(curv, C)
    Bdot = Bdot + Bdot.t()

    Cdot = B - torch.matmul(A, curv)  # e.g., minus A*curv

    # Pack back into flat ydot
    ydot[2*nmod:2*nmod + nmod**2] = Adot.reshape(-1)
    ydot[2*nmod + nmod**2:2*nmod + 2*nmod**2] = Bdot.reshape(-1)
    ydot[2*nmod + 2*nmod**2:2*nmod + 3*nmod**2] = Cdot.reshape(-1)

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
        t_eval = np.linspace(init_t,init_t+Time,NS)
        tspan = [init_t,init_t+Time]
        sol = solve_ivp(tdscha, tspan, y0, t_eval=t_eval, args=(phi, chi,  psi, field, gamma))
        save(label+'_%d' %i, sol.t, sol.y)

        y0 = sol.y[:,-1]
        init_t+=Time

    return sol

def torch_evolution(R, P, A, B, C,  field, gamma, phi, chi, psi, Time, NS, y0=None, init_t=0, chunks=1, label="solution"):
    # om_L in THz, Time in fs

    init_t = init_t/(4.8377687*1e-2)
    Time = Time/(4.8377687*1e-2)
    Time = Time/chunks
    NS = int(NS/chunks)

    phi, chi, psi, R, P, A, B, C, field = torch_init(phi, chi, psi, R, P, A, B, C, field)

    if y0 is None:
        y0 = get_y0_torch(R,P,A,B,C)

    device, dtype = y0.device, y0.dtype

    for i in range(chunks):
        tspan = torch.linspace(init_t,init_t+Time, NS, device=device, dtype=dtype)
        #sol = solve_ivp(func, tspan, y0, t_eval=t_eval, args=(phi, chi,  psi, field, gamma))
        with torch.no_grad():
            func = lambda t, y: tdscha_torch(t, y, phi, chi, psi, field, gamma)
            sol = odeint(func, y0, tspan, method='dopri5')
        save_torch(label+'_%d' %i, tspan,  sol)

        y0 = sol[-1]
        init_t+=Time

    return sol

def save(label, t, sol):
    sol = np.transpose(sol)
    sh = np.shape(sol)
    sh = [sh[0],sh[1]+1]
    y = np.zeros(sh)
    y[:,0] = t
    y[:,1:] = sol
    np.savez_compressed(label,y)


def save_torch(label, t, sol):
    t_np   = t.detach().cpu().numpy()
    sol_np = sol.detach().cpu().numpy() 

    y = np.concatenate([t_np[:, None], sol_np], axis=1)  
    np.savez_compressed(label, y)


def torch_init(phi, chi, psi, R, P, A, B, C, field):

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    phi    = torch.from_numpy(phi).to(device=device, dtype=dtype)
    chi    = torch.from_numpy(chi).to(device=device, dtype=dtype)
    psi    = torch.from_numpy(psi).to(device=device, dtype=dtype)
    R      = torch.from_numpy(R).to(device=device, dtype=dtype)
    P      = torch.from_numpy(P).to(device=device, dtype=dtype)
    A      = torch.from_numpy(A).to(device=device, dtype=dtype)
    B      = torch.from_numpy(B).to(device=device, dtype=dtype)
    C      = torch.from_numpy(C).to(device=device, dtype=dtype)

    Zeff = field['Zeff']
    field['Zeff'] = torch.from_numpy(Zeff).to(device=device, dtype=dtype) 

    edir = field['edir']
    field['edir'] = torch.tensor(edir).to(device=device, dtype=dtype) 

    return phi, chi, psi, R, P, A, B, C, field



