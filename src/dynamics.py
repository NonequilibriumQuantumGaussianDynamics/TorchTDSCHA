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
from averages import force, kappa, ext_for
from phonons import *

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
    ydot[nmod:2*nmod] = f + ext_for(t, field) - gamma*P #

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



