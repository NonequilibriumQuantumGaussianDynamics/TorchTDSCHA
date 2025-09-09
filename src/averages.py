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
#from dynamics import *
matplotlib.rc('axes',lw=1.5)
matplotlib.rcParams['axes.labelsize'] = 17
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13




RyToeV = 13.605703976


def force(R,A,phi,chi,psi):
    f1 = np.einsum('ij,j->i', phi, R)  
    f3 = 1/6*np.einsum('ijkl,j,k,l->i', psi, R, R, R, optimize = 'optimal')
    fq3 = 1/2*np.einsum('ijkl,j,kl->i', psi, R, A, optimize = 'optimal')

    f2 = 1/2*np.einsum('ijk,j,k->i', chi, R, R, optimize = 'optimal')
    fq2 = 1/2*np.einsum('ijk,jk->i', chi, A, optimize = 'optimal')

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

def av_d3(R, chi, psi):
    d3 = np.einsum('ijkl,l->ijk', psi, R)
    d3 = d3 + chi
    return d3

def V_classic(R, phi, chi, psi):
    V2 = 1/2*np.einsum('ij,i,j', phi, R, R)
    V3 = 1/6*np.einsum('ijk,i,j,k', chi, R, R, R, optimize = "optimal")
    V4 = 1/24*np.einsum('ijkl,i,j,k,l', psi, R, R, R, R, optimize = "optimal")

    return V2 + V3 + V4

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
    elif case=='sinc':
        return -force * np.sinc(t*freq*2)
    else:
        sys.exit("Field not implemented")

def kappa(R, A, phi, chi, psi):
    k1 = 1/2*np.einsum('ijkl, k,l->ij', psi, R, R, optimize = 'optimal')
    k2 = 1/2*np.einsum('ijkl, kl->ij', psi, A, optimize = 'optimal')

    k3 = np.einsum('ijk,k->ij', chi, R)
    #print_phonons_mat(phi + k1 + k2 + 0*k3)
    #sys.exit()
 
    return phi + k1 + k2 + k3

def kappa_t(R, A, phi, chi, psi):
    k1 = 1/2*np.einsum('ijkl, tk,tl->tij', psi, R, R, optimize = 'optimal')
    k2 = 1/2*np.einsum('ijkl, tkl->tij', psi, A, optimize = 'optimal')

    k3 = np.einsum('ijk,tk->tij', chi, R, optimize = 'optimal')
 
    return phi + k1 + k2 + k3


def d2V(R, phi, chi,  psi):
    k1 = 1/2*np.einsum('ijkl, k,l->ij', psi, R, R)
    k2 = np.einsum('ijk,k->ij', chi, R)
    return phi+k1+k2

def av_V(R, A, phi, chi, psi):
  
    V0 = 1/2*np.einsum('i,j,ij', R, R, phi)
    V1 = 1/2*np.einsum('ij,ij', A, phi)
    V2 = 1/24*np.einsum('ijkl,i,j,k,l', psi, R ,R ,R ,R, optimize="optimal")
    V3 = 1/4*np.einsum('ijkl,i,j,kl', psi, R ,R ,A, optimize="optimal")
    V4 = 1/8*np.einsum('ijkl,ij,kl', psi, A ,A, optimize="optimal")
 
    #Q = np.einsum('s, is,js,ks,ls -> ijkl', lamb**2, vect, vect, vect, vect)
    #V5 = 1/8*np.einsum('ijkl,ijkl', psi, Q)
    #V5 = 1/8*np.einsum('ijkl,im,jm,km,lm,m', psi, vect, vect, vect, vect, lamb**2, optimize= 'optimal')

    V6 = 1/6*np.einsum('ijk,i,j,k', chi, R, R, R, optimize="optimal")
    V7 = 1/2*np.einsum('ijk,i,jk', chi, R, A, optimize="optimal")

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
 
