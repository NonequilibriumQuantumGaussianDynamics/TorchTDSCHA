import numpy as np
import copy
import matplotlib
from matplotlib import pyplot as pl
from model_multi import *
from init import *

matplotlib.rcParams['axes.labelsize'] = 17
matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13

T=0
Time = 200
NS = 4000
Eamp = 0
om_L = 10
gamma = 0

nat, nmod, phi, psi, R, P,  masses, A, B, C = init(2,T)

R[0] = 0.4
R, om, A, B = minimize_free_energy(T,phi,psi, R)

R[0]=0.0
t, sol    = td_evolution(R, P, A, B, C,  Eamp, om_L, gamma, phi, psi, Time, NS)
#pl.plot(t,sol[:,2*nmod  ])
pl.plot(t,sol[:,0  ])
forc = []
ene = []
pot = []
kin = []
quant = []

for i in range(len(t)):
    #forc.append(ext_for(t[i], Eamp, om_L, nmod)[0])
    R = sol[i,:nmod]
    P = sol[i,nmod:2*nmod]
    A_lin = sol[i,2*nmod:2*nmod+nmod**2]
    B_lin = sol[i,2*nmod+nmod**2:2*nmod+2*nmod**2]

    A = np.reshape(A_lin, (nmod, nmod))
    B = np.reshape(B_lin, (nmod, nmod))
  
    ene.append(av_V(R,A,phi,psi) + np.sum(P**2/2) + np.trace(B)/2)   
    pot.append(av_V(R,A,phi,psi))
    kin.append(np.sum(P**2/2))
    quant.append(np.trace(B)/2)

ene = np.array(ene)*13600
pot = np.array(pot)*13600
quant = np.array(quant) *13600
kin = np.array(kin) *13600

"""
pl.plot(t, ene, label='t')
pl.plot(t, kin, label = 'k')
pl.plot(t, quant, label = 'q')
pl.plot(t, pot, label = 'p')
"""

pl.legend()
pl.show()

#pl.show()



