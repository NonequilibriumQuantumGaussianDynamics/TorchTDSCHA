import numpy as np
from scipy.optimize import minimize

def project_threefold_modes(u1,u2,u3, minim_thr = 1e-10, rounded = True):

    def Rx(r):
        R= np.array([[1,0,0],
                   [0,np.cos(r), np.sin(r)],
                   [0,-np.sin(r), np.cos(r)]])
        return R
    def Rz(r):
        R = np.array([[np.cos(r), np.sin(r),0],
                     [-np.sin(r), np.cos(r), 0],
                     [0,0,1]])
        return R

    def obj(vect):
        r, s, t = vect
        
        M = np.dot(Rx(r),np.dot(Rz(s),Rx(t)))
          
        e1 = M[0,0]*u1 + M[0,1]*u2 + M[0,2]*u3
        e2 = M[1,0]*u1 + M[1,1]*u2 + M[1,2]*u3
        e3 = M[2,0]*u1 + M[2,1]*u2 + M[2,2]*u3
        c1 = np.dot(e1[::3], e1[::3])
        c2 = np.dot(e2[1::3], e2[1::3])
        c3 = np.dot(e3[2::3], e3[2::3])
        print(c1**2 + c2**2 + c3**2)

        return c1**2 + c2**2 + c3**2

    popt = minimize(obj, [0,0,0], tol=minim_thr)
    print(popt)
    r, s, t = popt['x']
    M = np.dot(Rx(r),np.dot(Rz(s),Rx(t)))

    e1 = M[0,0]*u1 + M[0,1]*u2 + M[0,2]*u3
    e2 = M[1,0]*u1 + M[1,1]*u2 + M[1,2]*u3
    e3 = M[2,0]*u1 + M[2,1]*u2 + M[2,2]*u3

    nmod = len(u1)
    def round_vector(v):
        for i in range(nmod):
            if np.abs(v[i])<1e-4:
                v[i]=0
            else:
                v[i]=round(v[i],4)
        return v
    
    if rounded:
        e1 = round_vector(e1)
        e2 = round_vector(e2)
        e3 = round_vector(e3)
    return e1, e2, e3


