#import cellconstructor as CC, cellconstructor.Phonons
import sys
import os
import numpy as np
import copy



def cast_inside_supercell(s_crystal):
    # folds the coordinates in such a way that all the crystal coordinates are 0<u<1

    new = copy.copy(s_crystal)
    for i in range(len(new)):
        for j in range(3):
            if new[i,j]<0:
                new[i,j]+=1
            elif new[i,j]>=1:
                new[i,j]-=1

    return new

def cast_inside_unitcell(at, nn):
    # crystal coord of the supercell atom

    at_uc = [at[i]*nn[i] for i in range(3)]

    for i in range(3):
        while at_uc[i]>=1:
            at_uc[i]-=1
        while at_uc[i]<0:
            at_uc[i]+=1

    return at_uc


def fold(at_uc):

    for i in range(3):
        while at_uc[i]>=1:
            at_uc[i]-=1
        while at_uc[i]<0:
            at_uc[i]+=1

    return at_uc


def distance(at_uc, atoms_uc, atoms_uc_coords, nn):
    at_uc = fold(at_uc)
    #print(at_uc)

    minims = [1e5]*len(atoms_uc)
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                n_at_uc = at_uc + np.array([i,0,0]) + np.array([0,j,0]) + np.array([0,0,k])
                #print(n_at_uc)
                for m in range(len(atoms_uc)):
                    dist = np.linalg.norm(n_at_uc - atoms_uc_coords[m])
                    #print(i,j,k,m,dist)
                    if dist < minims[m]:
                        minims[m] = dist
    return minims


def find_atoms_unitcell(s_crystal, nn, thr=0.1):
    nat_uc = int(len(s_crystal)/np.prod(nn))
    atoms_uc = []
    atoms_uc_coords = []
    group_at = []
    for i, at in enumerate(s_crystal):
        at_uc = [at[j]*nn[j] for j in range(3)]
        at_uc = fold(at_uc)
        #print(i, at_uc)
        flag = True
        for j in range(3):
            flag = flag * (at_uc[j]<1 and at_uc[j]>=0)
        if flag and len(atoms_uc)<nat_uc:
            # check if the found atom is not already contained inside the uc
            if len(atoms_uc)>0:
                minims = distance(at_uc, atoms_uc, atoms_uc_coords, nn)
                if np.min(minims) > thr:
                    atoms_uc.append(i)
                    atoms_uc_coords.append(at_uc)
                    print(i, "belongs")
                    group_at.append(len(atoms_uc)-1)
                else:
                    group = np.argmin(minims)
                    print("Atom %d belongs to group %d" %(i, group)) 
                    group_at.append(group)
            else:
                atoms_uc.append(i)
                atoms_uc_coords.append(at_uc)
                print(i, "belongs")
                group_at.append(len(atoms_uc)-1)
        elif flag and len(atoms_uc)>=nat_uc:
            #print("atom %d in eccess" %i, at_uc)
            dist = distance(at_uc, atoms_uc, atoms_uc_coords, nn)
            group = np.argmin(dist)
            print("atom %d belongs to group %d" %(i, group)) 
            group_at.append(group)
        else:
            print("atom %d false " %i)
    print(group_at)
    for i in range(nat_uc):
        print(i, group_at.count(i))
        if group_at.count(i) != np.prod(nn):
            sys.exit("Mapping wrong")
    return group_at

                   
def find_gamma(eig, eigv, group_at,  nat_uc, nat, thr = 1e-3):

    gamma = []
    for i in range(len(eig)):
        coeff_jk = []
        for k in range(3):
            sum = 0
            for j in range(nat_uc) :
                for jj in range(nat):
                    if group_at[jj] == j:
                        sum+= eigv[3*jj+k,i]
                coeff_jk.append(sum)

        tot = np.dot(coeff_jk,coeff_jk)
        #print(tot)
        if tot > thr:
            print("%d gamma" %(i+1), eig[i]*13.605*241.8)
            gamma.append(i)
    if len(gamma)!=3*nat_uc:
        sys.exit("Number of modes different than 3*nat unitcell")
    return gamma

def find_gamma_(eig, eigv, s_angstrom, unit_cell, nn,  thr = 1e-3):

    A_inv = np.linalg.inv(np.transpose(unit_cell))
    s_crystal = np.transpose(np.dot(A_inv, np.transpose(s_angstrom)))
    nat = len(s_crystal)

    group_at = find_atoms_unitcell(s_crystal, nn)
    nat_uc = int(len(s_crystal)/np.prod(nn))

    gamma = []
    for i in range(len(eig)):
        coeff_jk = []
        for k in range(3):
            sum = 0
            for j in range(nat_uc) :
                for jj in range(nat):
                    if group_at[jj] == j:
                        sum+= eigv[3*jj+k,i]
                coeff_jk.append(sum)

        tot = np.dot(coeff_jk,coeff_jk)
        #print(tot)
        if tot > thr:
            print("%d gamma" %(i+1), eig[i]*13.605*241.8)
            gamma.append(i)
    if len(gamma)!=3*nat_uc:
        sys.exit("Number of modes different than 3*nat unitcell")
    return gamma

def group_phonons(gamma, eig):
    # eig in THz

    group = [[gamma[0]]]

    for i  in range(1,len(gamma)):
        om = eig[gamma[ i]]
        found = False
        for j, gr in enumerate(group):
            if np.abs(eig[gr[0]]-om) < 0.5:
                group[j].append(gamma[i])
                found = True
        if found ==False:
            group.append([gamma[i]])

    print(group)
    return group

