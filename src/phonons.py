import numpy as np
import sys
import scipy
import copy
import os


def get_phonons_THz(phi):
    om, eigv = np.linalg.eigh(phi)
    om = np.sqrt(om) * 13.6 * 241.8
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

    phi = np.einsum("s, is, js -> ij", om, eigv, eigv)
    return phi


def print_phonons(om):
    print("phonons")
    for i in range(len(om)):
        print("Mode %d" % (i + 1), om[i] * 241.8 * 13.6)


def print_phonons_mat(phi):
    om, eigv = np.linalg.eigh(phi)
    mask = np.where(om < 0)
    om = np.abs(om)
    om = np.sqrt(om)
    om[mask] *= -1
    print("phonons")
    for i in range(len(om)):
        print("Mode %d" % (i + 1), om[i] * 241.8 * 13.6, "THz", om[i] * 8065.54429 * 13.6, "cmm1")


def remove_translations(om, eigv, thr=1e-6):
    # thr of 1e-6 corresponds to around 0.01 THz and 0.3 cmm1
    nmod = len(om)
    nom = copy.deepcopy(om)
    neigv = copy.deepcopy(eigv)
    mask = np.where(np.abs(nom) > thr)
    nacoustic = nmod - len(mask[0])
    if nacoustic != 3:
        print("WARNING, n acoustic modes = ", nacoustic)
    nom = nom[mask]
    neigv = neigv[:, mask]
    neigv = neigv[:, 0, :]
    return nom, neigv


def remove_translations_from_mat(phi, thr=1e-6):
    # thr of 1e-6 corresponds to around 0.01 THz and 0.3 cmm1

    om, eigv = np.linalg.eigh(phi)
    mask = np.where(om < 0)
    om = np.abs(om)
    om = np.sqrt(om)
    om[mask] *= -1

    nmod = len(om)
    nom = copy.deepcopy(om)
    neigv = copy.deepcopy(eigv)
    mask = np.where(np.abs(nom) > thr)
    nacoustic = nmod - len(mask[0])
    if nacoustic != 3:
        print("WARNING, n acoustic modes = ", nacoustic)
    nom = nom[mask]
    neigv = neigv[:, mask]
    neigv = neigv[:, 0, :]
    return nom, neigv


def get_AB(fom, feigv, T):
    K_to_Ry = 6.336857346553283e-06

    om, eigv = remove_translations(fom, feigv)
    if T < 0.001:
        tanh = np.ones(len(om))
    else:
        arg = om / (T * K_to_Ry) / 2.0
        tanh = np.tanh(arg)

    lambd = 1 / tanh / (2 * om)
    A = np.einsum("s,is,js->ij", lambd, eigv, eigv)

    lambd = om / tanh / (2)
    B = np.einsum("s,is,js->ij", lambd, eigv, eigv)
    return A, B


def get_alpha(om, eigv, T):
    K_to_Ry = 6.336857346553283e-06

    if T < 0.001:
        tanh = np.ones(len(om))
    else:
        arg = om / (T * K_to_Ry) / 2.0
        tanh = np.tanh(arg)

    lambd = tanh * (2 * om)
    alpha = np.einsum("s,is,js->ij", lambd, eigv, eigv)

    return alpha


def inv_Phi(fom, feigv):
    om, eigv = remove_translations(fom, feigv)
    return np.einsum("k, ik, jk ->ij", 1 / om**2, eigv, eigv)


def displace_along_mode(mod, eigv, eta):
    eta = eta * 1.889725988 * np.sqrt(911.444175)
    return eigv[:, mod] * eta
