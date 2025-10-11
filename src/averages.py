import numpy as np
import sys
import math
import torch


def force(R, A, phi, chi, psi):

    f1 = np.einsum("ij,j->i", phi, R)
    f3 = 1 / 6 * np.einsum("ijkl,j,k,l->i", psi, R, R, R, optimize="optimal")
    fq3 = 1 / 2 * np.einsum("ijkl,j,kl->i", psi, R, A, optimize="optimal")

    f2 = 1 / 2 * np.einsum("ijk,j,k->i", chi, R, R, optimize="optimal")
    fq2 = 1 / 2 * np.einsum("ijk,jk->i", chi, A, optimize="optimal")
    return -f1 - f3 - fq3 - f2 - fq2


def torch_force(R, A, phi, chi, psi):

    n = R.shape[0]
    psi_flat = psi.view(n, n, n * n)  # they are contiguous
    chi_flat = chi.view(n, n * n)
    A_flat = A.view(n * n)
    R_flat = torch.kron(R, R)

    # Efficient replacement of einsum for psi, explotis contiguity

    f1 = torch.matmul(phi, R)
    f2 = 0.5 * torch.matmul(chi_flat, R_flat)
    fq2 = 0.5 * torch.matmul(chi_flat, A_flat)

    T = torch.matmul(psi_flat, R_flat)
    f3 = (1.0 / 6.0) * (T @ R)

    T = torch.matmul(psi_flat, A_flat)
    fq3 = (1.0 / 2.0) * (T @ R)

    return -(f1 + f2 + fq2 + f3 + fq3)


def kappa(R, A, phi, chi, psi):
    k1 = 1 / 2 * np.einsum("ijkl, k,l->ij", psi, R, R, optimize="optimal")
    k2 = 1 / 2 * np.einsum("ijkl, kl->ij", psi, A, optimize="optimal")

    k3 = np.einsum("ijk,k->ij", chi, R)

    return phi + k1 + k2 + k3


def torch_kappa(R, A, phi, chi, psi):

    n = R.shape[0]
    psi_flat = psi.view(n, n, n * n)
    R_flat = torch.kron(R, R)
    A_flat = A.view(-1)

    k1 = (1.0 / 2.0) * torch.matmul(psi_flat, R_flat)
    k2 = (1.0 / 2.0) * torch.matmul(psi_flat, A_flat)

    k3 = np.einsum("ijk,k->ij", chi, R)

    return phi + k1 + k2 + k3


def av_V(R, A, phi, chi, psi):

    V0 = 1 / 2 * np.einsum("i,j,ij", R, R, phi)
    V1 = 1 / 2 * np.einsum("ij,ij", A, phi)
    V2 = 1 / 24 * np.einsum("ijkl,i,j,k,l", psi, R, R, R, R, optimize="optimal")
    V3 = 1 / 4 * np.einsum("ijkl,i,j,kl", psi, R, R, A, optimize="optimal")
    V4 = 1 / 8 * np.einsum("ijkl,ij,kl", psi, A, A, optimize="optimal")

    V6 = 1 / 6 * np.einsum("ijk,i,j,k", chi, R, R, R, optimize="optimal")
    V7 = 1 / 2 * np.einsum("ijk,i,jk", chi, R, A, optimize="optimal")

    return V0 + V1 + V2 + V3 + V4 + V6 + V7


def torch_av_V(R, A, phi, chi, psi):

    V0 = (1.0 / 2.0) * (R @ (phi @ R))
    V1 = (1.0 / 2.0) * torch.sum(phi * A)

    t = torch.tensordot(psi, R, dims=([3], [0]))
    t_saved = torch.tensordot(t, R, dims=([2], [0]))
    t = torch.tensordot(t_saved, R, dims=([1], [0]))
    V2 = (1.0 / 24.0) * torch.dot(t, R)
    V3 = (1.0 / 4.0) * torch.sum(t_saved * A)

    t = torch.tensordot(psi, A, dims=([2, 3], [0, 1]))
    V4 = (1.0 / 8.0) * torch.sum(t * A)

    t_saved = torch.tensordot(chi, R, dims=([2], [0]))
    t = torch.tensordot(t_saved, R, dims=([1], [0]))
    V6 = (1.0 / 6.0) * torch.dot(t, R)
    V7 = (1.0 / 2.0) * torch.sum(t_saved * A)

    return V0 + V1 + V2 + V3 + V4 + V6 + V7


def ext_for(t, field):

    Eamp = field["amp"]
    om_L = field["freq"]
    edir = field["edir"]
    t0 = field["t0"]
    sig = field["sig"]
    case = field["type"]
    Zeff = field["Zeff"]
    eps = field["eps"]

    if np.abs(np.linalg.norm(edir) - 1) > 1e-7:
        sys.exit("Direction not normalized")

    Eeff = Eamp * 2.7502067 * 1e-7 * 2 / (1 + np.sqrt(eps))
    freq = om_L / (2.0670687 * 1e4)

    nmod = len(Zeff)
    nat = int(nmod / 3)

    force = []
    for i in range(nat):
        force.append(np.dot(Zeff[3 * i : 3 * i + 3, :], edir) * Eeff * np.sqrt(2))
    force = np.array(force)
    force = np.reshape(force, nmod)
    # force = force / np.sqrt(masses)  EFFECTIVE CHARGES ARE ALREADY RESCALED FOR MASSES

    if case == "sine":
        return force * np.sin(2 * np.pi * freq * t)
    elif case == "gaussian1":
        t0 = t0 / (4.8377687 * 1e-2)
        sig = 1 / (2 * np.pi * freq)
        return -force * (t - t0) / sig * np.exp(-0.5 * (t - t0) ** 2 / sig**2 + 0.5)
    elif case == "pulse":
        t0 = t0 / (4.8377687 * 1e-2)
        sig = sig / (4.8377687 * 1e-2)
        return -force * np.cos(2 * np.pi * freq * (t - t0)) * np.exp(-0.5 * (t - t0) ** 2 / sig**2)
    elif case == "gaussian2":
        t0 = t0 / (4.8377687 * 1e-2)
        sig = 1 / (np.sqrt(2) * np.pi * freq)
        return -force * (1 - (t - t0) ** 2 / sig**2) * np.exp(-0.5 * (t - t0) ** 2 / sig**2)
    elif case == "sinc":
        return -force * np.sinc(t * freq * 2)
    else:
        sys.exit("Field not implemented")


def torch_ext_for(t, field):

    # Extract field parameters
    Eamp = field["amp"]
    om_L = field["freq"]
    edir = field["edir"]
    t0 = field["t0"]
    sig = field["sig"]
    case = field["type"]
    Zeff = field["Zeff"]
    eps = field["eps"]

    device, dtype = Zeff.device, Zeff.dtype

    # Normalization check
    if torch.abs(torch.linalg.norm(edir) - 1) > 1e-7:
        raise ValueError("Direction not normalized")

    Eeff = Eamp * 2.7502067e-7 * 2.0 / (1.0 + math.sqrt(eps))
    freq = om_L / 2.0670687e4

    nmod = Zeff.shape[0]
    nat = nmod // 3

    root2 = math.sqrt(2.0)
    force_blocks = []
    for i in range(nat):
        blk = Zeff[3 * i : 3 * i + 3, :]
        proj = blk.transpose(0, 1) @ edir
        force_blocks.append(proj * Eeff * root2)
    force = torch.cat(force_blocks, dim=0).reshape(nmod)

    # Ensure time is a tensor
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=device, dtype=dtype)
    else:
        t = t.to(device=device, dtype=dtype)

    # Unit conversion constant
    conv = 4.8377687e-2
    two_pi = 2 * math.pi

    # Switch over waveform type
    if case == "sine":
        return force * torch.sin(two_pi * freq * t)

    elif case == "gaussian1":
        t0_ = t0 / conv
        sig_ = 1.0 / (two_pi * freq)
        return -force * (t - t0_) / sig_ * torch.exp(-0.5 * (t - t0_) ** 2 / sig_**2 + 0.5)

    elif case == "pulse":
        t0_ = t0 / conv
        sig_ = sig / conv
        return (
            -force
            * torch.cos(two_pi * freq * (t - t0_))
            * torch.exp(-0.5 * (t - t0_) ** 2 / sig_**2)
        )

    elif case == "gaussian2":
        t0_ = t0 / conv
        sig_ = 1.0 / (math.sqrt(2.0) * math.pi * freq)
        return (
            -force * (1.0 - (t - t0_) ** 2 / sig_**2) * torch.exp(-0.5 * (t - t0_) ** 2 / sig_**2)
        )

    elif case == "sinc":
        return -force * torch.sinc(t * freq * 2.0)

    else:
        raise ValueError("Field not implemented")


def force_t(R, A, phi, chi, psi):
    # now R and A are function of time, first index
    f1 = np.einsum("ij,tj->ti", phi, R)
    f3 = 1 / 6 * np.einsum("ijkl,tj,tk,tl->ti", psi, R, R, R, optimize="optimal")
    fq3 = 1 / 2 * np.einsum("ijkl,tj,tkl->ti", psi, R, A, optimize="optimal")

    f2 = 1 / 2 * np.einsum("ijk,tj,tk->ti", chi, R, R, optimize="optimal")
    fq2 = 1 / 2 * np.einsum("ijk,tjk->ti", chi, A)

    return -f1 - f3 - fq3 - f2 - fq2


def av_d3(R, chi, psi):
    d3 = np.einsum("ijkl,l->ijk", psi, R)
    d3 = d3 + chi
    return d3


def V_classic(R, phi, chi, psi):
    V2 = 1 / 2 * np.einsum("ij,i,j", phi, R, R)
    V3 = 1 / 6 * np.einsum("ijk,i,j,k", chi, R, R, R, optimize="optimal")
    V4 = 1 / 24 * np.einsum("ijkl,i,j,k,l", psi, R, R, R, R, optimize="optimal")

    return V2 + V3 + V4


def f_classic(R, phi, chi, psi):
    f1 = np.einsum("ij,j->i", phi, R)
    f3 = 1 / 6 * np.einsum("ijkl,j,k,l->i", psi, R, R, R)
    f2 = 1 / 2 * np.einsum("ijk,j,k->i", chi, R, R)

    return -f1 - f3 - f2


def kappa_t(R, A, phi, chi, psi):
    k1 = 1 / 2 * np.einsum("ijkl, tk,tl->tij", psi, R, R, optimize="optimal")
    k2 = 1 / 2 * np.einsum("ijkl, tkl->tij", psi, A, optimize="optimal")

    k3 = np.einsum("ijk,tk->tij", chi, R, optimize="optimal")

    return phi + k1 + k2 + k3


def d2V(R, phi, chi, psi):
    k1 = 1 / 2 * np.einsum("ijkl, k,l->ij", psi, R, R)
    k2 = np.einsum("ijk,k->ij", chi, R)
    return phi + k1 + k2


def av_V(R, A, phi, chi, psi):

    V0 = 1 / 2 * np.einsum("i,j,ij", R, R, phi)
    V1 = 1 / 2 * np.einsum("ij,ij", A, phi)
    V2 = 1 / 24 * np.einsum("ijkl,i,j,k,l", psi, R, R, R, R, optimize="optimal")
    V3 = 1 / 4 * np.einsum("ijkl,i,j,kl", psi, R, R, A, optimize="optimal")
    V4 = 1 / 8 * np.einsum("ijkl,ij,kl", psi, A, A, optimize="optimal")

    # Q = np.einsum('s, is,js,ks,ls -> ijkl', lamb**2, vect, vect, vect, vect)
    # V5 = 1/8*np.einsum('ijkl,ijkl', psi, Q)
    # V5 = 1/8*np.einsum('ijkl,im,jm,km,lm,m', psi, vect, vect, vect, vect, lamb**2, optimize= 'optimal')

    V6 = 1 / 6 * np.einsum("ijk,i,j,k", chi, R, R, R, optimize="optimal")
    V7 = 1 / 2 * np.einsum("ijk,i,jk", chi, R, A, optimize="optimal")

    return V0 + V1 + V2 + V3 + V4 + V6 + V7


def av_V_t(R, A, phi, chi, psi):

    V0 = 1 / 2 * np.einsum("ti,tj,ij->t", R, R, phi, optimize="optimal")
    V1 = 1 / 2 * np.einsum("tij,ij->t", A, phi, optimize="optimal")
    V2 = 1 / 24 * np.einsum("ijkl,ti,tj,tk,tl->t", psi, R, R, R, R, optimize="optimal")
    V3 = 1 / 4 * np.einsum("ijkl,ti,tj,tkl->t", psi, R, R, A, optimize="optimal")
    V4 = 1 / 8 * np.einsum("ijkl,tij,tkl->t", psi, A, A, optimize="optimal")

    # Q = np.einsum('s, is,js,ks,ls -> ijkl', lamb**2, vect, vect, vect, vect)
    # V5 = 1/8*np.einsum('ijkl,ijkl', psi, Q)
    # V5 = 1/8*np.einsum('ijkl,im,jm,km,lm,m', psi, vect, vect, vect, vect, lamb**2, optimize= 'optimal')

    V6 = 1 / 6 * np.einsum("ijk,ti,tj,tk->t", chi, R, R, R, optimize="optimal")
    V7 = 1 / 2 * np.einsum("ijk,ti,tjk->t", chi, R, A, optimize="optimal")

    return V0 + V1 + V2 + V3 + V4 + V6 + V7
