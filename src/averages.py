import numpy as np
import sys
import math
import torch


def force(R, A, phi, chi, psi):
    """
    Compute the total force acting on the nuclei from a 4th-order expansion
    of the potential energy surface.

    Parameters
    ----------
    R : ndarray (n,)
        Displacement vector from equilibrium positions.
    A : ndarray (n, n)
        Covariance (or quantum fluctuation) matrix.
    phi : ndarray (n, n)
        Harmonic (second-order) force-constant matrix.
    chi : ndarray (n, n, n)
        Third-order force-constant tensor.
    psi : ndarray (n, n, n, n)
        Fourth-order force-constant tensor.

    Returns
    -------
    ndarray (n,)
        Total force vector including harmonic, cubic, quartic,
        and quantum-correction terms.
    """

    f1 = np.einsum("ij,j->i", phi, R)
    f3 = 1 / 6 * np.einsum("ijkl,j,k,l->i", psi, R, R, R, optimize="optimal")
    fq3 = 1 / 2 * np.einsum("ijkl,j,kl->i", psi, R, A, optimize="optimal")

    f2 = 1 / 2 * np.einsum("ijk,j,k->i", chi, R, R, optimize="optimal")
    fq2 = 1 / 2 * np.einsum("ijk,jk->i", chi, A, optimize="optimal")
    return -f1 - f3 - fq3 - f2 - fq2


def torch_force(R, A, phi, chi, psi):
    """
    PyTorch implementation of `force`, enabling GPU execution.

    Parameters
    ----------
    R, A, phi, chi, psi : torch.Tensor
        Same physical quantities as in `force`, represented as tensors.

    Returns
    -------
    torch.Tensor
        Total force vector.
    """

    n = R.shape[0]
    psi_flat = psi.view(n, n, n * n)  # they are contiguous
    chi_flat = chi.view(n, n * n)
    A_flat = A.view(n * n)
    R_flat = torch.kron(R, R)

    f1 = torch.matmul(phi, R)
    f2 = 0.5 * torch.matmul(chi_flat, R_flat)
    fq2 = 0.5 * torch.matmul(chi_flat, A_flat)

    T = torch.matmul(psi_flat, R_flat)
    f3 = (1.0 / 6.0) * (T @ R)

    T = torch.matmul(psi_flat, A_flat)
    fq3 = (1.0 / 2.0) * (T @ R)

    return -(f1 + f2 + fq2 + f3 + fq3)


def kappa(R, A, phi, chi, psi):
    """
    Compute the effective force-constant matrix (curvature tensor).

    Parameters
    ----------
    R, A, phi, chi, psi : ndarray
        Same quantities as in `force`.

    Returns
    -------
    ndarray
        Effective force constants tensor including third- and fourth-order corrections and quantum effects.
    """

    k1 = 1 / 2 * np.einsum("ijkl, k,l->ij", psi, R, R, optimize="optimal")
    k2 = 1 / 2 * np.einsum("ijkl, kl->ij", psi, A, optimize="optimal")

    k3 = np.einsum("ijk,k->ij", chi, R)

    return phi + k1 + k2 + k3


def torch_kappa(R, A, phi, chi, psi):
    """
    PyTorch implementation of `kappa`, enabling GPU.

    Parameters
    ----------
    R, A, phi, chi, psi : torch.Tensor
        Same physical quantities as in `kappa`.

    Returns
    -------
    torch.Tensor
        Effective force-constants tensor.
    """

    n = R.shape[0]
    psi_flat = psi.view(n, n, n * n)
    R_flat = torch.kron(R, R)
    A_flat = A.view(-1)

    k1 = (1.0 / 2.0) * torch.matmul(psi_flat, R_flat)
    k2 = (1.0 / 2.0) * torch.matmul(psi_flat, A_flat)

    k3 = (chi.reshape(n * n, n) @ R).reshape(n, n)

    return phi + k1 + k2 + k3


def av_V(R, A, phi, chi, psi):
    """
    Compute the average potential energy for a Gaussian distribution
    centered at R with covariance A, up to fourth order in displacements.

    Parameters
    ----------
    R, A, phi, chi, psi : ndarray
        Displacement, covariance, and force-constant tensors.

    Returns
    -------
    float
        Expectation value of the quatnum potential energy.
    """

    V0 = 1 / 2 * np.einsum("i,j,ij", R, R, phi)
    V1 = 1 / 2 * np.einsum("ij,ij", A, phi)
    V2 = 1 / 24 * np.einsum("ijkl,i,j,k,l", psi, R, R, R, R, optimize="optimal")
    V3 = 1 / 4 * np.einsum("ijkl,i,j,kl", psi, R, R, A, optimize="optimal")
    V4 = 1 / 8 * np.einsum("ijkl,ij,kl", psi, A, A, optimize="optimal")

    V6 = 1 / 6 * np.einsum("ijk,i,j,k", chi, R, R, R, optimize="optimal")
    V7 = 1 / 2 * np.einsum("ijk,i,jk", chi, R, A, optimize="optimal")

    return V0 + V1 + V2 + V3 + V4 + V6 + V7


def torch_av_V(R, A, phi, chi, psi):
    """
    PyTorch implementation of `av_V`, computing the average potential
    energy using tensor operations.

    Parameters
    ----------
    R, A, phi, chi, psi : torch.Tensor

    Returns
    -------
    torch.Tensor
        Scalar potential energy.
    """

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
    """
    Compute the external driving force as a function of time.

    Parameters
    ----------
    t : float
        Time (in atomic units or fs converted accordingly).
    field : dict
        Dictionary describing the external field:
        - 'amp' : field amplitude (kV/cm)
        - 'freq' : field frequency (THz)
        - 'edir' : polarization direction (3-vector, normalized)
        - 't0' : pulse center (fs)
        - 'sig' : pulse width (fs)
        - 'type' : waveform ('sine', 'pulse', 'gaussian1', 'gaussian2', 'sinc')
        - 'Zeff' : effective charge tensor
        - 'eps' : dielectric constant

    Returns
    -------
    ndarray
        External force vector acting on all vibrational coordinates.
    """

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
    """
    PyTorch implementation of `ext_for`, supporting tensor operations.

    Parameters
    ----------
    t : float or torch.Tensor
        Time value(s).
    field : dict
        Same field dictionary as in `ext_for`, with `torch.Tensor` values
        for Zeff and edir.

    Returns
    -------
    torch.Tensor
        External force vector as a function of time.
    """

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
    """
    Compute the time-dependent force for trajectories R(t), A(t).

    Parameters
    ----------
    R : ndarray (nt, n)
        Time series of displacements.
    A : ndarray (nt, n, n)
        Time series of covariance matrices.
    phi, chi, psi : ndarray
        Harmonic, cubic, and quartic force-constant tensors.

    Returns
    -------
    ndarray (nt, n)
        Force vectors at each time step.
    """

    # now R and A are function of time, first index
    f1 = np.einsum("ij,tj->ti", phi, R)
    f3 = 1 / 6 * np.einsum("ijkl,tj,tk,tl->ti", psi, R, R, R, optimize="optimal")
    fq3 = 1 / 2 * np.einsum("ijkl,tj,tkl->ti", psi, R, A, optimize="optimal")

    f2 = 1 / 2 * np.einsum("ijk,tj,tk->ti", chi, R, R, optimize="optimal")
    fq2 = 1 / 2 * np.einsum("ijk,tjk->ti", chi, A)

    return -f1 - f3 - fq3 - f2 - fq2


def av_d3(R, chi, psi):
    """
    Compute the third-order derivative tensor averaged over displacements R.

    Parameters
    ----------
    R : ndarray (n,)
        Displacement vector.
    chi : ndarray (n, n, n)
        Third-order force constants.
    psi : ndarray (n, n, n, n)
        Fourth-order force constants.

    Returns
    -------
    ndarray (n, n, n)
        Averaged third-order tensor.
    """

    d3 = np.einsum("ijkl,l->ijk", psi, R)
    d3 = d3 + chi
    return d3


def V_classic(R, phi, chi, psi):
    """
    Evaluate the potential energy at configuration R using the classical
    Taylor expansion up to fourth order.

    Parameters
    ----------
    R, phi, chi, psi : ndarray

    Returns
    -------
    float
        Classical potential energy value.
    """

    V2 = 1 / 2 * np.einsum("ij,i,j", phi, R, R)
    V3 = 1 / 6 * np.einsum("ijk,i,j,k", chi, R, R, R, optimize="optimal")
    V4 = 1 / 24 * np.einsum("ijkl,i,j,k,l", psi, R, R, R, R, optimize="optimal")

    return V2 + V3 + V4


def f_classic(R, phi, chi, psi):
    """
    Compute the classical force from the Taylor-expanded potential
    up to quartic order.

    Parameters
    ----------
    R, phi, chi, psi : ndarray

    Returns
    -------
    ndarray
        Classical force vector.
    """

    f1 = np.einsum("ij,j->i", phi, R)
    f3 = 1 / 6 * np.einsum("ijkl,j,k,l->i", psi, R, R, R)
    f2 = 1 / 2 * np.einsum("ijk,j,k->i", chi, R, R)

    return -f1 - f3 - f2


def kappa_t(R, A, phi, chi, psi):
    """
    Time-dependent effective force-constants tensor kappa(t), computed for
    time series of displacements and covariances.

    Parameters
    ----------
    R : ndarray (nt, n)
    A : ndarray (nt, n, n)
    phi, chi, psi : ndarray

    Returns
    -------
    ndarray (nt, n, n)
        Effective force-constants tensors at each time.
    """

    k1 = 1 / 2 * np.einsum("ijkl, tk,tl->tij", psi, R, R, optimize="optimal")
    k2 = 1 / 2 * np.einsum("ijkl, tkl->tij", psi, A, optimize="optimal")

    k3 = np.einsum("ijk,tk->tij", chi, R, optimize="optimal")

    return phi + k1 + k2 + k3


def d2V(R, phi, chi, psi):
    """
    Compute the instantaneous second derivative of the potential energy
    (Hessian) at configuration R.

    Parameters
    ----------
    R : ndarray (n,)
    phi, chi, psi : ndarray

    Returns
    -------
    ndarray (n, n)
        Second-derivative (Hessian) matrix.
    """

    k1 = 1 / 2 * np.einsum("ijkl, k,l->ij", psi, R, R)
    k2 = np.einsum("ijk,k->ij", chi, R)
    return phi + k1 + k2



def av_V_t(R, A, phi, chi, psi):
    """
    Compute the time-dependent average potential energy ⟨V(t)⟩ for
    trajectories of displacements R(t) and covariances A(t).

    Parameters
    ----------
    R : ndarray (nt, n)
    A : ndarray (nt, n, n)
    phi, chi, psi : ndarray

    Returns
    -------
    ndarray (nt,)
        Average potential energy at each time step.
    """

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
