import numpy as np
import sys
import scipy
import copy
import os


def get_phonons_THz(phi):
    """
    Diagonalize a harmonic force-constant matrix and return phonon frequencies in THz.

    Parameters
    ----------
    phi : (3N, 3N) ndarray
        Harmonic force-constant matrix in atomic units (Ry/Bohr²).

    Returns
    -------
    om : (3N,) ndarray
        Phonon frequencies in terahertz (THz).
    eigv : (3N, 3N) ndarray
        Normal-mode eigenvectors (columns).
    """

    om, eigv = np.linalg.eigh(phi)
    om = np.sqrt(om) * 13.6 * 241.8
    return om, eigv


def get_phonons(phi):
    """
    Diagonalize a harmonic force-constant matrix and return phonon frequencies
    in internal atomic-unit scale (√Ry/Bohr).

    Parameters
    ----------
    phi : (3N, 3N) ndarray
        Harmonic force-constant matrix.

    Returns
    -------
    om : (3N,) ndarray
        Square-root eigenvalues of phi (frequencies in a.u.).
    eigv : (3N, 3N) ndarray
        Eigenvectors (columns).
    """
    om, eigv = np.linalg.eigh(phi)
    om = np.sqrt(om)
    return om, eigv


def get_phonons_r(phi):
    """
    Diagonalize the force-constant matrix using scipy.linalg.eigh and return
    positive phonon frequencies (absolute value).

    Parameters
    ----------
    phi : (3N, 3N) ndarray
        Harmonic force-constant matrix.

    Returns
    -------
    om : (3N,) ndarray
        Positive phonon frequencies (a.u.).
    eigv : (3N, 3N) ndarray
        Eigenvectors (columns).
    """

    om, eigv = scipy.linalg.eigh(phi)
    om = np.abs(om)
    om = np.sqrt(om)
    return om, eigv


def regularize(phi):
    """
    Replace negative eigenvalues of a force-constant matrix by their absolute
    values to enforce positive definiteness.

    Parameters
    ----------
    phi : (3N, 3N) ndarray
        Harmonic force-constant matrix (possibly with small negative modes).

    Returns
    -------
    phi_reg : (3N, 3N) ndarray
        Regularized force-constant matrix with all positive eigenvalues.
    """

    om, eigv = scipy.linalg.eigh(phi)
    om = np.abs(om)

    phi = np.einsum("s, is, js -> ij", om, eigv, eigv)
    return phi


def print_phonons(om):
    """
    Print phonon frequencies to standard output in THz.

    Parameters
    ----------
    om : (3N,) ndarray
        Frequencies in internal units (a.u.); converted to THz before printing.
    """

    print("phonons")
    for i in range(len(om)):
        print("Mode %d" % (i + 1), om[i] * 241.8 * 13.6)


def print_phonons_mat(phi):
    """
    Diagonalize a force-constant matrix and print phonon frequencies both in
    THz and cm⁻¹ units.

    Parameters
    ----------
    phi : (3N, 3N) ndarray
        Harmonic force-constant matrix.
    """

    om, eigv = np.linalg.eigh(phi)
    mask = np.where(om < 0)
    om = np.abs(om)
    om = np.sqrt(om)
    om[mask] *= -1
    print("phonons")
    for i in range(len(om)):
        print("Mode %d" % (i + 1), om[i] * 241.8 * 13.6, "THz", om[i] * 8065.54429 * 13.6, "cmm1")


def remove_translations(om, eigv, thr=1e-6):
    """
    Remove translational acoustic modes (ω≈0) from a phonon spectrum.

    Parameters
    ----------
    om : (3N,) ndarray
        Phonon frequencies (a.u.).
    eigv : (3N, 3N) ndarray
        Corresponding eigenvectors (columns).
    thr : float, optional
        Threshold below which a mode is considered acoustic. Default 1e-6.

    Returns
    -------
    nom : (3N-3,) ndarray
        Non-acoustic frequencies.
    neigv : (3N, 3N-3) ndarray
        Eigenvectors of optical modes.
    """

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
    """
    Remove translational acoustic modes directly from a force-constant matrix.

    Parameters
    ----------
    phi : (3N, 3N) ndarray
        Harmonic force-constant matrix.
    thr : float, optional
        Threshold below which eigenmodes are treated as acoustic. Default 1e-6.

    Returns
    -------
    nom : (3N-3,) ndarray
        Non-acoustic frequencies.
    neigv : (3N, 3N-3) ndarray
        Corresponding eigenvectors.
    """

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
    """
    Build Gaussian covariance matrices A and B for a harmonic system at temperature T.

    Parameters
    ----------
    fom : (3N,) ndarray
        Phonon frequencies (a.u.).
    feigv : (3N, 3N) ndarray
        Normal-mode eigenvectors (columns).
    T : float
        Temperature in kelvin.

    Returns
    -------
    A : (3N, 3N) ndarray
        Coordinate covariance matrix ⟨u_i u_j⟩.
    B : (3N, 3N) ndarray
        Momentum covariance matrix ⟨p_i p_j⟩.

    Notes
    -----
    - Converts T to Rydberg atomic units (K_to_Ry = 6.336857×10⁻⁶).
    - Acoustic modes (ω<1e-6) are removed before computing A and B.
    """

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
    """
    Compute α matrix = 2 ω tanh(ω / 2T) projected to Cartesian coordinates.

    Parameters
    ----------
    om : (3N,) ndarray
        Phonon frequencies (a.u.).
    eigv : (3N, 3N) ndarray
        Normal-mode eigenvectors.
    T : float
        Temperature in kelvin.

    Returns
    -------
    alpha : (3N, 3N) ndarray
        Dynamical matrix α in Cartesian coordinates.
    """

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
    """
    Construct the inverse of the harmonic force-constant matrix using
    normal-mode decomposition.

    Parameters
    ----------
    fom : (3N,) ndarray
        Phonon frequencies (a.u.).
    feigv : (3N, 3N) ndarray
        Corresponding eigenvectors.

    Returns
    -------
    inv_phi : (3N, 3N) ndarray
        Matrix ∑_k |e_k⟩⟨e_k| / ω_k² excluding acoustic modes.
    """

    om, eigv = remove_translations(fom, feigv)
    return np.einsum("k, ik, jk ->ij", 1 / om**2, eigv, eigv)


def displace_along_mode(mod, eigv, eta):
    """
    Build a Cartesian displacement vector corresponding to a single normal mode.

    Parameters
    ----------
    mod : int
        Mode index (0-based).
    eigv : (3N, 3N) ndarray
        Normal-mode eigenvectors (columns).
    eta : float
        Amplitude of displacement in Å.

    Returns
    -------
    disp : (3N,) ndarray
        Mass-weighted Cartesian displacement vector.

    Notes
    -----
    - Converts η from Å to Bohr·√amu using 1 Å = 1.889725988 Bohr and
      1 amu = 911.444175 m_e.
    """

    eta = eta * 1.889725988 * np.sqrt(911.444175)
    return eigv[:, mod] * eta
