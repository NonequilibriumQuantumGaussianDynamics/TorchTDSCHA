import numpy as np
import copy
import torch

# import h5py
from .phonons import get_AB

from ase.io.vasp import read_vasp
import cellconstructor as CC, cellconstructor.Phonons


def load_from_sscha(dyn_file, path, T, new_format=False, path_diff=""):
    """
    Load a TDSCHA model (φ, χ, ψ, R, P, masses, A, B, C) from SSCHA/CellConstructor outputs,
    convert to atomic units, and mass-rescale tensors.

    Parameters
    ----------
    dyn_file : str
        Path/prefix to the CellConstructor dynamical matrix (e.g. "final_result").
    path : str
        Directory containing SPOSCAR and (optionally) fc{2,3,4}.hdf5.
    T : float
        Temperature (K) used to build Gaussian widths A, B via normal modes.
    new_format : bool, optional
        If True, read φ/χ/ψ from NumPy files in `path_diff` (phi.npy, chi.npy, psi.npy).
        If False, read fc2/3/4 from HDF5 in `path`.
    path_diff : str, optional
        Folder with phi.npy/chi.npy/psi.npy when `new_format=True`.

    Returns
    -------
    nat : int
        Number of atoms.
    nmod : int
        Number of Cartesian DOF (3*nat).
    phi : (nmod, nmod) ndarray
        Mass-rescaled harmonic force constants in Ry/Bohr².
    chi : (nmod, nmod, nmod) ndarray
        Mass-rescaled third-order FCs in Ry/Bohr³.
    psi : (nmod, nmod, nmod, nmod) ndarray
        Mass-rescaled fourth-order FCs in Ry/Bohr⁴.
    R : (nmod,) ndarray
        Mass-weighted displacements (Bohr·sqrt(amu)).
    P : (nmod,) ndarray
        Initial conjugate momenta (zeros).
    masses : (nmod,) ndarray
        Cartesian mass array (amu) repeated per component.
    A, B : (nmod, nmod) ndarray
        Gaussian covariance/width matrices from normal modes at temperature T.
    C : (nmod, nmod) ndarray
        Initial mixed covariance (zeros).

    Notes
    -----
    - Converts units: Å→Bohr, eV→Ry, and applies 1/sqrt(mass) scaling to FC tensors.
    - Uses `dyn.DiagonalizeSupercell()` to obtain eigenvalues/eigenvectors,
      then constructs A and B via `get_AB(om, eigv, T)`.
    """

    Ry_to_eV = 13.60570397
    uma_to_Ry = 911.444175
    A_to_B = 1.889725988

    dyn = CC.Phonons.Phonons(dyn_file)

    masses = dyn.structure.get_masses_array()
    masses = np.repeat(masses, 3)

    positions = dyn.structure.coords
    atoms = read_vasp(path + "/SPOSCAR")

    nat = len(positions)
    nmod = 3 * nat
    R = np.reshape(positions - atoms.positions, nmod) * A_to_B
    sqrtm = np.sqrt(masses)
    sqrtm_inv = 1 / sqrtm
    R *= sqrtm

    if not new_format:
        phi = read_phi(path)
        phi = phi / Ry_to_eV / A_to_B**2
        phi *= sqrtm_inv[:, None]
        phi *= sqrtm_inv[None, :]

        chi = read_chi(path)
        chi = chi / Ry_to_eV / A_to_B**3
        chi *= sqrtm_inv[:, None, None]
        chi *= sqrtm_inv[None, :, None]
        chi *= sqrtm_inv[None, None, :]

        psi = read_psi(path)
        psi = psi / Ry_to_eV / A_to_B**4
        psi *= sqrtm_inv[:, None, None, None]
        psi *= sqrtm_inv[None, :, None, None]
        psi *= sqrtm_inv[None, None, :, None]
        psi *= sqrtm_inv[None, None, None, :]

    else:
        phi = np.load(path_diff + "/phi.npy")
        phi *= sqrtm_inv[:, None]
        phi *= sqrtm_inv[None, :]

        chi = np.load(path_diff + "/chi.npy")
        chi *= sqrtm_inv[:, None, None]
        chi *= sqrtm_inv[None, :, None]
        chi *= sqrtm_inv[None, None, :]

        psi = np.load(path_diff + "/psi.npy")
        psi *= sqrtm_inv[:, None, None, None]
        psi *= sqrtm_inv[None, :, None, None]
        psi *= sqrtm_inv[None, None, :, None]
        psi *= sqrtm_inv[None, None, None, :]

    P = np.zeros(nmod)
    C = np.zeros((nmod, nmod))

    om, eigv = dyn.DiagonalizeSupercell()

    A, B = get_AB(om, eigv, T)

    return nat, nmod, phi, chi, psi, R, P, masses, A, B, C


def read_phi(path):
    """
    Read and reshape harmonic force constants (fc2.hdf5) to a (3N, 3N) matrix.

    Parameters
    ----------
    path : str
        Directory containing 'fc2.hdf5' with dataset 'fc2' of shape (N, 3, N, 3).

    Returns
    -------
    phi : (3N, 3N) ndarray
        Harmonic force constants with indices reordered to Cartesian-major layout.
    """

    f2 = h5py.File(path + "/fc2.hdf5", "r")
    fc2 = f2["fc2"]
    nat = np.shape(fc2)[0]

    newfc2 = np.reshape(np.transpose(fc2, [0, 2, 1, 3]), (3 * nat, 3 * nat))
    return newfc2


def read_chi(path):
    """
    Read and reshape third-order force constants (fc3.hdf5) to a (3N, 3N, 3N) tensor.

    Parameters
    ----------
    path : str
        Directory containing 'fc3.hdf5' with dataset 'fc3' of shape (N, 3, N, 3, N, 3).

    Returns
    -------
    chi : (3N, 3N, 3N) ndarray
        Third-order force constants in Cartesian-major layout.
    """

    f3 = h5py.File(path + "/fc3.hdf5", "r")
    fc3 = f3["fc3"]
    nat = np.shape(fc3)[0]

    newfc3 = np.reshape(np.transpose(fc3, [0, 3, 1, 4, 2, 5]), (3 * nat, 3 * nat, 3 * nat))
    return newfc3


def read_psi(path):
    """
    Read and reshape fourth-order force constants (fc4.hdf5) to a (3N, 3N, 3N, 3N) tensor.

    Parameters
    ----------
    path : str
        Directory containing 'fc4.hdf5' with dataset 'fc4' of shape (N, 3, N, 3, N, 3, N, 3).

    Returns
    -------
    psi : (3N, 3N, 3N, 3N) ndarray
        Fourth-order force constants in Cartesian-major layout.
    """

    f4 = h5py.File(path + "/fc4.hdf5", "r")
    fc4 = f4["fc4"]
    nat = np.shape(fc4)[0]

    newfc4 = np.transpose(fc4, [0, 4, 1, 5, 2, 6, 3, 7])
    newfc4 = np.reshape(newfc4, (3 * nat, 3 * nat, 3 * nat, 3 * nat))
    return newfc4


def reduce_model(modes, nmod, phi, chi, psi, R, P, A, B, C, eigv, Zeff):
    """
    Project the model onto a subset of normal modes and return reduced tensors/state.

    Parameters
    ----------
    modes : list[int]
        1-based indices of modes to keep.
    nmod : int
        Total number of modes (3N).
    phi, chi, psi : ndarray
        Force-constant tensors in Cartesian coordinates.
    R, P : (nmod,) ndarray
        Mass-weighted displacements and momenta.
    A, B, C : (nmod, nmod) ndarray
        Gaussian covariance/width/mixed matrices.
    eigv : (nmod, nmod) ndarray
        Normal-mode eigenvectors (columns).
    Zeff : (nmod, 3) ndarray
        Mass-weighted Born effective charges in Cartesian basis.

    Returns
    -------
    nmod_red : int
        Reduced number of modes.
    phi_mu, chi_mu, psi_mu : ndarray
        Tensors projected and sliced to the selected modes.
    R_mu, P_mu : (nmod_red,) ndarray
        State vectors in the reduced modal basis.
    A_mu, B_mu, C_mu : (nmod_red, nmod_red) ndarray
        Reduced Gaussian matrices.
    Zeff_mu : (nmod_red, 3) ndarray
        Effective charges in the reduced modal basis.

    Notes
    -----
    - Uses eigenvector projections to go to modal basis, slices the selected
      modes, then returns reduced data still in modal coordinates.
    """

    print("Reducing model to ", modes)
    modes = [m - 1 for m in modes]

    phi_mu = np.einsum("ij,im,jn->mn", phi, eigv, eigv)
    chi_mu = np.einsum("ijk,im,jn,kp->mnp", chi, eigv, eigv, eigv, optimize="optimal")
    psi_mu = np.einsum("ijkl,im,jn,kp,lq->mnpq", psi, eigv, eigv, eigv, eigv, optimize="optimal")

    phi_mu = phi_mu[np.ix_(modes, modes)]
    chi_mu = chi_mu[np.ix_(modes, modes, modes)]
    psi_mu = psi_mu[np.ix_(modes, modes, modes, modes)]

    R_mu = np.einsum("i,is->s", R, eigv)
    P_mu = np.einsum("i,is->s", P, eigv)
    A_mu = np.einsum("ij,is,jt->st", A, eigv, eigv)
    B_mu = np.einsum("ij,is,jt->st", B, eigv, eigv)
    C_mu = np.einsum("ij,is,jt->st", C, eigv, eigv)

    R_mu = R_mu[np.ix_(modes)]
    P_mu = P_mu[np.ix_(modes)]
    A_mu = A_mu[np.ix_(modes, modes)]
    B_mu = B_mu[np.ix_(modes, modes)]
    C_mu = C_mu[np.ix_(modes, modes)]

    Zeff_mu = np.einsum("is, ij->sj", eigv, Zeff)
    Zeff_mu = Zeff_mu[modes, :]

    nmod = len(modes)

    return nmod, phi_mu, chi_mu, psi_mu, R_mu, P_mu, A_mu, B_mu, C_mu, Zeff_mu


def isolate_couplings(modes, phi, chi, psi, eigv, exclude_diag=[]):
    """
    Zero out all couplings except those involving a chosen set of modes.

    Parameters
    ----------
    modes : list[int]
        1-based indices of modes whose intra-set couplings are retained.
    phi, chi, psi : ndarray
        FC tensors in Cartesian basis.
    eigv : (nmod, nmod) ndarray
        Normal-mode eigenvectors (columns).
    exclude_diag : list[int], optional
        1-based mode indices whose pure self-terms (ii, iii, iiii) are set to zero.

    Returns
    -------
    phi_new, chi_new, psi_new : ndarray
        Modified FC tensors transformed back to Cartesian basis.

    Notes
    -----
    - Transforms FCs to modal basis, masks out-of-set couplings, optionally
      removes diagonal self-terms, and transforms back to Cartesian coordinates.
    """

    print("Reducing to ", modes)
    modes = [m - 1 for m in modes]

    phi_mu = np.einsum("ij,im,jn->mn", phi, eigv, eigv)
    chi_mu = np.einsum("ijk,im,jn,kp->mnp", chi, eigv, eigv, eigv, optimize="optimal")
    psi_mu = np.einsum("ijkl,im,jn,kp,lq->mnpq", psi, eigv, eigv, eigv, eigv, optimize="optimal")

    shape = phi_mu.shape
    mask = np.zeros(shape, dtype=bool)
    idx = np.ix_(modes, modes)
    mask[idx] = True
    phi_mu[~mask] = 0

    shape = chi_mu.shape
    mask = np.zeros(shape, dtype=bool)
    idx = np.ix_(modes, modes, modes)
    mask[idx] = True
    chi_mu[~mask] = 0

    shape = psi_mu.shape
    mask = np.zeros(shape, dtype=bool)
    idx = np.ix_(modes, modes, modes, modes)
    mask[idx] = True
    psi_mu[~mask] = 0

    for mod in exclude_diag:
        s = mod - 1
        print(phi_mu[s, s])
        phi_mu[s, s] = 0
        print(chi_mu[s, s, s])
        chi_mu[s, s, s] = 0
        print(psi_mu[s, s, s, s])
        psi_mu[s, s, s, s] = 0

    phi = np.einsum("mn,im,jn->ij", phi_mu, eigv, eigv, optimize="optimal")
    chi = np.einsum("mnp,im,jn,kp->ijk", chi_mu, eigv, eigv, eigv, optimize="optimal")
    psi = np.einsum("mnpq,im,jn,kp,lq->ijkl", psi_mu, eigv, eigv, eigv, eigv, optimize="optimal")

    return phi, chi, psi


def continue_evolution(fil):
    """
    Load a saved evolution chunk and return the final state as new initial conditions.

    Parameters
    ----------
    fil : str
        Path to a saved solution file ('.npz') produced by the evolution routines.

    Returns
    -------
    R, P, A, B, C : ndarray
        Last-time-step state reconstructed from the saved solution.
    newlabel : str
        Suggested label for the continuation output (original label + 'cont').
    tfin : float
        Final physical time of the loaded chunk (in fs).

    Notes
    -----
    - Infers number of modes from the flattened state size.
    - Converts back to matrices for A, B, C.
    """

    sol = np.load(fil)["arr_0"]
    tfin = sol[-1, 0] * (4.8377687 * 1e-2)

    sol = sol[:, 1:]

    N, n1 = np.shape(sol)
    nmod = int((-2 + np.sqrt(4 + 12 * n1)) / 6)

    R = sol[-1, :nmod]
    P = sol[-1, nmod : 2 * nmod]
    A = sol[-1, 2 * nmod : 2 * nmod + nmod**2]
    A = np.reshape(A, (nmod, nmod))
    B = sol[-1, 2 * nmod + nmod**2 : 2 * nmod + 2 * nmod**2]
    B = np.reshape(B, (nmod, nmod))
    C = sol[-1, 2 * nmod + 2 * nmod**2 : 2 * nmod + 3 * nmod**2]
    C = np.reshape(C, (nmod, nmod))

    newlabel = fil[:-5] + "cont"
    return R, P, A, B, C, newlabel, tfin


def merge_evolutions(fil, fil1, fil2=""):
    """
    Concatenate two (or three) time-evolution chunks into a single array.

    Parameters
    ----------
    fil : str
        First saved solution ('.npz') with key 'arr_0'.
    fil1 : str
        Second saved solution ('.npz').
    fil2 : str, optional
        Optional third saved solution ('.npz').

    Returns
    -------
    merged : (N, M) ndarray
        Concatenated array with time in the first column and state in the others;
        overlapping first rows of subsequent chunks are skipped.
    """

    sol = np.load(fil)["arr_0"]
    sol1 = np.load(fil1)["arr_0"]
    N1, x = np.shape(sol)
    N2, x = np.shape(sol1)
    merged = np.zeros((N1 + N2 - 1, x))

    if fil2 != "":
        sol2 = np.load(fil2)["arr_0"]
        N3, x = np.shape(sol2)
        merged = np.zeros((N1 + N2 + N3 - 2, x))

    merged[:N1, :] = sol

    if fil2 != "":
        merged[N1 : N1 + N2 - 1, :] = sol1[1:, :]
        merged[N1 + N2 - 1 :, :] = sol2[1:, :]
    else:
        merged[N1:, :] = sol1[1:, :]

    return merged


def read_charges(path, masses):
    """
    Read Born effective charges and dielectric constant from an input file.

    The file must contain the electronic dielectric tensor and the Born effective
    charge tensors for each atom.  This data is used to compute the coupling between
    atomic displacements and an external electric field.

    Parameters
    ----------
    filename : str
        Path to the file containing the dielectric constant and Born effective
        charges, typically generated by Density Functional Perturbation Theory
        (DFPT) or an equivalent calculation.
    masses : ndarray
        Array of atomic masses (in atomic mass units) for all atoms in the structure.

    Returns
    -------
    Zeff : ndarray, shape (3N, 3)
        Mass-rescaled Born effective charge tensor, expressed in atomic units.
    eps : float
        Macroscopic dielectric constant (scalar, averaged over the tensor).

    Notes
    -----
    - The Born effective charge tensor is divided by the square root of atomic
      masses to yield mass-weighted charges consistent with the dynamical
      variables of the TDSCHA formalism.
    - Units are automatically converted to Rydberg atomic units.
    """

    ff = open(path)
    lines = ff.readlines()
    ff.close()

    for i in range(len(lines)):
        if "number of atoms/cell" in lines[i]:
            nat = int(lines[i].split()[-1])
        if "Effective charges (d Force / dE) in cartesian axis with asr applied" in lines[i]:
            Zeff = []
            for j in range(nat):
                charge = []
                for l in range(3):
                    k = i + 4 * j + 3 + l
                    line = lines[k].split()
                    charge.append([float(line[2]), float(line[3]), float(line[4])])
                Zeff.append(charge)

            Zeff = np.array(Zeff)
            for i in range(3):
                print("check ", i + 1, np.sum(Zeff[:, i]))
        if "Dielectric constant in cartesian axis" in lines[i]:
            eps = float(lines[i + 2].split()[1])

    NewZeff = np.zeros((3 * nat, 3))
    for i in range(nat):
        NewZeff[3 * i : 3 * i + 3] = Zeff[i, :, :]
    NewZeff = np.einsum("ij,i->ij", NewZeff, 1 / np.sqrt(masses))
    return NewZeff, eps


def read_solution(label, chunks):
    """
    Load and stitch multiple saved evolution chunks into a single time series.

    Parameters
    ----------
    label : str
        Base label used when saving chunks (files named '{label}_{i}.npy' or '.npz').
    chunks : int
        Number of chunks to concatenate.

    Returns
    -------
    t : (N,) ndarray
        Time values concatenated across all chunks.
    sol : (N, D) ndarray
        State values (without the time column).

    Notes
    -----
    - Assumes each chunk has identical state dimensionality and contiguous time.
    - Skips duplicate first rows when concatenating is not handled here; use
      `merge_evolutions` if overlap should be removed.
    """

    for i in range(chunks):
        fil = np.load(label + "_%d.npy" % i)
        sh = np.shape(fil)
        N0 = sh[0]
        if i == 0:
            y = np.zeros((N0 * chunks, sh[1]))
        y[N0 * i : N0 * (i + 1), :] = fil
    t = y[:, 0]
    sol = y[:, 1:]
    return t, sol
