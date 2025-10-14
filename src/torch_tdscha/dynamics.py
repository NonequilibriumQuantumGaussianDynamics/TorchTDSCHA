import numpy as np
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
import torch
import math

from .averages import force, kappa, ext_for
from .averages import torch_force, torch_kappa, torch_ext_for



def get_y0(R, P, A, B, C):
    """
    Pack initial state (R, P, A, B, C) into a single 1D vector for ODE integration.

    State layout is:
        y = [R(0:n), P(0:n), vec(A), vec(B), vec(C)]

    Parameters
    ----------
    R : ndarray, shape (n,)
        Initial displacements.
    P : ndarray, shape (n,)
        Initial momenta.
    A : ndarray, shape (n, n)
        Initial covariance matrix ⟨u_i u_j⟩.
    B : ndarray, shape (n, n)
        Initial mixed moment ⟨u_i p_j⟩ (or equivalent correlation).
    C : ndarray, shape (n, n)
        Initial covariance ⟨p_i u_j⟩ (or related correlation).

    Returns
    -------
    ndarray, shape (2n + 3n^2,)
        Flattened initial state vector.
    """

    nmod = len(R)
    y0 = np.zeros(2 * nmod + 3 * nmod**2)
    y0[:nmod] = R
    y0[nmod : 2 * nmod] = P

    A_lin = np.reshape(A, nmod**2)
    B_lin = np.reshape(B, nmod**2)
    C_lin = np.reshape(C, nmod**2)

    y0[2 * nmod : 2 * nmod + nmod**2] = A_lin
    y0[2 * nmod + nmod**2 : 2 * nmod + 2 * nmod**2] = B_lin
    y0[2 * nmod + 2 * nmod**2 : 2 * nmod + 3 * nmod**2] = C_lin

    return y0


def get_y0_torch(R, P, A, B, C):
    """
    Torch version of `get_y0`: pack (R, P, A, B, C) into a 1D torch tensor.

    Parameters
    ----------
    R : torch.Tensor, shape (n,)
    P : torch.Tensor, shape (n,)
    A : torch.Tensor, shape (n, n)
    B : torch.Tensor, shape (n, n)
    C : torch.Tensor, shape (n, n)

    Returns
    -------
    torch.Tensor, shape (2n + 3n^2,)
        Flattened initial state vector on the same device/dtype as inputs.
    """

    nmod = len(R)
    y0 = torch.zeros(2 * nmod + 3 * nmod**2, dtype=R.dtype, device=R.device)

    y0[:nmod] = R
    y0[nmod : 2 * nmod] = P

    A_lin = A.reshape(-1)
    B_lin = B.reshape(-1)
    C_lin = C.reshape(-1)

    y0[2 * nmod : 2 * nmod + nmod**2] = A_lin
    y0[2 * nmod + nmod**2 : 2 * nmod + 2 * nmod**2] = B_lin
    y0[2 * nmod + 2 * nmod**2 : 2 * nmod + 3 * nmod**2] = C_lin

    return y0


def tdscha(t, y, phi, chi, psi, field, gamma):
    """
    Right-hand side of the TDSCHA ODE system (NumPy version).

    Given y(t) = [R, P, vec(A), vec(B), vec(C)], compute dy/dt, using `force`,
    `kappa`, and 'ext_for` as drive.

    Parameters
    ----------
    t : float
        Time (code units; input typically in a.u. after conversion).
    y : ndarray, shape (2n + 3n^2,)
        Current state vector.
    phi : ndarray, shape (n, n)
        Harmonic (second-order) force constants.
    chi : ndarray, shape (n, n, n)
        Third-order force constants.
    psi : ndarray, shape (n, n, n, n)
        Fourth-order force constants.
    field : dict
        External field specification; passed to `ext_for(t, field)`.
    gamma : float
        Linear damping coefficient for momenta.

    Returns
    -------
    ndarray, shape (2n + 3n^2,)
        Time derivative dy/dt.
    """

    nmod = int((-2 + np.sqrt(4 + 12 * len(y))) / 6)

    R = y[:nmod]
    P = y[nmod : 2 * nmod]

    A_lin = y[2 * nmod : 2 * nmod + nmod**2]
    B_lin = y[2 * nmod + nmod**2 : 2 * nmod + 2 * nmod**2]
    C_lin = y[2 * nmod + 2 * nmod**2 : 2 * nmod + 3 * nmod**2]

    A = np.reshape(A_lin, (nmod, nmod))
    B = np.reshape(B_lin, (nmod, nmod))
    C = np.reshape(C_lin, (nmod, nmod))

    f = force(R, A, phi, chi, psi)
    # f = f_classic(R,phi,psi)
    curv = kappa(R, A, phi, chi, psi)

    ydot = np.zeros(len(y))

    ydot[:nmod] = P
    ydot[nmod : 2 * nmod] = f + ext_for(t, field) - gamma * P  #

    Adot = C + np.transpose(C)

    Bdot = -np.dot(curv, C)
    Bdot = Bdot + np.transpose(Bdot)
    Cdot = B - np.dot(A, curv)  # -0.001*C

    ydot[2 * nmod : 2 * nmod + nmod**2] = np.reshape(Adot, nmod**2)
    ydot[2 * nmod + nmod**2 : 2 * nmod + 2 * nmod**2] = np.reshape(Bdot, nmod**2)
    ydot[2 * nmod + 2 * nmod**2 : 2 * nmod + 3 * nmod**2] = np.reshape(Cdot, nmod**2)

    return ydot


def tdscha_torch(t, y, phi, chi, psi, field, gamma):
    """
    Right-hand side of the TDSCHA ODE system (PyTorch version).

    Same dynamics as `tdscha` but computed with torch tensors (GPU/autograd-friendly).

    Parameters
    ----------
    t : torch.Tensor or float
        Time value.
    y : torch.Tensor, shape (2n + 3n^2,)
        Current state vector.
    phi : torch.Tensor, shape (n, n)
    chi : torch.Tensor, shape (n, n, n)
    psi : torch.Tensor, shape (n, n, n, n)
    field : dict
        External field parameters; used by `torch_ext_for`.
    gamma : float or torch.Tensor
        Damping coefficient.

    Returns
    -------
    torch.Tensor, shape (2n + 3n^2,)
        Time derivative dy/dt.
    """

    L = y.numel()
    nmod = int((-2 + math.sqrt(4 + 12 * L)) / 6)

    R = y[:nmod]
    P = y[nmod : 2 * nmod]

    A_lin = y[2 * nmod : 2 * nmod + nmod**2]
    B_lin = y[2 * nmod + nmod**2 : 2 * nmod + 2 * nmod**2]
    C_lin = y[2 * nmod + 2 * nmod**2 : 2 * nmod + 3 * nmod**2]

    A = A_lin.reshape(nmod, nmod)
    B = B_lin.reshape(nmod, nmod)
    C = C_lin.reshape(nmod, nmod)

    f = torch_force(R, A, phi, chi, psi)
    curv = torch_kappa(R, A, phi, chi, psi)

    ydot = torch.zeros_like(y)
    ydot[:nmod] = P
    ydot[nmod : 2 * nmod] = f + torch_ext_for(t, field) - gamma * P

    Adot = C + C.t()

    Bdot = -torch.matmul(curv, C)
    Bdot = Bdot + Bdot.t()

    Cdot = B - torch.matmul(A, curv)

    ydot[2 * nmod : 2 * nmod + nmod**2] = Adot.reshape(-1)
    ydot[2 * nmod + nmod**2 : 2 * nmod + 2 * nmod**2] = Bdot.reshape(-1)
    ydot[2 * nmod + 2 * nmod**2 : 2 * nmod + 3 * nmod**2] = Cdot.reshape(-1)

    return ydot


def td_evolution(
    R,
    P,
    A,
    B,
    C,
    field,
    gamma,
    phi,
    chi,
    psi,
    Time,
    NS,
    y0=None,
    init_t=0,
    chunks=1,
    label="solution",
):
    """
    Integrate TDSCHA dynamics with SciPy's `solve_ivp` (NumPy pipeline).

    Time (fs) is converted internally to code units. Integration can be chunked to
    save intermediate results to disk as compressed .npz files.

    Parameters
    ----------
    R, P : ndarray, shape (n,)
        Initial displacements and momenta.
    A, B, C : ndarray, shape (n, n)
        Initial second-moment/correlation matrices.
    field : dict
        External field specification (see `ext_for`).
    gamma : float
        Linear momentum damping.
    phi : ndarray, shape (n, n)
    chi : ndarray, shape (n, n, n)
    psi : ndarray, shape (n, n, n, n)
    Time : float
        Total physical time in femtoseconds.
    NS : int
        Total number of evaluation points.
    y0 : ndarray, optional
        Pre-packed initial state; if None, built from (R,P,A,B,C).
    init_t : float, default 0
        Initial time in femtoseconds.
    chunks : int, default 1
        Split integration into `chunks` segments (each saved to disk).
    label : str, default "solution"
        Prefix for saved files (e.g., label_0.npz, label_1.npz).

    Returns
    -------
    OdeResult
        Final SciPy solution object from the last chunk.
    """
    # om_L in THz, Time in fs

    init_t = init_t / (4.8377687 * 1e-2)
    Time = Time / (4.8377687 * 1e-2)
    Time = Time / chunks
    NS = int(NS / chunks)

    if y0 is None:
        y0 = get_y0(R, P, A, B, C)

    for i in range(chunks):
        t_eval = np.linspace(init_t, init_t + Time, NS)
        tspan = [init_t, init_t + Time]
        sol = solve_ivp(tdscha, tspan, y0, t_eval=t_eval, args=(phi, chi, psi, field, gamma))
        save(label + "_%d" % i, sol.t, sol.y)

        y0 = sol.y[:, -1]
        init_t += Time

    return sol


def torch_evolution(
    R,
    P,
    A,
    B,
    C,
    field,
    gamma,
    phi,
    chi,
    psi,
    Time,
    NS,
    y0=None,
    init_t=0,
    chunks=1,
    label="solution",
):
    """
    Integrate TDSCHA dynamics with `torchdiffeq.odeint` (PyTorch pipeline).

    Converts inputs to torch tensors (double precision), moves to CPU/GPU,
    and integrates in `chunks`. Each chunk is saved to compressed .npz.

    Parameters
    ----------
    R, P : ndarray, shape (n,)
    A, B, C : ndarray, shape (n, n)
    field : dict
        External field; `Zeff` and `edir` are converted to torch tensors.
    gamma : float
    phi : ndarray, shape (n, n)
    chi : ndarray, shape (n, n, n)
    psi : ndarray, shape (n, n, n, n)
    Time : float
        Total physical time in femtoseconds.
    NS : int
        Number of time points per full trajectory (split across chunks).
    y0 : torch.Tensor, optional
        Packed initial state; if None, built via `get_y0_torch`.
    init_t : float, default 0
        Initial time in femtoseconds.
    chunks : int, default 1
        Number of segments to split the integration.
    label : str, default "solution"
        Prefix used when saving each chunk.

    Returns
    -------
    torch.Tensor, shape (NS_chunk, 2n + 3n^2)
        Trajectory of the last chunk.
    """
    # om_L in THz, Time in fs

    init_t = init_t / (4.8377687 * 1e-2)
    Time = Time / (4.8377687 * 1e-2)
    Time = Time / chunks
    NS = int(NS / chunks)

    phi, chi, psi, R, P, A, B, C, field = torch_init(phi, chi, psi, R, P, A, B, C, field)

    if y0 is None:
        y0 = get_y0_torch(R, P, A, B, C)

    device, dtype = y0.device, y0.dtype

    for i in range(chunks):
        tspan = torch.linspace(init_t, init_t + Time, NS, device=device, dtype=dtype)
        # sol = solve_ivp(func, tspan, y0, t_eval=t_eval, args=(phi, chi,  psi, field, gamma))
        with torch.no_grad():
            func = lambda t, y: tdscha_torch(t, y, phi, chi, psi, field, gamma)
            sol = odeint(func, y0, tspan, method="rk4")
        save_torch(label + "_%d" % i, tspan, sol)

        y0 = sol[-1]
        init_t += Time

    return sol


def save(label, t, sol):
    """
    Save NumPy trajectory to a compressed .npz file as a single 2D array.

    The first column is time, the remaining columns are the state vector y.

    Parameters
    ----------
    label : str
        Output filename prefix ('.npz' is appended by `np.savez_compressed`).
    t : ndarray, shape (nt,)
        Time grid.
    sol : ndarray, shape (state_dim, nt)
        State matrix in column-major time (as returned by `solve_ivp`: y(t)).

    Returns
    -------
    None
    """

    sol = np.transpose(sol)
    sh = np.shape(sol)
    sh = [sh[0], sh[1] + 1]
    y = np.zeros(sh)
    y[:, 0] = t
    y[:, 1:] = sol
    np.savez_compressed(label, y)


def save_torch(label, t, sol):
    """
    Save torch trajectory to a compressed .npz file as a single 2D array.

    Converts tensors to NumPy and packs time as the first column.

    Parameters
    ----------
    label : str
        Output filename prefix.
    t : torch.Tensor, shape (nt,)
        Time grid.
    sol : torch.Tensor, shape (nt, state_dim) or (state_dim, nt)
        Trajectory; if time-major, it is concatenated with `t` appropriately.

    Returns
    -------
    None
    """
    t_np = t.detach().cpu().numpy()
    sol_np = sol.detach().cpu().numpy()

    y = np.concatenate([t_np[:, None], sol_np], axis=1)
    np.savez_compressed(label, y)


def torch_init(phi, chi, psi, R, P, A, B, C, field):
    """
    Convert NumPy inputs to torch tensors (float64) and move to CPU/GPU.

    Also converts `field['Zeff']` and `field['edir']` to tensors. Autograd is
    disabled (the integration uses `no_grad()`).

    Parameters
    ----------
    phi : ndarray, shape (n, n)
    chi : ndarray, shape (n, n, n)
    psi : ndarray, shape (n, n, n, n)
    R, P : ndarray, shape (n,)
    A, B, C : ndarray, shape (n, n)
    field : dict
        Must contain keys 'Zeff' (ndarray) and 'edir' (array-like).

    Returns
    -------
    phi, chi, psi, R, P, A, B, C, field : torch.Tensor, ..., dict
        Torch tensors on the selected device (GPU if available, else CPU),
        with dtype=torch.float64. `field` is the same dict with 'Zeff' and 'edir'
        replaced by torch tensors.
    """

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    phi = torch.from_numpy(phi).to(device=device, dtype=dtype)
    chi = torch.from_numpy(chi).to(device=device, dtype=dtype)
    psi = torch.from_numpy(psi).to(device=device, dtype=dtype)
    R = torch.from_numpy(R).to(device=device, dtype=dtype)
    P = torch.from_numpy(P).to(device=device, dtype=dtype)
    A = torch.from_numpy(A).to(device=device, dtype=dtype)
    B = torch.from_numpy(B).to(device=device, dtype=dtype)
    C = torch.from_numpy(C).to(device=device, dtype=dtype)

    Zeff = field["Zeff"]
    field["Zeff"] = torch.from_numpy(Zeff).to(device=device, dtype=dtype)

    edir = field["edir"]
    field["edir"] = torch.tensor(edir).to(device=device, dtype=dtype)

    return phi, chi, psi, R, P, A, B, C, field
