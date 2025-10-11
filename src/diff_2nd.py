import ase
from ase import Atoms
import numpy as np
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.Structure, cellconstructor.calculators


def diff_2nd(input_structure, calculator, what, eps=1e-5):
    """
    Compute second derivatives (harmonic force constants, φ) of a structure
    by finite differences of atomic forces.

    For each Cartesian coordinate of each atom, the routine displaces the atom
    by ±eps and evaluates the forces from the chosen calculator, assembling
    the dynamical matrix numerically.  The resulting φ is saved to "phi.npy".

    Parameters
    ----------
    input_structure : str
        Specifies the source of structural data:
          - "structure" : read atomic positions from a generic "POSCAR" file.
          - "dynamical_matrix" : load from an existing CellConstructor
            dynamical matrix stored in "final_result".
    calculator : ase.calculators.Calculator
        ASE calculator object used to compute forces.
    what : {"structure", "dynamical_matrix"}
        Type of input to read (see above).
    eps : float, optional
        Finite-difference displacement in Å (default 1e-5).

    Notes
    -----
    - Units: Forces from the calculator (in eV/Å) are converted to
      Rydberg/Bohr² using:
          1 Å = 1.889725988 Bohr, 1 Ry = 13.60570397 eV.
    - The resulting φ has dimensions (3N, 3N) where N is the number of atoms.
    - Saves "phi.npy" to disk in NumPy binary format.
    """

    Ang_to_Bohr = 1.889725988
    Ry_to_eV = 13.60570397

    if what == "structure":
        structure = CC.Structure.Structure()
        structure.read_generic_file("POSCAR")

    elif what == "dynamical_matrix":
        dyn = CC.Phonons.Phonons("final_result", 1)
        dyn.ForcePositiveDefinite()
        structure = dyn.structure
    else:
        sys.exit("Invalid input file")

    cell = structure.unit_cell
    ang_pos = structure.coords
    names = structure.atoms
    masses = np.repeat(structure.get_masses_array(), 3)

    crystal = Atoms(names, positions=ang_pos, pbc=True, cell=np.array(cell))
    crystal.calc = calculator

    nat = len(names)
    nmod = 3 * nat

    def get_index(i):
        icar = i % 3
        iat = int(np.floor(i / 3))
        return iat, icar

    phi = np.zeros((nmod, nmod))
    for i in range(nmod):
        iat, icar = get_index(i)

        crystal.positions[iat, icar] += eps
        fjp = crystal.get_forces()
        crystal.positions[iat, icar] -= 2 * eps
        fjm = crystal.get_forces()
        crystal.positions[iat, icar] += eps

        dfjdui = -(fjp - fjm) / (2 * eps)
        deriv = np.reshape(dfjdui, 3 * nat)

        phi[i, :] = deriv

    phi = phi / Ry_to_eV / Ang_to_Bohr**2  # Ry/B**2
    np.save("phi", phi)
    # phi = np.einsum('ij,i,j->ij', phi, 1/np.sqrt(masses), 1/np.sqrt(masses))
    # print_phonons_mat(phi)
