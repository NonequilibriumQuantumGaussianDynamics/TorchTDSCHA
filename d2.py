import os
import ICP
import ase
from ase import Atoms
from ase.build import molecule
from ICP.Calculator import ICPCalculator
import ICP.libs
import numpy as np
from ase.visualize import view
import glob
import sys
from quippy.potential import Potential
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammpslib import LAMMPSlib
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.Structure, cellconstructor.calculators
from model_multi import *



bohr_to_ang = 0.529177

structure = CC.Structure.Structure()
structure.read_generic_file('aiida.in')
structure.read_generic_file('POSCAR')

cell = structure.unit_cell
ang_pos = structure.coords
names = structure.atoms
masses = np.repeat(structure.get_masses_array(), 3)
print(cell)
print(ang_pos)
print(names)

crystal = Atoms(names, positions=ang_pos, pbc=True, cell=np.array(cell))

#calc = ICPCalculator(n1,n2,n3, re_init=True)
os.environ["ASE_LAMMPSRUN_COMMAND"] = "/home/flibbi/programs/lammps-stable_23Jun2022_update4/build/lmp"
calc = LAMMPS(
specorder=["Sr", "Ti", "O"],
keep_tmp_files=True,
tmp_dir="tmpflare",
keep_alive=False,
parameters={
  "pair_style": "flare",
"pair_coeff": [f"* * /scratch/flibbi/sscha/SrTiO3_flare/srtio3_34_160atoms.otf.flare"],
"atom_style": "atomic"
}
)

calc = LAMMPSlib(
        keep_alive=True,
        log_file="log.ase",
        atom_types={"Sr" : 1, "Ti" : 2, "O" : 3},
        lmpcmds=[
            "pair_style flare",
            "pair_coeff * * /scratch/flibbi/sscha/SrTiO3_flare/srtio3.otf.flare"
            ])

crystal.calc = calc


nat = len(names)
nmod = 3*nat
eps = 1e-5

def get_index(i):
    icar = i%3
    iat = int(np.floor(i/3))
    return iat, icar

phi = np.zeros((nmod, nmod))
for i in range(nmod):
    print(i)
    iat, icar  = get_index(i)
    
    crystal.positions[iat, icar] += eps
    fjp = crystal.get_forces()
    crystal.positions[iat, icar] -= 2*eps
    fjm = crystal.get_forces()
    crystal.positions[iat, icar] += eps

    dfjdui = -(fjp-fjm)/(2*eps)
    deriv = np.reshape(dfjdui, 3*nat)

    phi[i,:] = deriv


phi = phi / 13.60570397 / 1.889725988**2 # Ry/B**2
np.save('phi', phi)
phi = np.einsum('ij,i,j->ij', phi, 1/np.sqrt(masses), 1/np.sqrt(masses))
print_phonons_mat(phi)











