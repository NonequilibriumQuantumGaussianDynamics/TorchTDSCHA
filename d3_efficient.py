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
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

bohr_to_ang = 0.529177

structure = CC.Structure.Structure()
structure.read_generic_file('POSCAR')

cell = structure.unit_cell
ang_pos = structure.coords
names = structure.atoms
masses = np.repeat(structure.get_masses_array(), 3)

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
eps = 1e-4

def get_index(i):
    icar = i%3
    iat = int(np.floor(i/3))
    return iat, icar


# Parallelization

nm = nmod
numb = nm*(nm+1)/2

av_point = int(np.floor(numb/float(size)))
rest = numb%size
point_per_proc = []
rest_cp = rest
for i in range(size):
    if rest_cp > 0:
        point_per_proc.append(av_point+1)
        rest_cp = rest_cp - 1
    else:
        point_per_proc.append(av_point)

start_per_proc = []
end_per_proc = []
sum = 0
for i in range(len(point_per_proc)):
    start_per_proc.append(sum)
    sum = sum + point_per_proc[i]
    end_per_proc.append(sum)
if rank==0:
    print(numb)
    print(start_per_proc)
    print(end_per_proc)

# Derivative

count = 0
count_proc = 0
result = np.zeros((end_per_proc[rank]-start_per_proc[rank], nmod+2 ))

for i in range(nmod):
    print(i)
    iat, icar  = get_index(i)
    for k in range(i,nmod):
        print(k)
        kat, kcar = get_index(k)

        if count >= start_per_proc[rank] and count < end_per_proc[rank]:
            if rank==0:
                t0 = time.time()
   
            if i!=k:
                crystal.positions[iat, icar] += eps
                crystal.positions[kat, kcar] += eps
                fjpp = crystal.get_forces()
                crystal.positions[iat, icar] -= 2*eps
                fjmp = crystal.get_forces()
                crystal.positions[kat, kcar] -= 2*eps
                fjmm = crystal.get_forces()
                crystal.positions[iat, icar] += 2*eps
                fjpm = crystal.get_forces()
                crystal.positions[iat, icar] -= eps
                crystal.positions[kat, kcar] += eps

                deriv = -(fjpp+fjmm-fjmp-fjpm)/(4*eps**2)
                deriv = np.reshape(deriv, 3*nat)

            if i==k:
                crystal.positions[iat, icar] += eps
                fjp = crystal.get_forces()
                crystal.positions[iat, icar] -= eps
                fj0 = crystal.get_forces()
                crystal.positions[iat, icar] -= eps
                fjm = crystal.get_forces()
                crystal.positions[iat, icar] += eps

                deriv = -(fjp+fjm-2*fj0)/eps**2
                deriv = np.reshape(deriv, 3*nat)

            vect = np.zeros(nmod+2)
            vect[0] = i
            vect[1] = k
            vect[2:] = deriv
            result[count_proc,:] = vect

            if rank==0:
                print(count,'/', end_per_proc[rank], time.time()-t0)
            count_proc+=1

        count += 1

if rank==0:
    t0 = time.time()
tot_sol = comm.gather(result, root=0)
if rank==0:
    print("Communication ", time.time()-t0)

if rank ==0:

    chi = np.zeros((nmod, nmod, nmod))
    for pc in range(size):
        for count_proc in range(len(tot_sol[pc])):
            vect = tot_sol[pc][count_proc,:]
            i = int(vect[0])
            k = int(vect[1])
            deriv = vect[2:]
            chi[i,k,:] = deriv

    for l in range(nmod):
        for i in range(nmod):
            for k in range(i,nmod):
                chi[i,l,k] = chi[i,k,l]

                chi[l,i,k] = chi[i,k,l]
                chi[k,i,l] = chi[i,k,l]

                chi[k,l,i] = chi[i,k,l]
                chi[l,k,i] = chi[i,k,l]

    print("Number of derivatives", count, numb)


    chi = chi / 13.605 / 1.89**3 # Ry/B**3
    np.save("chi_efficient", chi)










