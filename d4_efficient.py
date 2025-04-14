import os
import ase
from ase import Atoms
from ase.build import molecule
import numpy as np
import sys
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammpslib import LAMMPSlib
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.Structure, cellconstructor.calculators
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
eps = 1e-3

def get_index(i):
    icar = i%3
    iat = int(np.floor(i/3))
    return iat, icar

# Parallelization

if nmod % 2!=0:
    nm = nmod-1
    numb = nm/3*(nm/2+1)*(nm+1)
    nm+=1
    numb = numb + nm*(nm+1)/2
else:
    nm = nmod
    numb = nm/3*(nm/2+1)*(nm+1)

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

psi_s = np.load('psi_symm.npy')

# Derivatives

result = np.zeros((end_per_proc[rank]-start_per_proc[rank], nmod+3 ))

count = 0
count_proc = 0
for i in range(nmod):
    iat, icar  = get_index(i)
    for k in range(i,nmod):
        kat, kcar = get_index(k)
        for l in range(k,nmod):
            lat, lcar = get_index(l)

            if count >= start_per_proc[rank] and count < end_per_proc[rank]:
                if rank==0:
                    t0 = time.time()
    
                if i!=k and i!=l and k!=l:
                    crystal.positions[iat, icar] += eps
                    crystal.positions[kat, kcar] += eps
                    crystal.positions[lat, lcar] += eps
                    fjppp = crystal.get_forces()
                    crystal.positions[iat, icar] -= 2*eps
                    fjmpp = crystal.get_forces()
                    crystal.positions[kat, kcar] -= 2*eps
                    fjmmp = crystal.get_forces()
                    crystal.positions[lat, lcar] -= 2*eps
                    fjmmm = crystal.get_forces()
                    crystal.positions[iat, icar] += 2*eps
                    fjpmm = crystal.get_forces()
                    crystal.positions[kat, kcar] += 2*eps
                    fjppm = crystal.get_forces()
                    crystal.positions[kat, kcar] -= 2*eps
                    crystal.positions[lat, lcar] += 2*eps
                    fjpmp = crystal.get_forces()
                    crystal.positions[iat, icar] -= 2*eps
                    crystal.positions[kat, kcar] += 2*eps
                    crystal.positions[lat, lcar] -= 2*eps
                    fjmpm = crystal.get_forces()
                    crystal.positions[iat, icar] += eps
                    crystal.positions[kat, kcar] -= eps
                    crystal.positions[lat, lcar] += eps

                    deriv = (fjppp-fjmpp+fjmmp-fjmmm+fjpmm-fjppm-fjpmp+fjmpm )/(8*eps**3)
                    deriv = np.reshape(deriv, 3*nat)
 
                if i==k and i!=l:
                    crystal.positions[iat, icar] += eps
                    crystal.positions[lat, lcar] += eps
                    fjpp = crystal.get_forces()  
                    crystal.positions[iat, icar] -= eps
                    fj0p = crystal.get_forces()
                    crystal.positions[iat, icar] -= eps
                    fjmp = crystal.get_forces()
                    crystal.positions[iat, icar] += 2*eps
                    crystal.positions[lat, lcar] -= 2*eps
                    fjpm = crystal.get_forces()
                    crystal.positions[iat, icar] -= eps
                    fj0m = crystal.get_forces()
                    crystal.positions[iat, icar] -= eps
                    fjmm = crystal.get_forces()
                    crystal.positions[iat, icar] += eps
                    crystal.positions[lat, lcar] += eps

                    deriv = (fjpp-fjpm+fjmp-fjmm-2*fj0p+2*fj0m)/(2*eps**3)
                    deriv = np.reshape(deriv, 3*nat)

                if i==l and i!=k:
                    crystal.positions[iat, icar] += eps                       
                    crystal.positions[kat, kcar] += eps
                    fjpp = crystal.get_forces()  
                    crystal.positions[iat, icar] -= eps
                    fj0p = crystal.get_forces()
                    crystal.positions[iat, icar] -= eps
                    fjmp = crystal.get_forces()
                    crystal.positions[iat, icar] += 2*eps
                    crystal.positions[kat, kcar] -= 2*eps
                    fjpm = crystal.get_forces()
                    crystal.positions[iat, icar] -= eps
                    fj0m = crystal.get_forces()
                    crystal.positions[iat, icar] -= eps
                    fjmm = crystal.get_forces()
                    crystal.positions[iat, icar] += eps
                    crystal.positions[kat, kcar] += eps

                    deriv = (fjpp-fjpm+fjmp-fjmm-2*fj0p+2*fj0m)/(2*eps**3)
                    deriv = np.reshape(deriv, 3*nat)


                if l==k and i!=l:
                    crystal.positions[lat, lcar] += eps
                    crystal.positions[iat, icar] += eps
                    fjpp = crystal.get_forces()  
                    crystal.positions[lat, lcar] -= eps
                    fj0p = crystal.get_forces()
                    crystal.positions[lat, lcar] -= eps
                    fjmp = crystal.get_forces()
                    crystal.positions[lat, lcar] += 2*eps
                    crystal.positions[iat, icar] -= 2*eps
                    fjpm = crystal.get_forces()
                    crystal.positions[lat, lcar] -= eps
                    fj0m = crystal.get_forces()
                    crystal.positions[lat, lcar] -= eps
                    fjmm = crystal.get_forces()
                    crystal.positions[lat, lcar] += eps
                    crystal.positions[iat, icar] += eps

                    deriv = (fjpp-fjpm+fjmp-fjmm-2*fj0p+2*fj0m)/(2*eps**3)
                    deriv = np.reshape(deriv, 3*nat)


                if i==l and k==l:
                    crystal.positions[iat, icar] += 2*eps
                    fjpp = crystal.get_forces()  
                    crystal.positions[iat, icar] -= eps
                    fjp = crystal.get_forces()  
                    crystal.positions[iat, icar] -= eps
                    fj0 = crystal.get_forces()  
                    crystal.positions[iat, icar] -= eps
                    fjm = crystal.get_forces()  
                    crystal.positions[iat, icar] -= eps
                    fjmm = crystal.get_forces()  
                    crystal.positions[iat, icar] += 2*eps

                    deriv = (fjpp/2 -fjp + fjm - fjmm/2)/eps**3
                    deriv = np.reshape(deriv, 3*nat)

                vect = np.zeros(nmod+3)
                vect[0] = i
                vect[1] = k
                vect[2] = l
                vect[3:] = -deriv
                result[count_proc,:] = vect

                if rank==0:
                    print(count,'/', end_per_proc[rank], time.time()-t0)
                count_proc += 1
            count += 1
if rank==0:
    t0 = time.time()
tot_sol = comm.gather(result, root=0)
if rank==0:
    print("Communication ", time.time()-t0)

if rank ==0:
    psi = np.zeros((nmod, nmod, nmod,nmod))

    for pc in range(size):
        for count_proc in range(len(tot_sol[pc])):
            vect = tot_sol[pc][count_proc,:]
            i = int(vect[0])
            k = int(vect[1])
            l = int(vect[2])
            deriv = vect[3:]
            psi[i,k,l,:] = deriv 


    for l in range(nmod):
        for i in range(nmod):
            for j in range(i,nmod):
                for k in range(j,nmod):
                    psi[i,j,l,k] = psi[i,j,k,l]
                    psi[i,k,j,l] = psi[i,j,k,l]
                    psi[i,k,l,j] = psi[i,j,k,l]
                    psi[i,l,j,k] = psi[i,j,k,l]
                    psi[i,l,k,j] = psi[i,j,k,l]
                    psi[j,i,k,l] = psi[i,j,k,l]
                    psi[j,i,l,k] = psi[i,j,k,l]
                    psi[j,k,i,l] = psi[i,j,k,l]
                    psi[j,k,l,i] = psi[i,j,k,l]
                    psi[j,l,i,k] = psi[i,j,k,l]
                    psi[j,l,k,i] = psi[i,j,k,l]
                    psi[k,i,j,l] = psi[i,j,k,l]
                    psi[k,i,l,j] = psi[i,j,k,l]
                    psi[k,j,i,l] = psi[i,j,k,l]
                    psi[k,j,l,i] = psi[i,j,k,l]
                    psi[k,l,i,j] = psi[i,j,k,l]
                    psi[k,l,j,i] = psi[i,j,k,l]
                    psi[l,i,j,k] = psi[i,j,k,l]
                    psi[l,i,k,j] = psi[i,j,k,l]
                    psi[l,j,i,k] = psi[i,j,k,l]
                    psi[l,j,k,i] = psi[i,j,k,l]
                    psi[l,k,i,j] = psi[i,j,k,l]
                    psi[l,k,j,i] = psi[i,j,k,l]



    psi = psi / 13.605 / 1.89**4 # Ry/B**4
    np.save("psi_efficient", psi)










