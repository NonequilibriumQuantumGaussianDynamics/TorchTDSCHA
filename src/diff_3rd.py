import os
import ase
from ase import Atoms
import numpy as np
import glob
import sys
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.Structure, cellconstructor.calculators
from mpi4py import MPI
import time
from ase.calculators.emt import EMT

def diff_3rd(input_structure, calculator, what, eps = 1e-4):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    Ang_to_Bohr =  1.889725988
    Ry_to_eV = 13.60570397

    if what == 'structure':
        structure = CC.Structure.Structure()
        structure.read_generic_file('POSCAR')

    elif what == 'dynamical_matrix':
        dyn = CC.Phonons.Phonons('final_result',1)
        dyn.ForcePositiveDefinite()
        structure = dyn.structure
    else:
        sys.exit('Invalid input file')

    cell = structure.unit_cell
    ang_pos = structure.coords
    names = structure.atoms
    masses = np.repeat(structure.get_masses_array(), 3)

    crystal = Atoms(names, positions=ang_pos, pbc=True, cell=np.array(cell))
    crystal.calc = calculator


    nat = len(names)
    nmod = 3*nat

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


        chi = chi / Ry_to_eV / Ang_to_Bohr**3 # Ry/B**3
        np.save("chi", chi)










