import numpy as np

from scipy.linalg import eigh

from mpi4py import MPI

import time

from numba import jit

# hamiltonian for tmdcs
from tight_binding_hamiltonian_python import *


### idos calculations for poisson solver

##### k-grid

red_k_intervall = 1.0 #0.06 

reduced_nk = 4*200 #4 * 200 

K_CENTER_reduced = np.array([0.5, 0.5 ])  #np.array([1.0/3.0, 1.0/3.0 ]) ##

K_ORIGIN_reduced = np.array([K_CENTER_reduced[0] - red_k_intervall/2, K_CENTER_reduced[1] - red_k_intervall/2])

degeneracy = 2 #4 #4 ## spin 2 center valley 2 = 4

#### mpi

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

size = comm.Get_size()


### parameter block

### calculation parameters

start_index = 117 ### relevant if in the bash script the first index is not 0, then we cannot create the initial numpy arrays for saving

full_output = True # prints out .txt files of DoS and which U has been produced 

max_iteration = 400  #1000  #500 #1000 #960


####

print_bandstructure = False #True # 1 for a bandstructure file

n_bands = 11

energy_deviation = 15 * pow(10,-3) #0.5 * pow(10,-3) #50 * pow(10,-3)

emin = -10.0 #-0.2  

emax = 10.0 #0.2

if rank == 0:

    print("emin, emax =", emin, emax)

en_intervall_count = round((emax - emin)/energy_deviation + 1)



####



dummy_iteration_counter = -1

if rank == 0 and False:

    fp = open("it_counter.txt", 'r')

    for line in fp:

        dummy_line = line.strip()

        dummy_line = dummy_line.split()

        dummy_iteration_counter = int(float(dummy_line[-1]))

    fp.close()
    
    
dummy_iteration_counter = 0


iteration_counter = comm.bcast(dummy_iteration_counter, root=0) # root: processor number; bcast broadcast

#### distribute k-points

count_kpts_per_thread = int(reduced_nk / size) 
kpts_remainder = reduced_nk % size 

kpts_per_thread = []
cummulative_kpts_count = []

for i in range(size):

	kpts_per_thread.append(count_kpts_per_thread)
	
for i in range(kpts_remainder):

	kpts_per_thread[-(i+1)] += 1
	

cummulative_kpts_count.append(0)

for i in range(1,size+1):

	cummulative_kpts_count.append(cummulative_kpts_count[i-1] + kpts_per_thread[i-1])


if rank == 0:
	print("cummulative_kpts_count = ", cummulative_kpts_count)

#### distribute energies


count_energies_per_thread = int(en_intervall_count / size)
energies_remainder = en_intervall_count % size #(reduced_nk - 1) % size why remove 1 point?
	
energies_per_thread = []
cummulative_energies_count = []



for i in range(size):

	energies_per_thread.append(count_energies_per_thread)
	
for i in range(energies_remainder):

	energies_per_thread[-(i+1)] += 1
	
	

cummulative_energies_count.append(0)

for i in range(1,size+1):

	cummulative_energies_count.append(cummulative_energies_count[i-1] + energies_per_thread[i-1])


if rank == 0:
	print("cummulative_energies_count = ", cummulative_energies_count)

### read iteration number


energy_intervall = np.linspace(emin, emax, en_intervall_count)



"""
if rank == 0:

    
    if dummy_iteration_counter == start_index: ##
    
    	energy_intervall = np.linspace(emin, emax, en_intervall_count)
    	
    	displacementfield_intervalls = np.linspace(0, max_U, max_iteration)
    
    	dos_array = np.zeros((max_iteration, en_intervall_count))
    	
    	it_tracker = np.zeros(max_iteration)

    	band_extremum = np.zeros(max_iteration)

    	band_extremum_k_point = np.zeros(2*max_iteration, dtype=float)

    	band_extremum_k_point_projected = np.zeros(2*max_iteration, dtype=float)
    	
    	np.save("../energy_intervall.npy", energy_intervall)
    	
    	np.save("../displacementfield_intervalls.npy", displacementfield_intervalls)
    	
    	np.save("../dos_blg.npy", dos_array)
    	
    	np.save("../it_tracker.npy", it_tracker)
    	
    else:
    
    	dos_array = np.load("../dos_blg.npy")
    	
    	it_tracker = np.load("../it_tracker.npy")
    	
    	it_tracker[dummy_iteration_counter] = dummy_iteration_counter
    	
    	np.save("../it_tracker.npy", it_tracker)

"""

if rank == 0:
    print("thread", rank, " has iteration_counter", iteration_counter)
    

### accurate blg parameters , see Jung & MacDonald Phys. Rev. B 89, 035405 (2014)

gamma_0 = 2.61

gamma_1 = 0.361

gamma_3 = -0.283

gamma_4 = 0.138

delta_prime = 0.015 #0.022 # if this is not the same as U, then different slopes


max_U = 200*pow(10,-3) #500*pow(10,-3) #250*pow(10,-3) #50*pow(10,-3)  #120*pow(10,-3)

U_incr = max_U / (max_iteration - 1)


### constants

r1 = np.zeros(2)
r2 = np.zeros(2)

a0_graphene = 2.461

#a0 = a # lattice constant graphene

r1[0] = a0_graphene
r1[1] = 0

r2[0] = r1[0] * np.cos(2.0 * np.pi / 3)
r2[1] = r1[0] * np.sin(2.0 * np.pi / 3)

g1 = np.zeros(2)
g2 = np.zeros(2)

g2[0] = 0
g2[1] = (2.0 * np.pi) / r2[1]

g1[0] = (2.0 * np.pi) / r1[0]
g1[1] = (-g1[0]*r2[0])/r2[1]

jacobi_det = np.abs(g1[0] * g2[1] - g1[1] * g2[0])/ (4.0 * np.pi * np.pi)

dos_prefactors = degeneracy * jacobi_det

K_CENTER_reduced = K_CENTER_reduced[0] * g1 + K_CENTER_reduced[1] * g2

K_REGION_ORIGIN = K_ORIGIN_reduced[0] * g1 + K_ORIGIN_reduced[1] * g2


U = U_incr * iteration_counter # U nicht relevant

kx_reduced = np.linspace(K_ORIGIN_reduced[0], K_ORIGIN_reduced[0] + red_k_intervall, reduced_nk)
ky_reduced = np.linspace(K_ORIGIN_reduced[1], K_ORIGIN_reduced[1] + red_k_intervall, reduced_nk)

kx_incr = red_k_intervall / (reduced_nk - 1.0)
ky_incr = red_k_intervall / (reduced_nk - 1.0)


	
if np.any(kx_reduced > 1.0) or np.any(kx_reduced < 0.0): ### outside of BZ, raise error

	raise ValueError("kx intervall exceeds Brillouin zone")
	
if np.any(ky_reduced > 1.0) or np.any(ky_reduced < 0.0): ### outside of BZ, raise error

	raise ValueError("ky intervall exceeds Brillouin zone")


### functions

def f_K(x,y):
    

    f = np.exp(complex(0,1) * y * a0_graphene / np.sqrt(3)) + 2 * np.exp(- complex(0,1) * y * a0_graphene / (2.0 * np.sqrt(3))) * np.cos(x * a0_graphene / 2.0)
    
    return f
    
    
def H_k(x,y):
    
    H = np.zeros((4,4), dtype = complex)
    
    ### the sites are A1, B1, A2, B2 , see figure 2 in review
    
    ### sites B1 and A2 form a dimer and are on top of each other
    
    ### diagonal elements
    
    H[0][0] = U/2   # A1
    
    H[1][1] = U/2 + delta_prime  # B1
    
    H[2][2] = -U/2 + delta_prime  # A2
    
    H[3][3] = -U/2    # B2
    
    
    ### dimer terms
    
    
    H[1][2] = gamma_1 #  B1, A2 term
    
    H[2][1] = gamma_1 #  A2, B1 term
    
    
    ### inplane terms
    
    H[0][1] = - gamma_0 * f_K(x,y) # A1, B1 term
    
    H[1][0] = - gamma_0 * np.conj(f_K(x,y)) # B1, A1 term
    
    H[2][3] = - gamma_0 * f_K(x,y) # A2, B2 term
    
    H[3][2] = - gamma_0 * np.conj(f_K(x,y)) # B2, A2 term
    
    ## skew terms same sublattice
    
    H[0][2] = gamma_4 * f_K(x,y) # A1, A2 term
    
    H[2][0] = gamma_4 * np.conj(f_K(x,y)) # A2, A1 term
    
    H[1][3] = gamma_4 * f_K(x,y) # A1, B2 term
    
    H[3][1] = gamma_4 * np.conj(f_K(x,y)) # B2, A1 term
    
    ## skew terms different sublattices
    
    H[0][3] = - gamma_3 * np.conj(f_K(x,y)) # A1, B2
    
    H[3][0] = - gamma_3 * f_K(x,y) # B2, A1
    
    return H


@jit
def gaussian(x0, x, sigma):

        return np.exp(-0.5 * ((x - x0) * (x - x0))/ (sigma*sigma)) / (sigma * np.sqrt(2.0*np.pi))


### get energies
    
    

kpath = np.zeros((reduced_nk, reduced_nk,2), dtype = np.float64)


k_grid_reduced = np.zeros((reduced_nk, reduced_nk,2))

k_grid_reduced[...,0] = np.copy(kx_reduced)

k_grid_reduced = np.swapaxes(k_grid_reduced, 0,1)

k_grid_reduced[...,1] = np.copy(ky_reduced)

k_grid_reduced = np.swapaxes(k_grid_reduced, 0,1)

kpath[...,0]  = k_grid_reduced[...,0] * g1[0] + k_grid_reduced[...,1] * g2[0] # k_x

kpath[...,1]  = k_grid_reduced[...,0] * g1[1] + k_grid_reduced[...,1] * g2[1] # k_y
	
"""

for j, ky in enumerate(ky_reduced):

	for i, kx in enumerate(kx_reduced):
	
	
		kpath[i, j] = kx * g1 + ky * g2



"""



time_start_bands = time.time()


dummy_bands = np.zeros((reduced_nk, reduced_nk, n_bands), dtype = np.float64)

bands = np.zeros(reduced_nk*reduced_nk*n_bands, dtype=np.float64)



# using the tight binding hamiltonian from 'tight_binding_hamiltonian_python.py' here
for j in np.arange(cummulative_kpts_count[rank], cummulative_kpts_count[rank+1]): # jeder ein k punkt; rank: welcher prozessor (id)

	for i in range(reduced_nk):
		
		dummy_bands[i, j] = np.real(eigh(hamiltonian_matrix(kpath[i, j][0], kpath[i, j][1]) + next_nearest_neighbour_hamitonian_matrix(kpath[i, j][0], kpath[i, j][1]), eigvals_only = True))	
		
		
"""		

TODO: make this work without for loops

dummy_thread_range = np.arange(cummulative_kpts_count[rank], cummulative_kpts_count[rank+1])

dummy_range = np.arange(reduced_nk)

index_map = np.zeros((dummy_thread_range.shape[0], reduced_nk, 2), dtype=np.int64)#, dtype=np.intp) #np.int64

index_map[..., 1] = np.copy(dummy_range)

index_map = np.swapaxes(index_map, 0,1)

index_map[...,0] = np.copy(dummy_thread_range)

index_map = np.swapaxes(index_map, 0,1)

#index_map = tuple(zip(*index_map))

#index_map = index_map.reshape(dummy_thread_range.shape[0]*reduced_nk, 2)

if rank ==0:

	#print(index_map)
	#print(index_map[0])
	print(index_map.shape)
	print(kpath.shape)
	print(kpath[index_map[0]].shape)
	
dummy_bands[index_map] = np.real(eigh(H_k(kpath[index_map][0], kpath[index_map][1]), eigvals_only = True))

"""


		
bands = comm.reduce(np.ravel(dummy_bands), op=MPI.SUM, root=0) # alle summieren, geliefert an root = 0
		
bands = comm.bcast(bands, root=0)

bands = bands.reshape((reduced_nk, reduced_nk, n_bands))

dummy_bands = 0

time_end_bands = time.time()

if rank == 0:

	print("bands computed in ", time_end_bands - time_start_bands, " seconds")


### calc dos


dummy_dos = np.zeros(en_intervall_count, dtype=np.float64)

dos = np.zeros(en_intervall_count, dtype=np.float64)


timer_dos_begin = time.time()


for i in np.arange(cummulative_energies_count[rank], cummulative_energies_count[rank+1]):

	rho = np.zeros((reduced_nk, reduced_nk), dtype=np.float64)
	
	current_dos_en = energy_intervall[i]
	
	for ind in range(n_bands):
		
		dummy_bands = np.copy(bands[...,ind])
		
		rho += gaussian(current_dos_en, dummy_bands, energy_deviation)
		
	dummy_dos[i] += np.trapezoid(np.trapezoid(rho, dx =  kx_incr), dx = ky_incr)
	
	



### a note on prefactors, if the prefactor is the degeneracy = 2, integrating out the bandstructure yields the total number of states, hence 8 for blg
### thus the dos is in units of 1/Energy
### multiplying by the jacobian and the degeneracy (which is called the dos_prefactors ), gives the number of electrons per unit-cell, hence in units of 1/(Energy * unit-cell area)

### from now on, only multiply by degeneracy, let the practitioner divide by unit-cell area/volume if need be

dummy_dos *= degeneracy  #dos_prefactors

dos = comm.reduce(dummy_dos, op=MPI.SUM, root=0)
		
	
timer_dos_end = time.time()	
	


if rank == 0:

	print("dos computed in ", timer_dos_end - timer_dos_begin, " seconds")

	np.savetxt('dos_MoS2.txt', np.array([ energy_intervall,  dos]).T)
	
	
	print("total number of states is ", np.trapezoid(dos, dx = energy_deviation))
		
		
	
### print results

if print_bandstructure == True and full_output == True and rank == 0:
	

	### print out the result
	
	
	"""
	fp = open("bandstructure.txt", 'w')
	
	
	for ind, ks in enumerate(kpath):
		
		for i in range(n_bands):
			
			print(ks[0], ks[1], bands[ind][i], file = fp)
			
	
				
	fp.close()
	
	"""	
	
	kx_band = np.ones((n_bands, reduced_nk , reduced_nk), dtype=np.float64)
	ky_band = np.ones((n_bands, reduced_nk , reduced_nk), dtype=np.float64)
	
	kx_band = kx_band * ((kpath.T)[0])
	ky_band = ky_band * ((kpath.T)[1])
	
	kx_band = kx_band.T
	ky_band = ky_band.T
	
	
	np.savetxt('bandstructure.txt', np.array([ np.ravel(kx_band), np.ravel(ky_band), np.ravel(bands)]).T)#, fmt='%10d')
	
