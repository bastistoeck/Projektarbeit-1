import numpy as np

from mpi4py import MPI

import time

import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD

rank = comm.Get_rank()

size = comm.Get_size()




read_energy_interval_shape = 0

read_displacement_field_shape = 0

read_dos = np.zeros(1)

read_energy_interval = np.zeros(1)

read_displacement_field = np.zeros(1)


if rank == 0:

	read_dos = np.load("dos_blg.npy")

	read_energy_interval = np.load("energy_intervall.npy")
	
	read_energy_interval_shape = read_energy_interval.shape[0]
	
	read_displacement_field = np.load("displacementfield_intervalls.npy")
	
	read_displacement_field_shape = read_displacement_field.shape[0]
	
	print("dos shape = ", read_dos.shape)
	
	print("energy interval shape = ", read_energy_interval.shape)

	print("displacement field shape = ", read_displacement_field.shape)
	
	
energy_shape = comm.bcast(read_energy_interval_shape, root = 0)

displacement_field_shape = comm.bcast(read_displacement_field_shape, root = 0)

energy_half_point = int(energy_shape/2)

energy_interval = np.zeros(energy_shape, dtype = np.float64)

displacement_field_interval = np.zeros(displacement_field_shape, dtype = np.float64)

dos = np.zeros((displacement_field_shape, energy_shape), dtype = np.float64)


###


energy_interval = comm.bcast(read_energy_interval, root = 0)

displacement_field_interval = comm.bcast(read_displacement_field, root = 0)

dos = comm.bcast(read_dos, root = 0)

#read_dos = 0

read_energy_interval = 0

read_displacement_field = 0

###


en_min = energy_interval[0]

en_max = energy_interval[-1]

en_incr = energy_interval[1] - energy_interval[0]


pot_min = displacement_field_interval[0]

pot_max = displacement_field_interval[-1]

pot_incr = displacement_field_interval[1] - displacement_field_interval[0]



if rank == 0:

	plt.plot(energy_interval*pow(10,3), dos[0], label=r"$U=$"+str(int(displacement_field_interval[0]*pow(10,3)))+r" $meV$")
	
	plt.plot(energy_interval*pow(10,3), dos[99], label=r"$U=$"+str(int(displacement_field_interval[99]*pow(10,3)))+r" $meV$")
	
	plt.plot(energy_interval*pow(10,3), dos[199], label=r"$U=$"+str(int(displacement_field_interval[199]*pow(10,3)))+r" $meV$")
	
	plt.plot(energy_interval*pow(10,3), dos[299], label=r"$U=$"+str(int(displacement_field_interval[299]*pow(10,3)))+r" $meV$")
	
	plt.plot(energy_interval*pow(10,3), dos[399], label=r"$U=$"+str(int(displacement_field_interval[399]*pow(10,3)))+r" $meV$")
	
	plt.xlabel("E [meV]")
	
	plt.legend()
	
	plt.title("density of states")
	
	plt.savefig("dos_0.png")

	plt.clf()
	
	

	plt.imshow(dos, extent = [en_min*pow(10,3), en_max*pow(10,3), pot_min*pow(10,3), pot_max*pow(10,3)], origin = "lower", cmap = 'YlOrBr', aspect = 2.0)
	
	plt.title("density of states")
	
	plt.xlabel("E [meV]")
	plt.ylabel("U [meV]")
	
	plt.colorbar()
	
	plt.savefig("dos.png")
	
	plt.clf()
	
	

#### idos


thread_idos = np.zeros(dos.shape, dtype=np.float64)

idos = np.zeros(dos.shape, dtype=np.float64)


#### distribute displacement fields among workers


count_displacementfield_per_thread = int(displacement_field_shape / size)
displacementfield_remainder = displacement_field_shape % size #(reduced_nk - 1) % size why remove 1 point?
	
displacementfield_per_thread = []
cummulative_displacementfield_count = []



for i in range(size):

	displacementfield_per_thread.append(count_displacementfield_per_thread)
	
for i in range(displacementfield_remainder):

	displacementfield_per_thread[-(i+1)] += 1
	
	

cummulative_displacementfield_count.append(0)

for i in range(1,size+1):

	cummulative_displacementfield_count.append(cummulative_displacementfield_count[i-1] + displacementfield_per_thread[i-1])


if rank == 0:
	print("cummulative_displacementfield_count = ", cummulative_displacementfield_count)
	
	
energy_counter = np.arange(energy_shape)

thread_electron_band_edge_indices = np.zeros(displacement_field_shape, dtype=np.int64)
thread_hole_band_edge_indices = np.zeros(displacement_field_shape, dtype=np.int64)

electron_band_edge_indices = np.zeros(displacement_field_shape, dtype=np.int64)
hole_band_edge_indices = np.zeros(displacement_field_shape, dtype=np.int64)

band_threshold = pow(10,-10)


begin_idos_timer = time.time()
	
	
for i in np.arange(cummulative_displacementfield_count[rank], cummulative_displacementfield_count[rank+1]):

	for j in range(energy_shape):

		thread_idos[i][j] = np.trapezoid(dos[i][energy_counter[:j]], dx = en_incr)
	
	thread_idos[i] -= thread_idos[i][energy_half_point]
	
	#https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
	
	thread_electron_band_edge_indices[i] = energy_half_point + np.argmax(thread_idos[i][energy_half_point:] > band_threshold)
	
	thread_hole_band_edge_indices[i] = energy_half_point - np.argmax(np.flip(thread_idos[i][:energy_half_point]) <= -band_threshold)

idos = comm.reduce(thread_idos, op=MPI.SUM, root=0)

electron_band_edge_indices = comm.reduce(thread_electron_band_edge_indices, op=MPI.SUM, root=0)

hole_band_edge_indices = comm.reduce(thread_hole_band_edge_indices, op=MPI.SUM, root=0)

thread_idos = 0	
	
end_idos_timer = time.time()


if rank == 0:

	print("idos calculated in ", end_idos_timer - begin_idos_timer)
	
	np.save("idos_blg.npy", idos)
	
	electron_band_edge = energy_interval[electron_band_edge_indices]
	
	hole_band_edge = energy_interval[hole_band_edge_indices]
	
	band_gap =  electron_band_edge - hole_band_edge
	
	np.save("electron_band_edge.npy", electron_band_edge)
	
	np.save("hole_band_edge.npy", hole_band_edge)
	
	np.save("band_gap.npy", band_gap)
	
	
	### idos
	
	plt.imshow(idos, extent = [en_min*pow(10,3), en_max*pow(10,3), pot_min*pow(10,3), pot_max*pow(10,3)], origin = "lower", cmap = 'RdYlBu', aspect = 2.0)
	
	plt.title("integrated density of states")
	
	plt.xlabel("E [meV]")
	plt.ylabel("U [meV]")
	
	plt.colorbar()
	
	plt.savefig("idos.png")
	
	plt.clf()
	
	
	### band gap
	
	plt.plot(displacement_field_interval*pow(10,3),  band_gap * pow(10,3))
	
	plt.title("band gap")
	
	plt.xlabel(r"$U$ $[meV]$")

	plt.ylabel(r"$E_{gap}$ $[meV]$")
		
	plt.savefig("band_gap.png")
	
	plt.clf()
	
	
	### band edges
	
	plt.plot(displacement_field_interval*pow(10,3),  electron_band_edge * pow(10,3), label="electron")
	
	plt.plot(displacement_field_interval*pow(10,3),  hole_band_edge * pow(10,3), label="hole")
	
	plt.title("band edge")
	
	plt.xlabel(r"$U$ $[meV]$")

	plt.ylabel(r"$E_{edge}$ $[meV]$")
	
	plt.legend()
		
	plt.savefig("band_edges.png")
	
	plt.clf()
	
	
	### idos with band edges
	
	plt.imshow(idos, extent = [en_min*pow(10,3), en_max*pow(10,3), pot_min*pow(10,3), pot_max*pow(10,3)], origin = "lower", cmap = 'RdYlBu', aspect = 2.0)
	
	plt.plot(electron_band_edge * pow(10,3), displacement_field_interval*pow(10,3), c="black", linestyle="dashed")
	
	plt.plot(hole_band_edge * pow(10,3), displacement_field_interval*pow(10,3),  c="black", linestyle="dashed")
	
	plt.title("integrated density of states")
	
	plt.xlabel("E [meV]")
	plt.ylabel("U [meV]")
	
	plt.colorbar()
	
	plt.savefig("idos_bandedges.png")
	
	plt.clf()
