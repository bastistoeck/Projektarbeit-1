# script to run blg calculation over range of hbn thicknesses and back gate voltages

import os
import multiprocessing

jupyter_true = False #True ## cluster variable

if jupyter_true == True: 

    import ipyparallel as ipp
    # Number of MPI engines (processes)
    num_engines =  4 #4  #10
    # Number of threads per engine (this is a separate concept; see note below)
    # threads_per_engine = 20

    # Create a cluster that uses MPI to launch engines.
    # The 'mpi_args' list is passed directly to the mpiexec/mpirun command.
    # '--use-hwthread-cpus' tells MPI to use hardware threads (cores * threads per core) as slots.
    # '-n' specifies the total number of MPI processes (engines) to start.
    c = ipp.Cluster(engines='mpi',
                    n=num_engines,
                    mpi_args=['--use-hwthread-cpus'])

    # Start the cluster and connect a client
    rc = c.start_and_connect_sync()

    # Your parallel code here...
    # For example, get a view of all engines
    view = rc[:]
    print(f"Successfully connected to {len(rc)} engines.")

    # When done, shutdown the cluster
    # c.shutdown()
    # rc.shutdown()

    view.push({'num_engines': num_engines}) 
    view.push({'jupyter_true': jupyter_true}) 

    #%autopx

    print(jupyter_true)

    ### 

from mpi4py import MPI

import os
import shutil
import multiprocessing

# Check MPI rank and size
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size
comm = MPI.COMM_WORLD

if jupyter_true == False:

    num_engines = size

# Check OMP_NUM_THREADS setting
os.environ["OMP_NUM_THREADS"] = "1"

omp_threads = os.environ.get('OMP_NUM_THREADS', 'Not set')

# Check available CPU threads (as seen by Python)
available_cpus = multiprocessing.cpu_count()

print(f"Rank {rank}/{size}: OMP_NUM_THREADS={omp_threads}, Available CPUs={available_cpus}")

max_threads = int(available_cpus / size)


import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

import matplotlib as mpl

from scipy import interpolate
import scipy.optimize

from copy import deepcopy
import time
from tqdm import tqdm

from ngsolve import *
from ngsolve.webgui import Draw
from netgen.csg import *

from netgen.occ import *
from netgen.meshing import IdentificationType

from pathlib import Path

# my functions
from helper_functions import *
from dipole_poisson_solver_BLG import dipole_poisson_solver
from my_split import *


### unit-system: Angstrom = 1, eV = 1, e_charge = 1

### choices, see lammps metal system: https://docs.lammps.org/units.html

Angstrom = 1
eV = 1
e_charge = 1
Volt = 1

### independent

finestructure_constant = 0.0072973525643 # Finestructure constant, independent of unit-system   , https://en.wikipedia.org/wiki/Fine-structure_constant

### derived

hbar = 6.582119569 * pow(10.0, -4.0) ## eV * pico seconds, 1 pico second = 10^(-12) seconds

### units of time are in pico seconds, this implies units of velocity are in Angstrom / picoseconds

c_speed_of_light = 2.99792458 * pow(10.0, 6.0) ## Angstrom/pico seconds

pico_second = 1

nm = 10 ## 1nm = 10 Angstrom


hbar_times_c = 1973  # eV * Angstrom  # https://en.wikipedia.org/wiki/Planck_constant#Reduced_Planck_constant
h_times_c = 2.0 * np.pi * hbar_times_c

epsilon_0 = (e_charge * e_charge) / (2.0 * finestructure_constant * h_times_c)    # https://en.wikipedia.org/wiki/Vacuum_permittivity
mu_0 = 1.0 / (c_speed_of_light * epsilon_0)    # https://en.wikipedia.org/wiki/Vacuum_permeability
mu_0 /= c_speed_of_light ### don't divide by c^2 immediately

######################
######################
######################


### constants
a_lattice = 2.46

grapheneheight = 3.35 # Angstrom

blg_grapheneheight = grapheneheight ### see Jung & MacDonald PRB 89, 035405 (2014) #2.0 * grapheneheight

bilayer_graphene_area, bilayer_graphene_volume = blg_area_vol(a_lattice, blg_grapheneheight, Angstrom)
#print(blg_grapheneheight)


# eps tensor
inplane_hBn_bulk = 4.98 ### see https://www.nature.com/articles/s41699-018-0050-x 
outofplane_hBn_bulk = 3.03

inplane_graphene = 1.8 ### https://pubs.acs.org/doi/10.1021/nl303611v
outofplane_graphene = 3

dummy_idos_shape = np.zeros(2, np.int64)


dos_folder = "new_ps_helper_functions/JungMacDonald_energy_range_m200meV_200meV_increment_0.5meV_U_range_0_200_meV_increment_0.5meV_VSC"

#dos_folder = "new_ps_helper_functions/JungMacDonald_energy_range_m200meV_200meV_increment_0.25meV_U_range_0_200_meV_increment_0.25meV_VSC"




### blg idos

idos = np.load(dos_folder+"/idos_blg.npy")
    
band_gaps = np.load(dos_folder+"/band_gap.npy")

displacement_field_intervals = np.load(dos_folder+"/displacementfield_intervalls.npy")

energy_interval = np.load(dos_folder+"/energy_intervall.npy")


correct_prefactor = 1.0 / (bilayer_graphene_volume) ### this dos is already in units of 1/Energy, convert to density/Energy

idos = -idos * correct_prefactor


en_min = energy_interval[0] ## -0.2 eV
en_max = energy_interval[-1]  ## 0.2 eV

pot_min = displacement_field_intervals[0]  ## 0 eV
pot_max = displacement_field_intervals[-1] ## 0.2 eV


en_steps = idos.shape[1] 

e_incr = (en_max - en_min) / (en_steps - 1)

pot_steps = idos.shape[0] 

pot_incr = (pot_max - pot_min) / (pot_steps - 1)

en_intervall = np.linspace(en_min, en_max, en_steps)

en_incr = (en_max - en_min) / (en_steps - 1)

pot_intervall = np.linspace(pot_min, pot_max, pot_steps)

en_mesh, pot_mesh = np.meshgrid(en_intervall, pot_intervall)  # 2D grid for interpolation

begin_barycentricscipy_interpolation_time = time.time()

idos_interpolated = scipy.interpolate.LinearNDInterpolator(np.array([np.ravel(pot_mesh), np.ravel(en_mesh)]).T,
                                                           idos.reshape(pot_steps*en_steps))

end_barycentricscipy_interpolation_time = time.time()


# scan parameters

number_hBn = 2
first_hBn = 15 * nm
last_hBn = 50 * nm
hBns = np.rint(np.linspace(first_hBn, last_hBn, num=number_hBn))

number_it = 6

if number_it % 3 != 0:
    print('Error: number_it must be divisible by 3')

number_points_per_range = int(number_it/3)
charge_record_factor = 3
number_charges = int(number_it/charge_record_factor)

if number_it % charge_record_factor != 0:
    print('Error: number_it must be divisible by charge record factor')

min_V = -0.1
min_V_middle = -0.001
max_V_middle = -1 * min_V_middle
max_V = -1 * min_V
back_gates_left = np.linspace(min_V, min_V_middle, number_points_per_range, endpoint=False)
back_gates_middle = np.linspace(min_V_middle, max_V_middle, num=number_points_per_range, endpoint=False)
back_gates_right = np.linspace(max_V_middle, max_V, num=number_points_per_range)
back_gates = np.concatenate((back_gates_left, back_gates_middle, back_gates_right))


# general parameters

dipole_moment_debye = 40

sphere_radius = 3 # 3
benzen_a = 1.4 # Angstrom
benzen_diameter = 2 * benzen_a
dipole_position = 6 #0.02 * vac_box_width # basically 0, should be independent of box width


dipole_d = 3 * benzen_diameter 

hbn_maxh = 0.7
blg_maxh = 0.2

cut_percentage = 0.2
larger_cut_percentage = 0.7
lam = 100000

fem_n = 300
epsilon_z = 1e-4

if rank == 0:
    print('number of hBn thicknesses: ', number_hBn)
    print('number of back gates per hBn thickness: ', number_it)
    print('hBns [Angstrom]: ')
    print(hBns)


# results files folder
folder_name = "BLG_calculations_back_gate_range_" + str(max_V)


indices = my_split(np.arange(number_it), size)[rank]


# loop over hBn thicknesses
for hBn_height in hBns:
    if rank == 0:
        print('current hBn thickness: ', hBn_height)
    # create solver with correct hBn height
    height_bottom = hBn_height
    height_top = hBn_height
    vac_box_width = height_bottom * 15 # box width is 15 times hbn thickness, to minimize boundary effects
    max_h = vac_box_width/15

    solver = dipole_poisson_solver(dipole_moment_debye, dipole_d, dipole_position, inplane_hBn_bulk, outofplane_hBn_bulk, inplane_graphene, outofplane_graphene, epsilon_0)
    solver.build_geometry(vac_box_width, height_bottom, height_top, blg_grapheneheight, sphere_radius, max_h, hbn_maxh, blg_maxh)
    
    # loop over back gate voltages
    
    thread_seps = np.zeros((number_it, ), dtype=np.float64)
    seps = np.zeros((number_it, ), dtype=np.float64)

    thread_charges = np.zeros((number_charges, fem_n, fem_n), dtype=np.float64)
    charges = np.zeros((number_charges, fem_n, fem_n), dtype=np.float64)

    thread_potentials = np.zeros((number_charges, fem_n, fem_n), dtype=np.float64)
    potentials = np.zeros((number_charges, fem_n, fem_n), dtype=np.float64)

    thread_successes = np.zeros((number_it, ), dtype=bool)
    successes = np.zeros((number_it, ), dtype=bool)

    thread_top_cocs = np.zeros((number_charges, 2), dtype=np.float64)
    top_cocs = np.zeros((number_it, 2), dtype=np.float64)

    thread_bot_cocs = np.zeros((number_charges, 2), dtype=np.float64)
    bot_cocs = np.zeros((number_it, 2), dtype=np.float64)

    thread_sep_idxs = np.zeros((number_charges, ), dtype=int)
    sep_idxs = np.zeros((number_charges, ), dtype=int)

    thread_y_cuts = np.zeros((number_charges, fem_n), dtype=np.float64)
    y_cuts = np.zeros((number_charges, fem_n), dtype=np.float64)

    thread_Q_tops = np.zeros((number_it, ), dtype=np.float64)
    Q_tops = np.zeros((number_it, ), dtype=np.float64)

    thread_Q_bots = np.zeros((number_it, 1), dtype=np.float64)
    Q_bots = np.zeros((number_it, ), dtype=np.float64)
    
    thread_ps = np.zeros((number_it, 2), dtype=np.float64)
    ps = np.zeros((number_it, 2), dtype=np.float64)

    thread_total_charges = np.zeros((number_it, ), dtype=np.float64)
    total_charges = np.zeros((number_it, ), dtype=np.float64)

    thread_total_charges_squared = np.zeros((number_it, ), dtype=np.float64)
    total_charges_squared = np.zeros((number_it, ), dtype=np.float64)

    for ind in indices:
        V = back_gates[ind]

        solver.solve_without_BLG(V)
        solver.solve(fem_n, epsilon_z, idos_interpolated, pot_min, pot_max, en_min, en_max, cut_percentage, larger_cut_percentage, lam, use_abs=True, starting_alpha=0.01)

        thread_seps[ind] = solver.separation
        thread_successes[ind] = solver.sol.success
        thread_Q_tops[ind] = solver.Q_top
        thread_Q_bots[ind] = solver.Q_bot
        thread_ps[ind, :] = solver.p_total
        thread_total_charges[ind] = solver.total_charge
        thread_total_charges_squared[ind] = solver.total_charge_squared

        if ind % charge_record_factor == 0:
            thread_charges[int(ind/charge_record_factor), :, :] = solver.gfQ(solver.meshptsMid).reshape((solver.fem_n, solver.fem_n))
            thread_potentials[int(ind/charge_record_factor), :, :] = solver.gfu(solver.meshptsMid).reshape((solver.fem_n, solver.fem_n))
            thread_top_cocs[int(ind/charge_record_factor), :] = solver.center_of_charge_top
            thread_bot_cocs[int(ind/charge_record_factor), :] = solver.center_of_charge_bot
            thread_sep_idxs[int(ind/charge_record_factor)] = solver.sep_index
            thread_y_cuts[int(ind/charge_record_factor), :] = solver.cut_through_charge
        
        #iterations.append(solver.global_it[0])
        #ds.append(solver.d_vec)
        
        

    seps = comm.reduce(thread_seps, root=0)
    charges = comm.reduce(thread_charges, root=0)
    successes = comm.reduce(thread_successes, root=0)
    potentials = comm.reduce(thread_potentials, root=0)
    top_cocs = comm.reduce(thread_top_cocs, root=0)
    bot_cocs = comm.reduce(thread_bot_cocs, root=0)
    sep_idxs = comm.reduce(thread_sep_idxs, root=0)
    y_cuts = comm.reduce(thread_y_cuts, root=0)
    Q_tops = comm.reduce(thread_Q_tops, root=0)
    Q_bots = comm.reduce(thread_Q_bots, root=0)
    ps = comm.reduce(thread_ps, root=0)
    total_charges = comm.reduce(thread_total_charges, root=0)
    total_charges_squared = comm.reduce(thread_total_charges_squared, root=0)

    # save back gate scan results for current hBn thickness
    hbn_folder_name = "hBn_" + str(hBn_height)

    if rank == 0:
        #newpath = r'C:\Program Files\arbitrary' 
        #if not os.path.exists(newpath):
        #    os.makedirs(newpath)

        results_folder = Path.cwd() / folder_name / hbn_folder_name
        results_folder.mkdir(exist_ok=True)


        np.save(results_folder / 'back_gates', back_gates)
        np.save(results_folder / 'separations', seps)
        np.save(results_folder / 'sep_indxs', sep_idxs)
        np.save(results_folder / 'charges', charges)
        np.save(results_folder / 'potentials', potentials)
        np.save(results_folder / 'top_cocs', top_cocs)
        np.save(results_folder / 'bot_cocs', bot_cocs)
        np.save(results_folder / 'successes', successes)
        np.save(results_folder / 'y_cuts', y_cuts)
        np.save(results_folder / 'Q_top', Q_tops)
        np.save(results_folder / 'Q_bot', Q_bots)
        np.save(results_folder / 'dipole_moments', ps)
        np.save(results_folder / 'total_charges', total_charges)
        np.save(results_folder / 'total_charges_squared', total_charges_squared)

        # safe other parameters
        other_parameters = np.array([vac_box_width, cut_percentage, larger_cut_percentage, height_bottom, charge_record_factor])
        np.save(results_folder / 'box_width-cut_percentage-larger_cut_percentage', other_parameters)

