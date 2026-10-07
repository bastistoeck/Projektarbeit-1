
import os
import multiprocessing

jupyter_true = True #True ## cluster variable

if jupyter_true == True: 

    import ipyparallel as ipp
    # Number of MPI engines (processes)
    num_engines =  2 #4  #10
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

    %autopx

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




def my_split(array, size): ### split array into lengths of size

    if array.shape[0] % size == 0: ### divisible

        return np.split(array, size)

    else: ## shave the end off then reattach:

        remainder = array.shape[0] % size

        shaved_array = array[:-remainder]

        split_array = np.split(shaved_array, size)

        split_array[-1] = np.concatenate([split_array[-1], array[-remainder:] ])

        return split_array
        



indices = my_split(np.arange(n_sweep), size)[rank]




#### cycle

u, v = fes.TnT()
a = BilinearForm(epsilon_0 * epsr * grad(u) * grad(v) * dx).Assemble() 

inv = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky")

for ind in indices:


	### DO THINGS
