"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cimport numpy as cnp
cimport cython

from libc.math cimport sin, cos, exp, log, sqrt, M_PI
from libc.stdlib cimport rand, RAND_MAX
from libcpp.random cimport mt19937, uniform_real_distribution

import random

from cython.parallel import prange, parallel
cimport openmp
from openmp cimport omp_get_thread_num, omp_get_num_threads, omp_get_max_threads, omp_set_num_threads;


def log_csv(folderpath, filename, type, size, steps, temp, order, nthreads, runtime):
    """
    Arguments:
      folderpath (string) = the path to the folder where the csv file will be saved;
      filename (string) = the name of the csv file;
      type (string) = the type of the simulation;
      size (int) = the size of the lattice;
      steps (int) = the number of Monte Carlo steps;
      temp (float) = the reduced temperature;
      order (float) = the order parameter;
      runtime (float) = the runtime of the simulation.
    Description:
      Function to save the data to a csv file.
    Returns:
      NULL
    """
    
    with open(folderpath + '/' + filename, 'a') as f:
        f.write(f"{type},{size},{steps},{temp},{order},{nthreads},{runtime}\n")

#=======================================================================
cdef double[:,::1] initdat(int nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    cdef double[:,::1] arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    arr = np.array(arr)

    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double one_energy(double[:,::1] arr, int ix, int iy, int nmax) nogil:
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    cdef:
        double en, ang, cos_ang = 0.0
        double cell_value = arr[ix,iy]
        int ixp = (ix+1)%nmax # These are the coordinates
        int ixm = (ix-1)%nmax # of the neighbours
        int iyp = (iy+1)%nmax # with wraparound
        int iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = cell_value-arr[ixp,iy]
    cos_ang = cos(ang)
    en += 0.5*(1.0 - 3.0*cos_ang**2)
    ang = cell_value-arr[ixm,iy]
    cos_ang = cos(ang)
    en += 0.5*(1.0 - 3.0*cos_ang**2)
    ang = cell_value-arr[ix,iyp]
    cos_ang = cos(ang)
    en += 0.5*(1.0 - 3.0*cos_ang**2)
    ang = cell_value-arr[ix,iym]
    cos_ang = cos(ang)
    en += 0.5*(1.0 - 3.0*cos_ang**2)

    return en
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double all_energy(double[:, ::1] arr, int nmax):
    """
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
    Returns:
      enall (float) = reduced energy of lattice.
    """
    cdef double enall = 0.0
    cdef int i, j

    for i in prange(nmax, nogil=True):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)

    return enall

#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_order(double[:,::1] arr, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    cdef double[:,::1] Qab = np.zeros((3, 3), dtype=np.float64)
    cdef double[:,::1] delta = np.eye(3, dtype=np.float64)

    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    cdef double[:,:,::1] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)

    cdef:
        int a,b,i,j = 0

    for a in range(3):                                            # not parallelising is faster here
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
            Qab[a,b] = Qab[a,b]/(2*nmax*nmax)

    cdef double[::1] eigenvalues = np.zeros(3, dtype=np.float64)

    eigenvalues = np.linalg.eigvalsh(Qab)                           # np.linalg.eigvalsh is faster than np.linalg.eigvals, and is also c array contiguous
    return max(eigenvalues)
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double MC_step(double[:,::1] arr, double Ts, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    cdef double scale = 0.1+Ts
    cdef double accept = 0
    cdef long[:,::1] xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    cdef long[:,::1] yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    cdef double[:,::1] aran = np.random.normal(scale=scale, size=(nmax,nmax))

    cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0, 1.0)
    cdef mt19937 generator = mt19937()

    cdef:
        int i,j,ix,iy = 0
        double ang,en0,en1,boltz = 0.0
        int[::1] odd_row_indices = np.arange(1, nmax, 2)
        int[::1] even_row_indices = np.arange(0, nmax, 2)


    # update odd rows
    for i in prange(nmax//2, nogil=True):
        for j in range(nmax):
            ix = xran[i*2+1,j]
            iy = yran[i*2+1,j]
            ang = aran[i*2+1,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = exp( -(en1 - en0) / Ts )                     # np.exp is slow, use libc.math.exp instead for single numbers

                if boltz >= dist(generator):                         # random.random or np.random.uniform is not allowed in nogil, use thread safe c++ 11 random instead
                    accept += 1
                else:
                    arr[ix,iy] -= ang

    # update even rows
    for i in prange(nmax//2 + nmax%2, nogil=True):
        for j in range(nmax):
            ix = xran[i*2,j]
            iy = yran[i*2,j]
            ang = aran[i*2,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = exp( -(en1 - en0) / Ts )

                if boltz >= dist(generator):
                    accept += 1
                else:
                    arr[ix,iy] -= ang

    return accept/(nmax*nmax)
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """

    omp_set_num_threads(4)
    total_threads = omp_get_max_threads()

    cdef:
        double runtime, c_temp = 0.0
        int it, c_nsteps, c_nmax, c_pflag = 0
    
    c_temp = float(temp)
    c_nsteps = int(nsteps)
    c_nmax = int(nmax)
    c_pflag = int(pflag)

    # Create and initialise lattice
    cdef double[:,::1] lattice = initdat(c_nmax)
    # Plot initial frame of lattice
    plotdat(lattice,c_pflag,c_nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    cdef double[::1] energy = np.zeros(c_nsteps+1,dtype=np.float64)
    cdef double[::1] ratio = np.zeros(c_nsteps+1,dtype=np.float64)
    cdef double[::1] order = np.zeros(c_nsteps+1,dtype=np.float64)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,c_nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,c_nmax)
    
    # Begin doing and timing some MC steps.
    MC_initial = 0
    MC_final = 0
    all_initial = 0
    all_final = 0
    order_initial = 0
    order_final = 0

    MC_times = np.zeros(c_nsteps,dtype=np.float64)
    all_times = np.zeros(c_nsteps,dtype=np.float64)
    order_times = np.zeros(c_nsteps,dtype=np.float64)

    initial = time.time()
    for it in range(1,c_nsteps+1):
        MC_initial = time.time()
        ratio[it] = MC_step(lattice,c_temp,c_nmax)
        MC_final = time.time()
        MC_times[it-1] = MC_final-MC_initial

        all_initial = time.time()
        energy[it] = all_energy(lattice,c_nmax)
        all_final = time.time()
        all_times[it-1] = all_final-all_initial

        order_initial = time.time()
        order[it] = get_order(lattice,c_nmax)
        order_final = time.time()
        order_times[it-1] = order_final-order_initial

    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program,c_nmax,c_nsteps,c_temp,order[c_nsteps-1],runtime))
    log_csv("../../log", "log.csv", "cython_omp", nmax, nsteps, temp, order[c_nsteps-1], total_threads, runtime)
    # Plot final frame of lattice and generate output file
    # savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,c_pflag,c_nmax)
    print("MC time: ", MC_times.sum())
    print("All time: ", all_times.sum())
    print("Order time: ", order_times.sum())
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
# if __name__ == '__main__':
#     if int(len(sys.argv)) == 5:
#         PROGNAME = sys.argv[0]
#         ITERATIONS = int(sys.argv[1])
#         SIZE = int(sys.argv[2])
#         TEMPERATURE = float(sys.argv[3])
#         PLOTFLAG = int(sys.argv[4])
#         main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
#     else:
#         print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
