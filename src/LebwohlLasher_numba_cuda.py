import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import cuda, njit, float64
from math import cos, sin, pi, exp, sqrt, ceil
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_uniform_float64


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


def to_device(arr):
    return cuda.to_device(arr)

def from_device(arr_device):
    return arr_device.copy_to_host()


# Initialize data
def initdat(nmax):
    arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    return arr.astype(np.float64)

# Plot data
def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))
    if pflag == 1:
        # Placeholder for energy coloring
        cols = arr  # Replace with actual energy data
        mpl.rc('image', cmap='rainbow')
    elif pflag == 2:
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
    
    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols, **quiveropts)
    ax.set_aspect('equal')
    plt.show()


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


@cuda.jit
def one_energy_kernel(arr, en, nmax):
    i, j = cuda.grid(2)
    if i < nmax and j < nmax:
        en_val = 0.0
        ixp = (i + 1) % nmax
        ixm = (i - 1) % nmax
        iyp = (j + 1) % nmax
        iym = (j - 1) % nmax

        angles = cuda.local.array(4, float64)
        angles[0] = arr[i, j] - arr[ixp, j]
        angles[1] = arr[i, j] - arr[ixm, j]
        angles[2] = arr[i, j] - arr[i, iyp]
        angles[3] = arr[i, j] - arr[i, iym]

        for k in range(4):
            cos_ang = cos(angles[k])
            en_val += 0.5 * (1.0 - 3.0 * cos_ang ** 2)
        
        en[i, j] = en_val



@njit(["double(double[:,:], int64, int64, int64)"], cache=True)          # using cache=True removes just in time compilation for later runs
def one_energy(arr,ix,iy,nmax):
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
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]                # compute cos_ang to avoid calling cos twice
    cos_ang = cos(ang)
    en += 0.5*(1.0 - 3.0*cos_ang**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    cos_ang = cos(ang)
    en += 0.5*(1.0 - 3.0*cos_ang**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    cos_ang = cos(ang)
    en += 0.5*(1.0 - 3.0*cos_ang**2)
    ang = arr[ix,iy]-arr[ix,iym]
    cos_ang = cos(ang)
    en += 0.5*(1.0 - 3.0*cos_ang**2)
    return en

@njit(["double(double[:,:], int64)"], cache=True)
def all_energy(arr,nmax):
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
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall


@cuda.jit
def all_energy_kernel(en, total_en, nmax):
    i, j = cuda.grid(2)
    if i < nmax and j < nmax:
        cuda.atomic.add(total_en, 0, en[i, j])



@njit(["double[:,:](double, int64)"], cache=True)
def rand_normal(scale, nmax):
    """
    Arguments:
      scale (float) = scale factor for normal distribution;
      nmax (int) = side length of square lattice.
    Description:
      Function to generate a 2D array of normally distributed
      random numbers. This is to replace the numpy.random.normal which
      is not supported by number njit.
  Returns:
    aran (float(nmax,nmax)) = array of random numbers.
    """
    aran = np.zeros((nmax,nmax))
    for i in range(nmax):
        for j in range(nmax):
            aran[i,j] = np.sqrt(-2*np.log(np.random.uniform(0.0,1.0)))*np.cos(2*np.pi*np.random.uniform(0.0,1.0)) * scale
    return aran


@njit(["double(double[:,:], int64)"], cache=True)
def get_order(arr,nmax):
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
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues = np.linalg.eigvalsh(Qab)
    return eigenvalues.max()

@cuda.jit
def MC_step_kernel(arr, Ts, accept, nmax, scale, xran, yran, aran, states):
    # checkerboard update to avoid race conditions, this loop updates the "0" cells
    i, j = cuda.grid(2)
    if i < nmax and j < nmax:
        if (i % 2 + j % 2) % 2 == 0:
            idx = i * arr.shape[1] + j
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]

            # one_energy function
            en0 = 0.0
            ixp = (ix+1)%nmax # These are the coordinates
            ixm = (ix-1)%nmax # of the neighbours
            iyp = (iy+1)%nmax # with wraparound
            iym = (iy-1)%nmax #

            ang = arr[ix,iy]-arr[ixp,iy]                # compute cos_ang to avoid calling cos twice
            cos_ang = cos(ang)
            en0 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ixm,iy]
            cos_ang = cos(ang)
            en0 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ix,iyp]
            cos_ang = cos(ang)
            en0 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ix,iym]
            cos_ang = cos(ang)
            en0 += 0.5*(1.0 - 3.0*cos_ang**2)

            # update arr
            arr[ix, iy] += ang

            # one_energy function
            en1 = 0.0
            ixp = (ix+1)%nmax # These are the coordinates
            ixm = (ix-1)%nmax # of the neighbours
            iyp = (iy+1)%nmax # with wraparound
            iym = (iy-1)%nmax #

            ang = arr[ix,iy]-arr[ixp,iy]                # compute cos_ang to avoid calling cos twice
            cos_ang = cos(ang)
            en1 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ixm,iy]
            cos_ang = cos(ang)
            en1 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ix,iyp]
            cos_ang = cos(ang)
            en1 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ix,iym]
            cos_ang = cos(ang)
            en1 += 0.5*(1.0 - 3.0*cos_ang**2)

            if en1 <= en0:
                cuda.atomic.add(accept, 0, 1)
            else:
                boltz = exp(-(en1 - en0) / Ts)
                rand_val = xoroshiro128p_uniform_float64(states, idx)
                if boltz >= (rand_val) :
                    cuda.atomic.add(accept, 0, 1)
                else:
                    arr[ix, iy] -= ang

    # checkerboard update to avoid race conditions, this loop updates the "1" cells
    i, j = cuda.grid(2)
    if i < nmax and j < nmax:
        if (i % 2 + j % 2) % 2 == 1:
            idx = i * arr.shape[1] + j
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]

            # one_energy function
            en0 = 0.0
            ixp = (ix+1)%nmax # These are the coordinates
            ixm = (ix-1)%nmax # of the neighbours
            iyp = (iy+1)%nmax # with wraparound
            iym = (iy-1)%nmax #

            ang = arr[ix,iy]-arr[ixp,iy]                # compute cos_ang to avoid calling cos twice
            cos_ang = cos(ang)
            en0 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ixm,iy]
            cos_ang = cos(ang)
            en0 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ix,iyp]
            cos_ang = cos(ang)
            en0 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ix,iym]
            cos_ang = cos(ang)
            en0 += 0.5*(1.0 - 3.0*cos_ang**2)

            # update arr
            arr[ix, iy] += ang

            # one_energy function
            en1 = 0.0
            ixp = (ix+1)%nmax # These are the coordinates
            ixm = (ix-1)%nmax # of the neighbours
            iyp = (iy+1)%nmax # with wraparound
            iym = (iy-1)%nmax #

            ang = arr[ix,iy]-arr[ixp,iy]                # compute cos_ang to avoid calling cos twice
            cos_ang = cos(ang)
            en1 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ixm,iy]
            cos_ang = cos(ang)
            en1 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ix,iyp]
            cos_ang = cos(ang)
            en1 += 0.5*(1.0 - 3.0*cos_ang**2)
            ang = arr[ix,iy]-arr[ix,iym]
            cos_ang = cos(ang)
            en1 += 0.5*(1.0 - 3.0*cos_ang**2)

            if en1 <= en0:
                cuda.atomic.add(accept, 0, 1)
            else:
                boltz = exp(-(en1 - en0) / Ts)
                rand_val = xoroshiro128p_uniform_float64(states, idx)
                if boltz >= (rand_val) :
                    cuda.atomic.add(accept, 0, 1)
                else:
                    arr[ix, iy] -= ang




def main(program, nsteps, nmax, temp, pflag):
    lattice = initdat(nmax)
    d_lattice = to_device(lattice)
    plotdat(lattice, pflag, nmax)

    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)

    energy[0] = all_energy(lattice, nmax)     # cpu calculation as its only once at start
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    threadsperblock = (16, 16)
    blockspergrid_x = ceil(nmax / threadsperblock[0])
    blockspergrid_y = ceil(nmax / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    num_states = blockspergrid_x * blockspergrid_y * threadsperblock[0] * threadsperblock[1]
    states = create_xoroshiro128p_states(num_states, seed=1)

    initial = time.time()
    for it in range(1, nsteps + 1):
        # Prepare random numbers
        xran = np.random.randint(0, high=nmax, size=(nmax, nmax)).astype(np.int32)
        yran = np.random.randint(0, high=nmax, size=(nmax, nmax)).astype(np.int32)
        aran = rand_normal(0.1 + temp, nmax).astype(np.float64)

        d_xran = cuda.to_device(xran)
        d_yran = cuda.to_device(yran)
        d_aran = cuda.to_device(aran)

        accept = cuda.to_device(np.array([0], dtype=np.float64))

        MC_step_kernel[blockspergrid, threadsperblock](d_lattice, temp, accept, nmax, 0.1 + temp, d_xran, d_yran, d_aran, states)
        ratio[it] = accept.copy_to_host()[0] / (nmax ** 2)

        # Calculate energy
        d_en = cuda.device_array((nmax, nmax), dtype=np.float64)
        one_energy_kernel[blockspergrid, threadsperblock](d_lattice, d_en, nmax)
        total_en = cuda.to_device(np.array([0.0], dtype=np.float64))
        all_energy_kernel[blockspergrid, threadsperblock](d_en, total_en, nmax)
        energy[it] = total_en.copy_to_host()[0]

        # Calculate order parameter cpu as it is relatively fast
        lattice = from_device(d_lattice)
        order[it] = get_order(lattice, nmax)

    final = time.time()
    runtime = final - initial

    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:.3f}: Order: {order[nsteps - 1]:.3f}, Time: {runtime:.6f} s")
    log_csv("../log", "log.csv", "cuda", nmax, nsteps, temp, order[nsteps-1], num_states, runtime)
    lattice = from_device(d_lattice)
    plotdat(lattice, pflag, nmax)


if __name__ == '__main__':
    if len(sys.argv) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
