#******************************************************************************
# FILE: mpi_mm.py
# DESCRIPTION:
#   MPI Matrix Multiply - Python Version
#   In this code, the master task distributes a matrix multiply
#   operation to numtasks-1 worker tasks.
#   NOTE:  Python arrays are row-major order.
# AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
#   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
#   Adapted to Python by SH (10/17)
#
# Run by typing:
# mpirun -np 4 python mpi_mm.py
#
#*****************************************************************************/
from mpi4py import MPI
import numpy as np

NRA = 62                 # number of rows in matrix A
NCA = 15                 # number of columns in matrix A
NCB = 7                  # number of columns in matrix B
MASTER = 0               # taskid of first task
FROM_MASTER = 1          # setting a message type
FROM_WORKER = 2          # setting a message type

a=np.zeros((NRA,NCA))
b=np.zeros((NCA,NCB))
c=np.zeros((NRA,NCB))

comm = MPI.COMM_WORLD
taskid = comm.Get_rank()
numtasks = comm.Get_size()
if numtasks < 2:
    print("Need at least two MPI tasks. Quitting...")
    comm.Abort()

numworkers = numtasks-1

#**************************** master task ************************************/
if taskid == MASTER:
    print("mpi_mm has started with %d tasks." % numtasks)
    print("Initializing arrays...");
    for i in range (NRA):
        for j in range (NCA):
            a[i,j]= i+j
    for i in range (NCA):
        for j in range (NCB):
            b[i,j]= i*j

    # Send matrix data to the worker tasks
    averow = NRA//numworkers
    extra = NRA%numworkers
    offset = 0
    for dest in range(1,numworkers+1):
        rows = averow
        if dest <= extra:
            rows+=1

        print("Sending %d rows to task %d offset=%d" % (rows,dest,offset))
        comm.send(offset, dest=dest, tag=FROM_MASTER)
        comm.send(rows, dest=dest, tag=FROM_MASTER)
        comm.Send(a[offset:offset+rows,:], dest=dest, tag=FROM_MASTER)
        comm.Send(b, dest=dest, tag=FROM_MASTER)
        offset += rows

    # Receive results from worker tasks
    for i in range(1,numworkers+1):
        source = i
        offset = comm.recv(source=source, tag=FROM_WORKER)
        rows = comm.recv(source=source, tag=FROM_WORKER)
        comm.Recv([c[offset:,:],rows*NCB,MPI.DOUBLE], source=source, tag=FROM_WORKER)
        print("Received results from task %d" % source)

# Print results
    print("******************************************************")
    print("Result Matrix:")
    print(c)
    print("******************************************************")
    print("Done.")


#**************************** worker task ************************************/
elif taskid > MASTER:

    offset = comm.recv(source=MASTER, tag=FROM_MASTER)
    rows = comm.recv(source=MASTER, tag=FROM_MASTER)
    comm.Recv([a,rows*NCA,MPI.DOUBLE], source=MASTER, tag=FROM_MASTER)
    comm.Recv(b, source=MASTER, tag=FROM_MASTER)

    for k in range(NCB):
        for i in range(rows):
            c[i,k] = 0.0
            for j in range(NCA):
                c[i,k] += a[i,j] * b[j,k]

    comm.send(offset, dest=MASTER, tag=FROM_WORKER)
    comm.send(rows, dest=MASTER, tag=FROM_WORKER)
    comm.Send(c[:rows,:], dest=MASTER, tag=FROM_WORKER)

