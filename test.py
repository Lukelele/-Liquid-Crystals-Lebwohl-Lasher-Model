from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


test_arr = np.array([[1,2,3,4], 
                     [5,6,7,8], 
                     [9,10,11,12], 
                     [13,14,15,16]], dtype='i')


if rank == 0:
    test_arr[3] = [0,0,0,0]
elif rank == 1:
    test_arr[2] = [1,1,1,1]
elif rank == 2:
    test_arr[1] = [2,2,2,2]
elif rank == 3:
    test_arr[0] = [3,3,3,3]


local_data = test_arr[3-rank]

recv_buffer = np.empty([size, 4], dtype='i')

comm.Allgather(local_data, recv_buffer)
comm.Barrier()

if rank == 0:
    print("Final combined array:")
    print(recv_buffer)