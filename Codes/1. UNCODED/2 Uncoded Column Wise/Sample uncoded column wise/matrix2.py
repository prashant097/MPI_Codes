
# Uncoded distributed matrix-vector multiplication
# Printed time is average over several iterations
# compile as: mpirun -n 4 python work.py

from mpi4py import MPI
from numpy import zeros
import numpy
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

M = 9000 # rows of A
N = 9000 # columns of A
n = size-1   # number of workers
niter = int(1)  #No of times the function is iterated

# # Partition the matrix A and store the partitions
# # at respective workers
# # Use tag = 0 for this communication phase
if rank == 0:
    A = numpy.random.rand(M,N)
    for i in range(1,n+1): # i is worker's rank 
        start_indx = (i-1)*N/n
        end_indx = i*N/n - 1
        comm.Send(numpy.ascontiguousarray(A[:,start_indx:end_indx+1]),dest=i,tag=0) # this comm phase has tag=1

if ((rank>=1) and (rank<=n)):
    Ai = numpy.empty([M,N/n])
    comm.Recv(Ai,source=0,tag=0)

if rank == 0:
    total_time = 0
    total_error = 0
for iter in range(0,niter):

    # # Generate and broadcast input column vector x to all workers
    if rank == 0:
        x = numpy.random.rand(N,1)
        # print x
    
    # # start time
    if rank == 0:
        tstart = time.time()
        for i in range(1,n+1):
	  start_indx =(i-1)*N/n
	  end_indx = i*N/n - 1 
	  comm.Send(x[start_indx:end_indx+1],dest=i,tag=1)
	
    
    # # Compute prodcuts Ai.x at workers
    # # and send results to master
    # # tag = 1 for this phase
    if rank != 0:
	xi = numpy.empty([N/n,1])
        comm.Recv(xi,source=0,tag=1)
        yi = Ai.dot(xi)
        comm.Send(yi,dest=0,tag=iter+1)
    
    # # Receive outputs from workers
    if rank == 0:
        y = numpy.zeros([M,1]) # formatted as a row vector
        info = MPI.Status()
        count_responses = 0
        #temp = numpy.empty([M/n,1])
	temp = numpy.empty([M,1])
        while count_responses < n:
            comm.Recv(temp,source=MPI.ANY_SOURCE,tag=iter+1,status=info)
            #i = info.Get_source() # rank of the sender
            #start_indx = (i-1)*M/n
            #end_indx = i*M/n - 1
            #y[start_indx:end_indx+1] = temp
             #y[M,1] += temp
	    y = y + temp
	    count_responses += 1
    
        time_this = time.time() - tstart
        total_time += time_this
        # # Check results: uncomment below lines to check if the result is correct
        total_error += numpy.linalg.norm(y-A.dot(x)) # Frobenius norm of the difference
    
    if rank > n:
        pass

if rank == 0:
    print 'time',total_time/niter
    print 'error',total_error/niter
    print 'Beta',straggling_mean

