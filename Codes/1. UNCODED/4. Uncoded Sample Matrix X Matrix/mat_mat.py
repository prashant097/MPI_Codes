# Coded distributed matrix-vector multiplication
# in the presence of stragglers.
# Printed time is average over several iterations
# compile as: mpiexec -np 4 python work.py

from mpi4py import MPI
import numpy
import time

M = 180 # rows of A
N = 10 # columns of A
s = 2    # number of stragglers
straggling_mean = 0.01 # mean of the exponential distribution
niter = int(10)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numpy.random.seed()

# Code specifications #
n = size-1   # number of workers
k = n - s
alpha = range(n)
G = numpy.transpose(numpy.vander(alpha,k,increasing=True))
Mp = int(M/k) # number of rows in a partition

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Partition the matrix A, encode and store the partitions
# # at respective workers
# # Use tag = 0 for this communication phase
if rank == 0:
    # Original matrix A #
    A = numpy.random.rand(M,N)
    # A = numpy.random.randint(3,size=[M,N])
    
    # Partition A into sub-matrices #
    Ap = numpy.empty([k,Mp,N])
    for i in range(0,k): # i is data partition index
        start_indx = i*Mp
        end_indx = (i+1)*Mp - 1
        Ap[i,:,:] = A[start_indx:end_indx+1] 
        # comm.Send(A[start_indx:end_indx+1],dest=i,tag=0) # this comm phase has tag=1
    
    # Encode Ap #
    Aenc = numpy.einsum('ijk,i...',Ap,G)
    for i in range(1,n+1): # i is worker's rank 
        comm.Send(Aenc[i-1,:,:],dest=i,tag=0) # this comm phase has tag=0

if ((rank>=1) and (rank<=n)):
    Aenci = numpy.empty([Mp,N])
    comm.Recv(Aenci,source=0,tag=0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
x = numpy.empty([N,N])

if rank == 0:
    total_time = 0

for iter in range(0,niter):

    # # Generate and broadcast input column vector x to all workers
    if rank == 0:
        x = numpy.random.rand(N,N)
        tstart = time.time()
    
    comm.Bcast(x,root=0)
    
    # # Compute prodcuts Ai.x at workers
    # # and send results to master
    # # tag = 1 for this phase
    if rank != 0:
        # perform computation #
        yenci = Aenci.dot(x)
        
        # Simulate straggling #
        wait_time = numpy.random.exponential(straggling_mean)
        time.sleep(wait_time)

        # communicate #
        comm.Send(yenci,dest=0,tag=iter+1)
    
    # # Receive outputs from workers
    if rank == 0:
        # # yrecv  =    G   .  y    # 
        # # k x Mp =  k x k  k x Mp #
        yrecv = numpy.empty([k,Mp,N]) # array of row vectors
        Grecv = numpy.empty([k,k]) # rows are the encoding vectors

        info = MPI.Status()
        count_responses = 0
        temp = numpy.empty([Mp,N])
        while count_responses < k:
            comm.Recv(temp,source=MPI.ANY_SOURCE,tag=iter+1,status=info)
            i = info.Get_source() # rank of the sender
            yrecv[count_responses,:,:] = temp
            Grecv[count_responses,:] = G[:,i-1] 
            count_responses += 1


        # # Decode # # 
        Grecv_inv = numpy.linalg.inv(Grecv)
        #yshaped = Grecv_inv.dot(yrecv)
	yshaped = numpy.einsum("ij,jkl->ikl",Grecv_inv,yrecv)
        y = yshaped.reshape(M,N)

        time_this = time.time() - tstart
        total_time += time_this
        
        # Check results: uncomment below lines to check if the result is correct
        print(numpy.linalg.norm(y-A.dot(x))/numpy.sqrt(M)) # Frobenius norm of the difference

        # Receive from other nodes as well #
        while count_responses < n:
            comm.Recv(temp,source=MPI.ANY_SOURCE,tag=iter+1,status=info)
            count_responses += 1
    
    if rank > n:
        pass

if rank == 0:
    print total_time/niter
