# Coded distributed matrix-vector multiplication
# Using Polynomial Codes
# in the presence of stragglers.
# Printed time is average over several iterations
# compile as: mpiexec -np 4 python work.py


from mpi4py import MPI
import numpy
import time
import random

s = 12 # rows of A and B
r = 12 # coloumns of A
t = 12 # coloumns of B
st = 2  # number of stragglers, No need here
straggling_mean = 0.01 # mean of the exponential distribution
niter = int(1)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numpy.random.seed()

# Code specifications #
N = size-1    # number of workers
m1 = 3        # Number of divisions of A into m coloumns
n1 = 3       # Number of divisions of B into n coloumns
              # So, the number of total workers must be > (m1*n1)
Mp = int(r/m1) #No of coloumns in each Ap
Np = int(t/n1)

# k = n - st
# alpha = range(n)
# G = numpy.transpose(numpy.vander(alpha,k,increasing=True))
# Mp = int(M/k) # number of rows in a partition

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # Partition the matrix A, encode and store the partitions
# # at respective workers
# # Use tag = 0 for this communication phase
if rank == 0:
    # Original matrix A and B
    A = numpy.random.rand(s,r)
    B = numpy.random.rand(s,t)
    # print("A = ", A)
    # print("B = ", B)
    # A = numpy.random.randint(3,size=[M,N])
    # B = numpy.random.randint(3,size=[M,N])

    # Partition A into sub-matrices, say Ap #
    Ap = numpy.empty([m1,s,Mp])
    for i in range(0,m1): # i is data partition index
        start_indx = i*Mp
        end_indx = (i+1)*Mp - 1
        Ap[i,:,:] = A[:, start_indx:end_indx+1]
	print("Ap = ",i+1)
	print(Ap[i,:,:]) 
        # comm.Send(A[start_indx:end_indx+1],dest=i,tag=0) # this comm phase has tag=1
    print("\n")

    # Partition B into sub-matrices #
    Bp = numpy.empty([n1,s,Np])
    for i in range(0,n1): # i is data partition index
        start_indx = i*Np
        end_indx = (i+1)*Np - 1
        Bp[i,:,:] = B[:, start_indx:end_indx+1]
        print("Bp = ",i+1)
	print(Bp[i,:,:]) 
    print("\n")
    
    # To Encode Ap and Bp #
    X = numpy.linspace(1,2,num=N)
 
    for i in range(1,N+1): # i is worker's rank
	Apenc = numpy.zeros([s,Mp])
        x = X[i-1]
	for j in range(0,m1-1):
	  Apenc += (x**j)*Ap[j,:,:]
	print ("Apenc =",i)
        print (Apenc)
        comm.Send(Apenc[:,:],dest=i,tag=0) # Here, tag=0
    print("\n")

    for i in range(1,N+1): # i is worker's rank
	Bpenc = numpy.zeros([s,Np])
        x = X[i-1]
	for j in range(0,n1-1):
	  Bpenc += (x**(j*m1))*Bp[j,:,:]
	print ("Bpenc =",i)
        print (Bpenc)
        comm.Send(Bpenc[:,:],dest=i,tag=1) # Here, tag=1
    print("\n")
        

if ((rank>=1) and (rank<=N)):
    Apenci = numpy.empty([s,Mp])
    Bpenci = numpy.empty([s,Np])
    comm.Recv(Apenci,source=0,tag=0)
    comm.Recv(Bpenci,source=0,tag=1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if rank == 0:
    total_time = 0

for iter in range(0,niter):

    # # Generate and broadcast input column vector x to all workers
    tstart = time.time()
        
    # # Compute prodcuts Ai.x at workers
    # # and send results to master
    # # tag = 1 for this phase
    if rank != 0:
        # perform computation #
        yenci = numpy.transpose(Apenci).dot(Bpenci)
        # print('yenci= ',rank)
	# print(yenci)
        # Simulate straggling #
       # wait_time = numpy.random.exponential(straggling_mean)
       # time.sleep(wait_time)

        # communicate #
        comm.Send(yenci,dest=0,tag=iter+1)
    
  
    # # Receive outputs from workers
    if rank == 0:
        # # Order of G = ((m1.n1)*(r/m1)) x ((m1.n1)*(r/m1))
        # # yrecv  =    G   .  y_actual    # 
        # # (r.n1 x t/n1) = (r.n1 x r.n1) . (r.n1 x t/n1) # Order of Matrices
        yrecv = numpy.empty([r*n1,Np]) # array of row vectors
        Grecv = numpy.empty([r*n1,r*n1]) # rows are the encoding vectors
        I = numpy.identity(Mp)
	y=numpy.empty([r,t])
        info = MPI.Status()
        cr = 1   # count_response
        temp = numpy.empty([Mp,Np])
        while cr < (m1*n1+1):
            comm.Recv(temp,source=MPI.ANY_SOURCE,tag=iter+1,status=info)
            i = info.Get_source() # rank of the sender
	    start_indx = (cr-1)*Mp
            end_indx = cr*Mp - 1
            yrecv[start_indx:end_indx+1,:] = temp
	    x = X[i-1]
	    print ("cr,x,i = ",cr,x,i)
	    for j in range(m1*n1):
		Grecv[start_indx:end_indx+1,j*Mp:(j+1)*Mp]=I*(x**j)
            cr += 1
	    
	    
        
        # # Decode # #
	print("Grecv = ", Grecv)
	print("Order of yrecv = ", yrecv.shape)
        print("yrecv = ", yrecv)
        Grecv_inv = numpy.linalg.inv(Grecv)
	print("Order of Grecv_inv = ", Grecv_inv.shape)
	print("Grecv_inv = ", Grecv_inv)
        #yshaped = Grecv_inv.dot(yrecv)
	yshaped = (Grecv_inv).dot(yrecv)
        print('yshaped = \n',yshaped)
        for i in range(n1):
	    y[:,i*Np:(i+1)*Np]=yshaped[i*r:(i+1)*r,:]
	print("y = ",y)
        time_this = time.time() - tstart
        total_time += time_this
        y_actual = numpy.transpose(A).dot(B)
        print("y_actual = \n", y_actual)
        # Check results: uncomment below lines to check if the result is correct
        print("Error : ",numpy.linalg.norm(y-y_actual)) # Frobenius norm of the difference

        # Receive from other nodes as well #
        while cr < N:
            comm.Recv(temp,source=MPI.ANY_SOURCE,tag=iter+1,status=info)
            cr += 1
    
    if rank > N:
        pass

if rank == 0:
    print ("Total_Time = ",total_time/niter)


