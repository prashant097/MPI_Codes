# Uncoded distributed matrix-vector multiplication
# Printed time is average over several iterations
# compile as: mpirun -n 4 python work.py


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from mpi4py import MPI
import numpy
import time
import xlsxwriter


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

SZ=[12,30,90,300,900]
SW=[5,10,15,20]
X1=5    # used only for graph 
zi=0

workbook = xlsxwriter.Workbook('poly_data.xlsx')
worksheet1 = workbook.add_worksheet("Sheet_time") 
worksheet2 = workbook.add_worksheet("Sheet_error")

s = 2    # number of stragglers
straggling_mean = 0.01 # mean of the exponential distribution
numpy.random.seed()

Y=numpy.empty([5,4])
Yn=numpy.empty([5,4])
for Z in SZ :
    z1i=0
    for W in SW :	
	r = Z # rows of A
	s = Z # columns of A and rowsin B
	t = Z
	m=3
	n=3

	N = W-1   # number of workers
	niter = int(1)

	# # Partition the matrix A and store the partitions
	# # at respective workers
	# # Use tag = 0 for this communication phase
	if rank == 0:
	    #x=numpy.arange(1,N+1)
	    A = numpy.random.rand(s,r)
	    B = numpy.random.rand(s,t)
	    #A=numpy.array([[1,2],[3,1]])
	    #B=numpy.array([[4,1],[2,3]] )
	    #print('A=\n',A)
	    #print('B=\n',B)
	    Ap=numpy.empty([m,s,r/m])
	    Bp=numpy.empty([n,s,t/n])
	    Aenc=numpy.zeros([N,s,r/m])
	    Benc=numpy.zeros([N,s,t/n])
	    x= numpy.linspace(2.0, 3.0, num=N)
	    for i in range(1,m+1): # i is worker's rank 
		start_indx = (i-1)*r/m
		end_indx = i*r/m - 1
		Ap[i-1,:,:]= A[:,start_indx:end_indx+1]
	    for i in range(1,n+1): # i is worker's rank 
		start_indx = (i-1)*t/n
		end_indx = i*t/n - 1
		Bp[i-1,:,:]= B[:,start_indx:end_indx+1]
	    for i in range(1,N+1): # i is worker's rank 
		for j in range(m):
		    Aenc[i-1,:,:] +=Ap[j,:,:]*(x[i-1]**j)
		comm.Send(numpy.ascontiguousarray(Aenc[i-1,:,:]),dest=i,tag=0)

		for j in range(n):
		    a=j*m
		    Benc[i-1,:,:] +=Bp[j,:,:]*(x[i-1]**a)
		comm.Send(numpy.ascontiguousarray(Benc[i-1,:,:]),dest=i,tag=1)
	    #print('Aenc=\n',Aenc)
	    #print('Benc=\n',Benc)
	if ((rank>=1) and (rank<=N)):
	    Ai = numpy.empty([s,r/m])
	    Bi = numpy.empty([s,t/n])
	    comm.Recv(Ai,source=0,tag=0)
	    comm.Recv(Bi,source=0,tag=1)
	if rank != 0:
	    print (Ai)
	if rank == 0:
	    total_time = 0
	    total_error = 0
	for iter in range(0,niter):

	    # # start time
	    if rank == 0:
		tstart = time.time()
		    
	    # # Compute prodcuts Ai.x at workers
	    # # and send results to master
	    # # tag = 1 for this phase
	    if rank != 0:
		yi = numpy.transpose(Ai).dot(Bi)
		comm.Send(yi,dest=0,tag=iter+1)
		#print('yi=\n',yi)
	    # # Receive outputs from workers
	    if rank == 0:
		y = numpy.empty([n*r,t/n]) # formatted as a row vector(mn*t/n)
		info = MPI.Status()
		cr = 1
		
		C=numpy.empty([r,t])
		temp = numpy.empty([r/m,t/n])
		I=numpy.identity(r/m)
		G=numpy.empty([n*r,n*r])
		while cr < m*n+1:
		    comm.Recv(temp,source=MPI.ANY_SOURCE,tag=iter+1,status=info)
		    i = info.Get_source() # rank of the sender
		    start_indx = (cr-1)*r/m
		    end_indx = cr*r/m - 1
		    y[start_indx:end_indx+1,:] = temp
		    #print('i=',i)
		    for j in range(m*n):
			G[start_indx:end_indx+1,j*r/m:(j+1)*r/m]=I*x[i-1]**j
		    cr += 1
		Grecv_inv = numpy.linalg.inv(G)
		#print(numpy.shape(yrecv))
		#print('G=\n',G)
		#print('Ginv=\n',Grecv_inv)
		#print('y=\n',y)
		yshaped = Grecv_inv.dot(y)
		#print('yshaped=\n',yshaped)
		for i in range(n):
		    C[:,i*t/n:(i+1)*t/n]=yshaped[i*r:(i+1)*r,:]
		
		#print('C=\n',C)
		#print('AT.B=\n',numpy.transpose(A).dot(B))

		time_this = time.time() - tstart
		total_time += time_this
		# # Check results: uncomment below lines to check if the result is correct
		total_error +=numpy.linalg.norm(C-numpy.transpose(A).dot(B)) # Frobenius norm of the difference


            if rank > n:
                pass

        if rank == 0:
            Y[zi,z1i] =total_time/niter
            Yn[zi,z1i] =total_error/niter
            worksheet1.write(zi, z1i, Y[zi,z1i])
            worksheet2.write(zi, z1i, Yn[zi,z1i])
            z1i+=1
    zi=zi+1
if rank==0:
    print('X=',X1,'Y=',Y)
    #print('x=',numpy.shape(X),'y=',numpy.shape(Y))
    workbook.close() 

