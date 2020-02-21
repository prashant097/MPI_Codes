# Uncoded distributed matrix-vector multiplication
# Printed time is average over several iterations
# compile as: mpirun -n 4 python work.py


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from mpi4py import MPI
import numpy
import time
import xlwt
from xlwt import Workbook

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
SZ=[150,180,300,600,900]#different sizes
SW=[8,10]#different number of workers

zi=0 #zi and zi1 are defined to store the data in excel sheet
wb = Workbook() 

#s = 2    # number of stragglers
straggling_mean = 0.04 # mean of the exponential distribution
numpy.random.seed()

  
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet time') 
sheet2 = wb.add_sheet('Sheet error') 
Y=numpy.empty([len(SZ),len(SW)]) #size,workers
Yn=numpy.empty([len(SZ),len(SW)])
for Z in SZ :
    z1i=0
    for W in SW :

	s = Z # rows of A and B
	r = Z # columns of A
	t = Z # columns of B
	N = W   # number of workers
	m=1
	n=1
	p=3
	K=p*m*n+p-1
	straggling_mean = 0.01 # mean of the exponential distribution
	#niter = int(1)


	if rank == 0:
	    A = numpy.random.rand(s,r)
	    B = numpy.random.rand(s,t)
	    x = numpy.linspace(1,2,N)
	   # At= numpy.empty([p,m,s/p,r/m])
	   # Bt= numpy.empty([p,n,s/p,t/n])
	 
	#ENCODE A AND B:
	    
	    Aenc=numpy.zeros((N,s/p,r/m))
	    Benc=numpy.zeros((N,s/p,t/n))
	    for h in range(N) :
		
		for i in range(p):
		    start_r = i*s/p
		    end_r = (i+1)*s/p
		    for j in range(m):
			start_c = j*r/m
		        end_c = (j+1)*r/m
		       # At[i,j,:,:]=A[start_r:end_r,start_c:end_c]
			Aenc[h,:,:]+=A[start_r:end_r,start_c:end_c]*(x[h]**(i+j*p))
		    for k in range(n):
			start_c = k*t/n
		        end_c = (k+1)*t/n
		       # Bt[i,k,:,:]=B[start_r:end_r,start_c:end_c]   
			Benc[h,:,:]+=B[start_r:end_r,start_c:end_c]*(x[h]**(p-1-i+k*p*m))
		comm.Send(Aenc[h,:,:],dest=h+1,tag=0)
		comm.Send(Benc[h,:,:],dest=h+1,tag=1)
	    tstart = time.time()
	#AT WORKER NODES

	if ((rank>=1) and (rank<=N)):
	    Ai = numpy.empty([s/p,r/m])
	    Bi = numpy.empty([s/p,t/n])
	    comm.Recv(Ai,source=0,tag=0)
	    comm.Recv(Bi,source=0,tag=1)
	    
	  # # Compute prodcuts Ai.Bi at workers

	    yi=numpy.transpose(Ai).dot(Bi)

	    # # and send results to master
	    # Simulate straggling #
	    wait_time = numpy.random.exponential(straggling_mean)
	    time.sleep(wait_time)
	    comm.Send(yi,dest=0,tag=2)
	    
	# Receiving
	 
	if rank==0:
	    info = MPI.Status()
	    C=numpy.empty([r,t])
	    Ci=numpy.empty([r*K/m,t/n])
	    temp=numpy.empty([r/m,t/n])
	    Grecv=numpy.empty([r*K/m,r*K/m])
	    I=numpy.identity(r/m)
	    count=0
	    while count<K :
		comm.Recv(temp,source=MPI.ANY_SOURCE,tag=2,status=info)
		start_r = count*r/m
		end_r = (count+1)*r/m
		Ci[start_r:end_r,:]=temp
		i=info.Get_source()
		for j in range(K) :
		    start_c = j*r/m
		    end_c = (j+1)*r/m 
		    Grecv[start_r:end_r,start_c:end_c]=(x[i-1]**j)*I
		count+=1
	    Ginv=numpy.linalg.inv(Grecv)
	    print('k=,ci, Ginv ',K,numpy.shape(Ci), numpy.shape(Ginv))
	    Yo=Ginv.dot(Ci)
	    print('Y ',numpy.shape(Yo))
	    for i in range(m) :
		start_r = i*r/m
		end_r = (i+1)*r/m
		for j in range(n) :
		    start_c = j*t/n
		    end_c = (j+1)*t/n	
		    kk=p-1+i*p+j*p*m
		    s=kk*r/m
		    e=(kk+1)*r/m
		    print('s,e ',s,' ',e)

		    C[start_r:end_r,start_c:end_c]=Yo[s:e,:]       
                time_this = time.time() - tstart
		
		
		# Check results: uncomment below lines to check if the result is correct
		#err=numpy.linalg.norm(y-A.dot(x)) # Frobenius norm of the difference

		# Receive from other nodes as well #
		while count < N:
		    comm.Recv(temp,source=MPI.ANY_SOURCE,tag=2,status=info)
		    count += 1
                err=numpy.linalg.norm(C-numpy.transpose(A).dot(B))
	    #Y[iter]=time.time() - tstart
        # # Check results: uncomment below lines to check if the result is correct
                print('error',err) # Frobenius norm of the difference
                              
		total_time = time_this
    	        
            if rank > N:
                pass

        if rank == 0:
            Y[zi,z1i] =total_time#/niter
            Yn[zi,z1i] =err#total_error/niter
            sheet1.write(zi, z1i, Y[zi,z1i])
            sheet2.write(zi, z1i, Yn[zi,z1i])
            z1i+=1
    zi=zi+1
if rank==0:
    print('X=',X1,'Y=',Y)
    #print('x=',numpy.shape(X),'y=',numpy.shape(Y))
    wb.save('entangled_square_1130.04.xls') 

