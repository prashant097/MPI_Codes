# Uncoded distributed matrix-matrix multiplication
# Printed time is average over several iterations
# compile as: mpirun -n k python script.py
# k = Number of workers
# Replace script.py with your python file name

import xlsxwriter
from mpi4py import MPI
import numpy
import time
import math

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

straggling_mean = 0.01

niter = int(15)
SZ = [180, 300, 600, 900, 1200]
SW = [5, 10, 17]
zi = 0
Data1 = numpy.empty([len(SZ), len(SW)])
Data2 = numpy.empty([len(SZ), len(SW)])

workbook = xlsxwriter.Workbook('Uncoded_row_time_error2.xlsx')
worksheet1 = workbook.add_worksheet('Sheet_time')
worksheet2 = workbook.add_worksheet('Sheet_error')

## Partition the matrix A and B and store the partitions
## at respective workers
## Use tag = 0 for A and tag =1 for B this communication phase

for Z in SZ:
    z1i = 0
    for W in SW:
        s = Z  # rows of A and B
        r = Z  # columns of A
        t = Z  # columns of A
        N = W - 1  # number of workers
        m = n = int(math.sqrt(N))
        if rank == 0:
            A = numpy.random.rand(s, r)
            B = numpy.random.rand(s, t)

            print ('A=', A)
            print ('B=', B)
            Ap = numpy.empty([m, s, r // m])
            Bp = numpy.empty([n, s, t // n])
            for i in range(1, m + 1):  # i is worker's rank
                start_indx = (i - 1) * r // m
                end_indx = i * r // m - 1
                Ap[i - 1, :, :] = A[:, start_indx:end_indx + 1]
            for i in range(1, n + 1):  # i is worker's rank
                start_indx = (i - 1) * t // n
                end_indx = i * t // n - 1
                Bp[i - 1, :, :] = B[:, start_indx:end_indx + 1]
            print ('Sending start')
            for i in range(m):
                for j in range(n):
                    k = i * n + j
                    print ('Worker Dest =', k)
                    comm.Send(Ap[i, :, :], dest=k + 1, tag=0)
                    print ('send A = ', i, j)
            for i in range(n):
                for j in range(m):
                    comm.Send(Bp[i, :, :], dest=j * n + i + 1, tag=1)
                    print ('send B = ', i, j)

        if rank >= 1 and rank <= N:
            Ai = numpy.empty([s, r // m])
            Bi = numpy.empty([s, t // n])
            comm.Recv(Ai, source=0, tag=0)
            comm.Recv(Bi, source=0, tag=1)

        if rank == 0:
            total_time = 0
            total_err = 0
        for iter in range(0, niter):

        # # start time

            if rank == 0:
                tstart = time.time()

            if rank >= 1 and rank <= N:
                yi = numpy.transpose(Ai).dot(Bi)
                print ('Ai =', Ai)
                print ('Bi =', Bi)
                print ('Yi =', yi)

        # Simulate straggling #

                wait_time = numpy.random.exponential(straggling_mean)
                time.sleep(wait_time)
                comm.Send(yi, dest=0, tag=iter + 1)

        # # Receive outputs from workers

            if rank == 0:
                C = numpy.empty([r, t])  # formatted as a row vector
                info = MPI.Status()
                count_responses = 0
                temp = numpy.empty([r // m, t // n])
                while count_responses < N:
                    comm.Recv(temp, source=MPI.ANY_SOURCE, tag=iter
                              + 1, status=info)
                    i = info.Get_source()  # rank of the sender
                    c = (i - 1) % n * (t // n)
                    r1 = (i - 1) // n * (r // m)
                    print ('c,r1,i =', c, r1, i)
                    C[r1:r1 + r // m, c:c + t // n] = temp
                    count_responses = count_responses + 1

        # print('error',numpy.linalg.norm(C-transpose(A).dot(B))) # Frobenius norm of the difference

                time_this = time.time() - tstart
                total_time += time_this
                C1 = numpy.transpose(A).dot(B)

        # # Check results: uncomment below lines to check if the result is correct

                print ('Actual Matrix product =', C1)
                print ('Distributed product =', C)
                total_err += numpy.linalg.norm(C - C1)  # Frobenius norm of the difference
        if rank == 0:
            Data1[zi, z1i] = total_time / niter
            Data2[zi, z1i] = total_err / niter
            worksheet1.write(zi, z1i, Data1[zi, z1i])
            worksheet2.write(zi, z1i, Data2[zi, z1i])
            z1i += 1
    zi = zi + 1
if rank == 0:
    print ('time=', total_time / niter)
    print ('error=', total_err / niter)
    print ('beta=', straggling_mean)
    print ('size of A=', numpy.shape(A))
    print ('size of B=', numpy.shape(B))
    workbook.close()
