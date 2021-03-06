# Uncoded distributed matrix-vector multiplication
# Printed time is average over several iterations
# compile as: mpirun -n k python script.py
# k = Number of workers
# Replace script.py with your python file name

import xlsxwriter
from mpi4py import MPI
import numpy
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
SZ = [1200, 2700, 6000, 12000] #Array of different Matrix Size
SW = [5, 10, 15, 20, 30] # Array of different number of workers
X = 5
zi = 0

workbook = xlsxwriter.Workbook('Uncoded_row_error.xlsx')
worksheet = workbook.add_worksheet('Sheet1')

Y = numpy.empty([4, 5])
for Z in SZ:
    z1i = 0
    for W in SW:
        M = Z  # rows of A
        N = Z  # columns of A
        n = W  # number of workers
        niter = int(50)

## Partition the matrix A and store the partitions
## at respective workers
## Use tag = 0 for this communication phase

        if rank == 0:
            A = numpy.random.rand(M, N)
            for i in range(1, n + 1):  # i is worker's rank
                start_indx = (i - 1) * M / n
                end_indx = i * M / n - 1
                comm.Send(A[start_indx:end_indx + 1], dest=i, tag=0)  # this comm phase has tag=1

        if rank >= 1 and rank <= n:
            Ai = numpy.empty([M / n, N])
            comm.Recv(Ai, source=0, tag=0)

        if rank == 0:
            total_time = 0
            total_error = 0
        for iter in range(0, niter):

    # # Generate and broadcast input column vector x to all workers

            if rank == 0:
                x = numpy.random.rand(N, 1)

                # print x

            if rank != 0:
                x = numpy.empty([N, 1])

    # # start time

            if rank == 0:
                tstart = time.time()

            comm.Bcast(x, root=0)

    # # Compute prodcuts Ai.x at workers
    # # and send results to master

            if rank >= 1 and rank <= n:
                yi = Ai.dot(x)
                comm.Send(yi, dest=0, tag=iter + 1)

    # # Receive outputs from workers

            if rank == 0:
                y = numpy.empty([M, 1])  # formatted as a row vector
                info = MPI.Status()
                count_responses = 0
                temp = numpy.empty([M / n, 1])
                while count_responses < n:
                    comm.Recv(temp, source=MPI.ANY_SOURCE, tag=iter
                              + 1, status=info)
                    i = info.Get_source()  # rank of the sender
                    start_indx = (i - 1) * M / n
                    end_indx = i * M / n - 1
                    y[start_indx:end_indx + 1] = temp
                    count_responses += 1

                time_this = time.time() - tstart
                total_time += time_this

            # # Check results: uncomment below lines to check if the result is correct
            # print('error',numpy.linalg.norm(y-A.dot(x)))
            # Frobenius norm of the difference

                total_error += numpy.linalg.norm(y - A.dot(x))
            if rank > n:
                pass

        if rank == 0:
            Y[zi, z1i] = total_error / niter
            row = zi
            col = z1i
            worksheet.write(row, col, Y[zi, z1i])
            z1i += 1
    print Y
    zi = zi + 1

if rank == 0:
    workbook.close()
