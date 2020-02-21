#!/usr/bin/python
# -*- coding: utf-8 -*-
# Uncoded distributed column wise matrix-vector multiplication
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

M = 88  # rows of A
N = 88  # columns of A
n = size - 1  # number of workers
m = 4
p = 4
niter = int(5)
X = range(niter)

## Partition the matrix A and store the partitions
## at respective workers
## Use tag = 0 for this communication phase

if rank == 0:
    A = numpy.random.rand(M, N)
    for h in range(n):
        for i in range(p):
            start_r = i * M / p
            end_r = (i + 1) * M / p
            for j in range(m):
                start_c = j * N / m
                end_c = (j + 1) * N / m
                comm.Send(numpy.ascontiguousarray(A[start_r:end_r,
                          start_c:end_c]), dest=h + 1, tag=0)  # A'(M/p,N/m)

if rank >= 1 and rank <= n:
    Ai = numpy.empty([M / p, N / m])
    comm.Recv(Ai, source=0, tag=0)

if rank == 0:
    total_time = 0
    Y = numpy.empty(5)
for iter in range(0, niter):

    # # Generate and broadcast input column vector x to all workers

    if rank == 0:
        x = numpy.random.rand(N, 1)

        # print x

    if rank != 0:
        xi = numpy.empty([N / m, 1])

    # # start time

    if rank == 0:
        tstart = time.time()
        for i in range(1, n + 1):  # i is worker's rank
            r = i % m
            if r == 0:
                r = m
            start_indx = (r - 1) * N / m
            end_indx = r * N / m
            comm.Send(x[start_indx:end_indx], dest=i, tag=1)  # this comm phase has tag=1
            print ('xi=', x[start_indx:end_indx])

    # # Compute prodcuts Ai.x at workers
    # # and send results to master
    # # tag = 1 for this phase

    if rank != 0:
        comm.Recv(xi, source=0, tag=1)
        yi = Ai.dot(xi)
        comm.Send(yi, dest=0, tag=iter + 1)

    # # Receive outputs from workers

    if rank == 0:
        y1 = numpy.empty([m, M])  # formatted as a row vector
        info = MPI.Status()
        count_responses = 0
        temp = numpy.empty(M / p)
        while count_responses < n:
            comm.Recv(temp, source=MPI.ANY_SOURCE, tag=iter + 1,
                      status=info)
            i = info.Get_source()  # rank of the sender
            r = i % m
            r1 = i // m
            if r == 0:
                r = m
                r1 = r1 - 1
            start_indx = r1 * M / p
            end_indx = (r1 + 1) * M / p - 1
            y1[r - 1, start_indx:end_indx + 1] = temp
            count_responses += 1
        print ('y1 size=', numpy.shape(y1), 'y1', y1)
        y = numpy.zeros((1, M))
        for i in range(m):
            y += y1[i, :]
        y = y.reshape(M, 1)
        time_this = time.time() - tstart
        total_time += time_this
        Y[iter] = time.time() - tstart

        # # Check results: uncomment below lines to check if the result is correct

        print ('error', numpy.linalg.norm(y - A.dot(x)))  # Frobenius norm of the difference

    if rank > n:
        pass

if rank == 0:
    print total_time / niter
    print ('X=', X, 'Y=', Y)
    print ('x=', numpy.shape(X), 'y=', numpy.shape(Y))

    workbook = xlsxwriter.Workbook('Blockwise.xlsx')
    worksheet = workbook.add_worksheet('First')

    row = 0
    col = 0
    for i in range(0, niter):
        worksheet.write(row, col, X[i])
        worksheet.write(row, col + 1, Y[i])
        row += 1
    workbook.close()
