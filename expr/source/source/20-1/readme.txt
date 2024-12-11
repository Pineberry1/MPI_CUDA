1. compile:
mpicc jacobi.c -o jacobi

2. run:
mpirun -np 4 jacobi

3. result (also in file dataOut.txt):
Input of file "dataIn.txt"
4       5
9.000000        -1.000000       -1.000000       1.000000        7.000000
0.000000        7.000000        -2.000000       0.000000        5.000000
-3.000000       -1.000000       3.000000        -1.000000       -1.000000
1.000000        1.000000        1.000000        9.000000        3.000000

0.000000        0.000000        1.000000        0.000000


Output of solution
x[0] = 0.999969
x[1] = 0.999960
x[2] = 0.999932
x[3] = 0.000031

Iteration num = 14
Whole running time    = 0.016051 seconds
Distribute data time  = 0.002984 seconds
Parallel compute time = 0.013067 seconds
