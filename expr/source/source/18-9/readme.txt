1. compile:
mpicc invert.c -o invert

2. run:
mpirun -np 4 invert

3. result( in file dataOut.txt):
Input of file "dataIn.txt"
3       3
1.000000        -1.000000       1.000000
5.000000        -4.000000       3.000000
2.000000        1.000000        1.000000

Output of Matrix A's inversion
-1.400000       0.400000        0.200000
0.200000        -0.200000       0.400000
2.600000        -0.600000       0.200000

Whole running time    = 0.008666 seconds
Distribute data time  = 0.001178 seconds
Parallel compute time = 0.007488 seconds
Parallel Process number = 4
