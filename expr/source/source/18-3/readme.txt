1. compile:
mpicc matrix.c -o matrix

2. run:
mpirun -np 4 matrix

3. result:
Input of file "dataIn.txt"
3        3
1.000000        2.000000        3.000000
4.000000        5.000000        6.000000
2.000000        0.000000        1.000000
3        3
10.000000       4.000000        6.000000
2.000000        1.000000        8.000000
5.000000        6.000000        12.000000

Output of Matrix C = AB
29.000000       24.000000       58.000000
80.000000       57.000000       136.000000
25.000000       14.000000       24.000000

Whole running time    = 0.012105 seconds
Distribute data time  = 0.007531 seconds
Parallel compute time = 0.004574 seconds
