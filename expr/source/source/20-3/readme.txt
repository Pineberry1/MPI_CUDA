1. compile:
mpicc relaxation.c -o relaxation

2. run:
mpirun -np 4 relaxation

3. result:
Input of file "dataIn.txt"
3       4
9.000000        -1.000000       -1.000000       7.000000
-1.000000       8.000000        0.000000        7.000000
-1.000000       0.000000        9.000000        8.000000

0.000000        0.000000        1.000000


Output of result
1  th total vaule = 0
2  th total vaule = 0
3  th total vaule = 0
4  th total vaule = 0
5  th total vaule = 0
6  th total vaule = 2
7  th total vaule = 3
x[0] = 0.999996
x[1] = 0.999999
x[2] = 0.999999

Iteration num = 7
Whole running time    = 0.013542 seconds
Distribute data time  = 0.003764 seconds
Parallel compute time = 0.009778 seconds
