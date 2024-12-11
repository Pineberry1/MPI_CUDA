1. compile:
mpicc jordan.c -o jordan

2. run:
mpirun -np 4 jordan

3. result(also in file dataOut.txt):
group size: 4
Input of file "dataIn.txt"
4
1.000000        4.000000        -2.000000       3.000000        6.000000
2.000000        2.000000        0.000000        4.000000        2.000000
3.000000        0.000000        -1.000000       2.000000        1.000000
1.000000        2.000000        2.000000        -3.000000       8.000000

Output of solution
x[0]=1.000000
x[1]=2.000000
x[2]=0.000000
x[3]=-1.000000

Whole running time    = 0.010401 seconds
Distribute data time  = 0.002502 seconds
Parallel compute time = 0.007899 seconds
