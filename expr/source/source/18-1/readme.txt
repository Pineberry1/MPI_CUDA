1.compile£º
mpicc transpose.c -o transpose

2.run£º
mpirun -np 4 transpose

3.result£º
Input of file "dataIn.txt"
3       3
1.000000        2.000000        3.000000
3.000000        4.000000        5.000000
5.000000        6.000000        7.000000

Output of Matrix AT
1.000000        3.000000        5.000000
2.000000        4.000000        6.000000
3.000000        5.000000        7.000000

Whole running time    = 0.012299 seconds
Distribute data time  = 0.011884 seconds
Parallel compute time = 0.000415 seconds
