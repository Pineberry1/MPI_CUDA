1. compile:
mpicc qr_value.c -o qr_value -lm

2. run:
mpirun -np 3 qr_value

3. result:
Input of file "dataIn.txt"
3        3
1.000000        3.000000        4.000000
3.000000        1.000000        2.000000
4.000000        2.000000        1.000000

Iteration num = 15
Whole running time    = 0.109047 seconds
Distribute data time  = 0.001712 seconds
Parallel compute time = 0.107335 seconds

the envalue is
7.074675        -3.187882       -0.886791
