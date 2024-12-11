1. compile:
mpicc single.c -o single -lm

2. run:
mpirun -np 4 single

3. result:
Input of file "dataIn.txt"
3       3
1.000000        3.000000        4.000000
3.000000        1.000000        2.000000
4.000000        2.000000        1.000000

the envalue of the matrix is 7.074389
the envalue of the matrix is -3.187124
the envalue of the matrix is -0.891775

Iteration num = 4
Whole running time    = 0.000000 s
Distribute data time  = 0.000000 s
Parallel compute time = 0.000000 s
