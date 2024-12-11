1. compile:
mpicc matrixvector.c -o matrixvector

2. run:
mpirun -np 4 matrixvector

3. result:
The matrix and the vector has been read from file "dataIn.txt" is: 5 width
 The element of the matrix is as follows:
row 0   1.000000        1.000000        1.000000        1.000000        1.000000
row 1   2.000000        2.000000        2.000000        2.000000        2.000000
row 2   3.000000        3.000000        3.000000        3.000000        3.000000
row 3   4.000000        4.000000        4.000000        4.000000        4.000000
row 4   5.000000        5.000000        5.000000        5.000000        5.000000
the element of the vector is as follows:
row 0 is 1.000000
row 1 is 2.000000
row 2 is 3.000000
row 3 is 4.000000
row 4 is 5.000000
now there is 3 processors are working on it and 2 rows per processors
the result is as follows:
row0 15.000000
row1 30.000000
row2 45.000000
row3 60.000000
row4 75.000000

Whole running time    = 0.004522 seconds
prepare time  = 0.002914 seconds
Parallel compute time = 0.000497 seconds
