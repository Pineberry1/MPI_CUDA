1. compile:
mpicc cholersky.c -o cholersky -lm

2. run:
mpirun -np 4 cholersky

3. result:
Cholersky Decomposion
Input Matrix A from dataIn.txt
  1.00000    0.00000    1.00000    2.00000
  0.00000    1.00000    0.00000    3.00000
  1.00000    0.00000    2.00000    4.00000
  2.00000    3.00000    4.00000   33.00000

After Cholersky Discomposion
Output Matrix G
  1.00000    0.00000    1.00000    2.00000
             1.00000    0.00000    3.00000
                        1.00000    2.00000
                                   4.00000
