#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define  MAX_PROCESSOR_NUM  12   /*set the max number of the Processor*/
#define  MAX_ARRAY_SIZE     32   /*set the max size of the array*/

int main(int argc, char *argv[])
{
    double a[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE], g[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
    int n;
    double transTime = 0,tempCurrentTime, beginTime;

    MPI_Status status;
    int rank, size;
    FILE *fin;
    int i, j, k;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(rank == 0)
    {
	fin = fopen("dataIn.txt","r");
        if (fin == NULL)     /* have no input source file */
        {
            puts("Not find input data file");     
	        puts("Please create a file \"dataIn.txt\"");
            puts("<example for dataIn.txt> ");
            puts("4");
            puts("1  3  4  5");
            puts("3  5  6  1");
            puts("4  6  2  7");
            puts("5  1  7  8");
            puts("\nArter\'s default data are running for you\n");
        }
        else
        {
            fscanf(fin, "%d", &n);
	    if ((n < 1)||(n > MAX_ARRAY_SIZE)) /* the matrix input error */
            {
                puts("Input the Matrix\'s size is out of range!");
                exit(-1);
            }
            for(i = 0; i < n; i ++) /* get the data of the matric in */ 
            {
                for(j = 0; j < n; j ++) fscanf(fin,"%lf", &a[i][j]);
            }
        }

        /* put out the matrix */
        puts("Cholersky Decomposion");
        puts("Input Matrix A from dataIn.txt");
        for(i = 0; i < n; i ++)
        {
            for(j = 0; j < n; j ++) printf("%9.5f  ", a[i][j]);
            printf("\n");
        }
        printf("\n");
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for(k = 0; k < n; k ++)
    {
        /* gathering the result ,and then broacasting to each processor */
        MPI_Bcast(a[k], (n-k)*MAX_ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                                                                                                            
        for(i = k+rank; i < n; i += size)
        {
            for(j = 0; j < k; j ++)
            {
                g[i][j] = a[i][j];
            }
            if (i == k)
            {
                for(j = k; j < n; j ++) g[i][j] = a[i][j]/sqrt(a[k][k]);
            }
            else
            {
                g[i][k] = a[i][k]/sqrt(a[k][k]);
                for(j = k+1; j < n; j ++) g[i][j] = a[i][j] - a[i][k]*a[k][j]/a[k][k];
            }
        }

        /* use the Cholersky Algorithm */
        for(i = k +rank; i < n; i ++)
        {
            MPI_Send(g[i], n, MPI_DOUBLE, 0, k*1000+i, MPI_COMM_WORLD);
        }

        if(rank == 0)
        {
            for(j = 0; j < size; j ++)
            {
                for(i = k + j; i < n; i += size)
                {
                    MPI_Recv(a[i], n, MPI_DOUBLE, j, k*1000+i, MPI_COMM_WORLD, &status);
                }
            }
        }
    }

    if (rank == 0)
    {
        puts("After Cholersky Discomposion");
        puts("Output Matrix G");
        for(i =0; i < n; i ++)
        {
            for(j = 0; j < i; j ++) printf("           ");
            for(j = i; j < n; j ++) printf("%9.5f  ", a[i][j]);
            printf("\n");
        } /* output the result */
    }

    MPI_Finalize();/* end of the program */
}

