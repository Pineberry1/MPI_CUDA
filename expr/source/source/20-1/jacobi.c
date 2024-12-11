#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"
#define E 0.0001
#define a(x,y) a[x*size+y]
#define b(x) b[x]
#define v(x) v[x]
#define v1(x) v1[x]
#define A(x,y) A[x*size+y]
#define B(x) B[x]
#define V(x) V[x]
#define intsize sizeof(int)
#define floatsize sizeof(float)
#define charsize sizeof(char)

int size,N;
int m;                                            /* the size of work on eath processor */
float *B;                                         /* store matrix B */
float *A;                                         /* store matrix A */
float *V;                                         /* store the value x */
double starttime;
double time1;
double time2;
int my_rank;                                      /* store the identifier of each processor */
int p;                                            /* store the number of processor */
MPI_Status status;
FILE *fdA,*fdB;

int i,j,group_size;
float sum;
float *b;
float *v;
float *a;
float *v1;
float lmax;                                       /* the differnce between v1 and v */
float max=1.0;                                    /* max of lmax */
int loop=0;                                       /* store the number of iteration */

/*name   Environment_Finalize
 *input  a,b,v,v1
 *usage  free a,b,v,v1
 */
void Environment_Finalize(float *a,float *b,float *v,float *v1)
{
    free(a);
    free(b);
    free(v);
    free(v1);
    if(my_rank==0)
    {
        free(A);
        free(B);
        free(V);
    }
}

/*name   Input
 *usage  read size,N,A,B,V from input file on the root
 */
void Input()
{
    if(my_rank==0)
    {
        starttime=MPI_Wtime();

        fdA=fopen("dataIn.txt","r");              /* Read from file "dataIn.txt" */
        fscanf(fdA,"%d %d", &size, &N);
        if (size != N-1)
        {
            printf("the input is wrong\n");
            exit(1);
        }

        A=(float *)malloc(floatsize*size*size);
        B=(float *)malloc(floatsize*size);
        V=(float *)malloc(floatsize*size);

        for(i = 0; i < size; i++)
        {
            for(j = 0; j < size; j++)
            {
                fscanf(fdA,"%f", A+i*size+j);
            }
            fscanf(fdA,"%f", B+i);
        }
        for(i = 0; i < size; i ++)
        {
            fscanf(fdA,"%f", V+i);
        }

        fclose(fdA);
    }
}

/*name   nodInit
 *usage  allocate spaces for v,a,b,v1 on each processor
 *       and store v value on the root
 */
void nodInit()
{
    m=size/p;if (size%p!=0) m++;
    v=(float *)malloc(floatsize*size);
    a=(float *)malloc(floatsize*m*size);
    b=(float *)malloc(floatsize*m);
    v1=(float *)malloc(floatsize*m);              /* store v value */

    if (a==NULL||b==NULL||v==NULL||v1==NULL)
        printf("allocate space fail!");

    if (my_rank==0)
    {
        for(i=0;i<size;i++)
            v(i)=V(i);
    }
}

/*name   Data_send
 *usage  initialize the value of a,b on root and broadcast
 */
void Data_send()
{
    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<size;j++)
                a(i,j)=A(i,j);

        for(i=0;i<m;i++)
            b(i)=B(i);
    }

    /* root sends all other nodes the corresponding data */
    if (my_rank==0)
    {
        for(i=1;i<p;i++)
        {
            MPI_Send(&(A(m*i,0)),m*size,MPI_FLOAT,i,i,MPI_COMM_WORLD);
            MPI_Send(&(B(m*i)),m,MPI_FLOAT,i,i,MPI_COMM_WORLD);
        }
        /* free(A); free(B); free(V); */
    }
    else
    {
        MPI_Recv(a,m*size,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);
        MPI_Recv(b,m,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);
    }
}

/*name   Compute
 *usage  compute v1(i) =( b(i)-¡Æa(i,j) * v(j) ) / a(i,my_rank * m + i)
 *       lmax = abs(v1(i)-v(my_rank * m + i))
 *	     collect the max lmax on each processor,if it is greater than E
 *	     then continue else end while
 */
void Compute()
{
    time1=MPI_Wtime();

    /* computing start */
    while (max>E)                                 /* The precision requirement */
    {
        lmax=0.0;

        for(i=0;i<m;i++)
        {
            if(my_rank*m+i<size)
            {
                sum=0.0;
                for(j=0;j<size;j++)
                    if (j!=(my_rank*m+i))
                        sum=sum+a(i,j)*v(j);

                                                  /* computes the new elements */
                v1(i)=(b(i) - sum)/a(i,my_rank * m + i);

                if (fabs(v1(i)-v(i))>lmax)
                    lmax=fabs(v1(i)-v(my_rank * m + i));
            }
        }

        /*Find the max element in the vector*/
        MPI_Allreduce(&lmax,&max,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);

        /*Gather all the elements of the vector from all nodes*/
        MPI_Allgather(v1,m,MPI_FLOAT,v,m,MPI_FLOAT,MPI_COMM_WORLD);
        loop++;
    }                                             /* while */
    time2=MPI_Wtime();
}

/*name   Output
 *usage  print messages
 */
void Output()
{
    if (my_rank==0)
    {
        printf("Input of file \"dataIn.txt\"\n");
        printf("%d\t%d\n", size, N);
        for(i = 0; i < size; i ++)
        {
            for(j = 0; j < size; j ++) printf("%f\t",A(i,j));
            printf("%f\n",B(i));
        }
        printf("\n");
        for(i = 0; i < size; i ++)
        {
            printf("%f\t", V(i));
        }
        printf("\n\n");
        printf("\nOutput of solution\n");
        for(i = 0; i < size; i ++) printf("x[%d] = %f\n",i,v(i));
        printf("\n");
        printf("Iteration num = %d\n",loop);
        printf("Whole running time    = %f seconds\n",time2-starttime);
        printf("Distribute data time  = %f seconds\n",time1-starttime);
        printf("Parallel compute time = %f seconds\n",time2-time1);

        fdB=fopen("dataOut.txt","w");
        fprintf(fdB,"Input of file \"dataIn.txt\"\n");
        fprintf(fdB,"%d\t%d\n", size, N);
        for(i = 0; i < size; i ++)
        {
            for(j = 0; j < size; j ++) fprintf(fdB,"%f\t",A(i,j));
            fprintf(fdB,"%f\n",B(i));
        }
        fprintf(fdB,"\n");
        for(i = 0; i < size; i ++)
        {
            fprintf(fdB,"%f\t", V(i));
        }
        fprintf(fdB,"\n\n");
        fprintf(fdB,"\nOutput of solution\n");
        for(i = 0; i < size; i ++) fprintf(fdB,"x[%d] = %f\n",i,v(i));
        fprintf(fdB,"\n");
        fprintf(fdB,"Iteration num = %d\n",loop);
        fprintf(fdB,"Whole running time    = %f seconds\n",time2-starttime);
        fprintf(fdB,"Distribute data time  = %f seconds\n",time1-starttime);
        fprintf(fdB,"Parallel compute time = %f seconds\n",time2-time1);
    }
}

/*name   main
 *input  argc:parameter number of command lines
 *       argv:array of each command lines' parameter
 *output return 0 if program is running correctly
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    p=group_size;
    Input();

    /*Broadcast the size of  factor Matrix*/
    MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);
    nodInit();

    /*Broadcast the initial vector*/
    MPI_Bcast(v,size,MPI_FLOAT,0,MPI_COMM_WORLD);
    Data_send();
    Compute();
    Output();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    Environment_Finalize(a,b,v,v1);
    return (0);
}
