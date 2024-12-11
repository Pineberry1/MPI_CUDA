#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"
#define E 0.0001
#define a(x,y) a[x*size+y]
#define b(x) b[x]
#define v(x) v[x]
#define A(x,y) A[x*size+y]
#define V(x) V[x]
#define intsize sizeof(int)
#define floatsize sizeof(float)
#define charsize sizeof(char)

int size,N;
int m;
float *A;
float *V;                                         /* store the value x */
double starttime;
double time1;
double time2;
int my_rank;
int p;
MPI_Status status;
FILE *fdA,*fdB;

void Environment_Finalize(float *a,float *b,float *v)
{
    free(a);
    free(b);
    free(v);
}


int main(int argc, char **argv)
{
    int i,j,my_rank,group_size;
    float sum;
    float *b;
    float *v;
    float *a;
    float differ,max,localmax,oldmax;
    int loop;
    loop=0;
    differ=1.0;
    oldmax=0.0;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    p=group_size;

    if(my_rank==0)
    {
        starttime=MPI_Wtime();

        fdA=fopen("dataIn.txt","r");
        fscanf(fdA,"%d %d", &size, &N);
        if (size != N-1)
        {
            printf("the input is wrong\n");
            exit(1);
        }

        A=(float *)malloc(floatsize*size*size);
        V=(float *)malloc(floatsize*size);

        for(i = 0; i < size; i++)
        {
            for(j = 0; j < size; j++)
            {
                fscanf(fdA,"%f", A+i*size+j);
            }
            fscanf(fdA,"%f", V+i);
        }
        fclose(fdA);
        printf("Input of file \"dataIn.txt\"\n");
        printf("%d\t%d\n", size, N);
        for(i = 0; i < size; i ++)
        {
            for(j = 0; j < size; j ++) printf("%f\t",A(i,j));
            printf("%f\n",V(i));
        }
        printf("\n");
        printf("Output of running\n");
    }

    MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);

    m=size/p;if (size%p!=0) m++;

    v=(float *)malloc(floatsize*size);
    a=(float *)malloc(floatsize*m*size);
    b=(float *)malloc(floatsize*m);

    if (a==NULL||b==NULL||v==NULL)
        printf("allocate space  fail!");

    if (my_rank==0)
    {
        for(i=0;i<size;i++)
            v(i)=V(i);
    }

    MPI_Bcast(v,size,MPI_FLOAT,0,MPI_COMM_WORLD);

    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<size;j++)
                a(i,j)=A(i,j);
    }

    if (my_rank==0)
    {
        for(i=1;i<p;i++)
            MPI_Send(&(A(m*i,0)),m*size,MPI_FLOAT,i,i,MPI_COMM_WORLD);

        free(A);free(V);
    }
    else
        MPI_Recv(a,m*size,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);

    time1=MPI_Wtime();

    while (differ>E)                              /* computing start */
    {
        for(i=0;i<m;i++)
        {
            sum=0.0;
            for(j=0;j<size;j++)
                sum=sum+a(i,j)*v(j);
            b(i)=sum;
        }

        localmax=fabs(b(0));

        for(i=1;i<m;i++)
            if (fabs(b(i))>localmax)
                localmax=fabs(b(i));

        MPI_Allreduce(&localmax,&max,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);

        for(i=0;i<m;i++)
            b(i)=b(i)/max;

        MPI_Allgather(b,m,MPI_FLOAT,v,m,MPI_FLOAT,MPI_COMM_WORLD);

        differ=fabs(max-oldmax);
        oldmax=max;
        loop++;

        if (my_rank==0)
            printf("%2d th  differ=%f\n",loop,differ);

    }                                             /* while */

    time2=MPI_Wtime();
    if (my_rank==0)
    {

        printf("the envalue is %f\n\n",max);
        printf("Iteration num = %d\n",loop);
        printf("Whole running time    = %f s\n",time2-starttime);
        printf("Distribute data time  = %f s\n",time1-starttime);
        printf("Parallel compute time = %f s\n",time2-time1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    Environment_Finalize(a,b,v);
    return (0);
}
