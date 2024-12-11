#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"

/* somethings convinient for programming use */
#define a(x,y) a[x*M+y]
#define b(x) b[x]
#define A(x,y) A[x*M+y]
#define B(x) B[x]
#define floatsize sizeof(float)
#define intsize sizeof(int)

/* M*M matrix */
int M;

/* dimision for per node */
int m;

/* A, B array pointer */
float *A;
float *B;

/* time starting to compute, and other virable for time use */
double starttime;
double time1;
double time2;

/* node rank */
int my_rank;

/* size of nodes */
int p;

int l;
MPI_Status status;

/*
 *Function: fatal()
 *Desc:     output some message for user to know, and exit
 *Param:    message-char pointer to msg going to print on the screen
 */
void
fatal(char *message)
{
    printf("%s\n",message);
    exit(1);
}

/*
 *Function: Environment_Finalize()
 *Desc:     release some pointer before all be finished
 *Param:    a,b,x,f - float pointer
 *
 */
void
Environment_Finalize(float *a,float *b,float *x,float *f)
{
    free(a);
    free(b);
    free(x);
    free(f);
}

/*
 *Function:  main()
 *Desc:      program entry
 *Param:
 */
int
main(int argc, char **argv)
{
    int i,j,t,k,my_rank,group_size;
    int i1,i2;
    int v,w;
    float s;
    float temp;
    int tem;
    float *f;
    float lmax;
    float *a;
    float *b;
    float *x;
    int *shift;
    FILE *fdA,*fdB;

    /* initialize MPI , group_size, my_rank */
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    if(my_rank==0)
        printf("group size: %d\n",group_size);

    /* node number */
    p=group_size;

    /*
     for the node which my_rank==0, set starttime
     and read A,B elements from file"dataIn.txt",
     p.s. M=N-1
    */
    if (my_rank==0)
    {
        starttime=MPI_Wtime();

        fdA=fopen("dataIn.txt","r");

        fscanf(fdA,"%d",&M);

        /* allocate memory for matrix A an B */
        A=(float *)malloc(floatsize*M*M);
        B=(float *)malloc(floatsize*M);

        /* read A&B elements(M*(M+1)) from the file */
        for(i = 0; i < M; i++)
        {
            /* the 0 to M-1 th element of each row is the element of A */
            for(j = 0; j < M; j++)
            {
                fscanf(fdA,"%f", A+i*M+j);
            }
            /* the last one of each row  is the element of B */
            fscanf(fdA,"%f", B+i);
        }
        fclose(fdA);
    }

    /* broadcast the value of M */
    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
    /* allocate area for each node */
    m=M/p;
    if (M%p!=0) m++;

    f=(float*)malloc(sizeof(float)*(M+1));
    a=(float*)malloc(sizeof(float)*m*M);
    b=(float*)malloc(sizeof(float)*m);
    x=(float*)malloc(sizeof(float)*M);
    shift=(int*)malloc(sizeof(int)*M);

    /* initialize shift value for each column */
    for(i=0;i<M;i++)
        shift[i]=i;

    /* node which my_rank==0 appoints the a,b matrix array */
    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<M;j++)
                a(i,j)=A(i*p,j);

        for(i=0;i<m;i++)
            b(i)=B(i*p);
    }

    if (my_rank==0)
    {
        for(i=0;i<M;i++)
            if ((i%p)!=0)
        {
            i1=i%p;
            i2=i/p+1;
            /* send A, B elements to node i1(i1!=0) */
            MPI_Send(&A(i,0),M,MPI_FLOAT,i1,i2,MPI_COMM_WORLD);
            MPI_Send(&B(i),1,MPI_FLOAT,i1,i2,MPI_COMM_WORLD);
        }
    }                                             /*  my_rank==0 */
    else                                          /*  my_rank !=0 */
    {
        /* receive A,B elements from node 0 */
        for(i=0;i<m;i++)
        {
            MPI_Recv(&a(i,0),M,MPI_FLOAT,0,i+1,MPI_COMM_WORLD,&status);
            MPI_Recv(&b(i),1,MPI_FLOAT,0,i+1,MPI_COMM_WORLD,&status);
        }
    }

    time1=MPI_Wtime();                            /* computing start */

    for(i=0;i<m;i++)
        for(j=0;j<p;j++)
    {
        if (my_rank==j)                           /* node rank= j*/
        {
            /* find lmax of matrix a on this node */
            v=i*p+j;
            lmax=a(i,v);
            l=v;

            for(k=v+1;k<M;k++)
                if (fabs(a(i,k))>lmax)
            {
                lmax=a(i,k);
                l=k;
            }

            /* exchange the with the max */
            if (l!=v)
            {
                for(t=0;t<m;t++)
                {
                    temp=a(t,v);
                    a(t,v)=a(t,l);
                    a(t,l)=temp;
                }
                /* record the shift in order to output the right x */
                tem=shift[v];
                shift[v]=shift[l];
                shift[l]=tem;
            }

            for(k=v+1;k<M;k++)
                a(i,k)=a(i,k)/a(i,v);

            b(i)=b(i)/a(i,v);
            a(i,v)=1;

            /* record f array for future use */
            for(k=v+1;k<M;k++)
                f[k]=a(i,k);
            f[M]=b(i);

            /* broadcast f & l value to node which rank= my_rank */
            MPI_Bcast(&f[0],M+1,MPI_FLOAT,my_rank,MPI_COMM_WORLD);
            MPI_Bcast(&l,1,MPI_INT,my_rank,MPI_COMM_WORLD);
        }
        else /*rank!=j*/
        {
            v=i*p+j;
            MPI_Bcast(&f[0],M+1,MPI_FLOAT,j,MPI_COMM_WORLD);
            MPI_Bcast(&l,1,MPI_INT,j,MPI_COMM_WORLD);
        }

        if(my_rank!=j)
            if (l!=v)
        {
            for(t=0;t<m;t++)
            {
                temp=a(t,v);
                a(t,v)=a(t,l);
                a(t,l)=temp;
            }

            tem=shift[v];
            shift[v]=shift[l];
            shift[l]=tem;
        }

        if (my_rank!=j)
            for(k=0;k<m;k++)
        {
            for(w=v+1;w<M;w++)
                a(k,w)=a(k,w)-f[w]*a(k,v);
            b(k)=b(k)-f[M]*a(k,v);
        }

        if (my_rank==j)
            for(k=0;k<m;k++)
                if (k!=i)
                {
                    for(w=v+1;w<M;w++)
                        a(k,w)=a(k,w)-f[w]*a(k,v);
                    b(k)=b(k)-f[M]*a(k,v);
                }
    } /* for i j */

    /* caculate x value at node which rank is zero */
    if (my_rank==0)
        for(i=0;i<m;i++)
            x[i*p]=b(i);

    /* node which rank is not zero send b() value to node zero */
    if (my_rank!=0)
        for(i=0;i<m;i++)
            MPI_Send(&b(i),1,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD);
    else
    {
        /*
         node which rank is  zero recieve b() value from node i
         and caculate x value
        */
        for(i=1;i<p;i++)
            for(j=0;j<m;j++)
        {
            MPI_Recv(&b(j),1,MPI_FLOAT,i,i,MPI_COMM_WORLD,&status);
            x[j*p+i]=b(j);
        }
    }

    /* write the result to the screen and to file "dataOut.txt" */
    fdA=fopen("dataOut.txt","w");

    if (my_rank==0)
    {
        printf("Input of file \"dataIn.txt\"\n");
        printf("%d\n", M);
        for(i=0;i<M;i++)
        {
            for(j=0;j<M;j++) printf("%f\t",A(i,j));
            printf("%f\n",B(i));
        }
        printf("\nOutput of solution\n");

        fprintf(fdA,"Output of solution\n");

        for(k=0;k<M;k++)
        {
            for(i=0;i<M;i++)
            {
                if (shift[i]==k)
                {
                    printf("x[%d]=%f\n",k,x[i]);
                    fprintf(fdA,"x[%d]=%f\n",k,x[i]);
                }
            }
        }
    }

    time2=MPI_Wtime();

    if (my_rank==0)
    {
        printf("\n");
        printf("Whole running time    = %f seconds\n",time2-starttime);
        printf("Distribute data time  = %f seconds\n",time1-starttime);
        printf("Parallel compute time = %f seconds\n",time2-time1);

        fprintf(fdA,"Whole running time    = %f seconds\n",time2-starttime);
        fprintf(fdA,"Distribute data time  = %f seconds\n",time1-starttime);
        fprintf(fdA,"Parallel compute time = %f seconds\n",time2-time1);
    }
    fclose(fdA);

    /* finalize and release pointer */
    MPI_Finalize();
    Environment_Finalize(a,b,x,f);
    return(0);
}
