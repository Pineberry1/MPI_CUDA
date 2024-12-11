#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#define E 0.0001
#define intsize sizeof(int)
#define floatsize sizeof(float)
#define A(x,y) A[x*col+y]                         /* A is matrix row*col */
#define I(x,y) I[x*col+y]                         /* I is matrix col*col */
#define U(x,y) U[x*col+y]
#define B(x)   B[x]
#define a(x,y) a[x*col+y]
#define e(x,y) e[x*col+y]

int col,row;
int m,n,p;
int myid,group_size;
float *A,*I,*B,*U;
float *a,*e;
MPI_Status status;
float starttime,endtime,time1,time2;

void read_fileA()
{
    int i,j;
    FILE *fdA;

    time1=MPI_Wtime();
    fdA=fopen("dataIn.txt","r");
    fscanf(fdA,"%d %d", &row, &col);

    A=(float*)malloc(floatsize*row*col);

    for(i = 0; i < row; i ++)
    {
        for(j = 0; j < col; j ++) fscanf(fdA, "%f", A+i*row+j);
    }
    fclose(fdA);

    printf("Input of file \"dataIn.txt\"\n");
    printf("%d\t %d\n",row, col);
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++) printf("%f\t",A(i,j));
        printf("\n");
    }

    I=(float*)malloc(floatsize*col*col);

    for(i=0;i<col;i++)
        for(j=0;j<col;j++)
            if (i==j) I(i,j)=1.0;
    else I(i,j)=0.0;

}

int main(int argc,char **argv)
{
    int loop;
    int i,j,v;
    int p,group_size,myid;
    int k;
    float *sum, *ss;
    float aa,bb,rr,c,s,t;
    float su;
    float *temp,*temp1;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    if (myid==0) starttime=MPI_Wtime();
    p=group_size;
    loop=0;
    k=0;

    if (myid==0)
        read_fileA();

    MPI_Bcast(&row,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&col,1,MPI_INT,0,MPI_COMM_WORLD);

    m=row/p; if (row%p!=0) m++;                   /*    for a    */
    n=col/p; if (col%p!=0) n++;                   /*    for e    */

    if (myid==0)
    {
        B=(float*)malloc(floatsize*col);
        U=(float*)malloc(floatsize*row*col);
    }

    a=(float*)malloc(floatsize*m*col);
    e=(float*)malloc(floatsize*n*col);
    temp=(float*)malloc(floatsize*m);
    temp1=(float*)malloc(floatsize*n);
    ss=(float*)malloc(floatsize*3);
    sum=(float*)malloc(floatsize*3);

    if (myid==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<col;j++)
                a(i,j)=A(i,j);

        for(i=0;i<n;i++)
            for(j=0;j<col;j++)
                e(i,j)=I(i,j);

        for(i=1;i<p;i++)
        {
            MPI_Send(&(A(m*i,0)),m*col,MPI_FLOAT,i,i,MPI_COMM_WORLD);
            MPI_Send(&(I(n*i,0)),n*col,MPI_FLOAT,i,i,MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(a,m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD,&status);
        MPI_Recv(e,n*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD,&status);
    }

    if (myid==0)                                  /* start computing now */
        time2=MPI_Wtime();

    while (k<=col*(col-1)/2)
    {
        for(i=0;i<col;i++)
            for(j=i+1;j<col;j++)
        {
            ss[0]=0; ss[1]=0; ss[2]=0;
            sum[0]=0; sum[1]=0; sum[2]=0;

            for(v=0;v<m;v++)
                ss[0]=ss[0]+a(v,i)*a(v,j);

            for(v=0;v<m;v++)
                ss[1]=ss[1]+a(v,i)*a(v,i);

            for(v=0;v<m;v++)
                ss[2]=ss[2]+a(v,j)*a(v,j);

            MPI_Allreduce(ss,sum,3,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

            if (fabs(sum[0])>E)
            {
                aa=2*sum[0];
                bb=sum[1]-sum[2];
                rr=sqrt(aa*aa+bb*bb);

                if (bb>=0)
                {
                    c=sqrt((bb+rr)/(2*rr));
                    s=aa/(2*rr*c);
                }
                if (bb<0)
                {
                    s=sqrt((rr-bb)/(2*rr));
                    c=aa/(2*rr*s);
                }

                for(v=0;v<m;v++)
                {
                    temp[v]=c*a(v,i)+s*a(v,j);
                    a(v,j)=(-s)*a(v,i)+c*a(v,j);
                }
                for(v=0;v<m;v++)
                    a(v,i)=temp[v];

                for(v=0;v<n;v++)
                {
                    temp1[v]=c*e(v,i)+s*e(v,j);
                    e(v,j)=(-s)*e(v,i)+c*e(v,j);
                }
                for(v=0;v<n;v++)
                    e(v,i)=temp1[v];
            }
            else
                k++;
        }                                         /* for */
        loop ++;
    }                                             /* while */

    if (myid==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<col;j++)
                A(i,j)=a(i,j);

        for(i=0;i<n;i++)
            for(j=0;j<col;j++)
                I(i,j)=e(i,j);
    }

    if (myid!=0)
        MPI_Send(a,m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD);
    else
    {
        for(j=1;j<p;j++)
        {
            MPI_Recv(a,m*col,MPI_FLOAT,j,j,MPI_COMM_WORLD,&status);

            for(i=0;i<m;i++)
                for(k=0;k<col;k++)
                    A((j*m+i),k)=a(i,k);
        }
    }

    if (myid!=0)
        MPI_Send(e,n*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD);
    else
    {
        for(j=1;j<p;j++)
        {
            MPI_Recv(e,n*col,MPI_FLOAT,j,j,MPI_COMM_WORLD,&status);

            for(i=0;i<n;i++)
                for(k=0;k<col;k++)
                    I((j*n+i),k)=e(i,k);
        }
    }

    if (myid==0)
    {
        for(j=0;j<col;j++)
        {
            su=0.0;
            for(i=0;i<row;i++)
                su=su+A(i,j)*A(i,j);
            B(j)=sqrt(su);
        }

        for(i=1;i<col;i++)
            for(j=0;j<i;j++)
        {
            t=I(i,j);
            I(i,j)=I(j,i);
            I(j,i)=t;
        }

        for(j=0;j<col;j++)
            for(i=0;i<row;i++)
                U(i,j )=A(i,j )/B(j);

        printf(".........U.........\n");
        for(i=0;i<row;i++)
        {
            for(j=0;j<col;j++)
                printf("%f\t",U(i,j));
            printf("\n");
        }

        printf("........E.........\n");
        for(i=0;i<col;i++)
            printf("%f\t",B(i));
        printf("\n");

        printf("........Vt........\n");
        for(i=0;i<col;i++)
        {
            for(j=0;j<col;j++)
                printf("%f\t",I(i,j));
            printf("\n");
        }
    }

    if (myid==0)
    {
        endtime=MPI_Wtime();

        printf("\n");
        printf("Iteration num = %d\n",loop);
        printf("Whole running time    = %f seconds\n",endtime-starttime);
        printf("Distribute data time  = %f seconds\n",time2-time1);
        printf("Parallel compute time = %f seconds\n",endtime-time2);

    }

    MPI_Finalize();
    free(a);
    free(e);
    free(A);
    free(U);
    free(I);
    free(B);
    free(temp);
    free(temp1);
    return(0);
}
