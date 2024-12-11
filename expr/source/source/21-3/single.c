#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#define E 0.0001
#define intsize sizeof(int)
#define floatsize sizeof(float)
#define A(x,y) A[x*col+y]
#define I(x,y) I[x*col+y]
#define B(x) B[x]
#define a(x,y) a[x*col+y]
#define e(x,y) e[x*col+y]

int col, N;
int m,p;
int myid,group_size;
float *A,*I,*B;
float *a,*e;
MPI_Status status;
float starttime,endtime,time1,time2;

/*读输入的矩阵A*/
void read_fileA()
{
    int i,j;
    FILE *fdA;
    time1=MPI_Wtime();

    fdA=fopen("dataIn.txt","r");
    fscanf(fdA,"%d %d", &col, &N);
    if (col != N)
    {
        printf("the input is wrong\n");
        exit(1);
    }

    A=(float *)malloc(floatsize*col*col);
    I=(float *)malloc(floatsize*col*col);

    for(i = 0; i < col; i++)
    {
        for(j = 0; j < col; j++)
        {
            fscanf(fdA,"%f", A+i*col+j);
        }
    }

    fclose(fdA);

    printf("Input of file \"dataIn.txt\"\n");
    printf("%d\t%d\n", col, N);
    for(i = 0; i < col; i ++)
    {
        for(j = 0; j < col; j ++) printf("%f\t",A(i,j));
        printf("\n");
    }

    for(i=0;i<col;i++)
    {
        for(j=0;j<col;j++)
        {
            if (i==j) I(i,j)=1.0;
            else I(i,j)=0.0;
        }
    }
}

int main(int argc,char **argv)
{
    int loop;
    int i,j,v;
    int p,group_size,myid;
    int k;
    float aa,bb,rr,c,s;
    float su;
    float *ss;
    float *sum;
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

    MPI_Bcast(&col,1,MPI_INT,0,MPI_COMM_WORLD);

    m=col/p; if (col%p!=0) m++;

    a=(float*)malloc(floatsize*m*col);
    e=(float*)malloc(floatsize*m*col);
    temp=(float*)malloc(floatsize*m);
    temp1=(float*)malloc(floatsize*m);
    ss=(float*)malloc(floatsize*3);
    sum=(float*)malloc(floatsize*3);

    if (myid==0)
        B=(float*)malloc(floatsize*col);

    if (myid==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<col;j++)
                a(i,j)=A(i,j);

        for(i=0;i<m;i++)
            for(j=0;j<col;j++)
                e(i,j)=I(i,j);

        for(i=1;i<p;i++)
        {
            MPI_Send(&(A(m*i,0)),m*col,MPI_FLOAT,i,i,MPI_COMM_WORLD);
            MPI_Send(&(I(m*i,0)),m*col,MPI_FLOAT,i,i,MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(a,m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD,&status);
        MPI_Recv(e,m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD,&status);
    }

    if (myid==0)                                  /*开始计时*/
        time2=MPI_Wtime();

    while (k<=col*(col-1)/2)
    {
        for(i=0;i<col;i++)
            for(j=i+1;j<col;j++)
        {
            ss[0]=0; ss[1]=0; ss[2]=0;
            sum[0]=0.0; sum[1]=0.0; sum[2]=0.0;

            for(v=0;v<m;v++)
                ss[0]=ss[0]+a(v,i)*a(v,j);        /* 计算本进程所存的列子向量a[*, i], a[*, j]的内积赋值给ss[0] */

            for(v=0;v<m;v++)
                ss[1]=ss[1]+a(v,i)*a(v,i);        /* 计算本进程所存的列子向量a[*, i], a[*, i]的内积赋值给ss[1] */

            for(v=0;v<m;v++)
                ss[2]=ss[2]+a(v,j)*a(v,j);        /* 计算本进程所存的列子向量a[*, j], a[*, j]的内积赋值给ss[2] */

            /* 计算所有进程中ss[i]的和赋给sum [i] (i = 0, 1, 2) */
            MPI_Allreduce(ss,sum,3,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

            if (fabs(sum[0])>E)                   /* 各个进程并行进行对i和j列正交化 */
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
                    temp1[v]=c*e(v,i)+s*e(v,j);
                    e(v,j)=(-s)*e(v,i)+c*e(v,j);
                }
                for(v=0;v<m;v++)
                {
                    a(v,i)=temp[v];
                    e(v,i)=temp1[v];
                }
            }
            else
                k++;

        }                                         /* for */
        loop ++;
    }                                             /* while */

    if (myid==0)                                  /*0号进程从其余各进程中接收子矩阵a和e，得到经过变换的矩阵A和I*/
    {
        for(i=0;i<m;i++)
            for(j=0;j<col;j++)
                A(i,j)=a(i,j);

        for(i=0;i<m;i++)
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
        MPI_Send(e,m*col,MPI_FLOAT,0,myid,MPI_COMM_WORLD);
    else
    {
        for(j=1;j<p;j++)
        {
            MPI_Recv(e,m*col,MPI_FLOAT,j,j,MPI_COMM_WORLD,&status);

            for(i=0;i<m;i++)
                for(k=0;k<col;k++)
                    I((j*m+i),k)=e(i,k);
        }
    }

    if (myid==0)                                  /*0号进程负责计算特征值*/
    {

        for(j=0;j<col;j++)
        {
            su=0.0;
            for(i=0;i<col;i++)
                su=su+A(i,j)*A(i,j);
            B(j)=sqrt(su);
        }

        printf("\n");
        for(j=0;j<col;j++)
        {
            if (I(0,j)*A(0,j)>0)                  /*判断矩阵A特征值的符号*/
                printf("the envalue of the matrix is %f\n",B(j));

            if (I(0,j)*A(0,j)<0)
                printf("the envalue of the matrix is %f\n",(-1)*B(j));

            if (I(0,j)*A(0,j)==0)
            {
                if (I(1,j)*A(1,j)>=0)             /*判断矩阵A特征值的符号*/
                    printf("the envalue of the matrix is %f\n",B(j));
                if (I(1,j)*A(1,j)<0)
                    printf("the envalue of the matrix is %f\n",(-1)*B(j));
            }
        }
    }

    if (myid==0)
    {
        endtime=MPI_Wtime();
        printf("\n");
        printf("Iteration num = %d\n",loop);

        printf("Whole running time    = %f s\n",time2-starttime); /*输出整个运行时间*/
        printf("Distribute data time  = %f s\n",time1-starttime); /*输出划分时间*/
        printf("Parallel compute time = %f s\n",time2-time1);     /*输出并行计算时间*/
    }

    MPI_Finalize();
    free(a);
    free(e);
    free(ss);
    free(sum);
    free(A);
    free(I);
    free(B);
    free(temp);
    free(temp1);
    return(0);
}
