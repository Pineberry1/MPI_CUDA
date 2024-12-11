#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"

#define E 0.0001
#define a(x,y) a[x*m+y]
#define b(x,y) b[x*m+y]
#define A(x,y) A[x*size+y]
#define B(x,y) B[x*size+y]
#define intsize sizeof(int)
#define floatsize sizeof(float)
#define charsize sizeof(char)

int size,N;                                       /* size:保存矩阵行数;N:保存矩阵列数 */
int m;                                            /* 保存子方阵的尺寸 */
int t;                                            /* 棋盘划分的分割数 */
float *A, *B;                                     /* A:保存原矩阵;B:保存转置后的矩阵 */
double starttime;                                 /* 保存开始时间 */
double time1;                                     /* 保存分发数据的结束时间 */
double time2;                                     /* 保存运行的结束时间 */
int my_rank;                                      /* 保存当前进程的进程号 */
int p;                                            /* 保存进程数 */
MPI_Status status;                                /* 保存MPI状态 */
FILE *fdA;                                        /* 输入文件 */

/* 运行结束前,调用本函数释放内存空间 */
void Environment_Finalize(float *a,float *b)
{
    free(a);
    free(b);
}

int main(int argc, char **argv)
{
    int i,j,k,my_rank,group_size;
    float *a,*b;
    int u,v;
    float temp;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    p=group_size;

    /* 如果是主进程(rank=0的进程),则进行读文件的操作,
       将待转置的矩阵读入内存,保存到全局变量A中
    */
    if(my_rank==0)
    {
        starttime=MPI_Wtime();
        fdA=fopen("dataIn.txt","r");
        /* 读入矩阵的行数和列数,并保存到size和N中 */
        fscanf(fdA,"%d %d", &size, &N);
        /* 判断是否是方阵,如果不是,程序退出 */
        if(size != N)
        {
            puts("The input is error!");
            exit(0);
        }
        A=(float*)malloc(floatsize*size*size);
        B=(float*)malloc(floatsize*size*size);
        /* 将矩阵的所有值读入,保存到A中 */
        for(i = 0; i < size; i ++)
        {
            for(j = 0; j < size; j ++) fscanf(fdA, "%f", A+i*size+j);
        }
        fclose(fdA);
    }
    /* 广播矩阵的尺寸 */
    MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);

    /* 获得棋盘划分的数目 */
    t=(int)sqrt(p);
    if (t>size)
        t=size;
    if(size%t!=0)
        for(;;)
    {
        t--;
        if(size%t==0)
            break;
    }
    /* 获得实际利用的处理器个数 */
    p=t*t;
    /* 每个子方阵的尺寸 */
    m=size/t;

    /* a保存子方阵,b是临时矩阵,是主进程用来保存待发送给别的进程的子方阵 */
    a=(float *)malloc(floatsize*m*m);
    b=(float *)malloc(floatsize*m*m);

    if (a==NULL||b==NULL)
        printf("allocate space  fail!");

    /* 对主进程,获得自己的子方阵(即左上角的子方阵) */
    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<m;j++)
                a(i,j)=A(i,j);
    }

    /* 主进程向其他进程发送数据 */
    if (my_rank==0)
    {
        for(i=1;i<p;i++)
        {
            v=i/t;                                /* 子方阵的行号 */
            u=i%t;                                /* 子方阵的列号 */

            for(j=v*m;j<(v+1)*m;j++)
                for(k=u*m;k<(u+1)*m;k++)
                    b((j%m),(k%m))=A(j,k);        /* 将子方阵暂存在b中 */

                                                  /* 将子方阵发送到相应的进程 */
            MPI_Send(b,m*m,MPI_FLOAT,i,i,MPI_COMM_WORLD);
        }
    }
    else if (my_rank<p)                           /* 对其他进程,从主进程接收数据 */
        MPI_Recv(a,m*m,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);

    time1=MPI_Wtime();

    /* 对下三角的子方阵进行处理 */
    if ((my_rank/t)>(my_rank%t)&&my_rank<p)
    {
        v=my_rank/t;                              /* 行号 */
        u=my_rank%t;                              /* 列号 */

        /* 发送子方阵到位于相应上三角位置的进程 */
        MPI_Send(a,m*m,MPI_FLOAT,(u*t+v),(u*t+v),MPI_COMM_WORLD);
        /* 从相应上三角位置的进程接收数据 */
        MPI_Recv(a,m*m,MPI_FLOAT,(u*t+v),my_rank,MPI_COMM_WORLD,&status);
    }

    /* 对上三角的子方阵进行处理 */
    if ((my_rank/t)<(my_rank%t)&&my_rank<p)
    {
        v=my_rank/t;                              /* 行号 */
        u=my_rank%t;                              /* 列号 */
        /* 将子方阵元素复制到b */
        for(i=0;i<m;i++)
            for(j=0;j<m;j++)
                b(i,j)=a(i,j);

        /* 从相应下三角位置的进程接收数据 */
        MPI_Recv(a,m*m,MPI_FLOAT,(u*t+v),my_rank,MPI_COMM_WORLD,&status);
        /* 子方阵发送到位于相应下三角位置的进程 */
        MPI_Send(b,m*m,MPI_FLOAT,(u*t+v),(u*t+v),MPI_COMM_WORLD);
    }

    /* 对每一个子方阵进行转置 */
    for(i=1;i<m;i++)
        for(j=0;j<i;j++)
    {
        temp=a(i,j);
        a(i,j)=a(j,i);
        a(j,i)=temp;
    }

    /* 主进程开始将转置的结果进行组合
       先将主进程的结果组合到B中左上角
    */
    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<m;j++)
                B(i,j)=a(i,j);
    }
    /* 主进程从其他进程接收结果,组合到B的相应位置 */
    if (my_rank==0)
    {
        for(i=1;i<p;i++)
        {
            /* 从其他进程接收结果 */
            MPI_Recv(a,m*m,MPI_FLOAT,i,i,MPI_COMM_WORLD,&status);

            v=i/t;                                /* 结果的行号 */
            u=i%t;                                /* 结果的列号 */

            for(j=v*m;j<(v+1)*m;j++)
                for(k=u*m;k<(u+1)*m;k++)
                    B(j,k)=a((j%m),(k%m));        /* 结果组合到B的相应位置 */
        }
    }
    else if(my_rank<p)                            /* 其他进程发送结果到主进程 */
        MPI_Send(a,m*m,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD);

    /* 由主进程打印计算结果 */
    if (my_rank==0)
    {
        printf("Input of file \"dataIn.txt\"\n");
        printf("%d\t%d\n", size, size);
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++) printf("%f\t",A(i,j));
            printf("\n");
        }
        printf("\nOutput of Matrix AT\n");
        for(i=0;i<size;i++)
        {
            for(j=0;j<size;j++) printf("%f\t",B(i,j));
            printf("\n");
        }
    }
    time2=MPI_Wtime();
    /* 由主进程打印时间信息 */
    if (my_rank==0)
    {
        printf("\n");
        printf("Whole running time    = %f seconds\n",time2-starttime);
        printf("Distribute data time  = %f seconds\n",time1-starttime);
        printf("Parallel compute time = %f seconds\n",time2-time1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    Environment_Finalize(a,b);
    return(0);
}
