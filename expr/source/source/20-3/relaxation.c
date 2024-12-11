#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"
#define E 0.0001
#define a(x,y) a[x*size+y]
#define b(x) b[x]
#define v(x) v[x]
#define A(x,y) A[x*size+y]
#define B(x) B[x]
#define V(x) V[x]
#define intsize sizeof(int)
#define floatsize sizeof(float)
#define charsize sizeof(char)

int size,N;                                       /*size 为系数矩阵的大小，N为size＋1*/
int m;
float *B;
float *A;
float *V;                                         /* store the value x */
double starttime;
double time1;
double time2;
int my_rank;
int p;
MPI_Status status;
FILE *fdA,*fdB,*fdB1;

void Environment_Finalize(float *a,float *b,float *v)
{
    free(a);
    free(b);
    free(v);
}


/*
 * 函数名: main
 * 输入：argc为命令行参数个数；
 *       argv为每个命令行参数组成的字符串数组。
 * 输出：返回0代表程序正常结束
 */
int main(int argc, char **argv)
{
    int i,j,my_rank,group_size;
    int k;
    float *sum;
    float *b;
    float *v;
    float *a;
    float *differ;
    float temp,w;
    int iteration,total,loop;
    int r,q;
    loop=0;
    total=0;
    w=0.9;

    MPI_Init(&argc,&argv);                        /* 启动计算 */
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);    /* 找进程数 */
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);       /* 找自己的id */
    p=group_size;

    if(my_rank==0)
    {
        starttime=MPI_Wtime();                    /* 开始计时 */

        fdA=fopen("dataIn.txt","r");              /* 读入输入的系数矩阵A，b */
        fscanf(fdA,"%d %d", &size, &N);

        /*size 要求 size=N+1*/
        if (size != N-1)
        {
            printf("the input is wrong\n");
            exit(1);
        }

        /*分配系数矩阵A，与b，解x的空间*/
        A=(float *)malloc(floatsize*size*size);
        B=(float *)malloc(floatsize*size);
        V=(float *)malloc(floatsize*size);

        /*存放A，b，x*/
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

        /*打印A，b，x*/
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
        printf("\nOutput of result\n");

    }                                             /* end of if (my_rank==0) */

    MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);  /* 广播size到所有进程*/

    m=size/p;if (size%p!=0) m++;                  /* m为每个进程保留的行数 */

    /*每个进程分配系数等的空间*/
    v=(float *)malloc(floatsize*size);
    a=(float *)malloc(floatsize*m*size);
    b=(float *)malloc(floatsize*m);
    sum=(float *)malloc(floatsize*m);

    if (a==NULL||b==NULL||v==NULL)
        printf("allocate space  fail!");

    if (my_rank==0)
    {
        for(i=0;i<size;i++)
            v(i)=V(i);
    }

    MPI_Bcast(v,size,MPI_FLOAT,0,MPI_COMM_WORLD); /* 将初始x广播到所有进程 */

    /* 给进程分配A,b,和x初值*/
    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<size;j++)
                a(i,j)=A(i,j);

        for(i=0;i<m;i++)
            b(i)=B(i);
    }

    /* 将A,b和x初值分配到所有进程,每个进程分配m行 */
    if (my_rank==0)                               /* 进程0将数据广播出去 */
    {
        for(i=1;i<p;i++)
        {
            MPI_Send(&(A(m*i,0)),m*size,MPI_FLOAT,i,i,MPI_COMM_WORLD);
            MPI_Send(&(B(m*i)),m,MPI_FLOAT,i,i,MPI_COMM_WORLD);
        }
        free(A); free(B); free(V);
    }
    else                                          /* 其他进程接收数据A,b和x的初值*/
    {
        MPI_Recv(a,m*size,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);
        MPI_Recv(b,m,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);
    }

    time1=MPI_Wtime();                            /* 取开始计算时间 */

    /* 求每行主对角线右边的元素和 */
    for(i=0;i<m;i++)                              /* 开始计算 */
    {
        sum[i]=0.0;
        for(j=0;j<size;j++)
        {
            if (j>(my_rank*m+i))
                sum[i]=sum[i]+a(i,j)*v(j);
        }
    }

    while (total<size)                            /* total为迭代差小于E的x分量个数，当所有解即所有x分量的迭代差均小于E时，停止计算 */
    {
        iteration=0;                              /* iteration为各个process中迭代差小于E的x分量个数 */
        total=0;

        for(j=0;j<size;j++)                       /* 依次以第0,1, ..., n-1行为主行 */

        {
            r=j%m; q=j/m;

            if (my_rank==q)                       /* 主行所在的处理器 */
            {
                temp=v(my_rank*m+r);

                /*sum[j]累加上本进程中已算出的新的x项，这些新项的行号小于当前处理的行号*/
                for(i=0;i<r;i++)
                    sum[r]=sum[r]+a(r,my_rank*m+i)*v(my_rank*m+i);

                /*计算新的x值*/
                v(my_rank*m+r)=(1-w)*temp+w*(b(r)-sum[r])/a(r,my_rank*m+r);

                if (fabs(v(my_rank*m+r)-temp)<E)  /*新旧值差小于E，累加iteration*/
                    iteration++;

                /*广播新的x值*/
                MPI_Bcast(&v(my_rank*m+r),1,MPI_FLOAT,my_rank,MPI_COMM_WORLD);

                sum[r]=0.0;                       /*当前行处理完毕后，sum清零，为下一次计算做准备*/
                for(i=0;i<r;i++)                  /*本process中前面行的sum累加新x值项，为下一次计算累加对角元素左边的项*/
                    sum[i]=sum[i]+a(i,my_rank*m+r)*v(my_rank*m+r);
            }
            else                                  /*非主行所在的处理器*/
            {
                                                  /*接收广播来的新x值*/
                MPI_Bcast(&v(q*m+r),1,MPI_FLOAT,q,MPI_COMM_WORLD);
                for(i=0;i<m;i++)                  /*累加新x值对应的项*/
                    sum[i]=sum[i]+a(i,q*m+r)*v(q*m+r);
            }
        }

        /* 求所有处理器中iteration值的和total并广播到所有处理器中 */
        MPI_Allreduce(&iteration,&total,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

        loop++;

        if (my_rank==0)                           /* 打印每次迭代的total值 */
            printf("%-2d th total vaule = %d\n",loop,total);
    }                                             /* while */

    time2=MPI_Wtime();                            /* 取结束时间 */

    if (my_rank==0)                               /* 打印解x值以及运行、分配和计算时间 */
    {

        for(i = 0; i < size; i ++) printf("x[%d] = %f\n",i,v(i));
        printf("\n");
        printf("Iteration num = %d\n",loop);
        printf("Whole running time    = %f seconds\n",time2-starttime);
        printf("Distribute data time  = %f seconds\n",time1-starttime);
        printf("Parallel compute time = %f seconds\n",time2-time1);
    }

    MPI_Barrier(MPI_COMM_WORLD);                  /* 同步所有进程 */
    MPI_Finalize();                               /* 结束计算 */

    Environment_Finalize(a,b,v);
    return (0);
}
