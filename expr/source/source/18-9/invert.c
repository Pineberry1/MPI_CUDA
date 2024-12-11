#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

#define a(x,y) a[x*M+y]
#define A(x,y) A[x*M+y]
#define B(x,y) B[x*M+y]
#define floatsize sizeof(float)
#define intsize sizeof(int)

int M, N;
float *A, *B;
double starttime;
double time1;
double time2;
int my_rank;
int p;
int m;
float* a;
float* f;
MPI_Status status;

/*
 * 函数名：fatal
 * 功能：程序出现严重错误时中止并给出错误信息；
 * 输入：message为存储错误信息的字符数组；
 */
void fatal(char *message)
{
    printf("%s\n", message);
    exit(1);
}

/*
 * 函数名：Environment_Finalize
 * 功能：清理程序所占用的内存空间；
 * 输入：a，f均为储存浮点数的数组，分别用于存放主行各元素和
 *       各进程将要处理的数据；
 */
void Environment_Finalize(float *a, float *f)
{
    free(a);
    free(f);
}


/*
 * 函数名：matrix_init
 * 功能：矩阵初始化。包括分配存储空间，从输入文件中读入数据；
 * 输入：无输入参数；
 * 输出：无返回值；
 */
void matrix_init()
{
    int i, j;
    FILE* fdA;
    if (my_rank==0)
    {
        /* starttime记录开始分配数据的时刻 */
        starttime=MPI_Wtime();
        fdA=fopen("dataIn.txt", "r");
        fscanf(fdA, "%d %d", &M, &N);
        /* 如果矩阵不是方阵，则报错 */
        if(M!=N)
        {
            puts("The input is error!");
            exit(0);
        }
        A=(float *)malloc(floatsize*M*M);
        for(i=0; i<M; i ++)
            for(j=0; j<M; j++)
                fscanf(fdA, "%f", A+i*M+j);
        fclose(fdA);
        B=(float *)malloc(floatsize*M*M);
    }
    /* 广播M(矩阵的维数)给所有进程 */
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

/*
 * 函数名：maxtrix_partition
 * 功能：矩阵划分。0号进程对矩阵A进行行交叉划分，将划分后的子矩阵
 *       分别传送给1至p-1号进程。
 */
void matrix_partition()
{
    int i, j, i1, i2;
    /* 各进程为主行元素建立发送和接收缓冲区 */
    f=(float*)malloc(sizeof(float)*M);
    /* 分配至各进程的子矩阵大小为m*M */
    a=(float*)malloc(sizeof(float)*m*M);
    if (a==NULL || f==NULL)
        fatal("allocate error\n");
     /* 0号进程拷贝相应数据到其子矩阵a */
    if (my_rank==0)
    {
        for(i=0; i<m; i++)
            for(j=0; j<M; j++)
                a(i,j)=A(i*p,j);
        /* 0号进程采用行交叉划分将矩阵A划分为m*M的p块子矩阵，依次发送给1至p-1号进程 */
        for(i=0; i<m*p; i++)
            if ((i%p)!=0)
        {
            i1=i%p;
            i2=i/p+1;
            if (i<M)
                MPI_Send(&A(i,0), M, MPI_FLOAT, i1, i2, MPI_COMM_WORLD);
            else
                MPI_Send(&A(0,0), M, MPI_FLOAT, i1, i2, MPI_COMM_WORLD);
        }
    }
    else
    {                                             /* 其它进程接收各自的子矩阵数据 */
        for(i=0; i<m; i++)
            MPI_Recv(&a(i,0), M, MPI_FLOAT, 0, i+1, MPI_COMM_WORLD, &status);
    }
}

/*
 * 函数名：broadcast_j
 * 功能：编号为j的进程广播主行元素
 * 输入：i为每个进程正在处理的子矩阵的行号，
 *       j为主元素所在的进程的编号，
 *       v为主元素(在矩阵A中)的行号；
 */
void broadcast_j(int i, int j, int v)
{
    int k;
    /* j号进程广播主行元素 */
    if (my_rank==j)
    {
        a(i,v)=1/a(i,v);
        for(k=0; k<M; k++)
        {
            if (k!=v)
                a(i,k)=a(i,k)*a(i,v);             /* 处理主行各元素 */
            f[k]=a(i,k);
        }
        /* 将变换后的主行元素(存于数组f中)广播到所有进程中 */
        MPI_Bcast(f, M, MPI_FLOAT, my_rank, MPI_COMM_WORLD);
    }
    else
    {
        /* 其余进程接收广播来的主行元素存于数组f中 */
        MPI_Bcast(f, M, MPI_FLOAT, j, MPI_COMM_WORLD);
    }
}


/*
 * 函数名：row_transform
 * 功能：各进程对其子矩阵的各行进行初等行变换
 * 输入：i为每个进程正在处理的子矩阵的行号，
 *       j为主元素所在的进程的编号，
 *       v为主元素(在矩阵A中)的行号；
 */
void row_transform(int i, int j, int v)
{
    int k, w;
    /* 编号不为j的进程利用主行对其m行行向量做行变换 */
    if (my_rank!=j)
    {
        /* 处理非主行、非主列元素 */
        for(k=0; k<m; k++)
        {
            for(w=0; w<M; w++)
                if(w!=v)
                    a(k,w)=a(k,w)-f[w]*a(k,v);
            a(k,v)=-f[v]*a(k,v);
        }
    }
    /* 发送主行数据的进程利用主行对其主行之外的m-1行行向量做行变换 */
    if (my_rank==j)
    {
        for(k=0; k<m; k++)
            if (k!=i)
        {
            for(w=0; w<M; w++)
                if (w!=v)
                    /* 处理主行所在的进程中的其它元素 */
                    a(k,w)=a(k,w)-f[w]*a(k,v);
            /* 处理主列元素 */
            a(k,v)=-f[v]*a(k,v);
        }
    }
}


/*
 * 函数名：matrix_inverse
 * 功能：得到输入矩阵的逆矩阵，结果存入矩阵B中；
 */
void matrix_inverse()
{
    int i, j, k;
    /* 0号进程将自己子矩阵a中的各行数据存入B */
    if (my_rank==0)
        for(i=0; i<m; i++)
            for(j=0; j<M; j++)
                B(i*p,j)=a(i,j);
    /* 0号进程从其余各进程中接收子矩阵a，得到经过变换的逆矩阵B */
    if (my_rank!=0)
    {
        for(i=0; i<m; i++)
            for(j=0; j<M; j++)
                MPI_Send(&a(i,j), 1, MPI_FLOAT, 0, my_rank, MPI_COMM_WORLD);
    }
    else
    {
        for(i=1; i<p; i++)
            for(j=0; j<m; j++)
                for(k=0; k<M; k++)
                {
                    MPI_Recv(&a(j,k), 1, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
                    if ((j*p+i)<M)
                        B((j*p+i),k)=a(j,k);
                }
    }
}

/*
 * 函数名：print_result
 * 功能：输出计算结果和时间统计；
 */
void print_result()
{
    FILE* fdOut;
    int i, j;
     /* 0号进程将运算结果写入目标文件 */
    if(my_rank==0)
    {
        fdOut=fopen("dataOut.txt", "w");
        fprintf(fdOut, "Input of file \"dataIn.txt\"\n");
        fprintf(fdOut, "%d\t%d\n", M, M);
        for(i=0;i<M;i++)
        {
            for(j=0; j<M; j++)
                fprintf(fdOut, "%f\t", A(i,j));
            fprintf(fdOut, "\n");
        }
        fprintf(fdOut, "\nOutput of Matrix A's inversion\n");
        for(i=0; i<M; i++)
        {
            for(j=0; j<M; j++)
                fprintf(fdOut, "%f\t", B(i,j));
            fprintf(fdOut, "\n");
        }
        /* 0号进程将时间统计写入目标文件 */
        fprintf(fdOut, "\n");
        fprintf(fdOut, "Whole running time    = %f seconds\n", time2-starttime);
        fprintf(fdOut, "Distribute data time  = %f seconds\n", time1-starttime);
        fprintf(fdOut, "Parallel compute time = %f seconds\n", time2-time1);
        fprintf(fdOut, "Parallel Process number = %d\n", p);
        fclose(fdOut);
    }
}

/*
 * 函数名: main
 * 功能：各个进程的主函数，处理程序的输入，计算并输出结果
 * 输入：argc为命令行参数个数；
 *       argv为每个命令行参数组成的字符串数组。
 * 输出：返回0代表程序正常结束；
 */
int main(int argc, char **argv)
{
    int i, j, group_size;
    int v;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    /* p为进程数 */
    p=group_size;
    matrix_init();
    /* m为每个进程需要处理的子矩阵行数 */
    m=M/p;
    if (M%p!=0) m++;
    matrix_partition();
    /* time1记录计算开始的时间 */
    time1=MPI_Wtime();
    /* 各个进程(当前编号为j的)轮流广播自己的第i行作为主行元素 */
    for(i=0; i<m; i++)
        for(j=0; j<p; j++)
    {
        v=i*p+j;
        if (v<M)
        {
            broadcast_j(i,j,v);
            row_transform(i,j,v);
        }
    }
    matrix_inverse();
    /* time2为计算结束的时刻 */
    time2=MPI_Wtime();
    print_result();
    MPI_Finalize();
    Environment_Finalize(a,f);
    return(0);
    free(A);
    free(B);
}
