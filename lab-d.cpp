#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// mpicc -o fox fox.c -lm   //将fox.c编译为可执行文件fox
// mpirun -np 4 ./fox  //选择线程数为4，并执行fox文件

const int N = 512;//矩阵A, B的维度

void Print_Mat(int *A, int n){//打印矩阵
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%d  ",A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int int_sqrt(int p){//计算p的根号，返回整数结果
    for(int i = 1; i <= p; i++){
        if(i * i == p){
            return i;
        }
    }
    return -1;
}

//获得A的第i,j个n*n方块,储存在a中
void Get_Block(int *A, int *a, int i, int j, int n){
    for(int k = 0; k < n; k++){
        for(int l = 0; l < n; l++){
            a[k * n + l] = A[i * n * N + j * n + k * N + l];
        }
    }
}

//矩阵相乘的串行算法，计算矩阵乘法 a*b，计算结果储存在c中
void Multi_Mat(int *a, int *b, int *c, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

//fox矩阵乘法
void Fox(int *a, int *b, int *c, int sp, int n, int myrank){
    int *temp_a = (int*)malloc(n * n * sizeof(int));//用来接收a
    int *temp_b = (int*)malloc(n * n * sizeof(int));//用来接收b

    //将处理器记为P_ij
    int j = myrank % sp;
    int i = (myrank - j) / sp;

    //对方块c初始化
    for(int k = 0; k < n * n; k++){
        c[k]=0;
    }
    int senddest = 0, recvdest = 0;
    //fox循环
    for(int round = 0; round < sp; round++){
        if(i == (j + round) % sp){//选出本轮在i行广播的A_ij
            for(int k = 0; k < sp; k++){
                if(k != j){
                    MPI_Send(a, n * n, MPI_INT, i * sp + k, (round + 1) * sp, MPI_COMM_WORLD);
                }
                else{
                    for(int l = 0; l < n * n; l++){
                        temp_a[l]=a[l];
                    }
                }
            }
        }
        else{
            if(i - round < 0){
                recvdest = i * sp + i - round + sp;
            }else{
                recvdest = i * sp + i -round;
            }
            MPI_Recv(temp_a, n * n, MPI_INT, recvdest, (round + 1) * sp, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        //做矩阵乘法
        Multi_Mat(temp_a, b, c, n);

        if(round == sp - 1)break;
        
        //将块B_ij上移
        senddest = i > 0 ? i - 1:i - 1 + sp;
        recvdest = i < sp - 1 ? i + 1:i + 1 - sp;
        MPI_Sendrecv(b, n * n, MPI_INT, senddest * sp + j, 0, 
                    temp_b, n * n, MPI_INT, recvdest * sp + j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int i = 0; i < n * n; i++){
            b[i] = temp_b[i];
        }
    }
}

//将C的第i,j个n*n方块用c赋值
void Copy_Block(int *c, int *C, int i, int j, int n){
    for(int k = 0; k < n; k++){
        for(int l = 0; l < n; l++){
            C[i * n * N + j * n + k * N + l] = c[k * n + l];
        }
    }
}

// processor0 接收其他处理器的计算结果
void Recv_Block(int *c, int *C, int sp, int n){
    for(int i = 0; i < sp; i++){
        for(int j = 0; j < sp; j++){
            if(i + j != 0) 
                MPI_Recv(c, n * n, MPI_INT, i * sp + j, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Copy_Block(c, C, i, j, n);
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int p, myrank;
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int *A, *B, *C;
    if(myrank == 0){
        // 由processor0 随机生成N*N矩阵A、B，和空矩阵C
        srand(time(NULL));
        A = (int *)malloc(N * N * sizeof(int));
        B = (int *)malloc(N * N * sizeof(int));
        C = (int *)malloc(N * N * sizeof(int));
        for(int i = 0; i < N * N; i++){
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
        //Print_Mat(A, N);
    }

    int sp = int_sqrt(p);
    if(sp == -1 && myrank == 0){
        printf("sp = %d, 处理器数不是完全平方数\n",sp);
        return(0);
    }
    int n = N / sp;//方块的维度
    if(sp * n != N && myrank == 0){
        printf("处理器数不能均分矩阵\n");
        return(0);
    }
    //printf("n = %d\n", n);
    int *a = (int*)malloc(n * n * sizeof(int));//a, b, c储存方块
    int *b = (int*)malloc(n * n * sizeof(int));
    int *c = (int*)malloc(n * n * sizeof(int));

    //计时开始
    MPI_Barrier(MPI_COMM_WORLD);
    double start_fox = MPI_Wtime();

    if(myrank == 0){
        //由processor0把A，B的n*n方块发送给另外sp*sp-1个处理器
        for(int i = 0; i < sp; i++){
            for(int j = 0; j < sp; j++){
                if(i == 0 && j == 0) continue;
                Get_Block(A, a, i, j, n);
                Get_Block(B, b, i, j, n);
                //Print_Mat(a, n);
                MPI_Send(a, n * n, MPI_INT, i * sp + j, 'a', MPI_COMM_WORLD);
                MPI_Send(b, n * n, MPI_INT, i * sp + j, 'b', MPI_COMM_WORLD);
            }
        }
        // processor0 储存方块A_00, B_00
        Get_Block(A, a, 0, 0, n);
        Get_Block(B, b, 0, 0, n);
    }
    else{// 其他处理器接收方块A_ij, B_ij
        MPI_Recv(a, n * n, MPI_INT, 0, 'a', MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, n * n, MPI_INT, 0, 'b', MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Fox矩阵乘法
    Fox(a, b, c, sp, n, myrank);

    if(myrank != 0){//其他处理器向 processor0 发送计算结果
        MPI_Send(c, n * n, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }
    else{//processor0 接收方块结果，并赋值给矩阵C
        Recv_Block(c, C, sp, n);
    }

    //计时结束
    double finish_fox = MPI_Wtime();

    // 由 processor0 打印计算结果
    if (myrank == 0)
    {   
        // 检验fox乘法结果正确性
        #if 0
        printf("Matrix A = \n");Print_Mat(A, N);
        printf("Matrix B = \n");Print_Mat(B, N);
        printf("fox计算结果 C = \n");Print_Mat(C, N);
        #endif
        printf("fox并行乘法耗时: %f\n", finish_fox - start_fox);
        
        //归零
        for(int i=0; i < N * N; i++){
            C[i]=0;
        }

        //串行乘法
        double start =  MPI_Wtime();
        //Multi_Mat(A, B, C, N);
        double finish = MPI_Wtime();

        #if 0
        printf("串行计算结果 C = \n");Print_Mat(C, N);
        #endif
        printf("串行乘法耗时: %f\n", finish - start);

        //计算加速比
        double rate = (finish - start)/(finish_fox - start_fox);
        printf("并行加速比为: %f\n", rate);
    }

    MPI_Finalize();
    return 0;
}

