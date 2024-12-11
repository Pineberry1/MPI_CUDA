/*
 * 程序功能：该程序并行计算矩阵和向量的乘积，并将结果向量输出，采用带状划分的算法
 * 变量说明：matrix[width,width]   存储从文件中读取的用于计算的矩阵的元素，矩阵元素为单精度浮点数
 *           vec[width]    存储从文件中读取的用于计算的向量的元素，向量元素为单精度浮点数
 *           width         存储从文件中读取的未知矩阵的宽度，和向量的长度，用来动态给
 *                                  矩阵头指针matrix，和向量头指针vec分配空间，      整型变量
 *           col            默认的值应该为width＋1，如果不是报错f           整型变量
 *           starttime，finishprepare，finishcompute，finishall
 *                         上述四个全局变量用来记录四个时间点                        双精度浮点数
 *           status         MPI接受消息时用到的变量      MPI_statuc 变量
 *           rom_per_pro    记录每一个进程实际需要计算的行数               整形变量
 *           fin            输入文件指针
 *           i,j            控制循环，并用作消息传递时的tag       整形变量
 *           my_rank        存储当前进程的标识                    整型变量
 *           group_size     存储当前总的进程的总数                整型变量
 *           vecin[width]   在每一个进程中用来暂存vec的变量       元素为单精度浮点数
 *           ma[rom_per_pro,width]
 *                          每一个进程中用来暂存被分配的那部分的矩阵的元素  元素为单精度浮点数
 *           result[rom_per_pro]
 *                          每一个进程中用来记录计算结果的变量    元素为单精度浮点数
 *           sum            计算过程中用到的中间变量              单精度浮点数
 *           p              处理器的总数                           整型变量
 *           realp          实际用到的处理器的数目                 整型变量
 */
#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "math.h"

#define matrix(x,y) matrix[x*width+y]
#define vec(x) vec[x]
#define vecin(x) vecin[x]
#define result(x) result[x]
#define ma(x,y) ma[x*width+y]
#define intsize sizeof(int)
#define floatsize sizeof(float)

int realp,p;
int width,col,row_per_pro;
float *matrix;
float *vec;
double starttime,finishprepare,finishcompute,finishall;
MPI_Status status;
FILE *fin;

/*
 函数名: free_all
 功能：释放每个进程中动态分配的三个变量空间
 输入：三个双精度浮点数指针变量ma，result，vecin
 输出：返回为空
*/
void free_all(float *ma,float *result,float *vecin)
{
    free(ma);
    free(result);
    free(vecin);
}

/*
 函数名: main
 功能：从文件中读入一个矩阵和一个向量，并行他们的乘积，在屏幕上输出结果变量
 输入：argc为命令行参数个数
       argv为每个命令行参数组成的字符串数组。
 输出：返回0代表程序正常结束
*/
int main(int argc, char **argv)
{
    int i,j,my_rank,group_size;
    float sum;
    float *ma;
    float *result;
    float *vecin;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    p=group_size;

    /*
     由进程0完成matrix,vec的动态空间分配
     同时将向量和矩阵读入到vec和matrix的空间中
    */
    if(my_rank==0)
    {
        /*
         记下初始运行的时间点
         打开文件dataIn.txt并读出width和col值
         行列数不对，报错并中断返回
        */
        starttime=MPI_Wtime();
        fin=fopen("dataIn.txt","r");
        fscanf(fin,"%d %d", &width, &col);
        if (width!=(col-1))
        {
            printf("The input is wrong,check the input matrix width and the total columns\n\n\n\n");
            exit(1);
        }

        /*
         在进程0中为matrix和vec分配空间从dataIn.txt中读出相应的值
         matrix 分配了width*width大小的浮点数空间
         vec分配了width长的浮点数空间
        */
        matrix=(float *)malloc(floatsize*width*width);
        vec=(float *)malloc(floatsize*width);

        for(i = 0; i < width; i++)
        {
            for(j = 0; j < width; j++)
            {
                fscanf(fin,"%f", matrix+i*width+j);
            }
            fscanf(fin,"%f", vec+i);
        }

        /* 将从文件dataIn.txt中读入的结果显示在屏幕上 */
        fclose(fin);
        printf("The matrix and the vector has been read from file \"dataIn.txt\" is: %d width\n ",width);
        printf("The element of the matrix is as follows:\n");
        for(i=0;i<width;i++)
        {
            printf("row %d\t",i);
            for(j=0;j<width;j++)
                printf("%f\t",matrix(i,j));
            printf("\n");
        }
        printf("the element of the vector is as follows:\n");
        for(i=0;i<width;i++)
        {
            printf("row %d is %f\n",i,vec(i));
        }

    }

    /* 进程0已经得到width值，所以它将这个结果广播给其他进程 */
    MPI_Bcast(&width,1,MPI_INT,0,MPI_COMM_WORLD);

    /*
     计算每一个进程要处理的行数，注意不能整除时要使得row_per_pro的值加1
     计算实际需要的最多的进程数，对于用不上的进程结束其执行
     将实际用到的进程数和每个进程需要处理的行数输出到屏幕上
    */
    row_per_pro=width/p;
    if (width%p!=0)
        row_per_pro++;

    realp=0;
    while(row_per_pro*realp<width)
        realp++;

    if(my_rank==0)
        printf("now there is %d processors are working on it and %d rows per processors\n",realp,row_per_pro);

    /* 现在为每一个进程中的三个浮点数指针变量分配空间 */
    vecin=(float *)malloc(floatsize*width);
    ma=(float *)malloc(floatsize*row_per_pro*width);
    result=(float *)malloc(floatsize*row_per_pro);

    /* 将进程0中的vec向量的值广播到各个进程的vecin变量中保存 */
    if (my_rank==0)
    {
        for(i=0;i<width;i++)
            vecin(i)=vec(i);
        free(vec);
    }
    MPI_Bcast(vecin,width,MPI_FLOAT,0,MPI_COMM_WORLD);

    /* 将进程0中的matrix的各个元素按各个进程的分工进行传播 */
    if (my_rank==0)
    {
        for(i=0;i<row_per_pro;i++)
            for(j=0;j<width;j++)
                ma(i,j)=matrix(i,j);
    }

    if (my_rank==0)
    {
        if(width%realp==0)
        {
            for(i=1;i<realp;i++)
                MPI_Send(&(matrix(row_per_pro*i,0)),row_per_pro*width,MPI_FLOAT,i,i,MPI_COMM_WORLD);
        }
        else
        {
            for(i=1;i<(realp-1);i++)
                MPI_Send(&(matrix(row_per_pro*i,0)),row_per_pro*width,MPI_FLOAT,i,i,MPI_COMM_WORLD);
            MPI_Send(&(matrix(row_per_pro*(realp-1),0)),
                     (width-row_per_pro*(realp-1))*width,
                     MPI_FLOAT,(realp-1),(realp-1),MPI_COMM_WORLD);
        }
        free(matrix);
    }
    else
    {
        if (my_rank<realp)
        {
            if(width%realp==0)
                MPI_Recv(ma,row_per_pro*width,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);
            else
            {
                if(my_rank!=realp-1)
                    MPI_Recv(ma,row_per_pro*width,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);
                else
                    MPI_Recv(ma,(width-row_per_pro*(realp-1))*width,MPI_FLOAT,0,realp-1,MPI_COMM_WORLD,&status);

            }

        }
    }

    /* 所有的数据都已经传送完毕，同步各个进程并，记录现在的时间点 */
    MPI_Barrier(MPI_COMM_WORLD);
    finishprepare=MPI_Wtime();

    /* 各个进程作自己那一部分的计算工作 */
    if (my_rank==0)
    {
        for(i=0;i<row_per_pro;i++)
        {
            sum=0.0;
            for(j=0;j<width;j++)
                sum=sum+ma(i,j)*vecin(j);
            result(i)=sum;
        }
    }
    else
    {
        if(my_rank<(realp-1))
        {
            for(i=0;i<row_per_pro;i++)
            {
                sum=0.0;
                for(j=0;j<width;j++)
                    sum=sum+ma(i,j)*vecin(j);
                result(i)=sum;
            }
        }
        else
        {
            if (my_rank<realp)
            {
                for(i=0;i<(width-row_per_pro*(realp-1));i++)
                {
                    sum=0.0;
                    for(j=0;j<width;j++)
                        sum=sum+ma(i,j)*vecin(j);
                    result(i)=sum;
                }
            }
        }
    }

    /* 每一个进程都计算完毕，记下此时的时间点 */
    MPI_Barrier(MPI_COMM_WORLD);

    finishcompute=MPI_Wtime();

    /* 将结果传回进程0 */
    if (my_rank==0)
    {
        for(i=0;i<row_per_pro;i++)
            vecin(i)=result(i);
        for(i=1;i<(realp-1);i++)
            MPI_Recv(&(vecin(i*row_per_pro)),row_per_pro,MPI_FLOAT,i,i,MPI_COMM_WORLD,&status);
        if(realp!=1)
        {
            MPI_Recv(&(vecin(row_per_pro*(realp-1))),
                     (width-row_per_pro*(realp-1)),
                     MPI_FLOAT,(realp-1),(realp-1),
                     MPI_COMM_WORLD,&status);
        }

    }
    else
    {
        if (my_rank<realp)
        {
            if (my_rank!=(realp-1))
                MPI_Send(result,row_per_pro,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD);
            else
                MPI_Send(result,(width-row_per_pro*(realp-1)),MPI_FLOAT,0,(realp-1),MPI_COMM_WORLD);

        }
    }

    /* 将进程0内的vecin内保存的结果输出 */
    if (my_rank==0)
    {
        printf("the result is as follows:\n");
        for(i=0;i<width;i++)
            printf("row%d %f\n",i,vecin(i));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    finishall=MPI_Wtime();

    /* 最后将几个时间输出 */
    if (my_rank==0)
    {
        printf("\n");
        printf("Whole running time    = %f seconds\n",finishall-starttime);
        printf("prepare time  = %f seconds\n",finishprepare-starttime);
        printf("Parallel compute time = %f seconds\n",finishcompute-finishprepare);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    free_all(ma,result,vecin);
    return (0);
}
