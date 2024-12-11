#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#define INTSIZE sizeof(int)
#define CHARSIZE sizeof(char)
#define M(i,j) M[i*n*p+j]
#define a(i,j) a[i*n*p+j]
#define b(i,j) b[i*n+j]
#define c(l,i,j) c[i*n*p+l*n+j]
int *a,*b,*c,*d;
/**输入的邻接矩阵**/
int *M;
/**处理器数目**/
int p;
/**顶点数目**/
int N;
/**分配该单个处理器的顶点数目**/
int n;
/**处理器编号**/
int myid;
MPI_Status status;

/* 
 * 函数名: readmatrix 
 * 功能: 读入邻接矩阵
 * 输入： 
 * 输出：返回0代表程序正常结束;否则程序出错。
 */ 
int readmatrix()
{
	int i,j;
	printf("Input the size of matrix:");
	scanf("%d",&N);
	n=N/p;
	if(N%p!=0)n++;
	/*给M矩阵分配空间*/
	M=(int*)malloc(INTSIZE*n*p*n*p);
	if(!M)
	{
		error("failed to allocate space for M");
	}/*if*/
	printf("Input the matrix:\n");
	/*输入矩阵*/
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			scanf("%d",&(M(i,j)));
			if (i==j) M(i,j)=1;
		}/*for*/
		for(j=N;j<n*p;j++) M(i,j)=0;
	}/*for*/
	for(i=N;i<n*p;i++)
		for(j=0;j<n*p;j++)
			if(i==j) M(i,j)=1;
			else M(i,j)=0;
	return(0);
} /*readmatrix*/

/* 
 * 函数名: error 
 * 功能: 提示错误并退出
 * 输入：message为要提示的消息
 * 输出：输出出错的处理器号和消息
 *       返回0代表程序正常结束;否则程序出错。
 */ 
int error(message)
char *message;
{
	printf("processor %d:%s\n",myid,message);
	/*退出*/
	exit(0);
	return(0);
} /*error*/

/* 
 * 函数名: sendmatrix 
 * 功能: 向各处理器发送邻接矩阵数据
 * 输入： 
 * 输出：返回0代表程序正常结束;否则程序出错。
 */ 
int sendmatrix()
{
	int i,j;
	for(i=1;i<p;i++)
	{
		MPI_Send(&(M(n*i,0)),n*n*p,MPI_INT,i,i,MPI_COMM_WORLD);
		for(j=0;j<n*p;j++)
			MPI_Send(&(M(j,n*i)),n,MPI_INT,i,i,MPI_COMM_WORLD);
	} /*for*/
	return(0);
} /*sendmatrix*/

/* 
 * 函数名: getmatrix 
 * 功能: 各处理器接收邻接矩阵数据
 * 输入： 
 * 输出：返回0代表程序正常结束;否则程序出错。
 */ 
int getmatrix()
{
	int i,j;
	if(myid==0)
	{
		for(i=0;i<n;i++)
			for(j=0;j<n*p;j++)
				a(i,j)=M(i,j);
		for(i=0;i<n*p;i++)
			for(j=0;j<n;j++)
				b(i,j)=M(i,j);
	}
	else
	{
		MPI_Recv(a,n*n*p,MPI_INT,0,myid,MPI_COMM_WORLD,&status);
		for(j=0;j<n*p;j++)
			MPI_Recv(&(b(j,0)),n,MPI_INT,0,myid,MPI_COMM_WORLD,&status);
	} /*if*/
	return(0);
}

/* 
 * 函数名: writeback 
 * 功能: 并行矩阵运算过程，将结果保存在c矩阵中
 * 输入： 
 * 输出：返回0代表程序正常结束;否则程序出错。
 */ 
int paramul()
{
	int l,i,j,s;
	/**以下两变量表示前一个和后一个处理器标识**/
	int last,next;
	for(l=0;l<p;l++)
	{
		for(i=0;i<n;i++)
			for(j=0;j<n;j++)
				for(c(((l+myid)%p),i,j)=0,s=0;s<n*p;s++)
					if(a(i,s)&&b(s,j))
					{c(((l+myid)%p),i,j)=1;break;}
		last=(p+myid-1)%p;
		next=(myid+1)%p;
		if(l!=p-1)
		{
			if(myid%2==0)
			/**偶数号处理器直接接收数据b**/
			{
				MPI_Send(b,n*n*p,MPI_INT,last,last,MPI_COMM_WORLD);
				MPI_Recv(b,n*n*p,MPI_INT,next,myid,MPI_COMM_WORLD,&status);
			}
			else
			{
			/**奇数号处理器需要先将原数据保存，防止丢失**/
				for(i=0;i<n*n*p;i++)
					d[i]=b[i];
				MPI_Recv(b,n*n*p,MPI_INT,next,myid,MPI_COMM_WORLD,&status);
				MPI_Send(d,n*n*p,MPI_INT,last,last,MPI_COMM_WORLD);
			}
		}
	}/*for*/		
	return(0);
}

/* 
 * 函数名: writeback 
 * 功能: 将c矩阵保存的矩阵相乘结果写回处理器0的数组M
 * 输入： 
 * 输出：返回0代表程序正常结束;否则程序出错。
 */ 
int writeback()
{ 
	int i;
	if(myid==0)
	{
	for(i=0;i<n*n*p;i++)
		M(0,i)=c(0,0,i);
		for(i=1;i<p;i++)
		{
			MPI_Recv(&(M(i*n,0)),n*n*p,MPI_INT,i,i,MPI_COMM_WORLD,&status);
		}/*for*/
	}/*for*/
	else
		MPI_Send(c,n*n*p,MPI_INT,0,myid,MPI_COMM_WORLD);
	return(0);
}

/* 
 * 函数名: output 
 * 功能: 输出结果
 * 输入： 
 * 输出：输出结果矩阵M
 *       返回0代表程序正常结束;否则程序出错。
 */ 
int output()
{
	int i,j;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			printf("%d ",M(i,j));
		printf("\n");
	}
	return(0);
}
/******************** main ********************/
/* 
 * 函数名: main 
 * 功能: 启动MPI计算；
 *       确定进程数和进程标识符；
 *       调用主进程和从进程程序并行求解传递闭包问题。
 * 输入：argc为命令行参数个数； 
 *       argv为每个命令行参数组成的字符串数组。 
 * 输出：返回0代表程序正常结束;否则程序出错。
 */ 
int main(argc,argv)
	int argc;
	char **argv;
{

	int i,j;
	int group_size;
	int *temp;
    /*初始化*/
	MPI_Init(&argc,&argv);
	/*确定工作组中的进程个数*/
	MPI_Comm_size(MPI_COMM_WORLD,&group_size);
	/*确定自己在工作组中的进程标识符*/
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	p=group_size;
	/**处理器0读入矩阵并发送，步骤(1)**/
	if(myid==0)
	{
		/*输入邻接矩阵*/
		readmatrix();
		for(i=1;i<p;i++)
			MPI_Send(&N,1,MPI_INT,i,i,MPI_COMM_WORLD);
	}
	else
		MPI_Recv(&N,1,MPI_INT,0,myid,MPI_COMM_WORLD,&status);
	n=N/p;
	if(N%p!=0)n++;
	/*分配存储空间*/
	a=(int*)malloc(INTSIZE*n*n*p);
	b=(int*)malloc(INTSIZE*n*n*p);
	c=(int*)malloc(INTSIZE*n*n*p);
	d=(int*)malloc(INTSIZE*n*n*p);
	/*分配失败*/
	if(a==NULL||b==NULL||c==NULL)
		error("failed to allocate space for a,b and c");
	/**logN次矩阵自乘，步骤(3)**/
	for(i=0;i<=log(N)/log(2);i++)
	{
		if(myid==0)printf("loop %d:\n",i);
		if(myid==0) output();
		/**处理器0向各处理器发送必要数据，步骤(3.1)(3.2)**/
		if(myid==0) sendmatrix();
		/**步骤(3.3)**/
		getmatrix();
		/**矩阵相乘，步骤(3.4)**/
		paramul();
		/**结果写回，步骤(3.5)**/
		writeback();
	};
	if(myid==0)printf("loop %d:\n",i);
	if(myid==0)output();
	/*结束MPI计算*/
	MPI_Finalize();
	/*释放存储空间*/
	free(a);
	free(b);
	free(c);
	free(M);
	return(0);
} /* main */
