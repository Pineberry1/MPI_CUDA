#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define  TRUE 1
 
/*
	* 函数名: main
	* 功能：实现快速排序的主程序
	* 输入：argc为命令行参数个数；
	*       argv为每个命令行参数组成的字符串数组。
	* 输出：返回0代表程序正常结束
*/
main(int argc,char *argv[])
{
	int DataSize;
	int *data;
	/*MyID表示进程标志符；SumID表示组内进程数*/
	int	MyID, SumID;
	int i, j;
	int m, r;

	MPI_Status status;
	/*启动MPI计算*/
	MPI_Init(&argc,&argv);

	/*MPI_COMM_WORLD是通信子*/
	/*确定自己的进程标志符MyID*/
	MPI_Comm_rank(MPI_COMM_WORLD,&MyID);	

	/*组内进程数是SumID*/
	MPI_Comm_size(MPI_COMM_WORLD,&SumID);	
	
	/*根处理机(MyID=0)获取必要信息，并分配各处理机进行工作*/
	if(MyID==0)
	{
		/*获取待排序数组的长度*/
		DataSize=GetDataSize();
		data=(int *)malloc(DataSize*sizeof(int));

		/*内存分配错误*/
		if(data==0) 
			ErrMsg("Malloc memory error!");

		/*动态生成待排序序列*/
		srand(396);
		for(i=0;i<DataSize;i++)
		{
			data[i]=(int)rand();
			printf("%10d",data[i]);
		}
		printf("\n");
	}

	m=log2(SumID);

	/* 从根处理器将数据序列广播到其他处理器*/
	/*{"1"表示传送的输入缓冲中的元素的个数,	   */
	/* "MPI_INT"表示输入元素的类型,			   */ 
	/* "0"表示root processor的ID  }			   */
	MPI_Bcast(&DataSize,1,MPI_INT,0,MPI_COMM_WORLD);
	
	/*ID号为0的处理器调度执行排序*/
	para_QuickSort(data,0,DataSize-1,m,0,MyID);
    
	/*ID号为0的处理器打印排序完的有序序列*/
	if(MyID==0)
	{
		for(i=0;i<DataSize;i++)
		{
			printf("%10d",data[i]);
		}
		printf("\n");
	}

	MPI_Finalize();		//结束计算
}


/*
	* 函数名: para_QuickSort
	* 功能：并行快速排序，对起止位置为start和end的序列，使用2的m次幂个处理器进行排序
	* 输入：无序数组data[1,n]，使用的处理器个数2^m
	* 输出：有序数组data[1,n]
*/
para_QuickSort(int *data,int start,int end,int m,int id,int MyID)
{
	int i, j;
	int r;
	int MyLength;
	int *tmp;
	MPI_Status status;

	MyLength=-1;

	/*如果可供选择的处理器只有一个，那么由处理器id调用串行排序，对应于算法13.4步骤(1.1)*/
	/*(1.1)	Pid call quicksort(data,i,j) */
	if(m==0)
	{
		if(MyID==id)
			QuickSort(data,start,end);
		return;
	}	

	/*由第id号处理器划分数据，并将后一部分数据发送到处理器id+exp2(m-1)，对应于算法13.4步骤(1.2,1.3)*/
	/*(1.2) Pid: r=patrition(data,i,j)*/	
	if(MyID==id)
	{
		/*将当前的无序区R[1，n]划分成左右两个无序的子区R[1，i-1]和R[i，n](1≤i≤n)*/
		r=Partition(data,start,end);	
		MyLength=end-r;	
		/*(1.3)	Pid send data[r+1,m-1] to P(id+2m-1) */
		/* {MyLength表示发送缓冲区地址；*/
		/*  发送元素数目为1;		   */
		/*  MyID是消息标签 }		   */
		MPI_Send(&MyLength,1,MPI_INT,id+exp2(m-1),MyID,MPI_COMM_WORLD);
		/*若缓冲区不空，则第id+2m-1号处理器取数据的首址是data[r+1]*/
		if(MyLength!=0)
			MPI_Send(data+r+1,MyLength,MPI_INT,id+exp2(m-1),MyID,MPI_COMM_WORLD);
	}

	/*处理器id+exp2(m-1)接受处理器id发送的消息*/
	if(MyID==id+exp2(m-1))
	{
		MPI_Recv(&MyLength,1,MPI_INT,id,id,MPI_COMM_WORLD,&status);
		if(MyLength!=0)
		{
			tmp=(int *)malloc(MyLength*sizeof(int));
			if(tmp==0) ErrMsg("Malloc memory error!");
			MPI_Recv(tmp,MyLength,MPI_INT,id,id,MPI_COMM_WORLD,&status);
		}
	}

	/*递归调用并行排序，对应于算法13.4步骤(1.4，1.5)*/

	/*用2^m-1个处理器对start--(r-1)的数据进行递归排序*/
	j=r-1-start;	
	MPI_Bcast(&j,1,MPI_INT,id,MPI_COMM_WORLD);
	/*(1.4)	para_quicksort(data,i,r-1,m-1,id)*/
	if(j>0)
		para_QuickSort(data,start,r-1,m-1,id,MyID);

	/*用2^m-1个处理器对(r+1)--end的数据进行递归排序*/
	j=MyLength;	
	MPI_Bcast(&j,1,MPI_INT,id,MPI_COMM_WORLD);
	/*(1.5)	para_quicksort(data,r+1,j,m-1,id+2m-1)*/
	if(j>0)
		para_QuickSort(tmp,0,MyLength-1,m-1,id+exp2(m-1),MyID);

	/*将排序好的数据由处理器id+exp2(m-1)发回id号处理器，对应于算法13.4步骤(1.6)*/
	/*(1.6)	P(id+2m-1) send data[r+1,m-1] back to Pid */
	if((MyID==id+exp2(m-1)) && (MyLength!=0))
		MPI_Send(tmp,MyLength,MPI_INT,id,id+exp2(m-1),MPI_COMM_WORLD);

	if((MyID==id) && (MyLength!=0))
		MPI_Recv(data+r+1,MyLength,MPI_INT,id+exp2(m-1),id+exp2(m-1),MPI_COMM_WORLD,&status);

}

/*
	* 函数名: QuickSort
	* 功能：对起止位置为start和end的数组序列，进行串行快速排序。
	* 输入：无序数组data[1,n]
	* 返回：有序数组data[1,n]
*/
QuickSort(int *data,int start,int end)
{
	int r;
	int i;

	if(start<end)
	{
		r=Partition(data,start,end);
		QuickSort(data,start,r-1);
		QuickSort(data,r+1,end);
	}
}

/*
	* 函数名: Partition
	* 功能：对起止位置为start和end的数组序列，将其分成两个非空子序列，
	*		其中前一个子序列中的任意元素小于后个子序列的元素。
	* 输入：无序数组data[1,n]
	* 返回: 两个非空子序列的分界下标
*/
int Partition(int *data,int start,int end)
{
	int pivo;
	int i, j;
	int tmp;

	pivo=data[end];			
	i=start-1;				/*i(活动指针)*/

	for(j=start;j<end;j++)
		if(data[j]<=pivo)
		{
			i++;			/*i表示比pivo小的元素的个数*/
			tmp=data[i];
			data[i]=data[j];
			data[j]=tmp;
		}
	
	tmp=data[i+1];
	data[i+1]=data[end];
	data[end]=tmp;			/*以pivo为分界，data[i+1]=pivo*/

	return i+1;
}

/*
	* 函数名: exp2
	* 功能：求2的num次幂
	* 输入：int型数据num
	* 返回: 2的num次幂
*/
int exp2(int num)
{
	int i;

	i=1;

	while(num>0)
	{
		num--;
		i=i*2;
	}
	
	return i;
}

/*
	* 函数名: log2
	* 功能：求以2为底的num的对数
	* 输入：int型数据num
	* 返回: 以2为底的num的对数
*/
int log2(int num)
{
	int i, j;

	i=1;	
	j=2;

	while(j<num)
	{
		j=j*2;
		i++;
	}

	if(j>num)
		i--;

	return i;
}

/*
	* 函数名: GetDataSize
	* 功能：读入待排序序列长度
*/
int GetDataSize()
{
	int i;

	while(TRUE)
	{
		printf("Input the Data Size :");
		scanf("%d",&i);
		/*读出正确的i，返回；否则，继续要求输入*/
		if((i>0) && (i<=65535))		

			break;
		ErrMsg("Wrong Data Size, must between [1..65535]");
	}
	return i;
}


/*输出错误信息*/
ErrMsg(char *msg)
{
	printf("Error: %s \n",msg);
}