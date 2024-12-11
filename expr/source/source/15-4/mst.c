#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <mpi.h>
#define A(i,j) A[i*N+j]
#define MST(i,j) MST[i*n*p+j]
#define MAX 1000
int N;
int n;
int p;
int i,j,k;

double l;

int *D,*C,*W,*J;

int *A;
int *MST;
int myid;
MPI_Status status;

/**输出结果MST**/
void printM()
{
	int i,j;
	if(myid==0){
		printf("the MST is:\n");
		for(i=0;i<N;i++)
		{
			for(j=0;j<N;j++)
				printf("%d ",MST[i*n*p+j]);
			printf("\n");
		}
	}
}

/**输出数组D内容**/
void printD()
{
	int i;
	if(myid==0){
		printf("the array D is:\n");
		for(i=0;i<N;i++)
		{
			printf("%d ",D[i]);
		}
		printf("\n");
	}

}


/**读入邻接矩阵**/
int readA()
{
	int i,j;
	int p1;
	p1=p;
	printf("Input the size of matrix:");
	scanf("%d",&N);
	n=N/p1;
	if(N%p1!=0) n++;
	A=(int*)malloc(sizeof(int)*(n*p1)*N);
	if(A==NULL){
		printf("Error when allocating memory\n");
		exit(0);
	}
	for(i=0;i<N;i++)
	for(j=0;j<N;j++)
		scanf("%d",&(A(i,j)));
	for(i=N;i<n*p1;i++)
	for(j=0;j<N;j++)
		A(i,j)=MAX-1;
	p=p1;
	return(0);

}


/**广播特定数组**/
void bcast(int *P)
{
  MPI_Bcast(P,N,MPI_INT,0,MPI_COMM_WORLD);
}

/**求最小的数学函数**/
int min(int a,int b)
{
  return(a<b?a:b);
}

/**检查MST是否已完成**/
int connected()
{
	int i;
	int flag;
	flag=1;
	for(i=1;i<N;i++)
		if(D[i]!=D[0])
		{
			flag=0;
			break;
		}
		return(flag);
}

/**为各顶点找出距离最近的树**/
void D_to_C()
{
	int i,j;
	for(i=0;i<n;i++){
	W[n*myid+i]=MAX;
	for(j=N-1;j>=0;j--)
		if((A(i,j)>0)&&(D[j]!=D[n*myid+i])&&(A(i,j)<=W[n*myid+i]))
		{
			C[n*myid+i]=D[j];
			W[n*myid+i]=A(i,j);
			J[n*myid+i]=j;
		}
	if(W[n*myid+i]==MAX) C[n*myid+i]=D[n*myid+i];
  }

}


/**为各树找出外连最短边**/
void C_to_C()
{
	int tempw,tempj;
	for(i=0;i<n;i++){
		tempj=N+1;
		tempw=MAX;
		for(j=N-1;j>=0;j--)
			if((D[j]==n*myid+i)&&(C[j]!=n*myid+i)&&(W[j]<=tempw))
			{
				C[myid*n+i]=C[j];
				tempw=W[j];
				tempj=j;
			}
		if(myid==0)
		{
			if((tempj<N)&&(J[tempj]<N)) 
			MST(tempj,J[tempj])=MST(J[tempj],tempj)=tempw;
			for(j=1;j<p;j++)
			{
				MPI_Recv(&tempj,1,MPI_INT,j,j,MPI_COMM_WORLD,&status);
				MPI_Recv(&tempw,1,MPI_INT,j,0,MPI_COMM_WORLD,&status);
				if((tempj<N)&&(tempw <N))MST(tempj,tempw)=MST(tempw,tempj)=A(tempj,tempw);
			}
		}
		else
		{
			MPI_Send(&tempj,1,MPI_INT,0,myid,MPI_COMM_WORLD);
			MPI_Send(&J[tempj],1,MPI_INT,0,0,MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

}


/**调整数组顶点的标识**/
void CC_to_C()
{
	for(i=0;i<n;i++)
    	C[myid*n+i]=C[C[myid*n+i]];

}


/**产生新树**/
void CD_to_D()
{
	for(i=0;i<n;i++)
		D[myid*n+i]=min(C[myid*n+i],D[C[myid*n+i]]);

}


/**释放动态内存**/
void freeall()
{
  free(A);
  free(D);
  free(C);

}


/**主函数**/
int main(int argc,char **argv)
{
	int i,j,k;
	int group_size;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&group_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	p=group_size;
	/**处理器0读入邻接矩阵**/
	if(myid==0)
	{
		readA();
		MST=(int*)malloc(sizeof(int)*n*p*n*p);
	}

	MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
	if(myid!=0){
		n=N/p;
		if(N%p!=0) n++;
	}

	D=(int*)malloc(sizeof(int)*(n*p));
	C=(int*)malloc(sizeof(int)*(n*p));
	W=(int*)malloc(sizeof(int)*(n*p));
	J=(int*)malloc(sizeof(int)*(n*p));

	if(myid!=0)
		A=(int*)malloc(sizeof(int)*n*N);

	/**数组D初始化，步骤(1)**/
	for(i=0;i<n;i++) D[myid*n+i]=myid*n+i;
	MPI_Gather(&D[myid*n],n,MPI_INT,D,n,MPI_INT,0,MPI_COMM_WORLD);
	bcast(D);
	/**处理器0向其它处理器发送必要信息**/
	if(myid==0)
		for(i=1;i<p;i++)
			MPI_Send(&A(i*n,0),n*N,MPI_INT,i,i,MPI_COMM_WORLD);
	else
		MPI_Recv(A,n*N,MPI_INT,0,myid,MPI_COMM_WORLD,&status);
	MPI_Barrier(MPI_COMM_WORLD);

	l=log(N)/log(2);
	i=1;
	/**主循环，步骤(2)**/
	while(!connected()){
		if(myid==0) printf("loop %d \n",i);
		printD();
		printM();

		/**步骤(2.1)**/
		D_to_C();
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gather(&C[n*myid],n,MPI_INT,C,n,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Gather(&W[n*myid],n,MPI_INT,W,n,MPI_INT,0,MPI_COMM_WORLD);
		bcast(C);
		bcast(W);
		MPI_Barrier(MPI_COMM_WORLD);

		/**步骤(2.2)**/
		C_to_C();
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Gather(&C[n*myid],n,MPI_INT,C,n,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Gather(&C[n*myid],n,MPI_INT,D,n,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		/**步骤(2.3)**/
		if(myid==0)
			for(j=0;j<n;j++) D[j]=C[j];

		/**步骤(2.4)**/
		for(k=0;k<l;k++){
			bcast(C);
			CC_to_C();
			MPI_Gather(&C[n*myid],n,MPI_INT,C,n,MPI_INT,0,MPI_COMM_WORLD);
		}
		bcast(C);
		bcast(D);

		/**步骤(2.5)**/
		CD_to_D();
		MPI_Gather(&D[n*myid],n,MPI_INT,D,n,MPI_INT,0,MPI_COMM_WORLD);
		bcast(D);
		
		i++;
	}
	if(myid==0)printf("loop %d \n",i);
	printD();
	printM(); 
	freeall();
	MPI_Finalize();
	return(0);
}
