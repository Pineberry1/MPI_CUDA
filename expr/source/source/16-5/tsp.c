#include <stdio.h>
#include <mpi.h>

/*定义距离的最大值*/
#define MAXDISTANCE 999999

/*定义点的最大个数*/
#define MAXPOINT    20

int my_rank,group_size,n;
int point[MAXPOINT];
double dist[MAXPOINT][MAXPOINT];
double maxvalue[MAXPOINT];
int flag=1;

void sub_tsp(int rank)
{
    int itemp,jtemp,ijtemp,k;
    double temp;
    int i,j;

    for(i=0;i<n-2;i++)
        for(j=i+2;j<n;j++)
    {
		/*分配给相应的处理器*/
        if(my_rank==((i+j)%group_size))
        {
			/*求出对边（i ，j）的改进权*/
            temp=dist[point[i]][point[i+1]]+dist[point[j]][point[j+1]]-dist[point[i]][point[j]]-dist[point[i+1]][point[j+1]];

			/*判断是不是更大的改进权*/
            if(temp>maxvalue[rank])
            {
                maxvalue[rank]=temp;
                itemp=i;
                jtemp=j;
            }
        }
    }

	/*如果最大的改进权大于0，相应的对边(itemp，jtemp)，则进行位置的调整，改良原来的Hamilton圈*/
    if(maxvalue[rank]>0)
    {
        for(k=itemp+1;k<=(itemp+1+jtemp)/2;k++)
        {
            ijtemp=point[k];
            point[k]=point[itemp+jtemp+1-k];
            point[itemp+jtemp+1-k]=ijtemp;
        }
    }

    return;
}


/*求最大改进权*/
int selectmax()
{
    int i,j;
    double temp=0;
    for(i=0;i<group_size;i++)
    {
        if(maxvalue[i]>temp)
        {
            j=i;
            temp=maxvalue[i];
        }

    }
    if(temp==0)
        return -1;
    return j;

}


/*输出较优的回路和回路的总长度*/
void output()
{
    int i;
    double sum=0.0;
    for(i=0;i<n;i++)
        sum+=dist[point[i]][point[i+1]];

	/*如果算法运行结束的时候，得到的Hamilton圈的长度大于距离的最大值，说明原图中不存在圈*/
    if((sum>=MAXDISTANCE)&&(flag==0))
    {
        printf("原图中不存在圈!  \n");
        return;
    }

    for(i=0;i<n;i++)
        printf("%d->",point[i]);

    printf("%d\n",point[n]);
    printf("距离的和是%.1lf\n",sum);

    return;
}


void main(int argc,char *argv[])
{
    int i,j;
    MPI_Status status;

	/*启动计算*/
    MPI_Init(&argc,&argv);

	/*找自己的id，存放在my_rank 中*/
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	/*找进程数，存放在group_size 中*/
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);

	/*输入点之间的距离矩阵*/
    if(my_rank==0)
    {
		/*输入点的个数，存放在n中*/
        printf("请输入点的个数:");
        scanf("%d",&n);

		/*点的个数不能大于MAXPOINT*/
        if(n>MAXPOINT)
        {
            printf("点的个数不能大于%d!  \n",MAXPOINT);
            goto terminal;
        }

		/*排除n＝0，1，2的情况*/
        if(n<3)
        {
            printf("TSP 问题在n＝%d的情况下没意义！！  \n",n);
            goto terminal;
        }

		/* 输入点 i和点j之间的 距离，存放在dist[i][j]*/

        for(i=0;i<n-1;i++)
        {
            for(j=i+1;j<n;j++)
            {
                printf("%d<->%d: ",i,j);
                scanf("%lf",&dist[i][j]);
                dist[j][i]=dist[i][j];
            }
        }

        for(i=0;i<=n;i++)
        {
            dist[i][i]=0;
            dist[n][i]=dist[0][i];
            dist[i][n]=dist[i][0];
        }
    }

	/*从根进程向所有进程发送n*/
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

	/*同步所有进程*/
    MPI_Barrier(MPI_COMM_WORLD);

	/*从根进程向所有进程发送点0到其他点的距离*/
    for(i=0;i<=n;i++)
        MPI_Bcast(&dist[i][0],n+1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

	/*构造初始的Hamilton圈0－>1－>2－> ...... －>n-1－>0*/

    for(i=0;i<=n;i++)
        point[i]=i%n;

	/*flag标志还能不能对Hamilton圈进行改良*/
    while(flag==1)
    {
		/*输出每次改良后的Hamilton圈及其总长度*/
        if(my_rank==0)
            output();

        maxvalue[my_rank]=0;
        sub_tsp(my_rank);

		/*非根进程将所有的改进权传递到0处理器*/
        if(my_rank>0)
            MPI_Send(&maxvalue[my_rank],1,MPI_DOUBLE,0,my_rank,MPI_COMM_WORLD);

		/*根进程接受其他进程的改进权，由其判断最大的改进权*/

        if(my_rank==0)
        {
            for(i=1;i<group_size;i++)
                MPI_Recv(&maxvalue[i],1,MPI_DOUBLE,i,i,MPI_COMM_WORLD,&status);
            j=selectmax();
        }
        MPI_Barrier(MPI_COMM_WORLD);

		/*从根进程向所有进程发送点最大的改进权*/
        MPI_Bcast(&j,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

		/*如果最大改进权为0，则表示没有任何改进，不能再对Hamilton圈进行改良*/
        if(j==-1)
            flag=0;
		/*否则将最大权对应的处理器rank值广播到处理器中，对应的处理器得到改进的圈*/
        else
            MPI_Bcast(point,n+1,MPI_INT,j,MPI_COMM_WORLD);
    }

	/*输出计算的最终结果*/
    if(my_rank==0)
        output();

	/*结束计算*/
    terminal:
    MPI_Finalize();
}
