#include <mpi.h>
#include <stdio.h>

int n;
double xtemp[20],ytemp[20];
double x,y;
int s,mys;
int group_size,my_rank;

/*判断射线与线段是否有交点*/
int cal_inter(int number,int i,double x,double y)
{
	double x1,y1,x2,y2,temp;
  	int result;
  	result=0;

  	if(number+i*group_size>=n)
		return result;
	x1=xtemp[number+i*group_size];
  	y1=ytemp[number+i*group_size];
  	x2=xtemp[(number+1+i*group_size)%n];
  	y2=ytemp[(number+1+i*group_size)%n];
 
 	if(y1>y2) 
  	{
   		temp=x1;
   		x1=x2;
   		x2=temp;
   		temp=y1;
   		y1=y2;
   		y2=temp;
  	}

    /*判断竖直边的情况*/
  	if(x1==x2)
  	{
   		if((x>x1)&&(y<=y2)&&(y>y1))
     		result=1;
   		else
			result=0;
        /*点在竖直边上，应该对result赋一个比较的大的值，这里是100*/
		if((x==x1)&&((y-y1)*(y2-y)>=0))
			result=100;
	}
	else
	{
		/*非竖直边，非水平边*/
        if (y1!=y2)
		{
			temp=x2+(y-y2)*(x2-x1)/(y2-y1);
			/*交点刚好在边上，且不为下顶点*/
     		if((temp<x)&&(y<=y2)&&(y>y1))
        			result=1;
     		else
        			result=0;

     		/*点在边上，应该对result赋一个比较的大的值，这里是100*/
			if((temp==x)&&((y-y2)*(y1-y)>=0))
				result=100;
		}
		else
		{
		/*点在水平边上，应该对result赋一个比较的大的值，这里是100*/
     		if((y==y1)&&((x1-x)*(x-x2)>=0)) result=100;
		}
	}
	return result;
}

main(int argc,char* argv[])
{
	int i;

    MPI_Init(&argc,&argv);  
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&group_size);

  	/*各处理器计数器初始化*/
  	mys=0;

    /*主处理器读入多边形顶点和要判断的点的坐标*/
  	if(my_rank==0)
  	{
  		printf("请输入点的个数:");
 		scanf("%d",&n);
  		printf("请输入各点的坐标\n");
  		for(i=0;i<n;i++)
   		{
   			printf("%d:",i);
   			scanf("%lf",&xtemp[i]);
   			scanf("%lf",&ytemp[i]);
   		}
  		printf("请输入要判断点的坐标\n");
  		scanf("%lf %lf",&x,&y);
  	}
	MPI_Barrier(MPI_COMM_WORLD);
	/*把多边形的顶点数、顶点坐标与要判别的点的坐标播送给所有进程*/
  	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
  	MPI_Bcast(&x,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  	MPI_Bcast(&y,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  	MPI_Bcast(xtemp,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
  	MPI_Bcast(ytemp,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

  	MPI_Barrier(MPI_COMM_WORLD);
    /*每一个处理器处理n/group_size条边上的情况并求和*/
  	for(i=0;i<n/group_size+1;i++)
    {
    		mys+=cal_inter(my_rank,i,x,y); 
	}

  	MPI_Barrier(MPI_COMM_WORLD);

    /*把mys的值规约到s*/
    MPI_Reduce(&mys,&s,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    /*根据s值确定输出结果*/
	if(my_rank==0)
   	{
		if(s>=100)  
			printf("vertex p is in polygon\n");
    	else
       		if(s%2==1)
        			printf("vertex p is in polygon\n");
       		else
         			printf("vertex p is out of polygon\n");
	}
   	MPI_Finalize();
} 

