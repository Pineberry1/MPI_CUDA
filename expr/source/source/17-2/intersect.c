#include <stdio.h>
#include <mpi.h>
//进程号和处理器数目
int my_rank,group_size;
//第一个多边形的端点
double pointx1[20],pointy1[20];
//第二个多边形的端点
double pointx2[20],pointy2[20];
//多边行的边数
int n_poly,m_poly;
//判断的结果
int result,localresult;
/*
 *函数名:if_not_parallel
 *功能:两条直线不垂直的情况下,判断是否相交
 *输入:(x1,y1),(x2,y2)是线段一的两个端点的坐标
 *     (x3,y3),(x4,y4)是线段二的两个端点的坐标
 *输出:返回整型值判断两条线段是否相交
 */
int if_intersect_np(double x1,double y1,double x2,double y2, double x3,double y3,double x4,double y4)
{
	double x;
	x=((x1*y2-x2*y1)/(x2-x1)+(x4*y3-x3*y4)/(x4-x3))/((y2-y1)/(x2-x1)-(y4-y3)/(x4-x3));
	if(((x1-x)*(x-x2)>=0)&&((x3-x)*(x-x4)>=0))
	return 1;
	else
	return 0;
}
/*
 *函数名:if_intersect_1v
 *功能:线段一是垂直的,判断线段二与一是否相交
 *输入:(x1,y1),(x2,y2)是线段一的两个端点的坐标
 *     (x3,y3),(x4,y4)是线段二的两个端点的坐标
 *输出:返回整型值判断两条线段是否相交
 */
int if_intersect_1v(double x1,double y1,double x2,double y2, double x3,double y3,double x4,double y4)
{
	double y;
	if(x3==x4)	
	{
		//两线段平行，不在一条直线上
		if(x3!=x1)	
		return 0;
		else if((y4<y1)||(y2<y3))
		return 0;
		else
		return 1;
	}
	//线段一垂直,线段二不垂直
	else
	{
		//线段一所在直线与线段二所在直线的交点纵坐标
		y = (y4 - y3)/(x4 - x3)*(x1 - x4)+ y4;
		if((y>=y1)&&(y<=y2)&&(y<=y4)&&(y>=y3)&&((x4-x1)*(x3-x1)<=0))
		return 1;
		else
		return 0;
	}
}
/*
 *函数名:if_intersect
 *功能:判断线段一(x1,y1),(x2,y2)与线段二(x3,y3),(x4,y4)是否相交
 *输入:(x1,y1),(x2,y2)是线段一的两个端点的坐标
 *     (x3,y3),(x4,y4)是线段二的两个端点的坐标
 *输出:返回整型值判断两条线段是否相交
 */
int if_intersect(double x1,double y1,double x2,double y2, double x3,double y3,double x4,double y4)
{
	double temp;
	int result;
	//将坐标大的换到第二个点
	if(y1>y2)
	{
		temp = x1;
		x1 = x2;
		x2 = temp;
		temp = y1;
		y1 = y2;
		y2 = temp;
	}
	if(y3>y4)
	{
		temp = x3;
		x3 = x4;
		x4 = temp;
		temp = y3;
		y3 = y4;
		y4 = temp;
	}
	//线段一为垂直线段
	if(x1==x2)	
	{
		result=if_intersect_1v(x1,y1,x2,y2,x3,y3,x4,y4);
	}	
	else
	{
		/*线段一不垂直，线段二垂直*/
		if(x3==x4)
		//交换线段一线段二位置，递归调用本函数判断两线段是否相交
		result= if_intersect(x3,y3,x4,y4,x1,y1,x2,y2);
		//线段一线段二皆不垂直
		else
		{
			//若两线段中存在平行于x轴的
			if((y1==y2)||(y3==y4))
			//交换线段一线段二中x,y位置，递归调用本函数判断两线段是否相交
			result= if_intersect(y1,x1,y2,x2,y3,x3,y4,x4);
			//线段一线段二平行
			else if((y4-y3)*(x2-x1)==(x4-x3)*(y2-y1))
			{
				//两线段所在直线在y轴上的截距相同，则两线段在同一直线上
				if((x4*y3-x3*y4)*(x2-x1)==(x2*y1-x1*y2)*(x4-x3))
				{
					//两线段在一条直线上，但不相交
					if((y4<y1)||(y2<y3))
					result = 0;
					//两线段在一条直线上，且相交
					else
					result = 1;
				}
				//两线段不在同一直线上
				else
				result = 0;
			}
			//两线段不平行
			else
			{
				result=if_intersect_np(x1,y1,x2,y2,x3,y3,x4,y4);
			}
		}
	}
	return result;
}
/*
 *函数名:Broadcast
 *功能:将两个多边形的数据传到个处理器
 *输入:无
 *输出:无
 */
void Broadcast()
{
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&n_poly,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&m_poly,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(pointx1,n_poly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(pointy1,n_poly,MPI_DOUBLE,0,MPI_COMM_WORLD); 
	MPI_Bcast(pointx2,m_poly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(pointy2,m_poly,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
}
/*
 *函数名:Judge()
 *功能:判断两个多边形是否相交
 *输入:无
 *输出:无
 */
void Judge()
{
	int i,j,k;
	/*对第二个多边形的每条边，若还未发现两多
	边形相交, 就判断是否与第一个多边形
	的边集相交, 对应于算法17.4步骤（3）
	*/
	result = 0;
	localresult = 0;
	for(i=0;(i<m_poly)&&(result==0);i++)
	{
		/*对处理器分配判断的第一个多边形的
		边集*/
		for(j=0;j<n_poly/group_size+1;j++)
		{
		/*累加第二个多边形当前边与第一
		个多边形边集相交的判断结
		果*/
			if(j*group_size+my_rank<n_poly)
			{
				k=j*group_size+my_rank;
				localresult+=if_intersect(pointx1[k],pointy1[k],pointx1[(k+1)%n_poly],pointy1[(k+1)%n_poly],pointx2[i],pointy2[i],pointx2[(i+1)%m_poly],pointy2[(i+1)%m_poly]);	
			}
		}
		/*统计第二个多边形与第一个多边形是
		否相交, 对应于算法17.4步骤(4)*/
		MPI_Scan(&localresult,&result,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	}
}
/*
 *函数名:GetData
 *功能:取得多边形的各种数据
 *输入:无
 *输出:无
 */
void GetData()
{
	int i;
	//0号处理器
	if(my_rank==0)
	{
		//输入多边形相关数据
		printf("please input first polygon\n");
		printf("please input the count of vertex:");
		scanf("%d",&n_poly);
		printf("please input the vertex\n");
		for(i=0;i<n_poly;i++)
		{
			scanf("%lf",&pointx1[i]);
			scanf("%lf",&pointy1[i]);
		}
		printf("please input second polygon\n");
		printf("please input the count of vertex:");
		scanf("%d",&m_poly);
		printf("please input the vertex\n");
		for(i=0;i<m_poly;i++)
		{
			scanf("%lf",&pointx2[i]);
			scanf("%lf",&pointy2[i]);
		}
	}
}
/*
 *函数名:main
 *功能:判断两个多边形是否相交
 *输入:argc为命令行参数个数
 *     argv为每个命令行参数组成的字符串数组。
 *输出：返回0代表程序正常结束
 */
int main(int argc,char * argv[])
{
	
    /*完成任务1:初始化*/   
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&group_size);
	/*完成任务2:从输入中获取数据*/
    GetData();
	/*完成任务3:将数据广播到每个处理器*/
	Broadcast();
	/*完成任务4:判断是否相交*/
    Judge();
    /*完成任务5:输出结果*/
	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank==0)
	{
		if(result!=0)
		printf("two polygons intersect\n");
		else
		printf("two polygons don't intersect\n");
	}
	MPI_Finalize();
}