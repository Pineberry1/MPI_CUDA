#include <stdio.h>
#include <mpi.h>
#include <math.h>

int my_rank,group_size,sub_group;
int n_all;
double pointx[30],pointy[30];
double tempx[2],tempy[2],xtemp[4],ytemp[4];
MPI_Status status;
int location[30];

/*获得y坐标最小的点,坐标值保存到保存到tempx,tempy数组中*/
void getymin( )
{
    int i,index=0;
    double tempvalue=pointy[0];

    for(i=1;i<n_all;i++)
    {
        if(pointy[i]<tempvalue)
        {
            tempvalue=pointy[i];
            index=i;
        }
        else
        {
			/*若y坐标相同,则取x坐标小者。这是为了保证(tempx[0],tempy[0])->(tempx[1],tempy[1])为逆时针方向*/
            if(pointy[i]==tempvalue)
                if(pointx[index]>pointx[i])
                    index = i;
        }
    }
    tempx[0]=pointx[index];
    tempy[0]=pointy[index];

	/*如果y坐标最小的极点只有1点,就复制该点*/
    tempx[1]=tempx[0];
    tempy[1]=tempy[0];

	/*考察极点是否多于1点*/
    for(i=0;i<n_all;i++)
        if((pointy[i]==tempy[1])&&(pointx[i]>tempx[1]))
    {
		/*极点多于1点,取x坐标大者,保存最后一个极点的坐标*/
        tempx[1]=pointx[i];
        tempy[1]=pointy[i];
    }
    return;
}


/*获得y坐标最大的点,保存到tempx,tempy数组中*/
void getymax()
{
    int i,index=0;
    double temp=pointy[0];

    for(i=1;i<n_all;i++)
    {
        if(pointy[i]>temp)
        {
            temp=pointy[i];
            index=i;
        }
        else
        {
			/*若y坐标相同,则取x坐标大者。这是为了保证(tempx[0],tempy[0])->(tempx[1],tempy[1])为逆时针方向*/
            if(temp==pointy[i])
                if(pointx[i]>pointx[index])
                    index=i;
        }
    }
	/*保存YMAX极点坐标*/
    tempx[0]=pointx[index];
    tempy[0]=pointy[index];

	/*如果y坐标最大的极点只有1点,就复制该点*/
    tempx[1]=tempx[0];
    tempy[1]=tempy[0];

	/*考察极点是否多于1点*/
    for(i=0;i<n_all;i++)
    {
        if((pointy[i]==tempy[1])&&(pointx[i]<tempx[1]))
        {
			/*极点多于1点,取x坐标小者,保存最后一个极点的坐标*/
            tempx[1]=pointx[i];
            tempy[1]=pointy[i];
        }
    }

    return;
}


/*获得x坐标最小的点,坐标保存到tempx,tempy数组中*/
void getxmin()
{
    int i,index=0;
    double temp=pointx[0];

    for(i=1;i<n_all;i++)
    {
        if(pointx[i]<temp)
        {
            temp=pointx[i];
            index=i;
        }
        else
        {
			/*若x坐标相同,则取y坐标大者。这是为了保证(tempx[0],tempy[0])->(tempx[1],tempy[1])为逆时针方向*/
            if(pointx[i]==temp)
                if(pointy[index]<pointy[i])
                    index=i;
        }
    }
	/*保存XMIN极点坐标*/
    tempx[0]=pointx[index];
    tempy[0]=pointy[index];

	/*如果x坐标最小的极点只有1点,就复制该点*/
    tempx[1]=tempx[0];
    tempy[1]=tempy[0];

	/*考察极点是否多于1点*/
    for(i=0;i<n_all;i++)
    {
        if((pointx[i]==tempx[1])&&(pointy[i]<tempy[1]))
        {
			/*极点多于1点,取y坐标小者,保存最后一个极点的坐标*/
            tempx[1]=pointx[i];
            tempy[1]=pointy[i];
        }
    }
    return;
}


/*获得x坐标最大的点,其坐标值保存保存到tempx,tempy数组中*/
void getxmax()
{
    int i,index=0;
    double temp=pointx[0];

    for(i=1;i<n_all;i++)
    {
        if(pointx[i]>temp)
        {
            temp=pointx[i];
            index=i;
        }
        else
        {
			/*若x坐标相同,则取y坐标小者。这是为了保证(tempx[0],tempy[0])->(tempx[1],tempy[1])为逆时针方向*/
            if(temp==pointx[i])
                if(pointy[index]>pointy[i])
                    index=i;
        }
    }
	/*保存XMAX极点坐标*/
    tempx[0]=pointx[index];
    tempy[0]=pointy[index];

	/*如果x坐标最大的极点只有1点,就复制该点*/
    tempx[1]=tempx[0];
    tempy[1]=tempy[0];

	/*考察极点是否多于1点*/
    for(i=0;i<n_all;i++)
    {
        if((pointx[i]==tempx[1])&&(pointy[i]>tempy[1]))
        {
			/*极点多于1点,取y坐标大者,保存第2个极点的坐标*/
            tempx[1]=pointx[i];
            tempy[1]=pointy[i];
        }
    }
    return;
}


/*temp[x],temp[y]设置两个值的作用是为了有多个最值时，保存首尾两个，后面划分
区域时分别以这两个为端点，中间就没有其他的点了*/

/*确定是属于哪一部分*/
void getincludedvertex(int tag)
{
    double linea,lineb,dist;
    int i,count=0;

	/*若该边已退化成一个点, 则除此点外无其它点在处理区域内*/
    if((tempx[1]==tempx[0])&&(tempy[1]==tempy[0]))
    {
        n_all=1;
        pointx[0]=tempx[1];
        pointy[0]=tempy[1];
    }
    else
    {
		/*根据两端点坐标确定直线的斜率和参数*/
        linea=(tempy[1]-tempy[0])/(tempx[1]-tempx[0]);
        lineb=tempy[1]-linea*tempx[1];
		/*根据点在直线的哪一侧来确定该顶点是否在处理区域内*/
        for(i=0;i<n_all;i++)
        {
            dist=linea*pointx[i]+lineb-pointy[i];
            if(tag*dist<0)
            {
                pointx[count]=pointx[i];
                pointy[count++]=pointy[i];
            }
        }
		/*调整各点的位置以加入两个端点*/
        for(i=0;i<count;i++)
        {
            pointx[count-i]=pointx[count-1-i];
            pointy[count-i]=pointy[count-1-i];
        }
		/*将两个端点放在两端*/
        pointx[0]=tempx[0];
        pointy[0]=tempy[0];
        pointx[count+1]=tempx[1];
        pointy[count+1]=tempy[1];
        count+=2;
		/*更新n_all为要处理的新的表列中的点的数量*/
        n_all=count;
    }
}


/*确定此点的后一点的序号，并放入location[]中*/
void nextindex(int i,int x,int y)
{
    double x1,y1,temp,valuemax=0;
    int j;

    x1=pointx[i];
    y1=pointy[i];

	/*遍历各顶, 将有最小极角点的序号作为自己的Nextindex值*/
    for(j=0;j<n_all;j++)
    {
        if(((pointx[j]-x1)*x+(pointy[j]-y1)*y)>0)
        {
			/*由极角的余弦或其余角的正弦来判断极角大小*/
            temp=((pointx[j]-x1)*x+(pointy[j]-y1)*y)/(sqrt((pointx[j]-x1)*(pointx[j]-x1)+(pointy[j]-y1)*(pointy[j]-y1)));
            if(temp>valuemax)
            {
                location[i]=j;
                valuemax=temp;
            }
        }
    }
    if(valuemax==0)
        location[i]=-1;
}


/*输出点的序列*/
void output(int rank)
{
    int j=0,index,temp;
    double xtem,ytem;
    int flag=0;

    printf("输出的是第%d部分的点\n",rank/sub_group);
    if(rank==0)
        flag=1;
    if(flag==0)
        if(!((pointx[n_all-1]==tempx[0])&&
        (pointy[n_all-1]==tempy[0])))
            flag=1;

	/*将nextindex反向,使输出的点按逆时针方向,以保证各行主处理器输出顺序相邻*/
    index=location[j];
    while(location[index]!=-1)
    {
        temp=location[index];
        location[index]=j;
        j=index;
        index=temp;
    }
    location[index]=j;
    location[0]=-1;

	/*按照索引顺序输出各点*/
    j=n_all-1;
    while(location[j]!=-1)
    {
        if(!((j==n_all-1)&&(flag==0)))
        {
            printf("%.2lf,%.2lf\n",pointx[j],pointy[j]);
            xtem=pointx[j];
            ytem=pointy[j];
        }
        j=location[j];
    }

	/*判断是不是重合*/
    if(!((j==n_all-1)&&(flag==0)))
    {
        printf("%.2lf,%.2lf\n",pointx[j],pointy[j]);
        xtem=pointx[j];
        ytem=pointy[j];
    }

    if((rank==3*sub_group)&&!((xtem==tempx[1])&&(ytem==tempy[1])))
        printf("%.2lf,%.2lf\n",tempx[1],tempy[1]);
    return;
}


main(int argc,char *argv[])
{
    int i;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	/*本程序至少要4个处理器才能正常执行*/
    if (group_size < 4)
    {
        if (my_rank==0)
        {
            printf("Need 4 or more processors to run!\n");
        }
        MPI_Finalize();
        exit(0);
    }

    if(my_rank==0)
    {
        printf("please input all the vertexes!\nfirst is number!\n");
        printf("please input the Number:");
        scanf("%d",&n_all);
        printf("please input the vertex:\n");
        for(i=0;i<n_all;i++)
        {
            scanf("%lf",&pointx[i]);
            scanf("%lf",&pointy[i]);
        }
    }
	/*处理器0接收输入的顶点坐标*/

    sub_group = group_size / 4;

    MPI_Bcast(&n_all,1,MPI_INT,0,MPI_COMM_WORLD);
	/*广播顶点总数*/
	/*第1行的行主处理器,向第2,3,4行的行主处理器发送顶点坐标，对应于算法17.6步骤(1.1)*/
    if(my_rank==0)
    {
        for(i=1;i<4;i++)
        {
            MPI_Send(pointx,n_all,MPI_DOUBLE,i*sub_group,i,MPI_COMM_WORLD);
            MPI_Send(pointy,n_all,MPI_DOUBLE,i*sub_group,i,MPI_COMM_WORLD);
        }
    }
    if((my_rank>0)&&(my_rank<4*sub_group)&&(my_rank%sub_group==0))
    {
        MPI_Recv(pointx,n_all,MPI_DOUBLE,0,my_rank/sub_group,MPI_COMM_WORLD,&status);
        MPI_Recv(pointy,n_all,MPI_DOUBLE,0,my_rank/sub_group,MPI_COMM_WORLD,&status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

	/*计算极点,并把他们的坐标分别存储在4个行主处理器中*/
	/*第1行的行主处理器计算YMIN*/
    if(my_rank==0)
        getymin();
	/*第2行的行主处理器计算XMAX*/
    if(my_rank==sub_group)
        getxmax();
	/*第3行的行主处理器计算YMAX*/
    if(my_rank==2*sub_group)
        getymax();
	/*第4行的行主处理器计算XMIN*/
    if(my_rank==3*sub_group)
        getxmin();

    MPI_Barrier(MPI_COMM_WORLD);

	/*将四条由极点组成的边存储到每一行的行主处理器上*/
    if((my_rank>0)&&(my_rank<4*sub_group)&&(my_rank%sub_group==0))
    {
		/*各行主处理器发送消息给进程0, 告知极点坐标*/
        MPI_Send(&tempx[1],1,MPI_DOUBLE,0,my_rank,MPI_COMM_WORLD);
        MPI_Send(&tempy[1],1,MPI_DOUBLE,0,my_rank,MPI_COMM_WORLD);
    }
    if(my_rank==0)
    {
        for(i=1;i<4;i++)
        {
            MPI_Recv(&xtemp[i],1,MPI_DOUBLE,i*sub_group,i*sub_group,MPI_COMM_WORLD,&status);
            MPI_Recv(&ytemp[i],1,MPI_DOUBLE,i*sub_group,i*sub_group,MPI_COMM_WORLD,&status);
        }
        xtemp[0]=tempx[1];
        ytemp[0]=tempy[1];
        tempx[1]=xtemp[3];
        tempy[1]=ytemp[3];

        for(i=1;i<4;i++)
        {
			/*进程0将相关的极点坐标发送给各行主处理器*/
            MPI_Send(&xtemp[i-1],1,MPI_DOUBLE,i*sub_group,i*sub_group,MPI_COMM_WORLD);
            MPI_Send(&ytemp[i-1],1,MPI_DOUBLE,i*sub_group,i*sub_group,MPI_COMM_WORLD);
        }
    }
    else
    {
        if((my_rank<4*sub_group)&&(my_rank%sub_group==0))
        {
			/*各行主处理器接收进程0的消息,得到并存储由极点构成的边*/
            MPI_Recv(&tempx[1],1,MPI_DOUBLE,0,my_rank,MPI_COMM_WORLD,&status);
            MPI_Recv(&tempy[1],1,MPI_DOUBLE,0,my_rank,MPI_COMM_WORLD,&status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

	/*确定四边形中的顶点，并将其余顶点归入四个三角区中*/
	/*四个行主处理器同时判断顶点是否处于自身所在的区域 */

	/*保留属于该区域内的点，每个行主处理器得到要处理的新的表列*/
	/*下面得到属于此范围的点*/
    if(my_rank==0)
        getincludedvertex(-1);
    if(my_rank==sub_group)
        getincludedvertex(-1);
    if(my_rank==2*sub_group)
        getincludedvertex(1);
    if(my_rank==3*sub_group)
        getincludedvertex(1);

    MPI_Barrier(MPI_COMM_WORLD);

    if((my_rank<4*sub_group)&&(my_rank%sub_group==0))
    {
        for(i=1;i<sub_group;i++)
        {
			/*各行主处理器向该行的其他处理器发送新的表列中的顶点数量*/
            MPI_Send(&n_all,1,MPI_INT,my_rank+i,i,MPI_COMM_WORLD);
        }
    }
    else if (my_rank<4*sub_group)
    {
		/*各处理器从该行的主处理器接收新的表列中的顶点数量*/
        MPI_Recv(&n_all,1,MPI_INT,my_rank/sub_group*sub_group,my_rank%sub_group,MPI_COMM_WORLD,&status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if((my_rank<4*sub_group)&&(my_rank%sub_group==0))
    {
        for(i=1;i<sub_group;i++)
        {
			/*各行主处理器向该行的其他处理器发送新的表列中的顶点坐标*/
            MPI_Send(pointx,n_all,MPI_DOUBLE,my_rank+i,i,MPI_COMM_WORLD);
            MPI_Send(pointy,n_all,MPI_DOUBLE,my_rank+i,i,MPI_COMM_WORLD);
        }
    }
    else if (my_rank<4*sub_group)
    {
        MPI_Recv(pointx,n_all,MPI_DOUBLE,my_rank/sub_group*sub_group,my_rank%sub_group,MPI_COMM_WORLD,&status);
        MPI_Recv(pointy,n_all,MPI_DOUBLE,my_rank/sub_group*sub_group,my_rank%sub_group,MPI_COMM_WORLD,&status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

	/*每一行上的处理器对属于同一区域的点计算极角；将有最小极角点的序号作为自己的Nexindex值建立各进程中点的nextindex索引*/
    if((n_all>1)&&(my_rank<4*sub_group))
    {
        for(i=my_rank%sub_group;i<n_all-1;i+=sub_group)
        {
            if(my_rank/sub_group==0)
                nextindex(i,-1,0);
            if(my_rank/sub_group==1)
                nextindex(i,0,-1);
            if(my_rank/sub_group==2)
                nextindex(i,1,0);
            if(my_rank/sub_group==3)
                nextindex(i,0,1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

	/*将个进程中点的nextindex索引发送到中心结点*/
    if((my_rank<4*sub_group)&&(my_rank%sub_group!=0))
    {
        for(i=my_rank%sub_group;i<n_all-1;i+=sub_group)
            MPI_Send(&location[i],1,MPI_INT,my_rank/sub_group*sub_group,my_rank,MPI_COMM_WORLD);
    }
    else if (my_rank<4*sub_group)
    {
        for(i=1;i<n_all-1;i++)
        {
            if(i%sub_group!=0)
            {
                MPI_Recv(&location[i],1,MPI_INT,my_rank+i%sub_group,my_rank+i%sub_group,MPI_COMM_WORLD,&status);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    location[n_all-1]=-1;

    MPI_Barrier(MPI_COMM_WORLD);

	/*每个行主处理器按照点的Nextindex索引输出自己处理器上的点)*/
    if(my_rank==0)
    {
		/*第1行输出各顶点的坐标*/
        output(my_rank);
		/*将最后的一个点传送给第2行主处理器判断是不是相同*/
        MPI_Send(&pointx[0],1,MPI_DOUBLE,sub_group,sub_group,MPI_COMM_WORLD);
        MPI_Send(&pointy[0],1,MPI_DOUBLE,sub_group,sub_group,MPI_COMM_WORLD);
		/*将起始点传送给第4行主处理器判断是不是相同*/
        MPI_Send(&pointx[n_all-1],1,MPI_DOUBLE,3*sub_group,3*sub_group,MPI_COMM_WORLD);
        MPI_Send(&pointy[n_all-1],1,MPI_DOUBLE,3*sub_group,3*sub_group,MPI_COMM_WORLD);
    }
    if(my_rank==sub_group)
    {
		/*接收第1行主处理器发送过来的最后一个点,也即本行起始点,以判断是不是相同*/
        MPI_Recv(&tempx[0],1,MPI_DOUBLE,0,sub_group,MPI_COMM_WORLD,&status);
        MPI_Recv(&tempy[0],1,MPI_DOUBLE,0,sub_group,MPI_COMM_WORLD,&status);
		/*将最后的一个点传送给第3行主处理器判断是不是相同*/
        MPI_Send(&pointx[0],1,MPI_DOUBLE,2*sub_group,2*sub_group,MPI_COMM_WORLD);
        MPI_Send(&pointy[0],1,MPI_DOUBLE,2*sub_group,2*sub_group,MPI_COMM_WORLD);
    }
    if(my_rank==2*sub_group)
    {

        MPI_Recv(&tempx[0],1,MPI_DOUBLE,sub_group,2*sub_group,MPI_COMM_WORLD,&status);
        MPI_Recv(&tempy[0],1,MPI_DOUBLE,sub_group,2*sub_group,MPI_COMM_WORLD,&status);

        MPI_Send(&pointx[0],1,MPI_DOUBLE,3*sub_group,3*sub_group,MPI_COMM_WORLD);
        MPI_Send(&pointy[0],1,MPI_DOUBLE,3*sub_group,3*sub_group,MPI_COMM_WORLD);
    }
    if(my_rank==3*sub_group)
    {

        MPI_Recv(&tempx[0],1,MPI_DOUBLE,2*sub_group,3*sub_group,MPI_COMM_WORLD,&status);
        MPI_Recv(&tempy[0],1,MPI_DOUBLE,2*sub_group,3*sub_group,MPI_COMM_WORLD,&status);

        MPI_Recv(&tempx[1],1,MPI_DOUBLE,0,3*sub_group,MPI_COMM_WORLD,&status);
        MPI_Recv(&tempy[1],1,MPI_DOUBLE,0,3*sub_group,MPI_COMM_WORLD,&status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

	/*第2行输出各顶点的坐标*/
    if(my_rank==sub_group)
        output(my_rank);

    MPI_Barrier(MPI_COMM_WORLD);

	/*第3行输出各顶点的坐标*/
    if(my_rank==2*sub_group)
        output(my_rank);

    MPI_Barrier(MPI_COMM_WORLD);

	/*第4行输出各顶点的坐标*/
    if(my_rank==3*sub_group)
        output(my_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
