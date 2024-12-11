/* FILE : sat.c*/
#include <stdio.h>
#include <stdlib.h>
/*包含有关时间的函数*/
#include <time.h>  
/*mpi库函数*/
#include "mpi.h"  
#include <math.h>

/*变量数目*/
#define NVARS 		100
/*子句数目*/
#define NCLAUSES 	3
/*每条子句的长度*/
#define LENGTH_CLAUSE 	3
/*fgets一行最多读入的字符数*/
#define	MAXLINE		60
#define	TRUE		1
#define	FALSE		0
/*既不是可满足的也不是不可满足的*/
#define	UNCERTAIN	-1
/*变量值为空*/
#define	NOVALUE		-1
#define	DEPTH		5
/*计算pow(x,y)*/
int Pow(int x, int y);
#define	COUNT		Pow(2,DEPTH)
#define	TASKTAG		11 


/**************/ 
/*主要数据结构*/
/**************/
/*变量*/
struct VARIABLE{
/*变量真值为１,假值为０,无值为－１*/
	int value;
};
/*文字*/
struct LITERAL{
	/*取值为VAR，－VAR和NOVALUE */
/* VAR取值从１到NVARS*/
	int value;
};
/*子句*/
struct CLAUSE{
	struct LITERAL lit[LENGTH_CLAUSE];
	int num_unassigned;
	/*被满足状态取值TRUE，FALSE或UNCERTAIN */
int state_of_satisfied;
};
/*合取范式*/
struct CNF{
	struct VARIABLE var[NVARS];
	struct CLAUSE clause[NCLAUSES];
	int num_uncertain;
	/*结果取值为TRUE，FALSE或UNCERTAIN*/
	int result;
};


/************/
/*进一步定义*/
/************/
/*测试一个子句是否可满足*/
int DavisPutnam();
/*当赋予一个新值时更新结构ＣＮＦ*/
void UpdateCNF();
/*检查该ＣＮＦ是否只有单一字句*/
int HasUnitClause();
/*检查该ＣＮＦ是否有纯文字*/
int HasPureLiteral();
/*从输入文件读数据初始化ＣＮＦ*/
void InitCNF();
/*从失败记号恢复ＣＮＦ*/
void Reverse();
/*返回一个整数的绝对值*/
int Abs();
/*将ＣＮＦ打包到一个整数缓冲*/
void PackCNF();
/*将一个整数缓冲释放到一个ＣＮＦ*/
void UnpackCNF();
/*根据任务更新该ＣＮＦ*/
void SetValue(); /*从Ｘ中取得ｎ位*/
unsigned GetBits(); /*从根接收任务然后计算并会送结果*/
void SlaveWork();
/*从ＣＮＦ中选变量*/
void PickVars();
/*检查变量是否在缓冲中*/
int NotInBuf(); 

/*主函数*/
int main(int argc,char ** argv)  
{   
	struct CNF  Cnf;  /*合取范式CNF*/
	int length=2+NVARS+NCLAUSES*(2+LENGTH_CLAUSE)+1;  	/*缓冲区长度*/
	int buf[length];  /*LENGHT_CLAUSE*NCLAUSE缓冲区*/
	MPI_Status status;  /*MPI--状态*/
	int i,j,myrank,size,BufResult;
	char* CnfFile;  /*输入CNF文件名*/
	FILE * in;  /*处理文件*/

	MPI_Init(&argc,&argv);  /*MPI--初始化*/
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);  /*MPI--次第号*/
	MPI_Comm_size(MPI_COMM_WORLD,&size);  /*MPI--数目*/

	/*root processor read data and initiate Cnf,then pack it to buff*/
	if(myrank==0)  /*进程0(主进程)*/
	{
		if (argc<2)  /*保证有两个参数*/
		{
			printf("You forgot to enter a parametes!\n");  /*参数数目不对*/
			MPI_Abort(MPI_COMM_WORLD,99);  /*MPI--退出*/
		}
		CnfFile=argv[1];  /*读取CNF文件名*/
		if ((in=fopen(CnfFile,"rt"))==NULL)	/*read file*/
		{
			printf("cannot open the result file\n");  /*文件不存在或没有相应的属性*/
			MPI_Abort(MPI_COMM_WORLD,99);  /*MPI--退出 */
		}

		if(myrank==0) printf("NVARS=%d NCLAUSES=%d LENGTH_CLAUSE=%d\n",NVARS,NCLAUSES,LENGTH_CLAUSE);
		InitCNF(&Cnf,in);  /*从输入文件中读取数据并初始化CNF*/		
		PackCNF(Cnf,buf);  /*将CNF打包到整型缓冲--便于MPI广播*/
	}

	/*broadcast buf to all the processors*/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(buf,length,MPI_INT,0,MPI_COMM_WORLD);  /*MPI--广播CNF包*/
	MPI_Barrier(MPI_COMM_WORLD);
	
	/*if it is root processor,distribute the task and wait result from slaves*/
	if ( myrank == 0)  /*进程0(主进程)*/
	{
		double start,end;
		start=MPI_Wtime();  /*MPI--时间(开始时间)*/
		/*send the first bulk of task*/
		for(i=0;i<size-1;i++)  /*对每一个从进程都发送消息*/
			MPI_Send(&i,1,MPI_INT,i+1,TASKTAG,MPI_COMM_WORLD);  /*发送CNF信息到从进程*/
		while (1)  /*一直执行*/
		{	             
			MPI_Recv(&BufResult,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);  /*从从进程接收结果*/
			if ( BufResult==TRUE )  /*SAT算法成功 */
			{
				end=MPI_Wtime();  /*MPI--时间(结束时间)*/
				i=COUNT+10;
				printf("\nSatisfied! time%f\n",end-start);  /*运行时间*/
				/*send i as the end signal*/
				for(j=0;j<size-1;j++)
				    MPI_Send(&i,1,MPI_INT,j+1,TASKTAG,MPI_COMM_WORLD);  /*向从进程发送终止信号*/
				break;  /*退出执行*/
			}
			else if(i>COUNT)  /*SAT算法超出规模*/
				{
					printf("\nUnsatisfied!\n");  /*SAT算法失败*/
					/*send i as the terminal signal*/
					for(j=0;j<size-1;j++)
						MPI_Send(&i,1,MPI_INT,j+1,TASKTAG,MPI_COMM_WORLD);  /*向从进程发送终止信号*/
					break;  /*退出执行*/
				}
			else
			{
				int dest=status.MPI_SOURCE;  /*确定消息发送目的地*/
				MPI_Send(&i,1,MPI_INT,dest,TASKTAG,MPI_COMM_WORLD);  /*向从进程发送相应CNF包*/
				i++;  /*下一步操作*/
			}
		}
	}
	else  /*其它进程(从进程)*/
	{
		unsigned	BufTask;  /*任务数*/
		UnpackCNF(&Cnf,buf);  /*将缓冲中的CNF解包*/
		while(1)  /*一直执行*/
		{
		MPI_Recv(&BufTask,1,MPI_INT,0,TASKTAG,MPI_COMM_WORLD,&status);  /*从进程0接受CNF包信息*/
			if ( BufTask <= COUNT)  /*未越界*/
				SlaveWork(Cnf,BufTask,&BufResult);  /*从0进程接受任务,计算,将结果返回0进程*/
			else break;  /*其它情况*/
		}
	}
	MPI_Finalize();  /*MPI--结束*/
	return(0);
}


/******************************************/ 
/*该函数从输入文件读入CNF数据并给Cnf赋值
输入：ｉｎ一包含CNF数据的文件
输出：ｐＣｎｆ一个指向Cnf结构的指针*/ 
/******************************************/ 
void InitCNF(struct CNF * pCnf,FILE* in)  /*从输入文件中读取数据并初始化CNF(参数:CNF式指针,处理文件)*/
{
	char prestr[MAXLINE];  /*文件缓存字符数组*/
	int i,j,temp;

	for(i=0;i<NVARS;i++)  /*在给定的变量数目范围(100)内*/
		pCnf->var[i].value=UNCERTAIN;  /*给CNF式变量赋初值(-1)*/

	for (i=0;i<NCLAUSES;i++) {  /*在给定的子句数目范围(350)内*/
		for (j=0;j<LENGTH_CLAUSE;j++) {  /*在给定的子句长度范围(3)内*/
			fscanf(in,"%d",&pCnf->clause[i].lit[j].value);  /*给CNF式文字赋初值*/
			/* printf("%d ",&ilauses[c].lit[l]); */
		}
		fscanf(in,"%d",&temp);  /*其它读入的值存入临时变量中*/
		/* printf("%d \n",temp); */
		pCnf->clause[i].num_unassigned=LENGTH_CLAUSE;  /*子句中未赋值数取为子句长度*/
		pCnf->clause[i].state_of_satisfied=UNCERTAIN;  /*子句状态取为不确定(-1)*/
	}
	pCnf->num_uncertain=NVARS;  /*CNF中未赋值数取为变量最大数目(100)*/
	pCnf->result=UNCERTAIN;  /*CNF结果取为不确定(-1)*/
	fclose(in);  /*关闭处理文件*/
}


/******************************************/ 
/*该函数计算结果是最主要的函数
输入：Cnf一个CNF结构
输出：DavisPutnam取值TRUE或FALSE*/ 
/******************************************/ 
int DavisPutnam(struct CNF  Cnf)  /*测试子句是否为SAT(参数:CNF式)*/
{
	int Serial;/*number of the var to be signed*/  /*待赋值的变量数*/
	/*test if this Cnf has had the result*/
	if (Cnf.result==TRUE) return TRUE;  /*如果CNF的一部分的结果取为真,返回真*/
	if (Cnf.result==FALSE) return FALSE;  /*如果CNF的一部分的结果取为假,返回假*/
	/*test if Cnf has unit clause*/
	if (HasUnitClause(Cnf,&Serial)==TRUE )  /*CNF含有单子句*/
	{
		Cnf.var[Abs(Serial)].value=(Serial>0)?TRUE:FALSE;  /*CNF变量赋值(0或1)*/
		printf("\nsingle%d",Serial);
		UpdateCNF(&Cnf,Abs(Serial));  /*为变量赋新值时更新CNF*/
		if (Cnf.result==TRUE) return TRUE;  /*如果CNF的一部分的结果取为真,返回真*/
		if (Cnf.result==FALSE) return FALSE;  /*如果CNF的一部分的结果取为假,返回假*/
		if (DavisPutnam(Cnf)==TRUE) return TRUE;  /*如果CNF的一部分测试子句后的结果取为真,返回真*/
		else return FALSE;  /*其它情况,返回假*/
	}
	/*test if Cnf has pure literal*/
	else if (HasPureLiteral(Cnf,&Serial)==TRUE )  /*CNF含有纯文字*/
		{
	        Cnf.var[Serial].value=TRUE;  /*CNF变量赋值为真(1)*/
			printf("\nPure %d",Serial);
			UpdateCNF(&Cnf,Serial);  /*为变量赋新值时更新CNF*/
			if (Cnf.result==TRUE) return TRUE;  /*如果CNF的一部分的结果取为真,返回真*/
			if (Cnf.result==FALSE) return FALSE;  /*如果CNF的一部分的结果取为假,返回假*/
			if (DavisPutnam(Cnf)==TRUE) return TRUE;  /*如果CNF的一部分测试子句后的结果取为真,返回真*/
			else return FALSE;  /*其它情况,返回假*/
		}
		/*pick a var without value*/
	     else  /*CNF不只含有单个文字且不含有纯文字*/
		 {
			for(Serial=0;Serial<NVARS;Serial++)  /*在给定的变量数目范围(100)内*/
				if (Cnf.var[Serial].value==NOVALUE)  break;  /*CNF式中变量没有值,退出*/
			printf("\ncommon%d",Serial);
			/*first assume this var is TRUE*/
			Cnf.var[Serial].value=TRUE;  /*CNF变量赋值为真(1)*/
			UpdateCNF(&Cnf,Serial);  /*为变量赋新值时更新CNF*/
			if (Cnf.result==TRUE) return TRUE;  /*CNF式中变量值为真(1),返回真(1)*/
			else if (Cnf.result==UNCERTAIN)  /*CNF式中变量值不确定*/
				if (DavisPutnam(Cnf)==TRUE) return TRUE;  /*如果CNF的一部分测试子句后的结果取为真,返回真*/
			/*Else try that #Serial is FALSE*/
			Cnf.var[Serial].value=FALSE;  /*CNF变量赋值为真(1)*/
			/*update Cnf when #Serial is FALSE*/
			Reverse(&Cnf,Serial);  /*将CNF从失败信号中恢复出来*/
			if (Cnf.result==TRUE) return TRUE;  /*如果CNF的一部分的结果取为真,返回真*/
			if (Cnf.result==FALSE) 	return FALSE;  /*如果CNF的一部分的结果取为假,返回假*/
			if (DavisPutnam(Cnf)==TRUE) return TRUE;  /*如果CNF的一部分测试子句后的结果取为真,返回真*/
 			return FALSE;  /*其它情况,返回假(0)*/
		 }
}


/******************************************/ 
/*该函数检查Cnf是否有单元字句
输入：Cnf一个CNF结构，Serial变量的编号
输出：HasUnitClause取值为TRUE或FALSE*/ 
/******************************************/ 
int HasUnitClause(struct CNF  Cnf,int *Serial)  /*测试CNF是否含有单子句(参数:CNF式,指向变量的指针)*/
{
	int i,j,k;
	for (i=0;i<NCLAUSES;i++)  /*在给定的子句数目范围(350)内*/
		if (Cnf.clause[i].num_unassigned==1)  /*子句中未赋值数为1*/
		{
			for(j=0;j<LENGTH_CLAUSE;j++)  /*在给定的子句长度范围(3)内*/
			{
				k=Cnf.clause[i].lit[j].value;  /*k暂时存放子句中相应的文字的值*/
				if (Cnf.var[Abs(k)].value==NOVALUE)  /*CNF中相应变量没有值*/
				{
					*Serial=k;  /*相应变量赋值*/
					return TRUE;  /*返回真(1)*/
				}
			}
		}
	*Serial=0;  /*变量指针格式化*/
	return FALSE;  /*其它情况,返回假(0)*/
}


/******************************************/ 
/*该函数检查Cnf是否有纯文字
输入：Cnf一个CNF结构，Serial变量的编号
输出：HasPureLiteral取值TRUE或FALSE*/ 
/******************************************/ 
int HasPureLiteral(struct CNF  Cnf,int* Serial)  /*测试CNF是否含有纯文字(参数:CNF式,指向变量的指针)*/
{
	int i,j,k,in_flag=FALSE;  /*标记初始设为假*/
	for(i=0;i<NVARS;i++)  /*在给定的变量数目范围(100)内*/
		if (Cnf.var[i].value==NOVALUE)  /*CNF中相应变量没有值*/
		{
			for(j=0;j<NCLAUSES;j++)  /*在给定的子句数目范围(350)内*/
			{
				for(k=0;k<LENGTH_CLAUSE;k++)  /*在给定的子句长度范围(3)内*/
					if (Cnf.clause[j].lit[k].value==-i) break;  /*子句中相应的文字的值为对应的负值,退出*/
					else if (Cnf.clause[j].lit[k].value==i) in_flag=TRUE;  /*子句中相应的文字的值为对应的正值,标记设为真*/
				if( k < LENGTH_CLAUSE ) break;  /*子句长度小于其最大值(3),退出循环*/
			}
			if(in_flag && j==NCLAUSES)  /*标记为真且在给定的子句数目范围(350)内*/
			{
				*Serial=i;  /*相应变量赋值*/
				return TRUE;  /*返回真(1)*/
			}
		}
	*Serial=0;  /*变量指针格式化*/
	return FALSE;  /*其它情况,返回假(0)*/
}


/******************************************/ 
/*该函数根据变量NO.Serial设置Cnf的值
输入：Cnf一个CNF结构，Serial变量编号*/ 
/******************************************/ 
void UpdateCNF(struct CNF  * Cnf,int Serial)  /*为变量赋新值时更新CNF(参数:CNF式,整型变量)*/
{
	int i,j;
	Cnf->num_uncertain--;  /*CNF式中未赋值数减1*/
	for(i=0;i<NCLAUSES;i++)  /*在给定的子句数目范围(350)内*/
		for(j=0;j<LENGTH_CLAUSE;j++)  /*在给定的子句长度范围(3)内*/
			if (Cnf->clause[i].lit[j].value==Serial || Cnf->clause[i].lit[j].value==-Serial)  /*子句中相应的文字的值为对应的变量正值或负值*/
			{
				Cnf->clause[i].num_unassigned--;  /*子句中未赋值数减1*/			
			if(Cnf->clause[i].lit[j].value==Abs(Serial)&&Cnf->var[Abs(Serial)].value==TRUE  /*子句中相应的文字的值为对应的变量正值且CNF中相应变量为真(1)*/
		|| Cnf->clause[i].lit[j].value==-Abs(Serial) && Cnf->var[Abs(Serial)].value==FALSE)  /*子句中相应的文字的值为对应的变量负值且CNF中相应变量为假(0)*/
					{
						/*printf("this is result %d\n",Cnf->result);*/
						Cnf->clause[i].state_of_satisfied=TRUE;  /*子句设为满足条件*/
					}
					else if (Cnf->clause[i].num_unassigned==0)  /*子句中未赋值文字为0*/
							if(Cnf->clause[i].state_of_satisfied==UNCERTAIN)  /*子句是否满足条件不确定*/
							{
								Cnf->clause[i].state_of_satisfied=FALSE;  /*子句设为不满足条件*/
								Cnf->result=FALSE;  /*CNF结果设为假（0）*/
							}
			}
	for(i=0;i<NCLAUSES;i++)  /*在给定的子句数目范围(350)内*/
		if(Cnf->clause[i].state_of_satisfied==FALSE)  /*子句不满足条件*/
		{
			Cnf->result=FALSE;  /*CNF结果设为假（0）*/
			return ;
		}
		else if(Cnf->clause[i].state_of_satisfied==UNCERTAIN)  /*子句是否满足条件不确定 */
		{
			Cnf->result=UNCERTAIN;  /*子句是否满足条件设为不确定*/
			return;
		}
	Cnf->result=TRUE;  /*CNF结果设为真（1）*/
}


/******************************************/ 
/*当我们将一个错误的值赋于一个变量时，我们必须将Cnf转置到可能争取的状态
输入：Cnf一个指向CNF结构的指针，Serial变量的编号
输出：Cnf*/ 
/******************************************/ 
void Reverse(struct CNF * Cnf,int Serial)  /*将CNF从失败信号中恢复出来(参数:CNF式,整型变量)*/
{
	int i,j,k,temp;
	for(i=0;i<NCLAUSES;i++)  /*在给定的子句数目范围(350)内*/
		for(j=0;j<LENGTH_CLAUSE;j++)  /*在给定的子句长度范围(3)内*/
				if (Cnf->clause[i].lit[j].value==Serial)  /*子句中相应的文字的值为对应的变量正值*/
			/*#Serial is in this clause and it is positive*/
				{
			/*judge if this clause can be satisfied*/
					for(k=0;k<LENGTH_CLAUSE;k++)  /*在给定的子句长度范围(3)内*/
						if((temp=Cnf->clause[i].lit[k].value)==Abs(temp) 
							&& Cnf->var[Abs(temp)].value==TRUE  /*子句中相应的文字的值为对应的变量正值且CNF中相应变量为真(1)*/
						||temp==-Abs(temp) && Cnf->var[Abs(temp)].value==FALSE)  /*子句中相应的文字的值为对应的变量负值且CNF中相应变量为假(0)*/
								break;
					if(k==LENGTH_CLAUSE)  /*到达子句末尾*/
	/*if k=LENGTH_CLAUSE then this clause can't be satisfied,so the CNF can't be satisfied*/
					{
						Cnf->clause[i].state_of_satisfied=FALSE;  /*子句是否满足条件设为假（0）*/
						Cnf->result=FALSE;  /*CNF结果设为假（0）*/
					}
				}
				else if (Cnf->clause[i].lit[j].value==-Serial)/*if*/  /*子句中相应的文字的值为对应的变量正值*/
					Cnf->clause[i].state_of_satisfied=TRUE;  /*子句是否满足条件设为真（1）*/
	for(i=0;i<NCLAUSES;i++)  /*在给定的子句数目范围(350)内*/
		if(Cnf->clause[i].state_of_satisfied==FALSE)  /*子句是否满足条件为假（0）*/
		{
			Cnf->result=FALSE;  /*CNF结果设为假（0）*/
			return ;
		}
		else if(Cnf->clause[i].state_of_satisfied==UNCERTAIN)  /*子句是否满足条件为不确定*/
		{
			Cnf->result=UNCERTAIN;  /*CNF结果设为不确定--以后确定*/
			return;
		}
	Cnf->result=TRUE;  /*CNF结果设为真（1）*/
}


/*****************************************/ 
/*该函数返回一个整数的绝对值*/ 
/*****************************************/ 
int Abs(int i)  /*返回变量的绝对值（参数：整型变量）*/
{
	if(i>0) return i;  /*正数情况*/
	return -i;  /*负数情况*/
}


/****************************/ 
/*该函数返回一个整数x的y次方*/ 
/****************************/ 
int Pow(int x,int y)
{
	int i,res=1;
	for(i=0;i<y;i++) res=res*x;
	return res;
}


/******************************************/ 
/*因为CNF结构不是一个MPI类型，当我们要广播它的时候我们必须将其压成另一种形式
输入：Cnf一个CNF结构
输出：一个整数缓冲*/ 
/******************************************/ 
void PackCNF(struct CNF Cnf,int buf[])  /*将CNF打包到整型缓冲--便于MPI广播（参数：CNF式，整型缓存）*/
{
	int i=0,j,k;
	buf[i++]=Cnf.num_uncertain;  /*整型缓冲第一个元素为CNF式不确定变量数*/
	buf[i++]=Cnf.result;  /*整型缓冲第二个元素为CNF式结果*/
	for(j=0;j<NVARS;j++)  /*在给定的变量数目范围(100)内*/
		buf[i++]=Cnf.var[j].value;  /*给整型缓冲其它元素赋CNF式相应变量的值*/
	for(j=0;j<NCLAUSES;j++)  /*在给定的子句数目范围(350)内*/
	{
		buf[i++]=Cnf.clause[j].num_unassigned;  /*整型缓冲接下来第一个元素为子句不确定文字数*/
		buf[i++]=Cnf.clause[j].state_of_satisfied;  /*整型缓冲接下来第二个元素为子句状态*/
		for(k=0;k<LENGTH_CLAUSE;k++)  /*在给定的子句长度范围(3)内*/
			buf[i++]=Cnf.clause[j].lit[k].value;  /*给整型缓冲其它元素赋子句相应文字的值*/
	}
}


/******************************************/ 
/*该函数是PackCNF函数的反函数
输入：buf一个整数缓冲
输出：pCnf一个指向CNF结构的指针*/ 
/******************************************/ 
void UnpackCNF(struct CNF * pCnf,int buf[])  /*将缓冲中的CNF解包（参数：CNF式，整型缓存）*/
{
	int i=0,j,k;
	pCnf->num_uncertain=buf[i++];  /*CNF式不确定变量数为整型缓冲第一个元素*/
	pCnf->result=buf[i++];  /*CNF式结果为整型缓冲第二个元素*/
	for (j=0;j<NVARS;j++)  /*在给定的变量数目范围(100)内*/
		pCnf->var[j].value=buf[i++];  /*给CNF式相应变量赋整型缓冲其它元素的值*/
	for (j=0;j<NCLAUSES;j++)  /*在给定的子句数目范围(350)内*/
	{
		pCnf->clause[j].num_unassigned=buf[i++];  /*子句不确定文字数为整型缓冲接下来第一个元素*/
		pCnf->clause[j].state_of_satisfied=buf[i++];  /*子句状态为整型缓冲接下来第二个元素*/
		for(k=0;k<LENGTH_CLAUSE;k++)  /*在给定的子句长度范围(3)内*/
			pCnf->clause[j].lit[k].value=buf[i++];  /*给子句相应文字赋整型缓冲其它元素的值*/
	}
}


/******************************************/ 
/*该函数将任务数转换为变量的值
输入：x任务
输出：pCnf一个指向CNF结构的指针*/ 
/******************************************/ 
void SetValue(struct CNF * pCnf,unsigned x)  /*在任务中为变量设值并更新CNF（参数：指向CNF式的指针，无符号整型变量--任务数）*/
{
	int i;
	for (i=0;i<DEPTH;i++)  /*在计算深度范围（5）内*/
	{
		pCnf->var[i].value=GetBits(x,i);  /*CNF式相应变量赋值为任务数*/
		UpdateCNF(pCnf,i);  /*为变量赋新值时更新CNF*/
		if(pCnf->result==TRUE)  /*CNF式结果为真*/
			return;
		else if (pCnf->result==FALSE)  /*CNF式结果为假*/
			return;
	}
}


/******************************************/ 
/*从x右边取出NO.n位
输入：x取出位的源
输出：GetBits取值为０或１*/ 
/******************************************/ 
unsigned GetBits(unsigned x,int n)  /*从变量相应位上取值（参数：无符号整型变量，整型变量--取值位置）*/
{
	return (x>>n) & ~(~0 << 1);  /*从变量相应位上取值*/
}


/******************************************/ 
/*该函数由从进程执行执行，计算该任务的结果并将其送回根进程
输入：Cnf一个CNF结构，ｉ任务*/
/******************************************/ 
void SlaveWork(struct CNF Cnf,unsigned i,int BufVar[])  /*从进程的工作--从0进程接受任务,计算,将结果返回0进程*/
{
		int BufResult;
		SetValue(&Cnf,i); /*,BufVar);*/  /*在任务中为变量设值并更新CNF*/
		if( DavisPutnam(Cnf)==TRUE )  /*子句为SAT*/
		{
			BufResult=TRUE;  /*缓存结果为真（1）*/
			MPI_Send(&BufResult,1,MPI_INT,0,0,MPI_COMM_WORLD);  /*MPI--发送缓存结果到0进程*/
		}
		else  /*子句不为SAT*/
		{
			BufResult=FALSE;  /*缓存结果为假（0）*/
			MPI_Send(&BufResult,1,MPI_INT,0,0,MPI_COMM_WORLD);  /*MPI--发送缓存结果到0进程*/
		}
}


/******************************************/
/*该函数从cnf中选取需要的变量
输入：Cnf一个CNF结构
输出：BufVar变量缓冲*/
/******************************************/
void PickVars(struct CNF Cnf,int BufVar[])  /*从CNF中选取相应变量（参数：CNF式，整型缓存）*/
{
	int i,j,k=0,l=0,m;
	for(i=0;i<DEPTH;i++)  /*在计算深度范围（5）内*/
		if(HasUnitClause(Cnf,&j)==TRUE && NotInBuf(BufVar,Abs(j))) /*CNF含有单子句且没有变量在缓冲中*/
		{
			/*printf("\nsingle!");*/
			BufVar[i]=Abs(j);  /*整型缓存中相应的值为单个子句的值*/
		}
		else if (HasPureLiteral(Cnf,&j)==TRUE && NotInBuf(BufVar,Abs(j)))  /*CNF含有纯文字且没有变量在缓冲中*/
		{	
			/*printf("\npure!");*/
			BufVar[i]=Abs(j);  /*整型缓存中相应的值为单个子句的值*/
		}
		else 
		{
			m=Abs(Cnf.clause[k].lit[l].value);  /*临时变量赋值为子句中相应文字的值*/
			while(!NotInBuf(BufVar,m))  /*当临时变量不在整型缓存中时循环*/
			{
				if (l>LENGTH_CLAUSE)  /*超出子句长度范围（3）*/
				{
					l=0;
					k++;
				}
				else l++;  /*子句中文字序号增加1*/
				m=Abs(Cnf.clause[k].lit[l].value);  /*临时变量赋值为子句中相应文字的值*/
				/*printf("\nm=%d",m);*/
			}
			BufVar[i]=m;  /*整型缓存中相应元素的值设为临时变量的值*/
		}
}


/******************************************/
/*该函数检查变量I是否在这个缓冲中
输入：Buf包含变量的缓冲，ｉ整数
输出：NotInBuf取值０或１*/
/******************************************/
int NotInBuf(int Buf[],int i)  /*测试变量是否在缓冲中（参数：整型缓存，整型变量）*/
{
	int j;
	for(j=0;j<DEPTH;j++)  /*在计算深度范围（5）内*/
		if (i==Buf[j]) return 0;  /*变量在缓冲中，返回0--判断时为假*/
	return 1;  /*变量不在缓冲中，返回1--判断时为真*/
} 
