#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>

typedef struct{
	int pedlen;
	int pednum;
	int psuffixlen;
}pntype;

/*
void gen_string(int stringlen,char *string,int seed)
输入：待生成的文本串长度: int stringlen;
       		
输出：待匹配的文本串: char *string;

功能：生成待匹配的文本串

void gen_pattern(char *text,int *patlen,char *pattern,int seed)

输入：待匹配文本 char *text;

      待生成的匹配串长度: int *patlen;

输出：生成的匹配串: char *pattern;

功能：从文本串中选择连续的一段作为待匹配的模式串

int Min(int x,int y,int z)
输入：三个整型数 int x,int y,int z;

输出：三个整数中选择最小的一个 int Min();

功能：从三个整数中选择最小的一个

void ModifiedDP(char *T,char *P,int textlen,int patternlen,int k,int *D)
输入：待匹配的文本串: char *T;
      匹配串:char *P;
	  文本串长:int textlen;
	  匹配串长:int patternlen

输出：模式串P的前缀子串p1…pi与文本串T的任意
      以tj结尾的子串之间的最小误差个数（即最小编辑距离）int *D;

功能：允许换位操作的动态规划近似串匹配算法

void Next(char *P,int patlen,int *nextval,pntype *pped)
输入：匹配串 char *P;
      匹配串长: int patlen;

输出：int *nextval;

功能：对输入串进行周期分析并生成next以及nextval值

void Search(char *T,char *P,int textlen,int patlen,int *nextval,
pntype *pped,int k,int *D,int SubpatternStart,char *pattern,int patternlen,int *VerifyCount)
输入：原文本串: char *T;
      匹配串: char *P;
	  原文本串长: int textlen;
	  匹配串长: int patlen;
	  近似匹配允许的最大误差数: int k ;
	  最小编辑距离: int *D;
输出：精确匹配点
功能：允许换位操作的过滤近似串匹配算法

int SubPatternLength(int patlen,int k,int num)
输入： 匹配串长: int patlen;
       近似匹配允许的最大误差数: int k ;
输出：子模式串长
功能：将模式串划分为2k+1个子模式串

*/
/*生成待匹配的文本串*/
void gen_string(int stringlen,char *string,int seed)
{
	int i,num;

 	srand(seed*100);
	for (i=0;i<stringlen;i++) {
		num=rand()%26;
		string[i]='a'+num;
	}
	string[stringlen]='\0';
}

/*从文本串中选择连续的一段作为待匹配的模式串*/
void gen_pattern(char *text,int *patlen,char **pattern,int seed) 
{
 	int start;

 	if ((*pattern=(char *)malloc((*patlen+1)*sizeof(char)))==NULL) {
		printf("Error alloc memory.\n");
		exit(1);
 	}
 	srand(seed*100);
 	start=rand()%strlen(text)/3;
 	strncpy(*pattern,text+start,*patlen);
 	*(*pattern+*patlen)='\0';
}

/*从三个整数中选择最小的一个*/
int Min(int x,int y,int z)
{
	int minimum;

 	minimum=(x<y?x:y);
 	minimum=(minimum<z?minimum:z);
 	return minimum;
}

/*允许换位操作的动态规划近似串匹配算法*/
void ModifiedDP(char *T,char *P,int textlen,int patternlen,int k,int *D)
{
 	int i,j;
 	int *D1,*D2;

 	if (((D1=(int *)malloc((textlen+1)*sizeof(int)))==NULL)||
		((D2=(int *)malloc((textlen+1)*sizeof(int)))==NULL)) {
		printf("Malloc error.\n");
		exit(1);
 	}
 	for (j=0;j<=textlen;j++) {
		D1[j]=0;
	 	D2[j]=0;
 	}	
 	for (i=1;i<=patternlen;i++) {
		D[0]=i;
     	for (j=1;j<=textlen;j++) {
	    	if (P[i-1]==T[j-1])
		    	D[j]=Min(D2[j]+1,D[j-1]+1,D2[j-1]);
		 	else if (i-2>=0 && j-2>=0 && P[i-1]==T[j-2] && P[i-2]==T[j-1])
		    	D[j]=Min(D2[j]+1,D[j-1]+1,D1[j-2]+1);
		   	else
            	D[j]=Min(D2[j]+1,D[j-1]+1,D2[j-1]+1);
     	}
	 	for (j=0;j<=textlen;j++) {
         	D1[j]=D2[j];
		 	D2[j]=D[j];
     	}
 	}
 	free(D1);
 	free(D2);
}

/*对输入串进行周期分析并生成next以及nextval值*/
void Next(char *P,int patlen,int *nextval,pntype *pped)
{
 	int i,j;
 	int *next;

 	if ((next=(int *)malloc((patlen+1)*sizeof(int)))==NULL) {
		printf("No enough memory.\n");
		exit(1);
 	}
 	next[0]=nextval[0]=-1;
 	j=1;
 	while (j<=patlen) {
	   	i=next[j-1];
	   	while (i!=-1 && P[i]!=P[j-1])
			i=next[i];
	   	next[j]=i+1;
	   	if (j!=patlen) {
		  	if (P[j]!=P[i+1])
				nextval[j]=i+1;
		  	else
			 	nextval[j]=nextval[i+1];
       	}
       	j++;
 	}
 	pped->pedlen=patlen-next[patlen];
 	pped->pednum=(int)(patlen/pped->pedlen);
 	pped->psuffixlen=patlen%pped->pedlen;
 	free(next);
}

/*允许换位操作的过滤近似串匹配算法*/
void Search(char *T,char *P,int textlen,int patlen,int *nextval,pntype *pped,int k,int *D,int SubpatternStart,char *pattern,int patternlen,int *VerifyCount)
{
 	int i,j,p;
 	int VerifyStart,VerifyEnd,VerifyLength;
 	int *TempD;

 	if ((TempD=(int *)malloc((patternlen+2*k+1)*sizeof(int)))==NULL) {
		printf("Malloc error.\n");
		exit(1);
 	}
 	i=0;
 	j=0;
 	while (i<textlen) {
	   	while ((j!=-1) && (P[j]!=T[i]))
			j=nextval[j];
	   	if (j==(patlen-1)) {
			(*VerifyCount)++;
			VerifyStart=i-patlen-SubpatternStart-k+1;
          	VerifyEnd=i-patlen-SubpatternStart+patternlen+k;
		  	VerifyStart=(VerifyStart>0?VerifyStart:0);
          	VerifyEnd=(VerifyEnd<textlen-1?VerifyEnd:textlen-1);
		  	VerifyLength=VerifyEnd-VerifyStart+1;
          	ModifiedDP(T+VerifyStart,pattern,VerifyLength,patternlen,k,TempD);
		  	for (p=1;p<=VerifyLength;p++)
				D[VerifyStart+p]=(D[VerifyStart+p]<TempD[p]?D[VerifyStart+p]:TempD[p]);
		   
		  	if (pped->pednum+pped->psuffixlen==1)
				j=-1;
		  	else
				j=patlen-1-pped->pedlen;
       	}
	   	j++;
	   	i++;
 	}
 	free(TempD);
}


/*将模式串划分为2k+1个子模式串*/
int SubPatternLength(int patlen,int k,int num)
{
 	int r;

 	r=patlen%(2*k+1);
 	if (r==0) 
 		return(patlen/(2*k+1));
 	else if (num<=r) 
 		return(patlen/(2*k+1)+1);
    else 
    	return(patlen/(2*k+1));
}

int main(int argc,char *argv[])
{
 	char *T,*P;
 	int alltextlen,textlen,patternlen,k,subpatlen,startpoint;
 	int *nextval,*D;
 	pntype pped;
 	int i,l,myrank,groupsize;
 	int VerifyCount=0;
 	int TotalMatchCount=0,MatchCount=0;
 	MPI_Status status;
	char ch[1];

 	MPI_Init(&argc,&argv);
 	MPI_Comm_size(MPI_COMM_WORLD,&groupsize);
 	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
 	
 	alltextlen=atoi(argv[1]);
 	patternlen=atoi(argv[2]);
 	k=atoi(argv[3]);
 	textlen=alltextlen/groupsize;

 	if (myrank==0) {
    	if (((T=(char *)malloc((textlen+patternlen+k)*sizeof(char)))==NULL)||
	  		((D=(int *)malloc((textlen+patternlen+k)*sizeof(int)))==NULL)) {
			printf("No enough memory.\n");
	   		exit(1);
		}

		for (i=0;i<=textlen;i++) D[i]=10000;
		
		/*生成本地文本串与模式串*/
    	gen_string(textlen,T+patternlen+k-1,myrank);
    	gen_pattern(T+patternlen+k-1,&patternlen,&P,myrank);

    	if ((nextval=(int *)malloc((patternlen/(2*k+1)+1)*sizeof(int)))==NULL) {
			printf("No enough memory.\n");
			exit(1);
		}
 	}

	MPI_Barrier(MPI_COMM_WORLD);

 	if (groupsize>1) {
		if (myrank!=0) {
			if (myrank==groupsize-1)
				textlen=alltextlen-(groupsize-1)*(alltextlen/groupsize);

			if (((T=(char *)malloc((textlen+patternlen+k)*sizeof(char)))==NULL)||
		  		((D=(int *)malloc((textlen+patternlen+k)*sizeof(int)))==NULL)||
          		((nextval=(int *)malloc((patternlen/(2*k+1)+1)*sizeof(int)))==NULL)||
          		((P=(char *)malloc((patternlen+1)*sizeof(char)))==NULL)) {
		  		printf("No enough memory.\n");
	      		exit(1);
	   		}

       		for (i=0;i<=textlen+patternlen+k-1;i++) D[i]=10000;
       		/*生成本地文本串*/
       		gen_string(textlen,T+patternlen+k-1,myrank);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		
		/*播送模式串给所有处理器，对应算法14.13步骤（2）*/
    	MPI_Bcast(P,patternlen+1,MPI_CHAR,0,MPI_COMM_WORLD);

		printf("on node %d the text is %s \n",myrank,T+patternlen+k-1);
		printf("on node %d the pattern is %s \n",myrank,P);
		
		/*播送进程间相关数据给相邻进程，对应算法14.13步骤（3）和（4）*/
		if(myrank==0) {
			MPI_Send(&T[textlen],patternlen+k-1,MPI_CHAR,myrank+1,myrank,MPI_COMM_WORLD);
		}
		else if(myrank==groupsize-1) {
	 		MPI_Recv(&T[0],patternlen+k-1,MPI_CHAR,myrank-1,myrank-1,MPI_COMM_WORLD,&status);
		}
		else {
	 		MPI_Recv(T,patternlen+k-1,MPI_CHAR,myrank-1,myrank-1,MPI_COMM_WORLD,&status);
			MPI_Send(T+textlen,patternlen+k-1,MPI_CHAR,myrank+1,myrank,MPI_COMM_WORLD);
		}
	}

 	MPI_Barrier(MPI_COMM_WORLD);	

 	/*各进程分别对本地数据执行允许换位操作的过滤近似串匹配算法，对应算法14.13步骤（5）*/
 	startpoint=0;
	if(myrank==0)
    	for (l=1;l<=2*k+1;l++) {
	   		subpatlen=SubPatternLength(patternlen,k,l);
	   		if (subpatlen==0) continue; 
	   		Next(P+startpoint,subpatlen,nextval,&pped);
	   		Search(T+patternlen+k-1,P+startpoint,textlen,subpatlen,nextval,&pped,k,D,startpoint,P,patternlen,&VerifyCount);
		   	startpoint=startpoint+subpatlen;
		}
	else
    	for (l=1;l<=2*k+1;l++) {
	   		subpatlen=SubPatternLength(patternlen,k,l);
	   		if (subpatlen==0) continue; 
	   		Next(P+startpoint,subpatlen,nextval,&pped);
	   		Search(T,P+startpoint,textlen+patternlen+k-1,subpatlen,nextval,&pped,k,D,startpoint,P,patternlen,&VerifyCount);
		   	startpoint=startpoint+subpatlen;
		}

 	MPI_Barrier(MPI_COMM_WORLD);
 
	if(myrank==0)
    	for (i=1;i<=textlen;i++) {
	    	if (D[i]>=0 && D[i]<=k)
         		MatchCount++;
		}
	else 
    	for (i=patternlen+k;i<=textlen+patternlen+k-1;i++) {
	    	if (D[i]>=0 && D[i]<=k)
         		MatchCount++;
		}
	
 	MPI_Barrier(MPI_COMM_WORLD);
	printf("Total %d matched on node %d \n",MatchCount,myrank);

 	free(T);
 	free(P);
 	free(D);
 	free(nextval);
 	MPI_Finalize();
 	
 	return;
}

