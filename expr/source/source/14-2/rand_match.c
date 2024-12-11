#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define PiM 21

/*初始化Fp矩阵*/
int f[2][2][2]={{{1,0},{1,1}},{{1,1},{0,1}}};
int g[2][2][2];
int pdata[]={2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,67,71,79,83};

/*生成字符串*/
void gen_string(int stringlen,char *String,int seed)
{
	int i,num;

	srand(seed*100);
	for(i=0;i<stringlen;i++){
    	num=rand()%2;
    	String[i]='0'+num;
  	}
  	String[stringlen]='\0';
}

/*从素数存放数组中随机选取一个素数*/
int drawp(int seed){
	srand(seed*100);	
	return pdata[rand()%PiM];
}

/*该算法中第一个参数n是正文串的长度，第二个参数m是模式串的长度。(约束0<m,n<=128)*/
main(int argc,char *argv[]){
  	int groupsize,myrank,n,m,p,i,j,h,tmp;
	int textlen;
  	char *Text,*Pattern;
  	int *MATCH;
  	int B[128][2][2],C[128][2][2],D[128][2][2],L[128][2][2];
  	int fp_pattern[2][2];
	MPI_Status status;
  
  
  	MPI_Init(&argc,&argv);
  	MPI_Comm_size(MPI_COMM_WORLD,&groupsize);
  	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  
  	n=atoi(argv[1]);
  	m=atoi(argv[2]);

	textlen=n/groupsize;

	if(myrank==groupsize-1)
		textlen=n-textlen*(groupsize-1);

  	if((Text=(char *)malloc(textlen*sizeof(char)+1))==NULL){
    	printf("no enough memory\n");
    	exit(1);
  	}

  	if((Pattern=(char *)malloc(m*sizeof(char)+1))==NULL){
    	printf("no enough memory\n");
    	exit(1);
  	}
  
  	if((MATCH=(int *)malloc(textlen*sizeof(int)+1))==NULL){
    	printf("no enough memory\n");
    	exit(1);
  	}

  	/*初始化Fp的逆矩阵Gp*/
  	g[0][0][0]=1;g[0][0][1]=0;g[0][1][0]=p-1;g[0][1][1]=1;
  	g[1][0][0]=1;g[1][0][1]=p-1;g[1][1][0]=0;g[1][1][1]=1;
  
  	/*产生正文串和模式串*/
  	gen_string(textlen,Text,m*n*myrank);

	/*随机选择一个素数，对应算法14.9步骤（2）*/
	if(myrank==0) {
  		p=drawp(n/m);
  		gen_string(m,Pattern,p*m*n);
	}
  
  	/*播送选中的素数与m给其余每个节点*/
	MPI_Bcast(&p,1,MPI_INT,0,MPI_COMM_WORLD);		
  	MPI_Bcast(Pattern,m,MPI_CHAR,0,MPI_COMM_WORLD);		
  
  	printf("one node %d n=%d,m=%d,p=%d\n",myrank,n,m,p);
  	printf("one node %d Text=%s\n",myrank,Text);
  	printf("one node %d Pattern=%s\n",myrank,Pattern);

  	MPI_Barrier(MPI_COMM_WORLD);
    
	/*计算模式串的指纹函数值，对应算法14.9步骤（3）*/
	if(myrank==0) {
  		for(i=1;i<=m;i++) {
    		B[i][0][0]=f[Pattern[i-1]-'0'][0][0];
    		B[i][0][1]=f[Pattern[i-1]-'0'][0][1];
    		B[i][1][0]=f[Pattern[i-1]-'0'][1][0];
    		B[i][1][1]=f[Pattern[i-1]-'0'][1][1];
  		}

      	B[0][0][0]=(B[1][0][0]*B[2][0][0]+B[1][0][1]*B[2][1][0])%p;
      	B[0][0][1]=(B[1][0][0]*B[2][0][1]+B[1][0][1]*B[2][1][1])%p;
      	B[0][1][0]=(B[1][1][0]*B[2][0][0]+B[1][1][1]*B[2][1][0])%p;
      	B[0][1][1]=(B[1][1][0]*B[2][0][1]+B[1][1][1]*B[2][1][1])%p;

    	for(j=1;j<=m-2;j++){
      		B[j][0][0]=(B[j-1][0][0]*B[j+2][0][0]+B[j-1][0][1]*B[j+2][1][0])%p;
      		B[j][0][1]=(B[j-1][0][0]*B[j+2][0][1]+B[j-1][0][1]*B[j+2][1][1])%p;
      		B[j][1][0]=(B[j-1][1][0]*B[j+2][0][0]+B[j-1][1][1]*B[j+2][1][0])%p;
      		B[j][1][1]=(B[j-1][1][0]*B[j+2][0][1]+B[j-1][1][1]*B[j+2][1][1])%p;
    	}

     	fp_pattern[0][0]=B[m-2][0][0];
     	fp_pattern[0][1]=B[m-2][0][1];
     	fp_pattern[1][0]=B[m-2][1][0];
     	fp_pattern[1][1]=B[m-2][1][1];
  	}
    	
  	MPI_Bcast(fp_pattern,4,MPI_INT,0,MPI_COMM_WORLD);
  	MPI_Barrier(MPI_COMM_WORLD);
    
  	/*计算Ni，存放在数组C中*/
  	for(i=1;i<=textlen;i++) {
    	B[i][0][0]=f[Text[i-1]-'0'][0][0];
    	B[i][0][1]=f[Text[i-1]-'0'][0][1];
    	B[i][1][0]=f[Text[i-1]-'0'][1][0];
    	B[i][1][1]=f[Text[i-1]-'0'][1][1];
  	}

   	C[1][0][0]=B[1][0][0];
    C[1][0][1]=B[1][0][1];
   	C[1][1][0]=B[1][1][0];
   	C[1][1][1]=B[1][1][1];

	for(i=2;i<=textlen;i++) {
      	C[i][0][0]=(C[i-1][0][0]*B[i][0][0]+C[i-1][0][1]*B[i][1][0])%p;
      	C[i][0][1]=(C[i-1][0][0]*B[i][0][1]+C[i-1][0][1]*B[i][1][1])%p;
      	C[i][1][0]=(C[i-1][1][0]*B[i][0][0]+C[i-1][1][1]*B[i][1][0])%p;
      	C[i][1][1]=(C[i-1][1][0]*B[i][0][1]+C[i-1][1][1]*B[i][1][1])%p;
	}

	if(groupsize>1) {
   		if(myrank==0) {
			MPI_Send(&C[textlen],4,MPI_INT,myrank+1,myrank,MPI_COMM_WORLD);
		}
		else if(myrank==groupsize-1) {
			MPI_Recv(&C[0],4,MPI_INT,myrank-1,myrank-1,MPI_COMM_WORLD,&status);
		}
		else {
			MPI_Recv(&C[0],4,MPI_INT,myrank-1,myrank-1,MPI_COMM_WORLD,&status);
	
			i=textlen;

      		C[textlen+1][0][0]=(C[0][0][0]*C[i][0][0]+C[0][0][1]*C[i][1][0])%p;
      		C[textlen+1][0][1]=(C[0][0][0]*C[i][0][1]+C[0][0][1]*C[i][1][1])%p;
      		C[textlen+1][1][0]=(C[0][1][0]*C[i][0][0]+C[0][1][1]*C[i][1][0])%p;
      		C[textlen+1][1][1]=(C[0][1][0]*C[i][0][1]+C[0][1][1]*C[i][1][1])%p;

   			C[i][0][0]=C[textlen+1][0][0];
   			C[i][0][1]=C[textlen+1][0][1];
	   		C[i][1][0]=C[textlen+1][1][0];
   			C[i][1][1]=C[textlen+1][1][1];

			MPI_Send(&C[textlen],4,MPI_INT,myrank+1,myrank,MPI_COMM_WORLD); 
		}
	}

  	MPI_Barrier(MPI_COMM_WORLD);

	if(myrank!=0) {
		for(i=1;i<textlen;i++) {
      		C[textlen+1][0][0]=(C[0][0][0]*C[i][0][0]+C[0][0][1]*C[i][1][0])%p;
      		C[textlen+1][0][1]=(C[0][0][0]*C[i][0][1]+C[0][0][1]*C[i][1][1])%p;
      		C[textlen+1][1][0]=(C[0][1][0]*C[i][0][0]+C[0][1][1]*C[i][1][0])%p;
      		C[textlen+1][1][1]=(C[0][1][0]*C[i][0][1]+C[0][1][1]*C[i][1][1])%p;
   			C[i][0][0]=C[textlen+1][0][0];
   			C[i][0][1]=C[textlen+1][0][1];
   			C[i][1][0]=C[textlen+1][1][0];
   			C[i][1][1]=C[textlen+1][1][1];
		} 
	}

  	MPI_Barrier(MPI_COMM_WORLD);

	if(groupsize>1) {
   		if(myrank==0) {
			MPI_Recv(&C[textlen+1],4*(m-1),MPI_INT,myrank+1,myrank,MPI_COMM_WORLD,&status);
		}
		else if(myrank==groupsize-1) {
			MPI_Send(&C[1],4*(m-1),MPI_INT,myrank-1,myrank-1,MPI_COMM_WORLD);
		}
		else {
			MPI_Recv(&C[textlen+1],4*(m-1),MPI_INT,myrank+1,myrank,MPI_COMM_WORLD,&status);
			MPI_Send(&C[1],4*(m-1),MPI_INT,myrank-1,myrank-1,MPI_COMM_WORLD); 
		}
	}

  	/*计算Ri，存放在数组D中*/
  	for(i=1;i<=textlen;i++) {
    	B[i][0][0]=g[Text[i-1]-'0'][0][0];
    	B[i][0][1]=g[Text[i-1]-'0'][0][1];
    	B[i][1][0]=g[Text[i-1]-'0'][1][0];
    	B[i][1][1]=g[Text[i-1]-'0'][1][1];
  	}

   	D[1][0][0]=B[1][0][0];
    D[1][0][1]=B[1][0][1];
   	D[1][1][0]=B[1][1][0];
   	D[1][1][1]=B[1][1][1];

	for(i=2;i<=textlen;i++) {
      	D[i][0][0]=(B[i][0][0]*D[i-1][0][0]+B[i][0][1]*D[i-1][1][0])%p;
      	D[i][0][1]=(B[i][0][0]*D[i-1][0][1]+B[i][0][1]*D[i-1][1][1])%p;
      	D[i][1][0]=(B[i][1][0]*D[i-1][0][0]+B[i][1][1]*D[i-1][1][0])%p;
      	D[i][1][1]=(B[i][1][0]*D[i-1][0][1]+B[i][1][1]*D[i-1][1][1])%p;
	}

	if(groupsize>1) {
   		if(myrank==0) {
			MPI_Send(&D[textlen],4,MPI_INT,myrank+1,myrank,MPI_COMM_WORLD);
		}
		else if(myrank==groupsize-1) {
			MPI_Recv(&D[0],4,MPI_INT,myrank-1,myrank-1,MPI_COMM_WORLD,&status);
		}
		else {
			MPI_Recv(&D[0],4,MPI_INT,myrank-1,myrank-1,MPI_COMM_WORLD,&status);

			i=textlen;
   		   	D[textlen+1][0][0]=(D[i][0][0]*D[0][0][0]+D[i][0][1]*D[0][1][0])%p;
   		   	D[textlen+1][0][1]=(D[i][0][0]*D[0][0][1]+D[i][0][1]*D[0][1][1])%p;
   		   	D[textlen+1][1][0]=(D[i][1][0]*D[0][0][0]+D[i][1][1]*D[0][1][0])%p;
   		   	D[textlen+1][1][1]=(D[i][1][0]*D[0][0][1]+D[i][1][1]*D[0][1][1])%p;
			
   			D[i][0][0]=D[textlen+1][0][0];
   			D[i][0][1]=D[textlen+1][0][1];
   			D[i][1][0]=D[textlen+1][1][0];
   			D[i][1][1]=D[textlen+1][1][1];

			MPI_Send(&D[textlen],4,MPI_INT,myrank+1,myrank,MPI_COMM_WORLD); 
		}
	}

  	MPI_Barrier(MPI_COMM_WORLD);

	if(myrank==0) {
   		D[0][0][0]=1;
   		D[0][0][1]=0;
   		D[0][1][0]=0;
   		D[0][1][1]=1;
	}
		
	if(myrank!=0) {
		for(i=1;i<textlen;i++) {
      		D[textlen+1][0][0]=(D[i][0][0]*D[0][0][0]+D[i][0][1]*D[0][1][0])%p;
      		D[textlen+1][0][1]=(D[i][0][0]*D[0][0][1]+D[i][0][1]*D[0][1][1])%p;
      		D[textlen+1][1][0]=(D[i][1][0]*D[0][0][0]+D[i][1][1]*D[0][1][0])%p;
      		D[textlen+1][1][1]=(D[i][1][0]*D[0][0][1]+D[i][1][1]*D[0][1][1])%p;
			
   			D[i][0][0]=D[textlen+1][0][0];
   			D[i][0][1]=D[textlen+1][0][1];
   			D[i][1][0]=D[textlen+1][1][0];
   			D[i][1][1]=D[textlen+1][1][1];
		} 
	}
  
  	MPI_Barrier(MPI_COMM_WORLD);

 	for(i=1;i<=textlen;i++)
  		MATCH[i]=0;

	/*各个处理器分别计算所分配的文本串的Li，并判别是否存在匹配位置，对应算法14.9的步骤（4）*/  
	if(myrank==groupsize-1) {
  		for(i=1;i<=textlen-m+1;i++){
    		L[i][0][0]=(D[i-1][0][0]*C[i+m-1][0][0]+D[i-1][0][1]*C[i+m-1][1][0])%p;      
    		L[i][0][1]=(D[i-1][0][0]*C[i+m-1][0][1]+D[i-1][0][1]*C[i+m-1][1][1])%p;      
    		L[i][1][0]=(D[i-1][1][0]*C[i+m-1][0][0]+D[i-1][1][1]*C[i+m-1][1][0])%p;      
    		L[i][1][1]=(D[i-1][1][0]*C[i+m-1][0][1]+D[i-1][1][1]*C[i+m-1][1][1])%p;      
    		if(((L[i][0][0]+p)%p==fp_pattern[0][0]) 
				&& ((L[i][0][1]+p)%p==fp_pattern[0][1])
				&& ((L[i][1][0]+p)%p==fp_pattern[1][0]) 
				&& ((L[i][1][1]+p)%p==fp_pattern[1][1])) {
      			MATCH[i]=1;
				printf("on node %d match pos is %d\n",myrank,i);
			}
		}
	}
	else {
  		for(i=1;i<=textlen;i++){
    		L[i][0][0]=(D[i-1][0][0]*C[i+m-1][0][0]+D[i-1][0][1]*C[i+m-1][1][0])%p;      
    		L[i][0][1]=(D[i-1][0][0]*C[i+m-1][0][1]+D[i-1][0][1]*C[i+m-1][1][1])%p;      
    		L[i][1][0]=(D[i-1][1][0]*C[i+m-1][0][0]+D[i-1][1][1]*C[i+m-1][1][0])%p;      
    		L[i][1][1]=(D[i-1][1][0]*C[i+m-1][0][1]+D[i-1][1][1]*C[i+m-1][1][1])%p;      
    		if(((L[i][0][0]+p)%p==fp_pattern[0][0]) 
				&& ((L[i][0][1]+p)%p==fp_pattern[0][1])
				&& ((L[i][1][0]+p)%p==fp_pattern[1][0]) 
				&& ((L[i][1][1]+p)%p==fp_pattern[1][1])) {
      			MATCH[i]=1;
				printf("on node %d match pos is %d\n",myrank,i);
			}
		}
	}

	MPI_Finalize();  
}
