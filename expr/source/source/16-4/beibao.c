#include "math.h"
#include "mpi.h"

main ( argc, argv )
int argc;
char * argv[];
{ 
  MPI_Status status;
  
  int i, j, m, tmp, ww, group_size, my_rank, pnumber, c, wb,
      pb;  /* to store number of used slave processors */
  int group_size1; /* number of rank*/
  int p[100] , w[100], z[100], f[100][100];
  
  MPI_Init ( &argc , &argv );/* Initialze MPI. */
  MPI_Comm_size ( MPI_COMM_WORLD , &group_size1 );/* Get the number of rank. */
  MPI_Comm_rank ( MPI_COMM_WORLD , &my_rank );  /* get id of rank*/
  
  /*if the number of slave processor is less than 2,Abort!*/
  if ( group_size1 < 3 ) 
  {
      printf ( "not enough processor!\n" );
      MPI_Abort ( MPI_COMM_WORLD,99 );
  }
  pnumber=group_size1-1;
  
  /*主进程:输入数组,输出结果*/
  if(my_rank == 0)
  {
      printf ( "my_rank %d\n" , my_rank );
    
      printf ( "knapscack of capacity = %d\n", pnumber);
    
      printf ( "Enter number of values:\n");
      scanf( "%d",&m);
      for(i=1;i<=pnumber;i++)
      {
          MPI_Send(&m,1,MPI_INT,i,i,MPI_COMM_WORLD);
      }
    
      printf("please input p:\n");
      for(i = 1;i <= m ; i++ )
      {
	  scanf( "%d" , &p[i] );
      }
      for(i=1;i<=pnumber;i++)
      {
          MPI_Send(&p[1],m,MPI_INT,i,i,MPI_COMM_WORLD);
      }
	
      printf("please input w:\n");
      for(i =1; i<=m;i++)
      {
	  scanf("%d",&w[i]);
      }
      for(i=1;i<=pnumber;i++)
      {
          MPI_Send(&w[1],m,MPI_INT,i,i,MPI_COMM_WORLD);
      }
     
  }
  
  /* 根据主进程传来的数据,计算,得出结果*/
  else
  {
      printf(" my rank is %d\n",my_rank);
     
      MPI_Recv(&m,1,MPI_INT,0,my_rank,MPI_COMM_WORLD,&status);
      MPI_Recv(&p[1],m,MPI_INT,0,my_rank,MPI_COMM_WORLD,&status);
      MPI_Recv(&w[1],m,MPI_INT,0,my_rank,MPI_COMM_WORLD,&status);
	  
      c=pnumber;
      for(i=0;i<=c;i++)
      {
	  f[0][i]=0;
      }
      for(i=1;i<=m;i++)
      {
 	  f[i][0]=0;
      }
      for(i=1;i<=m;i++)
      {
	 if(i>1)
	 {
    	    for(j=1;j<=pnumber;j++)
	    {
               MPI_Recv(&f[i-1][j],1,MPI_INT,j,(i-1)*10+j,MPI_COMM_WORLD,&status);
       	   }
	  }
         if(my_rank<w[i]) 
	    f[i][my_rank]=f[i-1][my_rank];
         if(my_rank>=w[i] && my_rank<=c)
	 {
	    ww=w[i];
	    tmp=f[i-1][my_rank-ww]+p[i];
	    if(f[i-1][my_rank]>tmp)
	        f[i][my_rank]=f[i-1][my_rank];
            else f[i][my_rank]=tmp;
	  }
         
	 if(i<m)
	 {
	    for(j=1;j<=pnumber;j++)
	    {
                MPI_Send(&f[i][my_rank],1,MPI_INT,j,i*10+my_rank,MPI_COMM_WORLD);
            }
         }
       }
	  
	  /* tracing backward to find the solution*/
	  wb=c;
	  pb=f[m][c];
	  for(i=m;i>=1;i--)
	  {
		  if(f[i-1][wb]==pb)
		  {
			  z[i]=0;
		  }
		  else
		  {
			  z[i]=1;
			  pb=pb-p[i];
			  wb=wb-w[i];
		  }
	  }
	  if(my_rank==c)
	  {
          printf(" the result:\n");
	       for(i=1;i<=m;i++)
		   {
		      printf("z %d:%d\n",i,z[i]);
		   }
	  }
  }
  MPI_Finalize();
}
