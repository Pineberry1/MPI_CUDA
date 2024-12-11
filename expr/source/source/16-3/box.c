#include <stdio.h>
#include "math.h"
#include "mpi.h"

int forint ( float temp)
{
    int outint;
    if( temp >= (floor(temp) + 0.5))
        outint=floor(temp) + 1;
    else
        outint=floor(temp);
    return outint;
}


main ( argc, argv )
int argc;
char * argv[];
{
    MPI_Status status;
    int     i,
        group_size,
        my_rank,
        pnumber,                                  /* to store number of used slave processors */
        h,
        lp,
        lop,
        j,
        t;
    int     group_size1,                          /* number of rank*/
        tmpp;
    FILE    *fp;
    int     lp1,
        h1,
        tmp1,
        lop1,
        high,
        mid,
        flag,
        fflag;
    int     tmpd,
        d[100][100],
        height,
        m,                                        /* number of box*/
        td;
    int     f[100][100],
        e[100][100],
        r[100];                                   /*output array*/
    float   a[100],                               /* input array*/
        s[100],
        b[100][100],
        c[100][100],
        tmp;
    double starttime,endtime;
    MPI_Init ( &argc , &argv );                   /* Initialze MPI. */
    starttime = MPI_Wtime();
                                                  /* Get the number of rank. */
    MPI_Comm_size ( MPI_COMM_WORLD , &group_size1 );
    MPI_Comm_rank ( MPI_COMM_WORLD , &my_rank );  /* get id of rank*/

	/*if the number of slave processor is less than 2,Abort!*/
    if ( group_size1 < 3 )
    {
        printf ( "not enough processor!\n" );
        MPI_Abort ( MPI_COMM_WORLD,99 );
    }

	/* calculate the number of use slave processor and the size of input array*/
    tmpp = 1;
    for ( i = 1 ; ; i++ )
    {
        tmpp = tmpp * 2;
        if ( tmpp > group_size1 - 1 ) break ;
    }

    pnumber = (int)( tmpp / 2 );
    printf ( "processor number is %d\n",pnumber );
    group_size = pnumber+1;

	/*主进程:输入数组,输出结果*/
    if(my_rank == 0)
    {
        printf ( "my_rank %d\n" , my_rank );
        printf ( "Enter %d values(<=1):\n" , pnumber );
        fp=fopen("data1","rb");
        for(i = 1;i <= pnumber ; i++ )
        {
            fscanf(fp,"%f",&a[i]);

            if ( a[i] > 1)
            {
                printf ( "input %f wrong!\n" , a[i] );
                i = i-1 ;
            }
            b[0][i] = a[i];
        }
        printf ( "input a[%d]:\n" , pnumber );
        for ( i = 1 ; i <= pnumber ; i++ )
        {
            printf ( "%f\n" , a[i] );
        }

        for ( i = 1; i <= pnumber ; i++ )
        {
            MPI_Send(&b[0][i],1,MPI_FLOAT,i,i,MPI_COMM_WORLD);
        }

        tmp = log( pnumber ) / log(2);
        lp = forint( tmp);
        printf("lp= %d \n",lp);
        tmp = 1 ;
        for ( h = 1;h <= lp ; h++ )
        {
            tmp = tmp * 2 ;
            lop =(int)( pnumber / tmp  );
            for ( j = 1 ;j <= lop ; j++ )
            {
                MPI_Send(&b[h-1][2*j-1],2,MPI_FLOAT,j,j+h*10,MPI_COMM_WORLD);

            }
            for( j = 1 ;j <= lop ; j++ )
            {
                MPI_Recv(&b[h][j],1,MPI_FLOAT,j,j+h*10+100,MPI_COMM_WORLD,&status);

            }

        }
        for ( i = 1 ; i <= pnumber ; i++ )
        {
            MPI_Recv(&c[0][i],1,MPI_FLOAT,i,0,MPI_COMM_WORLD,&status);
        }
        for ( i = 1 ; i <= pnumber ; i++ )
        {
            MPI_Send(&c[0][1],pnumber,MPI_FLOAT,i,1,MPI_COMM_WORLD);
        }

        for ( i = 1 ; i <= pnumber ; i++ )
        {
            MPI_Recv(&f[0][i],1,MPI_INT,i,0,MPI_COMM_WORLD,&status);
        }

        for ( i = 1 ; i <= pnumber ; i++ )
        {
            MPI_Send(&f[0][1],pnumber,MPI_INT,i,2,MPI_COMM_WORLD);
        }
        MPI_Recv(&m,1,MPI_INT,1,0,MPI_COMM_WORLD,&status);

        for ( i = 1 ; i <= m ; i++ )
        {
            MPI_Recv(&d[0][i],1,MPI_INT,i,0,MPI_COMM_WORLD,&status);
            printf("d 0 %d:%d\n",i,d[0][i]);
        }

        for ( i = 1 ; i <= pnumber ; i++ )
        {
            MPI_Send(&d[0][1],m,MPI_INT,i,1,MPI_COMM_WORLD);
        }
    }

	/* 根据主进程传来的数据,计算,得出结果*/
    else
    {
        if ( my_rank <= pnumber )
        {
			/* 求前缀和 c[0][j]<-a[1]+a[2]+...+a[j] */
            printf ( "my_rank %d\n" , my_rank );
            MPI_Recv(&b[0][my_rank],1,MPI_FLOAT,0,my_rank,MPI_COMM_WORLD,&status);

            tmp1=1;
            i=0;
            for(;;)
            {
                if ( tmp1 < my_rank )
                {
                    i++;
                    tmp1 = tmp1 * 2 ;
                }
                else break ;
            }

            tmp = log( group_size-1 ) / log(2);
            lp1 = forint ( tmp );

            for(h1=1;h1<=lp1-i;h1++)
            {

                MPI_Recv(&b[h1-1][2*my_rank-1],2,MPI_FLOAT,0,my_rank+h1*10,MPI_COMM_WORLD,&status);

                b[h1][my_rank] = b[h1-1][2*my_rank-1]+b[h1-1][2*my_rank];
                MPI_Send(&b[h1][my_rank],1,MPI_FLOAT,0,my_rank+100+h1*10,MPI_COMM_WORLD);

            }

            for(h1=lp1-i;h1>=0;h1--)
            {
                if ( my_rank == 1 )
                {
                    c[h1][my_rank] = b[h1][my_rank] ;
                    if ( h1 > 0 )
                        MPI_Send(&c[h1][my_rank],1,MPI_FLOAT,my_rank*2,my_rank+h1*10,MPI_COMM_WORLD);
                    if ( h1 > 0 && 2*my_rank+1 < (group_size-1) )
                        MPI_Send(&c[h1][my_rank],1,MPI_FLOAT,my_rank*2+1,my_rank+h1*10,MPI_COMM_WORLD);
                }
                if ( my_rank % 2 == 0 )
                {
                    MPI_Recv(&c[h1+1][my_rank/2],1,MPI_FLOAT,my_rank/2,my_rank/2+(h1+1)*10,MPI_COMM_WORLD,&status);
                    c[h1][my_rank] = c[h1+1][my_rank/2];
                    if ( h1 > 0 )
                        MPI_Send(&c[h1][my_rank],1,MPI_FLOAT,my_rank*2,my_rank+h1*10,MPI_COMM_WORLD);
                    if ( h1 > 0 && my_rank*2+1 < (group_size-1) )
                        MPI_Send(&c[h1][my_rank],1,MPI_FLOAT,my_rank*2+1,my_rank+h1*10,MPI_COMM_WORLD);
                }
                if ( my_rank % 2 != 0 && my_rank > 1 )
                {
                    MPI_Recv(&c[h1+1][(my_rank-1)/2],1,MPI_FLOAT,(my_rank-1)/2,(my_rank-1)/2+(h1+1)*10,MPI_COMM_WORLD,&status);
                    c[h1][my_rank] = c[h1+1][(my_rank-1)/2] + b[h1][my_rank] ;
                    if ( h1 > 0 )
                        MPI_Send(&c[h1][my_rank],1,MPI_FLOAT,my_rank*2,my_rank+h1*10,MPI_COMM_WORLD);
                    if ( h1 > 0 && my_rank*2+1 < (group_size-1) )
                        MPI_Send(&c[h1][my_rank],1,MPI_FLOAT,my_rank*2+1,my_rank+h1*10,MPI_COMM_WORLD);
                }
            }

            MPI_Send(&c[0][my_rank],1,MPI_FLOAT,0,0,MPI_COMM_WORLD);

			/*借助c[0][j],计算f[0][j]<-max{k|a[j]+a[j+1]+...+a[k] <= 1}*/
            MPI_Recv(&c[0][1],group_size-1,MPI_FLOAT,0,1,MPI_COMM_WORLD,&status);
            c[0][0]=0;
            for ( i = 1 ; i <= group_size - my_rank ; i++ )
            {
                s[i] = c[0][my_rank+i-1] - c[0][my_rank-1] ;
            }
            if ( s[group_size-my_rank] <= 1 ) mid = group_size - my_rank ;
            else
            {
                for ( i = 1 ; i <= group_size - my_rank ; i++ )
                {
                    if ( s[i] > 1 )  break ;
                }
                mid = i - 1 ;
            }
            f[0][my_rank] = mid + my_rank - 1 ;
            MPI_Send(&f[0][my_rank],1,MPI_INT,0,0,MPI_COMM_WORLD);

			/*计算下次适应算法使用的箱子数目 m*/
            MPI_Recv(&f[0][1],group_size-1,MPI_INT,0,2,MPI_COMM_WORLD,&status);
            f[0][group_size] = group_size - 1 ;
            h = 0 ;
            for ( i = 1 ; i < group_size ; i++ )
            {
                e[0][i] = 1 ;
            }
            fflag = 0 ;
            while ( fflag == 0 )
            {
                if ( h > 0 )
                {
                    for ( j = 1 ; j <= group_size - 1 ; j++ )
                    {
                        MPI_Recv(&e[h][j],1,MPI_INT,j,j+h*1000,MPI_COMM_WORLD,&status);
                        MPI_Recv(&f[h][j],1,MPI_INT,j,j+h*1000,MPI_COMM_WORLD,&status);
                    }
                }
                if ( f[h][1] == group_size - 1 )
                {
                    fflag=1 ;
                }
                else
                {
                    h = h + 1 ;
                    if ( f[h-1][my_rank] == group_size - 1 )
                    {
                        e[h][my_rank] = e[h-1][my_rank];
                        f[h][my_rank]=f[h-1][my_rank];
                    }
                    else
                    {
                        t = f[h-1][my_rank] + 1 ;
                        e[h][my_rank] = e[h-1][my_rank] + e[h-1][t] ;
                        f[h][my_rank] = f[h-1][t] ;
                    }
                    for ( i = 1 ; i < group_size ; i++ )
                    {
                        MPI_Send(&e[h][my_rank],1,MPI_INT,i,my_rank+h*1000,MPI_COMM_WORLD);
                        MPI_Send(&f[h][my_rank],1,MPI_INT,i,my_rank+h*1000,MPI_COMM_WORLD);
                    }
                }
            }
            height=h;
            if ( my_rank == 1 )
            {
                m = e[height][1] ;
                for ( i = 2 ; i < group_size ; i++ )
                {
                    MPI_Send(&m,1,MPI_INT,i,2,MPI_COMM_WORLD);
                }
            }
            else
            {
                MPI_Recv(&m,1,MPI_INT,1,2,MPI_COMM_WORLD,&status);
            }

			/*计算d[0][j]=第j个箱子中第一个物品在输入序列中的编号*/
            tmpd = 1 ;
            for ( h = height ; h >= 0 ; h-- )
            {
                if ( my_rank <= tmpd && my_rank <= m )
                {
                    if ( my_rank == 1 )
                    {
                        d[h][1] = 1 ;
                        MPI_Send(&d[h][1],1,MPI_INT,2,h*100+1,MPI_COMM_WORLD);
                    }
                    if ( my_rank % 2 == 0 )
                    {
                        MPI_Recv(&d[h+1][my_rank/2],1,MPI_INT,my_rank/2,(h+1)*100+my_rank/2,MPI_COMM_WORLD,&status);
                        td = d[h+1][my_rank/2] ;
                        d[h][my_rank] = f[h][td] + 1 ;
                        if ( my_rank*2 <= tmpd*2 && my_rank*2 <= m )
                        {
                            MPI_Send(&d[h][my_rank],1,MPI_INT,my_rank*2,h*100+my_rank,MPI_COMM_WORLD);
                        }
                        if ( my_rank*2 <= tmpd*2 && my_rank*2 - 1 <= m )
                            MPI_Send(&d[h][my_rank],1,MPI_INT,my_rank*2-1,h*100+my_rank,MPI_COMM_WORLD);
                    }
                    if ( my_rank % 2 !=0 && my_rank > 1 )
                    {
                        MPI_Recv(&d[h+1][(my_rank+1)/2],1,MPI_INT,(my_rank+1)/2,(h+1)*100+(my_rank+1)/2,MPI_COMM_WORLD,&status);
                        d[h][my_rank] = d[h+1][(my_rank+1)/2] ;
                        if ( my_rank*2 <= tmpd*2 && my_rank*2 <= m )
                        {
                            MPI_Send(&d[h][my_rank],1,MPI_INT,my_rank*2,h*100+my_rank,MPI_COMM_WORLD);
                        }
                        if ( my_rank*2 <= tmpd*2 && my_rank*2-1 <= m )
                            MPI_Send(&d[h][my_rank],1,MPI_INT,my_rank*2-1,h*100+my_rank,MPI_COMM_WORLD);
                    }
                }
                tmpd = tmpd * 2 ;
            }
            if ( my_rank == 1 ) MPI_Send(&m,1,MPI_INT,0,0,MPI_COMM_WORLD);
            if ( my_rank <= m )
            {
                MPI_Send(&d[0][my_rank],1,MPI_INT,0,0,MPI_COMM_WORLD);
            }

			/*计算r[j]=第j个物品所放入的箱子序号*/

            MPI_Recv(&d[0][1],m,MPI_INT,0,1,MPI_COMM_WORLD,&status);
            if ( d[0][m] <= my_rank )
            {
                r[my_rank] = m ;
            }
            else
            {
                for ( i = 0 ; i <= m ; i++ )
                {
                    if ( d[0][i] > my_rank ) break ;
                }
                r[my_rank] = i - 1 ;
            }
            printf ( "number %d goods put into box %d\n" , my_rank, r[my_rank] );
        }
    }
    endtime = MPI_Wtime();
    printf(" that tooks %f second!\n",endtime-starttime);
    MPI_Finalize();
}
