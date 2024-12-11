#include <mpi.h>
#include <stdio.h>

#define QUEENS 8
#define MAX_SOLUTIONS 92

typedef int bool;
const int true = 1;
const int false = 0;

/*枚举信号*/
enum msg_content
{
    READY,
    ACCOMPLISHED,
    NEW_TASK,
    TERMINATE
};

/*枚举信息标签*/
enum msg_tag
{
    REQUEST_TAG,
    SEED_TAG,
    REPLY_TAG,
    NUM_SOLUTIONS_TAG,
    SOLUTIONS_TAG
};

int solutions[MAX_SOLUTIONS][QUEENS];
int solution_count = 0;

/*
 *函数名：collides
 *功能：检查两个位置是否有列、对角线或反对角线冲突
 *输入：row1为位置1的行号；
 *      col1为位置1的列号；
 *      row2为位置2的行号；
 *      col2为位置2的列号。
 *输出：返回1代表两位置有行、对角线或反对角线冲突；
 *      否则返回0
 */
bool collides(int row1, int col1, int row2, int col2)
{
    return (col1 == col2)
        || (col1 - col2 == row1 - row2)
        || (col1 + row1 == col2 + row2);
}                                                 /* collides */


/*
 *函数名：generate_seed
 *功能：生成初始化棋盘，只初始化棋盘的前两列
 *输入：无
 *输出：返回0代表已没有可初始化的棋盘，
 *      否则返回生成的初始化的棋盘。
 */
int generate_seed()
{
    static int seed = 0;

    do
    {
        seed++;
    } while (seed <= QUEENS * QUEENS - 1
        && collides(0, seed / QUEENS, 1, seed % QUEENS));

    if (seed > QUEENS * QUEENS - 1)
        return 0;
    else
        return seed;
}                                                 /* generate_seed */


/*
 *函数名：print_solutions
 *功能：打印结果
 *输入：count为需要打印的结果的个数；
 *      solutions[][QUEENS]为所要打印出来的结果，即皇后所摆放的位置。
 *输出：无
 */
void print_solutions(int count, int solutions[][QUEENS])
{
    int i, j, k;

    for (i = 0; i < count; i++)
    {
		/*打印第i+1个结果*/
        printf("Solution %d :\n", i + 1);
        for (j = 0; j < QUEENS; j++)
        {
            printf("%d ", solutions[i][j]);
			/*打印棋盘，*表示皇后所在位置*/
            for (k = 0; k < solutions[i][j]; k++) printf("- ");
            printf("* ");
            for (k = QUEENS - 1; k > solutions[i][j]; k--) printf("- ");
            printf("\n");
        }
        printf("\n");
    }
}                                                 /* print_solutions */


/*
 *函数名：is_safe
 *功能：检查当前皇后所摆放的位置是否与已摆放的皇后的位置相冲突
 *输入：chessboard[]为存放皇后所摆放的位置的数组;
 *      row为当前位置的行；
 *      col为当前位置的列。
 *输出：返回0代表当前位置有冲突，不可摆放皇后；
        返回1代表当前位置没有冲突，可以摆放皇后。
 */
bool is_safe(int chessboard[], int row, int col)
{
    int i;

    for (i = 0; i < row; i++)
    {
		/*检查当前位置是否与第i行皇后的位置相冲突*/
        if (collides(i, chessboard[i], row, col))
            return false;
    }                                             /* for */
    return true;
}                                                 /* is_safe */


/*
 *函数名：place_queens
 *功能：为当前行的皇后寻找适当的摆放位置，
 *      如果皇后摆放完毕则记录所摆放的位置，
 *      递归的摆放后面的皇后，
 *      当所有皇后摆放完毕后，记录所得的解。
 *输入：chessboard[]为存放皇后所摆放位置的数组；
        row为当前皇后所要摆放的行号。
 *输出：无
 */
void place_queens(int chessboard[], int row)
{
    int i, col;

    if (row >= QUEENS)                            /*所有皇后均摆放完毕*/
    {
		/* 结束递归并记录当前解 */
        for (i = 0; i < QUEENS; i++)
        {
            solutions[solution_count][i] = chessboard[i];
        }
        solution_count++;
    }
    else
    {
		/*还有皇后没有摆好*/
        for (col = 0; col < QUEENS; col++)        /*在当前行寻找适当位置摆放皇后*/
        {
            if (is_safe(chessboard, row, col))    /*当前位置不冲突*/
            {
                chessboard[row] = col;            /* 在当前位置放置一个皇后 */
                place_queens(chessboard, row + 1);/* 递归放置下一个皇后 */
            }                                     /* if */
        }                                         /* for */
    }                                             /* else */
}                                                 /* place_queens */


/*
 *函数名：sequential_eight_queens
 *功能：串行地求解八皇后问题，并将解打印出来
 *输入：无
 *输出：无
 */
void sequential_eight_queens()
{
    int chessboard[QUEENS];

    solution_count = 0;
	/*开始求解八皇后问题*/
    place_queens(chessboard, 0);
	/*打印所求结果*/
    print_solutions(solution_count, solutions);
}                                                 /*sequential_eight_queens*/


/*
 *函数名：eight_queens_master
 *功能：生成初始化棋盘，并将其发送给空闲的子进程；
 *      从子进程中接受并记录解；
 *      当所有子进程都已终止且没有新的初始化棋盘时，打印所有的解。
 *输入：nodes为工作组中进程数
 *输出：无
 */
void eight_queens_master(int nodes)
{
    MPI_Status status;
    int active_slaves = nodes - 1;                // except the master itself
    int new_task = NEW_TASK;
    int terminate = TERMINATE;
    int reply;
    int child;
    int num_solutions;
    int seed;

    while (active_slaves)                         /*有未结束的子进程*/
    {
		/*从子进程中接受返回信号*/
        MPI_Recv(&reply, 1, MPI_INT, MPI_ANY_SOURCE, REPLY_TAG, MPI_COMM_WORLD, &status);

        child = status.MPI_SOURCE;

        if (reply == ACCOMPLISHED)                /*子进程已完成求解任务*/
        {
			/* 从子进程接收并记录解 */
            MPI_Recv(&num_solutions, 1, MPI_INT, child, NUM_SOLUTIONS_TAG, MPI_COMM_WORLD, &status);

            if (num_solutions > 0)
            {
                MPI_Recv(solutions[solution_count],
                    QUEENS * num_solutions, MPI_INT,
                    child, SOLUTIONS_TAG, MPI_COMM_WORLD, &status);
                solution_count += num_solutions;
            }
        }

        seed = generate_seed();                   /*产生初始棋盘*/
        if (seed)                                 /* 还有新的初始棋盘 */
        {
			/* 向子进程发送一个new_task信号 */
            MPI_Send(&new_task, 1, MPI_INT, child, REQUEST_TAG, MPI_COMM_WORLD);
			/* 向子进程发送一个合法的新棋盘 */
            MPI_Send(&seed, 1, MPI_INT, child, SEED_TAG, MPI_COMM_WORLD);
        }
        else                                      /* 已求出所有解 */
        {
			/* 向子进程发送终止信号 */
            MPI_Send(&terminate, 1, MPI_INT, child, REQUEST_TAG, MPI_COMM_WORLD);
            active_slaves--;
        }
    }                                             /* while */

	/*打印所有解*/
    print_solutions(solution_count, solutions);
}                                                 /* eight_queens_master */


/*
 *函数名：eight_queens_slave
 *功能：从主进程接受新的初始化的棋盘；
 *      在初始化的棋盘基础上求出所有合法的解；
 *      将求得的解发送给主进程。
 *输入：my_rank为子进程在整个工作组中的进程标识符
 *输出：无
 */
void eight_queens_slave(int my_rank)
{
    MPI_Status status;
    int ready = READY;
    int accomplished = ACCOMPLISHED;
    bool finished = false;
    int request;
    int seed;
    int num_solutions = 0;
    int chessboard[QUEENS];

	/*向主进程发送ready信号*/
    MPI_Send(&ready, 1, MPI_INT, 0, REPLY_TAG, MPI_COMM_WORLD);

    while (! finished)                            /*子进程未终止*/
    {
		/* 从主进程接收消息 */
        MPI_Recv(&request, 1, MPI_INT, 0, REQUEST_TAG, MPI_COMM_WORLD, &status);

        if (request == NEW_TASK)
        {
			/* 从主进程接收初始棋盘 */
            MPI_Recv(&seed, 1, MPI_INT, 0, SEED_TAG, MPI_COMM_WORLD, &status);

			/* 在初始棋盘基础上求解 */
            chessboard[0] = seed / QUEENS;
            chessboard[1] = seed % QUEENS;

            solution_count = 0;
            place_queens(chessboard, 2);

			/* 将解发送给主进程 */
            /*向主进程发送accomplished信号*/
            MPI_Send(&accomplished, 1, MPI_INT, 0, REPLY_TAG, MPI_COMM_WORLD);
            MPI_Send(&solution_count, 1, MPI_INT, 0, NUM_SOLUTIONS_TAG, MPI_COMM_WORLD);
            if (solution_count > 0)
            {
                MPI_Send(*solutions,
                    QUEENS * solution_count,
                    MPI_INT, 0, SOLUTIONS_TAG, MPI_COMM_WORLD);
            }
        }
        else                                      /* request == TERMINATE */
        {
			/*主进程要求子进程终止*/
            finished = true;
        }
    }                                             /* whlie */

}                                                 /* eight_queens_slave */


/******************** main ********************/

/*
 *函数名：main
 *功能：启动MPI计算；
 *      确定进程数和进程标识符；
 *      调用主进程和从进程程序并行求解八皇后问题。
 *输入：argc为命令行参数个数；
 *      argv为每个命令行参数组成的字符串数组。
 *输出：返回0代表程序正常结束；否则程序出错。
 */
int main(int argc, char* argv[])
{
    int nodes, my_rank;

	/*初始化*/
    MPI_Init(&argc, &argv);
	/*确定工作组中的进程个数*/
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
	/*确定自己在工作组中的进程标识符*/
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (nodes == 1)                               /*只有一个进程时，用串行算法求解*/
    {
        sequential_eight_queens();
    }

	/*并行求解八皇后问题*/
    if (! my_rank)
    {
        eight_queens_master(nodes);
    }
    else
    {
        eight_queens_slave(my_rank);
    }

	/*结束MPI计算*/
    MPI_Finalize();

    return 0;
}                                                 /* main */
