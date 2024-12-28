#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 假设 N 为 2 的幂次方
    int N = world_size;
    int local_sum = world_rank + 1;  // 每个进程的数据
    double start =  MPI_Wtime();//开始计时

    // 蝶式全和
    int step;
    for (step = 1; step < N; step *= 2) {
        int partner_rank = world_rank ^ step;  // 计算与当前进程通信的伙伴进程

        if (partner_rank < N) {
            int received_sum;
            MPI_Sendrecv(&local_sum, 1, MPI_INT, partner_rank, 0, 
                         &received_sum, 1, MPI_INT, partner_rank, 0, 
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 更新本地的全和
            local_sum += received_sum;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double finish = MPI_Wtime();
    // 输出每个进程最终的全和
    printf("Rank %d, Global sum: %d\n", world_rank, local_sum);
    if(world_rank == 0){
        printf("碟式全和耗时: %f\n", finish - start);
    }
    MPI_Finalize();
    return 0;
}