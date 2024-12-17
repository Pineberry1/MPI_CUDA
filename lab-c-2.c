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

    // 二叉树全和
    int step, recv;
    for (step = 1; step < N; step *= 2) {
        if(rank % (step << 1) == 0){
            partner = rank + step;
            if(partner < N){
                MPI_Recv(&recv, 1, MPI_INT, partner, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            local_sum += recv;
        }
        if(rank - step > 0 && (rank - step) % (step << 1) == 0){
            MPI_Send(&local_sum, 1, MPI_INT, rank - step, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    // 输出每个进程最终的全和
    printf("Rank %d, Global sum: %d\n", world_rank, local_sum);

    MPI_Finalize();
    return 0;
}