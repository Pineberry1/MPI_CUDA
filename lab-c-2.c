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
    for (step = 1; step < N; step <<= 1) {
        if(world_rank % (step << 1) == 0){
            int partner = world_rank + step;
            if(partner < N){
                MPI_Recv(&recv, 1, MPI_INT, partner, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            local_sum += recv;
        }
        if(world_rank - step >= 0 && (world_rank - step) % (step << 1) == 0){
            MPI_Send(&local_sum, 1, MPI_INT, world_rank - step, step, MPI_COMM_WORLD);
        }
    }
    for(;step > 1; step >>= 1){
        if(world_rank % step == 0){
            int partner = world_rank + (step >> 1);
            if(partner < N){
                MPI_Send(&local_sum, 1, MPI_INT, partner, step, MPI_COMM_WORLD);
            }
        }
        if(world_rank - (step >> 1) >= 0 && (world_rank - (step >> 1)) % (step) == 0){
            MPI_Recv(&recv, 1, MPI_INT, world_rank - (step >> 1), step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_sum = recv;
        }
    }
    printf("world_rank %d, Global sum: %d\n", world_rank, local_sum);

    MPI_Finalize();
    return 0;
}