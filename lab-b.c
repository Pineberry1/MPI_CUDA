#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int My_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, 
                void *recvbuf, int recvcount, MPI_Datatype recvtype, 
                MPI_Comm comm) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int sendtype_size, recvtype_size;
    MPI_Type_size(sendtype, &sendtype_size); 
    MPI_Type_size(recvtype, &recvtype_size);  

    for (int i = 0; i < world_size; ++i) {
        if(i != world_rank){
            MPI_Send(sendbuf + i * sendtype_size * sendcount, sendcount, sendtype, 
                    i, 0, comm);
            MPI_Recv(recvbuf + i * recvtype_size * recvcount, recvcount, recvtype, 
                    i, 0, comm, MPI_STATUS_IGNORE);
        }
        else{
            memcpy(recvbuf + i * recvtype_size * recvcount, sendbuf + i * sendtype_size * sendcount, sendtype_size * sendcount);
        }
    }


    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int *sendbuf = (int*)malloc(sizeof(int) * world_size);
    int *recvbuf = (int*)malloc(sizeof(int) * world_size);

    for (int i = 0; i < world_size; ++i) {
        sendbuf[i] = i;
        recvbuf[i] = -1;
    }
    double start =  MPI_Wtime();
    My_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);
    double finish = MPI_Wtime();
    printf("My_Alltoall耗时: %f\n", finish - start);
    
    MPI_Barrier(MPI_COMM_WORLD);

    start =  MPI_Wtime();
    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);
    finish = MPI_Wtime();
    printf("MPI_Alltoall耗时: %f\n", finish - start);

    printf("cur_rank: %d, recvdata: ", world_rank);
    for (int i = 0; i < world_size; ++i) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}