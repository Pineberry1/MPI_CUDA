#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
int MyBcastMPI(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
{
    return 0;
}
*/
int main(int argc, char *argv[])
{
    int id_procs, num_procs;
    char buf[16];
    int root = 0;
    MPI_Group world_group, new_group;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_procs);
    if (id_procs == root)
    {
        strcpy(buf, "hello,MPI!");
    }
    //1.1按节点分组
    MPI_Comm split_comm_world;
    MPI_Status status;

    int rank;
    int size;

    MPI_Comm_split(MPI_COMM_WORLD, id_procs % 4, id_procs, &split_comm_world);
    MPI_Comm_rank(split_comm_world, &rank);
    MPI_Comm_size(split_comm_world, &size);
    printf("MPI Comm rank %d, node id %d\n", id_procs, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    //1.2root发送到每个节点的0号进程
    //0号进程成一组
    MPI_Group group_world, zero_group;
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    if(rank == 0){
        MPI_Group_incl(group_world, 1, &id_procs, &zero_group);
    }
    if(rank != 0 && root == id_procs){
        MPI_Group_incl(group_world, 1, &id_procs, &zero_group);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm zero_comm;
    MPI_Comm_create(MPI_COMM_WORLD, zero_group, &zero_comm);
    //root向0组发送
    strcpy(buf, "none", 5);
    if(root == id_procs){
        strcpy(buf, "wake", 5);
    }
    MPI_Bcast(&buf, 16, MPI_CHAR, root, zero_comm);
    //0号进程接收并广播
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Bcast(&buf, 16, MPI_CHAR, 0, split_comm_world);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("MPI Comm rank %d, node id %d, buf: %s\n", id_procs, rank, buf);
    MPI_Finalize();
    return 0;
}