#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 获取每个进程的主机名
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);

    // 1.1 按节点分组
    // 每个进程的 hostname 作为键，创建一个新的通信器
    MPI_Comm node_comm;
    //MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &node_comm);
    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    printf("World Rank %d: Node Rank %d on Host %s\n", world_rank, node_rank, hostname);

    // 1.2 按节点分组后实现广播
    int root = 0;  // 定义全局广播根进程
    int message = 0;

    if (world_rank == root) {
        message = 42;  // Root 进程初始化消息
        printf("Root process broadcasting message: %d\n", message);
    }

    // 全局根进程将消息发送到每个节点的 0 号进程
    if (node_rank == 0) {
        MPI_Bcast(&message, 1, MPI_INT, root, MPI_COMM_WORLD);
    }

    // 每个节点的 0 号进程在本地节点内广播消息
    MPI_Bcast(&message, 1, MPI_INT, 0, node_comm);

    printf("World Rank %d (Node Rank %d): Received message %d\n", world_rank, node_rank, message);

    // 清理通信器
    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return 0;
}
