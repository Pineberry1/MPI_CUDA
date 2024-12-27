#include <mpi.h>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <cassert>
#include<random>
const int P = 4;
int N;
int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    N = world_size;
    assert(N > P);
    int Q = N - P;
    assert(Q > P);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(0, 1);
    MPI_Comm group_work, group_server;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank % P, world_rank, &group_work);//工作组
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < P ? 0 : 1, world_rank, &group_server);//参数服务器组
    double globalAvg;
    if(world_rank >= P){//工作进程
        while(1){
            double num = distrib(gen);
            double recvnum;
            MPI_Sendrecv(&num, 1, MPI_DOUBLE, 0, 0, &recvnum, 1, MPI_DOUBLE, 0, 1, group_work, MPI_STATUS_IGNORE);
        }
    }
    else{//参数服务器进程
        int group_size;
        double recvdata;
        MPI_Comm_size(group_work, &group_size);
        while(1){
            double recvsum = 0;
            for(int i = 0; i < group_size; ++ i){
                MPI_Recv(&recvdata, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, group_work, MPI_STATUS_IGNORE);
                recvsum += recvdata;
            }
            MPI_Allreduce(&recvsum, &globalAvg, 1, MPI_DOUBLE, MPI_SUM, group_server);
            globalAvg /= Q;
            MPI_Bcast(&globalAvg, 1, MPI_DOUBLE, 0, group_work);
        }
    }
}