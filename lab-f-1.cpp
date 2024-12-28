#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define N 16 // Matrix size

// Function to update matrix B based on matrix A
void update(const std::vector<double> &A, std::vector<double> &B, int start_row, int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < N - 1; j++) {
            B[i * N + j] = (A[(i - 1) * N + j] + A[i * N + j + 1] + A[(i + 1) * N + j] + A[i * N + j - 1]) / 4.0;
            std::cout << A[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if N can be divided by size
    if (N % size != 0) {
        if (rank == 0) std::cerr << "Matrix size N must be divisible by the number of processors." << std::endl;
        MPI_Finalize();
        return -1;
    }

    // Initialize matrices
    std::vector<double> A, B;
    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        for (int i = 0; i < N * N; i++) {
            //A[i] = rand() % 100;
            A[i] = 1;
        }
    }
    double start =  MPI_Wtime();//开始计时
    // Allocate local matrices
    int rows_per_proc = N / size;
    std::vector<double> local_A(rows_per_proc * N);
    std::vector<double> local_B(rows_per_proc * N);

    // Scatter rows of A to all processes
    MPI_Scatter(A.data(), rows_per_proc * N, MPI_DOUBLE, local_A.data(), rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute the range of rows for each process
    int start_row = (rank == 0) ? 1 : 0;
    int end_row = (rank == size - 1) ? rows_per_proc - 1 : rows_per_proc;

    // Update local matrix B based on local A
    update(local_A, local_B, start_row, end_row);

    // Gather results into B
    MPI_Gather(local_B.data(), rows_per_proc * N, MPI_DOUBLE, B.data(), rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double finish = MPI_Wtime();
    if (rank == 0) {
        // Optionally verify results or print
        std::cout << "Matrix update completed." << std::endl;
        printf("按行块划分耗时: %f\n", finish - start);
        // for (int i = 0; i < N; i++) {
        //     for(int j = 0; j < N; ++ j)
        //         std::cout << B[i*N + j] << " ";
        //     std::cout << std::endl;
        // }
    }

    MPI_Finalize();
    return 0;
}
