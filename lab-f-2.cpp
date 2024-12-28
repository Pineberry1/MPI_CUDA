
#include <mpi.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include<cmath>

#define N 2048 // Global matrix size

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute grid dimensions
    int grid_size = std::sqrt(size); // Assuming size is a perfect square
    if (grid_size * grid_size != size) {
        if (rank == 0) std::cerr << "Number of processes must be a perfect square." << std::endl;
        MPI_Finalize();
        return -1;
    }

    int rows_per_proc = N / grid_size;
    int cols_per_proc = N / grid_size;

    if (N % grid_size != 0) {
        if (rank == 0) std::cerr << "Matrix size must be divisible by grid size." << std::endl;
        MPI_Finalize();
        return -1;
    }

    std::vector<double> A, local_A;

    if (rank == 0) {
        // Initialize the global matrix A
        A.resize(N * N);
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() % 100;
        }
    }

    // Prepare send_counts and displs for Scatterv
    std::vector<int> send_counts(size, 0), displs(size, 0);
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int rank = i * grid_size + j;
            send_counts[rank] = rows_per_proc * cols_per_proc;
            displs[rank] = i * rows_per_proc * N + j * cols_per_proc;
        }
    }

    // Create MPI datatype for subarrays
    MPI_Datatype block_type;
    int sizes[2] = {N, N};
    int subsizes[2] = {rows_per_proc, cols_per_proc};
    int starts[2] = {0, 0};

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &block_type);
    MPI_Type_commit(&block_type);

    // Allocate local matrix
    local_A.resize(rows_per_proc * cols_per_proc);

    // Scatter the matrix
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, send_counts.data(), displs.data(),
                 block_type, local_A.data(), rows_per_proc * cols_per_proc, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Debug: Print local matrix size
    std::cout << "Process " << rank << " received " << local_A.size()
              << " elements." << std::endl;

    MPI_Type_free(&block_type);
    MPI_Finalize();
    return 0;
}