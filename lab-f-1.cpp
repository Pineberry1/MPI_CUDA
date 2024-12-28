#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define N 2048 // Matrix size

// Function to update matrix B based on matrix A
void update(const std::vector<double> &A, std::vector<double> &B, int row_start, int row_end, int col_start, int col_end) {
    for (int i = row_start; i < row_end; i++) {
        for (int j = col_start; j < col_end; j++) {
            B[i * N + j] = (A[(i - 1) * N + j] + A[i * N + j + 1] + A[(i + 1) * N + j] + A[i * N + j - 1]) / 4.0;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure size is a perfect square for chessboard decomposition
    int q = static_cast<int>(std::sqrt(size));
    if (q * q != size) {
        if (rank == 0) std::cerr << "Number of processors must be a perfect square." << std::endl;
        MPI_Finalize();
        return -1;
    }

    // Initialize matrices
    std::vector<double> A, B;
    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() % 100;
        }
    }

    // Determine local matrix dimensions
    int rows_per_proc = N / q;
    int cols_per_proc = N / q;
    std::vector<double> local_A((rows_per_proc + 2) * (cols_per_proc + 2)); // Include halo rows and columns
    std::vector<double> local_B(rows_per_proc * cols_per_proc);

    // Scatter submatrices of A to all processes
    std::vector<double> sub_A(rows_per_proc * cols_per_proc);
    if (rank == 0) {
        for (int p = 0; p < size; p++) {
            int proc_row = p / q;
            int proc_col = p % q;
            for (int i = 0; i < rows_per_proc; i++) {
                for (int j = 0; j < cols_per_proc; j++) {
                    sub_A[i * cols_per_proc + j] = A[(proc_row * rows_per_proc + i) * N + proc_col * cols_per_proc + j];
                }
            }
            if (p == 0) {
                local_A.assign(sub_A.begin(), sub_A.end());
            } else {
                MPI_Send(sub_A.data(), rows_per_proc * cols_per_proc, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_A.data(), rows_per_proc * cols_per_proc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Exchange halo rows and columns with neighbors
    MPI_Request requests[8];
    for (int i = 0; i < 8; i++) requests[i] = MPI_REQUEST_NULL;

    int proc_row = rank / q;
    int proc_col = rank % q;

    // Send/receive halo rows
    if (proc_row > 0) { // Top neighbor
        MPI_Isend(local_A.data() + cols_per_proc, cols_per_proc, MPI_DOUBLE, rank - q, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(local_A.data(), cols_per_proc, MPI_DOUBLE, rank - q, 1, MPI_COMM_WORLD, &requests[1]);
    }
    if (proc_row < q - 1) { // Bottom neighbor
        MPI_Isend(local_A.data() + rows_per_proc * cols_per_proc, cols_per_proc, MPI_DOUBLE, rank + q, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(local_A.data() + (rows_per_proc + 1) * cols_per_proc, cols_per_proc, MPI_DOUBLE, rank + q, 0, MPI_COMM_WORLD, &requests[3]);
    }

    // Send/receive halo columns
    if (proc_col > 0) { // Left neighbor
        MPI_Isend(local_A.data() + cols_per_proc, 1, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &requests[4]);
        MPI_Irecv(local_A.data(), 1, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &requests[5]);
    }
    if (proc_col < q - 1) { // Right neighbor
        MPI_Isend(local_A.data() + cols_per_proc - 1, 1, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &requests[6]);
        MPI_Irecv(local_A.data() + cols_per_proc, 1, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &requests[7]);
    }

    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

    // Update local matrix B based on local A
    update(local_A, local_B, 1, rows_per_proc + 1, 1, cols_per_proc + 1);

    // Gather results into B
    if (rank == 0) {
        for (int p = 0; p < size; p++) {
            if (p == 0) {
                for (int i = 0; i < rows_per_proc; i++) {
                    for (int j = 0; j < cols_per_proc; j++) {
                        B[(proc_row * rows_per_proc + i) * N + proc_col * cols_per_proc + j] = local_B[i * cols_per_proc + j];
                    }
                }
            } else {
                MPI_Recv(sub_A.data(), rows_per_proc * cols_per_proc, MPI_DOUBLE, p, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int proc_row = p / q;
                int proc_col = p % q;
                for (int i = 0; i < rows_per_proc; i++) {
                    for (int j = 0; j < cols_per_proc; j++) {
                        B[(proc_row * rows_per_proc + i) * N + proc_col * cols_per_proc + j] = sub_A[i * cols_per_proc + j];
                    }
                }
            }
        }
    } else {
        MPI_Send(local_B.data(), rows_per_proc * cols_per_proc, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Optionally verify results or print
        std::cout << "Matrix update completed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
