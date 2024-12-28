#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include<cassert>

#define N 1024 // Matrix size
int rank, size, rows_per_proc, cols_per_proc, grid_dim;
int proc_row, proc_col;

// Function to update matrix B based on matrix A
void update(const std::vector<double> &A, std::vector<double> &B, int start_row, int end_row, int start_col, int end_col, bool top_comm, bool bottom_comm, bool left_comm, bool right_comm) {
    MPI_Request requests[4];
    std::vector<double> recv_top(N), recv_bottom(N), recv_left(rows_per_proc), recv_right(rows_per_proc);
    std::vector<double> send_top(cols_per_proc), send_bottom(cols_per_proc), send_left(rows_per_proc), send_right(rows_per_proc);

    if (top_comm) {
        for (int j = 0; j < cols_per_proc; ++j) send_top[j] = A[j];
        MPI_Isend(send_top.data(), cols_per_proc, MPI_DOUBLE, rank - grid_dim, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(recv_top.data(), cols_per_proc, MPI_DOUBLE, rank - grid_dim, 1, MPI_COMM_WORLD, &requests[0]);
    }
    if (bottom_comm) {
        for (int j = 0; j < cols_per_proc; ++j) send_bottom[j] = A[(rows_per_proc - 1) * cols_per_proc + j];
        MPI_Isend(send_bottom.data(), cols_per_proc, MPI_DOUBLE, rank + grid_dim, 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Irecv(recv_bottom.data(), cols_per_proc, MPI_DOUBLE, rank + grid_dim, 0, MPI_COMM_WORLD, &requests[1]);
    }
    if (left_comm) {
        for (int i = 0; i < rows_per_proc; ++i) send_left[i] = A[i * cols_per_proc];
        MPI_Isend(send_left.data(), rows_per_proc, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(recv_left.data(), rows_per_proc, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &requests[2]);
    }
    if (right_comm) {
        for (int i = 0; i < rows_per_proc; ++i) send_right[i] = A[i * cols_per_proc + (cols_per_proc - 1)];
        MPI_Isend(send_right.data(), rows_per_proc, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &requests[3]);
        MPI_Irecv(recv_right.data(), rows_per_proc, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &requests[3]);
    }

    for (int i = start_row; i < end_row; i++) {
        for (int j = start_col; j < end_col; j++) {
            B[i * cols_per_proc + j] = 0;
            if (i > 0) B[i * cols_per_proc + j] += A[(i - 1) * cols_per_proc + j];
            if (i < rows_per_proc - 1) B[i * cols_per_proc + j] += A[(i + 1) * cols_per_proc + j];
            if (j > 0) B[i * cols_per_proc + j] += A[i * cols_per_proc + (j - 1)];
            if (j < cols_per_proc - 1) B[i * cols_per_proc + j] += A[i * cols_per_proc + (j + 1)];
        }
    }

    if (top_comm) {
        MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
        for (int j = 0; j < cols_per_proc; ++j) {
            B[j] += recv_top[j];
        }
    }
    if (bottom_comm) {
        MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
        for (int j = 0; j < cols_per_proc; ++j) {
            B[(rows_per_proc - 1) * cols_per_proc + j] += recv_bottom[j];
        }
    }
    if (left_comm) {
        MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
        for (int i = 0; i < rows_per_proc; ++i) {
            B[i * cols_per_proc] += recv_left[i];
        }
    }
    if (right_comm) {
        MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        for (int i = 0; i < rows_per_proc; ++i) {
            B[i * cols_per_proc + (cols_per_proc - 1)] += recv_right[i];
        }
    }

    for (auto &x : B) {
        x /= 4;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    grid_dim = std::sqrt(size);
    assert(grid_dim * grid_dim == size);

    rows_per_proc = N / grid_dim;
    cols_per_proc = N / grid_dim;
    proc_row = rank / grid_dim;
    proc_col = rank % grid_dim;

    std::vector<double> A, B;
    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() % 100;
        }
    }

    double start = MPI_Wtime();

    std::vector<double> local_A(rows_per_proc * cols_per_proc);
    std::vector<double> local_B(rows_per_proc * cols_per_proc);

    // MPI_Datatype block_type;
    // MPI_Type_vector(rows_per_proc, cols_per_proc, N, MPI_DOUBLE, &block_type);
    // MPI_Type_create_resized(block_type, 0, cols_per_proc * sizeof(double), &block_type);
    // MPI_Type_commit(&block_type);

    std::vector<int> send_counts(size, rows_per_proc * cols_per_proc);
    std::vector<int> displs(size);
    for (int i = 0; i < grid_dim; ++i) {
        for (int j = 0; j < grid_dim; ++j) {
            displs[i * grid_dim + j] = i * rows_per_proc * N + j * cols_per_proc;
        }
    }
    MPI_Scatterv(A.data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_A.data(), rows_per_proc * cols_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int start_row = (proc_row == 0) ? 1 : 0;
    int end_row = (proc_row == grid_dim - 1) ? rows_per_proc - 1 : rows_per_proc;
    int start_col = (proc_col == 0) ? 1 : 0;
    int end_col = (proc_col == grid_dim - 1) ? cols_per_proc - 1 : cols_per_proc;

    update(local_A, local_B, start_row, end_row, start_col, end_col, proc_row > 0, proc_row < grid_dim - 1, proc_col > 0, proc_col < grid_dim - 1);

    MPI_Gatherv(local_B.data(), rows_per_proc * cols_per_proc, MPI_DOUBLE, B.data(), send_counts.data(), displs.data(), block_type, 0, MPI_COMM_WORLD);

    double finish = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Matrix update completed." << std::endl;
        printf("棋盘划分%d核耗时: %f\n", size, finish - start);
        // for (int i = 0; i < N; i++) {
        //     for(int j = 0; j < N; ++ j)
        //         std::cout << B[i*N + j] << " ";
        //     std::cout << std::endl;
        // }
    }

    MPI_Type_free(&block_type);
    MPI_Finalize();
    return 0;
}
