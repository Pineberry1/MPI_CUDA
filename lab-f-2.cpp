#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstdlib>

// 计算每个进程上局部结果矩阵的值
void compute_local_result(int process_rank, int total_processes, int matrix_size, std::vector<std::vector<double>>& local_matrix, std::vector<std::vector<double>>& result_matrix) {
    // 计算每个进程所负责的子矩阵的大小
    int local_matrix_size = matrix_size / total_processes;

    // 计算当前进程的起始行和列
    int start_row = (process_rank / (total_processes / 2)) * local_matrix_size;
    int start_col = (process_rank % (total_processes / 2)) * local_matrix_size;

    // 遍历当前进程分配的局部矩阵范围，计算结果矩阵的每个值
    for (int i = start_row; i < start_row + local_matrix_size; i++) {
        for (int j = start_col; j < start_col + local_matrix_size; j++) {
            // 根据相邻元素的值计算结果矩阵的每个元素
            result_matrix[i - start_row][j - start_col] = (
                (i > 0 && j > 0 ? local_matrix[i - start_row - 1][j - start_col - 1] : local_matrix[i - start_row][j - start_col]) + // 上左
                (i > 0 ? local_matrix[i - start_row - 1][j - start_col] : local_matrix[i - start_row][j - start_col]) + // 上
                (i > 0 && j < local_matrix_size - 1 ? local_matrix[i - start_row - 1][j - start_col + 1] : local_matrix[i - start_row][j - start_col]) + // 上右
                (j > 0 ? local_matrix[i - start_row][j - start_col - 1] : local_matrix[i - start_row][j - start_col]) + // 左
                (j < local_matrix_size - 1 ? local_matrix[i - start_row][j - start_col + 1] : local_matrix[i - start_row][j - start_col]) + // 右
                (i < local_matrix_size - 1 && j > 0 ? local_matrix[i - start_row + 1][j - start_col - 1] : local_matrix[i - start_row][j - start_col]) + // 下左
                (i < local_matrix_size - 1 ? local_matrix[i - start_row + 1][j - start_col] : local_matrix[i - start_row][j - start_col]) + // 下
                (i < local_matrix_size - 1 && j < local_matrix_size - 1 ? local_matrix[i - start_row + 1][j - start_col + 1] : local_matrix[i - start_row][j - start_col]) // 下右
            ) / 4.0;  // 计算均值
        }
    }
}

int main(int argc, char** argv) {
    int process_rank, total_processes;
    MPI_Init(&argc, &argv); // 初始化 MPI 环境
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank); // 获取当前进程的 rank
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes); // 获取进程总数

    int matrix_size = 100; // 定义矩阵的大小
    std::vector<std::vector<double>> matrix_A(matrix_size, std::vector<double>(matrix_size)); // 初始化矩阵 A
    std::vector<std::vector<double>> matrix_B(matrix_size, std::vector<double>(matrix_size)); // 初始化矩阵 B
    std::vector<std::vector<double>> local_A, local_B; // 每个进程上局部的 A 和 B 矩阵

    if (process_rank == 0) {
        // 初始化矩阵 A，赋值为 [0, 1) 范围内的随机数
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                matrix_A[i][j] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }
    double start = MPI_Wtime();
    // 广播矩阵大小 N 到所有进程
    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_matrix_size = matrix_size / total_processes; // 计算每个进程分配的子矩阵大小
    local_A.resize(local_matrix_size, std::vector<double>(local_matrix_size)); // 为 local_A 分配内存
    local_B.resize(local_matrix_size, std::vector<double>(local_matrix_size)); // 为 local_B 分配内存

    // 定义 MPI 数据类型，表示子矩阵
    MPI_Datatype submatrix_type;
    MPI_Type_vector(local_matrix_size, local_matrix_size, matrix_size, MPI_DOUBLE, &submatrix_type);
    MPI_Type_commit(&submatrix_type);

    // 使用 MPI_Scatter 将矩阵 A 划分到各个进程
    MPI_Scatter(matrix_A.data(), 1, submatrix_type, local_A.data(), local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 每个进程根据自己的局部矩阵计算局部结果矩阵
    compute_local_result(process_rank, total_processes, matrix_size, local_A, local_B);

    // 将各个进程计算的局部结果矩阵收集到主进程
    MPI_Gather(local_B.data(), local_matrix_size * local_matrix_size, MPI_DOUBLE, matrix_B.data(), 1, submatrix_type, 0, MPI_COMM_WORLD);

    // 在主进程输出计算结果矩阵
    double finish = MPI_Wtime();
    if (process_rank == 0) {
        // Optionally verify results or print
        std::cout << "Matrix update completed." << std::endl;
        printf("按棋盘划分%d核耗时: %f\n", size, finish - start);
        // for (int i = 0; i < N; i++) {
        //     for(int j = 0; j < N; ++ j)
        //         std::cout << B[i*N + j] << " ";
        //     std::cout << std::endl;
        // }
    }

    MPI_Finalize(); // 结束 MPI 环境
    return 0;
}