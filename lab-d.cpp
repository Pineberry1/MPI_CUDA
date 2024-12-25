#include <cassert>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <ctime>

#define MATRIX_SIZE 4
#define GRID_DIM 2 // GRID_DIM * GRID_DIM = MATRIX_SIZE

int GlobalA[MATRIX_SIZE * MATRIX_SIZE], GlobalB[MATRIX_SIZE * MATRIX_SIZE], GlobalC[MATRIX_SIZE * MATRIX_SIZE];

// 矩阵乘法
void MultiplySubMatrices(int* matA, int* matB, int* matC, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                matC[i * dim + j] += matA[i * dim + k] * matB[k * dim + j];
            }
        }
    }
}

// Fox算法实现
void PerformFoxAlgorithm(int* localA, int* localB, int* localC, int processID) {
    int* bufferA = (int*)malloc(MATRIX_SIZE * sizeof(int));
    int* bufferB = (int*)malloc(MATRIX_SIZE * sizeof(int));

    int colIdx = processID % GRID_DIM;
    int rowIdx = processID / GRID_DIM;

    memset(localC, 0, sizeof(int) * MATRIX_SIZE);

    for (int step = 0; step < GRID_DIM; ++step) {
        int sender = (colIdx + step) % GRID_DIM;
        if (rowIdx == sender) {
            memcpy(bufferA, localA, MATRIX_SIZE * sizeof(int));
            for (int k = 0; k < GRID_DIM; ++k) {
                if (k != colIdx) {
                    MPI_Send(localA, MATRIX_SIZE, MPI_INT, rowIdx * GRID_DIM + k, step, MPI_COMM_WORLD);
                }
            }
        } else {
            int source = (rowIdx - step + GRID_DIM) % GRID_DIM;
            MPI_Recv(bufferA, MATRIX_SIZE, MPI_INT, source * GRID_DIM + colIdx, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MultiplySubMatrices(bufferA, localB, localC, GRID_DIM);

        if (step < GRID_DIM - 1) {
            int sendTo = rowIdx > 0 ? rowIdx - 1 : GRID_DIM - 1;
            int recvFrom = rowIdx < GRID_DIM - 1 ? rowIdx + 1 : 0;

            MPI_Sendrecv(localB, MATRIX_SIZE, MPI_INT, sendTo * GRID_DIM + colIdx, 0,
                         bufferB, MATRIX_SIZE, MPI_INT, recvFrom * GRID_DIM + colIdx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(localB, bufferB, MATRIX_SIZE * sizeof(int));
        }
    }

    free(bufferA);
    free(bufferB);
}

// 提取子矩阵
void ExtractSubMatrix(int* sourceMatrix, int* destMatrix, int startX, int startY, int subDim) {
    for (int i = 0; i < subDim; ++i) {
        for (int j = 0; j < subDim; ++j) {
            destMatrix[i * subDim + j] = sourceMatrix[(startX * subDim + i) * MATRIX_SIZE + startY * subDim + j];
        }
    }
}

// 分发子矩阵
void DistributeMatrices(int* localA, int* localB, int processID) {
    for (int i = 0; i < GRID_DIM; ++i) {
        for (int j = 0; j < GRID_DIM; ++j) {
            if (i == 0 && j == 0) continue;
            ExtractSubMatrix(GlobalA, localA, i, j, GRID_DIM);
            ExtractSubMatrix(GlobalB, localB, i, j, GRID_DIM);
            int targetProcess = i * GRID_DIM + j;
            MPI_Send(localA, GRID_DIM * GRID_DIM, MPI_INT, targetProcess, processID, MPI_COMM_WORLD);
            MPI_Send(localB, GRID_DIM * GRID_DIM, MPI_INT, targetProcess, processID, MPI_COMM_WORLD);
        }
    }
    ExtractSubMatrix(GlobalA, localA, 0, 0, GRID_DIM);
    ExtractSubMatrix(GlobalB, localB, 0, 0, GRID_DIM);
}

// 收集结果矩阵
void CollectResults(int* localC, int* resultMatrix) {
    for (int i = 0; i < GRID_DIM; ++i) {
        for (int j = 0; j < GRID_DIM; ++j) {
            if (i == 0 && j == 0) continue;
            MPI_Recv(localC, MATRIX_SIZE, MPI_INT, i * GRID_DIM + j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int x = 0; x < GRID_DIM; ++x) {
                for (int y = 0; y < GRID_DIM; ++y) {
                    resultMatrix[(i * GRID_DIM + x) * MATRIX_SIZE + (j * GRID_DIM + y)] = localC[x * GRID_DIM + y];
                }
            }
        }
    }
}

int main() {
    MPI_Init(NULL, NULL);

    int processID, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &processID);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    assert(numProcesses == MATRIX_SIZE);

    int* localA = (int*)malloc(MATRIX_SIZE * sizeof(int));
    int* localB = (int*)malloc(MATRIX_SIZE * sizeof(int));
    int* localC = (int*)malloc(MATRIX_SIZE * sizeof(int));

    if (processID == 0) {
        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) GlobalA[i] = 1, GlobalB[i] = 1;
        DistributeMatrices(localA, localB, processID);
    } else {
        MPI_Recv(localA, MATRIX_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localB, MATRIX_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    PerformFoxAlgorithm(localA, localB, localC, processID);
    MPI_Barrier(MPI_COMM_WORLD);

    if (processID == 0) {
        for (int i = 0; i < GRID_DIM; ++i) {
            for (int j = 0; j < GRID_DIM; ++j) {
                GlobalC[i * MATRIX_SIZE + j] = localC[i * GRID_DIM + j];
            }
        }
        CollectResults(localC, GlobalC);
        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) printf("%d ", GlobalC[i]);
    } else {
        MPI_Send(localC, MATRIX_SIZE, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    free(localA);
    free(localB);
    free(localC);
    MPI_Finalize();
    return 0;
}