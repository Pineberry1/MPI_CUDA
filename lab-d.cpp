#include <mpi.h>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <cassert>
using namespace std;
template<class type>
class matrix{
public:
    int row, col;
    type* mat;
    matrix():row(0), col(0), mat(nullptr){}
    matrix(int r, int c): row(r), col(c){
        mat = new type[r*c];
    }
    ~matrix(){
        delete[] mat;
        row = 0, col = 0;
    }
    void print(){
        for(int i = 0; i < row; ++ i){
            for(int j = 0; j < col; ++ j){
                cout << mat[i*col + j] << " ";
            }
            cout << endl;
        }
    }
    int serialize(char* res) const{
        *(int*)res = row;
        *(((int*)res) + 1) = col;
        type* data = (type*)(((int*)res) + 2);
        for(int i = 0; i < row*col; ++ i){
            data[i] = mat[i];
        }
        return sizeof(int) * 2 + sizeof(type) * row * col;
    }
    static matrix<type>& unserialize(char* stream){//unknown type
        int r = *(int*)stream;
        int c = *(((int*)stream) + 1);
        matrix<type>* res = new matrix<type>(r, c);//alloc
        type* data = (type*)(((int*)stream) + 2);
        for(int i = 0; i < r; ++ i){
            for(int j = 0; j < c; ++ j){
                (*res)[i][j] = data[i*c + j];
            }
        }
        return *res;
    }
    type* operator [](int x){
        if(x >= 0 && x < row){
            return (mat + x*col);
        }
        else
            assert(0);
    }
        const type* operator [](int x) const{
        if(x >= 0 && x < row){
            return (mat + x*col);
        }
        else
            assert(0);
    }
    matrix<type>& operator = (const matrix<type>& m){//shallow copy
        row = m.row;
        col = m.col;
        mat = m.mat;
    }
    matrix(const matrix<type>& m): row(m.row), col(m.col){
        row = m.row;
        col = m.col;
        mat = m.mat;
    }
    matrix<type>& split_copy(int x, int y, int tmp_row, int tmp_col){
        matrix* res = new matrix(tmp_row, tmp_col);//alloc
        for(int i = 0; i < tmp_row; ++ i){
            for(int j = 0; j < tmp_col; ++ j){
                (*res)[i][j] = (*this)[x + i][y + j];
            }
        }
        return *res;
    }
    matrix<type>& operator *(const matrix& m){
        matrix<type>* res = new matrix(row, m.col);//alloc
        memset(res->mat, 0, sizeof(type) * row * m.col);
        assert(col == m.row);
        for(int i = 0; i < row; ++ i){
            for(int j = 0; j < m.col; ++ j){
                for(int k = 0; k < m.row; ++ k){
                    (*res)[i][j] += (*this)[i][k] * m[k][j];
                }
            }
        }
        return *res;//maybe leak
    }
    matrix& operator +=(const matrix<type>& m){
        assert(row == m.row && col == m.col);
        for(int i = 0; i < row; ++ i){
            for(int j = 0; j < col; ++ j){
                mat[i*col + j] += m[i][j];
            }
        }
        return *this;
    }
    void clear(){
        memset(mat, 0, sizeof(type) * row * col);
    }
};
const int MAX_ROW = 1e4 + 5;
template<class type>
void sendmat(const matrix<type>& mat, char *buf, int senddest, int tag, MPI_Comm comm){
    int len = mat.serialize(buf);
    MPI_Send(&len, 1, MPI_INT, senddest, tag ^ (1 << 15), comm);
    MPI_Send(buf, len, MPI_CHAR, senddest, tag, comm);
}
template<class type>
void recvmat(matrix<type>& mat, char *buf, int recvdest, int tag, MPI_Comm comm){
    int len;
    MPI_Recv(&len, 1, MPI_INT, recvdest, tag ^ (1 << 15), comm, MPI_STATUS_IGNORE);
    MPI_Recv(buf, len, MPI_CHAR, recvdest, tag, comm, MPI_STATUS_IGNORE);
    mat = matrix<type>::unserialize(buf);
}
template<class type>
matrix<type>& FOX(matrix<type>&A, matrix<type>& B){
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int p = (int)sqrt(size);
    assert(p * p == size);
    int n = A.row;
    matrix<type>* c = new matrix<type>(n, n);
    c->clear();
    char* recv_bufa = new char[n * n * sizeof(type) + sizeof(int) * 2];//recv_a
    char* recv_bufb = new char[n * n * sizeof(type) + sizeof(int) * 2];//recv_b
    char* send_buf = new char[MAX_ROW * MAX_ROW];
    int j = rank % p;
    int i = (rank - j) / p;
    int senddest = 0, recvdest = 0;
    for(int round = 0; round < p; ++ round){
        if(i == (j + round) % p){
            for(int k = 0; k < p; ++ k){
                int len = A.serialize(send_buf);
                if(k != j){
                    MPI_Send(&len, 1, MPI_INT, i * p + k, (round + 1) * p, MPI_COMM_WORLD);
                    MPI_Send(send_buf, len, MPI_CHAR, i * p + k, (round + 1) * p + 1, MPI_COMM_WORLD);
                }
                else{
                    for(int l = 0; l < len; ++ l){
                        recv_bufa[l] = send_buf[l];
                    }
                }
            }
        }
        else{
            if(i < round){
                recvdest = i * p + i - round  + p;
            }
            else{
                recvdest = i * p + i - round;
            }
            int len;
            MPI_Recv(&len, 1, MPI_INT, recvdest, (round + 1) * p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_bufa, len, MPI_CHAR, recvdest, (round + 1) * p + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        matrix<type>& a = matrix<type>::unserialize(recv_bufa);
        (*c) += a*B;
        if(round == p - 1)break;
        senddest = i > 0 ? i - 1: i - 1 + p;
        recvdest = i < p - 1 ? i + 1: i + 1 - p;
        int len = B.serialize(send_buf);
        int len2;
        MPI_Sendrecv(&len, 1, MPI_INT, senddest * p + j, 0, &len2, 1, MPI_INT, recvdest * p + j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buf, len, MPI_CHAR, senddest * p + j, 1, recv_bufb, len2, MPI_CHAR, recvdest * p + j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        matrix<type>& tmp_b = matrix<type>::unserialize(recv_bufb);
        for(int i = 0; i < n; ++ i){
            for(int j = 0; j < n; ++ j){
                B[i][j] = tmp_b[i][j];
            }
        }
    }
    delete []recv_bufa;
    delete []recv_bufb;
    delete []send_buf;
    return *c;
}
const int mat_N = 128;
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    matrix<int>* A, *B, *C;
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == 0){
        A = new matrix<int>(mat_N, mat_N);
        B = new matrix<int>(mat_N, mat_N);
        C = new matrix<int>(mat_N, mat_N);
        for(int i = 0; i < mat_N; ++ i){
            for(int j = 0; j < mat_N; ++ j){
                (*A)[i][j] = rand() % 10;
                (*B)[i][j] = rand() % 10;
            }
        }
    }
    int p = (int)sqrt(world_size);
    assert(p * p == world_size);
    assert(mat_N % p == 0);
    int n = mat_N / p;
    matrix<int> tmpp;
    matrix<int>&a = tmpp, &b = tmpp, &c = tmpp;
    MPI_Barrier(MPI_COMM_WORLD);//start
    double start_fox = MPI_Wtime();
    char* bufa = new char[MAX_ROW * MAX_ROW];
    char* bufb = new char[MAX_ROW * MAX_ROW];
    if(world_rank == 0){
        for(int i = 0; i < p; ++ i){
            for(int j = 0; j < p; ++ j){
                if(i + j == 0)continue;
                a = A->split_copy(i * n, j * n, n, n);
                b = B->split_copy(i * n, j * n, n, n);
                sendmat(a, bufa, i * p + j, 'a', MPI_COMM_WORLD);
                sendmat(b, bufb, i * p + j, 'b', MPI_COMM_WORLD);
            }
        }
        a = A->split_copy(0, 0, n, n);
        b = B->split_copy(0, 0, n, n);
    }
    else{
        recvmat(a, bufa, 0, 'a', MPI_COMM_WORLD);
        recvmat(b, bufb, 0, 'b', MPI_COMM_WORLD);
    }
    c = FOX(a, b);
    if(world_rank != 0){
        sendmat(c, bufa, 0, 'c', MPI_COMM_WORLD);
    }
    else{
        for(int i = 0; i < p; ++ i){
            for(int j = 0; j < p; ++ j){
                if(i + j != 0){
                    recvmat(c, bufb, i*p + j, 'c', MPI_COMM_WORLD);
                }
                for(int ii = 0; ii < n; ++ ii){
                    for(int jj = 0; jj < n; ++ jj){
                        (*C)[i*n + ii][j*n + jj] = c[ii][jj];
                    }
                }
            }
        }
    }
    double finish_fox = MPI_Wtime();
    delete []bufa;
    delete []bufb;
    //#define DEBUG
    if(world_rank == 0){
        #ifdef DEBUG
        cout << "A:" << endl;
        A->print();
        cout << "B:" << endl;
        B->print();
        cout << "A*B:" << endl;
        C->print();
        #endif
        printf("fox并行乘法耗时: %f\n", finish_fox - start_fox);
                //串行乘法
        double start =  MPI_Wtime();
        *C = (*A)*(*B);
        double finish = MPI_Wtime();

        #ifdef DEBUG
        printf("串行计算结果 A*B : \n");
        C->print();
        #endif
        printf("串行乘法耗时: %f\n", finish - start);

        //计算加速比
        double rate = (finish - start)/(finish_fox - start_fox);
        printf("并行%d核加速比为: %f\n", world_size, rate);

    }
    MPI_Finalize();
    return 0;
}