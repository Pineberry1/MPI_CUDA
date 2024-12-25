#include <mpi.h>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <cassert>
using namespace std;
template<class type>
class matrix{
private:
    int row, col;
    type* mat;
public:
    matrix():row(0), col(0), mat(nullptr){}
    matrix(int r, int c): row(r), col(c){
        mat = new (type)[r*c];
    }
    ~matrix(){
        delete[] mat;
        row = 0, col = 0;
    }
    void print(){
        for(int i = 0; i < row; ++ i){
            for(int j = 0; j < col; ++ j){
                cout << mat[i][j] << " ";
            }
            cout << endl;
        }
    }
    int serialize(char* res){
        *(int*)res = row;
        *(((int*)res) + 1) = col;
        type* data = (type*)(((int*)res) + 2);
        for(int i = 0; i < r*c; ++ i){
            data[i] = mat[i];
        }
        return sizeof(int) * 2 + sizeof(type) * r * c;
    }
    static matrix unserialize(char* stream){//unknown type
        int r = *(int*)stream;
        int c = *(((int*)stream) + 1);
        matrix res = new matrix(r, c);//alloc
        type* data = (type*)(((int*)stream) + 2);
        for(int i = 0; i < r; ++ i){
            for(int j = 0; j < c; ++ j){
                res[i][j] = data[i*c + j];
            }
        }
        return res;
    }
    type& operator [][](int x, int y){
        if(x >= 0 && x < row && y >= 0 && y < col){
            return *(mat + x * col + y);
        }
        else
            throw "size overflow";
    }
    matrix& operator = (const matrix<type>& m){//shallow copy
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
                res[i][j] = this[x + i][y + j];
            }
        }
        return *res;
    }
    matrix<type> operator *(const matrix& m){
        matrix<type>* res = new matrix(x.row, y.col);//alloc
        memset(res.mat, 0, sizeof(type) * x.row * y.col);
        for(int i = 0; i < row; ++ i){
            for(int j = 0; j < col; ++ j){
                for(int k = 0; k < x.row; ++ k){
                    res[i][j] += x[i][k] * y[k][j];
                }
            }
        }
        return *res;//maybe leak
    }
    matrix& operator +=(const matrix<type>& m){
        assert(row == m.row && col == m.col);
        
    }
};
const int MAX_ROW = 1e3 + 5;
template<class type>
matrix<type> FOX(matrix<type>&A, matrix<type>& B){
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int p = (int)sqrt(size);
    assert(p * p = size);
    int n = A.row;
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
                    MPI_Send(&len, 1, MPI_INT, i * p + k, (round + 1) * 2, MPI_COMM_WORLD);
                    MPI_Send(send_buf, len, MPI_CHAR, i * p + k, (round + 1) * 2 + 1, MPI_COMM_WORLD);
                }
                else{
                    for(int l = 0; l < len; ++ l){
                        recv_bufa[l] = send_buf[l];
                    }
                }
            }
        }
        else{
            if(i - round < 0){
                recvdest = i * p + i - round  + p;
            }
            else{
                recvdest = i * p + i - round;
            }
            int len;
            MPI_Recv(&len, 1, MPI_INT, recvdest, (round + 1) * 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_bufa, len, MPI_CHAR, i * p + k, (round + 1) * 2 + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        matrix<type> a = matrix<type>::unserialize(recv_bufa);
        matrix<type> c = a*B;
        if(round == p - 1)break;
        senddest = i > 0 ? i - 1: i - 1 + p;
        recvdest = i < p - 1 ? i + 1: i + 1 - p;
        int len = B.serialize(send_buf);
        MPI_Send(&len, 1, MPI_INT, senddest * p + j, 0, MPI_COMM_WORLD);
        MPI_Send(send_buf, len, MPI_CHAR, senddest * sp + j, 1, MPI_COMM_WORLD);
        MPI_Recv(&len, 1, MPI_INT, recvdest * p + j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(recv_bufb, len, MPI_CHAR, recvdest * p + j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        matrix<type> tmp_b = matrix<type>::unserialize(recv_bufb);
        for(int i = 0; i < n; ++ i){
            for(int j = 0; j < n; ++ j){
                B[i][j] = tmp_b[i][j];
            }
        }
    }
    delete []recv_bufa;
    delete []recv_bufb;
    delete []send_buf;
}
const int mat_N = 16;
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == 0){
        matrix<int> A(mat_N, mat_N), B(mat_N, mat_N);
        for(int i = 0; i < mat_N; ++ i){
            for(int j = 0; j < mat_N; ++ j){
                A[i][j] = 1;
                B[i][j] = 1;
            }
        }
    }
    int p = (int)sqrt(world_size);
    assert(p * p = size);
    assert(mat_N % p == 0);
    int n = mat_N / p;
    matrix<int> a, b, c;
    MPI_Barrier(MPI_COMM_WORLD);//start
    double start_fox = MPI_Wtime();
    char* bufa = new char[MAX_ROW * MAX_ROW];
    char* bufb = new char[MAX_ROW * MAX_ROW];
    if(world_rank == 0){
        for(int i = 0; i < p; ++ i){
            for(int j = 0; j < p; ++ j){
                if(i == 0 && j == 0)continue;
                int len;
                a = A.split_copy(i * n, j * n, n, n);
                b = B.split_copy(i * n, j * n, n, n);
                len = a.serialize(bufa);
                MPI_Send(&len, 1, MPI_INT, i * p + j, 'a', MPI_COMM_WORLD);
                MPI_Send(bufa, len, MPI_CHAR, i * sp + j, 'a'+ 2, MPI_COMM_WORLD);
                len = b.serialize(bufb);
                MPI_Send(&len, 1, MPI_INT, i * p + j, 'b', MPI_COMM_WORLD);
                MPI_Send(bufb, len, MPI_CHAR, i * sp + j, 'b'+ 2, MPI_COMM_WORLD);
            }
        }
        a = A.split_copy(0, 0, n, n);
        b = B.split_copy(0, 0, n, n);
    }
    else{
        int len;
        MPI_Recv(&len, 1, MPI_INT, 0, 'a', MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(bufa, len, MPI_INT, 0, 'a' + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        a = matrix<int>::unserialize(bufa);
        MPI_Recv(&len, 1, MPI_INT, 0, 'b', MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(bufb, len, MPI_INT, 0, 'b' + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        b = matrix<int>::unserialize(bufb);
    }
    FOX(a, b);
    delete []bufa;
    delete []bufb;

    MPI_Finalize();
    return 0;
}