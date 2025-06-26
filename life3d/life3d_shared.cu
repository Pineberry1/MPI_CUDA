/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号:SA24011270 
 * 姓名:汤璇
 * 邮箱:txzzz@mail.ustc.edu.cn
 ------------------------------------------------*/

 #include <chrono>
 #include <cstring>
 #include <fstream>
 #include <iostream>
 #include <string>
 
 #define AT(x, y, z) universe[(x) * N * N + (y) * N + z]
 
 using std::cin, std::cout, std::endl;
 using std::ifstream, std::ofstream;
 
 // 存活细胞数
 int population(int N, char *universe)
 {
     int result = 0;
     for (int i = 0; i < N * N * N; i++)
         result += universe[i];
     return result;
 }
 
 // 打印世界状态
 void print_universe(int N, char *universe)
 {
     // 仅在N较小(<= 32)时用于Debug
     if (N > 32)
         return;
     for (int x = 0; x < N; x++)
     {
         for (int y = 0; y < N; y++)
         {
             for (int z = 0; z < N; z++)
             {
                 if (AT(x, y, z))
                     cout << "O ";
                 else
                     cout << "* ";
             }
             cout << endl;
         }
         cout << endl;
     }
     cout << "population: " << population(N, universe) << endl;
 }
 
 __device__ int mod(int x, int n) {
     return (x + n) % n;
 }
 

__global__ void update(int N, char *universe, char* next) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= N || y >= N || z >= N) return;

    extern __shared__ char shared_mem[];

    int shared_size = blockDim.x + 2;
    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;
    int local_z = threadIdx.z + 1;
    
    int global_idx = x * N * N + y * N + z;
    int shared_idx = local_x * shared_size * shared_size + local_y * shared_size + local_z;
    shared_mem[shared_idx] = universe[global_idx];

    // 边界面
    if (threadIdx.x == 0) {
        shared_mem[(local_x - 1) * shared_size * shared_size + local_y * shared_size + local_z] = universe[mod(x - 1, N) * N * N + y * N + z];
    }
    if (threadIdx.x == blockDim.x - 1 || x == N - 1) {
        shared_mem[(local_x + 1) * shared_size * shared_size + local_y * shared_size + local_z] = universe[mod(x + 1, N) * N * N + y * N + z];
    }
    if (threadIdx.y == 0) {
        shared_mem[local_x * shared_size * shared_size + (local_y - 1) * shared_size + local_z] = universe[x * N * N + mod(y - 1, N) * N + z];
    }
    if (threadIdx.y == blockDim.y - 1 || y == N - 1) {
        shared_mem[local_x * shared_size * shared_size + (local_y + 1) * shared_size + local_z] = universe[x * N * N + mod(y + 1, N) * N + z];
    }
    if (threadIdx.z == 0) {
        shared_mem[local_x * shared_size * shared_size + local_y * shared_size + (local_z - 1)] = universe[x * N * N + y * N + mod(z - 1, N)];
    }
    if (threadIdx.z == blockDim.z - 1 || z == N - 1) {
        shared_mem[local_x * shared_size * shared_size + local_y * shared_size + (local_z + 1)] = universe[x * N * N + y * N + mod(z + 1, N)];
    }

    // 边界边
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_mem[(local_x - 1) * shared_size * shared_size + (local_y - 1) * shared_size + local_z] = universe[mod(x - 1, N) * N * N + mod(y - 1, N) * N + z];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        shared_mem[(local_x + 1) * shared_size * shared_size + (local_y + 1) * shared_size + local_z] = universe[mod(x + 1, N) * N * N + mod(y + 1, N) * N + z];
    }
    if (threadIdx.x == 0 && threadIdx.z == 0) {
        shared_mem[(local_x - 1) * shared_size * shared_size + local_y * shared_size + (local_z - 1)] = universe[mod(x - 1, N) * N * N + y * N + mod(z - 1, N)];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.z == blockDim.z - 1) {
        shared_mem[(local_x + 1) * shared_size * shared_size + local_y * shared_size + (local_z + 1)] = universe[mod(x + 1, N) * N * N + y * N + mod(z + 1, N)];
    }
    if (threadIdx.y == 0 && threadIdx.z == 0) {
        shared_mem[local_x * shared_size * shared_size + (local_y - 1) * shared_size + (local_z - 1)] = universe[x * N * N + mod(y - 1, N) * N + mod(z - 1, N)];
    }
    if (threadIdx.y == blockDim.y - 1 && threadIdx.z == blockDim.z - 1) {
        shared_mem[local_x * shared_size * shared_size + (local_y + 1) * shared_size + (local_z + 1)] = universe[x * N * N + mod(y + 1, N) * N + mod(z + 1, N)];
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        shared_mem[(local_x - 1) * shared_size * shared_size + (local_y + 1) * shared_size + local_z] = universe[mod(x - 1, N) * N * N + mod(y + 1, N) * N + z];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        shared_mem[(local_x + 1) * shared_size * shared_size + (local_y - 1) * shared_size + local_z] = universe[mod(x + 1, N) * N * N + mod(y - 1, N) * N + z];
    }
    if (threadIdx.x == 0 && threadIdx.z == blockDim.z - 1) {
        shared_mem[(local_x - 1) * shared_size * shared_size + local_y * shared_size + (local_z + 1)] = universe[mod(x - 1, N) * N * N + y * N + mod(z + 1, N)];
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.z == 0) {
        shared_mem[(local_x + 1) * shared_size * shared_size + local_y * shared_size + (local_z - 1)] = universe[mod(x + 1, N) * N * N + y * N + mod(z - 1, N)];
    }
    if (threadIdx.y == 0 && threadIdx.z == blockDim.z - 1) {
        shared_mem[local_x * shared_size * shared_size + (local_y - 1) * shared_size + (local_z + 1)] = universe[x * N * N + mod(y - 1, N) * N + mod(z + 1, N)];
    }
    if (threadIdx.y == blockDim.y - 1 && threadIdx.z == 0) {
        shared_mem[local_x * shared_size * shared_size + (local_y + 1) * shared_size + (local_z - 1)] = universe[x * N * N + mod(y + 1, N) * N + mod(z - 1, N)];
    }
    // 边界点（只在一个线程处理，省去很多分支判断）
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        for (int corner = 0; corner < 8; ++corner) {
            int dx = (corner & 1) ? blockDim.x : -1;
            int dy = (corner & 2) ? blockDim.y : -1;
            int dz = (corner & 4) ? blockDim.z : -1;

            int neighbor_x = mod(x + dx, N);
            int neighbor_y = mod(y + dy, N);
            int neighbor_z = mod(z + dz, N);

            int global_neighbor_idx = neighbor_x * N * N + neighbor_y * N + neighbor_z;

            int local_neighbor_x = (dx == -1) ? 0 : (blockDim.x + 1);
            int local_neighbor_y = (dy == -1) ? 0 : (blockDim.y + 1);
            int local_neighbor_z = (dz == -1) ? 0 : (blockDim.z + 1);

            int shared_neighbor_idx = local_neighbor_x * shared_size * shared_size + 
                                      local_neighbor_y * shared_size + 
                                      local_neighbor_z;

            shared_mem[shared_neighbor_idx] = universe[global_neighbor_idx];
        }
    }

    __syncthreads();

    int alive = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                alive += shared_mem[(local_x + dx) * shared_size * shared_size + (local_y + dy) * shared_size + (local_z + dz)];
            }
        }
    }

    if (shared_mem[shared_idx] && (alive < 5 || alive > 7)) {
        next[global_idx] = 0;
    } else if (!shared_mem[shared_idx] && alive == 6) {
        next[global_idx] = 1;
    } else {
        next[global_idx] = shared_mem[shared_idx];
    }
}

 // 核心计算代码，将世界向前推进T个时刻
 const int blockSize = 8;
 __host__ void life3d_run(int N, char *universe, int T) {
     dim3 blockDim(blockSize, blockSize, blockSize);
     dim3 gridDim((N + blockSize - 1) / blockSize,
                  (N + blockSize - 1) / blockSize,
                  (N + blockSize - 1) / blockSize);
     size_t size = N * N * N * sizeof(char);
     char *universeInDevice, *next;
     cudaMalloc(&universeInDevice, size);
     cudaMalloc(&next, size);
     cudaMemcpy(universeInDevice, universe, size, cudaMemcpyHostToDevice);
 
     size_t shared_mem_size = (blockSize + 2) * (blockSize + 2) * (blockSize + 2) * sizeof(char);
 
     for (int t = 0; t < T; ++t) {
         update<<<gridDim, blockDim, shared_mem_size>>>(N, universeInDevice, next);
         cudaDeviceSynchronize();
         std::swap(next, universeInDevice);
     }
 
     cudaMemcpy(universe, universeInDevice, size, cudaMemcpyDeviceToHost);
     cudaFree(universeInDevice);
     cudaFree(next);
 }
 
 // 读取输入文件
 void read_file(char *input_file, char *buffer)
 {
     ifstream file(input_file, std::ios::binary | std::ios::ate);
     if (!file.is_open())
     {
         cout << "Error: Could not open file " << input_file << std::endl;
         exit(1);
     }
     std::streamsize file_size = file.tellg();
     file.seekg(0, std::ios::beg);
     if (!file.read(buffer, file_size))
     {
         std::cerr << "Error: Could not read file " << input_file << std::endl;
         exit(1);
     }
     file.close();
 }
 
 // 写入输出文件
 void write_file(char *output_file, char *buffer, int N)
 {
     ofstream file(output_file, std::ios::binary | std::ios::trunc);
     if (!file)
     {
         cout << "Error: Could not open file " << output_file << std::endl;
         exit(1);
     }
     file.write(buffer, N * N * N);
     file.close();
 }
 
 int main(int argc, char **argv)
 {
     // cmd args
     if (argc < 5)
     {
         cout << "usage: ./life3d N T input output" << endl;
         return 1;
     }
     int N = std::stoi(argv[1]);
     int T = std::stoi(argv[2]);
     char *input_file = argv[3];
     char *output_file = argv[4];
 
     char *universe = (char *)malloc(N * N * N);
     read_file(input_file, universe);
 
     int start_pop = population(N, universe);
     auto start_time = std::chrono::high_resolution_clock::now();
     life3d_run(N, universe, T);
     auto end_time = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> duration = end_time - start_time;
     int final_pop = population(N, universe);
     write_file(output_file, universe, N);
 
     cout << "start population: " << start_pop << endl;
     cout << "final population: " << final_pop << endl;
     double time = duration.count();
     cout << "time: " << time << "s" << endl;
     cout << "cell per sec: " << T / time * N * N * N << endl;
 
     free(universe);
     return 0;
 }
 