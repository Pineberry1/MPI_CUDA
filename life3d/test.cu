#include<iostream>
using namespace std;
__device__ 
void solve(){
    float* deviceMemory = NULL;
    int size = 16 * 16 * sizeof(float);
    cudaMalloc((void**)&deviceMemory, size);
    cudaFree(deviceMemory);
}
int main(){

}