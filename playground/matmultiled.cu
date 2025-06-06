#include <stdio.h>
#include <time.h>

#define N (1<<10)
#define tile_size 16

void printMatrices(int A[N][N], int B[N][N], int C[N][N]);

__global__ void matMulTiled(int *A, int *B, int*C){

    // allocate some memory
    __shared__ int A_tile[tile_size][tile_size];
    __shared__ int B_tile[tile_size][tile_size];

    // these are the row_num and col_num of the matrix C
    int tx = threadIdx.x; int bx = blockIdx.x;
    int ty = threadIdx.y; int by = blockIdx.y;
    // working on C[i][j]
    int i = ty + blockDim.y * by;
    int j = tx + blockDim.x * bx;
    if(i < N && j < N){
    int value = 0;
    for(int phase = 0; phase < N / tile_size; phase++){ 
            // start loading to shared memory: indexed by tx, ty
            A_tile[ty][tx] = A[i*N + phase*tile_size + tx];
            B_tile[ty][tx] = B[(phase*tile_size+ty)*N+j];
            __syncthreads(); // wait for all threads in a block

            for(int k=0; k < tile_size; k++){
                value += A_tile[ty][k] * B_tile[k][tx];
            }
            __syncthreads(); // wait for all threads in a block
        }
        C[i*N+j] = value;
    }
}

int main(){
    int *hA, *hB, *hC;
    // hA = (int*)malloc(sizeof(int)*N*N);
    // hB = (int*)malloc(sizeof(int)*N*N);
    // hC = (int*)malloc(sizeof(int)*N*N);
    // we can switch to pinned memory. this optimizes memory accesses to 
    cudaMallocHost(&hA, sizeof(int)*N*N);
    cudaMallocHost(&hB, sizeof(int)*N*N);
    cudaMallocHost(&hC, sizeof(int)*N*N);
    for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){hA[i*N+j] = rand()%13;}}
    for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){hB[i*N+j] = rand()%13;}}

    void *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(int)*N*N);
    cudaMalloc(&dB, sizeof(int)*N*N);
    cudaMalloc(&dC, sizeof(int)*N*N);
    
    cudaMemcpy(dA, hA, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    
    dim3 blockDim(tile_size, tile_size);
    dim3 gridDim(N / tile_size + 1, N / tile_size + 1);
    clock_t start = clock();
    matMulTiled<<<gridDim,  blockDim>>>(dA, dB, dC);
    clock_t end = clock();
    printf("\nTime Elapsed: %fs\n", (float)(end-start)/CLOCKS_PER_SEC);
    cudaMemcpy(hC, dC, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    // int A[N][N], B[N][N], C[N][N];
    // for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){A[i][j] = hA[i*N+j];}}
    // for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){B[i][j] = hB[i*N+j];}}
    // for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){C[i][j] = hC[i*N+j];}}
    // printMatrices(A, B, C);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void printMatrices(int A[N][N], int B[N][N], int C[N][N]){
       printf("\nMatrix A");
    for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", A[i][j]);}}
    printf("\nMatrix B");
    for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", B[i][j]);}}
    printf("\nMatrix C");
    for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", C[i][j]);}}
}