#include <stdio.h>

// we are mutliplying two NxN matrices
#define N (1<<2)

// Row Major conversion: A[i][j] --> A[i * N + j]

__global__ void matmul(int *A, int *B, int *C){
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i*N + j;
    if(i < N && j < N){
        C[idx] = 0;
            for(int k=0; k < N; k++){
                C[idx] += A[i*N+k] * B[k*N + j];
            }
    }
}

int main(){
    int **A, **B;

    A = (int**)malloc(sizeof(int*)*N*N);
    for(int i = 0; i < N; i++){*A = (int*)malloc(sizeof(int)*N); for(int j = 0; j<N;j++){A[i][j] = rand()%10;}}
    B = (int**)malloc(sizeof(int*)*N*N);
    for(int i = 0; i < N; i++){*B = (int*)malloc(sizeof(int)*N); for(int j = 0; j<N;j++){B[i][j] = rand()%10;}}

    int *hA, *hB, *hC;
    hA = (int*)malloc(sizeof(int)*N*N);
    hB = (int*)malloc(sizeof(int)*N*N);
    hC = (int*)malloc(sizeof(int)*N*N);
    for(int i = 0; i<N; i++){for(int j = 0; j<N; i++){hA[i*N+j] = A[i][j];}};
    for(int i = 0; i<N; i++){for(int j = 0; j<N; i++){hB[i*N+j] = B[i][j];}};

    int *dA, *dB, *dC;

    cudaMalloc(&dA, sizeof(int)*N*N);
    cudaMalloc(&dB, sizeof(int)*N*N);
    cudaMalloc(&dC, sizeof(int)*N*N);

    cudaMemcpy(dA, hA, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hA, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    
    dim3 blockDim(1, 32, 32);
    dim3 gridDim(1, N/32 + 1, N/32 + 1);
    
    matmul<<<gridDim, blockDim>>>(dA, dB, dC);
    
    // cudaMemcpy(hC, dC, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    // int **C;
    // C = (int**)malloc(sizeof(int*)*N*N);
    // for(int i = 0; i < N; i++){*C = (int*)malloc(sizeof(int)*N); for(int j = 0; j<N;j++){C[i][j] = hC[i*N+j];}}

    // cudaFree(dA);
    // cudaFree(dB);
    // cudaFree(dC);

    // printf("Matrix A");
    // for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", A[i][j]);}}
    // printf("Matrix B");
    // for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", B[i][j]);}}
    // printf("Matrix C");
    // for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", C[i][j]);}}
}