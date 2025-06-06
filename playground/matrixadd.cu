#include <stdio.h>

#define M (1<<10) // 1k is the size of the square matrix 
#define N (1<<10) // 1k is the size of the square matrix 

__global__ void matAdd(float* A, float* B, float* C){
    int i = threadIdx.y + blockIdx.y * blockDim. y;
    int j = threadIdx.z + blockIdx.z * blockDim. z;
    if(i < M && j < N){
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    float **A;
    A = (float**)malloc(sizeof(float*)*M); // allocate memory for M addresses
    for(int i=0;i<M;i++){A[i] = (float*)malloc(sizeof(float)*N);}
    float **B;
    B = (float**)malloc(sizeof(float*)*M);
    for(int i=0;i<M;i++){B[i] = (float*)malloc(sizeof(float)*N);}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            A[i][j] = rand() % 32;
            B[i][j] = rand() % 32;
        }
    }

    printf("%f, %f ", A[0][0], A[0][1]);

    float* hA = (float*)malloc(sizeof(float)*M*N);
    float* hB = (float*)malloc(sizeof(float)*M*N);
    for(int i = 0; i<M; i++){
        for(int j = 0; j<N; j++){
            int idx = i * N + j;
            hB[idx]=B[i][j];
            hA[idx]=A[i][j];
        }
    }
    float* hC = (float*)malloc(sizeof(float)*M*N);

    // init some pointers to float - they live in the CPU
    float *dA, *dB, *dC;
    // allocate some memory on GPU
    cudaMalloc(&dA, sizeof(float)*N*M); 
    cudaMalloc(&dB, sizeof(float)*N*M);
    cudaMalloc(&dC, sizeof(float)*N*M);

    cudaMemcpy(dA, hA, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*M*N, cudaMemcpyHostToDevice);

    dim3 blockDim(1, 32, 32);
    dim3 gridDim(1, M/32, N/32);
    matAdd<<<gridDim, blockDim>>>(dA, dB, dC);

    cudaMemcpy(dC, hC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    float **C = (float**)malloc(sizeof(float*)*M);
    for(int i=0;i<M;i++){C[i] = (float*)malloc(sizeof(float)*N);for(int j = 0; j < N; j++){C[i][j] = hC[j+i*M];}}

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}