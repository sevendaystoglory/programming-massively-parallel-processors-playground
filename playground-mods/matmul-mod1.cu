#include <stdio.h>
#include <time.h>

// This modification does coalesced memory acccesses

// we are mutliplying two NxN matrices ; C = A X B
#define N (1<<14) 

// Row Major conversion: A[i][j] --> A[i * N + j]
// Column Major conversion: B[i][j] --> B[j * M + i]

void printMatrices(int **A, int **B, int **C);

__global__ void matmul(int *A, int *B, int *C){
    // A is Row major, B is Column major
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = i*N + j;
    if(i < N && j < N){
        C[idx] = 0;
            for(int k=0; k < N; k++){
                C[idx] += A[i*N+k] * B[j*N + k]; // C is formed in a row major format
            }
    }
}

int main(){
    int **A, **B;

    A = (int**)malloc(sizeof(int*)*N);
    for(int i = 0; i < N; i++){A[i] = (int*)malloc(sizeof(int)*N); for(int j = 0; j<N;j++){A[i][j] = rand()%10;}}
    B = (int**)malloc(sizeof(int*)*N);
    for(int i = 0; i < N; i++){B[i] = (int*)malloc(sizeof(int)*N); for(int j = 0; j<N;j++){B[i][j] = rand()%10;}}

    int *hA, *hB, *hC;
    hA = (int*)malloc(sizeof(int)*N*N);
    hB = (int*)malloc(sizeof(int)*N*N);
    hC = (int*)malloc(sizeof(int)*N*N);
    for(int i = 0; i<N; i++){for(int j = 0; j<N; j++){hA[i*N+j] = A[i][j];}}; // row major
    for(int i = 0; i<N; i++){for(int j = 0; j<N; j++){hB[j*N+i] = B[i][j];}}; // column major

    int *dA, *dB, *dC;

    cudaMalloc(&dA, sizeof(int)*N*N);
    cudaMalloc(&dB, sizeof(int)*N*N);
    cudaMalloc(&dC, sizeof(int)*N*N);

    cudaMemcpy(dA, hA, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim(N/16 + 1, N/16 + 1);
    clock_t start = clock();
    matmul<<<gridDim, blockDim>>>(dA, dB, dC);
    clock_t end = clock();
    printf("\nTime Elapsed: %f\n", (double)(end-start) / CLOCKS_PER_SEC);
    cudaMemcpy(hC, dC, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    int **C;
    C = (int**)malloc(sizeof(int*)*N);
    for(int i = 0; i < N; i++){C[i] = (int*)malloc(sizeof(int)*N); for(int j = 0; j<N;j++){C[i][j] = hC[i*N+j];}}

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // printMatrices(A, B, C);
 
}

void printMatrices(int **A, int **B, int **C){
       printf("\nMatrix A");
    for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", A[i][j]);}}
    printf("\nMatrix B");
    for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", B[i][j]);}}
    printf("\nMatrix C");
    for(int i = 0; i < N; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%d, ", C[i][j]);}}

}