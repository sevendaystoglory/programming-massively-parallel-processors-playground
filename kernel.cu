#include <stdio.h>

// we shall understand thread hierarchy
__global__ void VecAdd(int* A, int* B, int * C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(){
    // initialize some variables in CPU memory
    int N = 4; // number of threads
    int A[3] = {1,2,3};
    int B[3] = {10,12,10};
    int C[3];

    int *dA, *dB, *dC; // these are pointers to int

    int size = N*sizeof(int);

    // allocate some CUDA memory
    cudaMalloc(&dA, size); // point to the GPU memory, bitch 
    cudaMalloc(&dB, size); 
    cudaMalloc(&dC, size); 

    // Copy input arrays to GPU
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    // launch kernel
    VecAdd<<<1, N>>>(dA, dB, dC);

    // Copy result back to host
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i<4 ; i++){printf("%d, ", C[i]);};
    return 0;
}