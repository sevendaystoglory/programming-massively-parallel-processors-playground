#include <stdio.h>
#include <time.h>

#define N 1 << 20 // 2^28 ~ 268M elements. ~3 GiB for 3 arrays.

// we shall understand thread hierarchy
__global__ void VecAdd(int* A, int* B, int * C){
    int start = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = start; i<N; i+=blockDim.x * gridDim.x){ // for a lower grid dim as imposed by the min, we have a longer running kernel
        C[i] = A[i] + B[i];    
    }
}

void print_array(int* A, int* B, int* C);
void seeCUDAerror();

int main(){
    // initialize some variables in CPU memory

    const int size = N;
    int* A = (int*)malloc(size * sizeof(int)); // allocate memory in heap as really large.
    int* B = (int*)malloc(size * sizeof(int));
    int* C = (int*)malloc(size * sizeof(int));
    for (int i = 0; i<size; ++i){A[i] = rand()%100; B[i] = rand()%100;}

    int *dA, *dB, *dC; // these are pointers to int

    // allocate some CUDA memory
    cudaMalloc(&dA, size * sizeof(int)); // point to the GPU memory, bitch 
    cudaMalloc(&dB, size * sizeof(int)); 
    cudaMalloc(&dC, size * sizeof(int)); 

    // Copy input arrays to GPU
    cudaMemcpy(dA, A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 64; //because 64 cores in one SM
    int sm_count = 14;
    int gridDim = min(32*sm_count , size / threads_per_block + 1); // 
    printf("gridDim: %d\n", gridDim);
    int blockDim = threads_per_block;
    // launch kernel
    clock_t start = clock();
    VecAdd<<<gridDim, blockDim>>>(dA, dB, dC);
    clock_t end = clock();
    seeCUDAerror;
    // Copy result back to host
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);
    print_array(A, B, C);

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nTime taken: %f seconds\n", time_spent);

    return 0;
}

// -------------------------helper functions ----------------------

void print_array(int* A, int* B, int* C){
    printf("A: ");
    for (int i = 0; i<4 ; i++){printf("%d, ", A[i]);};
    printf("\nB: ");
    for (int i = 0; i<4 ; i++){printf("%d, ", B[i]);};
    printf("\nC: ");
    for (int i = 0; i<4 ; i++){printf("%d, ", C[i]);};
}

void seeCUDAerror(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    };
    cudaDeviceSynchronize();   
}