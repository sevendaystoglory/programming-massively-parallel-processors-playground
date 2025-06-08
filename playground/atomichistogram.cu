#include <stdio.h>
#include <math.h>
#include <time.h>
#define N 100*1024*1024 // 100M element buffer
#define rnd (rand() % 256) // 0 --> 255 


void histo(unsigned char *buffer, unsigned int *hist){
    for(int i = 0; i < N; i++){
        hist[buffer[i]]+=1;
    }
}
__global__ void histo_kernel(unsigned char * buffer, unsigned int *hist){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            atomicAdd(&hist[buffer[idx]], 1);
        }
}

int main(){
    unsigned char *host_buffer = (unsigned char*)malloc(N * sizeof(unsigned char));
    unsigned int host_hist[256] = {0};
    for(int i = 0; i < N; i++){host_buffer[i] = rnd;}

    unsigned char *dev_buffer;
    unsigned int *dev_hist;
    cudaMalloc(&dev_buffer, N * sizeof(unsigned char));
    cudaMemcpy(dev_buffer, host_buffer, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_hist, 256 * sizeof(unsigned int));
    // cudaMemcpy(dev_hist, host_hist, 256, cudaMemcpyHostToDevice);
    cudaMemset(dev_hist, 0, 256 * sizeof(unsigned int));

    dim3 gridDim(N/512 + 1);
    dim3 blockDim(512);

    clock_t start = clock();
    histo(host_buffer, host_hist);
    clock_t end = clock();
    double time_spent = (double)(end-start)/CLOCKS_PER_SEC;
    printf("\nFirst Few CPU Counts: %d, %d, %d \n", host_hist[0], host_hist[1], host_hist[2]);
    printf("Time for CPU %fs:", time_spent);

    start = clock();
    histo_kernel<<<gridDim, blockDim>>>(dev_buffer, dev_hist);
    cudaDeviceSynchronize();  // Wait for kernel to finish
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    unsigned int host_hist_gpu[256];
    cudaMemcpy(host_hist_gpu, dev_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("\nFirst Few GPU Counts: %d, %d, %d \n", host_hist_gpu[0], host_hist_gpu[1], host_hist_gpu[2]);
    printf("Time for GPU %fs:\n", time_spent);

}