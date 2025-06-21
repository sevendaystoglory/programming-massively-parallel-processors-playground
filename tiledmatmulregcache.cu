//implements tiled matmul with register cache

#include <stdio.h>

#define N 1<<10
#define M 1<<10
#define K 1<<10
#define tile_size 128
#define BK 8


void printMatrices(float* A, float* B, float* C);

__global__ void matmul(float* A, float* B, float* C){
    __shared__ float sA[tile_size*BK];
    __shared__ float sB[BK*tile_size];

    // now think along these lines: We are multiplying two tile_size x BK, and BK x tile_size matrices.
    // we are going to tile these matrices using our micro tiles. we have fixed: BK x BK of these (from the blockDim)
    // register micro block tile is 1 dimensional if you think about it.
    const int micro_tile_size = tile_size / BK;
    const int micro_block_size = micro_tile_size * micro_tile_size;
    float threadResults[micro_block_size] = {0.0}; // register storage to store the results.
    float regA[micro_tile_size] = {0.0};
    float regB[micro_tile_size] = {0.0};

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    // i and j are global indices
    int i = ty + blockDim.y * by; 
    int j = tx + blockDim.x * bx;
    // we start by copying data from A and B to shared memory
    for(int phase = 0; phase < K; phase+=BK){ // iterate over common dimension of the matrix
        // we have lower threads than there are elements in sA. So we must loop. Now we have BK x BK as each micro_tile
        for(int A_offset = 0; A_offset < tile_size; A_offset+=BK){
            int tty = (ty + A_offset); int ttx = tx; //read ttx as 'translated tx' since threads to tile is no longer an onto mapping. We must transform to use.
            sA[tty*BK + ttx] = A[i*K + phase*BK + ttx];
        }
        for(int B_offset = 0; B_offset < tile_size; B_offset+=BK){
            int tty = (ty + B_offset); int ttx = tx;
            sB[tty*BK + ttx] = B[(phase*BK+ty)*N+j]; //store in row major layout in sB as well as sA
        }
        __syncthreads(); //we've successfully populated sA and sB for a block
    
        // if we exlude the 'offset' variable from accessing of A and B above, alternatively we could add that condition here to advance our blocktile along the common dimension
        // like this
        // A += BK;     // move BK columns to right
        // B += BK * N; // move BK rows down

        // calculate per-thread results
        
        for(int a = 0; a < (micro_block_size); a++){ // this loop will iterate over a single micro-tile
            regA[a] = sA[ty*micro_block_size + a];
            regB[a] = sB[tx*micro_block_size + a];
        }
        // we'll now be computing outer product of regA and regB to populate threadResults. 
        for(int a = 0; a < micro_tile_size; a++){ // note: i,j  span a microtile.
            for(int b = 0; b < micro_tile_size; b++){
                threadResults[a*BK + b] = regA[a] * regB[b];
            }
        }
        __syncthreads();

        // write the results to C
        for(int a = 0; a < micro_tile_size; a++){
            for(int b = 0; b < micro_tile_size; b++){
                C[(i + a)*tile_size+ j+b] += threadResults[a*micro_tile_size + b];
            }
        }
    }
    

}

int main(){
    float *hA = (float*)malloc(N*N*sizeof(float));
    float *hB = (float*)malloc(N*N*sizeof(float));
    float *hC = (float*)malloc(M*N*sizeof(float));
    for(int i = 0; i<M*N; i++){hA[i] = rand()%15;}
    for(int i = 0; i<M*N; i++){hB[i] = rand()%15;}
    for(int i = 0; i<M*N; i++){hB[i] = 0;}

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(float)*M*N);
    cudaMalloc(&dB, sizeof(float)*M*N);
    cudaMalloc(&dC, sizeof(float)*M*N);

    cudaMemcpy(dA, hA, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    
    dim3 blockDim(tile_size,tile_size); // each thread is to handle 8x8 elemnts of a micro tile. Each tile will have 16x16 microtiles. Size = 128x128
    dim3 gridDim(M/tile_size +1, N/tile_size + 1);
    matmul<<<gridDim, blockDim>>>(dA, dB, dC);

    
    cudaMemcpy(hC, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    // printMatrices(hA, hB, hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}


void printMatrices(float* A, float* B, float* C){
    printf("\nMatrix A");
    for(int i = 0; i < M; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%f, ", A[i*N +j]);}}
    printf("\nMatrix B");
    for(int i = 0; i < M; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%f, ", B[i*N +j]);}}
    printf("\nMatrix C");
    for(int i = 0; i < M; i++){for(int j = 0; j < N; j++){if(j == 0)printf("\n");printf("%f, ", C[i*N +j]);}}
}