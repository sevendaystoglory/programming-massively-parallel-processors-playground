#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

#define CEIL_DIV(M,N)  (((M) + (N) - 1) / (N))

constexpr int M_GLOBAL = 1 << 10;   // rows of C and A
constexpr int N_GLOBAL = 1 << 10;   // cols of C and B
constexpr int K_GLOBAL = 1 << 10;   // common dim

constexpr int BM = 128;     // rows  computed by one thread-block
constexpr int BN = 128;     // columns computed by one thread-block
constexpr int BK = 8;       // K depth for each iteration
constexpr int TM = 8;       // rows of C each thread produces
constexpr int TN = 8;       // cols of C each thread produces

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t status = (call);                                  \
        if (status != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                    cudaGetErrorString(status), __FILE__, __LINE__);  \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

#define CUDA_KERNEL_CHECK()  CUDA_CHECK(cudaGetLastError())

template<int BM_, int BN_, int BK_, int TM_, int TN_>
__global__ void __launch_bounds__((BM_ * BN_) / (TM_ * TN_), 1)
sgemm2DBlocktiling(int M, int N, int K,
                   float alpha,
                   const float * __restrict__ A,
                   const float * __restrict__ B,
                   float beta,
                   float *C)
{
    // ---------------- block identification ----------------
    const uint cRow = blockIdx.y;   // which BM-high tile of C
    const uint cCol = blockIdx.x;   // which BN-wide  tile of C

    // ---------------- thread identification ---------------
    constexpr uint totalResultsBlock   = BM_ * BN_;
    constexpr uint resultsPerThread    = TM_ * TN_;
    constexpr uint threadsPerBlock     = totalResultsBlock / resultsPerThread;

    // 1-D threadIdx.x → 2-D (threadRow, threadCol) inside the block-tile
    const uint threadCol = threadIdx.x % (BN_ / TN_);
    const uint threadRow = threadIdx.x / (BN_ / TN_);

    // ---------------- shared memory -----------------------
    extern __shared__ float smem[];
    float *As = smem;                         // size BM_ × BK_
    float *Bs = As + BM_ * BK_;               // size BK_ × BN_

    // ---------------- handy constants --------------------
    const float *Abase = A + cRow * BM_ * K;      // start of this A-tile
    const float *Bbase = B + cCol * BN_;          // start of this B-tile
    float       *Cbase = C + cRow * BM_ * N
                           + cCol * BN_;          // start of this C-tile

    // ---------------- indices each thread loads ----------
    const uint innerRowA = threadIdx.x / BK_;
    const uint innerColA = threadIdx.x % BK_;
    const uint strideA   = threadsPerBlock / BK_;

    const uint innerRowB = threadIdx.x / BN_;
    const uint innerColB = threadIdx.x % BN_;
    const uint strideB   = threadsPerBlock / BN_;

    // ---------------- registers --------------------------
    float threadResults[TM_ * TN_] = {0.f};
    float regM[TM_];
    float regN[TN_];

    // ---------------- loop over the K dimension ----------
    for (uint bk = 0; bk < static_cast<uint>(K); bk += BK_) {

        // ---- load A sub-tile ------------------------------------------------
        for (uint r = innerRowA; r < BM_; r += strideA)
            As[r * BK_ + innerColA] = Abase[(r * K) + bk + innerColA];

        // ---- load B sub-tile ------------------------------------------------
        for (uint r = innerRowB; r < BK_; r += strideB)
            Bs[r * BN_ + innerColB] = Bbase[(bk + r) * N + innerColB];

        __syncthreads();

        // ---- compute outer product ----------------------------------------
        #pragma unroll
        for (uint dot = 0; dot < BK_; ++dot) {

            #pragma unroll
            for (uint i = 0; i < TM_; ++i)
                regM[i] = As[(threadRow * TM_ + i) * BK_ + dot];

            #pragma unroll
            for (uint j = 0; j < TN_; ++j)
                regN[j] = Bs[dot * BN_ + threadCol * TN_ + j];

            #pragma unroll
            for (uint i = 0; i < TM_; ++i)
                #pragma unroll
                for (uint j = 0; j < TN_; ++j)
                    threadResults[i * TN_ + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    // ---------------- store the block-tile back to GMEM ---------------------
    #pragma unroll
    for (uint i = 0; i < TM_; ++i)
        #pragma unroll
        for (uint j = 0; j < TN_; ++j) {
            uint row = threadRow * TM_ + i;
            uint col = threadCol * TN_ + j;
            if (row < static_cast<uint>(M) && col < static_cast<uint>(N)) {
                Cbase[row * N + col] =
                    alpha * threadResults[i * TN_ + j] +
                    beta  * Cbase[row * N + col];
            }
        }
}

// -----------------------------------------------------------------------------
// main() – allocates matrices, runs the kernel once, prints timing & a sample
// -----------------------------------------------------------------------------
int main()
{
    // ------------- host allocation -----------------------------------------
    const size_t bytesA = size_t(M_GLOBAL) * K_GLOBAL * sizeof(float);
    const size_t bytesB = size_t(K_GLOBAL) * N_GLOBAL * sizeof(float);
    const size_t bytesC = size_t(M_GLOBAL) * N_GLOBAL * sizeof(float);

    float *hA = static_cast<float*>(malloc(bytesA));
    float *hB = static_cast<float*>(malloc(bytesB));
    float *hC = static_cast<float*>(malloc(bytesC));

    // ------------- initialise A & B with random data -----------------------
    for (size_t i = 0; i < (M_GLOBAL * K_GLOBAL); ++i) hA[i] = rand() / float(RAND_MAX);
    for (size_t i = 0; i < (K_GLOBAL * N_GLOBAL); ++i) hB[i] = rand() / float(RAND_MAX);
    for (size_t i = 0; i < (K_GLOBAL * N_GLOBAL); ++i) hC[i] = 0;
    

    // ------------- device allocation --------------------------------------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, bytesC, cudaMemcpyHostToDevice));

    // ------------- launch configuration -----------------------------------
    dim3 blockDim((BM * BN) / (TM * TN));   // 256 threads
    dim3 gridDim(CEIL_DIV(N_GLOBAL, BN),
                 CEIL_DIV(M_GLOBAL, BM));

    const size_t shmemBytes = (BM * BK + BK * BN) * sizeof(float);

    // ------------- run once & time it -------------------------------------
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim, shmemBytes>>>
        (M_GLOBAL, N_GLOBAL, K_GLOBAL,
         1.0f, dA, dB, 0.0f, dC);

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("SGEMM  %dx%d × %dx%d  finished in %.3f ms\n",
           M_GLOBAL, K_GLOBAL, K_GLOBAL, N_GLOBAL, ms);

    // ------------- read back one element so we know it worked --------------
    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(float), cudaMemcpyDeviceToHost));
    printf("C[0,0] = %f\n", hC[0]);

    // ------------- tidy up -------------------------------------------------
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
