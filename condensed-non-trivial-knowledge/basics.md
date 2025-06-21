We start by first learning about the GPGPU architecture. Refer to this [amazing series](https://www.youtube.com/watch?v=1Goq8Yc3dfo) on YT.

## Standard GPU Architecture
The standard consumer GPU, the GTX 1650ti for example has the following hardware components.
1. 14 Streaming Multiprocessors (SMs).
2. Each SM has 64 CUDA cores, organized into 4 warp schedulers, each with 16 cores. Totalling 896 CUDA cores across the 14 SMs.
3. 4GiB VRAM (or DRAM) on a 128-bit bus. 
4. SRAM side:
4.1. Each SM has 64KiB of shared memory (L1 cache) - closest to cores.
4.2. 1MiB of L2 cache for the entire GPU.

## Thread Hierarchy
<pre>
Grid
├── Block
│   ├── Thread
</pre>

Thread: A thread is a virtual component. ThreadIdx is a 3-component vector, indexed as (x, y, z). 
Thread Blocks: Each combination of these indices is limited to a 3D (or a 1D or a 2D) block of thread. There is a limit to the number of threads in a block - since they are expected to share the same SM and must share the same memory resources of that core. On current GPUs, each thread block may contain upto *1024* threads. Thread blocks are required to be executed independently.
Grid: Thread blocks can themselves be organized into 1D, 2D or 3D grids. Size of the grid is dictated by our data. 

<<<#(blocks), #(threads-per-block)>>> must be passed to the kernel. Now, a kernel is just a fancy name for a function that runs on the GPU. Within the kernel, we have access to the special threadIdX, blockIdx and the blockDim indices.
- threadIdx: index for a thread in a block
- blockIdx:index for a block in a grid
- blockDim: range of threadIdx

In the matrixadd.cu script, we prefer converting all matrices to a row major layout. This is because CUDA prefers to work with contiguous memory.

---

## Some Questions
(These questions naturally came to me while I was going through the material and thought I should document them here along with the answers I researched. Although, these might become trivial after some time. I hope the do :)

Q. How many blocks are possible? How many grids are possible?
A. The hardware limits the number of blocks to $2^31-1$ ~ 2.3 billion blocks per gridDim (regardless of the layout, i.e. how are these split across the three dimensions). In any scenario we want to use more blocks than that. E.g. when we naively launch a single thread inside a block (the parallelism is now due to SMs not cores!) and we need more indices in our kernel than $2^31-1$ - we take strides of gridDim with our index inside our kernel. Nowhere we ever define the number of grids. 

Q What is the use of multi-dimensional blocks or grids?
A. The limits on the blockDim of 1024 and on the gridDim of $2^31-1$ are imposed cumulatively, regardless of how these numbers multiply acorss dimensions. E.g. dim3 blockDim(32, 32) and dim3 blockDim(1024,1) - each uses the maximum allowed threads per block(1024) in a different thread layout (2D vs 1D).
These layouts are helpful when the very nature of our host object can be naturally represented by the layout, i.e. 1D for array, 2D for matrix, 3D for videos or volumes. 

P.S. Each thread dimension has its own individual limit!
$$
blockDim.x \leq 1024;
blockDim.y \leq 1024;
blockDim.z \leq 64;
$$

Q. Declaring floats explicitly in CUDA (i.e. 3.1415926535897932f vs 3.1415926535897932) and why it is important?
This is best explained using an example:
```C
__device__ float perimeter(float x):{
    return 2 * x * 3.1415826535897932; // BAD!!
}
```
Q. What is CuBLAS, and SGEMM?
A. We could've very easily renamed all our matmul kernels to a fancier - SGEMM, because it very literally stands for single precision general matrix multiplication. 
CuBLAS is just the CUDA implementation for basic linear algebra subprograms. In simple terms this library exposes basic methods such as vector addition and dot product an matrix multiplication as ready-to-use APIs.  

C/C++ implicitly assumes that 3.14.. is a double literal. Even though x is float, the product converts to double and since the return type is float we again truncate the result back to float. These conversions when happening across all threads slow down the computation. Better to write 3.14..f, which prevents unintentionally pulling in double math.

---
### Good Reads
[how fast can we perform as froward pass](https://bounded-regret.ghost.io/how-fast-can-we-perform-a-forward-pass)
[Cuda by Example](https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf)
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)