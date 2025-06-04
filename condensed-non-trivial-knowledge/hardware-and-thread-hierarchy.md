We start by first learning about the GPGPU architecture. Refer to this (amazing series)[https://www.youtube.com/watch?v=1Goq8Yc3dfo] on YT.

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
