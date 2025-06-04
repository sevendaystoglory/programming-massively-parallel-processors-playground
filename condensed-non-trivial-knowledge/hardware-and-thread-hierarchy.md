We start by first learning about the GPGPU architecture. Refer to this amazing series on YT - https://www.youtube.com/watch?v=1Goq8Yc3dfo

The standard consumer GPU, the GTX 1650ti for example has the following hardware components.
1. 14 Streaming Multiprocessors (SMs).
2. Each SM has 64 CUDA cores, totalling 896 CUDA cores.
3. 4GiB VRAM (or DRAM) on a 128-bit bus. 
4. SRAM side:
4.1. Each SM has 64KiB of shared memory (L1 cache) - closest to cores.
4.2. 1MiB of L2 cache for the entire GPU.