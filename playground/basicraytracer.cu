#include <stdio.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../local_libraries/stb_image_write.h"

#define INF 2e10f
#define NUM_SPHERES 20
#define rnd(x) ((float)rand() / (float)RAND_MAX ) * x
#define DIM 2048

struct Sphere{
    float r,g,b;
    float radius;
    float x,y,z;
    __device__ float hit(float ox, float oy, float *n){ // assumes an orthographic ray tracer. we get INF if we never hit a sphere but get the z value into the frame if we ever do hit a sphere.
        float dx = ox - x;
        float dy = oy - y;
        float d2 =dx*dx + dy*dy;
        if (d2 < radius*radius){ // we've an orthogonal hit!
            float dz = sqrtf(radius*radius - d2);
            *n = dz / sqrtf(radius*radius);
            return dz + z;
        }
        return -INF;
    }
};

__constant__ Sphere dS[NUM_SPHERES]; // pointer to a sphere object, this will point to our device

__global__ void raytrace(unsigned char * arr){ // each of my threads is generating one pixel using raytracing
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    float ox = x;
    float oy = y;
    if(x < DIM && y < DIM){ // check for overflows 
        int idx = x + y * blockDim.x * gridDim.x; // row major indexing to access arr
        float r = 0, g = 0, b = 0;
        float maxZ = -INF;
        for(int i = 0; i < NUM_SPHERES; i++){ // for each sphere. each of my threads in a warp is demanding the sphere essentials in the same order. If we define the sphere array at the file scope with a constant, and change the cudaMemcpy to cudaMemcpyToSymbol, we would be using the consant memory
            float n;
            float t = dS[i].hit(ox, oy, &n);
            if (t > maxZ){ // check if the ray projected orhogonally from ox,oy interects with it
                float fscale = n; // this value is higher the closer the ray is to the centre of the sphere.
                r = dS[i].r * fscale;
                g = dS[i].g * fscale;
                b = dS[i].b * fscale;
                maxZ = t;
            }
        }
        arr[idx * 4 + 0] = (int)(r*255);
        arr[idx * 4 + 1] = (int)(g*255);
        arr[idx * 4 + 2] = (int)(b*255);
        arr[idx * 4 + 3] = (int)255; // alpha values for all spheres is 255
    }
}

int main(){
    unsigned char *dev_bitmap; 
    // allocate memory on the GPU for the output bitmap
    cudaMalloc( (void**)&dev_bitmap, DIM * DIM * sizeof(unsigned char) * 4); 
    // allocate memory for the Sphere dataset
    
    cudaMalloc( (void**)&dS, sizeof(Sphere) * NUM_SPHERES );
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * NUM_SPHERES);
    for (int i=0; i<NUM_SPHERES; i++) {
    temp_s[i].r = rnd( 1.0f );
    temp_s[i].g = rnd( 1.0f );
    temp_s[i].b = rnd( 1.0f );
    temp_s[i].x = rnd( DIM );
    temp_s[i].y = rnd( DIM );
    temp_s[i].z = rnd( DIM );
    temp_s[i].radius = rnd( 400.0f ) + 20;
    };

    cudaMemcpyToSymbol(dS, temp_s, sizeof(Sphere) * NUM_SPHERES);
    free(temp_s);

    dim3 gridDim(DIM / 16, DIM / 16);
    dim3 blockDim(16, 16);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    raytrace<<<gridDim, blockDim>>>(dev_bitmap); // my image is being tiled by blocks
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("\nElapsed Time: %f\n", elapsed_time);

    cudaDeviceSynchronize();
    unsigned char *host_bitmap = (unsigned char *)malloc(DIM * DIM * sizeof(unsigned char) * 4);
    cudaMemcpy(host_bitmap, dev_bitmap, DIM * DIM * sizeof(unsigned char)*4, cudaMemcpyDeviceToHost);
    stbi_write_png("ray_trace.png", DIM, DIM, 4, host_bitmap, DIM * 4);
    free(host_bitmap);
    cudaFree(dev_bitmap);
    cudaFree(dS);
}

