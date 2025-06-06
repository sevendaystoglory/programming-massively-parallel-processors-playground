#include <stdio.h>

int main(){
    void *arr;
    // *arr = 12;
    printf("%p\n", arr);
    printf("%p\n", &arr);
    cudaMalloc((void**)&arr, sizeof(int)*3); // &arr stays same, and arr points to GPU memory
    printf("%p\n", arr);
    printf("%p\n", &arr);
}