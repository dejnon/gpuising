#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define N 3

__global__ void add(int *a, int *b, int *c) {
 c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
 }

void arr_print(int *a) {
  for (int i = 0; i < N; i++) {
    printf("%d ", a[i]);
  }
  printf("\n");
}

int main(void) {
 int *a, *b, *c; // host copies of a, b, c
 int *d_a, *d_b, *d_c; // device copies of a, b, c
 int size = sizeof(int) * N;

 // Allocate space for device copies of a, b, c
 cudaMalloc((void **)&d_a, size);
 cudaMalloc((void **)&d_b, size);
 cudaMalloc((void **)&d_c, size);

 // Setup input values
a = (int *)malloc(size);
a[0] = 1; a[1] = 2; a[2] = 3;
b = (int *)malloc(size);
b[0] = 2; b[1] = 2; b[2] = 2;
c = (int *)malloc(size);

// Copy inputs to device
 cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
 cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

arr_print(a);
arr_print(b);
arr_print(c);

 // Launch add() kernel on GPU
 add<<<1,N>>>(d_a, d_b, d_c);

 // Copy result back to host
 cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
 // Cleanup
arr_print(c);

free(a); free(b); free(c);
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
 return 0;
 }
