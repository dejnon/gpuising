// Test program to test if your GPGPU is capable of using curand (and kernel-side printf)
// Compile with: nvcc curand.cu -arch=sm_20
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>
 
 
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
 
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
 
__global__ void generate_kernel(curandStateMtgp32 *state)
{
    for(int i = 0; i < 1000; i++) {
        printf("%f, ", curand_normal(&state[blockIdx.x]));
    }
}
 
int main(int argc, char *argv[])
{
    int i;
    curandStateMtgp32 *devMTGPStates;
    mtgp32_kernel_params *devKernelParams;
        
    /* Allocate space for prng states on device */
    CUDA_CALL(cudaMalloc((void **)&devMTGPStates, 64 * 
              sizeof(curandStateMtgp32)));
        
    /* Allocate space for MTGP kernel parameters */
    CUDA_CALL(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));
    
    /* Reformat from predefined parameter sets to kernel format, */
    /* and copy kernel parameters to device memory               */
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));
    
    /* Initialize one state per thread block */
    CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, 
                mtgp32dc_params_fast_11213, devKernelParams, 64, 1234));
        
    /* Generate and use pseudo-random  */
    for(i = 0; i < 1; i++) {
        generate_kernel<<<1, 1>>>(devMTGPStates);
    }

    /* Cleanup */
    CUDA_CALL(cudaFree(devMTGPStates));
    return EXIT_SUCCESS;
}