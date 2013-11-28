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


#define SPIN_UP 0
#define SPIN_DOWN 1

#define FLIP_SPIN(s) ((s) == (0) ? (1) : (0))

#define LATICE_SIZE 51
#define W0 0.5
#define C 0.5

template <class T> void swap ( T& a, T& b )
{
  T c(a); a=b; b=c;
}

__global__ void generate_kernel(curandStateMtgp32 *state)
{
    short * LATICE           = (short *)malloc(LATICE_SIZE*sizeof(short));
    short * NEXT_STEP_LATICE = (short *)malloc(LATICE_SIZE*sizeof(short));
    short * swap = NULL;

    for (int i = 0; i < LATICE_SIZE; i++) {
        LATICE[i] = (i&1);
    }

    for (int t = 0; t < 1000; t++) {
        // bondDensity(LATICE)
        // miu, sigma via blockId or parameter?
        int first_i = (int)(LATICE_SIZE * curand_uniform(&state[blockIdx.x]));
        int last_i = (int)(first_i + (C * LATICE_SIZE));

        for (int i = 0; i < LATICE_SIZE; i++) {
            if (first_i <= i && i <= last_i) {
                int left  = (i-1) % LATICE_SIZE;
                int right = (i+1) % LATICE_SIZE;
                if ( LATICE[left] == LATICE[right] ) {
                    NEXT_STEP_LATICE[i] = LATICE[left];
                    // montecarlosteps++
                } else if ( W0 > curand_uniform(&state[blockIdx.x]) ) {
                    NEXT_STEP_LATICE[i] = FLIP_SPIN(LATICE[i]); 
                    // montecarlosteps++
                } // else flag for is_complete?
            } else {
                NEXT_STEP_LATICE[i] = LATICE[i];
            }
        }
        printf("[");
        for (int i = 0; i < LATICE_SIZE; i++) {
            if (i!=0){
                printf(",%d", NEXT_STEP_LATICE[i]);
            } else {
                printf("%d", NEXT_STEP_LATICE[i]);
            }
        }
        printf("]\n"); 
        swap = LATICE;
        LATICE = NEXT_STEP_LATICE;
        NEXT_STEP_LATICE = swap;
    }
}

int main(int argc, char *argv[])
{
    curandStateMtgp32 *devMTGPStates;
    mtgp32_kernel_params *devKernelParams;

    /* Allocate space for prng states on device */
    CUDA_CALL(cudaMalloc((void **)&devMTGPStates, 64 * sizeof(curandStateMtgp32)));

    /* Setup MTGP prng states */

    /* Allocate space for MTGP kernel parameters */
    CUDA_CALL(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));

    /* Reformat from predefined parameter sets to kernel format, */
    /* and copy kernel parameters to device memory               */
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213,devKernelParams));

    /* Initialize one state per thread block */
    CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates,mtgp32dc_params_fast_11213, devKernelParams, 64, 1234));

    generate_kernel<<<1, 1>>>(devMTGPStates);

    /* Cleanup */
    CUDA_CALL(cudaFree(devMTGPStates));
    return EXIT_SUCCESS;
}