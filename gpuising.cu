#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>

#define SPIN_UP 0
#define SPIN_DOWN 1

#define FLIP_SPIN(s) ((s) == (0) ? (1) : (0))

#define LATICE_SIZE 60
#define W0 0.5
#define C 0.5

#define MAX_MTS 10

#define TRUE 1
#define FALSE 0

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define PRINT_LATICE(LATICE) do {           \
    printf("[");                            \
    for (int i = 0; i < LATICE_SIZE; i++) { \
        (i!=0) ?                            \
            printf(",%d", LATICE[i])        \
          : printf("%d", LATICE[i]) ;       \
    }                                       \
    printf("]\n"); } while (0)

#define SWAP(A, B, C) do {  \
    C = A;                  \
    A = B;                  \
    B = C; } while (0)


__global__ void generate_kernel(curandStateMtgp32 *state)
{
    // Change short to char ?
    short * LATICE             = (short *)malloc(LATICE_SIZE*sizeof(short));
    short * NEXT_STEP_LATICE   = (short *)malloc(LATICE_SIZE*sizeof(short));
    short * SWAP               = NULL;

    long  latice_update_counter = 0;
    short is_latice_updated, is_ferromagnetic = FALSE;
    int first_i, last_i         = 0;
    long monte_carlo_steps      = 0;

    // Initialize LATICE as antiferromagnet [1,0,1,...]
    for (int i = 0; i < LATICE_SIZE; i++) {
        LATICE[i] = (i&1);
    }

    while ( monte_carlo_steps < MAX_MTS ) {
        if ( !is_latice_updated ) {
            // If latice is in ferromagnetic state simulation can stop
            is_ferromagnetic = TRUE;
            for (int i = 0; i < LATICE_SIZE; i++) {
                if (LATICE[0] != LATICE[i]) {
                    is_ferromagnetic = FALSE;
                }
            }
            if (is_ferromagnetic) {
                break;
            }
        }

        // bondDensity(LATICE)
        // miu, sigma via blockId or parameter?
        // generator per block / per watek / host api
        first_i = (int)(LATICE_SIZE * curand_uniform(&state[blockIdx.x]));
        last_i  = (int)(first_i + (C * LATICE_SIZE));
        is_latice_updated = FALSE; // ?

        for (int i = 0; i < LATICE_SIZE; i++) {
            if (first_i <= i && i <= last_i) {
                
                int left  = (i-1) % LATICE_SIZE;
                int right = (i+1) % LATICE_SIZE;

                if ( LATICE[left] == LATICE[right] && LATICE[i] != LATICE[left] ) {
                    NEXT_STEP_LATICE[i] = LATICE[left];
                    latice_update_counter++; is_latice_updated = TRUE;
                } else if ( W0 > curand_uniform(&state[blockIdx.x]) ) {
                    NEXT_STEP_LATICE[i] = FLIP_SPIN(LATICE[i]);
                    latice_update_counter++; is_latice_updated = TRUE;
                }
            } else {
                if (LATICE[i] != NEXT_STEP_LATICE[i]) {
                    latice_update_counter++; is_latice_updated = TRUE;
                }
                NEXT_STEP_LATICE[i] = LATICE[i];
            }
        }
        monte_carlo_steps = (int)(latice_update_counter / LATICE_SIZE);
        SWAP(LATICE, NEXT_STEP_LATICE, SWAP);
    }
    PRINT_LATICE(LATICE);
    printf("%d\n", monte_carlo_steps);
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