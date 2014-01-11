#include <iostream>
#include <algorithm>    // std::swap
#include <math.h>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <cctype>      // old <ctype.h>
#include <sys/types.h>
#include <time.h>
#include <fstream>
#include <unistd.h>
#include <ios>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <iostream>
#include <string>
#include <fstream>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define SPIN_UP     0
#define SPIN_DOWN   1

#define TRUE        1
#define FALSE       0

#define FLIP_SPIN(s) ((s) == (0) ? (1) : (0))

#define LATICE_SIZE 64
#define MAX_RNG_STATES 200

#define MAX_MCS     10000
#define W0_START    0.0 
#define W0_END      1.0 
#define W0_SIZE     10  
#define MIU_START   0.0 
#define MIU_END     1.0 
#define MIU_SIZE    20 
#define SIGMA_START 0.0 
#define SIGMA_END   1.0 
#define SIGMA_SIZE  20

#define X blockIdx.x
#define Y blockIdx.y
#define Z blockIdx.z
#define MAX_X MIU_SIZE
#define MAX_Y SIGMA_SIZE
#define MAX_Z W0_SIZE

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}

#define CURAND_CALL(ans) { curandAssert((ans), __FILE__, __LINE__); }
const char* curandGetErrorString(curandStatus_t status) {
    switch(status) {
        case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";    }
    return "unknown error";
}
inline void curandAssert(curandStatus_t code, char *file, int line, bool abort=true) {
   if (code != CURAND_STATUS_SUCCESS) {
    fprintf(stderr,"CUDAassert: %s %s %d\n", curandGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}

#define SETUP_PRNG ({                          \
    CUDA_CALL(cudaMalloc(                      \
        (void **)&devMTGPStates,               \
        PRNG_STATES * sizeof(curandStateMtgp32)\
    ));                                        \
    CUDA_CALL(cudaMalloc(                      \
        (void**)&devKernelParams,              \
        sizeof(mtgp32_kernel_params)           \
    ));                                        \
    CURAND_CALL(curandMakeMTGP32Constants(     \
        mtgp32dc_params_fast_11213,            \
        devKernelParams                        \
    ));                                        \
    for(int states=0; states<PRNG_STATES; states+=MAX_RNG_STATES){ \
        CURAND_CALL(curandMakeMTGP32KernelState(   \
            devMTGPStates + states,                \
            mtgp32dc_params_fast_11213,            \
            devKernelParams,                       \
            min((int)PRNG_STATES-states, MAX_RNG_STATES),\
            PRNG_SEED                              \
        ));                                        \
    }                                              \
})

#define DISPLAY_MC_RESULTS ({                               \
    printf("C_MEAN   C_SIGMA  W_0      RHO      MCS\n");    \
    for (int i = 0; i < BLOCKS_NEEDED; i++){                \
        int x = (int)(i % MAX_X);                           \
        int y = (int)(( i / MAX_X ) % MAX_Y);               \
        int z = (int)(i / ( MAX_X * MAX_Y ));               \
        printf(                                             \
            "%.6f %.6f %.6f %.6f %d\n",                     \
            x/(float)MAX_X, y/(float)MAX_Y, z/(float)MAX_Z, \
            HST_BOND_DENSITY[i], HST_MCS_NEEDED[i]          \
        );                                                  \
    }                                                       \
})

#define PRINT_LATICE(LATICE) do {              \
    printf("[");                               \
    for (int i = 0; i < LATICE_SIZE; i++) {    \
        (i!=0) ?                               \
            printf(",%d", LATICE[i])           \
          : printf("%d", LATICE[i]) ;          \
    }                                          \
    printf("]\n");                             \
} while (0)

#define TRIANGLE_DISTRIBUTION(miu, sigma) ({   \
    float start = max(miu-sigma, 0.0);         \
    float end   = min(miu+sigma, 1.0);         \
    float rand = (                            \
        curand_uniform(&state[BLOCK_ID])     \
      + curand_uniform(&state[BLOCK_ID])     \
    ) / 2.0;                                   \
    ((end-start) * rand) + start;              \
})

#define BOND_DENSITY(LATICE) ({                \
    int sum = 0;                               \
    for (int i = 0; i < LATICE_SIZE; i++) {    \
        int next = (i+1) % LATICE_SIZE;        \
        sum+=2*abs(LATICE[i]-LATICE[next]);    \
    }                                          \
    ((float)sum / (float)(2*LATICE_SIZE));     \
})

#define BLOCK_ID ({                             \
    X + Y * MAX_X + Z * MAX_X * MAX_Y;          \
})

#define SWAP(A, B, C) do {                     \
    C = A;                                     \
    A = B;                                     \
    B = C;                                     \
} while (0)

__global__ void generate_kernel(
    curandStateMtgp32 *state,
    int * DEV_MCS_NEEDED,
    float * DEV_BOND_DENSITY
) {
    __shared__ unsigned short LATICE_1[LATICE_SIZE];
    __shared__ unsigned short LATICE_2[LATICE_SIZE];
    __shared__ unsigned short first_i, last_i;
    __shared__ unsigned long long int latice_update_counter;
    __shared__ unsigned long monte_carlo_steps;
    __shared__ float W0;

    __shared__ unsigned short * LATICE;
    __shared__ unsigned short * NEXT_STEP_LATICE;
    __shared__ unsigned short * SWAP;

    if (threadIdx.x == 0) {
        LATICE = LATICE_1;
        NEXT_STEP_LATICE = LATICE_2;
        SWAP = NULL;
        latice_update_counter=0; 
        monte_carlo_steps=0;
        W0 = Z/(float)MAX_Z;
    }
    __syncthreads();
    // Initialize as antiferromagnetic
    NEXT_STEP_LATICE[threadIdx.x] = threadIdx.x&1;
    while (monte_carlo_steps < MAX_MCS) {
        __syncthreads();
        if (threadIdx.x == 0) {
            SWAP(LATICE, NEXT_STEP_LATICE, SWAP);
            // @todo reduction
            if ( BOND_DENSITY(LATICE) == 0.0 ) {
                // If latice is in ferromagnetic state, simulation can stop
                monte_carlo_steps = MAX_MCS;
                break;
            }
            float C = TRIANGLE_DISTRIBUTION((X / (float)MAX_X), (Y / (float)MAX_Y));
            first_i = (int)(LATICE_SIZE * curand_uniform(&state[BLOCK_ID]));
            last_i  = (int)(first_i + (C * LATICE_SIZE));
            monte_carlo_steps = (int)(latice_update_counter / LATICE_SIZE);
        }
        __syncthreads();
        NEXT_STEP_LATICE[threadIdx.x] = LATICE[threadIdx.x];
        if (first_i <= threadIdx.x && threadIdx.x <= last_i) {
            short left  = (threadIdx.x-1) % LATICE_SIZE;
            short right = (threadIdx.x+1) % LATICE_SIZE;
            // Neighbours are the same
            if ( LATICE[left] == LATICE[right] ) {
                NEXT_STEP_LATICE[threadIdx.x] = LATICE[left];
            }
            // Otherwise randomly flip the spin
            else if ( W0 > curand_uniform(&state[BLOCK_ID])) {
                NEXT_STEP_LATICE[threadIdx.x] = FLIP_SPIN(LATICE[threadIdx.x]);
            }
            atomicAdd(&latice_update_counter,1);
        }
    }
    if (threadIdx.x == 0) {
        DEV_BOND_DENSITY[BLOCK_ID] = BOND_DENSITY(LATICE);
        DEV_MCS_NEEDED[BLOCK_ID] = (int)(latice_update_counter / LATICE_SIZE);
    }
}


int main(int argc, char *argv[])    
{
    clock_t begin, end;

    // Blocks iterate over W0; Grid iterates over MIU and SIGMA
    // int THREADS_NEEDED = MIU_SIZE * SIGMA_SIZE * W0_SIZE * LATICE_SIZE;
    int BLOCKS_NEEDED  = MIU_SIZE * SIGMA_SIZE * W0_SIZE;
    dim3 blockDim(LATICE_SIZE,1,1);
    dim3 gridDim(MIU_SIZE,SIGMA_SIZE,W0_SIZE);

    long PRNG_STATES        = gridDim.x*gridDim.y*gridDim.z;
    long PRNG_SEED          = time (NULL) * getpid();

    int AVERAGES       = 1;
    
    // Setup configurables
    if (argc >= 2) {
        AVERAGES = atoi(argv[1]);
    }
    if (argc >= 3) {
        PRNG_SEED = atoi(argv[2]);
    }
    printf("%d\n", AVERAGES);

    curandStateMtgp32        *devMTGPStates;
    mtgp32_kernel_params     *devKernelParams;

    SETUP_PRNG;

    int * DEV_MCS_NEEDED; float * DEV_BOND_DENSITY;
    int * HST_MCS_NEEDED; float * HST_BOND_DENSITY;
    HST_MCS_NEEDED = (int *)malloc(BLOCKS_NEEDED * sizeof(int));
    CUDA_CALL(
        cudaMalloc(&DEV_MCS_NEEDED,BLOCKS_NEEDED * sizeof(int))
    );
    HST_BOND_DENSITY = (float *)malloc(BLOCKS_NEEDED * sizeof(float));
    CUDA_CALL(
        cudaMalloc(&DEV_BOND_DENSITY,BLOCKS_NEEDED * sizeof(float))
    ); 

    for (int avgs = 0; avgs < AVERAGES; avgs++) {
        begin = clock();

        generate_kernel<<<gridDim, blockDim>>>(
            devMTGPStates,
            DEV_MCS_NEEDED,
            DEV_BOND_DENSITY
        );

        CUDA_CALL(cudaMemcpy( 
            HST_BOND_DENSITY, 
            DEV_BOND_DENSITY, 
            BLOCKS_NEEDED * sizeof(int), 
            cudaMemcpyDeviceToHost
        ));

        CUDA_CALL(cudaMemcpy( 
            HST_MCS_NEEDED, 
            DEV_MCS_NEEDED, 
            BLOCKS_NEEDED * sizeof(int), 
            cudaMemcpyDeviceToHost
        ));
        end = clock();
        static double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

        int BLCK_SIZE = blockDim.x*blockDim.y*blockDim.z;
        int GRID_SIZE = gridDim.x*gridDim.y*gridDim.z;
        printf("%d\t%d\t%d\t%d\t%lf\n", LATICE_SIZE, MAX_MCS, BLCK_SIZE, GRID_SIZE, time_spent);
        // DISPLAY_MC_RESULTS;
    }

    /* Cleanup */
    CUDA_CALL(cudaFree(devMTGPStates));
    CUDA_CALL(cudaFree(DEV_MCS_NEEDED));
    CUDA_CALL(cudaFree(DEV_BOND_DENSITY));
    free(HST_MCS_NEEDED);
    free(HST_BOND_DENSITY);
    return EXIT_SUCCESS;

}