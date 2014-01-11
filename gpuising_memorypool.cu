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


// #define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
//     printf("Error at %s:%d\n",__FILE__,__LINE__); \
//     return EXIT_FAILURE;}} while(0)

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)


#define PRINT_ARR(LATICE, SIZE) do {           \
    printf("[");                               \
    for (int i = 0; i < SIZE; i++) {           \
        (i!=0) ?                               \
            printf(",%d", LATICE[i])           \
          : printf("%d", LATICE[i]) ;          \
    }                                          \
    printf("]\n");                             \
} while (0)

#define PRINT_ARR_F(LATICE, SIZE) do {         \
    printf("[");                               \
    for (int i = 0; i < SIZE; i++) {           \
        (i!=0) ?                               \
            printf(",%f", LATICE[i])           \
          : printf("%f", LATICE[i]) ;          \
    }                                          \
    printf("]\n");                             \
} while (0)

#define PRINT_LATICE(LATICE) do {              \
    printf("[");                               \
    for (int i = 0; i < LATICE_SIZE; i++) {    \
        (i!=0) ?                               \
            printf(",%d", LATICE[i])           \
          : printf("%d", LATICE[i]) ;          \
    }                                          \
    printf("]\n");                             \
} while (0)

#define PRINT_SHIFT_LATICE(LATICE, SHIFT) do { \
    printf("[");                               \
    for (int i = SHIFT; i < LATICE_SIZE+SHIFT; i++) {    \
        (i!=0) ?                               \
            printf(",%d", LATICE[i])           \
          : printf("%d", LATICE[i]) ;          \
    }                                          \
    printf("]\n");                             \
} while (0)

#define BOND_DENSITY(LATICE) ({                \
    int sum = 0;                               \
    for (int i = THREAD_LATICE_INDEX; i < LATICE_SIZE+THREAD_LATICE_INDEX; i++) { \
        int next = (i+1) % LATICE_SIZE;        \
        sum+=2*abs(LATICE[i]-LATICE[next]);    \
    }                                          \
    ((float)sum / (float)(2*LATICE_SIZE));     \
})

#define IS_FERROMAGNETIC(LATICE) ({            \
    short is_ferromagnetic = TRUE;             \
    for (int i = THREAD_LATICE_INDEX; i < LATICE_SIZE+THREAD_LATICE_INDEX; i++) {    \
        if (LATICE[THREAD_LATICE_INDEX] != LATICE[i]) {          \
            is_ferromagnetic = FALSE;          \
            break;                             \
        }                                      \
    }                                          \
    is_ferromagnetic;                          \
})

#define TRIANGLE_DISTRIBUTION(miu, sigma) ({   \
    float start = max(miu-sigma, 0.0);         \
    float end   = min(miu+sigma, 1.0);         \
    float rand = (                            \
        curand_uniform(&state[BLOCK_ID])     \
      + curand_uniform(&state[BLOCK_ID])     \
    ) / 2.0;                                   \
    ((end-start) * rand) + start;              \
})

#define SETUP_PRNG ({                               \
    CUDA_CALL(cudaMalloc(                           \
        (void **)&devMTGPStates,                    \
        PRNG_STATES * sizeof(curandStateMtgp32)     \
    ));                                             \
    CUDA_CALL(cudaMalloc(                           \
        (void**)&devKernelParams,                   \
        sizeof(mtgp32_kernel_params)                \
    ));                                             \
    CURAND_CALL(curandMakeMTGP32Constants(          \
        mtgp32dc_params_fast_11213,                 \
        devKernelParams                             \
    ));                                             \
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


// // [x,y,z] = [miu, sigma, w0] = [blockIdx.x, blockIdx.y, threadIdx.x]
#define X blockIdx.x
#define Y blockIdx.y
#define Z threadIdx.x
#define MAX_X MIU_SIZE
#define MAX_Y SIGMA_SIZE

// // https://coderwall.com/p/fzni3g
#define THREAD_ID ({                        \
    X + Y * MAX_X + Z * MAX_X * MAX_Y;      \
})

#define THREAD_LATICE_INDEX ({                 \
    X * LATICE_SIZE                             \
    + Y * MAX_X * LATICE_SIZE                   \
    + Z * MAX_X * MAX_Y *LATICE_SIZE;           \
}) 

#define BLOCK_ID ({                            \
    (blockIdx.x * gridDim.y) + blockIdx.y ;    \
})

#define SWAP(A, B, C) do {                     \
    C = A;                                     \
    A = B;                                     \
    B = C;                                     \
} while (0)


__global__ void generate_kernel(
    curandStateMtgp32 *state,
    short * LATICE, 
    short * NEXT_STEP_LATICE,
    int * DEV_MCS_NEEDED,
    float * DEV_BOND_DENSITY
) {
    // @todo change short to char ?
    // sizeof(GRID) = (32,32,1) | sizeof(BLOCK) = (32,1,1) | sizeof(LATICES) = (32^3, 60)
    // [GRIDX, GRIDY, BLOCKX] -> [tid]
    // short LATICE_1[LATICE_SIZE];
    // short LATICE_2[LATICE_SIZE];
    // short * LATICE           = LATICE_1;
    // short * NEXT_STEP_LATICE = LATICE_2;
    short * SWAP                = NULL;

    long  latice_update_counter = 0;
    short is_latice_updated     = FALSE;
    int   first_i, last_i       = 0;
    int   monte_carlo_steps     = 0;

    //Initialize LATICE as antiferromagnet [1,0,1,...]
    for (int i = THREAD_LATICE_INDEX; i < LATICE_SIZE + THREAD_LATICE_INDEX; i++) {
        LATICE[i] = (i&1);
    }
    while (monte_carlo_steps < MAX_MCS) {
        if ( is_latice_updated==FALSE && BOND_DENSITY(LATICE) == 0.0 ) {
            // If latice is in ferromagnetic state, simulation can stop
            break;
        }
        float C = TRIANGLE_DISTRIBUTION((X / (float)MAX_X), (Y / (float)MAX_Y));
        float W0 = Z/(float)MAX_Z;
        first_i = (int)(LATICE_SIZE * curand_uniform(&state[BLOCK_ID])) + THREAD_LATICE_INDEX;
        last_i  = (int)(first_i + (C * LATICE_SIZE)) + THREAD_LATICE_INDEX;
        is_latice_updated = FALSE; // ?
        for (int i = THREAD_LATICE_INDEX; i < LATICE_SIZE+THREAD_LATICE_INDEX; i++) {
            NEXT_STEP_LATICE[i] = LATICE[i];
            if (first_i <= i && i <= last_i) {
                int left  = (i-1) % LATICE_SIZE;
                int right = (i+1) % LATICE_SIZE;
                // Neighbours are the same and different than the current spin
                if ( LATICE[left] == LATICE[right] ) {
                    NEXT_STEP_LATICE[i] = LATICE[left];
                }
                // Otherwise randomly flip the spin
                else if ( W0 > curand_uniform(&state[BLOCK_ID])) {
                    NEXT_STEP_LATICE[i] = FLIP_SPIN(LATICE[i]);
                }
                latice_update_counter++; 
            }
            if (LATICE[i] != NEXT_STEP_LATICE[i]) {
                is_latice_updated = TRUE;
            }
        }
        monte_carlo_steps = (int)(latice_update_counter / LATICE_SIZE);
        SWAP(LATICE, NEXT_STEP_LATICE, SWAP);
    }
    DEV_BOND_DENSITY[THREAD_ID] = BOND_DENSITY(LATICE);
    DEV_MCS_NEEDED[THREAD_ID] = monte_carlo_steps; //printf("%d\n", monte_carlo_steps);
}

clock_t begin, end;
static double time_spent;

int main(int argc, char *argv[])    
{



    // Blocks iterate over W0; Grid iterates over MIU and SIGMA
    const int THREADS_NEEDED = MIU_SIZE * SIGMA_SIZE * W0_SIZE; // 10^3

    dim3 block_size(W0_SIZE,1,1);
    dim3 grid_size(MIU_SIZE,SIGMA_SIZE,1);

    // Maximum 256 threads per state
    // A given state may not be used by more than one block
    // @todo: upgrade for block_size > 256 threads ?
    const long PRNG_STATES    = grid_size.x*grid_size.y*grid_size.z;
    long PRNG_SEED      = time (NULL) * getpid();

    
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

    SETUP_PRNG; // setup devMTGPStates basing on devKernelParams, PRNG_STATES, PRNG_SEED

    short * DEV_LATICES;
    short * DEV_NEXT_STEP_LATICES;
    CUDA_CALL(cudaMalloc(                      
        &DEV_LATICES,               
        THREADS_NEEDED * sizeof(short) * LATICE_SIZE
    )); 
    CUDA_CALL(cudaMalloc(                      
        &DEV_NEXT_STEP_LATICES,               
        THREADS_NEEDED * sizeof(short) * LATICE_SIZE
    ));
    // @todo static memory allocation?
    int * DEV_MCS_NEEDED; float * DEV_BOND_DENSITY;
    int * HST_MCS_NEEDED; float * HST_BOND_DENSITY;
    HST_MCS_NEEDED = (int *)malloc(THREADS_NEEDED * sizeof(int));
    CUDA_CALL(
        cudaMalloc(&DEV_MCS_NEEDED,THREADS_NEEDED * sizeof(int))
    );

    HST_BOND_DENSITY = (float *)malloc(THREADS_NEEDED * sizeof(float));
    CUDA_CALL(
        cudaMalloc(&DEV_BOND_DENSITY,THREADS_NEEDED * sizeof(float))
    ); 

    for (int avgs = 0; avgs < AVERAGES; avgs++) {
        begin = clock();

        generate_kernel<<<grid_size, block_size>>>(
            devMTGPStates,
            DEV_LATICES, 
            DEV_NEXT_STEP_LATICES,
            DEV_MCS_NEEDED,
            DEV_BOND_DENSITY
        );

        CUDA_CALL(cudaMemcpy( 
            HST_BOND_DENSITY, 
            DEV_BOND_DENSITY, 
            THREADS_NEEDED * sizeof(int), 
            cudaMemcpyDeviceToHost
        ));

        CUDA_CALL(cudaMemcpy( 
            HST_MCS_NEEDED, 
            DEV_MCS_NEEDED, 
            THREADS_NEEDED * sizeof(int), 
            cudaMemcpyDeviceToHost
        ));

        end = clock();

        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        // printf("CPU time: %lf s - seed: %ld\n", time_spent, PRNG_SEED);
        int BLCK_SIZE = block_size.x*block_size.y*block_size.z;
        int GRID_SIZE = grid_size.x*grid_size.y*grid_size.z;
        printf("%d\t%d\t%d\t%d\t%lf\n", LATICE_SIZE, MAX_MCS, BLCK_SIZE, GRID_SIZE, time_spent);
        /* printf("C_MEAN   C_SIGMA  W_0      RHO      MCS\n");
        // for (int i = 0; i < THREADS_NEEDED; i++){
        //     int x = (int)(i % MAX_X);
        //     int y = (int)(( i / MAX_X ) % MAX_Y);
        //     int z = (int)(i / ( MAX_X * MAX_Y ));
        //     printf("%.6f %.6f %.6f %.6f %d\n", x*STEP_SIZE, y*STEP_SIZE, z*STEP_SIZE, HST_BOND_DENSITY[i], HST_MCS_NEEDED[i]);
        // } */
    }

    /* Cleanup */
    CUDA_CALL(cudaFree(devMTGPStates));
    CUDA_CALL(cudaFree(DEV_LATICES));
    CUDA_CALL(cudaFree(DEV_NEXT_STEP_LATICES));
    CUDA_CALL(cudaFree(DEV_MCS_NEEDED));
    CUDA_CALL(cudaFree(DEV_BOND_DENSITY));
    free(HST_MCS_NEEDED);
    free(HST_BOND_DENSITY);
    return EXIT_SUCCESS;
}