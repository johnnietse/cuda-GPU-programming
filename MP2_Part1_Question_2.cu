#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 2 //TILE_WIDTH can be changed for testing {2, 4, 8, 16, 32}

int main() {
    //GPU device properties
	cudaDeviceProp gpuProperties;
	
    cudaGetDeviceProperties(&gpuProperties, 0); // Assuming device 0

	//max blocks per SM calculation
    int maximumSharedMemoryPerSM = gpuProperties.sharedMemPerMultiprocessor;

    int maximumSharedMemoryPerSM = maximumSharedMemoryPerSM / (2 * TILE_WIDTH * TILE_WIDTH * sizeof(float));
   
	//max threads per block 
    int maxThreadsPerBlock = gpuProperties.maxThreadsPerBlock;
 
	//as the variable is named it is the calculation for the maximum total threads simultaneously scheduled/executing
    int maximumTotalThreadsSimulatenouslyScheduled = maximumSharedMemoryPerSM * maxThreadsPerBlock;


	//information asked by the question to print out
    printf("Total Number of Registers: %d\n", gpuProperties.regsPerBlock * TILE_WIDTH * TILE_WIDTH);
    
	printf("Shared Memory Size: %d bytes\n", 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float));
    
	printf("Number of Blocks per Streaming Multiprocessor (SM): %d\n", maximumSharedMemoryPerSM);
    
	printf("Maximum Total Threads Simultaneously Scheduled/Executing: %d\n", maximumTotalThreadsSimulatenouslyScheduled);

    return 0;
}



//similar to the tiledMatrixMultiplicationKernel() function from "MP2_Part1.cu", so refer to that function for commenting
__global__ void kernelForMatrixMultiplication(float* M, float* N, float* P, int matrixDimension) {
    __shared__ float matrix_M_Shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float matrix_N_Shared[TILE_WIDTH][TILE_WIDTH];

    float val_P = 0;

    int threadx = threadIdx.x;
    int thready = threadIdx.y;
    int blockx = blockIdx.x;
    int blocky = blockIdx.y;


    int row = blocky * TILE_WIDTH + thready;
    int col = blockx * TILE_WIDTH + threadx;


    for (int phase = 0; phase < matrixDimension / TILE_WIDTH; ++phase) {
        
		matrix_M_Shared[thready][threadx] = M[row * matrixDimension + phase * TILE_WIDTH + threadx];
        matrix_N_Shared[thready][threadx] = N[(phase * TILE_WIDTH + thready) * matrixDimension + col];
        
		__syncthreads();
		
        for (int z = 0; z < TILE_WIDTH; ++z) {
            val_P += matrix_M_Shared[thready][z] * matrix_N_Shared[z][threadx];
        }
		
		
		
        __syncthreads();
    }
	
	
	
    P[row * matrixDimension + col] = val_P;
}
