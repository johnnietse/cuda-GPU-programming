#include <stdio.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

//can change accordingly to test different TILE_WIDTH {2,4,8,16,32}
#define TILE_WIDTH 4

int main() {
	// Default Device ID (can be changed if we have multiple GPU devices available)
    int deviceID = 0; 
    
	cudaError_t error = cudaGetDeviceCount(&deviceID);
	
	//if it is not a valid CUDA device then return 1 and exit the program
    if (error != cudaSuccess) {
        printf("CUDA Error. Fail to set CUDA device: %s\n", cudaGetErrorString(error));
        return 1;
    }
	
	//set GPU device for usage
	cudaSetDevice(deviceID); 


	//get GPU properties 
    cudaDeviceProp gpuProperties;
    cudaGetDeviceProperties(&gpuProperties, deviceID);
	
	//make sure that GPU properties are retrieved 
	error = cudaGetDeviceProperties(&gpuProperties, deviceID);

    if (error != cudaSuccess) {
        printf("CUDA Error. Failed to get GPU Device Properties: %s\n", cudaGetErrorString(error));
    }


	//key GPU specs
	
	int maximumSharedMemoryPerSM = gpuProperties.sharedMemPerMultiprocessor;
	
	int warpSize = gpuProperties.warpSize;

    int maximum_Number_Of_Threads_Per_SM = gpuProperties.maxThreadsPerMultiProcessor;

    int number_Of_Streaming_Multiprocessors = gpuProperties.multiProcessorCount;
    

    //maximum blocks per SM based on shared memory size constraints
    int maximum_blocks_per_SM = maximumSharedMemoryPerSM / (2 * TILE_WIDTH * TILE_WIDTH * sizeof(float));

    //maximum threads scheduling capacity per SM
    int maximumThreadsPerBlock = maximum_Number_Of_Threads_Per_SM / warpSize;
    int totalSchedulableThreadsAcrossAllSMs = maximum_blocks_per_SM * maximumThreadsPerBlock;

	//display the information asked by the question
    printf("Number of Streaming Multiprocessors (SMs): %d\n", number_Of_Streaming_Multiprocessors);
    printf("Total Threads that can be scheduled across all SMs: %d\n", totalSchedulableThreadsAcrossAllSMs);
	printf("Maximum Threads per Multiprocessor: %d\n", maximum_Number_Of_Threads_Per_SM);
    printf("CUDA Warp Size: %d\n", warpSize);
    printf("Maximum Shared Memory per Multiprocessor: %d KB\n", maximumSharedMemoryPerSM/1024);
    printf("Maximum Number of Blocks per SM: %d\n", maximum_blocks_per_SM);

    return 0;
}


//convert CUDA compute capability version to the number of cores per SM
int getCudaCoresPerSM(int major, int minor) {
    
	int totalCoresCount = 0;

    switch (minor + (major << 4)) {
	
	
    case 0x10:
        totalCoresCount = 8;
        break;
    
	
	case 0x11:
    case 0x12:
        totalCoresCount = 8;
        break;
    
	
	case 0x13:
        totalCoresCount = 32;
        break;
    
	
	case 0x20:
        totalCoresCount = 32;
        break;
    
	
	default:
        totalCoresCount = 0;
        break;
    }


    return totalCoresCount;
}


