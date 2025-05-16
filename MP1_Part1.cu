#include <stdio.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

int main()
{
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
	
	//if it is not a valid CUDA device then return 1 and exit the program
    if (error != cudaSuccess) {
        printf("Cuda Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
	
	//sometimes there can be more than one device, so this "for loop" is for looping through all available devices and printing out their properties
    printf("Number of available devices %d \n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp gpuProperties;
        cudaGetDeviceProperties(&gpuProperties, i);
        printf("-----------------------------------------------------------------------------------\n");
        DisplayDeviceSpecificationsfn(gpuProperties);
    }
}



//function to display and print off all the GPU device properties

void DisplayDeviceSpecificationsfn(cudaDeviceProp gpuProperties) {
    printf("CUDA Device Name and Type: %s \n", gpuProperties.name);
    printf("Clock Rate: %d\n", gpuProperties.clockRate);
    printf("Number of Streaming Multiprocessors (SM's): %d\n", gpuProperties.multiProcessorCount);


    //initialize all the variables for calculating the number of CUDA cores
    int majorVersion = gpuProperties.major;
    int mPCount = gpuProperties.multiProcessorCount;
    int totalCoresCount = 0;


    if (majorVersion == 2) {
        totalCoresCount = 32 * mPCount;
        printf("Number of cores: %d\n", totalCoresCount);
    }


    else if (majorVersion == 3) {
        totalCoresCount = 192 * mPCount;
        printf("Number of cores: %d\n", totalCoresCount);
    }


    else if (majorVersion == 5) {
        totalCoresCount = 128 * mPCount;
        printf("Number of cores: %d\n", totalCoresCount);
    }

    else if (majorVersion == 6) {
        totalCoresCount = 64 * mPCount;
        printf("Number of cores: %d\n", totalCoresCount);
    }

    else if (majorVersion == 7) {
        totalCoresCount = 64 * mPCount;
        printf("Number of cores: %d\n", totalCoresCount);
    }

    else if (majorVersion == 8) {
        totalCoresCount = 64 * mPCount;
        printf("Number of cores: %d\n", totalCoresCount);
    }

	//unsupport architecture
    else {
        printf("Failed to retrieve the number of cores.\n");
		totalCoresCount = -1;

    }

	//some more additional GPU device properties
    printf("Warp Size: %d\n", gpuProperties.warpSize);
    printf("Total Amount of Global Memory: %d\n", gpuProperties.totalGlobalMem);
    printf("Total Amount of Constant Memory: %d\n", gpuProperties.totalConstMem);
    printf("Amount of Shared Memory per Block: %d\n", gpuProperties.sharedMemPerBlock);
    printf("Number of Registers Available per Block: %d\n", gpuProperties.regsPerBlock);
    printf("Maximum Number of Threads per Block: %d\n", gpuProperties.maxThreadsPerBlock);
    printf("Maximum Size of Each Dimension of a Block: %d x %d x %d\n", gpuProperties.maxThreadsDim[0], gpuProperties.maxThreadsDim[1], gpuProperties.maxThreadsDim[2]);
    printf("Maximum Size of Each Dimension of a Grid: %d x %d x %d\n", gpuProperties.maxGridSize[0], gpuProperties.maxGridSize[1], gpuProperties.maxGridSize[2]);
    printf("\n");
	printf("\n");
	printf("\n");


}

