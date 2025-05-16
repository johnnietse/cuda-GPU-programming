#include "cuda_runtime.h"
#include <stdio.h>
#include <stdbool.h>
#include "device_launch_parameters.h"
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define tolerance 1e-5


int main() {
    int matrixSizes[] = { 256, 512, 1024, 2048, 4096 };
   
    int blockSizes[] = { 2, 4, 8, 16, 32 };


    for (int i = 0; i < 5; i++) {
       
	   int width = matrixSizes[i];
        size_t bytes = width * width * sizeof(float);

        printf("This is the output for matrix size %dx%d\n", width, width);

        printf("------------------------------------------------------------------------------------------------------------------------------\n");


        float* d_M, * d_N, * d_P;


		//allocate memory for host matrices for the CPU
        float* h_M = (float*)malloc(bytes);
        float* h_N = (float*)malloc(bytes);
        float* h_P_CPU = (float*)malloc(bytes);
        float* h_P_GPU = (float*)malloc(bytes);


		//call the function that initialize matrices with random values
        matrixInit(h_M, width);
        matrixInit(h_N, width);


		//allocate memory for device matrices for the GPU
        cudaMalloc((void**)&d_M, bytes);
        cudaMalloc((void**)&d_N, bytes);
        cudaMalloc((void**)&d_P, bytes);


		//CUDA event timers for measuring time
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        float time_ms = 0;


		//host to device transfer
        cudaEventRecord(begin);
        
		cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        
		cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
        
		cudaEventRecord(end);
        
		cudaEventSynchronize(end);
        
		cudaEventElapsedTime(&time_ms, begin, end);
        
		printf("Host-to-Device transfer time for size %d x %d: %f ms\n", width, width, time_ms);




		//device to host transfer
        cudaEventRecord(begin);

        // Copy result back to host (Device -> Host)
        cudaMemcpy(h_P_GPU, d_P, bytes, cudaMemcpyDeviceToHost);

        cudaEventRecord(end);
        cudaEventSynchronize(end);

        cudaEventElapsedTime(&time_ms, begin, end);

        printf("Device to Host transfer time for size %d x %d: %f ms\n", width, width, time_ms);
        
		printf("-----------------------------------------------------------------------------------------------------------------------------\n");

		//gpu execution
        int block_width;
        for (int j = 0; j < 5; j++) {
            
			block_width = blockSizes[j];
            
			dim3 threadsPerBlock(block_width, block_width);
            
			dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

			//gpu kernel starting up
            cudaEventRecord(begin);
            
			GPUMatrixMultiply << <blocksPerGrid, threadsPerBlock >> > (d_P, d_M, d_N, width);
            
			cudaEventRecord(end);
            
			cudaEventSynchronize(end);
            
			cudaEventElapsedTime(&time_ms, begin, end);
            
            cudaError_t error = cudaGetLastError();
            
			if (error != cudaSuccess) {
                printf("Error in executing CUDA Kernel: %s\n", cudaGetErrorString(error));
            }
            printf("GPU calculation time for size %d x %d and block width %d: %f ms\n", width, width, block_width, time_ms);


			//perform matrix multiplication on CPU for comparison with GPU
            cudaEventRecord(begin);
            
			CPUMatrixMultiply(h_P_CPU, h_M, h_N, width);
            
			cudaEventRecord(end);
            
			cudaEventSynchronize(end);
            
			cudaEventElapsedTime(&time_ms, begin, end);
            
			printf("CPU calculation time for size %d x %d and block width %d: %f ms\n", width, width, block_width, time_ms);

            
			//copy results from the GPU to the host
			cudaMemcpy(h_P_GPU, d_P, bytes, cudaMemcpyDeviceToHost);




			//check gpu results with cpu results
            bool isitcorrect = true;
			
			
			
            for (int k = 0; k < width * width; k++) {
                if (fabs(h_P_CPU[k] - h_P_GPU[k]) > tolerance) {
                    isitcorrect = false;
                    break;
                }
            }

            if (isitcorrect) {
            
				printf("Test PASSED\n\n");
            
			} 
			
			else {
                
				
				printf("Test FAILED\n\n");
            
			
			}
            
			
			printf("--------------------------------------------------------------------------------------------------------------------------------\n");
        }


		//free memory
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        free(h_M);
        free(h_N);
        free(h_P_CPU);
        free(h_P_GPU);

        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }

    return 0;
}






//gpu kernel for matrix multiplication computation
__global__ void GPUMatrixMultiply(float* P, float* M, float* N, int width) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < width) {
	
        float sum = 0;
		
        for (int k = 0; k < width; k++) {

            sum += M[row * width + k] * N[k * width + col];
        }
        sum = P[row * width + col];
    }
}

//cpu kernel for matrix multiplication computation
void CPUMatrixMultiply(float* P, const float* M, const float* N, int width) {
    for (int row = 0; row < width; row++) {
	
	
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int n = 0; n < width; n++) {
                sum += M[row * width + n] * N[n * width + col];
            }
            sum = P[row * width + col];
        }
    }
	
	
}


//initialize the matrix with random values 
void matrixInit(float* arrayData, int width) {
    for (int i = 0; i < width * width; i++) {
        arrayData[i] = (rand() / (float)RAND_MAX);
    }
}