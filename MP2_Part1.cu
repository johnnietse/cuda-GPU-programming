#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include <device_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>





//define TITLE_WIDTH, change it according to what you want to test
#define TILE_WIDTH 2 //const int tile_width[number_Of_Configurations] = {2,4,8,16,32};

//define matrix size, change it according to what you want to test
#define MATRIX_DIMENSION (256) //const int matrixDim[number_Of_Configurations] = {256, 512, 1024, 2048, 4096};





int main() {

	//total matrix sizes in bytes (unit)
	int matrixBytes = MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(float);
	
	
	//device pointers
	float *d_M, *d_N, *d_P;

	//host pointers
	float *check_h_P, *h_M, *h_N, *h_P;

	//pinned host memory
	cudaMallocHost((void**)&check_h_P, matrixBytes);
	cudaMallocHost((void**)&h_P, matrixBytes);
	cudaMallocHost((void**)&h_M, matrixBytes);
	cudaMallocHost((void**)&h_N, matrixBytes);
	
	//grid dimensions
	int numberOfBlocks = MATRIX_DIMENSION / TILE_WIDTH;
	if (MATRIX_DIMENSION % TILE_WIDTH) {
		numberOfBlocks = numberOfBlocks + 1;
	}
	
	//define grid and block dimensions
	dim3 dimensionOfTheBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimensionOfTheGrid(numberOfBlocks, numberOfBlocks);

	//timing CUDA events
	cudaEvent_t startEvent; 
	cudaEvent_t stopEvent;
	
	
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	float tiled_Matrix_Multiplication_GPU_Elapsed_Time;

	//device memory
	cudaMalloc((void**)&d_P, matrixBytes);
	cudaMalloc((void**)&d_M, matrixBytes);
	cudaMalloc((void**)&d_N, matrixBytes);

	//initialize matrices with random val
	for (int x = 0; x < MATRIX_DIMENSION; x++) {
		
		
		for (int y = 0; y < MATRIX_DIMENSION; y++) {
			check_h_P[x * MATRIX_DIMENSION + y] = 0.0f;
			h_P[x * MATRIX_DIMENSION + y] = 0.0f;
			h_M[x * MATRIX_DIMENSION + y] = ((float)rand() / RAND_MAX) * 100.0f; 
			h_N[x * MATRIX_DIMENSION + y] = ((float)rand() / RAND_MAX) * 100.0f;
		}
		
		
	}

	//copying from input matrices to the device(s)
	cudaMemcpy(d_M, h_M, matrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, matrixBytes, cudaMemcpyHostToDevice);


	//execute the kernel with timing CUDA events
	cudaEventRecord(startEvent, 0);

	tiledMatrixMultiplicationKernel << <dimensionOfTheGrid, dimensionOfTheBlock, 0, 0 >> > (d_M, d_N, d_P, MATRIX_DIMENSION);

	cudaEventRecord(stopEvent, 0);
	
	cudaEventSynchronize(stopEvent);


	cudaEventElapsedTime(&tiled_Matrix_Multiplication_GPU_Elapsed_Time, startEvent, stopEvent);
	printf("Device Tiled Matrix Multiplication Time for matrix matrixBytes of [%dx%d]: %f ms\n", MATRIX_DIMENSION, MATRIX_DIMENSION, tiled_Matrix_Multiplication_GPU_Elapsed_Time);

	printf("TILE WIDTH: %d\n", TILE_WIDTH);
	
	
	//results back to host by copying
	cudaMemcpy(h_P, d_P, matrixBytes, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_M, d_M, matrixBytes, cudaMemcpyDeviceToHost);
	
	cudaMemcpy(h_N, d_N, matrixBytes, cudaMemcpyDeviceToHost);


	//CPU solution for reference
	matrixMultiplicationForCPU(h_M, h_N, check_h_P, MATRIX_DIMENSION);



	//comparing the results of GPU v.s. CPU
	int verificationFailed = 0;
	
	const float tolerance = 1e-3f;
	
	for (int x = 0; x < MATRIX_DIMENSION; x++) {
	
	
		for (int y = 0; y < MATRIX_DIMENSION; y++) {
			
			if (abs(h_P[x * MATRIX_DIMENSION + y] - check_h_P[x * MATRIX_DIMENSION + y]) > tolerance) {
				verificationFailed = 1;
				break;
			}
			
		}
	
	
	}

	
	printf("Test %s\n", verificationFailed ? "FAILED" : "PASSED");

	cudaFree(d_P);
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(h_P);
	cudaFree(h_M);
	cudaFree(h_N);
	cudaFree(check_h_P);
	
	return 0;
}



		
//Tiled Matrix Multiplication Kernel
__global__ void tiledMatrixMultiplicationKernel(float* M, float* N, float* P, int matrixDimension) {
	
	//declare shared memory for the tiles
	__shared__ float matrix_M_Shared[TILE_WIDTH][TILE_WIDTH];
	__shared__ float matrix_N_Shared[TILE_WIDTH][TILE_WIDTH];

	//thread and block indicies
	int threadx = threadIdx.x; 
	int thready = threadIdx.y;
	int blockx = blockIdx.x; 
	int blocky = blockIdx.y;
	
	//row and column indicies
	int row = blocky * TILE_WIDTH + thready;
	int col = blockx * TILE_WIDTH + threadx;

	float val_P = 0.0f;


	/tiled matrix multiplication but by phase
	for (int phase = 0; phase < (matrixDimension / TILE_WIDTH); ++phase) {
		matrix_M_Shared[thready][threadx] = M[row * matrixDimension + phase * TILE_WIDTH + threadx];
		matrix_N_Shared[thready][threadx] = N[(phase * TILE_WIDTH + thready) * matrixDimension + col];
		__syncthreads();
		
		
		//find partial product for current tile
		for (int z = 0; z < TILE_WIDTH; ++z) {
			val_P += matrix_M_Shared[thready][z] * matrix_N_Shared[z][threadx];
		}
		
		
		//synchronize threads
		__syncthreads();
		
		
	}
	
	
	
	P[row * matrixDimension + col] = val_P;

}



//use for CPU Matrix Multiplication
void matrixMultiplicationForCPU(float* M, float* N, float* P, int matrixDimension) {
	for (int x = 0; x < matrixDimension; ++x) {
		
		
		for (int y = 0; y < matrixDimension; ++y) {
			
			float sum = 0.0f;
			
			//basic and general way of doing matrix multiplication
			for (int z = 0; z < matrixDimension; ++z) {
				sum += M[x * matrixDimension + z] * N[z * matrixDimension + y];
			}
			
			P[x * matrixDimension + y] = sum;
		}
	
	
	}
}
