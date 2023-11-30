#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "charCountGpu.hpp"

#include <iostream>

void HandleError(cudaError_t cudaStatus, std::string functionCalling) {
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA ERROR: <" << functionCalling << "> " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

__global__ void charCountKernel(char* chars, char characterToCount, int length, int* count){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < length){
        if (chars[tid] == characterToCount) {
            atomicAdd(count, 1);
        }
        tid += blockDim.x * gridDim.x;
    }
}

//helper function for using CUDA to count a specific character in parallel
int charCountCuda(std::vector<char> board, char target) {
    int arraySize = board.size();
    int output = 0;

    //out
    int* dev_output = 0;
    //in
    char* dev_board = 0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const int blockSize = prop.maxThreadsPerBlock;
    const int gridSize = (arraySize + blockSize - 1) / blockSize;

    HandleError(cudaSetDevice(0), "cudaSetDevice");

    //allocate memory on gpu
    HandleError(cudaMalloc((void**)&dev_output, sizeof(int)), "cudaMalloc");

    HandleError(cudaMalloc((void**)&dev_board, arraySize * sizeof(char)), "cudaMalloc2");
    
    //copy data to gpu
    HandleError(cudaMemcpy(dev_board, board.data(), arraySize * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy Pre");

    //launch kernel
    charCountKernel <<<gridSize, blockSize>>> (dev_board, target, arraySize, dev_output);

    // blocks main thread until kernel is finished, returns any errors
    HandleError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    HandleError(cudaGetLastError(), "cudaGetLastError");

    HandleError(cudaMemcpy(&output, dev_output, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Post");

    cudaFree(dev_output);
    cudaFree(dev_board);

    return output;
}
