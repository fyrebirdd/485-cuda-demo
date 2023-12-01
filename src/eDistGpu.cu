#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "eDistGpu.hpp"

#include <iostream>

void HandleError(cudaError_t cudaStatus, std::string functionCalling) {
    if (cudaStatus != cudaSuccess) {
        std::cout << "CUDA ERROR: <" << functionCalling << "> " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

__global__ void euclideanDistanceKernel(char* vector, int size, float* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        int valueAsInt = (vector[tid] - '0');
        float square = (float)(valueAsInt * valueAsInt);
        atomicAdd(result,square);
    }
}

float euclideanDistanceCUDA(std::vector<char>& inputVec) {
    int size = static_cast<int>(inputVec.size());

    // Device vectors
    char* d_vector;
    float* d_result;
    float h_result = 0;

    HandleError(cudaSetDevice(0), "cudaSetDevice");

    // Allocate memory on the device
    HandleError(cudaMalloc((void**)&d_vector, size * sizeof(char)), "cudaMalloc");
    HandleError(cudaMalloc((void**)&d_result, sizeof(float)), "cudaMalloc2");

    // Copy input vector from host to device
    HandleError(cudaMemcpy(d_vector, inputVec.data(), size * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy pre");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Define grid and block sizes
    int blockSize = prop.maxThreadsPerBlock;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch the kernel to convert characters to floats
    euclideanDistanceKernel<<<gridSize, blockSize>>>(d_vector, size, d_result);

    HandleError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    HandleError(cudaGetLastError(), "cudaGetLastError");

    // Copy the result vector from device to host
    HandleError(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy post");
    // Free memory on the device
    cudaFree(d_vector);
    cudaFree(d_result);

    return sqrt(h_result);
}