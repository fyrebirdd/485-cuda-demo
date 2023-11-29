
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "testFileToArray.hpp"

#include <iostream>
#include <chrono>
#include <tuple>
#include <cmath>

__global__ void charSearchKernel(char* board, char character, int rows, int *output) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < rows) {
        if (board[tid] == character) {
            atomicAdd(output, 1);
        }
        tid += blockDim.x * gridDim.x;
    }
}

//helper function for using CUDA to search for a number in parallel
std::chrono::milliseconds charSearchCuda(std::vector<char> board, int* output, char target) {
    int row_amount = board.size();

    //out
    int* dev_output = 0;
    //in
    char* dev_board = 0;
    //err
    cudaError_t cudaStatus;

    const int blockSize = 256;
    const int gridSize = (row_amount + blockSize - 1) / blockSize;

    std::chrono::milliseconds procDuration;

    auto start = std::chrono::high_resolution_clock::now();

    cudaStatus = cudaSetDevice(0);

    //allocate memory on gpu
    cudaStatus = cudaMalloc((void**)&dev_output, sizeof(int));

    cudaStatus = cudaMalloc((void**)&dev_board, row_amount * sizeof(char));

    //copy data to gpu
    cudaStatus = cudaMemcpy(dev_board, board.data(), row_amount * sizeof(char), cudaMemcpyHostToDevice);

    //launch kernel
    charSearchKernel <<<gridSize, blockSize>>> (dev_board, target, row_amount, dev_output);

    // blocks main thread until kernel is finished, returns any errors
    cudaStatus = cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(output, dev_output, sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    procDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cudaFree(dev_output);
    cudaFree(dev_board);

    return procDuration;

}



//helper function for using CUDA to calc euclidain distance in parallel
__global__ void euclideanDistanceKernel(char* vector, int size, float* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        float square = static_cast<float>(vector[tid]) * static_cast<float>(vector[tid]);
        atomicAdd(result,square);
    }
}

std::chrono::milliseconds euclideanDistanceCUDA(std::vector<char>& inputVec, float* output) {
    int size = static_cast<int>(inputVec.size());

    // Device vectors
    char* d_vector;
    float* d_result;

    // Allocate memory on the device
    cudaMalloc((void**)&d_vector, size * sizeof(char));
    cudaMalloc((void**)&d_result, size * sizeof(float));

    // Copy input vector from host to device
    cudaMemcpy(d_vector, inputVec.data(), size * sizeof(char), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    std::chrono::milliseconds procDuration;
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel to convert characters to floats
    euclideanDistanceKernel<<<gridSize, blockSize>>>(d_vector, size, d_result);
    // Copy the result vector from device to host
    float* h_result = new float;
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    float dist = sqrt(*h_result);
    std::cout  << "FART " << dist <<std::endl;
    *output = dist;
    
    auto end = std::chrono::high_resolution_clock::now();
    procDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Free memory on the device
    cudaFree(d_vector);
    cudaFree(d_result);
    delete[] h_result;

    return procDuration;
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <input file>" << "<rows>" << "<number to find>" << std::endl;
        exit(1);
    }

    std::string fileName = argv[1];
    int rows = atoi(argv[2]);
    char numberToFind = *argv[3];

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<char> input = testFileToArray(fileName, rows);
    auto end = std::chrono::high_resolution_clock::now();

    int output = 0;
    float outputDistance;

    std::chrono::milliseconds readFileTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::chrono::milliseconds procTime = charSearchCuda(input, &output, numberToFind);

    std::cout << "Starting numberSearchCuda()" << std::endl;
    std::cout << "GPU: Load from file time (THIS RUNS ON CPU): " << readFileTime.count() << "ms" << std::endl;
    std::cout << "GPU: Proccessing Time: " << procTime.count() << "ms" << std::endl;
    std::cout << "GPU: Number of " << numberToFind << "'s in matrix: " << output << std::endl;
    std::cout <<"\n" << std::endl;

    procTime = euclideanDistanceCUDA(input, &outputDistance);

    std::cout << "Starting euclideanDistanceCUDA()" << std::endl;
    std::cout << "GPU: Proccessing Time: " << procTime.count() << "ms" << std::endl;
    std::cout << "GPU: Euclidian Distance: " << outputDistance << std::endl;
    return 0;
}


