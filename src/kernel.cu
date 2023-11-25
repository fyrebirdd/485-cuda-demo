
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "testFileToArray.hpp"

#include <iostream>
#include <chrono>
#include <tuple>

std::chrono::milliseconds numberSearchCuda(std::vector<char> board, int* output, char target);

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

    std::chrono::milliseconds readFileTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::chrono::milliseconds procTime = numberSearchCuda(input, &output, numberToFind);

    std::cout << "GPU: Load from file time (THIS RUNS ON CPU): " << readFileTime.count() << "ms" << std::endl;
    std::cout << "GPU: Proccessing Time: " << procTime.count() << "ms" << std::endl;
    std::cout << "GPU: Number of " << numberToFind << "'s in matrix: " << output << std::endl;
    
    return 0;
}

//helper function for using CUDA to search for a number in parallel
std::chrono::milliseconds numberSearchCuda(std::vector<char> board, int* output, char target) {
    int row_amount = board.size();

    //out
    int* dev_output = 0;
    //in
    char* dev_board = 0;
    //err
    cudaError_t cudaStatus;

    const int blockSize = 256;
    const int gridSize = (row_amount + blockSize - 1) / blockSize;



    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    std::chrono::milliseconds procDuration;

    start = std::chrono::high_resolution_clock::now();

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }
    //allocate memory on gpu
    cudaStatus = cudaMalloc((void**)&dev_output, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_board, row_amount * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //copy data to gpu
    cudaStatus = cudaMemcpy(dev_board, board.data(), row_amount * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }    

    //launch kernel
    charSearchKernel <<<gridSize, blockSize>>> (dev_board, target, row_amount, dev_output);


    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "charSearch launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // blocks main thread until kernel is finished, returns any errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching charSearchKernel!\n", cudaStatus);
        goto Error;
    }


    cudaStatus = cudaMemcpy(output, dev_output, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    end = std::chrono::high_resolution_clock::now();
    procDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

Error:
    cudaFree(dev_output);
    cudaFree(dev_board);

    return procDuration;

}