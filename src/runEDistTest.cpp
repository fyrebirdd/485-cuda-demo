#include <string>
#include <chrono>
#include <vector>

#include "eDistCpu.hpp"
#include "eDistGpu.hpp"
#include "testFileToArray.hpp"

int main(int argc, char* argv[]){
    if (argc != 2){
        std::cout << "Usage: " << argv[0] << " <input file>" << std::endl;
        exit(1);
    }
    std::string fileName = argv[1];

    // creating timers
    std::chrono::milliseconds mallocDuration;
    std::chrono::milliseconds cpuProcDuration;
    std::chrono::milliseconds gpuProcDuration;

    // pre-processing (loading array from file)
    auto fileStart = std::chrono::high_resolution_clock::now();
    auto input = testFileToArray(fileName);
    auto fileEnd = std::chrono::high_resolution_clock::now();
    mallocDuration = std::chrono::duration_cast<std::chrono::milliseconds>(fileEnd - fileStart);

    // running and timing CPU search of matrix
    auto procStart = std::chrono::high_resolution_clock::now();
    auto cpuDist = euclideanDistanceCpu(input);
    auto procEnd = std::chrono::high_resolution_clock::now();
    cpuProcDuration = std::chrono::duration_cast<std::chrono::milliseconds>(procEnd-procStart);

    // running and timing GPU search of matrix
    auto gpuProcStart = std::chrono::high_resolution_clock::now();

    float gpuDist = euclideanDistanceCUDA(input);

    auto gpuProcEnd = std::chrono::high_resolution_clock::now();
    gpuProcDuration = std::chrono::duration_cast<std::chrono::milliseconds>(gpuProcEnd-gpuProcStart);

    // Outputing times
    std::cout << "Euclidean Distance Test Results:" << std::endl;
    std::cout << "Load time (On Host): " << mallocDuration.count() << "ms" << std::endl;
    std::cout << "CPU:"<< std::endl;
    std::cout << "Proccessing Time: " << cpuProcDuration.count() << "ms" << std::endl;
    std::cout << "Euclidean Distance: "  << cpuDist << std::endl;
    std::cout << "GPU:"<< std::endl;
    std::cout << "Processing Time: " << gpuProcDuration.count() << "ms" << std::endl;
    std::cout << "Euclidean Distance: "  << gpuDist << std::endl;
    std::cout << std::endl;
    return 0;

    
}