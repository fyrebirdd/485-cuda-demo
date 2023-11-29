#include "testFileToArray.hpp"
#include <chrono>
#include <cstdlib>
#include <cmath>


int numberSearchCPU(std::vector<char> matrix, std::string target){
    int numCount = 0;
    for(auto character: matrix){
        if(character == target[0]){
            numCount++;
        }
    }return numCount;
}

// Function to calculate Euclidean distance of a vector
float euclideanDistance(const std::vector<char> vector) {
    float distance = 0.0f;

    // Sum the squares of each element in the vector
    for (float element : vector) {
        distance += static_cast<float>(element) * static_cast<float>(element);
    }

    // Take the square root of the sum to get the Euclidean distance
    distance = std::sqrt(distance);

    return distance;
}

int main(int argc, char* argv[]){
    // Initalizing user args
    if (argc != 4){
        std::cout << "Usage: " << argv[0] << " <input file>" << "<rows>" << "<number to find>" << std::endl;
        exit(1);
    }
    std::string fileName = argv[1];
    int rows = atoi(argv[2]);
    std::string targetChar = argv[3];

    // creating timers
    std::chrono::milliseconds mallocDuration;
    std::chrono::milliseconds procDuration;
    
    // pre-processing (loading array from file)
    auto fileStart = std::chrono::high_resolution_clock::now();
    auto matrix = testFileToArray(fileName, rows);
    auto fileEnd = std::chrono::high_resolution_clock::now();
    mallocDuration = std::chrono::duration_cast<std::chrono::milliseconds>(fileEnd - fileStart);

    // running and timing CPU search of matrix
    auto procStart = std::chrono::high_resolution_clock::now();
    auto numCount = numberSearchCPU(matrix, targetChar);
    auto procEnd = std::chrono::high_resolution_clock::now();
    procDuration = std::chrono::duration_cast<std::chrono::milliseconds>(procEnd-procStart);

    // Outputing times
    std::cout << "Starting numberSearchCPU()" << std::endl;
    std::cout << "CPU: Load from file time (THIS RUNS ON CPU): " << mallocDuration.count() << "ms" << std::endl;
    std::cout << "CPU: Proccessing Time: " << procDuration.count() << "ms" << std::endl;
    std::cout << "CPU: Number of " << targetChar << "'s in matrix: " << numCount << std::endl;
    std::cout <<"\n" << std::endl;
    
    // running and timing CPU calculaion on vector
    procStart = std::chrono::high_resolution_clock::now();
    float dist = euclideanDistance(matrix);
    procEnd = std::chrono::high_resolution_clock::now();
    procDuration = std::chrono::duration_cast<std::chrono::milliseconds>(procEnd-procStart);

    std::cout << "Starting euclideanDistanceCPU()" << std::endl;
    std::cout << "CPU: Proccessing Time: " << procDuration.count() << "ms" << std::endl;
    std::cout << "CPU: Euclidian Distance: " << dist << std::endl;

    return 0;
}