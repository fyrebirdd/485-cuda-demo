#include "testFileToArray.hpp"
#include <chrono>
#include <cstdlib>


int main(int argc, char* argv[]){
    
    if (argc != 4){
        std::cout << "Usage: " << argv[0] << " <input file>" << "<rows>" << "<number to find>" << std::endl;
        exit(1);
    }

    std::string fileName = argv[1];
    int rows = atoi(argv[2]);
    std::string numberToFind = argv[3];

    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    std::chrono::milliseconds mallocDuration;
    std::chrono::milliseconds procDuration;
    
    start = std::chrono::high_resolution_clock::now();

    auto matrix = testFileToArray(fileName, rows);

    end = std::chrono::high_resolution_clock::now();

    mallocDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();

    int numCount = 0;

    for(auto character: matrix){
        if(character == numberToFind[0]){
            numCount++;
        }
    }

    end = std::chrono::high_resolution_clock::now();
    procDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);

    std::cout << "CPU: Load from file time (THIS RUNS ON CPU): " << mallocDuration.count() << "ms" << std::endl;
    std::cout << "CPU: Proccessing Time: " << procDuration.count() << "ms" << std::endl;
    std::cout << "CPU: Number of " << numberToFind << "'s in matrix: " << numCount << std::endl;

    return 0;
}