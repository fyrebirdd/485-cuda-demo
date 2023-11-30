#include "charCountCpu.hpp"

int charCountCPU(std::vector<char> matrix, std::string target){
    int numCount = 0;
    
    for(auto character: matrix){
        if(character == target[0]){
            numCount++;
        }
    }
    return numCount;
}