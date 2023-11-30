#include "testFileToArray.hpp"

 std::vector<char> testFileToArray(const std::string& fileName){
    std::vector<char> output{ 0 };

    std::ifstream infile(fileName);
    std::string line;
    std::string fileContents;
    if(infile.is_open()){
        int linesAdded = 0;
        while(getline(infile, line)){            
            line.erase(std::remove_if(line.begin(), line.end(), [](char c) { return c == ' ' || c == '\n' || c == '\r'; }), line.end());
            output.insert(output.end(), line.begin(), line.end());
        }
        infile.close();
    }
    else{
        std::cout << "Unable to open file" << std::endl;
    }
    return output;
}