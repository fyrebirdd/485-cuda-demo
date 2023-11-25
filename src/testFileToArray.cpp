#include "testFileToArray.hpp"

std::vector<char> splitString(std::string input, const char& delimiter) {
    std::vector<char> output;
    std::istringstream f(input);
    std::string s;
    while (getline(f, s, delimiter)) {
        output.push_back(s[0]);
    }
    return output;
}
 std::vector<char> testFileToArray(const std::string fileName, int rows){
    std::vector<char> output( rows*rows, 0 );

    std::ifstream infile(fileName);
    std::string line;
    std::string fileContents;
    if(infile.is_open()){
        int linesAdded = 0;
        while(getline(infile, line)){            
            std::string::size_type pos = 0;
            while(pos < line.length()){
                pos = line.find('\n', pos);
                if (pos == std::string::npos){
                    break;
                }
                line.erase(pos,2);
            }
            auto stringSplit = splitString(line, ' ');
            for (int j = 0; j < int(stringSplit.size()); j++){
                output[(linesAdded * rows) + j] = stringSplit[j];
            }  
            linesAdded++;
        }
        infile.close();
        std::cout << "Allocated: " << ((output.size() * sizeof(char)) * 0.000001f) << " MB" << std::endl;
    }
    else{
        std::cout << "Unable to open file" << std::endl;
    }
    return output;
}