#pragma once

// Libraries
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

// Function declarations
std::vector<char> testFileToArray(std::string fileName, int rows);
std::vector<char> splitString(std::string input, const char& delimiter);