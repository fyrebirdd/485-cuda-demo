#include "eDistCpu.hpp"
#include <cmath>

// Function to calculate Euclidean distance of a vector
float euclideanDistanceCpu(const std::vector<char> vector) {
    double distance = 0.0f;

    // Sum the squares of each element in the vector
    for (char element : vector) {
        int eInt = (element - '0');
        distance += (double)(eInt * eInt);
    }

    // Take the square root of the sum to get the Euclidean distance
    distance = std::sqrt(distance);

    return distance;
}