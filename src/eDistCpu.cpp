#include "eDistCpu.hpp"
#include <cmath>

// Function to calculate Euclidean distance of a vector
float euclideanDistanceCpu(const std::vector<char> vector) {
    float distance = 0.0f;

    // Sum the squares of each element in the vector
    for (float element : vector) {
        distance += static_cast<float>(element) * static_cast<float>(element);
    }

    // Take the square root of the sum to get the Euclidean distance
    distance = std::sqrt(distance);

    return distance;
}