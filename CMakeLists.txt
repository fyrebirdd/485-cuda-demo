cmake_minimum_required(VERSION 3.12)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(cudaTesting)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Enable CUDA language
enable_language(CUDA)

# Set source files for the project
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testing)
set(BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Add executable for CPU version
add_executable(cpuNumberCounting ${SRC_DIR}/cpuNumberCounting.cpp ${SRC_DIR}/testFileToArray.cpp)

# Add executable for GPU version
add_executable(gpuNumberCounting ${SRC_DIR}/kernel.cu ${SRC_DIR}/testFileToArray.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates)

# Set output directories for executables
set_target_properties(
    cpuNumberCounting
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
    LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates	
    PDB_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
)

set_target_properties(
    gpuNumberCounting
    PROPERTIES
    CUDA_ARCHITECTURES 75
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
    LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
    PDB_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
)

# Custom command to copy executables to the Testing folder
add_custom_command(
    TARGET cpuNumberCounting gpuNumberCounting
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:cpuNumberCounting>
            $<TARGET_FILE:gpuNumberCounting>
            ${TEST_DIR}
)