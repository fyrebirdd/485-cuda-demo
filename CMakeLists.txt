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

add_executable(runCountTest ${SRC_DIR}/runCountTest.cpp ${SRC_DIR}/charCountGpu.cu ${SRC_DIR}/charCountGpu.hpp ${SRC_DIR}/charCountCpu.cpp ${SRC_DIR}/charCountCpu.hpp ${SRC_DIR}/testFileToArray.cpp ${SRC_DIR}/testFileToArray.hpp)
add_executable(runEDistTest ${SRC_DIR}/runEDistTest.cpp ${SRC_DIR}/eDistGpu.cu ${SRC_DIR}/eDistGpu.hpp ${SRC_DIR}/eDistCpu.cpp ${SRC_DIR}/eDistCpu.hpp ${SRC_DIR}/testFileToArray.cpp ${SRC_DIR}/testFileToArray.hpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates)

# Set output directories for executables
set_target_properties(
    runCountTest
    PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
    LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates	
    PDB_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
)

set_target_properties(
    runEDistTest
    PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
    LIBRARY_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates	
    PDB_OUTPUT_DIRECTORY ${BUILD_DIR}/intermediates
)

# Custom command to copy executables to the Testing folder
add_custom_command(
    TARGET runCountTest runEDistTest
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:runCountTest>
            $<TARGET_FILE:runEDistTest>
            ${TEST_DIR}
)