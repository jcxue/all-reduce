# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
cmake_minimum_required(VERSION 3.2.2 FATAL_ERROR)

# project setting
project(all-reduce C CXX)
set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# included dirs
include_directories("${PROJECT_SOURCE_DIR}")

# self level packages
add_subdirectory(src)
add_subdirectory(examples)
