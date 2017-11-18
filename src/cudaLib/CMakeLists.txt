project(cudaLib)

find_package(CUDA QUIET REQUIRED)

include_directories(
    ./
    /usr/local/cuda/include
)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_60,code=sm_60
)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")

set(${PROJECT_NAME}_SRC
    cuda_func.cu
)

set(${PROJECT_NAME}_HDR
    cuda_func.h
)

cuda_add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SRC} ${${PROJECT_NAME}_HDR})

target_link_libraries(
    ${PROJECT_NAME}
)

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} CACHE INTERNAL "${PROJECT_NAME}: Include Directory" FORCE)
