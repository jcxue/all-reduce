project(example)

include_directories(
    .
    ../src
    /usr/local/cuda/include
)

set(${PROJECT_NAME}_HDR
)

set(${PROJECT_NAME}_SRC
    main.cpp
)

cuda_add_executable(main ${${PROJECT_NAME}_SRC})

target_link_libraries(
    main
    commLib
    cudaLib
)
