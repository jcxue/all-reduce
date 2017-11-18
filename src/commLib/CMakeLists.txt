project(commLib)

include_directories(
    ./
    ../
)

set(${PROJECT_NAME}_SRC
    Communicator.cpp
)

set(${PROJECT_NAME}_HDR
    Communicator.h
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SRC} ${${PROJECT_NAME}_HDR})

target_link_libraries(
    ${PROJECT_NAME}
    cudaLib
)

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} CACHE INTERNAL "${PROJECT_NAME}: Include Directory" FORCE)
