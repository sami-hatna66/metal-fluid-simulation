cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

project(fluidsim)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(SDL2 REQUIRED)

add_subdirectory(metal-cmake)

include_directories(${SDL2_INCLUDE_DIR})

add_executable(fluidsim src/main.cpp src/Fluid.hpp src/Fluid.cpp)

target_link_libraries(fluidsim METAL_CPP ${SDL2_LIBRARY})
