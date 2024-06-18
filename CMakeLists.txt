cmake_minimum_required(VERSION 3.8)
project(reactgpu LANGUAGES CXX CUDA)

# Find packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

# Set compiler flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_FLAGS "-Ofast")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress 20012 --use_fast_math -ftz=true -prec-div=false -prec-sqrt=false")

# CUDA kernel
add_library(reactor-kernel SHARED src/reactor.cu src/reactor.cuh)
target_include_directories(reactor-kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIRS})
set_target_properties(reactor-kernel PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reactgpu
)

# Python bindings
pybind11_add_module(reactor src/reactor.cc src/reactor.cuh src/models.h src/utilities.h)
target_link_libraries(reactor PUBLIC pybind11::headers Eigen3::Eigen PRIVATE reactor-kernel)
target_include_directories(reactor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIRS})
set_target_properties(reactor PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reactgpu)