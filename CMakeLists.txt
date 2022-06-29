cmake_minimum_required(VERSION 3.22)
project(MarchingCube_C++)
set(CMAKE_C_FLAGS "-g -fopenmp -Ofast")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(run main.cpp)
target_link_libraries (run Eigen3::Eigen -fopenmp)
set_target_properties(run PROPERTIES CXX_STANDARD 17)