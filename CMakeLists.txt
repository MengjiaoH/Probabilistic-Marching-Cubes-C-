cmake_minimum_required(VERSION 3.16)
project(MarchingCube_C++)
set(CMAKE_C_FLAGS "-g -fopenmp -Ofast")

find_package(OpenMP)
find_package(TBB REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(run main.cpp)
target_link_libraries (run Eigen3::Eigen TBB::tbb OpenMP::OpenMP_CXX )

# if(OpenMP_CXX_FOUND)
#     target_link_libraries(run PUBLIC OpenMP::OpenMP_CXX)
# endif()
set_target_properties(run PROPERTIES CXX_STANDARD 11)

# OpenMP::OpenMP_CXX 