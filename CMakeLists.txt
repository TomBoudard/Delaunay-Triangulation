cmake_minimum_required(VERSION 3.12)

# set the project name and version
project(DELAUNAY_TRIANGULATION LANGUAGES CXX)

# set CXX standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SOURCES 
    ${SRC_DIR}/main.cu
    ${SRC_DIR}/point.cu
    ${SRC_DIR}/sort_array.cu)

# CUDA Libraries
find_package(CUDA)
if(CUDA_FOUND)
  enable_language(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS} )

  add_executable(main ${SOURCES})
endif()