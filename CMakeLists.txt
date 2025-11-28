cmake_minimum_required(VERSION 3.10)
project(MyCppProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the OpenCV_DIR to help CMake find the package
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/x64/mingw/lib")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories for OpenCV
include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(MyCppProject src/main.cpp src/my_header.cpp)

# Link OpenCV libraries
target_link_libraries(MyCppProject ${OpenCV_LIBS})
