cmake_minimum_required(VERSION 3.5)
project(mlearning)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2  -std=c++20 -Wall -Wpedantic -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 ")

file(GLOB sources "src/*.cpp" "src/*.c")

add_executable(mlearning ${sources})
