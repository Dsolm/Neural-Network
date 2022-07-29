cmake_minimum_required(VERSION 3.5)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wpedantic -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")

file(GLOB sources "src/*.cpp" "src/*.c")

add_executable(neuralnetwork ${sources})
