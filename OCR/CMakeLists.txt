cmake_minimum_required(VERSION 3.3)
project(OCR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(includes)

set(SOURCE_FILES
        includes/NeuralNetwork.hpp
        includes/indexed_vector.hpp
        includes/Trainer.hpp
        sources/NeuralNetwork.cpp
        sources/Trainer.cpp
        sources/main.cpp)

add_executable(OCR ${SOURCE_FILES})