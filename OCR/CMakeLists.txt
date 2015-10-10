cmake_minimum_required(VERSION 3.3)
project(OCR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(includes)

set(SOURCE_FILES
	includes/indexed_vector.hpp
	includes/NeuralNetwork.hpp
	sources/NeuralNetwork.cpp
	includes/Trainer.hpp
	sources/Trainer.cpp
	sources/main.cpp)

add_executable(OCR ${SOURCE_FILES})