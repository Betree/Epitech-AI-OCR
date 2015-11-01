cmake_minimum_required(VERSION 3.0)
project(OCR)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(includes)

set(TRAINER_SOURCE_FILES
        includes/NeuralNetwork.hpp
        includes/indexed_vector.hpp
        includes/Trainer.hpp
        includes/ocr_utils.hpp
	includes/dirent.h
        sources/ocr_utils.cpp
        sources/NeuralNetwork.cpp
        sources/Trainer.cpp
        sources/trainer_main.cpp)

add_executable(ocr_trainer ${TRAINER_SOURCE_FILES})

set(SOURCE_FILES
        includes/NeuralNetwork.hpp
        includes/indexed_vector.hpp
        includes/Trainer.hpp
        includes/ocr_utils.hpp
	includes/dirent.h
        sources/ocr_utils.cpp
        sources/NeuralNetwork.cpp
        sources/Trainer.cpp
        sources/main.cpp sources/ImagesLoader.cpp includes/ImagesLoader.h sources/ImageProcessor.cpp includes/ImageProcessor.h)

add_executable(ocr ${SOURCE_FILES})
target_link_libraries(OCR ${OpenCV_LIBS})
