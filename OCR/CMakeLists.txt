cmake_minimum_required(VERSION 3.0)
project(OCR)

find_package( OpenCV REQUIRED )

if (CMAKE_COMPILER_IS_GNUCXX)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -pthread")
endif (CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (MSVC)
	add_definitions(-D_USE_MATH_DEFINES)
endif (MSVC)

include_directories(includes)
include_directories(${OpenCV_INCLUDE_DIRS})

set(TRAINER_SOURCE_FILES
        includes/NeuralNetwork.hpp
        includes/indexed_vector.hpp
        includes/Trainer.hpp
        includes/ocr_utils.hpp
		includes/dirent.h
		includes/ImageProcessor.h
		includes/ImagesLoader.h
		includes/cvplot.h
		sources/cvplot.cpp
		sources/ImageProcessor.cpp
		sources/ImagesLoader.cpp
        sources/ocr_utils.cpp
        sources/NeuralNetwork.cpp
        sources/Trainer.cpp
        sources/trainer_main.cpp)

add_executable(ocr_trainer ${TRAINER_SOURCE_FILES})
target_link_libraries(ocr_trainer ${OpenCV_LIBS})

set(SOURCE_FILES
        includes/NeuralNetwork.hpp
        includes/indexed_vector.hpp
        includes/Trainer.hpp
        includes/ocr_utils.hpp
		includes/dirent.h
		includes/ImageProcessor.h
		includes/ImagesLoader.h
		includes/OCR.h
		sources/OCR.cpp
		sources/ImageProcessor.cpp
		sources/ImagesLoader.cpp
        sources/ocr_utils.cpp
        sources/NeuralNetwork.cpp
        sources/Trainer.cpp
        sources/main.cpp)

add_executable(ocr ${SOURCE_FILES})
target_link_libraries(ocr ${OpenCV_LIBS})
