cmake_minimum_required(VERSION 2.6)

SET(OpenCV_STATIC OFF)

IF(MSVC)
    ADD_DEFINITIONS(-DNOMINMAX)
ELSE()
    IF(APPLE)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++") # C++11 support on Apple clang
    ELSE()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")# C++11 support on GCC
    ENDIF()
ENDIF()

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../include ${OpenCV_INCLUDE_DIRS})
project(RANSAC)

SET(CPP_FILES main.cpp)
set(SOURCE_FILES main.cpp include/Model.h include/Ransac.h include/Ransac.h LineModel.h)
SET(VC_LIBS ${OpenCV_LIBS})

ADD_EXECUTABLE(${PROJECT_NAME} ${CPP_FILES} ${HPP_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})