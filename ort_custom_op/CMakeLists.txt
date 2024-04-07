cmake_minimum_required(VERSION 3.10)
project(customop)
add_definitions(-std=c++11)

set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.cc)

#Include path to header files for Custom Op
include_directories("../3rdparty/eigen3")
include_directories("../3rdparty/onnxruntime-linux-x64-1.13.1/include")

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "../3rdparty/onnxruntime-linux-x64-1.13.1/lib")

add_library(custom_group_norm SHARED ${HEADER} ${SOURCE})
target_link_libraries(custom_group_norm ${ONNXRUNTIME_LIBRARY})

add_executable(customop ${TEST_SOURCE} ${SOURCE})
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})


