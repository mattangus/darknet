cmake_minimum_required(VERSION 3.15)

project("Testing")
# Locate GTest
find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA  REQUIRED)
# include_directories("${CUDA_INCLUDE_DIRS}")
# find_package(CUDNN REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Link runTests with what we want to test and the GTest and pthread library
CUDA_ADD_EXECUTABLE(runTests test/tests.cpp utils/signal_handler.cpp gpu/dark.cu gpu/tensor.cu)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

target_link_libraries(runTests ${GTEST_LIBRARIES} pthread)

target_include_directories(runTests PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_INCLUDE_DIRS}
    "../3rdparty"
    ${GTEST_INCLUDE_DIRS}
    "."
)
TARGET_LINK_LIBRARIES(runTests ${CUDA_LIBRARIES})
add_test(runTests runTests)