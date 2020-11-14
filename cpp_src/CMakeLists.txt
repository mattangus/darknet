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


SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_compile_definitions(CUDA=1)

# Link runTests with what we want to test and the GTest and pthread library
# CUDA_ADD_EXECUTABLE(runTests test/tests.cpp utils/signal_handler.cpp gpu/dark.cu gpu/tensor.cu)

# target_include_directories(runTests PUBLIC
#     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#     ${CUDA_INCLUDE_DIRS}
#     "../3rdparty"
#     ${GTEST_INCLUDE_DIRS}
#     "."
# )
# TARGET_LINK_LIBRARIES(runTests ${CUDA_LIBRARIES} ${GTEST_LIBRARIES} pthread)

CUDA_ADD_EXECUTABLE(runTests2 test/tests2.cpp tensor/tensor_cpu.cpp types/enum.cpp tensor/tensor_gpu.cpp)

target_include_directories(runTests2 PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_INCLUDE_DIRS}
    "../3rdparty"
    ${GTEST_INCLUDE_DIRS}
    "."
)
TARGET_LINK_LIBRARIES(runTests2 ${CUDA_LIBRARIES} ${GTEST_LIBRARIES} pthread)


add_test(runTests runTests)