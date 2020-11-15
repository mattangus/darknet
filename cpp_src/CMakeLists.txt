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
set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
message("cuda falgs: " ${CUDA_ARCH_FLAGS})

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

CUDA_ADD_EXECUTABLE(runTests2 test/tests2.cpp tensor/tensor_cpu.cpp types/enum.cpp tensor/tensor_gpu.cpp gpu/tensor.cu gpu/dark.cu)

target_include_directories(runTests2 PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_INCLUDE_DIRS}
    "../3rdparty"
    ${GTEST_INCLUDE_DIRS}
    "."
)
TARGET_LINK_LIBRARIES(runTests2 ${CUDA_LIBRARIES} ${GTEST_LIBRARIES} pthread)

# target_compile_options(runTests2 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)



add_test(runTests runTests)