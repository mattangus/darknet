cmake_minimum_required(VERSION 3.15)

project("Testing")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules/")

# execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYTORCHLIB)
set(PYTORCHLIB ${CMAKE_SOURCE_DIR}/lib/libtorch)
message("-- PyTorchCmake: " ${PYTORCHLIB})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTORCHLIB})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate GTest
find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA  REQUIRED)
# include_directories("${CUDA_INCLUDE_DIRS}")
find_package(CUDNN REQUIRED)

set(CUDNN_INCLUDE_PATH ${CUDNN_INCLUDE_DIRS})
set(CUDNN_LIBRARY_PATH ${CUDNN_LIBRARIES})
find_package(Torch REQUIRED)
message("-- Troch flags: " ${TORCH_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -no-pie")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

# add_compile_definitions(CUDA=1)
set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
message("-- CUDA falgs: " ${CUDA_ARCH_FLAGS})

CUDA_ADD_EXECUTABLE(runTests3
    test/test3.cpp
)

target_link_libraries(runTests3 ${CUDA_LIBRARIES} ${GTEST_LIBRARIES} ${CUDNN_LIBRARIES} ${TORCH_LIBRARIES} pthread)
target_include_directories(runTests3 PUBLIC ${TORCH_INCLUDE_DIRS} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS})

add_test(runTests runTests)