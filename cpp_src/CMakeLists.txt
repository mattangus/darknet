cmake_minimum_required(VERSION 3.15)

project(Darknet)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules/")
message("-- CMake modules: " ${CMAKE_MODULE_PATH})

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate GTest
find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA  REQUIRED)
# include_directories("${CUDA_INCLUDE_DIRS}")
find_package(CUDNN REQUIRED)
find_package( OpenCV REQUIRED )

# execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYTORCHLIB)
set(PYTORCHLIB ${CMAKE_SOURCE_DIR}/lib/libtorch)
message("-- PyTorchCmake: " ${PYTORCHLIB})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTORCHLIB})
set(CUDNN_INCLUDE_PATH ${CUDNN_INCLUDE_DIRS})
set(CUDNN_LIBRARY_PATH ${CUDNN_LIBRARIES})
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

add_compile_definitions(WITH_CUDA=1)
message("-- Troch flags: " ${TORCH_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
message("-- CUDA flags: " ${CUDA_ARCH_FLAGS})
message("-- GTEST: " ${GTEST_LIBRARIES})
CUDA_ADD_EXECUTABLE(runTests3
    test/test3.cpp
    types/enum.cpp
)

target_include_directories(runTests3 PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "."
)
message("-- CUDNN libraries: " ${CUDNN_LIBRARIES})
message("-- OpenCV libraries: " ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(runTests3
    ${CUDA_LIBRARIES}
    ${GTEST_LIBRARIES}
    ${CUDNN_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    TorchVision::TorchVision
    pthread
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
)

target_compile_definitions(runTests3 PUBLIC USECUDA)

add_test(runTests runTests)