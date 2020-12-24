#pragma once

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
//#include <driver_types.h>

#ifdef CUDNN
#include <cudnn.h>
#endif // CUDNN

#define BLOCK 512

#define CUDA_1D_KERNEL_LOOP(index, nthreads) for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (nthreads); index += blockDim.x * gridDim.x)
#define CHECK_ERROR(X) darknet::gpu::check_error_extended(X, std::string(__FILE__) + " : " + std::string(__FUNCTION__), __LINE__,  std::string(__DATE__) + " - " + std::string(__TIME__) );


namespace darknet
{
namespace gpu
{
    /**
     * @brief Get the 2D grid size (z dim is set to 1) for the number of elements.
     * 
     * @param n number of elements
     * @param block_size size of compute blocks
     * @return dim3 3D grid size
     */
    dim3 cuda_gridsize(size_t n, size_t block_size = BLOCK);
    /**
     * @brief Get the number of blocks for the array size.
     * 
     * @param n number of elements
     * @param block_size size of compute blocks
     * @return int 
     */
    size_t get_number_of_blocks(size_t n, size_t block_size = BLOCK);

    /**
     * @brief Check the status of the current cuda error and display message if it is not a success.
     * 
     * @param status Current cuda status to check. 
     * @param file file name for caller.
     * @param line line number for caller.
     * @param date_time date and time of compilation.
     */
    void check_error_extended(cudaError_t status, const std::string file, int line, const std::string date_time);

#ifdef CUDNN
    /**
     * @brief Check the status of the current cudnn error and display message if it is not a success.
     * 
     * @param status Current cuda status to check. 
     * @param file file name for caller.
     * @param line line number for caller.
     * @param date_time date and time of compilation.
     */
    void check_error_extended(cudnnStatus_t status, const std::string file, int line, const std::string date_time);

#endif
} // namespace gpu
} // namespace darknet
