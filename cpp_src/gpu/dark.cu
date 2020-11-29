#include "gpu/dark.cuh"
#include <stdexcept>

namespace darknet
{
namespace gpu
{
    dim3 cuda_gridsize(size_t n, size_t block_size){
        size_t k = (n-1) / block_size + 1;
        size_t x = k;
        size_t y = 1;
        if(x > 65535){
            x = ceil(sqrt(k));
            y = (n-1)/(x*block_size) + 1;
        }
        
        dim3 d;
        d.x = x;
        d.y = y;
        d.z = 1;

        return d;
    }

    size_t get_number_of_blocks(size_t array_size, size_t block_size)
    {
        return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
    }


    void check_error_extended(cudaError_t status, const char *file, int line, const char *date_time)
    {
        if (status != cudaSuccess) {
            printf("CUDA status Error: file: %s() : line: %d : build time: %s \n", file, line, date_time);
            throw std::runtime_error(cudaGetErrorString(status));
        }
    }

#ifdef CUDNN
    void cudnn_check_error_extended(cudnnStatus_t status, const char *file, int line, const char *date_time)
    {
        if (status != CUDNN_STATUS_SUCCESS) {
            printf("\n cuDNN status Error in: file: %s() : line: %d : build time: %s \n", file, line, date_time);
            throw std::runtime_error(cudnnGetErrorString(status));
        }
    }
#endif
} // namespace gpu
} // namespace darknet
