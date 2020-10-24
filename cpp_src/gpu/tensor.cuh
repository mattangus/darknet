#pragma once

#include <cuda_runtime_api.h>
#include "gpu/dark.cuh"
#include "types/enum.hpp"

namespace darknet
{
namespace gpu
{
    template<typename T, ActivationType A>
    void applyFunctor(T* data, size_t n);
} // namespace gpu
} // namespace darknet
