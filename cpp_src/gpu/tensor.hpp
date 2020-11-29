#pragma once

#include <cuda_runtime_api.h>
#include "gpu/dark.cuh"
#include "types/enum.hpp"

namespace darknet
{
namespace gpu
{

    template<typename _Tp>
    struct plus
    {
      _GLIBCXX14_CONSTEXPR
      _Tp
      __host__ __device__ operator()(_Tp& __x, _Tp& __y) const
      { return __x + __y; }
    };

    template<typename _Tp>
    struct minus
    {
      _GLIBCXX14_CONSTEXPR
      _Tp
      __host__ __device__ operator()(const _Tp& __x, const _Tp& __y) const
      { return __x - __y; }
    };

    template<typename _Tp>
    struct divides
    {
      _GLIBCXX14_CONSTEXPR
      _Tp
      __host__ __device__ operator()(const _Tp& __x, const _Tp& __y) const
      { return __x / __y; }
    };

    template<typename _Tp>
    struct multiplies
    {
      _GLIBCXX14_CONSTEXPR
      _Tp
      __host__ __device__ operator()(const _Tp& __x, const _Tp& __y) const
      { return __x * __y; }
    };

    template<typename T, ActivationType A>
    void applyFunctor(T* data, size_t n);

    template<typename T, typename F>
    void elementwise(T* data, size_t n, T other);

    template<typename T, typename F>
    void elementwise(T* data, size_t n, T* other, size_t m);
} // namespace gpu
} // namespace darknet
