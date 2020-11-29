#pragma once

#include <cmath>
#include <cuda_runtime_api.h>

namespace darknet
{
namespace layer
{
    template <typename T>
    struct stair
    {
        _GLIBCXX14_CONSTEXPR
        T
            __device__ __host__
            operator()(T x)
        {
            int n = std::floor(x);
            if (n % 2 == 0)
                return std::floor(x / 2.f);
            else
                return (x - n) + std::floor(x / 2.f);
        }
    };

    template <typename T>
    struct hardtan
    {
        _GLIBCXX14_CONSTEXPR
        T
            __device__ __host__
            operator()(T x)
        {
            if (x < -1)
                return -1;
            if (x > 1)
                return 1;
            return x;
        }
    };

    template <typename T>
    struct linear
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return x;
        }
    };

    template <typename T>
    struct logistic
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return 1.f / (1.f + std::exp(-x));
        }
    };

    template <typename T>
    struct loggy
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return 2.f / (1.f + std::exp(-x)) - 1;
        }
    };

    template <typename T>
    struct relu
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return x * (x > 0);
        }
    };

    template <typename T>
    struct relu6
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return std::min(std::max(x, 0.0f), 6.0f);
        }
    };

    template <typename T>
    struct elu
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x >= 0) * x + (x < 0) * (std::exp(x) - 1);
        }
    };

    template <typename T>
    struct selu
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x >= 0) * 1.0507f * x + (x < 0) * 1.0507f * 1.6732f * (std::exp(x) - 1);
        }
    };

    template <typename T>
    struct relie
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x > 0) ? x : .01f * x;
        }
    };

    template <typename T>
    struct ramp
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return x * (x > 0) + .1f * x;
        }
    };

    template <typename T>
    struct leaky
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x > 0) ? x : .1f * x;
        }
    };

    template <typename T>
    struct tanh
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (2 / (1 + std::exp(-2 * x)) - 1);
        }
    };

    template <typename T>
    struct gelu
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (0.5 * x * (1 + std::tanh(0.797885 * x + 0.035677 * std::pow(x, 3))));
        }
    };

    template <typename T>
    struct softplus
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x, float threshold)
        {
            if (x > threshold)
                return x; // too large
            else if (x < -threshold)
                return std::exp(x); // too small
            return std::log(std::exp(x) + 1);
        }
    };

    template <typename T>
    struct plse
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            if (x < -4)
                return .01f * (x + 4);
            if (x > 4)
                return .01f * (x - 4) + 1;
            return .125f * x + .5f;
        }
    };

    template <typename T>
    struct lhtan
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            if (x < 0)
                return .001f * x;
            if (x > 1)
                return .001f * (x - 1) + 1;
            return x;
        }
    };

    template <typename T>
    struct lhtan_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            if (x > 0 && x < 1)
                return 1;
            return .001f;
        }
    };

    template <typename T>
    struct hardtan_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            if (x > -1 && x < 1)
                return 1;
            return 0;
        }
    };

    template <typename T>
    struct linear_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return 1;
        }
    };

    template <typename T>
    struct logistic_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (1 - x) * x;
        }
    };

    template <typename T>
    struct loggy_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            float y = (x + 1.f) / 2.f;
            return 2 * (1 - y) * y;
        }
    };

    template <typename T>
    struct stair_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            if (floor(x) == x)
                return 0;
            return 1.0f;
        }
    };

    template <typename T>
    struct relu_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x > 0);
        }
    };

    template <typename T>
    struct relu6_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x > 0 && x < 6);
        }
    };

    template <typename T>
    struct elu_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x >= 0) + (x < 0) * (x + 1);
        }
    };

    template <typename T>
    struct selu_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x >= 0) * 1.0507f + (x < 0) * (x + 1.0507f * 1.6732f);
        }
    };

    template <typename T>
    struct relie_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x > 0) ? 1 : .01f;
        }
    };

    template <typename T>
    struct ramp_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x > 0) + .1f;
        }
    };

    template <typename T>
    struct leaky_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x > 0) ? 1 : .1f;
        }
    };

    template <typename T>
    struct tanh_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return 1 - x * x;
        }
    };

    template <typename T>
    struct sech
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return 2 / (std::exp(x) + std::exp(-x));
        }
    };

    template <typename T>
    struct gelu_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            const float x3 = std::pow(x, 3);
            return 0.5 * std::tanh(0.0356774 * x3 + 0.797885 * x) + (0.0535161 * x3 + 0.398942 * x) * std::pow(sech<T>()(0.0356774 * x3 + 0.797885 * x), 2) + 0.5;
        }
    };

    template <typename T>
    struct plse_gradient
    {
        _GLIBCXX14_CONSTEXPR
        T

            __device__ __host__
            operator()(T x)
        {
            return (x < 0 || x > 1) ? .01f : .125f;
        }
    };

} // namespace layer
} // namespace darknet
