#include "layer/activation_fn.hpp"

namespace darknet
{
namespace layer
{
    template<typename T>
    __device__ __host__ static inline T stair(T x)
    {
        int n = std::floor(x);
        if (n%2 == 0) return std::floor(x/2.f);
        else return (x - n) + std::floor(x/2.f);
    }
    template<typename T>
    __device__ __host__ static inline T hardtan(T x)
    {
        if (x < -1) return -1;
        if (x > 1) return 1;
        return x;
    }
    template<typename T>
    __device__ __host__ static inline T linear(T x){return x;}
    template<typename T>
    __device__ __host__ static inline T logistic(T x){return 1.f/(1.f + std::exp(-x));}
    template<typename T>
    __device__ __host__ static inline T loggy(T x){return 2.f/(1.f + std::exp(-x)) - 1;}
    template<typename T>
    __device__ __host__ static inline T relu(T x){return x*(x>0);}
    template<typename T>
    __device__ __host__ static inline T relu6(T x) { return std::min(std::max(x, 0.0f), 6.0f); }
    template<typename T>
    __device__ __host__ static inline T elu(T x){return (x >= 0)*x + (x < 0)*(std::exp(x)-1);}
    template<typename T>
    __device__ __host__ static inline T selu(T x) { return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(std::exp(x) - 1); }
    template<typename T>
    __device__ __host__ static inline T relie(T x){return (x>0) ? x : .01f*x;}
    template<typename T>
    __device__ __host__ static inline T ramp(T x){return x*(x>0)+.1f*x;}
    template<typename T>
    __device__ __host__ static inline T leaky(T x){return (x>0) ? x : .1f*x;}
    template<typename T>
    __device__ __host__ static inline T tanh(T x) { return (2 / (1 + std::exp(-2 * x)) - 1); }
    template<typename T>
    __device__ __host__ static inline T gelu(T x) { return (0.5*x*(1 + std::tanh(0.797885*x + 0.035677*std::pow(x, 3)))); }
    template<typename T>
    __device__ __host__ static inline T softplus(T x, float threshold) {
        if (x > threshold) return x;                // too large
        else if (x < -threshold) return std::exp(x);    // too small
        return std::log(std::exp(x) + 1);
    }
    template<typename T>
    __device__ __host__ static inline T plse(T x)
    {
        if(x < -4) return .01f * (x + 4);
        if(x > 4)  return .01f * (x - 4) + 1;
        return .125f*x + .5f;
    }

    template<typename T>
    __device__ __host__ static inline T lhtan(T x)
    {
        if(x < 0) return .001f*x;
        if(x > 1) return .001f*(x-1) + 1;
        return x;
    }
    template<typename T>
    __device__ __host__ static inline T lhtan_gradient(T x)
    {
        if(x > 0 && x < 1) return 1;
        return .001f;
    }

    template<typename T>
    __device__ __host__ static inline T hardtan_gradient(T x)
    {
        if (x > -1 && x < 1) return 1;
        return 0;
    }
    template<typename T>
    __device__ __host__ static inline T linear_gradient(T x){return 1;}
    template<typename T>
    __device__ __host__ static inline T logistic_gradient(T x){return (1-x)*x;}
    template<typename T>
    __device__ __host__ static inline T loggy_gradient(T x)
    {
        float y = (x+1.f)/2.f;
        return 2*(1-y)*y;
    }
    template<typename T>
    __device__ __host__ static inline T stair_gradient(T x)
    {
        if (floor(x) == x) return 0;
        return 1.0f;
    }
    template<typename T>
    __device__ __host__ static inline T relu_gradient(T x){return (x>0);}
    template<typename T>
    __device__ __host__ static inline T relu6_gradient(T x) { return (x > 0 && x < 6); }
    template<typename T>
    __device__ __host__ static inline T elu_gradient(T x){return (x >= 0) + (x < 0)*(x + 1);}
    template<typename T>
    __device__ __host__ static inline T selu_gradient(T x) { return (x >= 0)*1.0507f + (x < 0)*(x + 1.0507f*1.6732f); }
    template<typename T>
    __device__ __host__ static inline T relie_gradient(T x){return (x>0) ? 1 : .01f;}
    template<typename T>
    __device__ __host__ static inline T ramp_gradient(T x){return (x>0)+.1f;}
    template<typename T>
    __device__ __host__ static inline T leaky_gradient(T x){return (x>0) ? 1 : .1f;}
    template<typename T>
    __device__ __host__ static inline T tanh_gradient(T x){return 1-x*x;}

    template<typename T>
    __device__ __host__ static inline T sech(T x) { return 2 / (std::exp(x) + std::exp(-x)); }
    template<typename T>
    __device__ __host__ static inline T gelu_gradient(T x) {
        const float x3 = std::pow(x, 3);
        return 0.5*std::tanh(0.0356774*x3 + 0.797885*x) + (0.0535161*x3 + 0.398942*x) * std::pow(sech(0.0356774*x3 + 0.797885*x), 2) + 0.5;
    }
    template<typename T>
    __device__ __host__ static inline T plse_gradient(T x){return (x < 0 || x > 1) ? .01f : .125f;}

} // namespace layer
} // namespace darknet
