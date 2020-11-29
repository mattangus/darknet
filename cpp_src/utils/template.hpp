#pragma once

#if CUDA
    #include <cuda_fp16.h>
#endif

#include "types/enum.hpp"

// template<typename B>
// using EnableIfB = typename std::enable_if<B>::type;

// #define TEMPLATEDEVICE(D, _dev) template<DeviceType D1 = D, EnableIfB<device == DeviceType::_dev> = 0>
// #define CPUTEMPLATE(D) TEMPLATEDEVICE(D, CPU)
// #define GPUTEMPLATE(D) TEMPLATEDEVICE(D, GPU)

#define NUMERIC_TYPES(MACRO)    MACRO(half); \
                                MACRO(float); \
                                MACRO(double); \
                                MACRO(int8_t); \
                                MACRO(int32_t); \
                                MACRO(int64_t); \
                                MACRO(uint8_t); \
                                MACRO(uint32_t); \
                                MACRO(uint64_t);

#define REAL_TYPES(DEF)     MACRO(half); \
                            MACRO(float); \
                            MACRO(double);

