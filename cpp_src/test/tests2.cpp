

#include <gtest/gtest.h>

// Seems a bit hacky but gets the job done.

#define ATENSOR CpuTensor
#define BTENSOR GpuTensor
#define TEST_NAME TestCPUTensor

#include "test/tensor.hpp"

#undef ATENSOR
#undef BTENSOR
#undef TEST_NAME

#define ATENSOR GpuTensor
#define BTENSOR CpuTensor
#define TEST_NAME TestGPUTensor

#include "test/tensor.hpp"

#undef ATENSOR
#undef BTENSOR
#undef TEST_NAME

int main(int argc, char **argv) {
    // register_handlers();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}