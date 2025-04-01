#pragma once

#include <stdexcept>
#include "device.h"
#include "dtype.h"


// CUDA 错误检查宏
#ifdef WITH_CUDA
#define CUDA_CHECK(call) {                                         \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                     \
        throw std::runtime_error(                                  \
            std::string("CUDA error at ") + __FILE__ + ":" +       \
            std::to_string(__LINE__) + ": " +                      \
            cudaGetErrorString(err));                              \
    }                                                              \
}
#else
#define CUDA_CHECK(call) {}
#endif

// 设备断言
#define ASSERT_ON_DEVICE(tensor, device)                           \
    if ((tensor).device() != (device)) {                           \
        throw std::runtime_error("Tensor is not on expected device"); \
    }

#define ASSERT_DTYPE(tensor, dtype)                                \
    if ((tensor).dtype() != (dtype)) {                            \
        throw std::runtime_error("Tensor has unexpected dtype");   \
    }
