#pragma once

#include "core/dtype.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace core {
namespace cuda {

// 基础逐元素加法
void elementwise_add(const void* a, const void* b, void* out, int64_t n,
                    DType dtype, cudaStream_t stream = 0);

// 规约加法(返回总和)
void reduce_add(const void* data, void* result, int64_t n, DType dtype,
               cudaStream_t stream = 0);

// 使用Tensor Core的矩阵加法(适合特定形状)
void tensorcore_add(const void* a, const void* b, void* out,
                   int m, int n, int k, DType dtype,
                   cudaStream_t stream = 0);

// 原子加法(原地操作)
void atomic_add(void* data, const void* value, int64_t n, DType dtype,
               cudaStream_t stream = 0);

} // namespace cuda
} // namespace core