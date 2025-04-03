#include "core/cuda/elementwise.cuh"
#include <cooperative_groups.h>
#include <mma.h>

namespace core {
namespace cuda {

// CUDA错误检查宏
#define CUDA_CHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if(e != cudaSuccess) {                            \
        throw std::runtime_error(                     \
            std::string("CUDA error at ") +           \
            __FILE__ + ":" + std::to_string(__LINE__) + \
            ": " + cudaGetErrorString(e));            \
    }                                                 \
} while(0)

// ==============================================
// 内核函数实现
// ==============================================

namespace kernel {

// 基础逐元素加法内核
template <typename T>
__global__ void elementwise_add(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// 针对float16的特化内核
__global__ void elementwise_add_half(const __half* a, const __half* b, __half* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

// 针对bfloat16的特化内核
__global__ void elementwise_add_bfloat16(const __nv_bfloat16* a, const __nv_bfloat16* b, 
                                       __nv_bfloat16* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

// 针对bool的特化内核(执行逻辑或操作)
__global__ void elementwise_add_bool(const bool* a, const bool* b, bool* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] || b[idx];
    }
}

// 带循环展开的优化内核
template <typename T, int UNROLL_FACTOR>
__global__ void elementwise_add_unrolled(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL_FACTOR;
    
    #pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        if (idx + i < n) {
            out[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// 向量化加载/存储的优化内核
template <typename T, int VECTOR_SIZE>
__global__ void elementwise_add_vectorized(const T* a, const T* b, T* out, int64_t n) {
    using VecType = typename std::conditional<VECTOR_SIZE == 2, 
                     typename std::conditional<std::is_same<T, float>::value, float2, double2>::type,
                     typename std::conditional<VECTOR_SIZE == 4, 
                     typename std::conditional<std::is_same<T, float>::value, float4, double4>::type, 
                     T>::type>::type;
    
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;
    
    if (idx + VECTOR_SIZE - 1 < n) {
        VecType a_vec = *reinterpret_cast<const VecType*>(a + idx);
        VecType b_vec = *reinterpret_cast<const VecType*>(b + idx);
        VecType out_vec;
        
        #pragma unroll
        for (int i = 0; i < VECTOR_SIZE; i++) {
            reinterpret_cast<T*>(&out_vec)[i] = 
                reinterpret_cast<T*>(&a_vec)[i] + reinterpret_cast<T*>(&b_vec)[i];
        }
        
        *reinterpret_cast<VecType*>(out + idx) = out_vec;
    } else {
        // 处理剩余元素
        for (int i = 0; i < VECTOR_SIZE && idx + i < n; i++) {
            out[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// 规约加法内核
template <typename T>
__global__ void reduce_add(const T* data, T* result, int64_t n) {
    // 使用动态共享内存
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);
    
    // 获取线程索引
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程计算部分和
    T sum = 0;
    while (i < n) {
        sum += data[i];
        i += blockDim.x * gridDim.x;
    }
    
    // 在共享内存中进行块内规约
    shared[tid] = sum;
    __syncthreads();
    
    // 使用并行规约算法
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // 第一个线程将结果原子添加到全局内存
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

// Tensor Core矩阵加法内核
template <typename T>
__global__ void tensorcore_add(const T* a, const T* b, T* out, int m, int n, int k) {
    using namespace nvcuda;
    
    const int warpSize = 32;
    const int blockRowTiles = 2;
    const int blockColTiles = 2;
    const int warpRowTiles = 2;
    const int warpColTiles = 1;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, T> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for (int i = 0; i < k; i += 16) {
        wmma::load_matrix_sync(a_frag, a + blockIdx.y * 16 * k + i * 16, k);
        wmma::load_matrix_sync(b_frag, b + i * n + blockIdx.x * 16, n);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    wmma::store_matrix_sync(out + blockIdx.y * 16 * n + blockIdx.x * 16,
                          acc_frag, n, wmma::mem_row_major);
}

// 原子加法内核
template <typename T>
__global__ void atomic_add(T* data, const T* value, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&data[idx], value[idx]);
    }
}

} // namespace kernel

// ==============================================
// 公共接口实现
// ==============================================

namespace {

// 分发函数，选择合适的内核
template <typename T>
void dispatch_elementwise_add(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    const int64_t block_size = 256;
    const int64_t grid_size = (n + block_size - 1) / block_size;
    
    if (n < 1024) {
        kernel::elementwise_add<<<grid_size, block_size, 0, stream>>>(a, b, out, n);
    } else if (n < 1000000) {
        constexpr int unroll_factor = 4;
        kernel::elementwise_add_unrolled<T, unroll_factor>
            <<<(n + block_size * unroll_factor - 1) / (block_size * unroll_factor), 
               block_size, 0, stream>>>(a, b, out, n);
    } else {
        if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
            constexpr int vector_size = std::is_same<T, float>::value ? 4 : 2;
            kernel::elementwise_add_vectorized<T, vector_size>
                <<<(n + block_size * vector_size - 1) / (block_size * vector_size), 
                   block_size, 0, stream>>>(a, b, out, n);
        } else {
            kernel::elementwise_add<<<grid_size, block_size, 0, stream>>>(a, b, out, n);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // anonymous namespace

// 主函数实现
void elementwise_add(const void* a, const void* b, void* out, int64_t n,
                    DType dtype, cudaStream_t stream) {
    switch (dtype) {
        case kFloat32:
            dispatch_elementwise_add<float>(
                reinterpret_cast<const float*>(a),
                reinterpret_cast<const float*>(b),
                reinterpret_cast<float*>(out), n, stream);
            break;
        case kFloat64:
            dispatch_elementwise_add<double>(
                reinterpret_cast<const double*>(a),
                reinterpret_cast<const double*>(b),
                reinterpret_cast<double*>(out), n, stream);
            break;
        case kFloat16:
            kernel::elementwise_add_half<<<(n + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __half*>(a),
                reinterpret_cast<const __half*>(b),
                reinterpret_cast<__half*>(out), n);
            break;
        case kBFloat16:
            kernel::elementwise_add_bfloat16<<<(n + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a),
                reinterpret_cast<const __nv_bfloat16*>(b),
                reinterpret_cast<__nv_bfloat16*>(out), n);
            break;
        case kInt32:
            dispatch_elementwise_add<int32_t>(
                reinterpret_cast<const int32_t*>(a),
                reinterpret_cast<const int32_t*>(b),
                reinterpret_cast<int32_t*>(out), n, stream);
            break;
        case kInt64:
            dispatch_elementwise_add<int64_t>(
                reinterpret_cast<const int64_t*>(a),
                reinterpret_cast<const int64_t*>(b),
                reinterpret_cast<int64_t*>(out), n, stream);
            break;
        case kUInt8:
            dispatch_elementwise_add<uint8_t>(
                reinterpret_cast<const uint8_t*>(a),
                reinterpret_cast<const uint8_t*>(b),
                reinterpret_cast<uint8_t*>(out), n, stream);
            break;
        case kBool:
            kernel::elementwise_add_bool<<<(n + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const bool*>(a),
                reinterpret_cast<const bool*>(b),
                reinterpret_cast<bool*>(out), n);
            break;
        default:
            throw std::runtime_error("Unsupported dtype in elementwise_add");
    }
}

void reduce_add(const void* data, void* result, int64_t n, DType dtype,
               cudaStream_t stream) {
    int minGridSize, blockSize;
    
    // 使用CUDA占用率API计算最优块大小
    switch (dtype) {
        case kFloat32:
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                &minGridSize, &blockSize, kernel::reduce_add<float>, 0, 0));
            break;
        case kFloat64:
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                &minGridSize, &blockSize, kernel::reduce_add<double>, 0, 0));
            break;
        case kInt32:
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                &minGridSize, &blockSize, kernel::reduce_add<int32_t>, 0, 0));
            break;
        case kInt64:
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                &minGridSize, &blockSize, kernel::reduce_add<int64_t>, 0, 0));
            break;
        default:
            throw std::runtime_error("Unsupported dtype for reduce_add");
    }
    
    // 限制块大小不超过最大线程数
    blockSize = min(blockSize, 1024);
    
    int gridSize = min((n + blockSize - 1) / blockSize, 65535);
    size_t sharedMemSize = blockSize * dtype_size(dtype);
    
    switch (dtype) {
        case kFloat32:
            kernel::reduce_add<float><<<gridSize, blockSize, sharedMemSize, stream>>>(
                reinterpret_cast<const float*>(data),
                reinterpret_cast<float*>(result), n);
            break;
        case kFloat64:
            kernel::reduce_add<double><<<gridSize, blockSize, sharedMemSize, stream>>>(
                reinterpret_cast<const double*>(data),
                reinterpret_cast<double*>(result), n);
            break;
        case kInt32:
            kernel::reduce_add<int32_t><<<gridSize, blockSize, sharedMemSize, stream>>>(
                reinterpret_cast<const int32_t*>(data),
                reinterpret_cast<int32_t*>(result), n);
            break;
        case kInt64:
            kernel::reduce_add<int64_t><<<gridSize, blockSize, sharedMemSize, stream>>>(
                reinterpret_cast<const int64_t*>(data),
                reinterpret_cast<int64_t*>(result), n);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for reduce_add");
    }
    CUDA_CHECK(cudaGetLastError());
}

void tensorcore_add(const void* a, const void* b, void* out,
                   int m, int n, int k, DType dtype,
                   cudaStream_t stream) {
    dim3 gridDim((n + 15) / 16, (m + 15) / 16);
    dim3 blockDim(32, 4);
    
    switch (dtype) {
        case kFloat16:
            kernel::tensorcore_add<__half><<<gridDim, blockDim, 0, stream>>>(
                reinterpret_cast<const __half*>(a),
                reinterpret_cast<const __half*>(b),
                reinterpret_cast<__half*>(out), m, n, k);
            break;
        case kBFloat16:
            kernel::tensorcore_add<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(a),
                reinterpret_cast<const __nv_bfloat16*>(b),
                reinterpret_cast<__nv_bfloat16*>(out), m, n, k);
            break;
        default:
            throw std::runtime_error("Tensor Core only supports float16 and bfloat16");
    }
    CUDA_CHECK(cudaGetLastError());
}

void atomic_add(void* data, const void* value, int64_t n, DType dtype,
               cudaStream_t stream) {
    int blockSize;
    int minGridSize;
    
    switch (dtype) {
        case kFloat32:
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                &minGridSize, &blockSize, kernel::atomic_add<float>, 0, 0));
            break;
        case kInt32:
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                &minGridSize, &blockSize, kernel::atomic_add<int32_t>, 0, 0));
            break;
        // ... 其他类型类似处理 ...
    }
    
    int gridSize = (n + blockSize - 1) / blockSize;
    
    switch (dtype) {
        case kFloat32:
            kernel::atomic_add<float><<<gridSize, blockSize, 0, stream>>>(
                reinterpret_cast<float*>(data),
                reinterpret_cast<const float*>(value), n);
            break;
        case kInt32:
            kernel::atomic_add<int32_t><<<gridSize, blockSize, 0, stream>>>(
                reinterpret_cast<int32_t*>(data),
                reinterpret_cast<const int32_t*>(value), n);
            break;
        // ... 其他类型类似处理 ...
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace core