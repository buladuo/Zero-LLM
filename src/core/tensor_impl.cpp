#include "tensor_impl.h"
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <algorithm>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace core {

// 构造函数实现
TensorImpl::TensorImpl(const std::vector<int64_t>& shape, 
                     DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    // 验证形状有效性
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty");
    }
    
    for (auto dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("Negative dimension in shape");
        }
    }
    
    compute_strides();
    allocate();
}

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, 
                     void* data, DType dtype, Device device)
    : shape_(shape), data_(data), dtype_(dtype), device_(device) {
    // 验证输入参数
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty");
    }
    
    if (data == nullptr) {
        throw std::invalid_argument("Data pointer cannot be null");
    }
    
    for (auto dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("Negative dimension in shape");
        }
    }
    
    compute_strides();
    nbytes_ = numel() * dtype_size(dtype_);
}
TensorImpl::TensorImpl(const std::vector<int64_t>& shape, 
                     void* data, DType dtype, Device device,
                     int64_t offset = 0)
    : shape_(shape), data_(data), dtype_(dtype), device_(device), offset_(offset) {
    compute_strides();
}

// 析构函数实现
TensorImpl::~TensorImpl() {
    try {
        if (data_ != nullptr) {
            deallocate();
        }
    } catch (...) {
        // 确保析构函数不会抛出异常
        std::cerr << "Error during tensor deallocation" << std::endl;
    }
}

// 内存管理实现
void TensorImpl::allocate() {
    if (data_ != nullptr) {
        throw std::runtime_error("Tensor already has allocated memory");
    }
    
    nbytes_ = numel() * dtype_size(dtype_);
    if (nbytes_ == 0) {
        return;  // 空张量不需要分配内存
    }
    
    try {
        switch (device_.type()) {
            case kCPU:
                cpu_allocate();
                break;
            case kCUDA:
                #ifdef WITH_CUDA
                cuda_allocate();
                #else
                throw std::runtime_error("CUDA support not compiled");
                #endif
                break;
            default:
                throw std::runtime_error("Unsupported device type");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Allocation failed: ") + e.what());
    }
}

void TensorImpl::deallocate() {
    if (data_ == nullptr) return;
    
    try {
        switch (device_.type()) {
            case kCPU:
                cpu_deallocate();
                break;
            case kCUDA:
                #ifdef WITH_CUDA
                cuda_deallocate();
                #else
                throw std::runtime_error("CUDA support not compiled");
                #endif
                break;
            default:
                throw std::runtime_error("Unsupported device type");
        }
    } catch (const std::exception& e) {
        std::cerr << "Deallocation error: " << e.what() << std::endl;
    }
    
    data_ = nullptr;
    nbytes_ = 0;
}

// CPU内存管理实现
void TensorImpl::cpu_allocate() {
    data_ = malloc(nbytes_);
    if (data_ == nullptr) {
        throw std::bad_alloc();
    }
    
    // 初始化内存为0
    std::memset(data_, 0, nbytes_);
}

void TensorImpl::cpu_deallocate() {
    if (data_ != nullptr) {
        free(data_);
    }
}

// 数据拷贝实现
void TensorImpl::copy_data_from(const TensorImpl& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch in copy_data_from");
    }
    
    if (dtype_ != other.dtype_) {
        throw std::invalid_argument("DType mismatch in copy_data_from");
    }
    
    if (data_ == nullptr) {
        allocate();
    }
    
    try {
        // 同设备拷贝
        if (device_ == other.device_) {
            if (device_ == kCPU) {
                std::memcpy(data_, other.data_, nbytes_);
            }
            #ifdef WITH_CUDA
            else if (device_ == kCUDA) {
                cudaError_t err = cudaMemcpy(data_, other.data_, nbytes_, 
                                           cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(err));
                }
            }
            #endif
        }
        // 跨设备拷贝
        else {
            #ifdef WITH_CUDA
            if (device_ == kCPU && other.device_ == kCUDA) {
                other.cuda_copy_to_host(data_);
            }
            else if (device_ == kCUDA && other.device_ == kCPU) {
                cuda_copy_from_host(other.data_);
            }
            else {
                throw std::runtime_error("Unsupported cross-device copy");
            }
            #else
            throw std::runtime_error("Cross-device copy requires CUDA support");
            #endif
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Copy failed: ") + e.what());
    }
}

// 移动数据实现
void TensorImpl::move_data_from(TensorImpl&& other) {
    if (this == &other) return;
    
    if (data_ != nullptr) {
        deallocate();
    }
    
    data_ = other.data_;
    nbytes_ = other.nbytes_;
    device_ = other.device_;
    
    other.data_ = nullptr;
    other.nbytes_ = 0;
}

// 形状计算实现
void TensorImpl::compute_strides() {
    strides_.resize(shape_.size());
    
    if (shape_.empty()) {
        return;
    }

    // 计算行优先(C-style)步长
    strides_.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
    
    // 更新连续性标志
    update_contiguity();
}

bool TensorImpl::check_contiguous() const {
    if (shape_.empty()) return true;
    
    int64_t expected_stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (strides_[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape_[i];
    }
    return true;
}

void TensorImpl::check_shape(const std::vector<int64_t>& shape) const {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty");
    }
    
    for (auto dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("Negative dimension in shape");
        }
    }
}

// CUDA实现
#ifdef WITH_CUDA

bool TensorImpl::is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return false;
    }
    return count > 0;
}

void TensorImpl::cuda_allocate() {
    cudaError_t err = cudaMalloc(&data_, nbytes_);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // 初始化设备内存为0
    err = cudaMemset(data_, 0, nbytes_);
    if (err != cudaSuccess) {
        cudaFree(data_);
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void TensorImpl::cuda_deallocate() {
    if (data_ != nullptr) {
        cudaError_t err = cudaFree(data_);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}

void TensorImpl::cuda_copy_from_host(const void* src) {
    cudaError_t err = cudaMemcpy(data_, src, nbytes_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void TensorImpl::cuda_copy_to_host(void* dst) const {
    cudaError_t err = cudaMemcpy(dst, data_, nbytes_, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

#endif // WITH_CUDA

} // namespace core