#include <numeric>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "tensor_impl.h"
#include "core/autograd/engine.h"
#include "core/autograd/function/base.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace core {

//=== 构造函数实现 ===//
TensorImpl::TensorImpl(const std::vector<int64_t>& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device), 
      memory_ownership_(MemoryOwnership::OWNED) {
    check_shape(shape);
    compute_strides();
    allocate();
}

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, void* data, DType dtype, 
                     Device device, MemoryOwnership ownership)
    : shape_(shape), data_(data), dtype_(dtype), device_(device),
      memory_ownership_(ownership) {
    check_shape(shape);
    if (data == nullptr) {
        throw_error("Data pointer cannot be null");
    }
    compute_strides();
    nbytes_ = numel() * dtype_size(dtype_);
}

//=== 析构函数实现 ===//
TensorImpl::~TensorImpl() {
    try {
        if (data_ != nullptr && memory_ownership_ == MemoryOwnership::OWNED) {
            deallocate();
        }
    } catch (...) {
        std::cerr << "Error during tensor deallocation" << std::endl;
    }
}

void TensorImpl::fill_(float value) {
    // 实现填充逻辑，根据设备类型调用不同的实现
    if (device_.type() == DeviceType::kCPU) {
        // CPU 实现
        float* data = static_cast<float*>(data_);
        std::fill(data, data + numel(), value);
    }
    #ifdef WITH_CUDA
    else if (device_.type() == DeviceType::kCUDA) {
        // CUDA 实现
        cuda_fill(data_, value, numel());
    }
    #endif
}

//=== 内存管理 ===//
void TensorImpl::allocate(AllocStrategy strategy, size_t alignment) {
    if (data_ != nullptr) {
        throw_error("Tensor already has allocated memory");
    }
    
    nbytes_ = numel() * dtype_size(dtype_);
    if (nbytes_ == 0) return;

    try {
        switch (device_.type()) {
            case DeviceType::kCPU:
                cpu_allocate(alignment);
                break;
            case DeviceType::kCUDA:
                #ifdef WITH_CUDA
                cuda_allocate(strategy == AllocStrategy::PINNED);
                #else
                throw_error("CUDA support not compiled");
                #endif
                break;
            default:
                throw_error("Unsupported device type");
        }
    } catch (const std::exception& e) {
        throw_error(std::string("Allocation failed: ") + e.what());
    }
}

void TensorImpl::deallocate() {
    if (data_ == nullptr) return;
    
    try {
        switch (device_.type()) {
            case DeviceType::kCPU:
                cpu_deallocate();
                break;
            case DeviceType::kCUDA:
                #ifdef WITH_CUDA
                cuda_deallocate();
                #else
                throw_error("CUDA support not compiled");
                #endif
                break;
            default:
                throw_error("Unsupported device type");
        }
    } catch (const std::exception& e) {
        std::cerr << "Deallocation error: " << e.what() << std::endl;
    }
    
    data_ = nullptr;
    nbytes_ = 0;
    alignment_ = 0;
    is_pinned_ = false;
}

//=== CPU内存管理 ===//
void TensorImpl::cpu_allocate(size_t alignment) {
    if (alignment > 0) {
        #if defined(_WIN32)
        data_ = _aligned_malloc(nbytes_, alignment);
        #else
        data_ = aligned_alloc(alignment, nbytes_);
        #endif
    } else {
        data_ = malloc(nbytes_);
    }

    if (data_ == nullptr) throw std::bad_alloc();
    std::memset(data_, 0, nbytes_);
    alignment_ = alignment;
}

void TensorImpl::cpu_deallocate() {
    if (data_ != nullptr) {
        if (alignment_ > 0) {
            #if defined(_WIN32)
            _aligned_free(data_);
            #else
            free(data_);
            #endif
        } else {
            free(data_);
        }
    }
}

//=== 数据操作 ===//
void TensorImpl::copy_data_from(const TensorImpl& other) {
    if (shape_ != other.shape_) {
        throw_error("Shape mismatch in copy_data_from");
    }
    
    if (dtype_ != other.dtype_) {
        throw_error("DType mismatch in copy_data_from");
    }
    
    if (data_ == nullptr) {
        allocate();
    }

    // 同设备拷贝
    if (device_ == other.device_) {
        if (device_.type() == DeviceType::kCPU) {
            std::memcpy(data_, other.data_, nbytes_);
        }
        #ifdef WITH_CUDA
        else if (device_.type() == DeviceType::kCUDA) {
            cudaError_t err = cudaMemcpy(data_, other.data_, nbytes_, 
                                       cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                throw_error(cudaGetErrorString(err));
            }
        }
        #endif
    }
    // 跨设备拷贝
    else {
        #ifdef WITH_CUDA
        if (device_.type() == DeviceType::kCPU && other.device_.type() == DeviceType::kCUDA) {
            cudaError_t err = cudaMemcpy(data_, other.data_, nbytes_,
                                        cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw_error(cudaGetErrorString(err));
            }
        }
        else if (device_.type() == DeviceType::kCUDA && other.device_.type() == DeviceType::kCPU) {
            cudaError_t err = cudaMemcpy(data_, other.data_, nbytes_,
                                        cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw_error(cudaGetErrorString(err));
            }
        }
        else {
            throw_error("Unsupported cross-device copy");
        }
        #else
        throw_error("Cross-device copy requires CUDA support");
        #endif
    }
}

//=== 视图操作 ===//
std::shared_ptr<TensorImpl> TensorImpl::create_view(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int64_t offset) {
    
    // 参数验证
    if (shape.size() != strides.size()) {
        throw_error("Shape and strides must have same dimensions");
    }
    
    // 创建视图
    auto view = std::make_shared<TensorImpl>(shape, data_, dtype_, device_, offset);
    view->strides_ = strides;
    view->offset_ = offset; 
    view->update_contiguity();
    view->memory_ownership_ = MemoryOwnership::BORROWED;
    
    return view;
}

void TensorImpl::copy_to_device(Device target_device) {
    if (device_ == target_device) {
        return; // 同设备无需拷贝
    }

    // 创建目标设备上的新tensor
    auto target_tensor = std::make_shared<TensorImpl>(shape_, dtype_, target_device);
    
    // 执行设备间拷贝
    if (device_.type() == DeviceType::kCPU && target_device.type() == DeviceType::kCUDA) {
#ifdef WITH_CUDA
        // CPU -> CUDA
        cudaError_t err = cudaMemcpy(target_tensor->data_ptr(), data_, 
                                   nbytes_, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw_error(std::string("CPU->CUDA copy failed: ") + cudaGetErrorString(err));
        }
#else
        throw_error("CUDA support not compiled");
#endif
    } 
    else if (device_.type() == DeviceType::kCUDA && target_device.type() == DeviceType::kCPU) {
#ifdef WITH_CUDA
        // CUDA -> CPU
        cudaError_t err = cudaMemcpy(target_tensor->data_ptr(), data_, 
                                   nbytes_, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw_error(std::string("CUDA->CPU copy failed: ") + cudaGetErrorString(err));
        }
#else
        throw_error("CUDA support not compiled");
#endif
    }
    else {
        throw_error("Unsupported device transfer");
    }

    // 更新当前tensor状态
    if (memory_ownership_ == MemoryOwnership::OWNED) {
        deallocate();
    }
    data_ = target_tensor->data_ptr();
    device_ = target_device;
    memory_ownership_ = MemoryOwnership::BORROWED; // 现在借用新tensor的内存
}


void TensorImpl::convert_dtype(DType target_dtype) {
    if (dtype_ == target_dtype) return;
    
    // 创建新tensor
    auto new_tensor = std::make_shared<TensorImpl>(shape_, target_dtype, device_);
    
    // 类型转换逻辑
    if (device_.type() == DeviceType::kCPU) {
        // CPU实现
        if (dtype_ == kFloat32 && target_dtype == kFloat64) {
            const float* src = static_cast<const float*>(data_);
            double* dst = static_cast<double*>(new_tensor->data_ptr());
            for (int64_t i = 0; i < numel(); ++i) {
                dst[i] = static_cast<double>(src[i]);
            }
        }
        // 其他类型转换...
    }
#ifdef WITH_CUDA
    else if (device_.type() == DeviceType::kCUDA) {
        // CUDA实现
        cuda_convert_dtype(data_, new_tensor->data_ptr(), 
                          numel(), dtype_, target_dtype);
    }
#endif
    
    // 替换当前tensor数据
    if (memory_ownership_ == MemoryOwnership::OWNED) {
        deallocate();
    }
    data_ = new_tensor->data_ptr();
    dtype_ = target_dtype;
    nbytes_ = numel() * dtype_size(dtype_);
    memory_ownership_ = MemoryOwnership::BORROWED; // 现在借用新tensor的内存
}


std::shared_ptr<TensorImpl> TensorImpl::clone() const {
    auto copy = std::make_shared<TensorImpl>(shape_, dtype_, device_);
    // 手动复制数据（跳过不可复制的成员）
    if (device_.type() == DeviceType::kCPU) {
        std::memcpy(copy->data_ptr(), data_ptr(), numel() * dtype_size(dtype_));
    }
#ifdef WITH_CUDA
    else if (device_.type() == DeviceType::kCUDA) {
        CUDA_CHECK(cudaMemcpy(
            copy->data_ptr(), data_ptr(),
            numel() * dtype_size(dtype_),
            cudaMemcpyDeviceToDevice
        ));
    }
#endif
    // 复制其他属性
    copy->set_requires_grad(requires_grad_);
    if (grad_) {
        copy->set_grad(grad_->clone()); // 递归克隆梯度
    }
    return copy;
}

//=== 形状操作 ===//
void TensorImpl::reshape(const std::vector<int64_t>& new_shape) {
    std::lock_guard<std::mutex> lock(shape_mutex_);
    check_shape(new_shape);
    
    int64_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int64_t>());
    if (new_numel != numel()) {
        throw_error("Shape size mismatch in reshape");
    }
    
    shape_ = new_shape;
    compute_strides();
}

void TensorImpl::set_shape(const std::vector<int64_t>& new_shape) {
    std::lock_guard<std::mutex> lock(shape_mutex_);
    check_shape(new_shape);
    shape_ = new_shape;
    compute_strides();
}

void TensorImpl::set_strides(const std::vector<int64_t>& new_strides) {
    std::lock_guard<std::mutex> lock(shape_mutex_);
    if (new_strides.size() != shape_.size()) {
        throw_error("Strides size must match shape size");
    }
    strides_ = new_strides;
    update_contiguity();
}

void TensorImpl::set_shape_and_strides(const std::vector<int64_t>& new_shape,
                                     const std::vector<int64_t>& new_strides) {
    std::lock_guard<std::mutex> lock(shape_mutex_);
    check_shape(new_shape);
    if (new_strides.size() != new_shape.size()) {
        throw_error("Strides size must match shape size");
    }
    shape_ = new_shape;
    strides_ = new_strides;
    update_contiguity();
}

void TensorImpl::set_offset(int64_t offset) {
    offset_ = offset;
}

void TensorImpl::set_contiguous(bool contiguous) {
    is_contiguous_ = contiguous;
}

//=== 数学操作 ===//
void TensorImpl::add_impl(const TensorImpl& other) {
    if (shape_ != other.shape_) {
        throw_error("Shape mismatch in add_impl");
    }
    
    if (device_.type() != DeviceType::kCPU || other.device_.type() != DeviceType::kCPU) {
        throw_error("add_impl only implemented on CPU");
    }
    
    if (dtype_ != kFloat32 || other.dtype_ != kFloat32) {
        throw_error("add_impl only implemented for float32");
    }
    
    float* dst = static_cast<float*>(data_);
    const float* src = static_cast<const float*>(other.data_);
    
    for (int64_t i = 0; i < numel(); ++i) {
        dst[i] += src[i];
    }
}

void TensorImpl::matmul_impl(const TensorImpl& a, const TensorImpl& b) {
    // 形状检查
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw_error("matmul_impl only supports 2D tensors");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw_error("Matrix dimension mismatch in matmul");
    }
    
    // 设备检查
    if (a.device() != b.device() || a.device() != device_) {
        throw_error("All tensors must be on same device");
    }
    
    // 类型检查
    if (a.dtype() != b.dtype() || a.dtype() != dtype_) {
        throw_error("All tensors must have same dtype");
    }
    
    // CPU实现
    if (device_.type() == DeviceType::kCPU) {
        if (dtype_ == kFloat32) {
            // 调用BLAS或其他矩阵乘法实现
            // 这里简化实现
            float* out = static_cast<float*>(data_);
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            
            // 朴素矩阵乘法
            for (int64_t i = 0; i < a.shape()[0]; ++i) {
                for (int64_t k = 0; k < a.shape()[1]; ++k) {
                    for (int64_t j = 0; j < b.shape()[1]; ++j) {
                        out[i * b.shape()[1] + j] += 
                            a_data[i * a.shape()[1] + k] * 
                            b_data[k * b.shape()[1] + j];
                    }
                }
            }
        } else {
            throw_error("Unsupported dtype for CPU matmul");
        }
    }
#ifdef WITH_CUDA
    // CUDA实现
    else if (device_.type() == DeviceType::kCUDA) {
        cuda_matmul(a.data_ptr(), b.data_ptr(), data_,
                   a.shape()[0], a.shape()[1], b.shape()[1],
                   dtype_);
    }
#endif
    else {
        throw_error("Unsupported device for matmul");
    }
}


//=== 自动微分 ===//
void TensorImpl::zero_() {
    if (data_ == nullptr) return;

    switch (device_.type()) {
        case DeviceType::kCPU:
            std::memset(data_, 0, nbytes_);
            break;
        case DeviceType::kCUDA:
            #ifdef WITH_CUDA
            cudaMemset(data_, 0, nbytes_);
            #else
            throw_error("CUDA support not compiled");
            #endif
            break;
        default:
            throw_error("Unsupported device type for zero_ operation");
    }
}
void TensorImpl::accumulate_grad(const TensorImpl& grad) {
    if (!requires_grad_) return;
    
    if (shape_ != grad.shape_) throw_error("Gradient shape mismatch");
    if (device_ != grad.device_) throw_error("Gradient device mismatch");
    if (dtype_ != grad.dtype_) throw_error("Gradient dtype mismatch");
    
    if (!grad_) {
        grad_ = std::make_shared<TensorImpl>(shape_, dtype_, device_);
        grad_->zero_();
    }
    grad_->add_impl(grad);
}

void TensorImpl::zero_grad() {
    if (grad_) {
        grad_->zero_(); 
    }
}
void TensorImpl::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
}

bool TensorImpl::requires_grad() const {
    return requires_grad_;
}

std::shared_ptr<TensorImpl> TensorImpl::grad() const {
    return grad_;
}

void TensorImpl::set_grad(std::shared_ptr<TensorImpl> grad) {
    grad_ = grad;
}


void TensorImpl::backward(const std::shared_ptr<TensorImpl>& grad) {
    if (!grad_fn_) return;
    
    if (!grad) {
        // 创建全1梯度
        grad_ = TensorImpl::create_autograd_aware(shape_, dtype_, device_);
        grad_->fill_(1.0f);  // 现在 fill_ 方法已定义
    } else {
        grad_ = grad;
    }
    
    autograd::Engine::get_default_engine().execute({grad_fn_});
}

std::shared_ptr<autograd::Function> TensorImpl::grad_fn() const {
    return grad_fn_;
}

void TensorImpl::set_grad_fn(const std::shared_ptr<autograd::Function>& grad_fn) {
    grad_fn_ = grad_fn;
}
std::shared_ptr<TensorImpl> TensorImpl::create_autograd_aware(
    const std::vector<int64_t>& shape,
    DType dtype,
    Device device,
    bool requires_grad) {
    auto impl = std::make_shared<TensorImpl>(shape, dtype, device);
    impl->requires_grad_ = requires_grad;
    return impl;
}

//=== 调试工具 ===//
std::string TensorImpl::debug_info() const {
    std::ostringstream oss;
    oss << "shape=" << shape_to_string(shape_)
        << ", dtype=" << dtype_name(dtype_)
        << ", device=" << device_.str()
        << ", contig=" << (is_contiguous_ ? "true" : "false")
        << ", owns_mem=" << (memory_ownership_ == MemoryOwnership::OWNED ? "true" : "false");
    return oss.str();
}

//=== 工具函数 ===//
std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

//=== 内部实现 ===//
void TensorImpl::compute_strides() {
    if (shape_.empty()) {
        strides_.clear();
        is_contiguous_ = true;
        return;
    }

    strides_.resize(shape_.size());
    
    // 按C连续内存布局计算步幅
    strides_.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
    
    is_contiguous_ = true;
}
bool TensorImpl::check_contiguous() const {
    if (shape_.empty() || strides_.empty()) {
        return true;
    }

    // 检查是否符合C连续条件
    int64_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (strides_[i] != stride) {
            return false;
        }
        stride *= shape_[i];
    }
    
    return true;
}
void TensorImpl::check_shape(const std::vector<int64_t>& shape) const {
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw_error("Negative dimension size not allowed");
        }
    }
    
    // 检查总元素数是否合理
    int64_t numel = 1;
    for (int64_t dim : shape) {
        if (dim > 0 && numel > std::numeric_limits<int64_t>::max() / dim) {
            throw_error("Shape too large, would cause integer overflow");
        }
        numel *= dim;
    }
}
void TensorImpl::update_contiguity() {
    if (shape_.empty()) {
        is_contiguous_ = true;
        return;
    }

    // 检查步幅是否单调递减且无间隔
    is_contiguous_ = true;
    int64_t expected_stride = 1;
    
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (strides_[i] != expected_stride) {
            is_contiguous_ = false;
            return;
        }
        if (shape_[i] == 0) {
            // 零尺寸的维度不影响连续性
            continue;
        }
        expected_stride *= shape_[i];
    }
}


//=== CUDA实现 ===//
#ifdef WITH_CUDA

void TensorImpl::cuda_allocate(bool pinned) {
    cudaError_t err;
    if (pinned) {
        err = cudaMallocHost(&data_, nbytes_);
        is_pinned_ = true;
    } else {
        err = cudaMalloc(&data_, nbytes_);
    }
    
    if (err != cudaSuccess) {
        throw_error(cudaGetErrorString(err));
    }
    
    // 初始化内存
    err = cudaMemset(data_, 0, nbytes_);
    if (err != cudaSuccess) {
        cuda_deallocate();
        throw_error(cudaGetErrorString(err));
    }
}

void TensorImpl::cuda_deallocate() {
    if (data_ != nullptr) {
        cudaError_t err;
        if (is_pinned_) {
            err = cudaFreeHost(data_);
        } else {
            err = cudaFree(data_);
        }
        
        if (err != cudaSuccess) {
            throw_error(cudaGetErrorString(err));
        }
    }
}

void TensorImpl::cuda_copy_from_host(const void* src) {
    cudaError_t err = cudaMemcpy(data_, src, nbytes_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw_error(cudaGetErrorString(err));
    }
}

void TensorImpl::cuda_copy_to_host(void* dst) const {
    cudaError_t err = cudaMemcpy(dst, data_, nbytes_, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw_error(cudaGetErrorString(err));
    }
}

#endif // WITH_CUDA

//=== 错误处理 ===//
[[noreturn]] void TensorImpl::throw_error(const std::string& msg) const {
    std::string full_msg = msg + "\nTensor metadata: " + debug_info();
    throw std::runtime_error(full_msg);
}

} // namespace core