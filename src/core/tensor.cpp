#include "tensor.h"
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace core {

// 构造函数和析构函数
Tensor::Tensor() : impl_(std::make_shared<TensorImpl>(std::vector<int64_t>{}, kFloat32, Device::CPU())) {}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, Device device)
    : impl_(std::make_shared<TensorImpl>(shape, dtype, device)) {}

Tensor::Tensor(const std::vector<int64_t>& shape, void* data, DType dtype, Device device)
    : impl_(std::make_shared<TensorImpl>(shape, data, dtype, device)) {}

Tensor::Tensor(const Tensor& other) : impl_(other.impl_) {}

Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

Tensor::~Tensor() = default;

inline Tensor make_tensor_from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(impl);  // 调用私有构造函数
}

// 赋值操作符
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        impl_ = other.impl_;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

// 基本属性访问
int64_t Tensor::dim() const { return impl_->shape().size(); }
const std::vector<int64_t>& Tensor::shape() const { return impl_->shape(); }
int64_t Tensor::size() const { return impl_->numel(); }
int64_t Tensor::numel() const { return impl_->numel(); }
DType Tensor::dtype() const { return impl_->dtype(); }
Device Tensor::device() const { return impl_->device(); }
bool Tensor::is_contiguous() const { return impl_->is_contiguous(); }
void* Tensor::data_ptr() { return impl_->data_ptr(); }
const void* Tensor::data_ptr() const { return impl_->data_ptr(); }

// 连续化方法
Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    
    // 创建连续副本
    Tensor result(shape(), dtype(), device());
    
    // 实现数据拷贝
    if (device().type() == kCPU) {
        std::memcpy(result.data_ptr(), data_ptr(), numel() * dtype_size(dtype()));
    }
    #ifdef WITH_CUDA
    else if (device().type() == kCUDA) {
        cudaMemcpy(result.data_ptr(), data_ptr(), 
                 numel() * dtype_size(dtype()),
                 cudaMemcpyDeviceToDevice);
    }
    #endif
    
    return result;
}


// 设备转换
Tensor Tensor::to(Device device) const {
    if (device == impl_->device()) {
        return *this;
    }

    Tensor result(shape(), dtype(), device);
    
    size_t nbytes = numel() * dtype_size(dtype());
    
    // 跨设备拷贝实现
    if (device.type() == kCPU && impl_->device().type() == kCUDA) {
        #ifdef WITH_CUDA
        cudaMemcpy(result.data_ptr(), data_ptr(), nbytes, cudaMemcpyDeviceToHost);
        #else
        throw std::runtime_error("CUDA support not compiled");
        #endif
    } 
    else if (device.type() == kCUDA && impl_->device().type() == kCPU) {
        #ifdef WITH_CUDA
        cudaMemcpy(result.data_ptr(), data_ptr(), nbytes, cudaMemcpyHostToDevice);
        #else
        throw std::runtime_error("CUDA support not compiled");
        #endif
    } 
    else {
        throw std::runtime_error("Unsupported device transfer");
    }
    
    return result;
}

Tensor Tensor::cpu() const { return to(Device::CPU()); }
Tensor Tensor::cuda() const { return to(Device::CUDA()); }

Tensor Tensor::to(DType dtype) const {
    if (dtype == impl_->dtype()) {
        return *this;
    }

    Tensor result(shape(), dtype, device());
    
    // 简单实现：仅支持CPU上的类型转换
    if (device().type() != kCPU) {
        throw std::runtime_error("Type conversion only supported on CPU");
    }
    
    // 获取源和目标数据类型大小
    size_t src_size = dtype_size(impl_->dtype());
    size_t dst_size = dtype_size(dtype);
    
    // 实现各种类型转换组合
    if (impl_->dtype() == kFloat32 && dtype == kFloat64) {
        const float* src = static_cast<const float*>(data_ptr());
        double* dst = static_cast<double*>(result.data_ptr());
        for (int64_t i = 0; i < numel(); ++i) {
            dst[i] = static_cast<double>(src[i]);
        }
    }
    else if (impl_->dtype() == kFloat64 && dtype == kFloat32) {
        const double* src = static_cast<const double*>(data_ptr());
        float* dst = static_cast<float*>(result.data_ptr());
        for (int64_t i = 0; i < numel(); ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
    }
    else if (impl_->dtype() == kFloat32 && dtype == kFloat16) {
        const float* src = static_cast<const float*>(data_ptr());
        uint16_t* dst = static_cast<uint16_t*>(result.data_ptr());
        // 实现float32到float16的转换 (这里简化处理)
        for (int64_t i = 0; i < numel(); ++i) {
            dst[i] = static_cast<uint16_t>(src[i]); // 实际需要更精确的转换
        }
    }
    else if (impl_->dtype() == kFloat16 && dtype == kFloat32) {
        const uint16_t* src = static_cast<const uint16_t*>(data_ptr());
        float* dst = static_cast<float*>(result.data_ptr());
        // 实现float16到float32的转换 (这里简化处理)
        for (int64_t i = 0; i < numel(); ++i) {
            dst[i] = static_cast<float>(src[i]); // 实际需要更精确的转换
        }
    }
    else {
        throw std::runtime_error(std::string("Unsupported type conversion from ") + 
                               dtype_name(impl_->dtype()) + " to " + dtype_name(dtype));
    }
    
    return result;
}

Tensor Tensor::float32() const { return to(kFloat32); }
Tensor Tensor::float16() const { return to(kFloat16); }
Tensor Tensor::bfloat16() const { return to(kBFloat16); }

// 形状操作
Tensor Tensor::view(const std::vector<int64_t>& shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("View is only supported for contiguous tensors");
    }

    int64_t new_numel = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    if (new_numel != numel()) {
        throw std::runtime_error("Shape size mismatch in view operation");
    }

    // 使用公共接口创建新视图
    auto new_impl = std::make_shared<TensorImpl>(shape, impl_->data_ptr(), dtype(), device());
    return make_tensor_from_impl(new_impl);
}

Tensor Tensor::reshape(const std::vector<int64_t>& shape) const {
    // 1. 检查形状有效性
    int64_t new_numel = 1;
    for (auto dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("Negative dimension in reshape");
        }
        new_numel *= dim;
    }

    // 2. 检查元素数量匹配
    if (new_numel != numel()) {
        throw std::runtime_error(std::string("Shape size mismatch in reshape: ") +
                              std::to_string(new_numel) + " vs " + std::to_string(numel()));
    }

    // 3. 处理连续张量
    if (is_contiguous()) {
        return view(shape);
    }

    // 4. 非连续张量需要创建副本
    Tensor contig = contiguous();
    return contig.view(shape);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    auto ndim = dim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::runtime_error("Invalid dimension for transpose");
    }
    
    auto new_shape = shape();
    auto new_strides = impl_->strides();  // 使用公共方法获取strides
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    
    auto new_impl = std::make_shared<TensorImpl>(*impl_);
    new_impl->set_shape_and_strides(new_shape, new_strides);
    
    return make_tensor_from_impl(new_impl);
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    auto ndim = dim();
    if (dims.size() != ndim) {
        throw std::runtime_error("Permute dimensions must match tensor rank");
    }
    
    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);
    const auto& old_strides = impl_->strides();
    
    for (size_t i = 0; i < dims.size(); ++i) {
        auto src_dim = dims[i];
        if (src_dim < 0) src_dim += ndim;
        if (src_dim < 0 || src_dim >= ndim) {
            throw std::runtime_error("Invalid dimension in permute");
        }
        new_shape[i] = shape()[src_dim];
        new_strides[i] = old_strides[src_dim];
    }
    
    auto new_impl = std::make_shared<TensorImpl>(*impl_);
    new_impl->set_shape_and_strides(new_shape, new_strides);
    
    return make_tensor_from_impl(new_impl);
}

Tensor Tensor::squeeze(int64_t dim) const {
    const auto& old_shape = shape();
    std::vector<int64_t> new_shape;
    
    if (dim == -1) {
        // 移除所有长度为1的维度
        std::copy_if(old_shape.begin(), old_shape.end(),
                    std::back_inserter(new_shape),
                    [](int64_t d) { return d != 1; });
    } else {
        // 检查指定维度
        const int64_t tensor_dim = this->dim();
        if (dim < 0) dim += tensor_dim;
        
        if (dim < 0 || dim >= tensor_dim) {
            throw std::runtime_error("Invalid dimension " + std::to_string(dim) + 
                                   " for squeeze with tensor dimension " + 
                                   std::to_string(tensor_dim));
        }
        
        if (old_shape[dim] != 1) {
            throw std::runtime_error("Can only squeeze dimension with size 1, but got " + 
                                   std::to_string(old_shape[dim]) + " at dim " + 
                                   std::to_string(dim));
        }
        
        new_shape = old_shape;
        new_shape.erase(new_shape.begin() + dim);
    }
    
    if (new_shape.empty()) {
        // 处理所有维度都被squeeze的情况
        new_shape.push_back(1);
    }
    
    return view(new_shape);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    // 获取当前维度数
    const int64_t current_dim = this->dim();
    
    // 处理负索引 (Python风格)
    if (dim < 0) dim += current_dim + 1;
    
    // 验证维度范围
    if (dim < 0 || dim > current_dim) {
        throw std::runtime_error(
            "Dimension out of range (expected to be in range [" + 
            std::to_string(-(current_dim + 1)) + ", " + 
            std::to_string(current_dim) + "], but got " + 
            std::to_string(dim) + ")"
        );
    }
    
    // 创建新形状
    auto new_shape = shape();
    new_shape.insert(new_shape.begin() + dim, 1);
    
    // 处理空张量特殊情况
    if (current_dim == 0) {
        return Tensor(new_shape, dtype(), device());
    }
    
    return view(new_shape);
}

// 索引操作
Tensor Tensor::operator[](int64_t index) const {
    return select(0, index);
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    // 参数验证
    const int64_t current_dim = this->dim();
    if (dim < 0) dim += current_dim;
    if (dim < 0 || dim >= current_dim) {
        throw std::runtime_error("Invalid dimension in select");
    }
    
    const int64_t dim_size = shape()[dim];
    if (index < 0) index += dim_size;
    if (index < 0 || index >= dim_size) {
        throw std::runtime_error("Index out of range in select");
    }
    
    // 创建新形状
    std::vector<int64_t> new_shape = shape();
    new_shape.erase(new_shape.begin() + dim);
    
    // 创建新实现对象
    auto new_impl = std::make_shared<TensorImpl>(*impl_);
    new_impl->set_shape(new_shape);
    new_impl->set_offset(impl_->offset() + index * impl_->strides()[dim]);
    
    return make_tensor_from_impl(new_impl);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    // 获取当前维度数和形状
    const int64_t current_dim = this->dim();
    const auto& current_shape = shape();
    
    // 处理负维度索引
    if (dim < 0) dim += current_dim;
    
    // 验证维度范围
    if (dim < 0 || dim >= current_dim) {
        throw std::runtime_error(
            "Dimension out of range (expected to be in range [" +
            std::to_string(-current_dim) + ", " +
            std::to_string(current_dim - 1) + "], but got " +
            std::to_string(dim) + ")"
        );
    }
    
    // 获取指定维度的长度
    const int64_t dim_size = current_shape[dim];
    
    // 处理负索引
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    
    // 边界检查
    start = std::max<int64_t>(0, std::min(start, dim_size));
    end = std::max<int64_t>(0, std::min(end, dim_size));
    
    // 验证参数有效性
    if (start > end) {
        throw std::runtime_error(
            "Invalid slice range [" + std::to_string(start) +
            ", " + std::to_string(end) + ") with step " +
            std::to_string(step)
        );
    }
    if (step <= 0) {
        throw std::runtime_error("Step must be positive, got " + std::to_string(step));
    }
    
    // 计算新长度
    const int64_t new_length = (end - start + step - 1) / step;
    
    // 创建新形状
    std::vector<int64_t> new_shape = current_shape;
    new_shape[dim] = new_length;
    
    // 创建新TensorImpl（假设有相应的构造函数）
    auto new_impl = std::make_shared<TensorImpl>(*impl_);
    
    // 使用公共接口修改属性
    new_impl->set_shape(new_shape);
    new_impl->set_offset(impl_->offset() + start * impl_->strides()[dim]);
    
    // 修改步长（假设有set_stride方法）
    auto new_strides = impl_->strides();
    new_strides[dim] *= step;
    new_impl->set_strides(new_strides);
    
    // 标记为非连续
    new_impl->set_contiguous(false);
    
    return make_tensor_from_impl(new_impl);
}

// 填充操作
void Tensor::fill_(float value) {
    if (dtype() == kFloat32) {
        float* data = static_cast<float*>(data_ptr());
        std::fill(data, data + numel(), value);
    } else if (dtype() == kFloat64) {
        double* data = static_cast<double*>(data_ptr());
        std::fill(data, data + numel(), static_cast<double>(value));
    } else {
        throw std::runtime_error("fill_ only implemented for float types");
    }
}

void Tensor::zero_() { fill_(0.0f); }

// 特殊张量创建
Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(0.0f);
    return result;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(1.0f);
    return result;
}

Tensor Tensor::eye(int64_t n, DType dtype, Device device) {
    Tensor result({n, n}, dtype, device);
    result.zero_();
    
    if (dtype == kFloat32) {
        float* data = static_cast<float*>(result.data_ptr());
        for (int64_t i = 0; i < n; ++i) {
            data[i * n + i] = 1.0f;
        }
    } else if (dtype == kFloat64) {
        double* data = static_cast<double*>(result.data_ptr());
        for (int64_t i = 0; i < n; ++i) {
            data[i * n + i] = 1.0;
        }
    } else {
        throw std::runtime_error("eye only implemented for float types");
    }
    
    return result;
}

Tensor Tensor::arange(int64_t end, DType dtype, Device device) {
    return arange(0, end, 1, dtype, device);
}

Tensor Tensor::arange(int64_t start, int64_t end, int64_t step, DType dtype, Device device) {
    if (step <= 0) {
        throw std::runtime_error("Step must be positive in arange");
    }
    
    int64_t size = (end - start + step - 1) / step;
    Tensor result({size}, dtype, device);
    
    if (dtype == kFloat32) {
        float* data = static_cast<float*>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = static_cast<float>(start + i * step);
        }
    } else if (dtype == kFloat64) {
        double* data = static_cast<double*>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = static_cast<double>(start + i * step);
        }
    } else if (dtype == kInt32) {
        int32_t* data = static_cast<int32_t*>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = static_cast<int32_t>(start + i * step);
        }
    } else if (dtype == kInt64) {
        int64_t* data = static_cast<int64_t*>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = start + i * step;
        }
    } else {
        throw std::runtime_error("arange only implemented for numeric types");
    }
    
    return result;
}

// 张量拼接
Tensor Tensor::cat(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }
    
    // 检查所有张量的兼容性
    const int64_t ndim = tensors[0].dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        std::ostringstream oss;
        oss << "Dimension out of range (expected to be in range ["
            << -ndim << ", " << ndim - 1 << "], but got " << dim << ")";
        throw std::runtime_error(oss.str());
    }
    
    const DType dtype = tensors[0].dtype();
    const Device device = tensors[0].device();
    
    // 验证所有张量的一致性
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].dim() != ndim) {
            std::ostringstream oss;
            oss << "All tensors must have same number of dimensions. "
                << ndim << "D vs " << tensors[i].dim() << "D at index " << i;
            throw std::runtime_error(oss.str());
        }
        if (tensors[i].dtype() != dtype) {
            std::ostringstream oss;
            oss << "All tensors must have same dtype. "
                << dtype_name(dtype) << " vs " << dtype_name(tensors[i].dtype())
                << " at index " << i;
            throw std::runtime_error(oss.str());
        }
        if (tensors[i].device() != device) {
            std::ostringstream oss;
            oss << "All tensors must be on same device. "
                << device.str() << " vs " << tensors[i].device().str()
                << " at index " << i;
            throw std::runtime_error(oss.str());
        }
        
        for (int64_t d = 0; d < ndim; ++d) {
            if (d != dim && tensors[i].shape()[d] != tensors[0].shape()[d]) {
                std::ostringstream oss;
                oss << "All tensors must have same shape except in concatenation dimension. "
                    << shape_to_string(tensors[0].shape()) << " vs "
                    << shape_to_string(tensors[i].shape()) << " at index " << i;
                throw std::runtime_error(oss.str());
            }
        }
    }
    
    // 计算输出形状
    std::vector<int64_t> output_shape = tensors[0].shape();
    output_shape[dim] = std::accumulate(
        tensors.begin(), tensors.end(), 0LL,
        [dim](int64_t sum, const Tensor& t) { return sum + t.shape()[dim]; }
    );
    
    // 创建输出张量
    Tensor result(output_shape, dtype, device);
    
    // 执行拼接操作
    if (device.type() == kCPU) {
        // 获取拼接维度的步长
        const int64_t stride = result.impl_->strides()[dim] * dtype_size(dtype);
        char* dst_ptr = static_cast<char*>(result.data_ptr());
        
        for (const auto& t : tensors) {
            const int64_t total_bytes = t.numel() * dtype_size(dtype);
            const char* src_ptr = static_cast<const char*>(t.data_ptr());
            
            std::memcpy(dst_ptr, src_ptr, total_bytes);
            dst_ptr += t.shape()[dim] * stride;
        }
    }
    #ifdef WITH_CUDA
    else if (device.type() == kCUDA) {
        // 获取拼接维度的步长
        const int64_t stride = result.impl_->strides()[dim] * dtype_size(dtype);
        char* dst_ptr = static_cast<char*>(result.data_ptr());
        
        for (const auto& t : tensors) {
            const int64_t total_bytes = t.numel() * dtype_size(dtype);
            const char* src_ptr = static_cast<const char*>(t.data_ptr());
            
            cudaMemcpy(dst_ptr, src_ptr, total_bytes, cudaMemcpyDeviceToDevice);
            dst_ptr += t.shape()[dim] * stride;
        }
    }
    #endif
    else {
        throw std::runtime_error("Unsupported device type for cat operation");
    }
    
    return result;
}

// 操作符重载
Tensor Tensor::operator+(const Tensor& other) const {
    // 简单实现：仅支持CPU上的float32加法
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("Operator+ only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("Operator+ only implemented for float32");
    }
    if (shape() != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator+");
    }

    Tensor result(shape(), dtype(), device());
    const float* lhs = static_cast<const float*>(data_ptr());
    const float* rhs = static_cast<const float*>(other.data_ptr());
    float* res = static_cast<float*>(result.data_ptr());

    for (int64_t i = 0; i < numel(); ++i) {
        res[i] = lhs[i] + rhs[i];
    }

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    // 实现类似operator+
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("Operator- only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("Operator- only implemented for float32");
    }
    if (shape() != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator-");
    }

    Tensor result(shape(), dtype(), device());
    const float* lhs = static_cast<const float*>(data_ptr());
    const float* rhs = static_cast<const float*>(other.data_ptr());
    float* res = static_cast<float*>(result.data_ptr());

    for (int64_t i = 0; i < numel(); ++i) {
        res[i] = lhs[i] - rhs[i];
    }

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    // 实现类似operator+
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("Operator* only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("Operator* only implemented for float32");
    }
    if (shape() != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator*");
    }

    Tensor result(shape(), dtype(), device());
    const float* lhs = static_cast<const float*>(data_ptr());
    const float* rhs = static_cast<const float*>(other.data_ptr());
    float* res = static_cast<float*>(result.data_ptr());

    for (int64_t i = 0; i < numel(); ++i) {
        res[i] = lhs[i] * rhs[i];
    }

    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    // 实现类似operator+
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("Operator/ only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("Operator/ only implemented for float32");
    }
    if (shape() != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator/");
    }

    Tensor result(shape(), dtype(), device());
    const float* lhs = static_cast<const float*>(data_ptr());
    const float* rhs = static_cast<const float*>(other.data_ptr());
    float* res = static_cast<float*>(result.data_ptr());

    for (int64_t i = 0; i < numel(); ++i) {
        res[i] = lhs[i] / rhs[i];
    }

    return result;
}

// 复合赋值操作符
Tensor& Tensor::operator+=(const Tensor& other) {
    *this = *this + other;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    *this = *this - other;
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other;
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other;
    return *this;
}

// 一元操作符
Tensor Tensor::operator-() const {
    if (device().type() != kCPU) {
        throw std::runtime_error("Unary operator- only implemented on CPU");
    }
    if (dtype() != kFloat32) {
        throw std::runtime_error("Unary operator- only implemented for float32");
    }

    Tensor result(shape(), dtype(), device());
    const float* src = static_cast<const float*>(data_ptr());
    float* dst = static_cast<float*>(result.data_ptr());

    for (int64_t i = 0; i < numel(); ++i) {
        dst[i] = -src[i];
    }

    return result;
}

// 比较操作符
Tensor Tensor::operator==(const Tensor& other) const {
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("Operator== only implemented on CPU");
    }
    if (shape() != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator==");
    }

    Tensor result(shape(), kBool, device());
    const float* lhs = static_cast<const float*>(data_ptr());
    const float* rhs = static_cast<const float*>(other.data_ptr());
    bool* res = static_cast<bool*>(result.data_ptr());

    for (int64_t i = 0; i < numel(); ++i) {
        res[i] = (lhs[i] == rhs[i]);
    }

    return result;
}

// 矩阵乘法
Tensor Tensor::matmul(const Tensor& other) const {
    if (dim() != 2 || other.dim() != 2) {
        throw std::runtime_error("matmul only implemented for 2D tensors");
    }
    if (shape()[1] != other.shape()[0]) {
        throw std::runtime_error("Shape mismatch in matmul");
    }
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("matmul only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("matmul only implemented for float32");
    }

    Tensor result({shape()[0], other.shape()[1]}, dtype(), device());
    const float* lhs = static_cast<const float*>(data_ptr());
    const float* rhs = static_cast<const float*>(other.data_ptr());
    float* res = static_cast<float*>(result.data_ptr());

    for (int64_t i = 0; i < shape()[0]; ++i) {
        for (int64_t j = 0; j < other.shape()[1]; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < shape()[1]; ++k) {
                sum += lhs[i * shape()[1] + k] * rhs[k * other.shape()[1] + j];
            }
            res[i * other.shape()[1] + j] = sum;
        }
    }

    return result;
}

// 点乘
Tensor Tensor::dot(const Tensor& other) const {
    if (dim() != 1 || other.dim() != 1) {
        throw std::runtime_error("dot only implemented for 1D tensors");
    }
    if (shape()[0] != other.shape()[0]) {
        throw std::runtime_error("Shape mismatch in dot");
    }
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("dot only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("dot only implemented for float32");
    }

    Tensor result({}, dtype(), device());
    const float* lhs = static_cast<const float*>(data_ptr());
    const float* rhs = static_cast<const float*>(other.data_ptr());
    float* res = static_cast<float*>(result.data_ptr());

    float sum = 0.0f;
    for (int64_t i = 0; i < shape()[0]; ++i) {
        sum += lhs[i] * rhs[i];
    }
    *res = sum;

    return result;
}

// 归约操作
Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    // 获取当前维度数和形状
    const int64_t ndim = this->dim();
    const auto& current_shape = shape();
    
    // 处理负维度索引
    if (dim < 0) dim += ndim;
    
    // 验证维度范围
    if (dim < 0 || dim >= ndim) {
        std::ostringstream oss;
        oss << "Dimension out of range (expected to be in range ["
            << -ndim << ", " << ndim-1 << "], but got " << dim << ")";
        throw std::runtime_error(oss.str());
    }
    
    // 设备检查
    if (device().type() != kCPU) {
        throw std::runtime_error("sum operation currently only supports CPU tensors");
    }
    
    // 数据类型检查
    if (dtype() != kFloat32) {
        throw std::runtime_error("sum operation currently only supports float32 tensors");
    }

    // 计算输出形状
    std::vector<int64_t> output_shape = current_shape;
    if (keepdim) {
        output_shape[dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + dim);
    }

    // 创建结果张量
    Tensor result(output_shape, dtype(), device());
    const float* src_data = static_cast<const float*>(data_ptr());
    float* dst_data = static_cast<float*>(result.data_ptr());

    // 计算各维度步长
    const auto& strides = impl_->strides();
    const int64_t stride_dim = strides[dim];
    
    // 计算归约前后的总元素数
    const int64_t num_reduced = current_shape[dim];
    const int64_t num_outer = std::accumulate(
        current_shape.begin(), current_shape.begin() + dim, 1LL, std::multiplies<int64_t>());
    const int64_t num_inner = std::accumulate(
        current_shape.begin() + dim + 1, current_shape.end(), 1LL, std::multiplies<int64_t>());

    // 通用实现（支持任意维度）
    for (int64_t outer = 0; outer < num_outer; ++outer) {
        for (int64_t inner = 0; inner < num_inner; ++inner) {
            float sum = 0.0f;
            const float* src_ptr = src_data + 
                outer * stride_dim * num_reduced + 
                inner * stride_dim;
            
            // 沿归约维度求和
            for (int64_t k = 0; k < num_reduced; ++k) {
                sum += src_ptr[k * stride_dim];
            }
            
            // 存储结果
            dst_data[outer * num_inner + inner] = sum;
        }
    }

    return result;
}

// 其他归约操作实现类似...

// 激活函数
Tensor Tensor::relu() const {
    if (device().type() != kCPU) {
        throw std::runtime_error("relu only implemented on CPU");
    }
    if (dtype() != kFloat32) {
        throw std::runtime_error("relu only implemented for float32");
    }

    Tensor result(shape(), dtype(), device());
    const float* src = static_cast<const float*>(data_ptr());
    float* dst = static_cast<float*>(result.data_ptr());

    for (int64_t i = 0; i < numel(); ++i) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }

    return result;
}

// 其他激活函数实现类似...

// 自动微分相关
void Tensor::backward(const Tensor& grad) {
    throw std::runtime_error("Autograd not implemented yet");
}

Tensor Tensor::grad() const {
    throw std::runtime_error("Autograd not implemented yet");
}

void Tensor::set_requires_grad(bool requires_grad) {
    impl_->set_requires_grad(requires_grad);
    
}

bool Tensor::requires_grad() const {
    return impl_->requires_grad();
    
}

// 打印张量
std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape().size(); ++i) {
        if (i != 0) oss << ", ";
        oss << shape()[i];
    }
    oss << "], dtype=" << dtype() << ", device=" << device().str() << ")";
    return oss.str();
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
}

// 序列化/反序列化
void Tensor::save(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing");
    }

    // 写入元数据
    int64_t ndim = dim();
    ofs.write(reinterpret_cast<const char*>(&ndim), sizeof(int64_t));
    ofs.write(reinterpret_cast<const char*>(shape().data()), ndim * sizeof(int64_t));
    
    DType dt = dtype();
    ofs.write(reinterpret_cast<const char*>(&dt), sizeof(DType));
    
    Device dev = device();
    DeviceType dev_type = dev.type();
    int dev_index = dev.index();
    ofs.write(reinterpret_cast<const char*>(&dev_type), sizeof(DeviceType));
    ofs.write(reinterpret_cast<const char*>(&dev_index), sizeof(int));

    // 写入数据
    if (device().type() == kCPU) {
        ofs.write(reinterpret_cast<const char*>(data_ptr()), numel() * dtype_size(dtype()));
    } else {
        throw std::runtime_error("save only implemented for CPU tensors");
    }
}

Tensor Tensor::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading");
    }

    // 读取元数据
    int64_t ndim;
    ifs.read(reinterpret_cast<char*>(&ndim), sizeof(int64_t));
    
    std::vector<int64_t> shape(ndim);
    ifs.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int64_t));
    
    DType dt;
    ifs.read(reinterpret_cast<char*>(&dt), sizeof(DType));
    
    DeviceType dev_type;
    int dev_index;
    ifs.read(reinterpret_cast<char*>(&dev_type), sizeof(DeviceType));
    ifs.read(reinterpret_cast<char*>(&dev_index), sizeof(int));
    Device dev(dev_type, dev_index);

    // 创建张量
    Tensor result(shape, dt, dev);

    // 读取数据
    if (dev.type() == kCPU) {
        ifs.read(reinterpret_cast<char*>(result.data_ptr()), result.numel() * dtype_size(dt));
    } else {
        throw std::runtime_error("load only implemented for CPU tensors");
    }

    return result;
}

} // namespace core
