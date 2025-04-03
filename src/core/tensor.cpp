#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "core/tensor.h"
#include "core/autograd/engine.h"
#include "core/autograd/function/base.h"



namespace core {

//=== 构造函数和析构函数 ===//
Tensor::Tensor()
    : impl_(std::make_shared<TensorImpl>(std::vector<int64_t>{}, kFloat32,
                                         Device::CPU())) {}

Tensor::Tensor(const std::vector<int64_t> &shape, DType dtype, Device device)
    : impl_(std::make_shared<TensorImpl>(shape, dtype, device)) {
    impl_->allocate(AllocStrategy::DEFAULT, 64);
}

Tensor::Tensor(const std::vector<int64_t> &shape, void *data, DType dtype,
               Device device)
    : impl_(std::make_shared<TensorImpl>(shape, data, dtype, device,
                                         MemoryOwnership::BORROWED)) {}

Tensor::Tensor(const Tensor &other) : impl_(other.impl_) {}

Tensor::Tensor(Tensor &&other) noexcept : impl_(std::move(other.impl_)) {}

Tensor::~Tensor() = default;

//=== 工厂函数 ===//
inline Tensor make_tensor_from_impl(std::shared_ptr<TensorImpl> impl) {
    return Tensor(impl);
}

//=== 赋值操作符 ===//
Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        impl_ = other.impl_;
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

// tensor.cpp
Tensor Tensor::clone() const {
    if (!impl_) {
        return Tensor();  // 空张量直接返回
    }

    // 创建新张量（相同形状、类型和设备）
    Tensor result(impl_->shape(), impl_->dtype(), impl_->device());
    
    if (impl_->device().type() == DeviceType::kCPU) {
        // CPU 内存拷贝
        std::memcpy(
            result.impl_->data_ptr(),
            impl_->data_ptr(),
            impl_->numel() * dtype_size(impl_->dtype())
        );
    }
#ifdef WITH_CUDA
    else if (impl_->device().type() == DeviceType::kCUDA) {
        // CUDA 内存拷贝
        CUDA_CHECK(cudaMemcpy(
            result.impl_->data_ptr(),
            impl_->data_ptr(),
            impl_->numel() * dtype_size(impl_->dtype()),
            cudaMemcpyDeviceToDevice
        ));
    }
#endif
    else {
        throw std::runtime_error("Unsupported device in clone");
    }

    // 复制梯度信息（如果存在）
    if (impl_->grad()) {
        result.impl_->set_grad(impl_->grad()->clone()); // 调用 TensorImpl::clone()
    }

    // 复制其他属性
    result.impl_->set_requires_grad(impl_->requires_grad());
    
    return result;
}

//=== 基本属性访问 ===//
int64_t Tensor::dim() const { return impl_->shape().size(); }
const std::vector<int64_t> &Tensor::shape() const { return impl_->shape(); }
int64_t Tensor::size() const { return impl_->numel(); }
int64_t Tensor::numel() const { return impl_->numel(); }
DType Tensor::dtype() const { return impl_->dtype(); }
Device Tensor::device() const { return impl_->device(); }
bool Tensor::is_contiguous() const { return impl_->is_contiguous(); }
void *Tensor::data_ptr() { return impl_->data_ptr(); }
const void *Tensor::data_ptr() const { return impl_->data_ptr(); }
bool Tensor::is_pinned() const { return impl_->is_pinned(); }

//=== 内存管理 ===//
Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    Tensor result(shape(), dtype(), device());
    result.impl_->copy_data_from(*impl_);
    return result;
}

void Tensor::pin_memory() {
    if (device().type() != DeviceType::kCPU) {
        throw std::runtime_error("Only CPU tensors can be pinned");
    }
    if (is_pinned()) return;
    
    // 保留原有数据
    Tensor tmp = this->clone();
    impl_->deallocate();
    impl_->allocate(AllocStrategy::PINNED);
    impl_->copy_data_from(*tmp.impl_);
}

//=== 设备转换 ===//
Tensor Tensor::to(Device device, AllocStrategy strategy) const {
    if (device == impl_->device()) {
        return *this;
    }

    Tensor result(shape(), dtype(), device);
    result.impl_->allocate(strategy);
    result.impl_->copy_data_from(*impl_);
    return result;
}

Tensor Tensor::cpu() const { return to(Device::CPU(), AllocStrategy::DEFAULT); }
Tensor Tensor::cuda() const {
    return to(Device::CUDA(), AllocStrategy::PINNED);
}

//=== 类型转换 ===//
Tensor Tensor::to(DType dtype) const {
    if (dtype == impl_->dtype()) {
        return *this;
    }

    Tensor result(shape(), dtype, device());
    result.impl_->convert_dtype(dtype); // 直接传递目标类型
    return result;
}

Tensor Tensor::float32() const { return to(kFloat32); }
Tensor Tensor::float16() const { return to(kFloat16); }
Tensor Tensor::bfloat16() const { return to(kBFloat16); }

//=== 形状操作 ===//
Tensor Tensor::view(const std::vector<int64_t> &shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error(
            "View is only supported for contiguous tensors");
    }

    int64_t new_numel = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<int64_t>());
    if (new_numel != numel()) {
        throw std::runtime_error("Shape size mismatch in view operation");
    }

    auto new_impl =
        impl_->create_view(shape, impl_->strides(), impl_->offset());
    return make_tensor_from_impl(new_impl);
}

Tensor Tensor::reshape(const std::vector<int64_t> &shape) const {
    int64_t new_numel = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<int64_t>());
    if (new_numel != numel()) {
        throw std::runtime_error("Shape size mismatch in reshape");
    }

    if (is_contiguous()) {
        return view(shape);
    }
    return contiguous().view(shape);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    auto ndim = dim();
    if (dim0 < 0)
        dim0 += ndim;
    if (dim1 < 0)
        dim1 += ndim;

    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::runtime_error("Invalid dimension for transpose");
    }

    auto new_shape = shape();
    auto new_strides = impl_->strides();
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    auto new_impl = impl_->create_view(new_shape, new_strides, impl_->offset());
    return make_tensor_from_impl(new_impl);
}

Tensor Tensor::permute(const std::vector<int64_t> &dims) const {
    auto ndim = dim();
    if (dims.size() != ndim) {
        throw std::runtime_error("Permute dimensions must match tensor rank");
    }

    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);
    const auto &old_strides = impl_->strides();

    for (size_t i = 0; i < dims.size(); ++i) {
        auto src_dim = dims[i];
        if (src_dim < 0)
            src_dim += ndim;
        if (src_dim < 0 || src_dim >= ndim) {
            throw std::runtime_error("Invalid dimension in permute");
        }
        new_shape[i] = shape()[src_dim];
        new_strides[i] = old_strides[src_dim];
    }

    auto new_impl = impl_->create_view(new_shape, new_strides, impl_->offset());
    return make_tensor_from_impl(new_impl);
}

Tensor Tensor::squeeze(int64_t dim) const {
    const auto &old_shape = shape();
    std::vector<int64_t> new_shape;

    if (dim == -1) {
        std::copy_if(old_shape.begin(), old_shape.end(),
                     std::back_inserter(new_shape),
                     [](int64_t d) { return d != 1; });
    } else {
        const int64_t tensor_dim = this->dim();
        if (dim < 0)
            dim += tensor_dim;

        if (dim < 0 || dim >= tensor_dim) {
            throw std::runtime_error("Invalid dimension for squeeze");
        }

        if (old_shape[dim] != 1) {
            throw std::runtime_error("Can only squeeze dimension with size 1");
        }

        new_shape = old_shape;
        new_shape.erase(new_shape.begin() + dim);
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
    }

    return view(new_shape);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    const int64_t current_dim = this->dim();
    if (dim < 0)
        dim += current_dim + 1;

    if (dim < 0 || dim > current_dim) {
        throw std::runtime_error("Dimension out of range in unsqueeze");
    }

    auto new_shape = shape();
    new_shape.insert(new_shape.begin() + dim, 1);

    if (current_dim == 0) {
        return Tensor(new_shape, dtype(), device());
    }

    return view(new_shape);
}

//=== 索引操作 ===//
Tensor Tensor::operator[](int64_t index) const { return select(0, index); }

Tensor Tensor::select(int64_t dim, int64_t index) const {
    const int64_t current_dim = this->dim();
    if (dim < 0)
        dim += current_dim;
    if (dim < 0 || dim >= current_dim) {
        throw std::runtime_error("Invalid dimension in select");
    }

    const int64_t dim_size = shape()[dim];
    if (index < 0)
        index += dim_size;
    if (index < 0 || index >= dim_size) {
        throw std::runtime_error("Index out of range in select");
    }

    std::vector<int64_t> new_shape = shape();
    new_shape.erase(new_shape.begin() + dim);

    int64_t new_offset = impl_->offset() + index * impl_->strides()[dim];
    auto new_impl = impl_->create_view(new_shape, impl_->strides(), new_offset);
    return make_tensor_from_impl(new_impl);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end,
                     int64_t step) const {
    const int64_t current_dim = this->dim();
    if (dim < 0)
        dim += current_dim;
    if (dim < 0 || dim >= current_dim) {
        throw std::runtime_error("Invalid dimension in slice");
    }

    const int64_t dim_size = shape()[dim];
    if (start < 0)
        start += dim_size;
    if (end < 0)
        end += dim_size;
    start = std::max<int64_t>(0, std::min(start, dim_size));
    end = std::max<int64_t>(0, std::min(end, dim_size));

    if (start > end) {
        throw std::runtime_error("Invalid slice range");
    }
    if (step <= 0) {
        throw std::runtime_error("Step must be positive");
    }

    const int64_t new_length = (end - start + step - 1) / step;
    std::vector<int64_t> new_shape = shape();
    new_shape[dim] = new_length;

    std::vector<int64_t> new_strides = impl_->strides();
    new_strides[dim] *= step;

    int64_t new_offset = impl_->offset() + start * impl_->strides()[dim];
    auto new_impl = impl_->create_view(new_shape, new_strides, new_offset);
    return make_tensor_from_impl(new_impl);
}

//=== 填充操作 ===//
void Tensor::fill_(float value) {
    if (dtype() == kFloat32) {
        if (device().type() == kCPU) {
            float *data = static_cast<float *>(data_ptr());
            std::fill(data, data + numel(), value);
        }
#ifdef WITH_CUDA
        else if (device().type() == kCUDA) {
            cudaMemset(data_ptr(), *reinterpret_cast<int *>(&value),
                       numel() * sizeof(float));
        }
#endif
    } else if (dtype() == kFloat64) {
        double *data = static_cast<double *>(data_ptr());
        std::fill(data, data + numel(), static_cast<double>(value));
    } else {
        throw std::runtime_error("fill_ only implemented for float types");
    }
}

void Tensor::zero_() {
    if (device().type() == kCPU) {
        std::memset(data_ptr(), 0, numel() * dtype_size(dtype()));
    }
#ifdef WITH_CUDA
    else if (device().type() == kCUDA) {
        cudaMemset(data_ptr(), 0, numel() * dtype_size(dtype()));
    }
#endif
}


//=== 随机初始化 ===//
Tensor Tensor::rand(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor result(shape, dtype, device);
    
    // 使用标准库的随机数生成器
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    if (device.type() == DeviceType::kCPU) {
        if (dtype == kFloat32) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float* data = static_cast<float*>(result.data_ptr());
            for (int64_t i = 0; i < result.numel(); ++i) {
                data[i] = dist(gen);
            }
        } else if (dtype == kFloat64) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double* data = static_cast<double*>(result.data_ptr());
            for (int64_t i = 0; i < result.numel(); ++i) {
                data[i] = dist(gen);
            }
        } else {
            throw std::runtime_error("rand only implemented for float types");
        }
    }
#ifdef WITH_CUDA
    else if (device.type() == DeviceType::kCUDA) {
        // CUDA实现 - 使用CUDA的随机数生成器
        if (dtype == kFloat32) {
            curandGenerator_t generator;
            CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, rd()));
            CURAND_CHECK(curandGenerateUniform(generator, static_cast<float*>(result.data_ptr()), result.numel()));
            CURAND_CHECK(curandDestroyGenerator(generator));
        } else if (dtype == kFloat64) {
            curandGenerator_t generator;
            CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, rd()));
            CURAND_CHECK(curandGenerateUniformDouble(generator, static_cast<double*>(result.data_ptr()), result.numel()));
            CURAND_CHECK(curandDestroyGenerator(generator));
        } else {
            throw std::runtime_error("rand only implemented for float types");
        }
    }
#endif
    else {
        throw std::runtime_error("Unsupported device for rand");
    }
    
    return result;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor result(shape, dtype, device);
    
    // 使用标准库的随机数生成器
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    if (device.type() == DeviceType::kCPU) {
        if (dtype == kFloat32) {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            float* data = static_cast<float*>(result.data_ptr());
            for (int64_t i = 0; i < result.numel(); ++i) {
                data[i] = dist(gen);
            }
        } else if (dtype == kFloat64) {
            std::normal_distribution<double> dist(0.0, 1.0);
            double* data = static_cast<double*>(result.data_ptr());
            for (int64_t i = 0; i < result.numel(); ++i) {
                data[i] = dist(gen);
            }
        } else {
            throw std::runtime_error("randn only implemented for float types");
        }
    }
#ifdef WITH_CUDA
    else if (device.type() == DeviceType::kCUDA) {
        // CUDA实现 - 使用CUDA的随机数生成器
        if (dtype == kFloat32) {
            curandGenerator_t generator;
            CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, rd()));
            CURAND_CHECK(curandGenerateNormal(generator, static_cast<float*>(result.data_ptr()), 
                                             result.numel(), 0.0f, 1.0f));
            CURAND_CHECK(curandDestroyGenerator(generator));
        } else if (dtype == kFloat64) {
            curandGenerator_t generator;
            CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, rd()));
            CURAND_CHECK(curandGenerateNormalDouble(generator, static_cast<double*>(result.data_ptr()), 
                                                  result.numel(), 0.0, 1.0));
            CURAND_CHECK(curandDestroyGenerator(generator));
        } else {
            throw std::runtime_error("randn only implemented for float types");
        }
    }
#endif
    else {
        throw std::runtime_error("Unsupported device for randn");
    }
    
    return result;
}

//=== 特殊张量创建 ===//
Tensor Tensor::zeros(const std::vector<int64_t> &shape, DType dtype,
                     Device device) {
    Tensor result(shape, dtype, device);
    result.zero_();
    return result;
}

Tensor Tensor::ones(const std::vector<int64_t> &shape, DType dtype,
                    Device device) {
    Tensor result(shape, dtype, device);
    result.fill_(1.0f);
    return result;
}

Tensor Tensor::eye(int64_t n, DType dtype, Device device) {
    Tensor result({n, n}, dtype, device);
    result.zero_();

    if (dtype == kFloat32) {
        float *data = static_cast<float *>(result.data_ptr());
        for (int64_t i = 0; i < n; ++i) {
            data[i * n + i] = 1.0f;
        }
    } else if (dtype == kFloat64) {
        double *data = static_cast<double *>(result.data_ptr());
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

Tensor Tensor::arange(int64_t start, int64_t end, int64_t step, DType dtype,
                      Device device) {
    if (step <= 0) {
        throw std::runtime_error("Step must be positive in arange");
    }

    int64_t size = (end - start + step - 1) / step;
    Tensor result({size}, dtype, device);

    if (dtype == kFloat32) {
        float *data = static_cast<float *>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = static_cast<float>(start + i * step);
        }
    } else if (dtype == kFloat64) {
        double *data = static_cast<double *>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = static_cast<double>(start + i * step);
        }
    } else if (dtype == kInt32) {
        int32_t *data = static_cast<int32_t *>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = static_cast<int32_t>(start + i * step);
        }
    } else if (dtype == kInt64) {
        int64_t *data = static_cast<int64_t *>(result.data_ptr());
        for (int64_t i = 0; i < size; ++i) {
            data[i] = start + i * step;
        }
    } else {
        throw std::runtime_error("arange only implemented for numeric types");
    }

    return result;
}

//=== 张量拼接 ===//
Tensor Tensor::cat(const std::vector<Tensor> &tensors, int64_t dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Cannot concatenate empty tensor list");
    }

    const int64_t ndim = tensors[0].dim();
    if (dim < 0)
        dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range in cat");
    }

    const DType dtype = tensors[0].dtype();
    const Device device = tensors[0].device();

    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].dim() != ndim || tensors[i].dtype() != dtype ||
            tensors[i].device() != device) {
            throw std::runtime_error(
                "All tensors must have same dimensions, type and device");
        }
    }

    std::vector<int64_t> output_shape = tensors[0].shape();
    output_shape[dim] = std::accumulate(
        tensors.begin(), tensors.end(), 0LL,
        [dim](int64_t sum, const Tensor &t) { return sum + t.shape()[dim]; });

    Tensor result(output_shape, dtype, device);

    if (device.type() == kCPU) {
        size_t dtype_size_val = dtype_size(dtype);
        char *dst_ptr = static_cast<char *>(result.data_ptr());
        const int64_t stride = result.impl_->strides()[dim] * dtype_size_val;

        for (const auto &t : tensors) {
            const size_t total_bytes = t.numel() * dtype_size_val;
            const char *src_ptr = static_cast<const char *>(t.data_ptr());

            std::memcpy(dst_ptr, src_ptr, total_bytes);
            dst_ptr += t.shape()[dim] * stride;
        }
    }
#ifdef WITH_CUDA
    else if (device.type() == kCUDA) {
        size_t dtype_size_val = dtype_size(dtype);
        char *dst_ptr = static_cast<char *>(result.data_ptr());
        const int64_t stride = result.impl_->strides()[dim] * dtype_size_val;

        for (const auto &t : tensors) {
            const size_t total_bytes = t.numel() * dtype_size_val;
            const char *src_ptr = static_cast<const char *>(t.data_ptr());

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

Tensor Tensor::ones_like(const Tensor& input) {
    auto result = Tensor(input.shape(), input.dtype(), input.device());
    result.fill_(1.0f);
    return result;
}

// tensor.cpp
Tensor Tensor::zeros_like(const Tensor& input) {
    auto result = Tensor(input.shape(), input.dtype(), input.device());
    result.fill_(0.0f);  // 填充0
    return result;
}

//=== 操作符重载 ===//
Tensor Tensor::operator+(const Tensor &other) const {
    Tensor result(shape(), dtype(), device());
    result.impl_->add_impl(*impl_);
    return result;
}

Tensor Tensor::operator-(const Tensor &other) const {
    Tensor result = *this;
    result -= other;
    return result;
}

Tensor Tensor::operator*(const Tensor &other) const {
    Tensor result = *this;
    result *= other;
    return result;
}

Tensor Tensor::operator/(const Tensor &other) const {
    Tensor result = *this;
    result /= other;
    return result;
}

Tensor &Tensor::operator+=(const Tensor &other) {
    impl_->add_impl(*other.impl_);
    return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
    // 实现减法操作
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("Operator-= only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("Operator-= only implemented for float32");
    }
    if (shape() != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator-=");
    }

    float *data = static_cast<float *>(data_ptr());
    const float *other_data = static_cast<const float *>(other.data_ptr());
    for (int64_t i = 0; i < numel(); ++i) {
        data[i] -= other_data[i];
    }
    return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
    // 实现乘法操作
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("Operator*= only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("Operator*= only implemented for float32");
    }
    if (shape() != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator*=");
    }

    float *data = static_cast<float *>(data_ptr());
    const float *other_data = static_cast<const float *>(other.data_ptr());
    for (int64_t i = 0; i < numel(); ++i) {
        data[i] *= other_data[i];
    }
    return *this;
}

Tensor &Tensor::operator/=(const Tensor &other) {
    // 实现除法操作
    if (device().type() != kCPU || other.device().type() != kCPU) {
        throw std::runtime_error("Operator/= only implemented on CPU");
    }
    if (dtype() != kFloat32 || other.dtype() != kFloat32) {
        throw std::runtime_error("Operator/= only implemented for float32");
    }
    if (shape() != other.shape()) {
        throw std::runtime_error("Shape mismatch in operator/=");
    }

    float *data = static_cast<float *>(data_ptr());
    const float *other_data = static_cast<const float *>(other.data_ptr());
    for (int64_t i = 0; i < numel(); ++i) {
        data[i] /= other_data[i];
    }
    return *this;
}

Tensor Tensor::operator-() const {
    Tensor result = *this;
    if (device().type() != kCPU) {
        throw std::runtime_error("Unary operator- only implemented on CPU");
    }
    if (dtype() != kFloat32) {
        throw std::runtime_error(
            "Unary operator- only implemented for float32");
    }

    float *data = static_cast<float *>(result.data_ptr());
    for (int64_t i = 0; i < numel(); ++i) {
        data[i] = -data[i];
    }
    return result;
}



//=== 自动微分 ===//
Tensor& Tensor::requires_grad_(bool requires_grad) {
    impl_->set_requires_grad(requires_grad);
    return *this;
}

bool Tensor::requires_grad() const {
    return impl_->requires_grad();
}

Tensor Tensor::grad() const {
    auto grad_impl = impl_->grad();
    return grad_impl ? Tensor(grad_impl) : Tensor();
}

void Tensor::set_grad(const Tensor& grad) {
    impl_->set_grad(grad.impl_);
}

void Tensor::backward(const Tensor& grad) {
    impl_->backward(grad.impl_);
}

std::shared_ptr<autograd::Function> Tensor::grad_fn() const {
    return impl_->grad_fn();
}

void Tensor::set_grad_fn(const std::shared_ptr<autograd::Function>& grad_fn) {
    impl_->set_grad_fn(grad_fn);
}

Tensor Tensor::make_autograd_aware(
    const std::vector<int64_t>& shape,
    DType dtype,
    Device device,
    bool requires_grad) {
    return Tensor(TensorImpl::create_autograd_aware(shape, dtype, device, requires_grad));
}

//=== 序列化 ===//
void Tensor::save(const std::string &filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing");
    }

    impl_->serialize(ofs);
}

Tensor Tensor::load(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading");
    }

    auto impl = TensorImpl::deserialize(ifs);
    return make_tensor_from_impl(impl);
}

//=== 调试工具 ===//
std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=" << shape_to_string(shape())
        << ", dtype=" << dtype_name(dtype()) << ", device=" << device().str()
        << ", requires_grad=" << (requires_grad() ? "true" : "false") << ")";
    return oss.str();
}

void Tensor::print() const { std::cout << to_string() << std::endl; }

} // namespace core