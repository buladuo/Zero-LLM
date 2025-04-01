#pragma once
#include <atomic>
#include <vector>
#include <memory>
#include <stdexcept>
#include "dtype.h"
#include "device.h"

namespace core {

class TensorImpl {
public:
    // 构造函数
    TensorImpl(const std::vector<int64_t>& shape, DType dtype, Device device);
    TensorImpl(const std::vector<int64_t>& shape, void* data, DType dtype, Device device);
    TensorImpl(const std::vector<int64_t>& shape, void* data, DType dtype, Device device, int64_t offset);
    ~TensorImpl();

    // 内存管理
    void allocate();
    void deallocate();
    void copy_data_from(const TensorImpl& other);
    void move_data_from(TensorImpl&& other);

    // 形状操作
    int64_t numel() const {
        if (shape_.empty()) return 0;
        int64_t size = 1;
        for (auto dim : shape_) {
            if (dim < 0) throw std::runtime_error("Negative dimension in shape");
            size *= dim;
        }
        return size;
    }

    void reshape(const std::vector<int64_t>& new_shape) {
        check_shape(new_shape);
        shape_ = new_shape;
        compute_strides();
    }

    void set_shape_and_strides(const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& strides) {
        check_shape(shape);
        if (shape.size() != strides.size()) {
            throw std::runtime_error("Shape and strides size mismatch");
        }
        shape_ = shape;
        strides_ = strides;
        update_contiguity();
    }
    void set_offset(int64_t offset) { offset_ = offset; }
    void set_contiguous(bool contiguous) {is_contiguous_ = contiguous;}
    void set_shape(const std::vector<int64_t>& shape) {shape_ = shape; compute_strides();}
    void set_strides(const std::vector<int64_t>& strides) {strides_ = strides; is_contiguous_ = check_contiguous();}


    // 数据访问
    template <DType T>
    typename dtype_traits<T>::ctype* data_as() {
        if (dtype_ != T) {
            throw std::runtime_error(std::string("Type mismatch in data_as: expected ") + 
                                   dtype_name(T) + " got " + dtype_name(dtype_));
        }
        return static_cast<typename dtype_traits<T>::ctype*>(data_);
    }

    template <DType T>
    const typename dtype_traits<T>::ctype* data_as() const {
        return const_cast<TensorImpl*>(this)->data_as<T>();
    }

    // 属性访问
    int64_t offset() const { return offset_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t nbytes() const { return nbytes_; }
    bool is_contiguous() const { return is_contiguous_; }
    void* data_ptr() { return data_; }
    const void* data_ptr() const { return data_; }

    // 自动微分支持
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    bool requires_grad() const { return requires_grad_; }
    std::shared_ptr<TensorImpl> grad() const { return grad_; }
    void set_grad(std::shared_ptr<TensorImpl> grad) { grad_ = grad; }

    // 工厂方法
    static std::shared_ptr<TensorImpl> create(const std::vector<int64_t>& shape, 
                                            DType dtype, Device device) {
        return std::make_shared<TensorImpl>(shape, dtype, device);
    }

    // CUDA支持
    #ifdef WITH_CUDA
    static bool is_cuda_available();
    #endif

private:
    // 内部实现方法
    void compute_strides();
    bool check_contiguous() const;
    void check_shape(const std::vector<int64_t>& shape) const;
    void update_contiguity() { is_contiguous_ = check_contiguous(); }

    // 内存分配
    void cpu_allocate();
    void cpu_deallocate();
    
    #ifdef WITH_CUDA
    void cuda_allocate();
    void cuda_deallocate();
    void cuda_copy_from_host(const void* src);
    void cuda_copy_to_host(void* dst) const;
    #endif

    // 数据成员
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t offset_ = 0;
    bool is_contiguous_ = true;
    void* data_ = nullptr;
    DType dtype_;
    Device device_;
    size_t nbytes_ = 0;
    std::atomic<int> refcount_{1};
    bool requires_grad_ = false;
    std::shared_ptr<TensorImpl> grad_;
};

// 辅助函数
inline bool is_shape_compatible(const std::vector<int64_t>& shape1, 
                              const std::vector<int64_t>& shape2) {
    if (shape1.size() != shape2.size()) return false;
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i]) return false;
    }
    return true;
}

} // namespace core