#pragma once
#include <atomic>
#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include "dtype.h"
#include "device.h"





namespace core {

namespace autograd {
    class Function;
    class Engine;
}

// 内存所有权标记
enum class MemoryOwnership {
    OWNED,
    BORROWED
};

// 内存分配策略
enum class AllocStrategy {
    DEFAULT,      // 普通分配
    PINNED,       // 页锁定内存（CUDA）
    SHARED        // 进程间共享
};

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    //=== 构造函数与析构 ===//
    TensorImpl(const std::vector<int64_t>& shape, DType dtype, Device device);
    TensorImpl(const std::vector<int64_t>& shape, void* data, DType dtype, 
               Device device, MemoryOwnership ownership = MemoryOwnership::BORROWED);
    ~TensorImpl();


    //=== 内存管理 ===//
    void allocate(AllocStrategy strategy = AllocStrategy::DEFAULT, size_t alignment = 64);
    void deallocate();
    void copy_data_from(const TensorImpl& other);
    void move_data_from(TensorImpl&& other);
    std::shared_ptr<TensorImpl> clone() const;


    //=== 形状操作 ===//
    void reshape(const std::vector<int64_t>& new_shape);
    void set_shape(const std::vector<int64_t>& new_shape);
    void set_strides(const std::vector<int64_t>& new_strides);
    void set_shape_and_strides(const std::vector<int64_t>& new_shape,
                              const std::vector<int64_t>& new_strides);
    void set_offset(int64_t offset);
    void set_contiguous(bool contiguous);
    
    //=== 视图操作 ===//
    std::shared_ptr<TensorImpl> create_view(const std::vector<int64_t>& shape,
                                    const std::vector<int64_t>& strides, int64_t offset);

    //=== 设备/类型转换 ===//
    void copy_to_device(Device target_device);
    void convert_dtype(DType target_dtype);

    //=== 数学操作 ===//
    void add_impl(const TensorImpl& other);
    void matmul_impl(const TensorImpl& a, const TensorImpl& b);

    //=== 自动微分 ===//
    void zero_();
    void fill_(float value);
    void accumulate_grad(const TensorImpl& grad);
    void zero_grad();

    void set_requires_grad(bool requires_grad);
    bool requires_grad() const;

    std::shared_ptr<TensorImpl> grad() const;
    void set_grad(std::shared_ptr<TensorImpl> grad);

    void backward(const std::shared_ptr<TensorImpl>& grad = nullptr);
    
    std::shared_ptr<autograd::Function> grad_fn() const;
    void set_grad_fn(const std::shared_ptr<autograd::Function>& grad_fn);

    // 工厂方法
    static std::shared_ptr<TensorImpl> create_autograd_aware(
        const std::vector<int64_t>& shape, 
        DType dtype, 
        Device device,
        bool requires_grad = false);

    //=== 序列化 ===//
    void serialize(std::ostream& os) const;
    static std::shared_ptr<TensorImpl> deserialize(std::istream& is);

    //=== 属性访问 ===//
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t offset() const { return offset_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t nbytes() const { return nbytes_; }
    size_t alignment() const { return alignment_; }
    bool is_pinned() const { return is_pinned_; }
    bool is_contiguous() const { return is_contiguous_; }
    void* data_ptr() { return data_; }
    const void* data_ptr() const { return data_; }
    int64_t numel() const;
    MemoryOwnership memory_ownership() const { return memory_ownership_; }

    //=== 调试工具 ===//
    std::string debug_info() const;

private:
    //=== 内部实现 ===//
    void compute_strides();
    bool check_contiguous() const;
    void check_shape(const std::vector<int64_t>& shape) const;
    void update_contiguity();
    
    // 内存分配
    void cpu_allocate(size_t alignment);
    void cpu_deallocate();
    
    #ifdef WITH_CUDA
    void cuda_allocate(bool pinned);
    void cuda_deallocate();
    void cuda_copy_from_host(const void* src);
    void cuda_copy_to_host(void* dst) const;
    #endif

    // 错误处理
    [[noreturn]] void throw_error(const std::string& msg) const;

    //=== 数据成员 ===//
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t offset_ = 0;
    bool is_contiguous_ = true;
    void* data_ = nullptr;
    DType dtype_;
    Device device_;
    size_t nbytes_ = 0;
    size_t alignment_ = 0;
    bool is_pinned_ = false;
    MemoryOwnership memory_ownership_;

    // 线程安全
    mutable std::mutex shape_mutex_;
    std::atomic<int> refcount_{1};

    // 自动微分
    bool requires_grad_ = false;
    std::shared_ptr<TensorImpl> grad_;
    std::shared_ptr<autograd::Function> grad_fn_;

};

// 工具函数
inline std::string shape_to_string(const std::vector<int64_t>& shape);

} // namespace core