#pragma once

#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <initializer_list>
#include "macros.h"
#include "device.h"
#include "dtype.h"
#include "tensor_impl.h"

namespace core {

// 前向声明
class TensorImpl;

class Tensor {
public:
    // 构造函数
    Tensor();
    explicit Tensor(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
    Tensor(const std::vector<int64_t>& shape, void* data, DType dtype = kFloat32, Device device = kCPU);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    
    // 从初始化列表构造
    Tensor(std::initializer_list<float> values);
    Tensor(std::initializer_list<std::initializer_list<float>> values);
    
    ~Tensor();

    // 赋值操作符
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // 基本属性访问
    int64_t dim() const;
    const std::vector<int64_t>& shape() const;
    int64_t size() const;
    int64_t numel() const;
    DType dtype() const;
    Device device() const;
    Tensor contiguous() const;
    bool is_contiguous() const;
    void* data_ptr();
    const void* data_ptr() const;
    
    // 设备转换
    Tensor to(Device device) const;
    Tensor cpu() const;
    Tensor cuda() const;
    
    // 类型转换
    Tensor to(DType dtype) const;
    Tensor float32() const;
    Tensor float16() const;
    Tensor bfloat16() const;
    
    // 形状操作
    Tensor view(const std::vector<int64_t>& shape) const;
    Tensor reshape(const std::vector<int64_t>& shape) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(const std::vector<int64_t>& dims) const;
    Tensor squeeze(int64_t dim = -1) const;
    Tensor unsqueeze(int64_t dim) const;
    
    // 索引操作
    Tensor operator[](int64_t index) const;
    Tensor select(int64_t dim, int64_t index) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    
    // 填充操作
    void fill_(float value);
    void zero_();
    
    // 随机初始化
    static Tensor rand(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
    static Tensor randn(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
    
    // 特殊张量创建
    static Tensor zeros(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
    static Tensor ones(const std::vector<int64_t>& shape, DType dtype = kFloat32, Device device = kCPU);
    static Tensor eye(int64_t n, DType dtype = kFloat32, Device device = kCPU);
    static Tensor arange(int64_t end, DType dtype = kFloat32, Device device = kCPU);
    static Tensor arange(int64_t start, int64_t end, int64_t step, DType dtype = kFloat32, Device device = kCPU);
    
    // 张量拼接
    static Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0);
    
    // 操作符重载
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    Tensor operator-() const;
    
    // 比较操作符
    Tensor operator==(const Tensor& other) const;
    Tensor operator!=(const Tensor& other) const;
    Tensor operator<(const Tensor& other) const;
    Tensor operator<=(const Tensor& other) const;
    Tensor operator>(const Tensor& other) const;
    Tensor operator>=(const Tensor& other) const;
    
    // 矩阵乘法
    Tensor matmul(const Tensor& other) const;
    
    // 点乘
    Tensor dot(const Tensor& other) const;
    
    // 归约操作
    Tensor sum(int64_t dim = -1, bool keepdim = false) const;
    Tensor mean(int64_t dim = -1, bool keepdim = false) const;
    Tensor max(int64_t dim = -1, bool keepdim = false) const;
    Tensor min(int64_t dim = -1, bool keepdim = false) const;
    
    // 激活函数
    Tensor relu() const;
    Tensor gelu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor softmax(int64_t dim = -1) const;
    Tensor log_softmax(int64_t dim = -1) const;
    
    // 自动微分相关
    void backward(const Tensor& grad = Tensor());
    Tensor grad() const;
    void set_requires_grad(bool requires_grad);
    bool requires_grad() const;
    
    // 打印张量
    std::string to_string() const;
    void print() const;
    
    // 序列化/反序列化
    void save(const std::string& filename) const;
    static Tensor load(const std::string& filename);

    static std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}
    
private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
    friend Tensor make_tensor_from_impl(std::shared_ptr<TensorImpl> impl);

    std::shared_ptr<TensorImpl> impl_;
    friend class TensorImpl;
};

} // namespace core