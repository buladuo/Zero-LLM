#pragma once

#include "tensor.h"

namespace core {

class TensorOps {
public:
    // 基本运算
    static Tensor add(const Tensor& a, const Tensor& b);
    static Tensor sub(const Tensor& a, const Tensor& b);
    static Tensor mul(const Tensor& a, const Tensor& b);
    static Tensor div(const Tensor& a, const Tensor& b);
    
    // 矩阵运算
    static Tensor matmul(const Tensor& a, const Tensor& b);
    static Tensor batch_matmul(const Tensor& a, const Tensor& b);
    
    // 激活函数
    static Tensor relu(const Tensor& input);
    static Tensor gelu(const Tensor& input);
    static Tensor sigmoid(const Tensor& input);
    static Tensor tanh(const Tensor& input);
    static Tensor silu(const Tensor& input);
    static Tensor softmax(const Tensor& input, int64_t dim);
    static Tensor log_softmax(const Tensor& input, int64_t dim);
    
    // 归约操作
    static Tensor sum(const Tensor& input, int64_t dim = -1, bool keepdim = false);
    static Tensor mean(const Tensor& input, int64_t dim = -1, bool keepdim = false);
    static Tensor max(const Tensor& input, int64_t dim = -1, bool keepdim = false);
    static Tensor min(const Tensor& input, int64_t dim = -1, bool keepdim = false);
    
    // 形状操作
    static Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1);
    static Tensor permute(const Tensor& input, const std::vector<int64_t>& dims);
    static Tensor reshape(const Tensor& input, const std::vector<int64_t>& shape);
    static Tensor view(const Tensor& input, const std::vector<int64_t>& shape);
    static Tensor squeeze(const Tensor& input, int64_t dim = -1);
    static Tensor unsqueeze(const Tensor& input, int64_t dim);
    
    // 设备/类型转换
    static Tensor to_device(const Tensor& input, Device device);
    static Tensor to_dtype(const Tensor& input, DType dtype);
    
    // 内存操作
    static Tensor contiguous(const Tensor& input);
    static Tensor clone(const Tensor& input);
    
    // 初始化操作
    static Tensor zeros_like(const Tensor& input);
    static Tensor ones_like(const Tensor& input);
    
    // 索引操作
    static Tensor index_select(const Tensor& input, int64_t dim, const Tensor& index);
    static Tensor masked_select(const Tensor& input, const Tensor& mask);
    
    // 比较操作
    static Tensor eq(const Tensor& a, const Tensor& b);
    static Tensor ne(const Tensor& a, const Tensor& b);
    static Tensor lt(const Tensor& a, const Tensor& b);
    static Tensor le(const Tensor& a, const Tensor& b);
    static Tensor gt(const Tensor& a, const Tensor& b);
    static Tensor ge(const Tensor& a, const Tensor& b);
    
    // 梯度相关
    static Tensor backward(const Tensor& output, const Tensor& grad = Tensor());
    static Tensor grad(const Tensor& input);
};

} // namespace core