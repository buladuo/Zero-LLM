#pragma once

#include "tensor.h"

namespace core {

class TensorOps {
public:
    // 基本运算
    static Tensor add(const Tensor& a, const Tensor& b);
    static Tensor sub(const Tensor& a, const Tensor& b);
    static Tensor mul(const Tensor& a, const Tensor& b);
    static Tensor dot(const Tensor& a, const Tensor& b);
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

private:
    // 辅助函数
    static int64_t normalize_dim(int64_t dim, int64_t ndim);
    static std::vector<int64_t> compute_reduced_shape(const std::vector<int64_t>& shape, int64_t dim, bool keepdim);
    template <typename T>
    static void stable_softmax(const T* input, T* output, int64_t size);
};

} // namespace core