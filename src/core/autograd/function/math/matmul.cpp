// core/autograd/function/math/matmul.cpp
#include "core/autograd/function/math/matmul.h"
#include "core/tensor_ops.h"

namespace core {
namespace autograd {

std::vector<Variable> MatMulFunction::apply(const std::vector<Variable>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("MatMulFunction expects exactly 2 inputs");
    }
    
    // 使用TensorOps::matmul进行前向传播
    Tensor result = TensorOps::matmul(inputs[0].data(), inputs[1].data());
    
    // 保存输入用于反向传播
    save_for_backward(inputs);
    
    return {Variable(result, inputs[0].requires_grad() || inputs[1].requires_grad())};
}

void MatMulFunction::backward(const std::vector<Variable>& grad_outputs) {
    if (grad_outputs.size() != 1) {
        throw std::runtime_error("MatMulFunction expects exactly 1 gradient output");
    }
    
    auto saved = get_saved_variables();
    if (saved.size() != 2) {
        throw std::runtime_error("Invalid saved variables in MatMulFunction");
    }
    
    const auto& grad = grad_outputs[0].data();
    const auto& a = saved[0].data();
    const auto& b = saved[1].data();
    
    if (saved[0].requires_grad()) {
        // 计算第一个输入的梯度: grad * b^T
        saved[0].set_grad(TensorOps::matmul(grad, TensorOps::transpose(b)));
    }
    if (saved[1].requires_grad()) {
        // 计算第二个输入的梯度: a^T * grad
        saved[1].set_grad(TensorOps::matmul(TensorOps::transpose(a), grad));
    }
}

std::shared_ptr<MatMulFunction> MatMulFunction::make() {
    return std::shared_ptr<MatMulFunction>(new MatMulFunction);
}

} // namespace autograd
} // namespace core