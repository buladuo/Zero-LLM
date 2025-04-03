// core/autograd/function/math/add.cpp
#include "core/autograd/function/math/add.h"
#include "core/tensor_ops.h"

namespace core {
namespace autograd {

std::vector<Variable> AddFunction::apply(const std::vector<Variable>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("AddFunction expects exactly 2 inputs");
    }
    
    // 前向传播
    Tensor result = inputs[0].data() + inputs[1].data();
    
    // 保存输入用于反向传播
    save_for_backward(inputs);
    
    return {Variable(result, inputs[0].requires_grad() || inputs[1].requires_grad())};
}

void AddFunction::backward(const std::vector<Variable>& grad_outputs) {
    if (grad_outputs.size() != 1) {
        throw std::runtime_error("AddFunction expects exactly 1 gradient output");
    }
    
    auto saved = get_saved_variables();
    if (saved.size() != 2) {
        throw std::runtime_error("Invalid saved variables in AddFunction");
    }
    
    // 加法操作的梯度就是输入的梯度
    if (saved[0].requires_grad()) {
        saved[0].set_grad(grad_outputs[0].data());
    }
    if (saved[1].requires_grad()) {
        saved[1].set_grad(grad_outputs[0].data());
    }
}

std::shared_ptr<AddFunction> AddFunction::make() {
    return std::shared_ptr<AddFunction>(new AddFunction);
}

} // namespace autograd
} // namespace core