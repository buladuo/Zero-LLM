// core/autograd/function/math/matmul.h
#pragma once
#include "core/autograd/function/base.h"

namespace core {
namespace autograd {

class MatMulFunction : public Function {
public:
    std::vector<Variable> apply(const std::vector<Variable>& inputs) override;
    void backward(const std::vector<Variable>& grad_outputs) override;
    
    static std::shared_ptr<MatMulFunction> make();
};

} // namespace autograd
} // namespace core