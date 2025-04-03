// core/autograd/function/math/add.h
#pragma once
#include "core/autograd/function/base.h"

namespace core {
namespace autograd {

class AddFunction : public Function {
public:
    std::vector<Variable> apply(const std::vector<Variable>& inputs) override;
    void backward(const std::vector<Variable>& grad_outputs) override;
    
    static std::shared_ptr<AddFunction> make();
};

} // namespace autograd
} // namespace core