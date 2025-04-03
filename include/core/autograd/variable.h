#pragma once
#include "core/tensor.h"
#include "core/autograd/function/base.h"

namespace core {
namespace autograd {

class Variable {
public:
    explicit Variable(Tensor data, bool requires_grad = false);
    
    Tensor& data();
    const Tensor& data() const;
    Tensor& grad();
    const Tensor& grad() const;
    
    void set_grad(const Tensor& grad);
    std::shared_ptr<Function> grad_fn() const;
    void set_grad_fn(const std::shared_ptr<Function>& grad_fn);
    
    void backward(const Tensor& grad = Tensor());
    
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    
private:
    Tensor data_;
    Tensor grad_;
    std::shared_ptr<Function> grad_fn_;
    bool requires_grad_;
};

} // namespace autograd
} // namespace core