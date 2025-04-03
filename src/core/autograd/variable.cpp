#include "core/autograd/variable.h"
#include "core/autograd/engine.h"
#include "core/autograd/function/base.h"

namespace core {
namespace autograd {

Variable::Variable(Tensor data, bool requires_grad)
    : data_(std::move(data)), requires_grad_(requires_grad) {}

Tensor& Variable::data() { return data_; }
const Tensor& Variable::data() const { return data_; }

Tensor& Variable::grad() { return grad_; }
void Variable::set_grad(const Tensor& grad) { grad_ = grad; }

bool Variable::requires_grad() const { return requires_grad_; }
void Variable::set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

std::shared_ptr<Function> Variable::grad_fn() const { return grad_fn_; }
void Variable::set_grad_fn(const std::shared_ptr<Function>& grad_fn) { grad_fn_ = grad_fn; }

void Variable::backward(const Tensor& grad) {
    if (!requires_grad_) return;
    
    if (!grad_fn_) {
        // 这是叶子节点
        if (!grad.defined()) {
            // 创建全1梯度
            grad_ = Tensor::ones_like(data_);
        } else {
            grad_ = grad;
        }
        return;
    }
    
    // 非叶子节点，通过引擎执行反向传播
    if (!grad.defined()) {
        grad_ = Tensor::ones_like(data_);
    } else {
        grad_ = grad;
    }
    
    Engine::get_default_engine().execute({grad_fn_});
}

} // namespace autograd
} // namespace core