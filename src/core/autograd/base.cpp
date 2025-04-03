// core/autograd/function/base.cpp
#include "core/autograd/function/base.h"

namespace core {
namespace autograd {

void Function::set_next_edges(const std::vector<std::shared_ptr<Function>>& next_functions) {
    next_functions_ = next_functions;
}

const std::vector<std::shared_ptr<Function>>& Function::get_next_functions() const {
    return next_functions_;
}

void Function::save_for_backward(const std::vector<Variable>& variables) {
    saved_variables_ = variables;
}

std::vector<Variable> Function::get_saved_variables() const {
    return saved_variables_;
}

void Function::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
}

bool Function::requires_grad() const {
    return requires_grad_;
}

} // namespace autograd
} // namespace core