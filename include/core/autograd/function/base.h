// core/autograd/function/base.h
#pragma once
#include <memory>
#include <vector>
#include "core/tensor.h"
#include "core/autograd/variable.h"

namespace core {
namespace autograd {

class Function : public std::enable_shared_from_this<Function> {
public:
    virtual ~Function() = default;
    
    virtual std::vector<Variable> apply(const std::vector<Variable>& inputs) = 0;
    virtual void backward(const std::vector<Variable>& grad_outputs) = 0;
    
    void set_next_edges(const std::vector<std::shared_ptr<Function>>& next_functions);
    const std::vector<std::shared_ptr<Function>>& get_next_functions() const;
    
    void save_for_backward(const std::vector<Variable>& variables);
    std::vector<Variable> get_saved_variables() const;
    
    void set_requires_grad(bool requires_grad);
    bool requires_grad() const;
    
protected:
    std::vector<std::shared_ptr<Function>> next_functions_;
    std::vector<Variable> saved_variables_;
    bool requires_grad_ = true;
};

} // namespace autograd
} // namespace core