#pragma once
#include <memory>
#include <vector>
#include <unordered_set>
#include "core/tensor.h"

namespace core {
namespace autograd {

class Function;

class Engine {
public:
    static Engine& get_default_engine();
    
    void execute(const std::vector<std::shared_ptr<Function>>& roots);
    
private:
    Engine() = default;
    void compute_dependencies(const std::shared_ptr<Function>& root,
                             std::unordered_set<std::shared_ptr<Function>>& seen);
    void compute_dependencies(const std::shared_ptr<Function>& root,
                                std::unordered_set<std::shared_ptr<Function>>& seen,
                                std::vector<std::shared_ptr<Function>>& order);
};

} // namespace autograd
} // namespace core