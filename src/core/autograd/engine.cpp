#include "core/autograd/engine.h"
#include "core/autograd/function/base.h"
#include <condition_variable>
#include <thread>
#include "core/autograd/variable.h"

namespace core {
namespace autograd {

Engine &Engine::get_default_engine() {
    static Engine engine;
    return engine;
}

void Engine::execute(const std::vector<std::shared_ptr<Function>> &roots) {
    std::unordered_set<std::shared_ptr<Function>> seen;
    std::vector<std::shared_ptr<Function>> order;

    // 构建计算图拓扑排序
    for (const auto &root : roots) {
        compute_dependencies(root, seen, order);
    }

    // 反向传播执行
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        // 获取梯度输出
        std::vector<Variable> grad_outputs;
        for (const auto &next_fn : (*it)->next_functions()) {
            if (auto next_var = next_fn.lock()) {
                grad_outputs.push_back(Variable(next_var->grad(), false));
            }
        }

        // 执行反向传播
        (*it)->backward(grad_outputs);
    }
}

void Engine::compute_dependencies(
    const std::shared_ptr<Function> &root,
    std::unordered_set<std::shared_ptr<Function>> &seen,
    std::vector<std::shared_ptr<Function>> &order) {

    if (seen.count(root))
        return;
    seen.insert(root);

    for (const auto &next_fn : root->next_functions()) {
        if (auto next = next_fn.lock()) {
            compute_dependencies(next, seen, order);
        }
    }

    order.push_back(root);
}

void Engine::compute_dependencies( 
    const std::shared_ptr<Function> &root,
    std::unordered_set<std::shared_ptr<Function>> &seen) {
    if (seen.find(root) != seen.end())
        return;

    seen.insert(root);
    for (const auto &next_fn : root->get_next_functions()) {
        if (next_fn) {
            compute_dependencies(next_fn, seen);
        }
    }
}

} // namespace autograd
} // namespace core