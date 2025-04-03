// tensor_ops.cpp
#include "tensor_ops.h"
#include <cmath>
#include <algorithm>


namespace core {

    // 辅助函数：计算归约后的shape
    std::vector<int64_t> TensorOps::compute_reduced_shape(
        const std::vector<int64_t>& shape, 
        int64_t dim, 
        bool keepdim) {
        std::vector<int64_t> new_shape = shape;
        if (keepdim) {
            new_shape[dim] = 1;
        } else {
            new_shape.erase(new_shape.begin() + dim);
        }
        return new_shape;
    }
    
    // 辅助函数：规范化维度
    int64_t TensorOps::normalize_dim(int64_t dim, int64_t ndim) {
        if (dim < 0) {
            dim += ndim;
        }
        if (dim < 0 || dim >= ndim) {
            throw std::runtime_error("Dimension out of range");
        }
        return dim;
    }
    
    template <typename T>
    void TensorOps::stable_softmax(const T* input, T* output, int64_t size) {
        if (size == 0) return;
        
        T max_val = input[0];
        for (int64_t i = 1; i < size; ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
        
        T sum = 0;
        for (int64_t i = 0; i < size; ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        
        // 避免除以零
        if (sum == 0) {
            std::fill(output, output + size, T(1.0) / size);
        } else {
            for (int64_t i = 0; i < size; ++i) {
                output[i] /= sum;
            }
        }
    }

    Tensor TensorOps::matmul(const Tensor& a, const Tensor& b) {
        if (a.dim() != 2 || b.dim() != 2) {
            throw std::runtime_error("matmul only implemented for 2D tensors");
        }
        if (a.shape()[1] != b.shape()[0]) {
            throw std::runtime_error("Shape mismatch in matmul");
        }
    
        Tensor result({a.shape()[0], b.shape()[1]}, a.dtype(), a.device());
        
        if (a.device().type() == DeviceType::kCPU) {
            if (a.dtype() == kFloat32) {
                const float* a_data = static_cast<const float*>(a.data_ptr());
                const float* b_data = static_cast<const float*>(b.data_ptr());
                float* r_data = static_cast<float*>(result.data_ptr());
                
                const int64_t m = a.shape()[0];
                const int64_t k = a.shape()[1];
                const int64_t n = b.shape()[1];
                
                for (int64_t i = 0; i < m; ++i) {
                    for (int64_t j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (int64_t l = 0; l < k; ++l) {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        r_data[i * n + j] = sum;
                    }
                }
            } else {
                throw std::runtime_error("matmul only implemented for float32");
            }
        } else {
            throw std::runtime_error("matmul only implemented on CPU");
        }
        
        return result;
    }
    
    Tensor TensorOps::dot(const Tensor& a, const Tensor& b) {
        if (a.dim() != 1 || b.dim() != 1) {
            throw std::runtime_error("dot only implemented for 1D tensors");
        }
        if (a.shape()[0] != b.shape()[0]) {
            throw std::runtime_error("Shape mismatch in dot");
        }
    
        Tensor result({}, a.dtype(), a.device());
        
        if (a.device().type() == DeviceType::kCPU) {
            if (a.dtype() == kFloat32) {
                const float* a_data = static_cast<const float*>(a.data_ptr());
                const float* b_data = static_cast<const float*>(b.data_ptr());
                float* r_data = static_cast<float*>(result.data_ptr());
                
                float sum = 0.0f;
                for (int64_t i = 0; i < a.shape()[0]; ++i) {
                    sum += a_data[i] * b_data[i];
                }
                *r_data = sum;
            } else {
                throw std::runtime_error("dot only implemented for float32");
            }
        } else {
            throw std::runtime_error("dot only implemented on CPU");
        }
        
        return result;
    }
    
//=== 归约操作 ===//

Tensor TensorOps::sum(const Tensor& input, int64_t dim, bool keepdim) {
    const int64_t ndim = input.dim();
    dim = normalize_dim(dim, ndim);
    
    auto result_shape = compute_reduced_shape(input.shape(), dim, keepdim);
    Tensor result(result_shape, input.dtype(), input.device());
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            const float* data = static_cast<const float*>(input.data_ptr());
            float* out_data = static_cast<float*>(result.data_ptr());
            
            const int64_t outer_size = std::accumulate(
                input.shape().begin(), input.shape().begin() + dim, 1LL, 
                std::multiplies<int64_t>());
            const int64_t inner_size = std::accumulate(
                input.shape().begin() + dim + 1, input.shape().end(), 1LL, 
                std::multiplies<int64_t>());
            const int64_t dim_size = input.shape()[dim];
            
            for (int64_t i = 0; i < outer_size; ++i) {
                for (int64_t k = 0; k < inner_size; ++k) {
                    float sum = 0.0f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t idx = i * dim_size * inner_size + j * inner_size + k;
                        sum += data[idx];
                    }
                    const int64_t out_idx = i * inner_size + k;
                    out_data[out_idx] = sum;
                }
            }
        } else {
            throw std::runtime_error("sum only implemented for float32");
        }
    } else {
        throw std::runtime_error("sum only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::mean(const Tensor& input, int64_t dim, bool keepdim) {
    Tensor result = sum(input, dim, keepdim);
    const int64_t ndim = input.dim();
    const int64_t normalized_dim = normalize_dim(dim, ndim);
    const int64_t dim_size = input.shape()[normalized_dim];
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            float* data = static_cast<float*>(result.data_ptr());
            const int64_t num_elements = result.numel();
            for (int64_t i = 0; i < num_elements; ++i) {
                data[i] /= dim_size;
            }
        } else {
            throw std::runtime_error("mean only implemented for float32");
        }
    } else {
        throw std::runtime_error("mean only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::max(const Tensor& input, int64_t dim, bool keepdim) {
    const int64_t ndim = input.dim();
    dim = normalize_dim(dim, ndim);
    
    auto result_shape = compute_reduced_shape(input.shape(), dim, keepdim);
    Tensor result(result_shape, input.dtype(), input.device());
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            const float* data = static_cast<const float*>(input.data_ptr());
            float* out_data = static_cast<float*>(result.data_ptr());
            
            const int64_t outer_size = std::accumulate(
                input.shape().begin(), input.shape().begin() + dim, 1LL, 
                std::multiplies<int64_t>());
            const int64_t inner_size = std::accumulate(
                input.shape().begin() + dim + 1, input.shape().end(), 1LL, 
                std::multiplies<int64_t>());
            const int64_t dim_size = input.shape()[dim];
            
            for (int64_t i = 0; i < outer_size; ++i) {
                for (int64_t k = 0; k < inner_size; ++k) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t idx = i * dim_size * inner_size + j * inner_size + k;
                        max_val = std::max(max_val, data[idx]);
                    }
                    const int64_t out_idx = i * inner_size + k;
                    out_data[out_idx] = max_val;
                }
            }
        } else {
            throw std::runtime_error("max only implemented for float32");
        }
    } else {
        throw std::runtime_error("max only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::min(const Tensor& input, int64_t dim, bool keepdim) {
    const int64_t ndim = input.dim();
    dim = normalize_dim(dim, ndim);
    
    auto result_shape = compute_reduced_shape(input.shape(), dim, keepdim);
    Tensor result(result_shape, input.dtype(), input.device());
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            const float* data = static_cast<const float*>(input.data_ptr());
            float* out_data = static_cast<float*>(result.data_ptr());
            
            const int64_t outer_size = std::accumulate(
                input.shape().begin(), input.shape().begin() + dim, 1LL, 
                std::multiplies<int64_t>());
            const int64_t inner_size = std::accumulate(
                input.shape().begin() + dim + 1, input.shape().end(), 1LL, 
                std::multiplies<int64_t>());
            const int64_t dim_size = input.shape()[dim];
            
            for (int64_t i = 0; i < outer_size; ++i) {
                for (int64_t k = 0; k < inner_size; ++k) {
                    float min_val = std::numeric_limits<float>::infinity();
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t idx = i * dim_size * inner_size + j * inner_size + k;
                        min_val = std::min(min_val, data[idx]);
                    }
                    const int64_t out_idx = i * inner_size + k;
                    out_data[out_idx] = min_val;
                }
            }
        } else {
            throw std::runtime_error("min only implemented for float32");
        }
    } else {
        throw std::runtime_error("min only implemented on CPU");
    }
    
    return result;
}





//=== 激活函数 ===//

Tensor TensorOps::relu(const Tensor& input) {
    Tensor result(input.shape(), input.dtype(), input.device());
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            const float* data = static_cast<const float*>(input.data_ptr());
            float* out_data = static_cast<float*>(result.data_ptr());
            
            for (int64_t i = 0; i < input.numel(); ++i) {
                out_data[i] = std::max(0.0f, data[i]);
            }
        } else {
            throw std::runtime_error("relu only implemented for float32");
        }
    } else {
        throw std::runtime_error("relu only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::gelu(const Tensor& input) {
    Tensor result(input.shape(), input.dtype(), input.device());
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            const float* data = static_cast<const float*>(input.data_ptr());
            float* out_data = static_cast<float*>(result.data_ptr());
            
            // GELU近似实现: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
            constexpr float sqrt_2_over_pi = 0.7978845608f;
            constexpr float gelu_coeff = 0.044715f;
            
            for (int64_t i = 0; i < input.numel(); ++i) {
                float x = data[i];
                float x_cubed = x * x * x;
                float inner = sqrt_2_over_pi * (x + gelu_coeff * x_cubed);
                out_data[i] = 0.5f * x * (1.0f + std::tanh(inner));
            }
        } else {
            throw std::runtime_error("gelu only implemented for float32");
        }
    } else {
        throw std::runtime_error("gelu only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::sigmoid(const Tensor& input) {
    Tensor result(input.shape(), input.dtype(), input.device());
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            const float* data = static_cast<const float*>(input.data_ptr());
            float* out_data = static_cast<float*>(result.data_ptr());
            
            for (int64_t i = 0; i < input.numel(); ++i) {
                out_data[i] = 1.0f / (1.0f + std::exp(-data[i]));
            }
        } else {
            throw std::runtime_error("sigmoid only implemented for float32");
        }
    } else {
        throw std::runtime_error("sigmoid only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::tanh(const Tensor& input) {
    Tensor result(input.shape(), input.dtype(), input.device());
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            const float* data = static_cast<const float*>(input.data_ptr());
            float* out_data = static_cast<float*>(result.data_ptr());
            
            for (int64_t i = 0; i < input.numel(); ++i) {
                out_data[i] = std::tanh(data[i]);
            }
        } else {
            throw std::runtime_error("tanh only implemented for float32");
        }
    } else {
        throw std::runtime_error("tanh only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::softmax(const Tensor& input, int64_t dim) {
    const int64_t ndim = input.dim();
    dim = normalize_dim(dim, ndim);
    
    Tensor result(input.shape(), input.dtype(), input.device());
    
    if (input.device().type() == DeviceType::kCPU) {
        if (input.dtype() == kFloat32) {
            const float* data = static_cast<const float*>(input.data_ptr());
            float* out_data = static_cast<float*>(result.data_ptr());
            
            const int64_t outer_size = std::accumulate(
                input.shape().begin(), input.shape().begin() + dim, 1LL, 
                std::multiplies<int64_t>());
            const int64_t inner_size = std::accumulate(
                input.shape().begin() + dim + 1, input.shape().end(), 1LL, 
                std::multiplies<int64_t>());
            const int64_t dim_size = input.shape()[dim];
            
            for (int64_t i = 0; i < outer_size; ++i) {
                for (int64_t k = 0; k < inner_size; ++k) {
                    const int64_t base_idx = i * dim_size * inner_size + k;
                    stable_softmax(data + base_idx, out_data + base_idx, dim_size);
                }
            }
        } else {
            throw std::runtime_error("softmax only implemented for float32");
        }
    } else {
        throw std::runtime_error("softmax only implemented on CPU");
    }
    
    return result;
}
Tensor TensorOps::add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch in tensor addition");
    }
    
    Tensor result(a.shape(), a.dtype(), a.device());
    
    if (a.device().type() == DeviceType::kCPU) {
        if (a.dtype() == kFloat32) {
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            float* r_data = static_cast<float*>(result.data_ptr());
            
            for (int64_t i = 0; i < a.numel(); ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
        } else {
            throw std::runtime_error("add only implemented for float32");
        }
    } else {
        throw std::runtime_error("add only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch in tensor subtraction");
    }
    
    Tensor result(a.shape(), a.dtype(), a.device());
    
    if (a.device().type() == DeviceType::kCPU) {
        if (a.dtype() == kFloat32) {
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            float* r_data = static_cast<float*>(result.data_ptr());
            
            for (int64_t i = 0; i < a.numel(); ++i) {
                r_data[i] = a_data[i] - b_data[i];
            }
        } else {
            throw std::runtime_error("sub only implemented for float32");
        }
    } else {
        throw std::runtime_error("sub only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch in tensor multiplication");
    }
    
    Tensor result(a.shape(), a.dtype(), a.device());
    
    if (a.device().type() == DeviceType::kCPU) {
        if (a.dtype() == kFloat32) {
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            float* r_data = static_cast<float*>(result.data_ptr());
            
            for (int64_t i = 0; i < a.numel(); ++i) {
                r_data[i] = a_data[i] * b_data[i];
            }
        } else {
            throw std::runtime_error("mul only implemented for float32");
        }
    } else {
        throw std::runtime_error("mul only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch in tensor division");
    }
    
    Tensor result(a.shape(), a.dtype(), a.device());
    
    if (a.device().type() == DeviceType::kCPU) {
        if (a.dtype() == kFloat32) {
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            float* r_data = static_cast<float*>(result.data_ptr());
            
            for (int64_t i = 0; i < a.numel(); ++i) {
                if (b_data[i] == 0.0f) {
                    throw std::runtime_error("Division by zero");
                }
                r_data[i] = a_data[i] / b_data[i];
            }
        } else {
            throw std::runtime_error("div only implemented for float32");
        }
    } else {
        throw std::runtime_error("div only implemented on CPU");
    }
    
    return result;
}

Tensor TensorOps::batch_matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() != 3 || b.dim() != 3) {
        throw std::runtime_error("batch_matmul only implemented for 3D tensors");
    }
    if (a.shape()[0] != b.shape()[0]) {
        throw std::runtime_error("Batch size mismatch in batch_matmul");
    }
    if (a.shape()[2] != b.shape()[1]) {
        throw std::runtime_error("Shape mismatch in batch_matmul");
    }

    Tensor result({a.shape()[0], a.shape()[1], b.shape()[2]}, a.dtype(), a.device());
    
    if (a.device().type() == DeviceType::kCPU) {
        if (a.dtype() == kFloat32) {
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            float* r_data = static_cast<float*>(result.data_ptr());
            
            const int64_t batch_size = a.shape()[0];
            const int64_t m = a.shape()[1];
            const int64_t k = a.shape()[2];
            const int64_t n = b.shape()[2];
            
            for (int64_t b = 0; b < batch_size; ++b) {
                for (int64_t i = 0; i < m; ++i) {
                    for (int64_t j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (int64_t l = 0; l < k; ++l) {
                            sum += a_data[b * m * k + i * k + l] * 
                                    b_data[b * k * n + l * n + j];
                        }
                        r_data[b * m * n + i * n + j] = sum;
                    }
                }
            }
        } else {
            throw std::runtime_error("batch_matmul only implemented for float32");
        }
    } else {
        throw std::runtime_error("batch_matmul only implemented on CPU");
    }
    
    return result;
}

// 交换两个维度的位置
static Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1) {
    return input.transpose(dim0, dim1);
}

// 重新排列张量的维度顺序
static Tensor permute(const Tensor& input, const std::vector<int64_t>& dims) {
    return input.permute(dims);
}

// 改变张量的形状 (假设Tensor类有reshape成员函数)
static Tensor reshape(const Tensor& input, const std::vector<int64_t>& shape) {
    return input.reshape(shape);
}

// 改变张量的视图 (假设Tensor类有view成员函数)
static Tensor view(const Tensor& input, const std::vector<int64_t>& shape) {
    return input.view(shape);
}

// 移除长度为1的维度
static Tensor squeeze(const Tensor& input, int64_t dim = -1) {
    return input.squeeze(dim);
}

// 在指定位置插入长度为1的维度
static Tensor unsqueeze(const Tensor& input, int64_t dim) {
    return input.unsqueeze(dim);
}


} // namespace core