#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tensor {

// ========== 比较操作 ==========

Tensor Tensor::gt(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] > threshold) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::lt(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] < threshold) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::ge(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] >= threshold) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::le(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] <= threshold) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::eq(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (std::abs(src[i] - threshold) < 1e-6f) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

} // namespace tensor
