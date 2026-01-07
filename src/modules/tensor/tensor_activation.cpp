#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tensor {

// ========== Activation 函数 ==========

Tensor Tensor::sigmoid() const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::softmax(int axis) const {
    check_cpu();
    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    Tensor a = contiguous();
    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    if (ax != static_cast<int>(shape_.size()) - 1) {
        throw std::runtime_error("Softmax only supports last axis for now");
    }

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t inner_size = shape_[ax];

    for (int64_t i = 0; i < outer_size; ++i) {
        const float* row = src + i * inner_size;
        float* out_row = result_data.data() + i * inner_size;

        float max_val = row[0];
        for (int64_t j = 1; j < inner_size; ++j) {
            max_val = std::max(max_val, row[j]);
        }

        float sum = 0.0f;
        for (int64_t j = 0; j < inner_size; ++j) {
            out_row[j] = std::exp(row[j] - max_val);
            sum += out_row[j];
        }

        float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < inner_size; ++j) {
            out_row[j] *= inv_sum;
        }
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::exp_() const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = std::exp(src[i]);
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::log_() const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = std::log(src[i]);
    }

    return Tensor(std::move(result_data), shape_);
}

} // namespace tensor
