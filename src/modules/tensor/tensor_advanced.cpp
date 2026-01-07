#include "tensor.h"
#include "cpu_storage.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tensor {

// ========== Gather ==========

Tensor Tensor::gather(int axis, const Tensor& indices) const {
    check_cpu();
    indices.check_cpu();

    // Normalize axis
    int ndim = static_cast<int>(shape_.size());
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw std::runtime_error("gather: axis out of range");
    }

    // Output has same shape as indices
    auto out_shape = indices.shape();
    int64_t total = indices.size();

    std::vector<float> result_data(total);
    Tensor src_contig = contiguous();
    Tensor idx_contig = indices.contiguous();

    const float* src_ptr = src_contig.raw_data();
    const float* idx_ptr = idx_contig.raw_data();
    float* dst_ptr = result_data.data();

    // Compute strides for source tensor
    std::vector<int64_t> src_strides = src_contig.strides();

    // For each element in indices, gather from source
    std::vector<int64_t> coords(ndim);
    for (int64_t i = 0; i < total; ++i) {
        // Convert flat index to multi-dimensional coordinates
        int64_t remaining = i;
        for (int d = ndim - 1; d >= 0; --d) {
            coords[d] = remaining % out_shape[d];
            remaining /= out_shape[d];
        }

        // Get the index value and replace coord at gather axis
        int64_t gather_idx = static_cast<int64_t>(idx_ptr[i]);
        if (gather_idx < 0) gather_idx += shape_[axis];
        if (gather_idx < 0 || gather_idx >= shape_[axis]) {
            throw std::runtime_error("gather: index out of bounds");
        }

        // Compute source offset
        int64_t src_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            int64_t coord = (d == axis) ? gather_idx : coords[d];
            src_offset += coord * src_strides[d];
        }

        dst_ptr[i] = src_ptr[src_offset];
    }

    return Tensor(std::move(result_data), out_shape);
}

// ========== Concat ==========

Tensor Tensor::concat(const std::vector<Tensor>& tensors, int axis) {
    if (tensors.empty()) {
        throw std::runtime_error("concat: empty tensor list");
    }

    // All tensors must have same ndim
    int ndim = static_cast<int>(tensors[0].shape().size());
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw std::runtime_error("concat: axis out of range");
    }

    // Validate shapes: all dims except concat axis must match
    auto base_shape = tensors[0].shape();
    int64_t concat_dim_size = 0;

    for (const auto& t : tensors) {
        t.check_cpu();
        auto shape = t.shape();
        if (static_cast<int>(shape.size()) != ndim) {
            throw std::runtime_error("concat: all tensors must have same ndim");
        }
        for (int d = 0; d < ndim; ++d) {
            if (d != axis && shape[d] != base_shape[d]) {
                throw std::runtime_error("concat: shape mismatch on non-concat dimension");
            }
        }
        concat_dim_size += shape[axis];
    }

    // Build output shape
    auto out_shape = base_shape;
    out_shape[axis] = concat_dim_size;

    // Compute total size and allocate
    int64_t total = 1;
    for (auto s : out_shape) total *= s;
    std::vector<float> result_data(total);

    // Compute sizes for efficient copy
    // outer_size = product of dims before axis
    // inner_size = product of dims after axis
    int64_t outer_size = 1, inner_size = 1;
    for (int d = 0; d < axis; ++d) outer_size *= base_shape[d];
    for (int d = axis + 1; d < ndim; ++d) inner_size *= base_shape[d];

    float* dst = result_data.data();
    int64_t out_axis_stride = concat_dim_size * inner_size;

    for (int64_t o = 0; o < outer_size; ++o) {
        float* dst_row = dst + o * out_axis_stride;
        for (const auto& t : tensors) {
            Tensor tc = t.contiguous();
            const float* src = tc.raw_data();
            int64_t t_axis_size = t.shape()[axis];
            int64_t copy_size = t_axis_size * inner_size;

            const float* src_row = src + o * copy_size;
            std::memcpy(dst_row, src_row, copy_size * sizeof(float));
            dst_row += copy_size;
        }
    }

    return Tensor(std::move(result_data), out_shape);
}

// ========== Split ==========

std::vector<Tensor> Tensor::split(int num_splits, int axis) const {
    check_cpu();

    int ndim = static_cast<int>(shape_.size());
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw std::runtime_error("split: axis out of range");
    }

    int64_t axis_size = shape_[axis];
    if (axis_size % num_splits != 0) {
        throw std::runtime_error("split: tensor size not evenly divisible");
    }

    int64_t split_size = axis_size / num_splits;

    // Compute sizes for efficient copy
    int64_t outer_size = 1, inner_size = 1;
    for (int d = 0; d < axis; ++d) outer_size *= shape_[d];
    for (int d = axis + 1; d < ndim; ++d) inner_size *= shape_[d];

    Tensor src_contig = contiguous();
    const float* src = src_contig.raw_data();

    // Build output shape for each split
    auto split_shape = shape_;
    split_shape[axis] = split_size;
    int64_t split_total = 1;
    for (auto s : split_shape) split_total *= s;

    std::vector<Tensor> results;
    results.reserve(num_splits);

    int64_t src_axis_stride = axis_size * inner_size;
    int64_t dst_axis_stride = split_size * inner_size;

    for (int s = 0; s < num_splits; ++s) {
        std::vector<float> split_data(split_total);
        float* dst = split_data.data();

        for (int64_t o = 0; o < outer_size; ++o) {
            const float* src_row = src + o * src_axis_stride + s * dst_axis_stride;
            float* dst_row = dst + o * dst_axis_stride;
            std::memcpy(dst_row, src_row, dst_axis_stride * sizeof(float));
        }

        results.emplace_back(std::move(split_data), split_shape);
    }

    return results;
}


} // namespace tensor
