#include "tensor.h"
#include <algorithm>
#include <stdexcept>

namespace tensor {

// ========== Level 1: 基础形状操作 ==========

Tensor Tensor::slice(int dim, int64_t start, int64_t end, int64_t step) const {
    if (dim < 0) dim += shape_.size();
    if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Dimension out of range");
    }

    if (start < 0) start += shape_[dim];
    if (end < 0) end += shape_[dim];

    start = std::max(int64_t(0), std::min(start, shape_[dim]));
    end = std::max(int64_t(0), std::min(end, shape_[dim]));

    if (start >= end || step <= 0) {
        throw std::runtime_error("Invalid slice parameters");
    }

    std::vector<int64_t> new_shape = shape_;
    new_shape[dim] = (end - start + step - 1) / step;

    std::vector<int64_t> new_strides = strides_;
    new_strides[dim] = strides_[dim] * step;

    int64_t new_offset = offset_ + start * strides_[dim];

    bool new_contiguous = false;
    if (contiguous_ && step == 1) {
        // Only contiguous if slicing the last dim AND taking all elements,
        // OR slicing non-last dim AND the slice size equals original size
        if (dim == static_cast<int>(shape_.size()) - 1 && new_shape[dim] == shape_[dim]) {
            new_contiguous = true;
        } else if (dim != static_cast<int>(shape_.size()) - 1 && new_shape[dim] == shape_[dim]) {
            new_contiguous = true;
        }
    }

    return Tensor(storage_, new_shape, new_strides, new_offset, new_contiguous);
}

Tensor Tensor::select_dim(int dim, int64_t index) const {
    if (dim < 0) dim += shape_.size();
    if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Dimension out of range");
    }

    if (index < 0) index += shape_[dim];
    if (index < 0 || index >= shape_[dim]) {
        throw std::runtime_error("Index out of range");
    }

    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (static_cast<int>(i) != dim) {
            new_shape.push_back(shape_[i]);
            new_strides.push_back(strides_[i]);
        }
    }

    int64_t new_offset = offset_ + index * strides_[dim];

    return Tensor(storage_, new_shape, new_strides, new_offset, false);
}

Tensor Tensor::get_column(int64_t col_idx) const {
    if (shape_.size() != 2) {
        throw std::runtime_error("get_column only works with 2D tensors");
    }
    return select_dim(1, col_idx);
}

Tensor Tensor::slice_columns(int64_t start, int64_t end) const {
    if (shape_.size() != 2) {
        throw std::runtime_error("slice_columns only works with 2D tensors");
    }
    return slice(1, start, end, 1);
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    int64_t new_size = 1;
    int infer_dim = -1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::runtime_error("Only one dimension can be -1");
            }
            infer_dim = i;
        } else {
            new_size *= new_shape[i];
        }
    }

    std::vector<int64_t> final_shape = new_shape;
    if (infer_dim != -1) {
        int64_t current_size = compute_size();
        if (current_size % new_size != 0) {
            throw std::runtime_error("Cannot infer dimension size");
        }
        final_shape[infer_dim] = current_size / new_size;
        new_size = current_size;
    }

    if (new_size != compute_size()) {
        throw std::runtime_error("Shape size mismatch");
    }

    if (!contiguous_) {
        return contiguous_copy().reshape(final_shape);
    }

    return Tensor(storage_, final_shape, compute_strides(final_shape), offset_, true);
}

Tensor Tensor::transpose(const std::vector<int>& dims) const {
    if (dims.size() != shape_.size()) {
        throw std::runtime_error("Transpose dimensions mismatch");
    }

    std::vector<bool> used(dims.size(), false);
    for (int dim : dims) {
        int d = dim;
        if (d < 0) d += dims.size();
        if (d < 0 || d >= static_cast<int>(dims.size()) || used[d]) {
            throw std::runtime_error("Invalid transpose dimensions");
        }
        used[d] = true;
    }

    std::vector<int64_t> new_shape(shape_.size());
    std::vector<int64_t> new_strides(strides_.size());

    for (size_t i = 0; i < dims.size(); ++i) {
        int dim = dims[i];
        if (dim < 0) dim += dims.size();
        new_shape[i] = shape_[dim];
        new_strides[i] = strides_[dim];
    }

    return Tensor(storage_, new_shape, new_strides, offset_, false);
}

Tensor Tensor::transpose() const {
    std::vector<int> dims(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
        dims[i] = shape_.size() - 1 - i;
    }
    return transpose(dims);
}

Tensor Tensor::squeeze(int dim) const {
    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;

    if (dim == -1) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] != 1) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        }
    } else {
        if (dim < 0) dim += shape_.size();
        if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
            throw std::runtime_error("Dimension out of range");
        }
        if (shape_[dim] != 1) {
            throw std::runtime_error("Cannot squeeze dimension with size != 1");
        }

        for (size_t i = 0; i < shape_.size(); ++i) {
            if (static_cast<int>(i) != dim) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        }
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
        new_strides.push_back(1);
    }

    return Tensor(storage_, new_shape, new_strides, offset_, contiguous_);
}

Tensor Tensor::unsqueeze(int dim) const {
    int ndim = shape_.size();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) {
        throw std::runtime_error("Dimension out of range");
    }

    std::vector<int64_t> new_shape = shape_;
    std::vector<int64_t> new_strides = strides_;

    new_shape.insert(new_shape.begin() + dim, 1);
    int64_t new_stride = (dim < ndim) ? strides_[dim] : 1;
    new_strides.insert(new_strides.begin() + dim, new_stride);

    return Tensor(storage_, new_shape, new_strides, offset_, contiguous_);
}

} // namespace tensor
