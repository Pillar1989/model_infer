#include "tensor.h"
#include "cpu_storage.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <cstring>

#include "LuaIntf.h"
#include <opencv2/opencv.hpp>

namespace tensor {

// ========== 构造函数 ==========

Tensor::Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape)
    : shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true) {
    // 分配 CPU Storage 并复制数据
    storage_ = CpuStorage::allocate(data.size() * sizeof(float));
    std::memcpy(storage_->data(), data.data(), data.size() * sizeof(float));
}

Tensor::Tensor(std::vector<float>&& data, const std::vector<int64_t>& shape)
    : shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true) {
    // 分配 CPU Storage 并移动数据
    storage_ = CpuStorage::allocate(data.size() * sizeof(float));
    std::memcpy(storage_->data(), data.data(), data.size() * sizeof(float));
}

Tensor::Tensor(const float* data, const std::vector<int64_t>& shape,
               std::shared_ptr<TensorStorage> owner)
    : shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true) {
    if (owner) {
        storage_ = owner;
    } else {
        int64_t total_size = compute_size();
        storage_ = CpuStorage::allocate(total_size * sizeof(float));
        std::memcpy(storage_->data(), data, total_size * sizeof(float));
    }
}

Tensor::Tensor(std::shared_ptr<TensorStorage> storage,
               const std::vector<int64_t>& shape,
               const std::vector<int64_t>& strides,
               int64_t offset,
               bool contiguous)
    : storage_(storage)
    , shape_(shape)
    , strides_(strides)
    , offset_(offset)
    , contiguous_(contiguous) {}

// Static factory method for Lua binding
Tensor Tensor::create(const std::vector<float>& data, const std::vector<int64_t>& shape) {
    return Tensor(data, shape);
}

// Lua factory method - manually parse LuaRef tables
Tensor Tensor::from_lua(lua_State* L, const LuaIntf::LuaRef& data_table, const LuaIntf::LuaRef& shape_table) {
    // Parse data table
    std::vector<float> data;
    if (data_table.isTable()) {
        int len = data_table.len();
        data.reserve(len);
        for (int i = 1; i <= len; ++i) {
            data.push_back(static_cast<float>(data_table[i].value<double>()));
        }
    } else {
        throw std::runtime_error("data must be a table");
    }

    // Parse shape table
    std::vector<int64_t> shape;
    if (shape_table.isTable()) {
        int len = shape_table.len();
        shape.reserve(len);
        for (int i = 1; i <= len; ++i) {
            shape.push_back(static_cast<int64_t>(shape_table[i].value<int>()));
        }
    } else {
        throw std::runtime_error("shape must be a table");
    }

    return Tensor(data, shape);
}

// Lua wrapper - get single index
float Tensor::get_lua(int idx) const {
    check_cpu();
    if (idx < 0 || idx >= size()) {
        throw std::runtime_error("Index out of bounds");
    }
    const float* data = raw_data();
    if (contiguous_) {
        return data[offset_ + idx];
    } else {
        // Non-contiguous: need to compute actual index
        std::vector<int64_t> indices(shape_.size());
        int64_t remaining = idx;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            indices[i] = remaining % shape_[i];
            remaining /= shape_[i];
        }
        int64_t actual_idx = offset_;
        for (size_t i = 0; i < shape_.size(); ++i) {
            actual_idx += indices[i] * strides_[i];
        }
        return data[actual_idx];
    }
}

// Lua wrapper - set single index
void Tensor::set_lua(int idx, float value) {
    check_cpu();
    if (idx < 0 || idx >= size()) {
        throw std::runtime_error("Index out of bounds");
    }
    float* data = raw_data();
    if (contiguous_) {
        data[offset_ + idx] = value;
    } else {
        std::vector<int64_t> indices(shape_.size());
        int64_t remaining = idx;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            indices[i] = remaining % shape_[i];
            remaining /= shape_[i];
        }
        int64_t actual_idx = offset_;
        for (size_t i = 0; i < shape_.size(); ++i) {
            actual_idx += indices[i] * strides_[i];
        }
        data[actual_idx] = value;
    }
}

// Extract columns as new Tensor (takes vector directly, auto-converted from Lua table)
Tensor Tensor::extract_columns_tensor(const std::vector<int64_t>& cols) const {
    check_cpu();
    if (shape_.size() != 2) {
        throw std::runtime_error("extract_columns requires 2D tensor");
    }

    int64_t rows = shape_[0];
    int64_t num_cols = static_cast<int64_t>(cols.size());

    std::vector<float> result_data;
    result_data.reserve(rows * num_cols);

    Tensor contig = contiguous();
    const float* src = contig.raw_data();
    int64_t src_cols = shape_[1];

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c : cols) {
            result_data.push_back(src[r * src_cols + c]);
        }
    }

    return Tensor(std::move(result_data), {rows, num_cols});
}

// ========== 辅助方法 ==========

int64_t Tensor::size() const {
    return compute_size();
}

int64_t Tensor::size(int dim) const {
    if (dim < 0) dim += shape_.size();
    if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Dimension out of range");
    }
    return shape_[dim];
}

int64_t Tensor::compute_size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
}

std::vector<int64_t> Tensor::compute_strides(const std::vector<int64_t>& shape) const {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;

    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

void Tensor::normalize_axis(int& axis) const {
    if (axis < 0) axis += shape_.size();
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }
}

int64_t Tensor::compute_offset(const std::vector<int64_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Indices size mismatch");
    }

    int64_t offset = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        int64_t idx = indices[i];
        if (idx < 0) idx += shape_[i];
        if (idx < 0 || idx >= shape_[i]) {
            throw std::runtime_error("Index out of range");
        }
        offset += idx * strides_[i];
    }
    return offset;
}

void Tensor::check_cpu() const {
    if (storage_->device() != DeviceType::CPU) {
        throw std::runtime_error("Operation requires CPU tensor");
    }
}

// ========== 数据访问 ==========

const float* Tensor::raw_data() const {
    check_cpu();
    return static_cast<const float*>(storage_->data());
}

float* Tensor::raw_data() {
    check_cpu();
    return static_cast<float*>(storage_->data());
}

const float* Tensor::data() const {
    check_cpu();
    return static_cast<const float*>(storage_->data()) + offset_;
}

float* Tensor::data() {
    check_cpu();
    return static_cast<float*>(storage_->data()) + offset_;
}

// ========== 统一元素访问 at() ==========

float Tensor::at(int64_t i) const {
    check_cpu();
    if (shape_.size() != 1) {
        throw std::runtime_error("at(i) requires 1D tensor");
    }
    if (i < 0) i += shape_[0];
    if (i < 0 || i >= shape_[0]) {
        throw std::runtime_error("Index out of range");
    }
    const float* ptr = static_cast<const float*>(storage_->data());
    return ptr[offset_ + i * strides_[0]];
}

float Tensor::at(int64_t i, int64_t j) const {
    check_cpu();
    if (shape_.size() != 2) {
        throw std::runtime_error("at(i,j) requires 2D tensor");
    }
    if (i < 0) i += shape_[0];
    if (j < 0) j += shape_[1];
    if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1]) {
        throw std::runtime_error("Index out of range");
    }
    const float* ptr = static_cast<const float*>(storage_->data());
    return ptr[offset_ + i * strides_[0] + j * strides_[1]];
}

float Tensor::at(const std::vector<int64_t>& indices) const {
    check_cpu();
    int64_t offset = compute_offset(indices);
    const float* ptr = static_cast<const float*>(storage_->data());
    return ptr[offset];
}

float& Tensor::at(int64_t i) {
    check_cpu();
    if (shape_.size() != 1) {
        throw std::runtime_error("at(i) requires 1D tensor");
    }
    if (i < 0) i += shape_[0];
    if (i < 0 || i >= shape_[0]) {
        throw std::runtime_error("Index out of range");
    }
    float* ptr = static_cast<float*>(storage_->data());
    return ptr[offset_ + i * strides_[0]];
}

float& Tensor::at(int64_t i, int64_t j) {
    check_cpu();
    if (shape_.size() != 2) {
        throw std::runtime_error("at(i,j) requires 2D tensor");
    }
    if (i < 0) i += shape_[0];
    if (j < 0) j += shape_[1];
    if (i < 0 || i >= shape_[0] || j < 0 || j >= shape_[1]) {
        throw std::runtime_error("Index out of range");
    }
    float* ptr = static_cast<float*>(storage_->data());
    return ptr[offset_ + i * strides_[0] + j * strides_[1]];
}

// ========== 设备操作 ==========

Tensor Tensor::to(DeviceType device) const {
    if (storage_->device() == device) {
        return *this;
    }

    // 先确保连续
    Tensor cont = contiguous();

    // 分配目标设备存储
    auto new_storage = TensorStorage::allocate(cont.storage_->size_bytes(), device);

    // 复制数据
    cont.storage_->copy_to(new_storage.get());

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::contiguous() const {
    if (contiguous_) {
        return *this;
    }
    return contiguous_copy();
}

// ========== 异步设备操作 ==========

Tensor Tensor::to_async(DeviceType device, Stream* stream) const {
    if (storage_->device() == device) {
        return *this;
    }

    // 先确保连续
    Tensor cont = contiguous();

    // 分配目标设备存储
    auto new_storage = TensorStorage::allocate(cont.storage_->size_bytes(), device);

    // 异步复制数据
    cont.storage_->copy_to_async(new_storage.get(), stream);

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

void Tensor::sync(Stream* stream) const {
    storage_->sync(stream);
}

bool Tensor::is_ready(Stream* stream) const {
    if (stream) {
        return stream->is_idle();
    }
    // 无流时默认已就绪（CPU 同步执行）
    return true;
}

Tensor Tensor::contiguous_copy() const {
    if (contiguous_) {
        return *this;
    }

    check_cpu();

    // 使用 strides 复制非连续 tensor 为连续存储
    int64_t total_size = compute_size();
    auto new_storage = CpuStorage::allocate(total_size * sizeof(float));
    float* new_data = static_cast<float*>(new_storage->data());
    const float* src = static_cast<const float*>(storage_->data());

    // 递归遍历所有索引，使用 strides 计算正确的源地址
    std::function<void(int64_t, std::vector<int64_t>&)> copy_recursive;
    copy_recursive = [&](int64_t dim, std::vector<int64_t>& indices) {
        if (dim == static_cast<int64_t>(shape_.size())) {
            int64_t src_offset = offset_;
            int64_t dst_offset = 0;
            int64_t stride = 1;
            for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i) {
                src_offset += indices[i] * strides_[i];
                dst_offset += indices[i] * stride;
                stride *= shape_[i];
            }
            new_data[dst_offset] = src[src_offset];
            return;
        }

        for (int64_t i = 0; i < shape_[dim]; ++i) {
            indices[dim] = i;
            copy_recursive(dim + 1, indices);
        }
    };

    std::vector<int64_t> indices(shape_.size(), 0);
    copy_recursive(0, indices);

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

// ========== 零拷贝视图 ==========

LuaIntf::TensorView<float> Tensor::view() {
    check_cpu();
    if (!contiguous_) {
        Tensor cont = contiguous_copy();
        return LuaIntf::TensorView<float>(
            static_cast<float*>(cont.storage_->data()),
            cont.compute_size(),
            cont.storage_
        );
    }
    return LuaIntf::TensorView<float>(
        static_cast<float*>(storage_->data()) + offset_,
        compute_size(),
        storage_
    );
}

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

// ========== Level 2: 数学运算 ==========

Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }

    check_cpu();
    other.check_cpu();

    Tensor a = contiguous();
    Tensor b = other.contiguous();

    std::vector<float> result_data(compute_size());
    const float* data1 = static_cast<const float*>(a.storage_->data()) + a.offset_;
    const float* data2 = static_cast<const float*>(b.storage_->data()) + b.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = data1[i] + data2[i];
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::add(float scalar) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = src[i] + scalar;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::sub(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }

    check_cpu();
    other.check_cpu();

    Tensor a = contiguous();
    Tensor b = other.contiguous();

    std::vector<float> result_data(compute_size());
    const float* data1 = static_cast<const float*>(a.storage_->data()) + a.offset_;
    const float* data2 = static_cast<const float*>(b.storage_->data()) + b.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = data1[i] - data2[i];
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::sub(float scalar) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = src[i] - scalar;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::mul(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }

    check_cpu();
    other.check_cpu();

    Tensor a = contiguous();
    Tensor b = other.contiguous();

    std::vector<float> result_data(compute_size());
    const float* data1 = static_cast<const float*>(a.storage_->data()) + a.offset_;
    const float* data2 = static_cast<const float*>(b.storage_->data()) + b.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = data1[i] * data2[i];
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::mul(float scalar) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = src[i] * scalar;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::div(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }

    check_cpu();
    other.check_cpu();

    Tensor a = contiguous();
    Tensor b = other.contiguous();

    std::vector<float> result_data(compute_size());
    const float* data1 = static_cast<const float*>(a.storage_->data()) + a.offset_;
    const float* data2 = static_cast<const float*>(b.storage_->data()) + b.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        if (std::abs(data2[i]) < 1e-7f) {
            throw std::runtime_error("Division by zero");
        }
        result_data[i] = data1[i] / data2[i];
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::div(float scalar) const {
    if (std::abs(scalar) < 1e-7f) {
        throw std::runtime_error("Division by zero");
    }

    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
    float inv_scalar = 1.0f / scalar;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = src[i] * inv_scalar;
    }

    return Tensor(std::move(result_data), shape_);
}

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

// ========== 比较操作 ==========

Tensor Tensor::gt(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] > threshold) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::lt(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] < threshold) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::ge(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] >= threshold) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::le(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] <= threshold) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::eq(float threshold) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<float> result_data(compute_size());
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (std::abs(src[i] - threshold) < 1e-6f) ? 1.0f : 0.0f;
    }

    return Tensor(std::move(result_data), shape_);
}

// ========== 辅助方法 ==========

float Tensor::get_item(const std::vector<int64_t>& indices) const {
    return at(indices);
}

void Tensor::set_item(const std::vector<int64_t>& indices, float value) {
    check_cpu();
    int64_t offset = compute_offset(indices);
    float* ptr = static_cast<float*>(storage_->data());
    ptr[offset] = value;
}

std::string Tensor::to_string(int max_elements) const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i < shape_.size() - 1) oss << ", ";
    }
    oss << "]";

    if (storage_->device() == DeviceType::CPU) {
        oss << ", data=[";
        Tensor cont = contiguous();
        const float* ptr = static_cast<const float*>(cont.storage_->data()) + cont.offset_;
        int64_t total = compute_size();
        int64_t show = std::min(static_cast<int64_t>(max_elements), total);

        for (int64_t i = 0; i < show; ++i) {
            oss << ptr[i];
            if (i < show - 1) oss << ", ";
        }
        if (total > show) {
            oss << ", ...";
        }
        oss << "]";
    }

    oss << ", device=" << device_type_to_string(storage_->device()) << ")";
    return oss.str();
}

// ========== Reduction 操作 ==========

Tensor Tensor::sum(int axis, bool keepdims) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        float total = 0.0f;
        const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
        for (int64_t i = 0; i < compute_size(); ++i) {
            total += src[i];
        }

        if (keepdims) {
            std::vector<int64_t> new_shape(shape_.size(), 1);
            return Tensor(std::vector<float>{total}, new_shape);
        } else {
            return Tensor(std::vector<float>{total}, {1});
        }
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        } else if (keepdims) {
            new_shape.push_back(1);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    std::vector<float> result_data(outer_size * inner_size, 0.0f);
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t j = 0; j < axis_size; ++j) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t src_idx = (i * axis_size + j) * inner_size + k;
                int64_t dst_idx = i * inner_size + k;
                result_data[dst_idx] += src[src_idx];
            }
        }
    }

    return Tensor(std::move(result_data), new_shape);
}

Tensor Tensor::mean(int axis, bool keepdims) const {
    Tensor sum_result = sum(axis, keepdims);

    int64_t count;
    if (axis == -1) {
        count = compute_size();
    } else {
        int ax = axis;
        if (ax < 0) ax += shape_.size();
        count = shape_[ax];
    }

    return sum_result.div(static_cast<float>(count));
}

Tensor Tensor::max(int axis, bool keepdims) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
        float max_val = src[0];
        for (int64_t i = 1; i < compute_size(); ++i) {
            max_val = std::max(max_val, src[i]);
        }

        if (keepdims) {
            std::vector<int64_t> new_shape(shape_.size(), 1);
            return Tensor(std::vector<float>{max_val}, new_shape);
        } else {
            return Tensor(std::vector<float>{max_val}, {1});
        }
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        } else if (keepdims) {
            new_shape.push_back(1);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    std::vector<float> result_data(outer_size * inner_size);
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;
            float max_val = src[src_idx];

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                max_val = std::max(max_val, src[src_idx]);
            }

            int64_t dst_idx = i * inner_size + k;
            result_data[dst_idx] = max_val;
        }
    }

    return Tensor(std::move(result_data), new_shape);
}

Tensor Tensor::min(int axis, bool keepdims) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
        float min_val = src[0];
        for (int64_t i = 1; i < compute_size(); ++i) {
            min_val = std::min(min_val, src[i]);
        }

        if (keepdims) {
            std::vector<int64_t> new_shape(shape_.size(), 1);
            return Tensor(std::vector<float>{min_val}, new_shape);
        } else {
            return Tensor(std::vector<float>{min_val}, {1});
        }
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        } else if (keepdims) {
            new_shape.push_back(1);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    std::vector<float> result_data(outer_size * inner_size);
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;
            float min_val = src[src_idx];

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                min_val = std::min(min_val, src[src_idx]);
            }

            int64_t dst_idx = i * inner_size + k;
            result_data[dst_idx] = min_val;
        }
    }

    return Tensor(std::move(result_data), new_shape);
}

// ========== 向量化过滤操作 ==========

std::vector<int64_t> Tensor::nonzero() const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<int64_t> indices;
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
    int64_t total = compute_size();

    indices.reserve(total / 10);

    for (int64_t i = 0; i < total; ++i) {
        if (std::abs(src[i]) > 1e-7f) {
            indices.push_back(i);
        }
    }

    return indices;
}

std::vector<int64_t> Tensor::where_indices(float threshold, const std::string& op) const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<int64_t> indices;
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
    int64_t total = compute_size();

    indices.reserve(total / 10);

    if (op == "ge" || op == ">=") {
        for (int64_t i = 0; i < total; ++i) {
            if (src[i] >= threshold) indices.push_back(i);
        }
    } else if (op == "gt" || op == ">") {
        for (int64_t i = 0; i < total; ++i) {
            if (src[i] > threshold) indices.push_back(i);
        }
    } else if (op == "le" || op == "<=") {
        for (int64_t i = 0; i < total; ++i) {
            if (src[i] <= threshold) indices.push_back(i);
        }
    } else if (op == "lt" || op == "<") {
        for (int64_t i = 0; i < total; ++i) {
            if (src[i] < threshold) indices.push_back(i);
        }
    } else if (op == "eq" || op == "==") {
        for (int64_t i = 0; i < total; ++i) {
            if (std::abs(src[i] - threshold) < 1e-6f) indices.push_back(i);
        }
    } else {
        throw std::runtime_error("Invalid operator: " + op);
    }

    return indices;
}

Tensor Tensor::index_select(int dim, const std::vector<int64_t>& indices) const {
    if (indices.empty()) {
        throw std::runtime_error("Empty indices for index_select");
    }

    check_cpu();
    Tensor a = contiguous();

    int ax = dim;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Dimension out of range");
    }

    std::vector<int64_t> new_shape = shape_;
    new_shape[ax] = indices.size();

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }

    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    int64_t new_size = outer_size * indices.size() * inner_size;
    std::vector<float> new_data(new_size);
    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < indices.size(); ++j) {
            int64_t idx = indices[j];
            if (idx < 0) idx += shape_[ax];
            if (idx < 0 || idx >= shape_[ax]) {
                throw std::runtime_error("Index out of range in index_select");
            }

            const float* src_ptr = src + (i * shape_[ax] + idx) * inner_size;
            float* dst_ptr = new_data.data() + (i * indices.size() + j) * inner_size;
            std::copy(src_ptr, src_ptr + inner_size, dst_ptr);
        }
    }

    return Tensor(std::move(new_data), new_shape);
}

LuaIntf::LuaRef Tensor::to_table(lua_State* L) const {
    check_cpu();

    if (shape_.size() == 1) {
        LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
        const float* base = data();

        for (int64_t i = 0; i < shape_[0]; ++i) {
            result[i + 1] = base[i * strides_[0]];
        }
        return result;
    } else if (shape_.size() == 2) {
        LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
        const float* base = data();

        for (int64_t i = 0; i < shape_[0]; ++i) {
            LuaIntf::LuaRef row = LuaIntf::LuaRef::createTable(L);
            for (int64_t j = 0; j < shape_[1]; ++j) {
                row[j + 1] = base[i * strides_[0] + j * strides_[1]];
            }
            result[i + 1] = row;
        }
        return result;
    } else {
        throw std::runtime_error("to_table only supports 1D and 2D tensors");
    }
}

// ========== Argmax/Argmin ==========

LuaIntf::LuaRef Tensor::argmax_lua(lua_State* L, int axis) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
        int64_t max_idx = 0;
        float max_val = src[0];

        for (int64_t i = 1; i < compute_size(); ++i) {
            if (src[i] > max_val) {
                max_val = src[i];
                max_idx = i;
            }
        }

        return LuaIntf::LuaRef::fromValue(L, max_idx);
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    int lua_idx = 1;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;
            float max_val = src[src_idx];
            int64_t max_pos = 0;

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                if (src[src_idx] > max_val) {
                    max_val = src[src_idx];
                    max_pos = j;
                }
            }

            result[lua_idx++] = max_pos;
        }
    }

    return result;
}

LuaIntf::LuaRef Tensor::argmin_lua(lua_State* L, int axis) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis == -1) {
        const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
        int64_t min_idx = 0;
        float min_val = src[0];

        for (int64_t i = 1; i < compute_size(); ++i) {
            if (src[i] < min_val) {
                min_val = src[i];
                min_idx = i;
            }
        }

        return LuaIntf::LuaRef::fromValue(L, min_idx);
    }

    int ax = axis;
    if (ax < 0) ax += shape_.size();

    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }

    const float* src = static_cast<const float*>(a.storage_->data()) + a.offset_;
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    int lua_idx = 1;

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            int64_t src_idx = (i * axis_size + 0) * inner_size + k;
            float min_val = src[src_idx];
            int64_t min_pos = 0;

            for (int64_t j = 1; j < axis_size; ++j) {
                src_idx = (i * axis_size + j) * inner_size + k;
                if (src[src_idx] < min_val) {
                    min_val = src[src_idx];
                    min_pos = j;
                }
            }

            result[lua_idx++] = min_pos;
        }
    }

    return result;
}

// ========== TopK (Optimized) ==========

LuaIntf::LuaRef Tensor::topk_lua(lua_State* L, int k, int axis, bool largest) const {
    check_cpu();
    Tensor a = contiguous();

    if (axis != -1 && axis != static_cast<int>(shape_.size()) - 1) {
        throw std::runtime_error("topk only supports last axis or axis=-1");
    }

    if (shape_.empty()) {
        throw std::runtime_error("Cannot apply topk to scalar");
    }

    int64_t inner_size = shape_.back();
    int64_t outer_size = compute_size() / inner_size;
    k = std::min(k, static_cast<int>(inner_size));

    const float* src = a.raw_data();

    // Pre-allocate Lua tables with exact size
    lua_createtable(L, static_cast<int>(outer_size * k), 0);
    LuaIntf::LuaRef values = LuaIntf::LuaRef::popFromStack(L);
    lua_createtable(L, static_cast<int>(outer_size * k), 0);
    LuaIntf::LuaRef indices = LuaIntf::LuaRef::popFromStack(L);

    // Reuse index buffer across rows to avoid repeated allocation
    std::vector<int32_t> idx_buf(inner_size);

    int lua_idx = 1;
    for (int64_t i = 0; i < outer_size; ++i) {
        const float* row = src + i * inner_size;

        // Initialize indices
        for (int64_t j = 0; j < inner_size; ++j) {
            idx_buf[j] = static_cast<int32_t>(j);
        }

        // Use nth_element for O(n) partitioning, then sort top-k for O(k log k)
        // Total: O(n + k log k) which is optimal for small k
        if (largest) {
            std::nth_element(idx_buf.begin(), idx_buf.begin() + k, idx_buf.end(),
                [row](int32_t a, int32_t b) { return row[a] > row[b]; });
            std::sort(idx_buf.begin(), idx_buf.begin() + k,
                [row](int32_t a, int32_t b) { return row[a] > row[b]; });
        } else {
            std::nth_element(idx_buf.begin(), idx_buf.begin() + k, idx_buf.end(),
                [row](int32_t a, int32_t b) { return row[a] < row[b]; });
            std::sort(idx_buf.begin(), idx_buf.begin() + k,
                [row](int32_t a, int32_t b) { return row[a] < row[b]; });
        }

        // Write results directly using raw Lua API for efficiency
        for (int j = 0; j < k; ++j, ++lua_idx) {
            values[lua_idx] = row[idx_buf[j]];
            indices[lua_idx] = static_cast<int64_t>(idx_buf[j]);
        }
    }

    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    result["values"] = values;
    result["indices"] = indices;
    return result;
}

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

// ========== Legacy 方法 ==========

LuaIntf::LuaRef Tensor::filter_yolo(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO output shape");
    }

    check_cpu();
    Tensor a = contiguous();
    const float* data_ptr = static_cast<const float*>(a.storage_->data()) + a.offset_;

    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];

    // Heuristic for [1, 84, 8400] (YOLOv8/11) vs [1, 25200, 85] (YOLOv5)
    bool transposed = (dim1 < dim2 && dim2 > 100);

    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;

    bool has_objectness = (box_dim == 85);
    int num_classes = has_objectness ? 80 : (static_cast<int>(box_dim) - 4);

    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h, objectness;

        if (transposed) {
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
            objectness = has_objectness ? data_ptr[4 * num_boxes + i] : 1.0f;
        } else {
            const float* box_data = data_ptr + i * box_dim;
            cx = box_data[0];
            cy = box_data[1];
            w  = box_data[2];
            h  = box_data[3];
            objectness = has_objectness ? box_data[4] : 1.0f;
        }

        int best_class_id = 0;
        float best_class_score = -1.0f;

        int class_start = has_objectness ? 5 : 4;

        if (transposed) {
             if (has_objectness && objectness < conf_thres) continue;
             best_class_score = data_ptr[(class_start + 0) * num_boxes + i];
             best_class_id = 0;
             for (int c = 1; c < num_classes; ++c) {
                 float score = data_ptr[(class_start + c) * num_boxes + i];
                 if (score > best_class_score) {
                     best_class_score = score;
                     best_class_id = c;
                 }
             }
        } else {
             const float* box_data = data_ptr + i * box_dim;
             const float* class_scores = box_data + class_start;
             if (has_objectness && objectness < conf_thres) continue;
             best_class_score = class_scores[0];
             for (int c = 1; c < num_classes; ++c) {
                 if (class_scores[c] > best_class_score) {
                     best_class_score = class_scores[c];
                     best_class_id = c;
                 }
             }
        }

        float final_score = objectness * best_class_score;
        if (final_score < conf_thres) continue;

        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        LuaIntf::LuaRef box = LuaIntf::LuaRef::createTable(L);
        box["x"] = x;
        box["y"] = y;
        box["w"] = w;
        box["h"] = h;
        box["score"] = final_score;
        box["cls"] = best_class_id;

        results[result_idx++] = box;
    }

    return results;
}

LuaIntf::LuaRef Tensor::filter_yolo_pose(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO Pose output shape");
    }

    check_cpu();
    Tensor a = contiguous();
    const float* data_ptr = static_cast<const float*>(a.storage_->data()) + a.offset_;

    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];

    bool transposed = (dim1 < dim2 && dim2 > 100);

    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;

    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h, score;

        if (transposed) {
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
            score = data_ptr[4 * num_boxes + i];
        } else {
            const float* box_data = data_ptr + i * box_dim;
            cx = box_data[0];
            cy = box_data[1];
            w  = box_data[2];
            h  = box_data[3];
            score = box_data[4];
        }

        if (score < conf_thres) continue;

        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        LuaIntf::LuaRef box = LuaIntf::LuaRef::createTable(L);
        box["x"] = x;
        box["y"] = y;
        box["w"] = w;
        box["h"] = h;
        box["score"] = score;
        box["class_id"] = 0;

        // Extract Keypoints
        LuaIntf::LuaRef kpts = LuaIntf::LuaRef::createTable(L);
        for (int k = 0; k < 17; ++k) {
            float kx, ky, kv;
            if (transposed) {
                kx = data_ptr[(5 + k * 3 + 0) * num_boxes + i];
                ky = data_ptr[(5 + k * 3 + 1) * num_boxes + i];
                kv = data_ptr[(5 + k * 3 + 2) * num_boxes + i];
            } else {
                const float* box_data = data_ptr + i * box_dim;
                kx = box_data[5 + k * 3 + 0];
                ky = box_data[5 + k * 3 + 1];
                kv = box_data[5 + k * 3 + 2];
            }

            LuaIntf::LuaRef kp = LuaIntf::LuaRef::createTable(L);
            kp["x"] = kx;
            kp["y"] = ky;
            kp["v"] = kv;
            kpts[k + 1] = kp;
        }
        box["keypoints"] = kpts;

        results[result_idx++] = box;
    }

    return results;
}

LuaIntf::LuaRef Tensor::filter_yolo_seg(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO Seg output shape");
    }

    check_cpu();
    Tensor a = contiguous();
    const float* data_ptr = static_cast<const float*>(a.storage_->data()) + a.offset_;

    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];

    bool transposed = (dim1 < dim2 && dim2 > 100);

    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;

    int num_classes = 80;
    int num_masks = 32;

    if (box_dim != (4 + num_classes + num_masks)) {
        num_masks = 32;
        num_classes = static_cast<int>(box_dim) - 4 - num_masks;
    }

    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h;

        if (transposed) {
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
        } else {
            const float* box_data = data_ptr + i * box_dim;
            cx = box_data[0];
            cy = box_data[1];
            w  = box_data[2];
            h  = box_data[3];
        }

        int best_class_id = 0;
        float best_class_score = -1.0f;

        int class_start = 4;

        if (transposed) {
             best_class_score = data_ptr[(class_start + 0) * num_boxes + i];
             best_class_id = 0;
             for (int c = 1; c < num_classes; ++c) {
                 float score = data_ptr[(class_start + c) * num_boxes + i];
                 if (score > best_class_score) {
                     best_class_score = score;
                     best_class_id = c;
                 }
             }
        } else {
             const float* box_data = data_ptr + i * box_dim;
             const float* class_scores = box_data + class_start;
             best_class_score = class_scores[0];
             for (int c = 1; c < num_classes; ++c) {
                 if (class_scores[c] > best_class_score) {
                     best_class_score = class_scores[c];
                     best_class_id = c;
                 }
             }
        }

        if (best_class_score < conf_thres) continue;

        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        LuaIntf::LuaRef box = LuaIntf::LuaRef::createTable(L);
        box["x"] = x;
        box["y"] = y;
        box["w"] = w;
        box["h"] = h;
        box["score"] = best_class_score;
        box["class_id"] = best_class_id;

        // Extract Mask Coefficients
        LuaIntf::LuaRef mask_coeffs_out = LuaIntf::LuaRef::createTable(L);
        int mask_start = 4 + num_classes;

        for (int m = 0; m < num_masks; ++m) {
            float val;
            if (transposed) {
                val = data_ptr[(mask_start + m) * num_boxes + i];
            } else {
                val = data_ptr[i * box_dim + mask_start + m];
            }
            mask_coeffs_out[m + 1] = val;
        }
        box["mask_coeffs"] = mask_coeffs_out;

        results[result_idx++] = box;
    }

    return results;
}

LuaIntf::LuaRef Tensor::process_mask(lua_State* L, const LuaIntf::LuaRef& mask_coeffs,
                                      const LuaIntf::LuaRef& box,
                                      int img_w, int img_h,
                                      int input_w, int input_h,
                                      int pad_x, int pad_y) {
    check_cpu();
    Tensor proto = contiguous();

    auto proto_shape = proto.shape();
    if (proto_shape.size() != 4 || proto_shape[1] != 32) {
        throw std::runtime_error("Invalid proto mask shape");
    }

    int mh = static_cast<int>(proto_shape[2]);
    int mw = static_cast<int>(proto_shape[3]);
    int num_masks = 32;

    const float* proto_data = static_cast<const float*>(proto.storage_->data()) + proto.offset_;

    // 1. Matrix Multiplication: Mask = Coeffs * Proto
    cv::Mat proto_mat(num_masks, mh * mw, CV_32F, const_cast<float*>(proto_data));
    cv::Mat coeffs_mat(1, num_masks, CV_32F);

    for (int i = 0; i < num_masks; ++i) {
        coeffs_mat.at<float>(0, i) = mask_coeffs.get<float>(i + 1);
    }

    cv::Mat mask_flat = coeffs_mat * proto_mat;
    cv::Mat mask = mask_flat.reshape(1, mh);

    // 2. Sigmoid
    cv::exp(-mask, mask);
    mask = 1.0f / (1.0f + mask);

    // 3. Resize to Input Size
    cv::Mat mask_input;
    cv::resize(mask, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

    // 4. Crop Padding
    int roi_w = input_w - 2 * pad_x;
    int roi_h = input_h - 2 * pad_y;

    cv::Rect roi(pad_x, pad_y, roi_w, roi_h);
    roi = roi & cv::Rect(0, 0, input_w, input_h);

    if (roi.area() == 0) {
         return LuaIntf::LuaRef::fromValue(L, Tensor(std::vector<float>(img_w * img_h, 0), {1, static_cast<int64_t>(img_h), static_cast<int64_t>(img_w)}));
    }

    cv::Mat mask_cropped = mask_input(roi);

    // 5. Resize to Original Image Size
    cv::Mat mask_original;
    cv::resize(mask_cropped, mask_original, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

    // 6. Crop by Box
    float bx = box.get<float>("x");
    float by = box.get<float>("y");
    float bw = box.get<float>("w");
    float bh = box.get<float>("h");

    cv::Rect box_rect(static_cast<int>(bx), static_cast<int>(by),
                      static_cast<int>(bw), static_cast<int>(bh));
    box_rect = box_rect & cv::Rect(0, 0, img_w, img_h);

    cv::Mat final_mask = cv::Mat::zeros(img_h, img_w, CV_32F);
    if (box_rect.area() > 0) {
        mask_original(box_rect).copyTo(final_mask(box_rect));
    }

    // 7. Threshold
    cv::threshold(final_mask, final_mask, 0.5, 1.0, cv::THRESH_BINARY);

    // Return as Tensor
    std::vector<float> mask_data(img_w * img_h);
    std::memcpy(mask_data.data(), final_mask.data, img_w * img_h * sizeof(float));

    return LuaIntf::LuaRef::fromValue(L, Tensor(std::move(mask_data), {1, static_cast<int64_t>(img_h), static_cast<int64_t>(img_w)}));
}

LuaIntf::LuaRef Tensor::argmax(lua_State* L) {
    if (shape_.size() != 2 || shape_[0] != 1) {
        throw std::runtime_error("Invalid classification output shape");
    }

    check_cpu();
    Tensor a = contiguous();

    int num_classes = static_cast<int>(shape_[1]);
    int max_idx = 0;
    const float* ptr = static_cast<const float*>(a.storage_->data()) + a.offset_;
    float max_val = ptr[0];

    for (int i = 1; i < num_classes; ++i) {
        if (ptr[i] > max_val) {
            max_val = ptr[i];
            max_idx = i;
        }
    }

    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    result["class_id"] = max_idx;
    result["confidence"] = max_val;
    return result;
}

LuaIntf::LuaRef Tensor::topk(lua_State* L, int k) {
    if (shape_.size() != 2 || shape_[0] != 1) {
        throw std::runtime_error("Invalid classification output shape");
    }

    check_cpu();
    Tensor a = contiguous();

    int num_classes = static_cast<int>(shape_[1]);
    if (k > num_classes) k = num_classes;

    const float* ptr = static_cast<const float*>(a.storage_->data()) + a.offset_;

    std::vector<std::pair<float, int>> scores(num_classes);
    for (int i = 0; i < num_classes; ++i) {
        scores[i] = {ptr[i], i};
    }

    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });

    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    for (int i = 0; i < k; ++i) {
        LuaIntf::LuaRef item = LuaIntf::LuaRef::createTable(L);
        item["class_id"] = scores[i].second;
        item["confidence"] = scores[i].first;
        result[i + 1] = item;
    }
    return result;
}

} // namespace tensor
