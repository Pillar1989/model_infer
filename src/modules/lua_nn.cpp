#include "lua_nn.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <opencv2/opencv.hpp>

// 启用vector的Lua table自动转换
namespace LuaIntf {
    LUA_USING_LIST_TYPE(std::vector)
}

namespace lua_nn {

// ========== Tensor构造函数 ==========
Tensor::Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape)
    : data_(std::make_shared<std::vector<float>>(data))
    , shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true) {}

Tensor::Tensor(std::vector<float>&& data, const std::vector<int64_t>& shape)
    : data_(std::make_shared<std::vector<float>>(std::move(data)))
    , shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true) {}

Tensor::Tensor(const float* data, const std::vector<int64_t>& shape, std::shared_ptr<std::vector<float>> owner)
    : shape_(shape)
    , strides_(compute_strides(shape))
    , offset_(0)
    , contiguous_(true) {
    if (owner) {
        data_ = owner;
    } else {
        int64_t total_size = compute_size();
        data_ = std::make_shared<std::vector<float>>(data, data + total_size);
    }
}

Tensor::Tensor(std::shared_ptr<std::vector<float>> data,
               const std::vector<int64_t>& shape,
               const std::vector<int64_t>& strides,
               int64_t offset,
               bool contiguous)
    : data_(data)
    , shape_(shape)
    , strides_(strides)
    , offset_(offset)
    , contiguous_(contiguous) {}

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

Tensor Tensor::contiguous_copy() const {
    if (contiguous_) {
        return *this;
    }
    
    std::vector<float> new_data(compute_size());
    // TODO: 实现非连续tensor的连续拷贝
    // 这里需要递归遍历所有维度
    throw std::runtime_error("contiguous_copy not yet implemented for non-contiguous tensors");
}

// ========== Level 1: 基础形状操作 ==========
Tensor Tensor::slice(int dim, int64_t start, int64_t end, int64_t step) const {
    if (dim < 0) dim += shape_.size();
    if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Dimension out of range");
    }
    
    // 处理负索引
    if (start < 0) start += shape_[dim];
    if (end < 0) end += shape_[dim];
    
    // 边界检查
    start = std::max(int64_t(0), std::min(start, shape_[dim]));
    end = std::max(int64_t(0), std::min(end, shape_[dim]));
    
    if (start >= end || step <= 0) {
        throw std::runtime_error("Invalid slice parameters");
    }
    
    // 创建新的shape和strides
    std::vector<int64_t> new_shape = shape_;
    new_shape[dim] = (end - start + step - 1) / step;
    
    std::vector<int64_t> new_strides = strides_;
    new_strides[dim] = strides_[dim] * step;
    
    int64_t new_offset = offset_ + start * strides_[dim];
    
    // 判断是否仍然连续（只有最后一维且step=1才连续）
    bool new_contiguous = contiguous_ && (dim == shape_.size() - 1) && (step == 1);
    
    return Tensor(data_, new_shape, new_strides, new_offset, new_contiguous);
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    // 计算新的总大小
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
    
    // 推断-1维度
    std::vector<int64_t> final_shape = new_shape;
    if (infer_dim != -1) {
        int64_t current_size = compute_size();
        if (current_size % new_size != 0) {
            throw std::runtime_error("Cannot infer dimension size");
        }
        final_shape[infer_dim] = current_size / new_size;
        new_size = current_size;
    }
    
    // 检查大小匹配
    if (new_size != compute_size()) {
        throw std::runtime_error("Shape size mismatch");
    }
    
    // Reshape要求tensor是连续的
    if (!contiguous_) {
        return contiguous_copy().reshape(final_shape);
    }
    
    // 零拷贝reshape
    return Tensor(data_, final_shape, compute_strides(final_shape), offset_, true);
}

Tensor Tensor::transpose(const std::vector<int>& dims) const {
    if (dims.size() != shape_.size()) {
        throw std::runtime_error("Transpose dimensions mismatch");
    }
    
    // 检查dims是否是有效的排列
    std::vector<bool> used(dims.size(), false);
    for (int dim : dims) {
        int d = dim;
        if (d < 0) d += dims.size();
        if (d < 0 || d >= static_cast<int>(dims.size()) || used[d]) {
            throw std::runtime_error("Invalid transpose dimensions");
        }
        used[d] = true;
    }
    
    // 创建新的shape和strides
    std::vector<int64_t> new_shape(shape_.size());
    std::vector<int64_t> new_strides(strides_.size());
    
    for (size_t i = 0; i < dims.size(); ++i) {
        int dim = dims[i];
        if (dim < 0) dim += dims.size();
        new_shape[i] = shape_[dim];
        new_strides[i] = strides_[dim];
    }
    
    // Transpose通常会破坏连续性
    return Tensor(data_, new_shape, new_strides, offset_, false);
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
        // 移除所有大小为1的维度
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] != 1) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        }
    } else {
        // 移除指定维度
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
    
    return Tensor(data_, new_shape, new_strides, offset_, contiguous_);
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
    // 新维度的stride可以是任意值（因为大小为1），我们使用相邻维度的stride
    int64_t new_stride = (dim < ndim) ? strides_[dim] : 1;
    new_strides.insert(new_strides.begin() + dim, new_stride);
    
    return Tensor(data_, new_shape, new_strides, offset_, contiguous_);
}

// ========== Level 2: 数学运算 ==========
// Element-wise加法
Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }
    
    std::vector<float> result_data(compute_size());
    const float* data1 = data();
    const float* data2 = other.data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = data1[i] + data2[i];
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::add(float scalar) const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = src[i] + scalar;
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::sub(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }
    
    std::vector<float> result_data(compute_size());
    const float* data1 = data();
    const float* data2 = other.data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = data1[i] - data2[i];
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::sub(float scalar) const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = src[i] - scalar;
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::mul(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }
    
    std::vector<float> result_data(compute_size());
    const float* data1 = data();
    const float* data2 = other.data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = data1[i] * data2[i];
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::mul(float scalar) const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = src[i] * scalar;
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::div(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch for element-wise operation");
    }
    
    std::vector<float> result_data(compute_size());
    const float* data1 = data();
    const float* data2 = other.data();
    
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
    
    std::vector<float> result_data(compute_size());
    const float* src = data();
    float inv_scalar = 1.0f / scalar;
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = src[i] * inv_scalar;
    }
    
    return Tensor(std::move(result_data), shape_);
}

// Activation函数
Tensor Tensor::sigmoid() const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::softmax(int axis) const {
    int ax = axis;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Axis out of range");
    }
    
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    // 简化实现：仅支持最后一维的softmax
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
        
        // 找到最大值（数值稳定性）
        float max_val = row[0];
        for (int64_t j = 1; j < inner_size; ++j) {
            max_val = std::max(max_val, row[j]);
        }
        
        // exp和sum
        float sum = 0.0f;
        for (int64_t j = 0; j < inner_size; ++j) {
            out_row[j] = std::exp(row[j] - max_val);
            sum += out_row[j];
        }
        
        // 归一化
        float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < inner_size; ++j) {
            out_row[j] *= inv_sum;
        }
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::exp_() const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = std::exp(src[i]);
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::log_() const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        if (src[i] <= 0.0f) {
            throw std::runtime_error("Log of non-positive value");
        }
        result_data[i] = std::log(src[i]);
    }
    
    return Tensor(std::move(result_data), shape_);
}

// 比较操作
Tensor Tensor::gt(float threshold) const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] > threshold) ? 1.0f : 0.0f;
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::lt(float threshold) const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] < threshold) ? 1.0f : 0.0f;
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::ge(float threshold) const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] >= threshold) ? 1.0f : 0.0f;
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::le(float threshold) const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (src[i] <= threshold) ? 1.0f : 0.0f;
    }
    
    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::eq(float threshold) const {
    std::vector<float> result_data(compute_size());
    const float* src = data();
    
    for (int64_t i = 0; i < compute_size(); ++i) {
        result_data[i] = (std::abs(src[i] - threshold) < 1e-6f) ? 1.0f : 0.0f;
    }
    
    return Tensor(std::move(result_data), shape_);
}

// Reduction操作
Tensor Tensor::sum(int axis, bool keepdims) const {
    if (axis == -1) {
        // 对所有元素求和
        float total = 0.0f;
        const float* src = data();
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
    
    // 计算新的shape
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        } else if (keepdims) {
            new_shape.push_back(1);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);
    
    // 计算输出大小
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
    const float* src = data();
    
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
    if (axis == -1) {
        // 对所有元素求max
        const float* src = data();
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
    
    // 计算新的shape
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        if (i != ax) {
            new_shape.push_back(shape_[i]);
        } else if (keepdims) {
            new_shape.push_back(1);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);
    
    // 计算尺寸
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
    const float* src = data();
    
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
    if (axis == -1) {
        const float* src = data();
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
    const float* src = data();
    
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

// Argmax/Argmin返回Lua表
LuaIntf::LuaRef Tensor::argmax_lua(lua_State* L, int axis) const {
    if (axis == -1) {
        // 返回单个最大值的索引
        const float* src = data();
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
    
    // 计算尺寸
    int64_t outer_size = 1;
    for (int i = 0; i < ax; ++i) {
        outer_size *= shape_[i];
    }
    int64_t axis_size = shape_[ax];
    int64_t inner_size = 1;
    for (int i = ax + 1; i < static_cast<int>(shape_.size()); ++i) {
        inner_size *= shape_[i];
    }
    
    const float* src = data();
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
    if (axis == -1) {
        const float* src = data();
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
    
    const float* src = data();
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

// ========== Level 3: 高级操作 ==========
LuaIntf::LuaRef Tensor::topk_lua(lua_State* L, int k, int axis, bool largest) const {
    // 简化实现：仅支持最后一维和axis=-1
    if (axis != -1 && axis != static_cast<int>(shape_.size()) - 1) {
        throw std::runtime_error("topk only supports last axis or axis=-1");
    }
    
    if (shape_.empty()) {
        throw std::runtime_error("Cannot apply topk to scalar");
    }
    
    int64_t inner_size = shape_[shape_.size() - 1];
    int64_t outer_size = compute_size() / inner_size;
    
    k = std::min(k, static_cast<int>(inner_size));
    
    const float* src = data();
    
    LuaIntf::LuaRef values = LuaIntf::LuaRef::createTable(L);
    LuaIntf::LuaRef indices = LuaIntf::LuaRef::createTable(L);
    
    int lua_idx = 1;
    for (int64_t i = 0; i < outer_size; ++i) {
        const float* row = src + i * inner_size;
        
        // 创建索引值对
        std::vector<std::pair<float, int64_t>> pairs(inner_size);
        for (int64_t j = 0; j < inner_size; ++j) {
            pairs[j] = {row[j], j};
        }
        
        // 排序
        if (largest) {
            std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
        } else {
            std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }
        
        // 提取前k个
        for (int j = 0; j < k; ++j) {
            values[lua_idx] = pairs[j].first;
            indices[lua_idx] = pairs[j].second;
            ++lua_idx;
        }
    }
    
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
    result["values"] = values;
    result["indices"] = indices;
    return result;
}

// 辅助方法
float Tensor::get_item(const std::vector<int64_t>& indices) const {
    int64_t offset = compute_offset(indices);
    return data_->at(offset);
}

void Tensor::set_item(const std::vector<int64_t>& indices, float value) {
    int64_t offset = compute_offset(indices);
    (*data_)[offset] = value;
}

std::string Tensor::to_string(int max_elements) const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], data=[";
    
    const float* src = data();
    int64_t total = compute_size();
    int64_t to_show = std::min(total, static_cast<int64_t>(max_elements));
    
    for (int64_t i = 0; i < to_show; ++i) {
        if (i > 0) oss << ", ";
        oss << src[i];
    }
    
    if (to_show < total) {
        oss << ", ...";
    }
    oss << "])";
    
    return oss.str();
}

LuaIntf::LuaRef Tensor::to_table(lua_State* L) const {
    // 简化实现：仅支持1D和2D tensor
    if (shape_.size() == 1) {
        LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
        const float* src = data();
        for (int64_t i = 0; i < shape_[0]; ++i) {
            result[i + 1] = src[i];
        }
        return result;
    } else if (shape_.size() == 2) {
        LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);
        const float* src = data();
        for (int64_t i = 0; i < shape_[0]; ++i) {
            LuaIntf::LuaRef row = LuaIntf::LuaRef::createTable(L);
            for (int64_t j = 0; j < shape_[1]; ++j) {
                row[j + 1] = src[i * shape_[1] + j];
            }
            result[i + 1] = row;
        }
        return result;
    } else {
        throw std::runtime_error("to_table only supports 1D and 2D tensors");
    }
}

// ========== 向量化过滤操作（方案3 - 通用API） ==========
// 返回非零元素的索引
std::vector<int64_t> Tensor::nonzero() const {
    std::vector<int64_t> indices;
    const float* src = data();
    int64_t total = compute_size();

    indices.reserve(total / 10);  // 预分配，假设10%非零

    for (int64_t i = 0; i < total; ++i) {
        if (std::abs(src[i]) > 1e-7f) {
            indices.push_back(i);
        }
    }

    return indices;
}

// 返回满足条件的索引（核心优化：避免创建bool tensor）
std::vector<int64_t> Tensor::where_indices(float threshold, const std::string& op) const {
    std::vector<int64_t> indices;
    const float* src = data();
    int64_t total = compute_size();

    indices.reserve(total / 10);  // 预分配

    // 根据操作符进行过滤
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

// 根据索引选择元素（批量gather）
Tensor Tensor::index_select(int dim, const std::vector<int64_t>& indices) const {
    if (indices.empty()) {
        throw std::runtime_error("Empty indices for index_select");
    }

    // 标准化维度
    int ax = dim;
    if (ax < 0) ax += shape_.size();
    if (ax < 0 || ax >= static_cast<int>(shape_.size())) {
        throw std::runtime_error("Dimension out of range");
    }

    // 计算新的shape
    std::vector<int64_t> new_shape = shape_;
    new_shape[ax] = indices.size();

    // 计算步长
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
    const float* src = data();

    // 批量复制数据
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

// 高效的多列提取（专为[C, N]格式优化，直接返回Lua table）
LuaIntf::LuaRef Tensor::extract_columns(lua_State* L, const std::vector<int64_t>& col_indices) const {
    if (shape_.size() != 2) {
        throw std::runtime_error("extract_columns only supports 2D tensors");
    }

    if (col_indices.empty()) {
        return LuaIntf::LuaRef::createTable(L);
    }

    int64_t num_rows = shape_[0];
    int64_t num_cols = shape_[1];
    const float* src = data();

    // 创建结果表：result[col_idx] = {row_values...}
    LuaIntf::LuaRef result = LuaIntf::LuaRef::createTable(L);

    for (size_t i = 0; i < col_indices.size(); ++i) {
        int64_t col = col_indices[i];
        if (col < 0) col += num_cols;
        if (col < 0 || col >= num_cols) {
            throw std::runtime_error("Column index out of range");
        }

        LuaIntf::LuaRef col_data = LuaIntf::LuaRef::createTable(L);
        for (int64_t row = 0; row < num_rows; ++row) {
            col_data[row + 1] = src[row * num_cols + col];
        }
        result[i + 1] = col_data;
    }

    return result;
}

// ========== Legacy方法（保留向后兼容） ==========
LuaIntf::LuaRef Tensor::filter_yolo(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO output shape");
    }
    
    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];
    
    // Heuristic for [1, 84, 8400] (YOLOv8/11) vs [1, 25200, 85] (YOLOv5)
    bool transposed = (dim1 < dim2 && dim2 > 100); 
    
    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;
    
    bool has_objectness = (box_dim == 85);
    int num_classes = has_objectness ? 80 : (box_dim - 4);
    
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;
    
    const float* data_ptr = data_->data();

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h, objectness;
        
        if (transposed) {
            // [1, C, N] -> stride is N (num_boxes)
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
            objectness = has_objectness ? data_ptr[4 * num_boxes + i] : 1.0f;
        } else {
            // [1, N, C] -> stride is C (box_dim)
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

             // Initialize with first class
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

LuaIntf::LuaRef Tensor::argmax(lua_State* L) {
    // 假设shape: [1, num_classes]
    if (shape_.size() != 2 || shape_[0] != 1) {
        throw std::runtime_error("Invalid classification output shape");
    }
    
    int num_classes = static_cast<int>(shape_[1]);
    int max_idx = 0;
    float max_val = (*data_)[0];
    
    for (int i = 1; i < num_classes; ++i) {
        if ((*data_)[i] > max_val) {
            max_val = (*data_)[i];
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
    
    int num_classes = static_cast<int>(shape_[1]);
    if (k > num_classes) k = num_classes;
    
    std::vector<std::pair<float, int>> scores(num_classes);
    for (int i = 0; i < num_classes; ++i) {
        scores[i] = {(*data_)[i], i};
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

LuaIntf::LuaRef Tensor::filter_yolo_pose(lua_State* L, float conf_thres) {
    if (shape_.size() != 3 || shape_[0] != 1) {
        throw std::runtime_error("Invalid YOLO Pose output shape");
    }
    
    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];
    
    // Heuristic for [1, 56, 8400]
    bool transposed = (dim1 < dim2 && dim2 > 100); 
    
    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;
    
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;
    
    const float* data_ptr = data_->data();

    for (int64_t i = 0; i < num_boxes; ++i) {
        float cx, cy, w, h, score;
        
        if (transposed) {
            // [1, C, N] -> stride is N (num_boxes)
            cx = data_ptr[0 * num_boxes + i];
            cy = data_ptr[1 * num_boxes + i];
            w  = data_ptr[2 * num_boxes + i];
            h  = data_ptr[3 * num_boxes + i];
            score = data_ptr[4 * num_boxes + i];
        } else {
            // [1, N, C] -> stride is C (box_dim)
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
        box["class_id"] = 0; // Pose usually has only 1 class (person)
        
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
    
    int64_t dim1 = shape_[1];
    int64_t dim2 = shape_[2];
    
    // Heuristic for [1, 116, 8400] (32 masks + 4 box + 80 classes)
    bool transposed = (dim1 < dim2 && dim2 > 100); 
    
    int64_t num_boxes = transposed ? dim2 : dim1;
    int64_t box_dim = transposed ? dim1 : dim2;
    
    // 4 box + 80 classes + 32 masks = 116
    int num_classes = 80; 
    int num_masks = 32;
    
    if (box_dim != (4 + num_classes + num_masks)) {
        // Try to infer
        num_masks = 32;
        num_classes = box_dim - 4 - num_masks;
    }
    
    LuaIntf::LuaRef results = LuaIntf::LuaRef::createTable(L);
    int result_idx = 1;
    
    const float* data_ptr = data_->data();

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
        LuaIntf::LuaRef mask_coeffs = LuaIntf::LuaRef::createTable(L);
        int mask_start = 4 + num_classes;
        
        for (int m = 0; m < num_masks; ++m) {
            float val;
            if (transposed) {
                val = data_ptr[(mask_start + m) * num_boxes + i];
            } else {
                val = data_ptr[i * box_dim + mask_start + m];
            }
            mask_coeffs[m + 1] = val;
        }
        box["mask_coeffs"] = mask_coeffs;
        
        results[result_idx++] = box;
    }
    
    return results;
}

LuaIntf::LuaRef Tensor::process_mask(lua_State* L, const LuaIntf::LuaRef& mask_coeffs, 
                                   const LuaIntf::LuaRef& box, 
                                   int img_w, int img_h,
                                   int input_w, int input_h,
                                   int pad_x, int pad_y) {
    const Tensor& proto = *this;
    // proto: [1, 32, 160, 160]
    // mask_coeffs: [32]
    
    auto proto_shape = proto.shape();
    if (proto_shape.size() != 4 || proto_shape[1] != 32) {
        throw std::runtime_error("Invalid proto mask shape");
    }
    
    int mh = proto_shape[2];
    int mw = proto_shape[3];
    int num_masks = 32;
    
    // 1. Matrix Multiplication: Mask = Coeffs * Proto
    cv::Mat proto_mat(num_masks, mh * mw, CV_32F, (void*)proto.raw_data());
    cv::Mat coeffs_mat(1, num_masks, CV_32F);
    
    for (int i = 0; i < num_masks; ++i) {
        coeffs_mat.at<float>(0, i) = mask_coeffs.get<float>(i + 1);
    }
    
    cv::Mat mask_flat = coeffs_mat * proto_mat; // 1 x 25600
    cv::Mat mask = mask_flat.reshape(1, mh); // 160 x 160
    
    // 2. Sigmoid
    cv::exp(-mask, mask);
    mask = 1.0f / (1.0f + mask);
    
    // 3. Resize to Input Size (e.g. 640x640)
    cv::Mat mask_input;
    cv::resize(mask, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
    
    // 4. Crop Padding (Remove letterbox padding)
    // ROI: [pad_x, pad_y, input_w - 2*pad_x, input_h - 2*pad_y]
    // Ensure ROI is within bounds
    int roi_w = input_w - 2 * pad_x;
    int roi_h = input_h - 2 * pad_y;
    
    cv::Rect roi(pad_x, pad_y, roi_w, roi_h);
    roi = roi & cv::Rect(0, 0, input_w, input_h);
    
    if (roi.area() == 0) {
         return LuaIntf::LuaRef::fromValue(L, Tensor(std::vector<float>(img_w * img_h, 0), {1, (int64_t)img_h, (int64_t)img_w}));
    }
    
    cv::Mat mask_cropped = mask_input(roi);
    
    // 5. Resize to Original Image Size
    cv::Mat mask_original;
    cv::resize(mask_cropped, mask_original, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);
    
    // 6. Crop by Box (set pixels outside box to 0)
    float bx = box.get<float>("x");
    float by = box.get<float>("y");
    float bw = box.get<float>("w");
    float bh = box.get<float>("h");
    
    cv::Rect box_rect(bx, by, bw, bh);
    box_rect = box_rect & cv::Rect(0, 0, img_w, img_h);
    
    cv::Mat final_mask = cv::Mat::zeros(img_h, img_w, CV_32F);
    if (box_rect.area() > 0) {
        mask_original(box_rect).copyTo(final_mask(box_rect));
    }
    
    // 7. Threshold (> 0.5)
    cv::threshold(final_mask, final_mask, 0.5, 1.0, cv::THRESH_BINARY);
    
    // Return as Tensor
    std::vector<float> mask_data(img_w * img_h);
    std::memcpy(mask_data.data(), final_mask.data, img_w * img_h * sizeof(float));
    
    return LuaIntf::LuaRef::fromValue(L, Tensor(std::move(mask_data), {1, (int64_t)img_h, (int64_t)img_w}));
}

// Session Implementation
Session::Session(const std::string& model_path)
    : env_(std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "model_infer")),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    // 会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // 创建会话
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    // 获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_.push_back(input_name.get());
        
        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_types_.push_back(tensor_info.GetElementType());
        input_shapes_.push_back(tensor_info.GetShape());
    }
    
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
}

LuaIntf::LuaRef Session::run(lua_State* L, const Tensor& input_tensor) {
    // 创建ONNX Runtime输入Tensor
    auto shape = input_tensor.shape();
    std::vector<Ort::Value> input_tensors;
    
    // Check expected type (assuming single input or first input matches)
    ONNXTensorElementDataType target_type = input_types_.empty() ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : input_types_[0];
    
    // Check expected shape and pad if necessary
    std::vector<float> padded_data;
    const float* input_data_ptr = input_tensor.raw_data();
    size_t input_data_size = input_tensor.size();
    
    if (!input_shapes_.empty() && input_shapes_[0].size() == 4) {
        int64_t model_h = input_shapes_[0][2];
        int64_t model_w = input_shapes_[0][3];
        
        if (model_h > 0 && model_w > 0 && shape.size() == 4) {
            int64_t input_h = shape[2];
            int64_t input_w = shape[3];
            
            if (input_h < model_h || input_w < model_w) {
                // Need padding
                // Assuming NCHW layout
                int64_t N = shape[0];
                int64_t C = shape[1];
                
                // New shape
                shape[2] = model_h;
                shape[3] = model_w;
                
                size_t new_size = N * C * model_h * model_w;
                padded_data.resize(new_size, 114.0f/255.0f); // Pad with gray
                
                // Copy data
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t c = 0; c < C; ++c) {
                        const float* src_ptr = input_data_ptr + (n * C + c) * input_h * input_w;
                        float* dst_ptr = padded_data.data() + (n * C + c) * model_h * model_w;
                        
                        for (int64_t h = 0; h < input_h; ++h) {
                            std::copy(src_ptr + h * input_w, src_ptr + h * input_w + input_w, dst_ptr + h * model_w);
                        }
                    }
                }
                
                // Update pointer and size
                input_data_ptr = padded_data.data();
                input_data_size = padded_data.size();
            }
        }
    }

    // Keep data alive during Run
    std::vector<Ort::Float16_t> fp16_data;
    
    if (target_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        size_t num_elements = input_data_size;
        const float* float_data = input_data_ptr;
        fp16_data.reserve(num_elements);
        
        for (size_t i = 0; i < num_elements; ++i) {
            fp16_data.emplace_back(float_data[i]);
        }
        
        input_tensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info_,
            fp16_data.data(),
            fp16_data.size(),
            shape.data(),
            shape.size()
        ));
    } else {
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input_data_ptr),
            input_data_size,
            shape.data(),
            shape.size()
        ));
    }
    
    // 执行推理
    std::vector<const char*> input_names_cstr, output_names_cstr;
    for (const auto& name : input_names_) input_names_cstr.push_back(name.c_str());
    for (const auto& name : output_names_) output_names_cstr.push_back(name.c_str());
    
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(), input_tensors.data(), input_tensors.size(),
        output_names_cstr.data(), output_names_cstr.size()
    );
    
    // 将输出转换为Lua table
    LuaIntf::LuaRef outputs = LuaIntf::LuaRef::createTable(L);
    
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        auto& ort_tensor = output_tensors[i];
        auto tensor_info = ort_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        
        // 复制数据到shared_ptr管理的vector
        ONNXTensorElementDataType output_type = tensor_info.GetElementType();
        size_t element_count = tensor_info.GetElementCount();
        
        std::vector<float> result_vec;
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
             const Ort::Float16_t* ort_data = ort_tensor.GetTensorData<Ort::Float16_t>();
             result_vec.reserve(element_count);
             for(size_t k=0; k<element_count; ++k) {
                 result_vec.push_back(ort_data[k].ToFloat());
             }
        } else {
             const float* ort_data = ort_tensor.GetTensorData<float>();
             result_vec.assign(ort_data, ort_data + element_count);
        }
        
        Tensor tensor(std::move(result_vec), shape);
        outputs[output_names_[i]] = tensor;
    }
    
    return outputs;
}

void register_module(lua_State* L) {
    using namespace LuaIntf;
    
    LuaBinding(L)
        .beginModule("lua_nn")
            // Tensor类绑定
            .beginClass<Tensor>("Tensor")
                .addConstructor(LUA_ARGS(
                    const std::vector<float>&,
                    const std::vector<int64_t>&
                ))
                
                // 属性
                .addProperty("ndim", &Tensor::ndim)
                .addFunction("shape", &Tensor::shape)
                .addFunction("strides", &Tensor::strides)
                .addFunction("size", static_cast<int64_t(Tensor::*)() const>(&Tensor::size))
                .addFunction("is_contiguous", &Tensor::is_contiguous)
                .addFunction("view", &Tensor::view)
                
                // Level 1: 基础形状操作
                .addFunction("slice", &Tensor::slice)
                .addFunction("reshape", &Tensor::reshape)
                .addFunction("transpose", 
                    static_cast<Tensor(Tensor::*)() const>(&Tensor::transpose))
                .addFunction("transpose_dims",
                    static_cast<Tensor(Tensor::*)(const std::vector<int>&) const>(&Tensor::transpose))
                .addFunction("squeeze", &Tensor::squeeze)
                .addFunction("unsqueeze", &Tensor::unsqueeze)
                
                // Level 2: 数学运算
                .addFunction("add", static_cast<Tensor(Tensor::*)(float) const>(&Tensor::add))
                .addFunction("add_tensor", static_cast<Tensor(Tensor::*)(const Tensor&) const>(&Tensor::add))
                .addFunction("sub", static_cast<Tensor(Tensor::*)(float) const>(&Tensor::sub))
                .addFunction("sub_tensor", static_cast<Tensor(Tensor::*)(const Tensor&) const>(&Tensor::sub))
                .addFunction("mul", static_cast<Tensor(Tensor::*)(float) const>(&Tensor::mul))
                .addFunction("mul_tensor", static_cast<Tensor(Tensor::*)(const Tensor&) const>(&Tensor::mul))
                .addFunction("div", static_cast<Tensor(Tensor::*)(float) const>(&Tensor::div))
                .addFunction("div_tensor", static_cast<Tensor(Tensor::*)(const Tensor&) const>(&Tensor::div))
                
                .addFunction("sum", &Tensor::sum)
                .addFunction("mean", &Tensor::mean)
                .addFunction("max", &Tensor::max)
                .addFunction("min", &Tensor::min)
                .addFunction("argmax", &Tensor::argmax_lua)
                .addFunction("argmin", &Tensor::argmin_lua)
                
                .addFunction("sigmoid", &Tensor::sigmoid)
                .addFunction("softmax", &Tensor::softmax)
                .addFunction("exp", &Tensor::exp_)
                .addFunction("log", &Tensor::log_)
                
                .addFunction("gt", &Tensor::gt)
                .addFunction("lt", &Tensor::lt)
                .addFunction("ge", &Tensor::ge)
                .addFunction("le", &Tensor::le)
                .addFunction("eq", &Tensor::eq)
                
                // Level 3: 高级操作
                .addFunction("topk_new", &Tensor::topk_lua)
                .addFunction("to_table", &Tensor::to_table)
                .addFunction("to_string", &Tensor::to_string)
                .addFunction("get", &Tensor::get_item)
                .addFunction("set", &Tensor::set_item)

                // 向量化过滤操作（方案3 - 通用API）
                .addFunction("nonzero", &Tensor::nonzero)
                .addFunction("where_indices", &Tensor::where_indices)
                .addFunction("index_select", &Tensor::index_select)
                .addFunction("extract_columns", &Tensor::extract_columns)

                // Legacy方法（向后兼容）
                .addFunction("filter_yolo", &Tensor::filter_yolo)
                .addFunction("filter_yolo_pose", &Tensor::filter_yolo_pose)
                .addFunction("filter_yolo_seg", &Tensor::filter_yolo_seg)
                .addFunction("process_mask", &Tensor::process_mask)
                .addFunction("argmax_old", &Tensor::argmax)
                .addFunction("topk", &Tensor::topk)
                
                // Metamethods
                .addMetaFunction("__len", [](const Tensor* t) { return t->size(); })
                .addMetaFunction("__tostring", [](const Tensor* t) {
                    return t->to_string(10);
                })
                .addMetaFunction("__add", [](Tensor& t, float scalar) { return t.add(scalar); })
                .addMetaFunction("__sub", [](Tensor& t, float scalar) { return t.sub(scalar); })
                .addMetaFunction("__mul", [](Tensor& t, float scalar) { return t.mul(scalar); })
                .addMetaFunction("__div", [](Tensor& t, float scalar) { return t.div(scalar); })
            .endClass()
            
            // TensorView绑定
            .beginClass<TensorView<float>>("FloatView")
                .addFunction("get", &TensorView<float>::get)
                .addFunction("set", &TensorView<float>::set)
                .addMetaFunction("__len", [](const TensorView<float>* t) { return t->length(); })
            .endClass()
            
            // Session绑定
            .beginClass<Session>("Session")
                .addConstructor(
                    LUA_SP(std::shared_ptr<Session>),
                    LUA_ARGS(const std::string&)
                )
                .addFunction("run", &Session::run)
                .addProperty("input_names", &Session::input_names)
                .addProperty("output_names", &Session::output_names)
            .endClass()
        .endModule();
}

} // namespace lua_nn
