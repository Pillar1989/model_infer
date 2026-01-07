#include "tensor.h"
#include "cpu_memory.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "LuaIntf.h"

namespace tensor {

// ========== 向量化过滤操作 ==========

std::vector<int64_t> Tensor::nonzero() const {
    check_cpu();
    Tensor a = contiguous();

    std::vector<int64_t> indices;
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
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
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;
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
    const float* src = static_cast<const float*>(a.buffer_->data()) + a.offset_;

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


} // namespace tensor
