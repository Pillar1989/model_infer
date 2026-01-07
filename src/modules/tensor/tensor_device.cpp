#include "tensor.h"
#include "cpu_memory.h"
#include <cstring>
#include <stdexcept>

#include "LuaIntf.h"

namespace tensor {

// ========== 设备操作 ==========

Tensor Tensor::to(DeviceType device) const {
    if (buffer_->device() == device) {
        return *this;
    }

    // 先确保连续
    Tensor cont = contiguous();

    // 分配目标设备存储
    auto new_storage = DeviceBuffer::allocate(cont.buffer_->size_bytes(), device);

    // 复制数据
    cont.buffer_->copy_to(new_storage.get());

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

Tensor Tensor::contiguous() const {
    if (contiguous_) {
        return *this;
    }
    return contiguous_copy();
}

// ========== 异步设备操作 ==========

Tensor Tensor::to_async(DeviceType device, SyncHandle* handle) const {
    if (buffer_->device() == device) {
        return *this;
    }

    // 先确保连续
    Tensor cont = contiguous();

    // 分配目标设备存储
    auto new_storage = DeviceBuffer::allocate(cont.buffer_->size_bytes(), device);

    // 异步复制数据
    cont.buffer_->copy_to_async(new_storage.get(), handle);

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

void Tensor::sync(SyncHandle* handle) const {
    buffer_->sync(handle);
}

bool Tensor::is_ready(SyncHandle* handle) const {
    if (handle) {
        return handle->is_idle();
    }
    // 无句柄时默认已就绪（CPU 同步执行）
    return true;
}

Tensor Tensor::contiguous_copy() const {
    if (contiguous_) {
        return *this;
    }

    check_cpu();

    int64_t total_size = compute_size();
    auto new_storage = CpuMemory::allocate(total_size * sizeof(float));
    float* dst = static_cast<float*>(new_storage->data());
    const float* src = static_cast<const float*>(buffer_->data()) + offset_;
    int ndim = static_cast<int>(shape_.size());

    // ========== 1D 特化 ==========
    if (ndim == 1) {
        int64_t n = shape_[0];
        int64_t stride = strides_[0];

        if (stride == 1) {
            std::memcpy(dst, src, n * sizeof(float));
        } else {
            // 编译器可以向量化此简单循环
            for (int64_t i = 0; i < n; ++i) {
                dst[i] = src[i * stride];
            }
        }
    }
    // ========== 2D 特化 ==========
    else if (ndim == 2) {
        int64_t n0 = shape_[0], n1 = shape_[1];
        int64_t s0 = strides_[0], s1 = strides_[1];

        if (s1 == 1 && s0 == n1) {
            // 完全连续
            std::memcpy(dst, src, total_size * sizeof(float));
        } else if (s1 == 1) {
            // 行连续：每行 memcpy
            for (int64_t i = 0; i < n0; ++i) {
                std::memcpy(dst + i * n1, src + i * s0, n1 * sizeof(float));
            }
        } else {
            // 通用步进：硬编码双循环
            int64_t dst_idx = 0;
            for (int64_t i = 0; i < n0; ++i) {
                for (int64_t j = 0; j < n1; ++j) {
                    dst[dst_idx++] = src[i * s0 + j * s1];
                }
            }
        }
    }
    // ========== 3D 特化 ==========
    else if (ndim == 3) {
        int64_t n0 = shape_[0], n1 = shape_[1], n2 = shape_[2];
        int64_t s0 = strides_[0], s1 = strides_[1], s2 = strides_[2];

        if (s2 == 1 && s1 == n2 && s0 == n1 * n2) {
            // 完全连续
            std::memcpy(dst, src, total_size * sizeof(float));
        } else if (s2 == 1 && s1 == n2) {
            // 平面连续
            int64_t plane_size = n1 * n2;
            for (int64_t i = 0; i < n0; ++i) {
                std::memcpy(dst + i * plane_size, src + i * s0, plane_size * sizeof(float));
            }
        } else if (s2 == 1) {
            // 行连续
            int64_t dst_idx = 0;
            for (int64_t i = 0; i < n0; ++i) {
                for (int64_t j = 0; j < n1; ++j) {
                    std::memcpy(dst + dst_idx, src + i * s0 + j * s1, n2 * sizeof(float));
                    dst_idx += n2;
                }
            }
        } else {
            // 通用步进：硬编码三循环
            int64_t dst_idx = 0;
            for (int64_t i = 0; i < n0; ++i) {
                for (int64_t j = 0; j < n1; ++j) {
                    for (int64_t k = 0; k < n2; ++k) {
                        dst[dst_idx++] = src[i * s0 + j * s1 + k * s2];
                    }
                }
            }
        }
    }
    // ========== 4D 特化 ==========
    else if (ndim == 4) {
        int64_t n0 = shape_[0], n1 = shape_[1], n2 = shape_[2], n3 = shape_[3];
        int64_t s0 = strides_[0], s1 = strides_[1], s2 = strides_[2], s3 = strides_[3];

        if (s3 == 1) {
            // 最内层连续：批量 memcpy
            int64_t dst_idx = 0;
            for (int64_t i = 0; i < n0; ++i) {
                for (int64_t j = 0; j < n1; ++j) {
                    for (int64_t k = 0; k < n2; ++k) {
                        std::memcpy(dst + dst_idx, src + i * s0 + j * s1 + k * s2, n3 * sizeof(float));
                        dst_idx += n3;
                    }
                }
            }
        } else {
            // 通用步进：硬编码四循环
            int64_t dst_idx = 0;
            for (int64_t i = 0; i < n0; ++i) {
                for (int64_t j = 0; j < n1; ++j) {
                    for (int64_t k = 0; k < n2; ++k) {
                        for (int64_t l = 0; l < n3; ++l) {
                            dst[dst_idx++] = src[i * s0 + j * s1 + k * s2 + l * s3];
                        }
                    }
                }
            }
        }
    }
    // ========== 通用路径 (ndim > 4, 罕见) ==========
    else {
        // 找到最内层连续块
        int64_t inner_size = 1;
        int contiguous_dims = 0;
        for (int i = ndim - 1; i >= 0; --i) {
            if (strides_[i] == inner_size) {
                inner_size *= shape_[i];
                contiguous_dims++;
            } else break;
        }

        if (contiguous_dims == ndim) {
            std::memcpy(dst, src, total_size * sizeof(float));
        } else {
            int64_t outer_size = total_size / inner_size;
            int64_t block_bytes = inner_size * sizeof(float);

            for (int64_t o = 0; o < outer_size; ++o) {
                int64_t src_offset = 0;
                int64_t idx = o;
                for (int d = ndim - contiguous_dims - 1; d >= 0; --d) {
                    int64_t coord = idx % shape_[d];
                    idx /= shape_[d];
                    src_offset += coord * strides_[d];
                }
                std::memcpy(dst, src + src_offset, block_bytes);
                dst += inner_size;
            }
        }
    }

    return Tensor(new_storage, shape_, compute_strides(shape_), 0, true);
}

// ========== 零拷贝视图 ==========

LuaIntf::TensorView<float> Tensor::view() {
    check_cpu();
    if (!contiguous_) {
        Tensor cont = contiguous_copy();
        return LuaIntf::TensorView<float>(
            static_cast<float*>(cont.buffer_->data()),
            cont.compute_size(),
            cont.buffer_
        );
    }
    return LuaIntf::TensorView<float>(
        static_cast<float*>(buffer_->data()) + offset_,
        compute_size(),
        buffer_
    );
}

} // namespace tensor
