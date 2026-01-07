#include "cpu_memory.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace tensor {

std::shared_ptr<CpuMemory> CpuMemory::allocate(size_t size_bytes, size_t alignment, bool zero_init) {
    if (size_bytes == 0) {
        throw std::invalid_argument("CpuMemory::allocate: size_bytes cannot be 0");
    }

    // 对齐要求必须是 2 的幂
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("CpuMemory::allocate: alignment must be power of 2");
    }

    auto storage = std::shared_ptr<CpuMemory>(new CpuMemory());

    // 使用 aligned_alloc 分配对齐内存
    // size_bytes 必须是 alignment 的倍数
    size_t aligned_size = ((size_bytes + alignment - 1) / alignment) * alignment;

#if defined(_WIN32)
    storage->data_ = _aligned_malloc(aligned_size, alignment);
#else
    storage->data_ = std::aligned_alloc(alignment, aligned_size);
#endif

    if (!storage->data_) {
        throw std::bad_alloc();
    }

    // 只在明确需要时初始化为 0（默认跳过，提高性能）
    if (zero_init) {
        std::memset(storage->data_, 0, aligned_size);
    }

    storage->size_bytes_ = size_bytes;
    storage->alignment_ = alignment;
    storage->owns_memory_ = true;

    return storage;
}

std::shared_ptr<CpuMemory> CpuMemory::from_external(void* ptr, size_t size_bytes, bool take_ownership) {
    if (!ptr) {
        throw std::invalid_argument("CpuMemory::from_external: ptr cannot be null");
    }
    if (size_bytes == 0) {
        throw std::invalid_argument("CpuMemory::from_external: size_bytes cannot be 0");
    }

    auto storage = std::shared_ptr<CpuMemory>(new CpuMemory());
    storage->data_ = ptr;
    storage->size_bytes_ = size_bytes;
    storage->alignment_ = 1;  // 外部内存对齐未知
    storage->owns_memory_ = take_ownership;

    return storage;
}

CpuMemory::~CpuMemory() {
    if (owns_memory_ && data_) {
#if defined(_WIN32)
        _aligned_free(data_);
#else
        std::free(data_);
#endif
        data_ = nullptr;
    }
}

CpuMemory::CpuMemory(CpuMemory&& other) noexcept
    : data_(other.data_),
      size_bytes_(other.size_bytes_),
      alignment_(other.alignment_),
      owns_memory_(other.owns_memory_) {
    other.data_ = nullptr;
    other.size_bytes_ = 0;
    other.owns_memory_ = false;
}

CpuMemory& CpuMemory::operator=(CpuMemory&& other) noexcept {
    if (this != &other) {
        // 释放当前资源
        if (owns_memory_ && data_) {
#if defined(_WIN32)
            _aligned_free(data_);
#else
            std::free(data_);
#endif
        }

        // 移动资源
        data_ = other.data_;
        size_bytes_ = other.size_bytes_;
        alignment_ = other.alignment_;
        owns_memory_ = other.owns_memory_;

        // 清空源对象
        other.data_ = nullptr;
        other.size_bytes_ = 0;
        other.owns_memory_ = false;
    }
    return *this;
}

void CpuMemory::copy_to(DeviceBuffer* dst) const {
    if (!dst) {
        throw std::invalid_argument("CpuMemory::copy_to: dst cannot be null");
    }
    if (dst->size_bytes() < size_bytes_) {
        throw std::invalid_argument("CpuMemory::copy_to: dst too small");
    }

    if (dst->device() == DeviceType::CPU) {
        // CPU -> CPU: 直接 memcpy
        std::memcpy(dst->data(), data_, size_bytes_);
    } else {
        // CPU -> NPU/TPU: 需要子类实现
        throw std::runtime_error("CpuMemory::copy_to: cross-device copy not implemented for target device");
    }
}

void CpuMemory::copy_from(const DeviceBuffer* src) {
    if (!src) {
        throw std::invalid_argument("CpuMemory::copy_from: src cannot be null");
    }
    if (size_bytes_ < src->size_bytes()) {
        throw std::invalid_argument("CpuMemory::copy_from: this buffer too small");
    }

    if (src->device() == DeviceType::CPU) {
        // CPU -> CPU: 直接 memcpy
        std::memcpy(data_, src->data(), src->size_bytes());
    } else {
        // NPU/TPU -> CPU: 需要子类实现
        throw std::runtime_error("CpuMemory::copy_from: cross-device copy not implemented for source device");
    }
}

// ========== 异步接口实现（CPU 同步降级） ==========

void CpuMemory::copy_to_async(DeviceBuffer* dst, Stream* stream) const {
    // CPU 实现：直接同步执行，忽略 stream
    (void)stream;  // 忽略未使用参数
    copy_to(dst);
}

void CpuMemory::copy_from_async(const DeviceBuffer* src, Stream* stream) {
    // CPU 实现：直接同步执行，忽略 stream
    (void)stream;  // 忽略未使用参数
    copy_from(src);
}

void CpuMemory::sync(Stream* stream) const {
    // CPU 操作同步执行，无需等待
    (void)stream;  // 忽略未使用参数
}

bool CpuMemory::supports_async() const {
    // CPU 不支持真正的异步操作
    return false;
}

} // namespace tensor
