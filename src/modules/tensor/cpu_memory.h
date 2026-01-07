#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include "device_buffer.h"

namespace tensor {

/**
 * CpuMemory - CPU 内存管理实现
 *
 * 特点：
 * - 支持 64 字节对齐（满足 SIMD 和大多数 NPU 要求）
 * - 支持引用外部内存（零拷贝）
 * - 使用 aligned_alloc/free 进行内存分配和释放
 */
class CpuMemory : public DeviceBuffer {
public:
    /**
     * 分配对齐内存
     * @param size_bytes 字节数
     * @param alignment 对齐要求（默认 64 字节）
     * @param zero_init 是否初始化为0（默认 false，提高性能）
     */
    static std::shared_ptr<CpuMemory> allocate(size_t size_bytes, size_t alignment = 64, bool zero_init = false);

    /**
     * 引用外部内存（零拷贝）
     * @param ptr 外部内存指针
     * @param size_bytes 字节数
     * @param take_ownership 是否接管所有权
     */
    static std::shared_ptr<CpuMemory> from_external(void* ptr, size_t size_bytes, bool take_ownership = false);

    ~CpuMemory() override;

    // 禁用拷贝
    CpuMemory(const CpuMemory&) = delete;
    CpuMemory& operator=(const CpuMemory&) = delete;

    // 允许移动
    CpuMemory(CpuMemory&& other) noexcept;
    CpuMemory& operator=(CpuMemory&& other) noexcept;

    // ========== DeviceBuffer 接口实现 ==========

    void* data() override { return data_; }
    const void* data() const override { return data_; }
    size_t size_bytes() const override { return size_bytes_; }
    DeviceType device() const override { return DeviceType::CPU; }
    size_t alignment() const override { return alignment_; }
    bool owns_memory() const override { return owns_memory_; }

    void copy_to(DeviceBuffer* dst) const override;
    void copy_from(const DeviceBuffer* src) override;

    // ========== 异步接口（CPU 实现为同步降级） ==========

    void copy_to_async(DeviceBuffer* dst, Stream* stream = nullptr) const override;
    void copy_from_async(const DeviceBuffer* src, Stream* stream = nullptr) override;
    void sync(Stream* stream = nullptr) const override;
    bool supports_async() const override;

private:
    CpuMemory() = default;

    void* data_ = nullptr;
    size_t size_bytes_ = 0;
    size_t alignment_ = 64;
    bool owns_memory_ = false;
};

} // namespace tensor
