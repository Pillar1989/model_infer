#pragma once

#include <cstddef>
#include <memory>
#include "device_type.h"

namespace tensor {

// 前向声明
class SyncHandle;

/**
 * DeviceBuffer - 设备缓冲区抽象基类
 *
 * 提供设备无关的数据缓冲区接口，支持：
 * - 多设备缓冲区管理 (CPU, NPU, TPU)
 * - 内存对齐要求
 * - 零拷贝引用外部内存
 * - 设备间数据传输
 */
class DeviceBuffer {
public:
    virtual ~DeviceBuffer() = default;

    // ========== 数据访问 ==========

    /**
     * 获取数据指针
     * 对于设备内存，返回设备端指针
     */
    virtual void* data() = 0;
    virtual const void* data() const = 0;

    /**
     * 获取数据大小（字节）
     */
    virtual size_t size_bytes() const = 0;

    // ========== 设备信息 ==========

    /**
     * 获取设备类型
     */
    virtual DeviceType device() const = 0;

    /**
     * 获取内存对齐要求
     */
    virtual size_t alignment() const = 0;

    // ========== 内存所有权 ==========

    /**
     * 是否拥有内存所有权
     * false 表示引用外部内存，不负责释放
     */
    virtual bool owns_memory() const = 0;

    // ========== 数据传输 ==========

    /**
     * 复制数据到目标缓冲区
     * @param dst 目标缓冲区（可以是不同设备）
     */
    virtual void copy_to(DeviceBuffer* dst) const = 0;

    /**
     * 从源缓冲区复制数据
     * @param src 源缓冲区（可以是不同设备）
     */
    virtual void copy_from(const DeviceBuffer* src) = 0;

    // ========== 异步数据传输 ==========

    /**
     * 异步复制数据到目标缓冲区
     * @param dst 目标缓冲区（可以是不同设备）
     * @param handle 同步句柄（nullptr 表示使用默认句柄）
     *
     * 注意：调用者需要在访问 dst 数据前调用 handle->synchronize()
     */
    virtual void copy_to_async(DeviceBuffer* dst, SyncHandle* handle = nullptr) const = 0;

    /**
     * 异步从源缓冲区复制数据
     * @param src 源缓冲区（可以是不同设备）
     * @param handle 同步句柄（nullptr 表示使用默认句柄）
     */
    virtual void copy_from_async(const DeviceBuffer* src, SyncHandle* handle = nullptr) = 0;

    /**
     * 同步等待所有挂起的异步操作完成
     * @param handle 要同步的句柄（nullptr 表示默认句柄）
     */
    virtual void sync(SyncHandle* handle = nullptr) const = 0;

    /**
     * 检查是否支持异步操作
     * @return CPU 返回 false，NPU/TPU 返回 true
     */
    virtual bool supports_async() const = 0;

    // ========== 工厂方法 ==========

    /**
     * 分配指定大小的缓冲区
     * @param size_bytes 字节数
     * @param device 设备类型（默认 CPU）
     * @param alignment 对齐要求（默认 64 字节）
     */
    static std::shared_ptr<DeviceBuffer> allocate(
        size_t size_bytes,
        DeviceType device = DeviceType::CPU,
        size_t alignment = 64
    );

    /**
     * 引用外部内存（零拷贝）
     * @param ptr 外部内存指针
     * @param size_bytes 字节数
     * @param device 设备类型
     * @param take_ownership 是否接管所有权
     */
    static std::shared_ptr<DeviceBuffer> from_external(
        void* ptr,
        size_t size_bytes,
        DeviceType device = DeviceType::CPU,
        bool take_ownership = false
    );
};

} // namespace tensor
