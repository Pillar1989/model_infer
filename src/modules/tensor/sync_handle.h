#pragma once

#include <memory>
#include "device_type.h"

namespace tensor {

/**
 * SyncHandle - 同步句柄抽象
 *
 * 用于管理异步操作的同步点，不同设备的实现：
 * - CPU: 无操作（同步执行）
 * - NPU (Rockchip): 使用 rknn_run_async() 的隐式同步
 * - TPU (Sophgo): 使用 bm_handle_t 的同步管理
 *
 * 使用模式：
 *   auto handle = SyncHandle::create(DeviceType::NPU);
 *   tensor.copy_to_async(dst, handle.get());
 *   // ... 其他操作 ...
 *   handle->synchronize();  // 等待所有操作完成
 */
class SyncHandle {
public:
    virtual ~SyncHandle() = default;

    /**
     * 等待所有挂起的操作完成（阻塞）
     */
    virtual void synchronize() = 0;

    /**
     * 检查是否所有操作已完成（非阻塞）
     * @return true 如果没有挂起的操作
     */
    virtual bool is_idle() const = 0;

    /**
     * 获取句柄所属的设备类型
     */
    virtual DeviceType device() const = 0;

    /**
     * 工厂方法：创建指定设备类型的同步句柄
     * @param device 目标设备类型
     * @return 新创建的同步句柄
     */
    static std::shared_ptr<SyncHandle> create(DeviceType device);
};

/**
 * CpuSyncHandle - CPU 同步句柄（无操作实现）
 *
 * CPU 上的操作是同步执行的，因此 CpuSyncHandle 的所有方法都是空操作。
 * 这种设计允许使用统一的异步接口，同时在 CPU 上保持高效。
 */
class CpuSyncHandle : public SyncHandle {
public:
    CpuSyncHandle() = default;
    ~CpuSyncHandle() override = default;

    /**
     * CPU 操作同步执行，无需等待
     */
    void synchronize() override {
        // CPU 操作同步执行，无需等待
    }

    /**
     * CPU 句柄始终空闲（操作立即完成）
     */
    bool is_idle() const override {
        return true;
    }

    /**
     * 返回 CPU 设备类型
     */
    DeviceType device() const override {
        return DeviceType::CPU;
    }
};

} // namespace tensor
