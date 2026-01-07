#pragma once

#include <memory>
#include "device_type.h"

namespace tensor {

/**
 * Stream - 异步操作流抽象
 *
 * 不同设备的 Stream 实现：
 * - CPU: 无操作（同步执行）
 * - NPU (Rockchip): 使用 rknn_run_async() 的隐式流
 * - TPU (Sophgo): 使用 bm_handle_t 的流管理
 *
 * 使用模式：
 *   auto stream = Stream::create(DeviceType::NPU);
 *   tensor.copy_to_async(dst, stream.get());
 *   // ... 其他操作 ...
 *   stream->synchronize();  // 等待所有操作完成
 */
class Stream {
public:
    virtual ~Stream() = default;

    /**
     * 等待流中所有操作完成（阻塞）
     */
    virtual void synchronize() = 0;

    /**
     * 检查流是否空闲（非阻塞）
     * @return true 如果流中没有挂起的操作
     */
    virtual bool is_idle() const = 0;

    /**
     * 获取流所属的设备类型
     */
    virtual DeviceType device() const = 0;

    /**
     * 工厂方法：创建指定设备类型的流
     * @param device 目标设备类型
     * @return 新创建的流对象
     */
    static std::shared_ptr<Stream> create(DeviceType device);
};

/**
 * CpuStream - CPU 同步流（无操作实现）
 *
 * CPU 上的操作是同步执行的，因此 CpuStream 的所有方法都是空操作。
 * 这种设计允许使用统一的异步接口，同时在 CPU 上保持高效。
 */
class CpuStream : public Stream {
public:
    CpuStream() = default;
    ~CpuStream() override = default;

    /**
     * CPU 操作同步执行，无需等待
     */
    void synchronize() override {
        // CPU 操作同步执行，无需等待
    }

    /**
     * CPU 流始终空闲（操作立即完成）
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
