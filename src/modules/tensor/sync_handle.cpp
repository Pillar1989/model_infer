#include "sync_handle.h"
#include <stdexcept>

namespace tensor {

std::shared_ptr<SyncHandle> SyncHandle::create(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
            return std::make_shared<CpuSyncHandle>();

        case DeviceType::NPU:
            // TODO: 实现 Rockchip NPU SyncHandle
            // 将使用 rknn_run_async() 的回调机制
            throw std::runtime_error("SyncHandle::create: NPU handle not implemented");

        case DeviceType::TPU:
            // TODO: 实现 Sophgo TPU SyncHandle
            // 将使用 bm_handle_t 的同步管理
            throw std::runtime_error("SyncHandle::create: TPU handle not implemented");

        default:
            throw std::invalid_argument("SyncHandle::create: unknown device type");
    }
}

} // namespace tensor
