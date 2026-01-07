#include "stream.h"
#include <stdexcept>

namespace tensor {

std::shared_ptr<Stream> Stream::create(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
            return std::make_shared<CpuStream>();

        case DeviceType::NPU:
            // TODO: 实现 Rockchip NPU Stream
            // 将使用 rknn_run_async() 的回调机制
            throw std::runtime_error("Stream::create: NPU stream not implemented");

        case DeviceType::TPU:
            // TODO: 实现 Sophgo TPU Stream
            // 将使用 bm_handle_t 的流管理
            throw std::runtime_error("Stream::create: TPU stream not implemented");

        default:
            throw std::invalid_argument("Stream::create: unknown device type");
    }
}

} // namespace tensor
