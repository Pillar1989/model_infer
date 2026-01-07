#include "device_buffer.h"
#include "cpu_memory.h"
#include <stdexcept>

namespace tensor {

std::shared_ptr<DeviceBuffer> DeviceBuffer::allocate(
    size_t size_bytes,
    DeviceType device,
    size_t alignment
) {
    switch (device) {
        case DeviceType::CPU:
            return CpuMemory::allocate(size_bytes, alignment);

        case DeviceType::NPU:
            // TODO: 实现 Rockchip NPU Memory
            throw std::runtime_error("DeviceBuffer::allocate: NPU buffer not implemented");

        case DeviceType::TPU:
            // TODO: 实现 Sophgo TPU Memory
            throw std::runtime_error("DeviceBuffer::allocate: TPU buffer not implemented");

        default:
            throw std::invalid_argument("DeviceBuffer::allocate: unknown device type");
    }
}

std::shared_ptr<DeviceBuffer> DeviceBuffer::from_external(
    void* ptr,
    size_t size_bytes,
    DeviceType device,
    bool take_ownership
) {
    switch (device) {
        case DeviceType::CPU:
            return CpuMemory::from_external(ptr, size_bytes, take_ownership);

        case DeviceType::NPU:
            // TODO: 实现 Rockchip NPU Memory
            throw std::runtime_error("DeviceBuffer::from_external: NPU buffer not implemented");

        case DeviceType::TPU:
            // TODO: 实现 Sophgo TPU Memory
            throw std::runtime_error("DeviceBuffer::from_external: TPU buffer not implemented");

        default:
            throw std::invalid_argument("DeviceBuffer::from_external: unknown device type");
    }
}

} // namespace tensor
