#include "tensor_storage.h"
#include "cpu_storage.h"
#include <stdexcept>

namespace tensor {

std::shared_ptr<TensorStorage> TensorStorage::allocate(
    size_t size_bytes,
    DeviceType device,
    size_t alignment
) {
    switch (device) {
        case DeviceType::CPU:
            return CpuStorage::allocate(size_bytes, alignment);

        case DeviceType::NPU:
            // TODO: 实现 Rockchip NPU Storage
            throw std::runtime_error("TensorStorage::allocate: NPU storage not implemented");

        case DeviceType::TPU:
            // TODO: 实现 Sophgo TPU Storage
            throw std::runtime_error("TensorStorage::allocate: TPU storage not implemented");

        default:
            throw std::invalid_argument("TensorStorage::allocate: unknown device type");
    }
}

std::shared_ptr<TensorStorage> TensorStorage::from_external(
    void* ptr,
    size_t size_bytes,
    DeviceType device,
    bool take_ownership
) {
    switch (device) {
        case DeviceType::CPU:
            return CpuStorage::from_external(ptr, size_bytes, take_ownership);

        case DeviceType::NPU:
            // TODO: 实现 Rockchip NPU Storage
            throw std::runtime_error("TensorStorage::from_external: NPU storage not implemented");

        case DeviceType::TPU:
            // TODO: 实现 Sophgo TPU Storage
            throw std::runtime_error("TensorStorage::from_external: TPU storage not implemented");

        default:
            throw std::invalid_argument("TensorStorage::from_external: unknown device type");
    }
}

} // namespace tensor
