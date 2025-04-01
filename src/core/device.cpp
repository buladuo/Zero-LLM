#include "core/device.h"
#include "core/macros.h"
#include "utils/logger.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace core {

// Device 实现
Device Device::CUDA(int index) { 
#ifdef WITH_CUDA
    return Device(kCUDA, index);
#else
    LOG_ERROR() << "CUDA support not compiled";
    return Device(kCPU);
#endif
}

Device::Properties Device::properties() const {
    Properties props;
    
    switch(type_) {
        case kCPU:
            props.name = "CPU";
            props.total_memory = 0; // 通常不限制
            break;
            
        case kCUDA:
#ifdef WITH_CUDA
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, index_));
            props.name = prop.name;
            props.total_memory = prop.totalGlobalMem;
            props.compute_capability = prop.major * 10 + prop.minor;
            props.multi_processor_count = prop.multiProcessorCount;
#else
            LOG_ERROR() << "CUDA support not compiled";
#endif
            break;
    }
    
    return props;
}

// DeviceManager 实现
DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

DeviceManager::DeviceManager() : current_device_(Device::CPU()) {
    // 初始化设备属性
#ifdef WITH_CUDA
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    for (int i = 0; i < device_count; ++i) {
        Device device = Device::CUDA(i);
        device_properties_.emplace(device, device.properties());
    }
#endif
}

void DeviceManager::set_current_device(const Device& device) {
    if (device != current_device_) {
        if (device.type() == kCUDA) {
#ifdef WITH_CUDA
            CUDA_CHECK(cudaSetDevice(device.index()));
#else
            LOG_ERROR() << "CUDA support not compiled";
#endif
        }
        current_device_ = device;
    }
}

int DeviceManager::device_count(DeviceType type) const {
    switch(type) {
        case kCPU: return 1;
        case kCUDA: 
#ifdef WITH_CUDA
            int count;
            CUDA_CHECK(cudaGetDeviceCount(&count));
            return count;
#else
            return 0;
#endif
        default: return 0;
    }
}

void DeviceManager::synchronize_device(const Device& device) {
    if (device.type() == kCUDA) {
#ifdef WITH_CUDA
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
    }
}

size_t DeviceManager::memory_allocated(const Device& device) const {
    if (device.type() == kCUDA) {
#ifdef WITH_CUDA
        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return total - free;
#else
        return 0;
#endif
    }
    return 0;
}

size_t DeviceManager::max_memory_allocated(const Device& device) const {
    // 实现可以使用静态变量跟踪最大值
    return 0;
}

const Device::Properties& DeviceManager::get_device_properties(const Device& device) const {
    static Device::Properties empty_props;
    auto it = device_properties_.find(device);
    if (it != device_properties_.end()) {
        return it->second;
    }
    return empty_props;
}

// DeviceGuard 实现
DeviceGuard::DeviceGuard(const Device& device) 
    : original_device_(DeviceManager::instance().current_device()) {
    DeviceManager::instance().set_current_device(device);
}

DeviceGuard::~DeviceGuard() {
    DeviceManager::instance().set_current_device(original_device_);
}

} // namespace core