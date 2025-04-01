#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <utility>
#include "macros.h"

namespace core {

// 设备类型枚举
enum DeviceType {
    kCPU,
    kCUDA
};

// 设备类
class Device {
public:
    // 设备属性结构体
    struct Properties {
        std::string name;
        size_t total_memory;
        int compute_capability;
        int multi_processor_count;
        
        Properties() : 
            name(""), 
            total_memory(0), 
            compute_capability(0), 
            multi_processor_count(0) {}
    };
    
    // 构造函数
    Device(DeviceType type = kCPU, int index = 0) : 
        type_(type), index_(index) {}
    
    // 获取设备类型
    DeviceType type() const { return type_; }
    
    // 获取设备索引
    int index() const { return index_; }
    
    // 设备比较
    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }
    
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }
    
    // 用于unordered_map的哈希支持
    struct Hash {
        size_t operator()(const Device& d) const {
            return std::hash<int>()(static_cast<int>(d.type())) ^ 
                   (std::hash<int>()(d.index()) << 1);
        }
    };
    
    // 设备字符串表示
    std::string str() const {
        switch(type_) {
            case kCPU: return "cpu";
            case kCUDA: return "cuda:" + std::to_string(index_);
            default: return "unknown";
        }
    }
    
    // 静态方法
    static Device CPU() { return Device(kCPU); }
    static Device CUDA(int index = 0);
    
    // 获取设备属性
    Properties properties() const;
    
private:
    DeviceType type_;
    int index_;
};

// 设备管理器
class DeviceManager {
public:
    // 获取单例实例
    static DeviceManager& instance();
    
    // 设置当前设备
    void set_current_device(const Device& device);
    
    // 获取当前设备
    Device current_device() const { return current_device_; }
    
    // 获取设备数量
    int device_count(DeviceType type) const;
    
    // 同步设备
    void synchronize_device(const Device& device);
    
    // 内存统计
    size_t memory_allocated(const Device& device) const;
    size_t max_memory_allocated(const Device& device) const;
    
    // 获取设备属性
    const Device::Properties& get_device_properties(const Device& device) const;
    
    // 禁用拷贝
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    
private:
    DeviceManager();
    
    Device current_device_;
    std::unordered_map<Device, Device::Properties, Device::Hash> device_properties_;
};

// 设备保护类 (用于临时切换设备)
class DeviceGuard {
public:
    explicit DeviceGuard(const Device& device);
    ~DeviceGuard();
    
    // 禁用拷贝
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    
private:
    Device original_device_;
};

} // namespace core