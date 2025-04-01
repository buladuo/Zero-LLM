#pragma once

#include <cstddef>  // for size_t
#include <stdexcept>
#include <string>

namespace core {

enum DType {
    kFloat64,   // 64位浮点数
    kFloat32,   // 32位浮点数 (常用)
    kFloat16,   // 16位浮点数 (节省内存，适合推理)
    kBFloat16,  // 16位脑浮点数 (训练友好)
    kInt32,     // 32位整数
    kInt64,     // 64位整数
    kUInt8,     // 8位无符号整数
    kBool       // 布尔类型
};

// 获取数据类型名称
inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case kFloat64:  return "float64";
        case kFloat32:  return "float32";
        case kFloat16:  return "float16";
        case kBFloat16: return "bfloat16";
        case kInt32:    return "int32";
        case kInt64:    return "int64";
        case kUInt8:    return "uint8";
        case kBool:     return "bool";
        default:
            throw std::runtime_error("Unknown dtype in dtype_name");
    }
}

// 获取数据类型大小(字节)
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case kFloat64:  return 8;
        case kFloat32:  return 4;
        case kFloat16:  return 2;
        case kBFloat16: return 2;
        case kInt32:    return 4;
        case kInt64:    return 8;
        case kUInt8:    return 1;
        case kBool:     return 1;
        default:
            throw std::runtime_error("Unsupported dtype in dtype_size");
    }
}

// 类型特征模板
template <DType> struct dtype_traits;
template <> struct dtype_traits<kFloat64>  { using ctype = double; };
template <> struct dtype_traits<kFloat32>  { using ctype = float; };
template <> struct dtype_traits<kFloat16>  { using ctype = uint16_t; };  // 特殊处理
template <> struct dtype_traits<kBFloat16> { using ctype = uint16_t; };  // 特殊处理
template <> struct dtype_traits<kInt32>    { using ctype = int32_t; };
template <> struct dtype_traits<kInt64>    { using ctype = int64_t; };
template <> struct dtype_traits<kUInt8>    { using ctype = uint8_t; };
template <> struct dtype_traits<kBool>     { using ctype = bool; };

// 辅助函数：判断是否为浮点类型
inline bool is_floating_point(DType dtype) {
    return dtype == kFloat64 || dtype == kFloat32 || 
           dtype == kFloat16 || dtype == kBFloat16;
}

// 辅助函数：判断是否为整数类型
inline bool is_integer(DType dtype) {
    return dtype == kInt32 || dtype == kInt64 || dtype == kUInt8;
}

// 辅助函数：判断是否为布尔类型
inline bool is_bool(DType dtype) {
    return dtype == kBool;
}

} // namespace core