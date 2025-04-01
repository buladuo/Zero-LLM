#include "utils/logger.h"
#include <ctime>
#include <iomanip>

namespace core {

Logger::LogLevel Logger::global_level_ = Logger::INFO;

Logger::Logger(const std::string& context, LogLevel level) 
    : level_(level), context_(context) {}

Logger::~Logger() {
    if (level_ < global_level_) return;

    const char* level_names[] = {"DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};
    
    // 获取当前时间
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    
    // 输出日志
    std::cerr << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] "
              << "[" << level_names[level_] << "] "
              << "[" << context_ << "] "
              << stream_.str() << std::endl;
    
    if (level_ == FATAL) {
        std::abort();
    }
}

void Logger::set_log_level(LogLevel level) {
    global_level_ = level;
}

} // namespace core