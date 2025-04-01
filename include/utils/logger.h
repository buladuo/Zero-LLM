#pragma once

#include <string>
#include <iostream>
#include <sstream>

namespace core {

class Logger {
public:
    enum LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL
    };

    explicit Logger(const std::string& context = "", LogLevel level = INFO);
    ~Logger();

    template<typename T>
    Logger& operator<<(const T& msg) {
        stream_ << msg;
        return *this;
    }

    static void set_log_level(LogLevel level);

private:
    std::ostringstream stream_;
    LogLevel level_;
    std::string context_;
    static LogLevel global_level_;
};

// 日志宏定义
#define LOG_DEBUG()   core::Logger(__FILE__, core::Logger::DEBUG)
#define LOG_INFO()    core::Logger(__FILE__, core::Logger::INFO)
#define LOG_WARNING() core::Logger(__FILE__, core::Logger::WARNING)
#define LOG_ERROR()   core::Logger(__FILE__, core::Logger::ERROR)
#define LOG_FATAL()   core::Logger(__FILE__, core::Logger::FATAL)

} // namespace core