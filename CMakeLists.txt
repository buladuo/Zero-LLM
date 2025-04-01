cmake_minimum_required(VERSION 3.18)  # Increased minimum version for better CUDA support
project(Zero-LLM LANGUAGES CXX C CUDA)  # Added CUDA as a language

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)  # CUDA standard version
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Project version
set(Zero-LLM_VERSION_MAJOR 0)
set(Zero-LLM_VERSION_MINOR 1)
set(Zero-LLM_VERSION_PATCH 0)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(WITH_GGML "Enable GGML backend" OFF)
option(WITH_CUDA "Enable CUDA support" OFF)  # New option for CUDA
option(WITH_CUBLAS "Enable cuBLAS support" OFF)  # Option for cuBLAS
option(WITH_CUDNN "Enable cuDNN support" OFF)  # Option for cuDNN

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA configuration
if(WITH_CUDA)
    enable_language(CUDA)
    
    find_package(CUDAToolkit REQUIRED)
    
    # Set CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v --default-stream per-thread")
    
    # Architecture flags (adjust based on your target GPUs)
    set(CUDA_ARCH "70;75;80;86" CACHE STRING "Target CUDA architectures")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
    
    if(WITH_CUBLAS)
        find_package(CUBLAS REQUIRED)
        add_definitions(-DWITH_CUBLAS)
    endif()
    
    if(WITH_CUDNN)
        find_package(CUDNN REQUIRED)
        add_definitions(-DWITH_CUDNN)
    endif()
    
    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA architectures: ${CUDA_ARCH}")
else()
    message(STATUS "CUDA support disabled")
endif()

# GGML dependency with CUDA support
if(WITH_GGML)
    include(FetchContent)
    
    FetchContent_Declare(
        ggml
        GIT_REPOSITORY https://github.com/ggerganov/ggml.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build GGML as static library" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable GGML tests" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable GGML examples" FORCE)
    
    # Enable CUDA in GGML if we have it
    if(WITH_CUDA)
        set(GGML_CUDA ON CACHE BOOL "Enable CUDA in GGML" FORCE)
    endif()
    
    FetchContent_MakeAvailable(ggml)
    include_directories(${ggml_SOURCE_DIR}/include)
    
    message(STATUS "GGML backend enabled")
else()
    message(STATUS "GGML backend disabled")
endif()

# Add subdirectories
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()