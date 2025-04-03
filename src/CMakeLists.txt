# Core library
add_library(core
    core/allocator.cpp
    core/device.cpp
    core/tensor.cpp
    core/tensor_ops.cpp
    core/autograd/engine.cpp
    core/autograd/variable.cpp
    core/autograd/function/math/add.cpp
    core/autograd/function/math/matmul.cpp
)

target_include_directories(core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/include/core/autograd
    ${CMAKE_SOURCE_DIR}/include/core/autograd/function
    ${CMAKE_SOURCE_DIR}/include/core/autograd/function/math
)

# Layers library
add_library(layers
    layers/activation.cpp
    layers/conv2d.cpp
    layers/dropout.cpp
    layers/linear.cpp
    layers/norm.cpp
)
target_include_directories(layers PUBLIC ${CMAKE_SOURCE_DIR}/include/layers)
target_link_libraries(layers PUBLIC core)

# LLM library
add_library(llm
    llm/attention/core.cpp
    llm/attention/flash.cpp
    llm/attention/rotary.cpp
    llm/model.cpp
    llm/tokenizer/bpe.cpp
    llm/tokenizer/tokenizer.cpp
    llm/transformer.cpp
)
target_include_directories(llm PUBLIC 
    ${CMAKE_SOURCE_DIR}/include/llm
    ${CMAKE_SOURCE_DIR}/include/llm/attention
    ${CMAKE_SOURCE_DIR}/include/llm/tokenizer
)
target_link_libraries(llm PUBLIC core layers)

# Loss library
add_library(loss
    loss/cross_entropy_loss.cpp
    loss/mse_loss.cpp
)
target_include_directories(loss PUBLIC ${CMAKE_SOURCE_DIR}/include/loss)
target_link_libraries(loss PUBLIC core)

# Optim library
add_library(optim
    optim/adam.cpp
    optim/adamw.cpp
    optim/sgd.cpp
)
target_include_directories(optim PUBLIC ${CMAKE_SOURCE_DIR}/include/optim)
target_link_libraries(optim PUBLIC core)

# Utils library
add_library(utils
    utils/checkpoint.cpp
    utils/dataloader.cpp
    utils/logger.cpp
)
target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR}/include/utils)
target_link_libraries(utils PUBLIC core)

# Main Zero-LLM library that combines everything
add_library(zero-llm
    $<TARGET_OBJECTS:core>
    $<TARGET_OBJECTS:layers>
    $<TARGET_OBJECTS:llm>
    $<TARGET_OBJECTS:loss>
    $<TARGET_OBJECTS:optim>
    $<TARGET_OBJECTS:utils>
)
target_link_libraries(zero-llm PUBLIC core layers llm loss optim utils)

# Install targets
install(TARGETS zero-llm core layers llm loss optim utils
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)