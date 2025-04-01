# CIFAR10 example
add_executable(cifar10_example cifar10/train.cpp)
target_link_libraries(cifar10_example PRIVATE zero-llm)

# MNIST example
add_executable(mnist_example mnist/train.cpp)
target_link_libraries(mnist_example PRIVATE zero-llm)

# LLM examples
add_executable(llm_pretrain_example llm/pretrain/train.cpp)
target_link_libraries(llm_pretrain_example PRIVATE zero-llm)

add_executable(llm_finetune_example llm/finetune/finetune.cpp)
target_link_libraries(llm_finetune_example PRIVATE zero-llm)

add_executable(llm_inference_example llm/inference/inference.cpp)
target_link_libraries(llm_inference_example PRIVATE zero-llm)

# Install examples
install(TARGETS 
    cifar10_example 
    mnist_example
    llm_pretrain_example
    llm_finetune_example
    llm_inference_example
    DESTINATION bin
)