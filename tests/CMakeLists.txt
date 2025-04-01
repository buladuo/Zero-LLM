# Find Catch2
find_package(Catch2 REQUIRED)

# Core tests
add_executable(test_core core/test_tensor.cpp)
target_link_libraries(test_core PRIVATE zero-llm Catch2::Catch2)
add_test(NAME test_core COMMAND test_core)

# Layers tests
add_executable(test_layers 
    layers/test_attention.cpp
    layers/test_linear.cpp
)
target_link_libraries(test_layers PRIVATE zero-llm Catch2::Catch2)
add_test(NAME test_layers COMMAND test_layers)

# LLM tests
add_executable(test_llm
    llm/test_tokenizer.cpp
    llm/test_transformer.cpp
)
target_link_libraries(test_llm PRIVATE zero-llm Catch2::Catch2)
add_test(NAME test_llm COMMAND test_llm)