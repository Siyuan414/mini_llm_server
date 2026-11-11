cmake_minimum_required(VERSION 3.18)
project(llm_server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Platform detection: Jetson (aarch64) vs desktop (x86_64) ─────────────────
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_CUDA_ARCHITECTURES 87)          # Jetson Orin
    set(TRT_LIB_DIR /usr/lib/aarch64-linux-gnu)
    set(TRT_INC_DIR /usr/include/aarch64-linux-gnu)
else()
    set(CMAKE_CUDA_ARCHITECTURES 120)         # RTX 5070 Ti (Blackwell)
    set(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
    set(TRT_INC_DIR /usr/include/x86_64-linux-gnu)
endif()

find_package(CUDAToolkit REQUIRED)

# ── runtime: single-request engine (Stage 1) ─────────────────────────────────
add_executable(runtime
    src/runtime.cpp
)
set_source_files_properties(src/runtime.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(runtime PRIVATE ${TRT_INC_DIR}
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/kernels)
target_link_directories(runtime PRIVATE ${TRT_LIB_DIR})
target_link_libraries(runtime
    nvinfer
    CUDA::cudart
    CUDA::cublas
)

# ── batch_runtime: continuous batching (Stage 2A) ────────────────────────────
add_executable(batch_runtime
    src/batch_runtime.cpp
)
set_source_files_properties(src/batch_runtime.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(batch_runtime PRIVATE ${TRT_INC_DIR}
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/kernels)
target_link_directories(batch_runtime PRIVATE ${TRT_LIB_DIR})
target_link_libraries(batch_runtime
    nvinfer
    CUDA::cudart
    CUDA::cublas
)

# ── paged_runtime: continuous batching on the paged cache (Stage 2B) ─────────
# No TensorRT: the forward pass is decode_layer.h + our kernels.
add_executable(paged_runtime
    src/paged_runtime.cpp
)
set_source_files_properties(src/paged_runtime.cpp PROPERTIES LANGUAGE CUDA)
target_compile_options(paged_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(paged_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/kernels)
target_link_libraries(paged_runtime CUDA::cudart CUDA::cublas)

# ── serve: HTTP front-end over the paged engine (Stage 2, S2.4) ──────────────
add_executable(serve
    src/serve.cpp
)
set_source_files_properties(src/serve.cpp PROPERTIES LANGUAGE CUDA)
target_compile_options(serve PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(serve PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/kernels)
target_link_libraries(serve CUDA::cudart CUDA::cublas)
find_package(Threads REQUIRED)
target_link_libraries(serve Threads::Threads)

# ── serve_chunked: chunked prefill over the varlen kernel (S3.4) ──────────────
add_executable(serve_chunked
    src/serve_chunked.cpp
)
set_source_files_properties(src/serve_chunked.cpp PROPERTIES LANGUAGE CUDA)
target_compile_options(serve_chunked PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(serve_chunked PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/kernels)
target_link_libraries(serve_chunked CUDA::cudart CUDA::cublas Threads::Threads)

# Future targets (uncomment as stages progress):
# add_executable(engine_builder tools/engine_builder.cpp)   # S1.1
# add_executable(bench tools/bench.cpp)                     # S1.6
# add_executable(server src/server.cpp ...)                 # Stage 2
