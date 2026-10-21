cmake_minimum_required(VERSION 3.16)
project(tensor_alloc VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build microbenchmarks" ON)
option(BUILD_EXAMPLES "Build example applications" ON)

# Compiler warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(ENABLE_SANITIZERS)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Core library source files
set(CORE_SOURCES
    src/request_validation.cpp
    src/size_class.cpp
    src/predictor_fallback.cpp
)

# Optional sources added as stages are completed
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/metrics.cpp")
    list(APPEND CORE_SOURCES src/metrics.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/allocation_registry.cpp")
    list(APPEND CORE_SOURCES src/allocation_registry.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_backend.cpp")
    list(APPEND CORE_SOURCES src/cpu_backend.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/trace_recorder.cpp")
    list(APPEND CORE_SOURCES src/trace_recorder.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/api.cpp")
    list(APPEND CORE_SOURCES src/api.cpp)
endif()

add_library(tensor_alloc_lib STATIC ${CORE_SOURCES})

target_include_directories(tensor_alloc_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Testing setup
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_size_class tests/test_size_class.cpp)
    target_link_libraries(test_size_class PRIVATE tensor_alloc_lib)
    add_test(NAME test_size_class COMMAND test_size_class)

    add_executable(test_validation tests/test_validation.cpp)
    target_link_libraries(test_validation PRIVATE tensor_alloc_lib)
    add_test(NAME test_validation COMMAND test_validation)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_api.cpp")
        add_executable(test_api tests/test_api.cpp)
        target_link_libraries(test_api PRIVATE tensor_alloc_lib)
        add_test(NAME test_api COMMAND test_api)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_alignment.cpp")
        add_executable(test_alignment tests/test_alignment.cpp)
        target_link_libraries(test_alignment PRIVATE tensor_alloc_lib)
        add_test(NAME test_alignment COMMAND test_alignment)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_reuse.cpp")
        add_executable(test_reuse tests/test_reuse.cpp)
        target_link_libraries(test_reuse PRIVATE tensor_alloc_lib)
        add_test(NAME test_reuse COMMAND test_reuse)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_invalid_inputs.cpp")
        add_executable(test_invalid_inputs tests/test_invalid_inputs.cpp)
        target_link_libraries(test_invalid_inputs PRIVATE tensor_alloc_lib)
        add_test(NAME test_invalid_inputs COMMAND test_invalid_inputs)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_multithreading.cpp")
        add_executable(test_multithreading tests/test_multithreading.cpp)
        target_link_libraries(test_multithreading PRIVATE tensor_alloc_lib)
        add_test(NAME test_multithreading COMMAND test_multithreading)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_trace_recorder.cpp")
        add_executable(test_trace_recorder tests/test_trace_recorder.cpp)
        target_link_libraries(test_trace_recorder PRIVATE tensor_alloc_lib)
        add_test(NAME test_trace_recorder COMMAND test_trace_recorder)
    endif()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_trace_pipeline.py")
        add_test(NAME test_trace_pipeline COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_trace_pipeline.py)
    endif()
endif()

# Benchmarks setup
if(BUILD_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/allocator_microbench.cpp")
        add_executable(allocator_microbench benchmarks/allocator_microbench.cpp)
        target_link_libraries(allocator_microbench PRIVATE tensor_alloc_lib)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/replay_trace.cpp")
        add_executable(replay_trace benchmarks/replay_trace.cpp)
        target_link_libraries(replay_trace PRIVATE tensor_alloc_lib)
    endif()
endif()

# Examples setup
if(BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_usage.cpp")
        add_executable(basic_usage examples/basic_usage.cpp)
        target_link_libraries(basic_usage PRIVATE tensor_alloc_lib)
    endif()
endif()
