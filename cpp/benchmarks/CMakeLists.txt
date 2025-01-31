add_executable(benchmarks)

add_subdirectory(core)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(pipelines)
add_subdirectory(t/geometry)
add_subdirectory(t/pipelines)

target_compile_definitions(benchmarks PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/test_data")
target_compile_definitions(benchmarks PRIVATE BENCHMARK_DATA_DIR="${PROJECT_SOURCE_DIR}/data/Benchmark")

target_link_libraries(benchmarks PRIVATE Open3D::Open3D)
open3d_show_and_abort_on_warning(benchmarks)
open3d_set_global_properties(benchmarks)
open3d_link_3rdparty_libraries(benchmarks)

if (BUILD_CUDA_MODULE)
    target_include_directories(benchmarks SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (WITH_IPPICV)
    target_compile_definitions(benchmarks PRIVATE IPP_CONDITIONAL_TEST_STR=) # Empty string (test not disabled)
else()
    target_compile_definitions(benchmarks PRIVATE IPP_CONDITIONAL_TEST_STR=DISABLED_)
endif()


################################################################################
#
# build benchmark dependency here so we can pass the _GLIBCXX_USE_CXX11_ABI
# definition
#

# turn off installing and testing of the benchmark lib
set(BENCHMARK_ENABLE_INSTALL  OFF CACHE BOOL "This should be OFF. Enables installing the benchmark lib")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "This should be OFF. Enables gtest framework for the benchmark lib")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "This should be OFF. Enables tests for the benchmark lib")

if(GLIBCXX_USE_CXX11_ABI)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()
add_subdirectory(${Open3D_3RDPARTY_DIR}/benchmark ${CMAKE_BINARY_DIR}/3rdparty/benchmark2)
# set the cache vars introduced by the benchmark lib as advanced to not
# clutter the cmake interfaces
mark_as_advanced(
    BENCHMARK_ENABLE_INSTALL
    BENCHMARK_ENABLE_GTEST_TESTS
    BENCHMARK_ENABLE_TESTING
    BENCHMARK_ENABLE_ASSEMBLY_TESTS
    BENCHMARK_DOWNLOAD_DEPENDENCIES
    BENCHMARK_BUILD_32_BITS
    BENCHMARK_ENABLE_EXCEPTIONS
    BENCHMARK_ENABLE_LTO
    BENCHMARK_USE_LIBCXX
)
################################################################################

target_link_libraries(benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main)

