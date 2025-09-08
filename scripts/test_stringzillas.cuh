/**
 *  @brief   Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test_stringzillas.cuh
 *  @author  Ash Vardanian
 */
#include <cstring> // `std::memcmp`
#include <thread>  // `std::thread::hardware_concurrency`

#define FU_ENABLE_NUMA 0
#include <fork_union.hpp> // Fork-join scoped thread pool

// #if !SZ_IS_CPP17_
// #error "This test requires C++17 or later."
// #endif

#include "test_stringzilla.hpp" // `arrow_strings_view_t`

namespace ashvardanian {
namespace stringzillas {
namespace scripts {

namespace fu = fork_union;
using namespace stringzilla;
using namespace stringzilla::scripts;

int log_environment() {
    std::printf("- Uses Haswell: %s \n", SZ_USE_HASWELL ? "yes" : "no");
    std::printf("- Uses Skylake: %s \n", SZ_USE_SKYLAKE ? "yes" : "no");
    std::printf("- Uses Ice Lake: %s \n", SZ_USE_ICE ? "yes" : "no");
    std::printf("- Uses NEON: %s \n", SZ_USE_NEON ? "yes" : "no");
    std::printf("- Uses SVE: %s \n", SZ_USE_SVE ? "yes" : "no");
    std::printf("- Uses SVE2: %s \n", SZ_USE_SVE2 ? "yes" : "no");
    std::printf("- Uses CUDA: %s \n", SZ_USE_CUDA ? "yes" : "no");
    std::printf("- Uses Kepler CUDA: %s \n", SZ_USE_KEPLER ? "yes" : "no");
    std::printf("- Uses Hopper CUDA: %s \n", SZ_USE_HOPPER ? "yes" : "no");

#if SZ_USE_CUDA
    cudaError_t cuda_error = cudaFree(0); // Force context initialization
    if (cuda_error != cudaSuccess) {
        std::printf("CUDA initialization error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    int device_count = 0;
    cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    std::printf("CUDA device count: %d\n", device_count);
    if (device_count == 0) {
        std::printf("No CUDA devices found.\n");
        return 1;
    }
    std::printf("- CUDA devices:\n");
    cudaDeviceProp prop;
    for (int i = 0; i < device_count; ++i) {
        cuda_error = cudaGetDeviceProperties(&prop, i);
        if (cuda_error != cudaSuccess) {
            std::printf("Error retrieving properties for device %d: %s\n", i, cudaGetErrorString(cuda_error));
            continue;
        }
        int warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        int shared_memory_per_warp = (warps_per_sm > 0) ? (prop.sharedMemPerMultiprocessor / warps_per_sm) : 0;
        std::printf("  - %s\n", prop.name);
        std::printf("    Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        std::printf("    Maximum Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        std::printf("    Warp Size: %d threads\n", prop.warpSize);
        std::printf("    Max Warps per SM: %d warps\n", warps_per_sm);
        std::printf("    Shared Memory per Warp: %d bytes\n", shared_memory_per_warp);
    }
    std::printf("- CUDA managed memory support: %s\n", prop.managedMemory == 1 ? "yes" : "no");
    std::printf("- CUDA unified memory support: %s\n", prop.unifiedAddressing == 1 ? "yes" : "no");
#endif
    return 0;
}

} // namespace scripts
} // namespace stringzillas
} // namespace ashvardanian
