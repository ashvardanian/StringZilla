/**
 *  @brief  Shared definitions for the StringZilla CUDA library.
 *  @file   types.cuh
 *  @author Ash Vardanian
 *
 *  The goal for this header is to provide absolutely-minimal set of types and forward-declarations for
 *  CUDA backends of higher-level complex templated algorithms implemented outside of the C layer, like:
 *
 *  - `unified_alloc` - a custom allocator that uses CUDA Unified Memory for allocation.
 */
#ifndef STRINGZILLA_TYPES_CUH_
#define STRINGZILLA_TYPES_CUH_

#include "stringzilla/types.hpp"

#include <cuda_runtime.h> // `cudaMallocManaged`, `cudaFree`, `cudaSuccess`, `cudaGetErrorString`
#include <optional>       // `std::optional`

#if !defined(SZ_USE_HOPPER)
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 11)
#define SZ_USE_HOPPER (1)
#else
#define SZ_USE_HOPPER (0)
#endif
#endif

#if !defined(SZ_USE_KEPLER)
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 3)
#define SZ_USE_KEPLER (1)
#else
#define SZ_USE_KEPLER (0)
#endif
#endif

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief  A custom allocator that uses CUDA Unified Memory for allocation.
 */
template <typename value_type_>
struct unified_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = sz_size_t;
    using difference_type = sz_ssize_t;

    template <typename other_value_type_>
    struct rebind {
        using other = unified_alloc<other_value_type_>;
    };

    constexpr unified_alloc() noexcept = default;
    constexpr unified_alloc(unified_alloc const &) noexcept = default;

    template <typename other_value_type_>
    constexpr unified_alloc(unified_alloc<other_value_type_> const &) noexcept {}

    value_type *allocate(size_type n) const noexcept {
        value_type *result = nullptr;
        auto error = cudaMallocManaged((value_type **)&result, n * sizeof(value_type));
        if (error != cudaSuccess) return nullptr;
        return result;
    }

    void deallocate(pointer p, size_type) const noexcept {
        if (!p) return;
        cudaFree(p);
    }

    template <typename other_type_>
    bool operator==(unified_alloc<other_type_> const &) const noexcept {
        return true;
    }

    template <typename other_type_>
    bool operator!=(unified_alloc<other_type_> const &) const noexcept {
        return false;
    }
};

inline std::optional<gpu_specs_t> gpu_specs(int device = 0) noexcept {
    gpu_specs_t specs;
    cudaDeviceProp prop;
    cudaError_t cuda_error = cudaGetDeviceProperties(&prop, device);
    if (cuda_error != cudaSuccess) return std::nullopt; // ! Failed to get device properties

    // Set the GPU specs
    specs.streaming_multiprocessors = prop.multiProcessorCount;
    specs.constant_memory_bytes = prop.totalConstMem;
    specs.vram_bytes = prop.totalGlobalMem;

    // Infer other global settings, that CUDA doesn't expose directly
    specs.shared_memory_bytes = prop.sharedMemPerMultiprocessor * prop.multiProcessorCount;
    specs.cuda_cores = gpu_specs_t::cores_per_multiprocessor(prop.major, prop.minor) * specs.streaming_multiprocessors;

    // Scheduling-related constants
    specs.max_blocks_per_multiprocessor = prop.maxBlocksPerMultiProcessor;
    specs.reserved_memory_per_block = prop.reservedSharedMemPerBlock;
    return specs;
}

struct cuda_status_t {
    status_t status = status_t::success_k;
    cudaError_t cuda_error = cudaSuccess;
    float elapsed_milliseconds = 0.0;

    inline operator status_t() const noexcept { return status; }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_TYPES_CUH_
