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

#include <cuda_runtime.h> // `cudaMallocManaged`, `cudaFree`, `cudaSuccess`, `cudaGetErrorString`

#include "types.hpp"

namespace ashvardanian {
namespace stringzilla {
namespace cuda {

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

struct specs_t {
    size_t total_sm_count = 108;              // ? On A100
    size_t blocks_per_sm = 128;               // ? Each, generally, with 32 threads
    size_t shared_memory_per_sm = 192 * 1024; // ? On A100 it's 192 KB per SM
};

} // namespace cuda
} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_TYPES_CUH_
