/**
 *  @brief  Shared definitions for the StringZilla CUDA library.
 *  @file   types.cuh
 *  @author Ash Vardanian
 *
 *  The goal for this header is to provide absolutely-minimal set of types and forward-declarations for
 *  CUDA backends of higher-level complex templated algorithms implemented outside of the C layer, like:
 *
 *  - `unified_alloc` - a custom allocator that uses CUDA Unified Memory for allocation.
 *  - `gpu_specs_t` - a structure that contains the GPU specifications, like number of SMs, VRAM size, etc.
 *  - `cuda_status_t` - a composite of the CUDA status, error code, and elapsed kernel execution time.
 */
#ifndef STRINGZILLAS_TYPES_CUH_
#define STRINGZILLAS_TYPES_CUH_

#include "stringzilla/types.hpp"

#include <cuda_runtime.h> // `cudaMallocManaged`, `cudaFree`, `cudaSuccess`, `cudaGetErrorString`

#include <optional>  // `std::optional`
#include <algorithm> // `std::sort`, `std::partition`

namespace ashvardanian {
namespace stringzillas {

using namespace stringzilla;

/**
 *  @brief  A custom allocator that uses CUDA Unified Memory for allocation.
 */
template <typename value_type_>
struct unified_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = sz_size_t;
    using difference_type = sz_ssize_t;

    /*  Those are needed for compatibility with our custom containers.
     *  @see https://en.cppreference.com/w/cpp/memory/allocator_traits
     */
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;

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

using unified_alloc_t = unified_alloc<char>;

/** @brief Returns `true` if the pointer refers to device-accessible memory (Device or Managed/Unified). */
inline bool is_device_accessible_memory(void const *ptr) noexcept {
    if (!ptr) return true;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) return false;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 10000) // Modern CUDA: use `type`
    return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
#else // Legacy CUDA: `memoryType` and `isManaged`
    return attr.memoryType == cudaMemoryTypeDevice || attr.isManaged;
#endif
}

struct cuda_status_t {
    status_t status = status_t::success_k;
    cudaError_t cuda_error = cudaSuccess;
    float elapsed_milliseconds = 0.0;

    inline operator status_t() const noexcept { return status; }
};

inline cuda_status_t gpu_specs_fetch(gpu_specs_t &specs, int device_id = 0) noexcept {
    cudaDeviceProp prop;
    cudaError_t cuda_error = cudaGetDeviceProperties(&prop, device_id);

    // Distinguish between "no GPU available" vs other CUDA errors for clearer handling upstream.
    if (cuda_error != cudaSuccess) {
        status_t status = status_t::unknown_k;
        if (cuda_error == cudaErrorNoDevice) status = status_t::missing_gpu_k;
        if (cuda_error == cudaErrorInvalidDevice) status = status_t::missing_gpu_k;
        if (cuda_error == cudaErrorInsufficientDriver) status = status_t::missing_gpu_k;
        return {status, cuda_error};
    }

    // Set the GPU specs
    specs.streaming_multiprocessors = prop.multiProcessorCount;
    specs.constant_memory_bytes = prop.totalConstMem;
    specs.vram_bytes = prop.totalGlobalMem;
    specs.warp_size = prop.warpSize;

    // Infer other global settings, that CUDA doesn't expose directly
    specs.shared_memory_bytes = prop.sharedMemPerMultiprocessor * prop.multiProcessorCount;
    specs.sm_code = gpu_specs_t::pack_sm_code(prop.major, prop.minor);
    specs.cuda_cores = gpu_specs_t::cores_per_multiprocessor(specs.sm_code) * specs.streaming_multiprocessors;

    // Scheduling-related constants
    specs.max_blocks_per_multiprocessor = prop.maxBlocksPerMultiProcessor;
    specs.reserved_memory_per_block = prop.reservedSharedMemPerBlock;
    return {status_t::success_k, cudaSuccess};
}

class cuda_executor_t {
    cudaStream_t stream_ = 0;
    int device_id_ = 0;

  public:
    constexpr cuda_executor_t() noexcept = default;
    constexpr cuda_executor_t(cuda_executor_t const &) noexcept = default;
    constexpr cuda_executor_t &operator=(cuda_executor_t const &) noexcept = default;

    cuda_status_t try_scheduling(int device_id) noexcept {
        device_id_ = -1; // ? Invalid device ID
        cudaError_t switching_error = cudaSetDevice(device_id);
        if (switching_error != cudaSuccess) return {status_t::unknown_k, switching_error};
        cudaError_t creation_error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (creation_error != cudaSuccess) return {status_t::unknown_k, creation_error};
        device_id_ = device_id;
        return {status_t::success_k, cudaSuccess};
    }

    explicit operator bool() const noexcept { return device_id_ >= 0; }
    inline cudaStream_t stream() const noexcept { return stream_; }
    inline int device_id() const noexcept { return device_id_; }
};

/**
 *  @brief  Loads 32 bits from an unaligned address using the well known @b `prmt` trick.
 *  @see    https://stackoverflow.com/a/40198552/2766161
 */
__forceinline__ __device__ sz_u32_vec_t sz_u32_load_unaligned(void const *ptr) noexcept {
    // In reality we load 64 bits, and then, with `.f4e`, we forward-extract
    // four consecutive bytes into a 32-bit register.
    sz_u32_vec_t result;
    asm("{\n\t"
        "   .reg .b64    aligned_ptr;\n\t"
        "   .reg .b32    low, high, alignment;\n\t"
        "   and.b64      aligned_ptr, %1, 0xfffffffffffffffc;\n\t"
        "   ld.u32       low, [aligned_ptr];\n\t"
        "   ld.u32       high, [aligned_ptr+4];\n\t"
        "   cvt.u32.u64  alignment, %1;\n\t"
        "   prmt.b32.f4e %0, low, high, alignment;\n\t"
        "}"
        : "=r"(result.u32)
        : "l"(ptr));
    return result;
}

/** @brief Number of threads per warp on the GPU. */
enum warp_size_t : unsigned {
    warp_size_nvidia_k = 32, // ? NVIDIA GPUs use 32 threads per warp
    warp_size_amd_k = 64,    // ? AMD GPUs use 64 threads per wave
};

/**
 *  @brief  Defines the upper bound on the number of warps per multi processor we may theoretically
 *          be able to run as part of one or many blocks. Generally this number depends on the amount
 *          of shared memory available on the device, and the amount of reserved memory per block.
 */
enum warp_tasks_density_t : unsigned {
    warps_working_together_k = 0,
    one_warp_per_multiprocessor_k = 1,
    two_warps_per_multiprocessor_k = 2,
    four_warps_per_multiprocessor_k = 4,
    eight_warps_per_multiprocessor_k = 8,
    sixteen_warps_per_multiprocessor_k = 16,
    thirty_two_warps_per_multiprocessor_k = 32,
    sixty_four_warps_per_multiprocessor_k = 64,
    infinite_warps_per_multiprocessor_k = 0xFFFFFFFF
};

inline warp_tasks_density_t warp_tasks_density(size_t task_memory_requirement, gpu_specs_t const &specs) noexcept {
    std::initializer_list<warp_tasks_density_t> densities {
        sixty_four_warps_per_multiprocessor_k, thirty_two_warps_per_multiprocessor_k,
        sixteen_warps_per_multiprocessor_k,    eight_warps_per_multiprocessor_k,
        four_warps_per_multiprocessor_k,       two_warps_per_multiprocessor_k,
        one_warp_per_multiprocessor_k,
    };
    if (task_memory_requirement == 0) return infinite_warps_per_multiprocessor_k;
    for (auto density : densities) {
        if (density > specs.max_blocks_per_multiprocessor) continue;
        size_t required_block_memory = task_memory_requirement * density + specs.reserved_memory_per_block * density;
        if (required_block_memory < specs.shared_memory_per_multiprocessor()) return density;
    }
    return warps_working_together_k;
}

struct speculative_warp_tasks_density_t {
    warp_tasks_density_t density = warps_working_together_k;
    size_t speculative_factor = 0;
};

/**
 *  @brief  Multiple warps can run concurrently on the same multiprocessor, which helps hide the latency
 *          of memory operations. It only happens, if we have enough shared memory, so we may want to reduce
 *          the density of the tasks proportional to the current GPU's speculative factor.
 */
inline speculative_warp_tasks_density_t speculation_friendly_density(warp_tasks_density_t maximum_density) noexcept {
    // if (maximum_density >= 16) return {static_cast<warp_tasks_density_t>(maximum_density / 16), 16};
    // if (maximum_density >= 8) return {static_cast<warp_tasks_density_t>(maximum_density / 8), 8};
    if (maximum_density >= 4) return {static_cast<warp_tasks_density_t>(maximum_density / 4), 4};
    if (maximum_density >= 2) return {static_cast<warp_tasks_density_t>(maximum_density / 2), 2};
    return {maximum_density, 1};
}

template <typename task_type_>
struct warp_tasks_groups {
    span<task_type_> device_level_tasks;
    span<task_type_> warp_level_tasks;
    span<task_type_> empty_tasks;
};

/**
 *  Let's say you have a list of GPU tasks of similar nature, but all of them require different amount of
 *  shared memory for efficiency. Ideally, we want each warp to receive it's own task (independent of the
 *  others), and have enough shared memory for more than one warp per multiprocessor.
 *
 *  To optimize the scheduling we will look into several factors of individual tasks.
 *
 *  @p `tasks[i].bytes_per_cell` - defines the actual kernel template to be used and serves as the natural boundary,
 *  as different kernels *kind of* can't run concurrently.
 *
 *      Theoretically, that isn't true. On paper, several CUDA kernels can run concurrently on the same physical device.
 *      It makes sense, assuming the Multi-Instance GPU (MIG) virtualization feature on them. But even when pulling
 *      the `cudaGetDeviceProperties` and logging @b `cudaDeviceProp::concurrentKernels` for the H200 GPU with 140 GB
 *      of VRAM, it states 1. 🤯
 *
 *  @p `tasks[i].density` - max number of warps per multiprocessor, executing simultaneously. It's inferred from the
 *  shared memory amount, supported by device, which is of the order of 100-200 KB, minus the reserved memory per block.
 *
 *      Theoretically, it's true, but in practice, to address more than 48 KB of shared memory from a single block,
 *      you need to override that @b `cudaFuncAttributeMaxDynamicSharedMemorySize` setting for each individual kernel,
 *      also subtracting the reserved memory for the expected number of blocks. 🤯
 *      Moreover, calling @b `cudaFuncSetAttribute` is a synchronous operation, so say goodbye to asynchronous kernel
 *      launches and latency hiding. So to reduce the overall runtime, we need to have as few individual launches
 *      as possible.
 *
 *  @p `specs.shared_memory_per_multiprocessor()` and @p `specs.reserved_memory_per_block` - our memory limits.
 *  @p `specs.streaming_multiprocessors` - the actual number of physical multiprocessors on the device. It defines
 *  the reasonable lower bound for a single kernel launch. On H100 it would be 132, so if a particular group of inputs
 *  can have 2 simultaneous warps per multiprocessor, but has fewer than (132 / 2 = 66) blocks, it would be better to
 *  merge it with a more memory-hungry group to avoid the overhead of additional kernel launches.
 *
 *  It's possible to have a very memory-efficient task, that allows for 32 warps per multiprocessor, and it would be
 *  best schedule them in the largest possible block size - 1024 threads.
 *
 *  @throws `std::bad_alloc` if the `std::sort` or `std::partition` fails to allocate memory.
 *
 *  @post The @p tasks are sorted in-place, first containing the returned number of device-wide tasks,
 *  and then the warp-wide tasks grouped by ( @p bytes_per_cell, @p density ) pairs. Use @b `group_by`
 *  to navigate the output.
 */
template <typename task_type_>
warp_tasks_groups<task_type_> warp_tasks_grouping(span<task_type_> tasks, gpu_specs_t const &specs) noexcept(false) {

    using task_t = task_type_;
    warp_tasks_groups<task_t> result;

    // Determine if there are tasks that require the whole device memory.
    size_t const device_level_tasks =
        std::partition(tasks.begin(), tasks.end(),
                       [](task_t const &task) { return task.density == warps_working_together_k; }) -
        tasks.begin();

    // Determine the number of empty tasks and put them aside.
    auto const warp_tasks_begin = tasks.begin() + device_level_tasks;
    size_t const non_empty_tasks =
        std::partition(warp_tasks_begin, tasks.end(),
                       [](task_t const &task) { return task.density != infinite_warps_per_multiprocessor_k; }) -
        warp_tasks_begin;

    // The remaining tasks will be sorted from smallest memory consumption to largest.
    auto const warp_tasks_end = warp_tasks_begin + non_empty_tasks;
    std::sort(warp_tasks_begin, warp_tasks_end, [](task_t const &lhs, task_t const &rhs) {
        return lhs.bytes_per_cell == rhs.bytes_per_cell ? lhs.density < rhs.density
                                                        : lhs.bytes_per_cell < rhs.bytes_per_cell;
    });

    result.device_level_tasks = {tasks.begin(), warp_tasks_begin};
    result.warp_level_tasks = {warp_tasks_begin, warp_tasks_end};
    result.empty_tasks = {warp_tasks_end, tasks.end()};

    // The naive next step would be to simply group them by their memory requirements,
    // but we our high-level goal isn't maximum utilization of the GPU, but rather
    // the fastest execution time. And assuming the scheduling & synchronization
    // costs, we may want to combine consecutive groups of tasks to ensure they are large enough.
    size_t tasks_remaining = result.warp_level_tasks.size() - device_level_tasks;
    while (tasks_remaining > 1) { // 1 task or less ~ nothing to merge
        size_t const first_task_index = result.warp_level_tasks.size() - tasks_remaining;
        task_t &indicative_task = tasks[first_task_index];
        size_t const tasks_with_same_density =
            std::find_if(&indicative_task, result.warp_level_tasks.end(),
                         [&](task_t const &task) {
                             return task.bytes_per_cell != indicative_task.bytes_per_cell ||
                                    task.density != indicative_task.density;
                         }) -
            &indicative_task;
        size_t const following_tasks = tasks_remaining - tasks_with_same_density;
        if (!following_tasks) break; // No more tasks to merge

        // Check if we have enough tasks to keep all the warps busy.
        size_t const possible_warps = size_t(indicative_task.density) * specs.streaming_multiprocessors;
        if (tasks_with_same_density > possible_warps) {
            tasks_remaining -= tasks_with_same_density;
            continue; // Jump to the next group
        }

        // If the next task has a different cell size, we can't merge them.
        task_t &next_indicative_task = tasks[first_task_index + tasks_with_same_density];
        if (indicative_task.bytes_per_cell != next_indicative_task.bytes_per_cell) {
            tasks_remaining -= tasks_with_same_density;
            continue; // Jump to the next group
        }

        // Update all the operations in the current group to have the same "sparser" density
        // as the next group.
        for (size_t i = 0; i < tasks_with_same_density; ++i) {
            task_t &task = tasks[first_task_index + i];
            task.density = next_indicative_task.density;
        }
    }

    return result;
}

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_TYPES_CUH_
