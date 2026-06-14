/**
 *  @brief Shared definitions for the StringZilla CUDA library.
 *  @file include/stringzillas/types.cuh
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

#include <cuda.h>         // `CUresult`, `cuLaunchKernelEx`, `cuFuncSetAttribute`, `cuGetErrorName`
#include <cuda_runtime.h> // `cudaMallocManaged`, `cudaFree`, `cudaSuccess`, `cudaGetErrorString`

#include <optional>  // `std::optional`
#include <algorithm> // `std::sort`, `std::partition`

namespace ashvardanian {
namespace stringzillas {

using namespace stringzilla;

/**
 *  @brief A custom allocator that uses CUDA Unified Memory for allocation.
 */
template <typename value_type_>
struct unified_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = size_t;
    using difference_type = ssize_t;

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
    // Driver query: `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` collapses both device and managed/unified memory onto
    // `CU_MEMORYTYPE_DEVICE` - exactly the two the runtime path accepted (`cudaMemoryTypeDevice`/`Managed`) - and
    // returns an error for unregistered host pointers, so a successful `CU_MEMORYTYPE_DEVICE` is the accept
    // decision and everything else (host memory, unregistered pointers, errors) is rejected.
    CUmemorytype memory_type = static_cast<CUmemorytype>(0);
    CUresult error = cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)ptr);
    if (error != CUDA_SUCCESS) return false;
    return memory_type == CU_MEMORYTYPE_DEVICE;
}

struct cuda_status_t {
    status_t status = status_t::success_k;
    cudaError_t cuda_error = cudaSuccess;
    CUresult driver_error = CUDA_SUCCESS;
    float elapsed_milliseconds = 0.0;

    inline operator status_t() const noexcept { return status; }
};

/**
 *  @brief Maps a CUDA runtime error into a first-class `cuda_status_t`, always carrying the `cudaError_t`.
 *
 *  Launch-configuration faults (too much shared memory, oversized grid, bad block geometry, etc.) used to
 *  collapse into an opaque `status_t::unknown_k`, which made misconfigured launches indistinguishable from
 *  genuine internal failures. We classify the common configuration faults into actionable statuses, while
 *  forwarding the raw `cudaError_t` so the caller can always print `cudaGetErrorName`/`cudaGetErrorString`.
 */
inline cuda_status_t make_cuda_status(cudaError_t cuda_error) noexcept {
    if (cuda_error == cudaSuccess) return {status_t::success_k, cudaSuccess};
    status_t status = status_t::unknown_k;
    switch (cuda_error) {
        // Out-of-resource faults map onto allocation failures: the launch asked for more registers,
        // shared memory, or device memory than the multiprocessor (or the device) can provide.
    case cudaErrorMemoryAllocation:
    case cudaErrorLaunchOutOfResources:
        status = status_t::bad_alloc_k;
        break;
        // Bad launch geometry (grid/block dimensions, shared-memory size, cooperative co-residency) is a
        // dimension/configuration problem, distinct from a generic unknown failure.
    case cudaErrorInvalidValue:
    case cudaErrorInvalidConfiguration: status = status_t::unexpected_dimensions_k; break;
    default: status = status_t::unknown_k; break;
    }
    return {status, cuda_error};
}

/**
 *  @brief Maps a CUDA @b driver error (`CUresult`) into a first-class `cuda_status_t`, carrying the `CUresult`.
 *
 *  The driver launch entry points (`cuLaunchKernelEx`, `cuFuncSetAttribute`) return `CUresult` rather than the
 *  runtime `cudaError_t`. We classify the common launch-configuration faults into actionable statuses, mirroring
 *  the `make_cuda_status(cudaError_t)` mapping, while forwarding the raw `CUresult` so the caller can always
 *  print `cuGetErrorName`/`cuGetErrorString`.
 */
inline cuda_status_t make_cuda_status(CUresult driver_error) noexcept {
    if (driver_error == CUDA_SUCCESS) return {status_t::success_k, cudaSuccess, CUDA_SUCCESS};
    status_t status = status_t::unknown_k;
    switch (driver_error) {
        // An out-of-memory fault maps onto an allocation failure, just like the runtime path.
    case CUDA_ERROR_OUT_OF_MEMORY:
        status = status_t::bad_alloc_k;
        break;
        // Bad launch geometry (oversized grid/block, shared-memory overflow, malformed image) is a
        // dimension/configuration problem, distinct from a generic unknown failure.
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    case CUDA_ERROR_INVALID_VALUE:
    case CUDA_ERROR_INVALID_IMAGE: status = status_t::unexpected_dimensions_k; break;
    default: status = status_t::unknown_k; break;
    }
    return {status, cudaSuccess, driver_error};
}

inline cuda_status_t gpu_specs_fetch(gpu_specs_t &specs, int device_id = 0) noexcept {
    // Runtime API by design: this is a one-time startup bulk query of ~9 device properties. The driver
    // equivalent is one `cuDeviceGetAttribute` per field for no benefit - the runtime (`cudart`) is already a
    // hard dependency via `cudaGetFuncBySymbol` (the only bridge from a registered `__global__` to a
    // `CUfunction`), so the whole launch/sync hot path is CU* while these cold setup calls stay runtime.
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

    // The executor owns its `cudaStream_t`. Copying would alias a single stream handle across instances and let
    // one copy's destructor tear down a stream the others still reference, so the executor is move-only: callers
    // hold one owner and hand it to the engines by `const &`.
    cuda_executor_t(cuda_executor_t const &) = delete;
    cuda_executor_t &operator=(cuda_executor_t const &) = delete;
    cuda_executor_t(cuda_executor_t &&other) noexcept : stream_(other.stream_), device_id_(other.device_id_) {
        other.stream_ = 0;
        other.device_id_ = 0;
    }
    cuda_executor_t &operator=(cuda_executor_t &&other) noexcept {
        if (this != &other) {
            if (stream_) cuStreamDestroy((CUstream)stream_);
            stream_ = other.stream_;
            device_id_ = other.device_id_;
            other.stream_ = 0;
            other.device_id_ = 0;
        }
        return *this;
    }
    ~cuda_executor_t() noexcept {
        if (stream_) cuStreamDestroy((CUstream)stream_);
    }

    cuda_status_t try_scheduling(int device_id) noexcept {
        device_id_ = -1; // ? Invalid device ID
        // `cudaSetDevice` selects the device and makes its primary context current on this thread - the same
        // context the runtime registers `__global__` kernels into, which `cudaGetFuncBySymbol` later resolves
        // against. The stream itself is then created through the driver so the whole launch/sync path is CU*.
        cudaError_t switching_error = cudaSetDevice(device_id);
        if (switching_error != cudaSuccess) return {status_t::unknown_k, switching_error};
        CUresult creation_error = cuStreamCreate((CUstream *)&stream_, CU_STREAM_NON_BLOCKING);
        if (creation_error != CUDA_SUCCESS) return make_cuda_status(creation_error);
        device_id_ = device_id;
        return {status_t::success_k, cudaSuccess};
    }

    explicit operator bool() const noexcept { return device_id_ >= 0; }
    inline cudaStream_t stream() const noexcept { return stream_; }
    inline int device_id() const noexcept { return device_id_; }
};

/**
 *  @brief Pair of CUDA driver events timing a launch; created once on first use and reused across calls.
 *
 *  Events need a current context, so they are created lazily inside the first `operator()` (via `ensure_created`)
 *  rather than at construction. `CU_EVENT_BLOCKING_SYNC` lets the drain sleep the host thread instead of spinning.
 */
struct cuda_timer_t {
    CUevent start_event = nullptr;
    CUevent stop_event = nullptr;

    cuda_timer_t() noexcept = default;
    ~cuda_timer_t() noexcept {
        if (start_event) cuEventDestroy(start_event);
        if (stop_event) cuEventDestroy(stop_event);
    }
    cuda_timer_t(cuda_timer_t const &) = delete;
    cuda_timer_t &operator=(cuda_timer_t const &) = delete;
    cuda_timer_t(cuda_timer_t &&other) noexcept : start_event(other.start_event), stop_event(other.stop_event) {
        other.start_event = nullptr, other.stop_event = nullptr;
    }
    cuda_timer_t &operator=(cuda_timer_t &&other) noexcept {
        if (this != &other) {
            if (start_event) cuEventDestroy(start_event);
            if (stop_event) cuEventDestroy(stop_event);
            start_event = other.start_event, stop_event = other.stop_event;
            other.start_event = nullptr, other.stop_event = nullptr;
        }
        return *this;
    }

    /** @brief Creates the two events on first use; a no-op on every later call. */
    inline CUresult ensure_created() noexcept {
        if (start_event) return CUDA_SUCCESS;
        CUresult start_error = cuEventCreate(&start_event, CU_EVENT_BLOCKING_SYNC);
        if (start_error != CUDA_SUCCESS) return start_error;
        return cuEventCreate(&stop_event, CU_EVENT_BLOCKING_SYNC);
    }

    inline CUresult record_start(cudaStream_t stream) noexcept { return cuEventRecord(start_event, (CUstream)stream); }
    inline CUresult record_stop(cudaStream_t stream) noexcept { return cuEventRecord(stop_event, (CUstream)stream); }
    inline CUresult synchronize(cudaStream_t stream) noexcept { return cuStreamSynchronize((CUstream)stream); }
    inline float elapsed_milliseconds() noexcept {
        float milliseconds = 0;
        cuEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }
};

/**
 *  @brief Sizes a grid for an already-resolved `CUfunction` via the driver occupancy query.
 *
 *  The warp tier's shared memory varies with the data, so its grid is sized per launch even though the handle was
 *  resolved up front; the driver query takes the `CUfunction` directly (the runtime query needs the host symbol).
 */
inline cuda_status_t occupancy_grid_for(unsigned &blocks_per_grid, CUfunction function, unsigned threads_per_block,
                                        unsigned shared_memory_bytes, gpu_specs_t const &specs) noexcept {
    int blocks_per_multiprocessor = 0;
    CUresult occupancy_error = cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, function,
                                                                           static_cast<int>(threads_per_block),
                                                                           static_cast<size_t>(shared_memory_bytes));
    if (occupancy_error != CUDA_SUCCESS) return make_cuda_status(occupancy_error);
    if (blocks_per_multiprocessor < 1) blocks_per_multiprocessor = 1;
    blocks_per_grid = static_cast<unsigned>(blocks_per_multiprocessor) * specs.streaming_multiprocessors;
    return {status_t::success_k, cudaSuccess};
}

/**
 *  @brief A resolved launch shape: a kernel's driver handle plus its co-resident block count.
 *
 *  Bridging a runtime `__global__` symbol to a `CUfunction`, raising its shared-memory ceiling, and querying its
 *  occupancy are invariant for a given (capability, kernel-shape, device). Each engine resolves its kernels once
 *  into a `kernel_table` and reads the handles back on every later launch - mirroring `sz_dispatch_table_*`.
 */
struct kernel_shape_t {
    CUfunction function = nullptr;
    unsigned blocks_per_multiprocessor = 0; // ? 0 when the grid depends on per-launch shared memory (warp tier)
};

/**
 *  @brief Resolves one kernel symbol into a `kernel_shape_t`: its `CUfunction`, an optionally-raised shared-memory
 *         ceiling, and - when @p precompute_occupancy - its co-resident block count for fixed-shape kernels.
 */
inline cuda_status_t resolve_kernel_shape(kernel_shape_t &shape, void const *kernel_symbol, unsigned threads_per_block,
                                          unsigned shared_memory_ceiling, bool precompute_occupancy) noexcept {
    cudaFunction_t function = nullptr;
    // Runtime-only bridge: maps a runtime-registered `__global__` host symbol to a `CUfunction`. The driver API
    // has no equivalent for runtime-compiled/fatbin-registered kernels, so the rest of the path stays CU*.
    cudaError_t function_error = cudaGetFuncBySymbol(&function, kernel_symbol);
    if (function_error != cudaSuccess) return make_cuda_status(function_error);
    shape.function = function;
    if (shared_memory_ceiling) {
        // The opt-in limit covers static + dynamic shared memory, so the dynamic ceiling we may raise is the
        // device opt-in maximum minus the kernel's own static shared usage (e.g. a substitution-cost table).
        int static_shared = 0;
        cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, (CUfunction)function);
        unsigned const dynamic_ceiling = shared_memory_ceiling > (unsigned)static_shared
                                             ? shared_memory_ceiling - (unsigned)static_shared
                                             : 0u;
        CUresult attribute_error = cuFuncSetAttribute((CUfunction)function,
                                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_ceiling);
        if (attribute_error != CUDA_SUCCESS) return make_cuda_status(attribute_error);
    }
    if (precompute_occupancy) {
        int blocks_per_multiprocessor = 0;
        // The driver query takes the already-resolved `CUfunction`, matching the rest of the launch path.
        CUresult occupancy_error = cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_multiprocessor, (CUfunction)function, static_cast<int>(threads_per_block),
            static_cast<size_t>(shared_memory_ceiling));
        if (occupancy_error != CUDA_SUCCESS) return make_cuda_status(occupancy_error);
        if (blocks_per_multiprocessor < 1) blocks_per_multiprocessor = 1;
        shape.blocks_per_multiprocessor = static_cast<unsigned>(blocks_per_multiprocessor);
    }
    return {status_t::success_k, cudaSuccess};
}

/**
 *  @brief Fluent builder over `CUlaunchConfig` + `cuLaunchKernelEx`, optionally cooperative.
 *
 *  Collapses the verbose `CUlaunchConfig{}` field-by-field setup (and the separate `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`
 *  attribute block) into a single expression. The cooperative attribute lives inside the builder, so the temporary
 *  must outlive the `launch()` call - which it does, since both happen within the same full expression.
 */
struct cuda_launch_t {
    CUlaunchConfig config_ {};
    CUlaunchAttribute attributes_[2] {}; // up to two of {cooperative, cluster-dimension}; must outlive `launch()`
    unsigned num_attributes_ = 0;

    cuda_launch_t() noexcept {
        config_.gridDimY = config_.gridDimZ = 1;
        config_.blockDimY = config_.blockDimZ = 1;
    }
    inline cuda_launch_t &grid(unsigned blocks) noexcept {
        config_.gridDimX = blocks;
        return *this;
    }
    inline cuda_launch_t &grid(unsigned blocks_x, unsigned blocks_y) noexcept {
        config_.gridDimX = blocks_x;
        config_.gridDimY = blocks_y; // cross-pair batching: `blockIdx.y` selects one (shorter, longer) pair
        return *this;
    }
    inline cuda_launch_t &block(unsigned threads) noexcept {
        config_.blockDimX = threads;
        return *this;
    }
    inline cuda_launch_t &shared(unsigned bytes) noexcept {
        config_.sharedMemBytes = bytes;
        return *this;
    }
    inline cuda_launch_t &stream(cudaStream_t stream) noexcept {
        config_.hStream = (CUstream)stream;
        return *this;
    }
    inline cuda_launch_t &cooperative() noexcept {
        CUlaunchAttribute &a = attributes_[num_attributes_++];
        a.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        a.value.cooperative = 1;
        config_.attrs = attributes_, config_.numAttrs = num_attributes_;
        return *this;
    }
    /** @brief Hopper thread-block clusters: `gridDimX` must be a multiple of `x` (each cluster spans `x` blocks). */
    inline cuda_launch_t &cluster(unsigned x, unsigned y = 1, unsigned z = 1) noexcept {
        CUlaunchAttribute &a = attributes_[num_attributes_++];
        a.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        a.value.clusterDim.x = x, a.value.clusterDim.y = y, a.value.clusterDim.z = z;
        config_.attrs = attributes_, config_.numAttrs = num_attributes_;
        return *this;
    }
    inline CUresult launch(cudaFunction_t function, void **arguments) noexcept {
        return cuLaunchKernelEx(&config_, function, arguments, nullptr);
    }
};

/**
 *  @brief Loads 32 bits from an unaligned address using the well known @b `prmt` trick.
 *  @see https://stackoverflow.com/a/40198552/2766161
 */
__forceinline__ __device__ u32_vec_t sz_u32_load_unaligned(void const *ptr) noexcept {
    // In reality we load 64 bits, and then, with `.f4e`, we forward-extract
    // four consecutive bytes into a 32-bit register.
    u32_vec_t result;
    asm("{\n\t"                                                    //
        "   .reg .b64    aligned_ptr;\n\t"                         //
        "   .reg .b32    low, high, alignment;\n\t"                //
        "   and.b64      aligned_ptr, %1, 0xfffffffffffffffc;\n\t" //
        "   ld.u32       low, [aligned_ptr];\n\t"                  //
        "   ld.u32       high, [aligned_ptr+4];\n\t"               //
        "   cvt.u32.u64  alignment, %1;\n\t"                       //
        "   prmt.b32.f4e %0, low, high, alignment;\n\t"            //
        "}"                                                        //
        : "=r"(result.u32)                                         //
        : "l"(ptr));
    return result;
}

/** @brief Number of threads per warp on the GPU. */
enum warp_size_t : unsigned {
    warp_size_nvidia_k = 32, // ? NVIDIA GPUs use 32 threads per warp
    warp_size_amd_k = 64,    // ? AMD GPUs use 64 threads per wave
};

/**
 *  @brief Defines the upper bound on the number of warps per multi processor we may theoretically
 *         be able to run as part of one or many blocks. Generally this number depends on the amount
 *         of shared memory available on the device, and the amount of reserved memory per block.
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
 *  @brief Multiple warps can run concurrently on the same multiprocessor, which helps hide the latency
 *         of memory operations. It only happens, if we have enough shared memory, so we may want to reduce
 *         the density of the tasks proportional to the current GPU's speculative factor.
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
    size_t const device_level_tasks = //
        std::partition(tasks.begin(), tasks.end(),
                       [](task_t const &task) { return task.density == warps_working_together_k; }) -
        tasks.begin();

    // Determine the number of empty tasks and put them aside.
    auto const warp_tasks_begin = tasks.begin() + device_level_tasks;
    size_t const non_empty_tasks = //
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
    // `tasks_remaining` counts only the WARP-level tasks; `first_task_index` is then an ABSOLUTE index into `tasks`,
    // offset past the device-level tasks that occupy the front of the array. (The earlier form subtracted
    // `device_level_tasks` from the warp count and omitted the offset, which is correct only when there are no
    // device-level tasks - otherwise it underflows to `SIZE_MAX` and spins forever the moment a device task exists.)
    size_t tasks_remaining = result.warp_level_tasks.size();
    while (tasks_remaining > 1) { // 1 task or less ~ nothing to merge
        size_t const first_task_index = device_level_tasks + (result.warp_level_tasks.size() - tasks_remaining);
        task_t &indicative_task = tasks[first_task_index];
        size_t const tasks_with_same_density = //
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
