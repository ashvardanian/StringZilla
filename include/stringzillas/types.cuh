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
#include "stringzillas/types.hpp" // `bytes_per_cell_t`, `one_byte_per_cell_k`

#include <cuda.h>         // `CUresult`, `cuLaunchKernelEx`, `cuFuncSetAttribute`, `cuGetErrorName`
#include <cuda_runtime.h> // `cudaMallocManaged`, `cudaFree`, `cudaSuccess`, `cudaGetErrorString`

#include <cub/block/block_reduce.cuh> // `cub::BlockReduce` — block collective behind our `_across_cuda_device_` reductions
#include <cub/block/block_scan.cuh> // `cub::BlockScan` — block collective behind our `exclusive_sum_across_cuda_device_`

namespace ashvardanian {
namespace stringzillas {

using namespace stringzilla;

/** @brief Minimal random-index iterator yielding `start + i` — the counting-sequence base the transform iterators and
 *         the `_across_cuda_device_` histograms/reductions index into. Only `operator[]` is used (no CUB consumers
 *         remain), so this replaces `cub::CountingInputIterator` / `cuda::counting_iterator` outright. */
template <typename value_type_>
struct counting_iterator {
    value_type_ start {};
    constexpr counting_iterator() noexcept = default;
    constexpr explicit counting_iterator(value_type_ start_value) noexcept : start(start_value) {}
    constexpr value_type_ operator[](size_t index) const noexcept { return start + static_cast<value_type_>(index); }
};

/** @brief Minimal transform iterator yielding `op(base[i])` — feeds a per-element field or derived value into a
 *         reduction / histogram without materializing it. Replaces `cub::TransformInputIterator` /
 *         `cuda::transform_iterator`. */
template <typename value_type_, typename conversion_op_, typename input_iterator_>
struct transform_input_iterator {
    input_iterator_ base {};
    conversion_op_ op {};
    constexpr transform_input_iterator() noexcept = default;
    constexpr transform_input_iterator(input_iterator_ base_iterator, conversion_op_ conversion) noexcept
        : base(base_iterator), op(conversion) {}
    constexpr value_type_ operator[](size_t index) const noexcept { return op(base[index]); }
};

/**
 *  @brief Ensures the calling thread has a current CUDA context, returning device 0's primary context (or `nullptr`).
 *
 *  The driver memory allocators (`cuMemAllocManaged`/`cuMemAlloc`/`cuMemHostAlloc`) require a current context, which -
 *  unlike the runtime's implicit lazy init behind `cudaMallocManaged` - the driver never creates on its own. Grow-only
 *  engine buffers may allocate before `cuda_executor_t::try_scheduling` retains a context, so every allocator ensures
 *  device 0's primary context here. `cuInit` + retain happen exactly once (C++ magic-static), while `cuCtxSetCurrent`
 *  is re-issued per call since the current context is per-thread and allocations may arrive on any thread.
 */
inline CUcontext ensure_primary_context_() noexcept {
    static CUcontext primary_context = []() -> CUcontext {
        if (cuInit(0) != CUDA_SUCCESS) return nullptr;
        CUdevice device = 0;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return nullptr;
        CUcontext context = nullptr;
        if (cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) return nullptr;
        return context;
    }();
    if (primary_context) cuCtxSetCurrent(primary_context);
    return primary_context;
}

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
        if (!ensure_primary_context_()) return nullptr;
        CUdeviceptr device_pointer = 0;
        CUresult error = cuMemAllocManaged(&device_pointer, n * sizeof(value_type), CU_MEM_ATTACH_GLOBAL);
        if (error != CUDA_SUCCESS) return nullptr;
        return reinterpret_cast<value_type *>(device_pointer);
    }

    void deallocate(pointer p, size_type) const noexcept {
        if (!p) return;
        cuMemFree(reinterpret_cast<CUdeviceptr>(p));
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

/** @brief Allocator over plain CUDA @b device memory via @b cuMemAlloc — not host-accessible, for buffers that live
 *         entirely on the GPU like materialized tasks and sort scratch, avoiding unified-memory page migration. */
template <typename value_type_>
struct device_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = size_t;
    using difference_type = ssize_t;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;
    template <typename other_value_type_>
    struct rebind {
        using other = device_alloc<other_value_type_>;
    };

    constexpr device_alloc() noexcept = default;
    constexpr device_alloc(device_alloc const &) noexcept = default;
    template <typename other_value_type_>
    constexpr device_alloc(device_alloc<other_value_type_> const &) noexcept {}

    value_type *allocate(size_type n) const noexcept {
        if (!ensure_primary_context_()) return nullptr;
        CUdeviceptr device_pointer = 0;
        CUresult error = cuMemAlloc(&device_pointer, n * sizeof(value_type));
        if (error != CUDA_SUCCESS) return nullptr;
        return reinterpret_cast<value_type *>(device_pointer);
    }
    void deallocate(pointer p, size_type) const noexcept {
        if (p) cuMemFree(reinterpret_cast<CUdeviceptr>(p));
    }
    template <typename other_type_>
    bool operator==(device_alloc<other_type_> const &) const noexcept {
        return true;
    }
    template <typename other_type_>
    bool operator!=(device_alloc<other_type_> const &) const noexcept {
        return false;
    }
};

using device_alloc_t = device_alloc<char>;

/** @brief Allocator over CUDA @b pinned page-locked host memory via @b cuMemHostAlloc — required for fast async
 *         @b cuMemcpyAsync staging of host-built descriptors bulk-copied to the device. */
template <typename value_type_>
struct pinned_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = size_t;
    using difference_type = ssize_t;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;
    template <typename other_value_type_>
    struct rebind {
        using other = pinned_alloc<other_value_type_>;
    };

    constexpr pinned_alloc() noexcept = default;
    constexpr pinned_alloc(pinned_alloc const &) noexcept = default;
    template <typename other_value_type_>
    constexpr pinned_alloc(pinned_alloc<other_value_type_> const &) noexcept {}

    value_type *allocate(size_type n) const noexcept {
        if (!ensure_primary_context_()) return nullptr;
        value_type *result = nullptr;
        CUresult error = cuMemHostAlloc((void **)&result, n * sizeof(value_type), 0u);
        if (error != CUDA_SUCCESS) return nullptr;
        return result;
    }
    void deallocate(pointer p, size_type) const noexcept {
        if (p) cuMemFreeHost(p);
    }
    template <typename other_type_>
    bool operator==(pinned_alloc<other_type_> const &) const noexcept {
        return true;
    }
    template <typename other_type_>
    bool operator!=(pinned_alloc<other_type_> const &) const noexcept {
        return false;
    }
};

using pinned_alloc_t = pinned_alloc<char>;

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

    /**
     *  @brief Padding that pushes `sizeof(cuda_status_t)` past 16 bytes so the SysV ABI returns it in
     *         memory (sret) rather than in the `RAX:RDX` register pair.
     *
     *  At exactly 16 bytes the struct is register-returned, and both NVCC 12.x and host g++ miscompile
     *  the leading `status`/`cuda_error` eightbyte of large translation units (the elapsed-time eightbyte
     *  survives) - the engine succeeds yet the caller reads a garbage status. Forcing the memory-return
     *  class side-steps the codegen bug for every engine without touching the call sites.
     */
    sz_u64_t reserved_ = 0;

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

/** @brief @b CUresult mirror of @ref make_cuda_status for the driver launch entry points @b cuLaunchKernelEx and
 *         @b cuFuncSetAttribute, forwarding the raw @b CUresult so callers can print @b cuGetErrorName. */
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
    // One cold, one-time startup query of ~9 device properties. The driver has no bulk `cudaGetDeviceProperties`
    // equivalent, so each field is read with its own `cuDeviceGetAttribute` (plus `cuDeviceTotalMem` for VRAM). The
    // driver never implicitly initializes, so `cuInit` (idempotent) precedes the device-handle lookup.
    CUresult init_error = cuInit(0);
    if (init_error != CUDA_SUCCESS) {
        status_t status = init_error == CUDA_ERROR_NO_DEVICE ? status_t::missing_gpu_k : status_t::unknown_k;
        return {status, cudaSuccess, init_error};
    }
    CUdevice device = 0;
    CUresult device_error = cuDeviceGet(&device, device_id);

    // Distinguish between "no GPU available" vs other CUDA errors for clearer handling upstream.
    if (device_error != CUDA_SUCCESS) {
        status_t status = status_t::unknown_k;
        if (device_error == CUDA_ERROR_NO_DEVICE) status = status_t::missing_gpu_k;
        if (device_error == CUDA_ERROR_INVALID_DEVICE) status = status_t::missing_gpu_k;
        return {status, cudaSuccess, device_error};
    }

    int multiprocessor_count = 0, warp_size = 0, major = 0, minor = 0;
    int constant_memory_bytes = 0, shared_per_multiprocessor = 0;
    int max_blocks_per_multiprocessor = 0, reserved_shared_per_block = 0;
    size_t total_global_memory = 0;
    cuDeviceGetAttribute(&multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    cuDeviceGetAttribute(&constant_memory_bytes, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);
    cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    cuDeviceGetAttribute(&shared_per_multiprocessor, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    cuDeviceGetAttribute(&max_blocks_per_multiprocessor, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, device);
    cuDeviceGetAttribute(&reserved_shared_per_block, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, device);
    cuDeviceTotalMem(&total_global_memory, device);

    // Set the GPU specs
    specs.streaming_multiprocessors = multiprocessor_count;
    specs.constant_memory_bytes = constant_memory_bytes;
    specs.vram_bytes = total_global_memory;
    specs.warp_size = warp_size;

    // Infer other global settings, that CUDA doesn't expose directly
    specs.shared_memory_bytes = static_cast<size_t>(shared_per_multiprocessor) * multiprocessor_count;
    specs.sm_code = gpu_specs_t::pack_sm_code(major, minor);
    specs.cuda_cores = gpu_specs_t::cores_per_multiprocessor(specs.sm_code) * specs.streaming_multiprocessors;

    // Scheduling-related constants
    specs.max_blocks_per_multiprocessor = max_blocks_per_multiprocessor;
    specs.reserved_memory_per_block = reserved_shared_per_block;
    return {status_t::success_k, cudaSuccess};
}

class cuda_executor_t {
    CUstream stream_ = 0;
    int device_id_ = 0;

  public:
    constexpr cuda_executor_t() noexcept = default;

    // The executor owns its `CUstream`. Copying would alias a single stream handle across instances and let
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
        // The driver never implicitly creates a context (the runtime's `cudaSetDevice` did). Initialize the driver,
        // retain the device's primary context, and make it current on this thread - the same context the runtime
        // registers `__global__` kernels into, which `cudaGetFuncBySymbol` later resolves against. The stream itself
        // is then created through the driver so the whole launch/sync path is CU*.
        CUresult init_error = cuInit(0);
        if (init_error != CUDA_SUCCESS) return make_cuda_status(init_error);
        CUdevice device = 0;
        CUresult device_error = cuDeviceGet(&device, device_id);
        if (device_error != CUDA_SUCCESS) return make_cuda_status(device_error);
        CUcontext context = nullptr;
        CUresult context_error = cuDevicePrimaryCtxRetain(&context, device);
        if (context_error != CUDA_SUCCESS) return make_cuda_status(context_error);
        CUresult current_error = cuCtxSetCurrent(context);
        if (current_error != CUDA_SUCCESS) return make_cuda_status(current_error);
        CUresult creation_error = cuStreamCreate((CUstream *)&stream_, CU_STREAM_NON_BLOCKING);
        if (creation_error != CUDA_SUCCESS) return make_cuda_status(creation_error);
        device_id_ = device_id;
        return {status_t::success_k, cudaSuccess};
    }

    explicit operator bool() const noexcept { return device_id_ >= 0; }
    inline CUstream stream() const noexcept { return stream_; }
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

    inline CUresult record_start(CUstream stream) noexcept { return cuEventRecord(start_event, (CUstream)stream); }
    inline CUresult record_stop(CUstream stream) noexcept { return cuEventRecord(stop_event, (CUstream)stream); }
    inline CUresult synchronize(CUstream stream) noexcept { return cuStreamSynchronize((CUstream)stream); }
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

/** @brief Selects the micro-tile march variant of the device-tiled scorer. `fast_k` is the store-free hot path
 *         for a full, non-corner tile (or any full tile when local); `checked_k` runs the bounds-checked path
 *         with result capture for corner / partial-edge tiles. */
enum class tile_march_t : bool { fast_k = true, checked_k = false };

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
    inline cuda_launch_t &stream(CUstream stream) noexcept {
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

#pragma region Device Wide Collective Primitives

/**
 *  @brief Driver-only replacements for `cub::Device*` host dispatch: each `__global__` template folds its grid-stride
 *         slice with a `cub::Block*` collective and merges into a pre-initialized output slot, so no device-wide
 *         temporary storage (and no "size then run" round-trip) is needed. Each host launcher fires one kernel through
 *         @ref cuda_launch_t from a resolved @ref kernel_shape_t plus the same data the CUB call took.
 */

/** @brief Block dimension of every collective primitive; the block collectives need it as a compile-time constant. */
static constexpr unsigned cuda_device_collective_threads_k = 256;

/** @brief Single-block per-segment block-max: for each @p segment in [0, segment_count) reduces
 *         @p input[segment_offsets[segment], segment_offsets[segment + 1]) into @p output[segment]. */
template <typename value_type_, typename input_iterator_, typename offset_type_>
__global__ void segmented_reduce_max_across_cuda_device_(input_iterator_ input, offset_type_ const *segment_offsets,
                                                         u32_t const *segment_count, value_type_ *output) {
    using block_reduce_t = cub::BlockReduce<value_type_, cuda_device_collective_threads_k>;
    __shared__ typename block_reduce_t::TempStorage temp_storage;
    u32_t const segments = *segment_count; // device-resident, so no host round-trip gates this launch
    for (u32_t segment = 0; segment < segments; ++segment) {
        size_t const begin = static_cast<size_t>(segment_offsets[segment]);
        size_t const end = static_cast<size_t>(segment_offsets[segment + 1]);
        value_type_ local = value_type_(0);
        for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
            value_type_ const value = static_cast<value_type_>(input[i]);
            local = local > value ? local : value;
        }
        value_type_ const block_max = block_reduce_t(temp_storage).Reduce(local, ::cuda::maximum<> {});
        if (threadIdx.x == 0) output[segment] = block_max;
        __syncthreads(); // reuse of `temp_storage` on the next segment must wait for this segment's readers
    }
}

/** @brief Single-block exclusive prefix sum of @p input[0, count) into @p output[0, count), with the inclusive total
 *         written to @p output[count] (so the segment-end array a segmented reduce needs comes for free). */
template <typename value_type_>
__global__ void exclusive_sum_across_cuda_device_(value_type_ const *input, size_t count, value_type_ *output) {
    using block_scan_t = cub::BlockScan<value_type_, cuda_device_collective_threads_k>;
    __shared__ typename block_scan_t::TempStorage temp_storage;
    value_type_ running = value_type_(0);
    for (size_t base = 0; base < count; base += blockDim.x) {
        size_t const i = base + threadIdx.x;
        value_type_ const value = i < count ? input[i] : value_type_(0);
        value_type_ exclusive = value_type_(0), block_aggregate = value_type_(0);
        block_scan_t(temp_storage).ExclusiveSum(value, exclusive, block_aggregate);
        if (i < count) output[i] = running + exclusive;
        running += block_aggregate;
        __syncthreads(); // reuse of `temp_storage` on the next tile must wait for this tile's readers
    }
    if (threadIdx.x == 0) output[count] = running;
}

/** @brief Three `u32` values reduced together — one per output field of a fused maxima pass. */
struct u32x3_t {
    u32_t a, b, c;
};

/** @brief Grid-stride block-max of a 3-field task projection: each task is loaded once and its three fields
 *         (from @p extract) folded through one `cub::BlockReduce` temp storage, then `atomicMax`-ed into the
 *         pre-zeroed 3-slot @p output — replacing three separate max-reduction passes over the
 *         same array. */
template <typename task_type_, typename extractor_>
__global__ void reduce_maxima3_across_cuda_device_(task_type_ const *tasks, size_t count, extractor_ extract,
                                                   u32_t *output) {
    using block_reduce_t = cub::BlockReduce<u32_t, cuda_device_collective_threads_k>;
    __shared__ typename block_reduce_t::TempStorage temp_storage;
    u32_t local_a = 0, local_b = 0, local_c = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        u32x3_t const value = extract(tasks[i]);
        local_a = local_a > value.a ? local_a : value.a;
        local_b = local_b > value.b ? local_b : value.b;
        local_c = local_c > value.c ? local_c : value.c;
    }
    u32_t const block_a = block_reduce_t(temp_storage).Reduce(local_a, ::cuda::maximum<> {});
    __syncthreads();
    u32_t const block_b = block_reduce_t(temp_storage).Reduce(local_b, ::cuda::maximum<> {});
    __syncthreads();
    u32_t const block_c = block_reduce_t(temp_storage).Reduce(local_c, ::cuda::maximum<> {});
    if (threadIdx.x == 0) {
        atomicMax(output + 0, block_a);
        atomicMax(output + 1, block_b);
        atomicMax(output + 2, block_c);
    }
}

/** @brief Launches @ref reduce_maxima3_across_cuda_device_ over @p count tasks, pre-zeroing the 3-slot @p output. */
template <typename task_type_, typename extractor_>
inline cuda_status_t cuda_launch_reduce_maxima3_(kernel_shape_t const &shape, task_type_ const *tasks, size_t count,
                                                 extractor_ extract, u32_t *output, CUstream stream) noexcept {
    CUresult memset_error = cuMemsetD8Async((CUdeviceptr)output, 0, 3 * sizeof(u32_t), (CUstream)stream);
    if (memset_error != CUDA_SUCCESS) return make_cuda_status(memset_error);
    task_type_ const *tasks_arg = tasks;
    size_t count_arg = count;
    extractor_ extract_arg = extract;
    u32_t *output_arg = output;
    void *args[4] = {(void *)&tasks_arg, (void *)&count_arg, (void *)&extract_arg, (void *)&output_arg};
    unsigned const blocks = static_cast<unsigned>(
        sz_min_of_two((count + cuda_device_collective_threads_k - 1) / cuda_device_collective_threads_k, size_t(1024)));
    CUresult error = cuda_launch_t {}
                         .grid(blocks ? blocks : 1u)
                         .block(cuda_device_collective_threads_k)
                         .shared(0)
                         .stream(stream)
                         .launch(shape.function, args);
    if (error != CUDA_SUCCESS) return make_cuda_status(error);
    return {status_t::success_k, cudaSuccess};
}

/** @brief Grid-stride block min-and-max of @p input[0, count) in one pass: `atomicMin` into the pre-filled (all-ones)
 *         @p out_min and `atomicMax` into the pre-zeroed @p out_max — replacing a separate min and max pass over the
 *         same iterator (and re-evaluating its transform functor once, not twice). */
template <typename value_type_, typename input_iterator_>
__global__ void reduce_minmax_across_cuda_device_(input_iterator_ input, size_t count, value_type_ *out_min,
                                                  value_type_ *out_max) {
    using block_reduce_t = cub::BlockReduce<value_type_, cuda_device_collective_threads_k>;
    __shared__ typename block_reduce_t::TempStorage temp_storage;
    value_type_ local_min = static_cast<value_type_>(~value_type_(0)), local_max = value_type_(0);
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        value_type_ const value = static_cast<value_type_>(input[i]);
        local_min = local_min < value ? local_min : value;
        local_max = local_max > value ? local_max : value;
    }
    value_type_ const block_min = block_reduce_t(temp_storage).Reduce(local_min, ::cuda::minimum<> {});
    __syncthreads();
    value_type_ const block_max = block_reduce_t(temp_storage).Reduce(local_max, ::cuda::maximum<> {});
    if (threadIdx.x == 0) {
        atomicMin(out_min, block_min);
        atomicMax(out_max, block_max);
    }
}

/** @brief Launches @ref reduce_minmax_across_cuda_device_ over @p input[0, count), pre-filling @p out_min all-ones
 *         and @p out_max zero. */
template <typename value_type_, typename input_iterator_>
inline cuda_status_t cuda_launch_reduce_minmax_(kernel_shape_t const &shape, input_iterator_ input, size_t count,
                                                value_type_ *out_min, value_type_ *out_max, CUstream stream) noexcept {
    CUresult min_memset_error = cuMemsetD8Async((CUdeviceptr)out_min, 0xFF, sizeof(value_type_), (CUstream)stream);
    if (min_memset_error != CUDA_SUCCESS) return make_cuda_status(min_memset_error);
    CUresult max_memset_error = cuMemsetD8Async((CUdeviceptr)out_max, 0, sizeof(value_type_), (CUstream)stream);
    if (max_memset_error != CUDA_SUCCESS) return make_cuda_status(max_memset_error);
    input_iterator_ input_arg = input;
    size_t count_arg = count;
    value_type_ *min_arg = out_min;
    value_type_ *max_arg = out_max;
    void *args[4] = {(void *)&input_arg, (void *)&count_arg, (void *)&min_arg, (void *)&max_arg};
    unsigned const blocks = static_cast<unsigned>(
        sz_min_of_two((count + cuda_device_collective_threads_k - 1) / cuda_device_collective_threads_k, size_t(1024)));
    CUresult error = cuda_launch_t {}
                         .grid(blocks ? blocks : 1u)
                         .block(cuda_device_collective_threads_k)
                         .shared(0)
                         .stream(stream)
                         .launch(shape.function, args);
    if (error != CUDA_SUCCESS) return make_cuda_status(error);
    return {status_t::success_k, cudaSuccess};
}

/** @brief Launches the single-block @ref segmented_reduce_max_across_cuda_device_ (writes directly, no pre-init). The
 *         segment count is read from device memory, so the launch needs no host round-trip. */
template <typename value_type_, typename input_iterator_, typename offset_type_>
inline cuda_status_t cuda_launch_segmented_reduce_max_(kernel_shape_t const &shape, input_iterator_ input,
                                                       offset_type_ const *segment_offsets, u32_t const *segment_count,
                                                       value_type_ *output, CUstream stream) noexcept {
    input_iterator_ input_arg = input;
    offset_type_ const *offsets_arg = segment_offsets;
    u32_t const *count_arg = segment_count;
    value_type_ *output_arg = output;
    void *args[4] = {(void *)&input_arg, (void *)&offsets_arg, (void *)&count_arg, (void *)&output_arg};
    CUresult error = cuda_launch_t {}
                         .grid(1u)
                         .block(cuda_device_collective_threads_k)
                         .shared(0)
                         .stream(stream)
                         .launch(shape.function, args);
    if (error != CUDA_SUCCESS) return make_cuda_status(error);
    return {status_t::success_k, cudaSuccess};
}

/** @brief Launches the single-block @ref exclusive_sum_across_cuda_device_ (writes directly, no pre-init). */
template <typename value_type_>
inline cuda_status_t cuda_launch_exclusive_sum_(kernel_shape_t const &shape, value_type_ const *input, size_t count,
                                                value_type_ *output, CUstream stream) noexcept {
    value_type_ const *input_arg = input;
    size_t count_arg = count;
    value_type_ *output_arg = output;
    void *args[3] = {(void *)&input_arg, (void *)&count_arg, (void *)&output_arg};
    CUresult error = cuda_launch_t {}
                         .grid(1u)
                         .block(cuda_device_collective_threads_k)
                         .shared(0)
                         .stream(stream)
                         .launch(shape.function, args);
    if (error != CUDA_SUCCESS) return make_cuda_status(error);
    return {status_t::success_k, cudaSuccess};
}

/** @brief Upper bound on the bucket count a @ref histogram_dense_across_cuda_device_ block-shared partial holds; the
 *         tier / launch-group consumers stay well under this. */
static constexpr u32_t cuda_dense_histogram_max_buckets_k = 64;

/** @brief Scatters each task into @p output at its bucket's running cursor (pre-set to the exclusive bucket offsets),
 *         grouping tasks by `bucket_of(task)` in one pass. Order within a bucket is unspecified — the consumers only
 *         need tasks grouped by bucket, and every task carries its own index for the result scatter. */
template <typename task_type_, typename bucket_functor_>
__global__ void scatter_tasks_by_bucket_across_cuda_device_(task_type_ const *tasks, size_t count,
                                                            bucket_functor_ bucket_of, u32_t *bucket_cursors,
                                                            task_type_ *output) {
    unsigned const lane = threadIdx.x & 31u;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        // Warp-aggregate: lanes hitting the same bucket reserve their whole run with one `atomicAdd`, then each takes
        // its slot within the run. With few buckets the cursors are highly contended, so this is the decisive win.
        u32_t const bucket = bucket_of(tasks[i]);
        unsigned const active = __activemask();
        unsigned const peers = __match_any_sync(active, bucket);
        unsigned const leader = static_cast<unsigned>(__ffs(static_cast<int>(peers)) - 1);
        u32_t base = 0;
        if (lane == leader) base = atomicAdd(&bucket_cursors[bucket], static_cast<u32_t>(__popc(peers)));
        base = __shfl_sync(active, base, leader);
        u32_t const rank = static_cast<u32_t>(__popc(peers & ((1u << lane) - 1u)));
        output[base + rank] = tasks[i];
    }
}

/** @brief Launches @ref scatter_tasks_by_bucket_across_cuda_device_ over @p count tasks into @p output. */
template <typename task_type_, typename bucket_functor_>
inline cuda_status_t cuda_launch_scatter_tasks_by_bucket_(kernel_shape_t const &shape, task_type_ const *tasks,
                                                          size_t count, bucket_functor_ bucket_of,
                                                          u32_t *bucket_cursors, task_type_ *output,
                                                          CUstream stream) noexcept {
    task_type_ const *tasks_arg = tasks;
    size_t count_arg = count;
    bucket_functor_ bucket_arg = bucket_of;
    u32_t *cursors_arg = bucket_cursors;
    task_type_ *output_arg = output;
    void *args[5] = {(void *)&tasks_arg, (void *)&count_arg, (void *)&bucket_arg, (void *)&cursors_arg,
                     (void *)&output_arg};
    unsigned const blocks = static_cast<unsigned>(
        sz_min_of_two((count + cuda_device_collective_threads_k - 1) / cuda_device_collective_threads_k, size_t(1024)));
    CUresult error = cuda_launch_t {}
                         .grid(blocks ? blocks : 1u)
                         .block(cuda_device_collective_threads_k)
                         .shared(0)
                         .stream(stream)
                         .launch(shape.function, args);
    if (error != CUDA_SUCCESS) return make_cuda_status(error);
    return {status_t::success_k, cudaSuccess};
}

/** @brief Per-bucket histogram of `bucket_of(task)` over @p count tasks into the pre-zeroed @p bucket_counts. Like
 *         @ref histogram_dense_across_cuda_device_ but for a runtime (dynamic-shared) @p bucket_count beyond the static
 *         @ref cuda_dense_histogram_max_buckets_k cap: each block accumulates into a shared partial, then flushes one
 *         atomic per non-empty bin — so the N global atomics collapse to `blocks * non_empty_bins`. Pairs with
 *         @ref scatter_tasks_by_bucket_across_cuda_device_. Launched with `sharedMemBytes = bucket_count * 4`. */
template <typename task_type_, typename bucket_functor_>
__global__ void histogram_tasks_by_bucket_across_cuda_device_(task_type_ const *tasks, size_t count,
                                                              bucket_functor_ bucket_of, u32_t bucket_count,
                                                              u32_t *bucket_counts) {
    extern __shared__ u32_t partial[];
    for (u32_t b = threadIdx.x; b < bucket_count; b += blockDim.x) partial[b] = 0u;
    __syncthreads();
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
        atomicAdd(&partial[bucket_of(tasks[i])], 1u);
    __syncthreads();
    for (u32_t b = threadIdx.x; b < bucket_count; b += blockDim.x)
        if (partial[b]) atomicAdd(&bucket_counts[b], partial[b]);
}

/** @brief One single-threaded pass over a counting-sort histogram that both exclusive-scans every bucket into
 *         @p bucket_cursors (the scatter's per-bucket offsets) and compacts the non-empty buckets into dense ascending
 *         run descriptors — for each non-zero bucket it appends (unique key = bucket id, run length = count, begin
 *         offset = running total), then writes the trailing end at `[groups]` and the group count. Fuses the exclusive
 *         sum and the compaction the counting sort would otherwise run as two launches. */
template <typename size_type_>
__global__ void scan_and_compact_bucket_runs_across_cuda_device_(u32_t const *bucket_counts, u32_t num_buckets,
                                                                 u32_t *bucket_cursors, u32_t *unique_keys,
                                                                 size_type_ *run_lengths, size_type_ *begin_offsets,
                                                                 u32_t *group_count_out) {
    if (blockIdx.x || threadIdx.x) return;
    u32_t groups = 0;
    size_type_ running = 0;
    for (u32_t bucket = 0; bucket < num_buckets; ++bucket) {
        bucket_cursors[bucket] = static_cast<u32_t>(running); // exclusive offset for the scatter (every bucket)
        u32_t const count = bucket_counts[bucket];
        if (count) {
            unique_keys[groups] = bucket;
            run_lengths[groups] = count;
            begin_offsets[groups] = running;
            ++groups;
        }
        running += count;
    }
    bucket_cursors[num_buckets] = static_cast<u32_t>(running);
    begin_offsets[groups] = running; // trailing segment end for the per-group segmented Max
    *group_count_out = groups;
}

/** @brief Launches @ref histogram_tasks_by_bucket_across_cuda_device_ over @p count tasks, pre-zeroing @p bucket_counts. */
template <typename task_type_, typename bucket_functor_>
inline cuda_status_t cuda_launch_histogram_tasks_by_bucket_(kernel_shape_t const &shape, task_type_ const *tasks,
                                                            size_t count, bucket_functor_ bucket_of, u32_t num_buckets,
                                                            u32_t *bucket_counts, CUstream stream) noexcept {
    CUresult memset_error = cuMemsetD8Async((CUdeviceptr)bucket_counts, 0,
                                            static_cast<size_t>(num_buckets) * sizeof(u32_t), (CUstream)stream);
    if (memset_error != CUDA_SUCCESS) return make_cuda_status(memset_error);
    task_type_ const *tasks_arg = tasks;
    size_t count_arg = count;
    bucket_functor_ bucket_arg = bucket_of;
    u32_t bucket_count_arg = num_buckets;
    u32_t *counts_arg = bucket_counts;
    void *args[5] = {(void *)&tasks_arg, (void *)&count_arg, (void *)&bucket_arg, (void *)&bucket_count_arg,
                     (void *)&counts_arg};
    unsigned const blocks = static_cast<unsigned>(
        sz_min_of_two((count + cuda_device_collective_threads_k - 1) / cuda_device_collective_threads_k, size_t(1024)));
    CUresult error = cuda_launch_t {}
                         .grid(blocks ? blocks : 1u)
                         .block(cuda_device_collective_threads_k)
                         .shared(num_buckets * static_cast<unsigned>(sizeof(u32_t)))
                         .stream(stream)
                         .launch(shape.function, args);
    if (error != CUDA_SUCCESS) return make_cuda_status(error);
    return {status_t::success_k, cudaSuccess};
}

/** @brief Launches the single-threaded @ref scan_and_compact_bucket_runs_across_cuda_device_. */
template <typename size_type_>
inline cuda_status_t cuda_launch_scan_and_compact_bucket_runs_(kernel_shape_t const &shape, u32_t const *bucket_counts,
                                                               u32_t num_buckets, u32_t *bucket_cursors,
                                                               u32_t *unique_keys, size_type_ *run_lengths,
                                                               size_type_ *begin_offsets, u32_t *group_count_out,
                                                               CUstream stream) noexcept {
    u32_t const *counts_arg = bucket_counts;
    u32_t num_buckets_arg = num_buckets;
    u32_t *cursors_arg = bucket_cursors;
    u32_t *keys_arg = unique_keys;
    size_type_ *lengths_arg = run_lengths;
    size_type_ *offsets_arg = begin_offsets;
    u32_t *group_count_arg = group_count_out;
    void *args[7] = {(void *)&counts_arg,  (void *)&num_buckets_arg, (void *)&cursors_arg,    (void *)&keys_arg,
                     (void *)&lengths_arg, (void *)&offsets_arg,     (void *)&group_count_arg};
    CUresult error = cuda_launch_t {}.grid(1u).block(1u).shared(0).stream(stream).launch(shape.function, args);
    if (error != CUDA_SUCCESS) return make_cuda_status(error);
    return {status_t::success_k, cudaSuccess};
}

/** @brief Small dense histogram: counts each element's bucket id (from @p buckets, any iterator yielding an id in
 *         `[0, bucket_count)`) into the pre-zeroed @p out through a block-shared partial, so the few global bins take
 *         one atomic per block, not per element. Gives a consumer its per-tier / per-group counts straight from the
 *         reordered tasks. */
template <typename input_iterator_>
__global__ void histogram_dense_across_cuda_device_(input_iterator_ buckets, size_t count, u32_t bucket_count,
                                                    u32_t *out) {
    __shared__ u32_t partial[cuda_dense_histogram_max_buckets_k];
    for (u32_t b = threadIdx.x; b < bucket_count; b += blockDim.x) partial[b] = 0;
    __syncthreads();
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
        atomicAdd(&partial[static_cast<u32_t>(buckets[i])], 1u);
    __syncthreads();
    for (u32_t b = threadIdx.x; b < bucket_count; b += blockDim.x) atomicAdd(&out[b], partial[b]);
}

/** @brief Launches @ref histogram_dense_across_cuda_device_ over @p count elements, pre-zeroing @p out[0, bucket_count). */
template <typename input_iterator_>
inline cuda_status_t cuda_launch_histogram_dense_(kernel_shape_t const &shape, input_iterator_ buckets, size_t count,
                                                  u32_t bucket_count, u32_t *out, CUstream stream) noexcept {
    CUresult memset_error = cuMemsetD8Async((CUdeviceptr)out, 0, static_cast<size_t>(bucket_count) * sizeof(u32_t),
                                            (CUstream)stream);
    if (memset_error != CUDA_SUCCESS) return make_cuda_status(memset_error);
    input_iterator_ buckets_arg = buckets;
    size_t count_arg = count;
    u32_t bucket_count_arg = bucket_count;
    u32_t *out_arg = out;
    void *args[4] = {(void *)&buckets_arg, (void *)&count_arg, (void *)&bucket_count_arg, (void *)&out_arg};
    unsigned const blocks = static_cast<unsigned>(
        sz_min_of_two((count + cuda_device_collective_threads_k - 1) / cuda_device_collective_threads_k, size_t(1024)));
    CUresult error = cuda_launch_t {}
                         .grid(blocks ? blocks : 1u)
                         .block(cuda_device_collective_threads_k)
                         .shared(0)
                         .stream(stream)
                         .launch(shape.function, args);
    if (error != CUDA_SUCCESS) return make_cuda_status(error);
    return {status_t::success_k, cudaSuccess};
}

#pragma endregion Device Wide Collective Primitives

/**
 *  @brief Loads 32 bits from an unaligned address using the well known @b `prmt` trick.
 *  @see https://stackoverflow.com/a/40198552/2766161
 */
SZ_DEVICE_INLINE u32_vec_t sz_u32_load_unaligned(void const *ptr) noexcept {
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

/** @brief Host-readable launch-shaping descriptor of one warp-tier group, run-length-encoded from the sorted
 *         warp tasks, so the host never dereferences a device task to drive a warp-tier launch. */
struct warp_tasks_group_descriptor_t {
    /** @brief DP cell-width tier shared by every task in the group (selects the warp kernel family). */
    bytes_per_cell_t bytes_per_cell = one_byte_per_cell_k;
    /** @brief Warps-per-multiprocessor tier shared by the group (post-merge launch density). */
    warp_tasks_density_t density = one_warp_per_multiprocessor_k;
    /** @brief Absolute offset of the group's first task into the original task array. */
    size_t begin_offset = 0;
    /** @brief Number of tasks in the group. */
    size_t count = 0;
    /** @brief Largest `memory_requirement` across the group (sizes the dynamic shared memory of the launch). */
    size_t max_memory_requirement = 0;
};

/** @brief Reads one warp task's `memory_requirement` for the per-group @ref segmented_reduce_max_across_cuda_device_. */
template <typename task_type_>
struct warp_tasks_memory_requirement_functor_ {
    __host__ SZ_DEVICE_INLINE size_t operator()(task_type_ const &task) const noexcept {
        return task.memory_requirement;
    }
};

/** @brief Classifies one task into the contiguous split bucket the counting sort lays out: device-cooperative (0),
 *         warp-level (1), or empty (2). Ascending buckets give the `[device | warp | empty]` layout directly. */
template <typename task_type_>
struct warp_group_tier_functor_ {
    constexpr u32_t operator()(task_type_ const &task) const noexcept {
        if (task.density == warps_working_together_k) return 0u;
        if (task.density == infinite_warps_per_multiprocessor_k) return 2u;
        return 1u;
    }
};

/** @brief `log2` of a power-of-two value (small inputs only; returns 0 for 0/1). */
static constexpr u32_t log2_pow2_(u32_t value) noexcept {
    u32_t bit = 0;
    while (value > 1u) {
        value >>= 1;
        ++bit;
    }
    return bit;
}

/** @brief Packs a warp task's @p bytes_per_cell and @p density into a dense 5-bit counting-sort key — 2 bits for
 *         the kernel family, 3 for the density tier — and implicitly converts back to that key for the sort. */
struct warp_group_key_t {
    static constexpr u32_t bytes_bits_k = 2;
    static constexpr u32_t density_bits_k = 3;
    static constexpr u32_t density_mask_k = (1u << density_bits_k) - 1u;

    u32_t packed = 0;

    constexpr warp_group_key_t() = default;
    constexpr explicit warp_group_key_t(u32_t packed_key) noexcept : packed(packed_key) {}
    constexpr warp_group_key_t(bytes_per_cell_t bytes_per_cell, warp_tasks_density_t density) noexcept
        : packed((log2_pow2_(static_cast<u32_t>(bytes_per_cell)) << density_bits_k) |
                 log2_pow2_(static_cast<u32_t>(density))) {}

    constexpr bytes_per_cell_t bytes_per_cell() const noexcept {
        return static_cast<bytes_per_cell_t>(1u << (packed >> density_bits_k));
    }
    constexpr warp_tasks_density_t density() const noexcept {
        return static_cast<warp_tasks_density_t>(1u << (packed & density_mask_k));
    }
    constexpr operator u32_t() const noexcept { return packed; }
};

/** @brief Reads one warp task's dense @ref warp_group_key_t for the counting sort's histogram/scatter. */
template <typename task_type_>
struct warp_group_key_functor_ {
    constexpr u32_t operator()(task_type_ const &task) const noexcept {
        return warp_group_key_t {task.bytes_per_cell, task.density};
    }
};

/** @brief Bucket count for the dense warp launch-group key: 2 bits `bytes_per_cell` | 3 bits `density` → 32. */
static constexpr u32_t warp_group_key_buckets_k = 1u << (warp_group_key_t::bytes_bits_k +
                                                         warp_group_key_t::density_bits_k);

/**
 *  @brief Reorders warp tasks into contiguous launch groups and describes each group for the host to launch.
 *
 *  Each warp ideally gets its own task, with enough shared memory for several warps per multiprocessor. Two task
 *  fields drive the scheduling:
 *
 *  @p bytes_per_cell selects the kernel template and is a hard group boundary, since distinct kernels do not run
 *  concurrently — even on an H200 with 140 GB of VRAM, @b cudaDeviceProp::concurrentKernels reports 1.
 *
 *  @p density is the max warps per multiprocessor, inferred from the device shared memory minus the per-block
 *  reserved memory. Addressing more than 48 KB per block needs a per-kernel
 *  @b cudaFuncAttributeMaxDynamicSharedMemorySize override, and @b cuFuncSetAttribute is synchronous — so we keep
 *  the launches few to stay asynchronous.
 *
 *  @p specs.streaming_multiprocessors sets the sensible lower bound per launch: on an H100 that is 132, so a group
 *  admitting 2 warps per multiprocessor but fewer than 66 blocks is better merged with a more memory-hungry group
 *  than launched on its own.
 *
 *  @post @p tasks are reordered in-place into [device-level | warp-level, sorted | empty], and @p group_count
 *  host-readable @ref warp_tasks_group_descriptor_t entries are written to @p group_descriptors — each a warp-tier
 *  group's absolute task subrange and launch-shaping fields, so the host launches from the descriptors alone,
 *  never by dereferencing a device task.
 */
template <typename task_type_, typename scratch_buffer_type_, typename task_buffer_type_, typename count_buffer_type_,
          typename key_buffer_type_, typename size_buffer_type_, typename descriptor_buffer_type_>
warp_tasks_groups<task_type_> warp_tasks_grouping( //
    span<task_type_> tasks, gpu_specs_t const &specs, CUstream stream,
    scratch_buffer_type_ &grouping_scratch, // grow-only device bucket histogram + cursors (bytes)
    task_buffer_type_ &partition_scratch,   // grow-only task ping-pong for the counting-sort scatters
    count_buffer_type_ &counts_scratch,     // device-accessible tier split counts + cursors
    key_buffer_type_ &group_keys_scratch,   // device-accessible unique composite keys + run count
    size_buffer_type_ &group_sizes_scratch, // device-accessible run lengths, begin offsets, per-group max memory
    descriptor_buffer_type_ &group_descriptors, size_t &group_count, kernel_shape_t const &segmented_reduce_max_shape,
    kernel_shape_t const &tier_histogram_shape, kernel_shape_t const &tier_scatter_shape,
    kernel_shape_t const &exclusive_sum_u32_shape, kernel_shape_t const &warp_group_histogram_shape,
    kernel_shape_t const &warp_group_scatter_shape, kernel_shape_t const &scan_compact_shape) noexcept {

    using task_t = task_type_;
    warp_tasks_groups<task_t> result;
    group_count = 0;

    size_t const total_tasks = tasks.size();
    result.device_level_tasks = {tasks.begin(), tasks.begin()};
    result.warp_level_tasks = {tasks.begin(), tasks.begin()};
    result.empty_tasks = {tasks.begin(), tasks.end()};
    if (!total_tasks) return result;

    // Split the array into [device-level | warp-level | empty] with a 3-bucket counting sort of the tier classifier:
    // histogram the three tiers, exclusive-scan them into cursors, scatter the tasks into the contiguous layout, and
    // copy it back into `tasks` (the scatter can't run in place). Only the two leading counts cross back to the host.
    if (partition_scratch.try_resize_uninitialized(total_tasks) == status_t::bad_alloc_k) return result;
    if (counts_scratch.try_resize_uninitialized(7) == status_t::bad_alloc_k) return result; // 3 counts + 4 cursors
    task_t *const partition_buffer = partition_scratch.data();
    u32_t *const tier_counts = counts_scratch.data();
    u32_t *const tier_cursors = tier_counts + 3; // exclusive offsets + the scan's trailing total

    warp_group_tier_functor_<task_t> const tier_of {};
    transform_input_iterator<u32_t, warp_group_tier_functor_<task_t>, task_t const *> tier_bucket_iterator(tasks.data(),
                                                                                                           tier_of);
    if (cuda_launch_histogram_dense_(tier_histogram_shape, tier_bucket_iterator, total_tasks, 3u, tier_counts, stream)
            .status != status_t::success_k)
        return result;
    if (cuda_launch_exclusive_sum_(exclusive_sum_u32_shape, tier_counts, 3u, tier_cursors, stream).status !=
        status_t::success_k)
        return result;
    if (cuda_launch_scatter_tasks_by_bucket_(tier_scatter_shape, tasks.data(), total_tasks, tier_of, tier_cursors,
                                             partition_buffer, stream)
            .status != status_t::success_k)
        return result;
    if (cuMemcpyDtoDAsync((CUdeviceptr)tasks.data(), (CUdeviceptr)partition_buffer, total_tasks * sizeof(task_t),
                          (CUstream)stream) != CUDA_SUCCESS)
        return result;
    if (cuStreamSynchronize((CUstream)stream) != CUDA_SUCCESS) return result;
    size_t const device_level_count = tier_counts[0];
    size_t const warp_level_count = tier_counts[1];

    auto const warp_tasks_begin = tasks.begin() + device_level_count;
    auto const warp_tasks_end = warp_tasks_begin + warp_level_count;
    result.device_level_tasks = {tasks.begin(), warp_tasks_begin};
    result.warp_level_tasks = {warp_tasks_begin, warp_tasks_end};
    result.empty_tasks = {warp_tasks_end, tasks.end()};

    if (!warp_level_count) return result;

    // Dense-key counting sort of the warp middle. Histogram the `(log2 bytes, log2 density)` groups, then in one pass
    // exclusive-scan them into scatter cursors AND compact the non-empty buckets into ascending run descriptors; scatter
    // the tasks into contiguous launch groups and copy the layout back (the scatter can't run in place). A segmented Max
    // of `memory_requirement` — sized by the device-resident group count, so no sync gates it — gives each group's
    // dynamic-shared-memory footprint. One synchronize drains the whole stage.
    if (group_keys_scratch.try_resize_uninitialized(warp_level_count + 1) == status_t::bad_alloc_k) return result;
    if (group_sizes_scratch.try_resize_uninitialized(3 * warp_level_count + 1) == status_t::bad_alloc_k) return result;
    if (grouping_scratch.try_resize_uninitialized((2u * warp_group_key_buckets_k + 1) * sizeof(u32_t)) ==
        status_t::bad_alloc_k)
        return result;
    u32_t *const unique_keys = group_keys_scratch.data();
    u32_t *const run_count_out = unique_keys + warp_level_count;
    size_t *const run_lengths = group_sizes_scratch.data();
    size_t *const begin_offsets = run_lengths + warp_level_count; // `warp_level_count + 1` slots (trailing end)
    size_t *const group_max_memory = begin_offsets + (warp_level_count + 1);
    u32_t *const group_bucket_counts = reinterpret_cast<u32_t *>(grouping_scratch.data());
    u32_t *const group_bucket_cursors = group_bucket_counts + warp_group_key_buckets_k;
    task_t *const warp_tasks_ptr = tasks.data() + device_level_count;

    warp_group_key_functor_<task_t> const group_key_of {};
    if (cuda_launch_histogram_tasks_by_bucket_(warp_group_histogram_shape, warp_tasks_ptr, warp_level_count,
                                               group_key_of, warp_group_key_buckets_k, group_bucket_counts, stream)
            .status != status_t::success_k)
        return result;
    if (cuda_launch_scan_and_compact_bucket_runs_(scan_compact_shape, group_bucket_counts, warp_group_key_buckets_k,
                                                  group_bucket_cursors, unique_keys, run_lengths, begin_offsets,
                                                  run_count_out, stream)
            .status != status_t::success_k)
        return result;
    if (cuda_launch_scatter_tasks_by_bucket_(warp_group_scatter_shape, warp_tasks_ptr, warp_level_count, group_key_of,
                                             group_bucket_cursors, partition_buffer, stream)
            .status != status_t::success_k)
        return result;
    if (cuMemcpyDtoDAsync((CUdeviceptr)warp_tasks_ptr, (CUdeviceptr)partition_buffer, warp_level_count * sizeof(task_t),
                          (CUstream)stream) != CUDA_SUCCESS)
        return result;
    {
        warp_tasks_memory_requirement_functor_<task_t> const memory_functor {};
        transform_input_iterator<size_t, warp_tasks_memory_requirement_functor_<task_t>, task_t const *>
            memory_iterator(warp_tasks_begin, memory_functor);
        cuda_status_t const reduce_status = cuda_launch_segmented_reduce_max_(
            segmented_reduce_max_shape, memory_iterator, begin_offsets, run_count_out, group_max_memory, stream);
        if (reduce_status.status != status_t::success_k) return result;
    }
    if (cuStreamSynchronize((CUstream)stream) != CUDA_SUCCESS) return result;
    size_t const raw_group_count = *run_count_out;
    if (!raw_group_count) return result;

    if (group_descriptors.try_resize_uninitialized(raw_group_count) == status_t::bad_alloc_k) return result;
    warp_tasks_group_descriptor_t *const descriptors = group_descriptors.data();

    // The naive next step would be to simply launch one kernel per raw group, but our high-level goal isn't maximum
    // utilization, it's the fastest end-to-end time. Accounting for scheduling & synchronization costs, we greedily
    // merge consecutive same-`bytes_per_cell` groups (adopting the sparser, more-memory-hungry density of the group
    // we merge into) until the running group has enough tasks to keep every multiprocessor's warps busy. This is the
    // device-resident equivalent of the former host scan that sparsened individual task densities in place.
    size_t emitted = 0;
    size_t raw_index = 0;
    while (raw_index < raw_group_count) {
        warp_group_key_t const head_key {unique_keys[raw_index]};
        warp_tasks_group_descriptor_t descriptor;
        descriptor.bytes_per_cell = head_key.bytes_per_cell();
        descriptor.density = head_key.density();
        descriptor.begin_offset = device_level_count + begin_offsets[raw_index];
        descriptor.count = run_lengths[raw_index];
        descriptor.max_memory_requirement = group_max_memory[raw_index];

        size_t next = raw_index + 1;
        while (next < raw_group_count) {
            size_t const possible_warps = size_t(descriptor.density) * specs.streaming_multiprocessors;
            if (descriptor.count > possible_warps) break; // enough tasks to saturate the device

            warp_group_key_t const next_key {unique_keys[next]};
            if (next_key.bytes_per_cell() != descriptor.bytes_per_cell) break; // different kernel family, can't merge

            // Merge the next group in: adopt its (sparser) density, grow the span, take the larger memory footprint.
            descriptor.density = next_key.density();
            descriptor.count += run_lengths[next];
            if (group_max_memory[next] > descriptor.max_memory_requirement)
                descriptor.max_memory_requirement = group_max_memory[next];
            ++next;
        }
        descriptors[emitted++] = descriptor;
        raw_index = next;
    }
    group_count = emitted;
    return result;
}

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_TYPES_CUH_
