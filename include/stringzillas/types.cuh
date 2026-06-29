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

#include <optional>  // `std::optional`
#include <algorithm> // `std::sort`

#include <cub/device/device_merge_sort.cuh> // `cub::DeviceMergeSort` — moves the O(n log n) grouping sort onto the GPU
#include <cub/device/device_partition.cuh>  // `cub::DevicePartition` — device/warp/empty tier split
#include <cub/device/device_run_length_encode.cuh> // `cub::DeviceRunLengthEncode` — warp-tier group run lengths
#include <cub/device/device_scan.cuh>              // `cub::DeviceScan` — group begin offsets via exclusive sum
#include <cub/device/device_segmented_reduce.cuh>  // `cub::DeviceSegmentedReduce` — per-group max memory requirement

// CUB 3+ (CCCL 3.x / CUDA 13+) moved iterator types from cub:: to cuda::
#if CUB_MAJOR_VERSION < 3
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#else
#include <cuda/iterator>
#endif

namespace ashvardanian {
namespace stringzillas {

using namespace stringzilla;

// Project-local iterator aliases so the rest of the codebase is CUB-version-agnostic.
#if CUB_MAJOR_VERSION < 3
template <typename value_type_>
using counting_iterator = cub::CountingInputIterator<value_type_>;
template <typename value_type_, typename conversion_op_, typename input_iterator_>
using transform_input_iterator = cub::TransformInputIterator<value_type_, conversion_op_, input_iterator_>;
#else
template <typename value_type_>
using counting_iterator = ::cuda::counting_iterator<value_type_>;
template <typename value_type_, typename conversion_op_, typename input_iterator_>
using transform_input_iterator = ::cuda::transform_iterator<conversion_op_, input_iterator_>;
#endif

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

/**
 *  @brief A custom allocator that uses plain CUDA @b device memory (`cudaMalloc`) - not host-accessible.
 *         Used for buffers that live entirely on the GPU (e.g. the materialized task array and sort/group
 *         scratch), so the host never touches them and no unified-memory page migration can occur.
 */
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
        value_type *result = nullptr;
        auto error = cudaMalloc((value_type **)&result, n * sizeof(value_type));
        if (error != cudaSuccess) return nullptr;
        return result;
    }
    void deallocate(pointer p, size_type) const noexcept {
        if (p) cudaFree(p);
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

/**
 *  @brief A custom allocator that uses CUDA @b pinned (page-locked) host memory (`cudaMallocHost`).
 *         Host-built staging (e.g. the per-string descriptors) that is bulk-copied to the device at full
 *         bandwidth - pinned memory is required for fast, async `cudaMemcpyAsync` transfers.
 */
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
        value_type *result = nullptr;
        auto error = cudaMallocHost((value_type **)&result, n * sizeof(value_type));
        if (error != cudaSuccess) return nullptr;
        return result;
    }
    void deallocate(pointer p, size_type) const noexcept {
        if (p) cudaFreeHost(p);
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

/**
 *  @brief Orders warp-level tasks from the least to the most shared-memory-hungry: primarily by `bytes_per_cell`
 *      (the kernel-family boundary), then by `density`. Marked `__host__ __device__` so the same predicate drives
 *      the GPU `thrust::sort` and the host `std::sort` fallback.
 */
template <typename task_type_>
struct warp_tasks_density_order_ {
    __host__ __device__ bool operator()(task_type_ const &lhs, task_type_ const &rhs) const noexcept {
        return lhs.bytes_per_cell == rhs.bytes_per_cell ? lhs.density < rhs.density
                                                        : lhs.bytes_per_cell < rhs.bytes_per_cell;
    }
};

/**
 *  @brief A small, host-readable descriptor of one warp-tier launch group, run-length-encoded from the sorted
 *         warp tasks. Carries only the launch-shaping fields, so the host never dereferences a device task to
 *         drive a warp-tier launch: the kernel family from @p bytes_per_cell, the occupancy from @p density and
 *         @p max_memory_requirement, and the task subrange from @p begin_offset / @p count (absolute into the
 *         original task array).
 */
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

/**
 *  @brief Composite RLE key for one warp task: `(bytes_per_cell << 8) | density`. The sort orders tasks by
 *         (`bytes_per_cell`, `density`), so equal composite keys are exactly the contiguous launch groups.
 *         Marked `__host__ __device__` so it drives the `cub` device run-length-encode.
 */
template <typename task_type_>
struct warp_tasks_group_key_functor_ {
    task_type_ const *tasks = nullptr;
    __host__ SZ_DEVICE_INLINE u32_t operator()(size_t index) const noexcept {
        task_type_ const &task = tasks[index];
        return (static_cast<u32_t>(task.bytes_per_cell) << 8) | static_cast<u32_t>(task.density);
    }
};

/** @brief Reads one warp task's `memory_requirement` for the per-group `cub::DeviceSegmentedReduce::Max`. */
template <typename task_type_>
struct warp_tasks_memory_requirement_functor_ {
    __host__ SZ_DEVICE_INLINE size_t operator()(task_type_ const &task) const noexcept {
        return task.memory_requirement;
    }
};

/** @brief `cub::DevicePartition::If` predicate selecting device-level tasks (whole-device cooperative). */
template <typename task_type_>
struct task_is_device_level_functor_ {
    __host__ SZ_DEVICE_INLINE bool operator()(task_type_ const &task) const noexcept {
        return task.density == warps_working_together_k;
    }
};

/** @brief `cub::DevicePartition::If` predicate selecting non-empty tasks (a pre-seeded cell is `infinite_*`). */
template <typename task_type_>
struct task_is_non_empty_functor_ {
    __host__ SZ_DEVICE_INLINE bool operator()(task_type_ const &task) const noexcept {
        return task.density != infinite_warps_per_multiprocessor_k;
    }
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
 *  @note Every data-touching stage runs on the GPU and the host reads only small scalars / the descriptor array:
 *  the device/warp/empty split is two `cub::DevicePartition::If` passes (the two split COUNTS are the only host
 *  reads), the warp middle is sorted with `cub::DeviceMergeSort`, and the launch groups are run-length-encoded
 *  with `cub::DeviceRunLengthEncode::Encode` (unique `(bytes_per_cell, density)` keys + run lengths), turned into
 *  begin offsets via `cub::DeviceScan::ExclusiveSum`, and given a per-group max `memory_requirement` via
 *  `cub::DeviceSegmentedReduce::Max`. All CUB temporaries come from the caller-owned, grow-only buffers (hoisted
 *  on the engine, reused across calls — never a per-call `cudaMalloc`). On any scratch-allocation failure the
 *  routine degrades gracefully: the middle stays partition-ordered and one descriptor per raw run is emitted.
 *
 *  @post The @p tasks are reordered in-place into [device-level | warp-level (sorted) | empty]; @p group_count
 *  host-readable @ref warp_tasks_group_descriptor_t entries are written to @p group_descriptors, each carrying a
 *  warp-tier launch group's absolute task subrange and launch-shaping fields. The host drives the warp-tier
 *  launches from those descriptors alone — never by dereferencing a device task.
 */
template <typename task_type_, typename scratch_buffer_type_, typename task_buffer_type_, typename count_buffer_type_,
          typename key_buffer_type_, typename size_buffer_type_, typename descriptor_buffer_type_>
warp_tasks_groups<task_type_> warp_tasks_grouping( //
    span<task_type_> tasks, gpu_specs_t const &specs, cudaStream_t stream,
    scratch_buffer_type_ &grouping_scratch, // grow-only CUB temporary storage (bytes)
    task_buffer_type_ &partition_scratch,   // grow-only task ping-pong for the two device partitions
    count_buffer_type_ &counts_scratch,     // device-accessible 2-slot partition selected-counts
    key_buffer_type_ &group_keys_scratch,   // device-accessible unique composite keys + run count
    size_buffer_type_ &group_sizes_scratch, // device-accessible run lengths, begin offsets, per-group max memory
    descriptor_buffer_type_ &group_descriptors, size_t &group_count) noexcept {

    using task_t = task_type_;
    warp_tasks_groups<task_t> result;
    group_count = 0;

    size_t const total_tasks = tasks.size();
    result.device_level_tasks = {tasks.begin(), tasks.begin()};
    result.warp_level_tasks = {tasks.begin(), tasks.begin()};
    result.empty_tasks = {tasks.begin(), tasks.end()};
    if (!total_tasks) return result;

    // Split the array into [device-level | warp-level | empty] entirely on the device with two
    // `cub::DevicePartition::If` passes; only the two selected-counts cross back to the host.
    if (partition_scratch.try_resize_uninitialized(total_tasks) == status_t::bad_alloc_k) return result;
    if (counts_scratch.try_resize_uninitialized(2) == status_t::bad_alloc_k) return result;
    task_t *const partition_buffer = partition_scratch.data();
    u32_t *const device_count_out = counts_scratch.data();
    u32_t *const warp_count_out = device_count_out + 1;

    auto const cub_partition = [&](task_t const *input, task_t *output, u32_t *count_out, auto predicate) noexcept {
        size_t partition_bytes = 0;
        if (cub::DevicePartition::If(nullptr, partition_bytes, input, output, count_out, static_cast<int>(total_tasks),
                                     predicate, stream) != cudaSuccess)
            return false;
        if (grouping_scratch.try_resize_uninitialized(partition_bytes) == status_t::bad_alloc_k) return false;
        return cub::DevicePartition::If(grouping_scratch.data(), partition_bytes, input, output, count_out,
                                        static_cast<int>(total_tasks), predicate, stream) == cudaSuccess;
    };

    // Pass 1: device-level tasks (`density == warps_working_together_k`) to the front of `partition_buffer`.
    if (!cub_partition(tasks.data(), partition_buffer, device_count_out, task_is_device_level_functor_<task_t> {}))
        return result;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return result;
    size_t const device_level_count = *device_count_out;
    size_t const rest_count = total_tasks - device_level_count;

    // Pass 2: of the remaining `partition_buffer[device_level_count, total_tasks)`, the non-empty (warp-level)
    // tasks to the front; the input count for `cub::DevicePartition::If` is the FULL `total_tasks`, so we restore
    // the device-level prefix into `partition_buffer` first and partition the whole array back into `tasks`.
    if (!cub_partition(partition_buffer, tasks.data(), warp_count_out, task_is_non_empty_functor_<task_t> {}))
        return result;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return result;
    // The whole-array pass also moved the device-level tasks (their `density == warps_working_together_k != infinite`
    // so they sort to the warp front); the warp-front then holds [device-level | warp-level]. Recover the warp count
    // by subtracting the device prefix the second predicate also kept ahead of the empties.
    size_t const non_empty_count = *warp_count_out;
    size_t const warp_level_count = non_empty_count - device_level_count;

    auto const warp_tasks_begin = tasks.begin() + device_level_count;
    auto const warp_tasks_end = warp_tasks_begin + warp_level_count;
    result.device_level_tasks = {tasks.begin(), warp_tasks_begin};
    result.warp_level_tasks = {warp_tasks_begin, warp_tasks_end};
    result.empty_tasks = {warp_tasks_end, tasks.end()};

    // Sort the warp-level middle from smallest to largest shared-memory footprint on the GPU; equal
    // `(bytes_per_cell, density)` keys become the contiguous launch groups the run-length-encode below discovers.
    if (warp_level_count > 1) {
        warp_tasks_density_order_<task_t> const order {};
        size_t temp_storage_bytes = 0;
        cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes, warp_tasks_begin, warp_level_count, order, stream);
        if (grouping_scratch.try_resize(temp_storage_bytes) == status_t::success_k)
            cub::DeviceMergeSort::SortKeys(grouping_scratch.data(), temp_storage_bytes, warp_tasks_begin,
                                           warp_level_count, order, stream);
        // No drain here: the run-length-encode below consumes the sort on the same stream, so it is already ordered
        // after it. On scratch-allocation failure the middle stays partition-ordered: correct, only less optimally
        // grouped.
    }
    if (!warp_level_count) return result;

    // Run-length-encode the sorted warp tasks on the composite `(bytes_per_cell << 8) | density` key to discover the
    // raw launch groups; an exclusive sum of the run lengths gives each group's begin offset, and a segmented Max of
    // `memory_requirement` gives each group's dynamic-shared-memory footprint. All on the device.
    if (group_keys_scratch.try_resize_uninitialized(warp_level_count + 1) == status_t::bad_alloc_k) return result;
    if (group_sizes_scratch.try_resize_uninitialized(3 * warp_level_count + 1) == status_t::bad_alloc_k) return result;
    u32_t *const unique_keys = group_keys_scratch.data();
    u32_t *const run_count_out = unique_keys + warp_level_count;
    size_t *const run_lengths = group_sizes_scratch.data();
    size_t *const begin_offsets = run_lengths + warp_level_count; // `warp_level_count + 1` slots (trailing end)
    size_t *const group_max_memory = begin_offsets + (warp_level_count + 1);

    warp_tasks_group_key_functor_<task_t> const key_functor {warp_tasks_begin};
    counting_iterator<size_t> iota(0);
    transform_input_iterator<u32_t, warp_tasks_group_key_functor_<task_t>, counting_iterator<size_t>> key_iterator(
        iota, key_functor);
    {
        size_t rle_bytes = 0;
        if (cub::DeviceRunLengthEncode::Encode(nullptr, rle_bytes, key_iterator, unique_keys, run_lengths,
                                               run_count_out, static_cast<int>(warp_level_count),
                                               stream) != cudaSuccess)
            return result;
        if (grouping_scratch.try_resize_uninitialized(rle_bytes) == status_t::bad_alloc_k) return result;
        if (cub::DeviceRunLengthEncode::Encode(grouping_scratch.data(), rle_bytes, key_iterator, unique_keys,
                                               run_lengths, run_count_out, static_cast<int>(warp_level_count),
                                               stream) != cudaSuccess)
            return result;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return result;
    size_t const raw_group_count = *run_count_out;
    if (!raw_group_count) return result;

    {
        warp_tasks_memory_requirement_functor_<task_t> const memory_functor {};
        transform_input_iterator<size_t, warp_tasks_memory_requirement_functor_<task_t>, task_t const *>
            memory_iterator(warp_tasks_begin, memory_functor);
        // Size BOTH the exclusive-sum (group begin offsets) and the segmented Max (per-group shared-memory footprint)
        // up front, then grow the shared CUB scratch ONCE to the larger - so no `try_resize` (a `cudaFree`/`cudaMalloc`
        // device barrier) ever runs while a previously-enqueued CUB kernel is still in flight on the stream. Both calls
        // pass their own byte count (<= capacity) and run sequentially on the stream, so they safely share the buffer.
        size_t scan_bytes = 0, reduce_bytes = 0;
        if (cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, run_lengths, begin_offsets,
                                          static_cast<int>(raw_group_count), stream) != cudaSuccess)
            return result;
        if (cub::DeviceSegmentedReduce::Max(nullptr, reduce_bytes, memory_iterator, group_max_memory,
                                            static_cast<int>(raw_group_count), begin_offsets, begin_offsets + 1,
                                            stream) != cudaSuccess)
            return result;
        if (grouping_scratch.try_resize_uninitialized(sz_max_of_two(scan_bytes, reduce_bytes)) == status_t::bad_alloc_k)
            return result;
        if (cub::DeviceScan::ExclusiveSum(grouping_scratch.data(), scan_bytes, run_lengths, begin_offsets,
                                          static_cast<int>(raw_group_count), stream) != cudaSuccess)
            return result;
        // The segmented Max needs each segment's [begin, end); `begin_offsets` is the begin array and `begin_offsets + 1`
        // the end array, so the trailing end (== `warp_level_count`) is appended on the stream after the scan. The host
        // source outlives the trailing `cudaStreamSynchronize` that drains this block.
        if (cudaMemcpyAsync(begin_offsets + raw_group_count, &warp_level_count, sizeof(size_t), cudaMemcpyHostToDevice,
                            stream) != cudaSuccess)
            return result;
        if (cub::DeviceSegmentedReduce::Max(grouping_scratch.data(), reduce_bytes, memory_iterator, group_max_memory,
                                            static_cast<int>(raw_group_count), begin_offsets, begin_offsets + 1,
                                            stream) != cudaSuccess)
            return result;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return result;

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
        u32_t const head_key = unique_keys[raw_index];
        warp_tasks_group_descriptor_t descriptor;
        descriptor.bytes_per_cell = static_cast<bytes_per_cell_t>(head_key >> 8);
        descriptor.density = static_cast<warp_tasks_density_t>(head_key & 0xFFu);
        descriptor.begin_offset = device_level_count + begin_offsets[raw_index];
        descriptor.count = run_lengths[raw_index];
        descriptor.max_memory_requirement = group_max_memory[raw_index];

        size_t next = raw_index + 1;
        while (next < raw_group_count) {
            size_t const possible_warps = size_t(descriptor.density) * specs.streaming_multiprocessors;
            if (descriptor.count > possible_warps) break; // enough tasks to saturate the device

            u32_t const next_key = unique_keys[next];
            bytes_per_cell_t const next_bytes_per_cell = static_cast<bytes_per_cell_t>(next_key >> 8);
            if (next_bytes_per_cell != descriptor.bytes_per_cell) break; // different kernel family, can't merge

            // Merge the next group in: adopt its (sparser) density, grow the span, take the larger memory footprint.
            descriptor.density = static_cast<warp_tasks_density_t>(next_key & 0xFFu);
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
