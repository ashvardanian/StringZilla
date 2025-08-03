/**
 *  @file       stringzillas.cu
 *  @brief      StringZillas library for parallel string operations using CUDA C++ and OpenMP backends.
 *  @author     Ash Vardanian
 *  @date       March 23, 2025
 */
#include <stringzillas/stringzillas.h> // StringZillas library header

#include <variant>        // For `std::variant`
#include <fork_union.hpp> // Fork-join scoped thread pool

#include <stringzillas/fingerprints.hpp> // C++ templates for string processing
#include <stringzillas/similarities.hpp> // C++ templates for string similarity

#if SZ_USE_CUDA
#include <stringzillas/fingerprints.cuh> // Parallel string processing in CUDA
#include <stringzillas/similarities.cuh> // Parallel string similarity in CUDA
#endif

namespace fu = ashvardanian::fork_union;
namespace sz = ashvardanian::stringzilla;
namespace szs = ashvardanian::stringzillas;

/** Helper class for `std::visit` to handle multiple callable types in a single variant. */
template <typename... callable_types_>
struct overloaded : callable_types_... {
    using callable_types_::operator()...;
};
template <typename... callable_types_>
overloaded(callable_types_...) -> overloaded<callable_types_...>;

/** Wraps a `sz_sequence_t` to feel like `std::vector<std::string_view>>` in the implementation layer. */
struct sz_sequence_as_cpp_container_t {
    sz_sequence_t const *sequence_ = nullptr;

    std::size_t size() const noexcept {
        sz_assert_(sequence_ != nullptr && "Sequence must not be null");
        return sequence_->count;
    }
    std::string_view operator[](std::size_t index) const noexcept {
        sz_assert_(sequence_ != nullptr && "Sequence must not be null");
        sz_assert_(index < sequence_->count && "Index out of bounds");
        return {sequence_->get_start(sequence_->handle, index), sequence_->get_length(sequence_->handle, index)};
    }
};

/** Wraps a `sz_arrow_u64tape_t` to feel like `std::vector<std::string_view>>` in the implementation layer. */
struct sz_arrow_u64tape_as_cpp_container_t {
    sz_arrow_u64tape_t const *tape_ = nullptr;

    std::size_t size() const noexcept {
        sz_assert_(tape_ != nullptr && "Tape must not be null");
        return tape_->count;
    }
    std::string_view operator[](std::size_t index) const noexcept {
        sz_assert_(tape_ != nullptr && "Tape must not be null");
        sz_assert_(index < tape_->count && "Index out of bounds");
        return {tape_->data + tape_->offsets[index], tape_->offsets[index + 1] - tape_->offsets[index]};
    }
};

/** Wraps a `sz_arrow_u32tape_t` to feel like `std::vector<std::string_view>>` in the implementation layer. */
struct sz_arrow_u32tape_as_cpp_container_t {
    sz_arrow_u32tape_t const *tape_ = nullptr;

    std::size_t size() const noexcept {
        sz_assert_(tape_ != nullptr && "Tape must not be null");
        return tape_->count;
    }
    std::string_view operator[](std::size_t index) const noexcept {
        sz_assert_(tape_ != nullptr && "Tape must not be null");
        sz_assert_(index < tape_->count && "Index out of bounds");
        return {tape_->data + tape_->offsets[index], tape_->offsets[index + 1] - tape_->offsets[index]};
    }
};

/** Convenience class for slicing a strided fingerprints output. */
template <typename element_type_>
struct strided_rows {
    using value_type = element_type_;

  private:
    sz_ptr_t data_ = nullptr;
    sz_size_t stride_bytes_ = 0;
    sz_size_t row_length_ = 0;
    sz_size_t count_ = 0;

  public:
    strided_rows(sz_ptr_t data, sz_size_t row_length, sz_size_t stride_bytes, sz_size_t count) noexcept
        : data_(data), stride_bytes_(stride_bytes), row_length_(row_length), count_(count) {}

    std::size_t size() const noexcept { return count_; }

    sz::span<value_type> operator[](std::size_t index) const noexcept {
        sz_assert_(index < count_ && "Index out of bounds");
        return sz::span<value_type>(reinterpret_cast<value_type *>(data_ + index * stride_bytes_), row_length_);
    }
};

#if SZ_USE_CUDA

/** @brief Redirects to CUDA's unified memory allocator. */
void *sz_memory_allocate_from_unified_(sz_size_t size_bytes, void *handle) {
    sz_unused_(handle);
    return szs::unified_alloc_t {}.allocate(size_bytes);
}

/** @brief Redirects to CUDA's unified memory allocator. */
void sz_memory_free_from_unified_(void *address, sz_size_t size_bytes, void *handle) {
    sz_unused_(handle);
    szs::unified_alloc_t {}.deallocate((char *)address, size_bytes);
}

#endif // SZ_USE_CUDA

struct default_scope_t {};
szs::dummy_executor_t get_executor(default_scope_t const &) noexcept { return {}; }
sz::cpu_specs_t get_specs(default_scope_t const &) noexcept { return {}; }

struct cpu_scope_t {
    std::unique_ptr<fu::basic_pool_t> executor_ptr;
    sz::cpu_specs_t specs;

    cpu_scope_t() = default;
    cpu_scope_t(std::unique_ptr<fu::basic_pool_t> exec_ptr, sz::cpu_specs_t cpu_specs) noexcept
        : executor_ptr(std::move(exec_ptr)), specs(cpu_specs) {}
};
fu::basic_pool_t &get_executor(cpu_scope_t &scope) noexcept { return *scope.executor_ptr; }
sz::cpu_specs_t get_specs(cpu_scope_t const &scope) noexcept { return scope.specs; }

#if SZ_USE_CUDA
struct gpu_scope_t {
    szs::cuda_executor_t executor;
    sz::gpu_specs_t specs;
};
szs::cuda_executor_t &get_executor(gpu_scope_t &scope) noexcept { return scope.executor; }
sz::gpu_specs_t get_specs(gpu_scope_t const &scope) noexcept { return scope.specs; }
#endif

struct device_scope_t {
#if SZ_USE_CUDA
    std::variant<default_scope_t, cpu_scope_t, gpu_scope_t> variants;
#else
    std::variant<default_scope_t, cpu_scope_t> variants;
#endif

    template <typename... variants_arguments_>
    device_scope_t(variants_arguments_ &&...args) noexcept : variants(std::forward<variants_arguments_>(args)...) {}
};

static constexpr size_t fingerprint_slice_k = 64;

template <typename element_type_>
using vec = szs::safe_vector<element_type_, std::allocator<element_type_>>;

struct fingerprints_t {
    using fallback_variant_t = szs::basic_rolling_hashers<szs::floating_rolling_hasher<sz::f64_t>, sz::u32_t>;

    /**
     *  On each hardware platform the contains a group of rolling hashers.
     *  Each rolling hasher produces `fingerprint_slice_k` worth of fingerprint dimensions.
     */
    std::variant<
#if SZ_USE_HASWELL
        vec<szs::floating_rolling_hashers<sz_cap_haswell_k, fingerprint_slice_k>>,
#endif
#if SZ_USE_SKYLAKE
        vec<szs::floating_rolling_hashers<sz_cap_skylake_k, fingerprint_slice_k>>,
#endif
#if SZ_USE_CUDA
        vec<szs::floating_rolling_hashers<sz_cap_cuda_k, fingerprint_slice_k>>,
#endif
        vec<szs::floating_rolling_hashers<sz_cap_serial_k, fingerprint_slice_k>>, fallback_variant_t>
        variants;

    sz_size_t dimensions = 0; // Total number of dimensions across all hashers

    template <typename... variants_arguments_>
    fingerprints_t(variants_arguments_ &&...args) noexcept : variants(std::forward<variants_arguments_>(args)...) {}
};

template <typename texts_type_>
sz_status_t sz_fingerprints_for_(                                     //
    sz_fingerprints_t engine_punned, sz_device_scope_t device_punned, //
    texts_type_ &&texts_container,                                    //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                //
    sz_u32_t *min_counts, sz_size_t min_counts_stride) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(min_hashes != nullptr && "Output min_hashes cannot be null");
    sz_assert_(min_counts != nullptr && "Output min_counts cannot be null");

    // Revert back from opaque pointer types
    auto *engine = reinterpret_cast<fingerprints_t *>(engine_punned);
    auto *device = reinterpret_cast<device_scope_t *>(device_punned);

    // Wrap our stable ABI sequences into C++ friendly containers
    auto const dims = engine->dimensions;
    auto const texts_count = texts_container.size();
    auto min_hashes_rows =
        strided_rows<sz_u32_t> {reinterpret_cast<sz_ptr_t>(min_hashes), dims, min_hashes_stride, texts_count};
    auto min_counts_rows =
        strided_rows<sz_u32_t> {reinterpret_cast<sz_ptr_t>(min_counts), dims, min_counts_stride, texts_count};

    // The simplest case, is having non-optimized non-unrolled hashers.
    sz_status_t result = sz_success_k;
    using fallback_variant_t = typename fingerprints_t::fallback_variant_t;
    auto fallback_logic = [&](fallback_variant_t &fallback_hashers) {
        // CPU fallback hashers can only work with CPU-compatible device scopes
        if (std::holds_alternative<default_scope_t>(device->variants)) {
            auto &device_scope = std::get<default_scope_t>(device->variants);
            sz::status_t status = fallback_hashers(                //
                texts_container, min_hashes_rows, min_counts_rows, //
                get_executor(device_scope), get_specs(device_scope));
            result = static_cast<sz_status_t>(status);
        }
        else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
            auto &device_scope = std::get<cpu_scope_t>(device->variants);
            sz::status_t status = fallback_hashers(                //
                texts_container, min_hashes_rows, min_counts_rows, //
                get_executor(device_scope), get_specs(device_scope));
            result = static_cast<sz_status_t>(status);
        }
        else { result = sz_status_unknown_k; }
    };

    // The unrolled logic is a bit more complex than `fallback_logic`, but in practice involves
    // just one additional loop level.
    auto unrolled_logic = [&](auto &&unrolled_hashers) { std::printf("Unrolled hashers with %zu dimensions\n", dims); };

    std::visit(overloaded {fallback_logic, unrolled_logic}, engine->variants);
    return result;
}

extern "C" {

SZ_DYNAMIC sz_status_t sz_memory_allocator_init_unified(sz_memory_allocator_t *alloc) {
#if SZ_USE_CUDA
    alloc->allocate = &sz_memory_allocate_from_unified_;
    alloc->free = &sz_memory_free_from_unified_;
    alloc->handle = nullptr;
    return sz_success_k;
#else
    return sz_missing_gpu_k;
#endif
}

SZ_DYNAMIC sz_status_t sz_device_scope_init_default(sz_device_scope_t *scope_punned) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");
    auto *scope = new device_scope_t {default_scope_t {}};
    if (!scope) return sz_bad_alloc_k;
    *scope_punned = reinterpret_cast<sz_device_scope_t>(scope);
    return sz_success_k;
}

SZ_DYNAMIC sz_status_t sz_device_scope_init_cpu_cores(sz_size_t cpu_cores, sz_device_scope_t *scope_punned) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");
    sz_assert_(cpu_cores > 0 && "CPU cores must be greater than zero");
    sz_assert_(cpu_cores > 1 && "For a single-threaded execution, use the default scope");

    sz::cpu_specs_t specs;
    auto executor = std::make_unique<fu::basic_pool_t>();
    if (!executor->try_spawn(cpu_cores)) return sz_bad_alloc_k;

    auto *scope =
        new (std::nothrow) device_scope_t(std::in_place_type_t<cpu_scope_t> {}, std::move(executor), std::move(specs));
    if (!scope) return sz_bad_alloc_k;
    *scope_punned = reinterpret_cast<sz_device_scope_t>(scope);
    return sz_success_k;
}

SZ_DYNAMIC sz_status_t sz_device_scope_init_gpu_device(sz_size_t gpu_device, sz_device_scope_t *scope_punned) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");

#if SZ_USE_CUDA
    sz::gpu_specs_t specs;
    auto specs_status = szs::gpu_specs_fetch(specs, static_cast<int>(gpu_device));
    if (specs_status.status != sz::status_t::success_k) return static_cast<sz_status_t>(specs_status.status);
    szs::cuda_executor_t executor;
    auto executor_status = executor.try_scheduling(static_cast<int>(gpu_device));
    if (executor_status.status != sz::status_t::success_k) return static_cast<sz_status_t>(executor_status.status);

    auto *scope =
        new (std::nothrow) device_scope_t {gpu_scope_t {.executor = std::move(executor), .specs = std::move(specs)}};
    if (!scope) return sz_bad_alloc_k;
    *scope_punned = reinterpret_cast<sz_device_scope_t>(scope);
    return sz_success_k;
#else
    sz_unused_(gpu_device);
    sz_unused_(scope_punned);
    return sz_missing_gpu_k;
#endif
}

SZ_DYNAMIC void sz_device_scope_free(sz_device_scope_t scope_punned) {
    if (scope_punned == nullptr) return;
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);
    delete scope;
}

SZ_DYNAMIC sz_status_t sz_fingerprints_init(                              //
    sz_size_t alphabet_size, sz_size_t const *window_widths,              //
    sz_size_t window_widths_count, sz_size_t dimensions_per_window_width, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,     //
    sz_fingerprints_t *engine_punned) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If window widths are not provided, let's pick some of the default configurations.
    auto const dimensions = window_widths_count * dimensions_per_window_width;
    auto const can_use_sliced_sketchers = dimensions_per_window_width % fingerprint_slice_k == 0;
    using fallback_variant_t = typename fingerprints_t::fallback_variant_t;

    if (!can_use_sliced_sketchers) {
        auto variant = fallback_variant_t();
        for (size_t window_width_index = 0; window_width_index < window_widths_count; ++window_width_index) {
            auto const window_width = window_widths[window_width_index];
            auto extend_status = variant.try_extend(window_width, dimensions_per_window_width, alphabet_size);
            if (extend_status != sz::status_t::success_k) return static_cast<sz_status_t>(extend_status);
        }

        auto engine = new (std::nothrow) fingerprints_t(std::in_place_type_t<fallback_variant_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        engine->dimensions = dimensions;
        *engine_punned = reinterpret_cast<sz_fingerprints_t>(engine);
        return sz_success_k;
    }

    // TODO: Implement unrolled logic
    return sz_status_unknown_k;
}

SZ_DYNAMIC sz_status_t sz_fingerprints_sequence(                      //
    sz_fingerprints_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_t const *texts,                                       //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                //
    sz_u32_t *min_counts, sz_size_t min_counts_stride) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_as_cpp_container_t {texts};
    return sz_fingerprints_for_(                       //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride);
}

SZ_DYNAMIC sz_status_t sz_fingerprints_u64tape(                       //
    sz_fingerprints_t engine_punned, sz_device_scope_t device_punned, //
    sz_arrow_u64tape_t const *texts,                                  //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                //
    sz_u32_t *min_counts, sz_size_t min_counts_stride) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_arrow_u64tape_as_cpp_container_t {texts};
    return sz_fingerprints_for_(                       //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride);
}

SZ_DYNAMIC sz_status_t sz_fingerprints_u32tape(                       //
    sz_fingerprints_t engine_punned, sz_device_scope_t device_punned, //
    sz_arrow_u32tape_t const *texts,                                  //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                //
    sz_u32_t *min_counts, sz_size_t min_counts_stride) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_arrow_u32tape_as_cpp_container_t {texts};
    return sz_fingerprints_for_(                       //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride);
}

SZ_DYNAMIC void sz_fingerprints_free(sz_fingerprints_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<fingerprints_t *>(engine_punned);
    delete engine;
}

} // extern "C"