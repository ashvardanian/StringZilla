/**
 *  @file c/stringzillas.cuh
 *  @brief StringZillas shared scaffolding (scopes, backend variant lists, dispatch) included by
 *         the per-algorithm CPU & CUDA shims.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#ifndef STRINGZILLAS_SCAFFOLDING_CUH_
#define STRINGZILLAS_SCAFFOLDING_CUH_

#if !defined(FU_ENABLE_NUMA)
#define FU_ENABLE_NUMA 0
#endif

#include <stringzillas/stringzillas.h> // StringZillas library header

#include <variant>        // For `std::variant`
#include <cstring>        // For `std::memcpy`
#include <string_view>    // For `std::string_view`
#include <thread>         // For `std::thread::hardware_concurrency`
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

using malloc_t = std::allocator<char>;
#if SZ_USE_CUDA
using ualloc_t = szs::unified_alloc_t;
#endif // SZ_USE_CUDA

/** Helper class for `std::visit` to handle multiple callable types in a single variant. */
template <typename... callable_types_>
struct overloaded : callable_types_... {
    using callable_types_::operator()...;
};
template <typename... callable_types_>
overloaded(callable_types_...) -> overloaded<callable_types_...>;

/** Wraps a `sz_sequence_t` to feel like `std::vector<std::string_view>>` in the implementation layer. */
struct sz_sequence_as_cpp_container_t {
    using value_type = std::string_view;
    sz_sequence_t const *sequence_ = nullptr;

    std::size_t size() const noexcept {
        sz_assert_(sequence_ != nullptr && "Sequence must not be null");
        return sequence_->count;
    }
    std::string_view operator[](std::size_t index) const noexcept {
        sz_assert_(sequence_ != nullptr && "Sequence must not be null");
        sz_assert_(index < sequence_->count && "Index out of bounds");
        sz_cptr_t start_ptr = sequence_->get_start(sequence_->handle, index);
        sz_size_t length = sequence_->get_length(sequence_->handle, index);
        return {start_ptr, length};
    }
};

/** Wraps a `sz_sequence_u64tape_t` to feel like `std::vector<std::string_view>>` in the implementation layer. */
struct sz_sequence_u64tape_as_cpp_container_t {
    using value_type = std::string_view;
    sz_sequence_u64tape_t const *tape_ = nullptr;

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

/** Wraps a `sz_sequence_u32tape_t` to feel like `std::vector<std::string_view>>` in the implementation layer. */
struct sz_sequence_u32tape_as_cpp_container_t {
    using value_type = std::string_view;
    sz_sequence_u32tape_t const *tape_ = nullptr;

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
template <typename element_type_, sz_size_t row_extent_ = SZ_SIZE_MAX>
struct strided_rows {
    using value_type = element_type_;

  private:
    static constexpr sz_size_t extent_k = row_extent_; // Extent of each row, default to SZ_SIZE_MAX

    sz_ptr_t data_ = nullptr;
    sz_size_t stride_bytes_ = 0;
    sz_size_t row_length_ = 0;
    sz_size_t count_ = 0;

  public:
    strided_rows(sz_ptr_t data, sz_size_t row_length, sz_size_t stride_bytes, sz_size_t count) noexcept
        : data_(data), stride_bytes_(stride_bytes), row_length_(row_length), count_(count) {}

    std::size_t size() const noexcept { return count_; }

    template <sz_size_t new_extent_ = extent_k>
    strided_rows<element_type_, new_extent_> shifted(std::ptrdiff_t offset) const noexcept {
        return strided_rows<element_type_, new_extent_>(data_ + offset, row_length_, stride_bytes_, count_);
    }

    sz::span<value_type, extent_k> operator[](std::size_t index) const noexcept {
        sz_assert_(index < count_ && "Index out of bounds");
        return sz::span<value_type, extent_k>(reinterpret_cast<value_type *>(data_ + index * stride_bytes_),
                                              row_length_);
    }
};

/**
 *  @brief Convenience class for strided pointer arithmetic.
 *  @see
 * https://github.com/ashvardanian/less_slow.cpp/blob/b21507f7143f8175b92d0b2b2d827b3bd4bb081b/less_slow.cpp#L2593-L2641
 */
template <typename value_type_>
class strided_ptr {
    sz_ptr_t data_;
    std::size_t stride_;

  public:
    using value_type = value_type_;
    using pointer = value_type_ *;
    using reference = value_type_ &;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    strided_ptr(sz_ptr_t data, std::size_t stride_bytes) : data_(data), stride_(stride_bytes) {
        assert(data_ && "Pointer must not be null, as NULL arithmetic is undefined");
    }
#if defined(__cpp_lib_assume_aligned) // Not available in Apple Clang
    reference operator*() const noexcept {
        return *std::launder(std::assume_aligned<1>(reinterpret_cast<pointer>(data_)));
    }
    reference operator[](difference_type i) const noexcept {
        return *std::launder(std::assume_aligned<1>(reinterpret_cast<pointer>(data_ + i * stride_)));
    }
#else
    reference operator*() const noexcept { return *reinterpret_cast<pointer>(data_); }
    reference operator[](difference_type i) const noexcept { return *reinterpret_cast<pointer>(data_ + i * stride_); }
#endif // defined(__cpp_lib_assume_aligned)

    pointer operator->() const noexcept { return &operator*(); }
    strided_ptr &operator++() noexcept {
        data_ += stride_;
        return *this;
    }
    strided_ptr operator++(int) noexcept {
        strided_ptr temp = *this;
        ++(*this);
        return temp;
    }
    strided_ptr &operator--() noexcept {
        data_ -= stride_;
        return *this;
    }
    strided_ptr operator--(int) noexcept {
        strided_ptr temp = *this;
        --(*this);
        return temp;
    }
    strided_ptr &operator+=(difference_type offset) noexcept {
        data_ += offset * stride_;
        return *this;
    }
    strided_ptr &operator-=(difference_type offset) noexcept {
        data_ -= offset * stride_;
        return *this;
    }
    strided_ptr operator+(difference_type offset) const noexcept {
        strided_ptr temp = *this;
        return temp += offset;
    }
    strided_ptr operator-(difference_type offset) const noexcept {
        strided_ptr temp = *this;
        return temp -= offset;
    }
    friend difference_type operator-(strided_ptr const &a, strided_ptr const &b) noexcept {
        assert(a.stride_ == b.stride_);
        return (a.data_ - b.data_) / static_cast<difference_type>(a.stride_);
    }
    friend bool operator==(strided_ptr const &a, strided_ptr const &b) noexcept { return a.data_ == b.data_; }
    friend bool operator<(strided_ptr const &a, strided_ptr const &b) noexcept { return a.data_ < b.data_; }
    friend bool operator!=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(a == b); }
    friend bool operator>(strided_ptr const &a, strided_ptr const &b) noexcept { return b < a; }
    friend bool operator<=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(b < a); }
    friend bool operator>=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(a < b); }
};

constexpr bool is_gpu_capability(sz_capability_t capability) noexcept {
    return (capability & sz_cap_cuda_k) != 0 || (capability & sz_cap_kepler_k) != 0 ||
           (capability & sz_cap_hopper_k) != 0;
}

inline sz_status_t propagate_error(sz::status_t status, char const **reporter_message,
                                   char const *optional_message = nullptr) noexcept {
    if (!reporter_message) return static_cast<sz_status_t>(status);

    // If the optional message is provided, use it verbatim
    if (optional_message && reporter_message) {
        *reporter_message = optional_message;
        return static_cast<sz_status_t>(status);
    }

    // Otherwise, map the status code to a predefined message
    switch (status) {
    case sz::status_t::success_k: *reporter_message = nullptr; break;
    case sz::status_t::bad_alloc_k: *reporter_message = "Memory allocation failed"; break;
    case sz::status_t::invalid_utf8_k: *reporter_message = "Invalid UTF-8 input"; break;
    case sz::status_t::contains_duplicates_k: *reporter_message = "Input contains duplicates"; break;
    case sz::status_t::overflow_risk_k: *reporter_message = "Overflow risk detected"; break;
    case sz::status_t::unexpected_dimensions_k: *reporter_message = "Input/output size mismatch"; break;
    case sz::status_t::missing_gpu_k: *reporter_message = "GPU device not available or CUDA not initialized"; break;
    case sz::status_t::device_code_mismatch_k: *reporter_message = "Backend and executor mismatch"; break;
    case sz::status_t::device_memory_mismatch_k: *reporter_message = "Use device-reachable or unified memory"; break;
    case sz::status_t::unknown_k: *reporter_message = "Unknown error"; break;
    default: *reporter_message = "Unrecognized error code"; break;
    }

    return static_cast<sz_status_t>(status);
}

#if SZ_USE_CUDA
inline sz_status_t propagate_error(szs::cuda_status_t cuda_status, char const **reporter_message,
                                   char const *optional_message = nullptr) noexcept {
    // Prefer the stable driver error *name* (e.g. "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES") so misconfigured driver
    // launches are diagnosable instead of collapsing into the generic "Unknown error" of the status-only path.
    if (cuda_status.driver_error != CUDA_SUCCESS) {
        if (reporter_message) {
            char const *driver_error_name = nullptr;
            if (cuGetErrorName(cuda_status.driver_error, &driver_error_name) != CUDA_SUCCESS)
                driver_error_name = "Unknown CUDA driver error";
            *reporter_message = driver_error_name;
        }
        return static_cast<sz_status_t>(cuda_status.status);
    }
    // Otherwise fall back to the runtime error *name* (e.g. "cudaErrorLaunchOutOfResources") for the runtime
    // calls we still keep (events, occupancy, malloc).
    else if (cuda_status.cuda_error != cudaSuccess) {
        if (reporter_message) *reporter_message = cudaGetErrorName(cuda_status.cuda_error);
        return static_cast<sz_status_t>(cuda_status.status);
    }
    else { return propagate_error(cuda_status.status, reporter_message, optional_message); }
}
#endif

#if SZ_USE_CUDA

/** @brief Redirects to CUDA's unified memory allocator. */
inline void *sz_memory_allocate_from_unified_(sz_size_t size_bytes, void *handle) {
    sz_unused_(handle);
    return szs::unified_alloc_t {}.allocate(size_bytes);
}

/** @brief Redirects to CUDA's unified memory allocator. */
inline void sz_memory_free_from_unified_(void *address, sz_size_t size_bytes, void *handle) {
    sz_unused_(handle);
    szs::unified_alloc_t {}.deallocate((char *)address, size_bytes);
}

#endif // SZ_USE_CUDA

struct default_scope_t {};
inline szs::dummy_executor_t get_executor(default_scope_t const &) noexcept { return {}; }
inline sz::cpu_specs_t get_specs(default_scope_t const &) noexcept { return {}; }

struct cpu_scope_t {
    std::unique_ptr<fu::basic_pool_t> executor_ptr;
    sz::cpu_specs_t specs;

    cpu_scope_t() = default;
    cpu_scope_t(std::unique_ptr<fu::basic_pool_t> exec_ptr, sz::cpu_specs_t cpu_specs) noexcept
        : executor_ptr(std::move(exec_ptr)), specs(cpu_specs) {}
};
inline fu::basic_pool_t &get_executor(cpu_scope_t &scope) noexcept { return *scope.executor_ptr; }
inline sz::cpu_specs_t get_specs(cpu_scope_t const &scope) noexcept { return scope.specs; }

#if SZ_USE_CUDA
struct gpu_scope_t {
    szs::cuda_executor_t executor;
    sz::gpu_specs_t specs;
};
inline szs::cuda_executor_t &get_executor(gpu_scope_t &scope) noexcept { return scope.executor; }
inline sz::gpu_specs_t get_specs(gpu_scope_t const &scope) noexcept { return scope.specs; }

/** Cached default GPU context (device 0) to avoid repeated scheduling boilerplate */
struct default_gpu_context_t {
    szs::cuda_status_t status {sz::status_t::unknown_k, cudaSuccess};
    szs::cuda_executor_t executor;
    sz::gpu_specs_t specs;
};

inline default_gpu_context_t &default_gpu_context() {
    static default_gpu_context_t ctx = [] {
        default_gpu_context_t result;
        szs::cuda_status_t specs_status = szs::gpu_specs_fetch(result.specs, 0);
        if (specs_status.status != sz::status_t::success_k) {
            result.status = specs_status;
            return result;
        }
        szs::cuda_status_t exec_status = result.executor.try_scheduling(0);
        result.status = exec_status;
        return result;
    }();
    return ctx;
}
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

struct levenshtein_backends_t {

    /**
     *  On each hardware platform we use a different backend for Levenshtein distances,
     *  separately covering:
     *  - Linear or Affine gap costs
     *  - Serial, Ice Lake, CUDA, CUDA Kepler, and CUDA Hopper backends
     */
    std::variant<
#if SZ_USE_ICELAKE
        szs::levenshtein_icelake_t, szs::affine_levenshtein_icelake_t,
#endif
#if SZ_USE_HASWELL
        szs::levenshtein_haswell_t, szs::affine_levenshtein_haswell_t,
#endif
#if SZ_USE_NEON
        szs::levenshtein_neon_t, szs::affine_levenshtein_neon_t,
#endif
#if SZ_USE_RVV
        szs::levenshtein_rvv_t, szs::affine_levenshtein_rvv_t,
#endif
#if SZ_USE_CUDA
        szs::levenshtein_cuda_t, szs::affine_levenshtein_cuda_t,
#endif
#if SZ_USE_KEPLER
        szs::levenshtein_kepler_t, szs::affine_levenshtein_kepler_t,
#endif
#if SZ_USE_HOPPER
        szs::levenshtein_hopper_t, szs::affine_levenshtein_hopper_t,
#endif
        szs::levenshtein_serial_t, szs::affine_levenshtein_serial_t>
        variants;

    template <typename... variants_arguments_>
    levenshtein_backends_t(variants_arguments_ &&...args) noexcept
        : variants(std::forward<variants_arguments_>(args)...) {}
};

template <typename texts_type_>
sz_status_t szs_levenshtein_distances_for_(                                      //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned, //
    texts_type_ const &a_container, texts_type_ const &b_container,              //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(results != nullptr && "Results must not be null");

    // Revert back from opaque pointer types
    auto *engine = reinterpret_cast<levenshtein_backends_t *>(engine_punned);
    auto *device = reinterpret_cast<device_scope_t *>(device_punned);

    // Wrap our stable ABI sequences into C++ friendly containers
    auto results_strided = strided_ptr<sz_size_t> {reinterpret_cast<sz_ptr_t>(results), results_stride};

    // The simplest case, is having non-optimized non-unrolled hashers.
    sz_status_t result = sz_success_k;
    auto variant_logic = [&](auto &engine_variant) {
        using engine_variant_t = std::decay_t<decltype(engine_variant)>;
        constexpr sz_capability_t engine_capability_k = engine_variant_t::capability_k;

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
#if SZ_USE_CUDA
            if (std::holds_alternative<gpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<gpu_scope_t>(device->variants);
                szs::cuda_status_t status = engine_variant(    //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            // Try ephemeral GPU on default scope (device 0)
            else if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &ctx = default_gpu_context();
                szs::cuda_status_t status = ctx.status != sz::status_t::success_k
                                                ? ctx.status
                                                : engine_variant( //
                                                      a_container, b_container, results_strided, ctx.executor,
                                                      ctx.specs);
                result = propagate_error(status, error_message);
            }
            else { result = propagate_error(sz::status_t::device_code_mismatch_k, error_message); }
#else
            result = propagate_error(sz::status_t::missing_gpu_k, error_message);
#endif // SZ_USE_CUDA
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else { result = propagate_error(sz::status_t::device_code_mismatch_k, error_message); }
        }
    };

    std::visit(variant_logic, engine->variants);
    return result;
}

struct levenshtein_utf8_backends_t {

    /**
     *  On each hardware platform we use a different backend for Levenshtein UTF8 distances,
     *  separately covering:
     *  - Serial, Ice Lake, CUDA backends
     */
    std::variant<
#if SZ_USE_ICELAKE
        szs::levenshtein_utf8_icelake_t,
#endif
#if SZ_USE_HASWELL
        szs::levenshtein_utf8_haswell_t,
#endif
#if SZ_USE_NEON
        szs::levenshtein_utf8_neon_t,
#endif
#if SZ_USE_RVV
        szs::levenshtein_utf8_rvv_t,
#endif
#if SZ_USE_CUDA
        szs::levenshtein_utf8_cuda_t,
#endif
        szs::levenshtein_utf8_serial_t, szs::affine_levenshtein_utf8_serial_t>
        variants;

    template <typename... variants_arguments_>
    levenshtein_utf8_backends_t(variants_arguments_ &&...args) noexcept
        : variants(std::forward<variants_arguments_>(args)...) {}
};

template <typename texts_type_>
sz_status_t szs_levenshtein_distances_utf8_for_(                                      //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    texts_type_ const &a_container, texts_type_ const &b_container,                   //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(results != nullptr && "Results must not be null");

    // Revert back from opaque pointer types
    auto *engine = reinterpret_cast<levenshtein_utf8_backends_t *>(engine_punned);
    auto *device = reinterpret_cast<device_scope_t *>(device_punned);

    // Wrap our stable ABI sequences into C++ friendly containers
    auto results_strided = strided_ptr<sz_size_t> {reinterpret_cast<sz_ptr_t>(results), results_stride};

    // The simplest case, is having non-optimized non-unrolled hashers.
    sz_status_t result = sz_success_k;
    auto variant_logic = [&](auto &engine_variant) {
        using engine_variant_t = std::decay_t<decltype(engine_variant)>;
        constexpr sz_capability_t engine_capability_k = engine_variant_t::capability_k;

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
#if SZ_USE_CUDA
            if (std::holds_alternative<gpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<gpu_scope_t>(device->variants);
                szs::cuda_status_t status = engine_variant(    //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            // Try ephemeral GPU on default scope (device 0)
            else if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &ctx = default_gpu_context();
                szs::cuda_status_t status = ctx.status != sz::status_t::success_k
                                                ? ctx.status
                                                : engine_variant( //
                                                      a_container, b_container, results_strided, ctx.executor,
                                                      ctx.specs);
                result = propagate_error(status, error_message);
            }
            else { result = propagate_error(sz::status_t::device_code_mismatch_k, error_message); }
#else
            result = propagate_error(sz::status_t::missing_gpu_k, error_message);
#endif // SZ_USE_CUDA
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else { result = sz_device_code_mismatch_k; }
        }
    };

    std::visit(variant_logic, engine->variants);
    return result;
}

struct needleman_wunsch_backends_t {

    /**
     *  On each hardware platform we use a different backend for Levenshtein distances,
     *  separately covering:
     *  - Linear or Affine gap costs
     *  - Serial, Ice Lake, CUDA, CUDA Kepler, and CUDA Hopper backends
     */
    std::variant<
#if SZ_USE_ICELAKE
        szs::needleman_wunsch_icelake_t, szs::affine_needleman_wunsch_icelake_t,
#endif
#if SZ_USE_HASWELL
        szs::needleman_wunsch_haswell_t, szs::affine_needleman_wunsch_haswell_t,
#endif
#if SZ_USE_NEON
        szs::needleman_wunsch_neon_t, szs::affine_needleman_wunsch_neon_t,
#endif
#if SZ_USE_RVV
        szs::needleman_wunsch_rvv_t, szs::affine_needleman_wunsch_rvv_t,
#endif
#if SZ_USE_CUDA
        szs::needleman_wunsch_cuda_t, szs::affine_needleman_wunsch_cuda_t,
#endif
#if SZ_USE_HOPPER
        szs::needleman_wunsch_hopper_t, szs::affine_needleman_wunsch_hopper_t,
#endif
        szs::needleman_wunsch_serial_t, szs::affine_needleman_wunsch_serial_t>
        variants;

    template <typename... variants_arguments_>
    needleman_wunsch_backends_t(variants_arguments_ &&...args) noexcept
        : variants(std::forward<variants_arguments_>(args)...) {}
};

template <typename texts_type_>
sz_status_t szs_needleman_wunsch_scores_for_(                                      //
    szs_needleman_wunsch_scores_t engine_punned, szs_device_scope_t device_punned, //
    texts_type_ const &a_container, texts_type_ const &b_container,                //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(results != nullptr && "Results must not be null");

    // Revert back from opaque pointer types
    auto *engine = reinterpret_cast<needleman_wunsch_backends_t *>(engine_punned);
    auto *device = reinterpret_cast<device_scope_t *>(device_punned);

    // Wrap our stable ABI sequences into C++ friendly containers
    auto results_strided = strided_ptr<sz_ssize_t> {reinterpret_cast<sz_ptr_t>(results), results_stride};

    // The simplest case, is having non-optimized non-unrolled hashers.
    sz_status_t result = sz_success_k;
    auto variant_logic = [&](auto &engine_variant) {
        using engine_variant_t = std::decay_t<decltype(engine_variant)>;
        constexpr sz_capability_t engine_capability_k = engine_variant_t::capability_k;

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
#if SZ_USE_CUDA
            if (std::holds_alternative<gpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<gpu_scope_t>(device->variants);
                szs::cuda_status_t status = engine_variant(    //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &ctx = default_gpu_context();
                szs::cuda_status_t status = ctx.status != sz::status_t::success_k
                                                ? ctx.status
                                                : engine_variant( //
                                                      a_container, b_container, results_strided, ctx.executor,
                                                      ctx.specs);
                result = propagate_error(status, error_message);
            }
            else { result = propagate_error(sz::status_t::unknown_k, error_message); }
#else
            result = propagate_error(sz::status_t::unknown_k, error_message); // GPU support is not enabled
#endif // SZ_USE_CUDA
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else { result = propagate_error(sz::status_t::unknown_k, error_message); }
        }
    };

    std::visit(variant_logic, engine->variants);
    return result;
}

struct smith_waterman_backends_t {

    /**
     *  On each hardware platform we use a different backend for Levenshtein distances,
     *  separately covering:
     *  - Linear or Affine gap costs
     *  - Serial, Ice Lake, CUDA, CUDA Kepler, and CUDA Hopper backends
     */
    std::variant<
#if SZ_USE_ICELAKE
        szs::smith_waterman_icelake_t, szs::affine_smith_waterman_icelake_t,
#endif
#if SZ_USE_HASWELL
        szs::smith_waterman_haswell_t, szs::affine_smith_waterman_haswell_t,
#endif
#if SZ_USE_NEON
        szs::smith_waterman_neon_t, szs::affine_smith_waterman_neon_t,
#endif
#if SZ_USE_RVV
        szs::smith_waterman_rvv_t, szs::affine_smith_waterman_rvv_t,
#endif
#if SZ_USE_CUDA
        szs::smith_waterman_cuda_t, szs::affine_smith_waterman_cuda_t,
#endif
#if SZ_USE_HOPPER
        szs::smith_waterman_hopper_t, szs::affine_smith_waterman_hopper_t,
#endif
        szs::smith_waterman_serial_t, szs::affine_smith_waterman_serial_t>
        variants;

    template <typename... variants_arguments_>
    smith_waterman_backends_t(variants_arguments_ &&...args) noexcept
        : variants(std::forward<variants_arguments_>(args)...) {}
};

template <typename texts_type_>
sz_status_t szs_smith_waterman_scores_for_(                                      //
    szs_smith_waterman_scores_t engine_punned, szs_device_scope_t device_punned, //
    texts_type_ const &a_container, texts_type_ const &b_container,              //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(results != nullptr && "Results must not be null");

    // Revert back from opaque pointer types
    auto *engine = reinterpret_cast<smith_waterman_backends_t *>(engine_punned);
    auto *device = reinterpret_cast<device_scope_t *>(device_punned);

    // Wrap our stable ABI sequences into C++ friendly containers
    auto results_strided = strided_ptr<sz_ssize_t> {reinterpret_cast<sz_ptr_t>(results), results_stride};

    // The simplest case, is having non-optimized non-unrolled hashers.
    sz_status_t result = sz_success_k;
    auto variant_logic = [&](auto &engine_variant) {
        using engine_variant_t = std::decay_t<decltype(engine_variant)>;
        constexpr sz_capability_t engine_capability_k = engine_variant_t::capability_k;

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
#if SZ_USE_CUDA
            if (std::holds_alternative<gpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<gpu_scope_t>(device->variants);
                szs::cuda_status_t status = engine_variant(    //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else if (std::holds_alternative<default_scope_t>(device->variants)) {
                sz::gpu_specs_t specs;
                auto specs_status = szs::gpu_specs_fetch(specs, 0);
                if (specs_status.status != sz::status_t::success_k) {
                    result = propagate_error(specs_status, error_message);
                }
                else {
                    szs::cuda_executor_t executor;
                    auto exec_status = executor.try_scheduling(0);
                    if (exec_status.status != sz::status_t::success_k) {
                        result = propagate_error(exec_status, error_message);
                    }
                    else {
                        szs::cuda_status_t status = engine_variant( //
                            a_container, b_container, results_strided, executor, specs);
                        result = propagate_error(status, error_message);
                    }
                }
            }
            else { result = propagate_error(sz::status_t::unknown_k, error_message); }
#else
            result = propagate_error(sz::status_t::unknown_k, error_message); // GPU support is not enabled
#endif // SZ_USE_CUDA
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else { result = propagate_error(sz::status_t::unknown_k, error_message); }
        }
    };

    std::visit(variant_logic, engine->variants);
    return result;
}

template <typename element_type_>
using vec = szs::safe_vector<element_type_, std::allocator<element_type_>>;

static constexpr size_t fingerprint_slice_k = 64;

struct fingerprints_backends_t {
    using fallback_variant_cpus_t = szs::basic_rolling_hashers<szs::floating_rolling_hasher<sz::f64_t>, sz::u32_t>;
#if SZ_USE_CUDA
    using fallback_variant_cuda_t = szs::basic_rolling_hashers<szs::floating_rolling_hasher<sz::f64_t>, sz::u32_t,
                                                               sz::u32_t, ualloc_t, sz_cap_cuda_k>;
#endif // SZ_USE_CUDA

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
        vec<szs::floating_rolling_hashers<sz_cap_cuda_k, fingerprint_slice_k>>, fallback_variant_cuda_t,
#endif
        vec<szs::floating_rolling_hashers<sz_cap_serial_k, fingerprint_slice_k>>, fallback_variant_cpus_t>
        variants;

    sz_size_t dimensions = 0; // Total number of dimensions across all hashers

    template <typename... variants_arguments_>
    fingerprints_backends_t(variants_arguments_ &&...args) noexcept
        : variants(std::forward<variants_arguments_>(args)...) {}
};

template <typename texts_type_>
sz_status_t szs_fingerprints_for_(                                      //
    szs_fingerprints_t engine_punned, szs_device_scope_t device_punned, //
    texts_type_ const &texts_container,                                 //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(min_hashes != nullptr && "Output min_hashes cannot be null");
    sz_assert_(min_counts != nullptr && "Output min_counts cannot be null");

    // Revert back from opaque pointer types
    auto *engine = reinterpret_cast<fingerprints_backends_t *>(engine_punned);
    auto *device = reinterpret_cast<device_scope_t *>(device_punned);

    // Wrap our stable ABI sequences into C++ friendly containers
    auto const dims = engine->dimensions;
    auto const texts_count = texts_container.size();

    // The simplest case, is having non-optimized non-unrolled hashers.
    sz_status_t result = sz_success_k;
    using fallback_variant_cpus_t = typename fingerprints_backends_t::fallback_variant_cpus_t;
    auto fallback_logic_cpus = [&](fallback_variant_cpus_t &fallback_hashers) {
        auto const min_hashes_rows = //
            strided_rows<sz_u32_t> {reinterpret_cast<sz_ptr_t>(min_hashes), dims, min_hashes_stride, texts_count};
        auto const min_counts_rows = //
            strided_rows<sz_u32_t> {reinterpret_cast<sz_ptr_t>(min_counts), dims, min_counts_stride, texts_count};

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
        else { result = propagate_error(sz::status_t::unknown_k, error_message); }
    };
#if SZ_USE_CUDA
    using fallback_variant_cuda_t = typename fingerprints_backends_t::fallback_variant_cuda_t;
    auto fallback_logic_gpus = [&](fallback_variant_cuda_t &fallback_hashers) {
        auto const min_hashes_rows = //
            strided_rows<sz_u32_t> {reinterpret_cast<sz_ptr_t>(min_hashes), dims, min_hashes_stride, texts_count};
        auto const min_counts_rows = //
            strided_rows<sz_u32_t> {reinterpret_cast<sz_ptr_t>(min_counts), dims, min_counts_stride, texts_count};

        // GPU fallback hashers can work with GPU scope, or default scope via an ephemeral GPU executor
        if (std::holds_alternative<gpu_scope_t>(device->variants)) {
            auto &device_scope = std::get<gpu_scope_t>(device->variants);
            sz::status_t status = fallback_hashers(                //
                texts_container, min_hashes_rows, min_counts_rows, //
                get_executor(device_scope), get_specs(device_scope));
            result = static_cast<sz_status_t>(status);
        }
        else if (std::holds_alternative<default_scope_t>(device->variants)) {
            auto &ctx = default_gpu_context();
            if (ctx.status.status != sz::status_t::success_k) { result = propagate_error(ctx.status, error_message); }
            else {
                sz::status_t status = fallback_hashers( //
                    texts_container, min_hashes_rows, min_counts_rows, ctx.executor, ctx.specs);
                result = propagate_error(status, error_message);
            }
        }
        else { result = propagate_error(sz::status_t::unknown_k, error_message); }
    };
#endif // SZ_USE_CUDA

    // The unrolled logic is a bit more complex than `fallback_logic_cpus`, but in practice involves
    // just one additional loop level.
    auto unrolled_logic = [&](auto &&unrolled_hashers) {
        using unrolled_hashers_t = std::decay_t<decltype(unrolled_hashers)>;
        using unrolled_hasher_t = typename unrolled_hashers_t::value_type;
        constexpr sz_capability_t engine_capability_k = unrolled_hasher_t::capability_k;
        constexpr size_t bytes_per_slice_k = fingerprint_slice_k * sizeof(sz_u32_t);

        // Each engine will produce only a few dimensions so the outputs should be defined
        // differently
        auto const min_hashes_rows = //
            strided_rows<sz_u32_t> {reinterpret_cast<sz_ptr_t>(min_hashes), fingerprint_slice_k, min_hashes_stride,
                                    texts_count};
        auto const min_counts_rows = //
            strided_rows<sz_u32_t> {reinterpret_cast<sz_ptr_t>(min_counts), fingerprint_slice_k, min_counts_stride,
                                    texts_count};

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
#if SZ_USE_CUDA
            if (std::holds_alternative<gpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<gpu_scope_t>(device->variants);
                for (std::size_t i = 0; i < unrolled_hashers.size(); ++i) {
                    auto &engine_variant = unrolled_hashers[i];
                    szs::cuda_status_t status = engine_variant(                                       //
                        texts_container,                                                              //
                        min_hashes_rows.template shifted<fingerprint_slice_k>(i * bytes_per_slice_k), //
                        min_counts_rows.template shifted<fingerprint_slice_k>(i * bytes_per_slice_k), //
                        get_executor(device_scope), get_specs(device_scope));
                    result = propagate_error(status, error_message);
                    if (result != sz_success_k) break;
                }
            }
            else if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &ctx = default_gpu_context();
                if (ctx.status != sz::status_t::success_k) { result = propagate_error(ctx.status, error_message); }
                else {
                    for (std::size_t i = 0; i < unrolled_hashers.size(); ++i) {
                        auto &engine_variant = unrolled_hashers[i];
                        szs::cuda_status_t status = engine_variant(                                       //
                            texts_container,                                                              //
                            min_hashes_rows.template shifted<fingerprint_slice_k>(i * bytes_per_slice_k), //
                            min_counts_rows.template shifted<fingerprint_slice_k>(i * bytes_per_slice_k), //
                            ctx.executor, ctx.specs);
                        result = propagate_error(status, error_message);
                        if (result != sz_success_k) break;
                    }
                }
            }
            else { result = propagate_error(sz::status_t::unknown_k, error_message); }
#else
            result = propagate_error(sz::status_t::unknown_k, error_message); // GPU support is not enabled
#endif // SZ_USE_CUDA
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                for (std::size_t i = 0; i < unrolled_hashers.size(); ++i) {
                    auto &engine_variant = unrolled_hashers[i];
                    sz::status_t status = engine_variant(                                             //
                        texts_container,                                                              //
                        min_hashes_rows.template shifted<fingerprint_slice_k>(i * bytes_per_slice_k), //
                        min_counts_rows.template shifted<fingerprint_slice_k>(i * bytes_per_slice_k), //
                        get_executor(device_scope), get_specs(device_scope));
                    result = propagate_error(status, error_message);
                }
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                for (std::size_t i = 0; i < unrolled_hashers.size(); ++i) {
                    auto &engine_variant = unrolled_hashers[i];
                    sz::status_t status = engine_variant(                                             //
                        texts_container,                                                              //
                        min_hashes_rows.template shifted<fingerprint_slice_k>(i * bytes_per_slice_k), //
                        min_counts_rows.template shifted<fingerprint_slice_k>(i * bytes_per_slice_k), //
                        get_executor(device_scope), get_specs(device_scope));
                    result = propagate_error(status, error_message);
                }
            }
            else { result = propagate_error(sz::status_t::unknown_k, error_message); }
        }
    };

#if SZ_USE_CUDA
    std::visit(overloaded {fallback_logic_cpus, fallback_logic_gpus, unrolled_logic}, engine->variants);
#else
    std::visit(overloaded {fallback_logic_cpus, unrolled_logic}, engine->variants);
#endif
    return result;
}

#endif // STRINGZILLAS_SCAFFOLDING_CUH_
