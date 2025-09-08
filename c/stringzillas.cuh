/**
 *  @file       stringzillas.cu
 *  @brief      StringZillas library shared code for parallel string operations using CPU & CUDA backends.
 *  @author     Ash Vardanian
 *  @date       March 23, 2025
 */

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

    // clang-format off
    pointer operator->() const noexcept { return &operator*(); }
    strided_ptr &operator++() noexcept { data_ += stride_; return *this; }
    strided_ptr operator++(int) noexcept { strided_ptr temp = *this; ++(*this); return temp; }
    strided_ptr &operator--() noexcept { data_ -= stride_; return *this; }
    strided_ptr operator--(int) noexcept { strided_ptr temp = *this; --(*this); return temp; }
    strided_ptr &operator+=(difference_type offset) noexcept { data_ += offset * stride_; return *this; }
    strided_ptr &operator-=(difference_type offset) noexcept { data_ -= offset * stride_; return *this; }
    strided_ptr operator+(difference_type offset) const noexcept { strided_ptr temp = *this; return temp += offset; }
    strided_ptr operator-(difference_type offset) const noexcept { strided_ptr temp = *this; return temp -= offset; }
    friend difference_type operator-(strided_ptr const &a, strided_ptr const &b) noexcept { assert(a.stride_ == b.stride_); return (a.data_ - b.data_) / static_cast<difference_type>(a.stride_); }
    friend bool operator==(strided_ptr const &a, strided_ptr const &b) noexcept { return a.data_ == b.data_; }
    friend bool operator<(strided_ptr const &a, strided_ptr const &b) noexcept { return a.data_ < b.data_; }
    friend bool operator!=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(a == b); }
    friend bool operator>(strided_ptr const &a, strided_ptr const &b) noexcept { return b < a; }
    friend bool operator<=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(b < a); }
    friend bool operator>=(strided_ptr const &a, strided_ptr const &b) noexcept { return !(a < b); }
    // clang-format on
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
    if (cuda_status.cuda_error != cudaSuccess) {
        if (reporter_message) *reporter_message = cudaGetErrorString(cuda_status.cuda_error);
        return static_cast<sz_status_t>(cuda_status.status);
    }
    else { return propagate_error(cuda_status.status, reporter_message, optional_message); }
}
#endif

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
#if SZ_USE_ICE
        szs::levenshtein_ice_t, szs::affine_levenshtein_ice_t,
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
                szs::cuda_status_t status =
                    ctx.status != sz::status_t::success_k
                        ? ctx.status
                        : engine_variant( //
                              a_container, b_container, results_strided, ctx.executor, ctx.specs);
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
#if SZ_USE_ICE
        szs::levenshtein_utf8_ice_t, // ! `szs::affine_levenshtein_utf8_ice_t` won't compile yet
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
            // No GPU backends for UTF8 Levenshtein distances yet
            result = propagate_error(sz::status_t::unknown_k, error_message);
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
#if SZ_USE_ICE
        szs::needleman_wunsch_ice_t, // ! No affine variant here yet
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
                szs::cuda_status_t status =
                    ctx.status != sz::status_t::success_k
                        ? ctx.status
                        : engine_variant( //
                              a_container, b_container, results_strided, ctx.executor, ctx.specs);
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
#if SZ_USE_ICE
        szs::smith_waterman_ice_t, // ! No affine variant here yet
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

extern "C" {

#pragma region Metadata

SZ_DYNAMIC int szs_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_DYNAMIC int szs_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_DYNAMIC int szs_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }

SZ_DYNAMIC sz_capability_t szs_capabilities(void) {
    // Preserve the static capabilities
    static sz_capability_t static_caps = sz_caps_none_k;
    if (static_caps == sz_caps_none_k) {
        sz_capability_t cpu_caps = sz_capabilities_implementation_();
#if SZ_USE_CUDA
        sz_capability_t gpu_caps = sz_caps_none_k;
        sz::gpu_specs_t first_gpu_specs;
        auto specs_status = static_cast<sz::status_t>(szs::gpu_specs_fetch(first_gpu_specs));
        if (specs_status == sz::status_t::missing_gpu_k) { return cpu_caps; }        // No GPUs available
        else if (specs_status != sz::status_t::success_k) { return sz_caps_none_k; } // Some bug
        gpu_caps = static_cast<sz_capability_t>(gpu_caps | sz_cap_cuda_k);
        if (first_gpu_specs.sm_code >= 30) gpu_caps = static_cast<sz_capability_t>(gpu_caps | sz_cap_kepler_k);
        if (first_gpu_specs.sm_code >= 90) gpu_caps = static_cast<sz_capability_t>(gpu_caps | sz_cap_hopper_k);
        static_caps = static_cast<sz_capability_t>(cpu_caps | gpu_caps);
#else
        static_caps = cpu_caps;
#endif // SZ_USE_CUDA
    }

    return static_caps;
}

SZ_DYNAMIC sz_status_t sz_memory_allocator_init_unified(sz_memory_allocator_t *alloc, char const **error_message) {
#if SZ_USE_CUDA
    alloc->allocate = &sz_memory_allocate_from_unified_;
    alloc->free = &sz_memory_free_from_unified_;
    alloc->handle = nullptr;
    return propagate_error(sz::status_t::success_k, error_message);
#else
    return propagate_error(sz::status_t::missing_gpu_k, error_message);
#endif
}

#pragma endregion Metadata

#pragma region Device Scopes

SZ_DYNAMIC sz_status_t szs_device_scope_init_default(szs_device_scope_t *scope_punned, char const **error_message) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");
    auto *scope = new device_scope_t {default_scope_t {}};
    if (!scope) return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate device scope");
    *scope_punned = reinterpret_cast<szs_device_scope_t>(scope);
    return propagate_error(sz::status_t::success_k, error_message);
}

SZ_DYNAMIC sz_status_t szs_device_scope_init_cpu_cores(sz_size_t cpu_cores, szs_device_scope_t *scope_punned,
                                                       char const **error_message) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");

    // If `cpu_cores` is 0, use all available cores
    if (cpu_cores == 0) cpu_cores = std::thread::hardware_concurrency();

    // If `cpu_cores` is 1, redirect to default scope
    if (cpu_cores == 1) return szs_device_scope_init_default(scope_punned, error_message);

    sz::cpu_specs_t specs;
    auto executor = std::make_unique<fu::basic_pool_t>();
    if (!executor->try_spawn(cpu_cores))
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to spawn thread pool");

    auto *scope =
        new (std::nothrow) device_scope_t(std::in_place_type_t<cpu_scope_t> {}, std::move(executor), std::move(specs));
    if (!scope) return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate CPU device scope");

    *scope_punned = reinterpret_cast<szs_device_scope_t>(scope);
    return propagate_error(sz::status_t::success_k, error_message);
}

SZ_DYNAMIC sz_status_t szs_device_scope_init_gpu_device(sz_size_t gpu_device, szs_device_scope_t *scope_punned,
                                                        char const **error_message) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");

#if SZ_USE_CUDA
    sz::gpu_specs_t specs;
    auto specs_status = szs::gpu_specs_fetch(specs, static_cast<int>(gpu_device));
    if (specs_status.status != sz::status_t::success_k) { return propagate_error(specs_status, error_message); }
    szs::cuda_executor_t executor;
    auto executor_status = executor.try_scheduling(static_cast<int>(gpu_device));
    if (executor_status.status != sz::status_t::success_k) return propagate_error(executor_status, error_message);

    auto *scope =
        new (std::nothrow) device_scope_t {gpu_scope_t {.executor = std::move(executor), .specs = std::move(specs)}};
    if (!scope) return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate GPU device scope");
    *scope_punned = reinterpret_cast<szs_device_scope_t>(scope);
    return propagate_error(sz::status_t::success_k, error_message);
#else
    sz_unused_(gpu_device);
    sz_unused_(scope_punned);
    return propagate_error(sz::status_t::missing_gpu_k, error_message, "CUDA support not compiled in");
#endif
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_cpu_cores(szs_device_scope_t scope_punned, sz_size_t *cpu_cores,
                                                      char const **error_message) {
    if (scope_punned == nullptr || cpu_cores == nullptr)
        return propagate_error(sz::status_t::unknown_k, error_message, "Invalid null pointer argument");
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);

    if (std::holds_alternative<cpu_scope_t>(scope->variants)) {
        auto &cpu_scope = std::get<cpu_scope_t>(scope->variants);
        if (cpu_scope.executor_ptr) {
            *cpu_cores = cpu_scope.executor_ptr->threads_count();
            return propagate_error(sz::status_t::success_k, error_message);
        }
    }
    // Default scope is single-threaded
    else if (std::holds_alternative<default_scope_t>(scope->variants)) {
        *cpu_cores = 1;
        return propagate_error(sz::status_t::success_k, error_message);
    }

    return propagate_error(sz::status_t::unknown_k, error_message, "Device scope is GPU-only");
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_gpu_device(szs_device_scope_t scope_punned, sz_size_t *gpu_device,
                                                       char const **error_message) {
    if (scope_punned == nullptr || gpu_device == nullptr)
        return propagate_error(sz::status_t::unknown_k, error_message, "Invalid null pointer argument");

#if SZ_USE_CUDA
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);
    if (std::holds_alternative<gpu_scope_t>(scope->variants)) {
        auto &gpu_scope = std::get<gpu_scope_t>(scope->variants);
        *gpu_device = static_cast<sz_size_t>(gpu_scope.executor.device_id());
        return propagate_error(sz::status_t::success_k, error_message);
    }
#else
    sz_unused_(scope_punned);
    sz_unused_(gpu_device);
#endif

    return propagate_error(sz::status_t::unknown_k, error_message, "Device scope is CPU-only");
}

SZ_DYNAMIC void szs_device_scope_free(szs_device_scope_t scope_punned) {
    if (scope_punned == nullptr) return;
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);
    delete scope;
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_capabilities(szs_device_scope_t scope_punned, sz_capability_t *capabilities,
                                                         char const **error_message) {

    if (scope_punned == nullptr || capabilities == nullptr)
        return propagate_error(sz::status_t::unknown_k, error_message, "Invalid null pointer argument");
    sz_capability_t system_caps = szs_capabilities();

#if SZ_USE_CUDA
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);
    if (std::holds_alternative<gpu_scope_t>(scope->variants)) {
        // For GPU scope, intersect system capabilities with CUDA capabilities
        *capabilities = static_cast<sz_capability_t>(system_caps & sz_caps_cuda_k);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif

    // For default and CPU scopes, intersect system capabilities with CPU capabilities
    *capabilities = static_cast<sz_capability_t>(system_caps & sz_caps_cpus_k);
    return propagate_error(sz::status_t::success_k, error_message);
}

#pragma endregion Device Scopes

#pragma region Unified Allocator

SZ_DYNAMIC void *szs_unified_alloc(sz_size_t size_bytes) {
#if SZ_USE_CUDA
    return szs::unified_alloc_t {}.allocate(size_bytes);
#else
    return std::malloc(size_bytes);
#endif
}

SZ_DYNAMIC void szs_unified_free(void *ptr, sz_size_t size_bytes) {
    if (!ptr) return;
#if SZ_USE_CUDA
    szs::unified_alloc_t {}.deallocate(static_cast<char *>(ptr), size_bytes);
#else
    sz_unused_(size_bytes);
    std::free(ptr);
#endif
}

#pragma endregion Unified Allocator

#pragma region Levenshtein Distances

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_init(                                             //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    szs_levenshtein_distances_t *engine_punned, char const **error_message) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const substitution_costs = szs::uniform_substitution_costs_t {match, mismatch};
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};

#if SZ_USE_ICE
    bool const can_use_ice = (capabilities & sz_cap_ice_k) == sz_cap_ice_k;
    if (can_use_ice && can_use_linear_costs) {
        auto variant = szs::levenshtein_ice_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_ice_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_ice) {
        auto variant = szs::affine_levenshtein_ice_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_ice_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_ICE

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) == sz_cap_cuda_k;
    if (can_use_cuda && can_use_linear_costs) {
        auto variant = szs::levenshtein_cuda_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_cuda_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_cuda) {
        auto variant = szs::affine_levenshtein_cuda_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_cuda_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_CUDA

#if SZ_USE_KEPLER
    bool const can_use_kepler = (capabilities & sz_caps_ck_k) == sz_caps_ck_k;
    if (can_use_kepler && can_use_linear_costs) {
        auto variant = szs::levenshtein_kepler_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_kepler_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_kepler) {
        auto variant = szs::affine_levenshtein_kepler_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_kepler_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_KEPLER

#if SZ_USE_HOPPER
    bool const can_use_hopper = (capabilities & sz_caps_ckh_k) == sz_caps_ckh_k;
    if (can_use_hopper && can_use_linear_costs) {
        auto variant = szs::levenshtein_hopper_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_hopper_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_hopper) {
        auto variant = szs::affine_levenshtein_hopper_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_hopper_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_HOPPER

    if (can_use_linear_costs) {
        auto variant = szs::levenshtein_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else {
        auto variant = szs::affine_levenshtein_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_sequence(                       //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                              //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return szs_levenshtein_distances_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_u32tape(                        //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,              //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return szs_levenshtein_distances_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_u64tape(                        //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,              //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return szs_levenshtein_distances_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC void szs_levenshtein_distances_free(szs_levenshtein_distances_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<levenshtein_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Levenshtein Distances

#pragma region Levenshtein UTF8 Distances

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_init(                                        //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    szs_levenshtein_distances_utf8_t *engine_punned, char const **error_message) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const substitution_costs = szs::uniform_substitution_costs_t {match, mismatch};
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};

#if SZ_USE_ICE
    bool const can_use_ice = (capabilities & sz_cap_ice_k) != 0;
    if (can_use_ice && can_use_linear_costs) {
        auto variant = szs::levenshtein_utf8_ice_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_utf8_backends_t(std::in_place_type_t<szs::levenshtein_utf8_ice_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate UTF-8 Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_utf8_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_ICE

    bool const can_use_serial = (capabilities & sz_cap_serial_k) == sz_cap_serial_k;
    if (can_use_serial && can_use_linear_costs) {
        auto variant = szs::levenshtein_utf8_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_utf8_backends_t(std::in_place_type_t<szs::levenshtein_utf8_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate UTF-8 Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_utf8_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else {
        auto variant = szs::affine_levenshtein_utf8_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) levenshtein_utf8_backends_t(
            std::in_place_type_t<szs::affine_levenshtein_utf8_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate UTF-8 Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_utf8_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }

    return propagate_error(sz::status_t::unknown_k, error_message, "No supported UTF-8 Levenshtein backends available");
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_sequence(                       //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                                   //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return szs_levenshtein_distances_utf8_for_(                 //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_u32tape(                        //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,                   //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return szs_levenshtein_distances_utf8_for_(                 //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_u64tape(                        //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,                   //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return szs_levenshtein_distances_utf8_for_(                 //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC void szs_levenshtein_distances_utf8_free(szs_levenshtein_distances_utf8_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<levenshtein_utf8_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Levenshtein UTF8 Distances

#pragma region Needleman Wunsch

SZ_DYNAMIC sz_status_t szs_needleman_wunsch_scores_init(                       //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,          //
    szs_needleman_wunsch_scores_t *engine_punned, char const **error_message) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};
    auto substitution_costs = szs::error_costs_256x256_t {};
    std::memcpy((void *)&substitution_costs, (void const *)subs, sizeof(substitution_costs));

#if SZ_USE_ICE
    bool const can_use_ice = (capabilities & sz_cap_ice_k) == sz_cap_ice_k;
    if (can_use_ice && can_use_linear_costs) {
        auto variant = szs::needleman_wunsch_ice_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            needleman_wunsch_backends_t(std::in_place_type_t<szs::needleman_wunsch_ice_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Needleman-Wunsch engine");

        *engine_punned = reinterpret_cast<szs_needleman_wunsch_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_ICE

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) != 0;
    if (can_use_cuda && can_use_linear_costs) {
        auto variant = szs::needleman_wunsch_cuda_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            needleman_wunsch_backends_t(std::in_place_type_t<szs::needleman_wunsch_cuda_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Needleman-Wunsch engine");

        *engine_punned = reinterpret_cast<szs_needleman_wunsch_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_cuda) {
        auto variant = szs::affine_needleman_wunsch_cuda_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) needleman_wunsch_backends_t(
            std::in_place_type_t<szs::affine_needleman_wunsch_cuda_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Needleman-Wunsch engine");

        *engine_punned = reinterpret_cast<szs_needleman_wunsch_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_CUDA

#if SZ_USE_HOPPER
    bool const can_use_hopper = (capabilities & sz_caps_ckh_k) == sz_caps_ckh_k;
    if (can_use_hopper && can_use_linear_costs) {
        auto variant = szs::needleman_wunsch_hopper_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            needleman_wunsch_backends_t(std::in_place_type_t<szs::needleman_wunsch_hopper_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Needleman-Wunsch engine");

        *engine_punned = reinterpret_cast<szs_needleman_wunsch_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_hopper) {
        auto variant = szs::affine_needleman_wunsch_hopper_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) needleman_wunsch_backends_t(
            std::in_place_type_t<szs::affine_needleman_wunsch_hopper_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Needleman-Wunsch engine");

        *engine_punned = reinterpret_cast<szs_needleman_wunsch_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_HOPPER

    if (can_use_linear_costs) {
        auto variant = szs::needleman_wunsch_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            needleman_wunsch_backends_t(std::in_place_type_t<szs::needleman_wunsch_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Needleman-Wunsch engine");

        *engine_punned = reinterpret_cast<szs_needleman_wunsch_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else {
        auto variant = szs::affine_needleman_wunsch_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) needleman_wunsch_backends_t(
            std::in_place_type_t<szs::affine_needleman_wunsch_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Needleman-Wunsch engine");

        *engine_punned = reinterpret_cast<szs_needleman_wunsch_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
}

SZ_DYNAMIC sz_status_t szs_needleman_wunsch_scores_sequence(                       //
    szs_needleman_wunsch_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                                //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return szs_needleman_wunsch_scores_for_(                    //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_needleman_wunsch_scores_u32tape(                        //
    szs_needleman_wunsch_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,                //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return szs_needleman_wunsch_scores_for_(                    //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_needleman_wunsch_scores_u64tape(                        //
    szs_needleman_wunsch_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,                //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return szs_needleman_wunsch_scores_for_(                    //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC void szs_needleman_wunsch_scores_free(szs_needleman_wunsch_scores_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<needleman_wunsch_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Needleman Wunsch

#pragma region Smith Waterman

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_init(                         //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,          //
    szs_smith_waterman_scores_t *engine_punned, char const **error_message) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};
    auto substitution_costs = szs::error_costs_256x256_t {};
    std::memcpy((void *)&substitution_costs, (void const *)subs, sizeof(substitution_costs));

#if SZ_USE_ICE
    bool const can_use_ice = (capabilities & sz_cap_ice_k) == sz_cap_ice_k;
    if (can_use_ice && can_use_linear_costs) {
        auto variant = szs::smith_waterman_ice_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            smith_waterman_backends_t(std::in_place_type_t<szs::smith_waterman_ice_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Smith-Waterman engine");

        *engine_punned = reinterpret_cast<szs_smith_waterman_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_ICE

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) != 0;
    if (can_use_cuda && can_use_linear_costs) {
        auto variant = szs::smith_waterman_cuda_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            smith_waterman_backends_t(std::in_place_type_t<szs::smith_waterman_cuda_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Smith-Waterman engine");

        *engine_punned = reinterpret_cast<szs_smith_waterman_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_cuda) {
        auto variant = szs::affine_smith_waterman_cuda_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            smith_waterman_backends_t(std::in_place_type_t<szs::affine_smith_waterman_cuda_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Smith-Waterman engine");

        *engine_punned = reinterpret_cast<szs_smith_waterman_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_CUDA

#if SZ_USE_HOPPER
    bool const can_use_hopper = (capabilities & sz_caps_ckh_k) == sz_caps_ckh_k;
    if (can_use_hopper && can_use_linear_costs) {
        auto variant = szs::smith_waterman_hopper_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            smith_waterman_backends_t(std::in_place_type_t<szs::smith_waterman_hopper_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Smith-Waterman engine");

        *engine_punned = reinterpret_cast<szs_smith_waterman_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_hopper) {
        auto variant = szs::affine_smith_waterman_hopper_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            smith_waterman_backends_t(std::in_place_type_t<szs::affine_smith_waterman_hopper_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Smith-Waterman engine");

        *engine_punned = reinterpret_cast<szs_smith_waterman_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_HOPPER

    if (can_use_linear_costs) {
        auto variant = szs::smith_waterman_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            smith_waterman_backends_t(std::in_place_type_t<szs::smith_waterman_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Smith-Waterman engine");

        *engine_punned = reinterpret_cast<szs_smith_waterman_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else {
        auto variant = szs::affine_smith_waterman_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            smith_waterman_backends_t(std::in_place_type_t<szs::affine_smith_waterman_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate Smith-Waterman engine");

        *engine_punned = reinterpret_cast<szs_smith_waterman_scores_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
}

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_sequence(                       //
    szs_smith_waterman_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                              //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return szs_smith_waterman_scores_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_u32tape(                        //
    szs_smith_waterman_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,              //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return szs_smith_waterman_scores_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_u64tape(                        //
    szs_smith_waterman_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,              //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return szs_smith_waterman_scores_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC void szs_smith_waterman_scores_free(szs_smith_waterman_scores_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<smith_waterman_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Smith Waterman

#pragma region Fingerprints

SZ_DYNAMIC sz_status_t szs_fingerprints_init(                         //
    sz_size_t dimensions, sz_size_t alphabet_size,                    //
    sz_size_t const *window_widths, sz_size_t window_widths_count,    //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities, //
    szs_fingerprints_t *engine_punned, char const **error_message) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // Use some default window widths if none are provided
    sz_size_t const default_window_widths[] = {3, 4, 5, 7, 9, 11, 15, 31};
    if (!window_widths || window_widths_count == 0) {
        window_widths = default_window_widths;
        window_widths_count = sizeof(default_window_widths) / sizeof(sz_size_t);
    }

    // For optimal performance the number of dimensions per window width must be divisible by the fingerprint slice.
    auto const dimensions_per_window_width_min = dimensions / window_widths_count;
    auto const dimensions_per_window_width_max = sz::divide_round_up(dimensions, window_widths_count);
    auto const can_use_sliced_sketchers = (dimensions_per_window_width_min == dimensions_per_window_width_max) &&
                                          (dimensions_per_window_width_min % fingerprint_slice_k == 0);
    using fallback_variant_cpus_t = typename fingerprints_backends_t::fallback_variant_cpus_t;

#if SZ_USE_HASWELL
    bool const can_use_haswell = (capabilities & sz_cap_haswell_k) == sz_cap_haswell_k;
    if (can_use_haswell && can_use_sliced_sketchers) {
        auto const count_hashers = dimensions / fingerprint_slice_k;
        using hasher_t = szs::floating_rolling_hashers<sz_cap_haswell_k, fingerprint_slice_k>;
        vec<hasher_t> hashers;
        if (hashers.try_resize(count_hashers) != sz::status_t::success_k) return sz_bad_alloc_k;

        // Populate the hashers with the given window widths
        for (size_t i = 0; i < count_hashers; ++i) {
            auto const window_width = window_widths[i % window_widths_count];
            auto const first_dimension_offset = i * fingerprint_slice_k;
            auto const seed_status = hashers[i].try_seed(window_width, alphabet_size, first_dimension_offset);
            if (seed_status != sz::status_t::success_k) return static_cast<sz_status_t>(seed_status);
        }

        auto engine =
            new (std::nothrow) fingerprints_backends_t(std::in_place_type_t<vec<hasher_t>>(), std::move(hashers));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Fingerprints engine");
        engine->dimensions = dimensions;
        *engine_punned = reinterpret_cast<szs_fingerprints_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_HASWELL

#if SZ_USE_SKYLAKE
    bool const can_use_skylake = (capabilities & sz_cap_skylake_k) == sz_cap_skylake_k;
    if (can_use_skylake && can_use_sliced_sketchers) {
        auto const count_hashers = dimensions / fingerprint_slice_k;
        using hasher_t = szs::floating_rolling_hashers<sz_cap_skylake_k, fingerprint_slice_k>;
        vec<hasher_t> hashers;
        if (hashers.try_resize(count_hashers) != sz::status_t::success_k) return sz_bad_alloc_k;

        // Populate the hashers with the given window widths
        for (size_t i = 0; i < count_hashers; ++i) {
            auto const window_width = window_widths[i % window_widths_count];
            auto const first_dimension_offset = i * fingerprint_slice_k;
            auto const seed_status = hashers[i].try_seed(window_width, alphabet_size, first_dimension_offset);
            if (seed_status != sz::status_t::success_k) return static_cast<sz_status_t>(seed_status);
        }

        auto engine =
            new (std::nothrow) fingerprints_backends_t(std::in_place_type_t<vec<hasher_t>>(), std::move(hashers));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Fingerprints engine");
        engine->dimensions = dimensions;
        *engine_punned = reinterpret_cast<szs_fingerprints_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_SKYLAKE

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) == sz_cap_cuda_k;
    if (can_use_cuda && can_use_sliced_sketchers) {
        auto const count_hashers = dimensions / fingerprint_slice_k;
        using hasher_t = szs::floating_rolling_hashers<sz_cap_cuda_k, fingerprint_slice_k>;
        vec<hasher_t> hashers;
        if (hashers.try_resize(count_hashers) != sz::status_t::success_k) return sz_bad_alloc_k;

        // Populate the hashers with the given window widths
        for (size_t i = 0; i < count_hashers; ++i) {
            auto const window_width = window_widths[i % window_widths_count];
            auto const first_dimension_offset = i * fingerprint_slice_k;
            auto const seed_status = hashers[i].try_seed(window_width, alphabet_size, first_dimension_offset);
            if (seed_status != sz::status_t::success_k) return static_cast<sz_status_t>(seed_status);
        }

        auto engine =
            new (std::nothrow) fingerprints_backends_t(std::in_place_type_t<vec<hasher_t>>(), std::move(hashers));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Fingerprints engine");
        engine->dimensions = dimensions;
        *engine_punned = reinterpret_cast<szs_fingerprints_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_cuda) {
        using fallback_variant_cuda_t = typename fingerprints_backends_t::fallback_variant_cuda_t;
        auto variant = fallback_variant_cuda_t();
        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            auto const window_width = window_widths[dimension % window_widths_count];
            auto const extend_status = variant.try_extend(window_width, 1, alphabet_size);
            if (extend_status != sz::status_t::success_k) return static_cast<sz_status_t>(extend_status);
        }

        auto engine = new (std::nothrow)
            fingerprints_backends_t(std::in_place_type_t<fallback_variant_cuda_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        engine->dimensions = dimensions;
        *engine_punned = reinterpret_cast<szs_fingerprints_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_CUDA

    // Build the vectorized, but serial backend
    if (can_use_sliced_sketchers) {
        auto const count_hashers = dimensions / fingerprint_slice_k;
        using hasher_t = szs::floating_rolling_hashers<sz_cap_serial_k, fingerprint_slice_k>;
        vec<hasher_t> hashers;
        if (hashers.try_resize(count_hashers) != sz::status_t::success_k) return sz_bad_alloc_k;

        // Populate the hashers with the given window widths
        for (size_t i = 0; i < count_hashers; ++i) {
            auto const window_width = window_widths[i % window_widths_count];
            auto const first_dimension_offset = i * fingerprint_slice_k;
            auto const seed_status = hashers[i].try_seed(window_width, alphabet_size, first_dimension_offset);
            if (seed_status != sz::status_t::success_k) return static_cast<sz_status_t>(seed_status);
        }

        auto engine =
            new (std::nothrow) fingerprints_backends_t(std::in_place_type_t<vec<hasher_t>>(), std::move(hashers));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Fingerprints engine");
        engine->dimensions = dimensions;
        *engine_punned = reinterpret_cast<szs_fingerprints_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }

    // Build the fallback variant with interleaving width dimensions
    auto variant = fallback_variant_cpus_t();
    for (size_t dimension = 0; dimension < dimensions; ++dimension) {
        auto const window_width = window_widths[dimension % window_widths_count];
        auto const extend_status = variant.try_extend(window_width, 1, alphabet_size);
        if (extend_status != sz::status_t::success_k) return static_cast<sz_status_t>(extend_status);
    }

    auto engine =
        new (std::nothrow) fingerprints_backends_t(std::in_place_type_t<fallback_variant_cpus_t>(), std::move(variant));
    if (!engine)
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Fingerprints engine");

    engine->dimensions = dimensions;
    *engine_punned = reinterpret_cast<szs_fingerprints_t>(engine);
    return propagate_error(sz::status_t::success_k, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_sequence(                       //
    szs_fingerprints_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *texts,                                         //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_as_cpp_container_t {texts};
    return szs_fingerprints_for_(                      //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_u32tape(                        //
    szs_fingerprints_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *texts,                                 //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_u32tape_as_cpp_container_t {texts};
    return szs_fingerprints_for_(                      //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_u64tape(                        //
    szs_fingerprints_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *texts,                                 //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_u64tape_as_cpp_container_t {texts};
    return szs_fingerprints_for_(                      //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC void szs_fingerprints_free(szs_fingerprints_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<fingerprints_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Fingerprints

#pragma region Fingerprints UTF8

SZ_DYNAMIC sz_status_t szs_fingerprints_utf8_init(                    //
    sz_size_t dimensions, sz_size_t alphabet_size,                    //
    sz_size_t const *window_widths, sz_size_t window_widths_count,    //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities, //
    szs_fingerprints_utf8_t *engine_punned, char const **error_message) {

    return szs_fingerprints_init( //
        dimensions, alphabet_size, window_widths, window_widths_count, alloc, capabilities, engine_punned,
        error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_utf8_sequence(                       //
    szs_fingerprints_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *texts,                                              //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    return szs_fingerprints_sequence(        //
        engine_punned, device_punned, texts, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_utf8_u32tape(                        //
    szs_fingerprints_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *texts,                                      //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    return szs_fingerprints_u32tape(         //
        engine_punned, device_punned, texts, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_utf8_u64tape(                        //
    szs_fingerprints_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *texts,                                      //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    return szs_fingerprints_u64tape(         //
        engine_punned, device_punned, texts, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC void szs_fingerprints_utf8_free(szs_fingerprints_utf8_t engine_punned) {
    return szs_fingerprints_free(engine_punned);
}

#pragma endregion Fingerprints UTF8
}
