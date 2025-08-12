/**
 *  @file       stringzillas.cu
 *  @brief      StringZillas library shared code for parallel string operations using CPU & CUDA backends.
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

using malloc_t = std::allocator<char>;
#if SZ_USE_CUDA
using ualloc_t = szs::unified_alloc<char>;
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

/** Wraps a `sz_sequence_u64tape_t` to feel like `std::vector<std::string_view>>` in the implementation layer. */
struct sz_sequence_u64tape_as_cpp_container_t {
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
sz_status_t sz_levenshtein_distances_for_(                                     //
    sz_levenshtein_distances_t engine_punned, sz_device_scope_t device_punned, //
    texts_type_ &&a_container, texts_type_ &&b_container,                      //
    sz_size_t *results, sz_size_t results_stride) {

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
        constexpr sz_capability_t engine_capability_k = engine_variant.capability_k;

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
#if SZ_USE_CUDA
            if (std::holds_alternative<gpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<gpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = static_cast<sz_status_t>(status);
            }
            else { result = sz_status_unknown_k; }
#else
            result = sz_status_unknown_k; // GPU support is not enabled
#endif // SZ_USE_CUDA
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = static_cast<sz_status_t>(status);
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = static_cast<sz_status_t>(status);
            }
            else { result = sz_status_unknown_k; }
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
        szs::levenshtein_utf8_ice_t, szs::affine_levenshtein_utf8_ice_t,
#endif
        szs::levenshtein_utf8_serial_t, szs::affine_levenshtein_utf8_serial_t>
        variants;

    template <typename... variants_arguments_>
    levenshtein_utf8_backends_t(variants_arguments_ &&...args) noexcept
        : variants(std::forward<variants_arguments_>(args)...) {}
};

template <typename texts_type_>
sz_status_t sz_levenshtein_distances_utf8_for_(                                     //
    sz_levenshtein_distances_utf8_t engine_punned, sz_device_scope_t device_punned, //
    texts_type_ &&a_container, texts_type_ &&b_container,                           //
    sz_size_t *results, sz_size_t results_stride) {

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
        constexpr sz_capability_t engine_capability_k = engine_variant.capability_k;

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
            // No GPU backends for UTF8 Levenshtein distances yet
            result = sz_status_unknown_k;
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = static_cast<sz_status_t>(status);
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = static_cast<sz_status_t>(status);
            }
            else { result = sz_status_unknown_k; }
        }
    };

    std::visit(variant_logic, engine->variants);
    return result;
}

struct needleman_wunsch_scores_backends_t {

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
    needleman_wunsch_scores_backends_t(variants_arguments_ &&...args) noexcept
        : variants(std::forward<variants_arguments_>(args)...) {}
};

template <typename texts_type_>
sz_status_t sz_needleman_wunsch_scores_for_(                                     //
    sz_needleman_wunsch_scores_t engine_punned, sz_device_scope_t device_punned, //
    texts_type_ &&a_container, texts_type_ &&b_container,                        //
    sz_ssize_t *results, sz_size_t results_stride) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(results != nullptr && "Results must not be null");

    // Revert back from opaque pointer types
    auto *engine = reinterpret_cast<needleman_wunsch_scores_backends_t *>(engine_punned);
    auto *device = reinterpret_cast<device_scope_t *>(device_punned);

    // Wrap our stable ABI sequences into C++ friendly containers
    auto results_strided = strided_ptr<sz_ssize_t> {reinterpret_cast<sz_ptr_t>(results), results_stride};

    // The simplest case, is having non-optimized non-unrolled hashers.
    sz_status_t result = sz_success_k;
    auto variant_logic = [&](auto &engine_variant) {
        constexpr sz_capability_t engine_capability_k = engine_variant.capability_k;

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
#if SZ_USE_CUDA
            if (std::holds_alternative<gpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<gpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = static_cast<sz_status_t>(status);
            }
            else { result = sz_status_unknown_k; }
#else
            result = sz_status_unknown_k; // GPU support is not enabled
#endif // SZ_USE_CUDA
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = static_cast<sz_status_t>(status);
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                sz::status_t status = engine_variant(          //
                    a_container, b_container, results_strided, //
                    get_executor(device_scope), get_specs(device_scope));
                result = static_cast<sz_status_t>(status);
            }
            else { result = sz_status_unknown_k; }
        }
    };

    std::visit(variant_logic, engine->variants);
    return result;
}

template <typename element_type_>
using vec = szs::safe_vector<element_type_, std::allocator<element_type_>>;

static constexpr size_t fingerprint_slice_k = 64;

struct fingerprints_backends_t {
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
    fingerprints_backends_t(variants_arguments_ &&...args) noexcept
        : variants(std::forward<variants_arguments_>(args)...) {}
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
    auto *engine = reinterpret_cast<fingerprints_backends_t *>(engine_punned);
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
    using fallback_variant_t = typename fingerprints_backends_t::fallback_variant_t;
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

#pragma region Metadata

SZ_DYNAMIC int szs_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_DYNAMIC int szs_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_DYNAMIC int szs_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }

SZ_DYNAMIC sz_capability_t szs_capabilities(void) {
    sz_capability_t cpu_capabilities = sz_capabilities_implementation_();
#if SZ_USE_CUDA
    return static_cast<sz_capability_t>(cpu_capabilities | sz_caps_ckh_k);
#else
    return cpu_capabilities;
#endif // SZ_USE_CUDA
}

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

#pragma endregion Metadata

#pragma region Device Scopes

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

#pragma endregion Device Scopes

#pragma region Levenshtein Distances

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_init(                                              //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    sz_levenshtein_distances_t *engine_punned) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const substitution_costs = szs::uniform_substitution_costs_t {match, mismatch};
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};

#if SZ_USE_ICE
    bool const can_use_ice = (capabilities & sz_cap_serial_k) == sz_cap_serial_k;
    if (can_use_ice && can_use_linear_costs) {
        auto variant = szs::levenshtein_ice_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_ice_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
    else if (can_use_ice) {
        auto variant = szs::affine_levenshtein_ice_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_ice_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_ICE

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) == sz_cap_cuda_k;
    if (can_use_cuda && can_use_linear_costs) {
        auto variant = szs::levenshtein_cuda_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_cuda_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
    else if (can_use_cuda) {
        auto variant = szs::affine_levenshtein_cuda_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_cuda_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_CUDA

#if SZ_USE_KEPLER
    bool const can_use_kepler = (capabilities & sz_cap_ck_k) == sz_cap_ck_k;
    if (can_use_kepler && can_use_linear_costs) {
        auto variant = szs::levenshtein_kepler_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_kepler_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
    else if (can_use_kepler) {
        auto variant = szs::affine_levenshtein_kepler_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_kepler_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_KEPLER

#if SZ_USE_HOPPER
    bool const can_use_hopper = (capabilities & sz_caps_ckh_k) == sz_caps_ckh_k;
    if (can_use_hopper && can_use_linear_costs) {
        auto variant = szs::levenshtein_hopper_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_hopper_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
    else if (can_use_hopper) {
        auto variant = szs::affine_levenshtein_hopper_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_hopper_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_HOPPER

    if (can_use_linear_costs) {
        auto variant = szs::levenshtein_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_serial_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
    else {
        auto variant = szs::affine_levenshtein_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_serial_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_t>(engine);
        return sz_success_k;
    }
}

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_sequence(                      //
    sz_levenshtein_distances_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                            //
    sz_size_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return sz_levenshtein_distances_for_(                       //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u32tape(                       //
    sz_levenshtein_distances_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,            //
    sz_size_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return sz_levenshtein_distances_for_(                       //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u64tape(                       //
    sz_levenshtein_distances_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,            //
    sz_size_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return sz_levenshtein_distances_for_(                       //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC void sz_levenshtein_distances_free(sz_levenshtein_distances_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<levenshtein_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Levenshtein Distances

#pragma region Levenshtein UTF8 Distances

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_init(                                         //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    sz_levenshtein_distances_utf8_t *engine_punned) {

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
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_utf8_t>(engine);
        return sz_success_k;
    }
    else {
        auto variant = szs::affine_levenshtein_utf8_ice_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_utf8_backends_t(std::in_place_type_t<szs::affine_levenshtein_utf8_ice_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_utf8_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_ICE

    bool const can_use_serial = (capabilities & sz_cap_serial_k) == sz_cap_serial_k;
    if (can_use_serial && can_use_linear_costs) {
        auto variant = szs::levenshtein_utf8_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_utf8_backends_t(std::in_place_type_t<szs::levenshtein_utf8_serial_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_utf8_t>(engine);
        return sz_success_k;
    }
    else {
        auto variant = szs::affine_levenshtein_utf8_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) levenshtein_utf8_backends_t(
            std::in_place_type_t<szs::affine_levenshtein_utf8_serial_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_levenshtein_distances_utf8_t>(engine);
        return sz_success_k;
    }

    return sz_status_unknown_k; // No supported backends available
}

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_sequence(                      //
    sz_levenshtein_distances_utf8_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                                 //
    sz_size_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return sz_levenshtein_distances_utf8_for_(                  //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_u32tape(                       //
    sz_levenshtein_distances_utf8_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,                 //
    sz_size_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return sz_levenshtein_distances_utf8_for_(                  //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_u64tape(                       //
    sz_levenshtein_distances_utf8_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,                 //
    sz_size_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return sz_levenshtein_distances_utf8_for_(                  //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC void sz_levenshtein_distances_utf8_free(sz_levenshtein_distances_utf8_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<levenshtein_utf8_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Levenshtein UTF8 Distances

#pragma region Needleman Wunsch

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_init(                        //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,          //
    sz_needleman_wunsch_scores_t *engine_punned) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const substitution_costs = *reinterpret_cast<szs::error_costs_256x256_t const *>(subs);
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};

#if SZ_USE_ICE
    bool const can_use_ice = (capabilities & sz_cap_serial_k) == sz_cap_serial_k;
    if (can_use_ice && can_use_linear_costs) {
        auto variant = szs::needleman_wunsch_ice_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            needleman_wunsch_scores_backends_t(std::in_place_type_t<szs::needleman_wunsch_ice_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_needleman_wunsch_scores_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_ICE

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) != 0;
    if (can_use_cuda && can_use_linear_costs) {
        auto variant = szs::needleman_wunsch_cuda_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow) needleman_wunsch_scores_backends_t(
            std::in_place_type_t<szs::needleman_wunsch_cuda_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_needleman_wunsch_scores_t>(engine);
        return sz_success_k;
    }
    else if (can_use_cuda) {
        auto variant = affine_needleman_wunsch_cuda_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) needleman_wunsch_scores_backends_t(
            std::in_place_type_t<affine_needleman_wunsch_cuda_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_needleman_wunsch_scores_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_CUDA

#if SZ_USE_HOPPER
    bool const can_use_hopper = (capabilities & sz_caps_ckh_k) == sz_caps_ckh_k;
    if (can_use_hopper && can_use_linear_costs) {
        auto variant = szs::needleman_wunsch_hopper_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow) needleman_wunsch_scores_backends_t(
            std::in_place_type_t<szs::needleman_wunsch_hopper_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_needleman_wunsch_scores_t>(engine);
        return sz_success_k;
    }
    else if (can_use_hopper) {
        auto variant = szs::affine_needleman_wunsch_hopper_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) needleman_wunsch_scores_backends_t(
            std::in_place_type_t<szs::affine_needleman_wunsch_hopper_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_needleman_wunsch_scores_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_HOPPER

    if (can_use_linear_costs) {
        auto variant = szs::needleman_wunsch_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow) needleman_wunsch_scores_backends_t(
            std::in_place_type_t<szs::needleman_wunsch_serial_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_needleman_wunsch_scores_t>(engine);
        return sz_success_k;
    }
    else {
        auto variant = szs::affine_needleman_wunsch_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) needleman_wunsch_scores_backends_t(
            std::in_place_type_t<szs::affine_needleman_wunsch_serial_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        *engine_punned = reinterpret_cast<sz_needleman_wunsch_scores_t>(engine);
        return sz_success_k;
    }
}

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_sequence(                      //
    sz_needleman_wunsch_scores_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                              //
    sz_ssize_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return sz_needleman_wunsch_scores_for_(                     //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u32tape(                       //
    sz_needleman_wunsch_scores_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,              //
    sz_ssize_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return sz_needleman_wunsch_scores_for_(                     //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u64tape(                       //
    sz_needleman_wunsch_scores_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,              //
    sz_ssize_t *results, sz_size_t results_stride) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return sz_needleman_wunsch_scores_for_(                     //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride);
}

SZ_DYNAMIC void sz_needleman_wunsch_scores_free(sz_needleman_wunsch_scores_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<needleman_wunsch_scores_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Needleman Wunsch

#pragma region Fingerprints

SZ_DYNAMIC sz_status_t sz_fingerprints_init(                              //
    sz_size_t alphabet_size, sz_size_t const *window_widths,              //
    sz_size_t window_widths_count, sz_size_t dimensions_per_window_width, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,     //
    sz_fingerprints_t *engine_punned) {

    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If window widths are not provided, let's pick some of the default configurations.
    auto const dimensions = window_widths_count * dimensions_per_window_width;
    auto const can_use_sliced_sketchers = dimensions_per_window_width % fingerprint_slice_k == 0;
    using fallback_variant_t = typename fingerprints_backends_t::fallback_variant_t;

    if (!can_use_sliced_sketchers) {
        auto variant = fallback_variant_t();
        for (size_t window_width_index = 0; window_width_index < window_widths_count; ++window_width_index) {
            auto const window_width = window_widths[window_width_index];
            auto extend_status = variant.try_extend(window_width, dimensions_per_window_width, alphabet_size);
            if (extend_status != sz::status_t::success_k) return static_cast<sz_status_t>(extend_status);
        }

        auto engine =
            new (std::nothrow) fingerprints_backends_t(std::in_place_type_t<fallback_variant_t>(), std::move(variant));
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

SZ_DYNAMIC sz_status_t sz_fingerprints_u32tape(                       //
    sz_fingerprints_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *texts,                               //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                //
    sz_u32_t *min_counts, sz_size_t min_counts_stride) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_u32tape_as_cpp_container_t {texts};
    return sz_fingerprints_for_(                       //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride);
}

SZ_DYNAMIC sz_status_t sz_fingerprints_u64tape(                       //
    sz_fingerprints_t engine_punned, sz_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *texts,                               //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                //
    sz_u32_t *min_counts, sz_size_t min_counts_stride) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_u64tape_as_cpp_container_t {texts};
    return sz_fingerprints_for_(                       //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride);
}

SZ_DYNAMIC void sz_fingerprints_free(sz_fingerprints_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<fingerprints_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Fingerprints

} // extern "C"