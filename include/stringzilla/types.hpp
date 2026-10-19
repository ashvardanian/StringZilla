/**
 *  @file include/stringzilla/types.hpp
 *  @author Ash Vardanian
 *  @date March 28, 2025
 *  @brief Shared definitions for the StringZilla C++ library.
 *
 *  The goal for this header is to provide an absolutely-minimal set of types and
 *  forward-declarations for CPU and GPU backends of higher-level complex templated algorithms
 *  implemented outside of the C layer. It includes the following primitive type aliases for
 *  the `types.h` header:
 *
 *  - @c u8_t, @c u16_t, @c u32_t, @c u64_t, @c i8_t, @c i16_t, @c i32_t, @c i64_t - sized integers.
 *  - @c ssize_t, @c ptr_t, @c cptr_t - address-related types.
 *  - @c status_t, @c bool_t, @c ordering_t, @c rune_t, @c rune_length_t, @c error_cost_t - logic.
 *
 *  The header also provides @c strings_view_t, a borrowed column of strings in any layout of
 *  @c sz_strings_view_t, and these higher-level types:
 *
 *  - `std::span<value_type>` - a view to a contiguous memory block of @c value_type elements.
 *  - `dummy_alloc<value_type>` - a dummy memory allocator shaped like @c std::allocator.
 *  - `vector<value_type, allocator_type>` - owning storage with fallible allocation.
 *  - `strings<allocator_type>` - an owned column of strings, an @c sz_strings_span_t tape.
 */
#ifndef STRINGZILLA_TYPES_HPP_
#define STRINGZILLA_TYPES_HPP_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capabilities_enabled_cpu`, `sz_capabilities_enabled_cuda`
#include "stringzilla/strings.h"

/** When set to 1, the library will include the C++ STL headers and implement automatic conversion
 *  from and to @c std::string_view and `std::basic_string<any_allocator>`. */
#if !defined(STRINGZILLA_WITH_STL)
#define STRINGZILLA_WITH_STL (1) // true or false
#endif

/*  MSVC pins @c __cplusplus at 199711L unless `/Zc:__cplusplus` is passed, so read
 *  @c _MSVC_LANG there. */
#if (defined(_MSVC_LANG) ? _MSVC_LANG : __cplusplus) < 202002L
#error "StringZilla's C++ interface requires C++20; the C interface in `stringzilla.h` has no such floor."
#endif

#if defined(_MSC_VER)
#define STRINGZILLA_INLINE_ __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define STRINGZILLA_INLINE_ inline __attribute__((always_inline))
#else
#define STRINGZILLA_INLINE_ inline
#endif

#if defined(_MSC_VER)
#define STRINGZILLA_NOINLINE_ __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define STRINGZILLA_NOINLINE_ __attribute__((noinline))
#else
#define STRINGZILLA_NOINLINE_
#endif

/** @c noipa disables interprocedural optimization across the function boundary - strictly stronger
 *  than @c noinline, which only forbids inlining. It is the durable guard against GCC's @c -O3
 *  miscompile of a by-value return that gets folded into a same-TU caller, where a @c status_t
 *  engine return reads back as garbage when the callee is folded into a large benchmark TU. Clang
 *  has no such miscompile and no @c noipa attribute, and NVCC rejects it on device code, so both
 *  fall back to @c noinline. */
#if defined(__GNUC__) && !defined(__clang__) && !STRINGZILLA_ARCH_CUDA_
#define STRINGZILLA_NOIPA_ __attribute__((noipa))
#else
#define STRINGZILLA_NOIPA_ STRINGZILLA_NOINLINE_
#endif

#if STRINGZILLA_WITH_STL
#include <initializer_list> // `std::initializer_list` is only ~100 LOC
#include <iterator>         // `std::random_access_iterator_tag` pulls 20K LOC
#include <limits>           // `std::numeric_limits`
#include <memory>           // `std::allocator_traits` for allocator rebinding
#include <new>              // Placement construction
#include <span>             // `std::span`
#include <type_traits>      // `std::is_const_v`, `std::is_arithmetic_v`, `std::is_trivially_destructible`
#include <utility>          // `std::move`
#endif

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief Forces the compiler to materialize @p value rather than fold or elide it across an
 *      optimization boundary - the read-side companion to @c STRINGZILLA_NOIPA_.
 *
 *  Mirrors Google Benchmark's @c DoNotOptimize; use it on a status/result just before
 *  returning it from a @c STRINGZILLA_NOIPA_ public entry point so the value survives a
 *  same-TU caller's interprocedural rewrite.
 *
 *  @sa sz_keep_alive_ in `types.h`, the C-callable sibling that pins stores to a buffer rather than
 *      a value, for the C headers, which cannot instantiate a template.
 */
template <typename value_type_>
STRINGZILLA_INLINE_ void sz_do_not_optimize(value_type_ &value) noexcept {
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#elif defined(__GNUC__)
    asm volatile("" : "+m,r"(value) : : "memory");
#else
    volatile value_type_ sink = value;
    sz_unused_(sink);
#endif
}

using i8_t = sz_i8_t;
using u8_t = sz_u8_t;
using i16_t = sz_i16_t;
using u16_t = sz_u16_t;
using i32_t = sz_i32_t;
using u32_t = sz_u32_t;
using u64_t = sz_u64_t;
using i64_t = sz_i64_t;

using byte_t = sz_byte_t;
using rune_t = sz_rune_t;

using ssize_t = sz_ssize_t;

/** A size or offset deliberately held in 32 bits, where the narrower arithmetic is cheaper -
 *  GPU address math above all. Every use pairs with a range check at the site that
 *  establishes the bound. */
using small_size_t = sz_u32_t;

using f32_t = float;
using f64_t = double;

using ptr_t = sz_ptr_t;
using cptr_t = sz_cptr_t;
using error_cost_t = sz_error_cost_t;
using error_cost_magnitude_t = sz_error_cost_magnitude_t;

using bool_t = sz_bool_t;
using ordering_t = sz_ordering_t;
using rune_length_t = sz_rune_length_t;

using u16_vec_t = sz_u16_vec_t;
using u32_vec_t = sz_u32_vec_t;
using u64_vec_t = sz_u64_vec_t;
using u128_vec_t = sz_u128_vec_t;
using u256_vec_t = sz_u256_vec_t;
using u512_vec_t = sz_u512_vec_t;

#pragma region Status

/**
 *  @brief Why a call failed, in the vocabulary the C ABI and every binding share.
 *
 *  Every enumerator takes its value from @c sz_status_t in @c types.h, so the C++ mirror and the
 *  ABI cannot drift, and the crossing is a cast rather than a table.
 */
enum class [[nodiscard]] status_t : int {

    /** Finished without error. */
    success_k = sz_success_k,

    /** An allocation failed. */
    bad_alloc_k = sz_bad_alloc_k,

    /** The input is not valid UTF-8. */
    invalid_utf8_k = sz_invalid_utf8_k,

    /** A collection that must hold unique elements holds duplicates. */
    contains_duplicates_k = sz_contains_duplicates_k,

    /** The input is too large for the counters or the offsets holding it. */
    overflow_risk_k = sz_overflow_risk_k,

    /** Operand sizes contradict each other, like an output span shorter than the input. */
    unexpected_dimensions_k = sz_unexpected_dimensions_k,

    /** No GPU of the vendor this build targets answers. */
    missing_gpu_k = sz_missing_gpu_k,

    /** The device code lacks the kernel, or it failed to build, launch or finish. */
    device_code_mismatch_k = sz_device_code_mismatch_k,

    /** An operand lies in memory the device cannot address. */
    device_memory_mismatch_k = sz_device_memory_mismatch_k,

    /** An authenticated decryption saw a tag that does not match its ciphertext. */
    authentication_failed_k = sz_authentication_failed_k,

    /** No capability in the capability mask has this kernel. */
    missing_kernel_k = sz_missing_kernel_k,

    /** A dispatch point or finder called from a header-only build, which links no library. */
    missing_library_k = sz_missing_library_k,

    /** A failure no other status describes. */
    unknown_k = sz_status_unknown_k,
};

/** Whether @p status reports success. */
constexpr bool succeeded(status_t status) noexcept { return status == status_t::success_k; }

/** Whether @p status reports failure. */
constexpr bool failed(status_t status) noexcept { return status != status_t::success_k; }

/** Static, English description of @p status. Never returns @c nullptr, never allocates. */
inline char const *status_name(status_t status) noexcept { return sz_status_name_(static_cast<sz_status_t>(status)); }

/** A result paired with the @c status_t explaining it; the value is only meaningful
 *  on @c success_k. */
template <typename value_type_>
struct [[nodiscard]] expected {
    value_type_ value;
    status_t status;

    explicit operator bool() const noexcept { return succeeded(status); }
};

/** The borrowing face of @ref expected for reference results: there is no null
 *  reference, so the borrow is stored as an address and @c value() binds it only on
 *  @c success_k. */
template <typename value_type_>
struct [[nodiscard]] expected<value_type_ &> {
    value_type_ *borrowed;
    status_t status;

    value_type_ &value() const noexcept { return *borrowed; }
    explicit operator bool() const noexcept { return succeeded(status); }
};

#pragma endregion Status

#pragma region Devices

/** Which runtime a device belongs to, as the `sz_<kind>_*` C functions name it. */
enum class device_kind_t : int { cpu_k, cuda_k, rocm_k, metal_k };

/** One device StringZilla runs kernels on: the host CPU, or a GPU by its runtime's own ordinal. */
class device_t {
    device_kind_t kind_;
    std::size_t ordinal_;

    constexpr device_t(device_kind_t kind, std::size_t ordinal) noexcept : kind_(kind), ordinal_(ordinal) {}

  public:
    /** The host CPU, which every build has. */
    static constexpr device_t cpu() noexcept { return {device_kind_t::cpu_k, 0}; }

    /** How many devices of @p kind the process sees: one CPU, or @c missing_gpu_k and zero GPUs. */
    static expected<std::size_t> count(device_kind_t kind) noexcept {
        sz_size_t count = 1;
        sz_status_t status = sz_success_k;
        switch (kind) {
        case device_kind_t::cpu_k: break;
        case device_kind_t::cuda_k: status = sz_device_count_cuda(&count); break;
        case device_kind_t::rocm_k: status = sz_device_count_rocm(&count); break;
        case device_kind_t::metal_k: status = sz_device_count_metal(&count); break;
        }
        return {static_cast<std::size_t>(count), static_cast<status_t>(status)};
    }

    /** Device @p ordinal of @p kind, or @c missing_gpu_k past the last one. */
    static expected<device_t> make(device_kind_t kind, std::size_t ordinal) noexcept {
        expected<std::size_t> const devices = count(kind);
        status_t const status = devices && ordinal >= devices.value ? status_t::missing_gpu_k : devices.status;
        return {device_t {kind, ordinal}, status};
    }

    constexpr device_kind_t kind() const noexcept { return kind_; }
    constexpr std::size_t ordinal() const noexcept { return ordinal_; }

    /** What this device runs, whether or not this binary holds kernels for it; zero on failure. */
    expected<sz_capability_t> capabilities_detected() const noexcept {
        sz_capability_t capabilities = 0;
        sz_status_t status = sz_success_k;
        switch (kind_) {
        case device_kind_t::cpu_k: status = sz_capabilities_detected_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = sz_capabilities_detected_cuda(ordinal_, &capabilities); break;
        case device_kind_t::rocm_k: status = sz_capabilities_detected_rocm(ordinal_, &capabilities); break;
        case device_kind_t::metal_k: status = sz_capabilities_detected_metal(ordinal_, &capabilities); break;
        }
        return {capabilities, static_cast<status_t>(status)};
    }

    /** What this binary holds kernels for on devices of this kind, whether or not this one runs. */
    sz_capability_t capabilities_compiled() const noexcept {
        sz_capability_t capabilities = 0;
        [[maybe_unused]] sz_status_t status = sz_success_k; // Never fails: the mask is fixed at build time
        switch (kind_) {
        case device_kind_t::cpu_k: status = sz_capabilities_compiled_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = sz_capabilities_compiled_cuda(&capabilities); break;
        case device_kind_t::rocm_k: status = sz_capabilities_compiled_rocm(&capabilities); break;
        case device_kind_t::metal_k: status = sz_capabilities_compiled_metal(&capabilities); break;
        }
        return capabilities;
    }

    /** Both at once: the mask for this device's dispatch points; zero on failure. */
    expected<sz_capability_t> capabilities_enabled() const noexcept {
        sz_capability_t capabilities = 0;
        sz_status_t status = sz_success_k;
        switch (kind_) {
        case device_kind_t::cpu_k: status = sz_capabilities_enabled_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = sz_capabilities_enabled_cuda(ordinal_, &capabilities); break;
        case device_kind_t::rocm_k: status = sz_capabilities_enabled_rocm(ordinal_, &capabilities); break;
        case device_kind_t::metal_k: status = sz_capabilities_enabled_metal(ordinal_, &capabilities); break;
        }
        return {capabilities, static_cast<status_t>(status)};
    }

    /** Prepares the calling thread for the CPU kernels of @p capabilities; GPUs have no such
     *  kernels, so they report @c missing_kernel_k. */
    status_t configure_thread(sz_capability_t capabilities) const noexcept {
        if (kind_ != device_kind_t::cpu_k) return status_t::missing_kernel_k;
        return static_cast<status_t>(sz_thread_configure_cpu(capabilities));
    }
};

/** The mask every wrapper dispatches over by default: the CPU's enabled capabilities, as
 *  @c device_t::cpu().capabilities_enabled() reports them, or none in header-only builds, whose
 *  dispatch points are stubs, so every wrapper runs the serial reference instead. */
inline sz_capability_t default_capabilities() noexcept {
#if STRINGZILLA_HEADER_ONLY
    return 0;
#else
    sz_capability_t capabilities = 0;
    return sz_capabilities_enabled_cpu(&capabilities) == sz_success_k ? capabilities : sz_cap_serial_k;
#endif
}

/** Calls the dispatch point @p best_ over a non-zero @p capabilities, or its serial twin @p serial_
 *  over an empty one, the mask of header-only builds, whose dispatch points are stubs. */
template <auto best_, auto serial_, typename... arguments_types_>
sz_status_t dispatch_(sz_capability_t capabilities, sz_stream_t stream, arguments_types_... arguments) noexcept {
    return capabilities ? best_(arguments..., capabilities, stream) : serial_(arguments..., stream);
}

#pragma endregion Devices

/**
 *  @brief A trivial function object for uniform character substitution costs in
 *      Levenshtein-like similarity algorithms.
 *
 *  @sa error_costs_32x32_t
 */
struct error_costs_unary_t {
    constexpr error_cost_t operator()(char a, char b) const noexcept { return a == b ? 0 : 1; }
    constexpr error_cost_t operator()(sz_rune_t a, sz_rune_t b) const noexcept { return a == b ? 0 : 1; }
    constexpr error_cost_magnitude_t magnitude() const noexcept { return 1; }
};

template <typename value_type_>
struct dummy_alloc {
    using value_type = value_type_;     // ? For STL compatibility
    using pointer = value_type *;       // ? For STL compatibility
    using size_type = std::size_t;      // ? For STL compatibility
    using difference_type = sz_ssize_t; // ? For STL compatibility

    template <typename other_value_type_>
    struct rebind {
        using other = dummy_alloc<other_value_type_>;
    };

    constexpr dummy_alloc() noexcept = default;
    constexpr dummy_alloc(dummy_alloc const &) noexcept = default;

    template <typename other_value_type_>
    constexpr dummy_alloc(dummy_alloc<other_value_type_> const &) noexcept {}

    constexpr value_type *allocate(size_type) const noexcept { return nullptr; }
    constexpr void deallocate(pointer, size_type) const noexcept {}

    template <typename other_type_>
    constexpr bool operator==(dummy_alloc<other_type_> const &) const noexcept {
        return true;
    }

    template <typename other_type_>
    constexpr bool operator!=(dummy_alloc<other_type_> const &) const noexcept {
        return false;
    }
};

using dummy_alloc_t = dummy_alloc<char>;

/** Nullable heap allocator using the C allocation policy. */
template <typename value_type_>
struct heap_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = std::size_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename other_value_type_>
    struct rebind {
        using other = heap_alloc<other_value_type_>;
    };

    constexpr heap_alloc() noexcept = default;
    template <typename other_value_type_>
    constexpr heap_alloc(heap_alloc<other_value_type_> const &) noexcept {}

    pointer allocate(size_type count) const noexcept {
        count = count ? count : 1;
        if (count > (std::numeric_limits<size_type>::max)() / sizeof(value_type)) return nullptr;
        sz_allocator_t allocator {};
        if (sz_allocator_init_heap(&allocator) != sz_success_k) return nullptr;
        pointer result = static_cast<pointer>(
            allocator.allocate(count * sizeof(value_type), allocator.handle, nullptr));
        if (!result || reinterpret_cast<sz_size_t>(result) % alignof(value_type) == 0) return result;
        allocator.free(result, count * sizeof(value_type), allocator.handle, nullptr);
        return nullptr;
    }
    void deallocate(pointer start, size_type count) const noexcept {
        if (!start) return;
        sz_allocator_t allocator {};
        if (sz_allocator_init_heap(&allocator) != sz_success_k) return;
        allocator.free(start, (count ? count : 1) * sizeof(value_type), allocator.handle, nullptr);
    }
    template <typename other_value_type_>
    constexpr bool operator==(heap_alloc<other_value_type_> const &) const noexcept {
        return true;
    }
};

/**
 *  @brief Random access iterator for any immutable container with indexed element lookup support.
 *
 *  @note Designed for @c strings_view_t and @c strings compatibility with STL
 *      algorithms and ranges.
 */
template <typename container_type_>
struct indexed_container_iterator {
    using container_t = container_type_;
    using value_t = typename container_t::value_type;

    using difference_type = sz_ssize_t;
    using value_type = value_t;
    using reference = value_t; // ! As our view returns by value
    using pointer = void;      // ! Not providing direct pointer semantics

#if STRINGZILLA_WITH_STL
    using iterator_category = std::random_access_iterator_tag;
#endif

  private:
    container_t const *parent_;
    std::size_t index_;

  public:
    constexpr indexed_container_iterator() noexcept : parent_(nullptr), index_(0) {}
    constexpr indexed_container_iterator(container_t const &parent, std::size_t index) noexcept
        : parent_(&parent), index_(index) {}
    constexpr reference operator*() const noexcept { return (*parent_)[index_]; }

    struct proxy {
        value_type value;
        constexpr proxy(value_type v) noexcept : value(v) {}
        constexpr value_type const *operator->() const noexcept { return &value; }
    };

    constexpr proxy operator->() const noexcept { return proxy(operator*()); }
    constexpr indexed_container_iterator &operator++() noexcept {
        ++index_;
        return *this;
    }

    constexpr indexed_container_iterator operator++(int) noexcept {
        indexed_container_iterator temp = *this;
        ++index_;
        return temp;
    }

    constexpr indexed_container_iterator &operator--() noexcept {
        --index_;
        return *this;
    }

    constexpr indexed_container_iterator operator--(int) noexcept {
        indexed_container_iterator temp = *this;
        --index_;
        return temp;
    }

    constexpr indexed_container_iterator &operator+=(difference_type n) noexcept {
        index_ += n;
        return *this;
    }

    constexpr indexed_container_iterator &operator-=(difference_type n) noexcept {
        index_ -= n;
        return *this;
    }

    constexpr indexed_container_iterator operator+(difference_type n) const noexcept {
        indexed_container_iterator temp = *this;
        return temp += n;
    }

    constexpr indexed_container_iterator operator-(difference_type n) const noexcept {
        indexed_container_iterator temp = *this;
        return temp -= n;
    }

    constexpr difference_type operator-(indexed_container_iterator const &other) const noexcept {
        return static_cast<difference_type>(index_) - static_cast<difference_type>(other.index_);
    }

    constexpr value_type operator[](difference_type n) const noexcept { return *(*this + n); }

    friend constexpr bool operator==(indexed_container_iterator const &lhs,
                                     indexed_container_iterator const &rhs) noexcept {
        return lhs.parent_ == rhs.parent_ && lhs.index_ == rhs.index_;
    }

    friend constexpr bool operator!=(indexed_container_iterator const &lhs,
                                     indexed_container_iterator const &rhs) noexcept {
        return !(lhs == rhs);
    }

    friend constexpr bool operator<(indexed_container_iterator const &lhs,
                                    indexed_container_iterator const &rhs) noexcept {
        return lhs.index_ < rhs.index_;
    }

    friend constexpr bool operator>(indexed_container_iterator const &lhs,
                                    indexed_container_iterator const &rhs) noexcept {
        return rhs < lhs;
    }

    friend constexpr bool operator<=(indexed_container_iterator const &lhs,
                                     indexed_container_iterator const &rhs) noexcept {
        return !(rhs < lhs);
    }

    friend constexpr bool operator>=(indexed_container_iterator const &lhs,
                                     indexed_container_iterator const &rhs) noexcept {
        return !(lhs < rhs);
    }
};

#pragma region String Collections

/** Allocates @p n bytes through the C++ allocator behind @p allocator_state, reading its throw as
 *  null, so the C side reports @c bad_alloc_k rather than unwinding through its frames. */
template <typename allocator_type_>
inline void *allocate_through_(sz_size_t n, void *allocator_state, void *) noexcept {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
    try {
        return std::allocator_traits<allocator_type_>::allocate(*reinterpret_cast<allocator_type_ *>(allocator_state),
                                                                n);
    }
    catch (...) {
        return nullptr;
    }
#else
    return std::allocator_traits<allocator_type_>::allocate(*reinterpret_cast<allocator_type_ *>(allocator_state), n);
#endif
}

template <typename allocator_type_>
inline void free_through_(void *ptr, sz_size_t n, void *allocator_state, void *) noexcept {
    using value_type_ = typename allocator_type_::value_type;
    return reinterpret_cast<allocator_type_ *>(allocator_state)->deallocate(reinterpret_cast<value_type_ *>(ptr), n);
}

/**
 *  @brief Lends the byte @p allocator to @p callback as a C-style @c sz_allocator_t.
 *
 *  A C++ allocator cannot resize a block where it lies, so @c reallocate stays null, as the C
 *  contract allows, and the C side moves a block that must change size through @c allocate and
 *  @c free.
 *
 *  @return The status the callback reports.
 */
template <typename allocator_type_, typename allocator_callback_>
inline status_t with_allocator_(allocator_type_ &allocator, allocator_callback_ &&callback) noexcept {
    sz_allocator_t c_allocator;
    c_allocator.allocate = &allocate_through_<allocator_type_>;
    c_allocator.reallocate = nullptr;
    c_allocator.free = &free_through_<allocator_type_>;
    c_allocator.handle = &allocator;
    return static_cast<status_t>(callback(c_allocator));
}

template <typename>
class string_slice;
template <typename>
class string;
class byteset_t;
class look_up_table_t;

using string_span_t = string_slice<char>;
using string_view_t = string_slice<char const>;

template <typename allocator_type_>
struct strings;

/**
 *  @brief A borrowed column in any layout of @c sz_strings_view_t, the twin of @c string_view_t.
 *
 *  Every conversion reads the whole column into a fresh @c strings, whose blocks come from
 *  the given allocator, and leaves it empty on failure.
 */
struct strings_view_t {
    using value_type = string_view_t;
    using iterator = indexed_container_iterator<strings_view_t>;

    /** The C view every member reads. */
    sz_strings_view_t raw;

    /** An empty column. */
    strings_view_t() noexcept : raw {} {}

    /** Wraps a C view, like one over records or NUL-padded rows. */
    explicit strings_view_t(sz_strings_view_t raw) noexcept : raw(raw) {}

    /** An Apache Arrow @c String column: string i is `[offsets[i], offsets[i + 1])` of @p data. */
    strings_view_t(char const *data, sz_u32_t const *offsets, std::size_t count) noexcept : raw {} {
        raw.layout = sz_strings_u32_tape_k, raw.count = count;
        raw.data.u32_tape.data = data, raw.data.u32_tape.offsets = offsets;
    }

    /** An Apache Arrow @c LargeString column, the same with 64-bit offsets. */
    strings_view_t(char const *data, sz_u64_t const *offsets, std::size_t count) noexcept : raw {} {
        raw.layout = sz_strings_u64_tape_k, raw.count = count;
        raw.data.u64_tape.data = data, raw.data.u64_tape.offsets = offsets;
    }

    /** Views @p strings through an accessor, copying no bytes, so they must outlive the view. */
    strings_view_t(std::span<string_view_t const> strings) noexcept;

    /** Views the tape of @p strings. */
    template <typename allocator_type_>
    strings_view_t(strings<allocator_type_> const &strings) noexcept : strings_view_t(strings.view()) {}

    std::size_t size() const noexcept { return raw.count; }
    string_view_t operator[](std::size_t index) const noexcept;
    iterator begin() const noexcept { return {*this, 0}; }
    iterator end() const noexcept { return {*this, size()}; }

    /**
     *  @brief Finds @p needle in every string, like a @c LIKE '%needle%' filter over a column.
     *  @param[out] positions The offset of each string's first match, or @c STRINGZILLA_SIZE_MAX.
     *  @return @c success_k, or @c unexpected_dimensions_k when @p positions is too short.
     *  @sa sz_find_strings_best
     */
    status_t find(string_view_t needle, std::span<std::size_t> positions,
                  sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) const noexcept;

    /**
     *  @brief Hashes every string under one @p seed, each as @c string_view_t::hash would.
     *  @return @c success_k, or @c unexpected_dimensions_k when @p hashes is too short.
     *  @sa sz_hash_strings_best
     */
    status_t hashes(std::span<u64_t> hashes, u64_t seed = 0, sz_capability_t capabilities = default_capabilities(),
                    sz_stream_t stream = nullptr) const noexcept;

    /**
     *  @brief Validates every string as UTF-8 and, unless @p runes is empty, counts its runes.
     *  @param[out] runes Empty, or each string's runes, unspecified from the first ill-formed one.
     *  @param[out] malformed_index The first ill-formed string, or @c size() if none is ill-formed.
     *  @return @c success_k, @c invalid_utf8_k when a string is ill-formed, or
     *      @c unexpected_dimensions_k when @p runes is too short.
     *  @sa sz_utf8_count_strings_best
     */
    status_t utf8_count(std::span<std::size_t> runes, std::size_t &malformed_index,
                        sz_capability_t capabilities = default_capabilities(),
                        sz_stream_t stream = nullptr) const noexcept;

    /**
     *  @brief Writes every string into a NUL-padded row of @p width bytes, truncating longer ones.
     *  @return The length of the longest string, to retry with wider rows, or
     *      @c unexpected_dimensions_k when @p rows holds fewer than @c size() rows.
     *  @sa sz_strings_to_rows_best
     */
    expected<std::size_t> to_rows(std::span<char> rows, std::size_t width,
                                  sz_capability_t capabilities = default_capabilities(),
                                  sz_stream_t stream = nullptr) const noexcept {
        if (raw.count && rows.size() / raw.count < width) return {0, status_t::unexpected_dimensions_k};
        sz_strings_rows_t const destination {.data = rows.data(), .stride = width, .width = width};
        sz_size_t longest = 0;
        sz_status_t const status = dispatch_<sz_strings_to_rows_best, sz_strings_to_rows_serial>(
            capabilities, stream, &raw, nullptr, raw.count, &destination, &longest);
        return {longest, static_cast<status_t>(status)};
    }

    /**
     *  @brief Copies every string into one tape, whatever the layout.
     *  @sa sz_strings_to_tape_best
     */
    template <typename allocator_type_ = heap_alloc<char>>
    expected<strings<allocator_type_>> gather(allocator_type_ allocator = {},
                                              sz_capability_t capabilities = default_capabilities(),
                                              sz_stream_t stream = nullptr) const noexcept {
        expected<strings<allocator_type_>> result {strings<allocator_type_>(allocator), status_t::success_k};
        result.status = with_allocator_(result.value.allocator, [&](sz_allocator_t &c_allocator) {
            return dispatch_<sz_strings_to_tape_best, sz_strings_to_tape_serial>(
                capabilities, stream, &raw, nullptr, raw.count, &c_allocator, &result.value.raw);
        });
        return result;
    }

    /**
     *  @brief Escapes every string for @p kind, like JSON or CSV.
     *  @sa sz_strings_escape_best
     */
    template <typename allocator_type_ = heap_alloc<char>>
    expected<strings<allocator_type_>> escape(sz_escape_kind_t kind, allocator_type_ allocator = {},
                                              sz_capability_t capabilities = default_capabilities(),
                                              sz_stream_t stream = nullptr) const noexcept {
        expected<strings<allocator_type_>> result {strings<allocator_type_>(allocator), status_t::success_k};
        result.status = with_allocator_(result.value.allocator, [&](sz_allocator_t &c_allocator) {
            return dispatch_<sz_strings_escape_best, sz_strings_escape_serial>(
                capabilities, stream, &raw, nullptr, raw.count, kind, &c_allocator, &result.value.raw);
        });
        return result;
    }

    /**
     *  @brief Unescapes every string from @p kind, checking all of them before writing any.
     *  @param[out] malformed_index The first malformed string, set on @c invalid_utf8_k.
     *  @sa sz_strings_unescape_best
     */
    template <typename allocator_type_ = heap_alloc<char>>
    expected<strings<allocator_type_>> unescape(sz_escape_kind_t kind, std::size_t &malformed_index,
                                                allocator_type_ allocator = {},
                                                sz_capability_t capabilities = default_capabilities(),
                                                sz_stream_t stream = nullptr) const noexcept {
        expected<strings<allocator_type_>> result {strings<allocator_type_>(allocator), status_t::success_k};
        result.status = with_allocator_(result.value.allocator, [&](sz_allocator_t &c_allocator) {
            return dispatch_<sz_strings_unescape_best, sz_strings_unescape_serial>(capabilities, stream, &raw, nullptr,
                                                                                   raw.count, kind, &c_allocator,
                                                                                   &result.value.raw, &malformed_index);
        });
        return result;
    }

    /**
     *  @brief Maps every byte through @p table, like @c tolower does.
     *  @sa sz_strings_lookup_best
     */
    template <typename allocator_type_ = heap_alloc<char>>
    expected<strings<allocator_type_>> lookup(look_up_table_t const &table, allocator_type_ allocator = {},
                                              sz_capability_t capabilities = default_capabilities(),
                                              sz_stream_t stream = nullptr) const noexcept;

    /**
     *  @brief Brings every string into the Unicode @p form, like NFC.
     *  @sa sz_strings_normalize_best
     */
    template <typename allocator_type_ = heap_alloc<char>>
    expected<strings<allocator_type_>> normalize(sz_normal_form_t form, allocator_type_ allocator = {},
                                                 sz_capability_t capabilities = default_capabilities(),
                                                 sz_stream_t stream = nullptr) const noexcept {
        expected<strings<allocator_type_>> result {strings<allocator_type_>(allocator), status_t::success_k};
        result.status = with_allocator_(result.value.allocator, [&](sz_allocator_t &c_allocator) {
            return dispatch_<sz_strings_normalize_best, sz_strings_normalize_serial>(
                capabilities, stream, &raw, nullptr, raw.count, form, &c_allocator, &result.value.raw);
        });
        return result;
    }

    /**
     *  @brief Joins every string with @p between into one, like Python's `str.join`.
     *  @sa sz_strings_join_best
     */
    template <typename allocator_type_ = std::allocator<char>>
    expected<string<allocator_type_>> join(string_view_t between, allocator_type_ allocator = {},
                                           sz_capability_t capabilities = default_capabilities(),
                                           sz_stream_t stream = nullptr) const noexcept;

    /**
     *  @brief Writes row r of @p columns as
     *      `frames[0] + columns[0][r] + frames[1] + … + frames[n]`, with @p between between rows,
     *      like CSV lines or JSON arrays.
     *  @return @c unexpected_dimensions_k for uneven columns or when @p frames is not one longer.
     *  @sa sz_strings_join_best
     */
    template <typename allocator_type_ = std::allocator<char>>
    static expected<string<allocator_type_>> join(std::span<sz_strings_view_t const> columns,
                                                  std::span<sz_string_view_t const> frames, string_view_t between,
                                                  allocator_type_ allocator = {},
                                                  sz_capability_t capabilities = default_capabilities(),
                                                  sz_stream_t stream = nullptr) noexcept;
};

/**
 *  @brief An owned column: the C span and the allocator its blocks came from, the twin of
 *      @c string.
 *
 *  Only the conversions of @c strings_view_t and the splits fill one, so it never grows and moves
 *  without copies.
 *  Its offsets are 32-bit while the payload fits in 4 GiB and 64-bit beyond.
 */
template <typename allocator_type_ = heap_alloc<char>>
struct strings {
    using allocator_type = allocator_type_;
    using value_type = string_view_t;
    using iterator = indexed_container_iterator<strings>;

    /** The tape, zeroed while empty. */
    sz_strings_span_t raw {};

    /** Where both blocks of @c raw come from and return to. */
    allocator_type_ allocator {};

    strings() noexcept = default;
    explicit strings(allocator_type_ allocator) noexcept : allocator(allocator) {}
    strings(strings const &) = delete;
    strings &operator=(strings const &) = delete;
    strings(strings &&other) noexcept : raw(other.raw), allocator(std::move(other.allocator)) { other.raw = {}; }
    strings &operator=(strings &&other) noexcept {
        if (this == &other) return *this;
        release_();
        raw = other.raw, allocator = std::move(other.allocator);
        other.raw = {};
        return *this;
    }
    ~strings() noexcept { release_(); }

    std::size_t size() const noexcept { return raw.count; }
    string_view_t operator[](std::size_t index) const noexcept;
    iterator begin() const noexcept { return {*this, 0}; }
    iterator end() const noexcept { return {*this, size()}; }

    strings_view_t view() const noexcept {
        strings_view_t result;
        sz_strings_from_span(&raw, &result.raw);
        return result;
    }

    /**
     *  @brief Splits @p text at the bytes of @p delimiters into its parts, like `bytes.split`.
     *  @sa sz_strings_split_best
     */
    static expected<strings> split(string_view_t text, byteset_t const &delimiters, sz_split_kind_t kind,
                                   allocator_type_ allocator = {},
                                   sz_capability_t capabilities = default_capabilities(),
                                   sz_stream_t stream = nullptr) noexcept;

    /**
     *  @brief Splits @p text into lines without their breaks, like `str.splitlines`.
     *  @sa sz_strings_split_lines_best
     */
    static expected<strings> split_lines(string_view_t text, allocator_type_ allocator = {},
                                         sz_capability_t capabilities = default_capabilities(),
                                         sz_stream_t stream = nullptr) noexcept;

  private:
    void release_() noexcept {
        [[maybe_unused]] status_t const status = with_allocator_(allocator, [&](sz_allocator_t &c_allocator) {
            sz_strings_span_free(&raw, &c_allocator, nullptr);
            return sz_success_k;
        });
    }
};

/** The owned column on the heap, the twin of @c string_t. */
using strings_t = strings<>;

#pragma endregion String Collections

template <typename first_, typename second_>
struct is_same_type;

template <typename first_>
struct is_same_type<first_, first_> {
    static constexpr bool value = true;
};

template <typename first_, typename second_>
struct is_same_type {
    static constexpr bool value = false;
};

struct cpu_specs_t {
    std::size_t l1_bytes = 32 * 1024;       // ? typically around 32 KB
    std::size_t l2_bytes = 256 * 1024;      // ? typically around 256 KB
    std::size_t l3_bytes = 8 * 1024 * 1024; // ? typically around 8 MB
    std::size_t cache_line_width = 64;      // ? 64 bytes on x86, sometimes 128 on ARM
    std::size_t cores_per_socket = 1;       // ? at least 1 core
    std::size_t sockets = 1;                // ? at least 1 socket

    std::size_t cores_total() const noexcept { return cores_per_socket * sockets; }
};

/**
 *  @brief Specifications of a typical NVIDIA GPU, such as A100 or H100.
 *  @note We recommend compiling the code for the 90a compute capability, the newest
 *      with specialized optimizations.
 *  @sa pack_sm_code, cores_per_multiprocessor
 */
struct gpu_specs_t {
    std::size_t vram_bytes = 40ul * 1024 * 1024 * 1024; // ? On A100 it's 40 GB
    std::size_t l2_bytes = 40ul * 1024 * 1024;          // ? On A100 it's 40 MB, shared by every multiprocessor
    std::size_t constant_memory_bytes = 64 * 1024;      // ? On A100 it's 64 KB
    std::size_t shared_memory_bytes = 192 * 1024 * 108; // ? On A100 it's 192 KB per SM
    std::size_t streaming_multiprocessors = 108;        // ? On A100
    std::size_t cuda_cores = 6912;                      // ? On A100 for f32/i32 logic
    std::size_t reserved_memory_per_block = 1024;       // ? Typically, 1 KB per block is reserved for bookkeeping
    std::size_t warp_size = 32;                         // ? Warp size is 32 threads on practically all GPUs
    std::size_t max_blocks_per_multiprocessor = 0;      // ? Maximum number of blocks per SM
    std::size_t sm_code = 0;                            // ? Compute capability code, e.g. 90a for Hopper (H100)

    inline std::size_t shared_memory_per_multiprocessor() const noexcept {
        return shared_memory_bytes / streaming_multiprocessors;
    }

    /**
     *  @brief Converts a compute capability (major, minor) to a single numeric code.
     *
     *  - 7.0, 7.2 is Volta, like V100                  - maps to 70, 72
     *  - 7.5 is Turing, like RTX 2080 Ti               - maps to 75
     *  - 8.0, 8.6, 8.7 is Ampere, like A100, RTX 3090  - maps to 80, 86, 87
     *  - 8.9 is Ada Lovelace, like RTX 4090            - maps to 89
     *  - 9.0 is Hopper, like H100                      - maps to 90
     *  - 12.0, 12.1 is Blackwell, like B200            - maps to 120, 121
     */
    inline static std::size_t pack_sm_code(int major, int minor) noexcept {
        return static_cast<std::size_t>((major * 10) + minor);
    }

    /**
     *  @brief Looks up hardware specs for a given compute capability (major, minor), used to
     *      populate the @c cuda_cores property.
     *  @param[in] sm The compute capability code obtained from `pack_sm_code(major, minor)`.
     */
    inline static std::size_t cores_per_multiprocessor(std::size_t sm) noexcept {
        typedef struct {
            std::size_t sm;
            std::size_t cores;
        } generation_to_core_count;
        generation_to_core_count generations_to_core_counts[] = {
            // Kepler architecture (2012-2014)
            {pack_sm_code(3, 0), 192}, // Capability 3.0 (GK104 - GTX 680, GTX 770)
            {pack_sm_code(3, 5), 192}, // Capability 3.5 (GK110 - GTX 780 Ti, GTX Titan, Tesla K20/K40)
            {pack_sm_code(3, 7), 192}, // Capability 3.7 (GK210 - Tesla K80)

            // Maxwell architecture (2014-2016)
            {pack_sm_code(5, 0), 128}, // Capability 5.0 (GM107/GM108 - GTX 750/750 Ti, GTX 850M/860M)
            {pack_sm_code(5, 2), 128}, // Capability 5.2 (GM200/GM204/GM206 - GTX 980/970, Titan X)
            {pack_sm_code(5, 3), 128}, // Capability 5.3 (GM20B - Jetson TX1, Tegra X1)

            // Pascal architecture (2016-2018)
            {pack_sm_code(6, 0), 64},  // Capability 6.0 (GP100 - Tesla P100) - HPC focused, different SM design
            {pack_sm_code(6, 1), 128}, // Capability 6.1 (GP102/GP104/GP106/GP107 - GTX 1080/1070/1060/1050, Titan X/Xp)
            {pack_sm_code(6, 2), 128}, // Capability 6.2 (GP10B - Jetson TX2, Tegra X2)

            // Volta architecture (2017-2018)
            {pack_sm_code(7, 0), 64}, // Capability 7.0 (GV100 - Tesla V100, Titan V) - Tensor Core architecture
            {pack_sm_code(7, 2), 64}, // Capability 7.2 (GV11B - Jetson AGX Xavier, Tegra Xavier)

            // Turing architecture (2018-2020)
            {pack_sm_code(7, 5), 64}, // Capability 7.5 (TU102/TU104/TU106/TU116/TU117 - RTX 20xx, GTX 16xx)

            // Ampere architecture (2020-2022)
            {pack_sm_code(8, 0), 64},  // Capability 8.0 (GA100 - A100) - HPC focused
            {pack_sm_code(8, 6), 128}, // Capability 8.6 (GA102/GA104/GA106/GA107 - RTX 3090/3080/3070/3060)
            {pack_sm_code(8, 7), 128}, // Capability 8.7 (GA10B - Jetson AGX Orin, Tegra Orin)

            // Ada Lovelace architecture (2022-2023)
            {pack_sm_code(8, 9), 128}, // Capability 8.9 (AD102/AD103/AD104/AD106/AD107 - RTX 40xx)

            // Hopper architecture (2022-2024)
            {pack_sm_code(9, 0), 128}, // Capability 9.0 (GH100 - H100, H200)

            // Blackwell architecture (2024+)
            {pack_sm_code(12, 0), 128}, // Capability 12.0 (GB100 - B100)
            {pack_sm_code(12, 1), 128}, // Capability 12.1 (GB200 - B200)

            {0, 0}};

        std::size_t index = 0;
        for (; generations_to_core_counts[index].sm != 0; ++index)
            if (generations_to_core_counts[index].sm == sm) return generations_to_core_counts[index].cores;

        // If exact match not found, return the most recent known architecture's core count
        // This provides forward compatibility for newer architectures
        return (index > 0) ? generations_to_core_counts[index - 1].cores : 128;
    }
};

/**
 *  @brief Divides @p x by @p divisor and rounds up to the nearest integer.
 *
 *  @note This is equivalent to `ceil(x / divisor)`, but avoids floating-point arithmetic.
 */
template <typename scalar_type_>
constexpr scalar_type_ divide_round_up(scalar_type_ x, scalar_type_ divisor) {
    sz_assert_(divisor > 0 && "Divisor must be positive");
    return (x + divisor - 1) / divisor;
}

/** Rounds @p x up to the nearest multiple of @p divisor. */
template <typename scalar_type_>
constexpr scalar_type_ round_up_to_multiple(scalar_type_ x, scalar_type_ divisor) {
    sz_assert_(divisor > 0 && "Divisor must be positive");
    return divide_round_up(x, divisor) * divisor;
}

/**
 *  @brief The number of significant bits in @p x, or zero when @p x is zero.
 *
 *  Widens to @c u64_t first, so one body serves every unsigned width - shifting a 32-bit value by
 *  32 would be undefined even on a branch that never runs.
 *
 *  @note Unrolled rather than delegating to @c sz_u64_clz, which is an intrinsic wrapper and
 *      not @c constexpr.
 */
template <typename scalar_type_>
constexpr int bit_width(scalar_type_ x) noexcept {
    static_assert(std::is_unsigned<scalar_type_>::value, "Value type must be unsigned integer");
    u64_t wide = (u64_t)x;
    int width = 0;
    if (wide >> 32) { width += 32, wide >>= 32; }
    if (wide >> 16) { width += 16, wide >>= 16; }
    if (wide >> 8) { width += 8, wide >>= 8; }
    if (wide >> 4) { width += 4, wide >>= 4; }
    if (wide >> 2) { width += 2, wide >>= 2; }
    if (wide >> 1) { width += 1, wide >>= 1; }
    return width + (int)(wide != 0);
}

/**
 *  @brief Exact ⌊√x⌋ over integers, without floating-point arithmetic and without a loop.
 *
 *  Seeds a power of two at or above the root, then runs five Newton steps: each doubles the count
 *  of correct bits, and a 64-bit input has a 32-bit root, so 1 → 2 → 4 → 8 → 16 → 32 covers every
 *  width. Newton converges from above, so one clamp lands on the floor. Unlike @c std::sqrt, it is
 *  @c constexpr, which keeps it callable from CUDA device code.
 */
template <typename scalar_type_>
constexpr scalar_type_ integer_square_root(scalar_type_ x) noexcept {
    static_assert(std::is_unsigned<scalar_type_>::value, "Value type must be unsigned integer");
    static_assert(sizeof(scalar_type_) <= sizeof(u64_t), "Newton ladder is sized for at most 64-bit inputs");
    if (x <= 1) return x;
    u64_t const wide = (u64_t)x;
    u64_t guess = (u64_t)1 << ((bit_width(wide) + 1) / 2);
    guess = (guess + wide / guess) / 2;
    guess = (guess + wide / guess) / 2;
    guess = (guess + wide / guess) / 2;
    guess = (guess + wide / guess) / 2;
    guess = (guess + wide / guess) / 2;
    return (scalar_type_)(guess > wide / guess ? guess - 1 : guess);
}

/** Equivalent to `(condition ? value : 0)`, but avoids branching. */
template <typename value_type_>
constexpr value_type_ non_zero_if(value_type_ value, value_type_ condition) noexcept {
    static_assert(std::is_unsigned<value_type_>::value, "Value type must be unsigned integer");
    sz_assert_((condition == 0 || condition == 1) && "Condition must be either 0 or 1 unsigned integer");
    return value * condition;
}

/** Analog to @c std::swap from `<utility>`, but also generates device code, unlike STL. */
template <typename value_type_>
constexpr void trivial_swap(value_type_ &x, value_type_ &y) noexcept {
    static_assert(std::is_trivially_copyable<value_type_>::value, "Value type must be trivially copyable");
    value_type_ temp = x;
    x = y;
    y = temp;
}

/** Helper structure for dividing a range of data into three parts: head, body, and tail, generally
 *  used to minimize misaligned (split) stores and operate on aligned pages. */
struct head_body_tail_t {
    std::size_t head = 0;
    std::size_t body = 0;
    std::size_t tail = 0;

    constexpr head_body_tail_t() = default;
    constexpr head_body_tail_t(std::size_t h, std::size_t b, std::size_t t) : head(h), body(b), tail(t) {}
};

template <std::size_t elements_per_page_, typename element_type_>
constexpr head_body_tail_t head_body_tail(element_type_ *first_address, std::size_t total_length) noexcept {
    constexpr std::size_t bytes_per_element = sizeof(element_type_);
    constexpr std::size_t bytes_per_page = elements_per_page_ * bytes_per_element;
    static_assert(bytes_per_page > 0, "Slice size must be positive");

    // To split into head, body, and tail, we need the `first_address` to be
    // a multiple of `bytes_per_element`, otherwise the `body` will always be a zero!
    sz_assert_((std::size_t)first_address % bytes_per_element == 0);
    std::size_t bytes_misalignment = (std::size_t)first_address % bytes_per_page;
    std::size_t bytes_in_head = (bytes_per_page - bytes_misalignment) % bytes_per_page;
    std::size_t elements_in_head = bytes_in_head / bytes_per_element;

    // Round down the remaining count to a multiple of `elements_per_page_`.
    std::size_t aligned_pages = (total_length - elements_in_head) / elements_per_page_;
    std::size_t elements_in_body = aligned_pages * elements_per_page_;

    // Tail is simply what remains:
    std::size_t elements_in_tail = total_length - elements_in_head - elements_in_body;
    sz_assert_(elements_in_head < elements_per_page_ && elements_in_head <= total_length);
    sz_assert_(elements_in_tail < elements_per_page_ && elements_in_tail <= total_length);
    sz_assert_(elements_in_body % elements_per_page_ == 0);

    return head_body_tail_t {elements_in_head, elements_in_body, elements_in_tail};
}

/** Safer alternative to @c std::vector, that avoids exceptions and copy constructors: every member
 *  that allocates returns a @c status_t instead of throwing. */
template <typename value_type_, typename allocator_type_ = heap_alloc<value_type_>>
class vector {
  public:
    using value_type = value_type_;
    using size_type = std::size_t;
    using allocator_type = allocator_type_;

    using allocator_traits = std::allocator_traits<allocator_type>;
    using allocated_type = typename allocator_traits::value_type;
    static_assert(sizeof(value_type) == sizeof(allocated_type),
                  "Allocator value type must be the same size as the vector value type");
    static_assert(allocator_traits::propagate_on_container_move_assignment::value,
                  "Allocator must propagate on move assignment, otherwise the move assignment won't be `noexcept`.");

    static_assert(std::is_nothrow_destructible<allocator_type>::value &&
                      std::is_nothrow_default_constructible<allocator_type>::value &&
                      std::is_nothrow_copy_constructible<allocator_type>::value &&
                      std::is_nothrow_move_constructible<allocator_type>::value &&
                      std::is_nothrow_move_assignable<allocator_type>::value &&
                      noexcept(std::declval<allocator_type &>().allocate(size_type {})) &&
                      noexcept(std::declval<allocator_type &>().deallocate(nullptr, size_type {})),
                  "Vector allocators must return nullptr on failure and never throw");
    static_assert(std::is_nothrow_destructible<value_type>::value, "Vector element destruction must be noexcept");

  private:
    value_type *data_;
    size_type size_;
    size_type capacity_;
    allocator_type alloc_;

    /**
     *  @brief Whether the host may dereference what @ref allocator_type hands out,
     *      which growing requires.
     *
     *  Growth moves live elements on the host, so an allocator over memory the host cannot touch
     *  opts out with `static constexpr bool host_accessible_k = false` and gets a build error here
     *  instead of a segmentation fault, as @ref device_alloc does. Allocators that say nothing -
     *  are assumed reachable.
     */
    static constexpr bool allocator_reachable_from_host_() noexcept {
        if constexpr (requires { allocator_type::host_accessible_k; }) return allocator_type::host_accessible_k;
        else return true;
    }

  public:
    vector() noexcept : data_(nullptr), size_(0), capacity_(0), alloc_() {}
    vector(allocator_type allocator) noexcept : data_(nullptr), size_(0), capacity_(0), alloc_(allocator) {}
    ~vector() noexcept { reset(); }

    void clear() noexcept {
        if constexpr (!std::is_trivially_destructible<value_type>::value)
            for (size_type i = 0; i < size_; ++i) data_[i].~value_type();
        size_ = 0;
    }

    void reset() noexcept {
        clear();
        if (data_) alloc_.deallocate((allocated_type *)data_, capacity_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    /** @warning Use @c assign instead to handle out-of-memory failures. */
    vector(vector const &other) = delete;

    /** @warning Use @c assign instead to handle out-of-memory failures. */
    vector &operator=(vector const &other) = delete;

    vector(vector &&other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_), alloc_(std::move(other.alloc_)) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    vector &operator=(vector &&other) noexcept {
        if (this != &other) {
            clear();
            if (data_) alloc_.deallocate((allocated_type *)data_, capacity_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            alloc_ = std::move(other.alloc_);
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    status_t assign(std::span<value_type const> const other) noexcept {
        static_assert(std::is_nothrow_copy_constructible<value_type>::value &&
                      std::is_nothrow_copy_assignable<value_type>::value);
        reset();

        if (other.size() == 0) return status_t::success_k; // Nothing to do :)

        // Allocate exact needed capacity
        size_type new_cap = other.size();
        allocated_type *raw = alloc_.allocate(new_cap);
        if (!raw) return status_t::bad_alloc_k;
        data_ = reinterpret_cast<value_type *>(raw);
        capacity_ = new_cap;

        // Copy‐construct each element
        if constexpr (!std::is_trivially_constructible<value_type>::value)
            for (size_type i = 0; i < other.size(); ++i) new (data_ + i) value_type(other[i]);
        else
            for (size_type i = 0; i < other.size(); ++i) data_[i] = other[i];
        size_ = other.size();
        return status_t::success_k;
    }

    template <typename other_allocator_type_>
    status_t assign(vector<value_type, other_allocator_type_> const &other) noexcept {
        return assign(std::span<value_type const>(other.data(), other.size()));
    }

    status_t reserve(size_type new_cap) noexcept {
        static_assert(std::is_nothrow_move_constructible<value_type>::value);
        static_assert(allocator_reachable_from_host_(),
                      "Growing host-moves live elements, so device-only storage must use `resize_uninitialized`");
        if (new_cap <= capacity_) return status_t::success_k;
        value_type *new_data = (value_type *)alloc_.allocate(new_cap);
        if (!new_data) return status_t::bad_alloc_k;
        for (size_type i = 0; i < size_; ++i) {
            new (new_data + i) value_type(std::move(data_[i]));
            if constexpr (!std::is_trivially_destructible<value_type>::value) data_[i].~value_type();
        }
        if (data_) alloc_.deallocate((allocated_type *)data_, capacity_);
        data_ = new_data;
        capacity_ = new_cap;
        return status_t::success_k;
    }

    status_t resize(size_type new_size) noexcept {
        static_assert(std::is_nothrow_default_constructible<value_type>::value);
        if (new_size > capacity_ && reserve(new_size) != status_t::success_k) return status_t::bad_alloc_k;

        if (new_size > size_) {
            if constexpr (!std::is_trivially_constructible<value_type>::value)
                for (size_type i = size_; i < new_size; ++i) new (data_ + i) value_type();
        }
        else if (new_size < size_) {
            if constexpr (!std::is_trivially_destructible<value_type>::value)
                for (size_type i = new_size; i < size_; ++i) data_[i].~value_type();
        }

        size_ = new_size;
        return status_t::success_k;
    }

    /**
     *  @brief Resizes without constructing, destroying, or moving any element - the caller
     *      guarantees to overwrite every live element before reading it.
     *
     *  On growth it allocates fresh storage and discards the old contents, with no element move, so
     *  it is safe even when the storage lives in @b device memory the host cannot dereference, like
     *  a task array backed by @ref device_alloc. Requires a trivially-destructible type.
     */
    status_t resize_uninitialized(size_type new_size) noexcept {
        static_assert(std::is_trivially_destructible<value_type>::value,
                      "resize_uninitialized requires a trivially-destructible value type");
        if (new_size > capacity_) {
            value_type *new_data = (value_type *)alloc_.allocate(new_size);
            if (!new_data) return status_t::bad_alloc_k;
            if (data_) alloc_.deallocate((allocated_type *)data_, capacity_);
            data_ = new_data;
            capacity_ = new_size;
        }
        size_ = new_size;
        return status_t::success_k;
    }

    status_t push_back(value_type const &value) noexcept {
        static_assert(std::is_nothrow_copy_constructible<value_type>::value);
        if (size_ == capacity_) {
            size_type new_cap = capacity_ ? capacity_ * 2 : 1;
            if (reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        new (data_ + size_) value_type(value);
        ++size_;
        return status_t::success_k;
    }

    status_t push_back(value_type &&value) noexcept {
        static_assert(std::is_nothrow_move_constructible<value_type>::value);
        if (size_ == capacity_) {
            size_type new_cap = capacity_ ? capacity_ * 2 : 1;
            if (reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        new (data_ + size_) value_type(std::move(value));
        ++size_;
        return status_t::success_k;
    }

    status_t append(std::span<value_type const> source) noexcept {
        static_assert(std::is_nothrow_copy_constructible<value_type>::value);
        size_type needed = size_ + source.size();
        if (needed > capacity_) {
            size_type new_cap = capacity_ ? capacity_ : 1;
            while (new_cap < needed) new_cap *= 2;
            if (reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        for (size_type i = 0; i < source.size(); ++i) new (data_ + size_ + i) value_type(source[i]);
        size_ = needed;
        return status_t::success_k;
    }

    value_type *begin() noexcept { return data_; }
    value_type const *begin() const noexcept { return data_; }
    value_type *end() noexcept { return data_ + size_; }
    value_type const *end() const noexcept { return data_ + size_; }
    value_type &operator[](size_type i) noexcept {
        sz_assert_(i < size_);
        return data_[i];
    }
    value_type const &operator[](size_type i) const noexcept {
        sz_assert_(i < size_);
        return data_[i];
    }
    value_type *data() noexcept { return data_; }
    value_type const *data() const noexcept { return data_; }
    value_type &front() noexcept {
        sz_assert_(size_ != 0);
        return data_[0];
    }
    value_type const &front() const noexcept {
        sz_assert_(size_ != 0);
        return data_[0];
    }
    value_type &back() noexcept {
        sz_assert_(size_ != 0);
        return data_[size_ - 1];
    }
    value_type const &back() const noexcept {
        sz_assert_(size_ != 0);
        return data_[size_ - 1];
    }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    operator std::span<value_type>() noexcept { return {data_, size_}; }
    operator std::span<value_type const>() const noexcept { return {data_, size_}; }
};

#pragma region Unified Allocators

/**
 *  @brief Allocator over the @b unified memory of one device group, which the host and that
 *      group's devices all address.
 *
 *  Nullable allocator for @ref strings and @ref vector.
 *  @p capabilities_ picks the default vendor; an initialized C allocator can select it at runtime.
 *  The allocator handle and stream must outlive its allocations; a null stream uses the vendor's
 *  default device. Failure returns null; @c make reports initialization errors.
 *
 *  @tparam capabilities_ One device's capabilities, like @c sz_cap_cuda_k.
 */
template <typename value_type_, sz_capability_t capabilities_ = sz_cap_serial_k>
struct unified_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;
    using is_always_equal = std::false_type;

    sz_allocator_t unified {};
    sz_stream_t stream = nullptr;

    template <typename other_value_type_>
    struct rebind {
        using other = unified_alloc<other_value_type_, capabilities_>;
    };

    constexpr unified_alloc() noexcept = default;
    static expected<unified_alloc> make(sz_stream_t stream = nullptr) noexcept {
        sz_allocator_t policy {};
        sz_status_t status = sz_success_k;
        if constexpr (capabilities_ == sz_cap_serial_k) status = sz_allocator_init_heap(&policy);
        else status = sz_allocator_init_unified_best(&policy, capabilities_);
        return {unified_alloc(policy, stream), static_cast<status_t>(status)};
    }
    constexpr unified_alloc(sz_allocator_t const &unified, sz_stream_t stream = nullptr) noexcept
        : unified(unified), stream(stream) {}
    template <typename other_value_type_, sz_capability_t other_capabilities_>
    constexpr unified_alloc(unified_alloc<other_value_type_, other_capabilities_> const &other) noexcept
        : unified(other.unified), stream(other.stream) {}

    value_type *allocate(size_type count) const noexcept {
        count = count ? count : 1;
        if (!unified.allocate || !unified.free || count > (std::numeric_limits<size_type>::max)() / sizeof(value_type))
            return nullptr;
        pointer result = static_cast<pointer>(unified.allocate(count * sizeof(value_type), unified.handle, stream));
        if (!result || reinterpret_cast<sz_size_t>(result) % alignof(value_type) == 0) return result;
        unified.free(result, count * sizeof(value_type), unified.handle, stream);
        return nullptr;
    }
    void deallocate(pointer start, size_type count) const noexcept {
        if (!start) return;
        unified.free(start, (count ? count : 1) * sizeof(value_type), unified.handle, stream);
    }
    template <typename other_type_, sz_capability_t other_capabilities_>
    bool operator==(unified_alloc<other_type_, other_capabilities_> const &other) const noexcept {
        return unified.allocate == other.unified.allocate && unified.reallocate == other.unified.reallocate &&
               unified.free == other.unified.free && unified.handle == other.unified.handle && stream == other.stream;
    }
    template <typename other_type_, sz_capability_t other_capabilities_>
    bool operator!=(unified_alloc<other_type_, other_capabilities_> const &other) const noexcept {
        return !(*this == other);
    }
};

#pragma endregion Unified Allocators

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_TYPES_HPP_
