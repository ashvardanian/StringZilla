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
 *  The header also provides these higher-level types:
 *
 *  - `std::span<value_type>` - a view to a contiguous memory block of @c value_type elements.
 *  - `dummy_alloc<value_type>` - a dummy memory allocator shaped like @c std::allocator.
 *  - `vector<value_type, allocator_type>` - owning storage with fallible allocation.
 *  - `tape<char_type, offset_type, allocator_type>` - an owning sequence of packed strings.
 *  - `tape_view<char_type, offset_type>` - a borrowed view over string bytes and offsets.
 */
#ifndef STRINGZILLA_TYPES_HPP_
#define STRINGZILLA_TYPES_HPP_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capabilities_enabled_cpu`, `sz_capabilities_enabled_cuda`
#include "stringzilla/memory.h"

/** When set to 1, the library will include the C++ STL headers and implement automatic conversion
 *  from and to @c std::string_view and `std::basic_string<any_allocator>`. */
#ifndef STRINGZILLA_WITH_STL
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
#include <exception>        // `std::terminate`
#include <initializer_list> // `std::initializer_list` is only ~100 LOC
#include <iterator>         // `std::random_access_iterator_tag` pulls 20K LOC
#include <limits>           // `std::numeric_limits`
#include <memory>           // `std::allocator_traits` for allocator rebinding
#include <new>              // `std::bad_alloc`
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
using sorted_idx_t = sz_sorted_idx_t;

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
 *  dispatch points are stubs. */
inline sz_capability_t default_capabilities() noexcept {
#if STRINGZILLA_HEADER_ONLY
    return 0;
#else
    sz_capability_t capabilities = 0;
    return sz_capabilities_enabled_cpu(&capabilities) == sz_success_k ? capabilities : sz_cap_serial_k;
#endif
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

/** Allocates @p count elements through @p allocator, reading a throw as null, so the @c noexcept
 *  containers report @c bad_alloc_k where @c std::allocator would terminate them. */
template <typename allocator_type_>
typename std::allocator_traits<allocator_type_>::pointer allocate_or_null_(allocator_type_ &allocator,
                                                                           std::size_t count) noexcept {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
    try {
        return std::allocator_traits<allocator_type_>::allocate(allocator, count);
    }
    catch (...) {
        return nullptr;
    }
#else
    return std::allocator_traits<allocator_type_>::allocate(allocator, count);
#endif
}

/**
 *  @brief Random access iterator for any immutable container with indexed element lookup support.
 *
 *  @note Designed for @c tape and @c tape_view compatibility with STL
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

/** Length convention for @ref tape_view element spans. */
enum class tape_termination_t {

    /** Matches @ref tape: a trailing @c '\0' is excluded, so elements
     *  read as C-strings. */
    nul_terminated_k,

    /** Raw Apache Arrow / cuDF / Parquet: the full `[offsets[i], offsets[i+1])`
     *  span, no terminator. */
    packed_k,
};

/**
 *  @brief Borrowed variable-length elements in a buffer, delimited by a `count + 1` offsets array.
 *
 *  @tparam char_type_ Buffer element type: @c char, @c std::byte, or a wider unit.
 *  @tparam offset_type_ Offset type into the buffer, typically @c int32_t or @c int64_t.
 *  @tparam termination_ Length convention, see @ref tape_termination_t.
 *  @sa tape
 *
 *  Elements exclude the trailing NUL by default, matching @ref tape. Use @c packed_k for
 *  terminator-free Apache Arrow and cuDF columns.
 */
template <typename char_type_, typename offset_type_,
          tape_termination_t termination_ = tape_termination_t::nul_terminated_k>
struct tape_view {
    using char_t = char_type_;
    using offset_t = offset_type_;
    using self_t = tape_view<char_t, offset_t, termination_>;

    using value_t = std::span<char_t const>;
    using value_type = value_t; // ? For STL compatibility
    using iterator_t = indexed_container_iterator<self_t>;
    using iterator = iterator_t; // ? For STL compatibility

    /** Bytes excluded from every element's length — one for the NULL terminator, zero for the
     *  terminator-free Apache Arrow / cuDF convention. */
    static constexpr std::size_t terminator_width_k = termination_ == tape_termination_t::nul_terminated_k ? 1u : 0u;

    std::span<char_t const> buffer_;
    std::span<offset_t const> offsets_;

    constexpr tape_view() noexcept : buffer_ {}, offsets_ {} {}
    constexpr tape_view(std::span<char_t const> buffer, std::span<offset_t const> offsets) noexcept
        : buffer_(buffer), offsets_(offsets) {}

    constexpr std::size_t size() const noexcept { return offsets_.size() != 0 ? offsets_.size() - 1 : 0; }
    constexpr value_t operator[](std::size_t i) const noexcept {
        return {buffer_.data() + offsets_[i],
                static_cast<std::size_t>(offsets_[i + 1] - offsets_[i]) - terminator_width_k};
    }

    /**
     *  @brief The contiguous block the elements slice, from the first element's start to the
     *      last one's end.
     *
     *  @note Starts at `offsets_[0]`, which a tape that is a slice of a wider one leaves non-zero.
     */
    constexpr std::span<char_t const> tape_bytes() const noexcept {
        return size() == 0 ? std::span<char_t const> {}
                           : std::span<char_t const> {buffer_.data() + offsets_[0],
                                                      static_cast<std::size_t>(offsets_[size()] - offsets_[0])};
    }

    /**
     *  @brief Every element's length summed, terminators excluded, without walking the elements.
     *
     *  @note Returns the tape's own offset width, so a 32-bit tape stays 32-bit until a
     *      caller needs more.
     */
    constexpr offset_t tape_total_bytes() const noexcept {
        return size() == 0 ? offset_t {}
                           : static_cast<offset_t>(offsets_[size()] - offsets_[0] -
                                                   static_cast<offset_t>(size()) * terminator_width_k);
    }

    /** One element's length, terminator excluded, in the tape's own offset width. */
    constexpr offset_t tape_length_at(std::size_t i) const noexcept {
        return static_cast<offset_t>(offsets_[i + 1] - offsets_[i] - terminator_width_k);
    }

    constexpr iterator_t begin() const noexcept { return iterator_t(*this, 0); }
    constexpr iterator_t end() const noexcept { return iterator_t(*this, size()); }
    constexpr iterator_t cbegin() const noexcept { return begin(); }
    constexpr iterator_t cend() const noexcept { return end(); }
};

/** The terminator-free Apache Arrow / cuDF / Parquet flavor of @ref tape_view: element
 *  @c i is the full `[offsets[i], offsets[i+1])` span, no NULL excluded. */
template <typename char_type_, typename offset_type_>
using packed_tape_view = tape_view<char_type_, offset_type_, tape_termination_t::packed_k>;

/**
 *  @brief Owns variable-length strings in NUL-terminated or canonical packed storage.
 *
 *  The default layout appends NUL-terminated strings with separate offsets. Packed tapes use one
 *  canonical block of 64-bit offsets and unterminated bytes. Packed append repacks the block;
 *  only the default layout promises amortized growth.
 */
template <typename char_type_, typename offset_type_, typename allocator_type_,
          tape_termination_t termination_ = tape_termination_t::nul_terminated_k>
struct tape {
    static_assert(termination_ != tape_termination_t::packed_k ||
                      (std::is_same<char_type_, char>::value && std::is_same<offset_type_, sz_u64_t>::value),
                  "Canonical packed tapes use char and sz_u64_t");
    using char_t = char_type_;
    using offset_t = offset_type_;
    using allocator_t = allocator_type_;
    using self_t = tape<char_t, offset_t, allocator_t, termination_>;

    using value_t = std::span<char_t const>;
    using view_t = tape_view<char_t, offset_t, termination_>;
    using value_type = value_t; // ? For STL compatibility
    using iterator_t = indexed_container_iterator<self_t>;
    using iterator = iterator_t; // ? For STL compatibility

    using char_alloc_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<char_t>;
    using offset_alloc_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<offset_t>;

    /** Largest offset the tape can address, past which an @c offset_t would wrap around. */
    static constexpr std::size_t max_offset_k = static_cast<std::size_t>((std::numeric_limits<offset_t>::max)());

  private:
    std::span<char_t> buffer_;
    std::span<offset_t> offsets_;
    char_alloc_t char_alloc_;
    offset_alloc_t offset_alloc_;
    std::size_t count_ = 0;

  public:
    constexpr tape() = default;

    tape(tape const &) = delete;
    tape &operator=(tape const &) = delete;

    constexpr tape(tape &&other) noexcept
        : buffer_(other.buffer_), offsets_(other.offsets_), char_alloc_(std::move(other.char_alloc_)),
          offset_alloc_(std::move(other.offset_alloc_)), count_(other.count_) {
        other.buffer_ = {}, other.offsets_ = {}, other.count_ = 0;
    }

    constexpr tape &operator=(tape &&other) noexcept {
        if (this == &other) return *this;
        reset();
        buffer_ = other.buffer_, offsets_ = other.offsets_;
        char_alloc_ = std::move(other.char_alloc_), offset_alloc_ = std::move(other.offset_alloc_);
        count_ = other.count_;
        other.buffer_ = {}, other.offsets_ = {}, other.count_ = 0;
        return *this;
    }

    explicit constexpr tape(allocator_t allocator) : char_alloc_(allocator), offset_alloc_(allocator) {}

    constexpr tape(std::span<char_t> buffer, std::span<offset_t> offsets, allocator_t allocator)
        requires(termination_ == tape_termination_t::nul_terminated_k)
        : buffer_(buffer), offsets_(offsets), char_alloc_(allocator), offset_alloc_(allocator) {}

    constexpr ~tape() noexcept { reset(); }
    constexpr void reset() noexcept {
        if (buffer_.data()) char_alloc_.deallocate(buffer_.data(), buffer_.size()), buffer_ = {};
        if constexpr (termination_ == tape_termination_t::nul_terminated_k)
            if (offsets_.data()) offset_alloc_.deallocate(offsets_.data(), offsets_.size());
        offsets_ = {};
        count_ = 0;
    }

    constexpr iterator_t begin() const noexcept { return iterator_t(*this, 0); }
    constexpr iterator_t end() const noexcept { return iterator_t(*this, size()); }
    constexpr iterator_t cbegin() const noexcept { return begin(); }
    constexpr iterator_t cend() const noexcept { return end(); }

    template <typename strings_iterator_type_>
    status_t assign(strings_iterator_type_ first, strings_iterator_type_ last) noexcept {
        static_assert(std::is_base_of<std::forward_iterator_tag,
                                      typename std::iterator_traits<strings_iterator_type_>::iterator_category>::value,
                      "tape::assign needs multi-pass (forward) iterators");
        std::size_t count = 0, combined_length = 0;
        constexpr std::size_t terminator = view_t::terminator_width_k;
        for (auto it = first; it != last; ++it, ++count) {
            std::size_t const length = it->size();
            if (combined_length > max_offset_k - terminator || length > max_offset_k - combined_length - terminator)
                return status_t::overflow_risk_k;
            combined_length += length + terminator;
        }
        if (count >= (std::numeric_limits<std::size_t>::max)() / sizeof(offset_t) ||
            combined_length > (std::numeric_limits<std::size_t>::max)() / sizeof(char_t))
            return status_t::overflow_risk_k;

        self_t next {allocator_t(char_alloc_)};
        if constexpr (termination_ == tape_termination_t::packed_k) {
            std::size_t const header = (count + 1) * sizeof(offset_t);
            if (combined_length > max_offset_k - header) return status_t::overflow_risk_k;
            char_t *block = allocate_or_null_(next.char_alloc_, header + combined_length);
            if (!block) return status_t::bad_alloc_k;
            next.buffer_ = {block, header + combined_length};
            if (reinterpret_cast<sz_size_t>(block) % alignof(offset_t)) return status_t::bad_alloc_k;
            next.offsets_ = {reinterpret_cast<offset_t *>(block), count + 1};
            next.offsets_[0] = static_cast<offset_t>(header);
        }
        else if (count) {
            char_t *buffer = allocate_or_null_(next.char_alloc_, combined_length);
            if (!buffer) return status_t::bad_alloc_k;
            next.buffer_ = {buffer, combined_length};
            offset_t *offsets = allocate_or_null_(next.offset_alloc_, count + 1);
            if (!offsets) return status_t::bad_alloc_k;
            next.offsets_ = {offsets, count + 1};
            next.offsets_[0] = 0;
        }

        std::size_t index = 0;
        for (auto it = first; it != last; ++it, ++index) {
            auto const &text = *it;
            char_t const *source = text.data();
            char_t *target = next.buffer_.data() + next.offsets_[index];
            for (std::size_t i = 0; i != text.size(); ++i) target[i] = source[i];
            if constexpr (termination_ == tape_termination_t::nul_terminated_k) target[text.size()] = char_t {};
            next.offsets_[index + 1] = static_cast<offset_t>(next.offsets_[index] + text.size() + terminator);
        }
        next.count_ = count;
        *this = std::move(next);
        return status_t::success_k;
    }

    constexpr std::size_t allocation_bytes() const noexcept {
        return buffer_.size() * sizeof(char_t) +
               (termination_ == tape_termination_t::nul_terminated_k ? offsets_.size() * sizeof(offset_t) : 0);
    }

    sz_sequence_t sequence() const noexcept
        requires(termination_ == tape_termination_t::packed_k)
    {
        return {buffer_.data(), count_, sz_sequence_tape_start, sz_sequence_tape_length};
    }

    status_t assign(sz_sequence_t const &source) noexcept
        requires(termination_ == tape_termination_t::packed_k)
    {
        if (source.count >= (std::numeric_limits<std::size_t>::max)() / sizeof(offset_t)) return status_t::bad_alloc_k;
        struct input_t {
            using value_type = std::span<char_t const>;
            sz_sequence_t const &source;
            value_type operator[](std::size_t index) const noexcept {
                return {source.get_start(source.handle, index), source.get_length(source.handle, index)};
            }
        } input {source};
        using input_iterator_t = indexed_container_iterator<input_t>;
        return assign(input_iterator_t(input, 0), input_iterator_t(input, source.count));
    }

    status_t assign(std::span<sz_string_view_t const> views) noexcept
        requires(termination_ == tape_termination_t::packed_k)
    {
        sz_sequence_t source {};
        sz_sequence_from_string_views(views.data(), views.size(), &source);
        return assign(source);
    }

#if STRINGZILLA_WITH_STL
    template <typename string_convertible_type_>
    status_t assign(std::initializer_list<string_convertible_type_> inits) noexcept {
        return assign(inits.begin(), inits.end());
    }
#endif

    status_t append(std::span<char_t const> string) noexcept {
        if constexpr (termination_ == tape_termination_t::packed_k) {
            if (count_ == (std::numeric_limits<std::size_t>::max)()) return status_t::overflow_risk_k;
            struct input_t {
                using value_type = std::span<char_t const>;
                self_t const &source;
                value_type tail;
                value_type operator[](std::size_t index) const noexcept {
                    return index == source.size() ? tail : source[index];
                }
            } input {*this, string};
            using input_iterator_t = indexed_container_iterator<input_t>;
            return assign(input_iterator_t(input, 0), input_iterator_t(input, count_ + 1));
        }

        std::size_t const string_length = string.size();
        std::size_t const current_used = count_ ? offsets_[count_] : 0;
        if (string_length >= max_offset_k - current_used) return status_t::overflow_risk_k;
        std::size_t const needed = current_used + string_length + 1;
        std::size_t const max_offsets = (std::numeric_limits<std::size_t>::max)() / sizeof(offset_t);
        if (count_ >= max_offsets - 1) return status_t::overflow_risk_k;

        if (count_ + 2 > offsets_.size()) {
            std::size_t const capacity = sz_size_bit_ceil(count_ + 2);
            if (!capacity || capacity > max_offsets) return status_t::overflow_risk_k;
            offset_t *offsets = allocate_or_null_(offset_alloc_, capacity);
            if (!offsets) return status_t::bad_alloc_k;
            if (offsets_.data()) {
                for (std::size_t i = 0; i <= count_; ++i) offsets[i] = offsets_[i];
                offset_alloc_.deallocate(offsets_.data(), offsets_.size());
            }
            offsets_ = {offsets, capacity};
        }

        std::span<char_t> next = buffer_;
        if (needed > buffer_.size()) {
            std::size_t const capacity = sz_size_bit_ceil(needed);
            if (!capacity) return status_t::overflow_risk_k;
            char_t *buffer = allocate_or_null_(char_alloc_, capacity);
            if (!buffer) return status_t::bad_alloc_k;
            next = {buffer, capacity};
            for (std::size_t i = 0; i < current_used; ++i) next[i] = buffer_[i];
        }
        // The input may point into the old buffer, so copy it before releasing that allocation.
        for (std::size_t i = 0; i < string_length; ++i) next[current_used + i] = string[i];
        next[needed - 1] = '\0';
        if (next.data() != buffer_.data() && buffer_.data()) char_alloc_.deallocate(buffer_.data(), buffer_.size());
        buffer_ = next;
        offsets_[count_] = static_cast<offset_t>(current_used);
        offsets_[++count_] = static_cast<offset_t>(needed);
        return status_t::success_k;
    }

    constexpr value_type operator[](std::size_t i) const noexcept {
        sz_assert_(i < count_ && "Index out of bounds");
        view_t const values = view();
        return values[i];
    }

    constexpr std::size_t size() const noexcept { return count_; }
    constexpr view_t view() const noexcept { return {buffer_, offsets_.first(offsets_.empty() ? 0 : count_ + 1)}; }
    constexpr std::span<char_t> const &buffer() const noexcept { return buffer_; }
    constexpr std::span<offset_t> const &offsets() const noexcept { return offsets_; }
};

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
template <typename value_type_, typename allocator_type_>
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
     *  @c std::allocator included - are assumed reachable, so nothing else needs changing.
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
        reset();

        if (other.size() == 0) return status_t::success_k; // Nothing to do :)

        // Allocate exact needed capacity
        size_type new_cap = other.size();
        allocated_type *raw = allocate_or_null_(alloc_, new_cap);
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
        static_assert(allocator_reachable_from_host_(),
                      "Growing host-moves live elements, so device-only storage must use `resize_uninitialized`");
        if (new_cap <= capacity_) return status_t::success_k;
        value_type *new_data = (value_type *)allocate_or_null_(alloc_, new_cap);
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
            value_type *new_data = (value_type *)allocate_or_null_(alloc_, new_size);
            if (!new_data) return status_t::bad_alloc_k;
            if (data_) alloc_.deallocate((allocated_type *)data_, capacity_);
            data_ = new_data;
            capacity_ = new_size;
        }
        size_ = new_size;
        return status_t::success_k;
    }

    status_t push_back(value_type const &value) noexcept {
        if (size_ == capacity_) {
            size_type new_cap = capacity_ ? capacity_ * 2 : 1;
            if (reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        new (data_ + size_) value_type(value);
        ++size_;
        return status_t::success_k;
    }

    status_t push_back(value_type &&value) noexcept {
        if (size_ == capacity_) {
            size_type new_cap = capacity_ ? capacity_ * 2 : 1;
            if (reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        new (data_ + size_) value_type(std::move(value));
        ++size_;
        return status_t::success_k;
    }

    status_t append(std::span<value_type const> source) noexcept {
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
 *  Compatible with @c std::vector, @ref tape and @ref vector.
 *  @p capabilities_ picks the default vendor; an initialized C allocator can select it at runtime.
 *  The allocator handle and stream must outlive its allocations; a null stream uses the vendor's
 *  default device. Allocation or alignment failure throws @c std::bad_alloc.
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

    unified_alloc() : unified_alloc(nullptr) {}
    explicit unified_alloc(sz_stream_t stream) : stream(stream) {
        if constexpr (capabilities_ == sz_cap_serial_k) {
            if (sz_allocator_init_heap(&unified) == sz_success_k) return;
        }
        else if (sz_allocator_init_unified_best(&unified, capabilities_) == sz_success_k) return;
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
        throw std::bad_alloc();
#else
        std::terminate();
#endif
    }
    constexpr unified_alloc(sz_allocator_t const &unified, sz_stream_t stream = nullptr) noexcept
        : unified(unified), stream(stream) {}
    template <typename other_value_type_, sz_capability_t other_capabilities_>
    constexpr unified_alloc(unified_alloc<other_value_type_, other_capabilities_> const &other) noexcept
        : unified(other.unified), stream(other.stream) {}

    value_type *allocate(size_type count) const {
        pointer result = nullptr;
        count = count ? count : 1;
        if (count > (std::numeric_limits<size_type>::max)() / sizeof(value_type)) goto failed;
        result = static_cast<pointer>(unified.allocate(count * sizeof(value_type), unified.handle, stream));
        if (!result) goto failed;
        if (reinterpret_cast<sz_size_t>(result) % alignof(value_type) == 0) return result;
        unified.free(result, count * sizeof(value_type), unified.handle, stream);
    failed:
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
        throw std::bad_alloc();
#else
        std::terminate();
#endif
    }
    void deallocate(pointer start, size_type count) const noexcept {
        unified.free(start, (count ? count : 1) * sizeof(value_type), unified.handle, stream);
    }
    template <typename other_type_, sz_capability_t other_capabilities_>
    bool operator==(unified_alloc<other_type_, other_capabilities_> const &other) const noexcept {
        return unified.allocate == other.unified.allocate && unified.free == other.unified.free &&
               unified.handle == other.unified.handle && stream == other.stream;
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
