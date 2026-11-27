/**
 *  @file include/stringzilla/types.hpp
 *  @author Ash Vardanian
 *  @date March 28, 2025
 *  @brief Shared definitions for the StringZilla C++ library.
 *
 *  The goal for this header is to provide an absolutely-minimal set of types and
 *  forward-declarations for CPU and GPU backends of higher-level complex templated algorithms
 *  implemented outside of the C layer. It includes the following primitive type aliases for
 *  the <tt>types.h</tt> header:
 *
 *  - @c u8_t, @c u16_t, @c u32_t, @c u64_t, @c i8_t, @c i16_t, @c i32_t, @c i64_t - sized integers.
 *  - @c ssize_t, @c ptr_t, @c cptr_t - address-related types.
 *  - @c status_t, @c bool_t, @c ordering_t, @c rune_t, @c rune_length_t, @c error_cost_t - logic.
 *
 *  The header also provides these higher-level types:
 *
 *  - @c std::span<value_type> - a view to a contiguous memory block of @c value_type elements.
 *  - @c dummy_alloc<value_type> - a dummy memory allocator shaped like @c std::allocator.
 *  - <tt>vector&lt;value_type, allocator_type&gt;</tt> - owning storage with fallible allocation.
 */
#ifndef STRINGZILLA_TYPES_HPP_
#define STRINGZILLA_TYPES_HPP_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_cap_serial_k`

#if STRINGZILLA_CXX_STANDARD_ < 202002L
#error "StringZilla's C++ interface requires C++20; the C interface in `stringzilla.h` has no such floor."
#endif

#include <initializer_list> // `std::initializer_list` is only ~100 LOC
#include <iterator>         // `std::random_access_iterator_tag` pulls 20K LOC
#include <limits>           // `std::numeric_limits`
#include <memory>           // `std::allocator_traits` for allocator rebinding
#include <new>              // Placement construction
#include <span>             // `std::span`
#include <type_traits>      // `std::is_const_v`, `std::is_arithmetic_v`, `std::is_trivially_destructible`
#include <utility>          // `std::move`

namespace ashvardanian {
namespace stringzilla {

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

/** The smaller of @p first and @p second; arithmetic only, so a caller's own @c min_of never
 *  meets this one by argument-dependent lookup. */
template <typename value_type_>
    requires std::is_arithmetic_v<value_type_>
constexpr value_type_ min_of(value_type_ first, value_type_ second) noexcept {
    return second < first ? second : first;
}

/** The larger of @p first and @p second, the @c max twin of @ref min_of. */
template <typename value_type_>
    requires std::is_arithmetic_v<value_type_>
constexpr value_type_ max_of(value_type_ first, value_type_ second) noexcept {
    return first < second ? second : first;
}

#pragma region Status

/**
 *  @brief Why a call failed, in the vocabulary the C ABI and every binding share.
 *
 *  Every enumerator takes its value from @c sz_status_t in @c types.h, so the C++ mirror and the
 *  ABI cannot drift, and the crossing is a cast rather than a table. Any verb that allocates, binds
 *  a device or touches a file may also return @c bad_alloc_k or a device status, so a verb's own
 *  block lists only what its contract adds.
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

    /** A pattern, path or document breaking its grammar, or using a construct no automaton runs. */
    invalid_syntax_k = sz_invalid_syntax_k,

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

#pragma endregion Status

#pragma region Devices

/** Which runtime a device belongs to, as the <tt>sz_&lt;kind&gt;_*</tt> C functions name it. */
enum class device_kind_t : int { cpu_k, cuda_k, rocm_k, metal_k };

/** Bytes a strided output touches, @p count entries @p stride bytes apart and @p entry_bytes long,
 *  saturating at @c SIZE_MAX, which no span holds, so an overflow reads as too short. */
constexpr std::size_t strided_bytes_(std::size_t count, std::size_t stride, std::size_t entry_bytes) noexcept {
    std::size_t const saturated = (std::numeric_limits<std::size_t>::max)();
    if (count == 0 || entry_bytes == 0) return 0;
    if (stride && count - 1 > (saturated - entry_bytes) / stride) return saturated;
    return (count - 1) * stride + entry_bytes;
}

#pragma endregion Devices

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

    using iterator_category = std::random_access_iterator_tag;

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

/**
 *  @brief Divides @p x by @p divisor and rounds up to the nearest integer.
 *
 *  @note This is equivalent to <tt>ceil(x / divisor)</tt>, but avoids floating-point arithmetic.
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

/** Analog to @c std::swap from <tt>&lt;utility&gt;</tt>, but also generates device code, unlike STL. */
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
     *  opts out with <tt>static constexpr bool host_accessible_k = false</tt> and gets a build error here
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
     *  a task array backed by @c device_alloc. Requires a trivially-destructible type.
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
 *  default device. Failure returns null; @c make_unified_alloc reports initialization errors.
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
