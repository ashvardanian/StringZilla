/**
 *  @brief  Shared definitions for the StringZilla C++ library.
 *  @file   types.hpp
 *  @author Ash Vardanian
 *
 *  The goal for this header is to provide absolutely-minimal set of types and forward-declarations for
 *  CPU and GPU backends of higher-level complex templated algorithms implemented outside of the C layer.
 *  It includes the following primitive type aliases for the @b "types.h" header:
 *
 *  - `u8_t`, `u16_t`, `u32_t`, `u64_t`, `i8_t`, `i16_t`, `i32_t`, `i64_t` - sized integers.
 *  - `size_t`, `ssize_t`, `ptr_t`, `cptr_t` - address-related types.
 *  - `status_t`, `bool_t`, `ordering_t`, `rune_t`, `rune_length_t`, `error_cost_t` - for logic.
 *
 *  The library also defines the following higher-level structures:
 *
 *  - `span<value_type>` - a view to a contiguous memory block of `value_type` elements.
 *  - `dummy_alloc<value_type>` - a dummy memory allocator that resembles the `std::allocator` interface.
 *  - `arrow_strings_tape<char_type, offset_type>` - a tape data-structure to efficiently store a sequence strings.
 */
#ifndef STRINGZILLA_TYPES_HPP_
#define STRINGZILLA_TYPES_HPP_

#include "types.h"

/**
 *  @brief  When set to 1, the library will include the C++ STL headers and implement
 *          automatic conversion from and to `std::stirng_view` and `std::basic_string<any_allocator>`.
 */
#ifndef SZ_AVOID_STL
#define SZ_AVOID_STL (0) // true or false
#endif

/*  We need to detect the version of the C++ language we are compiled with.
 *  This will affect recent features like `operator<=>` and tests against STL.
 */
#if __cplusplus >= 202101L
#define _SZ_IS_CPP23 1
#else
#define _SZ_IS_CPP23 0
#endif
#if __cplusplus >= 202002L
#define _SZ_IS_CPP20 1
#else
#define _SZ_IS_CPP20 0
#endif
#if __cplusplus >= 201703L
#define _SZ_IS_CPP17 1
#else
#define _SZ_IS_CPP17 0
#endif
#if __cplusplus >= 201402L
#define _SZ_IS_CPP14 1
#else
#define _SZ_IS_CPP14 0
#endif
#if __cplusplus >= 201103L
#define _SZ_IS_CPP11 1
#else
#define _SZ_IS_CPP11 0
#endif
#if __cplusplus >= 199711L
#define _SZ_IS_CPP98 1
#else
#define _SZ_IS_CPP98 0
#endif

/**
 *  @brief  Expands to `constexpr` in C++20 and later, and to nothing in older C++ versions.
 *          Useful for STL conversion operators, as several `std::string` members are `constexpr` in C++20.
 *
 *  The `constexpr` keyword has different applicability scope in different C++ versions.
 *  - C++11: Introduced `constexpr`, but no loops or multiple `return` statements were allowed.
 *  - C++14: Allowed loops, multiple statements, and local variables in `constexpr` functions.
 *  - C++17: Added the `if constexpr` construct for compile-time branching.
 *  - C++20: Added some dynamic memory allocations, `virtual` functions, and `try`/`catch` blocks.
 */
#if _SZ_IS_CPP14
#define sz_constexpr_if_cpp14 constexpr
#else
#define sz_constexpr_if_cpp14
#endif
#if _SZ_IS_CPP20
#define sz_constexpr_if_cpp20 constexpr
#else
#define sz_constexpr_if_cpp20
#endif

#if !SZ_AVOID_STL
#include <initializer_list> // `std::initializer_list` is only ~100 LOC
#endif

namespace ashvardanian {
namespace stringzilla {

using i8_t = sz_i8_t;
using u8_t = sz_u8_t;
using u16_t = sz_u16_t;
using i32_t = sz_i32_t;
using u32_t = sz_u32_t;
using u64_t = sz_u64_t;
using i64_t = sz_i64_t;
using size_t = sz_size_t;
using ssize_t = sz_ssize_t;

using ptr_t = sz_ptr_t;
using cptr_t = sz_cptr_t;
using error_cost_t = sz_error_cost_t;

using bool_t = sz_bool_t;
using ordering_t = sz_ordering_t;
using rune_length_t = sz_rune_length_t;
using rune_t = sz_rune_t;
using sorted_idx_t = sz_sorted_idx_t;

/** @sa sz_status_t */
enum class status_t {
    success_k = sz_success_k,
    bad_alloc_k = sz_bad_alloc_k,
    invalid_utf8_k = sz_invalid_utf8_k,
    contains_duplicates_k = sz_contains_duplicates_k,
    unknown_k = sz_status_unknown_k,
};

/**
 *  @brief A trivial function object for uniform character substitution costs in Levenshtein-like similarity algorithms.
 *  @sa error_costs_256x256_t, error_costs_26x26ascii_t
 */
struct error_costs_uniform_t {
    constexpr error_cost_t operator()(char a, char b) const noexcept { return a == b ? 0 : 1; }
};

template <typename value_type_>
struct span {
    using value_type = value_type_;     // ? For STL compatibility
    using size_type = sz_size_t;        // ? For STL compatibility
    using difference_type = sz_ssize_t; // ? For STL compatibility

    value_type *data_ {};
    size_type size_ {};

    constexpr value_type *begin() const noexcept { return data_; }
    constexpr value_type *end() const noexcept { return data_ + size_; }
    constexpr value_type *data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type length() const noexcept { return size_; }
    constexpr value_type &operator[](size_type i) const noexcept { return data_[i]; }
};

template <typename value_type_>
struct dummy_alloc {
    using value_type = value_type_;     // ? For STL compatibility
    using pointer = value_type *;       // ? For STL compatibility
    using size_type = size_t;           // ? For STL compatibility
    using difference_type = sz_ssize_t; // ? For STL compatibility

    template <typename other_value_type_>
    struct rebind {
        using other = dummy_alloc<other_value_type_>;
    };

    constexpr dummy_alloc() noexcept = default;
    constexpr dummy_alloc(dummy_alloc const &) noexcept = default;

    template <typename other_value_type_>
    constexpr dummy_alloc(dummy_alloc<other_value_type_> const &) noexcept {}

    sz_constexpr_if_cpp14 value_type *allocate(size_type) const noexcept { return nullptr; }
    sz_constexpr_if_cpp14 void deallocate(pointer, size_type) const noexcept {}

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

/**
 *  @brief  Apache @b Arrow-compatible tape data-structure to store a sequence of variable length strings.
 *          Doesn't own the memory, but provides a view to the strings stored in a contiguous memory block.
 *  @sa     arrow_strings_tape
 */
template <typename char_type_, typename offset_type_>
struct arrow_strings_view {
    using char_t = char_type_;
    using offset_t = offset_type_;
    using value_t = span<char_t>;
    using value_type = value_t; // ? For STL compatibility

    span<char_t> buffer_;
    span<offset_t> offsets_;

    constexpr arrow_strings_view() noexcept : buffer_ {}, offsets_ {} {}
    constexpr arrow_strings_view(span<char_t> buf, span<offset_t> offs) noexcept : buffer_(buf), offsets_(offs) {}
    constexpr size_t size() const noexcept { return offsets_.size() - 1; }

    constexpr span<char_t> operator[](size_t i) const noexcept {
        return {&buffer_[offsets_[i]], offsets_[i + 1] - offsets_[i] - 1};
    }
};

/**
 *  @brief  Apache @b Arrow-compatible tape data-structure to store a sequence of variable length strings.
 *          Each string is appended to a contiguous memory block, delimited by the NULL character.
 *          Provides @b ~O(1) access to each string by storing the offsets of each string in a separate array.
 */
template <typename char_type_, typename offset_type_, typename allocator_type_>
struct arrow_strings_tape {
    using char_t = char_type_;
    using offset_t = offset_type_;
    using allocator_t = allocator_type_;

    using value_t = span<char_t>;
    using view_t = arrow_strings_view<char_t, offset_t>;
    using value_type = value_t; // ? For STL compatibility

    using char_alloc_t = typename allocator_t::template rebind<char_t>::other;
    using offset_alloc_t = typename allocator_t::template rebind<offset_t>::other;

  private:
    span<char_t> buffer_;
    span<offset_t> offsets_;
    char_alloc_t char_alloc_;
    offset_alloc_t offset_alloc_;
    size_t count_ = 0;

  public:
    constexpr arrow_strings_tape() = default;
    constexpr arrow_strings_tape(arrow_strings_tape const &) = delete;
    constexpr arrow_strings_tape &operator=(arrow_strings_tape const &other) = delete;

    constexpr arrow_strings_tape(arrow_strings_tape &&) = delete;
    constexpr arrow_strings_tape &operator=(arrow_strings_tape &&) = delete;

    constexpr arrow_strings_tape(span<char_t> buffer, span<offset_t> offsets, allocator_t alloc)
        : buffer_(buffer), offsets_(offsets), char_alloc_(alloc), offset_alloc_(alloc) {}

    sz_constexpr_if_cpp20 ~arrow_strings_tape() noexcept { reset(); }
    sz_constexpr_if_cpp20 void reset() noexcept {
        if (buffer_.data_) char_alloc_.deallocate(const_cast<char_t *>(buffer_.data_), buffer_.size_), buffer_ = {};
        if (offsets_.data_)
            offset_alloc_.deallocate(const_cast<offset_t *>(offsets_.data_), offsets_.size_), offsets_ = {};
        count_ = 0;
    }

    template <typename strings_iterator_type_>
    status_t try_assign(strings_iterator_type_ first, strings_iterator_type_ last) noexcept {
        // Deallocate previous memory if allocated
        if (buffer_.data_ && buffer_.size_)
            char_alloc_.deallocate(const_cast<char_t *>(buffer_.data_), buffer_.size_), buffer_ = {};

        if (offsets_.data_ && offsets_.size_)
            offset_alloc_.deallocate(const_cast<offset_t *>(offsets_.data_), offsets_.size_), offsets_ = {};

        // Estimate required memory: total characters + one extra per string for the NULL.
        size_t count = 0;
        size_t combined_length = 0;
        for (auto it = first; it != last; ++it, ++count) combined_length += it->length();
        combined_length += count; // ? NULL-terminate every string

        // Allocate exactly the required memory
        buffer_ = {char_alloc_.allocate(combined_length), combined_length};
        offsets_ = {offset_alloc_.allocate(count + 1), count + 1};
        if (!buffer_.data_ || !offsets_.data_) return status_t::bad_alloc_k;

        // Copy the strings to the buffer and store the offsets
        char_t *buffer_ptr = buffer_.data_;
        offset_t *offsets_ptr = offsets_.data_;
        for (auto it = first; it != last; ++it) {
            *offsets_ptr++ = static_cast<offset_t>(buffer_ptr - buffer_.data_);
            // Perform a byte-level copy of the string, similar to `sz_copy`
            char_t const *from_ptr = it->data();
            size_t const from_length = it->length();
            for (size_t i = 0; i != from_length; ++i) *buffer_ptr++ = *from_ptr++;
            *buffer_ptr++ = '\0'; // ? NULL-terminated
        }
        *offsets_ptr = static_cast<offset_t>(buffer_ptr - buffer_.data_);
        count_ = count;
        return status_t::success_k;
    }

#if !SZ_AVOID_STL
    template <typename string_convertible_type_>
    status_t try_assign(std::initializer_list<string_convertible_type_> inits) noexcept {
        return try_assign(inits.begin(), inits.end());
    }
#endif

    status_t try_append(span<char_t const> string) noexcept {
        size_t const string_length = string.length();
        size_t const required = string_length + 1; // Space needed for the new string and its NULL
        size_t current_used = count_ > 0 ? offsets_.data_[count_] : 0;

        // Reallocate the buffer if needed (oversubscribe in powers of two).
        if (current_used + required > buffer_.size_) {
            size_t new_capacity = sz_size_bit_ceil(current_used + required);
            char_t *new_buffer = char_alloc_.allocate(new_capacity);
            if (!new_buffer) return status_t::bad_alloc_k;
            if (buffer_.data_) {
                // Copy the existing data to the new array, before deallocating the old one.
                char_t const *src = buffer_.data_, *end = buffer_.data_ + current_used;
                char_t *tgt = new_buffer;
                for (; src != end; ++src, ++tgt) *tgt = *src;
                char_alloc_.deallocate(const_cast<char_t *>(buffer_.data_), buffer_.size_);
            }
            buffer_.data_ = new_buffer;
            buffer_.size_ = new_capacity;
        }

        // Reallocate the offsets array if needed.
        if (count_ + 1 >= offsets_.size_) { // need one extra slot for the new offset
            size_t new_offsets_capacity = sz_size_bit_ceil(count_ + 1);
            offset_t *new_offsets = offset_alloc_.allocate(new_offsets_capacity);
            if (!new_offsets) return status_t::bad_alloc_k;
            if (offsets_.data_) {
                // Copy the existing offsets to the new array, before deallocating the old one.
                offset_t const *src = offsets_.data_, *end = offsets_.data_ + count_ + 1;
                offset_t *tgt = new_offsets;
                for (; src != end; ++src, ++tgt) *tgt = *src;
                offset_alloc_.deallocate(const_cast<offset_t *>(offsets_.data_), offsets_.size_);
            }
            offsets_.data_ = new_offsets;
            offsets_.size_ = new_offsets_capacity;
        }

        // Record the starting offset for the new string.
        offsets_.data_[count_] = static_cast<offset_t>(current_used);
        // Copy the string into the buffer.
        for (size_t i = 0; i < string_length; ++i) buffer_.data_[current_used++] = string[i];
        // Append the NULL terminator.
        buffer_.data_[current_used++] = '\0';
        // Update the offsets array with the new end-of-buffer position.
        offsets_.data_[++count_] = static_cast<offset_t>(current_used);
        return status_t::success_k;
    }

    constexpr value_type operator[](size_t i) const noexcept {
        _sz_assert(i < count_ && "Index out of bounds");
        return {buffer_.data_ + offsets_.data_[i], offsets_.data_[i + 1] - offsets_.data_[i] - 1};
    }

    constexpr size_t size() const noexcept { return count_; }
    constexpr view_t view() const noexcept { return {buffer_, {offsets_.data_, count_ + 1}}; }
    constexpr span<char_t> const &buffer() const noexcept { return buffer_; }
    constexpr span<offset_t> const &offsets() const noexcept { return offsets_; }
};

/**
 *  @brief  Similar to `thrust::constant_iterator`, always returning the same value.
 */
template <typename value_type_>
struct constant_iterator {

    using value_type = value_type_;
    using reference = value_type_ const &;
    using pointer = value_type_ const *;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    constant_iterator(value_type const &value, difference_type pos = 0) noexcept : value_(value), pos_(pos) {}

    reference operator*() const { return value_; }
    pointer operator->() const { return &value_; }

    constant_iterator &operator++() {
        ++pos_;
        return *this;
    }
    constant_iterator operator++(int) {
        constant_iterator tmp(*this);
        ++pos_;
        return tmp;
    }
    constant_iterator &operator--() {
        --pos_;
        return *this;
    }
    constant_iterator operator--(int) {
        constant_iterator tmp(*this);
        --pos_;
        return tmp;
    }

    constant_iterator operator+(difference_type n) const { return constant_iterator(value_, pos_ + n); }
    constant_iterator &operator+=(difference_type n) {
        pos_ += n;
        return *this;
    }
    constant_iterator operator-(difference_type n) const { return constant_iterator(value_, pos_ - n); }
    constant_iterator &operator-=(difference_type n) {
        pos_ -= n;
        return *this;
    }
    difference_type operator-(constant_iterator const &other) const { return pos_ - other.pos_; }

    reference operator[](difference_type) const { return value_; }
    bool operator==(constant_iterator const &other) const { return pos_ == other.pos_; }
    bool operator!=(constant_iterator const &other) const { return pos_ != other.pos_; }
    bool operator<(constant_iterator const &other) const { return pos_ < other.pos_; }
    bool operator>(constant_iterator const &other) const { return pos_ > other.pos_; }
    bool operator<=(constant_iterator const &other) const { return pos_ <= other.pos_; }
    bool operator>=(constant_iterator const &other) const { return pos_ >= other.pos_; }

  private:
    value_type value_;
    difference_type pos_;
};

template <typename first_, typename second_>
struct is_same_type;

template <typename first_>
struct is_same_type<first_, first_> {
    static constexpr bool value = true;
};

template <typename first_, typename second_>
struct is_same_type {
    static_assert(std::is_same<first_, second_>::value, "First and second types differ!");
    static constexpr bool value = false;
};

struct gpu_specs_t {
    size_t total_sm_count = 108;              // ? On A100
    size_t blocks_per_sm = 128;               // ? Each, generally, with 32 threads
    size_t shared_memory_per_sm = 192 * 1024; // ? On A100 it's 192 KB per SM
};

struct cpu_specs_t {
    size_t l1_bytes = 32 * 1024;       // ? typically around 32 KB
    size_t l2_bytes = 256 * 1024;      // ? typically around 256 KB
    size_t l3_bytes = 8 * 1024 * 1024; // ? typically around 8 MB
    size_t cache_line_width = 64;      // ? 64 bytes on x86, sometimes 128 on ARM
    size_t cores_per_socket = 1;       // ? at least 1 core
    size_t sockets = 1;                // ? at least 1 socket
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_TYPES_HPP_
