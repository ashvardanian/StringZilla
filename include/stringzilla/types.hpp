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
 *  - `span<value_type>` -
 *  - `dummy_alloc_t` -
 *  - `dummy_alloc<value_type>` -
 *  - `arrow_string_tape<char_type, offset_type>` -
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
};

struct uniform_substitution_cost_t {
    constexpr error_cost_t operator()(char a, char b) const noexcept { return a == b ? 0 : 1; }
};

struct lookup_substitution_cost_t {
    error_cost_t const *costs;
    constexpr error_cost_t operator()(char a, char b) const noexcept { return costs[(u8_t)a * 256 + (u8_t)b]; }
};

template <typename value_type_>
struct span {
    using value_type = value_type_;
    using size_type = sz_size_t;
    using difference_type = sz_ssize_t;

    value_type *data_;
    size_type size_;

    constexpr value_type *begin() const noexcept { return data_; }
    constexpr value_type *end() const noexcept { return data_ + size_; }
    constexpr value_type *data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type length() const noexcept { return size_; }
};

template <typename value_type_>
struct dummy_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = size_t;
    using difference_type = sz_ssize_t;

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
 *          Each string is appended to a contiguous memory block, delimited by the NULL character.
 *          Provides @b ~O(1) access to each string by storing the offsets of each string in a separate array.
 */
template <typename char_type_, typename offset_type_, typename allocator_type_>
struct arrow_string_tape {
    using char_type = char_type_;
    using offset_type = offset_type_;
    using allocator_type = allocator_type_;

    using value_type = span<char_type>;
    using view_type = arrow_string_tape<char_type, offset_type, dummy_alloc<char_type>>;

    using char_alloc_t = typename allocator_type::template rebind<char_type>::other;
    using offset_alloc_t = typename allocator_type::template rebind<offset_type>::other;

  private:
    span<char_type> buffer_;
    span<offset_type> offsets_;
    char_alloc_t char_alloc_;
    offset_alloc_t offset_alloc_;

  public:
    constexpr arrow_string_tape() = default;
    constexpr arrow_string_tape(span<char_type> buffer, span<offset_type> offsets, allocator_type alloc)
        : buffer_(buffer), offsets_(offsets), char_alloc_(alloc), offset_alloc_(alloc) {}

    template <typename strings_iterator_>
    sz_constexpr_if_cpp14 status_t try_assign(strings_iterator_ first, strings_iterator_ last) noexcept {
        // Deallocate the previous memory if it was allocated
        if (buffer_.data_) char_alloc_.deallocate(const_cast<char_type *>(buffer_.data_), buffer_.size_);
        if (offsets_.data_) offset_alloc_.deallocate(const_cast<offset_type *>(offsets_.data_), offsets_.size_);

        // Estimate the required memory size
        size_t buffer_capacity = 0;
        size_t max_count = 0;
        for (auto it = first; it != last; ++it) {
            buffer_capacity += it->size() + 1; // ? NULL-terminated
            ++max_count;
        }
        buffer_ = {char_alloc_.allocate(buffer_capacity), buffer_capacity};
        offsets_ = {offset_alloc_.allocate(max_count), max_count};
        if (!buffer_.data_ || !offsets_.data_) return status_t::bad_alloc_k;

        // Copy the strings to the buffer and store the offsets
        auto buffer_ptr = buffer_.data_;
        auto offsets_ptr = offsets_.data_;
        for (auto it = first; it != last; ++it) {
            *offsets_ptr++ = buffer_ptr - buffer_.data_;
            // Perform a byte-level copy of the string, similar to `sz_copy`
            for (size_t i = 0; i != it->size(); ++i) buffer_ptr[i] = it->data()[i];
            buffer_ptr[it->size()] = '\0'; // ? NULL-terminated
            buffer_ptr += it->size() + 1;
        }
        *offsets_ptr = static_cast<offset_type>(buffer_ptr - buffer_.data_);
        return status_t::success_k;
    }

    sz_constexpr_if_cpp20 ~arrow_string_tape() noexcept {
        if (buffer_.data_) char_alloc_.deallocate(const_cast<char_type *>(buffer_.data_), buffer_.size_);
        if (offsets_.data_) offset_alloc_.deallocate(const_cast<offset_type *>(offsets_.data_), offsets_.size_);
    }

    constexpr value_type operator[](size_t i) const noexcept {
        return {buffer_.data_ + offsets_.data_[i], offsets_.data_[i + 1] - offsets_.data_[i] - 1};
    }

    constexpr size_t size() const noexcept { return offsets_.size_ - 1; }
    constexpr view_type view() const noexcept { return {buffer_, offsets_, dummy_alloc_t {}}; }

    constexpr span<char_type> const &buffer() const noexcept { return buffer_; }
    constexpr span<offset_type> const &offsets() const noexcept { return offsets_; }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_TYPES_HPP_
