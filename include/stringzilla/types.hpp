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
 *          automatic conversion from and to `std::string_view` and `std::basic_string<any_allocator>`.
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
#include <iterator>         // `std::random_access_iterator_tag` pulls 20K LOC
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
    constexpr error_cost_t operator()(sz_rune_t a, sz_rune_t b) const noexcept { return a == b ? 0 : 1; }
    constexpr sz_size_t max_magnitude_change() const noexcept { return 1; }
};

template <typename value_type_>
struct span {
    using value_type = value_type_;     // ? For STL compatibility
    using size_type = sz_size_t;        // ? For STL compatibility
    using difference_type = sz_ssize_t; // ? For STL compatibility

    value_type *data_ {};
    size_type size_ {};

    constexpr span() noexcept = default;
    constexpr span(value_type *data, size_type size) noexcept : data_(data), size_(size) {}
    constexpr span(value_type *data, value_type *end) noexcept : data_(data), size_(end - data) {}

    constexpr value_type *begin() const noexcept { return data_; }
    constexpr value_type *end() const noexcept { return data_ + size_; }
    constexpr value_type *data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type length() const noexcept { return size_; }
    constexpr size_type size_bytes() const noexcept { return size_ * sizeof(value_type); }
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
 *  @brief  Random access iterator for any immutable container with indexed element lookup support.
 *  @note   Designed for `arrow_strings_tape` and `arrow_strings_view` compatibility with STL algorithms and ranges.
 */
template <typename container_type_>
struct indexed_container_iterator {
    using container_t = container_type_;
    using value_t = typename container_t::value_type;

    using difference_type = sz_ssize_t;
    using value_type = value_t;
    using reference = value_t; // ! As our view returns by value
    using pointer = void;      // ! Not providing direct pointer semantics

#if !SZ_AVOID_STL
    using iterator_category = std::random_access_iterator_tag;
#endif

  private:
    container_t const *parent_;
    size_t index_;

  public:
    constexpr indexed_container_iterator() noexcept : parent_(nullptr), index_(0) {}
    constexpr indexed_container_iterator(container_t const &parent, size_t index) noexcept
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

/**
 *  @brief  Apache @b Arrow-compatible tape data-structure to store a sequence of variable length strings.
 *          Doesn't own the memory, but provides a view to the strings stored in a contiguous memory block.
 *  @sa     arrow_strings_tape
 */
template <typename char_type_, typename offset_type_>
struct arrow_strings_view {
    using char_t = char_type_;
    using offset_t = offset_type_;
    using self_t = arrow_strings_view<char_t, offset_t>;

    using value_t = span<char_t const>;
    using value_type = value_t; // ? For STL compatibility
    using iterator_t = indexed_container_iterator<self_t>;
    using iterator = iterator_t; // ? For STL compatibility

    span<char_t const> buffer_;
    span<offset_t const> offsets_;

    constexpr arrow_strings_view() noexcept : buffer_ {}, offsets_ {} {}
    constexpr arrow_strings_view(span<char_t const> buf, span<offset_t const> offs) noexcept
        : buffer_(buf), offsets_(offs) {}

    constexpr size_t size() const noexcept { return offsets_.size() - 1; }
    constexpr value_t operator[](size_t i) const noexcept {
        return {&buffer_[offsets_[i]], offsets_[i + 1] - offsets_[i] - 1};
    }

    constexpr iterator_t begin() const noexcept { return iterator_t(*this, 0); }
    constexpr iterator_t end() const noexcept { return iterator_t(*this, size()); }
    constexpr iterator_t cbegin() const noexcept { return begin(); }
    constexpr iterator_t cend() const noexcept { return end(); }
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
    using self_t = arrow_strings_tape<char_t, offset_t, allocator_t>;

    using value_t = span<char_t const>;
    using view_t = arrow_strings_view<char_t, offset_t>;
    using value_type = value_t; // ? For STL compatibility
    using iterator_t = indexed_container_iterator<self_t>;
    using iterator = iterator_t; // ? For STL compatibility

#if _SZ_IS_CPP17
    using char_alloc_t = typename std::allocator_traits<allocator_t>::rebind_alloc<char_t>;
    using offset_alloc_t = typename std::allocator_traits<allocator_t>::rebind_alloc<offset_t>;
#else
    using char_alloc_t = typename allocator_t::template rebind<char_t>::other;
    using offset_alloc_t = typename allocator_t::template rebind<offset_t>::other;
#endif

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

    constexpr iterator_t begin() const noexcept { return iterator_t(*this, 0); }
    constexpr iterator_t end() const noexcept { return iterator_t(*this, size()); }
    constexpr iterator_t cbegin() const noexcept { return begin(); }
    constexpr iterator_t cend() const noexcept { return end(); }

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
    constexpr view_t view() const noexcept { return {{buffer_.data(), buffer_.size()}, {offsets_.data_, count_ + 1}}; }
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
    using difference_type = sz_ssize_t;
#if !SZ_AVOID_STL
    using iterator_category = std::random_access_iterator_tag;
#endif

    constexpr constant_iterator(value_type const &value, difference_type pos = 0) noexcept : value_(value), pos_(pos) {}
    constexpr reference operator*() const { return value_; }
    constexpr pointer operator->() const { return &value_; }

    constexpr constant_iterator &operator++() {
        ++pos_;
        return *this;
    }
    constexpr constant_iterator operator++(int) {
        constexpr constant_iterator tmp(*this);
        ++pos_;
        return tmp;
    }
    constexpr constant_iterator &operator--() {
        --pos_;
        return *this;
    }
    constexpr constant_iterator operator--(int) {
        constexpr constant_iterator tmp(*this);
        --pos_;
        return tmp;
    }

    constexpr constant_iterator operator+(difference_type n) const { return constant_iterator(value_, pos_ + n); }
    constexpr constant_iterator &operator+=(difference_type n) {
        pos_ += n;
        return *this;
    }
    constexpr constant_iterator operator-(difference_type n) const { return constant_iterator(value_, pos_ - n); }
    constexpr constant_iterator &operator-=(difference_type n) {
        pos_ -= n;
        return *this;
    }
    constexpr difference_type operator-(constant_iterator const &other) const { return pos_ - other.pos_; }

    constexpr reference operator[](difference_type) const { return value_; }
    constexpr bool operator==(constant_iterator const &other) const { return pos_ == other.pos_; }
    constexpr bool operator!=(constant_iterator const &other) const { return pos_ != other.pos_; }
    constexpr bool operator<(constant_iterator const &other) const { return pos_ < other.pos_; }
    constexpr bool operator>(constant_iterator const &other) const { return pos_ > other.pos_; }
    constexpr bool operator<=(constant_iterator const &other) const { return pos_ <= other.pos_; }
    constexpr bool operator>=(constant_iterator const &other) const { return pos_ >= other.pos_; }

  private:
    value_type value_;
    difference_type pos_;
};

template <typename begin_type_, typename end_type_>
struct random_access_range {

    using value_type = typename std::iterator_traits<begin_type_>::value_type;
    using reference_type = typename std::iterator_traits<begin_type_>::reference;
    using difference_type = typename std::iterator_traits<begin_type_>::difference_type;

    begin_type_ begin_;
    end_type_ end_;

    constexpr std::size_t size() const { return static_cast<std::size_t>(end_ - begin_); }
    constexpr begin_type_ begin() const { return begin_; }
    constexpr end_type_ end() const { return end_; }

    decltype(auto) operator[](std::size_t index) const {
        _sz_assert(index < size());
        return *(begin_ + index);
    }
};

template <typename begin_type_, typename end_type_>
random_access_range(begin_type_, end_type_) -> random_access_range<begin_type_, end_type_>;

template <typename value_type_, size_t count_>
struct safe_array {
    using value_type = value_type_;
    using size_type = size_t;
    using iterator = value_type *;
    using const_iterator = value_type const *;
    static constexpr size_type count_k = count_;

    value_type data_[count_k] = {};

    constexpr value_type &operator[](size_type i) noexcept { return data_[i]; }
    constexpr value_type const &operator[](size_type i) const noexcept { return data_[i]; }
    constexpr size_type size() const noexcept { return count_k; }
    constexpr value_type *data() noexcept { return data_; }
    constexpr value_type const *data() const noexcept { return data_; }
    constexpr iterator begin() noexcept { return data_; }
    constexpr const_iterator begin() const noexcept { return data_; }
    constexpr iterator end() noexcept { return data_ + count_k; }
    constexpr const_iterator end() const noexcept { return data_ + count_k; }
};

/**
 *  @brief  Safer alternative to `std::vector`, that avoids exceptions, copy constructors,
 *          and provides alternative `try_push_back` and `try_reserve` for faulty memory allocations.
 */
template <typename value_type_, typename allocator_type_>
class safe_vector {
  public:
    using value_type = value_type_;
    using size_type = std::size_t;
    using allocator_type = allocator_type_;

  private:
    value_type *data_;
    size_type size_;
    size_type capacity_;
    allocator_type alloc_;

  public:
    safe_vector() noexcept : data_(nullptr), size_(0), capacity_(0), alloc_() {}
    safe_vector(allocator_type alloc) noexcept : data_(nullptr), size_(0), capacity_(0), alloc_(alloc) {}
    ~safe_vector() noexcept { reset(); }

    void clear() noexcept {
        if constexpr (!std::is_trivially_destructible<value_type>::value)
            for (size_type i = 0; i < size_; ++i) data_[i].~value_type();
        size_ = 0;
    }

    void reset() noexcept {
        clear();
        if (data_) alloc_.deallocate(data_, capacity_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    safe_vector(safe_vector const &other) = delete;
    safe_vector &operator=(safe_vector const &other) noexcept = delete;

    safe_vector(safe_vector &&other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_), alloc_(std::move(other.alloc_)) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    safe_vector &operator=(safe_vector &&other) noexcept {
        if (this != &other) {
            clear();
            if (data_) alloc_.deallocate(data_, capacity_);
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

    status_t try_reserve(size_type new_cap) noexcept {
        if (new_cap <= capacity_) return status_t::success_k;
        value_type *new_data = alloc_.allocate(new_cap);
        if (!new_data) return status_t::bad_alloc_k;
        for (size_type i = 0; i < size_; ++i) {
            new (new_data + i) value_type(std::move(data_[i]));
            if constexpr (!std::is_trivially_destructible<value_type>::value) data_[i].~value_type();
        }
        if (data_) alloc_.deallocate(data_, capacity_);
        data_ = new_data;
        capacity_ = new_cap;
        return status_t::success_k;
    }

    status_t try_resize(size_type new_size) noexcept {
        if (new_size > capacity_ && try_reserve(new_size) != status_t::success_k) return status_t::bad_alloc_k;

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

    status_t try_push_back(value_type const &val) noexcept {
        if (size_ == capacity_) {
            size_type new_cap = capacity_ ? capacity_ * 2 : 1;
            if (try_reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        new (data_ + size_) value_type(val);
        ++size_;
        return status_t::success_k;
    }

    status_t try_push_back(value_type &&val) noexcept {
        if (size_ == capacity_) {
            size_type new_cap = capacity_ ? capacity_ * 2 : 1;
            if (try_reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        new (data_ + size_) value_type(std::move(val));
        ++size_;
        return status_t::success_k;
    }

    status_t try_append(span<value_type const> source) noexcept {
        size_type needed = size_ + source.size();
        if (needed > capacity_) {
            size_type new_cap = capacity_ ? capacity_ : 1;
            while (new_cap < needed) new_cap *= 2;
            if (try_reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        for (size_type i = 0; i < source.size(); ++i) new (data_ + size_ + i) value_type(source[i]);
        size_ = needed;
        return status_t::success_k;
    }

    value_type *begin() noexcept { return data_; }
    value_type const *begin() const noexcept { return data_; }
    value_type *end() noexcept { return data_ + size_; }
    value_type const *end() const noexcept { return data_ + size_; }
    value_type &operator[](size_type i) noexcept { return data_[i]; }
    value_type const &operator[](size_type i) const noexcept { return data_[i]; }
    value_type *data() noexcept { return data_; }
    value_type const *data() const noexcept { return data_; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    operator span<value_type>() noexcept { return {data_, size_}; }
    operator span<value_type const>() const noexcept { return {data_, size_}; }
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
    size_t vram_bytes = 40ul * 1024 * 1024 * 1024; // ? On A100 it's 40 GB
    size_t constant_memory_bytes = 64 * 1024;      // ? On A100 it's 64 KB
    size_t shared_memory_bytes = 192 * 1024 * 108; // ? On A100 it's 192 KB per SM
    size_t streaming_multiprocessors = 108;        // ? On A100
    size_t cuda_cores = 6912;                      // ? On A100 for f32/i32 logic
    size_t max_blocks_per_multiprocessor = 0;
    size_t reserved_memory_per_block = 0;

    inline size_t shared_memory_per_multiprocessor() const noexcept {
        return shared_memory_bytes / streaming_multiprocessors;
    }

    inline static size_t cores_per_multiprocessor(int major, int minor) noexcept {
        typedef struct {
            size_t sm;
            size_t cores;
        } generation_to_core_count;
        generation_to_core_count generations_to_core_counts[] = {
            {(7 << 4) + 0, 64},  // Compute Capability 7.0 (V100)
            {(7 << 4) + 5, 64},  // Compute Capability 7.5 (RTX 2080 Ti)
            {(8 << 4) + 0, 64},  // Compute Capability 8.0 (A100)
            {(8 << 4) + 6, 128}, // Compute Capability 8.6 (RTX 3090)
            {(9 << 4) + 0, 128}, // Compute Capability 9.0 (H100)
            {0, 0}};

        // Create a numeric code: for SM 3.5, SM = (3 << 4 + 5) = 0x35.
        size_t sm = ((major << 4) + minor);
        size_t index = 0;
        for (; generations_to_core_counts[index].sm != 0; ++index)
            if (generations_to_core_counts[index].sm == sm) return generations_to_core_counts[index].cores;
        return generations_to_core_counts[index - 1].cores;
    }
};

struct cpu_specs_t {
    size_t l1_bytes = 32 * 1024;       // ? typically around 32 KB
    size_t l2_bytes = 256 * 1024;      // ? typically around 256 KB
    size_t l3_bytes = 8 * 1024 * 1024; // ? typically around 8 MB
    size_t cache_line_width = 64;      // ? 64 bytes on x86, sometimes 128 on ARM
    size_t cores_per_socket = 1;       // ? at least 1 core
    size_t sockets = 1;                // ? at least 1 socket
};

/**
 *  @brief Rounds @p x up to the nearest multiple of @p divisor.
 */
template <typename scalar_type_>
constexpr scalar_type_ round_up_to_multiple(scalar_type_ x, scalar_type_ divisor) {
    _sz_assert(divisor > 0 && "Divisor must be positive");
    return ((x + divisor - 1) / divisor) * divisor;
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_TYPES_HPP_
