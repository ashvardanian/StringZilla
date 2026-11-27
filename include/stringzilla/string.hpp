/**
 *  @file include/stringzilla/string.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief String views, spans, and the owning @c string with a Small String Optimization.
 *
 *  The containers hold, slice, compare and grow strings, and call no kernel of the library: every
 *  operation a kernel backs, like @c find or @c hash, is a free function of the C++ SDK in
 *  <tt>cpp/stringzilla/</tt>, and its member spelling is deleted here, naming that function.
 */
#ifndef STRINGZILLA_STRING_HPP_
#define STRINGZILLA_STRING_HPP_

#include <cstdint> // `std::uint64_t`
#include <cstring> // `std::memcpy`, `std::memmove`, `std::memset`, `std::strlen`

#include <compare>     // `std::strong_ordering`
#include <concepts>    // `std::convertible_to`
#include <iosfwd>      // `std::basic_ostream`
#include <string>      // vendor-boundary `std::string`, `std::char_traits`
#include <string_view> // `std::string_view`

#include "stringzilla/small_string.h" // `sz_string_t`, `sz_string_init`, `sz_string_free`

#include "stringzilla/sets.hpp"
#include "stringzilla/types.hpp"

/**
 *  @brief For higher safety, we annotate the lifetime bound of the returned string slices.
 *  @see Clang lifetimebound attribute: https://clang.llvm.org/docs/AttributeReference.html#id11
 *  @see Lifetime bounds: https://lemire.me/blog/2024/07/26/safer-code-in-c-with-lifetime-bounds/
 */
#if !defined(__has_cpp_attribute)
#define sz_lifetime_bound_
#elif __has_cpp_attribute(msvc::lifetimebound)
#define sz_lifetime_bound_ [[msvc::lifetimebound]]
#elif __has_cpp_attribute(clang::lifetimebound)
#define sz_lifetime_bound_ [[clang::lifetimebound]]
#elif __has_cpp_attribute(lifetimebound)
#define sz_lifetime_bound_ [[lifetimebound]]
#else
#define sz_lifetime_bound_
#endif

/** Lets an empty allocator member occupy no bytes; MSVC ignores the standard spelling. */
#if defined(_MSC_VER)
#define sz_no_unique_address_ [[msvc::no_unique_address]]
#else
#define sz_no_unique_address_ [[no_unique_address]]
#endif

namespace ashvardanian {
namespace stringzilla {

template <typename>
class string_slice;

using string_span_t = string_slice<char>;
using string_view_t = string_slice<char const>;

/** Helper function using @c range_length to compute the unsigned distance. */
template <typename iterator_type_>
std::size_t range_length(iterator_type_ first, iterator_type_ last) {
    return static_cast<std::size_t>(std::distance(first, last));
}

/** A reverse iterator for mutable and immutable character buffers.
 *  Replaces @c std::reverse_iterator to avoid including <tt>&lt;iterator&gt;</tt>. */
template <typename value_type_>
class reversed_iterator_for {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = value_type_;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type_ *;
    using reference = value_type_ &;

    constexpr reversed_iterator_for(pointer ptr) noexcept : ptr_(ptr) {}
    constexpr reference operator*() const noexcept { return *ptr_; }

    constexpr bool operator==(reversed_iterator_for const &other) const noexcept { return ptr_ == other.ptr_; }
    constexpr bool operator!=(reversed_iterator_for const &other) const noexcept { return ptr_ != other.ptr_; }
    constexpr reference operator[](difference_type n) const noexcept { return *(*this + n); }
    constexpr reversed_iterator_for operator+(difference_type n) const noexcept {
        return reversed_iterator_for(ptr_ - n);
    }
    constexpr reversed_iterator_for operator-(difference_type n) const noexcept {
        return reversed_iterator_for(ptr_ + n);
    }
    constexpr difference_type operator-(reversed_iterator_for const &other) const noexcept { return other.ptr_ - ptr_; }

    constexpr reversed_iterator_for &operator++() noexcept {
        --ptr_;
        return *this;
    }

    constexpr reversed_iterator_for operator++(int) noexcept {
        reversed_iterator_for temp = *this;
        --ptr_;
        return temp;
    }

    constexpr reversed_iterator_for &operator--() noexcept {
        ++ptr_;
        return *this;
    }

    constexpr reversed_iterator_for operator--(int) noexcept {
        reversed_iterator_for temp = *this;
        ++ptr_;
        return temp;
    }

  private:
    value_type_ *ptr_;
};

template <typename first_type_, typename second_type_>
struct concatenation;

/**
 *  @brief Infers how an operand is stored inside a @c concatenation expression.
 *
 *  The @c concatenate function and @c operator| take forwarding references, so an lvalue
 *  @c sz::string_t deduces to <tt>sz::string_t &</tt> and a literal deduces to <tt>char const (&)[N]</tt>.
 *  Neither exposes the @c value_type, <tt>size()</tt> and <tt>copy()</tt> interface the expression template
 *  relies on, which is why baking the deduced type straight into @c concatenation fails to compile.
 *  Everything that is not already a nested expression is therefore normalized into a
 *  @c string_view_t, matching what the @c string_slice::operator| overloads produce.
 */
template <typename type_>
struct concatenation_operand {
    using type = string_view_t;
};

template <typename first_type_, typename second_type_>
struct concatenation_operand<concatenation<first_type_, second_type_>> {
    using type = concatenation<first_type_, second_type_>;
};

template <typename first_type_, typename second_type_>
struct concatenation_operand<concatenation<first_type_, second_type_> &>
    : concatenation_operand<concatenation<first_type_, second_type_>> {};

template <typename first_type_, typename second_type_>
struct concatenation_operand<concatenation<first_type_, second_type_> const &>
    : concatenation_operand<concatenation<first_type_, second_type_>> {};

/**
 *  @brief An "expression template" for lazy concatenation of strings using the @c operator|.
 *  @see https://en.wikipedia.org/wiki/Expression_templates
 *  @sa concatenate function for usage examples.
 */
template <typename first_type_, typename second_type_>
struct concatenation {

    using value_type = typename first_type_::value_type;
    using pointer = value_type *;
    using const_pointer = value_type const *;
    using size_type = typename first_type_::size_type;
    using difference_type = typename first_type_::difference_type;

    first_type_ first;
    second_type_ second;

    std::size_t size() const noexcept { return first.size() + second.size(); }
    std::size_t length() const noexcept { return first.size() + second.size(); }

    size_type copy(pointer destination) const noexcept {
        first.copy(destination);
        second.copy(destination + first.length());
        return first.length() + second.length();
    }

    size_type copy(pointer destination, size_type length) const noexcept {
        auto first_length = min_of(first.length(), length);
        auto second_length = min_of(second.length(), length - first_length);
        first.copy(destination, first_length);
        second.copy(destination + first_length, second_length);
        return first_length + second_length;
    }

    template <typename last_type_>
    concatenation<concatenation<first_type_, second_type_>, typename concatenation_operand<last_type_>::type> operator|(
        last_type_ &&last) const {
        return {*this, last};
    }
};

#pragma region String Views and Spans

/**
 *  @brief A string slice (view/span) class implementing a superset of C++23 functionality
 *      with much faster SIMD-accelerated substring search and approximate matching.
 *
 *  Construction, element access, slicing and comparisons are @c constexpr, serving @c _sv
 *  literals and kernels built with nvcc's @c --expt-relaxed-constexpr.
 *
 *  @tparam char_type_ The character type, usually <tt>char const</tt> or @c char, and a single byte
 *      long in any case.
 */
template <typename char_type_>
class string_slice {

    using char_type = char_type_;
    using mutable_char_type = typename std::remove_const<char_type_>::type;
    using immutable_char_type = typename std::add_const<char_type_>::type;

    char_type *start_;
    std::size_t length_;

  public:
    /*  STL compatibility */
    using traits_type = std::char_traits<mutable_char_type>;
    using value_type = mutable_char_type;
    using pointer = char_type *;
    using const_pointer = immutable_char_type *;
    using reference = char_type &;
    using const_reference = immutable_char_type &;
    using const_iterator = const_pointer;
    using iterator = pointer;
    using reverse_iterator = reversed_iterator_for<char_type>;
    using const_reverse_iterator = reversed_iterator_for<immutable_char_type>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    /*  Non-STL type definitions */
    using string_slice_t = string_slice<char_type>;
    using string_span_t = string_slice<mutable_char_type>;
    using string_view_t = string_slice<immutable_char_type>;

    /**
     *  @brief Special value for missing matches.
     *
     *  We take the largest 63-bit unsigned integer on 64-bit machines.
     *  We take the largest 31-bit unsigned integer on 32-bit machines.
     */
    static constexpr size_type npos = STRINGZILLA_SSIZE_MAX;

#pragma region Constructors and STL Utilities

    constexpr string_slice() noexcept : start_(nullptr), length_(0) {}
    constexpr string_slice(pointer c_string) noexcept : start_(c_string), length_(null_terminated_length(c_string)) {}
    constexpr string_slice(pointer c_string, size_type length) noexcept : start_(c_string), length_(length) {}

    string_slice(string_slice const &other) noexcept = default;
    string_slice &operator=(string_slice const &other) noexcept = default;
    string_slice(std::nullptr_t) = delete;

    /** Exchanges the view with that of the @p other. */
    void swap(string_slice_t &other) noexcept {
        trivial_swap(start_, other.start_);
        trivial_swap(length_, other.length_);
    }

    constexpr string_slice(std::string const &other) noexcept // vendor-boundary
        requires std::is_const_v<char_type>
        : string_slice(other.data(), other.size()) {}

    constexpr string_slice(std::string &other) noexcept // vendor-boundary
        requires(!std::is_const_v<char_type>)
        : string_slice(other.data(), other.size()) {}

    constexpr string_slice_t &operator=(std::string const &other) noexcept // vendor-boundary
        requires std::is_const_v<char_type>
    {
        return assign({other.data(), other.size()});
    }

    constexpr string_slice_t &operator=(std::string &other) noexcept // vendor-boundary
        requires(!std::is_const_v<char_type>)
    {
        return assign({other.data(), other.size()});
    }

    operator std::string() const { return {data(), size()}; } // vendor-boundary

    /**
     *  @brief Formatted output function for compatibility with STL's @c std::basic_ostream.
     *  @throws @c std::ios_base::failure if an exception occurred during output.
     */
    template <typename stream_traits_>
    friend std::basic_ostream<value_type, stream_traits_> &operator<<(
        std::basic_ostream<value_type, stream_traits_> &os, string_slice_t const &str) noexcept(false) {
        return os.write(str.data(), str.size());
    }

    constexpr string_slice(std::string_view const &other) noexcept
        requires std::is_const_v<char_type>
        : string_slice(other.data(), other.size()) {}

    constexpr string_slice_t &operator=(std::string_view const &other) noexcept
        requires std::is_const_v<char_type>
    {
        return assign({other.data(), other.size()});
    }
    operator std::string_view() const noexcept { return {data(), size()}; }

#pragma endregion

#pragma region Iterators and Element Access

    constexpr iterator begin() const noexcept { return iterator(start_); }
    constexpr iterator end() const noexcept { return iterator(start_ + length_); }
    constexpr const_iterator cbegin() const noexcept { return const_iterator(start_); }
    constexpr const_iterator cend() const noexcept { return const_iterator(start_ + length_); }
    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(start_ + length_ - 1); }
    constexpr reverse_iterator rend() const noexcept { return reverse_iterator(start_ - 1); }
    constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(start_ + length_ - 1); }
    constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(start_ - 1); }

    constexpr reference operator[](size_type pos) const noexcept { return start_[pos]; }
    constexpr reference at(size_type pos) const noexcept { return start_[pos]; }
    constexpr reference front() const noexcept { return start_[0]; }
    constexpr reference back() const noexcept { return start_[length_ - 1]; }
    constexpr pointer data() const noexcept { return start_; }

    constexpr difference_type ssize() const noexcept { return static_cast<difference_type>(length_); }
    constexpr size_type size() const noexcept { return length_; }
    constexpr size_type length() const noexcept { return length_; }
    constexpr size_type max_size() const noexcept { return npos - 1; }
    constexpr bool empty() const noexcept { return length_ == 0; }

#pragma endregion

#pragma region Slicing

#pragma region Safe and Signed Extensions

    /** Equivalent to Python's <tt>"abc"[-3:-1]</tt>. Exception-safe, unlike STL's @c substr.
     *  Supports signed and unsigned intervals. */
    constexpr string_slice_t operator[](std::initializer_list<difference_type> signed_offsets) const noexcept {
        sz_assert_(signed_offsets.size() == 2 && "operator[] can't take more than 2 offsets");
        difference_type const *bounds = signed_offsets.begin();
        return sub(bounds[0], bounds[1]);
    }

    /**
     *  @brief Signed alternative to <tt>at()</tt>. Handy if you often write <tt>str[str.size() - 2]</tt>.
     *  @warning The behavior is @b undefined if the position is beyond bounds.
     */
    constexpr reference sat(difference_type signed_offset) const noexcept {
        size_type pos = static_cast<size_type>(signed_offset < 0 ? size() + signed_offset : signed_offset);
        sz_assert_(pos < size() && "string_slice_t::sat(i) out of bounds");
        return start_[pos];
    }

    /**
     *  @brief The slice that would be dropped by @c remove_prefix, that accepts signed arguments
     *      and does no bounds checking. Equivalent to Python's <tt>"abc"[:2]</tt> and <tt>"abc"[:-1]</tt>.
     *
     *  @warning Undefined behavior if @p signed_offset is above size(), below −size(), or −0.
     */
    constexpr string_slice_t front(difference_type signed_offset) const noexcept {
        size_type pos = static_cast<size_type>(signed_offset < 0 ? size() + signed_offset : signed_offset);
        sz_assert_(pos <= size() && "string_slice_t::front(signed_offset) out of bounds");
        return {start_, pos};
    }

    /**
     *  @brief The slice that would be dropped by @c remove_suffix, that accepts signed arguments
     *      and does no bounds checking. Equivalent to Python's <tt>"abc"[2:]</tt> and <tt>"abc"[-1:]</tt>.
     *  @warning Undefined behavior if @p signed_offset is above size(), below −size(), or −0.
     */
    constexpr string_slice_t back(difference_type signed_offset) const noexcept {
        size_type pos = static_cast<size_type>(signed_offset < 0 ? size() + signed_offset : signed_offset);
        sz_assert_(pos <= size() && "string_slice_t::back(signed_offset) out of bounds");
        return {start_ + pos, length_ - pos};
    }

    /** Equivalent to Python's <tt>"abc"[-3:-1]</tt>. Exception-safe, unlike STL's @c substr.
     *  Supports signed and unsigned intervals. */
    constexpr string_slice_t sub(difference_type signed_start_offset,
                                 difference_type signed_end_offset = npos) const noexcept {
        sz_size_t normalized_offset, normalized_length;
        sz_ssize_clamp_interval(length_, signed_start_offset, signed_end_offset, &normalized_offset,
                                &normalized_length);
        return string_slice_t(start_ + normalized_offset, normalized_length);
    }

    /** Exports this entire view. Not an STL function, but useful for concatenations.
     *  The STL variant expects at least two arguments. */
    size_type copy(value_type *destination) const noexcept {
        if (length_) std::memcpy(destination, start_, length_); // ? An empty view may hold a null start
        return length_;
    }

#pragma endregion

#pragma region STL Style

    /**
     *  @brief Removes the first @p n bytes from the view.
     *  @warning The behavior is @b undefined if <tt>n > size()</tt>.
     */
    void remove_prefix(size_type n) noexcept {
        sz_assert_(n <= size());
        start_ += n, length_ -= n;
    }

    /**
     *  @brief Removes the last @p n bytes from the view.
     *  @warning The behavior is @b undefined if <tt>n > size()</tt>.
     */
    void remove_suffix(size_type n) noexcept {
        sz_assert_(n <= size());
        length_ -= n;
    }

    /** Added for STL compatibility. */
    constexpr string_slice_t substr() const noexcept { return *this; }

    /** Return a slice of this view after first @p n bytes, empty when @p n passes the end, like
     *  Python's <tt>text[n:]</tt>. */
    constexpr string_slice_t substr(size_type n) const noexcept {
        n = sz_min_of_two(n, length_);
        return string_slice_t(start_ + n, length_ - n);
    }

    /** Return a slice of this view after first @p n bytes, taking at most @p count bytes, empty
     *  when @p n passes the end, like Python's <tt>text[n:n + count]</tt>. */
    constexpr string_slice_t substr(size_type n, size_type count) const noexcept {
        n = sz_min_of_two(n, length_);
        return string_slice_t(start_ + n, sz_min_of_two(count, length_ - n));
    }

    /** Exports at most @p count bytes of this view after the first @p n, none when @p n passes
     *  the end. @return The number of bytes written. */
    size_type copy(value_type *destination, size_type count, size_type n = 0) const noexcept {
        string_slice_t const exported = substr(n, count);
        if (exported.size()) std::memcpy(destination, exported.data(), exported.size());
        return exported.size();
    }

#pragma endregion

#pragma endregion

#pragma region Comparisons

    /**
     *  @brief Compares two strings lexicographically, by unsigned bytes, then by length.
     *  @return -1 if <tt>*this</tt> is less than @p other, 0 if equal, 1 if greater.
     */
    constexpr int compare(string_view_t other) const noexcept {
        size_type const common = min_of(length_, other.size());
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) // ? a kernel's `__builtin_memcmp` compiles to nothing
        int order = 0;
        for (size_type position = 0; order == 0 && position != common; ++position)
            order = static_cast<unsigned char>(start_[position]) - static_cast<unsigned char>(other[position]);
#else
        int const order = common ? __builtin_memcmp(start_, other.data(), common) : 0;
#endif
        if (order != 0) return order < 0 ? -1 : 1;
        return length_ == other.size() ? 0 : (length_ < other.size() ? -1 : 1);
    }

    /**
     *  @brief Compares up to @p count1 bytes at @p pos1 with @p other, lexicographically,
     *      then by length on a common prefix.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     *  @note A @p pos1 past the end compares an empty slice.
     */
    constexpr int compare(size_type pos1, size_type count1, string_view_t other) const noexcept {
        return substr(pos1, count1).compare(other);
    }

    /**
     *  @brief Compares up to @p count1 bytes at @p pos1 with up to @p count2 bytes at @p pos2
     *      of @p other, lexicographically, then by length on a common prefix.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     *  @note A position past the end compares an empty slice.
     */
    constexpr int compare(size_type pos1, size_type count1, string_view_t other, size_type pos2,
                          size_type count2) const noexcept {
        return substr(pos1, count1).compare(other.substr(pos2, count2));
    }

    /**
     *  @brief Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     */
    constexpr int compare(const_pointer other) const noexcept { return compare(string_view_t(other)); }

    /**
     *  @brief Compares up to @p count1 bytes at @p pos1 with @p other, lexicographically,
     *      then by length on a common prefix.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     *  @note A @p pos1 past the end compares an empty slice.
     */
    constexpr int compare(size_type pos1, size_type count1, const_pointer other) const noexcept {
        return substr(pos1, count1).compare(string_view_t(other));
    }

    /**
     *  @brief Compares up to @p count1 bytes at @p pos1 with the first @p count2 bytes of
     *      @p other, lexicographically, then by length on a common prefix.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     *  @note A @p pos1 past the end compares an empty slice.
     */
    constexpr int compare(size_type pos1, size_type count1, const_pointer other, size_type count2) const noexcept {
        return substr(pos1, count1).compare(string_view_t(other, count2));
    }

    /** Checks if the string is equal to the other string. */
    constexpr bool operator==(string_view_t other) const noexcept {
        return length_ == other.size() && compare(other) == 0;
    }

    /** Checks if the string is equal to a concatenation of two strings. */
    bool operator==(concatenation<string_view_t, string_view_t> const &other) const noexcept {
        return length_ == other.size() && substr(0, other.first.size()) == other.first &&
               substr(other.first.size()) == other.second;
    }

    /** Computes the lexicographic ordering between this and the ::other string. */
    constexpr std::strong_ordering operator<=>(string_view_t other) const noexcept { return compare(other) <=> 0; }

#pragma endregion

    /**
     *  @brief Lazily concatenates two slices into an expression template, without allocating.
     *  @sa concatenate for the variadic form, and the converting constructor of @c string
     *      that materializes the expression.
     */
    concatenation<string_view_t, string_view_t> operator|(string_view_t other) const noexcept {
        return {string_view_t(data(), size()), other};
    }

    /** Populate a character set with characters present in this string. */
    byteset_t as_set() const noexcept {
        byteset_t set;
        for (auto it_ = this->begin(); it_ != this->end(); ++it_) set.add(*it_);
        return set;
    }

#pragma region Kernel Backed Operations

    string_slice_t remove_prefix(string_view_t) const = delete; // sz::remove_prefix(text, prefix)
    string_slice_t remove_suffix(string_view_t) const = delete; // sz::remove_suffix(text, suffix)
    template <typename... arguments_types_>
    void starts_with(arguments_types_ &&...) const = delete; // sz::starts_with(text, prefix)
    template <typename... arguments_types_>
    void ends_with(arguments_types_ &&...) const = delete; // sz::ends_with(text, suffix)
    template <typename... arguments_types_>
    void contains(arguments_types_ &&...) const = delete; // sz::contains(text, needle)
    template <typename... arguments_types_>
    void count(arguments_types_ &&...) const = delete; // sz::count(text, needle)
    template <typename... arguments_types_>
    void find(arguments_types_ &&...) const = delete; // sz::find(text, needle)
    template <typename... arguments_types_>
    void rfind(arguments_types_ &&...) const = delete; // sz::rfind(text, needle)
    template <typename... arguments_types_>
    void partition(arguments_types_ &&...) const = delete; // sz::partition(text, pattern)
    template <typename... arguments_types_>
    void rpartition(arguments_types_ &&...) const = delete; // sz::rpartition(text, pattern)
    template <typename... arguments_types_>
    void find_first_of(arguments_types_ &&...) const = delete; // sz::find_first_of(text, set)
    template <typename... arguments_types_>
    void find_first_not_of(arguments_types_ &&...) const = delete; // sz::find_first_not_of(text, set)
    template <typename... arguments_types_>
    void find_last_of(arguments_types_ &&...) const = delete; // sz::find_last_of(text, set)
    template <typename... arguments_types_>
    void find_last_not_of(arguments_types_ &&...) const = delete; // sz::find_last_not_of(text, set)
    template <typename... arguments_types_>
    void contains_only(arguments_types_ &&...) const = delete; // sz::contains_only(text, set)
    template <typename... arguments_types_>
    void is_alpha(arguments_types_ &&...) const = delete; // sz::is_alpha(text)
    template <typename... arguments_types_>
    void is_alnum(arguments_types_ &&...) const = delete; // sz::is_alnum(text)
    template <typename... arguments_types_>
    void is_ascii(arguments_types_ &&...) const = delete; // sz::is_ascii(text)
    template <typename... arguments_types_>
    void is_digit(arguments_types_ &&...) const = delete; // sz::is_digit(text)
    template <typename... arguments_types_>
    void is_lower(arguments_types_ &&...) const = delete; // sz::is_lower(text)
    template <typename... arguments_types_>
    void is_space(arguments_types_ &&...) const = delete; // sz::is_space(text)
    template <typename... arguments_types_>
    void is_upper(arguments_types_ &&...) const = delete; // sz::is_upper(text)
    template <typename... arguments_types_>
    void is_printable(arguments_types_ &&...) const = delete; // sz::is_printable(text)
    template <typename... arguments_types_>
    void lstrip(arguments_types_ &&...) const = delete; // sz::lstrip(text, set)
    template <typename... arguments_types_>
    void rstrip(arguments_types_ &&...) const = delete; // sz::rstrip(text, set)
    template <typename... arguments_types_>
    void strip(arguments_types_ &&...) const = delete; // sz::strip(text, set)
    template <typename... arguments_types_>
    void find_all(arguments_types_ &&...) const = delete; // sz::find_all(text, needle)
    template <typename... arguments_types_>
    void rfind_all(arguments_types_ &&...) const = delete; // sz::rfind_all(text, needle)
    template <typename... arguments_types_>
    void split(arguments_types_ &&...) const = delete; // sz::split(text, delimiter)
    template <typename... arguments_types_>
    void rsplit(arguments_types_ &&...) const = delete; // sz::rsplit(text, delimiter)
    template <typename... arguments_types_>
    void splitlines(arguments_types_ &&...) const = delete; // sz::split_characters(text, newlines_set())
    template <typename... arguments_types_>
    void hash(arguments_types_ &&...) const = delete; // sz::hash(text)
    template <typename... arguments_types_>
    void hash_multiseed(arguments_types_ &&...) const = delete; // sz::hash_multiseed(text, seeds, hashes)
    template <typename... arguments_types_>
    void bytesum(arguments_types_ &&...) const = delete; // sz::bytesum(text)
    template <typename... arguments_types_>
    void is_normalized(arguments_types_ &&...) const = delete; // sz::is_normalized(text, form)
    template <typename... arguments_types_>
    void utf8_find_denormalized(arguments_types_ &&...) const = delete; // sz::utf8_find_denormalized(text, form)
    template <typename... arguments_types_>
    void utf8_count(arguments_types_ &&...) const = delete; // sz::utf8_count(text)
    template <typename... arguments_types_>
    void utf8_seek(arguments_types_ &&...) const = delete; // sz::utf8_seek(text, runes)
    template <typename... arguments_types_>
    void utf8_runes(arguments_types_ &&...) const = delete; // sz::utf8_runes(text)
    template <typename... arguments_types_>
    void utf8_split_newlines(arguments_types_ &&...) const = delete; // sz::utf8_split_newlines(text)
    template <typename... arguments_types_>
    void utf8_newlines(arguments_types_ &&...) const = delete; // sz::utf8_newlines(text)
    template <typename... arguments_types_>
    void utf8_split_whitespaces(arguments_types_ &&...) const = delete; // sz::utf8_split_whitespaces(text)
    template <typename... arguments_types_>
    void utf8_whitespaces(arguments_types_ &&...) const = delete; // sz::utf8_whitespaces(text)
    template <typename... arguments_types_>
    void utf8_split_delimiters(arguments_types_ &&...) const = delete; // sz::utf8_split_delimiters(text)
    template <typename... arguments_types_>
    void utf8_delimiters(arguments_types_ &&...) const = delete; // sz::utf8_delimiters(text)
    template <typename... arguments_types_>
    void utf8_wordbreaks(arguments_types_ &&...) const = delete; // sz::utf8_wordbreaks(text)
    template <typename... arguments_types_>
    void utf8_graphemes(arguments_types_ &&...) const = delete; // sz::utf8_graphemes(text)
    template <typename... arguments_types_>
    void utf8_sentences(arguments_types_ &&...) const = delete; // sz::utf8_sentences(text)
    template <typename... arguments_types_>
    void utf8_linebreaks(arguments_types_ &&...) const = delete; // sz::utf8_linebreaks(text)
    template <typename... arguments_types_>
    void utf8_uncased_order(arguments_types_ &&...) const = delete; // sz::utf8_uncased_order(text, other)
    template <typename... arguments_types_>
    void utf8_uncased_search(arguments_types_ &&...) const = delete; // sz::utf8_uncased_search(text, needle)
    template <typename... arguments_types_>
    void utf8_uncased_matches(arguments_types_ &&...) const = delete; // sz::utf8_uncased_matches(text, needle)
#pragma endregion

  private:
    constexpr string_slice_t &assign(string_view_t const &other) noexcept {
        start_ = (pointer)other.data();
        length_ = other.size();
        return *this;
    }

    constexpr static size_type null_terminated_length(const_pointer s) noexcept {
        const_pointer p = s;
        while (*p) ++p;
        return p - s;
    }
};

#pragma endregion

/**
 *  @brief Memory-owning string class with a Small String Optimization.
 *
 *  @section sz_cpp_api API
 *
 *  Some APIs are different from @c string_slice:
 *
 *  - @c lstrip, @c rstrip, @c strip modify the string in-place, instead of returning a new view.
 *  - @c sat, @c sub, and element access have non-const overloads returning mutable references.
 *
 *  Functions defined for @c string, but not present in @c string_slice:
 *
 *  - @c replace, @c insert, @c erase, @c append, @c push_back, @c pop_back, @c resize,
 *  - @c insert_at, @c replace_slice and @c erase_slice twins over signed offsets,
 *  - @c replace_all and @c erase_all similar to Boost,
 *  - @c translate for character mapping,
 *  - @c fill_random for fast random string generation.
 *
 *  Functions defined for @c string_slice, but not present in @c string:
 *
 *  - @c partition, @c rpartition, @c split, @c rsplit, @c find_all and @c rfind_all, left out to
 *    enforce lifetimes over long operations,
 *  - @c remove_prefix and @c remove_suffix.
 *
 *  @section sz_cpp_errors Errors
 *
 *  Nothing throws. The default constructor is @c constexpr, and moves are @c noexcept. Copies
 *  and conversions go through @c copy, @c make and @c assign, and every allocating verb returns
 *  a @c status_t, or an @c expected beside an iterator. Positions past the end clamp to it, like
 *  Python's slices, and @c at checks its position with a debug assertion only.
 */
template <typename allocator_type_ = std::allocator<char>>
class string {

    using char_type = char;
    using sz_alloc_type = sz_allocator_t;
    using allocator_traits_t = std::allocator_traits<allocator_type_>;

    sz_string_t string_;
    sz_no_unique_address_ allocator_type_ allocator_;

    template <typename allocator_callback_>
    status_t with_allocator_(allocator_callback_ &&callback) noexcept {
        return ashvardanian::stringzilla::with_allocator_(allocator_, callback);
    }

    /** Returns the heap block, if any, to the allocator that granted it, leaving an empty SSO
     *  string. */
    void release() noexcept {
        [[maybe_unused]] status_t const status = with_allocator_([&](sz_alloc_type &allocator) {
            sz_string_free(&string_, &allocator);
            return sz_success_k;
        });
    }

    void move(string &other) noexcept {
        // Copying the other state as is fails, as its start may point into its own stack buffer.
        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_size_t string_space;
        sz_bool_t string_is_external;
        sz_string_unpack(&other.string_, &string_start, &string_length, &string_space, &string_is_external);

        // Acquire the old string's value bitwise
        *(&string_) = *(&other.string_);
        // Reposition the string start pointer to the stack if it fits.
        // Ternary condition may be optimized to a branchless version.
        string_.internal.start = string_is_external ? string_.internal.start : &string_.internal.chars[0];
        sz_string_init(&other.string_); // Discard the other string.
    }

  public:
    /*  STL compatibility */
    using traits_type = std::char_traits<char_type>;
    using value_type = char_type;
    using pointer = char_type *;
    using const_pointer = char_type const *;
    using reference = char_type &;
    using const_reference = char_type const &;
    using const_iterator = const_pointer;
    using iterator = pointer;
    using const_reverse_iterator = reversed_iterator_for<char_type const>;
    using reverse_iterator = reversed_iterator_for<char_type>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    /*  Non-STL type definitions */
    using allocator_type = allocator_type_;
    using string_span_t = string_slice<char_type>;
    using string_view_t = string_slice<typename std::add_const<char_type>::type>;

    /**
     *  @brief Special value for missing matches.
     *
     *  We take the largest 63-bit unsigned integer on 64-bit machines.
     *  We take the largest 31-bit unsigned integer on 32-bit machines.
     */
    static constexpr size_type npos = STRINGZILLA_SSIZE_MAX;

    /** The number of characters that can be stored in the internal buffer.
     *  Depends on the size of the internal buffer for the "Small String Optimization". */
    static constexpr size_type min_capacity = STRINGZILLA_STRING_INTERNAL_SPACE - 1;

#pragma region Constructors and STL Utilities

    constexpr string() noexcept : string(allocator_type_()) {}

    constexpr explicit string(allocator_type_ const &allocator) noexcept : allocator_(allocator) {
        // ! Reimplements `sz_string_init`, so that this constructor can stay `constexpr`.
        string_.internal.start = &string_.internal.chars[0];
        string_.words[1] = 0;
        string_.words[2] = 0;
        string_.words[3] = 0;
    }

    ~string() noexcept { release(); }

    string(string &&other) noexcept : allocator_(std::move(other.allocator_)) { move(other); }

    /** Steals the buffer, so the allocators must propagate on move or always compare equal; any
     *  other pair copies through @ref assign, which reports a failed allocation. */
    string &operator=(string &&other) noexcept
        requires(allocator_traits_t::propagate_on_container_move_assignment::value ||
                 allocator_traits_t::is_always_equal::value)
    {
        if (this == &other) return *this;
        release();
        if constexpr (allocator_traits_t::propagate_on_container_move_assignment::value)
            allocator_ = std::move(other.allocator_);
        move(other);
        return *this;
    }

    /** Copies go through @ref copy and @ref assign, which report a failed allocation. */
    string(string const &) = delete;
    string &operator=(string const &) = delete;
    string(std::nullptr_t) = delete;

    /** A copy of this string, in memory of the allocator that
     *  @c select_on_container_copy_construction picks. */
    expected<string> copy() const noexcept {
        string copied(allocator_traits_t::select_on_container_copy_construction(allocator_));
        status_t const status = copied.assign(view());
        return {std::move(copied), status};
    }

    /** A new string holding @p content, in memory of @p allocator. */
    static expected<string> make(string_view_t content, allocator_type_ const &allocator = allocator_type_()) noexcept {
        string made(allocator);
        status_t const status = made.assign(content);
        return {std::move(made), status};
    }

    /** A new string holding a lazy concatenation, like <tt>name | "\@" | domain</tt>, in one allocation. */
    template <typename first_type_, typename second_type_>
    static expected<string> make(concatenation<first_type_, second_type_> const &expression,
                                 allocator_type_ const &allocator = allocator_type_()) noexcept {
        string made(allocator);
        status_t const status = made.assign(expression);
        return {std::move(made), status};
    }

    operator string_view_t() const noexcept sz_lifetime_bound_ { return view(); }
    string_view_t view() const noexcept sz_lifetime_bound_ {
        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_string_range(&string_, &string_start, &string_length);
        return {string_start, string_length};
    }

    operator string_span_t() noexcept sz_lifetime_bound_ { return span(); }
    string_span_t span() noexcept sz_lifetime_bound_ {
        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_string_range(&string_, &string_start, &string_length);
        return {string_start, string_length};
    }

    /** Exchanges the string contents witt the @p other string. */
    void swap(string &other) noexcept {
        // A plain `swap(string_, other.string_)` breaks when either string is on the stack, as it
        // swaps the pointer to the stack-allocated memory instead of the contents.
        sz_ptr_t first_start, second_start;
        sz_size_t first_length, second_length;
        sz_size_t first_space, second_space;
        sz_bool_t first_is_external, second_is_external;
        sz_string_unpack(&string_, &first_start, &first_length, &first_space, &first_is_external);
        sz_string_unpack(&other.string_, &second_start, &second_length, &second_space, &second_is_external);
        trivial_swap(string_, other.string_);
        if (!first_is_external) other.string_.internal.start = &other.string_.internal.chars[0];
        if (!second_is_external) string_.internal.start = &string_.internal.chars[0];
        if constexpr (allocator_traits_t::propagate_on_container_swap::value)
            trivial_swap(allocator_, other.allocator_);
    }

    // As we are need both `data()` and `size()`, going through `operator string_view_t()`
    // and `sz_string_unpack` is faster than separate invocations.
    operator std::string() const { return view(); } // vendor-boundary

    /**
     *  @brief Formatted output function for compatibility with STL's @c std::basic_ostream.
     *  @throws @c std::ios_base::failure if an exception occurred during output.
     */
    template <typename stream_traits_>
    friend std::basic_ostream<value_type, stream_traits_> &operator<<(
        std::basic_ostream<value_type, stream_traits_> &os, string const &str) noexcept(false) {
        return os.write(str.data(), str.size());
    }

    operator std::string_view() const noexcept { return view(); }

#pragma endregion

#pragma region Iterators and Accessors

    iterator begin() noexcept sz_lifetime_bound_ { return iterator(data()); }
    const_iterator begin() const noexcept sz_lifetime_bound_ { return const_iterator(data()); }
    const_iterator cbegin() const noexcept sz_lifetime_bound_ { return const_iterator(data()); }

    // As we are need both `data()` and `size()`, going through `operator string_view_t()`
    // and `sz_string_unpack` is faster than separate invocations.
    iterator end() noexcept sz_lifetime_bound_ { return span().end(); }
    const_iterator end() const noexcept sz_lifetime_bound_ { return view().end(); }
    const_iterator cend() const noexcept sz_lifetime_bound_ { return view().end(); }

    reverse_iterator rbegin() noexcept sz_lifetime_bound_ { return span().rbegin(); }
    const_reverse_iterator rbegin() const noexcept sz_lifetime_bound_ { return view().rbegin(); }
    const_reverse_iterator crbegin() const noexcept sz_lifetime_bound_ { return view().crbegin(); }

    reverse_iterator rend() noexcept sz_lifetime_bound_ { return span().rend(); }
    const_reverse_iterator rend() const noexcept sz_lifetime_bound_ { return view().rend(); }
    const_reverse_iterator crend() const noexcept sz_lifetime_bound_ { return view().crend(); }

    reference operator[](size_type pos) noexcept sz_lifetime_bound_ { return string_.internal.start[pos]; }
    const_reference operator[](size_type pos) const noexcept sz_lifetime_bound_ { return string_.internal.start[pos]; }

    reference front() noexcept sz_lifetime_bound_ { return string_.internal.start[0]; }
    const_reference front() const noexcept sz_lifetime_bound_ { return string_.internal.start[0]; }
    reference back() noexcept sz_lifetime_bound_ { return string_.internal.start[size() - 1]; }
    const_reference back() const noexcept sz_lifetime_bound_ { return string_.internal.start[size() - 1]; }
    pointer data() noexcept sz_lifetime_bound_ { return string_.internal.start; }
    const_pointer data() const noexcept sz_lifetime_bound_ { return string_.internal.start; }
    pointer c_str() noexcept sz_lifetime_bound_ { return string_.internal.start; }
    const_pointer c_str() const noexcept sz_lifetime_bound_ { return string_.internal.start; }

    /** Element access checked by a debug assertion, as @c operator[] is unchecked. */
    reference at(size_type pos) noexcept sz_lifetime_bound_ {
        sz_assert_(pos < size());
        return string_.internal.start[pos];
    }
    const_reference at(size_type pos) const noexcept sz_lifetime_bound_ {
        sz_assert_(pos < size());
        return string_.internal.start[pos];
    }

    difference_type ssize() const noexcept { return static_cast<difference_type>(size()); }
    size_type size() const noexcept { return sz_string_length(&string_); }
    size_type length() const noexcept { return size(); }
    size_type max_size() const noexcept { return npos - 1; }
    bool empty() const noexcept { return sz_string_length(&string_) == 0; }
    size_type capacity() const noexcept {
        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_size_t string_space;
        sz_bool_t string_is_external;
        sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_external);
        return string_space - 1;
    }

    allocator_type get_allocator() const noexcept { return allocator_; }

#pragma endregion

#pragma region Slicing

#pragma region Safe and Signed Extensions

    /** Equivalent to Python's <tt>"abc"[-3:-1]</tt>. Exception-safe, unlike STL's @c substr.
     *  Supports signed and unsigned intervals. */
    string_view_t operator[](std::initializer_list<difference_type> offsets) const noexcept sz_lifetime_bound_ {
        string_view_t const whole = view();
        return whole[offsets];
    }
    string_span_t operator[](std::initializer_list<difference_type> offsets) noexcept sz_lifetime_bound_ {
        string_span_t const whole = span();
        return whole[offsets];
    }

    /**
     *  @brief Signed alternative to <tt>at()</tt>. Handy if you often write <tt>str[str.size() - 2]</tt>.
     *  @warning The behavior is @b undefined if the position is beyond bounds.
     */
    value_type sat(difference_type offset) const noexcept sz_lifetime_bound_ { return view().sat(offset); }
    reference sat(difference_type offset) noexcept sz_lifetime_bound_ { return span().sat(offset); }

    /**
     *  @brief The opposite operation to @c remove_prefix, that does no bounds checking.
     *  @warning The behavior is @b undefined if <tt>n > size()</tt>.
     */
    string_view_t front(difference_type n) const noexcept sz_lifetime_bound_ { return view().front(n); }
    string_span_t front(difference_type n) noexcept sz_lifetime_bound_ { return span().front(n); }

    /**
     *  @brief The opposite operation to @c remove_prefix, that does no bounds checking.
     *  @warning The behavior is @b undefined if <tt>n > size()</tt>.
     */
    string_view_t back(difference_type n) const noexcept sz_lifetime_bound_ { return view().back(n); }
    string_span_t back(difference_type n) noexcept sz_lifetime_bound_ { return span().back(n); }

    /** Equivalent to Python's <tt>"abc"[-3:-1]</tt>. Exception-safe, unlike STL's @c substr.
     *  Supports signed and unsigned intervals. @b Doesn't copy or allocate memory! */
    string_view_t sub(difference_type start, difference_type end = npos) const noexcept sz_lifetime_bound_ {
        return view().sub(start, end);
    }
    string_span_t sub(difference_type start, difference_type end = npos) noexcept sz_lifetime_bound_ {
        return span().sub(start, end);
    }

    /** Exports this entire view. Not an STL function, but useful for concatenations.
     *  The STL variant expects at least two arguments. */
    size_type copy(value_type *destination) const noexcept { return view().copy(destination); }

#pragma endregion

#pragma region STL Style

    /**
     *  @brief Removes the first @p n characters from the view.
     *  @warning The behavior is @b undefined if <tt>n > size()</tt>.
     */
    void remove_prefix(size_type n) noexcept {
        sz_assert_(n <= size());
        sz_string_erase(&string_, 0, n);
    }

    /**
     *  @brief Removes the last @p n characters from the view.
     *  @warning The behavior is @b undefined if <tt>n > size()</tt>.
     */
    void remove_suffix(size_type n) noexcept {
        sz_assert_(n <= size());
        sz_string_erase(&string_, size() - n, n);
    }

    /**
     *  @brief Exports at most @p count bytes after the first @p skip, and none at all when
     *      @p skip passes the end of the string.
     *  @return The number of bytes written.
     */
    size_type copy(value_type *destination, size_type count, size_type skip = 0) const noexcept {
        return view().copy(destination, count, skip);
    }

#pragma endregion

#pragma endregion

#pragma region Comparisons

#pragma region Whole String Comparisons

    /**
     *  @brief Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     */
    int compare(string_view_t other) const noexcept { return view().compare(other); }

    /**
     *  @brief Compares up to @p count1 bytes at @p pos1 with @p other, lexicographically,
     *      then by length on a common prefix.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     *  @note A @p pos1 past the end compares an empty slice.
     */
    int compare(size_type pos1, size_type count1, string_view_t other) const noexcept {
        return view().compare(pos1, count1, other);
    }

    /**
     *  @brief Compares up to @p count1 bytes at @p pos1 with up to @p count2 bytes at @p pos2
     *      of @p other, lexicographically, then by length on a common prefix.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     *  @note A position past the end compares an empty slice.
     */
    int compare(size_type pos1, size_type count1, string_view_t other, size_type pos2,
                size_type count2) const noexcept {
        return view().compare(pos1, count1, other, pos2, count2);
    }

    /**
     *  @brief Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     */
    int compare(const_pointer other) const noexcept { return view().compare(other); }

    /**
     *  @brief Compares up to @p count1 bytes at @p pos1 with @p other, lexicographically,
     *      then by length on a common prefix.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     *  @note A @p pos1 past the end compares an empty slice.
     */
    int compare(size_type pos1, size_type count1, const_pointer other) const noexcept {
        return view().compare(pos1, count1, other);
    }

    /**
     *  @brief Compares up to @p count1 bytes at @p pos1 with the first @p count2 bytes of
     *      @p other, lexicographically, then by length on a common prefix.
     *  @return 0 if equal, negative if <tt>*this</tt> is less than @p other, positive if <tt>*this</tt> is
     *      greater than @p other.
     *  @note A @p pos1 past the end compares an empty slice.
     */
    int compare(size_type pos1, size_type count1, const_pointer other, size_type count2) const noexcept {
        return view().compare(pos1, count1, other, count2);
    }

    /** Checks if the string is equal to the other string. */
    bool operator==(string const &other) const noexcept { return view() == other.view(); }
    bool operator==(string_view_t other) const noexcept { return view() == other; }
    bool operator==(const_pointer other) const noexcept { return view() == string_view_t(other); }

    /** Computes the lexicographic ordering between this and the @p other string. */
    std::strong_ordering operator<=>(string const &other) const noexcept { return view() <=> other.view(); }
    std::strong_ordering operator<=>(string_view_t other) const noexcept { return view() <=> other; }
    std::strong_ordering operator<=>(const_pointer other) const noexcept { return view() <=> string_view_t(other); }

#pragma endregion
    bool is_internal() const noexcept { return sz_string_is_on_stack(&string_); }

#pragma region Modifiers
#pragma region Non STL API

    /**
     *  @brief Resizes the string to a specified number of characters, padding with the specified
     *      character if needed.
     *  @param[in] count The new size of the string.
     *  @param[in] character The character to fill new elements with, if expanding, by default
     *      the null character.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k A @p count past @c max_size.
     */
    status_t resize(size_type count, value_type character = '\0') noexcept;

    /** Reduces memory usage by freeing unused memory. */
    status_t shrink_to_fit() noexcept {
        return with_allocator_([&](sz_alloc_type &allocator) {
            return sz_string_shrink_to_fit(&string_, &allocator) ? sz_success_k : sz_bad_alloc_k;
        });
    }

    /**
     *  @brief Reserves enough space for a specified number of characters.
     *  @param[in] capacity The new capacity to reserve.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k A @p capacity past @c max_size.
     */
    status_t reserve(size_type capacity) noexcept {
        if (capacity > max_size()) return status_t::overflow_risk_k;
        return with_allocator_([&](sz_alloc_type &allocator) {
            return sz_string_reserve(&string_, capacity, &allocator) ? sz_success_k : sz_bad_alloc_k;
        });
    }

    /**
     *  @brief Assigns a new value to the string, replacing its current contents.
     *  @param[in] other The string view whose contents to assign.
     */
    status_t assign(string_view_t other) noexcept;

    /**
     *  @brief Assigns a concatenated sequence to the string, replacing its current contents.
     *  @param[in] other The concatenation object representing the sequence to assign.
     */
    template <typename first_type_, typename second_type_>
    status_t assign(concatenation<first_type_, second_type_> const &other) noexcept;

    /**
     *  @brief Adds a single character to the end of the string.
     *  @param[in] c The character to add.
     */
    status_t push_back(char_type c) noexcept;

    /**
     *  @brief Appends a given character array to the string.
     *  @param[in] str The pointer to the array of characters to append.
     *  @param[in] length The number of characters to append.
     */
    status_t append(const_pointer str, size_type length) noexcept;

    /**
     *  @brief Appends a string view to the string.
     *  @param[in] str The string view to append.
     */
    status_t append(string_view_t str) noexcept { return append(str.data(), str.size()); }

    /**
     *  @brief Erases @b (in-place) the characters between two signed offsets, counted from the
     *      end when negative, like Python's <tt>del text[start:end]</tt>.
     *  @return Number of characters removed.
     */
    size_type erase_slice(difference_type signed_start_offset = 0, difference_type signed_end_offset = npos) noexcept {
        sz_size_t normalized_offset, normalized_length;
        sz_ssize_clamp_interval(size(), signed_start_offset, signed_end_offset, &normalized_offset, &normalized_length);
        if (normalized_length) sz_string_erase(&string_, normalized_offset, normalized_length);
        return normalized_length;
    }

    /**
     *  @brief Inserts @b (in-place) @p string at a signed offset, counted from the end when
     *      negative, like Python's <tt>text[offset:offset] = string</tt>.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k Past @c max_size.
     */
    status_t insert_at(difference_type signed_offset, string_view_t string) noexcept {
        sz_size_t normalized_offset, normalized_length;
        sz_ssize_clamp_interval(size(), signed_offset, signed_offset, &normalized_offset, &normalized_length);
        return insert(normalized_offset, string);
    }

    /**
     *  @brief Replaces @b (in-place) the characters between two signed offsets, counted from the
     *      end when negative, like Python's <tt>text[start:end] = replacement</tt>.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k Past @c max_size.
     */
    status_t replace_slice(difference_type signed_start_offset, difference_type signed_end_offset,
                           string_view_t replacement) noexcept {
        sz_size_t normalized_offset, normalized_length;
        sz_ssize_clamp_interval(size(), signed_start_offset, signed_end_offset, &normalized_offset, &normalized_length);
        return replace(normalized_offset, normalized_length, replacement);
    }

    /**
     *  @brief Resizes the string to a specified number of characters without initializing new
     *      elements, calling the provided callback to overwrite the resized contents.
     *  @param[in] count The new size of the string.
     *  @param[in] operation A callback that receives a pointer and the new size, and returns the
     *      actual new size.
     *  @see https://en.cppreference.com/w/cpp/string/string/resize_and_overwrite
     *
     *  @par Statuses:
     *  - @c overflow_risk_k A @p count past @c max_size.
     */
    template <typename operation_type_>
    status_t resize_and_overwrite(size_type count, operation_type_ operation) noexcept {
        if (count > max_size()) return status_t::overflow_risk_k;

        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_size_t string_space;
        sz_bool_t string_is_external;
        sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_external);

        // Allocate more space if needed, without initializing
        if (count >= string_space) {
            status_t const status = with_allocator_([&](sz_alloc_type &allocator) {
                return sz_string_expand(&string_, STRINGZILLA_SIZE_MAX, count - string_length, &allocator)
                           ? sz_success_k
                           : sz_bad_alloc_k;
            });
            if (failed(status)) return status;
            sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_external);
        }

        // The user's operation fills the buffer through a mutable pointer and the requested count.
        size_type actual_count = operation(reinterpret_cast<char_type *>(string_start), count);

        // Clamp the actual count to the requested count for safety
        if (actual_count > count) actual_count = count;

        // Update the string length appropriately
        if (actual_count > string_length) {
            string_start[actual_count] = '\0';
            // Safe for both SSO and heap strings: on little-endian, internal.length
            // overlaps with LSB of external.length, so += affects only the length byte.
            // On big-endian, the struct layout places them at matching positions.
            string_.external.length += actual_count - string_length;
        }
        else { sz_string_erase(&string_, actual_count, STRINGZILLA_SIZE_MAX); }

        return status_t::success_k;
    }
#pragma endregion

#pragma region STL Interfaces

    /** Clears the string contents, but @b no deallocations happen. */
    void clear() noexcept { sz_string_erase(&string_, 0, STRINGZILLA_SIZE_MAX); }

    /**
     *  @brief Erases ( @b in-place ) the given range of characters.
     *  @return Iterator following the erased character, or <tt>end()</tt> if no such character exists.
     */
    iterator erase(const_iterator first, const_iterator last) noexcept sz_lifetime_bound_ {
        auto start = begin();
        auto offset = first - start;
        sz_string_erase(&string_, offset, last - first);
        return start + offset;
    }

    /**
     *  @brief Erases @b (in-place) the one character at a given position.
     *  @return Iterator following the erased character, or <tt>end()</tt> if no such character exists.
     */
    iterator erase(const_iterator pos) noexcept sz_lifetime_bound_ { return erase(pos, pos + 1); }

    /**
     *  @brief Inserts @b (in-place) @p character @p repeats times at @p offset, clamped to the end.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k Past @c max_size.
     */
    status_t insert(size_type offset, size_type repeats, char_type character) noexcept {
        offset = sz_min_of_two(offset, size());
        status_t const status = preparing_replacement(offset, 0, repeats);
        if (succeeded(status)) std::memset(data() + offset, character, repeats);
        return status;
    }

    /**
     *  @brief Inserts @b (in-place) @p other at @p offset, clamped to the end.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k Past @c max_size.
     */
    status_t insert(size_type offset, string_view_t other) noexcept {
        offset = sz_min_of_two(offset, size());
        status_t const status = preparing_replacement(offset, 0, other.size());
        if (succeeded(status) && other.size()) std::memcpy(data() + offset, other.data(), other.size());
        return status;
    }

    /** Inserts @b (in-place) @p length characters from @p start at @p offset, clamped to the
     *  end. */
    status_t insert(size_type offset, const_pointer start, size_type length) noexcept {
        return insert(offset, string_view_t(start, length));
    }

    /** Inserts @b (in-place) at most @p count characters of @p other after its first @p other_index
     *  at @p offset, clamping both positions to their ends. */
    status_t insert(size_type offset, string_view_t other, size_type other_index, size_type count = npos) noexcept {
        return insert(offset, other.substr(other_index, count));
    }

    /**
     *  @brief Inserts @b (in-place) @p character before @p it.
     *  @return The iterator to the inserted character, with the status of the allocation.
     */
    expected<iterator> insert(const_iterator it, char_type character) noexcept sz_lifetime_bound_ {
        return insert(it, size_type(1), character);
    }

    /**
     *  @brief Inserts @b (in-place) @p character @p repeats times before @p it.
     *  @return The iterator to the first inserted character, with the status of the allocation.
     */
    expected<iterator> insert(const_iterator it, size_type repeats, char_type character) noexcept sz_lifetime_bound_ {
        size_type const offset = range_length(cbegin(), it);
        status_t const status = insert(offset, repeats, character);
        return {begin() + offset, status};
    }

    /**
     *  @brief Inserts @b (in-place) the characters between @p first and @p last before @p it.
     *  @return The iterator to the first inserted character, with the status of the allocation.
     */
    template <typename input_iterator_type_>
    expected<iterator> insert(const_iterator it, input_iterator_type_ first,
                              input_iterator_type_ last) noexcept sz_lifetime_bound_ {
        size_type const offset = range_length(cbegin(), it);
        status_t const status = preparing_replacement(offset, 0, range_length(first, last));
        iterator const inserted = begin() + offset;
        for (iterator output = inserted; succeeded(status) && first != last; ++first, ++output) *output = *first;
        return {inserted, status};
    }

    /**
     *  @brief Inserts @b (in-place) the characters of @p list before @p it.
     *  @return The iterator to the first inserted character, with the status of the allocation.
     */
    expected<iterator> insert(const_iterator it, std::initializer_list<char_type> list) noexcept sz_lifetime_bound_ {
        return insert(it, list.begin(), list.end());
    }

    /** Erases @b (in-place) at most @p count characters from @p pos, none when @p pos passes the
     *  end. */
    string &erase(size_type pos = 0, size_type count = npos) noexcept {
        if (pos < size()) sz_string_erase(&string_, pos, count);
        return *this;
    }

    /**
     *  @brief Replaces @b (in-place) at most @p count characters from @p pos with @p str, clamping
     *      @p pos to the end, where the replacement appends.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k Past @c max_size.
     */
    status_t replace(size_type pos, size_type count, string_view_t const &str) noexcept {
        pos = sz_min_of_two(pos, size()), count = sz_min_of_two(count, size() - pos);
        status_t const status = preparing_replacement(pos, count, str.size());
        if (succeeded(status) && str.size()) std::memcpy(data() + pos, str.data(), str.size());
        return status;
    }

    /** Replaces @b (in-place) the characters between @p first and @p last with @p str. */
    status_t replace(const_iterator first, const_iterator last, string_view_t const &str) noexcept {
        return replace(range_length(cbegin(), first), range_length(first, last), str);
    }

    /** Replaces @b (in-place) at most @p count characters from @p pos with at most @p count2
     *  characters of @p str after its first @p pos2, clamping every position to its end. */
    status_t replace(size_type pos, size_type count, string_view_t const &str, size_type pos2,
                     size_type count2 = npos) noexcept {
        return replace(pos, count, str.substr(pos2, count2));
    }

    /** Replaces @b (in-place) at most @p count characters from @p pos with @p count2 characters
     *  from @p cstr. */
    status_t replace(size_type pos, size_type count, const_pointer cstr, size_type count2) noexcept {
        return replace(pos, count, string_view_t(cstr, count2));
    }

    /** Replaces @b (in-place) the characters between @p first and @p last with @p count2
     *  characters from @p cstr. */
    status_t replace(const_iterator first, const_iterator last, const_pointer cstr, size_type count2) noexcept {
        return replace(first, last, string_view_t(cstr, count2));
    }

    /** Replaces @b (in-place) at most @p count characters from @p pos with the null-terminated
     *  @p cstr. */
    status_t replace(size_type pos, size_type count, const_pointer cstr) noexcept {
        return replace(pos, count, string_view_t(cstr));
    }

    /** Replaces @b (in-place) the characters between @p first and @p last with the
     *  null-terminated @p cstr. */
    status_t replace(const_iterator first, const_iterator last, const_pointer cstr) noexcept {
        return replace(first, last, string_view_t(cstr));
    }

    /**
     *  @brief Replaces @b (in-place) at most @p count characters from @p pos with @p character
     *      repeated @p count2 times, clamping @p pos to the end.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k Past @c max_size.
     */
    status_t replace(size_type pos, size_type count, size_type count2, char_type character) noexcept {
        pos = sz_min_of_two(pos, size()), count = sz_min_of_two(count, size() - pos);
        status_t const status = preparing_replacement(pos, count, count2);
        if (succeeded(status)) std::memset(data() + pos, character, count2);
        return status;
    }

    /** Replaces @b (in-place) the characters between @p first and @p last with @p character
     *  repeated @p count2 times. */
    status_t replace(const_iterator first, const_iterator last, size_type count2, char_type character) noexcept {
        return replace(range_length(cbegin(), first), range_length(first, last), count2, character);
    }

    /** Replaces @b (in-place) the characters between @p first and @p last with those between
     *  @p first2 and @p last2. */
    template <typename input_iterator_type_>
    status_t replace(const_iterator first, const_iterator last, input_iterator_type_ first2,
                     input_iterator_type_ last2) noexcept {
        size_type const pos = range_length(cbegin(), first);
        status_t const status = preparing_replacement(pos, range_length(first, last), range_length(first2, last2));
        for (iterator output = begin() + pos; succeeded(status) && first2 != last2; ++first2, ++output)
            *output = *first2;
        return status;
    }

    /** Replaces @b (in-place) the characters between @p first and @p last with those of @p list. */
    status_t replace(const_iterator first, const_iterator last, std::initializer_list<char_type> list) noexcept {
        return replace(first, last, list.begin(), list.end());
    }

    /**
     *  @brief Removes the last character from the string.
     *  @warning The behavior is @b undefined if the string is empty.
     */
    void pop_back() noexcept { sz_string_erase(&string_, size() - 1, 1); }

    /**
     *  @brief Overwrites the string with @p character repeated @p repeats times.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k Past @c max_size.
     */
    status_t assign(size_type repeats, char_type character) noexcept {
        status_t const status = resize(repeats, character);
        if (succeeded(status)) std::memset(data(), character, repeats);
        return status;
    }

    /** Overwrites the string with @p length characters from @p other. */
    status_t assign(const_pointer other, size_type length) noexcept { return assign(string_view_t(other, length)); }

    /** Overwrites the string with at most @p count characters of @p str after its first @p pos,
     *  clamping @p pos to its end. */
    status_t assign(string_view_t str, size_type pos, size_type count = npos) noexcept {
        return assign(str.substr(pos, count));
    }

    /** Overwrites the string with the characters between @p first and @p last. */
    template <typename input_iterator_type_>
    status_t assign(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        status_t const status = resize(range_length(first, last));
        for (iterator output = begin(); succeeded(status) && first != last; ++first, ++output) *output = *first;
        return status;
    }

    /** Overwrites the string with the characters of @p list. */
    status_t assign(std::initializer_list<char_type> list) noexcept { return assign(list.begin(), list.end()); }

    /** Appends at most @p length characters of @p str after its first @p pos, clamping @p pos to
     *  its end. */
    status_t append(string_view_t str, size_type pos, size_type length = npos) noexcept {
        return append(str.substr(pos, length));
    }

    /** Appends the null-terminated @p str. */
    status_t append(const_pointer str) noexcept { return append(string_view_t(str)); }

    /**
     *  @brief Appends @p character repeated @p repeats times.
     *
     *  @par Statuses:
     *  - @c overflow_risk_k Past @c max_size.
     */
    status_t append(size_type repeats, char_type character) noexcept {
        if (repeats > max_size() - size()) return status_t::overflow_risk_k;
        return resize(size() + repeats, character);
    }

    /** Appends the characters of @p other. */
    status_t append(std::initializer_list<char_type> other) noexcept { return append(other.begin(), other.end()); }

    /** Appends the characters between @p first and @p last. */
    template <typename input_iterator_type_>
    status_t append(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        return insert<input_iterator_type_>(cend(), first, last).status;
    }

#pragma endregion
#pragma endregion

    concatenation<string_view_t, string_view_t> operator|(string_view_t other) const noexcept {
        return {view(), other};
    }

#pragma region Kernel Backed Operations

    template <typename... arguments_types_>
    void starts_with(arguments_types_ &&...) const = delete; // sz::starts_with(text, prefix)
    template <typename... arguments_types_>
    void ends_with(arguments_types_ &&...) const = delete; // sz::ends_with(text, suffix)
    template <typename... arguments_types_>
    void contains(arguments_types_ &&...) const = delete; // sz::contains(text, needle)
    template <typename... arguments_types_>
    void count(arguments_types_ &&...) const = delete; // sz::count(text, needle)
    template <typename... arguments_types_>
    void find(arguments_types_ &&...) const = delete; // sz::find(text, needle)
    template <typename... arguments_types_>
    void rfind(arguments_types_ &&...) const = delete; // sz::rfind(text, needle)
    template <typename... arguments_types_>
    void partition(arguments_types_ &&...) const = delete; // sz::partition(text, pattern)
    template <typename... arguments_types_>
    void rpartition(arguments_types_ &&...) const = delete; // sz::rpartition(text, pattern)
    template <typename... arguments_types_>
    void find_first_of(arguments_types_ &&...) const = delete; // sz::find_first_of(text, set)
    template <typename... arguments_types_>
    void find_first_not_of(arguments_types_ &&...) const = delete; // sz::find_first_not_of(text, set)
    template <typename... arguments_types_>
    void find_last_of(arguments_types_ &&...) const = delete; // sz::find_last_of(text, set)
    template <typename... arguments_types_>
    void find_last_not_of(arguments_types_ &&...) const = delete; // sz::find_last_not_of(text, set)
    template <typename... arguments_types_>
    void contains_only(arguments_types_ &&...) const = delete; // sz::contains_only(text, set)
    template <typename... arguments_types_>
    void is_alpha(arguments_types_ &&...) const = delete; // sz::is_alpha(text)
    template <typename... arguments_types_>
    void is_alnum(arguments_types_ &&...) const = delete; // sz::is_alnum(text)
    template <typename... arguments_types_>
    void is_ascii(arguments_types_ &&...) const = delete; // sz::is_ascii(text)
    template <typename... arguments_types_>
    void is_digit(arguments_types_ &&...) const = delete; // sz::is_digit(text)
    template <typename... arguments_types_>
    void is_lower(arguments_types_ &&...) const = delete; // sz::is_lower(text)
    template <typename... arguments_types_>
    void is_space(arguments_types_ &&...) const = delete; // sz::is_space(text)
    template <typename... arguments_types_>
    void is_upper(arguments_types_ &&...) const = delete; // sz::is_upper(text)
    template <typename... arguments_types_>
    void is_printable(arguments_types_ &&...) const = delete; // sz::is_printable(text)
    template <typename... arguments_types_>
    void lstrip(arguments_types_ &&...) const = delete; // sz::lstrip(text, set)
    template <typename... arguments_types_>
    void rstrip(arguments_types_ &&...) const = delete; // sz::rstrip(text, set)
    template <typename... arguments_types_>
    void strip(arguments_types_ &&...) const = delete; // sz::strip(text, set)
    template <typename... arguments_types_>
    void find_all(arguments_types_ &&...) const = delete; // sz::find_all(text, needle)
    template <typename... arguments_types_>
    void rfind_all(arguments_types_ &&...) const = delete; // sz::rfind_all(text, needle)
    template <typename... arguments_types_>
    void split(arguments_types_ &&...) const = delete; // sz::split(text, delimiter)
    template <typename... arguments_types_>
    void rsplit(arguments_types_ &&...) const = delete; // sz::rsplit(text, delimiter)
    template <typename... arguments_types_>
    void splitlines(arguments_types_ &&...) const = delete; // sz::split_characters(text, newlines_set())
    template <typename... arguments_types_>
    void hash(arguments_types_ &&...) const = delete; // sz::hash(text)
    template <typename... arguments_types_>
    void hash_multiseed(arguments_types_ &&...) const = delete; // sz::hash_multiseed(text, seeds, hashes)
    template <typename... arguments_types_>
    void bytesum(arguments_types_ &&...) const = delete; // sz::bytesum(text)
    template <typename... arguments_types_>
    void is_normalized(arguments_types_ &&...) const = delete; // sz::is_normalized(text, form)
    template <typename... arguments_types_>
    void utf8_find_denormalized(arguments_types_ &&...) const = delete; // sz::utf8_find_denormalized(text, form)
    template <typename... arguments_types_>
    void utf8_count(arguments_types_ &&...) const = delete; // sz::utf8_count(text)
    template <typename... arguments_types_>
    void utf8_seek(arguments_types_ &&...) const = delete; // sz::utf8_seek(text, runes)
    template <typename... arguments_types_>
    void utf8_runes(arguments_types_ &&...) const = delete; // sz::utf8_runes(text)
    template <typename... arguments_types_>
    void utf8_split_newlines(arguments_types_ &&...) const = delete; // sz::utf8_split_newlines(text)
    template <typename... arguments_types_>
    void utf8_newlines(arguments_types_ &&...) const = delete; // sz::utf8_newlines(text)
    template <typename... arguments_types_>
    void utf8_split_whitespaces(arguments_types_ &&...) const = delete; // sz::utf8_split_whitespaces(text)
    template <typename... arguments_types_>
    void utf8_whitespaces(arguments_types_ &&...) const = delete; // sz::utf8_whitespaces(text)
    template <typename... arguments_types_>
    void utf8_split_delimiters(arguments_types_ &&...) const = delete; // sz::utf8_split_delimiters(text)
    template <typename... arguments_types_>
    void utf8_delimiters(arguments_types_ &&...) const = delete; // sz::utf8_delimiters(text)
    template <typename... arguments_types_>
    void utf8_wordbreaks(arguments_types_ &&...) const = delete; // sz::utf8_wordbreaks(text)
    template <typename... arguments_types_>
    void utf8_graphemes(arguments_types_ &&...) const = delete; // sz::utf8_graphemes(text)
    template <typename... arguments_types_>
    void utf8_sentences(arguments_types_ &&...) const = delete; // sz::utf8_sentences(text)
    template <typename... arguments_types_>
    void utf8_linebreaks(arguments_types_ &&...) const = delete; // sz::utf8_linebreaks(text)
    template <typename... arguments_types_>
    void utf8_uncased_order(arguments_types_ &&...) const = delete; // sz::utf8_uncased_order(text, other)
    template <typename... arguments_types_>
    void utf8_uncased_search(arguments_types_ &&...) const = delete; // sz::utf8_uncased_search(text, needle)
    template <typename... arguments_types_>
    void utf8_uncased_matches(arguments_types_ &&...) const = delete; // sz::utf8_uncased_matches(text, needle)
    template <typename... arguments_types_>
    void replace_all(arguments_types_ &&...) const = delete; // sz::replace_all(text, pattern, replacement)
    template <typename... arguments_types_>
    void fill_random(arguments_types_ &&...) const = delete; // sz::fill_random(text.span(), nonce)
    template <typename... arguments_types_>
    void lookup(arguments_types_ &&...) const = delete; // sz::lookup(text.span(), table)
    template <typename... arguments_types_>
    void utf8_uncased_fold(arguments_types_ &&...) const = delete; // sz::utf8_uncased_fold(text)
    template <typename... arguments_types_>
    void utf8_normalize(arguments_types_ &&...) const = delete; // sz::utf8_normalize(text, form)

#pragma endregion

  private:
    /**
     *  @brief Prepares the string for a replacement of a given range with a new string.
     *  @warning Allocates memory if the replacement is longer than the replaced range.
     */
    status_t preparing_replacement(size_type offset, size_type length, size_type new_length) noexcept;
};

using string_t = string<>;

static_assert(sizeof(string_t) == 4 * sizeof(void *), "String size must be 4 pointers.");

namespace literals {
constexpr string_view_t operator""_sv(char const *str, std::size_t length) noexcept { return {str, length}; }
} // namespace literals

template <typename allocator_>
status_t string<allocator_>::resize(size_type count, value_type character) noexcept {
    if (count > max_size()) return status_t::overflow_risk_k;
    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_external);

    // Allocate more space if needed.
    if (count >= string_space) {
        status_t const status = with_allocator_([&](sz_alloc_type &allocator) {
            return sz_string_expand(&string_, STRINGZILLA_SIZE_MAX, count - string_length, &allocator) ? sz_success_k
                                                                                                       : sz_bad_alloc_k;
        });
        if (failed(status)) return status;
        sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_external);
    }

    // Fill the trailing characters.
    if (count > string_length) {
        std::memset(string_start + string_length, character, count - string_length);
        string_start[count] = '\0';
        // Safe for both SSO and heap strings: on little-endian, internal.length
        // overlaps with LSB of external.length, so += affects only the length byte.
        // On big-endian, the struct layout places them at matching positions.
        string_.external.length += count - string_length;
    }
    else { sz_string_erase(&string_, count, STRINGZILLA_SIZE_MAX); }
    return status_t::success_k;
}

template <typename allocator_>
status_t string<allocator_>::assign(string_view_t other) noexcept {
    // Copying the other state as is fails, as its start may point into its own stack buffer.
    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_string_range(&string_, &string_start, &string_length);

    // One nasty special case is when the other string is a substring of this string.
    // It is handled separately, as `sz_string_expand` may invalidate the `other` pointer.
    if (other.data() >= string_start && other.data() < string_start + string_length) {
        auto offset_in_this = other.data() - string_start;
        sz_string_erase(&string_, 0, offset_in_this);
        sz_string_erase(&string_, other.length(), STRINGZILLA_SIZE_MAX);
    }
    // In some of the other cases, when the assigned string is short, we don't need to re-allocate.
    else if (string_length >= other.length()) {
        other.copy(string_start, other.length());
        sz_string_erase(&string_, other.length(), STRINGZILLA_SIZE_MAX);
    }
    // In the common case, however, we need to allocate.
    else {
        return with_allocator_([&](sz_alloc_type &allocator) {
            string_start = sz_string_expand(&string_, STRINGZILLA_SIZE_MAX, other.length() - string_length, &allocator);
            if (!string_start) return sz_bad_alloc_k;
            other.copy(string_start, other.length());
            return sz_success_k;
        });
    }
    return status_t::success_k;
}

template <typename allocator_>
status_t string<allocator_>::push_back(char_type c) noexcept {
    return with_allocator_([&](sz_alloc_type &allocator) {
        auto old_size = size();
        sz_ptr_t start = sz_string_expand(&string_, STRINGZILLA_SIZE_MAX, 1, &allocator);
        if (!start) return sz_bad_alloc_k;
        start[old_size] = c;
        return sz_success_k;
    });
}

template <typename allocator_>
status_t string<allocator_>::append(const_pointer str, size_type length) noexcept {
    return with_allocator_([&](sz_alloc_type &allocator) {
        // Sometimes we are inserting part of this string into itself.
        // By the time `sz_string_expand` finished, the old `str` pointer may be invalidated,
        // so we need to handle that special case separately.
        auto this_span = span();
        if (str >= this_span.begin() && str < this_span.end()) {
            auto offset_in_this = str - data();
            sz_ptr_t start = sz_string_expand(&string_, STRINGZILLA_SIZE_MAX, length, &allocator);
            if (!start) return sz_bad_alloc_k;
            std::memcpy(start + this_span.size(), start + offset_in_this, length);
        }
        else {
            sz_ptr_t start = sz_string_expand(&string_, STRINGZILLA_SIZE_MAX, length, &allocator);
            if (!start) return sz_bad_alloc_k;
            if (length) std::memcpy(start + this_span.size(), str, length);
        }
        return sz_success_k;
    });
}

template <typename allocator_>
template <typename first_type_, typename second_type_>
status_t string<allocator_>::assign(concatenation<first_type_, second_type_> const &other) noexcept {
    // Copying the other state as is fails, as its start may point into its own stack buffer.
    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_string_range(&string_, &string_start, &string_length);

    if (string_length >= other.length()) {
        sz_string_erase(&string_, other.length(), STRINGZILLA_SIZE_MAX);
        other.copy(string_start, other.length());
        return status_t::success_k;
    }
    return with_allocator_([&](sz_alloc_type &allocator) {
        string_start = sz_string_expand(&string_, STRINGZILLA_SIZE_MAX, other.length() - string_length, &allocator);
        if (!string_start) return sz_bad_alloc_k;
        other.copy(string_start, other.length());
        return sz_success_k;
    });
}

template <typename allocator_>
status_t string<allocator_>::preparing_replacement( //
    size_type offset, size_type length, size_type replacement_length) noexcept {

    // There are three cases:
    // 1. The replacement is the same length as the replaced range.
    // 2. The replacement is shorter than the replaced range.
    // 3. The replacement is longer than the replaced range. An allocation may occur.
    sz_assert_(offset + length <= size());

    // 1. The replacement is the same length as the replaced range.
    if (replacement_length == length) return status_t::success_k;

    // 2. The replacement is shorter than the replaced range.
    else if (replacement_length < length) {
        sz_string_erase(&string_, offset + replacement_length, length - replacement_length);
        return status_t::success_k;
    }
    // 3. The replacement is longer than the replaced range. An allocation may occur.
    else {
        if (replacement_length - length > max_size() - size()) return status_t::overflow_risk_k;
        return with_allocator_([&](sz_alloc_type &allocator) {
            return sz_string_expand(&string_, offset + length, replacement_length - length, &allocator)
                       ? sz_success_k
                       : sz_bad_alloc_k;
        });
    }
}

/**
 *  @brief Helper function-like object to order string-view convertible objects with StringZilla.
 *
 *  Similar to <tt>std::less&lt;std::string_view&gt;</tt>, but doesn't require including the heavy <tt>&lt;functional&gt;</tt>
 *  header. Can be used to combine STL classes with StringZilla logic, like:
 *  <tt>std::map&lt;std::string, int, sz::less_t&gt;</tt>.
 *
 *  @see std::less: https://en.cppreference.com/w/cpp/utility/functional/less
 */
struct less_t {

    /** Lets ordered containers find owning keys by views, without building a key to search for. */
    using is_transparent = void;
    constexpr bool operator()(string_view_t a, string_view_t b) const noexcept { return a < b; }
};

/**
 *  @brief Helper function-like object comparing string-view convertible objects with StringZilla.
 *
 *  Similar to <tt>std::equal_to&lt;std::string_view&gt;</tt>, but doesn't require including the heavy
 *  <tt>&lt;functional&gt;</tt> header. Can be used to combine STL classes with StringZilla logic, like:
 *  <tt>std::unordered_map&lt;std::string, int, sz::hash_t, sz::equal_to_t&gt;</tt>.
 *
 *  @see std::equal_to: https://en.cppreference.com/w/cpp/utility/functional/equal_to
 */
struct equal_to_t {

    /** Lets hashed containers find owning keys by views, beside a transparent @ref hash_t. */
    using is_transparent = void;
    constexpr bool operator()(string_view_t a, string_view_t b) const noexcept { return a == b; }
};

/** SFINAE-type used to infer the resulting type of concatenating multiple string together. */
template <typename... args_types_>
struct concatenation_result {};

template <typename first_type_, typename second_type_>
struct concatenation_result<first_type_, second_type_> {
    using type = concatenation<typename concatenation_operand<first_type_>::type,
                               typename concatenation_operand<second_type_>::type>;
};

template <typename first_type_, typename... following_types_>
struct concatenation_result<first_type_, following_types_...> {
    using type = concatenation<typename concatenation_operand<first_type_>::type,
                               typename concatenation_result<following_types_...>::type>;
};

/**
 *  @brief Concatenates two strings into a template expression.
 *  @sa concatenation class for more details.
 */
template <typename first_type_, typename second_type_>
concatenation<typename concatenation_operand<first_type_>::type, typename concatenation_operand<second_type_>::type>
concatenate(first_type_ &&first, second_type_ &&second) noexcept {
    return {first, second};
}

/**
 *  @brief Concatenates two or more strings into a template expression.
 *  @sa concatenation class for more details.
 */
template <typename first_type_, typename second_type_, typename... following_types_>
typename concatenation_result<first_type_, second_type_, following_types_...>::type concatenate(
    first_type_ &&first, second_type_ &&second, following_types_ &&...following) noexcept {
    // A fold expression like the one below would compile faster, but it would add an `if`
    // statement to every `append` call.
    //      std::size_t total_size = (strings.size() + ... + 0);
    //      std::string result;
    //      result.reserve(total_size);
    //      (result.append(strings), ...);
    return ashvardanian::stringzilla::concatenate( //
        std::forward<first_type_>(first),
        ashvardanian::stringzilla::concatenate( //
            std::forward<second_type_>(second), //
            std::forward<following_types_>(following)...));
}

/** Whether @p type_ reads as a run of bytes: a StringZilla or STL string, a span of characters or
 *  a NUL-terminated pointer like a string literal, which @ref view_of measures by @c strlen. */
template <typename type_>
concept a_string = std::convertible_to<type_ const &, char const *> || requires(type_ const &text) {
    { text.data() } -> std::convertible_to<char const *>;
    { text.size() } -> std::convertible_to<std::size_t>;
};

/** Views the bytes of any @ref a_string, measuring a NUL-terminated pointer with @c strlen. */
template <a_string string_type_>
string_view_t view_of(string_type_ const &text) noexcept {
    if constexpr (std::convertible_to<string_type_ const &, char const *>) {
        char const *start = text;
        return {start, std::strlen(start)};
    }
    else return {text.data(), static_cast<std::size_t>(text.size())};
}

/** Whether @p type_ names its own "not found" position, like @c std::string::npos. */
template <typename type_>
concept a_string_with_npos = requires { type_::npos; };

/** The "not found" position for @p string_type_: its own @c npos, like @c std::string::npos, or
 *  StringZilla's for types without one, like string literals. */
template <typename string_type_>
constexpr std::size_t npos_of() noexcept {
    if constexpr (a_string_with_npos<string_type_>) return static_cast<std::size_t>(string_type_::npos);
    else return string_view_t::npos;
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_STRING_HPP_
