/**
 *  @brief  StringZilla C++ wrapper improving over the performance of `std::string_view` and `std::string`,
 *          mostly for substring search, adding approximate matching functionality, and C++23 functionality
 *          to a C++11 compatible implementation.
 */
#ifndef STRINGZILLA_HPP_
#define STRINGZILLA_HPP_

#include <stringzilla/stringzilla.h>

namespace av {
namespace sz {

/**
 *  @brief  A string view class implementing with the superset of C++23 functionality
 *          with much faster SIMD-accelerated substring search and approximate matching.
 */

class string_view {
    sz_cptr_t start_;
    sz_size_t length_;

  public:
    // Member types
    using traits_type = std::char_traits<char>;
    using value_type = char;
    using pointer = char *;
    using const_pointer = char const *;
    using reference = char &;
    using const_reference = char const &;
    using const_iterator = void /* Implementation-defined constant LegacyRandomAccessIterator */;
    using iterator = const_iterator;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator = const_reverse_iterator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Static constant
    static constexpr size_type npos = size_type(-1);

    // Constructors and assignment
    string_view();
    string_view &operator=(string_view const &other); // Copy assignment operator

    // Iterators
    const_iterator begin() const noexcept;
    const_iterator end() const noexcept;
    const_iterator cbegin() const noexcept;
    const_iterator cend() const noexcept;
    const_reverse_iterator rbegin() const noexcept;
    const_reverse_iterator rend() const noexcept;
    const_reverse_iterator crbegin() const noexcept;
    const_reverse_iterator crend() const noexcept;

    // Element access
    reference operator[](size_type pos);
    const_reference operator[](size_type pos) const;
    reference at(size_type pos);
    const_reference at(size_type pos) const;
    reference front();
    const_reference front() const;
    reference back();
    const_reference back() const;
    const_pointer data() const noexcept;

    // Capacity
    size_type size() const noexcept;
    size_type length() const noexcept;
    size_type max_size() const noexcept;
    bool empty() const noexcept;

    // Modifiers
    void remove_prefix(size_type n);
    void remove_suffix(size_type n);
    void swap(string_view &other) noexcept;

    // Operations
    size_type copy(pointer dest, size_type count, size_type pos = 0) const;
    string_view substr(size_type pos = 0, size_type count = npos) const;
    int compare(string_view const &other) const noexcept;
    bool starts_with(string_view const &sv) const noexcept;
    bool ends_with(string_view const &sv) const noexcept;
    bool contains(string_view const &sv) const noexcept;

    // Search
    size_type find(string_view const &sv, size_type pos = 0) const noexcept;
    size_type find(value_type c, size_type pos = 0) const noexcept;
    size_type find(const_pointer s, size_type pos, size_type count) const noexcept;
    size_type find(const_pointer s, size_type pos = 0) const noexcept;

    // Reverse-order Search
    size_type rfind(string_view const &sv, size_type pos = 0) const noexcept;
    size_type rfind(value_type c, size_type pos = 0) const noexcept;
    size_type rfind(const_pointer s, size_type pos, size_type count) const noexcept;
    size_type rfind(const_pointer s, size_type pos = 0) const noexcept;
};

} // namespace sz
} // namespace av

#endif // STRINGZILLA_HPP_
