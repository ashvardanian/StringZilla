/**
 *  @brief  StringZilla C++ wrapper improving over the performance of `std::string_view` and `std::string`,
 *          mostly for substring search, adding approximate matching functionality, and C++23 functionality
 *          to a C++11 compatible implementation.
 *
 *  This implementation is aiming to be compatible with C++11, while imeplementing the C++23 functinoality.
 *  By default, it includes C++ STL headers, but that can be avoided to minimize compilation overhead.
 *  https://artificial-mind.net/projects/compile-health/
 */
#ifndef STRINGZILLA_HPP_
#define STRINGZILLA_HPP_

#include <stringzilla/stringzilla.h>

namespace av {
namespace sz {

/**
 *  @brief  A range of string views representing the matches of a substring search.
 *          Compatible with C++23 ranges, C++11 string views, and of course, StringZilla.
 */
template <typename string_view_>
class substring_matches_range {

  public:
    using string_view = string_view_;

    substring_matches_range(string_view haystack, string_view needle) : haystack_(haystack), needle_(needle) {}

    class iterator {
        string_view remaining_;
        string_view needle_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view;
        using pointer = string_view const *;
        using reference = string_view const &;

        iterator(string_view haystack, string_view needle) noexcept : remaining_(haystack), needle_(needle) {}
        value_type operator*() const noexcept { return remaining_.substr(0, needle_.size()); }

        iterator &operator++() noexcept {
            auto position = remaining_.find(needle_);
            remaining_ = remaining_.substr(position);
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator const &other) const noexcept { return !(*this == other); }
        bool operator==(iterator const &other) const noexcept {
            return remaining_.begin() == other.remaining_.begin() && remaining_.end() == other.remaining_.end();
        }
    };

    iterator begin() const noexcept {
        auto position = haystack_.find(needle_);
        return iterator(haystack_.substr(position), needle_);
    }

    iterator end() const noexcept { return iterator(string_view(), needle_); }

  private:
    string_view haystack_;
    string_view needle_;
};

// C++17 deduction guides

template <typename string_view_>
substring_matches_range(string_view_, string_view_) -> substring_matches_range<string_view_>;

/**
 *  @brief  A string view class implementing with the superset of C++23 functionality
 *          with much faster SIMD-accelerated substring search and approximate matching.
 *          Unlike STL, never raises exceptions.
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
    using const_iterator = char const *;
    using iterator = const_iterator;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using reverse_iterator = const_reverse_iterator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    /** @brief  Special value for missing matches. */
    static constexpr size_type npos = size_type(-1);

    string_view() noexcept : start_(nullptr), length_(0) {}
    string_view(const_pointer c_string) noexcept : start_(c_string), length_(null_terminated_length(c_string)) {}
    string_view(const_pointer c_string, size_type length) noexcept : start_(c_string), length_(length) {}
    string_view(string_view const &other) noexcept : start_(other.start_), length_(other.length_) {}
    string_view &operator=(string_view const &other) noexcept { return assign(other); }
    string_view(std::nullptr_t) = delete;

    string_view(std::string const &other) noexcept : string_view(other.data(), other.size()) {}
    string_view(std::string_view const &other) noexcept : string_view(other.data(), other.size()) {}
    string_view &operator=(std::string const &other) noexcept { return assign({other.data(), other.size()}); }
    string_view &operator=(std::string_view const &other) noexcept { return assign({other.data(), other.size()}); }

    const_iterator begin() const noexcept { return const_iterator(start_); }
    const_iterator end() const noexcept { return const_iterator(start_ + length_); }
    const_iterator cbegin() const noexcept { return const_iterator(start_); }
    const_iterator cend() const noexcept { return const_iterator(start_ + length_); }
    const_reverse_iterator rbegin() const noexcept;
    const_reverse_iterator rend() const noexcept;
    const_reverse_iterator crbegin() const noexcept;
    const_reverse_iterator crend() const noexcept;

    const_reference operator[](size_type pos) const noexcept { return start_[pos]; }
    const_reference at(size_type pos) const noexcept { return start_[pos]; }
    const_reference front() const noexcept { return start_[0]; }
    const_reference back() const noexcept { return start_[length_ - 1]; }
    const_pointer data() const noexcept { return start_; }

    size_type size() const noexcept { return length_; }
    size_type length() const noexcept { return length_; }
    size_type max_size() const noexcept { return sz_size_max; }
    bool empty() const noexcept { return length_ == 0; }

    /** @brief Removes the first `n` characters from the view. The behavior is undefined if `n > size()`. */
    void remove_prefix(size_type n) noexcept { assert(n <= size()), start_ += n, length_ -= n; }

    /** @brief Removes the last `n` characters from the view. The behavior is undefined if `n > size()`. */
    void remove_suffix(size_type n) noexcept { assert(n <= size()), length_ -= n; }

    /** @brief Exchanges the view with that of the `other`. */
    void swap(string_view &other) noexcept { std::swap(start_, other.start_), std::swap(length_, other.length_); }

    /** @brief  Added for STL compatibility. */
    string_view substr() const noexcept { return *this; }

    /** @brief  Equivalent of `remove_prefix(pos)`. The behavior is undefined if `pos > size()`. */
    string_view substr(size_type pos) const noexcept { return string_view(start_ + pos, length_ - pos); }

    /** @brief  Returns a subview [pos, pos + rlen), where `rlen` is the smaller of count and `size() - pos`.
     *          Equivalent to `substr(pos).substr(0, count)` or combining `remove_prefix` and `remove_suffix`.
     *          The behavior is undefined if `pos > size()`.  */
    string_view substr(size_type pos, size_type count) const noexcept {
        return string_view(start_ + pos, sz_min_of_two(count, length_ - pos));
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(string_view other) const noexcept {
        return (int)sz_order(start_, length_, other.start_, other.length_);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(size_type pos1, size_type count1, string_view other) const noexcept {
        return substr(pos1, count1).compare(other);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(size_type pos1, size_type count1, string_view other, size_type pos2, size_type count2) const noexcept {
        return substr(pos1, count1).compare(other.substr(pos2, count2));
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(const_pointer other) const noexcept { return compare(string_view(other)); }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(size_type pos1, size_type count1, const_pointer other) const noexcept {
        return substr(pos1, count1).compare(string_view(other));
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(size_type pos1, size_type count1, const_pointer other, size_type count2) const noexcept {
        return substr(pos1, count1).compare(string_view(other, count2));
    }

    /** @brief  Checks if the string is equal to the other string. */
    bool operator==(string_view other) const noexcept {
        return length_ == other.length_ && sz_equal(start_, other.start_, other.length_) == sz_true_k;
    }

#if __cplusplus >= 201402L
#define sz_deprecate_compare [[deprecated("Use the three-way comparison operator (<=>) in C++20 and later")]]
#else
#define sz_deprecate_compare
#endif

    /** @brief  Checks if the string is not equal to the other string. */
    sz_deprecate_compare bool operator!=(string_view other) const noexcept {
        return length_ != other.length_ || sz_equal(start_, other.start_, other.length_) == sz_false_k;
    }

    /** @brief  Checks if the string is lexicographically smaller than the other string. */
    sz_deprecate_compare bool operator<(string_view other) const noexcept { return compare(other) == sz_less_k; }

    /** @brief  Checks if the string is lexicographically equal or smaller than the other string. */
    sz_deprecate_compare bool operator<=(string_view other) const noexcept { return compare(other) != sz_greater_k; }

    /** @brief  Checks if the string is lexicographically greater than the other string. */
    sz_deprecate_compare bool operator>(string_view other) const noexcept { return compare(other) == sz_greater_k; }

    /** @brief  Checks if the string is lexicographically equal or greater than the other string. */
    sz_deprecate_compare bool operator>=(string_view other) const noexcept { return compare(other) != sz_less_k; }

#if __cplusplus >= 202002L

    /** @brief  Checks if the string is not equal to the other string. */
    int operator<=>(string_view other) const noexcept { return compare(other); }
#endif

    /** @brief  Checks if the string starts with the other string. */
    bool starts_with(string_view other) const noexcept {
        return length_ >= other.length_ && sz_equal(start_, other.start_, other.length_) == sz_true_k;
    }

    /** @brief  Checks if the string starts with the other string. */
    bool starts_with(const_pointer other) const noexcept {
        auto other_length = null_terminated_length(other);
        return length_ >= other_length && sz_equal(start_, other, other_length) == sz_true_k;
    }

    /** @brief  Checks if the string starts with the other character. */
    bool starts_with(value_type other) const noexcept { return length_ && start_[0] == other; }

    /** @brief  Checks if the string ends with the other string. */
    bool ends_with(string_view other) const noexcept {
        return length_ >= other.length_ &&
               sz_equal(start_ + length_ - other.length_, other.start_, other.length_) == sz_true_k;
    }

    /** @brief  Checks if the string ends with the other string. */
    bool ends_with(const_pointer other) const noexcept {
        auto other_length = null_terminated_length(other);
        return length_ >= other_length && sz_equal(start_ + length_ - other_length, other, other_length) == sz_true_k;
    }

    /** @brief  Checks if the string ends with the other character. */
    bool ends_with(value_type other) const noexcept { return length_ && start_[length_ - 1] == other; }

    /** @brief  Find the first occurence of a substring. */
    size_type find(string_view other) const noexcept {
        auto ptr = sz_find(start_, length_, other.start_, other.length_);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurence of a substring. The behavior is undefined if `pos > size()`. */
    size_type find(string_view other, size_type pos) const noexcept { return substr(pos).find(other); }

    /** @brief  Find the first occurence of a character. */
    size_type find(value_type character) const noexcept {
        auto ptr = sz_find_byte(start_, length_, &character);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurence of a character. The behavior is undefined if `pos > size()`. */
    size_type find(value_type character, size_type pos) const noexcept { return substr(pos).find(character); }

    /** @brief  Find the first occurence of a substring. The behavior is undefined if `pos > size()`. */
    size_type find(const_pointer other, size_type pos, size_type count) const noexcept {
        return substr(pos).find(string_view(other, count));
    }

    /** @brief  Find the first occurence of a substring. The behavior is undefined if `pos > size()`. */
    size_type find(const_pointer other, size_type pos = 0) const noexcept {
        return substr(pos).find(string_view(other));
    }

    /** @brief  Find the first occurence of a substring. */
    size_type rfind(string_view other) const noexcept {
        auto ptr = sz_find_last(start_, length_, other.start_, other.length_);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurence of a substring. The behavior is undefined if `pos > size()`. */
    size_type rfind(string_view other, size_type pos) const noexcept { return substr(pos).rfind(other); }

    /** @brief  Find the first occurence of a character. */
    size_type rfind(value_type character) const noexcept {
        auto ptr = sz_find_last_byte(start_, length_, &character);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurence of a character. The behavior is undefined if `pos > size()`. */
    size_type rfind(value_type character, size_type pos) const noexcept { return substr(pos).rfind(character); }

    /** @brief  Find the first occurence of a substring. The behavior is undefined if `pos > size()`. */
    size_type rfind(const_pointer other, size_type pos, size_type count) const noexcept {
        return substr(pos).rfind(string_view(other, count));
    }

    /** @brief  Find the first occurence of a substring. The behavior is undefined if `pos > size()`. */
    size_type rfind(const_pointer other, size_type pos = 0) const noexcept {
        return substr(pos).rfind(string_view(other));
    }

    bool contains(string_view other) const noexcept { return find(other) != npos; }
    bool contains(value_type character) const noexcept { return find(character) != npos; }
    bool contains(const_pointer other) const noexcept { return find(other) != npos; }

    /** @brief  Find the first occurence of a character from a set. */
    size_type find_first_of(string_view other) const noexcept {
        sz_u8_set_t set;
        sz_u8_set_init(&set);
        for (auto c : other) sz_u8_set_add(&set, c);
        auto ptr = sz_find_from_set(start_, length_, &set);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurence of a character outside of the set. */
    size_type find_first_not_of(string_view other) const noexcept {
        sz_u8_set_t set;
        sz_u8_set_init(&set);
        for (auto c : other) sz_u8_set_add(&set, c);
        sz_u8_set_invert(&set);
        auto ptr = sz_find_from_set(start_, length_, &set);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the last occurence of a character from a set. */
    size_type find_last_of(string_view other) const noexcept {
        sz_u8_set_t set;
        sz_u8_set_init(&set);
        for (auto c : other) sz_u8_set_add(&set, c);
        auto ptr = sz_find_last_from_set(start_, length_, &set);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the last occurence of a character outside of the set. */
    size_type find_last_not_of(string_view other) const noexcept {
        sz_u8_set_t set;
        sz_u8_set_init(&set);
        for (auto c : other) sz_u8_set_add(&set, c);
        sz_u8_set_invert(&set);
        auto ptr = sz_find_last_from_set(start_, length_, &set);
        return ptr ? ptr - start_ : npos;
    }

    size_type copy(pointer destination, size_type count, size_type pos = 0) const noexcept = delete;

  private:
    string_view &assign(string_view const &other) noexcept {
        start_ = other.start_;
        length_ = other.length_;
        return *this;
    }
    static size_type null_terminated_length(const_pointer s) noexcept {
        const_pointer p = s;
        while (*p) ++p;
        return p - s;
    }
};

} // namespace sz
} // namespace av

#endif // STRINGZILLA_HPP_
