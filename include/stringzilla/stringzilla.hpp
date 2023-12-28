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

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief  A set of characters represented as a bitset with 256 slots.
 */
class character_set {
    sz_u8_set_t bitset_;

  public:
    character_set() noexcept { sz_u8_set_init(&bitset_); }
    character_set(character_set const &other) noexcept : bitset_(other.bitset_) {}
    character_set &operator=(character_set const &other) noexcept {
        bitset_ = other.bitset_;
        return *this;
    }

    sz_u8_set_t &raw() noexcept { return bitset_; }
    bool contains(char c) const noexcept { return sz_u8_set_contains(&bitset_, c); }
    character_set &add(char c) noexcept {
        sz_u8_set_add(&bitset_, c);
        return *this;
    }
    character_set &invert() noexcept {
        sz_u8_set_invert(&bitset_);
        return *this;
    }
};

/**
 *  @brief  Zero-cost wrapper around the `.find` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find {
    using size_type = typename string_view_::size_type;
    string_view_ needle_;
    size_type needle_length() const noexcept { return needle_.length(); }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find(needle_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.rfind` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_rfind {
    using size_type = typename string_view_::size_type;
    string_view_ needle_;
    size_type needle_length() const noexcept { return needle_.length(); }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.rfind(needle_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_first_of` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find_first_of {
    using size_type = typename string_view_::size_type;
    string_view_ needles_;
    size_type needle_length() const noexcept { return 1; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find_first_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_last_of` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find_last_of {
    using size_type = typename string_view_::size_type;
    string_view_ needles_;
    size_type needle_length() const noexcept { return 1; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find_last_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_first_not_of` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find_first_not_of {
    using size_type = typename string_view_::size_type;
    string_view_ needles_;
    size_type needle_length() const noexcept { return 1; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find_first_not_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_last_not_of` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find_last_not_of {
    using size_type = typename string_view_::size_type;
    string_view_ needles_;
    size_type needle_length() const noexcept { return 1; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find_last_not_of(needles_); }
};

/**
 *  @brief  A range of string views representing the matches of a substring search.
 *          Compatible with C++23 ranges, C++11 string views, and of course, StringZilla.
 *
 *  TODO: Apply Galil rule to match repetitve patterns in strictly linear time.
 */
template <typename string_view_, template <typename> typename matcher_template_>
class range_matches {
    using string_view = string_view_;
    using matcher = matcher_template_<string_view>;

    string_view haystack_;
    matcher matcher_;
    std::size_t skip_after_match_ = 1;

  public:
    range_matches(string_view haystack, string_view needle, bool allow_interleaving = true)
        : haystack_(haystack), matcher_(needle), skip_after_match_(allow_interleaving ? 1 : matcher_.needle_length()) {}

    class iterator {
        string_view remaining_;
        matcher matcher_;
        std::size_t skip_after_match_ = 1;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view;
        using pointer = string_view const *;
        using reference = string_view const &;

        iterator(string_view haystack, matcher matcher, std::size_t skip_after_match = 1) noexcept
            : remaining_(haystack), matcher_(matcher), skip_after_match_(skip_after_match) {}
        value_type operator*() const noexcept { return remaining_.substr(0, matcher_.needle_length()); }

        iterator &operator++() noexcept {
            remaining_.remove_prefix(skip_after_match_);
            auto position = matcher_(remaining_);
            remaining_ = position != string_view::npos ? remaining_.substr(position) : string_view();
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator const &other) const noexcept { return remaining_.size() != other.remaining_.size(); }
        bool operator==(iterator const &other) const noexcept { return remaining_.size() == other.remaining_.size(); }
    };

    iterator begin() const noexcept {
        auto position = matcher_(haystack_);
        return iterator(position != string_view::npos ? haystack_.substr(position) : string_view(), matcher_,
                        skip_after_match_);
    }

    iterator end() const noexcept { return iterator(string_view(), matcher_, skip_after_match_); }
    iterator::difference_type size() const noexcept { return std::distance(begin(), end()); }
};

/**
 *  @brief  A range of string views representing the matches of a @b reverse-order substring search.
 *          Compatible with C++23 ranges, C++11 string views, and of course, StringZilla.
 *
 *  TODO: Apply Galil rule to match repetitve patterns in strictly linear time.
 */
template <typename string_view_, template <typename> typename matcher_template_>
class reverse_range_matches {
    using string_view = string_view_;
    using matcher = matcher_template_<string_view>;

    string_view haystack_;
    matcher matcher_;
    std::size_t skip_after_match_ = 1;

  public:
    reverse_range_matches(string_view haystack, string_view needle, bool allow_interleaving = true)
        : haystack_(haystack), matcher_(needle), skip_after_match_(allow_interleaving ? 1 : matcher_.needle_length()) {}

    class iterator {
        string_view remaining_;
        matcher matcher_;
        std::size_t skip_after_match_ = 1;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view;
        using pointer = string_view const *;
        using reference = string_view const &;

        iterator(string_view haystack, matcher matcher, std::size_t skip_after_match = 1) noexcept
            : remaining_(haystack), matcher_(matcher), skip_after_match_(skip_after_match) {}
        value_type operator*() const noexcept {
            return remaining_.substr(remaining_.size() - matcher_.needle_length());
        }

        iterator &operator++() noexcept {
            remaining_.remove_suffix(skip_after_match_);
            auto position = matcher_(remaining_);
            remaining_ = position != string_view::npos ? remaining_.substr(0, position + matcher_.needle_length())
                                                       : string_view();
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator const &other) const noexcept { return remaining_.data() != other.remaining_.data(); }
        bool operator==(iterator const &other) const noexcept { return remaining_.data() == other.remaining_.data(); }
    };

    iterator begin() const noexcept {
        auto position = matcher_(haystack_);
        return iterator(
            position != string_view::npos ? haystack_.substr(0, position + matcher_.needle_length()) : string_view(),
            matcher_, skip_after_match_);
    }

    iterator end() const noexcept { return iterator(string_view(), matcher_, skip_after_match_); }
    iterator::difference_type size() const noexcept { return std::distance(begin(), end()); }
};

template <typename string>
range_matches<string, matcher_find> find_all(string h, string n, bool allow_interleaving = true) {
    return {h, n};
}

template <typename string>
reverse_range_matches<string, matcher_rfind> rfind_all(string h, string n, bool allow_interleaving = true) {
    return {h, n};
}

template <typename string>
range_matches<string, matcher_find_first_of> find_all_characters(string h, string n) {
    return {h, n};
}

template <typename string>
reverse_range_matches<string, matcher_find_last_of> rfind_all_characters(string h, string n) {
    return {h, n};
}

template <typename string>
range_matches<string, matcher_find_first_not_of> find_all_other_characters(string h, string n) {
    return {h, n};
}

template <typename string>
reverse_range_matches<string, matcher_find_last_not_of> rfind_all_other_characters(string h, string n) {
    return {h, n};
}

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
    size_type find_first_of(string_view other) const noexcept { return find_first_of(other.as_set()); }

    /** @brief  Find the first occurence of a character outside of the set. */
    size_type find_first_not_of(string_view other) const noexcept { return find_first_not_of(other.as_set()); }

    /** @brief  Find the last occurence of a character from a set. */
    size_type find_last_of(string_view other) const noexcept { return find_last_of(other.as_set()); }

    /** @brief  Find the last occurence of a character outside of the set. */
    size_type find_last_not_of(string_view other) const noexcept { return find_last_not_of(other.as_set()); }

    /** @brief  Find the first occurence of a character from a set. */
    size_type find_first_of(character_set set) const noexcept {
        auto ptr = sz_find_from_set(start_, length_, &set.raw());
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurence of a character outside of the set. */
    size_type find_first_not_of(character_set set) const noexcept { return find_first_of(set.invert()); }

    /** @brief  Find the last occurence of a character from a set. */
    size_type find_last_of(character_set set) const noexcept {
        auto ptr = sz_find_last_from_set(start_, length_, &set.raw());
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the last occurence of a character outside of the set. */
    size_type find_last_not_of(character_set set) const noexcept { return find_last_of(set.invert()); }

    struct split_result {
        string_view before;
        string_view match;
        string_view after;
    };

    /** @brief  Split the string into three parts, before the match, the match itself, and after it. */
    template <typename pattern_>
    split_result split(pattern_ &&pattern) const noexcept {
        size_type pos = find(pattern);
        if (pos == npos) return {substr(), string_view(), string_view()};
        return {substr(0, pos), substr(pos, pattern.size()), substr(pos + pattern.size())};
    }

    /** @brief  Split the string into three parts, before the last match, the last match itself, and after it. */
    template <typename pattern_>
    split_result rsplit(pattern_ &&pattern) const noexcept {
        size_type pos = rfind(pattern);
        if (pos == npos) return {substr(), string_view(), string_view()};
        return {substr(0, pos), substr(pos, pattern.size()), substr(pos + pattern.size())};
    }

    size_type copy(pointer destination, size_type count, size_type pos = 0) const noexcept = delete;

    size_type hash() const noexcept { return static_cast<size_type>(sz_hash(start_, length_)); }

    character_set as_set() const noexcept {
        character_set set;
        for (auto c : *this) set.add(c);
        return set;
    }

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

template <>
struct matcher_find_first_of<string_view> {
    using size_type = typename string_view::size_type;
    sz_u8_set_t needles_set_;
    matcher_find_first_of(string_view needle) noexcept : needles_set_(needle.character_set()) {}
    size_type needle_length() const noexcept { return 1; }
    size_type operator()(string_view haystack) const noexcept { return haystack.find_first_of(needles_set_); }
};

template <>
struct matcher_find_last_of<string_view> {
    using size_type = typename string_view::size_type;
    sz_u8_set_t needles_set_;
    matcher_find_last_of(string_view needle) noexcept : needles_set_(needle.character_set()) {}
    size_type needle_length() const noexcept { return 1; }
    size_type operator()(string_view haystack) const noexcept { return haystack.find_last_of(needles_set_); }
};

template <>
struct matcher_find_first_not_of<string_view> {
    using size_type = typename string_view::size_type;
    sz_u8_set_t needles_set_;
    matcher_find_first_not_of(string_view needle) noexcept : needles_set_(needle.character_set()) {}
    size_type needle_length() const noexcept { return 1; }
    size_type operator()(string_view haystack) const noexcept { return haystack.find_first_not_of(needles_set_); }
};

template <>
struct matcher_find_last_not_of<string_view> {
    using size_type = typename string_view::size_type;
    sz_u8_set_t needles_set_;
    matcher_find_last_not_of(string_view needle) noexcept : needles_set_(needle.character_set()) {}
    size_type needle_length() const noexcept { return 1; }
    size_type operator()(string_view haystack) const noexcept { return haystack.find_last_not_of(needles_set_); }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_HPP_
