/**
 *  @brief  StringZilla C++ wrapper improving over the performance of `std::string_view` and `std::string`,
 *          mostly for substring search, adding approximate matching functionality, and C++23 functionality
 *          to a C++11 compatible implementation.
 *
 *  This implementation is aiming to be compatible with C++11, while implementing the C++23 functionality.
 *  By default, it includes C++ STL headers, but that can be avoided to minimize compilation overhead.
 *  https://artificial-mind.net/projects/compile-health/
 */
#ifndef STRINGZILLA_HPP_
#define STRINGZILLA_HPP_

#ifndef SZ_INCLUDE_STL_CONVERSIONS
#define SZ_INCLUDE_STL_CONVERSIONS 1
#endif

#if SZ_INCLUDE_STL_CONVERSIONS
#include <string>
#include <string_view>
#endif

#include <cassert> // `assert`
#include <cstddef> // `std::size_t`

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
    explicit character_set(char const *chars) noexcept : character_set() {
        for (std::size_t i = 0; chars[i]; ++i) add(chars[i]);
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
 *  @brief  A result of a string split operation, containing the string slice ::before,
 *          the ::match itself, and the slice ::after.
 */
template <typename string_>
struct string_split_result {
    string_ before;
    string_ match;
    string_ after;
};

/**
 *  @brief  Zero-cost wrapper around the `.find` member function of string-like classes.
 *
 *  TODO: Apply Galil rule to match repetitive patterns in strictly linear time.
 */
template <typename string_view_>
struct matcher_find {
    using size_type = typename string_view_::size_type;
    string_view_ needle_;
    std::size_t skip_after_match_ = 1;

    matcher_find(string_view_ needle = {}, bool allow_overlaps = true) noexcept
        : needle_(needle), skip_after_match_(allow_overlaps ? 1 : needle_.length()) {}
    size_type needle_length() const noexcept { return needle_.length(); }
    size_type skip_length() const noexcept { return skip_after_match_; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find(needle_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.rfind` member function of string-like classes.
 *
 *  TODO: Apply Galil rule to match repetitive patterns in strictly linear time.
 */
template <typename string_view_>
struct matcher_rfind {
    using size_type = typename string_view_::size_type;
    string_view_ needle_;
    std::size_t skip_after_match_ = 1;

    matcher_rfind(string_view_ needle = {}, bool allow_overlaps = true) noexcept
        : needle_(needle), skip_after_match_(allow_overlaps ? 1 : needle_.length()) {}
    size_type needle_length() const noexcept { return needle_.length(); }
    size_type skip_length() const noexcept { return skip_after_match_; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.rfind(needle_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_first_of` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find_first_of {
    using size_type = typename string_view_::size_type;
    string_view_ needles_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find_first_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_last_of` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find_last_of {
    using size_type = typename string_view_::size_type;
    string_view_ needles_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find_last_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_first_not_of` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find_first_not_of {
    using size_type = typename string_view_::size_type;
    string_view_ needles_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find_first_not_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_last_not_of` member function of string-like classes.
 */
template <typename string_view_>
struct matcher_find_last_not_of {
    using size_type = typename string_view_::size_type;
    string_view_ needles_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(string_view_ haystack) const noexcept { return haystack.find_last_not_of(needles_); }
};

struct end_sentinel_type {};
inline static constexpr end_sentinel_type end_sentinel;

/**
 *  @brief  A range of string slices representing the matches of a substring search.
 *          Compatible with C++23 ranges, C++11 string views, and of course, StringZilla.
 */
template <typename string_view_, template <typename> typename matcher_template_>
class range_matches {
    using string_view = string_view_;
    using matcher = matcher_template_<string_view>;

    string_view haystack_;
    matcher matcher_;

  public:
    range_matches(string_view haystack, matcher needle) : haystack_(haystack), matcher_(needle) {}

    class iterator {
        matcher matcher_;
        string_view remaining_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view;
        using pointer = string_view;   // Needed for compatibility with STL container constructors.
        using reference = string_view; // Needed for compatibility with STL container constructors.

        iterator(string_view haystack, matcher matcher) noexcept : matcher_(matcher), remaining_(haystack) {
            auto position = matcher_(remaining_);
            remaining_.remove_prefix(position != string_view::npos ? position : remaining_.size());
        }

        value_type operator*() const noexcept { return remaining_.substr(0, matcher_.needle_length()); }

        iterator &operator++() noexcept {
            remaining_.remove_prefix(matcher_.skip_length());
            auto position = matcher_(remaining_);
            remaining_.remove_prefix(position != string_view::npos ? position : remaining_.size());
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator const &other) const noexcept { return remaining_.begin() != other.remaining_.begin(); }
        bool operator==(iterator const &other) const noexcept { return remaining_.begin() == other.remaining_.begin(); }
        bool operator!=(end_sentinel_type) const noexcept { return !remaining_.empty(); }
        bool operator==(end_sentinel_type) const noexcept { return remaining_.empty(); }
    };

    iterator begin() const noexcept { return iterator(haystack_, matcher_); }
    iterator end() const noexcept { return iterator({haystack_.end(), 0}, matcher_); }
    typename iterator::difference_type size() const noexcept { return std::distance(begin(), end()); }
    bool empty() const noexcept { return begin() == end_sentinel; }
    bool allow_overlaps() const noexcept { return matcher_.skip_length() < matcher_.needle_length(); }

    /**
     *  @brief  Copies the matches into a container.
     */
    template <typename container_>
    void to(container_ &container) {
        for (auto match : *this) { container.push_back(match); }
    }

    /**
     *  @brief  Copies the matches into a consumed container, returning it at the end.
     */
    template <typename container_>
    container_ to() {
        return container_ {begin(), end()};
    }
};

/**
 *  @brief  A range of string slices representing the matches of a @b reverse-order substring search.
 *          Compatible with C++23 ranges, C++11 string views, and of course, StringZilla.
 */
template <typename string_view_, template <typename> typename matcher_template_>
class range_rmatches {
    using string_view = string_view_;
    using matcher = matcher_template_<string_view>;

    matcher matcher_;
    string_view haystack_;

  public:
    range_rmatches(string_view haystack, matcher needle) : haystack_(haystack), matcher_(needle) {}

    class iterator {
        string_view remaining_;
        matcher matcher_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view;
        using pointer = string_view;   // Needed for compatibility with STL container constructors.
        using reference = string_view; // Needed for compatibility with STL container constructors.

        iterator(string_view haystack, matcher matcher) noexcept : matcher_(matcher), remaining_(haystack) {
            auto position = matcher_(remaining_);
            remaining_.remove_suffix(position != string_view::npos
                                         ? remaining_.size() - position - matcher_.needle_length()
                                         : remaining_.size());
        }

        value_type operator*() const noexcept {
            return remaining_.substr(remaining_.size() - matcher_.needle_length());
        }

        iterator &operator++() noexcept {
            remaining_.remove_suffix(matcher_.skip_length());
            auto position = matcher_(remaining_);
            remaining_.remove_suffix(position != string_view::npos
                                         ? remaining_.size() - position - matcher_.needle_length()
                                         : remaining_.size());
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator const &other) const noexcept { return remaining_.end() != other.remaining_.end(); }
        bool operator==(iterator const &other) const noexcept { return remaining_.end() == other.remaining_.end(); }
        bool operator!=(end_sentinel_type) const noexcept { return !remaining_.empty(); }
        bool operator==(end_sentinel_type) const noexcept { return remaining_.empty(); }
    };

    iterator begin() const noexcept { return iterator(haystack_, matcher_); }
    iterator end() const noexcept { return iterator({haystack_.begin(), 0}, matcher_); }
    typename iterator::difference_type size() const noexcept { return std::distance(begin(), end()); }
    bool empty() const noexcept { return begin() == end_sentinel; }
    bool allow_overlaps() const noexcept { return matcher_.skip_length() < matcher_.needle_length(); }

    /**
     *  @brief  Copies the matches into a container.
     */
    template <typename container_>
    void to(container_ &container) {
        for (auto match : *this) { container.push_back(match); }
    }

    /**
     *  @brief  Copies the matches into a consumed container, returning it at the end.
     */
    template <typename container_>
    container_ to() {
        return container_ {begin(), end()};
    }
};

/**
 *  @brief  A range of string slices for different splits of the data.
 *          Compatible with C++23 ranges, C++11 string views, and of course, StringZilla.
 *
 *  In some sense, represents the inverse operation to `range_matches`, as it reports not the search matches
 *  but the data between them. Meaning that for `N` search matches, there will be `N+1` elements in the range.
 *  Unlike ::range_matches, this range can't be empty. It also can't report overlapping intervals.
 */
template <typename string_view_, template <typename> typename matcher_template_>
class range_splits {
    using string_view = string_view_;
    using matcher = matcher_template_<string_view>;

    string_view haystack_;
    matcher matcher_;

  public:
    range_splits(string_view haystack, matcher needle) : haystack_(haystack), matcher_(needle) {}

    class iterator {
        matcher matcher_;
        string_view remaining_;
        std::size_t length_within_remaining_;
        bool reached_tail_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view;
        using pointer = string_view;   // Needed for compatibility with STL container constructors.
        using reference = string_view; // Needed for compatibility with STL container constructors.

        iterator(string_view haystack, matcher matcher) noexcept : matcher_(matcher), remaining_(haystack) {
            auto position = matcher_(remaining_);
            length_within_remaining_ = position != string_view::npos ? position : remaining_.size();
            reached_tail_ = false;
        }

        iterator(string_view haystack, matcher matcher, end_sentinel_type) noexcept
            : matcher_(matcher), remaining_(haystack), reached_tail_(true) {}

        value_type operator*() const noexcept { return remaining_.substr(0, length_within_remaining_); }

        iterator &operator++() noexcept {
            remaining_.remove_prefix(length_within_remaining_);
            reached_tail_ = remaining_.empty();
            remaining_.remove_prefix(matcher_.needle_length() * !reached_tail_);
            auto position = matcher_(remaining_);
            length_within_remaining_ = position != string_view::npos ? position : remaining_.size();
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator const &other) const noexcept {
            return (remaining_.begin() != other.remaining_.begin()) || (reached_tail_ != other.reached_tail_);
        }
        bool operator==(iterator const &other) const noexcept {
            return (remaining_.begin() == other.remaining_.begin()) && (reached_tail_ == other.reached_tail_);
        }
        bool operator!=(end_sentinel_type) const noexcept { return !remaining_.empty() || !reached_tail_; }
        bool operator==(end_sentinel_type) const noexcept { return remaining_.empty() && reached_tail_; }
    };

    iterator begin() const noexcept { return iterator(haystack_, matcher_); }
    iterator end() const noexcept { return iterator({haystack_.end(), 0}, matcher_, end_sentinel); }
    typename iterator::difference_type size() const noexcept { return std::distance(begin(), end()); }
    constexpr bool empty() const noexcept { return false; }

    /**
     *  @brief  Copies the matches into a container.
     */
    template <typename container_>
    void to(container_ &container) {
        for (auto match : *this) { container.push_back(match); }
    }

    /**
     *  @brief  Copies the matches into a consumed container, returning it at the end.
     */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto match : *this) { container.push_back(match); }
        return std::move(container);
    }
};

/**
 *  @brief  A range of string slices for different splits of the data in @b reverse-order.
 *          Compatible with C++23 ranges, C++11 string views, and of course, StringZilla.
 *
 *  In some sense, represents the inverse operation to `range_matches`, as it reports not the search matches
 *  but the data between them. Meaning that for `N` search matches, there will be `N+1` elements in the range.
 *  Unlike ::range_matches, this range can't be empty. It also can't report overlapping intervals.
 */
template <typename string_view_, template <typename> typename matcher_template_>
class range_rsplits {
    using string_view = string_view_;
    using matcher = matcher_template_<string_view>;

    string_view haystack_;
    matcher matcher_;

  public:
    range_rsplits(string_view haystack, matcher needle) : haystack_(haystack), matcher_(needle) {}

    class iterator {
        matcher matcher_;
        string_view remaining_;
        std::size_t length_within_remaining_;
        bool reached_tail_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view;
        using pointer = string_view;   // Needed for compatibility with STL container constructors.
        using reference = string_view; // Needed for compatibility with STL container constructors.

        iterator(string_view haystack, matcher matcher) noexcept : matcher_(matcher), remaining_(haystack) {
            auto position = matcher_(remaining_);
            length_within_remaining_ = position != string_view::npos
                                           ? remaining_.size() - position - matcher_.needle_length()
                                           : remaining_.size();
            reached_tail_ = false;
        }

        iterator(string_view haystack, matcher matcher, end_sentinel_type) noexcept
            : matcher_(matcher), remaining_(haystack), reached_tail_(true) {}

        value_type operator*() const noexcept {
            return remaining_.substr(remaining_.size() - length_within_remaining_);
        }

        iterator &operator++() noexcept {
            remaining_.remove_suffix(length_within_remaining_);
            reached_tail_ = remaining_.empty();
            remaining_.remove_suffix(matcher_.needle_length() * !reached_tail_);
            auto position = matcher_(remaining_);
            length_within_remaining_ = position != string_view::npos
                                           ? remaining_.size() - position - matcher_.needle_length()
                                           : remaining_.size();
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator const &other) const noexcept {
            return (remaining_.end() != other.remaining_.end()) || (reached_tail_ != other.reached_tail_);
        }
        bool operator==(iterator const &other) const noexcept {
            return (remaining_.end() == other.remaining_.end()) && (reached_tail_ == other.reached_tail_);
        }
        bool operator!=(end_sentinel_type) const noexcept { return !remaining_.empty() || !reached_tail_; }
        bool operator==(end_sentinel_type) const noexcept { return remaining_.empty() && reached_tail_; }
    };

    iterator begin() const noexcept { return iterator(haystack_, matcher_); }
    iterator end() const noexcept { return iterator({haystack_.begin(), 0}, matcher_, end_sentinel); }
    typename iterator::difference_type size() const noexcept { return std::distance(begin(), end()); }
    constexpr bool empty() const noexcept { return false; }

    /**
     *  @brief  Copies the matches into a container.
     */
    template <typename container_>
    void to(container_ &container) {
        for (auto match : *this) { container.push_back(match); }
    }

    /**
     *  @brief  Copies the matches into a consumed container, returning it at the end.
     */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto match : *this) { container.push_back(match); }
        return std::move(container);
    }
};

template <typename string>
range_matches<string, matcher_find> find_all(string h, string n, bool interleaving = true) noexcept {
    return {h, n};
}

template <typename string>
range_rmatches<string, matcher_rfind> rfind_all(string h, string n, bool interleaving = true) noexcept {
    return {h, n};
}

template <typename string>
range_matches<string, matcher_find_first_of> find_all_characters(string h, string n) noexcept {
    return {h, n};
}

template <typename string>
range_rmatches<string, matcher_find_last_of> rfind_all_characters(string h, string n) noexcept {
    return {h, n};
}

template <typename string>
range_matches<string, matcher_find_first_not_of> find_all_other_characters(string h, string n) noexcept {
    return {h, n};
}

template <typename string>
range_rmatches<string, matcher_find_last_not_of> rfind_all_other_characters(string h, string n) noexcept {
    return {h, n};
}

template <typename string>
range_splits<string, matcher_find> split_all(string h, string n, bool interleaving = true) noexcept {
    return {h, n};
}

template <typename string>
range_rmatches<string, matcher_rfind> rsplit_all(string h, string n, bool interleaving = true) noexcept {
    return {h, n};
}

template <typename string>
range_splits<string, matcher_find_first_of> split_all_characters(string h, string n) noexcept {
    return {h, n};
}

template <typename string>
range_rsplits<string, matcher_find_last_of> rsplit_all_characters(string h, string n) noexcept {
    return {h, n};
}

template <typename string>
range_splits<string, matcher_find_first_not_of> split_all_other_characters(string h, string n) noexcept {
    return {h, n};
}

template <typename string>
range_rsplits<string, matcher_find_last_not_of> rsplit_all_other_characters(string h, string n) noexcept {
    return {h, n};
}

/**
 *  @brief  A reverse iterator for mutable and immutable character buffers.
 *          Replaces `std::reverse_iterator` to avoid including `<iterator>`.
 */
template <typename value_type_>
class reversed_iterator_for {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = value_type_;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type_ *;
    using reference = value_type_ &;

    reversed_iterator_for(pointer ptr) noexcept : ptr_(ptr) {}
    reference operator*() const noexcept { return *ptr_; }

    bool operator==(reversed_iterator_for const &other) const noexcept { return ptr_ == other.ptr_; }
    bool operator!=(reversed_iterator_for const &other) const noexcept { return ptr_ != other.ptr_; }
    reference operator[](difference_type n) const noexcept { return *(*this + n); }
    reversed_iterator_for operator+(difference_type n) const noexcept { return reversed_iterator_for(ptr_ - n); }
    reversed_iterator_for operator-(difference_type n) const noexcept { return reversed_iterator_for(ptr_ + n); }
    difference_type operator-(reversed_iterator_for const &other) const noexcept { return other.ptr_ - ptr_; }

    reversed_iterator_for &operator++() noexcept {
        --ptr_;
        return *this;
    }

    reversed_iterator_for operator++(int) const noexcept {
        reversed_iterator_for temp = *this;
        --ptr_;
        return temp;
    }

    reversed_iterator_for &operator--() const noexcept {
        ++ptr_;
        return *this;
    }

    reversed_iterator_for operator--(int) const noexcept {
        reversed_iterator_for temp = *this;
        ++ptr_;
        return temp;
    }

  private:
    value_type_ *ptr_;
};

/**
 *  @brief  A string view class implementing with the superset of C++23 functionality
 *          with much faster SIMD-accelerated substring search and approximate matching.
 *          Unlike STL, never raises exceptions. Constructors are `constexpr` enabling `_sz` literals.
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
    using const_reverse_iterator = reversed_iterator_for<char const>;
    using reverse_iterator = const_reverse_iterator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using split_result = string_split_result<string_view>;

    /** @brief  Special value for missing matches. */
    static constexpr size_type npos = size_type(-1);

    constexpr string_view() noexcept : start_(nullptr), length_(0) {}
    constexpr string_view(const_pointer c_string) noexcept
        : start_(c_string), length_(null_terminated_length(c_string)) {}
    constexpr string_view(const_pointer c_string, size_type length) noexcept : start_(c_string), length_(length) {}
    constexpr string_view(string_view const &other) noexcept : start_(other.start_), length_(other.length_) {}
    constexpr string_view &operator=(string_view const &other) noexcept { return assign(other); }
    string_view(std::nullptr_t) = delete;

#if SZ_INCLUDE_STL_CONVERSIONS
#if __cplusplus >= 202002L
#define sz_constexpr_if20 constexpr
#else
#define sz_constexpr_if20 inline
#endif

    sz_constexpr_if20 string_view(std::string const &other) noexcept : string_view(other.data(), other.size()) {}
    sz_constexpr_if20 string_view(std::string_view const &other) noexcept : string_view(other.data(), other.size()) {}
    sz_constexpr_if20 string_view &operator=(std::string const &other) noexcept {
        return assign({other.data(), other.size()});
    }
    sz_constexpr_if20 string_view &operator=(std::string_view const &other) noexcept {
        return assign({other.data(), other.size()});
    }
#endif

    inline const_iterator begin() const noexcept { return const_iterator(start_); }
    inline const_iterator end() const noexcept { return const_iterator(start_ + length_); }
    inline const_iterator cbegin() const noexcept { return const_iterator(start_); }
    inline const_iterator cend() const noexcept { return const_iterator(start_ + length_); }
    inline const_reverse_iterator rbegin() const noexcept;
    inline const_reverse_iterator rend() const noexcept;
    inline const_reverse_iterator crbegin() const noexcept;
    inline const_reverse_iterator crend() const noexcept;

    inline const_reference operator[](size_type pos) const noexcept { return start_[pos]; }
    inline const_reference at(size_type pos) const noexcept { return start_[pos]; }
    inline const_reference front() const noexcept { return start_[0]; }
    inline const_reference back() const noexcept { return start_[length_ - 1]; }
    inline const_pointer data() const noexcept { return start_; }

    inline size_type size() const noexcept { return length_; }
    inline size_type length() const noexcept { return length_; }
    inline size_type max_size() const noexcept { return sz_size_max; }
    inline bool empty() const noexcept { return length_ == 0; }

    /** @brief Removes the first `n` characters from the view. The behavior is undefined if `n > size()`. */
    inline void remove_prefix(size_type n) noexcept { assert(n <= size()), start_ += n, length_ -= n; }

    /** @brief Removes the last `n` characters from the view. The behavior is undefined if `n > size()`. */
    inline void remove_suffix(size_type n) noexcept { assert(n <= size()), length_ -= n; }

    /** @brief Exchanges the view with that of the `other`. */
    inline void swap(string_view &other) noexcept {
        std::swap(start_, other.start_), std::swap(length_, other.length_);
    }

    /** @brief  Added for STL compatibility. */
    inline string_view substr() const noexcept { return *this; }

    /** @brief  Equivalent of `remove_prefix(pos)`. The behavior is undefined if `pos > size()`. */
    inline string_view substr(size_type pos) const noexcept { return string_view(start_ + pos, length_ - pos); }

    /** @brief  Returns a sub-view [pos, pos + rlen), where `rlen` is the smaller of count and `size() - pos`.
     *          Equivalent to `substr(pos).substr(0, count)` or combining `remove_prefix` and `remove_suffix`.
     *          The behavior is undefined if `pos > size()`.  */
    inline string_view substr(size_type pos, size_type count) const noexcept {
        return string_view(start_ + pos, sz_min_of_two(count, length_ - pos));
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    inline int compare(string_view other) const noexcept {
        return (int)sz_order(start_, length_, other.start_, other.length_);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    inline int compare(size_type pos1, size_type count1, string_view other) const noexcept {
        return substr(pos1, count1).compare(other);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    inline int compare(size_type pos1, size_type count1, string_view other, size_type pos2,
                       size_type count2) const noexcept {
        return substr(pos1, count1).compare(other.substr(pos2, count2));
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    inline int compare(const_pointer other) const noexcept { return compare(string_view(other)); }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    inline int compare(size_type pos1, size_type count1, const_pointer other) const noexcept {
        return substr(pos1, count1).compare(string_view(other));
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    inline int compare(size_type pos1, size_type count1, const_pointer other, size_type count2) const noexcept {
        return substr(pos1, count1).compare(string_view(other, count2));
    }

    /** @brief  Checks if the string is equal to the other string. */
    inline bool operator==(string_view other) const noexcept {
        return length_ == other.length_ && sz_equal(start_, other.start_, other.length_) == sz_true_k;
    }

#if __cplusplus >= 201402L
#define sz_deprecate_compare [[deprecated("Use the three-way comparison operator (<=>) in C++20 and later")]]
#else
#define sz_deprecate_compare
#endif

    /** @brief  Checks if the string is not equal to the other string. */
    sz_deprecate_compare inline bool operator!=(string_view other) const noexcept {
        return length_ != other.length_ || sz_equal(start_, other.start_, other.length_) == sz_false_k;
    }

    /** @brief  Checks if the string is lexicographically smaller than the other string. */
    sz_deprecate_compare inline bool operator<(string_view other) const noexcept { return compare(other) == sz_less_k; }

    /** @brief  Checks if the string is lexicographically equal or smaller than the other string. */
    sz_deprecate_compare inline bool operator<=(string_view other) const noexcept {
        return compare(other) != sz_greater_k;
    }

    /** @brief  Checks if the string is lexicographically greater than the other string. */
    sz_deprecate_compare inline bool operator>(string_view other) const noexcept {
        return compare(other) == sz_greater_k;
    }

    /** @brief  Checks if the string is lexicographically equal or greater than the other string. */
    sz_deprecate_compare inline bool operator>=(string_view other) const noexcept {
        return compare(other) != sz_less_k;
    }

#if __cplusplus >= 202002L

    /** @brief  Checks if the string is not equal to the other string. */
    inline int operator<=>(string_view other) const noexcept { return compare(other); }
#endif

    /** @brief  Checks if the string starts with the other string. */
    inline bool starts_with(string_view other) const noexcept {
        return length_ >= other.length_ && sz_equal(start_, other.start_, other.length_) == sz_true_k;
    }

    /** @brief  Checks if the string starts with the other string. */
    inline bool starts_with(const_pointer other) const noexcept {
        auto other_length = null_terminated_length(other);
        return length_ >= other_length && sz_equal(start_, other, other_length) == sz_true_k;
    }

    /** @brief  Checks if the string starts with the other character. */
    inline bool starts_with(value_type other) const noexcept { return length_ && start_[0] == other; }

    /** @brief  Checks if the string ends with the other string. */
    inline bool ends_with(string_view other) const noexcept {
        return length_ >= other.length_ &&
               sz_equal(start_ + length_ - other.length_, other.start_, other.length_) == sz_true_k;
    }

    /** @brief  Checks if the string ends with the other string. */
    inline bool ends_with(const_pointer other) const noexcept {
        auto other_length = null_terminated_length(other);
        return length_ >= other_length && sz_equal(start_ + length_ - other_length, other, other_length) == sz_true_k;
    }

    /** @brief  Checks if the string ends with the other character. */
    inline bool ends_with(value_type other) const noexcept { return length_ && start_[length_ - 1] == other; }

    /** @brief  Find the first occurrence of a substring. */
    inline size_type find(string_view other) const noexcept {
        auto ptr = sz_find(start_, length_, other.start_, other.length_);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurrence of a substring. The behavior is undefined if `pos > size()`. */
    inline size_type find(string_view other, size_type pos) const noexcept { return substr(pos).find(other); }

    /** @brief  Find the first occurrence of a character. */
    inline size_type find(value_type character) const noexcept {
        auto ptr = sz_find_byte(start_, length_, &character);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurrence of a character. The behavior is undefined if `pos > size()`. */
    inline size_type find(value_type character, size_type pos) const noexcept { return substr(pos).find(character); }

    /** @brief  Find the first occurrence of a substring. The behavior is undefined if `pos > size()`. */
    inline size_type find(const_pointer other, size_type pos, size_type count) const noexcept {
        return substr(pos).find(string_view(other, count));
    }

    /** @brief  Find the first occurrence of a substring. The behavior is undefined if `pos > size()`. */
    inline size_type find(const_pointer other, size_type pos = 0) const noexcept {
        return substr(pos).find(string_view(other));
    }

    /** @brief  Find the first occurrence of a substring. */
    inline size_type rfind(string_view other) const noexcept {
        auto ptr = sz_find_last(start_, length_, other.start_, other.length_);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurrence of a substring. The behavior is undefined if `pos > size()`. */
    inline size_type rfind(string_view other, size_type pos) const noexcept { return substr(pos).rfind(other); }

    /** @brief  Find the first occurrence of a character. */
    inline size_type rfind(value_type character) const noexcept {
        auto ptr = sz_find_last_byte(start_, length_, &character);
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurrence of a character. The behavior is undefined if `pos > size()`. */
    inline size_type rfind(value_type character, size_type pos) const noexcept { return substr(pos).rfind(character); }

    /** @brief  Find the first occurrence of a substring. The behavior is undefined if `pos > size()`. */
    inline size_type rfind(const_pointer other, size_type pos, size_type count) const noexcept {
        return substr(pos).rfind(string_view(other, count));
    }

    /** @brief  Find the first occurrence of a substring. The behavior is undefined if `pos > size()`. */
    inline size_type rfind(const_pointer other, size_type pos = 0) const noexcept {
        return substr(pos).rfind(string_view(other));
    }

    inline bool contains(string_view other) const noexcept { return find(other) != npos; }
    inline bool contains(value_type character) const noexcept { return find(character) != npos; }
    inline bool contains(const_pointer other) const noexcept { return find(other) != npos; }

    /** @brief  Find the first occurrence of a character from a set. */
    inline size_type find_first_of(string_view other) const noexcept { return find_first_of(other.as_set()); }

    /** @brief  Find the first occurrence of a character outside of the set. */
    inline size_type find_first_not_of(string_view other) const noexcept { return find_first_not_of(other.as_set()); }

    /** @brief  Find the last occurrence of a character from a set. */
    inline size_type find_last_of(string_view other) const noexcept { return find_last_of(other.as_set()); }

    /** @brief  Find the last occurrence of a character outside of the set. */
    inline size_type find_last_not_of(string_view other) const noexcept { return find_last_not_of(other.as_set()); }

    /** @brief  Find the first occurrence of a character from a set. */
    inline size_type find_first_of(character_set set) const noexcept {
        auto ptr = sz_find_from_set(start_, length_, &set.raw());
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the first occurrence of a character from a set. */
    inline size_type find(character_set set) const noexcept { return find_first_of(set); }

    /** @brief  Find the first occurrence of a character outside of the set. */
    inline size_type find_first_not_of(character_set set) const noexcept { return find_first_of(set.invert()); }

    /** @brief  Find the last occurrence of a character from a set. */
    inline size_type find_last_of(character_set set) const noexcept {
        auto ptr = sz_find_last_from_set(start_, length_, &set.raw());
        return ptr ? ptr - start_ : npos;
    }

    /** @brief  Find the last occurrence of a character from a set. */
    inline size_type rfind(character_set set) const noexcept { return find_last_of(set); }

    /** @brief  Find the last occurrence of a character outside of the set. */
    inline size_type find_last_not_of(character_set set) const noexcept { return find_last_of(set.invert()); }

    /** @brief  Find all occurrences of a given string.
     *  @param  interleave  If true, interleaving offsets are returned as well. */
    inline range_matches<string_view, matcher_find> find_all(string_view, bool interleave = true) const noexcept;

    /** @brief  Find all occurrences of a given string in @b reverse order.
     *  @param  interleave  If true, interleaving offsets are returned as well. */
    inline range_rmatches<string_view, matcher_rfind> rfind_all(string_view, bool interleave = true) const noexcept;

    /** @brief  Find all occurrences of given characters. */
    inline range_matches<string_view, matcher_find_first_of> find_all(character_set) const noexcept;

    /** @brief  Find all occurrences of given characters in @b reverse order. */
    inline range_rmatches<string_view, matcher_find_last_of> rfind_all(character_set) const noexcept;

    /** @brief  Split the string into three parts, before the match, the match itself, and after it. */
    inline split_result split(string_view pattern) const noexcept { return split_(pattern, pattern.length()); }

    /** @brief  Split the string into three parts, before the match, the match itself, and after it. */
    inline split_result split(character_set pattern) const noexcept { return split_(pattern, 1); }

    /** @brief  Split the string into three parts, before the @b last match, the last match itself, and after it. */
    inline split_result rsplit(string_view pattern) const noexcept { return split_(pattern, pattern.length()); }

    /** @brief  Split the string into three parts, before the @b last match, the last match itself, and after it. */
    inline split_result rsplit(character_set pattern) const noexcept { return split_(pattern, 1); }

    /** @brief  Find all occurrences of a given string.
     *  @param  interleave  If true, interleaving offsets are returned as well. */
    inline range_splits<string_view, matcher_find> split_all(string_view) const noexcept;

    /** @brief  Find all occurrences of a given string in @b reverse order.
     *  @param  interleave  If true, interleaving offsets are returned as well. */
    inline range_rsplits<string_view, matcher_rfind> rsplit_all(string_view) const noexcept;

    /** @brief  Find all occurrences of given characters. */
    inline range_splits<string_view, matcher_find_first_of> split_all(character_set) const noexcept;

    /** @brief  Find all occurrences of given characters in @b reverse order. */
    inline range_rsplits<string_view, matcher_find_last_of> rsplit_all(character_set) const noexcept;

    inline size_type copy(pointer destination, size_type count, size_type pos = 0) const noexcept = delete;

    inline size_type hash() const noexcept { return static_cast<size_type>(sz_hash(start_, length_)); }

    inline character_set as_set() const noexcept {
        character_set set;
        for (auto c : *this) set.add(c);
        return set;
    }

#if SZ_INCLUDE_STL_CONVERSIONS
    inline operator std::string() const { return {data(), size()}; }
    inline operator std::string_view() const noexcept { return {data(), size()}; }
#endif

  private:
    constexpr string_view &assign(string_view const &other) noexcept {
        start_ = other.start_;
        length_ = other.length_;
        return *this;
    }
    constexpr static size_type null_terminated_length(const_pointer s) noexcept {
        const_pointer p = s;
        while (*p) ++p;
        return p - s;
    }

    template <typename pattern_>
    split_result split_(pattern_ &&pattern, std::size_t pattern_length) const noexcept {
        size_type pos = find(pattern);
        if (pos == npos) return {substr(), string_view(), string_view()};
        return {substr(0, pos), substr(pos, pattern_length), substr(pos + pattern_length)};
    }

    template <typename pattern_>
    split_result rsplit_(pattern_ &&pattern, std::size_t pattern_length) const noexcept {
        size_type pos = rfind(pattern);
        if (pos == npos) return {substr(), string_view(), string_view()};
        return {substr(0, pos), substr(pos, pattern_length), substr(pos + pattern_length)};
    }
};

/**
 *  @brief  Memory-owning string class with a Small String Optimization.
 *
 *  @section Exceptions
 *
 *  Default constructor is `constexpr`. Move constructor and move assignment operator are `noexcept`.
 *  Copy constructor and copy assignment operator are not! They may throw `std::bad_alloc` if the memory
 *  allocation fails. Alternatively, if exceptions are disabled, they may call `std::terminate`.
 */
template <typename allocator_ = std::allocator<char>>
class basic_string {
    sz_string_t string_;

    using alloc_t = sz_memory_allocator_t;

    static sz_ptr_t call_allocate(sz_size_t n, void *allocator_state) noexcept {
        return reinterpret_cast<allocator_ *>(allocator_state)->allocate(n);
    }
    static void call_free(sz_ptr_t ptr, sz_size_t n, void *allocator_state) noexcept {
        return reinterpret_cast<allocator_ *>(allocator_state)->deallocate(reinterpret_cast<char *>(ptr), n);
    }
    template <typename allocator_callback>
    static bool with_alloc(allocator_callback &&callback) noexcept {
        allocator_ allocator;
        sz_memory_allocator_t alloc;
        alloc.allocate = &call_allocate;
        alloc.free = &call_free;
        alloc.handle = &allocator;
        return callback(alloc) == sz_true_k;
    }

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
    using const_reverse_iterator = reversed_iterator_for<char const>;
    using reverse_iterator = const_reverse_iterator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using allocator_type = allocator_;

    /** @brief  Special value for missing matches. */
    static constexpr size_type npos = size_type(-1);

    constexpr basic_string() noexcept {
        // Instead of relying on the `sz_string_init`, we have to reimplement it to support `constexpr`.
        string_.on_stack.start = &string_.on_stack.chars[0];
        string_.on_stack.chars[0] = 0;
        string_.on_stack.length = 0;
    }

    ~basic_string() noexcept {
        with_alloc([&](alloc_t &alloc) {
            sz_string_free(&string_, &alloc);
            return sz_true_k;
        });
    }

    basic_string(basic_string &&other) noexcept : string_(other.string_) { sz_string_init(&other.string_); }
    basic_string &operator=(basic_string &&other) noexcept {
        string_ = other.string_;
        sz_string_init(&other.string_);
        return *this;
    }

    basic_string(basic_string const &other) noexcept(false) : basic_string() { assign(other); }
    basic_string &operator=(basic_string const &other) noexcept(false) { return assign(other); }
    basic_string(string_view view) noexcept(false) : basic_string() { assign(view); }
    basic_string &operator=(string_view view) noexcept(false) { return assign(view); }

    operator string_view() const noexcept {
        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_size_t string_space;
        sz_bool_t string_is_on_heap;
        sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_on_heap);
        return {string_start, string_length};
    }

    basic_string &assign(string_view other) noexcept(false) {
        if (!try_assign(other)) throw std::bad_alloc();
        return *this;
    }

    basic_string &append(string_view other) noexcept(false) {
        if (!try_append(other)) throw std::bad_alloc();
        return *this;
    }

    void push_back(char c) noexcept(false) {
        if (!try_push_back(c)) throw std::bad_alloc();
    }

    void clear() noexcept { sz_string_erase(&string_, 0, sz_size_max); }

    basic_string &erase(std::size_t pos = 0, std::size_t count = sz_size_max) noexcept {
        sz_string_erase(&string_, pos, count);
        return *this;
    }

    bool try_assign(string_view other) noexcept {
        clear();
        return try_append(other);
    }

    bool try_push_back(char c) noexcept {
        return with_alloc([&](alloc_t &alloc) { return sz_string_append(&string_, &c, 1, &alloc); });
    }

    bool try_append(char const *str, std::size_t length) noexcept {
        return with_alloc([&](alloc_t &alloc) { return sz_string_append(&string_, str, length, &alloc); });
    }

    bool try_append(string_view str) noexcept { return try_append(str.data(), str.size()); }

    size_type edit_distance(string_view other, size_type bound = npos) const noexcept {
        size_type distance;
        with_alloc([&](alloc_t &alloc) {
            distance = sz_edit_distance(string_.on_stack.start, string_.on_stack.length, other.data(), other.size(),
                                        bound, &alloc);
            return sz_true_k;
        });
        return distance;
    }
};

using string = basic_string<>;

static_assert(sizeof(string) == 4 * sizeof(void *), "String size must be 4 pointers.");

namespace literals {
constexpr string_view operator""_sz(char const *str, std::size_t length) noexcept { return {str, length}; }
} // namespace literals

template <>
struct matcher_find_first_of<string_view> {
    using size_type = typename string_view::size_type;
    character_set needles_set_;
    matcher_find_first_of() noexcept {}
    matcher_find_first_of(character_set set) noexcept : needles_set_(set) {}
    matcher_find_first_of(string_view needle) noexcept : needles_set_(needle.as_set()) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(string_view haystack) const noexcept { return haystack.find_first_of(needles_set_); }
};

template <>
struct matcher_find_last_of<string_view> {
    using size_type = typename string_view::size_type;
    character_set needles_set_;
    matcher_find_last_of() noexcept {}
    matcher_find_last_of(character_set set) noexcept : needles_set_(set) {}
    matcher_find_last_of(string_view needle) noexcept : needles_set_(needle.as_set()) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(string_view haystack) const noexcept { return haystack.find_last_of(needles_set_); }
};

template <>
struct matcher_find_first_not_of<string_view> {
    using size_type = typename string_view::size_type;
    character_set needles_set_;
    matcher_find_first_not_of() noexcept {}
    matcher_find_first_not_of(character_set set) noexcept : needles_set_(set) {}
    matcher_find_first_not_of(string_view needle) noexcept : needles_set_(needle.as_set()) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(string_view haystack) const noexcept { return haystack.find_first_not_of(needles_set_); }
};

template <>
struct matcher_find_last_not_of<string_view> {
    using size_type = typename string_view::size_type;
    character_set needles_set_;
    matcher_find_last_not_of() noexcept {}
    matcher_find_last_not_of(character_set set) noexcept : needles_set_(set) {}
    matcher_find_last_not_of(string_view needle) noexcept : needles_set_(needle.as_set()) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(string_view haystack) const noexcept { return haystack.find_last_not_of(needles_set_); }
};

inline range_matches<string_view, matcher_find> string_view::find_all(string_view n, bool i) const noexcept {
    return {*this, {n, i}};
}

inline range_rmatches<string_view, matcher_rfind> string_view::rfind_all(string_view n, bool i) const noexcept {
    return {*this, {n, i}};
}

inline range_matches<string_view, matcher_find_first_of> string_view::find_all(character_set n) const noexcept {
    return {*this, {n}};
}

inline range_rmatches<string_view, matcher_find_last_of> string_view::rfind_all(character_set n) const noexcept {
    return {*this, {n}};
}

inline range_splits<string_view, matcher_find> string_view::split_all(string_view n) const noexcept {
    return {*this, {n}};
}

inline range_rsplits<string_view, matcher_rfind> string_view::rsplit_all(string_view n) const noexcept {
    return {*this, {n}};
}

inline range_splits<string_view, matcher_find_first_of> string_view::split_all(character_set n) const noexcept {
    return {*this, {n}};
}

inline range_rsplits<string_view, matcher_find_last_of> string_view::rsplit_all(character_set n) const noexcept {
    return {*this, {n}};
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_HPP_
