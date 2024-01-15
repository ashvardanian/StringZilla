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

/**
 *  @brief  When set to 1, the library will include the C++ STL headers and implement
 *          automatic conversion from and to `std::stirng_view` and `std::basic_string<any_allocator>`.
 */
#ifndef SZ_INCLUDE_STL_CONVERSIONS
#define SZ_INCLUDE_STL_CONVERSIONS (1)
#endif

/**
 *  @brief  When set to 1, the strings `+` will return an expression template rather than a temporary string.
 *          This will improve performance, but may break some STL-specific code, so it's disabled by default.
 */
#ifndef SZ_LAZY_CONCAT
#define SZ_LAZY_CONCAT (0)
#endif

/**
 *  @brief  When set to 1, the library will change `substr` and several other member methods of `string`
 *          to return a view of its slice, rather than a copy, if the lifetime of the object is guaranteed.
 *          This will improve performance, but may break some STL-specific code, so it's disabled by default.
 */
#ifndef SZ_PREFER_VIEWS
#define SZ_PREFER_VIEWS (0)
#endif

/*  We need to detect the version of the C++ language we are compiled with.
 *  This will affect recent features like `operator<=>` and tests against STL.
 */
#define SZ_DETECT_CPP_23 (__cplusplus >= 202101L)
#define SZ_DETECT_CPP20 (__cplusplus >= 202002L)
#define SZ_DETECT_CPP_17 (__cplusplus >= 201703L)
#define SZ_DETECT_CPP14 (__cplusplus >= 201402L)
#define SZ_DETECT_CPP_11 (__cplusplus >= 201103L)
#define SZ_DETECT_CPP_98 (__cplusplus >= 199711L)

/**
 *  @brief  The `constexpr` keyword has different applicability scope in different C++ versions.
 *          Useful for STL conversion operators, as several `std::string` members are `constexpr` in C++20.
 */
#if SZ_DETECT_CPP20
#define sz_constexpr_if_cpp20 constexpr
#else
#define sz_constexpr_if_cpp20
#endif

#if SZ_INCLUDE_STL_CONVERSIONS
#include <string>
#if SZ_DETECT_CPP_17 && __cpp_lib_string_view
#include <string_view>
#endif
#endif

#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <iosfwd>  // `std::basic_ostream`
#include <utility> // `std::swap`

#include <stringzilla/stringzilla.h>

namespace ashvardanian {
namespace stringzilla {

template <typename>
class basic_char_set;
template <typename>
class basic_string_slice;
template <typename, typename>
class basic_string;

using string_span = basic_string_slice<char>;
using string_view = basic_string_slice<char const>;

#pragma region Character Sets

/**
 *  @brief  The concatenation of the `ascii_lowercase` and `ascii_uppercase`. This value is not locale-dependent.
 *          https://docs.python.org/3/library/string.html#string.ascii_letters
 */
inline static constexpr char ascii_letters[52] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                                                  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                                                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};

/**
 *  @brief  The lowercase letters "abcdefghijklmnopqrstuvwxyz". This value is not locale-dependent.
 *          https://docs.python.org/3/library/string.html#string.ascii_lowercase
 */
inline static constexpr char ascii_lowercase[26] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                                                    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

/**
 *  @brief  The uppercase letters "ABCDEFGHIJKLMNOPQRSTUVWXYZ". This value is not locale-dependent.
 *          https://docs.python.org/3/library/string.html#string.ascii_uppercase
 */
inline static constexpr char ascii_uppercase[26] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};

/**
 *  @brief  ASCII characters which are considered printable.
 *          A combination of `digits`, `ascii_letters`, `punctuation`, and `whitespace`.
 *          https://docs.python.org/3/library/string.html#string.printable
 */
inline static constexpr char ascii_printables[100] = {
    '0', '1', '2', '3', '4', '5',  '6', '7', '8',  '9', 'a', 'b', 'c', 'd', 'e', 'f',  'g',  'h',  'i',  'j',
    'k', 'l', 'm', 'n', 'o', 'p',  'q', 'r', 's',  't', 'u', 'v', 'w', 'x', 'y', 'z',  'A',  'B',  'C',  'D',
    'E', 'F', 'G', 'H', 'I', 'J',  'K', 'L', 'M',  'N', 'O', 'P', 'Q', 'R', 'S', 'T',  'U',  'V',  'W',  'X',
    'Y', 'Z', '!', '"', '#', '$',  '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',  '/',  ':',  ';',  '<',
    '=', '>', '?', '@', '[', '\\', ']', '^', '_',  '`', '{', '|', '}', '~', ' ', '\t', '\n', '\r', '\f', '\v'};

/**
 *  @brief  Non-printable ASCII control characters.
 *          Includes all codes from 0 to 31 and 127.
 */
inline static constexpr char ascii_controls[33] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 127};

/**
 *  @brief  The digits "0123456789".
 *          https://docs.python.org/3/library/string.html#string.digits
 */
inline static constexpr char digits[10] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

/**
 *  @brief  The letters "0123456789abcdefABCDEF".
 *          https://docs.python.org/3/library/string.html#string.hexdigits
 */
inline static constexpr char hexdigits[22] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', //
                                              'a', 'b', 'c', 'd', 'e', 'f', 'A', 'B', 'C', 'D', 'E', 'F'};

/**
 *  @brief  The letters "01234567".
 *          https://docs.python.org/3/library/string.html#string.octdigits
 */
inline static constexpr char octdigits[8] = {'0', '1', '2', '3', '4', '5', '6', '7'};

/**
 *  @brief  ASCII characters considered punctuation characters in the C locale:
 *          !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.
 *          https://docs.python.org/3/library/string.html#string.punctuation
 */
inline static constexpr char punctuation[32] = { //
    '!', '"', '#', '$', '%', '&', '\'', '(',  ')', '*', '+', ',', '-', '.', '/', ':',
    ';', '<', '=', '>', '?', '@', '[',  '\\', ']', '^', '_', '`', '{', '|', '}', '~'};

/**
 *  @brief  ASCII characters that are considered whitespace.
 *          This includes space, tab, linefeed, return, formfeed, and vertical tab.
 *          https://docs.python.org/3/library/string.html#string.whitespace
 */
inline static constexpr char whitespaces[6] = {' ', '\t', '\n', '\r', '\f', '\v'};

/**
 *  @brief  ASCII characters that are considered line delimiters.
 *          https://docs.python.org/3/library/stdtypes.html#str.splitlines
 */
inline static constexpr char newlines[8] = {'\n', '\r', '\f', '\v', '\x1C', '\x1D', '\x1E', '\x85'};

/**
 *  @brief  ASCII characters forming the BASE64 encoding alphabet.
 */
inline static constexpr char base64[64] = { //
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

/**
 *  @brief  A set of characters represented as a bitset with 256 slots.
 */
template <typename char_type_ = char>
class basic_char_set {
    sz_u8_set_t bitset_;

  public:
    using char_type = char_type_;

    basic_char_set() noexcept {
        // ! Instead of relying on the `sz_u8_set_init`, we have to reimplement it to support `constexpr`.
        bitset_._u64s[0] = 0, bitset_._u64s[1] = 0, bitset_._u64s[2] = 0, bitset_._u64s[3] = 0;
    }
    explicit basic_char_set(std::initializer_list<char_type> chars) noexcept : basic_char_set() {
        // ! Instead of relying on the `sz_u8_set_add(&bitset_, c)`, we have to reimplement it to support `constexpr`.
        for (auto c : chars) bitset_._u64s[sz_bitcast(sz_u8_t, c) >> 6] |= (1ull << (sz_bitcast(sz_u8_t, c) & 63u));
    }
    template <std::size_t count_characters>
    explicit basic_char_set(char_type const (&chars)[count_characters]) noexcept : basic_char_set() {
        static_assert(count_characters > 0, "Character array cannot be empty");
        for (std::size_t i = 0; i < count_characters - 1; ++i) { // count_characters - 1 to exclude the null terminator
            char_type c = chars[i];
            bitset_._u64s[sz_bitcast(sz_u8_t, c) >> 6] |= (1ull << (sz_bitcast(sz_u8_t, c) & 63u));
        }
    }

    basic_char_set(basic_char_set const &other) noexcept : bitset_(other.bitset_) {}
    basic_char_set &operator=(basic_char_set const &other) noexcept {
        bitset_ = other.bitset_;
        return *this;
    }

    basic_char_set operator|(basic_char_set other) const noexcept {
        basic_char_set result = *this;
        result.bitset_._u64s[0] |= other.bitset_._u64s[0], result.bitset_._u64s[1] |= other.bitset_._u64s[1],
            result.bitset_._u64s[2] |= other.bitset_._u64s[2], result.bitset_._u64s[3] |= other.bitset_._u64s[3];
        return *this;
    }

    inline basic_char_set &add(char_type c) noexcept {
        sz_u8_set_add(&bitset_, sz_bitcast(sz_u8_t, c));
        return *this;
    }
    inline sz_u8_set_t &raw() noexcept { return bitset_; }
    inline sz_u8_set_t const &raw() const noexcept { return bitset_; }
    inline bool contains(char_type c) const noexcept { return sz_u8_set_contains(&bitset_, sz_bitcast(sz_u8_t, c)); }
    inline basic_char_set inverted() const noexcept {
        basic_char_set result = *this;
        sz_u8_set_invert(&result.bitset_);
        return result;
    }
};

using char_set = basic_char_set<char>;

inline static char_set const ascii_letters_set {ascii_letters};
inline static char_set const ascii_lowercase_set {ascii_lowercase};
inline static char_set const ascii_uppercase_set {ascii_uppercase};
inline static char_set const ascii_printables_set {ascii_printables};
inline static char_set const ascii_controls_set {ascii_controls};
inline static char_set const digits_set {digits};
inline static char_set const hexdigits_set {hexdigits};
inline static char_set const octdigits_set {octdigits};
inline static char_set const punctuation_set {punctuation};
inline static char_set const whitespaces_set {whitespaces};
inline static char_set const newlines_set {newlines};
inline static char_set const base64_set {base64};

#pragma endregion

#pragma region Ranges of Search Matches

struct end_sentinel_type {};
inline static constexpr end_sentinel_type end_sentinel;

struct include_overlaps_type {};
inline static constexpr include_overlaps_type include_overlaps;

struct exclude_overlaps_type {};
inline static constexpr exclude_overlaps_type exclude_overlaps;

/**
 *  @brief  Zero-cost wrapper around the `.find` member function of string-like classes.
 */
template <typename string_type_, typename overlaps_type = include_overlaps_type>
struct matcher_find {
    using size_type = typename string_type_::size_type;
    string_type_ needle_;

    matcher_find(string_type_ needle = {}) noexcept : needle_(needle) {}
    size_type needle_length() const noexcept { return needle_.length(); }
    size_type operator()(string_type_ haystack) const noexcept { return haystack.find(needle_); }
    size_type skip_length() const noexcept {
        // TODO: Apply Galil rule to match repetitive patterns in strictly linear time.
        return std::is_same<overlaps_type, include_overlaps_type>() ? 1 : needle_.length();
    }
};

/**
 *  @brief  Zero-cost wrapper around the `.rfind` member function of string-like classes.
 */
template <typename string_type_, typename overlaps_type = include_overlaps_type>
struct matcher_rfind {
    using size_type = typename string_type_::size_type;
    string_type_ needle_;

    matcher_rfind(string_type_ needle = {}) noexcept : needle_(needle) {}
    size_type needle_length() const noexcept { return needle_.length(); }
    size_type operator()(string_type_ haystack) const noexcept { return haystack.rfind(needle_); }
    size_type skip_length() const noexcept {
        // TODO: Apply Galil rule to match repetitive patterns in strictly linear time.
        return std::is_same<overlaps_type, include_overlaps_type>() ? 1 : needle_.length();
    }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_first_of` member function of string-like classes.
 */
template <typename haystack_type, typename needles_type = haystack_type>
struct matcher_find_first_of {
    using size_type = typename haystack_type::size_type;
    needles_type needles_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(haystack_type haystack) const noexcept { return haystack.find_first_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_last_of` member function of string-like classes.
 */
template <typename haystack_type, typename needles_type = haystack_type>
struct matcher_find_last_of {
    using size_type = typename haystack_type::size_type;
    needles_type needles_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(haystack_type haystack) const noexcept { return haystack.find_last_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_first_not_of` member function of string-like classes.
 */
template <typename haystack_type, typename needles_type = haystack_type>
struct matcher_find_first_not_of {
    using size_type = typename haystack_type::size_type;
    needles_type needles_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(haystack_type haystack) const noexcept { return haystack.find_first_not_of(needles_); }
};

/**
 *  @brief  Zero-cost wrapper around the `.find_last_not_of` member function of string-like classes.
 */
template <typename haystack_type, typename needles_type = haystack_type>
struct matcher_find_last_not_of {
    using size_type = typename haystack_type::size_type;
    needles_type needles_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(haystack_type haystack) const noexcept { return haystack.find_last_not_of(needles_); }
};

/**
 *  @brief  A range of string slices representing the matches of a substring search.
 *          Compatible with C++23 ranges, C++11 string views, and of course, StringZilla.
 *          Similar to a pair of `boost::algorithm::find_iterator`.
 */
template <typename string_type_, typename matcher_type_>
class range_matches {
  public:
    using string_type = string_type_;
    using matcher_type = matcher_type_;

  private:
    matcher_type matcher_;
    string_type haystack_;

  public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = string_type;
    using pointer = string_type;   // Needed for compatibility with STL container constructors.
    using reference = string_type; // Needed for compatibility with STL container constructors.

    range_matches(string_type haystack, matcher_type needle) noexcept : matcher_(needle), haystack_(haystack) {}

    class iterator {
        matcher_type matcher_;
        string_type remaining_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_type;
        using pointer = string_type;   // Needed for compatibility with STL container constructors.
        using reference = string_type; // Needed for compatibility with STL container constructors.

        iterator(string_type haystack, matcher_type matcher) noexcept : matcher_(matcher), remaining_(haystack) {
            auto position = matcher_(remaining_);
            remaining_.remove_prefix(position != string_type::npos ? position : remaining_.size());
        }

        pointer operator->() const noexcept = delete;
        value_type operator*() const noexcept { return remaining_.substr(0, matcher_.needle_length()); }

        iterator &operator++() noexcept {
            remaining_.remove_prefix(matcher_.skip_length());
            auto position = matcher_(remaining_);
            remaining_.remove_prefix(position != string_type::npos ? position : remaining_.size());
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
    size_type size() const noexcept { return static_cast<size_type>(ssize()); }
    difference_type ssize() const noexcept { return std::distance(begin(), end()); }
    bool empty() const noexcept { return begin() == end_sentinel; }
    bool include_overlaps() const noexcept { return matcher_.skip_length() < matcher_.needle_length(); }

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
 *          Similar to a pair of `boost::algorithm::find_iterator`.
 */
template <typename string_type_, typename matcher_type_>
class range_rmatches {
  public:
    using string_type = string_type_;
    using matcher_type = matcher_type_;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = string_type;
    using pointer = string_type;   // Needed for compatibility with STL container constructors.
    using reference = string_type; // Needed for compatibility with STL container constructors.

  private:
    matcher_type matcher_;
    string_type haystack_;

  public:
    range_rmatches(string_type haystack, matcher_type needle) : matcher_(needle), haystack_(haystack) {}

    class iterator {
        matcher_type matcher_;
        string_type remaining_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_type;
        using pointer = string_type;   // Needed for compatibility with STL container constructors.
        using reference = string_type; // Needed for compatibility with STL container constructors.

        iterator(string_type haystack, matcher_type matcher) noexcept : matcher_(matcher), remaining_(haystack) {
            auto position = matcher_(remaining_);
            remaining_.remove_suffix(position != string_type::npos
                                         ? remaining_.size() - position - matcher_.needle_length()
                                         : remaining_.size());
        }

        pointer operator->() const noexcept = delete;
        value_type operator*() const noexcept {
            return remaining_.substr(remaining_.size() - matcher_.needle_length());
        }

        iterator &operator++() noexcept {
            remaining_.remove_suffix(matcher_.skip_length());
            auto position = matcher_(remaining_);
            remaining_.remove_suffix(position != string_type::npos
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
    size_type size() const noexcept { return static_cast<size_type>(ssize()); }
    difference_type ssize() const noexcept { return std::distance(begin(), end()); }
    bool empty() const noexcept { return begin() == end_sentinel; }
    bool include_overlaps() const noexcept { return matcher_.skip_length() < matcher_.needle_length(); }

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
 *          Similar to a pair of `boost::algorithm::split_iterator`.
 *
 *  In some sense, represents the inverse operation to `range_matches`, as it reports not the search matches
 *  but the data between them. Meaning that for `N` search matches, there will be `N+1` elements in the range.
 *  Unlike ::range_matches, this range can't be empty. It also can't report overlapping intervals.
 */
template <typename string_type_, typename matcher_type_>
class range_splits {
  public:
    using string_type = string_type_;
    using matcher_type = matcher_type_;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = string_type;
    using pointer = string_type;   // Needed for compatibility with STL container constructors.
    using reference = string_type; // Needed for compatibility with STL container constructors.

  private:
    matcher_type matcher_;
    string_type haystack_;

  public:
    range_splits(string_type haystack, matcher_type needle) noexcept : matcher_(needle), haystack_(haystack) {}

    class iterator {
        matcher_type matcher_;
        string_type remaining_;
        std::size_t length_within_remaining_;
        bool reached_tail_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_type;
        using pointer = string_type;   // Needed for compatibility with STL container constructors.
        using reference = string_type; // Needed for compatibility with STL container constructors.

        iterator(string_type haystack, matcher_type matcher) noexcept : matcher_(matcher), remaining_(haystack) {
            auto position = matcher_(remaining_);
            length_within_remaining_ = position != string_type::npos ? position : remaining_.size();
            reached_tail_ = false;
        }

        iterator(string_type haystack, matcher_type matcher, end_sentinel_type) noexcept
            : matcher_(matcher), remaining_(haystack), reached_tail_(true) {}

        pointer operator->() const noexcept = delete;
        value_type operator*() const noexcept { return remaining_.substr(0, length_within_remaining_); }

        iterator &operator++() noexcept {
            remaining_.remove_prefix(length_within_remaining_);
            reached_tail_ = remaining_.empty();
            remaining_.remove_prefix(matcher_.needle_length() * !reached_tail_);
            auto position = matcher_(remaining_);
            length_within_remaining_ = position != string_type::npos ? position : remaining_.size();
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
        bool is_last() const noexcept { return remaining_.size() == length_within_remaining_; }
    };

    iterator begin() const noexcept { return iterator(haystack_, matcher_); }
    iterator end() const noexcept { return iterator({haystack_.end(), 0}, matcher_, end_sentinel); }
    size_type size() const noexcept { return static_cast<size_type>(ssize()); }
    difference_type ssize() const noexcept { return std::distance(begin(), end()); }
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
 *          Similar to a pair of `boost::algorithm::split_iterator`.
 *
 *  In some sense, represents the inverse operation to `range_matches`, as it reports not the search matches
 *  but the data between them. Meaning that for `N` search matches, there will be `N+1` elements in the range.
 *  Unlike ::range_matches, this range can't be empty. It also can't report overlapping intervals.
 */
template <typename string_type_, typename matcher_type_>
class range_rsplits {
  public:
    using string_type = string_type_;
    using matcher_type = matcher_type_;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = string_type;
    using pointer = string_type;   // Needed for compatibility with STL container constructors.
    using reference = string_type; // Needed for compatibility with STL container constructors.

  private:
    matcher_type matcher_;
    string_type haystack_;

  public:
    range_rsplits(string_type haystack, matcher_type needle) noexcept : matcher_(needle), haystack_(haystack) {}

    class iterator {
        matcher_type matcher_;
        string_type remaining_;
        std::size_t length_within_remaining_;
        bool reached_tail_;

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_type;
        using pointer = string_type;   // Needed for compatibility with STL container constructors.
        using reference = string_type; // Needed for compatibility with STL container constructors.

        iterator(string_type haystack, matcher_type matcher) noexcept : matcher_(matcher), remaining_(haystack) {
            auto position = matcher_(remaining_);
            length_within_remaining_ = position != string_type::npos
                                           ? remaining_.size() - position - matcher_.needle_length()
                                           : remaining_.size();
            reached_tail_ = false;
        }

        iterator(string_type haystack, matcher_type matcher, end_sentinel_type) noexcept
            : matcher_(matcher), remaining_(haystack), reached_tail_(true) {}

        pointer operator->() const noexcept = delete;
        value_type operator*() const noexcept {
            return remaining_.substr(remaining_.size() - length_within_remaining_);
        }

        iterator &operator++() noexcept {
            remaining_.remove_suffix(length_within_remaining_);
            reached_tail_ = remaining_.empty();
            remaining_.remove_suffix(matcher_.needle_length() * !reached_tail_);
            auto position = matcher_(remaining_);
            length_within_remaining_ = position != string_type::npos
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
        bool is_last() const noexcept { return remaining_.size() == length_within_remaining_; }
    };

    iterator begin() const noexcept { return iterator(haystack_, matcher_); }
    iterator end() const noexcept { return iterator({haystack_.begin(), 0}, matcher_, end_sentinel); }
    size_type size() const noexcept { return static_cast<size_type>(ssize()); }
    difference_type ssize() const noexcept { return std::distance(begin(), end()); }
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
 *  @brief  Find all potentially @b overlapping inclusions of a needle substring.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_matches<string, matcher_find<string, include_overlaps_type>> find_all(string const &h, string const &n,
                                                                            include_overlaps_type = {}) noexcept {
    return {h, n};
}

/**
 *  @brief  Find all potentially @b overlapping inclusions of a needle substring in @b reverse order.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_rmatches<string, matcher_rfind<string, include_overlaps_type>> rfind_all(string const &h, string const &n,
                                                                               include_overlaps_type = {}) noexcept {
    return {h, n};
}

/**
 *  @brief  Find all @b non-overlapping inclusions of a needle substring.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_matches<string, matcher_find<string, exclude_overlaps_type>> find_all(string const &h, string const &n,
                                                                            exclude_overlaps_type) noexcept {
    return {h, n};
}

/**
 *  @brief  Find all @b non-overlapping inclusions of a needle substring in @b reverse order.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_rmatches<string, matcher_rfind<string, exclude_overlaps_type>> rfind_all(string const &h, string const &n,
                                                                               exclude_overlaps_type) noexcept {
    return {h, n};
}

/**
 *  @brief  Find all inclusions of characters from the second string.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_matches<string, matcher_find_first_of<string>> find_all_characters(string const &h, string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Find all inclusions of characters from the second string in @b reverse order.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_rmatches<string, matcher_find_last_of<string>> rfind_all_characters(string const &h, string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Find all characters except the ones in the second string.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_matches<string, matcher_find_first_not_of<string>> find_all_other_characters(string const &h,
                                                                                   string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Find all characters except the ones in the second string in @b reverse order.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_rmatches<string, matcher_find_last_not_of<string>> rfind_all_other_characters(string const &h,
                                                                                    string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Splits a string around every @b non-overlapping inclusion of the second string.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_splits<string, matcher_find<string, exclude_overlaps_type>> split(string const &h, string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Splits a string around every @b non-overlapping inclusion of the second string in @b reverse order.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_rmatches<string, matcher_rfind<string, exclude_overlaps_type>> rsplit(string const &h, string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Splits a string around every character from the second string.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_splits<string, matcher_find_first_of<string>> split_characters(string const &h, string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Splits a string around every character from the second string in @b reverse order.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_rsplits<string, matcher_find_last_of<string>> rsplit_characters(string const &h, string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Splits a string around every character except the ones from the second string.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_splits<string, matcher_find_first_not_of<string>> split_other_characters(string const &h,
                                                                               string const &n) noexcept {
    return {h, n};
}

/**
 *  @brief  Splits a string around every character except the ones from the second string in @b reverse order.
 *  @tparam string  A string-like type, ideally a view, like StringZilla or STL `string_view`.
 */
template <typename string>
range_rsplits<string, matcher_find_last_not_of<string>> rsplit_other_characters(string const &h,
                                                                                string const &n) noexcept {
    return {h, n};
}

/**  @brief  Helper function using `std::advance` iterator and return it back. */
template <typename iterator_type, typename distance_type>
iterator_type advanced(iterator_type &&it, distance_type n) {
    std::advance(it, n);
    return it;
}

/**  @brief  Helper function using `range_length` to compute the unsigned distance. */
template <typename iterator_type>
std::size_t range_length(iterator_type first, iterator_type last) {
    return static_cast<std::size_t>(std::distance(first, last));
}

#pragma endregion

#pragma region Helper Template Classes

/**
 *  @brief  A result of split a string once, containing the string slice ::before,
 *          the ::match itself, and the slice ::after.
 */
template <typename string_>
struct string_partition_result {
    string_ before;
    string_ match;
    string_ after;
};

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
 *  @brief  An "expression template" for lazy concatenation of strings using the `operator|`.
 *
 *  TODO: Ensure eqnership passing and move semantics are preserved.
 */
template <typename first_type, typename second_type>
struct concatenation {

    using value_type = typename first_type::value_type;
    using pointer = value_type *;
    using const_pointer = value_type const *;
    using size_type = typename first_type::size_type;
    using difference_type = typename first_type::difference_type;

    first_type first;
    second_type second;

    std::size_t size() const noexcept { return first.size() + second.size(); }
    std::size_t length() const noexcept { return first.size() + second.size(); }

    size_type copy(pointer destination) const noexcept {
        first.copy(destination);
        second.copy(destination + first.length());
        return first.length() + second.length();
    }

    size_type copy(pointer destination, size_type length) const noexcept {
        auto first_length = std::min(first.length(), length);
        auto second_length = std::min(second.length(), length - first_length);
        first.copy(destination, first_length);
        second.copy(destination + first_length, second_length);
        return first_length + second_length;
    }

    template <typename last_type>
    concatenation<concatenation<first_type, second_type>, last_type> operator|(last_type &&last) const {
        return {*this, last};
    }
};

#pragma endregion

#pragma region String Views/Spans

/**
 *  @brief  A string slice (view/span) class implementing a superset of C++23 functionality
 *          with much faster SIMD-accelerated substring search and approximate matching.
 *          Constructors are `constexpr` enabling `_sz` literals.
 *
 *  @tparam char_type_  The character type, usually `char const` or `char`. Must be a single byte long.
 */
template <typename char_type_>
class basic_string_slice {

    static_assert(sizeof(char_type_) == 1, "Characters must be a single byte long");
    static_assert(std::is_reference<char_type_>::value == false, "Characters can't be references");

    using char_type = char_type_;
    using mutable_char_type = typename std::remove_const<char_type_>::type;
    using immutable_char_type = typename std::add_const<char_type_>::type;

    char_type *start_;
    std::size_t length_;

  public:
    // STL compatibility
    using traits_type = std::char_traits<char_type_>;
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

    // Non-STL type definitions
    using string_slice = basic_string_slice<char_type>;
    using string_span = basic_string_slice<mutable_char_type>;
    using string_view = basic_string_slice<immutable_char_type>;
    using partition_type = string_partition_result<string_slice>;

    /**  @brief  Special value for missing matches.
     *          We take the largest 63-bit unsigned integer.
     */
    inline static constexpr size_type npos = 0x7FFFFFFFFFFFFFFFull;

#pragma region Constructors and STL Utilities

    constexpr basic_string_slice() noexcept : start_(nullptr), length_(0) {}
    constexpr basic_string_slice(pointer c_string) noexcept
        : start_(c_string), length_(null_terminated_length(c_string)) {}
    constexpr basic_string_slice(pointer c_string, size_type length) noexcept : start_(c_string), length_(length) {}

    constexpr basic_string_slice(basic_string_slice const &other) noexcept = default;
    constexpr basic_string_slice &operator=(basic_string_slice const &other) noexcept = default;

    basic_string_slice(std::nullptr_t) = delete;

    /**  @brief Exchanges the view with that of the `other`. */
    void swap(string_slice &other) noexcept { std::swap(start_, other.start_), std::swap(length_, other.length_); }

#if SZ_INCLUDE_STL_CONVERSIONS

    template <typename sfinae_ = char_type, typename std::enable_if<std::is_const<sfinae_>::value, int>::type = 0>
    sz_constexpr_if_cpp20 basic_string_slice(std::string const &other) noexcept
        : basic_string_slice(other.data(), other.size()) {}

    template <typename sfinae_ = char_type, typename std::enable_if<!std::is_const<sfinae_>::value, int>::type = 0>
    sz_constexpr_if_cpp20 basic_string_slice(std::string &other) noexcept
        : basic_string_slice(other.data(), other.size()) {}

    template <typename sfinae_ = char_type, typename std::enable_if<std::is_const<sfinae_>::value, int>::type = 0>
    sz_constexpr_if_cpp20 string_slice &operator=(std::string const &other) noexcept {
        return assign({other.data(), other.size()});
    }

    template <typename sfinae_ = char_type, typename std::enable_if<!std::is_const<sfinae_>::value, int>::type = 0>
    sz_constexpr_if_cpp20 string_slice &operator=(std::string &other) noexcept {
        return assign({other.data(), other.size()});
    }

    operator std::string() const { return {data(), size()}; }

    /**
     *  @brief  Formatted output function for compatibility with STL's `std::basic_ostream`.
     *  @throw  `std::ios_base::failure` if an exception occurred during output.
     */
    template <typename stream_traits>
    friend std::basic_ostream<value_type, stream_traits> &operator<<(std::basic_ostream<value_type, stream_traits> &os,
                                                                     string_slice const &str) noexcept(false) {
        return os.write(str.data(), str.size());
    }

#if SZ_DETECT_CPP_17 && __cpp_lib_string_view

    template <typename sfinae_ = char_type, typename std::enable_if<std::is_const<sfinae_>::value, int>::type = 0>
    sz_constexpr_if_cpp20 basic_string_slice(std::string_view const &other) noexcept
        : basic_string_slice(other.data(), other.size()) {}

    template <typename sfinae_ = char_type, typename std::enable_if<std::is_const<sfinae_>::value, int>::type = 0>
    sz_constexpr_if_cpp20 string_slice &operator=(std::string_view const &other) noexcept {
        return assign({other.data(), other.size()});
    }
    operator std::string_view() const noexcept { return {data(), size()}; }

#endif

#endif

#pragma endregion

#pragma region Iterators and Element Access

    iterator begin() const noexcept { return iterator(start_); }
    iterator end() const noexcept { return iterator(start_ + length_); }
    const_iterator cbegin() const noexcept { return const_iterator(start_); }
    const_iterator cend() const noexcept { return const_iterator(start_ + length_); }
    reverse_iterator rbegin() const noexcept { return reverse_iterator(start_ + length_ - 1); }
    reverse_iterator rend() const noexcept { return reverse_iterator(start_ - 1); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(start_ + length_ - 1); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(start_ - 1); }

    const_reference operator[](size_type pos) const noexcept { return start_[pos]; }
    const_reference at(size_type pos) const noexcept { return start_[pos]; }
    const_reference front() const noexcept { return start_[0]; }
    const_reference back() const noexcept { return start_[length_ - 1]; }
    const_pointer data() const noexcept { return start_; }

    difference_type ssize() const noexcept { return static_cast<difference_type>(length_); }
    size_type size() const noexcept { return length_; }
    size_type length() const noexcept { return length_; }
    size_type max_size() const noexcept { return npos - 1; }
    bool empty() const noexcept { return length_ == 0; }

#pragma endregion

#pragma region Slicing

#pragma region Safe and Signed Extensions

    /**
     *  @brief  Equivalent to Python's `"abc"[-3:-1]`. Exception-safe, unlike STL's `substr`.
     *          Supports signed and unsigned intervals.
     */
    string_slice operator[](std::initializer_list<difference_type> signed_offsets) const noexcept {
        assert(signed_offsets.size() == 2 && "operator[] can't take more than 2 offsets");
        return sub(signed_offsets.begin()[0], signed_offsets.begin()[1]);
    }

    /**
     *  @brief  Signed alternative to `at()`. Handy if you often write `str[str.size() - 2]`.
     *  @warning The behavior is @b undefined if the position is beyond bounds.
     */
    value_type sat(difference_type signed_offset) const noexcept {
        size_type pos = (signed_offset < 0) ? size() + signed_offset : signed_offset;
        assert(pos < size() && "string_slice::sat(i) out of bounds");
        return start_[pos];
    }

    /**
     *  @brief  The opposite operation to `remove_prefix`, that does no bounds checking.
     *  @warning The behavior is @b undefined if `n > size()`.
     */
    string_slice front(size_type n) const noexcept {
        assert(n <= size() && "string_slice::front(n) out of bounds");
        return {start_, n};
    }

    /**
     *  @brief  The opposite operation to `remove_prefix`, that does no bounds checking.
     *  @warning The behavior is @b undefined if `n > size()`.
     */
    string_slice back(size_type n) const noexcept {
        assert(n <= size() && "string_slice::back(n) out of bounds");
        return {start_ + length_ - n, n};
    }

    /**
     *  @brief  Equivalent to Python's `"abc"[-3:-1]`. Exception-safe, unlike STL's `substr`.
     *          Supports signed and unsigned intervals.
     */
    string_slice sub(difference_type signed_start_offset, difference_type signed_end_offset = npos) const noexcept {
        sz_size_t normalized_offset, normalized_length;
        sz_ssize_clamp_interval(length_, signed_start_offset, signed_end_offset, &normalized_offset,
                                &normalized_length);
        return string_slice(start_ + normalized_offset, normalized_length);
    }

    /**
     *  @brief  Exports this entire view. Not an STL function, but useful for concatenations.
     *          The STL variant expects at least two arguments.
     */
    size_type copy(value_type *destination) const noexcept {
        sz_copy((sz_ptr_t)destination, start_, length_);
        return length_;
    }

#pragma endregion

#pragma region STL Style

    /**
     *  @brief  Removes the first `n` characters from the view.
     *  @warning The behavior is @b undefined if `n > size()`.
     */
    void remove_prefix(size_type n) noexcept { assert(n <= size()), start_ += n, length_ -= n; }

    /**
     *  @brief  Removes the last `n` characters from the view.
     *  @warning The behavior is @b undefined if `n > size()`.
     */
    void remove_suffix(size_type n) noexcept { assert(n <= size()), length_ -= n; }

    /**  @brief  Added for STL compatibility. */
    string_slice substr() const noexcept { return *this; }

    /**
     *  @brief  Return a slice of this view after first `skip` bytes.
     *  @throws `std::out_of_range` if `skip > size()`.
     *  @see    `sub` for a cleaner exception-less alternative.
     */
    string_slice substr(size_type skip) const noexcept(false) {
        if (skip > size()) throw std::out_of_range("string_slice::substr");
        return string_slice(start_ + skip, length_ - skip);
    }

    /**
     *  @brief  Return a slice of this view after first `skip` bytes, taking at most `count` bytes.
     *  @throws `std::out_of_range` if `skip > size()`.
     *  @see    `sub` for a cleaner exception-less alternative.
     */
    string_slice substr(size_type skip, size_type count) const noexcept(false) {
        if (skip > size()) throw std::out_of_range("string_slice::substr");
        return string_slice(start_ + skip, sz_min_of_two(count, length_ - skip));
    }

    /**
     *  @brief  Exports a slice of this view after first `skip` bytes, taking at most `count` bytes.
     *  @throws `std::out_of_range` if `skip > size()`.
     *  @see    `sub` for a cleaner exception-less alternative.
     */
    size_type copy(value_type *destination, size_type count, size_type skip = 0) const noexcept(false) {
        if (skip > size()) throw std::out_of_range("string_slice::copy");
        count = sz_min_of_two(count, length_ - skip);
        sz_copy((sz_ptr_t)destination, start_ + skip, count);
        return count;
    }

#pragma endregion

#pragma endregion

#pragma region Comparisons

#pragma region Whole String Comparisons

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(string_view other) const noexcept {
        return (int)sz_order(start_, length_, other.start_, other.length_);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *          Equivalent to `substr(pos1, count1).compare(other)`.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     *  @throw  `std::out_of_range` if `pos1 > size()`.
     */
    int compare(size_type pos1, size_type count1, string_view other) const noexcept(false) {
        return substr(pos1, count1).compare(other);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *          Equivalent to `substr(pos1, count1).compare(other.substr(pos2, count2))`.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     *  @throw  `std::out_of_range` if `pos1 > size()` or if `pos2 > other.size()`.
     */
    int compare(size_type pos1, size_type count1, string_view other, size_type pos2, size_type count2) const
        noexcept(false) {
        return substr(pos1, count1).compare(other.substr(pos2, count2));
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(const_pointer other) const noexcept { return compare(string_view(other)); }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *          Equivalent to substr(pos1, count1).compare(other).
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     *  @throw  `std::out_of_range` if `pos1 > size()`.
     */
    int compare(size_type pos1, size_type count1, const_pointer other) const noexcept(false) {
        return substr(pos1, count1).compare(string_view(other));
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *          Equivalent to `substr(pos1, count1).compare({s, count2})`.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     *  @throw  `std::out_of_range` if `pos1 > size()`.
     */
    int compare(size_type pos1, size_type count1, const_pointer other, size_type count2) const noexcept(false) {
        return substr(pos1, count1).compare(string_view(other, count2));
    }

    /**  @brief  Checks if the string is equal to the other string. */
    bool operator==(string_view other) const noexcept {
        return length_ == other.length_ && sz_equal(start_, other.start_, other.length_) == sz_true_k;
    }

    /**  @brief  Checks if the string is equal to a concatenation of two strings. */
    bool operator==(concatenation<string_view, string_view> const &other) const noexcept {
        return length_ == other.length() && sz_equal(start_, other.first.data(), other.first.length()) == sz_true_k &&
               sz_equal(start_ + other.first.length(), other.second.data(), other.second.length()) == sz_true_k;
    }

#if SZ_DETECT_CPP20

    /**  @brief  Computes the lexicographic ordering between this and the ::other string. */
    std::strong_ordering operator<=>(string_view other) const noexcept {
        std::strong_ordering orders[3] {std::strong_ordering::less, std::strong_ordering::equal,
                                        std::strong_ordering::greater};
        return orders[compare(other) + 1];
    }

#else

    /**  @brief  Checks if the string is not equal to the other string. */
    bool operator!=(string_view other) const noexcept { return !operator==(other); }

    /**  @brief  Checks if the string is lexicographically smaller than the other string. */
    bool operator<(string_view other) const noexcept { return compare(other) == sz_less_k; }

    /**  @brief  Checks if the string is lexicographically equal or smaller than the other string. */
    bool operator<=(string_view other) const noexcept { return compare(other) != sz_greater_k; }

    /**  @brief  Checks if the string is lexicographically greater than the other string. */
    bool operator>(string_view other) const noexcept { return compare(other) == sz_greater_k; }

    /**  @brief  Checks if the string is lexicographically equal or greater than the other string. */
    bool operator>=(string_view other) const noexcept { return compare(other) != sz_less_k; }

#endif

#pragma endregion
#pragma region Prefix and Suffix Comparisons

    /**  @brief  Checks if the string starts with the other string. */
    bool starts_with(string_view other) const noexcept {
        return length_ >= other.length_ && sz_equal(start_, other.start_, other.length_) == sz_true_k;
    }

    /**  @brief  Checks if the string starts with the other string. */
    bool starts_with(const_pointer other) const noexcept {
        auto other_length = null_terminated_length(other);
        return length_ >= other_length && sz_equal(start_, other, other_length) == sz_true_k;
    }

    /**  @brief  Checks if the string starts with the other character. */
    bool starts_with(value_type other) const noexcept { return length_ && start_[0] == other; }

    /**  @brief  Checks if the string ends with the other string. */
    bool ends_with(string_view other) const noexcept {
        return length_ >= other.length_ &&
               sz_equal(start_ + length_ - other.length_, other.start_, other.length_) == sz_true_k;
    }

    /**  @brief  Checks if the string ends with the other string. */
    bool ends_with(const_pointer other) const noexcept {
        auto other_length = null_terminated_length(other);
        return length_ >= other_length && sz_equal(start_ + length_ - other_length, other, other_length) == sz_true_k;
    }

    /**  @brief  Checks if the string ends with the other character. */
    bool ends_with(value_type other) const noexcept { return length_ && start_[length_ - 1] == other; }

    /**  @brief  Python-like convenience function, dropping the matching prefix. */
    string_slice remove_prefix(string_view other) const noexcept {
        return starts_with(other) ? string_slice {start_ + other.length_, length_ - other.length_} : *this;
    }

    /**  @brief  Python-like convenience function, dropping the matching suffix. */
    string_slice remove_suffix(string_view other) const noexcept {
        return ends_with(other) ? string_slice {start_, length_ - other.length_} : *this;
    }

#pragma endregion
#pragma endregion

#pragma region Matching Substrings

    bool contains(string_view other) const noexcept { return find(other) != npos; }
    bool contains(value_type character) const noexcept { return find(character) != npos; }
    bool contains(const_pointer other) const noexcept { return find(other) != npos; }

#pragma region Returning offsets

    /**
     *  @brief  Find the first occurrence of a substring, skipping the first `skip` characters.
     *          The behavior is @b undefined if `skip > size()`.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type find(string_view other, size_type skip = 0) const noexcept {
        auto ptr = sz_find(start_ + skip, length_ - skip, other.start_, other.length_);
        return ptr ? ptr - start_ : npos;
    }

    /**
     *  @brief  Find the first occurrence of a character, skipping the first `skip` characters.
     *          The behavior is @b undefined if `skip > size()`.
     *  @return The offset of the match, or `npos` if not found.
     */
    size_type find(value_type character, size_type skip = 0) const noexcept {
        auto ptr = sz_find_byte(start_ + skip, length_ - skip, &character);
        return ptr ? ptr - start_ : npos;
    }

    /**
     *  @brief  Find the first occurrence of a substring, skipping the first `skip` characters.
     *          The behavior is @b undefined if `skip > size()`.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type find(const_pointer other, size_type pos, size_type count) const noexcept {
        return find(string_view(other, count), pos);
    }

    /**
     *  @brief  Find the last occurrence of a substring.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type rfind(string_view other) const noexcept {
        auto ptr = sz_find_last(start_, length_, other.start_, other.length_);
        return ptr ? ptr - start_ : npos;
    }

    /**
     *  @brief  Find the last occurrence of a substring, within first `until` characters.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type rfind(string_view other, size_type until) const noexcept {
        return until < length_ ? substr(0, until + 1).rfind(other) : rfind(other);
    }

    /**
     *  @brief  Find the last occurrence of a character.
     *  @return The offset of the match, or `npos` if not found.
     */
    size_type rfind(value_type character) const noexcept {
        auto ptr = sz_find_last_byte(start_, length_, &character);
        return ptr ? ptr - start_ : npos;
    }

    /**
     *  @brief  Find the last occurrence of a character, within first `until` characters.
     *  @return The offset of the match, or `npos` if not found.
     */
    size_type rfind(value_type character, size_type until) const noexcept {
        return until < length_ ? substr(0, until + 1).rfind(character) : rfind(character);
    }

    /**
     *  @brief  Find the last occurrence of a substring, within first `until` characters.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type rfind(const_pointer other, size_type until, size_type count) const noexcept {
        return rfind(string_view(other, count), until);
    }

    /**  @brief  Find the first occurrence of a character from a set. */
    size_type find(char_set set) const noexcept { return find_first_of(set); }

    /**  @brief  Find the last occurrence of a character from a set. */
    size_type rfind(char_set set) const noexcept { return find_last_of(set); }

#pragma endregion
#pragma region Returning Partitions

    /**  @brief  Split the string into three parts, before the match, the match itself, and after it. */
    partition_type partition(string_view pattern) const noexcept { return partition_(pattern, pattern.length()); }

    /**  @brief  Split the string into three parts, before the match, the match itself, and after it. */
    partition_type partition(char_set pattern) const noexcept { return partition_(pattern, 1); }

    /**  @brief  Split the string into three parts, before the @b last match, the last match itself, and after it. */
    partition_type rpartition(string_view pattern) const noexcept { return rpartition_(pattern, pattern.length()); }

    /**  @brief  Split the string into three parts, before the @b last match, the last match itself, and after it. */
    partition_type rpartition(char_set pattern) const noexcept { return rpartition_(pattern, 1); }

#pragma endregion
#pragma endregion

#pragma region Matching Character Sets

    bool contains_only(char_set set) const noexcept { return find_first_not_of(set) == npos; }
    bool isalpha() const noexcept { return !empty() && contains_only(ascii_letters_set); }
    bool isalnum() const noexcept { return !empty() && contains_only(ascii_letters_set | digits_set); }
    bool isascii() const noexcept { return empty() || contains_only(ascii_controls_set | ascii_printables_set); }
    bool isdigit() const noexcept { return !empty() && contains_only(digits_set); }
    bool islower() const noexcept { return !empty() && contains_only(ascii_lowercase_set); }
    bool isspace() const noexcept { return !empty() && contains_only(whitespaces_set); }
    bool isupper() const noexcept { return !empty() && contains_only(ascii_uppercase_set); }
    bool isprintable() const noexcept { return empty() || contains_only(ascii_printables_set); }

#pragma region Character Set Arguments
    /**
     *  @brief  Find the first occurrence of a character from a set.
     *  @param  skip Number of characters to skip before the search.
     *  @warning The behavior is @b undefined if `skip > size()`.
     */
    size_type find_first_of(char_set set, size_type skip = 0) const noexcept {
        auto ptr = sz_find_from_set(start_ + skip, length_ - skip, &set.raw());
        return ptr ? ptr - start_ : npos;
    }

    /**
     *  @brief  Find the first occurrence of a character outside a set.
     *  @param  skip  The number of first characters to be skipped.
     *  @warning The behavior is @b undefined if `skip > size()`.
     */
    size_type find_first_not_of(char_set set, size_type skip = 0) const noexcept {
        return find_first_of(set.inverted(), skip);
    }

    /**
     *  @brief  Find the last occurrence of a character from a set.
     */
    size_type find_last_of(char_set set) const noexcept {
        auto ptr = sz_find_last_from_set(start_, length_, &set.raw());
        return ptr ? ptr - start_ : npos;
    }

    /**
     *  @brief  Find the last occurrence of a character outside a set.
     */
    size_type find_last_not_of(char_set set) const noexcept { return find_last_of(set.inverted()); }

    /**
     *  @brief  Find the last occurrence of a character from a set.
     *  @param  until  The offset of the last character to be considered.
     */
    size_type find_last_of(char_set set, size_type until) const noexcept {
        return until < length_ ? substr(0, until + 1).find_last_of(set) : find_last_of(set);
    }

    /**
     *  @brief  Find the last occurrence of a character outside a set.
     *  @param  until  The offset of the last character to be considered.
     */
    size_type find_last_not_of(char_set set, size_type until) const noexcept {
        return find_last_of(set.inverted(), until);
    }

#pragma endregion
#pragma region String Arguments

    /**
     *  @brief  Find the first occurrence of a character from a ::set.
     *  @param  skip  The number of first characters to be skipped.
     */
    size_type find_first_of(string_view other, size_type skip = 0) const noexcept {
        return find_first_of(other.as_set(), skip);
    }

    /**
     *  @brief  Find the first occurrence of a character outside a ::set.
     *  @param  skip  The number of first characters to be skipped.
     */
    size_type find_first_not_of(string_view other, size_type skip = 0) const noexcept {
        return find_first_not_of(other.as_set(), skip);
    }

    /**
     *  @brief  Find the last occurrence of a character from a ::set.
     *  @param  until  The offset of the last character to be considered.
     */
    size_type find_last_of(string_view other, size_type until = npos) const noexcept {
        return find_last_of(other.as_set(), until);
    }

    /**
     *  @brief  Find the last occurrence of a character outside a ::set.
     *  @param  until  The offset of the last character to be considered.
     */
    size_type find_last_not_of(string_view other, size_type until = npos) const noexcept {
        return find_last_not_of(other.as_set(), until);
    }

#pragma endregion
#pragma region C-Style Arguments

    /**
     *  @brief  Find the first occurrence of a character from a set.
     *  @param  skip  The number of first characters to be skipped.
     *  @warning The behavior is @b undefined if `skip > size()`.
     */
    size_type find_first_of(const_pointer other, size_type skip, size_type count) const noexcept {
        return find_first_of(string_view(other, count), skip);
    }

    /**
     *  @brief  Find the first occurrence of a character outside a set.
     *  @param  skip  The number of first characters to be skipped.
     *  @warning The behavior is @b undefined if `skip > size()`.
     */
    size_type find_first_not_of(const_pointer other, size_type skip, size_type count) const noexcept {
        return find_first_not_of(string_view(other, count), skip);
    }

    /**
     *  @brief  Find the last occurrence of a character from a set.
     *  @param  until  The number of first characters to be considered.
     */
    size_type find_last_of(const_pointer other, size_type until, size_type count) const noexcept {
        return find_last_of(string_view(other, count), until);
    }

    /**
     *  @brief  Find the last occurrence of a character outside a set.
     *  @param  until  The number of first characters to be considered.
     */
    size_type find_last_not_of(const_pointer other, size_type until, size_type count) const noexcept {
        return find_last_not_of(string_view(other, count), until);
    }

#pragma endregion
#pragma region Slicing

    /**
     *  @brief  Python-like convenience function, dropping prefix formed of given characters.
     *          Similar to `boost::algorithm::trim_left_if(str, is_any_of(set))`.
     */
    string_slice lstrip(char_set set) const noexcept {
        set = set.inverted();
        auto new_start = sz_find_from_set(start_, length_, &set.raw());
        return new_start ? string_slice {new_start, length_ - static_cast<size_type>(new_start - start_)}
                         : string_slice();
    }

    /**
     *  @brief  Python-like convenience function, dropping suffix formed of given characters.
     *          Similar to `boost::algorithm::trim_right_if(str, is_any_of(set))`.
     */
    string_slice rstrip(char_set set) const noexcept {
        set = set.inverted();
        auto new_end = sz_find_last_from_set(start_, length_, &set.raw());
        return new_end ? string_slice {start_, static_cast<size_type>(new_end - start_ + 1)} : string_slice();
    }

    /**
     *  @brief  Python-like convenience function, dropping both the prefix & the suffix formed of given characters.
     *          Similar to `boost::algorithm::trim_if(str, is_any_of(set))`.
     */
    string_slice strip(char_set set) const noexcept {
        set = set.inverted();
        auto new_start = sz_find_from_set(start_, length_, &set.raw());
        return new_start
                   ? string_slice {new_start,
                                   static_cast<size_type>(
                                       sz_find_last_from_set(new_start, length_ - (new_start - start_), &set.raw()) -
                                       new_start + 1)}
                   : string_slice();
    }

#pragma endregion
#pragma endregion

#pragma region Search Ranges

    using find_all_type = range_matches<string_slice, matcher_find<string_view, include_overlaps_type>>;
    using rfind_all_type = range_rmatches<string_slice, matcher_rfind<string_view, include_overlaps_type>>;

    using find_disjoint_type = range_matches<string_slice, matcher_find<string_view, exclude_overlaps_type>>;
    using rfind_disjoint_type = range_rmatches<string_slice, matcher_rfind<string_view, exclude_overlaps_type>>;

    using find_all_chars_type = range_matches<string_slice, matcher_find_first_of<string_view, char_set>>;
    using rfind_all_chars_type = range_rmatches<string_slice, matcher_find_last_of<string_view, char_set>>;

    /**  @brief  Find all potentially @b overlapping occurrences of a given string. */
    find_all_type find_all(string_view needle, include_overlaps_type = {}) const noexcept { return {*this, needle}; }

    /**  @brief  Find all potentially @b overlapping occurrences of a given string in @b reverse order. */
    rfind_all_type rfind_all(string_view needle, include_overlaps_type = {}) const noexcept { return {*this, needle}; }

    /**  @brief  Find all @b non-overlapping occurrences of a given string. */
    find_disjoint_type find_all(string_view needle, exclude_overlaps_type) const noexcept { return {*this, needle}; }

    /**  @brief  Find all @b non-overlapping occurrences of a given string in @b reverse order. */
    rfind_disjoint_type rfind_all(string_view needle, exclude_overlaps_type) const noexcept { return {*this, needle}; }

    /**  @brief  Find all occurrences of given characters. */
    find_all_chars_type find_all(char_set set) const noexcept { return {*this, {set}}; }

    /**  @brief  Find all occurrences of given characters in @b reverse order. */
    rfind_all_chars_type rfind_all(char_set set) const noexcept { return {*this, {set}}; }

    using split_type = range_splits<string_slice, matcher_find<string_view, exclude_overlaps_type>>;
    using rsplit_type = range_rsplits<string_slice, matcher_rfind<string_view, exclude_overlaps_type>>;

    using split_chars_type = range_splits<string_slice, matcher_find_first_of<string_view, char_set>>;
    using rsplit_chars_type = range_rsplits<string_slice, matcher_find_last_of<string_view, char_set>>;

    /**  @brief  Split around occurrences of a given string. */
    split_type split(string_view delimiter) const noexcept { return {*this, delimiter}; }

    /**  @brief  Split around occurrences of a given string in @b reverse order. */
    rsplit_type rsplit(string_view delimiter) const noexcept { return {*this, delimiter}; }

    /**  @brief  Split around occurrences of given characters. */
    split_chars_type split(char_set set = whitespaces_set) const noexcept { return {*this, {set}}; }

    /**  @brief  Split around occurrences of given characters in @b reverse order. */
    rsplit_chars_type rsplit(char_set set = whitespaces_set) const noexcept { return {*this, {set}}; }

    /**  @brief  Split around the occurrences of all newline characters. */
    split_chars_type splitlines() const noexcept { return split(newlines_set); }

#pragma endregion

    /**  @brief  Hashes the string, equivalent to `std::hash<string_view>{}(str)`. */
    size_type hash() const noexcept { return static_cast<size_type>(sz_hash(start_, length_)); }

    /**  @brief  Populate a character set with characters present in this string. */
    char_set as_set() const noexcept {
        char_set set;
        for (auto c : *this) set.add(c);
        return set;
    }

  private:
    constexpr string_view &assign(string_view const &other) noexcept {
        start_ = other.start_;
        length_ = other.length_;
        return *this;
    }
    inline static constexpr size_type null_terminated_length(const_pointer s) noexcept {
        const_pointer p = s;
        while (*p) ++p;
        return p - s;
    }

    template <typename pattern_>
    partition_type partition_(pattern_ &&pattern, std::size_t pattern_length) const noexcept {
        size_type pos = find(pattern);
        if (pos == npos) return {substr(), string_view(), string_view()};
        return {substr(0, pos), substr(pos, pattern_length), substr(pos + pattern_length)};
    }

    template <typename pattern_>
    partition_type rpartition_(pattern_ &&pattern, std::size_t pattern_length) const noexcept {
        size_type pos = rfind(pattern);
        if (pos == npos) return {substr(), string_view(), string_view()};
        return {substr(0, pos), substr(pos, pattern_length), substr(pos + pattern_length)};
    }
};

#pragma endregion

/**
 *  @brief  Memory-owning string class with a Small String Optimization.
 *
 *  @section API
 *
 *  Some APIs are different from `basic_string_slice`:
 *      * `lstrip`, `rstrip`, `strip` modify the string in-place, instead of returning a new view.
 *      * `sat`, `sub`, and element access has non-const overloads returning references to mutable objects.
 *
 *  Functions defined for `basic_string`, but not present in `basic_string_slice`:
 *      * `replace`, `insert`, `erase`, `append`, `push_back`, `pop_back`, `resize`, `shrink_to_fit`... from STL,
 *      * `try_` exception-free "try" operations that returning non-zero values on succces,
 *      * `replace_all` and `erase_all` similar to Boost,
 *      * `edit_distance` - Levenshtein distance computation reusing the allocator,
 *      * `randomize`, `random` - for fast random string generation.
 *
 *  Functions defined for `basic_string_slice`, but not present in `basic_string`:
 *      * `[r]partition`, `[r]split`, `[r]find_all` missing to enforce lifetime on long operations.
 *      * `remove_prefix`, `remove_suffix` for now.
 *
 *  @section Exceptions
 *
 *  Default constructor is `constexpr`. Move constructor and move assignment operator are `noexcept`.
 *  Copy constructor and copy assignment operator are not! They may throw `std::bad_alloc` if the memory
 *  allocation fails. Similar to STL `std::out_of_range` if the position argument to some of the functions
 *  is out of bounds. Same as with STL, the bound checks are often assymetric, so pay attention to docs.
 *  If exceptions are disabled, on failure, `std::terminate` is called.
 */
template <typename char_type_, typename allocator_type_ = std::allocator<char_type_>>
class basic_string {

    static_assert(sizeof(char_type_) == 1, "Characters must be a single byte long");
    static_assert(std::is_reference<char_type_>::value == false, "Characters can't be references");
    static_assert(std::is_const<char_type_>::value == false, "Characters must be mutable");

    using char_type = char_type_;
    using calloc_type = sz_memory_allocator_t;

    sz_string_t string_;

    /**
     *  Stateful allocators and their support in C++ strings is extremely error-prone by design.
     *  Depending on traits like `propagate_on_container_copy_assignment` and `propagate_on_container_move_assignment`,
     *  its state will be copied from one string to another. It goes against the design of most string constructors,
     *  as they also receive allocator as the last argument!
     */
    static_assert(std::is_empty<allocator_type_>::value, "We currently only support stateless allocators");

    static void *call_allocate(sz_size_t n, void *allocator_state) noexcept {
        return reinterpret_cast<allocator_type_ *>(allocator_state)->allocate(n);
    }

    static void call_free(void *ptr, sz_size_t n, void *allocator_state) noexcept {
        return reinterpret_cast<allocator_type_ *>(allocator_state)->deallocate(reinterpret_cast<char *>(ptr), n);
    }

    template <typename allocator_callback>
    bool with_alloc(allocator_callback &&callback) const noexcept {
        allocator_type_ allocator;
        sz_memory_allocator_t alloc;
        alloc.allocate = &call_allocate;
        alloc.free = &call_free;
        alloc.handle = &allocator;
        return callback(alloc);
    }

    bool is_internal() const noexcept { return sz_string_is_on_stack(&string_); }

    void init(std::size_t length, char_type value) noexcept(false) {
        sz_ptr_t start;
        if (!with_alloc([&](calloc_type &alloc) { return (start = sz_string_init_length(&string_, length, &alloc)); }))
            throw std::bad_alloc();
        sz_fill(start, length, *(sz_u8_t *)&value);
    }

    void init(string_view other) noexcept(false) {
        sz_ptr_t start;
        if (!with_alloc(
                [&](calloc_type &alloc) { return (start = sz_string_init_length(&string_, other.size(), &alloc)); }))
            throw std::bad_alloc();
        sz_copy(start, (sz_cptr_t)other.data(), other.size());
    }

    void move(basic_string &other) noexcept {
        // We can't just assign the other string state, as its start address may be somewhere else on the stack.
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
    // STL compatibility
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

    // Non-STL type definitions
    using allocator_type = allocator_type_;
    using string_span = basic_string_slice<char_type>;
    using string_view = basic_string_slice<typename std::add_const<char_type>::type>;
    using partition_type = string_partition_result<string_view>;

    /**  @brief  Special value for missing matches.
     *          We take the largest 63-bit unsigned integer.
     */
    inline static constexpr size_type npos = 0x7FFFFFFFFFFFFFFFull;

#pragma region Constructors and STL Utilities

    constexpr basic_string() noexcept {
        // ! Instead of relying on the `sz_string_init`, we have to reimplement it to support `constexpr`.
        string_.internal.start = &string_.internal.chars[0];
        string_.u64s[1] = 0;
        string_.u64s[2] = 0;
        string_.u64s[3] = 0;
    }

    ~basic_string() noexcept {
        with_alloc([&](calloc_type &alloc) {
            sz_string_free(&string_, &alloc);
            return true;
        });
    }

    basic_string(basic_string &&other) noexcept { move(other); }
    basic_string &operator=(basic_string &&other) noexcept {
        if (!is_internal()) {
            with_alloc([&](calloc_type &alloc) {
                sz_string_free(&string_, &alloc);
                return true;
            });
        }
        move(other);
        return *this;
    }

    basic_string(basic_string const &other) noexcept(false) { init(other); }
    basic_string &operator=(basic_string const &other) noexcept(false) { return assign(other); }
    basic_string(string_view view) noexcept(false) { init(view); }
    basic_string &operator=(string_view view) noexcept(false) { return assign(view); }

    basic_string(const_pointer c_string) noexcept(false) : basic_string(string_view(c_string)) {}
    basic_string(const_pointer c_string, size_type length) noexcept(false)
        : basic_string(string_view(c_string, length)) {}
    basic_string &operator=(const_pointer other) noexcept(false) { return assign(string_view(other)); }

    basic_string(std::nullptr_t) = delete;

    /**  @brief  Construct a string by repeating a certain ::character ::count times. */
    basic_string(size_type count, value_type character) noexcept(false) { init(count, character); }

    basic_string(basic_string const &other, size_type pos) noexcept(false) { init(string_view(other).substr(pos)); }
    basic_string(basic_string const &other, size_type pos, size_type count) noexcept(false) {
        init(string_view(other).substr(pos, count));
    }

    basic_string(std::initializer_list<value_type> list) noexcept(false) {
        init(string_view(list.begin(), list.size()));
    }

    operator string_view() const noexcept { return view(); }
    string_view view() const noexcept {
        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_string_range(&string_, &string_start, &string_length);
        return {string_start, string_length};
    }

    operator string_span() noexcept { return span(); }
    string_span span() noexcept {
        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_string_range(&string_, &string_start, &string_length);
        return {string_start, string_length};
    }

    /**  @brief Exchanges the string contents witt the `other` string. */
    void swap(basic_string &other) noexcept {
        // If at least one of the strings is on the stack, a basic `std::swap(string_, other.string_)` won't work,
        // as the pointer to the stack-allocated memory will be swapped, instead of the contents.
        sz_ptr_t first_start, second_start;
        sz_size_t first_length, second_length;
        sz_size_t first_space, second_space;
        sz_bool_t first_is_external, second_is_external;
        sz_string_unpack(&string_, &first_start, &first_length, &first_space, &first_is_external);
        sz_string_unpack(&other.string_, &second_start, &second_length, &second_space, &second_is_external);
        std::swap(string_, other.string_);
        if (!first_is_external) other.string_.internal.start = &other.string_.internal.chars[0];
        if (!second_is_external) string_.internal.start = &string_.internal.chars[0];
    }

#if SZ_INCLUDE_STL_CONVERSIONS

    basic_string(std::string const &other) noexcept(false) : basic_string(other.data(), other.size()) {}
    basic_string &operator=(std::string const &other) noexcept(false) { return assign({other.data(), other.size()}); }

    // As we are need both `data()` and `size()`, going through `operator string_view()`
    // and `sz_string_unpack` is faster than separate invocations.
    operator std::string() const { return view(); }

    /**
     *  @brief  Formatted output function for compatibility with STL's `std::basic_ostream`.
     *  @throw  `std::ios_base::failure` if an exception occurred during output.
     */
    template <typename stream_traits>
    friend std::basic_ostream<value_type, stream_traits> &operator<<(std::basic_ostream<value_type, stream_traits> &os,
                                                                     basic_string const &str) noexcept(false) {
        return os.write(str.data(), str.size());
    }

#if SZ_DETECT_CPP_17 && __cpp_lib_string_view

    basic_string(std::string_view other) noexcept(false) : basic_string(other.data(), other.size()) {}
    basic_string &operator=(std::string_view other) noexcept(false) { return assign({other.data(), other.size()}); }
    operator std::string_view() const noexcept { return view(); }

#endif

#endif

    template <typename first_type, typename second_type>
    explicit basic_string(concatenation<first_type, second_type> const &expression) noexcept(false) {
        with_alloc([&](calloc_type &alloc) {
            sz_ptr_t ptr = sz_string_init_length(&string_, expression.length(), &alloc);
            if (!ptr) return false;
            expression.copy(ptr);
            return true;
        });
    }

    template <typename first_type, typename second_type>
    basic_string &operator=(concatenation<first_type, second_type> const &expression) noexcept(false) {
        if (!try_assign(expression)) throw std::bad_alloc();
        return *this;
    }

#pragma endregion

#pragma region Iterators and Accessors

    iterator begin() noexcept { return iterator(data()); }
    const_iterator begin() const noexcept { return const_iterator(data()); }
    const_iterator cbegin() const noexcept { return const_iterator(data()); }

    // As we are need both `data()` and `size()`, going through `operator string_view()`
    // and `sz_string_unpack` is faster than separate invocations.
    iterator end() noexcept { return span().end(); }
    const_iterator end() const noexcept { return view().end(); }
    const_iterator cend() const noexcept { return view().end(); }

    reverse_iterator rbegin() noexcept { return span().rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return view().rbegin(); }
    const_reverse_iterator crbegin() const noexcept { return view().crbegin(); }

    reverse_iterator rend() noexcept { return span().rend(); }
    const_reverse_iterator rend() const noexcept { return view().rend(); }
    const_reverse_iterator crend() const noexcept { return view().crend(); }

    reference operator[](size_type pos) noexcept { return string_.internal.start[pos]; }
    const_reference operator[](size_type pos) const noexcept { return string_.internal.start[pos]; }

    reference front() noexcept { return string_.internal.start[0]; }
    const_reference front() const noexcept { return string_.internal.start[0]; }
    reference back() noexcept { return string_.internal.start[size() - 1]; }
    const_reference back() const noexcept { return string_.internal.start[size() - 1]; }
    pointer data() noexcept { return string_.internal.start; }
    const_pointer data() const noexcept { return string_.internal.start; }
    pointer c_str() noexcept { return string_.internal.start; }
    const_pointer c_str() const noexcept { return string_.internal.start; }

    reference at(size_type pos) noexcept(false) {
        if (pos >= size()) throw std::out_of_range("sz::basic_string::at");
        return string_.internal.start[pos];
    }
    const_reference at(size_type pos) const noexcept(false) {
        if (pos >= size()) throw std::out_of_range("sz::basic_string::at");
        return string_.internal.start[pos];
    }

    difference_type ssize() const noexcept { return static_cast<difference_type>(size()); }
    size_type size() const noexcept { return view().size(); }
    size_type length() const noexcept { return size(); }
    size_type max_size() const noexcept { return npos - 1; }
    bool empty() const noexcept { return string_.external.length == 0; }
    size_type capacity() const noexcept {
        sz_ptr_t string_start;
        sz_size_t string_length;
        sz_size_t string_space;
        sz_bool_t string_is_external;
        sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_external);
        return string_space - 1;
    }

    allocator_type get_allocator() const noexcept { return {}; }

#pragma endregion

#pragma region Slicing

#pragma region Safe and Signed Extensions

    /**
     *  @brief  Equivalent to Python's `"abc"[-3:-1]`. Exception-safe, unlike STL's `substr`.
     *          Supports signed and unsigned intervals.
     */
    string_view operator[](std::initializer_list<difference_type> offsets) const noexcept { return view()[offsets]; }
    string_span operator[](std::initializer_list<difference_type> offsets) noexcept { return span()[offsets]; }

    /**
     *  @brief  Signed alternative to `at()`. Handy if you often write `str[str.size() - 2]`.
     *  @warning The behavior is @b undefined if the position is beyond bounds.
     */
    value_type sat(difference_type offset) const noexcept { return view().sat(offset); }
    reference sat(difference_type offset) noexcept { return span().sat(offset); }

    /**
     *  @brief  The opposite operation to `remove_prefix`, that does no bounds checking.
     *  @warning The behavior is @b undefined if `n > size()`.
     */
    string_view front(size_type n) const noexcept { return view().front(n); }
    string_span front(size_type n) noexcept { return span().front(n); }

    /**
     *  @brief  The opposite operation to `remove_prefix`, that does no bounds checking.
     *  @warning The behavior is @b undefined if `n > size()`.
     */
    string_view back(size_type n) const noexcept { return view().back(n); }
    string_span back(size_type n) noexcept { return span().back(n); }

    /**
     *  @brief  Equivalent to Python's `"abc"[-3:-1]`. Exception-safe, unlike STL's `substr`.
     *          Supports signed and unsigned intervals. @b Doesn't copy or allocate memory!
     */
    string_view sub(difference_type start, difference_type end = npos) const noexcept { return view().sub(start, end); }
    string_span sub(difference_type start, difference_type end = npos) noexcept { return span().sub(start, end); }

    /**
     *  @brief  Exports this entire view. Not an STL function, but useful for concatenations.
     *          The STL variant expects at least two arguments.
     */
    size_type copy(value_type *destination) const noexcept { return view().copy(destination); }

#pragma endregion

#pragma region STL Style

    /**
     *  @brief  Removes the first `n` characters from the view.
     *  @warning The behavior is @b undefined if `n > size()`.
     */
    void remove_prefix(size_type n) noexcept {
        assert(n <= size());
        sz_string_erase(&string_, 0, n);
    }

    /**
     *  @brief  Removes the last `n` characters from the view.
     *  @warning The behavior is @b undefined if `n > size()`.
     */
    void remove_suffix(size_type n) noexcept {
        assert(n <= size());
        sz_string_erase(&string_, size() - n, n);
    }

    /**  @brief  Added for STL compatibility. */
    basic_string substr() const noexcept { return *this; }

    /**
     *  @brief  Return a slice of this view after first `skip` bytes.
     *  @throws `std::out_of_range` if `skip > size()`.
     *  @see    `sub` for a cleaner exception-less alternative.
     */
    basic_string substr(size_type skip) const noexcept(false) { return view().substr(skip); }

    /**
     *  @brief  Return a slice of this view after first `skip` bytes, taking at most `count` bytes.
     *  @throws `std::out_of_range` if `skip > size()`.
     *  @see    `sub` for a cleaner exception-less alternative.
     */
    basic_string substr(size_type skip, size_type count) const noexcept(false) { return view().substr(skip, count); }

    /**
     *  @brief  Exports a slice of this view after first `skip` bytes, taking at most `count` bytes.
     *  @throws `std::out_of_range` if `skip > size()`.
     *  @see    `sub` for a cleaner exception-less alternative.
     */
    size_type copy(value_type *destination, size_type count, size_type skip = 0) const noexcept(false) {
        return view().copy(destination, count, skip);
    }

#pragma endregion

#pragma endregion

#pragma region Comparisons

#pragma region Whole String Comparisons

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(string_view other) const noexcept { return view().compare(other); }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *          Equivalent to `substr(pos1, count1).compare(other)`.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     *  @throw  `std::out_of_range` if `pos1 > size()`.
     */
    int compare(size_type pos1, size_type count1, string_view other) const noexcept(false) {
        return view().compare(pos1, count1, other);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *          Equivalent to `substr(pos1, count1).compare(other.substr(pos2, count2))`.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     *  @throw  `std::out_of_range` if `pos1 > size()` or if `pos2 > other.size()`.
     */
    int compare(size_type pos1, size_type count1, string_view other, size_type pos2, size_type count2) const
        noexcept(false) {
        return view().compare(pos1, count1, other, pos2, count2);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     */
    int compare(const_pointer other) const noexcept { return view().compare(other); }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *          Equivalent to substr(pos1, count1).compare(other).
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     *  @throw  `std::out_of_range` if `pos1 > size()`.
     */
    int compare(size_type pos1, size_type count1, const_pointer other) const noexcept(false) {
        return view().compare(pos1, count1, other);
    }

    /**
     *  @brief  Compares two strings lexicographically. If prefix matches, lengths are compared.
     *          Equivalent to `substr(pos1, count1).compare({s, count2})`.
     *  @return 0 if equal, negative if `*this` is less than `other`, positive if `*this` is greater than `other`.
     *  @throw  `std::out_of_range` if `pos1 > size()`.
     */
    int compare(size_type pos1, size_type count1, const_pointer other, size_type count2) const noexcept(false) {
        return view().compare(pos1, count1, other, count2);
    }

    /**  @brief  Checks if the string is equal to the other string. */
    bool operator==(basic_string const &other) const noexcept { return view() == other.view(); }
    bool operator==(string_view other) const noexcept { return view() == other; }
    bool operator==(const_pointer other) const noexcept { return view() == string_view(other); }

#if SZ_DETECT_CPP20

    /**  @brief  Computes the lexicographic ordering between this and the ::other string. */
    std::strong_ordering operator<=>(basic_string const &other) const noexcept { return view() <=> other.view(); }
    std::strong_ordering operator<=>(string_view other) const noexcept { return view() <=> other; }
    std::strong_ordering operator<=>(const_pointer other) const noexcept { return view() <=> string_view(other); }

#else

    /**  @brief  Checks if the string is not equal to the other string. */
    bool operator!=(string_view other) const noexcept { return !operator==(other); }

    /**  @brief  Checks if the string is lexicographically smaller than the other string. */
    bool operator<(string_view other) const noexcept { return compare(other) == sz_less_k; }

    /**  @brief  Checks if the string is lexicographically equal or smaller than the other string. */
    bool operator<=(string_view other) const noexcept { return compare(other) != sz_greater_k; }

    /**  @brief  Checks if the string is lexicographically greater than the other string. */
    bool operator>(string_view other) const noexcept { return compare(other) == sz_greater_k; }

    /**  @brief  Checks if the string is lexicographically equal or greater than the other string. */
    bool operator>=(string_view other) const noexcept { return compare(other) != sz_less_k; }

#endif

#pragma endregion
#pragma region Prefix and Suffix Comparisons

    /**  @brief  Checks if the string starts with the other string. */
    bool starts_with(string_view other) const noexcept { return view().starts_with(other); }

    /**  @brief  Checks if the string starts with the other string. */
    bool starts_with(const_pointer other) const noexcept { return view().starts_with(other); }

    /**  @brief  Checks if the string starts with the other character. */
    bool starts_with(value_type other) const noexcept { return view().starts_with(other); }

    /**  @brief  Checks if the string ends with the other string. */
    bool ends_with(string_view other) const noexcept { return view().ends_with(other); }

    /**  @brief  Checks if the string ends with the other string. */
    bool ends_with(const_pointer other) const noexcept { return view().ends_with(other); }

    /**  @brief  Checks if the string ends with the other character. */
    bool ends_with(value_type other) const noexcept { return view().ends_with(other); }

#pragma endregion
#pragma endregion

#pragma region Matching Substrings

    bool contains(string_view other) const noexcept { return view().contains(other); }
    bool contains(value_type character) const noexcept { return view().contains(character); }
    bool contains(const_pointer other) const noexcept { return view().contains(other); }

#pragma region Returning offsets

    /**
     *  @brief  Find the first occurrence of a substring, skipping the first `skip` characters.
     *          The behavior is @b undefined if `skip > size()`.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type find(string_view other, size_type skip = 0) const noexcept { return view().find(other, skip); }

    /**
     *  @brief  Find the first occurrence of a character, skipping the first `skip` characters.
     *          The behavior is @b undefined if `skip > size()`.
     *  @return The offset of the match, or `npos` if not found.
     */
    size_type find(value_type character, size_type skip = 0) const noexcept { return view().find(character, skip); }

    /**
     *  @brief  Find the first occurrence of a substring, skipping the first `skip` characters.
     *          The behavior is @b undefined if `skip > size()`.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type find(const_pointer other, size_type pos, size_type count) const noexcept {
        return view().find(other, pos, count);
    }

    /**
     *  @brief  Find the last occurrence of a substring.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type rfind(string_view other) const noexcept { return view().rfind(other); }

    /**
     *  @brief  Find the last occurrence of a substring, within first `until` characters.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type rfind(string_view other, size_type until) const noexcept { return view().rfind(other, until); }

    /**
     *  @brief  Find the last occurrence of a character.
     *  @return The offset of the match, or `npos` if not found.
     */
    size_type rfind(value_type character) const noexcept { return view().rfind(character); }

    /**
     *  @brief  Find the last occurrence of a character, within first `until` characters.
     *  @return The offset of the match, or `npos` if not found.
     */
    size_type rfind(value_type character, size_type until) const noexcept { return view().rfind(character, until); }

    /**
     *  @brief  Find the last occurrence of a substring, within first `until` characters.
     *  @return The offset of the first character of the match, or `npos` if not found.
     */
    size_type rfind(const_pointer other, size_type until, size_type count) const noexcept {
        return view().rfind(other, until, count);
    }

    /**  @brief  Find the first occurrence of a character from a set. */
    size_type find(char_set set) const noexcept { return view().find(set); }

    /**  @brief  Find the last occurrence of a character from a set. */
    size_type rfind(char_set set) const noexcept { return view().rfind(set); }

#pragma endregion
#pragma endregion

#pragma region Matching Character Sets

    bool contains_only(char_set set) const noexcept { return find_first_not_of(set) == npos; }
    bool isalpha() const noexcept { return !empty() && contains_only(ascii_letters_set); }
    bool isalnum() const noexcept { return !empty() && contains_only(ascii_letters_set | digits_set); }
    bool isascii() const noexcept { return empty() || contains_only(ascii_controls_set | ascii_printables_set); }
    bool isdigit() const noexcept { return !empty() && contains_only(digits_set); }
    bool islower() const noexcept { return !empty() && contains_only(ascii_lowercase_set); }
    bool isspace() const noexcept { return !empty() && contains_only(whitespaces_set); }
    bool isupper() const noexcept { return !empty() && contains_only(ascii_uppercase_set); }
    bool isprintable() const noexcept { return empty() || contains_only(ascii_printables_set); }

#pragma region Character Set Arguments

    /**
     *  @brief  Find the first occurrence of a character from a set.
     *  @param  skip Number of characters to skip before the search.
     *  @warning The behavior is @b undefined if `skip > size()`.
     */
    size_type find_first_of(char_set set, size_type skip = 0) const noexcept { return view().find_first_of(set, skip); }

    /**
     *  @brief  Find the first occurrence of a character outside a set.
     *  @param  skip  The number of first characters to be skipped.
     *  @warning The behavior is @b undefined if `skip > size()`.
     */
    size_type find_first_not_of(char_set set, size_type skip = 0) const noexcept {
        return view().find_first_not_of(set, skip);
    }

    /**
     *  @brief  Find the last occurrence of a character from a set.
     */
    size_type find_last_of(char_set set) const noexcept { return view().find_last_of(set); }

    /**
     *  @brief  Find the last occurrence of a character outside a set.
     */
    size_type find_last_not_of(char_set set) const noexcept { return view().find_last_not_of(set); }

    /**
     *  @brief  Find the last occurrence of a character from a set.
     *  @param  until  The offset of the last character to be considered.
     */
    size_type find_last_of(char_set set, size_type until) const noexcept { return view().find_last_of(set, until); }

    /**
     *  @brief  Find the last occurrence of a character outside a set.
     *  @param  until  The offset of the last character to be considered.
     */
    size_type find_last_not_of(char_set set, size_type until) const noexcept {
        return view().find_last_not_of(set, until);
    }

#pragma endregion
#pragma region String Arguments

    /**
     *  @brief  Find the first occurrence of a character from a ::set.
     *  @param  skip  The number of first characters to be skipped.
     */
    size_type find_first_of(string_view other, size_type skip = 0) const noexcept {
        return view().find_first_of(other, skip);
    }

    /**
     *  @brief  Find the first occurrence of a character outside a ::set.
     *  @param  skip  The number of first characters to be skipped.
     */
    size_type find_first_not_of(string_view other, size_type skip = 0) const noexcept {
        return view().find_first_not_of(other, skip);
    }

    /**
     *  @brief  Find the last occurrence of a character from a ::set.
     *  @param  until  The offset of the last character to be considered.
     */
    size_type find_last_of(string_view other, size_type until = npos) const noexcept {
        return view().find_last_of(other, until);
    }

    /**
     *  @brief  Find the last occurrence of a character outside a ::set.
     *  @param  until  The offset of the last character to be considered.
     */
    size_type find_last_not_of(string_view other, size_type until = npos) const noexcept {
        return view().find_last_not_of(other, until);
    }

#pragma endregion
#pragma region C-Style Arguments

    /**
     *  @brief  Find the first occurrence of a character from a set.
     *  @param  skip  The number of first characters to be skipped.
     *  @warning The behavior is @b undefined if `skip > size()`.
     */
    size_type find_first_of(const_pointer other, size_type skip, size_type count) const noexcept {
        return view().find_first_of(other, skip, count);
    }

    /**
     *  @brief  Find the first occurrence of a character outside a set.
     *  @param  skip  The number of first characters to be skipped.
     *  @warning The behavior is @b undefined if `skip > size()`.
     */
    size_type find_first_not_of(const_pointer other, size_type skip, size_type count) const noexcept {
        return view().find_first_not_of(other, skip, count);
    }

    /**
     *  @brief  Find the last occurrence of a character from a set.
     *  @param  until  The number of first characters to be considered.
     */
    size_type find_last_of(const_pointer other, size_type until, size_type count) const noexcept {
        return view().find_last_of(other, until, count);
    }

    /**
     *  @brief  Find the last occurrence of a character outside a set.
     *  @param  until  The number of first characters to be considered.
     */
    size_type find_last_not_of(const_pointer other, size_type until, size_type count) const noexcept {
        return view().find_last_not_of(other, until, count);
    }

#pragma endregion
#pragma region Slicing

    /**
     *  @brief  Python-like convenience function, dropping prefix formed of given characters.
     *          Similar to `boost::algorithm::trim_left_if(str, is_any_of(set))`.
     */
    basic_string &lstrip(char_set set) noexcept {
        auto remaining = view().lstrip(set);
        remove_prefix(size() - remaining.size());
        return *this;
    }

    /**
     *  @brief  Python-like convenience function, dropping suffix formed of given characters.
     *          Similar to `boost::algorithm::trim_right_if(str, is_any_of(set))`.
     */
    basic_string &rstrip(char_set set) noexcept {
        auto remaining = view().rstrip(set);
        remove_suffix(size() - remaining.size());
        return *this;
    }

    /**
     *  @brief  Python-like convenience function, dropping both the prefix & the suffix formed of given characters.
     *          Similar to `boost::algorithm::trim_if(str, is_any_of(set))`.
     */
    basic_string &strip(char_set set) noexcept { return lstrip(set).rstrip(set); }

#pragma endregion
#pragma endregion

#pragma region Modifiers
#pragma region Non-STL API

    bool try_resize(size_type count, value_type character = '\0') noexcept;

    bool try_reserve(size_type capacity) noexcept {
        return with_alloc([&](calloc_type &alloc) { return sz_string_reserve(&string_, capacity, &alloc); });
    }

    bool try_assign(string_view other) noexcept;

    template <typename first_type, typename second_type>
    bool try_assign(concatenation<first_type, second_type> const &other) noexcept;

    bool try_push_back(char_type c) noexcept;

    bool try_append(const_pointer str, size_type length) noexcept;

    bool try_append(string_view str) noexcept { return try_append(str.data(), str.size()); }

    /**
     *  @brief  Erases ( @b in-place ) a range of characters defined with signed offsets.
     *  @return Number of characters removed.
     */
    size_type try_erase(difference_type signed_start_offset = 0, difference_type signed_end_offset = npos) noexcept {
        sz_size_t normalized_offset, normalized_length;
        sz_ssize_clamp_interval(size(), signed_start_offset, signed_end_offset, &normalized_offset, &normalized_length);
        if (!normalized_length) return false;
        sz_string_erase(&string_, normalized_offset, normalized_length);
        return normalized_length;
    }

    /**
     *  @brief  Inserts ( @b in-place ) a range of characters at a given signed offset.
     *  @return `true` if the insertion was successful, `false` otherwise.
     */
    bool try_insert(difference_type signed_offset, string_view string) noexcept {
        sz_size_t normalized_offset, normalized_length;
        sz_ssize_clamp_interval(size(), signed_offset, 0, &normalized_offset, &normalized_length);
        if (!with_alloc([&](calloc_type &alloc) {
                return sz_string_expand(&string_, normalized_offset, string.size(), &alloc);
            }))
            return false;

        sz_copy(data() + normalized_offset, string.data(), string.size());
        return true;
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @return `true` if the replacement was successful, `false` otherwise.
     */
    bool try_replace(difference_type signed_start_offset, difference_type signed_end_offset,
                     string_view replacement) noexcept {

        sz_size_t normalized_offset, normalized_length;
        sz_ssize_clamp_interval(size(), signed_start_offset, signed_end_offset, &normalized_offset, &normalized_length);
        if (!try_preparing_replacement(normalized_offset, normalized_length, replacement)) return false;
        sz_copy(data() + normalized_offset, replacement.data(), replacement.size());
        return true;
    }

#pragma endregion

#pragma region STL Interfaces

    /**
     *  @brief  Clears the string contents, but @b no deallocations happen.
     */
    void clear() noexcept { sz_string_erase(&string_, 0, SZ_SIZE_MAX); }

    /**
     *  @brief  Resizes the string to the given size, filling the new space with the given character,
     *          or NULL-character if nothing is provided.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    void resize(size_type count, value_type character = '\0') noexcept(false) {
        if (count > max_size()) throw std::length_error("sz::basic_string::resize");
        if (!try_resize(count, character)) throw std::bad_alloc();
    }

    /**
     *  @brief  Informs the string object of a planned change in size, so that it pre-allocate once.
     *  @throw  `std::length_error` if the string is too long.
     */
    void reserve(size_type capacity) noexcept(false) {
        if (capacity > max_size()) throw std::length_error("sz::basic_string::reserve");
        if (!try_reserve(capacity)) throw std::bad_alloc();
    }

    /**
     *  @brief  Inserts ( @b in-place ) a ::character multiple times at the given offset.
     *  @throw  `std::out_of_range` if `offset > size()`.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    basic_string &insert(size_type offset, size_type repeats, char_type character) noexcept(false) {
        if (offset > size()) throw std::out_of_range("sz::basic_string::insert");
        if (size() + repeats > max_size()) throw std::length_error("sz::basic_string::insert");
        if (!with_alloc([&](calloc_type &alloc) { return sz_string_expand(&string_, offset, repeats, &alloc); }))
            throw std::bad_alloc();

        sz_fill(data() + offset, repeats, character);
        return *this;
    }

    /**
     *  @brief  Inserts ( @b in-place ) a range of characters at the given offset.
     *  @throw  `std::out_of_range` if `offset > size()`.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    basic_string &insert(size_type offset, string_view other) noexcept(false) {
        if (offset > size()) throw std::out_of_range("sz::basic_string::insert");
        if (size() + other.size() > max_size()) throw std::length_error("sz::basic_string::insert");
        if (!with_alloc([&](calloc_type &alloc) { return sz_string_expand(&string_, offset, other.size(), &alloc); }))
            throw std::bad_alloc();

        sz_copy(data() + offset, other.data(), other.size());
        return *this;
    }

    /**
     *  @brief  Inserts ( @b in-place ) a range of characters at the given offset.
     *  @throw  `std::out_of_range` if `offset > size()`.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    basic_string &insert(size_type offset, const_pointer start, size_type length) noexcept(false) {
        return insert(offset, string_view(start, length));
    }

    /**
     *  @brief  Inserts ( @b in-place ) a slice of another string at the given offset.
     *  @throw  `std::out_of_range` if `offset > size()` or `other_index > other.size()`.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    basic_string &insert(size_type offset, string_view other, size_type other_index,
                         size_type count = npos) noexcept(false) {
        return insert(offset, other.substr(other_index, count));
    }

    /**
     *  @brief  Inserts ( @b in-place ) one ::character at the given iterator position.
     *  @throw  `std::out_of_range` if `pos > size()` or `other_index > other.size()`.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    iterator insert(const_iterator it, char_type character) noexcept(false) {
        auto pos = range_length(cbegin(), it);
        insert(pos, string_view(&character, 1));
        return begin() + pos;
    }

    /**
     *  @brief  Inserts ( @b in-place ) a ::character multiple times at the given iterator position.
     *  @throw  `std::out_of_range` if `pos > size()` or `other_index > other.size()`.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    iterator insert(const_iterator it, size_type repeats, char_type character) noexcept(false) {
        auto pos = range_length(cbegin(), it);
        insert(pos, repeats, character);
        return begin() + pos;
    }

    /**
     *  @brief  Inserts ( @b in-place ) a range at the given iterator position.
     *  @throw  `std::out_of_range` if `pos > size()` or `other_index > other.size()`.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    template <typename input_iterator>
    iterator insert(const_iterator it, input_iterator first, input_iterator last) noexcept(false) {

        auto pos = range_length(cbegin(), it);
        if (pos > size()) throw std::out_of_range("sz::basic_string::insert");

        auto added_length = range_length(first, last);
        if (size() + added_length > max_size()) throw std::length_error("sz::basic_string::insert");

        if (!with_alloc([&](calloc_type &alloc) { return sz_string_expand(&string_, pos, added_length, &alloc); }))
            throw std::bad_alloc();

        iterator result = begin() + pos;
        for (iterator output = result; first != last; ++first, ++output) *output = *first;
        return result;
    }

    /**
     *  @brief  Inserts ( @b in-place ) an initializer list of characters.
     *  @throw  `std::out_of_range` if `pos > size()` or `other_index > other.size()`.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    iterator insert(const_iterator it, std::initializer_list<char_type> ilist) noexcept(false) {
        return insert(it, ilist.begin(), ilist.end());
    }

    /**
     *  @brief  Erases ( @b in-place ) the given range of characters.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @see    `try_erase_slice` for a cleaner exception-less alternative.
     */
    basic_string &erase(size_type pos = 0, size_type count = npos) noexcept(false) {
        if (!count || empty()) return *this;
        if (pos >= size()) throw std::out_of_range("sz::basic_string::erase");
        sz_string_erase(&string_, pos, count);
        return *this;
    }

    /**
     *  @brief  Erases ( @b in-place ) the given range of characters.
     *  @return Iterator pointing following the erased character, or end() if no such character exists.
     */
    iterator erase(const_iterator first, const_iterator last) noexcept {
        auto start = begin();
        auto offset = first - start;
        sz_string_erase(&string_, offset, last - first);
        return start + offset;
    }

    /**
     *  @brief  Erases ( @b in-place ) the one character at a given postion.
     *  @return Iterator pointing following the erased character, or end() if no such character exists.
     */
    iterator erase(const_iterator pos) noexcept { return erase(pos, pos + 1); }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(size_type pos, size_type count, string_view const &str) noexcept(false) {
        if (pos > size()) throw std::out_of_range("sz::basic_string::replace");
        if (size() - count + str.size() > max_size()) throw std::length_error("sz::basic_string::replace");
        if (!try_preparing_replacement(pos, count, str.size())) throw std::bad_alloc();
        sz_copy(data() + pos, str.data(), str.size());
        return *this;
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(const_iterator first, const_iterator last, string_view const &str) noexcept(false) {
        return replace(range_length(cbegin(), first), last - first, str);
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @throws `std::out_of_range` if `pos > size()` or `pos2 > str.size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(size_type pos, size_type count, string_view const &str, size_type pos2,
                          size_type count2 = npos) noexcept(false) {
        return replace(pos, count, str.substr(pos2, count2));
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(size_type pos, size_type count, const_pointer cstr, size_type count2) noexcept(false) {
        return replace(pos, count, string_view(cstr, count2));
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(const_iterator first, const_iterator last, const_pointer cstr,
                          size_type count2) noexcept(false) {
        return replace(range_length(cbegin(), first), last - first, string_view(cstr, count2));
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(size_type pos, size_type count, const_pointer cstr) noexcept(false) {
        return replace(pos, count, string_view(cstr));
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(const_iterator first, const_iterator last, const_pointer cstr) noexcept(false) {
        return replace(range_length(cbegin(), first), last - first, string_view(cstr));
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a repetition of given characters.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(size_type pos, size_type count, size_type count2, char_type character) noexcept(false) {
        if (pos > size()) throw std::out_of_range("sz::basic_string::replace");
        if (size() - count + count2 > max_size()) throw std::length_error("sz::basic_string::replace");
        if (!try_preparing_replacement(pos, count, count2)) throw std::bad_alloc();
        sz_fill(data() + pos, count2, character);
        return *this;
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a repetition of given characters.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(const_iterator first, const_iterator last, size_type count2,
                          char_type character) noexcept(false) {
        return replace(range_length(cbegin(), first), last - first, count2, character);
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given string.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    template <typename input_iterator>
    basic_string &replace(const_iterator first, const_iterator last, input_iterator first2,
                          input_iterator last2) noexcept(false) {
        auto pos = range_length(cbegin(), first);
        auto count = range_length(first, last);
        auto count2 = range_length(first2, last2);
        if (pos > size()) throw std::out_of_range("sz::basic_string::replace");
        if (size() - count + count2 > max_size()) throw std::length_error("sz::basic_string::replace");
        if (!try_preparing_replacement(pos, count, count2)) throw std::bad_alloc();
        for (iterator output = begin() + pos; first2 != last2; ++first2, ++output) *output = *first2;
        return *this;
    }

    /**
     *  @brief  Replaces ( @b in-place ) a range of characters with a given initializer list.
     *  @throws `std::out_of_range` if `pos > size()`.
     *  @throws `std::length_error` if the string is too long.
     *  @see    `try_replace` for a cleaner exception-less alternative.
     */
    basic_string &replace(const_iterator first, const_iterator last,
                          std::initializer_list<char_type> ilist) noexcept(false) {
        return replace(first, last, ilist.begin(), ilist.end());
    }

    /**
     *  @brief  Appends the given character at the end.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     */
    void push_back(char_type ch) noexcept(false) {
        if (size() == max_size()) throw std::length_error("string::push_back");
        if (!try_push_back(ch)) throw std::bad_alloc();
    }

    /**
     *  @brief  Removes the last character from the string.
     *  @warning The behavior is @b undefined if the string is empty.
     */
    void pop_back() noexcept { sz_string_erase(&string_, size() - 1, 1); }

    /**
     *  @brief  Overwrites the string with the given string.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_assign` for a cleaner exception-less alternative.
     */
    basic_string &assign(string_view other) noexcept(false) {
        if (!try_assign(other)) throw std::bad_alloc();
        return *this;
    }

    /**
     *  @brief  Overwrites the string with the given repeated character.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_assign` for a cleaner exception-less alternative.
     */
    basic_string &assign(size_type repeats, char_type character) noexcept(false) {
        resize(repeats, character);
        sz_fill(data(), repeats, *(sz_u8_t *)&character);
        return *this;
    }

    /**
     *  @brief  Overwrites the string with the given string.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_assign` for a cleaner exception-less alternative.
     */
    basic_string &assign(const_pointer other, size_type length) noexcept(false) { return assign({other, length}); }

    /**
     *  @brief  Overwrites the string with the given string.
     *  @throw  `std::length_error` if the string is too long or `pos > str.size()`.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_assign` for a cleaner exception-less alternative.
     */
    basic_string &assign(string_view str, size_type pos, size_type count = npos) noexcept(false) {
        return assign(str.substr(pos, count));
    }

    /**
     *  @brief  Overwrites the string with the given iterator range.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_assign` for a cleaner exception-less alternative.
     */
    template <typename input_iterator>
    basic_string &assign(input_iterator first, input_iterator last) noexcept(false) {
        resize(range_length(first, last));
        for (iterator output = begin(); first != last; ++first, ++output) *output = *first;
        return *this;
    }

    /**
     *  @brief  Overwrites the string with the given initializer list.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_assign` for a cleaner exception-less alternative.
     */
    basic_string &assign(std::initializer_list<char_type> ilist) noexcept(false) {
        return assign(ilist.begin(), ilist.end());
    }

    /**
     *  @brief  Appends to the end of the current string.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_append` for a cleaner exception-less alternative.
     */
    basic_string &append(string_view str) noexcept(false) {
        if (!try_append(str)) throw std::bad_alloc();
        return *this;
    }

    /**
     *  @brief  Appends to the end of the current string.
     *  @throw  `std::length_error` if the string is too long or `pos > str.size()`.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_append` for a cleaner exception-less alternative.
     */
    basic_string &append(string_view str, size_type pos, size_type length = npos) noexcept(false) {
        return append(str.substr(pos, length));
    }

    /**
     *  @brief  Appends to the end of the current string.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_append` for a cleaner exception-less alternative.
     */
    basic_string &append(const_pointer str, size_type length) noexcept(false) { return append({str, length}); }

    /**
     *  @brief  Appends to the end of the current string.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_append` for a cleaner exception-less alternative.
     */
    basic_string &append(const_pointer str) noexcept(false) { return append(string_view(str)); }

    /**
     *  @brief  Appends a repeated character to the end of the current string.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_append` for a cleaner exception-less alternative.
     */
    basic_string &append(size_type repeats, char_type ch) noexcept(false) {
        resize(size() + repeats, ch);
        return *this;
    }

    /**
     *  @brief  Appends to the end of the current string.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_append` for a cleaner exception-less alternative.
     */
    basic_string &append(std::initializer_list<char_type> other) noexcept(false) {
        return append(other.begin(), other.end());
    }

    /**
     *  @brief  Appends to the end of the current string.
     *  @throw  `std::length_error` if the string is too long.
     *  @throw  `std::bad_alloc` if the allocation fails.
     *  @see    `try_append` for a cleaner exception-less alternative.
     */
    template <typename input_iterator>
    basic_string &append(input_iterator first, input_iterator last) noexcept(false) {
        insert<input_iterator>(cend(), first, last);
        return *this;
    }

    basic_string &operator+=(string_view other) noexcept(false) { return append(other); }
    basic_string &operator+=(std::initializer_list<char_type> other) noexcept(false) { return append(other); }
    basic_string &operator+=(char_type character) noexcept(false) { return operator+=(string_view(&character, 1)); }
    basic_string &operator+=(const_pointer other) noexcept(false) { return operator+=(string_view(other)); }

    basic_string operator+(char_type character) const noexcept(false) { return operator+(string_view(&character, 1)); }
    basic_string operator+(const_pointer other) const noexcept(false) { return operator+(string_view(other)); }
    basic_string operator+(string_view other) const noexcept(false) {
        return basic_string {concatenation<string_view, string_view> {view(), other}};
    }
    basic_string operator+(std::initializer_list<char_type> other) const noexcept(false) {
        return basic_string {concatenation<string_view, string_view> {view(), other}};
    }

#pragma endregion
#pragma endregion

    concatenation<string_view, string_view> operator|(string_view other) const noexcept { return {view(), other}; }

    size_type edit_distance(string_view other, size_type bound = npos) const noexcept {
        size_type distance;
        with_alloc([&](calloc_type &alloc) {
            distance = sz_edit_distance(data(), size(), other.data(), other.size(), bound, &alloc);
            return true;
        });
        return distance;
    }

    /**  @brief  Hashes the string, equivalent to `std::hash<string_view>{}(str)`. */
    size_type hash() const noexcept { return view().hash(); }

    /**
     *  @brief  Overwrites the string with random characters from the given alphabet using the random generator.
     *
     *  @param  generator  A random generator function object that returns a random number in the range [0, 2^64).
     *  @param  alphabet   A string of characters to choose from.
     */
    template <typename generator_type>
    basic_string &randomize(generator_type &&generator, string_view alphabet = "abcdefghijklmnopqrstuvwxyz") noexcept {
        sz_ptr_t start;
        sz_size_t length;
        sz_string_range(&string_, &start, &length);
        sz_random_generator_t generator_callback = &random_generator<generator_type>;
        sz_generate(alphabet.data(), alphabet.size(), start, length, generator_callback, &generator);
        return *this;
    }

    /**
     *  @brief  Overwrites the string with random characters from the given alphabet
     *          using `std::rand` as the random generator.
     *
     *  @param  alphabet   A string of characters to choose from.
     */
    basic_string &randomize(string_view alphabet = "abcdefghijklmnopqrstuvwxyz") noexcept {
        return randomize(&std::rand, alphabet);
    }

    /**
     *  @brief  Generate a new random string of given length using `std::rand` as the random generator.
     *          May throw exceptions if the memory allocation fails.
     *
     *  @param  length     The length of the generated string.
     *  @param  alphabet   A string of characters to choose from.
     */
    static basic_string random(size_type length, string_view alphabet = "abcdefghijklmnopqrstuvwxyz") noexcept(false) {
        return basic_string(length, '\0').randomize(alphabet);
    }

    /**
     *  @brief  Replaces ( @b in-place ) all occurrences of a given string with the ::replacement string.
     *          Similar to `boost::algorithm::replace_all` and Python's `str.replace`.
     *
     *  The implementation is not as composable, as using search ranges combined with a replacing mapping for matches,
     *  and might be suboptimal, if you are exporting the cleaned-up string to another buffer.
     *  The algorithm is suboptimal when this string is made exclusively of the pattern.
     */
    basic_string &replace_all(string_view pattern, string_view replacement) noexcept(false) {
        if (!try_replace_all(pattern, replacement)) throw std::bad_alloc();
        return *this;
    }

    /**
     *  @brief  Replaces ( @b in-place ) all occurrences of a given character set with the ::replacement string.
     *          Similar to `boost::algorithm::replace_all` and Python's `str.replace`.
     *
     *  The implementation is not as composable, as using search ranges combined with a replacing mapping for matches,
     *  and might be suboptimal, if you are exporting the cleaned-up string to another buffer.
     *  The algorithm is suboptimal when this string is made exclusively of the pattern.
     */
    basic_string &replace_all(char_set pattern, string_view replacement) noexcept(false) {
        if (!try_replace_all(pattern, replacement)) throw std::bad_alloc();
        return *this;
    }

    /**
     *  @brief  Replaces ( @b in-place ) all occurrences of a given string with the ::replacement string.
     *          Similar to `boost::algorithm::replace_all` and Python's `str.replace`.
     *
     *  The implementation is not as composable, as using search ranges combined with a replacing mapping for matches,
     *  and might be suboptimal, if you are exporting the cleaned-up string to another buffer.
     *  The algorithm is suboptimal when this string is made exclusively of the pattern.
     */
    bool try_replace_all(string_view pattern, string_view replacement) noexcept {
        return try_replace_all_<string_view>(pattern, replacement);
    }

    /**
     *  @brief  Replaces ( @b in-place ) all occurrences of a given character set with the ::replacement string.
     *          Similar to `boost::algorithm::replace_all` and Python's `str.replace`.
     *
     *  The implementation is not as composable, as using search ranges combined with a replacing mapping for matches,
     *  and might be suboptimal, if you are exporting the cleaned-up string to another buffer.
     *  The algorithm is suboptimal when this string is made exclusively of the pattern.
     */
    bool try_replace_all(char_set pattern, string_view replacement) noexcept {
        return try_replace_all_<char_set>(pattern, replacement);
    }

  private:
    template <typename generator_type>
    static sz_u64_t random_generator(void *state) noexcept {
        generator_type &generator = *reinterpret_cast<generator_type *>(state);
        return generator();
    }

    template <typename pattern_type>
    bool try_replace_all_(pattern_type pattern, string_view replacement) noexcept;

    /**
     *  @brief  Tries to prepare the string for a replacement of a given range with a new string.
     *          The allocation may occur, if the replacement is longer than the replaced range.
     */
    bool try_preparing_replacement(size_type offset, size_type length, size_type new_length) noexcept;
};

using string = basic_string<char, std::allocator<char>>;

static_assert(sizeof(string) == 4 * sizeof(void *), "String size must be 4 pointers.");

namespace literals {
constexpr string_view operator""_sz(char const *str, std::size_t length) noexcept { return {str, length}; }
} // namespace literals

template <typename char_type_, typename allocator_>
bool basic_string<char_type_, allocator_>::try_resize(size_type count, value_type character) noexcept {
    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_external);

    // Allocate more space if needed.
    if (count >= string_space) {
        if (!with_alloc(
                [&](calloc_type &alloc) { return sz_string_expand(&string_, SZ_SIZE_MAX, count, &alloc) != NULL; }))
            return false;
        sz_string_unpack(&string_, &string_start, &string_length, &string_space, &string_is_external);
    }

    // Fill the trailing characters.
    if (count > string_length) {
        sz_fill(string_start + string_length, count - string_length, character);
        string_start[count] = '\0';
        // Knowing the layout of the string, we can perform this operation safely,
        // even if its located on stack.
        string_.external.length += count - string_length;
    }
    else { sz_string_erase(&string_, count, SZ_SIZE_MAX); }
    return true;
}

template <typename char_type_, typename allocator_>
bool basic_string<char_type_, allocator_>::try_assign(string_view other) noexcept {
    // We can't just assign the other string state, as its start address may be somewhere else on the stack.
    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_string_range(&string_, &string_start, &string_length);

    if (string_length >= other.length()) {
        other.copy(string_start, other.length());
        sz_string_erase(&string_, other.length(), SZ_SIZE_MAX);
    }
    else {
        if (!with_alloc([&](calloc_type &alloc) {
                string_start = sz_string_expand(&string_, SZ_SIZE_MAX, other.length(), &alloc);
                if (!string_start) return false;
                other.copy(string_start, other.length());
                return true;
            }))
            return false;
    }
    return true;
}

template <typename char_type_, typename allocator_>
bool basic_string<char_type_, allocator_>::try_push_back(char_type c) noexcept {
    return with_alloc([&](calloc_type &alloc) {
        auto old_size = size();
        sz_ptr_t start = sz_string_expand(&string_, SZ_SIZE_MAX, 1, &alloc);
        if (!start) return false;
        start[old_size] = c;
        return true;
    });
}

template <typename char_type_, typename allocator_>
bool basic_string<char_type_, allocator_>::try_append(const_pointer str, size_type length) noexcept {
    return with_alloc([&](calloc_type &alloc) {
        auto old_size = size();
        sz_ptr_t start = sz_string_expand(&string_, SZ_SIZE_MAX, length, &alloc);
        if (!start) return false;
        sz_copy(start + old_size, str, length);
        return true;
    });
}

template <typename char_type_, typename allocator_>
template <typename pattern_type>
bool basic_string<char_type_, allocator_>::try_replace_all_(pattern_type pattern, string_view replacement) noexcept {
    // Depending on the size of the pattern and the replacement, we may need to allocate more space.
    // There are 3 cases to consider:
    // 1. The pattern and the replacement are of the same length. Piece of cake!
    // 2. The pattern is longer than the replacement. We need to compact the strings.
    // 3. The pattern is shorter than the replacement. We may have to allocate more memory.
    using matcher_type = typename std::conditional<std::is_same<pattern_type, char_set>::value,
                                                   matcher_find_first_of<string_view, pattern_type>,
                                                   matcher_find<string_view, exclude_overlaps_type>>::type;
    matcher_type matcher({pattern});
    string_view this_view = view();

    // 1. The pattern and the replacement are of the same length.
    if (matcher.needle_length() == replacement.length()) {
        using matches_type = range_matches<string_view, matcher_type>;
        // Instead of iterating with `begin()` and `end()`, we could use the cheaper sentinel-based approach.
        //      for (string_view match : matches) { ... }
        matches_type matches = matches_type(this_view, {pattern});
        for (auto matches_iterator = matches.begin(); matches_iterator != end_sentinel; ++matches_iterator) {
            replacement.copy(const_cast<pointer>((*matches_iterator).data()));
        }
        return true;
    }

    // 2. The pattern is longer than the replacement. We need to compact the strings.
    else if (matcher.needle_length() > replacement.length()) {
        // Dealing with shorter replacements, we will avoid memory allocations, but we can also mimnimize the number
        // of `memmove`-s, by keeping one more iterator, pointing to the end of the last compacted area.
        // Having the split-ranges, however, we reuse their logic.
        using splits_type = range_splits<string_view, matcher_type>;
        splits_type splits = splits_type(this_view, {pattern});
        auto matches_iterator = splits.begin();
        auto compacted_end = (*matches_iterator).end();
        if (compacted_end == end()) return true; // No matches.

        ++matches_iterator; // Skip the first match.
        do {
            string_view match_view = *matches_iterator;
            replacement.copy(const_cast<pointer>(compacted_end));
            compacted_end += replacement.length();
            sz_move((sz_ptr_t)compacted_end, match_view.begin(), match_view.length());
            compacted_end += match_view.length();
            ++matches_iterator;
        } while (matches_iterator != end_sentinel);

        // Can't fail, so let's just return true :)
        try_resize(compacted_end - begin());
        return true;
    }

    // 3. The pattern is shorter than the replacement. We may have to allocate more memory.
    else {
        using rmatcher_type = typename std::conditional<std::is_same<pattern_type, char_set>::value,
                                                        matcher_find_last_of<string_view, pattern_type>,
                                                        matcher_rfind<string_view, exclude_overlaps_type>>::type;
        using rmatches_type = range_rmatches<string_view, rmatcher_type>;
        rmatches_type rmatches = rmatches_type(this_view, {pattern});

        // It's cheaper to iterate through the whole string once, counting the number of matches,
        // reserving memory once, than re-allocating and copying the string multiple times.
        auto matches_count = rmatches.size();
        if (matches_count == 0) return true; // No matches.

        // TODO: Resize without initializing the memory.
        auto replacement_delta_length = replacement.length() - matcher.needle_length();
        auto added_length = matches_count * replacement_delta_length;
        auto old_length = size();
        auto new_length = old_length + added_length;
        if (!try_resize(new_length)) return false;
        this_view = view().front(old_length);

        // Now iterate through splits similarly to the 2nd case, but in reverse order.
        using rsplits_type = range_rsplits<string_view, rmatcher_type>;
        rsplits_type splits = rsplits_type(this_view, {pattern});
        auto splits_iterator = splits.begin();

        // Put the compacted pointer to the end of the new string, and walk left.
        auto compacted_begin = this_view.data() + new_length;

        // By now we know that at least one match exists, which means the splits .
        do {
            string_view slice_view = *splits_iterator;
            compacted_begin -= slice_view.length();
            sz_move((sz_ptr_t)compacted_begin, slice_view.begin(), slice_view.length());
            compacted_begin -= replacement.length();
            replacement.copy(const_cast<pointer>(compacted_begin));
            ++splits_iterator;
        } while (!splits_iterator.is_last());

        return true;
    }
}

template <typename char_type_, typename allocator_>
template <typename first_type, typename second_type>
bool basic_string<char_type_, allocator_>::try_assign(concatenation<first_type, second_type> const &other) noexcept {
    // We can't just assign the other string state, as its start address may be somewhere else on the stack.
    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_string_range(&string_, &string_start, &string_length);

    if (string_length >= other.length()) {
        sz_string_erase(&string_, other.length(), SZ_SIZE_MAX);
        other.copy(string_start, other.length());
    }
    else {
        if (!with_alloc([&](calloc_type &alloc) {
                string_start = sz_string_expand(&string_, SZ_SIZE_MAX, other.length(), &alloc);
                if (!string_start) return false;
                other.copy(string_start, other.length());
                return true;
            }))
            return false;
    }
    return true;
}

template <typename char_type_, typename allocator_>
bool basic_string<char_type_, allocator_>::try_preparing_replacement(size_type offset, size_type length,
                                                                     size_type replacement_length) noexcept {
    // There are three cases:
    // 1. The replacement is the same length as the replaced range.
    // 2. The replacement is shorter than the replaced range.
    // 3. The replacement is longer than the replaced range. An allocation may occur.
    assert(offset + length <= size());

    // 1. The replacement is the same length as the replaced range.
    if (replacement_length == length) { return true; }

    // 2. The replacement is shorter than the replaced range.
    else if (replacement_length < length) {
        sz_string_erase(&string_, offset + replacement_length, length - replacement_length);
        return true;
    }
    // 3. The replacement is longer than the replaced range. An allocation may occur.
    else {
        return with_alloc([&](calloc_type &alloc) {
            return sz_string_expand(&string_, offset + length, replacement_length - length, &alloc);
        });
    }
}

/**  @brief  SFINAE-type used to infer the resulting type of concatenating multiple string together. */
template <typename... args_types>
struct concatenation_result {};

template <typename first_type, typename second_type>
struct concatenation_result<first_type, second_type> {
    using type = concatenation<first_type, second_type>;
};

template <typename first_type, typename... following_types>
struct concatenation_result<first_type, following_types...> {
    using type = concatenation<first_type, typename concatenation_result<following_types...>::type>;
};

/**
 *  @brief  Concatenates two strings into a template expression.
 */
template <typename first_type, typename second_type>
concatenation<first_type, second_type> concatenate(first_type &&first, second_type &&second) {
    return {first, second};
}

/**
 *  @brief  Concatenates two or more strings into a template expression.
 */
template <typename first_type, typename second_type, typename... following_types>
typename concatenation_result<first_type, second_type, following_types...>::type concatenate(
    first_type &&first, second_type &&second, following_types &&...following) {
    // Fold expression like the one below would result in faster compile times,
    // but would incur the penalty of additional `if`-statements in every `append` call.
    // Moreover, those are only supported in C++17 and later.
    //      std::size_t total_size = (strings.size() + ... + 0);
    //      std::string result;
    //      result.reserve(total_size);
    //      (result.append(strings), ...);
    return ashvardanian::stringzilla::concatenate(
        std::forward<first_type>(first),
        ashvardanian::stringzilla::concatenate(std::forward<second_type>(second),
                                               std::forward<following_types>(following)...));
}

} // namespace stringzilla
} // namespace ashvardanian

#pragma region STL Specializations

namespace std {

template <>
struct hash<ashvardanian::stringzilla::string_view> {
    size_t operator()(ashvardanian::stringzilla::string_view str) const noexcept { return str.hash(); }
};

template <>
struct hash<ashvardanian::stringzilla::string> {
    size_t operator()(ashvardanian::stringzilla::string const &str) const noexcept { return str.hash(); }
};

} // namespace std

#pragma endregion

#endif // STRINGZILLA_HPP_
