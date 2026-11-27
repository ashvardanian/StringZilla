/**
 *  @file include/stringzilla/sets.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Byte sets and byte look-up tables, as data the search verbs and the containers share.
 */
#ifndef STRINGZILLA_SETS_HPP_
#define STRINGZILLA_SETS_HPP_

#include <array> // `std::array`

#include "stringzilla/types.hpp"

namespace ashvardanian {
namespace stringzilla {

template <std::size_t count_characters_>
using carray = char[count_characters_];

#pragma region Character Sets

/**
 *  @brief The concatenation of @c ascii_lowercase and @c ascii_uppercase, independent of locale.
 *  @see https://docs.python.org/3/library/string.html#string.ascii_letters
 */
inline carray<52> const &ascii_letters() noexcept {
    static carray<52> const all = {
        //
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    };
    return all;
}

/**
 *  @brief The lowercase letters "abcdefghijklmnopqrstuvwxyz". This value is not locale-dependent.
 *  @see https://docs.python.org/3/library/string.html#string.ascii_lowercase
 */
inline carray<26> const &ascii_lowercase() noexcept {
    static carray<26> const all = {
        //
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    };
    return all;
}

/**
 *  @brief The uppercase letters "ABCDEFGHIJKLMNOPQRSTUVWXYZ". This value is not locale-dependent.
 *  @see https://docs.python.org/3/library/string.html#string.ascii_uppercase
 */
inline carray<26> const &ascii_uppercase() noexcept {
    static carray<26> const all = {
        //
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    };
    return all;
}

/**
 *  @brief Printable ASCII: @c digits, @c ascii_letters, @c punctuation and @c whitespace.
 *  @see https://docs.python.org/3/library/string.html#string.printable
 */
inline carray<100> const &ascii_printables() noexcept {
    static carray<100> const all = {
        //
        '0', '1', '2', '3', '4', '5',  '6', '7', '8',  '9', 'a', 'b', 'c', 'd', 'e', 'f',  'g',  'h',  'i',  'j',
        'k', 'l', 'm', 'n', 'o', 'p',  'q', 'r', 's',  't', 'u', 'v', 'w', 'x', 'y', 'z',  'A',  'B',  'C',  'D',
        'E', 'F', 'G', 'H', 'I', 'J',  'K', 'L', 'M',  'N', 'O', 'P', 'Q', 'R', 'S', 'T',  'U',  'V',  'W',  'X',
        'Y', 'Z', '!', '"', '#', '$',  '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',  '/',  ':',  ';',  '<',
        '=', '>', '?', '@', '[', '\\', ']', '^', '_',  '`', '{', '|', '}', '~', ' ', '\t', '\n', '\r', '\f', '\v',
    };
    return all;
}

/** Non-printable ASCII control characters. Includes all codes from 0 to 31 and 127. */
inline carray<33> const &ascii_controls() noexcept {
    static carray<33> const all = {
        //
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,  16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 127,
    };
    return all;
}

/**
 *  @brief The digits "0123456789".
 *  @see https://docs.python.org/3/library/string.html#string.digits
 */
inline carray<10> const &digits() noexcept {
    static carray<10> const all = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    return all;
}

/**
 *  @brief The letters "0123456789abcdefABCDEF".
 *  @see https://docs.python.org/3/library/string.html#string.hexdigits
 */
inline carray<22> const &hexdigits() noexcept {
    static carray<22> const all = {
        //
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', //
        'a', 'b', 'c', 'd', 'e', 'f', 'A', 'B', 'C', 'D', 'E', 'F',
    };
    return all;
}

/**
 *  @brief The letters "01234567".
 *  @see https://docs.python.org/3/library/string.html#string.octdigits
 */
inline carray<8> const &octdigits() noexcept {
    static carray<8> const all = {'0', '1', '2', '3', '4', '5', '6', '7'};
    return all;
}

/**
 *  @brief The 32 ASCII punctuation characters of the C locale: every printable character that is
 *      neither a letter, a digit, nor whitespace.
 *  @see https://docs.python.org/3/library/string.html#string.punctuation
 */
inline carray<32> const &punctuation() noexcept {
    static carray<32> const all = {
        //
        '!', '"', '#', '$', '%', '&', '\'', '(',  ')', '*', '+', ',', '-', '.', '/', ':',
        ';', '<', '=', '>', '?', '@', '[',  '\\', ']', '^', '_', '`', '{', '|', '}', '~',
    };
    return all;
}

/**
 *  @brief Whitespace ASCII characters: space, tab, linefeed, return, formfeed, and vertical tab.
 *  @see https://docs.python.org/3/library/string.html#string.whitespace
 */
inline carray<6> const &whitespaces() noexcept {
    static carray<6> const all = {' ', '\t', '\n', '\r', '\f', '\v'};
    return all;
}

/**
 *  @brief ASCII characters that Python's <tt>splitlines()</tt> treats as line delimiters.
 *  @see https://docs.python.org/3/library/stdtypes.html
 */
inline carray<8> const &newlines() noexcept {
    static carray<8> const all = {'\n', '\r', '\f', '\v', '\x1C', '\x1D', '\x1E', '\x85'};
    return all;
}

/**
 *  @brief ASCII characters forming the BASE64 encoding alphabet: a-z, A-Z, 0-9, +, and /.
 *  @see https://docs.python.org/3/library/base64.html
 */
inline carray<64> const &base64() noexcept {
    static carray<64> const all = {
        //
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
    };
    return all;
}

/** A set of characters represented as a bitset with 256 slots. */
class byteset_t {
    sz_byteset_t bitset_;

  public:
    using char_type = char;

    constexpr byteset_t() noexcept { sz_byteset_init(&bitset_); }
    explicit constexpr byteset_t(std::initializer_list<char_type> chars) noexcept : byteset_t() {
        for (auto c : chars) sz_byteset_add_u8(&bitset_, sz_bitcast_(sz_u8_t, c));
    }

    explicit constexpr byteset_t(char_type const *chars, std::size_t count_characters) noexcept : byteset_t() {
        for (std::size_t i = 0; i < count_characters; ++i) sz_byteset_add_u8(&bitset_, sz_bitcast_(sz_u8_t, chars[i]));
    }

    template <std::size_t count_characters_>
    explicit constexpr byteset_t(std::array<char_type, count_characters_> const &chars) noexcept : byteset_t() {
        static_assert(count_characters_ > 0, "Character array cannot be empty");
        for (std::size_t i = 0; i < count_characters_; ++i) sz_byteset_add_u8(&bitset_, sz_bitcast_(sz_u8_t, chars[i]));
    }

    constexpr byteset_t(byteset_t const &other) noexcept : bitset_(other.bitset_) {}
    constexpr byteset_t &operator=(byteset_t const &other) noexcept {
        bitset_ = other.bitset_;
        return *this;
    }

    constexpr byteset_t operator|(byteset_t other) const noexcept {
        byteset_t result = *this;
        result.bitset_._u64s[0] |= other.bitset_._u64s[0], result.bitset_._u64s[1] |= other.bitset_._u64s[1],
            result.bitset_._u64s[2] |= other.bitset_._u64s[2], result.bitset_._u64s[3] |= other.bitset_._u64s[3];
        return result;
    }

    inline byteset_t &add(char_type c) noexcept {
        sz_byteset_add(&bitset_, sz_bitcast_(sz_u8_t, c));
        return *this;
    }
    inline std::size_t size() const noexcept {
        return                                                                      //
            sz_u64_popcount(bitset_._u64s[0]) + sz_u64_popcount(bitset_._u64s[1]) + //
            sz_u64_popcount(bitset_._u64s[2]) + sz_u64_popcount(bitset_._u64s[3]);
    }
    inline sz_byteset_t &raw() noexcept { return bitset_; }
    inline sz_byteset_t const &raw() const noexcept { return bitset_; }
    inline bool contains(char_type c) const noexcept { return sz_byteset_contains(&bitset_, sz_bitcast_(sz_u8_t, c)); }
    inline byteset_t inverted() const noexcept {
        byteset_t result = *this;
        sz_byteset_invert(&result.bitset_);
        return result;
    }
};

inline byteset_t ascii_letters_set() { return byteset_t {ascii_letters(), sizeof(ascii_letters())}; }
inline byteset_t ascii_lowercase_set() { return byteset_t {ascii_lowercase(), sizeof(ascii_lowercase())}; }
inline byteset_t ascii_uppercase_set() { return byteset_t {ascii_uppercase(), sizeof(ascii_uppercase())}; }
inline byteset_t ascii_printables_set() { return byteset_t {ascii_printables(), sizeof(ascii_printables())}; }
inline byteset_t ascii_controls_set() { return byteset_t {ascii_controls(), sizeof(ascii_controls())}; }
inline byteset_t digits_set() { return byteset_t {digits(), sizeof(digits())}; }
inline byteset_t hexdigits_set() { return byteset_t {hexdigits(), sizeof(hexdigits())}; }
inline byteset_t octdigits_set() { return byteset_t {octdigits(), sizeof(octdigits())}; }
inline byteset_t punctuation_set() { return byteset_t {punctuation(), sizeof(punctuation())}; }
inline byteset_t whitespaces_set() { return byteset_t {whitespaces(), sizeof(whitespaces())}; }
inline byteset_t newlines_set() { return byteset_t {newlines(), sizeof(newlines())}; }
inline byteset_t base64_set() { return byteset_t {base64(), sizeof(base64())}; }

/** A look-up table for byte-to-byte replacement, exactly 256 bytes. */
class look_up_table_t {
    static constexpr std::size_t size_k = 256;

    char lut_[size_k] {};

  public:
    using char_type = char;

    look_up_table_t() noexcept = default;
    explicit look_up_table_t(char_type const (&chars)[size_k]) noexcept {
        for (std::size_t i = 0; i != size_k; ++i) lut_[i] = chars[i];
    }
    look_up_table_t(std::array<char_type, size_k> const &chars) noexcept {
        for (std::size_t i = 0; i != size_k; ++i) lut_[i] = chars[i];
    }

    /**
     *  @brief Creates a look-up table with a one-to-one mapping of characters to themselves.
     *  @note Similar to @c std::iota filling, but properly handles signed integer casts.
     */
    static look_up_table_t identity() noexcept {
        look_up_table_t result;
        for (std::size_t i = 0; i < size_k; ++i) { result.lut_[i] = static_cast<char>(static_cast<unsigned char>(i)); }
        return result;
    }

    inline sz_cptr_t raw() const noexcept { return &lut_[0]; }
    inline char_type &operator[](char_type c) noexcept { return lut_[sz_bitcast_(unsigned char, c)]; }
    inline char_type const &operator[](char_type c) const noexcept { return lut_[sz_bitcast_(unsigned char, c)]; }
};

#pragma endregion

namespace literals {
inline constexpr byteset_t operator""_bs(char const *str, std::size_t length) noexcept {
    return byteset_t {str, length};
}
} // namespace literals

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SETS_HPP_
