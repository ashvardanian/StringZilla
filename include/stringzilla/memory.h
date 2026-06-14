/**
 *  @brief Hardware-accelerated memory operations.
 *  @file include/stringzilla/memory.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs for contiguous memory operations:
 *
 *  - @b `sz_copy` - analog to @b `memcpy`, probably the most common operation in a computer
 *  - @b `sz_move` - analog to @b `memmove`, allowing overlapping memory regions, often used in string manipulation
 *  - @b `sz_fill` - analog to @b `memset`, often used to initialize memory with a constant value, like zero
 *  - @b `sz_lookup` - Look-Up Table @b (LUT) transformation of a string, mapping each byte to a new value
 *  - TODO: @b `sz_lookup_utf8` - LUT transformation of a UTF8 string, which can be used for normalization
 *
 *  All of the core APIs receive the target output buffer as the first argument,
 *  and aim to minimize the number of "store" instructions, especially unaligned ones,
 *  that can invalidate 2 cache lines.
 *
 *  Unlike many other libraries focusing on trivial SIMD transformations, like converting
 *  lowercase to uppercase, StringZilla generalizes those to basic lookup table transforms.
 *  For typical ASCII conversions, you can use the following @b LUT initialization functions:
 *
 *  - `sz_lookup_init_lower` for transforms like `tolower`
 *  - `sz_lookup_init_upper` for transforms like `toupper`
 *  - `sz_lookup_init_ascii` for transforms like `isascii`
 *
 *  The header also exposes a minimalistic @b `sz_isascii` which can be used in UTF-8 capable
 *  methods to select a simpler execution path for ASCII characters.
 */
#ifndef STRINGZILLA_MEMORY_H_
#define STRINGZILLA_MEMORY_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Similar to `memcpy`, copies contents of one string into another.
 *  @see https://en.cppreference.com/w/c/string/byte/memcpy
 *
 *  @param target String to copy into. Can be `NULL`, if the @p length is zero.
 *  @param source String to copy from. Can be `NULL`, if the @p length is zero.
 *  @param length Number of bytes to copy. Can be a zero.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/memory.h>
 *      int main() {
 *          char output[2];
 *          sz_copy(output, "hi", 2);
 *          return output[0] == 'h' && output[1] == 'i' ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @pre The @p target and @p source must not overlap.
 *  @sa sz_move
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_copy_serial, sz_copy_haswell, sz_copy_skylake, sz_copy_neon, sz_copy_sve, sz_copy_v128,
 *      sz_copy_v128relaxed, sz_copy_rvv, sz_copy_lasx, sz_copy_powervsx
 */
SZ_DYNAMIC void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length);

/**
 *  @brief Similar to `memmove`, copies (moves) contents of one string into another.
 *         Unlike `sz_copy`, allows overlapping strings as arguments.
 *  @see https://en.cppreference.com/w/c/string/byte/memmove
 *
 *  @param target String to copy into. Can be `NULL`, if the @p length is zero.
 *  @param source String to copy from. Can be `NULL`, if the @p length is zero.
 *  @param length Number of bytes to copy. Can be a zero.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/memory.h>
 *      int main() {
 *          char buffer[3] = {'a', 'b', 'c'};
 *          sz_move(buffer, buffer + 1, 2);
 *          return buffer[0] == 'b' && buffer[1] == 'c' && buffer[2] == 'c' ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_move_serial, sz_move_haswell, sz_move_skylake, sz_move_neon, sz_move_sve, sz_move_v128,
 *      sz_move_v128relaxed, sz_move_rvv, sz_move_lasx, sz_move_powervsx
 */
SZ_DYNAMIC void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length);

/**
 *  @brief Similar to `memset`, fills a string with a given value.
 *  @see https://en.cppreference.com/w/c/string/byte/memset
 *
 *  @param target String to fill. Can be `NULL`, if the @p length is zero.
 *  @param length Number of bytes to fill. Can be a zero.
 *  @param value Value to fill with.
 *
 *  Example usage:
 *
 *  @code{.c}
 *     #include <stringzilla/memory.h>
 *     int main() {
 *          char buffer[2];
 *          sz_fill(buffer, 2, 'x');
 *          return buffer[0] == 'x' && buffer[1] == 'x' ? 0 : 1;
 *     }
 *  @endcode
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_fill_serial, sz_fill_haswell, sz_fill_skylake, sz_fill_neon, sz_fill_sve, sz_fill_v128,
 *      sz_fill_v128relaxed, sz_fill_rvv, sz_fill_lasx, sz_fill_powervsx
 */
SZ_DYNAMIC void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value);

/**
 *  @brief Look Up Table @b (LUT) transformation of a @p source string. Same as `for (char &c : text) c = lut[c]`.
 *  @see https://en.wikipedia.org/wiki/Lookup_table
 *
 *  Can be used to implement some form of string normalization, partially masking punctuation marks,
 *  or converting between different character sets, like uppercase or lowercase. Surprisingly, also has
 *  broad implications in image processing, where image channel transformations are often done using LUTs.
 *
 *  @param target Output string, can point to the same address as @p source.
 *  @param length Number of bytes in the string.
 *  @param source String to be mapped using the @p lut table into the @p target.
 *  @param lut Look Up Table to apply. Must be exactly @b 256 bytes long.
 *
 *  Example usage:
 *
 *  @code{.c}
 *     #include <ctype.h> // for `tolower`
 *     #include <stringzilla/memory.h>
 *     int main() {
 *          char to_lower_lut[256];
 *          for (int i = 0; i < 256; ++i) to_lower_lut[i] = tolower(i);
 *          char buffer[3] = {'A', 'B', 'C'};
 *          sz_lookup(buffer, 3, buffer, to_lower_lut);
 *          return buffer[0] == 'a' && buffer[1] == 'b' && buffer[2] == 'c' ? 0 : 1;
 *     }
 *  @endcode
 *
 *  @pre The @p lut must be exactly 256 bytes long, even if the @p source string has no characters in the top range.
 *  @pre The @p target and @p source can be the same, but must not overlap.
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_lookup_serial, sz_lookup_haswell, sz_lookup_icelake, sz_lookup_neon, sz_lookup_sve, sz_lookup_v128,
 *      sz_lookup_v128relaxed, sz_lookup_rvv, sz_lookup_lasx, sz_lookup_powervsx
 */
SZ_DYNAMIC void sz_lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]);

/** @copydoc sz_copy */
SZ_PUBLIC void sz_copy_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_move */
SZ_PUBLIC void sz_move_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_fill */
SZ_PUBLIC void sz_fill_serial(sz_ptr_t target, sz_size_t length, sz_u8_t value);
/** @copydoc sz_lookup */
SZ_PUBLIC void sz_lookup_serial(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]);

#if SZ_USE_HASWELL
/** @copydoc sz_copy */
SZ_PUBLIC void sz_copy_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_move */
SZ_PUBLIC void sz_move_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_rfind_fill */
SZ_PUBLIC void sz_fill_haswell(sz_ptr_t target, sz_size_t length, sz_u8_t value);
/** @copydoc sz_lookup */
SZ_PUBLIC void sz_lookup_haswell(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                 char const lut[sz_at_least_(256)]);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_copy */
SZ_PUBLIC void sz_copy_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_move */
SZ_PUBLIC void sz_move_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_rfind_fill */
SZ_PUBLIC void sz_fill_skylake(sz_ptr_t target, sz_size_t length, sz_u8_t value);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_lookup */
SZ_PUBLIC void sz_lookup_icelake(sz_ptr_t target, sz_size_t length, sz_cptr_t source,
                                 char const lut[sz_at_least_(256)]);
#endif

#if SZ_USE_NEON
/** @copydoc sz_copy */
SZ_PUBLIC void sz_copy_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_move */
SZ_PUBLIC void sz_move_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_rfind_fill */
SZ_PUBLIC void sz_fill_neon(sz_ptr_t target, sz_size_t length, sz_u8_t value);
/** @copydoc sz_lookup */
SZ_PUBLIC void sz_lookup_neon(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]);
#endif

#pragma endregion // Core API

#pragma region Helper API

/**
 *  @brief Initializes a lookup table for converting ASCII characters to lowercase.
 *  @param lut Lookup table to be initialized. Must be exactly 256 bytes long.
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122].
 *  So there are 26 english letters, shifted by 32 values, meaning that a conversion
 *  can be done by flipping the 5th bit each inappropriate character byte.
 *  This, however, breaks for extended ASCII, so a different solution is needed.
 *  http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 */
SZ_PUBLIC void sz_lookup_init_lower(char lut[sz_at_least_(256)]) {
    static sz_u8_t const lowered[256] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  //
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  //
        64,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, //
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 91,  92,  93,  94,  95,  //
        96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, //
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, //
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, //
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, //
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, //
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 215, 248, 249, 250, 251, 252, 253, 254, 223, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, //
    };
    for (sz_size_t byte_index = 0; byte_index < 256; ++byte_index) lut[byte_index] = lowered[byte_index];
}

/**
 *  @brief Initializes a lookup table for converting ASCII characters to uppercase.
 *  @param lut Lookup table to be initialized. Must be exactly 256 bytes long.
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122].
 *  So there are 26 english letters, shifted by 32 values, meaning that a conversion
 *  can be done by flipping the 5th bit each inappropriate character byte.
 *  This, however, breaks for extended ASCII, so a different solution is needed.
 *  http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 */
SZ_PUBLIC void sz_lookup_init_upper(char lut[sz_at_least_(256)]) {
    static sz_u8_t const upped[256] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  //
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  //
        64,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, //
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 91,  92,  93,  94,  95,  //
        96,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  //
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  123, 124, 125, 126, 127, //
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, //
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, //
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, //
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 215, 248, 249, 250, 251, 252, 253, 254, 223, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, //
    };
    for (sz_size_t byte_index = 0; byte_index < 256; ++byte_index) lut[byte_index] = upped[byte_index];
}

/**
 *  @brief Initializes a lookup table for converting bytes to ASCII characters.
 *  @param lut Lookup table to be initialized. Must be exactly 256 bytes long.
 */
SZ_PUBLIC void sz_lookup_init_ascii(char lut[sz_at_least_(256)]) {
    for (sz_size_t byte_index = 0; byte_index < 256; ++byte_index) lut[byte_index] = (sz_u8_t)(byte_index & 0x7F);
}

/**
 *  @brief Checks if all characters in a @p text are valid ASCII characters.
 *  @param text String to be analyzed.
 *  @param length Number of bytes in the string.
 *  @return Whether all characters are valid ASCII characters.
 */
SZ_PUBLIC sz_bool_t sz_isascii(sz_cptr_t text, sz_size_t length) {

    if (!length) return sz_true_k;
    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_u8_t const *const text_end = text_cursor + length;

#if !SZ_USE_MISALIGNED_LOADS
    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text_cursor & 7ull) && text_cursor < text_end; ++text_cursor)
        if (*text_cursor & 0x80ull) return sz_false_k;
#endif

    // Validate eight bytes at once using SWAR.
    sz_u64_vec_t text_vec;
    for (; text_cursor + 8 <= text_end; text_cursor += 8) {
        text_vec.u64 = *(sz_u64_t const *)text_cursor;
        if (text_vec.u64 & 0x8080808080808080ull) return sz_false_k;
    }

    // Handle the misaligned tail.
    for (; text_cursor < text_end; ++text_cursor)
        if (*text_cursor & 0x80ull) return sz_false_k;
    return sz_true_k;
}

#pragma endregion // Helper API

#include "stringzilla/memory/serial.h"
#include "stringzilla/memory/haswell.h"
#include "stringzilla/memory/skylake.h"
#include "stringzilla/memory/icelake.h"
#include "stringzilla/memory/neon.h"
#include "stringzilla/memory/sve.h"
#include "stringzilla/memory/v128relaxed.h"
#include "stringzilla/memory/v128.h"
#include "stringzilla/memory/rvv.h"
#include "stringzilla/memory/lasx.h"
#include "stringzilla/memory/powervsx.h"

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

#pragma region Core Functionality

SZ_DYNAMIC void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_V128RELAXED
    sz_copy_v128relaxed(target, source, length);
#elif SZ_USE_V128
    sz_copy_v128(target, source, length);
#elif SZ_USE_RVV
    sz_copy_rvv(target, source, length);
#elif SZ_USE_LASX
    sz_copy_lasx(target, source, length);
#elif SZ_USE_POWERVSX
    sz_copy_powervsx(target, source, length);
#elif SZ_USE_SKYLAKE
    sz_copy_skylake(target, source, length);
#elif SZ_USE_HASWELL
    sz_copy_haswell(target, source, length);
#elif SZ_USE_SVE && SZ_ENFORCE_SVE_OVER_NEON
    sz_copy_sve(target, source, length);
#elif SZ_USE_NEON
    sz_copy_neon(target, source, length);
#else
    sz_copy_serial(target, source, length);
#endif
}

SZ_DYNAMIC void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_V128RELAXED
    sz_move_v128relaxed(target, source, length);
#elif SZ_USE_V128
    sz_move_v128(target, source, length);
#elif SZ_USE_RVV
    sz_move_rvv(target, source, length);
#elif SZ_USE_LASX
    sz_move_lasx(target, source, length);
#elif SZ_USE_POWERVSX
    sz_move_powervsx(target, source, length);
#elif SZ_USE_SKYLAKE
    sz_move_skylake(target, source, length);
#elif SZ_USE_HASWELL
    sz_move_haswell(target, source, length);
#elif SZ_USE_SVE && SZ_ENFORCE_SVE_OVER_NEON
    sz_move_sve(target, source, length);
#elif SZ_USE_NEON
    sz_move_neon(target, source, length);
#else
    sz_move_serial(target, source, length);
#endif
}

SZ_DYNAMIC void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
#if SZ_USE_V128RELAXED
    sz_fill_v128relaxed(target, length, value);
#elif SZ_USE_V128
    sz_fill_v128(target, length, value);
#elif SZ_USE_RVV
    sz_fill_rvv(target, length, value);
#elif SZ_USE_LASX
    sz_fill_lasx(target, length, value);
#elif SZ_USE_POWERVSX
    sz_fill_powervsx(target, length, value);
#elif SZ_USE_SKYLAKE
    sz_fill_skylake(target, length, value);
#elif SZ_USE_HASWELL
    sz_fill_haswell(target, length, value);
#elif SZ_USE_SVE && SZ_ENFORCE_SVE_OVER_NEON
    sz_fill_sve(target, length, value);
#elif SZ_USE_NEON
    sz_fill_neon(target, length, value);
#else
    sz_fill_serial(target, length, value);
#endif
}

SZ_DYNAMIC void sz_lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[sz_at_least_(256)]) {
#if SZ_USE_V128RELAXED
    sz_lookup_v128relaxed(target, length, source, lut);
#elif SZ_USE_V128
    sz_lookup_v128(target, length, source, lut);
#elif SZ_USE_RVV
    sz_lookup_rvv(target, length, source, lut);
#elif SZ_USE_LASX
    sz_lookup_lasx(target, length, source, lut);
#elif SZ_USE_POWERVSX
    sz_lookup_powervsx(target, length, source, lut);
#elif SZ_USE_ICELAKE
    sz_lookup_icelake(target, length, source, lut);
#elif SZ_USE_HASWELL
    sz_lookup_haswell(target, length, source, lut);
#elif SZ_USE_SVE && SZ_ENFORCE_SVE_OVER_NEON
    sz_lookup_sve(target, length, source, lut);
#elif SZ_USE_NEON
    sz_lookup_neon(target, length, source, lut);
#else
    sz_lookup_serial(target, length, source, lut);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_MEMORY_H_
