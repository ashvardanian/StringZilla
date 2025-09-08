/**
 *  @brief  Hardware-accelerated memory operations.
 *  @file   memory.h
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

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Similar to `memcpy`, copies contents of one string into another.
 *  @see    https://en.cppreference.com/w/c/string/byte/memcpy
 *
 *  @param[out] target String to copy into. Can be `NULL`, if the @p length is zero.
 *  @param[in] length Number of bytes to copy. Can be a zero.
 *  @param[in] source String to copy from. Can be `NULL`, if the @p length is zero.
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
 *  @pre    The @p target and @p source must not overlap.
 *  @sa     sz_move
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_copy_serial, sz_copy_haswell, sz_copy_skylake, sz_copy_neon
 */
SZ_DYNAMIC void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length);

/**
 *  @brief  Similar to `memmove`, copies (moves) contents of one string into another.
 *          Unlike `sz_copy`, allows overlapping strings as arguments.
 *  @see    https://en.cppreference.com/w/c/string/byte/memmove
 *
 *  @param[out] target String to copy into. Can be `NULL`, if the @p length is zero.
 *  @param[in] length Number of bytes to copy. Can be a zero.
 *  @param[in] source String to copy from. Can be `NULL`, if the @p length is zero.
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
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_move_serial, sz_move_haswell, sz_move_skylake, sz_move_neon
 */
SZ_DYNAMIC void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length);

/**
 *  @brief  Similar to `memset`, fills a string with a given value.
 *  @see    https://en.cppreference.com/w/c/string/byte/memset
 *
 *  @param[out] target String to fill. Can be `NULL`, if the @p length is zero.
 *  @param[in] length Number of bytes to fill. Can be a zero.
 *  @param[in] value Value to fill with.
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
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_fill_serial, sz_fill_haswell, sz_fill_skylake, sz_fill_neon
 */
SZ_DYNAMIC void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value);

/**
 *  @brief  Look Up Table @b (LUT) transformation of a @p source string. Same as `for (char &c : text) c = lut[c]`.
 *  @see    https://en.wikipedia.org/wiki/Lookup_table
 *
 *  Can be used to implement some form of string normalization, partially masking punctuation marks,
 *  or converting between different character sets, like uppercase or lowercase. Surprisingly, also has
 *  broad implications in image processing, where image channel transformations are often done using LUTs.
 *
 *  @param[out] target Output string, can point to the same address as @p source.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] source String to be mapped using the @p lut table into the @p target.
 *  @param[in] lut Look Up Table to apply. Must be exactly @b 256 bytes long.
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
 *  @pre    The @p lut must be exactly 256 bytes long, even if the @p source string has no characters in the top range.
 *  @pre    The @p target and @p source can be the same, but must not overlap.
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_lookup_serial, sz_lookup_haswell, sz_lookup_ice, sz_lookup_neon
 */
SZ_DYNAMIC void sz_lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut);

/** @copydoc sz_copy */
SZ_PUBLIC void sz_copy_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_move */
SZ_PUBLIC void sz_move_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_fill */
SZ_PUBLIC void sz_fill_serial(sz_ptr_t target, sz_size_t length, sz_u8_t value);
/** @copydoc sz_lookup */
SZ_PUBLIC void sz_lookup_serial(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut);

#if SZ_USE_HASWELL
/** @copydoc sz_copy */
SZ_PUBLIC void sz_copy_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_move */
SZ_PUBLIC void sz_move_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_rfind_fill */
SZ_PUBLIC void sz_fill_haswell(sz_ptr_t target, sz_size_t length, sz_u8_t value);
/** @copydoc sz_lookup */
SZ_PUBLIC void sz_lookup_haswell(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_copy */
SZ_PUBLIC void sz_copy_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_move */
SZ_PUBLIC void sz_move_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_rfind_fill */
SZ_PUBLIC void sz_fill_skylake(sz_ptr_t target, sz_size_t length, sz_u8_t value);
#endif

#if SZ_USE_ICE
/** @copydoc sz_lookup */
SZ_PUBLIC void sz_lookup_ice(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut);
#endif

#if SZ_USE_NEON
/** @copydoc sz_copy */
SZ_PUBLIC void sz_copy_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_move */
SZ_PUBLIC void sz_move_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
/** @copydoc sz_rfind_fill */
SZ_PUBLIC void sz_fill_neon(sz_ptr_t target, sz_size_t length, sz_u8_t value);
/** @copydoc sz_lookup */
SZ_PUBLIC void sz_lookup_neon(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut);
#endif

#pragma endregion // Core API

#pragma region Helper API

/**
 *  @brief Initializes a lookup table for converting ASCII characters to lowercase.
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122].
 *  So there are 26 english letters, shifted by 32 values, meaning that a conversion
 *  can be done by flipping the 5th bit each inappropriate character byte.
 *  This, however, breaks for extended ASCII, so a different solution is needed.
 *  http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 */
SZ_PUBLIC void sz_lookup_init_lower(sz_ptr_t lut) {
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
    for (sz_size_t i = 0; i < 256; ++i) lut[i] = lowered[i];
}

/**
 *  @brief Initializes a lookup table for converting ASCII characters to uppercase.
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122].
 *  So there are 26 english letters, shifted by 32 values, meaning that a conversion
 *  can be done by flipping the 5th bit each inappropriate character byte.
 *  This, however, breaks for extended ASCII, so a different solution is needed.
 *  http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 */
SZ_PUBLIC void sz_lookup_init_upper(sz_ptr_t lut) {
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
    for (sz_size_t i = 0; i < 256; ++i) lut[i] = upped[i];
}

/**
 *  @brief Initializes a lookup table for converting bytes to ASCII characters.
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 */
SZ_PUBLIC void sz_lookup_init_ascii(sz_ptr_t lut) {
    for (sz_size_t i = 0; i < 256; ++i) lut[i] = (sz_u8_t)(i & 0x7F);
}

/**
 *  @brief Checks if all characters in a @p text are valid ASCII characters.
 *  @param[in] text String to be analyzed.
 *  @param[in] length Number of bytes in the string.
 *  @return Whether all characters are valid ASCII characters.
 */
SZ_PUBLIC sz_bool_t sz_isascii(sz_cptr_t text, sz_size_t length) {

    if (!length) return sz_true_k;
    sz_u8_t const *h = (sz_u8_t const *)text;
    sz_u8_t const *const h_end = h + length;

#if !SZ_USE_MISALIGNED_LOADS
    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h < h_end; ++h)
        if (*h & 0x80ull) return sz_false_k;
#endif

    // Validate eight bytes at once using SWAR.
    sz_u64_vec_t text_vec;
    for (; h + 8 <= h_end; h += 8) {
        text_vec.u64 = *(sz_u64_t const *)h;
        if (text_vec.u64 & 0x8080808080808080ull) return sz_false_k;
    }

    // Handle the misaligned tail.
    for (; h < h_end; ++h)
        if (*h & 0x80ull) return sz_false_k;
    return sz_true_k;
}

#pragma endregion // Helper API

#pragma region Serial Implementation

SZ_PUBLIC void sz_lookup_serial(sz_ptr_t result, sz_size_t length, sz_cptr_t text, sz_cptr_t lut) {
    sz_u8_t const *unsigned_lut = (sz_u8_t const *)lut;
    sz_u8_t const *unsigned_text = (sz_u8_t const *)text;
    sz_u8_t *unsigned_result = (sz_u8_t *)result;
    sz_u8_t const *end = unsigned_text + length;
    for (; unsigned_text != end; ++unsigned_text, ++unsigned_result) *unsigned_result = unsigned_lut[*unsigned_text];
}

// When overriding libc, disable optimizations for this function because MSVC will optimize the loops into a `memset`.
// Which then causes a stack overflow due to infinite recursion (`memset` -> `sz_fill_serial` -> `memset`).
#if defined(_MSC_VER) && defined(SZ_OVERRIDE_LIBC) && SZ_OVERRIDE_LIBC
#pragma optimize("", off)
#endif
SZ_PUBLIC void sz_fill_serial(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_ptr_t end = target + length;
    // Dealing with short strings, a single sequential pass would be faster.
    // If the size is larger than 2 words, then at least 1 of them will be aligned.
    // But just one aligned word may not be worth SWAR.
    if (length < SZ_SWAR_THRESHOLD)
        while (target != end) *(target++) = value;

    // In case of long strings, skip unaligned bytes, and then fill the rest in 64-bit chunks.
    else {
        sz_u64_t value64 = (sz_u64_t)value * 0x0101010101010101ull;
        while ((sz_size_t)target & 7ull) *(target++) = value;
        while (target + 8 <= end) *(sz_u64_t *)target = value64, target += 8;
        while (target != end) *(target++) = value;
    }
}
#if defined(_MSC_VER) && defined(SZ_OVERRIDE_LIBC) && SZ_OVERRIDE_LIBC
#pragma optimize("", on)
#endif

SZ_PUBLIC void sz_copy_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_MISALIGNED_LOADS
    while (length >= 8) *(sz_u64_t *)target = *(sz_u64_t const *)source, target += 8, source += 8, length -= 8;
#endif
    while (length--) *(target++) = *(source++);
}

SZ_PUBLIC void sz_move_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // Implementing `memmove` is trickier, than `memcpy`, as the ranges may overlap.
    // Existing implementations often have two passes, in normal and reversed order,
    // depending on the relation of `target` and `source` addresses.
    // https://student.cs.uwaterloo.ca/~cs350/common/os161-src-html/doxygen/html/memmove_8c_source.html
    // https://marmota.medium.com/c-language-making-memmove-def8792bb8d5
    //
    // We can use the `memcpy` like left-to-right pass if we know that the `target` is before `source`.
    // Or if we know that they don't intersect! In that case the traversal order is irrelevant,
    // but older CPUs may predict and fetch forward-passes better.
    if (target < source || target >= source + length) {
#if SZ_USE_MISALIGNED_LOADS
        while (length >= 8) *(sz_u64_t *)target = *(sz_u64_t const *)(source), target += 8, source += 8, length -= 8;
#endif
        while (length--) *(target++) = *(source++);
    }
    else {
        // Jump to the end and walk backwards.
        target += length, source += length;
#if SZ_USE_MISALIGNED_LOADS
        while (length >= 8) *(sz_u64_t *)(target -= 8) = *(sz_u64_t const *)(source -= 8), length -= 8;
#endif
        while (length--) *(--target) = *(--source);
    }
}

#pragma endregion

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

SZ_PUBLIC void sz_fill_haswell(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    char value_char = *(char *)&value;
    __m256i value_vec = _mm256_set1_epi8(value_char);
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "stores".
    //
    //    for (; length >= 32; target += 32, length -= 32) _mm256_storeu_si256(target, value_vec);
    //    sz_fill_serial(target, length, value);
    //
    // When the buffer is small, there isn't much to innovate.
    if (length <= 32) sz_fill_serial(target, length, value);
    // When the buffer is aligned, we can avoid any split-stores.
    else {
        sz_size_t head_length = (32 - ((sz_size_t)target % 32)) % 32; // 31 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 32;    // 31 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 32.
        sz_u16_t value16 = (sz_u16_t)value * 0x0101u;
        sz_u32_t value32 = (sz_u32_t)value16 * 0x00010001u;
        sz_u64_t value64 = (sz_u64_t)value32 * 0x0000000100000001ull;

        // Fill the head of the buffer. This part is much cleaner with AVX-512.
        if (head_length & 1) *(sz_u8_t *)target = value, target++, head_length--;
        if (head_length & 2) *(sz_u16_t *)target = value16, target += 2, head_length -= 2;
        if (head_length & 4) *(sz_u32_t *)target = value32, target += 4, head_length -= 4;
        if (head_length & 8) *(sz_u64_t *)target = value64, target += 8, head_length -= 8;
        if (head_length & 16)
            _mm_store_si128((__m128i *)target, _mm_set1_epi8(value_char)), target += 16, head_length -= 16;
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");

        // Fill the aligned body of the buffer.
        for (; body_length >= 32; target += 32, body_length -= 32) _mm256_store_si256((__m256i *)target, value_vec);

        // Fill the tail of the buffer. This part is much cleaner with AVX-512.
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");
        if (tail_length & 16)
            _mm_store_si128((__m128i *)target, _mm_set1_epi8(value_char)), target += 16, tail_length -= 16;
        if (tail_length & 8) *(sz_u64_t *)target = value64, target += 8, tail_length -= 8;
        if (tail_length & 4) *(sz_u32_t *)target = value32, target += 4, tail_length -= 4;
        if (tail_length & 2) *(sz_u16_t *)target = value16, target += 2, tail_length -= 2;
        if (tail_length & 1) *(sz_u8_t *)target = value, target++, tail_length--;
    }
}

SZ_PUBLIC void sz_copy_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "stores" and "loads".
    //
    //    for (; length >= 32; target += 32, source += 32, length -= 32)
    //        _mm256_storeu_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
    //    sz_copy_serial(target, source, length);
    //
    // A typical AWS Skylake instance can have 32 KB x 2 blocks of L1 data cache per core,
    // 1 MB x 2 blocks of L2 cache per core, and one shared L3 cache buffer.
    // For now, let's avoid the cases beyond the L2 size.
    int is_huge = length > 1ull * 1024ull * 1024ull;
    if (length < 8) {
        while (length--) *(target++) = *(source++);
    }
    // The next few sections are identical here and in the `sz_move_haswell` function.
    // We can use 2x 64-bit interleaving loads for each string, and then compare them for equality.
    // The same approach is used in GLibC and was suggest by Denis Yaroshevskiy.
    // https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/multiarch/memcmp-avx2-movbe.S.html#518
    // It shouldn't improve performance on microbenchmarks, but should be better in practice.
    else if (length <= 16) {
        sz_u64_t source_first_word = *(sz_u64_t const *)(source);
        sz_u64_t source_second_word = *(sz_u64_t const *)(source + length - 8);
        sz_u64_t *target_first_word_ptr = (sz_u64_t *)(target);
        sz_u64_t *target_second_word_ptr = (sz_u64_t *)(target + length - 8);
        *target_first_word_ptr = source_first_word;
        *target_second_word_ptr = source_second_word;
    }
    // We can use 2x 128-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 32) {
        sz_u128_vec_t source_first_vec, source_second_vec;
        sz_u128_vec_t *target_first_word_ptr, *target_second_word_ptr;
        source_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(source));
        source_second_vec.xmm = _mm_lddqu_si128((__m128i const *)(source + length - 16));
        target_first_word_ptr = (sz_u128_vec_t *)(target);
        target_second_word_ptr = (sz_u128_vec_t *)(target + length - 16);
        _mm_storeu_si128(&target_first_word_ptr->xmm, source_first_vec.xmm);
        _mm_storeu_si128(&target_second_word_ptr->xmm, source_second_vec.xmm);
    }
    // We can use 2x 256-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 64) {
        sz_u256_vec_t source_first_vec, source_second_vec;
        sz_u256_vec_t *target_first_word_ptr, *target_second_word_ptr;
        source_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source));
        source_second_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source + length - 32));
        target_first_word_ptr = (sz_u256_vec_t *)(target);
        target_second_word_ptr = (sz_u256_vec_t *)(target + length - 32);
        _mm256_storeu_si256(&target_first_word_ptr->ymm, source_first_vec.ymm);
        _mm256_storeu_si256(&target_second_word_ptr->ymm, source_second_vec.ymm);
    }
    // When dealing with larger arrays, the optimization is not as simple as with the `sz_fill_haswell` function,
    // as both buffers may be unaligned. If we are lucky and the requested operation is some huge page transfer,
    // we can use aligned loads and stores, and the performance will be great.
    else if ((sz_size_t)target % 32 == 0 && (sz_size_t)source % 32 == 0 && !is_huge) {
        for (; length >= 32; target += 32, source += 32, length -= 32)
            _mm256_store_si256((__m256i *)target, _mm256_load_si256((__m256i const *)source));
        if (length) sz_copy_serial(target, source, length);
    }
    // The trickiest case is when both `source` and `target` are not aligned.
    // In such and simpler cases we can copy enough bytes into `target` to reach its cacheline boundary,
    // and then combine unaligned loads with aligned stores.
    else {
        sz_size_t head_length = (32 - ((sz_size_t)target % 32)) % 32; // 31 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 32;    // 31 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 32.

        // Fill the head of the buffer. This part is much cleaner with AVX-512.
        if (head_length & 1) *(sz_u8_t *)target = *(sz_u8_t *)source, target++, source++, head_length--;
        if (head_length & 2) *(sz_u16_t *)target = *(sz_u16_t *)source, target += 2, source += 2, head_length -= 2;
        if (head_length & 4) *(sz_u32_t *)target = *(sz_u32_t *)source, target += 4, source += 4, head_length -= 4;
        if (head_length & 8) *(sz_u64_t *)target = *(sz_u64_t *)source, target += 8, source += 8, head_length -= 8;
        if (head_length & 16)
            _mm_store_si128((__m128i *)target, _mm_lddqu_si128((__m128i const *)source)), target += 16, source += 16,
                head_length -= 16;
        sz_assert_(head_length == 0 && "The head length should be zero after the head copy.");
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");

        // Fill the aligned body of the buffer.
        if (!is_huge) {
            for (; body_length >= 32; target += 32, source += 32, body_length -= 32)
                _mm256_store_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
        }
        // When the buffer is huge, we can traverse it in 2 directions.
        else {
            size_t tails_bytes_skipped = 0;
            for (; body_length >= 64; target += 32, source += 32, body_length -= 64, tails_bytes_skipped += 32) {
                _mm256_store_si256((__m256i *)(target), _mm256_lddqu_si256((__m256i const *)(source)));
                _mm256_store_si256((__m256i *)(target + body_length - 32),
                                   _mm256_lddqu_si256((__m256i const *)(source + body_length - 32)));
            }
            if (body_length) {
                sz_assert_(body_length == 32 && "The only remaining body length should be 32 bytes.");
                _mm256_store_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
                target += 32, source += 32, body_length -= 32;
            }
            target += tails_bytes_skipped;
            source += tails_bytes_skipped;
        }

        // Fill the tail of the buffer. This part is much cleaner with AVX-512.
        sz_assert_((sz_size_t)target % 32 == 0 && "Target is supposed to be aligned to the YMM register size.");
        if (tail_length & 16)
            _mm_store_si128((__m128i *)target, _mm_lddqu_si128((__m128i const *)source)), target += 16, source += 16,
                tail_length -= 16;
        if (tail_length & 8) *(sz_u64_t *)target = *(sz_u64_t *)source, target += 8, source += 8, tail_length -= 8;
        if (tail_length & 4) *(sz_u32_t *)target = *(sz_u32_t *)source, target += 4, source += 4, tail_length -= 4;
        if (tail_length & 2) *(sz_u16_t *)target = *(sz_u16_t *)source, target += 2, source += 2, tail_length -= 2;
        if (tail_length & 1) *(sz_u8_t *)target = *(sz_u8_t *)source, target++, source++, tail_length--;
        sz_assert_(tail_length == 0 && "The tail length should be zero after the tail copy.");
    }
}

SZ_PUBLIC void sz_move_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {

    if (length < 8) {
        if (target < source)
            while (length--) *(target++) = *(source++);
        else {
            // Jump to the end and walk backwards:
            target += length, source += length;
            while (length--) *(--target) = *(--source);
        }
    }
    // The next few sections are identical here and in the `sz_copy_haswell` function.
    // We can use 2x 64-bit interleaving loads for each string, and then compare them for equality.
    // The same approach is used in GLibC and was suggest by Denis Yaroshevskiy.
    // https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/multiarch/memcmp-avx2-movbe.S.html#518
    // It shouldn't improve performance on microbenchmarks, but should be better in practice.
    else if (length <= 16) {
        sz_u64_t source_first_word = *(sz_u64_t const *)(source);
        sz_u64_t source_second_word = *(sz_u64_t const *)(source + length - 8);
        sz_u64_t *target_first_word_ptr = (sz_u64_t *)(target);
        sz_u64_t *target_second_word_ptr = (sz_u64_t *)(target + length - 8);
        *target_first_word_ptr = source_first_word;
        *target_second_word_ptr = source_second_word;
    }
    // We can use 2x 128-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 32) {
        sz_u128_vec_t source_first_vec, source_second_vec;
        sz_u128_vec_t *target_first_word_ptr, *target_second_word_ptr;
        source_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(source));
        source_second_vec.xmm = _mm_lddqu_si128((__m128i const *)(source + length - 16));
        target_first_word_ptr = (sz_u128_vec_t *)(target);
        target_second_word_ptr = (sz_u128_vec_t *)(target + length - 16);
        _mm_storeu_si128(&target_first_word_ptr->xmm, source_first_vec.xmm);
        _mm_storeu_si128(&target_second_word_ptr->xmm, source_second_vec.xmm);
    }
    // We can use 2x 256-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 64) {
        sz_u256_vec_t source_first_vec, source_second_vec;
        sz_u256_vec_t *target_first_word_ptr, *target_second_word_ptr;
        source_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source));
        source_second_vec.ymm = _mm256_lddqu_si256((__m256i const *)(source + length - 32));
        target_first_word_ptr = (sz_u256_vec_t *)(target);
        target_second_word_ptr = (sz_u256_vec_t *)(target + length - 32);
        _mm256_storeu_si256(&target_first_word_ptr->ymm, source_first_vec.ymm);
        _mm256_storeu_si256(&target_second_word_ptr->ymm, source_second_vec.ymm);
    }
    // When dealing with larger arrays, we keep things simple:
    else if (target < source || target >= source + length) {
        for (; length >= 32; target += 32, source += 32, length -= 32)
            _mm256_storeu_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
        while (length--) *(target++) = *(source++);
    }
    else {
        // Jump to the end and walk backwards:
        for (target += length, source += length; length >= 32; length -= 32)
            _mm256_storeu_si256((__m256i *)(target -= 32), _mm256_lddqu_si256((__m256i const *)(source -= 32)));
        while (length--) *(--target) = *(--source);
    }
}

SZ_PUBLIC void sz_lookup_haswell(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut) {

    // If the input is tiny (especially smaller than the look-up table itself), we may end up paying
    // more for organizing the SIMD registers and changing the CPU state, than for the actual computation.
    // But if at least 3 cache lines are touched, the AVX-2 implementation should be faster.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    // We need to pull the lookup table into 8x YMM registers.
    // The biggest issue is reorganizing the data in the lookup table, as AVX2 doesn't have 256-bit shuffle,
    // it only has 128-bit "within-lane" shuffle. Still, it's wiser to use full YMM registers, instead of XMM,
    // so that we can at least compensate high latency with twice larger window and one more level of lookup.
    sz_u256_vec_t lut_0_to_15_vec, lut_16_to_31_vec, lut_32_to_47_vec, lut_48_to_63_vec, //
        lut_64_to_79_vec, lut_80_to_95_vec, lut_96_to_111_vec, lut_112_to_127_vec,       //
        lut_128_to_143_vec, lut_144_to_159_vec, lut_160_to_175_vec, lut_176_to_191_vec,  //
        lut_192_to_207_vec, lut_208_to_223_vec, lut_224_to_239_vec, lut_240_to_255_vec;

    lut_0_to_15_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut)));
    lut_16_to_31_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 16)));
    lut_32_to_47_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 32)));
    lut_48_to_63_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 48)));
    lut_64_to_79_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 64)));
    lut_80_to_95_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 80)));
    lut_96_to_111_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 96)));
    lut_112_to_127_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 112)));
    lut_128_to_143_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 128)));
    lut_144_to_159_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 144)));
    lut_160_to_175_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 160)));
    lut_176_to_191_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 176)));
    lut_192_to_207_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 192)));
    lut_208_to_223_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 208)));
    lut_224_to_239_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 224)));
    lut_240_to_255_vec.ymm = _mm256_broadcastsi128_si256(_mm_lddqu_si128((__m128i const *)(lut + 240)));

    // Assuming each lookup is performed within 16 elements of 256, we need to reduce the scope by 16x = 2^4.
    sz_u256_vec_t not_first_bit_vec, not_second_bit_vec, not_third_bit_vec, not_fourth_bit_vec;

    /// Top and bottom nibbles of the source are used separately.
    sz_u256_vec_t source_vec, source_bot_vec;
    sz_u256_vec_t blended_0_to_31_vec, blended_32_to_63_vec, blended_64_to_95_vec, blended_96_to_127_vec,
        blended_128_to_159_vec, blended_160_to_191_vec, blended_192_to_223_vec, blended_224_to_255_vec;

    // Handling the head.
    while (length >= 32) {
        // Load and separate the nibbles of each byte in the source.
        source_vec.ymm = _mm256_lddqu_si256((__m256i const *)source);
        source_bot_vec.ymm = _mm256_and_si256(source_vec.ymm, _mm256_set1_epi8((char)0x0F));

        // In the first round, we select using the 4th bit.
        not_fourth_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x10), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8(                      //
            _mm256_shuffle_epi8(lut_16_to_31_vec.ymm, source_bot_vec.ymm), //
            _mm256_shuffle_epi8(lut_0_to_15_vec.ymm, source_bot_vec.ymm),  //
            not_fourth_bit_vec.ymm);
        blended_32_to_63_vec.ymm = _mm256_blendv_epi8(                     //
            _mm256_shuffle_epi8(lut_48_to_63_vec.ymm, source_bot_vec.ymm), //
            _mm256_shuffle_epi8(lut_32_to_47_vec.ymm, source_bot_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_64_to_95_vec.ymm = _mm256_blendv_epi8(                     //
            _mm256_shuffle_epi8(lut_80_to_95_vec.ymm, source_bot_vec.ymm), //
            _mm256_shuffle_epi8(lut_64_to_79_vec.ymm, source_bot_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_96_to_127_vec.ymm = _mm256_blendv_epi8(                      //
            _mm256_shuffle_epi8(lut_112_to_127_vec.ymm, source_bot_vec.ymm), //
            _mm256_shuffle_epi8(lut_96_to_111_vec.ymm, source_bot_vec.ymm),  //
            not_fourth_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8(                     //
            _mm256_shuffle_epi8(lut_144_to_159_vec.ymm, source_bot_vec.ymm), //
            _mm256_shuffle_epi8(lut_128_to_143_vec.ymm, source_bot_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_160_to_191_vec.ymm = _mm256_blendv_epi8(                     //
            _mm256_shuffle_epi8(lut_176_to_191_vec.ymm, source_bot_vec.ymm), //
            _mm256_shuffle_epi8(lut_160_to_175_vec.ymm, source_bot_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_192_to_223_vec.ymm = _mm256_blendv_epi8(                     //
            _mm256_shuffle_epi8(lut_208_to_223_vec.ymm, source_bot_vec.ymm), //
            _mm256_shuffle_epi8(lut_192_to_207_vec.ymm, source_bot_vec.ymm), //
            not_fourth_bit_vec.ymm);
        blended_224_to_255_vec.ymm = _mm256_blendv_epi8(                     //
            _mm256_shuffle_epi8(lut_240_to_255_vec.ymm, source_bot_vec.ymm), //
            _mm256_shuffle_epi8(lut_224_to_239_vec.ymm, source_bot_vec.ymm), //
            not_fourth_bit_vec.ymm);

        // Perform a tree-like reduction of the 8x "blended" YMM registers, depending on the "source" content.
        // The first round selects using the 3rd bit.
        not_third_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x20), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_32_to_63_vec.ymm,                 //
            blended_0_to_31_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_64_to_95_vec.ymm = _mm256_blendv_epi8( //
            blended_96_to_127_vec.ymm,                 //
            blended_64_to_95_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8( //
            blended_160_to_191_vec.ymm,                  //
            blended_128_to_159_vec.ymm,                  //
            not_third_bit_vec.ymm);
        blended_192_to_223_vec.ymm = _mm256_blendv_epi8( //
            blended_224_to_255_vec.ymm,                  //
            blended_192_to_223_vec.ymm,                  //
            not_third_bit_vec.ymm);

        // The second round selects using the 2nd bit.
        not_second_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x40), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_64_to_95_vec.ymm,                 //
            blended_0_to_31_vec.ymm,                  //
            not_second_bit_vec.ymm);
        blended_128_to_159_vec.ymm = _mm256_blendv_epi8( //
            blended_192_to_223_vec.ymm,                  //
            blended_128_to_159_vec.ymm,                  //
            not_second_bit_vec.ymm);

        // The third round selects using the 1st bit.
        not_first_bit_vec.ymm = _mm256_cmpeq_epi8( //
            _mm256_and_si256(_mm256_set1_epi8((char)0x80), source_vec.ymm), _mm256_setzero_si256());
        blended_0_to_31_vec.ymm = _mm256_blendv_epi8( //
            blended_128_to_159_vec.ymm,               //
            blended_0_to_31_vec.ymm,                  //
            not_first_bit_vec.ymm);

        // And dump the result into the target.
        _mm256_storeu_si256((__m256i *)target, blended_0_to_31_vec.ymm);
        source += 32, target += 32, length -= 32;
    }

    // Handle the tail.
    if (length) sz_lookup_serial(target, length, source, lut);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string search algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation

#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

SZ_PUBLIC void sz_fill_skylake(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    __m512i value_vec = _mm512_set1_epi8(value);
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "stores".
    //
    //    for (; length >= 64; target += 64, length -= 64) _mm512_storeu_si512(target, value_vec);
    //    _mm512_mask_storeu_epi8(target, sz_u64_mask_until_(length), value_vec);
    //
    // When the buffer is small, there isn't much to innovate.
    if (length <= 64) {
        __mmask64 mask = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask, value_vec);
    }
    // When the buffer is over 64 bytes, it's guaranteed to touch at least two cache lines - the head and tail,
    // and may include more cache-lines in-between. Knowing this, we can avoid expensive unaligned stores
    // by computing 2 masks - for the head and tail, using masked stores for the head and tail, and unmasked
    // for the body.
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 64.
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask, value_vec);
        for (target += head_length; body_length >= 64; target += 64, body_length -= 64)
            _mm512_store_si512(target, value_vec);
        _mm512_mask_storeu_epi8(target, tail_mask, value_vec);
    }
}

SZ_PUBLIC void sz_copy_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "stores" and "loads".
    //
    //    for (; length >= 64; target += 64, source += 64, length -= 64)
    //        _mm512_storeu_si512(target, _mm512_loadu_si512(source));
    //    __mmask64 mask = sz_u64_mask_until_(length);
    //    _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    //
    // A typical AWS Sapphire Rapids instance can have 48 KB x 2 blocks of L1 data cache per core,
    // 2 MB x 2 blocks of L2 cache per core, and one shared 60 MB buffer of L3 cache.
    // With two strings, we may consider the overal workload huge, if each exceeds 1 MB in length.
    int const is_huge = length >= 1ull * 1024ull * 1024ull;

    // When the buffer is small, there isn't much to innovate.
    if (length <= 64) {
        __mmask64 mask = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    }
    // When dealing with larger arrays, the optimization is not as simple as with the `sz_fill_skylake` function,
    // as both buffers may be unaligned. If we are lucky and the requested operation is some huge page transfer,
    // we can use aligned loads and stores, and the performance will be great.
    else if ((sz_size_t)target % 64 == 0 && (sz_size_t)source % 64 == 0 && !is_huge) {
        for (; length >= 64; target += 64, source += 64, length -= 64)
            _mm512_store_si512(target, _mm512_load_si512(source));
        // At this point the length is guaranteed to be under 64.
        __mmask64 mask = sz_u64_mask_until_(length);
        // Aligned load and stores would work too, but it's not defined.
        _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    }
    // The trickiest case is when both `source` and `target` are not aligned.
    // In such and simpler cases we can copy enough bytes into `target` to reach its cacheline boundary,
    // and then combine unaligned loads with aligned stores.
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 64.
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
        for (target += head_length, source += head_length; body_length >= 64;
             target += 64, source += 64, body_length -= 64)
            _mm512_store_si512(target, _mm512_loadu_si512(source)); // Unaligned load, but aligned store!
        _mm512_mask_storeu_epi8(target, tail_mask, _mm512_maskz_loadu_epi8(tail_mask, source));
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    //
    //      1. Moving in both directions to maximize the throughput, when fetching from multiple
    //         memory pages. Also helps with cache set-associativity issues, as we won't always
    //         be fetching the same entries in the lookup table.
    //      2. Using non-temporal stores to avoid polluting the cache.
    //      3. Prefetching the next cache line, to avoid stalling the CPU. This generally useless
    //         for predictable patterns, so disregard this advice.
    //
    // Bidirectional traversal adds about 10%, accelerating from 11 GB/s to 12 GB/s.
    // Using "streaming stores" boosts us from 12 GB/s to 19 GB/s.
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
        _mm512_mask_storeu_epi8(target + head_length + body_length, tail_mask,
                                _mm512_maskz_loadu_epi8(tail_mask, source + head_length + body_length));

        // Now in the main loop, we can use non-temporal loads and stores,
        // performing the operation in both directions.
        for (target += head_length, source += head_length; //
             body_length >= 128;                           //
             target += 64, source += 64, body_length -= 128) {
            _mm512_stream_si512((__m512i *)(target), _mm512_loadu_si512(source));
            _mm512_stream_si512((__m512i *)(target + body_length - 64), _mm512_loadu_si512(source + body_length - 64));
        }
        if (body_length >= 64) _mm512_stream_si512((__m512i *)target, _mm512_loadu_si512(source));
    }
}

SZ_PUBLIC void sz_move_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target == source) return; // Don't be silly, don't move the data if it's already there.

    // On very short buffers, that are one cache line in width or less, we don't need any loops.
    // We can also avoid any data-dependencies between iterations, assuming we have 32 registers
    // to pre-load the data, before writing it back.
    if (length <= 64) {
        __mmask64 mask = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    }
    else if (length <= 128) {
        sz_size_t last_length = length - 64;
        __mmask64 mask = sz_u64_mask_until_(last_length);
        __m512i source0 = _mm512_loadu_epi8(source);
        __m512i source1 = _mm512_maskz_loadu_epi8(mask, source + 64);
        _mm512_storeu_epi8(target, source0);
        _mm512_mask_storeu_epi8(target + 64, mask, source1);
    }
    else if (length <= 192) {
        sz_size_t last_length = length - 128;
        __mmask64 mask = sz_u64_mask_until_(last_length);
        __m512i source0 = _mm512_loadu_epi8(source);
        __m512i source1 = _mm512_loadu_epi8(source + 64);
        __m512i source2 = _mm512_maskz_loadu_epi8(mask, source + 128);
        _mm512_storeu_epi8(target, source0);
        _mm512_storeu_epi8(target + 64, source1);
        _mm512_mask_storeu_epi8(target + 128, mask, source2);
    }
    else if (length <= 256) {
        sz_size_t last_length = length - 192;
        __mmask64 mask = sz_u64_mask_until_(last_length);
        __m512i source0 = _mm512_loadu_epi8(source);
        __m512i source1 = _mm512_loadu_epi8(source + 64);
        __m512i source2 = _mm512_loadu_epi8(source + 128);
        __m512i source3 = _mm512_maskz_loadu_epi8(mask, source + 192);
        _mm512_storeu_epi8(target, source0);
        _mm512_storeu_epi8(target + 64, source1);
        _mm512_storeu_epi8(target + 128, source2);
        _mm512_mask_storeu_epi8(target + 192, mask, source3);
    }

    // If the regions don't overlap at all, just use "copy" and save some brain cells thinking about corner cases.
    else if (target + length < source || target >= source + length) { sz_copy_skylake(target, source, length); }

    // When the buffer is over 64 bytes, it's guaranteed to touch at least two cache lines - the head and tail,
    // and may include more cache-lines in-between. Knowing this, we can avoid expensive unaligned stores
    // by computing 2 masks - for the head and tail, using masked stores for the head and tail, and unmasked
    // for the body.
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 64.
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

        // The absolute most common case of using "moves" is shifting the data within a continuous buffer
        // when adding a removing some values in it. In such cases, a typical shift is by 1, 2, 4, 8, 16,
        // or 32 bytes, rarely larger. For small shifts, under the size of the ZMM register, we can use shuffles.
        //
        // Remember:
        //      - if we are shifting data left, that we are traversing to the right.
        //      - if we are shifting data right, that we are traversing to the left.
        int const left_to_right_traversal = source > target;

        // Now we guarantee, that the relative shift within registers is from 1 to 63 bytes and the output is aligned.
        // Hopefully, we need to shift more than two ZMM registers, so we could consider `valignr` instruction.
        // Sadly, using `_mm512_alignr_epi8` doesn't make sense, as it operates at a 128-bit granularity.
        //
        //      - `_mm256_alignr_epi8` shifts entire 256-bit register, but we need many of them.
        //      - `_mm512_alignr_epi32` shifts 512-bit chunks, but only if the `shift` is a multiple of 4 bytes.
        //      - `_mm512_alignr_epi64` shifts 512-bit chunks by 8 bytes.
        //
        // All of those have a latency of 1 cycle, and the shift amount must be an immediate value!
        // For 1-byte-shift granularity, the `_mm512_permutex2var_epi8` has a latency of 6 and needs VBMI!
        // The most efficient and broadly compatible alternative could be to use a combination of align and shuffle.
        // A similar approach was outlined in "Byte-wise alignr in AVX512F" by Wojciech Muła.
        // http://0x80.pl/notesen/2016-10-16-avx512-byte-alignr.html
        //
        // That solution, is extremely mouthful, assuming we need compile time constants for the shift amount.
        // A cleaner one, with a latency of 3 cycles, is to use `_mm512_permutexvar_epi8` or
        // `_mm512_mask_permutexvar_epi8`, which can be seen as combination of a cross-register shuffle and blend,
        // and is available with VBMI. That solution is still noticeably slower than AVX2.
        //
        // The GLibC implementation also uses non-temporal stores for larger buffers, we don't.
        // https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/multiarch/memmove-avx512-no-vzeroupper.S.html
        if (left_to_right_traversal) {
            // Head, body, and tail.
            _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
            for (target += head_length, source += head_length; body_length >= 64;
                 target += 64, source += 64, body_length -= 64)
                _mm512_store_si512(target, _mm512_loadu_si512(source));
            _mm512_mask_storeu_epi8(target, tail_mask, _mm512_maskz_loadu_epi8(tail_mask, source));
        }
        else {
            // Tail, body, and head.
            _mm512_mask_storeu_epi8(target + head_length + body_length, tail_mask,
                                    _mm512_maskz_loadu_epi8(tail_mask, source + head_length + body_length));
            for (; body_length >= 64; body_length -= 64)
                _mm512_store_si512(target + head_length + body_length - 64,
                                   _mm512_loadu_si512(source + head_length + body_length - 64));
            _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#endif

SZ_PUBLIC void sz_lookup_ice(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut) {

    // If the input is tiny (especially smaller than the look-up table itself), we may end up paying
    // more for organizing the SIMD registers and changing the CPU state, than for the actual computation.
    // But if at least 3 cache lines are touched, the AVX-512 implementation should be faster.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    // When the buffer is over 64 bytes, it's guaranteed to touch at least two cache lines - the head and tail,
    // and may include more cache-lines in-between. Knowing this, we can avoid expensive unaligned stores
    // by computing 2 masks - for the head and tail, using masked stores for the head and tail, and unmasked
    // for the body.
    sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
    sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
    __mmask64 head_mask = sz_u64_mask_until_(head_length);
    __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

    // We need to pull the lookup table into 4x ZMM registers.
    // We can use `vpermi2b` instruction to perform the look in two ZMM registers with `_mm512_permutex2var_epi8`
    // intrinsics, but it has a 6-cycle latency on Sapphire Rapids and requires AVX512-VBMI. Assuming we need to
    // operate on 4 registers, it might be cleaner to use 2x separate `_mm512_permutexvar_epi8` calls.
    // Combining the results with 2x `_mm512_test_epi8_mask` and 3x blends afterwards.
    //
    //  - 4x `_mm512_permutexvar_epi8` maps to "VPERMB (ZMM, ZMM, ZMM)":
    //      - On Ice Lake: 3 cycles latency, ports: 1*p5
    //      - On Genoa: 6 cycles latency, ports: 1*FP12
    //  - 3x `_mm512_mask_blend_epi8` maps to "VPBLENDMB_Z (ZMM, K, ZMM, ZMM)":
    //      - On Ice Lake: 3 cycles latency, ports: 1*p05
    //      - On Genoa: 1 cycle latency, ports: 1*FP0123
    //  - 2x `_mm512_test_epi8_mask` maps to "VPTESTMB (K, ZMM, ZMM)":
    //      - On Ice Lake: 3 cycles latency, ports: 1*p5
    //      - On Genoa: 4 cycles latency, ports: 1*FP01
    //
    sz_u512_vec_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec.zmm = _mm512_loadu_si512((lut));
    lut_64_to_127_vec.zmm = _mm512_loadu_si512((lut + 64));
    lut_128_to_191_vec.zmm = _mm512_loadu_si512((lut + 128));
    lut_192_to_255_vec.zmm = _mm512_loadu_si512((lut + 192));

    sz_u512_vec_t first_bit_vec, second_bit_vec;
    first_bit_vec.zmm = _mm512_set1_epi8((char)0x80);
    second_bit_vec.zmm = _mm512_set1_epi8((char)0x40);

    __mmask64 first_bit_mask, second_bit_mask;
    sz_u512_vec_t source_vec;
    // If the top bit is set in each word of `source_vec`, than we use `lookup_128_to_191_vec` or
    // `lookup_192_to_255_vec`. If the second bit is set, we use `lookup_64_to_127_vec` or `lookup_192_to_255_vec`.
    sz_u512_vec_t lookup_0_to_63_vec, lookup_64_to_127_vec, lookup_128_to_191_vec, lookup_192_to_255_vec;
    sz_u512_vec_t blended_0_to_127_vec, blended_128_to_255_vec, blended_0_to_255_vec;

    // Handling the head.
    if (head_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, source);
        lookup_0_to_63_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_0_to_63_vec.zmm);
        lookup_64_to_127_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_64_to_127_vec.zmm);
        lookup_128_to_191_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_128_to_191_vec.zmm);
        lookup_192_to_255_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_192_to_255_vec.zmm);
        first_bit_mask = _mm512_test_epi8_mask(source_vec.zmm, first_bit_vec.zmm);
        second_bit_mask = _mm512_test_epi8_mask(source_vec.zmm, second_bit_vec.zmm);
        blended_0_to_127_vec.zmm =
            _mm512_mask_blend_epi8(second_bit_mask, lookup_0_to_63_vec.zmm, lookup_64_to_127_vec.zmm);
        blended_128_to_255_vec.zmm =
            _mm512_mask_blend_epi8(second_bit_mask, lookup_128_to_191_vec.zmm, lookup_192_to_255_vec.zmm);
        blended_0_to_255_vec.zmm =
            _mm512_mask_blend_epi8(first_bit_mask, blended_0_to_127_vec.zmm, blended_128_to_255_vec.zmm);
        _mm512_mask_storeu_epi8(target, head_mask, blended_0_to_255_vec.zmm);
        source += head_length, target += head_length, length -= head_length;
    }

    // Handling the body in 64-byte chunks aligned to cache-line boundaries with respect to `target`.
    while (length >= 64) {
        source_vec.zmm = _mm512_loadu_si512(source);
        lookup_0_to_63_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_0_to_63_vec.zmm);
        lookup_64_to_127_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_64_to_127_vec.zmm);
        lookup_128_to_191_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_128_to_191_vec.zmm);
        lookup_192_to_255_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_192_to_255_vec.zmm);
        first_bit_mask = _mm512_test_epi8_mask(source_vec.zmm, first_bit_vec.zmm);
        second_bit_mask = _mm512_test_epi8_mask(source_vec.zmm, second_bit_vec.zmm);
        blended_0_to_127_vec.zmm =
            _mm512_mask_blend_epi8(second_bit_mask, lookup_0_to_63_vec.zmm, lookup_64_to_127_vec.zmm);
        blended_128_to_255_vec.zmm =
            _mm512_mask_blend_epi8(second_bit_mask, lookup_128_to_191_vec.zmm, lookup_192_to_255_vec.zmm);
        blended_0_to_255_vec.zmm =
            _mm512_mask_blend_epi8(first_bit_mask, blended_0_to_127_vec.zmm, blended_128_to_255_vec.zmm);
        _mm512_store_si512(target, blended_0_to_255_vec.zmm); //! Aligned store, our main weapon!
        source += 64, target += 64, length -= 64;
    }

    // Handling the tail.
    if (tail_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask, source);
        lookup_0_to_63_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_0_to_63_vec.zmm);
        lookup_64_to_127_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_64_to_127_vec.zmm);
        lookup_128_to_191_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_128_to_191_vec.zmm);
        lookup_192_to_255_vec.zmm = _mm512_permutexvar_epi8(source_vec.zmm, lut_192_to_255_vec.zmm);
        first_bit_mask = _mm512_test_epi8_mask(source_vec.zmm, first_bit_vec.zmm);
        second_bit_mask = _mm512_test_epi8_mask(source_vec.zmm, second_bit_vec.zmm);
        blended_0_to_127_vec.zmm =
            _mm512_mask_blend_epi8(second_bit_mask, lookup_0_to_63_vec.zmm, lookup_64_to_127_vec.zmm);
        blended_128_to_255_vec.zmm =
            _mm512_mask_blend_epi8(second_bit_mask, lookup_128_to_191_vec.zmm, lookup_192_to_255_vec.zmm);
        blended_0_to_255_vec.zmm =
            _mm512_mask_blend_epi8(first_bit_mask, blended_0_to_127_vec.zmm, blended_128_to_255_vec.zmm);
        _mm512_mask_storeu_epi8(target, tail_mask, blended_0_to_255_vec.zmm);
        source += tail_length, target += tail_length, length -= tail_length;
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Implementation of the string search algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#endif

SZ_PUBLIC void sz_copy_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // In most cases the `source` and the `target` are not aligned, but we should
    // at least make sure that writes don't touch many cache lines.
    // NEON has an instruction to load and write 64 bytes at once.
    //
    //    sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
    //    sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
    //    for (; head_length; target += 1, source += 1, head_length -= 1) *target = *source;
    //    length -= head_length;
    //    for (; length >= 64; target += 64, source += 64, length -= 64)
    //        vst4q_u8((sz_u8_t *)target, vld1q_u8_x4((sz_u8_t const *)source));
    //    for (; tail_length; target += 1, source += 1, tail_length -= 1) *target = *source;
    //
    // Sadly, those instructions end up being 20% slower than the code processing 16 bytes at a time:
    for (; length >= 16; target += 16, source += 16, length -= 16)
        vst1q_u8((sz_u8_t *)target, vld1q_u8((sz_u8_t const *)source));
    if (length) sz_copy_serial(target, source, length);
}

SZ_PUBLIC void sz_move_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // When moving small buffers, using a small buffer on stack as a temporary storage is faster.

    if (target < source || target >= source + length) {
        // Non-overlapping, proceed forward
        sz_copy_neon(target, source, length);
    }
    else {
        // Overlapping, proceed backward
        target += length;
        source += length;

        sz_u128_vec_t src_vec;
        while (length >= 16) {
            target -= 16, source -= 16, length -= 16;
            src_vec.u8x16 = vld1q_u8((sz_u8_t const *)source);
            vst1q_u8((sz_u8_t *)target, src_vec.u8x16);
        }
        while (length) {
            target -= 1, source -= 1, length -= 1;
            *target = *source;
        }
    }
}

SZ_PUBLIC void sz_fill_neon(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    uint8x16_t fill_vec = vdupq_n_u8(value); // Broadcast the value across the register

    while (length >= 16) {
        vst1q_u8((sz_u8_t *)target, fill_vec);
        target += 16;
        length -= 16;
    }

    // Handle remaining bytes
    if (length) sz_fill_serial(target, length, value);
}

SZ_PUBLIC void sz_lookup_neon(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut) {

    // If the input is tiny (especially smaller than the look-up table itself), we may end up paying
    // more for organizing the SIMD registers and changing the CPU state, than for the actual computation.
    if (length <= 128) {
        sz_lookup_serial(target, length, source, lut);
        return;
    }

    sz_size_t head_length = (16 - ((sz_size_t)target % 16)) % 16; // 15 or less.
    sz_size_t tail_length = (sz_size_t)(target + length) % 16;    // 15 or less.

    // We need to pull the lookup table into 16x NEON registers. We have a total of 32 such registers.
    // According to the Neoverse V2 manual, the 4-table lookup has a latency of 6 cycles, and 4x throughput.
    uint8x16x4_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 0));
    lut_64_to_127_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 64));
    lut_128_to_191_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 128));
    lut_192_to_255_vec = vld1q_u8_x4((sz_u8_t const *)(lut + 192));

    sz_u128_vec_t source_vec;
    // If the top bit is set in each word of `source_vec`, than we use `lookup_128_to_191_vec` or
    // `lookup_192_to_255_vec`. If the second bit is set, we use `lookup_64_to_127_vec` or `lookup_192_to_255_vec`.
    sz_u128_vec_t lookup_0_to_63_vec, lookup_64_to_127_vec, lookup_128_to_191_vec, lookup_192_to_255_vec;
    sz_u128_vec_t blended_0_to_255_vec;

    // Process the head with serial code
    for (; head_length; target += 1, source += 1, head_length -= 1) *target = lut[*(sz_u8_t const *)source];

    // Table lookups on Arm are much simpler to use than on x86, as we can use the `vqtbl4q_u8` instruction
    // to perform a 4-table lookup in a single instruction. The XORs are used to adjust the lookup position
    // within each 64-byte range of the table.
    // Details on the 4-table lookup: https://lemire.me/blog/2019/07/23/arbitrary-byte-to-byte-maps-using-arm-neon/
    length -= head_length;
    length -= tail_length;
    for (; length >= 16; source += 16, target += 16, length -= 16) {
        source_vec.u8x16 = vld1q_u8((sz_u8_t const *)source);
        lookup_0_to_63_vec.u8x16 = vqtbl4q_u8(lut_0_to_63_vec, source_vec.u8x16);
        lookup_64_to_127_vec.u8x16 = vqtbl4q_u8(lut_64_to_127_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0x40)));
        lookup_128_to_191_vec.u8x16 = vqtbl4q_u8(lut_128_to_191_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0x80)));
        lookup_192_to_255_vec.u8x16 = vqtbl4q_u8(lut_192_to_255_vec, veorq_u8(source_vec.u8x16, vdupq_n_u8(0xc0)));
        blended_0_to_255_vec.u8x16 = vorrq_u8( //
            vorrq_u8(lookup_0_to_63_vec.u8x16, lookup_64_to_127_vec.u8x16),
            vorrq_u8(lookup_128_to_191_vec.u8x16, lookup_192_to_255_vec.u8x16));
        vst1q_u8((sz_u8_t *)target, blended_0_to_255_vec.u8x16);
    }

    // Process the tail with serial code
    for (; tail_length; target += 1, source += 1, tail_length -= 1) *target = lut[*(sz_u8_t const *)source];
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

/*  Implementation of the memory operations using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

SZ_PUBLIC void sz_fill_sve(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    svuint8_t value_vec = svdup_u8(value);
    sz_size_t vec_len = svcntb(); // Vector length in bytes (scalable)

    if (length <= vec_len) {
        // Small buffer case: use mask to handle small writes
        svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svst1_u8(mask, (sz_u8_t *)target, value_vec);
    }
    else {
        // Calculate head, body, and tail sizes
        sz_size_t head_length = vec_len - ((sz_size_t)target % vec_len);
        sz_size_t tail_length = (sz_size_t)(target + length) % vec_len;
        sz_size_t body_length = length - head_length - tail_length;

        // Handle unaligned head
        svbool_t head_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)head_length);
        svst1_u8(head_mask, (sz_u8_t *)target, value_vec);
        target += head_length;

        // Aligned body loop
        for (; body_length >= vec_len; target += vec_len, body_length -= vec_len) {
            svst1_u8(svptrue_b8(), (sz_u8_t *)target, value_vec);
        }

        // Handle unaligned tail
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)tail_length);
        svst1_u8(tail_mask, (sz_u8_t *)target, value_vec);
    }
}

SZ_PUBLIC void sz_copy_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_size_t vec_len = svcntb(); // Vector length in bytes

    // Arm Neoverse V2 cores in Graviton 4, for example, come with 256 KB of L1 data cache per core,
    // and 8 MB of L2 cache per core. Moreover, the L1 cache is fully associative.
    // With two strings, we may consider the overal workload huge, if each exceeds 1 MB in length.
    //
    //      int is_huge = length >= 4ull * 1024ull * 1024ull;
    //
    // When the buffer is small, there isn't much to innovate.
    if (length <= vec_len) {
        // Small buffer case: use mask to handle small writes
        svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svuint8_t data = svld1_u8(mask, (sz_u8_t *)source);
        svst1_u8(mask, (sz_u8_t *)target, data);
    }
    // When dealing with larger buffers, similar to AVX-512, we want minimize unaligned operations
    // and handle the head, body, and tail separately. We can also traverse the buffer in both directions
    // as Arm generally supports more simultaneous stores than x86 CPUs.
    //
    // For gigantic datasets, similar to AVX-512, non-temporal "loads" and "stores" can be used.
    // Sadly, if the register size (16 byte or larger) is smaller than a cache-line (64 bytes)
    // we will pay a huge penalty on loads, fetching the same content many times.
    // It may be better to allow caching (and subsequent eviction), in favor of using four-element
    // tuples, wich will be guaranteed to be a multiple of a cache line.
    //
    // Another approach is to use the `LD4B` instructions, which will populate four registers at once.
    // This however, further decreases the performance from LibC-like 29 GB/s to 20 GB/s.
    else {
        // Calculating head, body, and tail sizes depends on the `vec_len`,
        // but it's runtime constant, and the modulo operation is expensive!
        // Instead we use the fact, that it's always a multiple of 128 bits or 16 bytes.
        sz_size_t head_length = 16 - ((sz_size_t)target % 16);
        sz_size_t tail_length = (sz_size_t)(target + length) % 16;
        sz_size_t body_length = length - head_length - tail_length;

        // Handle unaligned parts
        svbool_t head_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)head_length);
        svuint8_t head_data = svld1_u8(head_mask, (sz_u8_t *)source);
        svst1_u8(head_mask, (sz_u8_t *)target, head_data);
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)tail_length);
        svuint8_t tail_data = svld1_u8(tail_mask, (sz_u8_t *)source + head_length + body_length);
        svst1_u8(tail_mask, (sz_u8_t *)target + head_length + body_length, tail_data);
        target += head_length;
        source += head_length;

        // Aligned body loop, walking in two directions
        for (; body_length >= vec_len * 2; target += vec_len, source += vec_len, body_length -= vec_len * 2) {
            svuint8_t forward_data = svld1_u8(svptrue_b8(), (sz_u8_t *)source);
            svuint8_t backward_data = svld1_u8(svptrue_b8(), (sz_u8_t *)source + body_length - vec_len);
            svst1_u8(svptrue_b8(), (sz_u8_t *)target, forward_data);
            svst1_u8(svptrue_b8(), (sz_u8_t *)target + body_length - vec_len, backward_data);
        }
        // Up to (vec_len * 2 - 1) bytes of data may be left in the body,
        // so we can unroll the last two optional loop iterations.
        if (body_length > vec_len) {
            svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)body_length);
            svuint8_t data = svld1_u8(mask, (sz_u8_t *)source);
            svst1_u8(mask, (sz_u8_t *)target, data);
            body_length -= vec_len;
            source += body_length;
            target += body_length;
        }
        if (body_length) {
            svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)body_length);
            svuint8_t data = svld1_u8(mask, (sz_u8_t *)source);
            svst1_u8(mask, (sz_u8_t *)target, data);
        }
    }
}

SZ_PUBLIC void sz_move_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_NEON
    sz_move_neon(target, source, length);
#else
    sz_move_serial(target, source, length);
#endif
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

#pragma region Core Functionality

SZ_DYNAMIC void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_SKYLAKE
    sz_copy_skylake(target, source, length);
#elif SZ_USE_HASWELL
    sz_copy_haswell(target, source, length);
#elif SZ_USE_SVE
    sz_copy_sve(target, source, length);
#elif SZ_USE_NEON
    sz_copy_neon(target, source, length);
#else
    sz_copy_serial(target, source, length);
#endif
}

SZ_DYNAMIC void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_SKYLAKE
    sz_move_skylake(target, source, length);
#elif SZ_USE_HASWELL
    sz_move_haswell(target, source, length);
#elif SZ_USE_SVE
    sz_move_sve(target, source, length);
#elif SZ_USE_NEON
    sz_move_neon(target, source, length);
#else
    sz_move_serial(target, source, length);
#endif
}

SZ_DYNAMIC void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
#if SZ_USE_SKYLAKE
    sz_fill_skylake(target, length, value);
#elif SZ_USE_HASWELL
    sz_fill_haswell(target, length, value);
#elif SZ_USE_SVE
    sz_fill_sve(target, length, value);
#elif SZ_USE_NEON
    sz_fill_neon(target, length, value);
#else
    sz_fill_serial(target, length, value);
#endif
}

SZ_DYNAMIC void sz_lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source, sz_cptr_t lut) {
#if SZ_USE_ICE
    sz_lookup_ice(target, length, source, lut);
#elif SZ_USE_HASWELL
    sz_lookup_haswell(target, length, source, lut);
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
