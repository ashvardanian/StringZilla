/**
 *  @brief  Shared definitions for the StringZilla library.
 *  @file   types.h
 *  @author Ash Vardanian
 *
 *  Includes the following types:
 *
 *  - `sz_u8_t`, `sz_u16_t`, `sz_u32_t`, `sz_u64_t` - unsigned integers of 8, 16, 32, and 64 bits.
 *  - `sz_i8_t`, `sz_i16_t`, `sz_i32_t`, `sz_i64_t` - signed integers of 8, 16, 32, and 64 bits.
 *  - `sz_size_t`, `sz_ssize_t` - unsigned and signed integers of the same size as a pointer.
 *  - `sz_ptr_t`, `sz_cptr_t` - pointer and constant pointer to a C-style string.
 *  - `sz_bool_t` - boolean type, `sz_true_k` and `sz_false_k` constants.
 *  - `sz_ordering_t` - for comparison results, `sz_less_k`, `sz_equal_k`, `sz_greater_k`.
 *  - @b `sz_u8_vec_t`, `sz_u16_vec_t`, `sz_u32_vec_t`, `sz_u64_vec_t` - @b SWAR vector types.
 *  - @b `sz_u128_vec_t`, `sz_u256_vec_t`, `sz_u512_vec_t` - @b SIMD vector types for x86 and Arm.
 *  - @b `sz_rune_t` - for 32-bit Unicode code points ~ @b runes.
 *  - `sz_rune_length_t` - to describe the number of bytes in a UTF8-encoded rune.
 *  - `sz_error_cost_t` - for substitution costs in string alignment and scoring algorithms.
 *
 *  The library also defines the following higher-level structures:
 *
 *  - `sz_string_view_t` - for a C-style `std::string_view`-like structure.
 *  - `sz_memory_allocator_t` - a wrapper for memory-management functions.
 *  - `sz_sequence_t` - a wrapper to access strings forming a sequential container.
 *  - `sz_charset_t` - a bitset for 256 possible byte values.
 */
#ifndef STRINGZILLA_TYPES_H_
#define STRINGZILLA_TYPES_H_

/*
 *  Debugging and testing.
 */
#ifndef SZ_DEBUG
#if defined(DEBUG) || defined(_DEBUG) // This means "Not using DEBUG information".
#define SZ_DEBUG (1)
#else
#define SZ_DEBUG (0)
#endif
#endif

/**
 *  @brief  When set to 1, the library will include the following LibC headers: <stddef.h> and <stdint.h>.
 *          In debug builds (SZ_DEBUG=1), the library will also include <stdio.h> and <stdlib.h>.
 *
 *  You may want to disable this compiling for use in the kernel, or in embedded systems.
 *  You may also avoid them, if you are very sensitive to compilation time and avoid pre-compiled headers.
 *  https://artificial-mind.net/projects/compile-health/
 */
#ifndef SZ_AVOID_LIBC
#define SZ_AVOID_LIBC (0) // true or false
#endif

/**
 *  @brief  Removes compile-time dispatching, and replaces it with runtime dispatching.
 *          So the `sz_find` function will invoke the most advanced backend supported by the CPU,
 *          that runs the program, rather than the most advanced backend supported by the CPU
 *          used to compile the library or the downstream application.
 */
#ifndef SZ_DYNAMIC_DISPATCH
#define SZ_DYNAMIC_DISPATCH (0) // true or false
#endif

/**
 *  @brief  A misaligned load can be - trying to fetch eight consecutive bytes from an address
 *          that is not divisible by eight. On x86 enabled by default. On ARM it's not.
 *
 *  Most platforms support it, but there is no industry standard way to check for those.
 *  This value will mostly affect the performance of the serial (SWAR) backend.
 */
#ifndef SZ_USE_MISALIGNED_LOADS
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define SZ_USE_MISALIGNED_LOADS (1) // true or false
#else
#define SZ_USE_MISALIGNED_LOADS (0) // true or false
#endif
#endif

/**
 *  @brief  Analogous to `size_t` and `std::size_t`, unsigned integer, identical to pointer size.
 *          64-bit on most platforms where pointers are 64-bit.
 *          32-bit on platforms where pointers are 32-bit.
 */
#if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_WIN64)
#define _SZ_IS_64_BIT (1)
#define SZ_SIZE_MAX (0xFFFFFFFFFFFFFFFFull)  // Largest unsigned integer that fits into 64 bits.
#define SZ_SSIZE_MAX (0x7FFFFFFFFFFFFFFFull) // Largest signed integer that fits into 64 bits.
#else
#define _SZ_IS_64_BIT (0)
#define SZ_SIZE_MAX (0xFFFFFFFFu)  // Largest unsigned integer that fits into 32 bits.
#define SZ_SSIZE_MAX (0x7FFFFFFFu) // Largest signed integer that fits into 32 bits.
#endif

/**
 *  @brief  On Big-Endian machines StringZilla will work in compatibility mode.
 *          This disables SWAR hacks to minimize code duplication, assuming practically
 *          all modern popular platforms are Little-Endian.
 *
 *  This variable is hard to infer from macros reliably. It's best to set it manually.
 *  For that CMake provides the `TestBigEndian` and `CMAKE_<LANG>_BYTE_ORDER` (from 3.20 onwards).
 *  In Python one can check `sys.byteorder == 'big'` in the `setup.py` script and pass the appropriate macro.
 *  https://stackoverflow.com/a/27054190
 */
#ifndef _SZ_IS_BIG_ENDIAN
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || defined(__BIG_ENDIAN__) || defined(__ARMEB__) || \
    defined(__THUMBEB__) || defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
#define _SZ_IS_BIG_ENDIAN (1) //< It's a big-endian target architecture
#else
#define _SZ_IS_BIG_ENDIAN (0) //< It's a little-endian target architecture
#endif
#endif

/**
 *  @brief  Threshold for switching to SWAR (8-bytes at a time) backend over serial byte-level for-loops.
 *          On very short strings, under 16 bytes long, at most a single word will be processed with SWAR.
 *          Assuming potentially misaligned loads, SWAR makes sense only after ~24 bytes.
 */
#ifndef SZ_SWAR_THRESHOLD
#if SZ_DEBUG
#define SZ_SWAR_THRESHOLD (8u) // 8 bytes in debug builds
#else
#define SZ_SWAR_THRESHOLD (24u) // 24 bytes in release builds
#endif
#endif

/*  Annotation for the public API symbols:
 *
 *  - `SZ_PUBLIC` is used for functions that are part of the public API.
 *  - `SZ_INTERNAL` is used for internal helper functions with unstable APIs.
 *  - `SZ_DYNAMIC` is used for functions that are part of the public API, but are dispatched at runtime.
 */
#ifndef SZ_DYNAMIC
#if SZ_DYNAMIC_DISPATCH
#if defined(_WIN32) || defined(__CYGWIN__)
#define SZ_DYNAMIC __declspec(dllexport)
#define SZ_EXTERNAL __declspec(dllimport)
#define SZ_PUBLIC inline static
#define SZ_INTERNAL inline static
#else
#define SZ_DYNAMIC __attribute__((visibility("default")))
#define SZ_EXTERNAL extern
#define SZ_PUBLIC __attribute__((unused)) inline static
#define SZ_INTERNAL __attribute__((always_inline)) inline static
#endif // _WIN32 || __CYGWIN__
#else
#define SZ_DYNAMIC inline static
#define SZ_EXTERNAL extern
#define SZ_PUBLIC inline static
#define SZ_INTERNAL inline static
#endif // SZ_DYNAMIC_DISPATCH
#endif // SZ_DYNAMIC

/**
 *  @brief  Alignment macro for 64-byte alignment.
 */
#if defined(_MSC_VER)
#define _SZ_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define _SZ_ALIGN64 __attribute__((aligned(64)))
#else
#define _SZ_ALIGN64
#endif

/**
 *  @brief  Largest prime number that fits into 31 bits.
 *  @see    https://mersenneforum.org/showthread.php?t=3471
 */
#define SZ_U32_MAX_PRIME (2147483647u)

/**
 *  @brief  Largest prime number that fits into 64 bits.
 *  @see    https://mersenneforum.org/showthread.php?t=3471
 *
 *  2^64 = 18,446,744,073,709,551,616
 *  this = 18,446,744,073,709,551,557
 *  diff = 59
 */
#define SZ_U64_MAX_PRIME (18446744073709551557ull)

#if !SZ_AVOID_LIBC
#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`
#endif

/*  The headers needed for the `_sz_assert_failure` function. */
#if SZ_DEBUG && defined(SZ_AVOID_LIBC) && !SZ_AVOID_LIBC && !defined(SZ_PIC)
#include <stdio.h>  // `fprintf`, `stderr`
#include <stdlib.h> // `EXIT_FAILURE`
#endif

/*  Compile-time hardware features detection.
 *  All of those can be controlled by the user.
 */
#ifndef SZ_USE_HASWELL
#ifdef __AVX2__
#define SZ_USE_HASWELL 1
#else
#define SZ_USE_HASWELL 0
#endif
#endif

#ifndef SZ_USE_SKYLAKE
#ifdef __AVX512F__
#define SZ_USE_SKYLAKE 1
#else
#define SZ_USE_SKYLAKE 0
#endif
#endif

#ifndef SZ_USE_ICE
#ifdef __AVX512BW__
#define SZ_USE_ICE 1
#else
#define SZ_USE_ICE 0
#endif
#endif

#ifndef SZ_USE_NEON
#ifdef __ARM_NEON
#define SZ_USE_NEON 1
#else
#define SZ_USE_NEON 0
#endif
#endif

#ifndef SZ_USE_SVE
#ifdef __ARM_FEATURE_SVE
#define SZ_USE_SVE 1
#else
#define SZ_USE_SVE 0
#endif
#endif

/*  Hardware-specific headers for different SIMD intrinsics and register wrappers.
 */
#if SZ_USE_HASWELL || SZ_USE_SKYLAKE || SZ_USE_ICE
#include <immintrin.h>
#endif // SZ_USE_X86...
#if SZ_USE_NEON
#if !defined(_MSC_VER)
#include <arm_acle.h>
#endif
#include <arm_neon.h>
#endif // SZ_USE_NEON
#if SZ_USE_SVE
#if !defined(_MSC_VER)
#include <arm_sve.h>
#endif
#endif // SZ_USE_SVE

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  Let's infer the integer types or pull them from LibC,
 *  if that is allowed by the user.
 */
#if !SZ_AVOID_LIBC
typedef int8_t sz_i8_t;       // Always 8 bits
typedef uint8_t sz_u8_t;      // Always 8 bits
typedef uint16_t sz_u16_t;    // Always 16 bits
typedef int32_t sz_i32_t;     // Always 32 bits
typedef uint32_t sz_u32_t;    // Always 32 bits
typedef uint64_t sz_u64_t;    // Always 64 bits
typedef int64_t sz_i64_t;     // Always 64 bits
typedef size_t sz_size_t;     // Pointer-sized unsigned integer, 32 or 64 bits
typedef ptrdiff_t sz_ssize_t; // Signed version of `sz_size_t`, 32 or 64 bits

#else // if SZ_AVOID_LIBC:

// ! The C standard doesn't specify the signedness of char.
// ! On x86 char is signed by default while on Arm it is unsigned by default.
// ! That's why we don't define `sz_char_t` and generally use explicit `sz_i8_t` and `sz_u8_t`.
typedef signed char sz_i8_t;         // Always 8 bits
typedef unsigned char sz_u8_t;       // Always 8 bits
typedef unsigned short sz_u16_t;     // Always 16 bits
typedef int sz_i32_t;                // Always 32 bits
typedef unsigned int sz_u32_t;       // Always 32 bits
typedef long long sz_i64_t;          // Always 64 bits
typedef unsigned long long sz_u64_t; // Always 64 bits

// Now we need to redefine the `size_t`.
// Microsoft Visual C++ (MSVC) typically follows LLP64 data model on 64-bit platforms,
// where integers, pointers, and long types have different sizes:
//
//  > `int` is 32 bits
//  > `long` is 32 bits
//  > `long long` is 64 bits
//  > pointer (thus, `size_t`) is 64 bits
//
// In contrast, GCC and Clang on 64-bit Unix-like systems typically follow the LP64 model, where:
//
//  > `int` is 32 bits
//  > `long` and pointer (thus, `size_t`) are 64 bits
//  > `long long` is also 64 bits
//
// Source: https://learn.microsoft.com/en-us/windows/win32/winprog64/abstract-data-models
#if _SZ_IS_64_BIT
typedef unsigned long long sz_size_t; // 64-bit.
typedef long long sz_ssize_t;         // 64-bit.
#else
typedef unsigned sz_size_t;  // 32-bit.
typedef unsigned sz_ssize_t; // 32-bit.
#endif // _SZ_IS_64_BIT

#endif // SZ_AVOID_LIBC

/**
 *  @brief  Compile-time assert macro similar to `static_assert` in C++.
 */
#define sz_static_assert(condition, name)                \
    typedef struct {                                     \
        int static_assert_##name : (condition) ? 1 : -1; \
    } sz_static_assert_##name##_t

sz_static_assert(sizeof(sz_size_t) == sizeof(void *), sz_size_t_must_be_pointer_size);
sz_static_assert(sizeof(sz_ssize_t) == sizeof(void *), sz_ssize_t_must_be_pointer_size);

typedef char *sz_ptr_t;          // A type alias for `char *`
typedef char const *sz_cptr_t;   // A type alias for `char const *`
typedef sz_i8_t sz_error_cost_t; // Character mismatch cost for fuzzy matching functions

typedef sz_u64_t sz_sorted_idx_t; // Index of a sorted string in a list of strings

typedef enum { sz_false_k = 0, sz_true_k = 1 } sz_bool_t;                        // Only one relevant bit
typedef enum { sz_less_k = -1, sz_equal_k = 0, sz_greater_k = 1 } sz_ordering_t; // Only three possible states: <=>

/**
 *  @brief  Describes the length of a UTF8 @b rune / character / codepoint in bytes.
 */
typedef enum {
    sz_utf8_invalid_k = 0,     //!< Invalid UTF8 character.
    sz_utf8_rune_1byte_k = 1,  //!< 1-byte UTF8 character.
    sz_utf8_rune_2bytes_k = 2, //!< 2-byte UTF8 character.
    sz_utf8_rune_3bytes_k = 3, //!< 3-byte UTF8 character.
    sz_utf8_rune_4bytes_k = 4, //!< 4-byte UTF8 character.
} sz_rune_length_t;

typedef sz_u32_t sz_rune_t;

/**
 *  @brief  Tiny string-view structure. It's POD type, unlike the `std::string_view`.
 */
typedef struct sz_string_view_t {
    sz_cptr_t start;
    sz_size_t length;
} sz_string_view_t;

#pragma region Character Sets

/**
 *  @brief  Bit-set structure for 256 possible byte values. Useful for filtering and search.
 *  @see    sz_charset_init, sz_charset_add, sz_charset_contains, sz_charset_invert
 */
typedef union sz_charset_t {
    sz_u64_t _u64s[4];
    sz_u32_t _u32s[8];
    sz_u16_t _u16s[16];
    sz_u8_t _u8s[32];
} sz_charset_t;

/** @brief  Initializes a bit-set to an empty collection, meaning - all characters are banned. */
SZ_PUBLIC void sz_charset_init(sz_charset_t *s) { s->_u64s[0] = s->_u64s[1] = s->_u64s[2] = s->_u64s[3] = 0; }

/** @brief  Initializes a bit-set to all ASCII character. */
SZ_PUBLIC void sz_charset_init_ascii(sz_charset_t *s) {
    s->_u64s[0] = s->_u64s[1] = 0xFFFFFFFFFFFFFFFFull;
    s->_u64s[2] = s->_u64s[3] = 0;
}

/** @brief  Adds a character to the set and accepts @b unsigned integers. */
SZ_PUBLIC void sz_charset_add_u8(sz_charset_t *s, sz_u8_t c) { s->_u64s[c >> 6] |= (1ull << (c & 63u)); }

/** @brief  Adds a character to the set. Consider @b sz_charset_add_u8. */
SZ_PUBLIC void sz_charset_add(sz_charset_t *s, char c) { sz_charset_add_u8(s, *(sz_u8_t *)(&c)); } // bitcast

/** @brief  Checks if the set contains a given character and accepts @b unsigned integers. */
SZ_PUBLIC sz_bool_t sz_charset_contains_u8(sz_charset_t const *s, sz_u8_t c) {
    // Checking the bit can be done in different ways:
    // - (s->_u64s[c >> 6] & (1ull << (c & 63u))) != 0
    // - (s->_u32s[c >> 5] & (1u << (c & 31u))) != 0
    // - (s->_u16s[c >> 4] & (1u << (c & 15u))) != 0
    // - (s->_u8s[c >> 3] & (1u << (c & 7u))) != 0
    return (sz_bool_t)((s->_u64s[c >> 6] & (1ull << (c & 63u))) != 0);
}

/** @brief  Checks if the set contains a given character. Consider @b sz_charset_contains_u8. */
SZ_PUBLIC sz_bool_t sz_charset_contains(sz_charset_t const *s, char c) {
    return sz_charset_contains_u8(s, *(sz_u8_t *)(&c)); // bitcast
}

/** @brief  Inverts the contents of the set, so allowed character get disallowed, and vice versa. */
SZ_PUBLIC void sz_charset_invert(sz_charset_t *s) {
    s->_u64s[0] ^= 0xFFFFFFFFFFFFFFFFull, s->_u64s[1] ^= 0xFFFFFFFFFFFFFFFFull, //
        s->_u64s[2] ^= 0xFFFFFFFFFFFFFFFFull, s->_u64s[3] ^= 0xFFFFFFFFFFFFFFFFull;
}

#pragma endregion

#pragma region Memory Management

typedef void *(*sz_memory_allocate_t)(sz_size_t, void *);
typedef void (*sz_memory_free_t)(void *, sz_size_t, void *);
typedef sz_u64_t (*sz_random_generator_t)(void *);

/**
 *  @brief  Some complex pattern matching algorithms may require memory allocations.
 *          This structure is used to pass the memory allocator to those functions.
 *  @see    sz_memory_allocator_init_fixed
 */
typedef struct sz_memory_allocator_t {
    sz_memory_allocate_t allocate;
    sz_memory_free_t free;
    void *handle;
} sz_memory_allocator_t;

/**
 *  @brief  Initializes a memory allocator to use the system default `malloc` and `free`.
 *          ! The function is not available if the library was compiled with `SZ_AVOID_LIBC`.
 *
 *  @param alloc    Memory allocator to initialize.
 */
SZ_PUBLIC void sz_memory_allocator_init_default(sz_memory_allocator_t *alloc);

/**
 *  @brief  Initializes a memory allocator to use a static-capacity buffer.
 *          No dynamic allocations will be performed.
 *
 *  @param alloc    Memory allocator to initialize.
 *  @param buffer   Buffer to use for allocations.
 *  @param length   Length of the buffer. @b Must be greater than 8 bytes. Different values would be optimal for
 *                  different algorithms and input lengths, but 4096 bytes (one RAM page) is a good default.
 */
SZ_PUBLIC void sz_memory_allocator_init_fixed(sz_memory_allocator_t *alloc, void *buffer, sz_size_t length);

#pragma endregion

#pragma region API Signature Types

typedef sz_u64_t (*sz_hash_t)(sz_cptr_t, sz_size_t);
typedef sz_u64_t (*sz_checksum_t)(sz_cptr_t, sz_size_t);
typedef sz_bool_t (*sz_equal_t)(sz_cptr_t, sz_cptr_t, sz_size_t);
typedef sz_ordering_t (*sz_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);
typedef void (*sz_to_converter_t)(sz_cptr_t, sz_size_t, sz_ptr_t);

typedef void (*sz_look_up_transform_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_ptr_t);

typedef void (*sz_move_t)(sz_ptr_t, sz_cptr_t, sz_size_t);

typedef void (*sz_fill_t)(sz_ptr_t, sz_size_t, sz_u8_t);

typedef sz_cptr_t (*sz_find_byte_t)(sz_cptr_t, sz_size_t, sz_cptr_t);
typedef sz_cptr_t (*sz_find_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);
typedef sz_cptr_t (*sz_find_set_t)(sz_cptr_t, sz_size_t, sz_charset_t const *);

typedef sz_size_t (*sz_hamming_distance_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t, sz_size_t);

typedef sz_size_t (*sz_edit_distance_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t, sz_size_t, sz_memory_allocator_t *);

typedef sz_ssize_t (*sz_alignment_score_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t, sz_error_cost_t const *,
                                           sz_error_cost_t, sz_memory_allocator_t *);

typedef void (*sz_hash_callback_t)(sz_cptr_t, sz_size_t, sz_u64_t, void *user);

typedef void (*sz_hashes_t)(sz_cptr_t, sz_size_t, sz_size_t, sz_size_t, sz_hash_callback_t, void *);

typedef void (*sz_hashes_fingerprint_t)(sz_cptr_t, sz_size_t, sz_size_t, sz_ptr_t, sz_size_t);

typedef sz_size_t (*sz_hashes_intersection_t)(sz_cptr_t, sz_size_t, sz_size_t, sz_cptr_t, sz_size_t);

#pragma endregion

#pragma region Helper Structures

/**
 *  @brief  Helper structure to simplify work with 16-bit words.
 *  @see    sz_u16_load
 */
typedef union sz_u16_vec_t {
    sz_u16_t u16;
    sz_u8_t u8s[2];
} sz_u16_vec_t;

/**
 *  @brief  Helper structure to simplify work with 32-bit words.
 *  @see    sz_u32_load
 */
typedef union sz_u32_vec_t {
    sz_u32_t u32;
    sz_u16_t u16s[2];
    sz_u8_t u8s[4];
} sz_u32_vec_t;

/**
 *  @brief  Helper structure to simplify work with 64-bit words.
 *  @see    sz_u64_load
 */
typedef union sz_u64_vec_t {
    sz_u64_t u64;
    sz_u32_t u32s[2];
    sz_u16_t u16s[4];
    sz_u8_t u8s[8];
} sz_u64_vec_t;

/**
 *  @brief  Helper structure to simplify work with @b 128-bit registers.
 *          It can help view the contents as 8-bit, 16-bit, 32-bit, or 64-bit integers,
 *          as well as 1x XMM register.
 */
typedef union sz_u128_vec_t {
#if SZ_USE_HASWELL
    __m128i xmm;
#endif
#if SZ_USE_NEON
    uint8x16_t u8x16;
    uint16x8_t u16x8;
    uint32x4_t u32x4;
    uint64x2_t u64x2;
#endif
    sz_u64_t u64s[2];
    sz_u32_t u32s[4];
    sz_u16_t u16s[8];
    sz_u8_t u8s[16];
} sz_u128_vec_t;

/**
 *  @brief  Helper structure to simplify work with @b 256-bit registers.
 *          It can help view the contents as 8-bit, 16-bit, 32-bit, or 64-bit integers,
 *          as well as 2x XMM registers or 1x YMM register.
 */
typedef union sz_u256_vec_t {
#if SZ_USE_HASWELL
    __m256i ymm;
    __m128i xmms[2];
#endif
    sz_u64_t u64s[4];
    sz_u32_t u32s[8];
    sz_u16_t u16s[16];
    sz_u8_t u8s[32];
} sz_u256_vec_t;

/**
 *  @brief  Helper structure to simplify work with @b 512-bit registers.
 *          It can help view the contents as 8-bit, 16-bit, 32-bit, or 64-bit integers,
 *          as well as 4x XMM registers or 2x YMM registers or 1x ZMM register.
 */
typedef union sz_u512_vec_t {
#if SZ_USE_SKYLAKE || SZ_USE_ICE
    __m512i zmm;
#endif
#if SZ_USE_HASWELL || SZ_USE_SKYLAKE || SZ_USE_ICE
    __m256i ymms[2];
    __m128i xmms[4];
#endif
    sz_u64_t u64s[8];
    sz_i64_t i64s[8];
    sz_u32_t u32s[16];
    sz_i32_t i32s[16];
    sz_u16_t u16s[32];
    sz_u8_t u8s[64];
} sz_u512_vec_t;

#pragma endregion

#pragma region UTF8

/**
 *  @brief  Extracts just one UTF8 codepoint from a UTF8 string into a 32-bit unsigned integer.
 */
SZ_INTERNAL void _sz_extract_utf8_rune(sz_cptr_t utf8, sz_rune_t *code, sz_rune_length_t *code_length) {
    sz_u8_t const *current = (sz_u8_t const *)utf8;
    sz_u8_t leading_byte = *current++;
    sz_rune_t ch;
    sz_rune_length_t ch_length;

    // TODO: This can be made entirely branchless using 32-bit SWAR.
    if (leading_byte < 0x80) {
        // Single-byte rune (0xxxxxxx)
        ch = leading_byte;
        ch_length = sz_utf8_rune_1byte_k;
    }
    else if ((leading_byte & 0xE0) == 0xC0) {
        // Two-byte rune (110xxxxx 10xxxxxx)
        ch = (leading_byte & 0x1F) << 6;
        ch |= (*current++ & 0x3F);
        ch_length = sz_utf8_rune_2bytes_k;
    }
    else if ((leading_byte & 0xF0) == 0xE0) {
        // Three-byte rune (1110xxxx 10xxxxxx 10xxxxxx)
        ch = (leading_byte & 0x0F) << 12;
        ch |= (*current++ & 0x3F) << 6;
        ch |= (*current++ & 0x3F);
        ch_length = sz_utf8_rune_3bytes_k;
    }
    else if ((leading_byte & 0xF8) == 0xF0) {
        // Four-byte rune (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
        ch = (leading_byte & 0x07) << 18;
        ch |= (*current++ & 0x3F) << 12;
        ch |= (*current++ & 0x3F) << 6;
        ch |= (*current++ & 0x3F);
        ch_length = sz_utf8_rune_4bytes_k;
    }
    else {
        // Invalid UTF8 rune.
        ch = 0;
        ch_length = sz_utf8_invalid_k;
    }
    *code = ch;
    *code_length = ch_length;
}

/**
 *  @brief  Exports a UTF8 string into a UTF32 buffer.
 *          ! The result is undefined id the UTF8 string is corrupted.
 *  @return The length in the number of codepoints.
 */
SZ_INTERNAL sz_size_t _sz_export_utf8_to_utf32(sz_cptr_t utf8, sz_size_t utf8_length, sz_rune_t *utf32) {
    sz_cptr_t const end = utf8 + utf8_length;
    sz_size_t count = 0;
    sz_rune_length_t rune_length;
    for (; utf8 != end; utf8 += rune_length, utf32++, count++) _sz_extract_utf8_rune(utf8, utf32, &rune_length);
    return count;
}

#pragma endregion

#pragma region String Sequences API

struct sz_sequence_t;

typedef sz_cptr_t (*sz_sequence_member_start_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_size_t (*sz_sequence_member_length_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_bool_t (*sz_sequence_predicate_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_bool_t (*sz_sequence_comparator_t)(struct sz_sequence_t const *, sz_size_t, sz_size_t);
typedef sz_bool_t (*sz_string_is_less_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

typedef struct sz_sequence_t {
    sz_sorted_idx_t *order;
    sz_size_t count;
    sz_sequence_member_start_t get_start;
    sz_sequence_member_length_t get_length;
    void const *handle;
} sz_sequence_t;

/**
 *  @brief  Initiates the sequence structure from a tape layout, used by Apache Arrow.
 *          Expects ::offsets to contains `count + 1` entries, the last pointing at the end
 *          of the last string, indicating the total length of the ::tape.
 */
SZ_PUBLIC void sz_sequence_from_u32tape( //
    sz_cptr_t *start, sz_u32_t const *offsets, sz_size_t count, sz_sequence_t *sequence);

/**
 *  @brief  Initiates the sequence structure from a tape layout, used by Apache Arrow.
 *          Expects ::offsets to contains `count + 1` entries, the last pointing at the end
 *          of the last string, indicating the total length of the ::tape.
 */
SZ_PUBLIC void sz_sequence_from_u64tape( //
    sz_cptr_t *start, sz_u64_t const *offsets, sz_size_t count, sz_sequence_t *sequence);

#pragma endregion

#pragma region Helper Functions

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC visibility push(hidden)

/*
 **********************************************************************************************************************
 **********************************************************************************************************************
 **********************************************************************************************************************
 *
 *  This is where we the actual implementation begins.
 *  The rest of the file is hidden from the public API.
 *
 **********************************************************************************************************************
 **********************************************************************************************************************
 **********************************************************************************************************************
 */

/**
 *  @brief  Helper-macro to mark potentially unused variables.
 */
#define sz_unused(x) ((void)(x))

/**
 *  @brief  Helper-macro casting a variable to another type of the same size.
 */
#define sz_bitcast(type, value) (*((type *)&(value)))

/**
 *  @brief  Defines `SZ_NULL`, analogous to `NULL`.
 *          The default often comes from locale.h, stddef.h,
 *          stdio.h, stdlib.h, string.h, time.h, or wchar.h.
 */
#ifdef __GNUG__
#define SZ_NULL __null
#define SZ_NULL_CHAR __null
#else
#define SZ_NULL ((void *)0)
#define SZ_NULL_CHAR ((char *)0)
#endif

/**
 *  @brief  Cache-line width, that will affect the execution of some algorithms,
 *          like equality checks and relative order computing.
 */
#define SZ_CACHE_LINE_WIDTH (64) // bytes

/**
 *  @brief  Similar to `assert`, the `_sz_assert` is used in the `SZ_DEBUG` mode
 *          to check the invariants of the library. It's a no-op in the "Release" mode.
 *  @note   If you want to catch it, put a breakpoint at @b `__GI_exit`
 */
#if SZ_DEBUG && defined(SZ_AVOID_LIBC) && !SZ_AVOID_LIBC && !defined(SZ_PIC)
SZ_PUBLIC void _sz_assert_failure(char const *condition, char const *file, int line) {
    fprintf(stderr, "Assertion failed: %s, in file %s, line %d\n", condition, file, line);
    exit(EXIT_FAILURE);
}
#define _sz_assert(condition)                                                     \
    do {                                                                          \
        if (!(condition)) { _sz_assert_failure(#condition, __FILE__, __LINE__); } \
    } while (0)
#else
#define _sz_assert(condition) ((void)(condition))
#endif

/*  Intrinsics aliases for MSVC, GCC, Clang, and Clang-Cl.
 *  The following section of compiler intrinsics comes in 2 flavors.
 */
#if defined(_MSC_VER) && !defined(__clang__) // On Clang-CL
#include <intrin.h>

// Sadly, when building Win32 images, we can't use the `_tzcnt_u64`, `_lzcnt_u64`,
// `_BitScanForward64`, or `_BitScanReverse64` intrinsics. For now it's a simple `for`-loop.
// TODO: In the future we can switch to a more efficient De Bruijn's algorithm.
// https://www.chessprogramming.org/BitScan
// https://www.chessprogramming.org/De_Bruijn_Sequence
// https://gist.github.com/resilar/e722d4600dbec9752771ab4c9d47044f
//
// Use the serial version on 32-bit x86 and on Arm.
#if (defined(_WIN32) && !defined(_WIN64)) || defined(_M_ARM) || defined(_M_ARM64)
SZ_INTERNAL int sz_u64_ctz(sz_u64_t x) {
    _sz_assert(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
SZ_INTERNAL int sz_u64_clz(sz_u64_t x) {
    _sz_assert(x != 0);
    int n = 0;
    while ((x & 0x8000000000000000ull) == 0) { n++, x <<= 1; }
    return n;
}
SZ_INTERNAL int sz_u64_popcount(sz_u64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    return (((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full) * 0x0101010101010101ull) >> 56;
}
SZ_INTERNAL int sz_u32_ctz(sz_u32_t x) {
    _sz_assert(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
SZ_INTERNAL int sz_u32_clz(sz_u32_t x) {
    _sz_assert(x != 0);
    int n = 0;
    while ((x & 0x80000000u) == 0) { n++, x <<= 1; }
    return n;
}
SZ_INTERNAL int sz_u32_popcount(sz_u32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
#else
SZ_INTERNAL int sz_u64_ctz(sz_u64_t x) { return (int)_tzcnt_u64(x); }
SZ_INTERNAL int sz_u64_clz(sz_u64_t x) { return (int)_lzcnt_u64(x); }
SZ_INTERNAL int sz_u64_popcount(sz_u64_t x) { return (int)__popcnt64(x); }
SZ_INTERNAL int sz_u32_ctz(sz_u32_t x) { return (int)_tzcnt_u32(x); }
SZ_INTERNAL int sz_u32_clz(sz_u32_t x) { return (int)_lzcnt_u32(x); }
SZ_INTERNAL int sz_u32_popcount(sz_u32_t x) { return (int)__popcnt(x); }
#endif
// Force the byteswap functions to be intrinsics, because when /Oi- is given, these will turn into CRT function calls,
// which breaks when `SZ_AVOID_LIBC` is given
#pragma intrinsic(_byteswap_uint64)
SZ_INTERNAL sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return _byteswap_uint64(val); }
#pragma intrinsic(_byteswap_ulong)
SZ_INTERNAL sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return _byteswap_ulong(val); }
#else
SZ_INTERNAL int sz_u64_popcount(sz_u64_t x) { return __builtin_popcountll(x); }
SZ_INTERNAL int sz_u32_popcount(sz_u32_t x) { return __builtin_popcount(x); }
SZ_INTERNAL int sz_u64_ctz(sz_u64_t x) { return __builtin_ctzll(x); }
SZ_INTERNAL int sz_u64_clz(sz_u64_t x) { return __builtin_clzll(x); }
SZ_INTERNAL int sz_u32_ctz(sz_u32_t x) { return __builtin_ctz(x); } // ! Undefined if `x == 0`
SZ_INTERNAL int sz_u32_clz(sz_u32_t x) { return __builtin_clz(x); } // ! Undefined if `x == 0`
SZ_INTERNAL sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return __builtin_bswap64(val); }
SZ_INTERNAL sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return __builtin_bswap32(val); }
#endif

SZ_INTERNAL sz_u64_t sz_u64_rotl(sz_u64_t x, sz_u64_t r) { return (x << r) | (x >> (64 - r)); }

/**
 *  @brief  Select bits from either ::a or ::b depending on the value of ::mask bits.
 *
 *  Similar to `_mm_blend_epi16` intrinsic on x86.
 *  Described in the "Bit Twiddling Hacks" by Sean Eron Anderson.
 *  https://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
 */
SZ_INTERNAL sz_u64_t sz_u64_blend(sz_u64_t a, sz_u64_t b, sz_u64_t mask) { return a ^ ((a ^ b) & mask); }

/*
 *  Efficiently computing the minimum and maximum of two or three values can be tricky.
 *  The simple branching baseline would be:
 *
 *      x < y ? x : y                               // can replace with 1 conditional move
 *
 *  Branchless approach is well known for signed integers, but it doesn't apply to unsigned ones.
 *  https://stackoverflow.com/questions/514435/templatized-branchless-int-max-min-function
 *  https://graphics.stanford.edu/~seander/bithacks.html#IntegerMinOrMax
 *  Using only bit-shifts for singed integers it would be:
 *
 *      y + ((x - y) & (x - y) >> 31)               // 4 unique operations
 *
 *  Alternatively, for any integers using multiplication:
 *
 *      (x > y) * y + (x <= y) * x                  // 5 operations
 *
 *  Alternatively, to avoid multiplication:
 *
 *      x & ~((x < y) - 1) + y & ((x < y) - 1)      // 6 unique operations
 */
#define sz_min_of_two(x, y) (x < y ? x : y)
#define sz_max_of_two(x, y) (x < y ? y : x)
#define sz_min_of_three(x, y, z) sz_min_of_two(x, sz_min_of_two(y, z))
#define sz_max_of_three(x, y, z) sz_max_of_two(x, sz_max_of_two(y, z))

/**
 *  One option to avoid branching is to use conditional moves and lookup the comparison result in a table:
 *       sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
 *       for (; a != min_end; ++a, ++b)
 *           if (*a != *b) return ordering_lookup[*a < *b];
 *  That, however, introduces a data-dependency.
 *  A cleaner option is to perform two comparisons and a subtraction.
 *  One instruction more, but no data-dependency.
 */
#define _sz_order_scalars(a, b) ((sz_ordering_t)((a > b) - (a < b)))

/** @brief  Branchless minimum function for two signed 32-bit integers. */
SZ_INTERNAL sz_i32_t sz_i32_min_of_two(sz_i32_t x, sz_i32_t y) { return y + ((x - y) & (x - y) >> 31); }

/** @brief  Branchless minimum function for two signed 32-bit integers. */
SZ_INTERNAL sz_i32_t sz_i32_max_of_two(sz_i32_t x, sz_i32_t y) { return x - ((x - y) & (x - y) >> 31); }

/*  In AVX-512 we actively use masked operations and the "K mask registers".
 *  Producing a mask for the first N elements of a sequence can be done using the `1 << N - 1` idiom.
 *  It, however, induces undefined behavior if `N == 64` or `N == 32` on 64-bit or 32-bit systems respectively.
 *  Alternatively, the BZHI instruction can be used to clear the bits above N.
 */
#if SZ_USE_SKYLAKE || SZ_USE_ICE
SZ_INTERNAL __mmask16 _sz_u16_mask_until(sz_size_t n) { return (__mmask16)_bzhi_u32(0xFFFFu, n); }
SZ_INTERNAL __mmask32 _sz_u32_mask_until(sz_size_t n) { return (__mmask32)_bzhi_u64(0xFFFFFFFFu, n); }
SZ_INTERNAL __mmask64 _sz_u64_mask_until(sz_size_t n) { return (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n); }
SZ_INTERNAL __mmask16 _sz_u16_clamp_mask_until(sz_size_t n) { return n < 16 ? _sz_u16_mask_until(n) : 0xFFFFu; }
SZ_INTERNAL __mmask32 _sz_u32_clamp_mask_until(sz_size_t n) { return n < 32 ? _sz_u32_mask_until(n) : 0xFFFFFFFFu; }
SZ_INTERNAL __mmask64 _sz_u64_clamp_mask_until(sz_size_t n) {
    return n < 64 ? _sz_u64_mask_until(n) : 0xFFFFFFFFFFFFFFFFull;
}
#endif

/**
 *  @brief  Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each byte signifies a match.
 */
SZ_INTERNAL sz_u64_vec_t _sz_u64_each_byte_equal(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each byte is set.
    // For that take the bottom 7 bits of each byte, add one to them,
    // and if this sets the top bit to one, then all the 7 bits are ones as well.
    vec.u64 = ((vec.u64 & 0x7F7F7F7F7F7F7F7Full) + 0x0101010101010101ull) & ((vec.u64 & 0x8080808080808080ull));
    return vec;
}

/**
 *  @brief  Clamps signed offsets in a string to a valid range. Used for Pythonic-style slicing.
 */
SZ_INTERNAL void sz_ssize_clamp_interval( //
    sz_size_t length, sz_ssize_t start, sz_ssize_t end, sz_size_t *normalized_offset, sz_size_t *normalized_length) {
    // TODO: Remove branches.
    // Normalize negative indices
    if (start < 0) start += length;
    if (end < 0) end += length;

    // Clamp indices to a valid range
    if (start < 0) start = 0;
    if (end < 0) end = 0;
    if (start > (sz_ssize_t)length) start = length;
    if (end > (sz_ssize_t)length) end = length;

    // Ensure start <= end
    if (start > end) start = end;

    *normalized_offset = start;
    *normalized_length = end - start;
}

/**
 *  @brief  Compute the logarithm base 2 of a positive integer, rounding down.
 */
SZ_INTERNAL sz_size_t sz_size_log2i_nonzero(sz_size_t x) {
    _sz_assert(x > 0 && "Non-positive numbers have no defined logarithm");
    sz_size_t leading_zeros = sz_u64_clz(x);
    return 63 - leading_zeros;
}

/**
 *  @brief  Compute the smallest power of two greater than or equal to ::x.
 */
SZ_INTERNAL sz_size_t sz_size_bit_ceil(sz_size_t x) {
    // Unlike the commonly used trick with `clz` intrinsics, is valid across the whole range of `x`.
    // https://stackoverflow.com/a/10143264
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if _SZ_IS_64_BIT
    x |= x >> 32;
#endif
    x++;
    return x;
}

/**
 *  @brief  Transposes an 8x8 bit matrix packed in a `sz_u64_t`.
 *
 *  There is a well known SWAR sequence for that known to chess programmers,
 *  willing to flip a bit-matrix of pieces along the main A1-H8 diagonal.
 *  https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating
 *  https://lukas-prokop.at/articles/2021-07-23-transpose
 */
SZ_INTERNAL sz_u64_t sz_u64_transpose(sz_u64_t x) {
    sz_u64_t t;
    t = x ^ (x << 36);
    x ^= 0xf0f0f0f00f0f0f0full & (t ^ (x >> 36));
    t = 0xcccc0000cccc0000ull & (x ^ (x << 18));
    x ^= t ^ (t >> 18);
    t = 0xaa00aa00aa00aa00ull & (x ^ (x << 9));
    x ^= t ^ (t >> 9);
    return x;
}

/**
 *  @brief  Helper, that swaps two 64-bit integers representing the order of elements in the sequence.
 */
SZ_INTERNAL void sz_u64_swap(sz_u64_t *a, sz_u64_t *b) {
    sz_u64_t t = *a;
    *a = *b;
    *b = t;
}

/**
 *  @brief  Helper, that swaps two 64-bit integers representing the order of elements in the sequence.
 */
SZ_INTERNAL void sz_pointer_swap(void **a, void **b) {
    void *t = *a;
    *a = *b;
    *b = t;
}

/**
 *  @brief Load a 16-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u16_vec_t sz_u16_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u16_vec_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    return result;
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The __unaligned modifier isn't valid for the x86 platform.
    return *((sz_u16_vec_t *)ptr);
#else
    return *((__unaligned sz_u16_vec_t *)ptr);
#endif
#else
    __attribute__((aligned(1))) sz_u16_vec_t const *result = (sz_u16_vec_t const *)ptr;
    return *result;
#endif
}

/**
 *  @brief Load a 32-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u32_vec_t sz_u32_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u32_vec_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    result.u8s[2] = ptr[2];
    result.u8s[3] = ptr[3];
    return result;
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The __unaligned modifier isn't valid for the x86 platform.
    return *((sz_u32_vec_t *)ptr);
#else
    return *((__unaligned sz_u32_vec_t *)ptr);
#endif
#else
    __attribute__((aligned(1))) sz_u32_vec_t const *result = (sz_u32_vec_t const *)ptr;
    return *result;
#endif
}

/**
 *  @brief Load a 64-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u64_vec_t sz_u64_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u64_vec_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    result.u8s[2] = ptr[2];
    result.u8s[3] = ptr[3];
    result.u8s[4] = ptr[4];
    result.u8s[5] = ptr[5];
    result.u8s[6] = ptr[6];
    result.u8s[7] = ptr[7];
    return result;
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The __unaligned modifier isn't valid for the x86 platform.
    return *((sz_u64_vec_t *)ptr);
#else
    return *((__unaligned sz_u64_vec_t *)ptr);
#endif
#else
    __attribute__((aligned(1))) sz_u64_vec_t const *result = (sz_u64_vec_t const *)ptr;
    return *result;
#endif
}

/** @brief  Helper function, using the supplied fixed-capacity buffer to allocate memory. */
SZ_INTERNAL sz_ptr_t _sz_memory_allocate_fixed(sz_size_t length, void *handle) {
    sz_size_t capacity;
    *(sz_ptr_t)&capacity = *(sz_cptr_t)handle;
    sz_size_t consumed_capacity = sizeof(sz_size_t);
    if (consumed_capacity + length > capacity) return SZ_NULL_CHAR;
    return (sz_ptr_t)handle + consumed_capacity;
}

/** @brief  Helper "no-op" function, simulating memory deallocation when we use a "static" memory buffer. */
SZ_INTERNAL void _sz_memory_free_fixed(sz_ptr_t start, sz_size_t length, void *handle) {
    sz_unused(start && length && handle);
}

#pragma GCC visibility pop
#pragma endregion

#pragma region Serial Implementation

#if !SZ_AVOID_LIBC
#include <stdio.h>  // `fprintf`
#include <stdlib.h> // `malloc`, `EXIT_FAILURE`

SZ_PUBLIC void *_sz_memory_allocate_default(sz_size_t length, void *handle) {
    sz_unused(handle);
    return malloc(length);
}
SZ_PUBLIC void _sz_memory_free_default(sz_ptr_t start, sz_size_t length, void *handle) {
    sz_unused(handle && length);
    free(start);
}

#endif

SZ_PUBLIC void sz_memory_allocator_init_default(sz_memory_allocator_t *alloc) {
#if !SZ_AVOID_LIBC
    alloc->allocate = (sz_memory_allocate_t)_sz_memory_allocate_default;
    alloc->free = (sz_memory_free_t)_sz_memory_free_default;
#else
    alloc->allocate = (sz_memory_allocate_t)SZ_NULL;
    alloc->free = (sz_memory_free_t)SZ_NULL;
#endif
    alloc->handle = SZ_NULL;
}

SZ_PUBLIC void sz_memory_allocator_init_fixed(sz_memory_allocator_t *alloc, void *buffer, sz_size_t length) {
    // The logic here is simple - put the buffer length in the first slots of the buffer.
    // Later use it for bounds checking.
    alloc->allocate = (sz_memory_allocate_t)_sz_memory_allocate_fixed;
    alloc->free = (sz_memory_free_t)_sz_memory_free_fixed;
    alloc->handle = &buffer;
    *(sz_ptr_t)buffer = *(sz_cptr_t)&length;
}

#pragma endregion

#ifdef __cplusplus
#pragma GCC diagnostic pop
}
#endif // __cplusplus

#endif // STRINGZILLA_TYPES_H_
