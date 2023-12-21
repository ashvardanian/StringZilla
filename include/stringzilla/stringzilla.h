/**
 *  @brief  StringZilla is a collection of simple string algorithms, designed to be used in Big Data applications.
 *          It may be slower than LibC, but has a broader & cleaner interface, and a very short implementation
 *          targeting modern CPUs with AVX-512 and SVE and older CPUs with SWAR and auto-vecotrization.
 *
 *  @section    Operations potentially not worth optimizing in StringZilla
 *
 *  Some operations, like equality comparisons and relative order checking, almost always fail on some of the very
 *  first bytes in either string. This makes vectorization almost useless, unless huge strings are considered.
 *  Examples would be - computing the checksum of a long string, or checking 2 large binary strings for exact equality.
 *
 *  @section    Uncommon operations covered by StringZilla
 *
 *  * Reverse order search is rarely supported on par with normal order string scans.
 *  * Approximate string-matching is not the most common functionality for general-purpose string libraries.
 *
 *  @section    Compatibility with LibC and STL
 *
 *  The C++ Standard Templates Library provides an `std::string` and `std::string_view` classes with similar
 *  functionality. LibC, in turn, provides the "string.h" header with a set of functions for working with C strings.
 *  Both of those have a fairly constrained interface, as well as poor utilization of SIMD and SWAR techniques.
 *  StringZilla improves on both of those, by providing a more flexible interface, and better performance.
 *  If you are well familiar use the following index to find the equivalent functionality:
 *
 *  Covered:
 *      - void    *memchr(const void *, int, size_t); -> sz_find_byte
 *      - int      memcmp(const void *, const void *, size_t); -> sz_order, sz_equal
 *      - char    *strchr(const char *, int); -> sz_find_byte
 *      - int      strcmp(const char *, const char *); -> sz_order, sz_equal
 *      - size_t   strcspn(const char *, const char *); -> sz_find_last_from_set
 *      - size_t   strlen(const char *);-> sz_find_byte
 *      - size_t   strspn(const char *, const char *); -> sz_find_from_set
 *      - char    *strstr(const char *, const char *); -> sz_find
 *
 *  Not implemented:
 *      - void    *memccpy(void *restrict, const void *restrict, int, size_t);
 *      - void    *memcpy(void *restrict, const void *restrict, size_t);
 *      - void    *memmove(void *, const void *, size_t);
 *      - void    *memset(void *, int, size_t);
 *      - char    *strcat(char *restrict, const char *restrict);
 *      - int      strcoll(const char *, const char *);
 *      - char    *strcpy(char *restrict, const char *restrict);
 *      - char    *strdup(const char *);
 *      - char    *strerror(int);
 *      - int     *strerror_r(int, char *, size_t);
 *      - char    *strncat(char *restrict, const char *restrict, size_t);
 *      - int      strncmp(const char *, const char *, size_t);
 *      - char    *strncpy(char *restrict, const char *restrict, size_t);
 *      - char    *strpbrk(const char *, const char *);
 *      - char    *strrchr(const char *, int);
 *      - char    *strtok(char *restrict, const char *restrict);
 *      - char    *strtok_r(char *, const char *, char **);
 *      - size_t   strxfrm(char *restrict, const char *restrict, size_t);
 *
 *  LibC documentation: https://pubs.opengroup.org/onlinepubs/009695399/basedefs/string.h.html
 *  STL documentation: https://en.cppreference.com/w/cpp/header/string_view
 */
#ifndef STRINGZILLA_H_
#define STRINGZILLA_H_

/**
 *  @brief  Annotation for the public API symbols.
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#define SZ_PUBLIC __declspec(dllexport)
#elif __GNUC__ >= 4
#define SZ_PUBLIC __attribute__((visibility("default")))
#else
#define SZ_PUBLIC
#endif
#define SZ_INTERNAL inline static

/**
 *  @brief  Generally `NULL` is coming from locale.h, stddef.h, stdio.h, stdlib.h, string.h, time.h,
 *          and wchar.h, according to the C standard.
 */
#ifndef NULL
#define NULL ((void *)0)
#endif

/**
 *  @brief  Generally `CHAR_BIT` is coming from limits.h, according to the C standard.
 */
#ifndef CHAR_BIT
#define CHAR_BIT (8)
#endif

/**
 *  @brief  Compile-time assert macro similar to `static_assert` in C++.
 */
#define SZ_STATIC_ASSERT(condition, name)                \
    typedef struct {                                     \
        int static_assert_##name : (condition) ? 1 : -1; \
    } sz_static_assert_##name##_t

/**
 *  @brief  A misaligned load can be - trying to fetch eight consecutive bytes from an address
 *          that is not divisble by eight.
 *
 *  Most platforms support it, but there is no industry standard way to check for those.
 *  This value will mostly affect the performance of the serial (SWAR) backend.
 */
#ifndef SZ_USE_MISALIGNED_LOADS
#define SZ_USE_MISALIGNED_LOADS (1)
#endif

/**
 *  @brief  Cache-line width, that will affect the execution of some algorithms,
 *          like equality checks and relative order computing.
 */
#ifndef SZ_CACHE_LINE_WIDRTH
#define SZ_CACHE_LINE_WIDRTH (64)
#endif

/*
 *  Hardware feature detection.
 */
#ifndef SZ_USE_X86_AVX512
#ifdef __AVX512BW__
#define SZ_USE_X86_AVX512 1
#else
#define SZ_USE_X86_AVX512 0
#endif
#endif

#ifndef SZ_USE_X86_AVX2
#ifdef __AVX2__
#define SZ_USE_X86_AVX2 1
#else
#define SZ_USE_X86_AVX2 0
#endif
#endif

#ifndef SZ_USE_X86_SSE42
#ifdef __SSE4_2__
#define SZ_USE_X86_SSE42 1
#else
#define SZ_USE_X86_SSE42 0
#endif
#endif

#ifndef SZ_USE_ARM_NEON
#ifdef __ARM_NEON
#define SZ_USE_ARM_NEON 1
#else
#define SZ_USE_ARM_NEON 0
#endif
#endif

#ifndef SZ_USE_ARM_CRC32
#ifdef __ARM_FEATURE_CRC32
#define SZ_USE_ARM_CRC32 1
#else
#define SZ_USE_ARM_CRC32 0
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Analogous to `size_t` and `std::size_t`, unsigned integer, identical to pointer size.
 *          64-bit on most platforms where pointers are 64-bit.
 *          32-bit on platforms where pointers are 32-bit.
 */
#if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_WIN64)
#define sz_size_max 0xFFFFFFFFFFFFFFFFull
typedef unsigned long long sz_size_t;
typedef long long sz_ssize_t;
#else
#define sz_size_max 0xFFFFFFFFu
typedef unsigned sz_size_t;
typedef unsigned sz_ssize_t;
#endif
SZ_STATIC_ASSERT(sizeof(sz_size_t) == sizeof(void *), sz_size_t_must_be_pointer_size);
SZ_STATIC_ASSERT(sizeof(sz_ssize_t) == sizeof(void *), sz_ssize_t_must_be_pointer_size);

typedef unsigned char sz_u8_t;       /// Always 8 bits
typedef unsigned short sz_u16_t;     /// Always 16 bits
typedef int sz_i32_t;                /// Always 32 bits
typedef unsigned int sz_u32_t;       /// Always 32 bits
typedef unsigned long long sz_u64_t; /// Always 64 bits

typedef char *sz_ptr_t;        /// A type alias for `char *`
typedef char const *sz_cptr_t; /// A type alias for `char const *`

typedef signed char sz_error_cost_t; /// Character mismatch cost for fuzzy matching functions

typedef enum { sz_false_k = 0, sz_true_k = 1 } sz_bool_t;                        /// Only one relevant bit
typedef enum { sz_less_k = -1, sz_equal_k = 0, sz_greater_k = 1 } sz_ordering_t; /// Only three possible states: <=>

/**
 *  @brief  Tiny string-view structure. It's POD type, unlike the `std::string_view`.
 */
typedef struct sz_string_view_t {
    sz_cptr_t start;
    sz_size_t length;
} sz_string_view_t;

typedef union sz_u8_set_t {
    sz_u64_t _u64s[4];
    sz_u8_t _u8s[32];
} sz_u8_set_t;

SZ_PUBLIC void sz_u8_set_init(sz_u8_set_t *f) { f->_u64s[0] = f->_u64s[1] = f->_u64s[2] = f->_u64s[3] = 0; }
SZ_PUBLIC void sz_u8_set_add(sz_u8_set_t *f, sz_u8_t c) { f->_u64s[c >> 6] |= (1ull << (c & 63u)); }
SZ_PUBLIC sz_bool_t sz_u8_set_contains(sz_u8_set_t const *f, sz_u8_t c) {
    return (sz_bool_t)((f->_u64s[c >> 6] & (1ull << (c & 63u))) != 0);
}
SZ_PUBLIC void sz_u8_set_invert(sz_u8_set_t *f) {
    f->_u64s[0] ^= 0xFFFFFFFFFFFFFFFFull, f->_u64s[1] ^= 0xFFFFFFFFFFFFFFFFull, //
        f->_u64s[2] ^= 0xFFFFFFFFFFFFFFFFull, f->_u64s[3] ^= 0xFFFFFFFFFFFFFFFFull;
}

typedef sz_ptr_t (*sz_memory_allocate_t)(sz_size_t, void *);
typedef void (*sz_memory_free_t)(sz_ptr_t, sz_size_t, void *);

/**
 *  @brief  Some complex pattern matching algorithms may require memory allocations.
 */
typedef struct sz_memory_allocator_t {
    sz_memory_allocate_t allocate;
    sz_memory_free_t free;
    void *user_data;
} sz_memory_allocator_t;

#pragma region Basic Functionality

typedef sz_u64_t (*sz_hash_t)(sz_cptr_t, sz_size_t);
typedef sz_bool_t (*sz_equal_t)(sz_cptr_t, sz_cptr_t, sz_size_t);
typedef sz_ordering_t (*sz_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/**
 *  @brief  Computes the hash of a string.
 *
 *  @section    Why not use CRC32?
 *
 *  Cyclic Redundancy Check 32 is one of the most commonly used hash functions in Computer Science.
 *  It has in-hardware support on both x86 and Arm, for both 8-bit, 16-bit, 32-bit, and 64-bit words.
 *  In case of Arm more than one polynomial is supported. It is, however, somewhat limiting for Big Data
 *  usecases, which often have to deal with more than 4 Billion strings, making collisions unavoidable.
 *  Moreover, the existing SIMD approaches are tricky, combining general purpose computations with
 *  specialized intstructions, to utilize more silicon in every cycle.
 *
 *  Some of the best articles on CRC32:
 *  - Comprehensive derivation of approaches: https://github.com/komrad36/CRC
 *  - Faster computation for 4 KB buffers on x86: https://www.corsix.org/content/fast-crc32c-4k
 *  - Comparing different lookup tables: https://create.stephan-brumme.com/crc32
 *
 *  Some of the best open-source implementations:
 *  - Peter Cawley: https://github.com/corsix/fast-crc32
 *  - Stephan Brumme: https://github.com/stbrumme/crc32
 *
 *  @section    Modern Algorithms
 *
 *  MurmurHash from 2008 by Austin Appleby is one of the best known non-cryptographic hashes.
 *  It has a very short implementation and is capable of producing 32-bit and 128-bit hashes.
 *  https://github.com/aappleby/smhasher/tree/61a0530f28277f2e850bfc39600ce61d02b518de
 *
 *  The CityHash from 2011 by Google and the xxHash improve on that, better leveraging
 *  the superscalar nature of modern CPUs and producing 64-bit and 128-bit hashes.
 *  https://opensource.googleblog.com/2011/04/introducing-cityhash
 *  https://github.com/Cyan4973/xxHash
 *
 *  Neither of those functions are cryptographic, unlike MD5, SHA, and BLAKE algorithms.
 *  Most of those are based on the Merkle–Damgård construction, and aren't resistant to
 *  the length-extension attacks. Current state of the Art, might be the BLAKE3 algorithm.
 *  It's resistant to a broad range of attacks, can process 2 bytes per CPU cycle, and comes
 *  with a very optimized official implementation for C and Rust. It has the same 128-bit
 *  security level as the BLAKE2, and achieves its performance gains by reducing the number
 *  of mixing rounds, and processing data in 1 KiB chunks, which is great for longer strings,
 *  but may result in poor performance on short ones.
 *  https://en.wikipedia.org/wiki/BLAKE_(hash_function)#BLAKE3
 *  https://github.com/BLAKE3-team/BLAKE3
 *
 *  As shown, choosing the right hashing algorithm for your application can be crucial from
 *  both performance and security standpoint. Assuming, this functionality will be mostly used on
 *  multi-word short UTF8 strings, StringZilla implements a very simple scheme derived from MurMur3.
 *
 *  @param text     String to hash.
 *  @param length   Number of bytes in the text.
 *  @return         64-bit hash value.
 */
SZ_PUBLIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_u64_t sz_hash_avx512(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_u64_t sz_hash_neon(sz_cptr_t text, sz_size_t length);

/**
 *  @brief  Checks if two string are equal.
 *          Similar to `memcmp(a, b, length) == 0` in LibC and `a == b` in STL.
 *
 *  The implementation of this function is very similar to `sz_order`, but the usage patterns are different.
 *  This function is more often used in parsing, while `sz_order` is often used in sorting.
 *  It works best on platforms with cheap
 *
 *  @param a        First string to compare.
 *  @param b        Second string to compare.
 *  @param length   Number of bytes in both strings.
 *  @return         1 if strings match, 0 otherwise.
 */
SZ_PUBLIC sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
SZ_PUBLIC sz_bool_t sz_equal_serial(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
SZ_PUBLIC sz_bool_t sz_equal_avx512(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
SZ_PUBLIC sz_bool_t sz_equal_neon(sz_cptr_t a, sz_cptr_t b, sz_size_t length);

/**
 *  @brief  Estimates the relative order of two strings. Equivalent to `memcmp(a, b, length)` in LibC.
 *          Can be used on different length strings.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *  @return         Negative if (a < b), positive if (a > b), zero if they are equal.
 */
SZ_PUBLIC sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);
SZ_PUBLIC sz_ordering_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/**
 *  @brief  Equivalent to `for (char & c : text) c = tolower(c)`.
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122].
 *  So there are 26 english letters, shifted by 32 values, meaning that a conversion
 *  can be done by flipping the 5th bit each inappropriate character byte. This, however,
 *  breaks for extended ASCII, so a different solution is needed.
 *  http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 *
 *  @param text     String to be normalized.
 *  @param length   Number of bytes in the string.
 *  @param result   Output string, can point to the same address as ::text.
 */
SZ_PUBLIC void sz_tolower(sz_cptr_t text, sz_size_t length, sz_ptr_t result);

/**
 *  @brief  Equivalent to `for (char & c : text) c = toupper(c)`.
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122].
 *  So there are 26 english letters, shifted by 32 values, meaning that a conversion
 *  can be done by flipping the 5th bit each inappropriate character byte. This, however,
 *  breaks for extended ASCII, so a different solution is needed.
 *  http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 *
 *  @param text     String to be normalized.
 *  @param length   Number of bytes in the string.
 *  @param result   Output string, can point to the same address as ::text.
 */
SZ_PUBLIC void sz_toupper(sz_cptr_t text, sz_size_t length, sz_ptr_t result);

/**
 *  @brief  Equivalent to `for (char & c : text) c = toascii(c)`.
 *
 *  @param text     String to be normalized.
 *  @param length   Number of bytes in the string.
 *  @param result   Output string, can point to the same address as ::text.
 */
SZ_PUBLIC void sz_toascii(sz_cptr_t text, sz_size_t length, sz_ptr_t result);

#pragma endregion

#pragma region Fast Substring Search

typedef sz_cptr_t (*sz_find_byte_t)(sz_cptr_t, sz_size_t, sz_cptr_t);
typedef sz_cptr_t (*sz_find_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/**
 *  @brief  Locates first matching byte in a string. Equivalent to `memchr(haystack, *needle, h_length)` in LibC.
 *
 *  X86_64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memchr.S
 *  Aarch64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/aarch64/memchr.S
 *
 *  @param haystack Haystack - the string to search in.
 *  @param h_length Number of bytes in the haystack.
 *  @param needle   Needle - single-byte substring to find.
 *  @return         Address of the first match.
 */
SZ_PUBLIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_avx512(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/**
 *  @brief  Locates last matching byte in a string. Equivalent to `memrchr(haystack, *needle, h_length)` in LibC.
 *
 *  X86_64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memrchr.S
 *  Aarch64 implementation: missing
 *
 *  @param haystack Haystack - the string to search in.
 *  @param h_length Number of bytes in the haystack.
 *  @param needle   Needle - single-byte substring to find.
 *  @return         Address of the last match.
 */
SZ_PUBLIC sz_cptr_t sz_find_last_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/** @copydoc sz_find_last_byte */
SZ_PUBLIC sz_cptr_t sz_find_last_byte_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/** @copydoc sz_find_last_byte */
SZ_PUBLIC sz_cptr_t sz_find_last_byte_avx512(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/**
 *  @brief  Locates first matching substring.
 *          Equivalient to `memmem(haystack, h_length, needle, n_length)` in LibC.
 *          Similar to `strstr(haystack, needle)` in LibC, but requires known length.
 *
 *  Uses different algorithms for different needle lengths and backends:
 *
 *  > Naive exact matching for 1-, 2-, 3-, and 4-character-long needles.
 *  > Bitap "Shift Or" (Baeza-Yates-Gonnet) algorithm for serial (SWAR) backend.
 *  > Two-way heuristic for longer needles with SIMD backends.
 *
 *  @section Reading Materials
 *
 *  Exact String Matching Algorithms in Java: https://www-igm.univ-mlv.fr/~lecroq/string
 *  SIMD-friendly algorithms for substring searching: http://0x80.pl/articles/simd-strfind.html
 *
 *  @param haystack Haystack - the string to search in.
 *  @param h_length Number of bytes in the haystack.
 *  @param needle   Needle - substring to find.
 *  @param n_length Number of bytes in the needle.
 *  @return         Address of the first match.
 */
SZ_PUBLIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_avx512(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/**
 *  @brief  Locates the last matching substring.
 *
 *  Uses different algorithms for different needle lengths and backends:
 *
 *  > Naive exact matching for 1-, 2-, 3-, and 4-character-long needles.
 *  > Bitap "Shift Or" (Baeza-Yates-Gonnet) algorithm for serial (SWAR) backend.
 *  > Two-way heuristic for longer needles with SIMD backends.
 *
 *  @section Reading Materials
 *
 *  Exact String Matching Algorithms in Java: https://www-igm.univ-mlv.fr/~lecroq/string
 *  SIMD-friendly algorithms for substring searching: http://0x80.pl/articles/simd-strfind.html
 *
 *  @param haystack Haystack - the string to search in.
 *  @param h_length Number of bytes in the haystack.
 *  @param needle   Needle - substring to find.
 *  @param n_length Number of bytes in the needle.
 *  @return         Address of the last match.
 */
SZ_PUBLIC sz_cptr_t sz_find_last(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find_last */
SZ_PUBLIC sz_cptr_t sz_find_last_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find_last */
SZ_PUBLIC sz_cptr_t sz_find_last_avx512(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find_last */
SZ_PUBLIC sz_cptr_t sz_find_last_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/**
 *  @brief  Finds the first character present from the ::set, present in ::text.
 *          Equivalent to `strspn(text, accepted)` and `strcspn(text, rejected)` in LibC.
 *          May have identical implementation and performance to ::sz_find_last_from_set.
 *
 *  @param text     String to be trimmed.
 *  @param accepted Set of accepted characters.
 *  @return         Number of bytes forming the prefix.
 */
SZ_PUBLIC sz_cptr_t sz_find_from_set(sz_cptr_t text, sz_size_t length, sz_u8_set_t *set);
SZ_PUBLIC sz_cptr_t sz_find_from_set_serial(sz_cptr_t text, sz_size_t length, sz_u8_set_t *set);

/**
 *  @brief  Finds the last character present from the ::set, present in ::text.
 *          Equivalent to `strspn(text, accepted)` and `strcspn(text, rejected)` in LibC.
 *          May have identical implementation and performance to ::sz_find_from_set.
 *
 *  Useful for parsing, when we want to skip a set of characters. Examples:
 *  * 6 whitespaces: " \t\n\r\v\f".
 *  * 16 digits forming a float number: "0123456789,.eE+-".
 *  * 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  * 2 JSON string special characters useful to locate the end of the string: "\"\\".
 *
 *  @param text     String to be trimmed.
 *  @param rejected Set of rejected characters.
 *  @return         Number of bytes forming the prefix.
 */
SZ_PUBLIC sz_cptr_t sz_find_last_from_set(sz_cptr_t text, sz_size_t length, sz_u8_set_t *set);
SZ_PUBLIC sz_cptr_t sz_find_last_from_set_serial(sz_cptr_t text, sz_size_t length, sz_u8_set_t *set);

SZ_PUBLIC sz_cptr_t sz_find_bounded_regex(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length,
                                          sz_size_t bound, sz_memory_allocator_t const *alloc);
SZ_PUBLIC sz_cptr_t sz_find_last_bounded_regex(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle,
                                               sz_size_t n_length, sz_size_t bound, sz_memory_allocator_t const *alloc);

#pragma endregion

#pragma region String Similarity Measures

/**
 *  @brief  Computes Levenshtein edit-distance between two strings.
 *          Similar to the Needleman–Wunsch algorithm. Often used in fuzzy string matching.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *  @param alloc    Temporary memory allocator, that will allocate at most two rows of the Levenshtein matrix.
 *  @param bound    Upper bound on the distance, that allows us to exit early.
 *  @return         Unsigned edit distance.
 */
SZ_PUBLIC sz_size_t sz_levenshtein(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                   sz_size_t bound, sz_memory_allocator_t const *alloc);

/** @copydoc sz_levenshtein */
SZ_PUBLIC sz_size_t sz_levenshtein_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                          sz_size_t bound, sz_memory_allocator_t const *alloc);

/** @copydoc sz_levenshtein */
SZ_PUBLIC sz_size_t sz_levenshtein_avx512(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                          sz_size_t bound, sz_memory_allocator_t const *alloc);

/**
 *  @brief  Estimates the amount of temporary memory required to efficiently compute the weighted edit distance.
 *
 *  @param a_length Number of bytes in the first string.
 *  @param b_length Number of bytes in the second string.
 *  @return         Number of bytes to allocate for temporary memory.
 */
SZ_PUBLIC sz_size_t sz_alignment_score_memory_needed(sz_size_t a_length, sz_size_t b_length);

/**
 *  @brief  Computes Levenshtein edit-distance between two strings, parameterized for gap and substitution penalties.
 *          Similar to the Needleman–Wunsch algorithm. Often used in bioinformatics and cheminformatics.
 *
 *  This function is equivalent to the default Levenshtein distance implementation with the ::gap parameter set
 *  to one, and the ::subs matrix formed of all ones except for the main diagonal, which is zeros.
 *  Unlike the default Levenshtein implementaion, this can't be bounded, as the substitution costs can be both positive
 *  and negative, meaning that the distance isn't monotonically growing as we go through the strings.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *  @param gap      Penalty cost for gaps - insertions and removals.
 *  @param subs     Substitution costs matrix with 256 x 256 values for all pais of characters.
 *  @param alloc    Temporary memory allocator, that will allocate at most two rows of the Levenshtein matrix.
 *  @return         Signed score ~ edit distance.
 */
SZ_PUBLIC sz_ssize_t sz_alignment_score(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                        sz_error_cost_t gap, sz_error_cost_t const *subs,                 //
                                        sz_memory_allocator_t const *alloc);

/** @copydoc sz_alignment_score */
SZ_PUBLIC sz_ssize_t sz_alignment_score_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                               sz_error_cost_t gap, sz_error_cost_t const *subs,                 //
                                               sz_memory_allocator_t const *alloc);
/** @copydoc sz_alignment_score */
SZ_PUBLIC sz_ssize_t sz_alignment_score_avx512(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                               sz_error_cost_t gap, sz_error_cost_t const *subs,                 //
                                               sz_memory_allocator_t const *alloc);

#pragma endregion

#pragma region String Sequences

struct sz_sequence_t;

typedef sz_cptr_t (*sz_sequence_member_start_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_size_t (*sz_sequence_member_length_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_bool_t (*sz_sequence_predicate_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_bool_t (*sz_sequence_comparator_t)(struct sz_sequence_t const *, sz_size_t, sz_size_t);
typedef sz_bool_t (*sz_string_is_less_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

typedef struct sz_sequence_t {
    sz_u64_t *order;
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
SZ_PUBLIC void sz_sequence_from_u32tape(sz_cptr_t *start, sz_u32_t const *offsets, sz_size_t count,
                                        sz_sequence_t *sequence);

/**
 *  @brief  Initiates the sequence structure from a tape layout, used by Apache Arrow.
 *          Expects ::offsets to contains `count + 1` entries, the last pointing at the end
 *          of the last string, indicating the total length of the ::tape.
 */
SZ_PUBLIC void sz_sequence_from_u64tape(sz_cptr_t *start, sz_u64_t const *offsets, sz_size_t count,
                                        sz_sequence_t *sequence);

/**
 *  @brief  Similar to `std::partition`, given a predicate splits the sequence into two parts.
 *          The algorithm is unstable, meaning that elements may change relative order, as long
 *          as they are in the right partition. This is the simpler algorithm for partitioning.
 */
SZ_PUBLIC sz_size_t sz_partition(sz_sequence_t *sequence, sz_sequence_predicate_t predicate);

/**
 *  @brief  Inplace `std::set_union` for two consecutive chunks forming the same continuous `sequence`.
 *
 *  @param partition The number of elements in the first sub-sequence in `sequence`.
 *  @param less Comparison function, to determine the lexicographic ordering.
 */
SZ_PUBLIC void sz_merge(sz_sequence_t *sequence, sz_size_t partition, sz_sequence_comparator_t less);

/**
 *  @brief  Sorting algorithm, combining Radix Sort for the first 32 bits of every word
 *          and a follow-up by a more conventional sorting procedure on equally prefixed parts.
 */
SZ_PUBLIC void sz_sort(sz_sequence_t *sequence);

/**
 *  @brief  Partial sorting algorithm, combining Radix Sort for the first 32 bits of every word
 *          and a follow-up by a more conventional sorting procedure on equally prefixed parts.
 */
SZ_PUBLIC void sz_sort_partial(sz_sequence_t *sequence, sz_size_t n);

/**
 *  @brief  Intro-Sort algorithm that supports custom comparators.
 */
SZ_PUBLIC void sz_sort_intro(sz_sequence_t *sequence, sz_sequence_comparator_t less);

#pragma endregion

#pragma region Compiler Extensions and Helper Functions

/*
 *  Intrinsics aliases for MSVC, GCC, and Clang.
 */
#if defined(_MSC_VER)
SZ_INTERNAL int sz_u64_popcount(sz_u64_t x) { return __popcnt64(x); }
SZ_INTERNAL int sz_u64_ctz(sz_u64_t x) { return _tzcnt_u64(x); }
SZ_INTERNAL int sz_u64_clz(sz_u64_t x) { return _lzcnt_u64(x); }
SZ_INTERNAL sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return _byteswap_uint64(val); }
#else
SZ_INTERNAL int sz_u64_popcount(sz_u64_t x) { return __builtin_popcountll(x); }
SZ_INTERNAL int sz_u64_ctz(sz_u64_t x) { return __builtin_ctzll(x); }
SZ_INTERNAL int sz_u64_clz(sz_u64_t x) { return __builtin_clzll(x); }
SZ_INTERNAL sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return __builtin_bswap64(val); }
#endif

SZ_INTERNAL sz_u64_t sz_u64_rotl(sz_u64_t x, sz_u64_t r) { return (x << r) | (x >> (64 - r)); }

/*
 *  Efficiently computing the minimum and maximum of two or three values can be tricky.
 *  The simple branching baseline would be:
 *
 *      x < y ? x : y                               // 1 conditional move
 *
 *  Branchless approach is well known for signed integers, but it doesn't apply to unsigned ones.
 *  https://stackoverflow.com/questions/514435/templatized-branchless-int-max-min-function
 *  https://graphics.stanford.edu/~seander/bithacks.html#IntegerMinOrMax
 *  Using only bitshifts for singed integers it would be:
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
#define sz_min_of_three(x, y, z) sz_min_of_two(x, sz_min_of_two(y, z))

/**
 *  @brief Branchless minimum function for two integers.
 */
SZ_INTERNAL sz_i32_t sz_i32_min_of_two(sz_i32_t x, sz_i32_t y) { return y + ((x - y) & (x - y) >> 31); }

/**
 *  @brief  Compute the logarithm base 2 of an integer.
 *
 *  @note If n is 0, the function returns 0 to avoid undefined behavior.
 *  @note This function uses compiler-specific intrinsics or built-ins
 *        to achieve the computation. It's designed to work with GCC/Clang and MSVC.
 */
SZ_INTERNAL sz_size_t sz_size_log2i(sz_size_t n) {
    if (n == 0) return 0;

#ifdef _WIN64
#if defined(_MSC_VER)
    unsigned long index;
    if (_BitScanReverse64(&index, n)) return index;
    return 0; // This line might be redundant due to the initial check, but it's safer to include it.
#else
    return 63 - __builtin_clzll(n);
#endif
#elif defined(_WIN32)
#if defined(_MSC_VER)
    unsigned long index;
    if (_BitScanReverse(&index, n)) return index;
    return 0; // Same note as above.
#else
    return 31 - __builtin_clz(n);
#endif
#else
// Handle non-Windows platforms. You can further differentiate between 32-bit and 64-bit if needed.
#if defined(__LP64__)
    return 63 - __builtin_clzll(n);
#else
    return 31 - __builtin_clz(n);
#endif
#endif
}

/**
 *  @brief  Helper structure to simpify work with 16-bit words.
 *  @see    sz_u16_load
 */
typedef union sz_u16_parts_t {
    sz_u16_t u16;
    sz_u8_t u8s[2];
} sz_u16_parts_t;

/**
 *  @brief Load a 16-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u16_parts_t sz_u16_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u16_parts_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    return result;
#elif defined(_MSC_VER)
    return *((__unaligned sz_u16_parts_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u16_parts_t const *result = (sz_u16_parts_t const *)ptr;
    return *result;
#endif
}

/**
 *  @brief  Helper structure to simpify work with 32-bit words.
 *  @see    sz_u32_load
 */
typedef union sz_u32_parts_t {
    sz_u32_t u32;
    sz_u16_t u16s[2];
    sz_u8_t u8s[4];
} sz_u32_parts_t;

/**
 *  @brief Load a 32-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u32_parts_t sz_u32_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u32_parts_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    result.u8s[2] = ptr[2];
    result.u8s[3] = ptr[3];
    return result;
#elif defined(_MSC_VER)
    return *((__unaligned sz_u32_parts_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u32_parts_t const *result = (sz_u32_parts_t const *)ptr;
    return *result;
#endif
}

/**
 *  @brief  Helper structure to simpify work with 64-bit words.
 *  @see    sz_u64_load
 */
typedef union sz_u64_parts_t {
    sz_u64_t u64;
    sz_u32_t u32s[2];
    sz_u16_t u16s[4];
    sz_u8_t u8s[8];
} sz_u64_parts_t;

/**
 *  @brief Load a 64-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u64_parts_t sz_u64_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u64_parts_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    result.u8s[2] = ptr[2];
    result.u8s[3] = ptr[3];
    result.u8s[4] = ptr[4];
    result.u8s[5] = ptr[5];
    result.u8s[6] = ptr[6];
    result.u8s[7] = ptr[7];
    return result;
#elif defined(_MSC_VER)
    return *((__unaligned sz_u64_parts_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u64_parts_t const *result = (sz_u64_parts_t const *)ptr;
    return *result;
#endif
}

#pragma endregion

#pragma region Serial Implementation

SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t start, sz_size_t length) {

    sz_u64_t const c1 = 0x87c37b91114253d5ull;
    sz_u64_t const c2 = 0x4cf5ad432745937full;
    sz_u64_parts_t k1, k2;
    sz_u64_t h1, h2;

    k1.u64 = k2.u64 = 0;
    h1 = h2 = length;

    for (; length >= 16; length -= 16, start += 16) {
        k1 = sz_u64_load(start);
        k2 = sz_u64_load(start + 8);

        k1.u64 *= c1;
        k1.u64 = sz_u64_rotl(k1.u64, 31);
        k1.u64 *= c2;
        h1 ^= k1.u64;

        h1 = sz_u64_rotl(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2.u64 *= c2;
        k2.u64 = sz_u64_rotl(k2.u64, 33);
        k2.u64 *= c1;
        h2 ^= k2.u64;

        h2 = sz_u64_rotl(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    // Similar to xxHash, WaterHash:
    // 0 - 3 bytes: https://github.com/Cyan4973/xxHash/blob/f91df681b034d78c7ce87de66f0f78a1e40e7bfb/xxhash.h#L4515
    // 4 - 8 bytes: https://github.com/Cyan4973/xxHash/blob/f91df681b034d78c7ce87de66f0f78a1e40e7bfb/xxhash.h#L4537
    // 9 - 16 bytes: https://github.com/Cyan4973/xxHash/blob/f91df681b034d78c7ce87de66f0f78a1e40e7bfb/xxhash.h#L4553
    // 17 - 128 bytes: https://github.com/Cyan4973/xxHash/blob/f91df681b034d78c7ce87de66f0f78a1e40e7bfb/xxhash.h#L4640
    // Long sequences: https://github.com/Cyan4973/xxHash/blob/f91df681b034d78c7ce87de66f0f78a1e40e7bfb/xxhash.h#L5906
    switch (length & 15) {
    case 15: k2.u8s[6] = start[14];
    case 14: k2.u8s[5] = start[13];
    case 13: k2.u8s[4] = start[12];
    case 12: k2.u8s[3] = start[11];
    case 11: k2.u8s[2] = start[10];
    case 10: k2.u8s[1] = start[9];
    case 9:
        k2.u8s[0] = start[8];
        k2.u64 *= c2;
        k2.u64 = sz_u64_rotl(k2.u64, 33);
        k2.u64 *= c1;
        h2 ^= k2.u64;

    case 8: k1.u8s[7] = start[7];
    case 7: k1.u8s[6] = start[6];
    case 6: k1.u8s[5] = start[5];
    case 5: k1.u8s[4] = start[4];
    case 4: k1.u8s[3] = start[3];
    case 3: k1.u8s[2] = start[2];
    case 2: k1.u8s[1] = start[1];
    case 1:
        k1.u8s[0] = start[0];
        k1.u64 *= c1;
        k1.u64 = sz_u64_rotl(k1.u64, 31);
        k1.u64 *= c2;
        h1 ^= k1.u64;
    };

    // We almost entirely avoid the final mixing step
    // https://github.com/aappleby/smhasher/blob/61a0530f28277f2e850bfc39600ce61d02b518de/src/MurmurHash3.cpp#L317
    return h1 + h2;
}

/**
 *  @brief  Byte-level equality comparison between two strings.
 *          If unaligned loads are allowed, uses a switch-table to avoid loops on short strings.
 */
SZ_PUBLIC sz_bool_t sz_equal_serial(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    sz_cptr_t const a_end = a + length;
    while (a != a_end && *a == *b) a++, b++;
    return (sz_bool_t)(a_end == a);
}

SZ_PUBLIC sz_cptr_t sz_find_from_set_serial(sz_cptr_t text, sz_size_t length, sz_u8_set_t *set) {
    for (sz_cptr_t const end = text + length; text != end; ++text)
        if (sz_u8_set_contains(set, *text)) return text;
    return NULL;
}

SZ_PUBLIC sz_cptr_t sz_find_last_from_set_serial(sz_cptr_t text, sz_size_t length, sz_u8_set_t *set) {
    sz_cptr_t const end = text;
    for (text += length; text != end; --text)
        if (sz_u8_set_contains(set, *(text - 1))) return text - 1;
    return NULL;
}

/**
 *  @brief  Byte-level lexicographic order comparison of two strings.
 */
SZ_PUBLIC sz_ordering_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
#if SZ_USE_MISALIGNED_LOADS
    sz_bool_t a_shorter = (sz_bool_t)(a_length < b_length);
    sz_size_t min_length = a_shorter ? a_length : b_length;
    sz_cptr_t min_end = a + min_length;
    for (sz_u64_parts_t a_vec, b_vec; a + 8 <= min_end; a += 8, b += 8) {
        a_vec.u64 = sz_u64_bytes_reverse(sz_u64_load(a).u64);
        b_vec.u64 = sz_u64_bytes_reverse(sz_u64_load(b).u64);
        if (a_vec.u64 != b_vec.u64) return ordering_lookup[a_vec.u64 < b_vec.u64];
    }
#endif
    for (; a != min_end; ++a, ++b)
        if (*a != *b) return ordering_lookup[*a < *b];
    return a_length != b_length ? ordering_lookup[a_shorter] : sz_equal_k;
}

/**
 *  @brief  Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each byte signifies a match.
 */
SZ_INTERNAL sz_u64_t sz_u64_each_byte_equal(sz_u64_t a, sz_u64_t b) {
    sz_u64_t match_indicators = ~(a ^ b);
    // The match is valid, if every bit within each byte is set.
    // For that take the bottom 7 bits of each byte, add one to them,
    // and if this sets the top bit to one, then all the 7 bits are ones as well.
    match_indicators = ((match_indicators & 0x7F7F7F7F7F7F7F7Full) + 0x0101010101010101ull) &
                       ((match_indicators & 0x8080808080808080ull));
    return match_indicators;
}

/**
 *  @brief  Find the first occurrence of a @b single-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *          Identical to `memchr(haystack, needle[0], haystack_length)`.
 */
SZ_PUBLIC sz_cptr_t sz_find_byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_cptr_t const h_end = h + h_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h < h_end; ++h)
        if (*h == *n) return h;

    // Broadcast the n into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_parts_t n_vec;
    n_vec.u64 = (sz_u64_t)n[0] * 0x0101010101010101ull;
    for (; h + 8 <= h_end; h += 8) {
        sz_u64_t h_vec = *(sz_u64_t const *)h;
        sz_u64_t match_indicators = sz_u64_each_byte_equal(h_vec, n_vec.u64);
        if (match_indicators != 0) return h + sz_u64_ctz(match_indicators) / 8;
    }

    // Handle the misaligned tail.
    for (; h < h_end; ++h)
        if (*h == *n) return h;
    return NULL;
}

/**
 *  @brief  Find the last occurrence of a @b single-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *          Identical to `memrchr(haystack, needle[0], haystack_length)`.
 */
sz_cptr_t sz_find_last_byte_serial(sz_cptr_t h, sz_size_t h_len, sz_cptr_t needle) {

    sz_cptr_t const h_start = h;

    // Reposition the `h` pointer to the last character, as we will be walking backwards.
    h = h + h_len - 1;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)(h + 1) & 7ull) && h >= h_start; --h)
        if (*h == *needle) return h;

    // Broadcast the needle into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_parts_t n_vec;
    n_vec.u64 = (sz_u64_t)needle[0] * 0x0101010101010101ull;
    for (; h >= h_start + 8; h -= 8) {
        sz_u64_t h_vec = *(sz_u64_t const *)(h - 8);
        sz_u64_t match_indicators = sz_u64_each_byte_equal(h_vec, n_vec.u64);
        if (match_indicators != 0) return h - 8 + sz_u64_clz(match_indicators) / 8;
    }

    for (; h >= h_start; --h)
        if (*h == *needle) return h;
    return NULL;
}

/**
 *  @brief  2Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 2byte signifies a match.
 */
SZ_INTERNAL sz_u64_t sz_u64_each_2byte_equal(sz_u64_t a, sz_u64_t b) {
    sz_u64_t match_indicators = ~(a ^ b);
    // The match is valid, if every bit within each 2byte is set.
    // For that take the bottom 15 bits of each 2byte, add one to them,
    // and if this sets the top bit to one, then all the 15 bits are ones as well.
    match_indicators = ((match_indicators & 0x7FFF7FFF7FFF7FFFull) + 0x0001000100010001ull) &
                       ((match_indicators & 0x8000800080008000ull));
    return match_indicators;
}

/**
 *  @brief  Find the first occurrence of a @b two-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
SZ_INTERNAL sz_cptr_t sz_find_2byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_cptr_t const h_end = h + h_length;

    // This code simulates hyper-scalar execution, analyzing 7 offsets at a time.
    sz_u64_parts_t h_vec, n_vec, matches_odd_vec, matches_even_vec;
    n_vec.u64 = 0;
    n_vec.u8s[0] = n[0];
    n_vec.u8s[1] = n[1];
    n_vec.u64 *= 0x0001000100010001ull;

    for (; h + 8 <= h_end; h += 7) {
        h_vec = sz_u64_load(h);
        matches_even_vec.u64 = sz_u64_each_2byte_equal(h_vec.u64, n_vec.u64);
        matches_odd_vec.u64 = sz_u64_each_2byte_equal(h_vec.u64 >> 8, n_vec.u64);

        if (matches_even_vec.u64 + matches_odd_vec.u64) {
            sz_u64_t match_indicators = (matches_even_vec.u64 >> 8) | (matches_odd_vec.u64);
            return h + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; h + 2 <= h_end; ++h)
        if (h[0] == n[0] && h[1] == n[1]) return h;
    return NULL;
}

/**
 *  @brief  Find the first occurrence of a three-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
SZ_INTERNAL sz_cptr_t sz_find_3byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_cptr_t const h_end = h + h_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h + 3 <= h_end; ++h)
        if (h[0] == n[0] && h[1] == n[1] && h[2] == n[2]) return h;

    // This code simulates hyper-scalar execution, analyzing 6 offsets at a time.
    // We have two unused bytes at the end.
    sz_u64_parts_t h_vec, n_vec, matches_first_vec, matches_second_vec, matches_third_vec;
    n_vec.u8s[2] = n[0]; // broadcast `n` into `nn`
    n_vec.u8s[3] = n[1]; // broadcast `n` into `nn`
    n_vec.u8s[4] = n[2]; // broadcast `n` into `nn`
    n_vec.u8s[5] = n[0]; // broadcast `n` into `nn`
    n_vec.u8s[6] = n[1]; // broadcast `n` into `nn`
    n_vec.u8s[7] = n[2]; // broadcast `n` into `nn`

    for (; h + 8 <= h_end; h += 6) {
        h_vec = sz_u64_load(h);
        matches_first_vec.u64 = ~(h_vec.u64 ^ n_vec.u64);
        matches_second_vec.u64 = ~((h_vec.u64 << 8) ^ n_vec.u64);
        matches_third_vec.u64 = ~((h_vec.u64 << 16) ^ n_vec.u64);
        // For every first match - 3 chars (24 bits) must be identical.
        // For that merge every byte state and then combine those three-way.
        matches_first_vec.u64 &= matches_first_vec.u64 >> 1;
        matches_first_vec.u64 &= matches_first_vec.u64 >> 2;
        matches_first_vec.u64 &= matches_first_vec.u64 >> 4;
        matches_first_vec.u64 = (matches_first_vec.u64 >> 16) & (matches_first_vec.u64 >> 8) &
                                (matches_first_vec.u64 >> 0) & 0x0000010000010000;

        // For every second match - 3 chars (24 bits) must be identical.
        // For that merge every byte state and then combine those three-way.
        matches_second_vec.u64 &= matches_second_vec.u64 >> 1;
        matches_second_vec.u64 &= matches_second_vec.u64 >> 2;
        matches_second_vec.u64 &= matches_second_vec.u64 >> 4;
        matches_second_vec.u64 = (matches_second_vec.u64 >> 16) & (matches_second_vec.u64 >> 8) &
                                 (matches_second_vec.u64 >> 0) & 0x0000010000010000;

        // For every third match - 3 chars (24 bits) must be identical.
        // For that merge every byte state and then combine those three-way.
        matches_third_vec.u64 &= matches_third_vec.u64 >> 1;
        matches_third_vec.u64 &= matches_third_vec.u64 >> 2;
        matches_third_vec.u64 &= matches_third_vec.u64 >> 4;
        matches_third_vec.u64 = (matches_third_vec.u64 >> 16) & (matches_third_vec.u64 >> 8) &
                                (matches_third_vec.u64 >> 0) & 0x0000010000010000;

        sz_u64_t match_indicators =
            matches_first_vec.u64 | (matches_second_vec.u64 >> 8) | (matches_third_vec.u64 >> 16);
        if (match_indicators != 0) return h + sz_u64_ctz(match_indicators) / 8;
    }

    for (; h + 3 <= h_end; ++h)
        if (h[0] == n[0] && h[1] == n[1] && h[2] == n[2]) return h;
    return NULL;
}

/**
 *  @brief  Find the first occurrence of a @b four-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
SZ_INTERNAL sz_cptr_t sz_find_4byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_cptr_t const h_end = h + h_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h + 4 <= h_end; ++h)
        if (h[0] == n[0] && h[1] == n[1] && h[2] == n[2] && h[3] == n[3]) return h;

    // This code simulates hyper-scalar execution, analyzing 4 offsets at a time.
    sz_u64_t nn = (sz_u64_t)(n[0] << 0) | ((sz_u64_t)(n[1]) << 8) | ((sz_u64_t)(n[2]) << 16) | ((sz_u64_t)(n[3]) << 24);
    nn |= nn << 32;

    //
    unsigned char offset_in_slice[16] = {0};
    offset_in_slice[0x2] = offset_in_slice[0x6] = offset_in_slice[0xA] = offset_in_slice[0xE] = 1;
    offset_in_slice[0x4] = offset_in_slice[0xC] = 2;
    offset_in_slice[0x8] = 3;

    // We can perform 5 comparisons per load, but it's easier to perform 4, minimizing the size of the lookup table.
    for (; h + 8 <= h_end; h += 4) {
        sz_u64_t h_vec = sz_u64_load(h).u64;
        sz_u64_t h01 = (h_vec & 0x00000000FFFFFFFF) | ((h_vec & 0x000000FFFFFFFF00) << 24);
        sz_u64_t h23 = ((h_vec & 0x0000FFFFFFFF0000) >> 16) | ((h_vec & 0x00FFFFFFFF000000) << 8);
        sz_u64_t h01_indicators = ~(h01 ^ nn);
        sz_u64_t h23_indicators = ~(h23 ^ nn);

        // For every first match - 4 chars (32 bits) must be identical.
        h01_indicators &= h01_indicators >> 1;
        h01_indicators &= h01_indicators >> 2;
        h01_indicators &= h01_indicators >> 4;
        h01_indicators &= h01_indicators >> 8;
        h01_indicators &= h01_indicators >> 16;
        h01_indicators &= 0x0000000100000001;

        // For every first match - 4 chars (32 bits) must be identical.
        h23_indicators &= h23_indicators >> 1;
        h23_indicators &= h23_indicators >> 2;
        h23_indicators &= h23_indicators >> 4;
        h23_indicators &= h23_indicators >> 8;
        h23_indicators &= h23_indicators >> 16;
        h23_indicators &= 0x0000000100000001;

        if (h01_indicators + h23_indicators) {
            // Assuming we have performed 4 comparisons, we can only have 2^4=16 outcomes.
            // Which is small enough for a lookup table.
            unsigned char match_indicators = (unsigned char)(    //
                (h01_indicators >> 31) | (h01_indicators << 0) | //
                (h23_indicators >> 29) | (h23_indicators << 2));
            return h + offset_in_slice[match_indicators];
        }
    }

    for (; h + 4 <= h_end; ++h)
        if (h[0] == n[0] && h[1] == n[1] && h[2] == n[2] && h[3] == n[3]) return h;
    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns under @b 8-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t sz_find_under8byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    sz_u8_t running_match = 0xFF;
    sz_u8_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { pattern_mask[n[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[h[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns under @b 8-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t sz_find_last_under8byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    sz_u8_t running_match = 0xFF;
    sz_u8_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { pattern_mask[n[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[h[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns under @b 16-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t sz_find_under16byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    sz_u16_t running_match = 0xFFFF;
    sz_u16_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { pattern_mask[n[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[h[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns under @b 16-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t sz_find_last_under16byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                      sz_size_t n_length) {

    sz_u16_t running_match = 0xFFFF;
    sz_u16_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { pattern_mask[n[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[h[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns under @b 32-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t sz_find_under32byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    sz_u32_t running_match = 0xFFFFFFFF;
    sz_u32_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFFFFFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { pattern_mask[n[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[h[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns under @b 32-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t sz_find_last_under32byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                      sz_size_t n_length) {

    sz_u32_t running_match = 0xFFFFFFFF;
    sz_u32_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFFFFFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { pattern_mask[n[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[h[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns under @b 64-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t sz_find_under64byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { pattern_mask[n[i]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[h[i]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns under @b 64-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t sz_find_last_under64byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                      sz_size_t n_length) {

    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { pattern_mask[n[n_length - i - 1]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[h[h_length - i - 1]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

SZ_INTERNAL sz_cptr_t sz_find_over64byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    sz_size_t const prefix_length = 64;
    sz_size_t const suffix_length = n_length - prefix_length;
    while (true) {
        sz_cptr_t found = sz_find_under64byte_serial(h, h_length, n, prefix_length);
        if (!found) return NULL;

        // Verify the remaining part of the needle
        sz_size_t remaining = h_length - (found - h);
        if (remaining < suffix_length) return NULL;
        if (sz_equal_serial(found + prefix_length, n + prefix_length, suffix_length)) return found;

        // Adjust the position.
        h = found + 1;
        h_length = remaining - 1;
    }
}

SZ_INTERNAL sz_cptr_t sz_find_last_over64byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    sz_size_t const suffix_length = 64;
    sz_size_t const prefix_length = n_length - suffix_length;
    while (true) {
        sz_cptr_t found = sz_find_under64byte_serial(h, h_length, n + prefix_length, suffix_length);
        if (!found) return NULL;

        // Verify the remaining part of the needle
        sz_size_t remaining = found - h;
        if (remaining < prefix_length) return NULL;
        if (sz_equal_serial(found - prefix_length, n, prefix_length)) return found;

        // Adjust the position.
        h_length = remaining - 1;
    }
}

SZ_PUBLIC sz_cptr_t sz_find_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return NULL;

    sz_find_t backends[] = {
        // For very short strings a lookup table for an optimized backend makes a lot of sense.
        (sz_find_t)sz_find_byte_serial,
        (sz_find_t)sz_find_2byte_serial,
        (sz_find_t)sz_find_3byte_serial,
        (sz_find_t)sz_find_4byte_serial,
        // For needle lengths up to 64, use the Bitap algorithm variation for exact search.
        (sz_find_t)sz_find_under8byte_serial,
        (sz_find_t)sz_find_under16byte_serial,
        (sz_find_t)sz_find_under32byte_serial,
        (sz_find_t)sz_find_under64byte_serial,
        // For longer needles, use Bitap for the first 64 bytes and then check the rest.
        (sz_find_t)sz_find_over64byte_serial,
    };

    return backends[
        // For very short strings a lookup table for an optimized backend makes a lot of sense.
        (n_length > 1) + (n_length > 2) + (n_length > 3) +
        // For needle lengths up to 64, use the Bitap algorithm variation for exact search.
        (n_length > 4) + (n_length > 8) + (n_length > 16) + (n_length > 32) +
        // For longer needles, use Bitap for the first 64 bytes and then check the rest.
        (n_length > 64)](h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_last_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return NULL;

    sz_find_t backends[] = {
        // For very short strings a lookup table for an optimized backend makes a lot of sense.
        (sz_find_t)sz_find_last_byte_serial,
        // For needle lengths up to 64, use the Bitap algorithm variation for reverse-order exact search.
        (sz_find_t)sz_find_last_under8byte_serial,
        (sz_find_t)sz_find_last_under16byte_serial,
        (sz_find_t)sz_find_last_under32byte_serial,
        (sz_find_t)sz_find_last_under64byte_serial,
        // For longer needles, use Bitap for the last 64 bytes and then check the rest.
        (sz_find_t)sz_find_last_over64byte_serial,
    };

    return backends[
        // For very short strings a lookup table for an optimized backend makes a lot of sense.
        0 +
        // For needle lengths up to 64, use the Bitap algorithm variation for reverse-order exact search.
        (n_length > 1) + (n_length > 8) + (n_length > 16) + (n_length > 32) +
        // For longer needles, use Bitap for the last 64 bytes and then check the rest.
        (n_length > 64)](h, h_length, n, n_length);
}

SZ_INTERNAL sz_size_t _sz_levenshtein_serial_upto256bytes( //
    sz_cptr_t a, sz_size_t a_length,                       //
    sz_cptr_t b, sz_size_t b_length,                       //
    sz_size_t bound, sz_memory_allocator_t const *alloc) {

    // When dealing with short strings, we won't need to allocate memory on heap,
    // as everythin would easily fit on the stack. Let's just make sure that
    // we use the amount proportional to the number of elements in the shorter string,
    // not the larger.
    if (b_length > a_length) return _sz_levenshtein_serial_upto256bytes(b, b_length, a, a_length, bound, alloc);

    // If the strings are under 256-bytes long, the distance can never exceed 256,
    // and will fit into `sz_u8_t` reducing our memory requirements.
    sz_u8_t levenshtein_matrx_rows[(b_length + 1) * 2];
    sz_u8_t *previous_distances = &levenshtein_matrx_rows[0];
    sz_u8_t *current_distances = &levenshtein_matrx_rows[b_length + 1];

    // The very first row of the matrix is equivalent to `std::iota` outputs.
    for (sz_size_t idx_b = 0; idx_b != (b_length + 1); ++idx_b) previous_distances[idx_b] = idx_b;

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        current_distances[0] = idx_a + 1;

        // Initialize min_distance with a value greater than bound.
        sz_size_t min_distance = bound;

        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            sz_u8_t cost_deletion = previous_distances[idx_b + 1] + 1;
            sz_u8_t cost_insertion = current_distances[idx_b] + 1;
            sz_u8_t cost_substitution = previous_distances[idx_b] + (a[idx_a] != b[idx_b]);
            current_distances[idx_b + 1] = sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);

            // Keep track of the minimum distance seen so far in this row.
            min_distance = sz_min_of_two(current_distances[idx_b + 1], min_distance);
        }

        // If the minimum distance in this row exceeded the bound, return early
        if (min_distance >= bound) return bound;

        // Swap previous_distances and current_distances pointers
        sz_u8_t *temp = previous_distances;
        previous_distances = current_distances;
        current_distances = temp;
    }

    return previous_distances[b_length] < bound ? previous_distances[b_length] : bound;
}

SZ_INTERNAL sz_size_t _sz_levenshtein_serial_over256bytes( //
    sz_cptr_t a, sz_size_t a_length,                       //
    sz_cptr_t b, sz_size_t b_length,                       //
    sz_size_t bound, sz_memory_allocator_t const *alloc) {

    // Let's make sure that we use the amount proportional to the number of elements in the shorter string,
    // not the larger.
    if (b_length > a_length) return _sz_levenshtein_serial_upto256bytes(b, b_length, a, a_length, bound, alloc);

    sz_size_t buffer_length = (b_length + 1) * 2;
    sz_ptr_t buffer = alloc->allocate(buffer_length, alloc->user_data);
    sz_size_t *previous_distances = (sz_size_t *)buffer;
    sz_size_t *current_distances = previous_distances + b_length + 1;

    for (sz_size_t idx_b = 0; idx_b != (b_length + 1); ++idx_b) previous_distances[idx_b] = idx_b;

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        current_distances[0] = idx_a + 1;

        // Initialize min_distance with a value greater than bound
        sz_size_t min_distance = bound;

        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            sz_size_t cost_deletion = previous_distances[idx_b + 1] + 1;
            sz_size_t cost_insertion = current_distances[idx_b] + 1;
            sz_size_t cost_substitution = previous_distances[idx_b] + (a[idx_a] != b[idx_b]);
            current_distances[idx_b + 1] = sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);

            // Keep track of the minimum distance seen so far in this row
            min_distance = sz_min_of_two(current_distances[idx_b + 1], min_distance);
        }

        // If the minimum distance in this row exceeded the bound, return early
        if (min_distance >= bound) {
            alloc->free(buffer, buffer_length, alloc->user_data);
            return bound;
        }

        // Swap previous_distances and current_distances pointers
        sz_size_t *temp = previous_distances;
        previous_distances = current_distances;
        current_distances = temp;
    }

    alloc->free(buffer, buffer_length, alloc->user_data);
    return previous_distances[b_length] < bound ? previous_distances[b_length] : bound;
}

SZ_PUBLIC sz_size_t sz_levenshtein_serial( //
    sz_cptr_t a, sz_size_t a_length,       //
    sz_cptr_t b, sz_size_t b_length,       //
    sz_size_t bound, sz_memory_allocator_t const *alloc) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one.
    if (a_length == 0) return b_length <= bound ? b_length : bound;
    if (b_length == 0) return a_length <= bound ? a_length : bound;

    // If the difference in length is beyond the `bound`, there is no need to check at all.
    if (a_length > b_length) {
        if (a_length - b_length > bound) return bound;
    }
    else {
        if (b_length - a_length > bound) return bound;
    }

    // Depending on the length, we may be able to use the optimized implementation.
    if (a_length < 256 && b_length < 256)
        return _sz_levenshtein_serial_upto256bytes(a, a_length, b, b_length, bound, alloc);
    else
        return _sz_levenshtein_serial_over256bytes(a, a_length, b, b_length, bound, alloc);
}

SZ_PUBLIC sz_ssize_t sz_alignment_score_serial(       //
    sz_cptr_t a, sz_size_t a_length,                  //
    sz_cptr_t b, sz_size_t b_length,                  //
    sz_error_cost_t gap, sz_error_cost_t const *subs, //
    sz_memory_allocator_t const *alloc) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one
    if (a_length == 0) return b_length;
    if (b_length == 0) return a_length;

    // Let's make sure that we use the amount proportional to the number of elements in the shorter string,
    // not the larger.
    if (b_length > a_length) return sz_alignment_score_serial(b, b_length, a, a_length, gap, subs, alloc);

    sz_size_t buffer_length = (b_length + 1) * 2;
    sz_ptr_t buffer = alloc->allocate(buffer_length, alloc->user_data);
    sz_ssize_t *previous_distances = (sz_ssize_t *)buffer;
    sz_ssize_t *current_distances = previous_distances + b_length + 1;

    for (sz_size_t idx_b = 0; idx_b != (b_length + 1); ++idx_b) previous_distances[idx_b] = idx_b;

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        current_distances[0] = idx_a + 1;

        // Initialize min_distance with a value greater than bound
        sz_error_cost_t const *a_subs = subs + a[idx_a] * 256ul;
        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            sz_ssize_t cost_deletion = previous_distances[idx_b + 1] + gap;
            sz_ssize_t cost_insertion = current_distances[idx_b] + gap;
            sz_ssize_t cost_substitution = previous_distances[idx_b] + a_subs[b[idx_b]];
            current_distances[idx_b + 1] = sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);
        }

        // Swap previous_distances and current_distances pointers
        sz_ssize_t *temp = previous_distances;
        previous_distances = current_distances;
        current_distances = temp;
    }

    alloc->free(buffer, buffer_length, alloc->user_data);
    return previous_distances[b_length];
}

SZ_INTERNAL sz_u8_t sz_u8_tolower(sz_u8_t c) {
    static sz_u8_t lowered[256] = {
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
    return lowered[c];
}

SZ_INTERNAL sz_u8_t sz_u8_toupper(sz_u8_t c) {
    static sz_u8_t upped[256] = {
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
    return upped[c];
}

SZ_PUBLIC void sz_tolower_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    for (sz_cptr_t end = text + length; text != end; ++text, ++result) {
        *result = sz_u8_tolower(*(sz_u8_t const *)text);
    }
}

SZ_PUBLIC void sz_toupper_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    for (sz_cptr_t end = text + length; text != end; ++text, ++result) {
        *result = sz_u8_toupper(*(sz_u8_t const *)text);
    }
}

SZ_PUBLIC void sz_toascii_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    for (sz_cptr_t end = text + length; text != end; ++text, ++result) { *result = *text & 0x7F; }
}

#pragma endregion

/*
 *  @brief  Serial implementation for strings sequence processing.
 */
#pragma region Serial Implementation for Sequences

/**
 *  @brief  Helper, that swaps two 64-bit integers representing the order of elements in the sequence.
 */
SZ_INTERNAL void _sz_swap_order(sz_u64_t *a, sz_u64_t *b) {
    sz_u64_t t = *a;
    *a = *b;
    *b = t;
}

SZ_PUBLIC sz_size_t sz_partition(sz_sequence_t *sequence, sz_sequence_predicate_t predicate) {

    sz_size_t matches = 0;
    while (matches != sequence->count && predicate(sequence, sequence->order[matches])) ++matches;

    for (sz_size_t i = matches + 1; i < sequence->count; ++i)
        if (predicate(sequence, sequence->order[i]))
            _sz_swap_order(sequence->order + i, sequence->order + matches), ++matches;

    return matches;
}

SZ_PUBLIC void sz_merge(sz_sequence_t *sequence, sz_size_t partition, sz_sequence_comparator_t less) {

    sz_size_t start_b = partition + 1;

    // If the direct merge is already sorted
    if (!less(sequence, sequence->order[start_b], sequence->order[partition])) return;

    sz_size_t start_a = 0;
    while (start_a <= partition && start_b <= sequence->count) {

        // If element 1 is in right place
        if (!less(sequence, sequence->order[start_b], sequence->order[start_a])) { start_a++; }
        else {
            sz_size_t value = sequence->order[start_b];
            sz_size_t index = start_b;

            // Shift all the elements between element 1
            // element 2, right by 1.
            while (index != start_a) { sequence->order[index] = sequence->order[index - 1], index--; }
            sequence->order[start_a] = value;

            // Update all the pointers
            start_a++;
            partition++;
            start_b++;
        }
    }
}

SZ_PUBLIC void sz_sort_insertion(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    sz_u64_t *keys = sequence->order;
    sz_size_t keys_count = sequence->count;
    for (sz_size_t i = 1; i < keys_count; i++) {
        sz_u64_t i_key = keys[i];
        sz_size_t j = i;
        for (; j > 0 && less(sequence, i_key, keys[j - 1]); --j) keys[j] = keys[j - 1];
        keys[j] = i_key;
    }
}

SZ_INTERNAL void _sz_sift_down(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_u64_t *order, sz_size_t start,
                               sz_size_t end) {
    sz_size_t root = start;
    while (2 * root + 1 <= end) {
        sz_size_t child = 2 * root + 1;
        if (child + 1 <= end && less(sequence, order[child], order[child + 1])) { child++; }
        if (!less(sequence, order[root], order[child])) { return; }
        _sz_swap_order(order + root, order + child);
        root = child;
    }
}

SZ_INTERNAL void _sz_heapify(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_u64_t *order, sz_size_t count) {
    sz_size_t start = (count - 2) / 2;
    while (1) {
        _sz_sift_down(sequence, less, order, start, count - 1);
        if (start == 0) return;
        start--;
    }
}

SZ_INTERNAL void _sz_heapsort(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_size_t first, sz_size_t last) {
    sz_u64_t *order = sequence->order;
    sz_size_t count = last - first;
    _sz_heapify(sequence, less, order + first, count);
    sz_size_t end = count - 1;
    while (end > 0) {
        _sz_swap_order(order + first, order + first + end);
        end--;
        _sz_sift_down(sequence, less, order + first, 0, end);
    }
}

SZ_INTERNAL void _sz_introsort(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_size_t first, sz_size_t last,
                               sz_size_t depth) {

    sz_size_t length = last - first;
    switch (length) {
    case 0:
    case 1: return;
    case 2:
        if (less(sequence, sequence->order[first + 1], sequence->order[first]))
            _sz_swap_order(&sequence->order[first], &sequence->order[first + 1]);
        return;
    case 3: {
        sz_u64_t a = sequence->order[first];
        sz_u64_t b = sequence->order[first + 1];
        sz_u64_t c = sequence->order[first + 2];
        if (less(sequence, b, a)) _sz_swap_order(&a, &b);
        if (less(sequence, c, b)) _sz_swap_order(&c, &b);
        if (less(sequence, b, a)) _sz_swap_order(&a, &b);
        sequence->order[first] = a;
        sequence->order[first + 1] = b;
        sequence->order[first + 2] = c;
        return;
    }
    }
    // Until a certain length, the quadratic-complexity insertion-sort is fine
    if (length <= 16) {
        sz_sequence_t sub_seq = *sequence;
        sub_seq.order += first;
        sub_seq.count = length;
        sz_sort_insertion(&sub_seq, less);
        return;
    }

    // Fallback to N-logN-complexity heap-sort
    if (depth == 0) {
        _sz_heapsort(sequence, less, first, last);
        return;
    }

    --depth;

    // Median-of-three logic to choose pivot
    sz_size_t median = first + length / 2;
    if (less(sequence, sequence->order[median], sequence->order[first]))
        _sz_swap_order(&sequence->order[first], &sequence->order[median]);
    if (less(sequence, sequence->order[last - 1], sequence->order[first]))
        _sz_swap_order(&sequence->order[first], &sequence->order[last - 1]);
    if (less(sequence, sequence->order[median], sequence->order[last - 1]))
        _sz_swap_order(&sequence->order[median], &sequence->order[last - 1]);

    // Partition using the median-of-three as the pivot
    sz_u64_t pivot = sequence->order[median];
    sz_size_t left = first;
    sz_size_t right = last - 1;
    while (1) {
        while (less(sequence, sequence->order[left], pivot)) left++;
        while (less(sequence, pivot, sequence->order[right])) right--;
        if (left >= right) break;
        _sz_swap_order(&sequence->order[left], &sequence->order[right]);
        left++;
        right--;
    }

    // Recursively sort the partitions
    _sz_introsort(sequence, less, first, left, depth);
    _sz_introsort(sequence, less, right + 1, last, depth);
}

SZ_PUBLIC void sz_sort_introsort(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    sz_size_t depth_limit = 2 * sz_size_log2i(sequence->count);
    _sz_introsort(sequence, less, 0, sequence->count, depth_limit);
}

SZ_INTERNAL void _sz_sort_recursion( //
    sz_sequence_t *sequence, sz_size_t bit_idx, sz_size_t bit_max, sz_sequence_comparator_t comparator,
    sz_size_t partial_order_length) {

    if (!sequence->count) return;

    // Partition a range of integers according to a specific bit value
    sz_size_t split = 0;
    {
        sz_u64_t mask = (1ull << 63) >> bit_idx;
        while (split != sequence->count && !(sequence->order[split] & mask)) ++split;
        for (sz_size_t i = split + 1; i < sequence->count; ++i)
            if (!(sequence->order[i] & mask)) _sz_swap_order(sequence->order + i, sequence->order + split), ++split;
    }

    // Go down recursively
    if (bit_idx < bit_max) {
        sz_sequence_t a = *sequence;
        a.count = split;
        _sz_sort_recursion(&a, bit_idx + 1, bit_max, comparator, partial_order_length);

        sz_sequence_t b = *sequence;
        b.order += split;
        b.count -= split;
        _sz_sort_recursion(&b, bit_idx + 1, bit_max, comparator, partial_order_length);
    }
    // Reached the end of recursion
    else {
        // Discard the prefixes
        sz_u32_t *order_half_words = (sz_u32_t *)sequence->order;
        for (sz_size_t i = 0; i != sequence->count; ++i) { order_half_words[i * 2 + 1] = 0; }

        sz_sequence_t a = *sequence;
        a.count = split;
        sz_sort_introsort(&a, comparator);

        sz_sequence_t b = *sequence;
        b.order += split;
        b.count -= split;
        sz_sort_introsort(&b, comparator);
    }
}

SZ_INTERNAL sz_bool_t _sz_sort_is_less(sz_sequence_t *sequence, sz_size_t i_key, sz_size_t j_key) {
    sz_cptr_t i_str = sequence->get_start(sequence, i_key);
    sz_cptr_t j_str = sequence->get_start(sequence, j_key);
    sz_size_t i_len = sequence->get_length(sequence, i_key);
    sz_size_t j_len = sequence->get_length(sequence, j_key);
    return (sz_bool_t)(sz_order_serial(i_str, i_len, j_str, j_len) == sz_less_k);
}

SZ_PUBLIC void sz_sort_partial(sz_sequence_t *sequence, sz_size_t partial_order_length) {

    // Export up to 4 bytes into the `sequence` bits themselves
    for (sz_size_t i = 0; i != sequence->count; ++i) {
        sz_cptr_t begin = sequence->get_start(sequence, sequence->order[i]);
        sz_size_t length = sequence->get_length(sequence, sequence->order[i]);
        length = length > 4ull ? 4ull : length;
        sz_ptr_t prefix = (sz_ptr_t)&sequence->order[i];
        for (sz_size_t j = 0; j != length; ++j) prefix[7 - j] = begin[j];
    }

    // Perform optionally-parallel radix sort on them
    _sz_sort_recursion(sequence, 0, 32, (sz_sequence_comparator_t)_sz_sort_is_less, partial_order_length);
}

SZ_PUBLIC void sz_sort(sz_sequence_t *sequence) { sz_sort_partial(sequence, sequence->count); }

#pragma endregion

/*
 *  @brief  AVX-512 implementation of the string search algorithms.
 *
 *  Different subsets of AVX-512 were introduced in different years:
 *  * 2017 SkyLake: F, CD, ER, PF, VL, DQ, BW
 *  * 2018 CannonLake: IFMA, VBMI
 *  * 2019 IceLake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES
 *  * 2020 TigerLake: VP2INTERSECT
 */
#pragma region AVX-512 Implementation

#if SZ_USE_X86_AVX512
#include <x86intrin.h>

/**
 *  @brief  Helper structure to simpify work with 64-bit words.
 */
typedef union sz_u512_parts_t {
    __m512i zmm;
    sz_u64_t u64s[8];
    sz_u32_t u32s[16];
    sz_u16_t u16s[32];
    sz_u8_t u8s[64];
} sz_u512_parts_t;

SZ_INTERNAL __mmask64 sz_u64_clamp_mask_until(sz_size_t n) {
    // The simplest approach to compute this if we know that `n` is blow or equal 64:
    //      return (1ull << n) - 1;
    // A slighly more complex approach, if we don't know that `n` is under 64:
    return _bzhi_u64(0xFFFFFFFFFFFFFFFF, n < 64 ? n : 64);
}

SZ_INTERNAL __mmask64 sz_u64_mask_until(sz_size_t n) {
    // The simplest approach to compute this if we know that `n` is blow or equal 64:
    //      return (1ull << n) - 1;
    // A slighly more complex approach, if we don't know that `n` is under 64:
    return _bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
}

/**
 *  @brief  Variation of AVX-512 relative order check for different length strings.
 */
SZ_PUBLIC sz_ordering_t sz_order_avx512(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
    __m512i a_vec, b_vec;

sz_order_avx512_cycle:
    // In most common scenarios at least one of the strings is under 64 bytes.
    if ((a_length < 64) + (b_length < 64)) {
        __mmask64 a_mask = sz_u64_clamp_mask_until(a_length);
        __mmask64 b_mask = sz_u64_clamp_mask_until(b_length);
        a_vec = _mm512_maskz_loadu_epi8(a_mask, a);
        b_vec = _mm512_maskz_loadu_epi8(b_mask, b);
        // The AVX-512 `_mm512_mask_cmpneq_epi8_mask` intrinsics are generally handy in such environments.
        // They, however, have latency 3 on most modern CPUs. Using AVX2: `_mm256_cmpeq_epi8` would have
        // been cheaper, if we didn't have to apply `_mm256_movemask_epi8` afterwards.
        __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec, b_vec);
        if (mask_not_equal != 0) {
            int first_diff = _tzcnt_u64(mask_not_equal);
            char a_char = a[first_diff];
            char b_char = b[first_diff];
            return ordering_lookup[a_char < b_char];
        }
        else
            // From logic perspective, the hardest cases are "abc\0" and "abc".
            // The result must be `sz_greater_k`, as the latter is shorter.
            return a_length != b_length ? ordering_lookup[a_length < b_length] : sz_equal_k;
    }
    else {
        a_vec = _mm512_loadu_epi8(a);
        b_vec = _mm512_loadu_epi8(b);
        __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec, b_vec);
        if (mask_not_equal != 0) {
            int first_diff = _tzcnt_u64(mask_not_equal);
            char a_char = a[first_diff];
            char b_char = b[first_diff];
            return ordering_lookup[a_char < b_char];
        }
        a += 64, b += 64, a_length -= 64, b_length -= 64;
        if ((a_length > 0) + (b_length > 0)) goto sz_order_avx512_cycle;
        return a_length != b_length ? ordering_lookup[a_length < b_length] : sz_equal_k;
    }
}

/**
 *  @brief  Variation of AVX-512 equality check between equivalent length strings.
 */
SZ_PUBLIC sz_bool_t sz_equal_avx512(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {

    // In the absolute majority of the cases, the first mismatch is
    __m512i a_vec, b_vec;
    __mmask64 mask;

sz_equal_avx512_cycle:
    if (length < 64) {
        mask = sz_u64_mask_until(length);
        a_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_vec = _mm512_maskz_loadu_epi8(mask, b);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpneq_epi8_mask(mask, a_vec, b_vec);
        return (sz_bool_t)(mask == 0);
    }
    else {
        a_vec = _mm512_loadu_epi8(a);
        b_vec = _mm512_loadu_epi8(b);
        mask = _mm512_cmpneq_epi8_mask(a_vec, b_vec);
        if (mask != 0) return sz_false_k;
        a += 64, b += 64, length -= 64;
        if (length) goto sz_equal_avx512_cycle;
        return sz_true_k;
    }
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns up to 1 bytes included.
 */
SZ_PUBLIC sz_cptr_t sz_find_byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __mmask64 mask;
    sz_u512_parts_t h_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi8(n[0]);

sz_find_byte_avx512_cycle:
    if (h_length < 64) {
        mask = sz_u64_mask_until(h_length);
        h_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec.zmm, n_vec.zmm);
        if (mask) return h + sz_u64_ctz(mask);
    }
    else {
        h_vec.zmm = _mm512_loadu_epi8(h);
        mask = _mm512_cmpeq_epi8_mask(h_vec.zmm, n_vec.zmm);
        if (mask) return h + sz_u64_ctz(mask);
        h += 64, h_length -= 64;
        if (h_length) goto sz_find_byte_avx512_cycle;
    }
    return NULL;
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns up to 2 bytes included.
 */
SZ_INTERNAL sz_cptr_t sz_find_2byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];

    // A simpler approach would ahve been to use two separate registers for
    // different characters of the needle, but that would use more registers.
    __m512i h0_vec, h1_vec, n_vec = _mm512_set1_epi16(n_parts.u16s[0]);
    __mmask64 mask;
    __mmask32 matches0, matches1;

sz_find_2byte_avx512_cycle:
    if (h_length < 2) { return NULL; }
    else if (h_length < 66) {
        mask = sz_u64_mask_until(h_length);
        h0_vec = _mm512_maskz_loadu_epi8(mask, h);
        h1_vec = _mm512_maskz_loadu_epi8(mask, h + 1);
        matches0 = _mm512_mask_cmpeq_epi16_mask(mask, h0_vec, n_vec);
        matches1 = _mm512_mask_cmpeq_epi16_mask(mask, h1_vec, n_vec);
        if (matches0 | matches1)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x5555555555555555ull) | //
                                  _pdep_u64(matches1, 0xAAAAAAAAAAAAAAAAull));
        return NULL;
    }
    else {
        h0_vec = _mm512_loadu_epi8(h);
        h1_vec = _mm512_loadu_epi8(h + 1);
        matches0 = _mm512_cmpeq_epi16_mask(h0_vec, n_vec);
        matches1 = _mm512_cmpeq_epi16_mask(h1_vec, n_vec);
        // https://lemire.me/blog/2018/01/08/how-fast-can-you-bit-interleave-32-bit-integers/
        if (matches0 | matches1)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x5555555555555555ull) | //
                                  _pdep_u64(matches1, 0xAAAAAAAAAAAAAAAAull));
        h += 64, h_length -= 64;
        goto sz_find_2byte_avx512_cycle;
    }
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns up to 4 bytes included.
 */
SZ_INTERNAL sz_cptr_t sz_find_4byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];
    n_parts.u8s[2] = n[2];
    n_parts.u8s[3] = n[3];

    __m512i h0_vec, h1_vec, h2_vec, h3_vec, n_vec = _mm512_set1_epi32(n_parts.u32s[0]);
    __mmask64 mask;
    __mmask16 matches0, matches1, matches2, matches3;

sz_find_4byte_avx512_cycle:
    if (h_length < 4) { return NULL; }
    else if (h_length < 68) {
        mask = sz_u64_mask_until(h_length);
        h0_vec = _mm512_maskz_loadu_epi8(mask, h);
        h1_vec = _mm512_maskz_loadu_epi8(mask, h + 1);
        h2_vec = _mm512_maskz_loadu_epi8(mask, h + 2);
        h3_vec = _mm512_maskz_loadu_epi8(mask, h + 3);
        matches0 = _mm512_mask_cmpeq_epi32_mask(mask, h0_vec, n_vec);
        matches1 = _mm512_mask_cmpeq_epi32_mask(mask, h1_vec, n_vec);
        matches2 = _mm512_mask_cmpeq_epi32_mask(mask, h2_vec, n_vec);
        matches3 = _mm512_mask_cmpeq_epi32_mask(mask, h3_vec, n_vec);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x1111111111111111ull) | //
                                  _pdep_u64(matches1, 0x2222222222222222ull) | //
                                  _pdep_u64(matches2, 0x4444444444444444ull) | //
                                  _pdep_u64(matches3, 0x8888888888888888ull));
        return NULL;
    }
    else {
        h0_vec = _mm512_loadu_epi8(h);
        h1_vec = _mm512_loadu_epi8(h + 1);
        h2_vec = _mm512_loadu_epi8(h + 2);
        h3_vec = _mm512_loadu_epi8(h + 3);
        matches0 = _mm512_cmpeq_epi32_mask(h0_vec, n_vec);
        matches1 = _mm512_cmpeq_epi32_mask(h1_vec, n_vec);
        matches2 = _mm512_cmpeq_epi32_mask(h2_vec, n_vec);
        matches3 = _mm512_cmpeq_epi32_mask(h3_vec, n_vec);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x1111111111111111) | //
                                  _pdep_u64(matches1, 0x2222222222222222) | //
                                  _pdep_u64(matches2, 0x4444444444444444) | //
                                  _pdep_u64(matches3, 0x8888888888888888));
        h += 64, h_length -= 64;
        goto sz_find_4byte_avx512_cycle;
    }
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns up to 3 bytes included.
 */
SZ_INTERNAL sz_cptr_t sz_find_3byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    sz_u64_parts_t n_parts;
    n_parts.u64 = 0;
    n_parts.u8s[0] = n[0];
    n_parts.u8s[1] = n[1];
    n_parts.u8s[2] = n[2];

    // A simpler approach would ahve been to use two separate registers for
    // different characters of the needle, but that would use more registers.
    __m512i h0_vec, h1_vec, h2_vec, h3_vec, n_vec = _mm512_set1_epi32(n_parts.u32s[0]);
    __mmask64 mask;
    __mmask16 matches0, matches1, matches2, matches3;

sz_find_3byte_avx512_cycle:
    if (h_length < 3) { return NULL; }
    else if (h_length < 67) {
        mask = sz_u64_mask_until(h_length);
        // This implementation is more complex than the `sz_find_4byte_avx512`,
        // as we are going to match only 3 bytes within each 4-byte word.
        h0_vec = _mm512_maskz_loadu_epi8(mask & 0x7777777777777777, h);
        h1_vec = _mm512_maskz_loadu_epi8(mask & 0x7777777777777777, h + 1);
        h2_vec = _mm512_maskz_loadu_epi8(mask & 0x7777777777777777, h + 2);
        h3_vec = _mm512_maskz_loadu_epi8(mask & 0x7777777777777777, h + 3);
        matches0 = _mm512_mask_cmpeq_epi32_mask(mask, h0_vec, n_vec);
        matches1 = _mm512_mask_cmpeq_epi32_mask(mask, h1_vec, n_vec);
        matches2 = _mm512_mask_cmpeq_epi32_mask(mask, h2_vec, n_vec);
        matches3 = _mm512_mask_cmpeq_epi32_mask(mask, h3_vec, n_vec);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x1111111111111111) | //
                                  _pdep_u64(matches1, 0x2222222222222222) | //
                                  _pdep_u64(matches2, 0x4444444444444444) | //
                                  _pdep_u64(matches3, 0x8888888888888888));
        return NULL;
    }
    else {
        h0_vec = _mm512_maskz_loadu_epi8(0x7777777777777777, h);
        h1_vec = _mm512_maskz_loadu_epi8(0x7777777777777777, h + 1);
        h2_vec = _mm512_maskz_loadu_epi8(0x7777777777777777, h + 2);
        h3_vec = _mm512_maskz_loadu_epi8(0x7777777777777777, h + 3);
        matches0 = _mm512_cmpeq_epi32_mask(h0_vec, n_vec);
        matches1 = _mm512_cmpeq_epi32_mask(h1_vec, n_vec);
        matches2 = _mm512_cmpeq_epi32_mask(h2_vec, n_vec);
        matches3 = _mm512_cmpeq_epi32_mask(h3_vec, n_vec);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x1111111111111111) | //
                                  _pdep_u64(matches1, 0x2222222222222222) | //
                                  _pdep_u64(matches2, 0x4444444444444444) | //
                                  _pdep_u64(matches3, 0x8888888888888888));
        h += 64, h_length -= 64;
        goto sz_find_3byte_avx512_cycle;
    }
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns up to 66 bytes included.
 */
SZ_INTERNAL sz_cptr_t sz_find_under66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    __mmask64 mask, n_length_body_mask = sz_u64_mask_until(n_length - 2);
    __mmask64 matches;
    sz_u512_parts_t h_first_vec, h_last_vec, h_body_vec, n_first_vec, n_last_vec, n_body_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);
    n_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, n + 1);

sz_find_under66byte_avx512_cycle:
    if (h_length < n_length) { return NULL; }
    else if (h_length < n_length + 64) {
        mask = sz_u64_mask_until(h_length);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            h_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, h + potential_offset + 1);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_under66byte_avx512_cycle;
        }
        else
            return NULL;
    }
    else {
        h_first_vec.zmm = _mm512_loadu_epi8(h);
        h_last_vec.zmm = _mm512_loadu_epi8(h + n_length - 1);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            h_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, h + potential_offset + 1);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_under66byte_avx512_cycle;
        }
        else {
            h += 64, h_length -= 64;
            goto sz_find_under66byte_avx512_cycle;
        }
    }
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns longer than 66 bytes.
 */
SZ_INTERNAL sz_cptr_t sz_find_over66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    __mmask64 mask;
    __mmask64 matches;
    sz_u512_parts_t h_first_vec, h_last_vec, n_first_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);

sz_find_over66byte_avx512_cycle:
    if (h_length < n_length) { return NULL; }
    else if (h_length < n_length + 64) {
        mask = sz_u64_mask_until(h_length);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (sz_equal_avx512(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_over66byte_avx512_cycle;
        }
        else
            return NULL;
    }
    else {
        h_first_vec.zmm = _mm512_loadu_epi8(h);
        h_last_vec.zmm = _mm512_loadu_epi8(h + n_length - 1);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (sz_equal_avx512(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_over66byte_avx512_cycle;
        }
        else {
            h += 64, h_length -= 64;
            goto sz_find_over66byte_avx512_cycle;
        }
    }
}

SZ_PUBLIC sz_cptr_t sz_find_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return NULL;

    sz_find_t backends[] = {
        // For very short strings a lookup table for an optimized backend makes a lot of sense.
        (sz_find_t)sz_find_byte_avx512,
        (sz_find_t)sz_find_2byte_avx512,
        (sz_find_t)sz_find_3byte_avx512,
        (sz_find_t)sz_find_4byte_avx512,
        // For longer needles we use a Two-Way heurstic with a follow-up check in-between.
        (sz_find_t)sz_find_under66byte_avx512,
        (sz_find_t)sz_find_over66byte_avx512,
    };

    return backends[
        // For very short strings a lookup table for an optimized backend makes a lot of sense.
        (n_length > 1) + (n_length > 2) + (n_length > 3) +
        // For longer needles we use a Two-Way heurstic with a follow-up check in-between.
        (n_length > 4) + (n_length > 66)](h, h_length, n, n_length);
}

/**
 *  @brief  Variation of AVX-512 exact reverse-order search for patterns up to 1 bytes included.
 */
SZ_PUBLIC sz_cptr_t sz_find_last_byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __mmask64 mask;
    sz_u512_parts_t h_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi8(n[0]);

sz_find_last_byte_avx512_cycle:
    if (h_length < 64) {
        mask = sz_u64_mask_until(h_length);
        h_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec.zmm, n_vec.zmm);
        int potential_offset = sz_u64_clz(mask);
        if (mask) return h + 64 - sz_u64_clz(mask) - 1;
    }
    else {
        h_vec.zmm = _mm512_loadu_epi8(h + h_length - 64);
        mask = _mm512_cmpeq_epi8_mask(h_vec.zmm, n_vec.zmm);
        int potential_offset = sz_u64_clz(mask);
        if (mask) return h + h_length - 1 - sz_u64_clz(mask);
        h_length -= 64;
        if (h_length) goto sz_find_last_byte_avx512_cycle;
    }
    return NULL;
}

/**
 *  @brief  Variation of AVX-512 reverse-order exact search for patterns up to 66 bytes included.
 */
SZ_INTERNAL sz_cptr_t sz_find_last_under66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                      sz_size_t n_length) {

    __mmask64 mask, n_length_body_mask = sz_u64_mask_until(n_length - 2);
    __mmask64 matches;
    sz_u512_parts_t h_first_vec, h_last_vec, h_body_vec, n_first_vec, n_last_vec, n_body_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);
    n_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, n + 1);

sz_find_under66byte_avx512_cycle:
    if (h_length < n_length) { return NULL; }
    else if (h_length < n_length + 64) {
        mask = sz_u64_mask_until(h_length);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_clz(matches);
            h_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, h + 64 - potential_offset);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm)) return h + 64 - potential_offset - 1;

            h_length = 64 - potential_offset - 1;
            goto sz_find_under66byte_avx512_cycle;
        }
        else
            return NULL;
    }
    else {
        h_first_vec.zmm = _mm512_loadu_epi8(h + h_length - n_length - 64 + 1);
        h_last_vec.zmm = _mm512_loadu_epi8(h + h_length - 64);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_clz(matches);
            h_body_vec.zmm =
                _mm512_maskz_loadu_epi8(n_length_body_mask, h + h_length - n_length - potential_offset + 1);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm))
                return h + h_length - n_length - potential_offset;

            h_length -= potential_offset + 1;
            goto sz_find_under66byte_avx512_cycle;
        }
        else {
            h_length -= 64;
            goto sz_find_under66byte_avx512_cycle;
        }
    }
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns longer than 66 bytes.
 */
SZ_INTERNAL sz_cptr_t sz_find_last_over66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    __mmask64 mask;
    __mmask64 matches;
    sz_u512_parts_t h_first_vec, h_last_vec, n_first_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);

sz_find_over66byte_avx512_cycle:
    if (h_length < n_length) { return NULL; }
    else if (h_length < n_length + 64) {
        mask = sz_u64_mask_until(h_length);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (sz_equal_avx512(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_over66byte_avx512_cycle;
        }
        else
            return NULL;
    }
    else {
        h_first_vec.zmm = _mm512_loadu_epi8(h);
        h_last_vec.zmm = _mm512_loadu_epi8(h + n_length - 1);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (sz_equal_avx512(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;

            h += potential_offset + 1, h_length -= potential_offset + 1;
            goto sz_find_over66byte_avx512_cycle;
        }
        else {
            h += 64, h_length -= 64;
            goto sz_find_over66byte_avx512_cycle;
        }
    }
}

SZ_PUBLIC sz_cptr_t sz_find_last_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return NULL;

    sz_find_t backends[] = {
        // For very short strings a lookup table for an optimized backend makes a lot of sense.
        (sz_find_t)sz_find_last_byte_avx512,
        // For longer needles we use a Two-Way heurstic with a follow-up check in-between.
        (sz_find_t)sz_find_last_under66byte_avx512,
        (sz_find_t)sz_find_last_over66byte_avx512,
    };

    return backends[
        // For very short strings a lookup table for an optimized backend makes a lot of sense.
        0 +
        // For longer needles we use a Two-Way heurstic with a follow-up check in-between.
        (n_length > 1) + (n_length > 66)](h, h_length, n, n_length);
}

#endif

#pragma endregion

/*
 *  @brief  Pick the right implementation for the string search algorithms.
 */
#pragma region Compile-Time Dispatching

#include <stringzilla/stringzilla.h>

SZ_PUBLIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length) {
#if defined(__NEON__)
    return sz_hash_neon(text, length);
#elif defined(__AVX512__)
    return sz_hash_avx512(text, length);
#else
    return sz_hash_serial(text, length);
#endif
}

SZ_PUBLIC sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
#if defined(__AVX512__)
    return sz_order_avx512(a, a_length, b, b_length);
#else
    return sz_order_serial(a, a_length, b, b_length);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#if defined(__AVX512__)
    return sz_find_byte_avx512(haystack, h_length, needle);
#else
    return sz_find_byte_serial(haystack, h_length, needle);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
#if defined(__AVX512__)
    return sz_find_avx512(haystack, h_length, needle, n_length);
#elif defined(__NEON__)
    return sz_find_neon(haystack, h_length, needle, n_length);
#else
    return sz_find_serial(haystack, h_length, needle, n_length);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_from_set(sz_cptr_t text, sz_size_t length, sz_u8_set_t *set) {
    return sz_find_from_set_serial(text, length, set);
}

SZ_PUBLIC sz_cptr_t sz_find_last_from_set(sz_cptr_t text, sz_size_t length, sz_u8_set_t *set) {
    return sz_find_last_from_set_serial(text, length, set);
}

SZ_PUBLIC void sz_tolower(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#if defined(__AVX512__)
    sz_tolower_avx512(text, length, result);
#else
    sz_tolower_serial(text, length, result);
#endif
}

SZ_PUBLIC void sz_toupper(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#if defined(__AVX512__)
    sz_toupper_avx512(text, length, result);
#else
    sz_toupper_serial(text, length, result);
#endif
}

SZ_PUBLIC void sz_toascii(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
#if defined(__AVX512__)
    sz_toascii_avx512(text, length, result);
#else
    sz_toascii_serial(text, length, result);
#endif
}

SZ_PUBLIC sz_size_t sz_levenshtein(  //
    sz_cptr_t a, sz_size_t a_length, //
    sz_cptr_t b, sz_size_t b_length, //
    sz_size_t bound, sz_memory_allocator_t const *alloc) {
#if defined(__AVX512__)
    return sz_levenshtein_avx512(a, a_length, b, b_length, bound, alloc);
#else
    return sz_levenshtein_serial(a, a_length, b, b_length, bound, alloc);
#endif
}

SZ_PUBLIC sz_ssize_t sz_alignment_score(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                        sz_error_cost_t gap, sz_error_cost_t const *subs,
                                        sz_memory_allocator_t const *alloc) {

#if defined(__AVX512__)
    return sz_alignment_score_avx512(a, a_length, b, b_length, gap, subs, alloc);
#else
    return sz_alignment_score_serial(a, a_length, b, b_length, gap, subs, alloc);
#endif
}

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_H_
