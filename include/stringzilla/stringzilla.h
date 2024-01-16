/**
 *  @brief  StringZilla is a collection of simple string algorithms, designed to be used in Big Data applications.
 *          It may be slower than LibC, but has a broader & cleaner interface, and a very short implementation
 *          targeting modern x86 CPUs with AVX-512 and Arm NEON and older CPUs with SWAR and auto-vectorization.
 *
 *  @section    Operations potentially not worth optimizing in StringZilla
 *
 *  Some operations, like equality comparisons and relative order checking, almost always fail on some of the very
 *  first bytes in either string. This makes vectorization almost useless, unless huge strings are considered.
 *  Examples would be - computing the checksum of a long string, or checking 2 large binary strings for exact equality.
 *
 *  @section    Uncommon operations covered by StringZilla
 *
 *  Every in-order search/matching operations has a reverse order counterpart, a rare feature in string libraries.
 *  That way `sz_find` and `sz_find_last` are similar to `strstr` and `strrstr` in LibC, but `sz_find_byte` and
 *  `sz_find_last_byte` are equivalent to `memchr` and `memrchr`. The same goes for `sz_find_from_set` and
 *  `sz_find_last_from_set`, which are equivalent to `strspn` and `strcspn` in LibC.
 *
 *  Edit distance computations can be parameterized with the substitution matrix and gap (insertion & deletion)
 *  penalties. This allows for more flexible usecases, like scoring fuzzy string matches, and bioinformatics.

 *  @section    Exact substring search algorithms
 *
 *  Uses different algorithms for different needle lengths and backends:
 *
 *  > Naive exact matching for 1-, 2-, 3-, and 4-character-long needles using SIMD.
 *  > Bitap "Shift Or" Baeza-Yates-Gonnet (BYG) algorithm for mid-length needles on a serial backend.
 *  > Boyer-Moore-Horspool (BMH) algorithm with Raita heuristic variation for longer needles.
 *  > Apostolico-Giancarlo algorithm for longer needles (TODO), if needle preprocessing time isn't an issue.
 *
 *  Substring search algorithms are generally divided into: comparison-based, automaton-based, and bit-parallel.
 *  Different families are effective for different alphabet sizes and needle lengths. The more operations are
 *  needed per-character - the more effective SIMD would be. The longer the needle - the more effective the
 *  skip-tables are.
 *
 *  On very short needles, especially 1-4 characters long, brute force with SIMD is the fastest solution.
 *  On mid-length needles, bit-parallel algorithms are very effective, as the character masks fit into 32-bit
 *  or 64-bit words. Either way, if the needle is under 64-bytes long, on haystack traversal we will still fetch
 *  every CPU cache line. So the only way to improve performance is to reduce the number of comparisons.
 *
 *  Going beyond that, to long needles, Boyer-Moore (BM) and its variants are often the best choice. It has two tables:
 *  the good-suffix shift and the bad-character shift. Common choice is to use the simplified BMH algorithm,
 *  which only uses the bad-character shift table, reducing the pre-processing time. In the C++ Standards Library,
 *  the `std::string::find` function uses the BMH algorithm with Raita's heuristic. We do the same for longer needles.
 *
 *  All those, still, have O(hn) worst case complexity, and struggle with repetitive needle patterns.
 *  To guarantee O(h) worst case time complexity, the Apostolico-Giancarlo (AG) algorithm adds an additional skip-table.
 *  Preprocessing phase is O(n+sigma) in time and space. On traversal, performs from (h/n) to (3h/2) comparisons.
 *  We should consider implementing it if we can:
 *      - accelerate the preprocessing phase of the needle.
 *      - simplify the control-flow of the main loop.
 *      - replace the array of shift values with a circular buffer.
 *
 *  Reading materials:
 *      - Exact String Matching Algorithms in Java: https://www-igm.univ-mlv.fr/~lecroq/string
 *      - SIMD-friendly algorithms for substring searching: http://0x80.pl/articles/simd-strfind.html
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
 *      - void    *memrchr(const void *, int, size_t); -> sz_find_last_byte
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
#define SZ_PUBLIC inline static
#elif __GNUC__ >= 4
#define SZ_PUBLIC inline static
#else
#define SZ_PUBLIC inline static
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
 *  @brief  A misaligned load can be - trying to fetch eight consecutive bytes from an address
 *          that is not divisible by eight.
 *
 *  Most platforms support it, but there is no industry standard way to check for those.
 *  This value will mostly affect the performance of the serial (SWAR) backend.
 */
#ifndef SZ_USE_MISALIGNED_LOADS
#define SZ_USE_MISALIGNED_LOADS (1) // true or false
#endif

/**
 *  @brief  Cache-line width, that will affect the execution of some algorithms,
 *          like equality checks and relative order computing.
 */
#ifndef SZ_CACHE_LINE_WIDTH
#define SZ_CACHE_LINE_WIDTH (64) // bytes
#endif

/**
 *  @brief  Threshold for switching to SWAR (8-bytes at a time) backend over serial byte-level for-loops.
 *          On very short strings, under 16 bytes long, at most a single word will be processed with SWAR.
 *          Assuming potentially misaligned loads, SWAR makes sense only after ~24 bytes.
 */
#ifndef SZ_SWAR_THRESHOLD
#define SZ_SWAR_THRESHOLD (24) // bytes
#endif

/**
 *  @brief  Analogous to `size_t` and `std::size_t`, unsigned integer, identical to pointer size.
 *          64-bit on most platforms where pointers are 64-bit.
 *          32-bit on platforms where pointers are 32-bit.
 */
#if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_WIN64)
#define SZ_DETECT_64_BIT (1)
#define SZ_SIZE_MAX (0xFFFFFFFFFFFFFFFFull)
#else
#define SZ_DETECT_64_BIT (0)
#define SZ_SIZE_MAX (0xFFFFFFFFu)
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

#ifndef SZ_USE_ARM_NEON
#ifdef __ARM_NEON
#define SZ_USE_ARM_NEON 1
#else
#define SZ_USE_ARM_NEON 0
#endif
#endif

#ifndef SZ_USE_ARM_SVE
#ifdef __ARM_FEATURE_SVE
#define SZ_USE_ARM_SVE 1
#else
#define SZ_USE_ARM_SVE 0
#endif
#endif

/*
 *  Debugging and testing.
 */
#ifndef SZ_DEBUG
#ifndef NDEBUG
#define SZ_DEBUG 1
#else
#define SZ_DEBUG 0
#endif
#endif

#if SZ_DEBUG
#define sz_assert(condition)                                                                                \
    do {                                                                                                    \
        if (!(condition)) {                                                                                 \
            fprintf(stderr, "Assertion failed: %s, in file %s, line %d\n", #condition, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
    } while (0)
#else
#define sz_assert(condition) ((void)0)
#endif

/**
 *  @brief  Compile-time assert macro similar to `static_assert` in C++.
 */
#define sz_static_assert(condition, name)                \
    typedef struct {                                     \
        int static_assert_##name : (condition) ? 1 : -1; \
    } sz_static_assert_##name##_t

#define sz_unused(x) ((void)(x))

#define sz_bitcast(type, value) (*((type *)&(value)))

#if __has_attribute(__fallthrough__)
#define SZ_FALLTHROUGH __attribute__((__fallthrough__))
#else
#define SZ_FALLTHROUGH \
    do {               \
    } while (0) /* fallthrough */
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_DETECT_64_BIT
typedef unsigned long long sz_size_t;
typedef long long sz_ssize_t;
#else
typedef unsigned sz_size_t;
typedef unsigned sz_ssize_t;
#endif

sz_static_assert(sizeof(sz_size_t) == sizeof(void *), sz_size_t_must_be_pointer_size);
sz_static_assert(sizeof(sz_ssize_t) == sizeof(void *), sz_ssize_t_must_be_pointer_size);

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

/**
 *  @brief  Bit-set structure for 256 ASCII characters. Useful for filtering and search.
 */
typedef union sz_u8_set_t {
    sz_u64_t _u64s[4];
    sz_u32_t _u32s[8];
    sz_u16_t _u16s[16];
    sz_u8_t _u8s[32];
} sz_u8_set_t;

SZ_PUBLIC void sz_u8_set_init(sz_u8_set_t *f) { f->_u64s[0] = f->_u64s[1] = f->_u64s[2] = f->_u64s[3] = 0; }
SZ_PUBLIC void sz_u8_set_add(sz_u8_set_t *f, sz_u8_t c) { f->_u64s[c >> 6] |= (1ull << (c & 63u)); }
SZ_PUBLIC sz_bool_t sz_u8_set_contains(sz_u8_set_t const *f, sz_u8_t c) {
    // Checking the bit can be done in different ways:
    // - (f->_u64s[c >> 6] & (1ull << (c & 63u))) != 0
    // - (f->_u32s[c >> 5] & (1u << (c & 31u))) != 0
    // - (f->_u16s[c >> 4] & (1u << (c & 15u))) != 0
    // - (f->_u8s[c >> 3] & (1u << (c & 7u))) != 0
    return (sz_bool_t)((f->_u64s[c >> 6] & (1ull << (c & 63u))) != 0);
}
SZ_PUBLIC void sz_u8_set_invert(sz_u8_set_t *f) {
    f->_u64s[0] ^= 0xFFFFFFFFFFFFFFFFull, f->_u64s[1] ^= 0xFFFFFFFFFFFFFFFFull, //
        f->_u64s[2] ^= 0xFFFFFFFFFFFFFFFFull, f->_u64s[3] ^= 0xFFFFFFFFFFFFFFFFull;
}

typedef void *(*sz_memory_allocate_t)(sz_size_t, void *);
typedef void (*sz_memory_free_t)(void *, sz_size_t, void *);
typedef sz_u64_t (*sz_random_generator_t)(void *);

/**
 *  @brief  Some complex pattern matching algorithms may require memory allocations.
 */
typedef struct sz_memory_allocator_t {
    sz_memory_allocate_t allocate;
    sz_memory_free_t free;
    void *handle;
} sz_memory_allocator_t;

/**
 *  @brief  The number of bytes a stack-allocated string can hold, including the NULL termination character.
 */
#define sz_string_stack_space (23)

/**
 *  @brief  Tiny memory-owning string structure with a Small String Optimization (SSO).
 *          Differs in layout from Folly, Clang, GCC, and probably most other implementations.
 *          It's designed to avoid any branches on read-only operations, and can store up
 *          to 22 characters on stack, followed by the NULL-termination character.
 *
 *  @section Changing Length
 *
 *  One nice thing about this design, is that you can, in many cases, change the length of the string
 *  without any branches, invoking a `+=` or `-=` on the 64-bit `length` field. If the string is on heap,
 *  the solution is obvious. If it's on stack, inplace decrement wouldn't affect the top bytes of the string,
 *  only changing the last byte containing the length.
 */
typedef union sz_string_t {

    struct internal {
        sz_ptr_t start;
        sz_u8_t length;
        char chars[sz_string_stack_space];
    } internal;

    struct external {
        sz_ptr_t start;
        sz_size_t length;
        /// @brief Number of bytes, that have been allocated for this string, equals to (capacity + 1).
        sz_size_t space;
        sz_size_t padding;
    } external;

    sz_u64_t u64s[4];

} sz_string_t;

#pragma region API

typedef sz_u64_t (*sz_hash_t)(sz_cptr_t, sz_size_t);
typedef sz_bool_t (*sz_equal_t)(sz_cptr_t, sz_cptr_t, sz_size_t);
typedef sz_ordering_t (*sz_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/**
 *  @brief  Computes the hash of a string.
 *
 *  Preferences for the ideal hash:
 *  - 64 bits long.
 *  - Fast on short strings.
 *  - Short implementation.
 *  - Supports rolling computation.
 *  - For two strings with known hashes, the hash of their concatenation can be computed in sublinear time.
 *  - Invariance to zero characters? Maybe only at start/end?
 *
 *  @section    Why not use vanilla CRC32?
 *
 *  Cyclic Redundancy Check 32 is one of the most commonly used hash functions in Computer Science.
 *  It has in-hardware support on both x86 and Arm, for both 8-bit, 16-bit, 32-bit, and 64-bit words.
 *  The `0x1EDC6F41` polynomial is used in iSCSI, Btrfs, ext4, and the `0x04C11DB7` in SATA, Ethernet, Zlib, PNG.
 *  In case of Arm more than one polynomial is supported. It is, however, somewhat limiting for Big Data
 *  usecases, which often have to deal with more than 4 Billion strings, making collisions unavoidable.
 *  Moreover, the existing SIMD approaches are tricky, combining general purpose computations with
 *  specialized instructions, to utilize more silicon in every cycle.
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
 *  the super-scalar nature of modern CPUs and producing 64-bit and 128-bit hashes.
 *  https://opensource.googleblog.com/2011/04/introducing-cityhash
 *  https://github.com/Cyan4973/xxHash
 *
 *  Neither of those functions are cryptographic, unlike MD5, SHA, and BLAKE algorithms.
 *  Most of those are based on the Merkle-Damgård construction, and aren't resistant to
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
SZ_PUBLIC sz_u64_t sz_hash_avx512(sz_cptr_t text, sz_size_t length) { return sz_hash_serial(text, length); }
SZ_PUBLIC sz_u64_t sz_hash_neon(sz_cptr_t text, sz_size_t length) { return sz_hash_serial(text, length); }

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

/**
 *  @brief  Generates a random string for a given alphabet, avoiding integer division and modulo operations.
 *          Similar to `text[i] = alphabet[rand() % cardinality]`.
 *
 *  The modulo operation is expensive, and should be avoided in performance-critical code.
 *  We avoid it using small lookup tables and replacing it with a multiplication and shifts, similar to `libdivide`.
 *  Alternative algorithms would include:
 *      - Montgomery form: https://en.algorithmica.org/hpc/number-theory/montgomery/
 *      - Barret reduction: https://www.nayuki.io/page/barrett-reduction-algorithm
 *      - Lemire's trick: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
 *
 *  @param alphabet     Set of characters to sample from.
 *  @param cardinality  Number of characters to sample from.
 *  @param text         Output string, can point to the same address as ::text.
 *  @param generate     Callback producing random numbers given the generator state.
 *  @param generator    Generator state, can be a pointer to a seed, or a pointer to a random number generator.
 */
SZ_PUBLIC void sz_generate(sz_cptr_t alphabet, sz_size_t cardinality, sz_ptr_t text, sz_size_t length,
                           sz_random_generator_t generate, void *generator);

/**
 *  @brief  Similar to `memcpy`, copies contents of one string into another.
 *          The behavior is undefined if the strings overlap.
 *
 *  @param target   String to copy into.
 *  @param length   Number of bytes to copy.
 *  @param source   String to copy from.
 */
SZ_PUBLIC void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
SZ_PUBLIC void sz_copy_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
SZ_PUBLIC void sz_copy_avx512(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
SZ_PUBLIC void sz_copy_avx2(sz_ptr_t target, sz_cptr_t source, sz_size_t length);

/**
 *  @brief  Similar to `memmove`, copies (moves) contents of one string into another.
 *          Unlike `sz_copy`, allows overlapping strings as arguments.
 *
 *  @param target   String to copy into.
 *  @param length   Number of bytes to copy.
 *  @param source   String to copy from.
 */
SZ_PUBLIC void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
SZ_PUBLIC void sz_move_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
SZ_PUBLIC void sz_move_avx512(sz_ptr_t target, sz_cptr_t source, sz_size_t length);
SZ_PUBLIC void sz_move_avx2(sz_ptr_t target, sz_cptr_t source, sz_size_t length);

/**
 *  @brief  Similar to `memset`, fills a string with a given value.
 *
 *  @param target   String to fill.
 *  @param length   Number of bytes to fill.
 *  @param value    Value to fill with.
 */
SZ_PUBLIC void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value);
SZ_PUBLIC void sz_fill_serial(sz_ptr_t target, sz_size_t length, sz_u8_t value);
SZ_PUBLIC void sz_fill_avx512(sz_ptr_t target, sz_size_t length, sz_u8_t value);
SZ_PUBLIC void sz_fill_avx2(sz_ptr_t target, sz_size_t length, sz_u8_t value);

/**
 *  @brief  Initializes a string class instance to an empty value.
 */
SZ_PUBLIC void sz_string_init(sz_string_t *string);

/**
 *  @brief  Convenience function checking if the provided string is stored inside of the ::string instance itself,
 *          alternative being - allocated in a remote region of the heap.
 */
SZ_PUBLIC sz_bool_t sz_string_is_on_stack(sz_string_t const *string);

/**
 *  @brief  Unpacks the opaque instance of a string class into its components.
 *          Recommended to use only in read-only operations.
 *
 *  @param string       String to unpack.
 *  @param start        Pointer to the start of the string.
 *  @param length       Number of bytes in the string, before the NULL character.
 *  @param space        Number of bytes allocated for the string (heap or stack), including the NULL character.
 *  @param is_external  Whether the string is allocated on the heap externally, or fits withing ::string instance.
 */
SZ_PUBLIC void sz_string_unpack(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length, sz_size_t *space,
                                sz_bool_t *is_external);

/**
 *  @brief  Unpacks only the start and length of the string.
 *          Recommended to use only in read-only operations.
 *
 * @param string       String to unpack.
 * @param start        Pointer to the start of the string.
 * @param length       Number of bytes in the string, before the NULL character.
 */
SZ_PUBLIC void sz_string_range(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length);

/**
 *  @brief  Constructs a string of a given ::length with noisy contents.
 *          Use the returned character pointer to populate the string.
 *
 *  @param string       String to initialize.
 *  @param length       Number of bytes in the string, before the NULL character.
 *  @param allocator    Memory allocator to use for the allocation.
 *  @return             NULL if the operation failed, pointer to the start of the string otherwise.
 */
SZ_PUBLIC sz_ptr_t sz_string_init_length(sz_string_t *string, sz_size_t length, sz_memory_allocator_t *allocator);

/**
 *  @brief  Doesn't change the contents or the length of the string, but grows the available memory capacity.
 *          This is beneficial, if several insertions are expected, and we want to minimize allocations.
 *
 *  @param string       String to grow.
 *  @param new_capacity The number of characters to reserve space for, including existing ones.
 *  @param allocator    Memory allocator to use for the allocation.
 *  @return             True if the operation succeeded. False if memory allocation failed.
 */
SZ_PUBLIC sz_bool_t sz_string_reserve(sz_string_t *string, sz_size_t new_capacity, sz_memory_allocator_t *allocator);

/**
 *  @brief  Grows the string by adding an uninitialized region of ::added_length at the given ::offset.
 *          Would often be used in conjunction with one or more `sz_copy` calls to populate the allocated region.
 *          Similar to `sz_string_reserve`, but changes the length of the ::string.
 *
 *  @param string       String to grow.
 *  @param offset       Offset of the first byte to reserve space for.
 *                      If provided offset is larger than the length, it will be capped.
 *  @param added_length The number of new characters to reserve space for.
 *  @param allocator    Memory allocator to use for the allocation.
 *  @return             NULL if the operation failed, pointer to the new start of the string otherwise.
 */
SZ_PUBLIC sz_ptr_t sz_string_expand(sz_string_t *string, sz_size_t offset, sz_size_t added_length,
                                    sz_memory_allocator_t *allocator);

/**
 *  @brief  Removes a range from a string. Changes the length, but not the capacity.
 *          Performs no allocations or deallocations and can't fail.
 *
 *  @param string       String to clean.
 *  @param offset       Offset of the first byte to remove.
 *  @param length       Number of bytes to remove. Out-of-bound ranges will be capped.
 *  @return             Number of bytes removed.
 */
SZ_PUBLIC sz_size_t sz_string_erase(sz_string_t *string, sz_size_t offset, sz_size_t length);

/**
 *  @brief  Shrinks the string to fit the current length, if it's allocated on the heap.
 *          Teh reverse operation of ::sz_string_reserve.
 *
 *  @param string       String to shrink.
 *  @param allocator    Memory allocator to use for the allocation.
 *  @return             Whether the operation was successful. The only failures can come from the allocator.
 */
SZ_PUBLIC sz_ptr_t sz_string_shrink_to_fit(sz_string_t *string, sz_memory_allocator_t *allocator);

/**
 *  @brief  Frees the string, if it's allocated on the heap.
 *          If the string is on the stack, the function clears/resets the state.
 */
SZ_PUBLIC void sz_string_free(sz_string_t *string, sz_memory_allocator_t *allocator);

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

/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_avx2(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

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

/** @copydoc sz_find_last_byte */
SZ_PUBLIC sz_cptr_t sz_find_last_byte_avx2(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/** @copydoc sz_find_last_byte */
SZ_PUBLIC sz_cptr_t sz_find_last_byte_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/**
 *  @brief  Locates first matching substring.
 *          Equivalent to `memmem(haystack, h_length, needle, n_length)` in LibC.
 *          Similar to `strstr(haystack, needle)` in LibC, but requires known length.
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
SZ_PUBLIC sz_cptr_t sz_find_avx2(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/**
 *  @brief  Locates the last matching substring.
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
SZ_PUBLIC sz_cptr_t sz_find_last_avx2(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

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
SZ_PUBLIC sz_cptr_t sz_find_from_set(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

/** @copydoc sz_find_from_set */
SZ_PUBLIC sz_cptr_t sz_find_from_set_serial(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

/** @copydoc sz_find_from_set */
SZ_PUBLIC sz_cptr_t sz_find_from_set_avx512(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

/** @copydoc sz_find_from_set */
SZ_PUBLIC sz_cptr_t sz_find_from_set_avx2(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

/** @copydoc sz_find_from_set */
SZ_PUBLIC sz_cptr_t sz_find_from_set_neon(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

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
SZ_PUBLIC sz_cptr_t sz_find_last_from_set(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

/** @copydoc sz_find_last_from_set */
SZ_PUBLIC sz_cptr_t sz_find_last_from_set_serial(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

/** @copydoc sz_find_last_from_set */
SZ_PUBLIC sz_cptr_t sz_find_last_from_set_avx512(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

/** @copydoc sz_find_last_from_set */
SZ_PUBLIC sz_cptr_t sz_find_last_from_set_avx2(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

/** @copydoc sz_find_last_from_set */
SZ_PUBLIC sz_cptr_t sz_find_last_from_set_neon(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set);

#pragma endregion

#pragma region String Similarity Measures

/**
 *  @brief  Computes the Levenshtein edit-distance between two strings using the Wagner-Fisher algorithm.
 *          Similar to the Needleman-Wunsch alignment algorithm. Often used in fuzzy string matching.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *  @param alloc    Temporary memory allocator, that will allocate at most two rows of the Levenshtein matrix.
 *  @param bound    Upper bound on the distance, that allows us to exit early.
 *  @return         Unsigned edit distance.
 */
SZ_PUBLIC sz_size_t sz_edit_distance(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                     sz_size_t bound, sz_memory_allocator_t const *alloc);

/** @copydoc sz_edit_distance */
SZ_PUBLIC sz_size_t sz_edit_distance_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                            sz_size_t bound, sz_memory_allocator_t const *alloc);

/** @copydoc sz_edit_distance */
SZ_PUBLIC sz_size_t sz_edit_distance_avx512(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
                                            sz_size_t bound, sz_memory_allocator_t const *alloc);

/**
 *  @brief  Computes Needleman–Wunsch alignment score for two string. Often used in bioinformatics and cheminformatics.
 *          Similar to the Levenshtein edit-distance, parameterized for gap and substitution penalties.
 *
 *  This function is equivalent to the default Levenshtein distance implementation with the ::gap parameter set
 *  to one, and the ::subs matrix formed of all ones except for the main diagonal, which is zeros.
 *  Unlike the default Levenshtein implementation, this can't be bounded, as the substitution costs can be both positive
 *  and negative, meaning that the distance isn't monotonically growing as we go through the strings.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *  @param gap      Penalty cost for gaps - insertions and removals.
 *  @param subs     Substitution costs matrix with 256 x 256 values for all pairs of characters.
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

#if 0
/**
 *  @brief  Computes the Karp-Rabin rolling hash of a string outputting a binary fingerprint.
 *          Such fingerprints can be compared with Hamming or Jaccard (Tanimoto) distance for similarity.
 */
SZ_PUBLIC sz_ssize_t sz_fingerprint_rolling(sz_cptr_t text, sz_size_t length,                  //
                                            sz_ptr_t fingerprint, sz_size_t fingerprint_bytes, //
                                            sz_size_t window_length) {
    /// The size of our alphabet.
    sz_u64_t base = 256;
    /// Define a large prime number that we are going to use for modulo arithmetic.
    /// Fun fact, the largest signed 32-bit signed integer (2147483647) is a prime number.
    /// But we are going to use a larger one, to reduce collisions.
    /// https://www.mersenneforum.org/showthread.php?t=3471
    sz_u64_t prime = 18446744073709551557ull;
    /// The `prime ^ window_length` value, that we are going to use for modulo arithmetic.
    sz_u64_t prime_power = 1;
    for (sz_size_t i = 0; i <= w; ++i) prime_power = (prime_power * base) % prime;
    /// Here we stick to 32-bit hashes as 64-bit modulo arithmetic is expensive.
    sz_u64_t hash = 0;
    /// Compute the initial hash value for the first window.
    sz_cptr_t text_end = text + length;
    for (sz_cptr_t first_end = text + window_length; text < first_end; ++text) hash = (hash * base + *text) % prime;

    /// In most cases the fingerprint length will be a power of two.
    sz_bool_t fingerprint_length_is_power_of_two = fingerprint_bytes & (fingerprint_bytes - 1);
    sz_u8_t *fingerprint_u8s = (sz_u8_t *)fingerprint;
    if (!fingerprint_length_is_power_of_two) {
        /// Compute the hash value for every window, exporting into the fingerprint,
        /// using the expensive modulo operation.
        for (; text < text_end; ++text) {
            hash = (base * (hash - *(text - window_length) * h) + *text) % prime;
            sz_size_t byte_offset = (hash / 8) % fingerprint_bytes;
            fingerprint_u8s[byte_offset] |= (1 << (hash & 7));
        }
    }
    else {
        /// Compute the hash value for every window, exporting into the fingerprint,
        /// using a cheap bitwise-and operation to determine the byte offset
        for (; text < text_end; ++text) {
            hash = (base * (hash - *(text - window_length) * h) + *text) % prime;
            sz_size_t byte_offset = (hash / 8) & (fingerprint_bytes - 1);
            fingerprint_u8s[byte_offset] |= (1 << (hash & 7));
        }
    }
}

#endif

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
SZ_INTERNAL int sz_u32_popcount(sz_u32_t x) { return __popcnt32(x); }
SZ_INTERNAL int sz_u32_ctz(sz_u32_t x) { return _tzcnt_u32(x); }
SZ_INTERNAL int sz_u32_clz(sz_u32_t x) { return _lzcnt_u32(x); }
SZ_INTERNAL sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return _byteswap_uint32(val); }
#else
SZ_INTERNAL int sz_u64_popcount(sz_u64_t x) { return __builtin_popcountll(x); }
SZ_INTERNAL int sz_u64_ctz(sz_u64_t x) { return __builtin_ctzll(x); }
SZ_INTERNAL int sz_u64_clz(sz_u64_t x) { return __builtin_clzll(x); }
SZ_INTERNAL sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return __builtin_bswap64(val); }
SZ_INTERNAL int sz_u32_popcount(sz_u32_t x) { return __builtin_popcount(x); }
SZ_INTERNAL int sz_u32_ctz(sz_u32_t x) { return __builtin_ctz(x); } // ! Undefined if `x == 0`
SZ_INTERNAL int sz_u32_clz(sz_u32_t x) { return __builtin_clz(x); } // ! Undefined if `x == 0`
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
 *  @brief  Branchless minimum function for two integers.
 */
SZ_INTERNAL sz_i32_t sz_i32_min_of_two(sz_i32_t x, sz_i32_t y) { return y + ((x - y) & (x - y) >> 31); }

/**
 *  @brief  Clamps signed offsets in a string to a valid range. Used for Pythonic-style slicing.
 */
SZ_INTERNAL void sz_ssize_clamp_interval(sz_size_t length, sz_ssize_t start, sz_ssize_t end,
                                         sz_size_t *normalized_offset, sz_size_t *normalized_length) {
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
    sz_assert(x > 0 && "Non-positive numbers have no defined logarithm");
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
    x |= x >> 32;
    x++;
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
 *  @brief  Helper structure to simplify work with 16-bit words.
 *  @see    sz_u16_load
 */
typedef union sz_u16_vec_t {
    sz_u16_t u16;
    sz_u8_t u8s[2];
} sz_u16_vec_t;

/**
 *  @brief Load a 16-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u16_vec_t sz_u16_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u16_vec_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    return result;
#elif defined(_MSC_VER)
    return *((__unaligned sz_u16_vec_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u16_vec_t const *result = (sz_u16_vec_t const *)ptr;
    return *result;
#endif
}

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
#elif defined(_MSC_VER)
    return *((__unaligned sz_u32_vec_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u32_vec_t const *result = (sz_u32_vec_t const *)ptr;
    return *result;
#endif
}

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
#elif defined(_MSC_VER)
    return *((__unaligned sz_u64_vec_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u64_vec_t const *result = (sz_u64_vec_t const *)ptr;
    return *result;
#endif
}

SZ_INTERNAL sz_ptr_t _sz_memory_allocate_for_static_buffer(sz_size_t length, sz_string_view_t *string_view) {
    if (length > string_view->length) return NULL;
    return (sz_ptr_t)string_view->start;
}

SZ_INTERNAL void _sz_memory_free_for_static_buffer(sz_ptr_t start, sz_size_t length, sz_string_view_t *string_view) {
    sz_unused(start && length && string_view);
}

SZ_PUBLIC void sz_memory_allocator_init_for_static_buffer(sz_string_view_t buffer, sz_memory_allocator_t *alloc) {
    alloc->allocate = (sz_memory_allocate_t)_sz_memory_allocate_for_static_buffer;
    alloc->free = (sz_memory_free_t)_sz_memory_free_for_static_buffer;
    alloc->handle = &buffer;
}

#pragma endregion

#pragma region Serial Implementation

SZ_PUBLIC sz_u64_t sz_hash_serial(sz_cptr_t start, sz_size_t length) {

    sz_u64_t const c1 = 0x87c37b91114253d5ull;
    sz_u64_t const c2 = 0x4cf5ad432745937full;
    sz_u64_vec_t k1, k2;
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
    case 15: k2.u8s[6] = start[14]; SZ_FALLTHROUGH;
    case 14: k2.u8s[5] = start[13]; SZ_FALLTHROUGH;
    case 13: k2.u8s[4] = start[12]; SZ_FALLTHROUGH;
    case 12: k2.u8s[3] = start[11]; SZ_FALLTHROUGH;
    case 11: k2.u8s[2] = start[10]; SZ_FALLTHROUGH;
    case 10: k2.u8s[1] = start[9]; SZ_FALLTHROUGH;
    case 9:
        k2.u8s[0] = start[8];
        k2.u64 *= c2;
        k2.u64 = sz_u64_rotl(k2.u64, 33);
        k2.u64 *= c1;
        h2 ^= k2.u64;
        SZ_FALLTHROUGH;

    case 8: k1.u8s[7] = start[7]; SZ_FALLTHROUGH;
    case 7: k1.u8s[6] = start[6]; SZ_FALLTHROUGH;
    case 6: k1.u8s[5] = start[5]; SZ_FALLTHROUGH;
    case 5: k1.u8s[4] = start[4]; SZ_FALLTHROUGH;
    case 4: k1.u8s[3] = start[3]; SZ_FALLTHROUGH;
    case 3: k1.u8s[2] = start[2]; SZ_FALLTHROUGH;
    case 2: k1.u8s[1] = start[1]; SZ_FALLTHROUGH;
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

SZ_PUBLIC sz_cptr_t sz_find_from_set_serial(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set) {
    for (sz_cptr_t const end = text + length; text != end; ++text)
        if (sz_u8_set_contains(set, *text)) return text;
    return NULL;
}

SZ_PUBLIC sz_cptr_t sz_find_last_from_set_serial(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
    sz_cptr_t const end = text;
    for (text += length; text != end;)
        if (sz_u8_set_contains(set, *(text -= 1))) return text;
    return NULL;
#pragma GCC diagnostic pop
}

SZ_PUBLIC sz_ordering_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
    sz_bool_t a_shorter = (sz_bool_t)(a_length < b_length);
    sz_size_t min_length = a_shorter ? a_length : b_length;
    sz_cptr_t min_end = a + min_length;
#if SZ_USE_MISALIGNED_LOADS
    for (sz_u64_vec_t a_vec, b_vec; a + 8 <= min_end; a += 8, b += 8) {
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
 *  @brief  Find the first occurrence of a @b single-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *          Identical to `memchr(haystack, needle[0], haystack_length)`.
 */
SZ_PUBLIC sz_cptr_t sz_find_byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    if (!h_length) return NULL;
    sz_cptr_t const h_end = h + h_length;

#if !SZ_USE_MISALIGNED_LOADS
    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h < h_end; ++h)
        if (*h == *n) return h;
#endif

    // Broadcast the n into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t h_vec, n_vec, match_vec;
    match_vec.u64 = 0;
    n_vec.u64 = (sz_u64_t)n[0] * 0x0101010101010101ull;
    for (; h + 8 <= h_end; h += 8) {
        h_vec.u64 = *(sz_u64_t const *)h;
        match_vec = _sz_u64_each_byte_equal(h_vec, n_vec);
        if (match_vec.u64) return h + sz_u64_ctz(match_vec.u64) / 8;
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
sz_cptr_t sz_find_last_byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    if (!h_length) return NULL;
    sz_cptr_t const h_start = h;

    // Reposition the `h` pointer to the end, as we will be walking backwards.
    h = h + h_length - 1;

#if !SZ_USE_MISALIGNED_LOADS
    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)(h + 1) & 7ull) && h >= h_start; --h)
        if (*h == *n) return h;
#endif

    // Broadcast the n into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t h_vec, n_vec, match_vec;
    n_vec.u64 = (sz_u64_t)n[0] * 0x0101010101010101ull;
    for (; h >= h_start + 7; h -= 8) {
        h_vec.u64 = *(sz_u64_t const *)(h - 7);
        match_vec = _sz_u64_each_byte_equal(h_vec, n_vec);
        if (match_vec.u64) return h - sz_u64_clz(match_vec.u64) / 8;
    }

    for (; h >= h_start; --h)
        if (*h == *n) return h;
    return NULL;
}

/**
 *  @brief  2Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 2byte signifies a match.
 */
SZ_INTERNAL sz_u64_vec_t _sz_u64_each_2byte_equal(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each 2byte is set.
    // For that take the bottom 15 bits of each 2byte, add one to them,
    // and if this sets the top bit to one, then all the 15 bits are ones as well.
    vec.u64 = ((vec.u64 & 0x7FFF7FFF7FFF7FFFull) + 0x0001000100010001ull) & ((vec.u64 & 0x8000800080008000ull));
    return vec;
}

/**
 *  @brief  Find the first occurrence of a @b two-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 offsets at a time.
 */
SZ_INTERNAL sz_cptr_t _sz_find_2byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    // This is an internal method, and the haystack is guaranteed to be at least 2 bytes long.
    sz_assert(h_length >= 2 && "The haystack is too short.");
    sz_cptr_t const h_end = h + h_length;

#if !SZ_USE_MISALIGNED_LOADS
    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h < h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) == 2) return h;
#endif

    sz_u64_vec_t h_even_vec, h_odd_vec, n_vec, matches_even_vec, matches_odd_vec;
    n_vec.u64 = 0;
    n_vec.u8s[0] = n[0], n_vec.u8s[1] = n[1];
    n_vec.u64 *= 0x0001000100010001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time.
    for (; h + 9 <= h_end; h += 8) {
        h_even_vec.u64 = *(sz_u64_t *)h;
        h_odd_vec.u64 = (h_even_vec.u64 >> 8) | (*(sz_u64_t *)&h[8] << 56);
        matches_even_vec = _sz_u64_each_2byte_equal(h_even_vec, n_vec);
        matches_odd_vec = _sz_u64_each_2byte_equal(h_odd_vec, n_vec);

        if (matches_even_vec.u64 + matches_odd_vec.u64) {
            matches_even_vec.u64 >>= 8;
            sz_u64_t match_indicators = matches_even_vec.u64 | matches_odd_vec.u64;
            return h + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; h + 2 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) == 2) return h;
    return NULL;
}

/**
 *  @brief  4Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 4byte signifies a match.
 */
SZ_INTERNAL sz_u64_vec_t _sz_u64_each_4byte_equal(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each 4byte is set.
    // For that take the bottom 31 bits of each 4byte, add one to them,
    // and if this sets the top bit to one, then all the 31 bits are ones as well.
    vec.u64 = ((vec.u64 & 0x7FFFFFFF7FFFFFFFull) + 0x0000000100000001ull) & ((vec.u64 & 0x8000000080000000ull));
    return vec;
}

/**
 *  @brief  Find the first occurrence of a @b four-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 offsets at a time.
 */
SZ_INTERNAL sz_cptr_t _sz_find_4byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    // This is an internal method, and the haystack is guaranteed to be at least 4 bytes long.
    sz_assert(h_length >= 4 && "The haystack is too short.");
    sz_cptr_t const h_end = h + h_length;

#if !SZ_USE_MISALIGNED_LOADS
    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h < h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) + (h[2] == n[2]) + (h[3] == n[3]) == 4) return h;
#endif

    sz_u64_vec_t h0_vec, h1_vec, h2_vec, h3_vec, n_vec, matches0_vec, matches1_vec, matches2_vec, matches3_vec;
    n_vec.u64 = 0;
    n_vec.u8s[0] = n[0], n_vec.u8s[1] = n[1], n_vec.u8s[2] = n[2], n_vec.u8s[3] = n[3];
    n_vec.u64 *= 0x0000000100000001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time using four 64-bit words.
    // We load the subsequent word at onceto minimize the data dependency.
    sz_u64_t h_page_current, h_page_next;
    for (; h + 16 <= h_end; h += 8) {
        h_page_current = *(sz_u64_t *)h;
        h_page_next = *(sz_u64_t *)(h + 8);
        h0_vec.u64 = (h_page_current);
        h1_vec.u64 = (h_page_current >> 8) | (h_page_next << 56);
        h2_vec.u64 = (h_page_current >> 16) | (h_page_next << 48);
        h3_vec.u64 = (h_page_current >> 24) | (h_page_next << 40);
        matches0_vec = _sz_u64_each_4byte_equal(h0_vec, n_vec);
        matches1_vec = _sz_u64_each_4byte_equal(h1_vec, n_vec);
        matches2_vec = _sz_u64_each_4byte_equal(h2_vec, n_vec);
        matches3_vec = _sz_u64_each_4byte_equal(h3_vec, n_vec);

        if (matches0_vec.u64 + matches1_vec.u64 + matches2_vec.u64 + matches3_vec.u64) {
            matches0_vec.u64 >>= 24;
            matches1_vec.u64 >>= 16;
            matches2_vec.u64 >>= 8;
            sz_u64_t match_indicators = matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64;
            return h + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; h + 4 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) + (h[2] == n[2]) + (h[3] == n[3]) == 4) return h;
    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 8-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_8bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                        sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u8_t running_match = 0xFF;
    sz_u8_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 8-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_last_bitap_upto_8bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                             sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u8_t running_match = 0xFF;
    sz_u8_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 16-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_16bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u16_t running_match = 0xFFFF;
    sz_u16_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 16-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_last_bitap_upto_16bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                              sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u16_t running_match = 0xFFFF;
    sz_u16_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 32-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_32bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u32_t running_match = 0xFFFFFFFF;
    sz_u32_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 32-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_last_bitap_upto_32bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                              sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u32_t running_match = 0xFFFFFFFF;
    sz_u32_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 64-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 64-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_last_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                              sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for approximate matching of patterns up to @b 64-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bounded_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                                 sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for approximate matching of patterns up to @b 64-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bounded_last_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                                      sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Boyer-Moore-Horspool algorithm for exact matching of patterns up to @b 256-bytes long.
 *          Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
 */
SZ_INTERNAL sz_cptr_t _sz_find_horspool_upto_256bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                             sz_size_t n_length) {

    // Several popular string matching algorithms are using a bad-character shift table.
    // Boyer Moore: https://www-igm.univ-mlv.fr/~lecroq/string/node14.html
    // Quick Search: https://www-igm.univ-mlv.fr/~lecroq/string/node19.html
    // Smith: https://www-igm.univ-mlv.fr/~lecroq/string/node21.html
    sz_u8_t bad_shift_table[256] = {(sz_u8_t)n_length};
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    for (sz_size_t i = 0; i + 1 < n_length; ++i) bad_shift_table[n_unsigned[i]] = (sz_u8_t)(n_length - i - 1);

    // Another common heuristic is to match a few characters from different parts of a string.
    // Raita suggests to use the first two, the last, and the middle character of the pattern.
    sz_size_t n_midpoint = n_length / 2 + 1;
    sz_u32_vec_t h_vec, n_vec;
    n_vec.u8s[0] = n[0];
    n_vec.u8s[1] = n[1];
    n_vec.u8s[2] = n[n_midpoint];
    n_vec.u8s[3] = n[n_length - 1];

    // Scan through the whole haystack, skipping the last `n_length` bytes.
    for (sz_size_t i = 0; i <= h_length - n_length;) {
        h_vec.u8s[0] = h[i + 0];
        h_vec.u8s[1] = h[i + 1];
        h_vec.u8s[2] = h[i + n_midpoint];
        h_vec.u8s[3] = h[i + n_length - 1];
        if (h_vec.u32 == n_vec.u32 && sz_equal_serial(h + i + 2, n + 2, n_length - 3)) return h + i;
        i += bad_shift_table[h_vec.u8s[3]];
    }
    return NULL;
}

SZ_INTERNAL sz_cptr_t _sz_find_last_horspool_upto_256bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                                  sz_size_t n_length) {
    sz_u8_t bad_shift_table[256] = {(sz_u8_t)n_length};
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    for (sz_size_t i = 0; i + 1 < n_length; ++i) bad_shift_table[n_unsigned[i]] = (sz_u8_t)(i + 1);

    sz_size_t n_midpoint = n_length / 2;
    sz_u32_vec_t h_vec, n_vec;
    n_vec.u8s[0] = n[n_length - 1];
    n_vec.u8s[1] = n[n_length - 2];
    n_vec.u8s[2] = n[n_midpoint];
    n_vec.u8s[3] = n[0];

    for (sz_size_t j = 0; j <= h_length - n_length;) {
        sz_size_t i = h_length - n_length - j;
        h_vec.u8s[0] = h[i + n_length - 1];
        h_vec.u8s[1] = h[i + n_length - 2];
        h_vec.u8s[2] = h[i + n_midpoint];
        h_vec.u8s[3] = h[i];
        if (h_vec.u32 == n_vec.u32 && sz_equal_serial(h + i + 1, n + 1, n_length - 3)) return h + i;
        j += bad_shift_table[h_vec.u8s[0]];
    }
    return NULL;
}

/**
 *  @brief  Exact substring search helper function, that finds the first occurrence of a prefix of the needle
 *          using a given search function, and then verifies the remaining part of the needle.
 */
SZ_INTERNAL sz_cptr_t _sz_find_with_prefix(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length,
                                           sz_find_t find_prefix, sz_size_t prefix_length) {

    sz_size_t suffix_length = n_length - prefix_length;
    while (1) {
        sz_cptr_t found = find_prefix(h, h_length, n, prefix_length);
        if (!found) return NULL;

        // Verify the remaining part of the needle
        sz_size_t remaining = h_length - (found - h);
        if (remaining < suffix_length) return NULL;
        if (sz_equal_serial(found + prefix_length, n + prefix_length, suffix_length)) return found;

        // Adjust the position.
        h = found + 1;
        h_length = remaining - 1;
    }

    // Unreachable, but helps silence compiler warnings:
    return NULL;
}

/**
 *  @brief  Exact reverse-order substring search helper function, that finds the last occurrence of a suffix of the
 *          needle using a given search function, and then verifies the remaining part of the needle.
 */
SZ_INTERNAL sz_cptr_t _sz_find_last_with_suffix(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length,
                                                sz_find_t find_suffix, sz_size_t suffix_length) {

    sz_size_t prefix_length = n_length - suffix_length;
    while (1) {
        sz_cptr_t found = find_suffix(h, h_length, n + prefix_length, suffix_length);
        if (!found) return NULL;

        // Verify the remaining part of the needle
        sz_size_t remaining = found - h;
        if (remaining < prefix_length) return NULL;
        if (sz_equal_serial(found - prefix_length, n, prefix_length)) return found - prefix_length;

        // Adjust the position.
        h_length = remaining - 1;
    }

    // Unreachable, but helps silence compiler warnings:
    return NULL;
}

SZ_INTERNAL sz_cptr_t _sz_find_horspool_over_256bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                             sz_size_t n_length) {
    return _sz_find_with_prefix(h, h_length, n, n_length, _sz_find_horspool_upto_256bytes_serial, 256);
}

SZ_INTERNAL sz_cptr_t _sz_find_last_horspool_over_256bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                                  sz_size_t n_length) {
    return _sz_find_last_with_suffix(h, h_length, n, n_length, _sz_find_last_horspool_upto_256bytes_serial, 256);
}

SZ_PUBLIC sz_cptr_t sz_find_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return NULL;

    sz_find_t backends[] = {
        // For very short strings brute-force SWAR makes sense.
        (sz_find_t)sz_find_byte_serial,
        (sz_find_t)_sz_find_2byte_serial,
        (sz_find_t)_sz_find_bitap_upto_8bytes_serial,
        (sz_find_t)_sz_find_4byte_serial,
        // For needle lengths up to 64, use the Bitap algorithm variation for exact search.
        (sz_find_t)_sz_find_bitap_upto_8bytes_serial,
        (sz_find_t)_sz_find_bitap_upto_16bytes_serial,
        (sz_find_t)_sz_find_bitap_upto_32bytes_serial,
        (sz_find_t)_sz_find_bitap_upto_64bytes_serial,
        // For longer needles - use skip tables.
        (sz_find_t)_sz_find_horspool_upto_256bytes_serial,
        (sz_find_t)_sz_find_horspool_over_256bytes_serial,
    };

    return backends[
        // For very short strings brute-force SWAR makes sense.
        (n_length > 1) + (n_length > 2) + (n_length > 3) +
        // For needle lengths up to 64, use the Bitap algorithm variation for exact search.
        (n_length > 4) + (n_length > 8) + (n_length > 16) + (n_length > 32) +
        // For longer needles - use skip tables.
        (n_length > 64) + (n_length > 256)](h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_last_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return NULL;

    sz_find_t backends[] = {
        // For very short strings brute-force SWAR makes sense.
        (sz_find_t)sz_find_last_byte_serial,
        // For needle lengths up to 64, use the Bitap algorithm variation for reverse-order exact search.
        (sz_find_t)_sz_find_last_bitap_upto_8bytes_serial,
        (sz_find_t)_sz_find_last_bitap_upto_16bytes_serial,
        (sz_find_t)_sz_find_last_bitap_upto_32bytes_serial,
        (sz_find_t)_sz_find_last_bitap_upto_64bytes_serial,
        // For longer needles - use skip tables.
        (sz_find_t)_sz_find_last_horspool_upto_256bytes_serial,
        (sz_find_t)_sz_find_last_horspool_over_256bytes_serial,
    };

    return backends[
        // For very short strings brute-force SWAR makes sense.
        0 +
        // For needle lengths up to 64, use the Bitap algorithm variation for reverse-order exact search.
        (n_length > 1) + (n_length > 8) + (n_length > 16) + (n_length > 32) +
        // For longer needles - use skip tables.
        (n_length > 64) + (n_length > 256)](h, h_length, n, n_length);
}

SZ_PUBLIC sz_size_t sz_edit_distance_serial(     //
    sz_cptr_t longer, sz_size_t longer_length,   //
    sz_cptr_t shorter, sz_size_t shorter_length, //
    sz_size_t bound, sz_memory_allocator_t const *alloc) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one.
    if (longer_length == 0) return shorter_length <= bound ? shorter_length : bound;
    if (shorter_length == 0) return longer_length <= bound ? longer_length : bound;

    // Let's make sure that we use the amount proportional to the
    // number of elements in the shorter string, not the larger.
    if (shorter_length > longer_length) {
        sz_u64_swap((sz_u64_t *)&longer_length, (sz_u64_t *)&shorter_length);
        sz_u64_swap((sz_u64_t *)&longer, (sz_u64_t *)&shorter);
    }

    // If the difference in length is beyond the `bound`, there is no need to check at all.
    if (bound && longer_length - shorter_length > bound) return bound;

    // Skip the matching prefixes and suffixes, they won't affect the distance.
    for (sz_cptr_t a_end = longer + longer_length, b_end = shorter + shorter_length;
         longer != a_end && shorter != b_end && *longer == *shorter;
         ++longer, ++shorter, --longer_length, --shorter_length)
        ;
    for (; longer_length && shorter_length && longer[longer_length - 1] == shorter[shorter_length - 1];
         --longer_length, --shorter_length)
        ;

    // If a buffering memory-allocator is provided, this operation is practically free,
    // and cheaper than allocating even 512 bytes (for small distance matrices) on stack.
    sz_size_t buffer_length = sizeof(sz_size_t) * ((shorter_length + 1) * 2);
    sz_size_t *distances = (sz_size_t *)alloc->allocate(buffer_length, alloc->handle);
    sz_size_t *previous_distances = distances;
    sz_size_t *current_distances = previous_distances + shorter_length + 1;

    for (sz_size_t idx_shorter = 0; idx_shorter != (shorter_length + 1); ++idx_shorter)
        previous_distances[idx_shorter] = idx_shorter;

    // Keeping track of the bound parameter introduces a very noticeable performance penalty.
    // So if it's not provided, we can skip the check altogether.
    if (!bound) {
        for (sz_size_t idx_longer = 0; idx_longer != longer_length; ++idx_longer) {
            current_distances[0] = idx_longer + 1;
            for (sz_size_t idx_shorter = 0; idx_shorter != shorter_length; ++idx_shorter) {
                sz_size_t cost_deletion = previous_distances[idx_shorter + 1] + 1;
                sz_size_t cost_insertion = current_distances[idx_shorter] + 1;
                sz_size_t cost_substitution =
                    previous_distances[idx_shorter] + (longer[idx_longer] != shorter[idx_shorter]);
                // ? It might be a good idea to enforce branchless execution here.
                // ? The caveat being that the benchmarks on longer sequences backfire and more research is needed.
                current_distances[idx_shorter + 1] = sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);
            }
            sz_u64_swap((sz_u64_t *)&previous_distances, (sz_u64_t *)&current_distances);
        }
        sz_size_t result = previous_distances[shorter_length];
        alloc->free(distances, buffer_length, alloc->handle);
        return result;
    }
    //
    else {
        for (sz_size_t idx_longer = 0; idx_longer != longer_length; ++idx_longer) {
            current_distances[0] = idx_longer + 1;

            // Initialize min_distance with a value greater than bound
            sz_size_t min_distance = bound - 1;

            for (sz_size_t idx_shorter = 0; idx_shorter != shorter_length; ++idx_shorter) {
                sz_size_t cost_deletion = previous_distances[idx_shorter + 1] + 1;
                sz_size_t cost_insertion = current_distances[idx_shorter] + 1;
                sz_size_t cost_substitution =
                    previous_distances[idx_shorter] + (longer[idx_longer] != shorter[idx_shorter]);
                current_distances[idx_shorter + 1] = sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);

                // Keep track of the minimum distance seen so far in this row
                min_distance = sz_min_of_two(current_distances[idx_shorter + 1], min_distance);
            }

            // If the minimum distance in this row exceeded the bound, return early
            if (min_distance >= bound) {
                alloc->free(distances, buffer_length, alloc->handle);
                return bound;
            }

            // Swap previous_distances and current_distances pointers
            sz_u64_swap((sz_u64_t *)&previous_distances, (sz_u64_t *)&current_distances);
        }
        sz_size_t result = previous_distances[shorter_length] < bound ? previous_distances[shorter_length] : bound;
        alloc->free(distances, buffer_length, alloc->handle);
        return result;
    }
}

SZ_PUBLIC sz_ssize_t sz_alignment_score_serial(       //
    sz_cptr_t longer, sz_size_t longer_length,        //
    sz_cptr_t shorter, sz_size_t shorter_length,      //
    sz_error_cost_t gap, sz_error_cost_t const *subs, //
    sz_memory_allocator_t const *alloc) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one
    if (longer_length == 0) return shorter_length;
    if (shorter_length == 0) return longer_length;

    // Let's make sure that we use the amount proportional to the
    // number of elements in the shorter string, not the larger.
    if (shorter_length > longer_length) {
        sz_u64_swap((sz_u64_t *)&longer_length, (sz_u64_t *)&shorter_length);
        sz_u64_swap((sz_u64_t *)&longer, (sz_u64_t *)&shorter);
    }

    sz_size_t buffer_length = sizeof(sz_ssize_t) * (shorter_length + 1) * 2;
    sz_ssize_t *distances = (sz_ssize_t *)alloc->allocate(buffer_length, alloc->handle);
    sz_ssize_t *previous_distances = distances;
    sz_ssize_t *current_distances = previous_distances + shorter_length + 1;

    for (sz_size_t idx_shorter = 0; idx_shorter != (shorter_length + 1); ++idx_shorter)
        previous_distances[idx_shorter] = idx_shorter;

    sz_u8_t const *shorter_unsigned = (sz_u8_t const *)shorter;
    for (sz_size_t idx_longer = 0; idx_longer != longer_length; ++idx_longer) {
        current_distances[0] = idx_longer + 1;

        // Initialize min_distance with a value greater than bound
        sz_error_cost_t const *a_subs = subs + longer[idx_longer] * 256ul;
        for (sz_size_t idx_shorter = 0; idx_shorter != shorter_length; ++idx_shorter) {
            sz_ssize_t cost_deletion = previous_distances[idx_shorter + 1] + gap;
            sz_ssize_t cost_insertion = current_distances[idx_shorter] + gap;
            sz_ssize_t cost_substitution = previous_distances[idx_shorter] + a_subs[shorter_unsigned[idx_shorter]];
            current_distances[idx_shorter + 1] = sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);
        }

        // Swap previous_distances and current_distances pointers
        sz_u64_swap((sz_u64_t *)&previous_distances, (sz_u64_t *)&current_distances);
    }

    alloc->free(distances, buffer_length, alloc->handle);
    return previous_distances[shorter_length];
}

/**
 *  @brief  Uses a small lookup-table to convert a lowercase character to uppercase.
 */
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

/**
 *  @brief  Uses a small lookup-table to convert an uppercase character to lowercase.
 */
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

/**
 *  @brief  Uses two small lookup tables (768 bytes total) to accelerate division by a small
 *          unsigned integer. Performs two lookups, one multiplication, two shifts, and two accumulations.
 *
 *  @param  divisor Integral value larger than one.
 *  @param  number  Integral value to divide.
 */
SZ_INTERNAL sz_u8_t sz_u8_divide(sz_u8_t number, sz_u8_t divisor) {
    static sz_u16_t multipliers[256] = {
        0,     0,     0,     21846, 0,     39322, 21846, 9363,  0,     50973, 39322, 29790, 21846, 15124, 9363,  4370,
        0,     57826, 50973, 44841, 39322, 34329, 29790, 25645, 21846, 18351, 15124, 12137, 9363,  6780,  4370,  2115,
        0,     61565, 57826, 54302, 50973, 47824, 44841, 42011, 39322, 36765, 34329, 32006, 29790, 27671, 25645, 23705,
        21846, 20063, 18351, 16706, 15124, 13602, 12137, 10725, 9363,  8049,  6780,  5554,  4370,  3224,  2115,  1041,
        0,     63520, 61565, 59668, 57826, 56039, 54302, 52614, 50973, 49377, 47824, 46313, 44841, 43407, 42011, 40649,
        39322, 38028, 36765, 35532, 34329, 33154, 32006, 30885, 29790, 28719, 27671, 26647, 25645, 24665, 23705, 22766,
        21846, 20945, 20063, 19198, 18351, 17520, 16706, 15907, 15124, 14356, 13602, 12863, 12137, 11424, 10725, 10038,
        9363,  8700,  8049,  7409,  6780,  6162,  5554,  4957,  4370,  3792,  3224,  2665,  2115,  1573,  1041,  517,
        0,     64520, 63520, 62535, 61565, 60609, 59668, 58740, 57826, 56926, 56039, 55164, 54302, 53452, 52614, 51788,
        50973, 50169, 49377, 48595, 47824, 47063, 46313, 45572, 44841, 44120, 43407, 42705, 42011, 41326, 40649, 39982,
        39322, 38671, 38028, 37392, 36765, 36145, 35532, 34927, 34329, 33738, 33154, 32577, 32006, 31443, 30885, 30334,
        29790, 29251, 28719, 28192, 27671, 27156, 26647, 26143, 25645, 25152, 24665, 24182, 23705, 23233, 22766, 22303,
        21846, 21393, 20945, 20502, 20063, 19628, 19198, 18772, 18351, 17933, 17520, 17111, 16706, 16305, 15907, 15514,
        15124, 14738, 14356, 13977, 13602, 13231, 12863, 12498, 12137, 11779, 11424, 11073, 10725, 10380, 10038, 9699,
        9363,  9030,  8700,  8373,  8049,  7727,  7409,  7093,  6780,  6470,  6162,  5857,  5554,  5254,  4957,  4662,
        4370,  4080,  3792,  3507,  3224,  2943,  2665,  2388,  2115,  1843,  1573,  1306,  1041,  778,   517,   258,
    };
    // This table can be avoided using a single addition and counting trailing zeros.
    static sz_u8_t shifts[256] = {
        0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, //
        4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, //
        5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, //
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, //
        6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, //
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, //
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, //
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, //
    };
    sz_u32_t multiplier = multipliers[divisor];
    sz_u8_t shift = shifts[divisor];

    sz_u16_t q = (sz_u16_t)((multiplier * number) >> 16);
    sz_u16_t t = ((number - q) >> 1) + q;
    return (sz_u8_t)(t >> shift);
}

SZ_PUBLIC void sz_tolower_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    sz_u8_t *unsigned_result = (sz_u8_t *)result;
    sz_u8_t const *unsigned_text = (sz_u8_t const *)text;
    sz_u8_t const *end = unsigned_text + length;
    for (; unsigned_text != end; ++unsigned_text, ++unsigned_result) *unsigned_result = sz_u8_tolower(*unsigned_text);
}

SZ_PUBLIC void sz_toupper_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    sz_u8_t *unsigned_result = (sz_u8_t *)result;
    sz_u8_t const *unsigned_text = (sz_u8_t const *)text;
    sz_u8_t const *end = unsigned_text + length;
    for (; unsigned_text != end; ++unsigned_text, ++unsigned_result) *unsigned_result = sz_u8_toupper(*unsigned_text);
}

SZ_PUBLIC void sz_toascii_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    sz_u8_t *unsigned_result = (sz_u8_t *)result;
    sz_u8_t const *unsigned_text = (sz_u8_t const *)text;
    sz_u8_t const *end = unsigned_text + length;
    for (; unsigned_text != end; ++unsigned_text, ++unsigned_result) *unsigned_result = *unsigned_text & 0x7F;
}

SZ_PUBLIC void sz_generate(sz_cptr_t alphabet, sz_size_t alphabet_size, sz_ptr_t result, sz_size_t result_length,
                           sz_random_generator_t generator, void *generator_user_data) {

    sz_assert(alphabet_size > 0 && alphabet_size <= 256 && "Inadequate alphabet size");

    if (alphabet_size == 1)
        for (sz_cptr_t end = result + result_length; result != end; ++result) *result = *alphabet;

    else {
        sz_assert(generator && "Expects a valid random generator");
        for (sz_cptr_t end = result + result_length; result != end; ++result)
            *result = alphabet[sz_u8_divide(generator(generator_user_data) & 0xFF, alphabet_size)];
    }
}

#pragma endregion

/*
 *  Serial implementation of string class operations.
 */
#pragma region Serial Implementation for the String Class

SZ_PUBLIC sz_bool_t sz_string_is_on_stack(sz_string_t const *string) {
    // It doesn't matter if it's on stack or heap, the pointer location is the same.
    return (sz_bool_t)((sz_cptr_t)string->internal.start == (sz_cptr_t)&string->internal.chars[0]);
}

SZ_PUBLIC void sz_string_range(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length) {
    sz_size_t is_small = (sz_cptr_t)string->internal.start == (sz_cptr_t)&string->internal.chars[0];
    sz_size_t is_big_mask = is_small - 1ull;
    *start = string->external.start; // It doesn't matter if it's on stack or heap, the pointer location is the same.
    // If the string is small, use branch-less approach to mask-out the top 7 bytes of the length.
    *length = string->external.length & (0x00000000000000FFull | is_big_mask);
}

SZ_PUBLIC void sz_string_unpack(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length, sz_size_t *space,
                                sz_bool_t *is_external) {
    sz_size_t is_small = (sz_cptr_t)string->internal.start == (sz_cptr_t)&string->internal.chars[0];
    sz_size_t is_big_mask = is_small - 1ull;
    *start = string->external.start; // It doesn't matter if it's on stack or heap, the pointer location is the same.
    // If the string is small, use branch-less approach to mask-out the top 7 bytes of the length.
    *length = string->external.length & (0x00000000000000FFull | is_big_mask);
    // In case the string is small, the `is_small - 1ull` will become 0xFFFFFFFFFFFFFFFFull.
    *space = sz_u64_blend(sz_string_stack_space, string->external.space, is_big_mask);
    *is_external = (sz_bool_t)!is_small;
}

SZ_PUBLIC sz_bool_t sz_string_equal(sz_string_t const *a, sz_string_t const *b) {
    // Tempting to say that the external.length is bitwise the same even if it includes
    // some bytes of the on-stack payload, but we don't at this writing maintain that invariant.
    // (An on-stack string includes noise bytes in the high-order bits of external.length. So do this
    // the hard/correct way.

#if SZ_USE_MISALIGNED_LOADS
    // Dealing with StringZilla strings, we know that the `start` pointer always points
    // to a word at least 8 bytes long. Therefore, we can compare the first 8 bytes at once.

#endif
    // Alternatively, fall back to byte-by-byte comparison.
    sz_ptr_t a_start, b_start;
    sz_size_t a_length, b_length;
    sz_string_range(a, &a_start, &a_length);
    sz_string_range(b, &b_start, &b_length);
    return (sz_bool_t)(a_length == b_length && sz_equal(a_start, b_start, b_length));
}

SZ_PUBLIC sz_ordering_t sz_string_order(sz_string_t const *a, sz_string_t const *b) {
#if SZ_USE_MISALIGNED_LOADS
    // Dealing with StringZilla strings, we know that the `start` pointer always points
    // to a word at least 8 bytes long. Therefore, we can compare the first 8 bytes at once.

#endif
    // Alternatively, fall back to byte-by-byte comparison.
    sz_ptr_t a_start, b_start;
    sz_size_t a_length, b_length;
    sz_string_range(a, &a_start, &a_length);
    sz_string_range(b, &b_start, &b_length);
    return sz_order(a_start, a_length, b_start, b_length);
}

SZ_PUBLIC void sz_string_init(sz_string_t *string) {
    sz_assert(string && "String can't be NULL.");

    // Only 8 + 1 + 1 need to be initialized.
    string->internal.start = &string->internal.chars[0];
    // But for safety let's initialize the entire structure to zeros.
    // string->internal.chars[0] = 0;
    // string->internal.length = 0;
    string->u64s[1] = 0;
    string->u64s[2] = 0;
    string->u64s[3] = 0;
}

SZ_PUBLIC sz_ptr_t sz_string_init_length(sz_string_t *string, sz_size_t length, sz_memory_allocator_t *allocator) {
    sz_size_t space_needed = length + 1; // space for trailing \0
    sz_assert(string && allocator && "String and allocator can't be NULL.");
    // Initialize the string to zeros for safety.
    string->u64s[1] = 0;
    string->u64s[2] = 0;
    string->u64s[3] = 0;
    // If we are lucky, no memory allocations will be needed.
    if (space_needed <= sz_string_stack_space) {
        string->internal.start = &string->internal.chars[0];
        string->internal.length = length;
    }
    else {
        // If we are not lucky, we need to allocate memory.
        string->external.start = (sz_ptr_t)allocator->allocate(space_needed, allocator->handle);
        if (!string->external.start) return NULL;
        string->external.length = length;
        string->external.space = space_needed;
    }
    sz_assert(&string->internal.start == &string->external.start && "Alignment confusion");
    string->external.start[length] = 0;
    return string->external.start;
}

SZ_PUBLIC sz_bool_t sz_string_reserve(sz_string_t *string, sz_size_t new_capacity, sz_memory_allocator_t *allocator) {

    sz_assert(string && "String can't be NULL.");

    sz_size_t new_space = new_capacity + 1;
    if (new_space <= sz_string_stack_space) return sz_true_k;

    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_external);
    sz_assert(new_space > string_space && "New space must be larger than current.");

    sz_ptr_t new_start = (sz_ptr_t)allocator->allocate(new_space, allocator->handle);
    if (!new_start) return sz_false_k;

    sz_copy(new_start, string_start, string_length);
    string->external.start = new_start;
    string->external.space = new_space;
    string->external.padding = 0;
    string->external.length = string_length;

    // Deallocate the old string.
    if (string_is_external) allocator->free(string_start, string_space, allocator->handle);
    return sz_true_k;
}

SZ_PUBLIC sz_ptr_t sz_string_expand(sz_string_t *string, sz_size_t offset, sz_size_t added_length,
                                    sz_memory_allocator_t *allocator) {

    sz_assert(string && allocator && "String and allocator can't be NULL.");

    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_external);

    // The user intended to extend the string.
    offset = sz_min_of_two(offset, string_length);

    // If we are lucky, no memory allocations will be needed.
    if (offset + string_length + added_length < string_space) {
        sz_move(string_start + offset + added_length, string_start + offset, string_length - offset);
        string_start[string_length + added_length] = 0;
        // Even if the string is on the stack, the `+=` won't affect the tail of the string.
        string->external.length += added_length;
    }
    // If we are not lucky, we need to allocate more memory.
    else {
        sz_size_t next_planned_size = sz_max_of_two(SZ_CACHE_LINE_WIDTH, string_space * 2ull);
        sz_size_t min_needed_space = sz_size_bit_ceil(offset + string_length + added_length + 1);
        sz_size_t new_space = sz_max_of_two(min_needed_space, next_planned_size);
        if (!sz_string_reserve(string, new_space - 1, allocator)) return NULL;

        // Copy into the new buffer.
        string_start = string->external.start;
        sz_move(string_start + offset + added_length, string_start + offset, string_length - offset);
        string_start[string_length + added_length] = 0;
        string->external.length = string_length + added_length;
    }

    return string_start;
}

SZ_PUBLIC sz_size_t sz_string_erase(sz_string_t *string, sz_size_t offset, sz_size_t length) {

    sz_assert(string && "String can't be NULL.");

    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_external);

    // Normalize the offset, it can't be larger than the length.
    offset = sz_min_of_two(offset, string_length);

    // We shouldn't normalize the length, to avoid overflowing on `offset + length >= string_length`,
    // if receiving `length == SZ_SIZE_MAX`. After following expression the `length` will contain
    // exactly the delta between original and final length of this `string`.
    length = sz_min_of_two(length, string_length - offset);

    // There are 2 common cases, that wouldn't even require a `memmove`:
    //      1.  Erasing the entire contents of the string.
    //          In that case `length` argument will be equal or greater than `length` member.
    //      2.  Removing the tail of the string with something like `string.pop_back()` in C++.
    //
    // In both of those, regardless of the location of the string - stack or heap,
    // the erasing is as easy as setting the length to the offset.
    // In every other case, we must `memmove` the tail of the string to the left.
    if (offset + length < string_length)
        sz_move(string_start + offset, string_start + offset + length, string_length - offset - length);

    // The `string->external.length = offset` assignment would discard last characters
    // of the on-the-stack string, but inplace subtraction would work.
    string->external.length -= length;
    string_start[string_length - length] = 0;
    return length;
}

SZ_PUBLIC void sz_string_free(sz_string_t *string, sz_memory_allocator_t *allocator) {
    if (!sz_string_is_on_stack(string))
        allocator->free(string->external.start, string->external.space, allocator->handle);
    sz_string_init(string);
}

SZ_PUBLIC void sz_fill_serial(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_ptr_t end = target + length;
    // Dealing with short strings, a single sequential pass would be faster.
    // If the size is larger than 2 words, then at least 1 of them will be aligned.
    // But just one aligned word may not be worth SWAR.
    if (length < SZ_SWAR_THRESHOLD)
        while (target != end) *(target++) = value;

    // In case of long strings, skip unaligned bytes, and then fill the rest in 64-bit chunks.
    else {
        sz_u64_t value64 = (sz_u64_t)(value) * 0x0101010101010101ull;
        while ((sz_size_t)target & 7ull) *(target++) = value;
        while (target + 8 <= end) *(sz_u64_t *)target = value64, target += 8;
        while (target != end) *(target++) = value;
    }
}

SZ_PUBLIC void sz_copy_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_MISALIGNED_LOADS
    while (length >= 8) *(sz_u64_t *)target = *(sz_u64_t *)source, target += 8, source += 8, length -= 8;
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
        while (length >= 8) *(sz_u64_t *)target = *(sz_u64_t *)source, target += 8, source += 8, length -= 8;
#endif
        while (length--) *(target++) = *(source++);
    }
    else {
        // Jump to the end and walk backwards.
        target += length, source += length;
#if SZ_USE_MISALIGNED_LOADS
        while (length >= 8) *(sz_u64_t *)(target -= 8) = *(sz_u64_t *)(source -= 8), length -= 8;
#endif
        while (length--) *(--target) = *(--source);
    }
}

#pragma endregion

/*
 *  @brief  Serial implementation for strings sequence processing.
 */
#pragma region Serial Implementation for Sequences

SZ_PUBLIC sz_size_t sz_partition(sz_sequence_t *sequence, sz_sequence_predicate_t predicate) {

    sz_size_t matches = 0;
    while (matches != sequence->count && predicate(sequence, sequence->order[matches])) ++matches;

    for (sz_size_t i = matches + 1; i < sequence->count; ++i)
        if (predicate(sequence, sequence->order[i]))
            sz_u64_swap(sequence->order + i, sequence->order + matches), ++matches;

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
        sz_u64_swap(order + root, order + child);
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
        sz_u64_swap(order + first, order + first + end);
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
            sz_u64_swap(&sequence->order[first], &sequence->order[first + 1]);
        return;
    case 3: {
        sz_u64_t a = sequence->order[first];
        sz_u64_t b = sequence->order[first + 1];
        sz_u64_t c = sequence->order[first + 2];
        if (less(sequence, b, a)) sz_u64_swap(&a, &b);
        if (less(sequence, c, b)) sz_u64_swap(&c, &b);
        if (less(sequence, b, a)) sz_u64_swap(&a, &b);
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
        sz_u64_swap(&sequence->order[first], &sequence->order[median]);
    if (less(sequence, sequence->order[last - 1], sequence->order[first]))
        sz_u64_swap(&sequence->order[first], &sequence->order[last - 1]);
    if (less(sequence, sequence->order[median], sequence->order[last - 1]))
        sz_u64_swap(&sequence->order[median], &sequence->order[last - 1]);

    // Partition using the median-of-three as the pivot
    sz_u64_t pivot = sequence->order[median];
    sz_size_t left = first;
    sz_size_t right = last - 1;
    while (1) {
        while (less(sequence, sequence->order[left], pivot)) left++;
        while (less(sequence, pivot, sequence->order[right])) right--;
        if (left >= right) break;
        sz_u64_swap(&sequence->order[left], &sequence->order[right]);
        left++;
        right--;
    }

    // Recursively sort the partitions
    _sz_introsort(sequence, less, first, left, depth);
    _sz_introsort(sequence, less, right + 1, last, depth);
}

SZ_PUBLIC void sz_sort_introsort(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    if (sequence->count == 0) return;
    sz_size_t size_is_not_power_of_two = (sequence->count & (sequence->count - 1)) != 0;
    sz_size_t depth_limit = sz_size_log2i_nonzero(sequence->count) + size_is_not_power_of_two;
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
            if (!(sequence->order[i] & mask)) sz_u64_swap(sequence->order + i, sequence->order + split), ++split;
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
 *  @brief  AVX2 implementation of the string search algorithms.
 *          Very minimalistic, but still faster than the serial implementation.
 */
#pragma region AVX2 Implementation

#if SZ_USE_X86_AVX2
#include <x86intrin.h>

/**
 *  @brief  Helper structure to simplify work with 256-bit registers.
 */
typedef union sz_u256_vec_t {
    __m256i ymm;
    __m128i xmms[2];
    sz_u64_t u64s[4];
    sz_u32_t u32s[8];
    sz_u16_t u16s[16];
    sz_u8_t u8s[32];
} sz_u256_vec_t;

SZ_PUBLIC void sz_fill_avx2(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    for (; length >= 32; target += 32, length -= 32) _mm256_storeu_si256((__m256i *)target, _mm256_set1_epi8(value));
    return sz_fill_serial(target, length, value);
}

SZ_PUBLIC void sz_copy_avx2(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    for (; length >= 32; target += 32, source += 32, length -= 32)
        _mm256_storeu_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
    return sz_copy_serial(target, source, length);
}

SZ_PUBLIC void sz_move_avx2(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target < source || target >= source + length) {
        for (; length >= 32; target += 32, source += 32, length -= 32)
            _mm256_storeu_si256((__m256i *)target, _mm256_lddqu_si256((__m256i const *)source));
        while (length--) *(target++) = *(source++);
    }
    else {
        // Jump to the end and walk backwards.
        for (target += length, source += length; length >= 32; length -= 32)
            _mm256_storeu_si256((__m256i *)(target -= 32), _mm256_lddqu_si256((__m256i const *)(source -= 32)));
        while (length--) *(--target) = *(--source);
    }
}

SZ_PUBLIC sz_cptr_t sz_find_byte_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    int mask;
    sz_u256_vec_t h_vec, n_vec;
    n_vec.ymm = _mm256_set1_epi8(n[0]);

    while (h_length >= 32) {
        h_vec.ymm = _mm256_lddqu_si256((__m256i const *)h);
        mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_vec.ymm, n_vec.ymm));
        if (mask) return h + sz_u32_ctz(mask);
        h += 32, h_length -= 32;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_last_byte_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    int mask;
    sz_u256_vec_t h_vec, n_vec;
    n_vec.ymm = _mm256_set1_epi8(n[0]);

    while (h_length >= 32) {
        h_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + h_length - 32));
        mask = _mm256_cmpeq_epi8_mask(h_vec.ymm, n_vec.ymm);
        if (mask) return h + h_length - 1 - sz_u32_clz(mask);
        h_length -= 32;
    }

    return sz_find_last_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    if (n_length == 1) return sz_find_byte_avx2(h, h_length, n);

    int matches;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(n[0]);
    n_mid_vec.ymm = _mm256_set1_epi8(n[n_length / 2]);
    n_last_vec.ymm = _mm256_set1_epi8(n[n_length - 1]);

    for (; h_length >= n_length + 32; h += 32, h_length -= 32) {
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + n_length / 2));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + n_length - 1));
        matches = _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
                  _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
                  _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches) {
            int potential_offset = sz_u32_ctz(matches);
            if (sz_equal(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;
            matches &= matches - 1;
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_last_avx2(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    if (n_length == 1) return sz_find_last_byte_avx2(h, h_length, n);

    int matches;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(n[0]);
    n_mid_vec.ymm = _mm256_set1_epi8(n[n_length / 2]);
    n_last_vec.ymm = _mm256_set1_epi8(n[n_length - 1]);

    for (; h_length >= n_length + 32; h_length -= 32) {
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + h_length - n_length - 32 + 1));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + h_length - n_length - 32 + 1 + n_length / 2));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + h_length - 32));
        matches = _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
                  _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
                  _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches) {
            int potential_offset = sz_u32_clz(matches);
            if (sz_equal(h + h_length - n_length - potential_offset + 1, n + 1, n_length - 2))
                return h + h_length - n_length - potential_offset;
            matches &= ~(1 << (31 - potential_offset));
        }
    }

    return sz_find_last_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_from_set_avx2(sz_cptr_t h, sz_size_t h_length, sz_u8_set_t const *set) {
    return sz_find_from_set_serial(h, h_length, set);
}

SZ_PUBLIC sz_cptr_t sz_find_last_from_set_avx2(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set) {
    return sz_find_last_from_set_serial(text, length, set);
}

#endif
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
 *  @brief  Helper structure to simplify work with 512-bit registers.
 */
typedef union sz_u512_vec_t {
    __m512i zmm;
    __m256i ymms[2];
    __m128i xmms[4];
    sz_u64_t u64s[8];
    sz_u32_t u32s[16];
    sz_u16_t u16s[32];
    sz_u8_t u8s[64];
} sz_u512_vec_t;

SZ_INTERNAL __mmask64 sz_u64_clamp_mask_until(sz_size_t n) {
    // The simplest approach to compute this if we know that `n` is blow or equal 64:
    //      return (1ull << n) - 1;
    // A slightly more complex approach, if we don't know that `n` is under 64:
    return _bzhi_u64(0xFFFFFFFFFFFFFFFF, n < 64 ? n : 64);
}

SZ_INTERNAL __mmask64 sz_u64_mask_until(sz_size_t n) {
    // The simplest approach to compute this if we know that `n` is blow or equal 64:
    //      return (1ull << n) - 1;
    // A slightly more complex approach, if we don't know that `n` is under 64:
    return _bzhi_u64(0xFFFFFFFFFFFFFFFF, n);
}

/**
 *  @brief  Variation of AVX-512 relative order check for different length strings.
 */
SZ_PUBLIC sz_ordering_t sz_order_avx512(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
    sz_u512_vec_t a_vec, b_vec;
    __mmask64 a_mask, b_mask, mask_not_equal;

    // The rare case, when both string are very long.
    while ((a_length >= 64) & (b_length >= 64)) {
        a_vec.zmm = _mm512_loadu_epi8(a);
        b_vec.zmm = _mm512_loadu_epi8(b);
        mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        if (mask_not_equal != 0) {
            int first_diff = _tzcnt_u64(mask_not_equal);
            char a_char = a[first_diff];
            char b_char = b[first_diff];
            return ordering_lookup[a_char < b_char];
        }
        a += 64, b += 64, a_length -= 64, b_length -= 64;
    }

    // In most common scenarios at least one of the strings is under 64 bytes.
    if (a_length | b_length) {
        a_mask = sz_u64_clamp_mask_until(a_length);
        b_mask = sz_u64_clamp_mask_until(b_length);
        a_vec.zmm = _mm512_maskz_loadu_epi8(a_mask, a);
        b_vec.zmm = _mm512_maskz_loadu_epi8(b_mask, b);
        // The AVX-512 `_mm512_mask_cmpneq_epi8_mask` intrinsics are generally handy in such environments.
        // They, however, have latency 3 on most modern CPUs. Using AVX2: `_mm256_cmpeq_epi8` would have
        // been cheaper, if we didn't have to apply `_mm256_movemask_epi8` afterwards.
        mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
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
    else
        return sz_equal_k;
}

/**
 *  @brief  Variation of AVX-512 equality check between equivalent length strings.
 */
SZ_PUBLIC sz_bool_t sz_equal_avx512(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    __mmask64 mask;
    sz_u512_vec_t a_vec, b_vec;

    while (length >= 64) {
        a_vec.zmm = _mm512_loadu_epi8(a);
        b_vec.zmm = _mm512_loadu_epi8(b);
        mask = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        if (mask != 0) return sz_false_k;
        a += 64, b += 64, length -= 64;
    }

    if (length) {
        mask = sz_u64_mask_until(length);
        a_vec.zmm = _mm512_maskz_loadu_epi8(mask, a);
        b_vec.zmm = _mm512_maskz_loadu_epi8(mask, b);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpneq_epi8_mask(mask, a_vec.zmm, b_vec.zmm);
        return (sz_bool_t)(mask == 0);
    }
    else
        return sz_true_k;
}

SZ_PUBLIC void sz_fill_avx512(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    for (; length >= 64; target += 64, length -= 64) _mm512_storeu_epi8(target, _mm512_set1_epi8(value));
    // At this point the length is guaranteed to be under 64.
    _mm512_mask_storeu_epi8(target, sz_u64_mask_until(length), _mm512_set1_epi8(value));
}

SZ_PUBLIC void sz_copy_avx512(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    for (; length >= 64; target += 64, source += 64, length -= 64)
        _mm512_storeu_epi8(target, _mm512_loadu_epi8(source));
    // At this point the length is guaranteed to be under 64.
    __mmask64 mask = sz_u64_mask_until(length);
    _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
}

SZ_PUBLIC void sz_move_avx512(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target < source || target >= source + length) {
        for (; length >= 64; target += 64, source += 64, length -= 64)
            _mm512_storeu_epi8(target, _mm512_loadu_epi8(source));
        // At this point the length is guaranteed to be under 64.
        __mmask64 mask = sz_u64_mask_until(length);
        _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    }
    else {
        // Jump to the end and walk backwards.
        for (target += length, source += length; length >= 64; length -= 64)
            _mm512_storeu_epi8(target -= 64, _mm512_loadu_epi8(source -= 64));
        // At this point the length is guaranteed to be under 64.
        __mmask64 mask = sz_u64_mask_until(length);
        _mm512_mask_storeu_epi8(target - length, mask, _mm512_maskz_loadu_epi8(mask, source - length));
    }
}

SZ_PUBLIC sz_cptr_t sz_find_byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __mmask64 mask;
    sz_u512_vec_t h_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi8(n[0]);

    while (h_length >= 64) {
        h_vec.zmm = _mm512_loadu_epi8(h);
        mask = _mm512_cmpeq_epi8_mask(h_vec.zmm, n_vec.zmm);
        if (mask) return h + sz_u64_ctz(mask);
        h += 64, h_length -= 64;
    }

    if (h_length) {
        mask = sz_u64_mask_until(h_length);
        h_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec.zmm, n_vec.zmm);
        if (mask) return h + sz_u64_ctz(mask);
    }

    return NULL;
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns up to 2 bytes included.
 */
SZ_INTERNAL sz_cptr_t sz_find_2byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    // A simpler approach would ahve been to use two separate registers for
    // different characters of the needle, but that would use more registers.
    __mmask64 mask;
    __mmask32 matches0, matches1;
    sz_u512_vec_t h0_vec, h1_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi16(sz_u16_load(n).u16);

    while (h_length >= 65) {
        h0_vec.zmm = _mm512_loadu_epi8(h);
        h1_vec.zmm = _mm512_loadu_epi8(h + 1);
        matches0 = _mm512_cmpeq_epi16_mask(h0_vec.zmm, n_vec.zmm);
        matches1 = _mm512_cmpeq_epi16_mask(h1_vec.zmm, n_vec.zmm);
        // https://lemire.me/blog/2018/01/08/how-fast-can-you-bit-interleave-32-bit-integers/
        if (matches0 | matches1)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x5555555555555555ull) | //
                                  _pdep_u64(matches1, 0xAAAAAAAAAAAAAAAAull));
        h += 64, h_length -= 64;
    }

    if (h_length >= 2) {
        mask = sz_u64_mask_until(h_length);
        h0_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h1_vec.zmm = _mm512_maskz_loadu_epi8(mask >> 1, h + 1);
        matches0 = _mm512_mask_cmpeq_epi16_mask(mask, h0_vec.zmm, n_vec.zmm);
        matches1 = _mm512_mask_cmpeq_epi16_mask(mask, h1_vec.zmm, n_vec.zmm);
        if (matches0 | matches1)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x5555555555555555ull) | //
                                  _pdep_u64(matches1, 0xAAAAAAAAAAAAAAAAull));
    }

    return NULL;
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns up to 4 bytes included.
 */
SZ_INTERNAL sz_cptr_t sz_find_4byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    __mmask64 mask;
    __mmask16 matches0, matches1, matches2, matches3;
    sz_u512_vec_t h0_vec, h1_vec, h2_vec, h3_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi32(sz_u32_load(n).u32);

    while (h_length >= 64) {
        h0_vec.zmm = _mm512_loadu_epi8(h + 0);
        h1_vec.zmm = _mm512_loadu_epi8(h + 1);
        h2_vec.zmm = _mm512_loadu_epi8(h + 2);
        h3_vec.zmm = _mm512_loadu_epi8(h + 3);
        matches0 = _mm512_cmpeq_epi32_mask(h0_vec.zmm, n_vec.zmm);
        matches1 = _mm512_cmpeq_epi32_mask(h1_vec.zmm, n_vec.zmm);
        matches2 = _mm512_cmpeq_epi32_mask(h2_vec.zmm, n_vec.zmm);
        matches3 = _mm512_cmpeq_epi32_mask(h3_vec.zmm, n_vec.zmm);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x1111111111111111) | //
                                  _pdep_u64(matches1, 0x2222222222222222) | //
                                  _pdep_u64(matches2, 0x4444444444444444) | //
                                  _pdep_u64(matches3, 0x8888888888888888));
        h += 64, h_length -= 64;
    }

    if (h_length >= 4) {
        mask = sz_u64_mask_until(h_length);
        h0_vec.zmm = _mm512_maskz_loadu_epi8(mask >> 0, h + 0);
        h1_vec.zmm = _mm512_maskz_loadu_epi8(mask >> 1, h + 1);
        h2_vec.zmm = _mm512_maskz_loadu_epi8(mask >> 2, h + 2);
        h3_vec.zmm = _mm512_maskz_loadu_epi8(mask >> 3, h + 3);
        matches0 = _mm512_mask_cmpeq_epi32_mask(mask, h0_vec.zmm, n_vec.zmm);
        matches1 = _mm512_mask_cmpeq_epi32_mask(mask, h1_vec.zmm, n_vec.zmm);
        matches2 = _mm512_mask_cmpeq_epi32_mask(mask, h2_vec.zmm, n_vec.zmm);
        matches3 = _mm512_mask_cmpeq_epi32_mask(mask, h3_vec.zmm, n_vec.zmm);
        if (matches0 | matches1 | matches2 | matches3)
            return h + sz_u64_ctz(_pdep_u64(matches0, 0x1111111111111111ull) | //
                                  _pdep_u64(matches1, 0x2222222222222222ull) | //
                                  _pdep_u64(matches2, 0x4444444444444444ull) | //
                                  _pdep_u64(matches3, 0x8888888888888888ull));
    }

    return NULL;
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns up to 66 bytes included.
 */
SZ_INTERNAL sz_cptr_t sz_find_under66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    __mmask64 matches;
    __mmask64 mask, n_length_body_mask = sz_u64_mask_until(n_length - 2);
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, h_body_vec, n_first_vec, n_mid_vec, n_last_vec, n_body_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_mid_vec.zmm = _mm512_set1_epi8(n[n_length / 2]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);
    n_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, n + 1);

    while (h_length >= n_length + 64) {
        h_first_vec.zmm = _mm512_loadu_epi8(h);
        h_mid_vec.zmm = _mm512_loadu_epi8(h + n_length / 2);
        h_last_vec.zmm = _mm512_loadu_epi8(h + n_length - 1);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            h_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, h + potential_offset + 1);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm)) return h + potential_offset;
            h += potential_offset + 1, h_length -= potential_offset + 1;
        }
        else { h += 64, h_length -= 64; }
    }

    while (h_length >= n_length) {
        mask = sz_u64_mask_until(h_length - n_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length / 2);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_mid_vec.zmm, n_mid_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            h_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, h + potential_offset + 1);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm)) return h + potential_offset;
            h += potential_offset + 1, h_length -= potential_offset + 1;
        }
        else { break; }
    }

    return NULL;
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns longer than 66 bytes.
 */
SZ_INTERNAL sz_cptr_t sz_find_over66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    __mmask64 mask;
    __mmask64 matches;
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_mid_vec.zmm = _mm512_set1_epi8(n[n_length / 2]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);

    while (h_length >= n_length + 64) {
        h_first_vec.zmm = _mm512_loadu_epi8(h);
        h_mid_vec.zmm = _mm512_loadu_epi8(h + n_length / 2);
        h_last_vec.zmm = _mm512_loadu_epi8(h + n_length - 1);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (sz_equal_avx512(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;
            h += potential_offset + 1, h_length -= potential_offset + 1;
        }
        else { h += 64, h_length -= 64; }
    }

    while (h_length >= n_length) {
        mask = sz_u64_mask_until(h_length - n_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length / 2);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_mid_vec.zmm, n_mid_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (sz_equal_avx512(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;
            h += potential_offset + 1, h_length -= potential_offset + 1;
        }
        else { break; }
    }

    return NULL;
}

SZ_PUBLIC sz_cptr_t sz_find_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return NULL;

    sz_find_t backends[] = {
        // For very short strings brute-force SWAR makes sense.
        (sz_find_t)sz_find_byte_avx512,
        (sz_find_t)sz_find_2byte_avx512,
        (sz_find_t)sz_find_under66byte_avx512,
        (sz_find_t)sz_find_4byte_avx512,
        // For longer needles we use a Two-Way heuristic with a follow-up check in-between.
        (sz_find_t)sz_find_under66byte_avx512,
        (sz_find_t)sz_find_over66byte_avx512,
    };

    return backends[
        // For very short strings brute-force SWAR makes sense.
        (n_length > 1) + (n_length > 2) + (n_length > 3) +
        // For longer needles we use a Two-Way heuristic with a follow-up check in-between.
        (n_length > 4) + (n_length > 66)](h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_last_byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __mmask64 mask;
    sz_u512_vec_t h_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi8(n[0]);

    while (h_length >= 64) {
        h_vec.zmm = _mm512_loadu_epi8(h + h_length - 64);
        mask = _mm512_cmpeq_epi8_mask(h_vec.zmm, n_vec.zmm);
        int potential_offset = sz_u64_clz(mask);
        if (mask) return h + h_length - 1 - potential_offset;
        h_length -= 64;
    }

    if (h_length) {
        mask = sz_u64_mask_until(h_length);
        h_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec.zmm, n_vec.zmm);
        int potential_offset = sz_u64_clz(mask);
        if (mask) return h + 64 - potential_offset - 1;
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
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, h_body_vec, n_first_vec, n_mid_vec, n_last_vec, n_body_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_mid_vec.zmm = _mm512_set1_epi8(n[n_length / 2]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);
    n_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, n + 1);

    while (h_length >= n_length + 64) {

        h_first_vec.zmm = _mm512_loadu_epi8(h + h_length - n_length - 64 + 1);
        h_mid_vec.zmm = _mm512_loadu_epi8(h + h_length - n_length - 64 + 1 + n_length / 2);
        h_last_vec.zmm = _mm512_loadu_epi8(h + h_length - 64);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_clz(matches);
            h_body_vec.zmm =
                _mm512_maskz_loadu_epi8(n_length_body_mask, h + h_length - n_length - potential_offset + 1);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm))
                return h + h_length - n_length - potential_offset;
            h_length -= potential_offset + 1;
        }
        else { h_length -= 64; }
    }

    while (h_length >= n_length) {
        mask = sz_u64_mask_until(h_length - n_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length / 2);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_mid_vec.zmm, n_mid_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_clz(matches);
            h_body_vec.zmm = _mm512_maskz_loadu_epi8(n_length_body_mask, h + 64 - potential_offset);
            if (!_mm512_cmpneq_epi8_mask(h_body_vec.zmm, n_body_vec.zmm)) return h + 64 - potential_offset - 1;
            h_length = 64 - potential_offset - 1;
        }
        else { break; }
    }

    return NULL;
}

/**
 *  @brief  Variation of AVX-512 exact search for patterns longer than 66 bytes.
 */
SZ_INTERNAL sz_cptr_t sz_find_last_over66byte_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    __mmask64 mask;
    __mmask64 matches;
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[0]);
    n_mid_vec.zmm = _mm512_set1_epi8(n[n_length / 2]);
    n_last_vec.zmm = _mm512_set1_epi8(n[n_length - 1]);

    while (h_length >= n_length + 64) {
        h_first_vec.zmm = _mm512_loadu_epi8(h + h_length - n_length - 64 + 1);
        h_mid_vec.zmm = _mm512_loadu_epi8(h + h_length - n_length - 64 + 1 + n_length / 2);
        h_last_vec.zmm = _mm512_loadu_epi8(h + h_length - 64);
        matches = _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm) &
                  _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_clz(matches);
            if (sz_equal_avx512(h + h_length - n_length - potential_offset + 1, n + 1, n_length - 2))
                return h + h_length - n_length - potential_offset;
            h_length -= potential_offset + 1;
        }
        else { h_length -= 64; }
    }

    while (h_length >= n_length) {
        mask = sz_u64_mask_until(h_length - n_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length / 2);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + n_length - 1);
        matches = _mm512_mask_cmpeq_epi8_mask(mask, h_first_vec.zmm, n_first_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_mid_vec.zmm, n_mid_vec.zmm) &
                  _mm512_mask_cmpeq_epi8_mask(mask, h_last_vec.zmm, n_last_vec.zmm);
        if (matches) {
            int potential_offset = sz_u64_clz(matches);
            if (sz_equal_avx512(h + 64 - potential_offset, n + 1, n_length - 2)) return h + 64 - potential_offset - 1;
            h_length = 64 - potential_offset - 1;
        }
        else { break; };
    }

    return NULL;
}

SZ_PUBLIC sz_cptr_t sz_find_last_avx512(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return NULL;

    sz_find_t backends[] = {
        // For very short strings brute-force SWAR makes sense.
        (sz_find_t)sz_find_last_byte_avx512,
        // For longer needles we use a Two-Way heuristic with a follow-up check in-between.
        (sz_find_t)sz_find_last_under66byte_avx512,
        (sz_find_t)sz_find_last_over66byte_avx512,
    };

    return backends[
        // For very short strings brute-force SWAR makes sense.
        0 +
        // For longer needles we use a Two-Way heuristic with a follow-up check in-between.
        (n_length > 1) + (n_length > 66)](h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_from_set_avx512(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *filter) {

    sz_size_t load_length;
    __mmask32 load_mask, matches_mask;
    // To store the set in the register we need just 256 bits, but the `VPERMB` instruction
    // we are going to invoke is surprisingly cheaper on ZMM registers.
    sz_u512_vec_t text_vec, filter_vec;
    filter_vec.ymms[0] = _mm256_loadu_epi64(&filter->_u64s[0]);

    // We are going to view the `filter` at 8-bit word granularity.
    sz_u512_vec_t filter_slice_offsets_vec;
    sz_u512_vec_t filter_slice_vec;
    sz_u512_vec_t offset_within_slice_vec;
    sz_u512_vec_t mask_in_filter_slice_vec;
    sz_u512_vec_t matches_vec;

    while (length) {
        // For every byte:
        // 1. Find corresponding word in a set.
        // 2. Produce a bitmask to check against that word.
        load_length = sz_min_of_two(length, 32);
        load_mask = sz_u64_mask_until(load_length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(load_mask, text);

        // To shift right every byte by 3 bits we can use the GF2 affine transformations.
        // https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/
        // After next line, all 8-bit offsets in the `filter_slice_offsets_vec` should be under 32.
        filter_slice_offsets_vec.ymms[0] =
            _mm256_gf2p8affine_epi64_epi8(text_vec.ymms[0], _mm256_set1_epi64x(0x0102040810204080ull << (3 * 8)), 0);

        // After next line, `filter_slice_vec` will contain the right word from the set,
        // needed to filter the presence of the byte in the set.
        filter_slice_vec.ymms[0] = _mm256_permutexvar_epi8(filter_slice_offsets_vec.ymms[0], filter_vec.ymms[0]);

        // After next line, all 8-bit offsets in the `filter_slice_offsets_vec` should be under 8.
        offset_within_slice_vec.ymms[0] = _mm256_and_si256(text_vec.ymms[0], _mm256_set1_epi64x(0x0707070707070707ull));

        // Instead of performing one more Galois Field operation, we can upcast to 16-bit integers,
        // and perform the fift and intersection there.
        filter_slice_vec.zmm = _mm512_cvtepi8_epi16(filter_slice_vec.ymms[0]);
        offset_within_slice_vec.zmm = _mm512_cvtepi8_epi16(offset_within_slice_vec.ymms[0]);
        mask_in_filter_slice_vec.zmm = _mm512_sllv_epi16(_mm512_set1_epi16(1), offset_within_slice_vec.zmm);
        matches_vec.zmm = _mm512_and_si512(filter_slice_vec.zmm, mask_in_filter_slice_vec.zmm);

        matches_mask = _mm512_mask_cmpneq_epi16_mask(load_mask, matches_vec.zmm, _mm512_setzero_si512());
        if (matches_mask) {
            int offset = sz_u32_ctz(matches_mask);
            return text + offset;
        }
        else { text += load_length, length -= load_length; }
    }

    return NULL;
}

SZ_PUBLIC sz_cptr_t sz_find_last_from_set_avx512(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *filter) {

    sz_size_t load_length;
    __mmask32 load_mask, matches_mask;
    // To store the set in the register we need just 256 bits, but the `VPERMB` instruction
    // we are going to invoke is surprisingly cheaper on ZMM registers.
    sz_u512_vec_t text_vec, filter_vec;
    filter_vec.ymms[0] = _mm256_loadu_epi64(&filter->_u64s[0]);

    // We are going to view the `filter` at 8-bit word granularity.
    sz_u512_vec_t filter_slice_offsets_vec;
    sz_u512_vec_t filter_slice_vec;
    sz_u512_vec_t offset_within_slice_vec;
    sz_u512_vec_t mask_in_filter_slice_vec;
    sz_u512_vec_t matches_vec;

    while (length) {
        // For every byte:
        // 1. Find corresponding word in a set.
        // 2. Produce a bitmask to check against that word.
        load_length = sz_min_of_two(length, 32);
        load_mask = sz_u64_mask_until(load_length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(load_mask, text + length - load_length);

        // To shift right every byte by 3 bits we can use the GF2 affine transformations.
        // https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/
        // After next line, all 8-bit offsets in the `filter_slice_offsets_vec` should be under 32.
        filter_slice_offsets_vec.ymms[0] =
            _mm256_gf2p8affine_epi64_epi8(text_vec.ymms[0], _mm256_set1_epi64x(0x0102040810204080ull << (3 * 8)), 0);

        // After next line, `filter_slice_vec` will contain the right word from the set,
        // needed to filter the presence of the byte in the set.
        filter_slice_vec.ymms[0] = _mm256_permutexvar_epi8(filter_slice_offsets_vec.ymms[0], filter_vec.ymms[0]);

        // After next line, all 8-bit offsets in the `filter_slice_offsets_vec` should be under 8.
        offset_within_slice_vec.ymms[0] = _mm256_and_si256(text_vec.ymms[0], _mm256_set1_epi64x(0x0707070707070707ull));

        // Instead of performing one more Galois Field operation, we can upcast to 16-bit integers,
        // and perform the fift and intersection there.
        filter_slice_vec.zmm = _mm512_cvtepi8_epi16(filter_slice_vec.ymms[0]);
        offset_within_slice_vec.zmm = _mm512_cvtepi8_epi16(offset_within_slice_vec.ymms[0]);
        mask_in_filter_slice_vec.zmm = _mm512_sllv_epi16(_mm512_set1_epi16(1), offset_within_slice_vec.zmm);
        matches_vec.zmm = _mm512_and_si512(filter_slice_vec.zmm, mask_in_filter_slice_vec.zmm);

        matches_mask = _mm512_mask_cmpneq_epi16_mask(load_mask, matches_vec.zmm, _mm512_setzero_si512());
        if (matches_mask) {
            int offset = sz_u32_clz(matches_mask);
            return text + length - load_length + 32 - offset - 1;
        }
        else { length -= load_length; }
    }

    return NULL;
}

#if 0
SZ_PUBLIC sz_size_t sz_edit_distance_avx512(     //
    sz_cptr_t const a, sz_size_t const a_length, //
    sz_cptr_t const b, sz_size_t const b_length, //
    sz_size_t const bound, sz_memory_allocator_t const *alloc) {

    sz_u512_vec_t a_vec, b_vec, previous_vec, current_vec, permutation_vec;
    sz_u512_vec_t cost_deletion_vec, cost_insertion_vec, cost_substitution_vec;
    sz_size_t min_distance;

    b_vec.zmm = _mm512_maskz_loadu_epi8(sz_u64_mask_until(b_length), b);
    previous_vec.zmm = _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
                                       47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
                                       31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
                                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    // Shifting bytes across the whole ZMM register is quite complicated, so let's use a permutation for that.
    permutation_vec.zmm = _mm512_set_epi8(62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, //
                                          46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, //
                                          30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, //
                                          14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 63);

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        min_distance = bound - 1;

        a_vec.zmm = _mm512_set1_epi8(a[idx_a]);
        // We first start by computing the cost of deletions and substitutions
        // for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
        //     sz_u8_t cost_deletion = previous_vec.u8s[idx_b + 1] + 1;
        //     sz_u8_t cost_substitution = previous_vec.u8s[idx_b] + (a[idx_a] != b[idx_b]);
        //     current_vec.u8s[idx_b + 1] = sz_min_of_two(cost_deletion, cost_substitution);
        // }
        cost_deletion_vec.zmm = _mm512_add_epi8(previous_vec.zmm, _mm512_set1_epi8(1));
        cost_substitution_vec.zmm =
            _mm512_mask_set1_epi8(_mm512_setzero_si512(), _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm), 0x01);
        cost_substitution_vec.zmm = _mm512_add_epi8(previous_vec.zmm, cost_substitution_vec.zmm);
        cost_substitution_vec.zmm = _mm512_permutexvar_epi8(permutation_vec.zmm, cost_substitution_vec.zmm);
        current_vec.zmm = _mm512_min_epu8(cost_deletion_vec.zmm, cost_substitution_vec.zmm);
        current_vec.u8s[0] = idx_a + 1;

        // Now we need to compute the inclusive prefix sums using the minimum operator
        // In one line:
        //      current_vec.u8s[idx_b + 1] = sz_min_of_two(current_vec.u8s[idx_b + 1], current_vec.u8s[idx_b] + 1)
        //
        // Unrolling this:
        //      current_vec.u8s[0 + 1] = sz_min_of_two(current_vec.u8s[0 + 1], current_vec.u8s[0] + 1)
        //      current_vec.u8s[1 + 1] = sz_min_of_two(current_vec.u8s[1 + 1], current_vec.u8s[1] + 1)
        //      current_vec.u8s[2 + 1] = sz_min_of_two(current_vec.u8s[2 + 1], current_vec.u8s[2] + 1)
        //      current_vec.u8s[3 + 1] = sz_min_of_two(current_vec.u8s[3 + 1], current_vec.u8s[3] + 1)
        //
        // Alternatively, using a tree-like reduction in log2 steps:
        //      - 6 cycles of reductions shifting by 1, 2, 4, 8, 16, 32, 64 bytes;
        //      - with each cycle containing at least one shift, min, add, blend.
        //
        // Which adds meaningless complexity without any performance gains.
        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            sz_u8_t cost_insertion = current_vec.u8s[idx_b] + 1;
            current_vec.u8s[idx_b + 1] = sz_min_of_two(current_vec.u8s[idx_b + 1], cost_insertion);
        }

        // Swap previous_distances and current_distances pointers
        sz_u512_vec_t temp_vec;
        temp_vec.zmm = previous_vec.zmm;
        previous_vec.zmm = current_vec.zmm;
        current_vec.zmm = temp_vec.zmm;
    }

    return previous_vec.u8s[b_length] < bound ? previous_vec.u8s[b_length] : bound;
}
#endif

#endif

#pragma endregion

/*  @brief  Implementation of the string search algorithms using the Arm NEON instruction set, available on 64-bit
 *          Arm processors. Implements: {substring search, character search, character set search} x {forward, reverse}.
 */
#pragma region ARM NEON

#if SZ_USE_ARM_NEON
#include <arm_neon.h>

/**
 *  @brief  Helper structure to simplify work with 64-bit words.
 */
typedef union sz_u128_vec_t {
    uint8x16_t u8x16;
    uint32x4_t u32x4;
    sz_u64_t u64s[2];
    sz_u32_t u32s[4];
    sz_u16_t u16s[8];
    sz_u8_t u8s[16];
} sz_u128_vec_t;

SZ_PUBLIC sz_cptr_t sz_find_byte_neon(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u8_t offsets[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    sz_u128_vec_t h_vec, n_vec, offsets_vec, matches_vec;
    n_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)n);
    offsets_vec.u8x16 = vld1q_u8(offsets);

    while (h_length >= 16) {
        h_vec.u8x16 = vld1q_u8((sz_u8_t const *)h);
        matches_vec.u8x16 = vceqq_u8(h_vec.u8x16, n_vec.u8x16);
        // In Arm NEON we don't have a `movemask` to combine it with `ctz` and get the offset of the match.
        // But assuming the `vmaxvq` is cheap, we can use it to find the first match, by blending (bitwise selecting)
        // the vector with a relative offsets array.
        if (vmaxvq_u8(matches_vec.u8x16)) {
            matches_vec.u8x16 = vbslq_u8(matches_vec.u8x16, offsets_vec.u8x16, vdupq_n_u8(0xFF));
            return h + vminvq_u8(matches_vec.u8x16);
        }
        h += 16, h_length -= 16;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_last_byte_neon(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u8_t offsets[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    sz_u128_vec_t h_vec, n_vec, offsets_vec, matches_vec;
    n_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)n);
    offsets_vec.u8x16 = vld1q_u8(offsets);

    while (h_length >= 16) {
        h_vec.u8x16 = vld1q_u8((sz_u8_t const *)h + h_length - 16);
        matches_vec.u8x16 = vceqq_u8(h_vec.u8x16, n_vec.u8x16);
        // In Arm NEON we don't have a `movemask` to combine it with `clz` and get the offset of the match.
        // But assuming the `vmaxvq` is cheap, we can use it to find the first match, by blending (bitwise selecting)
        // the vector with a relative offsets array.
        if (vmaxvq_u8(matches_vec.u8x16)) {
            matches_vec.u8x16 = vbslq_u8(matches_vec.u8x16, offsets_vec.u8x16, vdupq_n_u8(0));
            return h + h_length - 16 + vmaxvq_u8(matches_vec.u8x16);
        }
        h_length -= 16;
    }

    return sz_find_last_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_neon(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    if (n_length == 1) return sz_find_byte_neon(h, h_length, n);

    // Will contain 4 bits per character.
    sz_u64_t matches;
    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
    n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[0]);
    n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[n_length / 2]);
    n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[n_length - 1]);

    for (; h_length >= n_length + 16; h += 16, h_length -= 16) {
        h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h));
        h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + n_length / 2));
        h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + n_length - 1));
        matches_vec.u8x16 = vandq_u8(                           //
            vandq_u8(                                           //
                vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
            vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
        if (vmaxvq_u8(matches_vec.u8x16)) {
            // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
            // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
            matches = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(matches_vec.u8x16), 4)), 0) &
                      0x8888888888888888ull;
            while (matches) {
                int potential_offset = sz_u64_ctz(matches) / 4;
                if (sz_equal(h + potential_offset + 1, n + 1, n_length - 2)) return h + potential_offset;
                matches &= matches - 1;
            }
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_last_neon(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    if (n_length == 1) return sz_find_last_byte_neon(h, h_length, n);

    // Will contain 4 bits per character.
    sz_u64_t matches;
    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
    n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[0]);
    n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[n_length / 2]);
    n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[n_length - 1]);

    for (; h_length >= n_length + 16; h_length -= 16) {
        h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + h_length - n_length - 16 + 1));
        h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + h_length - n_length - 16 + 1 + n_length / 2));
        h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + h_length - 16));
        matches_vec.u8x16 = vandq_u8(                           //
            vandq_u8(                                           //
                vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
            vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
        if (vmaxvq_u8(matches_vec.u8x16)) {
            // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
            // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
            matches = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(matches_vec.u8x16), 4)), 0) &
                      0x8888888888888888ull;
            while (matches) {
                int potential_offset = sz_u64_clz(matches) / 4;
                if (sz_equal(h + h_length - n_length - potential_offset + 1, n + 1, n_length - 2))
                    return h + h_length - n_length - potential_offset;
                sz_assert((matches & (1ull << (63 - potential_offset * 4))) != 0 &&
                          "The bit must be set before we squash it");
                matches &= ~(1ull << (63 - potential_offset * 4));
            }
        }
    }

    return sz_find_last_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_from_set_neon(sz_cptr_t h, sz_size_t h_length, sz_u8_set_t const *set) {
    return sz_find_from_set_serial(h, h_length, set);
}

SZ_PUBLIC sz_cptr_t sz_find_last_from_set_neon(sz_cptr_t h, sz_size_t h_length, sz_u8_set_t const *set) {
    return sz_find_last_from_set_serial(h, h_length, set);
}

#endif // Arm Neon

#pragma endregion

/*
 *  @brief  Pick the right implementation for the string search algorithms.
 */
#pragma region Compile-Time Dispatching

SZ_PUBLIC sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length) { return sz_hash_serial(text, length); }

SZ_PUBLIC sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
#if SZ_USE_X86_AVX512
    return sz_equal_avx512(a, b, length);
#else
    return sz_equal_serial(a, b, length);
#endif
}

SZ_PUBLIC sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
#if SZ_USE_X86_AVX512
    return sz_order_avx512(a, a_length, b, b_length);
#else
    return sz_order_serial(a, a_length, b, b_length);
#endif
}

SZ_PUBLIC void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_X86_AVX512
    sz_copy_avx512(target, source, length);
#elif SZ_USE_X86_AVX2
    sz_copy_avx2(target, source, length);
#else
    sz_copy_serial(target, source, length);
#endif
}

SZ_PUBLIC void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if SZ_USE_X86_AVX512
    sz_move_avx512(target, source, length);
#elif SZ_USE_X86_AVX2
    sz_move_avx2(target, source, length);
#else
    sz_move_serial(target, source, length);
#endif
}

SZ_PUBLIC void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
#if SZ_USE_X86_AVX512
    sz_fill_avx512(target, length, value);
#elif SZ_USE_X86_AVX2
    sz_fill_avx2(target, length, value);
#else
    sz_fill_serial(target, length, value);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#if SZ_USE_X86_AVX512
    return sz_find_byte_avx512(haystack, h_length, needle);
#elif SZ_USE_X86_AVX2
    return sz_find_byte_avx2(haystack, h_length, needle);
#elif SZ_USE_ARM_NEON
    return sz_find_byte_neon(haystack, h_length, needle);
#else
    return sz_find_byte_serial(haystack, h_length, needle);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_last_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#if SZ_USE_X86_AVX512
    return sz_find_last_byte_avx512(haystack, h_length, needle);
#elif SZ_USE_X86_AVX2
    return sz_find_last_byte_avx2(haystack, h_length, needle);
#elif SZ_USE_ARM_NEON
    return sz_find_last_byte_neon(haystack, h_length, needle);
#else
    return sz_find_last_byte_serial(haystack, h_length, needle);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
#if SZ_USE_X86_AVX512
    return sz_find_avx512(haystack, h_length, needle, n_length);
#elif SZ_USE_X86_AVX2
    return sz_find_avx2(haystack, h_length, needle, n_length);
#elif SZ_USE_ARM_NEON
    return sz_find_neon(haystack, h_length, needle, n_length);
#else
    return sz_find_serial(haystack, h_length, needle, n_length);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_last(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
#if SZ_USE_X86_AVX512
    return sz_find_last_avx512(haystack, h_length, needle, n_length);
#elif SZ_USE_X86_AVX2
    return sz_find_last_avx2(haystack, h_length, needle, n_length);
#elif SZ_USE_ARM_NEON
    return sz_find_last_neon(haystack, h_length, needle, n_length);
#else
    return sz_find_last_serial(haystack, h_length, needle, n_length);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_from_set(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set) {
#if SZ_USE_X86_AVX512
    return sz_find_from_set_avx512(text, length, set);
#elif SZ_USE_X86_AVX2
    return sz_find_from_set_avx2(text, length, set);
#elif SZ_USE_ARM_NEON
    return sz_find_from_set_neon(text, length, set);
#else
    return sz_find_from_set_serial(text, length, set);
#endif
}

SZ_PUBLIC sz_cptr_t sz_find_last_from_set(sz_cptr_t text, sz_size_t length, sz_u8_set_t const *set) {
#if SZ_USE_X86_AVX512
    return sz_find_last_from_set_avx512(text, length, set);
#elif SZ_USE_X86_AVX2
    return sz_find_last_from_set_avx2(text, length, set);
#elif SZ_USE_ARM_NEON
    return sz_find_last_from_set_neon(text, length, set);
#else
    return sz_find_last_from_set_serial(text, length, set);
#endif
}

SZ_PUBLIC void sz_tolower(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    sz_tolower_serial(text, length, result);
}

SZ_PUBLIC void sz_toupper(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    sz_toupper_serial(text, length, result);
}

SZ_PUBLIC void sz_toascii(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    sz_toascii_serial(text, length, result);
}

SZ_PUBLIC sz_size_t sz_edit_distance( //
    sz_cptr_t a, sz_size_t a_length,  //
    sz_cptr_t b, sz_size_t b_length,  //
    sz_size_t bound, sz_memory_allocator_t const *alloc) {
    return sz_edit_distance_serial(a, a_length, b, b_length, bound, alloc);
}

SZ_PUBLIC sz_ssize_t sz_alignment_score(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                        sz_error_cost_t gap, sz_error_cost_t const *subs,
                                        sz_memory_allocator_t const *alloc) {
    return sz_alignment_score_serial(a, a_length, b, b_length, gap, subs, alloc);
}

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_H_
