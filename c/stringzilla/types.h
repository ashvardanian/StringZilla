/**
 *  @file c/stringzilla/types.h
 *  @author Ash Vardanian
 *  @date August 14, 2020
 *  @brief Shared definitions for the StringZilla library.
 *
 *  Includes the following types:
 *
 *  - @c sz_u8_t, @c sz_u16_t, @c sz_u32_t, @c sz_u64_t - unsigned integers of 8, 16, 32, 64 bits.
 *  - @c sz_i8_t, @c sz_i16_t, @c sz_i32_t, @c sz_i64_t - signed integers of 8, 16, 32, 64 bits.
 *  - @c sz_size_t, @c sz_ssize_t - unsigned and signed integers of the same size as a pointer.
 *  - @c sz_ptr_t, @c sz_cptr_t - pointer and constant pointer to a C-style string.
 *  - @c sz_bool_t - boolean type, @c sz_true_k and @c sz_false_k constants.
 *  - @c sz_ordering_t - for comparison results, @c sz_less_k, @c sz_equal_k, @c sz_greater_k.
 *  - @c sz_u8_vec_t, @c sz_u16_vec_t, @c sz_u32_vec_t, @c sz_u64_vec_t - @b SWAR vector types.
 *  - @c sz_u128_vec_t, @c sz_u256_vec_t, @c sz_u512_vec_t - @b SIMD vector types for x86 and Arm.
 *  - @c sz_rune_t - for 32-bit Unicode code points ~ @b runes.
 *  - @c sz_rune_length_t - to describe the number of bytes in a UTF8-encoded rune.
 *  - @c sz_error_cost_t - for substitution costs in string alignment and scoring algorithms.
 *
 *  The library also defines the following higher-level structures:
 *
 *  - @c sz_string_view_t - a C-style structure like @c std::string_view.
 *  - @c sz_allocator_t - a wrapper for memory-management functions.
 *  - @c sz_strings_view_t - an ordered collection of strings, as a tape or behind an accessor.
 *  - @c sz_byteset_t - a bitset for 256 possible byte values.
 */
#if !defined(STRINGZILLA_TYPES_H_)
#define STRINGZILLA_TYPES_H_

#define STRINGZILLA_VERSION_MAJOR 6
#define STRINGZILLA_VERSION_MINOR 0
#define STRINGZILLA_VERSION_PATCH 0

/*  Debugging and testing. */
#if !defined(STRINGZILLA_DEBUG)
#if defined(DEBUG) || defined(_DEBUG) // This means "Not using DEBUG information".
#define STRINGZILLA_DEBUG (1)
#else
#define STRINGZILLA_DEBUG (0)
#endif
#endif

/**
 *  @brief When set to 1, the library will include the following LibC headers: <stddef.h>
 *      and <stdint.h>. In debug builds, with STRINGZILLA_DEBUG=1, it will also include
 *      <stdio.h> and <stdlib.h>.
 *
 *  You may want to disable this compiling for use in the kernel, or in embedded systems. You may
 *  also avoid them, if you are very sensitive to compilation time and avoid pre-compiled headers.
 *
 *  @see Compile Health: https://artificial-mind.net/projects/compile-health/
 */
#if !defined(STRINGZILLA_WITH_LIBC)
#define STRINGZILLA_WITH_LIBC (1) // true or false
#endif

/** Compiles every kernel into this translation unit and stubs every dispatch point and finder with
 *  @c sz_missing_library_k, instead of linking them from the StringZilla library. */
#if !defined(STRINGZILLA_HEADER_ONLY)
#define STRINGZILLA_HEADER_ONLY (0) // true or false
#endif

/** 0 folds each dispatch table to its best kernel; 1, the default, keeps every compiled one. */
#if !defined(STRINGZILLA_MULTIVERSIONED)
#define STRINGZILLA_MULTIVERSIONED 1
#endif

/**
 *  @brief A misaligned load can be - trying to fetch eight consecutive bytes from an address that
 *      is not divisible by eight. On x86 enabled by default. On Arm it's not.
 *
 *  Most platforms support it, but there is no industry standard way to check for those. This value
 *  will mostly affect the performance of the serial (SWAR) backend.
 */
#if !defined(STRINGZILLA_ALLOW_MISALIGNED_LOADS)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define STRINGZILLA_ALLOW_MISALIGNED_LOADS (1) // true or false
#else
#define STRINGZILLA_ALLOW_MISALIGNED_LOADS (0) // true or false
#endif
#endif

/**
 *  @brief Analogous to @c size_t and @c std::size_t, unsigned integer, identical to pointer size.
 *      64-bit on most platforms where pointers are 64-bit, 32-bit where pointers are 32-bit.
 *
 *  @note Do not use @c STRINGZILLA_ARCH_X8664_ or @c STRINGZILLA_ARCH_ARM64_ here — those
 *      indicate the CPU family, not pointer width. Rely on compiler/OS macros only.
 */
#if defined(__SIZEOF_POINTER__)
#define STRINGZILLA_ARCH_64BIT_ (__SIZEOF_POINTER__ == 8)
#elif defined(__LP64__) || defined(_LP64) || defined(_WIN64)
#define STRINGZILLA_ARCH_64BIT_ (1)
#else
#define STRINGZILLA_ARCH_64BIT_ (0)
#endif

/**
 *  @brief On Big-Endian machines StringZilla will work in compatibility mode. This disables SWAR
 *      hacks to minimize code duplication, assuming all popular modern platforms are Little-Endian.
 *
 *  Modern compilers typically define @c __BYTE_ORDER__ and @c __ORDER_BIG_ENDIAN__. Fall back to
 *  legacy macros and known arch tags when unavailable.
 *
 *  @see Detecting endianness: https://stackoverflow.com/a/27054190
 */
#if !defined(STRINGZILLA_ARCH_BIG_ENDIAN_)
#if (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)) ||                                         \
    (defined(__BYTE_ORDER) && (__BYTE_ORDER == __BIG_ENDIAN)) || defined(__BIG_ENDIAN__) || defined(_BIG_ENDIAN) ||  \
    defined(__ARMEB__) || defined(__THUMBEB__) || defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) || \
    defined(__MIBSEB__) || defined(__s390x__) || defined(__s390__)
#define STRINGZILLA_ARCH_BIG_ENDIAN_ (1) // It's a big-endian target architecture
#else
#define STRINGZILLA_ARCH_BIG_ENDIAN_ (0) // It's a little-endian target architecture
#endif
#endif

/** Infer the target architecture, unless it's overridden by the build system. */
#if !defined(STRINGZILLA_ARCH_X8664_)
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !defined(__METAL_VERSION__)
#define STRINGZILLA_ARCH_X8664_ (1)
#else
#define STRINGZILLA_ARCH_X8664_ (0)
#endif
#endif
#if !defined(STRINGZILLA_ARCH_ARM64_)
#if (defined(__aarch64__) || defined(__arm64__) || defined(__arm64) || defined(_M_ARM64)) && \
    !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__METAL_VERSION__)
#define STRINGZILLA_ARCH_ARM64_ (1)
#else
#define STRINGZILLA_ARCH_ARM64_ (0)
#endif
#endif
#if !defined(STRINGZILLA_ARCH_RISCV64_)
#if defined(__riscv) && (__riscv_xlen == 64)
#define STRINGZILLA_ARCH_RISCV64_ (1)
#else
#define STRINGZILLA_ARCH_RISCV64_ (0)
#endif
#endif
#if !defined(STRINGZILLA_ARCH_LOONGARCH64_)
#if defined(__loongarch__)
#define STRINGZILLA_ARCH_LOONGARCH64_ (1)
#else
#define STRINGZILLA_ARCH_LOONGARCH64_ (0)
#endif
#endif
#if !defined(STRINGZILLA_ARCH_PPC64_)
#if defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
#define STRINGZILLA_ARCH_PPC64_ (1)
#else
#define STRINGZILLA_ARCH_PPC64_ (0)
#endif
#endif
#if !defined(STRINGZILLA_ARCH_WASM_)
#if defined(__wasm__) || defined(__EMSCRIPTEN__)
#define STRINGZILLA_ARCH_WASM_ (1)
#else
#define STRINGZILLA_ARCH_WASM_ (0)
#endif
#endif

#pragma region Platform Identity

/*  Derived once from the compiler's predefines, which no other file reads and no build sets. */
#if defined(STRINGZILLA_OS_LINUX_) || defined(STRINGZILLA_OS_APPLE_) || defined(STRINGZILLA_OS_WINDOWS_) || \
    defined(STRINGZILLA_OS_FREEBSD_) || defined(STRINGZILLA_OS_POSIX_)
#error "STRINGZILLA_OS_*_ are identity, derived from the compiler; no build sets them"
#endif

/** Whether the kernel is Linux, Android included, as 1 or 0. */
#if defined(__linux__)
#define STRINGZILLA_OS_LINUX_ 1
#else
#define STRINGZILLA_OS_LINUX_ 0
#endif

/** Whether the kernel is Darwin: macOS, iOS, tvOS, watchOS or visionOS, as 1 or 0. */
#if defined(__APPLE__)
#define STRINGZILLA_OS_APPLE_ 1
#else
#define STRINGZILLA_OS_APPLE_ 0
#endif

/** Whether the target is Windows, MinGW included and Cygwin not, as 1 or 0. */
#if defined(_WIN32)
#define STRINGZILLA_OS_WINDOWS_ 1
#else
#define STRINGZILLA_OS_WINDOWS_ 0
#endif

/** Whether the kernel is FreeBSD, as 1 or 0. */
#if defined(__FreeBSD__)
#define STRINGZILLA_OS_FREEBSD_ 1
#else
#define STRINGZILLA_OS_FREEBSD_ 0
#endif

/** Whether the POSIX interfaces exist: Linux, Darwin, FreeBSD or Cygwin, as 1 or 0. */
#if STRINGZILLA_OS_LINUX_ || STRINGZILLA_OS_APPLE_ || STRINGZILLA_OS_FREEBSD_ || defined(__CYGWIN__)
#define STRINGZILLA_OS_POSIX_ 1
#else
#define STRINGZILLA_OS_POSIX_ 0
#endif

#if STRINGZILLA_OS_LINUX_ + STRINGZILLA_OS_APPLE_ + STRINGZILLA_OS_WINDOWS_ + STRINGZILLA_OS_FREEBSD_ > 1
#error "Two operating-system identities hold at once"
#endif
#if STRINGZILLA_OS_POSIX_ && STRINGZILLA_OS_WINDOWS_
#error "POSIX and Windows identities hold at once"
#endif

#pragma endregion Platform Identity

/** Threshold for switching to SWAR (8-bytes at a time) backend over serial byte-level for-loops. On
 *  very short strings, under 16 bytes long, at most a single word will be processed with SWAR.
 *  Assuming potentially misaligned loads, SWAR makes sense only after ~24 bytes. */
#if !defined(STRINGZILLA_SWAR_THRESHOLD)
#if STRINGZILLA_DEBUG
#define STRINGZILLA_SWAR_THRESHOLD (8u) // 8 bytes in debug builds
#else
#define STRINGZILLA_SWAR_THRESHOLD (24u) // 24 bytes in release builds
#endif
#endif

/**
 *  @brief Function annotations on two axes, role and, for internal helpers, inlining policy:
 *
 *  @verbatim
 *  STRINGZILLA_API           public function or kernel: once in the library, header-inline in header-only builds
 *  STRINGZILLA_CONSTEXPR     internal helper, compiler decides inlining
 *  STRINGZILLA_INLINE        internal helper forced inline, structural: devirtualizing driver loops
 *  STRINGZILLA_OUTLINED     internal helper forced out-of-line
 *  STRINGZILLA_DEVICE        CUDA or HIP device helper forced inline
 *  @endverbatim
 *
 *  @c STRINGZILLA_OUTLINED omits @c inline, as GCC ignores @c noinline on an @c inline function.
 */
#define STRINGZILLA_C_INLINE_ inline static

#if defined(__GNUC__) || defined(__clang__)
#define STRINGZILLA_MAYBE_UNUSED_ __attribute__((unused))
#else
#define STRINGZILLA_MAYBE_UNUSED_
#endif

#if defined(_MSC_VER)
#define STRINGZILLA_INLINE __forceinline static
#define STRINGZILLA_OUTLINED __declspec(noinline) STRINGZILLA_C_INLINE_
#elif defined(__clang__)
#define STRINGZILLA_INLINE STRINGZILLA_MAYBE_UNUSED_ __attribute__((always_inline)) STRINGZILLA_C_INLINE_
#define STRINGZILLA_OUTLINED STRINGZILLA_MAYBE_UNUSED_ static __attribute__((noinline))
#else

/** GCC checks @c always_inline at the call as written, so a generic caller outside a target region
 *  needs plain inline. */
#define STRINGZILLA_INLINE STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_
#define STRINGZILLA_OUTLINED STRINGZILLA_MAYBE_UNUSED_ static __attribute__((noinline))
#endif

/** The C++ standard in effect, or zero in C. MSVC reports it in @c _MSVC_LANG and leaves
 *  @c __cplusplus at C++98. */
#if defined(_MSVC_LANG)
#define STRINGZILLA_CXX_STANDARD_ _MSVC_LANG
#elif defined(__cplusplus)
#define STRINGZILLA_CXX_STANDARD_ __cplusplus
#else
#define STRINGZILLA_CXX_STANDARD_ 0
#endif

/**
 *  @brief A portable scalar helper, inline in whichever translation unit uses it, and @c constexpr
 *      from C++20 onwards.
 *
 *  That qualifier is what lets a caller fold the helper at compile time, and what lets a CUDA
 *  kernel call it at all - @c --expt-relaxed-constexpr reaches a host @c constexpr function from
 *  device code, so this layer never has to name an execution space of its own.
 *
 *  A helper reaching an intrinsic can never be constant-evaluated, and a @c constexpr function with
 *  no constant-evaluated path is ill-formed: Clang and MSVC reject the definition, GCC 12 too, and
 *  only GCC 13+ softens it to @c -Winvalid-constexpr. Every such helper - the whole of each ISA
 *  backend, plus the few portable ones wrapping a builtin or a type-punned load - carries
 *  @c STRINGZILLA_INLINE instead, which is why no translation unit needs a @c -Wno- flag to
 *  compile this header.
 *
 *  The qualifier waits for C++20 because these helpers declare their locals before filling them,
 *  which only C++20 permits in a @c constexpr function. C, and an older C++ dialect reaching the C
 *  API, get the plain helper.
 */
#if STRINGZILLA_CXX_STANDARD_ >= 202002L
#define STRINGZILLA_CONSTEXPR STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_ constexpr
#else
#define STRINGZILLA_CONSTEXPR STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_
#endif

/** Every public function: header-inline in header-only builds, otherwise defined once in the
 *  library. Windows DLLs export every symbol through CMake, so no import or export spelling. */
#if STRINGZILLA_HEADER_ONLY
#define STRINGZILLA_API STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_
#elif defined(__clang__)
#define STRINGZILLA_API __attribute__((visibility("default")))
#elif defined(__GNUC__)

/** GCC inlines the plain-inline helpers into an exported kernel only when told to. */
#define STRINGZILLA_API __attribute__((visibility("default"), flatten))
#else
#define STRINGZILLA_API
#endif

/** Translation-unit-local lookup table storage; read-only types carry @c const separately. */
#define STRINGZILLA_CONSTANT static

#if defined(__CUDACC__) || defined(__HIP__)

/** Forces a CUDA or HIP device helper inline. */
#define STRINGZILLA_DEVICE static __device__ __forceinline__

/** Keeps a CUDA or HIP device helper out of line. */
#define STRINGZILLA_DEVICE_NOINLINE __device__ __noinline__

/** Device lookup table storage inside the one device function reading it, so a unit that never
 *  calls that function emits no table; a file-scope @c __device__ table ships in every unit. */
#define STRINGZILLA_DEVICE_CONSTANT static
#endif

/**
 *  @brief Disables stack protection for performance-critical functions.
 *
 *  GCC's @c -fstack-protector-strong inserts stack canary checks for functions with local arrays
 *  or buffers. For hash functions that use fixed-size state structures, this is unnecessary
 *  overhead (~10 cycles per call). This macro opts out of stack protection for such functions.
 */
#if defined(__GNUC__) || defined(__clang__)
#define STRINGZILLA_NO_STACK_PROTECTOR_ __attribute__((no_stack_protector))
#else
#define STRINGZILLA_NO_STACK_PROTECTOR_
#endif

/** Alignment macro for N-byte alignment. */
#if defined(_MSC_VER)
#define sz_align_(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
#define sz_align_(n) __attribute__((aligned(n)))
#else
#define sz_align_(n)
#endif

/** Lets a type legally alias any other, so the SIMD vector unions' differently-typed views
 *  of one storage, like reading a @c sz_u8_t state buffer back through @c sz_u128_vec_t, are
 *  well-defined. Without it GCC at @c -O3 assumes the views never alias and may reorder
 *  them, corrupting results. */
#if defined(__GNUC__) || defined(__clang__)
#define STRINGZILLA_MAY_ALIAS_ __attribute__((__may_alias__))
#else
#define STRINGZILLA_MAY_ALIAS_
#endif

/**
 *  @brief Forbids the compiler from eliding stores to @p pointer that nothing
 *      reads back afterwards.
 *
 *  Zeroing a buffer that is about to die is a dead store, and an optimizer is entitled to drop it.
 *  That is fatal when the buffer held key material, so a scrub places ordinary wide stores and then
 *  this barrier, rather than paying for @c volatile on every byte - @c volatile also forbids
 *  vectorization, so a scrub written that way cannot use more than one byte per store.
 */
#if defined(__GNUC__) || defined(__clang__)
#define sz_keep_alive_(pointer) __asm__ __volatile__("" : : "r"(pointer) : "memory")
#elif defined(_MSC_VER)
#include <intrin.h> // `_ReadWriteBarrier`
#define sz_keep_alive_(pointer) (_ReadWriteBarrier(), (void)(pointer))
#else
#define sz_keep_alive_(pointer) ((void)(pointer))
#endif

/**
 *  @brief C99 static array parameter annotation for minimum array size. In C, expands to
 *      <tt>static n</tt> enabling compiler bounds checking. In C++, expands to nothing as this syntax
 *      is not supported.
 *
 *  It annotates declarations like these:
 *
 *  @code{.c}
 *      void hash_digest(uint8_t digest[sz_at_least_(32)]);
 *      void lookup(uint8_t const lut[sz_at_least_(256)]);
 *  @endcode
 *
 *  @see Static array parameters in C: https://lwn.net/Articles/1046840/
 */
#if defined(__cplusplus) || defined(_MSC_VER)
#define sz_at_least_(n)
#else
#define sz_at_least_(n) static n
#endif

/** Largest value that fits into 8 bits. */
#define STRINGZILLA_U8_MAX (255u)

/** Largest value that fits into 16 bits. */
#define STRINGZILLA_U16_MAX (65535u)

/** Largest prime number that fits into 16 bits. */
#define STRINGZILLA_U16_MAX_PRIME (65521u)

/** Largest prime number that fits into 31 bits. */
#define STRINGZILLA_U32_MAX_PRIME (2147483647u)

/**
 *  @brief Largest prime number that fits into 64 bits.
 *
 *  @verbatim
 *  2⁶⁴  = 18,446,744,073,709,551,616
 *  this = 18,446,744,073,709,551,557
 *  diff = 59
 *  @endverbatim
 *
 *  @see Largest 64-bit prime: https://mersenneforum.org/showthread.php?t=3471
 */
#define STRINGZILLA_U64_MAX_PRIME (18446744073709551557ull)

/** Opt into glibc's "misc" extensions, @c _DEFAULT_SOURCE implying @c __USE_MISC, before the first
 *  LibC header is pulled in. A strict @c -std=cNN otherwise hides declarations like <tt>syscall()</tt>,
 *  which the RISC-V @c riscv_hwprobe capability probe in <tt>capabilities.h</tt> relies on. As the first
 *  StringZilla header every translation unit includes, this is the one place that covers every
 *  build system, be it CMake, Cargo, or setuptools. */
#if STRINGZILLA_OS_LINUX_ && STRINGZILLA_WITH_LIBC && !defined(_DEFAULT_SOURCE) && !defined(_GNU_SOURCE)
#define _DEFAULT_SOURCE 1
#endif

#if STRINGZILLA_WITH_LIBC
#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`
#include <string.h> // `memcpy`, `memmove`, `memset`
#endif

/*  The headers needed for the @c sz_assert_failure_ function. */
#if STRINGZILLA_DEBUG && STRINGZILLA_WITH_LIBC
#include <stdio.h>  // `fprintf`, `stderr`
#include <stdlib.h> // `abort`
#endif

/*  Two flags per capability. @c STRINGZILLA_ARCH_<ARCH>_<CAPABILITY>_ says its code compiles in
 *  this unit: detected here from the unit's own flags, a sound lower bound, and raised or zeroed
 *  only by this library's build with its probe's verdict, as each tier header scopes its code to
 *  its kit with a target pragma; it never reads a target. The unit has the capability's kernels
 *  where @c STRINGZILLA_TARGET_<CAPABILITY> says so, by default a literal from the same detection.
 *  LASX and POWER9 compile only under `-mlasx` and `-mcpu=power9`, as `lasxintrin.h` and
 *  `altivec.h` hide without them. */
#if !defined(STRINGZILLA_ARCH_X8664_WESTMERE_) || (STRINGZILLA_ARCH_X8664_WESTMERE_ && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_ARCH_X8664_WESTMERE_
#if STRINGZILLA_ARCH_X8664_ && defined(__SSE4_2__) && defined(__AES__)
#define STRINGZILLA_ARCH_X8664_WESTMERE_ (1)
#elif STRINGZILLA_ARCH_X8664_ && defined(_MSC_VER) && defined(__AVX__)
#define STRINGZILLA_ARCH_X8664_WESTMERE_ (1) // ! MSVC doesn't expose `__SSE4_2__`, `__AES__` macros
#else
#define STRINGZILLA_ARCH_X8664_WESTMERE_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_WESTMERE) || (STRINGZILLA_TARGET_WESTMERE && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_TARGET_WESTMERE
#if STRINGZILLA_ARCH_X8664_WESTMERE_
#define STRINGZILLA_TARGET_WESTMERE (1)
#else
#define STRINGZILLA_TARGET_WESTMERE (0)
#endif
#endif

#if !defined(STRINGZILLA_ARCH_X8664_HASWELL_) || (STRINGZILLA_ARCH_X8664_HASWELL_ && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_ARCH_X8664_HASWELL_
#if STRINGZILLA_ARCH_X8664_ && defined(__AVX2__)
#define STRINGZILLA_ARCH_X8664_HASWELL_ (1)
#else
#define STRINGZILLA_ARCH_X8664_HASWELL_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_HASWELL) || (STRINGZILLA_TARGET_HASWELL && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_TARGET_HASWELL
#if STRINGZILLA_ARCH_X8664_HASWELL_
#define STRINGZILLA_TARGET_HASWELL (1)
#else
#define STRINGZILLA_TARGET_HASWELL (0)
#endif
#endif

#if !defined(STRINGZILLA_ARCH_X8664_GOLDMONT_) || (STRINGZILLA_ARCH_X8664_GOLDMONT_ && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_ARCH_X8664_GOLDMONT_
#if STRINGZILLA_ARCH_X8664_ && defined(__SHA__)
#define STRINGZILLA_ARCH_X8664_GOLDMONT_ (1)
#elif STRINGZILLA_ARCH_X8664_ && defined(_MSC_VER) && defined(__AVX2__)
#define STRINGZILLA_ARCH_X8664_GOLDMONT_ (1) // ! MSVC doesn't expose `__SHA__` macros
#else
#define STRINGZILLA_ARCH_X8664_GOLDMONT_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_GOLDMONT) || (STRINGZILLA_TARGET_GOLDMONT && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_TARGET_GOLDMONT
#if STRINGZILLA_ARCH_X8664_GOLDMONT_
#define STRINGZILLA_TARGET_GOLDMONT (1)
#else
#define STRINGZILLA_TARGET_GOLDMONT (0)
#endif
#endif

#if !defined(STRINGZILLA_ARCH_X8664_SKYLAKE_) || (STRINGZILLA_ARCH_X8664_SKYLAKE_ && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_ARCH_X8664_SKYLAKE_
#if STRINGZILLA_ARCH_X8664_ && defined(__AVX512F__)
#define STRINGZILLA_ARCH_X8664_SKYLAKE_ (1)
#else
#define STRINGZILLA_ARCH_X8664_SKYLAKE_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_SKYLAKE) || (STRINGZILLA_TARGET_SKYLAKE && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_TARGET_SKYLAKE
#if STRINGZILLA_ARCH_X8664_SKYLAKE_
#define STRINGZILLA_TARGET_SKYLAKE (1)
#else
#define STRINGZILLA_TARGET_SKYLAKE (0)
#endif
#endif

#if !defined(STRINGZILLA_ARCH_X8664_ICELAKE_) || (STRINGZILLA_ARCH_X8664_ICELAKE_ && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_ARCH_X8664_ICELAKE_
#if STRINGZILLA_ARCH_X8664_ && defined(__AVX512BW__) && defined(__VAES__)
#define STRINGZILLA_ARCH_X8664_ICELAKE_ (1)
#elif STRINGZILLA_ARCH_X8664_ && defined(_MSC_VER) && defined(__AVX512BW__)
#define STRINGZILLA_ARCH_X8664_ICELAKE_ (1) // ! MSVC doesn't expose `__VAES__` macros
#else
#define STRINGZILLA_ARCH_X8664_ICELAKE_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_ICELAKE) || (STRINGZILLA_TARGET_ICELAKE && !STRINGZILLA_ARCH_X8664_)
#undef STRINGZILLA_TARGET_ICELAKE
#if STRINGZILLA_ARCH_X8664_ICELAKE_
#define STRINGZILLA_TARGET_ICELAKE (1)
#else
#define STRINGZILLA_TARGET_ICELAKE (0)
#endif
#endif

/*  NEON support is optional in Armv7/AArch32, but mandatory from 8.0 onwards. The
 *  `!STRINGZILLA_ARCH_BIG_ENDIAN_` guard keeps the SIMD stack off exotic @c aarch64_be targets: the
 *  shared byte-lane bridges, like `sz_utf8_rune_pred_to_u64_*` and
 *  @c sz_utf8_vreinterpretq_u8_u4_neon_, assume little-endian lane ↔ byte order and would
 *  mis-execute on big-endian. */
#if !defined(STRINGZILLA_ARCH_ARM64_NEON_) || (STRINGZILLA_ARCH_ARM64_NEON_ && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_ARCH_ARM64_NEON_
#if STRINGZILLA_ARCH_ARM64_ && defined(__ARM_NEON) && !STRINGZILLA_ARCH_BIG_ENDIAN_
#define STRINGZILLA_ARCH_ARM64_NEON_ (1)
#elif STRINGZILLA_ARCH_ARM64_ && defined(_MSC_VER) && defined(_M_ARM64)
#define STRINGZILLA_ARCH_ARM64_NEON_ (1) // ! MSVC lacks `__ARM_NEON`, and its AArch64 is always little-endian
#else
#define STRINGZILLA_ARCH_ARM64_NEON_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_NEON) || (STRINGZILLA_TARGET_NEON && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_TARGET_NEON
#if STRINGZILLA_ARCH_ARM64_NEON_
#define STRINGZILLA_TARGET_NEON (1)
#else
#define STRINGZILLA_TARGET_NEON (0)
#endif
#endif

/*  SVE is optional since Armv8.2-A, but never became mandatory, and MSVC cannot probe for it. */
#if !defined(STRINGZILLA_ARCH_ARM64_SVE_) || (STRINGZILLA_ARCH_ARM64_SVE_ && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_ARCH_ARM64_SVE_
#if STRINGZILLA_ARCH_ARM64_ && defined(__ARM_FEATURE_SVE) && !STRINGZILLA_ARCH_BIG_ENDIAN_
#define STRINGZILLA_ARCH_ARM64_SVE_ (1)
#else
#define STRINGZILLA_ARCH_ARM64_SVE_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_SVE) || (STRINGZILLA_TARGET_SVE && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_TARGET_SVE
#if STRINGZILLA_ARCH_ARM64_SVE_
#define STRINGZILLA_TARGET_SVE (1)
#else
#define STRINGZILLA_TARGET_SVE (0)
#endif
#endif

/*  SVE2 is optional since Armv9.0-A, but never became mandatory, and MSVC cannot probe for it. */
#if !defined(STRINGZILLA_ARCH_ARM64_SVE2_) || (STRINGZILLA_ARCH_ARM64_SVE2_ && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_ARCH_ARM64_SVE2_
#if STRINGZILLA_ARCH_ARM64_ && defined(__ARM_FEATURE_SVE2) && !STRINGZILLA_ARCH_BIG_ENDIAN_
#define STRINGZILLA_ARCH_ARM64_SVE2_ (1)
#else
#define STRINGZILLA_ARCH_ARM64_SVE2_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_SVE2) || (STRINGZILLA_TARGET_SVE2 && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_TARGET_SVE2
#if STRINGZILLA_ARCH_ARM64_SVE2_
#define STRINGZILLA_TARGET_SVE2 (1)
#else
#define STRINGZILLA_TARGET_SVE2 (0)
#endif
#endif

/*  AES is optional since Armv8.0-A, but never became mandatory, and MSVC cannot probe for it. */
#if !defined(STRINGZILLA_ARCH_ARM64_NEONAES_) || (STRINGZILLA_ARCH_ARM64_NEONAES_ && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_ARCH_ARM64_NEONAES_
#if STRINGZILLA_ARCH_ARM64_ && (defined(__ARM_FEATURE_AES) || defined(__ARM_FEATURE_CRYPTO) || STRINGZILLA_OS_APPLE_)
#define STRINGZILLA_ARCH_ARM64_NEONAES_ (1)
#else
#define STRINGZILLA_ARCH_ARM64_NEONAES_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_NEONAES) || (STRINGZILLA_TARGET_NEONAES && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_TARGET_NEONAES
#if STRINGZILLA_ARCH_ARM64_NEONAES_
#define STRINGZILLA_TARGET_NEONAES (1)
#else
#define STRINGZILLA_TARGET_NEONAES (0)
#endif
#endif

/*  SHA2 is optional since Armv8.0-A, but never became mandatory, and MSVC cannot probe for it. */
#if !defined(STRINGZILLA_ARCH_ARM64_NEONSHA_) || (STRINGZILLA_ARCH_ARM64_NEONSHA_ && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_ARCH_ARM64_NEONSHA_
#if STRINGZILLA_ARCH_ARM64_ && (defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO) || STRINGZILLA_OS_APPLE_)
#define STRINGZILLA_ARCH_ARM64_NEONSHA_ (1)
#else
#define STRINGZILLA_ARCH_ARM64_NEONSHA_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_NEONSHA) || (STRINGZILLA_TARGET_NEONSHA && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_TARGET_NEONSHA
#if STRINGZILLA_ARCH_ARM64_NEONSHA_
#define STRINGZILLA_TARGET_NEONSHA (1)
#else
#define STRINGZILLA_TARGET_NEONSHA (0)
#endif
#endif

/*  SVE2 AES is optional since Armv9.0-A, but never became mandatory, and MSVC cannot probe for it.
 *  GCC spells the feature macro @c __ARM_FEATURE_SVE2AES; Clang spells it
 *  @c __ARM_FEATURE_SVE2_AES. Accept both. */
#if !defined(STRINGZILLA_ARCH_ARM64_SVE2AES_) || (STRINGZILLA_ARCH_ARM64_SVE2AES_ && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_ARCH_ARM64_SVE2AES_
#if STRINGZILLA_ARCH_ARM64_ && (defined(__ARM_FEATURE_SVE2AES) || defined(__ARM_FEATURE_SVE2_AES))
#define STRINGZILLA_ARCH_ARM64_SVE2AES_ (1)
#else
#define STRINGZILLA_ARCH_ARM64_SVE2AES_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_SVE2AES) || (STRINGZILLA_TARGET_SVE2AES && !STRINGZILLA_ARCH_ARM64_)
#undef STRINGZILLA_TARGET_SVE2AES
#if STRINGZILLA_ARCH_ARM64_SVE2AES_
#define STRINGZILLA_TARGET_SVE2AES (1)
#else
#define STRINGZILLA_TARGET_SVE2AES (0)
#endif
#endif

/** LLVM 18 through 21 carry @c evex512 as a separate target feature, split out of AVX-512 for the
 *  AVX10 transition; ZMM codegen in a per-function @c target attribute needs it named. LLVM 17 and
 *  older never knew the token, LLVM 22 retired it again, and Clang drops the whole attribute over
 *  one unknown feature - @c -Wignored-attributes, silently costing every AVX-512 kernel - so the
 *  fork is a closed version window, not a floor. Apple Clang 17 is built on LLVM 19, so it falls
 *  inside that window too. */
#if defined(__clang__) && __clang_major__ < 22 && \
    (__clang_major__ >= 18 || (defined(__apple_build_version__) && __clang_major__ >= 17))
#define STRINGZILLA_HAS_CLANG_EVEX512_ (1)
#else
#define STRINGZILLA_HAS_CLANG_EVEX512_ (0)
#endif

/** Compiled as CUDA, host and device passes alike: nvcc, or Clang through its CUDA runtime wrapper.
 *  HIP on NVIDIA goes through nvcc, so it is CUDA here too. Unlike the CPU facts, both GPU facts
 *  hold beside the host's architecture, so they never follow one in an @c #elif chain. The library
 *  also sets it for its host units, which list the kernels of its CUDA units and include no <tt>.cuh</tt>,
 *  so device code compiles only where the compiler set it. */
#if !defined(STRINGZILLA_ARCH_CUDA_)
#if defined(__CUDACC__) && !defined(__HIP__)
#define STRINGZILLA_ARCH_CUDA_ (1)
#else
#define STRINGZILLA_ARCH_CUDA_ (0)
#endif
#endif

/** Compiled as HIP for AMD GPUs, host and device passes alike; Clang predefines @c __HIP__ itself,
 *  unlike @c __HIP_PLATFORM_AMD__, which only <tt>hip_common.h</tt> defines. The library also sets it for
 *  its host units, which list the kernels of its one HIP unit. */
#if !defined(STRINGZILLA_ARCH_ROCM_)
#if defined(__HIP__)
#define STRINGZILLA_ARCH_ROCM_ (1)
#else
#define STRINGZILLA_ARCH_ROCM_ (0)
#endif
#endif

/** Whether the Metal host API is compiled in. The build stamps it and nothing infers it: no C
 *  compiler predefines anything for Apple GPUs, and the layer links Metal and Foundation. */
#if !defined(STRINGZILLA_ARCH_METAL_)
#define STRINGZILLA_ARCH_METAL_ (0)
#endif

/** Defining the serial kernels: STRINGZILLA_TARGET_SERIAL. Header-only builds define them in every
 *  translation unit; in the library only its serial unit does, so units that include a serial
 *  header for its helpers leave its kernels to the library. */
#if !defined(STRINGZILLA_TARGET_SERIAL)
#define STRINGZILLA_TARGET_SERIAL STRINGZILLA_HEADER_ONLY
#endif

/*  Every compiler of a vendor builds every generation's building blocks: static inline device
 *  functions, emitted only where a kernel compiled for that generation calls them. */
#if STRINGZILLA_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)
#define STRINGZILLA_ARCH_CUDA_AMPERE_ (1)
#define STRINGZILLA_ARCH_CUDA_HOPPER_ (1)
#define STRINGZILLA_ARCH_CUDA_BLACKWELL_ (1)
#else
#define STRINGZILLA_ARCH_CUDA_AMPERE_ (0)
#define STRINGZILLA_ARCH_CUDA_HOPPER_ (0)
#define STRINGZILLA_ARCH_CUDA_BLACKWELL_ (0)
#endif
#if STRINGZILLA_ARCH_ROCM_ && defined(__HIP__)
#define STRINGZILLA_ARCH_ROCM_CDNA3_ (1)
#define STRINGZILLA_ARCH_ROCM_CDNA4_ (1)
#define STRINGZILLA_ARCH_ROCM_CDNA5_ (1)
#else
#define STRINGZILLA_ARCH_ROCM_CDNA3_ (0)
#define STRINGZILLA_ARCH_ROCM_CDNA4_ (0)
#define STRINGZILLA_ARCH_ROCM_CDNA5_ (0)
#endif

/*  Defining the CUDA kernels, the SIMT baseline every NVIDIA device runs: STRINGZILLA_TARGET_CUDA.
 *  Forced off where the unit is not compiled as CUDA. */
#if !defined(STRINGZILLA_TARGET_CUDA) || (STRINGZILLA_TARGET_CUDA && !STRINGZILLA_ARCH_CUDA_)
#undef STRINGZILLA_TARGET_CUDA
#define STRINGZILLA_TARGET_CUDA STRINGZILLA_ARCH_CUDA_
#endif

/*  Defining the Ampere kernels, which reduce across a warp with @c redux.sync and copy into shared
 *  memory with @c cp.async , for compute capability 8.0 on: STRINGZILLA_TARGET_AMPERE. Off unless
 *  the build says its codes run them, and forced off where the unit is not compiled as CUDA. */
#if !defined(STRINGZILLA_TARGET_AMPERE) || (STRINGZILLA_TARGET_AMPERE && !STRINGZILLA_ARCH_CUDA_)
#undef STRINGZILLA_TARGET_AMPERE
#define STRINGZILLA_TARGET_AMPERE 0
#endif

/*  Defining the Hopper kernels, which group blocks into clusters sharing their shared memory, for
 *  compute capability 9.0 on: STRINGZILLA_TARGET_HOPPER. Off unless the build says its codes run
 *  them, and forced off where the unit is not compiled as CUDA. */
#if !defined(STRINGZILLA_TARGET_HOPPER) || (STRINGZILLA_TARGET_HOPPER && !STRINGZILLA_ARCH_CUDA_)
#undef STRINGZILLA_TARGET_HOPPER
#define STRINGZILLA_TARGET_HOPPER 0
#endif

/*  Defining the Blackwell kernels, whose blocks take over the tiles of unstarted blocks through
 *  cluster launch control, for compute capability 10.0 on: STRINGZILLA_TARGET_BLACKWELL. Off
 *  unless the build says its codes run them, and forced off where a unit is not CUDA. */
#if !defined(STRINGZILLA_TARGET_BLACKWELL) || (STRINGZILLA_TARGET_BLACKWELL && !STRINGZILLA_ARCH_CUDA_)
#undef STRINGZILLA_TARGET_BLACKWELL
#define STRINGZILLA_TARGET_BLACKWELL 0
#endif

/*  Defining the ROCm kernels, the SIMT baseline every AMD device runs: STRINGZILLA_TARGET_ROCM.
 *  Forced off where the unit is not compiled as HIP. */
#if !defined(STRINGZILLA_TARGET_ROCM) || (STRINGZILLA_TARGET_ROCM && !STRINGZILLA_ARCH_ROCM_)
#undef STRINGZILLA_TARGET_ROCM
#define STRINGZILLA_TARGET_ROCM STRINGZILLA_ARCH_ROCM_
#endif

/*  Defining the CDNA3 kernels, whose one-lane shifts move through a DPP shift of the whole 64-lane
 *  wavefront, for gfx942: STRINGZILLA_TARGET_CDNA3. Off unless the build says its targets run
 *  them, and forced off where the unit is not compiled as HIP. */
#if !defined(STRINGZILLA_TARGET_CDNA3) || (STRINGZILLA_TARGET_CDNA3 && !STRINGZILLA_ARCH_ROCM_)
#undef STRINGZILLA_TARGET_CDNA3
#define STRINGZILLA_TARGET_CDNA3 0
#endif

/*  Defining the CDNA4 kernels, which take the CDNA3 wavefront shift on gfx950:
 *  STRINGZILLA_TARGET_CDNA4. Off unless the build says its targets run them, and forced off where
 *  the unit is not compiled as HIP. */
#if !defined(STRINGZILLA_TARGET_CDNA4) || (STRINGZILLA_TARGET_CDNA4 && !STRINGZILLA_ARCH_ROCM_)
#undef STRINGZILLA_TARGET_CDNA4
#define STRINGZILLA_TARGET_CDNA4 0
#endif

/*  Defining the CDNA5 kernels, whose one-lane shifts move through DPP row shifts bridged by
 *  @c v_permlanex16 on a 32-lane wavefront, for gfx1250 and gfx1251: STRINGZILLA_TARGET_CDNA5. Off
 *  unless the build says its targets run them, and forced off where a unit is not HIP. */
#if !defined(STRINGZILLA_TARGET_CDNA5) || (STRINGZILLA_TARGET_CDNA5 && !STRINGZILLA_ARCH_ROCM_)
#undef STRINGZILLA_TARGET_CDNA5
#define STRINGZILLA_TARGET_CDNA5 0
#endif

/*  Defining the Metal kernels, the SIMT baseline every Apple7 device runs:
 *  STRINGZILLA_TARGET_METAL. Forced off where Metal is not linked. */
#if !defined(STRINGZILLA_TARGET_METAL) || (STRINGZILLA_TARGET_METAL && !STRINGZILLA_ARCH_METAL_)
#undef STRINGZILLA_TARGET_METAL
#define STRINGZILLA_TARGET_METAL STRINGZILLA_ARCH_METAL_
#endif

/*  WebAssembly SIMD128 is opt-in via @c -msimd128 with no runtime probe: an engine validates a
 *  module whole, so the flag alone decides, and a request without it is dropped. */
#if !defined(STRINGZILLA_ARCH_WASM_V128_) || \
    (STRINGZILLA_ARCH_WASM_V128_ && !(STRINGZILLA_ARCH_WASM_ && defined(__wasm_simd128__)))
#undef STRINGZILLA_ARCH_WASM_V128_
#if STRINGZILLA_ARCH_WASM_ && defined(__wasm_simd128__)
#define STRINGZILLA_ARCH_WASM_V128_ (1)
#else
#define STRINGZILLA_ARCH_WASM_V128_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_V128) || \
    (STRINGZILLA_TARGET_V128 && !(STRINGZILLA_ARCH_WASM_ && defined(__wasm_simd128__)))
#undef STRINGZILLA_TARGET_V128
#if STRINGZILLA_ARCH_WASM_V128_
#define STRINGZILLA_TARGET_V128 (1)
#else
#define STRINGZILLA_TARGET_V128 (0)
#endif
#endif

/*  WebAssembly @b relaxed SIMD — opt-in via @c -mrelaxed-simd; a level above baseline SIMD128
 *  adding relaxed swizzle, fused multiply-add, lane-select, and integer dot-products. Some
 *  runtimes lower a few relaxed ops sub-optimally, but the level is exposed so native engines
 *  can use them. As with SIMD128, the flag alone decides. */
#if !defined(STRINGZILLA_ARCH_WASM_V128RELAXED_) || \
    (STRINGZILLA_ARCH_WASM_V128RELAXED_ && !(STRINGZILLA_ARCH_WASM_ && defined(__wasm_relaxed_simd__)))
#undef STRINGZILLA_ARCH_WASM_V128RELAXED_
#if STRINGZILLA_ARCH_WASM_ && defined(__wasm_relaxed_simd__)
#define STRINGZILLA_ARCH_WASM_V128RELAXED_ (1)
#else
#define STRINGZILLA_ARCH_WASM_V128RELAXED_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_V128RELAXED) || \
    (STRINGZILLA_TARGET_V128RELAXED && !(STRINGZILLA_ARCH_WASM_ && defined(__wasm_relaxed_simd__)))
#undef STRINGZILLA_TARGET_V128RELAXED
#if STRINGZILLA_ARCH_WASM_V128RELAXED_
#define STRINGZILLA_TARGET_V128RELAXED (1)
#else
#define STRINGZILLA_TARGET_V128RELAXED (0)
#endif
#endif

/*  RISC-V Vector extension (RVV 1.0) — `-march=rv64gcv`. Length-agnostic registers. */
#if !defined(STRINGZILLA_ARCH_RISCV64_RVV_) || (STRINGZILLA_ARCH_RISCV64_RVV_ && !STRINGZILLA_ARCH_RISCV64_)
#undef STRINGZILLA_ARCH_RISCV64_RVV_
#if STRINGZILLA_ARCH_RISCV64_ && defined(__riscv_vector)
#define STRINGZILLA_ARCH_RISCV64_RVV_ (1)
#else
#define STRINGZILLA_ARCH_RISCV64_RVV_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_RVV) || (STRINGZILLA_TARGET_RVV && !STRINGZILLA_ARCH_RISCV64_)
#undef STRINGZILLA_TARGET_RVV
#if STRINGZILLA_ARCH_RISCV64_RVV_
#define STRINGZILLA_TARGET_RVV (1)
#else
#define STRINGZILLA_TARGET_RVV (0)
#endif
#endif

/*  RISC-V Vector Crypto (Zvkned, Zvknhb, Zvkg) — `-march=rv64gcv_zvkned_zvknhb_zvkg`. */
#if !defined(STRINGZILLA_ARCH_RISCV64_RVVCRYPTO_) || (STRINGZILLA_ARCH_RISCV64_RVVCRYPTO_ && !STRINGZILLA_ARCH_RISCV64_)
#undef STRINGZILLA_ARCH_RISCV64_RVVCRYPTO_
#if STRINGZILLA_ARCH_RISCV64_ && defined(__riscv_vector) && defined(__riscv_zvkned) && defined(__riscv_zvknhb) && \
    defined(__riscv_zvkg)
#define STRINGZILLA_ARCH_RISCV64_RVVCRYPTO_ (1)
#else
#define STRINGZILLA_ARCH_RISCV64_RVVCRYPTO_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_RVVCRYPTO) || (STRINGZILLA_TARGET_RVVCRYPTO && !STRINGZILLA_ARCH_RISCV64_)
#undef STRINGZILLA_TARGET_RVVCRYPTO
#if STRINGZILLA_ARCH_RISCV64_RVVCRYPTO_
#define STRINGZILLA_TARGET_RVVCRYPTO (1)
#else
#define STRINGZILLA_TARGET_RVVCRYPTO (0)
#endif
#endif

/*  LoongArch Advanced SIMD eXtension (LASX, 256-bit) — `-mlasx`. */
#if !defined(STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_) || \
    (STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_ && !STRINGZILLA_ARCH_LOONGARCH64_)
#undef STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_
#if STRINGZILLA_ARCH_LOONGARCH64_ && defined(__loongarch_asx)
#define STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_ (1)
#else
#define STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_LOONGSONASX) || (STRINGZILLA_TARGET_LOONGSONASX && !STRINGZILLA_ARCH_LOONGARCH64_)
#undef STRINGZILLA_TARGET_LOONGSONASX
#if STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_
#define STRINGZILLA_TARGET_LOONGSONASX (1)
#else
#define STRINGZILLA_TARGET_LOONGSONASX (0)
#endif
#endif

/*  IBM Power Vector-Scalar eXtension at the POWER9 level, ISA 3.0 — `-mcpu=power9`. */
#if !defined(STRINGZILLA_ARCH_PPC64_POWERVSX_) || (STRINGZILLA_ARCH_PPC64_POWERVSX_ && !STRINGZILLA_ARCH_PPC64_)
#undef STRINGZILLA_ARCH_PPC64_POWERVSX_
#if STRINGZILLA_ARCH_PPC64_ && defined(__VSX__) && defined(__POWER9_VECTOR__)
#define STRINGZILLA_ARCH_PPC64_POWERVSX_ (1)
#else
#define STRINGZILLA_ARCH_PPC64_POWERVSX_ (0)
#endif
#endif
#if !defined(STRINGZILLA_TARGET_POWERVSX) || (STRINGZILLA_TARGET_POWERVSX && !STRINGZILLA_ARCH_PPC64_)
#undef STRINGZILLA_TARGET_POWERVSX
#if STRINGZILLA_ARCH_PPC64_POWERVSX_
#define STRINGZILLA_TARGET_POWERVSX (1)
#else
#define STRINGZILLA_TARGET_POWERVSX (0)
#endif
#endif

/*  Building blocks follow what is built on them, top-down. Each target above took a literal 0 or
 *  1 from its own detection, so raising a base's flag opens its blocks, never its kernels. */
#if STRINGZILLA_ARCH_X8664_ICELAKE_
#undef STRINGZILLA_ARCH_X8664_SKYLAKE_
#define STRINGZILLA_ARCH_X8664_SKYLAKE_ (1)
#endif
#if STRINGZILLA_ARCH_X8664_SKYLAKE_
#undef STRINGZILLA_ARCH_X8664_HASWELL_
#define STRINGZILLA_ARCH_X8664_HASWELL_ (1)
#endif
#if STRINGZILLA_ARCH_X8664_HASWELL_
#undef STRINGZILLA_ARCH_X8664_WESTMERE_
#define STRINGZILLA_ARCH_X8664_WESTMERE_ (1)
#endif
#if STRINGZILLA_ARCH_ARM64_SVE2AES_
#undef STRINGZILLA_ARCH_ARM64_SVE2_
#define STRINGZILLA_ARCH_ARM64_SVE2_ (1)
#undef STRINGZILLA_ARCH_ARM64_NEONAES_
#define STRINGZILLA_ARCH_ARM64_NEONAES_ (1)
#endif
#if STRINGZILLA_ARCH_ARM64_SVE2_
#undef STRINGZILLA_ARCH_ARM64_SVE_
#define STRINGZILLA_ARCH_ARM64_SVE_ (1)
#endif
#if STRINGZILLA_ARCH_ARM64_NEONAES_ || STRINGZILLA_ARCH_ARM64_NEONSHA_ || STRINGZILLA_ARCH_ARM64_SVE_
#undef STRINGZILLA_ARCH_ARM64_NEON_
#define STRINGZILLA_ARCH_ARM64_NEON_ (1)
#endif
#if STRINGZILLA_ARCH_RISCV64_RVVCRYPTO_
#undef STRINGZILLA_ARCH_RISCV64_RVV_
#define STRINGZILLA_ARCH_RISCV64_RVV_ (1)
#endif
#if STRINGZILLA_ARCH_WASM_V128RELAXED_
#undef STRINGZILLA_ARCH_WASM_V128_
#define STRINGZILLA_ARCH_WASM_V128_ (1)
#endif

/*  Hardware-specific headers for different SIMD intrinsics and register wrappers. */
#if STRINGZILLA_ARCH_WASM_V128_
#include <wasm_simd128.h>
#endif // STRINGZILLA_ARCH_WASM_V128_
#if STRINGZILLA_ARCH_RISCV64_RVV_
#include <riscv_vector.h>
#endif // STRINGZILLA_ARCH_RISCV64_RVV_
/*  The LASX and POWER9 intrinsics headers open in the same target region as their kernels.
 *  GCC 15 defines @c __loongarch_asx and @c __POWER9_VECTOR__ there, while Clang defines them
 *  only under `-mlasx` or `-mcpu=power9`. */
#if STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_
#if !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("lasx")
#endif
#include <lsxintrin.h>  // 128-bit `__lsx_*` intrinsics and the `__m128i` register type, for sub-32-byte inputs
#include <lasxintrin.h> // 256-bit `__lasx_*` intrinsics and the `__m256i` register type
#if !defined(__clang__)
#pragma GCC pop_options
#endif
#endif
#if STRINGZILLA_ARCH_PPC64_POWERVSX_
#if !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif
#include <altivec.h>
#if !defined(__clang__)
#pragma GCC pop_options
#endif
#endif
#if STRINGZILLA_ARCH_X8664_WESTMERE_ || STRINGZILLA_ARCH_X8664_GOLDMONT_
#include <immintrin.h>
#if !defined(_MSC_VER) || defined(__clang__)
#include <x86intrin.h> // `_bit_scan_forward`, `_bit_scan_reverse`
#endif
#endif // STRINGZILLA_ARCH_X8664_WESTMERE_ || STRINGZILLA_ARCH_X8664_GOLDMONT_
#if STRINGZILLA_ARCH_ARM64_NEON_
#if !defined(_MSC_VER)
#include <arm_acle.h>
#endif
#include <arm_neon.h>
#endif // STRINGZILLA_ARCH_ARM64_NEON_
#if STRINGZILLA_ARCH_ARM64_SVE_
#if !defined(_MSC_VER)
#include <arm_sve.h>
#endif
#endif // STRINGZILLA_ARCH_ARM64_SVE_

#ifdef __cplusplus
extern "C" {
#endif

typedef float sz_f32_t;  // 32-bit floating-point number
typedef double sz_f64_t; // 64-bit floating-point number

/*  Let's infer the integer types or pull them from LibC, if that is allowed by the user. */
#if STRINGZILLA_WITH_LIBC
typedef int8_t sz_i8_t;       // Always 8 bits
typedef uint8_t sz_u8_t;      // Always 8 bits
typedef int16_t sz_i16_t;     // Always 16 bits
typedef uint16_t sz_u16_t;    // Always 16 bits
typedef int32_t sz_i32_t;     // Always 32 bits
typedef uint32_t sz_u32_t;    // Always 32 bits
typedef uint64_t sz_u64_t;    // Always 64 bits
typedef int64_t sz_i64_t;     // Always 64 bits
typedef size_t sz_size_t;     // Pointer-sized unsigned integer, 32 or 64 bits
typedef ptrdiff_t sz_ssize_t; // Signed version of `sz_size_t`, 32 or 64 bits

#else // if !STRINGZILLA_WITH_LIBC:

/**
 *  @brief Even when LibC is not available, compiler macros let us infer the size of integer types.
 *
 *  The C standard doesn't specify the signedness of @c char. On x86 @c char is signed by default
 *  while on Arm it is unsigned by default. That's why we don't define @c sz_char_t and generally
 *  use explicit @c sz_i8_t and @c sz_u8_t.
 *
 *  @see Common Predefined Macros: https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
 */
#if defined(__INT8_TYPE__)
typedef __INT8_TYPE__ sz_i8_t;
#else
typedef signed char sz_i8_t;
#endif
#if defined(__UINT8_TYPE__)
typedef __UINT8_TYPE__ sz_u8_t;
#else
typedef unsigned char sz_u8_t;
#endif
#if defined(__INT16_TYPE__)
typedef __INT16_TYPE__ sz_i16_t;
#else
typedef short sz_i16_t;
#endif
#if defined(__UINT16_TYPE__)
typedef __UINT16_TYPE__ sz_u16_t;
#else
typedef unsigned short sz_u16_t;
#endif
#if defined(__INT32_TYPE__)
typedef __INT32_TYPE__ sz_i32_t;
#else
typedef int sz_i32_t;
#endif
#if defined(__UINT32_TYPE__)
typedef __UINT32_TYPE__ sz_u32_t;
#else
typedef unsigned int sz_u32_t;
#endif
#if defined(__INT64_TYPE__)
typedef __INT64_TYPE__ sz_i64_t;
#else
typedef long long sz_i64_t;
#endif
#if defined(__UINT64_TYPE__)
typedef __UINT64_TYPE__ sz_u64_t;
#else
typedef unsigned long long sz_u64_t;
#endif

/**
 *  @brief Now we need to redefine the @c size_t. Microsoft Visual C++ (MSVC) typically follows the
 *      LLP64 data model on 64-bit platforms, where integers, pointers, and longs differ in size.
 *
 *  GCC and Clang on 64-bit Unix-like systems typically follow the LP64 model instead:
 *
 *  @verbatim
 *                   LLP64 (MSVC)   LP64 (GCC, Clang)
 *  int              32 bits        32 bits
 *  long             32 bits        64 bits
 *  long long        64 bits        64 bits
 *  pointer, size_t  64 bits        64 bits
 *  @endverbatim
 *
 *  @see Abstract Data Models: https://learn.microsoft.com/en-us/windows/win32/winprog64/abstract-data-models
 */
#if STRINGZILLA_ARCH_64BIT_
typedef sz_u64_t sz_size_t;  // ? Preferred over the `__SIZE_TYPE__` and `__UINTMAX_TYPE__` macros
typedef sz_i64_t sz_ssize_t; // ? Preferred over the `__PTRDIFF_TYPE__` and `__INTMAX_TYPE__` macros
#else
typedef sz_u32_t sz_size_t;  // ? Preferred over the `__SIZE_TYPE__` and `__UINTMAX_TYPE__` macros
typedef sz_i32_t sz_ssize_t; // ? Preferred over the `__PTRDIFF_TYPE__` and `__INTMAX_TYPE__` macros
#endif // STRINGZILLA_ARCH_64BIT_

/*  Freestanding GCC and Clang still emit calls to these three, so every environment provides them.
 *  Their length is the compiler's own @c size_t, which may differ from @c sz_size_t in spelling. */
#if defined(__SIZE_TYPE__)
void *memcpy(void *target, void const *source, __SIZE_TYPE__ length);
void *memmove(void *target, void const *source, __SIZE_TYPE__ length);
void *memset(void *target, int value, __SIZE_TYPE__ length);
#else
void *memcpy(void *target, void const *source, sz_size_t length);
void *memmove(void *target, void const *source, sz_size_t length);
void *memset(void *target, int value, sz_size_t length);
#endif
#endif // STRINGZILLA_WITH_LIBC

/** Defines @c STRINGZILLA_NULL, analogous to @c NULL. The default often comes from locale.h,
 *  stddef.h, stdio.h, stdlib.h, string.h, time.h, or wchar.h. */
#if defined(__cplusplus)
#define STRINGZILLA_NULL nullptr
#define STRINGZILLA_NULL_CHAR nullptr
#else
#define STRINGZILLA_NULL ((void *)0)
#define STRINGZILLA_NULL_CHAR ((char *)0)
#endif

#define STRINGZILLA_SIZE_MAX ((sz_size_t)(-1))
#define STRINGZILLA_SSIZE_MAX ((sz_ssize_t)(STRINGZILLA_SIZE_MAX >> 1))
#define STRINGZILLA_SSIZE_MIN ((sz_ssize_t)(-STRINGZILLA_SSIZE_MAX - 1))

/** Default alignment policy for padding and allocation-size heuristics. */
#if !defined(STRINGZILLA_DEFAULT_ALIGNMENT)
#if defined(__s390x__)
#define STRINGZILLA_DEFAULT_ALIGNMENT 256
#elif STRINGZILLA_ARCH_WASM_
#define STRINGZILLA_DEFAULT_ALIGNMENT 64
#else
#define STRINGZILLA_DEFAULT_ALIGNMENT 128
#endif
#endif
#if STRINGZILLA_DEFAULT_ALIGNMENT < 64 || (STRINGZILLA_DEFAULT_ALIGNMENT & (STRINGZILLA_DEFAULT_ALIGNMENT - 1))
#error "STRINGZILLA_DEFAULT_ALIGNMENT must be a power of two and at least 64 bytes"
#endif

enum { sz_default_alignment_k = STRINGZILLA_DEFAULT_ALIGNMENT };

/** Compile-time assert akin to C++ @c static_assert. Uses the native assertion where available
 *  (C++11 @c static_assert, C11 @c _Static_assert); the older-C typedef fallback must sit at file
 *  scope to stay clear of @c -Wunused-local-typedef. */
#if STRINGZILLA_CXX_STANDARD_ >= 201103L
#define sz_static_assert_(condition, name) static_assert(condition, #name)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define sz_static_assert_(condition, name) _Static_assert(condition, #name)
#elif defined(_MSC_VER)
#define sz_static_assert_(condition, name) static_assert(condition, #name)
#else
#define sz_static_assert_(condition, name) typedef char sz_static_assert_##name[(condition) ? 1 : -1]
#endif

sz_static_assert_(sizeof(sz_size_t) == sizeof(void *), sz_size_t_must_be_pointer_size);
sz_static_assert_(sizeof(sz_ssize_t) == sizeof(void *), sz_ssize_t_must_be_pointer_size);

typedef unsigned char sz_byte_t;            // A byte is an 8-bit unsigned integer
typedef char *sz_ptr_t;                     // A type alias for `char *`
typedef char const *sz_cptr_t;              // A type alias for `char const *`
typedef sz_i8_t sz_error_cost_t;            // Character mismatch cost for fuzzy matching functions
typedef sz_u16_t sz_error_cost_magnitude_t; // The smallest type that can hold unsigned `abs(sz_error_cost_t {})`

/** Eight booleans in one byte, item 0 in the least significant bit, the order of Arrow and cuDF
 *  validity masks and the reverse of NumKong's @c nk_u1x8_t. */
typedef sz_u8_t sz_u1x8_t;

struct sz_hash_state_t;           // Forward declaration of a hash state structure
struct sz_sha256_state_t;         // Forward declaration of a SHA256 hash state structure
struct sz_aes256_key_t;           // Forward declaration of an AES-256 round-key schedule
struct sz_aes256_gcm_key_t;       // Forward declaration of an AES-256-GCM key, schedule plus hash powers
struct sz_aes256_gcm_state_t;     // Forward declaration of the payload both streaming directions share
struct sz_aes256_gcm_encryptor_t; // Forward declaration of an AES-256-GCM sealing state
struct sz_aes256_gcm_decryptor_t; // Forward declaration of an AES-256-GCM opening state
struct sz_substrings_bm25_t;      // Forward declaration of BM25's continuous parameters
struct sz_levenshtein_engine_t;   // Forward declaration of a batch of prepared Levenshtein queries
struct sz_ngrams_engine_t;        // Forward declaration of prepared window tables and the scratch both verbs grow
struct sz_substrings_engine_t;    // Forward declaration of a compiled multi-pattern vocabulary
struct sz_regex_engine_t;         // Forward declaration of a compiled vocabulary of regular expressions
struct sz_subdocuments_engine_t;  // Forward declaration of a compiled vocabulary of JSONPath or XPath queries
typedef sz_size_t sz_pgram_t;     // "Pointer-sized N-gram" of a string

/**
 *  @brief Simple boolean type, until @c _Bool in C 99 and @c true and @c false in C 23.
 *  @see Using boolean values in C: https://stackoverflow.com/questions/1921539/using-boolean-values-in-c
 */
typedef enum { sz_false_k = 0, sz_true_k = 1 } sz_bool_t;

/**
 *  @brief Describes the result of a comparison, like @c std::strong_ordering in C++20.
 *  @see std::strong_ordering: https://en.cppreference.com/w/cpp/utility/compare/strong_ordering
 */
typedef enum { sz_less_k = -1, sz_equal_k = 0, sz_greater_k = 1 } sz_ordering_t;

/**
 *  @brief Describes the alignment scope for string similarity algorithms.
 *  @sa sz_similarity_global_k, sz_similarity_local_k
 */
typedef enum sz_similarity_locality_t {
    sz_similarity_global_k = 0,
    sz_similarity_local_k = 1
} sz_similarity_locality_t;

/**
 *  @brief Describes the alignment objective for string similarity algorithms.
 *  @sa sz_minimize_distance_k, sz_maximize_score_k
 */
typedef enum sz_similarity_objective_t {
    sz_minimize_distance_k = 0,
    sz_maximize_score_k = 1
} sz_similarity_objective_t;

/**
 *  @brief Describes how a similarity kernel packs work across SIMD lanes or a GPU warp's threads.
 *
 *  @b one_pair_per_walk_k vectorizes within a single (query, candidate) pair, in anti-diagonal
 *  lanes — the intra-sequence form, used for one-off pairs and the ragged tail of a cross-product.
 *
 *  @b candidates_across_lanes_k vectorizes across many candidates of one shared query, one
 *  candidate per lane — the inter-sequence form that keeps every lane busy regardless of string
 *  length, the cross-product workhorse.
 *
 *  @sa sz_packing_one_pair_per_walk_k, sz_packing_candidates_across_lanes_k
 */
typedef enum sz_similarity_packing_t {
    sz_packing_one_pair_per_walk_k = 0,
    sz_packing_candidates_across_lanes_k = 1
} sz_similarity_packing_t;

/**
 *  @brief Describes the cost model for gap opening vs extension in string similarity algorithms.
 *  @sa sz_gaps_linear_k, sz_gaps_affine_k
 */
typedef enum sz_similarity_gaps_t {

    /** Linear costs require us to build only 1 DP matrix. */
    sz_gaps_linear_k = 1,

    /** Affine costs require us to build 3 DP matrices. */
    sz_gaps_affine_k = 3
} sz_similarity_gaps_t;

/** Unicode normalization form selector, see <tt>utf8_norm.h</tt>. */
typedef enum sz_normal_form_t {

    /** Canonical decomposition. */
    sz_normal_form_nfd_k = 0,

    /** Canonical decomposition followed by canonical composition. */
    sz_normal_form_nfc_k = 1,

    /** Compatibility decomposition. */
    sz_normal_form_nfkd_k = 2,

    /** Compatibility decomposition followed by canonical composition. */
    sz_normal_form_nfkc_k = 3,

    /** NFKC with full case folding and without default-ignorable codepoints, recomposed to NFC:
     *  the Unicode toNFKC_Casefold that caseless identifier and search matching rely on. */
    sz_normal_form_nfkc_casefold_k = 4,
} sz_normal_form_t;

/** Which symbols a batch counts, as the alphabet picks the transpose and the mask layout alike. */
typedef enum sz_levenshtein_symbol_t {

    /** Every byte is its own symbol, and a distance counts bytes. */
    sz_levenshtein_bytes_k = 0,

    /** Every UTF-8 rune is one symbol, each maximal ill-formed subpart decoding to one U+FFFD. */
    sz_levenshtein_runes_k = 1,
} sz_levenshtein_symbol_t;

/** Whether a vocabulary matches needles byte-for-byte, or folds both sides to one case first. */
typedef enum sz_case_sensitivity_t {

    /** Byte-exact matching; needles may be arbitrary bytes, including malformed UTF-8. */
    sz_cased_k = 0,

    /** Full Unicode case folding as <tt>CaseFolding.txt</tt> defines it; needles must be valid UTF-8. */
    sz_uncased_k = 1,
} sz_case_sensitivity_t;

/** One reported match, locating it by haystack, by needle, and by byte span. */
typedef struct sz_match_t {

    /** Which haystack of the collection this match was found in. */
    sz_size_t haystack_index;

    /** Which needle of the vocabulary matched: a substring, a pattern or a path. */
    sz_size_t needle_index;

    /** Where the match starts inside that haystack, in its own bytes. */
    sz_size_t byte_offset;

    /** Haystack bytes the match spans, which folding can make differ from the needle's length. */
    sz_size_t byte_length;
} sz_match_t;

/** The document grammar a subdocuments engine reads, and the path language that addresses it. */
typedef enum sz_subdocuments_format_t {

    /** RFC 8259 documents, addressed by RFC 9535 JSONPath. */
    sz_subdocuments_json_k = 0,

    /** XML 1.0 documents, addressed by XPath 1.0 along forward axes. */
    sz_subdocuments_xml_k = 1,
} sz_subdocuments_format_t;

/** How a subdocuments engine reads what RFC 8259 leaves open, combined by bitwise or. */
typedef enum sz_subdocuments_options_t {

    /** Every member a name selector names is entered. */
    sz_subdocuments_options_none_k = 0,

    /** A name selector enters only the first member of an object with that name, so each field
     *  a path names is at most one node. */
    sz_subdocuments_duplicate_names_first_k = 1,
} sz_subdocuments_options_t;

/** How matches that share bytes resolve: reported in full, or thinned to a leftmost run. */
typedef enum sz_substrings_overlap_policy_t {

    /** Every match of every needle, including ones that share bytes and ones nested in others. */
    sz_substrings_overlapping_k = 0,

    /** Matches sharing no bytes: earliest start, then longest span, then lower needle index. */
    sz_substrings_leftmost_longest_k = 1,

    /** Matches sharing no bytes: earliest start, then lower needle index, whatever the lengths. */
    sz_substrings_leftmost_first_k = 2,
} sz_substrings_overlap_policy_t;

/**
 *  @brief Outcome of every StringZilla call that can fail: zero on success, negative when the
 *      call failed and its outputs hold no result. Positive values are reserved for results
 *      with a caveat.
 *
 *  Statuses fall into four categories, with no interruptions to report. Resources:
 *  @c sz_bad_alloc_k, @c sz_missing_gpu_k, @c sz_device_code_mismatch_k, @c sz_missing_kernel_k
 *  and @c sz_missing_library_k. Input: @c sz_invalid_utf8_k, @c sz_contains_duplicates_k,
 *  @c sz_unexpected_dimensions_k, @c sz_device_memory_mismatch_k, @c sz_authentication_failed_k
 *  and @c sz_invalid_syntax_k. State: @c sz_overflow_risk_k. Defects: @c sz_status_unknown_k. Any
 *  verb that allocates, binds a device or touches a file may also return a Resources status. A
 *  verb's own block lists only what its contract adds.
 */
typedef enum sz_status_t {

    /** For algorithms that return a status, indicates that the operation was successful. */
    sz_success_k = 0,

    /** For algorithms that require memory allocation, indicates that the allocation failed. */
    sz_bad_alloc_k = -10,

    /** For algorithms that require UTF8 input, indicates that the input is invalid. */
    sz_invalid_utf8_k = -12,

    /** For algorithms taking collections of unique elements, reports the presence of duplicates. */
    sz_contains_duplicates_k = -13,

    /** For algorithms on large inputs, reports the need to upcast the logic to larger types. */
    sz_overflow_risk_k = -14,

    /** For algorithms with multi-stage pipelines indicates input/output size mismatch. */
    sz_unexpected_dimensions_k = -15,

    /** GPU support is missing in the library. */
    sz_missing_gpu_k = -16,

    /** Backend-device mismatch: e.g., GPU kernel with CPU/default executor or vice versa. */
    sz_device_code_mismatch_k = -17,

    /** Device memory mismatch: e.g., GPU kernel requires unified/device-accessible memory. */
    sz_device_memory_mismatch_k = -18,

    /** An authenticated decryption saw a tag that does not match the ciphertext it accompanies. */
    sz_authentication_failed_k = -19,

    /** No capability in the capability mask has this kernel. */
    sz_missing_kernel_k = -20,

    /** A dispatch point or finder called from a header-only build, which links no library. */
    sz_missing_library_k = -21,

    /** A pattern, path or document breaking its grammar, or using a construct no automaton runs. */
    sz_invalid_syntax_k = -22,

    /** A sink-hole status for unknown errors. */
    sz_status_unknown_k = -1,
} sz_status_t;

/** Static English description of @p status, behind @c sz_status_name. */
STRINGZILLA_CONSTEXPR char const *sz_status_name_(sz_status_t status) {
    switch (status) {
    case sz_success_k: return "success";
    case sz_bad_alloc_k: return "out of memory";
    case sz_invalid_utf8_k: return "input is not valid UTF-8";
    case sz_contains_duplicates_k: return "collection contains duplicates";
    case sz_overflow_risk_k: return "input too large for the counters";
    case sz_unexpected_dimensions_k: return "unexpected dimensions";
    case sz_missing_gpu_k: return "no GPU of this vendor";
    case sz_device_code_mismatch_k: return "no kernel ran on this device";
    case sz_device_memory_mismatch_k: return "memory the device cannot reach";
    case sz_authentication_failed_k: return "authentication tag mismatch";
    case sz_missing_kernel_k: return "no kernel for these capabilities";
    case sz_missing_library_k: return "the StringZilla library is not linked; call a capability's kernel or link it";
    case sz_invalid_syntax_k: return "pattern, path or document breaks its grammar";
    case sz_status_unknown_k: return "unknown failure";
    }
    return "an unrecognized status";
}

/** Static English description of @p status, never null. */
STRINGZILLA_API char const *sz_status_name(sz_status_t status);

#if STRINGZILLA_HEADER_ONLY
STRINGZILLA_API char const *sz_status_name(sz_status_t status) { return sz_status_name_(status); }
#endif

/** Compares an explicit-length string against a NUL-terminated literal. */
STRINGZILLA_CONSTEXPR int sz_same_literal_(char const *name, sz_size_t length, char const *literal) {
    sz_size_t position = 0;
    for (; position != length; ++position)
        if (literal[position] == '\0' || name[position] != literal[position]) return 0;
    return literal[position] == '\0';
}

/**
 *  @brief Describes the length of a UTF-8 @b rune / character / codepoint in bytes, from 1 to 4.
 *  @see UTF-8: https://en.wikipedia.org/wiki/UTF-8
 */
typedef enum sz_rune_length_t {

    /** Invalid UTF8 character. */
    sz_rune_invalid_k = 0,

    /** 1-byte UTF8 character. */
    sz_rune_1byte_k = 1,

    /** 2-byte UTF8 character. */
    sz_rune_2bytes_k = 2,

    /** 3-byte UTF8 character. */
    sz_rune_3bytes_k = 3,

    /** 4-byte UTF8 character. */
    sz_rune_4bytes_k = 4,
} sz_rune_length_t;

/**
 *  @brief Stores a single UTF-8 @b rune / character / codepoint unpacked into @b UTF-32.
 *
 *  The underlying numeric type holds 4 bytes, with over 4 billion possible states, but:
 *
 *  - UTF-8 in its largest 4-byte form has only 3 + 6 + 6 + 6 = 21 bits of usable space, for
 *    2 million states.
 *  - Unicode, in turn, has only @b 1'114'112 possible code points from U+0000 to U+10FFFF.
 *  - Of those, in Unicode 16, only @b 155'063 are assigned characters, just over 17 bits' worth.
 *
 *  That's @b 0.004% of the 32-bit space, so sparse data-structures suit UTF-8 oriented algorithms.
 *
 *  @see UTF-32: https://en.wikipedia.org/wiki/UTF-32
 */
typedef sz_u32_t sz_rune_t;

/** The Unicode @b replacement character U+FFFD, emitted once per maximal ill-formed UTF-8 run. */
enum { sz_rune_replacement_k = 0xFFFD };

STRINGZILLA_CONSTEXPR sz_rune_t sz_rune_perfect_hash(sz_rune_t rune) {
    // TODO: A perfect hashing scheme can be constructed to map a 32-bit rune into an 18-bit representation,
    // TODO: that can fit all of the unique values in the Unicode 16 standard.
    return rune;
}

/** The unsigned value of the byte at @p offset. A <tt>char const *</tt> cannot be reinterpreted as
 *  <tt>sz_u8_t const *</tt> inside a constant expression, and that qualifier is what carries these helpers
 *  into device code, so the sign is taken off one byte at a time instead. */
STRINGZILLA_CONSTEXPR sz_u8_t sz_utf8_byte_at_(sz_cptr_t utf8, sz_size_t offset) { return (sz_u8_t)utf8[offset]; }

/** Whether @p byte is a UTF-8 continuation byte (<tt>0x80..0xBF</tt>). The single low-level predicate
 *  every decode path shares, so @c sz_rune_decode and @c sz_utf8_maximal_subpart_ can never
 *  disagree on validity. */
STRINGZILLA_CONSTEXPR sz_bool_t sz_utf8_is_continuation_(sz_u8_t byte) { return (sz_bool_t)((byte & 0xC0) == 0x80); }

/** Whether @p second is a valid @b first continuation for lead byte @p lead: a continuation byte
 *  that also satisfies the E0/ED/F0/F4 overlong/surrogate/range constraint (Unicode Table 3-7). For
 *  C2..DF and the unconstrained 3/4-byte leads it is just the continuation test. Shared by
 *  @c sz_rune_decode and @c sz_utf8_maximal_subpart_ so the "is byte 2 in range" rule lives in
 *  exactly one place. */
STRINGZILLA_CONSTEXPR sz_bool_t sz_utf8_first_continuation_ok_(sz_u8_t lead, sz_u8_t second) {
    if ((second & 0xC0) != 0x80) return sz_false_k;
    if (lead == 0xE0) return (sz_bool_t)(second >= 0xA0); // overlong 3-byte
    if (lead == 0xED) return (sz_bool_t)(second < 0xA0);  // surrogate U+D800..U+DFFF
    if (lead == 0xF0) return (sz_bool_t)(second >= 0x90); // overlong 4-byte
    if (lead == 0xF4) return (sz_bool_t)(second < 0x90);  // > U+10FFFF
    return sz_true_k;
}

/** Bounds-checked, validating UTF-8 decode. Returns the codepoint's 1-4 byte length in @p rune,
 *  or @c sz_rune_invalid_k (0) when the bytes at @p utf8 do not begin a well-formed,
 *  shortest-form, non-surrogate, in-range codepoint (a truncated trailing sequence before
 *  @p utf8_end included). The single authority for "is this a foldable/normalizable rune"; the
 *  decode-side mirror of @c sz_rune_encode. On failure use @c sz_utf8_maximal_subpart_ for how
 *  many bytes the resulting U+FFFD consumes. */
STRINGZILLA_CONSTEXPR sz_rune_length_t sz_rune_decode(sz_cptr_t utf8, sz_cptr_t utf8_end, sz_rune_t *rune) {
    sz_size_t const available = (sz_size_t)(utf8_end - utf8);
    sz_u8_t const lead = sz_utf8_byte_at_(utf8, 0);
    if (lead < 0x80) {
        *rune = lead;
        return sz_rune_1byte_k;
    }
    if (lead < 0xC2) return sz_rune_invalid_k; // continuation byte, or C0/C1 (overlong 2-byte)
    if (lead < 0xE0) {                         // C2..DF
        if (available < 2 || !sz_utf8_first_continuation_ok_(lead, sz_utf8_byte_at_(utf8, 1))) return sz_rune_invalid_k;
        *rune = (sz_rune_t)(lead & 0x1F) << 6 | (sz_utf8_byte_at_(utf8, 1) & 0x3F);
        return sz_rune_2bytes_k;
    }
    if (lead < 0xF0) { // E0..EF
        if (available < 3 || !sz_utf8_first_continuation_ok_(lead, sz_utf8_byte_at_(utf8, 1)) ||
            !sz_utf8_is_continuation_(sz_utf8_byte_at_(utf8, 2)))
            return sz_rune_invalid_k;
        *rune = (sz_rune_t)(lead & 0x0F) << 12 | (sz_rune_t)(sz_utf8_byte_at_(utf8, 1) & 0x3F) << 6 |
                (sz_utf8_byte_at_(utf8, 2) & 0x3F);
        return sz_rune_3bytes_k;
    }
    if (lead <= 0xF4) { // F0..F4
        if (available < 4 || !sz_utf8_first_continuation_ok_(lead, sz_utf8_byte_at_(utf8, 1)) ||
            !sz_utf8_is_continuation_(sz_utf8_byte_at_(utf8, 2)) ||
            !sz_utf8_is_continuation_(sz_utf8_byte_at_(utf8, 3)))
            return sz_rune_invalid_k;
        *rune = (sz_rune_t)(lead & 0x07) << 18 | (sz_rune_t)(sz_utf8_byte_at_(utf8, 1) & 0x3F) << 12 |
                (sz_rune_t)(sz_utf8_byte_at_(utf8, 2) & 0x3F) << 6 | (sz_utf8_byte_at_(utf8, 3) & 0x3F);
        return sz_rune_4bytes_k;
    }
    return sz_rune_invalid_k; // F5..FF
}

/**
 *  @brief Encode a UTF-32 codepoint to UTF-8 (1-4 bytes).
 *  @return Byte count, or @c sz_rune_invalid_k if invalid.
 */
STRINGZILLA_CONSTEXPR sz_rune_length_t sz_rune_encode(sz_rune_t rune, sz_u8_t *utf8s) {
    if (rune <= 0x7F) {
        utf8s[0] = (sz_u8_t)rune;
        return sz_rune_1byte_k;
    }
    else if (rune <= 0x7FF) {
        utf8s[0] = (sz_u8_t)(0xC0 | (rune >> 6));
        utf8s[1] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_rune_2bytes_k;
    }
    else if (rune <= 0xFFFF) {
        if (rune >= 0xD800 && rune <= 0xDFFF) return sz_rune_invalid_k; // reject surrogates
        utf8s[0] = (sz_u8_t)(0xE0 | (rune >> 12));
        utf8s[1] = (sz_u8_t)(0x80 | ((rune >> 6) & 0x3F));
        utf8s[2] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_rune_3bytes_k;
    }
    else if (rune <= 0x10FFFF) {
        utf8s[0] = (sz_u8_t)(0xF0 | (rune >> 18));
        utf8s[1] = (sz_u8_t)(0x80 | ((rune >> 12) & 0x3F));
        utf8s[2] = (sz_u8_t)(0x80 | ((rune >> 6) & 0x3F));
        utf8s[3] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_rune_4bytes_k;
    }
    return sz_rune_invalid_k;
}

/** Locate the first ill-formed byte in <tt>[text, text+length)</tt>; @c STRINGZILLA_NULL_CHAR if
 *  entirely well-formed UTF-8. */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_utf8_find_malformed(sz_cptr_t text, sz_size_t length) {
    sz_cptr_t const end = text + length;
    while (text < end) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode(text, end, &rune);
        if (consumed == sz_rune_invalid_k) return text;
        text += consumed;
    }
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Get the UTF-8 sequence length from a lead byte, branchlessly.
 *
 *  Three comparisons rather than a lookup: 0x00-0xBF is 1 (ASCII and, for robustness, stray
 *  continuation bytes treated as single bytes), 0xC0-0xDF is 2, 0xE0-0xEF is 3, and 0xF0 and
 *  above is 4. The same form @c sz_rune_decode_unchecked uses, so the two can never disagree on
 *  how far a lead byte advances.
 */
STRINGZILLA_CONSTEXPR sz_size_t sz_utf8_lead_length(sz_u8_t lead_byte) {
    return (sz_size_t)(1 + (lead_byte >= 0xC0U) + (lead_byte >= 0xE0U) + (lead_byte >= 0xF0U));
}

/**
 *  @brief Tiny string-view structure. It's a Plain-Old Datatype @b (POD) type, unlike the
 *      @c std::string_view.
 *  @see PODType: https://en.cppreference.com/w/cpp/named_req/PODType
 */
typedef struct sz_string_view_t {
    sz_cptr_t start;
    sz_size_t length;
} sz_string_view_t;

#pragma region Character Sets

/**
 *  @brief Semi-opaque bit-set for 256 possible byte values, useful for filtering and search.
 *
 *  It can be filled and probed like this:
 *
 *  @code{.c}
 *      #include <stringzilla/types.h>
 *      int main() {
 *          char const *alphabet = "abcdefghijklmnopqrstuvwxyz";
 *          sz_byteset_t byteset;
 *          sz_byteset_init(&byteset);
 *          for (sz_size_t i = 0; i < 26; ++i)
 *              sz_byteset_add(&byteset, alphabet[i]);
 *          return sz_byteset_contains(&byteset, 'a') && !sz_byteset_contains(&byteset, 'A') ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @sa sz_byteset_init, sz_byteset_add, sz_byteset_contains, sz_byteset_invert
 */
typedef union sz_byteset_t {
    sz_u64_t _u64s[4];
    sz_u32_t _u32s[8];
    sz_u16_t _u16s[16];
    sz_u8_t _u8s[32];
} sz_byteset_t;

/** Initializes a bit-set to an empty collection, meaning - all characters are banned. */
STRINGZILLA_CONSTEXPR void sz_byteset_init(sz_byteset_t *s) {
    s->_u64s[0] = s->_u64s[1] = s->_u64s[2] = s->_u64s[3] = 0;
}

/** Initializes a bit-set to all ASCII characters. */
STRINGZILLA_CONSTEXPR void sz_byteset_init_ascii(sz_byteset_t *s) {
    s->_u64s[0] = s->_u64s[1] = 0xFFFFFFFFFFFFFFFFull;
    s->_u64s[2] = s->_u64s[3] = 0;
}

/** Adds a character to the set and accepts @b unsigned integers. */
STRINGZILLA_CONSTEXPR void sz_byteset_add_u8(sz_byteset_t *s, sz_u8_t c) { s->_u64s[c >> 6] |= (1ull << (c & 63u)); }

/** Adds a character to the set. Consider @b sz_byteset_add_u8. */
STRINGZILLA_INLINE void sz_byteset_add(sz_byteset_t *s, char c) { sz_byteset_add_u8(s, *(sz_u8_t *)(&c)); } // bitcast

/** Checks if the set contains a given character and accepts @b unsigned integers. */
STRINGZILLA_CONSTEXPR sz_bool_t sz_byteset_contains_u8(sz_byteset_t const *s, sz_u8_t c) {
    // Checking the bit can be done in different ways:
    // - (s->_u64s[c >> 6] & (1ull << (c & 63u))) != 0
    // - (s->_u32s[c >> 5] & (1u << (c & 31u))) != 0
    // - (s->_u16s[c >> 4] & (1u << (c & 15u))) != 0
    // - (s->_u8s[c >> 3] & (1u << (c & 7u))) != 0
    return (sz_bool_t)((s->_u64s[c >> 6] & (1ull << (c & 63u))) != 0);
}

/** Checks if the set contains a given character. Consider @b sz_byteset_contains_u8. */
STRINGZILLA_INLINE sz_bool_t sz_byteset_contains(sz_byteset_t const *s, char c) {
    return sz_byteset_contains_u8(s, *(sz_u8_t *)(&c)); // bitcast
}

/** Inverts the contents of the set, so allowed characters get disallowed, and vice versa. */
STRINGZILLA_CONSTEXPR void sz_byteset_invert(sz_byteset_t *s) {
    s->_u64s[0] ^= 0xFFFFFFFFFFFFFFFFull, s->_u64s[1] ^= 0xFFFFFFFFFFFFFFFFull, //
        s->_u64s[2] ^= 0xFFFFFFFFFFFFFFFFull, s->_u64s[3] ^= 0xFFFFFFFFFFFFFFFFull;
}

#pragma endregion

#pragma region Memory Management

/** Native CUDA cudaStream_t, ROCm hipStream_t, or Metal MTLCommandQueue handle.
 *  Null selects the current device's default CUDA/ROCm stream, Metal's default device queue,
 *  or synchronous execution on the CPU. The caller owns the handle. */
typedef void *sz_stream_t;

typedef void *(*sz_allocate_t)(sz_size_t bytes, void *handle, sz_stream_t stream);
typedef void *(*sz_reallocate_t)(void *pointer, sz_size_t old_bytes, sz_size_t new_bytes, void *handle,
                                 sz_stream_t stream);
typedef void (*sz_free_t)(void *pointer, sz_size_t bytes, void *handle, sz_stream_t stream);

/**
 *  @brief Some complex pattern matching algorithms may require memory allocations. This structure
 *      passes the memory allocator to those functions.
 *
 *  Every function takes the caller's stream, which on a GPU names the device a block is made on and
 *  the work a release waits behind; host allocators ignore it, and on the CPU it is null. The
 *  optional @c reallocate keeps the first <tt>min(old_bytes, new_bytes)</tt> bytes, resizing the block
 *  where it lies when it can and moving it otherwise, and leaves the block as it was on failure.
 *  Without it, a block that must change size moves through @c allocate and @c free.
 *
 *  @sa sz_allocator_init_arena, sz_allocator_init_unified_best
 */
typedef struct sz_allocator_t {
    sz_allocate_t allocate;
    sz_reallocate_t reallocate;
    sz_free_t free;
    void *handle;
} sz_allocator_t;

/**
 *  @brief Initializes a memory allocator to use the system default @c malloc and @c free.
 *  @param[out] allocator Memory allocator to initialize.
 *
 *  @note Unlike the C standard library, <tt>malloc(0)</tt> is guaranteed to return a null pointer.
 *  @see malloc: https://en.cppreference.com/w/c/memory/malloc
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k Without libc. Failure preserves the allocator.
 */
STRINGZILLA_INLINE sz_status_t sz_allocator_init_heap(sz_allocator_t *allocator);

/**
 *  @brief Initializes a memory allocator that serves every request from a static-capacity buffer,
 *      @b without any dynamic allocations.
 *  @param[out] allocator Memory allocator to initialize.
 *  @param[in] buffer Buffer to use for allocations.
 *  @param[in] bytes Buffer size, including metadata and alignment padding.
 *  @param[in] alignment Nonzero power-of-two alignment guaranteed for every returned block.
 *
 *  Failure leaves both the allocator and buffer unchanged. Three aligned @c sz_size_t values store
 *  capacity, consumption and alignment inside the buffer.
 *
 *  @par Statuses:
 *  - @c sz_unexpected_dimensions_k Invalid alignment.
 *  - @c sz_bad_alloc_k Missing storage.
 *  - @c sz_overflow_risk_k An overflowing address range.
 */
STRINGZILLA_INLINE sz_status_t sz_allocator_init_arena(sz_allocator_t *allocator, void *buffer, sz_size_t bytes,
                                                       sz_size_t alignment);

/**
 *  @brief Checks if two memory allocators are equivalent.
 *  @param[in] a First memory allocator.
 *  @param[in] b Second memory allocator.
 *  @return True if the allocators are the same, false otherwise.
 */
STRINGZILLA_CONSTEXPR sz_bool_t sz_allocator_equal(sz_allocator_t const *a, sz_allocator_t const *b);

#pragma endregion

#pragma region Helper Structures

/**
 *  @brief Helper structure to simplify work with 16-bit words.
 *  @sa sz_u16_load
 */
typedef union sz_u16_vec_t {
    sz_u16_t u16;
    sz_u8_t u8s[2];
} sz_u16_vec_t;

/**
 *  @brief Helper structure to simplify work with 32-bit words.
 *  @sa sz_u32_load
 */
typedef union sz_u32_vec_t {
    sz_u32_t u32;
    sz_i32_t i32;
    sz_u16_t u16s[2];
    sz_i16_t i16s[2];
    sz_u8_t u8s[4];
    sz_i8_t i8s[4];
} sz_u32_vec_t;

/**
 *  @brief Helper structure to simplify work with 64-bit words.
 *  @sa sz_u64_load
 */
typedef union sz_u64_vec_t {
    sz_u64_t u64;
    sz_i64_t i64;
    sz_u32_t u32s[2];
    sz_i32_t i32s[2];
    sz_u16_t u16s[4];
    sz_i16_t i16s[4];
    sz_u8_t u8s[8];
    sz_i8_t i8s[8];
} sz_u64_vec_t;

/** Helper structure to simplify work with @b 128-bit registers. It can help view the contents as
 *  8-bit, 16-bit, 32-bit, or 64-bit integers, as well as 1x XMM register. */
typedef union STRINGZILLA_MAY_ALIAS_ sz_u128_vec_t {
#if STRINGZILLA_ARCH_X8664_WESTMERE_
    __m128i xmm;
    __m128d xmm_pd;
    __m128 xmm_ps;
#endif
#if STRINGZILLA_ARCH_ARM64_NEON_
    uint8x16_t u8x16;
    uint16x8_t u16x8;
    uint32x4_t u32x4;
    uint64x2_t u64x2;
    float64x2_t f64x2;
    float32x4_t f32x4;
#endif
#if STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_
    __m128i lsx;
#endif
#if STRINGZILLA_ARCH_WASM_V128_
    v128_t v128;
#endif
#if STRINGZILLA_ARCH_PPC64_POWERVSX_
    __vector unsigned char vsx_u8;
    __vector unsigned short vsx_u16;
    __vector unsigned int vsx_u32;
    __vector unsigned long long vsx_u64;
#endif
    sz_f64_t f64s[2];
    sz_f32_t f32s[4];
    sz_u64_t u64s[2];
    sz_i64_t i64s[2];
    sz_u32_t u32s[4];
    sz_i32_t i32s[4];
    sz_u16_t u16s[8];
    sz_i16_t i16s[8];
    sz_u8_t u8s[16];
    sz_i8_t i8s[16];
} sz_u128_vec_t;

/** Helper structure to simplify work with @b 256-bit registers. It can help view the contents as
 *  8-bit, 16-bit, 32-bit, or 64-bit integers, as well as 2x XMM registers or 1x YMM register. */
typedef union STRINGZILLA_MAY_ALIAS_ sz_u256_vec_t {
#if STRINGZILLA_ARCH_X8664_HASWELL_
    __m256i ymm;
    __m256d ymm_pd;
    __m256 ymm_ps;
#endif
#if STRINGZILLA_ARCH_X8664_WESTMERE_
    __m128i xmms[2];
#endif
#if STRINGZILLA_ARCH_ARM64_NEON_
    uint8x16_t u8x16s[2];
    uint16x8_t u16x8s[2];
    uint32x4_t u32x4s[2];
    uint64x2_t u64x2s[2];
#endif
#if STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_
    __m256i lasx;
#endif
#if STRINGZILLA_ARCH_WASM_V128_
    v128_t v128s[2];
#endif
    sz_f64_t f64s[4];
    sz_f32_t f32s[8];
    sz_u64_t u64s[4];
    sz_i64_t i64s[4];
    sz_u32_t u32s[8];
    sz_i32_t i32s[8];
    sz_u16_t u16s[16];
    sz_i16_t i16s[16];
    sz_u8_t u8s[32];
    sz_i8_t i8s[32];
} sz_u256_vec_t;

/** Helper structure to simplify work with @b 512-bit registers. It can help view the contents as
 *  8-bit, 16-bit, 32-bit, or 64-bit integers, as well as 4x XMM registers or 2x YMM registers or
 *  1x ZMM register. */
typedef union STRINGZILLA_MAY_ALIAS_ sz_u512_vec_t {
#if STRINGZILLA_ARCH_X8664_SKYLAKE_
    __m512i zmm;
    __m512d zmm_pd;
    __m512 zmm_ps;
#endif
#if STRINGZILLA_ARCH_X8664_HASWELL_
    __m256i ymms[2];
#endif
#if STRINGZILLA_ARCH_X8664_WESTMERE_
    __m128i xmms[4];
#endif
#if STRINGZILLA_ARCH_ARM64_NEON_
    uint8x16_t u8x16s[4];
    uint16x8_t u16x8s[4];
    uint32x4_t u32x4s[4];
    uint64x2_t u64x2s[4];
#endif
    sz_f64_t f64s[8];
    sz_f32_t f32s[16];
    sz_u64_t u64s[8];
    sz_i64_t i64s[8];
    sz_u32_t u32s[16];
    sz_i32_t i32s[16];
    sz_u16_t u16s[32];
    sz_i16_t i16s[32];
    sz_u8_t u8s[64];
    sz_i8_t i8s[64];

    sz_u128_vec_t u128s[4];
    sz_u128_vec_t u256s[2];
} sz_u512_vec_t;

#pragma endregion

#pragma region Lane Compaction Tables

/** Left-pack table for 32-bit permutes: row <tt>[m]</tt> holds the 8 dword indices that gather the
 *  @c m -selected u64 lanes of 4, each a dword pair, to the front. 512 bytes stay cache-resident;
 *  an 8-wide variant would need 16 KB. */
sz_align_(64) static sz_u32_t const sz_compact4_dword_indices_[16][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 0, 0, 0, 0},
    {4, 5, 0, 0, 0, 0, 0, 0}, {0, 1, 4, 5, 0, 0, 0, 0}, {2, 3, 4, 5, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 0, 0},
    {6, 7, 0, 0, 0, 0, 0, 0}, {0, 1, 6, 7, 0, 0, 0, 0}, {2, 3, 6, 7, 0, 0, 0, 0}, {0, 1, 2, 3, 6, 7, 0, 0},
    {4, 5, 6, 7, 0, 0, 0, 0}, {0, 1, 4, 5, 6, 7, 0, 0}, {2, 3, 4, 5, 6, 7, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7},
};

/** Byte form of @c sz_compact4_dword_indices_ for @c vqtbl2q_u8: row <tt>[m]</tt> holds the 32 byte
 *  indices that gather the @c m -selected u64 lanes of 4 to the front. The first 16 bytes feed
 *  lookup 0, for output lanes 0-1, and the last 16 feed lookup 1, for output lanes 2-3. */
sz_align_(64) static sz_u8_t const sz_compact4_byte_indices_[16][32] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7},
    {24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7},
    {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7},
    {8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
     24, 25, 26, 27, 28, 29, 30, 31, 0,  1,  2,  3,  4,  5,  6,  7},
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
};

/** Population count of every 4-bit mask, for the targets without a scalar popcount instruction. */
sz_align_(64) static sz_u8_t const sz_popcount4_lut_[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

#pragma endregion

#pragma region String Collections API

/** Signature of the @c sz_strings_view_t accessor, writing the view of every string <tt>indices[i]</tt>.
 *  Kernels call it once per block of up to 64 indices, and it must accept any @c count. */
typedef void (*sz_strings_get_t)(void const *handle, sz_size_t const *indices, sz_size_t count,
                                 sz_string_view_t *views);

/** How a collection holds its strings, naming the member of its @c data union to read. */
typedef enum sz_strings_layout_t {

    /** Apache Arrow @c String, where string i spans <tt>[offsets[i], offsets[i + 1])</tt> of the bytes. */
    sz_strings_u32_tape_k = 0,

    /** Apache Arrow @c LargeString, the same with 64-bit offsets. */
    sz_strings_u64_tape_k = 1,

    /** Strings found through a @c sz_strings_get_t accessor, which only the host can call. */
    sz_strings_accessor_k = 2,
} sz_strings_layout_t;

/**
 *  @brief A read-only, ordered collection of borrowed strings, in one of three layouts.
 *
 *  Tapes are what Apache Arrow, the Python @c Strings class and GPU kernels read directly. The
 *  accessor covers every other shape, from arrays of views to the nodes of a parsed tree, at one
 *  indirect call per block of strings. A slice of a tape keeps its parent's offsets, so its first
 *  offset may be above zero.
 *
 *  @sa sz_strings_from_string_views, sz_strings_from_records, sz_strings_from_span
 */
typedef struct sz_strings_view_t {

    /** Which member of @c data describes the strings. */
    sz_strings_layout_t layout;

    /** Number of strings. */
    sz_size_t count;

    /** The description of the strings, read through the member that @c layout names. */
    union {

        /** The accessor of a @c sz_strings_accessor_k collection. */
        struct {

            /** Writes the views of the requested strings. */
            sz_strings_get_t get;

            /** The state @c get reads, like an array of views. */
            void const *handle;
        } accessor;

        /** The tape of a @c sz_strings_u32_tape_k collection. */
        struct {

            /** The bytes of every string, back to back. */
            sz_cptr_t data;

            /** Offsets into @c data, one per string and one past the end of the last string. */
            sz_u32_t const *offsets;
        } u32_tape;

        /** The tape of a @c sz_strings_u64_tape_k collection. */
        struct {

            /** The bytes of every string, back to back. */
            sz_cptr_t data;

            /** Offsets into @c data, one per string and one past the end of the last string. */
            sz_u64_t const *offsets;
        } u64_tape;
    } data;
} sz_strings_view_t;

/**
 *  @brief A writable tape, the output of every collection conversion, in place or not.
 *
 *  Its payload and offsets are two blocks of exactly the sizes they need, so both sizes follow from
 *  the offsets, which the host can always read. A zeroed span is an empty tape holding no blocks.
 */
typedef struct sz_strings_span_t {

    /** Either @c sz_strings_u32_tape_k or @c sz_strings_u64_tape_k, never the accessor. */
    sz_strings_layout_t layout;

    /** Number of strings. */
    sz_size_t count;

    /** The tape, read through the member that @c layout names. */
    union {

        /** The tape of a @c sz_strings_u32_tape_k span. */
        struct {

            /** The bytes of every string, back to back. */
            sz_ptr_t data;

            /** Offsets into @c data, one per string and one past the end of the last string. */
            sz_u32_t *offsets;
        } u32_tape;

        /** The tape of a @c sz_strings_u64_tape_k span. */
        struct {

            /** The bytes of every string, back to back. */
            sz_ptr_t data;

            /** Offsets into @c data, one per string and one past the end of the last string. */
            sz_u64_t *offsets;
        } u64_tape;
    } data;
} sz_strings_span_t;

/** Where every record of an array keeps its string, the handle of @c sz_strings_from_records and
 *  @c sz_strings_from_record_pointers. */
typedef struct sz_strings_crecords_t {

    /** The first record, or the first pointer to a record. */
    void const *records;

    /** Bytes between consecutive records, or between consecutive record pointers. */
    sz_size_t stride;

    /** Byte offset of the @c sz_cptr_t member pointing at the string inside a record. */
    sz_size_t start_member;

    /** Byte offset of the @c sz_size_t member holding the string length inside a record. */
    sz_size_t length_member;
} sz_strings_crecords_t;

/** Read-only NUL-padded rows, like NumPy @c S arrays or @c char fields of structs, each string
 *  ending before its row's trailing NUL bytes. */
typedef struct sz_strings_crows_t {

    /** The first row. */
    sz_cptr_t data;

    /** Bytes between the starts of consecutive rows, at least @c width. */
    sz_size_t stride;

    /** Bytes in every row. */
    sz_size_t width;
} sz_strings_crows_t;

/** Writable NUL-padded rows, laid out like @c sz_strings_crows_t. */
typedef struct sz_strings_rows_t {

    /** The first row. */
    sz_ptr_t data;

    /** Bytes between the starts of consecutive rows, at least @c width. */
    sz_size_t stride;

    /** Bytes in every row. */
    sz_size_t width;
} sz_strings_rows_t;

/**
 *  @brief Describes an array of NUL-terminated strings, like <tt>char const *argv[]</tt>.
 *  @param[in] start The array of strings, which must outlive @p strings.
 *  @param[in] count Number of strings in the array.
 *  @param[out] strings The accessor-backed collection to initialize.
 */
STRINGZILLA_INLINE void sz_strings_from_null_terminated_strings(sz_cptr_t const *start, sz_size_t count,
                                                                sz_strings_view_t *strings);

/**
 *  @brief Describes an array of pointer-length pairs, like a list of borrowed string views.
 *  @param[in] views The array of views, which must outlive @p strings.
 *  @param[in] count Number of views in the array.
 *  @param[out] strings The accessor-backed collection to initialize.
 */
STRINGZILLA_INLINE void sz_strings_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                     sz_strings_view_t *strings);

/**
 *  @brief Describes the string members of fixed-stride records, like an array of structs that each
 *      hold a pointer and a length.
 *  @param[in] records Where the members sit, which must outlive @p strings.
 *  @param[in] count Number of records.
 *  @param[out] strings The accessor-backed collection to initialize.
 */
STRINGZILLA_INLINE void sz_strings_from_records(sz_strings_crecords_t const *records, sz_size_t count,
                                                sz_strings_view_t *strings);

/**
 *  @brief Describes the string members of records reached through an array of pointers, like the
 *      nodes a query over a parsed tree returns.
 *  @param[in] records Where the pointers and members sit, which must outlive @p strings.
 *  @param[in] count Number of record pointers.
 *  @param[out] strings The accessor-backed collection to initialize.
 */
STRINGZILLA_INLINE void sz_strings_from_record_pointers(sz_strings_crecords_t const *records, sz_size_t count,
                                                        sz_strings_view_t *strings);

/**
 *  @brief Describes NUL-padded rows, trimming the trailing NUL bytes of every row.
 *  @param[in] rows Where the rows sit, which must outlive @p strings.
 *  @param[in] count Number of rows.
 *  @param[out] strings The accessor-backed collection to initialize.
 */
STRINGZILLA_INLINE void sz_strings_from_rows(sz_strings_crows_t const *rows, sz_size_t count,
                                             sz_strings_view_t *strings);

/**
 *  @brief Selects the @p count strings at <tt>start + i * step</tt>, copying no string bytes.
 *
 *  A step of one over a tape yields a tape over the same bytes and offsets, leaving @p views
 *  untouched. Every other selection fills @p views and points the accessor of @p slice at them.
 *
 *  @param[in] strings The collection to slice, which must outlive @p slice.
 *  @param[in] start Index of the first selected string.
 *  @param[in] step Distance between selected indices, negative to walk backward.
 *  @param[in] count Number of strings to select.
 *  @param[out] views Room for @p count views, or null for a step of one over a tape.
 *  @param[out] slice The selected strings.
 */
STRINGZILLA_INLINE void sz_strings_slice(sz_strings_view_t const *strings, sz_size_t start, sz_ssize_t step,
                                         sz_size_t count, sz_string_view_t *views, sz_strings_view_t *slice);

/**
 *  @brief Reorders @p strings by the order @c sz_sort_order_best emits, copying no bytes.
 *  @param[in] strings The collection to reorder, which must outlive @p permuted.
 *  @param[in] order Indices into @p strings, possibly fewer than all of them.
 *  @param[in] count Number of indices in @p order.
 *  @param[out] views Receives the view of <tt>strings[order[i]]</tt> for each index.
 *  @param[out] permuted Points its accessor at @p views.
 */
STRINGZILLA_INLINE void sz_strings_permute(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                           sz_string_view_t *views, sz_strings_view_t *permuted);

/**
 *  @brief Views the spans a substrings, regex or subdocuments search located, copying no bytes.
 *  @param[in] haystacks The collection the matches were found in, which must outlive @p strings.
 *  @param[in] matches The located matches, in any order.
 *  @param[in] count Number of matches.
 *  @param[out] views Receives the bytes of <tt>matches[i]</tt> inside its haystack for each match.
 *  @param[out] strings Points its accessor at @p views.
 */
STRINGZILLA_INLINE void sz_strings_from_matches(sz_strings_view_t const *haystacks, sz_match_t const *matches,
                                                sz_size_t count, sz_string_view_t *views, sz_strings_view_t *strings);

/**
 *  @brief Views the tape of @p span, so the output of one conversion can feed the next.
 *  @param[in] span The writable tape to read, which must outlive @p strings.
 *  @param[out] strings The read-only collection to initialize.
 */
STRINGZILLA_INLINE void sz_strings_from_span(sz_strings_span_t const *span, sz_strings_view_t *strings);

/**
 *  @brief Returns both blocks of @p span to the @p allocator they came from, leaving it zeroed.
 *  @param[in,out] span The tape to release, whose offsets the host reads to size both blocks.
 *  @param[in] allocator The allocator both blocks came from.
 *  @param[in] stream Null on the CPU, or the GPU stream the release waits behind.
 */
STRINGZILLA_INLINE void sz_strings_span_free(sz_strings_span_t *span, sz_allocator_t *allocator, sz_stream_t stream);

#pragma endregion

#pragma region Engines and States

/** 64-bit bitmask of the capabilities of one device. */
typedef sz_u64_t sz_capability_t;

/**
 *  @brief The state for incremental construction of a hash.
 *  @sa sz_hash_state_init_best, sz_hash_state_update_best, sz_hash_state_digest_best
 *
 *  @note Uses the @c packed attribute to allow placement at arbitrary addresses without UBSAN
 *      warnings, and plain byte arrays to avoid implicit alignment requirements from SIMD types.
 *      The layout matches @c sz_hash_state_aligned_t for safe casting between them.
 *
 *  @c aes, @c sum and @c ins each hold a @c sz_u512_vec_t, and @c key a @c sz_u128_vec_t.
 */
#if defined(_MSC_VER)
#pragma pack(push, 1)
typedef struct sz_hash_state_t {
    sz_u8_t aes[64];
    sz_u8_t sum[64];
    sz_u8_t ins[64];
    sz_u8_t key[16];
    sz_size_t ins_length;
} sz_hash_state_t;
#pragma pack(pop)
#else
typedef struct __attribute__((packed)) sz_hash_state_t {
    sz_u8_t aes[64];
    sz_u8_t sum[64];
    sz_u8_t ins[64];
    sz_u8_t key[16];
    sz_size_t ins_length;
} sz_hash_state_t;
#endif

/** Bytes in a SHA256 digest, fixed by FIPS 180-4. */
#define STRINGZILLA_SHA256_DIGEST_LENGTH (32)

/** Bytes in a SHA256 message block, fixed by FIPS 180-4. */
#define STRINGZILLA_SHA256_BLOCK_LENGTH (64)

/**
 *  @brief The state for incremental construction of a SHA256 hash.
 *  @sa sz_sha256_state_init_best, sz_sha256_state_update_best, sz_sha256_state_digest_best
 */
typedef struct sz_sha256_state_t {

    /** Message block buffer. */
    sz_u8_t block[STRINGZILLA_SHA256_BLOCK_LENGTH];

    /** Current hash state: 8x 32-bit values. */
    sz_u32_t hash[8];

    /** Total message length in bytes. */
    sz_u64_t total_length;

    /** Current bytes in block (0-63). */
    sz_u8_t block_length;

    /** Rounds the state to 128 bytes. */
    sz_u8_t padding_[23];
} sz_sha256_state_t;

/*  The batched kernels walk an array of these, so the layout is load-bearing rather than
 *  incidental. A power-of-two stride turns lane indexing into a shift, and putting @c block first
 *  gives every lane the same cache-line phase as the array itself — at 112 bytes the phase rotated
 *  per lane, splitting the 64-byte block read across two lines for most of them.
 *  @c block_length is a byte because it never exceeds 63, which also makes the struct identical on
 *  32- and 64-bit builds. Alignment stays natural on purpose: @c malloc only promises 16 bytes, so
 *  demanding 64 would under-align every heap-allocated batch. */
sz_static_assert_(sizeof(sz_sha256_state_t) == 128, sha256_state_is_two_cache_lines);

/** Bytes in an AES block, fixed by FIPS 197. */
#define STRINGZILLA_AES_BLOCK_LENGTH (16)

/** Bytes in an AES-256 secret key. */
#define STRINGZILLA_AES256_KEY_LENGTH (32)

/** Bytes in the nonce both modes accept, as NIST SP 800-38D recommends for Galois/counter mode. */
#define STRINGZILLA_AES256_NONCE_LENGTH (12)

/** Bytes in a Galois/counter mode authentication tag. */
#define STRINGZILLA_AES256_TAG_LENGTH (16)

/** Round keys in an AES-256 schedule: 15 of them, four 32-bit words each. */
#define STRINGZILLA_AES256_ROUND_KEYS (60)

/**
 *  @brief An expanded AES-256 round-key schedule, all that counter mode, or CTR, needs.
 *  @sa sz_aes256_key_init_best, sz_aes256_ctr_xor_best
 *
 *  @note Deliberately separate from @c sz_aes256_gcm_key_t. Folding the Galois hash powers in here
 *      would make every counter-mode caller carry 128 bytes it never reads, and pay eight field
 *      multiplications at key setup for a mode that has no authentication tag.
 */
typedef struct sz_aes256_key_t {

    /** 15 round keys of 4 words each, in encryption order. */
    sz_u32_t round_keys[STRINGZILLA_AES256_ROUND_KEYS];
} sz_aes256_key_t;

/**
 *  @brief An AES-256 schedule together with the Galois hash subkey powers, the key Galois/counter
 *      mode needs. Galois/counter mode is abbreviated GCM, and its hash is GHASH.
 *  @sa sz_aes256_gcm_key_init_best, sz_aes256_gcm_encrypt_best, sz_aes256_gcm_encryptor_init_best
 *
 *  The powers H¹ through H⁸ are derived once per key so the wide backends can aggregate eight
 *  blocks between field reductions. Counter mode never touches them, which is why they live here
 *  rather than in @c sz_aes256_key_t.
 */
typedef struct sz_aes256_gcm_key_t {

    /** The round-key schedule. */
    sz_aes256_key_t block;

    /** H¹ through H⁸, ascending. */
    sz_u8_t powers[8 * STRINGZILLA_AES_BLOCK_LENGTH];
} sz_aes256_gcm_key_t;

/**
 *  @brief What both directions of a chunked transformation carry.
 *  @sa sz_aes256_gcm_encryptor_t, sz_aes256_gcm_decryptor_t
 *
 *  Callers construct an encryptor or a decryptor, never this. It is public because the per-backend
 *  kernels operate on it and the differential tests compare it field by field.
 *
 *  @note The key is embedded rather than referenced. A pointer would save the copy, but nothing in
 *      a transparent C struct can stop a state from outliving the key it points at, and that
 *      failure is silent. The copy costs about two percent of a 16 KB record.
 */
typedef struct sz_aes256_gcm_state_t {

    /** Embedded, so the state cannot outlive its key. */
    sz_aes256_gcm_key_t key;

    /** Running Galois hash value. */
    sz_u8_t accumulator[STRINGZILLA_AES_BLOCK_LENGTH];

    /** Current counter block, big-endian in the low four bytes. */
    sz_u8_t counter[STRINGZILLA_AES_BLOCK_LENGTH];

    /** The encrypted initial counter, folded in at digest time. */
    sz_u8_t tag_mask[STRINGZILLA_AES_BLOCK_LENGTH];

    /** Hash bytes awaiting a full block, associated then ciphertext. */
    sz_u8_t partial[STRINGZILLA_AES_BLOCK_LENGTH];

    /** Current keystream block, carried across chunk boundaries. */
    sz_u8_t keystream[STRINGZILLA_AES_BLOCK_LENGTH];

    /** Total associated bytes absorbed. */
    sz_u64_t associated_length;

    /** Total message bytes transformed. */
    sz_u64_t text_length;

    /** Valid bytes in @c partial, 0 to 15. */
    sz_u8_t buffered;

    /** Bytes of @c keystream already spent, 0 to 16. */
    sz_u8_t keystream_used;
} sz_aes256_gcm_state_t;

/**
 *  @brief Seals a chunked message under Galois/counter mode, or GCM, emitting a tag at the end.
 *  @sa sz_aes256_gcm_encryptor_init_best, sz_aes256_gcm_encryptor_update_best,
 *      sz_aes256_gcm_encryptor_digest_best
 *
 *  Distinct from @c sz_aes256_gcm_decryptor_t so that the direction is carried by the type rather
 *  than by a field. The Galois hash absorbs ciphertext either way - the output buffer when sealing,
 *  the input buffer when opening - and a single state that could be pointed either way makes that a
 *  runtime question the compiler cannot help with. Passing one of these where the other belongs
 *  will not compile.
 *
 *  A chunk boundary must be invisible to the result. The keystream block and the hash block are
 *  both sixteen bytes wide and a caller's chunks are not, so both have to survive between calls -
 *  which is why @c keystream_used exists alongside @c buffered instead of one shared counter.
 */
typedef struct sz_aes256_gcm_encryptor_t {

    /** The shared payload. */
    sz_aes256_gcm_state_t state;
} sz_aes256_gcm_encryptor_t;

/**
 *  @brief Opens a chunked message under Galois/counter mode, or GCM, checking its tag at the end.
 *  @sa sz_aes256_gcm_decryptor_init_best, sz_aes256_gcm_decryptor_update_unverified_best,
 *      sz_aes256_gcm_decryptor_verify_best
 */
typedef struct sz_aes256_gcm_decryptor_t {

    /** The shared payload. */
    sz_aes256_gcm_state_t state;
} sz_aes256_gcm_decryptor_t;

/** The text formats that the escape and unescape families convert strings to and from. */
typedef enum sz_escape_kind_t {

    /** RFC 8259 string bodies: <tt>\\"</tt>, <tt>\\\\</tt>, short forms like <tt>\\n</tt>, and <tt>\\u00XX</tt> for other bytes
     *  below 0x20. */
    sz_escape_json_k = 0,

    /** Two lowercase hexadecimal digits per byte, decoding either case. */
    sz_escape_hex_k = 1,

    /** HTML and XML text and attribute values: <tt>&amp;amp;</tt>, <tt>&amp;lt;</tt>, <tt>&amp;gt;</tt>, <tt>&amp;quot;</tt> and <tt>&amp;#39;</tt>. */
    sz_escape_html_k = 2,

    /** RFC 3986 percent-encoding of every byte outside <tt>A-Z a-z 0-9 - _ ~</tt> and the dot. */
    sz_escape_url_k = 3,

    /** Terminal-safe logs: short forms like <tt>\\n</tt>, <tt>\\\\</tt> and <tt>\\"</tt>, <tt>\\xHH</tt> for other C0 control
     *  bytes, DEL and ill-formed UTF-8, and <tt>\\uXXXX</tt> for C1 controls and the bidirectional
     *  overrides U+202A to U+202E and U+2066 to U+2069, which reorder what a reader sees. */
    sz_escape_console_k = 4,

    /** RFC 4180 fields, quoted only when holding a comma, a quote, CR or LF, with every inner
     *  quote doubled. */
    sz_escape_csv_k = 5,

    /** RFC 8259 string bodies in ASCII alone, as Python's <tt>json.dumps</tt> writes them by default:
     *  every codepoint outside printable ASCII without a short form, DEL included, becomes
     *  <tt>\\uXXXX</tt>, a surrogate pair beyond the Basic Multilingual Plane, and every ill-formed byte
     *  becomes the escape of U+FFFD. */
    sz_escape_json_ascii_k = 6,
} sz_escape_kind_t;

/** How a split ends its parts. */
typedef enum sz_split_kind_t {

    /** Every delimiter ends a part, so neighbouring delimiters leave empty parts, as
     *  <tt>str.split(sep)</tt> does. */
    sz_split_each_k = 0,

    /** A run of delimiters ends one part and empty parts are skipped, as <tt>str.split()</tt> does. */
    sz_split_runs_k = 1,
} sz_split_kind_t;

/**
 *  @brief A needle analysed once by @c sz_utf8_uncased_needle_init_best for any number of
 *      uncased searches, under any capabilities.
 *
 *  Preparing picks the per-script kernel for the needle in two steps:
 *
 *  1. locating the longest "safe" slice of the needle for the different SIMD folding kernels,
 *  2. shrinking it to the most diverse slice that fits into @c folded_slice when case-folded.
 *
 *  Unlike the exact substring search kernels, it uses 4 probe positions instead of 3:
 *
 *  - first: implicit at <tt>folded_slice[0]</tt>
 *  - second: @c probe_second
 *  - third: @c probe_third
 *  - last: implicit at <tt>folded_slice[folded_slice_length - 1]</tt>
 *
 *  The struct points at the needle rather than copying it, so the needle must outlive it.
 */
typedef struct sz_utf8_uncased_needle_t {

    /** The needle's text, read by every search. */
    sz_cptr_t start;
    sz_size_t length;

    /** Bytes of the needle before the slice that @c folded_slice holds folded. */
    sz_size_t offset_in_unfolded;

    /** Bytes of the needle that @c folded_slice holds folded. */
    sz_size_t length_in_unfolded;
    sz_u8_t folded_slice[16];
    sz_u8_t folded_slice_length;

    /** Offset of the second probe in @c folded_slice. */
    sz_u8_t probe_second;

    /** Offset of the third probe in @c folded_slice. */
    sz_u8_t probe_third;

    /** The @c sz_utf8_uncased_rune_*_k profile whose kernel searches this needle. */
    sz_u8_t script;
} sz_utf8_uncased_needle_t;

/**
 *  @brief A batch of prepared queries, the block it lives in, and the round's scratch beside it.
 *
 *  The first four members are @ref sz_levenshtein_query_t in tensor form: every query's mask plane
 *  back to back, which a kernel indexes by arithmetic where an array of structs would need a
 *  pointer chase. The launch geometry lives in @c memory 's head, which only the tier named by
 *  @c capability reads.
 */
typedef struct sz_levenshtein_engine_t {

    /** Every query's mask plane, back to back, @c masks_offsets addressing them. */
    sz_u64_t const *masks;

    /** The @b [count+1] word offsets into @c masks, the last being its length. */
    sz_size_t const *masks_offsets;

    /** The mask row each symbol reads, laid out as @c symbol spells it. */
    void const *symbol_to_class;

    /** The @b [count] symbols each query spans, which seeds its score bit. */
    sz_u32_t const *lengths;

    /** Queries prepared, which is the first axis of every output. */
    sz_size_t count;

    /** The alphabet the batch was prepared over, and how to read the class map. */
    sz_levenshtein_symbol_t symbol;

    /** The capability whose init kernel prepared the batch, which picks every round's kernel. */
    sz_capability_t capability;

    /** What built both blocks below and what grows the second. */
    sz_allocator_t allocator;

    /** The batch's block, fixed for the engine's life, its head tier-private. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** The round's block, grown by a compute verb and never shrunk. */
    void *scratch;

    /** Bytes of that block, zero until the first round sizes it. */
    sz_size_t scratch_bytes;
} sz_levenshtein_engine_t;

/** Words a query of @p length symbols spans - every class's row holds that many, one
 *  bit per symbol. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_query_words(sz_size_t length) {
    return length / 64 + (length % 64 != 0);
}

/** Maximum query words held in registers by the fixed-width GPU kernels. */
enum { sz_levenshtein_gpu_words_max_k = 256 };

/** N-grams a text of @p length bytes holds, summed over @p widths_count @p widths; a width of zero
 *  or past the text holds none. */
STRINGZILLA_CONSTEXPR sz_size_t sz_ngrams_count(sz_size_t length, sz_size_t const *widths, sz_size_t widths_count) {
    sz_size_t ngrams = 0;
    for (sz_size_t index = 0; index != widths_count; ++index)
        if (widths[index] && widths[index] <= length) ngrams += length - widths[index] + 1;
    return ngrams;
}

/** One prepared query's table of distinct N-grams at every width, and what a candidate's share
 *  needs beside it: what @ref sz_ngrams_engine_query reads off an engine, or what a device walk
 *  prepares inside a span its caller owns. */
typedef struct sz_ngrams_query_t {

    /** Slots of the query's distinct N-grams, zero if empty, read by @ref sz_ngrams_query_probe. */
    sz_u64_t const *slots;

    /** A power of two, at least twice @c ngrams and at least 64. */
    sz_size_t slots_count;

    /** The miss filter after the slots, eight bits a slot, so most N-grams the query lacks are told
     *  by one load. */
    sz_u32_t const *filter;

    /** The @b [widths_count] window widths, in bytes. */
    sz_size_t const *widths;

    /** The @b [widths_count] @ref sz_ngrams_power values. */
    sz_u32_t const *powers;

    /** Entries in @c widths and @c powers. */
    sz_size_t widths_count;

    /** The query's N-grams summed over every width, repeats included, which is the number of bits
     *  a candidate's hits span. */
    sz_size_t ngrams;

    /** The query's distinct N-grams, the share's denominator. */
    sz_size_t distinct;

    /** The query's bytes, so a width past them probes nothing. */
    sz_size_t length;
} sz_ngrams_query_t;

/**
 *  @brief Every prepared query's table over one block, the widths every one of them holds, and the
 *      scratch both verbs grow.
 *
 *  Only @c count, @c widths_count, @c capability and @c distinct_counts are public; the rest is the
 *  family's own layout, read through @ref sz_ngrams_engine_query.
 */
typedef struct sz_ngrams_engine_t {

    /** Every query's slots, back to back, @c slots_offsets addressing them. */
    sz_u64_t const *slots;

    /** The @b [count+1] slot offsets into @c slots, the last being its length. */
    sz_size_t const *slots_offsets;

    /** Every query's miss filter, eight bits a slot, the query @c i 's from word
     *  @c slots_offsets[i]/4. */
    sz_u32_t const *filters;

    /** The @b [count,widths_count] distinct N-grams each query holds at each width, which is a
     *  share's denominator. */
    sz_u32_t const *distinct_counts;

    /** The @b [widths_count] window widths, in bytes. */
    sz_size_t const *widths;

    /** The @b [widths_count] @ref sz_ngrams_power values. */
    sz_u32_t const *powers;

    /** The @b [count] bytes each query spans. */
    sz_u32_t const *lengths;

    /** Queries prepared, which is the first axis of every overlap output; zero for an engine that
     *  only sketches. */
    sz_size_t count;

    /** Window widths, the last axis of an overlap output. */
    sz_size_t widths_count;

    /** The capability whose init kernel prepared the engine, which picks every verb's kernel. */
    sz_capability_t capability;

    /** What built both blocks below and what grows the second. */
    sz_allocator_t allocator;

    /** The tables' block, fixed for the engine's life, its head tier-private. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** A round's chains, residues and hit words, or a sketch's ranks, grown and never shrunk. */
    void *scratch;

    /** Bytes of that block, zero until a verb sizes it. */
    sz_size_t scratch_bytes;
} sz_ngrams_engine_t;

/**
 *  @brief One needle ending on one state, as the engine stores it.
 *
 *  The engine walks @b folded bytes, so this length is the folded one - the needle's own, identical
 *  for every match of that needle. The @b source span it corresponds to is not: needle "k" matches
 *  both the 1-byte "k" and the 3-byte Kelvin sign U+212A. Recovering that span is the walk's job.
 */
typedef struct sz_substrings_output_t {

    /** Which needle ends here, as an index into the strings the engine was built from. */
    sz_u32_t needle_index;

    /** Folded bytes this match spans; a walk traverses one edge per byte, so it fits a state id. */
    sz_u32_t folded_match_bytes;
} sz_substrings_output_t;

/**
 *  @brief What a round found, written by the device and read after the caller's own join.
 *
 *  A device verb enqueues and returns, so nothing it discovers can reach the caller through a
 *  status code. The sizing walk always runs, so @c matches_emitted is the truth whatever the
 *  caller's output could hold, and a nonzero @c shortfall is the one signal that the output is
 *  incomplete rather than wrong. Under a leftmost policy a cover thins the matches after that walk,
 *  so for a round that stayed inside its budget the matches a caller could have read are
 *  @c matches_stored plus @c shortfall, which is also what the last boundary of a
 *  @ref sz_substrings_find names; a round that outran its budget ran no cover at all, and its
 *  @c shortfall counts the matches the budget could not hold.
 */
typedef struct sz_substrings_report_t {

    /** Matches the sizing walk found, which is the truth whatever the output held. */
    sz_size_t matches_emitted;

    /** Matches written out, which is @c matches_emitted clipped at the capacity. */
    sz_size_t matches_stored;

    /** Bytes a rewrite needs, which is the truth whatever the target held. */
    sz_size_t target_length;

    /** Matches or bytes the round could not hold, zero when everything fit. */
    sz_size_t shortfall;
} sz_substrings_report_t;

/**
 *  @brief The compiled vocabulary, its sizing policy, and the round's arena, in one lifetime.
 *
 *  Transitions are split into two tiers by how often a state is visited. Text keeps resetting the
 *  walk toward the root, so a small set of states absorbs most byte steps whatever the dictionary
 *  size, and the tiers are sized to that skew rather than to the vocabulary as a whole.
 *
 *  The @b hot tier is a dense goto-completed table, one row per state with one target per byte
 *  @b class, so a step is a single load with no branch and no failure chasing. A class is a byte
 *  some needle spells, or the one shared class of every byte none does, so a row is as wide as the
 *  vocabulary's own alphabet: five targets for nucleotides, a few hundred for multilingual text,
 *  and never more than 256. The @b cold tier is a double array: @c base and @c check encode
 *  transitions as address arithmetic plus an ownership test, and @c fail restores the failure links
 *  that goto completion would otherwise have folded away. States are numbered so the hot ones come
 *  first, which makes the tier test state < hot_count with no lookup.
 *
 *  Both blocks are built by an init kernel and live until @ref sz_substrings_engine_free, so no
 *  compute verb rebuilds them and a device backend has no host clone to stage.
 */
typedef struct sz_substrings_engine_t {

    /** Hot tier: hot_count × classes_count goto-completed targets, row-major, shallow first. */
    sz_u32_t const *hot_rows;

    /** Each byte's column in @c hot_rows: its own if a needle spells it, else one shared column. */
    sz_u8_t const *byte_to_class;

    /** Cold tier: transition target for a state on a byte is base[state] + byte, if owned. */
    sz_u32_t const *base;

    /** Cold tier: owner of each slot, so a collision reads as a missing edge, not a wrong one. */
    sz_u32_t const *check;

    /** Cold tier: failure link, followed when @c check denies ownership. */
    sz_u32_t const *fail;

    /** One bit per slot: whether any needle ends there, so a non-matching byte touches no count. */
    sz_u32_t const *accepts_words;

    /** Matches ending at each state, flattened and merged along failure chains at build time. */
    sz_substrings_output_t const *outputs;

    /** Outputs per slot, in the published numbering. */
    sz_u32_t const *outputs_counts;

    /** Exclusive prefix sum of @c outputs_counts; a nested-suffix vocabulary drives the pool to
     *  O(states²), so these stay pointer-wide where the counts do not. */
    sz_size_t const *outputs_offsets;

    /** Length of @c outputs, so a consumer never rescans the CSR to recover it. */
    sz_size_t outputs_total;

    /** Slots every cold-tier array holds: @c state_count plus the alphabet's address headroom. */
    sz_size_t slots_count;

    /** States below @c hot_count live in @c hot_rows; the rest live in the double array. */
    sz_u32_t hot_count;

    /** Columns of a hot row, at most 256. */
    sz_u32_t classes_count;

    /** Published state ceiling, above the trie's, as a packed child's id is address arithmetic. */
    sz_u32_t state_count;

    /** The root's published id, which is always zero. */
    sz_u32_t root;

    /** Needles the vocabulary holds, which bounds every reported @c needle_index. */
    sz_u32_t needles_count;

    /** Most @b haystack bytes one match can span, sizing every slice, halo and warm-up. */
    sz_u32_t max_source_match_bytes;

    /** Fewest haystack bytes one match can span; the mirror bound. */
    sz_u32_t min_source_match_bytes;

    /** Most merged outputs any state carries, so a consumer can bound one pass's match count. */
    sz_u32_t max_outputs_per_state;

    /** Whether a walk folds the haystack as it consumes it, or steps it byte for byte. */
    sz_case_sensitivity_t case_sensitivity;

    /** Bytes that move a walk off the root, which a byte search can skip to while it sits there. */
    sz_byteset_t root_live;

    /** The policy the arena was sized for, and the only one it runs. */
    sz_substrings_overlap_policy_t overlap_policy;

    /** Matches one round may emit; past it every later kernel retires. */
    sz_size_t matches_budget;

    /** Chunks one round may cut the haystacks into, fixed here, not discovered. */
    sz_size_t chunk_budget;

    /** Haystacks one round may carry, read by a device tier only; its arena is sized for them. */
    sz_size_t haystacks_budget;

    /** Device-resident counts every verb writes and no verb joins to read. */
    sz_substrings_report_t *report;

    /** The capability that compiled the vocabulary, whose kernels every round then runs. */
    sz_capability_t capability;

    /** What built both blocks below. */
    sz_allocator_t allocator;

    /** The automaton's block, fixed for the engine's life. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** The round's arena: ring, chunks, emitted, keep, scan scratch. */
    void *scratch;

    /** Bytes of that block. */
    sz_size_t scratch_bytes;
} sz_substrings_engine_t;

/** BM25's continuous parameters. */
typedef struct sz_substrings_bm25_t {

    /** The literature's k₁: how slowly repeated occurrences stop adding score; 1.2 is customary. */
    sz_f32_t term_frequency_saturation;

    /** The literature's b, in [0, 1]: 0 ignores document length and 1 normalizes it fully, while
     *  0.75 is the customary choice. */
    sz_f32_t length_normalization;

    /** The corpus-wide mean document length, in the unit of the lengths scored; read only when
     *  @c length_normalization is positive. */
    sz_f32_t average_document_length;
} sz_substrings_bm25_t;

/** The @c hot_states value that sizes the hot tier itself, so zero stays a genuine all-cold ask. */
#define STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO (STRINGZILLA_SIZE_MAX)

/** The kinds of NFA states. */
typedef enum sz_regex_nfa_kind_t {

    /** Consumes one byte along every link whose range holds it; with no links it never matches. */
    sz_regex_nfa_ranges_k = 0,

    /** Moves along every link without consuming, in link order. */
    sz_regex_nfa_union_k = 1,

    /** Moves along every link without consuming; links are prepended, so later patches go first. */
    sz_regex_nfa_union_lazy_k = 2,

    /** Moves to @c target without consuming. */
    sz_regex_nfa_empty_k = 3,

    /** Moves to @c target without consuming where its assertion holds. */
    sz_regex_nfa_look_k = 4,

    /** A pattern matched; @c target is the pattern's index. */
    sz_regex_nfa_match_k = 5,
} sz_regex_nfa_kind_t;

/** One NFA state. */
typedef struct sz_regex_nfa_state_t {

    /** What the state does. */
    sz_regex_nfa_kind_t kind;

    /** An assertion's bit. */
    sz_u32_t look;

    /** The next state of an empty or look state, or a match state's pattern. */
    sz_u32_t target;

    /** The capture slot an empty state writes as a walk passes it, twice its group plus one where
     *  the group closes, or @c sz_regex_none_k. */
    sz_u32_t slot;

    /** The first outgoing link of a ranges or union state, or @c sz_regex_none_k. */
    sz_u32_t first_link;

    /** The last outgoing link, which an appended link follows. */
    sz_u32_t last_link;
} sz_regex_nfa_state_t;

/** How a lookaround reads its body around the position it is tested at. */
typedef enum sz_regex_lookaround_kind_t {

    /** <tt>(?=…)</tt>: the body matches from the position on. */
    sz_regex_lookahead_k = 0,

    /** <tt>(?</tt><tt>!</tt>…<tt>)</tt>: the body matches nowhere from the position on. */
    sz_regex_lookahead_negated_k = 1,

    /** <tt>(?</tt><tt>&lt;=…)</tt>: the body matches up to the position. */
    sz_regex_lookbehind_k = 2,

    /** <tt>(?</tt><tt>&lt;!</tt>…<tt>)</tt>: the body matches nowhere up to the position. */
    sz_regex_lookbehind_negated_k = 3,
} sz_regex_lookaround_kind_t;

/** One lookaround: its body, compiled into the engine's NFA as a sub-vocabulary of its own, and how
 *  it reads it. */
typedef struct sz_regex_lookaround_t {

    /** Where a walk of the body enters @c nfa_states, compiled forward for a lookbehind and
     *  reversed for a lookahead; while the vocabulary builds, the root of its syntax tree. */
    sz_u32_t start;

    /** How the lookaround reads its body: odd kinds are negated, and those from
     *  @c sz_regex_lookbehind_k look behind. */
    sz_regex_lookaround_kind_t kind;

    /** One past the highest level among the lookarounds of its body, so levels computed in
     *  ascending order find every bit a body tests known. */
    sz_u32_t level;
} sz_regex_lookaround_t;

/** One lookaround body's DFA, unanchored and keeping every match, which fills the table of where
 *  its lookaround holds a byte per step instead of the PikeVM. */
typedef struct sz_regex_body_t {

    /** Rows of <tt>classes_count + 1</tt> next states, the last column taken at the far edge of a
     *  haystack, ids being row offsets; null for a body over other lookarounds, of one byte, or
     *  past its budget. */
    sz_u32_t const *rows;

    /** The state a pass starts from at the haystack's edge it walks from. */
    sz_u32_t start;

    /** Ids at or above this one accept, which the step over the unit after a match of the body
     *  lands on. */
    sz_u32_t accepts_begin;
} sz_regex_body_t;

/** One outgoing edge, chained per state. */
typedef struct sz_regex_nfa_link_t {

    /** The state the link leads to. */
    sz_u32_t target;

    /** The owner's next link, or @c sz_regex_none_k. */
    sz_u32_t next;

    /** The lowest byte a ranges state's link consumes. */
    sz_u8_t low;

    /** The highest byte a ranges state's link consumes. */
    sz_u8_t high;
} sz_regex_nfa_link_t;

/** Bytes from a candidate start that the literal prefilter tests. */
enum { sz_regex_prefixes_depth_k = 3 };

/** Paths of byte sets the literal prefilter tells apart, one bucket bit each. */
enum { sz_regex_prefixes_paths_k = 8 };

/**
 *  @brief The paths of byte sets the first bytes of every match follow, as the end-finding DFA's
 *      first steps allow them under any lookaround, each path one bucket bit.
 *
 *  A search skips to the next position whose bytes keep some bucket at every offset, and the end
 *  walk from there settles whether a match starts there, so a superset of the paths stays exact.
 *  A path shorter than the deepest holds every byte past its end. Vector tiers test a byte's two
 *  nibbles apart, which holds a superset of its bucket's bytes where a path's set has several.
 */
typedef struct sz_regex_prefixes_t {

    /** Per offset from a candidate start, per byte, the buckets whose paths hold the byte there. */
    sz_u8_t buckets[sz_regex_prefixes_depth_k][256];

    /** Per offset, the buckets holding some byte of each low nibble, then of each high nibble;
     *  every bucket at offsets past @c depth. */
    sz_u8_t nibbles[sz_regex_prefixes_depth_k][32];

    /** Offsets of the deepest path. */
    sz_u32_t depth;
} sz_regex_prefixes_t;

/**
 *  @brief A compiled vocabulary of regular expressions: two automata over one byte-class map.
 *
 *  Both automata are dense tables of <tt>classes_count + 1</tt> columns per block, the last column taken
 *  at a haystack's edge, and one block per set of lookarounds that may hold at the boundary a step
 *  crosses, as @c sz_regex_stride_ and @c sz_regex_looks_block_ lay them out. A state id is its
 *  row's offset, so a step is one load with no multiply; the dead state is zero, and accepting
 *  states are numbered last, so acceptance is one compare.
 *
 *  A start-finding DFA that would pass its state budget is replaced by a position automaton, with
 *  @c reverse_rows null: a set of @c positions_words words over the classes of the reversed
 *  patterns, stepped over whole codepoints. A boundary kind is the kind of the unit left of a
 *  boundary times four plus the kind right of it, each zero at the haystack's edge, one for a line
 *  feed, two for a word byte and three otherwise; a mask of them is a 16-bit word.
 *
 *  A lookaround is a look state testing bit <tt>sz_regex_look_lookaround_k << i</tt> of its index @c i,
 *  which a table of the walk holds per haystack position, as @c lookarounds describes; a DFA step
 *  reads it to pick its block, except within one codepoint's folds, which take the first block.
 *
 *  Both blocks are owned and live until @ref sz_regex_engine_free; the round's scratch grows on
 *  demand and is never shrunk.
 */
typedef struct sz_regex_engine_t {

    /** Each byte's column in both automata; bytes share one only if no pattern tells them apart. */
    sz_u8_t const *byte_to_class;

    /** Start-finding automaton: the reversed patterns, unanchored, keeping every match, walked
     *  right to left. Rows of blocks of <tt>classes_count + 1</tt> next states, the last column taken at
     *  the start of a haystack, ids being row offsets. A state at or above @c reverse_accepts_begin
     *  after the byte at @c i starts a match at <tt>i + 1</tt>; that step takes the block of the
     *  lookarounds holding at <tt>i + 1</tt>. */
    sz_u32_t const *reverse_rows;

    /** End-finding automaton: the patterns anchored and leftmost-first, walked left to right from
     *  one start and laid out the same, a step over the byte at @c i taking the block of the
     *  lookarounds holding at @c i. The last accepting state before it dies names the match's end
     *  and pattern. Null past its state budget or past the lookarounds its capability takes as
     *  columns, where end walks run a PikeVM over @c nfa_states. */
    sz_u32_t const *forward_rows;

    /** Per forward accepting state, in id order, the pattern leftmost-first priority picks. */
    sz_u32_t const *forward_patterns;

    /** Per forward state, in id order, the bytes that move it elsewhere under any block, which an
     *  end walk searches ahead for where the state stays put on all but a few printable ASCII
     *  bytes; every byte for other states. */
    sz_byteset_t const *forward_escapes;

    /** The paths every match's first bytes follow, which a search skips to instead of marking
     *  starts; null where they would pass too many positions, where some match is empty, for an
     *  uncased engine, and where end walks run the PikeVM. */
    sz_regex_prefixes_t const *prefixes;

    /** Forward start by the byte before a match: none, a line feed, a word byte, or another. */
    sz_u32_t forward_starts[4];

    /** Reverse start by the byte after a haystack: none, a line feed, a word byte, or another. */
    sz_u32_t reverse_starts[4];

    /** Forward ids at or above this one accept, so acceptance is a compare rather than a load. */
    sz_u32_t forward_accepts_begin;

    /** Reverse ids at or above this one mark a start. */
    sz_u32_t reverse_accepts_begin;

    /** Columns of a block before its haystack-edge column. */
    sz_u32_t classes_count;

    /** Rows of the end-finding automaton. */
    sz_u32_t forward_states_count;

    /** Rows of the start-finding automaton, all of which priming walks at once. */
    sz_u32_t reverse_states_count;

    /** Patterns the vocabulary holds, which bounds every reported @c needle_index. */
    sz_u32_t patterns_count;

    /** Bytes that move the start pass off its idle states, which a reverse byte-set search skips
     *  to: for the DFA, the starts after a line feed, a word byte and another byte, which any other
     *  byte maps into one another under every block; for positions, the loop alone, or every byte
     *  when the loop alone can match. */
    sz_byteset_t reverse_live;

    /** Words of one position set of the bit-parallel start pass, which replaces @c reverse_rows
     *  past their budget. */
    sz_u32_t positions_words;

    /** Records in @c positions_shifts: a signed distance word, then the positions moving by it
     *  on every unit. */
    sz_u32_t positions_shifts_count;

    /** Records in @c positions_groups: a mask of boundary kinds, the source positions, then the
     *  targets they enter. */
    sz_u32_t positions_groups_count;

    /** Pairs in @c positions_ranges. */
    sz_u32_t positions_ranges_count;

    /** Per unit class, the positions able to consume it; the loop and match positions are
     *  in every row. */
    sz_u64_t const *positions_reach;

    /** The unconditional follows that a constant distance between positions covers,
     *  one record per distance. */
    sz_u64_t const *positions_shifts;

    /** Every other follow, its sources grouped by the targets they share under one condition: a
     *  mask of boundary kinds in the low 16 bits of the record's first word, then the look bits of
     *  the lookarounds that must hold. */
    sz_u64_t const *positions_groups;

    /** Every position a walk can hold, which priming's upper run starts from. */
    sz_u64_t const *positions_all;

    /** Per boundary kind, the positions the always-live loop enters, itself among them,
     *  @c positions_words each. */
    sz_u64_t const *positions_initial;

    /** Each ASCII codepoint's unit class. */
    sz_u32_t const *positions_ascii;

    /** Past ASCII, ascending pairs of a codepoint range's first codepoint and its unit class. */
    sz_u32_t const *positions_ranges;

    /** The Thompson NFA the end-finding automaton was built from, which a PikeVM walks where
     *  @c forward_rows is null and which groups are read from; a capture is an empty state
     *  carrying a slot. */
    sz_regex_nfa_state_t const *nfa_states;

    /** Every link of @c nfa_states, chained per state. */
    sz_regex_nfa_link_t const *nfa_links;

    /** States of @c nfa_states. */
    sz_u32_t nfa_states_count;

    /** Links of @c nfa_links. */
    sz_u32_t nfa_links_count;

    /** Where a walk of every pattern at once enters the NFA, the patterns as alternatives
     *  in index order. */
    sz_u32_t nfa_start;

    /** Per pattern, where a walk of that pattern alone enters the NFA. */
    sz_u32_t const *patterns_starts;

    /** Per pattern, its first capturing group among all groups numbered pattern after pattern,
     *  then their total; group @c g of a pattern, counting from one, writes slots @c 2g and
     *  <tt>2g + 1</tt> of a walk. */
    sz_u32_t const *patterns_groups;

    /** Per lookaround, in the order of the look bits naming them, its body's entry into
     *  @c nfa_states and its kind. */
    sz_regex_lookaround_t const *lookarounds;

    /** Per lookaround, its body's DFA, which fills the table where it is built; null without
     *  lookarounds. */
    sz_regex_body_t const *bodies;

    /** The lookarounds whose bodies match one byte of a set and assert nothing, so the byte beside
     *  a position alone decides whether they hold; then per byte, those holding right after it;
     *  then per byte, those holding right before it. Null where no body is one byte, as in an
     *  uncased engine, whose lookarounds hold between whole codepoints only. */
    sz_u32_t const *looks_bytes;

    /** Lookarounds the vocabulary holds, nested ones included; up to
     *  @c sz_regex_columned_lookarounds_k of them widen the rows of both DFAs to a block per set,
     *  and more leave walks on positions and the PikeVM. */
    sz_u32_t lookarounds_count;

    /** Every capturing group's name, in the order @c patterns_groups numbers them,
     *  empty for an unnamed group. */
    sz_strings_view_t groups_names;

    /** Whether walks fold the haystack as they consume it, or step it byte for byte. */
    sz_case_sensitivity_t case_sensitivity;

    /** The capability that compiled the vocabulary, whose kernels every round then runs. */
    sz_capability_t capability;

    /** What built both blocks below. */
    sz_allocator_t allocator;

    /** Both automata and their side arrays, fixed for the engine's life. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** A round's start bits, priming sets and window edges,
     *  grown on demand like @c levenshtein's. */
    void *scratch;

    /** Bytes of that block. */
    sz_size_t scratch_bytes;
} sz_regex_engine_t;

/** The array positions one selector picks, RFC 9535's <tt>[begin:end:step]</tt>, an index @c n being
 *  <tt>[n:n+1:1]</tt>, or <tt>[-1::1]</tt> for the last element; negative bounds count from the array's end. */
typedef struct sz_subdocuments_slice_t {

    /** The first position, or 2⁵³ past the end that a negative step walks from when omitted. */
    sz_i64_t begin;

    /** The position the slice stops before, or 2⁵³ past the end its step walks to when omitted. */
    sz_i64_t end;

    /** Distance between selected positions, negative walking backward; zero selects none. */
    sz_i64_t step;
} sz_subdocuments_slice_t;

/** One instruction of a filter program. */
typedef struct sz_subdocuments_instruction_t {

    /** What it does, a @c sz_subdocuments_operation_t. */
    sz_u32_t operation;

    /** A step, a pattern, a comparison or a literal's kind, as @c operation reads it. */
    sz_u32_t operand;

    /** A number literal's value. */
    sz_f64_t number;

    /** Where a string literal or a <tt>match()</tt> or <tt>search()</tt> pattern starts in @c strings_data. */
    sz_u32_t string_offset;

    /** Its unescaped bytes. */
    sz_u32_t string_length;
} sz_subdocuments_instruction_t;

/** The filters of one bracketed selection, joined by disjunction, and the program deciding them. */
typedef struct sz_subdocuments_filter_t {

    /** The step whose selection they decide. */
    sz_u32_t step;

    /** The program's first instruction in @c instructions. */
    sz_u32_t instructions_begin;

    /** One past its last. */
    sz_u32_t instructions_end;
} sz_subdocuments_filter_t;

/**
 *  @brief The compiled paths: one bit-parallel automaton over keys and positions, with no table.
 *
 *  A step is one segment of one path, numbered path after path, the first step of every path
 *  being its root; "live" is a bitset of steps. Entering a member, an element, an attribute or a
 *  text node advances every live step that is not final to the next one, keeps those whose
 *  selectors accept the child's key, name, kind or position, and carries the steps a descendant
 *  segment keeps live at every depth below.
 */
typedef struct sz_subdocuments_engine_t {

    /** Bytes of every key, element or attribute name the paths spell, back to back, to
     *  confirm a hash hit. */
    sz_cptr_t names_data;

    /** Offsets into @c names_data, one per name and one past the last. */
    sz_u32_t const *names_offsets;

    /** Open-addressed name hash table, each slot a name index or @c sz_subdocuments_no_name_k. */
    sz_u32_t const *names_slots;

    /** Per name, the steps whose selector is that name, @c words_count words each. */
    sz_u64_t const *names_steps;

    /** Steps any key or index enters: wildcards. */
    sz_u64_t const *wildcard_steps;

    /** Steps a descendant segment keeps live at every depth below: each one
     *  precedes a <tt>..</tt> selector. */
    sz_u64_t const *descendant_steps;

    /** Steps with a slice the array's length settles: negative indices or bounds, or a step
     *  under -1. An array such a step may enter opens a pending run, settled when it closes. */
    sz_u64_t const *relative_steps;

    /** XPath steps selecting attributes, <tt>\@name</tt> and <tt>\@*</tt>, which select no element. */
    sz_u64_t const *attribute_steps;

    /** XPath steps selecting text nodes, <tt>text()</tt>. */
    sz_u64_t const *text_steps;

    /** Steps with a filter selector, whose selection of a node waits for that node's close. */
    sz_u64_t const *filter_steps;

    /** Per filter, the root steps of its queries, @c words_count words each: the
     *  steps <tt>\@</tt> stands for. */
    sz_u64_t const *filters_roots;

    /** Every filter, in the order compiled. */
    sz_subdocuments_filter_t const *filters;

    /** Every filter's program, back to back. */
    sz_subdocuments_instruction_t const *instructions;

    /** Bytes of every string literal and pattern the filters spell, unescaped. */
    sz_cptr_t strings_data;

    /** The last step of every path and filter query, which never advances. */
    sz_u64_t const *final_steps;

    /** The first step of every path, live at the document's root; a filter query's first step is
     *  live only at the nodes its filter decides. */
    sz_u64_t const *root_steps;

    /** The first step of every bare filter path, live at a holder above the document's root whose
     *  one element the root is, so the filter decides the root. */
    sz_u64_t const *holder_steps;

    /** The array positions of every index and slice selector, step after step. */
    sz_subdocuments_slice_t const *slices;

    /** Per step, where its slices start in @c slices, and one past the last step's. */
    sz_u32_t const *slices_offsets;

    /** Per step, the path it belongs to, reported as @c needle_index. */
    sz_u32_t const *steps_paths;

    /** Steps across all paths. */
    sz_u32_t steps_count;

    /** 64-bit words per step bitset. */
    sz_u32_t words_count;

    /** Slots of @c names_slots, a power of two, or zero when no step selects by name. */
    sz_u32_t names_slots_count;

    /** Paths the vocabulary holds; the queries of filters take steps after
     *  theirs, and report nothing. */
    sz_u32_t paths_count;

    /** Filters the paths hold. */
    sz_u32_t filters_count;

    /** Patterns of <tt>match()</tt> and <tt>search()</tt>, each compiled behind a tag naming it, so
     *  that one engine answers for each alone: a tested string carries the tag of its
     *  pattern. Empty without any. */
    sz_regex_engine_t regex;

    /** Patterns @c regex holds, zero when it holds none and was never compiled. */
    sz_u32_t patterns_count;

    /** The document grammar and the path language that goes with it. */
    sz_subdocuments_format_t format;

    /** How the JSON walks read what RFC 8259 leaves open. */
    sz_subdocuments_options_t options;

    /** The capability that compiled the paths. */
    sz_capability_t capability;

    /** What built both blocks below. */
    sz_allocator_t allocator;

    /** The compiled paths. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** The host depth stack of live sets and the nodes of a pending run, grown on demand like the
     *  round scratch of @c levenshtein. */
    void *scratch;

    /** Bytes of that block. */
    sz_size_t scratch_bytes;
} sz_subdocuments_engine_t;

#pragma endregion

/*  This is where the actual implementation begins, hidden from the public API. */
#pragma region Helper Functions

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC visibility push(hidden)
#endif

/** Helper-macro to mark potentially unused variables. */
#define sz_unused_(x) ((void)(x))

/** Helper-macro casting a variable to another type of the same size. */
#if !defined(_MSC_VER) && defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
#define sz_bitcast_(type, value) __builtin_bit_cast(type, (value))
#else
#define sz_bitcast_(type, value) (*((type *)&(value)))
#endif
#else
#define sz_bitcast_(type, value) (*((type *)&(value)))
#endif

STRINGZILLA_CONSTEXPR sz_size_t sz_size_max_(void) { return STRINGZILLA_SIZE_MAX; }
STRINGZILLA_CONSTEXPR sz_ssize_t sz_ssize_max_(void) { return STRINGZILLA_SSIZE_MAX; }

/**
 *  @brief Similar to @c assert, the @c sz_assert_ checks library invariants in @c STRINGZILLA_DEBUG
 *      builds, aborting on failure; in release it type-checks the condition without evaluating it.
 *  @note If you want to catch it, put a breakpoint at @c abort.
 */
#if STRINGZILLA_DEBUG && (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) // ? CUDA or HIP code for GPUs
STRINGZILLA_DEVICE_NOINLINE void sz_assert_cuda_failure_(char const *condition, char const *file, int line) {
    printf("Assertion failed: %s, in file %s, line %d\n", condition, file, line);
    __builtin_trap();
}
#define sz_assert_(condition)                                                          \
    do {                                                                               \
        if (!(condition)) { sz_assert_cuda_failure_(#condition, __FILE__, __LINE__); } \
    } while (0)
#elif STRINGZILLA_DEBUG && STRINGZILLA_WITH_LIBC // ? CPU code with LibC, PIC included
STRINGZILLA_OUTLINED void sz_assert_failure_(char const *condition, char const *file, int line) {
    fprintf(stderr, "Assertion failed: %s, in file %s, line %d\n", condition, file, line);
    abort();
}
#define sz_assert_(condition)                                                     \
    do {                                                                          \
        if (!(condition)) { sz_assert_failure_(#condition, __FILE__, __LINE__); } \
    } while (0)
#elif STRINGZILLA_DEBUG && defined(_MSC_VER) && !defined(__clang__) // ? No LibC, and MSVC has no `__builtin_trap`
#define sz_assert_(condition)             \
    do {                                  \
        if (!(condition)) __debugbreak(); \
    } while (0)
#elif STRINGZILLA_DEBUG // ? No LibC: nothing to print with, so trap in place
#define sz_assert_(condition)               \
    do {                                    \
        if (!(condition)) __builtin_trap(); \
    } while (0)
#else
#define sz_assert_(condition) sz_unused_(sizeof(!(condition)))
#endif

/** Asserts that @p output either is @p input, over as many bytes, or shares no byte with it: the
 *  aliasing every in-place-or-disjoint transform accepts. It compares distances rather than ends,
 *  so no sum can wrap. */
#define sz_assert_no_overlap_(output, output_length, input, input_length)                          \
    sz_assert_((output_length) == 0 || (input_length) == 0 ||                                      \
               ((sz_cptr_t)(output) == (sz_cptr_t)(input) && (output_length) == (input_length)) || \
               ((sz_size_t)(output) <= (sz_size_t)(input)                                          \
                    ? (sz_size_t)(input) - (sz_size_t)(output) >= (output_length)                  \
                    : (sz_size_t)(output) - (sz_size_t)(input) >= (input_length)))

/**
 *  @brief Whether one batch from a UTF-8 decoder or token finder keeps the contract its resuming
 *      callers loop on.
 *  @param[in] length Bytes offered to the call.
 *  @param[in] capacity Entries the caller had room for.
 *  @param[in] count Entries the call reported.
 *  @param[in] consumed Bytes the call covered, where its caller resumes.
 *  @param[in] starts Span offsets, or @c STRINGZILLA_NULL for the decoder, which reports
 *      runes, not spans.
 *  @param[in] lengths Span lengths, read only alongside @p starts.
 *
 *  The batch fits @p capacity and never runs past @p length, its spans ascend inside the consumed
 *  prefix, and it makes progress whenever it had room for an entry and bytes to cover, because a
 *  caller resuming from @p consumed would otherwise loop forever.
 */
STRINGZILLA_CONSTEXPR sz_bool_t sz_utf8_batch_consistent_(sz_size_t length, sz_size_t capacity, sz_size_t count,
                                                          sz_size_t consumed, sz_size_t const *starts,
                                                          sz_size_t const *lengths) {
    if (count > capacity || consumed > length) return sz_false_k;
    if (consumed == 0 && capacity != 0 && length != 0) return sz_false_k;
    for (sz_size_t index = 0; starts && index != count; ++index) {
        sz_size_t const earliest = index == 0 ? 0 : starts[index - 1] + lengths[index - 1];
        if (starts[index] < earliest || starts[index] > consumed || lengths[index] > consumed - starts[index])
            return sz_false_k;
    }
    return sz_true_k;
}

/*  Intrinsics aliases for MSVC, GCC, Clang, and Clang-Cl. The following section of compiler
 *  intrinsics comes in 2 flavors. */
#if defined(_MSC_VER) && !defined(__clang__) // On Clang-CL
#include <intrin.h>

/*
 *  Sadly, when building Win32 images, we can't use the @c _tzcnt_u64, @c _lzcnt_u64,
 *  @c _BitScanForward64, or @c _BitScanReverse64 intrinsics, so 32-bit x86 and Arm use the serial
 *  version. For now it's a simple @c for loop; a De Bruijn's algorithm would be more efficient.
 *
 *  @see BitScan: https://www.chessprogramming.org/BitScan
 *  @see De Bruijn Sequence: https://www.chessprogramming.org/De_Bruijn_Sequence
 *  @see Bit-scan with De Bruijn: https://gist.github.com/resilar/e722d4600dbec9752771ab4c9d47044f
 */
#if !STRINGZILLA_ARCH_X8664_
STRINGZILLA_INLINE int sz_u64_ctz(sz_u64_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
STRINGZILLA_INLINE int sz_u64_clz(sz_u64_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 0x8000000000000000ull) == 0) { n++, x <<= 1; }
    return n;
}
STRINGZILLA_INLINE int sz_u64_popcount(sz_u64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    return (((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full) * 0x0101010101010101ull) >> 56;
}
STRINGZILLA_INLINE int sz_u32_ctz(sz_u32_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
STRINGZILLA_INLINE int sz_u32_clz(sz_u32_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 0x80000000u) == 0) { n++, x <<= 1; }
    return n;
}
STRINGZILLA_INLINE int sz_u32_popcount(sz_u32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
#else
/*  Before Haswell @c lzcnt decodes as @c bsr and miscounts, so leading zeros go through @c bsr. */
STRINGZILLA_INLINE int sz_u64_ctz(sz_u64_t x) { return (int)_tzcnt_u64(x); }
STRINGZILLA_INLINE int sz_u64_clz(sz_u64_t x) {
    unsigned long index;
    _BitScanReverse64(&index, x);
    return 63 - (int)index;
}
STRINGZILLA_INLINE int sz_u64_popcount(sz_u64_t x) { return (int)__popcnt64(x); }
STRINGZILLA_INLINE int sz_u32_ctz(sz_u32_t x) { return (int)_tzcnt_u32(x); }
STRINGZILLA_INLINE int sz_u32_clz(sz_u32_t x) {
    unsigned long index;
    _BitScanReverse(&index, x);
    return 31 - (int)index;
}
STRINGZILLA_INLINE int sz_u32_popcount(sz_u32_t x) { return (int)__popcnt(x); }
#endif
/*  Force the byteswap functions to be intrinsics, because when @c /Oi- is given, these will turn
 *  into CRT function calls, which breaks when @c STRINGZILLA_WITH_LIBC is 0. */
#pragma intrinsic(_byteswap_uint64)
STRINGZILLA_INLINE sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return _byteswap_uint64(val); }
#pragma intrinsic(_byteswap_ulong)
STRINGZILLA_INLINE sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return _byteswap_ulong(val); }
#else
STRINGZILLA_INLINE int sz_u64_popcount(sz_u64_t x) { return __builtin_popcountll(x); }
STRINGZILLA_INLINE int sz_u32_popcount(sz_u32_t x) { return __builtin_popcount(x); }
STRINGZILLA_INLINE int sz_u64_ctz(sz_u64_t x) { return __builtin_ctzll(x); }
STRINGZILLA_INLINE int sz_u64_clz(sz_u64_t x) { return __builtin_clzll(x); }
STRINGZILLA_INLINE int sz_u32_ctz(sz_u32_t x) { return __builtin_ctz(x); } // ! Undefined if `x == 0`
STRINGZILLA_INLINE int sz_u32_clz(sz_u32_t x) { return __builtin_clz(x); } // ! Undefined if `x == 0`
STRINGZILLA_INLINE sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return __builtin_bswap64(val); }
STRINGZILLA_INLINE sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return __builtin_bswap32(val); }
#endif

/*  Arm NEON kernel files call these directly instead of @c sz_u32_ctz, @c sz_u32_clz, and
 *  @c sz_u32_popcount: unlike those, they are explicitly ISA-named and never fall back to a
 *  portable loop, so a missing case here is a compile error, not a silent downgrade. Real MSVC, not
 *  clang-cl, compiles NEON code on Windows-on-Arm, where @c STRINGZILLA_TARGET_NEON is
 *  unconditionally 1, and has no `__builtin_*` or ACLE support, so the MSVC branch is required, not
 *  optional; `_CountTrailingZeros[64]` needs VS2022 17.7+, which matches this repo's CI floor, the
 *  @c windows-11-arm runner. */
#if defined(_MSC_VER) && !defined(__clang__)
#define sz_u32_ctz_neon_(x) ((int)_CountTrailingZeros(x))
#define sz_u64_ctz_neon_(x) ((int)_CountTrailingZeros64(x))
#define sz_u32_clz_neon_(x) ((int)_CountLeadingZeros(x))
#define sz_u64_clz_neon_(x) ((int)_CountLeadingZeros64(x))
#define sz_u32_popcount_neon_(x) ((int)_CountOneBits(x))
#define sz_u64_popcount_neon_(x) ((int)_CountOneBits64(x))
#else
#define sz_u32_ctz_neon_(x) ((int)__clz(__rbit(x))) // ACLE, byte-identical to `sz_u32_ctz`'s `rbit`+`clz`
#define sz_u64_ctz_neon_(x) ((int)__clzll(__rbitll(x)))
#define sz_u32_clz_neon_(x) ((int)__clz(x))
#define sz_u64_clz_neon_(x) ((int)__clzll(x))
#define sz_u32_popcount_neon_(x) (__builtin_popcount(x)) // no ACLE popcount intrinsic exists
#define sz_u64_popcount_neon_(x) (__builtin_popcountll(x))
#endif

/**
 *  @brief Returns @p pointer unchanged, hiding its origin so table reads stay memory operands.
 *
 *  Left visible, a <tt>static const</tt> table folds into immediates and every constant costs a
 *  @c vpbroadcastd where an AVX-512 <tt>{1toN}</tt> embedded broadcast would have been free. No use
 *  outside x86, which has no equivalent operand to protect.
 */
STRINGZILLA_INLINE void const *sz_x86_hide_pointer_origin_(void const *pointer) {
#if defined(__GNUC__)
    __asm__("" : "+r"(pointer));
#endif
    return pointer;
}

/** Reverse the 64 bits of @p value, so bit i moves to bit 63 - i: swap adjacent bits, then
 *  bit-pairs within nibbles, then nibbles within bytes, then the bytes. Lets an ascending-only
 *  byte-compress, like @c vpcompressb, pack lanes in descending order. */
STRINGZILLA_INLINE sz_u64_t sz_u64_bits_reverse(sz_u64_t value) {
    value = ((value & 0x5555555555555555ull) << 1) | ((value >> 1) & 0x5555555555555555ull);
    value = ((value & 0x3333333333333333ull) << 2) | ((value >> 2) & 0x3333333333333333ull);
    value = ((value & 0x0F0F0F0F0F0F0F0Full) << 4) | ((value >> 4) & 0x0F0F0F0F0F0F0F0Full);
    return sz_u64_bytes_reverse(value);
}

/** Bit index of the n-th (0-based) set bit of @p bits; clears the @p n lowest set bits, then
 *  @c ctz. @p bits must hold more than @p n set bits. One tested home for the per-ISA SIMD
 *  "n-th lane" locate. */
STRINGZILLA_INLINE int sz_u64_nth_set_bit(sz_u64_t bits, sz_size_t n) {
    while (n--) bits &= bits - 1;
    return sz_u64_ctz(bits);
}
STRINGZILLA_INLINE int sz_u32_nth_set_bit(sz_u32_t bits, sz_size_t n) {
    while (n--) bits &= bits - 1;
    return sz_u32_ctz(bits);
}

/** Branchless <tt>value | (bit if condition)</tt>: OR @p bit into @p value when @p condition holds, with
 *  no branch - a mask-select rather than a CMOV, so there is no flag dependency. For threading a
 *  per-window carry signal into a lane mask. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_u64_or_if_(sz_u64_t value, sz_u64_t bit, int condition) {
    return value | (bit & ((sz_u64_t)0 - (sz_u64_t)(condition != 0)));
}

STRINGZILLA_CONSTEXPR sz_u64_t sz_u64_rotl(sz_u64_t x, sz_u64_t r) { return (x << r) | (x >> (64 - r)); }

/**
 *  @brief Select bits from either @p a or @p b depending on the value of @p mask bits.
 *
 *  Similar to the @c _mm_blend_epi16 intrinsic on x86.
 *
 *  @see Bit Twiddling Hacks by Sean Eron Anderson: https://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
 */
STRINGZILLA_CONSTEXPR sz_u64_t sz_u64_blend(sz_u64_t a, sz_u64_t b, sz_u64_t mask) { return a ^ ((a ^ b) & mask); }

/**
 *  @brief Efficiently computing the minimum and maximum of two or three values can be tricky.
 *
 *  A branchless approach is well known for signed integers, but it doesn't apply to unsigned ones.
 *  The simple branching baseline comes first, then the shift-only form for signed integers, then
 *  two forms for any integers, with and without multiplication:
 *
 *  @verbatim
 *  x < y ? x : y                                   can replace with 1 conditional move
 *  y + ((x - y) & (x - y) >> 31)                   4 unique operations
 *  (x > y) * y + (x <= y) * x                      5 operations
 *  x & ~((x < y) - 1) + y & ((x < y) - 1)          6 unique operations
 *  @endverbatim
 *
 *  @see Branchless min and max: https://stackoverflow.com/questions/514435/templatized-branchless-int-max-min-function
 *  @see Bit Twiddling Hacks: https://graphics.stanford.edu/~seander/bithacks.html#IntegerMinOrMax
 */
#define sz_min_of_two(x, y) ((x) < (y) ? (x) : (y))
#define sz_max_of_two(x, y) ((x) < (y) ? (y) : (x))
#define sz_min_of_three(x, y, z) sz_min_of_two(x, sz_min_of_two(y, z))
#define sz_max_of_three(x, y, z) sz_max_of_two(x, sz_max_of_two(y, z))

/**
 *  @brief Three-way comparison of two scalars, as two comparisons and a subtraction.
 *
 *  One way to avoid branching is to look up the comparison result in a table via conditional moves:
 *
 *  @code{.c}
 *      sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
 *      for (; a != min_end; ++a, ++b)
 *          if (*a != *b) return ordering_lookup[*a < *b];
 *  @endcode
 *
 *  That, however, introduces a data-dependency. Two comparisons and a subtraction take one
 *  instruction more, but have no data-dependency.
 */
#define sz_order_scalars_(a, b) ((sz_ordering_t)((a > b) - (a < b)))

/** Convenience macro to swap two values of the same type. */
#define sz_swap_(type, a, b) \
    do {                     \
        type _tmp = (a);     \
        (a) = (b);           \
        (b) = _tmp;          \
    } while (0)

/** Branchless minimum function for two signed 32-bit integers. */
STRINGZILLA_CONSTEXPR sz_i32_t sz_i32_min_of_two(sz_i32_t x, sz_i32_t y) { return y + ((x - y) & (x - y) >> 31); }

/** Branchless maximum function for two signed 32-bit integers. */
STRINGZILLA_CONSTEXPR sz_i32_t sz_i32_max_of_two(sz_i32_t x, sz_i32_t y) { return x - ((x - y) & (x - y) >> 31); }

/** Low @p count bits set (<tt>[0, count)</tt>), 0 for <tt>count==0</tt>, all-ones for <tt>count>=64</tt>. Portable,
 *  branch-light replacement for the BMI2 @c sz_u64_mask_until_ so the shared boundary algebra
 *  compiles on every backend. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_u64_mask_until_serial_(sz_size_t count) {
    return count >= 64 ? (sz_u64_t) ~(sz_u64_t)0 : (((sz_u64_t)1 << count) - 1);
}

/*  In AVX-512 we actively use masked operations and the "K mask registers". Producing a mask for
 *  the first N elements of a sequence can be done using the `1 << N - 1` idiom. It, however,
 *  induces undefined behavior if N is 64 or 32 on 64-bit or 32-bit systems respectively.
 *  Alternatively, the BZHI instruction can be used to clear the bits above N. */
#if STRINGZILLA_ARCH_X8664_SKYLAKE_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("bmi", "bmi2")
#endif
STRINGZILLA_INLINE __mmask8 sz_u8_mask_until_(sz_size_t n) { return (__mmask8)_bzhi_u32(0xFFu, (unsigned char)n); }
STRINGZILLA_INLINE __mmask16 sz_u16_mask_until_(sz_size_t n) { return (__mmask16)_bzhi_u32(0xFFFFu, (unsigned char)n); }
STRINGZILLA_INLINE __mmask32 sz_u32_mask_until_(sz_size_t n) {
    return (__mmask32)_bzhi_u64(0xFFFFFFFFu, (unsigned char)n);
}
STRINGZILLA_INLINE __mmask64 sz_u64_mask_until_(sz_size_t n) {
    return (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned char)n);
}
STRINGZILLA_CONSTEXPR __mmask8 sz_u8_clamp_mask_until_(sz_size_t n) { return n < 8 ? sz_u8_mask_until_(n) : 0xFFu; }
STRINGZILLA_CONSTEXPR __mmask16 sz_u16_clamp_mask_until_(sz_size_t n) {
    return n < 16 ? sz_u16_mask_until_(n) : 0xFFFFu;
}
STRINGZILLA_CONSTEXPR __mmask32 sz_u32_clamp_mask_until_(sz_size_t n) {
    return n < 32 ? sz_u32_mask_until_(n) : 0xFFFFFFFFu;
}
STRINGZILLA_CONSTEXPR __mmask64 sz_u64_clamp_mask_until_(sz_size_t n) {
    return n < 64 ? sz_u64_mask_until_(n) : 0xFFFFFFFFFFFFFFFFull;
}
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_X8664_SKYLAKE_

/**
 *  @brief Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each byte signifies a match.
 */
STRINGZILLA_CONSTEXPR sz_u64_vec_t sz_u64_each_byte_equal_(sz_u64_vec_t a_vec, sz_u64_vec_t b_vec) {
    sz_u64_vec_t result_vec;
    result_vec.u64 = ~(a_vec.u64 ^ b_vec.u64);
    // The match is valid, if every bit within each byte is set.
    // For that take the bottom 7 bits of each byte, add one to them,
    // and if this sets the top bit to one, then all the 7 bits are ones as well.
    result_vec.u64 = ((result_vec.u64 & 0x7F7F7F7F7F7F7F7Full) + 0x0101010101010101ull) &
                     ((result_vec.u64 & 0x8080808080808080ull));
    return result_vec;
}

/** Clamps signed offsets to a Python-style <tt>[offset, offset + length)</tt> slice and reports whether
 *  the requested window was non-degenerate. Mirrors CPython's @c ADJUST_INDICES: negative
 *  indices count from the end, @p end is clamped into the string, and a window with @p start
 *  past @p end, out of range or inverted, is "empty" - exactly when <tt>str.find("")</tt> and
 *  <tt>str.rfind("")</tt> return -1. */
STRINGZILLA_CONSTEXPR sz_bool_t sz_ssize_clamp_interval_checked( //
    sz_size_t length, sz_ssize_t start, sz_ssize_t end, sz_size_t *normalized_offset, sz_size_t *normalized_length) {
    sz_ssize_t const signed_length = (sz_ssize_t)length;
    if (start < 0) start += signed_length;
    if (end < 0) end += signed_length;
    if (start < 0) start = 0;
    if (end < 0) end = 0;
    if (end > signed_length) end = signed_length;
    sz_bool_t const window_valid = start <= end ? sz_true_k : sz_false_k;
    if (start > end) start = end; // Collapse the degenerate window to a usable zero-length slice
    *normalized_offset = (sz_size_t)start;
    *normalized_length = (sz_size_t)(end - start);
    return window_valid;
}

/**
 *  @brief Clamps signed offsets in a string to a valid range. Used for Pythonic-style slicing.
 *  @note Thin wrapper over @ref sz_ssize_clamp_interval_checked for callers ignoring validity.
 */
STRINGZILLA_CONSTEXPR void sz_ssize_clamp_interval( //
    sz_size_t length, sz_ssize_t start, sz_ssize_t end, sz_size_t *normalized_offset, sz_size_t *normalized_length) {
    sz_ssize_clamp_interval_checked(length, start, end, normalized_offset, normalized_length);
}

/**
 *  @brief Compute the logarithm base 2 of a positive integer, rounding down.
 *  @pre Input must be a positive number, as the logarithm of zero is undefined.
 */
STRINGZILLA_INLINE sz_size_t sz_size_log2i_nonzero(sz_size_t x) {
    sz_assert_(x > 0 && "Non-positive numbers have no defined logarithm");
    int leading_zeros = sz_u64_clz(x);
    return (sz_size_t)(63 - leading_zeros);
}

/** Computes the ceiling of @p x divided by @p divisor - the number of chunks of that size needed to
 *  cover @p x. Assumes a non-zero @p divisor and no overflow on <tt>x + divisor</tt>. */
STRINGZILLA_CONSTEXPR sz_size_t sz_size_divide_round_up(sz_size_t x, sz_size_t divisor) {
    return (x + divisor - 1) / divisor;
}

/** Divides rounding up in 32 bits, for device code where the @c sz_size_t form costs registers. */
STRINGZILLA_CONSTEXPR sz_u32_t sz_u32_divide_round_up(sz_u32_t x, sz_u32_t divisor) {
    return (x + divisor - 1) / divisor;
}

/** Turns a byte stride into entries, refusing one that splits an entry or spans fewer than
 *  @p minimum_entries. */
STRINGZILLA_CONSTEXPR sz_status_t sz_stride_entries_(sz_size_t stride_bytes, sz_size_t entry_bytes,
                                                     sz_size_t minimum_entries, sz_size_t *stride_entries) {
    if (stride_bytes % entry_bytes != 0 || stride_bytes / entry_bytes < minimum_entries)
        return sz_unexpected_dimensions_k;
    *stride_entries = stride_bytes / entry_bytes;
    return sz_success_k;
}

/**
 *  @brief Compute the smallest power of two greater than or equal to @p x.
 *  @note Uses LZCNT/CLZ for efficient computation on modern CPUs. Edge cases: bit_ceil(0) = 0,
 *      bit_ceil(1) = 1.
 *  @see Rounding up to a power of two: https://stackoverflow.com/a/10143264
 */
STRINGZILLA_INLINE sz_size_t sz_size_bit_ceil(sz_size_t x) {
#if defined(__LZCNT__) || defined(__BMI__)
    // Edge cases: 0 and 1 return themselves, avoids undefined clz(0).
    if (x <= 1) return x;
#if STRINGZILLA_ARCH_64BIT_
    return (sz_size_t)1 << (64 - sz_u64_clz(x - 1));
#else
    return (sz_size_t)1 << (32 - sz_u32_clz((sz_u32_t)(x - 1)));
#endif
#else
    // The following trick is valid for 0 input as well.
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if STRINGZILLA_ARCH_64BIT_
    x |= x >> 32;
#endif
    x++;
    return x;
#endif
}

/**
 *  @brief Transposes an 8×8 bit matrix packed in a @c sz_u64_t.
 *
 *  There is a well known SWAR sequence for that known to chess programmers, willing to flip a
 *  bit-matrix of pieces along the main A1-H8 diagonal.
 *
 *  @see Flipping, Mirroring and Rotating: https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating
 *  @see Transposing a bit matrix: https://lukas-prokop.at/articles/2021-07-23-transpose
 */
STRINGZILLA_CONSTEXPR sz_u64_t sz_u64_transpose(sz_u64_t x) {
    sz_u64_t t;
    t = x ^ (x << 36);
    x ^= 0xf0f0f0f00f0f0f0full & (t ^ (x >> 36));
    t = 0xcccc0000cccc0000ull & (x ^ (x << 18));
    x ^= t ^ (t >> 18);
    t = 0xaa00aa00aa00aa00ull & (x ^ (x << 9));
    x ^= t ^ (t >> 9);
    return x;
}

#if STRINGZILLA_ALLOW_MISALIGNED_LOADS && (!defined(_MSC_VER) || defined(__clang__))

/** The words at byte alignment, which the loads and stores below access memory through. The
 *  attribute must sit on a typedef: on a pointer's declaration it aligns the pointer variable, and
 *  GCC keeps assuming the pointee is word-aligned. */
typedef sz_u16_t __attribute__((aligned(1))) sz_u16_unaligned_t;
typedef sz_u32_t __attribute__((aligned(1))) sz_u32_unaligned_t;
typedef sz_u64_t __attribute__((aligned(1))) sz_u64_unaligned_t;
#endif

/** Load a 16-bit unsigned integer from a potentially unaligned pointer, slow on some platforms. */
STRINGZILLA_INLINE sz_u16_vec_t sz_u16_load(sz_cptr_t ptr) {
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
    sz_u16_vec_t result_vec;
    result_vec.u8s[0] = ptr[0];
    result_vec.u8s[1] = ptr[1];
    return result_vec;
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    return *((sz_u16_vec_t *)ptr);
#else
    return *((__unaligned sz_u16_vec_t *)ptr);
#endif
#else
    sz_u16_vec_t result_vec;
    result_vec.u16 = *(sz_u16_unaligned_t const *)ptr;
    return result_vec;
#endif
}

/** Load a 32-bit unsigned integer from a potentially unaligned pointer, slow on some platforms. */
STRINGZILLA_INLINE sz_u32_vec_t sz_u32_load(sz_cptr_t ptr) {
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
    sz_u32_vec_t result_vec;
    result_vec.u8s[0] = ptr[0];
    result_vec.u8s[1] = ptr[1];
    result_vec.u8s[2] = ptr[2];
    result_vec.u8s[3] = ptr[3];
    return result_vec;
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    return *((sz_u32_vec_t *)ptr);
#else
    return *((__unaligned sz_u32_vec_t *)ptr);
#endif
#else
    sz_u32_vec_t result_vec;
    result_vec.u32 = *(sz_u32_unaligned_t const *)ptr;
    return result_vec;
#endif
}

/** Load a 64-bit unsigned integer from a potentially unaligned pointer, slow on some platforms. */
STRINGZILLA_INLINE sz_u64_vec_t sz_u64_load(sz_cptr_t ptr) {
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
    sz_u64_vec_t result_vec;
    result_vec.u8s[0] = ptr[0];
    result_vec.u8s[1] = ptr[1];
    result_vec.u8s[2] = ptr[2];
    result_vec.u8s[3] = ptr[3];
    result_vec.u8s[4] = ptr[4];
    result_vec.u8s[5] = ptr[5];
    result_vec.u8s[6] = ptr[6];
    result_vec.u8s[7] = ptr[7];
    return result_vec;
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    return *((sz_u64_vec_t *)ptr);
#else
    return *((__unaligned sz_u64_vec_t *)ptr);
#endif
#else
    sz_u64_vec_t result_vec;
    result_vec.u64 = *(sz_u64_unaligned_t const *)ptr;
    return result_vec;
#endif
}

/** Store a 16-bit unsigned integer to a potentially unaligned pointer, slow on some platforms. */
STRINGZILLA_INLINE void sz_u16_store(sz_ptr_t ptr, sz_u16_t value) {
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
    sz_u16_vec_t vec;
    vec.u16 = value;
    ptr[0] = vec.u8s[0];
    ptr[1] = vec.u8s[1];
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    ((sz_u16_vec_t *)ptr)->u16 = value;
#else
    ((__unaligned sz_u16_vec_t *)ptr)->u16 = value;
#endif
#else
    *(sz_u16_unaligned_t *)ptr = value;
#endif
}

/** Store a 32-bit unsigned integer to a potentially unaligned pointer, slow on some platforms. */
STRINGZILLA_INLINE void sz_u32_store(sz_ptr_t ptr, sz_u32_t value) {
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
    sz_u32_vec_t vec;
    vec.u32 = value;
    ptr[0] = vec.u8s[0];
    ptr[1] = vec.u8s[1];
    ptr[2] = vec.u8s[2];
    ptr[3] = vec.u8s[3];
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    ((sz_u32_vec_t *)ptr)->u32 = value;
#else
    ((__unaligned sz_u32_vec_t *)ptr)->u32 = value;
#endif
#else
    *(sz_u32_unaligned_t *)ptr = value;
#endif
}

/** Store a 64-bit unsigned integer to a potentially unaligned pointer, slow on some platforms. */
STRINGZILLA_INLINE void sz_u64_store(sz_ptr_t ptr, sz_u64_t value) {
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
    sz_u64_vec_t vec;
    vec.u64 = value;
    ptr[0] = vec.u8s[0];
    ptr[1] = vec.u8s[1];
    ptr[2] = vec.u8s[2];
    ptr[3] = vec.u8s[3];
    ptr[4] = vec.u8s[4];
    ptr[5] = vec.u8s[5];
    ptr[6] = vec.u8s[6];
    ptr[7] = vec.u8s[7];
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    ((sz_u64_vec_t *)ptr)->u64 = value;
#else
    ((__unaligned sz_u64_vec_t *)ptr)->u64 = value;
#endif
#else
    *(sz_u64_unaligned_t *)ptr = value;
#endif
}

typedef struct STRINGZILLA_MAY_ALIAS_ sz_arena_t_ {
    sz_size_t capacity, consumed, alignment;
} sz_arena_t_;

/** Allocates an aligned block from the arena, leaving its consumption unchanged on failure. */
STRINGZILLA_CONSTEXPR void *sz_allocate_arena_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    sz_unused_(stream);
    sz_arena_t_ *metadata = (sz_arena_t_ *)handle;
    sz_size_t const capacity = metadata->capacity, consumed = metadata->consumed, alignment = metadata->alignment;
    if (consumed > capacity) return STRINGZILLA_NULL;
    sz_size_t const padding = (0 - ((sz_size_t)handle + consumed)) & (alignment - 1);
    if (padding > capacity - consumed || bytes > capacity - consumed - padding) return STRINGZILLA_NULL;
    metadata->consumed = consumed + padding + bytes;
    return (sz_ptr_t)handle + consumed + padding;
}

/** Helper "no-op" function, simulating memory deallocation when we use a "static" memory buffer. */
STRINGZILLA_CONSTEXPR void sz_free_arena_(void *start, sz_size_t length, void *handle, sz_stream_t stream) {
    sz_unused_(start && length && handle && stream);
}

/** Resizes the arena's last block where it lies, and moves any other block into a new one. */
STRINGZILLA_CONSTEXPR void *sz_reallocate_arena_(void *start, sz_size_t old_bytes, sz_size_t new_bytes, void *handle,
                                                 sz_stream_t stream) {
    sz_arena_t_ *metadata = (sz_arena_t_ *)handle;
    if (!start) return sz_allocate_arena_(new_bytes, handle, stream);
    sz_size_t const start_offset = (sz_size_t)((sz_ptr_t)start - (sz_ptr_t)handle);
    if (start_offset + old_bytes == metadata->consumed) {
        if (new_bytes > metadata->capacity - start_offset) return STRINGZILLA_NULL;
        metadata->consumed = start_offset + new_bytes;
        return start;
    }
    sz_ptr_t const moved = (sz_ptr_t)sz_allocate_arena_(new_bytes, handle, stream);
    if (!moved) return STRINGZILLA_NULL;
    sz_cptr_t const kept = (sz_cptr_t)start;
    for (sz_size_t byte_index = 0; byte_index != sz_min_of_two(old_bytes, new_bytes); ++byte_index)
        moved[byte_index] = kept[byte_index];
    return moved;
}

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#pragma endregion

#pragma region Serial Implementation

#if STRINGZILLA_WITH_LIBC
#include <stdio.h>  // `fprintf`
#include <stdlib.h> // `malloc`, `EXIT_FAILURE`

STRINGZILLA_INLINE void *sz_allocate_heap_(sz_size_t length, void *handle, sz_stream_t stream) {
    sz_unused_(handle && stream);
    if (length == 0) return STRINGZILLA_NULL;
    return malloc(length);
}
STRINGZILLA_INLINE void sz_free_heap_(void *start, sz_size_t length, void *handle, sz_stream_t stream) {
    sz_unused_(handle && length && stream);
    free(start);
}
STRINGZILLA_INLINE void *sz_reallocate_heap_(void *start, sz_size_t old_bytes, sz_size_t new_bytes, void *handle,
                                             sz_stream_t stream) {
    sz_unused_(handle && old_bytes && stream);
    if (new_bytes == 0) {
        free(start);
        return STRINGZILLA_NULL;
    }
    return realloc(start, new_bytes);
}

#endif

STRINGZILLA_INLINE sz_status_t sz_allocator_init_heap(sz_allocator_t *allocator) {
#if STRINGZILLA_WITH_LIBC
    allocator->allocate = sz_allocate_heap_;
    allocator->reallocate = sz_reallocate_heap_;
    allocator->free = sz_free_heap_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
#else
    sz_unused_(allocator);
    return sz_missing_kernel_k;
#endif
}

STRINGZILLA_INLINE sz_status_t sz_allocator_init_arena(sz_allocator_t *allocator, void *buffer, sz_size_t bytes,
                                                       sz_size_t alignment) {
    if (!alignment || (alignment & (alignment - 1))) return sz_unexpected_dimensions_k;
    if (!buffer) return sz_bad_alloc_k;
    if (bytes > STRINGZILLA_SIZE_MAX - (sz_size_t)buffer) return sz_overflow_risk_k;
    sz_size_t const padding = (0 - (sz_size_t)buffer) & (sizeof(sz_size_t) - 1);
    if (padding > bytes || bytes - padding < sizeof(sz_arena_t_)) return sz_bad_alloc_k;
    sz_arena_t_ *metadata = (sz_arena_t_ *)((sz_ptr_t)buffer + padding);
    metadata->capacity = bytes - padding;
    metadata->consumed = sizeof(sz_arena_t_);
    metadata->alignment = alignment;
    allocator->allocate = sz_allocate_arena_;
    allocator->reallocate = sz_reallocate_arena_;
    allocator->free = sz_free_arena_;
    allocator->handle = metadata;
    return sz_success_k;
}

STRINGZILLA_CONSTEXPR sz_bool_t sz_allocator_equal(sz_allocator_t const *a, sz_allocator_t const *b) {
    if (!a || !b) return sz_false_k;

    // Two allocators are considered equal if they have the same function pointers and handle
    return (a->allocate == b->allocate) && (a->reallocate == b->reallocate) && (a->free == b->free) &&
                   (a->handle == b->handle)
               ? sz_true_k
               : sz_false_k;
}

/** Writes the views of the strings @p indices of @p strings, in any of its three layouts. */
STRINGZILLA_INLINE void sz_strings_get_(sz_strings_view_t const *strings, sz_size_t const *indices, sz_size_t count,
                                        sz_string_view_t *views) {
    switch (strings->layout) {
    case sz_strings_accessor_k: strings->data.accessor.get(strings->data.accessor.handle, indices, count, views); break;
    case sz_strings_u32_tape_k:
        for (sz_size_t position = 0; position != count; ++position) {
            sz_u32_t const *offsets = strings->data.u32_tape.offsets + indices[position];
            views[position].start = strings->data.u32_tape.data + offsets[0];
            views[position].length = offsets[1] - offsets[0];
        }
        break;
    case sz_strings_u64_tape_k:
        for (sz_size_t position = 0; position != count; ++position) {
            sz_u64_t const *offsets = strings->data.u64_tape.offsets + indices[position];
            views[position].start = strings->data.u64_tape.data + offsets[0];
            views[position].length = (sz_size_t)(offsets[1] - offsets[0]);
        }
        break;
    }
}

/** Writes the views of up to 64 strings of @p strings from @p start on, returning how many. */
STRINGZILLA_INLINE sz_size_t sz_strings_get_block_(sz_strings_view_t const *strings, sz_size_t start,
                                                   sz_string_view_t *views) {
    sz_size_t const count = sz_min_of_two(strings->count - start, (sz_size_t)64);
    sz_size_t indices[64];
    for (sz_size_t position = 0; position != count; ++position) indices[position] = start + position;
    sz_strings_get_(strings, indices, count, views);
    return count;
}

/** The indices of the @p count keys leading the @b [3 × count] @p scratch, ascending and stable: a
 *  least significant digit radix over only the bytes that differ across the keys, so keys under 256
 *  cost one counting pass and one scatter. */
STRINGZILLA_INLINE sz_size_t const *sz_keys_order_(sz_size_t *scratch, sz_size_t count) {
    sz_size_t const *const keys = scratch;
    sz_size_t *source = scratch + count, *target = source + count;
    sz_size_t all_ones = ~(sz_size_t)0, any_ones = 0;
    for (sz_size_t rank = 0; rank != count; ++rank) source[rank] = rank, all_ones &= keys[rank], any_ones |= keys[rank];
    sz_size_t const varying = all_ones ^ any_ones;
    for (sz_size_t shift = 0; shift != sizeof(sz_size_t) * 8; shift += 8) {
        if (((varying >> shift) & 0xFF) == 0) continue;
        sz_size_t offsets[256] = {0};
        for (sz_size_t rank = 0; rank != count; ++rank) ++offsets[(keys[source[rank]] >> shift) & 0xFF];
        for (sz_size_t digit = 0, running = 0; digit != 256; ++digit) {
            sz_size_t const bucket = offsets[digit];
            offsets[digit] = running, running += bucket;
        }
        for (sz_size_t rank = 0; rank != count; ++rank)
            target[offsets[(keys[source[rank]] >> shift) & 0xFF]++] = source[rank];
        sz_size_t *const sorted = target;
        target = source, source = sorted;
    }
    return source;
}

/** The indices of @p strings ascending by length, stably, in the @b [3 × count] @p scratch. */
STRINGZILLA_INLINE sz_size_t const *sz_strings_order_by_length_(sz_strings_view_t const *strings, sz_size_t *scratch) {
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < strings->count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(strings, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position)
            scratch[block_start + position] = views[position].length;
    }
    return sz_keys_order_(scratch, strings->count);
}

/** Offset @p index of the tape @p strings views, which is never an accessor. */
STRINGZILLA_INLINE sz_size_t sz_strings_view_offset_(sz_strings_view_t const *strings, sz_size_t index) {
    return strings->layout == sz_strings_u64_tape_k ? (sz_size_t)strings->data.u64_tape.offsets[index]
                                                    : (sz_size_t)strings->data.u32_tape.offsets[index];
}

/** The payload of the tape @p strings views, which is never an accessor. */
STRINGZILLA_INLINE sz_cptr_t sz_strings_view_data_(sz_strings_view_t const *strings) {
    return strings->layout == sz_strings_u64_tape_k ? strings->data.u64_tape.data : strings->data.u32_tape.data;
}

STRINGZILLA_INLINE void sz_strings_from_null_terminated_strings_get_(void const *handle, sz_size_t const *indices,
                                                                     sz_size_t count, sz_string_view_t *views) {
    sz_cptr_t const *starts = (sz_cptr_t const *)handle;
    for (sz_size_t position = 0; position != count; ++position) {
        sz_cptr_t const start = starts[indices[position]];
        sz_size_t length = 0;
        while (start[length]) ++length;
        views[position].start = start;
        views[position].length = length;
    }
}

STRINGZILLA_INLINE void sz_strings_from_string_views_get_(void const *handle, sz_size_t const *indices, sz_size_t count,
                                                          sz_string_view_t *views) {
    sz_string_view_t const *source_views = (sz_string_view_t const *)handle;
    for (sz_size_t position = 0; position != count; ++position) views[position] = source_views[indices[position]];
}

STRINGZILLA_INLINE void sz_strings_from_records_get_(void const *handle, sz_size_t const *indices, sz_size_t count,
                                                     sz_string_view_t *views) {
    sz_strings_crecords_t const *records = (sz_strings_crecords_t const *)handle;
    for (sz_size_t position = 0; position != count; ++position) {
        sz_cptr_t const record = (sz_cptr_t)records->records + indices[position] * records->stride;
        views[position].start = *(sz_cptr_t const *)(record + records->start_member);
        views[position].length = *(sz_size_t const *)(record + records->length_member);
    }
}

STRINGZILLA_INLINE void sz_strings_from_record_pointers_get_(void const *handle, sz_size_t const *indices,
                                                             sz_size_t count, sz_string_view_t *views) {
    sz_strings_crecords_t const *records = (sz_strings_crecords_t const *)handle;
    for (sz_size_t position = 0; position != count; ++position) {
        sz_cptr_t const record = *(sz_cptr_t const *)((sz_cptr_t)records->records +
                                                      indices[position] * records->stride);
        views[position].start = *(sz_cptr_t const *)(record + records->start_member);
        views[position].length = *(sz_size_t const *)(record + records->length_member);
    }
}

STRINGZILLA_INLINE void sz_strings_from_rows_get_(void const *handle, sz_size_t const *indices, sz_size_t count,
                                                  sz_string_view_t *views) {
    sz_strings_crows_t const *rows = (sz_strings_crows_t const *)handle;
    for (sz_size_t position = 0; position != count; ++position) {
        sz_cptr_t const row = rows->data + indices[position] * rows->stride;
        sz_size_t length = rows->width;
        while (length && !row[length - 1]) --length;
        views[position].start = row;
        views[position].length = length;
    }
}

STRINGZILLA_INLINE void sz_strings_from_null_terminated_strings(sz_cptr_t const *start, sz_size_t count,
                                                                sz_strings_view_t *strings) {
    strings->layout = sz_strings_accessor_k;
    strings->count = count;
    strings->data.accessor.get = sz_strings_from_null_terminated_strings_get_;
    strings->data.accessor.handle = start;
}

STRINGZILLA_INLINE void sz_strings_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                     sz_strings_view_t *strings) {
    strings->layout = sz_strings_accessor_k;
    strings->count = count;
    strings->data.accessor.get = sz_strings_from_string_views_get_;
    strings->data.accessor.handle = views;
}

STRINGZILLA_INLINE void sz_strings_from_records(sz_strings_crecords_t const *records, sz_size_t count,
                                                sz_strings_view_t *strings) {
    strings->layout = sz_strings_accessor_k;
    strings->count = count;
    strings->data.accessor.get = sz_strings_from_records_get_;
    strings->data.accessor.handle = records;
}

STRINGZILLA_INLINE void sz_strings_from_record_pointers(sz_strings_crecords_t const *records, sz_size_t count,
                                                        sz_strings_view_t *strings) {
    strings->layout = sz_strings_accessor_k;
    strings->count = count;
    strings->data.accessor.get = sz_strings_from_record_pointers_get_;
    strings->data.accessor.handle = records;
}

STRINGZILLA_INLINE void sz_strings_from_rows(sz_strings_crows_t const *rows, sz_size_t count,
                                             sz_strings_view_t *strings) {
    strings->layout = sz_strings_accessor_k;
    strings->count = count;
    strings->data.accessor.get = sz_strings_from_rows_get_;
    strings->data.accessor.handle = rows;
}

STRINGZILLA_INLINE void sz_strings_slice(sz_strings_view_t const *strings, sz_size_t start, sz_ssize_t step,
                                         sz_size_t count, sz_string_view_t *views, sz_strings_view_t *slice) {
    if (step == 1 && strings->layout != sz_strings_accessor_k) {
        *slice = *strings;
        slice->count = count;
        if (strings->layout == sz_strings_u32_tape_k) slice->data.u32_tape.offsets += start;
        else slice->data.u64_tape.offsets += start;
        return;
    }
    sz_size_t indices[64];
    for (sz_size_t block_start = 0; block_start < count; block_start += 64) {
        sz_size_t const block_count = sz_min_of_two(count - block_start, (sz_size_t)64);
        for (sz_size_t position = 0; position != block_count; ++position)
            indices[position] = (sz_size_t)((sz_ssize_t)start + (sz_ssize_t)(block_start + position) * step);
        sz_strings_get_(strings, indices, block_count, views + block_start);
    }
    sz_strings_from_string_views(views, count, slice);
}

STRINGZILLA_INLINE void sz_strings_permute(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                           sz_string_view_t *views, sz_strings_view_t *permuted) {
    sz_strings_get_(strings, order, count, views);
    sz_strings_from_string_views(views, count, permuted);
}

STRINGZILLA_INLINE void sz_strings_from_matches(sz_strings_view_t const *haystacks, sz_match_t const *matches,
                                                sz_size_t count, sz_string_view_t *views, sz_strings_view_t *strings) {
    sz_size_t indices[64];
    for (sz_size_t block_start = 0; block_start < count; block_start += 64) {
        sz_size_t const block_count = sz_min_of_two(count - block_start, (sz_size_t)64);
        for (sz_size_t position = 0; position != block_count; ++position)
            indices[position] = matches[block_start + position].haystack_index;
        sz_strings_get_(haystacks, indices, block_count, views + block_start);
        for (sz_size_t position = 0; position != block_count; ++position) {
            views[block_start + position].start += matches[block_start + position].byte_offset;
            views[block_start + position].length = matches[block_start + position].byte_length;
        }
    }
    sz_strings_from_string_views(views, count, strings);
}

STRINGZILLA_INLINE void sz_strings_from_span(sz_strings_span_t const *span, sz_strings_view_t *strings) {
    strings->layout = span->layout;
    strings->count = span->count;
    if (span->layout == sz_strings_u64_tape_k)
        strings->data.u64_tape.data = span->data.u64_tape.data,
        strings->data.u64_tape.offsets = span->data.u64_tape.offsets;
    else
        strings->data.u32_tape.data = span->data.u32_tape.data,
        strings->data.u32_tape.offsets = span->data.u32_tape.offsets;
}

STRINGZILLA_INLINE void sz_strings_span_free(sz_strings_span_t *span, sz_allocator_t *allocator, sz_stream_t stream) {
    if (span->layout == sz_strings_u32_tape_k && span->data.u32_tape.offsets) {
        sz_u32_t *const offsets = span->data.u32_tape.offsets;
        allocator->free(span->data.u32_tape.data, offsets[span->count], allocator->handle, stream);
        allocator->free(offsets, (span->count + 1) * sizeof(sz_u32_t), allocator->handle, stream);
    }
    else if (span->layout == sz_strings_u64_tape_k && span->data.u64_tape.offsets) {
        sz_u64_t *const offsets = span->data.u64_tape.offsets;
        allocator->free(span->data.u64_tape.data, (sz_size_t)offsets[span->count], allocator->handle, stream);
        allocator->free(offsets, (span->count + 1) * sizeof(sz_u64_t), allocator->handle, stream);
    }
    span->layout = sz_strings_u32_tape_k, span->count = 0;
    span->data.u32_tape.data = STRINGZILLA_NULL, span->data.u32_tape.offsets = STRINGZILLA_NULL;
}

#pragma endregion

#ifdef __cplusplus
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}
#endif // __cplusplus

#endif // STRINGZILLA_TYPES_H_
