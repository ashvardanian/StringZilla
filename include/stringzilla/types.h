/**
 *  @brief Shared definitions for the StringZilla library.
 *  @file include/stringzilla/types.h
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
 *  - `sz_byteset_t` - a bitset for 256 possible byte values.
 */
#if !defined(STRINGZILLA_TYPES_H_)
#define STRINGZILLA_TYPES_H_

/*
 *  Debugging and testing.
 */
#if !defined(SZ_DEBUG)
#if defined(DEBUG) || defined(_DEBUG) // This means "Not using DEBUG information".
#define SZ_DEBUG (1)
#else
#define SZ_DEBUG (0)
#endif
#endif

/**
 *  @brief When set to 1, the library will include the following LibC headers: <stddef.h> and <stdint.h>.
 *         In debug builds (SZ_DEBUG=1), the library will also include <stdio.h> and <stdlib.h>.
 *
 *  You may want to disable this compiling for use in the kernel, or in embedded systems.
 *  You may also avoid them, if you are very sensitive to compilation time and avoid pre-compiled headers.
 *  https://artificial-mind.net/projects/compile-health/
 */
#if !defined(SZ_AVOID_LIBC)
#define SZ_AVOID_LIBC (0) // true or false
#endif

/**
 *  @brief Removes compile-time dispatching, and replaces it with runtime dispatching.
 *         So the `sz_find` function will invoke the most advanced backend supported by the CPU,
 *         that runs the program, rather than the most advanced backend supported by the CPU
 *         used to compile the library or the downstream application.
 */
#if !defined(SZ_DYNAMIC_DISPATCH)
#define SZ_DYNAMIC_DISPATCH (0) // true or false
#endif

/**
 *  @brief A misaligned load can be - trying to fetch eight consecutive bytes from an address
 *         that is not divisible by eight. On x86 enabled by default. On ARM it's not.
 *
 *  Most platforms support it, but there is no industry standard way to check for those.
 *  This value will mostly affect the performance of the serial (SWAR) backend.
 */
#if !defined(SZ_USE_MISALIGNED_LOADS)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define SZ_USE_MISALIGNED_LOADS (1) // true or false
#else
#define SZ_USE_MISALIGNED_LOADS (0) // true or false
#endif
#endif

/**
 *  @brief Analogous to `size_t` and `std::size_t`, unsigned integer, identical to pointer size.
 *         64-bit on most platforms where pointers are 64-bit.
 *         32-bit on platforms where pointers are 32-bit.
 *
 *  @note Do not use `defined(SZ_IS_64BIT_X86_)` or `defined(SZ_IS_64BIT_ARM_)` here — those indicate
 *        the CPU family, not pointer width. Rely on compiler/OS macros only.
 */
#if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_WIN64) || defined(__aarch64__) || \
    defined(__arm64__) || defined(__arm64) || defined(_M_ARM64)
#define SZ_IS_64BIT_ (1)
#else
#define SZ_IS_64BIT_ (0)
#endif

/**
 *  @brief On Big-Endian machines StringZilla will work in compatibility mode.
 *         This disables SWAR hacks to minimize code duplication, assuming practically
 *         all modern popular platforms are Little-Endian.
 *
 *  This variable is hard to infer from macros reliably. It's best to set it manually.
 *  For that CMake provides the `TestBigEndian` and `CMAKE_<LANG>_BYTE_ORDER` (from 3.20 onwards).
 *  In Python one can check `sys.byteorder == 'big'` in the `setup.py` script and pass the appropriate macro.
 *  https://stackoverflow.com/a/27054190
 *
 *  Modern compilers typically define __BYTE_ORDER__ and __ORDER_BIG_ENDIAN__.
 *  Fall back to legacy macros and known arch tags when unavailable.
 */
#if !defined(SZ_IS_BIG_ENDIAN_)
#if (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)) ||                                         \
    (defined(__BYTE_ORDER) && (__BYTE_ORDER == __BIG_ENDIAN)) || defined(__BIG_ENDIAN__) || defined(_BIG_ENDIAN) ||  \
    defined(__ARMEB__) || defined(__THUMBEB__) || defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) || \
    defined(__MIBSEB__) || defined(__s390x__) || defined(__s390__)
#define SZ_IS_BIG_ENDIAN_ (1) //< It's a big-endian target architecture
#else
#define SZ_IS_BIG_ENDIAN_ (0) //< It's a little-endian target architecture
#endif
#endif

/**
 *  @brief Infer the target architecture, unless it's overriden by the build system.
 *         At this point we only provide optimized backends for x86_64 and ARM64.
 */
#if !defined(SZ_IS_64BIT_X86_)
#if defined(__x86_64__) || defined(_M_X64)
#define SZ_IS_64BIT_X86_ (1)
#else
#define SZ_IS_64BIT_X86_ (0)
#endif
#endif
#if !defined(SZ_IS_64BIT_ARM_)
#if defined(__aarch64__) || defined(__arm64__) || defined(__arm64) || defined(_M_ARM64)
#define SZ_IS_64BIT_ARM_ (1)
#else
#define SZ_IS_64BIT_ARM_ (0)
#endif
#endif

/**
 *  @brief Threshold for switching to SWAR (8-bytes at a time) backend over serial byte-level for-loops.
 *         On very short strings, under 16 bytes long, at most a single word will be processed with SWAR.
 *         Assuming potentially misaligned loads, SWAR makes sense only after ~24 bytes.
 */
#if !defined(SZ_SWAR_THRESHOLD)
#if SZ_DEBUG
#define SZ_SWAR_THRESHOLD (8u) // 8 bytes in debug builds
#else
#define SZ_SWAR_THRESHOLD (24u) // 24 bytes in release builds
#endif
#endif

/*  Function annotations on two axes — role and (for internal helpers) inlining policy:
 *  - `SZ_API_COMPTIME`    public API, ISA tier resolved at compile time (header-inline).
 *  - `SZ_API_RUNTIME`     public API dispatched at runtime — the only role with cross-TU linkage.
 *  - `SZ_HELPER_AUTO`     internal helper; compiler decides inlining (same expansion as `SZ_API_COMPTIME`).
 *  - `SZ_HELPER_INLINE`   internal helper forced inline (structural: devirtualizing driver loops).
 *  - `SZ_HELPER_NOINLINE` internal helper forced out-of-line. Omits `inline` deliberately — GCC ignores
 *    `noinline` on an `inline` function. */

#if defined(__cplusplus)
#define SZ_C_INLINE inline
#else
#define SZ_C_INLINE inline static
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SZ_MAYBE_UNUSED __attribute__((unused))
#else
#define SZ_MAYBE_UNUSED
#endif

#if defined(_MSC_VER)
#define SZ_HELPER_INLINE __forceinline static
#define SZ_HELPER_NOINLINE __declspec(noinline) static
#else
#define SZ_HELPER_INLINE __attribute__((always_inline)) SZ_C_INLINE
#define SZ_HELPER_NOINLINE static __attribute__((noinline))
#endif

#define SZ_API_COMPTIME SZ_MAYBE_UNUSED SZ_C_INLINE
#define SZ_HELPER_AUTO SZ_MAYBE_UNUSED SZ_C_INLINE

// Exported symbol under dynamic dispatch or `SZ_EXPORT` (emitted from one amalgamation TU, links like a
// normal C library — the Rust binding without `dynamic-dispatch`); otherwise a header-inline tier.
#if SZ_DYNAMIC_DISPATCH || (defined(SZ_EXPORT) && SZ_EXPORT)
#if defined(_WIN32) || defined(__CYGWIN__)
#define SZ_API_RUNTIME __declspec(dllexport)
#else
#define SZ_API_RUNTIME extern __attribute__((visibility("default")))
#endif // _WIN32 || __CYGWIN__
#else
#define SZ_API_RUNTIME SZ_C_INLINE
#endif // SZ_DYNAMIC_DISPATCH || SZ_EXPORT

// CUDA device-side inlining policy (only meaningful under nvcc; these functions exist only on the device).
#if defined(__CUDACC__)
#define SZ_DEVICE_INLINE __device__ __forceinline__
#define SZ_DEVICE_NOINLINE __device__ __noinline__
#endif

/**
 *  @brief Disables stack protection for performance-critical functions.
 *
 *  GCC's `-fstack-protector-strong` inserts stack canary checks for functions with local arrays
 *  or buffers. For hash functions that use fixed-size state structures, this is unnecessary
 *  overhead (~10 cycles per call). This macro opts out of stack protection for such functions.
 */
#if defined(__GNUC__) || defined(__clang__)
#define SZ_NO_STACK_PROTECTOR __attribute__((no_stack_protector))
#else
#define SZ_NO_STACK_PROTECTOR
#endif

/**
 *  @brief Alignment macro for N-byte alignment.
 */
#if defined(_MSC_VER)
#define sz_align_(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
#define sz_align_(n) __attribute__((aligned(n)))
#else
#define sz_align_(n)
#endif

/**
 *  @brief Lets a type legally alias any other, so the SIMD vector unions' differently-typed views of one
 *         storage (e.g. reading a `sz_u8_t` state buffer back through `sz_u128_vec_t`) are well-defined.
 *         Without it GCC at `-O3` assumes the views never alias and may reorder them, corrupting results.
 */
#if defined(__GNUC__) || defined(__clang__)
#define SZ_MAY_ALIAS __attribute__((__may_alias__))
#else
#define SZ_MAY_ALIAS
#endif

/**
 *  @brief C99 static array parameter annotation for minimum array size.
 *         In C, expands to `static n` enabling compiler bounds checking.
 *         In C++, expands to nothing as this syntax is not supported.
 *
 *  @see https://lwn.net/Articles/1046840/
 *
 *  Example usage:
 *  @code{.c}
 *      void hash_digest(uint8_t digest[sz_at_least_(32)]);
 *      void lookup(uint8_t const lut[sz_at_least_(256)]);
 *  @endcode
 */
#if defined(__cplusplus) || defined(_MSC_VER)
#define sz_at_least_(n)
#else
#define sz_at_least_(n) static n
#endif

/**
 *  @brief Largest value that fits into 16 bits.
 */
#define SZ_U16_MAX (65535u)

/**
 *  @brief Largest prime number that fits into 16 bits.
 */
#define SZ_U16_MAX_PRIME (65521u)

/**
 *  @brief Largest prime number that fits into 31 bits.
 */
#define SZ_U32_MAX_PRIME (2147483647u)

/**
 *  @brief Largest prime number that fits into 64 bits.
 *  @see https://mersenneforum.org/showthread.php?t=3471
 *
 *  2^64 = 18,446,744,073,709,551,616
 *  this = 18,446,744,073,709,551,557
 *  diff = 59
 */
#define SZ_U64_MAX_PRIME (18446744073709551557ull)

/*  Opt into glibc's "misc" extensions (`_DEFAULT_SOURCE` => `__USE_MISC`) before the first libc header is
 *  pulled in. A strict `-std=cNN` otherwise hides declarations like `syscall()`, which the RISC-V
 *  `riscv_hwprobe` capability probe in `stringzilla.h` relies on. As the first StringZilla header every
 *  translation unit includes, this is the one place that covers every build system (CMake/Cargo/setuptools). */
#if defined(__linux__) && !SZ_AVOID_LIBC && !defined(_DEFAULT_SOURCE) && !defined(_GNU_SOURCE)
#define _DEFAULT_SOURCE 1
#endif

#if !SZ_AVOID_LIBC
#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`
#endif

/*  The headers needed for the `sz_assert_failure_` function. */
#if SZ_DEBUG && defined(SZ_AVOID_LIBC) && !SZ_AVOID_LIBC && !defined(SZ_PIC)
#include <stdio.h>  // `fprintf`, `stderr`
#include <stdlib.h> // `EXIT_FAILURE`
#endif

/*  Compile-time hardware features detection.
 *  All of those can be controlled by the user.
 */
#if !defined(SZ_USE_WESTMERE)
#if SZ_IS_64BIT_X86_ && defined(__SSE4_2__) && defined(__AES__)
#define SZ_USE_WESTMERE (1)
#elif SZ_IS_64BIT_X86_ && defined(_MSC_VER) && defined(__AVX__)
#define SZ_USE_WESTMERE (1) // ! MSVC doesn't expose `__SSE4_2__`, `__AES__` macros
#else
#define SZ_USE_WESTMERE (0)
#endif
#endif

#if !defined(SZ_USE_HASWELL)
#if SZ_IS_64BIT_X86_ && defined(__AVX2__)
#define SZ_USE_HASWELL (1)
#else
#define SZ_USE_HASWELL (0)
#endif
#endif

#if !defined(SZ_USE_GOLDMONT)
#if SZ_IS_64BIT_X86_ && defined(__SHA__)
#define SZ_USE_GOLDMONT (1)
#elif SZ_IS_64BIT_X86_ && defined(_MSC_VER) && defined(__AVX2__)
#define SZ_USE_GOLDMONT (1) // ! MSVC doesn't expose `__SHA__` macros
#else
#define SZ_USE_GOLDMONT (0)
#endif
#endif

#if !defined(SZ_USE_SKYLAKE)
#if SZ_IS_64BIT_X86_ && defined(__AVX512F__)
#define SZ_USE_SKYLAKE (1)
#else
#define SZ_USE_SKYLAKE (0)
#endif
#endif

#if !defined(SZ_USE_ICELAKE)
#if SZ_IS_64BIT_X86_ && defined(__AVX512BW__) && defined(__VAES__)
#define SZ_USE_ICELAKE (1)
#elif SZ_IS_64BIT_X86_ && defined(_MSC_VER) && defined(__AVX512BW__)
#define SZ_USE_ICELAKE (1) // ! MSVC doesn't expose `__VAES__` macros
#else
#define SZ_USE_ICELAKE (0)
#endif
#endif

/*  NEON support is optional in Armv7/AArch32, but mandatory from 8.0 onwards. */
#if !defined(SZ_USE_NEON)
#if SZ_IS_64BIT_ARM_ && defined(__ARM_NEON)
#define SZ_USE_NEON (1)
#elif SZ_IS_64BIT_ARM_ && defined(_MSC_VER) && defined(_M_ARM64)
#define SZ_USE_NEON (1) // ! MSVC doesn't expose `__ARM_NEON` macros
#else
#define SZ_USE_NEON (0)
#endif
#endif

/*  SVE is optional since Armv8.2-A, but never became mandatory and MSVC has no way to probe for it. */
#if !defined(SZ_USE_SVE)
#if SZ_IS_64BIT_ARM_ && defined(__ARM_FEATURE_SVE)
#define SZ_USE_SVE (1)
#else
#define SZ_USE_SVE (0)
#endif
#endif

/*  SVE2 is optional since Armv9.0-A, but never became mandatory and MSVC has no way to probe for it. */
#if !defined(SZ_USE_SVE2)
#if SZ_IS_64BIT_ARM_ && defined(__ARM_FEATURE_SVE2)
#define SZ_USE_SVE2 (1)
#else
#define SZ_USE_SVE2 (0)
#endif
#endif

/*  AES is optional since Armv8.0-A, but never became mandatory and MSVC has no way to probe for it. */
#if !defined(SZ_USE_NEONAES)
#if SZ_IS_64BIT_ARM_ && (defined(__ARM_FEATURE_AES) || defined(__ARM_FEATURE_CRYPTO) || defined(__APPLE__))
#define SZ_USE_NEONAES (1)
#else
#define SZ_USE_NEONAES (0)
#endif
#endif

/*  SHA2 is optional since Armv8.0-A, but never became mandatory and MSVC has no way to probe for it. */
#if !defined(SZ_USE_NEONSHA)
#if SZ_IS_64BIT_ARM_ && (defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO) || defined(__APPLE__))
#define SZ_USE_NEONSHA (1)
#else
#define SZ_USE_NEONSHA (0)
#endif
#endif

/*  SVE2 AES is optional since Armv9.0-A, but never became mandatory and MSVC has no way to probe for it.
 *  GCC spells the feature macro `__ARM_FEATURE_SVE2AES`; Clang spells it `__ARM_FEATURE_SVE2_AES`. Accept both. */
#if !defined(SZ_USE_SVE2AES)
#if SZ_IS_64BIT_ARM_ && (defined(__ARM_FEATURE_SVE2AES) || defined(__ARM_FEATURE_SVE2_AES))
#define SZ_USE_SVE2AES (1)
#else
#define SZ_USE_SVE2AES (0)
#endif
#endif

/**  SVE isn't a silver bullet. Oftentimes its still slower and with this flag you can disable it. */
#if !defined(SZ_ENFORCE_SVE_OVER_NEON)
#define SZ_ENFORCE_SVE_OVER_NEON (0)
#endif

#if !defined(SZ_USE_CUDA)
#if defined(__NVCC__)
#define SZ_USE_CUDA (1)
#else
#define SZ_USE_CUDA (0)
#endif
#endif

/**
 *  Kepler-generation logic requires SM30+ (i.e. `__CUDA_ARCH__ >= 300`).
 *  For dynamic dispatch, however, it's more sensible to check the CUDA version (i.e. `__CUDACC_VER_MAJOR__ >= 11`).
 */
#if !defined(SZ_USE_KEPLER)
#if defined(__NVCC__) && defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 3)
#define SZ_USE_KEPLER (1)
#else
#define SZ_USE_KEPLER (0)
#endif
#endif

/**
 *  Hopper-generation logic requires SM90+ (i.e. `__CUDA_ARCH__ >= 900`).
 *  For dynamic dispatch, however, it's more sensible to check the CUDA version (i.e. `__CUDACC_VER_MAJOR__ >= 11`).
 */
#if !defined(SZ_USE_HOPPER)
#if defined(__NVCC__) && defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 11)
#define SZ_USE_HOPPER (1)
#else
#define SZ_USE_HOPPER (0)
#endif
#endif

/*  WebAssembly SIMD128 — opt-in at compile time via `-msimd128`; there is no runtime probe. */
#if !defined(SZ_USE_V128)
#if defined(__wasm__) && defined(__wasm_simd128__)
#define SZ_USE_V128 (1)
#else
#define SZ_USE_V128 (0)
#endif
#endif

/*  WebAssembly @b relaxed SIMD — opt-in via `-mrelaxed-simd`; a level above baseline SIMD128 adding
 *  relaxed swizzle, fused multiply-add, lane-select, and integer dot-products. Some runtimes lower a
 *  few relaxed ops sub-optimally, but the level is exposed so native engines can use them. */
#if !defined(SZ_USE_V128RELAXED)
#if defined(__wasm__) && defined(__wasm_relaxed_simd__)
#define SZ_USE_V128RELAXED (1)
#else
#define SZ_USE_V128RELAXED (0)
#endif
#endif

/*  RISC-V Vector extension (RVV 1.0) — `-march=rv64gcv`. Length-agnostic registers. */
#if !defined(SZ_USE_RVV)
#if defined(__riscv) && (__riscv_xlen == 64) && defined(__riscv_vector)
#define SZ_USE_RVV (1)
#else
#define SZ_USE_RVV (0)
#endif
#endif

/*  RISC-V Vector Crypto (Zvk: Zvkned AES + Zvknhb SHA) — `-march=rv64gcv_zvkned_zvknhb`. */
#if !defined(SZ_USE_RVVCRYPTO)
#if SZ_USE_RVV && defined(__riscv_zvkned) && defined(__riscv_zvknhb)
#define SZ_USE_RVVCRYPTO (1)
#else
#define SZ_USE_RVVCRYPTO (0)
#endif
#endif

/*  LoongArch Advanced SIMD eXtension (LASX, 256-bit) — `-mlasx`. */
#if !defined(SZ_USE_LASX)
#if defined(__loongarch__) && defined(__loongarch_asx)
#define SZ_USE_LASX (1)
#else
#define SZ_USE_LASX (0)
#endif
#endif

/*  IBM Power Vector-Scalar eXtension (VSX, Power8+) — `-mvsx`. */
#if !defined(SZ_USE_POWERVSX)
#if (defined(__powerpc__) || defined(__powerpc64__)) && defined(__VSX__)
#define SZ_USE_POWERVSX (1)
#else
#define SZ_USE_POWERVSX (0)
#endif
#endif

/*  SIMD micro-architecture tiers are cumulative supersets - there is no CPU with AVX-512 VBMI but
 *  not AVX2, nor SVE without NEON - and higher-tier kernels call into lower-tier helpers (the Ice
 *  Lake hash reuses the Westmere routines, the Ice Lake intersect kernel uses the 128/256-bit
 *  register unions). Enabling a tier therefore REQUIRES its whole substrate. Close the set
 *  downward so an isolated `-D SZ_USE_ICELAKE=1`, or a hand-edited development override, can never
 *  name a tier without the ones it is built on. This is the single source of truth for the
 *  nesting; everything downstream may assume a lower tier is on whenever a higher one is. Goldmont
 *  (SHA) and the NEON/SVE crypto extensions are orthogonal feature flags, not part of the nesting.
 */
#if SZ_USE_ICELAKE && !SZ_USE_SKYLAKE
#undef SZ_USE_SKYLAKE
#define SZ_USE_SKYLAKE (1)
#endif
#if SZ_USE_SKYLAKE && !SZ_USE_HASWELL
#undef SZ_USE_HASWELL
#define SZ_USE_HASWELL (1)
#endif
#if SZ_USE_HASWELL && !SZ_USE_WESTMERE
#undef SZ_USE_WESTMERE
#define SZ_USE_WESTMERE (1)
#endif
#if SZ_USE_SVE2 && !SZ_USE_SVE
#undef SZ_USE_SVE
#define SZ_USE_SVE (1)
#endif
#if SZ_USE_SVE && !SZ_USE_NEON
#undef SZ_USE_NEON
#define SZ_USE_NEON (1)
#endif
/*  WebAssembly relaxed-SIMD is a tier above baseline SIMD128 (it requires `simd128`). Close the set
 *  downward so backends that ship only a `_v128` kernel can dispatch on `#if SZ_USE_V128` alone and
 *  still be reached when the build enabled relaxed-SIMD. */
#if SZ_USE_V128RELAXED && !SZ_USE_V128
#undef SZ_USE_V128
#define SZ_USE_V128 (1)
#endif

/*  Hardware-specific headers for different SIMD intrinsics and register wrappers.
 */
#if SZ_USE_V128
#include <wasm_simd128.h>
#endif // SZ_USE_V128
#if SZ_USE_RVV
#include <riscv_vector.h>
#endif // SZ_USE_RVV
#if SZ_USE_LASX
#include <lasxintrin.h> // 256-bit `__lasx_*` intrinsics and the `__m256i` register type
#include <lsxintrin.h>  // 128-bit `__lsx_*` intrinsics and the `__m128i` register type, for sub-32-byte inputs
#endif                  // SZ_USE_LASX
#if SZ_USE_POWERVSX
#include <altivec.h>
#endif // SZ_USE_POWERVSX
#if SZ_USE_WESTMERE || SZ_USE_HASWELL || SZ_USE_SKYLAKE || SZ_USE_ICELAKE
#include <immintrin.h>
#endif // SZ_USE_WESTMERE || SZ_USE_HASWELL || SZ_USE_SKYLAKE || SZ_USE_ICELAKE
#if SZ_USE_NEON
#if !defined(_MSC_VER)
#include <arm_acle.h>
#endif
#include <arm_neon.h>
#endif // SZ_USE_NEON
#if SZ_USE_SVE || SZ_USE_SVE2
#if !defined(_MSC_VER)
#include <arm_sve.h>
#endif
#endif // SZ_USE_SVE || SZ_USE_SVE2

#ifdef __cplusplus
extern "C" {
#endif

typedef float sz_f32_t;  // 32-bit floating-point number
typedef double sz_f64_t; // 64-bit floating-point number

/*
 *  Let's infer the integer types or pull them from LibC,
 *  if that is allowed by the user.
 */
#if !SZ_AVOID_LIBC
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

#else // if SZ_AVOID_LIBC:
/**
 *  Even when LibC is not available, we can use compiler macros to infer the size of integer types.
 *  https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
 *
 *  ! The C standard doesn't specify the signedness of char.
 *  ! On x86 char is signed by default while on Arm it is unsigned by default.
 *  ! That's why we don't define `sz_char_t` and generally use explicit `sz_i8_t` and `sz_u8_t`.
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
 *  Now we need to redefine the `size_t`.
 *  Microsoft Visual C++ (MSVC) typically follows LLP64 data model on 64-bit platforms,
 *  where integers, pointers, and long types have different sizes:
 *
 *   > `int` is 32 bits
 *   > `long` is 32 bits
 *   > `long long` is 64 bits
 *   > pointer (thus, `size_t`) is 64 bits
 *
 *  In contrast, GCC and Clang on 64-bit Unix-like systems typically follow the LP64 model, where:
 *
 *   > `int` is 32 bits
 *   > `long` and pointer (thus, `size_t`) are 64 bits
 *   > `long long` is also 64 bits
 *
 *  Source: https://learn.microsoft.com/en-us/windows/win32/winprog64/abstract-data-models
 */
#if SZ_IS_64BIT_
typedef sz_u64_t sz_size_t;  // ? Preferred over the `__SIZE_TYPE__` and `__UINTMAX_TYPE__` macros
typedef sz_i64_t sz_ssize_t; // ? Preferred over the `__PTRDIFF_TYPE__` and `__INTMAX_TYPE__` macros
#else
typedef sz_u32_t sz_size_t;  // ? Preferred over the `__SIZE_TYPE__` and `__UINTMAX_TYPE__` macros
typedef sz_i32_t sz_ssize_t; // ? Preferred over the `__PTRDIFF_TYPE__` and `__INTMAX_TYPE__` macros
#endif // SZ_IS_64BIT_
#endif // SZ_AVOID_LIBC

/**
 *  @brief Compile-time assert akin to C++ `static_assert`. Uses the native assertion where available (C++11
 *  `static_assert`, C11 `_Static_assert`); the older-C typedef fallback must sit at file scope to stay clear of
 *  `-Wunused-local-typedef`.
 */
#if defined(__cplusplus) && __cplusplus >= 201103L
#define sz_static_assert(condition, name) static_assert(condition, #name)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define sz_static_assert(condition, name) _Static_assert(condition, #name)
#elif defined(_MSC_VER)
#define sz_static_assert(condition, name) static_assert(condition, #name)
#else
#define sz_static_assert(condition, name) typedef char sz_static_assert_##name[(condition) ? 1 : -1]
#endif

sz_static_assert(sizeof(sz_size_t) == sizeof(void *), sz_size_t_must_be_pointer_size);
sz_static_assert(sizeof(sz_ssize_t) == sizeof(void *), sz_ssize_t_must_be_pointer_size);

typedef unsigned char sz_byte_t;            // A byte is an 8-bit unsigned integer
typedef char *sz_ptr_t;                     // A type alias for `char *`
typedef char const *sz_cptr_t;              // A type alias for `char const *`
typedef sz_i8_t sz_error_cost_t;            // Character mismatch cost for fuzzy matching functions
typedef sz_u16_t sz_error_cost_magnitude_t; // The smallest type that can hold unsigned `abs(sz_error_cost_t {})`

struct sz_hash_state_t;            // Forward declaration of a hash state structure
struct sz_sha256_state_t;          // Forward declaration of a SHA256 hash state structure
struct sz_sequence_t;              // Forward declaration of an ordered collection of strings
typedef sz_size_t sz_sorted_idx_t; // Index of a sorted string in a list of strings
typedef sz_size_t sz_pgram_t;      // "Pointer-sized N-gram" of a string

/**
 *  @brief Simple boolean type, until `_Bool` in C 99 and `true` and `false` in C 23.
 *  @see https://stackoverflow.com/questions/1921539/using-boolean-values-in-c
 */
typedef enum { sz_false_k = 0, sz_true_k = 1 } sz_bool_t;

/**
 *  @brief Describes the result of a comparison operation. Equivalent to @b `std::strong_ordering` in C++20.
 *  @see https://en.cppreference.com/w/cpp/utility/compare/strong_ordering
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
 *  @brief Describes how a similarity kernel packs work across SIMD lanes (or a GPU warp's threads).
 *
 *  @b one_pair_per_walk_k vectorizes within a single (query, candidate) pair (anti-diagonal lanes) — the
 *  intra-sequence form, used for one-off pairs and the ragged tail of a cross-product.
 *  @b candidates_across_lanes_k vectorizes across many candidates of one shared query (one candidate per lane)
 *  — the inter-sequence form that keeps every lane busy regardless of string length, the cross-product workhorse.
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

/**
 *  @brief A simple signed integer type describing the status of a faulty operation.
 *  @sa sz_success_k, sz_bad_alloc_k, sz_invalid_utf8_k, sz_contains_duplicates_k
 */
typedef enum sz_status_t {
    /** For algorithms that return a status, this status indicates that the operation was successful. */
    sz_success_k = 0,
    /** For algorithms that require memory allocation, this status indicates that the allocation failed. */
    sz_bad_alloc_k = -10,
    /** For algorithms that require UTF8 input, this status indicates that the input is invalid. */
    sz_invalid_utf8_k = -12,
    /** For algorithms that take collections of unique elements, this status indicates presence of duplicates. */
    sz_contains_duplicates_k = -13,
    /** For algorithms dealing with large inputs, this error reports the need to upcast the logic to larger types. */
    sz_overflow_risk_k = -14,
    /** For algorithms with multi-stage pipelines indicates input/output size mismatch. */
    sz_unexpected_dimensions_k = -15,
    /** GPU support is missing in the library. */
    sz_missing_gpu_k = -16,
    /** Backend-device mismatch: e.g., GPU kernel with CPU/default executor or vice versa. */
    sz_device_code_mismatch_k = -17,
    /** Device memory mismatch: e.g., GPU kernel requires unified/device-accessible memory. */
    sz_device_memory_mismatch_k = -18,
    /** A sink-hole status for unknown errors. */
    sz_status_unknown_k = -1,
} sz_status_t;

/**
 *  @brief Enumeration of SIMD capabilities of the target architecture.
 *         Used to introspect the supported functionality of the dynamic library.
 */
typedef enum sz_capability_t {
    sz_cap_serial_k = 1,        ///< Serial (non-SIMD) capability
    sz_cap_parallel_k = 1 << 2, ///< Multi-threading via Fork Union or other OpenMP-like engines
    sz_cap_any_k = 0x7FFFFFFF,  ///< Mask representing any capability with `INT_MAX`

    sz_cap_goldmont_k = 1 << 3, ///< x86 SHA-NI capability for accelerated SHA-256 hashing
    sz_cap_westmere_k = 1 << 4, ///< x86 SSE4.2 + AES-NI capability
    sz_cap_haswell_k = 1 << 5,  ///< x86 AVX2 capability with FMA and F16C extensions
    sz_cap_skylake_k = 1 << 6,  ///< x86 AVX512 baseline capability
    sz_cap_icelake_k = 1 << 7,  ///< x86 AVX512 capability with advanced integer algos and AES extensions

    sz_cap_neon_k = 1 << 10,    ///< ARM NEON baseline capability
    sz_cap_neonaes_k = 1 << 11, ///< ARM NEON baseline capability with AES extensions
    sz_cap_neonsha_k = 1 << 15, ///< ARM NEON baseline capability with SHA2 extensions
    sz_cap_sve_k = 1 << 12,     ///< ARM SVE baseline capability
    sz_cap_sve2_k = 1 << 13,    ///< ARM SVE2 capability
    sz_cap_sve2aes_k = 1 << 14, ///< ARM SVE2 capability with AES extensions

    sz_cap_v128_k = 1 << 16,        ///< WebAssembly SIMD128 capability
    sz_cap_v128relaxed_k = 1 << 17, ///< WebAssembly relaxed-SIMD capability (above SIMD128)
    sz_cap_lasx_k = 1 << 18,        ///< LoongArch LASX (256-bit) capability
    sz_cap_powervsx_k = 1 << 19,    ///< IBM Power VSX capability
    sz_cap_rvv_k = 1 << 20,         ///< RISC-V Vector (RVV 1.0) capability

    sz_cap_cuda_k = 1 << 21,   ///< CUDA capability
    sz_cap_kepler_k = 1 << 22, ///< CUDA capability with support with in-warp register shuffles
    sz_cap_hopper_k = 1 << 23, ///< CUDA capability with support for Hopper's DPX instructions

    sz_cap_rvvcrypto_k = 1 << 24, ///< RISC-V vector crypto (Zvk: Zvkned AES + Zvknhb SHA) capability

    sz_caps_none_k = 0,

    sz_caps_sp_k = sz_cap_serial_k | sz_cap_parallel_k, ///< Serial code with Fork Union
    sz_caps_sh_k = sz_cap_serial_k | sz_cap_haswell_k,  ///< Serial code with Haswell
    sz_caps_sn_k = sz_cap_serial_k | sz_cap_neon_k,     ///< Serial code with NEON
    sz_caps_sr_k = sz_cap_serial_k | sz_cap_rvv_k,      ///< Serial code with RISC-V Vector
    sz_caps_sil_k = sz_cap_serial_k | sz_cap_icelake_k, ///< Serial code with Ice Lake

    sz_caps_spil_k = sz_cap_serial_k | sz_cap_parallel_k |
                     sz_cap_icelake_k,                                  ///< Serial code with Fork Union and Ice Lake
    sz_caps_sps_k = sz_cap_serial_k | sz_cap_parallel_k | sz_cap_sve_k, ///< Serial code with Fork Union and SVE
    sz_caps_ck_k = sz_cap_cuda_k | sz_cap_kepler_k,                     ///< CUDA code with Kepler
    sz_caps_ckh_k = sz_cap_cuda_k | sz_cap_kepler_k | sz_cap_hopper_k,  ///< CUDA code with Kepler and Hopper

    // Aggregates for different StringZillas builds
    sz_caps_cpus_k = sz_cap_serial_k | sz_cap_parallel_k | sz_cap_haswell_k | sz_cap_skylake_k | sz_cap_icelake_k |
                     sz_cap_westmere_k | sz_cap_goldmont_k | sz_cap_neon_k | sz_cap_neonaes_k | sz_cap_neonsha_k |
                     sz_cap_sve_k | sz_cap_sve2_k | sz_cap_sve2aes_k | sz_cap_v128_k | sz_cap_v128relaxed_k |
                     sz_cap_rvv_k | sz_cap_rvvcrypto_k | sz_cap_lasx_k | sz_cap_powervsx_k,
    sz_caps_cuda_k = sz_cap_cuda_k | sz_cap_kepler_k | sz_cap_hopper_k,

} sz_capability_t;

/**
 *  @brief Maximum number of individual capability flags that can be represented.
 *  @sa sz_capabilities_to_strings_implementation_ - not intended for public use, but a valid example.
 */
#define SZ_CAPABILITIES_COUNT 22

/**
 *  @brief Describes the length of a UTF-8 @b rune / character / codepoint in bytes, which can be 1 to 4.
 *  @see https://en.wikipedia.org/wiki/UTF-8
 */
typedef enum sz_rune_length_t {
    sz_rune_invalid_k = 0, //!< Invalid UTF8 character.
    sz_rune_1byte_k = 1,   //!< 1-byte UTF8 character.
    sz_rune_2bytes_k = 2,  //!< 2-byte UTF8 character.
    sz_rune_3bytes_k = 3,  //!< 3-byte UTF8 character.
    sz_rune_4bytes_k = 4,  //!< 4-byte UTF8 character.
} sz_rune_length_t;

/**
 *  @brief Stores a single UTF-8 @b rune / character / codepoint unpacked into @b UTF-32.
 *  @see https://en.wikipedia.org/wiki/UTF-32
 *
 *  The theoretical capacity of the underlying numeric type is 4 bytes, with over 4 billion possible states, but:
 *  - UTF-8, however, in its' largest 4-byte form has only 3+6+6+6 = 21 bits of usable space for 2 million states.
 *  - Unicode, in turn, has only @b 1'114'112 possible code points from U+0000 to U+10FFFF.
 *  - Of those, in Unicode 16, only @b 155'063 are assigned characters ~ a little over 17 bits of content.
 *  That's @b 0.004% of the 32-bit space, so sparse data-structures are encouraged for UTF-8 oriented algorithms.
 */
typedef sz_u32_t sz_rune_t;

/** @brief The Unicode @b replacement character U+FFFD, emitted once per maximal ill-formed UTF-8 subpart. */
enum { sz_rune_replacement_k = 0xFFFD };

SZ_API_COMPTIME sz_rune_t sz_rune_perfect_hash(sz_rune_t rune) {
    // TODO: A perfect hashing scheme can be constructed to map a 32-bit rune into an 18-bit representation,
    // TODO: that can fit all of the unique values in the Unicode 16 standard.
    return rune;
}

/**
 *  @brief Tiny string-view structure. It's Plain-Old Datatype @b (POD) type, unlike the `std::string_view`.
 *  @see https://en.cppreference.com/w/cpp/named_req/PODType
 */
typedef struct sz_string_view_t {
    sz_cptr_t start;
    sz_size_t length;
} sz_string_view_t;

#pragma region Character Sets

/**
 *  @brief Bit-set semi-opaque structure for 256 possible byte values. Useful for filtering and search.
 *  @sa sz_byteset_init, sz_byteset_add, sz_byteset_contains, sz_byteset_invert
 *
 *  Example usage:
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
 */
typedef union sz_byteset_t {
    sz_u64_t _u64s[4];
    sz_u32_t _u32s[8];
    sz_u16_t _u16s[16];
    sz_u8_t _u8s[32];
} sz_byteset_t;

/** @brief Initializes a bit-set to an empty collection, meaning - all characters are banned. */
SZ_API_COMPTIME void sz_byteset_init(sz_byteset_t *s) { s->_u64s[0] = s->_u64s[1] = s->_u64s[2] = s->_u64s[3] = 0; }

/** @brief Initializes a bit-set to all ASCII character. */
SZ_API_COMPTIME void sz_byteset_init_ascii(sz_byteset_t *s) {
    s->_u64s[0] = s->_u64s[1] = 0xFFFFFFFFFFFFFFFFull;
    s->_u64s[2] = s->_u64s[3] = 0;
}

/** @brief Adds a character to the set and accepts @b unsigned integers. */
SZ_API_COMPTIME void sz_byteset_add_u8(sz_byteset_t *s, sz_u8_t c) { s->_u64s[c >> 6] |= (1ull << (c & 63u)); }

/** @brief Adds a character to the set. Consider @b sz_byteset_add_u8. */
SZ_API_COMPTIME void sz_byteset_add(sz_byteset_t *s, char c) { sz_byteset_add_u8(s, *(sz_u8_t *)(&c)); } // bitcast

/** @brief Checks if the set contains a given character and accepts @b unsigned integers. */
SZ_API_COMPTIME sz_bool_t sz_byteset_contains_u8(sz_byteset_t const *s, sz_u8_t c) {
    // Checking the bit can be done in different ways:
    // - (s->_u64s[c >> 6] & (1ull << (c & 63u))) != 0
    // - (s->_u32s[c >> 5] & (1u << (c & 31u))) != 0
    // - (s->_u16s[c >> 4] & (1u << (c & 15u))) != 0
    // - (s->_u8s[c >> 3] & (1u << (c & 7u))) != 0
    return (sz_bool_t)((s->_u64s[c >> 6] & (1ull << (c & 63u))) != 0);
}

/** @brief Checks if the set contains a given character. Consider @b sz_byteset_contains_u8. */
SZ_API_COMPTIME sz_bool_t sz_byteset_contains(sz_byteset_t const *s, char c) {
    return sz_byteset_contains_u8(s, *(sz_u8_t *)(&c)); // bitcast
}

/** @brief Inverts the contents of the set, so allowed character get disallowed, and vice versa. */
SZ_API_COMPTIME void sz_byteset_invert(sz_byteset_t *s) {
    s->_u64s[0] ^= 0xFFFFFFFFFFFFFFFFull, s->_u64s[1] ^= 0xFFFFFFFFFFFFFFFFull, //
        s->_u64s[2] ^= 0xFFFFFFFFFFFFFFFFull, s->_u64s[3] ^= 0xFFFFFFFFFFFFFFFFull;
}

#pragma endregion

#pragma region Memory Management

typedef void *(*sz_memory_allocate_t)(sz_size_t, void *);
typedef void (*sz_memory_free_t)(void *, sz_size_t, void *);

/**
 *  @brief Some complex pattern matching algorithms may require memory allocations.
 *         This structure is used to pass the memory allocator to those functions.
 *  @sa sz_memory_allocator_init_fixed
 */
typedef struct sz_memory_allocator_t {
    sz_memory_allocate_t allocate;
    sz_memory_free_t free;
    void *handle;
} sz_memory_allocator_t;

/**
 *  @brief Initializes a memory allocator to use the system default `malloc` and `free`.
 *  @warning The function is not available if the library was compiled with `SZ_AVOID_LIBC`.
 *  @param allocator Memory allocator to initialize.
 *
 *  @note Unlike the C standard library, the `malloc(0)` is guaranteed to return a non-null pointer.
 *  @see https://en.cppreference.com/w/c/memory/malloc
 */
SZ_API_COMPTIME void sz_memory_allocator_init_default(sz_memory_allocator_t *allocator);

/**
 *  @brief Initializes a memory allocator to use only a static-capacity buffer @b w/out any dynamic allocations.
 *  @param allocator Memory allocator to initialize.
 *  @param buffer Buffer to use for allocations.
 *  @param length Length of the buffer. @b Must be greater than 16, at least 4KB (one RAM page) is recommended.
 *
 *  The `buffer` itself will be prepended with the capacity and the consumed size. Those values shouldn't be
 *  modified.
 */
SZ_API_COMPTIME void sz_memory_allocator_init_fixed(sz_memory_allocator_t *allocator, void *buffer, sz_size_t length);

/**
 *  @brief Checks if two memory allocators are equivalent.
 *  @param a First memory allocator.
 *  @param b Second memory allocator.
 *  @return True if the allocators are the same, false otherwise.
 */
SZ_API_COMPTIME sz_bool_t sz_memory_allocator_equal(sz_memory_allocator_t const *a, sz_memory_allocator_t const *b);

#pragma endregion

#pragma region API Signature Types

/** @brief Signature of `sz_hash`. */
typedef sz_u64_t (*sz_hash_t)(sz_cptr_t, sz_size_t, sz_u64_t);

/** @brief Signature of `sz_hash_multiseed`. */
typedef void (*sz_hash_multiseed_t)(sz_cptr_t, sz_size_t, sz_u64_t const *, sz_size_t, sz_u64_t *);

/** @brief Signature of `sz_hash_state_init`. */
typedef void (*sz_hash_state_init_t)(struct sz_hash_state_t *, sz_u64_t);

/** @brief Signature of `sz_hash_state_update` (legacy) / `sz_hash_state_update` (preferred). */
typedef void (*sz_hash_state_update_t)(struct sz_hash_state_t *, sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_hash_state_digest` (legacy) / `sz_hash_state_digest` (preferred). */
typedef sz_u64_t (*sz_hash_state_digest_t)(struct sz_hash_state_t const *);

/** @brief Signature of `sz_bytesum`. */
typedef sz_u64_t (*sz_bytesum_t)(sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_utf8_count`. */
typedef sz_size_t (*sz_utf8_count_t)(sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_utf8_seek`. */
typedef sz_cptr_t (*sz_utf8_seek_t)(sz_cptr_t, sz_size_t, sz_size_t);

/** @brief Signature of `sz_utf8_decode`. */
typedef sz_cptr_t (*sz_utf8_decode_t)(sz_cptr_t, sz_size_t, sz_rune_t *, sz_size_t, sz_size_t *);

/** @brief Signature of `sz_utf8_uncased_fold`. */
typedef sz_size_t (*sz_utf8_uncased_fold_t)(sz_cptr_t, sz_size_t, sz_ptr_t);

/** @brief Unicode normalization form selector (see `utf8_norm.h`). */
typedef enum sz_normal_form_t {
    sz_normal_form_nfd_k = 0,  /**< Canonical decomposition. */
    sz_normal_form_nfc_k = 1,  /**< Canonical decomposition + canonical composition. */
    sz_normal_form_nfkd_k = 2, /**< Compatibility decomposition. */
    sz_normal_form_nfkc_k = 3, /**< Compatibility decomposition + canonical composition. */
} sz_normal_form_t;

/** @brief Signature of `sz_utf8_norm` (single-pass normalizer). */
typedef sz_size_t (*sz_utf8_norm_t)(sz_cptr_t, sz_size_t, sz_normal_form_t, sz_ptr_t);

/** @brief Signature of `sz_utf8_find_denormalized`. */
typedef sz_cptr_t (*sz_utf8_find_denormalized_t)(sz_cptr_t, sz_size_t, sz_normal_form_t);

/** @brief Forward declaration for uncased needle metadata. */
struct sz_utf8_uncased_needle_metadata_t;

/** @brief Signature of `sz_utf8_uncased_search`. */
typedef sz_cptr_t (*sz_utf8_uncased_search_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t,
                                              struct sz_utf8_uncased_needle_metadata_t *, sz_size_t *);

/** @brief Signature of `sz_utf8_uncased_order`. */
typedef sz_ordering_t (*sz_utf8_uncased_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_utf8_find_cased`. */
typedef sz_cptr_t (*sz_utf8_find_cased_t)(sz_cptr_t, sz_size_t);

/** @brief Signature of every UTF-8 "find boundaries" kernel - words (forward/reverse), graphemes, sentences,
 *         lines, newlines, whitespace, delimiters. Emits parallel (offset, length) arrays for each
 *         segment/delimiter plus a resume `bytes_consumed`. */
typedef sz_size_t (*sz_utf8_segmenter_t)(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t *, sz_size_t, sz_size_t *);

/** @brief Signature of `sz_fill_random`. */
typedef void (*sz_fill_random_t)(sz_ptr_t, sz_size_t, sz_u64_t);

/** @brief Signature of `sz_sha256_state_init`. */
typedef void (*sz_sha256_state_init_t)(struct sz_sha256_state_t *);

/** @brief Signature of `sz_sha256_state_update`. */
typedef void (*sz_sha256_state_update_t)(struct sz_sha256_state_t *, sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_sha256_state_digest`. */
typedef void (*sz_sha256_state_digest_t)(struct sz_sha256_state_t const *, sz_u8_t *);

/** @brief Signature of `sz_equal`. */
typedef sz_bool_t (*sz_equal_t)(sz_cptr_t, sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_order`. */
typedef sz_ordering_t (*sz_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_lookup`. */
typedef void (*sz_lookup_t)(sz_ptr_t, sz_size_t, sz_cptr_t, sz_cptr_t);

/** @brief Signature of `sz_copy`. */
typedef void (*sz_copy_t)(sz_ptr_t, sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_move`. */
typedef void (*sz_move_t)(sz_ptr_t, sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_fill`. */
typedef void (*sz_fill_t)(sz_ptr_t, sz_size_t, sz_u8_t);

/** @brief Signature of `sz_find_byte`. */
typedef sz_cptr_t (*sz_find_byte_t)(sz_cptr_t, sz_size_t, sz_cptr_t);

/** @brief Signature of `sz_find`. */
typedef sz_cptr_t (*sz_find_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/** @brief Signature of `sz_find_byteset`. */
typedef sz_cptr_t (*sz_find_byteset_t)(sz_cptr_t, sz_size_t, sz_byteset_t const *);

/** @brief Signature of `sz_sequence_argsort` and `sz_sequence_argsort_uncased`. */
typedef sz_status_t (*sz_sequence_argsort_t)(struct sz_sequence_t const *, sz_memory_allocator_t *, sz_sorted_idx_t *,
                                             sz_size_t, sz_bool_t);

/** @brief Signature of the internal `sz_pgrams_sort_serial`/`_skylake`/`_sve` integer-sort helpers. */
typedef sz_status_t (*sz_pgrams_sort_t)(sz_pgram_t *, sz_size_t, sz_memory_allocator_t *, sz_sorted_idx_t *);

/** @brief Signature of `sz_sequence_intersect`. */
typedef sz_status_t (*sz_sequence_intersect_t)(struct sz_sequence_t const *, struct sz_sequence_t const *,
                                               sz_memory_allocator_t *, sz_u64_t, sz_size_t *, sz_sorted_idx_t *,
                                               sz_sorted_idx_t *);

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

/**
 *  @brief Helper structure to simplify work with @b 128-bit registers.
 *         It can help view the contents as 8-bit, 16-bit, 32-bit, or 64-bit integers,
 *         as well as 1x XMM register.
 */
typedef union SZ_MAY_ALIAS sz_u128_vec_t {
#if SZ_USE_WESTMERE
    __m128i xmm;
    __m128d xmm_pd;
    __m128 xmm_ps;
#endif
#if SZ_USE_NEON
    uint8x16_t u8x16;
    uint16x8_t u16x8;
    uint32x4_t u32x4;
    uint64x2_t u64x2;
    float64x2_t f64x2;
    float32x4_t f32x4;
#endif
#if SZ_USE_LASX
    __m128i lsx;
#endif
#if SZ_USE_V128
    v128_t v128;
#endif
#if SZ_USE_POWERVSX
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

/**
 *  @brief Helper structure to simplify work with @b 256-bit registers.
 *         It can help view the contents as 8-bit, 16-bit, 32-bit, or 64-bit integers,
 *         as well as 2x XMM registers or 1x YMM register.
 */
typedef union SZ_MAY_ALIAS sz_u256_vec_t {
#if SZ_USE_HASWELL
    __m256i ymm;
    __m256d ymm_pd;
    __m256 ymm_ps;
#endif
#if SZ_USE_WESTMERE
    __m128i xmms[2];
#endif
#if SZ_USE_NEON
    uint8x16_t u8x16s[2];
    uint16x8_t u16x8s[2];
    uint32x4_t u32x4s[2];
    uint64x2_t u64x2s[2];
#endif
#if SZ_USE_LASX
    __m256i lasx;
#endif
#if SZ_USE_V128
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

/**
 *  @brief Helper structure to simplify work with @b 512-bit registers.
 *         It can help view the contents as 8-bit, 16-bit, 32-bit, or 64-bit integers,
 *         as well as 4x XMM registers or 2x YMM registers or 1x ZMM register.
 */
typedef union SZ_MAY_ALIAS sz_u512_vec_t {
#if SZ_USE_SKYLAKE
    __m512i zmm;
    __m512d zmm_pd;
    __m512 zmm_ps;
#endif
#if SZ_USE_HASWELL
    __m256i ymms[2];
#endif
#if SZ_USE_WESTMERE
    __m128i xmms[4];
#endif
#if SZ_USE_NEON
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

#pragma region String Sequences API

/** @brief Signature of `sz_sequence_t::get_start` used to get the start of a member string at a given index. */
typedef sz_cptr_t (*sz_sequence_member_start_t)(void const *, sz_sorted_idx_t);
/** @brief Signature of `sz_sequence_t::get_length` used to get the length of a member string at a given index. */
typedef sz_size_t (*sz_sequence_member_length_t)(void const *, sz_sorted_idx_t);

/**
 *  @brief Structure to represent an ordered collection of strings.
 *         It's a generic structure that can be used to represent a sequence of strings in different layouts.
 *         It can be easily combined with Apache Arrow and its tape-like concatenated strings.
 *  @sa sz_sequence_from_null_terminated_strings
 */
typedef struct sz_sequence_t {
    void const *handle;
    sz_size_t count;
    sz_sequence_member_start_t get_start;
    sz_sequence_member_length_t get_length;
} sz_sequence_t;

/**
 *  @brief Initiates the sequence structure from a typical C-style strings array, like `char *[]`.
 *  @param start Pointer to the array of strings.
 *  @param count Number of strings in the array.
 *  @param sequence Sequence structure to initialize.
 */
SZ_API_COMPTIME void sz_sequence_from_null_terminated_strings(sz_cptr_t *start, sz_size_t count,
                                                              sz_sequence_t *sequence);

#pragma endregion

#pragma region Helper Functions

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC visibility push(hidden)
#endif

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

/** @brief Helper-macro to mark potentially unused variables. */
#define sz_unused_(x) ((void)(x))

/** @brief Helper-macro casting a variable to another type of the same size. */
#if !defined(_MSC_VER) && defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
#define sz_bitcast_(type, value) __builtin_bit_cast(type, (value))
#else
#define sz_bitcast_(type, value) (*((type *)&(value)))
#endif
#else
#define sz_bitcast_(type, value) (*((type *)&(value)))
#endif

/**
 *  @brief Defines `SZ_NULL`, analogous to `NULL`.
 *         The default often comes from locale.h, stddef.h,
 *         stdio.h, stdlib.h, string.h, time.h, or wchar.h.
 */
#ifdef __GNUG__
#define SZ_NULL __null
#define SZ_NULL_CHAR __null
#elif defined(__cplusplus)
#define SZ_NULL nullptr
#define SZ_NULL_CHAR nullptr
#else
#define SZ_NULL ((void *)0)
#define SZ_NULL_CHAR ((char *)0)
#endif

/**
 *  @brief Cache-line width, that will affect the execution of some algorithms,
 *         like equality checks and relative order computing.
 */
#define SZ_CACHE_LINE_WIDTH (64)   // bytes
#define SZ_MAX_REGISTER_WIDTH (64) // bytes
#define SZ_SIZE_MAX ((sz_size_t)(-1))
#define SZ_SSIZE_MAX ((sz_ssize_t)(SZ_SIZE_MAX >> 1))
#define SZ_SSIZE_MIN ((sz_ssize_t)(-SZ_SSIZE_MAX - 1))

SZ_HELPER_AUTO sz_size_t sz_size_max_(void) { return SZ_SIZE_MAX; }
SZ_HELPER_AUTO sz_ssize_t sz_ssize_max_(void) { return SZ_SSIZE_MAX; }

/**
 *  @brief Similar to `assert`, the `sz_assert_` is used in the `SZ_DEBUG` mode
 *         to check the invariants of the library. It's a no-op in the "Release" mode.
 *  @note If you want to catch it, put a breakpoint at @b `__GI_exit`
 */
#if SZ_DEBUG && defined(SZ_AVOID_LIBC) && !SZ_AVOID_LIBC && !defined(SZ_PIC) && \
    !defined(__CUDA_ARCH__) // ? CPU code w/out LibC access
SZ_API_COMPTIME void sz_assert_failure_(char const *condition, char const *file, int line) {
    fprintf(stderr, "Assertion failed: %s, in file %s, line %d\n", condition, file, line);
    exit(EXIT_FAILURE);
}
#define sz_assert_(condition)                                                     \
    do {                                                                          \
        if (!(condition)) { sz_assert_failure_(#condition, __FILE__, __LINE__); } \
    } while (0)
#elif SZ_DEBUG && defined(__CUDA_ARCH__) // ? CUDA code for GPUs
SZ_DEVICE_NOINLINE void sz_assert_cuda_failure_(char const *condition, char const *file, int line) {
    printf("Assertion failed: %s, in file %s, line %d\n", condition, file, line);
    __trap();
}
#define sz_assert_(condition)                                                          \
    do {                                                                               \
        if (!(condition)) { sz_assert_cuda_failure_(#condition, __FILE__, __LINE__); } \
    } while (0)
#else
#define sz_assert_(condition) ((void)(condition))
#endif

/*  Intrinsics aliases for MSVC, GCC, Clang, and Clang-Cl.
 *  The following section of compiler intrinsics comes in 2 flavors.
 */
#if defined(_MSC_VER) && !defined(__clang__) // On Clang-CL
#include <intrin.h>
/*
 *  Sadly, when building Win32 images, we can't use the `_tzcnt_u64`, `_lzcnt_u64`,
 *  `_BitScanForward64`, or `_BitScanReverse64` intrinsics. For now it's a simple `for`-loop.
 *  TODO: In the future we can switch to a more efficient De Bruijn's algorithm.
 *  https://www.chessprogramming.org/BitScan
 *  https://www.chessprogramming.org/De_Bruijn_Sequence
 *  https://gist.github.com/resilar/e722d4600dbec9752771ab4c9d47044f
 *
 *  Use the serial version on 32-bit x86 and on Arm.
 */
#if (defined(_WIN32) && !defined(_WIN64)) || defined(_M_ARM) || defined(_M_ARM64)
SZ_HELPER_AUTO int sz_u64_ctz(sz_u64_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
SZ_HELPER_AUTO int sz_u64_clz(sz_u64_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 0x8000000000000000ull) == 0) { n++, x <<= 1; }
    return n;
}
SZ_HELPER_AUTO int sz_u64_popcount(sz_u64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    return (((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full) * 0x0101010101010101ull) >> 56;
}
SZ_HELPER_AUTO int sz_u32_ctz(sz_u32_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
SZ_HELPER_AUTO int sz_u32_clz(sz_u32_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 0x80000000u) == 0) { n++, x <<= 1; }
    return n;
}
SZ_HELPER_AUTO int sz_u32_popcount(sz_u32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
#else
SZ_HELPER_AUTO int sz_u64_ctz(sz_u64_t x) { return (int)_tzcnt_u64(x); }
SZ_HELPER_AUTO int sz_u64_clz(sz_u64_t x) { return (int)_lzcnt_u64(x); }
SZ_HELPER_AUTO int sz_u64_popcount(sz_u64_t x) { return (int)__popcnt64(x); }
SZ_HELPER_AUTO int sz_u32_ctz(sz_u32_t x) { return (int)_tzcnt_u32(x); }
SZ_HELPER_AUTO int sz_u32_clz(sz_u32_t x) { return (int)_lzcnt_u32(x); }
SZ_HELPER_AUTO int sz_u32_popcount(sz_u32_t x) { return (int)__popcnt(x); }
#endif
/*
 *  Force the byteswap functions to be intrinsics, because when `/Oi-` is given,
 *  these will turn into CRT function calls, which breaks when `SZ_AVOID_LIBC` is given.
 */
#pragma intrinsic(_byteswap_uint64)
SZ_HELPER_AUTO sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return _byteswap_uint64(val); }
#pragma intrinsic(_byteswap_ulong)
SZ_HELPER_AUTO sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return _byteswap_ulong(val); }
#else
SZ_HELPER_AUTO int sz_u64_popcount(sz_u64_t x) { return __builtin_popcountll(x); }
SZ_HELPER_AUTO int sz_u32_popcount(sz_u32_t x) { return __builtin_popcount(x); }
SZ_HELPER_AUTO int sz_u64_ctz(sz_u64_t x) { return __builtin_ctzll(x); }
SZ_HELPER_AUTO int sz_u64_clz(sz_u64_t x) { return __builtin_clzll(x); }
SZ_HELPER_AUTO int sz_u32_ctz(sz_u32_t x) { return __builtin_ctz(x); } // ! Undefined if `x == 0`
SZ_HELPER_AUTO int sz_u32_clz(sz_u32_t x) { return __builtin_clz(x); } // ! Undefined if `x == 0`
SZ_HELPER_AUTO sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return __builtin_bswap64(val); }
SZ_HELPER_AUTO sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return __builtin_bswap32(val); }
#endif

/** @brief Reverse the 64 bits of @p value (bit `i` moves to bit `63 - i`): swap adjacent bits, then bit-pairs
 *         within nibbles, then nibbles within bytes, then the bytes. Lets an ascending-only byte-compress
 *         (`vpcompressb`) pack lanes in descending order. */
SZ_HELPER_AUTO sz_u64_t sz_u64_bits_reverse(sz_u64_t value) {
    value = ((value & 0x5555555555555555ull) << 1) | ((value >> 1) & 0x5555555555555555ull);
    value = ((value & 0x3333333333333333ull) << 2) | ((value >> 2) & 0x3333333333333333ull);
    value = ((value & 0x0F0F0F0F0F0F0F0Full) << 4) | ((value >> 4) & 0x0F0F0F0F0F0F0F0Full);
    return sz_u64_bytes_reverse(value);
}

/** @brief Bit index of the n-th (0-based) set bit of @p bits; clears the @p n lowest set bits, then `ctz`.
 *         @p bits must hold more than @p n set bits. One tested home for the per-ISA SIMD "n-th lane" locate. */
SZ_HELPER_AUTO int sz_u64_nth_set_bit(sz_u64_t bits, sz_size_t n) {
    while (n--) bits &= bits - 1;
    return sz_u64_ctz(bits);
}
SZ_HELPER_AUTO int sz_u32_nth_set_bit(sz_u32_t bits, sz_size_t n) {
    while (n--) bits &= bits - 1;
    return sz_u32_ctz(bits);
}

/** @brief Branchless `value | (bit if condition)`: OR @p bit into @p value when @p condition holds, with no
 *         branch (mask-select rather than a CMOV, so there is no flag dependency). For threading a per-window
 *         carry signal into a lane mask. */
SZ_HELPER_AUTO sz_u64_t sz_u64_or_if_(sz_u64_t value, sz_u64_t bit, int condition) {
    return value | (bit & ((sz_u64_t)0 - (sz_u64_t)(condition != 0)));
}

SZ_HELPER_AUTO sz_u64_t sz_u64_rotl(sz_u64_t x, sz_u64_t r) { return (x << r) | (x >> (64 - r)); }

/**
 *  @brief Select bits from either @p a or @p b depending on the value of @p mask bits.
 *
 *  Similar to `_mm_blend_epi16` intrinsic on x86.
 *  Described in the "Bit Twiddling Hacks" by Sean Eron Anderson.
 *  https://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
 */
SZ_HELPER_AUTO sz_u64_t sz_u64_blend(sz_u64_t a, sz_u64_t b, sz_u64_t mask) { return a ^ ((a ^ b) & mask); }

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
#define sz_order_scalars_(a, b) ((sz_ordering_t)((a > b) - (a < b)))

/**
 *  Convenience macro to swap two values of the same type.
 */
#define sz_swap_(type, a, b) \
    do {                     \
        type _tmp = (a);     \
        (a) = (b);           \
        (b) = _tmp;          \
    } while (0)

/** @brief  Branchless minimum function for two signed 32-bit integers. */
SZ_HELPER_AUTO sz_i32_t sz_i32_min_of_two(sz_i32_t x, sz_i32_t y) { return y + ((x - y) & (x - y) >> 31); }

/** @brief  Branchless minimum function for two signed 32-bit integers. */
SZ_HELPER_AUTO sz_i32_t sz_i32_max_of_two(sz_i32_t x, sz_i32_t y) { return x - ((x - y) & (x - y) >> 31); }

/*  In AVX-512 we actively use masked operations and the "K mask registers".
 *  Producing a mask for the first N elements of a sequence can be done using the `1 << N - 1` idiom.
 *  It, however, induces undefined behavior if `N == 64` or `N == 32` on 64-bit or 32-bit systems respectively.
 *  Alternatively, the BZHI instruction can be used to clear the bits above N.
 */
#if SZ_USE_SKYLAKE || SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("bmi", "bmi2")
#endif
SZ_HELPER_AUTO __mmask8 sz_u8_mask_until_(sz_size_t n) { return (__mmask8)_bzhi_u32(0xFFu, (unsigned char)n); }
SZ_HELPER_AUTO __mmask16 sz_u16_mask_until_(sz_size_t n) { return (__mmask16)_bzhi_u32(0xFFFFu, (unsigned char)n); }
SZ_HELPER_AUTO __mmask32 sz_u32_mask_until_(sz_size_t n) { return (__mmask32)_bzhi_u64(0xFFFFFFFFu, (unsigned char)n); }
SZ_HELPER_AUTO __mmask64 sz_u64_mask_until_(sz_size_t n) {
    return (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned char)n);
}
SZ_HELPER_AUTO __mmask8 sz_u8_clamp_mask_until_(sz_size_t n) { return n < 8 ? sz_u8_mask_until_(n) : 0xFFu; }
SZ_HELPER_AUTO __mmask16 sz_u16_clamp_mask_until_(sz_size_t n) { return n < 16 ? sz_u16_mask_until_(n) : 0xFFFFu; }
SZ_HELPER_AUTO __mmask32 sz_u32_clamp_mask_until_(sz_size_t n) { return n < 32 ? sz_u32_mask_until_(n) : 0xFFFFFFFFu; }
SZ_HELPER_AUTO __mmask64 sz_u64_clamp_mask_until_(sz_size_t n) {
    return n < 64 ? sz_u64_mask_until_(n) : 0xFFFFFFFFFFFFFFFFull;
}
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SKYLAKE || SZ_USE_ICELAKE

/**
 *  @brief Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each byte signifies a match.
 */
SZ_HELPER_AUTO sz_u64_vec_t sz_u64_each_byte_equal_(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each byte is set.
    // For that take the bottom 7 bits of each byte, add one to them,
    // and if this sets the top bit to one, then all the 7 bits are ones as well.
    vec.u64 = ((vec.u64 & 0x7F7F7F7F7F7F7F7Full) + 0x0101010101010101ull) & ((vec.u64 & 0x8080808080808080ull));
    return vec;
}

/**
 *  @brief Clamps signed offsets to a Python-style `[offset, offset + length)` slice and reports whether
 *         the requested window was non-degenerate. Mirrors CPython's `ADJUST_INDICES`: negative indices
 *         count from the end, `end` is clamped into the string, and a window with `start > end` (out of
 *         range or inverted) is "empty" -- exactly when `str.find("")`/`rfind("")` return -1.
 */
SZ_HELPER_AUTO sz_bool_t sz_ssize_clamp_interval_checked( //
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
 *  @note Thin wrapper over @ref sz_ssize_clamp_interval_checked for callers that ignore window validity.
 */
SZ_HELPER_AUTO void sz_ssize_clamp_interval( //
    sz_size_t length, sz_ssize_t start, sz_ssize_t end, sz_size_t *normalized_offset, sz_size_t *normalized_length) {
    sz_ssize_clamp_interval_checked(length, start, end, normalized_offset, normalized_length);
}

/**
 *  @brief Compute the logarithm base 2 of a positive integer, rounding down.
 *  @pre Input must be a positive number, as the logarithm of zero is undefined.
 */
SZ_HELPER_AUTO sz_size_t sz_size_log2i_nonzero(sz_size_t x) {
    sz_assert_(x > 0 && "Non-positive numbers have no defined logarithm");
    int leading_zeros = sz_u64_clz(x);
    return (sz_size_t)(63 - leading_zeros);
}

/**
 *  @brief Computes the ceiling of @p x divided by @p divisor - the number of @p divisor-sized
 *         chunks needed to cover @p x. Assumes a non-zero @p divisor and no overflow on `x + divisor`.
 */
SZ_HELPER_AUTO sz_size_t sz_size_divide_round_up(sz_size_t x, sz_size_t divisor) { return (x + divisor - 1) / divisor; }

/**
 *  @brief Compute the smallest power of two greater than or equal to @p x.
 *  @note Uses LZCNT/CLZ for efficient computation on modern CPUs.
 *        Edge cases: bit_ceil(0) = 0, bit_ceil(1) = 1.
 *  @see https://stackoverflow.com/a/10143264
 */
SZ_HELPER_AUTO sz_size_t sz_size_bit_ceil(sz_size_t x) {
#if defined(__LZCNT__) || defined(__BMI__)
    // Edge cases: 0 and 1 return themselves, avoids undefined clz(0).
    if (x <= 1) return x;
#if SZ_IS_64BIT_
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
#if SZ_IS_64BIT_
    x |= x >> 32;
#endif
    x++;
    return x;
#endif
}

/**
 *  @brief Transposes an 8x8 bit matrix packed in a `sz_u64_t`.
 *
 *  There is a well known SWAR sequence for that known to chess programmers,
 *  willing to flip a bit-matrix of pieces along the main A1-H8 diagonal.
 *  https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating
 *  https://lukas-prokop.at/articles/2021-07-23-transpose
 */
SZ_HELPER_AUTO sz_u64_t sz_u64_transpose(sz_u64_t x) {
    sz_u64_t t;
    t = x ^ (x << 36);
    x ^= 0xf0f0f0f00f0f0f0full & (t ^ (x >> 36));
    t = 0xcccc0000cccc0000ull & (x ^ (x << 18));
    x ^= t ^ (t >> 18);
    t = 0xaa00aa00aa00aa00ull & (x ^ (x << 9));
    x ^= t ^ (t >> 9);
    return x;
}

/** @brief Load a 16-bit unsigned integer from a potentially unaligned pointer. Can be expensive on some platforms.
 */
SZ_HELPER_AUTO sz_u16_vec_t sz_u16_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u16_vec_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    return result;
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    return *((sz_u16_vec_t *)ptr);
#else
    return *((__unaligned sz_u16_vec_t *)ptr);
#endif
#else
    __attribute__((aligned(1))) sz_u16_vec_t const *result = (sz_u16_vec_t const *)ptr;
    return *result;
#endif
}

/** @brief Load a 32-bit unsigned integer from a potentially unaligned pointer. Can be expensive on some platforms.
 */
SZ_HELPER_AUTO sz_u32_vec_t sz_u32_load(sz_cptr_t ptr) {
#if !SZ_USE_MISALIGNED_LOADS
    sz_u32_vec_t result;
    result.u8s[0] = ptr[0];
    result.u8s[1] = ptr[1];
    result.u8s[2] = ptr[2];
    result.u8s[3] = ptr[3];
    return result;
#elif defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    return *((sz_u32_vec_t *)ptr);
#else
    return *((__unaligned sz_u32_vec_t *)ptr);
#endif
#else
    __attribute__((aligned(1))) sz_u32_vec_t const *result = (sz_u32_vec_t const *)ptr;
    return *result;
#endif
}

/** @brief Load a 64-bit unsigned integer from a potentially unaligned pointer. Can be expensive on some platforms.
 */
SZ_HELPER_AUTO sz_u64_vec_t sz_u64_load(sz_cptr_t ptr) {
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
#if defined(_M_IX86) //< The `__unaligned` modifier isn't valid for the x86 platform.
    return *((sz_u64_vec_t *)ptr);
#else
    return *((__unaligned sz_u64_vec_t *)ptr);
#endif
#else
    __attribute__((aligned(1))) sz_u64_vec_t const *result = (sz_u64_vec_t const *)ptr;
    return *result;
#endif
}

/** @brief Store a 16-bit unsigned integer to a potentially unaligned pointer. Can be expensive on some platforms. */
SZ_HELPER_AUTO void sz_u16_store(sz_ptr_t ptr, sz_u16_t value) {
#if !SZ_USE_MISALIGNED_LOADS
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
    __attribute__((aligned(1))) sz_u16_vec_t *result = (sz_u16_vec_t *)ptr;
    result->u16 = value;
#endif
}

/** @brief Store a 32-bit unsigned integer to a potentially unaligned pointer. Can be expensive on some platforms. */
SZ_HELPER_AUTO void sz_u32_store(sz_ptr_t ptr, sz_u32_t value) {
#if !SZ_USE_MISALIGNED_LOADS
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
    __attribute__((aligned(1))) sz_u32_vec_t *result = (sz_u32_vec_t *)ptr;
    result->u32 = value;
#endif
}

/** @brief Store a 64-bit unsigned integer to a potentially unaligned pointer. Can be expensive on some platforms. */
SZ_HELPER_AUTO void sz_u64_store(sz_ptr_t ptr, sz_u64_t value) {
#if !SZ_USE_MISALIGNED_LOADS
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
    __attribute__((aligned(1))) sz_u64_vec_t *result = (sz_u64_vec_t *)ptr;
    result->u64 = value;
#endif
}

/** @brief Helper function, using the supplied fixed-capacity buffer to allocate memory. */
SZ_HELPER_AUTO sz_ptr_t sz_memory_allocate_fixed_(sz_size_t length, void *handle) {

    sz_size_t const capacity = *(sz_size_t *)handle;
    sz_size_t const consumed_capacity = *((sz_size_t *)handle + 1);
    if (consumed_capacity + length > capacity) return SZ_NULL_CHAR;
    // Increase the consumed capacity.
    *((sz_size_t *)handle + 1) += length;
    return (sz_ptr_t)handle + consumed_capacity;
}

/** @brief Helper "no-op" function, simulating memory deallocation when we use a "static" memory buffer. */
SZ_HELPER_AUTO void sz_memory_free_fixed_(sz_ptr_t start, sz_size_t length, void *handle) {
    sz_unused_(start && length && handle);
}

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#pragma endregion

#pragma region Serial Implementation

#if !SZ_AVOID_LIBC
#include <stdio.h>  // `fprintf`
#include <stdlib.h> // `malloc`, `EXIT_FAILURE`

SZ_API_COMPTIME void *sz_memory_allocate_default_(sz_size_t length, void *handle) {
    sz_unused_(handle);
    if (length == 0) return SZ_NULL;
    return malloc(length);
}
SZ_API_COMPTIME void sz_memory_free_default_(sz_ptr_t start, sz_size_t length, void *handle) {
    sz_unused_(handle && length);
    free(start);
}

#endif

SZ_API_COMPTIME void sz_memory_allocator_init_default(sz_memory_allocator_t *allocator) {
#if !SZ_AVOID_LIBC
    allocator->allocate = (sz_memory_allocate_t)sz_memory_allocate_default_;
    allocator->free = (sz_memory_free_t)sz_memory_free_default_;
#else
    allocator->allocate = (sz_memory_allocate_t)SZ_NULL;
    allocator->free = (sz_memory_free_t)SZ_NULL;
#endif
    allocator->handle = SZ_NULL;
}

SZ_API_COMPTIME void sz_memory_allocator_init_fixed(sz_memory_allocator_t *allocator, void *buffer, sz_size_t length) {
    // The logic here is simple - put the buffer capacity in the first slots of the buffer.
    // The second slot is used to store the current consumed capacity.
    // The rest of the buffer is used for the actual data.
    allocator->allocate = (sz_memory_allocate_t)sz_memory_allocate_fixed_;
    allocator->free = (sz_memory_free_t)sz_memory_free_fixed_;
    allocator->handle = buffer;
    sz_size_t *pointer = (sz_size_t *)buffer;
    pointer[0] = length;
    pointer[1] = sizeof(sz_size_t) * 2; // The capacity and consumption so far
}

SZ_API_COMPTIME sz_bool_t sz_memory_allocator_equal(sz_memory_allocator_t const *a, sz_memory_allocator_t const *b) {
    if (!a || !b) return sz_false_k;

    // Two allocators are considered equal if they have the same function pointers and handle
    return (a->allocate == b->allocate) && (a->free == b->free) && (a->handle == b->handle) ? sz_true_k : sz_false_k;
}

SZ_API_COMPTIME sz_cptr_t sz_sequence_from_null_terminated_strings_get_start_(void const *handle, sz_size_t i) {
    sz_cptr_t const *start = (sz_cptr_t const *)handle;
    return start[i];
}

SZ_API_COMPTIME sz_size_t sz_sequence_from_null_terminated_strings_get_length_(void const *handle, sz_size_t i) {
    sz_cptr_t const *start = (sz_cptr_t const *)handle;
    sz_size_t length = 0;
    for (sz_cptr_t ptr = start[i]; *ptr; ptr++) length++;
    return length;
}

SZ_API_COMPTIME void sz_sequence_from_null_terminated_strings(sz_cptr_t *start, sz_size_t count,
                                                              sz_sequence_t *sequence) {
    sequence->handle = start;
    sequence->count = count;
    sequence->get_start = sz_sequence_from_null_terminated_strings_get_start_;
    sequence->get_length = sz_sequence_from_null_terminated_strings_get_length_;
}

#pragma endregion

#ifdef __cplusplus
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}
#endif // __cplusplus

#endif // STRINGZILLA_TYPES_H_
