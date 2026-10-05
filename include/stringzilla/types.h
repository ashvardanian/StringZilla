/**
 *  @file include/stringzilla/types.h
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
 *  - @c sz_sequence_t - a wrapper to access strings forming a sequential container.
 *  - @c sz_byteset_t - a bitset for 256 possible byte values.
 */
#if !defined(STRINGZILLA_TYPES_H_)
#define STRINGZILLA_TYPES_H_

#define STRINGZILLA_H_VERSION_MAJOR 5
#define STRINGZILLA_H_VERSION_MINOR 1
#define STRINGZILLA_H_VERSION_PATCH 2

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
#define STRINGZILLA_ARCH_BIG_ENDIAN_ (1) //< It's a big-endian target architecture
#else
#define STRINGZILLA_ARCH_BIG_ENDIAN_ (0) //< It's a little-endian target architecture
#endif
#endif

/** Infer the target architecture, unless it's overridden by the build system. At this point we only
 *  provide optimized backends for x86_64 and AArch64. */
#if !defined(STRINGZILLA_ARCH_X8664_)
#if defined(__x86_64__) || defined(_M_X64)
#define STRINGZILLA_ARCH_X8664_ (1)
#else
#define STRINGZILLA_ARCH_X8664_ (0)
#endif
#endif
#if !defined(STRINGZILLA_ARCH_ARM64_)
#if defined(__aarch64__) || defined(__arm64__) || defined(__arm64) || defined(_M_ARM64)
#define STRINGZILLA_ARCH_ARM64_ (1)
#else
#define STRINGZILLA_ARCH_ARM64_ (0)
#endif
#endif

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
 *  STRINGZILLA_OUTLINED_     internal helper forced out-of-line
 *  STRINGZILLA_DEVICE        CUDA or HIP device helper forced inline
 *  @endverbatim
 *
 *  @c STRINGZILLA_OUTLINED_ omits @c inline, as GCC ignores @c noinline on an @c inline function.
 */
#if defined(__cplusplus)
#define STRINGZILLA_C_INLINE_ inline
#else
#define STRINGZILLA_C_INLINE_ inline static
#endif

#if defined(__GNUC__) || defined(__clang__)
#define STRINGZILLA_MAYBE_UNUSED_ __attribute__((unused))
#else
#define STRINGZILLA_MAYBE_UNUSED_
#endif

#if defined(_MSC_VER)
#define STRINGZILLA_INLINE __forceinline static
#define STRINGZILLA_OUTLINED_ __declspec(noinline) static
#else
#define STRINGZILLA_INLINE __attribute__((always_inline)) STRINGZILLA_C_INLINE_
#define STRINGZILLA_OUTLINED_ STRINGZILLA_MAYBE_UNUSED_ static __attribute__((noinline))
#endif

/**
 *  @brief A portable scalar helper, inline in whichever translation unit uses it, and @c constexpr
 *      from C++20 onwards.
 *
 *  That qualifier is what lets a caller fold the helper at compile time, and what lets a CUDA
 *  kernel call it at all - `--expt-relaxed-constexpr` reaches a host @c constexpr function from
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
 *
 *  MSVC is the one front end that gets the plain helper: its bit-scan and byte-swap intrinsics are
 *  not constant-evaluable, so @c sz_u64_ctz and every helper that reaches one -
 *  @c sz_size_bit_ceil, the folded rune iterators, the uncased search - is rejected with C3615, an
 *  error no @c /wd can silence. The qualifier buys compile-time folding and @c nvcc reach, and MSVC
 *  hosts neither, so nothing is lost by dropping it.
 */
#if defined(__cplusplus) && __cplusplus >= 202002L && !(defined(_MSC_VER) && !defined(__clang__))
#define STRINGZILLA_CONSTEXPR STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_ constexpr
#else
#define STRINGZILLA_CONSTEXPR STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_
#endif

/** Every public function: header-inline in header-only builds, otherwise defined once in the
 *  library. Windows DLLs export every symbol through CMake, so no import or export spelling. */
#if STRINGZILLA_HEADER_ONLY
#define STRINGZILLA_API STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_
#elif defined(__GNUC__) || defined(__clang__)
#define STRINGZILLA_API __attribute__((visibility("default")))
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

/** Device-resident lookup table storage, without the size limit of CUDA constant memory. */
#define STRINGZILLA_DEVICE_CONSTANT static __device__
#endif

/**
 *  @brief Disables stack protection for performance-critical functions.
 *
 *  GCC's `-fstack-protector-strong` inserts stack canary checks for functions with local arrays
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
 *
 *  @sa sz_do_not_optimize in `types.hpp`, its C++ read-side companion for a value, not a buffer.
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
 *      `static n` enabling compiler bounds checking. In C++, expands to nothing as this syntax
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
 *  LibC header is pulled in. A strict @c -std=cNN otherwise hides declarations like `syscall()`,
 *  which the RISC-V @c riscv_hwprobe capability probe in `capabilities.h` relies on. As the first
 *  StringZilla header every translation unit includes, this is the one place that covers every
 *  build system, be it CMake, Cargo, or setuptools. */
#if defined(__linux__) && STRINGZILLA_WITH_LIBC && !defined(_DEFAULT_SOURCE) && !defined(_GNU_SOURCE)
#define _DEFAULT_SOURCE 1
#endif

#if STRINGZILLA_WITH_LIBC
#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`
#endif

/*  The headers needed for the @c sz_assert_failure_ function. */
#if STRINGZILLA_DEBUG && STRINGZILLA_WITH_LIBC
#include <stdio.h>  // `fprintf`, `stderr`
#include <stdlib.h> // `abort`
#endif

/*  Which kits' kernels a unit defines, by default those the compiler's own flags enable. Each tier
 *  header scopes its kernels to their kit with a target pragma, so a unit at the baseline flags may
 *  define any kit the toolchain builds, as the CMake build does from its probes, except LASX and
 *  POWER9, as `lasxintrin.h` and `altivec.h` hide without `-mlasx` and `-mcpu=power9`. */
#if !defined(STRINGZILLA_TARGET_WESTMERE)
#if STRINGZILLA_ARCH_X8664_ && defined(__SSE4_2__) && defined(__AES__)
#define STRINGZILLA_TARGET_WESTMERE (1)
#elif STRINGZILLA_ARCH_X8664_ && defined(_MSC_VER) && defined(__AVX__)
#define STRINGZILLA_TARGET_WESTMERE (1) // ! MSVC doesn't expose `__SSE4_2__`, `__AES__` macros
#else
#define STRINGZILLA_TARGET_WESTMERE (0)
#endif
#endif

#if !defined(STRINGZILLA_TARGET_HASWELL)
#if STRINGZILLA_ARCH_X8664_ && defined(__AVX2__)
#define STRINGZILLA_TARGET_HASWELL (1)
#else
#define STRINGZILLA_TARGET_HASWELL (0)
#endif
#endif

#if !defined(STRINGZILLA_TARGET_GOLDMONT)
#if STRINGZILLA_ARCH_X8664_ && defined(__SHA__)
#define STRINGZILLA_TARGET_GOLDMONT (1)
#elif STRINGZILLA_ARCH_X8664_ && defined(_MSC_VER) && defined(__AVX2__)
#define STRINGZILLA_TARGET_GOLDMONT (1) // ! MSVC doesn't expose `__SHA__` macros
#else
#define STRINGZILLA_TARGET_GOLDMONT (0)
#endif
#endif

#if !defined(STRINGZILLA_TARGET_SKYLAKE)
#if STRINGZILLA_ARCH_X8664_ && defined(__AVX512F__)
#define STRINGZILLA_TARGET_SKYLAKE (1)
#else
#define STRINGZILLA_TARGET_SKYLAKE (0)
#endif
#endif

#if !defined(STRINGZILLA_TARGET_ICELAKE)
#if STRINGZILLA_ARCH_X8664_ && defined(__AVX512BW__) && defined(__VAES__)
#define STRINGZILLA_TARGET_ICELAKE (1)
#elif STRINGZILLA_ARCH_X8664_ && defined(_MSC_VER) && defined(__AVX512BW__)
#define STRINGZILLA_TARGET_ICELAKE (1) // ! MSVC doesn't expose `__VAES__` macros
#else
#define STRINGZILLA_TARGET_ICELAKE (0)
#endif
#endif

/*  NEON support is optional in Armv7/AArch32, but mandatory from 8.0 onwards. The
 *  `!STRINGZILLA_ARCH_BIG_ENDIAN_` guard keeps the SIMD stack off exotic @c aarch64_be targets: the
 *  shared byte-lane bridges, like `sz_utf8_rune_pred_to_u64_*` and
 *  @c sz_utf8_vreinterpretq_u8_u4_neon_, assume little-endian lane ↔ byte order and would
 *  mis-execute on big-endian. */
#if !defined(STRINGZILLA_TARGET_NEON)
#if STRINGZILLA_ARCH_ARM64_ && defined(__ARM_NEON) && !STRINGZILLA_ARCH_BIG_ENDIAN_
#define STRINGZILLA_TARGET_NEON (1)
#elif STRINGZILLA_ARCH_ARM64_ && defined(_MSC_VER) && defined(_M_ARM64)
#define STRINGZILLA_TARGET_NEON (1) // ! MSVC doesn't expose `__ARM_NEON` macros; MSVC AArch64 is always little-endian
#else
#define STRINGZILLA_TARGET_NEON (0)
#endif
#endif

/*  SVE is optional since Armv8.2-A, but never became mandatory, and MSVC cannot probe for it. */
#if !defined(STRINGZILLA_TARGET_SVE)
#if STRINGZILLA_ARCH_ARM64_ && defined(__ARM_FEATURE_SVE) && !STRINGZILLA_ARCH_BIG_ENDIAN_
#define STRINGZILLA_TARGET_SVE (1)
#else
#define STRINGZILLA_TARGET_SVE (0)
#endif
#endif

/*  SVE2 is optional since Armv9.0-A, but never became mandatory, and MSVC cannot probe for it. */
#if !defined(STRINGZILLA_TARGET_SVE2)
#if STRINGZILLA_ARCH_ARM64_ && defined(__ARM_FEATURE_SVE2) && !STRINGZILLA_ARCH_BIG_ENDIAN_
#define STRINGZILLA_TARGET_SVE2 (1)
#else
#define STRINGZILLA_TARGET_SVE2 (0)
#endif
#endif

/*  AES is optional since Armv8.0-A, but never became mandatory, and MSVC cannot probe for it. */
#if !defined(STRINGZILLA_TARGET_NEONAES)
#if STRINGZILLA_ARCH_ARM64_ && (defined(__ARM_FEATURE_AES) || defined(__ARM_FEATURE_CRYPTO) || defined(__APPLE__))
#define STRINGZILLA_TARGET_NEONAES (1)
#else
#define STRINGZILLA_TARGET_NEONAES (0)
#endif
#endif

/*  SHA2 is optional since Armv8.0-A, but never became mandatory, and MSVC cannot probe for it. */
#if !defined(STRINGZILLA_TARGET_NEONSHA)
#if STRINGZILLA_ARCH_ARM64_ && (defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO) || defined(__APPLE__))
#define STRINGZILLA_TARGET_NEONSHA (1)
#else
#define STRINGZILLA_TARGET_NEONSHA (0)
#endif
#endif

/*  SVE2 AES is optional since Armv9.0-A, but never became mandatory, and MSVC cannot probe for it.
 *  GCC spells the feature macro @c __ARM_FEATURE_SVE2AES; Clang spells it
 *  @c __ARM_FEATURE_SVE2_AES. Accept both. */
#if !defined(STRINGZILLA_TARGET_SVE2AES)
#if STRINGZILLA_ARCH_ARM64_ && (defined(__ARM_FEATURE_SVE2AES) || defined(__ARM_FEATURE_SVE2_AES))
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
 *  also sets it for its host units, which list the kernels of its one CUDA unit. */
#if !defined(STRINGZILLA_ARCH_CUDA_)
#if defined(__CUDACC__) && !defined(__HIP__)
#define STRINGZILLA_ARCH_CUDA_ (1)
#else
#define STRINGZILLA_ARCH_CUDA_ (0)
#endif
#endif

/** Compiled as HIP for AMD GPUs, host and device passes alike; Clang predefines @c __HIP__ itself,
 *  unlike @c __HIP_PLATFORM_AMD__, which only `hip_common.h` defines. The library also sets it for
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
#if !defined(STRINGZILLA_WITH_METAL)
#define STRINGZILLA_WITH_METAL (0)
#endif

/** Defining the serial kernels: STRINGZILLA_TARGET_SERIAL. Header-only builds define them in every
 *  translation unit; in the library only its serial unit does, so units that include a serial
 *  header for its helpers leave its kernels to the library. */
#if !defined(STRINGZILLA_TARGET_SERIAL)
#define STRINGZILLA_TARGET_SERIAL STRINGZILLA_HEADER_ONLY
#endif

/*  Defining the CUDA kernels, the SIMT baseline every NVIDIA device runs: STRINGZILLA_TARGET_CUDA.
 *  Forced off where the unit is not compiled as CUDA. */
#if !defined(STRINGZILLA_TARGET_CUDA) || (STRINGZILLA_TARGET_CUDA && !STRINGZILLA_ARCH_CUDA_)
#undef STRINGZILLA_TARGET_CUDA
#define STRINGZILLA_TARGET_CUDA STRINGZILLA_ARCH_CUDA_
#endif

/*  Defining the ROCm kernels, the SIMT baseline every AMD device runs: STRINGZILLA_TARGET_ROCM.
 *  Forced off where the unit is not compiled as HIP. */
#if !defined(STRINGZILLA_TARGET_ROCM) || (STRINGZILLA_TARGET_ROCM && !STRINGZILLA_ARCH_ROCM_)
#undef STRINGZILLA_TARGET_ROCM
#define STRINGZILLA_TARGET_ROCM STRINGZILLA_ARCH_ROCM_
#endif

/*  Defining the Metal kernels, the SIMT baseline every Apple7 device runs:
 *  STRINGZILLA_TARGET_METAL. Forced off where Metal is not linked. */
#if !defined(STRINGZILLA_TARGET_METAL) || (STRINGZILLA_TARGET_METAL && !STRINGZILLA_WITH_METAL)
#undef STRINGZILLA_TARGET_METAL
#define STRINGZILLA_TARGET_METAL STRINGZILLA_WITH_METAL
#endif

/*  WebAssembly SIMD128 is opt-in via @c -msimd128 with no runtime probe: an engine validates a
 *  module whole, so the flag alone decides, and a request without it is dropped. */
#if !defined(STRINGZILLA_TARGET_V128) || (STRINGZILLA_TARGET_V128 && !(defined(__wasm__) && defined(__wasm_simd128__)))
#undef STRINGZILLA_TARGET_V128
#if defined(__wasm__) && defined(__wasm_simd128__)
#define STRINGZILLA_TARGET_V128 (1)
#else
#define STRINGZILLA_TARGET_V128 (0)
#endif
#endif

/*  WebAssembly @b relaxed SIMD — opt-in via @c -mrelaxed-simd; a level above baseline SIMD128
 *  adding relaxed swizzle, fused multiply-add, lane-select, and integer dot-products. Some
 *  runtimes lower a few relaxed ops sub-optimally, but the level is exposed so native engines
 *  can use them. As with SIMD128, the flag alone decides. */
#if !defined(STRINGZILLA_TARGET_V128RELAXED) || \
    (STRINGZILLA_TARGET_V128RELAXED && !(defined(__wasm__) && defined(__wasm_relaxed_simd__)))
#undef STRINGZILLA_TARGET_V128RELAXED
#if defined(__wasm__) && defined(__wasm_relaxed_simd__)
#define STRINGZILLA_TARGET_V128RELAXED (1)
#else
#define STRINGZILLA_TARGET_V128RELAXED (0)
#endif
#endif

/*  RISC-V Vector extension (RVV 1.0) — `-march=rv64gcv`. Length-agnostic registers. */
#if !defined(STRINGZILLA_TARGET_RVV)
#if defined(__riscv) && (__riscv_xlen == 64) && defined(__riscv_vector)
#define STRINGZILLA_TARGET_RVV (1)
#else
#define STRINGZILLA_TARGET_RVV (0)
#endif
#endif

/*  RISC-V Vector Crypto (Zvk: Zvkned AES + Zvknhb SHA) — `-march=rv64gcv_zvkned_zvknhb`. */
#if !defined(STRINGZILLA_TARGET_RVVCRYPTO)
#if STRINGZILLA_TARGET_RVV && defined(__riscv_zvkned) && defined(__riscv_zvknhb)
#define STRINGZILLA_TARGET_RVVCRYPTO (1)
#else
#define STRINGZILLA_TARGET_RVVCRYPTO (0)
#endif
#endif

/*  LoongArch Advanced SIMD eXtension (LASX, 256-bit) — `-mlasx`. */
#if !defined(STRINGZILLA_TARGET_LOONGSONASX)
#if defined(__loongarch__) && defined(__loongarch_asx)
#define STRINGZILLA_TARGET_LOONGSONASX (1)
#else
#define STRINGZILLA_TARGET_LOONGSONASX (0)
#endif
#endif

/*  IBM Power Vector-Scalar eXtension at the POWER9 level, ISA 3.0 — `-mcpu=power9`. */
#if !defined(STRINGZILLA_TARGET_POWERVSX)
#if (defined(__powerpc__) || defined(__powerpc64__)) && defined(__VSX__) && defined(__POWER9_VECTOR__)
#define STRINGZILLA_TARGET_POWERVSX (1)
#else
#define STRINGZILLA_TARGET_POWERVSX (0)
#endif
#endif

/** Whether a capability's helpers compile here: its own target, or any capability built on it.
 *  @c STRINGZILLA_TARGET_* alone decides where its kernels are defined, and a capability nothing
 *  builds on guards its helpers with its target. */
#define STRINGZILLA_ARCH_X8664_SKYLAKE_ (STRINGZILLA_TARGET_SKYLAKE || STRINGZILLA_TARGET_ICELAKE)
#define STRINGZILLA_ARCH_X8664_HASWELL_ (STRINGZILLA_TARGET_HASWELL || STRINGZILLA_ARCH_X8664_SKYLAKE_)
#define STRINGZILLA_ARCH_X8664_WESTMERE_ (STRINGZILLA_TARGET_WESTMERE || STRINGZILLA_ARCH_X8664_HASWELL_)
#define STRINGZILLA_ARCH_ARM64_SVE2_ (STRINGZILLA_TARGET_SVE2 || STRINGZILLA_TARGET_SVE2AES)
#define STRINGZILLA_ARCH_ARM64_SVE_ (STRINGZILLA_TARGET_SVE || STRINGZILLA_ARCH_ARM64_SVE2_)
#define STRINGZILLA_ARCH_ARM64_NEONAES_ (STRINGZILLA_TARGET_NEONAES || STRINGZILLA_TARGET_SVE2AES)
#define STRINGZILLA_ARCH_ARM64_NEON_                                                             \
    (STRINGZILLA_TARGET_NEON || STRINGZILLA_ARCH_ARM64_NEONAES_ || STRINGZILLA_TARGET_NEONSHA || \
     STRINGZILLA_ARCH_ARM64_SVE_)
#define STRINGZILLA_ARCH_RISCV64_RVV_ (STRINGZILLA_TARGET_RVV || STRINGZILLA_TARGET_RVVCRYPTO)
#define STRINGZILLA_ARCH_WASM_V128_ (STRINGZILLA_TARGET_V128 || STRINGZILLA_TARGET_V128RELAXED)

/*  Hardware-specific headers for different SIMD intrinsics and register wrappers. */
#if STRINGZILLA_ARCH_WASM_V128_
#include <wasm_simd128.h>
#endif // STRINGZILLA_ARCH_WASM_V128_
#if STRINGZILLA_ARCH_RISCV64_RVV_
#include <riscv_vector.h>
#endif // STRINGZILLA_ARCH_RISCV64_RVV_
#if defined(__loongarch_asx)
#include <lsxintrin.h>  // 128-bit `__lsx_*` intrinsics and the `__m128i` register type, for sub-32-byte inputs
#include <lasxintrin.h> // 256-bit `__lasx_*` intrinsics and the `__m256i` register type
#endif
#if defined(__POWER9_VECTOR__)
#include <altivec.h>
#endif
#if STRINGZILLA_ARCH_X8664_WESTMERE_ || STRINGZILLA_TARGET_GOLDMONT
#include <immintrin.h>
#endif // STRINGZILLA_ARCH_X8664_WESTMERE_ || STRINGZILLA_TARGET_GOLDMONT
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
#endif // STRINGZILLA_WITH_LIBC

/** Compile-time assert akin to C++ @c static_assert. Uses the native assertion where available
 *  (C++11 @c static_assert, C11 @c _Static_assert); the older-C typedef fallback must sit at file
 *  scope to stay clear of @c -Wunused-local-typedef. */
#if defined(__cplusplus) && __cplusplus >= 201103L
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

struct sz_hash_state_t;            // Forward declaration of a hash state structure
struct sz_sha256_state_t;          // Forward declaration of a SHA256 hash state structure
struct sz_aes256_key_t;            // Forward declaration of an AES-256 round-key schedule
struct sz_aes256_gcm_key_t;        // Forward declaration of an AES-256-GCM key, schedule plus hash powers
struct sz_aes256_gcm_state_t;      // Forward declaration of the payload both streaming directions share
struct sz_aes256_gcm_encryptor_t;  // Forward declaration of an AES-256-GCM sealing state
struct sz_aes256_gcm_decryptor_t;  // Forward declaration of an AES-256-GCM opening state
struct sz_sequence_t;              // Forward declaration of an ordered collection of strings
struct sz_substrings_match_t;      // Forward declaration of one located multi-pattern match
struct sz_substrings_bm25_t;       // Forward declaration of BM25's continuous parameters
struct sz_levenshtein_engine_t;    // Forward declaration of a batch of prepared Levenshtein queries
struct sz_overlap_engine_t;        // Forward declaration of a forest of prepared overlap query trees
struct sz_substrings_engine_t;     // Forward declaration of a compiled multi-pattern vocabulary
typedef sz_size_t sz_sorted_idx_t; // Index of a sorted string in a list of strings
typedef sz_size_t sz_pgram_t;      // "Pointer-sized N-gram" of a string

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

/** Unicode normalization form selector, see `utf8_norm.h`. */
typedef enum sz_normal_form_t {

    /** Canonical decomposition. */
    sz_normal_form_nfd_k = 0,

    /** Canonical decomposition followed by canonical composition. */
    sz_normal_form_nfc_k = 1,

    /** Compatibility decomposition. */
    sz_normal_form_nfkd_k = 2,

    /** Compatibility decomposition followed by canonical composition. */
    sz_normal_form_nfkc_k = 3,
} sz_normal_form_t;

/** Which symbols a batch counts, as the alphabet picks the transpose and the mask layout alike. */
typedef enum sz_levenshtein_symbol_t {

    /** Every byte is its own symbol, and a distance counts bytes. */
    sz_levenshtein_bytes_k = 0,

    /** Every UTF-8 rune is one symbol, an ill-formed byte decoding to U+FFFD. */
    sz_levenshtein_runes_k = 1,
} sz_levenshtein_symbol_t;

/** Whether a vocabulary matches needles byte-for-byte, or folds both sides to one case first. */
typedef enum sz_substrings_case_sensitivity_t {

    /** Byte-exact matching; needles may be arbitrary bytes, including malformed UTF-8. */
    sz_substrings_cased_k = 0,

    /** Full Unicode case folding as `CaseFolding.txt` defines it; needles must be valid UTF-8. */
    sz_substrings_uncased_k = 1,
} sz_substrings_case_sensitivity_t;

/** How matches that share bytes resolve: reported in full, or thinned to a leftmost run. */
typedef enum sz_substrings_overlap_policy_t {

    /** Every match of every needle, including ones that share bytes and ones nested in others. */
    sz_substrings_overlapping_k = 0,

    /** Matches sharing no bytes: earliest start, then longest span, then lower needle index. */
    sz_substrings_leftmost_longest_k = 1,

    /** Matches sharing no bytes: earliest start, then lower needle index, whatever the lengths. */
    sz_substrings_leftmost_first_k = 2,
} sz_substrings_overlap_policy_t;

/** Outcome of every StringZilla call that can fail: zero on success, negative when the call failed
 *  and its outputs hold no result. Positive values are reserved for results with a caveat. */
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
typedef void (*sz_free_t)(void *pointer, sz_size_t bytes, void *handle, sz_stream_t stream);

/**
 *  @brief Some complex pattern matching algorithms may require memory allocations. This structure
 *      passes the memory allocator to those functions.
 *
 *  Both functions take the caller's stream, which on a GPU names the device a block is made on and
 *  the work a release waits behind; host allocators ignore it, and on the CPU it is null.
 *
 *  @sa sz_allocator_init_arena, sz_allocator_init_unified_best
 */
typedef struct sz_allocator_t {
    sz_allocate_t allocate;
    sz_free_t free;
    void *handle;
} sz_allocator_t;

/**
 *  @brief Initializes a memory allocator to use the system default @c malloc and @c free.
 *  @param[out] allocator Memory allocator to initialize.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k without libc.
 *      Failure preserves the allocator.
 *  @note Unlike the C standard library, `malloc(0)` is guaranteed to return a null pointer.
 *  @see malloc: https://en.cppreference.com/w/c/memory/malloc
 */
STRINGZILLA_INLINE sz_status_t sz_allocator_init_heap(sz_allocator_t *allocator);

/**
 *  @brief Initializes a memory allocator that serves every request from a static-capacity buffer,
 *      @b without any dynamic allocations.
 *  @param[out] allocator Memory allocator to initialize.
 *  @param[in] buffer Buffer to use for allocations.
 *  @param[in] bytes Buffer size, including metadata and alignment padding.
 *  @param[in] alignment Nonzero power-of-two alignment guaranteed for every returned block.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k for invalid alignment, @c sz_bad_alloc_k
 *      for missing storage, or @c sz_overflow_risk_k for an overflowing address range. Failure
 *      leaves both the allocator and buffer unchanged.
 *
 *  Three aligned @c sz_size_t values store capacity, consumption and alignment inside the buffer.
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
#if defined(__loongarch_asx)
    __m128i lsx;
#endif
#if STRINGZILLA_ARCH_WASM_V128_
    v128_t v128;
#endif
#if defined(__POWER9_VECTOR__)
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
#if defined(__loongarch_asx)
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

#pragma region String Sequences API

/** Signature of @c sz_sequence_t::get_start, returning the start of the string at an index. */
typedef sz_cptr_t (*sz_sequence_member_start_t)(void const *, sz_sorted_idx_t);

/** Signature of @c sz_sequence_t::get_length, returning the length of the string at an index. */
typedef sz_size_t (*sz_sequence_member_length_t)(void const *, sz_sorted_idx_t);

/**
 *  @brief Structure to represent an ordered collection of strings, in whatever layout.
 *
 *  It can be easily combined with Apache Arrow and its tape-like concatenated strings.
 *
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
 *  @param[in] start Pointer to the array of strings.
 *  @param[in] count Number of strings in the array.
 *  @param[out] sequence Sequence structure to initialize.
 */
STRINGZILLA_INLINE void sz_sequence_from_null_terminated_strings(sz_cptr_t *start, sz_size_t count,
                                                                 sz_sequence_t *sequence);

/**
 *  @brief Initiates the sequence structure from an array of pointer-length pairs, like
 *      `sz_string_view_t[]`.
 *  @param[in] views Pointer to the array of views, which must outlive @p sequence.
 *  @param[in] count Number of views in the array.
 *  @param[out] sequence Sequence structure to initialize.
 */
STRINGZILLA_INLINE void sz_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                      sz_sequence_t *sequence);

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

/** Defines @c STRINGZILLA_NULL, analogous to @c NULL. The default often comes from locale.h,
 *  stddef.h, stdio.h, stdlib.h, string.h, time.h, or wchar.h. */
#if defined(__cplusplus)
#define STRINGZILLA_NULL nullptr
#define STRINGZILLA_NULL_CHAR nullptr
#else
#define STRINGZILLA_NULL ((void *)0)
#define STRINGZILLA_NULL_CHAR ((char *)0)
#endif

/** Default alignment policy for padding and allocation-size heuristics. */
#if !defined(STRINGZILLA_DEFAULT_ALIGNMENT)
#if defined(__s390x__)
#define STRINGZILLA_DEFAULT_ALIGNMENT 256
#elif defined(__wasm__) || defined(__EMSCRIPTEN__)
#define STRINGZILLA_DEFAULT_ALIGNMENT 64
#else
#define STRINGZILLA_DEFAULT_ALIGNMENT 128
#endif
#endif
#if STRINGZILLA_DEFAULT_ALIGNMENT < 64 || (STRINGZILLA_DEFAULT_ALIGNMENT & (STRINGZILLA_DEFAULT_ALIGNMENT - 1))
#error "STRINGZILLA_DEFAULT_ALIGNMENT must be a power of two and at least 64 bytes"
#endif

enum { sz_default_alignment_k = STRINGZILLA_DEFAULT_ALIGNMENT };

#define STRINGZILLA_SIZE_MAX ((sz_size_t)(-1))
#define STRINGZILLA_SSIZE_MAX ((sz_ssize_t)(STRINGZILLA_SIZE_MAX >> 1))
#define STRINGZILLA_SSIZE_MIN ((sz_ssize_t)(-STRINGZILLA_SSIZE_MAX - 1))

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
STRINGZILLA_OUTLINED_ void sz_assert_failure_(char const *condition, char const *file, int line) {
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
#if (defined(_WIN32) && !defined(_WIN64)) || defined(_M_ARM) || defined(_M_ARM64)
STRINGZILLA_CONSTEXPR int sz_u64_ctz(sz_u64_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
STRINGZILLA_CONSTEXPR int sz_u64_clz(sz_u64_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 0x8000000000000000ull) == 0) { n++, x <<= 1; }
    return n;
}
STRINGZILLA_CONSTEXPR int sz_u64_popcount(sz_u64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    return (((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full) * 0x0101010101010101ull) >> 56;
}
STRINGZILLA_CONSTEXPR int sz_u32_ctz(sz_u32_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
STRINGZILLA_CONSTEXPR int sz_u32_clz(sz_u32_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 0x80000000u) == 0) { n++, x <<= 1; }
    return n;
}
STRINGZILLA_CONSTEXPR int sz_u32_popcount(sz_u32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
#else
STRINGZILLA_INLINE int sz_u64_ctz(sz_u64_t x) { return (int)_tzcnt_u64(x); }
STRINGZILLA_INLINE int sz_u64_clz(sz_u64_t x) { return (int)_lzcnt_u64(x); }
STRINGZILLA_INLINE int sz_u64_popcount(sz_u64_t x) { return (int)__popcnt64(x); }
STRINGZILLA_INLINE int sz_u32_ctz(sz_u32_t x) { return (int)_tzcnt_u32(x); }
STRINGZILLA_INLINE int sz_u32_clz(sz_u32_t x) { return (int)_lzcnt_u32(x); }
STRINGZILLA_INLINE int sz_u32_popcount(sz_u32_t x) { return (int)__popcnt(x); }
#endif
/*  Force the byteswap functions to be intrinsics, because when @c /Oi- is given, these will turn
 *  into CRT function calls, which breaks when @c STRINGZILLA_WITH_LIBC is 0. */
#pragma intrinsic(_byteswap_uint64)
STRINGZILLA_INLINE sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return _byteswap_uint64(val); }
#pragma intrinsic(_byteswap_ulong)
STRINGZILLA_INLINE sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return _byteswap_ulong(val); }
#else
STRINGZILLA_CONSTEXPR int sz_u64_popcount(sz_u64_t x) { return __builtin_popcountll(x); }
STRINGZILLA_CONSTEXPR int sz_u32_popcount(sz_u32_t x) { return __builtin_popcount(x); }
STRINGZILLA_CONSTEXPR int sz_u64_ctz(sz_u64_t x) { return __builtin_ctzll(x); }
STRINGZILLA_CONSTEXPR int sz_u64_clz(sz_u64_t x) { return __builtin_clzll(x); }
STRINGZILLA_CONSTEXPR int sz_u32_ctz(sz_u32_t x) { return __builtin_ctz(x); } // ! Undefined if `x == 0`
STRINGZILLA_CONSTEXPR int sz_u32_clz(sz_u32_t x) { return __builtin_clz(x); } // ! Undefined if `x == 0`
STRINGZILLA_CONSTEXPR sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return __builtin_bswap64(val); }
STRINGZILLA_CONSTEXPR sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return __builtin_bswap32(val); }
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
 *  Left visible, a `static const` table folds into immediates and every constant costs a
 *  @c vpbroadcastd where an AVX-512 `{1toN}` embedded broadcast would have been free. No use
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
STRINGZILLA_CONSTEXPR sz_u64_t sz_u64_bits_reverse(sz_u64_t value) {
    value = ((value & 0x5555555555555555ull) << 1) | ((value >> 1) & 0x5555555555555555ull);
    value = ((value & 0x3333333333333333ull) << 2) | ((value >> 2) & 0x3333333333333333ull);
    value = ((value & 0x0F0F0F0F0F0F0F0Full) << 4) | ((value >> 4) & 0x0F0F0F0F0F0F0F0Full);
    return sz_u64_bytes_reverse(value);
}

/** Bit index of the n-th (0-based) set bit of @p bits; clears the @p n lowest set bits, then
 *  @c ctz. @p bits must hold more than @p n set bits. One tested home for the per-ISA SIMD
 *  "n-th lane" locate. */
STRINGZILLA_CONSTEXPR int sz_u64_nth_set_bit(sz_u64_t bits, sz_size_t n) {
    while (n--) bits &= bits - 1;
    return sz_u64_ctz(bits);
}
STRINGZILLA_CONSTEXPR int sz_u32_nth_set_bit(sz_u32_t bits, sz_size_t n) {
    while (n--) bits &= bits - 1;
    return sz_u32_ctz(bits);
}

/** Branchless `value | (bit if condition)`: OR @p bit into @p value when @p condition holds, with
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
#define sz_min_of_two(x, y) (x < y ? x : y)
#define sz_max_of_two(x, y) (x < y ? y : x)
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

/** Clamps signed offsets to a Python-style `[offset, offset + length)` slice and reports whether
 *  the requested window was non-degenerate. Mirrors CPython's @c ADJUST_INDICES: negative
 *  indices count from the end, @p end is clamped into the string, and a window with @p start
 *  past @p end, out of range or inverted, is "empty" - exactly when `str.find("")` and
 *  `str.rfind("")` return -1. */
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
STRINGZILLA_CONSTEXPR sz_size_t sz_size_log2i_nonzero(sz_size_t x) {
    sz_assert_(x > 0 && "Non-positive numbers have no defined logarithm");
    int leading_zeros = sz_u64_clz(x);
    return (sz_size_t)(63 - leading_zeros);
}

/** Computes the ceiling of @p x divided by @p divisor - the number of chunks of that size needed to
 *  cover @p x. Assumes a non-zero @p divisor and no overflow on `x + divisor`. */
STRINGZILLA_CONSTEXPR sz_size_t sz_size_divide_round_up(sz_size_t x, sz_size_t divisor) {
    return (x + divisor - 1) / divisor;
}

/** Divides rounding up in 32 bits, for device code where the @c sz_size_t form costs registers. */
STRINGZILLA_CONSTEXPR sz_u32_t sz_u32_divide_round_up(sz_u32_t x, sz_u32_t divisor) {
    return (x + divisor - 1) / divisor;
}

/**
 *  @brief Compute the smallest power of two greater than or equal to @p x.
 *  @note Uses LZCNT/CLZ for efficient computation on modern CPUs. Edge cases: bit_ceil(0) = 0,
 *      bit_ceil(1) = 1.
 *  @see Rounding up to a power of two: https://stackoverflow.com/a/10143264
 */
STRINGZILLA_CONSTEXPR sz_size_t sz_size_bit_ceil(sz_size_t x) {
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
    __attribute__((aligned(1))) sz_u16_vec_t const *result_vec = (sz_u16_vec_t const *)ptr;
    return *result_vec;
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
    __attribute__((aligned(1))) sz_u32_vec_t const *result_vec = (sz_u32_vec_t const *)ptr;
    return *result_vec;
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
    __attribute__((aligned(1))) sz_u64_vec_t const *result_vec = (sz_u64_vec_t const *)ptr;
    return *result_vec;
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
    __attribute__((aligned(1))) sz_u16_vec_t *result_vec = (sz_u16_vec_t *)ptr;
    result_vec->u16 = value;
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
    __attribute__((aligned(1))) sz_u32_vec_t *result_vec = (sz_u32_vec_t *)ptr;
    result_vec->u32 = value;
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
    __attribute__((aligned(1))) sz_u64_vec_t *result_vec = (sz_u64_vec_t *)ptr;
    result_vec->u64 = value;
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
    sz_size_t const padding = (-((sz_size_t)handle + consumed)) & (alignment - 1);
    if (padding > capacity - consumed || bytes > capacity - consumed - padding) return STRINGZILLA_NULL;
    metadata->consumed = consumed + padding + bytes;
    return (sz_ptr_t)handle + consumed + padding;
}

/** Helper "no-op" function, simulating memory deallocation when we use a "static" memory buffer. */
STRINGZILLA_CONSTEXPR void sz_free_arena_(void *start, sz_size_t length, void *handle, sz_stream_t stream) {
    sz_unused_(start && length && handle && stream);
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

#endif

STRINGZILLA_INLINE sz_status_t sz_allocator_init_heap(sz_allocator_t *allocator) {
#if STRINGZILLA_WITH_LIBC
    allocator->allocate = sz_allocate_heap_;
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
    sz_size_t const padding = (-(sz_size_t)buffer) & (sizeof(sz_size_t) - 1);
    if (padding > bytes || bytes - padding < sizeof(sz_arena_t_)) return sz_bad_alloc_k;
    sz_arena_t_ *metadata = (sz_arena_t_ *)((sz_ptr_t)buffer + padding);
    metadata->capacity = bytes - padding;
    metadata->consumed = sizeof(sz_arena_t_);
    metadata->alignment = alignment;
    allocator->allocate = sz_allocate_arena_;
    allocator->free = sz_free_arena_;
    allocator->handle = metadata;
    return sz_success_k;
}

STRINGZILLA_CONSTEXPR sz_bool_t sz_allocator_equal(sz_allocator_t const *a, sz_allocator_t const *b) {
    if (!a || !b) return sz_false_k;

    // Two allocators are considered equal if they have the same function pointers and handle
    return (a->allocate == b->allocate) && (a->free == b->free) && (a->handle == b->handle) ? sz_true_k : sz_false_k;
}

STRINGZILLA_INLINE sz_cptr_t sz_sequence_from_null_terminated_strings_get_start_(void const *handle, sz_size_t i) {
    sz_cptr_t const *start = (sz_cptr_t const *)handle;
    return start[i];
}

STRINGZILLA_INLINE sz_size_t sz_sequence_from_null_terminated_strings_get_length_(void const *handle, sz_size_t i) {
    sz_cptr_t const *start = (sz_cptr_t const *)handle;
    sz_size_t length = 0;
    for (sz_cptr_t ptr = start[i]; *ptr; ptr++) length++;
    return length;
}

STRINGZILLA_INLINE void sz_sequence_from_null_terminated_strings(sz_cptr_t *start, sz_size_t count,
                                                                 sz_sequence_t *sequence) {
    sequence->handle = start;
    sequence->count = count;
    sequence->get_start = sz_sequence_from_null_terminated_strings_get_start_;
    sequence->get_length = sz_sequence_from_null_terminated_strings_get_length_;
}

STRINGZILLA_INLINE sz_cptr_t sz_sequence_from_string_views_get_start_(void const *handle, sz_size_t i) {
    sz_string_view_t const *views = (sz_string_view_t const *)handle;
    return views[i].start;
}

STRINGZILLA_INLINE sz_size_t sz_sequence_from_string_views_get_length_(void const *handle, sz_size_t i) {
    sz_string_view_t const *views = (sz_string_view_t const *)handle;
    return views[i].length;
}

STRINGZILLA_INLINE void sz_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                      sz_sequence_t *sequence) {
    sequence->handle = views;
    sequence->count = count;
    sequence->get_start = sz_sequence_from_string_views_get_start_;
    sequence->get_length = sz_sequence_from_string_views_get_length_;
}

/** Reads one string out of a tape: one block of count + 1 @c sz_u64_t offsets from the block's own
 *  start, then the bytes, string @p i spanning `[offsets[i], offsets[i + 1])`. Exported once, so
 *  every unit of the library knows a tape by this one address. */
STRINGZILLA_API sz_cptr_t sz_sequence_tape_start(void const *handle, sz_size_t i);

/** Reads one length out of a tape, laid out as @ref sz_sequence_tape_start reads it. */
STRINGZILLA_API sz_size_t sz_sequence_tape_length(void const *handle, sz_size_t i);

#if STRINGZILLA_HEADER_ONLY
STRINGZILLA_API sz_cptr_t sz_sequence_tape_start(void const *handle, sz_size_t i) {
    sz_u64_t const *offsets = (sz_u64_t const *)handle;
    return (sz_cptr_t)handle + offsets[i];
}
STRINGZILLA_API sz_size_t sz_sequence_tape_length(void const *handle, sz_size_t i) {
    sz_u64_t const *offsets = (sz_u64_t const *)handle;
    return (sz_size_t)(offsets[i + 1] - offsets[i]);
}
#endif

#pragma endregion

#ifdef __cplusplus
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}
#endif // __cplusplus

#endif // STRINGZILLA_TYPES_H_
