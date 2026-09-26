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
 *  - @c sz_memory_allocator_t - a wrapper for memory-management functions.
 *  - @c sz_sequence_t - a wrapper to access strings forming a sequential container.
 *  - @c sz_byteset_t - a bitset for 256 possible byte values.
 */
#if !defined(STRINGZILLA_TYPES_H_)
#define STRINGZILLA_TYPES_H_

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

/** Removes compile-time dispatching, and replaces it with runtime dispatching. So @c sz_find will
 *  invoke the most advanced backend supported by the CPU that runs the program, rather than the one
 *  supported by the CPU used to compile the library or the downstream application. */
#if !defined(STRINGZILLA_RUNTIME_DISPATCH)
#define STRINGZILLA_RUNTIME_DISPATCH (0) // true or false
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
 *  @note Do not use @c STRINGZILLA_ARCH_X86_64_ or @c STRINGZILLA_ARCH_ARM64_ here — those
 *      indicate the CPU family, not pointer width. Rely on compiler/OS macros only.
 */
#if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_WIN64) || defined(__aarch64__) || \
    defined(__arm64__) || defined(__arm64) || defined(_M_ARM64)
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
#if !defined(STRINGZILLA_ARCH_X86_64_)
#if defined(__x86_64__) || defined(_M_X64)
#define STRINGZILLA_ARCH_X86_64_ (1)
#else
#define STRINGZILLA_ARCH_X86_64_ (0)
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
 *  STRINGZILLA_API_COMPTIME     public API, ISA tier resolved at compile time, header-inline
 *  STRINGZILLA_API_RUNTIME      public API dispatched at runtime, the only role with cross-TU linkage
 *  STRINGZILLA_HELPER_AUTO      internal helper, compiler decides inlining, expands like STRINGZILLA_API_COMPTIME
 *  STRINGZILLA_HELPER_INLINE    internal helper forced inline, structural: devirtualizing driver loops
 *  STRINGZILLA_HELPER_NOINLINE  internal helper forced out-of-line
 *  @endverbatim
 *
 *  The last one omits @c inline deliberately, as GCC ignores @c noinline on an @c inline function.
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
#define STRINGZILLA_HELPER_INLINE __forceinline static
#define STRINGZILLA_HELPER_NOINLINE __declspec(noinline) static
#else
#define STRINGZILLA_HELPER_INLINE __attribute__((always_inline)) STRINGZILLA_C_INLINE_
#define STRINGZILLA_HELPER_NOINLINE static __attribute__((noinline))
#endif

#define STRINGZILLA_API_COMPTIME STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_

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
 *  @c STRINGZILLA_HELPER_INLINE instead, which is why no translation unit needs a @c -Wno- flag to
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
#define STRINGZILLA_HELPER_AUTO STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_ constexpr
#else
#define STRINGZILLA_HELPER_AUTO STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_
#endif

#if !defined(STRINGZILLA_EXPORT_)
#define STRINGZILLA_EXPORT_ (0)
#endif

/** Exported symbol under dynamic dispatch or @c STRINGZILLA_EXPORT_ (emitted from one
 *  amalgamation TU, links like a normal C library — the Rust binding without
 *  @c dynamic-dispatch); otherwise a header-inline tier. */
#if STRINGZILLA_RUNTIME_DISPATCH || STRINGZILLA_EXPORT_
#if defined(_WIN32) || defined(__CYGWIN__)
#define STRINGZILLA_API_RUNTIME __declspec(dllexport)
#else
#define STRINGZILLA_API_RUNTIME extern __attribute__((visibility("default")))
#endif // _WIN32 || __CYGWIN__
#else
#define STRINGZILLA_API_RUNTIME STRINGZILLA_C_INLINE_
#endif // STRINGZILLA_RUNTIME_DISPATCH || STRINGZILLA_EXPORT_

/** CUDA device-side inlining policy. It is only meaningful under @c nvcc, since the functions it
 *  marks exist only on the device. */
#if defined(__CUDACC__)
#define STRINGZILLA_DEVICE_INLINE __device__ __forceinline__
#define STRINGZILLA_DEVICE_NOINLINE __device__ __noinline__
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
 *  which the RISC-V @c riscv_hwprobe capability probe in `stringzilla.h` relies on. As the first
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

/*  The toolkit version behind the NVIDIA GPU tiers below. No compiler macro can report the toolkit
 *  from a host-compiled unit - NVCC's implicit `cuda_runtime.h` reaches `.cu` files only - so the
 *  question is put to the include path itself. A build that never configured CUDA cannot see the
 *  header and pays nothing for asking. */
#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h> // `CUDART_VERSION`
#endif
#endif

/*  Compile-time hardware features detection. All of those can be controlled by the user. */
#if !defined(STRINGZILLA_TARGET_WESTMERE)
#if STRINGZILLA_ARCH_X86_64_ && defined(__SSE4_2__) && defined(__AES__)
#define STRINGZILLA_TARGET_WESTMERE (1)
#elif STRINGZILLA_ARCH_X86_64_ && defined(_MSC_VER) && defined(__AVX__)
#define STRINGZILLA_TARGET_WESTMERE (1) // ! MSVC doesn't expose `__SSE4_2__`, `__AES__` macros
#else
#define STRINGZILLA_TARGET_WESTMERE (0)
#endif
#endif

#if !defined(STRINGZILLA_TARGET_HASWELL)
#if STRINGZILLA_ARCH_X86_64_ && defined(__AVX2__)
#define STRINGZILLA_TARGET_HASWELL (1)
#else
#define STRINGZILLA_TARGET_HASWELL (0)
#endif
#endif

#if !defined(STRINGZILLA_TARGET_GOLDMONT)
#if STRINGZILLA_ARCH_X86_64_ && defined(__SHA__)
#define STRINGZILLA_TARGET_GOLDMONT (1)
#elif STRINGZILLA_ARCH_X86_64_ && defined(_MSC_VER) && defined(__AVX2__)
#define STRINGZILLA_TARGET_GOLDMONT (1) // ! MSVC doesn't expose `__SHA__` macros
#else
#define STRINGZILLA_TARGET_GOLDMONT (0)
#endif
#endif

#if !defined(STRINGZILLA_TARGET_SKYLAKE)
#if STRINGZILLA_ARCH_X86_64_ && defined(__AVX512F__)
#define STRINGZILLA_TARGET_SKYLAKE (1)
#else
#define STRINGZILLA_TARGET_SKYLAKE (0)
#endif
#endif

#if !defined(STRINGZILLA_TARGET_ICELAKE)
#if STRINGZILLA_ARCH_X86_64_ && defined(__AVX512BW__) && defined(__VAES__)
#define STRINGZILLA_TARGET_ICELAKE (1)
#elif STRINGZILLA_ARCH_X86_64_ && defined(_MSC_VER) && defined(__AVX512BW__)
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

/** SVE isn't a silver bullet: in the length-sensitive kernel families (compare, memory, find,
 *  UTF-8 tokens) the scalable kernels only outrun NEON when the registers are wider than NEON's
 *  128 bits, while the crypto-heavy families (like hashing) win on SVE2 at any width. At compile
 *  time the width is only known when pinned via @c -msve-vector-bits=N, reported as
 *  @c __ARM_FEATURE_SVE_BITS; unpinned builds assume the common 128-bit case and keep NEON.
 *  Runtime dispatch measures the actual width with @c sz_sve_wider_than_neon_ instead of trusting
 *  this compile-time assumption. */
#if defined(__ARM_FEATURE_SVE_BITS) && (__ARM_FEATURE_SVE_BITS > 128)
#define STRINGZILLA_SVE_WIDER_THAN_NEON_ (1)
#else
#define STRINGZILLA_SVE_WIDER_THAN_NEON_ (0)
#endif

/** LLVM 18 through 21 carry @c evex512 as a separate target feature, split out of AVX-512 for the
 *  AVX10 transition; ZMM codegen in a per-function @c target attribute needs it named. LLVM 17 and
 *  older never knew the token, LLVM 22 retired it again, and Clang drops the whole attribute over
 *  one unknown feature - @c -Wignored-attributes, silently costing every AVX-512 kernel - so the
 *  fork is a closed version window, not a floor. Apple Clang 17 is LLVM-19-based and sits inside
 *  it. The same window is spelled out in `probes/x86_skylake.c` and `probes/x86_icelake.c`, which
 *  stay freestanding for Cargo. */
#if defined(__clang__) && __clang_major__ < 22 && \
    (__clang_major__ >= 18 || (defined(__apple_build_version__) && __clang_major__ >= 17))
#define STRINGZILLA_HAS_CLANG_EVEX512_ (1)
#else
#define STRINGZILLA_HAS_CLANG_EVEX512_ (0)
#endif

/*  Whether a CUDA layer exists at all is a build-wide switch both build systems stamp
 *  explicitly; this fallback only serves header-only use. It keys on @c __CUDACC__ rather than
 *  @c __NVCC__ so that Clang's CUDA mode - which defines the former and not the latter - is
 *  recognized as a CUDA compilation too. */
#if !defined(STRINGZILLA_TARGET_CUDA)
#if defined(__CUDACC__)
#define STRINGZILLA_TARGET_CUDA (1)
#else
#define STRINGZILLA_TARGET_CUDA (0)
#endif
#endif

/*  Whether a ROCm layer exists at all is a build-wide switch every build system stamps explicitly,
 *  beside @c STRINGZILLA_TARGET_CUDA and never with it; this fallback only serves header-only use.
 *  @c __HIP__ alone would also catch HIP targeting NVIDIA, so the CUDA switch decides that case -
 *  and @c __HIP_PLATFORM_AMD__ cannot, being defined by `hip_common.h` rather than by the compiler,
 *  long after this header is read. */
#if !defined(STRINGZILLA_TARGET_ROCM)
#if defined(__HIP__) && !STRINGZILLA_TARGET_CUDA
#define STRINGZILLA_TARGET_ROCM (1)
#else
#define STRINGZILLA_TARGET_ROCM (0)
#endif
#endif

/*  The Kepler tier is reached through @c __shfl_sync, @c __ballot_sync and @c __popc. Every
 *  architecture a current toolkit can target has them - CUDA 13 dropped everything below SM75 - so
 *  a CUDA compilation brings the tier with it. */
#if !defined(STRINGZILLA_TARGET_KEPLER)
#if STRINGZILLA_TARGET_CUDA
#define STRINGZILLA_TARGET_KEPLER (1)
#else
#define STRINGZILLA_TARGET_KEPLER (0)
#endif
#endif

/*  The Hopper tier needs SM90 code in the shipped binary, which is a property of the
 *  architectures this build compiles for and not of the toolkit that compiles them - a CUDA 13
 *  toolkit asked for SM75 alone emits no DPX and no bulk copies. Both build systems stamp this
 *  explicitly; the device pass knows its own target, and the host pass cannot scan
 *  @c __CUDA_ARCH_LIST__ in the preprocessor, so an unstamped host pass under-reports rather than
 *  claiming code it may not have emitted. */
#if !defined(STRINGZILLA_TARGET_HOPPER)
#if STRINGZILLA_TARGET_CUDA && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define STRINGZILLA_TARGET_HOPPER (1)
#else
#define STRINGZILLA_TARGET_HOPPER (0)
#endif
#endif

/*  WebAssembly SIMD128 — opt-in at compile time via @c -msimd128; there is no runtime probe. */
#if !defined(STRINGZILLA_TARGET_V128)
#if defined(__wasm__) && defined(__wasm_simd128__)
#define STRINGZILLA_TARGET_V128 (1)
#else
#define STRINGZILLA_TARGET_V128 (0)
#endif
#endif

/*  WebAssembly @b relaxed SIMD — opt-in via @c -mrelaxed-simd; a level above baseline SIMD128
 *  adding relaxed swizzle, fused multiply-add, lane-select, and integer dot-products. Some
 *  runtimes lower a few relaxed ops sub-optimally, but the level is exposed so native engines
 *  can use them. */
#if !defined(STRINGZILLA_TARGET_V128RELAXED)
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
#if !defined(STRINGZILLA_TARGET_LASX)
#if defined(__loongarch__) && defined(__loongarch_asx)
#define STRINGZILLA_TARGET_LASX (1)
#else
#define STRINGZILLA_TARGET_LASX (0)
#endif
#endif

/*  IBM Power Vector-Scalar eXtension (VSX, Power8+) — `-mvsx`. */
#if !defined(STRINGZILLA_TARGET_POWERVSX)
#if (defined(__powerpc__) || defined(__powerpc64__)) && defined(__VSX__)
#define STRINGZILLA_TARGET_POWERVSX (1)
#else
#define STRINGZILLA_TARGET_POWERVSX (0)
#endif
#endif

/*  SIMD micro-architecture tiers are cumulative supersets - there is no CPU with AVX-512 VBMI but
 *  not AVX2, nor SVE without NEON - and higher-tier kernels call into lower-tier helpers (the Ice
 *  Lake hash reuses the Westmere routines, the Ice Lake intersect kernel uses the 128/256-bit
 *  register unions). Enabling a tier therefore requires its whole substrate. Close the set
 *  downward so an isolated `-D STRINGZILLA_TARGET_ICELAKE=1`, or a hand-edited development
 *  override, can never name a tier without the ones it is built on. This is the single source of
 *  truth for the nesting; everything downstream may assume a lower tier is on whenever a higher
 *  one is. Goldmont (SHA) and the NEON/SVE crypto extensions are orthogonal feature flags, not
 *  part of the nesting. */
#if STRINGZILLA_TARGET_ICELAKE && !STRINGZILLA_TARGET_SKYLAKE
#undef STRINGZILLA_TARGET_SKYLAKE
#define STRINGZILLA_TARGET_SKYLAKE (1)
#endif
#if STRINGZILLA_TARGET_SKYLAKE && !STRINGZILLA_TARGET_HASWELL
#undef STRINGZILLA_TARGET_HASWELL
#define STRINGZILLA_TARGET_HASWELL (1)
#endif
#if STRINGZILLA_TARGET_HASWELL && !STRINGZILLA_TARGET_WESTMERE
#undef STRINGZILLA_TARGET_WESTMERE
#define STRINGZILLA_TARGET_WESTMERE (1)
#endif
#if STRINGZILLA_TARGET_SVE2 && !STRINGZILLA_TARGET_SVE
#undef STRINGZILLA_TARGET_SVE
#define STRINGZILLA_TARGET_SVE (1)
#endif
#if STRINGZILLA_TARGET_SVE && !STRINGZILLA_TARGET_NEON
#undef STRINGZILLA_TARGET_NEON
#define STRINGZILLA_TARGET_NEON (1)
#endif
/*  WebAssembly relaxed-SIMD is a tier above baseline SIMD128 (it requires @c simd128). Close the
 *  set downward so backends that ship only a @c _v128 kernel can dispatch on
 *  `#if STRINGZILLA_TARGET_V128` alone and still be reached when the build enabled relaxed-SIMD. */
#if STRINGZILLA_TARGET_V128RELAXED && !STRINGZILLA_TARGET_V128
#undef STRINGZILLA_TARGET_V128
#define STRINGZILLA_TARGET_V128 (1)
#endif

/*  Hardware-specific headers for different SIMD intrinsics and register wrappers. */
#if STRINGZILLA_TARGET_V128
#include <wasm_simd128.h>
#endif // STRINGZILLA_TARGET_V128
#if STRINGZILLA_TARGET_RVV
#include <riscv_vector.h>
#endif // STRINGZILLA_TARGET_RVV
#if STRINGZILLA_TARGET_LASX
#include <lasxintrin.h> // 256-bit `__lasx_*` intrinsics and the `__m256i` register type
#include <lsxintrin.h>  // 128-bit `__lsx_*` intrinsics and the `__m128i` register type, for sub-32-byte inputs
#endif                  // STRINGZILLA_TARGET_LASX
#if STRINGZILLA_TARGET_POWERVSX
#include <altivec.h>
#endif // STRINGZILLA_TARGET_POWERVSX
#if STRINGZILLA_TARGET_WESTMERE || STRINGZILLA_TARGET_HASWELL || STRINGZILLA_TARGET_SKYLAKE || \
    STRINGZILLA_TARGET_ICELAKE
#include <immintrin.h>
#endif // STRINGZILLA_TARGET_WESTMERE || STRINGZILLA_TARGET_HASWELL || STRINGZILLA_TARGET_SKYLAKE || STRINGZILLA_TARGET_ICELAKE
#if STRINGZILLA_TARGET_NEON
#if !defined(_MSC_VER)
#include <arm_acle.h>
#endif
#include <arm_neon.h>
#endif // STRINGZILLA_TARGET_NEON
#if STRINGZILLA_TARGET_SVE || STRINGZILLA_TARGET_SVE2
#if !defined(_MSC_VER)
#include <arm_sve.h>
#endif
#endif // STRINGZILLA_TARGET_SVE || STRINGZILLA_TARGET_SVE2

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

/** A simple signed integer type describing the status of a faulty operation. */
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

    /** A sink-hole status for unknown errors. */
    sz_status_unknown_k = -1,
} sz_status_t;

/**
 *  @brief Enumeration of SIMD capabilities of the target architecture, used to introspect the
 *      supported functionality of the dynamic library.
 *
 *  Each single capability is a bit named `sz_cap_<name>_k`:
 *
 *  @verbatim
 *  serial          serial, non-SIMD code
 *  parallel        multi-threading via Fork Union or other OpenMP-like engines
 *  any             mask representing any capability, equal to INT_MAX
 *
 *  goldmont        x86 SHA-NI for accelerated SHA-256 hashing
 *  westmere        x86 SSE4.2 and AES-NI
 *  haswell         x86 AVX2 with FMA and F16C extensions
 *  skylake         x86 AVX-512 baseline
 *  icelake         x86 AVX-512 with advanced integer algorithms and AES extensions
 *
 *  neon            Arm NEON baseline
 *  neonaes         Arm NEON with AES extensions
 *  neonsha         Arm NEON with SHA2 extensions
 *  sve             Arm SVE baseline
 *  sve2            Arm SVE2
 *  sve2aes         Arm SVE2 with AES extensions
 *
 *  v128            WebAssembly SIMD128
 *  v128relaxed     WebAssembly relaxed-SIMD, above SIMD128
 *  lasx            LoongArch LASX, 256-bit
 *  powervsx        IBM Power VSX
 *  rvv             RISC-V Vector, RVV 1.0
 *  rvvcrypto       RISC-V Vector Crypto, Zvk: Zvkned AES and Zvknhb SHA
 *
 *  cuda            CUDA
 *  kepler          CUDA with in-warp register shuffles
 *  hopper          CUDA with Hopper's DPX instructions
 *  @endverbatim
 *
 *  Each combination is named `sz_caps_<name>_k`:
 *
 *  @verbatim
 *  none            no capabilities
 *  sp              serial code with Fork Union
 *  sh              serial code with Haswell
 *  sn              serial code with NEON
 *  sr              serial code with RISC-V Vector
 *  sil             serial code with Ice Lake
 *  spil            serial code with Fork Union and Ice Lake
 *  sps             serial code with Fork Union and SVE
 *  ck              CUDA code with Kepler
 *  ckh             CUDA code with Kepler and Hopper
 *  cpus            aggregate for the CPU StringZillas builds
 *  cuda            aggregate for the CUDA StringZillas builds
 *  @endverbatim
 */
typedef enum sz_capability_t {
    sz_cap_serial_k = 1,
    sz_cap_parallel_k = 1 << 2,
    sz_cap_any_k = 0x7FFFFFFF,

    sz_cap_goldmont_k = 1 << 3,
    sz_cap_westmere_k = 1 << 4,
    sz_cap_haswell_k = 1 << 5,
    sz_cap_skylake_k = 1 << 6,
    sz_cap_icelake_k = 1 << 7,

    sz_cap_neon_k = 1 << 10,
    sz_cap_neonaes_k = 1 << 11,
    sz_cap_neonsha_k = 1 << 15,
    sz_cap_sve_k = 1 << 12,
    sz_cap_sve2_k = 1 << 13,
    sz_cap_sve2aes_k = 1 << 14,

    sz_cap_v128_k = 1 << 16,
    sz_cap_v128relaxed_k = 1 << 17,
    sz_cap_lasx_k = 1 << 18,
    sz_cap_powervsx_k = 1 << 19,
    sz_cap_rvv_k = 1 << 20,

    sz_cap_cuda_k = 1 << 21,
    sz_cap_kepler_k = 1 << 22,
    sz_cap_hopper_k = 1 << 23,

    sz_cap_rvvcrypto_k = 1 << 24,

    sz_caps_none_k = 0,

    sz_caps_sp_k = sz_cap_serial_k | sz_cap_parallel_k,
    sz_caps_sh_k = sz_cap_serial_k | sz_cap_haswell_k,
    sz_caps_sn_k = sz_cap_serial_k | sz_cap_neon_k,
    sz_caps_sr_k = sz_cap_serial_k | sz_cap_rvv_k,
    sz_caps_sil_k = sz_cap_serial_k | sz_cap_icelake_k,

    sz_caps_spil_k = sz_cap_serial_k | sz_cap_parallel_k | sz_cap_icelake_k,
    sz_caps_sps_k = sz_cap_serial_k | sz_cap_parallel_k | sz_cap_sve_k,
    sz_caps_ck_k = sz_cap_cuda_k | sz_cap_kepler_k,
    sz_caps_ckh_k = sz_cap_cuda_k | sz_cap_kepler_k | sz_cap_hopper_k,

    sz_caps_cpus_k = sz_cap_serial_k | sz_cap_parallel_k | sz_cap_haswell_k | sz_cap_skylake_k | sz_cap_icelake_k |
                     sz_cap_westmere_k | sz_cap_goldmont_k | sz_cap_neon_k | sz_cap_neonaes_k | sz_cap_neonsha_k |
                     sz_cap_sve_k | sz_cap_sve2_k | sz_cap_sve2aes_k | sz_cap_v128_k | sz_cap_v128relaxed_k |
                     sz_cap_rvv_k | sz_cap_rvvcrypto_k | sz_cap_lasx_k | sz_cap_powervsx_k,
    sz_caps_cuda_k = sz_cap_cuda_k | sz_cap_kepler_k | sz_cap_hopper_k,
} sz_capability_t;

/**
 *  @brief Maximum number of individual capability flags that can be represented.
 *  @sa sz_capabilities_to_strings_implementation_, internal, but a valid example.
 */
#define STRINGZILLA_CAPABILITIES_COUNT 22

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

STRINGZILLA_API_COMPTIME sz_rune_t sz_rune_perfect_hash(sz_rune_t rune) {
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
STRINGZILLA_API_COMPTIME void sz_byteset_init(sz_byteset_t *s) {
    s->_u64s[0] = s->_u64s[1] = s->_u64s[2] = s->_u64s[3] = 0;
}

/** Initializes a bit-set to all ASCII characters. */
STRINGZILLA_API_COMPTIME void sz_byteset_init_ascii(sz_byteset_t *s) {
    s->_u64s[0] = s->_u64s[1] = 0xFFFFFFFFFFFFFFFFull;
    s->_u64s[2] = s->_u64s[3] = 0;
}

/** Adds a character to the set and accepts @b unsigned integers. */
STRINGZILLA_API_COMPTIME void sz_byteset_add_u8(sz_byteset_t *s, sz_u8_t c) { s->_u64s[c >> 6] |= (1ull << (c & 63u)); }

/** Adds a character to the set. Consider @b sz_byteset_add_u8. */
STRINGZILLA_API_COMPTIME void sz_byteset_add(sz_byteset_t *s, char c) {
    sz_byteset_add_u8(s, *(sz_u8_t *)(&c));
} // bitcast

/** Checks if the set contains a given character and accepts @b unsigned integers. */
STRINGZILLA_API_COMPTIME sz_bool_t sz_byteset_contains_u8(sz_byteset_t const *s, sz_u8_t c) {
    // Checking the bit can be done in different ways:
    // - (s->_u64s[c >> 6] & (1ull << (c & 63u))) != 0
    // - (s->_u32s[c >> 5] & (1u << (c & 31u))) != 0
    // - (s->_u16s[c >> 4] & (1u << (c & 15u))) != 0
    // - (s->_u8s[c >> 3] & (1u << (c & 7u))) != 0
    return (sz_bool_t)((s->_u64s[c >> 6] & (1ull << (c & 63u))) != 0);
}

/** Checks if the set contains a given character. Consider @b sz_byteset_contains_u8. */
STRINGZILLA_API_COMPTIME sz_bool_t sz_byteset_contains(sz_byteset_t const *s, char c) {
    return sz_byteset_contains_u8(s, *(sz_u8_t *)(&c)); // bitcast
}

/** Inverts the contents of the set, so allowed characters get disallowed, and vice versa. */
STRINGZILLA_API_COMPTIME void sz_byteset_invert(sz_byteset_t *s) {
    s->_u64s[0] ^= 0xFFFFFFFFFFFFFFFFull, s->_u64s[1] ^= 0xFFFFFFFFFFFFFFFFull, //
        s->_u64s[2] ^= 0xFFFFFFFFFFFFFFFFull, s->_u64s[3] ^= 0xFFFFFFFFFFFFFFFFull;
}

#pragma endregion

#pragma region Memory Management

typedef void *(*sz_memory_allocate_t)(sz_size_t, void *);
typedef void (*sz_memory_free_t)(void *, sz_size_t, void *);

/**
 *  @brief Some complex pattern matching algorithms may require memory allocations. This structure
 *      passes the memory allocator to those functions.
 *  @sa sz_memory_allocator_init_fixed
 */
typedef struct sz_memory_allocator_t {
    sz_memory_allocate_t allocate;
    sz_memory_free_t free;
    void *handle;
} sz_memory_allocator_t;

/**
 *  @brief Initializes a memory allocator to use the system default @c malloc and @c free.
 *  @param[out] allocator Memory allocator to initialize.
 *  @warning The function is not available if the library was compiled with
 *      @c STRINGZILLA_WITH_LIBC=0.
 *  @note Unlike the C standard library, `malloc(0)` is guaranteed to return a non-null pointer.
 *  @see malloc: https://en.cppreference.com/w/c/memory/malloc
 */
STRINGZILLA_API_COMPTIME void sz_memory_allocator_init_default(sz_memory_allocator_t *allocator);

/**
 *  @brief Initializes a memory allocator that serves every request from a static-capacity buffer,
 *      @b without any dynamic allocations.
 *  @param[out] allocator Memory allocator to initialize.
 *  @param[in] buffer Buffer to use for allocations.
 *  @param[in] length Length of the buffer. @b Must be greater than 16, at least 4KB (one RAM
 *      page) is recommended.
 *
 *  The @p buffer itself will be prepended with the capacity and the consumed size. Those values
 *  shouldn't be modified.
 */
STRINGZILLA_API_COMPTIME void sz_memory_allocator_init_fixed(sz_memory_allocator_t *allocator, void *buffer,
                                                             sz_size_t length);

/**
 *  @brief Checks if two memory allocators are equivalent.
 *  @param[in] a First memory allocator.
 *  @param[in] b Second memory allocator.
 *  @return True if the allocators are the same, false otherwise.
 */
STRINGZILLA_API_COMPTIME sz_bool_t sz_memory_allocator_equal(sz_memory_allocator_t const *a,
                                                             sz_memory_allocator_t const *b);

#pragma endregion

#pragma region API Signature Types

/** Signature of @c sz_hash. */
typedef sz_u64_t (*sz_hash_t)(sz_cptr_t, sz_size_t, sz_u64_t);

/** Signature of @c sz_hash_multiseed. */
typedef void (*sz_hash_multiseed_t)(sz_cptr_t, sz_size_t, sz_u64_t const *, sz_size_t, sz_u64_t *);

/** Signature of @c sz_hash_state_init. */
typedef void (*sz_hash_state_init_t)(struct sz_hash_state_t *, sz_u64_t);

/** Signature of @c sz_hash_state_update (legacy) / @c sz_hash_state_update (preferred). */
typedef void (*sz_hash_state_update_t)(struct sz_hash_state_t *, sz_cptr_t, sz_size_t);

/** Signature of @c sz_hash_state_digest (legacy) / @c sz_hash_state_digest (preferred). */
typedef sz_u64_t (*sz_hash_state_digest_t)(struct sz_hash_state_t const *);

/** Signature of @c sz_bytesum. */
typedef sz_u64_t (*sz_bytesum_t)(sz_cptr_t, sz_size_t);

/** Signature of @c sz_utf8_count. */
typedef sz_size_t (*sz_utf8_count_t)(sz_cptr_t, sz_size_t);

/** Signature of @c sz_utf8_seek. */
typedef sz_cptr_t (*sz_utf8_seek_t)(sz_cptr_t, sz_size_t, sz_size_t);

/** Signature of @c sz_utf8_decode. */
typedef sz_cptr_t (*sz_utf8_decode_t)(sz_cptr_t, sz_size_t, sz_rune_t *, sz_size_t, sz_size_t *);

/** Signature of @c sz_utf8_uncased_fold. */
typedef sz_size_t (*sz_utf8_uncased_fold_t)(sz_cptr_t, sz_size_t, sz_ptr_t);

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

/** Signature of @c sz_utf8_norm (single-pass normalizer). */
typedef sz_size_t (*sz_utf8_norm_t)(sz_cptr_t, sz_size_t, sz_normal_form_t, sz_ptr_t);

/** Signature of @c sz_utf8_find_denormalized. */
typedef sz_cptr_t (*sz_utf8_find_denormalized_t)(sz_cptr_t, sz_size_t, sz_normal_form_t);

/** Forward declaration for uncased needle metadata. */
struct sz_utf8_uncased_needle_metadata_t;

/** Signature of @c sz_utf8_uncased_search. */
typedef sz_cptr_t (*sz_utf8_uncased_search_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t,
                                              struct sz_utf8_uncased_needle_metadata_t *, sz_size_t *);

/** Signature of @c sz_utf8_uncased_order. */
typedef sz_ordering_t (*sz_utf8_uncased_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/** Signature of @c sz_utf8_find_cased. */
typedef sz_cptr_t (*sz_utf8_find_cased_t)(sz_cptr_t, sz_size_t);

/** Signature of every UTF-8 "find boundaries" kernel - words (forward/reverse), graphemes,
 *  sentences, lines, newlines, whitespace, delimiters. Emits parallel (offset, length) arrays for
 *  each segment/delimiter plus a resume @c bytes_consumed. */
typedef sz_size_t (*sz_utf8_segmenter_t)(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t *, sz_size_t, sz_size_t *);

/** Signature of @c sz_fill_random. */
typedef void (*sz_fill_random_t)(sz_ptr_t, sz_size_t, sz_u64_t);

/** Signature of @c sz_sha256_state_init. */
typedef void (*sz_sha256_state_init_t)(struct sz_sha256_state_t *);

/** Signature of @c sz_sha256_state_update. */
typedef void (*sz_sha256_state_update_t)(struct sz_sha256_state_t *, sz_cptr_t, sz_size_t);

/** Signature of @c sz_sha256_state_digest. */
typedef void (*sz_sha256_state_digest_t)(struct sz_sha256_state_t const *, sz_u8_t *);

/** Signature of @c sz_sha256_multistate_update. */
typedef void (*sz_sha256_multistate_update_t)(struct sz_sha256_state_t *, struct sz_sequence_t const *);

/** Signature of @c sz_sha256_multistate_digest. */
typedef void (*sz_sha256_multistate_digest_t)(struct sz_sha256_state_t const *, sz_size_t, sz_u8_t *);

/** Signature of @c sz_aes256_key_init. */
typedef void (*sz_aes256_key_init_t)(struct sz_aes256_key_t *, sz_u8_t const *);

/** Signature of @c sz_aes256_gcm_key_init. */
typedef void (*sz_aes256_gcm_key_init_t)(struct sz_aes256_gcm_key_t *, sz_u8_t const *);

/** Signature of @c sz_aes256_ctr_xor. */
typedef void (*sz_aes256_ctr_xor_t)(struct sz_aes256_key_t const *, sz_u8_t const *, sz_u64_t, sz_cptr_t, sz_size_t,
                                    sz_ptr_t);

/** Signature of @c sz_aes256_gcm_encrypt. */
typedef void (*sz_aes256_gcm_encrypt_t)(struct sz_aes256_gcm_key_t const *, sz_u8_t const *, sz_cptr_t, sz_size_t,
                                        sz_cptr_t, sz_size_t, sz_ptr_t, sz_u8_t *);

/** Signature of @c sz_aes256_gcm_decrypt. */
typedef sz_status_t (*sz_aes256_gcm_decrypt_t)(struct sz_aes256_gcm_key_t const *, sz_u8_t const *, sz_cptr_t,
                                               sz_size_t, sz_cptr_t, sz_size_t, sz_ptr_t, sz_u8_t const *);

/** Signature of @c sz_aes256_gcm_encryptor_init. */
typedef void (*sz_aes256_gcm_encryptor_init_t)(struct sz_aes256_gcm_encryptor_t *, struct sz_aes256_gcm_key_t const *,
                                               sz_u8_t const *);

/** Signature of @c sz_aes256_gcm_encryptor_associate. */
typedef void (*sz_aes256_gcm_encryptor_associate_t)(struct sz_aes256_gcm_encryptor_t *, sz_cptr_t, sz_size_t);

/** Signature of @c sz_aes256_gcm_encryptor_update. */
typedef void (*sz_aes256_gcm_encryptor_update_t)(struct sz_aes256_gcm_encryptor_t *, sz_cptr_t, sz_size_t, sz_ptr_t);

/** Signature of @c sz_aes256_gcm_encryptor_digest. */
typedef void (*sz_aes256_gcm_encryptor_digest_t)(struct sz_aes256_gcm_encryptor_t const *, sz_u8_t *);

/** Signature of @c sz_aes256_gcm_decryptor_init. */
typedef void (*sz_aes256_gcm_decryptor_init_t)(struct sz_aes256_gcm_decryptor_t *, struct sz_aes256_gcm_key_t const *,
                                               sz_u8_t const *);

/** Signature of @c sz_aes256_gcm_decryptor_associate. */
typedef void (*sz_aes256_gcm_decryptor_associate_t)(struct sz_aes256_gcm_decryptor_t *, sz_cptr_t, sz_size_t);

/** Signature of @c sz_aes256_gcm_decryptor_update_unverified. */
typedef void (*sz_aes256_gcm_decryptor_update_unverified_t)(struct sz_aes256_gcm_decryptor_t *, sz_cptr_t, sz_size_t,
                                                            sz_ptr_t);

/** Signature of @c sz_aes256_gcm_decryptor_verify. */
typedef sz_status_t (*sz_aes256_gcm_decryptor_verify_t)(struct sz_aes256_gcm_decryptor_t const *, sz_u8_t const *);

/** Signature of @c sz_equal. */
typedef sz_bool_t (*sz_equal_t)(sz_cptr_t, sz_cptr_t, sz_size_t);

/** Signature of @c sz_order. */
typedef sz_ordering_t (*sz_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/** Signature of @c sz_lookup. */
typedef void (*sz_lookup_t)(sz_ptr_t, sz_size_t, sz_cptr_t, sz_cptr_t);

/** Signature of @c sz_copy. */
typedef void (*sz_copy_t)(sz_ptr_t, sz_cptr_t, sz_size_t);

/** Signature of @c sz_move. */
typedef void (*sz_move_t)(sz_ptr_t, sz_cptr_t, sz_size_t);

/** Signature of @c sz_fill. */
typedef void (*sz_fill_t)(sz_ptr_t, sz_size_t, sz_u8_t);

/** Signature of @c sz_find_byte. */
typedef sz_cptr_t (*sz_find_byte_t)(sz_cptr_t, sz_size_t, sz_cptr_t);

/** Signature of @c sz_find. */
typedef sz_cptr_t (*sz_find_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t);

/** Signature of @c sz_find_byteset. */
typedef sz_cptr_t (*sz_find_byteset_t)(sz_cptr_t, sz_size_t, sz_byteset_t const *);

/** Signature of @c sz_sequence_argsort and @c sz_sequence_argsort_uncased. */
typedef sz_status_t (*sz_sequence_argsort_t)(struct sz_sequence_t const *, sz_memory_allocator_t *, sz_sorted_idx_t *,
                                             sz_size_t, sz_bool_t);

/** Signature of the internal @c sz_pgrams_sort_serial, @c _skylake, and @c _sve integer sorts. */
typedef sz_status_t (*sz_pgrams_sort_t)(sz_pgram_t *, sz_size_t, sz_memory_allocator_t *, sz_sorted_idx_t *);

/** Signature of @c sz_sequence_intersect. */
typedef sz_status_t (*sz_sequence_intersect_t)(struct sz_sequence_t const *, struct sz_sequence_t const *,
                                               sz_memory_allocator_t *, sz_u64_t, sz_size_t *, sz_sorted_idx_t *,
                                               sz_sorted_idx_t *);

/** Which symbols a batch counts, as the alphabet picks the transpose and the mask layout alike. */
typedef enum sz_levenshtein_symbol_t {

    /** Every byte is its own symbol, and a distance counts bytes. */
    sz_levenshtein_bytes_k = 0,

    /** Every UTF-8 rune is one symbol, an ill-formed byte decoding to U+FFFD. */
    sz_levenshtein_runes_k = 1,
} sz_levenshtein_symbol_t;

/** Signature of @c sz_levenshtein_distances, at either alphabet. */
typedef sz_status_t (*sz_levenshtein_distances_t)(struct sz_levenshtein_engine_t *, struct sz_sequence_t const *,
                                                  sz_size_t *, sz_size_t);

/** Signature of @c sz_overlap_scores. */
typedef sz_status_t (*sz_overlap_scores_t)(struct sz_overlap_engine_t *, struct sz_sequence_t const *, sz_f32_t *,
                                           sz_size_t, sz_size_t);

/** How matches that share bytes resolve: reported in full, or thinned to a leftmost run. */
typedef enum sz_substrings_overlap_policy_t {

    /** Every match of every needle, including ones that share bytes and ones nested in others. */
    sz_substrings_overlapping_k = 0,

    /** Matches sharing no bytes: earliest start, then longest span, then lower needle index. */
    sz_substrings_leftmost_longest_k = 1,

    /** Matches sharing no bytes: earliest start, then lower needle index, whatever the lengths. */
    sz_substrings_leftmost_first_k = 2,
} sz_substrings_overlap_policy_t;

/** Signature of @c sz_substrings_counts. */
typedef sz_status_t (*sz_substrings_counts_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                              sz_size_t *, sz_size_t);

/** Signature of @c sz_substrings_find. */
typedef sz_status_t (*sz_substrings_find_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                            struct sz_substrings_match_t *, sz_size_t, sz_size_t *);

/** Signature of @c sz_substrings_replace. */
typedef sz_status_t (*sz_substrings_replace_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                               struct sz_sequence_t const *, sz_ptr_t, sz_size_t, sz_size_t *);

/** Signature of @c sz_substrings_bm25_scores. */
typedef sz_status_t (*sz_substrings_bm25_scores_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                                   sz_f32_t const *, struct sz_substrings_bm25_t const *,
                                                   sz_f32_t const *, sz_f32_t *, sz_size_t);

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
#if STRINGZILLA_TARGET_WESTMERE
    __m128i xmm;
    __m128d xmm_pd;
    __m128 xmm_ps;
#endif
#if STRINGZILLA_TARGET_NEON
    uint8x16_t u8x16;
    uint16x8_t u16x8;
    uint32x4_t u32x4;
    uint64x2_t u64x2;
    float64x2_t f64x2;
    float32x4_t f32x4;
#endif
#if STRINGZILLA_TARGET_LASX
    __m128i lsx;
#endif
#if STRINGZILLA_TARGET_V128
    v128_t v128;
#endif
#if STRINGZILLA_TARGET_POWERVSX
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
#if STRINGZILLA_TARGET_HASWELL
    __m256i ymm;
    __m256d ymm_pd;
    __m256 ymm_ps;
#endif
#if STRINGZILLA_TARGET_WESTMERE
    __m128i xmms[2];
#endif
#if STRINGZILLA_TARGET_NEON
    uint8x16_t u8x16s[2];
    uint16x8_t u16x8s[2];
    uint32x4_t u32x4s[2];
    uint64x2_t u64x2s[2];
#endif
#if STRINGZILLA_TARGET_LASX
    __m256i lasx;
#endif
#if STRINGZILLA_TARGET_V128
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
#if STRINGZILLA_TARGET_SKYLAKE
    __m512i zmm;
    __m512d zmm_pd;
    __m512 zmm_ps;
#endif
#if STRINGZILLA_TARGET_HASWELL
    __m256i ymms[2];
#endif
#if STRINGZILLA_TARGET_WESTMERE
    __m128i xmms[4];
#endif
#if STRINGZILLA_TARGET_NEON
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
STRINGZILLA_API_COMPTIME void sz_sequence_from_null_terminated_strings(sz_cptr_t *start, sz_size_t count,
                                                                       sz_sequence_t *sequence);

/**
 *  @brief Initiates the sequence structure from an array of pointer-length pairs, like
 *      `sz_string_view_t[]`.
 *  @param[in] views Pointer to the array of views, which must outlive @p sequence.
 *  @param[in] count Number of views in the array.
 *  @param[out] sequence Sequence structure to initialize.
 */
STRINGZILLA_API_COMPTIME void sz_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
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

/** The one cache-line width the library assumes, in bytes: the stride of equality checks and
 *  relative-order heuristics, and the smallest heap buffer a growing string asks for. Derived from
 *  the target; define it from outside to override. */
#if !defined(STRINGZILLA_CACHE_LINE_BYTES)
#if defined(__s390x__)
#define STRINGZILLA_CACHE_LINE_BYTES (256) // bytes
#elif defined(__APPLE__) && defined(__aarch64__)
#define STRINGZILLA_CACHE_LINE_BYTES (128) // bytes - Apple's cores carry a whole 128-byte line
#else
#define STRINGZILLA_CACHE_LINE_BYTES (64) // bytes
#endif
#endif

#define STRINGZILLA_SIZE_MAX ((sz_size_t)(-1))
#define STRINGZILLA_SSIZE_MAX ((sz_ssize_t)(STRINGZILLA_SIZE_MAX >> 1))
#define STRINGZILLA_SSIZE_MIN ((sz_ssize_t)(-STRINGZILLA_SSIZE_MAX - 1))

STRINGZILLA_HELPER_AUTO sz_size_t sz_size_max_(void) { return STRINGZILLA_SIZE_MAX; }
STRINGZILLA_HELPER_AUTO sz_ssize_t sz_ssize_max_(void) { return STRINGZILLA_SSIZE_MAX; }

/**
 *  @brief Similar to @c assert, the @c sz_assert_ checks library invariants in @c STRINGZILLA_DEBUG
 *      builds, aborting on failure; in release it type-checks the condition without evaluating it.
 *  @note If you want to catch it, put a breakpoint at @c abort.
 */
#if STRINGZILLA_DEBUG && defined(__CUDA_ARCH__) // ? CUDA code for GPUs
STRINGZILLA_DEVICE_NOINLINE void sz_assert_cuda_failure_(char const *condition, char const *file, int line) {
    printf("Assertion failed: %s, in file %s, line %d\n", condition, file, line);
    __trap();
}
#define sz_assert_(condition)                                                          \
    do {                                                                               \
        if (!(condition)) { sz_assert_cuda_failure_(#condition, __FILE__, __LINE__); } \
    } while (0)
#elif STRINGZILLA_DEBUG && STRINGZILLA_WITH_LIBC // ? CPU code with LibC, PIC included
STRINGZILLA_API_COMPTIME void sz_assert_failure_(char const *condition, char const *file, int line) {
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
 *  @brief Whether one batch from a UTF-8 decoder or segmenter keeps the contract its resuming
 *      callers loop on.
 *  @param[in] length Bytes offered to the call.
 *  @param[in] capacity Entries the caller had room for.
 *  @param[in] count Entries the call reported.
 *  @param[in] consumed Bytes the call covered, where its caller resumes.
 *  @param[in] starts Span offsets, or @c STRINGZILLA_NULL for the decoder, which reports
 *      runes, not spans.
 *  @param[in] lengths Span lengths, read only alongside @p starts.
 *  @param[in] resumable_tail Bytes a call may leave unconsumed without progress: a truncated
 *      sequence for the decoder, none for a segmenter.
 *  @param[in] tiling Whether the spans cover the text from its first byte, with no gap and no empty
 *      span, as grapheme, word, sentence and line segments do and the separators of a split do not.
 *
 *  The batch fits @p capacity and never runs past @p length, its spans ascend inside the consumed
 *  prefix, and it makes progress whenever it had room for an entry and more than @p resumable_tail
 *  bytes to cover, because a caller resuming from @p consumed would otherwise loop forever.
 */
STRINGZILLA_HELPER_AUTO sz_bool_t sz_utf8_batch_consistent_(sz_size_t length, sz_size_t capacity, sz_size_t count,
                                                            sz_size_t consumed, sz_size_t const *starts,
                                                            sz_size_t const *lengths, sz_size_t resumable_tail,
                                                            sz_bool_t tiling) {
    if (count > capacity || consumed > length) return sz_false_k;
    if (consumed == 0 && capacity != 0 && length > resumable_tail) return sz_false_k;
    for (sz_size_t index = 0; starts && index != count; ++index) {
        sz_size_t const earliest = index == 0 ? 0 : starts[index - 1] + lengths[index - 1];
        if (starts[index] < earliest || starts[index] > consumed || lengths[index] > consumed - starts[index])
            return sz_false_k;
        if (tiling && (starts[index] != earliest || lengths[index] == 0)) return sz_false_k;
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
STRINGZILLA_HELPER_AUTO int sz_u64_ctz(sz_u64_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
STRINGZILLA_HELPER_AUTO int sz_u64_clz(sz_u64_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 0x8000000000000000ull) == 0) { n++, x <<= 1; }
    return n;
}
STRINGZILLA_HELPER_AUTO int sz_u64_popcount(sz_u64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    return (((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full) * 0x0101010101010101ull) >> 56;
}
STRINGZILLA_HELPER_AUTO int sz_u32_ctz(sz_u32_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 1) == 0) { n++, x >>= 1; }
    return n;
}
STRINGZILLA_HELPER_AUTO int sz_u32_clz(sz_u32_t x) {
    sz_assert_(x != 0);
    int n = 0;
    while ((x & 0x80000000u) == 0) { n++, x <<= 1; }
    return n;
}
STRINGZILLA_HELPER_AUTO int sz_u32_popcount(sz_u32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
#else
STRINGZILLA_HELPER_INLINE int sz_u64_ctz(sz_u64_t x) { return (int)_tzcnt_u64(x); }
STRINGZILLA_HELPER_INLINE int sz_u64_clz(sz_u64_t x) { return (int)_lzcnt_u64(x); }
STRINGZILLA_HELPER_INLINE int sz_u64_popcount(sz_u64_t x) { return (int)__popcnt64(x); }
STRINGZILLA_HELPER_INLINE int sz_u32_ctz(sz_u32_t x) { return (int)_tzcnt_u32(x); }
STRINGZILLA_HELPER_INLINE int sz_u32_clz(sz_u32_t x) { return (int)_lzcnt_u32(x); }
STRINGZILLA_HELPER_INLINE int sz_u32_popcount(sz_u32_t x) { return (int)__popcnt(x); }
#endif
/*  Force the byteswap functions to be intrinsics, because when @c /Oi- is given, these will turn
 *  into CRT function calls, which breaks when @c STRINGZILLA_WITH_LIBC is 0. */
#pragma intrinsic(_byteswap_uint64)
STRINGZILLA_HELPER_INLINE sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return _byteswap_uint64(val); }
#pragma intrinsic(_byteswap_ulong)
STRINGZILLA_HELPER_INLINE sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return _byteswap_ulong(val); }
#else
STRINGZILLA_HELPER_AUTO int sz_u64_popcount(sz_u64_t x) { return __builtin_popcountll(x); }
STRINGZILLA_HELPER_AUTO int sz_u32_popcount(sz_u32_t x) { return __builtin_popcount(x); }
STRINGZILLA_HELPER_AUTO int sz_u64_ctz(sz_u64_t x) { return __builtin_ctzll(x); }
STRINGZILLA_HELPER_AUTO int sz_u64_clz(sz_u64_t x) { return __builtin_clzll(x); }
STRINGZILLA_HELPER_AUTO int sz_u32_ctz(sz_u32_t x) { return __builtin_ctz(x); } // ! Undefined if `x == 0`
STRINGZILLA_HELPER_AUTO int sz_u32_clz(sz_u32_t x) { return __builtin_clz(x); } // ! Undefined if `x == 0`
STRINGZILLA_HELPER_AUTO sz_u64_t sz_u64_bytes_reverse(sz_u64_t val) { return __builtin_bswap64(val); }
STRINGZILLA_HELPER_AUTO sz_u32_t sz_u32_bytes_reverse(sz_u32_t val) { return __builtin_bswap32(val); }
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
STRINGZILLA_HELPER_INLINE void const *sz_x86_hide_pointer_origin_(void const *pointer) {
#if defined(__GNUC__)
    __asm__("" : "+r"(pointer));
#endif
    return pointer;
}

/** Reverse the 64 bits of @p value, so bit i moves to bit 63 - i: swap adjacent bits, then
 *  bit-pairs within nibbles, then nibbles within bytes, then the bytes. Lets an ascending-only
 *  byte-compress, like @c vpcompressb, pack lanes in descending order. */
STRINGZILLA_HELPER_AUTO sz_u64_t sz_u64_bits_reverse(sz_u64_t value) {
    value = ((value & 0x5555555555555555ull) << 1) | ((value >> 1) & 0x5555555555555555ull);
    value = ((value & 0x3333333333333333ull) << 2) | ((value >> 2) & 0x3333333333333333ull);
    value = ((value & 0x0F0F0F0F0F0F0F0Full) << 4) | ((value >> 4) & 0x0F0F0F0F0F0F0F0Full);
    return sz_u64_bytes_reverse(value);
}

/** Bit index of the n-th (0-based) set bit of @p bits; clears the @p n lowest set bits, then
 *  @c ctz. @p bits must hold more than @p n set bits. One tested home for the per-ISA SIMD
 *  "n-th lane" locate. */
STRINGZILLA_HELPER_AUTO int sz_u64_nth_set_bit(sz_u64_t bits, sz_size_t n) {
    while (n--) bits &= bits - 1;
    return sz_u64_ctz(bits);
}
STRINGZILLA_HELPER_AUTO int sz_u32_nth_set_bit(sz_u32_t bits, sz_size_t n) {
    while (n--) bits &= bits - 1;
    return sz_u32_ctz(bits);
}

/** Branchless `value | (bit if condition)`: OR @p bit into @p value when @p condition holds, with
 *  no branch - a mask-select rather than a CMOV, so there is no flag dependency. For threading a
 *  per-window carry signal into a lane mask. */
STRINGZILLA_HELPER_AUTO sz_u64_t sz_u64_or_if_(sz_u64_t value, sz_u64_t bit, int condition) {
    return value | (bit & ((sz_u64_t)0 - (sz_u64_t)(condition != 0)));
}

STRINGZILLA_HELPER_AUTO sz_u64_t sz_u64_rotl(sz_u64_t x, sz_u64_t r) { return (x << r) | (x >> (64 - r)); }

/**
 *  @brief Select bits from either @p a or @p b depending on the value of @p mask bits.
 *
 *  Similar to the @c _mm_blend_epi16 intrinsic on x86.
 *
 *  @see Bit Twiddling Hacks by Sean Eron Anderson: https://graphics.stanford.edu/~seander/bithacks.html#ConditionalSetOrClearBitsWithoutBranching
 */
STRINGZILLA_HELPER_AUTO sz_u64_t sz_u64_blend(sz_u64_t a, sz_u64_t b, sz_u64_t mask) { return a ^ ((a ^ b) & mask); }

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
STRINGZILLA_HELPER_AUTO sz_i32_t sz_i32_min_of_two(sz_i32_t x, sz_i32_t y) { return y + ((x - y) & (x - y) >> 31); }

/** Branchless maximum function for two signed 32-bit integers. */
STRINGZILLA_HELPER_AUTO sz_i32_t sz_i32_max_of_two(sz_i32_t x, sz_i32_t y) { return x - ((x - y) & (x - y) >> 31); }

/*  In AVX-512 we actively use masked operations and the "K mask registers". Producing a mask for
 *  the first N elements of a sequence can be done using the `1 << N - 1` idiom. It, however,
 *  induces undefined behavior if N is 64 or 32 on 64-bit or 32-bit systems respectively.
 *  Alternatively, the BZHI instruction can be used to clear the bits above N. */
#if STRINGZILLA_TARGET_SKYLAKE || STRINGZILLA_TARGET_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("bmi", "bmi2")
#endif
STRINGZILLA_HELPER_INLINE __mmask8 sz_u8_mask_until_(sz_size_t n) {
    return (__mmask8)_bzhi_u32(0xFFu, (unsigned char)n);
}
STRINGZILLA_HELPER_INLINE __mmask16 sz_u16_mask_until_(sz_size_t n) {
    return (__mmask16)_bzhi_u32(0xFFFFu, (unsigned char)n);
}
STRINGZILLA_HELPER_INLINE __mmask32 sz_u32_mask_until_(sz_size_t n) {
    return (__mmask32)_bzhi_u64(0xFFFFFFFFu, (unsigned char)n);
}
STRINGZILLA_HELPER_INLINE __mmask64 sz_u64_mask_until_(sz_size_t n) {
    return (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned char)n);
}
STRINGZILLA_HELPER_AUTO __mmask8 sz_u8_clamp_mask_until_(sz_size_t n) { return n < 8 ? sz_u8_mask_until_(n) : 0xFFu; }
STRINGZILLA_HELPER_AUTO __mmask16 sz_u16_clamp_mask_until_(sz_size_t n) {
    return n < 16 ? sz_u16_mask_until_(n) : 0xFFFFu;
}
STRINGZILLA_HELPER_AUTO __mmask32 sz_u32_clamp_mask_until_(sz_size_t n) {
    return n < 32 ? sz_u32_mask_until_(n) : 0xFFFFFFFFu;
}
STRINGZILLA_HELPER_AUTO __mmask64 sz_u64_clamp_mask_until_(sz_size_t n) {
    return n < 64 ? sz_u64_mask_until_(n) : 0xFFFFFFFFFFFFFFFFull;
}
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_SKYLAKE || STRINGZILLA_TARGET_ICELAKE

/**
 *  @brief Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each byte signifies a match.
 */
STRINGZILLA_HELPER_AUTO sz_u64_vec_t sz_u64_each_byte_equal_(sz_u64_vec_t a_vec, sz_u64_vec_t b_vec) {
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
STRINGZILLA_HELPER_AUTO sz_bool_t sz_ssize_clamp_interval_checked( //
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
STRINGZILLA_HELPER_AUTO void sz_ssize_clamp_interval( //
    sz_size_t length, sz_ssize_t start, sz_ssize_t end, sz_size_t *normalized_offset, sz_size_t *normalized_length) {
    sz_ssize_clamp_interval_checked(length, start, end, normalized_offset, normalized_length);
}

/**
 *  @brief Compute the logarithm base 2 of a positive integer, rounding down.
 *  @pre Input must be a positive number, as the logarithm of zero is undefined.
 */
STRINGZILLA_HELPER_AUTO sz_size_t sz_size_log2i_nonzero(sz_size_t x) {
    sz_assert_(x > 0 && "Non-positive numbers have no defined logarithm");
    int leading_zeros = sz_u64_clz(x);
    return (sz_size_t)(63 - leading_zeros);
}

/** Computes the ceiling of @p x divided by @p divisor - the number of chunks of that size needed to
 *  cover @p x. Assumes a non-zero @p divisor and no overflow on `x + divisor`. */
STRINGZILLA_HELPER_AUTO sz_size_t sz_size_divide_round_up(sz_size_t x, sz_size_t divisor) {
    return (x + divisor - 1) / divisor;
}

/**
 *  @brief Compute the smallest power of two greater than or equal to @p x.
 *  @note Uses LZCNT/CLZ for efficient computation on modern CPUs. Edge cases: bit_ceil(0) = 0,
 *      bit_ceil(1) = 1.
 *  @see Rounding up to a power of two: https://stackoverflow.com/a/10143264
 */
STRINGZILLA_HELPER_AUTO sz_size_t sz_size_bit_ceil(sz_size_t x) {
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
STRINGZILLA_HELPER_AUTO sz_u64_t sz_u64_transpose(sz_u64_t x) {
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
STRINGZILLA_HELPER_INLINE sz_u16_vec_t sz_u16_load(sz_cptr_t ptr) {
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
STRINGZILLA_HELPER_INLINE sz_u32_vec_t sz_u32_load(sz_cptr_t ptr) {
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
STRINGZILLA_HELPER_INLINE sz_u64_vec_t sz_u64_load(sz_cptr_t ptr) {
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
STRINGZILLA_HELPER_INLINE void sz_u16_store(sz_ptr_t ptr, sz_u16_t value) {
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
STRINGZILLA_HELPER_INLINE void sz_u32_store(sz_ptr_t ptr, sz_u32_t value) {
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
STRINGZILLA_HELPER_INLINE void sz_u64_store(sz_ptr_t ptr, sz_u64_t value) {
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

/** Bytes a fixed buffer rounds each block up to, so an odd length can't misalign the next block. */
enum { sz_memory_alignment_k = 64 };

/**
 *  @brief Helper function, using the supplied fixed-capacity buffer to allocate memory.
 *
 *  Offsets are rounded to @ref sz_memory_alignment_k, so a block is aligned as far as the caller's
 *  own buffer is: pass one aligned to 64 and every block is, and a malloc'd buffer still carries
 *  its own guarantee.
 */
STRINGZILLA_HELPER_AUTO sz_ptr_t sz_memory_allocate_fixed_(sz_size_t length, void *handle) {

    sz_size_t const capacity = *(sz_size_t *)handle;
    sz_size_t const consumed_capacity = *((sz_size_t *)handle + 1);
    sz_size_t const aligned_capacity =
        (consumed_capacity + sz_memory_alignment_k - 1) & ~(sz_size_t)(sz_memory_alignment_k - 1);
    if (aligned_capacity + length > capacity) return STRINGZILLA_NULL_CHAR;
    // Increase the consumed capacity.
    *((sz_size_t *)handle + 1) = aligned_capacity + length;
    return (sz_ptr_t)handle + aligned_capacity;
}

/** Helper "no-op" function, simulating memory deallocation when we use a "static" memory buffer. */
STRINGZILLA_HELPER_AUTO void sz_memory_free_fixed_(sz_ptr_t start, sz_size_t length, void *handle) {
    sz_unused_(start && length && handle);
}

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#pragma endregion

#pragma region Serial Implementation

#if STRINGZILLA_WITH_LIBC
#include <stdio.h>  // `fprintf`
#include <stdlib.h> // `malloc`, `EXIT_FAILURE`

STRINGZILLA_API_COMPTIME void *sz_memory_allocate_default_(sz_size_t length, void *handle) {
    sz_unused_(handle);
    if (length == 0) return STRINGZILLA_NULL;
    return malloc(length);
}
STRINGZILLA_API_COMPTIME void sz_memory_free_default_(sz_ptr_t start, sz_size_t length, void *handle) {
    sz_unused_(handle && length);
    free(start);
}

#endif

STRINGZILLA_API_COMPTIME void sz_memory_allocator_init_default(sz_memory_allocator_t *allocator) {
#if STRINGZILLA_WITH_LIBC
    allocator->allocate = (sz_memory_allocate_t)sz_memory_allocate_default_;
    allocator->free = (sz_memory_free_t)sz_memory_free_default_;
#else
    allocator->allocate = (sz_memory_allocate_t)STRINGZILLA_NULL;
    allocator->free = (sz_memory_free_t)STRINGZILLA_NULL;
#endif
    allocator->handle = STRINGZILLA_NULL;
}

STRINGZILLA_API_COMPTIME void sz_memory_allocator_init_fixed(sz_memory_allocator_t *allocator, void *buffer,
                                                             sz_size_t length) {
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

STRINGZILLA_API_COMPTIME sz_bool_t sz_memory_allocator_equal(sz_memory_allocator_t const *a,
                                                             sz_memory_allocator_t const *b) {
    if (!a || !b) return sz_false_k;

    // Two allocators are considered equal if they have the same function pointers and handle
    return (a->allocate == b->allocate) && (a->free == b->free) && (a->handle == b->handle) ? sz_true_k : sz_false_k;
}

STRINGZILLA_API_COMPTIME sz_cptr_t sz_sequence_from_null_terminated_strings_get_start_(void const *handle,
                                                                                       sz_size_t i) {
    sz_cptr_t const *start = (sz_cptr_t const *)handle;
    return start[i];
}

STRINGZILLA_API_COMPTIME sz_size_t sz_sequence_from_null_terminated_strings_get_length_(void const *handle,
                                                                                        sz_size_t i) {
    sz_cptr_t const *start = (sz_cptr_t const *)handle;
    sz_size_t length = 0;
    for (sz_cptr_t ptr = start[i]; *ptr; ptr++) length++;
    return length;
}

STRINGZILLA_API_COMPTIME void sz_sequence_from_null_terminated_strings(sz_cptr_t *start, sz_size_t count,
                                                                       sz_sequence_t *sequence) {
    sequence->handle = start;
    sequence->count = count;
    sequence->get_start = sz_sequence_from_null_terminated_strings_get_start_;
    sequence->get_length = sz_sequence_from_null_terminated_strings_get_length_;
}

STRINGZILLA_API_COMPTIME sz_cptr_t sz_sequence_from_string_views_get_start_(void const *handle, sz_size_t i) {
    sz_string_view_t const *views = (sz_string_view_t const *)handle;
    return views[i].start;
}

STRINGZILLA_API_COMPTIME sz_size_t sz_sequence_from_string_views_get_length_(void const *handle, sz_size_t i) {
    sz_string_view_t const *views = (sz_string_view_t const *)handle;
    return views[i].length;
}

STRINGZILLA_API_COMPTIME void sz_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                            sz_sequence_t *sequence) {
    sequence->handle = views;
    sequence->count = count;
    sequence->get_start = sz_sequence_from_string_views_get_start_;
    sequence->get_length = sz_sequence_from_string_views_get_length_;
}

#pragma endregion

#ifdef __cplusplus
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}
#endif // __cplusplus

#endif // STRINGZILLA_TYPES_H_
