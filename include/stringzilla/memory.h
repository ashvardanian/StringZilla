/**
 *  @file include/stringzilla/memory.h
 *  @author Ash Vardanian
 *  @date October 7, 2023
 *  @brief Hardware-accelerated memory operations.
 *
 *  Includes core APIs for contiguous memory operations:
 *
 *  - @c sz_copy_best - analog to @c memcpy, probably the most common operation in a computer.
 *  - @c sz_move_best - analog to @c memmove, allowing overlaps, often used in string edits.
 *  - @c sz_fill_best - analog to @c memset, often used to initialize memory with a constant.
 *  - @c sz_lookup_best - Look-Up Table @b (LUT) transformation, mapping every byte to a new value.
 *  - @c sz_lookup_utf8 - planned LUT transformation of a UTF-8 string, usable for normalization.
 *  - @c sz_allocator_init_unified_best - an allocator of blocks the host and a device share.
 *  - @c sz_sequence_realloc_best - reallocates a sequence as a tape for the selected device.
 *
 *  All of the core APIs receive the target output buffer as the first argument, and aim to minimize
 *  the number of "store" instructions, especially unaligned ones that can invalidate 2 cache lines.
 *
 *  Unlike many other libraries focusing on trivial SIMD transformations, like converting lowercase
 *  to uppercase, StringZilla generalizes those to basic lookup table transforms. For typical ASCII
 *  conversions, you can use the following @b LUT initialization functions:
 *
 *  - @c sz_lookup_init_lower for transforms like @c tolower
 *  - @c sz_lookup_init_upper for transforms like @c toupper
 *  - @c sz_lookup_init_ascii for transforms like @c isascii
 */
#ifndef STRINGZILLA_MEMORY_H_
#define STRINGZILLA_MEMORY_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`
#include "stringzilla/cuda.cuh"       // Ahead of `extern "C"`, as the GPU runtimes' headers declare templates
#include "stringzilla/rocm.cuh"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Similar to @c memcpy, copies contents of one string into another.
 *
 *  @param[out] target String to copy into. Can be @c NULL, if the @p length is zero.
 *  @param[in] source String to copy from. Can be @c NULL, if the @p length is zero.
 *  @param[in] length Number of bytes to copy. Can be a zero.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @see memcpy: https://en.cppreference.com/w/c/string/byte/memcpy
 *
 *  Copying two bytes into a buffer:
 *
 *  @code{.c}
 *      #include <stringzilla/memory.h>
 *      int main() {
 *          sz_capability_t capabilities;
 *          char output[2];
 *          sz_capabilities_enabled_cpu(&capabilities);
 *          sz_copy_best(output, "hi", 2, capabilities, NULL);
 *          return output[0] == 'h' && output[1] == 'i' ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @pre The @p target and @p source must not overlap.
 *  @sa sz_move_best
 *
 *  @sa sz_copy_serial, sz_copy_haswell, sz_copy_skylake, sz_copy_neon, sz_copy_sve, sz_copy_v128,
 *      sz_copy_v128relaxed, sz_copy_rvv, sz_copy_loongsonasx, sz_copy_powervsx
 */
STRINGZILLA_API sz_status_t sz_copy_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                         sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Similar to @c memmove, copies (moves) contents of one string into another. Unlike
 *      @c sz_copy_best, allows overlapping strings as arguments.
 *
 *  @param[out] target String to copy into. Can be @c NULL, if the @p length is zero.
 *  @param[in] source String to copy from. Can be @c NULL, if the @p length is zero.
 *  @param[in] length Number of bytes to copy. Can be a zero.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @see memmove: https://en.cppreference.com/w/c/string/byte/memmove
 *
 *  Shifting a buffer left by one byte in place:
 *
 *  @code{.c}
 *      #include <stringzilla/memory.h>
 *      int main() {
 *          sz_capability_t capabilities;
 *          char buffer[3] = {'a', 'b', 'c'};
 *          sz_capabilities_enabled_cpu(&capabilities);
 *          sz_move_best(buffer, buffer + 1, 2, capabilities, NULL);
 *          return buffer[0] == 'b' && buffer[1] == 'c' && buffer[2] == 'c' ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @sa sz_move_serial, sz_move_haswell, sz_move_skylake, sz_move_neon, sz_move_sve, sz_move_v128,
 *      sz_move_v128relaxed, sz_move_rvv, sz_move_loongsonasx, sz_move_powervsx
 */
STRINGZILLA_API sz_status_t sz_move_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                         sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Similar to @c memset, fills a string with a given value.
 *
 *  @param[out] target String to fill. Can be @c NULL, if the @p length is zero.
 *  @param[in] length Number of bytes to fill. Can be a zero.
 *  @param[in] value Value to fill with.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @see memset: https://en.cppreference.com/w/c/string/byte/memset
 *
 *  Filling a buffer with one character:
 *
 *  @code{.c}
 *     #include <stringzilla/memory.h>
 *     int main() {
 *          sz_capability_t capabilities;
 *          char buffer[2];
 *          sz_capabilities_enabled_cpu(&capabilities);
 *          sz_fill_best(buffer, 2, 'x', capabilities, NULL);
 *          return buffer[0] == 'x' && buffer[1] == 'x' ? 0 : 1;
 *     }
 *  @endcode
 *
 *  @sa sz_fill_serial, sz_fill_haswell, sz_fill_skylake, sz_fill_neon, sz_fill_sve, sz_fill_v128,
 *      sz_fill_v128relaxed, sz_fill_rvv, sz_fill_loongsonasx, sz_fill_powervsx
 */
STRINGZILLA_API sz_status_t sz_fill_best(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_capability_t capabilities,
                                         sz_stream_t stream);

/**
 *  @brief Look Up Table @b (LUT) transformation of a @p source string, the same as
 *      `for (char &c : text) c = lut[c]`.
 *
 *  @param[out] target Output string, can point to the same address as @p source.
 *  @param[in] source String to be mapped using the @p lut table into the @p target.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] lut Look Up Table to apply. Must be exactly @b 256 bytes long.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @see Lookup table: https://en.wikipedia.org/wiki/Lookup_table
 *
 *  Can be used to implement some form of string normalization, partially masking punctuation marks,
 *  or converting between different character sets, like uppercase or lowercase. Surprisingly, also
 *  has broad implications in image processing, where channel transformations often use LUTs.
 *
 *  Lowercasing a buffer in place:
 *
 *  @code{.c}
 *     #include <ctype.h> // for `tolower`
 *     #include <stringzilla/memory.h>
 *     int main() {
 *          sz_capability_t capabilities;
 *          char to_lower_lut[256];
 *          for (int i = 0; i < 256; ++i) to_lower_lut[i] = tolower(i);
 *          char buffer[3] = {'A', 'B', 'C'};
 *          sz_capabilities_enabled_cpu(&capabilities);
 *          sz_lookup_best(buffer, buffer, 3, to_lower_lut, capabilities, NULL);
 *          return buffer[0] == 'a' && buffer[1] == 'b' && buffer[2] == 'c' ? 0 : 1;
 *     }
 *  @endcode
 *
 *  @pre The @p lut must be exactly 256 bytes long, even if @p source has no bytes in the top range.
 *  @pre The @p target and @p source can be the same, but must not overlap.
 *
 *  @sa sz_lookup_serial, sz_lookup_haswell, sz_lookup_icelake, sz_lookup_neon,
 *      sz_lookup_sve, sz_lookup_v128, sz_lookup_v128relaxed, sz_lookup_rvv,
 *      sz_lookup_loongsonasx, sz_lookup_powervsx
 */
STRINGZILLA_API sz_status_t sz_lookup_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                           char const lut[sz_at_least_(256)], sz_capability_t capabilities,
                                           sz_stream_t stream);

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_serial(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                             char const lut[sz_at_least_(256)], sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_haswell(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_haswell(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                              char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_skylake(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_icelake(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                              char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_neon(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_neon(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                           char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_sve(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                          char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_rvv(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_rvv(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_rvv(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_rvv(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                          char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_v128(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_v128(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_v128(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_v128(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                           char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_v128relaxed(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_v128relaxed(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_v128relaxed(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_v128relaxed(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                  char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_loongsonasx(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_loongsonasx(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_loongsonasx(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_loongsonasx(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                  char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_copy_best */
STRINGZILLA_API sz_status_t sz_copy_powervsx(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_move_best */
STRINGZILLA_API sz_status_t sz_move_powervsx(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream);

/** @copydoc sz_fill_best */
STRINGZILLA_API sz_status_t sz_fill_powervsx(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream);

/** @copydoc sz_lookup_best */
STRINGZILLA_API sz_status_t sz_lookup_powervsx(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                               char const lut[sz_at_least_(256)], sz_stream_t stream);
#endif

/**
 *  @brief Finds the memory kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_memory_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#pragma region Memory Management

/**
 *  @brief Initializes an allocator whose blocks both the host and the device of @p capabilities
 *      address: the host heap on the CPU, managed memory on CUDA and ROCm, shared buffers on Metal.
 *
 *  The allocator is stateless, its @c handle null, and each call allocates on the device of the
 *  stream it is given, so one allocator serves every device of its vendor.
 *
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cuda reports;
 *      its group picks the CPU or a GPU vendor.
 *  @return @c sz_success_k, or @c sz_missing_gpu_k for a GPU vendor this library was built without.
 */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_best(sz_allocator_t *allocator, sz_capability_t capabilities);

/** Initializes device storage on CUDA and ROCm, or unified storage on Metal.
 *  Each callback uses its stream's device. CPU returns @c sz_missing_kernel_k; GPU vendors omitted
 *  from the build return @c sz_missing_gpu_k. */
STRINGZILLA_API sz_status_t sz_allocator_init_device_best(sz_allocator_t *allocator, sz_capability_t capabilities);

/** Initializes page-locked CUDA or ROCm host storage, using each callback's stream.
 *  CPU and Metal return @c sz_missing_kernel_k; unbuilt GPU vendors return @c sz_missing_gpu_k. */
STRINGZILLA_API sz_status_t sz_allocator_init_pinned_best(sz_allocator_t *allocator, sz_capability_t capabilities);

/**
 *  @brief Reallocates @p source as a tape from @p allocator when the selected device cannot use
 *      its storage, and points @p target at it through the canonical tape accessors.
 *
 *  A tape is count + 1 @c sz_u64_t offsets from the block's own start, then every string's bytes
 *  back to back. Its accessors always remain host-callable, including CUDA and ROCm tapes.
 *  Device kernels read the tape directly. A source tape can therefore be repacked for another
 *  vendor or device after its producer stream has completed.
 *
 *  @param[out] target The tape, untouched unless the call succeeds; it may be @p source itself.
 *      Neither the source allocation nor any previous target allocation is freed. Save their
 *      ownership information before replacing a descriptor in place.
 *  @param[in] source The strings, through host-callable accessors over host-readable texts, or a
 *      host-readable tape from any group. Complete prior device writes before this call.
 *  @param[in] allocator Where the block comes from, host-writable, like the one
 *      @ref sz_allocator_init_unified_best initializes.
 *  @param[out] allocated_bytes Bytes of the block, which the caller frees by passing
 *      `target->handle`, these bytes, `allocator->handle` and @p stream to `allocator->free`, or
 *      zero when the source tape is borrowed; its owner must keep it alive until work completes.
 *  @param[in] capabilities One device's capabilities; its group selects the backend.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, also naming the device:
 *      a @c cudaStream_t, a @c hipStream_t, or an @c id<MTLCommandQueue>; null for the default.
 *  @return @c sz_success_k; @c sz_bad_alloc_k when the block cannot be taken; or on a device
 *      @c sz_device_memory_mismatch_k for a @p stream it cannot use, @c sz_device_code_mismatch_k
 *      for a runtime failure, and @c sz_missing_gpu_k for a vendor this library was built without
 *      or a device that does not answer.
 *  @note Never joins: the host writes the block, and a device migration is queued on @p stream.
 */
STRINGZILLA_API sz_status_t sz_sequence_realloc_best(sz_sequence_t *target, sz_sequence_t const *source,
                                                     sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                     sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_allocator_init_unified_best */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_serial(sz_allocator_t *allocator);
/** @copydoc sz_sequence_realloc_best */
STRINGZILLA_API sz_status_t sz_sequence_realloc_serial(sz_sequence_t *target, sz_sequence_t const *source,
                                                       sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                       sz_stream_t stream);

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_allocator_init_unified_best */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_cuda(sz_allocator_t *allocator);
/** @copydoc sz_allocator_init_device_best */
STRINGZILLA_API sz_status_t sz_allocator_init_device_cuda(sz_allocator_t *allocator);
/** @copydoc sz_allocator_init_pinned_best */
STRINGZILLA_API sz_status_t sz_allocator_init_pinned_cuda(sz_allocator_t *allocator);
/** @copydoc sz_sequence_realloc_best */
STRINGZILLA_API sz_status_t sz_sequence_realloc_cuda(sz_sequence_t *target, sz_sequence_t const *source,
                                                     sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                     sz_stream_t stream);

#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_allocator_init_unified_best */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_rocm(sz_allocator_t *allocator);
/** @copydoc sz_allocator_init_device_best */
STRINGZILLA_API sz_status_t sz_allocator_init_device_rocm(sz_allocator_t *allocator);
/** @copydoc sz_allocator_init_pinned_best */
STRINGZILLA_API sz_status_t sz_allocator_init_pinned_rocm(sz_allocator_t *allocator);
/** @copydoc sz_sequence_realloc_best */
STRINGZILLA_API sz_status_t sz_sequence_realloc_rocm(sz_sequence_t *target, sz_sequence_t const *source,
                                                     sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                     sz_stream_t stream);

#endif

#if STRINGZILLA_TARGET_METAL
/** @copydoc sz_allocator_init_unified_best */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_metal(sz_allocator_t *allocator);
/** @copydoc sz_sequence_realloc_best */
STRINGZILLA_API sz_status_t sz_sequence_realloc_metal(sz_sequence_t *target, sz_sequence_t const *source,
                                                      sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                      sz_stream_t stream);

#endif

#pragma endregion Memory Management

#pragma region Helper API

/**
 *  @brief Initializes a lookup table for converting ASCII characters to lowercase.
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 *  @see SWAR swap case: http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122]. So there are 26
 *  English letters, shifted by 32 values, meaning that a conversion can be done by flipping the 5th
 *  bit of each inappropriate character byte. This, however, breaks for extended ASCII, so a
 *  different solution is needed.
 */
STRINGZILLA_INLINE void sz_lookup_init_lower(char lut[sz_at_least_(256)]) {
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
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 *  @see SWAR swap case: http://0x80.pl/notesen/2016-01-06-swar-swap-case.html
 *
 *  ASCII characters [A, Z] map to decimals [65, 90], and [a, z] map to [97, 122]. So there are 26
 *  English letters, shifted by 32 values, meaning that a conversion can be done by flipping the 5th
 *  bit of each inappropriate character byte. This, however, breaks for extended ASCII, so a
 *  different solution is needed.
 */
STRINGZILLA_INLINE void sz_lookup_init_upper(char lut[sz_at_least_(256)]) {
    static sz_u8_t const upped[256] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  //
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  //
        64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  //
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  //
        96,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  //
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  123, 124, 125, 126, 127, //
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, //
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, //
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, //
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, //
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, //
        208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, //
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, //
        208, 209, 210, 211, 212, 213, 214, 247, 216, 217, 218, 219, 220, 221, 222, 255, //
    };
    for (sz_size_t byte_index = 0; byte_index < 256; ++byte_index) lut[byte_index] = upped[byte_index];
}

/**
 *  @brief Initializes a lookup table for converting bytes to ASCII characters.
 *  @param[out] lut Lookup table to be initialized. Must be exactly 256 bytes long.
 */
STRINGZILLA_CONSTEXPR void sz_lookup_init_ascii(char lut[sz_at_least_(256)]) {
    for (sz_size_t byte_index = 0; byte_index < 256; ++byte_index) lut[byte_index] = (sz_u8_t)(byte_index & 0x7F);
}

#pragma endregion Helper API

#include "stringzilla/memory/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/memory/haswell.h"
#include "stringzilla/memory/skylake.h"
#include "stringzilla/memory/icelake.h"
#include "stringzilla/memory/neon.h"
#include "stringzilla/memory/sve.h"
#include "stringzilla/memory/v128relaxed.h"
#include "stringzilla/memory/v128.h"
#include "stringzilla/memory/rvv.h"
#include "stringzilla/memory/loongsonasx.h"
#include "stringzilla/memory/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

/*  Header-only builds link no library, so their dispatch points only report it missing. */
#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_copy_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                         sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(target), sz_unused_(source), sz_unused_(length), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_move_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                         sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(target), sz_unused_(source), sz_unused_(length), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_fill_best(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_capability_t capabilities,
                                         sz_stream_t stream) {
    sz_unused_(target), sz_unused_(length), sz_unused_(value), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_lookup_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                           char const lut[sz_at_least_(256)], sz_capability_t capabilities,
                                           sz_stream_t stream) {
    sz_unused_(target), sz_unused_(source), sz_unused_(length), sz_unused_(lut), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_memory_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator), sz_unused_(capabilities);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator), sz_unused_(capabilities);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator), sz_unused_(capabilities);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sequence_realloc_best(sz_sequence_t *target, sz_sequence_t const *source,
                                                     sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                     sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(target), sz_unused_(source), sz_unused_(allocator), sz_unused_(allocated_bytes),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_MEMORY_H_
