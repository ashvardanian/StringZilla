/**
 *  @file include/stringzilla/utf8_uncased_fold.h
 *  @author Ash Vardanian
 *  @date November 23, 2025
 *  @brief Hardware-accelerated UTF-8 case folding.
 *
 *  Includes core APIs:
 *
 *  - @c sz_utf8_uncased_fold_best - Unicode case folding for uncased comparisons
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_H_

#include "stringzilla/utf8_runes/serial.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Apply Unicode case folding to a UTF-8 string.
 *
 *  Case folding normalizes text for uncased comparisons by mapping uppercase letters to their
 *  lowercase equivalents and handling special expansions defined in Unicode CaseFolding.txt.
 *
 *  @section utf8_uncased_fold_buffer_sizing Buffer Sizing
 *
 *  The target buffer must be at least `source_length * 3` bytes to guarantee sufficient space for
 *  worst-case expansion. The maximum expansion ratio is 3:1, which occurs with Greek characters
 *  that expand to three codepoints under case folding. For example, 'ΐ' (U+0390, CE 90) folds to
 *  "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81), so 2 bytes → 6 bytes, and a string of N such
 *  characters expands from 2N to 6N bytes.
 *
 *  Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is
 *  copied through unchanged (it folds to itself) and folding resyncs at the next byte. Valid UTF-8
 *  folds as usual. Basic usage:
 *
 *  @code{.c}
 *      char const *source = "HELLO";
 *      char target[15]; // 5 * 3, a safe overestimate
 *      sz_capability_t capabilities;
 *      sz_cpu_capabilities_enabled(&capabilities);
 *      sz_size_t target_length;
 *      sz_utf8_uncased_fold_best(source, 5, target, &target_length, capabilities, NULL);
 *      // target now contains "hello", target_length = 5
 *  @endcode
 *
 *  On CUDA and ROCm the fold is enqueued on @p stream, which names the device: the source, the
 *  target and the length slot must all be memory that device reaches, and the length lands once
 *  @p stream is joined, the call itself allocating nothing and joining nothing.
 *
 *  @param[in] source UTF-8 string to be case-folded.
 *  @param[in] source_length Number of bytes in the source buffer.
 *  @param[out] target Buffer to write the case-folded UTF-8 string.
 *  @param[out] target_length Number of bytes written to the target buffer.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, which also names the
 *      device; null for the default.
 *  @return @c sz_success_k, @c sz_missing_kernel_k when no capability in @p capabilities has it,
 *      and on a GPU @c sz_device_memory_mismatch_k for a buffer or a @p stream the device cannot
 *      use or @c sz_unexpected_dimensions_k for a text of more than five tebibytes.
 *  @warning The caller must ensure the target buffer is large enough. No bounds checking is
 *      performed. Use `source_length * 3` for safety.
 */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_best( //
    sz_cptr_t source, sz_size_t source_length,         //
    sz_ptr_t target, sz_size_t *target_length,         //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Finds the UTF-8 case folding kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                             sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_serial(                                  //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_haswell(                                 //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_icelake(                                 //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_neon(                                    //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_sve2(                                    //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_rvv(                                     //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_v128(                                    //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_loongsonasx(                             //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_powervsx(                                //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_cuda(                                    //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_utf8_uncased_fold_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_rocm(                                    //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t target, sz_size_t *target_length, //
    void *stream);
#endif

#pragma endregion

#pragma region Backends

#include "stringzilla/utf8_uncased_fold/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_uncased_fold/icelake.h"
#include "stringzilla/utf8_uncased_fold/haswell.h"
#include "stringzilla/utf8_uncased_fold/neon.h"
#include "stringzilla/utf8_uncased_fold/sve2.h"
#include "stringzilla/utf8_uncased_fold/v128.h"
#include "stringzilla/utf8_uncased_fold/rvv.h"
#include "stringzilla/utf8_uncased_fold/loongsonasx.h"
#include "stringzilla/utf8_uncased_fold/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#pragma endregion

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_best( //
    sz_cptr_t source, sz_size_t source_length,         //
    sz_ptr_t target, sz_size_t *target_length,         //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(source), sz_unused_(source_length), sz_unused_(target), sz_unused_(target_length),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                             sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_H_
