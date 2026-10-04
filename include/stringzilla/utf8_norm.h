/**
 *  @file include/stringzilla/utf8_norm.h
 *  @author Ash Vardanian
 *  @date June 14, 2026
 *  @brief Hardware-accelerated single-pass Unicode normalization (NFD / NFC / NFKD / NFKC).
 *
 *  Includes core APIs:
 *
 *  - @c sz_utf8_norm_best - transform UTF-8 text into a Unicode normalization form
 *  - @c sz_utf8_find_denormalized_best - find the first byte breaking a normalization form, if any
 *
 *  Normalization is built from three UAX #15 primitives - decomposition, canonical ordering, and
 *  composition - sharing one ISA-agnostic table set (`utf8_norm/tables.h`, generated from the UCD
 *  by the recipe embedded in that header). It is intentionally locale-independent.
 */
#ifndef STRINGZILLA_UTF8_NORM_H_
#define STRINGZILLA_UTF8_NORM_H_

#include "stringzilla/types.h"        // `sz_normal_form_t`, `sz_size_t`, `sz_cptr_t`
#include "stringzilla/capabilities.h" // `sz_capability_t`
#include "stringzilla/cuda.cuh"       // Ahead of `extern "C"`, as the GPU runtimes' headers declare templates
#include "stringzilla/rocm.cuh"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Transform a UTF-8 string into a Unicode normalization form.
 *
 *  @section utf8_norm_buffer_sizing Buffer Sizing
 *
 *  Decomposition forms (NFD, NFKD) can expand the input; the target must hold up to
 *  `source_length * 18` bytes for the worst single-codepoint compatibility decomposition. The
 *  composing forms (NFC, NFKC) never exceed the decomposed length, so the same bound is safe.
 *
 *  Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is
 *  an opaque 1-byte barrier - it is passed through unchanged, does not decompose, compose, or take
 *  part in canonical ordering, and processing resyncs at the next byte.
 *
 *  On CUDA and ROCm the normalization is enqueued on @p stream, which names the device: the source,
 *  the target and the length slot must all be memory that device reaches, and the length lands once
 *  @p stream is joined. Each call copies the normalization tables to the device on @p stream ahead
 *  of its kernel, allocating nothing and joining nothing.
 *
 *  @param[in] source UTF-8 string to normalize.
 *  @param[in] source_length Number of bytes in @p source.
 *  @param[in] form One of @c sz_normal_form_nfd_k, @c _nfc_k, @c _nfkd_k, @c _nfkc_k.
 *  @param[out] target Buffer to receive the normalized UTF-8 string.
 *  @param[out] target_length Number of bytes written to @p target.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, which also names the
 *      device; null for the default.
 *  @return @c sz_success_k, @c sz_missing_kernel_k when no capability in @p capabilities has it,
 *      and on a GPU @c sz_device_memory_mismatch_k for a buffer or a @p stream the device cannot
 *      use or @c sz_unexpected_dimensions_k for a text of more than nine hundred gibibytes.
 *  @warning No bounds checking is performed on @p target.
 */
STRINGZILLA_API sz_status_t sz_utf8_norm_best(                        //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, //
    sz_ptr_t target, sz_size_t *target_length,                        //
    sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Locate the first byte that breaks a normalization form.
 *
 *  Malformed UTF-8 is treated losslessly: any byte that does not begin a well-formed codepoint is
 *  an opaque, inert 1-byte barrier (never a violation), and the scan resyncs at the next byte.
 *
 *  @param[in] source UTF-8 string to test.
 *  @param[in] source_length Number of bytes in @p source.
 *  @param[in] form One of @c sz_normal_form_nfd_k, @c _nfc_k, @c _nfkd_k, @c _nfkc_k.
 *  @param[out] match @c STRINGZILLA_NULL_CHAR if @p source is already in @p form, else a pointer to
 *      the first byte of the first codepoint breaking it: a non-Yes QC or out of canonical order.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_best(                             //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, //
    sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Finds the UTF-8 normalization kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_norm_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                     sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_serial(                                       //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_serial( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_haswell(                                      //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_haswell( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SKYLAKE
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_skylake(                                      //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_skylake( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_icelake(                                      //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_icelake( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_neon(                                         //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_neon( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_sve(                                          //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_sve( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_sve2(                                         //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_sve2( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_rvv(                                          //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_rvv( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_v128(                                         //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_v128( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_v128relaxed(                                  //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_v128relaxed( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_loongsonasx(                                  //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_loongsonasx( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_powervsx(                                     //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);

/** @copydoc sz_utf8_find_denormalized_best */
STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_powervsx( //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_cuda(                                         //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_utf8_norm_best */
STRINGZILLA_API sz_status_t sz_utf8_norm_rocm(                                         //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_ptr_t target, //
    sz_size_t *target_length, sz_stream_t stream);
#endif

#pragma endregion

#pragma region Backends

#include "stringzilla/utf8_norm/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_norm/haswell.h"
#include "stringzilla/utf8_norm/skylake.h"
#include "stringzilla/utf8_norm/icelake.h" // includes skylake.h; the guard makes the double-include safe
#include "stringzilla/utf8_norm/neon.h"
#include "stringzilla/utf8_norm/sve.h"
#include "stringzilla/utf8_norm/sve2.h" // includes sve.h; the guard makes the double-include safe
#include "stringzilla/utf8_norm/rvv.h"
#include "stringzilla/utf8_norm/v128.h"
#include "stringzilla/utf8_norm/v128relaxed.h" // includes v128.h; the guard makes the double-include safe
#include "stringzilla/utf8_norm/loongsonasx.h"
#include "stringzilla/utf8_norm/powervsx.h"
#include "stringzilla/utf8_norm/cuda.cuh"
#include "stringzilla/utf8_norm/rocm.cuh"
#endif // STRINGZILLA_HEADER_ONLY

#pragma endregion

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_norm_best(                        //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, //
    sz_ptr_t target, sz_size_t *target_length,                        //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(source), sz_unused_(source_length), sz_unused_(form), sz_unused_(target), sz_unused_(target_length),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_best(                             //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(source), sz_unused_(source_length), sz_unused_(form), sz_unused_(match), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_norm_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                     sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_H_
