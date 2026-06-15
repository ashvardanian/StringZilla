/**
 *  @brief Hardware-accelerated single-pass Unicode normalization (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_utf8_norm` - transform UTF-8 text into a Unicode normalization form
 *  - `sz_utf8_norm_violation` - locate the first byte that breaks a normalization form (NULL if none)
 *
 *  Normalization is built from three UAX #15 primitives - decomposition, canonical ordering, and
 *  composition - sharing one ISA-agnostic table set (`utf8_norm/tables.h`, generated from the UCD by
 *  the recipe embedded in that header). It is intentionally locale-independent.
 */
#ifndef STRINGZILLA_UTF8_NORM_H_
#define STRINGZILLA_UTF8_NORM_H_

#include "stringzilla/types.h" // `sz_normal_form_t`, `sz_size_t`, `sz_cptr_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Transform a UTF-8 string into a Unicode normalization form.
 *
 *  @section Buffer Sizing
 *
 *  Decomposition forms (NFD, NFKD) can expand the input; the destination must hold up to
 *  `source_length * 18` bytes for the worst single-codepoint compatibility decomposition. The
 *  composing forms (NFC, NFKC) never exceed the decomposed length, so the same bound is safe.
 *
 *  @param source UTF-8 string to normalize. Must be valid UTF-8.
 *  @param source_length Number of bytes in @p source.
 *  @param form One of `sz_normal_form_nfd_k`, `_nfc_k`, `_nfkd_k`, `_nfkc_k`.
 *  @param destination Buffer to receive the normalized UTF-8 string.
 *  @return Number of bytes written to @p destination.
 *
 *  @warning No bounds checking is performed on @p destination. The source must be valid UTF-8.
 */
SZ_DYNAMIC sz_size_t sz_utf8_norm(      //
    sz_cptr_t source, sz_size_t length, //
    sz_normal_form_t form, sz_ptr_t destination);

/**
 *  @brief Locate the first byte that breaks a normalization form.
 *  @param source UTF-8 string to test. Must be valid UTF-8.
 *  @param length Number of bytes in @p source.
 *  @param form One of `sz_normal_form_nfd_k`, `_nfc_k`, `_nfkd_k`, `_nfkc_k`.
 *  @return `SZ_NULL_CHAR` if @p source is already in @p form; otherwise a pointer to the first byte
 *          that begins a codepoint breaking the form (first non-Yes QC or canonical-order violation).
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_norm_violation( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);

/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_serial( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_serial( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);

#if SZ_USE_SKYLAKE
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_skylake( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_skylake( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_icelake( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_icelake( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_haswell( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_haswell( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_neon( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_neon( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_SVE
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_sve( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_sve( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_SVE2
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_sve2( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_sve2( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_RVV
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_rvv( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_rvv( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_V128
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_v128( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_v128( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_V128RELAXED
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_v128relaxed( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_v128relaxed( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_LASX
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_lasx( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_lasx( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#if SZ_USE_POWERVSX
/** @copydoc sz_utf8_norm */
SZ_PUBLIC sz_size_t sz_utf8_norm_powervsx( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination);

/** @copydoc sz_utf8_norm_violation */
SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_powervsx( //
    sz_cptr_t source, sz_size_t length, sz_normal_form_t form);
#endif

#pragma endregion // Core API

#pragma region Backends

#include "stringzilla/utf8_norm/serial.h"
#include "stringzilla/utf8_norm/haswell.h"
#include "stringzilla/utf8_norm/skylake.h"
#include "stringzilla/utf8_norm/icelake.h" // includes skylake.h; the guard makes the double-include safe
#include "stringzilla/utf8_norm/neon.h"
#include "stringzilla/utf8_norm/sve.h"
#include "stringzilla/utf8_norm/sve2.h" // includes sve.h; the guard makes the double-include safe
#include "stringzilla/utf8_norm/rvv.h"
#include "stringzilla/utf8_norm/v128.h"
#include "stringzilla/utf8_norm/v128relaxed.h" // includes v128.h; the guard makes the double-include safe
#include "stringzilla/utf8_norm/lasx.h"
#include "stringzilla/utf8_norm/powervsx.h"

#pragma endregion // Backends

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_utf8_norm(sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t destination) {
#if SZ_USE_ICELAKE
    return sz_utf8_norm_icelake(source, length, form, destination);
#elif SZ_USE_SKYLAKE
    return sz_utf8_norm_skylake(source, length, form, destination);
#elif SZ_USE_HASWELL
    return sz_utf8_norm_haswell(source, length, form, destination);
#elif SZ_USE_SVE2
    return sz_utf8_norm_sve2(source, length, form, destination);
#elif SZ_USE_SVE
    return sz_utf8_norm_sve(source, length, form, destination);
#elif SZ_USE_NEON
    return sz_utf8_norm_neon(source, length, form, destination);
#elif SZ_USE_RVV
    return sz_utf8_norm_rvv(source, length, form, destination);
#elif SZ_USE_LASX
    return sz_utf8_norm_lasx(source, length, form, destination);
#elif SZ_USE_POWERVSX
    return sz_utf8_norm_powervsx(source, length, form, destination);
#elif SZ_USE_V128RELAXED
    return sz_utf8_norm_v128relaxed(source, length, form, destination);
#elif SZ_USE_V128
    return sz_utf8_norm_v128(source, length, form, destination);
#else
    return sz_utf8_norm_serial(source, length, form, destination);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_norm_violation(sz_cptr_t source, sz_size_t length, sz_normal_form_t form) {
#if SZ_USE_ICELAKE
    return sz_utf8_norm_violation_icelake(source, length, form);
#elif SZ_USE_SKYLAKE
    return sz_utf8_norm_violation_skylake(source, length, form);
#elif SZ_USE_HASWELL
    return sz_utf8_norm_violation_haswell(source, length, form);
#elif SZ_USE_SVE2
    return sz_utf8_norm_violation_sve2(source, length, form);
#elif SZ_USE_SVE
    return sz_utf8_norm_violation_sve(source, length, form);
#elif SZ_USE_NEON
    return sz_utf8_norm_violation_neon(source, length, form);
#elif SZ_USE_RVV
    return sz_utf8_norm_violation_rvv(source, length, form);
#elif SZ_USE_LASX
    return sz_utf8_norm_violation_lasx(source, length, form);
#elif SZ_USE_POWERVSX
    return sz_utf8_norm_violation_powervsx(source, length, form);
#elif SZ_USE_V128RELAXED
    return sz_utf8_norm_violation_v128relaxed(source, length, form);
#elif SZ_USE_V128
    return sz_utf8_norm_violation_v128(source, length, form);
#else
    return sz_utf8_norm_violation_serial(source, length, form);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_H_
