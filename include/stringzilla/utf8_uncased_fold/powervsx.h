/**
 *  @file include/stringzilla/utf8_uncased_fold/powervsx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief IBM Power (VSX) backend for UTF-8 case folding (delegates to serial).
 *
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_POWERVSX_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_POWERVSX_H_

#include "stringzilla/utf8_uncased_fold/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_powervsx(sz_cptr_t source, sz_size_t source_length,
                                                                 sz_ptr_t destination) {
    return sz_utf8_uncased_fold_serial(source, source_length, destination);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_POWERVSX_H_
