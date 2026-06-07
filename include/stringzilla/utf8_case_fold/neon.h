/**
 *  @brief Arm NEON backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_case_fold/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_fold.h
 */
#ifndef STRINGZILLA_UTF8_CASE_FOLD_NEON_H_
#define STRINGZILLA_UTF8_CASE_FOLD_NEON_H_

#include "stringzilla/utf8_case_fold/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON

SZ_PUBLIC sz_size_t sz_utf8_case_fold_neon(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    return sz_utf8_case_fold_serial(source, source_length, destination);
}

#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_FOLD_NEON_H_
