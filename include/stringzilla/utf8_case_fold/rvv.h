/**
 *  @brief RISC-V Vector backend for UTF-8 case folding (delegates to serial).
 *  @file include/stringzilla/utf8_case_fold/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_fold.h
 */
#ifndef STRINGZILLA_UTF8_CASE_FOLD_RVV_H_
#define STRINGZILLA_UTF8_CASE_FOLD_RVV_H_

#include "stringzilla/utf8_case_fold/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

SZ_PUBLIC sz_size_t sz_utf8_case_fold_rvv(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    return sz_utf8_case_fold_serial(source, source_length, destination);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_FOLD_RVV_H_
