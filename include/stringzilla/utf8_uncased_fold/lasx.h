/**
 *  @file include/stringzilla/utf8_uncased_fold/lasx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief LoongArch ASX backend for UTF-8 case folding (delegates to serial).
 *
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_LASX_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_LASX_H_

#include "stringzilla/utf8_uncased_fold/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_lasx(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    return sz_utf8_uncased_fold_serial(source, source_length, destination);
}

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_LASX_H_
