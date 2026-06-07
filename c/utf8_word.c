/**
 *  @file       utf8_word.c
 *  @brief      StringZilla UTF-8 word functions compiled into library
 *  @author     Raul Marin
 *  @date       February 5, 2026
 */

#ifdef SZ_DYNAMIC_DISPATCH
#undef SZ_DYNAMIC_DISPATCH
#endif
#define SZ_DYNAMIC_DISPATCH 0

// Include types.h first to let it define the macros
#include <stringzilla/types.h>

// Use weak symbols to allow multiple definitions (linker will merge them)
#undef SZ_PUBLIC
#undef SZ_C_INLINE
#undef SZ_INTERNAL
#define SZ_C_INLINE __attribute__((weak))
#define SZ_PUBLIC __attribute__((weak))
#define SZ_INTERNAL static inline
#undef SZ_DYNAMIC
#define SZ_DYNAMIC static inline


#include <stringzilla/utf8_word.h>
