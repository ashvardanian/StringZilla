/**
*   @file       compare.c
 *  @brief      StringZilla compare functions compiled into library
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

// Now include compare.h which will use our overridden macros
#include <stringzilla/compare.h>
