/**
 *  @file c/cpu/v128relaxed.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c v128relaxed kernels, defined once for the StringZilla library.
 */
#undef STRINGZILLA_TARGET_V128
#define STRINGZILLA_TARGET_V128 0
#include "stringzilla/stringzilla.h"

#include "stringzilla/compare/v128relaxed.h"
#include "stringzilla/memory/v128relaxed.h"
#include "stringzilla/hash/v128relaxed.h"
#include "stringzilla/cipher/v128relaxed.h"
#include "stringzilla/find/v128relaxed.h"
#include "stringzilla/utf8_runes/v128relaxed.h"
#include "stringzilla/utf8_norm/v128relaxed.h"
