/**
 *  @file c/cpu/skylake.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c skylake kernels, defined once for the StringZilla library.
 */
#undef STRINGZILLA_TARGET_WESTMERE
#define STRINGZILLA_TARGET_WESTMERE 0
#include "stringzilla/stringzilla.h"

#include "stringzilla/compare/skylake.h"
#include "stringzilla/memory/skylake.h"
#include "stringzilla/hash/skylake.h"
#include "stringzilla/find/skylake.h"
#include "stringzilla/sort/skylake.h"
#include "stringzilla/levenshtein/skylake.h"
#include "stringzilla/overlap/skylake.h"
#include "stringzilla/utf8_norm/skylake.h"
