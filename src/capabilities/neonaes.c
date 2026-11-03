/**
 *  @file src/capabilities/neonaes.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c neonaes kernels, defined once for the StringZilla library.
 */
#undef STRINGZILLA_TARGET_NEON
#define STRINGZILLA_TARGET_NEON 0
#include "stringzilla/stringzilla.h"

#include "stringzilla/hash/neonaes.h"
#include "stringzilla/cipher/neonaes.h"
#include "stringzilla/intersect/neonaes.h"
