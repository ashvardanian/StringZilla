/**
 *  @file c/target/sve2aes.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c sve2aes kernels, defined once for the StringZilla library.
 */
#undef STRINGZILLA_TARGET_NEONAES
#define STRINGZILLA_TARGET_NEONAES 0
#include "stringzilla/stringzilla.h"

#include "stringzilla/hash/sve2aes.h"
#include "stringzilla/cipher/sve2aes.h"
