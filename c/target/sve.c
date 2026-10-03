/**
 *  @file c/target/sve.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c sve kernels, defined once for the StringZilla library.
 */
#undef STRINGZILLA_TARGET_NEON
#define STRINGZILLA_TARGET_NEON 0
#include "stringzilla/stringzilla.h"

#include "stringzilla/compare/sve.h"
#include "stringzilla/memory/sve.h"
#include "stringzilla/hash/sve.h"
#include "stringzilla/find/sve.h"
#include "stringzilla/sort/sve.h"
#include "stringzilla/utf8_norm/sve.h"
