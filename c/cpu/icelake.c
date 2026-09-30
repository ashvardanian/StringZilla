/**
 *  @file c/cpu/icelake.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c icelake kernels, defined once for the StringZilla library.
 */
#undef STRINGZILLA_TARGET_SKYLAKE
#define STRINGZILLA_TARGET_SKYLAKE 0
#undef STRINGZILLA_TARGET_WESTMERE
#define STRINGZILLA_TARGET_WESTMERE 0
#include "stringzilla/stringzilla.h"

#include "stringzilla/memory/icelake.h"
#include "stringzilla/hash/icelake.h"
#include "stringzilla/cipher/icelake.h"
#include "stringzilla/find/icelake.h"
#include "stringzilla/intersect/icelake.h"
#include "stringzilla/levenshtein/icelake.h"
#include "stringzilla/substrings/icelake.h"
#include "stringzilla/utf8_runes/icelake.h"
#include "stringzilla/utf8_tokens/icelake.h"
#include "stringzilla/utf8_wordbreaks/icelake.h"
#include "stringzilla/utf8_graphemes/icelake.h"
#include "stringzilla/utf8_sentences/icelake.h"
#include "stringzilla/utf8_linebreaks/icelake.h"
#include "stringzilla/utf8_uncased_fold/icelake.h"
#include "stringzilla/utf8_uncased/icelake.h"
#include "stringzilla/utf8_norm/icelake.h"
