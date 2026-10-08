/**
 *  @file c/target/sve2.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c sve2 kernels, defined once for the StringZilla library.
 */
#undef STRINGZILLA_TARGET_NEON
#define STRINGZILLA_TARGET_NEON 0
#undef STRINGZILLA_TARGET_SVE
#define STRINGZILLA_TARGET_SVE 0
#include "stringzilla/stringzilla.h"

#include "stringzilla/hash/sve2.h"
#include "stringzilla/find/sve2.h"
#include "stringzilla/substrings/sve2.h"
#include "stringzilla/utf8_runes/sve2.h"
#include "stringzilla/utf8_tokens/sve2.h"
#include "stringzilla/utf8_wordbreaks/sve2.h"
#include "stringzilla/utf8_graphemes/sve2.h"
#include "stringzilla/utf8_sentences/sve2.h"
#include "stringzilla/utf8_linebreaks/sve2.h"
#include "stringzilla/utf8_uncased_fold/sve2.h"
#include "stringzilla/utf8_uncased/sve2.h"
#include "stringzilla/utf8_norm/sve2.h"
