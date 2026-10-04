/**
 *  @file c/target/neon.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c neon kernels, defined once for the StringZilla library.
 */
#include "stringzilla/stringzilla.h"

#include "stringzilla/compare/neon.h"
#include "stringzilla/memory/neon.h"
#include "stringzilla/hash/neon.h"
#include "stringzilla/find/neon.h"
#include "stringzilla/sort/neon.h"
#include "stringzilla/levenshtein/neon.h"
#include "stringzilla/overlap/neon.h"
#include "stringzilla/substrings/neon.h"
#include "stringzilla/utf8_runes/neon.h"
#include "stringzilla/utf8_tokens/neon.h"
#include "stringzilla/utf8_wordbreaks/neon.h"
#include "stringzilla/utf8_graphemes/neon.h"
#include "stringzilla/utf8_sentences/neon.h"
#include "stringzilla/utf8_linebreaks/neon.h"
#include "stringzilla/utf8_uncased_fold/neon.h"
#include "stringzilla/utf8_uncased/neon.h"
#include "stringzilla/utf8_norm/neon.h"
