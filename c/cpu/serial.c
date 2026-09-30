/**
 *  @file c/cpu/serial.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Every family's @c serial kernels, defined once for the StringZilla library.
 */
#undef STRINGZILLA_TARGET_SERIAL
#define STRINGZILLA_TARGET_SERIAL 1
#include "stringzilla/stringzilla.h"

#include "stringzilla/compare/serial.h"
#include "stringzilla/memory/serial.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/cipher/serial.h"
#include "stringzilla/find/serial.h"
#include "stringzilla/sort/serial.h"
#include "stringzilla/intersect/serial.h"
#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/overlap/serial.h"
#include "stringzilla/substrings/serial.h"
#include "stringzilla/utf8_runes/serial.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_graphemes/serial.h"
#include "stringzilla/utf8_sentences/serial.h"
#include "stringzilla/utf8_linebreaks/serial.h"
#include "stringzilla/utf8_uncased_fold/serial.h"
#include "stringzilla/utf8_uncased/serial.h"
#include "stringzilla/utf8_norm/serial.h"
