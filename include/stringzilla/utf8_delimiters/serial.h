/**
 *  @brief Serial backend for UTF-8 delimiter (punctuation/symbol/separator/whitespace) scanning.
 *  @file include/stringzilla/utf8_delimiters/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_DELIMITERS_SERIAL_H_
#define STRINGZILLA_UTF8_DELIMITERS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_delimiters/tables.h" // `sz_rune_is_delimiter_`
#include "stringzilla/utf8_runes/serial.h"      // `sz_rune_decode`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Serial

/**
 *  @brief Reference scan: returns the first delimiter codepoint and writes its byte length.
 *
 *  Decodes each codepoint with the bounds-checked `sz_rune_decode`, so a truncated trailing UTF-8
 *  sequence never over-reads past @p text + @p length. A byte that does not begin a well-formed
 *  codepoint (lone continuation, overlong, surrogate, truncated tail) is skipped one byte at a time
 *  and is never reported as a delimiter.
 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_cptr_t const end = text + length;
    sz_cptr_t position = text;
    while (position < end) {
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode(position, end, &rune);
        if (rune_length == sz_rune_invalid_k) { position += 1; }
        else if (sz_rune_is_delimiter_(rune)) {
            if (matched_length) *matched_length = (sz_size_t)rune_length;
            return position;
        }
        else { position += rune_length; }
    }
    if (matched_length) *matched_length = 0;
    return SZ_NULL_CHAR;
}

#pragma endregion // Serial

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_DELIMITERS_SERIAL_H_
