/**
 *  @brief Serial backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_SERIAL_H_
#define STRINGZILLA_UTF8_TOKENS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Scalar newline scan emitting every delimiter into parallel offset/length arrays.
 *
 *  A @c "\r\n" CRLF is one match of length 2 (its trailing LF is never emitted alone). `base` is added to every
 *  emitted offset and to `*bytes_consumed`, the resume offset, which is always a true delimiter boundary.
 */
SZ_INTERNAL sz_size_t sz_utf8_newlines_serial_(         //
    sz_cptr_t text, sz_size_t length, sz_size_t base,   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_bytes = (sz_u8_t const *)text;
    sz_size_t match_count = 0, position = 0;
    while (position < length && match_count < matches_capacity) {
        sz_u8_t const lead_byte = text_bytes[position];
        sz_size_t match_length = 0; // 0 means "no delimiter starts at this position"
        switch (lead_byte) {
        case '\n':
        case '\v':
        case '\f': match_length = 1; break;
        case '\r': match_length = (position + 1 < length && text_bytes[position + 1] == '\n') ? 2 : 1; break;
        case 0xC2: // U+0085 NEXT LINE
            if (position + 1 < length && text_bytes[position + 1] == 0x85) match_length = 2;
            break;
        case 0xE2: // U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR
            if (position + 2 < length && text_bytes[position + 1] == 0x80 &&
                (text_bytes[position + 2] == 0xA8 || text_bytes[position + 2] == 0xA9))
                match_length = 3;
            break;
        default: break;
        }
        if (match_length) {
            match_offsets[match_count] = base + position, match_lengths[match_count] = match_length, ++match_count;
            position += match_length;
        }
        else { position += 1; }
    }
    if (bytes_consumed) *bytes_consumed = base + position;
    return match_count;
}

/**
 *  @brief Scalar multistep whitespace scan: classify each codepoint inline and emit every delimiter.
 *
 *  Same contract as `sz_utf8_newlines_serial_` but for the Unicode White_Space set. There is no CRLF
 *  merging here - CR and LF are independent length-1 matches.
 */
SZ_INTERNAL sz_size_t sz_utf8_whitespaces_serial_(      //
    sz_cptr_t text, sz_size_t length, sz_size_t base,   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_bytes = (sz_u8_t const *)text;
    sz_size_t match_count = 0, position = 0;
    while (position < length && match_count < matches_capacity) {
        sz_u8_t const lead_byte = text_bytes[position];
        sz_size_t match_length = 0;
        switch (lead_byte) {
        case ' ':
        case '\t':
        case '\n':
        case '\v':
        case '\f':
        case '\r': match_length = 1; break;
        case 0xC2: // U+0085 NEL, U+00A0 NBSP
            if (position + 1 < length && (text_bytes[position + 1] == 0x85 || text_bytes[position + 1] == 0xA0))
                match_length = 2;
            break;
        case 0xE1: // U+1680 OGHAM SPACE MARK (E1 9A 80)
            if (position + 2 < length && text_bytes[position + 1] == 0x9A && text_bytes[position + 2] == 0x80)
                match_length = 3;
            break;
        case 0xE2: // U+2000..U+200A, U+2028, U+2029, U+202F, U+205F
            if (position + 2 < length) {
                sz_u8_t const second = text_bytes[position + 1], third = text_bytes[position + 2];
                if (second == 0x80 &&
                    ((third >= 0x80 && third <= 0x8D) || third == 0xA8 || third == 0xA9 || third == 0xAF))
                    match_length = 3;
                else if (second == 0x81 && third == 0x9F) match_length = 3;
            }
            break;
        case 0xE3: // U+3000 IDEOGRAPHIC SPACE (E3 80 80)
            if (position + 2 < length && text_bytes[position + 1] == 0x80 && text_bytes[position + 2] == 0x80)
                match_length = 3;
            break;
        default: break;
        }
        if (match_length) {
            match_offsets[match_count] = base + position, match_lengths[match_count] = match_length, ++match_count;
            position += match_length;
        }
        else { position += 1; }
    }
    if (bytes_consumed) *bytes_consumed = base + position;
    return match_count;
}

SZ_PUBLIC sz_size_t sz_utf8_newlines_serial(            //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_newlines_serial_(text, length, 0, match_offsets, match_lengths, matches_capacity, bytes_consumed);
}

SZ_PUBLIC sz_size_t sz_utf8_whitespaces_serial(         //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_whitespaces_serial_(text, length, 0, match_offsets, match_lengths, matches_capacity, bytes_consumed);
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_SERIAL_H_
