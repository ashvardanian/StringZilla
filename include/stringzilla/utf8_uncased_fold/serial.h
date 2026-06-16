/**
 *  @brief Serial backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_uncased_fold/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_SERIAL_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Helper function performing case-folding under the constraint, that no output may be incomplete.
 *
 *  @param source Pointer to the source UTF-8 data, must be valid UTF-8.
 *  @param source_length Length of the source data in bytes.
 *  @param destination Pointer to the destination buffer.
 *  @param destination_length Length of the destination buffer in bytes.
 *  @param codepoints_consumed Number of codepoints read from source.
 *  @param codepoints_exported Number of codepoints written to destination.
 *  @param bytes_consumed Number of bytes read from source.
 *  @param bytes_exported Number of bytes written to destination.
 */
SZ_INTERNAL void sz_utf8_uncased_fold_upto_(                           //
    sz_cptr_t source, sz_size_t source_length,                      //
    sz_ptr_t destination, sz_size_t destination_length,             //
    sz_size_t *codepoints_consumed, sz_size_t *codepoints_exported, //
    sz_size_t *bytes_consumed, sz_size_t *bytes_exported) {

    sz_u8_t const *const source_start = (sz_u8_t const *)source;
    sz_u8_t const *const source_limit = source_start + source_length;
    sz_u8_t *const destination_start = (sz_u8_t *)destination;
    sz_u8_t *const destination_limit = destination_start + destination_length;

    sz_u8_t const *source_ptr = source_start;
    sz_u8_t *destination_ptr = destination_start;
    sz_size_t codepoints_read = 0;
    sz_size_t codepoints_written = 0;

    while (source_ptr < source_limit && destination_ptr < destination_limit) {

        // Fast path for ASCII optimization
        while (source_ptr < source_limit && destination_ptr < destination_limit && *source_ptr < 0x80) {
            *destination_ptr++ = sz_ascii_fold_(*source_ptr++);
            codepoints_read++;
            codepoints_written++;
        }

        // Check if we hit boundaries
        if (source_ptr >= source_limit || destination_ptr >= destination_limit) break;

        // Multi-byte UTF-8 sequence
        sz_rune_t source_rune;
        sz_rune_length_t source_rune_length;
        sz_rune_parse((sz_cptr_t)source_ptr, (sz_cptr_t)source_limit, &source_rune, &source_rune_length);

        // Perform Unicode folding
        sz_rune_t target_runes[3];
        sz_size_t target_runes_count = sz_unicode_fold_codepoint_(source_rune, target_runes);

        // In the worst case scenario, when folded, the text becomes 3x longer.
        // That's the story of 'ΐ' (U+0390, CE 90) becoming three codepoints (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81).
        sz_u8_t target_bytes[12]; // 3 runes max, each up to 4 bytes
        sz_size_t target_bytes_count = 0;
        for (sz_size_t rune_index = 0; rune_index < target_runes_count; ++rune_index)
            target_bytes_count += sz_rune_export(target_runes[rune_index], target_bytes + target_bytes_count);

        if (destination_ptr + target_bytes_count > destination_limit) break;
        for (sz_size_t byte_index = 0; byte_index < target_bytes_count; ++byte_index)
            *destination_ptr++ = target_bytes[byte_index];

        source_ptr += source_rune_length;
        codepoints_read++;
        codepoints_written += target_runes_count;
    }

    if (codepoints_consumed) *codepoints_consumed = codepoints_read;
    if (codepoints_exported) *codepoints_exported = codepoints_written;
    if (bytes_consumed) *bytes_consumed = (sz_size_t)(source_ptr - source_start);
    if (bytes_exported) *bytes_exported = (sz_size_t)(destination_ptr - destination_start);
}

SZ_PUBLIC sz_size_t sz_utf8_uncased_fold_serial(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {

    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;
    sz_u8_t *destination_ptr = (sz_u8_t *)destination;

    // Assumes valid UTF-8 input; use sz_utf8_valid() first if validation is needed.
    while (source_ptr < source_end) {
        // ASCII fast-path: process consecutive ASCII bytes without UTF-8 decode/encode overhead.
        // This handles ~95% of typical text content with minimal branching.
        while (source_ptr < source_end && *source_ptr < 0x80) *destination_ptr++ = sz_ascii_fold_(*source_ptr++);
        if (source_ptr >= source_end) break;

        // Multi-byte UTF-8 sequence: use full decode/fold/encode path
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)source_ptr, (sz_cptr_t)source_end, &rune, &rune_length);
        source_ptr += rune_length;

        sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            destination_ptr += sz_rune_export(folded_runes[rune_index], destination_ptr);
    }

    return (sz_size_t)(destination_ptr - (sz_u8_t *)destination);
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_SERIAL_H_
