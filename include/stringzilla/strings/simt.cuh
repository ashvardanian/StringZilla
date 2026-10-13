/**
 *  @file include/stringzilla/strings/simt.cuh
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief The string collection kernels CUDA and ROCm share: measuring every output string, prefix
 *      sums of those lengths into tape offsets, and writing every string where its offset points.
 *
 *  A verb measures its output with one thread per string, scans the lengths into offsets, and once
 *  its host side has read the total back and sized the tape, writes every string. Escaping,
 *  unescaping and normalizing run one thread per string through the very helper that measured it,
 *  @c sz_utf8_norm_span_simt_ for the last; copies and lookups run a warp per string, or the whole
 *  grid over one run of bytes when a tape is read in its own order.
 *
 *  The escape helpers repeat those of `serial.h` step for step, with a plain byte-set search in
 *  place of a tier's vectorized one, and decode UTF-8 through the same @c sz_rune_decode.
 *
 *  A verb's scratch holds `count + 1` offsets, which its measuring kernel first fills with lengths,
 *  then the first malformed position and the payload of the strings before it, then the block sums
 *  of the scan.
 *
 *  @sa include/stringzilla/strings.h
 *  @sa include/stringzilla/strings/cuda.cuh
 *  @sa include/stringzilla/strings/rocm.cuh
 */
#ifndef STRINGZILLA_STRINGS_SIMT_CUH_
#define STRINGZILLA_STRINGS_SIMT_CUH_

#include "stringzilla/types.cuh"
#include "stringzilla/strings/serial.h"   // `sz_escape_kind_t`, `sz_rune_decode`, `sz_rune_encode`
#include "stringzilla/utf8_norm/simt.cuh" // `sz_utf8_norm_span_simt_`

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

/** The launch shape every strings kernel shares. */
enum {

    /** Threads to a block, which the block scan and the staged lookup table assume. */
    sz_strings_threads_simt_k = 256,

    /** Lanes sharing a string in the copying kernels: a warp on NVIDIA, half a wavefront on AMD. */
    sz_strings_lanes_simt_k = 32,

    /** Blocks a launch takes at most, within every vendor's grid, its kernel striding past them. */
    sz_strings_blocks_max_simt_k = 65535,
};

/** A 256-entry lookup table, passed to a kernel by value. */
typedef struct sz_strings_lut_simt_t {

    /** The byte each byte value maps to. */
    sz_u8_t bytes[256];
} sz_strings_lut_simt_t;

#pragma region Escaping

/** The names of the five XML character references, as @c sz_strings_unescape_html_ matches them. */
static __constant__ char const sz_strings_entities_simt_[5][6] = {"amp;", "lt;", "gt;", "quot;", "apos;"};

/** The characters those five references stand for. */
static __constant__ char const sz_strings_entity_characters_simt_[5] = {'&', '<', '>', '"', '\''};

/** The first byte of @p text in @p set, or null. */
STRINGZILLA_DEVICE sz_cptr_t sz_strings_find_simt_(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    for (sz_size_t index = 0; index != length; ++index)
        if (sz_byteset_contains_u8(set, (sz_u8_t)text[index])) return text + index;
    return STRINGZILLA_NULL;
}

/** Appends @p length bytes at @p written into @p target, or only counts them when it is null. */
STRINGZILLA_DEVICE sz_size_t sz_strings_emit_simt_(sz_ptr_t target, sz_size_t written, sz_cptr_t source,
                                                   sz_size_t length) {
    for (sz_size_t index = 0; target && index != length; ++index) target[written + index] = source[index];
    return written + length;
}

/** The lowercase hexadecimal digit of the low four bits of @p value. */
STRINGZILLA_DEVICE char sz_strings_hex_lower_simt_(sz_u32_t value) {
    sz_u32_t const nibble = value & 15;
    return (char)(nibble < 10 ? '0' + nibble : 'a' - 10 + nibble);
}

/** The uppercase hexadecimal digit of the low four bits of @p value. */
STRINGZILLA_DEVICE char sz_strings_hex_upper_simt_(sz_u32_t value) {
    sz_u32_t const nibble = value & 15;
    return (char)(nibble < 10 ? '0' + nibble : 'A' - 10 + nibble);
}

/** The value of the hexadecimal digit @p digit of either case, or 16 for any other byte. */
STRINGZILLA_DEVICE sz_u32_t sz_strings_hex_digit_simt_(char digit) {
    sz_u32_t const byte = (sz_u8_t)digit, lower = byte | 0x20;
    if (byte >= '0' && byte <= '9') return byte - '0';
    if (lower >= 'a' && lower <= 'f') return lower - 'a' + 10;
    return 16;
}

/** Parses @p count hexadecimal digits into @p value, failing on any other byte. */
STRINGZILLA_DEVICE sz_bool_t sz_strings_hex_parse_simt_(sz_cptr_t digits, sz_size_t count, sz_u32_t *value) {
    sz_u32_t parsed = 0;
    for (sz_size_t index = 0; index != count; ++index) {
        sz_u32_t const digit = sz_strings_hex_digit_simt_(digits[index]);
        if (digit > 15) return sz_false_k;
        parsed = parsed << 4 | digit;
    }
    *value = parsed;
    return sz_true_k;
}

/** Writes @p rune as `\uXXXX` in lowercase, a surrogate pair beyond U+FFFF, returning 6 or 12. */
STRINGZILLA_DEVICE sz_size_t sz_strings_write_utf16_simt_(sz_rune_t rune, char *sequence) {
    sz_u32_t units[2] = {rune, 0};
    sz_size_t units_count = 1;
    if (rune > 0xFFFF)
        units[0] = 0xD800 + ((rune - 0x10000) >> 10), units[1] = 0xDC00 + ((rune - 0x10000) & 0x3FF), units_count = 2;
    for (sz_size_t index = 0; index != units_count; ++index) {
        sz_u32_t const unit = units[index];
        char *const escape = sequence + index * 6;
        escape[0] = '\\', escape[1] = 'u', escape[2] = sz_strings_hex_lower_simt_(unit >> 12);
        escape[3] = sz_strings_hex_lower_simt_(unit >> 8), escape[4] = sz_strings_hex_lower_simt_(unit >> 4);
        escape[5] = sz_strings_hex_lower_simt_(unit);
    }
    return units_count * 6;
}

/** Escapes one string for JSON into @p target, or measures it for a null @p target; for
 *  @c sz_escape_json_ascii_k also DEL and every non-ASCII codepoint, ill-formed bytes as U+FFFD. */
STRINGZILLA_DEVICE sz_size_t sz_strings_escape_json_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                          sz_escape_kind_t kind) {
    // `"`, `\` and the C0 controls, and for ASCII output DEL and every byte from 0x80 on
    sz_u64_t const high = kind == sz_escape_json_ascii_k ? ~(sz_u64_t)0 : 0;
    sz_byteset_t const set = {
        {0x00000004FFFFFFFFull, 0x0000000010000000ull | (high & 0x8000000000000000ull), high, high}};
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_simt_(source, (sz_size_t)(end - source), &set)) != STRINGZILLA_NULL;) {
        char sequence[12] = {'\\'};
        sz_size_t sequence_length = 2, consumed = 1;
        sz_rune_t rune = (sz_u8_t)*hit;
        switch (rune) {
        case '"': sequence[1] = '"'; break;
        case '\\': sequence[1] = '\\'; break;
        case '\b': sequence[1] = 'b'; break;
        case '\f': sequence[1] = 'f'; break;
        case '\n': sequence[1] = 'n'; break;
        case '\r': sequence[1] = 'r'; break;
        case '\t': sequence[1] = 't'; break;
        default:
            if (rune >= 0x80) {
                sz_rune_length_t const rune_length = sz_rune_decode(hit, end, &rune);
                if (rune_length) consumed = rune_length;
                else rune = 0xFFFD;
            }
            sequence_length = sz_strings_write_utf16_simt_(rune, sequence);
        }
        written = sz_strings_emit_simt_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_simt_(target, written, sequence, sequence_length);
        source = hit + consumed;
    }
    return sz_strings_emit_simt_(target, written, source, (sz_size_t)(end - source));
}

/** Escapes one string as lowercase hexadecimal digit pairs into @p target, or measures it. */
STRINGZILLA_DEVICE sz_size_t sz_strings_escape_hex_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    for (sz_size_t index = 0; target && index != length; ++index) {
        sz_u8_t const byte = (sz_u8_t)source[index];
        target[index * 2] = sz_strings_hex_lower_simt_(byte >> 4);
        target[index * 2 + 1] = sz_strings_hex_lower_simt_(byte);
    }
    return length * 2;
}

/** Escapes one string for HTML text and attributes into @p target, or measures it. */
STRINGZILLA_DEVICE sz_size_t sz_strings_escape_html_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    // `"`, `&`, `'`, `<` and `>`
    sz_byteset_t const set = {{0x500000C400000000ull, 0, 0, 0}};
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_simt_(source, (sz_size_t)(end - source), &set)) != STRINGZILLA_NULL;) {
        sz_cptr_t entity;
        switch (*hit) {
        case '&': entity = "&amp;"; break;
        case '<': entity = "&lt;"; break;
        case '>': entity = "&gt;"; break;
        case '"': entity = "&quot;"; break;
        default: entity = "&#39;"; break;
        }
        sz_size_t entity_length = 0;
        while (entity[entity_length] != ';') ++entity_length;
        written = sz_strings_emit_simt_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_simt_(target, written, entity, entity_length + 1);
        source = hit + 1;
    }
    return sz_strings_emit_simt_(target, written, source, (sz_size_t)(end - source));
}

/** Percent-encodes one string for URLs into @p target, or measures it. */
STRINGZILLA_DEVICE sz_size_t sz_strings_escape_url_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    // Every byte outside `A-Z a-z 0-9 - . _ ~`
    sz_byteset_t const set = {{0xFC009FFFFFFFFFFFull, 0xB800000178000001ull, ~0ull, ~0ull}};
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_simt_(source, (sz_size_t)(end - source), &set)) != STRINGZILLA_NULL;) {
        sz_u8_t const byte = (sz_u8_t)*hit;
        char const sequence[3] = {'%', sz_strings_hex_upper_simt_(byte >> 4), sz_strings_hex_upper_simt_(byte)};
        written = sz_strings_emit_simt_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_simt_(target, written, sequence, 3);
        source = hit + 1;
    }
    return sz_strings_emit_simt_(target, written, source, (sz_size_t)(end - source));
}

/** Escapes one string for terminals into @p target, or measures it, keeping well-formed UTF-8
 *  verbatim apart from the C1 controls and the bidirectional overrides. */
STRINGZILLA_DEVICE sz_size_t sz_strings_escape_console_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    // `"`, `\`, the C0 controls, DEL and every byte from 0x80 on
    sz_byteset_t const set = {{0x00000004FFFFFFFFull, 0x8000000010000000ull, ~0ull, ~0ull}};
    sz_cptr_t const end = source + length;
    sz_cptr_t run = source, cursor = source;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_simt_(cursor, (sz_size_t)(end - cursor), &set)) != STRINGZILLA_NULL;) {
        char sequence[6] = {'\\'};
        sz_size_t sequence_length = 2, consumed = 1;
        sz_rune_t rune = (sz_u8_t)*hit;
        sz_rune_length_t const rune_length = rune >= 0x80 ? sz_rune_decode(hit, end, &rune) : sz_rune_1byte_k;
        if (rune == '\n') sequence[1] = 'n';
        else if (rune == '\t') sequence[1] = 't';
        else if (rune == '\r') sequence[1] = 'r';
        else if (rune == '\\' || rune == '"') sequence[1] = (char)rune;
        else if (rune < 0x80 || !rune_length) {
            sz_u8_t const byte = (sz_u8_t)*hit;
            sequence[1] = 'x', sequence[2] = sz_strings_hex_lower_simt_(byte >> 4);
            sequence[3] = sz_strings_hex_lower_simt_(byte), sequence_length = 4;
        }
        else if (rune <= 0x9F || (rune >= 0x202A && rune <= 0x202E) || (rune >= 0x2066 && rune <= 0x2069))
            sequence_length = sz_strings_write_utf16_simt_(rune, sequence), consumed = rune_length;
        else {
            cursor = hit + rune_length;
            continue;
        }
        written = sz_strings_emit_simt_(target, written, run, (sz_size_t)(hit - run));
        written = sz_strings_emit_simt_(target, written, sequence, sequence_length);
        run = cursor = hit + consumed;
    }
    return sz_strings_emit_simt_(target, written, run, (sz_size_t)(end - run));
}

/** Escapes one CSV field into @p target, or measures it, quoting it only when it must be. */
STRINGZILLA_DEVICE sz_size_t sz_strings_escape_csv_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    // `,`, `"`, CR and LF
    sz_byteset_t const set = {{0x0000100400002400ull, 0, 0, 0}};
    sz_cptr_t const end = source + length;
    sz_cptr_t cursor = sz_strings_find_simt_(source, length, &set);
    if (!cursor) return sz_strings_emit_simt_(target, 0, source, length);
    sz_size_t written = sz_strings_emit_simt_(target, 0, "\"", 1);
    sz_cptr_t run = source;
    for (; cursor; cursor = sz_strings_find_simt_(cursor, (sz_size_t)(end - cursor), &set)) {
        if (*cursor++ != '"') continue;
        written = sz_strings_emit_simt_(target, written, run, (sz_size_t)(cursor - run));
        written = sz_strings_emit_simt_(target, written, "\"", 1);
        run = cursor;
    }
    written = sz_strings_emit_simt_(target, written, run, (sz_size_t)(end - run));
    return sz_strings_emit_simt_(target, written, "\"", 1);
}

/** Escapes one string for @p kind into @p target, or only measures it for a null @p target. */
STRINGZILLA_DEVICE sz_size_t sz_strings_escape_one_simt_(sz_escape_kind_t kind, sz_cptr_t source, sz_size_t length,
                                                         sz_ptr_t target) {
    switch (kind) {
    case sz_escape_hex_k: return sz_strings_escape_hex_simt_(source, length, target);
    case sz_escape_html_k: return sz_strings_escape_html_simt_(source, length, target);
    case sz_escape_url_k: return sz_strings_escape_url_simt_(source, length, target);
    case sz_escape_console_k: return sz_strings_escape_console_simt_(source, length, target);
    case sz_escape_csv_k: return sz_strings_escape_csv_simt_(source, length, target);
    default: return sz_strings_escape_json_simt_(source, length, target, kind);
    }
}

/** Unescapes one JSON string body, or for @c sz_escape_console_k a terminal log line, into
 *  @p target, or measures it, returning @c STRINGZILLA_SIZE_MAX when it is malformed. */
STRINGZILLA_DEVICE sz_size_t sz_strings_unescape_backslashes_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                                   sz_escape_kind_t kind) {
    // `\`
    sz_byteset_t const set = {{0, 0x0000000010000000ull, 0, 0}};
    sz_bool_t const console = (sz_bool_t)(kind == sz_escape_console_k);
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_simt_(source, (sz_size_t)(end - source), &set)) != STRINGZILLA_NULL;) {
        sz_size_t const left = (sz_size_t)(end - hit);
        sz_u8_t sequence[4];
        sz_size_t sequence_length = 1, consumed = 2;
        sz_u32_t rune, low;
        if (left < 2) return STRINGZILLA_SIZE_MAX;
        switch (hit[1]) {
        case '"':
        case '\\': sequence[0] = (sz_u8_t)hit[1]; break;
        case 'n': sequence[0] = '\n'; break;
        case 'r': sequence[0] = '\r'; break;
        case 't': sequence[0] = '\t'; break;
        case '/':
            if (console) return STRINGZILLA_SIZE_MAX;
            sequence[0] = '/';
            break;
        case 'b':
            if (console) return STRINGZILLA_SIZE_MAX;
            sequence[0] = '\b';
            break;
        case 'f':
            if (console) return STRINGZILLA_SIZE_MAX;
            sequence[0] = '\f';
            break;
        case 'x':
            if (!console || left < 4 || !sz_strings_hex_parse_simt_(hit + 2, 2, &rune)) return STRINGZILLA_SIZE_MAX;
            sequence[0] = (sz_u8_t)rune, consumed = 4;
            break;
        case 'u':
            if (left < 6 || !sz_strings_hex_parse_simt_(hit + 2, 4, &rune) || (rune >= 0xDC00 && rune <= 0xDFFF))
                return STRINGZILLA_SIZE_MAX;
            consumed = 6;
            if (rune >= 0xD800 && rune <= 0xDBFF) {
                if (console || left < 12 || hit[6] != '\\' || hit[7] != 'u' ||
                    !sz_strings_hex_parse_simt_(hit + 8, 4, &low) || low < 0xDC00 || low > 0xDFFF)
                    return STRINGZILLA_SIZE_MAX;
                rune = 0x10000 + ((rune - 0xD800) << 10) + (low - 0xDC00), consumed = 12;
            }
            sequence_length = sz_rune_encode(rune, sequence);
            break;
        default: return STRINGZILLA_SIZE_MAX;
        }
        written = sz_strings_emit_simt_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_simt_(target, written, (sz_cptr_t)sequence, sequence_length);
        source = hit + consumed;
    }
    return sz_strings_emit_simt_(target, written, source, (sz_size_t)(end - source));
}

/** Decodes one string of hexadecimal digit pairs into @p target, or measures it, returning
 *  @c STRINGZILLA_SIZE_MAX for an odd length or a byte that is not a digit. */
STRINGZILLA_DEVICE sz_size_t sz_strings_unescape_hex_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    if (length & 1) return STRINGZILLA_SIZE_MAX;
    for (sz_size_t index = 0; index != length / 2; ++index) {
        sz_u32_t byte;
        if (!sz_strings_hex_parse_simt_(source + index * 2, 2, &byte)) return STRINGZILLA_SIZE_MAX;
        if (target) target[index] = (char)byte;
    }
    return length / 2;
}

/** Unescapes the HTML character references of one string into @p target, or measures it, copying
 *  every `&` that starts no reference; it is never malformed. */
STRINGZILLA_DEVICE sz_size_t sz_strings_unescape_html_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    // `&`
    sz_byteset_t const set = {{0x0000004000000000ull, 0, 0, 0}};
    sz_cptr_t const end = source + length;
    sz_cptr_t run = source, cursor = source;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_simt_(cursor, (sz_size_t)(end - cursor), &set)) != STRINGZILLA_NULL;) {
        sz_size_t const left = (sz_size_t)(end - hit) - 1;
        sz_u8_t sequence[4];
        sz_size_t sequence_length = 0, consumed = 0;
        for (sz_size_t name_index = 0; name_index != 5 && !consumed; ++name_index) {
            sz_cptr_t const name = sz_strings_entities_simt_[name_index];
            sz_size_t name_length = 0, matched = 0;
            while (name[name_length]) ++name_length;
            while (left >= name_length && matched != name_length && hit[1 + matched] == name[matched]) ++matched;
            if (left >= name_length && matched == name_length)
                sequence[0] = (sz_u8_t)sz_strings_entity_characters_simt_[name_index], sequence_length = 1,
                consumed = name_length + 1;
        }
        if (!consumed && left >= 3 && hit[1] == '#') {
            sz_bool_t const hexadecimal = (sz_bool_t)(hit[2] == 'x' || hit[2] == 'X');
            sz_u32_t const base = hexadecimal ? 16 : 10;
            sz_cptr_t const digits = hit + 2 + hexadecimal;
            sz_cptr_t digit = digits;
            sz_u32_t rune = 0;
            for (; digit != end; ++digit) {
                sz_u32_t const value = hexadecimal ? sz_strings_hex_digit_simt_(*digit)
                                                   : (sz_u32_t)((sz_u8_t)*digit - '0');
                if (value >= base) break;
                // Saturate past the last codepoint, so an overlong reference cannot wrap around.
                rune = sz_min_of_two(rune * base + value, (sz_u32_t)0x110000);
            }
            if (digit != digits && digit != end && *digit == ';') {
                if (!rune || (rune >= 0xD800 && rune <= 0xDFFF) || rune > 0x10FFFF) rune = 0xFFFD;
                sequence_length = sz_rune_encode(rune, sequence), consumed = (sz_size_t)(digit + 1 - hit);
            }
        }
        if (!consumed) {
            cursor = hit + 1;
            continue;
        }
        written = sz_strings_emit_simt_(target, written, run, (sz_size_t)(hit - run));
        written = sz_strings_emit_simt_(target, written, (sz_cptr_t)sequence, sequence_length);
        run = cursor = hit + consumed;
    }
    return sz_strings_emit_simt_(target, written, run, (sz_size_t)(end - run));
}

/** Decodes the percent-escapes of one string into @p target, or measures it, returning
 *  @c STRINGZILLA_SIZE_MAX for a `%` without two hexadecimal digits. */
STRINGZILLA_DEVICE sz_size_t sz_strings_unescape_url_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    // `%`
    sz_byteset_t const set = {{0x0000002000000000ull, 0, 0, 0}};
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_simt_(source, (sz_size_t)(end - source), &set)) != STRINGZILLA_NULL;) {
        sz_u32_t byte;
        if (end - hit < 3 || !sz_strings_hex_parse_simt_(hit + 1, 2, &byte)) return STRINGZILLA_SIZE_MAX;
        char const decoded = (char)byte;
        written = sz_strings_emit_simt_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_simt_(target, written, &decoded, 1);
        source = hit + 3;
    }
    return sz_strings_emit_simt_(target, written, source, (sz_size_t)(end - source));
}

/** Unquotes one CSV field into @p target, or measures it, returning @c STRINGZILLA_SIZE_MAX for a
 *  quoted field without its closing quote or with an inner quote that is not doubled. */
STRINGZILLA_DEVICE sz_size_t sz_strings_unescape_csv_simt_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    // `"`
    sz_byteset_t const set = {{0x0000000400000000ull, 0, 0, 0}};
    if (!length || *source != '"') return sz_strings_emit_simt_(target, 0, source, length);
    if (length < 2 || source[length - 1] != '"') return STRINGZILLA_SIZE_MAX;
    sz_cptr_t const end = source + length - 1;
    sz_cptr_t run = source + 1;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_simt_(run, (sz_size_t)(end - run), &set)) != STRINGZILLA_NULL;) {
        if (hit + 1 == end || hit[1] != '"') return STRINGZILLA_SIZE_MAX;
        written = sz_strings_emit_simt_(target, written, run, (sz_size_t)(hit + 1 - run));
        run = hit + 2;
    }
    return sz_strings_emit_simt_(target, written, run, (sz_size_t)(end - run));
}

/** Unescapes one string from @p kind into @p target, or only measures it for a null @p target,
 *  returning @c STRINGZILLA_SIZE_MAX when it is malformed. */
STRINGZILLA_DEVICE sz_size_t sz_strings_unescape_one_simt_(sz_escape_kind_t kind, sz_cptr_t source, sz_size_t length,
                                                           sz_ptr_t target) {
    switch (kind) {
    case sz_escape_hex_k: return sz_strings_unescape_hex_simt_(source, length, target);
    case sz_escape_html_k: return sz_strings_unescape_html_simt_(source, length, target);
    case sz_escape_url_k: return sz_strings_unescape_url_simt_(source, length, target);
    case sz_escape_csv_k: return sz_strings_unescape_csv_simt_(source, length, target);
    default: return sz_strings_unescape_backslashes_simt_(source, length, target, kind);
    }
}

#pragma endregion Escaping

#pragma region Kernels

/** The payload block of @p tape. */
STRINGZILLA_DEVICE sz_ptr_t sz_strings_payload_simt_(sz_strings_span_t tape) {
    return tape.layout == sz_strings_u64_tape_k ? tape.data.u64_tape.data : tape.data.u32_tape.data;
}

/** Sets offset @p index of @p tape. */
STRINGZILLA_DEVICE void sz_strings_offset_set_simt_(sz_strings_span_t tape, sz_size_t index, sz_size_t offset) {
    if (tape.layout == sz_strings_u64_tape_k) tape.data.u64_tape.offsets[index] = offset;
    else tape.data.u32_tape.offsets[index] = (sz_u32_t)offset;
}

static __global__ void sz_strings_scan_local_simt_kernel_(sz_u64_t *offsets, sz_u64_t const *lengths, sz_size_t count,
                                                          sz_size_t *blocks) {
    __shared__ sz_size_t shared[sz_strings_threads_simt_k];
    sz_size_t const row = (sz_size_t)blockIdx.x * sz_strings_threads_simt_k + threadIdx.x;
    sz_size_t total = 0;
    sz_size_t const before = sz_block_scan_simt_(row < count ? lengths[row] : 0, shared, &total);
    if (row < count) offsets[row] = before;
    if (!threadIdx.x) blocks[blockIdx.x] = total;
}

static __global__ void sz_strings_scan_blocks_simt_kernel_(sz_size_t *blocks, sz_size_t block_count, sz_u64_t *offsets,
                                                           sz_size_t count, sz_size_t target_offset) {
    __shared__ sz_size_t shared[sz_strings_threads_simt_k];
    sz_size_t const chunk = block_count / sz_strings_threads_simt_k + (block_count % sz_strings_threads_simt_k != 0);
    sz_size_t const first = threadIdx.x * chunk, end = sz_min_of_two(first + chunk, block_count);
    sz_size_t mine = 0, total = 0;
    for (sz_size_t index = first; index < end; ++index) mine += blocks[index];
    sz_size_t running = target_offset + sz_block_scan_simt_(mine, shared, &total);
    for (sz_size_t index = first; index < end; ++index) {
        sz_size_t const value = blocks[index];
        blocks[index] = running;
        running += value;
    }
    if (!threadIdx.x) offsets[count] = target_offset + total;
}

static __global__ void sz_strings_scan_offsets_simt_kernel_(sz_u64_t *offsets, sz_size_t count,
                                                            sz_size_t const *blocks) {
    sz_size_t const row = (sz_size_t)blockIdx.x * sz_strings_threads_simt_k + threadIdx.x;
    if (row < count) offsets[row] += blocks[blockIdx.x];
}

/** Puts the payload of the strings before the first malformed one right after that position. */
static __global__ void sz_strings_payload_simt_kernel_(sz_u64_t *offsets, sz_size_t count) {
    sz_u64_t const malformed = offsets[count + 1];
    offsets[count + 2] = offsets[malformed < count ? malformed : count];
}

static __global__ void sz_strings_lengths_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                       sz_size_t count, sz_u64_t *lengths) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t position = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; position < count; position += stride)
        lengths[position] = sz_strings_length_simt_(strings, order ? order[position] : position);
}

static __global__ void sz_strings_escape_lengths_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                              sz_size_t count, sz_escape_kind_t kind,
                                                              sz_u64_t *lengths) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t position = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; position < count; position += stride) {
        sz_size_t const index = order ? order[position] : position;
        lengths[position] = sz_strings_escape_one_simt_(kind, sz_strings_start_simt_(strings, index),
                                                        sz_strings_length_simt_(strings, index), STRINGZILLA_NULL);
    }
}

/** Measures unescaped strings, taking a malformed one as empty and keeping the earliest. */
static __global__ void sz_strings_unescape_lengths_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                                sz_size_t count, sz_escape_kind_t kind,
                                                                sz_u64_t *lengths) {
    unsigned long long *const malformed = (unsigned long long *)(lengths + count + 1);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t position = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; position < count; position += stride) {
        sz_size_t const index = order ? order[position] : position;
        sz_size_t const unescaped = sz_strings_unescape_one_simt_(
            kind, sz_strings_start_simt_(strings, index), sz_strings_length_simt_(strings, index), STRINGZILLA_NULL);
        if (unescaped == STRINGZILLA_SIZE_MAX) atomicMin(malformed, (unsigned long long)position);
        lengths[position] = unescaped == STRINGZILLA_SIZE_MAX ? 0 : unescaped;
    }
}

static __global__ void sz_strings_escape_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                      sz_size_t count, sz_escape_kind_t kind, sz_u64_t const *offsets,
                                                      sz_strings_span_t tape) {
    sz_ptr_t const payload = sz_strings_payload_simt_(tape);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t position = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; position < count; position += stride) {
        sz_size_t const index = order ? order[position] : position;
        sz_strings_escape_one_simt_(kind, sz_strings_start_simt_(strings, index),
                                    sz_strings_length_simt_(strings, index), payload + offsets[position]);
        sz_strings_offset_set_simt_(tape, position + 1, offsets[position + 1]);
    }
}

static __global__ void sz_strings_unescape_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                        sz_size_t count, sz_escape_kind_t kind, sz_u64_t const *offsets,
                                                        sz_strings_span_t tape) {
    sz_ptr_t const payload = sz_strings_payload_simt_(tape);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t position = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; position < count; position += stride) {
        sz_size_t const index = order ? order[position] : position;
        sz_strings_unescape_one_simt_(kind, sz_strings_start_simt_(strings, index),
                                      sz_strings_length_simt_(strings, index), payload + offsets[position]);
        sz_strings_offset_set_simt_(tape, position + 1, offsets[position + 1]);
    }
}

/** Copies every string through @p lut, a warp to a string, where @p offsets places it. */
static __global__ void sz_strings_gather_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                      sz_size_t count, sz_strings_lut_simt_t lut,
                                                      sz_u64_t const *offsets, sz_strings_span_t tape) {
    __shared__ sz_u8_t table[sz_strings_threads_simt_k];
    table[threadIdx.x] = lut.bytes[threadIdx.x];
    __syncthreads();
    sz_ptr_t const payload = sz_strings_payload_simt_(tape);
    sz_size_t const lane = threadIdx.x % sz_strings_lanes_simt_k;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x / sz_strings_lanes_simt_k;
    for (sz_size_t position = ((sz_size_t)blockIdx.x * blockDim.x + threadIdx.x) / sz_strings_lanes_simt_k;
         position < count; position += stride) {
        sz_size_t const index = order ? order[position] : position;
        sz_u8_t const *const source = (sz_u8_t const *)sz_strings_start_simt_(strings, index);
        sz_size_t const length = sz_strings_length_simt_(strings, index);
        sz_u8_t *const target = (sz_u8_t *)payload + offsets[position];
        for (sz_size_t offset = lane; offset < length; offset += sz_strings_lanes_simt_k)
            target[offset] = table[source[offset]];
        if (!lane) sz_strings_offset_set_simt_(tape, position + 1, offsets[position + 1]);
    }
}

/** Maps one run of bytes through @p lut, where @p target may be @p source itself. */
static __global__ void sz_strings_lookup_simt_kernel_(sz_u8_t const *source, sz_size_t length,
                                                      sz_strings_lut_simt_t lut, sz_u8_t *target) {
    __shared__ sz_u8_t table[sz_strings_threads_simt_k];
    table[threadIdx.x] = lut.bytes[threadIdx.x];
    __syncthreads();
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t offset = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; offset < length; offset += stride)
        target[offset] = table[source[offset]];
}

/** Writes the offsets of @p strings into @p tape, moved down so the first one is zero. */
static __global__ void sz_strings_rebase_simt_kernel_(sz_strings_view_t strings, sz_size_t count,
                                                      sz_strings_span_t tape) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t position = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; position <= count; position += stride)
        sz_strings_offset_set_simt_(
            tape, position,
            (sz_size_t)(sz_strings_start_simt_(strings, position) - sz_strings_start_simt_(strings, 0)));
}

/** Writes every string into its NUL-padded row, a warp to a row, raising @p longest to fit it. */
static __global__ void sz_strings_to_rows_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                       sz_size_t count, sz_strings_rows_t rows, sz_size_t *longest) {
    unsigned long long *const longest_slot = (unsigned long long *)longest;
    sz_size_t const lane = threadIdx.x % sz_strings_lanes_simt_k;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x / sz_strings_lanes_simt_k;
    for (sz_size_t position = ((sz_size_t)blockIdx.x * blockDim.x + threadIdx.x) / sz_strings_lanes_simt_k;
         position < count; position += stride) {
        sz_size_t const index = order ? order[position] : position;
        sz_cptr_t const source = sz_strings_start_simt_(strings, index);
        sz_size_t const length = sz_strings_length_simt_(strings, index);
        sz_size_t const kept = sz_min_of_two(length, rows.width);
        sz_ptr_t const row = rows.data + position * rows.stride;
        for (sz_size_t offset = lane; offset < rows.width; offset += sz_strings_lanes_simt_k)
            row[offset] = offset < kept ? source[offset] : 0;
        if (!lane) atomicMax(longest_slot, (unsigned long long)length);
    }
}

#if STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM

static __global__ void sz_strings_normalize_lengths_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                                 sz_size_t count, sz_normal_form_t form,
                                                                 sz_u64_t *lengths) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t position = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; position < count; position += stride) {
        sz_size_t const index = order ? order[position] : position;
        sz_size_t const length = sz_strings_length_simt_(strings, index);
        lengths[position] = sz_utf8_norm_span_simt_((sz_u8_t const *)sz_strings_start_simt_(strings, index), length, 0,
                                                    length, form, STRINGZILLA_NULL);
    }
}

static __global__ void sz_strings_normalize_simt_kernel_(sz_strings_view_t strings, sz_size_t const *order,
                                                         sz_size_t count, sz_normal_form_t form,
                                                         sz_u64_t const *offsets, sz_strings_span_t tape) {
    sz_u8_t *const payload = (sz_u8_t *)sz_strings_payload_simt_(tape);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t position = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; position < count; position += stride) {
        sz_size_t const index = order ? order[position] : position;
        sz_size_t const length = sz_strings_length_simt_(strings, index);
        sz_utf8_norm_span_simt_((sz_u8_t const *)sz_strings_start_simt_(strings, index), length, 0, length, form,
                                payload + offsets[position]);
        sz_strings_offset_set_simt_(tape, position + 1, offsets[position + 1]);
    }
}

#endif // STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM

#pragma endregion Kernels

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_STRINGS_SIMT_CUH_
