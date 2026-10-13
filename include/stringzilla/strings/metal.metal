/**
 *  @file include/stringzilla/strings/metal.metal
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief The string collection kernels on Apple GPUs, step for step those of `strings/simt.cuh`:
 *      measuring every output string, prefix sums of the lengths into offsets, and writing.
 *
 *  Every dispatch binds the strings read as two buffers, their payload and their offsets, then the
 *  order, the scratch, and the payload and offsets written, with the launch record after them. The
 *  scratch holds `count + 1` offsets, the block sums of the scan, and one report per threadgroup of
 *  the measuring dispatch, the first malformed position in it, as Apple7 has no 64-bit atomics.
 *
 *  It compiles behind `utf8_norm/tables.h` and `utf8_norm/metal.metal`, whose
 *  @c sz_utf8_norm_span_metal_ normalizes a string to a thread.
 *
 *  @sa include/stringzilla/strings/metal.h, which embeds and launches this source
 *  @sa include/stringzilla/strings/simt.cuh, the CUDA sibling this mirrors
 */

/** The launch record the host copies in, as @c sz_strings_arguments_metal_t on the host. */
struct sz_strings_arguments_metal_t {
    ulong count, order, layout, target_layout, mode, length, width, stride, tiles;
    uchar lut[256];
};

/** The escape kinds, numbered as @c sz_escape_kind_t numbers them. */
enum sz_escape_kind_metal_t : ulong {

    /** RFC 8259 string bodies. */
    sz_escape_json_metal_k = 0,

    /** Two lowercase hexadecimal digits per byte. */
    sz_escape_hex_metal_k = 1,

    /** HTML and XML text and attribute values. */
    sz_escape_html_metal_k = 2,

    /** RFC 3986 percent-encoding. */
    sz_escape_url_metal_k = 3,

    /** Terminal-safe logs. */
    sz_escape_console_metal_k = 4,

    /** RFC 4180 fields. */
    sz_escape_csv_metal_k = 5,

    /** RFC 8259 string bodies in ASCII alone. */
    sz_escape_json_ascii_metal_k = 6,
};

/** Threads to a threadgroup, and strings to one in the kernels giving each string a simdgroup. */
constant uint sz_strings_threads_metal_k = 256, sz_strings_simdgroups_metal_k = 8;

/** The names of the five XML character references, as @c sz_strings_unescape_html_ matches them. */
constant char sz_strings_entities_metal_[5][6] = {"amp;", "lt;", "gt;", "quot;", "apos;"};

/** The characters those five references stand for. */
constant uchar sz_strings_entity_characters_metal_[5] = {'&', '<', '>', '"', '\''};

/** A set of byte values, one bit each. */
struct sz_strings_byteset_metal_t {

    /** The bits of values 0 to 63, 64 to 127, 128 to 191 and 192 to 255. */
    ulong words[4];
};

/** The first byte of @p text in @p set, or null. */
inline device uchar const *sz_strings_find_metal_(device uchar const *text, ulong length,
                                                  sz_strings_byteset_metal_t set) {
    for (ulong index = 0; index != length; ++index)
        if ((set.words[text[index] >> 6] >> (text[index] & 63)) & 1) return text + index;
    return nullptr;
}

/** Appends @p length bytes at @p written into @p target, or only counts them when it is null. */
template <typename bytes_type_>
inline ulong sz_strings_emit_metal_(device uchar *target, ulong written, bytes_type_ source, ulong length) {
    for (ulong index = 0; target && index != length; ++index) target[written + index] = uchar(source[index]);
    return written + length;
}

/** The lowercase hexadecimal digit of the low four bits of @p value. */
inline uchar sz_strings_hex_lower_metal_(uint value) {
    uint const nibble = value & 15;
    return uchar(nibble < 10 ? '0' + nibble : 'a' - 10 + nibble);
}

/** The uppercase hexadecimal digit of the low four bits of @p value. */
inline uchar sz_strings_hex_upper_metal_(uint value) {
    uint const nibble = value & 15;
    return uchar(nibble < 10 ? '0' + nibble : 'A' - 10 + nibble);
}

/** The value of the hexadecimal digit @p byte of either case, or 16 for any other byte. */
inline uint sz_strings_hex_digit_metal_(uchar byte) {
    uint const lower = byte | 0x20;
    if (byte >= '0' && byte <= '9') return byte - '0';
    if (lower >= 'a' && lower <= 'f') return lower - 'a' + 10;
    return 16;
}

/** Parses @p count hexadecimal digits into @p value, failing on any other byte. */
inline bool sz_strings_hex_parse_metal_(device uchar const *digits, ulong count, thread uint &value) {
    uint parsed = 0;
    for (ulong index = 0; index != count; ++index) {
        uint const digit = sz_strings_hex_digit_metal_(digits[index]);
        if (digit > 15) return false;
        parsed = parsed << 4 | digit;
    }
    value = parsed;
    return true;
}

/** Writes @p rune as UTF-8 into @p bytes, returning 1 to 4, as @c sz_rune_encode does. */
inline ulong sz_strings_encode_rune_metal_(uint rune, thread uchar *bytes) {
    if (rune <= 0x7F) {
        bytes[0] = uchar(rune);
        return 1;
    }
    if (rune <= 0x7FF) {
        bytes[0] = uchar(0xC0 | (rune >> 6)), bytes[1] = uchar(0x80 | (rune & 0x3F));
        return 2;
    }
    if (rune <= 0xFFFF) {
        bytes[0] = uchar(0xE0 | (rune >> 12)), bytes[1] = uchar(0x80 | ((rune >> 6) & 0x3F));
        bytes[2] = uchar(0x80 | (rune & 0x3F));
        return 3;
    }
    bytes[0] = uchar(0xF0 | (rune >> 18)), bytes[1] = uchar(0x80 | ((rune >> 12) & 0x3F));
    bytes[2] = uchar(0x80 | ((rune >> 6) & 0x3F)), bytes[3] = uchar(0x80 | (rune & 0x3F));
    return 4;
}

/** Writes @p rune as `\uXXXX` in lowercase, a surrogate pair beyond U+FFFF, returning 6 or 12. */
inline ulong sz_strings_write_utf16_metal_(uint rune, thread uchar *sequence) {
    uint units[2] = {rune, 0};
    ulong units_count = 1;
    if (rune > 0xFFFF)
        units[0] = 0xD800 + ((rune - 0x10000) >> 10), units[1] = 0xDC00 + ((rune - 0x10000) & 0x3FF), units_count = 2;
    for (ulong index = 0; index != units_count; ++index) {
        uint const unit = units[index];
        thread uchar *const escape = sequence + index * 6;
        escape[0] = '\\', escape[1] = 'u', escape[2] = sz_strings_hex_lower_metal_(unit >> 12);
        escape[3] = sz_strings_hex_lower_metal_(unit >> 8), escape[4] = sz_strings_hex_lower_metal_(unit >> 4);
        escape[5] = sz_strings_hex_lower_metal_(unit);
    }
    return units_count * 6;
}

/** Escapes one string for JSON into @p target, or measures it for a null @p target; for
 *  @c sz_escape_json_ascii_metal_k also DEL and every non-ASCII codepoint, writing U+FFFD for
 *  each ill-formed byte. */
inline ulong sz_strings_escape_json_metal_(device uchar const *source, ulong length, device uchar *target, ulong kind) {
    // `"`, `\` and the C0 controls, and for ASCII output DEL and every byte from 0x80 on
    ulong const high = kind == sz_escape_json_ascii_metal_k ? ~0ul : 0ul;
    sz_strings_byteset_metal_t const set = {
        {0x00000004FFFFFFFFul, 0x0000000010000000ul | (high & 0x8000000000000000ul), high, high}};
    device uchar const *const end = source + length;
    ulong written = 0;
    for (device uchar const *hit; (hit = sz_strings_find_metal_(source, ulong(end - source), set)) != nullptr;) {
        uchar sequence[12] = {'\\'};
        ulong sequence_length = 2, consumed = 1;
        uint rune = *hit;
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
                uint const rune_length = sz_utf8_decode_rune_metal_(hit, ulong(end - hit), 0, rune);
                if (rune_length) consumed = rune_length;
                else rune = 0xFFFD;
            }
            sequence_length = sz_strings_write_utf16_metal_(rune, sequence);
        }
        written = sz_strings_emit_metal_(target, written, source, ulong(hit - source));
        written = sz_strings_emit_metal_(target, written, sequence, sequence_length);
        source = hit + consumed;
    }
    return sz_strings_emit_metal_(target, written, source, ulong(end - source));
}

/** Escapes one string as lowercase hexadecimal digit pairs into @p target, or measures it. */
inline ulong sz_strings_escape_hex_metal_(device uchar const *source, ulong length, device uchar *target) {
    for (ulong index = 0; target && index != length; ++index) {
        uchar const byte = source[index];
        target[index * 2] = sz_strings_hex_lower_metal_(byte >> 4);
        target[index * 2 + 1] = sz_strings_hex_lower_metal_(byte);
    }
    return length * 2;
}

/** Escapes one string for HTML text and attributes into @p target, or measures it. */
inline ulong sz_strings_escape_html_metal_(device uchar const *source, ulong length, device uchar *target) {
    // `"`, `&`, `'`, `<` and `>`
    sz_strings_byteset_metal_t const set = {{0x500000C400000000ul, 0, 0, 0}};
    device uchar const *const end = source + length;
    ulong written = 0;
    for (device uchar const *hit; (hit = sz_strings_find_metal_(source, ulong(end - source), set)) != nullptr;) {
        constant char *entity;
        switch (*hit) {
        case '&': entity = "&amp;"; break;
        case '<': entity = "&lt;"; break;
        case '>': entity = "&gt;"; break;
        case '"': entity = "&quot;"; break;
        default: entity = "&#39;"; break;
        }
        ulong entity_length = 0;
        while (entity[entity_length] != ';') ++entity_length;
        written = sz_strings_emit_metal_(target, written, source, ulong(hit - source));
        written = sz_strings_emit_metal_(target, written, entity, entity_length + 1);
        source = hit + 1;
    }
    return sz_strings_emit_metal_(target, written, source, ulong(end - source));
}

/** Percent-encodes one string for URLs into @p target, or measures it. */
inline ulong sz_strings_escape_url_metal_(device uchar const *source, ulong length, device uchar *target) {
    // Every byte outside `A-Z a-z 0-9 - . _ ~`
    sz_strings_byteset_metal_t const set = {{0xFC009FFFFFFFFFFFul, 0xB800000178000001ul, ~0ul, ~0ul}};
    device uchar const *const end = source + length;
    ulong written = 0;
    for (device uchar const *hit; (hit = sz_strings_find_metal_(source, ulong(end - source), set)) != nullptr;) {
        uchar const byte = *hit;
        uchar const sequence[3] = {'%', sz_strings_hex_upper_metal_(byte >> 4), sz_strings_hex_upper_metal_(byte)};
        written = sz_strings_emit_metal_(target, written, source, ulong(hit - source));
        written = sz_strings_emit_metal_(target, written, sequence, 3);
        source = hit + 1;
    }
    return sz_strings_emit_metal_(target, written, source, ulong(end - source));
}

/** Escapes one string for terminals into @p target, or measures it, keeping well-formed UTF-8
 *  verbatim apart from the C1 controls and the bidirectional overrides. */
inline ulong sz_strings_escape_console_metal_(device uchar const *source, ulong length, device uchar *target) {
    // `"`, `\`, the C0 controls, DEL and every byte from 0x80 on
    sz_strings_byteset_metal_t const set = {{0x00000004FFFFFFFFul, 0x8000000010000000ul, ~0ul, ~0ul}};
    device uchar const *const end = source + length;
    device uchar const *run = source, *cursor = source;
    ulong written = 0;
    for (device uchar const *hit; (hit = sz_strings_find_metal_(cursor, ulong(end - cursor), set)) != nullptr;) {
        uchar sequence[6] = {'\\'};
        ulong sequence_length = 2, consumed = 1;
        uint rune = *hit;
        uint const rune_length = rune >= 0x80 ? sz_utf8_decode_rune_metal_(hit, ulong(end - hit), 0, rune) : 1;
        if (rune == '\n') sequence[1] = 'n';
        else if (rune == '\t') sequence[1] = 't';
        else if (rune == '\r') sequence[1] = 'r';
        else if (rune == '\\' || rune == '"') sequence[1] = uchar(rune);
        else if (rune < 0x80 || !rune_length) {
            uchar const byte = *hit;
            sequence[1] = 'x', sequence[2] = sz_strings_hex_lower_metal_(byte >> 4);
            sequence[3] = sz_strings_hex_lower_metal_(byte), sequence_length = 4;
        }
        else if (rune <= 0x9F || (rune >= 0x202A && rune <= 0x202E) || (rune >= 0x2066 && rune <= 0x2069))
            sequence_length = sz_strings_write_utf16_metal_(rune, sequence), consumed = rune_length;
        else {
            cursor = hit + rune_length;
            continue;
        }
        written = sz_strings_emit_metal_(target, written, run, ulong(hit - run));
        written = sz_strings_emit_metal_(target, written, sequence, sequence_length);
        run = cursor = hit + consumed;
    }
    return sz_strings_emit_metal_(target, written, run, ulong(end - run));
}

/** Escapes one CSV field into @p target, or measures it, quoting it only when it must be. */
inline ulong sz_strings_escape_csv_metal_(device uchar const *source, ulong length, device uchar *target) {
    // `,`, `"`, CR and LF
    sz_strings_byteset_metal_t const set = {{0x0000100400002400ul, 0, 0, 0}};
    constant char *quote = "\"";
    device uchar const *const end = source + length;
    device uchar const *cursor = sz_strings_find_metal_(source, length, set);
    if (!cursor) return sz_strings_emit_metal_(target, 0, source, length);
    ulong written = sz_strings_emit_metal_(target, 0, quote, 1);
    device uchar const *run = source;
    for (; cursor; cursor = sz_strings_find_metal_(cursor, ulong(end - cursor), set)) {
        if (*cursor++ != '"') continue;
        written = sz_strings_emit_metal_(target, written, run, ulong(cursor - run));
        written = sz_strings_emit_metal_(target, written, quote, 1);
        run = cursor;
    }
    written = sz_strings_emit_metal_(target, written, run, ulong(end - run));
    return sz_strings_emit_metal_(target, written, quote, 1);
}

/** Escapes one string for @p kind into @p target, or only measures it for a null @p target. */
inline ulong sz_strings_escape_one_metal_(ulong kind, device uchar const *source, ulong length, device uchar *target) {
    switch (kind) {
    case sz_escape_hex_metal_k: return sz_strings_escape_hex_metal_(source, length, target);
    case sz_escape_html_metal_k: return sz_strings_escape_html_metal_(source, length, target);
    case sz_escape_url_metal_k: return sz_strings_escape_url_metal_(source, length, target);
    case sz_escape_console_metal_k: return sz_strings_escape_console_metal_(source, length, target);
    case sz_escape_csv_metal_k: return sz_strings_escape_csv_metal_(source, length, target);
    default: return sz_strings_escape_json_metal_(source, length, target, kind);
    }
}

/** Unescapes one JSON string body, or for @c sz_escape_console_metal_k a terminal log line, into
 *  @p target, or measures it, returning all ones when it is malformed. */
inline ulong sz_strings_unescape_backslashes_metal_(device uchar const *source, ulong length, device uchar *target,
                                                    ulong kind) {
    // `\`
    sz_strings_byteset_metal_t const set = {{0, 0x0000000010000000ul, 0, 0}};
    bool const console = kind == sz_escape_console_metal_k;
    device uchar const *const end = source + length;
    ulong written = 0;
    for (device uchar const *hit; (hit = sz_strings_find_metal_(source, ulong(end - source), set)) != nullptr;) {
        ulong const left = ulong(end - hit);
        uchar sequence[4];
        ulong sequence_length = 1, consumed = 2;
        uint rune = 0, low = 0;
        if (left < 2) return ~0ul;
        switch (hit[1]) {
        case '"':
        case '\\': sequence[0] = hit[1]; break;
        case 'n': sequence[0] = '\n'; break;
        case 'r': sequence[0] = '\r'; break;
        case 't': sequence[0] = '\t'; break;
        case '/':
            if (console) return ~0ul;
            sequence[0] = '/';
            break;
        case 'b':
            if (console) return ~0ul;
            sequence[0] = '\b';
            break;
        case 'f':
            if (console) return ~0ul;
            sequence[0] = '\f';
            break;
        case 'x':
            if (!console || left < 4 || !sz_strings_hex_parse_metal_(hit + 2, 2, rune)) return ~0ul;
            sequence[0] = uchar(rune), consumed = 4;
            break;
        case 'u':
            if (left < 6 || !sz_strings_hex_parse_metal_(hit + 2, 4, rune) || (rune >= 0xDC00 && rune <= 0xDFFF))
                return ~0ul;
            consumed = 6;
            if (rune >= 0xD800 && rune <= 0xDBFF) {
                if (console || left < 12 || hit[6] != '\\' || hit[7] != 'u' ||
                    !sz_strings_hex_parse_metal_(hit + 8, 4, low) || low < 0xDC00 || low > 0xDFFF)
                    return ~0ul;
                rune = 0x10000 + ((rune - 0xD800) << 10) + (low - 0xDC00), consumed = 12;
            }
            sequence_length = sz_strings_encode_rune_metal_(rune, sequence);
            break;
        default: return ~0ul;
        }
        written = sz_strings_emit_metal_(target, written, source, ulong(hit - source));
        written = sz_strings_emit_metal_(target, written, sequence, sequence_length);
        source = hit + consumed;
    }
    return sz_strings_emit_metal_(target, written, source, ulong(end - source));
}

/** Decodes one string of hexadecimal digit pairs into @p target, or measures it, returning all ones
 *  for an odd length or a byte that is not a digit. */
inline ulong sz_strings_unescape_hex_metal_(device uchar const *source, ulong length, device uchar *target) {
    if (length & 1) return ~0ul;
    for (ulong index = 0; index != length / 2; ++index) {
        uint byte = 0;
        if (!sz_strings_hex_parse_metal_(source + index * 2, 2, byte)) return ~0ul;
        if (target) target[index] = uchar(byte);
    }
    return length / 2;
}

/** Unescapes the HTML character references of one string into @p target, or measures it, copying
 *  every `&` that starts no reference; it is never malformed. */
inline ulong sz_strings_unescape_html_metal_(device uchar const *source, ulong length, device uchar *target) {
    // `&`
    sz_strings_byteset_metal_t const set = {{0x0000004000000000ul, 0, 0, 0}};
    device uchar const *const end = source + length;
    device uchar const *run = source, *cursor = source;
    ulong written = 0;
    for (device uchar const *hit; (hit = sz_strings_find_metal_(cursor, ulong(end - cursor), set)) != nullptr;) {
        ulong const left = ulong(end - hit) - 1;
        uchar sequence[4];
        ulong sequence_length = 0, consumed = 0;
        for (ulong name_index = 0; name_index != 5 && !consumed; ++name_index) {
            constant char *const name = sz_strings_entities_metal_[name_index];
            ulong name_length = 0, matched = 0;
            while (name[name_length]) ++name_length;
            while (left >= name_length && matched != name_length && hit[1 + matched] == uchar(name[matched])) ++matched;
            if (left >= name_length && matched == name_length)
                sequence[0] = sz_strings_entity_characters_metal_[name_index], sequence_length = 1,
                consumed = name_length + 1;
        }
        if (!consumed && left >= 3 && hit[1] == '#') {
            bool const hexadecimal = hit[2] == 'x' || hit[2] == 'X';
            uint const base = hexadecimal ? 16 : 10;
            device uchar const *const digits = hit + 2 + hexadecimal;
            device uchar const *digit = digits;
            uint rune = 0;
            for (; digit != end; ++digit) {
                uint const value = hexadecimal ? sz_strings_hex_digit_metal_(*digit) : uint(*digit) - '0';
                if (value >= base) break;
                // Saturate past the last codepoint, so an overlong reference cannot wrap around.
                rune = min(rune * base + value, 0x110000u);
            }
            if (digit != digits && digit != end && *digit == ';') {
                if (!rune || (rune >= 0xD800 && rune <= 0xDFFF) || rune > 0x10FFFF) rune = 0xFFFD;
                sequence_length = sz_strings_encode_rune_metal_(rune, sequence), consumed = ulong(digit + 1 - hit);
            }
        }
        if (!consumed) {
            cursor = hit + 1;
            continue;
        }
        written = sz_strings_emit_metal_(target, written, run, ulong(hit - run));
        written = sz_strings_emit_metal_(target, written, sequence, sequence_length);
        run = cursor = hit + consumed;
    }
    return sz_strings_emit_metal_(target, written, run, ulong(end - run));
}

/** Decodes the percent-escapes of one string into @p target, or measures it, returning all ones for
 *  a `%` without two hexadecimal digits. */
inline ulong sz_strings_unescape_url_metal_(device uchar const *source, ulong length, device uchar *target) {
    // `%`
    sz_strings_byteset_metal_t const set = {{0x0000002000000000ul, 0, 0, 0}};
    device uchar const *const end = source + length;
    ulong written = 0;
    for (device uchar const *hit; (hit = sz_strings_find_metal_(source, ulong(end - source), set)) != nullptr;) {
        uint byte = 0;
        if (end - hit < 3 || !sz_strings_hex_parse_metal_(hit + 1, 2, byte)) return ~0ul;
        uchar const decoded[1] = {uchar(byte)};
        written = sz_strings_emit_metal_(target, written, source, ulong(hit - source));
        written = sz_strings_emit_metal_(target, written, decoded, 1);
        source = hit + 3;
    }
    return sz_strings_emit_metal_(target, written, source, ulong(end - source));
}

/** Unquotes one CSV field into @p target, or measures it, returning all ones for a quoted field
 *  without its closing quote or with an inner quote that is not doubled. */
inline ulong sz_strings_unescape_csv_metal_(device uchar const *source, ulong length, device uchar *target) {
    // `"`
    sz_strings_byteset_metal_t const set = {{0x0000000400000000ul, 0, 0, 0}};
    if (!length || *source != '"') return sz_strings_emit_metal_(target, 0, source, length);
    if (length < 2 || source[length - 1] != '"') return ~0ul;
    device uchar const *const end = source + length - 1;
    device uchar const *run = source + 1;
    ulong written = 0;
    for (device uchar const *hit; (hit = sz_strings_find_metal_(run, ulong(end - run), set)) != nullptr;) {
        if (hit + 1 == end || hit[1] != '"') return ~0ul;
        written = sz_strings_emit_metal_(target, written, run, ulong(hit + 1 - run));
        run = hit + 2;
    }
    return sz_strings_emit_metal_(target, written, run, ulong(end - run));
}

/** Unescapes one string from @p kind into @p target, or only measures it for a null @p target,
 *  returning all ones when it is malformed. */
inline ulong sz_strings_unescape_one_metal_(ulong kind, device uchar const *source, ulong length,
                                            device uchar *target) {
    switch (kind) {
    case sz_escape_hex_metal_k: return sz_strings_unescape_hex_metal_(source, length, target);
    case sz_escape_html_metal_k: return sz_strings_unescape_html_metal_(source, length, target);
    case sz_escape_url_metal_k: return sz_strings_unescape_url_metal_(source, length, target);
    case sz_escape_csv_metal_k: return sz_strings_unescape_csv_metal_(source, length, target);
    default: return sz_strings_unescape_backslashes_metal_(source, length, target, kind);
    }
}

/** Sets offset @p index of offsets @p layout wide. */
inline void sz_strings_offset_set_metal_(device uchar *offsets, ulong layout, ulong index, ulong offset) {
    device uint *const offsets_u32 = (device uint *)offsets;
    device ulong *const offsets_u64 = (device ulong *)offsets;
    if (layout == sz_strings_u64_tape_metal_k) offsets_u64[index] = offset;
    else offsets_u32[index] = uint(offset);
}

struct sz_strings_scan_arguments_metal_t {
    ulong offset, count, blocks;
};

inline ulong sz_strings_scan_metal_(ulong value, uint lane, threadgroup ulong *totals) {
    uint const low = uint(value);
    uint const low_prefix = simd_prefix_exclusive_sum(low);
    uint const carries = simd_prefix_exclusive_sum(uint(low_prefix + low < low_prefix));
    uint const high_prefix = simd_prefix_exclusive_sum(uint(value >> 32));
    ulong const prefix = (ulong(high_prefix + carries) << 32) | low_prefix;
    if ((lane & 31) == 31) totals[lane / 32] = prefix + value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong preceding = 0;
    for (uint group = 0; group < lane / 32; ++group) preceding += totals[group];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return preceding + prefix;
}

kernel void sz_strings_scan_local_metal_kernel_(device ulong *offsets [[buffer(0)]],
                                                device ulong const *lengths [[buffer(1)]],
                                                device ulong *blocks [[buffer(2)]],
                                                constant sz_strings_scan_arguments_metal_t &arguments [[buffer(3)]],
                                                uint group [[threadgroup_position_in_grid]],
                                                uint lane [[thread_index_in_threadgroup]]) {
    threadgroup ulong totals[8];
    ulong const row = ulong(group) * 256 + lane;
    ulong const length = row < arguments.count ? lengths[row] : 0;
    ulong const prefix = sz_strings_scan_metal_(length, lane, totals);
    if (row < arguments.count) offsets[row] = prefix;
    if (lane == 255) blocks[group] = prefix + length;
}

kernel void sz_strings_scan_blocks_metal_kernel_(device ulong *offsets [[buffer(0)]],
                                                 device ulong const *lengths [[buffer(1)]],
                                                 device ulong *blocks [[buffer(2)]],
                                                 constant sz_strings_scan_arguments_metal_t &arguments [[buffer(3)]],
                                                 uint lane [[thread_index_in_threadgroup]]) {
    threadgroup ulong totals[8];
    ulong const chunk = arguments.blocks / 256 + (arguments.blocks % 256 != 0);
    ulong const first = min(ulong(lane) * chunk, arguments.blocks);
    ulong const end = min(first + chunk, arguments.blocks);
    ulong sum = 0;
    for (ulong block = first; block < end; ++block) {
        ulong const value = blocks[block];
        blocks[block] = sum;
        sum += value;
    }
    ulong const prefix = sz_strings_scan_metal_(sum, lane, totals);
    for (ulong block = first; block < end; ++block) blocks[block] += arguments.offset + prefix;
    if (lane == 255) offsets[arguments.count] = arguments.offset + prefix + sum;
}

kernel void sz_strings_scan_offsets_metal_kernel_(device ulong *offsets [[buffer(0)]],
                                                  device ulong const *lengths [[buffer(1)]],
                                                  device ulong const *blocks [[buffer(2)]],
                                                  constant sz_strings_scan_arguments_metal_t &arguments [[buffer(3)]],
                                                  uint group [[threadgroup_position_in_grid]],
                                                  uint lane [[thread_index_in_threadgroup]]) {
    ulong const row = ulong(group) * 256 + lane;
    if (row < arguments.count) offsets[row] += blocks[group];
}

kernel void sz_strings_lengths_metal_kernel_(device uchar const *data [[buffer(0)]],
                                             device uchar const *offsets [[buffer(1)]],
                                             device ulong const *order [[buffer(2)]],
                                             device ulong *lengths [[buffer(3)]],
                                             constant sz_strings_arguments_metal_t &arguments [[buffer(6)]],
                                             uint group [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_threadgroup]]) {
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_threads_metal_k + lane;
    if (position < arguments.count)
        lengths[position] = sz_strings_length_metal_(strings, arguments.order ? order[position] : position);
}

kernel void sz_strings_escape_lengths_metal_kernel_(device uchar const *data [[buffer(0)]],
                                                    device uchar const *offsets [[buffer(1)]],
                                                    device ulong const *order [[buffer(2)]],
                                                    device ulong *lengths [[buffer(3)]],
                                                    constant sz_strings_arguments_metal_t &arguments [[buffer(6)]],
                                                    uint group [[threadgroup_position_in_grid]],
                                                    uint lane [[thread_index_in_threadgroup]]) {
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_threads_metal_k + lane;
    if (position >= arguments.count) return;
    ulong const index = arguments.order ? order[position] : position;
    lengths[position] = sz_strings_escape_one_metal_(arguments.mode, sz_strings_start_metal_(strings, index),
                                                     sz_strings_length_metal_(strings, index), nullptr);
}

/** Measures unescaped strings, counting a malformed one as empty and reporting the first malformed
 *  position of every threadgroup that holds one. */
kernel void sz_strings_unescape_lengths_metal_kernel_(device uchar const *data [[buffer(0)]],
                                                      device uchar const *offsets [[buffer(1)]],
                                                      device ulong const *order [[buffer(2)]],
                                                      device ulong *lengths [[buffer(3)]],
                                                      constant sz_strings_arguments_metal_t &arguments [[buffer(6)]],
                                                      uint group [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint first_malformed;
    if (lane == 0) atomic_store_explicit(&first_malformed, sz_strings_threads_metal_k, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_threads_metal_k + lane;
    if (position < arguments.count) {
        ulong const index = arguments.order ? order[position] : position;
        ulong const unescaped = sz_strings_unescape_one_metal_(arguments.mode, sz_strings_start_metal_(strings, index),
                                                               sz_strings_length_metal_(strings, index), nullptr);
        if (unescaped == ~0ul) atomic_fetch_min_explicit(&first_malformed, lane, memory_order_relaxed);
        lengths[position] = unescaped == ~0ul ? 0 : unescaped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device ulong *const reports = lengths + arguments.count + 1 + arguments.tiles;
    uint const found = atomic_load_explicit(&first_malformed, memory_order_relaxed);
    if (lane == 0 && found != sz_strings_threads_metal_k)
        reports[group] = ulong(group) * sz_strings_threads_metal_k + found;
}

kernel void sz_strings_escape_metal_kernel_(
    device uchar const *data [[buffer(0)]], device uchar const *offsets [[buffer(1)]],
    device ulong const *order [[buffer(2)]], device ulong const *scanned [[buffer(3)]],
    device uchar *target [[buffer(4)]], device uchar *target_offsets [[buffer(5)]],
    constant sz_strings_arguments_metal_t &arguments [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_threads_metal_k + lane;
    if (position >= arguments.count) return;
    ulong const index = arguments.order ? order[position] : position;
    sz_strings_escape_one_metal_(arguments.mode, sz_strings_start_metal_(strings, index),
                                 sz_strings_length_metal_(strings, index), target + scanned[position]);
    sz_strings_offset_set_metal_(target_offsets, arguments.target_layout, position + 1, scanned[position + 1]);
}

kernel void sz_strings_unescape_metal_kernel_(
    device uchar const *data [[buffer(0)]], device uchar const *offsets [[buffer(1)]],
    device ulong const *order [[buffer(2)]], device ulong const *scanned [[buffer(3)]],
    device uchar *target [[buffer(4)]], device uchar *target_offsets [[buffer(5)]],
    constant sz_strings_arguments_metal_t &arguments [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_threads_metal_k + lane;
    if (position >= arguments.count) return;
    ulong const index = arguments.order ? order[position] : position;
    sz_strings_unescape_one_metal_(arguments.mode, sz_strings_start_metal_(strings, index),
                                   sz_strings_length_metal_(strings, index), target + scanned[position]);
    sz_strings_offset_set_metal_(target_offsets, arguments.target_layout, position + 1, scanned[position + 1]);
}

kernel void sz_strings_normalize_lengths_metal_kernel_(device uchar const *data [[buffer(0)]],
                                                       device uchar const *offsets [[buffer(1)]],
                                                       device ulong const *order [[buffer(2)]],
                                                       device ulong *lengths [[buffer(3)]],
                                                       constant sz_strings_arguments_metal_t &arguments [[buffer(6)]],
                                                       uint group [[threadgroup_position_in_grid]],
                                                       uint lane [[thread_index_in_threadgroup]]) {
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_threads_metal_k + lane;
    if (position >= arguments.count) return;
    ulong const index = arguments.order ? order[position] : position;
    ulong const length = sz_strings_length_metal_(strings, index);
    lengths[position] = sz_utf8_norm_span_metal_(sz_strings_start_metal_(strings, index), length, 0, length,
                                                 uint(arguments.mode), nullptr);
}

kernel void sz_strings_normalize_metal_kernel_(
    device uchar const *data [[buffer(0)]], device uchar const *offsets [[buffer(1)]],
    device ulong const *order [[buffer(2)]], device ulong const *scanned [[buffer(3)]],
    device uchar *target [[buffer(4)]], device uchar *target_offsets [[buffer(5)]],
    constant sz_strings_arguments_metal_t &arguments [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_threads_metal_k + lane;
    if (position >= arguments.count) return;
    ulong const index = arguments.order ? order[position] : position;
    ulong const length = sz_strings_length_metal_(strings, index);
    sz_utf8_norm_span_metal_(sz_strings_start_metal_(strings, index), length, 0, length, uint(arguments.mode),
                             target + scanned[position]);
    sz_strings_offset_set_metal_(target_offsets, arguments.target_layout, position + 1, scanned[position + 1]);
}

/** Copies every string through the table, a simdgroup to a string, where @p scanned places it. */
kernel void sz_strings_gather_metal_kernel_(
    device uchar const *data [[buffer(0)]], device uchar const *offsets [[buffer(1)]],
    device ulong const *order [[buffer(2)]], device ulong const *scanned [[buffer(3)]],
    device uchar *target [[buffer(4)]], device uchar *target_offsets [[buffer(5)]],
    constant sz_strings_arguments_metal_t &arguments [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uchar table[256];
    table[lane] = arguments.lut[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_simdgroups_metal_k + lane / 32;
    uint const simd_lane = lane % 32;
    if (position >= arguments.count) return;
    ulong const index = arguments.order ? order[position] : position;
    device uchar const *const source = sz_strings_start_metal_(strings, index);
    ulong const length = sz_strings_length_metal_(strings, index);
    device uchar *const written = target + scanned[position];
    for (ulong offset = simd_lane; offset < length; offset += 32) written[offset] = table[source[offset]];
    if (simd_lane == 0)
        sz_strings_offset_set_metal_(target_offsets, arguments.target_layout, position + 1, scanned[position + 1]);
}

/** Maps one run of bytes through the table, where @p target may be @p source itself. */
kernel void sz_strings_lookup_metal_kernel_(device uchar const *source [[buffer(0)]],
                                            device uchar *target [[buffer(4)]],
                                            constant sz_strings_arguments_metal_t &arguments [[buffer(6)]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint groups [[threadgroups_per_grid]],
                                            uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uchar table[256];
    table[lane] = arguments.lut[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong const stride = ulong(groups) * sz_strings_threads_metal_k;
    for (ulong offset = ulong(group) * sz_strings_threads_metal_k + lane; offset < arguments.length; offset += stride)
        target[offset] = table[source[offset]];
}

/** Writes the offsets of the strings read, moved down so the first one is zero. */
kernel void sz_strings_rebase_metal_kernel_(device uchar const *data [[buffer(0)]],
                                            device uchar const *offsets [[buffer(1)]],
                                            device uchar *target_offsets [[buffer(5)]],
                                            constant sz_strings_arguments_metal_t &arguments [[buffer(6)]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_threadgroup]]) {
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_threads_metal_k + lane;
    if (position <= arguments.count)
        sz_strings_offset_set_metal_(
            target_offsets, arguments.target_layout, position,
            sz_strings_offset_metal_(strings, position) - sz_strings_offset_metal_(strings, 0));
}

/** Writes every string into its NUL-padded row, a simdgroup to a row, and the longest length of
 *  every threadgroup into @p longest_per_group. */
kernel void sz_strings_to_rows_metal_kernel_(
    device uchar const *data [[buffer(0)]], device uchar const *offsets [[buffer(1)]],
    device ulong const *order [[buffer(2)]], device ulong *longest_per_group [[buffer(3)]],
    device uchar *rows [[buffer(4)]], constant sz_strings_arguments_metal_t &arguments [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup ulong lengths[sz_strings_simdgroups_metal_k];
    sz_strings_tape_metal_t const strings = {data, offsets, arguments.layout};
    ulong const position = ulong(group) * sz_strings_simdgroups_metal_k + lane / 32;
    uint const simd_lane = lane % 32;
    ulong length = 0;
    if (position < arguments.count) {
        ulong const index = arguments.order ? order[position] : position;
        device uchar const *const source = sz_strings_start_metal_(strings, index);
        length = sz_strings_length_metal_(strings, index);
        ulong const kept = min(length, arguments.width);
        device uchar *const row = rows + position * arguments.stride;
        for (ulong offset = simd_lane; offset < arguments.width; offset += 32)
            row[offset] = offset < kept ? source[offset] : 0;
    }
    if (simd_lane == 0) lengths[lane / 32] = length;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0) return;
    ulong longest = 0;
    for (uint simdgroup = 0; simdgroup != sz_strings_simdgroups_metal_k; ++simdgroup)
        longest = max(longest, lengths[simdgroup]);
    longest_per_group[group] = longest;
}

/** Reduces the longest length of every threadgroup of the rows dispatch into @p longest. */
kernel void sz_strings_longest_metal_kernel_(device ulong const *longest_per_group [[buffer(3)]],
                                             device ulong *longest [[buffer(5)]],
                                             constant sz_strings_arguments_metal_t &arguments [[buffer(6)]],
                                             uint lane [[thread_index_in_threadgroup]]) {
    threadgroup ulong shared[256];
    ulong mine = 0;
    for (ulong tile = lane; tile < arguments.tiles; tile += sz_strings_threads_metal_k)
        mine = max(mine, longest_per_group[tile]);
    shared[lane] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_width = 128; half_width; half_width >>= 1) {
        if (lane < half_width) shared[lane] = max(shared[lane], shared[lane + half_width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) *longest = shared[0];
}
