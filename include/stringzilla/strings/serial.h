/**
 *  @file include/stringzilla/strings/serial.h
 *  @author Ash Vardanian
 *  @date January 1, 2024
 *  @brief Serial backend for collections of strings, and the drivers every other tier shares.
 *
 *  Every verb but normalization measures its output, sizes the tape once, and then writes it, while
 *  normalization writes each string once into a payload of its own that it grows. Every failure
 *  comes before the first write into a block the tape holds, so a failed call leaves the tape as it
 *  was. Tiers differ only in the @ref sz_strings_walks_t they hand to the drivers: a byte-set
 *  search that skips the bytes needing no escaping, and a lookup-table kernel. Normalization takes
 *  the tier's own normalizer from <tt>utf8_norm/</tt> instead.
 *
 *  @sa include/stringzilla/strings.h
 */
/*  Above the guard: entered first, this body must finish before the hub includes the tiers. */
#include "stringzilla/strings.h"

#ifndef STRINGZILLA_STRINGS_SERIAL_H_
#define STRINGZILLA_STRINGS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/find/serial.h"       // `sz_find_byteset_serial_`, `sz_equal_serial_`
#include "stringzilla/utf8_runes/serial.h" // `sz_rune_decode`, `sz_rune_encode`
#include "stringzilla/utf8_norm/serial.h"  // `sz_utf8_norm_serial`, `sz_utf8_norm_decomp_max_k`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Walks

/** The stages a tier may replace, handed to every driver so one tier's kernels reach all verbs. */
typedef struct sz_strings_walks_t {

    /** The first byte of a haystack in a set, or null, skipping the bytes that need no escaping. */
    sz_cptr_t (*find_byteset)(sz_cptr_t, sz_size_t, sz_byteset_t const *);

    /** Maps bytes through a 256-entry table, into the same buffer or a disjoint one. */
    void (*lookup_bytes)(sz_ptr_t, sz_cptr_t, sz_size_t, char const *);
} sz_strings_walks_t;

STRINGZILLA_INLINE void sz_strings_lookup_bytes_serial_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                        char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);
    sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
    sz_u8_t const *source_u8 = (sz_u8_t const *)source;
    sz_u8_t *target_u8 = (sz_u8_t *)target;
    sz_u8_t const *source_end = source_u8 + length;
    for (; source_u8 != source_end; ++source_u8, ++target_u8) *target_u8 = lut_u8[*source_u8];
}

/** The serial stages, which every tier starts from. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_serial_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_serial_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_serial_;
    return walks;
}

#pragma endregion Walks

#pragma region Tapes

/** Writes the views of up to 64 output positions from @p start on, returning how many. */
STRINGZILLA_INLINE sz_size_t sz_strings_block_(sz_strings_view_t const *strings, sz_size_t const *order,
                                               sz_size_t count, sz_size_t start, sz_string_view_t *views) {
    sz_size_t const block_count = sz_min_of_two(count - start, (sz_size_t)64);
    if (order) {
        sz_strings_get_(strings, order + start, block_count, views);
        return block_count;
    }
    sz_size_t indices[64];
    for (sz_size_t position = 0; position != block_count; ++position) indices[position] = start + position;
    sz_strings_get_(strings, indices, block_count, views);
    return block_count;
}

/** Offset @p index of the tape @p span holds. */
STRINGZILLA_INLINE sz_size_t sz_strings_offset_(sz_strings_span_t const *span, sz_size_t index) {
    if (span->layout == sz_strings_u64_tape_k) return (sz_size_t)span->data.u64_tape.offsets[index];
    return span->data.u32_tape.offsets[index];
}

/** Sets offset @p index of the tape @p span holds. */
STRINGZILLA_INLINE void sz_strings_offset_set_(sz_strings_span_t *span, sz_size_t index, sz_size_t offset) {
    if (span->layout == sz_strings_u64_tape_k) span->data.u64_tape.offsets[index] = offset;
    else span->data.u32_tape.offsets[index] = (sz_u32_t)offset;
}

/** The payload block of @p span. */
STRINGZILLA_INLINE sz_ptr_t sz_strings_payload_(sz_strings_span_t const *span) {
    return span->layout == sz_strings_u64_tape_k ? span->data.u64_tape.data : span->data.u32_tape.data;
}

/** The offsets block of @p span, null for a zeroed span that holds no blocks. */
STRINGZILLA_INLINE void *sz_strings_offsets_block_(sz_strings_span_t const *span) {
    return span->layout == sz_strings_u64_tape_k ? (void *)span->data.u64_tape.offsets
                                                 : (void *)span->data.u32_tape.offsets;
}

/** Points @p span at a payload and an offsets block, in the layout it already names. */
STRINGZILLA_INLINE void sz_strings_point_(sz_strings_span_t *span, void *payload, void *offsets) {
    if (span->layout == sz_strings_u64_tape_k)
        span->data.u64_tape.data = (sz_ptr_t)payload, span->data.u64_tape.offsets = (sz_u64_t *)offsets;
    else span->data.u32_tape.data = (sz_ptr_t)payload, span->data.u32_tape.offsets = (sz_u32_t *)offsets;
}

/** Bytes in the offsets block of @p count strings in @p layout. */
STRINGZILLA_INLINE sz_size_t sz_strings_offsets_bytes_(sz_strings_layout_t layout, sz_size_t count) {
    return (count + 1) * (layout == sz_strings_u64_tape_k ? sizeof(sz_u64_t) : sizeof(sz_u32_t));
}

/** The layout of a written tape: 32-bit offsets while the @p payload fits them, 64-bit beyond. */
STRINGZILLA_INLINE sz_strings_layout_t sz_strings_layout_for_(sz_size_t payload) {
    return (payload >> 16 >> 16) ? sz_strings_u64_tape_k : sz_strings_u32_tape_k;
}

/** Whether @p strings reads the bytes or the offsets of the blocks @p tape holds. */
STRINGZILLA_INLINE sz_bool_t sz_strings_shares_(sz_strings_view_t const *strings, sz_strings_span_t const *tape) {
    void const *const offsets = sz_strings_offsets_block_(tape);
    if (strings->layout == sz_strings_accessor_k || !offsets) return sz_false_k;
    sz_cptr_t const data = strings->layout == sz_strings_u64_tape_k ? strings->data.u64_tape.data
                                                                    : strings->data.u32_tape.data;
    void const *const strings_offsets = strings->layout == sz_strings_u64_tape_k
                                            ? (void const *)strings->data.u64_tape.offsets
                                            : (void const *)strings->data.u32_tape.offsets;
    return (sz_bool_t)((data && data == sz_strings_payload_(tape)) || strings_offsets == offsets);
}

/** Whether @p strings is the very tape @p tape holds: the same layout, payload and offsets. */
STRINGZILLA_INLINE sz_bool_t sz_strings_aliases_(sz_strings_view_t const *strings, sz_strings_span_t const *tape) {
    if (strings->layout != tape->layout || !sz_strings_offsets_block_(tape)) return sz_false_k;
    if (tape->layout == sz_strings_u64_tape_k)
        return (sz_bool_t)(strings->data.u64_tape.data == tape->data.u64_tape.data &&
                           strings->data.u64_tape.offsets == tape->data.u64_tape.offsets);
    return (sz_bool_t)(strings->data.u32_tape.data == tape->data.u32_tape.data &&
                       strings->data.u32_tape.offsets == tape->data.u32_tape.offsets);
}

/** Gives @p block @p new_bytes and keeps its first bytes: through @c reallocate when the allocator
 *  has one and it succeeds, or else through a fresh block. On failure @p block stays unchanged. */
STRINGZILLA_INLINE sz_bool_t sz_strings_resize_(void *block, sz_size_t old_bytes, sz_size_t new_bytes,
                                                sz_allocator_t *allocator, sz_stream_t stream, void **resized) {
    if (old_bytes == new_bytes) {
        *resized = block;
        return sz_true_k;
    }
    if (block && allocator->reallocate) {
        void *const result = allocator->reallocate(block, old_bytes, new_bytes, allocator->handle, stream);
        if (result || !new_bytes) {
            *resized = result;
            return sz_true_k;
        }
    }
    void *const fresh = new_bytes ? allocator->allocate(new_bytes, allocator->handle, stream) : STRINGZILLA_NULL;
    if (new_bytes && !fresh) return sz_false_k;
    if (block && fresh) memcpy(fresh, block, sz_min_of_two(old_bytes, new_bytes));
    if (block) allocator->free(block, old_bytes, allocator->handle, stream);
    *resized = fresh;
    return sz_true_k;
}

/** Whether the blocks @p tape holds fit @p count strings over @p payload bytes exactly, and
 *  @p strings does not read them, so a call may write straight into them. */
STRINGZILLA_INLINE sz_bool_t sz_strings_reusable_(sz_strings_view_t const *strings, sz_strings_span_t const *tape,
                                                  sz_size_t count, sz_size_t payload) {
    return (sz_bool_t)(sz_strings_offsets_block_(tape) && !sz_strings_shares_(strings, tape) &&
                       tape->layout == sz_strings_layout_for_(payload) && tape->count == count &&
                       sz_strings_offset_(tape, count) == payload);
}

/**
 *  @brief Points @p output at blocks for @p count strings over @p payload bytes without touching
 *      @p tape: its blocks when reusable, or else fresh ones from @p allocator on @p stream.
 *  @return @c sz_success_k, or @c sz_bad_alloc_k holding no new block.
 */
STRINGZILLA_INLINE sz_status_t sz_strings_acquire_(sz_strings_view_t const *strings, sz_strings_span_t const *tape,
                                                   sz_size_t count, sz_size_t payload, sz_allocator_t *allocator,
                                                   sz_strings_span_t *output, sz_stream_t stream) {
    if (count >= STRINGZILLA_SIZE_MAX / sizeof(sz_u64_t)) return sz_bad_alloc_k;
    output->layout = sz_strings_layout_for_(payload), output->count = count;
    if (sz_strings_reusable_(strings, tape, count, payload)) {
        *output = *tape;
        return sz_success_k;
    }
    void *const data = payload ? allocator->allocate(payload, allocator->handle, stream) : STRINGZILLA_NULL;
    if (payload && !data) return sz_bad_alloc_k;
    void *const offsets = allocator->allocate(sz_strings_offsets_bytes_(output->layout, count), allocator->handle,
                                              stream);
    if (!offsets) {
        if (data) allocator->free(data, payload, allocator->handle, stream);
        return sz_bad_alloc_k;
    }
    sz_strings_point_(output, data, offsets);
    return sz_success_k;
}

/** Hands the written @p output to @p tape, freeing the blocks @p tape held on @p stream unless
 *  @p output reuses them. */
STRINGZILLA_INLINE void sz_strings_commit_(sz_strings_span_t *tape, sz_strings_span_t const *output,
                                           sz_allocator_t *allocator, sz_stream_t stream) {
    if (sz_strings_offsets_block_(tape) != sz_strings_offsets_block_(output))
        sz_strings_span_free(tape, allocator, stream);
    *tape = *output;
}

/** Frees the blocks a failed call acquired into @p output, @p payload bytes of them, unless it
 *  holds none, as when acquiring failed or never ran, or the very blocks @p tape holds. */
STRINGZILLA_INLINE void sz_strings_release_(sz_strings_span_t const *tape, sz_strings_span_t const *output,
                                            sz_size_t payload, sz_allocator_t *allocator, sz_stream_t stream) {
    void *const offsets = sz_strings_offsets_block_(output);
    if (!offsets || offsets == sz_strings_offsets_block_(tape)) return;
    if (sz_strings_payload_(output)) allocator->free(sz_strings_payload_(output), payload, allocator->handle, stream);
    allocator->free(offsets, sz_strings_offsets_bytes_(output->layout, output->count), allocator->handle, stream);
}

/** Moves the first @p written offsets of @p span into a new block of @p layout and frees the old
 *  one, or leaves @p span as it was when @p allocator refuses the block. */
STRINGZILLA_INLINE sz_bool_t sz_strings_relayout_(sz_strings_span_t *span, sz_size_t written,
                                                  sz_strings_layout_t layout, sz_allocator_t *allocator) {
    void *const offsets = allocator->allocate(sz_strings_offsets_bytes_(layout, span->count), allocator->handle,
                                              STRINGZILLA_NULL);
    if (!offsets) return sz_false_k;
    sz_strings_span_t moved = *span;
    moved.layout = layout;
    sz_strings_point_(&moved, sz_strings_payload_(span), offsets);
    for (sz_size_t index = 0; index != written; ++index)
        sz_strings_offset_set_(&moved, index, sz_strings_offset_(span, index));
    allocator->free(sz_strings_offsets_block_(span), sz_strings_offsets_bytes_(span->layout, span->count),
                    allocator->handle, STRINGZILLA_NULL);
    *span = moved;
    return sz_true_k;
}

/** Gathers strings into a tape, mapping every byte through a non-null @p lut. A tape read in its
 *  own order is one run of bytes, measured by its offsets and mapped in a single call. */
STRINGZILLA_INLINE sz_status_t sz_strings_gather_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                  sz_size_t count, char const *lut, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_strings_walks_t const *walks) {
    sz_bool_t const flat = (sz_bool_t)(!order && count && strings->layout != sz_strings_accessor_k);
    sz_size_t const base = flat ? sz_strings_view_offset_(strings, 0) : 0;
    sz_size_t payload = flat ? sz_strings_view_offset_(strings, count) - base : 0;
    sz_string_view_t views[64];
    for (sz_size_t start = 0; !flat && start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            if (views[position].length > STRINGZILLA_SIZE_MAX - payload) return sz_bad_alloc_k;
            payload += views[position].length;
        }
    }
    sz_strings_span_t output;
    sz_status_t const status = sz_strings_acquire_(strings, tape, count, payload, allocator, &output, STRINGZILLA_NULL);
    if (status != sz_success_k) return status;

    sz_ptr_t const data = sz_strings_payload_(&output);
    sz_strings_offset_set_(&output, 0, 0);
    for (sz_size_t index = 1; flat && index <= count; ++index)
        sz_strings_offset_set_(&output, index, sz_strings_view_offset_(strings, index) - base);
    if (flat && payload && lut) walks->lookup_bytes(data, sz_strings_view_data_(strings) + base, payload, lut);
    else if (flat && payload) memcpy(data, sz_strings_view_data_(strings) + base, payload);
    sz_size_t offset = 0;
    for (sz_size_t start = 0; !flat && start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_size_t const length = views[position].length;
            if (length && lut) walks->lookup_bytes(data + offset, views[position].start, length, lut);
            else if (length) memcpy(data + offset, views[position].start, length);
            offset += length;
            sz_strings_offset_set_(&output, start + position + 1, offset);
        }
    }
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

#pragma endregion Tapes

#pragma region Escaping

/** The first byte of @p text in @p set, or null. A short run is probed by hand first, so dense
 *  escapes never pay the setup of the tier's vectorized search. */
STRINGZILLA_INLINE sz_cptr_t sz_strings_find_(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set,
                                              sz_strings_walks_t const *walks) {
    sz_size_t const probed = sz_min_of_two(length, (sz_size_t)16);
    for (sz_size_t index = 0; index != probed; ++index)
        if (sz_byteset_contains_u8(set, (sz_u8_t)text[index])) return text + index;
    return length > probed ? walks->find_byteset(text + probed, length - probed, set) : STRINGZILLA_NULL;
}

/** Appends @p length bytes at @p written into @p target, or only counts them when it is null. */
STRINGZILLA_INLINE sz_size_t sz_strings_emit_(sz_ptr_t target, sz_size_t written, sz_cptr_t source, sz_size_t length) {
    if (target && length) memmove(target + written, source, length);
    return written + length;
}

/** The value of the hexadecimal digit @p digit of either case, or 16 for any other byte. */
STRINGZILLA_INLINE sz_u32_t sz_strings_hex_digit_(char digit) {
    sz_u32_t const byte = (sz_u8_t)digit, lower = byte | 0x20;
    if (byte >= '0' && byte <= '9') return byte - '0';
    if (lower >= 'a' && lower <= 'f') return lower - 'a' + 10;
    return 16;
}

/** Parses @p count hexadecimal digits into @p value, failing on any other byte. */
STRINGZILLA_INLINE sz_bool_t sz_strings_hex_parse_(sz_cptr_t digits, sz_size_t count, sz_u32_t *value) {
    sz_u32_t parsed = 0;
    for (sz_size_t index = 0; index != count; ++index) {
        sz_u32_t const digit = sz_strings_hex_digit_(digits[index]);
        if (digit > 15) return sz_false_k;
        parsed = parsed << 4 | digit;
    }
    *value = parsed;
    return sz_true_k;
}

/** Writes @p rune as <tt>\\uXXXX</tt> in lowercase, a surrogate pair beyond U+FFFF, returning 6 or 12. */
STRINGZILLA_INLINE sz_size_t sz_strings_write_utf16_(sz_rune_t rune, char *sequence) {
    static char const digits[] = "0123456789abcdef";
    sz_u32_t units[2] = {rune, 0};
    sz_size_t units_count = 1;
    if (rune > 0xFFFF)
        units[0] = 0xD800 + ((rune - 0x10000) >> 10), units[1] = 0xDC00 + ((rune - 0x10000) & 0x3FF), units_count = 2;
    for (sz_size_t index = 0; index != units_count; ++index) {
        sz_u32_t const unit = units[index];
        char *const escape = sequence + index * 6;
        escape[0] = '\\', escape[1] = 'u', escape[2] = digits[unit >> 12 & 15], escape[3] = digits[unit >> 8 & 15];
        escape[4] = digits[unit >> 4 & 15], escape[5] = digits[unit & 15];
    }
    return units_count * 6;
}

/** The escape of @p byte, one that @p kind escapes alone, as up to 8 little-endian bytes, adding
 *  the bytes it outgrows @p byte by to @p extra: JSON, HTML or URL escapes. */
STRINGZILLA_INLINE sz_u64_t sz_strings_escape_sequence_(sz_escape_kind_t kind, sz_u8_t byte, sz_size_t *extra) {
    static sz_u8_t const upper[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
    static sz_u8_t const lower[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    // The letter after the backslash for each control, `u` for `\u00XX`
    static sz_u8_t const letters[32] = {'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'b', 't', 'n', 'u', 'f', 'r', 'u', 'u',
                                        'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u'};
    // The entities of `>`, `"`, `<`, `'` and `&` at `((byte >> 2) ^ byte) & 7`
    static char const entities[8][8] = {"", "&gt;", "&quot;", "&lt;", "", "", "&#39;", "&amp;"};
    if (kind == sz_escape_url_k) {
        *extra += 2;
        return '%' | (sz_u64_t)upper[byte >> 4] << 8 | (sz_u64_t)upper[byte & 15] << 16;
    }
    if (kind == sz_escape_html_k) {
        sz_size_t const hash = ((byte >> 2) ^ byte) & 7;
        // A nibble per hash holds the bytes its entity adds.
        *extra += 0x44003530u >> (hash * 4) & 15;
        return sz_u64_load(entities[hash]).u64;
    }
    sz_u8_t const letter = byte < 32 ? letters[byte] : byte;
    *extra += letter == 'u' ? 5 : 1;
    return '\\' | (sz_u64_t)letter << 8 | 0x30300000ull | (sz_u64_t)lower[byte >> 4] << 32 |
           (sz_u64_t)lower[byte & 15] << 40;
}

/** Escapes one string for JSON into @p target, or measures it for a null @p target; @p ascii also
 *  writes DEL and every non-ASCII codepoint as <tt>\\uXXXX</tt>, and every ill-formed byte as U+FFFD. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_json_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                     sz_bool_t ascii, sz_strings_walks_t const *walks) {
    // `"`, `\` and the C0 controls, and for ASCII output DEL and every byte from 0x80 on
    static sz_byteset_t const json_set = {{0x00000004FFFFFFFFull, 0x0000000010000000ull, 0, 0}};
    static sz_byteset_t const ascii_set = {{0x00000004FFFFFFFFull, 0x8000000010000000ull, ~0ull, ~0ull}};
    sz_byteset_t const *set = ascii ? &ascii_set : &json_set;
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_(source, (sz_size_t)(end - source), set, walks)) != STRINGZILLA_NULL;) {
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
            sequence_length = sz_strings_write_utf16_(rune, sequence);
        }
        written = sz_strings_emit_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_(target, written, sequence, sequence_length);
        source = hit + consumed;
    }
    return sz_strings_emit_(target, written, source, (sz_size_t)(end - source));
}

/** Escapes one string as lowercase hexadecimal digit pairs into @p target, or measures it. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_hex_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    static char const digits[] = "0123456789abcdef";
    for (sz_size_t index = 0; target && index != length; ++index) {
        sz_u8_t const byte = (sz_u8_t)source[index];
        target[index * 2] = digits[byte >> 4], target[index * 2 + 1] = digits[byte & 15];
    }
    return length * 2;
}

/** Escapes one string for HTML text and attributes into @p target, or measures it. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_html_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                     sz_strings_walks_t const *walks) {
    // `"`, `&`, `'`, `<` and `>`
    static sz_byteset_t const set = {{0x500000C400000000ull, 0, 0, 0}};
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_(source, (sz_size_t)(end - source), &set, walks)) != STRINGZILLA_NULL;) {
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
        written = sz_strings_emit_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_(target, written, entity, entity_length + 1);
        source = hit + 1;
    }
    return sz_strings_emit_(target, written, source, (sz_size_t)(end - source));
}

/** Percent-encodes one string for URLs into @p target, or measures it. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_url_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                    sz_strings_walks_t const *walks) {
    // Every byte outside `A-Z a-z 0-9 - . _ ~`
    static sz_byteset_t const set = {{0xFC009FFFFFFFFFFFull, 0xB800000178000001ull, ~0ull, ~0ull}};
    static char const digits[] = "0123456789ABCDEF";
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_(source, (sz_size_t)(end - source), &set, walks)) != STRINGZILLA_NULL;) {
        sz_u8_t const byte = (sz_u8_t)*hit;
        char const sequence[3] = {'%', digits[byte >> 4], digits[byte & 15]};
        written = sz_strings_emit_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_(target, written, sequence, 3);
        source = hit + 1;
    }
    return sz_strings_emit_(target, written, source, (sz_size_t)(end - source));
}

/** Escapes one string for terminals into @p target, or measures it, keeping well-formed UTF-8
 *  verbatim apart from the C1 controls and the bidirectional overrides. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_console_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                        sz_strings_walks_t const *walks) {
    // `"`, `\`, the C0 controls, DEL and every byte from 0x80 on
    static sz_byteset_t const set = {{0x00000004FFFFFFFFull, 0x8000000010000000ull, ~0ull, ~0ull}};
    static char const digits[] = "0123456789abcdef";
    sz_cptr_t const end = source + length;
    sz_cptr_t run = source, cursor = source;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_(cursor, (sz_size_t)(end - cursor), &set, walks)) != STRINGZILLA_NULL;) {
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
            sequence[1] = 'x', sequence[2] = digits[byte >> 4], sequence[3] = digits[byte & 15], sequence_length = 4;
        }
        else if (rune <= 0x9F || (rune >= 0x202A && rune <= 0x202E) || (rune >= 0x2066 && rune <= 0x2069))
            sequence_length = sz_strings_write_utf16_(rune, sequence), consumed = rune_length;
        else {
            cursor = hit + rune_length;
            continue;
        }
        written = sz_strings_emit_(target, written, run, (sz_size_t)(hit - run));
        written = sz_strings_emit_(target, written, sequence, sequence_length);
        run = cursor = hit + consumed;
    }
    return sz_strings_emit_(target, written, run, (sz_size_t)(end - run));
}

/** Escapes one CSV field into @p target, or measures it, quoting it only when it must be. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_csv_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                    sz_strings_walks_t const *walks) {
    // `,`, `"`, CR and LF, then `"` alone
    static sz_byteset_t const set = {{0x0000100400002400ull, 0, 0, 0}};
    static sz_byteset_t const quotes = {{0x0000000400000000ull, 0, 0, 0}};
    sz_cptr_t const end = source + length;
    sz_cptr_t cursor = sz_strings_find_(source, length, &set, walks);
    if (!cursor) return sz_strings_emit_(target, 0, source, length);
    sz_size_t written = sz_strings_emit_(target, 0, "\"", 1);
    sz_cptr_t run = source;
    // Once quoted, only quotes need doubling, so commas and line breaks stop the walk no more.
    for (cursor = sz_strings_find_(cursor, (sz_size_t)(end - cursor), &quotes, walks); cursor;
         cursor = sz_strings_find_(cursor, (sz_size_t)(end - cursor), &quotes, walks)) {
        ++cursor;
        written = sz_strings_emit_(target, written, run, (sz_size_t)(cursor - run));
        written = sz_strings_emit_(target, written, "\"", 1);
        run = cursor;
    }
    written = sz_strings_emit_(target, written, run, (sz_size_t)(end - run));
    return sz_strings_emit_(target, written, "\"", 1);
}

/** Escapes one string for @p kind into @p target, or only measures it for a null @p target. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_one_(sz_escape_kind_t kind, sz_cptr_t source, sz_size_t length,
                                                    sz_ptr_t target, sz_strings_walks_t const *walks) {
    switch (kind) {
    case sz_escape_hex_k: return sz_strings_escape_hex_(source, length, target);
    case sz_escape_html_k: return sz_strings_escape_html_(source, length, target, walks);
    case sz_escape_url_k: return sz_strings_escape_url_(source, length, target, walks);
    case sz_escape_console_k: return sz_strings_escape_console_(source, length, target, walks);
    case sz_escape_csv_k: return sz_strings_escape_csv_(source, length, target, walks);
    case sz_escape_json_ascii_k: return sz_strings_escape_json_(source, length, target, sz_true_k, walks);
    default: return sz_strings_escape_json_(source, length, target, sz_false_k, walks);
    }
}

/** Unescapes one JSON string body, or with @p console a terminal log line, into @p target, or
 *  measures it for a null @p target, returning @c STRINGZILLA_SIZE_MAX when it is malformed. */
STRINGZILLA_INLINE sz_size_t sz_strings_unescape_backslashes_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                              sz_bool_t console, sz_strings_walks_t const *walks) {
    // `\`
    static sz_byteset_t const set = {{0, 0x0000000010000000ull, 0, 0}};
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_(source, (sz_size_t)(end - source), &set, walks)) != STRINGZILLA_NULL;) {
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
            if (!console || left < 4 || !sz_strings_hex_parse_(hit + 2, 2, &rune)) return STRINGZILLA_SIZE_MAX;
            sequence[0] = (sz_u8_t)rune, consumed = 4;
            break;
        case 'u':
            if (left < 6 || !sz_strings_hex_parse_(hit + 2, 4, &rune) || (rune >= 0xDC00 && rune <= 0xDFFF))
                return STRINGZILLA_SIZE_MAX;
            consumed = 6;
            if (rune >= 0xD800 && rune <= 0xDBFF) {
                if (console || left < 12 || hit[6] != '\\' || hit[7] != 'u' ||
                    !sz_strings_hex_parse_(hit + 8, 4, &low) || low < 0xDC00 || low > 0xDFFF)
                    return STRINGZILLA_SIZE_MAX;
                rune = 0x10000 + ((rune - 0xD800) << 10) + (low - 0xDC00), consumed = 12;
            }
            sequence_length = sz_rune_encode(rune, sequence);
            break;
        default: return STRINGZILLA_SIZE_MAX;
        }
        written = sz_strings_emit_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_(target, written, (sz_cptr_t)sequence, sequence_length);
        source = hit + consumed;
    }
    return sz_strings_emit_(target, written, source, (sz_size_t)(end - source));
}

/** Decodes one string of hexadecimal digit pairs into @p target, or measures it, returning
 *  @c STRINGZILLA_SIZE_MAX for an odd length or a byte that is not a digit. */
STRINGZILLA_INLINE sz_size_t sz_strings_unescape_hex_(sz_cptr_t source, sz_size_t length, sz_ptr_t target) {
    if (length & 1) return STRINGZILLA_SIZE_MAX;
    for (sz_size_t index = 0; index != length / 2; ++index) {
        sz_u32_t byte;
        if (!sz_strings_hex_parse_(source + index * 2, 2, &byte)) return STRINGZILLA_SIZE_MAX;
        if (target) target[index] = (char)byte;
    }
    return length / 2;
}

/** Unescapes the HTML character references of one string into @p target, or measures it, copying
 *  every <tt>&amp;</tt> that starts no reference; it is never malformed. */
STRINGZILLA_INLINE sz_size_t sz_strings_unescape_html_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                       sz_strings_walks_t const *walks) {
    // `&`
    static sz_byteset_t const set = {{0x0000004000000000ull, 0, 0, 0}};
    static char const names[5][6] = {"amp;", "lt;", "gt;", "quot;", "apos;"};
    static char const characters[5] = {'&', '<', '>', '"', '\''};
    sz_cptr_t const end = source + length;
    sz_cptr_t run = source, cursor = source;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_(cursor, (sz_size_t)(end - cursor), &set, walks)) != STRINGZILLA_NULL;) {
        sz_size_t const left = (sz_size_t)(end - hit) - 1;
        sz_u8_t sequence[4];
        sz_size_t sequence_length = 0, consumed = 0;
        for (sz_size_t name_index = 0; name_index != 5 && !consumed; ++name_index) {
            sz_cptr_t const name = names[name_index];
            sz_size_t name_length = 0;
            while (name[name_length]) ++name_length;
            if (left >= name_length && sz_equal_serial_(hit + 1, name, name_length))
                sequence[0] = (sz_u8_t)characters[name_index], sequence_length = 1, consumed = name_length + 1;
        }
        if (!consumed && left >= 3 && hit[1] == '#') {
            sz_bool_t const hexadecimal = (sz_bool_t)(hit[2] == 'x' || hit[2] == 'X');
            sz_u32_t const base = hexadecimal ? 16 : 10;
            sz_cptr_t const digits = hit + 2 + hexadecimal;
            sz_cptr_t digit = digits;
            sz_u32_t rune = 0;
            for (; digit != end; ++digit) {
                sz_u32_t const value = hexadecimal ? sz_strings_hex_digit_(*digit) : (sz_u32_t)((sz_u8_t)*digit - '0');
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
        written = sz_strings_emit_(target, written, run, (sz_size_t)(hit - run));
        written = sz_strings_emit_(target, written, (sz_cptr_t)sequence, sequence_length);
        run = cursor = hit + consumed;
    }
    return sz_strings_emit_(target, written, run, (sz_size_t)(end - run));
}

/** Decodes the percent-escapes of one string into @p target, or measures it, returning
 *  @c STRINGZILLA_SIZE_MAX for a <tt>\%</tt> without two hexadecimal digits. */
STRINGZILLA_INLINE sz_size_t sz_strings_unescape_url_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                      sz_strings_walks_t const *walks) {
    // `%`
    static sz_byteset_t const set = {{0x0000002000000000ull, 0, 0, 0}};
    sz_cptr_t const end = source + length;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_(source, (sz_size_t)(end - source), &set, walks)) != STRINGZILLA_NULL;) {
        sz_u32_t byte;
        if (end - hit < 3 || !sz_strings_hex_parse_(hit + 1, 2, &byte)) return STRINGZILLA_SIZE_MAX;
        char const decoded = (char)byte;
        written = sz_strings_emit_(target, written, source, (sz_size_t)(hit - source));
        written = sz_strings_emit_(target, written, &decoded, 1);
        source = hit + 3;
    }
    return sz_strings_emit_(target, written, source, (sz_size_t)(end - source));
}

/** Unquotes one CSV field into @p target, or measures it, returning @c STRINGZILLA_SIZE_MAX for a
 *  quoted field without its closing quote or with an inner quote that is not doubled. */
STRINGZILLA_INLINE sz_size_t sz_strings_unescape_csv_(sz_cptr_t source, sz_size_t length, sz_ptr_t target,
                                                      sz_strings_walks_t const *walks) {
    // `"`
    static sz_byteset_t const set = {{0x0000000400000000ull, 0, 0, 0}};
    if (!length || *source != '"') return sz_strings_emit_(target, 0, source, length);
    if (length < 2 || source[length - 1] != '"') return STRINGZILLA_SIZE_MAX;
    sz_cptr_t const end = source + length - 1;
    sz_cptr_t run = source + 1;
    sz_size_t written = 0;
    for (sz_cptr_t hit; (hit = sz_strings_find_(run, (sz_size_t)(end - run), &set, walks)) != STRINGZILLA_NULL;) {
        if (hit + 1 == end || hit[1] != '"') return STRINGZILLA_SIZE_MAX;
        written = sz_strings_emit_(target, written, run, (sz_size_t)(hit + 1 - run));
        run = hit + 2;
    }
    return sz_strings_emit_(target, written, run, (sz_size_t)(end - run));
}

/** Unescapes one string from @p kind into @p target, or only measures it for a null @p target,
 *  returning @c STRINGZILLA_SIZE_MAX when it is malformed. */
STRINGZILLA_INLINE sz_size_t sz_strings_unescape_one_(sz_escape_kind_t kind, sz_cptr_t source, sz_size_t length,
                                                      sz_ptr_t target, sz_strings_walks_t const *walks) {
    switch (kind) {
    case sz_escape_hex_k: return sz_strings_unescape_hex_(source, length, target);
    case sz_escape_html_k: return sz_strings_unescape_html_(source, length, target, walks);
    case sz_escape_url_k: return sz_strings_unescape_url_(source, length, target, walks);
    case sz_escape_console_k: return sz_strings_unescape_backslashes_(source, length, target, sz_true_k, walks);
    case sz_escape_csv_k: return sz_strings_unescape_csv_(source, length, target, walks);
    default: return sz_strings_unescape_backslashes_(source, length, target, sz_false_k, walks);
    }
}

#pragma endregion Escaping

#pragma region Splitting

/** Bytes in the line break at @p index of @p text, or zero: LF, VT, FF, CR, FS, GS, RS, CR LF, NEL,
 *  LS or PS. */
STRINGZILLA_INLINE sz_size_t sz_strings_line_break_(sz_u8_t const *text, sz_size_t length, sz_size_t index) {
    sz_u8_t const byte = text[index];
    sz_size_t const left = length - index;
    if ((sz_u8_t)(byte - 0x0A) < 4 || (sz_u8_t)(byte - 0x1C) < 3)
        return 1 + (byte == '\r' && left > 1 && text[index + 1] == '\n');
    if (byte == 0xC2) return left > 1 && text[index + 1] == 0x85 ? 2 : 0;
    if (byte == 0xE2) return left > 2 && text[index + 1] == 0x80 && (text[index + 2] | 1) == 0xA9 ? 3 : 0;
    return 0;
}

/** Whether the 8 bytes at @p text are all printable ASCII, so none can be a byte of a line break:
 *  a byte below 0x20 borrows into its top bit, and a byte from 0x80 already has it set. */
STRINGZILLA_INLINE sz_bool_t sz_strings_line_clean_(sz_cptr_t text) {
    sz_u64_t const word = sz_u64_load(text).u64;
    return (sz_bool_t) !(((word - 0x2020202020202020ull) | word) & 0x8080808080808080ull);
}

#pragma endregion Splitting

#pragma region Drivers

/** Gathers strings into a tape, leaving a whole tape gathered onto itself where it lies. */
STRINGZILLA_INLINE sz_status_t sz_strings_to_tape_with_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape) {
    if (!order && sz_strings_aliases_(strings, tape) && count == tape->count && !sz_strings_offset_(tape, 0))
        return sz_success_k;
    return sz_strings_gather_(strings, order, count, STRINGZILLA_NULL, allocator, tape, STRINGZILLA_NULL);
}

/** Writes strings into NUL-padded rows, truncating longer ones and reporting the longest. */
STRINGZILLA_INLINE sz_status_t sz_strings_to_rows_with_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_strings_rows_t const *rows,
                                                        sz_size_t *longest) {
    sz_string_view_t views[64];
    sz_size_t longest_length = 0;
    for (sz_size_t start = 0; start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_ptr_t const row = rows->data + (start + position) * rows->stride;
            sz_size_t const length = views[position].length, kept = sz_min_of_two(length, rows->width);
            // Rows may be the very rows @p strings reads, so the copy may overlap itself.
            if (kept) memmove(row, views[position].start, kept);
            if (kept != rows->width) memset(row + kept, 0, rows->width - kept);
            longest_length = sz_max_of_two(longest_length, length);
        }
    }
    *longest = longest_length;
    return sz_success_k;
}

/** Maps every byte of @p strings through @p lut into a tape, in place when they view all of it. */
STRINGZILLA_INLINE sz_status_t sz_strings_lookup_with_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, char const lut[sz_at_least_(256)],
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_strings_walks_t const *walks) {
    if (order || !sz_strings_aliases_(strings, tape) || count != tape->count)
        return sz_strings_gather_(strings, order, count, lut, allocator, tape, walks);
    sz_size_t const first = sz_strings_offset_(tape, 0), end = sz_strings_offset_(tape, count);
    sz_ptr_t const data = sz_strings_payload_(tape);
    if (end != first) walks->lookup_bytes(data + first, data + first, end - first, lut);
    return sz_success_k;
}

/** Whether escaping the @p count strings of @p strings into @p payload bytes grows the tape
 *  @p tape holds where it lies: they must be that whole tape, keeping its offsets width, and the
 *  tape must not shrink. */
STRINGZILLA_INLINE sz_bool_t sz_strings_escapes_in_place_(sz_strings_view_t const *strings,
                                                          sz_strings_span_t const *tape, sz_size_t count,
                                                          sz_size_t payload) {
    return (sz_bool_t)(sz_strings_aliases_(strings, tape) && count == tape->count &&
                       sz_strings_layout_for_(payload) == tape->layout && payload >= sz_strings_offset_(tape, count));
}

/** Grows the payload block of @p tape to @p payload bytes, no fewer than it holds, and moves its
 *  bytes to the end, where @p reader then points, so escaping them front to back never overtakes
 *  a byte yet to be read. Points @p output at the grown tape, or leaves @p tape as it was. */
STRINGZILLA_INLINE sz_status_t sz_strings_escape_grow_(sz_strings_span_t *tape, sz_size_t payload,
                                                       sz_allocator_t *allocator, sz_strings_span_t *output,
                                                       sz_cptr_t *reader) {
    sz_size_t const first = sz_strings_offset_(tape, 0), end = sz_strings_offset_(tape, tape->count);
    void *data = sz_strings_payload_(tape);
    if (!sz_strings_resize_(data, end, payload, allocator, STRINGZILLA_NULL, &data)) return sz_bad_alloc_k;
    sz_strings_point_(tape, data, sz_strings_offsets_block_(tape));
    sz_ptr_t const bytes = (sz_ptr_t)data;
    if (end != first) memmove(bytes + payload - (end - first), bytes + first, end - first);
    *reader = bytes + payload - (end - first), *output = *tape;
    return sz_success_k;
}

/** Escapes the whole of @p tape where it lies, into @p payload bytes: the block grows first, so
 *  nothing can fail once writing starts. */
STRINGZILLA_INLINE sz_status_t sz_strings_escape_in_place_(sz_strings_span_t *tape, sz_escape_kind_t kind,
                                                           sz_size_t payload, sz_allocator_t *allocator,
                                                           sz_strings_walks_t const *walks) {
    sz_strings_span_t grown;
    sz_cptr_t reader;
    if (sz_strings_escape_grow_(tape, payload, allocator, &grown, &reader) != sz_success_k) return sz_bad_alloc_k;
    sz_size_t const count = tape->count;
    sz_ptr_t const bytes = sz_strings_payload_(tape);
    sz_size_t offset = 0, old_start = sz_strings_offset_(tape, 0);
    for (sz_size_t index = 0; index != count; ++index) {
        sz_size_t const old_end = sz_strings_offset_(tape, index + 1);
        offset += sz_strings_escape_one_(kind, reader, old_end - old_start, bytes + offset, walks);
        reader += old_end - old_start, old_start = old_end;
        sz_strings_offset_set_(tape, index + 1, offset);
    }
    sz_strings_offset_set_(tape, 0, 0);
    return sz_success_k;
}

/** Escapes strings for @p kind into a tape, growing the tape @p strings views where it lies. */
STRINGZILLA_INLINE sz_status_t sz_strings_escape_with_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_escape_kind_t kind,
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_strings_walks_t const *walks) {
    sz_string_view_t views[64];
    sz_size_t payload = 0;
    for (sz_size_t start = 0; start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_size_t const escaped = sz_strings_escape_one_(kind, views[position].start, views[position].length,
                                                             STRINGZILLA_NULL, walks);
            if (escaped > STRINGZILLA_SIZE_MAX - payload) return sz_bad_alloc_k;
            payload += escaped;
        }
    }
    if (!order && sz_strings_escapes_in_place_(strings, tape, count, payload))
        return sz_strings_escape_in_place_(tape, kind, payload, allocator, walks);

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_acquire_(strings, tape, count, payload, allocator, &output, STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_ptr_t const data = sz_strings_payload_(&output);
    sz_size_t offset = 0;
    sz_strings_offset_set_(&output, 0, 0);
    for (sz_size_t start = 0; start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_ptr_t const target = data ? data + offset : STRINGZILLA_NULL;
            offset += sz_strings_escape_one_(kind, views[position].start, views[position].length, target, walks);
            sz_strings_offset_set_(&output, start + position + 1, offset);
        }
    }
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

/** Unescapes strings from @p kind into a tape, writing nothing when any string is malformed. A tape
 *  unescaped onto itself only shrinks, so it takes fresh blocks rather than shrink once written. */
STRINGZILLA_INLINE sz_status_t sz_strings_unescape_with_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_escape_kind_t kind,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_size_t *malformed_index, sz_strings_walks_t const *walks) {
    sz_string_view_t views[64];
    sz_size_t payload = 0, valid = 0;
    for (sz_size_t start = 0; valid == start && start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position, ++valid) {
            sz_size_t const unescaped = sz_strings_unescape_one_(kind, views[position].start, views[position].length,
                                                                 STRINGZILLA_NULL, walks);
            if (unescaped == STRINGZILLA_SIZE_MAX) break;
            payload += unescaped;
        }
    }
    if (valid != count) {
        *malformed_index = valid;
        return sz_invalid_utf8_k;
    }
    sz_strings_span_t output;
    sz_status_t const status = sz_strings_acquire_(strings, tape, count, payload, allocator, &output, STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_ptr_t const data = sz_strings_payload_(&output);
    sz_size_t offset = 0;
    sz_strings_offset_set_(&output, 0, 0);
    for (sz_size_t start = 0; start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_ptr_t const target = data ? data + offset : STRINGZILLA_NULL;
            offset += sz_strings_unescape_one_(kind, views[position].start, views[position].length, target, walks);
            sz_strings_offset_set_(&output, start + position + 1, offset);
        }
    }
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

/** Normalizes one string into @p target, which has room for @c sz_utf8_norm_decomp_max_k bytes per
 *  byte of it or exactly for the length measured for it, and returns the bytes it wrote. */
STRINGZILLA_INLINE sz_size_t sz_strings_normalize_one_(sz_cptr_t source, sz_size_t length, sz_normal_form_t form,
                                                       sz_ptr_t target, sz_kernel_utf8_norm_t normalize) {
    sz_size_t written = 0;
    if (length) normalize(source, length, form, target, &written, STRINGZILLA_NULL);
    return written;
}

/** Scratch that strings normalize into while their output length is unknown: the stack's for up to
 *  256 flagged bytes, and beyond them a heap block of @c allocator that grows by doubling. */
typedef struct sz_strings_scratch_t {

    /** Where the heap block comes from and returns to. */
    sz_allocator_t *allocator;

    /** The heap block, null until a flagged part outgrows the stack. */
    void *heap;

    /** Bytes in @c heap. */
    sz_size_t heap_bytes;

    /** The stack's part, for the worst case of 256 flagged bytes. */
    char stack[256 * sz_utf8_norm_decomp_max_k];
} sz_strings_scratch_t;

/** Points into @p scratch with room to normalize @p length bytes in the worst case, or returns null
 *  when it cannot grow to them. */
STRINGZILLA_INLINE sz_ptr_t sz_strings_scratch_reserve_(sz_strings_scratch_t *scratch, sz_size_t length) {
    if (length > STRINGZILLA_SIZE_MAX / (2 * sz_utf8_norm_decomp_max_k)) return STRINGZILLA_NULL;
    sz_size_t const bound = length * sz_utf8_norm_decomp_max_k;
    if (bound <= sizeof(scratch->stack)) return scratch->stack;
    if (bound > scratch->heap_bytes) {
        sz_size_t const grown = sz_max_of_two(bound, scratch->heap_bytes * 2);
        if (!sz_strings_resize_(scratch->heap, scratch->heap_bytes, grown, scratch->allocator, STRINGZILLA_NULL,
                                &scratch->heap))
            return STRINGZILLA_NULL;
        scratch->heap_bytes = grown;
    }
    return (sz_ptr_t)scratch->heap;
}

/** Grows the payload of @p output from @p capacity bytes to hold @p needed, at least doubling it
 *  and to no fewer than @p least. Its @p written offsets widen once 32 bits cannot reach
 *  @p needed, and 32-bit ones keep the block within their reach. */
STRINGZILLA_INLINE sz_bool_t sz_strings_grow_(sz_strings_span_t *output, sz_size_t *capacity, sz_size_t needed,
                                              sz_size_t least, sz_size_t written, sz_allocator_t *allocator) {
    if (output->layout == sz_strings_u32_tape_k && sz_strings_layout_for_(needed) != sz_strings_u32_tape_k &&
        !sz_strings_relayout_(output, written, sz_strings_u64_tape_k, allocator))
        return sz_false_k;
    sz_size_t grown = sz_max_of_two(needed, least);
    if (*capacity <= STRINGZILLA_SIZE_MAX / 2) grown = sz_max_of_two(grown, *capacity * 2);
    if (output->layout == sz_strings_u32_tape_k) grown = sz_min_of_two(grown, (sz_size_t)0xFFFFFFFFu);
    void *data = sz_strings_payload_(output);
    if (!sz_strings_resize_(data, *capacity, grown, allocator, STRINGZILLA_NULL, &data)) return sz_false_k;
    sz_strings_point_(output, data, sz_strings_offsets_block_(output));
    *capacity = grown;
    return sz_true_k;
}

/**
 *  @brief Normalizes strings into @p form through a tier's @p normalize, writing each string once.
 *
 *  Writes into blocks of its own, never the ones @p tape holds, as they grow while being written:
 *  a payload that starts at the total input length takes a string straight while it has room for
 *  the worst case of @c sz_utf8_norm_decomp_max_k bytes per byte, and through scratch otherwise,
 *  growing only when even the normalized string does not fit. Fitted to its exact size, the
 *  payload is handed to @p tape last, so a failure leaves @p tape as it was.
 */
STRINGZILLA_INLINE sz_status_t sz_strings_normalize_with_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                          sz_size_t count, sz_normal_form_t form,
                                                          sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                          sz_kernel_utf8_norm_t normalize) {
    sz_string_view_t views[64];
    sz_size_t read = 0;
    for (sz_size_t start = 0; start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            if (views[position].length > STRINGZILLA_SIZE_MAX - read) return sz_bad_alloc_k;
            read += views[position].length;
        }
    }
    sz_strings_span_t none, output;
    memset(&none, 0, sizeof(none));
    sz_status_t status = sz_strings_acquire_(strings, &none, count, read, allocator, &output, STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_strings_scratch_t scratch;
    scratch.allocator = allocator, scratch.heap = STRINGZILLA_NULL, scratch.heap_bytes = 0;
    sz_size_t capacity = read, cursor = 0;
    sz_strings_offset_set_(&output, 0, 0);
    for (sz_size_t start = 0; status == sz_success_k && start < count; start += 64) {
        sz_size_t const block_count = sz_strings_block_(strings, order, count, start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_cptr_t const source = views[position].start;
            sz_size_t const length = views[position].length;
            sz_ptr_t const data = sz_strings_payload_(&output);
            if (length && length <= (capacity - cursor) / sz_utf8_norm_decomp_max_k)
                cursor += sz_strings_normalize_one_(source, length, form, data + cursor, normalize);
            else if (length) {
                sz_ptr_t const normalized = sz_strings_scratch_reserve_(&scratch, length);
                sz_size_t const written = normalized
                                              ? sz_strings_normalize_one_(source, length, form, normalized, normalize)
                                              : 0;
                if (!normalized || written > STRINGZILLA_SIZE_MAX - cursor ||
                    (written > capacity - cursor &&
                     !sz_strings_grow_(&output, &capacity, cursor + written, read, start + position + 1, allocator))) {
                    status = sz_bad_alloc_k;
                    break;
                }
                memcpy(sz_strings_payload_(&output) + cursor, normalized, written);
                cursor += written;
            }
            sz_strings_offset_set_(&output, start + position + 1, cursor);
        }
    }
    if (scratch.heap) allocator->free(scratch.heap, scratch.heap_bytes, allocator->handle, STRINGZILLA_NULL);

    // Fit the payload exactly, then the offsets to the width that payload takes.
    void *data = sz_strings_payload_(&output);
    if (status == sz_success_k) {
        if (sz_strings_resize_(data, capacity, cursor, allocator, STRINGZILLA_NULL, &data)) {
            sz_strings_point_(&output, data, sz_strings_offsets_block_(&output));
            capacity = cursor;
            if (output.layout != sz_strings_layout_for_(cursor) &&
                !sz_strings_relayout_(&output, count + 1, sz_strings_layout_for_(cursor), allocator))
                status = sz_bad_alloc_k;
        }
        else status = sz_bad_alloc_k;
    }
    if (status != sz_success_k) {
        sz_strings_release_(&none, &output, capacity, allocator, STRINGZILLA_NULL);
        return status;
    }
    // Blocks of exactly the final sizes, like a borrowed span's, are filled rather than replaced.
    if (sz_strings_reusable_(strings, tape, count, cursor)) {
        if (cursor) memcpy(sz_strings_payload_(tape), sz_strings_payload_(&output), cursor);
        memcpy(sz_strings_offsets_block_(tape), sz_strings_offsets_block_(&output),
               sz_strings_offsets_bytes_(output.layout, count));
        sz_strings_release_(&none, &output, cursor, allocator, STRINGZILLA_NULL);
        return sz_success_k;
    }
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

/** Whether the @p length bytes at @p text overlap either block @p tape holds. */
STRINGZILLA_INLINE sz_bool_t sz_strings_holds_(sz_strings_span_t const *tape, sz_cptr_t text, sz_size_t length) {
    void const *const offsets = sz_strings_offsets_block_(tape);
    if (!offsets || !length) return sz_false_k;
    sz_size_t const start = (sz_size_t)text, end = start + length;
    sz_size_t const payload = (sz_size_t)sz_strings_payload_(tape), offsets_start = (sz_size_t)offsets;
    sz_size_t const payload_end = payload + sz_strings_offset_(tape, tape->count);
    sz_size_t const offsets_end = offsets_start + sz_strings_offsets_bytes_(tape->layout, tape->count);
    return (sz_bool_t)((start < payload_end && payload < end) || (start < offsets_end && offsets_start < end));
}

/** Points @p output at blocks for @p parts over @p payload bytes: those @p tape holds when they fit
 *  and @p text lies outside them, or else fresh ones, so writing never overwrites what it reads. */
STRINGZILLA_INLINE sz_status_t sz_strings_split_acquire_(sz_cptr_t text, sz_size_t length,
                                                         sz_strings_span_t const *tape, sz_size_t parts,
                                                         sz_size_t payload, sz_allocator_t *allocator,
                                                         sz_strings_span_t *output) {
    sz_strings_span_t none;
    sz_strings_view_t unread;
    memset(&none, 0, sizeof(none)), memset(&unread, 0, sizeof(unread));
    unread.layout = sz_strings_accessor_k;
    return sz_strings_acquire_(&unread, sz_strings_holds_(tape, text, length) ? &none : tape, parts, payload, allocator,
                               output, STRINGZILLA_NULL);
}

/** Sets the first and the last offsets of a written split @p output and hands it to @p tape. */
STRINGZILLA_INLINE void sz_strings_split_commit_(sz_strings_span_t *tape, sz_strings_span_t *output, sz_size_t payload,
                                                 sz_allocator_t *allocator) {
    sz_strings_offset_set_(output, 0, 0), sz_strings_offset_set_(output, output->count, payload);
    sz_strings_commit_(tape, output, allocator, STRINGZILLA_NULL);
}

/** Measures into @p length the join of @p rows rows of @p columns, multiplying out the frames and
 *  separators, reading a tape column's first and last offsets and an accessor 64 views a call. */
STRINGZILLA_INLINE sz_status_t sz_strings_join_measure_(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                        sz_size_t rows, sz_string_view_t const *frames,
                                                        sz_string_view_t between, sz_size_t *length) {
    sz_size_t framing = 0, cells = 0, overflow = 0;
    sz_string_view_t views[64];
    for (sz_size_t column = 0; column <= columns_count; ++column)
        overflow |= frames[column].length > STRINGZILLA_SIZE_MAX - framing, framing += frames[column].length;
    for (sz_size_t column = 0; column != columns_count && rows; ++column) {
        sz_strings_view_t const *const strings = columns + column;
        sz_bool_t const tape = (sz_bool_t)(strings->layout != sz_strings_accessor_k);
        sz_size_t const bytes = tape ? sz_strings_view_offset_(strings, rows) - sz_strings_view_offset_(strings, 0) : 0;
        overflow |= bytes > STRINGZILLA_SIZE_MAX - cells, cells += bytes;
        // Accessor cells arrive 64 at a time, as each fetch costs one indirect call.
        for (sz_size_t row = 0; !tape && row != rows; ++row) {
            if (row % 64 == 0) sz_strings_get_block_(strings, row, views);
            overflow |= views[row % 64].length > STRINGZILLA_SIZE_MAX - cells, cells += views[row % 64].length;
        }
    }
    sz_size_t const separators = rows ? rows - 1 : 0;
    overflow |= framing && rows > STRINGZILLA_SIZE_MAX / framing;
    overflow |= between.length && separators > STRINGZILLA_SIZE_MAX / between.length;
    sz_size_t const framed = framing * rows, separated = between.length * separators;
    overflow |= framed > STRINGZILLA_SIZE_MAX - cells || separated > STRINGZILLA_SIZE_MAX - cells - framed;
    if (overflow) return sz_bad_alloc_k;
    *length = cells + framed + separated;
    return sz_success_k;
}

/** Writes the measured join of @p columns into @p output row by row, fetching each cell alone, in
 *  any layouts, which always succeeds. */
STRINGZILLA_INLINE sz_status_t sz_strings_join_rows_serial_(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                            sz_size_t rows, sz_string_view_t const *frames,
                                                            sz_string_view_t between, sz_ptr_t output) {
    sz_size_t written = 0;
    for (sz_size_t row = 0; row != rows; ++row) {
        // Every cell follows its frame, and the last frame and the row separator follow the last.
        for (sz_size_t column = 0; column <= columns_count; ++column) {
            sz_string_view_t cell = {between.start, row + 1 != rows ? between.length : 0};
            if (column != columns_count) sz_strings_get_(columns + column, &row, 1, &cell);
            if (frames[column].length) memcpy(output + written, frames[column].start, frames[column].length);
            written += frames[column].length;
            if (cell.length) memcpy(output + written, cell.start, cell.length);
            written += cell.length;
        }
    }
    return sz_success_k;
}

#pragma endregion Drivers

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_strings_to_tape_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_strings_rows_t const *rows,
                                                      sz_size_t *longest, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_serial_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_escape_kind_t kind,
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_size_t *malformed_index, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_serial_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, char const lut[sz_at_least_(256)],
                                                     sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                     sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_serial_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_normal_form_t form,
                                                        sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                        sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_serial);
}

STRINGZILLA_API sz_status_t sz_strings_split_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                    sz_split_kind_t kind, sz_allocator_t *allocator,
                                                    sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_size_t const each = kind != sz_split_runs_k;
    sz_size_t removed_count = 0, ends_count = 0, previous_removed = 1;
    for (sz_size_t index = 0; index != length; ++index) {
        sz_size_t const removed = sz_byteset_contains_u8(delimiters, bytes[index]);
        removed_count += removed, ends_count += removed & (previous_removed ^ 1), previous_removed = removed;
    }
    sz_size_t const payload = length - removed_count;
    sz_size_t const parts = each ? removed_count + 1 : ends_count + (previous_removed ^ 1);

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_split_acquire_(text, length, tape, parts, payload, allocator, &output);
    if (status != sz_success_k) return status;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_size_t written = 0, ended = 0;
    previous_removed = 1;
    // A delimiter ending no part directly follows one that does, so rewriting its slot is harmless.
    for (sz_size_t index = 0; index != length; ++index) {
        sz_size_t const removed = sz_byteset_contains_u8(delimiters, bytes[index]);
        ended += removed & ((previous_removed ^ 1) | each), previous_removed = removed;
        if (removed) sz_strings_offset_set_(&output, ended, written);
        else data[written++] = bytes[index];
    }
    sz_strings_split_commit_(tape, &output, payload, allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_strings_split_lines_serial(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                          sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_size_t removed_count = 0, ends_count = 0, trailing = 0;
    for (sz_size_t index = 0; index < length;) {
        sz_size_t const clean = length - index >= 8 && sz_strings_line_clean_(text + index) ? 8 : 0;
        sz_size_t const broken = clean ? 0 : sz_strings_line_break_(bytes, length, index);
        removed_count += broken, ends_count += broken != 0, trailing = broken == 0;
        index += clean + broken + (!clean && !broken);
    }
    sz_size_t const payload = length - removed_count, parts = ends_count + trailing;

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_split_acquire_(text, length, tape, parts, payload, allocator, &output);
    if (status != sz_success_k) return status;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_size_t written = 0, ended = 0;
    for (sz_size_t index = 0; index < length;) {
        sz_size_t const clean = length - index >= 8 && sz_strings_line_clean_(text + index) ? 8 : 0;
        sz_size_t const broken = clean ? 0 : sz_strings_line_break_(bytes, length, index);
        if (clean) memcpy(data + written, bytes + index, 8), written += 8;
        else if (broken) sz_strings_offset_set_(&output, ++ended, written);
        else data[written++] = bytes[index];
        index += clean + broken + (!clean && !broken);
    }
    sz_strings_split_commit_(tape, &output, payload, allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_strings_join_serial(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                   sz_size_t rows, sz_string_view_t const *frames,
                                                   sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                   sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    if (!output) return sz_strings_join_measure_(columns, columns_count, rows, frames, between, length);
    return sz_strings_join_rows_serial_(columns, columns_count, rows, frames, between, output);
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_serial(sz_allocator_t *allocator) {
    return sz_allocator_init_heap(allocator);
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_serial(sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SERIAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_SERIAL_H_
