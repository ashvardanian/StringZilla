/**
 *  @brief Uncased UTF-8 search, comparison & invariance checks: serial scaffolding.
 *  @file include/stringzilla/utf8_uncased/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_SERIAL_H_
#define STRINGZILLA_UTF8_UNCASED_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"
#include "stringzilla/utf8_uncased_fold/serial.h" // `sz_unicode_fold_codepoint_`
#include "stringzilla/find/serial.h"              // `sz_find_serial`

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Iterator state for streaming through folded UTF-8 runes.
 *  Handles one-to-many case folding expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) transparently.
 */
typedef struct {
    sz_cptr_t ptr;           // Current position in UTF-8 string
    sz_cptr_t end;           // End of string
    sz_rune_t pending[4];    // Buffered folded runes from one-to-many expansions
    sz_size_t pending_count; // Number of pending folded runes
    sz_size_t pending_idx;   // Current index into pending buffer
} sz_utf8_folded_iter_t_;

/** @brief Initialize a folded rune iterator. */
SZ_HELPER_AUTO void sz_utf8_folded_iter_init_(sz_utf8_folded_iter_t_ *iterator, sz_cptr_t string, sz_size_t length) {
    iterator->ptr = string;
    iterator->end = string + length;
    iterator->pending_count = 0;
    iterator->pending_idx = 0;
}

/**
 *  @brief Get next folded rune. Returns `sz_false_k` when exhausted.
 *  Malformed UTF-8 is handled losslessly: a byte that does not begin a well-formed codepoint is emitted as a
 *  single literal byte (tagged so it compares byte-for-byte and never collides with a real folded codepoint) and
 *  the iterator resyncs by one byte, never reading past `end`.
 */
SZ_HELPER_AUTO sz_bool_t sz_utf8_folded_iter_next_(sz_utf8_folded_iter_t_ *it, sz_rune_t *out_rune) {
    // Refill pending buffer if exhausted
    if (it->pending_idx >= it->pending_count) {
        if (it->ptr >= it->end) return sz_false_k;

        // ASCII fast-path: fold inline without buffering
        sz_u8_t lead = *(sz_u8_t const *)it->ptr;
        if (lead < 0x80) {
            *out_rune = sz_ascii_fold_(lead);
            it->ptr++;
            it->pending_count = 0; // Clear pending buffer
            it->pending_idx = 0;   // Signal first rune of new codepoint for source tracking
            return sz_true_k;
        }

        // Multi-byte UTF-8: decode (bounds-checked), fold, and buffer. A byte that does not begin a
        // well-formed codepoint folds to itself (>= 0x80 bytes are unchanged by `sz_ascii_fold_`) and resyncs
        // by one byte, never over-reading past `end`.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode(it->ptr, it->end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            *out_rune = sz_rune_malformed_byte_(lead);
            it->ptr++;
            it->pending_count = 0;
            it->pending_idx = 0;
            return sz_true_k;
        }

        it->ptr += rune_length;
        // Pre-fill pending buffer with sentinel values to prevent stale data from causing false matches.
        // The fold function will overwrite positions it uses; unused positions keep the sentinel.
        // This follows the same pattern as sz_utf8_uncased_search_2folded_serial_ and
        // sz_utf8_uncased_search_3folded_serial_.
        it->pending[0] = 0xFFFFFFFFu;
        it->pending[1] = 0xFFFFFFFEu;
        it->pending[2] = 0xFFFFFFFDu;
        it->pending[3] = 0xFFFFFFFCu;
        it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
        it->pending_idx = 0;
    }

    *out_rune = it->pending[it->pending_idx++];
    return sz_true_k;
}

/**
 *  @brief Reverse iterator state for streaming through folded UTF-8 runes backwards.
 * Handles one-to-many case folding expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) transparently
 * in reverse order.
 */
typedef struct {
    sz_cptr_t ptr;           // Current position (points to byte AFTER current sequence)
    sz_cptr_t start;         // Start of string (stop when ptr reaches this)
    sz_rune_t pending[4];    // Buffered folded runes from one-to-many expansions (in reverse order)
    sz_size_t pending_count; // Number of pending folded runes
    sz_size_t pending_idx;   // Current index into pending buffer
} sz_utf8_folded_reverse_iter_t_;

/** @brief Initialize a reverse folded rune iterator. Iterates from end towards start. */
SZ_HELPER_AUTO void sz_utf8_folded_reverse_iter_init_(sz_utf8_folded_reverse_iter_t_ *it, sz_cptr_t start,
                                                      sz_cptr_t end) {
    it->ptr = end;
    it->start = start;
    it->pending_count = 0;
    it->pending_idx = 0;
}

/**
 *  @brief Get previous folded rune (walking backwards). Returns `sz_false_k` when exhausted.
 * When a codepoint folds to multiple runes (like 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)), returns them in
 * reverse order ('s', then 's'). Malformed UTF-8 is handled losslessly and byte-identically to the forward
 * iterator: a byte that does not begin/end a well-formed codepoint is emitted as a single tagged literal byte and
 * the iterator resyncs by one byte, so the backward rune stream is exactly the reverse of the forward stream.
 */
SZ_HELPER_AUTO sz_bool_t sz_utf8_folded_reverse_iter_prev_(sz_utf8_folded_reverse_iter_t_ *it, sz_rune_t *out_rune) {
    // Return pending runes if any (stored in reverse order, consumed in reverse)
    if (it->pending_idx < it->pending_count) {
        *out_rune = it->pending[it->pending_count - 1 - it->pending_idx];
        it->pending_idx++;
        return sz_true_k;
    }

    // Refill: find previous codepoint
    if (it->ptr <= it->start) return sz_false_k;

    // Remember one-past-the-end of the sequence we are about to decode, so the strict decode is bounded
    // and a malformed run resyncs one byte at a time - mirroring the forward iterator byte-for-byte.
    sz_cptr_t const sequence_end = it->ptr;

    // The byte immediately before `sequence_end` is the last byte of whatever codepoint ends here.
    sz_u8_t const last_byte = *(sz_u8_t const *)(sequence_end - 1);

    // ASCII fast-path: a byte < 0x80 is always its own complete 1-byte codepoint.
    if (last_byte < 0x80) {
        it->ptr = sequence_end - 1;
        *out_rune = sz_ascii_fold_(last_byte);
        it->pending_count = 0;
        it->pending_idx = 0;
        return sz_true_k;
    }

    // Otherwise walk backwards over up to 3 continuation bytes (0x80-0xBF) to locate a candidate lead.
    // A well-formed multi-byte rune is at most 4 bytes, so stop after considering 4 positions.
    sz_cptr_t candidate = sequence_end - 1;
    for (sz_size_t back = 0; back < 3 && candidate > it->start && (*(sz_u8_t const *)candidate & 0xC0) == 0x80; ++back)
        candidate--;

    // Multi-byte UTF-8: decode (bounded) and fold only if the bytes from the candidate lead form a well-formed
    // codepoint that ends EXACTLY at `sequence_end`. Otherwise the last byte does not begin/end a valid rune, so
    // treat it as a literal folded-to-itself byte and resync by one - matching the forward iterator byte-for-byte.
    sz_rune_t rune;
    sz_rune_length_t const rune_length = sz_rune_decode(candidate, sequence_end, &rune);
    if (rune_length == sz_rune_invalid_k || candidate + rune_length != sequence_end) {
        it->ptr = sequence_end - 1;
        *out_rune = sz_rune_malformed_byte_(last_byte);
        it->pending_count = 0;
        it->pending_idx = 0;
        return sz_true_k;
    }
    it->ptr = candidate;

    // Store folded runes in pending buffer
    it->pending[0] = 0xFFFFFFFFu;
    it->pending[1] = 0xFFFFFFFEu;
    it->pending[2] = 0xFFFFFFFDu;
    it->pending[3] = 0xFFFFFFFCu;
    it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
    it->pending_idx = 1; // We'll return the last one now, then the rest in subsequent calls

    // Return the LAST folded rune first (since we're going backwards)
    *out_rune = it->pending[it->pending_count - 1];
    return sz_true_k;
}

#pragma region Case Invariance & Ordering

/**
 *  @brief Internal helper: checks if a single Unicode codepoint is case-agnostic.
 *
 *  A codepoint is case-agnostic if ALL of the following are true:
 *  1. It folds to exactly itself (no transformation, no expansion)
 *  2. It does NOT belong to any bicameral (cased) script
 *  3. It does NOT appear in any case fold expansion as a target character
 *
 *  The third condition is critical. Consider 'ʾ' (U+02BE, CA BE):
 *  - It has no case variant and folds to itself
 *  - However, 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
 *  - A needle containing 'ʾ' must match at position 1 of the folded expansion of 'ẚ'
 *  - Binary search cannot handle this - it only sees 'ẚ' as a 3-byte sequence (E1 BA 9A)
 *  - Therefore 'ʾ' must NOT be treated as case-agnostic
 *
 *  This function implements the check via explicit range exclusions for all bicameral
 *  scripts and all Unicode blocks containing case fold expansion target characters.
 *
 *  @param rune Unicode codepoint to check.
 *  @return sz_true_k if the codepoint is case-agnostic, sz_false_k otherwise.
 *
 *  @warning This is an internal function. Use sz_utf8_find_cased_serial() for string checking.
 *  @see sz_utf8_find_cased_serial
 *  @see sz_unicode_fold_codepoint_
 */
SZ_HELPER_AUTO sz_bool_t sz_rune_is_uncased_(sz_rune_t rune) {

    // Check if this rune participates in case folding
    sz_rune_t folded_runes[3];
    sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);

    // If it expands or changes, it's not caseless
    if (folded_count != 1 || folded_runes[0] != rune) return sz_false_k;

    // Check if this rune is a lowercase target of some uppercase letter.
    // Lowercase letters that don't change when folded still participate in case
    // because uppercase versions fold TO them. We must mark entire bicameral
    // script ranges as "not caseless" to enable proper uncased matching.
    //
    // Important: Combining diacritical marks (U+0300-U+036F) can appear as non-first
    // runes in multi-rune case fold expansions. Example: ǰ (U+01F0) → j + ̌ (U+030C).
    // A needle starting with combining caron could match inside such an expansion,
    // so combining marks must NOT be treated as case-agnostic.
    //
    // Bicameral scripts organized by UTF-8 lead byte for efficient checking:
    //
    // 1-byte sequences with upper and lower case (U+0000-007F): 00-7F
    if (rune >= 0x0041 && rune <= 0x005A) return sz_false_k; // Basic Latin (A-Z)
    if (rune >= 0x0061 && rune <= 0x007A) return sz_false_k; // Basic Latin (a-z)
    //
    // 2-byte sequences (U+0080-07FF): C2-DF lead bytes
    if (rune >= 0x00C0 && rune <= 0x00FF) return sz_false_k; // Latin-1 Supplement (À-ÿ)
    if (rune >= 0x0100 && rune <= 0x024F) return sz_false_k; // Latin Extended-A/B
    if (rune >= 0x0250 && rune <= 0x02AF) return sz_false_k; // IPA Extensions
    if (rune >= 0x02B0 && rune <= 0x02FF) return sz_false_k; // Spacing Modifier Letters (ʾ U+02BE appears in ẚ→aʾ)
    if (rune >= 0x0300 && rune <= 0x036F) return sz_false_k; // Combining Diacritical Marks (can appear in expansions!)
    if (rune >= 0x0370 && rune <= 0x03FF) return sz_false_k; // Greek and Coptic
    if (rune >= 0x0400 && rune <= 0x04FF) return sz_false_k; // Cyrillic
    if (rune >= 0x0500 && rune <= 0x052F) return sz_false_k; // Cyrillic Supplement
    if (rune >= 0x0531 && rune <= 0x0587) return sz_false_k; // Armenian (uppercase + lowercase + ligature)
    //
    // 3-byte sequences (U+0800-FFFF): E0-EF lead bytes
    if (rune >= 0x10A0 && rune <= 0x10FF) return sz_false_k; // Georgian (Asomtavruli + Mkhedruli)
    if (rune >= 0x13A0 && rune <= 0x13FD) return sz_false_k; // Cherokee (folds to uppercase!)
    if (rune >= 0x1C80 && rune <= 0x1C8F) return sz_false_k; // Cyrillic Extended-C
    if (rune >= 0x1C90 && rune <= 0x1CBF) return sz_false_k; // Georgian Extended (Mtavruli)
    if (rune == 0x1D79 || rune == 0x1D7D || rune == 0x1D8E)
        return sz_false_k; // Phonetic Extensions ᵹ ᵽ ᶎ (fold targets of Ᵹ U+A77D, Ᵽ U+2C63, Ᶎ U+A7C6)
    if (rune >= 0x1E00 && rune <= 0x1EFF) return sz_false_k; // Latin Extended Additional
    if (rune >= 0x1F00 && rune <= 0x1FFF) return sz_false_k; // Greek Extended
    if (rune == 0x214E) return sz_false_k;                   // ⅎ (fold target of Ⅎ U+2132)
    if (rune >= 0x2170 && rune <= 0x217F) return sz_false_k; // small Roman numerals (fold targets of U+2160-216F)
    if (rune == 0x2184) return sz_false_k;                   // ↄ (fold target of Ↄ U+2183)
    if (rune >= 0x24D0 && rune <= 0x24E9) return sz_false_k; // circled small Latin (fold targets of U+24B6-24CF)
    if (rune >= 0x2C00 && rune <= 0x2C5F) return sz_false_k; // Glagolitic
    if (rune >= 0x2C60 && rune <= 0x2C7F) return sz_false_k; // Latin Extended-C
    if (rune >= 0x2C80 && rune <= 0x2CFF) return sz_false_k; // Coptic
    if (rune >= 0x2D00 && rune <= 0x2D2F) return sz_false_k; // Georgian Supplement (Nuskhuri)
    if (rune >= 0x2DE0 && rune <= 0x2DFF) return sz_false_k; // Cyrillic Extended-A
    if (rune >= 0xA640 && rune <= 0xA69F) return sz_false_k; // Cyrillic Extended-B
    if (rune >= 0xA720 && rune <= 0xA7FF) return sz_false_k; // Latin Extended-D
    if (rune >= 0xAB30 && rune <= 0xAB6F) return sz_false_k; // Latin Extended-E
    if (rune >= 0xAB70 && rune <= 0xABBF) return sz_false_k; // Cherokee Supplement (lowercase)
    if (rune >= 0xFB00 && rune <= 0xFB06) return sz_false_k; // Alphabetic Presentation (ligatures)
    if (rune >= 0xFB13 && rune <= 0xFB17) return sz_false_k; // Armenian ligatures
    if (rune >= 0xFF21 && rune <= 0xFF5A) return sz_false_k; // Fullwidth Latin
    //
    // 4-byte sequences (U+10000-10FFFF): F0-F4 lead bytes
    if (rune >= 0x10400 && rune <= 0x1044F) return sz_false_k; // Deseret
    if (rune >= 0x104B0 && rune <= 0x104FF) return sz_false_k; // Osage
    if (rune >= 0x10570 && rune <= 0x105BF) return sz_false_k; // Vithkuqi
    if (rune >= 0x10780 && rune <= 0x107BF) return sz_false_k; // Latin Extended-F
    if (rune >= 0x10C80 && rune <= 0x10CFF) return sz_false_k; // Old Hungarian
    if (rune >= 0x10D70 && rune <= 0x10D85) return sz_false_k; // Garay small letters (fold targets of U+10D50-10D65)
    if (rune >= 0x118A0 && rune <= 0x118FF) return sz_false_k; // Warang Citi
    if (rune >= 0x16E40 && rune <= 0x16E9F) return sz_false_k; // Medefaidrin
    if (rune >= 0x16EBB && rune <= 0x16ED3) return sz_false_k; // Beria Erfe small letters (Unicode 17 fold targets)
    if (rune >= 0x1DF00 && rune <= 0x1DFFF) return sz_false_k; // Latin Extended-G
    if (rune >= 0x1E000 && rune <= 0x1E02F) return sz_false_k; // Glagolitic Supplement
    if (rune >= 0x1E030 && rune <= 0x1E08F) return sz_false_k; // Cyrillic Extended-D
    if (rune >= 0x1E900 && rune <= 0x1E95F) return sz_false_k; // Adlam

    return sz_true_k;
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_find_cased_serial(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *text_cursor = (sz_u8_t const *)str;
    sz_u8_t const *text_end = text_cursor + length;

    while (text_cursor < text_end) {
        sz_u8_t lead = *text_cursor;

        // ASCII fast path: only digits, punctuation, and control chars are caseless
        // A-Z (0x41-0x5A) and a-z (0x61-0x7A) participate in case folding
        if (lead < 0x80) {
            if ((lead >= 'A' && lead <= 'Z') || (lead >= 'a' && lead <= 'z')) return (sz_cptr_t)text_cursor;
            text_cursor++;
            continue;
        }

        // Multi-byte: decode and check. A byte that does not begin a well-formed codepoint is its own
        // 1-byte maximal subpart - it folds to itself, so it is caseless and never a violation; resync by one byte.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)text_cursor, (sz_cptr_t)text_end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            text_cursor++;
            continue;
        }
        if (sz_rune_is_uncased_(rune) == sz_false_k) return (sz_cptr_t)text_cursor;
        text_cursor += rune_length;
    }

    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_ordering_t sz_utf8_uncased_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                           sz_size_t b_length) {
    sz_utf8_folded_iter_t_ a_iterator, b_iterator;
    sz_utf8_folded_iter_init_(&a_iterator, a, a_length);
    sz_utf8_folded_iter_init_(&b_iterator, b, b_length);

    sz_rune_t a_rune = 0,
              b_rune = 0; // Initialized to satisfy GCC's -Wmaybe-uninitialized; the iterators always set them.
    for (;;) {
        sz_bool_t pulled_from_a = sz_utf8_folded_iter_next_(&a_iterator, &a_rune);
        sz_bool_t pulled_from_b = sz_utf8_folded_iter_next_(&b_iterator, &b_rune);

        if (!pulled_from_a && !pulled_from_b) return sz_equal_k;
        if (!pulled_from_a) return sz_less_k;
        if (!pulled_from_b) return sz_greater_k;
        if (a_rune != b_rune) return sz_order_scalars_(a_rune, b_rune);
    }
}

#pragma endregion // Case Invariance & Ordering

#pragma region Substring Search

/**
 *  @brief Verify head region uncasedly (backward iteration).
 *
 *  Walks backward from needle_end/haystack_end, comparing folded runes.
 *  Returns true if needle region exhausts (matched), with haystack bytes consumed.
 *
 *  @param needle_start Start of needle head region.
 *  @param needle_end End of needle head region (where safe window begins).
 *  @param haystack_start Start of haystack (lower bound for backward scan).
 *  @param haystack_end End of haystack head region (where safe window was found).
 *  @param match_length Haystack bytes consumed by this match.
 */
SZ_HELPER_AUTO sz_bool_t sz_utf8_uncased_verify_head_(sz_cptr_t needle_start, sz_cptr_t needle_end,
                                                      sz_cptr_t haystack_start, sz_cptr_t haystack_end,
                                                      sz_size_t *match_length) {

    // If needle head is empty, no haystack bytes needed
    if (needle_end <= needle_start) {
        *match_length = 0;
        return sz_true_k;
    }

    sz_utf8_folded_reverse_iter_t_ needle_riter, haystack_riter;
    sz_utf8_folded_reverse_iter_init_(&needle_riter, needle_start, needle_end);
    sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack_start, haystack_end);

    sz_rune_t needle_rune = 0, haystack_rune = 0;
    for (;;) {
        sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_rune);

        // Needle exhausted - success! Unconsumed haystack runes are OK.
        // Example: "fi" matches suffix of "ﬃ" (folds to "ffi"), leaving first 'f' unused.
        if (!have_needle) {
            *match_length = (sz_size_t)(haystack_end - haystack_riter.ptr);
            return sz_true_k;
        }

        sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_rune);
        if (!have_haystack) return sz_false_k;
        if (needle_rune != haystack_rune) return sz_false_k;
    }
}

/**
 *  @brief Verify tail region uncasedly (forward iteration).
 *
 *  Walks forward, comparing folded runes. Returns true if needle exhausts.
 *
 *  @param needle_start Start of needle tail region.
 *  @param needle_end End of needle tail region (= needle + needle_length).
 *  @param haystack_start Start of haystack tail region.
 *  @param haystack_end End of haystack (upper bound for forward scan).
 *  @param match_length Haystack bytes consumed by this match.
 */
SZ_HELPER_AUTO sz_bool_t sz_utf8_uncased_verify_tail_(sz_cptr_t needle_start, sz_cptr_t needle_end,
                                                      sz_cptr_t haystack_start, sz_cptr_t haystack_end,
                                                      sz_size_t *match_length) {

    sz_size_t needle_length = (sz_size_t)(needle_end - needle_start);

    // Empty tail is trivially matched
    if (needle_length == 0) {
        *match_length = 0;
        return sz_true_k;
    }

    sz_utf8_folded_iter_t_ needle_iter, haystack_iter;
    sz_utf8_folded_iter_init_(&needle_iter, needle_start, needle_length);
    sz_utf8_folded_iter_init_(&haystack_iter, haystack_start, (sz_size_t)(haystack_end - haystack_start));

    sz_rune_t needle_rune = 0, haystack_rune = 0;
    for (;;) {
        sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_rune);

        if (!have_needle) {
            // Needle exhausted - success!
            *match_length = (sz_size_t)(haystack_iter.ptr - haystack_start);
            return sz_true_k;
        }

        sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_rune);
        if (!have_haystack) return sz_false_k;
        if (needle_rune != haystack_rune) return sz_false_k;
    }
}

/**
 *  @brief Verify a complete match around a SIMD-detected window.
 *
 *  Verifies two regions: "head" (before window) and "tail" (after window).
 *  It's important to note that the middle part may still be in part unprocessed, if its larger
 *  than the "folded slice" of the needle. We handle it as part of the "tail" and the `needle_tail_bytes`
 *  must be calculated accordingly.
 *
 *  @param haystack Haystack start pointer, arbitrary case.
 *  @param haystack_length Haystack length in bytes.
 *  @param needle Needle start pointer, arbitrary case.
 *  @param needle_length Needle length in bytes.
 *  @param haystack_matched_offset Start offset of matched safe window in haystack in bytes.
 *  @param haystack_matched_length Length of matched safe window in haystack in bytes.
 *  @param needle_head_bytes Start of matched safe window in needle in bytes.
 *  @param needle_tail_bytes Number of bytes in the needle remaining after the matched part.
 *  @param match_length Total length of the verified match in haystack bytes.
 *  @return Match start pointer, or SZ_NULL_CHAR if validation fails.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_verify_match_(                   //
    sz_cptr_t haystack, sz_size_t haystack_length,                        //
    sz_cptr_t needle, sz_size_t needle_length,                            //
    sz_size_t haystack_matched_offset, sz_size_t haystack_matched_length, //
    sz_size_t needle_head_bytes, sz_size_t needle_tail_bytes,             //
    sz_size_t *match_length) {

    sz_cptr_t needle_end = needle + needle_length;
    sz_cptr_t haystack_end = haystack + haystack_length;

    // Verify head using backward iterators
    sz_size_t head_match_length = 0;
    if (needle_head_bytes)
        if (!sz_utf8_uncased_verify_head_(                    //
                needle, needle + needle_head_bytes,           // needle head region
                haystack, haystack + haystack_matched_offset, // haystack head region
                &head_match_length))
            return SZ_NULL_CHAR;

    // Verify tail using forward iterators
    sz_size_t tail_match_length = 0;
    sz_cptr_t haystack_tail_start = haystack + haystack_matched_offset + haystack_matched_length;
    if (needle_tail_bytes)
        if (!sz_utf8_uncased_verify_tail_(                              //
                needle + needle_length - needle_tail_bytes, needle_end, // needle tail region
                haystack_tail_start, haystack_end,                      // haystack tail region
                &tail_match_length))
            return SZ_NULL_CHAR;

    *match_length = head_match_length + haystack_matched_length + tail_match_length;
    return haystack + haystack_matched_offset - head_match_length;
}

/**
 *  @brief Hash-free uncased search for needles that fold to exactly 1 rune.
 *      Examples: 'a', 'A', 'б', 'Б' (but NOT 'ß' (U+00DF, C3 9F) → "ss" = 2 runes).
 *
 *  Single-pass algorithm: parses each source rune, folds it, checks if it produces
 *  exactly one rune matching the target. No iterator overhead, no verification needed.
 *
 *  @param haystack Pointer to the haystack string to search within.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle_folded The single folded rune to search for.
 *  @param match_length Output: length of the matched rune in haystack bytes on success.
 *  @return Pointer to the first matching rune, or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_1folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,               //
    sz_rune_t needle_folded, sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    // Each haystack rune may fold in up to 3 runes
    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;

    // If we simply initialize the runes for zero, the code will break
    // when the needle itself is the NUL character
    sz_rune_t haystack_folded_runes[3] = {~needle_folded};
    while (haystack < haystack_end) {
        // A byte that does not begin a well-formed codepoint folds to itself and matches byte-for-byte;
        // resync by one byte. Fill the unused fold slots with sentinels so they never false-match.
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[0] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            haystack_folded_runes[1] = ~needle_folded;
            haystack_folded_runes[2] = ~needle_folded;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes); }

        // Perform branchless equality check via arithmetic
        sz_u32_t has_match =                              //
            (haystack_folded_runes[0] == needle_folded) + //
            (haystack_folded_runes[1] == needle_folded) + //
            (haystack_folded_runes[2] == needle_folded);

        if (has_match) {
            *match_length = haystack_rune_length;
            return haystack;
        }

        haystack += haystack_rune_length;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

/**
 *  @brief Search a "danger zone" region using 1-folded candidate search + validation.
 *
 *  When SIMD kernels detect potentially problematic bytes (ligatures, Greek Extended, etc.),
 *  they fall back to this serial search within the affected chunk. This function:
 *  1. Extracts the first folded rune from the needle's safe window
 *  2. Searches for candidates matching that rune
 *  3. Validates each candidate using the full verification pipeline
 *
 *  @param haystack Full haystack string, arbitrary case.
 *  @param haystack_length Full haystack length in bytes.
 *  @param needle Full needle string, arbitrary case.
 *  @param needle_length Full needle length.
 *  @param danger_cursor Start of the danger zone region to search.
 *  @param danger_length Length of the danger zone region in bytes.
 *  @param needle_first_safe_folded_rune The first rune of the safe window, folded.
 *  @param needle_first_safe_folded_rune_offset Offset of the safe window within the needle.
 *  @param match_length Haystack bytes consumed by the match.
 *  @return Pointer to match start, or SZ_NULL_CHAR if not found in this region.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_in_danger_zone_( //
    sz_cptr_t haystack, sz_size_t haystack_length,               //
    sz_cptr_t needle, sz_size_t needle_length,                   //
    sz_cptr_t danger_cursor, sz_size_t danger_length,            //
    sz_rune_t needle_first_safe_folded_rune,                     //
    sz_size_t needle_first_safe_folded_rune_offset,              //
    sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_cptr_t const danger_end = sz_min_of_two(danger_cursor + danger_length, haystack_end);
    while (danger_cursor < danger_end) {

        // Skip continuation bytes - they are mid-sequence, not valid rune starts.
        // Without this check, a continuation byte like 0xBA could be misinterpreted as U+00BA (º),
        // causing false matches when the danger zone starts mid-character.
        sz_u8_t lead_byte = *(sz_u8_t const *)danger_cursor;
        if ((lead_byte & 0xC0) == 0x80) {
            danger_cursor++;
            continue;
        }

        // The following part is practically the unpacked variant of `sz_utf8_uncased_search_1folded_serial_`,
        // that finds the first occurrence of the `needle_first_safe_folded_rune` haystack. The issue is that each one
        // `haystack_rune` may unpack into multiple `haystack_folded_runes`.
        sz_rune_t haystack_rune;
        sz_rune_length_t haystack_rune_length;
        sz_rune_t haystack_folded_runes[3] = {~needle_first_safe_folded_rune};
        // A byte that does not begin a well-formed codepoint folds to itself and resyncs by one byte.
        haystack_rune_length = sz_rune_decode(danger_cursor, haystack_end, &haystack_rune);
        sz_size_t haystack_folded_runes_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[0] = sz_rune_malformed_byte_((sz_u8_t)*danger_cursor);
            haystack_folded_runes_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { haystack_folded_runes_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes); }

        // The simplest case is when the very first in `haystack_folded_runes` is our target:
        if (haystack_folded_runes[0] == needle_first_safe_folded_rune) {
            // Validate the full match using the unified validator
            sz_cptr_t match = sz_utf8_uncased_verify_match_( //
                haystack, haystack_length,                   //
                needle, needle_length,                       //
                danger_cursor - haystack, 0,                 // No pre-matched middle
                needle_first_safe_folded_rune_offset,
                needle_length - needle_first_safe_folded_rune_offset, // Verify everything after head serially
                match_length);

            if (match) return match;
            else { goto consider_second_haystack_folded_rune; } // We fall through here anyways :)
        }

    consider_second_haystack_folded_rune:

        // Check for a match at the second position in the folded haystack rune sequence
        if (haystack_folded_runes_count > 1 && haystack_folded_runes[1] == needle_first_safe_folded_rune) {
            sz_cptr_t haystack_match_start = 0, haystack_match_end = 0;

            // Check if the previous characters in the needle match the haystack before the danger zone began
            sz_rune_t needle_riter_rune = 0, haystack_riter_rune = 0;
            sz_utf8_folded_reverse_iter_t_ needle_riter, haystack_riter;
            sz_utf8_folded_reverse_iter_init_(&needle_riter, needle, needle + needle_first_safe_folded_rune_offset);
            sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack, danger_cursor);

            // Check if we even have needle bytes to check
            {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[0])
                    goto consider_third_haystack_folded_rune;
            }

            // Loop backwards until we exhaust the needle head or find a mismatch
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);

                // Needle exhausted - success!
                if (!have_needle) {
                    haystack_match_start = haystack_riter.ptr;
                    break;
                }

                sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_riter_rune);
                if (!have_haystack) goto consider_third_haystack_folded_rune;
                if (needle_riter_rune != haystack_riter_rune) goto consider_third_haystack_folded_rune;
            }

            // First match the tail (from safe window start forward)
            sz_rune_t needle_iter_rune = 0, haystack_iter_rune = 0;
            sz_utf8_folded_iter_t_ needle_iter, haystack_iter;
            sz_utf8_folded_iter_init_(&needle_iter, needle + needle_first_safe_folded_rune_offset,
                                      needle_length - needle_first_safe_folded_rune_offset);
            sz_utf8_folded_iter_init_(&haystack_iter, danger_cursor + haystack_rune_length,
                                      haystack_end - (danger_cursor + haystack_rune_length));

            // Pop the `needle_first_safe_folded_rune` from the forward iterator
            {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                sz_assert_(have_needle && needle_iter_rune == needle_first_safe_folded_rune);
            }

            // In some cases we already have the first point of comparison in the `haystack_folded_runes[2]`
            if (haystack_folded_runes_count == 3) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                if (have_needle && needle_iter_rune != haystack_folded_runes[2])
                    goto consider_third_haystack_folded_rune;
            }

            // Match the remaining tail runes
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);

                // Needle exhausted - success!
                if (!have_needle) {
                    haystack_match_end = haystack_iter.ptr;
                    break;
                }

                sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_iter_rune);
                if (!have_haystack) goto consider_third_haystack_folded_rune;
                if (needle_iter_rune != haystack_iter_rune) goto consider_third_haystack_folded_rune;
            }

            // Check if we have a match to report
            if (haystack_match_start != 0 && haystack_match_end != 0) {
                *match_length = (sz_size_t)(haystack_match_end - haystack_match_start);
                return haystack_match_start;
            }
        }

    consider_third_haystack_folded_rune:

        // Check for a match at the second position in the folded haystack rune sequence
        if (haystack_folded_runes_count > 2 && haystack_folded_runes[2] == needle_first_safe_folded_rune) {
            sz_cptr_t haystack_match_start = 0, haystack_match_end = 0;

            // Check if the previous characters in the needle match the haystack before the danger zone began
            sz_rune_t needle_riter_rune = 0, haystack_riter_rune = 0;
            sz_utf8_folded_reverse_iter_t_ needle_riter, haystack_riter;
            sz_utf8_folded_reverse_iter_init_(&needle_riter, needle, needle + needle_first_safe_folded_rune_offset);
            sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack, danger_cursor);

            // Check if we even have needle bytes to check
            {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[1])
                    goto consider_following_haystack_runes;
                have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[0])
                    goto consider_following_haystack_runes;
            }

            // Loop backwards until we exhaust the needle head or find a mismatch
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);

                // Needle exhausted - success!
                if (!have_needle) {
                    haystack_match_start = haystack_riter.ptr;
                    break;
                }

                sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_riter_rune);
                if (!have_haystack) goto consider_following_haystack_runes;
                if (needle_riter_rune != haystack_riter_rune) goto consider_following_haystack_runes;
            }

            // First match the tail (from safe window start forward)
            sz_rune_t needle_iter_rune = 0, haystack_iter_rune = 0;
            sz_utf8_folded_iter_t_ needle_iter, haystack_iter;
            sz_utf8_folded_iter_init_(&needle_iter, needle + needle_first_safe_folded_rune_offset,
                                      needle_length - needle_first_safe_folded_rune_offset);
            sz_utf8_folded_iter_init_(&haystack_iter, danger_cursor + haystack_rune_length,
                                      haystack_end - (danger_cursor + haystack_rune_length));

            // Pop the `needle_first_safe_folded_rune` from the forward iterator
            {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                sz_assert_(have_needle && needle_iter_rune == needle_first_safe_folded_rune);
            }

            // Match the remaining tail runes
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);

                // Needle exhausted - success!
                if (!have_needle) {
                    haystack_match_end = haystack_iter.ptr;
                    break;
                }

                sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_iter_rune);
                if (!have_haystack) goto consider_following_haystack_runes;
                if (needle_iter_rune != haystack_iter_rune) goto consider_following_haystack_runes;
            }

            // Check if we have a match to report
            if (haystack_match_start != 0 && haystack_match_end != 0) {
                *match_length = (sz_size_t)(haystack_match_end - haystack_match_start);
                return haystack_match_start;
            }
        }

    consider_following_haystack_runes:
        // Move to next candidate
        danger_cursor += haystack_rune_length;
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief Hash-free uncased search for needles that fold to exactly 2 runes.
 *      Examples: 'ab', 'AB', 'ß' (U+00DF) → "ss", 'ﬁ' (U+FB01) → "fi".
 *
 *  Single-pass sliding window over the folded rune stream. Handles expansions
 *  by buffering folded runes from each source and tracking source boundaries.
 *
 *  @param haystack Pointer to the haystack string to search within.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param first_needle_folded First folded rune of the 2-rune needle.
 *  @param second_needle_folded Second folded rune of the 2-rune needle.
 *  @param match_length Output: length of the matched region in haystack bytes on success.
 *  @return Pointer to the first match, or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_2folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,               //
    sz_rune_t first_needle_folded, sz_rune_t second_needle_folded, sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;

    // Source-codepoint begin pointer for the single history slot (slot [0]). It is never read
    // until at least one codepoint has been processed, because the sentinel `~first_needle_folded`
    // in slot [0] can never equal `first_needle_folded`, so `match_at_01` is 0 on the first step.
    sz_cptr_t first_history_source_begin = haystack;

    // If we simply initialize the runes for zero, the code will break
    // when the needle itself is the NUL character
    sz_rune_t haystack_folded_runes[4] = {~first_needle_folded};
    while (haystack < haystack_end) {
        // A byte that does not begin a well-formed codepoint folds to itself and resyncs by one byte.
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);

        // Pre-fill positions [2] and [3] with sentinels before folding.
        // The fold will overwrite positions it uses; unused positions keep the sentinel.
        // This branchlessly prevents stale data from causing false matches.
        sz_rune_t sentinel = ~second_needle_folded;
        haystack_folded_runes[2] = sentinel;
        haystack_folded_runes[3] = sentinel;
        // Export into the last 3 rune entries of the 4-element array,
        // keeping the first position with historical data untouched
        sz_size_t folded_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[1] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            folded_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { folded_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes + 1); }

        // Perform branchless equality check via arithmetic
        sz_u32_t has_match_f0 = first_needle_folded == haystack_folded_runes[0];
        sz_u32_t has_match_f1 = first_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_f2 = first_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s1 = second_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_s2 = second_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s3 = second_needle_folded == haystack_folded_runes[3];

        // Branchless match detection: each product is 0 or 1
        sz_u32_t match_at_01 = has_match_f0 * has_match_s1;
        sz_u32_t match_at_12 = has_match_f1 * has_match_s2;
        sz_u32_t match_at_23 = has_match_f2 * has_match_s3;
        sz_u32_t has_match = match_at_01 + match_at_12 + match_at_23;

        if (has_match) {
            // The first matched rune is in history slot [0] only for `match_at_01`; for `match_at_12`
            // and `match_at_23` it is in the current codepoint. The last matched rune is always within
            // the current codepoint, so the match always ends at the current codepoint's end.
            sz_cptr_t match_begin = match_at_01 ? first_history_source_begin : haystack;
            sz_cptr_t match_end = haystack + haystack_rune_length;
            *match_length = (sz_size_t)(match_end - match_begin);
            return match_begin;
        }

        // The history slot is always fed by the codepoint just processed.
        haystack_folded_runes[0] = haystack_folded_runes[folded_count];
        first_history_source_begin = haystack;
        haystack += haystack_rune_length;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

/**
 *  @brief Hash-free uncased search for needles that fold to exactly 3 runes.
 *      Examples: 'abc', 'ABC', "aß" → "ass", "ﬁa" (U+FB01) → "fia".
 *
 *  Single-pass sliding window of 3 folded runes over the haystack's folded stream.
 *  Handles expansions by buffering folded runes and tracking source boundaries.
 *
 *  @param haystack Pointer to the haystack string to search within.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param first_needle_folded First folded rune of the 3-rune needle.
 *  @param second_needle_folded Second folded rune of the 3-rune needle.
 *  @param third_needle_folded Third folded rune of the 3-rune needle.
 *  @param match_length Output: length of the matched region in haystack bytes on success.
 *  @return Pointer to the first match, or SZ_NULL_CHAR if not found.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_3folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,               //
    sz_rune_t first_needle_folded, sz_rune_t second_needle_folded, sz_rune_t third_needle_folded,
    sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;

    // Source-codepoint begin pointers for the two history slots ([0] and [1]). Never read until the
    // corresponding slot holds a real (non-sentinel) rune, because the sentinels in slots [0],[1] can
    // never match their needle runes.
    sz_cptr_t history_source_begin[2] = {haystack, haystack};

    // Initialize historical slots with sentinels that can never match their respective needle positions
    // This prevents false matches on first iterations when history is not yet populated
    sz_rune_t haystack_folded_runes[5] = {~first_needle_folded, ~second_needle_folded, 0, 0, 0};
    while (haystack < haystack_end) {
        // A byte that does not begin a well-formed codepoint folds to itself and resyncs by one byte.
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);

        // Pre-fill positions [3] and [4] with sentinels before folding.
        // The fold will overwrite positions it uses; unused positions keep the sentinel.
        // This branchlessly prevents stale data from causing false matches.
        sz_rune_t sentinel = ~third_needle_folded;
        haystack_folded_runes[3] = sentinel;
        haystack_folded_runes[4] = sentinel;
        // Export into the last 3 rune entries of the 5-element array,
        // keeping the first two positions with historical data untouched
        sz_size_t folded_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[2] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            folded_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { folded_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes + 2); }

        // Perform branchless equality check via arithmetic
        sz_u32_t has_match_f0 = first_needle_folded == haystack_folded_runes[0];
        sz_u32_t has_match_f1 = first_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_f2 = first_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s1 = second_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_s2 = second_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s3 = second_needle_folded == haystack_folded_runes[3];
        sz_u32_t has_match_t2 = third_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_t3 = third_needle_folded == haystack_folded_runes[3];
        sz_u32_t has_match_t4 = third_needle_folded == haystack_folded_runes[4];

        // Branchless match detection: each product is 0 or 1
        sz_u32_t match_at_012 = has_match_f0 * has_match_s1 * has_match_t2;
        sz_u32_t match_at_123 = has_match_f1 * has_match_s2 * has_match_t3;
        sz_u32_t match_at_234 = has_match_f2 * has_match_s3 * has_match_t4;
        sz_u32_t has_match = match_at_012 + match_at_123 + match_at_234;

        if (has_match) {
            // First matched rune slot: [0] for `match_at_012`, [1] for `match_at_123` (both history),
            // [2] for `match_at_234` (the current codepoint). The last matched rune is always within
            // the current codepoint, so the match always ends at the current codepoint's end.
            sz_cptr_t match_begin;
            if (match_at_012) match_begin = history_source_begin[0];
            else if (match_at_123) match_begin = history_source_begin[1];
            else match_begin = haystack;
            sz_cptr_t match_end = haystack + haystack_rune_length;
            *match_length = (sz_size_t)(match_end - match_begin);
            return match_begin;
        }

        // Mirror the folded-rune shift for the per-slot source-begin pointers.
        if (folded_count >= 2) {
            // Both new history runes come from the current codepoint.
            haystack_folded_runes[0] = haystack_folded_runes[folded_count];
            haystack_folded_runes[1] = haystack_folded_runes[folded_count + 1];
            history_source_begin[0] = haystack;
            history_source_begin[1] = haystack;
        }
        else {
            sz_assert_(folded_count == 1);
            // Slot [0] inherits old slot [1]; slot [1] takes the current codepoint.
            haystack_folded_runes[0] = haystack_folded_runes[1];
            haystack_folded_runes[1] = haystack_folded_runes[2];
            history_source_begin[0] = history_source_begin[1];
            history_source_begin[1] = haystack;
        }

        haystack += haystack_rune_length;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_uncased_search_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,       //
    sz_cptr_t needle, sz_size_t needle_length,           //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *match_length) {

    (void)needle_metadata; // Only used by SIMD kernels for debugging

    if (needle_length == 0) {
        *match_length = 0;
        return haystack;
    }

    if (sz_utf8_find_cased_serial(needle, needle_length) == SZ_NULL_CHAR) {
        sz_cptr_t result = sz_find_serial(haystack, haystack_length, needle, needle_length);
        if (result) {
            *match_length = needle_length;
            return result;
        }
        *match_length = 0;
        return SZ_NULL_CHAR;
    }

    // For short needles (up to 12 bytes which can fold to at most ~6 runes), try hash-free search.
    // We fold the needle first and dispatch based on the folded rune count.
    // This avoids ring buffer setup, hash multiplier computation, and rolling hash updates.
    if (needle_length <= 12) {
        sz_rune_t folded_runes[4]; // 4th slot accessed before loop exit
        sz_size_t folded_count = 0;
        sz_utf8_folded_iter_t_ iter;
        sz_utf8_folded_iter_init_(&iter, needle, needle_length);
        sz_rune_t rune;
        while (folded_count < 4 && sz_utf8_folded_iter_next_(&iter, &rune)) folded_runes[folded_count++] = rune;

        // Dispatch based on folded rune count
        switch (folded_count) {
        case 1:
            return sz_utf8_uncased_search_1folded_serial_( //
                haystack, haystack_length,                 //
                folded_runes[0], match_length);
        case 2:
            return sz_utf8_uncased_search_2folded_serial_( //
                haystack, haystack_length,                 //
                folded_runes[0], folded_runes[1], match_length);
        case 3:
            return sz_utf8_uncased_search_3folded_serial_( //
                haystack, haystack_length,                 //
                folded_runes[0], folded_runes[1], folded_runes[2], match_length);
        default: break; // 4+ folded runes: fall through to Rabin-Karp
        }
    }

    sz_size_t const ring_capacity = 32;
    sz_rune_t needle_runes[32];
    sz_size_t needle_prefix_count = 0, needle_total_count = 0;
    sz_u64_t needle_hash = 0;
    {
        sz_utf8_folded_iter_t_ needle_iter;
        sz_utf8_folded_iter_init_(&needle_iter, needle, needle_length);
        sz_rune_t rune;
        while (needle_prefix_count < ring_capacity && sz_utf8_folded_iter_next_(&needle_iter, &rune)) {
            needle_runes[needle_prefix_count++] = rune;
            needle_hash = needle_hash * 257 + rune;
        }
        needle_total_count = needle_prefix_count;
        // For long needles, count remaining runes beyond ring buffer
        while (sz_utf8_folded_iter_next_(&needle_iter, &rune)) needle_total_count++;
    }
    if (!needle_prefix_count) {
        *match_length = 0;
        return SZ_NULL_CHAR;
    }

    sz_u64_t hash_multiplier = 1;
    for (sz_size_t lane_index = 1; lane_index < needle_prefix_count; ++lane_index) hash_multiplier *= 257;

    sz_rune_t window_runes[32];
    sz_cptr_t window_sources[32];     // Byte position of character that produced each window rune
    sz_size_t window_skip_counts[32]; // Runes to skip from first character's expansion
    sz_size_t ring_head = 0;
    sz_u64_t window_hash = 0;
    sz_utf8_folded_iter_t_ haystack_iter;
    sz_utf8_folded_iter_init_(&haystack_iter, haystack, haystack_length);

    sz_cptr_t window_start = haystack;
    sz_cptr_t current_source = haystack;
    sz_size_t current_skip = 0;
    sz_size_t window_count = 0;

    while (window_count < needle_prefix_count) {
        sz_cptr_t pre_advance_cursor = haystack_iter.ptr;
        sz_rune_t rune;
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &rune)) break;
        window_runes[window_count] = rune;
        // Update source and skip only when starting a new character (not mid-expansion)
        if (haystack_iter.pending_idx <= 1 || haystack_iter.pending_count == 0) {
            current_source = pre_advance_cursor;
            current_skip = 0;
        }
        window_sources[window_count] = current_source;
        window_skip_counts[window_count] = current_skip;
        window_hash = window_hash * 257 + rune;
        window_count++;
        // For next rune from same expansion, increment skip
        if (haystack_iter.pending_idx > 0 && haystack_iter.pending_idx < haystack_iter.pending_count)
            current_skip = haystack_iter.pending_idx;
    }
    if (window_count < needle_prefix_count) {
        *match_length = 0;
        return SZ_NULL_CHAR;
    }
    sz_cptr_t window_end = haystack_iter.ptr;

    for (;;) {
        if (window_hash == needle_hash) {
            // The ring buffer is a circular array where `ring_head` points to the oldest (first) element.
            // A naive approach would use `window_runes[(ring_head + i) % needle_prefix_count]` for each comparison,
            // but modulo is expensive. Instead, we compare in two contiguous segments:
            //   - First segment:  window_runes[ring_head..needle_prefix_count) maps to needle_runes[0..first_segment)
            //   - Second segment: window_runes[0..ring_head) maps to needle_runes[first_segment..needle_prefix_count)
            sz_size_t first_segment = needle_prefix_count - ring_head;
            sz_size_t mismatches = 0;
            for (sz_size_t lane_index = 0; lane_index < first_segment; ++lane_index)
                mismatches += window_runes[ring_head + lane_index] != needle_runes[lane_index];
            for (sz_size_t lane_index = 0; lane_index < ring_head; ++lane_index)
                mismatches += window_runes[lane_index] != needle_runes[first_segment + lane_index];

            if (!mismatches) {
                sz_size_t skip_runes = window_skip_counts[ring_head];
                // Short needle: rune comparison above is sufficient verification
                if (needle_total_count <= ring_capacity) {
                    *match_length = (sz_size_t)(window_end - window_start);
                    return window_start;
                }
                // Long needle: verify FULL needle from window_start, skipping runes if match
                // starts mid-expansion. Example: ẚ→"aʾ", needle starting with "ʾ" must skip "a".
                sz_utf8_folded_iter_t_ verify_haystack_iter;
                sz_utf8_folded_iter_init_(&verify_haystack_iter, window_start,
                                          (sz_size_t)(haystack + haystack_length - window_start));
                // Skip runes within first character's expansion
                sz_rune_t skip_rune;
                for (sz_size_t skip_index = 0; skip_index < skip_runes; ++skip_index)
                    sz_utf8_folded_iter_next_(&verify_haystack_iter, &skip_rune);
                // Now verify full needle against remaining haystack
                sz_utf8_folded_iter_t_ verify_needle_iter;
                sz_utf8_folded_iter_init_(&verify_needle_iter, needle, needle_length);
                sz_rune_t needle_rune_v, haystack_rune_v;
                sz_bool_t match_ok = sz_true_k;
                while (sz_utf8_folded_iter_next_(&verify_needle_iter, &needle_rune_v)) {
                    if (!sz_utf8_folded_iter_next_(&verify_haystack_iter, &haystack_rune_v) ||
                        needle_rune_v != haystack_rune_v) {
                        match_ok = sz_false_k;
                        break;
                    }
                }
                if (match_ok) {
                    *match_length = (sz_size_t)(verify_haystack_iter.ptr - window_start);
                    return window_start;
                }
            }
        }

        sz_cptr_t pre_advance_cursor = haystack_iter.ptr;
        sz_rune_t new_rune;
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &new_rune)) break;

        window_hash -= window_runes[ring_head] * hash_multiplier;
        window_hash = window_hash * 257 + new_rune;

        // Advance ring head, avoiding modulo with a conditional (cheaper than integer division)
        sz_size_t next_head = ring_head + 1;
        next_head = next_head == needle_prefix_count ? 0 : next_head;

        window_runes[ring_head] = new_rune;
        // Update source and skip only when starting a new character (not mid-expansion)
        if (haystack_iter.pending_idx <= 1 || haystack_iter.pending_count == 0) {
            current_source = pre_advance_cursor;
            current_skip = 0;
        }
        window_sources[ring_head] = current_source;
        window_skip_counts[ring_head] = current_skip;
        // For next rune from same expansion, increment skip
        if (haystack_iter.pending_idx > 0 && haystack_iter.pending_idx < haystack_iter.pending_count)
            current_skip = haystack_iter.pending_idx;

        ring_head = next_head;
        window_start = window_sources[ring_head];
        window_end = haystack_iter.ptr;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

#pragma endregion // Substring Search

#pragma region Character Safety Profiles

/*  The character safety classifier and needle-metadata builder are ISA-agnostic: they only depend
 *  on the serial Unicode core (rune parsing & folding). The SIMD kernels (Ice Lake, etc.) consume
 *  the metadata it produces, so it lives here in the serial scaffolding rather than behind any
 *  `SZ_USE_*` gate, keeping it reachable for every backend including pure serial builds. */

/**
 *  @brief Determine safety profile for a character across all script contexts.
 *
 *  This function encodes the contextual safety rules from the ASCII selector
 *  and applies them consistently to all paths that include ASCII.
 *
 *  @param rune The decoded codepoint.
 *  @param rune_bytes UTF-8 byte length of this codepoint (1-4).
 *  @param prev_rune Previous codepoint (0 if at start).
 *  @param next_rune Next codepoint (0 if at end).
 *  @param prev_prev_rune Codepoint before prev_rune (0 if prev is at start).
 *  @param next_next_rune Codepoint after next_rune (0 if next is at end).
 *  @param safety_profiles Safety flags for each script path.
 *  @return The primary fast path preferred for this rune.
 *
 *  @note Using 0 for boundary markers is safe even though NUL (U+0000) is a valid
 *        codepoint in StringZilla's length-based strings. This works because:
 *        1. NUL is valid ASCII (< 0x80), so boundary and actual NUL are treated identically
 *        2. Ligature checks use inequality (lower_prev != 'f'), and 0 never matches letters
 *        3. NUL doesn't participate in any Unicode case folding or ligature expansions
 *
 *  @note The neighbor-of-neighbor context (prev_prev, next_next) enables position-1 and
 *        position-N-2 detection for the 's' rule: if prev_prev==0 && prev!=0, we're at
 *        position 1; if next_next==0 && next!=0, we're at position N-2.
 */
SZ_HELPER_AUTO sz_utf8_uncased_rune_safety_profile_t_ sz_utf8_uncased_rune_safety_profile_( //
    sz_rune_t rune, sz_size_t rune_bytes,                                                   //
    sz_rune_t prev_rune, sz_rune_t next_rune,                                               //
    sz_rune_t prev_prev_rune, sz_rune_t next_next_rune,                                     //
    unsigned int *safety_profiles) {

    unsigned safety = 0;

    // Bitmasks for profiles that share identical ASCII rules
    unsigned int western_group = //
        (1 << sz_utf8_uncased_rune_safe_western_europe_k);
    unsigned int central_viet_group =                       //
        (1 << sz_utf8_uncased_rune_safe_central_europe_k) | //
        (1 << sz_utf8_uncased_rune_safe_vietnamese_k);
    unsigned int strict_ascii_group =                   //
        (1 << sz_utf8_uncased_rune_ascii_invariant_k) | //
        (1 << sz_utf8_uncased_rune_safe_cyrillic_k) |   //
        (1 << sz_utf8_uncased_rune_safe_greek_k) |      //
        (1 << sz_utf8_uncased_rune_safe_armenian_k) |   //
        (1 << sz_utf8_uncased_rune_safe_georgian_k);

    // Helper: lowercase ASCII
    sz_rune_t lower = (rune >= 'A' && rune <= 'Z') ? (rune + 0x20) : rune;
    sz_rune_t lower_prev = (prev_rune >= 'A' && prev_rune <= 'Z') ? (prev_rune + 0x20) : prev_rune;
    sz_rune_t lower_next = (next_rune >= 'A' && next_rune <= 'Z') ? (next_rune + 0x20) : next_rune;

    // Helper: is neighbor ASCII letter? (explicit conversion for C++ compatibility)
    // Note: prev_rune/next_rune == 0 means boundary (start/end of needle)
    sz_bool_t prev_ascii = (prev_rune != 0 && prev_rune < 0x80) ? sz_true_k : sz_false_k;
    sz_bool_t next_ascii = (next_rune != 0 && next_rune < 0x80) ? sz_true_k : sz_false_k;
    sz_bool_t at_start = (prev_rune == 0) ? sz_true_k : sz_false_k;
    sz_bool_t at_end = (next_rune == 0) ? sz_true_k : sz_false_k;

    // Helper: position detection for 's' rule (mid-ß-expansion avoidance in Western profile)
    // Position 1: prev exists but prev_prev doesn't (prev is at position 0)
    // Position N-2: next exists but next_next doesn't (next is at position N-1)
    sz_bool_t at_pos_1 = (prev_rune != 0 && prev_prev_rune == 0) ? sz_true_k : sz_false_k;
    sz_bool_t at_pos_n_minus_2 = (next_rune != 0 && next_next_rune == 0) ? sz_true_k : sz_false_k;

    // ASCII character (1-byte UTF-8)
    if (rune < 0x80) {
        if (lower >= 'a' && lower <= 'z') {
            switch (lower) {

            // Unconditionally safe for all profiles.
            // No Unicode chars fold to sequences containing these,
            // and they don't participate in dangerous ligatures.
            case 'b':
            case 'c':
            case 'd':
            case 'e':
            case 'g':
            case 'm':
            case 'o':
            case 'p':
            case 'q':
            case 'r':
            case 'u':
            case 'v':
            case 'x':
            case 'z': safety |= strict_ascii_group | central_viet_group | western_group; break;

            // 'k':
            // - Strict: UNSAFE. 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B).
            // - Western/Central/Viet: SAFE. Kelvin sign detected in haystack.
            case 'k': safety |= central_viet_group | western_group; break;

            // 'a':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede 'ʾ' (U+02BE, CA BE).
            //   Avoids: 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE).
            // - Western: SAFE. Expansion detected in haystack.
            case 'a':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'h':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̱' (U+0331, CC B1).
            //   Avoids: 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1).
            // - Western: SAFE. Expansion detected in haystack.
            case 'h':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'j':
            // - All: Contextual. Can't be last; can't precede '̌' (U+030C).
            //   Avoids: 'ǰ' (U+01F0) → "ǰ" (U+006A U+030C, 6A CC 8C).
            //   Western profile does NOT detect this in haystack scan.
            case 'j':
                if (at_end == sz_false_k && next_ascii)
                    safety |= strict_ascii_group | central_viet_group | western_group;
                break;

            // 'w':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̊' (U+030A).
            //   Avoids: 'ẘ' (U+1E98) → "ẘ" (U+0077 U+030A, 77 CC 8A).
            // - Western: SAFE. Expansion detected in haystack.
            case 'w':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'y':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̊' (U+030A).
            //   Avoids: 'ẙ' (U+1E99) → "ẙ" (U+0079 U+030A, 79 CC 8A).
            // - Western: SAFE. Expansion detected in haystack.
            case 'y':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'n':
            // - ASCII/Cyrillic/Greek: Contextual. Can't be first; can't follow 'ʼ' (U+02BC, CA BC).
            //   Avoids: 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
            // - Armenian: UNSAFE. Armenian kernel cannot handle 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
            //   The character 'n' can match the 2nd part of the expansion, causing false positives.
            // - Western/Central/Viet: Contextual, same as above.
            //   Western profile does NOT detect this in haystack scan.
            case 'n':
                // Exclude Armenian - it cannot handle 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
                if (at_start == sz_false_k && prev_ascii) {
                    safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k) | //
                              (1 << sz_utf8_uncased_rune_safe_cyrillic_k) |   //
                              (1 << sz_utf8_uncased_rune_safe_greek_k);       //
                    // Armenian EXCLUDED: sz_utf8_uncased_rune_safe_armenian_k
                    safety |= central_viet_group | western_group;
                }
                break;

            // 'i':
            // - All: Contextual. Can't be first or last; can't follow 'f'; can't precede '̇' (U+0307, CC 87).
            //   Avoids: 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87),
            //   and 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69).
            //   Western profile does NOT detect Turkish 'İ' expansion.
            case 'i':
                if (at_start == sz_false_k && at_end == sz_false_k && next_ascii && lower_prev != 'f')
                    safety |= strict_ascii_group | central_viet_group | western_group;
                break;

            // 'l':
            // - Strict/Central/Viet: Contextual. Can't be first; can't follow 'f'.
            //   Avoids: 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C).
            // - Western: SAFE. Ligatures detected in haystack.
            case 'l':
                if (at_start == sz_false_k && lower_prev != 'f') safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 't':
            // - Strict/Central/Viet: Contextual. Can't be first/last; can't follow 's';
            //   can't precede '̈' (U+0308, CC 88).
            //   Avoids: 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74),
            //   'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74),
            //   and 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88).
            // - Western: SAFE. Ligatures/expansion detected in haystack.
            case 't':
                if (at_start == sz_false_k && at_end == sz_false_k && next_ascii && lower_prev != 's')
                    safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'f':
            // - Strict/Central/Viet: Contextual. Can't be first/last; can't follow 'f';
            //   can't precede 'f', 'i', 'l'.
            //   Avoids:
            //   - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
            //   - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
            //   - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
            //   - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
            //   - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
            // - Western: SAFE. Ligatures detected in haystack.
            case 'f':
                if (at_start == sz_false_k && at_end == sz_false_k && prev_ascii && next_ascii && lower_prev != 'f' &&
                    lower_next != 'f' && lower_next != 'i' && lower_next != 'l')
                    safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 's'
            // - Strict: UNSAFE. 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
            // - Central/Vietnamese: Contextual. Can't be first/last; can't be adjacent to 's'/'t'.
            //   Avoids: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73),
            //   'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74),
            //   'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74), and 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
            // - Western: Contextual. Can't be at positions 0, 1 (if prev='s'), N-1, or N-2 (if next='s').
            //   Avoids mid-ß-expansion matches: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) in-place means
            //   needle with 's' at these positions could match at byte offset 1 (UTF-8 continuation byte 0x9F).
            //   Example: "ßStra" → "ssstra", needle "sstra" matches at pos 1 = mid-character.
            //   Interior 's' like "tesst" or "masse" are safe for SIMD.
            case 's':
                if (at_start == sz_false_k && at_end == sz_false_k && prev_ascii && next_ascii && lower_prev != 's' &&
                    lower_next != 's' && lower_next != 't')
                    safety |= central_viet_group;
                // Western: ban pos 0, pos 1 if prev='s', pos N-1, pos N-2 if next='s'
                if (at_start == sz_false_k && at_end == sz_false_k && //
                    !(at_pos_1 == sz_true_k && lower_prev == 's') &&  //
                    !(at_pos_n_minus_2 == sz_true_k && lower_next == 's'))
                    safety |= western_group;
                break;

            default:
                // Should not happen for a-z
                safety |= strict_ascii_group | central_viet_group | western_group;
                break;
            }
        }
        else {
            // Non-letters (digits, punctuation, whitespace) - always safe for all profiles
            safety |= strict_ascii_group | central_viet_group | western_group;
        }

        *safety_profiles = safety;
        return sz_utf8_uncased_rune_ascii_invariant_k;
    }

    // 2-byte UTF-8 (U+0080 to U+07FF)
    // Must check EXACT ranges that the fold functions handle, not just lead bytes
    if (rune_bytes == 2) {
        sz_u8_t lead = (rune >> 6) | 0xC0;     // Reconstruct lead byte
        sz_u8_t second = (rune & 0x3F) | 0x80; // Reconstruct continuation byte

        // Latin-1 Supplement (C2/C3 lead bytes)
        // Exclude: 'å' (U+00E5, C3 A5) - Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5) also folds to it
        if (lead == 0xC2 || lead == 0xC3) {
            if (rune == 0x00E5) {
                // 'å' excluded from all Latin profiles due to Angstrom ambiguity
            }
            else if (rune == 0x00DF) {
                // 'ß' excluded from Central Europe and Vietnamese, allowed in Western Europe
                safety |= western_group;
            }
            else if (rune == 0x00B5) {
                // 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC).
                // Allow only the Greek SIMD path; Latin paths remain unsafe.
                safety |= (1 << sz_utf8_uncased_rune_safe_greek_k);
            }
            else { safety |= western_group | central_viet_group; }
        }

        // Latin Extended-A (C4/C5 lead bytes) - for central_europe and vietnamese
        if (lead == 0xC4 || lead == 0xC5) {
            // Exclude expansions/length-changes:
            // - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
            // - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
            // - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
            if (rune != 0x0130 && rune != 0x0149 && rune != 0x017F) { safety |= central_viet_group; }
        }

        // Latin Extended-B (C6 lead byte) - for vietnamese (supports ơ/ư)
        if (lead == 0xC6) { safety |= (1 << sz_utf8_uncased_rune_safe_vietnamese_k); }

        // Cyrillic - check exact ranges handled by sz_utf8_uncased_search_icelake_cyrillic_fold_zmm_
        // D0 80-BF: U+0400-U+043F (includes uppercase and lowercase)
        // D1 80-9F: U+0440-U+045F (lowercase continuation)
        // Note: D2/D3 Extended Cyrillic BANNED from SIMD kernel - needles with D2/D3 use serial fallback
        if ((lead == 0xD0 && second >= 0x80 && second <= 0xBF) || //
            (lead == 0xD1 && second >= 0x80 && second <= 0x9F)) { //
            safety |= (1 << sz_utf8_uncased_rune_safe_cyrillic_k);
        }

        // Greek - check exact ranges handled by sz_utf8_uncased_search_icelake_greek_fold_zmm_
        // CE 86-8F: accented uppercase Ά-Ώ (with gaps at 87, 8B, 8D)
        //   - EXCLUDE CE 90: 'ΐ' (U+0390) expands to 3 codepoints
        // CE 91-A9: basic uppercase Α-Ω
        // CE AA-AB: dialytika uppercase Ϊ-Ϋ
        // CE AC-AF: accented lowercase ά-ί
        //   - EXCLUDE CE B0: 'ΰ' (U+03B0) expands to 3 codepoints
        // CE B1-BF: basic lowercase α-ο
        // CF 80-89: basic lowercase π-ω (includes final sigma at 82, sigma at 83)
        // CF 8A-8E: accented/dialytika lowercase ϊ-ώ
        if (lead == 0xCE) {
            // Accented uppercase (with gaps) - exclude 87, 8B, 8D, 90
            if ((second >= 0x86 && second <= 0x8F) && second != 0x87 && second != 0x8B && second != 0x8D &&
                second != 0x90) {
                safety |= (1 << sz_utf8_uncased_rune_safe_greek_k);
            }
            // Basic uppercase Α-Ω
            if (second >= 0x91 && second <= 0xA9) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            // Dialytika uppercase Ϊ-Ϋ
            if (second >= 0xAA && second <= 0xAB) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            // Accented lowercase ά-ί
            if (second >= 0xAC && second <= 0xAF) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            // Basic lowercase α-ο - exclude B0 (ΰ expands)
            if (second >= 0xB1 && second <= 0xBF) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
        }
        if (lead == 0xCF) {
            // Basic lowercase π-ω
            if (second >= 0x80 && second <= 0x89) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            // Accented/dialytika lowercase ϊ-ώ
            if (second >= 0x8A && second <= 0x8E) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
        }

        // Armenian - check exact ranges with contextual constraints for ligatures
        // D4 B1-BF: uppercase Ա-Ի (U+0531-U+053F)
        // D5 80-96: uppercase Լ-Ֆ (U+0540-U+0556)
        // D5 A1-BF: lowercase ա-տ (U+0561-U+057F)
        // D6 80-86: lowercase ր-ֆ (U+0580-U+0586)
        //
        // Ligature constraints (from spec):
        // - 'ե' (U+0565): can't be first; can't follow 'մ'; can't precede 'ւ'
        // - 'ւ' (U+0582): can't be last; can't follow 'ե'
        // - 'մ' (U+0574): can't be last; can't precede 'ն', 'ե', 'ի', 'խ'
        // - 'ն' (U+0576): can't be first; can't follow 'մ', 'վ'
        // - 'ի' (U+056B): can't be first; can't follow 'մ'
        // - 'վ' (U+057E): can't be first; can't precede 'ն'
        // - 'խ' (U+056D): can't be first; can't follow 'մ'
        {
            sz_bool_t is_armenian_range = sz_false_k;
            sz_bool_t armenian_safe = sz_true_k;

            if ((lead == 0xD4 && second >= 0xB1 && second <= 0xBF) ||
                (lead == 0xD5 && second >= 0x80 && second <= 0x96) ||
                (lead == 0xD5 && second >= 0xA1 && second <= 0xBF) ||
                (lead == 0xD6 && second >= 0x80 && second <= 0x86)) {
                is_armenian_range = sz_true_k;

                // Helper: get lowercase Armenian codepoint for neighbor checks
                sz_rune_t lower_prev_arm = prev_rune;
                sz_rune_t lower_next_arm = next_rune;
                if (prev_rune >= 0x0531 && prev_rune <= 0x0556) lower_prev_arm = prev_rune + 0x30;
                if (next_rune >= 0x0531 && next_rune <= 0x0556) lower_next_arm = next_rune + 0x30;

                // Check ligature constraints
                switch (rune) {
                case 0x0565: // U+0565 ech - can't be first; can't follow U+0574 men; can't precede U+0582 yiwn
                case 0x0535: // U+0535 Ech uppercase
                    if (at_start || lower_prev_arm == 0x0574 || lower_next_arm == 0x0582) armenian_safe = sz_false_k;
                    break;
                case 0x0582: // U+0582 yiwn - can't be first; can't be last; can't follow U+0565 ech
                    // Armenian ligature և (U+0587) → ech + yiwn; needle starting with yiwn matches mid-expansion
                    if (at_start || at_end || lower_prev_arm == 0x0565) armenian_safe = sz_false_k;
                    break;
                case 0x0574: // U+0574 men - can't be last; can't precede U+0576, U+0565, U+056B, U+056D
                case 0x0544: // U+0544 Men uppercase
                    if (at_end || lower_next_arm == 0x0576 || lower_next_arm == 0x0565 || lower_next_arm == 0x056B ||
                        lower_next_arm == 0x056D)
                        armenian_safe = sz_false_k;
                    break;
                case 0x0576: // U+0576 now - can't be first; can't follow U+0574 men, U+057E vew
                case 0x0546: // U+0546 Now uppercase
                    if (at_start || lower_prev_arm == 0x0574 || lower_prev_arm == 0x057E) armenian_safe = sz_false_k;
                    break;
                case 0x056B: // U+056B ini - can't be first; can't follow U+0574 men
                case 0x053B: // U+053B Ini uppercase
                    if (at_start || lower_prev_arm == 0x0574) armenian_safe = sz_false_k;
                    break;
                case 0x057E: // U+057E vew - can't be last; can't precede U+0576 now
                case 0x054E: // U+054E Vew uppercase
                    if (at_end || lower_next_arm == 0x0576) armenian_safe = sz_false_k;
                    break;
                case 0x056D: // U+056D xeh - can't be first; can't follow U+0574 men
                case 0x053D: // U+053D Xeh uppercase
                    if (at_start || lower_prev_arm == 0x0574) armenian_safe = sz_false_k;
                    break;
                default: break;
                }
            }

            if (is_armenian_range && armenian_safe) { safety |= (1 << sz_utf8_uncased_rune_safe_armenian_k); }
        }

        // Output safety and determine primary script for 2-byte runes
        // For case-invariant non-ASCII runes, add the ASCII-invariant bit.
        // This enables fast ASCII kernel for needles like "中文字" that contain no cased characters.
        // ASCII fold only affects bytes 0x41-0x5A (A-Z), so all other bytes pass through unchanged.
        if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
        *safety_profiles = safety;
        if (rune >= 0x0080 && rune <= 0x00FF) return sz_utf8_uncased_rune_safe_western_europe_k; // Latin-1 Supplement
        if (rune >= 0x0100 && rune <= 0x024F) return sz_utf8_uncased_rune_safe_central_europe_k; // Latin Extended-A/B
        if (rune >= 0x0370 && rune <= 0x03FF) return sz_utf8_uncased_rune_safe_greek_k;          // Greek
        if (rune >= 0x0400 && rune <= 0x04FF) return sz_utf8_uncased_rune_safe_cyrillic_k;       // Cyrillic
        if (rune >= 0x0530 && rune <= 0x058F) return sz_utf8_uncased_rune_safe_armenian_k;       // Armenian
        return sz_utf8_uncased_rune_invariant_k;
    }

    // 3-byte UTF-8 (U+0800 to U+FFFF)
    if (rune_bytes == 3) {
        sz_u8_t lead = (rune >> 12) | 0xE0;
        sz_u8_t second = ((rune >> 6) & 0x3F) | 0x80;
        sz_u8_t third = (rune & 0x3F) | 0x80;

        // Vietnamese/Latin Extended Additional (E1 B8-BB range)
        // U+1E00-U+1EFF maps to E1 B8 80 - E1 BB BF
        if (lead == 0xE1 && (second >= 0xB8 && second <= 0xBB)) {
            // Need detailed check for exclusions in U+1E96-U+1E9F
            // 1E96-1E9F: E1 BA 96 - E1 BA 9F
            if (second == 0xBA && third >= 0x96 && third <= 0x9F) {
                // Excluded: expansions or irregulars
            }
            else { safety |= (1 << sz_utf8_uncased_rune_safe_vietnamese_k); }
        }

        // Georgian Mkhedruli (E1 83 90-BF range)
        // U+10D0-U+10FF maps to E1 83 90 - E1 83 BF
        // Mkhedruli is caseless, so all characters are safe for the Georgian kernel.
        if (lead == 0xE1 && second == 0x83 && third >= 0x90) { safety |= (1 << sz_utf8_uncased_rune_safe_georgian_k); }

        // Output safety and determine primary script for 3-byte runes
        // For case-invariant non-ASCII runes (like CJK), add the ASCII-invariant bit.
        if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
        *safety_profiles = safety;
        if (rune >= 0x10D0 && rune <= 0x10FF) return sz_utf8_uncased_rune_safe_georgian_k; // Georgian Mkhedruli
        if (rune >= 0x1E00 && rune <= 0x1EFF)
            return sz_utf8_uncased_rune_safe_vietnamese_k; // Latin Extended Additional
        return sz_utf8_uncased_rune_invariant_k;
    }

    // 4-byte UTF-8 - currently no fast paths, but case-invariant 4-byte runes can use ASCII kernel
    if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
    *safety_profiles = safety;
    return sz_utf8_uncased_rune_invariant_k;
}

/**
 *  @brief Compute diversity score for a byte sequence.
 *
 *  Uses a 256-bit bitmap to efficiently count distinct byte values.
 *  Higher scores indicate more diverse byte values, which lead to better
 *  filtering during SIMD search (fewer false positives).
 *
 *  @param data Pointer to byte sequence.
 *  @param length Length of byte sequence.
 *  @return Count of distinct byte values (0-256).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_probe_diversity_score_(sz_u8_t const *data, sz_size_t length) {
    if (length <= 1) return length;
    sz_u64_t seen[4] = {0, 0, 0, 0}; // 256-bit bitmap
    sz_size_t distinct = 0;
    for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) {
        sz_u8_t byte = data[byte_index];
        sz_size_t word = byte >> 6;                // Which 64-bit word (0-3)
        sz_u64_t bit = (sz_u64_t)1 << (byte & 63); // Bit within the word
        if (!(seen[word] & bit)) {
            seen[word] |= bit;
            ++distinct;
        }
    }
    return distinct;
}

/**
 *  @brief Find the "best safe window" in the needle for each script path.
 *
 *  The objective is as follows. For a given needle, find a slice, that when folded fits into 16 bytes
 *  and where all characters are "safe" with respect to a certain path. If no such path can be found,
 *  an empty result is returned. It might be the case for a search query like "s" or "n", that by itself
 *  isn't safe for any path given the number of Unicode characters expanding into multiple 's'- or 'n'-containing
 *  sequences. The selected safe folded slice will never begin mid-character in the needle, so if it starts with
 *  an 'ŉ' (U+0149, C5 89), we can't choose - 'n' (6E) - the second half of its folded sequence as a starting point.
 *
 *  The algorithm is as follows. Iterate through the arbitrary-case "ŉEeDlE_WITH_LONG_SUFFIX", unpacking runes.
 *  For each input rune, perform folding, expanding into a sequence, like:
 *
 *      'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
 *
 *  Continue unpacking the rest, until we reach a 16-byte limit, like:
 *
 *      ʼ     n  e  e  d  l  e  _  w  i  t  h  _  l  o  n  g
 *      CA BC 6E 45 45 44 4C 45 5F 57 49 54 48 5F 4C 4F 4E 47
 *
 *  At this point, we need to trim it to make sure - its characters satisfy boundary conditions.
 *  Assuming at the next step we'll move the iterator to the next input rune to point to 'E' (U+0045) input character,
 *  we only trim from the end. But also invalidate the whole starting position if a bad character is chosen at start.
 *  For safe window starting position we can have multiple length variants, assuming different safe paths can have
 *  different rules for the last symbol in the safe sequence.
 *
 *  Once we have safe window for a certain script, we evaluate its diversity score - the number of distinct byte
 *  values in the folded window. The more diverse - the better! We keep track of best seen window for each script.
 *
 *  We also track not only the safety with respect to a certain profile, but also applicability. For example,
 *  the needle "xyz" is safe with respect to the Western European path, as well as Central European, Vietnamese,
 *  and potentially others. But it's pure ASCII. We shouldn't pay the cost of complex Vietnamese case-folding of
 *  triple-byte Latin extensions for just "xyz". So we must invalidate the "safe path" if its just "safe", but
 *  not ideal.
 *
 *  In the end, we'll have up to 7 best safe windows, one per script path.
 *  The heuristic is:
 *
 *  - Prefer ASCII, if there is an ASCII-safe path at least 4 bytes wide with at least 4 distinct byte values.
 *    It's only one subtraction, a comparison, and a masked addition. Cheapest of all kernels.
 *  - Pick the most diverse variant from all others, if ASCII variant isn't good enough.
 *
 *  We then identify the four "probe" positions within the <= 16 byte folded safe window, one more than
 *  in exact substring search kernels with Raita heuristics:
 *
 *  1. implicit at `refined->folded_slice[0]`
 *  2. stored in `refined->probe_second` - targets last byte of 2nd character when 4+ chars
 *  3. stored in `refined->probe_third` - targets last byte of 3rd character when 4+ chars
 *  4. implicit at `refined->folded_slice[refined->folded_slice_length - 1]`
 *
 *  By aiming at the last byte of each UTF-8 codepoint we maximize diversity, as in a Russian text almost
 *  all letters will have the same first byte, but mostly different second byte. The same is true for many
 *  other languages. For short strings (< 4 bytes), probes will necessarily overlap - this is expected.
 *  The function also sets `offset_in_unfolded` and `length_in_unfolded` to track where the
 *  selected folded slice came from in the original unfolded input.
 *
 *  @param needle Pointer to needle string (original, not folded).
 *  @param needle_length Length in bytes.
 *  @param refined Output metadata structure to populate.
 */
SZ_HELPER_AUTO void sz_utf8_uncased_needle_metadata_(sz_cptr_t needle, sz_size_t needle_length, //
                                                     sz_utf8_uncased_needle_metadata_t *refined) {

    // Per-script window state during iteration
    typedef struct {
        sz_size_t start_offset;   // Byte offset in original needle
        sz_size_t input_length;   // Bytes consumed from original needle
        sz_u8_t folded_bytes[16]; // Folded content
        sz_size_t folded_length;  // Length of folded content (bytes)
        sz_bool_t applicable;     // Has >=1 primary-script character
        sz_bool_t broken;         // Window continuity broken - skip further extension
        sz_size_t diversity;      // Distinct byte count (computed at end of each starting position)
    } script_window_t_;

    // Number of script kernels (indices 1-8 used, index 0 reserved)
    sz_size_t const num_scripts = 9;

    // Best window found so far for each script
    script_window_t_ best[9];
    for (sz_size_t script_index = 0; script_index < num_scripts; ++script_index) {
        best[script_index].start_offset = 0;
        best[script_index].input_length = 0;
        best[script_index].folded_length = 0;
        best[script_index].applicable = sz_false_k;
        best[script_index].broken = sz_false_k;
        best[script_index].diversity = 0;
    }

    // Handle empty needle
    if (needle_length == 0) {
        refined->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
        refined->offset_in_unfolded = 0;
        refined->length_in_unfolded = 0;
        refined->folded_slice_length = 0;
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }

    // A needle containing any byte that does not begin a well-formed codepoint cannot be window-analyzed by the
    // unchecked decode below; route it to the serial kernel, which handles malformed bytes losslessly (each is
    // folded to itself and resyncs by one byte), keeping SIMD and serial results identical.
    if (sz_utf8_find_malformed(needle, needle_length) != SZ_NULL_CHAR) {
        refined->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
        refined->offset_in_unfolded = 0;
        refined->length_in_unfolded = 0;
        refined->folded_slice_length = 0;
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }

    sz_u8_t const *needle_start = (sz_u8_t const *)needle;
    sz_u8_t const *needle_end = needle_start + needle_length;

    // Iterate through each starting position in the needle (stepping by rune)
    for (sz_u8_t const *needle_cursor = needle_start; needle_cursor < needle_end;) {
        // Current window being built for each script at this starting position
        script_window_t_ current[9];
        for (sz_size_t script_index = 0; script_index < num_scripts; ++script_index) {
            current[script_index].start_offset = (sz_size_t)(needle_cursor - needle_start);
            current[script_index].input_length = 0;
            current[script_index].folded_length = 0;
            current[script_index].applicable = sz_false_k;
            current[script_index].broken = sz_false_k;
            current[script_index].diversity = 0;
        }

        // Track context for safety profile evaluation
        sz_rune_t prev_prev_rune = 0;
        sz_rune_t prev_rune = 0;

        // Fold forward from needle_cursor until 16 bytes or needle end
        sz_u8_t const *position = needle_cursor;
        sz_bool_t any_active = sz_true_k;

        while (position < needle_end && any_active) {
            // Parse current rune
            sz_rune_t rune;
            sz_rune_length_t const rune_bytes = sz_rune_decode_unchecked((sz_cptr_t)position, &rune);
            if (position + rune_bytes > needle_end) break; // Incomplete rune

            // Parse next rune for context (if available)
            sz_rune_t next_rune = 0;
            sz_rune_length_t next_bytes = sz_rune_invalid_k;
            if (position + rune_bytes < needle_end) {
                next_bytes = sz_rune_decode_unchecked((sz_cptr_t)(position + rune_bytes), &next_rune);
                if (position + rune_bytes + next_bytes > needle_end) next_rune = 0;
            }

            // Parse next-next rune for context
            sz_rune_t next_next_rune = 0;
            if (next_rune != 0 && position + rune_bytes + next_bytes < needle_end) {
                sz_rune_length_t const next_next_bytes = sz_rune_decode_unchecked(
                    (sz_cptr_t)(position + rune_bytes + next_bytes), &next_next_rune);
                if (position + rune_bytes + next_bytes + next_next_bytes > needle_end) next_next_rune = 0;
            }

            // Get safety mask and primary script for this rune
            unsigned safety_mask = 0;
            sz_utf8_uncased_rune_safety_profile_t_ primary_script = sz_utf8_uncased_rune_safety_profile_( //
                rune, rune_bytes, prev_rune, next_rune, prev_prev_rune, next_next_rune, &safety_mask);

            // Fold this rune
            sz_rune_t folded_runes[4];
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);

            // Convert folded runes to UTF-8 bytes
            sz_u8_t folded_utf8[16];
            sz_size_t folded_utf8_length = 0;
            for (sz_size_t rune_index = 0; rune_index < folded_count; ++rune_index) {
                folded_utf8_length += sz_rune_encode(folded_runes[rune_index], folded_utf8 + folded_utf8_length);
            }

            // Update each script's window
            any_active = sz_false_k;
            for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
                if (current[script_index].broken) continue;

                // Check if this rune is safe for this script
                sz_bool_t is_safe = (safety_mask & (1u << script_index)) ? sz_true_k : sz_false_k;

                // Check if adding this rune would exceed 16 bytes
                if (is_safe && current[script_index].folded_length + folded_utf8_length <= 16) {
                    // Extend this script's window
                    for (sz_size_t byte_index = 0; byte_index < folded_utf8_length; ++byte_index) {
                        current[script_index].folded_bytes[current[script_index].folded_length + byte_index] =
                            folded_utf8[byte_index];
                    }
                    current[script_index].folded_length += folded_utf8_length;
                    current[script_index].input_length += rune_bytes;

                    // Mark as applicable if primary script matches
                    if (primary_script == script_index) { current[script_index].applicable = sz_true_k; }
                    any_active = sz_true_k;
                }
                else {
                    // Window broken for this script
                    current[script_index].broken = sz_true_k;
                }
            }

            // Update context for next iteration
            prev_prev_rune = prev_rune;
            prev_rune = rune;
            position += rune_bytes;
        }

        // Compare current to best for each script
        for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
            if (!current[script_index].applicable || current[script_index].folded_length == 0) continue;

            // Compute diversity score
            current[script_index].diversity = sz_utf8_probe_diversity_score_(current[script_index].folded_bytes,
                                                                             current[script_index].folded_length);

            // Update best if this is better (prefer higher diversity, then longer length)
            if (current[script_index].diversity > best[script_index].diversity ||
                (current[script_index].diversity == best[script_index].diversity &&
                 current[script_index].folded_length > best[script_index].folded_length)) {
                best[script_index] = current[script_index];
            }
        }

        // Advance to next rune for next starting position
        sz_rune_t skip_rune;
        sz_rune_length_t const skip_length = sz_rune_decode_unchecked((sz_cptr_t)needle_cursor, &skip_rune);
        needle_cursor += skip_length;
    }

    // Select final kernel based on best windows
    // Rule: Prefer ASCII if >=4 bytes with >=4 diversity; otherwise pick most diverse applicable
    sz_size_t chosen_script = 0;
    sz_size_t best_diversity = 0;

    // Check ASCII preference
    if (best[sz_utf8_uncased_rune_ascii_invariant_k].applicable &&
        best[sz_utf8_uncased_rune_ascii_invariant_k].folded_length >= 4 &&
        best[sz_utf8_uncased_rune_ascii_invariant_k].diversity >= 4) {
        chosen_script = sz_utf8_uncased_rune_ascii_invariant_k;
    }
    else {
        // Find most diverse applicable script
        for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
            if (best[script_index].applicable && best[script_index].diversity > best_diversity) {
                best_diversity = best[script_index].diversity;
                chosen_script = script_index;
            }
        }
    }

    // If no applicable window found, fall back to serial
    if (chosen_script == 0) {
        refined->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
        refined->offset_in_unfolded = 0;
        refined->length_in_unfolded = 0;
        refined->folded_slice_length = 0;
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }

    // Populate output metadata
    refined->kernel_id = (sz_u8_t)chosen_script;
    refined->offset_in_unfolded = best[chosen_script].start_offset;
    refined->length_in_unfolded = best[chosen_script].input_length;
    refined->folded_slice_length = (sz_u8_t)best[chosen_script].folded_length;

    // Copy folded bytes
    for (sz_size_t byte_index = 0; byte_index < best[chosen_script].folded_length; ++byte_index) {
        refined->folded_slice[byte_index] = best[chosen_script].folded_bytes[byte_index];
    }

    // Compute probe positions - target last bytes of UTF-8 codepoints for maximum diversity
    sz_size_t folded_length = best[chosen_script].folded_length;
    if (folded_length == 0) {
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }

    // Find character end positions in the folded slice
    // A byte is a character's last byte if the next byte is a UTF-8 leader (not continuation)
    sz_size_t char_ends[16];
    sz_size_t char_count = 0;
    for (sz_size_t byte_index = 0; byte_index < folded_length; ++byte_index) {
        sz_u8_t next = (byte_index + 1 < folded_length) ? refined->folded_slice[byte_index + 1]
                                                        : 0xC0; // Fake leader at end
        if ((next & 0xC0) != 0x80) {                            // Next is not a continuation byte
            if (char_count < 16) char_ends[char_count++] = byte_index;
        }
    }

    // Determine probe positions
    if (char_count >= 4) {
        // 4+ characters: target last bytes of 2nd and 3rd characters
        refined->probe_second = (sz_u8_t)char_ends[1];
        refined->probe_third = (sz_u8_t)char_ends[2];
    }
    else if (folded_length <= 3) {
        // Very short: probes overlap
        refined->probe_second = (folded_length > 1) ? 1 : 0;
        refined->probe_third = (folded_length > 1) ? 1 : 0;
    }
    else {
        // 1-3 characters but 4+ bytes: use byte diversity search
        sz_u8_t byte_first = refined->folded_slice[0];
        sz_u8_t byte_last = refined->folded_slice[folded_length - 1];

        sz_size_t probe_second = folded_length / 3;
        sz_size_t probe_third = (folded_length * 2) / 3;

        // Try to find positions with bytes distinct from first/last
        for (sz_size_t byte_index = 1; byte_index < folded_length - 1; ++byte_index) {
            if (refined->folded_slice[byte_index] != byte_first && refined->folded_slice[byte_index] != byte_last) {
                probe_second = byte_index;
                break;
            }
        }

        sz_u8_t byte_second = refined->folded_slice[probe_second];
        for (sz_size_t byte_index = probe_second + 1; byte_index < folded_length - 1; ++byte_index) {
            if (refined->folded_slice[byte_index] != byte_first && refined->folded_slice[byte_index] != byte_last &&
                refined->folded_slice[byte_index] != byte_second) {
                probe_third = byte_index;
                break;
            }
        }

        // Clamp bounds
        if (probe_second == 0) probe_second = 1;
        if (probe_third >= folded_length - 1) probe_third = folded_length - 2;
        if (probe_third <= probe_second && probe_second + 1 < folded_length - 1) probe_third = probe_second + 1;

        refined->probe_second = (sz_u8_t)probe_second;
        refined->probe_third = (sz_u8_t)probe_third;
    }
}

#pragma endregion // Character Safety Profiles

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_SERIAL_H_
