/**
 *  @brief Serial backend for the single-pass Unicode normalizer (NFD / NFC / NFKD / NFKC).
 *  @file include/stringzilla/utf8_norm/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_norm.h
 *
 *  Implements the three UAX #15 primitives over UTF-8:
 *
 *  - canonical / compatibility @b decomposition (table-driven, plus algorithmic Hangul),
 *  - canonical @b ordering (stable insertion sort of each non-starter run by combining class),
 *  - canonical @b composition (Hangul algorithm + the partner-indexed primary-composite table).
 *
 *  Work is streamed one combining segment at a time - a starter followed by its trailing
 *  non-starters - so no whole-string buffer is needed. Real-world non-starter runs are 1-2 long
 *  (Stream-Safe Text caps them at 30); the buffer falls back to a flush if a pathological run
 *  exceeds @b SZ_UTF8_NORM_SEG_CAP_.
 *
 *  The engine reads the unified `utf8_norm/tables.h` record set and exposes two public entry points -
 *  a normalizer and a violation finder - that share a single scan primitive
 *  (`sz_utf8_norm_classify_serial_`), the one point a NEON (or other ISA) backend overrides. The
 *  composition step is partner-indexed, so the compose key is a dense small-range search instead of a
 *  42-bit `(starter<<21)|combiner` probe.
 */
#ifndef STRINGZILLA_UTF8_NORM_SERIAL_H_
#define STRINGZILLA_UTF8_NORM_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_norm/tables.h"
#include "stringzilla/utf8_runes.h" // `sz_rune_parse`, `sz_rune_export`

#ifdef __cplusplus
extern "C" {
#endif

#define SZ_UTF8_NORM_SEG_CAP_ 256u   /**< Max runes buffered per combining segment. */
#define SZ_UTF8_NORM_DECOMP_MAX_ 18u /**< Longest single-codepoint decomposition in the UCD. */
#define SZ_UTF8_NORM_HANGUL_L_BASE_ 0x1100u
#define SZ_UTF8_NORM_HANGUL_V_BASE_ 0x1161u
#define SZ_UTF8_NORM_HANGUL_T_BASE_ 0x11A7u
#define SZ_UTF8_NORM_HANGUL_L_COUNT_ 19u
#define SZ_UTF8_NORM_HANGUL_V_COUNT_ 21u
#define SZ_UTF8_NORM_HANGUL_T_COUNT_ 28u
#define SZ_UTF8_NORM_HANGUL_N_COUNT_ 588u // V_COUNT * T_COUNT

/** @brief 3-stage trie index for @p codepoint (0 for out-of-range / default). Shared by the props and scan lookups. */
SZ_INTERNAL sz_u16_t sz_utf8_norm_index_(sz_rune_t codepoint) {
    if (codepoint >= SZ_UTF8_NORM_TABLE_MAX_) return 0;
    sz_size_t leaf = codepoint >> SZ_UTF8_NORM_LOW_BITS_;
    sz_u16_t mid = sz_utf8_norm_stage1_[leaf >> SZ_UTF8_NORM_MID_BITS_];
    sz_u16_t block = sz_utf8_norm_stage2_[(sz_size_t)mid * SZ_UTF8_NORM_MID_ + (leaf & SZ_UTF8_NORM_MID_MASK_)];
    return sz_utf8_norm_stage3_[(sz_size_t)block * SZ_UTF8_NORM_LOW_ + (codepoint & SZ_UTF8_NORM_LOW_MASK_)];
}

/** @brief Look up the per-codepoint normalization properties (canonical combining class, quick-check, decomposition, compose). */
SZ_INTERNAL sz_utf8_norm_props_t sz_utf8_norm_lookup_(sz_rune_t codepoint) {
    return sz_utf8_norm_props_[sz_utf8_norm_index_(codepoint)];
}

/**
 *  @brief Cold-verify value `(quick_check_flags << 8) | canonical_combining_class` for @p codepoint, via the compact scan trie.
 *
 *  A dedicated palette trie (u8 stage3 + a small u16 palette) keeps this cache-tight: the props trie's
 *  stage3 is u16 (it indexes 4k+ records), and walking it for a full-CJK scan doubled the working set
 *  (~20% slower). Hangul's decomposition bits are baked into the generated values, so no runtime Hangul
 *  test is needed here.
 */
SZ_INTERNAL sz_u16_t sz_utf8_norm_value_(sz_rune_t codepoint) {
    if (codepoint >= SZ_UTF8_NORM_TABLE_MAX_) return 0;
    sz_size_t leaf = codepoint >> SZ_UTF8_NORM_SCAN_LOW_BITS_;
    sz_u16_t mid = sz_utf8_norm_scan_stage1_[leaf >> SZ_UTF8_NORM_SCAN_MID_BITS_];
    sz_u16_t block =
        sz_utf8_norm_scan_stage2_[(sz_size_t)mid * SZ_UTF8_NORM_SCAN_MID_ + (leaf & SZ_UTF8_NORM_SCAN_MID_MASK_)];
    sz_u8_t palette_index = sz_utf8_norm_scan_stage3_[(sz_size_t)block * SZ_UTF8_NORM_SCAN_LOW_ +
                                                      (codepoint & SZ_UTF8_NORM_SCAN_LOW_MASK_)];
    return sz_utf8_norm_scan_palette_[palette_index];
}

/** @brief Canonical_Combining_Class of a codepoint (0 for starters and all Hangul jamo). */
SZ_INTERNAL sz_u8_t sz_utf8_norm_ccc_(sz_rune_t codepoint) {
    return sz_utf8_norm_lookup_(codepoint).canonical_combining_class;
}

/**
 *  @brief Decompose one codepoint into 1-18 runes and their combining classes in a single lookup.
 *
 *  Fusing decomposition with the combining-class read avoids a second trie probe per codepoint on the
 *  common (non-decomposing) path - which is every base letter and every Hangul jamo.
 *  @return Number of runes written to @p out / @p out_canonical_combining_class (>= 1).
 */
SZ_INTERNAL sz_size_t sz_utf8_norm_decompose_rune_(sz_rune_t codepoint, sz_bool_t compat, sz_rune_t *out,
                                                   sz_u8_t *out_canonical_combining_class) {
    // Hangul syllables decompose algorithmically - they are absent from the tables; jamo are starters.
    if (codepoint >= SZ_UTF8_NORM_HANGUL_S_BASE_ &&
        codepoint < SZ_UTF8_NORM_HANGUL_S_BASE_ + SZ_UTF8_NORM_HANGUL_S_COUNT_) {
        sz_u32_t syllable = codepoint - SZ_UTF8_NORM_HANGUL_S_BASE_;
        out[0] = SZ_UTF8_NORM_HANGUL_L_BASE_ + syllable / SZ_UTF8_NORM_HANGUL_N_COUNT_,
        out_canonical_combining_class[0] = 0;
        out[1] = SZ_UTF8_NORM_HANGUL_V_BASE_ + (syllable % SZ_UTF8_NORM_HANGUL_N_COUNT_) / SZ_UTF8_NORM_HANGUL_T_COUNT_,
        out_canonical_combining_class[1] = 0;
        sz_u32_t trailing = syllable % SZ_UTF8_NORM_HANGUL_T_COUNT_;
        if (trailing) {
            out[2] = SZ_UTF8_NORM_HANGUL_T_BASE_ + trailing, out_canonical_combining_class[2] = 0;
            return 3;
        }
        return 2;
    }
    sz_utf8_norm_props_t properties = sz_utf8_norm_lookup_(codepoint);
    sz_u16_t index = compat ? properties.nfkd : properties.nfd;
    if (index == 0) { // no decomposition: emit self with its own combining class (no second lookup)
        out[0] = codepoint, out_canonical_combining_class[0] = properties.canonical_combining_class;
        return 1;
    }
    sz_utf8_norm_decomp_t decomposition = sz_utf8_norm_decomp_[index];
    for (sz_size_t i = 0; i != decomposition.length; ++i) {
        sz_u16_t value = sz_utf8_norm_pool_[decomposition.offset + i];
        sz_rune_t rune = value < SZ_UTF8_NORM_POOL_ASTRAL_
                             ? (sz_rune_t)value
                             : sz_utf8_norm_pool_astral_[value - SZ_UTF8_NORM_POOL_ASTRAL_];
        out[i] = rune,
        out_canonical_combining_class[i] = sz_utf8_norm_ccc_(
            rune); // decomposed runes are atomic; their class is a real lookup
    }
    return decomposition.length;
}

/**
 *  @brief Compose a starter @p a with a following codepoint @p b into a primary composite.
 *  @return The composed codepoint, or 0 if the pair does not compose.
 */
SZ_INTERNAL sz_rune_t sz_utf8_norm_compose_pair_(sz_rune_t a, sz_rune_t b) {
    // Hangul: leading + vowel jamo → LV syllable.
    if (a >= SZ_UTF8_NORM_HANGUL_L_BASE_ && a < SZ_UTF8_NORM_HANGUL_L_BASE_ + SZ_UTF8_NORM_HANGUL_L_COUNT_ && //
        b >= SZ_UTF8_NORM_HANGUL_V_BASE_ && b < SZ_UTF8_NORM_HANGUL_V_BASE_ + SZ_UTF8_NORM_HANGUL_V_COUNT_) {
        sz_u32_t leading = a - SZ_UTF8_NORM_HANGUL_L_BASE_, vowel = b - SZ_UTF8_NORM_HANGUL_V_BASE_;
        return SZ_UTF8_NORM_HANGUL_S_BASE_ +
               (leading * SZ_UTF8_NORM_HANGUL_V_COUNT_ + vowel) * SZ_UTF8_NORM_HANGUL_T_COUNT_;
    }
    // Hangul: LV syllable + trailing jamo → LVT syllable.
    if (a >= SZ_UTF8_NORM_HANGUL_S_BASE_ && a < SZ_UTF8_NORM_HANGUL_S_BASE_ + SZ_UTF8_NORM_HANGUL_S_COUNT_ && //
        (a - SZ_UTF8_NORM_HANGUL_S_BASE_) % SZ_UTF8_NORM_HANGUL_T_COUNT_ == 0 &&                              //
        b > SZ_UTF8_NORM_HANGUL_T_BASE_ && b < SZ_UTF8_NORM_HANGUL_T_BASE_ + SZ_UTF8_NORM_HANGUL_T_COUNT_)
        return a + (b - SZ_UTF8_NORM_HANGUL_T_BASE_);

    // Primary-composite table (#3): the starter exposes a dense slice of partner ids; the combiner
    // carries its own dense partner id. A small range search keyed on the partner replaces the old
    // 42-bit `(a<<21)|b` probe - no runtime key pack, and the search space is per-starter tiny.
    sz_utf8_norm_props_t props_a = sz_utf8_norm_lookup_(a), props_b = sz_utf8_norm_lookup_(b);
    if (props_a.starter == 0xFFFF || props_b.partner == 0xFFFF) return 0;
    sz_utf8_norm_compose_starter_t starter_entry = sz_utf8_norm_compose_starters_[props_a.starter];
    sz_size_t lo = starter_entry.offset, hi = (sz_size_t)starter_entry.offset + starter_entry.count;
    while (lo < hi) {
        sz_size_t middle = lo + ((hi - lo) >> 1);
        if (sz_utf8_norm_compose_partner_[middle] < props_b.partner) { lo = middle + 1; }
        else { hi = middle; }
    }
    if (lo < (sz_size_t)((sz_size_t)starter_entry.offset + starter_entry.count) &&
        sz_utf8_norm_compose_partner_[lo] == props_b.partner)
        return sz_utf8_norm_compose_value_[lo];
    return 0;
}

/** @brief Stable insertion sort of a combining segment by canonical combining class (canonical ordering). */
SZ_INTERNAL void sz_utf8_norm_canonical_order_(sz_rune_t *runes, sz_u8_t *canonical_combining_classes,
                                               sz_size_t count) {
    for (sz_size_t i = 1; i < count; ++i) {
        sz_rune_t rune = runes[i];
        sz_u8_t canonical_combining_class = canonical_combining_classes[i];
        if (canonical_combining_class == 0) continue; // starters never move (only the leading one can be a starter)
        sz_size_t j = i;
        while (j > 0 && canonical_combining_classes[j - 1] > canonical_combining_class) {
            runes[j] = runes[j - 1];
            canonical_combining_classes[j] = canonical_combining_classes[j - 1];
            --j;
        }
        runes[j] = rune;
        canonical_combining_classes[j] = canonical_combining_class;
    }
}

/** @brief Output sink: either appends UTF-8 to a destination, or compares against a source. */
typedef struct sz_utf8_norm_out_t {
    sz_u8_t *dst;           /**< Destination cursor, or NULL in compare mode. */
    sz_u8_t const *cmp;     /**< Source comparison cursor (compare mode). */
    sz_u8_t const *cmp_end; /**< End of the source buffer (compare mode). */
    sz_size_t written;      /**< Bytes written so far (write mode). */
    sz_bool_t matches;      /**< Still byte-identical to the source (compare mode). */
} sz_utf8_norm_out_t;

SZ_INTERNAL void sz_utf8_norm_emit_(sz_utf8_norm_out_t *out, sz_rune_t rune) {
    sz_u8_t bytes[4];
    sz_size_t length = (sz_size_t)sz_rune_export(rune, bytes);
    if (out->dst) {
        for (sz_size_t i = 0; i != length; ++i) out->dst[i] = bytes[i];
        out->dst += length;
        out->written += length;
    }
    else if (out->matches) {
        for (sz_size_t i = 0; i != length; ++i)
            if (out->cmp == out->cmp_end || *out->cmp++ != bytes[i]) {
                out->matches = sz_false_k;
                return;
            }
    }
}

/**
 *  @brief Emit one literal byte to the sink, bypassing rune re-encoding.
 *
 *  A malformed byte is not a codepoint - it is its own 1-byte maximal subpart and must reach the
 *  output verbatim, never round-tripped through `sz_rune_export`. It is an opaque barrier: it does
 *  not decompose, compose, or participate in canonical ordering.
 */
SZ_INTERNAL void sz_utf8_norm_emit_byte_(sz_utf8_norm_out_t *out, sz_u8_t byte) {
    if (out->dst) { *out->dst++ = byte, ++out->written; }
    else if (out->matches) {
        if (out->cmp == out->cmp_end || *out->cmp++ != byte) out->matches = sz_false_k;
    }
}

/** @brief Order, optionally compose, and emit one buffered combining segment. */
SZ_INTERNAL void sz_utf8_norm_flush_(sz_rune_t *runes, sz_u8_t *canonical_combining_classes, sz_size_t count,
                                     sz_bool_t compose, sz_utf8_norm_out_t *out) {
    if (count == 0) return;
    sz_utf8_norm_canonical_order_(runes, canonical_combining_classes, count);

    if (compose && canonical_combining_classes[0] == 0) {
        // Standard canonical composition: fold each non-blocked non-starter into the active starter.
        sz_rune_t starter = runes[0];
        sz_size_t produced = 1;
        int last_canonical_combining_class = 0;
        for (sz_size_t i = 1; i < count; ++i) {
            sz_rune_t codepoint = runes[i];
            sz_u8_t canonical_combining_class = canonical_combining_classes[i];
            if ((last_canonical_combining_class < (int)canonical_combining_class ||
                 last_canonical_combining_class == 0)) {
                sz_rune_t composed = sz_utf8_norm_compose_pair_(starter, codepoint);
                if (composed) {
                    starter = composed;
                    runes[0] = composed;
                    continue; // composed away: do not append, do not advance the blocking class
                }
            }
            last_canonical_combining_class = canonical_combining_class;
            runes[produced] = codepoint;
            canonical_combining_classes[produced] = canonical_combining_class;
            ++produced;
        }
        count = produced;
    }

    for (sz_size_t i = 0; i < count; ++i) sz_utf8_norm_emit_(out, runes[i]);
}

/** @brief Core normalization engine, shared by the write and compare entry points. */
SZ_INTERNAL void sz_utf8_norm_run_(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                   sz_utf8_norm_out_t *out) {
    sz_bool_t compat = (form == sz_normal_form_nfkd_k || form == sz_normal_form_nfkc_k) ? sz_true_k : sz_false_k;
    sz_bool_t compose = (form == sz_normal_form_nfc_k || form == sz_normal_form_nfkc_k) ? sz_true_k : sz_false_k;

    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;

    sz_rune_t segment[SZ_UTF8_NORM_SEG_CAP_];
    sz_u8_t segment_canonical_combining_classes[SZ_UTF8_NORM_SEG_CAP_];
    sz_size_t segment_length = 0;

    sz_rune_t decomposed[SZ_UTF8_NORM_DECOMP_MAX_];
    sz_u8_t decomposed_canonical_combining_classes[SZ_UTF8_NORM_DECOMP_MAX_];
    while (source_ptr < source_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_parse((sz_cptr_t)source_ptr, (sz_cptr_t)source_end, &rune);
        if (rune_length == sz_utf8_invalid_k) {
            // A malformed byte is an opaque 1-byte barrier: flush the buffered segment so the valid
            // runes before it order/compose normally, emit the byte verbatim, and resync one byte.
            sz_utf8_norm_flush_(segment, segment_canonical_combining_classes, segment_length, compose, out);
            segment_length = 0;
            sz_utf8_norm_emit_byte_(out, *source_ptr);
            ++source_ptr;
            continue;
        }
        source_ptr += rune_length;

        sz_size_t parts = sz_utf8_norm_decompose_rune_(rune, compat, decomposed,
                                                       decomposed_canonical_combining_classes);
        for (sz_size_t p = 0; p != parts; ++p) {
            sz_rune_t rune_part = decomposed[p];
            sz_u8_t canonical_combining_class = decomposed_canonical_combining_classes[p];
            if (canonical_combining_class == 0) {
                // A starter ends the current segment - unless it can merge with a lone adjacent starter.
                if (compose && segment_length == 1 && segment_canonical_combining_classes[0] == 0) {
                    sz_rune_t merged = sz_utf8_norm_compose_pair_(segment[0], rune_part);
                    if (merged) {
                        segment[0] = merged;
                        continue;
                    }
                }
                sz_utf8_norm_flush_(segment, segment_canonical_combining_classes, segment_length, compose, out);
                segment[0] = rune_part, segment_canonical_combining_classes[0] = 0, segment_length = 1;
            }
            else {
                if (segment_length >= SZ_UTF8_NORM_SEG_CAP_) { // pathological overflow: flush and restart
                    sz_utf8_norm_flush_(segment, segment_canonical_combining_classes, segment_length, compose, out);
                    segment_length = 0;
                }
                segment[segment_length] = rune_part,
                segment_canonical_combining_classes[segment_length] = canonical_combining_class, ++segment_length;
            }
        }
    }
    sz_utf8_norm_flush_(segment, segment_canonical_combining_classes, segment_length, compose, out);
}

/**
 *  @brief Is @p codepoint a normalization-safe break boundary for @p form?
 *
 *  A boundary is safe to split before iff @p codepoint is a starter (canonical combining class == 0) AND its Quick_Check for the
 *  form is Yes. The Quick_Check=Yes condition is essential: a starter that is QC=Maybe (e.g. a Hangul
 *  vowel/trailing jamo, which composes backward) must NOT be a split point, or `가` + `ᆨ` would be
 *  separated mid-composition.
 */
SZ_INTERNAL sz_bool_t sz_utf8_norm_is_safe_boundary_(sz_rune_t codepoint, sz_normal_form_t form) {
    sz_bool_t hangul = (codepoint >= SZ_UTF8_NORM_HANGUL_S_BASE_ &&
                        codepoint < SZ_UTF8_NORM_HANGUL_S_BASE_ + SZ_UTF8_NORM_HANGUL_S_COUNT_)
                           ? sz_true_k
                           : sz_false_k;
    sz_utf8_norm_props_t properties = sz_utf8_norm_lookup_(codepoint);
    if (properties.canonical_combining_class != 0) return sz_false_k;
    switch (form) {
    case sz_normal_form_nfc_k: return (properties.quick_check & 3) ? sz_false_k : sz_true_k;   // NFC_QC == Yes
    case sz_normal_form_nfkc_k: return (properties.quick_check & 12) ? sz_false_k : sz_true_k; // NFKC_QC == Yes
    case sz_normal_form_nfd_k: return (properties.nfd || hangul) ? sz_false_k : sz_true_k;     // no canonical decomp
    default: return (properties.nfkd || hangul) ? sz_false_k : sz_true_k;                      // no compat decomp
    }
}

/**
 *  @brief Scan primitive shared by both public entry points - the single point a NEON backend overrides.
 *
 *  Returns the first byte that begins a codepoint that is NOT provably inert for @p form (QC != Yes,
 *  or canonical combining class != 0, or has a relevant decomposition for the D-forms), or @b SZ_NULL_CHAR if the whole span
 *  is inert. This is the scalar reference; the NEON backend will replace just this with a `vqtbl4q`
 *  lead-classify plus a 64-byte gate. Semantics match the old module's `sz_utf8_find_denormalized`,
 *  but computed from the unified props trie - no dependency on `utf8_iterate`.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_classify_serial_(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) {
    sz_u8_t const *ptr = (sz_u8_t const *)text;
    sz_u8_t const *end = ptr + length;
    while (ptr < end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_parse((sz_cptr_t)ptr, (sz_cptr_t)end, &rune);
        if (rune_length == sz_utf8_invalid_k) { ++ptr; continue; } // malformed byte: inert, passed through

        sz_bool_t hangul = (rune >= SZ_UTF8_NORM_HANGUL_S_BASE_ &&
                            rune < SZ_UTF8_NORM_HANGUL_S_BASE_ + SZ_UTF8_NORM_HANGUL_S_COUNT_)
                               ? sz_true_k
                               : sz_false_k;
        sz_utf8_norm_props_t properties = sz_utf8_norm_lookup_(rune);
        switch (form) {
        case sz_normal_form_nfc_k:
            if ((properties.quick_check & 3) || properties.canonical_combining_class != 0) return (sz_cptr_t)ptr;
            break;
        case sz_normal_form_nfkc_k:
            if ((properties.quick_check & 12) || properties.canonical_combining_class != 0) return (sz_cptr_t)ptr;
            break;
        case sz_normal_form_nfd_k:
            if (properties.nfd || hangul || properties.canonical_combining_class != 0) return (sz_cptr_t)ptr;
            break;
        default: // sz_normal_form_nfkd_k
            if (properties.nfkd || hangul || properties.canonical_combining_class != 0) return (sz_cptr_t)ptr;
            break;
        }
        ptr += rune_length;
    }
    return SZ_NULL_CHAR;
}

/** @brief Map a normalization form to its hot-path `SZ_UTF8_NORM_QUICK_CHECK_*` flag bit. */
SZ_INTERNAL sz_u8_t sz_utf8_norm_form_flag_(sz_normal_form_t form) {
    switch (form) {
    case sz_normal_form_nfc_k: return SZ_UTF8_NORM_QUICK_CHECK_NFC_;
    case sz_normal_form_nfkc_k: return SZ_UTF8_NORM_QUICK_CHECK_NFKC_;
    case sz_normal_form_nfd_k: return SZ_UTF8_NORM_QUICK_CHECK_NFD_;
    default: return SZ_UTF8_NORM_QUICK_CHECK_NFKD_;
    }
}

/**
 *  @brief Cold per-codepoint verify shared by every vector scanner: walk `[*position_io, block_end)`
 *  and return the first byte that begins a non-inert codepoint for @p form_flag (a canonical-ordering
 *  violation or a quick-check No/Maybe), else @b SZ_NULL_CHAR. Updates `*position_io` to where it
 *  stopped and carries `*previous_canonical_combining_class_io` across SIMD-block boundaries.
 *
 *  ASCII resets the combining class; 2-byte runes use the flat `sz_utf8_norm_twobyte_` table (one load,
 *  no general parse); 3-/4-byte runes parse and read `sz_utf8_norm_value_`. This is the exact body the
 *  NEON backend ran inline, lifted here so all backends share one copy of the carry-sensitive logic.
 *  A malformed byte is an opaque 1-byte barrier: it is inert (never flagged, passed through unchanged)
 *  and resets the carried combining class, exactly like ASCII.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_verify_block_(sz_u8_t const **position_io, sz_u8_t const *block_end,
                                                 sz_u8_t const *end, sz_u8_t form_flag,
                                                 sz_u8_t *previous_canonical_combining_class_io) {
    sz_u8_t const *position = *position_io;
    sz_u8_t previous_canonical_combining_class = *previous_canonical_combining_class_io;
    while (position < block_end) {
        sz_u8_t const *codepoint_start = position;
        sz_u8_t lead = *position;
        sz_u16_t value;
        if (lead < 0x80) {
            previous_canonical_combining_class = 0, ++position;
            continue;
        }
        else if (lead >= 0xE0) { // 3-/4-byte: validating parse + props trie
            sz_rune_t rune;
            sz_rune_length_t rune_length = sz_rune_parse((sz_cptr_t)position, (sz_cptr_t)end, &rune);
            if (rune_length == sz_utf8_invalid_k) { previous_canonical_combining_class = 0, ++position; continue; }
            value = sz_utf8_norm_value_(rune);
            position += rune_length;
        }
        else if (position + 2 <= end && (position[1] & 0xC0u) == 0x80u && lead >= 0xC2u) {
            // 2-byte well-formed: inlined decode + flat lookup (one load, no parse)
            value = sz_utf8_norm_twobyte_[((sz_rune_t)(lead & 0x1Fu) << 6) | (position[1] & 0x3Fu)];
            position += 2;
        }
        else { // malformed lead/continuation or truncated 2-byte: a 1-byte barrier, passed through
            previous_canonical_combining_class = 0, ++position;
            continue;
        }
        sz_u8_t canonical_combining_class = (sz_u8_t)(value & 0xFFu);
        if ((canonical_combining_class != 0 && canonical_combining_class < previous_canonical_combining_class) ||
            ((sz_u8_t)(value >> 8) & form_flag)) {
            *position_io = position;
            *previous_canonical_combining_class_io = previous_canonical_combining_class;
            return (sz_cptr_t)codepoint_start;
        }
        previous_canonical_combining_class = canonical_combining_class;
    }
    *position_io = position;
    *previous_canonical_combining_class_io = previous_canonical_combining_class;
    return SZ_NULL_CHAR;
}

/**
 *  @brief A scan primitive: returns the first non-inert byte for @p form, or @b SZ_NULL_CHAR if the
 *  span is provably already normalized. The single ISA-specific point both engines force-inline.
 *
 *  `sz_utf8_norm_classify_serial_` is the scalar reference; `sz_utf8_norm_classify_neon_` (in `neon.h`) is the
 *  vectorized override. Passing a constant function address into the `SZ_INTERNAL` (always-inline)
 *  engines below devirtualizes the call at -O2/-O3, so each backend pays no indirection - this is the
 *  same force-inlined function-pointer idiom the case-folding family uses.
 */
typedef sz_cptr_t (*sz_utf8_norm_scan_t)(sz_cptr_t, sz_size_t, sz_normal_form_t);

/**
 *  @brief Does the codepoint (or malformed byte) at @p position open a safe split boundary for @p form?
 *
 *  A malformed byte is an opaque 1-byte barrier - it never decomposes/composes/reorders - so it is
 *  always a safe boundary. A well-formed rune defers to `sz_utf8_norm_is_safe_boundary_`.
 */
SZ_INTERNAL sz_bool_t sz_utf8_norm_boundary_at_(sz_u8_t const *position, sz_u8_t const *end, sz_normal_form_t form) {
    sz_rune_t rune;
    sz_rune_length_t const rune_length = sz_rune_parse((sz_cptr_t)position, (sz_cptr_t)end, &rune);
    if (rune_length == sz_utf8_invalid_k) return sz_true_k; // malformed byte: opaque barrier
    return sz_utf8_norm_is_safe_boundary_(rune, form);
}

/**
 *  @brief Step @p position back to the start of the preceding codepoint, never crossing @p begin.
 *
 *  A backward UTF-8 scan that is malformed-safe: a continuation run that does not resolve to a
 *  well-formed lead (or that would cross @p begin) is treated as single literal bytes, so the cursor
 *  retreats exactly one byte rather than over-reading.
 */
SZ_INTERNAL sz_u8_t const *sz_utf8_norm_step_back_(sz_u8_t const *position, sz_u8_t const *begin) {
    sz_u8_t const *probe = position - 1;
    while (probe > begin && (*probe & 0xC0u) == 0x80u && (position - probe) < 4) --probe;
    sz_rune_t rune;
    sz_rune_length_t const rune_length = sz_rune_parse((sz_cptr_t)probe, (sz_cptr_t)position, &rune);
    // A clean rune ending exactly at `position` is the real predecessor; otherwise retreat one byte.
    if (rune_length != sz_utf8_invalid_k && probe + rune_length == position) return probe;
    return position - 1;
}

/**
 *  @brief Normalize via skip-and-fix: copy the already-normalized runs verbatim (located by the
 *  @p scan primitive) and run the decompose/reorder/compose engine only on the short dirty regions,
 *  each delimited by safe boundaries so composition never crosses a split. Shared across ISAs.
 */
SZ_INTERNAL sz_size_t sz_utf8_norm_engine_(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                           sz_ptr_t destination, sz_utf8_norm_scan_t scan) {
    sz_u8_t const *const begin = (sz_u8_t const *)source;
    sz_u8_t const *const end = begin + source_length;
    sz_u8_t *out = (sz_u8_t *)destination;
    sz_u8_t const *ptr = begin;

    while (ptr < end) {
        sz_cptr_t dirty = scan((sz_cptr_t)ptr, (sz_size_t)(end - ptr), form);
        if (dirty == SZ_NULL_CHAR) { // rest is provably normalized - copy verbatim
            while (ptr < end) *out++ = *ptr++;
            break;
        }
        sz_u8_t const *dirty_ptr = (sz_u8_t const *)dirty;

        // Back up to the safe boundary that begins the affected segment (a Yes-starter, or a malformed
        // barrier, at/before dirty_ptr).
        sz_u8_t const *segment = dirty_ptr;
        while (segment > ptr) {
            if (sz_utf8_norm_boundary_at_(segment, end, form)) break;
            segment = sz_utf8_norm_step_back_(segment, ptr);
        }

        // Find the next safe boundary strictly after dirty_ptr - the end of the dirty region.
        sz_u8_t const *tail = dirty_ptr;
        {
            sz_rune_t rune;
            sz_rune_length_t rune_length = sz_rune_parse((sz_cptr_t)tail, (sz_cptr_t)end, &rune);
            tail += rune_length == sz_utf8_invalid_k ? 1 : rune_length;
        }
        while (tail < end) {
            if (sz_utf8_norm_boundary_at_(tail, end, form)) break;
            sz_rune_t rune;
            sz_rune_length_t rune_length = sz_rune_parse((sz_cptr_t)tail, (sz_cptr_t)end, &rune);
            tail += rune_length == sz_utf8_invalid_k ? 1 : rune_length;
        }

        // Emit the clean prefix verbatim, then normalize the bounded dirty region.
        while (ptr < segment) *out++ = *ptr++;
        sz_utf8_norm_out_t sink;
        sink.dst = out;
        sink.cmp = SZ_NULL, sink.cmp_end = SZ_NULL;
        sink.written = 0, sink.matches = sz_true_k;
        sz_utf8_norm_run_((sz_cptr_t)segment, (sz_size_t)(tail - segment), form, &sink);
        out += sink.written;
        ptr = tail;
    }
    return (sz_size_t)(out - (sz_u8_t *)destination);
}

/**
 *  @brief Find the first byte that proves @p source is not in @p form, or @b SZ_NULL_CHAR if it is.
 *  Shared across ISAs; the dirty runs are located by the @p scan primitive.
 *
 *  The @p scan scanner is a conservative superset (it flags every `canonical combining class != 0` byte so the normalizer
 *  can blindly re-fix), so a stop is not by itself a violation - a well-ordered bare combining mark is
 *  valid NFD. We therefore verify exactly: a combining segment (delimited by safe boundaries, so the
 *  check is compositional) is in @p form @b iff normalizing it reproduces it byte-for-byte. If it
 *  changes, the segment is the violation; otherwise we skip it and keep scanning. This single rule is
 *  exact for all four forms, subsuming the No/Maybe/order cases.
 *
 *  We return the @b safe-boundary start of the first non-conforming segment, not the @p scan stop:
 *  this is backend-independent (a conservative serial scan and an exact NEON scan walk past the same
 *  benign segments and back up to the same boundary), and it carries the clean guarantee that every
 *  byte before the returned pointer is provably in @p form.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_norm_violation_engine_(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                                     sz_utf8_norm_scan_t scan) {
    sz_u8_t const *const end = (sz_u8_t const *)source + source_length;
    sz_u8_t const *cur = (sz_u8_t const *)source;

    while (cur < end) {
        // Fast path: the scanner spans the already-normalized run (NEON: 26-94 GB/s).
        sz_cptr_t stop = scan((sz_cptr_t)cur, (sz_size_t)(end - cur), form);
        if (stop == SZ_NULL_CHAR) return SZ_NULL_CHAR;

        // Resolve the flagged byte exactly by normalizing its combining segment and comparing.
        sz_u8_t const *stop_ptr = (sz_u8_t const *)stop;
        sz_u8_t const *segment = stop_ptr;
        while (segment > cur) {
            if (sz_utf8_norm_boundary_at_(segment, end, form)) break;
            segment = sz_utf8_norm_step_back_(segment, cur);
        }
        // The next safe boundary strictly after the flagged codepoint ends the segment.
        sz_u8_t const *tail = stop_ptr;
        {
            sz_rune_t rune;
            sz_rune_length_t rune_length = sz_rune_parse((sz_cptr_t)tail, (sz_cptr_t)end, &rune);
            tail += rune_length == sz_utf8_invalid_k ? 1 : rune_length;
        }
        while (tail < end) {
            if (sz_utf8_norm_boundary_at_(tail, end, form)) break;
            sz_rune_t rune;
            sz_rune_length_t rune_length = sz_rune_parse((sz_cptr_t)tail, (sz_cptr_t)end, &rune);
            tail += rune_length == sz_utf8_invalid_k ? 1 : rune_length;
        }
        sz_utf8_norm_out_t out;
        out.dst = SZ_NULL;
        out.cmp = segment, out.cmp_end = tail;
        out.written = 0, out.matches = sz_true_k;
        sz_utf8_norm_run_((sz_cptr_t)segment, (sz_size_t)(tail - segment), form, &out);
        if (!out.matches || out.cmp != tail) return (sz_cptr_t)segment; // segment changes => not normalized
        cur = tail;                                                     // already-normalized here; keep scanning
    }
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_size_t sz_utf8_norm_serial(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                        sz_ptr_t destination) {
    return sz_utf8_norm_engine_(source, source_length, form, destination, &sz_utf8_norm_classify_serial_);
}

SZ_PUBLIC sz_cptr_t sz_utf8_norm_violation_serial(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form) {
    return sz_utf8_norm_violation_engine_(source, source_length, form, &sz_utf8_norm_classify_serial_);
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_NORM_SERIAL_H_
