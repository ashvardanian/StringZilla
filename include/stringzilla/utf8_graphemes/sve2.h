/**
 *  @brief SVE2 (AArch64 scalable) backend for UAX-29 grapheme clusters.
 *  @file include/stringzilla/utf8_graphemes/sve2.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_SVE2_H_
#define STRINGZILLA_UTF8_GRAPHEMES_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_graphemes/tables.h"
#include "stringzilla/utf8_graphemes/serial.h"
#include "stringzilla/utf8_runes/sve2.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

#pragma region Grapheme Cluster Break classifier

/** @brief  Packed descriptor byte for one chunk of ASTRAL codepoints over offset = cp - 0x10000 (5-nibble cascade),
 *          the SVE2 twin of @ref sz_grapheme_astral_descriptor_neon_. Per-lane bytes: @p plane = (offset>>16)&0xFF
 *          (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF. Bit-exact. */
SZ_HELPER_AUTO svuint8_t sz_grapheme_astral_descriptor_sve2_(svuint8_t plane_u8x, svuint8_t high_u8x,
                                                             svuint8_t low_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const n4_u8x = svand_n_u8_x(all_b8x, plane_u8x, 0x0F);
    svuint8_t const n3_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, high_u8x, 4), 0x0F);
    svuint8_t const stage1_index_u8x = svorr_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, n4_u8x, 4), n3_u8x);
    svuint8_t const page_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_grapheme_break_haswell_astral_stage1_, 256,
                                                      stage1_index_u8x);
    svuint8_t const n2_u8x = svand_n_u8_x(all_b8x, high_u8x, 0x0F);
    svuint8_t const leaf2_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_grapheme_break_haswell_astral_stage2_lo_,
        (int)sz_utf8_grapheme_break_haswell_astral_stage2_lo_count_k / 16, page_u8x, n2_u8x);
    svuint8_t const n1_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, low_u8x, 4), 0x0F);
    svuint8_t const leaf_lo_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_grapheme_break_haswell_astral_stage3_lo_,
        (int)sz_utf8_grapheme_break_haswell_astral_stage3_lo_count_k / 16, leaf2_u8x, n1_u8x);
    svuint8_t const leaf_hi_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_grapheme_break_haswell_astral_stage3_hi_,
        (int)sz_utf8_grapheme_break_haswell_astral_stage3_hi_count_k / 16, leaf2_u8x, n1_u8x);
    svuint8_t const n0_u8x = svand_n_u8_x(all_b8x, low_u8x, 0x0F);
    svuint8_t const leaf_group_u8x = svorr_u8_x(all_b8x,
                                                svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, leaf_lo_u8x, 4), 0x0F),
                                                svlsl_n_u8_x(all_b8x, leaf_hi_u8x, 4));
    svuint8_t const stage4_index_u8x = svorr_u8_x(
        all_b8x, svlsl_n_u8_x(all_b8x, svand_n_u8_x(all_b8x, leaf_lo_u8x, 0x0F), 4), n0_u8x);
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_grapheme_break_haswell_astral_leaf_groups_k; ++group) {
        svuint8_t const value_u8x = sz_utf8_rune_lut_sve2_(
            sz_utf8_grapheme_break_haswell_astral_stage4_groups_ + group * 256, 256, stage4_index_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(all_b8x, leaf_group_u8x, (sz_u8_t)group), value_u8x, result_u8x);
    }
    return result_u8x;
}

/** @brief  Predicate of lanes whose BMP codepoint `(high << 8) | low` lies in the inclusive range [@p lo, @p hi] -
 *          the SVE2 twin of @ref sz_grapheme_cp_in_range_neon_ confined to one chunk. */
SZ_HELPER_INLINE svbool_t sz_grapheme_cp_in_range_sve2_(svuint8_t high_u8x, svuint8_t low_u8x, sz_u16_t lo,
                                                        sz_u16_t hi) {
    svbool_t const all_b8x = svptrue_b8();
    sz_u8_t const lo_high = (sz_u8_t)(lo >> 8), lo_low = (sz_u8_t)(lo & 0xFF);
    sz_u8_t const hi_high = (sz_u8_t)(hi >> 8), hi_low = (sz_u8_t)(hi & 0xFF);
    svbool_t const high_eq_lo_b8x = svcmpeq_n_u8(all_b8x, high_u8x, lo_high);
    svbool_t const high_eq_hi_b8x = svcmpeq_n_u8(all_b8x, high_u8x, hi_high);
    svbool_t const ge_low_b8x = svorr_b_z(all_b8x,
                                          svbic_b_z(all_b8x, svcmpge_n_u8(all_b8x, high_u8x, lo_high), high_eq_lo_b8x),
                                          svand_b_z(all_b8x, high_eq_lo_b8x, svcmpge_n_u8(all_b8x, low_u8x, lo_low)));
    svbool_t const le_high_b8x = svorr_b_z(all_b8x,
                                           svbic_b_z(all_b8x, svcmple_n_u8(all_b8x, high_u8x, hi_high), high_eq_hi_b8x),
                                           svand_b_z(all_b8x, high_eq_hi_b8x, svcmple_n_u8(all_b8x, low_u8x, hi_low)));
    return svand_b_z(all_b8x, ge_low_b8x, le_high_b8x);
}

/** @brief  Lanes whose BMP codepoint resolves uniformly to GCB=Other via the CJK / Kana arithmetic ranges - the
 *          SVE2 twin of @ref sz_grapheme_cjk_other_neon_. Such lanes need no cold cascade (descriptor 0). */
SZ_HELPER_AUTO svbool_t sz_grapheme_cjk_other_sve2_(svuint8_t high_u8x, svuint8_t low_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    svbool_t const run_a_b8x = sz_grapheme_cp_in_range_sve2_(high_u8x, low_u8x, 0x3000, 0xA66E);
    svbool_t const run_b_b8x = sz_grapheme_cp_in_range_sve2_(high_u8x, low_u8x, 0xD7FC, 0xFB1D);
    svbool_t const exc_b8x = svorr_b_z(
        all_b8x,
        svorr_b_z(all_b8x, sz_grapheme_cp_in_range_sve2_(high_u8x, low_u8x, 0x302A, 0x3030),
                  sz_grapheme_cp_in_range_sve2_(high_u8x, low_u8x, 0x3099, 0x309A)),
        svorr_b_z(all_b8x,
                  svorr_b_z(all_b8x, sz_grapheme_cp_in_range_sve2_(high_u8x, low_u8x, 0x303D, 0x303D),
                            sz_grapheme_cp_in_range_sve2_(high_u8x, low_u8x, 0x3297, 0x3297)),
                  sz_grapheme_cp_in_range_sve2_(high_u8x, low_u8x, 0x3299, 0x3299)));
    return svbic_b_z(all_b8x, svorr_b_z(all_b8x, run_a_b8x, run_b_b8x), exc_b8x);
}

#pragma endregion Grapheme Cluster Break classifier

#pragma region Grapheme forward driver

/**
 *  @brief  Forward UAX-29 grapheme-cluster kernel (SVE2, vector-length agnostic). Bit-exact with
 *          `sz_utf8_graphemes_serial` and the other ISA fronts: a chunked-window classify feeds the shared
 *          portable engine @ref sz_grapheme_window_boundaries_ through the shared bit-route compaction.
 *
 *  Each window streams as `64 / svcntb()` register chunks with one peeked vector ahead; every byte lane
 *  classifies BLINDLY (matching the family's malformed-input policy) to the packed descriptor
 *  `gcb | incb << 4 | extpict << 6`, whose seven bits lower to window-wide bit-planes through the predicate
 *  bridge. The planes compact to the codepoint-dense domain with ONE shared bit-route built from the start
 *  lanes - seven gathers instead of eighteen - and the per-class masks assemble from the dense planes with
 *  scalar algebra. Clusters are DENSE (a boundary every 1-4 codepoints) and callers stream small capacities,
 *  so the window clamps to the remaining-capacity budget whenever the clamped edge is structurally clean
 *  (every continuation byte inside is claimed by a lead's declared length), retrying unclamped otherwise -
 *  classify work stays proportional to what the caller can consume.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_graphemes_sve2(          //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed) {

    sz_size_t clusters = 0;
    if (length == 0 || clusters_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const vector_bytes = svcntb();
    sz_size_t const chunk_bytes = vector_bytes < 64 ? vector_bytes : 64;
    sz_size_t const window_capacity = (64 / chunk_bytes) * chunk_bytes;
    svbool_t const all_b8x = svptrue_b8();

    sz_grapheme_carry_t carry = sz_grapheme_carry_empty_();
    sz_size_t cluster_start = 0;
    sz_size_t base = 0;
    sz_size_t bytes_per_cluster = 2; // running density estimate, updated from each window's own yield

    while (base < length) {
        sz_size_t const available = length - base;
        // Capacity budget: the remaining output slots bound how many clusters this window can emit, so scale the
        // window to the observed cluster density instead of classifying bytes the caller cannot consume - the
        // dominant cost at the small capacities the bindings stream. The clamp is only taken when the clamped
        // window is structurally clean (checked below).
        sz_size_t const budget = (clusters_capacity - clusters) * bytes_per_cluster + 4;
        sz_size_t window_target = available < window_capacity ? available : window_capacity;
        int clamped = budget < window_target;
        if (clamped) window_target = budget;

        for (int attempt = 0; attempt < 2; ++attempt) {
            sz_size_t const loaded = window_target;
            sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);

            // Stream the window: per chunk, lower the lead-shape masks and classify every lane to a descriptor.
            sz_u64_t continuation = 0, two_byte = 0, three_byte = 0, four_byte = 0;
            sz_u64_t desc_planes[7] = {0, 0, 0, 0, 0, 0, 0};
            for (sz_size_t chunk_base = 0; chunk_base < loaded; chunk_base += chunk_bytes) {
                sz_size_t const ahead = loaded - chunk_base;
                sz_size_t const chunk_span = ahead < chunk_bytes ? ahead : chunk_bytes;
                sz_size_t const peek_span =
                    ahead > chunk_bytes ? (ahead - chunk_bytes < chunk_bytes ? ahead - chunk_bytes : chunk_bytes) : 0;
                svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
                svbool_t const peek_b8x = svwhilelt_b8_u64(0, (sz_u64_t)peek_span);
                svuint8_t const bytes_u8x = svld1_u8(loaded_b8x, text_u8 + base + chunk_base);
                svuint8_t const peek_u8x = svld1_u8(peek_b8x,
                                                    text_u8 + base + chunk_base + (peek_span ? chunk_bytes : 0));
                svuint8_t const next1_u8x = svext_u8(bytes_u8x, peek_u8x, 1);
                svuint8_t const next2_u8x = svext_u8(bytes_u8x, peek_u8x, 2);
                svuint8_t const next3_u8x = svext_u8(bytes_u8x, peek_u8x, 3);

                svbool_t const continuation_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xC0),
                                                               0x80);
                svbool_t const three_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF0), 0xE0);
                svbool_t const four_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF8), 0xF0);
                svbool_t const ascii_b8x = svcmplt_n_u8(loaded_b8x, bytes_u8x, 0x80);
                continuation |= sz_utf8_rune_pred_to_u64_sve2_(continuation_b8x) << chunk_base;
                two_byte |= sz_utf8_rune_pred_to_u64_sve2_(
                                svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xE0), 0xC0))
                            << chunk_base;
                three_byte |= sz_utf8_rune_pred_to_u64_sve2_(three_b8x) << chunk_base;
                four_byte |= sz_utf8_rune_pred_to_u64_sve2_(four_b8x) << chunk_base;

                // Blind codepoint reconstruction: 2-byte fold by default on non-ASCII lanes, the 3-byte fold on
                // E0..EF leads, and the astral (plane, mid, alo) triple for F0..F7 leads whose BMP view seats
                // (mid, alo) so a plane-0 overlong 4-byte lead resolves through the BMP path exactly like serial.
                svuint8_t const two_high_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, bytes_u8x, 2), 0x07);
                svuint8_t const two_low_u8x = svorr_u8_x(
                    all_b8x, svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, bytes_u8x, 6), 0xC0),
                    svand_n_u8_x(all_b8x, next1_u8x, 0x3F));
                svuint8_t const three_high_u8x = svorr_u8_x(
                    all_b8x, svlsl_n_u8_x(all_b8x, svand_n_u8_x(all_b8x, bytes_u8x, 0x0F), 4),
                    svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next1_u8x, 2), 0x0F));
                svuint8_t const three_low_u8x = svorr_u8_x(
                    all_b8x, svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next1_u8x, 6), 0xC0),
                    svand_n_u8_x(all_b8x, next2_u8x, 0x3F));
                svuint8_t const plane_u8x = svorr_u8_x(
                    all_b8x, svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, bytes_u8x, 2), 0x1C),
                    svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next1_u8x, 4), 0x03));
                svuint8_t const mid_u8x = svorr_u8_x(all_b8x,
                                                     svlsl_n_u8_x(all_b8x, svand_n_u8_x(all_b8x, next1_u8x, 0x0F), 4),
                                                     svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next2_u8x, 2), 0x0F));
                svuint8_t const alo_u8x = svorr_u8_x(all_b8x,
                                                     svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next2_u8x, 6), 0xC0),
                                                     svand_n_u8_x(all_b8x, next3_u8x, 0x3F));

                svuint8_t high_u8x = svsel_u8(three_b8x, three_high_u8x, two_high_u8x);
                svuint8_t low_u8x = svsel_u8(three_b8x, three_low_u8x, two_low_u8x);
                high_u8x = svsel_u8(ascii_b8x, svdup_n_u8(0), high_u8x);
                low_u8x = svsel_u8(ascii_b8x, bytes_u8x, low_u8x);
                high_u8x = svsel_u8(four_b8x, mid_u8x, high_u8x);
                low_u8x = svsel_u8(four_b8x, alo_u8x, low_u8x);

                // ASCII fast path: one 256-LUT on the raw byte; the cold full-BMP cascade only runs when the
                // chunk holds a non-ASCII, non-CJK-other lane (a pure-ASCII or pure-CJK chunk skips it).
                svuint8_t const ascii_desc_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_grapheme_break_haswell_ascii_desc_, 256,
                                                                        bytes_u8x);
                svuint8_t desc_u8x = ascii_desc_u8x;
                svbool_t const non_ascii_b8x = svbic_b_z(loaded_b8x, loaded_b8x, ascii_b8x);
                if (svptest_any(loaded_b8x, non_ascii_b8x)) {
                    svbool_t const cjk_other_b8x = svand_b_z(loaded_b8x, sz_grapheme_cjk_other_sve2_(high_u8x, low_u8x),
                                                             non_ascii_b8x);
                    desc_u8x = svsel_u8(cjk_other_b8x, svdup_n_u8(0), desc_u8x);
                    if (svptest_any(loaded_b8x, svbic_b_z(loaded_b8x, non_ascii_b8x, cjk_other_b8x))) {
                        svuint8_t const bmp_u8x = sz_utf8_rune_flat_lookup_sve2_(
                            sz_utf8_grapheme_break_bmp_page_lut_, sz_utf8_grapheme_break_flat_bmp_, high_u8x, low_u8x);
                        desc_u8x = svsel_u8(ascii_b8x, ascii_desc_u8x, bmp_u8x);
                    }
                }

                // A 4-byte lead splits on its blind plane: 0 already went through the BMP path, [1,16] takes the
                // astral trie, above 16 forces Other; `0xF8..0xFF` leads classify as Other regardless.
                svbool_t const plane_nonzero_b8x = svand_b_z(loaded_b8x, four_b8x, svcmpgt_n_u8(all_b8x, plane_u8x, 0));
                svbool_t const plane_le16_b8x = svcmple_n_u8(all_b8x, plane_u8x, 0x10);
                svbool_t const is_astral_b8x = svand_b_z(loaded_b8x, plane_nonzero_b8x, plane_le16_b8x);
                if (svptest_any(loaded_b8x, is_astral_b8x)) {
                    svuint8_t const plane_off_u8x = svsub_n_u8_x(all_b8x,
                                                                 svsel_u8(is_astral_b8x, plane_u8x, svdup_n_u8(1)), 1);
                    svuint8_t const astral_u8x = sz_grapheme_astral_descriptor_sve2_(plane_off_u8x, mid_u8x, alo_u8x);
                    desc_u8x = svsel_u8(is_astral_b8x, astral_u8x, desc_u8x);
                }
                svbool_t const clear_b8x = svorr_b_z(loaded_b8x,
                                                     svbic_b_z(loaded_b8x, plane_nonzero_b8x, plane_le16_b8x),
                                                     svcmpge_n_u8(loaded_b8x, bytes_u8x, 0xF8));
                desc_u8x = svsel_u8(clear_b8x, svdup_n_u8(0), desc_u8x);

                for (int bit = 0; bit < 7; ++bit)
                    desc_planes[bit] |= sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(
                                            loaded_b8x, svand_n_u8_x(loaded_b8x, desc_u8x, (sz_u8_t)(1 << bit)), 0))
                                        << chunk_base;
            }

            // The capacity clamp is only safe when the clamped window is structurally clean: every continuation
            // byte inside is claimed by a lead's declared length, so no U+FFFD grouping can depend on the
            // artificial edge. Otherwise retry this window unclamped.
            sz_u64_t const claimed = (((two_byte | three_byte | four_byte) << 1) | ((three_byte | four_byte) << 2) |
                                      (four_byte << 3)) &
                                     loaded_mask;
            if (clamped && (claimed & loaded_mask) != continuation) {
                clamped = 0;
                window_target = available < window_capacity ? available : window_capacity;
                continue;
            }

            sz_u64_t start_lanes = loaded_mask & ~continuation;
            sz_size_t byte_span = loaded;
            if (loaded >= window_capacity || clamped) {
                byte_span = sz_grapheme_byte_span_serial_(two_byte, three_byte, four_byte, loaded);
                start_lanes &= sz_u64_mask_until_serial_(byte_span);
            }

            // ONE shared bit-route compacts the seven descriptor planes to the codepoint-dense domain; the class,
            // InCB, and Extended_Pictographic masks then assemble from the dense planes with scalar algebra.
            sz_grapheme_bit_route_t const route = sz_grapheme_bit_route_build_(start_lanes);
            sz_u64_t const dense_ones = sz_grapheme_bit_gather_(start_lanes, &route);
            if (!dense_ones) {
                base = length;
                break;
            } // defensive: cannot happen for valid input
            int const count = (int)(64 - sz_u64_clz(dense_ones));
            sz_u64_t const valid = dense_ones;
            sz_u64_t dense_planes[7];
            for (int bit = 0; bit < 7; ++bit) dense_planes[bit] = sz_grapheme_bit_gather_(desc_planes[bit], &route);

            sz_grapheme_window_masks_t masks;
            for (int cls = 0; cls < 14; ++cls)
                masks.class_bit[cls] = ((cls & 1) ? dense_planes[0] : ~dense_planes[0]) &
                                       ((cls & 2) ? dense_planes[1] : ~dense_planes[1]) &
                                       ((cls & 4) ? dense_planes[2] : ~dense_planes[2]) &
                                       ((cls & 8) ? dense_planes[3] : ~dense_planes[3]) & valid;
            masks.indic_consonant = dense_planes[4] & ~dense_planes[5] & valid;
            masks.indic_extend = ~dense_planes[4] & dense_planes[5] & valid;
            masks.indic_linker = dense_planes[4] & dense_planes[5] & valid;
            masks.extended_pictographic = dense_planes[6] & valid;

            sz_u64_t const dense_boundary = sz_grapheme_window_boundaries_(&masks, count, valid, &carry);
            sz_u64_t boundary = sz_grapheme_bit_scatter_(dense_boundary, &route);
            if (base == 0) boundary &= ~1ull; // GB1: the first cluster's own start is not a new break

            sz_size_t const clusters_before = clusters;
            clusters = sz_utf8_rune_drain_forward_serial_(boundary, base, cluster_starts, cluster_lengths, clusters,
                                                          clusters_capacity, &cluster_start);
            if (clusters == clusters_capacity) {
                if (bytes_consumed) *bytes_consumed = cluster_start;
                return clusters;
            }
            if (clusters > clusters_before) {
                bytes_per_cluster = (byte_span + (clusters - clusters_before) - 1) / (clusters - clusters_before);
                if (bytes_per_cluster < 1) bytes_per_cluster = 1;
                if (bytes_per_cluster > 8) bytes_per_cluster = 8;
            }
            base += byte_span;
            break;
        }
    }

    cluster_starts[clusters] = cluster_start;
    cluster_lengths[clusters] = length - cluster_start;
    ++clusters;
    if (bytes_consumed) *bytes_consumed = length;
    return clusters;
}

#pragma endregion Grapheme forward driver

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_SVE2_H_
