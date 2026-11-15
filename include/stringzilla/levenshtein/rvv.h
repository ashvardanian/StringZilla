/**
 *  @file include/stringzilla/levenshtein/rvv.h
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief RISC-V Vector bit-vector edit distances over bytes and UTF-8 runes.
 *
 *  @sa c/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_RVV_H_
#define STRINGZILLA_LEVENSHTEIN_RVV_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_RISCV64_
#if STRINGZILLA_ARCH_RISCV64_RVV_

#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/utf8_runes/rvv.h" // `sz_utf8_decode_rvv_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

/** Candidates one sweep holds at most, whatever the vector length: the bits of the mask that
 *  tracks its unread ones. */
enum { sz_levenshtein_u64m2_candidates_max_rvv_k = sizeof(sz_u64_t) * 8 };

/** Candidates a sweep steps per position: one LMUL-2 group of 64-bit lanes, the widest whose step
 *  keeps its live vectors in registers, from four at VLEN 128 to sixteen at 512. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64m2_lanes_rvv_(void) {
    return sz_min_of_two(__riscv_vsetvlmax_e64m2(), (sz_size_t)sz_levenshtein_u64m2_candidates_max_rvv_k);
}

STRINGZILLA_INLINE vuint64m2_t sz_levenshtein_u64m2_init_rvv_(sz_u64_t *verticals, sz_size_t words, sz_size_t lanes,
                                                              sz_levenshtein_query_t const *query) {
    vuint64m2_t const ones_u64m2 = __riscv_vmv_v_x_u64m2(~(sz_u64_t)0, lanes);
    vuint64m2_t const zeros_u64m2 = __riscv_vmv_v_x_u64m2(0, lanes);
    for (sz_size_t word = 0; word != words; ++word) {
        __riscv_vse64_v_u64m2(verticals + word * 2 * lanes, ones_u64m2, lanes);
        __riscv_vse64_v_u64m2(verticals + word * 2 * lanes + lanes, zeros_u64m2, lanes);
    }
    return __riscv_vmv_v_x_u64m2(query->length, lanes);
}

STRINGZILLA_INLINE vuint64m2_t sz_levenshtein_u64m2_step_rvv_(vuint64m2_t scores_u64m2, sz_u64_t *verticals,
                                                              sz_size_t words, sz_size_t lanes,
                                                              sz_levenshtein_query_t const *query,
                                                              vuint32m1_t offsets_u32m1, sz_size_t vector_length) {
    // The top boundary: the row above the first word is one edit higher than the cell to its left.
    vuint64m2_t positive_carry_u64m2 = __riscv_vmv_v_x_u64m2(1, vector_length);
    vuint64m2_t negative_carry_u64m2 = __riscv_vmv_v_x_u64m2(0, vector_length);
    vuint64m2_t horizontal_positive_u64m2 = positive_carry_u64m2, horizontal_negative_u64m2 = negative_carry_u64m2;
    for (sz_size_t word = 0; word != words; ++word) {
        sz_u64_t *const positive = verticals + word * 2 * lanes, *const negative = positive + lanes;
        vuint64m2_t const positive_u64m2 = __riscv_vle64_v_u64m2(positive, vector_length);
        vuint64m2_t const negative_u64m2 = __riscv_vle64_v_u64m2(negative, vector_length);
        vuint64m2_t const equality_u64m2 = __riscv_vluxei32_v_u64m2(query->masks + word * query->stride, offsets_u32m1,
                                                                    vector_length);
        vuint64m2_t const vertical_carry_u64m2 = __riscv_vor_vv_u64m2(equality_u64m2, negative_u64m2, vector_length);
        vuint64m2_t const matched_u64m2 = __riscv_vor_vv_u64m2(equality_u64m2, negative_carry_u64m2, vector_length);
        vuint64m2_t const sum_u64m2 = __riscv_vadd_vv_u64m2(
            __riscv_vand_vv_u64m2(matched_u64m2, positive_u64m2, vector_length), positive_u64m2, vector_length);
        vuint64m2_t const diagonal_u64m2 = __riscv_vor_vv_u64m2(
            __riscv_vxor_vv_u64m2(sum_u64m2, positive_u64m2, vector_length), matched_u64m2, vector_length);
        horizontal_positive_u64m2 = __riscv_vor_vv_u64m2(
            negative_u64m2,
            __riscv_vnot_v_u64m2(__riscv_vor_vv_u64m2(diagonal_u64m2, positive_u64m2, vector_length), vector_length),
            vector_length);
        horizontal_negative_u64m2 = __riscv_vand_vv_u64m2(positive_u64m2, diagonal_u64m2, vector_length);
        vuint64m2_t const shifted_positive_u64m2 = __riscv_vor_vv_u64m2(
            __riscv_vsll_vx_u64m2(horizontal_positive_u64m2, 1, vector_length), positive_carry_u64m2, vector_length);
        vuint64m2_t const shifted_negative_u64m2 = __riscv_vor_vv_u64m2(
            __riscv_vsll_vx_u64m2(horizontal_negative_u64m2, 1, vector_length), negative_carry_u64m2, vector_length);
        positive_carry_u64m2 = __riscv_vsrl_vx_u64m2(horizontal_positive_u64m2, 63, vector_length);
        negative_carry_u64m2 = __riscv_vsrl_vx_u64m2(horizontal_negative_u64m2, 63, vector_length);
        __riscv_vse64_v_u64m2(
            positive,
            __riscv_vor_vv_u64m2(
                shifted_negative_u64m2,
                __riscv_vnot_v_u64m2(__riscv_vor_vv_u64m2(vertical_carry_u64m2, shifted_positive_u64m2, vector_length),
                                     vector_length),
                vector_length),
            vector_length);
        __riscv_vse64_v_u64m2(negative,
                              __riscv_vand_vv_u64m2(shifted_positive_u64m2, vertical_carry_u64m2, vector_length),
                              vector_length);
    }
    // The last word's deltas, before their shift, move the score at the query's last symbol.
    sz_u64_t const last_symbol_bit = sz_levenshtein_last_symbol_bit_(query->length);
    vbool32_t const raised_b32 = __riscv_vmsne_vx_u64m2_b32(
        __riscv_vand_vx_u64m2(horizontal_positive_u64m2, last_symbol_bit, vector_length), 0, vector_length);
    vbool32_t const lowered_b32 = __riscv_vmsne_vx_u64m2_b32(
        __riscv_vand_vx_u64m2(horizontal_negative_u64m2, last_symbol_bit, vector_length), 0, vector_length);
    scores_u64m2 = __riscv_vadd_vx_u64m2_mu(raised_b32, scores_u64m2, scores_u64m2, 1, vector_length);
    return __riscv_vsub_vx_u64m2_mu(lowered_b32, scores_u64m2, scores_u64m2, 1, vector_length);
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64m2_score_rvv_(vuint64m2_t scores_u64m2, sz_size_t candidate) {
    return (sz_size_t)__riscv_vmv_x_s_u64m2_u64(__riscv_vslidedown_vx_u64m2(scores_u64m2, candidate, 1));
}

/** Starts a group of @p lanes candidates at the top boundary, returning their scores at the
 *  query's length; each of @p words verticals holds a word's positive deltas for every lane,
 *  then its negative deltas for every lane. */
STRINGZILLA_INLINE vuint64m2_t sz_levenshtein_u64m2_init_rvv(sz_u64_t *verticals, sz_size_t words, sz_size_t lanes,
                                                             sz_levenshtein_query_t const *query) {
    return sz_levenshtein_u64m2_init_rvv_(verticals, words, lanes, query);
}

/** Advances the first @p vector_length of a group's candidates by the mask offsets of their
 *  classes, returning their scores; the score moves on the last of @p words. */
STRINGZILLA_INLINE vuint64m2_t sz_levenshtein_u64m2_step_rvv(vuint64m2_t scores_u64m2, sz_u64_t *verticals,
                                                             sz_size_t words, sz_size_t lanes,
                                                             sz_levenshtein_query_t const *query,
                                                             vuint32m1_t offsets_u32m1, sz_size_t vector_length) {
    return sz_levenshtein_u64m2_step_rvv_(scores_u64m2, verticals, words, lanes, query, offsets_u32m1, vector_length);
}

/** Tests whether any unfinished one of the first @p vector_length candidates can still reach the
 *  radius, their @p symbol_counts read from memory. */
STRINGZILLA_INLINE sz_bool_t sz_levenshtein_u64m2_any_active_rvv(vuint64m2_t scores_u64m2,
                                                                 sz_u64_t const *symbol_counts, sz_size_t position,
                                                                 sz_ssize_t radius, sz_size_t vector_length) {
    vuint64m2_t const counts_u64m2 = __riscv_vle64_v_u64m2(symbol_counts, vector_length);
    vint64m2_t const floor_i64m2 = __riscv_vreinterpret_v_u64m2_i64m2(__riscv_vsub_vv_u64m2(
        __riscv_vadd_vx_u64m2(scores_u64m2, position, vector_length), counts_u64m2, vector_length));
    vbool32_t const reachable_b32 = __riscv_vmsle_vx_i64m2_b32(floor_i64m2, radius, vector_length);
    vbool32_t const unfinished_b32 = __riscv_vmsgtu_vx_u64m2_b32(counts_u64m2, position, vector_length);
    vbool32_t const active_b32 = __riscv_vmand_mm_b32(reachable_b32, unfinished_b32, vector_length);
    return __riscv_vfirst_m_b32(active_b32, vector_length) >= 0 ? sz_true_k : sz_false_k;
}

/** Reads a candidate's running score at its text's end. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64m2_score_rvv(vuint64m2_t scores_u64m2, sz_size_t candidate) {
    return sz_levenshtein_u64m2_score_rvv_(scores_u64m2, candidate);
}

STRINGZILLA_INLINE vuint32m1_t sz_levenshtein_u64m2_classes_u8_rvv_(sz_u8_t const *classes, sz_size_t vector_length) {
    vuint32m1_t const classes_u32m1 = __riscv_vzext_vf4_u32m1(__riscv_vle8_v_u8mf4(classes, vector_length),
                                                              vector_length);
    return __riscv_vsll_vx_u32m1(classes_u32m1, 3, vector_length);
}

/** Byte-wide class ids, as the byte transpose emits them, widened to the byte offsets of their
 *  eight-byte masks. */
STRINGZILLA_INLINE vuint32m1_t sz_levenshtein_u64m2_classes_u8_rvv(sz_u8_t const *classes, sz_size_t vector_length) {
    return sz_levenshtein_u64m2_classes_u8_rvv_(classes, vector_length);
}

STRINGZILLA_INLINE vuint32m1_t sz_levenshtein_u64m2_classes_u32_rvv_(sz_u32_t const *classes, sz_size_t vector_length) {
    return __riscv_vsll_vx_u32m1(__riscv_vle32_v_u32m1(classes, vector_length), 3, vector_length);
}

/** Four-byte class ids, as the UTF-8 transpose emits them, turned into the byte offsets of their
 *  eight-byte masks. */
STRINGZILLA_INLINE vuint32m1_t sz_levenshtein_u64m2_classes_u32_rvv(sz_u32_t const *classes, sz_size_t vector_length) {
    return sz_levenshtein_u64m2_classes_u32_rvv_(classes, vector_length);
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8m8_transpose_rvv_(sz_levenshtein_query_t const *query,
                                                                sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                sz_size_t candidates, sz_size_t *cursors,
                                                                sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                sz_size_t positions, void *transpose_classes) {
    sz_unused_(symbol_counts), sz_unused_(transpose_start);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u8_t *const classes = (sz_u8_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != candidates; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    vuint8m8_t const zeros_u8m8 = __riscv_vmv_v_x_u8m8(0, __riscv_vsetvlmax_e8m8());
    for (sz_size_t candidate = 0; candidate != candidates; ++candidate) {
        sz_size_t const taken = sz_min_of_two(filled, (sz_size_t)byte_counts[candidate] - cursors[candidate]);
        sz_size_t vector_length = 0;
        // Each strip of a candidate's bytes is classed by a table lookup, stored down its column.
        for (sz_size_t position = 0; position < taken; position += vector_length) {
            vector_length = __riscv_vsetvl_e8m8(taken - position);
            vuint8m8_t const bytes_u8m8 = __riscv_vle8_v_u8m8(
                (sz_u8_t const *)texts[candidate] + cursors[candidate] + position, vector_length);
            __riscv_vsse8_v_u8m8(classes + position * candidates + candidate, (ptrdiff_t)candidates,
                                 __riscv_vluxei8_v_u8m8(byte_to_class, bytes_u8m8, vector_length), vector_length);
        }
        for (sz_size_t position = taken; position < filled; position += vector_length) {
            vector_length = __riscv_vsetvl_e8m8(filled - position);
            __riscv_vsse8_v_u8m8(classes + position * candidates + candidate, (ptrdiff_t)candidates, zeros_u8m8,
                                 vector_length);
        }
        cursors[candidate] += taken;
    }
    return filled;
}

/** The byte transpose for any number of candidates: each candidate's bytes in strips as wide as
 *  the vector, classed by an indexed table load and stored down that candidate's column by a
 *  strided store, zeros past its end; emits the @c sz_u8_t class of every byte. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8m8_transpose_rvv(sz_levenshtein_query_t const *query,
                                                               sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                               sz_size_t candidates, sz_size_t *cursors,
                                                               sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                               sz_size_t positions, void *transpose_classes) {
    return sz_levenshtein_u8m8_transpose_rvv_(query, texts, byte_counts, candidates, cursors, symbol_counts,
                                              transpose_start, positions, transpose_classes);
}

/** Sweeps up to @p lanes candidates, one per lane of a register group, through every transpose in
 *  @p transpose_classes ; @p words and @p width are constants, keeping the class loads unbranched.
 *  Each transpose is consumed in runs that step the lanes through the highest unread candidate from
 *  one event to the next: a candidate's text ending, where its score is read and the stepped lanes
 *  may shrink, or a freeze test, which ends the sweep once every unread candidate is past
 *  @p radius, after which the unread ones read their running scores where they stand. */
STRINGZILLA_INLINE void sz_levenshtein_u64m2_sweep_rvv_(sz_levenshtein_query_t const *shared_query,
                                                        sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                        sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                        sz_levenshtein_transpose_t_ transpose,
                                                        sz_levenshtein_classes_width_t width, sz_ssize_t radius,
                                                        sz_size_t lanes, sz_u64_t *verticals, sz_size_t words,
                                                        sz_u32_t *transpose_classes, sz_size_t *distances) {
    enum { positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[sz_levenshtein_u64m2_candidates_max_rvv_k] = {0};
    sz_u64_t unread = sz_u64_mask_until_serial_(sweep_count);
    // Lanes through the highest one holding an unread candidate, all stepped by every run.
    sz_size_t vector_length = sweep_count;
    vuint64m2_t scores_u64m2 = sz_levenshtein_u64m2_init_rvv_(verticals, words, lanes, query);
    sz_u8_t const *const transpose_bytes = (sz_u8_t const *)transpose_classes;
    sz_size_t freeze_position = sz_levenshtein_freeze_start_(radius);
    for (sz_size_t transpose_start = 0, filled = positions_per_transpose_k; filled == positions_per_transpose_k;
         transpose_start += filled) {
        filled = transpose(query, texts, byte_counts, lanes, cursors, symbol_counts, transpose_start,
                           positions_per_transpose_k, transpose_classes);
        sz_u64_t counts[sz_levenshtein_u64m2_candidates_max_rvv_k];
        for (sz_size_t candidate = 0; candidate != lanes; ++candidate) counts[candidate] = symbol_counts[candidate];
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(unread, counts);
        for (sz_size_t position = 0;;) {
            sz_size_t const event = sz_min_of_two(deadline.position, freeze_position);
            sz_size_t const run_end = sz_min_of_two(event - transpose_start, filled);
            for (; position != run_end; ++position) {
                vuint32m1_t const offsets_u32m1 =
                    width == sz_levenshtein_classes_u8_k
                        ? sz_levenshtein_u64m2_classes_u8_rvv_(transpose_bytes + position * lanes, vector_length)
                        : sz_levenshtein_u64m2_classes_u32_rvv_(transpose_classes + position * lanes, vector_length);
                scores_u64m2 = sz_levenshtein_u64m2_step_rvv_(scores_u64m2, verticals, words, lanes, query,
                                                              offsets_u32m1, vector_length);
            }
            if (position == filled) break;
            if (event == deadline.position) {
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)sz_u64_ctz(ending);
                    distances[candidate] = sz_levenshtein_u64m2_score_rvv_(scores_u64m2, candidate);
                }
                unread &= ~deadline.retiring;
                vector_length = unread ? (sz_size_t)(64 - sz_u64_clz(unread)) : 0;
                deadline = sz_levenshtein_deadline_(unread, counts);
            }
            if (event == freeze_position) {
                // A zero fill ends the transposes, as when every candidate runs dry.
                if (sz_levenshtein_u64m2_any_active_rvv(scores_u64m2, counts, event, radius, vector_length) ==
                    sz_false_k) {
                    filled = 0;
                    break;
                }
                freeze_position += sz_levenshtein_freeze_positions_k;
            }
        }
    }
    // Unread candidates end where the transposes ran dry, or froze and read their running score.
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)sz_u64_ctz(unread);
        distances[candidate] = sz_levenshtein_u64m2_score_rvv_(scores_u64m2, candidate);
    }
}

/** Streams every candidate through a prepared @p query, @p lanes at a time in @p order, with
 *  @p transpose emitting their classes at @p width into @p transpose_classes , each distance under
 *  @p bound ; @p verticals holds enough for the query's words. */
STRINGZILLA_INLINE void sz_levenshtein_u64m2_distances_rvv_(
    sz_levenshtein_query_t const *query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_size_t lanes, sz_u64_t *verticals, sz_u32_t *transpose_classes, sz_size_t *distances) {
    enum { candidates_max_k = sz_levenshtein_u64m2_candidates_max_rvv_k };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    for (sz_size_t next_rank = 0; next_rank != candidates->count;) {
        sz_cptr_t texts[candidates_max_k];
        sz_u64_t byte_counts[candidates_max_k], sweep_symbol_counts[candidates_max_k];
        sz_size_t indices[candidates_max_k], sweep_distances[candidates_max_k];
        sz_size_t const sweep_count = sz_levenshtein_sweep_seed_(candidates, symbol_counts, order, query, bound,
                                                                 &next_rank, lanes, indices, texts, byte_counts,
                                                                 sweep_symbol_counts, distances);
        if (!sweep_count) break;
        if (words == 1)
            sz_levenshtein_u64m2_sweep_rvv_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                            width, radius, lanes, verticals, 1, transpose_classes, sweep_distances);
        else if (words == 2)
            sz_levenshtein_u64m2_sweep_rvv_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                            width, radius, lanes, verticals, 2, transpose_classes, sweep_distances);
        else
            sz_levenshtein_u64m2_sweep_rvv_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                            width, radius, lanes, verticals, words, transpose_classes, sweep_distances);
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
            distances[indices[candidate]] = sz_levenshtein_bounded_(sweep_distances[candidate], bound);
    }
}

#if STRINGZILLA_TARGET_RVV

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_rvv(sz_levenshtein_engine_t *engine,
                                                           sz_strings_view_t const *queries,
                                                           sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                           sz_stream_t stream) {
    return sz_levenshtein_engine_init_cpu_(engine, queries, symbol, sz_cap_rvv_k, allocator, stream);
}

/** Scores every candidate of a round against @p engine 's query @p index on RVV. */
STRINGZILLA_INLINE void sz_levenshtein_run_rvv_(sz_levenshtein_engine_t const *engine, sz_size_t index,
                                                sz_strings_view_t const *scored, sz_size_t const *symbol_counts,
                                                sz_size_t const *order, sz_size_t bound, sz_size_t lanes,
                                                sz_u64_t *verticals, sz_u32_t *transpose_classes,
                                                sz_size_t *distances) {
    if (engine->lengths[index] == 0) {
        sz_levenshtein_engine_empty_(scored, symbol_counts, bound, distances);
        return;
    }
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
    // Each alphabet passes its transpose and class width as literals, which the sweeps fold.
    if (engine->symbol == sz_levenshtein_bytes_k)
        sz_levenshtein_u64m2_distances_rvv_(&query, scored, symbol_counts, order, sz_levenshtein_u8m8_transpose_rvv_,
                                            sz_levenshtein_classes_u8_k, bound, lanes, verticals, transpose_classes,
                                            distances);
    else
        sz_levenshtein_u64m2_distances_rvv_(&query, scored, symbol_counts, order, sz_levenshtein_transpose_runes_,
                                            sz_levenshtein_classes_u32_k, bound, lanes, verticals, transpose_classes,
                                            distances);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_rvv(sz_levenshtein_engine_t *engine,
                                                         sz_strings_view_t const *candidates,
                                                         sz_size_t const *symbol_counts, sz_size_t const *order,
                                                         sz_size_t bound, sz_size_t *distances,
                                                         sz_size_t distances_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_assert_((engine->capability & sz_cap_cpus_k) != 0 &&
               "A host tier never scores a device-prepared engine, whose head only its GPU tier reads");
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    sz_size_t const lanes = sz_levenshtein_u64m2_lanes_rvv_();
    sz_size_t const words_max = sz_levenshtein_engine_words_max_(engine);
    // The transposes ride past the verticals, as their rows widen with the vector length.
    sz_size_t const verticals_bytes = sz_levenshtein_engine_verticals_bytes_(1, words_max,
                                                                             2 * lanes * sizeof(sz_u64_t)) +
                                      sz_levenshtein_positions_per_transpose_k * lanes * sizeof(sz_u32_t);
    sz_strings_view_t scored;
    sz_size_t const *candidate_order;
    sz_status_t const prepared = sz_levenshtein_engine_round_(
        engine, candidates, order, verticals_bytes, sz_utf8_decode_rvv_, &scored, &symbol_counts, &candidate_order);
    if (prepared != sz_success_k) return prepared;
    sz_u64_t *const verticals = (sz_u64_t *)sz_levenshtein_engine_verticals_(engine);
    sz_u32_t *const transpose_classes = (sz_u32_t *)(verticals + words_max * 2 * lanes);
    for (sz_size_t index = 0; index != engine->count; ++index)
        sz_levenshtein_run_rvv_(engine, index, &scored, symbol_counts, candidate_order, bound, lanes, verticals,
                                transpose_classes, distances + index * distances_stride);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_RVV

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_RISCV64_RVV_
#endif // STRINGZILLA_ARCH_RISCV64_

#endif // STRINGZILLA_LEVENSHTEIN_RVV_H_
