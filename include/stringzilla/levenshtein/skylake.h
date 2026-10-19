/**
 *  @file include/stringzilla/levenshtein/skylake.h
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Skylake-X (AVX-512F) backend for Levenshtein edit distances: eight candidates per ZMM,
 *      one 64-bit Myers word per candidate, the Myers booleans folded into @c VPTERNLOGQ and the
 *      score deltas into masked adds.
 *
 *  Every intrinsic here is AVX-512F, so Skylake-SP, Cascade Lake and Cooper Lake run the
 *  eight-wide kernel too.
 *
 *  @sa include/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_SKYLAKE_H_
#define STRINGZILLA_LEVENSHTEIN_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/utf8_runes/haswell.h" // `sz_utf8_decode_haswell_`

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_SKYLAKE_
#pragma region Skylake Implementation
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(__attribute__((target("avx,avx512f,bmi,popcnt,evex512"))), apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,bmi,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "bmi", "popcnt")
#endif

/** Eight candidates' running scores, one per 64-bit ZMM position. */
typedef struct sz_levenshtein_u64x8_state_skylake_t {

    /** The running edit distance per candidate. */
    sz_u512_vec_t scores_vec;
} sz_levenshtein_u64x8_state_skylake_t;

/** One query word of eight candidates' Myers states - the vertical deltas of that word. */
typedef struct sz_levenshtein_u64x8_vertical_skylake_t {

    /** Myers' VP per candidate. */
    sz_u512_vec_t positive_vec;

    /** Myers' VN per candidate. */
    sz_u512_vec_t negative_vec;
} sz_levenshtein_u64x8_vertical_skylake_t;

STRINGZILLA_INLINE void sz_levenshtein_u64x8_init_skylake_(sz_levenshtein_u64x8_state_skylake_t *state,
                                                           sz_levenshtein_u64x8_vertical_skylake_t *verticals,
                                                           sz_size_t words, sz_levenshtein_query_t const *query) {
    state->scores_vec.zmm = _mm512_set1_epi64((long long)query->length);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.zmm = _mm512_set1_epi64(-1);
        verticals[word].negative_vec.zmm = _mm512_setzero_si512();
    }
}

/** Starts eight candidates: the scores at the query's length, and @p words verticals at
 *  the top boundary. */
STRINGZILLA_INLINE void sz_levenshtein_u64x8_init_skylake(sz_levenshtein_u64x8_state_skylake_t *state,
                                                          sz_levenshtein_u64x8_vertical_skylake_t *verticals,
                                                          sz_size_t words, sz_levenshtein_query_t const *query) {
    sz_levenshtein_u64x8_init_skylake_(state, verticals, words, query);
}

STRINGZILLA_INLINE sz_u512_vec_t sz_levenshtein_u64x8_classes_u8_skylake_(sz_u8_t const *classes) {
    sz_u512_vec_t classes_vec;
    classes_vec.zmm = _mm512_cvtepu8_epi64(_mm_loadl_epi64((__m128i const *)classes));
    return classes_vec;
}

/** Eight byte-wide class ids, as the byte transpose emits them, widened to gather indices. */
STRINGZILLA_INLINE sz_u512_vec_t sz_levenshtein_u64x8_classes_u8_skylake(sz_u8_t const *classes) {
    return sz_levenshtein_u64x8_classes_u8_skylake_(classes);
}

STRINGZILLA_INLINE sz_u512_vec_t sz_levenshtein_u64x8_classes_u32_skylake_(sz_u32_t const *classes) {
    sz_u512_vec_t classes_vec;
    classes_vec.zmm = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i const *)classes));
    return classes_vec;
}

/** Eight four-byte class ids, as the UTF-8 transpose emits them, widened to gather indices. */
STRINGZILLA_INLINE sz_u512_vec_t sz_levenshtein_u64x8_classes_u32_skylake(sz_u32_t const *classes) {
    return sz_levenshtein_u64x8_classes_u32_skylake_(classes);
}

STRINGZILLA_INLINE void sz_levenshtein_u64x8_step_skylake_(sz_levenshtein_u64x8_state_skylake_t *state,
                                                           sz_levenshtein_u64x8_vertical_skylake_t *verticals,
                                                           sz_size_t words, sz_levenshtein_query_t const *query,
                                                           sz_u512_vec_t classes_vec) {
    enum { ternary_xor_or_k = 0xBE, ternary_or_nor_k = 0xF1 };
    sz_u512_vec_t last_symbol_bit_vec, last_symbol_shift_vec, positive_carry_vec, negative_carry_vec;
    last_symbol_bit_vec.zmm = _mm512_set1_epi64((long long)sz_levenshtein_last_symbol_bit_(query->length));
    last_symbol_shift_vec.zmm = _mm512_set1_epi64((long long)sz_levenshtein_last_symbol_shift_(query->length));
    positive_carry_vec.zmm = _mm512_set1_epi64(1);
    negative_carry_vec.zmm = _mm512_setzero_si512();
    for (sz_size_t word = 0; word != words; ++word) {
        sz_levenshtein_u64x8_vertical_skylake_t *const vertical = verticals + word;
        sz_u512_vec_t equality_vec, vertical_carry_vec, matched_vec, sum_vec, diagonal_vec;
        sz_u512_vec_t horizontal_positive_vec, horizontal_negative_vec, next_positive_vec, next_negative_vec;
        // A word's classes sit side by side, so a lane's class indexes that word's table directly.
        equality_vec.zmm = _mm512_i64gather_epi64(classes_vec.zmm, query->masks + word * query->stride, 8);
        vertical_carry_vec.zmm = _mm512_or_si512(equality_vec.zmm, vertical->negative_vec.zmm);
        matched_vec.zmm = _mm512_or_si512(equality_vec.zmm, negative_carry_vec.zmm);
        sum_vec.zmm = _mm512_add_epi64(_mm512_and_si512(matched_vec.zmm, vertical->positive_vec.zmm),
                                       vertical->positive_vec.zmm);
        diagonal_vec.zmm = _mm512_ternarylogic_epi64(sum_vec.zmm, vertical->positive_vec.zmm, matched_vec.zmm,
                                                     ternary_xor_or_k);
        horizontal_positive_vec.zmm = _mm512_ternarylogic_epi64(vertical->negative_vec.zmm, diagonal_vec.zmm,
                                                                vertical->positive_vec.zmm, ternary_or_nor_k);
        horizontal_negative_vec.zmm = _mm512_and_si512(vertical->positive_vec.zmm, diagonal_vec.zmm);
        if (word + 1 == words) {
            state->scores_vec.zmm = _mm512_add_epi64(
                state->scores_vec.zmm,
                _mm512_srlv_epi64(_mm512_and_si512(horizontal_positive_vec.zmm, last_symbol_bit_vec.zmm),
                                  last_symbol_shift_vec.zmm));
            state->scores_vec.zmm = _mm512_sub_epi64(
                state->scores_vec.zmm,
                _mm512_srlv_epi64(_mm512_and_si512(horizontal_negative_vec.zmm, last_symbol_bit_vec.zmm),
                                  last_symbol_shift_vec.zmm));
        }
        next_positive_vec.zmm = _mm512_srli_epi64(horizontal_positive_vec.zmm, 63);
        next_negative_vec.zmm = _mm512_srli_epi64(horizontal_negative_vec.zmm, 63);
        horizontal_positive_vec.zmm = _mm512_or_si512(_mm512_slli_epi64(horizontal_positive_vec.zmm, 1),
                                                      positive_carry_vec.zmm);
        horizontal_negative_vec.zmm = _mm512_or_si512(_mm512_slli_epi64(horizontal_negative_vec.zmm, 1),
                                                      negative_carry_vec.zmm);
        positive_carry_vec = next_positive_vec, negative_carry_vec = next_negative_vec;
        vertical->positive_vec.zmm = _mm512_ternarylogic_epi64(horizontal_negative_vec.zmm, vertical_carry_vec.zmm,
                                                               horizontal_positive_vec.zmm, ternary_or_nor_k);
        vertical->negative_vec.zmm = _mm512_and_si512(horizontal_positive_vec.zmm, vertical_carry_vec.zmm);
    }
}

/** Advances eight candidates one symbol through exactly @p words verticals; the score moves on the
 *  last word. A candidate past its text keeps stepping whatever class the transpose emits; its
 *  score is read where its text ends. */
STRINGZILLA_INLINE void sz_levenshtein_u64x8_step_skylake(sz_levenshtein_u64x8_state_skylake_t *state,
                                                          sz_levenshtein_u64x8_vertical_skylake_t *verticals,
                                                          sz_size_t words, sz_levenshtein_query_t const *query,
                                                          sz_u512_vec_t classes_vec) {
    sz_levenshtein_u64x8_step_skylake_(state, verticals, words, query, classes_vec);
}

/** Whether any of the eight candidates can still come under @p radius at @p position: a score
 *  falls by at most one per remaining symbol. Monotone, so once false it stays false;
 *  @c STRINGZILLA_SSIZE_MAX bounds nothing. */
STRINGZILLA_INLINE sz_bool_t sz_levenshtein_u64x8_any_active_skylake(sz_levenshtein_u64x8_state_skylake_t const *state,
                                                                     sz_u512_vec_t symbol_counts_vec,
                                                                     sz_size_t position, sz_ssize_t radius) {
    sz_u512_vec_t position_vec, floor_vec;
    position_vec.zmm = _mm512_set1_epi64((long long)position);
    floor_vec.zmm = _mm512_sub_epi64(_mm512_add_epi64(state->scores_vec.zmm, position_vec.zmm), symbol_counts_vec.zmm);
    __mmask8 const active_mask_m8 = _mm512_cmpgt_epi64_mask(symbol_counts_vec.zmm, position_vec.zmm) &
                                    _mm512_cmple_epi64_mask(floor_vec.zmm, _mm512_set1_epi64((long long)radius));
    return active_mask_m8 ? sz_true_k : sz_false_k;
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x8_score_skylake_(sz_levenshtein_u64x8_state_skylake_t const *state,
                                                                 sz_size_t candidate) {
    return state->scores_vec.u64s[candidate];
}

/** The running score of candidate @p candidate, read at the position where its text ends. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x8_score_skylake(sz_levenshtein_u64x8_state_skylake_t const *state,
                                                                sz_size_t candidate) {
    return sz_levenshtein_u64x8_score_skylake_(state, candidate);
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x8_transpose_skylake_(sz_levenshtein_query_t const *query,
                                                                    sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                    sz_size_t candidates, sz_size_t *cursors,
                                                                    sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                    sz_size_t positions, void *transpose_classes) {
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u8_t *const classes = (sz_u8_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != 8; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    sz_size_t position = 0;
    for (; position + 8 <= filled; position += 8) {
        sz_bool_t all_full = sz_true_k;
        for (sz_size_t candidate = 0; candidate != 8; ++candidate)
            all_full = all_full && cursors[candidate] + 8 <= byte_counts[candidate] ? sz_true_k : sz_false_k;
        if (!all_full) break;
        sz_u128_vec_t candidates_vec[8], pairs_vec[4], quads_vec[4];
        for (sz_size_t candidate = 0; candidate != 8; ++candidate)
            candidates_vec[candidate].xmm = _mm_cvtsi64_si128((long long)sz_levenshtein_octet_classes_(
                byte_to_class, sz_u64_load(texts[candidate] + cursors[candidate]).u64));
        for (sz_size_t pair = 0; pair != 4; ++pair)
            pairs_vec[pair].xmm = _mm_unpacklo_epi8(candidates_vec[2 * pair].xmm, candidates_vec[2 * pair + 1].xmm);
        quads_vec[0].xmm = _mm_unpacklo_epi16(pairs_vec[0].xmm, pairs_vec[1].xmm);
        quads_vec[1].xmm = _mm_unpackhi_epi16(pairs_vec[0].xmm, pairs_vec[1].xmm);
        quads_vec[2].xmm = _mm_unpacklo_epi16(pairs_vec[2].xmm, pairs_vec[3].xmm);
        quads_vec[3].xmm = _mm_unpackhi_epi16(pairs_vec[2].xmm, pairs_vec[3].xmm);
        _mm_storeu_si128((__m128i *)(classes + position * 8), _mm_unpacklo_epi32(quads_vec[0].xmm, quads_vec[2].xmm));
        _mm_storeu_si128((__m128i *)(classes + position * 8 + 16),
                         _mm_unpackhi_epi32(quads_vec[0].xmm, quads_vec[2].xmm));
        _mm_storeu_si128((__m128i *)(classes + position * 8 + 32),
                         _mm_unpacklo_epi32(quads_vec[1].xmm, quads_vec[3].xmm));
        _mm_storeu_si128((__m128i *)(classes + position * 8 + 48),
                         _mm_unpackhi_epi32(quads_vec[1].xmm, quads_vec[3].xmm));
        for (sz_size_t candidate = 0; candidate != 8; ++candidate) cursors[candidate] += 8;
    }
    for (; position != filled; ++position)
        for (sz_size_t candidate = 0; candidate != 8; ++candidate)
            classes[position * 8 + candidate] = cursors[candidate] < byte_counts[candidate]
                                                    ? byte_to_class[(sz_u8_t)texts[candidate][cursors[candidate]++]]
                                                    : 0;
    return filled;
}

/** The byte transpose for eight candidates: eight positions per eight loads and three rounds of
 *  unpacks while every candidate has eight bytes left, one byte at a time after that; emits the
 *  @c sz_u8_t class of every byte. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x8_transpose_skylake(sz_levenshtein_query_t const *query,
                                                                   sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                   sz_size_t candidates, sz_size_t *cursors,
                                                                   sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                   sz_size_t positions, void *transpose_classes) {
    return sz_levenshtein_u8x8_transpose_skylake_(query, texts, byte_counts, candidates, cursors, symbol_counts,
                                                  transpose_start, positions, transpose_classes);
}

/** One ZMM register per position; unmeasured, the Haswell reading carried over. */
enum {
    sz_levenshtein_u64x8_candidates_per_step_skylake_k = 8,
    sz_levenshtein_u64x8_registers_per_position_skylake_k = 1
};

/** Sweeps eight candidates through every transpose; @p words and @p width are constants, keeping
 *  short queries' verticals in registers and the class loads unbranched. Each transpose is consumed
 *  in runs from one event to the next: a candidate's text ending, where its score is read, or a
 *  freeze test, which ends the sweep once every unread candidate is past @p radius. Whatever is
 *  still unread then reads its running score; @p symbol_counts seeds as byte counts and the
 *  transpose refines them as it goes. */
STRINGZILLA_INLINE void sz_levenshtein_u64x8_sweep_skylake_(sz_levenshtein_query_t const *shared_query,
                                                            sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                            sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                            sz_levenshtein_transpose_t_ transpose,
                                                            sz_levenshtein_classes_width_t width, sz_ssize_t radius,
                                                            sz_levenshtein_u64x8_vertical_skylake_t *verticals,
                                                            sz_size_t words, sz_size_t *distances) {
    enum { candidates_per_position_k = 8, positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k };
    // A local copy: nothing stored through the verticals can alias it, so the step keeps its fields in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_u64_t unread = ((sz_u64_t)1 << sweep_count) - 1;
    sz_levenshtein_u64x8_state_skylake_t state;
    sz_levenshtein_u64x8_init_skylake_(&state, verticals, words, query);
    sz_u32_t transpose_classes[positions_per_transpose_k][candidates_per_position_k];
    sz_size_t freeze_position = sz_levenshtein_freeze_start_(radius);
    for (sz_size_t transpose_start = 0, filled = positions_per_transpose_k; filled == positions_per_transpose_k;
         transpose_start += filled) {
        filled = transpose(query, texts, byte_counts, candidates_per_position_k, cursors, symbol_counts,
                           transpose_start, positions_per_transpose_k, &transpose_classes[0][0]);
        // A local copy: the transpose may refine the counts, and stores into `distances` must not force reloads.
        sz_u64_t counts[candidates_per_position_k];
        for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
            counts[candidate] = symbol_counts[candidate];
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(unread, counts);
        for (sz_size_t position = 0;;) {
            sz_size_t const event = sz_min_of_two(deadline.position, freeze_position);
            sz_size_t const run_end = sz_min_of_two(event - transpose_start, filled);
            for (; position != run_end; ++position) {
                sz_u512_vec_t const classes_vec =
                    width == sz_levenshtein_classes_u8_k
                        ? sz_levenshtein_u64x8_classes_u8_skylake_((sz_u8_t const *)&transpose_classes[0][0] +
                                                                   position * candidates_per_position_k)
                        : sz_levenshtein_u64x8_classes_u32_skylake_(transpose_classes[position]);
                sz_levenshtein_u64x8_step_skylake_(&state, verticals, words, query, classes_vec);
            }
            if (position == filled) break;
            if (event == deadline.position) {
                // The only read of the scores by lane index, so the run keeps them in a register.
                sz_levenshtein_u64x8_state_skylake_t const ended = state;
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)_tzcnt_u64(ending);
                    distances[candidate] = sz_levenshtein_u64x8_score_skylake_(&ended, candidate);
                }
                unread &= ~deadline.retiring;
                deadline = sz_levenshtein_deadline_(unread, counts);
            }
            if (event == freeze_position) {
                sz_u512_vec_t counts_vec;
                counts_vec.zmm = _mm512_loadu_si512((void const *)counts);
                // A zero fill ends the transposes, as when every candidate runs dry.
                if (sz_levenshtein_u64x8_any_active_skylake(&state, counts_vec, event, radius) == sz_false_k) {
                    filled = 0;
                    break;
                }
                freeze_position += sz_levenshtein_freeze_positions_k;
            }
        }
    }
    // Unread candidates end where the transposes ran dry, or froze and read their running score.
    sz_levenshtein_u64x8_state_skylake_t const ended = state;
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)_tzcnt_u64(unread);
        distances[candidate] = sz_levenshtein_u64x8_score_skylake_(&ended, candidate);
    }
}

/** Streams every candidate through a prepared @p query, eight at a time in @p order, with
 *  @p transpose emitting their classes at @p width, each distance under @p bound; @p verticals
 *  holds enough for a runtime word count. */
STRINGZILLA_INLINE void sz_levenshtein_u64x8_distances_skylake_(
    sz_levenshtein_query_t const *query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_levenshtein_u64x8_vertical_skylake_t *verticals, sz_size_t *distances) {
    enum { candidates_per_position_k = 8 };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_levenshtein_u64x8_vertical_skylake_t resident_verticals[2];
    for (sz_size_t next_rank = 0; next_rank != candidates->count;) {
        sz_cptr_t texts[candidates_per_position_k];
        sz_u64_t byte_counts[candidates_per_position_k], sweep_symbol_counts[candidates_per_position_k];
        sz_size_t indices[candidates_per_position_k], sweep_distances[candidates_per_position_k];
        sz_size_t const sweep_count = sz_levenshtein_sweep_seed_(candidates, symbol_counts, order, query, bound,
                                                                 &next_rank, candidates_per_position_k, indices, texts,
                                                                 byte_counts, sweep_symbol_counts, distances);
        if (!sweep_count) break;
        if (words == 1)
            sz_levenshtein_u64x8_sweep_skylake_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                width, radius, resident_verticals, 1, sweep_distances);
        else if (words == 2)
            sz_levenshtein_u64x8_sweep_skylake_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                width, radius, resident_verticals, 2, sweep_distances);
        else
            sz_levenshtein_u64x8_sweep_skylake_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                width, radius, verticals, words, sweep_distances);
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
            distances[indices[candidate]] = sz_levenshtein_bounded_(sweep_distances[candidate], bound);
    }
}

/** The lanes of eight candidates whose floor, their running score less the symbols left before
 *  their @p ends_vec, has passed @p radius at @p position. */
STRINGZILLA_INLINE __mmask8 sz_levenshtein_u64x8_frozen_skylake_(sz_levenshtein_u64x8_state_skylake_t const *state,
                                                                 sz_u512_vec_t ends_vec, sz_size_t position,
                                                                 sz_ssize_t radius) {
    __m512i const floor_i64x8 = _mm512_sub_epi64(
        _mm512_add_epi64(state->scores_vec.zmm, _mm512_set1_epi64((long long)position)), ends_vec.zmm);
    return _mm512_cmpgt_epi64_mask(floor_i64x8, _mm512_set1_epi64((long long)radius));
}

/** Restarts the lanes of @p fresh_m8 at the top boundary: their scores at the query's length and
 *  their verticals at all positive, leaving every other lane as it is. */
STRINGZILLA_INLINE void sz_levenshtein_u64x8_restart_skylake_(sz_levenshtein_u64x8_state_skylake_t *state,
                                                              sz_levenshtein_u64x8_vertical_skylake_t *verticals,
                                                              sz_size_t words, sz_levenshtein_query_t const *query,
                                                              __mmask8 fresh_m8) {
    state->scores_vec.zmm = _mm512_mask_set1_epi64(state->scores_vec.zmm, fresh_m8, (long long)query->length);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.zmm = _mm512_mask_set1_epi64(verticals[word].positive_vec.zmm, fresh_m8, -1);
        verticals[word].negative_vec.zmm = _mm512_maskz_mov_epi64((__mmask8)~fresh_m8,
                                                                  verticals[word].negative_vec.zmm);
    }
}

/** Streams every candidate through eight lanes that recycle: a lane whose candidate ends or freezes
 *  past @p bound is reported and, at the next freeze test, takes the next candidate in @p order, so
 *  lanes idle at most a freeze interval rather than until their group's longest ends. @p words and
 *  @p width are constants, as in @ref sz_levenshtein_u64x8_sweep_skylake_. */
STRINGZILLA_INLINE void sz_levenshtein_u64x8_recycle_skylake_(
    sz_levenshtein_query_t const *shared_query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_levenshtein_u64x8_vertical_skylake_t *verticals, sz_size_t words, sz_size_t *distances) {
    enum { lanes_k = 8, block_k = sz_levenshtein_freeze_positions_k };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_cptr_t texts[lanes_k] = {STRINGZILLA_NULL};
    sz_u64_t byte_counts[lanes_k] = {0}, ends[lanes_k] = {0};
    sz_size_t cursors[lanes_k] = {0}, slots[lanes_k] = {0};
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_u32_t transpose_classes[block_k][lanes_k] = {{0}};
    sz_levenshtein_u64x8_state_skylake_t state;
    sz_levenshtein_u64x8_init_skylake_(&state, verticals, words, query);
    sz_u64_t live = 0, idle = (1u << lanes_k) - 1;
    sz_size_t next_rank = 0;
    for (sz_size_t block_start = 0;; block_start += block_k) {
        // Idle lanes take the next candidates in order, starting at this block.
        if (idle && next_rank != candidates->count) {
            sz_cptr_t fresh_texts[lanes_k];
            sz_u64_t fresh_bytes[lanes_k], fresh_symbols[lanes_k];
            sz_size_t fresh_slots[lanes_k];
            sz_size_t const fresh_count = sz_levenshtein_sweep_seed_(
                candidates, symbol_counts, order, query, bound, &next_rank, (sz_size_t)_mm_popcnt_u64(idle),
                fresh_slots, fresh_texts, fresh_bytes, fresh_symbols, distances);
            sz_u64_t fresh = 0;
            for (sz_size_t taken = 0; taken != fresh_count; ++taken) {
                sz_size_t const lane = (sz_size_t)_tzcnt_u64(idle);
                idle &= idle - 1, fresh |= (sz_u64_t)1 << lane;
                texts[lane] = fresh_texts[taken], byte_counts[lane] = fresh_bytes[taken], cursors[lane] = 0;
                slots[lane] = fresh_slots[taken], ends[lane] = block_start + fresh_symbols[taken];
            }
            sz_levenshtein_u64x8_restart_skylake_(&state, verticals, words, query, (__mmask8)fresh);
            live |= fresh;
        }
        if (!live) break;
        for (sz_u64_t retired = idle; retired; retired &= retired - 1) {
            sz_size_t const lane = (sz_size_t)_tzcnt_u64(retired);
            texts[lane] = STRINGZILLA_NULL, byte_counts[lane] = 0, cursors[lane] = 0;
        }
        transpose(query, texts, byte_counts, lanes_k, cursors, STRINGZILLA_NULL, block_start, block_k,
                  &transpose_classes[0][0]);
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(live, ends);
        for (sz_size_t position = 0;;) {
            sz_size_t const run_end = sz_min_of_two(deadline.position - block_start, (sz_size_t)block_k);
            for (; position < run_end; ++position) {
                sz_u512_vec_t const classes_vec =
                    width == sz_levenshtein_classes_u8_k
                        ? sz_levenshtein_u64x8_classes_u8_skylake_((sz_u8_t const *)&transpose_classes[0][0] +
                                                                   position * lanes_k)
                        : sz_levenshtein_u64x8_classes_u32_skylake_(transpose_classes[position]);
                sz_levenshtein_u64x8_step_skylake_(&state, verticals, words, query, classes_vec);
            }
            if (deadline.position != block_start + position) break;
            for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                sz_size_t const lane = (sz_size_t)_tzcnt_u64(ending);
                distances[slots[lane]] = sz_levenshtein_bounded_(state.scores_vec.u64s[lane], bound);
            }
            live &= ~deadline.retiring, idle |= deadline.retiring;
            if (!live) break;
            deadline = sz_levenshtein_deadline_(live, ends);
        }
        // At every block's end a lane already past the bound reports it and frees its lane.
        sz_u512_vec_t ends_vec;
        ends_vec.zmm = _mm512_loadu_si512((void const *)ends);
        sz_u64_t const frozen = live & (sz_u64_t)sz_levenshtein_u64x8_frozen_skylake_(&state, ends_vec,
                                                                                      block_start + block_k, radius);
        for (sz_u64_t ending = frozen; ending; ending &= ending - 1)
            distances[slots[(sz_size_t)_tzcnt_u64(ending)]] = bound + 1;
        live &= ~frozen, idle |= frozen;
    }
}

/** Streams every candidate through recycling lanes, specializing one- and two-word queries to keep
 *  their verticals in registers. */
STRINGZILLA_INLINE void sz_levenshtein_u64x8_recycled_skylake_(
    sz_levenshtein_query_t const *query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_levenshtein_u64x8_vertical_skylake_t *verticals, sz_size_t *distances) {
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_levenshtein_u64x8_vertical_skylake_t resident_verticals[2];
    if (words == 1)
        sz_levenshtein_u64x8_recycle_skylake_(query, candidates, symbol_counts, order, transpose, width, bound,
                                              resident_verticals, 1, distances);
    else if (words == 2)
        sz_levenshtein_u64x8_recycle_skylake_(query, candidates, symbol_counts, order, transpose, width, bound,
                                              resident_verticals, 2, distances);
    else
        sz_levenshtein_u64x8_recycle_skylake_(query, candidates, symbol_counts, order, transpose, width, bound,
                                              verticals, words, distances);
}

/** Scores every candidate of a round against @p engine 's query @p index in eight-wide lanes. */
STRINGZILLA_INLINE void sz_levenshtein_run_skylake_(sz_levenshtein_engine_t const *engine, sz_size_t index,
                                                    sz_strings_view_t const *scored, sz_size_t const *symbol_counts,
                                                    sz_size_t const *order, sz_size_t bound,
                                                    sz_levenshtein_u64x8_vertical_skylake_t *verticals,
                                                    sz_size_t *distances) {
    if (engine->lengths[index] == 0) {
        sz_levenshtein_engine_empty_(scored, symbol_counts, bound, distances);
        return;
    }
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
    // A bounded run frees a lane as soon as its candidate passes the bound, where a fixed group of
    // eight keeps it stepping until all eight do; an unbounded run's sorted lengths end together.
    // Each alphabet passes its transpose and class width as literals, which the sweeps fold.
    if (engine->symbol == sz_levenshtein_bytes_k && bound != STRINGZILLA_SIZE_MAX)
        sz_levenshtein_u64x8_recycled_skylake_(&query, scored, symbol_counts, order,
                                               sz_levenshtein_u8x8_transpose_skylake_, sz_levenshtein_classes_u8_k,
                                               bound, verticals, distances);
    else if (engine->symbol == sz_levenshtein_bytes_k)
        sz_levenshtein_u64x8_distances_skylake_(&query, scored, symbol_counts, order,
                                                sz_levenshtein_u8x8_transpose_skylake_, sz_levenshtein_classes_u8_k,
                                                bound, verticals, distances);
    else if (bound != STRINGZILLA_SIZE_MAX)
        sz_levenshtein_u64x8_recycled_skylake_(&query, scored, symbol_counts, order, sz_levenshtein_transpose_runes_,
                                               sz_levenshtein_classes_u32_k, bound, verticals, distances);
    else
        sz_levenshtein_u64x8_distances_skylake_(&query, scored, symbol_counts, order, sz_levenshtein_transpose_runes_,
                                                sz_levenshtein_classes_u32_k, bound, verticals, distances);
}

/** Bytes one round's eight-wide verticals take, sized for the batch's widest query. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_verticals_bytes_skylake_(sz_levenshtein_engine_t const *engine) {
    return sz_levenshtein_engine_verticals_bytes_(sz_levenshtein_u64x8_registers_per_position_skylake_k,
                                                  sz_levenshtein_engine_words_max_(engine),
                                                  sizeof(sz_levenshtein_u64x8_vertical_skylake_t));
}

STRINGZILLA_INLINE sz_status_t sz_levenshtein_distances_skylake_(sz_levenshtein_engine_t *engine,
                                                                 sz_strings_view_t const *candidates,
                                                                 sz_size_t const *symbol_counts, sz_size_t const *order,
                                                                 sz_size_t bound, sz_size_t *distances,
                                                                 sz_size_t distances_stride,
                                                                 sz_levenshtein_decode_t_ decode) {
    sz_assert_((engine->capability & sz_cap_cpus_k) != 0 &&
               "A host tier never scores a device-prepared engine, whose head only its GPU tier reads");
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    sz_size_t const verticals_bytes = sz_levenshtein_verticals_bytes_skylake_(engine);
    sz_strings_view_t scored;
    sz_size_t const *candidate_order;
    sz_status_t const prepared = sz_levenshtein_engine_round_(engine, candidates, order, verticals_bytes, decode,
                                                              &scored, &symbol_counts, &candidate_order);
    if (prepared != sz_success_k) return prepared;
    sz_levenshtein_u64x8_vertical_skylake_t *const verticals =
        (sz_levenshtein_u64x8_vertical_skylake_t *)sz_levenshtein_engine_verticals_(engine);
    for (sz_size_t index = 0; index != engine->count; ++index)
        sz_levenshtein_run_skylake_(engine, index, &scored, symbol_counts, candidate_order, bound, verticals,
                                    distances + index * distances_stride);
    return sz_success_k;
}

#if STRINGZILLA_TARGET_SKYLAKE

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_skylake(sz_levenshtein_engine_t *engine,
                                                               sz_strings_view_t const *queries,
                                                               sz_levenshtein_symbol_t symbol,
                                                               sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_levenshtein_engine_init_cpu_(engine, queries, symbol, sz_cap_skylake_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_skylake(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *candidates,
                                                             sz_size_t const *symbol_counts, sz_size_t const *order,
                                                             sz_size_t bound, sz_size_t *distances,
                                                             sz_size_t distances_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_levenshtein_distances_skylake_(engine, candidates, symbol_counts, order, bound, distances,
                                             distances_stride, sz_utf8_decode_haswell_);
}

#endif // STRINGZILLA_TARGET_SKYLAKE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#pragma endregion Skylake Implementation
#endif // STRINGZILLA_ARCH_X8664_SKYLAKE_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_LEVENSHTEIN_SKYLAKE_H_
