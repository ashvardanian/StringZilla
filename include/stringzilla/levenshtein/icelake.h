/**
 *  @file include/stringzilla/levenshtein/icelake.h
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Ice Lake (AVX-512 VBMI) backend for Levenshtein edit distances.
 *
 *  A query of at most eight symbols runs in byte lanes, sixty-four candidates per ZMM, the match
 *  masks read by a @c VPERMB; every wider query runs the Skylake eight-wide kernel, which needs
 *  nothing above AVX-512F.
 *
 *  @sa include/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_ICELAKE_H_
#define STRINGZILLA_LEVENSHTEIN_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/levenshtein/skylake.h"
#include "stringzilla/utf8_runes/icelake.h" // `sz_utf8_decode_icelake_`

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_TARGET_ICELAKE
#pragma region Ice Lake Implementation
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(                                                                                    \
    __attribute__((                                                                                              \
        target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,popcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                       \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,popcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "lzcnt", "popcnt")
#endif

#pragma region Narrow Lanes

/** Sixty-four candidates' score movement since the last flush, one signed byte per candidate. */
typedef struct sz_levenshtein_u8x64_state_icelake_t {

    /** How far each candidate's score has moved since the last flush. */
    sz_u512_vec_t deltas_vec;
} sz_levenshtein_u8x64_state_icelake_t;

/** Sixty-four candidates' Myers state for a query of at most eight symbols - one byte-wide
 *  word per candidate. */
typedef struct sz_levenshtein_u8x64_vertical_icelake_t {

    /** Myers' VP per candidate, the query's symbols in the low bits. */
    sz_u512_vec_t positive_vec;

    /** Myers' VN per candidate, the query's symbols in the low bits. */
    sz_u512_vec_t negative_vec;
} sz_levenshtein_u8x64_vertical_icelake_t;

/** A query of at most eight symbols as the byte lanes read it, so a step looks its masks up
 *  without a gather. */
typedef struct sz_levenshtein_u8x64_query_icelake_t {

    /** Class @c c 's match mask at byte @c c, every other byte zero. */
    sz_u512_vec_t masks_vec;

    /** The query's last symbol's bit, repeated in all sixty-four lanes. */
    sz_u512_vec_t last_symbol_bit_vec;
} sz_levenshtein_u8x64_query_icelake_t;

/** One ZMM per position, sixty-four candidates in it, and the vertical never leaves a register. */
enum {
    sz_levenshtein_u8x64_candidates_per_step_icelake_k = 64,
    sz_levenshtein_u8x64_registers_per_position_icelake_k = 1
};

/** Positions the byte-lane deltas span before a signed byte could wrap: a score moves by at most
 *  one per step. */
enum { sz_levenshtein_u8x64_positions_per_flush_icelake_k = 64 };

STRINGZILLA_INLINE void sz_levenshtein_u8x64_pack_icelake_(sz_levenshtein_query_t const *query,
                                                           sz_levenshtein_u8x64_query_icelake_t *packed) {
    sz_u512_vec_t masks_vec;
    masks_vec.zmm = _mm512_setzero_si512();
    for (sz_size_t class_id = 0; class_id != query->classes; ++class_id)
        masks_vec.u8s[class_id] = (sz_u8_t)query->masks[class_id];
    packed->masks_vec = masks_vec;
    packed->last_symbol_bit_vec.zmm = _mm512_set1_epi8((char)sz_levenshtein_last_symbol_bit_(query->length));
}

/** Packs a query of at most eight symbols into one byte per class, the table @c VPERMB indexes. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_pack_icelake(sz_levenshtein_query_t const *query,
                                                          sz_levenshtein_u8x64_query_icelake_t *packed) {
    sz_levenshtein_u8x64_pack_icelake_(query, packed);
}

STRINGZILLA_INLINE void sz_levenshtein_u8x64_init_icelake_(sz_levenshtein_u8x64_state_icelake_t *state,
                                                           sz_levenshtein_u8x64_vertical_icelake_t *vertical) {
    state->deltas_vec.zmm = _mm512_setzero_si512();
    vertical->positive_vec.zmm = _mm512_set1_epi8((char)-1);
    vertical->negative_vec.zmm = _mm512_setzero_si512();
}

/** Starts sixty-four candidates: no score movement yet, and the vertical at the top boundary. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_init_icelake(sz_levenshtein_u8x64_state_icelake_t *state,
                                                          sz_levenshtein_u8x64_vertical_icelake_t *vertical) {
    sz_levenshtein_u8x64_init_icelake_(state, vertical);
}

STRINGZILLA_INLINE sz_u512_vec_t sz_levenshtein_u8x64_classes_u8_icelake_(sz_u8_t const *classes) {
    sz_u512_vec_t classes_vec;
    classes_vec.zmm = _mm512_loadu_si512((void const *)classes);
    return classes_vec;
}

/** Sixty-four byte-wide class ids, as the byte transpose emits them, already the indices the
 *  mask table takes. */
STRINGZILLA_INLINE sz_u512_vec_t sz_levenshtein_u8x64_classes_u8_icelake(sz_u8_t const *classes) {
    return sz_levenshtein_u8x64_classes_u8_icelake_(classes);
}

STRINGZILLA_INLINE void sz_levenshtein_u8x64_step_icelake_(sz_levenshtein_u8x64_state_icelake_t *state,
                                                           sz_levenshtein_u8x64_vertical_icelake_t *vertical,
                                                           sz_levenshtein_u8x64_query_icelake_t const *packed,
                                                           sz_u512_vec_t classes_vec) {
    enum { ternary_xor_or_k = 0xBE, ternary_or_nor_k = 0xF1 };
    __m512i const ones_u8x64 = _mm512_set1_epi8((char)-1);
    __m512i const equality_u8x64 = _mm512_permutexvar_epi8(classes_vec.zmm, packed->masks_vec.zmm);
    __m512i const vertical_carry_u8x64 = _mm512_or_si512(equality_u8x64, vertical->negative_vec.zmm);
    __m512i const sum_u8x64 = _mm512_add_epi8(_mm512_and_si512(equality_u8x64, vertical->positive_vec.zmm),
                                              vertical->positive_vec.zmm);
    __m512i const diagonal_u8x64 = _mm512_ternarylogic_epi64(sum_u8x64, vertical->positive_vec.zmm, equality_u8x64,
                                                             ternary_xor_or_k);
    __m512i const horizontal_positive_u8x64 = _mm512_ternarylogic_epi64(vertical->negative_vec.zmm, diagonal_u8x64,
                                                                        vertical->positive_vec.zmm, ternary_or_nor_k);
    __m512i const horizontal_negative_u8x64 = _mm512_and_si512(vertical->positive_vec.zmm, diagonal_u8x64);
    __mmask64 const rose_m64 = _mm512_test_epi8_mask(horizontal_positive_u8x64, packed->last_symbol_bit_vec.zmm);
    __mmask64 const fell_m64 = _mm512_test_epi8_mask(horizontal_negative_u8x64, packed->last_symbol_bit_vec.zmm);
    state->deltas_vec.zmm = _mm512_mask_sub_epi8(state->deltas_vec.zmm, rose_m64, state->deltas_vec.zmm, ones_u8x64);
    state->deltas_vec.zmm = _mm512_mask_add_epi8(state->deltas_vec.zmm, fell_m64, state->deltas_vec.zmm, ones_u8x64);
    // Doubling a byte lane shifts it up by one, and the top symbol's carry leaves the lane exactly as it leaves a word.
    __m512i const shifted_positive_u8x64 = _mm512_sub_epi8(
        _mm512_add_epi8(horizontal_positive_u8x64, horizontal_positive_u8x64), ones_u8x64);
    __m512i const shifted_negative_u8x64 = _mm512_add_epi8(horizontal_negative_u8x64, horizontal_negative_u8x64);
    vertical->positive_vec.zmm = _mm512_ternarylogic_epi64(shifted_negative_u8x64, vertical_carry_u8x64,
                                                           shifted_positive_u8x64, ternary_or_nor_k);
    vertical->negative_vec.zmm = _mm512_and_si512(shifted_positive_u8x64, vertical_carry_u8x64);
}

/** Advances sixty-four candidates one symbol through one byte-wide Myers word, the masks read by a
 *  single permute. A candidate past its text keeps stepping whatever class the transpose emits; its
 *  score is read where its text ends. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_step_icelake(sz_levenshtein_u8x64_state_icelake_t *state,
                                                          sz_levenshtein_u8x64_vertical_icelake_t *vertical,
                                                          sz_levenshtein_u8x64_query_icelake_t const *packed,
                                                          sz_u512_vec_t classes_vec) {
    sz_levenshtein_u8x64_step_icelake_(state, vertical, packed, classes_vec);
}

STRINGZILLA_INLINE void sz_levenshtein_u8x64_flush_icelake_(sz_levenshtein_u8x64_state_icelake_t *state,
                                                            sz_size_t *scores) {
    enum { lanes_k = sz_levenshtein_u8x64_candidates_per_step_icelake_k };
    sz_u512_vec_t const deltas_vec = state->deltas_vec;
    for (sz_size_t candidate = 0; candidate != lanes_k; ++candidate)
        scores[candidate] = (sz_size_t)((sz_ssize_t)scores[candidate] + deltas_vec.i8s[candidate]);
    state->deltas_vec.zmm = _mm512_setzero_si512();
}

/** Folds the byte-lane deltas into @p scores and clears them, so every candidate's score
 *  is exact again. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_flush_icelake(sz_levenshtein_u8x64_state_icelake_t *state,
                                                           sz_size_t *scores) {
    sz_levenshtein_u8x64_flush_icelake_(state, scores);
}

/** Whether any of the @p unread candidates among the first @p lanes can still come under @p radius
 *  at @p position, read from flushed @p scores; every unread candidate ends past @p position. */
STRINGZILLA_INLINE sz_bool_t sz_levenshtein_flushed_any_active_icelake_(sz_size_t const *scores,
                                                                        sz_u64_t const *byte_counts, sz_size_t lanes,
                                                                        sz_u64_t unread, sz_size_t position,
                                                                        sz_ssize_t radius) {
    __m512i const position_u64x8 = _mm512_set1_epi64((long long)position);
    __m512i const radius_i64x8 = _mm512_set1_epi64((long long)radius);
    sz_u64_t active = 0;
    for (sz_size_t first = 0; first != lanes; first += 8) {
        __m512i const scores_u64x8 = _mm512_loadu_si512((void const *)(scores + first));
        __m512i const counts_u64x8 = _mm512_loadu_si512((void const *)(byte_counts + first));
        __m512i const floor_i64x8 = _mm512_sub_epi64(_mm512_add_epi64(scores_u64x8, position_u64x8), counts_u64x8);
        active |= (sz_u64_t)_mm512_cmple_epi64_mask(floor_i64x8, radius_i64x8) << first;
    }
    return (active & unread) ? sz_true_k : sz_false_k;
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x64_transpose_icelake_(sz_levenshtein_query_t const *query,
                                                                     sz_cptr_t const *texts,
                                                                     sz_u64_t const *byte_counts, sz_size_t candidates,
                                                                     sz_size_t *cursors, sz_u64_t *symbol_counts,
                                                                     sz_size_t transpose_start, sz_size_t positions,
                                                                     void *transpose_classes) {
    enum { lanes_k = sz_levenshtein_u8x64_candidates_per_step_icelake_k };
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u8_t *const classes = (sz_u8_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != lanes_k; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    for (sz_size_t candidate = 0; candidate != lanes_k; ++candidate) {
        sz_size_t const taken = sz_min_of_two(filled, (sz_size_t)byte_counts[candidate] - cursors[candidate]);
        for (sz_size_t position = 0; position != taken; ++position)
            classes[position * lanes_k + candidate] = (sz_u8_t)texts[candidate][cursors[candidate] + position];
        for (sz_size_t position = taken; position != filled; ++position) classes[position * lanes_k + candidate] = 0;
        cursors[candidate] += taken;
    }
    // A pair of permutes classes sixty-four staged bytes, the byte's top bit picking which half answers.
    __m512i const table_first_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 0));
    __m512i const table_second_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 64));
    __m512i const table_third_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 128));
    __m512i const table_fourth_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 192));
    for (sz_size_t staged = 0; staged != filled * lanes_k; staged += 64) {
        __m512i const bytes_u8x64 = _mm512_loadu_si512((void const *)(classes + staged));
        __m512i const low_half_u8x64 = _mm512_permutex2var_epi8(table_first_quarter_u8x64, bytes_u8x64,
                                                                table_second_quarter_u8x64);
        __m512i const high_half_u8x64 = _mm512_permutex2var_epi8(table_third_quarter_u8x64, bytes_u8x64,
                                                                 table_fourth_quarter_u8x64);
        _mm512_storeu_si512((void *)(classes + staged),
                            _mm512_mask_blend_epi8(_mm512_movepi8_mask(bytes_u8x64), low_half_u8x64, high_half_u8x64));
    }
    return filled;
}

/** The byte transpose for sixty-four candidates, transposed as every transpose is: position @c p of
 *  candidate @c c lands at @c p * 64 + c. The bytes are staged first and classed a whole position
 *  at a time, so a class id costs a lane of a permute rather than a scalar load; a lane past its
 *  text pads with a zero byte, classed like any other. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x64_transpose_icelake(sz_levenshtein_query_t const *query,
                                                                    sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                    sz_size_t candidates, sz_size_t *cursors,
                                                                    sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                    sz_size_t positions, void *transpose_classes) {
    return sz_levenshtein_u8x64_transpose_icelake_(query, texts, byte_counts, candidates, cursors, symbol_counts,
                                                   transpose_start, positions, transpose_classes);
}

/** Sweeps sixty-four candidates of a query of at most eight symbols through every transpose.
 *  Between a retirement and a flush the step loop carries no scalar work, so a run of positions
 *  costs only its permutes and logic. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_sweep_icelake_(sz_levenshtein_query_t const *shared_query,
                                                            sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                            sz_size_t sweep_count, sz_ssize_t radius,
                                                            sz_size_t *distances) {
    enum {
        candidates_per_position_k = sz_levenshtein_u8x64_candidates_per_step_icelake_k,
        positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k,
        positions_per_flush_k = sz_levenshtein_u8x64_positions_per_flush_icelake_k
    };
    // A local copy: nothing stored through the transpose can alias it, so the step keeps the packed query in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_u8x64_query_icelake_t packed;
    sz_levenshtein_u8x64_pack_icelake_(&local_query, &packed);
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_size_t scores[candidates_per_position_k];
    for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
        scores[candidate] = local_query.length;
    sz_u64_t unread = sweep_count == candidates_per_position_k ? ~(sz_u64_t)0
                                                               : ((sz_u64_t)1 << sweep_count) - (sz_u64_t)1;
    sz_levenshtein_u8x64_state_icelake_t state;
    sz_levenshtein_u8x64_vertical_icelake_t vertical;
    sz_levenshtein_u8x64_init_icelake_(&state, &vertical);
    sz_size_t positions_since_flush = 0;
    sz_u8_t transpose_classes[positions_per_transpose_k][candidates_per_position_k];
    for (sz_size_t transpose_start = 0, filled = positions_per_transpose_k; filled == positions_per_transpose_k;
         transpose_start += filled) {
        filled = sz_levenshtein_u8x64_transpose_icelake_(&local_query, texts, byte_counts, candidates_per_position_k,
                                                         cursors, STRINGZILLA_NULL, transpose_start,
                                                         positions_per_transpose_k, &transpose_classes[0][0]);
        // When scores must next be read, and whose, so a position costs one compare and retiring costs no test.
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(unread, byte_counts);
        for (sz_size_t position = 0; position != filled;) {
            sz_size_t const run_length = sz_min_of_two(
                sz_min_of_two(filled - position, positions_per_flush_k - positions_since_flush),
                deadline.position - (transpose_start + position));
            for (sz_size_t taken = 0; taken != run_length; ++taken, ++position)
                sz_levenshtein_u8x64_step_icelake_(
                    &state, &vertical, &packed, sz_levenshtein_u8x64_classes_u8_icelake_(transpose_classes[position]));
            positions_since_flush += run_length;
            sz_size_t const flushed_position = transpose_start + position;
            if (positions_since_flush != positions_per_flush_k && flushed_position != deadline.position) continue;
            sz_levenshtein_u8x64_flush_icelake_(&state, scores), positions_since_flush = 0;
            if (flushed_position == deadline.position) {
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)_tzcnt_u64(ending);
                    distances[candidate] = scores[candidate];
                }
                unread &= ~deadline.retiring;
                deadline = sz_levenshtein_deadline_(unread, byte_counts);
            }
            // Every flush makes the scores exact, where a bounded sweep tests for a freeze, and a
            // zero fill ends the transposes as when every candidate runs dry.
            if (radius < STRINGZILLA_SSIZE_MAX &&
                sz_levenshtein_flushed_any_active_icelake_(scores, byte_counts, candidates_per_position_k, unread,
                                                           flushed_position, radius) == sz_false_k) {
                filled = 0;
                break;
            }
        }
    }
    // Candidates as long as the sweep itself end at the position the transposes never reached, and
    // frozen ones read their running score.
    sz_levenshtein_u8x64_flush_icelake_(&state, scores);
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)_tzcnt_u64(unread);
        distances[candidate] = scores[candidate];
    }
}

/** Streams every candidate through a prepared byte @p query of at most eight symbols, sixty-four
 *  at a time in @p order, each distance under @p bound. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_distances_icelake_(sz_levenshtein_query_t const *query,
                                                                sz_strings_view_t const *candidates,
                                                                sz_size_t const *order, sz_size_t bound,
                                                                sz_size_t *distances) {
    enum { candidates_per_position_k = sz_levenshtein_u8x64_candidates_per_step_icelake_k };
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    for (sz_size_t next_rank = 0; next_rank != candidates->count;) {
        sz_cptr_t texts[candidates_per_position_k];
        sz_u64_t byte_counts[candidates_per_position_k], symbol_counts[candidates_per_position_k];
        sz_size_t indices[candidates_per_position_k], sweep_distances[candidates_per_position_k];
        sz_size_t const sweep_count = sz_levenshtein_sweep_seed_(candidates, STRINGZILLA_NULL, order, query, bound,
                                                                 &next_rank, candidates_per_position_k, indices, texts,
                                                                 byte_counts, symbol_counts, distances);
        if (!sweep_count) break;
        sz_levenshtein_u8x64_sweep_icelake_(query, texts, byte_counts, sweep_count, radius, sweep_distances);
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
            distances[indices[candidate]] = sz_levenshtein_bounded_(sweep_distances[candidate], bound);
    }
}

/** Thirty-two candidates' score movement since the last flush, one signed 16-bit integer
 *  for each candidate. */
typedef struct sz_levenshtein_u16x32_state_icelake_t {

    /** How far each candidate's score has moved since the last flush. */
    sz_u512_vec_t deltas_vec;
} sz_levenshtein_u16x32_state_icelake_t;

/** Thirty-two candidates' Myers state for a query of at most sixteen symbols - one 16-bit
 *  word per candidate. */
typedef struct sz_levenshtein_u16x32_vertical_icelake_t {

    /** Myers' VP per candidate, the query's symbols in the low bits. */
    sz_u512_vec_t positive_vec;

    /** Myers' VN per candidate, the query's symbols in the low bits. */
    sz_u512_vec_t negative_vec;
} sz_levenshtein_u16x32_vertical_icelake_t;

/** A query of at most sixteen symbols as the 16-bit lanes read it, so a step looks its masks up
 *  with @c VPERMW instead of a gather (since $\le 16$ byte symbols produce at most 17 classes). */
typedef struct sz_levenshtein_u16x32_query_icelake_t {

    /** Class @c c 's match mask at 16-bit lane @c c, every other lane zero. */
    sz_u512_vec_t masks_vec;

    /** The query's last symbol's bit, repeated in all thirty-two lanes. */
    sz_u512_vec_t last_symbol_bit_vec;
} sz_levenshtein_u16x32_query_icelake_t;

/** One ZMM per position, thirty-two candidates in it, and the vertical never leaves a register. */
enum {
    sz_levenshtein_u16x32_candidates_per_step_icelake_k = 32,
    sz_levenshtein_u16x32_registers_per_position_icelake_k = 1,
    sz_levenshtein_u16x32_positions_per_flush_icelake_k = 64
};

STRINGZILLA_INLINE void sz_levenshtein_u16x32_pack_icelake_(sz_levenshtein_query_t const *query,
                                                            sz_levenshtein_u16x32_query_icelake_t *packed) {
    sz_u512_vec_t masks_vec;
    masks_vec.zmm = _mm512_setzero_si512();
    for (sz_size_t class_id = 0; class_id != query->classes; ++class_id)
        masks_vec.u16s[class_id] = (sz_u16_t)query->masks[class_id];
    packed->masks_vec = masks_vec;
    packed->last_symbol_bit_vec.zmm = _mm512_set1_epi16((short)sz_levenshtein_last_symbol_bit_(query->length));
}

/** Packs a query of at most sixteen symbols into one 16-bit lane per class, the table
 *  @c VPERMW indexes. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_pack_icelake(sz_levenshtein_query_t const *query,
                                                           sz_levenshtein_u16x32_query_icelake_t *packed) {
    sz_levenshtein_u16x32_pack_icelake_(query, packed);
}

STRINGZILLA_INLINE void sz_levenshtein_u16x32_init_icelake_(sz_levenshtein_u16x32_state_icelake_t *state,
                                                            sz_levenshtein_u16x32_vertical_icelake_t *vertical) {
    state->deltas_vec.zmm = _mm512_setzero_si512();
    vertical->positive_vec.zmm = _mm512_set1_epi16((short)-1);
    vertical->negative_vec.zmm = _mm512_setzero_si512();
}

/** Starts thirty-two candidates: no score movement yet, and the vertical at the top boundary. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_init_icelake(sz_levenshtein_u16x32_state_icelake_t *state,
                                                           sz_levenshtein_u16x32_vertical_icelake_t *vertical) {
    sz_levenshtein_u16x32_init_icelake_(state, vertical);
}

STRINGZILLA_INLINE sz_u512_vec_t sz_levenshtein_u16x32_classes_u8_icelake_(sz_u8_t const *classes) {
    sz_u512_vec_t classes_vec;
    classes_vec.zmm = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i const *)classes));
    return classes_vec;
}

/** Thirty-two byte-wide class ids, widened to 16-bit indices for @c VPERMW. */
STRINGZILLA_INLINE sz_u512_vec_t sz_levenshtein_u16x32_classes_u8_icelake(sz_u8_t const *classes) {
    return sz_levenshtein_u16x32_classes_u8_icelake_(classes);
}

STRINGZILLA_INLINE void sz_levenshtein_u16x32_step_icelake_(sz_levenshtein_u16x32_state_icelake_t *state,
                                                            sz_levenshtein_u16x32_vertical_icelake_t *vertical,
                                                            sz_levenshtein_u16x32_query_icelake_t const *packed,
                                                            sz_u512_vec_t classes_vec) {
    enum { ternary_xor_or_k = 0xBE, ternary_or_nor_k = 0xF1 };
    __m512i const ones_u16x32 = _mm512_set1_epi16((short)-1);
    __m512i const equality_u16x32 = _mm512_permutexvar_epi16(classes_vec.zmm, packed->masks_vec.zmm);
    __m512i const vertical_carry_u16x32 = _mm512_or_si512(equality_u16x32, vertical->negative_vec.zmm);
    __m512i const sum_u16x32 = _mm512_add_epi16(_mm512_and_si512(equality_u16x32, vertical->positive_vec.zmm),
                                                vertical->positive_vec.zmm);
    __m512i const diagonal_u16x32 = _mm512_ternarylogic_epi64(sum_u16x32, vertical->positive_vec.zmm, equality_u16x32,
                                                              ternary_xor_or_k);
    __m512i const horizontal_positive_u16x32 = _mm512_ternarylogic_epi64(vertical->negative_vec.zmm, diagonal_u16x32,
                                                                         vertical->positive_vec.zmm, ternary_or_nor_k);
    __m512i const horizontal_negative_u16x32 = _mm512_and_si512(vertical->positive_vec.zmm, diagonal_u16x32);
    __mmask32 const rose_m32 = _mm512_test_epi16_mask(horizontal_positive_u16x32, packed->last_symbol_bit_vec.zmm);
    __mmask32 const fell_m32 = _mm512_test_epi16_mask(horizontal_negative_u16x32, packed->last_symbol_bit_vec.zmm);
    state->deltas_vec.zmm = _mm512_mask_sub_epi16(state->deltas_vec.zmm, rose_m32, state->deltas_vec.zmm, ones_u16x32);
    state->deltas_vec.zmm = _mm512_mask_add_epi16(state->deltas_vec.zmm, fell_m32, state->deltas_vec.zmm, ones_u16x32);
    __m512i const shifted_positive_u16x32 = _mm512_sub_epi16(
        _mm512_add_epi16(horizontal_positive_u16x32, horizontal_positive_u16x32), ones_u16x32);
    __m512i const shifted_negative_u16x32 = _mm512_add_epi16(horizontal_negative_u16x32, horizontal_negative_u16x32);
    vertical->positive_vec.zmm = _mm512_ternarylogic_epi64(shifted_negative_u16x32, vertical_carry_u16x32,
                                                           shifted_positive_u16x32, ternary_or_nor_k);
    vertical->negative_vec.zmm = _mm512_and_si512(shifted_positive_u16x32, vertical_carry_u16x32);
}

/** Advances thirty-two candidates one symbol through one 16-bit Myers word, the masks read by a
 *  single @c VPERMW permute. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_step_icelake(sz_levenshtein_u16x32_state_icelake_t *state,
                                                           sz_levenshtein_u16x32_vertical_icelake_t *vertical,
                                                           sz_levenshtein_u16x32_query_icelake_t const *packed,
                                                           sz_u512_vec_t classes_vec) {
    sz_levenshtein_u16x32_step_icelake_(state, vertical, packed, classes_vec);
}

STRINGZILLA_INLINE void sz_levenshtein_u16x32_flush_icelake_(sz_levenshtein_u16x32_state_icelake_t *state,
                                                             sz_size_t *scores) {
    enum { lanes_k = sz_levenshtein_u16x32_candidates_per_step_icelake_k };
    sz_u512_vec_t const deltas_vec = state->deltas_vec;
    for (sz_size_t candidate = 0; candidate != lanes_k; ++candidate)
        scores[candidate] = (sz_size_t)((sz_ssize_t)scores[candidate] + deltas_vec.i16s[candidate]);
    state->deltas_vec.zmm = _mm512_setzero_si512();
}

/** Folds the 16-bit lane deltas into @p scores and clears them. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_flush_icelake(sz_levenshtein_u16x32_state_icelake_t *state,
                                                            sz_size_t *scores) {
    sz_levenshtein_u16x32_flush_icelake_(state, scores);
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u16x32_transpose_icelake_(sz_levenshtein_query_t const *query,
                                                                      sz_cptr_t const *texts,
                                                                      sz_u64_t const *byte_counts, sz_size_t candidates,
                                                                      sz_size_t *cursors, sz_u64_t *symbol_counts,
                                                                      sz_size_t transpose_start, sz_size_t positions,
                                                                      void *transpose_classes) {
    enum { lanes_k = sz_levenshtein_u16x32_candidates_per_step_icelake_k };
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u8_t *const classes = (sz_u8_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != lanes_k; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    for (sz_size_t candidate = 0; candidate != lanes_k; ++candidate) {
        sz_size_t const taken = sz_min_of_two(filled, (sz_size_t)byte_counts[candidate] - cursors[candidate]);
        for (sz_size_t position = 0; position != taken; ++position)
            classes[position * lanes_k + candidate] = (sz_u8_t)texts[candidate][cursors[candidate] + position];
        for (sz_size_t position = taken; position != filled; ++position) classes[position * lanes_k + candidate] = 0;
        cursors[candidate] += taken;
    }
    __m512i const table_first_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 0));
    __m512i const table_second_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 64));
    __m512i const table_third_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 128));
    __m512i const table_fourth_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 192));
    sz_size_t staged = 0;
    for (; staged + 64 <= filled * lanes_k; staged += 64) {
        __m512i const bytes_u8x64 = _mm512_loadu_si512((void const *)(classes + staged));
        __m512i const low_half_u8x64 = _mm512_permutex2var_epi8(table_first_quarter_u8x64, bytes_u8x64,
                                                                table_second_quarter_u8x64);
        __m512i const high_half_u8x64 = _mm512_permutex2var_epi8(table_third_quarter_u8x64, bytes_u8x64,
                                                                 table_fourth_quarter_u8x64);
        _mm512_storeu_si512((void *)(classes + staged),
                            _mm512_mask_blend_epi8(_mm512_movepi8_mask(bytes_u8x64), low_half_u8x64, high_half_u8x64));
    }
    for (; staged != filled * lanes_k; ++staged) classes[staged] = byte_to_class[classes[staged]];
    return filled;
}

/** The byte transpose for thirty-two candidates: position @c p of candidate @c c lands at
 *  @c p * 32 + c. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u16x32_transpose_icelake(sz_levenshtein_query_t const *query,
                                                                     sz_cptr_t const *texts,
                                                                     sz_u64_t const *byte_counts, sz_size_t candidates,
                                                                     sz_size_t *cursors, sz_u64_t *symbol_counts,
                                                                     sz_size_t transpose_start, sz_size_t positions,
                                                                     void *transpose_classes) {
    return sz_levenshtein_u16x32_transpose_icelake_(query, texts, byte_counts, candidates, cursors, symbol_counts,
                                                    transpose_start, positions, transpose_classes);
}

/** Sweeps thirty-two candidates of a query of 9..16 symbols through every transpose. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_sweep_icelake_(sz_levenshtein_query_t const *shared_query,
                                                             sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                             sz_size_t sweep_count, sz_ssize_t radius,
                                                             sz_size_t *distances) {
    enum {
        candidates_per_position_k = sz_levenshtein_u16x32_candidates_per_step_icelake_k,
        positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k,
        positions_per_flush_k = sz_levenshtein_u16x32_positions_per_flush_icelake_k
    };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_u16x32_query_icelake_t packed;
    sz_levenshtein_u16x32_pack_icelake_(&local_query, &packed);
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_size_t scores[candidates_per_position_k];
    for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
        scores[candidate] = local_query.length;
    sz_u64_t unread = ((sz_u64_t)1 << sweep_count) - (sz_u64_t)1;
    sz_levenshtein_u16x32_state_icelake_t state;
    sz_levenshtein_u16x32_vertical_icelake_t vertical;
    sz_levenshtein_u16x32_init_icelake_(&state, &vertical);
    sz_size_t positions_since_flush = 0;
    sz_u8_t transpose_classes[positions_per_transpose_k][candidates_per_position_k];
    for (sz_size_t transpose_start = 0, filled = positions_per_transpose_k; filled == positions_per_transpose_k;
         transpose_start += filled) {
        filled = sz_levenshtein_u16x32_transpose_icelake_(&local_query, texts, byte_counts, candidates_per_position_k,
                                                          cursors, STRINGZILLA_NULL, transpose_start,
                                                          positions_per_transpose_k, &transpose_classes[0][0]);
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(unread, byte_counts);
        for (sz_size_t position = 0; position != filled;) {
            sz_size_t const run_length = sz_min_of_two(
                sz_min_of_two(filled - position, positions_per_flush_k - positions_since_flush),
                deadline.position - (transpose_start + position));
            for (sz_size_t taken = 0; taken != run_length; ++taken, ++position)
                sz_levenshtein_u16x32_step_icelake_(
                    &state, &vertical, &packed, sz_levenshtein_u16x32_classes_u8_icelake_(transpose_classes[position]));
            positions_since_flush += run_length;
            sz_size_t const flushed_position = transpose_start + position;
            if (positions_since_flush != positions_per_flush_k && flushed_position != deadline.position) continue;
            sz_levenshtein_u16x32_flush_icelake_(&state, scores), positions_since_flush = 0;
            if (flushed_position == deadline.position) {
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)_tzcnt_u64(ending);
                    distances[candidate] = scores[candidate];
                }
                unread &= ~deadline.retiring;
                deadline = sz_levenshtein_deadline_(unread, byte_counts);
            }
            // Every flush makes the scores exact, where a bounded sweep tests for a freeze, and a
            // zero fill ends the transposes as when every candidate runs dry.
            if (radius < STRINGZILLA_SSIZE_MAX &&
                sz_levenshtein_flushed_any_active_icelake_(scores, byte_counts, candidates_per_position_k, unread,
                                                           flushed_position, radius) == sz_false_k) {
                filled = 0;
                break;
            }
        }
    }
    // Candidates as long as the sweep itself end at the position the transposes never reached, and
    // frozen ones read their running score.
    sz_levenshtein_u16x32_flush_icelake_(&state, scores);
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)_tzcnt_u64(unread);
        distances[candidate] = scores[candidate];
    }
}

/** Streams every candidate through a prepared byte @p query of 9..16 symbols, thirty-two
 *  at a time in @p order, each distance under @p bound. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_distances_icelake_(sz_levenshtein_query_t const *query,
                                                                 sz_strings_view_t const *candidates,
                                                                 sz_size_t const *order, sz_size_t bound,
                                                                 sz_size_t *distances) {
    enum { candidates_per_position_k = sz_levenshtein_u16x32_candidates_per_step_icelake_k };
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    for (sz_size_t next_rank = 0; next_rank != candidates->count;) {
        sz_cptr_t texts[candidates_per_position_k];
        sz_u64_t byte_counts[candidates_per_position_k], symbol_counts[candidates_per_position_k];
        sz_size_t indices[candidates_per_position_k], sweep_distances[candidates_per_position_k];
        sz_size_t const sweep_count = sz_levenshtein_sweep_seed_(candidates, STRINGZILLA_NULL, order, query, bound,
                                                                 &next_rank, candidates_per_position_k, indices, texts,
                                                                 byte_counts, symbol_counts, distances);
        if (!sweep_count) break;
        sz_levenshtein_u16x32_sweep_icelake_(query, texts, byte_counts, sweep_count, radius, sweep_distances);
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
            distances[indices[candidate]] = sz_levenshtein_bounded_(sweep_distances[candidate], bound);
    }
}

/** The lanes among the first @p lanes whose floor, their flushed @p scores less the symbols left
 *  before their @p ends, has passed @p radius at @p position. */
STRINGZILLA_INLINE sz_u64_t sz_levenshtein_flushed_frozen_icelake_(sz_size_t const *scores, sz_u64_t const *ends,
                                                                   sz_size_t lanes, sz_size_t position,
                                                                   sz_ssize_t radius) {
    __m512i const position_u64x8 = _mm512_set1_epi64((long long)position);
    __m512i const radius_i64x8 = _mm512_set1_epi64((long long)radius);
    sz_u64_t frozen = 0;
    for (sz_size_t first = 0; first != lanes; first += 8) {
        __m512i const scores_u64x8 = _mm512_loadu_si512((void const *)(scores + first));
        __m512i const ends_u64x8 = _mm512_loadu_si512((void const *)(ends + first));
        __m512i const floor_i64x8 = _mm512_sub_epi64(_mm512_add_epi64(scores_u64x8, position_u64x8), ends_u64x8);
        frozen |= (sz_u64_t)_mm512_cmpgt_epi64_mask(floor_i64x8, radius_i64x8) << first;
    }
    return frozen;
}

/** Streams every candidate through a prepared byte @p query in sixty-four recycling lanes: a
 *  lane whose candidate ends or freezes past @p bound is reported and, at the next freeze test,
 *  takes the next candidate in @p order. Every freeze test flushes, so the deltas never wrap. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_recycled_icelake_(sz_levenshtein_query_t const *shared_query,
                                                               sz_strings_view_t const *candidates,
                                                               sz_size_t const *order, sz_size_t bound,
                                                               sz_size_t *distances) {
    enum { lanes_k = 64, block_k = sz_levenshtein_freeze_positions_k };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_u8x64_query_icelake_t packed;
    sz_levenshtein_u8x64_pack_icelake_(&local_query, &packed);
    sz_cptr_t texts[lanes_k] = {STRINGZILLA_NULL};
    sz_u64_t byte_counts[lanes_k] = {0}, ends[lanes_k] = {0};
    sz_size_t cursors[lanes_k] = {0}, slots[lanes_k] = {0}, scores[lanes_k] = {0};
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_u8_t transpose_classes[block_k][lanes_k];
    sz_levenshtein_u8x64_state_icelake_t state;
    sz_levenshtein_u8x64_vertical_icelake_t vertical;
    sz_levenshtein_u8x64_init_icelake_(&state, &vertical);
    sz_u64_t live = 0, idle = sz_u64_mask_until_(lanes_k);
    sz_size_t next_rank = 0;
    for (sz_size_t block_start = 0;; block_start += block_k) {
        // Idle lanes take the next candidates in order from this block on, with no deltas pending.
        if (idle && next_rank != candidates->count) {
            sz_cptr_t fresh_texts[lanes_k];
            sz_u64_t fresh_bytes[lanes_k], fresh_symbols[lanes_k];
            sz_size_t fresh_slots[lanes_k];
            sz_size_t const fresh_count = sz_levenshtein_sweep_seed_(
                candidates, STRINGZILLA_NULL, order, &local_query, bound, &next_rank, (sz_size_t)_mm_popcnt_u64(idle),
                fresh_slots, fresh_texts, fresh_bytes, fresh_symbols, distances);
            sz_u64_t fresh = 0;
            for (sz_size_t taken = 0; taken != fresh_count; ++taken) {
                sz_size_t const lane = (sz_size_t)_tzcnt_u64(idle);
                idle &= idle - 1, fresh |= (sz_u64_t)1 << lane;
                texts[lane] = fresh_texts[taken], byte_counts[lane] = fresh_bytes[taken], cursors[lane] = 0;
                slots[lane] = fresh_slots[taken], ends[lane] = block_start + fresh_symbols[taken];
                scores[lane] = local_query.length;
            }
            vertical.positive_vec.zmm = _mm512_mask_set1_epi8(vertical.positive_vec.zmm, (__mmask64)fresh, -1);
            vertical.negative_vec.zmm = _mm512_maskz_mov_epi8((__mmask64)~fresh, vertical.negative_vec.zmm);
            live |= fresh;
        }
        if (!live) break;
        for (sz_u64_t retired = idle; retired; retired &= retired - 1) {
            sz_size_t const lane = (sz_size_t)_tzcnt_u64(retired);
            texts[lane] = STRINGZILLA_NULL, byte_counts[lane] = 0, cursors[lane] = 0;
        }
        sz_levenshtein_u8x64_transpose_icelake_(&local_query, texts, byte_counts, lanes_k, cursors, STRINGZILLA_NULL,
                                                block_start, block_k, &transpose_classes[0][0]);
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(live, ends);
        for (sz_size_t position = 0;;) {
            sz_size_t const run_end = sz_min_of_two(deadline.position - block_start, (sz_size_t)block_k);
            for (; position < run_end; ++position)
                sz_levenshtein_u8x64_step_icelake_(
                    &state, &vertical, &packed, sz_levenshtein_u8x64_classes_u8_icelake_(transpose_classes[position]));
            if (deadline.position != block_start + position) break;
            sz_levenshtein_u8x64_flush_icelake_(&state, scores);
            for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                sz_size_t const lane = (sz_size_t)_tzcnt_u64(ending);
                distances[slots[lane]] = sz_levenshtein_bounded_(scores[lane], bound);
            }
            live &= ~deadline.retiring, idle |= deadline.retiring;
            if (!live) break;
            deadline = sz_levenshtein_deadline_(live, ends);
        }
        // At every block's end a lane already past the bound reports it and frees its lane.
        sz_levenshtein_u8x64_flush_icelake_(&state, scores);
        sz_u64_t const frozen = live & sz_levenshtein_flushed_frozen_icelake_(scores, ends, lanes_k,
                                                                              block_start + block_k, radius);
        for (sz_u64_t ending = frozen; ending; ending &= ending - 1)
            distances[slots[(sz_size_t)_tzcnt_u64(ending)]] = bound + 1;
        live &= ~frozen, idle |= frozen;
    }
}

/** Streams every candidate through a prepared byte @p query in thirty-two recycling lanes: a
 *  lane whose candidate ends or freezes past @p bound is reported and, at the next freeze test,
 *  takes the next candidate in @p order. Every freeze test flushes, so the deltas never wrap. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_recycled_icelake_(sz_levenshtein_query_t const *shared_query,
                                                                sz_strings_view_t const *candidates,
                                                                sz_size_t const *order, sz_size_t bound,
                                                                sz_size_t *distances) {
    enum { lanes_k = 32, block_k = sz_levenshtein_freeze_positions_k };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_u16x32_query_icelake_t packed;
    sz_levenshtein_u16x32_pack_icelake_(&local_query, &packed);
    sz_cptr_t texts[lanes_k] = {STRINGZILLA_NULL};
    sz_u64_t byte_counts[lanes_k] = {0}, ends[lanes_k] = {0};
    sz_size_t cursors[lanes_k] = {0}, slots[lanes_k] = {0}, scores[lanes_k] = {0};
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_u8_t transpose_classes[block_k][lanes_k];
    sz_levenshtein_u16x32_state_icelake_t state;
    sz_levenshtein_u16x32_vertical_icelake_t vertical;
    sz_levenshtein_u16x32_init_icelake_(&state, &vertical);
    sz_u64_t live = 0, idle = sz_u64_mask_until_(lanes_k);
    sz_size_t next_rank = 0;
    for (sz_size_t block_start = 0;; block_start += block_k) {
        // Idle lanes take the next candidates in order from this block on, with no deltas pending.
        if (idle && next_rank != candidates->count) {
            sz_cptr_t fresh_texts[lanes_k];
            sz_u64_t fresh_bytes[lanes_k], fresh_symbols[lanes_k];
            sz_size_t fresh_slots[lanes_k];
            sz_size_t const fresh_count = sz_levenshtein_sweep_seed_(
                candidates, STRINGZILLA_NULL, order, &local_query, bound, &next_rank, (sz_size_t)_mm_popcnt_u64(idle),
                fresh_slots, fresh_texts, fresh_bytes, fresh_symbols, distances);
            sz_u64_t fresh = 0;
            for (sz_size_t taken = 0; taken != fresh_count; ++taken) {
                sz_size_t const lane = (sz_size_t)_tzcnt_u64(idle);
                idle &= idle - 1, fresh |= (sz_u64_t)1 << lane;
                texts[lane] = fresh_texts[taken], byte_counts[lane] = fresh_bytes[taken], cursors[lane] = 0;
                slots[lane] = fresh_slots[taken], ends[lane] = block_start + fresh_symbols[taken];
                scores[lane] = local_query.length;
            }
            vertical.positive_vec.zmm = _mm512_mask_set1_epi16(vertical.positive_vec.zmm, (__mmask32)fresh, -1);
            vertical.negative_vec.zmm = _mm512_maskz_mov_epi16((__mmask32)~fresh, vertical.negative_vec.zmm);
            live |= fresh;
        }
        if (!live) break;
        for (sz_u64_t retired = idle; retired; retired &= retired - 1) {
            sz_size_t const lane = (sz_size_t)_tzcnt_u64(retired);
            texts[lane] = STRINGZILLA_NULL, byte_counts[lane] = 0, cursors[lane] = 0;
        }
        sz_levenshtein_u16x32_transpose_icelake_(&local_query, texts, byte_counts, lanes_k, cursors, STRINGZILLA_NULL,
                                                 block_start, block_k, &transpose_classes[0][0]);
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(live, ends);
        for (sz_size_t position = 0;;) {
            sz_size_t const run_end = sz_min_of_two(deadline.position - block_start, (sz_size_t)block_k);
            for (; position < run_end; ++position)
                sz_levenshtein_u16x32_step_icelake_(
                    &state, &vertical, &packed, sz_levenshtein_u16x32_classes_u8_icelake_(transpose_classes[position]));
            if (deadline.position != block_start + position) break;
            sz_levenshtein_u16x32_flush_icelake_(&state, scores);
            for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                sz_size_t const lane = (sz_size_t)_tzcnt_u64(ending);
                distances[slots[lane]] = sz_levenshtein_bounded_(scores[lane], bound);
            }
            live &= ~deadline.retiring, idle |= deadline.retiring;
            if (!live) break;
            deadline = sz_levenshtein_deadline_(live, ends);
        }
        // At every block's end a lane already past the bound reports it and frees its lane.
        sz_levenshtein_u16x32_flush_icelake_(&state, scores);
        sz_u64_t const frozen = live & sz_levenshtein_flushed_frozen_icelake_(scores, ends, lanes_k,
                                                                              block_start + block_k, radius);
        for (sz_u64_t ending = frozen; ending; ending &= ending - 1)
            distances[slots[(sz_size_t)_tzcnt_u64(ending)]] = bound + 1;
        live &= ~frozen, idle |= frozen;
    }
}

/** How many symbols of a query one Myers lane holds, which is how many candidates
 *  a step advances. */
typedef enum sz_levenshtein_lanes_icelake_t {

    /** Byte lanes: sixty-four candidates a step, up to an eight-symbol query. */
    sz_levenshtein_lanes_u8x64_k,

    /** Halfword lanes: thirty-two candidates a step, up to a sixteen-symbol query. */
    sz_levenshtein_lanes_u16x32_k,

    /** Word lanes: eight candidates a step, at any query length. */
    sz_levenshtein_lanes_u64x8_k,
} sz_levenshtein_lanes_icelake_t;

/** The narrowest lanes @p length symbols fit, so the shortest queries advance the most
 *  candidates per step. */
STRINGZILLA_CONSTEXPR sz_levenshtein_lanes_icelake_t sz_levenshtein_lanes_icelake(sz_size_t length) {
    return length <= 8 ? sz_levenshtein_lanes_u8x64_k
                       : (length <= 16 ? sz_levenshtein_lanes_u16x32_k : sz_levenshtein_lanes_u64x8_k);
}

#pragma endregion Narrow Lanes

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_icelake(sz_levenshtein_engine_t *engine,
                                                               sz_strings_view_t const *queries,
                                                               sz_levenshtein_symbol_t symbol,
                                                               sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_levenshtein_engine_init_cpu_(engine, queries, symbol, sz_cap_icelake_k, allocator, stream);
}

/** Scores every candidate of a round against @p engine 's byte query @p index in the narrowest
 *  lanes its length fits. */
STRINGZILLA_INLINE void sz_levenshtein_run_icelake_(sz_levenshtein_engine_t const *engine, sz_size_t index,
                                                    sz_strings_view_t const *scored, sz_size_t const *symbol_counts,
                                                    sz_size_t const *order, sz_size_t bound,
                                                    sz_levenshtein_u64x8_vertical_skylake_t *verticals,
                                                    sz_size_t *distances) {
    sz_levenshtein_lanes_icelake_t const lanes = sz_levenshtein_lanes_icelake(engine->lengths[index]);
    if (engine->lengths[index] == 0 || lanes == sz_levenshtein_lanes_u64x8_k) {
        sz_levenshtein_run_skylake_(engine, index, scored, symbol_counts, order, bound, verticals, distances);
        return;
    }
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
    // A bounded run recycles lanes as their candidates pass the bound, as on Skylake.
    if (lanes == sz_levenshtein_lanes_u8x64_k && bound != STRINGZILLA_SIZE_MAX)
        sz_levenshtein_u8x64_recycled_icelake_(&query, scored, order, bound, distances);
    else if (lanes == sz_levenshtein_lanes_u8x64_k)
        sz_levenshtein_u8x64_distances_icelake_(&query, scored, order, bound, distances);
    else if (bound != STRINGZILLA_SIZE_MAX)
        sz_levenshtein_u16x32_recycled_icelake_(&query, scored, order, bound, distances);
    else sz_levenshtein_u16x32_distances_icelake_(&query, scored, order, bound, distances);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_icelake(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *candidates,
                                                             sz_size_t const *symbol_counts, sz_size_t const *order,
                                                             sz_size_t bound, sz_size_t *distances,
                                                             sz_size_t distances_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_assert_((engine->capability & sz_cap_cpus_k) != 0 &&
               "A host tier never scores a device-prepared engine, whose head only its GPU tier reads");
    // The narrow lanes read one word of one class row, so a rune batch is Skylake's whole and not
    // one query at a time.
    if (engine->symbol != sz_levenshtein_bytes_k)
        return sz_levenshtein_distances_skylake_(engine, candidates, symbol_counts, order, bound, distances,
                                                 distances_stride, sz_utf8_decode_icelake_);
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    // Queries up to 16 symbols use 8-bit or 16-bit lanes; wider ones keep the Skylake verticals.
    sz_strings_view_t scored;
    sz_size_t const *candidate_order;
    sz_status_t const prepared = sz_levenshtein_engine_round_(
        engine, candidates, order, sz_levenshtein_verticals_bytes_skylake_(engine), sz_utf8_decode_icelake_, &scored,
        &symbol_counts, &candidate_order);
    if (prepared != sz_success_k) return prepared;
    sz_levenshtein_u64x8_vertical_skylake_t *const verticals =
        (sz_levenshtein_u64x8_vertical_skylake_t *)sz_levenshtein_engine_verticals_(engine);
    for (sz_size_t index = 0; index != engine->count; ++index)
        sz_levenshtein_run_icelake_(engine, index, &scored, symbol_counts, candidate_order, bound, verticals,
                                    distances + index * distances_stride);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#pragma endregion Ice Lake Implementation
#endif // STRINGZILLA_TARGET_ICELAKE
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_LEVENSHTEIN_ICELAKE_H_
