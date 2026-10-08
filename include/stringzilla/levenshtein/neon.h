/**
 *  @file include/stringzilla/levenshtein/neon.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief NEON bit-vector edit distances over bytes and UTF-8 runes.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_NEON_H_
#define STRINGZILLA_LEVENSHTEIN_NEON_H_
#include "stringzilla/types.h"
#include "stringzilla/levenshtein/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEON_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

typedef struct sz_levenshtein_u64x2_state_neon_t {
    sz_u128_vec_t scores_vec;
} sz_levenshtein_u64x2_state_neon_t;
typedef struct sz_levenshtein_u64x2_vertical_neon_t {
    sz_u128_vec_t positive_vec, negative_vec;
} sz_levenshtein_u64x2_vertical_neon_t;

STRINGZILLA_INLINE void sz_levenshtein_u64x2_init_neon_(sz_levenshtein_u64x2_state_neon_t *state,
                                                        sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                        sz_size_t words, sz_levenshtein_query_t const *query) {
    state->scores_vec.u64x2 = vdupq_n_u64(query->length);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.u64x2 = vdupq_n_u64(~(sz_u64_t)0);
        verticals[word].negative_vec.u64x2 = vdupq_n_u64(0);
    }
}

STRINGZILLA_INLINE void sz_levenshtein_u64x2_steps_neon_(sz_levenshtein_u64x2_state_neon_t *states,
                                                         sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                         sz_size_t words, sz_levenshtein_query_t const *query,
                                                         sz_u64_t const *const *rows, sz_size_t registers,
                                                         sz_u64_t active) {
    uint64x2_t positive_carry_u64x2[4], negative_carry_u64x2[4];
    for (sz_size_t reg = 0; reg != registers; ++reg) {
        positive_carry_u64x2[reg] = vdupq_n_u64(1);
        negative_carry_u64x2[reg] = vdupq_n_u64(0);
    }
    uint64x2_t const last_symbol_bit_u64x2 = vdupq_n_u64(sz_levenshtein_last_symbol_bit_(query->length));
    int64x2_t const last_symbol_shift_i64x2 = vdupq_n_s64(-(sz_i64_t)sz_levenshtein_last_symbol_shift_(query->length));
    for (sz_size_t word = 0; word != words; ++word) {
        for (sz_size_t reg = 0; reg != registers; ++reg) {
            if (!(active & ((sz_u64_t)3 << (reg * 2)))) continue;
            sz_levenshtein_u64x2_state_neon_t *state = states + reg;
            sz_levenshtein_u64x2_vertical_neon_t *vertical = verticals + reg * words + word;
            sz_u64_t const masks[2] = {rows[reg * 2][word], rows[reg * 2 + 1][word]};
            uint64x2_t const equality_u64x2 = vld1q_u64(masks);
            uint64x2_t const vertical_carry_u64x2 = vorrq_u64(equality_u64x2, vertical->negative_vec.u64x2);
            uint64x2_t const matched_u64x2 = vorrq_u64(equality_u64x2, negative_carry_u64x2[reg]);
            uint64x2_t const sum_u64x2 = vaddq_u64(vandq_u64(matched_u64x2, vertical->positive_vec.u64x2),
                                                   vertical->positive_vec.u64x2);
            uint64x2_t const diagonal_u64x2 = vorrq_u64(veorq_u64(sum_u64x2, vertical->positive_vec.u64x2),
                                                        matched_u64x2);
            uint64x2_t horizontal_positive_u64x2 = vorrq_u64(
                vertical->negative_vec.u64x2, vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(
                                                  vorrq_u64(diagonal_u64x2, vertical->positive_vec.u64x2)))));
            uint64x2_t horizontal_negative_u64x2 = vandq_u64(vertical->positive_vec.u64x2, diagonal_u64x2);
            if (word + 1 == words) {
                state->scores_vec.u64x2 = vaddq_u64(
                    state->scores_vec.u64x2,
                    vshlq_u64(vandq_u64(horizontal_positive_u64x2, last_symbol_bit_u64x2), last_symbol_shift_i64x2));
                state->scores_vec.u64x2 = vsubq_u64(
                    state->scores_vec.u64x2,
                    vshlq_u64(vandq_u64(horizontal_negative_u64x2, last_symbol_bit_u64x2), last_symbol_shift_i64x2));
            }
            uint64x2_t const next_positive_u64x2 = vshrq_n_u64(horizontal_positive_u64x2, 63),
                             next_negative_u64x2 = vshrq_n_u64(horizontal_negative_u64x2, 63);
            horizontal_positive_u64x2 = vorrq_u64(vshlq_n_u64(horizontal_positive_u64x2, 1), positive_carry_u64x2[reg]);
            horizontal_negative_u64x2 = vorrq_u64(vshlq_n_u64(horizontal_negative_u64x2, 1), negative_carry_u64x2[reg]);
            positive_carry_u64x2[reg] = next_positive_u64x2, negative_carry_u64x2[reg] = next_negative_u64x2;
            vertical->positive_vec.u64x2 = vorrq_u64(horizontal_negative_u64x2,
                                                     vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(
                                                         vorrq_u64(vertical_carry_u64x2, horizontal_positive_u64x2)))));
            vertical->negative_vec.u64x2 = vandq_u64(horizontal_positive_u64x2, vertical_carry_u64x2);
        }
    }
}

STRINGZILLA_INLINE void sz_levenshtein_u64x2_step_neon_(sz_levenshtein_u64x2_state_neon_t *state,
                                                        sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                        sz_size_t words, sz_levenshtein_query_t const *query,
                                                        sz_u128_vec_t classes_vec) {
    sz_u64_t const *rows[2] = {query->masks + vgetq_lane_u64(classes_vec.u64x2, 0) * query->stride,
                               query->masks + vgetq_lane_u64(classes_vec.u64x2, 1) * query->stride};
    sz_levenshtein_u64x2_steps_neon_(state, verticals, words, query, rows, 1, 3);
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x2_score_neon_(sz_levenshtein_u64x2_state_neon_t const *state,
                                                              sz_size_t candidate) {
    sz_u64_t scores[2];
    vst1q_u64(scores, state->scores_vec.u64x2);
    return scores[candidate];
}

/** Initializes two candidates' Myers states for a prepared query. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_init_neon(sz_levenshtein_u64x2_state_neon_t *state,
                                                       sz_levenshtein_u64x2_vertical_neon_t *verticals, sz_size_t words,
                                                       sz_levenshtein_query_t const *query) {
    sz_levenshtein_u64x2_init_neon_(state, verticals, words, query);
}

/** Advances two candidates by their query class IDs. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_step_neon(sz_levenshtein_u64x2_state_neon_t *state,
                                                       sz_levenshtein_u64x2_vertical_neon_t *verticals, sz_size_t words,
                                                       sz_levenshtein_query_t const *query, sz_u128_vec_t classes_vec) {
    sz_levenshtein_u64x2_step_neon_(state, verticals, words, query, classes_vec);
}

/** Tests whether either unfinished candidate can still reach the radius. */
STRINGZILLA_INLINE sz_bool_t sz_levenshtein_u64x2_any_active_neon(sz_levenshtein_u64x2_state_neon_t const *state,
                                                                  sz_u128_vec_t symbol_counts_vec, sz_size_t position,
                                                                  sz_ssize_t radius) {
    uint64x2_t const positions_u64x2 = vdupq_n_u64(position);
    int64x2_t const floor_i64x2 = vreinterpretq_s64_u64(
        vsubq_u64(vaddq_u64(state->scores_vec.u64x2, positions_u64x2), symbol_counts_vec.u64x2));
    uint64x2_t const active_u64x2 = vandq_u64(vcleq_s64(floor_i64x2, vdupq_n_s64(radius)),
                                              vcgtq_u64(symbol_counts_vec.u64x2, positions_u64x2));
    return (vgetq_lane_u64(active_u64x2, 0) | vgetq_lane_u64(active_u64x2, 1)) != 0 ? sz_true_k : sz_false_k;
}

/** Reads a candidate's running score at its text's end. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x2_score_neon(sz_levenshtein_u64x2_state_neon_t const *state,
                                                             sz_size_t candidate) {
    return sz_levenshtein_u64x2_score_neon_(state, candidate);
}

enum { sz_levenshtein_u64x2_candidates_per_step_neon_k = 2, sz_levenshtein_u64x2_registers_per_position_neon_k = 4 };

STRINGZILLA_INLINE void sz_levenshtein_u64x2_sweep_neon_(sz_levenshtein_query_t const *shared_query,
                                                         sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                         sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                         sz_levenshtein_transpose_t_ transpose,
                                                         sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                         sz_size_t words, sz_size_t *distances) {
    enum {
        registers_k = sz_levenshtein_u64x2_registers_per_position_neon_k,
        candidates_per_position_k = 2 * registers_k,
        positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k
    };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_u64_t unread = ((sz_u64_t)1 << sweep_count) - 1;
    sz_levenshtein_u64x2_state_neon_t states[registers_k];
    for (sz_size_t reg = 0; reg != registers_k; ++reg)
        sz_levenshtein_u64x2_init_neon_(&states[reg], verticals + reg * words, words, query);
    sz_u32_t transpose_classes[positions_per_transpose_k][candidates_per_position_k];
    for (sz_size_t transpose_start = 0, filled = positions_per_transpose_k; filled == positions_per_transpose_k;
         transpose_start += filled) {
        filled = transpose(query, texts, byte_counts, candidates_per_position_k, cursors, symbol_counts,
                           transpose_start, positions_per_transpose_k, &transpose_classes[0][0]);
        sz_u64_t counts[candidates_per_position_k];
        for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
            counts[candidate] = symbol_counts[candidate];
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(unread, counts);
        for (sz_size_t position = 0; position != filled; ++position) {
            if (transpose_start + position == deadline.position) {
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)sz_u64_ctz_neon_(ending);
                    distances[candidate] = sz_levenshtein_u64x2_score_neon_(&states[candidate / 2], candidate % 2);
                }
                unread &= ~deadline.retiring;
                deadline = sz_levenshtein_deadline_(unread, counts);
            }
            sz_u64_t const *rows[candidates_per_position_k];
            for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
                rows[candidate] = query->masks + transpose_classes[position][candidate] * query->stride;
            if (words <= 2) {
                for (sz_size_t reg = 0; reg != registers_k; ++reg) {
                    if (!(unread & ((sz_u64_t)3 << (reg * 2)))) continue;
                    sz_levenshtein_u64x2_steps_neon_(states + reg, verticals + reg * words, words, query,
                                                     rows + reg * 2, 1, 3);
                }
            }
            else if (sweep_count <= 2) sz_levenshtein_u64x2_steps_neon_(states, verticals, words, query, rows, 1, 3);
            else sz_levenshtein_u64x2_steps_neon_(states, verticals, words, query, rows, registers_k, unread);
        }
    }
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)sz_u64_ctz_neon_(unread);
        distances[candidate] = sz_levenshtein_u64x2_score_neon_(&states[candidate / 2], candidate % 2);
    }
}

STRINGZILLA_INLINE void sz_levenshtein_u64x2_distances_neon_(sz_levenshtein_query_t const *query,
                                                             sz_sequence_t const *candidates,
                                                             sz_levenshtein_transpose_t_ transpose,
                                                             sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                             sz_size_t *distances) {
    enum { candidates_per_position_k = 2 * sz_levenshtein_u64x2_registers_per_position_neon_k };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_levenshtein_u64x2_vertical_neon_t resident_verticals[sz_levenshtein_u64x2_registers_per_position_neon_k * 2];
    for (sz_size_t sweep_first = 0; sweep_first < candidates->count; sweep_first += candidates_per_position_k) {
        sz_size_t const sweep_count = sz_min_of_two((sz_size_t)candidates_per_position_k,
                                                    candidates->count - sweep_first);
        sz_cptr_t texts[candidates_per_position_k] = {0};
        sz_u64_t byte_counts[candidates_per_position_k] = {0}, symbol_counts[candidates_per_position_k] = {0};
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate) {
            texts[candidate] = candidates->get_start(candidates->handle, sweep_first + candidate);
            byte_counts[candidate] = candidates->get_length(candidates->handle, sweep_first + candidate);
            symbol_counts[candidate] = byte_counts[candidate];
        }
        if (words == 1)
            sz_levenshtein_u64x2_sweep_neon_(query, texts, byte_counts, symbol_counts, sweep_count, transpose,
                                             resident_verticals, 1, distances + sweep_first);
        else if (words == 2)
            sz_levenshtein_u64x2_sweep_neon_(query, texts, byte_counts, symbol_counts, sweep_count, transpose,
                                             resident_verticals, 2, distances + sweep_first);
        else
            sz_levenshtein_u64x2_sweep_neon_(query, texts, byte_counts, symbol_counts, sweep_count, transpose,
                                             verticals, words, distances + sweep_first);
    }
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_neon(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                            sz_stream_t stream) {
    return sz_levenshtein_engine_init_cpu_(engine, queries, symbol, sz_cap_neon_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_neon(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_assert_((engine->capability & sz_cap_cpus_k) != 0 &&
               "A host tier never scores a device-prepared engine, whose head only its GPU tier reads");
    enum { registers_k = sz_levenshtein_u64x2_registers_per_position_neon_k };
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    sz_bool_t const over_bytes = engine->symbol == sz_levenshtein_bytes_k ? sz_true_k : sz_false_k;
    sz_levenshtein_transpose_t_ const transpose = over_bytes ? sz_levenshtein_transpose_
                                                             : sz_levenshtein_transpose_utf8_;
    sz_status_t const grown = sz_levenshtein_engine_scratch_(
        engine, sz_levenshtein_engine_verticals_bytes_(registers_k, sz_levenshtein_engine_words_max_(engine),
                                                       sizeof(sz_levenshtein_u64x2_vertical_neon_t)));
    if (grown != sz_success_k) return grown;
    sz_levenshtein_u64x2_vertical_neon_t *const verticals =
        (sz_levenshtein_u64x2_vertical_neon_t *)sz_levenshtein_engine_verticals_(engine);

    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t *const row = distances + index * distances_stride;
        if (engine->lengths[index] == 0) {
            sz_levenshtein_engine_empty_row_(engine, candidates, row);
            continue;
        }
        sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
        sz_levenshtein_u64x2_distances_neon_(&query, candidates, transpose, verticals, row);
    }
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_NEON_
#endif // STRINGZILLA_ARCH_ARM64_
#ifdef __cplusplus
}
#endif
#endif
