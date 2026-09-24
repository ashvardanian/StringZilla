/**
 *  @brief Haswell (AVX2) backend for Levenshtein edit distances: four candidates per YMM, one 64-bit Myers
 *      word per candidate, every candidate reading the same query's match masks.
 *  @file include/stringzilla/levenshtein/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_HASWELL_H_
#define STRINGZILLA_LEVENSHTEIN_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/levenshtein/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,lzcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "lzcnt")
#endif

/** Four candidates' running scores, one per 64-bit YMM position. */
typedef struct sz_levenshtein_u64x4_state_haswell_t {
    sz_u256_vec_t scores_vec; /**< The running edit distance per candidate. */
} sz_levenshtein_u64x4_state_haswell_t;

/** One query word of four candidates' Myers states - the vertical deltas of that word. */
typedef struct sz_levenshtein_u64x4_vertical_haswell_t {
    sz_u256_vec_t positive_vec; /**< Myers' VP per candidate. */
    sz_u256_vec_t negative_vec; /**< Myers' VN per candidate. */
} sz_levenshtein_u64x4_vertical_haswell_t;

/** @c first | ~(second | third) - the AVX2 lowering of the shared Myers @c Ph / @c Pv' shape. */
SZ_HELPER_INLINE __m256i sz_levenshtein_haswell_or_nor_(__m256i first, __m256i second, __m256i third) {
    return _mm256_or_si256(first, _mm256_andnot_si256(_mm256_or_si256(second, third), _mm256_set1_epi64x(-1)));
}

/** Starts four candidates: the scores at the query's length, and @p words verticals at the top boundary. */
SZ_API_COMPTIME void sz_levenshtein_u64x4_init_haswell(sz_levenshtein_u64x4_state_haswell_t *state,
                                                       sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                       sz_size_t words, sz_levenshtein_query_t const *query) {
    state->scores_vec.ymm = _mm256_set1_epi64x((long long)query->length);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.ymm = _mm256_set1_epi64x(-1);
        verticals[word].negative_vec.ymm = _mm256_setzero_si256();
    }
}

/** Four byte-wide class ids, as the byte transpose emits them, widened to gather indices. */
SZ_API_COMPTIME sz_u256_vec_t sz_levenshtein_u64x4_classes_u8_haswell(sz_u8_t const *classes) {
    sz_u256_vec_t classes_vec;
    classes_vec.ymm = _mm256_cvtepu8_epi64(_mm_loadu_si32(classes));
    return classes_vec;
}

/** Four four-byte class ids, as the UTF-8 transpose emits them, widened to gather indices. */
SZ_API_COMPTIME sz_u256_vec_t sz_levenshtein_u64x4_classes_u32_haswell(sz_u32_t const *classes) {
    sz_u256_vec_t classes_vec;
    classes_vec.ymm = _mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i const *)classes));
    return classes_vec;
}

/**
 *  @brief Advances four candidates one symbol through exactly @p words verticals; the score moves on the last word.
 *      A candidate past its text keeps stepping whatever class the transpose emits; its score is read where its text ends.
 *  @param[in] words Exactly @c sz_levenshtein_query_words(query->length); a constant keeps the verticals in registers.
 *  @param[in] classes_vec The four candidates' classes at this position, one gather index each.
 */
SZ_API_COMPTIME void sz_levenshtein_u64x4_step_haswell(sz_levenshtein_u64x4_state_haswell_t *state,
                                                       sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                       sz_size_t words, sz_levenshtein_query_t const *query,
                                                       sz_u256_vec_t classes_vec) {
    sz_u256_vec_t last_symbol_bit_vec, last_symbol_shift_vec, positive_carry_vec, negative_carry_vec, rows_vec;
    last_symbol_bit_vec.ymm = _mm256_set1_epi64x((long long)sz_levenshtein_last_symbol_bit_(query->length));
    last_symbol_shift_vec.ymm = _mm256_set1_epi64x((long long)sz_levenshtein_last_symbol_shift_(query->length));
    positive_carry_vec.ymm = _mm256_set1_epi64x(1);
    negative_carry_vec.ymm = _mm256_setzero_si256();
    rows_vec.ymm = _mm256_mul_epu32(classes_vec.ymm, _mm256_set1_epi64x((long long)query->stride));
    for (sz_size_t word = 0; word != words; ++word) {
        sz_levenshtein_u64x4_vertical_haswell_t *const vertical = verticals + word;
        sz_u256_vec_t equality_vec, vertical_carry_vec, matched_vec, sum_vec, diagonal_vec;
        sz_u256_vec_t horizontal_positive_vec, horizontal_negative_vec, next_positive_vec, next_negative_vec;
        equality_vec.ymm = _mm256_i64gather_epi64((long long const *)(query->masks + word), rows_vec.ymm, 8);
        vertical_carry_vec.ymm = _mm256_or_si256(equality_vec.ymm, vertical->negative_vec.ymm);
        matched_vec.ymm = _mm256_or_si256(equality_vec.ymm, negative_carry_vec.ymm);
        sum_vec.ymm = _mm256_add_epi64(_mm256_and_si256(matched_vec.ymm, vertical->positive_vec.ymm),
                                       vertical->positive_vec.ymm);
        diagonal_vec.ymm = _mm256_or_si256(_mm256_xor_si256(sum_vec.ymm, vertical->positive_vec.ymm), matched_vec.ymm);
        horizontal_positive_vec.ymm = sz_levenshtein_haswell_or_nor_(vertical->negative_vec.ymm, diagonal_vec.ymm,
                                                                     vertical->positive_vec.ymm);
        horizontal_negative_vec.ymm = _mm256_and_si256(vertical->positive_vec.ymm, diagonal_vec.ymm);
        if (word + 1 == words) {
            state->scores_vec.ymm = _mm256_add_epi64(
                state->scores_vec.ymm,
                _mm256_srlv_epi64(_mm256_and_si256(horizontal_positive_vec.ymm, last_symbol_bit_vec.ymm),
                                  last_symbol_shift_vec.ymm));
            state->scores_vec.ymm = _mm256_sub_epi64(
                state->scores_vec.ymm,
                _mm256_srlv_epi64(_mm256_and_si256(horizontal_negative_vec.ymm, last_symbol_bit_vec.ymm),
                                  last_symbol_shift_vec.ymm));
        }
        next_positive_vec.ymm = _mm256_srli_epi64(horizontal_positive_vec.ymm, 63);
        next_negative_vec.ymm = _mm256_srli_epi64(horizontal_negative_vec.ymm, 63);
        horizontal_positive_vec.ymm = _mm256_or_si256(_mm256_slli_epi64(horizontal_positive_vec.ymm, 1),
                                                      positive_carry_vec.ymm);
        horizontal_negative_vec.ymm = _mm256_or_si256(_mm256_slli_epi64(horizontal_negative_vec.ymm, 1),
                                                      negative_carry_vec.ymm);
        positive_carry_vec = next_positive_vec, negative_carry_vec = next_negative_vec;
        vertical->positive_vec.ymm = sz_levenshtein_haswell_or_nor_(horizontal_negative_vec.ymm, vertical_carry_vec.ymm,
                                                                    horizontal_positive_vec.ymm);
        vertical->negative_vec.ymm = _mm256_and_si256(horizontal_positive_vec.ymm, vertical_carry_vec.ymm);
    }
}

/** @brief Whether any of the four candidates can still come under @p radius at @p position: a score falls by at
 *      most one per remaining symbol. Monotone, so once false it stays false; @c SZ_SSIZE_MAX bounds nothing. */
SZ_API_COMPTIME sz_bool_t sz_levenshtein_u64x4_any_active_haswell(sz_levenshtein_u64x4_state_haswell_t const *state,
                                                                  sz_u256_vec_t symbol_counts_vec, sz_size_t position,
                                                                  sz_ssize_t radius) {
    sz_u256_vec_t position_vec, floor_vec, active_vec;
    position_vec.ymm = _mm256_set1_epi64x((long long)position);
    floor_vec.ymm = _mm256_sub_epi64(_mm256_add_epi64(state->scores_vec.ymm, position_vec.ymm), symbol_counts_vec.ymm);
    active_vec.ymm = _mm256_andnot_si256(_mm256_cmpgt_epi64(floor_vec.ymm, _mm256_set1_epi64x((long long)radius)),
                                         _mm256_cmpgt_epi64(symbol_counts_vec.ymm, position_vec.ymm));
    return _mm256_testz_si256(active_vec.ymm, active_vec.ymm) ? sz_false_k : sz_true_k;
}

/** The running score of candidate @p candidate, read at the position where its text ends. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_u64x4_score_haswell(sz_levenshtein_u64x4_state_haswell_t const *state,
                                                             sz_size_t candidate) {
    return state->scores_vec.u64s[candidate];
}

/** The byte transpose for four candidates: eight positions per four loads and three unpacks while every candidate has
 *  eight bytes left, one byte at a time after that; emits the @c sz_u8_t class of every byte. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_u8x4_transpose_haswell(sz_levenshtein_query_t const *query,
                                                                sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                sz_size_t candidates, sz_size_t *cursors,
                                                                sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                sz_size_t positions, void *transpose_classes) {
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u8_t *const classes = (sz_u8_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != 4; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    sz_size_t position = 0;
    for (; position + 8 <= filled; position += 8) {
        if (cursors[0] + 8 > byte_counts[0] || cursors[1] + 8 > byte_counts[1] || cursors[2] + 8 > byte_counts[2] ||
            cursors[3] + 8 > byte_counts[3])
            break;
        sz_u128_vec_t candidate0_vec, candidate1_vec, candidate2_vec, candidate3_vec, pair01_vec, pair23_vec;
        candidate0_vec.xmm = _mm_cvtsi64_si128(
            (long long)sz_levenshtein_octet_classes_(byte_to_class, sz_u64_load(texts[0] + cursors[0]).u64));
        candidate1_vec.xmm = _mm_cvtsi64_si128(
            (long long)sz_levenshtein_octet_classes_(byte_to_class, sz_u64_load(texts[1] + cursors[1]).u64));
        candidate2_vec.xmm = _mm_cvtsi64_si128(
            (long long)sz_levenshtein_octet_classes_(byte_to_class, sz_u64_load(texts[2] + cursors[2]).u64));
        candidate3_vec.xmm = _mm_cvtsi64_si128(
            (long long)sz_levenshtein_octet_classes_(byte_to_class, sz_u64_load(texts[3] + cursors[3]).u64));
        pair01_vec.xmm = _mm_unpacklo_epi8(candidate0_vec.xmm, candidate1_vec.xmm);
        pair23_vec.xmm = _mm_unpacklo_epi8(candidate2_vec.xmm, candidate3_vec.xmm);
        _mm_storeu_si128((__m128i *)(classes + position * 4), _mm_unpacklo_epi16(pair01_vec.xmm, pair23_vec.xmm));
        _mm_storeu_si128((__m128i *)(classes + position * 4 + 16), _mm_unpackhi_epi16(pair01_vec.xmm, pair23_vec.xmm));
        cursors[0] += 8, cursors[1] += 8, cursors[2] += 8, cursors[3] += 8;
    }
    for (; position != filled; ++position)
        for (sz_size_t candidate = 0; candidate != 4; ++candidate)
            classes[position * 4 + candidate] = cursors[candidate] < byte_counts[candidate]
                                                    ? byte_to_class[(sz_u8_t)texts[candidate][cursors[candidate]++]]
                                                    : 0;
    return filled;
}

/** One YMM register per position: a second measured no faster on one-word queries and slower on two. */
enum {
    sz_levenshtein_haswell_u64x4_candidates_per_step_k = 4,
    sz_levenshtein_haswell_u64x4_registers_per_position_k = 1
};

/** Sweeps four candidates through every transpose; @p words is a constant, keeping short queries' verticals in registers.
 *  A candidate's score is read where its text ends; @p symbol_counts seeds as byte counts, refined by the transpose. */
SZ_HELPER_INLINE void sz_levenshtein_haswell_u64x4_sweep_(sz_levenshtein_query_t const *shared_query,
                                                          sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                          sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                          sz_levenshtein_transpose_t transpose,
                                                          sz_levenshtein_classes_width_t width,
                                                          sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                          sz_size_t words, sz_size_t *distances) {
    enum { candidates_per_position_k = 4, positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k };
    // A local copy: nothing stored through the verticals can alias it, so the step keeps its fields in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_u64_t unread = ((sz_u64_t)1 << sweep_count) - 1;
    sz_levenshtein_u64x4_state_haswell_t state;
    sz_levenshtein_u64x4_init_haswell(&state, verticals, words, query);
    sz_u32_t transpose_classes[positions_per_transpose_k][candidates_per_position_k];
    for (sz_size_t transpose_start = 0, filled = positions_per_transpose_k; filled == positions_per_transpose_k;
         transpose_start += filled) {
        filled = transpose(query, texts, byte_counts, candidates_per_position_k, cursors, symbol_counts,
                           transpose_start, positions_per_transpose_k, &transpose_classes[0][0]);
        // A local copy: the transpose may refine the counts, and stores into `distances` must not force reloads.
        sz_u64_t counts[candidates_per_position_k];
        for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
            counts[candidate] = symbol_counts[candidate];
        // When scores must next be read, and whose, so a position costs one compare and retiring costs no test.
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(unread, counts);
        for (sz_size_t position = 0; position != filled; ++position) {
            if (transpose_start + position == deadline.position) {
                // The only read of the scores by lane index, so the sweep keeps them in a register between here.
                sz_levenshtein_u64x4_state_haswell_t const ended = state;
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)_tzcnt_u64(ending);
                    distances[candidate] = sz_levenshtein_u64x4_score_haswell(&ended, candidate);
                }
                unread &= ~deadline.retiring;
                deadline = sz_levenshtein_deadline_(unread, counts);
            }
            sz_u256_vec_t const classes_vec =
                width == sz_levenshtein_classes_u8_k
                    ? sz_levenshtein_u64x4_classes_u8_haswell((sz_u8_t const *)&transpose_classes[0][0] +
                                                              position * candidates_per_position_k)
                    : sz_levenshtein_u64x4_classes_u32_haswell(transpose_classes[position]);
            sz_levenshtein_u64x4_step_haswell(&state, verticals, words, query, classes_vec);
        }
    }
    // Candidates as long as the sweep itself end at the position the transposes never reached.
    sz_levenshtein_u64x4_state_haswell_t const ended = state;
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)_tzcnt_u32(unread);
        distances[candidate] = sz_levenshtein_u64x4_score_haswell(&ended, candidate);
    }
}

/** Streams every candidate through a prepared @p query, four at a time, with @p transpose emitting their
 *  classes at @p width; @p verticals holds enough for a runtime word count. */
SZ_HELPER_INLINE void sz_levenshtein_haswell_u64x4_distances_(
    sz_levenshtein_query_t const *query, sz_sequence_t const *candidates, sz_levenshtein_transpose_t transpose,
    sz_levenshtein_classes_width_t width, sz_levenshtein_u64x4_vertical_haswell_t *verticals, sz_size_t *distances) {
    enum { candidates_per_position_k = 4 };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_levenshtein_u64x4_vertical_haswell_t resident_verticals[2];
    for (sz_size_t sweep_first = 0; sweep_first < candidates->count; sweep_first += candidates_per_position_k) {
        sz_size_t const sweep_count = sz_min_of_two(candidates_per_position_k, candidates->count - sweep_first);
        sz_cptr_t texts[candidates_per_position_k] = {0};
        sz_u64_t byte_counts[candidates_per_position_k] = {0}, symbol_counts[candidates_per_position_k] = {0};
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate) {
            texts[candidate] = candidates->get_start(candidates->handle, sweep_first + candidate);
            byte_counts[candidate] = candidates->get_length(candidates->handle, sweep_first + candidate);
            symbol_counts[candidate] = byte_counts[candidate];
        }
        if (words == 1)
            sz_levenshtein_haswell_u64x4_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, transpose, width,
                                                resident_verticals, 1, distances + sweep_first);
        else if (words == 2)
            sz_levenshtein_haswell_u64x4_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, transpose, width,
                                                resident_verticals, 2, distances + sweep_first);
        else
            sz_levenshtein_haswell_u64x4_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, transpose, width,
                                                verticals, words, distances + sweep_first);
    }
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_haswell(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride) {
    enum { registers_k = sz_levenshtein_haswell_u64x4_registers_per_position_k };
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    sz_bool_t const over_bytes = engine->symbol == sz_levenshtein_bytes_k ? sz_true_k : sz_false_k;
    sz_levenshtein_transpose_t const transpose = over_bytes ? sz_levenshtein_u8x4_transpose_haswell
                                                            : sz_levenshtein_transpose_utf8;
    sz_levenshtein_classes_width_t const width = over_bytes ? sz_levenshtein_classes_u8_k
                                                            : sz_levenshtein_classes_u32_k;
    sz_status_t const grown = sz_levenshtein_engine_scratch_(
        engine, sz_levenshtein_engine_verticals_bytes_(registers_k, sz_levenshtein_engine_words_max_(engine),
                                                       sizeof(sz_levenshtein_u64x4_vertical_haswell_t)));
    if (grown != sz_success_k) return grown;
    sz_levenshtein_u64x4_vertical_haswell_t *const verticals =
        (sz_levenshtein_u64x4_vertical_haswell_t *)sz_levenshtein_engine_verticals_(engine);

    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t *const row = distances + index * distances_stride;
        if (engine->lengths[index] == 0) {
            sz_levenshtein_engine_empty_row_(engine, candidates, row);
            continue;
        }
        sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
        sz_levenshtein_haswell_u64x4_distances_(&query, candidates, transpose, width, verticals, row);
    }
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL
#pragma endregion Haswell Implementation

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_LEVENSHTEIN_HASWELL_H_
