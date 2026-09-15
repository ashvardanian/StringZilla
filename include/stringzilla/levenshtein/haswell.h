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

/** Four byte-wide class ids, as the byte stripe emits them, widened to gather indices. */
SZ_API_COMPTIME sz_u256_vec_t sz_levenshtein_u64x4_classes_u8_haswell(sz_u8_t const *classes) {
    sz_u256_vec_t classes_vec;
    classes_vec.ymm = _mm256_cvtepu8_epi64(_mm_loadu_si32(classes));
    return classes_vec;
}

/** Four four-byte class ids, as the UTF-8 stripe emits them, widened to gather indices. */
SZ_API_COMPTIME sz_u256_vec_t sz_levenshtein_u64x4_classes_u32_haswell(sz_u32_t const *classes) {
    sz_u256_vec_t classes_vec;
    classes_vec.ymm = _mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i const *)classes));
    return classes_vec;
}

/**
 *  @brief Advances four candidates one symbol through exactly @p words verticals; the score moves on the last word.
 *      A candidate past its text keeps stepping whatever class the stripe emits; its score is read where its text ends.
 *  @param[in] words Exactly @c sz_levenshtein_query_words(query->length); a constant keeps the verticals in registers.
 *  @param[in] classes_vec The four candidates' classes at this position, one gather index each.
 */
SZ_API_COMPTIME void sz_levenshtein_u64x4_step_haswell(sz_levenshtein_u64x4_state_haswell_t *state,
                                                       sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                       sz_size_t words, sz_levenshtein_query_t const *query,
                                                       sz_u256_vec_t classes_vec) {
    sz_u256_vec_t last_symbol_bit_vec, last_symbol_shift_vec, positive_carry_vec, negative_carry_vec;
    last_symbol_bit_vec.ymm = _mm256_set1_epi64x((long long)sz_levenshtein_last_symbol_bit_(query->length));
    last_symbol_shift_vec.ymm = _mm256_set1_epi64x((long long)sz_levenshtein_last_symbol_shift_(query->length));
    positive_carry_vec.ymm = _mm256_set1_epi64x(1);
    negative_carry_vec.ymm = _mm256_setzero_si256();
    for (sz_size_t word = 0; word != words; ++word) {
        sz_levenshtein_u64x4_vertical_haswell_t *const vertical = verticals + word;
        sz_u256_vec_t equality_vec, vertical_carry_vec, matched_vec, sum_vec, diagonal_vec;
        sz_u256_vec_t horizontal_positive_vec, horizontal_negative_vec, next_positive_vec, next_negative_vec;
        equality_vec.ymm = _mm256_i64gather_epi64((long long const *)(query->masks + word * query->classes),
                                                  classes_vec.ymm, 8);
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

/** The byte stripe for four candidates: eight positions per four loads and three unpacks while every candidate has
 *  eight bytes left, one byte at a time after that; emits @c sz_u8_t classes. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_u8x4_stripe_haswell(sz_levenshtein_query_t const *query,
                                                             sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                             sz_size_t candidates, sz_size_t *cursors,
                                                             sz_u64_t *symbol_counts, sz_size_t stripe_start,
                                                             sz_size_t positions, void *stripe_classes) {
    sz_unused_(query), sz_unused_(symbol_counts), sz_unused_(stripe_start), sz_unused_(candidates);
    sz_u8_t *const classes = (sz_u8_t *)stripe_classes;
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
        candidate0_vec.xmm = _mm_loadl_epi64((__m128i const *)(texts[0] + cursors[0]));
        candidate1_vec.xmm = _mm_loadl_epi64((__m128i const *)(texts[1] + cursors[1]));
        candidate2_vec.xmm = _mm_loadl_epi64((__m128i const *)(texts[2] + cursors[2]));
        candidate3_vec.xmm = _mm_loadl_epi64((__m128i const *)(texts[3] + cursors[3]));
        pair01_vec.xmm = _mm_unpacklo_epi8(candidate0_vec.xmm, candidate1_vec.xmm);
        pair23_vec.xmm = _mm_unpacklo_epi8(candidate2_vec.xmm, candidate3_vec.xmm);
        _mm_storeu_si128((__m128i *)(classes + position * 4), _mm_unpacklo_epi16(pair01_vec.xmm, pair23_vec.xmm));
        _mm_storeu_si128((__m128i *)(classes + position * 4 + 16), _mm_unpackhi_epi16(pair01_vec.xmm, pair23_vec.xmm));
        cursors[0] += 8, cursors[1] += 8, cursors[2] += 8, cursors[3] += 8;
    }
    for (; position != filled; ++position)
        for (sz_size_t candidate = 0; candidate != 4; ++candidate)
            classes[position * 4 + candidate] = cursors[candidate] < byte_counts[candidate]
                                                    ? (sz_u8_t)texts[candidate][cursors[candidate]++]
                                                    : 0;
    return filled;
}

/** One YMM register per position: a second measured no faster on one-word queries and slower on two. */
enum {
    sz_levenshtein_haswell_u64x4_candidates_per_step_k = 4,
    sz_levenshtein_haswell_u64x4_registers_per_position_k = 1
};

/** Sweeps four candidates through every stripe; @p words is a constant, keeping short queries' verticals in registers.
 *  A candidate's score is read where its text ends; @p symbol_counts seeds as byte counts, refined by the stripe. */
SZ_HELPER_INLINE void sz_levenshtein_haswell_u64x4_sweep_(sz_levenshtein_query_t const *shared_query,
                                                          sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                          sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                          sz_levenshtein_stripe_t stripe,
                                                          sz_levenshtein_classes_width_t width,
                                                          sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                          sz_size_t words, sz_size_t *distances) {
    enum { candidates_per_position_k = 4, positions_per_stripe_k = sz_levenshtein_positions_per_stripe_k };
    // A local copy: nothing stored through the verticals can alias it, so the step keeps its fields in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    unsigned unread = (1u << sweep_count) - 1u;
    sz_levenshtein_u64x4_state_haswell_t state;
    sz_levenshtein_u64x4_init_haswell(&state, verticals, words, query);
    sz_u32_t stripe_classes[positions_per_stripe_k][candidates_per_position_k];
    for (sz_size_t stripe_start = 0, filled = positions_per_stripe_k; filled == positions_per_stripe_k;
         stripe_start += filled) {
        filled = stripe(query, texts, byte_counts, candidates_per_position_k, cursors, symbol_counts, stripe_start,
                        positions_per_stripe_k, &stripe_classes[0][0]);
        for (sz_size_t position = 0; position != filled; ++position) {
            for (unsigned pending = unread; pending; pending &= pending - 1) {
                sz_size_t const candidate = (sz_size_t)sz_u32_ctz(pending);
                if (stripe_start + position != symbol_counts[candidate]) continue;
                distances[candidate] = sz_levenshtein_u64x4_score_haswell(&state, candidate),
                unread &= ~(1u << candidate);
            }
            sz_u256_vec_t const classes_vec = width == sz_levenshtein_classes_u8_k
                                                  ? sz_levenshtein_u64x4_classes_u8_haswell(
                                                        (sz_u8_t const *)&stripe_classes[0][0] +
                                                        position * candidates_per_position_k)
                                                  : sz_levenshtein_u64x4_classes_u32_haswell(stripe_classes[position]);
            sz_levenshtein_u64x4_step_haswell(&state, verticals, words, query, classes_vec);
        }
    }
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)sz_u32_ctz(unread);
        distances[candidate] = sz_levenshtein_u64x4_score_haswell(&state, candidate);
    }
}

/** Streams every candidate through a prepared @p query, four at a time, with @p stripe emitting their
 *  classes at @p width; @p verticals holds enough for a runtime word count. */
SZ_HELPER_INLINE void sz_levenshtein_haswell_u64x4_distances_(
    sz_levenshtein_query_t const *query, sz_sequence_t const *candidates, sz_levenshtein_stripe_t stripe,
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
            sz_levenshtein_haswell_u64x4_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe, width,
                                                resident_verticals, 1, distances + sweep_first);
        else if (words == 2)
            sz_levenshtein_haswell_u64x4_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe, width,
                                                resident_verticals, 2, distances + sweep_first);
        else
            sz_levenshtein_haswell_u64x4_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe, width,
                                                verticals, words, distances + sweep_first);
    }
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_haswell(sz_cptr_t query_text, sz_size_t query_length,
                                                             sz_sequence_t const *candidates,
                                                             sz_memory_allocator_t *alloc, sz_size_t *distances) {
    enum { registers_k = sz_levenshtein_haswell_u64x4_registers_per_position_k };
    if (query_length == 0) return sz_levenshtein_byte_counts_as_distances_(candidates, distances), sz_success_k;
    sz_size_t const words = sz_levenshtein_query_words(query_length);
    sz_size_t const scratch_bytes = sz_levenshtein_distances_scratch_bytes_(
        words * 256, 0, registers_k, words, sizeof(sz_levenshtein_u64x4_vertical_haswell_t));
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_levenshtein_u64x4_vertical_haswell_t *const verticals =
        (sz_levenshtein_u64x4_vertical_haswell_t *)((sz_ptr_t)masks +
                                                    sz_levenshtein_align64_(words * 256 * sizeof(sz_u64_t)));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare(query_text, query_length, masks, &query);
    sz_levenshtein_haswell_u64x4_distances_(&query, candidates, sz_levenshtein_u8x4_stripe_haswell,
                                            sz_levenshtein_classes_u8_k, verticals, distances);
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_utf8_haswell(sz_cptr_t query_text, sz_size_t query_length,
                                                                  sz_sequence_t const *candidates,
                                                                  sz_memory_allocator_t *alloc, sz_size_t *distances) {
    enum { registers_k = sz_levenshtein_haswell_u64x4_registers_per_position_k };
    sz_size_t const runes = sz_levenshtein_utf8_runes(query_text, query_length);
    if (runes == 0) return sz_levenshtein_rune_counts_as_distances_(candidates, distances), sz_success_k;
    sz_size_t const words = sz_levenshtein_query_words(runes);
    sz_size_t const pages_bytes = sz_levenshtein_utf8_pages_bytes(runes);
    sz_size_t const scratch_bytes = sz_levenshtein_distances_scratch_bytes_(
        words * (runes + 1), pages_bytes, registers_k, words, sizeof(sz_levenshtein_u64x4_vertical_haswell_t));
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_ptr_t const pages = (sz_ptr_t)masks + sz_levenshtein_align64_(words * (runes + 1) * sizeof(sz_u64_t));
    sz_levenshtein_u64x4_vertical_haswell_t *const verticals =
        (sz_levenshtein_u64x4_vertical_haswell_t *)(pages + sz_levenshtein_align64_(pages_bytes));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare_utf8(query_text, query_length, masks, pages, &query);
    sz_levenshtein_haswell_u64x4_distances_(&query, candidates, sz_levenshtein_stripe_utf8,
                                            sz_levenshtein_classes_u32_k, verticals, distances);
    alloc->free(scratch, scratch_bytes, alloc->handle);
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
