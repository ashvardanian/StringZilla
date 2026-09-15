/**
 *  @brief Ice Lake (AVX-512) backend for Levenshtein edit distances: eight candidates per ZMM, one 64-bit Myers
 *      word per candidate, the Myers booleans folded into @c VPTERNLOGQ and the score deltas into masked adds.
 *  @file include/stringzilla/levenshtein/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_ICELAKE_H_
#define STRINGZILLA_LEVENSHTEIN_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/levenshtein/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Ice Lake Implementation
#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                                        \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,evex512"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "lzcnt")
#endif

/** Eight candidates' running scores, one per 64-bit ZMM position. */
typedef struct sz_levenshtein_u64x8_state_icelake_t {
    sz_u512_vec_t scores_vec; /**< The running edit distance per candidate. */
} sz_levenshtein_u64x8_state_icelake_t;

/** One query word of eight candidates' Myers states - the vertical deltas of that word. */
typedef struct sz_levenshtein_u64x8_vertical_icelake_t {
    sz_u512_vec_t positive_vec; /**< Myers' VP per candidate. */
    sz_u512_vec_t negative_vec; /**< Myers' VN per candidate. */
} sz_levenshtein_u64x8_vertical_icelake_t;

/** Starts eight candidates: the scores at the query's length, and @p words verticals at the top boundary. */
SZ_API_COMPTIME void sz_levenshtein_u64x8_init_icelake(sz_levenshtein_u64x8_state_icelake_t *state,
                                                       sz_levenshtein_u64x8_vertical_icelake_t *verticals,
                                                       sz_size_t words, sz_levenshtein_query_t const *query) {
    state->scores_vec.zmm = _mm512_set1_epi64((long long)query->length);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.zmm = _mm512_set1_epi64(-1);
        verticals[word].negative_vec.zmm = _mm512_setzero_si512();
    }
}

/** Eight byte-wide class ids, as the byte stripe emits them, widened to gather indices. */
SZ_API_COMPTIME sz_u512_vec_t sz_levenshtein_u64x8_classes_u8_icelake(sz_u8_t const *classes) {
    sz_u512_vec_t classes_vec;
    classes_vec.zmm = _mm512_cvtepu8_epi64(_mm_loadl_epi64((__m128i const *)classes));
    return classes_vec;
}

/** Eight four-byte class ids, as the UTF-8 stripe emits them, widened to gather indices. */
SZ_API_COMPTIME sz_u512_vec_t sz_levenshtein_u64x8_classes_u32_icelake(sz_u32_t const *classes) {
    sz_u512_vec_t classes_vec;
    classes_vec.zmm = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i const *)classes));
    return classes_vec;
}

/** Advances eight candidates one symbol through exactly @p words verticals; the score moves on the last word.
 *  A candidate past its text keeps stepping whatever class the stripe emits; its score is read where its text ends. */
SZ_API_COMPTIME void sz_levenshtein_u64x8_step_icelake(sz_levenshtein_u64x8_state_icelake_t *state,
                                                       sz_levenshtein_u64x8_vertical_icelake_t *verticals,
                                                       sz_size_t words, sz_levenshtein_query_t const *query,
                                                       sz_u512_vec_t classes_vec) {
    enum { ternary_xor_or_k = 0xBE, ternary_or_nor_k = 0xF1 };
    sz_u512_vec_t last_symbol_bit_vec, last_symbol_shift_vec, positive_carry_vec, negative_carry_vec;
    last_symbol_bit_vec.zmm = _mm512_set1_epi64((long long)sz_levenshtein_last_symbol_bit_(query->length));
    last_symbol_shift_vec.zmm = _mm512_set1_epi64((long long)sz_levenshtein_last_symbol_shift_(query->length));
    positive_carry_vec.zmm = _mm512_set1_epi64(1);
    negative_carry_vec.zmm = _mm512_setzero_si512();
    for (sz_size_t word = 0; word != words; ++word) {
        sz_levenshtein_u64x8_vertical_icelake_t *const vertical = verticals + word;
        sz_u512_vec_t equality_vec, vertical_carry_vec, matched_vec, sum_vec, diagonal_vec;
        sz_u512_vec_t horizontal_positive_vec, horizontal_negative_vec, next_positive_vec, next_negative_vec;
        equality_vec.zmm = _mm512_i64gather_epi64(classes_vec.zmm, query->masks + word * query->classes, 8);
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

/** Whether any of the eight candidates can still come under @p radius at @p position: a score falls by at most one
 *  per remaining symbol. Monotone, so once false it stays false; @c SZ_SSIZE_MAX bounds nothing. */
SZ_API_COMPTIME sz_bool_t sz_levenshtein_u64x8_any_active_icelake(sz_levenshtein_u64x8_state_icelake_t const *state,
                                                                  sz_u512_vec_t symbol_counts_vec, sz_size_t position,
                                                                  sz_ssize_t radius) {
    sz_u512_vec_t position_vec, floor_vec;
    position_vec.zmm = _mm512_set1_epi64((long long)position);
    floor_vec.zmm = _mm512_sub_epi64(_mm512_add_epi64(state->scores_vec.zmm, position_vec.zmm), symbol_counts_vec.zmm);
    __mmask8 const active_mask_m8 = _mm512_cmpgt_epi64_mask(symbol_counts_vec.zmm, position_vec.zmm) &
                                    _mm512_cmple_epi64_mask(floor_vec.zmm, _mm512_set1_epi64((long long)radius));
    return active_mask_m8 ? sz_true_k : sz_false_k;
}

/** The running score of candidate @p candidate, read at the position where its text ends. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_u64x8_score_icelake(sz_levenshtein_u64x8_state_icelake_t const *state,
                                                             sz_size_t candidate) {
    return state->scores_vec.u64s[candidate];
}

/** The byte stripe for eight candidates: eight positions per eight loads and three rounds of unpacks while every
 *  candidate has eight bytes left, one byte at a time after that; emits @c sz_u8_t classes. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_u8x8_stripe_icelake(sz_levenshtein_query_t const *query,
                                                             sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                             sz_size_t candidates, sz_size_t *cursors,
                                                             sz_u64_t *symbol_counts, sz_size_t stripe_start,
                                                             sz_size_t positions, void *stripe_classes) {
    sz_unused_(query), sz_unused_(symbol_counts), sz_unused_(stripe_start), sz_unused_(candidates);
    sz_u8_t *const classes = (sz_u8_t *)stripe_classes;
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
            candidates_vec[candidate].xmm = _mm_loadl_epi64((__m128i const *)(texts[candidate] + cursors[candidate]));
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
                                                    ? (sz_u8_t)texts[candidate][cursors[candidate]++]
                                                    : 0;
    return filled;
}

/** One ZMM register per position; unmeasured, the Haswell reading carried over. */
enum {
    sz_levenshtein_icelake_u64x8_candidates_per_step_k = 8,
    sz_levenshtein_icelake_u64x8_registers_per_position_k = 1
};

/** Sweeps eight candidates through every stripe; @p words is a constant, keeping short queries' verticals in registers.
 *  A candidate's score is read where its text ends; @p symbol_counts seeds as byte counts, refined by the stripe. */
SZ_HELPER_INLINE void sz_levenshtein_icelake_u64x8_sweep_(sz_levenshtein_query_t const *shared_query,
                                                          sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                          sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                          sz_levenshtein_stripe_t stripe,
                                                          sz_levenshtein_classes_width_t width,
                                                          sz_levenshtein_u64x8_vertical_icelake_t *verticals,
                                                          sz_size_t words, sz_size_t *distances) {
    enum { candidates_per_position_k = 8, positions_per_stripe_k = sz_levenshtein_positions_per_stripe_k };
    // A local copy: nothing stored through the verticals can alias it, so the step keeps its fields in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    unsigned unread = (1u << sweep_count) - 1u;
    sz_levenshtein_u64x8_state_icelake_t state;
    sz_levenshtein_u64x8_init_icelake(&state, verticals, words, query);
    sz_u32_t stripe_classes[positions_per_stripe_k][candidates_per_position_k];
    for (sz_size_t stripe_start = 0, filled = positions_per_stripe_k; filled == positions_per_stripe_k;
         stripe_start += filled) {
        filled = stripe(query, texts, byte_counts, candidates_per_position_k, cursors, symbol_counts, stripe_start,
                        positions_per_stripe_k, &stripe_classes[0][0]);
        for (sz_size_t position = 0; position != filled; ++position) {
            for (unsigned pending = unread; pending; pending &= pending - 1) {
                sz_size_t const candidate = (sz_size_t)sz_u32_ctz(pending);
                if (stripe_start + position != symbol_counts[candidate]) continue;
                distances[candidate] = sz_levenshtein_u64x8_score_icelake(&state, candidate),
                unread &= ~(1u << candidate);
            }
            sz_u512_vec_t const classes_vec = width == sz_levenshtein_classes_u8_k
                                                  ? sz_levenshtein_u64x8_classes_u8_icelake(
                                                        (sz_u8_t const *)&stripe_classes[0][0] +
                                                        position * candidates_per_position_k)
                                                  : sz_levenshtein_u64x8_classes_u32_icelake(stripe_classes[position]);
            sz_levenshtein_u64x8_step_icelake(&state, verticals, words, query, classes_vec);
        }
    }
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)sz_u32_ctz(unread);
        distances[candidate] = sz_levenshtein_u64x8_score_icelake(&state, candidate);
    }
}

/** Streams every candidate through a prepared @p query, eight at a time, with @p stripe emitting their
 *  classes at @p width; @p verticals holds enough for a runtime word count. */
SZ_HELPER_INLINE void sz_levenshtein_icelake_u64x8_distances_(
    sz_levenshtein_query_t const *query, sz_sequence_t const *candidates, sz_levenshtein_stripe_t stripe,
    sz_levenshtein_classes_width_t width, sz_levenshtein_u64x8_vertical_icelake_t *verticals, sz_size_t *distances) {
    enum { candidates_per_position_k = 8 };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_levenshtein_u64x8_vertical_icelake_t resident_verticals[2];
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
            sz_levenshtein_icelake_u64x8_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe, width,
                                                resident_verticals, 1, distances + sweep_first);
        else if (words == 2)
            sz_levenshtein_icelake_u64x8_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe, width,
                                                resident_verticals, 2, distances + sweep_first);
        else
            sz_levenshtein_icelake_u64x8_sweep_(query, texts, byte_counts, symbol_counts, sweep_count, stripe, width,
                                                verticals, words, distances + sweep_first);
    }
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_icelake(sz_cptr_t query_text, sz_size_t query_length,
                                                             sz_sequence_t const *candidates,
                                                             sz_memory_allocator_t *alloc, sz_size_t *distances) {
    enum { registers_k = sz_levenshtein_icelake_u64x8_registers_per_position_k };
    if (query_length == 0) return sz_levenshtein_byte_counts_as_distances_(candidates, distances), sz_success_k;
    sz_size_t const words = sz_levenshtein_query_words(query_length);
    sz_size_t const scratch_bytes = sz_levenshtein_distances_scratch_bytes_(
        words * 256, 0, registers_k, words, sizeof(sz_levenshtein_u64x8_vertical_icelake_t));
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_levenshtein_u64x8_vertical_icelake_t *const verticals =
        (sz_levenshtein_u64x8_vertical_icelake_t *)((sz_ptr_t)masks +
                                                    sz_levenshtein_align64_(words * 256 * sizeof(sz_u64_t)));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare(query_text, query_length, masks, &query);
    sz_levenshtein_icelake_u64x8_distances_(&query, candidates, sz_levenshtein_u8x8_stripe_icelake,
                                            sz_levenshtein_classes_u8_k, verticals, distances);
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_utf8_icelake(sz_cptr_t query_text, sz_size_t query_length,
                                                                  sz_sequence_t const *candidates,
                                                                  sz_memory_allocator_t *alloc, sz_size_t *distances) {
    enum { registers_k = sz_levenshtein_icelake_u64x8_registers_per_position_k };
    sz_size_t const runes = sz_levenshtein_utf8_runes(query_text, query_length);
    if (runes == 0) return sz_levenshtein_rune_counts_as_distances_(candidates, distances), sz_success_k;
    sz_size_t const words = sz_levenshtein_query_words(runes);
    sz_size_t const pages_bytes = sz_levenshtein_utf8_pages_bytes(runes);
    sz_size_t const scratch_bytes = sz_levenshtein_distances_scratch_bytes_(
        words * (runes + 1), pages_bytes, registers_k, words, sizeof(sz_levenshtein_u64x8_vertical_icelake_t));
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_ptr_t const pages = (sz_ptr_t)masks + sz_levenshtein_align64_(words * (runes + 1) * sizeof(sz_u64_t));
    sz_levenshtein_u64x8_vertical_icelake_t *const verticals =
        (sz_levenshtein_u64x8_vertical_icelake_t *)(pages + sz_levenshtein_align64_(pages_bytes));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare_utf8(query_text, query_length, masks, pages, &query);
    sz_levenshtein_icelake_u64x8_distances_(&query, candidates, sz_levenshtein_stripe_utf8,
                                            sz_levenshtein_classes_u32_k, verticals, distances);
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE
#pragma endregion Ice Lake Implementation

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_LEVENSHTEIN_ICELAKE_H_
