/**
 *  @brief Ice Lake (AVX-512 VBMI) backend for Levenshtein edit distances: a query of at most eight symbols runs
 *      in byte lanes - sixty-four candidates per ZMM, the match masks read by a @c VPERMB - and every wider query
 *      runs the Skylake eight-wide kernel, which needs nothing above AVX-512F.
 *  @file include/stringzilla/levenshtein/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_ICELAKE_H_
#define STRINGZILLA_LEVENSHTEIN_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/levenshtein/skylake.h"

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

#pragma region Narrow Lanes

/** Sixty-four candidates' score movement since the last flush, one signed byte per candidate. */
typedef struct sz_levenshtein_u8x64_state_icelake_t {
    sz_u512_vec_t deltas_vec; /**< How far each candidate's score has moved since the last flush. */
} sz_levenshtein_u8x64_state_icelake_t;

/** Sixty-four candidates' Myers state for a query of at most eight symbols - one byte-wide word per candidate. */
typedef struct sz_levenshtein_u8x64_vertical_icelake_t {
    sz_u512_vec_t positive_vec; /**< Myers' VP per candidate, the query's symbols in the low bits. */
    sz_u512_vec_t negative_vec; /**< Myers' VN per candidate, the query's symbols in the low bits. */
} sz_levenshtein_u8x64_vertical_icelake_t;

/** A query of at most eight symbols as the byte lanes read it, so a step looks its masks up without a gather. */
typedef struct sz_levenshtein_u8x64_query_icelake_t {
    sz_u512_vec_t masks_vec;           /**< Class @c c 's match mask at byte @c c, every other byte zero. */
    sz_u512_vec_t last_symbol_bit_vec; /**< The query's last symbol's bit, repeated in all sixty-four lanes. */
} sz_levenshtein_u8x64_query_icelake_t;

/** One ZMM per position, sixty-four candidates in it, and the vertical never leaves a register. */
enum {
    sz_levenshtein_icelake_u8x64_candidates_per_step_k = 64,
    sz_levenshtein_icelake_u8x64_registers_per_position_k = 1
};

/** Positions the byte-lane deltas span before a signed byte could wrap: a score moves by at most one per step. */
enum { sz_levenshtein_icelake_u8x64_positions_per_flush_k = 64 };

/** Packs a query of at most eight symbols into one byte per class, the table @c VPERMB indexes. */
SZ_API_COMPTIME void sz_levenshtein_u8x64_pack_icelake(sz_levenshtein_query_t const *query,
                                                       sz_levenshtein_u8x64_query_icelake_t *packed) {
    sz_u512_vec_t masks_vec;
    masks_vec.zmm = _mm512_setzero_si512();
    for (sz_size_t class_id = 0; class_id != query->classes; ++class_id)
        masks_vec.u8s[class_id] = (sz_u8_t)query->masks[class_id * query->stride];
    packed->masks_vec = masks_vec;
    packed->last_symbol_bit_vec.zmm = _mm512_set1_epi8((char)sz_levenshtein_last_symbol_bit_(query->length));
}

/** Starts sixty-four candidates: no score movement yet, and the vertical at the top boundary. */
SZ_API_COMPTIME void sz_levenshtein_u8x64_init_icelake(sz_levenshtein_u8x64_state_icelake_t *state,
                                                       sz_levenshtein_u8x64_vertical_icelake_t *vertical) {
    state->deltas_vec.zmm = _mm512_setzero_si512();
    vertical->positive_vec.zmm = _mm512_set1_epi8((char)-1);
    vertical->negative_vec.zmm = _mm512_setzero_si512();
}

/** Sixty-four byte-wide class ids, as the byte stripe emits them, already the indices the mask table takes. */
SZ_API_COMPTIME sz_u512_vec_t sz_levenshtein_u8x64_classes_u8_icelake(sz_u8_t const *classes) {
    sz_u512_vec_t classes_vec;
    classes_vec.zmm = _mm512_loadu_si512((void const *)classes);
    return classes_vec;
}

/** Advances sixty-four candidates one symbol through one byte-wide Myers word, the masks read by a single permute.
 *  A candidate past its text keeps stepping whatever class the stripe emits; its score is read where its text ends. */
SZ_API_COMPTIME void sz_levenshtein_u8x64_step_icelake(sz_levenshtein_u8x64_state_icelake_t *state,
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

/** Folds the byte-lane deltas into @p scores and clears them, so every candidate's score is exact again. */
SZ_API_COMPTIME void sz_levenshtein_u8x64_flush_icelake(sz_levenshtein_u8x64_state_icelake_t *state,
                                                        sz_size_t *scores) {
    enum { lanes_k = sz_levenshtein_icelake_u8x64_candidates_per_step_k };
    sz_u512_vec_t const deltas_vec = state->deltas_vec;
    for (sz_size_t candidate = 0; candidate != lanes_k; ++candidate)
        scores[candidate] = (sz_size_t)((sz_ssize_t)scores[candidate] + deltas_vec.i8s[candidate]);
    state->deltas_vec.zmm = _mm512_setzero_si512();
}

/** The byte stripe for sixty-four candidates, transposed as every stripe is: position @c p of candidate @c c lands
 *  at @c p * 64 + c. The bytes are staged first and classed a whole position at a time, so a class id costs a lane
 *  of a permute rather than a scalar load; a lane past its text pads with a zero byte, classed like any other. */
SZ_API_COMPTIME sz_size_t sz_levenshtein_u8x64_stripe_icelake(sz_levenshtein_query_t const *query,
                                                              sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                              sz_size_t candidates, sz_size_t *cursors,
                                                              sz_u64_t *symbol_counts, sz_size_t stripe_start,
                                                              sz_size_t positions, void *stripe_classes) {
    enum { lanes_k = sz_levenshtein_icelake_u8x64_candidates_per_step_k };
    sz_unused_(symbol_counts), sz_unused_(stripe_start), sz_unused_(candidates);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u8_t *const classes = (sz_u8_t *)stripe_classes;
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

/** Sweeps sixty-four candidates of a query of at most eight symbols through every stripe. Between a retirement and
 *  a flush the step loop carries no scalar work, so a run of positions costs only its permutes and logic. */
SZ_HELPER_INLINE void sz_levenshtein_icelake_u8x64_sweep_(sz_levenshtein_query_t const *shared_query,
                                                          sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                          sz_size_t sweep_count, sz_size_t *distances) {
    enum {
        candidates_per_position_k = sz_levenshtein_icelake_u8x64_candidates_per_step_k,
        positions_per_stripe_k = sz_levenshtein_positions_per_stripe_k,
        positions_per_flush_k = sz_levenshtein_icelake_u8x64_positions_per_flush_k
    };
    // A local copy: nothing stored through the stripe can alias it, so the step keeps the packed query in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_u8x64_query_icelake_t packed;
    sz_levenshtein_u8x64_pack_icelake(&local_query, &packed);
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_size_t scores[candidates_per_position_k];
    for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
        scores[candidate] = local_query.length;
    sz_u64_t unread = sweep_count == candidates_per_position_k ? ~(sz_u64_t)0
                                                               : ((sz_u64_t)1 << sweep_count) - (sz_u64_t)1;
    sz_levenshtein_u8x64_state_icelake_t state;
    sz_levenshtein_u8x64_vertical_icelake_t vertical;
    sz_levenshtein_u8x64_init_icelake(&state, &vertical);
    sz_size_t positions_since_flush = 0;
    sz_u8_t stripe_classes[positions_per_stripe_k][candidates_per_position_k];
    for (sz_size_t stripe_start = 0, filled = positions_per_stripe_k; filled == positions_per_stripe_k;
         stripe_start += filled) {
        filled = sz_levenshtein_u8x64_stripe_icelake(&local_query, texts, byte_counts, candidates_per_position_k,
                                                     cursors, SZ_NULL, stripe_start, positions_per_stripe_k,
                                                     &stripe_classes[0][0]);
        // When scores must next be read, and whose, so a position costs one compare and retiring costs no test.
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(unread, byte_counts);
        for (sz_size_t position = 0; position != filled;) {
            sz_size_t const run_length = sz_min_of_two(
                sz_min_of_two(filled - position, positions_per_flush_k - positions_since_flush),
                deadline.position - (stripe_start + position));
            for (sz_size_t taken = 0; taken != run_length; ++taken, ++position)
                sz_levenshtein_u8x64_step_icelake(&state, &vertical, &packed,
                                                  sz_levenshtein_u8x64_classes_u8_icelake(stripe_classes[position]));
            positions_since_flush += run_length;
            if (positions_since_flush == positions_per_flush_k || stripe_start + position == deadline.position)
                sz_levenshtein_u8x64_flush_icelake(&state, scores), positions_since_flush = 0;
            if (stripe_start + position != deadline.position) continue;
            for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                sz_size_t const candidate = (sz_size_t)_tzcnt_u64(ending);
                distances[candidate] = scores[candidate];
            }
            unread &= ~deadline.retiring;
            deadline = sz_levenshtein_deadline_(unread, byte_counts);
        }
    }
    // Candidates as long as the sweep itself end at the position the stripes never reached.
    sz_levenshtein_u8x64_flush_icelake(&state, scores);
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)_tzcnt_u64(unread);
        distances[candidate] = scores[candidate];
    }
}

/** Streams every candidate through a prepared byte @p query of at most eight symbols, sixty-four at a time. */
SZ_HELPER_INLINE void sz_levenshtein_icelake_u8x64_distances_(sz_levenshtein_query_t const *query,
                                                              sz_sequence_t const *candidates, sz_size_t *distances) {
    enum { candidates_per_position_k = sz_levenshtein_icelake_u8x64_candidates_per_step_k };
    for (sz_size_t sweep_first = 0; sweep_first < candidates->count; sweep_first += candidates_per_position_k) {
        sz_size_t const sweep_count = sz_min_of_two(candidates_per_position_k, candidates->count - sweep_first);
        sz_cptr_t texts[candidates_per_position_k] = {0};
        sz_u64_t byte_counts[candidates_per_position_k] = {0};
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate) {
            texts[candidate] = candidates->get_start(candidates->handle, sweep_first + candidate);
            byte_counts[candidate] = candidates->get_length(candidates->handle, sweep_first + candidate);
        }
        sz_levenshtein_icelake_u8x64_sweep_(query, texts, byte_counts, sweep_count, distances + sweep_first);
    }
}

/** How many symbols of a query one Myers lane holds, which is how many candidates a step advances. */
typedef enum sz_levenshtein_lanes_icelake_t {
    sz_levenshtein_lanes_u8x64_k, /**< Byte lanes: sixty-four candidates a step, up to an eight-symbol query. */
    sz_levenshtein_lanes_u64x8_k, /**< Word lanes: eight candidates a step, at any query length. */
} sz_levenshtein_lanes_icelake_t;

/** The narrowest lanes @p length symbols fit, so the shortest queries advance the most candidates per step. */
SZ_HELPER_AUTO sz_levenshtein_lanes_icelake_t sz_levenshtein_lanes_icelake(sz_size_t length) {
    return length <= 8 ? sz_levenshtein_lanes_u8x64_k : sz_levenshtein_lanes_u64x8_k;
}

#pragma endregion Narrow Lanes

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_icelake(sz_cptr_t query_text, sz_size_t query_length,
                                                             sz_sequence_t const *candidates,
                                                             sz_memory_allocator_t *alloc, sz_size_t *distances) {
    // Only a query short enough for a byte lane is this tier's own; every wider one, and an empty one, is Skylake's.
    if (query_length == 0 || sz_levenshtein_lanes_icelake(query_length) != sz_levenshtein_lanes_u8x64_k)
        return sz_levenshtein_distances_skylake(query_text, query_length, candidates, alloc, distances);
    // The byte lanes keep their vertical in a register, so the scratch is an alignment head, the masks, and the map.
    sz_size_t const mask_entries = sz_levenshtein_query_mask_entries(query_length);
    sz_size_t const scratch_bytes = 64 + sz_levenshtein_align64_(mask_entries * sizeof(sz_u64_t)) +
                                    sz_levenshtein_byte_classes_k;
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    sz_u64_t *const masks = (sz_u64_t *)sz_levenshtein_align64_((sz_size_t)scratch);
    sz_u8_t *const byte_to_class = (sz_u8_t *)masks + sz_levenshtein_align64_(mask_entries * sizeof(sz_u64_t));

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare(query_text, query_length, masks, byte_to_class, &query);
    sz_levenshtein_icelake_u8x64_distances_(&query, candidates, distances);
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
