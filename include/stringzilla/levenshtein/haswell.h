/**
 *  @file include/stringzilla/levenshtein/haswell.h
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Haswell (AVX2) backend for Levenshtein edit distances: four candidates per YMM, one
 *      64-bit Myers word per candidate, every candidate reading the same query's match masks.
 *
 *  @sa include/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_HASWELL_H_
#define STRINGZILLA_LEVENSHTEIN_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/utf8_runes/haswell.h" // `sz_utf8_decode_haswell_`

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_HASWELL_
#pragma region Haswell Implementation
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,lzcnt,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "lzcnt", "popcnt")
#endif

/** Four candidates' running scores, one per 64-bit YMM position. */
typedef struct sz_levenshtein_u64x4_state_haswell_t {

    /** The running edit distance per candidate. */
    sz_u256_vec_t scores_vec;
} sz_levenshtein_u64x4_state_haswell_t;

/** One query word of four candidates' Myers states - the vertical deltas of that word. */
typedef struct sz_levenshtein_u64x4_vertical_haswell_t {

    /** Myers' VP per candidate. */
    sz_u256_vec_t positive_vec;

    /** Myers' VN per candidate. */
    sz_u256_vec_t negative_vec;
} sz_levenshtein_u64x4_vertical_haswell_t;

/** The AVX2 lowering of the shared Myers @c Ph and @c Pv' shape: first | ~(second | third). */
STRINGZILLA_INLINE __m256i sz_levenshtein_or_nor_haswell_(__m256i first_u64x4, __m256i second_u64x4,
                                                          __m256i third_u64x4) {
    return _mm256_or_si256(first_u64x4,
                           _mm256_andnot_si256(_mm256_or_si256(second_u64x4, third_u64x4), _mm256_set1_epi64x(-1)));
}

STRINGZILLA_INLINE void sz_levenshtein_u64x4_init_haswell_(sz_levenshtein_u64x4_state_haswell_t *state,
                                                           sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                           sz_size_t words, sz_levenshtein_query_t const *query) {
    state->scores_vec.ymm = _mm256_set1_epi64x((long long)query->length);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.ymm = _mm256_set1_epi64x(-1);
        verticals[word].negative_vec.ymm = _mm256_setzero_si256();
    }
}

/** Starts four candidates: the scores at the query's length, and @p words verticals at
 *  the top boundary. */
STRINGZILLA_INLINE void sz_levenshtein_u64x4_init_haswell(sz_levenshtein_u64x4_state_haswell_t *state,
                                                          sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                          sz_size_t words, sz_levenshtein_query_t const *query) {
    sz_levenshtein_u64x4_init_haswell_(state, verticals, words, query);
}

STRINGZILLA_INLINE sz_u256_vec_t sz_levenshtein_u64x4_classes_u8_haswell_(sz_u8_t const *classes) {
    sz_u256_vec_t classes_vec;
    classes_vec.ymm = _mm256_cvtepu8_epi64(_mm_loadu_si32(classes));
    return classes_vec;
}

/** Four byte-wide class ids, as the byte transpose emits them, widened to gather indices. */
STRINGZILLA_INLINE sz_u256_vec_t sz_levenshtein_u64x4_classes_u8_haswell(sz_u8_t const *classes) {
    return sz_levenshtein_u64x4_classes_u8_haswell_(classes);
}

STRINGZILLA_INLINE sz_u256_vec_t sz_levenshtein_u64x4_classes_u32_haswell_(sz_u32_t const *classes) {
    sz_u256_vec_t classes_vec;
    classes_vec.ymm = _mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i const *)classes));
    return classes_vec;
}

/** Four four-byte class ids, as the UTF-8 transpose emits them, widened to gather indices. */
STRINGZILLA_INLINE sz_u256_vec_t sz_levenshtein_u64x4_classes_u32_haswell(sz_u32_t const *classes) {
    return sz_levenshtein_u64x4_classes_u32_haswell_(classes);
}

STRINGZILLA_INLINE void sz_levenshtein_u64x4_step_haswell_(sz_levenshtein_u64x4_state_haswell_t *state,
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
        // A word's classes sit side by side, so a lane's class indexes that word's table directly.
        equality_vec.ymm = _mm256_i64gather_epi64((long long const *)(query->masks + word * query->stride),
                                                  classes_vec.ymm, 8);
        vertical_carry_vec.ymm = _mm256_or_si256(equality_vec.ymm, vertical->negative_vec.ymm);
        matched_vec.ymm = _mm256_or_si256(equality_vec.ymm, negative_carry_vec.ymm);
        sum_vec.ymm = _mm256_add_epi64(_mm256_and_si256(matched_vec.ymm, vertical->positive_vec.ymm),
                                       vertical->positive_vec.ymm);
        diagonal_vec.ymm = _mm256_or_si256(_mm256_xor_si256(sum_vec.ymm, vertical->positive_vec.ymm), matched_vec.ymm);
        horizontal_positive_vec.ymm = sz_levenshtein_or_nor_haswell_(vertical->negative_vec.ymm, diagonal_vec.ymm,
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
        vertical->positive_vec.ymm = sz_levenshtein_or_nor_haswell_(horizontal_negative_vec.ymm, vertical_carry_vec.ymm,
                                                                    horizontal_positive_vec.ymm);
        vertical->negative_vec.ymm = _mm256_and_si256(horizontal_positive_vec.ymm, vertical_carry_vec.ymm);
    }
}

/**
 *  @brief Advances four candidates one symbol through exactly @p words verticals; the score moves
 *      on the last word. A candidate past its text keeps stepping whatever class the transpose
 *      emits; its score is read where its text ends.
 *  @param[in] words Exactly <tt>sz_levenshtein_query_words(query->length)</tt>; a constant keeps the
 *      verticals in registers.
 *  @param[in] classes_vec The four candidates' classes at this position, one gather index each.
 */
STRINGZILLA_INLINE void sz_levenshtein_u64x4_step_haswell(sz_levenshtein_u64x4_state_haswell_t *state,
                                                          sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                          sz_size_t words, sz_levenshtein_query_t const *query,
                                                          sz_u256_vec_t classes_vec) {
    sz_levenshtein_u64x4_step_haswell_(state, verticals, words, query, classes_vec);
}

/** Whether any of the four candidates can still come under @p radius at @p position: a score
 *  falls by at most one per remaining symbol. Monotone, so once false it stays false;
 *  @c STRINGZILLA_SSIZE_MAX bounds nothing. */
STRINGZILLA_INLINE sz_bool_t sz_levenshtein_u64x4_any_active_haswell(sz_levenshtein_u64x4_state_haswell_t const *state,
                                                                     sz_u256_vec_t symbol_counts_vec,
                                                                     sz_size_t position, sz_ssize_t radius) {
    sz_u256_vec_t position_vec, floor_vec, active_vec;
    position_vec.ymm = _mm256_set1_epi64x((long long)position);
    floor_vec.ymm = _mm256_sub_epi64(_mm256_add_epi64(state->scores_vec.ymm, position_vec.ymm), symbol_counts_vec.ymm);
    active_vec.ymm = _mm256_andnot_si256(_mm256_cmpgt_epi64(floor_vec.ymm, _mm256_set1_epi64x((long long)radius)),
                                         _mm256_cmpgt_epi64(symbol_counts_vec.ymm, position_vec.ymm));
    return _mm256_testz_si256(active_vec.ymm, active_vec.ymm) ? sz_false_k : sz_true_k;
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x4_score_haswell_(sz_levenshtein_u64x4_state_haswell_t const *state,
                                                                 sz_size_t candidate) {
    return state->scores_vec.u64s[candidate];
}

/** The running score of candidate @p candidate, read at the position where its text ends. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x4_score_haswell(sz_levenshtein_u64x4_state_haswell_t const *state,
                                                                sz_size_t candidate) {
    return sz_levenshtein_u64x4_score_haswell_(state, candidate);
}

/** Fills @p slots for the byte query @p byte_to_class maps, laid out as @c byte_slots, answering
 *  whether its present bytes differ in their low nibbles: the absent class is the largest, so a
 *  scan for it finds every byte outside it, and at most sixteen take a store each. */
STRINGZILLA_INLINE sz_bool_t sz_levenshtein_byte_slots_haswell_(sz_u8_t const *byte_to_class, sz_u8_t *slots) {
    __m256i largest_u8x32 = _mm256_setzero_si256();
    for (sz_size_t chunk = 0; chunk != 8; ++chunk)
        largest_u8x32 = _mm256_max_epu8(largest_u8x32,
                                        _mm256_loadu_si256((__m256i const *)(byte_to_class + chunk * 32)));
    __m128i largest_u8x16 = _mm_max_epu8(_mm256_castsi256_si128(largest_u8x32),
                                         _mm256_extracti128_si256(largest_u8x32, 1));
    largest_u8x16 = _mm_max_epu8(largest_u8x16, _mm_srli_si128(largest_u8x16, 8));
    largest_u8x16 = _mm_max_epu8(largest_u8x16, _mm_srli_si128(largest_u8x16, 4));
    largest_u8x16 = _mm_max_epu8(largest_u8x16, _mm_srli_si128(largest_u8x16, 2));
    largest_u8x16 = _mm_max_epu8(largest_u8x16, _mm_srli_si128(largest_u8x16, 1));
    sz_u8_t const absent = (sz_u8_t)_mm_cvtsi128_si32(largest_u8x16);
    __m256i const absent_u8x32 = _mm256_set1_epi8((char)absent);
    sz_u32_t present[8];
    sz_size_t present_count = 0;
    for (sz_size_t chunk = 0; chunk != 8; ++chunk) {
        __m256i const classes_u8x32 = _mm256_loadu_si256((__m256i const *)(byte_to_class + chunk * 32));
        present[chunk] = ~(sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(classes_u8x32, absent_u8x32));
        present_count += (sz_size_t)_mm_popcnt_u32(present[chunk]);
    }
    if (present_count > 16) return sz_false_k;
    for (sz_size_t slot = 0; slot != 16; ++slot) slots[slot] = 0, slots[16 + slot] = absent;
    slots[32] = absent;
    sz_u32_t taken = 0;
    for (sz_size_t chunk = 0; chunk != 8; ++chunk)
        for (sz_u32_t bits = present[chunk]; bits; bits &= bits - 1) {
            sz_size_t const byte = chunk * 32 + _tzcnt_u32(bits);
            if (taken & (1u << (byte & 15))) return sz_false_k;
            taken |= 1u << (byte & 15);
            slots[byte & 15] = (sz_u8_t)byte, slots[16 + (byte & 15)] = byte_to_class[byte];
        }
    return sz_true_k;
}

/** Sixteen bytes of a @p length -byte @p text from @p cursor on, zeros past its end, never read
 *  outside it: a text of sixteen bytes or more loads the sixteen ending at its own end once the
 *  cursor nears it and shuffles out the ones passed; a shorter text reads two clamped octets. */
STRINGZILLA_INLINE __m128i sz_levenshtein_sixteen_haswell_(sz_cptr_t text, sz_size_t length, sz_size_t cursor) {
    if (length < 16)
        return _mm_set_epi64x((long long)sz_levenshtein_octet_(text, length, cursor + 8),
                              (long long)sz_levenshtein_octet_(text, length, cursor));
    sz_size_t const start = sz_min_of_two(cursor, length - 16);
    sz_size_t const passed = sz_min_of_two(cursor, length) - start;
    // Byte `i` takes byte `i + passed`, and a control past fifteen has its top bit set, which
    // zeroes the byte.
    __m128i const control_u8x16 = _mm_add_epi8(_mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                               _mm_set1_epi8((char)passed));
    return _mm_shuffle_epi8(_mm_loadu_si128((__m128i const *)(text + start)),
                            _mm_or_si128(control_u8x16, _mm_cmpgt_epi8(control_u8x16, _mm_set1_epi8(15))));
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x4_transpose_haswell_(sz_levenshtein_query_t const *query,
                                                                    sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                    sz_size_t candidates, sz_size_t *cursors,
                                                                    sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                    sz_size_t positions, void *transpose_classes) {
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u8_t const *const byte_slots = query->byte_slots;
    sz_u8_t *const classes = (sz_u8_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != 4; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    __m256i slot_bytes_u8x32 = _mm256_setzero_si256(), slot_classes_u8x32 = _mm256_setzero_si256();
    __m256i absent_u8x32 = _mm256_setzero_si256();
    if (byte_slots) {
        slot_bytes_u8x32 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)byte_slots));
        slot_classes_u8x32 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(byte_slots + 16)));
        absent_u8x32 = _mm256_set1_epi8((char)byte_slots[32]);
    }
    for (sz_size_t position = 0; position < filled; position += 16) {
        __m128i lanes_u8x16[4];
        for (sz_size_t candidate = 0; candidate != 4; ++candidate)
            lanes_u8x16[candidate] = sz_levenshtein_sixteen_haswell_(
                texts[candidate], (sz_size_t)byte_counts[candidate], cursors[candidate] + position);
        // Two rounds of unpacks turn four lanes of sixteen bytes into sixteen positions of four.
        __m128i const low_pairs_u8x16 = _mm_unpacklo_epi8(lanes_u8x16[0], lanes_u8x16[1]);
        __m128i const high_pairs_u8x16 = _mm_unpackhi_epi8(lanes_u8x16[0], lanes_u8x16[1]);
        __m128i const low_other_pairs_u8x16 = _mm_unpacklo_epi8(lanes_u8x16[2], lanes_u8x16[3]);
        __m128i const high_other_pairs_u8x16 = _mm_unpackhi_epi8(lanes_u8x16[2], lanes_u8x16[3]);
        sz_u256_vec_t halves_vec[2];
        halves_vec[0].ymm = _mm256_setr_m128i(_mm_unpacklo_epi16(low_pairs_u8x16, low_other_pairs_u8x16),
                                              _mm_unpackhi_epi16(low_pairs_u8x16, low_other_pairs_u8x16));
        halves_vec[1].ymm = _mm256_setr_m128i(_mm_unpacklo_epi16(high_pairs_u8x16, high_other_pairs_u8x16),
                                              _mm_unpackhi_epi16(high_pairs_u8x16, high_other_pairs_u8x16));
        for (sz_size_t half = 0; half != 2; ++half) {
            // At most sixteen bytes differing in their low nibbles class in registers.
            if (byte_slots) {
                __m256i const bytes_u8x32 = halves_vec[half].ymm;
                __m256i const nibbles_u8x32 = _mm256_and_si256(bytes_u8x32, _mm256_set1_epi8(15));
                __m256i const slot_matches_u8x32 = _mm256_cmpeq_epi8(
                    _mm256_shuffle_epi8(slot_bytes_u8x32, nibbles_u8x32), bytes_u8x32);
                halves_vec[half].ymm = _mm256_blendv_epi8(
                    absent_u8x32, _mm256_shuffle_epi8(slot_classes_u8x32, nibbles_u8x32), slot_matches_u8x32);
            }
            else
                for (sz_size_t octet = 0; octet != 4; ++octet)
                    halves_vec[half].u64s[octet] = sz_levenshtein_octet_classes_(byte_to_class,
                                                                                 halves_vec[half].u64s[octet]);
        }
        // A last chunk past `positions` stores only the rows the buffer holds.
        sz_size_t const rows = sz_min_of_two(positions - position, (sz_size_t)16);
        if (rows == 16) {
            _mm256_storeu_si256((__m256i *)(classes + position * 4), halves_vec[0].ymm);
            _mm256_storeu_si256((__m256i *)(classes + position * 4 + 32), halves_vec[1].ymm);
        }
        else
            for (sz_size_t byte = 0; byte != rows * 4; ++byte)
                classes[position * 4 + byte] = halves_vec[byte / 32].u8s[byte % 32];
    }
    for (sz_size_t candidate = 0; candidate != 4; ++candidate)
        cursors[candidate] += sz_min_of_two(filled, (sz_size_t)byte_counts[candidate] - cursors[candidate]);
    return filled;
}

/** The byte transpose for four candidates: sixteen positions per four loads, each clamped to its
 *  own candidate's end, classed by the query's byte slots when it has them and by its byte map
 *  otherwise; emits the @c sz_u8_t class of every byte. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x4_transpose_haswell(sz_levenshtein_query_t const *query,
                                                                   sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                   sz_size_t candidates, sz_size_t *cursors,
                                                                   sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                   sz_size_t positions, void *transpose_classes) {
    return sz_levenshtein_u8x4_transpose_haswell_(query, texts, byte_counts, candidates, cursors, symbol_counts,
                                                  transpose_start, positions, transpose_classes);
}

/** One YMM register per position: a second measured no faster on one-word queries and
 *  slower on two. */
enum {
    sz_levenshtein_u64x4_candidates_per_step_haswell_k = 4,
    sz_levenshtein_u64x4_registers_per_position_haswell_k = 1
};

/** Sweeps four candidates through every transpose; @p words and @p width are constants, keeping
 *  short queries' verticals in registers and the class loads unbranched. Each transpose is consumed
 *  in runs from one event to the next: a candidate's text ending, where its score is read, or a
 *  freeze test, which ends the sweep once every unread candidate is past @p radius. Whatever is
 *  still unread then reads its running score; @p symbol_counts seeds as byte counts and the
 *  transpose refines them as it goes. */
STRINGZILLA_INLINE void sz_levenshtein_u64x4_sweep_haswell_(sz_levenshtein_query_t const *shared_query,
                                                            sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                            sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                            sz_levenshtein_transpose_t_ transpose,
                                                            sz_levenshtein_classes_width_t width, sz_ssize_t radius,
                                                            sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                            sz_size_t words, sz_size_t *distances) {
    enum { candidates_per_position_k = 4, positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k };
    // A local copy: nothing stored through the verticals can alias it, so the step keeps its fields in registers.
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_u64_t unread = ((sz_u64_t)1 << sweep_count) - 1;
    sz_levenshtein_u64x4_state_haswell_t state;
    sz_levenshtein_u64x4_init_haswell_(&state, verticals, words, query);
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
                sz_u256_vec_t const classes_vec =
                    width == sz_levenshtein_classes_u8_k
                        ? sz_levenshtein_u64x4_classes_u8_haswell_((sz_u8_t const *)&transpose_classes[0][0] +
                                                                   position * candidates_per_position_k)
                        : sz_levenshtein_u64x4_classes_u32_haswell_(transpose_classes[position]);
                sz_levenshtein_u64x4_step_haswell_(&state, verticals, words, query, classes_vec);
            }
            if (position == filled) break;
            if (event == deadline.position) {
                // The only read of the scores by lane index, so the run keeps them in a register.
                sz_levenshtein_u64x4_state_haswell_t const ended = state;
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)_tzcnt_u64(ending);
                    distances[candidate] = sz_levenshtein_u64x4_score_haswell_(&ended, candidate);
                }
                unread &= ~deadline.retiring;
                deadline = sz_levenshtein_deadline_(unread, counts);
            }
            if (event == freeze_position) {
                sz_u256_vec_t counts_vec;
                counts_vec.ymm = _mm256_loadu_si256((__m256i const *)counts);
                // A zero fill ends the transposes, as when every candidate runs dry.
                if (sz_levenshtein_u64x4_any_active_haswell(&state, counts_vec, event, radius) == sz_false_k) {
                    filled = 0;
                    break;
                }
                freeze_position += sz_levenshtein_freeze_positions_k;
            }
        }
    }
    // Unread candidates end where the transposes ran dry, or froze and read their running score.
    sz_levenshtein_u64x4_state_haswell_t const ended = state;
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)_tzcnt_u64(unread);
        distances[candidate] = sz_levenshtein_u64x4_score_haswell_(&ended, candidate);
    }
}

/** Streams every candidate through a prepared @p query, four at a time in @p order, with
 *  @p transpose emitting their classes at @p width, each distance under @p bound; @p verticals
 *  holds enough for a runtime word count. */
STRINGZILLA_INLINE void sz_levenshtein_u64x4_distances_haswell_(
    sz_levenshtein_query_t const *query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_levenshtein_u64x4_vertical_haswell_t *verticals, sz_size_t *distances) {
    enum { candidates_per_position_k = 4 };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_levenshtein_u64x4_vertical_haswell_t resident_verticals[2];
    for (sz_size_t next_rank = 0; next_rank != candidates->count;) {
        sz_cptr_t texts[candidates_per_position_k];
        sz_u64_t byte_counts[candidates_per_position_k], sweep_symbol_counts[candidates_per_position_k];
        sz_size_t indices[candidates_per_position_k], sweep_distances[candidates_per_position_k];
        sz_size_t const sweep_count = sz_levenshtein_sweep_seed_(candidates, symbol_counts, order, query, bound,
                                                                 &next_rank, candidates_per_position_k, indices, texts,
                                                                 byte_counts, sweep_symbol_counts, distances);
        if (!sweep_count) break;
        if (words == 1)
            sz_levenshtein_u64x4_sweep_haswell_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                width, radius, resident_verticals, 1, sweep_distances);
        else if (words == 2)
            sz_levenshtein_u64x4_sweep_haswell_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                width, radius, resident_verticals, 2, sweep_distances);
        else
            sz_levenshtein_u64x4_sweep_haswell_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                width, radius, verticals, words, sweep_distances);
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
            distances[indices[candidate]] = sz_levenshtein_bounded_(sweep_distances[candidate], bound);
    }
}

#if STRINGZILLA_TARGET_HASWELL

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_haswell(sz_levenshtein_engine_t *engine,
                                                               sz_strings_view_t const *queries,
                                                               sz_levenshtein_symbol_t symbol,
                                                               sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_levenshtein_engine_init_cpu_(engine, queries, symbol, sz_cap_haswell_k, allocator, stream);
}

/** The lanes of four candidates whose floor, their running score less the symbols left before
 *  their @p ends_vec, has passed @p radius at @p position. */
STRINGZILLA_INLINE sz_u64_t sz_levenshtein_u64x4_frozen_haswell_(sz_levenshtein_u64x4_state_haswell_t const *state,
                                                                 sz_u256_vec_t ends_vec, sz_size_t position,
                                                                 sz_ssize_t radius) {
    __m256i const floor_i64x4 = _mm256_sub_epi64(
        _mm256_add_epi64(state->scores_vec.ymm, _mm256_set1_epi64x((long long)position)), ends_vec.ymm);
    return (sz_u64_t)_mm256_movemask_pd(
        _mm256_castsi256_pd(_mm256_cmpgt_epi64(floor_i64x4, _mm256_set1_epi64x((long long)radius))));
}

/** Restarts the lanes of @p fresh at the top boundary: their scores at the query's length and their
 *  verticals at all positive, leaving every other lane as it is. */
STRINGZILLA_INLINE void sz_levenshtein_u64x4_restart_haswell_(sz_levenshtein_u64x4_state_haswell_t *state,
                                                              sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                              sz_size_t words, sz_levenshtein_query_t const *query,
                                                              sz_u64_t fresh) {
    __m256i const fresh_u64x4 = _mm256_sub_epi64(
        _mm256_setzero_si256(),
        _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x((long long)fresh), _mm256_setr_epi64x(0, 1, 2, 3)),
                         _mm256_set1_epi64x(1)));
    state->scores_vec.ymm = _mm256_blendv_epi8(state->scores_vec.ymm, _mm256_set1_epi64x((long long)query->length),
                                               fresh_u64x4);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.ymm = _mm256_or_si256(verticals[word].positive_vec.ymm, fresh_u64x4);
        verticals[word].negative_vec.ymm = _mm256_andnot_si256(fresh_u64x4, verticals[word].negative_vec.ymm);
    }
}

/** Streams every candidate through four lanes that recycle: a lane whose candidate ends or freezes
 *  past @p bound is reported and, at the next freeze test, takes the next candidate in @p order, so
 *  lanes idle at most a freeze interval rather than until their group's longest ends. @p words and
 *  @p width are constants, as in @ref sz_levenshtein_u64x4_sweep_haswell_. */
STRINGZILLA_INLINE void sz_levenshtein_u64x4_recycle_haswell_(
    sz_levenshtein_query_t const *shared_query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_levenshtein_u64x4_vertical_haswell_t *verticals, sz_size_t words, sz_size_t *distances) {
    enum { lanes_k = 4, block_k = sz_levenshtein_freeze_positions_k };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_cptr_t texts[lanes_k] = {STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL};
    sz_u64_t byte_counts[lanes_k] = {0, 0, 0, 0}, ends[lanes_k] = {0, 0, 0, 0};
    sz_size_t cursors[lanes_k] = {0, 0, 0, 0}, slots[lanes_k] = {0, 0, 0, 0};
    // Candidates are seeded sixty-four at a time, so one seed's gap tests serve many refills.
    sz_cptr_t fresh_texts[64];
    sz_u64_t fresh_bytes[64], fresh_symbols[64];
    sz_size_t fresh_slots[64], fresh_taken = 0, fresh_count = 0;
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_u32_t transpose_classes[block_k][lanes_k] = {{0}};
    sz_levenshtein_u64x4_state_haswell_t state;
    sz_levenshtein_u64x4_init_haswell_(&state, verticals, words, query);
    sz_u64_t live = 0, idle = (1u << lanes_k) - 1;
    sz_size_t next_rank = 0;
    for (sz_size_t block_start = 0;; block_start += block_k) {
        // Idle lanes take the next seeded candidates in order, starting at this block.
        sz_u64_t fresh = 0;
        for (; idle; idle &= idle - 1) {
            if (fresh_taken == fresh_count) {
                if (next_rank == candidates->count) break;
                fresh_count = sz_levenshtein_sweep_seed_(candidates, symbol_counts, order, query, bound, &next_rank, 64,
                                                         fresh_slots, fresh_texts, fresh_bytes, fresh_symbols,
                                                         distances);
                fresh_taken = 0;
                if (!fresh_count) break;
            }
            sz_size_t const lane = (sz_size_t)_tzcnt_u64(idle);
            fresh |= (sz_u64_t)1 << lane;
            texts[lane] = fresh_texts[fresh_taken], byte_counts[lane] = fresh_bytes[fresh_taken], cursors[lane] = 0;
            slots[lane] = fresh_slots[fresh_taken], ends[lane] = block_start + fresh_symbols[fresh_taken];
            ++fresh_taken;
        }
        if (fresh) sz_levenshtein_u64x4_restart_haswell_(&state, verticals, words, query, fresh);
        live |= fresh;
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
                sz_u256_vec_t const classes_vec =
                    width == sz_levenshtein_classes_u8_k
                        ? sz_levenshtein_u64x4_classes_u8_haswell_((sz_u8_t const *)&transpose_classes[0][0] +
                                                                   position * lanes_k)
                        : sz_levenshtein_u64x4_classes_u32_haswell_(transpose_classes[position]);
                sz_levenshtein_u64x4_step_haswell_(&state, verticals, words, query, classes_vec);
            }
            if (deadline.position != block_start + position) break;
            sz_levenshtein_u64x4_state_haswell_t const ended = state;
            for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                sz_size_t const lane = (sz_size_t)_tzcnt_u64(ending);
                distances[slots[lane]] = sz_levenshtein_bounded_(ended.scores_vec.u64s[lane], bound);
            }
            live &= ~deadline.retiring, idle |= deadline.retiring;
            if (!live) break;
            deadline = sz_levenshtein_deadline_(live, ends);
        }
        // At every block's end a lane already past its bound reports it and frees its lane.
        sz_u256_vec_t ends_vec;
        ends_vec.ymm = _mm256_loadu_si256((__m256i const *)ends);
        sz_u64_t const frozen = live &
                                sz_levenshtein_u64x4_frozen_haswell_(&state, ends_vec, block_start + block_k, radius);
        for (sz_u64_t ending = frozen; ending; ending &= ending - 1)
            distances[slots[(sz_size_t)_tzcnt_u64(ending)]] = bound + 1;
        live &= ~frozen, idle |= frozen;
    }
}

/** Streams every candidate through recycling lanes, specializing one- and two-word queries to keep
 *  their verticals in registers. */
STRINGZILLA_INLINE void sz_levenshtein_u64x4_recycled_haswell_(
    sz_levenshtein_query_t const *query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_levenshtein_u64x4_vertical_haswell_t *verticals, sz_size_t *distances) {
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_levenshtein_u64x4_vertical_haswell_t resident_verticals[2];
    if (words == 1)
        sz_levenshtein_u64x4_recycle_haswell_(query, candidates, symbol_counts, order, transpose, width, bound,
                                              resident_verticals, 1, distances);
    else if (words == 2)
        sz_levenshtein_u64x4_recycle_haswell_(query, candidates, symbol_counts, order, transpose, width, bound,
                                              resident_verticals, 2, distances);
    else
        sz_levenshtein_u64x4_recycle_haswell_(query, candidates, symbol_counts, order, transpose, width, bound,
                                              verticals, words, distances);
}

/** Scores every candidate of a round against @p engine 's query @p index on Haswell, a byte query
 *  of few distinct bytes classing them by its byte slots. */
STRINGZILLA_INLINE void sz_levenshtein_run_haswell_(sz_levenshtein_engine_t const *engine, sz_size_t index,
                                                    sz_strings_view_t const *scored, sz_size_t const *symbol_counts,
                                                    sz_size_t const *order, sz_size_t bound,
                                                    sz_levenshtein_u64x4_vertical_haswell_t *verticals,
                                                    sz_size_t *distances) {
    if (engine->lengths[index] == 0) {
        sz_levenshtein_engine_empty_(scored, symbol_counts, bound, distances);
        return;
    }
    sz_levenshtein_query_t query = sz_levenshtein_engine_row_(engine, index);
    sz_u8_t byte_slots[33];
    if (engine->symbol == sz_levenshtein_bytes_k &&
        sz_levenshtein_byte_slots_haswell_(query.byte_to_class, byte_slots) == sz_true_k)
        query.byte_slots = byte_slots;
    // A bounded run frees a lane as soon as its candidate passes the bound, where a fixed group of
    // four keeps it stepping until all four do; an unbounded run's sorted lengths end together.
    // Each alphabet passes its transpose and class width as literals, which the sweeps fold.
    if (engine->symbol == sz_levenshtein_bytes_k && bound != STRINGZILLA_SIZE_MAX)
        sz_levenshtein_u64x4_recycled_haswell_(&query, scored, symbol_counts, order,
                                               sz_levenshtein_u8x4_transpose_haswell_, sz_levenshtein_classes_u8_k,
                                               bound, verticals, distances);
    else if (engine->symbol == sz_levenshtein_bytes_k)
        sz_levenshtein_u64x4_distances_haswell_(&query, scored, symbol_counts, order,
                                                sz_levenshtein_u8x4_transpose_haswell_, sz_levenshtein_classes_u8_k,
                                                bound, verticals, distances);
    else if (bound != STRINGZILLA_SIZE_MAX)
        sz_levenshtein_u64x4_recycled_haswell_(&query, scored, symbol_counts, order, sz_levenshtein_transpose_runes_,
                                               sz_levenshtein_classes_u32_k, bound, verticals, distances);
    else
        sz_levenshtein_u64x4_distances_haswell_(&query, scored, symbol_counts, order, sz_levenshtein_transpose_runes_,
                                                sz_levenshtein_classes_u32_k, bound, verticals, distances);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_haswell(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *candidates,
                                                             sz_size_t const *symbol_counts, sz_size_t const *order,
                                                             sz_size_t bound, sz_size_t *distances,
                                                             sz_size_t distances_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_assert_((engine->capability & sz_cap_cpus_k) != 0 &&
               "A host tier never scores a device-prepared engine, whose head only its GPU tier reads");
    enum { registers_k = sz_levenshtein_u64x4_registers_per_position_haswell_k };
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    sz_size_t const verticals_bytes = sz_levenshtein_engine_verticals_bytes_(
        registers_k, sz_levenshtein_engine_words_max_(engine), sizeof(sz_levenshtein_u64x4_vertical_haswell_t));
    sz_strings_view_t scored;
    sz_size_t const *candidate_order;
    sz_status_t const prepared = sz_levenshtein_engine_round_(
        engine, candidates, order, verticals_bytes, sz_utf8_decode_haswell_, &scored, &symbol_counts, &candidate_order);
    if (prepared != sz_success_k) return prepared;
    sz_levenshtein_u64x4_vertical_haswell_t *const verticals =
        (sz_levenshtein_u64x4_vertical_haswell_t *)sz_levenshtein_engine_verticals_(engine);
    for (sz_size_t index = 0; index != engine->count; ++index)
        sz_levenshtein_run_haswell_(engine, index, &scored, symbol_counts, candidate_order, bound, verticals,
                                    distances + index * distances_stride);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#pragma endregion Haswell Implementation
#endif // STRINGZILLA_ARCH_X8664_HASWELL_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_LEVENSHTEIN_HASWELL_H_
