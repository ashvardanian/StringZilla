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

/** The step, moving the score deltas of only the @p active_m64 candidates, so a candidate past its
 *  text keeps the delta it had where the text ended. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_step_icelake_(sz_levenshtein_u8x64_state_icelake_t *state,
                                                           sz_levenshtein_u8x64_vertical_icelake_t *vertical,
                                                           sz_levenshtein_u8x64_query_icelake_t const *packed,
                                                           sz_u512_vec_t classes_vec, __mmask64 active_m64) {
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
    __mmask64 const rose_m64 = _mm512_mask_test_epi8_mask(active_m64, horizontal_positive_u8x64,
                                                          packed->last_symbol_bit_vec.zmm);
    __mmask64 const fell_m64 = _mm512_mask_test_epi8_mask(active_m64, horizontal_negative_u8x64,
                                                          packed->last_symbol_bit_vec.zmm);
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
    sz_levenshtein_u8x64_step_icelake_(state, vertical, packed, classes_vec, (__mmask64) ~(sz_u64_t)0);
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

/** Classes sixteen positions of the first @p lanes, eight or a multiple of eight up to sixty-four,
 *  of @p texts from their @p cursors on, storing the first @p rows transposed: position @c p of
 *  lane @c c at @c p * lanes + c. Each lane's sixteen bytes take one masked load into its quarter
 *  of a register, so a lane past its text reads zero bytes, classed like any other, and touches no
 *  memory; then four lanes a register become every lane a position. */
STRINGZILLA_INLINE void sz_levenshtein_transpose_block_icelake_(sz_u8_t const *byte_to_class, sz_cptr_t const *texts,
                                                                sz_u64_t const *byte_counts, sz_size_t const *cursors,
                                                                sz_size_t lanes, sz_size_t rows, sz_u8_t *classes) {
    sz_u64_t starts[64], masks[64];
    // Each load lands in its lane's quarter, so its address and mask shift by the quarter's offset.
    __m512i const quarter_offsets_u64x8 = _mm512_setr_epi64(0, 16, 32, 48, 0, 16, 32, 48);
    for (sz_size_t chunk = 0; chunk != lanes / 8; ++chunk) {
        __m512i const cursors_u64x8 = _mm512_loadu_si512((void const *)(cursors + chunk * 8));
        __m512i const left_u64x8 = _mm512_sub_epi64(_mm512_loadu_si512((void const *)(byte_counts + chunk * 8)),
                                                    cursors_u64x8);
        __m512i const taken_u64x8 = _mm512_min_epu64(left_u64x8, _mm512_set1_epi64(16));
        __m512i const bits_u64x8 = _mm512_sub_epi64(_mm512_sllv_epi64(_mm512_set1_epi64(1), taken_u64x8),
                                                    _mm512_set1_epi64(1));
        _mm512_storeu_si512((void *)(masks + chunk * 8), _mm512_sllv_epi64(bits_u64x8, quarter_offsets_u64x8));
        _mm512_storeu_si512(
            (void *)(starts + chunk * 8),
            _mm512_sub_epi64(_mm512_add_epi64(_mm512_loadu_si512((void const *)(texts + chunk * 8)), cursors_u64x8),
                             quarter_offsets_u64x8));
    }
    // A pair of permutes classes sixty-four bytes, the byte's top bit picking which half answers.
    __m512i const table_first_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 0));
    __m512i const table_second_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 64));
    __m512i const table_third_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 128));
    __m512i const table_fourth_quarter_u8x64 = _mm512_loadu_si512((void const *)(byte_to_class + 192));
    __m512i quads_u8x64[16];
    for (sz_size_t quad = 0; quad != lanes / 4; ++quad) {
        __m512i bytes_u8x64 = _mm512_setzero_si512();
        for (sz_size_t lane = quad * 4; lane != quad * 4 + 4; ++lane)
            bytes_u8x64 = _mm512_mask_loadu_epi8(bytes_u8x64, (__mmask64)masks[lane], (void const *)starts[lane]);
        quads_u8x64[quad] = _mm512_mask_blend_epi8(
            _mm512_movepi8_mask(bytes_u8x64),
            _mm512_permutex2var_epi8(table_first_quarter_u8x64, bytes_u8x64, table_second_quarter_u8x64),
            _mm512_permutex2var_epi8(table_third_quarter_u8x64, bytes_u8x64, table_fourth_quarter_u8x64));
    }
    // Eight lanes fill two registers, so two two-source permutes transpose them.
    if (lanes == 8) {
        __m512i const first_half_u8x64 = _mm512_set_epi8(
            119, 103, 87, 71, 55, 39, 23, 7, 118, 102, 86, 70, 54, 38, 22, 6, 117, 101, 85, 69, 53, 37, 21, 5, 116, 100,
            84, 68, 52, 36, 20, 4, 115, 99, 83, 67, 51, 35, 19, 3, 114, 98, 82, 66, 50, 34, 18, 2, 113, 97, 81, 65, 49,
            33, 17, 1, 112, 96, 80, 64, 48, 32, 16, 0);
        __m512i const second_half_u8x64 = _mm512_add_epi8(first_half_u8x64, _mm512_set1_epi8(8));
        _mm512_mask_storeu_epi8((void *)classes, sz_u64_mask_until_(rows * 8),
                                _mm512_permutex2var_epi8(quads_u8x64[0], first_half_u8x64, quads_u8x64[1]));
        _mm512_mask_storeu_epi8((void *)(classes + 64), sz_u64_mask_until_(rows > 8 ? rows * 8 - 64 : 0),
                                _mm512_permutex2var_epi8(quads_u8x64[0], second_half_u8x64, quads_u8x64[1]));
        return;
    }
    // New byte `4 * position + quarter` takes old byte `16 * quarter + position`, so dword
    // `position` holds four lanes.
    __m512i const position_major_u8x64 = _mm512_set_epi8(63, 47, 31, 15, 62, 46, 30, 14, 61, 45, 29, 13, 60, 44, 28, 12,
                                                         59, 43, 27, 11, 58, 42, 26, 10, 57, 41, 25, 9, 56, 40, 24, 8,
                                                         55, 39, 23, 7, 54, 38, 22, 6, 53, 37, 21, 5, 52, 36, 20, 4, 51,
                                                         35, 19, 3, 50, 34, 18, 2, 49, 33, 17, 1, 48, 32, 16, 0);
    __m512i quads_u32x16[16];
    for (sz_size_t quad = 0; quad != 16; ++quad)
        quads_u32x16[quad] = quad < lanes / 4 ? _mm512_permutexvar_epi8(position_major_u8x64, quads_u8x64[quad])
                                              : _mm512_setzero_si512();
    // A 16 x 16 dword transpose: two rounds of unpacks within 128-bit lanes, two of lane shuffles.
    __m512i pairs_u32x16[16], fours_u32x16[16], eights_u32x16[16], positions_u8x64[16];
    for (sz_size_t first = 0; first != 16; first += 2) {
        pairs_u32x16[first] = _mm512_unpacklo_epi32(quads_u32x16[first], quads_u32x16[first + 1]);
        pairs_u32x16[first + 1] = _mm512_unpackhi_epi32(quads_u32x16[first], quads_u32x16[first + 1]);
    }
    for (sz_size_t first = 0; first != 16; first += 4) {
        fours_u32x16[first] = _mm512_unpacklo_epi64(pairs_u32x16[first], pairs_u32x16[first + 2]);
        fours_u32x16[first + 1] = _mm512_unpackhi_epi64(pairs_u32x16[first], pairs_u32x16[first + 2]);
        fours_u32x16[first + 2] = _mm512_unpacklo_epi64(pairs_u32x16[first + 1], pairs_u32x16[first + 3]);
        fours_u32x16[first + 3] = _mm512_unpackhi_epi64(pairs_u32x16[first + 1], pairs_u32x16[first + 3]);
    }
    for (sz_size_t column = 0; column != 4; ++column) {
        eights_u32x16[column] = _mm512_shuffle_i32x4(fours_u32x16[column], fours_u32x16[column + 4], 0x88);
        eights_u32x16[column + 4] = _mm512_shuffle_i32x4(fours_u32x16[column], fours_u32x16[column + 4], 0xDD);
        eights_u32x16[column + 8] = _mm512_shuffle_i32x4(fours_u32x16[column + 8], fours_u32x16[column + 12], 0x88);
        eights_u32x16[column + 12] = _mm512_shuffle_i32x4(fours_u32x16[column + 8], fours_u32x16[column + 12], 0xDD);
    }
    for (sz_size_t column = 0; column != 4; ++column) {
        positions_u8x64[column] = _mm512_shuffle_i32x4(eights_u32x16[column], eights_u32x16[column + 8], 0x88);
        positions_u8x64[column + 8] = _mm512_shuffle_i32x4(eights_u32x16[column], eights_u32x16[column + 8], 0xDD);
        positions_u8x64[column + 4] = _mm512_shuffle_i32x4(eights_u32x16[column + 4], eights_u32x16[column + 12], 0x88);
        positions_u8x64[column + 12] = _mm512_shuffle_i32x4(eights_u32x16[column + 4], eights_u32x16[column + 12],
                                                            0xDD);
    }
    for (sz_size_t row = 0; row != rows; ++row)
        _mm512_mask_storeu_epi8((void *)(classes + row * lanes), sz_u64_mask_until_(lanes), positions_u8x64[row]);
}

/** The byte transpose for the first @p lanes candidates, sixteen positions at a time: each
 *  candidate's cursor moves past what it emitted, and the positions filled, the most any lane
 *  emitted, are returned. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_transpose_lanes_icelake_(sz_u8_t const *byte_to_class,
                                                                     sz_cptr_t const *texts,
                                                                     sz_u64_t const *byte_counts, sz_size_t *cursors,
                                                                     sz_size_t lanes, sz_size_t positions,
                                                                     sz_u8_t *classes) {
    __m512i filled_u64x8 = _mm512_setzero_si512();
    for (sz_size_t chunk = 0; chunk != lanes / 8; ++chunk)
        filled_u64x8 = _mm512_max_epu64(filled_u64x8,
                                        _mm512_sub_epi64(_mm512_loadu_si512((void const *)(byte_counts + chunk * 8)),
                                                         _mm512_loadu_si512((void const *)(cursors + chunk * 8))));
    sz_size_t const filled = sz_min_of_two(positions, (sz_size_t)_mm512_reduce_max_epu64(filled_u64x8));
    for (sz_size_t first = 0; first < filled; first += 16) {
        sz_size_t const rows = sz_min_of_two(filled - first, (sz_size_t)16);
        sz_levenshtein_transpose_block_icelake_(byte_to_class, texts, byte_counts, cursors, lanes, rows,
                                                classes + first * lanes);
        __m512i const rows_u64x8 = _mm512_set1_epi64((long long)rows);
        for (sz_size_t chunk = 0; chunk != lanes / 8; ++chunk) {
            __m512i const cursors_u64x8 = _mm512_loadu_si512((void const *)(cursors + chunk * 8));
            __m512i const left_u64x8 = _mm512_sub_epi64(_mm512_loadu_si512((void const *)(byte_counts + chunk * 8)),
                                                        cursors_u64x8);
            _mm512_storeu_si512((void *)(cursors + chunk * 8),
                                _mm512_add_epi64(cursors_u64x8, _mm512_min_epu64(left_u64x8, rows_u64x8)));
        }
    }
    return filled;
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x64_transpose_icelake_(sz_levenshtein_query_t const *query,
                                                                     sz_cptr_t const *texts,
                                                                     sz_u64_t const *byte_counts, sz_size_t candidates,
                                                                     sz_size_t *cursors, sz_u64_t *symbol_counts,
                                                                     sz_size_t transpose_start, sz_size_t positions,
                                                                     void *transpose_classes) {
    enum { lanes_k = sz_levenshtein_u8x64_candidates_per_step_icelake_k };
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    return sz_levenshtein_transpose_lanes_icelake_(query->byte_to_class, texts, byte_counts, cursors, lanes_k,
                                                   positions, (sz_u8_t *)transpose_classes);
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

/** The byte transpose for the eight lanes of the Skylake kernels, sixteen positions per pair of
 *  permutes and classed by @c VPERMI2B rather than a byte at a time. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x8_transpose_icelake_(sz_levenshtein_query_t const *query,
                                                                    sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                    sz_size_t candidates, sz_size_t *cursors,
                                                                    sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                    sz_size_t positions, void *transpose_classes) {
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    return sz_levenshtein_transpose_lanes_icelake_(query->byte_to_class, texts, byte_counts, cursors, 8, positions,
                                                   (sz_u8_t *)transpose_classes);
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

/** The step, moving the score deltas of only the @p active_m32 candidates, so a candidate past its
 *  text keeps the delta it had where the text ended. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_step_icelake_(sz_levenshtein_u16x32_state_icelake_t *state,
                                                            sz_levenshtein_u16x32_vertical_icelake_t *vertical,
                                                            sz_levenshtein_u16x32_query_icelake_t const *packed,
                                                            sz_u512_vec_t classes_vec, __mmask32 active_m32) {
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
    __mmask32 const rose_m32 = _mm512_mask_test_epi16_mask(active_m32, horizontal_positive_u16x32,
                                                           packed->last_symbol_bit_vec.zmm);
    __mmask32 const fell_m32 = _mm512_mask_test_epi16_mask(active_m32, horizontal_negative_u16x32,
                                                           packed->last_symbol_bit_vec.zmm);
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
    sz_levenshtein_u16x32_step_icelake_(state, vertical, packed, classes_vec, (__mmask32) ~(sz_u32_t)0);
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
    return sz_levenshtein_transpose_lanes_icelake_(query->byte_to_class, texts, byte_counts, cursors, lanes_k,
                                                   positions, (sz_u8_t *)transpose_classes);
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

/** Moves the first @p fresh_count seeded candidates into the lowest of the @p idle lanes among the
 *  first @p lanes, answering which lanes took one: each starts at its text's first byte with the
 *  score @p length. */
STRINGZILLA_INLINE sz_u64_t sz_levenshtein_refill_icelake_(sz_u64_t idle, sz_size_t fresh_count, sz_size_t lanes,
                                                           sz_cptr_t const *fresh_texts, sz_u64_t const *fresh_bytes,
                                                           sz_size_t const *fresh_slots, sz_size_t length,
                                                           sz_cptr_t *texts, sz_u64_t *byte_counts, sz_size_t *cursors,
                                                           sz_size_t *slots, sz_size_t *scores) {
    sz_u64_t const fresh = _pdep_u64(sz_u64_mask_until_(fresh_count), idle);
    sz_size_t taken = 0;
    for (sz_size_t chunk = 0; chunk != lanes / 8; ++chunk) {
        __mmask8 const fresh_m8 = (__mmask8)(fresh >> (chunk * 8));
        _mm512_storeu_si512((void *)(texts + chunk * 8),
                            _mm512_mask_expandloadu_epi64(_mm512_loadu_si512((void const *)(texts + chunk * 8)),
                                                          fresh_m8, (void const *)(fresh_texts + taken)));
        _mm512_storeu_si512((void *)(byte_counts + chunk * 8),
                            _mm512_mask_expandloadu_epi64(_mm512_loadu_si512((void const *)(byte_counts + chunk * 8)),
                                                          fresh_m8, (void const *)(fresh_bytes + taken)));
        _mm512_storeu_si512((void *)(slots + chunk * 8),
                            _mm512_mask_expandloadu_epi64(_mm512_loadu_si512((void const *)(slots + chunk * 8)),
                                                          fresh_m8, (void const *)(fresh_slots + taken)));
        _mm512_storeu_si512(
            (void *)(cursors + chunk * 8),
            _mm512_maskz_mov_epi64((__mmask8)~fresh_m8, _mm512_loadu_si512((void const *)(cursors + chunk * 8))));
        _mm512_storeu_si512((void *)(scores + chunk * 8),
                            _mm512_mask_set1_epi64(_mm512_loadu_si512((void const *)(scores + chunk * 8)), fresh_m8,
                                                   (long long)length));
        taken += (sz_size_t)_mm_popcnt_u32(fresh_m8);
    }
    return fresh;
}

/** Where each of the first @p lanes stands at a block's start: the bytes left of its text,
 *  saturated to a byte. */
STRINGZILLA_INLINE void sz_levenshtein_lefts_icelake_(sz_u64_t const *byte_counts, sz_size_t const *cursors,
                                                      sz_size_t lanes, sz_u8_t *lefts) {
    for (sz_size_t chunk = 0; chunk != lanes / 8; ++chunk)
        _mm512_mask_cvtusepi64_storeu_epi8(lefts + chunk * 8, 0xFF,
                                           _mm512_sub_epi64(_mm512_loadu_si512((void const *)(byte_counts + chunk * 8)),
                                                            _mm512_loadu_si512((void const *)(cursors + chunk * 8))));
}

/** Folds a block's @p deltas_vec, sixty-four signed bytes or thirty-two signed halfwords as @p
 *  lanes says, into the lanes' @p scores, then retires every @p live lane whose text ended within
 *  the block or whose floor, its score less the symbols left after the block, has passed @p radius:
 *  its score, or @p bound + 1 past @p bound, lands at its slot of @p distances and its cursor at
 *  its text's end. Every other cursor moves a block on. Answers the retired lanes. */
STRINGZILLA_INLINE sz_u64_t sz_levenshtein_retire_icelake_(sz_u512_vec_t deltas_vec, sz_size_t lanes, sz_u64_t live,
                                                           sz_u64_t const *byte_counts, sz_size_t *cursors,
                                                           sz_size_t const *slots, sz_size_t *scores, sz_ssize_t radius,
                                                           sz_size_t bound, sz_size_t *distances) {
    __m512i const block_u64x8 = _mm512_set1_epi64(sz_levenshtein_freeze_positions_k);
    __m512i const radius_i64x8 = _mm512_set1_epi64((long long)radius);
    __m512i const bound_u64x8 = _mm512_set1_epi64((long long)bound);
    __m512i const over_u64x8 = _mm512_set1_epi64((long long)(bound + 1));
    sz_u64_t retired = 0;
    for (sz_size_t chunk = 0; chunk != lanes / 8; ++chunk) {
        __m512i const deltas_i64x8 =
            lanes == 64 ? _mm512_cvtepi8_epi64(_mm_loadl_epi64((__m128i const *)(deltas_vec.u8s + chunk * 8)))
                        : _mm512_cvtepi16_epi64(_mm_loadu_si128((__m128i const *)(deltas_vec.u16s + chunk * 8)));
        __m512i const scores_u64x8 = _mm512_add_epi64(_mm512_loadu_si512((void const *)(scores + chunk * 8)),
                                                      deltas_i64x8);
        __m512i const cursors_u64x8 = _mm512_loadu_si512((void const *)(cursors + chunk * 8));
        __m512i const counts_u64x8 = _mm512_loadu_si512((void const *)(byte_counts + chunk * 8));
        __m512i const left_u64x8 = _mm512_sub_epi64(counts_u64x8, cursors_u64x8);
        __m512i const floor_i64x8 = _mm512_sub_epi64(_mm512_add_epi64(scores_u64x8, block_u64x8), left_u64x8);
        __mmask8 const live_m8 = (__mmask8)(live >> (chunk * 8));
        __mmask8 const retiring_m8 = _mm512_mask_cmple_epu64_mask(live_m8, left_u64x8, block_u64x8) |
                                     _mm512_mask_cmpgt_epi64_mask(live_m8, floor_i64x8, radius_i64x8);
        __m512i const reported_u64x8 = _mm512_mask_mov_epi64(
            scores_u64x8, _mm512_cmpgt_epu64_mask(scores_u64x8, bound_u64x8), over_u64x8);
        _mm512_mask_i64scatter_epi64(distances, retiring_m8, _mm512_loadu_si512((void const *)(slots + chunk * 8)),
                                     reported_u64x8, 8);
        _mm512_storeu_si512((void *)(scores + chunk * 8), scores_u64x8);
        _mm512_storeu_si512(
            (void *)(cursors + chunk * 8),
            _mm512_mask_mov_epi64(_mm512_min_epu64(_mm512_add_epi64(cursors_u64x8, block_u64x8), counts_u64x8),
                                  retiring_m8, counts_u64x8));
        retired |= (sz_u64_t)retiring_m8 << (chunk * 8);
    }
    return retired;
}

/** Streams every candidate through a prepared byte @p query in sixty-four recycling lanes, a block
 *  of sixteen positions at a time: a lane whose candidate ends within a block or freezes past @p
 *  bound is reported at the block's end and takes the next candidate in @p order at the next
 *  block's start. A lane's score delta stops moving where its text ends, so a block steps straight
 *  through and its bookkeeping is one vector pass. */
STRINGZILLA_INLINE void sz_levenshtein_u8x64_recycled_icelake_(sz_levenshtein_query_t const *shared_query,
                                                               sz_strings_view_t const *candidates,
                                                               sz_size_t const *order, sz_size_t bound,
                                                               sz_size_t *distances) {
    enum { lanes_k = 64, block_k = sz_levenshtein_freeze_positions_k };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_u8x64_query_icelake_t packed;
    sz_levenshtein_u8x64_pack_icelake_(&local_query, &packed);
    sz_cptr_t texts[lanes_k] = {STRINGZILLA_NULL};
    sz_u64_t byte_counts[lanes_k] = {0};
    sz_size_t cursors[lanes_k] = {0}, slots[lanes_k] = {0}, scores[lanes_k] = {0};
    // The seed writes eight entries at a time, so seven of slack.
    sz_cptr_t fresh_texts[lanes_k + 7];
    sz_u64_t fresh_bytes[lanes_k + 7], fresh_symbols[lanes_k + 7];
    sz_size_t fresh_slots[lanes_k + 7];
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_u8_t transpose_classes[block_k][lanes_k];
    sz_u512_vec_t lefts_vec;
    sz_levenshtein_u8x64_state_icelake_t state;
    sz_levenshtein_u8x64_vertical_icelake_t vertical;
    sz_levenshtein_u8x64_init_icelake_(&state, &vertical);
    sz_u64_t live = 0;
    sz_size_t next_rank = 0;
    for (;;) {
        sz_u64_t const idle = ~live;
        if (idle && next_rank != candidates->count) {
            sz_size_t const fresh_count = sz_levenshtein_seed_skylake_(
                candidates, STRINGZILLA_NULL, order, local_query.length, bound, &next_rank,
                (sz_size_t)_mm_popcnt_u64(idle), fresh_slots, fresh_texts, fresh_bytes, fresh_symbols, distances);
            sz_u64_t const fresh = sz_levenshtein_refill_icelake_(idle, fresh_count, lanes_k, fresh_texts, fresh_bytes,
                                                                  fresh_slots, local_query.length, texts, byte_counts,
                                                                  cursors, slots, scores);
            vertical.positive_vec.zmm = _mm512_mask_set1_epi8(vertical.positive_vec.zmm, (__mmask64)fresh, -1);
            vertical.negative_vec.zmm = _mm512_maskz_mov_epi8((__mmask64)~fresh, vertical.negative_vec.zmm);
            live |= fresh;
        }
        if (!live) break;
        sz_levenshtein_transpose_block_icelake_(local_query.byte_to_class, texts, byte_counts, cursors, lanes_k,
                                                block_k, &transpose_classes[0][0]);
        sz_levenshtein_lefts_icelake_(byte_counts, cursors, lanes_k, lefts_vec.u8s);
        __m512i position_u8x64 = _mm512_setzero_si512();
        for (sz_size_t position = 0; position != block_k; ++position) {
            sz_levenshtein_u8x64_step_icelake_(&state, &vertical, &packed,
                                               sz_levenshtein_u8x64_classes_u8_icelake_(transpose_classes[position]),
                                               _mm512_cmpgt_epu8_mask(lefts_vec.zmm, position_u8x64));
            position_u8x64 = _mm512_add_epi8(position_u8x64, _mm512_set1_epi8(1));
        }
        live &= ~sz_levenshtein_retire_icelake_(state.deltas_vec, lanes_k, live, byte_counts, cursors, slots, scores,
                                                radius, bound, distances);
        state.deltas_vec.zmm = _mm512_setzero_si512();
    }
}

/** Streams every candidate through a prepared byte @p query in thirty-two recycling lanes, as
 *  @ref sz_levenshtein_u8x64_recycled_icelake_ does in sixty-four, a block of sixteen at a time. */
STRINGZILLA_INLINE void sz_levenshtein_u16x32_recycled_icelake_(sz_levenshtein_query_t const *shared_query,
                                                                sz_strings_view_t const *candidates,
                                                                sz_size_t const *order, sz_size_t bound,
                                                                sz_size_t *distances) {
    enum { lanes_k = 32, block_k = sz_levenshtein_freeze_positions_k };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_u16x32_query_icelake_t packed;
    sz_levenshtein_u16x32_pack_icelake_(&local_query, &packed);
    sz_cptr_t texts[lanes_k] = {STRINGZILLA_NULL};
    sz_u64_t byte_counts[lanes_k] = {0};
    sz_size_t cursors[lanes_k] = {0}, slots[lanes_k] = {0}, scores[lanes_k] = {0};
    // The seed writes eight entries at a time, so seven of slack.
    sz_cptr_t fresh_texts[lanes_k + 7];
    sz_u64_t fresh_bytes[lanes_k + 7], fresh_symbols[lanes_k + 7];
    sz_size_t fresh_slots[lanes_k + 7];
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_u8_t transpose_classes[block_k][lanes_k];
    sz_u256_vec_t lefts_vec;
    sz_levenshtein_u16x32_state_icelake_t state;
    sz_levenshtein_u16x32_vertical_icelake_t vertical;
    sz_levenshtein_u16x32_init_icelake_(&state, &vertical);
    sz_u64_t live = 0;
    sz_size_t next_rank = 0;
    for (;;) {
        sz_u64_t const idle = sz_u64_mask_until_(lanes_k) & ~live;
        if (idle && next_rank != candidates->count) {
            sz_size_t const fresh_count = sz_levenshtein_seed_skylake_(
                candidates, STRINGZILLA_NULL, order, local_query.length, bound, &next_rank,
                (sz_size_t)_mm_popcnt_u64(idle), fresh_slots, fresh_texts, fresh_bytes, fresh_symbols, distances);
            sz_u64_t const fresh = sz_levenshtein_refill_icelake_(idle, fresh_count, lanes_k, fresh_texts, fresh_bytes,
                                                                  fresh_slots, local_query.length, texts, byte_counts,
                                                                  cursors, slots, scores);
            vertical.positive_vec.zmm = _mm512_mask_set1_epi16(vertical.positive_vec.zmm, (__mmask32)fresh, -1);
            vertical.negative_vec.zmm = _mm512_maskz_mov_epi16((__mmask32)~fresh, vertical.negative_vec.zmm);
            live |= fresh;
        }
        if (!live) break;
        sz_levenshtein_transpose_block_icelake_(local_query.byte_to_class, texts, byte_counts, cursors, lanes_k,
                                                block_k, &transpose_classes[0][0]);
        sz_levenshtein_lefts_icelake_(byte_counts, cursors, lanes_k, lefts_vec.u8s);
        __m512i const lefts_u16x32 = _mm512_cvtepu8_epi16(lefts_vec.ymm);
        __m512i position_u16x32 = _mm512_setzero_si512();
        for (sz_size_t position = 0; position != block_k; ++position) {
            sz_levenshtein_u16x32_step_icelake_(&state, &vertical, &packed,
                                                sz_levenshtein_u16x32_classes_u8_icelake_(transpose_classes[position]),
                                                _mm512_cmpgt_epu16_mask(lefts_u16x32, position_u16x32));
            position_u16x32 = _mm512_add_epi16(position_u16x32, _mm512_set1_epi16(1));
        }
        live &= ~sz_levenshtein_retire_icelake_(state.deltas_vec, lanes_k, live, byte_counts, cursors, slots, scores,
                                                radius, bound, distances);
        state.deltas_vec.zmm = _mm512_setzero_si512();
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
        sz_levenshtein_run_skylake_(engine, index, scored, symbol_counts, order, bound,
                                    sz_levenshtein_u8x8_transpose_icelake_, verticals, distances);
        return;
    }
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
    if (lanes == sz_levenshtein_lanes_u8x64_k)
        sz_levenshtein_u8x64_recycled_icelake_(&query, scored, order, bound, distances);
    else sz_levenshtein_u16x32_recycled_icelake_(&query, scored, order, bound, distances);
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
