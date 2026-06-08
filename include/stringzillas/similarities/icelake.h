/**
 *  @file include/stringzillas/similarities/icelake.h
 *  @brief Plain-C AVX-512 (Ice Lake) similarity kernels.
 *  @author Ash Vardanian
 *
 *  Ice Lake backend for the four pairwise alignment operations. The byte/rune Levenshtein diagonals
 *  are SIMD-specialized here: a per-width AVX-512 slice computes one anti-diagonal's interior, and
 *  per-width diagonal bodies drive the triangle/band/triangle geometry around it. Byte-level inputs
 *  cover both linear (`szs_levenshtein_linear_{u8,u16,u32}_icelake_`) and affine
 *  (`szs_levenshtein_affine_{u8,u16,u32}_icelake_`) gaps; rune inputs cover the linear gaps
 *  (`szs_levenshtein_utf8_linear_{u8,u16}_icelake_`). Everything the C++ Ice Lake engine did NOT
 *  specialize - the widest cell width, affine UTF-8, and all of NW/SW - falls through to the serial
 *  backend in `serial.h`, which runs the identical recurrence and is bit-exact against the C++ Ice
 *  Lake instantiations.
 */
#ifndef STRINGZILLAS_SIMILARITIES_ICELAKE_H_
#define STRINGZILLAS_SIMILARITIES_ICELAKE_H_

#include "stringzillas/similarities/serial.h" // Serial fall-through bodies + entry helpers (costs, decode, dispatch)

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#pragma region - Ice Lake Implementation

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#endif

#pragma region - Levenshtein Linear Diagonal Slices

// Each slice computes a contiguous run of `n` cells of one anti-diagonal of the Levenshtein DP
// matrix: cell = min(pre_substitution + (a==b ? match : mismatch), min(pre_insert, pre_delete) + gap).
// Inputs are masked-loaded, results masked-stored; one string is already reversed by the caller, so
// both buffers are traversed in the same order. Register locals are dtype/shape suffixed.

// One byte-level u8-cell diagonal slice (step 64).
SZ_INTERNAL void sz_icelake_slice_u8_linear(                                       //
    sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice, sz_size_t n, //
    sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion,   //
    sz_u8_t const *scores_pre_deletion, sz_u8_t *scores_new,                       //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t gap) {

    sz_u512_vec_t match_cost_u8x64, mismatch_cost_u8x64, gap_cost_u8x64;
    match_cost_u8x64.zmm = _mm512_set1_epi8(match);
    mismatch_cost_u8x64.zmm = _mm512_set1_epi8(mismatch);
    gap_cost_u8x64.zmm = _mm512_set1_epi8(gap);

    for (sz_size_t progress = 0; progress < n; progress += 64) {
        __mmask64 load_mask = sz_u64_clamp_mask_until_(n - progress);
        sz_u512_vec_t first_u8x64, second_u8x64;
        sz_u512_vec_t pre_substitution_u8x64, pre_insert_u8x64, pre_delete_u8x64;
        sz_u512_vec_t cost_of_substitution_u8x64, cost_if_substitution_u8x64, cost_if_gap_u8x64, cell_score_u8x64;
        __mmask64 match_mask;

        first_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice + progress);
        second_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice + progress);
        pre_substitution_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_substitution + progress);
        pre_insert_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_insertion + progress);
        pre_delete_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_deletion + progress);

        match_mask = _mm512_cmpeq_epi8_mask(first_u8x64.zmm, second_u8x64.zmm);
        cost_of_substitution_u8x64.zmm = _mm512_mask_blend_epi8(match_mask, mismatch_cost_u8x64.zmm,
                                                                match_cost_u8x64.zmm);
        cost_if_substitution_u8x64.zmm = _mm512_add_epi8(pre_substitution_u8x64.zmm, cost_of_substitution_u8x64.zmm);
        cost_if_gap_u8x64.zmm = _mm512_add_epi8(_mm512_min_epu8(pre_insert_u8x64.zmm, pre_delete_u8x64.zmm),
                                                gap_cost_u8x64.zmm);
        cell_score_u8x64.zmm = _mm512_min_epu8(cost_if_substitution_u8x64.zmm, cost_if_gap_u8x64.zmm);
        _mm512_mask_storeu_epi8(scores_new + progress, load_mask, cell_score_u8x64.zmm);
    }
}

// One byte-level u16-cell diagonal slice (step 32).
SZ_INTERNAL void sz_icelake_slice_u16_linear(                                      //
    sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice, sz_size_t n, //
    sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion, //
    sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new,                     //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t gap) {

    sz_u512_vec_t match_cost_u16x32, mismatch_cost_u16x32, gap_cost_u16x32;
    match_cost_u16x32.zmm = _mm512_set1_epi16(match);
    mismatch_cost_u16x32.zmm = _mm512_set1_epi16(mismatch);
    gap_cost_u16x32.zmm = _mm512_set1_epi16(gap);

    for (sz_size_t progress = 0; progress < n; progress += 32) {
        __mmask32 load_mask = sz_u32_clamp_mask_until_(n - progress);
        sz_u256_vec_t first_u8x32, second_u8x32;
        sz_u512_vec_t pre_substitution_u16x32, pre_insert_u16x32, pre_delete_u16x32;
        sz_u512_vec_t cost_of_substitution_u16x32, cost_if_substitution_u16x32, cost_if_gap_u16x32, cell_score_u16x32;
        __mmask32 match_mask;

        first_u8x32.ymm = _mm256_maskz_loadu_epi8(load_mask, first_reversed_slice + progress);
        second_u8x32.ymm = _mm256_maskz_loadu_epi8(load_mask, second_slice + progress);
        pre_substitution_u16x32.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_substitution + progress);
        pre_insert_u16x32.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_insertion + progress);
        pre_delete_u16x32.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_deletion + progress);

        match_mask = _mm256_cmpeq_epi8_mask(first_u8x32.ymm, second_u8x32.ymm);
        cost_of_substitution_u16x32.zmm = _mm512_mask_blend_epi16(match_mask, mismatch_cost_u16x32.zmm,
                                                                  match_cost_u16x32.zmm);
        cost_if_substitution_u16x32.zmm = _mm512_add_epi16(pre_substitution_u16x32.zmm,
                                                           cost_of_substitution_u16x32.zmm);
        cost_if_gap_u16x32.zmm = _mm512_add_epi16(_mm512_min_epu16(pre_insert_u16x32.zmm, pre_delete_u16x32.zmm),
                                                  gap_cost_u16x32.zmm);
        cell_score_u16x32.zmm = _mm512_min_epu16(cost_if_substitution_u16x32.zmm, cost_if_gap_u16x32.zmm);
        _mm512_mask_storeu_epi16(scores_new + progress, load_mask, cell_score_u16x32.zmm);
    }
}

// One byte-level u32-cell diagonal slice (step 16).
SZ_INTERNAL void sz_icelake_slice_u32_linear(                                      //
    sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice, sz_size_t n, //
    sz_u32_t const *scores_pre_substitution, sz_u32_t const *scores_pre_insertion, //
    sz_u32_t const *scores_pre_deletion, sz_u32_t *scores_new,                     //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t gap) {

    sz_u512_vec_t match_cost_u32x16, mismatch_cost_u32x16, gap_cost_u32x16;
    match_cost_u32x16.zmm = _mm512_set1_epi32(match);
    mismatch_cost_u32x16.zmm = _mm512_set1_epi32(mismatch);
    gap_cost_u32x16.zmm = _mm512_set1_epi32(gap);

    for (sz_size_t progress = 0; progress < n; progress += 16) {
        __mmask16 load_mask = sz_u16_clamp_mask_until_(n - progress);
        sz_u128_vec_t first_u8x16, second_u8x16;
        sz_u512_vec_t pre_substitution_u32x16, pre_insert_u32x16, pre_delete_u32x16;
        sz_u512_vec_t cost_of_substitution_u32x16, cost_if_substitution_u32x16, cost_if_gap_u32x16, cell_score_u32x16;
        __mmask16 match_mask;

        first_u8x16.xmm = _mm_maskz_loadu_epi8(load_mask, first_reversed_slice + progress);
        second_u8x16.xmm = _mm_maskz_loadu_epi8(load_mask, second_slice + progress);
        pre_substitution_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_substitution + progress);
        pre_insert_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_insertion + progress);
        pre_delete_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_deletion + progress);

        match_mask = _mm_cmpeq_epi8_mask(first_u8x16.xmm, second_u8x16.xmm);
        cost_of_substitution_u32x16.zmm = _mm512_mask_blend_epi32(match_mask, mismatch_cost_u32x16.zmm,
                                                                  match_cost_u32x16.zmm);
        cost_if_substitution_u32x16.zmm = _mm512_add_epi32(pre_substitution_u32x16.zmm,
                                                           cost_of_substitution_u32x16.zmm);
        cost_if_gap_u32x16.zmm = _mm512_add_epi32(_mm512_min_epu32(pre_insert_u32x16.zmm, pre_delete_u32x16.zmm),
                                                  gap_cost_u32x16.zmm);
        cell_score_u32x16.zmm = _mm512_min_epu32(cost_if_substitution_u32x16.zmm, cost_if_gap_u32x16.zmm);
        _mm512_mask_storeu_epi32(scores_new + progress, load_mask, cell_score_u32x16.zmm);
    }
}

// One rune-level u8-cell diagonal slice (step 16).
SZ_INTERNAL void sz_icelake_slice_rune_u8_linear(                                      //
    sz_rune_t const *first_reversed_slice, sz_rune_t const *second_slice, sz_size_t n, //
    sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion,       //
    sz_u8_t const *scores_pre_deletion, sz_u8_t *scores_new,                           //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t gap) {

    sz_u128_vec_t match_cost_u8x16, mismatch_cost_u8x16, gap_cost_u8x16;
    match_cost_u8x16.xmm = _mm_set1_epi8(match);
    mismatch_cost_u8x16.xmm = _mm_set1_epi8(mismatch);
    gap_cost_u8x16.xmm = _mm_set1_epi8(gap);

    for (sz_size_t progress = 0; progress < n; progress += 16) {
        __mmask16 load_mask = sz_u16_clamp_mask_until_(n - progress);
        sz_u512_vec_t first_u32x16, second_u32x16;
        sz_u128_vec_t pre_substitution_u8x16, pre_insert_u8x16, pre_delete_u8x16;
        sz_u128_vec_t cost_of_substitution_u8x16, cost_if_substitution_u8x16, cost_if_gap_u8x16, cell_score_u8x16;
        __mmask16 match_mask;

        first_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, first_reversed_slice + progress);
        second_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, second_slice + progress);
        pre_substitution_u8x16.xmm = _mm_maskz_loadu_epi8(load_mask, scores_pre_substitution + progress);
        pre_insert_u8x16.xmm = _mm_maskz_loadu_epi8(load_mask, scores_pre_insertion + progress);
        pre_delete_u8x16.xmm = _mm_maskz_loadu_epi8(load_mask, scores_pre_deletion + progress);

        match_mask = _mm512_cmpeq_epi32_mask(first_u32x16.zmm, second_u32x16.zmm);
        cost_of_substitution_u8x16.xmm = _mm_mask_blend_epi8(match_mask, mismatch_cost_u8x16.xmm, match_cost_u8x16.xmm);
        cost_if_substitution_u8x16.xmm = _mm_add_epi8(pre_substitution_u8x16.xmm, cost_of_substitution_u8x16.xmm);
        cost_if_gap_u8x16.xmm = _mm_add_epi8(_mm_min_epu8(pre_insert_u8x16.xmm, pre_delete_u8x16.xmm),
                                             gap_cost_u8x16.xmm);
        cell_score_u8x16.xmm = _mm_min_epu8(cost_if_substitution_u8x16.xmm, cost_if_gap_u8x16.xmm);
        _mm_mask_storeu_epi8(scores_new + progress, load_mask, cell_score_u8x16.xmm);
    }
}

// One rune-level u16-cell diagonal slice (step 16).
SZ_INTERNAL void sz_icelake_slice_rune_u16_linear(                                     //
    sz_rune_t const *first_reversed_slice, sz_rune_t const *second_slice, sz_size_t n, //
    sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion,     //
    sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new,                         //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t gap) {

    sz_u256_vec_t match_cost_u16x16, mismatch_cost_u16x16, gap_cost_u16x16;
    match_cost_u16x16.ymm = _mm256_set1_epi16(match);
    mismatch_cost_u16x16.ymm = _mm256_set1_epi16(mismatch);
    gap_cost_u16x16.ymm = _mm256_set1_epi16(gap);

    for (sz_size_t progress = 0; progress < n; progress += 16) {
        __mmask16 load_mask = sz_u16_clamp_mask_until_(n - progress);
        sz_u512_vec_t first_u32x16, second_u32x16;
        sz_u256_vec_t pre_substitution_u16x16, pre_insert_u16x16, pre_delete_u16x16;
        sz_u256_vec_t cost_of_substitution_u16x16, cost_if_substitution_u16x16, cost_if_gap_u16x16, cell_score_u16x16;
        __mmask16 match_mask;

        first_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, first_reversed_slice + progress);
        second_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, second_slice + progress);
        pre_substitution_u16x16.ymm = _mm256_maskz_loadu_epi16(load_mask, scores_pre_substitution + progress);
        pre_insert_u16x16.ymm = _mm256_maskz_loadu_epi16(load_mask, scores_pre_insertion + progress);
        pre_delete_u16x16.ymm = _mm256_maskz_loadu_epi16(load_mask, scores_pre_deletion + progress);

        match_mask = _mm512_cmpeq_epi32_mask(first_u32x16.zmm, second_u32x16.zmm);
        cost_of_substitution_u16x16.ymm = _mm256_mask_blend_epi16(match_mask, mismatch_cost_u16x16.ymm,
                                                                  match_cost_u16x16.ymm);
        cost_if_substitution_u16x16.ymm = _mm256_add_epi16(pre_substitution_u16x16.ymm,
                                                           cost_of_substitution_u16x16.ymm);
        cost_if_gap_u16x16.ymm = _mm256_add_epi16(_mm256_min_epu16(pre_insert_u16x16.ymm, pre_delete_u16x16.ymm),
                                                  gap_cost_u16x16.ymm);
        cell_score_u16x16.ymm = _mm256_min_epu16(cost_if_substitution_u16x16.ymm, cost_if_gap_u16x16.ymm);
        _mm256_mask_storeu_epi16(scores_new + progress, load_mask, cell_score_u16x16.ymm);
    }
}

#pragma endregion

#pragma region - Levenshtein Affine Diagonal Slices

// Each affine slice computes a contiguous run of `n` cells of one anti-diagonal of the Gotoh DP:
// insert = min(up_insert + extend, up_main + open); delete = min(left_delete + extend, left_main + open);
// cell = min(pre_substitution + (a==b ? match : mismatch), min(insert, delete)). The up-neighbor main &
// insert tracks arrive as `scores_pre_insertion` / `scores_running_insertions`; the left-neighbor main &
// delete tracks as `scores_pre_deletion` / `scores_running_deletions`; the caller offsets the left-side
// pointers by one cell so both runs traverse identically. Results write back the cell plus its next
// insert/delete tracks. One string is already reversed by the caller. Register locals are suffixed.

// One byte-level u8-cell affine diagonal slice (step 64).
SZ_INTERNAL void sz_icelake_slice_u8_affine(                                       //
    sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice, sz_size_t n, //
    sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion,   //
    sz_u8_t const *scores_pre_deletion, sz_u8_t const *scores_running_insertions,  //
    sz_u8_t const *scores_running_deletions, sz_u8_t *scores_new,                  //
    sz_u8_t *scores_new_insertions, sz_u8_t *scores_new_deletions,                 //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend) {

    sz_u512_vec_t match_cost_u8x64, mismatch_cost_u8x64, gap_open_u8x64, gap_extend_u8x64;
    match_cost_u8x64.zmm = _mm512_set1_epi8(match);
    mismatch_cost_u8x64.zmm = _mm512_set1_epi8(mismatch);
    gap_open_u8x64.zmm = _mm512_set1_epi8(open);
    gap_extend_u8x64.zmm = _mm512_set1_epi8(extend);

    for (sz_size_t progress = 0; progress < n; progress += 64) {
        __mmask64 load_mask = sz_u64_clamp_mask_until_(n - progress);
        sz_u512_vec_t first_u8x64, second_u8x64;
        sz_u512_vec_t pre_substitution_u8x64, pre_insert_open_u8x64, pre_delete_open_u8x64;
        sz_u512_vec_t pre_insert_extend_u8x64, pre_delete_extend_u8x64;
        sz_u512_vec_t cost_of_substitution_u8x64, cost_if_substitution_u8x64;
        sz_u512_vec_t cost_if_insert_u8x64, cost_if_delete_u8x64, cell_score_u8x64;
        __mmask64 match_mask;

        first_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice + progress);
        second_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice + progress);
        pre_substitution_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_substitution + progress);
        pre_insert_open_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_insertion + progress);
        pre_delete_open_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_deletion + progress);
        pre_insert_extend_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_running_insertions + progress);
        pre_delete_extend_u8x64.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_running_deletions + progress);

        match_mask = _mm512_cmpeq_epi8_mask(first_u8x64.zmm, second_u8x64.zmm);
        cost_of_substitution_u8x64.zmm = _mm512_mask_blend_epi8(match_mask, mismatch_cost_u8x64.zmm,
                                                                match_cost_u8x64.zmm);
        cost_if_substitution_u8x64.zmm = _mm512_add_epi8(pre_substitution_u8x64.zmm, cost_of_substitution_u8x64.zmm);
        cost_if_insert_u8x64.zmm = _mm512_min_epu8(_mm512_add_epi8(pre_insert_extend_u8x64.zmm, gap_extend_u8x64.zmm),
                                                   _mm512_add_epi8(pre_insert_open_u8x64.zmm, gap_open_u8x64.zmm));
        cost_if_delete_u8x64.zmm = _mm512_min_epu8(_mm512_add_epi8(pre_delete_extend_u8x64.zmm, gap_extend_u8x64.zmm),
                                                   _mm512_add_epi8(pre_delete_open_u8x64.zmm, gap_open_u8x64.zmm));
        cell_score_u8x64.zmm = _mm512_min_epu8(cost_if_substitution_u8x64.zmm,
                                               _mm512_min_epu8(cost_if_insert_u8x64.zmm, cost_if_delete_u8x64.zmm));
        _mm512_mask_storeu_epi8(scores_new + progress, load_mask, cell_score_u8x64.zmm);
        _mm512_mask_storeu_epi8(scores_new_insertions + progress, load_mask, cost_if_insert_u8x64.zmm);
        _mm512_mask_storeu_epi8(scores_new_deletions + progress, load_mask, cost_if_delete_u8x64.zmm);
    }
}

// One byte-level u16-cell affine diagonal slice (step 32).
SZ_INTERNAL void sz_icelake_slice_u16_affine(                                       //
    sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice, sz_size_t n,  //
    sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion,  //
    sz_u16_t const *scores_pre_deletion, sz_u16_t const *scores_running_insertions, //
    sz_u16_t const *scores_running_deletions, sz_u16_t *scores_new,                 //
    sz_u16_t *scores_new_insertions, sz_u16_t *scores_new_deletions,                //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend) {

    sz_u512_vec_t match_cost_u16x32, mismatch_cost_u16x32, gap_open_u16x32, gap_extend_u16x32;
    match_cost_u16x32.zmm = _mm512_set1_epi16(match);
    mismatch_cost_u16x32.zmm = _mm512_set1_epi16(mismatch);
    gap_open_u16x32.zmm = _mm512_set1_epi16(open);
    gap_extend_u16x32.zmm = _mm512_set1_epi16(extend);

    for (sz_size_t progress = 0; progress < n; progress += 32) {
        __mmask32 load_mask = sz_u32_clamp_mask_until_(n - progress);
        sz_u256_vec_t first_u8x32, second_u8x32;
        sz_u512_vec_t pre_substitution_u16x32, pre_insert_open_u16x32, pre_delete_open_u16x32;
        sz_u512_vec_t pre_insert_extend_u16x32, pre_delete_extend_u16x32;
        sz_u512_vec_t cost_of_substitution_u16x32, cost_if_substitution_u16x32;
        sz_u512_vec_t cost_if_insert_u16x32, cost_if_delete_u16x32, cell_score_u16x32;
        __mmask32 match_mask;

        first_u8x32.ymm = _mm256_maskz_loadu_epi8(load_mask, first_reversed_slice + progress);
        second_u8x32.ymm = _mm256_maskz_loadu_epi8(load_mask, second_slice + progress);
        pre_substitution_u16x32.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_substitution + progress);
        pre_insert_open_u16x32.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_insertion + progress);
        pre_delete_open_u16x32.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_deletion + progress);
        pre_insert_extend_u16x32.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_running_insertions + progress);
        pre_delete_extend_u16x32.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_running_deletions + progress);

        match_mask = _mm256_cmpeq_epi8_mask(first_u8x32.ymm, second_u8x32.ymm);
        cost_of_substitution_u16x32.zmm = _mm512_mask_blend_epi16(match_mask, mismatch_cost_u16x32.zmm,
                                                                  match_cost_u16x32.zmm);
        cost_if_substitution_u16x32.zmm = _mm512_add_epi16(pre_substitution_u16x32.zmm,
                                                           cost_of_substitution_u16x32.zmm);
        cost_if_insert_u16x32.zmm = _mm512_min_epu16(
            _mm512_add_epi16(pre_insert_extend_u16x32.zmm, gap_extend_u16x32.zmm),
            _mm512_add_epi16(pre_insert_open_u16x32.zmm, gap_open_u16x32.zmm));
        cost_if_delete_u16x32.zmm = _mm512_min_epu16(
            _mm512_add_epi16(pre_delete_extend_u16x32.zmm, gap_extend_u16x32.zmm),
            _mm512_add_epi16(pre_delete_open_u16x32.zmm, gap_open_u16x32.zmm));
        cell_score_u16x32.zmm = _mm512_min_epu16(
            cost_if_substitution_u16x32.zmm, _mm512_min_epu16(cost_if_insert_u16x32.zmm, cost_if_delete_u16x32.zmm));
        _mm512_mask_storeu_epi16(scores_new + progress, load_mask, cell_score_u16x32.zmm);
        _mm512_mask_storeu_epi16(scores_new_insertions + progress, load_mask, cost_if_insert_u16x32.zmm);
        _mm512_mask_storeu_epi16(scores_new_deletions + progress, load_mask, cost_if_delete_u16x32.zmm);
    }
}

// One byte-level u32-cell affine diagonal slice (step 16).
SZ_INTERNAL void sz_icelake_slice_u32_affine(                                       //
    sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice, sz_size_t n,  //
    sz_u32_t const *scores_pre_substitution, sz_u32_t const *scores_pre_insertion,  //
    sz_u32_t const *scores_pre_deletion, sz_u32_t const *scores_running_insertions, //
    sz_u32_t const *scores_running_deletions, sz_u32_t *scores_new,                 //
    sz_u32_t *scores_new_insertions, sz_u32_t *scores_new_deletions,                //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend) {

    sz_u512_vec_t match_cost_u32x16, mismatch_cost_u32x16, gap_open_u32x16, gap_extend_u32x16;
    match_cost_u32x16.zmm = _mm512_set1_epi32(match);
    mismatch_cost_u32x16.zmm = _mm512_set1_epi32(mismatch);
    gap_open_u32x16.zmm = _mm512_set1_epi32(open);
    gap_extend_u32x16.zmm = _mm512_set1_epi32(extend);

    for (sz_size_t progress = 0; progress < n; progress += 16) {
        __mmask16 load_mask = sz_u16_clamp_mask_until_(n - progress);
        sz_u128_vec_t first_u8x16, second_u8x16;
        sz_u512_vec_t pre_substitution_u32x16, pre_insert_open_u32x16, pre_delete_open_u32x16;
        sz_u512_vec_t pre_insert_extend_u32x16, pre_delete_extend_u32x16;
        sz_u512_vec_t cost_of_substitution_u32x16, cost_if_substitution_u32x16;
        sz_u512_vec_t cost_if_insert_u32x16, cost_if_delete_u32x16, cell_score_u32x16;
        __mmask16 match_mask;

        first_u8x16.xmm = _mm_maskz_loadu_epi8(load_mask, first_reversed_slice + progress);
        second_u8x16.xmm = _mm_maskz_loadu_epi8(load_mask, second_slice + progress);
        pre_substitution_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_substitution + progress);
        pre_insert_open_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_insertion + progress);
        pre_delete_open_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_deletion + progress);
        pre_insert_extend_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_running_insertions + progress);
        pre_delete_extend_u32x16.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_running_deletions + progress);

        match_mask = _mm_cmpeq_epi8_mask(first_u8x16.xmm, second_u8x16.xmm);
        cost_of_substitution_u32x16.zmm = _mm512_mask_blend_epi32(match_mask, mismatch_cost_u32x16.zmm,
                                                                  match_cost_u32x16.zmm);
        cost_if_substitution_u32x16.zmm = _mm512_add_epi32(pre_substitution_u32x16.zmm,
                                                           cost_of_substitution_u32x16.zmm);
        cost_if_insert_u32x16.zmm = _mm512_min_epu32(
            _mm512_add_epi32(pre_insert_extend_u32x16.zmm, gap_extend_u32x16.zmm),
            _mm512_add_epi32(pre_insert_open_u32x16.zmm, gap_open_u32x16.zmm));
        cost_if_delete_u32x16.zmm = _mm512_min_epu32(
            _mm512_add_epi32(pre_delete_extend_u32x16.zmm, gap_extend_u32x16.zmm),
            _mm512_add_epi32(pre_delete_open_u32x16.zmm, gap_open_u32x16.zmm));
        cell_score_u32x16.zmm = _mm512_min_epu32(
            cost_if_substitution_u32x16.zmm, _mm512_min_epu32(cost_if_insert_u32x16.zmm, cost_if_delete_u32x16.zmm));
        _mm512_mask_storeu_epi32(scores_new + progress, load_mask, cell_score_u32x16.zmm);
        _mm512_mask_storeu_epi32(scores_new_insertions + progress, load_mask, cost_if_insert_u32x16.zmm);
        _mm512_mask_storeu_epi32(scores_new_deletions + progress, load_mask, cost_if_delete_u32x16.zmm);
    }
}

#pragma endregion

#pragma region - Levenshtein Linear Diagonal Bodies

// The five bodies below are the explicit per-width Levenshtein linear-gap diagonal sweeps the macro
// once stamped. They share the triangle/band/triangle anti-diagonal geometry, the boundary init via
// `sz_init_score_linear`, and the 3-buffer rotation via `sz_rotate3_ptr` with the serial body in
// `serial.h`; only the per-cell inner loop differs - one AVX-512 `sz_icelake_slice_*_linear` call per
// anti-diagonal interior instead of a scalar loop. The cost model is the uniform Levenshtein one
// (match/mismatch + linear gap, minimize/global) read straight off `szs_cell_costs_t`.

// Byte-level Levenshtein, u8 cells (64 lanes/ZMM).
SZ_INTERNAL sz_status_t szs_levenshtein_linear_u8_icelake_(sz_u8_t const *first, sz_size_t first_length,
                                                           sz_u8_t const *second, sz_size_t second_length,
                                                           szs_cell_costs_t const *costs, sz_memory_allocator_t *alloc,
                                                           sz_u8_t *result) {
    sz_error_cost_t const match = costs->match, mismatch = costs->mismatch, gap = costs->gap_open;
    if (first_length == 0 || second_length == 0) {
        sz_u8_t value = 0;
        if (first_length != 0) value = (sz_u8_t)((sz_i64_t)gap * (sz_i64_t)first_length);
        else if (second_length != 0) value = (sz_u8_t)((sz_i64_t)gap * (sz_i64_t)second_length);
        *result = value;
        return sz_success_k;
    }

    sz_u8_t const *shorter = first;
    sz_u8_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_u8_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const bytes_per_diagonal = sz_round_up_to_multiple(max_diagonal_length * sizeof(sz_u8_t), 64);
    sz_size_t const buffer_length = bytes_per_diagonal * 3 + shorter_length * sizeof(sz_u8_t);
    sz_u8_t *buffer = (sz_u8_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_u8_t *previous_scores = (sz_u8_t *)(buffer);
    sz_u8_t *current_scores = (sz_u8_t *)(buffer + bytes_per_diagonal);
    sz_u8_t *next_scores = (sz_u8_t *)(buffer + bytes_per_diagonal * 2);
    sz_u8_t *shorter_reversed = (sz_u8_t *)(buffer + bytes_per_diagonal * 3);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = (sz_u8_t)sz_init_score_linear(0, gap, 0);
    current_scores[0] = (sz_u8_t)sz_init_score_linear(1, gap, 0);
    current_scores[1] = (sz_u8_t)sz_init_score_linear(1, gap, 0);
    sz_u8_t last_score = current_scores[1];
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        sz_icelake_slice_u8_linear(first_reversed, longer, interior_length, previous_scores, current_scores,
                                   current_scores + 1, next_scores + 1, match, mismatch, gap);
        next_scores[0] = (sz_u8_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        next_scores[next_diagonal_length - 1] = (sz_u8_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        sz_icelake_slice_u8_linear(first_reversed, second_slice, interior_length, previous_scores, current_scores,
                                   current_scores + 1, next_scores, match, mismatch, gap);
        next_scores[next_diagonal_length - 1] = (sz_u8_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_icelake_slice_u8_linear(first_reversed, second_slice, next_diagonal_length, previous_scores, current_scores,
                                   current_scores + 1, next_scores, match, mismatch, gap);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        previous_scores++;
    }

    *result = last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

// Byte-level Levenshtein, u16 cells (32 lanes/ZMM).
SZ_INTERNAL sz_status_t szs_levenshtein_linear_u16_icelake_(sz_u8_t const *first, sz_size_t first_length,
                                                            sz_u8_t const *second, sz_size_t second_length,
                                                            szs_cell_costs_t const *costs, sz_memory_allocator_t *alloc,
                                                            sz_u16_t *result) {
    sz_error_cost_t const match = costs->match, mismatch = costs->mismatch, gap = costs->gap_open;
    if (first_length == 0 || second_length == 0) {
        sz_u16_t value = 0;
        if (first_length != 0) value = (sz_u16_t)((sz_i64_t)gap * (sz_i64_t)first_length);
        else if (second_length != 0) value = (sz_u16_t)((sz_i64_t)gap * (sz_i64_t)second_length);
        *result = value;
        return sz_success_k;
    }

    sz_u8_t const *shorter = first;
    sz_u8_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_u8_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const bytes_per_diagonal = sz_round_up_to_multiple(max_diagonal_length * sizeof(sz_u16_t), 64);
    sz_size_t const buffer_length = bytes_per_diagonal * 3 + shorter_length * sizeof(sz_u8_t);
    sz_u8_t *buffer = (sz_u8_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_u16_t *previous_scores = (sz_u16_t *)(buffer);
    sz_u16_t *current_scores = (sz_u16_t *)(buffer + bytes_per_diagonal);
    sz_u16_t *next_scores = (sz_u16_t *)(buffer + bytes_per_diagonal * 2);
    sz_u8_t *shorter_reversed = (sz_u8_t *)(buffer + bytes_per_diagonal * 3);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = (sz_u16_t)sz_init_score_linear(0, gap, 0);
    current_scores[0] = (sz_u16_t)sz_init_score_linear(1, gap, 0);
    current_scores[1] = (sz_u16_t)sz_init_score_linear(1, gap, 0);
    sz_u16_t last_score = current_scores[1];
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        sz_icelake_slice_u16_linear(first_reversed, longer, interior_length, previous_scores, current_scores,
                                    current_scores + 1, next_scores + 1, match, mismatch, gap);
        next_scores[0] = (sz_u16_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        next_scores[next_diagonal_length - 1] = (sz_u16_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        sz_icelake_slice_u16_linear(first_reversed, second_slice, interior_length, previous_scores, current_scores,
                                    current_scores + 1, next_scores, match, mismatch, gap);
        next_scores[next_diagonal_length - 1] = (sz_u16_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_icelake_slice_u16_linear(first_reversed, second_slice, next_diagonal_length, previous_scores, current_scores,
                                    current_scores + 1, next_scores, match, mismatch, gap);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        previous_scores++;
    }

    *result = last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

// Byte-level Levenshtein, u32 cells (16 lanes/ZMM).
SZ_INTERNAL sz_status_t szs_levenshtein_linear_u32_icelake_(sz_u8_t const *first, sz_size_t first_length,
                                                            sz_u8_t const *second, sz_size_t second_length,
                                                            szs_cell_costs_t const *costs, sz_memory_allocator_t *alloc,
                                                            sz_u32_t *result) {
    sz_error_cost_t const match = costs->match, mismatch = costs->mismatch, gap = costs->gap_open;
    if (first_length == 0 || second_length == 0) {
        sz_u32_t value = 0;
        if (first_length != 0) value = (sz_u32_t)((sz_i64_t)gap * (sz_i64_t)first_length);
        else if (second_length != 0) value = (sz_u32_t)((sz_i64_t)gap * (sz_i64_t)second_length);
        *result = value;
        return sz_success_k;
    }

    sz_u8_t const *shorter = first;
    sz_u8_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_u8_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const bytes_per_diagonal = sz_round_up_to_multiple(max_diagonal_length * sizeof(sz_u32_t), 64);
    sz_size_t const buffer_length = bytes_per_diagonal * 3 + shorter_length * sizeof(sz_u8_t);
    sz_u8_t *buffer = (sz_u8_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_u32_t *previous_scores = (sz_u32_t *)(buffer);
    sz_u32_t *current_scores = (sz_u32_t *)(buffer + bytes_per_diagonal);
    sz_u32_t *next_scores = (sz_u32_t *)(buffer + bytes_per_diagonal * 2);
    sz_u8_t *shorter_reversed = (sz_u8_t *)(buffer + bytes_per_diagonal * 3);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = (sz_u32_t)sz_init_score_linear(0, gap, 0);
    current_scores[0] = (sz_u32_t)sz_init_score_linear(1, gap, 0);
    current_scores[1] = (sz_u32_t)sz_init_score_linear(1, gap, 0);
    sz_u32_t last_score = current_scores[1];
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        sz_icelake_slice_u32_linear(first_reversed, longer, interior_length, previous_scores, current_scores,
                                    current_scores + 1, next_scores + 1, match, mismatch, gap);
        next_scores[0] = (sz_u32_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        next_scores[next_diagonal_length - 1] = (sz_u32_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        sz_icelake_slice_u32_linear(first_reversed, second_slice, interior_length, previous_scores, current_scores,
                                    current_scores + 1, next_scores, match, mismatch, gap);
        next_scores[next_diagonal_length - 1] = (sz_u32_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_icelake_slice_u32_linear(first_reversed, second_slice, next_diagonal_length, previous_scores, current_scores,
                                    current_scores + 1, next_scores, match, mismatch, gap);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        previous_scores++;
    }

    *result = last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

// Rune-level (UTF-8) Levenshtein, u8 cells.
SZ_INTERNAL sz_status_t szs_levenshtein_utf8_linear_u8_icelake_(sz_rune_t const *first, sz_size_t first_length,
                                                                sz_rune_t const *second, sz_size_t second_length,
                                                                szs_cell_costs_t const *costs,
                                                                sz_memory_allocator_t *alloc, sz_u8_t *result) {
    sz_error_cost_t const match = costs->match, mismatch = costs->mismatch, gap = costs->gap_open;
    if (first_length == 0 || second_length == 0) {
        sz_u8_t value = 0;
        if (first_length != 0) value = (sz_u8_t)((sz_i64_t)gap * (sz_i64_t)first_length);
        else if (second_length != 0) value = (sz_u8_t)((sz_i64_t)gap * (sz_i64_t)second_length);
        *result = value;
        return sz_success_k;
    }

    sz_rune_t const *shorter = first;
    sz_rune_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_rune_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const bytes_per_diagonal = sz_round_up_to_multiple(max_diagonal_length * sizeof(sz_u8_t), 64);
    sz_size_t const buffer_length = bytes_per_diagonal * 3 + shorter_length * sizeof(sz_rune_t);
    sz_u8_t *buffer = (sz_u8_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_u8_t *previous_scores = (sz_u8_t *)(buffer);
    sz_u8_t *current_scores = (sz_u8_t *)(buffer + bytes_per_diagonal);
    sz_u8_t *next_scores = (sz_u8_t *)(buffer + bytes_per_diagonal * 2);
    sz_rune_t *shorter_reversed = (sz_rune_t *)(buffer + bytes_per_diagonal * 3);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = (sz_u8_t)sz_init_score_linear(0, gap, 0);
    current_scores[0] = (sz_u8_t)sz_init_score_linear(1, gap, 0);
    current_scores[1] = (sz_u8_t)sz_init_score_linear(1, gap, 0);
    sz_u8_t last_score = current_scores[1];
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_rune_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        sz_icelake_slice_rune_u8_linear(first_reversed, longer, interior_length, previous_scores, current_scores,
                                        current_scores + 1, next_scores + 1, match, mismatch, gap);
        next_scores[0] = (sz_u8_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        next_scores[next_diagonal_length - 1] = (sz_u8_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_rune_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_rune_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        sz_icelake_slice_rune_u8_linear(first_reversed, second_slice, interior_length, previous_scores, current_scores,
                                        current_scores + 1, next_scores, match, mismatch, gap);
        next_scores[next_diagonal_length - 1] = (sz_u8_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_rune_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_rune_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_icelake_slice_rune_u8_linear(first_reversed, second_slice, next_diagonal_length, previous_scores,
                                        current_scores, current_scores + 1, next_scores, match, mismatch, gap);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        previous_scores++;
    }

    *result = last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

// Rune-level (UTF-8) Levenshtein, u16 cells.
SZ_INTERNAL sz_status_t szs_levenshtein_utf8_linear_u16_icelake_(sz_rune_t const *first, sz_size_t first_length,
                                                                 sz_rune_t const *second, sz_size_t second_length,
                                                                 szs_cell_costs_t const *costs,
                                                                 sz_memory_allocator_t *alloc, sz_u16_t *result) {
    sz_error_cost_t const match = costs->match, mismatch = costs->mismatch, gap = costs->gap_open;
    if (first_length == 0 || second_length == 0) {
        sz_u16_t value = 0;
        if (first_length != 0) value = (sz_u16_t)((sz_i64_t)gap * (sz_i64_t)first_length);
        else if (second_length != 0) value = (sz_u16_t)((sz_i64_t)gap * (sz_i64_t)second_length);
        *result = value;
        return sz_success_k;
    }

    sz_rune_t const *shorter = first;
    sz_rune_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_rune_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const bytes_per_diagonal = sz_round_up_to_multiple(max_diagonal_length * sizeof(sz_u16_t), 64);
    sz_size_t const buffer_length = bytes_per_diagonal * 3 + shorter_length * sizeof(sz_rune_t);
    sz_u8_t *buffer = (sz_u8_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_u16_t *previous_scores = (sz_u16_t *)(buffer);
    sz_u16_t *current_scores = (sz_u16_t *)(buffer + bytes_per_diagonal);
    sz_u16_t *next_scores = (sz_u16_t *)(buffer + bytes_per_diagonal * 2);
    sz_rune_t *shorter_reversed = (sz_rune_t *)(buffer + bytes_per_diagonal * 3);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = (sz_u16_t)sz_init_score_linear(0, gap, 0);
    current_scores[0] = (sz_u16_t)sz_init_score_linear(1, gap, 0);
    current_scores[1] = (sz_u16_t)sz_init_score_linear(1, gap, 0);
    sz_u16_t last_score = current_scores[1];
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_rune_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        sz_icelake_slice_rune_u16_linear(first_reversed, longer, interior_length, previous_scores, current_scores,
                                         current_scores + 1, next_scores + 1, match, mismatch, gap);
        next_scores[0] = (sz_u16_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        next_scores[next_diagonal_length - 1] = (sz_u16_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_rune_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_rune_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        sz_icelake_slice_rune_u16_linear(first_reversed, second_slice, interior_length, previous_scores, current_scores,
                                         current_scores + 1, next_scores, match, mismatch, gap);
        next_scores[next_diagonal_length - 1] = (sz_u16_t)sz_init_score_linear(next_diagonal_index, gap, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_rune_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_rune_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_icelake_slice_rune_u16_linear(first_reversed, second_slice, next_diagonal_length, previous_scores,
                                         current_scores, current_scores + 1, next_scores, match, mismatch, gap);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        previous_scores++;
    }

    *result = last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

#pragma endregion

#pragma region - Levenshtein Affine Diagonal Bodies

// The three bodies below are the explicit per-width Levenshtein affine-gap diagonal sweeps. They share
// the triangle/band/triangle anti-diagonal geometry with the linear bodies, but track seven diagonals
// instead of three: the main scores (previous/current/next) plus the running insert & delete tracks
// (current/next of each). Boundary init uses `sz_init_score_affine` / `sz_init_gap_affine`; the main
// scores rotate via `sz_rotate3_ptr`, while the two gap tracks swap current<->next. Each anti-diagonal
// interior is one AVX-512 `sz_icelake_slice_*_affine` call, bit-exact with the serial recurrence in
// `serial.h` (`sz_cell_affine_i32`): insert/delete tracks come from the up/left neighbors, the left-side
// pointers offset by one cell so both runs traverse identically.

// Byte-level Levenshtein, affine gaps, u8 cells (64 lanes/ZMM).
SZ_INTERNAL sz_status_t szs_levenshtein_affine_u8_icelake_(sz_u8_t const *first, sz_size_t first_length,
                                                           sz_u8_t const *second, sz_size_t second_length,
                                                           szs_cell_costs_t const *costs, sz_memory_allocator_t *alloc,
                                                           sz_u8_t *result) {
    sz_error_cost_t const match = costs->match, mismatch = costs->mismatch;
    sz_error_cost_t const open = costs->gap_open, extend = costs->gap_extend;
    if (first_length == 0 || second_length == 0) {
        sz_u8_t value = 0;
        if (first_length != 0) value = (sz_u8_t)((sz_i64_t)open + (sz_i64_t)extend * (sz_i64_t)(first_length - 1));
        else if (second_length != 0)
            value = (sz_u8_t)((sz_i64_t)open + (sz_i64_t)extend * (sz_i64_t)(second_length - 1));
        *result = value;
        return sz_success_k;
    }

    sz_u8_t const *shorter = first;
    sz_u8_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_u8_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const bytes_per_diagonal = sz_round_up_to_multiple(max_diagonal_length * sizeof(sz_u8_t), 64);
    sz_size_t const buffer_length = bytes_per_diagonal * 7 + shorter_length * sizeof(sz_u8_t);
    sz_u8_t *buffer = (sz_u8_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_u8_t *previous_scores = (sz_u8_t *)(buffer);
    sz_u8_t *current_scores = (sz_u8_t *)(buffer + bytes_per_diagonal);
    sz_u8_t *next_scores = (sz_u8_t *)(buffer + bytes_per_diagonal * 2);
    sz_u8_t *current_inserts = (sz_u8_t *)(buffer + bytes_per_diagonal * 3);
    sz_u8_t *next_inserts = (sz_u8_t *)(buffer + bytes_per_diagonal * 4);
    sz_u8_t *current_deletes = (sz_u8_t *)(buffer + bytes_per_diagonal * 5);
    sz_u8_t *next_deletes = (sz_u8_t *)(buffer + bytes_per_diagonal * 6);
    sz_u8_t *shorter_reversed = (sz_u8_t *)(buffer + bytes_per_diagonal * 7);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = (sz_u8_t)sz_init_score_affine(0, open, extend, 0);
    current_scores[0] = (sz_u8_t)sz_init_score_affine(1, open, extend, 0);
    current_scores[1] = (sz_u8_t)sz_init_score_affine(1, open, extend, 0);
    current_inserts[0] = (sz_u8_t)sz_init_gap_affine(1, open, extend, 0);
    current_deletes[1] = (sz_u8_t)sz_init_gap_affine(1, open, extend, 0);
    sz_u8_t last_score = current_scores[1];
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        sz_icelake_slice_u8_affine(first_reversed, longer, interior_length, previous_scores, current_scores,
                                   current_scores + 1, current_inserts, current_deletes + 1, next_scores + 1,
                                   next_inserts + 1, next_deletes + 1, match, mismatch, open, extend);
        next_scores[0] = (sz_u8_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_scores[next_diagonal_length - 1] = (sz_u8_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_inserts[0] = (sz_u8_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        next_deletes[next_diagonal_length - 1] = (sz_u8_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u8_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        sz_icelake_slice_u8_affine(first_reversed, second_slice, interior_length, previous_scores, current_scores,
                                   current_scores + 1, current_inserts, current_deletes + 1, next_scores, next_inserts,
                                   next_deletes, match, mismatch, open, extend);
        next_scores[next_diagonal_length - 1] = (sz_u8_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_deletes[next_diagonal_length - 1] = (sz_u8_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u8_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_icelake_slice_u8_affine(first_reversed, second_slice, next_diagonal_length, previous_scores, current_scores,
                                   current_scores + 1, current_inserts, current_deletes + 1, next_scores, next_inserts,
                                   next_deletes, match, mismatch, open, extend);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u8_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
        previous_scores++;
    }

    *result = last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

// Byte-level Levenshtein, affine gaps, u16 cells (32 lanes/ZMM).
SZ_INTERNAL sz_status_t szs_levenshtein_affine_u16_icelake_(sz_u8_t const *first, sz_size_t first_length,
                                                            sz_u8_t const *second, sz_size_t second_length,
                                                            szs_cell_costs_t const *costs, sz_memory_allocator_t *alloc,
                                                            sz_u16_t *result) {
    sz_error_cost_t const match = costs->match, mismatch = costs->mismatch;
    sz_error_cost_t const open = costs->gap_open, extend = costs->gap_extend;
    if (first_length == 0 || second_length == 0) {
        sz_u16_t value = 0;
        if (first_length != 0) value = (sz_u16_t)((sz_i64_t)open + (sz_i64_t)extend * (sz_i64_t)(first_length - 1));
        else if (second_length != 0)
            value = (sz_u16_t)((sz_i64_t)open + (sz_i64_t)extend * (sz_i64_t)(second_length - 1));
        *result = value;
        return sz_success_k;
    }

    sz_u8_t const *shorter = first;
    sz_u8_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_u8_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const bytes_per_diagonal = sz_round_up_to_multiple(max_diagonal_length * sizeof(sz_u16_t), 64);
    sz_size_t const buffer_length = bytes_per_diagonal * 7 + shorter_length * sizeof(sz_u8_t);
    sz_u8_t *buffer = (sz_u8_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_u16_t *previous_scores = (sz_u16_t *)(buffer);
    sz_u16_t *current_scores = (sz_u16_t *)(buffer + bytes_per_diagonal);
    sz_u16_t *next_scores = (sz_u16_t *)(buffer + bytes_per_diagonal * 2);
    sz_u16_t *current_inserts = (sz_u16_t *)(buffer + bytes_per_diagonal * 3);
    sz_u16_t *next_inserts = (sz_u16_t *)(buffer + bytes_per_diagonal * 4);
    sz_u16_t *current_deletes = (sz_u16_t *)(buffer + bytes_per_diagonal * 5);
    sz_u16_t *next_deletes = (sz_u16_t *)(buffer + bytes_per_diagonal * 6);
    sz_u8_t *shorter_reversed = (sz_u8_t *)(buffer + bytes_per_diagonal * 7);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = (sz_u16_t)sz_init_score_affine(0, open, extend, 0);
    current_scores[0] = (sz_u16_t)sz_init_score_affine(1, open, extend, 0);
    current_scores[1] = (sz_u16_t)sz_init_score_affine(1, open, extend, 0);
    current_inserts[0] = (sz_u16_t)sz_init_gap_affine(1, open, extend, 0);
    current_deletes[1] = (sz_u16_t)sz_init_gap_affine(1, open, extend, 0);
    sz_u16_t last_score = current_scores[1];
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        sz_icelake_slice_u16_affine(first_reversed, longer, interior_length, previous_scores, current_scores,
                                    current_scores + 1, current_inserts, current_deletes + 1, next_scores + 1,
                                    next_inserts + 1, next_deletes + 1, match, mismatch, open, extend);
        next_scores[0] = (sz_u16_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_scores[next_diagonal_length - 1] = (sz_u16_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_inserts[0] = (sz_u16_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        next_deletes[next_diagonal_length - 1] = (sz_u16_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u16_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        sz_icelake_slice_u16_affine(first_reversed, second_slice, interior_length, previous_scores, current_scores,
                                    current_scores + 1, current_inserts, current_deletes + 1, next_scores, next_inserts,
                                    next_deletes, match, mismatch, open, extend);
        next_scores[next_diagonal_length - 1] = (sz_u16_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_deletes[next_diagonal_length - 1] = (sz_u16_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u16_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_icelake_slice_u16_affine(first_reversed, second_slice, next_diagonal_length, previous_scores, current_scores,
                                    current_scores + 1, current_inserts, current_deletes + 1, next_scores, next_inserts,
                                    next_deletes, match, mismatch, open, extend);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u16_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
        previous_scores++;
    }

    *result = last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

// Byte-level Levenshtein, affine gaps, u32 cells (16 lanes/ZMM).
SZ_INTERNAL sz_status_t szs_levenshtein_affine_u32_icelake_(sz_u8_t const *first, sz_size_t first_length,
                                                            sz_u8_t const *second, sz_size_t second_length,
                                                            szs_cell_costs_t const *costs, sz_memory_allocator_t *alloc,
                                                            sz_u32_t *result) {
    sz_error_cost_t const match = costs->match, mismatch = costs->mismatch;
    sz_error_cost_t const open = costs->gap_open, extend = costs->gap_extend;
    if (first_length == 0 || second_length == 0) {
        sz_u32_t value = 0;
        if (first_length != 0) value = (sz_u32_t)((sz_i64_t)open + (sz_i64_t)extend * (sz_i64_t)(first_length - 1));
        else if (second_length != 0)
            value = (sz_u32_t)((sz_i64_t)open + (sz_i64_t)extend * (sz_i64_t)(second_length - 1));
        *result = value;
        return sz_success_k;
    }

    sz_u8_t const *shorter = first;
    sz_u8_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_u8_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const bytes_per_diagonal = sz_round_up_to_multiple(max_diagonal_length * sizeof(sz_u32_t), 64);
    sz_size_t const buffer_length = bytes_per_diagonal * 7 + shorter_length * sizeof(sz_u8_t);
    sz_u8_t *buffer = (sz_u8_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_u32_t *previous_scores = (sz_u32_t *)(buffer);
    sz_u32_t *current_scores = (sz_u32_t *)(buffer + bytes_per_diagonal);
    sz_u32_t *next_scores = (sz_u32_t *)(buffer + bytes_per_diagonal * 2);
    sz_u32_t *current_inserts = (sz_u32_t *)(buffer + bytes_per_diagonal * 3);
    sz_u32_t *next_inserts = (sz_u32_t *)(buffer + bytes_per_diagonal * 4);
    sz_u32_t *current_deletes = (sz_u32_t *)(buffer + bytes_per_diagonal * 5);
    sz_u32_t *next_deletes = (sz_u32_t *)(buffer + bytes_per_diagonal * 6);
    sz_u8_t *shorter_reversed = (sz_u8_t *)(buffer + bytes_per_diagonal * 7);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = (sz_u32_t)sz_init_score_affine(0, open, extend, 0);
    current_scores[0] = (sz_u32_t)sz_init_score_affine(1, open, extend, 0);
    current_scores[1] = (sz_u32_t)sz_init_score_affine(1, open, extend, 0);
    current_inserts[0] = (sz_u32_t)sz_init_gap_affine(1, open, extend, 0);
    current_deletes[1] = (sz_u32_t)sz_init_gap_affine(1, open, extend, 0);
    sz_u32_t last_score = current_scores[1];
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        sz_icelake_slice_u32_affine(first_reversed, longer, interior_length, previous_scores, current_scores,
                                    current_scores + 1, current_inserts, current_deletes + 1, next_scores + 1,
                                    next_inserts + 1, next_deletes + 1, match, mismatch, open, extend);
        next_scores[0] = (sz_u32_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_scores[next_diagonal_length - 1] = (sz_u32_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_inserts[0] = (sz_u32_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        next_deletes[next_diagonal_length - 1] = (sz_u32_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u32_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        sz_icelake_slice_u32_affine(first_reversed, second_slice, interior_length, previous_scores, current_scores,
                                    current_scores + 1, current_inserts, current_deletes + 1, next_scores, next_inserts,
                                    next_deletes, match, mismatch, open, extend);
        next_scores[next_diagonal_length - 1] = (sz_u32_t)sz_init_score_affine(next_diagonal_index, open, extend, 0);
        next_deletes[next_diagonal_length - 1] = (sz_u32_t)sz_init_gap_affine(next_diagonal_index, open, extend, 0);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u32_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u8_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u8_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_icelake_slice_u32_affine(first_reversed, second_slice, next_diagonal_length, previous_scores, current_scores,
                                    current_scores + 1, current_inserts, current_deletes + 1, next_scores, next_inserts,
                                    next_deletes, match, mismatch, open, extend);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_u32_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
        previous_scores++;
    }

    *result = last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

#pragma endregion

#pragma region - Per-Op Dispatch Entries

// NOTE: the Levenshtein Ice Lake bodies below call AVX-512 slice kernels, so the per-pair body
// functions must stay INSIDE the target attribute region; the target pragma is popped only after
// them, before the NW/SW delegators (which route to the serial backend).

// Byte-level Levenshtein per-pair body: linear- or affine-gap width-dispatch into the narrow Ice Lake
// bodies, the widest cell width falling through to the serial i32 diagonal body.
static sz_status_t szs_levenshtein_icelake_body_(void *ctx_punned, sz_size_t i) {
    szs_pairwise_ctx_t *ctx = (szs_pairwise_ctx_t *)ctx_punned;
    sz_memory_allocator_t *alloc = ctx->alloc;
    szs_pair_spans_t pair = szs_inputs_pair_(&ctx->inputs, i);
    szs_cell_costs_t const *costs = &ctx->costs;

    sz_size_t const sub_magnitude = sz_max_of_two(
        (sz_size_t)(costs->match < 0 ? -costs->match : costs->match),
        (sz_size_t)(costs->mismatch < 0 ? -costs->mismatch : costs->mismatch));
    sz_size_t const gap_magnitude = sz_max_of_two(
        (sz_size_t)(costs->gap_open < 0 ? -costs->gap_open : costs->gap_open),
        (sz_size_t)(costs->gap_extend < 0 ? -costs->gap_extend : costs->gap_extend));
    sz_similarity_gaps_t const gap_mode = ctx->linear ? sz_gaps_linear_k : sz_gaps_affine_k;
    sz_dp_memory_t mem = sz_dp_memory(pair.a_len, pair.b_len, gap_mode, sub_magnitude, gap_magnitude, 1,
                                      SZ_MAX_REGISTER_WIDTH, 0);
    sz_u8_t const *ap = (sz_u8_t const *)pair.a_ptr, *bp = (sz_u8_t const *)pair.b_ptr;

    sz_size_t result = 0;
    sz_status_t status = sz_success_k;
    if (mem.bytes_per_cell == sz_one_byte_per_cell_k) {
        sz_u8_t cell = 0;
        status = ctx->linear ? szs_levenshtein_linear_u8_icelake_(ap, pair.a_len, bp, pair.b_len, costs, alloc, &cell)
                             : szs_levenshtein_affine_u8_icelake_(ap, pair.a_len, bp, pair.b_len, costs, alloc, &cell);
        result = cell;
    }
    else if (mem.bytes_per_cell == sz_two_bytes_per_cell_k) {
        sz_u16_t cell = 0;
        status = ctx->linear ? szs_levenshtein_linear_u16_icelake_(ap, pair.a_len, bp, pair.b_len, costs, alloc, &cell)
                             : szs_levenshtein_affine_u16_icelake_(ap, pair.a_len, bp, pair.b_len, costs, alloc, &cell);
        result = cell;
    }
    else if (mem.bytes_per_cell == sz_four_bytes_per_cell_k) {
        sz_u32_t cell = 0;
        status = ctx->linear ? szs_levenshtein_linear_u32_icelake_(ap, pair.a_len, bp, pair.b_len, costs, alloc, &cell)
                             : szs_levenshtein_affine_u32_icelake_(ap, pair.a_len, bp, pair.b_len, costs, alloc, &cell);
        result = cell;
    }
    else {
        // The widest cell width (and the empty-input zero-width case, where a narrow body would
        // truncate the boundary distance) have no SIMD specialization; widen to u32 and run the
        // serial i32 body, whose result fits the public `sz_size_t` slot untruncated.
        sz_size_t const widen_capacity = pair.a_len + pair.b_len;
        sz_u32_t *elements = (sz_u32_t *)alloc->allocate((widen_capacity ? widen_capacity : 1) * sizeof(sz_u32_t),
                                                         alloc->handle);
        if (!elements) {
            if (ctx->error) *ctx->error = "Levenshtein element buffer allocation failed";
            return sz_bad_alloc_k;
        }
        sz_u32_t *a_elements = elements;
        sz_u32_t *b_elements = elements + pair.a_len;
        for (sz_size_t k = 0; k < pair.a_len; ++k) a_elements[k] = ap[k];
        for (sz_size_t k = 0; k < pair.b_len; ++k) b_elements[k] = bp[k];
        sz_i32_t score = 0;
        status = ctx->linear
                     ? szs_diagonal_linear_serial_(a_elements, pair.a_len, b_elements, pair.b_len, costs, alloc, &score)
                     : szs_diagonal_affine_serial_(a_elements, pair.a_len, b_elements, pair.b_len, costs, alloc,
                                                   &score);
        alloc->free(elements, (widen_capacity ? widen_capacity : 1) * sizeof(sz_u32_t), alloc->handle);
        result = (sz_size_t)(sz_u32_t)score;
    }
    if (status != sz_success_k) {
        if (ctx->error) *ctx->error = "Levenshtein Ice Lake kernel failed";
        return status;
    }
    *(sz_size_t *)((sz_u8_t *)ctx->results + i * ctx->results_stride) = result;
    return sz_success_k;
}

// Rune-level (UTF-8) Levenshtein per-pair body: same width-dispatch, decoding the inputs to runes
// first and routing the widest cell width to the serial i32 diagonal body.
static sz_status_t szs_levenshtein_utf8_icelake_body_(void *ctx_punned, sz_size_t i) {
    szs_pairwise_ctx_t *ctx = (szs_pairwise_ctx_t *)ctx_punned;
    sz_memory_allocator_t *alloc = ctx->alloc;
    szs_pair_spans_t pair = szs_inputs_pair_(&ctx->inputs, i);
    szs_cell_costs_t const *costs = &ctx->costs;

    sz_size_t const widen_capacity = pair.a_len + pair.b_len;
    sz_rune_t *runes = (sz_rune_t *)alloc->allocate((widen_capacity ? widen_capacity : 1) * sizeof(sz_rune_t),
                                                    alloc->handle);
    if (!runes) {
        if (ctx->error) *ctx->error = "Levenshtein UTF-8 rune buffer allocation failed";
        return sz_bad_alloc_k;
    }
    sz_rune_t *a_runes = runes;
    sz_rune_t *b_runes = runes + pair.a_len;
    sz_size_t a_count = 0, b_count = 0;
    if (!sz_utf8_to_runes(pair.a_ptr, pair.a_len, a_runes, &a_count) ||
        !sz_utf8_to_runes(pair.b_ptr, pair.b_len, b_runes, &b_count)) {
        alloc->free(runes, (widen_capacity ? widen_capacity : 1) * sizeof(sz_rune_t), alloc->handle);
        if (ctx->error) *ctx->error = "Invalid UTF-8 input";
        return sz_invalid_utf8_k;
    }

    sz_size_t const sub_magnitude = sz_max_of_two(
        (sz_size_t)(costs->match < 0 ? -costs->match : costs->match),
        (sz_size_t)(costs->mismatch < 0 ? -costs->mismatch : costs->mismatch));
    sz_size_t const gap_magnitude = (sz_size_t)(costs->gap_open < 0 ? -costs->gap_open : costs->gap_open);
    sz_dp_memory_t mem = sz_dp_memory(pair.a_len, pair.b_len, sz_gaps_linear_k, sub_magnitude, gap_magnitude,
                                      sizeof(sz_rune_t), SZ_MAX_REGISTER_WIDTH, 0);

    sz_size_t result = 0;
    sz_status_t status = sz_success_k;
    if (mem.bytes_per_cell == sz_one_byte_per_cell_k) {
        sz_u8_t cell = 0;
        status = szs_levenshtein_utf8_linear_u8_icelake_(a_runes, a_count, b_runes, b_count, costs, alloc, &cell);
        result = cell;
    }
    else if (mem.bytes_per_cell == sz_two_bytes_per_cell_k) {
        sz_u16_t cell = 0;
        status = szs_levenshtein_utf8_linear_u16_icelake_(a_runes, a_count, b_runes, b_count, costs, alloc, &cell);
        result = cell;
    }
    else {
        // The wider cell widths, and the empty-input zero-width case (a narrow body would truncate the
        // boundary distance), have no rune SIMD specialization; run the serial i32 diagonal body.
        sz_i32_t score = 0;
        status = szs_diagonal_linear_serial_(a_runes, a_count, b_runes, b_count, costs, alloc, &score);
        result = (sz_size_t)(sz_u32_t)score;
    }

    alloc->free(runes, (widen_capacity ? widen_capacity : 1) * sizeof(sz_rune_t), alloc->handle);
    if (status != sz_success_k) {
        if (ctx->error) *ctx->error = "Levenshtein UTF-8 Ice Lake kernel failed";
        return status;
    }
    *(sz_size_t *)((sz_u8_t *)ctx->results + i * ctx->results_stride) = result;
    return sz_success_k;
}

SZ_INTERNAL sz_status_t szs_levenshtein_icelake(szs_levenshtein_engine_t const *engine,
                                                szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                sz_size_t *results, sz_size_t results_stride, char const **error) {
    // Both linear and affine gaps have byte-level Ice Lake specializations; `szs_levenshtein_preset_` sets
    // `ctx.linear` so the per-pair body width-dispatches into the matching narrow kernel.
    sz_memory_allocator_t fallback_alloc;
    szs_pairwise_ctx_t ctx;
    ctx.inputs = inputs;
    ctx.alloc = szs_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc, &fallback_alloc);
    szs_levenshtein_preset_(&ctx, engine, /*decode_runes*/ 0);
    ctx.results = results, ctx.results_stride = results_stride, ctx.error = error;
    return szs_parallel_for_(device, szs_inputs_count_(&inputs), &szs_levenshtein_icelake_body_, &ctx);
}

SZ_INTERNAL sz_status_t szs_levenshtein_utf8_icelake(szs_levenshtein_engine_t const *engine,
                                                     szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                     sz_size_t *results, sz_size_t results_stride, char const **error) {
    // The C++ Ice Lake engine specialized only the linear-gap rune walker; affine UTF-8 has no rune
    // SIMD path, so an affine pass routes to the serial backend (bit-exact with the same recurrence).
    if (engine->gap_mode == sz_gaps_affine_k && engine->gaps.open != engine->gaps.extend)
        return szs_levenshtein_utf8_serial(engine, device, inputs, results, results_stride, error);
    sz_memory_allocator_t fallback_alloc;
    szs_pairwise_ctx_t ctx;
    ctx.inputs = inputs;
    ctx.alloc = szs_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc, &fallback_alloc);
    szs_levenshtein_preset_(&ctx, engine, /*decode_runes*/ 1);
    ctx.results = results, ctx.results_stride = results_stride, ctx.error = error;
    return szs_parallel_for_(device, szs_inputs_count_(&inputs), &szs_levenshtein_utf8_icelake_body_, &ctx);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

// The C++ Ice Lake engine had no NW/SW or affine specialization; the serial recurrence is identical
// and bit-exact, so both alignment scores route straight through to the serial backend.
SZ_INTERNAL sz_status_t szs_needleman_wunsch_icelake(szs_needleman_wunsch_engine_t const *engine,
                                                     szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                     sz_ssize_t *results, sz_size_t results_stride,
                                                     char const **error) {
    return szs_needleman_wunsch_serial(engine, device, inputs, results, results_stride, error);
}

SZ_INTERNAL sz_status_t szs_smith_waterman_icelake(szs_smith_waterman_engine_t const *engine,
                                                   szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                   sz_ssize_t *results, sz_size_t results_stride, char const **error) {
    return szs_smith_waterman_serial(engine, device, inputs, results, results_stride, error);
}

#pragma endregion

#pragma endregion // Ice Lake Implementation
#endif            // SZ_USE_ICELAKE

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLAS_SIMILARITIES_ICELAKE_H_
