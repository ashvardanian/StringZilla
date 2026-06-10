/**
 *  @brief AVX2 (Haswell) string-similarity backend.
 *  @file include/stringzillas/similarities/haswell.hpp
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 *  @sa include/stringzillas/similarities/icelake.hpp
 */
#ifndef STRINGZILLAS_SIMILARITIES_HASWELL_HPP_
#define STRINGZILLAS_SIMILARITIES_HASWELL_HPP_

#include "stringzillas/similarities/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

/*  AVX2 implementation of the string similarity algorithms for Haswell processors and newer.
 *  Mirrors the Ice Lake backend, but without K-mask registers or `VPERMB`: a full 256-entry table
 *  lookup is emulated with high-nibble-selected `VPSHUFB` blends, and the 32-entry cost row with two.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "bmi", "bmi2")
#endif

/** @brief Serial fallback combined with the Haswell AVX2 backend, mirroring `sz_caps_sil_k` for Ice Lake. */
static constexpr sz_capability_t sz_caps_sh_k = (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k);

/**
 *  @brief Helper object optimizing the most expensive part of class-based variable-substitution-cost
 *         alignment methods for Haswell CPUs. It's designed for horizontal layout "walkers", where we
 *         look at just one row of the (32 x 32) `class_substitution_costs` matrix while mapping each input
 *         byte to one of the @b `error_costs_classes_count_k` (32) classes via the `byte_to_class[256]` map.
 *
 *  AVX2 has no `VPERMB`, so a 256-entry table lookup is emulated in two stages per block of 32 input bytes:
 *  1. Map byte to class: `VPSHUFB` only addresses 16 bytes per 128-bit lane, so we shuffle each of the 16
 *     high-nibble groups of `byte_to_class` by the low nibble and blend the group whose high nibble matches.
 *  2. Map class to substitution cost: shuffle the low and high 16 entries of the cost row by the class and
 *     blend by whether the class is below or above @b 16.
 *
 *  This is a common abstraction for both:
 *  - Local SW and global NW alignment.
 *  - Serial and parallel implementations.
 *  - 8-bit, 16-bit, and 32-bit costs.
 *  - Any memory allocator used.
 */
struct lookup_in256classes_haswell_t_ {
    sz_u256_vec_t byte_to_class_group_vecs_[16]; // ! Each holds one 16-byte group broadcast to both lanes
    sz_u256_vec_t row_subs_low_vec_, row_subs_high_vec_;
    sz_u256_vec_t low_nibble_mask_vec_;

    inline lookup_in256classes_haswell_t_() noexcept { low_nibble_mask_vec_.ymm = _mm256_set1_epi8(0x0f); }

    inline void reload_classes(sz_u8_t const *byte_to_class) noexcept {
        for (int group = 0; group != 16; ++group) {
            __m128i group_xmm = _mm_loadu_si128((__m128i const *)(byte_to_class + group * 16));
            byte_to_class_group_vecs_[group].ymm = _mm256_set_m128i(group_xmm, group_xmm);
        }
    }

    inline void reload_row(sz_error_cost_t const *row_subs) noexcept {
        // The cost row holds only `error_costs_classes_count_k` (32) entries, split into two 16-byte halves
        // broadcast to both lanes so the second-stage `VPSHUFB` works regardless of which lane a class lands in.
        __m128i low_xmm = _mm_loadu_si128((__m128i const *)(row_subs + 0));
        __m128i high_xmm = _mm_loadu_si128((__m128i const *)(row_subs + 16));
        row_subs_low_vec_.ymm = _mm256_set_m128i(low_xmm, low_xmm);
        row_subs_high_vec_.ymm = _mm256_set_m128i(high_xmm, high_xmm);
    }

    inline sz_u256_vec_t lookup32(sz_u256_vec_t const &text_vec) const noexcept {

        sz_u256_vec_t low_nibbles_vec, high_nibbles_vec;
        sz_u256_vec_t class_vec, substituted_vec;

        // First stage: map each input byte to its class. `VPSHUFB` ignores the upper nibble of the index, so
        // we shuffle every high-nibble group by the low nibble and keep the group whose high nibble matches.
        low_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, low_nibble_mask_vec_.ymm);
        // There is no `_mm256_srli_epi8` intrinsic, so we shift as 16-bit lanes and mask off the carried bits.
        high_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), low_nibble_mask_vec_.ymm);

        class_vec.ymm = _mm256_setzero_si256();
        for (int group = 0; group != 16; ++group) {
            __m256i shuffled = _mm256_shuffle_epi8(byte_to_class_group_vecs_[group].ymm, low_nibbles_vec.ymm);
            __m256i is_group = _mm256_cmpeq_epi8(high_nibbles_vec.ymm, _mm256_set1_epi8((char)group));
            class_vec.ymm = _mm256_blendv_epi8(class_vec.ymm, shuffled, is_group);
        }

        // Second stage: map each class to its substitution cost. `VPSHUFB` uses the low 4 bits of the class,
        // so the low-half shuffle is valid for classes below 16 and the high-half for the rest.
        __m256i cost_if_low = _mm256_shuffle_epi8(row_subs_low_vec_.ymm, class_vec.ymm);
        __m256i cost_if_high = _mm256_shuffle_epi8(row_subs_high_vec_.ymm, class_vec.ymm);
        __m256i is_high_class = _mm256_cmpgt_epi8(class_vec.ymm, _mm256_set1_epi8(15));
        substituted_vec.ymm = _mm256_blendv_epi8(cost_if_low, cost_if_high, is_high_class);
        return substituted_vec;
    }
};

/**
 *  @brief Helper object for Haswell CPUs, designed for @b diagonal layout "walkers", where @b both operands
 *         of every substitution vary per cell, so a single resident cost row is not enough and the entire
 *         (32 x 32) `class_substitution_costs` matrix must stay resident.
 *
 *  AVX2 has neither `VPERMB` nor mask registers, so both stages of the Ice Lake helper are emulated with
 *  in-128-bit-lane `_mm256_shuffle_epi8` (VPSHUFB) and high-bit `_mm256_blendv_epi8` blends:
 *  1. Byte to class: shared with the horizontal helper, the 256-entry `byte_to_class` table is addressed via
 *     16 low-nibble shuffles blended by the high nibble (see `classify32`).
 *  2. Two varying class operands to cost: the (32 x 32) matrix is kept resident as 32 cost rows, each row split
 *     into a low (classes 0..15) and a high (classes 16..31) 16-byte half broadcast to both lanes. The second
 *     class indexes each row's halves with one `VPSHUFB`; a balanced tree of blends over the 5 bits of the first
 *     class then selects the matching row, and a final blend on `second_class >= 16` joins the two halves. This
 *     costs @b 64 `VPSHUFB` + @b 64 blends per 32 cells, versus Ice Lake's 16 `VPERMB` + 15 blends per 64 cells.
 *
 *  This is a common abstraction for both:
 *  - Local SW and global NW alignment.
 *  - Serial and parallel implementations.
 *  - 16-bit and 32-bit costs.
 *  - Any memory allocator used.
 */
struct substitution_matrix_lookup_haswell_t_ {
    sz_u256_vec_t byte_to_class_group_vecs_[16]; // ! Each holds one 16-byte group broadcast to both lanes
    sz_u256_vec_t cost_rows_low_vecs_[error_costs_classes_count_k], cost_rows_high_vecs_[error_costs_classes_count_k];
    sz_u256_vec_t low_nibble_mask_vec_;

    inline substitution_matrix_lookup_haswell_t_() noexcept { low_nibble_mask_vec_.ymm = _mm256_set1_epi8(0x0f); }

    inline void reload_classes(sz_u8_t const *byte_to_class) noexcept {
        for (int group = 0; group != 16; ++group) {
            __m128i group_xmm = _mm_loadu_si128((__m128i const *)(byte_to_class + group * 16));
            byte_to_class_group_vecs_[group].ymm = _mm256_set_m128i(group_xmm, group_xmm);
        }
    }

    /**
     *  @brief Keeps the (32 x 32) class cost matrix resident as 32 low/high row halves for `lookup32`.
     *  @param transpose When set, the matrix is loaded as @b transposed, so that `lookup32(first, second)`
     *         still returns the original `class_substitution_costs[first][second]` even when the diagonal walker
     *         has swapped the shorter and longer strings (which flips the order of the two class operands).
     */
    inline void reload_costs(
        error_cost_t const (&class_substitution_costs)[error_costs_classes_count_k][error_costs_classes_count_k],
        bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class) {
            error_cost_t row[error_costs_classes_count_k];
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                row[second_class] = transpose ? class_substitution_costs[second_class][first_class]
                                              : class_substitution_costs[first_class][second_class];
            __m128i low_xmm = _mm_loadu_si128((__m128i const *)(row + 0));
            __m128i high_xmm = _mm_loadu_si128((__m128i const *)(row + 16));
            cost_rows_low_vecs_[first_class].ymm = _mm256_set_m128i(low_xmm, low_xmm);
            cost_rows_high_vecs_[first_class].ymm = _mm256_set_m128i(high_xmm, high_xmm);
        }
    }

    inline sz_u256_vec_t classify32(sz_u256_vec_t const &text_vec) const noexcept {

        sz_u256_vec_t low_nibbles_vec, high_nibbles_vec, class_vec;

        // Map each input byte to its class. `VPSHUFB` ignores the upper nibble of the index, so we shuffle every
        // high-nibble group by the low nibble and keep the group whose high nibble matches.
        low_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, low_nibble_mask_vec_.ymm);
        high_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), low_nibble_mask_vec_.ymm);

        class_vec.ymm = _mm256_setzero_si256();
        for (int group = 0; group != 16; ++group) {
            __m256i shuffled = _mm256_shuffle_epi8(byte_to_class_group_vecs_[group].ymm, low_nibbles_vec.ymm);
            __m256i is_group = _mm256_cmpeq_epi8(high_nibbles_vec.ymm, _mm256_set1_epi8((char)group));
            class_vec.ymm = _mm256_blendv_epi8(class_vec.ymm, shuffled, is_group);
        }
        return class_vec;
    }

    inline sz_u256_vec_t lookup32(sz_u256_vec_t const &first_class_vec,
                                  sz_u256_vec_t const &second_class_vec) const noexcept {

        sz_u256_vec_t cost_low_vecs[error_costs_classes_count_k], cost_high_vecs[error_costs_classes_count_k];
        sz_u256_vec_t substituted_vec;

        // `VPSHUFB` only looks at the low 4 bits of the index, so the same `second_class` byte indexes each row's
        // low half (classes 0..15) and high half (classes 16..31). Both are resolved per row up front.
        __m256i const second_idx = second_class_vec.ymm;
        for (size_t row = 0; row != error_costs_classes_count_k; ++row) {
            cost_low_vecs[row].ymm = _mm256_shuffle_epi8(cost_rows_low_vecs_[row].ymm, second_idx);
            cost_high_vecs[row].ymm = _mm256_shuffle_epi8(cost_rows_high_vecs_[row].ymm, second_idx);
        }

        // Select the matching row per lane via a balanced tree of blends over the 5 bits of the first class.
        // `_mm256_blendv_epi8` keeps the second argument where the mask byte's high bit is set, and `cmpeq`
        // produces an all-ones byte exactly where the tested first-class bit is set.
        for (int bit = 0; bit != 5; ++bit) {
            size_t const survivors = (size_t)error_costs_classes_count_k >> (bit + 1);
            __m256i const bit_value = _mm256_set1_epi8((char)(1 << bit));
            __m256i const is_bit_set = _mm256_cmpeq_epi8(_mm256_and_si256(first_class_vec.ymm, bit_value), bit_value);
            for (size_t pair = 0; pair != survivors; ++pair) {
                cost_low_vecs[pair].ymm = _mm256_blendv_epi8(cost_low_vecs[2 * pair].ymm,
                                                             cost_low_vecs[2 * pair + 1].ymm, is_bit_set);
                cost_high_vecs[pair].ymm = _mm256_blendv_epi8(cost_high_vecs[2 * pair].ymm,
                                                              cost_high_vecs[2 * pair + 1].ymm, is_bit_set);
            }
        }

        // Join the two halves: classes below 16 take the low shuffle, the rest take the high shuffle.
        __m256i const is_high_class = _mm256_cmpgt_epi8(second_class_vec.ymm, _mm256_set1_epi8(15));
        substituted_vec.ymm = _mm256_blendv_epi8(cost_low_vecs[0].ymm, cost_high_vecs[0].ymm, is_high_class);
        return substituted_vec;
    }
};

/**
 *  @brief Helper object for Haswell CPUs, designed for horizontal layout "walkers", operating over 16-bit costs.
 *         It's based on the idea, that substitutions are the most expensive part of the algorithm, so those are
 *         parallelized, while the running minimums within a row are computed in a serial fashion.
 *
 *  This is a common abstraction for both:
 *  - Local SW and global NW alignment.
 *  - Serial and parallel implementations.
 *  - Any memory allocator used.
 */
template <sz_similarity_locality_t locality_>
struct tile_scorer<constant_iterator<char>, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_haswell_k>
    : public tile_scorer<constant_iterator<char>, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    lookup_in256classes_haswell_t_ lookup_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                   //
        constant_iterator<char> first_char, char const *second_slice, size_t n,        //
        sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
        sz_i16_t const *scores_pre_deletion, sz_i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // Load a new substitution row, addressed by the class of the current first character.
        sz_i16_t const gap = static_cast<sz_i16_t>(this->gap_costs_.open_or_extend);
        sz_u8_t const first_class = this->substituter_.byte_to_class[(sz_u8_t)*first_char];
        lookup_.reload_classes(this->substituter_.byte_to_class);
        lookup_.reload_row(&this->substituter_.class_substitution_costs[first_class][0]);

        // Progress through the row 32 characters at a time.
        size_t const count_slices = n / 32;
        executor.for_n(count_slices, [&](size_t idx_slice) noexcept {
            slice_32chars(second_slice, idx_slice * 32, gap, scores_pre_substitution, scores_pre_insertion, scores_new);
        });

        // Handle the tail with a less efficient scalar kernel.
        for (size_t i = count_slices * 32; i < n; ++i)
            slice_1char(second_slice, i, gap, scores_pre_substitution, scores_pre_insertion, scores_new);

        // Horizontally propagate the deletion cost across the last row, just like the serial scorer.
        sz_assert_(scores_pre_substitution + 1 == scores_pre_insertion && "Expects horizontal traversal of DP matrix");
        sz_assert_(scores_pre_deletion + 1 == scores_new && "Expects horizontal traversal of DP matrix");
        sz_i16_t last_in_row = scores_pre_deletion[0];
        if constexpr (locality_ == sz_similarity_global_k) {
            for (size_t i = 0; i < n; ++i)
                scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
            this->last_score_ = last_in_row;
        }
        else {
            // In Local Alignment for SW the deletion still propagates horizontally, but every cell stays
            // non-negative and the running best across the whole matrix is the reported score.
            sz_i16_t row_best = this->best_score_;
            for (size_t i = 0; i < n; ++i) {
                scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
                row_best = sz_max_of_two(row_best, scores_new[i]);
            }
            this->best_score_ = row_best;
        }
    }

    void slice_32chars(char const *second_slice, size_t i, sz_i16_t gap,                              //
                       sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
                       sz_i16_t *scores_new) const noexcept {

        sz_u256_vec_t second_vec;
        sz_u256_vec_t pre_substitution_vecs[2], pre_gap_vecs[2];
        sz_u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        sz_u256_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        // Initialize constats:
        sz_u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi16(gap);

        // Load the data without any masks:
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)(second_slice + i));
        pre_substitution_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + i + 0));
        pre_substitution_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + i + 16));
        pre_gap_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + i + 0));
        pre_gap_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + i + 16));

        // First, sign-extend the substitution cost vector.
        cost_of_substitution_i8_vec = lookup_.lookup32(second_vec);
        cost_of_substitution_i16_vecs[0].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0));
        cost_of_substitution_i16_vecs[1].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1));

        // Then compute the data-parallel part, assuming the cost of deletions will be propagated
        // left to right outside of this loop.
        cost_if_substitution_vecs[0].ymm = _mm256_add_epi16(pre_substitution_vecs[0].ymm,
                                                            cost_of_substitution_i16_vecs[0].ymm);
        cost_if_substitution_vecs[1].ymm = _mm256_add_epi16(pre_substitution_vecs[1].ymm,
                                                            cost_of_substitution_i16_vecs[1].ymm);
        cost_if_gap_vecs[0].ymm = _mm256_add_epi16(pre_gap_vecs[0].ymm, gap_cost_vec.ymm);
        cost_if_gap_vecs[1].ymm = _mm256_add_epi16(pre_gap_vecs[1].ymm, gap_cost_vec.ymm);
        cell_score_vecs[0].ymm = _mm256_max_epi16(cost_if_substitution_vecs[0].ymm, cost_if_gap_vecs[0].ymm);
        cell_score_vecs[1].ymm = _mm256_max_epi16(cost_if_substitution_vecs[1].ymm, cost_if_gap_vecs[1].ymm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        if constexpr (locality_ == sz_similarity_local_k)
            cell_score_vecs[0].ymm = _mm256_max_epi16(cell_score_vecs[0].ymm, _mm256_setzero_si256()),
            cell_score_vecs[1].ymm = _mm256_max_epi16(cell_score_vecs[1].ymm, _mm256_setzero_si256());

        // Dump partial results to the output buffer.
        _mm256_storeu_si256((__m256i *)(scores_new + i + 0), cell_score_vecs[0].ymm);
        _mm256_storeu_si256((__m256i *)(scores_new + i + 16), cell_score_vecs[1].ymm);
    }

    void slice_1char(char const *second_slice, size_t i, sz_i16_t gap,                              //
                     sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
                     sz_i16_t *scores_new) const noexcept {
        // The substitution row is already addressed by the first character's class, so we only need
        // the second character's class to fetch the cost from the broadcast cost row.
        sz_u8_t const second_class = this->substituter_.byte_to_class[(sz_u8_t)second_slice[i]];
        sz_i16_t const cost_of_substitution = second_class < 16 ? lookup_.row_subs_low_vec_.i8s[second_class]
                                                                : lookup_.row_subs_high_vec_.i8s[second_class - 16];
        sz_i16_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        sz_i16_t const if_gap = scores_pre_insertion[i] + gap;
        sz_i16_t cell_score = sz_max_of_two(if_substitution, if_gap);
        if constexpr (locality_ == sz_similarity_local_k) cell_score = sz_max_of_two(cell_score, (sz_i16_t)0);
        scores_new[i] = cell_score;
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<constant_iterator<char>, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_haswell_k, void>
    : public tile_scorer<constant_iterator<char>, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    lookup_in256classes_haswell_t_ lookup_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                   //
        constant_iterator<char> first_char, char const *second_slice, size_t n,        //
        sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
        sz_i32_t const *scores_pre_deletion, sz_i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // Load a new substitution row, addressed by the class of the current first character.
        sz_i32_t const gap = static_cast<sz_i32_t>(this->gap_costs_.open_or_extend);
        sz_u8_t const first_class = this->substituter_.byte_to_class[(sz_u8_t)*first_char];
        lookup_.reload_classes(this->substituter_.byte_to_class);
        lookup_.reload_row(&this->substituter_.class_substitution_costs[first_class][0]);

        // Progress through the row 32 characters at a time.
        size_t const count_slices = n / 32;
        executor.for_n(count_slices, [&](size_t idx_slice) noexcept {
            slice_32chars(second_slice, idx_slice * 32, gap, scores_pre_substitution, scores_pre_insertion, scores_new);
        });

        // Handle the tail with a less efficient scalar kernel.
        for (size_t i = count_slices * 32; i < n; ++i)
            slice_1char(second_slice, i, gap, scores_pre_substitution, scores_pre_insertion, scores_new);

        // Horizontally propagate the deletion cost across the last row, just like the serial scorer.
        sz_assert_(scores_pre_substitution + 1 == scores_pre_insertion && "Expects horizontal traversal of DP matrix");
        sz_assert_(scores_pre_deletion + 1 == scores_new && "Expects horizontal traversal of DP matrix");
        sz_i32_t last_in_row = scores_pre_deletion[0];
        if constexpr (locality_ == sz_similarity_global_k) {
            for (size_t i = 0; i < n; ++i)
                scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
            this->last_score_ = last_in_row;
        }
        else {
            // In Local Alignment for SW the deletion still propagates horizontally, but every cell stays
            // non-negative and the running best across the whole matrix is the reported score.
            sz_i32_t row_best = this->best_score_;
            for (size_t i = 0; i < n; ++i) {
                scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
                row_best = sz_max_of_two(row_best, scores_new[i]);
            }
            this->best_score_ = row_best;
        }
    }

    void slice_32chars(char const *second_slice, size_t i, sz_i32_t gap,                              //
                       sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
                       sz_i32_t *scores_new) const noexcept {

        sz_u256_vec_t second_vec;
        sz_u256_vec_t pre_substitution_vecs[4], pre_gap_vecs[4];
        sz_u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i32_vecs[4];
        sz_u256_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        // Initialize constats:
        sz_u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi32(gap);

        // Load the data without any masks:
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)(second_slice + i));
        pre_substitution_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + i + 8 * 0));
        pre_substitution_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + i + 8 * 1));
        pre_substitution_vecs[2].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + i + 8 * 2));
        pre_substitution_vecs[3].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + i + 8 * 3));
        pre_gap_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + i + 8 * 0));
        pre_gap_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + i + 8 * 1));
        pre_gap_vecs[2].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + i + 8 * 2));
        pre_gap_vecs[3].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + i + 8 * 3));

        // First, sign-extend the substitution cost vector. The 32 packed `i8` costs split into 4 lanes of 8.
        cost_of_substitution_i8_vec = lookup_.lookup32(second_vec);
        __m128i cost_low_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0);
        __m128i cost_high_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1);
        cost_of_substitution_i32_vecs[0].ymm = _mm256_cvtepi8_epi32(cost_low_xmm);
        cost_of_substitution_i32_vecs[1].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_low_xmm, 8));
        cost_of_substitution_i32_vecs[2].ymm = _mm256_cvtepi8_epi32(cost_high_xmm);
        cost_of_substitution_i32_vecs[3].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_high_xmm, 8));

        // Then compute the data-parallel part, assuming the cost of deletions will be propagated
        // left to right outside of this loop.
        cost_if_substitution_vecs[0].ymm = _mm256_add_epi32(pre_substitution_vecs[0].ymm,
                                                            cost_of_substitution_i32_vecs[0].ymm);
        cost_if_substitution_vecs[1].ymm = _mm256_add_epi32(pre_substitution_vecs[1].ymm,
                                                            cost_of_substitution_i32_vecs[1].ymm);
        cost_if_substitution_vecs[2].ymm = _mm256_add_epi32(pre_substitution_vecs[2].ymm,
                                                            cost_of_substitution_i32_vecs[2].ymm);
        cost_if_substitution_vecs[3].ymm = _mm256_add_epi32(pre_substitution_vecs[3].ymm,
                                                            cost_of_substitution_i32_vecs[3].ymm);
        cost_if_gap_vecs[0].ymm = _mm256_add_epi32(pre_gap_vecs[0].ymm, gap_cost_vec.ymm);
        cost_if_gap_vecs[1].ymm = _mm256_add_epi32(pre_gap_vecs[1].ymm, gap_cost_vec.ymm);
        cost_if_gap_vecs[2].ymm = _mm256_add_epi32(pre_gap_vecs[2].ymm, gap_cost_vec.ymm);
        cost_if_gap_vecs[3].ymm = _mm256_add_epi32(pre_gap_vecs[3].ymm, gap_cost_vec.ymm);
        cell_score_vecs[0].ymm = _mm256_max_epi32(cost_if_substitution_vecs[0].ymm, cost_if_gap_vecs[0].ymm);
        cell_score_vecs[1].ymm = _mm256_max_epi32(cost_if_substitution_vecs[1].ymm, cost_if_gap_vecs[1].ymm);
        cell_score_vecs[2].ymm = _mm256_max_epi32(cost_if_substitution_vecs[2].ymm, cost_if_gap_vecs[2].ymm);
        cell_score_vecs[3].ymm = _mm256_max_epi32(cost_if_substitution_vecs[3].ymm, cost_if_gap_vecs[3].ymm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        if constexpr (locality_ == sz_similarity_local_k)
            cell_score_vecs[0].ymm = _mm256_max_epi32(cell_score_vecs[0].ymm, _mm256_setzero_si256()),
            cell_score_vecs[1].ymm = _mm256_max_epi32(cell_score_vecs[1].ymm, _mm256_setzero_si256()),
            cell_score_vecs[2].ymm = _mm256_max_epi32(cell_score_vecs[2].ymm, _mm256_setzero_si256()),
            cell_score_vecs[3].ymm = _mm256_max_epi32(cell_score_vecs[3].ymm, _mm256_setzero_si256());

        // Dump partial results to the output buffer.
        _mm256_storeu_si256((__m256i *)(scores_new + i + 8 * 0), cell_score_vecs[0].ymm);
        _mm256_storeu_si256((__m256i *)(scores_new + i + 8 * 1), cell_score_vecs[1].ymm);
        _mm256_storeu_si256((__m256i *)(scores_new + i + 8 * 2), cell_score_vecs[2].ymm);
        _mm256_storeu_si256((__m256i *)(scores_new + i + 8 * 3), cell_score_vecs[3].ymm);
    }

    void slice_1char(char const *second_slice, size_t i, sz_i32_t gap,                              //
                     sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
                     sz_i32_t *scores_new) const noexcept {
        sz_u8_t const second_class = this->substituter_.byte_to_class[(sz_u8_t)second_slice[i]];
        sz_i32_t const cost_of_substitution = second_class < 16 ? lookup_.row_subs_low_vec_.i8s[second_class]
                                                                : lookup_.row_subs_high_vec_.i8s[second_class - 16];
        sz_i32_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        sz_i32_t const if_gap = scores_pre_insertion[i] + gap;
        sz_i32_t cell_score = sz_max_of_two(if_substitution, if_gap);
        if constexpr (locality_ == sz_similarity_local_k) cell_score = sz_max_of_two(cell_score, (sz_i32_t)0);
        scores_new[i] = cell_score;
    }
};

/** @brief No 64-bit Haswell specialization yet, fall back to the serial scalar tile scorer. */
template <sz_similarity_locality_t locality_>
struct tile_scorer<constant_iterator<char>, char const *, sz_i64_t, error_costs_32x32_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_haswell_k>
    : public tile_scorer<constant_iterator<char>, char const *, sz_i64_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, sz_i64_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b global Needleman-Wunsch score with class-based
 *         substitution costs over a @b diagonal walker, for inputs whose diagonal exceeds the tiny threshold.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 *
 *  Mirrors the Ice Lake `sz_i16_t` diagonal class scorer (reversed-first / forward-second class streams, the
 *  `max`/`add` recurrence `cell = max(pre_sub + sub_cost, max(pre_ins, pre_del) + gap)`), but works over 256-bit
 *  vectors with the AVX2 two-stage `substitution_matrix_lookup_haswell_t_`. AVX2 has no masked loads or stores,
 *  so each diagonal is processed in full 32-cell `loadu`/`storeu` slices with a @b scalar tail epilogue, exactly
 *  like the horizontal Haswell scorers above.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_matrix_lookup_haswell_t_ lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                      //
        sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice,              //
        sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
        sz_i16_t const *scores_pre_deletion, sz_i16_t *scores_new,                     //
        sz_u256_vec_t gap_cost_vec) const noexcept {

        sz_u256_vec_t first_vec, second_vec;
        sz_u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        sz_u256_vec_t pre_substitution_vecs[2], pre_insert_vecs[2], pre_delete_vecs[2];
        sz_u256_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        // ? Both buffers are traversed in the same order, because one of the strings has been reversed beforehand.
        first_vec.ymm = _mm256_loadu_si256((__m256i const *)first_reversed_slice);
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)second_slice);
        pre_substitution_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + 0));
        pre_substitution_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + 16));
        pre_insert_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + 0));
        pre_insert_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + 16));
        pre_delete_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + 0));
        pre_delete_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + 16));

        // Sign-extend the 32 class-pair substitution costs into two `i16` halves.
        cost_of_substitution_i8_vec = lookup_.lookup32(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0));
        cost_of_substitution_i16_vecs[1].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1));

        cost_if_substitution_vecs[0].ymm = _mm256_add_epi16(pre_substitution_vecs[0].ymm,
                                                            cost_of_substitution_i16_vecs[0].ymm);
        cost_if_substitution_vecs[1].ymm = _mm256_add_epi16(pre_substitution_vecs[1].ymm,
                                                            cost_of_substitution_i16_vecs[1].ymm);
        cost_if_gap_vecs[0].ymm = _mm256_add_epi16(_mm256_max_epi16(pre_insert_vecs[0].ymm, pre_delete_vecs[0].ymm),
                                                   gap_cost_vec.ymm);
        cost_if_gap_vecs[1].ymm = _mm256_add_epi16(_mm256_max_epi16(pre_insert_vecs[1].ymm, pre_delete_vecs[1].ymm),
                                                   gap_cost_vec.ymm);
        cell_score_vecs[0].ymm = _mm256_max_epi16(cost_if_substitution_vecs[0].ymm, cost_if_gap_vecs[0].ymm);
        cell_score_vecs[1].ymm = _mm256_max_epi16(cost_if_substitution_vecs[1].ymm, cost_if_gap_vecs[1].ymm);
        _mm256_storeu_si256((__m256i *)(scores_new + 0), cell_score_vecs[0].ymm);
        _mm256_storeu_si256((__m256i *)(scores_new + 16), cell_score_vecs[1].ymm);
    }

    SZ_INLINE void slice_1cell(                                                        //
        sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice,              //
        sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
        sz_i16_t const *scores_pre_deletion, sz_i16_t *scores_new, sz_i16_t gap) const noexcept {
        sz_u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        sz_i16_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                    : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        sz_i16_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        sz_i16_t const if_gap = sz_max_of_two(scores_pre_insertion[0], scores_pre_deletion[0]) + gap;
        scores_new[0] = sz_max_of_two(if_substitution, if_gap);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion,   //
        sz_i16_t const *scores_pre_deletion, sz_i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        sz_u8_t const *first_reversed_classes = (sz_u8_t const *)first_reversed_slice;
        sz_u8_t const *second_classes = (sz_u8_t const *)second_slice;
        sz_i16_t const gap = static_cast<sz_i16_t>(this->gap_costs_.open_or_extend);

        sz_u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi16(gap);

        // AVX2 has no masked loads/stores, so we process full 32-cell slices and finish with a scalar tail.
        size_t const count_slices = length / step_k;
        executor.for_n(count_slices, [&](size_t idx_slice) noexcept {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                                       //
                first_reversed_classes + progress, second_classes + progress,                    //
                scores_pre_substitution + progress, scores_pre_insertion + progress,             //
                scores_pre_deletion + progress, scores_new + progress, gap_cost_vec);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(first_reversed_classes + i, second_classes + i, scores_pre_substitution + i,
                        scores_pre_insertion + i, scores_pre_deletion + i, scores_new + i, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of the diagonal class scorer that maximizes the @b local Smith-Waterman score.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 *
 *  Identical to the global diagonal scorer above, but adds the per-cell zero-clamp and tracks the running best
 *  across the whole matrix, mirroring the Ice Lake local diagonal scorer and the horizontal class scorer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_i16_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_matrix_lookup_haswell_t_ lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                      //
        sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice,              //
        sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
        sz_i16_t const *scores_pre_deletion, sz_i16_t *scores_new,                     //
        sz_u256_vec_t gap_cost_vec) const noexcept {

        sz_u256_vec_t first_vec, second_vec;
        sz_u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        sz_u256_vec_t pre_substitution_vecs[2], pre_insert_vecs[2], pre_delete_vecs[2];
        sz_u256_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        first_vec.ymm = _mm256_loadu_si256((__m256i const *)first_reversed_slice);
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)second_slice);
        pre_substitution_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + 0));
        pre_substitution_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + 16));
        pre_insert_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + 0));
        pre_insert_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + 16));
        pre_delete_vecs[0].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + 0));
        pre_delete_vecs[1].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + 16));

        cost_of_substitution_i8_vec = lookup_.lookup32(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0));
        cost_of_substitution_i16_vecs[1].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1));

        cost_if_substitution_vecs[0].ymm = _mm256_add_epi16(pre_substitution_vecs[0].ymm,
                                                            cost_of_substitution_i16_vecs[0].ymm);
        cost_if_substitution_vecs[1].ymm = _mm256_add_epi16(pre_substitution_vecs[1].ymm,
                                                            cost_of_substitution_i16_vecs[1].ymm);
        cost_if_gap_vecs[0].ymm = _mm256_add_epi16(_mm256_max_epi16(pre_insert_vecs[0].ymm, pre_delete_vecs[0].ymm),
                                                   gap_cost_vec.ymm);
        cost_if_gap_vecs[1].ymm = _mm256_add_epi16(_mm256_max_epi16(pre_insert_vecs[1].ymm, pre_delete_vecs[1].ymm),
                                                   gap_cost_vec.ymm);
        cell_score_vecs[0].ymm = _mm256_max_epi16(cost_if_substitution_vecs[0].ymm, cost_if_gap_vecs[0].ymm);
        cell_score_vecs[1].ymm = _mm256_max_epi16(cost_if_substitution_vecs[1].ymm, cost_if_gap_vecs[1].ymm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        cell_score_vecs[0].ymm = _mm256_max_epi16(cell_score_vecs[0].ymm, _mm256_setzero_si256());
        cell_score_vecs[1].ymm = _mm256_max_epi16(cell_score_vecs[1].ymm, _mm256_setzero_si256());
        _mm256_storeu_si256((__m256i *)(scores_new + 0), cell_score_vecs[0].ymm);
        _mm256_storeu_si256((__m256i *)(scores_new + 16), cell_score_vecs[1].ymm);
    }

    SZ_INLINE void slice_1cell(                                                        //
        sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice,              //
        sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
        sz_i16_t const *scores_pre_deletion, sz_i16_t *scores_new, sz_i16_t gap) const noexcept {
        sz_u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        sz_i16_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                    : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        sz_i16_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        sz_i16_t const if_gap = sz_max_of_two(scores_pre_insertion[0], scores_pre_deletion[0]) + gap;
        scores_new[0] = sz_max_of_two(sz_max_of_two(if_substitution, if_gap), (sz_i16_t)0);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion,   //
        sz_i16_t const *scores_pre_deletion, sz_i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        sz_u8_t const *first_reversed_classes = (sz_u8_t const *)first_reversed_slice;
        sz_u8_t const *second_classes = (sz_u8_t const *)second_slice;
        sz_i16_t const gap = static_cast<sz_i16_t>(this->gap_costs_.open_or_extend);
        sz_i16_t *const scores_new_begin = scores_new;

        sz_u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi16(gap);

        size_t const count_slices = length / step_k;
        executor.for_n(count_slices, [&](size_t idx_slice) noexcept {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                                       //
                first_reversed_classes + progress, second_classes + progress,                    //
                scores_pre_substitution + progress, scores_pre_insertion + progress,             //
                scores_pre_deletion + progress, scores_new + progress, gap_cost_vec);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(first_reversed_classes + i, second_classes + i, scores_pre_substitution + i,
                        scores_pre_insertion + i, scores_pre_deletion + i, scores_new + i, gap);

        // The running best across the whole matrix is the reported local-alignment score.
        sz_i16_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Variant of the global diagonal class scorer over @b `sz_i32_t` cells, for inputs whose scores exceed
 *         the 16-bit range. Mirrors the `sz_i16_t` scorer but folds the 32 class-pair costs into four 8-lane
 *         `i32` quarters via `_mm256_cvtepi8_epi32`.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_matrix_lookup_haswell_t_ lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                      //
        sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice,              //
        sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
        sz_i32_t const *scores_pre_deletion, sz_i32_t *scores_new,                     //
        sz_u256_vec_t gap_cost_vec) const noexcept {

        sz_u256_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        sz_u256_vec_t cost_of_substitution_i32_vecs[4];
        sz_u256_vec_t pre_substitution_vecs[4], pre_insert_vecs[4], pre_delete_vecs[4];
        sz_u256_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        first_vec.ymm = _mm256_loadu_si256((__m256i const *)first_reversed_slice);
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)second_slice);
        for (size_t part = 0; part != 4; ++part) {
            pre_substitution_vecs[part].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + part * 8));
            pre_insert_vecs[part].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + part * 8));
            pre_delete_vecs[part].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + part * 8));
        }

        // Sign-extend the 32 class-pair costs into four 8-lane `i32` quarters.
        cost_of_substitution_i8_vec = lookup_.lookup32(first_vec, second_vec);
        __m128i cost_low_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0);
        __m128i cost_high_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1);
        cost_of_substitution_i32_vecs[0].ymm = _mm256_cvtepi8_epi32(cost_low_xmm);
        cost_of_substitution_i32_vecs[1].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_low_xmm, 8));
        cost_of_substitution_i32_vecs[2].ymm = _mm256_cvtepi8_epi32(cost_high_xmm);
        cost_of_substitution_i32_vecs[3].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_high_xmm, 8));

        for (size_t part = 0; part != 4; ++part) {
            cost_if_substitution_vecs[part].ymm = _mm256_add_epi32(pre_substitution_vecs[part].ymm,
                                                                   cost_of_substitution_i32_vecs[part].ymm);
            cost_if_gap_vecs[part].ymm = _mm256_add_epi32(
                _mm256_max_epi32(pre_insert_vecs[part].ymm, pre_delete_vecs[part].ymm), gap_cost_vec.ymm);
            cell_score_vecs[part].ymm = _mm256_max_epi32(cost_if_substitution_vecs[part].ymm,
                                                         cost_if_gap_vecs[part].ymm);
            _mm256_storeu_si256((__m256i *)(scores_new + part * 8), cell_score_vecs[part].ymm);
        }
    }

    SZ_INLINE void slice_1cell(                                                        //
        sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice,              //
        sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
        sz_i32_t const *scores_pre_deletion, sz_i32_t *scores_new, sz_i32_t gap) const noexcept {
        sz_u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        sz_i32_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                    : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        sz_i32_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        sz_i32_t const if_gap = sz_max_of_two(scores_pre_insertion[0], scores_pre_deletion[0]) + gap;
        scores_new[0] = sz_max_of_two(if_substitution, if_gap);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion,   //
        sz_i32_t const *scores_pre_deletion, sz_i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        sz_u8_t const *first_reversed_classes = (sz_u8_t const *)first_reversed_slice;
        sz_u8_t const *second_classes = (sz_u8_t const *)second_slice;
        sz_i32_t const gap = static_cast<sz_i32_t>(this->gap_costs_.open_or_extend);

        sz_u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi32(gap);

        size_t const count_slices = length / step_k;
        executor.for_n(count_slices, [&](size_t idx_slice) noexcept {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                                       //
                first_reversed_classes + progress, second_classes + progress,                    //
                scores_pre_substitution + progress, scores_pre_insertion + progress,             //
                scores_pre_deletion + progress, scores_new + progress, gap_cost_vec);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(first_reversed_classes + i, second_classes + i, scores_pre_substitution + i,
                        scores_pre_insertion + i, scores_pre_deletion + i, scores_new + i, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of the local diagonal class scorer over @b `sz_i32_t` cells. Mirrors the `sz_i32_t` global
 *         scorer, plus the per-cell zero-clamp and the running-best reduction of Smith-Waterman.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_i32_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_matrix_lookup_haswell_t_ lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                      //
        sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice,              //
        sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
        sz_i32_t const *scores_pre_deletion, sz_i32_t *scores_new,                     //
        sz_u256_vec_t gap_cost_vec) const noexcept {

        sz_u256_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        sz_u256_vec_t cost_of_substitution_i32_vecs[4];
        sz_u256_vec_t pre_substitution_vecs[4], pre_insert_vecs[4], pre_delete_vecs[4];
        sz_u256_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        first_vec.ymm = _mm256_loadu_si256((__m256i const *)first_reversed_slice);
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)second_slice);
        for (size_t part = 0; part != 4; ++part) {
            pre_substitution_vecs[part].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + part * 8));
            pre_insert_vecs[part].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + part * 8));
            pre_delete_vecs[part].ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + part * 8));
        }

        cost_of_substitution_i8_vec = lookup_.lookup32(first_vec, second_vec);
        __m128i cost_low_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0);
        __m128i cost_high_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1);
        cost_of_substitution_i32_vecs[0].ymm = _mm256_cvtepi8_epi32(cost_low_xmm);
        cost_of_substitution_i32_vecs[1].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_low_xmm, 8));
        cost_of_substitution_i32_vecs[2].ymm = _mm256_cvtepi8_epi32(cost_high_xmm);
        cost_of_substitution_i32_vecs[3].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_high_xmm, 8));

        for (size_t part = 0; part != 4; ++part) {
            cost_if_substitution_vecs[part].ymm = _mm256_add_epi32(pre_substitution_vecs[part].ymm,
                                                                   cost_of_substitution_i32_vecs[part].ymm);
            cost_if_gap_vecs[part].ymm = _mm256_add_epi32(
                _mm256_max_epi32(pre_insert_vecs[part].ymm, pre_delete_vecs[part].ymm), gap_cost_vec.ymm);
            cell_score_vecs[part].ymm = _mm256_max_epi32(cost_if_substitution_vecs[part].ymm,
                                                         cost_if_gap_vecs[part].ymm);
            // In Local Alignment for SW we also clamp every cell to zero.
            cell_score_vecs[part].ymm = _mm256_max_epi32(cell_score_vecs[part].ymm, _mm256_setzero_si256());
            _mm256_storeu_si256((__m256i *)(scores_new + part * 8), cell_score_vecs[part].ymm);
        }
    }

    SZ_INLINE void slice_1cell(                                                        //
        sz_u8_t const *first_reversed_slice, sz_u8_t const *second_slice,              //
        sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
        sz_i32_t const *scores_pre_deletion, sz_i32_t *scores_new, sz_i32_t gap) const noexcept {
        sz_u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        sz_i32_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                    : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        sz_i32_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        sz_i32_t const if_gap = sz_max_of_two(scores_pre_insertion[0], scores_pre_deletion[0]) + gap;
        scores_new[0] = sz_max_of_two(sz_max_of_two(if_substitution, if_gap), (sz_i32_t)0);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion,   //
        sz_i32_t const *scores_pre_deletion, sz_i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        sz_u8_t const *first_reversed_classes = (sz_u8_t const *)first_reversed_slice;
        sz_u8_t const *second_classes = (sz_u8_t const *)second_slice;
        sz_i32_t const gap = static_cast<sz_i32_t>(this->gap_costs_.open_or_extend);
        sz_i32_t *const scores_new_begin = scores_new;

        sz_u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi32(gap);

        size_t const count_slices = length / step_k;
        executor.for_n(count_slices, [&](size_t idx_slice) noexcept {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                                       //
                first_reversed_classes + progress, second_classes + progress,                    //
                scores_pre_substitution + progress, scores_pre_insertion + progress,             //
                scores_pre_deletion + progress, scores_new + progress, gap_cost_vec);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(first_reversed_classes + i, second_classes + i, scores_pre_substitution + i,
                        scores_pre_insertion + i, scores_pre_deletion + i, scores_new + i, gap);

        sz_i32_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/** @brief Redirects the Haswell template specialization to the serial version. */
template <typename char_type_, typename score_type_, typename substituter_type_, typename gap_costs_type_,
          typename allocator_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, allocator_type_, objective_,
                         locality_, sz_cap_haswell_k, void>
    : public horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, allocator_type_, objective_,
                               locality_, sz_cap_serial_k, void> {

    using base_t = horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, allocator_type_,
                                     objective_, locality_, sz_cap_serial_k, void>;

    using base_t::base_t;
    using base_t::operator();
};

/**
 *  @brief Haswell horizontal "walker" for class-based substitution costs with linear gaps.
 *         Mirrors the serial walker, but binds its `tile_scorer_t` to the AVX2 class lookup above,
 *         which the generic redirect above would otherwise leave on the scalar serial path.
 */
template <typename char_type_, typename score_type_, typename allocator_type_, sz_similarity_objective_t objective_,
          sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, error_costs_32x32_t, linear_gap_costs_t, allocator_type_, objective_,
                         locality_, sz_cap_haswell_k, void> {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    using walker_t = horizontal_walker<char_t, score_t, substituter_t, gap_costs_t, allocator_t, objective_k,
                                       locality_k, capability_k, void>;

    using allocated_t = typename allocator_t::value_type;
    static_assert(sizeof(allocated_t) == sizeof(char), "Allocator must be byte-aligned");
    using tile_scorer_t = tile_scorer<constant_iterator<char_t>, char_t const *, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    mutable allocator_t alloc_ {};

    horizontal_walker(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    horizontal_walker(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) { result_ref = gap_costs_.open_or_extend * first.size(); }
                else if (first.empty() && !second.empty()) { result_ref = gap_costs_.open_or_extend * second.size(); }
            }
            return status_t::success_k;
        }

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
        size_t shorter_length = first.size(), longer_length = second.size();
        if (shorter_length > longer_length) {
            trivial_swap(shorter, longer);
            trivial_swap(shorter_length, longer_length);
        }

        // We are going to store 2 rows of the matrix. It will be either 2 rows of length `shorter_length + 1`
        // or 2 rows of length `longer_length + 1`, depending on our preference - either minimizing the memory
        // consumption or the inner loop performance.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;

        // We decide to use less memory!
        size_t const buffer_length = sizeof(score_t) * shorter_dim * 2;
        score_t *const buffer = (score_t *)alloc_.allocate(buffer_length);
        if (!buffer) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = buffer;
        score_t *current_scores = previous_scores + shorter_dim;

        // Initialize the first row:
        tile_scorer_t scorer {substituter_, gap_costs_};
        for (size_t col_idx = 0; col_idx < shorter_dim; ++col_idx) scorer.init_score(previous_scores[col_idx], col_idx);

        // Progress through the matrix row-by-row:
        for (size_t row_idx = 1; row_idx < longer_dim; ++row_idx) {

            // Don't forget to populate the first column of each row:
            scorer.init_score(current_scores[0], row_idx);

            scorer(                                              //
                constant_iterator<char_t> {longer[row_idx - 1]}, // first sequence of characters
                shorter,                                         // second sequence of characters
                shorter_dim - 1,                                 // number of elements to compute with the `scorer`
                previous_scores,                                 // costs pre substitution
                previous_scores + 1,                             // costs pre insertion
                current_scores,                                  // costs pre deletion
                current_scores + 1,                              // new scores
                executor                                         // ! note, most horizontal scorers are not parallel
            );

            // Reuse the memory.
            trivial_swap(previous_scores, current_scores);
        }

        // Export the scalar before `free` call.
        result_ref = scorer.score();
        alloc_.deallocate((allocated_t *)buffer, buffer_length);
        return status_t::success_k;
    }
};

/**
 *  @brief Haswell diagonal "walker" for class-based substitution costs with linear gaps. Mirrors the Ice Lake
 *         diagonal walker: it pre-classifies both strings @b once into class streams and keeps a 3-diagonal
 *         buffer, but feeds the AVX2 `substitution_matrix_lookup_haswell_t_` instead of the `VPERMB` one.
 *
 *  When the diagonal walker swaps the shorter and longer strings, the order of the two class operands flips;
 *  to keep the recurrence reading the original `class_substitution_costs[first][second]`, the swap bit is fed
 *  into `tile_scorer_t::prepare`, which folds the resident table transposed (one-time, off the hot path).
 */
template <typename score_type_, typename allocator_type_, sz_similarity_objective_t objective_,
          sz_similarity_locality_t locality_>
struct diagonal_walker<char, score_type_, error_costs_32x32_t, linear_gap_costs_t, allocator_type_, objective_,
                       locality_, sz_cap_haswell_k, void> {

    using char_t = char;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    using allocated_t = typename allocator_t::value_type;
    static_assert(sizeof(allocated_t) == sizeof(char), "Allocator must be byte-aligned");
    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    mutable allocator_t alloc_ {};

    diagonal_walker(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    diagonal_walker(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) { result_ref = gap_costs_.open_or_extend * first.size(); }
                else if (first.empty() && !second.empty()) { result_ref = gap_costs_.open_or_extend * second.size(); }
            }
            return status_t::success_k;
        }

        // Make sure the size relation between the strings is correct. When the strings get swapped, the two class
        // operands change order, so the cost table is folded @b transposed to keep the recurrence reading the
        // original `class_substitution_costs[first][second]`, matching the serial scorer bit-for-bit.
        char_t const *shorter = first.data(), *longer = second.data();
        size_t shorter_length = first.size(), longer_length = second.size();
        bool transpose = false;
        if (shorter_length > longer_length) {
            trivial_swap(shorter, longer);
            trivial_swap(shorter_length, longer_length);
            transpose = true;
        }

        // We are going to store 3 diagonals of the matrix.
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;
        size_t const diagonals_count = shorter_dim + longer_dim - 1;
        size_t const max_diagonal_length = shorter_length + 1;

        // Beyond the 3 score diagonals, we also keep the @b reversed shorter string as class bytes and the
        // longer string as class bytes. Both class streams are over-allocated by one slice, so the full-width
        // unaligned loads in the body never read out of bounds.
        size_t const padding = step_classes_k;
        size_t const buffer_length = sizeof(score_t) * max_diagonal_length * 3 + (shorter_length + padding) * 2 +
                                     (longer_length + padding);
        score_t *const buffer = (score_t *)alloc_.allocate(buffer_length);
        if (!buffer) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = buffer;
        score_t *current_scores = previous_scores + max_diagonal_length;
        score_t *next_scores = current_scores + max_diagonal_length;
        char_t *const shorter_reversed = (char_t *)(next_scores + max_diagonal_length);
        char_t *const shorter_reversed_classes = shorter_reversed + shorter_length + padding;
        char_t *const longer_classes = shorter_reversed_classes + shorter_length + padding;

        // Export the reversed shorter string, then classify both strings @b once into their class streams.
        for (size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

        tile_scorer_t scorer {substituter_, gap_costs_};
        scorer.lookup_.reload_classes(substituter_.byte_to_class);
        scorer.prepare(transpose);
        classify_into_(scorer.lookup_, shorter_reversed, shorter_length, shorter_reversed_classes);
        classify_into_(scorer.lookup_, longer, longer_length, longer_classes);

        // Initialize the first two diagonals:
        scorer.init_score(previous_scores[0], 0);
        scorer.init_score(current_scores[0], 1);
        scorer.init_score(current_scores[1], 1);

        size_t next_diagonal_index = 2;

        // Progress through the upper-left triangle of the matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = next_diagonal_index + 1;
            scorer(                                                                  //
                shorter_reversed_classes + shorter_length - next_diagonal_index + 1, // first sequence of classes
                longer_classes,                                                      // second sequence of classes
                next_diagonal_length - 2,           // number of elements to compute with the `scorer`
                previous_scores,                    // costs pre substitution
                current_scores, current_scores + 1, // costs pre insertion/deletion
                next_scores + 1,                    // new scores for the next diagonal
                executor);                          // parallel execution within the diagonal

            // Don't forget to populate the first row and the first column of the matrix.
            scorer.init_score(next_scores[0], next_diagonal_index);
            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            rotate_three(previous_scores, current_scores, next_scores);
        }

        // Now let's handle the anti-diagonal band of the matrix.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = shorter_dim;
            scorer(                                                          //
                shorter_reversed_classes + shorter_length - shorter_dim + 1, // first sequence of classes
                longer_classes + next_diagonal_index - shorter_dim,          // second sequence of classes
                next_diagonal_length - 1,                                    // number of elements to compute
                previous_scores,                                             // costs pre substitution
                current_scores, current_scores + 1,                          // costs pre insertion/deletion
                next_scores,                                                 // new scores for the next diagonal
                executor);                                                   // parallel execution within the diagonal

            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            rotate_three(previous_scores, current_scores, next_scores);
            sz_move_serial((sz_ptr_t)(previous_scores), (sz_ptr_t)(previous_scores + 1),
                           (max_diagonal_length - 1) * sizeof(score_t));
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
            scorer(                                                          //
                shorter_reversed_classes + shorter_length - shorter_dim + 1, // first sequence of classes
                longer_classes + next_diagonal_index - shorter_dim,          // second sequence of classes
                next_diagonal_length,                                        // number of elements to compute
                previous_scores,                                             // costs pre substitution
                current_scores, current_scores + 1,                          // costs pre insertion/deletion
                next_scores,                                                 // new scores for the next diagonal
                executor);                                                   // parallel execution within the diagonal

            rotate_three(previous_scores, current_scores, next_scores);
            previous_scores++;
        }

        // Export the scalar before `free` call.
        result_ref = scorer.score();
        alloc_.deallocate((allocated_t *)buffer, buffer_length);
        return status_t::success_k;
    }

  private:
    static constexpr size_t step_classes_k = 32;

    /** @brief Maps a raw byte string into class bytes using the resident `byte_to_class` lookup, @b amortized. */
    static void classify_into_(substitution_matrix_lookup_haswell_t_ const &lookup, char_t const *source,
                               size_t length, char_t *classes) noexcept {
        sz_u256_vec_t source_vec, classes_vec;
        size_t progress = 0;
        for (; progress + step_classes_k <= length; progress += step_classes_k) {
            source_vec.ymm = _mm256_loadu_si256((__m256i const *)(source + progress));
            classes_vec = lookup.classify32(source_vec);
            _mm256_storeu_si256((__m256i *)(classes + progress), classes_vec.ymm);
        }
        // AVX2 has no masked store, so the tail is classified one byte at a time.
        for (; progress != length; ++progress)
            classes[progress] = (char_t)lookup.byte_to_class_group_vecs_[((sz_u8_t)source[progress]) >> 4]
                                    .u8s[((sz_u8_t)source[progress]) & 0x0f];
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score between two strings using the Haswell backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <typename allocator_type_>
struct needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, allocator_type_, sz_caps_sh_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;

    using diagonal_i16_t =                                                         //
        diagonal_walker<char_t, sz_i16_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_global_k, sz_cap_haswell_k>;
    using diagonal_i32_t =                                                         //
        diagonal_walker<char_t, sz_i32_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_global_k, sz_cap_haswell_k>;
    using diagonal_i64_t =                                                         //
        diagonal_walker<char_t, sz_i64_t, substituter_t, gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    needleman_wunsch_score(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_score(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_ssize_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, true>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        // The diagonal class scorer owns all input sizes; we only differentiate the cell type, since smaller
        // ones overflow for larger inputs while larger-than-needed types waste memory and bandwidth. The 64-bit
        // path has no AVX2 diagonal scorer yet, so it falls back to the serial diagonal walker.
        if (requirements.bytes_per_cell <= 2) {
            sz_i16_t result_i16;
            status_t status = diagonal_i16_t {substituter_, gap_costs_, alloc_}(first, second, result_i16, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_i32_t result_i32;
            status_t status = diagonal_i32_t {substituter_, gap_costs_, alloc_}(first, second, result_i32, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_i64_t result_i64;
            status_t status = diagonal_i64_t {substituter_, gap_costs_, alloc_}(first, second, result_i64, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Computes the @b byte-level Smith-Waterman score between two strings using the Haswell backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <typename allocator_type_>
struct smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, allocator_type_, sz_caps_sh_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;

    using diagonal_i16_t =                                                                //
        diagonal_walker<char_t, sz_i16_t, substituter_t, linear_gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_local_k, sz_cap_haswell_k>;
    using diagonal_i32_t =                                                                //
        diagonal_walker<char_t, sz_i32_t, substituter_t, linear_gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_local_k, sz_cap_haswell_k>;
    using diagonal_i64_t =                                                                //
        diagonal_walker<char_t, sz_i64_t, substituter_t, linear_gap_costs_t, allocator_t, //
                        sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    smith_waterman_score(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    smith_waterman_score(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_ssize_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, true>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        // The diagonal class scorer owns all input sizes; we only differentiate the cell type, since smaller
        // ones overflow for larger inputs while larger-than-needed types waste memory and bandwidth. The 64-bit
        // path has no AVX2 diagonal scorer yet, so it falls back to the serial diagonal walker.
        if (requirements.bytes_per_cell <= 2) {
            sz_i16_t result_i16;
            status_t status = diagonal_i16_t {substituter_, gap_costs_, alloc_}(first, second, result_i16, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_i32_t result_i32;
            status_t status = diagonal_i32_t {substituter_, gap_costs_, alloc_}(first, second, result_i32, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_i64_t result_i64;
            status_t status = diagonal_i64_t {substituter_, gap_costs_, alloc_}(first, second, result_i64, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i64;
        }

        return status_t::success_k;
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_HASWELL_HPP_
