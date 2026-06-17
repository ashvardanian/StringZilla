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
struct class_lookup_haswell_t {
    u256_vec_t byte_to_class_group_vecs_[16]; // ! Each holds one 16-byte group broadcast to both lanes
    u256_vec_t row_subs_low_vec_, row_subs_high_vec_;
    u256_vec_t low_nibble_mask_vec_;

    inline class_lookup_haswell_t() noexcept { low_nibble_mask_vec_.ymm = _mm256_set1_epi8(0x0f); }

    inline void reload_classes(u8_t const *byte_to_class) noexcept {
        for (int group = 0; group != 16; ++group) {
            __m128i group_xmm = _mm_loadu_si128((__m128i const *)(byte_to_class + group * 16));
            byte_to_class_group_vecs_[group].ymm = _mm256_set_m128i(group_xmm, group_xmm);
        }
    }

    inline void reload_row(error_cost_t const *row_subs) noexcept {
        // The cost row holds only `error_costs_classes_count_k` (32) entries, split into two 16-byte halves
        // broadcast to both lanes so the second-stage `VPSHUFB` works regardless of which lane a class lands in.
        __m128i low_xmm = _mm_loadu_si128((__m128i const *)(row_subs + 0));
        __m128i high_xmm = _mm_loadu_si128((__m128i const *)(row_subs + 16));
        row_subs_low_vec_.ymm = _mm256_set_m128i(low_xmm, low_xmm);
        row_subs_high_vec_.ymm = _mm256_set_m128i(high_xmm, high_xmm);
    }

    inline u256_vec_t lookup32(u256_vec_t const &text_vec) const noexcept {

        u256_vec_t low_nibbles_vec, high_nibbles_vec;
        u256_vec_t class_vec, substituted_vec;

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
struct substitution_lookup_haswell_t {
    u256_vec_t byte_to_class_group_vecs_[16]; // ! Each holds one 16-byte group broadcast to both lanes
    u256_vec_t cost_rows_low_vecs_[error_costs_classes_count_k], cost_rows_high_vecs_[error_costs_classes_count_k];
    u256_vec_t low_nibble_mask_vec_;

    inline substitution_lookup_haswell_t() noexcept { low_nibble_mask_vec_.ymm = _mm256_set1_epi8(0x0f); }

    inline void reload_classes(u8_t const *byte_to_class) noexcept {
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

    inline u256_vec_t classify32(u256_vec_t const &text_vec) const noexcept {

        u256_vec_t low_nibbles_vec, high_nibbles_vec, class_vec;

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

    inline u256_vec_t lookup32(u256_vec_t const &first_class_vec, u256_vec_t const &second_class_vec) const noexcept {

        u256_vec_t cost_low_vecs[error_costs_classes_count_k], cost_high_vecs[error_costs_classes_count_k];
        u256_vec_t substituted_vec;

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
struct tile_scorer<constant_iterator<char>, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_haswell_k>
    : public tile_scorer<constant_iterator<char>, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    class_lookup_haswell_t lookup_;

    /** @brief Executor-independent trampoline, computing one row of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        char const *second_slice, i16_t gap,                                     //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t *scores_new, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice)
            slice_32chars(second_slice, idx_slice * 32, gap, scores_pre_substitution, scores_pre_insertion, scores_new);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                             //
        constant_iterator<char> first_char, char const *second_slice, size_t n,  //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // Load a new substitution row, addressed by the class of the current first character.
        i16_t const gap = static_cast<i16_t>(this->gap_costs_.open_or_extend);
        u8_t const first_class = this->substituter_.byte_to_class[(u8_t)*first_char];
        lookup_.reload_classes(this->substituter_.byte_to_class);
        lookup_.reload_row(&this->substituter_.class_substitution_costs[first_class][0]);

        // Progress through the row 32 characters at a time.
        size_t const count_slices = n / 32;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(second_slice, gap, scores_pre_substitution, scores_pre_insertion, scores_new, from,
                                    to);
        });

        // Handle the tail with a less efficient scalar kernel.
        for (size_t i = count_slices * 32; i < n; ++i)
            slice_1char(second_slice, i, gap, scores_pre_substitution, scores_pre_insertion, scores_new);

        // Horizontally propagate the deletion cost across the last row, just like the serial scorer.
        sz_assert_(scores_pre_substitution + 1 == scores_pre_insertion && "Expects horizontal traversal of DP matrix");
        sz_assert_(scores_pre_deletion + 1 == scores_new && "Expects horizontal traversal of DP matrix");
        i16_t last_in_row = scores_pre_deletion[0];
        if constexpr (locality_ == sz_similarity_global_k) {
            for (size_t i = 0; i < n; ++i)
                scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
            this->last_score_ = last_in_row;
        }
        else {
            // In Local Alignment for SW the deletion still propagates horizontally, but every cell stays
            // non-negative and the running best across the whole matrix is the reported score.
            i16_t row_best = this->best_score_;
            for (size_t i = 0; i < n; ++i) {
                scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
                row_best = sz_max_of_two(row_best, scores_new[i]);
            }
            this->best_score_ = row_best;
        }
    }

    void slice_32chars(char const *second_slice, size_t i, i16_t gap,                           //
                       i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
                       i16_t *scores_new) const noexcept {

        u256_vec_t second_vec;
        u256_vec_t pre_substitution_vecs[2], pre_gap_vecs[2];
        u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        u256_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        // Initialize constats:
        u256_vec_t gap_cost_vec;
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

    void slice_1char(char const *second_slice, size_t i, i16_t gap,                           //
                     i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
                     i16_t *scores_new) const noexcept {
        // The substitution row is already addressed by the first character's class, so we only need
        // the second character's class to fetch the cost from the broadcast cost row.
        u8_t const second_class = this->substituter_.byte_to_class[(u8_t)second_slice[i]];
        i16_t const cost_of_substitution = second_class < 16 ? lookup_.row_subs_low_vec_.i8s[second_class]
                                                             : lookup_.row_subs_high_vec_.i8s[second_class - 16];
        i16_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        i16_t const if_gap = scores_pre_insertion[i] + gap;
        i16_t cell_score = sz_max_of_two(if_substitution, if_gap);
        if constexpr (locality_ == sz_similarity_local_k) cell_score = sz_max_of_two(cell_score, (i16_t)0);
        scores_new[i] = cell_score;
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<constant_iterator<char>, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_haswell_k, void>
    : public tile_scorer<constant_iterator<char>, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    class_lookup_haswell_t lookup_;

    /** @brief Executor-independent trampoline, computing one row of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        char const *second_slice, i32_t gap,                                     //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t *scores_new, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice)
            slice_32chars(second_slice, idx_slice * 32, gap, scores_pre_substitution, scores_pre_insertion, scores_new);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                             //
        constant_iterator<char> first_char, char const *second_slice, size_t n,  //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // Load a new substitution row, addressed by the class of the current first character.
        i32_t const gap = static_cast<i32_t>(this->gap_costs_.open_or_extend);
        u8_t const first_class = this->substituter_.byte_to_class[(u8_t)*first_char];
        lookup_.reload_classes(this->substituter_.byte_to_class);
        lookup_.reload_row(&this->substituter_.class_substitution_costs[first_class][0]);

        // Progress through the row 32 characters at a time.
        size_t const count_slices = n / 32;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(second_slice, gap, scores_pre_substitution, scores_pre_insertion, scores_new, from,
                                    to);
        });

        // Handle the tail with a less efficient scalar kernel.
        for (size_t i = count_slices * 32; i < n; ++i)
            slice_1char(second_slice, i, gap, scores_pre_substitution, scores_pre_insertion, scores_new);

        // Horizontally propagate the deletion cost across the last row, just like the serial scorer.
        sz_assert_(scores_pre_substitution + 1 == scores_pre_insertion && "Expects horizontal traversal of DP matrix");
        sz_assert_(scores_pre_deletion + 1 == scores_new && "Expects horizontal traversal of DP matrix");
        i32_t last_in_row = scores_pre_deletion[0];
        if constexpr (locality_ == sz_similarity_global_k) {
            for (size_t i = 0; i < n; ++i)
                scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
            this->last_score_ = last_in_row;
        }
        else {
            // In Local Alignment for SW the deletion still propagates horizontally, but every cell stays
            // non-negative and the running best across the whole matrix is the reported score.
            i32_t row_best = this->best_score_;
            for (size_t i = 0; i < n; ++i) {
                scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
                row_best = sz_max_of_two(row_best, scores_new[i]);
            }
            this->best_score_ = row_best;
        }
    }

    void slice_32chars(char const *second_slice, size_t i, i32_t gap,                           //
                       i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
                       i32_t *scores_new) const noexcept {

        u256_vec_t second_vec;
        u256_vec_t pre_substitution_vecs[4], pre_gap_vecs[4];
        u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i32_vecs[4];
        u256_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        // Initialize constats:
        u256_vec_t gap_cost_vec;
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

    void slice_1char(char const *second_slice, size_t i, i32_t gap,                           //
                     i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
                     i32_t *scores_new) const noexcept {
        u8_t const second_class = this->substituter_.byte_to_class[(u8_t)second_slice[i]];
        i32_t const cost_of_substitution = second_class < 16 ? lookup_.row_subs_low_vec_.i8s[second_class]
                                                             : lookup_.row_subs_high_vec_.i8s[second_class - 16];
        i32_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        i32_t const if_gap = scores_pre_insertion[i] + gap;
        i32_t cell_score = sz_max_of_two(if_substitution, if_gap);
        if constexpr (locality_ == sz_similarity_local_k) cell_score = sz_max_of_two(cell_score, (i32_t)0);
        scores_new[i] = cell_score;
    }
};

/** @brief No 64-bit Haswell specialization yet, fall back to the serial scalar tile scorer. */
template <sz_similarity_locality_t locality_>
struct tile_scorer<constant_iterator<char>, char const *, i64_t, error_costs_32x32_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_haswell_k>
    : public tile_scorer<constant_iterator<char>, char const *, i64_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, i64_t, error_costs_32x32_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b global Needleman-Wunsch score with class-based
 *         substitution costs over a @b diagonal walker, for inputs whose diagonal exceeds the tiny threshold.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 *
 *  Mirrors the Ice Lake `i16_t` diagonal class scorer (reversed-first / forward-second class-index buffers, the
 *  `max`/`add` recurrence `cell = max(pre_sub + sub_cost, max(pre_ins, pre_del) + gap)`), but works over 256-bit
 *  vectors with the AVX2 two-stage `substitution_lookup_haswell_t`. AVX2 has no masked loads or stores,
 *  so each diagonal is processed in full 32-cell `loadu`/`storeu` slices with a @b scalar tail epilogue, exactly
 *  like the horizontal Haswell scorers above.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_lookup_haswell_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        u256_vec_t gap_cost_vec) const noexcept {

        u256_vec_t first_vec, second_vec;
        u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        u256_vec_t pre_substitution_vecs[2], pre_insert_vecs[2], pre_delete_vecs[2];
        u256_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

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

    SZ_INLINE void slice_1cell(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new, i16_t gap) const noexcept {
        u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        i16_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                 : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        i16_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        i16_t const if_gap = sz_max_of_two(scores_pre_insertion[0], scores_pre_deletion[0]) + gap;
        scores_new[0] = sz_max_of_two(if_substitution, if_gap);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new, i16_t gap, size_t from, size_t to) const noexcept {
        u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi16(gap);
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                           //
                first_reversed_classes + progress, second_classes + progress,        //
                scores_pre_substitution + progress, scores_pre_insertion + progress, //
                scores_pre_deletion + progress, scores_new + progress, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i16_t const gap = static_cast<i16_t>(this->gap_costs_.open_or_extend);

        // AVX2 has no masked loads/stores, so we process full 32-cell slices and finish with a scalar tail.
        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap, from, to);
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
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_lookup_haswell_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        u256_vec_t gap_cost_vec) const noexcept {

        u256_vec_t first_vec, second_vec;
        u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        u256_vec_t pre_substitution_vecs[2], pre_insert_vecs[2], pre_delete_vecs[2];
        u256_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

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

    SZ_INLINE void slice_1cell(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new, i16_t gap) const noexcept {
        u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        i16_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                 : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        i16_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        i16_t const if_gap = sz_max_of_two(scores_pre_insertion[0], scores_pre_deletion[0]) + gap;
        scores_new[0] = sz_max_of_two(sz_max_of_two(if_substitution, if_gap), (i16_t)0);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new, i16_t gap, size_t from, size_t to) const noexcept {
        u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi16(gap);
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                           //
                first_reversed_classes + progress, second_classes + progress,        //
                scores_pre_substitution + progress, scores_pre_insertion + progress, //
                scores_pre_deletion + progress, scores_new + progress, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i16_t const gap = static_cast<i16_t>(this->gap_costs_.open_or_extend);
        i16_t *const scores_new_begin = scores_new;

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(first_reversed_classes + i, second_classes + i, scores_pre_substitution + i,
                        scores_pre_insertion + i, scores_pre_deletion + i, scores_new + i, gap);

        // The running best across the whole matrix is the reported local-alignment score.
        i16_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Variant of the global diagonal class scorer over @b `i32_t` cells, for inputs whose scores exceed
 *         the 16-bit range. Mirrors the `i16_t` scorer but folds the 32 class-pair costs into four 8-lane
 *         `i32` quarters via `_mm256_cvtepi8_epi32`.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_lookup_haswell_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        u256_vec_t gap_cost_vec) const noexcept {

        u256_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        u256_vec_t cost_of_substitution_i32_vecs[4];
        u256_vec_t pre_substitution_vecs[4], pre_insert_vecs[4], pre_delete_vecs[4];
        u256_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

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

    SZ_INLINE void slice_1cell(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new, i32_t gap) const noexcept {
        u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        i32_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                 : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        i32_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        i32_t const if_gap = sz_max_of_two(scores_pre_insertion[0], scores_pre_deletion[0]) + gap;
        scores_new[0] = sz_max_of_two(if_substitution, if_gap);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new, i32_t gap, size_t from, size_t to) const noexcept {
        u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi32(gap);
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                           //
                first_reversed_classes + progress, second_classes + progress,        //
                scores_pre_substitution + progress, scores_pre_insertion + progress, //
                scores_pre_deletion + progress, scores_new + progress, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t const gap = static_cast<i32_t>(this->gap_costs_.open_or_extend);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(first_reversed_classes + i, second_classes + i, scores_pre_substitution + i,
                        scores_pre_insertion + i, scores_pre_deletion + i, scores_new + i, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of the local diagonal class scorer over @b `i32_t` cells. Mirrors the `i32_t` global
 *         scorer, plus the per-cell zero-clamp and the running-best reduction of Smith-Waterman.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_lookup_haswell_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        u256_vec_t gap_cost_vec) const noexcept {

        u256_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        u256_vec_t cost_of_substitution_i32_vecs[4];
        u256_vec_t pre_substitution_vecs[4], pre_insert_vecs[4], pre_delete_vecs[4];
        u256_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

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

    SZ_INLINE void slice_1cell(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new, i32_t gap) const noexcept {
        u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        i32_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                 : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        i32_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        i32_t const if_gap = sz_max_of_two(scores_pre_insertion[0], scores_pre_deletion[0]) + gap;
        scores_new[0] = sz_max_of_two(sz_max_of_two(if_substitution, if_gap), (i32_t)0);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new, i32_t gap, size_t from, size_t to) const noexcept {
        u256_vec_t gap_cost_vec;
        gap_cost_vec.ymm = _mm256_set1_epi32(gap);
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                           //
                first_reversed_classes + progress, second_classes + progress,        //
                scores_pre_substitution + progress, scores_pre_insertion + progress, //
                scores_pre_deletion + progress, scores_new + progress, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t const gap = static_cast<i32_t>(this->gap_costs_.open_or_extend);
        i32_t *const scores_new_begin = scores_new;

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(first_reversed_classes + i, second_classes + i, scores_pre_substitution + i,
                        scores_pre_insertion + i, scores_pre_deletion + i, scores_new + i, gap);

        i32_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Haswell @b affine-gap diagonal class scorer - maximizes the global Needleman-Wunsch score over
 *         `i16_t` cells. Mirrors the linear class scorer above, but threads the separate insertion and
 *         deletion gap diagonals of the Gotoh recurrence (open vs extend) alongside the main score diagonal,
 *         exactly like the Ice Lake affine scorer, over 256-bit AVX2 vectors with a scalar tail epilogue.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_lookup_haswell_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        u256_vec_t gap_open_vec, u256_vec_t gap_expand_vec) const noexcept {

        u256_vec_t first_vec, second_vec;
        u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];

        // ? Both buffers are traversed in the same order, because one of the strings has been reversed beforehand.
        first_vec.ymm = _mm256_loadu_si256((__m256i const *)first_reversed_slice);
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)second_slice);
        cost_of_substitution_i8_vec = lookup_.lookup32(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0));
        cost_of_substitution_i16_vecs[1].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1));

        for (size_t part = 0; part != 2; ++part) {
            size_t const offset = part * 16;
            u256_vec_t pre_substitution, pre_insert_open, pre_delete_open, run_insert, run_delete;
            u256_vec_t cost_if_insert, cost_if_delete, cell_score;
            pre_substitution.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + offset));
            pre_insert_open.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + offset));
            pre_delete_open.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + offset));
            run_insert.ymm = _mm256_loadu_si256((__m256i const *)(scores_running_insertions + offset));
            run_delete.ymm = _mm256_loadu_si256((__m256i const *)(scores_running_deletions + offset));
            cost_if_insert.ymm = _mm256_max_epi16(_mm256_add_epi16(run_insert.ymm, gap_expand_vec.ymm),
                                                  _mm256_add_epi16(pre_insert_open.ymm, gap_open_vec.ymm));
            cost_if_delete.ymm = _mm256_max_epi16(_mm256_add_epi16(run_delete.ymm, gap_expand_vec.ymm),
                                                  _mm256_add_epi16(pre_delete_open.ymm, gap_open_vec.ymm));
            cell_score.ymm = _mm256_max_epi16(
                _mm256_add_epi16(pre_substitution.ymm, cost_of_substitution_i16_vecs[part].ymm),
                _mm256_max_epi16(cost_if_insert.ymm, cost_if_delete.ymm));
            _mm256_storeu_si256((__m256i *)(scores_new + offset), cell_score.ymm);
            _mm256_storeu_si256((__m256i *)(scores_new_insertions + offset), cost_if_insert.ymm);
            _mm256_storeu_si256((__m256i *)(scores_new_deletions + offset), cost_if_delete.ymm);
        }
    }

    SZ_INLINE void slice_1cell(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        i16_t gap_open, i16_t gap_extend) const noexcept {
        u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        i16_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                 : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        i16_t const if_insertion = sz_max_of_two((i16_t)(scores_running_insertions[0] + gap_extend),
                                                 (i16_t)(scores_pre_insertion[0] + gap_open));
        i16_t const if_deletion = sz_max_of_two((i16_t)(scores_running_deletions[0] + gap_extend),
                                                (i16_t)(scores_pre_deletion[0] + gap_open));
        i16_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        scores_new[0] = sz_max_of_two(if_substitution, sz_max_of_two(if_insertion, if_deletion));
        scores_new_insertions[0] = if_insertion;
        scores_new_deletions[0] = if_deletion;
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        i16_t gap_open, i16_t gap_extend, size_t from, size_t to) const noexcept {
        u256_vec_t gap_open_vec, gap_expand_vec;
        gap_open_vec.ymm = _mm256_set1_epi16(gap_open);
        gap_expand_vec.ymm = _mm256_set1_epi16(gap_extend);
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                            //
                first_reversed_classes + progress, second_classes + progress,         //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions,        //
        i16_t const *scores_running_deletions, i16_t *scores_new,                        //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i16_t const gap_open = static_cast<i16_t>(this->gap_costs_.open);
        i16_t const gap_extend = static_cast<i16_t>(this->gap_costs_.extend);

        // AVX2 has no masked loads/stores, so we process full 32-cell slices and finish with a scalar tail.
        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open, gap_extend, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(                                                //
                first_reversed_classes + i, second_classes + i,         //
                scores_pre_substitution + i, scores_pre_insertion + i,  //
                scores_pre_deletion + i, scores_running_insertions + i, //
                scores_running_deletions + i, scores_new + i,           //
                scores_new_insertions + i, scores_new_deletions + i,    //
                gap_open, gap_extend);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Haswell @b affine-gap diagonal class scorer - maximizes the local Smith-Waterman score over
 *         `i16_t` cells. Mirrors the global affine class scorer above, but adds the Smith-Waterman-Gotoh
 *         zero-reset on @b only the substitution term and tracks the running best across the whole matrix.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_lookup_haswell_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        u256_vec_t gap_open_vec, u256_vec_t gap_expand_vec) const noexcept {

        u256_vec_t first_vec, second_vec;
        u256_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];

        first_vec.ymm = _mm256_loadu_si256((__m256i const *)first_reversed_slice);
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)second_slice);
        cost_of_substitution_i8_vec = lookup_.lookup32(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0));
        cost_of_substitution_i16_vecs[1].ymm = _mm256_cvtepi8_epi16(
            _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1));

        for (size_t part = 0; part != 2; ++part) {
            size_t const offset = part * 16;
            u256_vec_t pre_substitution, pre_insert_open, pre_delete_open, run_insert, run_delete;
            u256_vec_t cost_if_insert, cost_if_delete, cell_score;
            pre_substitution.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + offset));
            pre_insert_open.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + offset));
            pre_delete_open.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + offset));
            run_insert.ymm = _mm256_loadu_si256((__m256i const *)(scores_running_insertions + offset));
            run_delete.ymm = _mm256_loadu_si256((__m256i const *)(scores_running_deletions + offset));
            cost_if_insert.ymm = _mm256_max_epi16(_mm256_add_epi16(run_insert.ymm, gap_expand_vec.ymm),
                                                  _mm256_add_epi16(pre_insert_open.ymm, gap_open_vec.ymm));
            cost_if_delete.ymm = _mm256_max_epi16(_mm256_add_epi16(run_delete.ymm, gap_expand_vec.ymm),
                                                  _mm256_add_epi16(pre_delete_open.ymm, gap_open_vec.ymm));
            // In Local Alignment for SW the zero-reset is applied to @b only the substitution term;
            // the insertion/deletion gap matrices are not clamped, exactly like the serial scorer.
            cell_score.ymm = _mm256_max_epi16(
                _mm256_max_epi16(_mm256_add_epi16(pre_substitution.ymm, cost_of_substitution_i16_vecs[part].ymm),
                                 _mm256_setzero_si256()),
                _mm256_max_epi16(cost_if_insert.ymm, cost_if_delete.ymm));
            _mm256_storeu_si256((__m256i *)(scores_new + offset), cell_score.ymm);
            _mm256_storeu_si256((__m256i *)(scores_new_insertions + offset), cost_if_insert.ymm);
            _mm256_storeu_si256((__m256i *)(scores_new_deletions + offset), cost_if_delete.ymm);
        }
    }

    SZ_INLINE void slice_1cell(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        i16_t gap_open, i16_t gap_extend) const noexcept {
        u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        i16_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                 : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        i16_t const if_insertion = sz_max_of_two((i16_t)(scores_running_insertions[0] + gap_extend),
                                                 (i16_t)(scores_pre_insertion[0] + gap_open));
        i16_t const if_deletion = sz_max_of_two((i16_t)(scores_running_deletions[0] + gap_extend),
                                                (i16_t)(scores_pre_deletion[0] + gap_open));
        i16_t const if_substitution = sz_max_of_two((i16_t)(scores_pre_substitution[0] + sub_cost), (i16_t)0);
        scores_new[0] = sz_max_of_two(if_substitution, sz_max_of_two(if_insertion, if_deletion));
        scores_new_insertions[0] = if_insertion;
        scores_new_deletions[0] = if_deletion;
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        i16_t gap_open, i16_t gap_extend, size_t from, size_t to) const noexcept {
        u256_vec_t gap_open_vec, gap_expand_vec;
        gap_open_vec.ymm = _mm256_set1_epi16(gap_open);
        gap_expand_vec.ymm = _mm256_set1_epi16(gap_extend);
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                            //
                first_reversed_classes + progress, second_classes + progress,         //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions,        //
        i16_t const *scores_running_deletions, i16_t *scores_new,                        //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i16_t const gap_open = static_cast<i16_t>(this->gap_costs_.open);
        i16_t const gap_extend = static_cast<i16_t>(this->gap_costs_.extend);
        i16_t *const scores_new_begin = scores_new;

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open, gap_extend, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(                                                //
                first_reversed_classes + i, second_classes + i,         //
                scores_pre_substitution + i, scores_pre_insertion + i,  //
                scores_pre_deletion + i, scores_running_insertions + i, //
                scores_running_deletions + i, scores_new + i,           //
                scores_new_insertions + i, scores_new_deletions + i,    //
                gap_open, gap_extend);

        // The running best across the whole matrix is the reported local-alignment score.
        i16_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Haswell @b affine-gap diagonal class scorer - maximizes the global Needleman-Wunsch score over
 *         `i32_t` cells, for inputs whose scores exceed the 16-bit range. Mirrors the `i16_t` affine
 *         scorer but folds the 32 class-pair costs into four 8-lane `i32` quarters via `_mm256_cvtepi8_epi32`.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_lookup_haswell_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        u256_vec_t gap_open_vec, u256_vec_t gap_expand_vec) const noexcept {

        u256_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        u256_vec_t cost_of_substitution_i32_vecs[4];

        first_vec.ymm = _mm256_loadu_si256((__m256i const *)first_reversed_slice);
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)second_slice);

        // Sign-extend the 32 class-pair costs into four 8-lane `i32` quarters.
        cost_of_substitution_i8_vec = lookup_.lookup32(first_vec, second_vec);
        __m128i cost_low_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0);
        __m128i cost_high_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1);
        cost_of_substitution_i32_vecs[0].ymm = _mm256_cvtepi8_epi32(cost_low_xmm);
        cost_of_substitution_i32_vecs[1].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_low_xmm, 8));
        cost_of_substitution_i32_vecs[2].ymm = _mm256_cvtepi8_epi32(cost_high_xmm);
        cost_of_substitution_i32_vecs[3].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_high_xmm, 8));

        for (size_t part = 0; part != 4; ++part) {
            size_t const offset = part * 8;
            u256_vec_t pre_substitution, pre_insert_open, pre_delete_open, run_insert, run_delete;
            u256_vec_t cost_if_insert, cost_if_delete, cell_score;
            pre_substitution.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + offset));
            pre_insert_open.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + offset));
            pre_delete_open.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + offset));
            run_insert.ymm = _mm256_loadu_si256((__m256i const *)(scores_running_insertions + offset));
            run_delete.ymm = _mm256_loadu_si256((__m256i const *)(scores_running_deletions + offset));
            cost_if_insert.ymm = _mm256_max_epi32(_mm256_add_epi32(run_insert.ymm, gap_expand_vec.ymm),
                                                  _mm256_add_epi32(pre_insert_open.ymm, gap_open_vec.ymm));
            cost_if_delete.ymm = _mm256_max_epi32(_mm256_add_epi32(run_delete.ymm, gap_expand_vec.ymm),
                                                  _mm256_add_epi32(pre_delete_open.ymm, gap_open_vec.ymm));
            cell_score.ymm = _mm256_max_epi32(
                _mm256_add_epi32(pre_substitution.ymm, cost_of_substitution_i32_vecs[part].ymm),
                _mm256_max_epi32(cost_if_insert.ymm, cost_if_delete.ymm));
            _mm256_storeu_si256((__m256i *)(scores_new + offset), cell_score.ymm);
            _mm256_storeu_si256((__m256i *)(scores_new_insertions + offset), cost_if_insert.ymm);
            _mm256_storeu_si256((__m256i *)(scores_new_deletions + offset), cost_if_delete.ymm);
        }
    }

    SZ_INLINE void slice_1cell(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        i32_t gap_open, i32_t gap_extend) const noexcept {
        u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        i32_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                 : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        i32_t const if_insertion = sz_max_of_two(scores_running_insertions[0] + gap_extend,
                                                 scores_pre_insertion[0] + gap_open);
        i32_t const if_deletion = sz_max_of_two(scores_running_deletions[0] + gap_extend,
                                                scores_pre_deletion[0] + gap_open);
        i32_t const if_substitution = scores_pre_substitution[0] + sub_cost;
        scores_new[0] = sz_max_of_two(if_substitution, sz_max_of_two(if_insertion, if_deletion));
        scores_new_insertions[0] = if_insertion;
        scores_new_deletions[0] = if_deletion;
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        i32_t gap_open, i32_t gap_extend, size_t from, size_t to) const noexcept {
        u256_vec_t gap_open_vec, gap_expand_vec;
        gap_open_vec.ymm = _mm256_set1_epi32(gap_open);
        gap_expand_vec.ymm = _mm256_set1_epi32(gap_extend);
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                            //
                first_reversed_classes + progress, second_classes + progress,         //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions,        //
        i32_t const *scores_running_deletions, i32_t *scores_new,                        //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t const gap_open = static_cast<i32_t>(this->gap_costs_.open);
        i32_t const gap_extend = static_cast<i32_t>(this->gap_costs_.extend);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open, gap_extend, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(                                                //
                first_reversed_classes + i, second_classes + i,         //
                scores_pre_substitution + i, scores_pre_insertion + i,  //
                scores_pre_deletion + i, scores_running_insertions + i, //
                scores_running_deletions + i, scores_new + i,           //
                scores_new_insertions + i, scores_new_deletions + i,    //
                gap_open, gap_extend);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Haswell @b affine-gap diagonal class scorer - maximizes the local Smith-Waterman score over
 *         `i32_t` cells. Mirrors the `i32_t` global affine scorer, plus the Smith-Waterman-Gotoh
 *         zero-reset on the substitution term and the running-best reduction.
 *  @note Requires AVX2-capable Haswell generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    substitution_lookup_haswell_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_32cells(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        u256_vec_t gap_open_vec, u256_vec_t gap_expand_vec) const noexcept {

        u256_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        u256_vec_t cost_of_substitution_i32_vecs[4];

        first_vec.ymm = _mm256_loadu_si256((__m256i const *)first_reversed_slice);
        second_vec.ymm = _mm256_loadu_si256((__m256i const *)second_slice);

        cost_of_substitution_i8_vec = lookup_.lookup32(first_vec, second_vec);
        __m128i cost_low_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 0);
        __m128i cost_high_xmm = _mm256_extracti128_si256(cost_of_substitution_i8_vec.ymm, 1);
        cost_of_substitution_i32_vecs[0].ymm = _mm256_cvtepi8_epi32(cost_low_xmm);
        cost_of_substitution_i32_vecs[1].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_low_xmm, 8));
        cost_of_substitution_i32_vecs[2].ymm = _mm256_cvtepi8_epi32(cost_high_xmm);
        cost_of_substitution_i32_vecs[3].ymm = _mm256_cvtepi8_epi32(_mm_srli_si128(cost_high_xmm, 8));

        for (size_t part = 0; part != 4; ++part) {
            size_t const offset = part * 8;
            u256_vec_t pre_substitution, pre_insert_open, pre_delete_open, run_insert, run_delete;
            u256_vec_t cost_if_insert, cost_if_delete, cell_score;
            pre_substitution.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_substitution + offset));
            pre_insert_open.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_insertion + offset));
            pre_delete_open.ymm = _mm256_loadu_si256((__m256i const *)(scores_pre_deletion + offset));
            run_insert.ymm = _mm256_loadu_si256((__m256i const *)(scores_running_insertions + offset));
            run_delete.ymm = _mm256_loadu_si256((__m256i const *)(scores_running_deletions + offset));
            cost_if_insert.ymm = _mm256_max_epi32(_mm256_add_epi32(run_insert.ymm, gap_expand_vec.ymm),
                                                  _mm256_add_epi32(pre_insert_open.ymm, gap_open_vec.ymm));
            cost_if_delete.ymm = _mm256_max_epi32(_mm256_add_epi32(run_delete.ymm, gap_expand_vec.ymm),
                                                  _mm256_add_epi32(pre_delete_open.ymm, gap_open_vec.ymm));
            // In Local Alignment for SW the zero-reset is applied to @b only the substitution term;
            // the insertion/deletion gap matrices are not clamped, exactly like the serial scorer.
            cell_score.ymm = _mm256_max_epi32(
                _mm256_max_epi32(_mm256_add_epi32(pre_substitution.ymm, cost_of_substitution_i32_vecs[part].ymm),
                                 _mm256_setzero_si256()),
                _mm256_max_epi32(cost_if_insert.ymm, cost_if_delete.ymm));
            _mm256_storeu_si256((__m256i *)(scores_new + offset), cell_score.ymm);
            _mm256_storeu_si256((__m256i *)(scores_new_insertions + offset), cost_if_insert.ymm);
            _mm256_storeu_si256((__m256i *)(scores_new_deletions + offset), cost_if_delete.ymm);
        }
    }

    SZ_INLINE void slice_1cell(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        i32_t gap_open, i32_t gap_extend) const noexcept {
        u8_t const first_class = first_reversed_slice[0], second_class = second_slice[0];
        // Read the resident cost rows so the scalar tail honors the same transpose as the vector `lookup32`.
        i32_t const sub_cost = second_class < 16 ? lookup_.cost_rows_low_vecs_[first_class].i8s[second_class]
                                                 : lookup_.cost_rows_high_vecs_[first_class].i8s[second_class - 16];
        i32_t const if_insertion = sz_max_of_two(scores_running_insertions[0] + gap_extend,
                                                 scores_pre_insertion[0] + gap_open);
        i32_t const if_deletion = sz_max_of_two(scores_running_deletions[0] + gap_extend,
                                                scores_pre_deletion[0] + gap_open);
        i32_t const if_substitution = sz_max_of_two(scores_pre_substitution[0] + sub_cost, 0);
        scores_new[0] = sz_max_of_two(if_substitution, sz_max_of_two(if_insertion, if_deletion));
        scores_new_insertions[0] = if_insertion;
        scores_new_deletions[0] = if_deletion;
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        i32_t gap_open, i32_t gap_extend, size_t from, size_t to) const noexcept {
        u256_vec_t gap_open_vec, gap_expand_vec;
        gap_open_vec.ymm = _mm256_set1_epi32(gap_open);
        gap_expand_vec.ymm = _mm256_set1_epi32(gap_extend);
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_32cells(                                                            //
                first_reversed_classes + progress, second_classes + progress,         //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions,        //
        i32_t const *scores_running_deletions, i32_t *scores_new,                        //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t const gap_open = static_cast<i32_t>(this->gap_costs_.open);
        i32_t const gap_extend = static_cast<i32_t>(this->gap_costs_.extend);
        i32_t *const scores_new_begin = scores_new;

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open, gap_extend, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1cell(                                                //
                first_reversed_classes + i, second_classes + i,         //
                scores_pre_substitution + i, scores_pre_insertion + i,  //
                scores_pre_deletion + i, scores_running_insertions + i, //
                scores_running_deletions + i, scores_new + i,           //
                scores_new_insertions + i, scores_new_deletions + i,    //
                gap_open, gap_extend);

        // The running best across the whole matrix is the reported local-alignment score.
        i32_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/** @brief Redirects the Haswell template specialization to the serial version. */
template <typename char_type_, typename score_type_, typename substituter_type_, typename gap_costs_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                         sz_cap_haswell_k, void>
    : public horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                               sz_cap_serial_k, void> {

    using base_t = horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                                     sz_cap_serial_k, void>;

    using base_t::base_t;
    using base_t::operator();
};

/**
 *  @brief Haswell horizontal "walker" for class-based substitution costs with linear gaps.
 *         Mirrors the serial walker, but binds its `tile_scorer_t` to the AVX2 class lookup above,
 *         which the generic redirect above would otherwise leave on the scalar serial path.
 */
template <typename char_type_, typename score_type_, sz_similarity_objective_t objective_,
          sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, error_costs_32x32_t, linear_gap_costs_t, objective_, locality_,
                         sz_cap_haswell_k, void> {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    using tile_scorer_t = tile_scorer<constant_iterator<char_t>, char_t const *, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    horizontal_walker() noexcept {}
    horizontal_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

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

        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;
        size_t const padded_shorter_dim = round_up_to_multiple(sizeof(score_t) * shorter_dim, specs.cache_line_width) /
                                          sizeof(score_t);
        size_t const scratch_required = sizeof(score_t) * padded_shorter_dim * 2;
        if (scratch_space.size() < scratch_required) return status_t::bad_alloc_k;

        score_t *previous_scores = (score_t *)scratch_space.data();
        score_t *current_scores = previous_scores + padded_shorter_dim;

        // Initialize the first row:
        tile_scorer_t scorer {substituter_, gap_costs_};
        for (size_t col_idx = 0; col_idx < shorter_dim; ++col_idx) scorer.init_score(previous_scores[col_idx], col_idx);

        // Progress through the matrix row-by-row:
        for (size_t row_idx = 1; row_idx < longer_dim; ++row_idx) {
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
            trivial_swap(previous_scores, current_scores);
        }

        result_ref = scorer.score();
        return status_t::success_k;
    }
};

/**
 *  @brief Haswell diagonal "walker" for class-based substitution costs with linear gaps. Mirrors the Ice Lake
 *         diagonal walker: it pre-classifies both strings @b once into class-index buffers and keeps a 3-diagonal
 *         buffer, but feeds the AVX2 `substitution_lookup_haswell_t` instead of the `VPERMB` one.
 *
 *  When the diagonal walker swaps the shorter and longer strings, the order of the two class operands flips;
 *  to keep the recurrence reading the original `class_substitution_costs[first][second]`, the swap bit is fed
 *  into `tile_scorer_t::prepare`, which folds the resident table transposed (one-time, off the hot path).
 */
template <typename score_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct diagonal_walker<char, score_type_, error_costs_32x32_t, linear_gap_costs_t, objective_, locality_,
                       sz_cap_haswell_k, void> {

    using char_t = char;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    diagonal_walker() noexcept {}
    diagonal_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Byte offsets of this walker's scratch sub-buffers within its `scratch_space`. */
    struct layout_t {
        size_t previous_scores = 0; // ? The 3 rotating score diagonals.
        size_t current_scores = 0;
        size_t next_scores = 0;
        size_t shorter_reversed = 0;         // ? Reversed shorter string (carries a step_classes_k overread guard).
        size_t shorter_reversed_classes = 0; // ? Its pre-classified class indices.
        size_t longer_classes = 0;           // ? The longer string's pre-classified class indices.
        size_t total = 0;                    // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        size_t const longer_length = sz_max_of_two(first.size(), second.size());
        size_t const diagonal_bytes = sizeof(score_t) * (shorter_length + 1); // one anti-diagonal, unpadded
        size_t const shorter_stream_bytes = shorter_length + step_classes_k;  // string + SIMD overread guard
        size_t const longer_stream_bytes = longer_length + step_classes_k;
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.previous_scores = amount, amount += diagonal_bytes;
        at.current_scores = amount, amount += diagonal_bytes;
        at.next_scores = amount, amount += diagonal_bytes;
        at.shorter_reversed = amount, amount += shorter_stream_bytes;
        at.shorter_reversed_classes = amount, amount += shorter_stream_bytes;
        at.longer_classes = amount, amount += longer_stream_bytes;
        at.total = amount;
        return at;
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

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

        // One `layout()` describes every sub-buffer (3 diagonals, the reversed shorter string, and both
        // class-index streams, each padded for the step_classes_k SIMD overread). We validate the walker's own
        // footprint and place the pointers from it.
        layout_t const at = layout(first, second, specs);
        if (scratch_space.size() < at.total) return status_t::bad_alloc_k;
        score_t *previous_scores = (score_t *)(scratch_space.data() + at.previous_scores);
        score_t *current_scores = (score_t *)(scratch_space.data() + at.current_scores);
        score_t *next_scores = (score_t *)(scratch_space.data() + at.next_scores);
        char_t *const shorter_reversed = (char_t *)(scratch_space.data() + at.shorter_reversed);
        char_t *const shorter_reversed_classes = (char_t *)(scratch_space.data() + at.shorter_reversed_classes);
        char_t *const longer_classes = (char_t *)(scratch_space.data() + at.longer_classes);

        // Export the reversed shorter string, then classify both strings @b once into their class-index buffers.
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
            sz_move_serial((ptr_t)(previous_scores), (ptr_t)(previous_scores + 1),
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

        result_ref = scorer.score();
        return status_t::success_k;
    }

    static constexpr size_t step_classes_k = 32;

  private:
    /** @brief Maps a raw byte string into class bytes using the resident `byte_to_class` lookup, @b amortized. */
    static void classify_into_(substitution_lookup_haswell_t const &lookup, char_t const *source, size_t length,
                               char_t *classes) noexcept {
        u256_vec_t source_vec, classes_vec;
        size_t progress = 0;
        for (; progress + step_classes_k <= length; progress += step_classes_k) {
            source_vec.ymm = _mm256_loadu_si256((__m256i const *)(source + progress));
            classes_vec = lookup.classify32(source_vec);
            _mm256_storeu_si256((__m256i *)(classes + progress), classes_vec.ymm);
        }
        // AVX2 has no masked store, so the tail is classified one byte at a time.
        for (; progress != length; ++progress)
            classes[progress] = (char_t)lookup.byte_to_class_group_vecs_[((u8_t)source[progress]) >> 4]
                                    .u8s[((u8_t)source[progress]) & 0x0f];
    }
};

/**
 *  @brief Haswell diagonal "walker" for class-based substitution costs with @b affine gaps. Mirrors the linear
 *         class walker above for the classify-once + `prepare(transpose)` machinery, but threads the 7-diagonal
 *         Gotoh layout (3 main score diagonals + 2 insertion + 2 deletion) of the serial affine walker, feeding
 *         the AVX2 affine diagonal scorers (5-in / 3-out).
 *
 *  When the diagonal walker swaps the shorter and longer strings, the order of the two class operands flips;
 *  to keep the recurrence reading the original `class_substitution_costs[first][second]`, the swap bit is fed
 *  into `tile_scorer_t::prepare`, which folds the resident table transposed (one-time, off the hot path).
 */
template <typename score_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct diagonal_walker<char, score_type_, error_costs_32x32_t, affine_gap_costs_t, objective_, locality_,
                       sz_cap_haswell_k, void> {

    using char_t = char;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    diagonal_walker() noexcept {}
    diagonal_walker(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Byte offsets of this walker's scratch sub-buffers within its `scratch_space`. */
    struct layout_t {
        size_t previous_scores = 0; // ? The 3 rotating score diagonals.
        size_t current_scores = 0;
        size_t next_scores = 0;
        size_t current_inserts = 0; // ? The 2 rotating insertion-gap diagonals.
        size_t next_inserts = 0;
        size_t current_deletes = 0; // ? The 2 rotating deletion-gap diagonals.
        size_t next_deletes = 0;
        size_t shorter_reversed = 0;         // ? Reversed shorter string (carries a step_classes_k overread guard).
        size_t shorter_reversed_classes = 0; // ? Its pre-classified class indices.
        size_t longer_classes = 0;           // ? The longer string's pre-classified class indices.
        size_t total = 0;                    // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        size_t const longer_length = sz_max_of_two(first.size(), second.size());
        size_t const diagonal_bytes = sizeof(score_t) * (shorter_length + 1); // one anti-diagonal, unpadded
        size_t const shorter_stream_bytes = shorter_length + step_classes_k;  // string + SIMD overread guard
        size_t const longer_stream_bytes = longer_length + step_classes_k;
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.previous_scores = amount, amount += diagonal_bytes;
        at.current_scores = amount, amount += diagonal_bytes;
        at.next_scores = amount, amount += diagonal_bytes;
        at.current_inserts = amount, amount += diagonal_bytes;
        at.next_inserts = amount, amount += diagonal_bytes;
        at.current_deletes = amount, amount += diagonal_bytes;
        at.next_deletes = amount, amount += diagonal_bytes;
        at.shorter_reversed = amount, amount += shorter_stream_bytes;
        at.shorter_reversed_classes = amount, amount += shorter_stream_bytes;
        at.longer_classes = amount, amount += longer_stream_bytes;
        at.total = amount;
        return at;
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) {
                    result_ref = gap_costs_.open + gap_costs_.extend * (first.size() - 1);
                }
                else if (first.empty() && !second.empty()) {
                    result_ref = gap_costs_.open + gap_costs_.extend * (second.size() - 1);
                }
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

        // We are going to store 7 diagonals of the matrix (3 main + 2 insert + 2 delete).
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;
        size_t const diagonals_count = shorter_dim + longer_dim - 1;
        size_t const max_diagonal_length = shorter_length + 1;

        // One `layout()` describes every sub-buffer (7 diagonals, the reversed shorter string, and both
        // class-index streams, each padded for the step_classes_k SIMD overread). We validate the walker's own
        // footprint and place the pointers from it.
        layout_t const at = layout(first, second, specs);
        if (scratch_space.size() < at.total) return status_t::bad_alloc_k;
        score_t *previous_scores = (score_t *)(scratch_space.data() + at.previous_scores);
        score_t *current_scores = (score_t *)(scratch_space.data() + at.current_scores);
        score_t *next_scores = (score_t *)(scratch_space.data() + at.next_scores);
        score_t *current_inserts = (score_t *)(scratch_space.data() + at.current_inserts);
        score_t *next_inserts = (score_t *)(scratch_space.data() + at.next_inserts);
        score_t *current_deletes = (score_t *)(scratch_space.data() + at.current_deletes);
        score_t *next_deletes = (score_t *)(scratch_space.data() + at.next_deletes);
        char_t *const shorter_reversed = (char_t *)(scratch_space.data() + at.shorter_reversed);
        char_t *const shorter_reversed_classes = (char_t *)(scratch_space.data() + at.shorter_reversed_classes);
        char_t *const longer_classes = (char_t *)(scratch_space.data() + at.longer_classes);

        // Export the reversed shorter string, then classify both strings @b once into their class-index buffers.
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
        scorer.init_gap(current_inserts[0], 1);
        scorer.init_gap(current_deletes[1], 1);

        size_t next_diagonal_index = 2;

        // Progress through the upper-left triangle of the matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = next_diagonal_index + 1;
            scorer(                                                                  //
                shorter_reversed_classes + shorter_length - next_diagonal_index + 1, // first sequence of classes
                longer_classes,                                                      // second sequence of classes
                next_diagonal_length - 2,             // number of elements to compute with the `scorer`
                previous_scores,                      // costs pre substitution
                current_scores, current_scores + 1,   // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1, // costs pre insertion/deletion extension
                next_scores + 1,                      // updated similarity scores
                next_inserts + 1, next_deletes + 1,   // updated insertion/deletion extensions
                executor);                            // parallel execution within the diagonal

            // Don't forget to populate the first row and the first column of the matrix.
            scorer.init_score(next_scores[0], next_diagonal_index);
            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            scorer.init_gap(next_inserts[0], next_diagonal_index);
            scorer.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);

            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);
        }

        // Now let's handle the anti-diagonal band of the matrix.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = shorter_dim;
            scorer(                                                          //
                shorter_reversed_classes + shorter_length - shorter_dim + 1, // first sequence of classes
                longer_classes + next_diagonal_index - shorter_dim,          // second sequence of classes
                next_diagonal_length - 1,                                    // number of elements to compute
                previous_scores,                                             // costs pre substitution
                current_scores, current_scores + 1,                          // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,                        // costs pre insertion/deletion extension
                next_scores,                                                 // updated similarity scores
                next_inserts, next_deletes,                                  // updated insertion/deletion extensions
                executor);                                                   // parallel execution within the diagonal

            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            scorer.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);

            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);
            sz_move_serial((ptr_t)(previous_scores), (ptr_t)(previous_scores + 1),
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
                current_scores, current_scores + 1,                          // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,                        // costs pre insertion/deletion extension
                next_scores,                                                 // updated similarity scores
                next_inserts, next_deletes,                                  // updated insertion/deletion extensions
                executor);                                                   // parallel execution within the diagonal

            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);
            previous_scores++;
        }

        result_ref = scorer.score();
        return status_t::success_k;
    }

    static constexpr size_t step_classes_k = 32;

  private:
    /** @brief Maps a raw byte string into class bytes using the resident `byte_to_class` lookup, @b amortized. */
    static void classify_into_(substitution_lookup_haswell_t const &lookup, char_t const *source, size_t length,
                               char_t *classes) noexcept {
        u256_vec_t source_vec, classes_vec;
        size_t progress = 0;
        for (; progress + step_classes_k <= length; progress += step_classes_k) {
            source_vec.ymm = _mm256_loadu_si256((__m256i const *)(source + progress));
            classes_vec = lookup.classify32(source_vec);
            _mm256_storeu_si256((__m256i *)(classes + progress), classes_vec.ymm);
        }
        // AVX2 has no masked store, so the tail is classified one byte at a time.
        for (; progress != length; ++progress)
            classes[progress] = (char_t)lookup.byte_to_class_group_vecs_[((u8_t)source[progress]) >> 4]
                                    .u8s[((u8_t)source[progress]) & 0x0f];
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score between two strings using the Haswell backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sh_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 3;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_haswell_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_haswell_k>;
    using diagonal_i64_t = diagonal_walker<char_t, i64_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    needleman_wunsch_score() noexcept {}
    needleman_wunsch_score(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = std::min(first.size(), second.size());
        size_t const longer_length = std::max(first.size(), second.size());
        size_t const max_diagonal_length = shorter_length + 1;
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(i64_t) * max_diagonal_length, specs.cache_line_width) / sizeof(i64_t);
        size_t const padded_shorter_stream_length = round_up_to_multiple(
            shorter_length + diagonal_i16_t::step_classes_k, specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + diagonal_i16_t::step_classes_k,
                                                                        specs.cache_line_width);
        return sizeof(i64_t) * padded_diagonal_length * diagonal_buffers_count_k + padded_shorter_stream_length * 2 +
               padded_longer_stream_length;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status_t status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status_t status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status_t status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space,
                                                                        executor, specs);
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
template <>
struct smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sh_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 3;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, linear_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_haswell_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, linear_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_haswell_k>;
    using diagonal_i64_t = diagonal_walker<char_t, i64_t, substituter_t, linear_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    smith_waterman_score() noexcept {}
    smith_waterman_score(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = std::min(first.size(), second.size());
        size_t const longer_length = std::max(first.size(), second.size());
        size_t const max_diagonal_length = shorter_length + 1;
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(i64_t) * max_diagonal_length, specs.cache_line_width) / sizeof(i64_t);
        size_t const padded_shorter_stream_length = round_up_to_multiple(
            shorter_length + diagonal_i16_t::step_classes_k, specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + diagonal_i16_t::step_classes_k,
                                                                        specs.cache_line_width);
        return sizeof(i64_t) * padded_diagonal_length * diagonal_buffers_count_k + padded_shorter_stream_length * 2 +
               padded_longer_stream_length;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status_t status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status_t status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status_t status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score with @b affine gaps using the Haswell backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sh_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 7;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_haswell_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_haswell_k>;
    using diagonal_i64_t = diagonal_walker<char_t, i64_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    needleman_wunsch_score() noexcept {}
    needleman_wunsch_score(substituter_t subs, affine_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = std::min(first.size(), second.size());
        size_t const longer_length = std::max(first.size(), second.size());
        size_t const max_diagonal_length = shorter_length + 1;
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(i64_t) * max_diagonal_length, specs.cache_line_width) / sizeof(i64_t);
        size_t const padded_shorter_stream_length = round_up_to_multiple(
            shorter_length + diagonal_i16_t::step_classes_k, specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + diagonal_i16_t::step_classes_k,
                                                                        specs.cache_line_width);
        return sizeof(i64_t) * padded_diagonal_length * diagonal_buffers_count_k + padded_shorter_stream_length * 2 +
               padded_longer_stream_length;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status_t status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status_t status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status_t status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Computes the @b byte-level Smith-Waterman score with @b affine gaps using the Haswell backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sh_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 7;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, affine_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_haswell_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, affine_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_haswell_k>;
    using diagonal_i64_t = diagonal_walker<char_t, i64_t, substituter_t, affine_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    smith_waterman_score() noexcept {}
    smith_waterman_score(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = std::min(first.size(), second.size());
        size_t const longer_length = std::max(first.size(), second.size());
        size_t const max_diagonal_length = shorter_length + 1;
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(i64_t) * max_diagonal_length, specs.cache_line_width) / sizeof(i64_t);
        size_t const padded_shorter_stream_length = round_up_to_multiple(
            shorter_length + diagonal_i16_t::step_classes_k, specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + diagonal_i16_t::step_classes_k,
                                                                        specs.cache_line_width);
        return sizeof(i64_t) * padded_diagonal_length * diagonal_buffers_count_k + padded_shorter_stream_length * 2 +
               padded_longer_stream_length;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status_t status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status_t status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status_t status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i64;
        }

        return status_t::success_k;
    }
};

#pragma region - Inter-Sequence Candidate Lanes

/**
 *  @brief Inter-sequence Haswell walker: one query against up to 32 candidates packed one-per-lane.
 *
 *  Computes the @b global unit-cost Levenshtein distance of a single shared query against a transposed
 *  `candidate_lanes_block_t` of up to 32 candidates. Each `__m256i` holds 32 `u8` cells - lane @p lane_index
 *  carries that candidate's running Dynamic Programming column. The query characters index the rows; for every
 *  row the candidate column is broadcast-compared against the query character and the SWIPE recurrence
 *  `cell = min(substitution, min(deletion, insertion))` advances all 32 lanes in lockstep. This is the AVX2 twin
 *  of the 64-lane Ice Lake `u8` kernel; AVX2 has no mask registers, so the mismatch penalty is folded in with
 *  `_mm256_cmpeq_epi8` (yielding `0xFF` on a match, `0x00` on a mismatch) added to `set1_epi8(1)` to obtain a
 *  per-lane addend of `0` on match and `1` on mismatch.
 *
 *  @note The cells are `u8` and the recurrence's `cost + 1` intermediate wraps once a cell reaches 255, so this
 *      kernel is only valid when the query and every candidate are at most 254 characters long (the all-gap path
 *      sets the longest reachable cell). Enforcing that bound is the caller's dispatch contract; the kernel
 *      performs no runtime length check.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u8_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 32, void> {

    using char_t = char;
    using score_t = u8_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 32;

    // The `u8` lane recurrence hardcodes `_mm256_min_epu8`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 8-bit candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (32 `u8` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 32 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 32`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        __m256i const one_vec = _mm256_set1_epi8(1);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                _mm256_set1_epi8(static_cast<char>(static_cast<u8_t>(column))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m256i const query_char_vec = _mm256_set1_epi8(query[query_position - 1]);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row),
                                _mm256_set1_epi8(static_cast<char>(static_cast<u8_t>(query_position))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m256i const candidate_chars_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const deletion_source_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const insertion_source_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                // AVX2 has no mask registers: `cmpeq` yields `0xFF` (-1) on a match, `0x00` on a mismatch.
                // Adding `1` collapses that to `0` on match and `1` on mismatch - the substitution penalty.
                __m256i const equal_vec = _mm256_cmpeq_epi8(query_char_vec, candidate_chars_vec);
                __m256i const mismatch_addend_vec = _mm256_add_epi8(one_vec, equal_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi8(diagonal_vec, mismatch_addend_vec);
                __m256i const cost_if_deletion_vec = _mm256_add_epi8(deletion_source_vec, one_vec);
                __m256i const cost_if_insertion_vec = _mm256_add_epi8(insertion_source_vec, one_vec);
                __m256i const cell_score_vec = _mm256_min_epu8(
                    cost_if_substitution_vec, _mm256_min_epu8(cost_if_deletion_vec, cost_if_insertion_vec));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row + column * candidate_lanes_k),
                                    cell_score_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        // Latch each live lane's result from its own final column; ragged lengths mean different columns per lane.
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
            size_t const candidate_length = candidates.lengths[lane_index];
            result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
        }
        return status_t::success_k;
    }
};

/**
 *  @brief Inter-sequence Haswell walker: one query against up to 16 candidates packed one-per-lane, 16-bit cells.
 *
 *  Structural twin of the 32-lane `u8` walker, widened to `u16` cells so each `__m256i` holds 16 lanes. The
 *  candidate characters remain `char`/`u8`: 16 of them load into the low half of a `__m128i` and compare against
 *  the broadcast query character with `_mm_cmpeq_epi8`, yielding 16 bytes of `0xFF` (match) or `0x00` (mismatch)
 *  that sign-extend to 16 `i16` lanes; adding `1` collapses that to a per-lane substitution penalty of `0` on
 *  match and `1` on mismatch. The SWIPE recurrence `cell = min(substitution, min(deletion, insertion))` advances
 *  all 16 lanes in lockstep over `_mm256_min_epu16`.
 *
 *  @note The cells are `u16` and the recurrence's `cost + 1` intermediate wraps once a cell reaches 65535, so this
 *      kernel is only valid when the query and every candidate are at most 65534 characters long (the all-gap path
 *      sets the longest reachable cell). Enforcing that bound is the caller's dispatch contract; the kernel
 *      performs no runtime length check.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 16, void> {

    using char_t = char;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 16;

    // The `u16` lane recurrence hardcodes `_mm256_min_epu16`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 16-bit candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (16 `u16` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 16 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 16`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        __m256i const one_vec = _mm256_set1_epi16(1);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                _mm256_set1_epi16(static_cast<short>(static_cast<u16_t>(column))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m128i const query_char_vec = _mm_set1_epi8(query[query_position - 1]);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row),
                                _mm256_set1_epi16(static_cast<short>(static_cast<u16_t>(query_position))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                // Only 16 candidate bytes are live this column; load them into the low half of a `__m128i`.
                __m128i const candidate_chars_vec =
                    _mm_loadu_si128(reinterpret_cast<__m128i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const deletion_source_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const insertion_source_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                // `cmpeq` yields `0xFF` (-1) on a match, `0x00` on a mismatch, per byte; sign-extend to 16 `i16`.
                // Adding `1` collapses that to `0` on match and `1` on mismatch - the substitution penalty.
                __m128i const equal_i8_vec = _mm_cmpeq_epi8(query_char_vec, candidate_chars_vec);
                __m256i const equal_i16_vec = _mm256_cvtepi8_epi16(equal_i8_vec);
                __m256i const mismatch_addend_vec = _mm256_add_epi16(one_vec, equal_i16_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi16(diagonal_vec, mismatch_addend_vec);
                __m256i const cost_if_deletion_vec = _mm256_add_epi16(deletion_source_vec, one_vec);
                __m256i const cost_if_insertion_vec = _mm256_add_epi16(insertion_source_vec, one_vec);
                __m256i const cell_score_vec = _mm256_min_epu16(
                    cost_if_substitution_vec, _mm256_min_epu16(cost_if_deletion_vec, cost_if_insertion_vec));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row + column * candidate_lanes_k),
                                    cell_score_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        // Latch each live lane's result from its own final column; ragged lengths mean different columns per lane.
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
            size_t const candidate_length = candidates.lengths[lane_index];
            result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
        }
        return status_t::success_k;
    }
};

/**
 *  @brief Inter-sequence Haswell walker: one query against up to 16 candidates packed one-per-lane, weighted
 *         Needleman-Wunsch with 32-class substitution costs and a linear gap, 16-bit signed cells.
 *
 *  Structural twin of the 16-lane uniform `u16` walker, but the unit `cmpeq -> +1` substitution term is replaced
 *  by the resident-row class lookup of `class_lookup_haswell_t`, and the objective is @b maximization
 *  (Needleman-Wunsch), so the recurrence is `cell = max(diag + substitution, max(up, left) + gap)` over signed
 *  `i16` cells with a (typically negative) `error_cost_t` gap.
 *
 *  The query character is fixed for an entire Dynamic Programming row, so its class is scalar: per row the cost
 *  row `class_substitution_costs[query_class][0..31]` is made resident with `reload_row`. Inside the column loop
 *  the 16 candidate bytes index that resident row through the two-stage `lookup32`, yielding 16 signed `i8`
 *  substitution costs that sign-extend to 16 `i16` lanes.
 *
 *  @note The cells are signed `i16`, so the kernel is only valid while every reachable score stays within
 *      `[-32768, 32767]`; enforcing that bound is the caller's dispatch contract, the kernel performs no runtime
 *      range check.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 16, void> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 16;

    // The signed `i16` recurrence hardcodes `_mm256_max_epi16`; minimization would need a different blend.
    static_assert(objective_ == sz_maximize_score_k,
                  "The weighted candidate-lane kernel only implements score maximization (Needleman-Wunsch).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (16 `i16` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous row
        amount += score_row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 16 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        error_cost_t const gap = gap_costs_.open_or_extend;
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        class_lookup_haswell_t lookup;
        lookup.reload_classes(substituter_.byte_to_class);

        __m256i const gap_vec = _mm256_set1_epi16(static_cast<short>(gap));

        // Row 0: the empty query prefix against every candidate prefix is a run of `gap * column`, identical across
        // lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                _mm256_set1_epi16(static_cast<short>(static_cast<i16_t>(gap * (i16_t)column))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            // The query character is fixed for the whole row, so its class and cost row are scalar.
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            lookup.reload_row(&substituter_.class_substitution_costs[query_class][0]);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row),
                                _mm256_set1_epi16(static_cast<short>(static_cast<i16_t>(gap * (i16_t)query_position))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                // Only 16 candidate bytes are live this column; place them in the low half of a `u256_vec_t` so
                // `lookup32` produces this column's 16 signed `i8` substitution costs in its low `__m128i`.
                u256_vec_t candidate_chars_vec;
                candidate_chars_vec.ymm = _mm256_castsi128_si256(
                    _mm_loadu_si128(reinterpret_cast<__m128i const *>(candidates.position(column - 1))));
                __m256i const diagonal_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const left_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                u256_vec_t const cost_i8_vec = lookup.lookup32(candidate_chars_vec);
                __m256i const cost_i16_vec = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(cost_i8_vec.ymm));

                __m256i const cost_if_substitution_vec = _mm256_add_epi16(diagonal_vec, cost_i16_vec);
                __m256i const cost_if_gap_vec = _mm256_add_epi16(_mm256_max_epi16(up_vec, left_vec), gap_vec);
                __m256i const cell_score_vec = _mm256_max_epi16(cost_if_substitution_vec, cost_if_gap_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row + column * candidate_lanes_k),
                                    cell_score_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        // Latch each live lane's result from its own final column; ragged lengths mean different columns per lane.
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
            size_t const candidate_length = candidates.lengths[lane_index];
            result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
        }
        return status_t::success_k;
    }
};

/**
 *  @brief Inter-sequence Haswell walker: one query against up to 16 candidates packed one-per-lane, weighted
 *         @b Smith-Waterman (local) alignment with 32-class substitution costs and a linear gap, 16-bit signed cells.
 *
 *  Local sibling of the weighted Needleman-Wunsch walker: the recurrence clamps to zero,
 *  `cell = max(0, diag + substitution, max(up, left) + gap)`, the boundary row and column are zero (an alignment may
 *  start anywhere), and the score is the maximum cell over the whole matrix rather than the bottom-right corner. A
 *  per-lane running maximum accumulates that, updated only for columns within each lane's own candidate length so a
 *  shorter candidate's zero-padded tail columns cannot inflate its score.
 *
 *  @note Signed `i16` cells; valid while every reachable score stays within `[0, 32767]` (caller's dispatch contract).
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, linear_gap_costs_t, objective_,
                             sz_similarity_local_k, sz_cap_haswell_k, 16, void> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 16;

    static_assert(objective_ == sz_maximize_score_k,
                  "The weighted local candidate-lane kernel only implements score maximization (Smith-Waterman).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two `i16` score rows of `longest_candidate + 1` lane-vectors (16 cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes;
        amount += score_row_bytes;
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 16 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One local-alignment score per live lane (`candidates.lanes_count` of them).
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        error_cost_t const gap = gap_costs_.open_or_extend;
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        class_lookup_haswell_t lookup;
        lookup.reload_classes(substituter_.byte_to_class);

        __m256i const gap_vec = _mm256_set1_epi16(static_cast<short>(gap));
        __m256i const zero_vec = _mm256_setzero_si256();

        // Per-lane candidate lengths gate the running-maximum so a shorter candidate's padded columns are excluded.
        alignas(32) i16_t lane_lengths[candidate_lanes_k] = {0};
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
            lane_lengths[lane_index] = static_cast<i16_t>(candidates.lengths[lane_index]);
        __m256i const lane_lengths_vec = _mm256_load_si256(reinterpret_cast<__m256i const *>(lane_lengths));
        __m256i running_max_vec = zero_vec;

        // Local alignment: the boundary row and column are zero (an alignment may begin at any cell).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k), zero_vec);

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            lookup.reload_row(&substituter_.class_substitution_costs[query_class][0]);

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), zero_vec);
            for (size_t column = 1; column <= longest_candidate; ++column) {
                u256_vec_t candidate_chars_vec;
                candidate_chars_vec.ymm = _mm256_castsi128_si256(
                    _mm_loadu_si128(reinterpret_cast<__m128i const *>(candidates.position(column - 1))));
                __m256i const diagonal_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const left_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                u256_vec_t const cost_i8_vec = lookup.lookup32(candidate_chars_vec);
                __m256i const cost_i16_vec = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(cost_i8_vec.ymm));

                __m256i const cost_if_substitution_vec = _mm256_add_epi16(diagonal_vec, cost_i16_vec);
                __m256i const cost_if_gap_vec = _mm256_add_epi16(_mm256_max_epi16(up_vec, left_vec), gap_vec);
                __m256i const cell_score_vec =
                    _mm256_max_epi16(zero_vec, _mm256_max_epi16(cost_if_substitution_vec, cost_if_gap_vec));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row + column * candidate_lanes_k),
                                    cell_score_vec);

                // Fold this column into the running maximum only for lanes whose candidate reaches it. AVX2 has no
                // mask registers, so a `cmpgt` mask blends the unchanged running max with the candidate per lane.
                __m256i const column_live_vec =
                    _mm256_cmpgt_epi16(lane_lengths_vec, _mm256_set1_epi16(static_cast<short>(column - 1)));
                __m256i const folded_max_vec = _mm256_max_epi16(running_max_vec, cell_score_vec);
                running_max_vec = _mm256_blendv_epi8(running_max_vec, folded_max_vec, column_live_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        alignas(32) i16_t final_max[candidate_lanes_k];
        _mm256_store_si256(reinterpret_cast<__m256i *>(final_max), running_max_vec);
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
            result_lanes[lane_index] = final_max[lane_index];
        return status_t::success_k;
    }
};

#pragma endregion - Inter-Sequence Candidate Lanes

#pragma region - Inter-Sequence Cross-Product Engines

/**
 *  @brief Batched byte-level Levenshtein distances on Haswell, packing up to 32 candidates of a shared query into
 *      the unit-cost `u8` lane walker (or 16 candidates into the `u16` lane walker for longer strings) and falling
 *      back to the per-pair anti-diagonal Dynamic Programming scorer for the long tail (distances that escape `u16`,
 *      or non-unit costs).
 *
 *  Mirrors the structure of the Ice Lake `needleman_wunsch_scores` cross-product engine - live cells walked
 *  query-major, grouped into shared-query blocks, transposed into a reusable scratch buffer, scored, and scattered
 *  into the strided result matrix (plus the mirrored slot for symmetric self-similarity) - but Haswell has no
 *  lockstep Myers family, so the unit-cost lane walkers do the in-range work and a `size_t` value type carries the
 *  distance. Non-unit substitution or gap costs route entirely through the per-pair fallback.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 &&
                                              (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t lanes_u8_k = 32;  // ? `__m256i` holds 32 `u8` cells.
    static constexpr size_t lanes_u16_k = 16; // ? `__m256i` holds 16 `u16` cells.
    // The `u8` walker's `cost + 1` recurrence wraps once a cell reaches 255, so the all-gap path (the longer side)
    // must stay <= 254 to keep every intermediate within `u8`; 255 routes to the `u16` tier instead.
    static constexpr size_t length_limit_u8_k = 254;
    static constexpr size_t length_limit_u16_k = 65534; // ? Same `cost + 1` wrap headroom as the `u8` tier.

    // Haswell has no lockstep Myers family, so the long-tail per-pair fallback serializes to the scalar scorer
    // (its `scratch_space_needed` would otherwise instantiate a non-existent Haswell `levenshtein_distance_myers`).
    using scoring_t = levenshtein_distance<char, gap_costs_t, sz_cap_serial_k>; // ? Per-pair DP fallback.
    using lane_walker_u8_t =
        candidate_lane_walker<char, u8_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, (int)lanes_u8_k, void>;
    using lane_walker_u16_t =
        candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, (int)lanes_u16_k, void>;

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    uniform_substitution_costs_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    levenshtein_distances(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances(uniform_substitution_costs_t subs, linear_gap_costs_t gaps,
                          allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    bool is_unit_cost_() const noexcept {
        return substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1;
    }

    /** @brief Whether a `(query, candidate)` cell fits the unit-cost lane walkers' `u8` / `u16` distance range. */
    bool fits_u8_range_(size_t query_length, size_t candidate_length) const noexcept {
        return sz_max_of_two(query_length, candidate_length) <= length_limit_u8_k;
    }
    bool fits_u16_range_(size_t query_length, size_t candidate_length) const noexcept {
        return sz_max_of_two(query_length, candidate_length) <= length_limit_u16_k;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the transposed-block + lane-walker
     *      arena for the longest candidate (sized for the wider `u16` lanes, which is the larger of the two), or the
     *      anti-diagonal DP fallback for the longest query × longest candidate.
     */
    template <typename queries_type_, typename candidates_type_>
    size_t worst_cell_scratch_(queries_type_ const &queries, candidates_type_ const &candidates,
                               cpu_specs_t const &specs) const noexcept {
        size_t longest_query = 0, longest_query_index = 0, longest_candidate = 0, longest_candidate_index = 0;
        for (size_t index = 0; index < queries.size(); ++index)
            if (to_view(queries[index]).size() > longest_query)
                longest_query = to_view(queries[index]).size(), longest_query_index = index;
        for (size_t index = 0; index < candidates.size(); ++index)
            if (to_view(candidates[index]).size() > longest_candidate)
                longest_candidate = to_view(candidates[index]).size(), longest_candidate_index = index;
        lane_walker_u16_t lane_walker {substituter_, gap_costs_};
        // The transpose arena must hold the wider 32-lane `u8` block (`lanes_u8_k` candidates side by side); the
        // 16-lane `u16` block is narrower, so sizing for `lanes_u8_k` is a safe upper bound for both tiers.
        size_t const transpose_bytes = lanes_u8_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size() &&
            (!is_unit_cost_() || !fits_u16_range_(longest_query, longest_candidate))) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(transpose_bytes + walker_scratch, dp_scratch);
    }

#pragma region - Cross-Product Cell Addressing

    /** @brief A destination for one scored cell: the primary matrix slot plus an optional mirror slot. */
    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    /** @brief The number of live cells: the full rectangle, or the lower triangle (incl. diagonal) when symmetric. */
    static size_t live_cells_count_(size_t queries_count, size_t candidates_count,
                                    cross_similarities_t cross_kind) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) return queries_count * (queries_count + 1) / 2;
        return queries_count * candidates_count;
    }

    /** @brief Decodes a flat live-cell index into its `(query_index, candidate_index)` grid coordinates. */
    static void cell_to_indices_(size_t cell_index, size_t candidates_count, cross_similarities_t cross_kind,
                                 size_t &query_index, size_t &candidate_index) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) {
            size_t row = 0;
            while ((row + 1) * (row + 2) / 2 <= cell_index) ++row;
            query_index = row;
            candidate_index = cell_index - row * (row + 1) / 2;
        }
        else {
            query_index = cell_index / candidates_count;
            candidate_index = cell_index % candidates_count;
        }
    }

#pragma endregion - Cross-Product Cell Addressing

#pragma region - Cross-Product Scoring

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the shared-query unit-cost
     *      lane walkers, falling back to the anti-diagonal Dynamic Programming scorer for the long tail. Each cell
     *      scores `dist(query, candidate)` and writes it into the strided @p results matrix (plus the mirror slot
     *      for symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    [[gnu::noinline]] status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                            results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                            size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        scoring_t dp {substituter_, gap_costs_};
        lane_walker_u8_t lane_walker_u8 {substituter_, gap_costs_};
        lane_walker_u16_t lane_walker_u16 {substituter_, gap_costs_};
        bool const unit_cost = is_unit_cost_();

        // Carve the transpose and walker sub-arenas from the longest lane-eligible candidate in this slice; the
        // global-worst sizing in `worst_cell_scratch_` (computed for the wider `u16` lanes) bounds these offsets.
        size_t longest_candidate = 0;
        if (unit_cost)
            for (size_t cell_index = cell_begin; cell_index != cell_end; ++cell_index) {
                size_t query_index = 0, candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
                size_t const query_length = to_view(queries[query_index]).size();
                size_t const candidate_length = to_view(candidates[candidate_index]).size();
                if (query_length != 0 && candidate_length != 0 && fits_u16_range_(query_length, candidate_length))
                    longest_candidate = sz_max_of_two(longest_candidate, candidate_length);
            }

        // The transpose arena must hold the wider 32-lane `u8` block; sizing for `lanes_u8_k` bounds both tiers.
        size_t const transpose_bytes = lanes_u8_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch =
            longest_candidate ? lane_walker_u16.scratch_space_needed(longest_candidate, specs) : 0;
        char_t *transposed = reinterpret_cast<char_t *>(scratch.data());
        scratch_space_t walker_scratch_space = scratch.subspan(transpose_bytes, walker_scratch);
        scratch_space_t dp_scratch_space = scratch;

        auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
            cross_cell_destination_<value_t> destination;
            destination.primary = results.data + query_index * results.row_stride + candidate_index;
            if (cross_kind == cross_similarities_t::symmetric_k && candidate_index != query_index)
                destination.mirror = results.data + candidate_index * results.row_stride + query_index;
            return destination;
        };

        auto const scatter = [&](cross_cell_destination_<value_t> const &destination, size_t distance) noexcept {
            *destination.primary = static_cast<value_t>(distance);
            if (destination.mirror) *destination.mirror = static_cast<value_t>(distance);
        };

        dummy_executor_t dummy;
        size_t lengths[lanes_u8_k];
        cross_cell_destination_<value_t> destinations[lanes_u8_k];
        u8_t result_lanes_u8[lanes_u8_k];
        u16_t result_lanes_u16[lanes_u16_k];
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const query_length = query.size(), candidate_length = candidate.size();

            // Empty cell: the only alignment is all-gaps, so the distance is `max(query, candidate)` length.
            if (query_length == 0 || candidate_length == 0) {
                scatter(destination_for(query_index, candidate_index), sz_max_of_two(query_length, candidate_length));
                ++cell_index;
                continue;
            }

            // Long tail: non-unit costs, or a distance that escapes `u16`, walks the per-pair scorer.
            if (!unit_cost || !fits_u16_range_(query_length, candidate_length)) {
                size_t result_distance = 0;
                if (status_t status = dp(query, candidate, result_distance, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_distance);
                ++cell_index;
                continue;
            }

            // Pick the lane tier from this seed cell: the narrow `u8` lanes when both sides fit 255, else `u16`. A
            // block stays on one tier; same-query candidates that also fit it extend the block up to its capacity.
            bool const seed_fits_u8 = fits_u8_range_(query_length, candidate_length);
            size_t const lanes_count_max = seed_fits_u8 ? lanes_u8_k : lanes_u16_k;
            size_t const seed_query_index = query_index;
            size_t const block_begin = cell_index;
            size_t block_longest = candidate_length;
            lengths[0] = candidate_length;
            destinations[0] = destination_for(query_index, candidate_index);
            index_t lanes_count = 1;
            ++cell_index;
            for (; cell_index != cell_end && lanes_count != lanes_count_max; ++cell_index, ++lanes_count) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                if (next_query_index != seed_query_index) break;
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                size_t const next_length = next_candidate.size();
                if (next_length == 0) break;
                bool const next_fits_u8 = fits_u8_range_(query_length, next_length);
                bool const next_fits_u16 = fits_u16_range_(query_length, next_length);
                if (seed_fits_u8 ? !next_fits_u8 : !next_fits_u16) break;
                lengths[lanes_count] = next_length;
                destinations[lanes_count] = destination_for(next_query_index, next_candidate_index);
                block_longest = sz_max_of_two(block_longest, next_length);
            }

            size_t const lane_stride = seed_fits_u8 ? lanes_u8_k : lanes_u16_k;
            // Transpose the block into column-major lane order, zero-padding the ragged tails of every lane.
            for (size_t position = 0; position != lane_stride * block_longest; ++position) transposed[position] = 0;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index) {
                size_t fill_query_index = 0, fill_candidate_index = 0;
                cell_to_indices_(block_begin + lane_index, candidates_count, cross_kind, fill_query_index,
                                 fill_candidate_index);
                auto const lane_candidate = to_view(candidates[fill_candidate_index]);
                for (size_t position = 0; position < lane_candidate.size(); ++position)
                    transposed[position * lane_stride + lane_index] = lane_candidate[position];
            }

            candidate_lanes_block_t<char_t> block;
            block.transposed = transposed;
            block.lane_capacity = lane_stride;
            block.lanes_count = lanes_count;
            block.lengths = lengths;
            block.longest_candidate = block_longest;

            if (seed_fits_u8) {
                if (status_t status = lane_walker_u8(query, block, result_lanes_u8, walker_scratch_space, specs);
                    status != status_t::success_k)
                    return status;
                for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index)
                    scatter(destinations[lane_index], (size_t)result_lanes_u8[lane_index]);
            }
            else {
                if (status_t status = lane_walker_u16(query, block, result_lanes_u16, walker_scratch_space, specs);
                    status != status_t::success_k)
                    return status;
                for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index)
                    scatter(destinations[lane_index], (size_t)result_lanes_u16[lane_index]);
            }
        }
        return status_t::success_k;
    }

    /** @brief Scores the cross-product in parallel: each worker takes a contiguous slice of the live-cell range. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    [[gnu::noinline]] status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                               results_type_ &&results, cross_similarities_t cross_kind,
                                               executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        size_t const worker_scratch = worst_cell_scratch_(queries, candidates, specs);
        size_t const workers = sz_max_of_two(sz_min_of_two(executor.threads_count(), cells_count), (size_t)1);
        if (status_t status = score_scratch_.try_resize(worker_scratch * workers); status != status_t::success_k)
            return status;
        std::atomic<size_t> next_worker {0};
        std::atomic<status_t> error {status_t::success_k};
        executor.for_slices(cells_count, [&](size_t cell_begin, size_t length) noexcept {
            if (length == 0) return; // empty slice: no work, and it must not consume a scratch partition
            size_t const worker = next_worker.fetch_add(1, std::memory_order_relaxed);
            scratch_space_t slice = scratch_space_t(score_scratch_).subspan(worker * worker_scratch, worker_scratch);
            status_t status =
                score_range_(queries, candidates, results, cross_kind, cell_begin, cell_begin + length, slice, specs);
            if (status != status_t::success_k) error.store(status);
        });
        return error.load();
    }

#pragma endregion - Cross-Product Scoring

#pragma region - Public Cross-Product Overloads

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(queries, candidates, specs));
            status != status_t::success_k)
            return status;
        return score_range_(queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                            live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(sequences, sequences, specs));
            status != status_t::success_k)
            return status;
        return score_range_(sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                            live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion - Public Cross-Product Overloads
};

/**
 *  @brief Batched byte-level @b weighted Needleman-Wunsch scores on Haswell, packing up to 16 candidates of a
 *      shared query into the signed-`i16` `candidate_lane_walker` and falling back to the per-pair anti-diagonal
 *      Dynamic Programming scorer for the long tail (scores that escape `i16`).
 *
 *  Structural twin of the Ice Lake `needleman_wunsch_scores` cross-product engine, narrowed to the 16-lane AVX2
 *  weighted walker: live cells are walked query-major, grouped into shared-query blocks, transposed into a reusable
 *  scratch buffer, scored, and scattered into the strided result matrix (plus the mirrored slot for symmetric
 *  self-similarity). The lane walker maximizes the signed score; the value type is the wide `ssize_t`, so the
 *  `i16` lane results are widened on scatter.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
                               std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 &&
                                                (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, sz_caps_sh_k>; // ? Per-pair DP fallback.
    using lane_walker_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_haswell_k, (int)candidate_lanes_k, void>; // ? AVX2 shared-query lanes.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    needleman_wunsch_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_scores(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief The largest per-cell substitution/gap magnitude; scales the worst-case score against the `i16` range. */
    error_cost_magnitude_t cost_magnitude_() const noexcept {
        return sz_max_of_two(substituter_.magnitude(), gap_costs_.magnitude());
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the lane walker's `i16` headroom. */
    bool fits_lane_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the transposed-block + lane-walker
     *      arena for the longest candidate, or the anti-diagonal DP fallback for the longest query × longest candidate.
     */
    template <typename queries_type_, typename candidates_type_>
    size_t worst_cell_scratch_(queries_type_ const &queries, candidates_type_ const &candidates,
                               cpu_specs_t const &specs) const noexcept {
        size_t longest_query = 0, longest_query_index = 0, longest_candidate = 0, longest_candidate_index = 0;
        for (size_t index = 0; index < queries.size(); ++index)
            if (to_view(queries[index]).size() > longest_query)
                longest_query = to_view(queries[index]).size(), longest_query_index = index;
        for (size_t index = 0; index < candidates.size(); ++index)
            if (to_view(candidates[index]).size() > longest_candidate)
                longest_candidate = to_view(candidates[index]).size(), longest_candidate_index = index;
        lane_walker_t lane_walker {substituter_, gap_costs_};
        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size() && !fits_lane_range_(longest_query, longest_candidate)) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(transpose_bytes + walker_scratch, dp_scratch);
    }

#pragma region - Cross-Product Cell Addressing

    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    /** @brief The number of live cells: the full rectangle, or the lower triangle (incl. diagonal) when symmetric. */
    static size_t live_cells_count_(size_t queries_count, size_t candidates_count,
                                    cross_similarities_t cross_kind) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) return queries_count * (queries_count + 1) / 2;
        return queries_count * candidates_count;
    }

    /** @brief Decodes a flat live-cell index into its `(query_index, candidate_index)` grid coordinates. */
    static void cell_to_indices_(size_t cell_index, size_t candidates_count, cross_similarities_t cross_kind,
                                 size_t &query_index, size_t &candidate_index) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) {
            size_t row = 0;
            while ((row + 1) * (row + 2) / 2 <= cell_index) ++row;
            query_index = row;
            candidate_index = cell_index - row * (row + 1) / 2;
        }
        else {
            query_index = cell_index / candidates_count;
            candidate_index = cell_index % candidates_count;
        }
    }

#pragma endregion - Cross-Product Cell Addressing

#pragma region - Cross-Product Scoring

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the shared-query lane walker,
     *      falling back to the anti-diagonal Dynamic Programming scorer for the long tail. Each cell scores
     *      `score(query, candidate)` and writes it into the strided @p results matrix (plus the mirror slot for
     *      symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    [[gnu::noinline]] status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                            results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                            size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        scoring_t dp {substituter_, gap_costs_};
        lane_walker_t lane_walker {substituter_, gap_costs_};
        size_t longest_candidate = 0;
        for (size_t cell_index = cell_begin; cell_index != cell_end; ++cell_index) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            size_t const query_length = to_view(queries[query_index]).size();
            size_t const candidate_length = to_view(candidates[candidate_index]).size();
            if (query_length != 0 && candidate_length != 0 && fits_lane_range_(query_length, candidate_length))
                longest_candidate = sz_max_of_two(longest_candidate, candidate_length);
        }

        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        char_t *transposed = reinterpret_cast<char_t *>(scratch.data());
        scratch_space_t walker_scratch_space = scratch.subspan(transpose_bytes, walker_scratch);
        scratch_space_t dp_scratch_space = scratch;

        auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
            cross_cell_destination_<value_t> destination;
            destination.primary = results.data + query_index * results.row_stride + candidate_index;
            if (cross_kind == cross_similarities_t::symmetric_k && candidate_index != query_index)
                destination.mirror = results.data + candidate_index * results.row_stride + query_index;
            return destination;
        };

        auto const scatter = [&](cross_cell_destination_<value_t> const &destination, ssize_t score) noexcept {
            *destination.primary = static_cast<value_t>(score);
            if (destination.mirror) *destination.mirror = static_cast<value_t>(score);
        };

        dummy_executor_t dummy;
        size_t lengths[candidate_lanes_k];
        cross_cell_destination_<value_t> destinations[candidate_lanes_k];
        i16_t result_lanes[candidate_lanes_k];
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const query_length = query.size(), candidate_length = candidate.size();

            // Empty cell: the only alignment is all-gaps, so the score is `gap * other_length`.
            if (query_length == 0 || candidate_length == 0) {
                ssize_t const other_length = (ssize_t)sz_max_of_two(query_length, candidate_length);
                scatter(destination_for(query_index, candidate_index),
                        (ssize_t)gap_costs_.open_or_extend * other_length);
                ++cell_index;
                continue;
            }

            // Long tail: any cell whose worst-case score escapes the `i16` range walks the per-pair scorer.
            if (!fits_lane_range_(query_length, candidate_length)) {
                ssize_t result_score = 0;
                if (status_t status = dp(query, candidate, result_score, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_score);
                ++cell_index;
                continue;
            }

            // Seed a shared-query block with this cell, then extend over consecutive same-query candidates that
            // also fit the `i16` range, up to the 16-lane capacity. Seeding first guarantees forward progress.
            size_t const seed_query_index = query_index;
            size_t const block_begin = cell_index;
            size_t block_longest = candidate_length;
            lengths[0] = candidate_length;
            destinations[0] = destination_for(query_index, candidate_index);
            index_t lanes_count = 1;
            ++cell_index;
            for (; cell_index != cell_end && lanes_count != candidate_lanes_k; ++cell_index, ++lanes_count) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                if (next_query_index != seed_query_index) break;
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                if (next_candidate.size() == 0 || !fits_lane_range_(query_length, next_candidate.size())) break;
                lengths[lanes_count] = next_candidate.size();
                destinations[lanes_count] = destination_for(next_query_index, next_candidate_index);
                block_longest = sz_max_of_two(block_longest, next_candidate.size());
            }

            // Transpose the block into column-major lane order, zero-padding the ragged tails of every lane.
            for (size_t position = 0; position != candidate_lanes_k * block_longest; ++position) transposed[position] = 0;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index) {
                size_t fill_query_index = 0, fill_candidate_index = 0;
                cell_to_indices_(block_begin + lane_index, candidates_count, cross_kind, fill_query_index,
                                 fill_candidate_index);
                auto const lane_candidate = to_view(candidates[fill_candidate_index]);
                for (size_t position = 0; position < lane_candidate.size(); ++position)
                    transposed[position * candidate_lanes_k + lane_index] = lane_candidate[position];
            }

            candidate_lanes_block_t<char_t> block;
            block.transposed = transposed;
            block.lane_capacity = candidate_lanes_k;
            block.lanes_count = lanes_count;
            block.lengths = lengths;
            block.longest_candidate = block_longest;

            if (status_t status = lane_walker(query, block, result_lanes, walker_scratch_space, specs);
                status != status_t::success_k)
                return status;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index)
                scatter(destinations[lane_index], (ssize_t)result_lanes[lane_index]);
        }
        return status_t::success_k;
    }

    /** @brief Scores the cross-product in parallel: each worker takes a contiguous slice of the live-cell range. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    [[gnu::noinline]] status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                               results_type_ &&results, cross_similarities_t cross_kind,
                                               executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        size_t const worker_scratch = worst_cell_scratch_(queries, candidates, specs);
        size_t const workers = sz_max_of_two(sz_min_of_two(executor.threads_count(), cells_count), (size_t)1);
        if (status_t status = score_scratch_.try_resize(worker_scratch * workers); status != status_t::success_k)
            return status;
        std::atomic<size_t> next_worker {0};
        std::atomic<status_t> error {status_t::success_k};
        executor.for_slices(cells_count, [&](size_t cell_begin, size_t length) noexcept {
            if (length == 0) return; // empty slice: no work, and it must not consume a scratch partition
            size_t const worker = next_worker.fetch_add(1, std::memory_order_relaxed);
            scratch_space_t slice = scratch_space_t(score_scratch_).subspan(worker * worker_scratch, worker_scratch);
            status_t status =
                score_range_(queries, candidates, results, cross_kind, cell_begin, cell_begin + length, slice, specs);
            if (status != status_t::success_k) error.store(status);
        });
        return error.load();
    }

#pragma endregion - Cross-Product Scoring

#pragma region - Public Cross-Product Overloads

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(queries, candidates, specs));
            status != status_t::success_k)
            return status;
        return score_range_(queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                            live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(sequences, sequences, specs));
            status != status_t::success_k)
            return status;
        return score_range_(sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                            live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion - Public Cross-Product Overloads
};

/**
 *  @brief Batched byte-level @b weighted Smith-Waterman (local) scores on Haswell, packing up to 16 candidates of a
 *      shared query into the signed-`i16` local `candidate_lane_walker` and falling back to the per-pair anti-diagonal
 *      Dynamic Programming scorer for the long tail (scores that escape `i16`).
 *
 *  Local sibling of the Haswell `needleman_wunsch_scores` cross-product engine: identical query-major grouping,
 *  transpose, and scatter, but it dispatches the @b local lane walker (zero-clamped recurrence, score = the maximum
 *  cell of the matrix) and an empty `(query, candidate)` cell scores @b 0 (a local alignment may align nothing).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 &&
                                              (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, sz_caps_sh_k>; // ? Per-pair DP fallback.
    using lane_walker_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_haswell_k, (int)candidate_lanes_k, void>; // ? AVX2 shared-query local lanes.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    smith_waterman_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    smith_waterman_scores(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief The largest per-cell substitution/gap magnitude; scales the worst-case score against the `i16` range. */
    error_cost_magnitude_t cost_magnitude_() const noexcept {
        return sz_max_of_two(substituter_.magnitude(), gap_costs_.magnitude());
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the lane walker's `i16` headroom. */
    bool fits_lane_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the transposed-block + lane-walker
     *      arena for the longest candidate, or the anti-diagonal DP fallback for the longest query × longest candidate.
     */
    template <typename queries_type_, typename candidates_type_>
    size_t worst_cell_scratch_(queries_type_ const &queries, candidates_type_ const &candidates,
                               cpu_specs_t const &specs) const noexcept {
        size_t longest_query = 0, longest_query_index = 0, longest_candidate = 0, longest_candidate_index = 0;
        for (size_t index = 0; index < queries.size(); ++index)
            if (to_view(queries[index]).size() > longest_query)
                longest_query = to_view(queries[index]).size(), longest_query_index = index;
        for (size_t index = 0; index < candidates.size(); ++index)
            if (to_view(candidates[index]).size() > longest_candidate)
                longest_candidate = to_view(candidates[index]).size(), longest_candidate_index = index;
        lane_walker_t lane_walker {substituter_, gap_costs_};
        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size() && !fits_lane_range_(longest_query, longest_candidate)) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(transpose_bytes + walker_scratch, dp_scratch);
    }

#pragma region - Cross-Product Cell Addressing

    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    /** @brief The number of live cells: the full rectangle, or the lower triangle (incl. diagonal) when symmetric. */
    static size_t live_cells_count_(size_t queries_count, size_t candidates_count,
                                    cross_similarities_t cross_kind) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) return queries_count * (queries_count + 1) / 2;
        return queries_count * candidates_count;
    }

    /** @brief Decodes a flat live-cell index into its `(query_index, candidate_index)` grid coordinates. */
    static void cell_to_indices_(size_t cell_index, size_t candidates_count, cross_similarities_t cross_kind,
                                 size_t &query_index, size_t &candidate_index) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) {
            size_t row = 0;
            while ((row + 1) * (row + 2) / 2 <= cell_index) ++row;
            query_index = row;
            candidate_index = cell_index - row * (row + 1) / 2;
        }
        else {
            query_index = cell_index / candidates_count;
            candidate_index = cell_index % candidates_count;
        }
    }

#pragma endregion - Cross-Product Cell Addressing

#pragma region - Cross-Product Scoring

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` with the shared-query local lane walker, falling back
     *      to the anti-diagonal Dynamic Programming scorer for the long tail. Writes each `score(query, candidate)`
     *      into the strided @p results matrix (plus the mirror slot for symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    [[gnu::noinline]] status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                            results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                            size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        scoring_t dp {substituter_, gap_costs_};
        lane_walker_t lane_walker {substituter_, gap_costs_};
        size_t longest_candidate = 0;
        for (size_t cell_index = cell_begin; cell_index != cell_end; ++cell_index) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            size_t const query_length = to_view(queries[query_index]).size();
            size_t const candidate_length = to_view(candidates[candidate_index]).size();
            if (query_length != 0 && candidate_length != 0 && fits_lane_range_(query_length, candidate_length))
                longest_candidate = sz_max_of_two(longest_candidate, candidate_length);
        }

        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        char_t *transposed = reinterpret_cast<char_t *>(scratch.data());
        scratch_space_t walker_scratch_space = scratch.subspan(transpose_bytes, walker_scratch);
        scratch_space_t dp_scratch_space = scratch;

        auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
            cross_cell_destination_<value_t> destination;
            destination.primary = results.data + query_index * results.row_stride + candidate_index;
            if (cross_kind == cross_similarities_t::symmetric_k && candidate_index != query_index)
                destination.mirror = results.data + candidate_index * results.row_stride + query_index;
            return destination;
        };

        auto const scatter = [&](cross_cell_destination_<value_t> const &destination, ssize_t score) noexcept {
            *destination.primary = static_cast<value_t>(score);
            if (destination.mirror) *destination.mirror = static_cast<value_t>(score);
        };

        dummy_executor_t dummy;
        size_t lengths[candidate_lanes_k];
        cross_cell_destination_<value_t> destinations[candidate_lanes_k];
        i16_t result_lanes[candidate_lanes_k];
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const query_length = query.size(), candidate_length = candidate.size();

            // Empty cell: a local alignment may align nothing, so the score is 0.
            if (query_length == 0 || candidate_length == 0) {
                scatter(destination_for(query_index, candidate_index), 0);
                ++cell_index;
                continue;
            }

            // Long tail: any cell whose worst-case score escapes the `i16` range walks the per-pair scorer.
            if (!fits_lane_range_(query_length, candidate_length)) {
                ssize_t result_score = 0;
                if (status_t status = dp(query, candidate, result_score, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_score);
                ++cell_index;
                continue;
            }

            size_t const seed_query_index = query_index;
            size_t const block_begin = cell_index;
            size_t block_longest = candidate_length;
            lengths[0] = candidate_length;
            destinations[0] = destination_for(query_index, candidate_index);
            index_t lanes_count = 1;
            ++cell_index;
            for (; cell_index != cell_end && lanes_count != candidate_lanes_k; ++cell_index, ++lanes_count) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                if (next_query_index != seed_query_index) break;
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                if (next_candidate.size() == 0 || !fits_lane_range_(query_length, next_candidate.size())) break;
                lengths[lanes_count] = next_candidate.size();
                destinations[lanes_count] = destination_for(next_query_index, next_candidate_index);
                block_longest = sz_max_of_two(block_longest, next_candidate.size());
            }

            for (size_t position = 0; position != candidate_lanes_k * block_longest; ++position) transposed[position] = 0;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index) {
                size_t fill_query_index = 0, fill_candidate_index = 0;
                cell_to_indices_(block_begin + lane_index, candidates_count, cross_kind, fill_query_index,
                                 fill_candidate_index);
                auto const lane_candidate = to_view(candidates[fill_candidate_index]);
                for (size_t position = 0; position < lane_candidate.size(); ++position)
                    transposed[position * candidate_lanes_k + lane_index] = lane_candidate[position];
            }

            candidate_lanes_block_t<char_t> block;
            block.transposed = transposed;
            block.lane_capacity = candidate_lanes_k;
            block.lanes_count = lanes_count;
            block.lengths = lengths;
            block.longest_candidate = block_longest;

            if (status_t status = lane_walker(query, block, result_lanes, walker_scratch_space, specs);
                status != status_t::success_k)
                return status;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index)
                scatter(destinations[lane_index], (ssize_t)result_lanes[lane_index]);
        }
        return status_t::success_k;
    }

    /** @brief Scores the cross-product in parallel: each worker takes a contiguous slice of the live-cell range. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    [[gnu::noinline]] status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                               results_type_ &&results, cross_similarities_t cross_kind,
                                               executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        size_t const worker_scratch = worst_cell_scratch_(queries, candidates, specs);
        size_t const workers = sz_max_of_two(sz_min_of_two(executor.threads_count(), cells_count), (size_t)1);
        if (status_t status = score_scratch_.try_resize(worker_scratch * workers); status != status_t::success_k)
            return status;
        std::atomic<size_t> next_worker {0};
        std::atomic<status_t> error {status_t::success_k};
        executor.for_slices(cells_count, [&](size_t cell_begin, size_t length) noexcept {
            if (length == 0) return; // empty slice: no work, and it must not consume a scratch partition
            size_t const worker = next_worker.fetch_add(1, std::memory_order_relaxed);
            scratch_space_t slice = scratch_space_t(score_scratch_).subspan(worker * worker_scratch, worker_scratch);
            status_t status =
                score_range_(queries, candidates, results, cross_kind, cell_begin, cell_begin + length, slice, specs);
            if (status != status_t::success_k) error.store(status);
        });
        return error.load();
    }

#pragma endregion - Cross-Product Scoring

#pragma region - Public Cross-Product Overloads

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(queries, candidates, specs));
            status != status_t::success_k)
            return status;
        return score_range_(queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                            live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(sequences, sequences, specs));
            status != status_t::success_k)
            return status;
        return score_range_(sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                            live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion - Public Cross-Product Overloads
};

#pragma endregion - Inter-Sequence Cross-Product Engines

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
