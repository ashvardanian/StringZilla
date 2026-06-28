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

    /**
     *  @brief First stage only: map each input byte to its class. Loop-invariant across query rows, so diagonal
     *      walkers pre-classify candidate columns @b once and feed the result into `costs_for_classes32`.
     */
    inline u256_vec_t classify32(u256_vec_t const &text_vec) const noexcept {

        u256_vec_t low_nibbles_vec, high_nibbles_vec, class_vec;

        // Map each input byte to its class. `VPSHUFB` ignores the upper nibble of the index, so we shuffle every
        // high-nibble group by the low nibble and keep the group whose high nibble matches.
        low_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, low_nibble_mask_vec_.ymm);
        // There is no `_mm256_srli_epi8` intrinsic, so we shift as 16-bit lanes and mask off the carried bits.
        high_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), low_nibble_mask_vec_.ymm);

        class_vec.ymm = _mm256_setzero_si256();
        for (int group = 0; group != 16; ++group) {
            __m256i shuffled = _mm256_shuffle_epi8(byte_to_class_group_vecs_[group].ymm, low_nibbles_vec.ymm);
            __m256i is_group = _mm256_cmpeq_epi8(high_nibbles_vec.ymm, _mm256_set1_epi8((char)group));
            class_vec.ymm = _mm256_blendv_epi8(class_vec.ymm, shuffled, is_group);
        }
        return class_vec;
    }

    /**
     *  @brief Second stage only: map each precomputed class to its substitution cost using the resident cost row
     *      set by `reload_row`. Cheap enough to re-run per cell while the class stage is hoisted out of the row loop.
     */
    inline u256_vec_t costs_for_classes32(u256_vec_t const &class_vec) const noexcept {

        u256_vec_t substituted_vec;

        // Map each class to its substitution cost. `VPSHUFB` uses the low 4 bits of the class, so the low-half
        // shuffle is valid for classes below 16 and the high-half for the rest.
        __m256i cost_if_low = _mm256_shuffle_epi8(row_subs_low_vec_.ymm, class_vec.ymm);
        __m256i cost_if_high = _mm256_shuffle_epi8(row_subs_high_vec_.ymm, class_vec.ymm);
        __m256i is_high_class = _mm256_cmpgt_epi8(class_vec.ymm, _mm256_set1_epi8(15));
        substituted_vec.ymm = _mm256_blendv_epi8(cost_if_low, cost_if_high, is_high_class);
        return substituted_vec;
    }

    inline u256_vec_t lookup32(u256_vec_t const &text_vec) const noexcept {
        return costs_for_classes32(classify32(text_vec));
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

#pragma region Inter Sequence Candidate Lanes

/**
 *  @brief Inter-sequence Haswell walker: one query against up to 32 candidates packed one-per-lane.
 *
 *  Computes the @b global unit-cost Levenshtein distance of a single shared query against a transposed
 *  `candidate_lanes_block` of up to 32 candidates. Each `__m256i` holds 32 `u8` cells - lane @p lane_index
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
     *  @param[in] candidates Transposed block of up to 32 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
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
                __m256i const candidate_chars_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const deletion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const insertion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

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
 *  that sign-extend to 16 `i16` lanes; `_mm256_blendv_epi8` then selects the configured `match` cost on equal lanes
 *  and `mismatch` on the rest as the diagonal substitution penalty, while deletion and insertion each add the
 *  uniform `gap` cost. The recurrence `cell = min(substitution, min(deletion, insertion))` advances all 16 lanes in
 *  lockstep over `_mm256_min_epu16`. This serves the @b non-unit uniform-cost Levenshtein cells; the unit-cost case
 *  (match 0, mismatch 1, gap 1) is handled by the Myers fast path and never reaches here.
 *
 *  @note The cells are `u16`, so the kernel is only valid while every reachable score stays below 65535; the longest
 *      reachable cell is the all-gap path `max(query, candidate) * gap`. Enforcing that bound (a cost-dependent
 *      length limit) is the caller's dispatch contract; the kernel performs no runtime range check.
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
     *  @param[in] candidates Transposed block of up to 16 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 16`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        // The recurrence honors the engine's configured uniform costs: `match`/`mismatch` on the diagonal and a
        // single `gap` on either deletion or insertion. The unit-cost case (match 0, mismatch 1, gap 1) is served by
        // the Myers fast path, so this kernel only ever runs for non-unit costs.
        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const gap_cost = static_cast<score_t>(gap_costs_.open_or_extend);
        __m256i const match_vec = _mm256_set1_epi16(static_cast<short>(match_cost));
        __m256i const mismatch_vec = _mm256_set1_epi16(static_cast<short>(mismatch_cost));
        __m256i const gap_vec = _mm256_set1_epi16(static_cast<short>(gap_cost));

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                _mm256_set1_epi16(static_cast<short>(static_cast<u16_t>(column * gap_cost))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m128i const query_char_vec = _mm_set1_epi8(query[query_position - 1]);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row),
                                _mm256_set1_epi16(static_cast<short>(static_cast<u16_t>(query_position * gap_cost))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                // Only 16 candidate bytes are live this column; load them into the low half of a `__m128i`.
                __m128i const candidate_chars_vec = _mm_loadu_si128(
                    reinterpret_cast<__m128i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const deletion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const insertion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                // `cmpeq` yields `0xFF` (-1) on a match, `0x00` on a mismatch, per byte; sign-extend to 16 `i16` so the
                // mask selects `match` on equal lanes and `mismatch` on the rest as the diagonal substitution penalty.
                __m128i const equal_i8_vec = _mm_cmpeq_epi8(query_char_vec, candidate_chars_vec);
                __m256i const equal_i16_vec = _mm256_cvtepi8_epi16(equal_i8_vec);
                __m256i const mismatch_addend_vec = _mm256_blendv_epi8(mismatch_vec, match_vec, equal_i16_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi16(diagonal_vec, mismatch_addend_vec);
                __m256i const cost_if_deletion_vec = _mm256_add_epi16(deletion_source_vec, gap_vec);
                __m256i const cost_if_insertion_vec = _mm256_add_epi16(insertion_source_vec, gap_vec);
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
 *  @brief Inter-sequence Haswell walker: one query against up to 8 candidates packed one-per-lane, 32-bit cells.
 *
 *  Width-mirror of the 16-lane `u16` linear uniform Levenshtein walker, widened to `u32` cells so each `__m256i`
 *  holds 8 lanes. The candidate characters remain `char`/`u8`: only 8 of them are live per column, loaded into the
 *  low 8 bytes of a `__m128i` (`_mm_loadl_epi64`) and compared against the broadcast query character with
 *  `_mm_cmpeq_epi8`, yielding 8 bytes of `0xFF` (match) or `0x00` (mismatch) that sign-extend to 8 `i32` lanes via
 *  `_mm256_cvtepi8_epi32`; `_mm256_blendv_epi8` then selects the configured `match` cost on equal lanes and
 *  `mismatch` on the rest as the diagonal substitution penalty, while deletion and insertion each add the uniform
 *  `gap` cost. The recurrence `cell = min(substitution, min(deletion, insertion))` advances all 8 lanes in lockstep
 *  over `_mm256_min_epu32`. This serves the @b non-unit uniform-cost Levenshtein cells whose all-gap path would
 *  overflow `u16`; the unit-cost case is handled by the Myers fast path and never reaches here.
 *
 *  @note The cells are `u32`, so the kernel is only valid while every reachable score stays below 4294967295; the
 *      longest reachable cell is the all-gap path `max(query, candidate) * gap`. Enforcing that bound (a cost-dependent
 *      length limit) is the caller's dispatch contract; the kernel performs no runtime range check.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u32_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 8, void> {

    using char_t = char;
    using score_t = u32_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The `u32` lane recurrence hardcodes `_mm256_min_epu32`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 32-bit candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (8 `u32` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 8 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 8`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        // The recurrence honors the engine's configured uniform costs: `match`/`mismatch` on the diagonal and a
        // single `gap` on either deletion or insertion. The unit-cost case (match 0, mismatch 1, gap 1) is served by
        // the Myers fast path, so this kernel only ever runs for non-unit costs.
        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const gap_cost = static_cast<score_t>(gap_costs_.open_or_extend);
        __m256i const match_vec = _mm256_set1_epi32(static_cast<int>(match_cost));
        __m256i const mismatch_vec = _mm256_set1_epi32(static_cast<int>(mismatch_cost));
        __m256i const gap_vec = _mm256_set1_epi32(static_cast<int>(gap_cost));

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                _mm256_set1_epi32(static_cast<int>(static_cast<u32_t>(column * gap_cost))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m128i const query_char_vec = _mm_set1_epi8(query[query_position - 1]);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row),
                                _mm256_set1_epi32(static_cast<int>(static_cast<u32_t>(query_position * gap_cost))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                // Only 8 candidate bytes are live this column; load them into the low 8 bytes of a `__m128i`.
                __m128i const candidate_chars_vec = _mm_loadl_epi64(
                    reinterpret_cast<__m128i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const deletion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const insertion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                // `cmpeq` yields `0xFF` (-1) on a match, `0x00` on a mismatch, per byte; sign-extend to 8 `i32` so the
                // mask selects `match` on equal lanes and `mismatch` on the rest as the diagonal substitution penalty.
                __m128i const equal_i8_vec = _mm_cmpeq_epi8(query_char_vec, candidate_chars_vec);
                __m256i const equal_i32_vec = _mm256_cvtepi8_epi32(equal_i8_vec);
                __m256i const mismatch_addend_vec = _mm256_blendv_epi8(mismatch_vec, match_vec, equal_i32_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi32(diagonal_vec, mismatch_addend_vec);
                __m256i const cost_if_deletion_vec = _mm256_add_epi32(deletion_source_vec, gap_vec);
                __m256i const cost_if_insertion_vec = _mm256_add_epi32(insertion_source_vec, gap_vec);
                __m256i const cell_score_vec = _mm256_min_epu32(
                    cost_if_substitution_vec, _mm256_min_epu32(cost_if_deletion_vec, cost_if_insertion_vec));
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
 *  @brief Inter-sequence Haswell walker: one query against up to 16 candidates packed one-per-lane, @b affine-gap
 *         (Gotoh) unit-class Levenshtein over unsigned 16-bit cells, distance minimization, global alignment.
 *
 *  Affine sibling of the 16-lane uniform `u16` linear walker: the diagonal substitution term is identical (a
 *  `_mm_cmpeq_epi8` mask selecting the `match` cost on equal lanes and `mismatch` on the rest), but the single
 *  gap term is replaced by the branchless Gotoh E/F tracks of the signed weighted affine walker, here over
 *  `_mm256_min_epu16` so the recurrence stays in non-negative distance space:
 *      F[column] = min(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *      E         = min(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *      M[column] = min(M_diagonal + substitution, min(E, F))
 *  The `F` track is materialized as a third scratch row indexed exactly like `M`; the `E` track only depends on the
 *  cell to its left, so it lives in a single rolling lane-register reseeded per row from the discarded boundary. A
 *  large `discard_bias` is added to any track that cannot have been opened yet, keeping `min` from selecting it.
 *
 *  @note The cells are unsigned `u16`, so the kernel is only valid while every reachable score - plus the
 *      `discard_bias` headroom - stays below 65535; enforcing that bound is the caller's dispatch contract.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, affine_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 16, void> {

    using char_t = char;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 16;

    // The `u16` lane recurrence hardcodes `_mm256_min_epu16`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 16-bit affine candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, affine_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds three `u16` rows of `longest_candidate + 1` lane-vectors: previous/current M plus the F. */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous M row
        amount += score_row_bytes; // current M row
        amount += score_row_bytes; // F (vertical gap) row, carried across rows
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 16 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Three row buffers carved from the byte span; each lane-vector lives at `row + column * 16`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        score_t *vertical_row = current_row + row_stride;

        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const open = static_cast<score_t>(gap_costs_.open);
        score_t const extend = static_cast<score_t>(gap_costs_.extend);
        __m256i const match_vec = _mm256_set1_epi16(static_cast<short>(match_cost));
        __m256i const mismatch_vec = _mm256_set1_epi16(static_cast<short>(mismatch_cost));
        __m256i const open_vec = _mm256_set1_epi16(static_cast<short>(open));
        __m256i const extend_vec = _mm256_set1_epi16(static_cast<short>(extend));
        // A magnitude above any in-range score so `min` never selects a track that has not been opened yet; the
        // caller's reach bound keeps real scores below it, and `discard_bias + extend` must not wrap `u16`.
        __m256i const discard_bias_vec = _mm256_set1_epi16(static_cast<short>(static_cast<u16_t>(60000)));

        // Row 0 (global): `M[0] = 0`; `M[column] = open + extend * (column - 1)`. The vertical `F` row cannot have
        // been entered from above at row 0, so it is seeded with the discarded magnitude added to the boundary.
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row), _mm256_setzero_si256());
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row), discard_bias_vec);
        for (size_t column = 1; column <= longest_candidate; ++column) {
            __m256i const boundary_vec = _mm256_set1_epi16(
                static_cast<short>(static_cast<u16_t>(open + extend * (u16_t)(column - 1))));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k), boundary_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                _mm256_add_epi16(discard_bias_vec, boundary_vec));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m128i const query_char_vec = _mm_set1_epi8(query[query_position - 1]);

            // First-column boundary of this row: `M = open + extend * (query_position - 1)`, and the rolling
            // horizontal `E` register reseeded with the discarded magnitude added to that left boundary.
            __m256i const left_boundary_vec = _mm256_set1_epi16(
                static_cast<short>(static_cast<u16_t>(open + extend * (u16_t)(query_position - 1))));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), left_boundary_vec);
            __m256i horizontal_vec = _mm256_add_epi16(discard_bias_vec, left_boundary_vec);

            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m128i const candidate_chars_vec = _mm_loadu_si128(
                    reinterpret_cast<__m128i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const left_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vertical_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(vertical_row + column * candidate_lanes_k));

                // `cmpeq` yields `0xFF` (-1) on a match, `0x00` on a mismatch, per byte; sign-extend to 16 `i16` so the
                // mask selects `match` on equal lanes and `mismatch` on the rest as the diagonal substitution penalty.
                __m128i const equal_i8_vec = _mm_cmpeq_epi8(query_char_vec, candidate_chars_vec);
                __m256i const equal_i16_vec = _mm256_cvtepi8_epi16(equal_i8_vec);
                __m256i const substitution_addend_vec = _mm256_blendv_epi8(mismatch_vec, match_vec, equal_i16_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi16(diagonal_vec, substitution_addend_vec);

                // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                __m256i const vertical_vec = _mm256_min_epu16(_mm256_add_epi16(up_vec, open_vec),
                                                              _mm256_add_epi16(up_vertical_vec, extend_vec));
                horizontal_vec = _mm256_min_epu16(_mm256_add_epi16(left_vec, open_vec),
                                                  _mm256_add_epi16(horizontal_vec, extend_vec));
                __m256i const cost_if_gap_vec = _mm256_min_epu16(vertical_vec, horizontal_vec);

                __m256i const cell_score_vec = _mm256_min_epu16(cost_if_substitution_vec, cost_if_gap_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                    vertical_vec);
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
 *  @brief Inter-sequence Haswell walker: one query against up to 8 candidates packed one-per-lane, @b affine-gap
 *         (Gotoh) unit-class Levenshtein over unsigned 32-bit cells, distance minimization, global alignment.
 *
 *  Width-mirror of the 16-lane uniform `u16` affine walker, widened to `u32` cells so each `__m256i` holds 8 lanes.
 *  The diagonal substitution term is identical (a `_mm_cmpeq_epi8` mask, here sign-extended to 8 `i32` lanes via
 *  `_mm256_cvtepi8_epi32`, selecting the `match` cost on equal lanes and `mismatch` on the rest), and the single gap
 *  term is the branchless Gotoh E/F tracks over `_mm256_min_epu32` so the recurrence stays in non-negative distance
 *  space:
 *      F[column] = min(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *      E         = min(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *      M[column] = min(M_diagonal + substitution, min(E, F))
 *  The `F` track is materialized as a third scratch row indexed exactly like `M`; the `E` track only depends on the
 *  cell to its left, so it lives in a single rolling lane-register reseeded per row from the discarded boundary. A
 *  large `discard_bias` is added to any track that cannot have been opened yet, keeping `min` from selecting it.
 *
 *  @note The cells are unsigned `u32`, so the kernel is only valid while every reachable score - plus the
 *      `discard_bias` headroom - stays below 4294967295; enforcing that bound is the caller's dispatch contract.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u32_t, uniform_substitution_costs_t, affine_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 8, void> {

    using char_t = char;
    using score_t = u32_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The `u32` lane recurrence hardcodes `_mm256_min_epu32`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 32-bit affine candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, affine_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds three `u32` rows of `longest_candidate + 1` lane-vectors: previous/current M plus the F. */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous M row
        amount += score_row_bytes; // current M row
        amount += score_row_bytes; // F (vertical gap) row, carried across rows
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 8 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Three row buffers carved from the byte span; each lane-vector lives at `row + column * 8`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        score_t *vertical_row = current_row + row_stride;

        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const open = static_cast<score_t>(gap_costs_.open);
        score_t const extend = static_cast<score_t>(gap_costs_.extend);
        __m256i const match_vec = _mm256_set1_epi32(static_cast<int>(match_cost));
        __m256i const mismatch_vec = _mm256_set1_epi32(static_cast<int>(mismatch_cost));
        __m256i const open_vec = _mm256_set1_epi32(static_cast<int>(open));
        __m256i const extend_vec = _mm256_set1_epi32(static_cast<int>(extend));
        // A magnitude above any in-range score so `min` never selects a track that has not been opened yet; the
        // caller's reach bound keeps real scores below it, and `discard_bias + extend` must not wrap `u32`.
        __m256i const discard_bias_vec = _mm256_set1_epi32(static_cast<int>(static_cast<u32_t>(2000000000)));

        // Row 0 (global): `M[0] = 0`; `M[column] = open + extend * (column - 1)`. The vertical `F` row cannot have
        // been entered from above at row 0, so it is seeded with the discarded magnitude added to the boundary.
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row), _mm256_setzero_si256());
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row), discard_bias_vec);
        for (size_t column = 1; column <= longest_candidate; ++column) {
            __m256i const boundary_vec = _mm256_set1_epi32(
                static_cast<int>(static_cast<u32_t>(open + extend * (u32_t)(column - 1))));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k), boundary_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                _mm256_add_epi32(discard_bias_vec, boundary_vec));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m128i const query_char_vec = _mm_set1_epi8(query[query_position - 1]);

            // First-column boundary of this row: `M = open + extend * (query_position - 1)`, and the rolling
            // horizontal `E` register reseeded with the discarded magnitude added to that left boundary.
            __m256i const left_boundary_vec = _mm256_set1_epi32(
                static_cast<int>(static_cast<u32_t>(open + extend * (u32_t)(query_position - 1))));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), left_boundary_vec);
            __m256i horizontal_vec = _mm256_add_epi32(discard_bias_vec, left_boundary_vec);

            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m128i const candidate_chars_vec = _mm_loadl_epi64(
                    reinterpret_cast<__m128i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const left_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vertical_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(vertical_row + column * candidate_lanes_k));

                // `cmpeq` yields `0xFF` (-1) on a match, `0x00` on a mismatch, per byte; sign-extend to 8 `i32` so the
                // mask selects `match` on equal lanes and `mismatch` on the rest as the diagonal substitution penalty.
                __m128i const equal_i8_vec = _mm_cmpeq_epi8(query_char_vec, candidate_chars_vec);
                __m256i const equal_i32_vec = _mm256_cvtepi8_epi32(equal_i8_vec);
                __m256i const substitution_addend_vec = _mm256_blendv_epi8(mismatch_vec, match_vec, equal_i32_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi32(diagonal_vec, substitution_addend_vec);

                // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                __m256i const vertical_vec = _mm256_min_epu32(_mm256_add_epi32(up_vec, open_vec),
                                                              _mm256_add_epi32(up_vertical_vec, extend_vec));
                horizontal_vec = _mm256_min_epu32(_mm256_add_epi32(left_vec, open_vec),
                                                  _mm256_add_epi32(horizontal_vec, extend_vec));
                __m256i const cost_if_gap_vec = _mm256_min_epu32(vertical_vec, horizontal_vec);

                __m256i const cell_score_vec = _mm256_min_epu32(cost_if_substitution_vec, cost_if_gap_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                    vertical_vec);
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
 *  @brief Inter-sequence Haswell walker: one @b rune (UTF-32) query against up to 16 rune candidates packed
 *         one-per-lane, 16-bit cells, @b non-unit uniform-cost @b Levenshtein. Distance minimization only.
 *
 *  Rune twin of the 16-lane byte `u16` linear uniform walker: the score recurrence, boundaries, latch, and the
 *  `_mm256_min_epu16` lockstep are byte-for-byte identical; only the diagonal substitution test differs. The 16 live
 *  candidate runes for a column occupy two `__m256i` of 8 `u32` each (lanes 0..7 and 8..15), compared against the
 *  broadcast query rune with `_mm256_cmpeq_epi32` to yield two vectors of `0xFFFFFFFF` (match) / `0x00000000`
 *  (mismatch). Saturating-packing those down to one `__m256i` of 16 `i16` (`_mm256_packs_epi32` keeps `-1`/`0`
 *  exactly) and `_mm256_permute4x64_epi64(.., 0xD8)` repairs the AVX2 cross-128-bit-lane interleave of `packs`,
 *  restoring lane order 0..15; `_mm256_blendv_epi8` then selects the configured `match` cost on equal lanes and
 *  `mismatch` on the rest as the diagonal substitution penalty, while deletion and insertion each add the uniform
 *  `gap` cost. The unit-cost case (match 0, mismatch 1, gap 1) is served by the rune Myers fast path, so this kernel
 *  only ever runs for non-unit costs.
 *
 *  @note The cells are `u16`, so the kernel is only valid while every reachable score stays below 65535; the longest
 *      reachable cell is the all-gap path `max(query, candidate) * gap`. Enforcing that bound (a cost-dependent rune
 *      length limit) is the caller's dispatch contract; the kernel performs no runtime range check.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<rune_t, u16_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 16, void> {

    using char_t = rune_t;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 16;
    static constexpr size_t runes_per_vec_k = 8; // ? `__m256i` holds 8 `u32` runes; 16 lanes span two vectors.

    // The `u16` lane recurrence hardcodes `_mm256_min_epu16`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 16-bit rune candidate-lane kernel only implements distance minimization (Levenshtein).");

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
     *  @param[in] query The shared rune query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 16 rune candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 16`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        // The recurrence honors the engine's configured uniform costs: `match`/`mismatch` on the diagonal and a
        // single `gap` on either deletion or insertion. The unit-cost case (match 0, mismatch 1, gap 1) is served by
        // the rune Myers fast path, so this kernel only ever runs for non-unit costs.
        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const gap_cost = static_cast<score_t>(gap_costs_.open_or_extend);
        __m256i const match_vec = _mm256_set1_epi16(static_cast<short>(match_cost));
        __m256i const mismatch_vec = _mm256_set1_epi16(static_cast<short>(mismatch_cost));
        __m256i const gap_vec = _mm256_set1_epi16(static_cast<short>(gap_cost));

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                _mm256_set1_epi16(static_cast<short>(static_cast<u16_t>(column * gap_cost))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m256i const query_rune_vec = _mm256_set1_epi32(
                static_cast<int>(static_cast<u32_t>(query[query_position - 1])));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row),
                                _mm256_set1_epi16(static_cast<short>(static_cast<u16_t>(query_position * gap_cost))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                // The 16 live candidate runes for this column occupy two `__m256i` of 8 `u32` each.
                __m256i const candidate_runes_low_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidates.position(column - 1)));
                __m256i const candidate_runes_high_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidates.position(column - 1) + runes_per_vec_k));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const deletion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const insertion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                // `cmpeq_epi32` yields `0xFFFFFFFF` (-1) on a match, `0x00000000` on a mismatch, per 32-bit rune lane.
                // Saturating-pack the two 8-lane `i32` masks down to one 16-lane `i16` mask (the `-1`/`0` values stay
                // in range), then `permute4x64` repairs the AVX2 cross-128-bit-lane interleave so lane order is 0..15;
                // `blendv` selects `match` on equal lanes and `mismatch` on the rest as the diagonal penalty.
                __m256i const equal_low_vec = _mm256_cmpeq_epi32(query_rune_vec, candidate_runes_low_vec);
                __m256i const equal_high_vec = _mm256_cmpeq_epi32(query_rune_vec, candidate_runes_high_vec);
                __m256i const equal_packed_vec = _mm256_packs_epi32(equal_low_vec, equal_high_vec);
                __m256i const equal_i16_vec = _mm256_permute4x64_epi64(equal_packed_vec, 0xD8);
                __m256i const mismatch_addend_vec = _mm256_blendv_epi8(mismatch_vec, match_vec, equal_i16_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi16(diagonal_vec, mismatch_addend_vec);
                __m256i const cost_if_deletion_vec = _mm256_add_epi16(deletion_source_vec, gap_vec);
                __m256i const cost_if_insertion_vec = _mm256_add_epi16(insertion_source_vec, gap_vec);
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
 *  @brief Inter-sequence Haswell walker: one @b rune (UTF-32) query against up to 8 rune candidates packed
 *         one-per-lane, 32-bit cells, @b non-unit uniform-cost @b Levenshtein. Distance minimization only.
 *
 *  Width-twin of the 16-lane uniform `u16` linear rune walker, widened to `u32` cells so each `__m256i` holds 8
 *  lanes. The 8 live candidate runes for a column fill a single `__m256i`, compared against the broadcast query rune
 *  with one `_mm256_cmpeq_epi32` yielding 8 `i32` lanes of `0xFFFFFFFF` (match) / `0x00000000` (mismatch); the cell
 *  width already matches the rune-mask width, so no pack/permute is needed and `_mm256_blendv_epi8` selects the
 *  configured `match` cost on equal lanes and `mismatch` on the rest as the diagonal substitution penalty, while
 *  deletion and insertion each add the uniform `gap` cost. The recurrence `cell = min(substitution, min(deletion,
 *  insertion))` advances all 8 lanes in lockstep over `_mm256_min_epu32`. Used when the worst-case score escapes the
 *  `u16` walker's 65535 range but stays below the much wider `u32` ceiling; the unit-cost case is handled by the rune
 *  Myers fast path and never reaches here.
 *
 *  @note The cells are `u32`, so the kernel is only valid while every reachable score stays below 4294967295; the
 *      longest reachable cell is the all-gap path `max(query, candidate) * gap`. Enforcing that bound (a cost-dependent
 *      rune length limit) is the caller's dispatch contract; the kernel performs no runtime range check.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<rune_t, u32_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 8, void> {

    using char_t = rune_t;
    using score_t = u32_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The `u32` lane recurrence hardcodes `_mm256_min_epu32`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 32-bit rune candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (8 `u32` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared rune query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 8 rune candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 8`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const gap_cost = static_cast<score_t>(gap_costs_.open_or_extend);
        __m256i const match_vec = _mm256_set1_epi32(static_cast<int>(match_cost));
        __m256i const mismatch_vec = _mm256_set1_epi32(static_cast<int>(mismatch_cost));
        __m256i const gap_vec = _mm256_set1_epi32(static_cast<int>(gap_cost));

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                _mm256_set1_epi32(static_cast<int>(static_cast<u32_t>(column * gap_cost))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m256i const query_rune_vec = _mm256_set1_epi32(
                static_cast<int>(static_cast<u32_t>(query[query_position - 1])));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row),
                                _mm256_set1_epi32(static_cast<int>(static_cast<u32_t>(query_position * gap_cost))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                // The 8 live candidate runes for this column fill one `__m256i`.
                __m256i const candidate_runes_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const deletion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const insertion_source_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                // `cmpeq_epi32` yields `0xFFFFFFFF` (-1) on a match, `0x00000000` on a mismatch, per 32-bit rune lane;
                // the cell width matches the rune-mask width, so the mask drives `blendv` directly: `match` on equal
                // lanes and `mismatch` on the rest as the diagonal substitution penalty.
                __m256i const equal_i32_vec = _mm256_cmpeq_epi32(query_rune_vec, candidate_runes_vec);
                __m256i const mismatch_addend_vec = _mm256_blendv_epi8(mismatch_vec, match_vec, equal_i32_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi32(diagonal_vec, mismatch_addend_vec);
                __m256i const cost_if_deletion_vec = _mm256_add_epi32(deletion_source_vec, gap_vec);
                __m256i const cost_if_insertion_vec = _mm256_add_epi32(insertion_source_vec, gap_vec);
                __m256i const cell_score_vec = _mm256_min_epu32(
                    cost_if_substitution_vec, _mm256_min_epu32(cost_if_deletion_vec, cost_if_insertion_vec));
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
 *  @brief Inter-sequence Haswell walker: one @b rune (UTF-32) query against up to 16 rune candidates packed
 *         one-per-lane, unit-class @b affine-gap Levenshtein distance, 16-bit unsigned cells. Distance minimization only.
 *
 *  Rune twin of the 16-lane byte `u16` affine walker: the branchless Gotoh E/F tracks over `_mm256_min_epu16`, the
 *  boundaries, the `discard_bias` headroom, and the latch are byte-for-byte identical:
 *      F[column] = min(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *      E         = min(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *      M[column] = min(M_diagonal + substitution, min(E, F))
 *  The only delta from the byte walker is the substitution test: 16 candidate runes occupy two `__m256i` of 8 `u32`,
 *  compared against the broadcast query rune with two `_mm256_cmpeq_epi32`, saturating-packed to one 16-lane `i16`
 *  mask (`_mm256_packs_epi32` + `_mm256_permute4x64_epi64(.., 0xD8)` to repair the AVX2 cross-128-bit-lane interleave),
 *  driving the `_mm256_blendv_epi8` that selects `match` on equal lanes and `mismatch` on the rest.
 *
 *  @note The cells are unsigned `u16`, so the kernel is only valid while every reachable score - plus the
 *      `discard_bias` headroom - stays below 65535; enforcing that bound is the caller's dispatch contract.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<rune_t, u16_t, uniform_substitution_costs_t, affine_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 16, void> {

    using char_t = rune_t;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 16;
    static constexpr size_t runes_per_vec_k = 8; // ? `__m256i` holds 8 `u32` runes; 16 lanes span two vectors.

    // The `u16` lane recurrence hardcodes `_mm256_min_epu16`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 16-bit affine rune candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, affine_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds three `u16` rows of `longest_candidate + 1` lane-vectors: previous/current M plus the F. */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous M row
        amount += score_row_bytes; // current M row
        amount += score_row_bytes; // F (vertical gap) row, carried across rows
        return amount;
    }

    /**
     *  @param[in] query The shared rune query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 16 rune candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Three row buffers carved from the byte span; each lane-vector lives at `row + column * 16`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        score_t *vertical_row = current_row + row_stride;

        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const open = static_cast<score_t>(gap_costs_.open);
        score_t const extend = static_cast<score_t>(gap_costs_.extend);
        __m256i const match_vec = _mm256_set1_epi16(static_cast<short>(match_cost));
        __m256i const mismatch_vec = _mm256_set1_epi16(static_cast<short>(mismatch_cost));
        __m256i const open_vec = _mm256_set1_epi16(static_cast<short>(open));
        __m256i const extend_vec = _mm256_set1_epi16(static_cast<short>(extend));
        // A magnitude above any in-range score so `min` never selects a track that has not been opened yet; the
        // caller's reach bound keeps real scores below it, and `discard_bias + extend` must not wrap `u16`.
        __m256i const discard_bias_vec = _mm256_set1_epi16(static_cast<short>(static_cast<u16_t>(60000)));

        // Row 0 (global): `M[0] = 0`; `M[column] = open + extend * (column - 1)`. The vertical `F` row cannot have
        // been entered from above at row 0, so it is seeded with the discarded magnitude added to the boundary.
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row), _mm256_setzero_si256());
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row), discard_bias_vec);
        for (size_t column = 1; column <= longest_candidate; ++column) {
            __m256i const boundary_vec = _mm256_set1_epi16(
                static_cast<short>(static_cast<u16_t>(open + extend * (u16_t)(column - 1))));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k), boundary_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                _mm256_add_epi16(discard_bias_vec, boundary_vec));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m256i const query_rune_vec = _mm256_set1_epi32(
                static_cast<int>(static_cast<u32_t>(query[query_position - 1])));

            // First-column boundary of this row: `M = open + extend * (query_position - 1)`, and the rolling
            // horizontal `E` register reseeded with the discarded magnitude added to that left boundary.
            __m256i const left_boundary_vec = _mm256_set1_epi16(
                static_cast<short>(static_cast<u16_t>(open + extend * (u16_t)(query_position - 1))));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), left_boundary_vec);
            __m256i horizontal_vec = _mm256_add_epi16(discard_bias_vec, left_boundary_vec);

            for (size_t column = 1; column <= longest_candidate; ++column) {
                // The 16 live candidate runes for this column occupy two `__m256i` of 8 `u32` each.
                __m256i const candidate_runes_low_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidates.position(column - 1)));
                __m256i const candidate_runes_high_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidates.position(column - 1) + runes_per_vec_k));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const left_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vertical_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(vertical_row + column * candidate_lanes_k));

                // Two `cmpeq_epi32` masks set the equal lanes for the low and high 8 candidate runes; saturating-pack
                // them down to one 16-lane `i16` mask and `permute4x64` repairs the AVX2 cross-128-bit-lane interleave,
                // so `blendv` selects `match` on equal lanes and `mismatch` on the rest as the diagonal penalty.
                __m256i const equal_low_vec = _mm256_cmpeq_epi32(query_rune_vec, candidate_runes_low_vec);
                __m256i const equal_high_vec = _mm256_cmpeq_epi32(query_rune_vec, candidate_runes_high_vec);
                __m256i const equal_packed_vec = _mm256_packs_epi32(equal_low_vec, equal_high_vec);
                __m256i const equal_i16_vec = _mm256_permute4x64_epi64(equal_packed_vec, 0xD8);
                __m256i const substitution_addend_vec = _mm256_blendv_epi8(mismatch_vec, match_vec, equal_i16_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi16(diagonal_vec, substitution_addend_vec);

                // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                __m256i const vertical_vec = _mm256_min_epu16(_mm256_add_epi16(up_vec, open_vec),
                                                              _mm256_add_epi16(up_vertical_vec, extend_vec));
                horizontal_vec = _mm256_min_epu16(_mm256_add_epi16(left_vec, open_vec),
                                                  _mm256_add_epi16(horizontal_vec, extend_vec));
                __m256i const cost_if_gap_vec = _mm256_min_epu16(vertical_vec, horizontal_vec);

                __m256i const cell_score_vec = _mm256_min_epu16(cost_if_substitution_vec, cost_if_gap_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                    vertical_vec);
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
 *  @brief Inter-sequence Haswell walker: one @b rune (UTF-32) query against up to 8 rune candidates packed
 *         one-per-lane, unit-class @b affine-gap Levenshtein distance, 32-bit unsigned cells. Distance minimization only.
 *
 *  Width-twin of the 16-lane uniform `u16` affine rune walker, widened to `u32` cells so each `__m256i` holds 8 lanes.
 *  The 8 live candidate runes for a column fill a single `__m256i`, compared against the broadcast query rune with one
 *  `_mm256_cmpeq_epi32` yielding 8 `i32` lanes; the cell width matches the rune-mask width, so the mask drives
 *  `_mm256_blendv_epi8` directly (no pack/permute), and the single gap term is the branchless Gotoh E/F tracks over
 *  `_mm256_min_epu32`:
 *      F[column] = min(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *      E         = min(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *      M[column] = min(M_diagonal + substitution, min(E, F))
 *  A large `discard_bias` is added to any track that cannot have been opened yet, keeping `min` from selecting it.
 *
 *  @note The cells are unsigned `u32`, so the kernel is only valid while every reachable score - plus the
 *      `discard_bias` headroom - stays below 4294967295; enforcing that bound is the caller's dispatch contract.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<rune_t, u32_t, uniform_substitution_costs_t, affine_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_haswell_k, 8, void> {

    using char_t = rune_t;
    using score_t = u32_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The `u32` lane recurrence hardcodes `_mm256_min_epu32`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 32-bit affine rune candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, affine_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds three `u32` rows of `longest_candidate + 1` lane-vectors: previous/current M plus the F. */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous M row
        amount += score_row_bytes; // current M row
        amount += score_row_bytes; // F (vertical gap) row, carried across rows
        return amount;
    }

    /**
     *  @param[in] query The shared rune query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 8 rune candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Three row buffers carved from the byte span; each lane-vector lives at `row + column * 8`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        score_t *vertical_row = current_row + row_stride;

        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const open = static_cast<score_t>(gap_costs_.open);
        score_t const extend = static_cast<score_t>(gap_costs_.extend);
        __m256i const match_vec = _mm256_set1_epi32(static_cast<int>(match_cost));
        __m256i const mismatch_vec = _mm256_set1_epi32(static_cast<int>(mismatch_cost));
        __m256i const open_vec = _mm256_set1_epi32(static_cast<int>(open));
        __m256i const extend_vec = _mm256_set1_epi32(static_cast<int>(extend));
        // A magnitude above any in-range score so `min` never selects a track that has not been opened yet; the
        // caller's reach bound keeps real scores below it, and `discard_bias + extend` must not wrap `u32`.
        __m256i const discard_bias_vec = _mm256_set1_epi32(static_cast<int>(static_cast<u32_t>(2000000000)));

        // Row 0 (global): `M[0] = 0`; `M[column] = open + extend * (column - 1)`. The vertical `F` row cannot have
        // been entered from above at row 0, so it is seeded with the discarded magnitude added to the boundary.
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row), _mm256_setzero_si256());
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row), discard_bias_vec);
        for (size_t column = 1; column <= longest_candidate; ++column) {
            __m256i const boundary_vec = _mm256_set1_epi32(
                static_cast<int>(static_cast<u32_t>(open + extend * (u32_t)(column - 1))));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k), boundary_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                _mm256_add_epi32(discard_bias_vec, boundary_vec));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m256i const query_rune_vec = _mm256_set1_epi32(
                static_cast<int>(static_cast<u32_t>(query[query_position - 1])));

            // First-column boundary of this row: `M = open + extend * (query_position - 1)`, and the rolling
            // horizontal `E` register reseeded with the discarded magnitude added to that left boundary.
            __m256i const left_boundary_vec = _mm256_set1_epi32(
                static_cast<int>(static_cast<u32_t>(open + extend * (u32_t)(query_position - 1))));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), left_boundary_vec);
            __m256i horizontal_vec = _mm256_add_epi32(discard_bias_vec, left_boundary_vec);

            for (size_t column = 1; column <= longest_candidate; ++column) {
                // The 8 live candidate runes for this column fill one `__m256i`.
                __m256i const candidate_runes_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidates.position(column - 1)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const left_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vertical_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(vertical_row + column * candidate_lanes_k));

                // `cmpeq_epi32` yields a per-lane `0xFFFFFFFF`/`0x00000000` mask; the cell width matches the rune-mask
                // width, so `blendv` selects `match` on equal lanes and `mismatch` on the rest as the diagonal penalty.
                __m256i const equal_i32_vec = _mm256_cmpeq_epi32(query_rune_vec, candidate_runes_vec);
                __m256i const substitution_addend_vec = _mm256_blendv_epi8(mismatch_vec, match_vec, equal_i32_vec);
                __m256i const cost_if_substitution_vec = _mm256_add_epi32(diagonal_vec, substitution_addend_vec);

                // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                __m256i const vertical_vec = _mm256_min_epu32(_mm256_add_epi32(up_vec, open_vec),
                                                              _mm256_add_epi32(up_vertical_vec, extend_vec));
                horizontal_vec = _mm256_min_epu32(_mm256_add_epi32(left_vec, open_vec),
                                                  _mm256_add_epi32(horizontal_vec, extend_vec));
                __m256i const cost_if_gap_vec = _mm256_min_epu32(vertical_vec, horizontal_vec);

                __m256i const cell_score_vec = _mm256_min_epu32(cost_if_substitution_vec, cost_if_gap_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                    vertical_vec);
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
 *         Needleman-Wunsch / Smith-Waterman with 32-class substitution costs, @b linear @b or @b affine gaps,
 *         16-bit signed cells. One body covering both gap models and both localities, resolved with `if constexpr`.
 *
 *  Structural twin of the 16-lane uniform `u16` walker, but the unit `cmpeq -> +1` substitution term is replaced by
 *  the resident-row class lookup of `class_lookup_haswell_t`, and the objective is @b maximization, so the base
 *  recurrence is `cell = max(diag + substitution, gap_term)` over signed `i16` cells. The query character is fixed
 *  for an entire Dynamic Programming row, so its class is scalar: per row the cost row
 *  `class_substitution_costs[query_class][0..31]` is made resident with `reload_row`. Inside the column loop the 16
 *  candidate bytes index that resident row through the two-stage `lookup32`, yielding 16 signed `i8` substitution
 *  costs that sign-extend to 16 `i16` lanes.
 *
 *  The two compile-time axes:
 *  - @b gap_costs_type_ : linear collapses the gap term to `max(up, left) + gap`; affine (Gotoh) replaces it with the
 *    branchless E/F tracks (serial reference `serial.hpp:1116-1150`):
 *        F[column] = max(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *        E         = max(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *        M[column] = max(M_diagonal + substitution, max(E, F))
 *    The `F` track is materialized as a third scratch row indexed exactly like `M`; the `E` track only depends on the
 *    cell to its left, so it lives in a single rolling lane-register reseeded per row from the discarded boundary.
 *  - @b locality_ : global (Needleman-Wunsch) seeds the boundary with the gap run and latches each lane's bottom-right
 *    corner; local (Smith-Waterman) zeroes the boundary, clamps every cell to zero, and latches a per-lane running
 *    maximum updated only for columns within each lane's own candidate length so a shorter candidate's zero-padded
 *    tail columns cannot inflate its score.
 *
 *  @note The cells are signed `i16`, so the kernel is only valid while every reachable score stays within
 *      `[-32768, 32767]` (global) or `[0, 32767]` (local); enforcing that bound is the caller's dispatch contract,
 *      the kernel performs no runtime range check.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_, typename gap_costs_type_, sz_similarity_locality_t locality_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, gap_costs_type_, objective_, locality_, sz_cap_haswell_k,
                             16, void> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 16;

    static constexpr bool is_affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    // The signed `i16` recurrence hardcodes `_mm256_max_epi16`; minimization would need a different blend.
    static_assert(
        objective_ == sz_maximize_score_k,
        "The weighted candidate-lane kernel only implements score " "maximization (Needleman-Wunsch / " "Smith-" "Water" "man)" ".");

    substituter_t substituter_ {};
    gap_costs_type_ gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, gap_costs_type_ gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Scratch holds the `i16` score rows of `longest_candidate + 1` lane-vectors (16 cells each): two rows for
     *      a linear gap (previous/current M), three for an affine gap (previous/current M plus the carried F track),
     *      plus one cached candidate-class lane-vector (16 `u8`) per candidate position.
     */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        size_t const class_bytes = candidate_lanes_k * longest_candidate * sizeof(u8_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes;                            // previous M row
        amount += score_row_bytes;                            // current M row
        if constexpr (is_affine_k) amount += score_row_bytes; // F (vertical gap) row, carried across rows
        amount += class_bytes;                                // cached candidate classes
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 16 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        class_lookup_haswell_t lookup;
        lookup.reload_classes(substituter_.byte_to_class);

        __m256i const zero_vec = _mm256_setzero_si256();

        // Per-gap-model constants and the affine F track row; each is consumed only in the matching `if constexpr`
        // branch, so the unused ones in the other instantiation are explicitly tolerated.
        [[maybe_unused]] error_cost_t gap = 0, open = 0, extend = 0;
        [[maybe_unused]] __m256i gap_vec = zero_vec, open_vec = zero_vec, extend_vec = zero_vec,
                                 discard_bias_vec = zero_vec;
        [[maybe_unused]] score_t *vertical_row = nullptr;
        u8_t *candidate_classes = nullptr;
        if constexpr (is_affine_k) {
            open = gap_costs_.open;
            extend = gap_costs_.extend;
            open_vec = _mm256_set1_epi16(static_cast<short>(open));
            extend_vec = _mm256_set1_epi16(static_cast<short>(extend));
            // The supplementary tracks are seeded with a value of higher magnitude than the primary so they are
            // discarded until a real opening exists; mirrors the serial `init_gap` of `(open + extend) + boundary`.
            discard_bias_vec = _mm256_set1_epi16(static_cast<short>(open + extend));
            vertical_row = current_row + row_stride;
            candidate_classes = reinterpret_cast<u8_t *>(vertical_row + row_stride);
        }
        else {
            gap = gap_costs_.open_or_extend;
            gap_vec = _mm256_set1_epi16(static_cast<short>(gap));
            candidate_classes = reinterpret_cast<u8_t *>(current_row + row_stride);
        }

        // Pre-classify every candidate column once; the 16 candidate bytes per column are loop-invariant in rows, so
        // only the cheap class-to-cost stage stays in the hot loop. The 16 live bytes go in the low `__m128i` and the
        // resulting 16 class bytes are stored back into the per-column cache slot.
        for (size_t column = 0; column < longest_candidate; ++column) {
            u256_vec_t candidate_chars_vec;
            candidate_chars_vec.ymm = _mm256_castsi128_si256(
                _mm_loadu_si128(reinterpret_cast<__m128i const *>(candidates.position(column))));
            u256_vec_t const candidate_classes_vec = lookup.classify32(candidate_chars_vec);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(candidate_classes + column * candidate_lanes_k),
                             _mm256_castsi256_si128(candidate_classes_vec.ymm));
        }

        // Local alignment gates a per-lane running maximum by candidate length so a shorter candidate's padded columns
        // are excluded; AVX2 has no mask registers, so a `cmpgt` mask blends the running max with the candidate.
        [[maybe_unused]] alignas(32) i16_t lane_lengths[candidate_lanes_k] = {0};
        [[maybe_unused]] __m256i lane_lengths_vec = zero_vec;
        [[maybe_unused]] __m256i running_max_vec = zero_vec;
        if constexpr (is_local_k) {
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                lane_lengths[lane_index] = static_cast<i16_t>(candidates.lengths[lane_index]);
            lane_lengths_vec = _mm256_load_si256(reinterpret_cast<__m256i const *>(lane_lengths));
        }

        // Row 0 boundary: local zeroes both tracks (an alignment may begin anywhere); global seeds the gap run.
        if constexpr (is_local_k) {
            for (size_t column = 0; column <= longest_candidate; ++column) {
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k), zero_vec);
                if constexpr (is_affine_k)
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                        discard_bias_vec);
            }
        }
        else if constexpr (is_affine_k) {
            // Row 0: one opening plus `column - 1` extensions; the vertical-gap row is seeded with the discarded
            // magnitude so the first real row cannot reuse it.
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row), zero_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row),
                                _mm256_add_epi16(discard_bias_vec, zero_vec));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m256i const boundary_vec = _mm256_set1_epi16(
                    static_cast<short>(static_cast<i16_t>(open + extend * (i16_t)(column - 1))));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                    boundary_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                    _mm256_add_epi16(discard_bias_vec, boundary_vec));
            }
        }
        else {
            // Row 0: a run of `gap * column`, identical across lanes (masked per-lane at latch by each lane's column).
            for (size_t column = 0; column <= longest_candidate; ++column)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                    _mm256_set1_epi16(static_cast<short>(static_cast<i16_t>(gap * (i16_t)column))));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            // The query character is fixed for the whole row, so its class and cost row are scalar.
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            lookup.reload_row(&substituter_.class_substitution_costs[query_class][0]);

            // First-column boundary of the M (and affine E) tracks for this row.
            [[maybe_unused]] __m256i horizontal_vec = zero_vec;
            if constexpr (is_local_k) { _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), zero_vec); }
            else if constexpr (is_affine_k) {
                __m256i const left_boundary_vec = _mm256_set1_epi16(
                    static_cast<short>(static_cast<i16_t>(open + extend * (i16_t)(query_position - 1))));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), left_boundary_vec);
                horizontal_vec = _mm256_add_epi16(discard_bias_vec, left_boundary_vec);
            }
            else {
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(current_row),
                    _mm256_set1_epi16(static_cast<short>(static_cast<i16_t>(gap * (i16_t)query_position))));
            }
            if constexpr (is_local_k && is_affine_k) horizontal_vec = discard_bias_vec;

            for (size_t column = 1; column <= longest_candidate; ++column) {
                // The candidate classes were precomputed once; load this column's 16 cached class bytes into the low
                // half of a `u256_vec_t` and run only the class-to-cost stage to get the 16 signed `i8` costs.
                u256_vec_t candidate_classes_vec;
                candidate_classes_vec.ymm = _mm256_castsi128_si256(_mm_loadu_si128(
                    reinterpret_cast<__m128i const *>(candidate_classes + (column - 1) * candidate_lanes_k)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const left_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                u256_vec_t const cost_i8_vec = lookup.costs_for_classes32(candidate_classes_vec);
                __m256i const cost_i16_vec = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(cost_i8_vec.ymm));

                __m256i const cost_if_substitution_vec = _mm256_add_epi16(diagonal_vec, cost_i16_vec);
                __m256i cost_if_gap_vec;
                if constexpr (is_affine_k) {
                    // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                    __m256i const up_vertical_vec = _mm256_loadu_si256(
                        reinterpret_cast<__m256i const *>(vertical_row + column * candidate_lanes_k));
                    __m256i const vertical_vec = _mm256_max_epi16(_mm256_add_epi16(up_vec, open_vec),
                                                                  _mm256_add_epi16(up_vertical_vec, extend_vec));
                    horizontal_vec = _mm256_max_epi16(_mm256_add_epi16(left_vec, open_vec),
                                                      _mm256_add_epi16(horizontal_vec, extend_vec));
                    cost_if_gap_vec = _mm256_max_epi16(vertical_vec, horizontal_vec);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                        vertical_vec);
                }
                else { cost_if_gap_vec = _mm256_add_epi16(_mm256_max_epi16(up_vec, left_vec), gap_vec); }

                __m256i cell_score_vec = _mm256_max_epi16(cost_if_substitution_vec, cost_if_gap_vec);
                if constexpr (is_local_k) cell_score_vec = _mm256_max_epi16(zero_vec, cell_score_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row + column * candidate_lanes_k),
                                    cell_score_vec);

                // Fold this column into the running maximum only for lanes whose candidate reaches it.
                if constexpr (is_local_k) {
                    __m256i const column_live_vec = _mm256_cmpgt_epi16(
                        lane_lengths_vec, _mm256_set1_epi16(static_cast<short>(column - 1)));
                    __m256i const folded_max_vec = _mm256_max_epi16(running_max_vec, cell_score_vec);
                    running_max_vec = _mm256_blendv_epi8(running_max_vec, folded_max_vec, column_live_vec);
                }
            }
            trivial_swap(previous_row, current_row);
        }

        if constexpr (is_local_k) {
            alignas(32) i16_t final_max[candidate_lanes_k];
            _mm256_store_si256(reinterpret_cast<__m256i *>(final_max), running_max_vec);
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                result_lanes[lane_index] = final_max[lane_index];
        }
        else {
            // Latch each live lane's result from its own final column; ragged lengths mean different columns per lane.
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
                size_t const candidate_length = candidates.lengths[lane_index];
                result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
            }
        }
        return status_t::success_k;
    }
};

/**
 *  @brief Inter-sequence Haswell walker: one query against up to 8 candidates packed one-per-lane, weighted
 *         Needleman-Wunsch / Smith-Waterman with 32-class substitution costs, @b linear @b or @b affine gaps,
 *         32-bit signed cells. One body covering both gap models and both localities, resolved with `if constexpr`.
 *
 *  Width-mirror of the 16-lane weighted `i16` walker, widened to `i32` cells so each `__m256i` holds 8 lanes. The
 *  unit `cmpeq -> +1` substitution term is the resident-row class lookup of `class_lookup_haswell_t`, and the
 *  objective is @b maximization, so the base recurrence is `cell = max(diag + substitution, gap_term)` over signed
 *  `i32` cells. The query character is fixed for an entire Dynamic Programming row, so its class is scalar: per row
 *  the cost row `class_substitution_costs[query_class][0..31]` is made resident with `reload_row`. Inside the column
 *  loop only the 8 live candidate bytes index that resident row through the two-stage `lookup32`, yielding 8 signed
 *  `i8` substitution costs that sign-extend to 8 `i32` lanes via `_mm256_cvtepi8_epi32`.
 *
 *  The two compile-time axes:
 *  - @b gap_costs_type_ : linear collapses the gap term to `max(up, left) + gap`; affine (Gotoh) replaces it with the
 *    branchless E/F tracks:
 *        F[column] = max(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *        E         = max(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *        M[column] = max(M_diagonal + substitution, max(E, F))
 *    The `F` track is materialized as a third scratch row indexed exactly like `M`; the `E` track only depends on the
 *    cell to its left, so it lives in a single rolling lane-register reseeded per row from the discarded boundary.
 *  - @b locality_ : global (Needleman-Wunsch) seeds the boundary with the gap run and latches each lane's bottom-right
 *    corner; local (Smith-Waterman) zeroes the boundary, clamps every cell to zero, and latches a per-lane running
 *    maximum updated only for columns within each lane's own candidate length so a shorter candidate's zero-padded
 *    tail columns cannot inflate its score.
 *
 *  @note The cells are signed `i32`, so the kernel is only valid while every reachable score stays within
 *      `[-2147483648, 2147483647]` (global) or `[0, 2147483647]` (local); enforcing that bound is the caller's
 *      dispatch contract, the kernel performs no runtime range check.
 *  @note Requires Intel Haswell generation CPUs or newer (AVX2).
 */
template <sz_similarity_objective_t objective_, typename gap_costs_type_, sz_similarity_locality_t locality_>
struct candidate_lane_walker<char, i32_t, error_costs_32x32_t, gap_costs_type_, objective_, locality_, sz_cap_haswell_k,
                             8, void> {

    using char_t = char;
    using score_t = i32_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr size_t candidate_lanes_k = 8;

    static constexpr bool is_affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    // The signed `i32` recurrence hardcodes `_mm256_max_epi32`; minimization would need a different blend.
    static_assert(
        objective_ == sz_maximize_score_k,
        "The weighted candidate-lane kernel only implements score " "maximization (Needleman-Wunsch / " "Smith-" "Water" "man)" ".");

    substituter_t substituter_ {};
    gap_costs_type_ gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, gap_costs_type_ gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Scratch holds the `i32` score rows of `longest_candidate + 1` lane-vectors (8 cells each): two rows for
     *      a linear gap (previous/current M), three for an affine gap (previous/current M plus the carried F track),
     *      plus one cached candidate-class lane-vector (8 `u8`) per candidate position.
     */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        size_t const class_bytes = candidate_lanes_k * longest_candidate * sizeof(u8_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes;                            // previous M row
        amount += score_row_bytes;                            // current M row
        if constexpr (is_affine_k) amount += score_row_bytes; // F (vertical gap) row, carried across rows
        amount += class_bytes;                                // cached candidate classes
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 8 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        class_lookup_haswell_t lookup;
        lookup.reload_classes(substituter_.byte_to_class);

        __m256i const zero_vec = _mm256_setzero_si256();

        // Per-gap-model constants and the affine F track row; each is consumed only in the matching `if constexpr`
        // branch, so the unused ones in the other instantiation are explicitly tolerated.
        [[maybe_unused]] error_cost_t gap = 0, open = 0, extend = 0;
        [[maybe_unused]] __m256i gap_vec = zero_vec, open_vec = zero_vec, extend_vec = zero_vec,
                                 discard_bias_vec = zero_vec;
        [[maybe_unused]] score_t *vertical_row = nullptr;
        u8_t *candidate_classes = nullptr;
        if constexpr (is_affine_k) {
            open = gap_costs_.open;
            extend = gap_costs_.extend;
            open_vec = _mm256_set1_epi32(static_cast<int>(open));
            extend_vec = _mm256_set1_epi32(static_cast<int>(extend));
            // The supplementary tracks are seeded with a value of higher magnitude than the primary so they are
            // discarded until a real opening exists; mirrors the serial `init_gap` of `(open + extend) + boundary`.
            discard_bias_vec = _mm256_set1_epi32(static_cast<int>(open + extend));
            vertical_row = current_row + row_stride;
            candidate_classes = reinterpret_cast<u8_t *>(vertical_row + row_stride);
        }
        else {
            gap = gap_costs_.open_or_extend;
            gap_vec = _mm256_set1_epi32(static_cast<int>(gap));
            candidate_classes = reinterpret_cast<u8_t *>(current_row + row_stride);
        }

        // Pre-classify every candidate column once; the 8 candidate bytes per column are loop-invariant in rows, so
        // only the cheap class-to-cost stage stays in the hot loop. The 8 live bytes go in the low quadword and the
        // resulting 8 class bytes are stored back into the per-column cache slot.
        for (size_t column = 0; column < longest_candidate; ++column) {
            u256_vec_t candidate_chars_vec;
            candidate_chars_vec.ymm = _mm256_castsi128_si256(
                _mm_loadl_epi64(reinterpret_cast<__m128i const *>(candidates.position(column))));
            u256_vec_t const candidate_classes_vec = lookup.classify32(candidate_chars_vec);
            _mm_storel_epi64(reinterpret_cast<__m128i *>(candidate_classes + column * candidate_lanes_k),
                             _mm256_castsi256_si128(candidate_classes_vec.ymm));
        }

        // Local alignment gates a per-lane running maximum by candidate length so a shorter candidate's padded columns
        // are excluded; AVX2 has no mask registers, so a `cmpgt` mask blends the running max with the candidate.
        [[maybe_unused]] alignas(32) i32_t lane_lengths[candidate_lanes_k] = {0};
        [[maybe_unused]] __m256i lane_lengths_vec = zero_vec;
        [[maybe_unused]] __m256i running_max_vec = zero_vec;
        if constexpr (is_local_k) {
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                lane_lengths[lane_index] = static_cast<i32_t>(candidates.lengths[lane_index]);
            lane_lengths_vec = _mm256_load_si256(reinterpret_cast<__m256i const *>(lane_lengths));
        }

        // Row 0 boundary: local zeroes both tracks (an alignment may begin anywhere); global seeds the gap run.
        if constexpr (is_local_k) {
            for (size_t column = 0; column <= longest_candidate; ++column) {
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k), zero_vec);
                if constexpr (is_affine_k)
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                        discard_bias_vec);
            }
        }
        else if constexpr (is_affine_k) {
            // Row 0: one opening plus `column - 1` extensions; the vertical-gap row is seeded with the discarded
            // magnitude so the first real row cannot reuse it.
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row), zero_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row),
                                _mm256_add_epi32(discard_bias_vec, zero_vec));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m256i const boundary_vec = _mm256_set1_epi32(
                    static_cast<int>(static_cast<i32_t>(open + extend * (i32_t)(column - 1))));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                    boundary_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                    _mm256_add_epi32(discard_bias_vec, boundary_vec));
            }
        }
        else {
            // Row 0: a run of `gap * column`, identical across lanes (masked per-lane at latch by each lane's column).
            for (size_t column = 0; column <= longest_candidate; ++column)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(previous_row + column * candidate_lanes_k),
                                    _mm256_set1_epi32(static_cast<int>(static_cast<i32_t>(gap * (i32_t)column))));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            // The query character is fixed for the whole row, so its class and cost row are scalar.
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            lookup.reload_row(&substituter_.class_substitution_costs[query_class][0]);

            // First-column boundary of the M (and affine E) tracks for this row.
            [[maybe_unused]] __m256i horizontal_vec = zero_vec;
            if constexpr (is_local_k) { _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), zero_vec); }
            else if constexpr (is_affine_k) {
                __m256i const left_boundary_vec = _mm256_set1_epi32(
                    static_cast<int>(static_cast<i32_t>(open + extend * (i32_t)(query_position - 1))));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row), left_boundary_vec);
                horizontal_vec = _mm256_add_epi32(discard_bias_vec, left_boundary_vec);
            }
            else {
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(current_row),
                    _mm256_set1_epi32(static_cast<int>(static_cast<i32_t>(gap * (i32_t)query_position))));
            }
            if constexpr (is_local_k && is_affine_k) horizontal_vec = discard_bias_vec;

            for (size_t column = 1; column <= longest_candidate; ++column) {
                // The candidate classes were precomputed once; load this column's 8 cached class bytes into the low
                // quadword of a `u256_vec_t` and run only the class-to-cost stage to get the 8 signed `i8` costs.
                u256_vec_t candidate_classes_vec;
                candidate_classes_vec.ymm = _mm256_castsi128_si256(_mm_loadl_epi64(
                    reinterpret_cast<__m128i const *>(candidate_classes + (column - 1) * candidate_lanes_k)));
                __m256i const diagonal_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + (column - 1) * candidate_lanes_k));
                __m256i const up_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(previous_row + column * candidate_lanes_k));
                __m256i const left_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(current_row + (column - 1) * candidate_lanes_k));

                u256_vec_t const cost_i8_vec = lookup.costs_for_classes32(candidate_classes_vec);
                __m256i const cost_i32_vec = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(cost_i8_vec.ymm));

                __m256i const cost_if_substitution_vec = _mm256_add_epi32(diagonal_vec, cost_i32_vec);
                __m256i cost_if_gap_vec;
                if constexpr (is_affine_k) {
                    // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                    __m256i const up_vertical_vec = _mm256_loadu_si256(
                        reinterpret_cast<__m256i const *>(vertical_row + column * candidate_lanes_k));
                    __m256i const vertical_vec = _mm256_max_epi32(_mm256_add_epi32(up_vec, open_vec),
                                                                  _mm256_add_epi32(up_vertical_vec, extend_vec));
                    horizontal_vec = _mm256_max_epi32(_mm256_add_epi32(left_vec, open_vec),
                                                      _mm256_add_epi32(horizontal_vec, extend_vec));
                    cost_if_gap_vec = _mm256_max_epi32(vertical_vec, horizontal_vec);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(vertical_row + column * candidate_lanes_k),
                                        vertical_vec);
                }
                else { cost_if_gap_vec = _mm256_add_epi32(_mm256_max_epi32(up_vec, left_vec), gap_vec); }

                __m256i cell_score_vec = _mm256_max_epi32(cost_if_substitution_vec, cost_if_gap_vec);
                if constexpr (is_local_k) cell_score_vec = _mm256_max_epi32(zero_vec, cell_score_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(current_row + column * candidate_lanes_k),
                                    cell_score_vec);

                // Fold this column into the running maximum only for lanes whose candidate reaches it.
                if constexpr (is_local_k) {
                    __m256i const column_live_vec = _mm256_cmpgt_epi32(lane_lengths_vec,
                                                                       _mm256_set1_epi32(static_cast<int>(column - 1)));
                    __m256i const folded_max_vec = _mm256_max_epi32(running_max_vec, cell_score_vec);
                    running_max_vec = _mm256_blendv_epi8(running_max_vec, folded_max_vec, column_live_vec);
                }
            }
            trivial_swap(previous_row, current_row);
        }

        if constexpr (is_local_k) {
            alignas(32) i32_t final_max[candidate_lanes_k];
            _mm256_store_si256(reinterpret_cast<__m256i *>(final_max), running_max_vec);
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                result_lanes[lane_index] = final_max[lane_index];
        }
        else {
            // Latch each live lane's result from its own final column; ragged lengths mean different columns per lane.
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
                size_t const candidate_length = candidates.lengths[lane_index];
                result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
            }
        }
        return status_t::success_k;
    }
};

#pragma endregion Inter Sequence Candidate Lanes

#pragma region Inter Sequence Byte Myers

/**
 *  @brief AVX2 Myers/Hyyrö unit-cost Levenshtein for Haswell, scoring four independent pairs at once with one Myers
 *      integer per 64-bit YMM lane. The structural twin of the Ice Lake eight-lane byte Myers at 1/4 the width -
 *      `__m256i` (four `u64` lanes) instead of `__m512i` (eight) - and with the AVX-512 mask machinery emulated in
 *      vector algebra, since AVX2 has no `__mmask` registers, no `VPTERNLOGQ`, and no unsigned `VPCMP`:
 *      - `distances_4x64_`             - four single-word Myers (shorter <= 64), one 64-bit integer per lane;
 *      - `distances_4x_multiword_<words_count_>` - four multi-word Myers with a compile-time word count, covering
 *        shorter in `(64, 64 * words_count_]`; the word state lives in stack arrays so the loop unrolls and the
 *        `vertical_positive` / `vertical_negative` words register-promote;
 *      - `distances_4x_multiword_large_` - the runtime-`words_count` sibling for the long tail (shorter > 512).
 *      The multi-word kernels carry two intra-lane ripples (the 65-bit `(Eq & VP) + VP + addition_carry` and the
 *      `horizontal_positive` / `horizontal_negative` bit63->bit0 shift); neither ever crosses a lane. Variable
 *      per-lane lengths are handled with an active mask (an all-ones `__m256i` lane while that pair still has text)
 *      that freezes finished lanes.
 *
 *  AVX-512 -> AVX2 boolean lowering used throughout:
 *      - `Xh = (((Eq & VP) + VP) ^ VP) | Eq` -> `_mm256_or_si256(_mm256_xor_si256(sum, VP), Eq)`.
 *      - `Ph = Mv | ~(Xh | VP)` (VPTERNLOGQ 0xF1) -> `_mm256_or_si256(Mv, _mm256_andnot_si256(Xh | VP, ones))`.
 *      - `Pv' = Mh | ~(Xv | Ph)` (same 0xF1 shape) -> the same `or(andnot(or(a, b), ones))` lowering.
 *      - a per-lane mask add/sub `score +/- 1` -> `_mm256_add/sub_epi64(score, _mm256_and_si256(one, lane_mask))`.
 *      - `_mm512_test_epi64_mask(x, top)` -> `~_mm256_cmpeq_epi64(_mm256_and_si256(x, top), zero)` (nonzero lanes).
 *      - `_mm512_cmplt_epu64_mask(a, b)` -> a signed `_mm256_cmpgt_epi64` after flipping both sign bits.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<
    char, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using index_t = u32_t;
    static constexpr index_t lanes_k = 4;
    static constexpr size_t match_masks_bytes_k = sizeof(u64_t) * lanes_k * 256; // ? 8 KB `match_masks` table.

    levenshtein_distance_myers() noexcept {}

    /** @brief Single-pair scratch sizing for the per-pair `levenshtein_distance` engine - delegated to the scalar
     *      serial Myers (one pair gains nothing from AVX2 batching; the 4-lane kernels serve the cross-product). */
    auto layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        return levenshtein_distance_myers<char, sz_cap_serial_k> {}.layout(first, second, specs);
    }

    /** @brief Single-pair Myers (per-pair engine path) - delegated to the scalar serial Myers. */
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        scratch_space_t scratch_space) noexcept {
        return levenshtein_distance_myers<char, sz_cap_serial_k> {}(first, second, result_ref, scratch_space);
    }

    /** @brief Per-lane "nonzero" predicate: all-ones in lanes where `(value & probe) != 0`, else all-zeros. */
    static __m256i lane_test_(__m256i value, __m256i probe, __m256i zero) noexcept {
        __m256i const masked = _mm256_and_si256(value, probe);
        // `cmpeq` is all-ones where `masked == 0`; the nonzero predicate is its complement.
        return _mm256_andnot_si256(_mm256_cmpeq_epi64(masked, zero), _mm256_set1_epi64x(-1));
    }

    /** @brief Per-lane unsigned `a < b` predicate via a signed `cmpgt` after flipping both sign bits. */
    static __m256i lane_less_unsigned_(__m256i first, __m256i second, __m256i sign_bit) noexcept {
        return _mm256_cmpgt_epi64(_mm256_xor_si256(second, sign_bit), _mm256_xor_si256(first, sign_bit));
    }

    /** @brief `first | ~(second | third)` - the AVX2 lowering of the shared `Ph` / `Pv'` VPTERNLOGQ(0xF1) shape. */
    static __m256i or_nor_(__m256i first, __m256i second, __m256i third, __m256i ones) noexcept {
        return _mm256_or_si256(first, _mm256_andnot_si256(_mm256_or_si256(second, third), ones));
    }

    /**
     *  @brief Four independent single-word Myers distances, one per 64-bit YMM lane (each shorter side <= 64).
     *      The hot path for short words: no carry, shift, or boundary masking crosses lanes - every lane is its
     *      own 64-bit Myers integer. @p scratch_space holds the `match_masks[4][256]` table (`match_masks_bytes_k`)
     *      followed by a transposed-text buffer of at least `max_longer * 4` bytes.
     */
    template <typename results_writer_>
    status_t distances_4x64_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                             scratch_space_t scratch_space) const noexcept {

        size_t max_longer = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
        if (scratch_space.size() < match_masks_bytes_k + max_longer * lanes_k) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data()); // ? Indexed `lane * 256 + symbol`.
        u8_t *const transposed_text = reinterpret_cast<u8_t *>(scratch_space.data() + match_masks_bytes_k);
        alignas(32) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (size_t position = 0; position != max_longer * lanes_k; ++position) transposed_text[position] = 0;

        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            size_t const longer_length = pairs.longers[lane_index].size();
            char_t const *const shorter = pairs.shorters[lane_index].data();
            char_t const *const longer = pairs.longers[lane_index].data();
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[lane_index * 256 + (u8_t)shorter[position]] = 0;
            for (size_t position = 0; position != longer_length; ++position)
                match_masks[lane_index * 256 + (u8_t)longer[position]] = 0;
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[lane_index * 256 + (u8_t)shorter[position]] |= (u64_t)1 << position;
            top_bits[lane_index] = (u64_t)1 << (shorter_length - 1);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = longer_length;
            for (size_t position = 0; position != longer_length; ++position)
                transposed_text[position * lanes_k + lane_index] = (u8_t)longer[position];
        }

        __m256i const lane_offsets = _mm256_set_epi64x(3 * 256, 2 * 256, 1 * 256, 0);
        __m256i const one = _mm256_set1_epi64x(1);
        __m256i const ones = _mm256_set1_epi64x(-1);
        __m256i const zero = _mm256_setzero_si256();
        __m256i const top_mask = _mm256_load_si256((__m256i const *)top_bits);
        __m256i const longer_vec = _mm256_load_si256((__m256i const *)longer_lengths);
        __m256i const length_vec = _mm256_load_si256((__m256i const *)shorter_lengths);
        // VP = the low `shorter_length` bits set (a shift count of 64 yields 0, so `(1 << 64) - 1` == ~0).
        __m256i vertical_positive = _mm256_sub_epi64(_mm256_sllv_epi64(one, length_vec), one);
        __m256i vertical_negative = _mm256_setzero_si256();
        __m256i score = length_vec;

        for (size_t position = 0; position != max_longer; ++position) {
            __m256i const active = _mm256_cmpgt_epi64(longer_vec, _mm256_set1_epi64x((long long)position));
            __m256i const symbols = _mm256_cvtepu8_epi64(
                _mm_loadl_epi64((__m128i const *)(transposed_text + position * lanes_k)));
            __m256i const equality = _mm256_i64gather_epi64((long long const *)match_masks,
                                                            _mm256_add_epi64(lane_offsets, symbols), 8);
            __m256i const carry_in = _mm256_or_si256(equality, vertical_negative);
            // Xh = (((Eq & VP) + VP) ^ VP) | Eq.
            __m256i const sum = _mm256_add_epi64(_mm256_and_si256(equality, vertical_positive), vertical_positive);
            __m256i const diagonal = _mm256_or_si256(_mm256_xor_si256(sum, vertical_positive), equality);
            __m256i horizontal_positive = or_nor_(vertical_negative, diagonal, vertical_positive, ones); // Mv | ~(D|VP)
            __m256i horizontal_negative = _mm256_and_si256(vertical_positive, diagonal);                 // VP & D
            __m256i const add_mask = _mm256_and_si256(active, lane_test_(horizontal_positive, top_mask, zero));
            __m256i const sub_mask = _mm256_and_si256(active, lane_test_(horizontal_negative, top_mask, zero));
            score = _mm256_add_epi64(score, _mm256_and_si256(one, add_mask));
            score = _mm256_sub_epi64(score, _mm256_and_si256(one, sub_mask));
            horizontal_positive = _mm256_or_si256(_mm256_slli_epi64(horizontal_positive, 1), one);
            horizontal_negative = _mm256_slli_epi64(horizontal_negative, 1);
            // Pv' = Mh | ~(Xv | Ph).
            __m256i const next_positive = or_nor_(horizontal_negative, carry_in, horizontal_positive, ones);
            __m256i const next_negative = _mm256_and_si256(horizontal_positive, carry_in);
            vertical_positive = _mm256_blendv_epi8(vertical_positive, next_positive, active);
            vertical_negative = _mm256_blendv_epi8(vertical_negative, next_negative, active);
        }

        alignas(32) u64_t final_scores[lanes_k];
        _mm256_store_si256((__m256i *)final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief Four independent compile-time multi-word Myers distances, one pair per 64-bit YMM lane, covering
     *      shorter sides in `(64, 64 * words_count_]`. Each lane is one register-resident `words_count_`-word Myers
     *      integer; the per-word `vertical_positive` / `vertical_negative` state lives in `__m256i[words_count_]`
     *      stack arrays so the word loop unrolls and the words promote to registers. Two carries cross words @b within
     *      a lane (never across lanes): the 65-bit `(Eq & VP) + VP + addition_carry` ripple, tracked low->high via
     *      unsigned overflow detection, and the `horizontal_positive` / `horizontal_negative` bit63->bit0 shift.
     *
     *      @p scratch_space holds the per-lane multi-word `match_masks` table (`match_masks_bytes_k * words_count_`,
     *      base `match_masks + lane * 256 * words_count_`, entry `[symbol * words_count_ + word]`).
     */
    template <size_t words_count_, typename results_writer_>
    status_t distances_4x_multiword_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                     scratch_space_t scratch_space) const noexcept {

        constexpr size_t words_count = words_count_;

        size_t max_longer = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
        size_t const match_masks_words = (size_t)256 * words_count * lanes_k;
        if (scratch_space.size() < match_masks_words * sizeof(u64_t)) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data());
        for (size_t element = 0; element != match_masks_words; ++element) match_masks[element] = 0;

        alignas(32) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            char_t const *const shorter = pairs.shorters[lane_index].data();
            u64_t *const lane_table = match_masks + (size_t)lane_index * 256 * words_count;
            for (index_t position = 0; position != shorter_length; ++position)
                lane_table[(size_t)(u8_t)shorter[position] * words_count + (position >> 6)] |= (u64_t)1
                                                                                               << (position & 63);
            top_bits[lane_index] = (u64_t)1 << ((shorter_length - 1) & 63);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
        }

        // Each lane keeps `words_count_` 64-bit Myers words on the stack; word `w` of every lane lives in one YMM.
        __m256i vertical_positive[words_count_];
        __m256i vertical_negative[words_count_];
        for (size_t word = 0; word != words_count; ++word) {
            vertical_positive[word] = _mm256_set1_epi64x(-1);
            vertical_negative[word] = _mm256_setzero_si256();
        }

        __m256i const one = _mm256_set1_epi64x(1);
        __m256i const ones = _mm256_set1_epi64x(-1);
        __m256i const zero = _mm256_setzero_si256();
        __m256i const sign_bit = _mm256_set1_epi64x((long long)((u64_t)1 << 63));
        __m256i const top_mask = _mm256_load_si256((__m256i const *)top_bits);
        __m256i const longer_vec = _mm256_load_si256((__m256i const *)longer_lengths);
        __m256i score = _mm256_load_si256((__m256i const *)shorter_lengths);
        constexpr size_t last_word = words_count_ - 1;

        for (size_t position = 0; position != max_longer; ++position) {
            __m256i const active = _mm256_cmpgt_epi64(longer_vec, _mm256_set1_epi64x((long long)position));
            // Per-lane base offset of the current character's word row inside that lane's table.
            alignas(32) u64_t base_offsets[lanes_k] = {0};
            for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index) {
                bool const lane_active = lane_index < pairs.lanes_count() &&
                                         position < pairs.longers[lane_index].size();
                u8_t const symbol = lane_active ? (u8_t)pairs.longers[lane_index].data()[position] : 0;
                base_offsets[lane_index] = lane_active
                                               ? (u64_t)lane_index * 256 * words_count + (u64_t)symbol * words_count
                                               : 0;
            }

            __m256i addition_carry = _mm256_setzero_si256(); // ? 0/1 per lane, the `(Eq&VP)+VP` ripple.
            __m256i horizontal_positive_carry = one;         // ? Word 0 seeds the leading 1.
            __m256i horizontal_negative_carry = _mm256_setzero_si256();

            for (size_t word = 0; word != words_count; ++word) {
                alignas(32) u64_t equality_words[lanes_k];
                for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index)
                    equality_words[lane_index] = (lane_index < pairs.lanes_count() &&
                                                  position < pairs.longers[lane_index].size())
                                                     ? match_masks[(size_t)base_offsets[lane_index] + word]
                                                     : 0;
                __m256i const equality = _mm256_load_si256((__m256i const *)equality_words);

                __m256i const vertical_positive_word = vertical_positive[word];
                __m256i const vertical_negative_word = vertical_negative[word];

                // sum = (Eq & VP) + VP + addition_carry, tracking the per-lane carry-out low->high across words.
                __m256i const summand = _mm256_and_si256(equality, vertical_positive_word);
                __m256i const sum_low = _mm256_add_epi64(summand, vertical_positive_word);
                __m256i const carry_from_summand = lane_less_unsigned_(sum_low, summand, sign_bit);
                __m256i const sum = _mm256_add_epi64(sum_low, addition_carry);
                __m256i const carry_from_incoming = lane_less_unsigned_(sum, sum_low, sign_bit);
                addition_carry = _mm256_and_si256(one, _mm256_or_si256(carry_from_summand, carry_from_incoming));

                __m256i const carry_in = _mm256_or_si256(equality, vertical_negative_word); // ? Eq | VN
                // Xh = (sum ^ VP) | Eq | VN == (sum ^ VP) | carry_in.
                __m256i const diagonal = _mm256_or_si256(_mm256_xor_si256(sum, vertical_positive_word), carry_in);
                __m256i horizontal_positive = or_nor_(vertical_negative_word, diagonal, vertical_positive_word,
                                                      ones);                                      // ? VN | ~(D|VP)
                __m256i horizontal_negative = _mm256_and_si256(vertical_positive_word, diagonal); // ? VP & D

                if (word == last_word) {
                    __m256i const add_mask = _mm256_and_si256(active, lane_test_(horizontal_positive, top_mask, zero));
                    __m256i const sub_mask = _mm256_and_si256(active, lane_test_(horizontal_negative, top_mask, zero));
                    score = _mm256_add_epi64(score, _mm256_and_si256(one, add_mask));
                    score = _mm256_sub_epi64(score, _mm256_and_si256(one, sub_mask));
                }

                // Save each word's bit63 as the next word's bit0 carry, then shift up by one.
                __m256i const next_positive_carry = _mm256_srli_epi64(horizontal_positive, 63);
                __m256i const next_negative_carry = _mm256_srli_epi64(horizontal_negative, 63);
                horizontal_positive = _mm256_or_si256(_mm256_slli_epi64(horizontal_positive, 1),
                                                      horizontal_positive_carry);
                horizontal_negative = _mm256_or_si256(_mm256_slli_epi64(horizontal_negative, 1),
                                                      horizontal_negative_carry);
                horizontal_positive_carry = next_positive_carry;
                horizontal_negative_carry = next_negative_carry;

                // Pv' = Mh | ~(Xv | Ph); applied only to active lanes.
                __m256i const next_positive = or_nor_(horizontal_negative, carry_in, horizontal_positive, ones);
                __m256i const next_negative = _mm256_and_si256(horizontal_positive, carry_in);
                vertical_positive[word] = _mm256_blendv_epi8(vertical_positive_word, next_positive, active);
                vertical_negative[word] = _mm256_blendv_epi8(vertical_negative_word, next_negative, active);
            }
        }

        alignas(32) u64_t final_scores[lanes_k];
        _mm256_store_si256((__m256i *)final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief The runtime-`words_count` sibling of `distances_4x_multiword_<words_count_>` for the long tail
     *      (shorter > 512), where instantiating one variant per exact word count is not worthwhile. Each lane
     *      carries its own `ceil(shorter / 64)`-word Myers integer; the group shares a single @p words_count so every
     *      lane loops the
     *      same word range. The per-word state lives in a fixed-capacity stack array indexed at runtime; the math is
     *      identical to the templated variant.
     *
     *      @p scratch_space holds the per-lane multi-word `match_masks` table (`match_masks_bytes_k * words_count`,
     *      base `match_masks + lane * 256 * words_count`, entry `[symbol * words_count + word]`).
     */
    template <typename results_writer_>
    status_t distances_4x_multiword_large_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                           scratch_space_t scratch_space) const noexcept {

        static constexpr size_t stack_words_capacity_k = 64; // ? Covers shorter sides up to 4096 on the stack.

        size_t max_longer = 0, max_shorter = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
            max_shorter = sz_max_of_two(max_shorter, pairs.shorters[lane_index].size());
        }
        size_t const words_count = divide_round_up(max_shorter, (size_t)64);
        if (words_count == 0 || words_count > stack_words_capacity_k) return status_t::bad_alloc_k;
        size_t const match_masks_words = (size_t)256 * words_count * lanes_k;
        if (scratch_space.size() < match_masks_words * sizeof(u64_t)) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data());
        for (size_t element = 0; element != match_masks_words; ++element) match_masks[element] = 0;

        alignas(32) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            char_t const *const shorter = pairs.shorters[lane_index].data();
            u64_t *const lane_table = match_masks + (size_t)lane_index * 256 * words_count;
            for (index_t position = 0; position != shorter_length; ++position)
                lane_table[(size_t)(u8_t)shorter[position] * words_count + (position >> 6)] |= (u64_t)1
                                                                                               << (position & 63);
            top_bits[lane_index] = (u64_t)1 << ((shorter_length - 1) & 63);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
        }

        // Each lane keeps `words_count` 64-bit Myers words; word `w` of every lane lives in one YMM register.
        __m256i vertical_positive[stack_words_capacity_k];
        __m256i vertical_negative[stack_words_capacity_k];
        for (size_t word = 0; word != words_count; ++word) {
            vertical_positive[word] = _mm256_set1_epi64x(-1);
            vertical_negative[word] = _mm256_setzero_si256();
        }

        __m256i const one = _mm256_set1_epi64x(1);
        __m256i const ones = _mm256_set1_epi64x(-1);
        __m256i const zero = _mm256_setzero_si256();
        __m256i const sign_bit = _mm256_set1_epi64x((long long)((u64_t)1 << 63));
        __m256i const top_mask = _mm256_load_si256((__m256i const *)top_bits);
        __m256i const longer_vec = _mm256_load_si256((__m256i const *)longer_lengths);
        __m256i score = _mm256_load_si256((__m256i const *)shorter_lengths);
        size_t const last_word = words_count - 1;

        for (size_t position = 0; position != max_longer; ++position) {
            __m256i const active = _mm256_cmpgt_epi64(longer_vec, _mm256_set1_epi64x((long long)position));
            alignas(32) u64_t base_offsets[lanes_k] = {0};
            for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index) {
                bool const lane_active = lane_index < pairs.lanes_count() &&
                                         position < pairs.longers[lane_index].size();
                u8_t const symbol = lane_active ? (u8_t)pairs.longers[lane_index].data()[position] : 0;
                base_offsets[lane_index] = lane_active
                                               ? (u64_t)lane_index * 256 * words_count + (u64_t)symbol * words_count
                                               : 0;
            }

            __m256i addition_carry = _mm256_setzero_si256();
            __m256i horizontal_positive_carry = one;
            __m256i horizontal_negative_carry = _mm256_setzero_si256();

            for (size_t word = 0; word != words_count; ++word) {
                alignas(32) u64_t equality_words[lanes_k];
                for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index)
                    equality_words[lane_index] = (lane_index < pairs.lanes_count() &&
                                                  position < pairs.longers[lane_index].size())
                                                     ? match_masks[(size_t)base_offsets[lane_index] + word]
                                                     : 0;
                __m256i const equality = _mm256_load_si256((__m256i const *)equality_words);

                __m256i const vertical_positive_word = vertical_positive[word];
                __m256i const vertical_negative_word = vertical_negative[word];

                __m256i const summand = _mm256_and_si256(equality, vertical_positive_word);
                __m256i const sum_low = _mm256_add_epi64(summand, vertical_positive_word);
                __m256i const carry_from_summand = lane_less_unsigned_(sum_low, summand, sign_bit);
                __m256i const sum = _mm256_add_epi64(sum_low, addition_carry);
                __m256i const carry_from_incoming = lane_less_unsigned_(sum, sum_low, sign_bit);
                addition_carry = _mm256_and_si256(one, _mm256_or_si256(carry_from_summand, carry_from_incoming));

                __m256i const carry_in = _mm256_or_si256(equality, vertical_negative_word);
                __m256i const diagonal = _mm256_or_si256(_mm256_xor_si256(sum, vertical_positive_word), carry_in);
                __m256i horizontal_positive = or_nor_(vertical_negative_word, diagonal, vertical_positive_word, ones);
                __m256i horizontal_negative = _mm256_and_si256(vertical_positive_word, diagonal);

                if (word == last_word) {
                    __m256i const add_mask = _mm256_and_si256(active, lane_test_(horizontal_positive, top_mask, zero));
                    __m256i const sub_mask = _mm256_and_si256(active, lane_test_(horizontal_negative, top_mask, zero));
                    score = _mm256_add_epi64(score, _mm256_and_si256(one, add_mask));
                    score = _mm256_sub_epi64(score, _mm256_and_si256(one, sub_mask));
                }

                __m256i const next_positive_carry = _mm256_srli_epi64(horizontal_positive, 63);
                __m256i const next_negative_carry = _mm256_srli_epi64(horizontal_negative, 63);
                horizontal_positive = _mm256_or_si256(_mm256_slli_epi64(horizontal_positive, 1),
                                                      horizontal_positive_carry);
                horizontal_negative = _mm256_or_si256(_mm256_slli_epi64(horizontal_negative, 1),
                                                      horizontal_negative_carry);
                horizontal_positive_carry = next_positive_carry;
                horizontal_negative_carry = next_negative_carry;

                __m256i const next_positive = or_nor_(horizontal_negative, carry_in, horizontal_positive, ones);
                __m256i const next_negative = _mm256_and_si256(horizontal_positive, carry_in);
                vertical_positive[word] = _mm256_blendv_epi8(vertical_positive_word, next_positive, active);
                vertical_negative[word] = _mm256_blendv_epi8(vertical_negative_word, next_negative, active);
            }
        }

        alignas(32) u64_t final_scores[lanes_k];
        _mm256_store_si256((__m256i *)final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }
};

#pragma endregion Inter Sequence Byte Myers

#pragma region Inter Sequence Rune Myers

/**
 *  @brief AVX2 Myers/Hyyr\xC3\xB6 unit-cost @b rune (UTF-32) Levenshtein for Haswell, scoring four independent pairs at
 *      once with one Myers integer per 64-bit YMM lane - the rune twin of the byte `levenshtein_distance_myers` and the
 *      four-lane sibling of the Ice Lake eight-lane rune Myers. The scan is bit-for-bit identical to the AVX2 byte
 *      4x64 / 4xN family; the only delta is the `match_masks` source. A byte pattern indexes a dense 256-row table, but a rune
 *      pattern (up to 0x10FFFF distinct keys) cannot, so each lane builds a small @b open-addressing hash over its own
 *      pattern's distinct runes (capacity `next_pow2(2*distinct)`, load factor <= 0.5), and per text position each of
 *      the four lanes probes its lane's hash for the current text rune to assemble the `Eq` bitmask words. A text rune
 *      absent from a lane's pattern hits the lane's permanently-zero `absent_row`, i.e. an all-zero `Eq` (matches
 *      nothing) - the same semantics as a byte miss. The hash math (multiply-shift `hash_rune`, the `empty_slot_k`
 *      sentinel, the linear-probe build/lookup) is the @b verbatim machinery of the Ice Lake rune Myers (the hash is
 *      scalar host-side per lane; only the per-lane SIMD scan differs), so the two stay bit-exact.
 *      - `distances_4x64_`             - four single-word Myers (shorter <= 64 runes), one 64-bit integer per lane;
 *      - `distances_4x_multiword_<words_count_>` - four multi-word Myers with a compile-time word count, covering
 *        shorter in `(64, 64 * words_count_]` runes;
 *      - `distances_4x_multiword_large_` - the runtime-`words_count` sibling for the long tail (shorter > 512 runes).
 *
 *  AVX-512 -> AVX2 boolean lowering matches the byte four-lane Myers (no `__mmask`, no `VPTERNLOGQ`, no unsigned
 *  `VPCMP`): `lane_test_` for the nonzero top-bit predicate, `lane_less_unsigned_` for the multi-word carry, and
 *  `or_nor_` for the shared `Ph` / `Pv'` shape.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<
    rune_t, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = rune_t;
    using index_t = u32_t;
    static constexpr index_t lanes_k = 4;
    static constexpr size_t match_masks_bytes_k = sizeof(u64_t) * lanes_k * 256; // ? Reported for parity with the byte
    // ? Myers; the rune `match_masks` is hash-sized
    // ? via `scratch_bytes_for` instead.

    /** @brief Sentinel rune marking an empty hash slot (`rune_t` never reaches `0xFFFFFFFF`; valid <= 0x10FFFF). */
    static constexpr rune_t empty_slot_k = static_cast<rune_t>(0xFFFFFFFFu);

    levenshtein_distance_myers() noexcept {}

    /** @brief Per-lane "nonzero" predicate: all-ones in lanes where `(value & probe) != 0`, else all-zeros. */
    static __m256i lane_test_(__m256i value, __m256i probe, __m256i zero) noexcept {
        __m256i const masked = _mm256_and_si256(value, probe);
        return _mm256_andnot_si256(_mm256_cmpeq_epi64(masked, zero), _mm256_set1_epi64x(-1));
    }

    /** @brief Per-lane unsigned `a < b` predicate via a signed `cmpgt` after flipping both sign bits. */
    static __m256i lane_less_unsigned_(__m256i first, __m256i second, __m256i sign_bit) noexcept {
        return _mm256_cmpgt_epi64(_mm256_xor_si256(second, sign_bit), _mm256_xor_si256(first, sign_bit));
    }

    /** @brief `first | ~(second | third)` - the AVX2 lowering of the shared `Ph` / `Pv'` VPTERNLOGQ(0xF1) shape. */
    static __m256i or_nor_(__m256i first, __m256i second, __m256i third, __m256i ones) noexcept {
        return _mm256_or_si256(first, _mm256_andnot_si256(_mm256_or_si256(second, third), ones));
    }

    /** @brief Number of 64-bit words spanning a @p shorter_length-rune pattern, i.e. `ceil(shorter_length / 64)`. */
    static size_t words_count_for(size_t shorter_length) noexcept {
        return divide_round_up<size_t>(shorter_length, 64);
    }

    /**
     *  @brief Open-addressing capacity (a power of two) for a pattern of @p distinct_upper_bound distinct runes. The
     *      pattern has at most `shorter_length` distinct runes, so passing the longest shorter side is a safe upper
     *      bound for the whole 4-lane group. The `2 *` keeps the load factor <= 0.5 for cheap linear probing;
     *      `sz_size_bit_ceil` rounds to a power of two so the multiply-shift hash maps cleanly onto the slot range.
     */
    static index_t hash_capacity_for(size_t distinct_upper_bound) noexcept {
        size_t const slots_wanted = sz_max_of_two(2 * distinct_upper_bound, (size_t)1);
        return static_cast<index_t>(sz_size_bit_ceil(slots_wanted));
    }

    /**
     *  @brief Multiply-shift hash of a rune into `[0, capacity)`; @p capacity must be a power of two. The 64-bit
     *      Fibonacci multiply spreads the rune's bits into the high word, then a single mask selects the slot - no
     *      `>> 64` undefined shift at the `capacity == 1` boundary.
     */
    static index_t hash_rune(rune_t rune, index_t capacity) noexcept {
        u64_t const mixed = static_cast<u64_t>(static_cast<u32_t>(rune)) * 0x9E3779B97F4A7C15ull;
        return static_cast<index_t>((mixed >> 32) & static_cast<u64_t>(capacity - 1));
    }

    /**
     *  @brief Scratch bytes the 4-lane rune `match_masks` needs for a group whose longest shorter side is @p max_shorter
     *      runes: four per-lane hash tables (slot keys + `words_count` bitmask words per slot) plus one shared
     *      permanently-zero `absent_row`. This is the single source of truth for both the scratch sizing in the
     *      cross-product driver and the carving inside the kernels, so the two can never disagree.
     */
    static size_t scratch_bytes_for(size_t max_shorter) noexcept {
        size_t const words_count = words_count_for(sz_max_of_two(max_shorter, (size_t)1));
        size_t const capacity = hash_capacity_for(sz_max_of_two(max_shorter, (size_t)1));
        size_t const slot_keys_bytes = sizeof(rune_t) * capacity * lanes_k;
        size_t const slot_masks_bytes = sizeof(u64_t) * capacity * words_count * lanes_k;
        size_t const absent_row_bytes = sizeof(u64_t) * words_count;
        return slot_keys_bytes + slot_masks_bytes + absent_row_bytes;
    }

#pragma region Per Lane Hash match_masks

    /**
     *  @brief Carves the 4-lane hash `match_masks` out of @p scratch_space and builds it from each lane's shorter side. Every
     *      lane shares the group-wide @p capacity and @p words_count (so the inner scan loops one common word range,
     *      exactly like the byte 4xN kernel); a lane's pattern is hashed into its own slot region. Returns `false`
     *      when @p scratch_space is too small. On success @p slot_keys / @p slot_masks / @p absent_row point into
     *      @p scratch_space and the lane tables are populated; @p absent_row is the all-zero row probed for misses.
     */
    static bool build_lane_hashes_(span<char_t const> const *shorters, index_t pairs_active, index_t capacity,
                                   size_t words_count, scratch_space_t scratch_space, rune_t *&slot_keys,
                                   u64_t *&slot_masks, u64_t *&absent_row) noexcept {
        size_t const slot_keys_bytes = sizeof(rune_t) * capacity * lanes_k;
        size_t const slot_masks_bytes = sizeof(u64_t) * capacity * words_count * lanes_k;
        size_t const absent_row_bytes = sizeof(u64_t) * words_count;
        if (scratch_space.size() < slot_keys_bytes + slot_masks_bytes + absent_row_bytes) return false;

        slot_keys = reinterpret_cast<rune_t *>(scratch_space.data());
        slot_masks = reinterpret_cast<u64_t *>(scratch_space.data() + slot_keys_bytes);
        absent_row = reinterpret_cast<u64_t *>(scratch_space.data() + slot_keys_bytes + slot_masks_bytes);

        for (size_t word = 0; word != words_count; ++word) absent_row[word] = 0;
        // Clear every slot of every lane (including dead lanes, so their probes land on the empty sentinel -> miss).
        for (size_t slot = 0; slot != (size_t)capacity * lanes_k; ++slot) slot_keys[slot] = empty_slot_k;

        for (index_t lane = 0; lane != pairs_active; ++lane) {
            rune_t *const lane_keys = slot_keys + (size_t)lane * capacity;
            u64_t *const lane_masks = slot_masks + (size_t)lane * capacity * words_count;
            index_t const shorter_length = (index_t)shorters[lane].size();
            char_t const *const shorter = shorters[lane].data();
            for (index_t position = 0; position != shorter_length; ++position) {
                rune_t const rune = shorter[position];
                index_t slot = hash_rune(rune, capacity);
                for (;; slot = (slot + 1) & (capacity - 1)) {
                    if (lane_keys[slot] == rune) break;
                    if (lane_keys[slot] == empty_slot_k) {
                        lane_keys[slot] = rune;
                        for (size_t word = 0; word != words_count; ++word)
                            lane_masks[(size_t)slot * words_count + word] = 0;
                        break;
                    }
                }
                lane_masks[(size_t)slot * words_count + (position >> 6)] |= (u64_t)1 << (position & 63);
            }
        }
        return true;
    }

    /**
     *  @brief Probes lane @p lane's hash for text rune @p symbol, returning a pointer to its `words_count` bitmask
     *      words - the slot's row if the rune is in the lane's pattern, or the shared all-zero @p absent_row (a
     *      whole-row miss) otherwise. Mirrors the serial rune oracle's lookup verbatim.
     */
    static u64_t const *lane_match_row_(rune_t const *slot_keys, u64_t const *slot_masks, u64_t const *absent_row,
                                        index_t lane, index_t capacity, size_t words_count, rune_t symbol) noexcept {
        rune_t const *const lane_keys = slot_keys + (size_t)lane * capacity;
        u64_t const *const lane_masks = slot_masks + (size_t)lane * capacity * words_count;
        for (index_t slot = hash_rune(symbol, capacity);; slot = (slot + 1) & (capacity - 1)) {
            rune_t const key = lane_keys[slot];
            if (key == symbol) return &lane_masks[(size_t)slot * words_count];
            if (key == empty_slot_k) break;
        }
        return absent_row;
    }

#pragma endregion Per Lane Hash match_masks

    /**
     *  @brief Four independent single-word rune Myers distances, one per 64-bit YMM lane (each shorter side <= 64
     *      runes). The scan is verbatim the byte `distances_4x64_`; only the `Eq` source differs - per text position
     *      each lane hash-probes its rune to a single 64-bit bitmask, and the four masks are assembled into one YMM.
     *      @p scratch_space holds the 4-lane hash `match_masks` (`scratch_bytes_for(max_shorter)`).
     */
    template <typename results_writer_>
    status_t distances_4x64_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                             scratch_space_t scratch_space) const noexcept {

        size_t max_longer = 0, max_shorter = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
            max_shorter = sz_max_of_two(max_shorter, pairs.shorters[lane_index].size());
        }
        index_t const capacity = hash_capacity_for(sz_max_of_two(max_shorter, (size_t)1));
        rune_t *slot_keys = nullptr;
        u64_t *slot_masks = nullptr, *absent_row = nullptr;
        if (!build_lane_hashes_(pairs.shorters.data(), (index_t)pairs.lanes_count(), capacity, 1, scratch_space,
                                slot_keys, slot_masks, absent_row))
            return status_t::bad_alloc_k;

        alignas(32) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            top_bits[lane_index] = (u64_t)1 << (shorter_length - 1);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
        }

        __m256i const one = _mm256_set1_epi64x(1);
        __m256i const ones = _mm256_set1_epi64x(-1);
        __m256i const zero = _mm256_setzero_si256();
        __m256i const top_mask = _mm256_load_si256((__m256i const *)top_bits);
        __m256i const longer_vec = _mm256_load_si256((__m256i const *)longer_lengths);
        __m256i const length_vec = _mm256_load_si256((__m256i const *)shorter_lengths);
        // VP = the low `shorter_length` bits set (a shift count of 64 yields 0, so `(1 << 64) - 1` == ~0).
        __m256i vertical_positive = _mm256_sub_epi64(_mm256_sllv_epi64(one, length_vec), one);
        __m256i vertical_negative = _mm256_setzero_si256();
        __m256i score = length_vec;

        for (size_t position = 0; position != max_longer; ++position) {
            __m256i const active = _mm256_cmpgt_epi64(longer_vec, _mm256_set1_epi64x((long long)position));
            // Per-lane hash probe of the current text rune -> one 64-bit `Eq` mask per lane, assembled into one YMM.
            alignas(32) u64_t equality_words[lanes_k];
            for (index_t lane = 0; lane != lanes_k; ++lane) {
                bool const lane_active = lane < pairs.lanes_count() && position < pairs.longers[lane].size();
                if (!lane_active) {
                    equality_words[lane] = 0;
                    continue;
                }
                rune_t const symbol = pairs.longers[lane].data()[position];
                equality_words[lane] = lane_match_row_(slot_keys, slot_masks, absent_row, lane, capacity, 1, symbol)[0];
            }
            __m256i const equality = _mm256_load_si256((__m256i const *)equality_words);
            __m256i const carry_in = _mm256_or_si256(equality, vertical_negative);
            // Xh = (((Eq & VP) + VP) ^ VP) | Eq.
            __m256i const sum = _mm256_add_epi64(_mm256_and_si256(equality, vertical_positive), vertical_positive);
            __m256i const diagonal = _mm256_or_si256(_mm256_xor_si256(sum, vertical_positive), equality);
            __m256i horizontal_positive = or_nor_(vertical_negative, diagonal, vertical_positive, ones); // Mv | ~(D|VP)
            __m256i horizontal_negative = _mm256_and_si256(vertical_positive, diagonal);                 // VP & D
            __m256i const add_mask = _mm256_and_si256(active, lane_test_(horizontal_positive, top_mask, zero));
            __m256i const sub_mask = _mm256_and_si256(active, lane_test_(horizontal_negative, top_mask, zero));
            score = _mm256_add_epi64(score, _mm256_and_si256(one, add_mask));
            score = _mm256_sub_epi64(score, _mm256_and_si256(one, sub_mask));
            horizontal_positive = _mm256_or_si256(_mm256_slli_epi64(horizontal_positive, 1), one);
            horizontal_negative = _mm256_slli_epi64(horizontal_negative, 1);
            // Pv' = Mh | ~(Xv | Ph).
            __m256i const next_positive = or_nor_(horizontal_negative, carry_in, horizontal_positive, ones);
            __m256i const next_negative = _mm256_and_si256(horizontal_positive, carry_in);
            vertical_positive = _mm256_blendv_epi8(vertical_positive, next_positive, active);
            vertical_negative = _mm256_blendv_epi8(vertical_negative, next_negative, active);
        }

        alignas(32) u64_t final_scores[lanes_k];
        _mm256_store_si256((__m256i *)final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief Four independent compile-time multi-word rune Myers distances, one pair per 64-bit YMM lane, covering
     *      shorter sides in `(64, 64 * words_count_]` runes. The scan is verbatim the byte
     *      `distances_4x_multiword_<words_count_>` (two intra-lane ripples: the 65-bit `(Eq & VP) + VP + addition_carry`
     *      and the bit63->bit0 shift carries; neither crosses a lane); only the `Eq` source differs - each lane probes
     *      its hash once per text rune and reads the resulting row's `words_count_` bitmask words. @p scratch_space
     *      holds the 4-lane hash `match_masks` (`scratch_bytes_for(max_shorter)`).
     */
    template <size_t words_count_, typename results_writer_>
    status_t distances_4x_multiword_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                     scratch_space_t scratch_space) const noexcept {

        constexpr size_t words_count = words_count_;

        size_t max_longer = 0, max_shorter = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
            max_shorter = sz_max_of_two(max_shorter, pairs.shorters[lane_index].size());
        }
        index_t const capacity = hash_capacity_for(sz_max_of_two(max_shorter, (size_t)1));
        rune_t *slot_keys = nullptr;
        u64_t *slot_masks = nullptr, *absent_row = nullptr;
        if (!build_lane_hashes_(pairs.shorters.data(), (index_t)pairs.lanes_count(), capacity, words_count,
                                scratch_space, slot_keys, slot_masks, absent_row))
            return status_t::bad_alloc_k;

        alignas(32) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            top_bits[lane_index] = (u64_t)1 << ((shorter_length - 1) & 63);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
        }

        // Each lane keeps `words_count_` 64-bit Myers words on the stack; word `w` of every lane lives in one YMM.
        __m256i vertical_positive[words_count_];
        __m256i vertical_negative[words_count_];
        for (size_t word = 0; word != words_count; ++word) {
            vertical_positive[word] = _mm256_set1_epi64x(-1);
            vertical_negative[word] = _mm256_setzero_si256();
        }

        __m256i const one = _mm256_set1_epi64x(1);
        __m256i const ones = _mm256_set1_epi64x(-1);
        __m256i const zero = _mm256_setzero_si256();
        __m256i const sign_bit = _mm256_set1_epi64x((long long)((u64_t)1 << 63));
        __m256i const top_mask = _mm256_load_si256((__m256i const *)top_bits);
        __m256i const longer_vec = _mm256_load_si256((__m256i const *)longer_lengths);
        __m256i score = _mm256_load_si256((__m256i const *)shorter_lengths);
        constexpr size_t last_word = words_count_ - 1;

        for (size_t position = 0; position != max_longer; ++position) {
            __m256i const active = _mm256_cmpgt_epi64(longer_vec, _mm256_set1_epi64x((long long)position));
            // Per-lane hash probe of the current text rune -> the lane's `words_count` bitmask row (or `absent_row`).
            u64_t const *match_rows[lanes_k];
            for (index_t lane = 0; lane != lanes_k; ++lane) {
                bool const lane_active = lane < pairs.lanes_count() && position < pairs.longers[lane].size();
                rune_t const symbol = lane_active ? pairs.longers[lane].data()[position] : empty_slot_k;
                match_rows[lane] = lane_active ? lane_match_row_(slot_keys, slot_masks, absent_row, lane, capacity,
                                                                 words_count, symbol)
                                               : absent_row;
            }

            __m256i addition_carry = _mm256_setzero_si256(); // ? 0/1 per lane, the `(Eq&VP)+VP` ripple.
            __m256i horizontal_positive_carry = one;         // ? Word 0 seeds the leading 1.
            __m256i horizontal_negative_carry = _mm256_setzero_si256();

            for (size_t word = 0; word != words_count; ++word) {
                alignas(32) u64_t equality_words[lanes_k];
                for (index_t lane = 0; lane != lanes_k; ++lane)
                    equality_words[lane] = (lane < pairs.lanes_count() && position < pairs.longers[lane].size())
                                               ? match_rows[lane][word]
                                               : 0;
                __m256i const equality = _mm256_load_si256((__m256i const *)equality_words);

                __m256i const vertical_positive_word = vertical_positive[word];
                __m256i const vertical_negative_word = vertical_negative[word];

                // sum = (Eq & VP) + VP + addition_carry, tracking the per-lane carry-out low->high across words.
                __m256i const summand = _mm256_and_si256(equality, vertical_positive_word);
                __m256i const sum_low = _mm256_add_epi64(summand, vertical_positive_word);
                __m256i const carry_from_summand = lane_less_unsigned_(sum_low, summand, sign_bit);
                __m256i const sum = _mm256_add_epi64(sum_low, addition_carry);
                __m256i const carry_from_incoming = lane_less_unsigned_(sum, sum_low, sign_bit);
                addition_carry = _mm256_and_si256(one, _mm256_or_si256(carry_from_summand, carry_from_incoming));

                __m256i const carry_in = _mm256_or_si256(equality, vertical_negative_word); // ? Eq | VN
                // Xh = (sum ^ VP) | Eq | VN == (sum ^ VP) | carry_in.
                __m256i const diagonal = _mm256_or_si256(_mm256_xor_si256(sum, vertical_positive_word), carry_in);
                __m256i horizontal_positive = or_nor_(vertical_negative_word, diagonal, vertical_positive_word,
                                                      ones);                                      // ? VN | ~(D|VP)
                __m256i horizontal_negative = _mm256_and_si256(vertical_positive_word, diagonal); // ? VP & D

                if (word == last_word) {
                    __m256i const add_mask = _mm256_and_si256(active, lane_test_(horizontal_positive, top_mask, zero));
                    __m256i const sub_mask = _mm256_and_si256(active, lane_test_(horizontal_negative, top_mask, zero));
                    score = _mm256_add_epi64(score, _mm256_and_si256(one, add_mask));
                    score = _mm256_sub_epi64(score, _mm256_and_si256(one, sub_mask));
                }

                // Save each word's bit63 as the next word's bit0 carry, then shift up by one.
                __m256i const next_positive_carry = _mm256_srli_epi64(horizontal_positive, 63);
                __m256i const next_negative_carry = _mm256_srli_epi64(horizontal_negative, 63);
                horizontal_positive = _mm256_or_si256(_mm256_slli_epi64(horizontal_positive, 1),
                                                      horizontal_positive_carry);
                horizontal_negative = _mm256_or_si256(_mm256_slli_epi64(horizontal_negative, 1),
                                                      horizontal_negative_carry);
                horizontal_positive_carry = next_positive_carry;
                horizontal_negative_carry = next_negative_carry;

                // Pv' = Mh | ~(Xv | Ph); applied only to active lanes.
                __m256i const next_positive = or_nor_(horizontal_negative, carry_in, horizontal_positive, ones);
                __m256i const next_negative = _mm256_and_si256(horizontal_positive, carry_in);
                vertical_positive[word] = _mm256_blendv_epi8(vertical_positive_word, next_positive, active);
                vertical_negative[word] = _mm256_blendv_epi8(vertical_negative_word, next_negative, active);
            }
        }

        alignas(32) u64_t final_scores[lanes_k];
        _mm256_store_si256((__m256i *)final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief The runtime-`words_count` sibling of `distances_4x_multiword_<words_count_>` for the long tail (shorter >
     *      512 runes). Each lane carries its own `ceil(shorter / 64)`-word Myers integer; the group shares a single
     *      @p words_count so every lane loops the same word range. The math is identical to the templated variant; the
     *      per-word state lives in a fixed-capacity stack array indexed at runtime. @p scratch_space holds the 4-lane
     *      hash `match_masks` (`scratch_bytes_for(max_shorter)`).
     */
    template <typename results_writer_>
    status_t distances_4x_multiword_large_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                           scratch_space_t scratch_space) const noexcept {

        static constexpr size_t stack_words_capacity_k = 64; // ? Covers shorter sides up to 4096 runes on the stack.

        size_t max_longer = 0, max_shorter = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
            max_shorter = sz_max_of_two(max_shorter, pairs.shorters[lane_index].size());
        }
        size_t const words_count = words_count_for(sz_max_of_two(max_shorter, (size_t)1));
        if (words_count == 0 || words_count > stack_words_capacity_k) return status_t::bad_alloc_k;
        index_t const capacity = hash_capacity_for(sz_max_of_two(max_shorter, (size_t)1));
        rune_t *slot_keys = nullptr;
        u64_t *slot_masks = nullptr, *absent_row = nullptr;
        if (!build_lane_hashes_(pairs.shorters.data(), (index_t)pairs.lanes_count(), capacity, words_count,
                                scratch_space, slot_keys, slot_masks, absent_row))
            return status_t::bad_alloc_k;

        alignas(32) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            top_bits[lane_index] = (u64_t)1 << ((shorter_length - 1) & 63);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
        }

        // Each lane keeps `words_count` 64-bit Myers words; word `w` of every lane lives in one YMM register.
        __m256i vertical_positive[stack_words_capacity_k];
        __m256i vertical_negative[stack_words_capacity_k];
        for (size_t word = 0; word != words_count; ++word) {
            vertical_positive[word] = _mm256_set1_epi64x(-1);
            vertical_negative[word] = _mm256_setzero_si256();
        }

        __m256i const one = _mm256_set1_epi64x(1);
        __m256i const ones = _mm256_set1_epi64x(-1);
        __m256i const zero = _mm256_setzero_si256();
        __m256i const sign_bit = _mm256_set1_epi64x((long long)((u64_t)1 << 63));
        __m256i const top_mask = _mm256_load_si256((__m256i const *)top_bits);
        __m256i const longer_vec = _mm256_load_si256((__m256i const *)longer_lengths);
        __m256i score = _mm256_load_si256((__m256i const *)shorter_lengths);
        size_t const last_word = words_count - 1;

        for (size_t position = 0; position != max_longer; ++position) {
            __m256i const active = _mm256_cmpgt_epi64(longer_vec, _mm256_set1_epi64x((long long)position));
            u64_t const *match_rows[lanes_k];
            for (index_t lane = 0; lane != lanes_k; ++lane) {
                bool const lane_active = lane < pairs.lanes_count() && position < pairs.longers[lane].size();
                rune_t const symbol = lane_active ? pairs.longers[lane].data()[position] : empty_slot_k;
                match_rows[lane] = lane_active ? lane_match_row_(slot_keys, slot_masks, absent_row, lane, capacity,
                                                                 words_count, symbol)
                                               : absent_row;
            }

            __m256i addition_carry = _mm256_setzero_si256();
            __m256i horizontal_positive_carry = one;
            __m256i horizontal_negative_carry = _mm256_setzero_si256();

            for (size_t word = 0; word != words_count; ++word) {
                alignas(32) u64_t equality_words[lanes_k];
                for (index_t lane = 0; lane != lanes_k; ++lane)
                    equality_words[lane] = (lane < pairs.lanes_count() && position < pairs.longers[lane].size())
                                               ? match_rows[lane][word]
                                               : 0;
                __m256i const equality = _mm256_load_si256((__m256i const *)equality_words);

                __m256i const vertical_positive_word = vertical_positive[word];
                __m256i const vertical_negative_word = vertical_negative[word];

                __m256i const summand = _mm256_and_si256(equality, vertical_positive_word);
                __m256i const sum_low = _mm256_add_epi64(summand, vertical_positive_word);
                __m256i const carry_from_summand = lane_less_unsigned_(sum_low, summand, sign_bit);
                __m256i const sum = _mm256_add_epi64(sum_low, addition_carry);
                __m256i const carry_from_incoming = lane_less_unsigned_(sum, sum_low, sign_bit);
                addition_carry = _mm256_and_si256(one, _mm256_or_si256(carry_from_summand, carry_from_incoming));

                __m256i const carry_in = _mm256_or_si256(equality, vertical_negative_word);
                __m256i const diagonal = _mm256_or_si256(_mm256_xor_si256(sum, vertical_positive_word), carry_in);
                __m256i horizontal_positive = or_nor_(vertical_negative_word, diagonal, vertical_positive_word, ones);
                __m256i horizontal_negative = _mm256_and_si256(vertical_positive_word, diagonal);

                if (word == last_word) {
                    __m256i const add_mask = _mm256_and_si256(active, lane_test_(horizontal_positive, top_mask, zero));
                    __m256i const sub_mask = _mm256_and_si256(active, lane_test_(horizontal_negative, top_mask, zero));
                    score = _mm256_add_epi64(score, _mm256_and_si256(one, add_mask));
                    score = _mm256_sub_epi64(score, _mm256_and_si256(one, sub_mask));
                }

                __m256i const next_positive_carry = _mm256_srli_epi64(horizontal_positive, 63);
                __m256i const next_negative_carry = _mm256_srli_epi64(horizontal_negative, 63);
                horizontal_positive = _mm256_or_si256(_mm256_slli_epi64(horizontal_positive, 1),
                                                      horizontal_positive_carry);
                horizontal_negative = _mm256_or_si256(_mm256_slli_epi64(horizontal_negative, 1),
                                                      horizontal_negative_carry);
                horizontal_positive_carry = next_positive_carry;
                horizontal_negative_carry = next_negative_carry;

                __m256i const next_positive = or_nor_(horizontal_negative, carry_in, horizontal_positive, ones);
                __m256i const next_negative = _mm256_and_si256(horizontal_positive, carry_in);
                vertical_positive[word] = _mm256_blendv_epi8(vertical_positive_word, next_positive, active);
                vertical_negative[word] = _mm256_blendv_epi8(vertical_negative_word, next_negative, active);
            }
        }

        alignas(32) u64_t final_scores[lanes_k];
        _mm256_store_si256((__m256i *)final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }
};

#pragma endregion Inter Sequence Rune Myers

#pragma region Inter Sequence Cross Product Engines

/**
 *  @brief Batched byte-level Levenshtein distances on Haswell, scoring four unit-cost pairs at once with the
 *      @b AVX2 Myers family - `distances_4x64_` (shorter <= 64), `distances_4x_multiword_<words_count>` (compile-time
 *      word count, shorter <= 512), and `distances_4x_multiword_large_` (runtime word count, shorter > 512) - and the
 *      anti-diagonal
 *      DP for the lone over-512 cell. Cells are grouped by their exact `ceil(shorter / 64)` word bucket so each group
 *      loops one common word count, the four-lane twin of the Ice Lake eight-lane byte Myers engine. Non-unit
 *      substitution or gap costs cannot use the `+-1`-delta Myers recurrence, so they route through the inter-sequence
 *      `u16` shared-query `candidate_lane_walker` (16 candidates per block), with the per-pair anti-diagonal serial DP
 *      as the long-tail fallback for cells whose worst-case distance escapes the `u16` range.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<
    linear_gap_costs_t, allocator_type_, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;    // ? `u16` lanes for the non-unit candidate-lane walker.
    static constexpr size_t u16_reach_limit_k = 60000; // ? `u16` headroom for the non-unit lane walker.
    using scoring_t = levenshtein_distance<char, gap_costs_t, sz_cap_serial_k>; // ? Per-pair DP fallback.
    using myers_t = levenshtein_distance_myers<char, capability_k>;             // ? AVX2 four-lane Myers.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, (int)candidate_lanes_k,
                              void>; // ? AVX2 non-unit shared-query `u16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, u32_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, 8,
                              void>; // ? AVX2 non-unit shared-query `u32` lanes.
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

    /** @brief Whether a `(query, candidate)` cell's worst-case distance stays inside the narrow walker's `u16` headroom. */
    bool fits_u16_(size_t query_length, size_t candidate_length) const noexcept {
        size_t const magnitude = sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open_or_extend);
        return (query_length + candidate_length) * magnitude <= u16_reach_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case distance stays inside the wide walker's `u32` headroom. */
    bool fits_u32_(size_t query_length, size_t candidate_length) const noexcept {
        size_t const magnitude = sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open_or_extend);
        return (query_length + candidate_length) * magnitude <= 1500000000;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the Myers `match_masks` + transposed-text
     *      buffer for the longest string, or the anti-diagonal DP fallback for the longest query × longest candidate.
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
        size_t const max_longer = sz_max_of_two(longest_query, longest_candidate);
        size_t const myers_scratch = myers_t::match_masks_bytes_k + max_longer * (size_t)myers_t::lanes_k;
        size_t dp_scratch = 0, fourxN_scratch = 0;
        size_t const shortest_longest = sz_min_of_two(longest_query, longest_candidate);
        if (queries.size() && candidates.size() && shortest_longest > 64) {
            // The 4xN kernels keep a per-lane multi-word `match_masks` of `match_masks_bytes_k * words_bound`, where the word
            // bound is `ceil(shorter / 64)` for the largest shorter side any group can present (every shorter > 64).
            size_t const words_bound = divide_round_up(shortest_longest, (size_t)64);
            fourxN_scratch = myers_t::match_masks_bytes_k * words_bound;
        }
        if (queries.size() && candidates.size() && shortest_longest > 512) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(sz_max_of_two(myers_scratch, dp_scratch), fourxN_scratch);
    }

#pragma region Cross Product Cell Addressing

    /** @brief A destination for one scored cell: the primary matrix slot plus an optional mirror slot. */
    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    /**
     *  @brief An indexable adapter handed to the Myers kernels so they stay grouping-agnostic: a lane's group-local
     *      index selects its destination, and assigning a score writes the primary cell and, for symmetric
     *      self-similarity, the mirrored cell too.
     */
    template <typename value_type_>
    struct cross_cell_writer_ {
        cross_cell_destination_<value_type_> const *destinations = nullptr;

        struct cell_proxy_ {
            cross_cell_destination_<value_type_> destination;
            cell_proxy_ &operator=(size_t value) noexcept {
                *destination.primary = static_cast<value_type_>(value);
                if (destination.mirror) *destination.mirror = static_cast<value_type_>(value);
                return *this;
            }
        };

        cell_proxy_ operator[](size_t group_local_index) const noexcept {
            return cell_proxy_ {destinations[group_local_index]};
        }
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

#pragma endregion Cross Product Cell Addressing

#pragma region Cross Product Scoring

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the four-lane AVX2 Myers
     *      family, falling back to the anti-diagonal DP for a lone over-512 cell. Each cell scores
     *      `dist(query, candidate)` and writes it into the strided @p results matrix (plus the mirror slot for
     *      symmetric self-similarity). Only the unit-cost path reaches here; non-unit costs go through the per-pair
     *      serial fallback at the public-overload level.
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    SZ_NOINLINE status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                      results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                      size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        // `scratch` is provided by the caller, already sized to the worst single cell (`worst_cell_scratch_`).
        scoring_t dp {substituter_, gap_costs_};

        // Maps a query row and candidate column to their primary (and mirrored) destination slots.
        auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
            cross_cell_destination_<value_t> destination;
            destination.primary = results.data + query_index * results.row_stride + candidate_index;
            if (cross_kind == cross_similarities_t::symmetric_k && candidate_index != query_index)
                destination.mirror = results.data + candidate_index * results.row_stride + query_index;
            return destination;
        };

        myers_t myers;
        cross_cell_writer_<value_t> writer;
        dummy_executor_t dummy;
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const shorter = sz_min_of_two(query.size(), candidate.size());

            if (shorter == 0) {
                cross_cell_destination_<value_t> const destination = destination_for(query_index, candidate_index);
                cross_cell_writer_<value_t> {&destination}[0] = sz_max_of_two(query.size(), candidate.size());
                ++cell_index;
                continue;
            }

            // Single-word Myers: gather up to `lanes_k` consecutive live cells whose shorter side fits one 64-bit
            // Myers word. Per-lane `top_bits` handle the differing exact lengths inside the group.
            if (shorter <= 64) {
                span<char const> group_shorters[myers_t::lanes_k], group_longers[myers_t::lanes_k];
                size_t group_positions[myers_t::lanes_k];
                cross_cell_destination_<value_t> group_destinations[myers_t::lanes_k];
                bool const seed_query_shorter = query.size() <= candidate.size();
                group_shorters[0] = seed_query_shorter ? query : candidate;
                group_longers[0] = seed_query_shorter ? candidate : query;
                group_positions[0] = 0;
                group_destinations[0] = destination_for(query_index, candidate_index);
                index_t group = 1;
                ++cell_index;
                for (; cell_index != cell_end && group != (index_t)myers_t::lanes_k; ++cell_index, ++group) {
                    size_t next_query_index = 0, next_candidate_index = 0;
                    cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                    auto const next_query = to_view(queries[next_query_index]);
                    auto const next_candidate = to_view(candidates[next_candidate_index]);
                    size_t const next_shorter = sz_min_of_two(next_query.size(), next_candidate.size());
                    if (next_shorter == 0 || next_shorter > 64) break;
                    bool const next_query_shorter = next_query.size() <= next_candidate.size();
                    group_shorters[group] = next_query_shorter ? next_query : next_candidate;
                    group_longers[group] = next_query_shorter ? next_candidate : next_query;
                    group_positions[group] = group;
                    group_destinations[group] = destination_for(next_query_index, next_candidate_index);
                }

                writer.destinations = group_destinations;
                status_t const status = myers.distances_4x64_(
                    lane_pairs_view<char> {{group_shorters, group}, {group_longers, group}, {group_positions, group}},
                    writer, scratch);
                if (status != status_t::success_k) return status;
                continue;
            }

            // Batched multi-word Myers: gather up to `lanes_k` consecutive live cells whose shorter side is also > 64
            // and shares this seed cell's exact `ceil(shorter / 64)` word bucket, so the whole group loops one common
            // `words_count` and `distances_4x_multiword_<words_count>` can be selected at compile time.
            size_t const seed_bucket = divide_round_up(shorter, (size_t)64);
            span<char const> group_shorters[myers_t::lanes_k], group_longers[myers_t::lanes_k];
            size_t group_positions[myers_t::lanes_k];
            cross_cell_destination_<value_t> group_destinations[myers_t::lanes_k];
            bool const seed_query_shorter = query.size() <= candidate.size();
            group_shorters[0] = seed_query_shorter ? query : candidate;
            group_longers[0] = seed_query_shorter ? candidate : query;
            group_positions[0] = 0;
            group_destinations[0] = destination_for(query_index, candidate_index);
            index_t group = 1;
            ++cell_index;
            for (; cell_index != cell_end && group != (index_t)myers_t::lanes_k; ++cell_index, ++group) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                auto const next_query = to_view(queries[next_query_index]);
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                size_t const next_shorter = sz_min_of_two(next_query.size(), next_candidate.size());
                if (next_shorter <= 64 || divide_round_up(next_shorter, (size_t)64) != seed_bucket) break;
                bool const next_query_shorter = next_query.size() <= next_candidate.size();
                group_shorters[group] = next_query_shorter ? next_query : next_candidate;
                group_longers[group] = next_query_shorter ? next_candidate : next_query;
                group_positions[group] = group;
                group_destinations[group] = destination_for(next_query_index, next_candidate_index);
            }

            cross_cell_writer_<value_t> group_writer;
            group_writer.destinations = group_destinations;
            lane_pairs_view<char> const group_pairs {{group_shorters, group},
                                                     {group_longers, group},
                                                     {group_positions, group}};
            // A lone shorter > 512 cell keeps the single-pair anti-diagonal DP, avoiding a ragged regression.
            if (seed_bucket > 8 && group < 2) {
                size_t result_score = 0;
                if (status_t const lone_status = dp(group_shorters[0], group_longers[0], result_score, scratch, dummy,
                                                    specs);
                    lone_status != status_t::success_k)
                    return lone_status;
                cross_cell_writer_<value_t> {&group_destinations[0]}[0] = result_score;
                continue;
            }
            // Buckets 2..8 (shorter <= 512) hit the compile-time variant; longer groups take the runtime sibling.
            status_t status = dispatch_word_bucket_<2, 8>(
                seed_bucket,
                [&](auto bucket) {
                    return myers.template distances_4x_multiword_<bucket.value>(group_pairs, group_writer, scratch);
                },
                [&] { return myers.distances_4x_multiword_large_(group_pairs, group_writer, scratch); });
            if (status == status_t::success_k) continue;
            // Defensive scratch-shortfall fallback: score every grouped pair through the DP.
            for (index_t lane = 0; lane != group; ++lane) {
                size_t lane_score = 0;
                if (status_t const lane_status = dp(group_shorters[lane], group_longers[lane], lane_score, scratch,
                                                    dummy, specs);
                    lane_status != status_t::success_k)
                    return lane_status;
                cross_cell_writer_<value_t> {&group_destinations[lane]}[0] = lane_score;
            }
        }
        return status_t::success_k;
    }

    /** @brief Scores the cross-product in parallel: each worker takes a contiguous slice of the live-cell range. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    SZ_NOINLINE status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
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
            status_t status = score_range_(queries, candidates, results, cross_kind, cell_begin, cell_begin + length,
                                           slice, specs);
            if (status != status_t::success_k) error.store(status);
        });
        return error.load();
    }

#pragma endregion Cross Product Scoring

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case distance fits the narrow `u16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_u16_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case distance fits the wide `u32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_u32_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: an empty cell is a run of `gap` per remaining character. */
    auto empty_cell_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept -> ssize_t {
            return (ssize_t)gap_costs_.open_or_extend * (ssize_t)sz_max_of_two(query_length, candidate_length);
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_()) {
            lane_walker_narrow_t narrow {substituter_, gap_costs_};
            lane_walker_wide_t wide {substituter_, gap_costs_};
            scoring_t fallback {substituter_, gap_costs_};
            if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                    narrow, wide, fallback, queries, candidates, fits_wide_policy_(), specs));
                status != status_t::success_k)
                return status;
            return cross_product_candidate_lanes_range_(
                narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_),
                specs);
        }
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(queries, candidates, specs));
            status != status_t::success_k)
            return status;
        return score_range_(queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                            live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_()) {
            lane_walker_narrow_t narrow {substituter_, gap_costs_};
            lane_walker_wide_t wide {substituter_, gap_costs_};
            scoring_t fallback {substituter_, gap_costs_};
            return cross_product_candidate_lanes_parallel_(
                narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, score_scratch_,
                std::forward<executor_type_>(executor), fits_narrow_policy_(), fits_wide_policy_(),
                empty_cell_policy_(), specs);
        }
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_()) {
            lane_walker_narrow_t narrow {substituter_, gap_costs_};
            lane_walker_wide_t wide {substituter_, gap_costs_};
            scoring_t fallback {substituter_, gap_costs_};
            if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                    narrow, wide, fallback, sequences, sequences, fits_wide_policy_(), specs));
                status != status_t::success_k)
                return status;
            return cross_product_candidate_lanes_range_(
                narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_),
                specs);
        }
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(sequences, sequences, specs));
            status != status_t::success_k)
            return status;
        return score_range_(sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                            live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_()) {
            lane_walker_narrow_t narrow {substituter_, gap_costs_};
            lane_walker_wide_t wide {substituter_, gap_costs_};
            scoring_t fallback {substituter_, gap_costs_};
            return cross_product_candidate_lanes_parallel_(
                narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k,
                score_scratch_, std::forward<executor_type_>(executor), fits_narrow_policy_(), fits_wide_policy_(),
                empty_cell_policy_(), specs);
        }
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion Public Cross Product Overloads
};

/**
 *  @brief Batched byte-level @b affine-gap Levenshtein distances on Haswell, packing up to 16 candidates of a shared
 *      query into the unsigned-`u16` affine `candidate_lane_walker` and falling back to the per-pair anti-diagonal
 *      Dynamic Programming scorer for the long tail (distances that escape `u16`).
 *
 *  Affine sibling of the Haswell linear byte `levenshtein_distances` cross-product engine: identical query-major
 *  grouping, transpose, scatter, and mirror handling via the shared candidate-lane driver. There is no Myers fast
 *  path - the bit-parallel `+-1`-delta recurrence is linear-only - so @b every cost routes through the affine lane
 *  walker, and an empty `(query, candidate)` cell scores the single-gap-run `open + extend * (L - 1)`.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<
    affine_gap_costs_t, allocator_type_, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;    // ? `u16` lanes for the affine candidate-lane walker.
    static constexpr size_t u16_reach_limit_k = 50000; // ? `u16` headroom below the lane walker's discard bias.

    using scoring_t = levenshtein_distance<char, affine_gap_costs_t, sz_cap_serial_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, affine_gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, (int)candidate_lanes_k,
                              void>; // ? AVX2 affine shared-query `u16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, u32_t, uniform_substitution_costs_t, affine_gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, 8,
                              void>; // ? AVX2 affine shared-query `u32` lanes.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    uniform_substitution_costs_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    levenshtein_distances(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances(uniform_substitution_costs_t subs, affine_gap_costs_t gaps,
                          allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief Whether a `(query, candidate)` cell's worst-case distance stays inside the narrow walker's `u16` headroom. */
    bool fits_u16_(size_t query_length, size_t candidate_length) const noexcept {
        return (query_length + candidate_length) *
                       sz_max_of_two(sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open),
                                     (size_t)gap_costs_.extend) +
                   (size_t)gap_costs_.open <=
               u16_reach_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case distance stays inside the wide walker's `u32` headroom. */
    bool fits_u32_(size_t query_length, size_t candidate_length) const noexcept {
        return (query_length + candidate_length) *
                       sz_max_of_two(sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open),
                                     (size_t)gap_costs_.extend) +
                   (size_t)gap_costs_.open <=
               1500000000;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case distance fits the narrow `u16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_u16_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case distance fits the wide `u32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_u32_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: an affine empty cell is one open plus extensions. */
    auto empty_cell_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept -> ssize_t {
            size_t const other = sz_max_of_two(query_length, candidate_length);
            return other == 0 ? 0 : (ssize_t)((size_t)gap_costs_.open + (size_t)gap_costs_.extend * (other - 1));
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, queries, candidates, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, queries, candidates, results,
                                                       cross_similarities_t::all_pairs_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, sequences, sequences, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, sequences, sequences, results,
                                                       cross_similarities_t::symmetric_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

#pragma endregion Public Cross Product Overloads
};

/**
 *  @brief Batched @b rune-level (UTF-8) unit-cost Levenshtein distances on Haswell, scoring four unit-cost rune pairs
 *      at once with the @b AVX2 four-lane rune Myers family - `distances_4x64_` (shorter <= 64 runes),
 *      `distances_4x_multiword_<words_count>` (compile-time word count, shorter <= 512 runes), and
 *      `distances_4x_multiword_large_` (runtime word count, shorter > 512 runes). Cells are transcoded to runes and
 *      grouped by their exact `ceil(shorter / 64)` word bucket so each group loops one common word count, the four-lane
 *      twin of the Ice Lake eight-lane rune Myers UTF-8 engine. Empty cells, invalid UTF-8, and lone shorter > 4096
 *      cells fall back to the per-pair rune scorer.
 *
 *  Structural mirror of the Ice Lake `levenshtein_distances_utf8` cross-product engine - live cells walked query-major,
 *  transcoded into a reusable rune arena, grouped into same-bucket Myers calls, scored, and scattered into the strided
 *  result matrix (plus the mirrored slot for symmetric self-similarity) - narrowed to the four-lane AVX2 rune Myers.
 *  Only unit-cost @b linear gaps take the Myers fast path; non-unit or affine UTF-8 keeps the per-pair path (the
 *  primary `levenshtein_distances_utf8` template via the per-pair `levenshtein_distance_utf8` scorer).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances_utf8<
    linear_gap_costs_t, allocator_type_, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;

    using scoring_t = levenshtein_distance_utf8<gap_costs_t, sz_cap_serial_k>; // ? Per-pair UTF-8 serial fallback.
    using myers_t = levenshtein_distance_myers<rune_t, capability_k>;          // ? AVX2 four-lane rune Myers fast path.
    using lane_walker_t =
        candidate_lane_walker<rune_t, u16_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, (int)candidate_lanes_k, void>;
    // The non-unit cost path routes the transcoded rune views through the shared width-tiered candidate-lane driver:
    // a 16-lane `u16` narrow walker plus an 8-lane `u32` wide walker, with the per-pair rune DP as the long-tail
    // fallback. Unit-cost linear stays on the rune Myers fast path (`score_range_`), untouched.
    using lane_walker_narrow_t = lane_walker_t; // ? AVX2 16-lane `u16` non-unit rune shared query.
    using lane_walker_wide_t =
        candidate_lane_walker<rune_t, u32_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, 8, void>; // ? 8-lane `u32` non-unit rune.
    // The driver's per-pair fallback receives @b rune views, so it is a rune-typed `levenshtein_distance`; the serial
    // capability covers every cell width (the Haswell rune diagonal walker only goes up to `u16`), and this long-tail
    // path is rare. It stays bit-exact with the serial reference oracle.
    using rune_scoring_t = levenshtein_distance<rune_t, gap_costs_t, sz_cap_serial_k>; // ? Per-pair rune DP fallback.
    static constexpr index_t myers_lanes_k = myers_t::lanes_k;
    static constexpr size_t u16_reach_limit_k = 60000; // ? `u16` headroom for the non-unit rune lane walker.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    using rune_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<rune_t>;
    using rune_view_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<span<rune_t const>>;

    uniform_substitution_costs_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker
    // The non-unit path transcodes every query/candidate to UTF-32 once and exposes each as a `span<rune_t const>`
    // view, so the driver's `to_view` yields rune spans. Queries and candidates own @b separate arenas so the second
    // transcode does not invalidate the first set of views; the symmetric self-similarity case reuses the query arena.
    safe_vector<rune_t, rune_allocator_t> query_arena_ {alloc_};
    safe_vector<rune_t, rune_allocator_t> candidate_arena_ {alloc_};
    safe_vector<span<rune_t const>, rune_view_allocator_t> query_runes_ {alloc_};
    safe_vector<span<rune_t const>, rune_view_allocator_t> candidate_runes_ {alloc_};

    levenshtein_distances_utf8(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances_utf8(uniform_substitution_costs_t subs, linear_gap_costs_t gaps,
                               allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief Whether the substitution/gap costs are the unit-cost edit distance the rune Myers fast path assumes. */
    bool is_unit_cost_() const noexcept {
        return substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1;
    }

    /** @brief Whether a `(query, candidate)` rune cell's worst-case distance fits the narrow `u16` walker's headroom. */
    bool fits_u16_(size_t query_runes, size_t candidate_runes) const noexcept {
        size_t const magnitude = sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open_or_extend);
        return (query_runes + candidate_runes) * magnitude <= u16_reach_limit_k;
    }

    /** @brief Whether a `(query, candidate)` rune cell's worst-case distance fits the wide `u32` walker's headroom. */
    bool fits_u32_(size_t query_runes, size_t candidate_runes) const noexcept {
        size_t const magnitude = sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open_or_extend);
        return (query_runes + candidate_runes) * magnitude <= 1500000000;
    }

    /**
     *  @brief Transcodes every UTF-8 sequence in @p sequences to UTF-32 runes appended to @p arena, recording each
     *      sequence's rune span in @p views. The spans point into @p arena, so @p arena must outlive the driver call.
     *      Returns `false` on invalid UTF-8 (the caller then routes the whole call through the per-pair UTF-8 serial
     *      fallback, which scores invalid bytes directly), or on allocation failure.
     */
    template <typename sequences_type_>
    bool transcode_views_(sequences_type_ const &sequences, safe_vector<rune_t, rune_allocator_t> &arena,
                          safe_vector<span<rune_t const>, rune_view_allocator_t> &views) const noexcept {
        size_t total_bytes = 0;
        for (size_t index = 0; index < sequences.size(); ++index) total_bytes += to_view(sequences[index]).size();
        // A rune occupies at least one UTF-8 byte, so the byte total bounds the rune count.
        if (arena.try_reserve(total_bytes) != status_t::success_k) return false;
        if (arena.try_resize(0) != status_t::success_k) return false;
        if (views.try_resize(sequences.size()) != status_t::success_k) return false;
        // The arena is reserved to its full size up front, so it never reallocates inside the loop and every span
        // recorded into it stays valid for the whole driver call.
        for (size_t index = 0; index < sequences.size(); ++index) {
            auto const bytes = to_view(sequences[index]);
            size_t const rune_begin = arena.size();
            rune_length_t rune_length;
            for (size_t progress = 0; progress < bytes.size(); progress += rune_length) {
                rune_t rune;
                rune_length = sz_rune_decode_unchecked(bytes.data() + progress, &rune);
                if (rune_length == sz_rune_invalid_k) return false;
                if (arena.try_resize(arena.size() + 1) != status_t::success_k) return false;
                arena[arena.size() - 1] = rune;
            }
            views[index] = span<rune_t const> {arena.data() + rune_begin, arena.size() - rune_begin};
        }
        return true;
    }

    /** @brief Byte size of the cache-line-aligned UTF-32 transcode buffer for a string of @p byte_length bytes. */
    size_t query_runes_bytes_(size_t byte_length, cpu_specs_t const &specs) const noexcept {
        return round_up_to_multiple(sizeof(rune_t) * byte_length, specs.cache_line_width);
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the query-rune buffer plus the
     *      transposed rune block plus the lane-walker arena for the longest sides, or the per-pair rune DP fallback
     *      for the longest query × longest candidate (a real cell in both grids, so a safe upper bound per slice).
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
        size_t const query_rune_bytes = query_runes_bytes_(longest_query, specs);
        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(rune_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs)
                                                        : 0;
        // The rune-Myers fast path transcodes up to `myers_lanes_k` pairs' runes into one buffer (a rune is at least
        // one byte, so byte length bounds rune count) and builds the 4-lane hash `match_masks` for the longest shorter side.
        size_t const myers_transcode_bytes = round_up_to_multiple(
            (size_t)myers_lanes_k * (longest_query + longest_candidate) * sizeof(rune_t), specs.cache_line_width);
        size_t const myers_match_masks_bytes = myers_t::scratch_bytes_for(
            sz_min_of_two(longest_query, longest_candidate));
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size()) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        size_t const lane_walker_path = query_rune_bytes + transpose_bytes + walker_scratch;
        size_t const myers_path = myers_transcode_bytes + myers_match_masks_bytes;
        return sz_max_of_two(sz_max_of_two(lane_walker_path, myers_path), dp_scratch);
    }

#pragma region Cross Product Cell Addressing

    /**
     *  @brief A destination for one scored cell: the primary matrix slot plus an optional mirror slot. The lane
     *      walker writes one score per lane, and the scatter fans it out to both slots on assignment.
     */
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

#pragma endregion Cross Product Cell Addressing

#pragma region Cross Product Scoring

    /**
     *  @brief Transcodes a UTF-8 string into a contiguous UTF-32 rune buffer, returning the rune count (or
     *      `SZ_SIZE_MAX` on invalid UTF-8). Mirrors the per-pair UTF-8 engine's export loop.
     */
    static size_t transcode_runes_(span<char_t const> utf8, rune_t *runes_out) noexcept {
        rune_length_t rune_length;
        size_t rune_count = 0;
        for (size_t progress_utf8 = 0; progress_utf8 < utf8.size(); progress_utf8 += rune_length, ++rune_count) {
            rune_length = sz_rune_decode_unchecked(utf8.data() + progress_utf8, runes_out + rune_count);
            if (rune_length == sz_rune_invalid_k) return SZ_SIZE_MAX;
        }
        return rune_count;
    }

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the 4-lane rune Myers - the
     *      unit-cost-linear UTF-8 fast path. Each cell is transcoded to runes, and consecutive cells whose shorter
     *      rune side shares the same `ceil(shorter / 64)` word bucket are batched into one rune-Myers call (single
     *      word for shorter <= 64, the compile-time `distances_4x_multiword_<W>` for buckets 2..8, the runtime sibling
     *      beyond), exactly the byte Myers driver's grouping. Empty cells, invalid UTF-8, and lone shorter > 4096 cells
     *      fall back to the per-pair rune scorer. Writes each distance into the strided @p results matrix (plus the
     *      mirror slot for symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    SZ_NOINLINE status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                      results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                      size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        // Carve the slice's worst transcode area off the front of scratch (a rune is >= 1 byte, so byte length bounds
        // rune count) and leave the tail for the per-call 4-lane Myers `match_masks`. The global-worst sizing from
        // `worst_cell_scratch_` guarantees this split stays in bounds for any slice.
        size_t longest_query = 0, longest_candidate = 0;
        for (size_t cell_index = cell_begin; cell_index != cell_end; ++cell_index) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            longest_query = sz_max_of_two(longest_query, to_view(queries[query_index]).size());
            longest_candidate = sz_max_of_two(longest_candidate, to_view(candidates[candidate_index]).size());
        }
        size_t const transcode_bytes = round_up_to_multiple(
            (size_t)myers_lanes_k * (longest_query + longest_candidate) * sizeof(rune_t), specs.cache_line_width);
        rune_t *const rune_arena = reinterpret_cast<rune_t *>(scratch.data());
        size_t const rune_arena_runes = transcode_bytes / sizeof(rune_t);
        scratch_space_t const match_masks_scratch = transcode_bytes <= scratch.size()
                                                        ? scratch.subspan(transcode_bytes,
                                                                          scratch.size() - transcode_bytes)
                                                        : scratch_space_t {};
        scratch_space_t const dp_scratch_space = scratch;

        scoring_t dp {substituter_, gap_costs_};
        dummy_executor_t dummy;

        auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
            cross_cell_destination_<value_t> destination;
            destination.primary = results.data + query_index * results.row_stride + candidate_index;
            if (cross_kind == cross_similarities_t::symmetric_k && candidate_index != query_index)
                destination.mirror = results.data + candidate_index * results.row_stride + query_index;
            return destination;
        };
        auto const scatter = [&](cross_cell_destination_<value_t> const &destination, size_t score) noexcept {
            *destination.primary = static_cast<value_t>(score);
            if (destination.mirror) *destination.mirror = static_cast<value_t>(score);
        };

        // The cross-cell writer the rune-Myers kernels assign through: lane-local index -> destination slot(s).
        struct cross_cell_writer_ {
            cross_cell_destination_<value_t> const *destinations = nullptr;
            struct cell_proxy_ {
                cross_cell_destination_<value_t> destination;
                cell_proxy_ &operator=(size_t value) noexcept {
                    *destination.primary = static_cast<value_t>(value);
                    if (destination.mirror) *destination.mirror = static_cast<value_t>(value);
                    return *this;
                }
            };
            cell_proxy_ operator[](size_t lane_index) const noexcept { return cell_proxy_ {destinations[lane_index]}; }
        };

        myers_t myers;

        // Transcode one cell's shorter/longer sides into the rune arena at byte offset `arena_used` (in runes),
        // returning false on invalid UTF-8 or arena overflow. The shorter rune side is the Myers pattern.
        auto const transcode_cell = [&](span<char_t const> query, span<char_t const> candidate, size_t &arena_used,
                                        span<rune_t const> &shorter_runes,
                                        span<rune_t const> &longer_runes) noexcept -> bool {
            size_t const query_offset = arena_used;
            size_t query_runes_count = 0;
            rune_length_t rune_length;
            for (size_t progress = 0; progress < query.size(); progress += rune_length, ++query_runes_count) {
                if (query_offset + query_runes_count >= rune_arena_runes) return false;
                rune_length = sz_rune_decode_unchecked(query.data() + progress,
                                                       rune_arena + query_offset + query_runes_count);
                if (rune_length == sz_rune_invalid_k) return false;
            }
            size_t const candidate_offset = query_offset + query_runes_count;
            size_t candidate_runes_count = 0;
            for (size_t progress = 0; progress < candidate.size(); progress += rune_length, ++candidate_runes_count) {
                if (candidate_offset + candidate_runes_count >= rune_arena_runes) return false;
                rune_length = sz_rune_decode_unchecked(candidate.data() + progress,
                                                       rune_arena + candidate_offset + candidate_runes_count);
                if (rune_length == sz_rune_invalid_k) return false;
            }
            arena_used = candidate_offset + candidate_runes_count;
            span<rune_t const> const query_view {rune_arena + query_offset, query_runes_count};
            span<rune_t const> const candidate_view {rune_arena + candidate_offset, candidate_runes_count};
            bool const query_is_shorter = query_runes_count <= candidate_runes_count;
            shorter_runes = query_is_shorter ? query_view : candidate_view;
            longer_runes = query_is_shorter ? candidate_view : query_view;
            return true;
        };

        static constexpr size_t stack_words_capacity_k =
            64; // ? `distances_4x_multiword_large_` covers shorter <= 4096 runes.
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);

            // Empty cell, or a side whose byte length might escape the rune Myers stack capacity, takes the per-pair
            // rune scorer. An empty side makes the distance the other side's rune count, which the per-pair scorer
            // returns; the byte length is a safe conservative gate for the > 4096-rune long tail (rune >= 1 byte).
            bool const fits_myers = query.size() != 0 && candidate.size() != 0 &&
                                    sz_min_of_two(query.size(), candidate.size()) <= stack_words_capacity_k * 64;
            if (!fits_myers || !match_masks_scratch.size()) {
                size_t result_distance = 0;
                if (status_t status = dp(query, candidate, result_distance, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_distance);
                ++cell_index;
                continue;
            }

            // Seed the group with this cell. Transcoding determines the exact shorter rune count and thus the word
            // bucket every grouped lane must share.
            size_t arena_used = 0;
            span<rune_t const> group_shorters[myers_lanes_k], group_longers[myers_lanes_k];
            size_t group_positions[myers_lanes_k];
            cross_cell_destination_<value_t> group_destinations[myers_lanes_k];
            size_t group_query_indices[myers_lanes_k], group_candidate_indices[myers_lanes_k];
            span<rune_t const> seed_shorter, seed_longer;
            if (!transcode_cell(query, candidate, arena_used, seed_shorter, seed_longer)) {
                size_t result_distance = 0;
                if (status_t status = dp(query, candidate, result_distance, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_distance);
                ++cell_index;
                continue;
            }
            size_t const seed_shorter_runes = seed_shorter.size();
            // A lone empty-after-transcode shorter side cannot seed Myers (the top-bit read underflows); the per-pair
            // scorer handles it (its distance is the longer side's rune count).
            if (seed_shorter_runes == 0) {
                size_t result_distance = 0;
                if (status_t status = dp(query, candidate, result_distance, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_distance);
                ++cell_index;
                continue;
            }
            size_t const seed_bucket = divide_round_up<size_t>(seed_shorter_runes, 64);
            group_shorters[0] = seed_shorter;
            group_longers[0] = seed_longer;
            group_positions[0] = 0;
            group_destinations[0] = destination_for(query_index, candidate_index);
            group_query_indices[0] = query_index;
            group_candidate_indices[0] = candidate_index;
            index_t group = 1;
            ++cell_index;
            for (; cell_index != cell_end && group != myers_lanes_k; ++cell_index) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                auto const next_query = to_view(queries[next_query_index]);
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                if (next_query.size() == 0 || next_candidate.size() == 0 ||
                    sz_min_of_two(next_query.size(), next_candidate.size()) > stack_words_capacity_k * 64)
                    break;
                size_t const arena_before = arena_used;
                span<rune_t const> next_shorter, next_longer;
                if (!transcode_cell(next_query, next_candidate, arena_used, next_shorter, next_longer)) {
                    arena_used = arena_before; // ? Invalid UTF-8 or arena full: do not consume this cell here.
                    break;
                }
                if (next_shorter.size() == 0 || divide_round_up<size_t>(next_shorter.size(), 64) != seed_bucket) {
                    arena_used = arena_before; // ? Different bucket / empty pattern: leave it for the next group.
                    break;
                }
                group_shorters[group] = next_shorter;
                group_longers[group] = next_longer;
                group_positions[group] = group;
                group_destinations[group] = destination_for(next_query_index, next_candidate_index);
                group_query_indices[group] = next_query_index;
                group_candidate_indices[group] = next_candidate_index;
                ++group;
            }

            cross_cell_writer_ group_writer;
            group_writer.destinations = group_destinations;
            lane_pairs_view<rune_t> const group_pairs {{group_shorters, group},
                                                       {group_longers, group},
                                                       {group_positions, group}};
            // Bucket 1 (shorter <= 64 runes) takes the single-word kernel, 2..8 the compile-time multiword variant.
            status_t status = dispatch_word_bucket_<1, 8>(
                seed_bucket,
                [&](auto bucket) {
                    if constexpr (bucket.value == 1)
                        return myers.distances_4x64_(group_pairs, group_writer, match_masks_scratch);
                    else
                        return myers.template distances_4x_multiword_<bucket.value>(group_pairs, group_writer,
                                                                                    match_masks_scratch);
                },
                [&] { return myers.distances_4x_multiword_large_(group_pairs, group_writer, match_masks_scratch); });
            if (status == status_t::success_k) continue;
            // Defensive scratch-shortfall fallback: score every grouped pair through the per-pair UTF-8 rune DP over
            // its original cell (the `dp` scorer owns its own scratch via `dp_scratch_space`). Mirrors the byte
            // Myers driver's group fallback; `worst_cell_scratch_` sizes the `match_masks` so this should not trigger.
            for (index_t lane = 0; lane != group; ++lane) {
                size_t lane_score = 0;
                auto const lane_query = to_view(queries[group_query_indices[lane]]);
                auto const lane_candidate = to_view(candidates[group_candidate_indices[lane]]);
                if (status_t lane_status = dp(lane_query, lane_candidate, lane_score, dp_scratch_space, dummy, specs);
                    lane_status != status_t::success_k)
                    return lane_status;
                scatter(group_destinations[lane], lane_score);
            }
        }
        return status_t::success_k;
    }

    /** @brief Scores the cross-product in parallel: each worker takes a contiguous slice of the live-cell range. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    SZ_NOINLINE status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                         results_type_ &&results, cross_similarities_t cross_kind,
                                         executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        // One hoisted buffer carved into a per-worker slice; `for_slices` invokes the body at most `threads_count`
        // times, and the atomic counter hands each invocation a disjoint slice (no aliasing, no per-worker alloc).
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
            status_t status = score_range_(queries, candidates, results, cross_kind, cell_begin, cell_begin + length,
                                           slice, specs);
            if (status != status_t::success_k) error.store(status);
        });
        return error.load();
    }

#pragma endregion Cross Product Scoring

#pragma region Non Unit Cross Product via Rune Lane Driver

    /** @brief `(query_runes, candidate_runes) -> bool`: whether a cell fits the narrow `u16` rune walker's range. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_runes, size_t candidate_runes) noexcept {
            return fits_u16_(query_runes, candidate_runes);
        };
    }

    /** @brief `(query_runes, candidate_runes) -> bool`: whether a cell fits the wide `u32` rune walker's range. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_runes, size_t candidate_runes) noexcept {
            return fits_u32_(query_runes, candidate_runes);
        };
    }

    /** @brief `(query_runes, candidate_runes) -> score`: an empty cell is a run of `gap` per rune of the longer. */
    auto empty_cell_policy_() const noexcept {
        return [this](size_t query_runes, size_t candidate_runes) noexcept -> ssize_t {
            return (ssize_t)gap_costs_.open_or_extend * (ssize_t)sz_max_of_two(query_runes, candidate_runes);
        };
    }

    /**
     *  @brief Non-unit serial cross-product: transcode to runes once, then run the shared candidate-lane driver
     *      over the rune views. On invalid UTF-8 (or transcode alloc failure) routes the whole call through the
     *      per-pair UTF-8 serial scorer, which handles invalid bytes directly.
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    status_t cross_via_lanes_(queries_type_ const &queries, candidates_type_ const &candidates, results_type_ &&results,
                              cross_similarities_t cross_kind, cpu_specs_t const &specs) noexcept {
        bool const same = static_cast<void const *>(&queries) == static_cast<void const *>(&candidates);
        if (!transcode_views_(queries, query_arena_, query_runes_) ||
            (!same && !transcode_views_(candidates, candidate_arena_, candidate_runes_)))
            return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                               cross_kind, score_scratch_, specs);
        // Symmetric self-similarity scores one set against itself: reuse the query arena/views as the candidate side.
        auto const &candidate_views = same ? query_runes_ : candidate_runes_;

        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        rune_scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, query_runes_, candidate_views, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, query_runes_, candidate_views, results, cross_kind, 0,
            cross_live_cells_count_(query_runes_.size(), candidate_views.size(), cross_kind), fits_narrow_policy_(),
            fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    /**
     *  @brief Non-unit parallel cross-product: transcode to runes once, then run the parallel candidate-lane driver
     *      over the rune views. Falls back to the per-pair UTF-8 serial scorer on invalid UTF-8 / alloc failure.
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    status_t cross_via_lanes_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                       results_type_ &&results, cross_similarities_t cross_kind,
                                       executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        bool const same = static_cast<void const *>(&queries) == static_cast<void const *>(&candidates);
        if (!transcode_views_(queries, query_arena_, query_runes_) ||
            (!same && !transcode_views_(candidates, candidate_arena_, candidate_runes_)))
            return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                              cross_kind, score_scratch_, std::forward<executor_type_>(executor),
                                              specs);
        auto const &candidate_views = same ? query_runes_ : candidate_runes_;

        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        rune_scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, query_runes_, candidate_views, results,
                                                       cross_kind, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

#pragma endregion Non Unit Cross Product via Rune Lane Driver

#pragma region Public Cross Product Overloads

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_via_lanes_(queries, candidates, results, cross_similarities_t::all_pairs_k, specs);
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(queries, candidates, specs));
            status != status_t::success_k)
            return status;
        return score_range_(queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                            live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_via_lanes_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                                             std::forward<executor_type_>(executor), specs);
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_via_lanes_(sequences, sequences, results, cross_similarities_t::symmetric_k, specs);
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(sequences, sequences, specs));
            status != status_t::success_k)
            return status;
        return score_range_(sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                            live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_via_lanes_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                                             std::forward<executor_type_>(executor), specs);
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion Public Cross Product Overloads
};

/**
 *  @brief Batched @b rune-level @b affine-gap Levenshtein distances on Haswell, transcoding UTF-8 to UTF-32 once and
 *      packing up to 16 candidates of a shared query into the unsigned-`u16` affine rune `candidate_lane_walker` (or
 *      the 8-lane `u32` sibling), falling back to the per-pair rune anti-diagonal Dynamic Programming scorer for the
 *      long tail (rune lengths that escape the `u32` cell range, or candidates that cannot fill a lane block).
 *
 *  Affine sibling of the Haswell linear UTF-8 `levenshtein_distances_utf8` cross-product engine. There is no Myers
 *  fast path - the bit-parallel `+-1`-delta recurrence is linear-only - so @b every cost routes through the affine
 *  rune lane driver, and an empty `(query, candidate)` cell scores the single-gap-run `open + extend * (L - 1)`. On
 *  invalid UTF-8 (or transcode alloc failure) the whole call routes through the per-pair UTF-8 serial scorer, which
 *  scores invalid bytes directly.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances_utf8<
    affine_gap_costs_t, allocator_type_, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;    // ? `u16` lanes for the affine rune candidate-lane walker.
    static constexpr size_t u16_reach_limit_k = 50000; // ? `u16` headroom below the affine walker's discard bias.

    using scoring_t = levenshtein_distance_utf8<gap_costs_t, sz_cap_serial_k>; // ? Per-pair UTF-8 serial fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<rune_t, u16_t, uniform_substitution_costs_t, affine_gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, (int)candidate_lanes_k,
                              void>; // ? AVX2 16-lane `u16` affine rune shared query.
    using lane_walker_wide_t =
        candidate_lane_walker<rune_t, u32_t, uniform_substitution_costs_t, affine_gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_haswell_k, 8, void>;      // ? 8-lane `u32` affine rune.
    using rune_scoring_t = levenshtein_distance<rune_t, gap_costs_t, sz_cap_serial_k>; // ? Per-pair rune DP fallback.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    using rune_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<rune_t>;
    using rune_view_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<span<rune_t const>>;

    uniform_substitution_costs_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker
    // Queries and candidates own separate rune arenas so the second transcode does not invalidate the first set of
    // views; the symmetric self-similarity case reuses the query arena.
    safe_vector<rune_t, rune_allocator_t> query_arena_ {alloc_};
    safe_vector<rune_t, rune_allocator_t> candidate_arena_ {alloc_};
    safe_vector<span<rune_t const>, rune_view_allocator_t> query_runes_ {alloc_};
    safe_vector<span<rune_t const>, rune_view_allocator_t> candidate_runes_ {alloc_};

    levenshtein_distances_utf8(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances_utf8(uniform_substitution_costs_t subs, affine_gap_costs_t gaps,
                               allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief Whether a `(query, candidate)` rune cell's worst-case distance fits the narrow `u16` walker's headroom. */
    bool fits_u16_(size_t query_runes, size_t candidate_runes) const noexcept {
        return (query_runes + candidate_runes) *
                       sz_max_of_two(sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open),
                                     (size_t)gap_costs_.extend) +
                   (size_t)gap_costs_.open <=
               u16_reach_limit_k;
    }

    /** @brief Whether a `(query, candidate)` rune cell's worst-case distance fits the wide `u32` walker's headroom. */
    bool fits_u32_(size_t query_runes, size_t candidate_runes) const noexcept {
        return (query_runes + candidate_runes) *
                       sz_max_of_two(sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open),
                                     (size_t)gap_costs_.extend) +
                   (size_t)gap_costs_.open <=
               1500000000;
    }

    /**
     *  @brief Transcodes every UTF-8 sequence in @p sequences to UTF-32 runes appended to @p arena, recording each
     *      sequence's rune span in @p views (pointing into @p arena). Returns `false` on invalid UTF-8 / alloc failure.
     */
    template <typename sequences_type_>
    bool transcode_views_(sequences_type_ const &sequences, safe_vector<rune_t, rune_allocator_t> &arena,
                          safe_vector<span<rune_t const>, rune_view_allocator_t> &views) const noexcept {
        size_t total_bytes = 0;
        for (size_t index = 0; index < sequences.size(); ++index) total_bytes += to_view(sequences[index]).size();
        if (arena.try_reserve(total_bytes) != status_t::success_k) return false;
        if (arena.try_resize(0) != status_t::success_k) return false;
        if (views.try_resize(sequences.size()) != status_t::success_k) return false;
        for (size_t index = 0; index < sequences.size(); ++index) {
            auto const bytes = to_view(sequences[index]);
            size_t const rune_begin = arena.size();
            rune_length_t rune_length;
            for (size_t progress = 0; progress < bytes.size(); progress += rune_length) {
                rune_t rune;
                rune_length = sz_rune_decode_unchecked(bytes.data() + progress, &rune);
                if (rune_length == sz_rune_invalid_k) return false;
                if (arena.try_resize(arena.size() + 1) != status_t::success_k) return false;
                arena[arena.size() - 1] = rune;
            }
            views[index] = span<rune_t const> {arena.data() + rune_begin, arena.size() - rune_begin};
        }
        return true;
    }

#pragma region Cross Product Policies

    /** @brief `(query_runes, candidate_runes) -> bool`: whether a cell fits the narrow `u16` rune walker's range. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_runes, size_t candidate_runes) noexcept {
            return fits_u16_(query_runes, candidate_runes);
        };
    }

    /** @brief `(query_runes, candidate_runes) -> bool`: whether a cell fits the wide `u32` rune walker's range. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_runes, size_t candidate_runes) noexcept {
            return fits_u32_(query_runes, candidate_runes);
        };
    }

    /** @brief `(query_runes, candidate_runes) -> score`: an affine empty cell is one open plus extensions. */
    auto empty_cell_policy_() const noexcept {
        return [this](size_t query_runes, size_t candidate_runes) noexcept -> ssize_t {
            size_t const other = sz_max_of_two(query_runes, candidate_runes);
            return other == 0 ? 0 : (ssize_t)((size_t)gap_costs_.open + (size_t)gap_costs_.extend * (other - 1));
        };
    }

    /** @brief Serial cross-product: transcode to runes once, then run the shared candidate-lane driver. */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    status_t cross_via_lanes_(queries_type_ const &queries, candidates_type_ const &candidates, results_type_ &&results,
                              cross_similarities_t cross_kind, cpu_specs_t const &specs) noexcept {
        bool const same = static_cast<void const *>(&queries) == static_cast<void const *>(&candidates);
        if (!transcode_views_(queries, query_arena_, query_runes_) ||
            (!same && !transcode_views_(candidates, candidate_arena_, candidate_runes_)))
            return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                               cross_kind, score_scratch_, specs);
        // Symmetric self-similarity scores one set against itself: reuse the query arena/views as the candidate side.
        auto const &candidate_views = same ? query_runes_ : candidate_runes_;

        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        rune_scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, query_runes_, candidate_views, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, query_runes_, candidate_views, results, cross_kind, 0,
            cross_live_cells_count_(query_runes_.size(), candidate_views.size(), cross_kind), fits_narrow_policy_(),
            fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    /** @brief Parallel cross-product: transcode to runes once, then run the parallel candidate-lane driver. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    status_t cross_via_lanes_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                       results_type_ &&results, cross_similarities_t cross_kind,
                                       executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        bool const same = static_cast<void const *>(&queries) == static_cast<void const *>(&candidates);
        if (!transcode_views_(queries, query_arena_, query_runes_) ||
            (!same && !transcode_views_(candidates, candidate_arena_, candidate_runes_)))
            return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                              cross_kind, score_scratch_, std::forward<executor_type_>(executor),
                                              specs);
        auto const &candidate_views = same ? query_runes_ : candidate_runes_;

        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        rune_scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, query_runes_, candidate_views, results,
                                                       cross_kind, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

#pragma endregion Cross Product Policies

#pragma region Public Cross Product Overloads

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        return cross_via_lanes_(queries, candidates, results, cross_similarities_t::all_pairs_k, specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_via_lanes_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                                         std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_via_lanes_(sequences, sequences, results, cross_similarities_t::symmetric_k, specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return cross_via_lanes_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                                         std::forward<executor_type_>(executor), specs);
    }

#pragma endregion Public Cross Product Overloads
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
struct needleman_wunsch_scores<
    error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, sz_caps_sh_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_haswell_k, (int)candidate_lanes_k, void>; // ? AVX2 shared-query `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_haswell_k, 8, void>; // ? AVX2 shared-query `i32` lanes.

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

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the narrow walker's `i16` headroom. */
    bool fits_lane_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the wide walker's `i32` headroom. */
    bool fits_i32_range_(size_t query_length, size_t candidate_length) const noexcept {
        return (ssize_t)(query_length + candidate_length) * (ssize_t)cost_magnitude_() <= 2000000000;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the narrow `i16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_lane_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the wide `i32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i32_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: a global linear-gap empty cell is a run of `gap` per char. */
    auto empty_cell_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept -> ssize_t {
            return (ssize_t)gap_costs_.open_or_extend * (ssize_t)sz_max_of_two(query_length, candidate_length);
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, queries, candidates, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, queries, candidates, results,
                                                       cross_similarities_t::all_pairs_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, sequences, sequences, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, sequences, sequences, results,
                                                       cross_similarities_t::symmetric_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

#pragma endregion Public Cross Product Overloads
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
struct smith_waterman_scores<
    error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, sz_caps_sh_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_haswell_k, (int)candidate_lanes_k, void>; // ? AVX2 shared-query local `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_haswell_k, 8, void>; // ? AVX2 shared-query local `i32` lanes.

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

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the narrow walker's `i16` headroom. */
    bool fits_lane_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the wide walker's `i32` headroom. */
    bool fits_i32_range_(size_t query_length, size_t candidate_length) const noexcept {
        return (ssize_t)(query_length + candidate_length) * (ssize_t)cost_magnitude_() <= 2000000000;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the narrow `i16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_lane_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the wide `i32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i32_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: a local alignment may align nothing, so an empty cell is 0. */
    auto empty_cell_policy_() const noexcept {
        return []([[maybe_unused]] size_t query_length, [[maybe_unused]] size_t candidate_length) noexcept -> ssize_t {
            return 0;
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, queries, candidates, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, queries, candidates, results,
                                                       cross_similarities_t::all_pairs_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, sequences, sequences, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, sequences, sequences, results,
                                                       cross_similarities_t::symmetric_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

#pragma endregion Public Cross Product Overloads
};

/**
 *  @brief Batched byte-level @b weighted @b affine-gap Needleman-Wunsch scores on Haswell, packing up to 16 candidates
 *      of a shared query into the signed-`i16` affine `candidate_lane_walker` and falling back to the per-pair
 *      anti-diagonal Dynamic Programming scorer for the long tail (scores that escape `i16`).
 *
 *  Affine sibling of the Haswell linear `needleman_wunsch_scores` cross-product engine: identical query-major
 *  grouping, transpose, scatter, and mirror handling, but it dispatches the affine (Gotoh E/F) lane walker and the
 *  affine per-pair scorer, and an empty `(query, candidate)` cell scores the single-gap-run `open + extend*(L-1)`.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<
    error_costs_32x32_t, affine_gap_costs_t, allocator_type_, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, sz_caps_sh_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_haswell_k, (int)candidate_lanes_k, void>; // ? AVX2 shared-query `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_haswell_k, 8, void>; // ? AVX2 shared-query `i32` lanes.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    needleman_wunsch_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_scores(substituter_t subs, affine_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief The largest per-cell substitution/gap magnitude; scales the worst-case score against the `i16` range. */
    error_cost_magnitude_t cost_magnitude_() const noexcept {
        return sz_max_of_two(substituter_.magnitude(), gap_costs_.magnitude());
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the narrow walker's `i16` headroom. */
    bool fits_lane_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the wide walker's `i32` headroom. */
    bool fits_i32_range_(size_t query_length, size_t candidate_length) const noexcept {
        return (ssize_t)(query_length + candidate_length) * (ssize_t)cost_magnitude_() <= 2000000000;
    }

    /** @brief The score of an all-gap alignment of one empty side against `length` characters: one open + extensions. */
    ssize_t empty_cell_score_(size_t length) const noexcept {
        if (length == 0) return 0;
        return (ssize_t)gap_costs_.open + (ssize_t)gap_costs_.extend * (ssize_t)(length - 1);
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the narrow `i16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_lane_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the wide `i32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i32_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: a global affine empty cell is one open plus extensions. */
    auto empty_cell_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept -> ssize_t {
            return empty_cell_score_(sz_max_of_two(query_length, candidate_length));
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, queries, candidates, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, queries, candidates, results,
                                                       cross_similarities_t::all_pairs_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, sequences, sequences, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, sequences, sequences, results,
                                                       cross_similarities_t::symmetric_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

#pragma endregion Public Cross Product Overloads
};

/**
 *  @brief Batched byte-level @b weighted @b affine-gap Smith-Waterman (local) scores on Haswell, packing up to 16
 *      candidates of a shared query into the signed-`i16` local affine `candidate_lane_walker` and falling back to the
 *      per-pair anti-diagonal Dynamic Programming scorer for the long tail (scores that escape `i16`).
 *
 *  Local sibling of the Haswell affine `needleman_wunsch_scores` cross-product engine: identical query-major
 *  grouping, transpose, and scatter, but it dispatches the @b local affine lane walker (zero-clamped recurrence,
 *  score = the maximum cell of the matrix) and an empty `(query, candidate)` cell scores @b 0 (a local alignment may
 *  align nothing).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<
    error_costs_32x32_t, affine_gap_costs_t, allocator_type_, capability_,
    std::enable_if_t<(capability_ & sz_cap_haswell_k) != 0 && (capability_ & sz_cap_icelake_k) == 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 16;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, sz_caps_sh_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_haswell_k, (int)candidate_lanes_k, void>; // ? AVX2 shared-query local `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_haswell_k, 8, void>; // ? AVX2 shared-query local `i32` lanes.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    smith_waterman_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    smith_waterman_scores(substituter_t subs, affine_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief The largest per-cell substitution/gap magnitude; scales the worst-case score against the `i16` range. */
    error_cost_magnitude_t cost_magnitude_() const noexcept {
        return sz_max_of_two(substituter_.magnitude(), gap_costs_.magnitude());
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the narrow walker's `i16` headroom. */
    bool fits_lane_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the wide walker's `i32` headroom. */
    bool fits_i32_range_(size_t query_length, size_t candidate_length) const noexcept {
        return (ssize_t)(query_length + candidate_length) * (ssize_t)cost_magnitude_() <= 2000000000;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the narrow `i16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_lane_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the wide `i32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i32_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: a local alignment may align nothing, so an empty cell is 0. */
    auto empty_cell_policy_() const noexcept {
        return []([[maybe_unused]] size_t query_length, [[maybe_unused]] size_t candidate_length) noexcept -> ssize_t {
            return 0;
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, queries, candidates, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, queries, candidates, results,
                                                       cross_similarities_t::all_pairs_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                narrow, wide, fallback, sequences, sequences, fits_wide_policy_(), specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide_policy_(), empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        return cross_product_candidate_lanes_parallel_(narrow, wide, fallback, sequences, sequences, results,
                                                       cross_similarities_t::symmetric_k, score_scratch_,
                                                       std::forward<executor_type_>(executor), fits_narrow_policy_(),
                                                       fits_wide_policy_(), empty_cell_policy_(), specs);
    }

#pragma endregion Public Cross Product Overloads
};

#pragma endregion Inter Sequence Cross Product Engines

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
