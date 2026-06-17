/**
 *  @brief Arm NEON (AArch64) string-similarity backend.
 *  @file include/stringzillas/similarities/neon.hpp
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 *  @sa include/stringzillas/similarities/icelake.hpp
 */
#ifndef STRINGZILLAS_SIMILARITIES_NEON_HPP_
#define STRINGZILLAS_SIMILARITIES_NEON_HPP_

#include "stringzillas/similarities/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

/*  NEON implementation of the string similarity algorithms for 64-bit Arm CPUs.
 *  Mirrors the Ice Lake @b diagonal class scorers, but uses NEON's native byte-gather `vqtbl4q_u8`
 *  (a 64-byte = 4x16 table indexed by 16 bytes, with out-of-range indices returning zero) instead of
 *  `VPERMB`: the 256-entry `byte_to_class` map is resolved with four 64-byte windows blended by XOR/OR,
 *  and the (32 x 32) cost matrix is folded into 16 resident 64-byte windows blended by a balanced tree.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/**
 *  @brief Until a bit-parallel Myers fast path is vectorized for NEON, reuse the serial one. The serial
 *         Myers is SIMD-free (64-bit-word lanes), so it runs unchanged on AArch64; the byte-level
 *         Levenshtein dispatch reaches for it on short unit-cost pairs before the anti-diagonal scorers.
 */
template <>
struct levenshtein_distance_myers<char, sz_cap_neon_k> : public levenshtein_distance_myers<char, sz_cap_serial_k> {
    using levenshtein_distance_myers<char, sz_cap_serial_k>::levenshtein_distance_myers;
};

/** @brief Redirects the NEON horizontal-walker specialization to the serial version. The Wagner-Fischer
 *         tiny-input path is not worth vectorizing, so it shares the serial implementation. */
template <typename char_type_, typename score_type_, typename substituter_type_, typename gap_costs_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                         sz_cap_neon_k, void>
    : public horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                               sz_cap_serial_k, void> {

    using base_t = horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                                     sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

/**
 *  @brief Helper object optimizing the most expensive part of class-based variable-substitution-cost
 *         alignment methods for NEON CPUs. It's designed for @b diagonal layout "walkers", where both
 *         class operands of the (32 x 32) `class_substitution_costs` matrix vary lane-by-lane while
 *         mapping each input byte to one of the @b `error_costs_classes_count_k` (32) classes via the
 *         `byte_to_class[256]` map.
 *
 *  The first stage maps a byte to its class using `vqtbl4q_u8`: each lookup addresses one 64-byte window of
 *  the 256-entry table, and indices that fall outside [0, 63] return zero, so XOR-ing the index by 0x40 / 0x80
 *  / 0xc0 and OR-ing the four windows reconstructs the full 256-entry map. This is exposed via `classify16`,
 *  so diagonal walkers can pre-classify both strings @b once and feed class-index buffers into the hot loop.
 *
 *  The second stage looks up the cost for two varying class operands at once by keeping the matrix folded into
 *  16 loop-invariant 64-byte windows, addressed as `idx = ((first_class & 1) << 5) | second_class` with the
 *  window `window = first_class >> 1`, resolved through 16x `vqtbl4q_u8` and a balanced tree of mask-blends.
 *  Every `idx` is below 64 by construction, so the window lookups need no XOR offset.
 *
 *  This is a common abstraction for both:
 *  - Local SW and global NW alignment.
 *  - Serial and parallel implementations.
 *  - 16-bit and 32-bit costs.
 *  - Any memory allocator used.
 */
struct substitution_lookup_neon_t {
    uint8x16x4_t byte_to_class_vecs_[4];
    uint8x16x4_t cost_windows_vecs_[16];

    inline substitution_lookup_neon_t() noexcept {}

    inline void reload_classes(u8_t const *byte_to_class) noexcept {
        byte_to_class_vecs_[0] = vld1q_u8_x4(byte_to_class + 64 * 0);
        byte_to_class_vecs_[1] = vld1q_u8_x4(byte_to_class + 64 * 1);
        byte_to_class_vecs_[2] = vld1q_u8_x4(byte_to_class + 64 * 2);
        byte_to_class_vecs_[3] = vld1q_u8_x4(byte_to_class + 64 * 3);
    }

    /**
     *  @brief Folds the (32 x 32) class cost matrix into 16 resident 64-byte windows for `lookup16`.
     *  @param transpose When set, the matrix is loaded as @b transposed, so that `lookup16(first, second)`
     *         still returns the original `class_substitution_costs[first][second]` even when the diagonal walker
     *         has swapped the shorter and longer strings (which flips the order of the two class operands).
     */
    inline void reload_costs(
        error_cost_t const (&class_substitution_costs)[error_costs_classes_count_k][error_costs_classes_count_k],
        bool transpose) noexcept {
        alignas(16) error_cost_t windows[16 * 64];
        for (size_t window = 0; window != 16; ++window)
            for (size_t low_bit = 0; low_bit != 2; ++low_bit) {
                size_t const first_class = window * 2 + low_bit;
                for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                    windows[window * 64 + low_bit * 32 + second_class] =
                        transpose ? class_substitution_costs[second_class][first_class]
                                  : class_substitution_costs[first_class][second_class];
            }
        for (size_t window = 0; window != 16; ++window)
            cost_windows_vecs_[window] = vld1q_u8_x4((u8_t const *)(windows + window * 64));
    }

    inline uint8x16_t classify16(uint8x16_t text_vec) const noexcept {

        // Map each input byte to its class using the 256-entry `byte_to_class` table. Each `vqtbl4q_u8`
        // addresses one 64-byte window and returns zero for indices outside [0, 63], so XOR-ing the index by
        // 0x40 / 0x80 / 0xc0 selects the matching window and OR-ing the four results reconstructs the map.
        uint8x16_t lookup_0_to_63 = vqtbl4q_u8(byte_to_class_vecs_[0], text_vec);
        uint8x16_t lookup_64_to_127 = vqtbl4q_u8(byte_to_class_vecs_[1], veorq_u8(text_vec, vdupq_n_u8(0x40)));
        uint8x16_t lookup_128_to_191 = vqtbl4q_u8(byte_to_class_vecs_[2], veorq_u8(text_vec, vdupq_n_u8(0x80)));
        uint8x16_t lookup_192_to_255 = vqtbl4q_u8(byte_to_class_vecs_[3], veorq_u8(text_vec, vdupq_n_u8(0xc0)));
        return vorrq_u8(vorrq_u8(lookup_0_to_63, lookup_64_to_127), vorrq_u8(lookup_128_to_191, lookup_192_to_255));
    }

    inline int8x16_t lookup16(uint8x16_t first_class_vec, uint8x16_t second_class_vec) const noexcept {

        // The permute index inside every window is `((first_class & 1) << 5) | second_class`, always below 64.
        uint8x16_t index_vec = vorrq_u8( //
            vshlq_n_u8(vandq_u8(first_class_vec, vdupq_n_u8(1)), 5), second_class_vec);

        // Gather one candidate from each of the 16 windows. `vqtbl4q_u8` returns zero outside [0, 63], but
        // here the index is always valid, so each window yields its own (32 x 32) cost cell.
        uint8x16_t permuted_vecs[16];
        for (size_t window = 0; window != 16; ++window)
            permuted_vecs[window] = vqtbl4q_u8(cost_windows_vecs_[window], index_vec);

        // Select the correct window per lane via a balanced tree of bit-select blends over `window = first_class >> 1`.
        uint8x16_t window_vec = vandq_u8(vshrq_n_u8(first_class_vec, 1), vdupq_n_u8(15));
        uint8x16_t const window_bit0 = vtstq_u8(window_vec, vdupq_n_u8(1));
        uint8x16_t blend4_vecs[8];
        for (size_t pair = 0; pair != 8; ++pair)
            blend4_vecs[pair] = vbslq_u8(window_bit0, permuted_vecs[2 * pair + 1], permuted_vecs[2 * pair]);
        uint8x16_t const window_bit1 = vtstq_u8(window_vec, vdupq_n_u8(2));
        uint8x16_t blend3_vecs[4];
        for (size_t pair = 0; pair != 4; ++pair)
            blend3_vecs[pair] = vbslq_u8(window_bit1, blend4_vecs[2 * pair + 1], blend4_vecs[2 * pair]);
        uint8x16_t const window_bit2 = vtstq_u8(window_vec, vdupq_n_u8(4));
        uint8x16_t blend2_vecs[2];
        for (size_t pair = 0; pair != 2; ++pair)
            blend2_vecs[pair] = vbslq_u8(window_bit2, blend3_vecs[2 * pair + 1], blend3_vecs[2 * pair]);
        uint8x16_t const window_bit3 = vtstq_u8(window_vec, vdupq_n_u8(8));
        return vreinterpretq_s8_u8(vbslq_u8(window_bit3, blend2_vecs[1], blend2_vecs[0]));
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b global Needleman-Wunsch score with class-based
 *         substitution costs, over `i16_t` cells.
 *  @note Requires Arm NEON CPUs.
 *
 *  Mirrors the uniform `u16_t` diagonal scorer (reversed-first / forward-second loads, scalar tail),
 *  but replaces the `cmpeq(first, second) -> select(match, mismatch)` substitution term with
 *  `sign_extend_i8_i16(lookup(first_class, second_class))`, evaluated over @b pre-classified class-index buffers
 *  and the resident (32 x 32) cost table built by `prepare`. The objective is maximization, so the
 *  recurrence uses `max`/`add` with a negative gap.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    substitution_lookup_neon_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
        this->transpose_ = transpose; // The scalar tail reads the cost table directly, so it needs the bit too.
    }

    SZ_INLINE void slice_16chars(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        int16x8_t gap_cost_vec) const noexcept {

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);

        // Sign-extend the 16 class-pair substitution costs into two `i16` halves.
        int8x16_t cost_of_substitution_i8_vec = lookup_.lookup16(first_vec, second_vec);
        int16x8_t cost_of_substitution_i16_vecs[2];
        cost_of_substitution_i16_vecs[0] = vmovl_s8(vget_low_s8(cost_of_substitution_i8_vec));
        cost_of_substitution_i16_vecs[1] = vmovl_high_s8(cost_of_substitution_i8_vec);

        for (size_t part = 0; part != 2; ++part) {
            int16x8_t pre_substitution_vec = vld1q_s16(scores_pre_substitution + part * 8);
            int16x8_t pre_insert_vec = vld1q_s16(scores_pre_insertion + part * 8);
            int16x8_t pre_delete_vec = vld1q_s16(scores_pre_deletion + part * 8);
            int16x8_t cost_if_substitution_vec = vaddq_s16(pre_substitution_vec, cost_of_substitution_i16_vecs[part]);
            int16x8_t cost_if_gap_vec = vaddq_s16(vmaxq_s16(pre_insert_vec, pre_delete_vec), gap_cost_vec);
            int16x8_t cell_score_vec = vmaxq_s16(cost_if_substitution_vec, cost_if_gap_vec);
            vst1q_s16(scores_new + part * 8, cell_score_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,    //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new, i16_t gap) const noexcept {
        i16_t const cost_of_substitution =
            this->transpose_ ? this->substituter_.class_substitution_costs[second_slice[i]][first_reversed_slice[i]]
                             : this->substituter_.class_substitution_costs[first_reversed_slice[i]][second_slice[i]];
        i16_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        i16_t const if_gap = sz_max_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap;
        scores_new[i] = sz_max_of_two(if_substitution, if_gap);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        int16x8_t gap_cost_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(                                                           //
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
        int16x8_t const gap_cost_vec = vdupq_n_s16(gap);

        // NEON has no mask loads, so we process full 16-char vectors and finish the remainder with a scalar tail.
        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap_cost_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_classes, second_classes, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_new, gap);

        // The last element of the last chunk is the result of the global alignment.
        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b local Smith-Waterman score with class-based
 *         substitution costs, over `i16_t` cells.
 *  @note Requires Arm NEON CPUs.
 *
 *  Identical to the global diagonal scorer above, but adds the local zero-clamp on every cell and tracks
 *  the running best across the whole matrix.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    substitution_lookup_neon_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
        this->transpose_ = transpose; // The scalar tail reads the cost table directly, so it needs the bit too.
    }

    SZ_INLINE void slice_16chars(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        int16x8_t gap_cost_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);

        int8x16_t cost_of_substitution_i8_vec = lookup_.lookup16(first_vec, second_vec);
        int16x8_t cost_of_substitution_i16_vecs[2];
        cost_of_substitution_i16_vecs[0] = vmovl_s8(vget_low_s8(cost_of_substitution_i8_vec));
        cost_of_substitution_i16_vecs[1] = vmovl_high_s8(cost_of_substitution_i8_vec);

        for (size_t part = 0; part != 2; ++part) {
            int16x8_t pre_substitution_vec = vld1q_s16(scores_pre_substitution + part * 8);
            int16x8_t pre_insert_vec = vld1q_s16(scores_pre_insertion + part * 8);
            int16x8_t pre_delete_vec = vld1q_s16(scores_pre_deletion + part * 8);
            int16x8_t cost_if_substitution_vec = vaddq_s16(pre_substitution_vec, cost_of_substitution_i16_vecs[part]);
            int16x8_t cost_if_gap_vec = vaddq_s16(vmaxq_s16(pre_insert_vec, pre_delete_vec), gap_cost_vec);
            int16x8_t cell_score_vec = vmaxq_s16(cost_if_substitution_vec, cost_if_gap_vec);
            // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
            cell_score_vec = vmaxq_s16(cell_score_vec, vdupq_n_s16(0));
            vst1q_s16(scores_new + part * 8, cell_score_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,    //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new, i16_t gap) const noexcept {
        i16_t const cost_of_substitution =
            this->transpose_ ? this->substituter_.class_substitution_costs[second_slice[i]][first_reversed_slice[i]]
                             : this->substituter_.class_substitution_costs[first_reversed_slice[i]][second_slice[i]];
        i16_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        i16_t const if_gap = sz_max_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap;
        i16_t cell_score = sz_max_of_two(if_substitution, if_gap);
        scores_new[i] = sz_max_of_two(cell_score, (i16_t)0);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        int16x8_t gap_cost_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(                                                           //
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
        i16_t *const scores_new_begin = scores_new;
        i16_t const gap = static_cast<i16_t>(this->gap_costs_.open_or_extend);
        int16x8_t const gap_cost_vec = vdupq_n_s16(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap_cost_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_classes, second_classes, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_new, gap);

        // The running best across the whole matrix is the reported local-alignment score.
        i16_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Variant of the global diagonal class scorer over @b `i32_t` cells, for inputs whose scores
 *         exceed the 16-bit range. Mirrors the `i16_t` scorer but folds the 16 class-pair costs into
 *         four 4-lane `i32` halves via two-step sign extension.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    substitution_lookup_neon_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
        this->transpose_ = transpose; // The scalar tail reads the cost table directly, so it needs the bit too.
    }

    SZ_INLINE void slice_16chars(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        int32x4_t gap_cost_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);

        // Sign-extend the 16 packed `i8` costs into four 4-lane `i32` quarters.
        int8x16_t cost_of_substitution_i8_vec = lookup_.lookup16(first_vec, second_vec);
        int16x8_t cost_low_i16 = vmovl_s8(vget_low_s8(cost_of_substitution_i8_vec));
        int16x8_t cost_high_i16 = vmovl_high_s8(cost_of_substitution_i8_vec);
        int32x4_t cost_of_substitution_i32_vecs[4];
        cost_of_substitution_i32_vecs[0] = vmovl_s16(vget_low_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[1] = vmovl_high_s16(cost_low_i16);
        cost_of_substitution_i32_vecs[2] = vmovl_s16(vget_low_s16(cost_high_i16));
        cost_of_substitution_i32_vecs[3] = vmovl_high_s16(cost_high_i16);

        for (size_t part = 0; part != 4; ++part) {
            int32x4_t pre_substitution_vec = vld1q_s32(scores_pre_substitution + part * 4);
            int32x4_t pre_insert_vec = vld1q_s32(scores_pre_insertion + part * 4);
            int32x4_t pre_delete_vec = vld1q_s32(scores_pre_deletion + part * 4);
            int32x4_t cost_if_substitution_vec = vaddq_s32(pre_substitution_vec, cost_of_substitution_i32_vecs[part]);
            int32x4_t cost_if_gap_vec = vaddq_s32(vmaxq_s32(pre_insert_vec, pre_delete_vec), gap_cost_vec);
            int32x4_t cell_score_vec = vmaxq_s32(cost_if_substitution_vec, cost_if_gap_vec);
            vst1q_s32(scores_new + part * 4, cell_score_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,    //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new, i32_t gap) const noexcept {
        i32_t const cost_of_substitution =
            this->transpose_ ? this->substituter_.class_substitution_costs[second_slice[i]][first_reversed_slice[i]]
                             : this->substituter_.class_substitution_costs[first_reversed_slice[i]][second_slice[i]];
        i32_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        i32_t const if_gap = sz_max_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap;
        scores_new[i] = sz_max_of_two(if_substitution, if_gap);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        int32x4_t gap_cost_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(                                                           //
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

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t const gap = static_cast<i32_t>(this->gap_costs_.open_or_extend);
        int32x4_t const gap_cost_vec = vdupq_n_s32(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap_cost_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_classes, second_classes, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_new, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of the @b local diagonal class scorer over `i32_t` cells. Mirrors the global `i32_t`
 *         scorer, but zero-clamps every cell and tracks the running best across the matrix.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    substitution_lookup_neon_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
        this->transpose_ = transpose; // The scalar tail reads the cost table directly, so it needs the bit too.
    }

    SZ_INLINE void slice_16chars(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        int32x4_t gap_cost_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);

        int8x16_t cost_of_substitution_i8_vec = lookup_.lookup16(first_vec, second_vec);
        int16x8_t cost_low_i16 = vmovl_s8(vget_low_s8(cost_of_substitution_i8_vec));
        int16x8_t cost_high_i16 = vmovl_high_s8(cost_of_substitution_i8_vec);
        int32x4_t cost_of_substitution_i32_vecs[4];
        cost_of_substitution_i32_vecs[0] = vmovl_s16(vget_low_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[1] = vmovl_high_s16(cost_low_i16);
        cost_of_substitution_i32_vecs[2] = vmovl_s16(vget_low_s16(cost_high_i16));
        cost_of_substitution_i32_vecs[3] = vmovl_high_s16(cost_high_i16);

        for (size_t part = 0; part != 4; ++part) {
            int32x4_t pre_substitution_vec = vld1q_s32(scores_pre_substitution + part * 4);
            int32x4_t pre_insert_vec = vld1q_s32(scores_pre_insertion + part * 4);
            int32x4_t pre_delete_vec = vld1q_s32(scores_pre_deletion + part * 4);
            int32x4_t cost_if_substitution_vec = vaddq_s32(pre_substitution_vec, cost_of_substitution_i32_vecs[part]);
            int32x4_t cost_if_gap_vec = vaddq_s32(vmaxq_s32(pre_insert_vec, pre_delete_vec), gap_cost_vec);
            int32x4_t cell_score_vec = vmaxq_s32(cost_if_substitution_vec, cost_if_gap_vec);
            // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
            cell_score_vec = vmaxq_s32(cell_score_vec, vdupq_n_s32(0));
            vst1q_s32(scores_new + part * 4, cell_score_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,    //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new, i32_t gap) const noexcept {
        i32_t const cost_of_substitution =
            this->transpose_ ? this->substituter_.class_substitution_costs[second_slice[i]][first_reversed_slice[i]]
                             : this->substituter_.class_substitution_costs[first_reversed_slice[i]][second_slice[i]];
        i32_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        i32_t const if_gap = sz_max_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap;
        i32_t cell_score = sz_max_of_two(if_substitution, if_gap);
        scores_new[i] = sz_max_of_two(cell_score, (i32_t)0);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        int32x4_t gap_cost_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(                                                           //
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

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t *const scores_new_begin = scores_new;
        i32_t const gap = static_cast<i32_t>(this->gap_costs_.open_or_extend);
        int32x4_t const gap_cost_vec = vdupq_n_s32(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap_cost_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_classes, second_classes, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_new, gap);

        // The running best across the whole matrix is the reported local-alignment score.
        i32_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief NEON @b affine-gap diagonal class scorer - maximizes the @b global Needleman-Wunsch score with
 *         class-based substitution costs, over `i16_t` cells.
 *  @note Requires Arm NEON CPUs.
 *
 *  Mirrors the linear `i16_t` global class scorer (TBL lookup, step-16 loop, scalar tail), but threads the
 *  separate insertion and deletion gap diagonals of the Gotoh recurrence (open vs extend) alongside the main
 *  score diagonal: `if_insertion = max(pre_insertion_opening + open, running_insertions + extend)`,
 *  `if_deletion = max(pre_deletion_opening + open, running_deletions + extend)`, and
 *  `cell = max(max(if_insertion, if_deletion), pre_substitution + cost)`.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    substitution_lookup_neon_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
        this->transpose_ = transpose;
    }

    SZ_INLINE void slice_16chars(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        int16x8_t gap_open_vec, int16x8_t gap_extend_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);

        // Sign-extend the 16 class-pair substitution costs into two `i16` halves.
        int8x16_t cost_of_substitution_i8_vec = lookup_.lookup16(first_vec, second_vec);
        int16x8_t cost_of_substitution_i16_vecs[2];
        cost_of_substitution_i16_vecs[0] = vmovl_s8(vget_low_s8(cost_of_substitution_i8_vec));
        cost_of_substitution_i16_vecs[1] = vmovl_high_s8(cost_of_substitution_i8_vec);

        for (size_t part = 0; part != 2; ++part) {
            int16x8_t pre_substitution_vec = vld1q_s16(scores_pre_substitution + part * 8);
            int16x8_t pre_insert_open_vec = vld1q_s16(scores_pre_insertion + part * 8);
            int16x8_t pre_delete_open_vec = vld1q_s16(scores_pre_deletion + part * 8);
            int16x8_t run_insert_vec = vld1q_s16(scores_running_insertions + part * 8);
            int16x8_t run_delete_vec = vld1q_s16(scores_running_deletions + part * 8);
            int16x8_t cost_if_insert_vec = vmaxq_s16(vaddq_s16(pre_insert_open_vec, gap_open_vec),
                                                     vaddq_s16(run_insert_vec, gap_extend_vec));
            int16x8_t cost_if_delete_vec = vmaxq_s16(vaddq_s16(pre_delete_open_vec, gap_open_vec),
                                                     vaddq_s16(run_delete_vec, gap_extend_vec));
            int16x8_t cost_if_substitution_vec = vaddq_s16(pre_substitution_vec, cost_of_substitution_i16_vecs[part]);
            int16x8_t cell_score_vec = vmaxq_s16(vmaxq_s16(cost_if_insert_vec, cost_if_delete_vec),
                                                 cost_if_substitution_vec);
            vst1q_s16(scores_new + part * 8, cell_score_vec);
            vst1q_s16(scores_new_insertions + part * 8, cost_if_insert_vec);
            vst1q_s16(scores_new_deletions + part * 8, cost_if_delete_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,     //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        i16_t gap_open, i16_t gap_extend) const noexcept {
        // The transposed cost table is folded only into the SIMD lookup, so the scalar tail must swap the
        // two class operands itself to stay correct on @b asymmetric matrices.
        i16_t const cost_of_substitution =
            this->transpose_ ? this->substituter_.class_substitution_costs[second_slice[i]][first_reversed_slice[i]]
                             : this->substituter_.class_substitution_costs[first_reversed_slice[i]][second_slice[i]];
        i16_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        i16_t const if_insertion = sz_max_of_two(scores_pre_insertion[i] + gap_open,
                                                 scores_running_insertions[i] + gap_extend);
        i16_t const if_deletion = sz_max_of_two(scores_pre_deletion[i] + gap_open,
                                                scores_running_deletions[i] + gap_extend);
        scores_new[i] = sz_max_of_two(sz_max_of_two(if_insertion, if_deletion), if_substitution);
        scores_new_insertions[i] = if_insertion;
        scores_new_deletions[i] = if_deletion;
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        int16x8_t gap_open_vec, int16x8_t gap_extend_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(                                                            //
                first_reversed_classes + progress, second_classes + progress,         //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_extend_vec);
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
        int16x8_t const gap_open_vec = vdupq_n_s16(gap_open);
        int16x8_t const gap_extend_vec = vdupq_n_s16(gap_extend);

        // NEON has no mask loads, so we process full 16-char vectors and finish the remainder with a scalar tail.
        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open_vec, gap_extend_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_classes, second_classes, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_running_insertions, scores_running_deletions, scores_new,
                        scores_new_insertions, scores_new_deletions, gap_open, gap_extend);

        // The last element of the last chunk is the result of the global alignment.
        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b affine-gap diagonal class scorer - maximizes the @b local Smith-Waterman score with
 *         class-based substitution costs, over `i16_t` cells.
 *  @note Requires Arm NEON CPUs.
 *
 *  Identical to the global affine scorer above, but adds the Smith-Waterman-Gotoh zero-reset on @b only the
 *  substitution term (the insertion/deletion gap matrices stay unclamped) and tracks the running best across
 *  the whole matrix.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    substitution_lookup_neon_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
        this->transpose_ = transpose;
    }

    SZ_INLINE void slice_16chars(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        int16x8_t gap_open_vec, int16x8_t gap_extend_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);

        int8x16_t cost_of_substitution_i8_vec = lookup_.lookup16(first_vec, second_vec);
        int16x8_t cost_of_substitution_i16_vecs[2];
        cost_of_substitution_i16_vecs[0] = vmovl_s8(vget_low_s8(cost_of_substitution_i8_vec));
        cost_of_substitution_i16_vecs[1] = vmovl_high_s8(cost_of_substitution_i8_vec);

        for (size_t part = 0; part != 2; ++part) {
            int16x8_t pre_substitution_vec = vld1q_s16(scores_pre_substitution + part * 8);
            int16x8_t pre_insert_open_vec = vld1q_s16(scores_pre_insertion + part * 8);
            int16x8_t pre_delete_open_vec = vld1q_s16(scores_pre_deletion + part * 8);
            int16x8_t run_insert_vec = vld1q_s16(scores_running_insertions + part * 8);
            int16x8_t run_delete_vec = vld1q_s16(scores_running_deletions + part * 8);
            int16x8_t cost_if_insert_vec = vmaxq_s16(vaddq_s16(pre_insert_open_vec, gap_open_vec),
                                                     vaddq_s16(run_insert_vec, gap_extend_vec));
            int16x8_t cost_if_delete_vec = vmaxq_s16(vaddq_s16(pre_delete_open_vec, gap_open_vec),
                                                     vaddq_s16(run_delete_vec, gap_extend_vec));
            // In Local Alignment for SW the zero-reset is applied to @b only the substitution term;
            // the insertion/deletion gap matrices are not clamped, exactly like the serial scorer.
            int16x8_t cost_if_substitution_vec = vmaxq_s16(
                vaddq_s16(pre_substitution_vec, cost_of_substitution_i16_vecs[part]), vdupq_n_s16(0));
            int16x8_t cell_score_vec = vmaxq_s16(vmaxq_s16(cost_if_insert_vec, cost_if_delete_vec),
                                                 cost_if_substitution_vec);
            vst1q_s16(scores_new + part * 8, cell_score_vec);
            vst1q_s16(scores_new_insertions + part * 8, cost_if_insert_vec);
            vst1q_s16(scores_new_deletions + part * 8, cost_if_delete_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,     //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        i16_t gap_open, i16_t gap_extend) const noexcept {
        // The transposed cost table is folded only into the SIMD lookup, so the scalar tail must swap the
        // two class operands itself to stay correct on @b asymmetric matrices.
        i16_t const cost_of_substitution =
            this->transpose_ ? this->substituter_.class_substitution_costs[second_slice[i]][first_reversed_slice[i]]
                             : this->substituter_.class_substitution_costs[first_reversed_slice[i]][second_slice[i]];
        i16_t const if_substitution = sz_max_of_two(scores_pre_substitution[i] + cost_of_substitution, (i16_t)0);
        i16_t const if_insertion = sz_max_of_two(scores_pre_insertion[i] + gap_open,
                                                 scores_running_insertions[i] + gap_extend);
        i16_t const if_deletion = sz_max_of_two(scores_pre_deletion[i] + gap_open,
                                                scores_running_deletions[i] + gap_extend);
        scores_new[i] = sz_max_of_two(sz_max_of_two(if_insertion, if_deletion), if_substitution);
        scores_new_insertions[i] = if_insertion;
        scores_new_deletions[i] = if_deletion;
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        int16x8_t gap_open_vec, int16x8_t gap_extend_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(                                                            //
                first_reversed_classes + progress, second_classes + progress,         //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_extend_vec);
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
        i16_t *const scores_new_begin = scores_new;
        i16_t const gap_open = static_cast<i16_t>(this->gap_costs_.open);
        i16_t const gap_extend = static_cast<i16_t>(this->gap_costs_.extend);
        int16x8_t const gap_open_vec = vdupq_n_s16(gap_open);
        int16x8_t const gap_extend_vec = vdupq_n_s16(gap_extend);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open_vec, gap_extend_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_classes, second_classes, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_running_insertions, scores_running_deletions, scores_new,
                        scores_new_insertions, scores_new_deletions, gap_open, gap_extend);

        // The running best across the whole matrix is the reported local-alignment score.
        i16_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Variant of the @b global affine diagonal class scorer over `i32_t` cells, for inputs whose scores
 *         exceed the 16-bit range. Mirrors the `i16_t` affine scorer but folds the 16 class-pair costs into
 *         four 4-lane `i32` quarters via two-step sign extension.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    substitution_lookup_neon_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
        this->transpose_ = transpose;
    }

    SZ_INLINE void slice_16chars(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        int32x4_t gap_open_vec, int32x4_t gap_extend_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);

        // Sign-extend the 16 packed `i8` costs into four 4-lane `i32` quarters.
        int8x16_t cost_of_substitution_i8_vec = lookup_.lookup16(first_vec, second_vec);
        int16x8_t cost_low_i16 = vmovl_s8(vget_low_s8(cost_of_substitution_i8_vec));
        int16x8_t cost_high_i16 = vmovl_high_s8(cost_of_substitution_i8_vec);
        int32x4_t cost_of_substitution_i32_vecs[4];
        cost_of_substitution_i32_vecs[0] = vmovl_s16(vget_low_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[1] = vmovl_high_s16(cost_low_i16);
        cost_of_substitution_i32_vecs[2] = vmovl_s16(vget_low_s16(cost_high_i16));
        cost_of_substitution_i32_vecs[3] = vmovl_high_s16(cost_high_i16);

        for (size_t part = 0; part != 4; ++part) {
            int32x4_t pre_substitution_vec = vld1q_s32(scores_pre_substitution + part * 4);
            int32x4_t pre_insert_open_vec = vld1q_s32(scores_pre_insertion + part * 4);
            int32x4_t pre_delete_open_vec = vld1q_s32(scores_pre_deletion + part * 4);
            int32x4_t run_insert_vec = vld1q_s32(scores_running_insertions + part * 4);
            int32x4_t run_delete_vec = vld1q_s32(scores_running_deletions + part * 4);
            int32x4_t cost_if_insert_vec = vmaxq_s32(vaddq_s32(pre_insert_open_vec, gap_open_vec),
                                                     vaddq_s32(run_insert_vec, gap_extend_vec));
            int32x4_t cost_if_delete_vec = vmaxq_s32(vaddq_s32(pre_delete_open_vec, gap_open_vec),
                                                     vaddq_s32(run_delete_vec, gap_extend_vec));
            int32x4_t cost_if_substitution_vec = vaddq_s32(pre_substitution_vec, cost_of_substitution_i32_vecs[part]);
            int32x4_t cell_score_vec = vmaxq_s32(vmaxq_s32(cost_if_insert_vec, cost_if_delete_vec),
                                                 cost_if_substitution_vec);
            vst1q_s32(scores_new + part * 4, cell_score_vec);
            vst1q_s32(scores_new_insertions + part * 4, cost_if_insert_vec);
            vst1q_s32(scores_new_deletions + part * 4, cost_if_delete_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,     //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        i32_t gap_open, i32_t gap_extend) const noexcept {
        // The transposed cost table is folded only into the SIMD lookup, so the scalar tail must swap the
        // two class operands itself to stay correct on @b asymmetric matrices.
        i32_t const cost_of_substitution =
            this->transpose_ ? this->substituter_.class_substitution_costs[second_slice[i]][first_reversed_slice[i]]
                             : this->substituter_.class_substitution_costs[first_reversed_slice[i]][second_slice[i]];
        i32_t const if_substitution = scores_pre_substitution[i] + cost_of_substitution;
        i32_t const if_insertion = sz_max_of_two(scores_pre_insertion[i] + gap_open,
                                                 scores_running_insertions[i] + gap_extend);
        i32_t const if_deletion = sz_max_of_two(scores_pre_deletion[i] + gap_open,
                                                scores_running_deletions[i] + gap_extend);
        scores_new[i] = sz_max_of_two(sz_max_of_two(if_insertion, if_deletion), if_substitution);
        scores_new_insertions[i] = if_insertion;
        scores_new_deletions[i] = if_deletion;
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        int32x4_t gap_open_vec, int32x4_t gap_extend_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(                                                            //
                first_reversed_classes + progress, second_classes + progress,         //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_extend_vec);
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

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t const gap_open = static_cast<i32_t>(this->gap_costs_.open);
        i32_t const gap_extend = static_cast<i32_t>(this->gap_costs_.extend);
        int32x4_t const gap_open_vec = vdupq_n_s32(gap_open);
        int32x4_t const gap_extend_vec = vdupq_n_s32(gap_extend);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open_vec, gap_extend_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_classes, second_classes, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_running_insertions, scores_running_deletions, scores_new,
                        scores_new_insertions, scores_new_deletions, gap_open, gap_extend);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of the @b local affine diagonal class scorer over `i32_t` cells. Mirrors the `i32_t`
 *         global affine scorer, plus the Smith-Waterman-Gotoh zero-reset on the substitution term and the
 *         running-best reduction.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    substitution_lookup_neon_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
        this->transpose_ = transpose;
    }

    SZ_INLINE void slice_16chars(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        int32x4_t gap_open_vec, int32x4_t gap_extend_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);

        int8x16_t cost_of_substitution_i8_vec = lookup_.lookup16(first_vec, second_vec);
        int16x8_t cost_low_i16 = vmovl_s8(vget_low_s8(cost_of_substitution_i8_vec));
        int16x8_t cost_high_i16 = vmovl_high_s8(cost_of_substitution_i8_vec);
        int32x4_t cost_of_substitution_i32_vecs[4];
        cost_of_substitution_i32_vecs[0] = vmovl_s16(vget_low_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[1] = vmovl_high_s16(cost_low_i16);
        cost_of_substitution_i32_vecs[2] = vmovl_s16(vget_low_s16(cost_high_i16));
        cost_of_substitution_i32_vecs[3] = vmovl_high_s16(cost_high_i16);

        for (size_t part = 0; part != 4; ++part) {
            int32x4_t pre_substitution_vec = vld1q_s32(scores_pre_substitution + part * 4);
            int32x4_t pre_insert_open_vec = vld1q_s32(scores_pre_insertion + part * 4);
            int32x4_t pre_delete_open_vec = vld1q_s32(scores_pre_deletion + part * 4);
            int32x4_t run_insert_vec = vld1q_s32(scores_running_insertions + part * 4);
            int32x4_t run_delete_vec = vld1q_s32(scores_running_deletions + part * 4);
            int32x4_t cost_if_insert_vec = vmaxq_s32(vaddq_s32(pre_insert_open_vec, gap_open_vec),
                                                     vaddq_s32(run_insert_vec, gap_extend_vec));
            int32x4_t cost_if_delete_vec = vmaxq_s32(vaddq_s32(pre_delete_open_vec, gap_open_vec),
                                                     vaddq_s32(run_delete_vec, gap_extend_vec));
            // In Local Alignment for SW the zero-reset is applied to @b only the substitution term;
            // the insertion/deletion gap matrices are not clamped, exactly like the serial scorer.
            int32x4_t cost_if_substitution_vec = vmaxq_s32(
                vaddq_s32(pre_substitution_vec, cost_of_substitution_i32_vecs[part]), vdupq_n_s32(0));
            int32x4_t cell_score_vec = vmaxq_s32(vmaxq_s32(cost_if_insert_vec, cost_if_delete_vec),
                                                 cost_if_substitution_vec);
            vst1q_s32(scores_new + part * 4, cell_score_vec);
            vst1q_s32(scores_new_insertions + part * 4, cost_if_insert_vec);
            vst1q_s32(scores_new_deletions + part * 4, cost_if_delete_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,     //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        i32_t gap_open, i32_t gap_extend) const noexcept {
        // The transposed cost table is folded only into the SIMD lookup, so the scalar tail must swap the
        // two class operands itself to stay correct on @b asymmetric matrices.
        i32_t const cost_of_substitution =
            this->transpose_ ? this->substituter_.class_substitution_costs[second_slice[i]][first_reversed_slice[i]]
                             : this->substituter_.class_substitution_costs[first_reversed_slice[i]][second_slice[i]];
        i32_t const if_substitution = sz_max_of_two(scores_pre_substitution[i] + cost_of_substitution, (i32_t)0);
        i32_t const if_insertion = sz_max_of_two(scores_pre_insertion[i] + gap_open,
                                                 scores_running_insertions[i] + gap_extend);
        i32_t const if_deletion = sz_max_of_two(scores_pre_deletion[i] + gap_open,
                                                scores_running_deletions[i] + gap_extend);
        scores_new[i] = sz_max_of_two(sz_max_of_two(if_insertion, if_deletion), if_substitution);
        scores_new_insertions[i] = if_insertion;
        scores_new_deletions[i] = if_deletion;
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        int32x4_t gap_open_vec, int32x4_t gap_extend_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(                                                            //
                first_reversed_classes + progress, second_classes + progress,         //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_extend_vec);
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

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t *const scores_new_begin = scores_new;
        i32_t const gap_open = static_cast<i32_t>(this->gap_costs_.open);
        i32_t const gap_extend = static_cast<i32_t>(this->gap_costs_.extend);
        int32x4_t const gap_open_vec = vdupq_n_s32(gap_open);
        int32x4_t const gap_extend_vec = vdupq_n_s32(gap_extend);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open_vec, gap_extend_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_classes, second_classes, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_running_insertions, scores_running_deletions, scores_new,
                        scores_new_insertions, scores_new_deletions, gap_open, gap_extend);

        // The running best across the whole matrix is the reported local-alignment score.
        i32_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

#pragma region Uniform Cost Levenshtein

/**
 *  @brief Redirects any NEON uniform-cost minimize-distance scorer to the serial version. The explicit
 *         `u16`/`u32` specializations below are more specialized and win by partial ordering for the hot
 *         cell widths; this catches the rarely-instantiated `u8`/`u64` cells (and any not-yet-vectorized
 *         iterator/gap combination) so the dispatch never lands on an undefined template.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_, typename gap_costs_type_,
          sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, uniform_substitution_costs_t,
                   gap_costs_type_, sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, uniform_substitution_costs_t,
                         gap_costs_type_, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, uniform_substitution_costs_t,
                               gap_costs_type_, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

/**
 *  @brief NEON @b uniform-cost diagonal scorer - minimizes the Levenshtein distance over `u16_t` cells.
 *  @note Requires Arm NEON CPUs.
 *
 *  The Levenshtein twin of the class-based scorer above, but the (32 x 32) `vqtbl4q_u8` cost lookup
 *  collapses to a single `vceqq_u8` + `vbslq_u8` (match vs. mismatch), and the objective is
 *  minimization, so the recurrence uses `vminq_u16` with a positive gap. Driven by the generic
 *  serial diagonal walker, which hands it the raw reversed-first / forward-second byte buffers.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion, //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                     //
        uint8x16_t match_cost_u8_vec, uint8x16_t mismatch_cost_u8_vec, uint16x8_t gap_cost_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);
        uint8x16_t equal_vec = vceqq_u8(first_vec, second_vec);
        uint8x16_t cost_u8_vec = vbslq_u8(equal_vec, match_cost_u8_vec, mismatch_cost_u8_vec);
        uint16x8_t cost_u16_vecs[2];
        cost_u16_vecs[0] = vmovl_u8(vget_low_u8(cost_u8_vec));
        cost_u16_vecs[1] = vmovl_high_u8(cost_u8_vec);

        for (size_t part = 0; part != 2; ++part) {
            uint16x8_t pre_substitution_vec = vld1q_u16(scores_pre_substitution + part * 8);
            uint16x8_t pre_insert_vec = vld1q_u16(scores_pre_insertion + part * 8);
            uint16x8_t pre_delete_vec = vld1q_u16(scores_pre_deletion + part * 8);
            uint16x8_t cost_if_substitution_vec = vaddq_u16(pre_substitution_vec, cost_u16_vecs[part]);
            uint16x8_t cost_if_gap_vec = vaddq_u16(vminq_u16(pre_insert_vec, pre_delete_vec), gap_cost_vec);
            uint16x8_t cell_score_vec = vminq_u16(cost_if_substitution_vec, cost_if_gap_vec);
            vst1q_u16(scores_new + part * 8, cell_score_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,    //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion, //
        u16_t const *scores_pre_deletion, u16_t *scores_new, u16_t gap) const noexcept {
        u16_t const cost = first_reversed_slice[i] == second_slice[i] ? (u16_t)this->substituter_.match
                                                                      : (u16_t)this->substituter_.mismatch;
        u16_t const if_substitution = (u16_t)(scores_pre_substitution[i] + cost);
        u16_t const if_gap = (u16_t)(sz_min_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap);
        scores_new[i] = sz_min_of_two(if_substitution, if_gap);
    }

    SZ_NOINLINE void score_slice_trampoline_(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice,                             //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,                //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                                    //
        uint8x16_t match_cost_u8_vec, uint8x16_t mismatch_cost_u8_vec, uint16x8_t gap_cost_vec, //
        size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,
                          match_cost_u8_vec, mismatch_cost_u8_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,         //
        u16_t const *scores_pre_deletion, u16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed = (u8_t const *)first_reversed_slice;
        u8_t const *second = (u8_t const *)second_slice;
        u16_t const gap = static_cast<u16_t>(this->gap_costs_.open_or_extend);
        uint8x16_t const match_cost_u8_vec = vdupq_n_u8((u8_t)this->substituter_.match);
        uint8x16_t const mismatch_cost_u8_vec = vdupq_n_u8((u8_t)this->substituter_.mismatch);
        uint16x8_t const gap_cost_vec = vdupq_n_u16(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed, second, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_u8_vec, mismatch_cost_u8_vec,
                                    gap_cost_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed, second, i, scores_pre_substitution, scores_pre_insertion, scores_pre_deletion,
                        scores_new, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b uniform-cost diagonal scorer - minimizes the Levenshtein distance over `u32_t` cells.
 *  @note Requires Arm NEON CPUs. Same recurrence as the `u16_t` variant, widened to four `u32x4` quarters.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion, //
        u32_t const *scores_pre_deletion, u32_t *scores_new,                     //
        uint8x16_t match_cost_u8_vec, uint8x16_t mismatch_cost_u8_vec, uint32x4_t gap_cost_vec) const noexcept {

        uint8x16_t first_vec = vld1q_u8(first_reversed_slice);
        uint8x16_t second_vec = vld1q_u8(second_slice);
        uint8x16_t equal_vec = vceqq_u8(first_vec, second_vec);
        uint8x16_t cost_u8_vec = vbslq_u8(equal_vec, match_cost_u8_vec, mismatch_cost_u8_vec);
        uint16x8_t cost_low_u16 = vmovl_u8(vget_low_u8(cost_u8_vec));
        uint16x8_t cost_high_u16 = vmovl_high_u8(cost_u8_vec);
        uint32x4_t cost_u32_vecs[4];
        cost_u32_vecs[0] = vmovl_u16(vget_low_u16(cost_low_u16));
        cost_u32_vecs[1] = vmovl_high_u16(cost_low_u16);
        cost_u32_vecs[2] = vmovl_u16(vget_low_u16(cost_high_u16));
        cost_u32_vecs[3] = vmovl_high_u16(cost_high_u16);

        for (size_t part = 0; part != 4; ++part) {
            uint32x4_t pre_substitution_vec = vld1q_u32(scores_pre_substitution + part * 4);
            uint32x4_t pre_insert_vec = vld1q_u32(scores_pre_insertion + part * 4);
            uint32x4_t pre_delete_vec = vld1q_u32(scores_pre_deletion + part * 4);
            uint32x4_t cost_if_substitution_vec = vaddq_u32(pre_substitution_vec, cost_u32_vecs[part]);
            uint32x4_t cost_if_gap_vec = vaddq_u32(vminq_u32(pre_insert_vec, pre_delete_vec), gap_cost_vec);
            uint32x4_t cell_score_vec = vminq_u32(cost_if_substitution_vec, cost_if_gap_vec);
            vst1q_u32(scores_new + part * 4, cell_score_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                  //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,    //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion, //
        u32_t const *scores_pre_deletion, u32_t *scores_new, u32_t gap) const noexcept {
        u32_t const cost = first_reversed_slice[i] == second_slice[i] ? (u32_t)this->substituter_.match
                                                                      : (u32_t)this->substituter_.mismatch;
        u32_t const if_substitution = scores_pre_substitution[i] + cost;
        u32_t const if_gap = sz_min_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap;
        scores_new[i] = sz_min_of_two(if_substitution, if_gap);
    }

    SZ_NOINLINE void score_slice_trampoline_(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice,                             //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,                //
        u32_t const *scores_pre_deletion, u32_t *scores_new,                                    //
        uint8x16_t match_cost_u8_vec, uint8x16_t mismatch_cost_u8_vec, uint32x4_t gap_cost_vec, //
        size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,
                          match_cost_u8_vec, mismatch_cost_u8_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,         //
        u32_t const *scores_pre_deletion, u32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed = (u8_t const *)first_reversed_slice;
        u8_t const *second = (u8_t const *)second_slice;
        u32_t const gap = static_cast<u32_t>(this->gap_costs_.open_or_extend);
        uint8x16_t const match_cost_u8_vec = vdupq_n_u8((u8_t)this->substituter_.match);
        uint8x16_t const mismatch_cost_u8_vec = vdupq_n_u8((u8_t)this->substituter_.mismatch);
        uint32x4_t const gap_cost_vec = vdupq_n_u32(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed, second, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_u8_vec, mismatch_cost_u8_vec,
                                    gap_cost_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed, second, i, scores_pre_substitution, scores_pre_insertion, scores_pre_deletion,
                        scores_new, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b affine-gap uniform-cost diagonal scorer - minimizes Levenshtein over `u16_t` cells.
 *  @note Requires Arm NEON CPUs. Gotoh recurrence with separate insertion/deletion gap planes.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,  //
        u16_t const *scores_pre_deletion, u16_t const *scores_running_insertions, //
        u16_t const *scores_running_deletions, u16_t *scores_new,                 //
        u16_t *scores_new_insertions, u16_t *scores_new_deletions,                //
        uint8x16_t match_cost_u8_vec, uint8x16_t mismatch_cost_u8_vec,            //
        uint16x8_t gap_open_vec, uint16x8_t gap_extend_vec) const noexcept {

        uint8x16_t equal_vec = vceqq_u8(vld1q_u8(first_reversed_slice), vld1q_u8(second_slice));
        uint8x16_t cost_u8_vec = vbslq_u8(equal_vec, match_cost_u8_vec, mismatch_cost_u8_vec);
        uint16x8_t cost_u16_vecs[2];
        cost_u16_vecs[0] = vmovl_u8(vget_low_u8(cost_u8_vec));
        cost_u16_vecs[1] = vmovl_high_u8(cost_u8_vec);

        for (size_t part = 0; part != 2; ++part) {
            uint16x8_t pre_substitution_vec = vld1q_u16(scores_pre_substitution + part * 8);
            uint16x8_t pre_insert_open_vec = vld1q_u16(scores_pre_insertion + part * 8);
            uint16x8_t pre_delete_open_vec = vld1q_u16(scores_pre_deletion + part * 8);
            uint16x8_t run_insert_vec = vld1q_u16(scores_running_insertions + part * 8);
            uint16x8_t run_delete_vec = vld1q_u16(scores_running_deletions + part * 8);
            uint16x8_t cost_if_insert_vec = vminq_u16(vaddq_u16(pre_insert_open_vec, gap_open_vec),
                                                      vaddq_u16(run_insert_vec, gap_extend_vec));
            uint16x8_t cost_if_delete_vec = vminq_u16(vaddq_u16(pre_delete_open_vec, gap_open_vec),
                                                      vaddq_u16(run_delete_vec, gap_extend_vec));
            uint16x8_t cost_if_substitution_vec = vaddq_u16(pre_substitution_vec, cost_u16_vecs[part]);
            uint16x8_t cell_score_vec = vminq_u16(vminq_u16(cost_if_insert_vec, cost_if_delete_vec),
                                                  cost_if_substitution_vec);
            vst1q_u16(scores_new + part * 8, cell_score_vec);
            vst1q_u16(scores_new_insertions + part * 8, cost_if_insert_vec);
            vst1q_u16(scores_new_deletions + part * 8, cost_if_delete_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,     //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,  //
        u16_t const *scores_pre_deletion, u16_t const *scores_running_insertions, //
        u16_t const *scores_running_deletions, u16_t *scores_new,                 //
        u16_t *scores_new_insertions, u16_t *scores_new_deletions,                //
        u16_t gap_open, u16_t gap_extend) const noexcept {
        u16_t const cost = first_reversed_slice[i] == second_slice[i] ? (u16_t)this->substituter_.match
                                                                      : (u16_t)this->substituter_.mismatch;
        u16_t const if_substitution = (u16_t)(scores_pre_substitution[i] + cost);
        u16_t const if_insertion = sz_min_of_two((u16_t)(scores_pre_insertion[i] + gap_open),
                                                 (u16_t)(scores_running_insertions[i] + gap_extend));
        u16_t const if_deletion = sz_min_of_two((u16_t)(scores_pre_deletion[i] + gap_open),
                                                (u16_t)(scores_running_deletions[i] + gap_extend));
        scores_new[i] = sz_min_of_two(sz_min_of_two(if_insertion, if_deletion), if_substitution);
        scores_new_insertions[i] = if_insertion;
        scores_new_deletions[i] = if_deletion;
    }

    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,  //
        u16_t const *scores_pre_deletion, u16_t const *scores_running_insertions, //
        u16_t const *scores_running_deletions, u16_t *scores_new,                 //
        u16_t *scores_new_insertions, u16_t *scores_new_deletions,                //
        uint8x16_t match_cost_u8_vec, uint8x16_t mismatch_cost_u8_vec, uint16x8_t gap_open_vec,
        uint16x8_t gap_extend_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress,
                          scores_running_insertions + progress, scores_running_deletions + progress,
                          scores_new + progress, scores_new_insertions + progress, scores_new_deletions + progress,
                          match_cost_u8_vec, mismatch_cost_u8_vec, gap_open_vec, gap_extend_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,         //
        u16_t const *scores_pre_deletion, u16_t const *scores_running_insertions,        //
        u16_t const *scores_running_deletions, u16_t *scores_new,                        //
        u16_t *scores_new_insertions, u16_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed = (u8_t const *)first_reversed_slice;
        u8_t const *second = (u8_t const *)second_slice;
        u16_t const gap_open = static_cast<u16_t>(this->gap_costs_.open);
        u16_t const gap_extend = static_cast<u16_t>(this->gap_costs_.extend);
        uint8x16_t const match_cost_u8_vec = vdupq_n_u8((u8_t)this->substituter_.match);
        uint8x16_t const mismatch_cost_u8_vec = vdupq_n_u8((u8_t)this->substituter_.mismatch);
        uint16x8_t const gap_open_vec = vdupq_n_u16(gap_open);
        uint16x8_t const gap_extend_vec = vdupq_n_u16(gap_extend);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed, second, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_running_insertions, scores_running_deletions,
                                    scores_new, scores_new_insertions, scores_new_deletions, match_cost_u8_vec,
                                    mismatch_cost_u8_vec, gap_open_vec, gap_extend_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed, second, i, scores_pre_substitution, scores_pre_insertion, scores_pre_deletion,
                        scores_running_insertions, scores_running_deletions, scores_new, scores_new_insertions,
                        scores_new_deletions, gap_open, gap_extend);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b affine-gap uniform-cost diagonal scorer - minimizes Levenshtein over `u32_t` cells.
 *  @note Requires Arm NEON CPUs. Same Gotoh recurrence as the `u16_t` variant, in four `u32x4` quarters.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,  //
        u32_t const *scores_pre_deletion, u32_t const *scores_running_insertions, //
        u32_t const *scores_running_deletions, u32_t *scores_new,                 //
        u32_t *scores_new_insertions, u32_t *scores_new_deletions,                //
        uint8x16_t match_cost_u8_vec, uint8x16_t mismatch_cost_u8_vec,            //
        uint32x4_t gap_open_vec, uint32x4_t gap_extend_vec) const noexcept {

        uint8x16_t equal_vec = vceqq_u8(vld1q_u8(first_reversed_slice), vld1q_u8(second_slice));
        uint8x16_t cost_u8_vec = vbslq_u8(equal_vec, match_cost_u8_vec, mismatch_cost_u8_vec);
        uint16x8_t cost_low_u16 = vmovl_u8(vget_low_u8(cost_u8_vec));
        uint16x8_t cost_high_u16 = vmovl_high_u8(cost_u8_vec);
        uint32x4_t cost_u32_vecs[4];
        cost_u32_vecs[0] = vmovl_u16(vget_low_u16(cost_low_u16));
        cost_u32_vecs[1] = vmovl_high_u16(cost_low_u16);
        cost_u32_vecs[2] = vmovl_u16(vget_low_u16(cost_high_u16));
        cost_u32_vecs[3] = vmovl_high_u16(cost_high_u16);

        for (size_t part = 0; part != 4; ++part) {
            uint32x4_t pre_substitution_vec = vld1q_u32(scores_pre_substitution + part * 4);
            uint32x4_t pre_insert_open_vec = vld1q_u32(scores_pre_insertion + part * 4);
            uint32x4_t pre_delete_open_vec = vld1q_u32(scores_pre_deletion + part * 4);
            uint32x4_t run_insert_vec = vld1q_u32(scores_running_insertions + part * 4);
            uint32x4_t run_delete_vec = vld1q_u32(scores_running_deletions + part * 4);
            uint32x4_t cost_if_insert_vec = vminq_u32(vaddq_u32(pre_insert_open_vec, gap_open_vec),
                                                      vaddq_u32(run_insert_vec, gap_extend_vec));
            uint32x4_t cost_if_delete_vec = vminq_u32(vaddq_u32(pre_delete_open_vec, gap_open_vec),
                                                      vaddq_u32(run_delete_vec, gap_extend_vec));
            uint32x4_t cost_if_substitution_vec = vaddq_u32(pre_substitution_vec, cost_u32_vecs[part]);
            uint32x4_t cell_score_vec = vminq_u32(vminq_u32(cost_if_insert_vec, cost_if_delete_vec),
                                                  cost_if_substitution_vec);
            vst1q_u32(scores_new + part * 4, cell_score_vec);
            vst1q_u32(scores_new_insertions + part * 4, cost_if_insert_vec);
            vst1q_u32(scores_new_deletions + part * 4, cost_if_delete_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,     //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,  //
        u32_t const *scores_pre_deletion, u32_t const *scores_running_insertions, //
        u32_t const *scores_running_deletions, u32_t *scores_new,                 //
        u32_t *scores_new_insertions, u32_t *scores_new_deletions,                //
        u32_t gap_open, u32_t gap_extend) const noexcept {
        u32_t const cost = first_reversed_slice[i] == second_slice[i] ? (u32_t)this->substituter_.match
                                                                      : (u32_t)this->substituter_.mismatch;
        u32_t const if_substitution = scores_pre_substitution[i] + cost;
        u32_t const if_insertion = sz_min_of_two(scores_pre_insertion[i] + gap_open,
                                                 scores_running_insertions[i] + gap_extend);
        u32_t const if_deletion = sz_min_of_two(scores_pre_deletion[i] + gap_open,
                                                scores_running_deletions[i] + gap_extend);
        scores_new[i] = sz_min_of_two(sz_min_of_two(if_insertion, if_deletion), if_substitution);
        scores_new_insertions[i] = if_insertion;
        scores_new_deletions[i] = if_deletion;
    }

    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_slice, u8_t const *second_slice,               //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,  //
        u32_t const *scores_pre_deletion, u32_t const *scores_running_insertions, //
        u32_t const *scores_running_deletions, u32_t *scores_new,                 //
        u32_t *scores_new_insertions, u32_t *scores_new_deletions,                //
        uint8x16_t match_cost_u8_vec, uint8x16_t mismatch_cost_u8_vec, uint32x4_t gap_open_vec,
        uint32x4_t gap_extend_vec, size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress,
                          scores_running_insertions + progress, scores_running_deletions + progress,
                          scores_new + progress, scores_new_insertions + progress, scores_new_deletions + progress,
                          match_cost_u8_vec, mismatch_cost_u8_vec, gap_open_vec, gap_extend_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,         //
        u32_t const *scores_pre_deletion, u32_t const *scores_running_insertions,        //
        u32_t const *scores_running_deletions, u32_t *scores_new,                        //
        u32_t *scores_new_insertions, u32_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed = (u8_t const *)first_reversed_slice;
        u8_t const *second = (u8_t const *)second_slice;
        u32_t const gap_open = static_cast<u32_t>(this->gap_costs_.open);
        u32_t const gap_extend = static_cast<u32_t>(this->gap_costs_.extend);
        uint8x16_t const match_cost_u8_vec = vdupq_n_u8((u8_t)this->substituter_.match);
        uint8x16_t const mismatch_cost_u8_vec = vdupq_n_u8((u8_t)this->substituter_.mismatch);
        uint32x4_t const gap_open_vec = vdupq_n_u32(gap_open);
        uint32x4_t const gap_extend_vec = vdupq_n_u32(gap_extend);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed, second, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_running_insertions, scores_running_deletions,
                                    scores_new, scores_new_insertions, scores_new_deletions, match_cost_u8_vec,
                                    mismatch_cost_u8_vec, gap_open_vec, gap_extend_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed, second, i, scores_pre_substitution, scores_pre_insertion, scores_pre_deletion,
                        scores_running_insertions, scores_running_deletions, scores_new, scores_new_insertions,
                        scores_new_deletions, gap_open, gap_extend);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b UTF-8 uniform-cost diagonal scorer - minimizes rune-level Levenshtein over `u16_t` cells.
 *  @note Requires Arm NEON CPUs. Runes are 32-bit, so four `vceqq_u32` compares narrow (`vmovn`) and pack
 *        (`vcombine`) into the two `u16x8` cost halves - no `vget_low`/`vget_high` round-trips.
 */
template <sz_capability_t capability_>
struct tile_scorer<rune_t const *, rune_t const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<rune_t const *, rune_t const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<rune_t const *, rune_t const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                                //
        rune_t const *first_reversed_slice, rune_t const *second_slice,          //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion, //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                     //
        uint16x8_t match_cost_vec, uint16x8_t mismatch_cost_vec, uint16x8_t gap_cost_vec) const noexcept {

        u32_t const *first = (u32_t const *)first_reversed_slice;
        u32_t const *second = (u32_t const *)second_slice;
        uint16x8_t equal_vecs[2];
        equal_vecs[0] = vcombine_u16(vmovn_u32(vceqq_u32(vld1q_u32(first + 0), vld1q_u32(second + 0))),
                                     vmovn_u32(vceqq_u32(vld1q_u32(first + 4), vld1q_u32(second + 4))));
        equal_vecs[1] = vcombine_u16(vmovn_u32(vceqq_u32(vld1q_u32(first + 8), vld1q_u32(second + 8))),
                                     vmovn_u32(vceqq_u32(vld1q_u32(first + 12), vld1q_u32(second + 12))));

        for (size_t part = 0; part != 2; ++part) {
            uint16x8_t cost_vec = vbslq_u16(equal_vecs[part], match_cost_vec, mismatch_cost_vec);
            uint16x8_t pre_substitution_vec = vld1q_u16(scores_pre_substitution + part * 8);
            uint16x8_t pre_insert_vec = vld1q_u16(scores_pre_insertion + part * 8);
            uint16x8_t pre_delete_vec = vld1q_u16(scores_pre_deletion + part * 8);
            uint16x8_t cost_if_substitution_vec = vaddq_u16(pre_substitution_vec, cost_vec);
            uint16x8_t cost_if_gap_vec = vaddq_u16(vminq_u16(pre_insert_vec, pre_delete_vec), gap_cost_vec);
            uint16x8_t cell_score_vec = vminq_u16(cost_if_substitution_vec, cost_if_gap_vec);
            vst1q_u16(scores_new + part * 8, cell_score_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                   //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t i, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,  //
        u16_t const *scores_pre_deletion, u16_t *scores_new, u16_t gap) const noexcept {
        u16_t const cost = first_reversed_slice[i] == second_slice[i] ? (u16_t)this->substituter_.match
                                                                      : (u16_t)this->substituter_.mismatch;
        u16_t const if_substitution = (u16_t)(scores_pre_substitution[i] + cost);
        u16_t const if_gap = (u16_t)(sz_min_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap);
        scores_new[i] = sz_min_of_two(if_substitution, if_gap);
    }

    SZ_NOINLINE void score_slice_trampoline_(                                             //
        rune_t const *first_reversed_slice, rune_t const *second_slice,                   //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,          //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                              //
        uint16x8_t match_cost_vec, uint16x8_t mismatch_cost_vec, uint16x8_t gap_cost_vec, //
        size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,
                          match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                         //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t const length, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,             //
        u16_t const *scores_pre_deletion, u16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u16_t const gap = static_cast<u16_t>(this->gap_costs_.open_or_extend);
        uint16x8_t const match_cost_vec = vdupq_n_u16((u16_t)this->substituter_.match);
        uint16x8_t const mismatch_cost_vec = vdupq_n_u16((u16_t)this->substituter_.mismatch);
        uint16x8_t const gap_cost_vec = vdupq_n_u16(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_vec, mismatch_cost_vec, gap_cost_vec,
                                    from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_slice, second_slice, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_new, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b UTF-8 uniform-cost diagonal scorer - minimizes rune-level Levenshtein over `u32_t` cells.
 *  @note Requires Arm NEON CPUs. Rune compares stay in `u32x4`, so the cost select needs no narrowing at all.
 */
template <sz_capability_t capability_>
struct tile_scorer<rune_t const *, rune_t const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<rune_t const *, rune_t const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<rune_t const *, rune_t const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                                //
        rune_t const *first_reversed_slice, rune_t const *second_slice,          //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion, //
        u32_t const *scores_pre_deletion, u32_t *scores_new,                     //
        uint32x4_t match_cost_vec, uint32x4_t mismatch_cost_vec, uint32x4_t gap_cost_vec) const noexcept {

        u32_t const *first = (u32_t const *)first_reversed_slice;
        u32_t const *second = (u32_t const *)second_slice;
        for (size_t part = 0; part != 4; ++part) {
            uint32x4_t equal_vec = vceqq_u32(vld1q_u32(first + part * 4), vld1q_u32(second + part * 4));
            uint32x4_t cost_vec = vbslq_u32(equal_vec, match_cost_vec, mismatch_cost_vec);
            uint32x4_t pre_substitution_vec = vld1q_u32(scores_pre_substitution + part * 4);
            uint32x4_t pre_insert_vec = vld1q_u32(scores_pre_insertion + part * 4);
            uint32x4_t pre_delete_vec = vld1q_u32(scores_pre_deletion + part * 4);
            uint32x4_t cost_if_substitution_vec = vaddq_u32(pre_substitution_vec, cost_vec);
            uint32x4_t cost_if_gap_vec = vaddq_u32(vminq_u32(pre_insert_vec, pre_delete_vec), gap_cost_vec);
            uint32x4_t cell_score_vec = vminq_u32(cost_if_substitution_vec, cost_if_gap_vec);
            vst1q_u32(scores_new + part * 4, cell_score_vec);
        }
    }

    SZ_INLINE void slice_1char(                                                   //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t i, //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,  //
        u32_t const *scores_pre_deletion, u32_t *scores_new, u32_t gap) const noexcept {
        u32_t const cost = first_reversed_slice[i] == second_slice[i] ? (u32_t)this->substituter_.match
                                                                      : (u32_t)this->substituter_.mismatch;
        u32_t const if_substitution = scores_pre_substitution[i] + cost;
        u32_t const if_gap = sz_min_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap;
        scores_new[i] = sz_min_of_two(if_substitution, if_gap);
    }

    SZ_NOINLINE void score_slice_trampoline_(                                             //
        rune_t const *first_reversed_slice, rune_t const *second_slice,                   //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,          //
        u32_t const *scores_pre_deletion, u32_t *scores_new,                              //
        uint32x4_t match_cost_vec, uint32x4_t mismatch_cost_vec, uint32x4_t gap_cost_vec, //
        size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,
                          match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                         //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t const length, //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,             //
        u32_t const *scores_pre_deletion, u32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u32_t const gap = static_cast<u32_t>(this->gap_costs_.open_or_extend);
        uint32x4_t const match_cost_vec = vdupq_n_u32((u32_t)this->substituter_.match);
        uint32x4_t const mismatch_cost_vec = vdupq_n_u32((u32_t)this->substituter_.mismatch);
        uint32x4_t const gap_cost_vec = vdupq_n_u32(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_vec, mismatch_cost_vec, gap_cost_vec,
                                    from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_slice, second_slice, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_new, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b uniform-cost diagonal scorer - minimizes the Levenshtein distance over `u8_t` cells.
 *  @note Requires Arm NEON CPUs. The narrow cell width handles short pairs (max distance < 256) with @b no
 *        widening at all: 16 cells live in one register, so the recurrence is a single `vminq_u8` pass.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                              //
        u8_t const *first_reversed_slice, u8_t const *second_slice,            //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion, //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                     //
        uint8x16_t match_cost_vec, uint8x16_t mismatch_cost_vec, uint8x16_t gap_cost_vec) const noexcept {

        uint8x16_t equal_vec = vceqq_u8(vld1q_u8(first_reversed_slice), vld1q_u8(second_slice));
        uint8x16_t cost_vec = vbslq_u8(equal_vec, match_cost_vec, mismatch_cost_vec);
        uint8x16_t pre_substitution_vec = vld1q_u8(scores_pre_substitution);
        uint8x16_t pre_insert_vec = vld1q_u8(scores_pre_insertion);
        uint8x16_t pre_delete_vec = vld1q_u8(scores_pre_deletion);
        uint8x16_t cost_if_substitution_vec = vaddq_u8(pre_substitution_vec, cost_vec);
        uint8x16_t cost_if_gap_vec = vaddq_u8(vminq_u8(pre_insert_vec, pre_delete_vec), gap_cost_vec);
        vst1q_u8(scores_new, vminq_u8(cost_if_substitution_vec, cost_if_gap_vec));
    }

    SZ_INLINE void slice_1char(                                                //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,  //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion, //
        u8_t const *scores_pre_deletion, u8_t *scores_new, u8_t gap) const noexcept {
        u8_t const cost = first_reversed_slice[i] == second_slice[i] ? (u8_t)this->substituter_.match
                                                                     : (u8_t)this->substituter_.mismatch;
        u8_t const if_substitution = (u8_t)(scores_pre_substitution[i] + cost);
        u8_t const if_gap = (u8_t)(sz_min_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap);
        scores_new[i] = sz_min_of_two(if_substitution, if_gap);
    }

    SZ_NOINLINE void score_slice_trampoline_(                                             //
        u8_t const *first_reversed_slice, u8_t const *second_slice,                       //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,            //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                                //
        uint8x16_t match_cost_vec, uint8x16_t mismatch_cost_vec, uint8x16_t gap_cost_vec, //
        size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,
                          match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,           //
        u8_t const *scores_pre_deletion, u8_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed = (u8_t const *)first_reversed_slice;
        u8_t const *second = (u8_t const *)second_slice;
        u8_t const gap = static_cast<u8_t>(this->gap_costs_.open_or_extend);
        uint8x16_t const match_cost_vec = vdupq_n_u8((u8_t)this->substituter_.match);
        uint8x16_t const mismatch_cost_vec = vdupq_n_u8((u8_t)this->substituter_.mismatch);
        uint8x16_t const gap_cost_vec = vdupq_n_u8(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed, second, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_vec, mismatch_cost_vec, gap_cost_vec,
                                    from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed, second, i, scores_pre_substitution, scores_pre_insertion, scores_pre_deletion,
                        scores_new, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b affine-gap uniform-cost diagonal scorer - minimizes Levenshtein over `u8_t` cells.
 *  @note Requires Arm NEON CPUs. Gotoh recurrence with separate insertion/deletion gap planes, all in `u8`.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                               //
        u8_t const *first_reversed_slice, u8_t const *second_slice,             //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,  //
        u8_t const *scores_pre_deletion, u8_t const *scores_running_insertions, //
        u8_t const *scores_running_deletions, u8_t *scores_new,                 //
        u8_t *scores_new_insertions, u8_t *scores_new_deletions,                //
        uint8x16_t match_cost_vec, uint8x16_t mismatch_cost_vec, uint8x16_t gap_open_vec,
        uint8x16_t gap_extend_vec) const noexcept {

        uint8x16_t equal_vec = vceqq_u8(vld1q_u8(first_reversed_slice), vld1q_u8(second_slice));
        uint8x16_t cost_vec = vbslq_u8(equal_vec, match_cost_vec, mismatch_cost_vec);
        uint8x16_t pre_insert_open_vec = vld1q_u8(scores_pre_insertion);
        uint8x16_t pre_delete_open_vec = vld1q_u8(scores_pre_deletion);
        uint8x16_t run_insert_vec = vld1q_u8(scores_running_insertions);
        uint8x16_t run_delete_vec = vld1q_u8(scores_running_deletions);
        uint8x16_t cost_if_insert_vec = vminq_u8(vaddq_u8(pre_insert_open_vec, gap_open_vec),
                                                 vaddq_u8(run_insert_vec, gap_extend_vec));
        uint8x16_t cost_if_delete_vec = vminq_u8(vaddq_u8(pre_delete_open_vec, gap_open_vec),
                                                 vaddq_u8(run_delete_vec, gap_extend_vec));
        uint8x16_t cost_if_substitution_vec = vaddq_u8(vld1q_u8(scores_pre_substitution), cost_vec);
        uint8x16_t cell_score_vec = vminq_u8(vminq_u8(cost_if_insert_vec, cost_if_delete_vec),
                                             cost_if_substitution_vec);
        vst1q_u8(scores_new, cell_score_vec);
        vst1q_u8(scores_new_insertions, cost_if_insert_vec);
        vst1q_u8(scores_new_deletions, cost_if_delete_vec);
    }

    SZ_INLINE void slice_1char(                                                 //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t i,   //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,  //
        u8_t const *scores_pre_deletion, u8_t const *scores_running_insertions, //
        u8_t const *scores_running_deletions, u8_t *scores_new,                 //
        u8_t *scores_new_insertions, u8_t *scores_new_deletions,                //
        u8_t gap_open, u8_t gap_extend) const noexcept {
        u8_t const cost = first_reversed_slice[i] == second_slice[i] ? (u8_t)this->substituter_.match
                                                                     : (u8_t)this->substituter_.mismatch;
        u8_t const if_substitution = (u8_t)(scores_pre_substitution[i] + cost);
        u8_t const if_insertion = sz_min_of_two((u8_t)(scores_pre_insertion[i] + gap_open),
                                                (u8_t)(scores_running_insertions[i] + gap_extend));
        u8_t const if_deletion = sz_min_of_two((u8_t)(scores_pre_deletion[i] + gap_open),
                                               (u8_t)(scores_running_deletions[i] + gap_extend));
        scores_new[i] = sz_min_of_two(sz_min_of_two(if_insertion, if_deletion), if_substitution);
        scores_new_insertions[i] = if_insertion;
        scores_new_deletions[i] = if_deletion;
    }

    SZ_NOINLINE void score_slice_trampoline_(                                   //
        u8_t const *first_reversed_slice, u8_t const *second_slice,             //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,  //
        u8_t const *scores_pre_deletion, u8_t const *scores_running_insertions, //
        u8_t const *scores_running_deletions, u8_t *scores_new,                 //
        u8_t *scores_new_insertions, u8_t *scores_new_deletions,                //
        uint8x16_t match_cost_vec, uint8x16_t mismatch_cost_vec, uint8x16_t gap_open_vec, uint8x16_t gap_extend_vec,
        size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress,
                          scores_running_insertions + progress, scores_running_deletions + progress,
                          scores_new + progress, scores_new_insertions + progress, scores_new_deletions + progress,
                          match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_extend_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,           //
        u8_t const *scores_pre_deletion, u8_t const *scores_running_insertions,          //
        u8_t const *scores_running_deletions, u8_t *scores_new,                          //
        u8_t *scores_new_insertions, u8_t *scores_new_deletions,                         //
        executor_type_ &&executor = {}) noexcept {

        u8_t const *first_reversed = (u8_t const *)first_reversed_slice;
        u8_t const *second = (u8_t const *)second_slice;
        u8_t const gap_open = static_cast<u8_t>(this->gap_costs_.open);
        u8_t const gap_extend = static_cast<u8_t>(this->gap_costs_.extend);
        uint8x16_t const match_cost_vec = vdupq_n_u8((u8_t)this->substituter_.match);
        uint8x16_t const mismatch_cost_vec = vdupq_n_u8((u8_t)this->substituter_.mismatch);
        uint8x16_t const gap_open_vec = vdupq_n_u8(gap_open);
        uint8x16_t const gap_extend_vec = vdupq_n_u8(gap_extend);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed, second, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_running_insertions, scores_running_deletions,
                                    scores_new, scores_new_insertions, scores_new_deletions, match_cost_vec,
                                    mismatch_cost_vec, gap_open_vec, gap_extend_vec, from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed, second, i, scores_pre_substitution, scores_pre_insertion, scores_pre_deletion,
                        scores_running_insertions, scores_running_deletions, scores_new, scores_new_insertions,
                        scores_new_deletions, gap_open, gap_extend);

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief NEON @b UTF-8 uniform-cost diagonal scorer - minimizes rune-level Levenshtein over `u8_t` cells.
 *  @note Requires Arm NEON CPUs. Runes are 32-bit, so four `vceqq_u32` compares narrow to one `u8x16` match
 *        mask via `vmovn`/`vcombine` - the natural inverse of the cost widening the wider cells need.
 */
template <sz_capability_t capability_>
struct tile_scorer<rune_t const *, rune_t const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>>
    : public tile_scorer<rune_t const *, rune_t const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<rune_t const *, rune_t const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t step_k = 16;

    SZ_INLINE void slice_16chars(                                              //
        rune_t const *first_reversed_slice, rune_t const *second_slice,        //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion, //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                     //
        uint8x16_t match_cost_vec, uint8x16_t mismatch_cost_vec, uint8x16_t gap_cost_vec) const noexcept {

        u32_t const *first = (u32_t const *)first_reversed_slice;
        u32_t const *second = (u32_t const *)second_slice;
        uint16x8_t equal_lo = vcombine_u16(vmovn_u32(vceqq_u32(vld1q_u32(first + 0), vld1q_u32(second + 0))),
                                           vmovn_u32(vceqq_u32(vld1q_u32(first + 4), vld1q_u32(second + 4))));
        uint16x8_t equal_hi = vcombine_u16(vmovn_u32(vceqq_u32(vld1q_u32(first + 8), vld1q_u32(second + 8))),
                                           vmovn_u32(vceqq_u32(vld1q_u32(first + 12), vld1q_u32(second + 12))));
        uint8x16_t equal_vec = vcombine_u8(vmovn_u16(equal_lo), vmovn_u16(equal_hi));
        uint8x16_t cost_vec = vbslq_u8(equal_vec, match_cost_vec, mismatch_cost_vec);
        uint8x16_t cost_if_substitution_vec = vaddq_u8(vld1q_u8(scores_pre_substitution), cost_vec);
        uint8x16_t cost_if_gap_vec = vaddq_u8(vminq_u8(vld1q_u8(scores_pre_insertion), vld1q_u8(scores_pre_deletion)),
                                              gap_cost_vec);
        vst1q_u8(scores_new, vminq_u8(cost_if_substitution_vec, cost_if_gap_vec));
    }

    SZ_INLINE void slice_1char(                                                   //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t i, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,    //
        u8_t const *scores_pre_deletion, u8_t *scores_new, u8_t gap) const noexcept {
        u8_t const cost = first_reversed_slice[i] == second_slice[i] ? (u8_t)this->substituter_.match
                                                                     : (u8_t)this->substituter_.mismatch;
        u8_t const if_substitution = (u8_t)(scores_pre_substitution[i] + cost);
        u8_t const if_gap = (u8_t)(sz_min_of_two(scores_pre_insertion[i], scores_pre_deletion[i]) + gap);
        scores_new[i] = sz_min_of_two(if_substitution, if_gap);
    }

    SZ_NOINLINE void score_slice_trampoline_(                                             //
        rune_t const *first_reversed_slice, rune_t const *second_slice,                   //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,            //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                                //
        uint8x16_t match_cost_vec, uint8x16_t mismatch_cost_vec, uint8x16_t gap_cost_vec, //
        size_t from, size_t to) const noexcept {
        for (size_t idx_slice = from; idx_slice < to; ++idx_slice) {
            size_t const progress = idx_slice * step_k;
            slice_16chars(first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress,
                          scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,
                          match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                         //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t const length, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,               //
        u8_t const *scores_pre_deletion, u8_t *scores_new, executor_type_ &&executor = {}) noexcept {

        u8_t const gap = static_cast<u8_t>(this->gap_costs_.open_or_extend);
        uint8x16_t const match_cost_vec = vdupq_n_u8((u8_t)this->substituter_.match);
        uint8x16_t const mismatch_cost_vec = vdupq_n_u8((u8_t)this->substituter_.mismatch);
        uint8x16_t const gap_cost_vec = vdupq_n_u8(gap);

        size_t const count_slices = length / step_k;
        executor.for_slices(count_slices, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_vec, mismatch_cost_vec, gap_cost_vec,
                                    from, to);
        });
        for (size_t i = count_slices * step_k; i < length; ++i)
            slice_1char(first_reversed_slice, second_slice, i, scores_pre_substitution, scores_pre_insertion,
                        scores_pre_deletion, scores_new, gap);

        this->last_score_ = scores_new[length - 1];
    }
};

#pragma endregion // Uniform Cost Levenshtein

/**
 *  @brief NEON diagonal "walker" for class-based substitution costs with linear gaps. Mirrors the Ice Lake
 *         diagonal walker: classify both strings @b once into class-index buffers, fold the resident (32 x 32)
 *         cost table (possibly transposed after the shorter/longer swap) into `tile_scorer_t::prepare`, then
 *         sweep three rolling anti-diagonals through the matrix.
 */
template <typename score_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct diagonal_walker<char, score_type_, error_costs_32x32_t, linear_gap_costs_t, objective_, locality_, sz_cap_neon_k,
                       void> {

    using char_t = char;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t step_classes_k = 16;

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

        // Make sure the size relation between the strings is correct.
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
        classify_into_(scorer.lookup_, substituter_.byte_to_class, shorter_reversed, shorter_length,
                       shorter_reversed_classes);
        classify_into_(scorer.lookup_, substituter_.byte_to_class, longer, longer_length, longer_classes);

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

  private:
    /** @brief Maps a raw byte string into class bytes using the resident `byte_to_class` lookup, @b amortized. */
    static void classify_into_(substitution_lookup_neon_t const &lookup, u8_t const *byte_to_class,
                               char_t const *source, size_t length, char_t *classes) noexcept {
        size_t progress = 0;
        for (; progress + step_classes_k <= length; progress += step_classes_k) {
            uint8x16_t source_vec = vld1q_u8((u8_t const *)(source + progress));
            vst1q_u8((u8_t *)(classes + progress), lookup.classify16(source_vec));
        }
        for (; progress < length; ++progress) classes[progress] = (char_t)byte_to_class[(u8_t)source[progress]];
    }
};

/**
 *  @brief NEON @b affine-gap diagonal "walker" for class-based substitution costs. Mirrors the linear NEON
 *         diagonal walker, but threads the separate insertion and deletion gap diagonals of the Gotoh
 *         recurrence (open vs extend) through the affine scorer's 5-in/3-out call signature, keeping 7 rolling
 *         anti-diagonals (3 main + 2 insert + 2 delete).
 */
template <typename score_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct diagonal_walker<char, score_type_, error_costs_32x32_t, affine_gap_costs_t, objective_, locality_, sz_cap_neon_k,
                       void> {

    using char_t = char;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t step_classes_k = 16;

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

        // Make sure the size relation between the strings is correct.
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
        classify_into_(scorer.lookup_, substituter_.byte_to_class, shorter_reversed, shorter_length,
                       shorter_reversed_classes);
        classify_into_(scorer.lookup_, substituter_.byte_to_class, longer, longer_length, longer_classes);

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

  private:
    /** @brief Maps a raw byte string into class bytes using the resident `byte_to_class` lookup, @b amortized. */
    static void classify_into_(substitution_lookup_neon_t const &lookup, u8_t const *byte_to_class,
                               char_t const *source, size_t length, char_t *classes) noexcept {
        size_t progress = 0;
        for (; progress + step_classes_k <= length; progress += step_classes_k) {
            uint8x16_t source_vec = vld1q_u8((u8_t const *)(source + progress));
            vst1q_u8((u8_t *)(classes + progress), lookup.classify16(source_vec));
        }
        for (; progress < length; ++progress) classes[progress] = (char_t)byte_to_class[(u8_t)source[progress]];
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score between two strings using the NEON backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sn_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 3;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_neon_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_neon_k>;
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

        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score with @b affine gaps using the NEON backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sn_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 7;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_neon_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_neon_k>;
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

        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief Computes the @b byte-level Smith-Waterman score between two strings using the NEON backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sn_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 3;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, linear_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_neon_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, linear_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_neon_k>;
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

        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief Computes the @b byte-level Smith-Waterman score with @b affine gaps using the NEON backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sn_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 7;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, affine_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_neon_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, affine_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_neon_k>;
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

        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

#pragma region NEON Inter-Sequence Candidate Lanes

/**
 *  @brief Inter-sequence NEON walker: one query against up to 16 candidates packed one-per-lane, 8-bit cells.
 *
 *  Computes the @b global unit-cost Levenshtein distance of a single shared query against a transposed
 *  `candidate_lanes_block_t` of up to 16 candidates. Each `uint8x16_t` holds 16 `u8` cells - lane @p lane_index
 *  carries that candidate's running Dynamic Programming column. The query characters index the rows; for every
 *  row the candidate column is broadcast-compared against the query character and the SWIPE recurrence
 *  `cell = min(substitution, min(deletion, insertion))` advances all 16 lanes in lockstep over `vminq_u8`.
 *
 *  @note The cells are `u8`, so distances saturate at 255: this kernel is only valid when the query and every
 *      candidate are at most 255 characters long. Enforcing that bound is the caller's dispatch contract; the
 *      kernel performs no runtime length check.
 *  @note Requires Arm NEON CPUs. Structural twin of the Ice Lake 64-lane `u8` candidate-lane walker, narrowed
 *      to the 16-wide NEON byte vector.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u8_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 16, void> {

    using char_t = char;
    using score_t = u8_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 16;

    // The `u8` lane recurrence hardcodes `vminq_u8`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 8-bit candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (16 `u8` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1);
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

        uint8x16_t const one_vec = vdupq_n_u8(1);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            vst1q_u8(previous_row + column * candidate_lanes_k, vdupq_n_u8(static_cast<u8_t>(column)));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint8x16_t const query_char_vec = vdupq_n_u8(static_cast<u8_t>(query[query_position - 1]));
            vst1q_u8(current_row, vdupq_n_u8(static_cast<u8_t>(query_position)));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x16_t const candidate_chars_vec = vld1q_u8((u8_t const *)candidates.position(column - 1));
                uint8x16_t const diagonal_vec = vld1q_u8(previous_row + (column - 1) * candidate_lanes_k);
                uint8x16_t const deletion_source_vec = vld1q_u8(previous_row + column * candidate_lanes_k);
                uint8x16_t const insertion_source_vec = vld1q_u8(current_row + (column - 1) * candidate_lanes_k);

                // `vceqq_u8` yields 0xFF on match; its inverse is 0xFF on mismatch, and subtracting that mask
                // (modular `- 0xFF == + 1`) adds 1 to the diagonal only on a mismatch.
                uint8x16_t const mismatch_mask = vmvnq_u8(vceqq_u8(query_char_vec, candidate_chars_vec));
                uint8x16_t const cost_if_substitution_vec = vsubq_u8(diagonal_vec, mismatch_mask);
                uint8x16_t const cost_if_deletion_vec = vaddq_u8(deletion_source_vec, one_vec);
                uint8x16_t const cost_if_insertion_vec = vaddq_u8(insertion_source_vec, one_vec);
                uint8x16_t const cell_score_vec =
                    vminq_u8(cost_if_substitution_vec, vminq_u8(cost_if_deletion_vec, cost_if_insertion_vec));
                vst1q_u8(current_row + column * candidate_lanes_k, cell_score_vec);
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
 *  @brief Inter-sequence NEON walker: one query against up to 8 candidates packed one-per-lane, 16-bit cells.
 *
 *  Structural twin of the 16-lane `u8` walker, widened to `u16` cells so each `uint16x8_t` holds 8 lanes. The
 *  candidate characters remain `char`/`u8`: 8 of them load as a 64-bit `uint8x8_t` and compare against the
 *  broadcast query character, the 8 `u8` match flags widen to 8 `u16` lanes that drive the substitution add.
 *  The SWIPE recurrence `cell = min(substitution, min(deletion, insertion))` advances all 8 lanes over `vminq_u16`.
 *
 *  @note The cells are `u16`, so distances saturate at 65535: this kernel is only valid when the query and every
 *      candidate are at most 65535 characters long. Enforcing that bound is the caller's dispatch contract; the
 *      kernel performs no runtime length check.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 8, void> {

    using char_t = char;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The `u16` lane recurrence hardcodes `vminq_u16`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 16-bit candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (8 `u16` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 8 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 8`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        uint16x8_t const one_vec = vdupq_n_u16(1);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            vst1q_u16(previous_row + column * candidate_lanes_k, vdupq_n_u16(static_cast<u16_t>(column)));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint8x8_t const query_char_vec = vdup_n_u8(static_cast<u8_t>(query[query_position - 1]));
            vst1q_u16(current_row, vdupq_n_u16(static_cast<u16_t>(query_position)));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x8_t const candidate_chars_vec = vld1_u8((u8_t const *)candidates.position(column - 1));
                uint16x8_t const diagonal_vec = vld1q_u16(previous_row + (column - 1) * candidate_lanes_k);
                uint16x8_t const deletion_source_vec = vld1q_u16(previous_row + column * candidate_lanes_k);
                uint16x8_t const insertion_source_vec = vld1q_u16(current_row + (column - 1) * candidate_lanes_k);

                // Reduce the 8 `0xFF`/`0x00` match flags to a 1-on-mismatch increment, widen to 8 `u16` lanes,
                // and add it to the diagonal (the modular `- 0xFF` trick is `u8`-only, so we add explicitly here).
                uint8x8_t const mismatch_increment_u8 = vshr_n_u8(vmvn_u8(vceq_u8(query_char_vec, candidate_chars_vec)), 7);
                uint16x8_t const mismatch_increment = vmovl_u8(mismatch_increment_u8);
                uint16x8_t const cost_if_substitution_vec = vaddq_u16(diagonal_vec, mismatch_increment);
                uint16x8_t const cost_if_deletion_vec = vaddq_u16(deletion_source_vec, one_vec);
                uint16x8_t const cost_if_insertion_vec = vaddq_u16(insertion_source_vec, one_vec);
                uint16x8_t const cell_score_vec =
                    vminq_u16(cost_if_substitution_vec, vminq_u16(cost_if_deletion_vec, cost_if_insertion_vec));
                vst1q_u16(current_row + column * candidate_lanes_k, cell_score_vec);
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
 *  @brief Inter-sequence NEON walker: one query against up to 8 candidates packed one-per-lane, weighted
 *         Needleman-Wunsch with 32-class substitution costs and a linear gap, 16-bit signed cells.
 *
 *  Structural twin of the 8-lane uniform `u16` walker, but the unit `cmpeq -> +1` substitution term is replaced by
 *  a two-level class lookup against the resident (32 x 32) `class_substitution_costs` matrix folded into
 *  `substitution_lookup_neon_t`, and the objective is @b maximization (Needleman-Wunsch), so the recurrence is
 *  `cell = max(diag + substitution, max(up, left) + gap)` over signed `i16` cells with a (typically negative) gap.
 *
 *  The 8 candidate characters (one per lane) are loop-invariant across rows, so their classes are computed @b once
 *  before the row loop via `classify16` and cached. Inside the column loop the cached candidate-class vector and the
 *  broadcast query-class vector index the resident cost matrix with a single `lookup16`, yielding 16 signed `i8`
 *  substitution costs whose low 8 lanes sign-extend to 8 `i16` cells.
 *
 *  @note Signed `i16` cells; valid while every reachable score stays within `[-32768, 32767]` (caller's contract).
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 8, void> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The signed `i16` recurrence hardcodes `vmaxq_s16`; minimization would need a different blend.
    static_assert(objective_ == sz_maximize_score_k,
                  "The weighted candidate-lane kernel only implements score maximization (Needleman-Wunsch).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (8 `i16` cells each), plus one
     *      cached candidate-class lane-vector (8 `u8`) per candidate position.
     */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        size_t const class_bytes = candidate_lanes_k * longest_candidate * sizeof(u8_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous row
        amount += score_row_bytes; // current row
        amount += class_bytes;     // cached candidate classes
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 8 candidates (see `candidate_lanes_block_t`).
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

        // Two `i16` score row buffers, then a `u8` candidate-class cache, all carved from the byte span.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        u8_t *candidate_classes = reinterpret_cast<u8_t *>(current_row + row_stride);

        // The (32 x 32) cost matrix is symmetric across the two operand orderings here, so no transpose is needed:
        // the query class is the first operand and the candidate class the second, matching `class_substitution_costs`.
        substitution_lookup_neon_t lookup;
        lookup.reload_classes(substituter_.byte_to_class);
        lookup.reload_costs(substituter_.class_substitution_costs, false);

        // Pre-classify every candidate column once; the 8 candidate bytes per column are loop-invariant in rows.
        // Only the low 8 lanes carry live candidates, so we classify a 16-lane vector and store its low half.
        for (size_t column = 0; column < longest_candidate; ++column) {
            uint8x8_t const candidate_chars_low = vld1_u8((u8_t const *)candidates.position(column));
            uint8x16_t const candidate_chars_vec = vcombine_u8(candidate_chars_low, vdup_n_u8(0));
            uint8x16_t const candidate_classes_vec = lookup.classify16(candidate_chars_vec);
            vst1_u8(candidate_classes + column * candidate_lanes_k, vget_low_u8(candidate_classes_vec));
        }

        int16x8_t const gap_vec = vdupq_n_s16(static_cast<i16_t>(gap));

        // Row 0: the empty query prefix against every candidate prefix is a run of `gap * column`, identical across
        // lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            vst1q_s16(previous_row + column * candidate_lanes_k,
                      vdupq_n_s16(static_cast<i16_t>(gap * (i16_t)column)));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            // The query character is fixed for the whole row, so its class is scalar and broadcast across lanes.
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            uint8x16_t const query_class_vec = vdupq_n_u8(query_class);

            vst1q_s16(current_row, vdupq_n_s16(static_cast<i16_t>(gap * (i16_t)query_position)));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x8_t const candidate_classes_low =
                    vld1_u8(candidate_classes + (column - 1) * candidate_lanes_k);
                uint8x16_t const candidate_classes_vec = vcombine_u8(candidate_classes_low, vdup_n_u8(0));
                int16x8_t const diagonal_vec = vld1q_s16(previous_row + (column - 1) * candidate_lanes_k);
                int16x8_t const up_vec = vld1q_s16(previous_row + column * candidate_lanes_k);
                int16x8_t const left_vec = vld1q_s16(current_row + (column - 1) * candidate_lanes_k);

                // Gather 16 substitution costs and sign-extend the low 8 `i8` lanes into 8 `i16` cells.
                int8x16_t const cost_i8_vec = lookup.lookup16(query_class_vec, candidate_classes_vec);
                int16x8_t const cost_i16_vec = vmovl_s8(vget_low_s8(cost_i8_vec));

                int16x8_t const cost_if_substitution_vec = vaddq_s16(diagonal_vec, cost_i16_vec);
                int16x8_t const cost_if_gap_vec = vaddq_s16(vmaxq_s16(up_vec, left_vec), gap_vec);
                int16x8_t const cell_score_vec = vmaxq_s16(cost_if_substitution_vec, cost_if_gap_vec);
                vst1q_s16(current_row + column * candidate_lanes_k, cell_score_vec);
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
 *  @brief Inter-sequence NEON walker: one query against up to 8 candidates packed one-per-lane, weighted
 *         @b Smith-Waterman (local) alignment with 32-class substitution costs and a linear gap, 16-bit signed cells.
 *
 *  Local sibling of the weighted Needleman-Wunsch walker: the recurrence clamps to zero,
 *  `cell = max(0, diag + substitution, max(up, left) + gap)`, the boundary row and column are zero (an alignment may
 *  start anywhere), and the score is the maximum cell over the whole matrix rather than the bottom-right corner. A
 *  per-lane running maximum accumulates that, updated only for columns within each lane's own candidate length so a
 *  shorter candidate's zero-padded tail columns cannot inflate its score.
 *
 *  @note Signed `i16` cells; valid while every reachable score stays within `[0, 32767]` (caller's contract).
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, linear_gap_costs_t, objective_,
                             sz_similarity_local_k, sz_cap_neon_k, 8, void> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 8;

    static_assert(objective_ == sz_maximize_score_k,
                  "The weighted local candidate-lane kernel only implements score maximization (Smith-Waterman).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch: two `i16` score rows of `longest_candidate + 1` lane-vectors plus one `u8` candidate-class
     *      cache per candidate position (same layout as the global weighted walker). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        size_t const class_bytes = candidate_lanes_k * longest_candidate * sizeof(u8_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes;
        amount += score_row_bytes;
        amount += class_bytes;
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 8 candidates (see `candidate_lanes_block_t`).
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
        u8_t *candidate_classes = reinterpret_cast<u8_t *>(current_row + row_stride);

        substitution_lookup_neon_t lookup;
        lookup.reload_classes(substituter_.byte_to_class);
        lookup.reload_costs(substituter_.class_substitution_costs, false);

        for (size_t column = 0; column < longest_candidate; ++column) {
            uint8x8_t const candidate_chars_low = vld1_u8((u8_t const *)candidates.position(column));
            uint8x16_t const candidate_chars_vec = vcombine_u8(candidate_chars_low, vdup_n_u8(0));
            uint8x16_t const candidate_classes_vec = lookup.classify16(candidate_chars_vec);
            vst1_u8(candidate_classes + column * candidate_lanes_k, vget_low_u8(candidate_classes_vec));
        }

        int16x8_t const gap_vec = vdupq_n_s16(static_cast<i16_t>(gap));
        int16x8_t const zero_vec = vdupq_n_s16(0);

        // Per-lane candidate lengths gate the running-maximum so a shorter candidate's padded columns are excluded.
        alignas(16) i16_t lane_lengths[candidate_lanes_k] = {0};
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
            lane_lengths[lane_index] = static_cast<i16_t>(candidates.lengths[lane_index]);
        int16x8_t const lane_lengths_vec = vld1q_s16(lane_lengths);
        int16x8_t running_max_vec = zero_vec;

        // Local alignment: the boundary row and column are zero (an alignment may begin at any cell).
        for (size_t column = 0; column <= longest_candidate; ++column)
            vst1q_s16(previous_row + column * candidate_lanes_k, zero_vec);

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            uint8x16_t const query_class_vec = vdupq_n_u8(query_class);

            vst1q_s16(current_row, zero_vec);
            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x8_t const candidate_classes_low =
                    vld1_u8(candidate_classes + (column - 1) * candidate_lanes_k);
                uint8x16_t const candidate_classes_vec = vcombine_u8(candidate_classes_low, vdup_n_u8(0));
                int16x8_t const diagonal_vec = vld1q_s16(previous_row + (column - 1) * candidate_lanes_k);
                int16x8_t const up_vec = vld1q_s16(previous_row + column * candidate_lanes_k);
                int16x8_t const left_vec = vld1q_s16(current_row + (column - 1) * candidate_lanes_k);

                int8x16_t const cost_i8_vec = lookup.lookup16(query_class_vec, candidate_classes_vec);
                int16x8_t const cost_i16_vec = vmovl_s8(vget_low_s8(cost_i8_vec));

                int16x8_t const cost_if_substitution_vec = vaddq_s16(diagonal_vec, cost_i16_vec);
                int16x8_t const cost_if_gap_vec = vaddq_s16(vmaxq_s16(up_vec, left_vec), gap_vec);
                int16x8_t const cell_score_vec =
                    vmaxq_s16(zero_vec, vmaxq_s16(cost_if_substitution_vec, cost_if_gap_vec));
                vst1q_s16(current_row + column * candidate_lanes_k, cell_score_vec);

                // Fold this column into the running maximum only for lanes whose candidate reaches it.
                int16x8_t const column_live = vcgtq_s16(lane_lengths_vec, vdupq_n_s16(static_cast<i16_t>(column - 1)));
                int16x8_t const masked_cell_vec = vandq_s16(cell_score_vec, column_live);
                running_max_vec = vmaxq_s16(running_max_vec, masked_cell_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        alignas(16) i16_t final_max[candidate_lanes_k];
        vst1q_s16(final_max, running_max_vec);
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
            result_lanes[lane_index] = final_max[lane_index];
        return status_t::success_k;
    }
};

/**
 *  @brief Batched byte-level @b Levenshtein distances on NEON, packing a shared query's candidates into the
 *      unit-cost `candidate_lane_walker` (16-lane `u8` for short cells, 8-lane `u16` for medium cells) and falling
 *      back to the per-pair anti-diagonal Dynamic Programming scorer for the long tail (distances that escape `u16`).
 *
 *  Mirrors the Ice Lake `levenshtein_distances` cross-product engine's structure - live cells walked query-major,
 *  grouped into shared-query blocks, transposed into a reusable scratch buffer (the column-major layout the lane
 *  walker reads), scored, and scattered into the strided result matrix (plus the mirror slot for symmetric
 *  self-similarity) - but where Ice Lake dispatches the bit-parallel Myers family, NEON has no vectorized Myers and
 *  instead reaches for the SWIPE-style row-walking lane walker, which is the NEON inter-sequence win. Non-unit
 *  substitution or gap costs route to the serial per-pair cross-product helpers.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t lanes_u8_k = 16;  // ? 16 candidates per block while distances fit `u8`.
    static constexpr size_t lanes_u16_k = 8;  // ? 8 candidates per block while distances fit `u16`.
    static constexpr size_t fits_u8_k = 254;  // ? Both lengths within this keep the worst-case distance under 255.
    static constexpr size_t fits_u16_k = 512; // ? Tier limit before falling back to the per-pair DP scorer.

    using scoring_t = levenshtein_distance<char, gap_costs_t, capability_k>; // ? Per-pair DP fallback.
    using lane_u8_t = candidate_lane_walker<char, u8_t, uniform_substitution_costs_t, gap_costs_t,
                                            sz_minimize_distance_k, sz_similarity_global_k, sz_cap_neon_k,
                                            (int)lanes_u8_k, void>;
    using lane_u16_t = candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, gap_costs_t,
                                             sz_minimize_distance_k, sz_similarity_global_k, sz_cap_neon_k,
                                             (int)lanes_u16_k, void>;
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

    /** @brief Which lane tier a `(query, candidate)` cell fits: 1 for the `u8` block, 2 for `u16`, 0 for the DP tail. */
    static int lane_tier_(size_t query_length, size_t candidate_length) noexcept {
        size_t const shorter = sz_min_of_two(query_length, candidate_length);
        size_t const longer = sz_max_of_two(query_length, candidate_length);
        if (shorter == 0) return 0;
        if (longer <= fits_u8_k) return 1;
        if (longer <= fits_u16_k) return 2;
        return 0;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the transposed-block plus the
     *      `u16` lane-walker arena for the longest candidate (a superset of the `u8` arena), or the anti-diagonal DP
     *      fallback for the longest query x longest candidate.
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
        lane_u16_t lane_walker {substituter_, gap_costs_};
        size_t const transpose_bytes = lanes_u8_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size() && sz_min_of_two(longest_query, longest_candidate) > fits_u16_k) {
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

    /** @brief Transposes a shared-query block of candidates into the zero-padded column-major lane layout, then runs
     *      the matching lane walker and scatters its per-lane distances into the destination cells. */
    template <typename candidates_type_, typename value_type_, typename lane_walker_type_>
    status_t score_block_(span<char_t const> query, candidates_type_ const &candidates, size_t candidates_count,
                          cross_similarities_t cross_kind, size_t block_begin, index_t lanes_count, size_t lane_capacity,
                          size_t block_longest, char_t *transposed, scratch_space_t walker_scratch_space,
                          size_t const *lengths, cross_cell_destination_<value_type_> const *destinations,
                          lane_walker_type_ &lane_walker, cpu_specs_t const &specs) const noexcept {
        using lane_score_t = typename lane_walker_type_::score_t;
        for (size_t position = 0; position != lane_capacity * block_longest; ++position) transposed[position] = 0;
        for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index) {
            size_t fill_query_index = 0, fill_candidate_index = 0;
            cell_to_indices_(block_begin + lane_index, candidates_count, cross_kind, fill_query_index,
                             fill_candidate_index);
            auto const lane_candidate = to_view(candidates[fill_candidate_index]);
            for (size_t position = 0; position < lane_candidate.size(); ++position)
                transposed[position * lane_capacity + lane_index] = lane_candidate[position];
        }

        candidate_lanes_block_t<char_t> block;
        block.transposed = transposed;
        block.lane_capacity = lane_capacity;
        block.lanes_count = lanes_count;
        block.lengths = lengths;
        block.longest_candidate = block_longest;

        lane_score_t result_lanes[lanes_u8_k];
        if (status_t status = lane_walker(query, block, result_lanes, walker_scratch_space, specs);
            status != status_t::success_k)
            return status;
        for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index) {
            *destinations[lane_index].primary = static_cast<value_type_>(result_lanes[lane_index]);
            if (destinations[lane_index].mirror)
                *destinations[lane_index].mirror = static_cast<value_type_>(result_lanes[lane_index]);
        }
        return status_t::success_k;
    }

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the shared-query lane walker,
     *      falling back to the anti-diagonal Dynamic Programming scorer for the long tail. Each cell scores
     *      `dist(query, candidate)` and writes it into the strided @p results matrix (plus the mirror slot for
     *      symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    [[gnu::noinline]] status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                            results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                            size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        scoring_t dp {substituter_, gap_costs_};
        lane_u8_t lane_u8 {substituter_, gap_costs_};
        lane_u16_t lane_u16 {substituter_, gap_costs_};

        // Scan the slice for its longest lane-eligible candidate to carve the transpose/walker sub-arenas; the
        // global-worst sizing (`worst_cell_scratch_`) guarantees these offsets stay in bounds.
        size_t longest_candidate = 0;
        for (size_t cell_index = cell_begin; cell_index != cell_end; ++cell_index) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            size_t const query_length = to_view(queries[query_index]).size();
            size_t const candidate_length = to_view(candidates[candidate_index]).size();
            if (lane_tier_(query_length, candidate_length) != 0)
                longest_candidate = sz_max_of_two(longest_candidate, candidate_length);
        }

        size_t const transpose_bytes = lanes_u8_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_u16.scratch_space_needed(longest_candidate, specs) : 0;
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

        auto const scatter = [&](cross_cell_destination_<value_t> const &destination, size_t score) noexcept {
            *destination.primary = static_cast<value_t>(score);
            if (destination.mirror) *destination.mirror = static_cast<value_t>(score);
        };

        dummy_executor_t dummy;
        size_t lengths[lanes_u8_k];
        cross_cell_destination_<value_t> destinations[lanes_u8_k];
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const query_length = query.size(), candidate_length = candidate.size();
            int const tier = lane_tier_(query_length, candidate_length);

            // Empty cell: the only alignment is all-gaps, so the distance is `max(query, candidate)` length.
            if (query_length == 0 || candidate_length == 0) {
                scatter(destination_for(query_index, candidate_index), sz_max_of_two(query_length, candidate_length));
                ++cell_index;
                continue;
            }

            // Long tail: any cell whose distance can escape `u16` walks the per-pair anti-diagonal scorer.
            if (tier == 0) {
                size_t result_score = 0;
                if (status_t status = dp(query, candidate, result_score, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_score);
                ++cell_index;
                continue;
            }

            // Seed a shared-query block with this cell, then extend over consecutive same-query candidates that
            // share the same lane tier, up to that tier's lane capacity. Seeding first guarantees forward progress.
            size_t const lane_capacity = tier == 1 ? lanes_u8_k : lanes_u16_k;
            size_t const seed_query_index = query_index;
            size_t const block_begin = cell_index;
            size_t block_longest = candidate_length;
            lengths[0] = candidate_length;
            destinations[0] = destination_for(query_index, candidate_index);
            index_t lanes_count = 1;
            ++cell_index;
            for (; cell_index != cell_end && lanes_count != lane_capacity; ++cell_index, ++lanes_count) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                if (next_query_index != seed_query_index) break;
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                if (lane_tier_(query_length, next_candidate.size()) != tier) break;
                lengths[lanes_count] = next_candidate.size();
                destinations[lanes_count] = destination_for(next_query_index, next_candidate_index);
                block_longest = sz_max_of_two(block_longest, next_candidate.size());
            }

            status_t status = tier == 1
                                  ? score_block_(query, candidates, candidates_count, cross_kind, block_begin,
                                                 lanes_count, lane_capacity, block_longest, transposed,
                                                 walker_scratch_space, lengths, destinations, lane_u8, specs)
                                  : score_block_(query, candidates, candidates_count, cross_kind, block_begin,
                                                 lanes_count, lane_capacity, block_longest, transposed,
                                                 walker_scratch_space, lengths, destinations, lane_u16, specs);
            if (status != status_t::success_k) return status;
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
        if (!is_unit_cost_())
            return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                               cross_similarities_t::all_pairs_k, score_scratch_, specs);
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
        if (!is_unit_cost_())
            return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                              cross_similarities_t::all_pairs_k, score_scratch_,
                                              std::forward<executor_type_>(executor), specs);
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                               cross_similarities_t::symmetric_k, score_scratch_, specs);
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
        if (!is_unit_cost_())
            return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                              cross_similarities_t::symmetric_k, score_scratch_,
                                              std::forward<executor_type_>(executor), specs);
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion - Public Cross-Product Overloads
};

/**
 *  @brief Batched byte-level @b weighted Needleman-Wunsch scores on NEON, packing up to 8 candidates of a shared
 *      query into the signed-`i16` `candidate_lane_walker` and falling back to the per-pair anti-diagonal Dynamic
 *      Programming scorer for the long tail (scores that escape `i16`, or empty cells).
 *
 *  Mirrors the Ice Lake `needleman_wunsch_scores` cross-product engine, narrowed to the 8-wide NEON `i16` lane.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
                               std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 8;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, sz_caps_sn_k>; // ? Per-pair DP fallback.
    using lane_walker_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_neon_k, (int)candidate_lanes_k, void>; // ? NEON shared-query lanes.

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
     *      arena for the longest candidate, or the anti-diagonal DP fallback for the longest query x longest candidate.
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
     *      falling back to the anti-diagonal Dynamic Programming scorer for the long tail. Writes each
     *      `score(query, candidate)` into the strided @p results matrix (plus the mirror slot for symmetric cells).
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
 *  @brief Batched byte-level @b weighted Smith-Waterman (local) scores on NEON, packing up to 8 candidates of a
 *      shared query into the signed-`i16` local `candidate_lane_walker` and falling back to the per-pair anti-diagonal
 *      Dynamic Programming scorer for the long tail (scores that escape `i16`).
 *
 *  Local sibling of the NEON `needleman_wunsch_scores` cross-product engine: identical query-major grouping,
 *  transpose, and scatter, but it dispatches the @b local lane walker (zero-clamped recurrence, score = the maximum
 *  cell of the matrix) and an empty `(query, candidate)` cell scores @b 0 (a local alignment may align nothing).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 8;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, sz_caps_sn_k>; // ? Per-pair DP fallback.
    using lane_walker_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_neon_k, (int)candidate_lanes_k, void>; // ? NEON shared-query local lanes.

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
     *      arena for the longest candidate, or the anti-diagonal DP fallback for the longest query x longest candidate.
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_NEON_HPP_
