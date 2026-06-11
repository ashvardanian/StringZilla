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

/** @brief Serial fallback combined with the NEON backend, mirroring `sz_caps_sil_k` for Ice Lake. */
static constexpr sz_capability_t sz_caps_sn_k = (sz_capability_t)(sz_cap_serial_k | sz_cap_neon_k);

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
        cost_of_substitution_i16_vecs[1] = vmovl_s8(vget_high_s8(cost_of_substitution_i8_vec));

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
        cost_of_substitution_i16_vecs[1] = vmovl_s8(vget_high_s8(cost_of_substitution_i8_vec));

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
        int16x8_t cost_high_i16 = vmovl_s8(vget_high_s8(cost_of_substitution_i8_vec));
        int32x4_t cost_of_substitution_i32_vecs[4];
        cost_of_substitution_i32_vecs[0] = vmovl_s16(vget_low_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[1] = vmovl_s16(vget_high_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[2] = vmovl_s16(vget_low_s16(cost_high_i16));
        cost_of_substitution_i32_vecs[3] = vmovl_s16(vget_high_s16(cost_high_i16));

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
        int16x8_t cost_high_i16 = vmovl_s8(vget_high_s8(cost_of_substitution_i8_vec));
        int32x4_t cost_of_substitution_i32_vecs[4];
        cost_of_substitution_i32_vecs[0] = vmovl_s16(vget_low_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[1] = vmovl_s16(vget_high_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[2] = vmovl_s16(vget_low_s16(cost_high_i16));
        cost_of_substitution_i32_vecs[3] = vmovl_s16(vget_high_s16(cost_high_i16));

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
        cost_of_substitution_i16_vecs[1] = vmovl_s8(vget_high_s8(cost_of_substitution_i8_vec));

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
        cost_of_substitution_i16_vecs[1] = vmovl_s8(vget_high_s8(cost_of_substitution_i8_vec));

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
        int16x8_t cost_high_i16 = vmovl_s8(vget_high_s8(cost_of_substitution_i8_vec));
        int32x4_t cost_of_substitution_i32_vecs[4];
        cost_of_substitution_i32_vecs[0] = vmovl_s16(vget_low_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[1] = vmovl_s16(vget_high_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[2] = vmovl_s16(vget_low_s16(cost_high_i16));
        cost_of_substitution_i32_vecs[3] = vmovl_s16(vget_high_s16(cost_high_i16));

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
        int16x8_t cost_high_i16 = vmovl_s8(vget_high_s8(cost_of_substitution_i8_vec));
        int32x4_t cost_of_substitution_i32_vecs[4];
        cost_of_substitution_i32_vecs[0] = vmovl_s16(vget_low_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[1] = vmovl_s16(vget_high_s16(cost_low_i16));
        cost_of_substitution_i32_vecs[2] = vmovl_s16(vget_low_s16(cost_high_i16));
        cost_of_substitution_i32_vecs[3] = vmovl_s16(vget_high_s16(cost_high_i16));

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
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(score_t) * max_diagonal_length, specs.cache_line_width) / sizeof(score_t);

        // Beyond the 3 score diagonals, we keep the reversed shorter string and both class-index buffers.
        // Each class-index buffer is padded by a step_classes_k-byte SIMD overread guard.
        size_t const padded_shorter_stream_length = round_up_to_multiple(shorter_length + step_classes_k,
                                                                         specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + step_classes_k,
                                                                        specs.cache_line_width);
        size_t const scratch_required = sizeof(score_t) * padded_diagonal_length * 3 +
                                        padded_shorter_stream_length * 2 + padded_longer_stream_length;
        if (scratch_space.size() < scratch_required) return status_t::bad_alloc_k;

        score_t *previous_scores = (score_t *)scratch_space.data();
        score_t *current_scores = previous_scores + padded_diagonal_length;
        score_t *next_scores = current_scores + padded_diagonal_length;
        char_t *const shorter_reversed = (char_t *)(next_scores + padded_diagonal_length);
        char_t *const shorter_reversed_classes = shorter_reversed + padded_shorter_stream_length;
        char_t *const longer_classes = shorter_reversed_classes + padded_shorter_stream_length;

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
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(score_t) * max_diagonal_length, specs.cache_line_width) / sizeof(score_t);

        // Beyond the 7 score diagonals, we keep the reversed shorter string and both class-index buffers.
        // Each class-index buffer is padded by a step_classes_k-byte SIMD overread guard.
        size_t const padded_shorter_stream_length = round_up_to_multiple(shorter_length + step_classes_k,
                                                                         specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + step_classes_k,
                                                                        specs.cache_line_width);
        size_t const scratch_required = sizeof(score_t) * padded_diagonal_length * 7 +
                                        padded_shorter_stream_length * 2 + padded_longer_stream_length;
        if (scratch_space.size() < scratch_required) return status_t::bad_alloc_k;

        score_t *previous_scores = (score_t *)scratch_space.data();
        score_t *current_scores = previous_scores + padded_diagonal_length;
        score_t *next_scores = current_scores + padded_diagonal_length;
        score_t *current_inserts = next_scores + padded_diagonal_length;
        score_t *next_inserts = current_inserts + padded_diagonal_length;
        score_t *current_deletes = next_inserts + padded_diagonal_length;
        score_t *next_deletes = current_deletes + padded_diagonal_length;
        char_t *const shorter_reversed = (char_t *)(next_deletes + padded_diagonal_length);
        char_t *const shorter_reversed_classes = shorter_reversed + padded_shorter_stream_length;
        char_t *const longer_classes = shorter_reversed_classes + padded_shorter_stream_length;

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
                        scratch_space_t scratch_space, executor_type_ &&executor,
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
                        scratch_space_t scratch_space, executor_type_ &&executor,
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
                        scratch_space_t scratch_space, executor_type_ &&executor,
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
                        scratch_space_t scratch_space, executor_type_ &&executor,
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
