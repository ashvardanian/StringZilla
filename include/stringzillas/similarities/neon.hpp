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

/** @brief Per-lane nonzero test: an all-ones lane where any bit of @p value is set, else all-zeros. */
SZ_INLINE uint64x2_t lane_nonzero_(uint64x2_t value) noexcept { return vcgtq_u64(value, vdupq_n_u64(0)); }

#pragma region Bit Parallel Myers

/**
 *  @brief NEON Myers/Hyyr\xc3\xb6 unit-cost Levenshtein for AArch64, scoring two independent pairs at once with one
 *      Myers integer per `uint64x2_t` lane. Mirrors the Ice Lake eight-lane byte Myers at a quarter of the width:
 *      there is no cross-lane carry machinery anywhere - every lane is its own register-resident 64-bit Myers
 *      integer, scaled in @b words instead of in lanes:
 *      - `distances_2x64_`            - 2 single-word Myers (shorter <= 64), one 64-bit integer per lane;
 *      - `distances_2x_multiword_<words_count_>` - 2 multi-word Myers with a compile-time word count, covering shorter
 *        in `(64, 64 * words_count_]`; the word state lives in stack arrays so the loop unrolls and the
 *        `vertical_positive` / `vertical_negative` words register-promote;
 *      - `distances_2x_multiword_large_` - the runtime-`words_count` sibling for the long tail (shorter > 512), where
 *        instantiating one variant per word count is not worthwhile.
 *      The multi-word kernels carry two intra-lane ripples (the 65-bit `(Eq & VP) + VP + addition_carry` and the
 *      `horizontal_positive` / `horizontal_negative` bit63->bit0 shift); neither ever crosses a lane. NEON has no
 *      lane-mask registers and no qword gather, so the per-lane `active` predicate is a `uint64x2_t` of all-ones /
 *      all-zeros built with `vcgtq_u64`, the `match_masks` rows are gathered scalar per lane (as in the Ice Lake multi-word
 *      kernels), and the boolean recurrence uses raw `vandq_u64` / `vorrq_u64` / `veorq_u64` / `vbicq_u64` instead
 *      of VPTERNLOGQ. Variable per-lane lengths are handled with that active mask, which freezes finished lanes.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<char, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using index_t = u32_t;
    static constexpr index_t lanes_k = 2;
    static constexpr size_t match_masks_bytes_k = sizeof(u64_t) * lanes_k *
                                                  256; // ? 4 KB single-word `match_masks` table.

    levenshtein_distance_myers() noexcept {}

    /** @brief Single-pair scratch sizing for the per-pair `levenshtein_distance` engine — delegated to the scalar
     *      serial Myers (one pair gains nothing from NEON batching; the 2-lane kernels serve the cross-product). */
    auto layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        return levenshtein_distance_myers<char, sz_cap_serial_k> {}.layout(first, second, specs);
    }

    /** @brief Single-pair Myers (per-pair engine path) — delegated to the scalar serial Myers. */
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        scratch_space_t scratch_space) noexcept {
        return levenshtein_distance_myers<char, sz_cap_serial_k> {}(first, second, result_ref, scratch_space);
    }

    /**
     *  @brief Two independent single-word Myers distances, one per `uint64x2_t` lane (each shorter side <= 64).
     *      The hot path for short words: no carry, shift, or boundary masking crosses lanes - every lane is its own
     *      64-bit Myers integer. @p scratch_space holds the `match_masks[2][256]` table (`match_masks_bytes_k`).
     */
    template <typename results_writer_>
    status_t distances_2x64_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                             scratch_space_t scratch_space) const noexcept {

        if (scratch_space.size() < match_masks_bytes_k) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data()); // ? Indexed `lane * 256 + symbol`.
        alignas(16) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        alignas(16) u64_t vertical_positive_init[lanes_k] = {0};

        size_t max_longer = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            size_t const longer_length = pairs.longers[lane_index].size();
            char_t const *const shorter = pairs.shorters[lane_index].data();
            char_t const *const longer = pairs.longers[lane_index].data();
            // Zero this lane's `match_masks` entries for every character its text may read (see the scalar Myers).
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[lane_index * 256 + (u8_t)shorter[position]] = 0;
            for (size_t position = 0; position != longer_length; ++position)
                match_masks[lane_index * 256 + (u8_t)longer[position]] = 0;
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[lane_index * 256 + (u8_t)shorter[position]] |= (u64_t)1 << position;
            top_bits[lane_index] = (u64_t)1 << (shorter_length - 1);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = longer_length;
            // VP = the low `shorter_length` bits set; a 64-bit shift is undefined, so build it as `~0 >> (64 - len)`
            // for full-width and the plain mask otherwise (the scalar oracle's `(1 << len) - 1`).
            vertical_positive_init[lane_index] = shorter_length == 64 ? ~(u64_t)0 : (((u64_t)1 << shorter_length) - 1);
            max_longer = sz_max_of_two(max_longer, longer_length);
        }

        uint64x2_t const one = vdupq_n_u64(1);
        uint64x2_t const top_mask = vld1q_u64(top_bits), longer_vec = vld1q_u64(longer_lengths);
        uint64x2_t vertical_positive = vld1q_u64(vertical_positive_init);
        uint64x2_t vertical_negative = vdupq_n_u64(0);
        uint64x2_t score = vld1q_u64(shorter_lengths);

        for (size_t position = 0; position != max_longer; ++position) {
            uint64x2_t const active = vcgtq_u64(longer_vec, vdupq_n_u64(position));
            alignas(16) u64_t equality_lanes[lanes_k] = {0};
            for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
                if (position < pairs.longers[lane_index].size())
                    equality_lanes[lane_index] =
                        match_masks[lane_index * 256 + (u8_t)pairs.longers[lane_index].data()[position]];
            uint64x2_t const equality = vld1q_u64(equality_lanes);

            uint64x2_t const carry_in = vorrq_u64(equality, vertical_negative);
            // Xh = (((Eq & VP) + VP) ^ VP) | Eq.
            uint64x2_t const sum = vaddq_u64(vandq_u64(equality, vertical_positive), vertical_positive);
            uint64x2_t const diagonal = vorrq_u64(veorq_u64(sum, vertical_positive), equality);
            // Ph = VN | ~(D | VP) = VN | ~D & ~VP -> `vbic(vbic(ones, D), VP)` then OR; expressed via OR-NOT.
            uint64x2_t horizontal_positive = vorrq_u64(
                vertical_negative, vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), diagonal), vertical_positive));
            uint64x2_t horizontal_negative = vandq_u64(vertical_positive, diagonal);

            // score += active & ((Ph & top) != 0); score -= active & ((Hn & top) != 0).
            uint64x2_t const add_step = vandq_u64(
                vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_positive, top_mask))), one);
            uint64x2_t const sub_step = vandq_u64(
                vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_negative, top_mask))), one);
            score = vsubq_u64(vaddq_u64(score, add_step), sub_step);

            horizontal_positive = vorrq_u64(vshlq_n_u64(horizontal_positive, 1), one);
            horizontal_negative = vshlq_n_u64(horizontal_negative, 1);
            // Pv' = Mh | ~(Xv | Ph) = Hn | ~(carry_in | Hp).
            uint64x2_t const next_positive = vorrq_u64(
                horizontal_negative, vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), carry_in), horizontal_positive));
            uint64x2_t const next_negative = vandq_u64(horizontal_positive, carry_in);
            vertical_positive = vbslq_u64(active, next_positive, vertical_positive);
            vertical_negative = vbslq_u64(active, next_negative, vertical_negative);
        }

        alignas(16) u64_t final_scores[lanes_k];
        vst1q_u64(final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief Two independent compile-time multi-word Myers distances, one pair per `uint64x2_t` lane, covering
     *      shorter sides in `(64, 64 * words_count_]`. Each lane is one register-resident `words_count_`-word Myers
     *      integer; the per-word `vertical_positive` / `vertical_negative` state lives in `uint64x2_t[words_count_]`
     *      stack arrays so the word loop unrolls and the words promote to registers. Two carries cross words @b within
     *      a lane (never across lanes): the 65-bit `(Eq & VP) + VP + addition_carry` ripple, tracked low->high via
     *      unsigned overflow detection, and the `horizontal_positive` / `horizontal_negative` bit63->bit0 shift.
     *
     *      @p scratch_space holds the per-lane multi-word `match_masks` table (`match_masks_bytes_k * words_count_`, base
     *      `match_masks + lane * 256 * words_count_`, entry `[symbol * words_count_ + word]`).
     */
    template <size_t words_count_, typename results_writer_>
    status_t distances_2x_multiword_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                     scratch_space_t scratch_space) const noexcept {

        constexpr size_t words_count = words_count_;

        size_t max_longer = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
        size_t const match_masks_words = (size_t)256 * words_count * lanes_k;
        if (scratch_space.size() < match_masks_words * sizeof(u64_t)) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data());
        for (size_t element = 0; element != match_masks_words; ++element) match_masks[element] = 0;

        alignas(16) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
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

        // Each lane keeps `words_count_` 64-bit Myers words on the stack; word `w` of every lane lives in one vector.
        uint64x2_t vertical_positive[words_count_];
        uint64x2_t vertical_negative[words_count_];
        for (size_t word = 0; word != words_count; ++word) {
            vertical_positive[word] = vdupq_n_u64(~(u64_t)0);
            vertical_negative[word] = vdupq_n_u64(0);
        }

        uint64x2_t const one = vdupq_n_u64(1);
        uint64x2_t const top_mask = vld1q_u64(top_bits), longer_vec = vld1q_u64(longer_lengths);
        uint64x2_t score = vld1q_u64(shorter_lengths);
        constexpr size_t last_word = words_count_ - 1;

        for (size_t position = 0; position != max_longer; ++position) {
            uint64x2_t const active = vcgtq_u64(longer_vec, vdupq_n_u64(position));
            // Per-lane base offset of the current character's word row inside that lane's table.
            u64_t base_offsets[lanes_k] = {0};
            for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index) {
                bool const lane_active = lane_index < pairs.lanes_count() &&
                                         position < pairs.longers[lane_index].size();
                u8_t const symbol = lane_active ? (u8_t)pairs.longers[lane_index].data()[position] : 0;
                base_offsets[lane_index] = lane_active
                                               ? (u64_t)lane_index * 256 * words_count + (u64_t)symbol * words_count
                                               : 0;
            }

            uint64x2_t addition_carry = vdupq_n_u64(0); // ? 0/1 per lane, the `(Eq&VP)+VP` ripple.
            uint64x2_t horizontal_positive_carry = one; // ? Word 0 seeds the leading 1.
            uint64x2_t horizontal_negative_carry = vdupq_n_u64(0);

            for (size_t word = 0; word != words_count; ++word) {
                alignas(16) u64_t equality_words[lanes_k] = {0};
                for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index)
                    equality_words[lane_index] = (lane_index < pairs.lanes_count() &&
                                                  position < pairs.longers[lane_index].size())
                                                     ? match_masks[(size_t)base_offsets[lane_index] + word]
                                                     : 0;
                uint64x2_t const equality = vld1q_u64(equality_words);

                uint64x2_t const vertical_positive_word = vertical_positive[word];
                uint64x2_t const vertical_negative_word = vertical_negative[word];

                // sum = (Eq & VP) + VP + addition_carry, tracking the per-lane carry-out low->high across words.
                uint64x2_t const summand = vandq_u64(equality, vertical_positive_word);
                uint64x2_t const sum_low = vaddq_u64(summand, vertical_positive_word);
                uint64x2_t const carry_from_summand = vcltq_u64(sum_low, summand);
                uint64x2_t const sum = vaddq_u64(sum_low, addition_carry);
                uint64x2_t const carry_from_incoming = vcltq_u64(sum, sum_low);
                addition_carry = vandq_u64(vorrq_u64(carry_from_summand, carry_from_incoming), one);

                uint64x2_t const carry_in = vorrq_u64(equality, vertical_negative_word); // ? Eq | VN
                // Xh = (sum ^ VP) | Eq | VN == (sum ^ VP) | carry_in.
                uint64x2_t const diagonal = vorrq_u64(veorq_u64(sum, vertical_positive_word), carry_in);
                uint64x2_t horizontal_positive = vorrq_u64( // ? VN | ~(D | VP)
                    vertical_negative_word,
                    vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), diagonal), vertical_positive_word));
                uint64x2_t horizontal_negative = vandq_u64(vertical_positive_word, diagonal); // ? VP & D

                if (word == last_word) {
                    uint64x2_t const add_step = vandq_u64(
                        vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_positive, top_mask))), one);
                    uint64x2_t const sub_step = vandq_u64(
                        vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_negative, top_mask))), one);
                    score = vsubq_u64(vaddq_u64(score, add_step), sub_step);
                }

                // Save each word's bit63 as the next word's bit0 carry, then shift up by one.
                uint64x2_t const next_positive_carry = vshrq_n_u64(horizontal_positive, 63);
                uint64x2_t const next_negative_carry = vshrq_n_u64(horizontal_negative, 63);
                horizontal_positive = vorrq_u64(vshlq_n_u64(horizontal_positive, 1), horizontal_positive_carry);
                horizontal_negative = vorrq_u64(vshlq_n_u64(horizontal_negative, 1), horizontal_negative_carry);
                horizontal_positive_carry = next_positive_carry;
                horizontal_negative_carry = next_negative_carry;

                // Pv' = Mh | ~(Xv | Ph) = Hn | ~(carry_in | Hp); applied only to active lanes.
                uint64x2_t const next_positive = vorrq_u64(
                    horizontal_negative, vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), carry_in), horizontal_positive));
                uint64x2_t const next_negative = vandq_u64(horizontal_positive, carry_in);
                vertical_positive[word] = vbslq_u64(active, next_positive, vertical_positive_word);
                vertical_negative[word] = vbslq_u64(active, next_negative, vertical_negative_word);
            }
        }

        alignas(16) u64_t final_scores[lanes_k];
        vst1q_u64(final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief The runtime-`words_count` sibling of `distances_2x_multiword_<words_count_>` for the long tail
     *      (shorter > 512), where instantiating one variant per exact word count is not worthwhile. Each lane
     *      carries its own `ceil(shorter / 64)`-word Myers integer; the group shares a single @p words_count so every
     *      lane loops the same word range. The per-word state lives in a fixed-capacity stack array indexed at runtime;
     *      the math is identical to the templated variant.
     *
     *      @p scratch_space holds the per-lane multi-word `match_masks` table (`match_masks_bytes_k * words_count`, base
     *      `match_masks + lane * 256 * words_count`, entry `[symbol * words_count + word]`).
     */
    template <typename results_writer_>
    status_t distances_2x_multiword_large_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                           scratch_space_t scratch_space) const noexcept {

        static constexpr size_t stack_words_capacity_k = 64; // ? Covers shorter sides up to 4096 on the stack.

        size_t max_longer = 0, max_shorter = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
            max_shorter = sz_max_of_two(max_shorter, pairs.shorters[lane_index].size());
        }
        size_t const words_count = divide_round_up<size_t>(max_shorter, 64);
        if (words_count == 0 || words_count > stack_words_capacity_k) return status_t::bad_alloc_k;
        size_t const match_masks_words = (size_t)256 * words_count * lanes_k;
        if (scratch_space.size() < match_masks_words * sizeof(u64_t)) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data());
        for (size_t element = 0; element != match_masks_words; ++element) match_masks[element] = 0;

        alignas(16) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
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

        // Each lane keeps `words_count` 64-bit Myers words; word `w` of every lane lives in one vector.
        uint64x2_t vertical_positive[stack_words_capacity_k];
        uint64x2_t vertical_negative[stack_words_capacity_k];
        for (size_t word = 0; word != words_count; ++word) {
            vertical_positive[word] = vdupq_n_u64(~(u64_t)0);
            vertical_negative[word] = vdupq_n_u64(0);
        }

        uint64x2_t const one = vdupq_n_u64(1);
        uint64x2_t const top_mask = vld1q_u64(top_bits), longer_vec = vld1q_u64(longer_lengths);
        uint64x2_t score = vld1q_u64(shorter_lengths);
        size_t const last_word = words_count - 1;

        for (size_t position = 0; position != max_longer; ++position) {
            uint64x2_t const active = vcgtq_u64(longer_vec, vdupq_n_u64(position));
            u64_t base_offsets[lanes_k] = {0};
            for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index) {
                bool const lane_active = lane_index < pairs.lanes_count() &&
                                         position < pairs.longers[lane_index].size();
                u8_t const symbol = lane_active ? (u8_t)pairs.longers[lane_index].data()[position] : 0;
                base_offsets[lane_index] = lane_active
                                               ? (u64_t)lane_index * 256 * words_count + (u64_t)symbol * words_count
                                               : 0;
            }

            uint64x2_t addition_carry = vdupq_n_u64(0);
            uint64x2_t horizontal_positive_carry = one;
            uint64x2_t horizontal_negative_carry = vdupq_n_u64(0);

            for (size_t word = 0; word != words_count; ++word) {
                alignas(16) u64_t equality_words[lanes_k] = {0};
                for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index)
                    equality_words[lane_index] = (lane_index < pairs.lanes_count() &&
                                                  position < pairs.longers[lane_index].size())
                                                     ? match_masks[(size_t)base_offsets[lane_index] + word]
                                                     : 0;
                uint64x2_t const equality = vld1q_u64(equality_words);

                uint64x2_t const vertical_positive_word = vertical_positive[word];
                uint64x2_t const vertical_negative_word = vertical_negative[word];

                uint64x2_t const summand = vandq_u64(equality, vertical_positive_word);
                uint64x2_t const sum_low = vaddq_u64(summand, vertical_positive_word);
                uint64x2_t const carry_from_summand = vcltq_u64(sum_low, summand);
                uint64x2_t const sum = vaddq_u64(sum_low, addition_carry);
                uint64x2_t const carry_from_incoming = vcltq_u64(sum, sum_low);
                addition_carry = vandq_u64(vorrq_u64(carry_from_summand, carry_from_incoming), one);

                uint64x2_t const carry_in = vorrq_u64(equality, vertical_negative_word);
                uint64x2_t const diagonal = vorrq_u64(veorq_u64(sum, vertical_positive_word), carry_in);
                uint64x2_t horizontal_positive = vorrq_u64(
                    vertical_negative_word,
                    vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), diagonal), vertical_positive_word));
                uint64x2_t horizontal_negative = vandq_u64(vertical_positive_word, diagonal);

                if (word == last_word) {
                    uint64x2_t const add_step = vandq_u64(
                        vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_positive, top_mask))), one);
                    uint64x2_t const sub_step = vandq_u64(
                        vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_negative, top_mask))), one);
                    score = vsubq_u64(vaddq_u64(score, add_step), sub_step);
                }

                uint64x2_t const next_positive_carry = vshrq_n_u64(horizontal_positive, 63);
                uint64x2_t const next_negative_carry = vshrq_n_u64(horizontal_negative, 63);
                horizontal_positive = vorrq_u64(vshlq_n_u64(horizontal_positive, 1), horizontal_positive_carry);
                horizontal_negative = vorrq_u64(vshlq_n_u64(horizontal_negative, 1), horizontal_negative_carry);
                horizontal_positive_carry = next_positive_carry;
                horizontal_negative_carry = next_negative_carry;

                uint64x2_t const next_positive = vorrq_u64(
                    horizontal_negative, vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), carry_in), horizontal_positive));
                uint64x2_t const next_negative = vandq_u64(horizontal_positive, carry_in);
                vertical_positive[word] = vbslq_u64(active, next_positive, vertical_positive_word);
                vertical_negative[word] = vbslq_u64(active, next_negative, vertical_negative_word);
            }
        }

        alignas(16) u64_t final_scores[lanes_k];
        vst1q_u64(final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }
};

/**
 *  @brief NEON Myers/Hyyr\xc3\xb6 unit-cost @b rune (UTF-32) Levenshtein for AArch64, scoring two independent pairs at
 *      once with one Myers integer per `uint64x2_t` lane - the rune twin of the byte `levenshtein_distance_myers`, and
 *      the NEON 2-lane shrink of the Ice Lake eight-lane rune Myers. The scan is bit-for-bit identical to the byte
 *      2x64 / 2xN family; the only delta is the `match_masks` source. A byte pattern indexes a dense 256-row table, but a rune
 *      pattern (up to 0x10FFFF distinct keys) cannot, so each lane builds a small @b open-addressing hash over its own
 *      pattern's distinct runes (capacity `next_pow2(2*distinct)`, load factor <= 0.5), and per text position each lane
 *      scalar-probes its hash for the current text rune to assemble that lane's `Eq` bitmask word(s). A text rune absent
 *      from a lane's pattern hits the lane's permanently-zero `absent_row`, i.e. an all-zero `Eq` (matches nothing) -
 *      the same semantics as a byte miss. The hash math (multiply-shift `hash_rune`, the `empty_slot_k` sentinel, the
 *      linear-probe build/lookup) is reused verbatim from the Ice Lake rune Myers and the serial rune oracle
 *      (`levenshtein_distance_myers<rune_t, sz_cap_serial_k>`) so all three stay bit-exact.
 *      - `distances_2x64_`            - 2 single-word rune Myers (shorter <= 64 runes), one 64-bit integer per lane;
 *      - `distances_2x_multiword_<words_count_>` - 2 multi-word rune Myers with a compile-time word count, covering
 *        shorter in `(64, 64 * words_count_]` runes;
 *      - `distances_2x_multiword_large_` - the runtime-`words_count` sibling for the long tail (shorter > 512 runes).
 *      As in the NEON byte Myers, NEON has no lane-mask registers and no qword gather, so the per-lane `active`
 *      predicate is a `uint64x2_t` of all-ones / all-zeros built with `vcgtq_u64`, the `match_masks` rows are gathered scalar
 *      per lane, and the boolean recurrence uses raw `vandq_u64` / `vorrq_u64` / `veorq_u64` / `vbicq_u64`.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<rune_t, capability_, std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = rune_t;
    using index_t = u32_t;
    static constexpr index_t lanes_k = 2;

    /** @brief Sentinel rune marking an empty hash slot (`rune_t` never reaches `0xFFFFFFFF`; valid <= 0x10FFFF). */
    static constexpr rune_t empty_slot_k = static_cast<rune_t>(0xFFFFFFFFu);

    levenshtein_distance_myers() noexcept {}

    /** @brief Number of 64-bit words spanning a @p shorter_length-rune pattern, i.e. `ceil(shorter_length / 64)`. */
    static size_t words_count_for(size_t shorter_length) noexcept {
        return divide_round_up<size_t>(shorter_length, 64);
    }

    /**
     *  @brief Open-addressing capacity (a power of two) for a pattern of @p distinct_upper_bound distinct runes. The
     *      pattern has at most `shorter_length` distinct runes, so passing the longest shorter side is a safe upper
     *      bound for the whole 2-lane group. The `2 *` keeps the load factor <= 0.5 for cheap linear probing;
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
     *  @brief Scratch bytes the 2-lane rune `match_masks` needs for a group whose longest shorter side is @p max_shorter
     *      runes: two per-lane hash tables (slot keys + `words_count` bitmask words per slot) plus one shared
     *      permanently-zero `absent_row`. This is the single source of truth for both the scratch sizing in the
     *      cross-product engine and the carving inside the kernels, so the two can never disagree.
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
     *  @brief Carves the 2-lane hash `match_masks` out of @p scratch_space and builds it from each lane's shorter side. Every
     *      lane shares the group-wide @p capacity and @p words_count (so the inner scan loops one common word range,
     *      exactly like the byte 2xN kernel); a lane's pattern is hashed into its own slot region. Returns `false`
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
     *  @brief Two independent single-word rune Myers distances, one per `uint64x2_t` lane (each shorter side <= 64
     *      runes). The scan is verbatim the byte `distances_2x64_`; only the `Eq` source differs - per text position
     *      each lane hash-probes its rune to a single 64-bit bitmask, and the two masks are assembled into one vector.
     *      @p scratch_space holds the 2-lane hash `match_masks` (`scratch_bytes_for(max_shorter)`).
     */
    template <typename results_writer_>
    status_t distances_2x64_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
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

        alignas(16) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        alignas(16) u64_t vertical_positive_init[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            top_bits[lane_index] = (u64_t)1 << (shorter_length - 1);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
            // VP = the low `shorter_length` bits set; a 64-bit shift is undefined, so build full-width as `~0`.
            vertical_positive_init[lane_index] = shorter_length == 64 ? ~(u64_t)0 : (((u64_t)1 << shorter_length) - 1);
        }

        uint64x2_t const one = vdupq_n_u64(1);
        uint64x2_t const top_mask = vld1q_u64(top_bits), longer_vec = vld1q_u64(longer_lengths);
        uint64x2_t vertical_positive = vld1q_u64(vertical_positive_init);
        uint64x2_t vertical_negative = vdupq_n_u64(0);
        uint64x2_t score = vld1q_u64(shorter_lengths);

        for (size_t position = 0; position != max_longer; ++position) {
            uint64x2_t const active = vcgtq_u64(longer_vec, vdupq_n_u64(position));
            // Per-lane hash probe of the current text rune -> one 64-bit `Eq` mask per lane, assembled into one vector.
            alignas(16) u64_t equality_lanes[lanes_k] = {0};
            for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
                if (position < pairs.longers[lane_index].size()) {
                    rune_t const symbol = pairs.longers[lane_index].data()[position];
                    equality_lanes[lane_index] = lane_match_row_(slot_keys, slot_masks, absent_row, lane_index,
                                                                 capacity, 1, symbol)[0];
                }
            uint64x2_t const equality = vld1q_u64(equality_lanes);

            uint64x2_t const carry_in = vorrq_u64(equality, vertical_negative);
            // Xh = (((Eq & VP) + VP) ^ VP) | Eq.
            uint64x2_t const sum = vaddq_u64(vandq_u64(equality, vertical_positive), vertical_positive);
            uint64x2_t const diagonal = vorrq_u64(veorq_u64(sum, vertical_positive), equality);
            // Ph = VN | ~(D | VP).
            uint64x2_t horizontal_positive = vorrq_u64(
                vertical_negative, vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), diagonal), vertical_positive));
            uint64x2_t horizontal_negative = vandq_u64(vertical_positive, diagonal);

            uint64x2_t const add_step = vandq_u64(
                vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_positive, top_mask))), one);
            uint64x2_t const sub_step = vandq_u64(
                vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_negative, top_mask))), one);
            score = vsubq_u64(vaddq_u64(score, add_step), sub_step);

            horizontal_positive = vorrq_u64(vshlq_n_u64(horizontal_positive, 1), one);
            horizontal_negative = vshlq_n_u64(horizontal_negative, 1);
            // Pv' = Hn | ~(carry_in | Hp).
            uint64x2_t const next_positive = vorrq_u64(
                horizontal_negative, vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), carry_in), horizontal_positive));
            uint64x2_t const next_negative = vandq_u64(horizontal_positive, carry_in);
            vertical_positive = vbslq_u64(active, next_positive, vertical_positive);
            vertical_negative = vbslq_u64(active, next_negative, vertical_negative);
        }

        alignas(16) u64_t final_scores[lanes_k];
        vst1q_u64(final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief Two independent compile-time multi-word rune Myers distances, one pair per `uint64x2_t` lane, covering
     *      shorter sides in `(64, 64 * words_count_]` runes. The scan is verbatim the byte
     *      `distances_2x_multiword_<words_count_>` (two intra-lane ripples: the 65-bit `(Eq & VP) + VP + addition_carry`
     *      and the bit63->bit0 shift carries; neither crosses a lane); only the `Eq` source differs - each lane probes
     *      its hash once per text rune and reads the resulting row's `words_count_` bitmask words.
     *      @p scratch_space holds the 2-lane hash `match_masks` (`scratch_bytes_for(max_shorter)`).
     */
    template <size_t words_count_, typename results_writer_>
    status_t distances_2x_multiword_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
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

        alignas(16) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            top_bits[lane_index] = (u64_t)1 << ((shorter_length - 1) & 63);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
        }

        // Each lane keeps `words_count_` 64-bit Myers words on the stack; word `w` of every lane lives in one vector.
        uint64x2_t vertical_positive[words_count_];
        uint64x2_t vertical_negative[words_count_];
        for (size_t word = 0; word != words_count; ++word) {
            vertical_positive[word] = vdupq_n_u64(~(u64_t)0);
            vertical_negative[word] = vdupq_n_u64(0);
        }

        uint64x2_t const one = vdupq_n_u64(1);
        uint64x2_t const top_mask = vld1q_u64(top_bits), longer_vec = vld1q_u64(longer_lengths);
        uint64x2_t score = vld1q_u64(shorter_lengths);
        constexpr size_t last_word = words_count_ - 1;

        for (size_t position = 0; position != max_longer; ++position) {
            uint64x2_t const active = vcgtq_u64(longer_vec, vdupq_n_u64(position));
            // Per-lane hash probe of the current text rune -> the lane's `words_count` bitmask row (or `absent_row`).
            u64_t const *match_rows[lanes_k];
            for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index) {
                bool const lane_active = lane_index < pairs.lanes_count() &&
                                         position < pairs.longers[lane_index].size();
                rune_t const symbol = lane_active ? pairs.longers[lane_index].data()[position] : empty_slot_k;
                match_rows[lane_index] = lane_active ? lane_match_row_(slot_keys, slot_masks, absent_row, lane_index,
                                                                       capacity, words_count, symbol)
                                                     : absent_row;
            }

            uint64x2_t addition_carry = vdupq_n_u64(0); // ? 0/1 per lane, the `(Eq&VP)+VP` ripple.
            uint64x2_t horizontal_positive_carry = one; // ? Word 0 seeds the leading 1.
            uint64x2_t horizontal_negative_carry = vdupq_n_u64(0);

            for (size_t word = 0; word != words_count; ++word) {
                alignas(16) u64_t equality_words[lanes_k] = {0};
                for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index)
                    equality_words[lane_index] = (lane_index < pairs.lanes_count() &&
                                                  position < pairs.longers[lane_index].size())
                                                     ? match_rows[lane_index][word]
                                                     : 0;
                uint64x2_t const equality = vld1q_u64(equality_words);

                uint64x2_t const vertical_positive_word = vertical_positive[word];
                uint64x2_t const vertical_negative_word = vertical_negative[word];

                // sum = (Eq & VP) + VP + addition_carry, tracking the per-lane carry-out low->high across words.
                uint64x2_t const summand = vandq_u64(equality, vertical_positive_word);
                uint64x2_t const sum_low = vaddq_u64(summand, vertical_positive_word);
                uint64x2_t const carry_from_summand = vcltq_u64(sum_low, summand);
                uint64x2_t const sum = vaddq_u64(sum_low, addition_carry);
                uint64x2_t const carry_from_incoming = vcltq_u64(sum, sum_low);
                addition_carry = vandq_u64(vorrq_u64(carry_from_summand, carry_from_incoming), one);

                uint64x2_t const carry_in = vorrq_u64(equality, vertical_negative_word); // ? Eq | VN
                // Xh = (sum ^ VP) | Eq | VN == (sum ^ VP) | carry_in.
                uint64x2_t const diagonal = vorrq_u64(veorq_u64(sum, vertical_positive_word), carry_in);
                uint64x2_t horizontal_positive = vorrq_u64( // ? VN | ~(D | VP)
                    vertical_negative_word,
                    vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), diagonal), vertical_positive_word));
                uint64x2_t horizontal_negative = vandq_u64(vertical_positive_word, diagonal); // ? VP & D

                if (word == last_word) {
                    uint64x2_t const add_step = vandq_u64(
                        vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_positive, top_mask))), one);
                    uint64x2_t const sub_step = vandq_u64(
                        vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_negative, top_mask))), one);
                    score = vsubq_u64(vaddq_u64(score, add_step), sub_step);
                }

                // Save each word's bit63 as the next word's bit0 carry, then shift up by one.
                uint64x2_t const next_positive_carry = vshrq_n_u64(horizontal_positive, 63);
                uint64x2_t const next_negative_carry = vshrq_n_u64(horizontal_negative, 63);
                horizontal_positive = vorrq_u64(vshlq_n_u64(horizontal_positive, 1), horizontal_positive_carry);
                horizontal_negative = vorrq_u64(vshlq_n_u64(horizontal_negative, 1), horizontal_negative_carry);
                horizontal_positive_carry = next_positive_carry;
                horizontal_negative_carry = next_negative_carry;

                // Pv' = Hn | ~(carry_in | Hp); applied only to active lanes.
                uint64x2_t const next_positive = vorrq_u64(
                    horizontal_negative, vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), carry_in), horizontal_positive));
                uint64x2_t const next_negative = vandq_u64(horizontal_positive, carry_in);
                vertical_positive[word] = vbslq_u64(active, next_positive, vertical_positive_word);
                vertical_negative[word] = vbslq_u64(active, next_negative, vertical_negative_word);
            }
        }

        alignas(16) u64_t final_scores[lanes_k];
        vst1q_u64(final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief The runtime-`words_count` sibling of `distances_2x_multiword_<words_count_>` for the long tail (shorter >
     *      512 runes). Each lane carries its own `ceil(shorter / 64)`-word Myers integer; the group shares a single
     *      @p words_count so every lane loops the same word range. The per-word state lives in a fixed-capacity stack
     *      array indexed at runtime; the math is identical to the templated variant.
     *      @p scratch_space holds the 2-lane hash `match_masks` (`scratch_bytes_for(max_shorter)`).
     */
    template <typename results_writer_>
    status_t distances_2x_multiword_large_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
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

        alignas(16) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            top_bits[lane_index] = (u64_t)1 << ((shorter_length - 1) & 63);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
        }

        // Each lane keeps `words_count` 64-bit Myers words; word `w` of every lane lives in one vector.
        uint64x2_t vertical_positive[stack_words_capacity_k];
        uint64x2_t vertical_negative[stack_words_capacity_k];
        for (size_t word = 0; word != words_count; ++word) {
            vertical_positive[word] = vdupq_n_u64(~(u64_t)0);
            vertical_negative[word] = vdupq_n_u64(0);
        }

        uint64x2_t const one = vdupq_n_u64(1);
        uint64x2_t const top_mask = vld1q_u64(top_bits), longer_vec = vld1q_u64(longer_lengths);
        uint64x2_t score = vld1q_u64(shorter_lengths);
        size_t const last_word = words_count - 1;

        for (size_t position = 0; position != max_longer; ++position) {
            uint64x2_t const active = vcgtq_u64(longer_vec, vdupq_n_u64(position));
            u64_t const *match_rows[lanes_k];
            for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index) {
                bool const lane_active = lane_index < pairs.lanes_count() &&
                                         position < pairs.longers[lane_index].size();
                rune_t const symbol = lane_active ? pairs.longers[lane_index].data()[position] : empty_slot_k;
                match_rows[lane_index] = lane_active ? lane_match_row_(slot_keys, slot_masks, absent_row, lane_index,
                                                                       capacity, words_count, symbol)
                                                     : absent_row;
            }

            uint64x2_t addition_carry = vdupq_n_u64(0);
            uint64x2_t horizontal_positive_carry = one;
            uint64x2_t horizontal_negative_carry = vdupq_n_u64(0);

            for (size_t word = 0; word != words_count; ++word) {
                alignas(16) u64_t equality_words[lanes_k] = {0};
                for (index_t lane_index = 0; lane_index != lanes_k; ++lane_index)
                    equality_words[lane_index] = (lane_index < pairs.lanes_count() &&
                                                  position < pairs.longers[lane_index].size())
                                                     ? match_rows[lane_index][word]
                                                     : 0;
                uint64x2_t const equality = vld1q_u64(equality_words);

                uint64x2_t const vertical_positive_word = vertical_positive[word];
                uint64x2_t const vertical_negative_word = vertical_negative[word];

                uint64x2_t const summand = vandq_u64(equality, vertical_positive_word);
                uint64x2_t const sum_low = vaddq_u64(summand, vertical_positive_word);
                uint64x2_t const carry_from_summand = vcltq_u64(sum_low, summand);
                uint64x2_t const sum = vaddq_u64(sum_low, addition_carry);
                uint64x2_t const carry_from_incoming = vcltq_u64(sum, sum_low);
                addition_carry = vandq_u64(vorrq_u64(carry_from_summand, carry_from_incoming), one);

                uint64x2_t const carry_in = vorrq_u64(equality, vertical_negative_word);
                uint64x2_t const diagonal = vorrq_u64(veorq_u64(sum, vertical_positive_word), carry_in);
                uint64x2_t horizontal_positive = vorrq_u64(
                    vertical_negative_word,
                    vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), diagonal), vertical_positive_word));
                uint64x2_t horizontal_negative = vandq_u64(vertical_positive_word, diagonal);

                if (word == last_word) {
                    uint64x2_t const add_step = vandq_u64(
                        vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_positive, top_mask))), one);
                    uint64x2_t const sub_step = vandq_u64(
                        vandq_u64(active, lane_nonzero_(vandq_u64(horizontal_negative, top_mask))), one);
                    score = vsubq_u64(vaddq_u64(score, add_step), sub_step);
                }

                uint64x2_t const next_positive_carry = vshrq_n_u64(horizontal_positive, 63);
                uint64x2_t const next_negative_carry = vshrq_n_u64(horizontal_negative, 63);
                horizontal_positive = vorrq_u64(vshlq_n_u64(horizontal_positive, 1), horizontal_positive_carry);
                horizontal_negative = vorrq_u64(vshlq_n_u64(horizontal_negative, 1), horizontal_negative_carry);
                horizontal_positive_carry = next_positive_carry;
                horizontal_negative_carry = next_negative_carry;

                uint64x2_t const next_positive = vorrq_u64(
                    horizontal_negative, vbicq_u64(vbicq_u64(vdupq_n_u64(~(u64_t)0), carry_in), horizontal_positive));
                uint64x2_t const next_negative = vandq_u64(horizontal_positive, carry_in);
                vertical_positive[word] = vbslq_u64(active, next_positive, vertical_positive_word);
                vertical_negative[word] = vbslq_u64(active, next_negative, vertical_negative_word);
            }
        }

        alignas(16) u64_t final_scores[lanes_k];
        vst1q_u64(final_scores, score);
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }
};

#pragma endregion Bit Parallel Myers

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

    substitution_lookup_neon_t() noexcept {}

    void reload_classes(u8_t const *byte_to_class) noexcept {
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
    void reload_costs(
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

    SZ_INLINE uint8x16_t classify16(uint8x16_t text_vec) const noexcept {

        // Map each input byte to its class using the 256-entry `byte_to_class` table. Each `vqtbl4q_u8`
        // addresses one 64-byte window and returns zero for indices outside [0, 63], so XOR-ing the index by
        // 0x40 / 0x80 / 0xc0 selects the matching window and OR-ing the four results reconstructs the map.
        uint8x16_t lookup_0_to_63 = vqtbl4q_u8(byte_to_class_vecs_[0], text_vec);
        uint8x16_t lookup_64_to_127 = vqtbl4q_u8(byte_to_class_vecs_[1], veorq_u8(text_vec, vdupq_n_u8(0x40)));
        uint8x16_t lookup_128_to_191 = vqtbl4q_u8(byte_to_class_vecs_[2], veorq_u8(text_vec, vdupq_n_u8(0x80)));
        uint8x16_t lookup_192_to_255 = vqtbl4q_u8(byte_to_class_vecs_[3], veorq_u8(text_vec, vdupq_n_u8(0xc0)));
        return vorrq_u8(vorrq_u8(lookup_0_to_63, lookup_64_to_127), vorrq_u8(lookup_128_to_191, lookup_192_to_255));
    }

    SZ_INLINE int8x16_t lookup16(uint8x16_t first_class_vec, uint8x16_t second_class_vec) const noexcept {

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

#pragma region NEON Inter Sequence Candidate Lanes

/**
 *  @brief Inter-sequence NEON walker: one query against up to 16 candidates packed one-per-lane, 8-bit cells.
 *
 *  Computes the @b global unit-cost Levenshtein distance of a single shared query against a transposed
 *  `candidate_lanes_block` of up to 16 candidates. Each `uint8x16_t` holds 16 `u8` cells - lane @p lane_index
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
                uint8x16_t const cell_score_vec = vminq_u8(cost_if_substitution_vec,
                                                           vminq_u8(cost_if_deletion_vec, cost_if_insertion_vec));
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
 *  broadcast query character with `vceq_u8`, yielding 8 bytes of `0xFF` (match) or `0x00` (mismatch) that widen
 *  with `vmovl_u8` to an 8-lane `u16` mask; `vbslq_u16` then selects the configured `match` cost on equal lanes
 *  and `mismatch` on the rest as the diagonal substitution penalty, while deletion and insertion each add the
 *  uniform `gap` cost. The recurrence `cell = min(substitution, min(deletion, insertion))` advances all 8 lanes
 *  over `vminq_u16`. This serves the @b non-unit uniform-cost Levenshtein cells; the unit-cost case (match 0,
 *  mismatch 1, gap 1) is handled by the Myers fast path and never reaches here.
 *
 *  @note The cells are `u16`, so the kernel is only valid while every reachable score stays below 65535; the longest
 *      reachable cell is the all-gap path `max(query, candidate) * gap`. Enforcing that bound (a cost-dependent
 *      length limit) is the caller's dispatch contract; the kernel performs no runtime range check.
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
        uint16x8_t const match_vec = vdupq_n_u16(match_cost);
        uint16x8_t const mismatch_vec = vdupq_n_u16(mismatch_cost);
        uint16x8_t const gap_vec = vdupq_n_u16(gap_cost);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            vst1q_u16(previous_row + column * candidate_lanes_k, vdupq_n_u16(static_cast<u16_t>(column * gap_cost)));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint16x8_t const query_char_vec = vdupq_n_u16(static_cast<u16_t>((u8_t)query[query_position - 1]));
            vst1q_u16(current_row, vdupq_n_u16(static_cast<u16_t>(query_position * gap_cost)));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x8_t const candidate_chars_u8_vec = vld1_u8((u8_t const *)candidates.position(column - 1));
                uint16x8_t const candidate_chars_vec = vmovl_u8(candidate_chars_u8_vec);
                uint16x8_t const diagonal_vec = vld1q_u16(previous_row + (column - 1) * candidate_lanes_k);
                uint16x8_t const deletion_source_vec = vld1q_u16(previous_row + column * candidate_lanes_k);
                uint16x8_t const insertion_source_vec = vld1q_u16(current_row + (column - 1) * candidate_lanes_k);

                // Widen the 8 candidate bytes to `u16` and compare against the broadcast query character so the mask
                // is a full-width `0xFFFF` (match) / `0x0000` (mismatch); `vbslq_u16` then selects `match` on equal
                // lanes and `mismatch` on the rest as the diagonal substitution penalty.
                uint16x8_t const equal_u16_vec = vceqq_u16(query_char_vec, candidate_chars_vec);
                uint16x8_t const substitution_addend_vec = vbslq_u16(equal_u16_vec, match_vec, mismatch_vec);
                uint16x8_t const cost_if_substitution_vec = vaddq_u16(diagonal_vec, substitution_addend_vec);
                uint16x8_t const cost_if_deletion_vec = vaddq_u16(deletion_source_vec, gap_vec);
                uint16x8_t const cost_if_insertion_vec = vaddq_u16(insertion_source_vec, gap_vec);
                uint16x8_t const cell_score_vec = vminq_u16(cost_if_substitution_vec,
                                                            vminq_u16(cost_if_deletion_vec, cost_if_insertion_vec));
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
 *  @brief Inter-sequence NEON walker: one query against up to 4 candidates packed one-per-lane, 32-bit cells.
 *
 *  Width-mirror of the 8-lane `u16` walker, widened to `u32` cells so each `uint32x4_t` holds 4 lanes. The candidate
 *  characters remain `char`/`u8`: 4 of them widen through `vmovl_u8` then `vmovl_u16(vget_low_u16(...))` into a 4-lane
 *  `u32` vector and compare against the broadcast query character with `vceqq_u32`, yielding a full-width `0xFFFFFFFF`
 *  (match) / `0x00000000` (mismatch) mask; `vbslq_u32` then selects the configured `match` cost on equal lanes and
 *  `mismatch` on the rest as the diagonal substitution penalty, while deletion and insertion each add the uniform
 *  `gap` cost. The recurrence `cell = min(substitution, min(deletion, insertion))` advances all 4 lanes over
 *  `vminq_u32`. Halving the lane count to 4 quadruples the representable score range to the `u32` window, serving
 *  candidates whose reachable scores exceed the `u16` contract.
 *
 *  @note The cells are `u32`, so the kernel is only valid while every reachable score stays below 4294967295; the
 *      longest reachable cell is the all-gap path `max(query, candidate) * gap`. Enforcing that bound (a cost-dependent
 *      length limit) is the caller's dispatch contract; the kernel performs no runtime range check.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u32_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 4, void> {

    using char_t = char;
    using score_t = u32_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 4;

    // The `u32` lane recurrence hardcodes `vminq_u32`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 32-bit candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (4 `u32` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 4 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 4`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        // The recurrence honors the engine's configured uniform costs: `match`/`mismatch` on the diagonal and a
        // single `gap` on either deletion or insertion. The unit-cost case (match 0, mismatch 1, gap 1) is served by
        // the Myers fast path, so this kernel only ever runs for non-unit costs.
        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const gap_cost = static_cast<score_t>(gap_costs_.open_or_extend);
        uint32x4_t const match_vec = vdupq_n_u32(match_cost);
        uint32x4_t const mismatch_vec = vdupq_n_u32(mismatch_cost);
        uint32x4_t const gap_vec = vdupq_n_u32(gap_cost);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            vst1q_u32(previous_row + column * candidate_lanes_k, vdupq_n_u32(static_cast<u32_t>(column * gap_cost)));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint32x4_t const query_char_vec = vdupq_n_u32(static_cast<u32_t>((u8_t)query[query_position - 1]));
            vst1q_u32(current_row, vdupq_n_u32(static_cast<u32_t>(query_position * gap_cost)));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x8_t const candidate_chars_u8_vec = vld1_u8((u8_t const *)candidates.position(column - 1));
                uint16x8_t const candidate_chars_u16_vec = vmovl_u8(candidate_chars_u8_vec);
                uint32x4_t const candidate_chars_vec = vmovl_u16(vget_low_u16(candidate_chars_u16_vec));
                uint32x4_t const diagonal_vec = vld1q_u32(previous_row + (column - 1) * candidate_lanes_k);
                uint32x4_t const deletion_source_vec = vld1q_u32(previous_row + column * candidate_lanes_k);
                uint32x4_t const insertion_source_vec = vld1q_u32(current_row + (column - 1) * candidate_lanes_k);

                // Widen the 4 candidate bytes to `u32` and compare against the broadcast query character so the mask
                // is a full-width `0xFFFFFFFF` (match) / `0x00000000` (mismatch); `vbslq_u32` then selects `match` on
                // equal lanes and `mismatch` on the rest as the diagonal substitution penalty.
                uint32x4_t const equal_u32_vec = vceqq_u32(query_char_vec, candidate_chars_vec);
                uint32x4_t const substitution_addend_vec = vbslq_u32(equal_u32_vec, match_vec, mismatch_vec);
                uint32x4_t const cost_if_substitution_vec = vaddq_u32(diagonal_vec, substitution_addend_vec);
                uint32x4_t const cost_if_deletion_vec = vaddq_u32(deletion_source_vec, gap_vec);
                uint32x4_t const cost_if_insertion_vec = vaddq_u32(insertion_source_vec, gap_vec);
                uint32x4_t const cell_score_vec = vminq_u32(cost_if_substitution_vec,
                                                            vminq_u32(cost_if_deletion_vec, cost_if_insertion_vec));
                vst1q_u32(current_row + column * candidate_lanes_k, cell_score_vec);
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
 *  @brief Inter-sequence NEON walker: one @b rune (UTF-32) query against up to 8 rune candidates packed
 *         one-per-lane, 16-bit cells, @b non-unit uniform-cost @b Levenshtein.
 *
 *  Rune sibling of the byte-keyed 8-lane `u16` uniform candidate-lane walker: the only difference from the byte
 *  body is the lane key - 32-bit Unicode code points (`rune_t`) instead of bytes - and the rune-equality compare.
 *  Eight candidate runes per column occupy 256 bits, so they load as two `uint32x4_t` halves; each half compares
 *  against the broadcast query rune with `vceqq_u32`, the two `u32` match masks narrow with `vmovn_u32` and
 *  recombine into one full-width 8-lane `u16` match mask, and `vbslq_u16` then selects the configured `match`
 *  cost on equal lanes and `mismatch` on the rest as the diagonal substitution penalty, while deletion and
 *  insertion each add the uniform `gap` cost. The recurrence `cell = min(substitution, min(deletion, insertion))`
 *  advances all 8 lanes over `vminq_u16`. This serves the @b non-unit uniform-cost Levenshtein cells; the
 *  unit-cost case (match 0, mismatch 1, gap 1) is handled by the rune Myers fast path and never reaches here.
 *
 *  @note Eight lanes is the natural count here: the cells are `u16` (one `uint16x8_t` per row column holds 8),
 *      matching the byte-keyed `u16` walker. The wider `u32` keys cost two `uint32x4_t` loads/compares per
 *      column instead of one `uint8x8_t`, but the lane width that gates throughput is the score width, not the
 *      key width, so the rune walker keeps 8 lanes rather than dropping to 4.
 *  @note The cells are `u16`, so the kernel is only valid while every reachable score stays below 65535; the
 *      longest reachable cell is the all-gap path `max(query, candidate) * gap`. Enforcing that bound (a
 *      cost-dependent rune length limit) is the caller's dispatch contract; the kernel performs no range check.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<rune_t, u16_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 8, void> {

    using char_t = rune_t;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The `u16` lane recurrence hardcodes `vminq_u16`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 16-bit rune candidate-lane kernel only implements distance minimization (Levenshtein).");

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

        // The recurrence honors the engine's configured uniform costs: `match`/`mismatch` on the diagonal and a
        // single `gap` on either deletion or insertion. The unit-cost case (match 0, mismatch 1, gap 1) is served by
        // the rune Myers fast path, so this kernel only ever runs for non-unit costs.
        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const gap_cost = static_cast<score_t>(gap_costs_.open_or_extend);
        uint16x8_t const match_vec = vdupq_n_u16(match_cost);
        uint16x8_t const mismatch_vec = vdupq_n_u16(mismatch_cost);
        uint16x8_t const gap_vec = vdupq_n_u16(gap_cost);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            vst1q_u16(previous_row + column * candidate_lanes_k, vdupq_n_u16(static_cast<u16_t>(column * gap_cost)));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint32x4_t const query_rune_vec = vdupq_n_u32(static_cast<u32_t>(query[query_position - 1]));
            vst1q_u16(current_row, vdupq_n_u16(static_cast<u16_t>(query_position * gap_cost)));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                // Eight `u32` candidate runes for this column occupy two `uint32x4_t` halves.
                u32_t const *candidate_runes = (u32_t const *)candidates.position(column - 1);
                uint32x4_t const candidate_runes_low = vld1q_u32(candidate_runes);
                uint32x4_t const candidate_runes_high = vld1q_u32(candidate_runes + 4);
                uint16x8_t const diagonal_vec = vld1q_u16(previous_row + (column - 1) * candidate_lanes_k);
                uint16x8_t const deletion_source_vec = vld1q_u16(previous_row + column * candidate_lanes_k);
                uint16x8_t const insertion_source_vec = vld1q_u16(current_row + (column - 1) * candidate_lanes_k);

                // Compare both `u32` halves for rune equality, narrow each mask to `u16`, and recombine into one
                // full-width 8-lane `u16` match mask (`0xFFFF` match / `0x0000` mismatch); `vbslq_u16` then selects
                // `match` on equal lanes and `mismatch` on the rest as the diagonal substitution penalty.
                uint16x4_t const match_low = vmovn_u32(vceqq_u32(query_rune_vec, candidate_runes_low));
                uint16x4_t const match_high = vmovn_u32(vceqq_u32(query_rune_vec, candidate_runes_high));
                uint16x8_t const equal_u16_vec = vcombine_u16(match_low, match_high);
                uint16x8_t const substitution_addend_vec = vbslq_u16(equal_u16_vec, match_vec, mismatch_vec);
                uint16x8_t const cost_if_substitution_vec = vaddq_u16(diagonal_vec, substitution_addend_vec);
                uint16x8_t const cost_if_deletion_vec = vaddq_u16(deletion_source_vec, gap_vec);
                uint16x8_t const cost_if_insertion_vec = vaddq_u16(insertion_source_vec, gap_vec);
                uint16x8_t const cell_score_vec = vminq_u16(cost_if_substitution_vec,
                                                            vminq_u16(cost_if_deletion_vec, cost_if_insertion_vec));
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
 *  @brief Inter-sequence NEON walker: one @b rune (UTF-32) query against up to 4 rune candidates packed
 *         one-per-lane, 32-bit cells, @b non-unit uniform-cost @b Levenshtein.
 *
 *  Width-twin of the 8-lane uniform `u16` linear rune walker, widened to `u32` cells so each `uint32x4_t` holds 4
 *  lanes. The 4 candidate runes occupy a single `uint32x4_t`, compared against the broadcast query rune with one
 *  `vceqq_u32` yielding a full-width `0xFFFFFFFF` (match) / `0x00000000` (mismatch) mask; `vbslq_u32` then selects
 *  the configured `match` cost on equal lanes and `mismatch` on the rest as the diagonal substitution penalty,
 *  while deletion and insertion each add the uniform `gap` cost. The recurrence `cell = min(substitution,
 *  min(deletion, insertion))` advances all 4 lanes over `vminq_u32`. Used when the worst-case score escapes the
 *  `u16` walker's 65535 range but stays below the much wider `u32` ceiling; the unit-cost case is handled by the
 *  rune Myers fast path and never reaches here.
 *
 *  @note The cells are `u32`, so the kernel is only valid while every reachable score stays below 4294967295; the
 *      longest reachable cell is the all-gap path `max(query, candidate) * gap`. Enforcing that bound (a
 *      cost-dependent rune length limit) is the caller's dispatch contract; the kernel performs no range check.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<rune_t, u32_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 4, void> {

    using char_t = rune_t;
    using score_t = u32_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 4;

    // The `u32` lane recurrence hardcodes `vminq_u32`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 32-bit rune candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (4 `u32` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared rune query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 4 rune candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 4`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const gap_cost = static_cast<score_t>(gap_costs_.open_or_extend);
        uint32x4_t const match_vec = vdupq_n_u32(match_cost);
        uint32x4_t const mismatch_vec = vdupq_n_u32(mismatch_cost);
        uint32x4_t const gap_vec = vdupq_n_u32(gap_cost);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            vst1q_u32(previous_row + column * candidate_lanes_k, vdupq_n_u32(static_cast<u32_t>(column * gap_cost)));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint32x4_t const query_rune_vec = vdupq_n_u32(static_cast<u32_t>(query[query_position - 1]));
            vst1q_u32(current_row, vdupq_n_u32(static_cast<u32_t>(query_position * gap_cost)));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                // The 4 `u32` candidate runes for this column fill a single `uint32x4_t`.
                uint32x4_t const candidate_runes_vec = vld1q_u32((u32_t const *)candidates.position(column - 1));
                uint32x4_t const diagonal_vec = vld1q_u32(previous_row + (column - 1) * candidate_lanes_k);
                uint32x4_t const deletion_source_vec = vld1q_u32(previous_row + column * candidate_lanes_k);
                uint32x4_t const insertion_source_vec = vld1q_u32(current_row + (column - 1) * candidate_lanes_k);

                // Compare the 4 candidate runes against the broadcast query rune so the mask is a full-width
                // `0xFFFFFFFF` (match) / `0x00000000` (mismatch); `vbslq_u32` then selects `match` on equal lanes and
                // `mismatch` on the rest as the diagonal substitution penalty.
                uint32x4_t const equal_u32_vec = vceqq_u32(query_rune_vec, candidate_runes_vec);
                uint32x4_t const substitution_addend_vec = vbslq_u32(equal_u32_vec, match_vec, mismatch_vec);
                uint32x4_t const cost_if_substitution_vec = vaddq_u32(diagonal_vec, substitution_addend_vec);
                uint32x4_t const cost_if_deletion_vec = vaddq_u32(deletion_source_vec, gap_vec);
                uint32x4_t const cost_if_insertion_vec = vaddq_u32(insertion_source_vec, gap_vec);
                uint32x4_t const cell_score_vec = vminq_u32(cost_if_substitution_vec,
                                                            vminq_u32(cost_if_deletion_vec, cost_if_insertion_vec));
                vst1q_u32(current_row + column * candidate_lanes_k, cell_score_vec);
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
 *  @brief Inter-sequence NEON walker: one @b rune (UTF-32) query against up to 8 rune candidates packed
 *         one-per-lane, unit-class @b affine-gap Levenshtein distance, 16-bit unsigned cells. Minimization only.
 *
 *  Rune twin of the 8-lane byte `u16` affine walker: the lane keys are 32-bit `rune_t` code points rather than
 *  bytes, the score cells stay `u16` (8 lanes), and the branchless Gotoh E/F tracks over `vminq_u16` are
 *  identical:
 *      F[column] = min(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *      E         = min(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *      M[column] = min(M_diagonal + substitution, min(E, F))
 *  The only delta from the byte walker is the substitution test: 8 candidate runes occupy two `uint32x4_t`, so the
 *  broadcast query rune is compared against them with two `vceqq_u32` masks narrowed and recombined into one
 *  full-width 8-lane `u16` mask, and `vbslq_u16` selects `match` on equal lanes and `mismatch` on the rest. A large
 *  `discard_bias` is added to any track that cannot have been opened yet, keeping `min` from selecting it.
 *
 *  @note The cells are unsigned `u16`, so the kernel is only valid while every reachable score - plus the
 *      `discard_bias` headroom - stays below 65535; enforcing that bound is the caller's dispatch contract.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<rune_t, u16_t, uniform_substitution_costs_t, affine_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 8, void> {

    using char_t = rune_t;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The `u16` lane recurrence hardcodes `vminq_u16`; maximization would need a different blend.
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
        uint16x8_t const match_vec = vdupq_n_u16(match_cost);
        uint16x8_t const mismatch_vec = vdupq_n_u16(mismatch_cost);
        uint16x8_t const open_vec = vdupq_n_u16(open);
        uint16x8_t const extend_vec = vdupq_n_u16(extend);
        // A magnitude above any in-range score so `min` never selects a track that has not been opened yet; the
        // caller's reach bound keeps real scores below it, and `discard_bias + extend` must not wrap `u16`.
        uint16x8_t const discard_bias_vec = vdupq_n_u16(static_cast<u16_t>(60000));

        // Row 0 (global): `M[0] = 0`; `M[column] = open + extend * (column - 1)`. The vertical `F` row cannot have
        // been entered from above at row 0, so it is seeded with the discarded magnitude added to the boundary.
        vst1q_u16(previous_row, vdupq_n_u16(0));
        vst1q_u16(vertical_row, discard_bias_vec);
        for (size_t column = 1; column <= longest_candidate; ++column) {
            uint16x8_t const boundary_vec = vdupq_n_u16(static_cast<u16_t>(open + extend * (u16_t)(column - 1)));
            vst1q_u16(previous_row + column * candidate_lanes_k, boundary_vec);
            vst1q_u16(vertical_row + column * candidate_lanes_k, vaddq_u16(discard_bias_vec, boundary_vec));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint32x4_t const query_rune_vec = vdupq_n_u32(static_cast<u32_t>(query[query_position - 1]));

            // First-column boundary of this row: `M = open + extend * (query_position - 1)`, and the rolling
            // horizontal `E` register reseeded with the discarded magnitude added to that left boundary.
            uint16x8_t const left_boundary_vec = vdupq_n_u16(
                static_cast<u16_t>(open + extend * (u16_t)(query_position - 1)));
            vst1q_u16(current_row, left_boundary_vec);
            uint16x8_t horizontal_vec = vaddq_u16(discard_bias_vec, left_boundary_vec);

            for (size_t column = 1; column <= longest_candidate; ++column) {
                // Eight `u32` candidate runes for this column occupy two `uint32x4_t` halves.
                u32_t const *candidate_runes = (u32_t const *)candidates.position(column - 1);
                uint32x4_t const candidate_runes_low = vld1q_u32(candidate_runes);
                uint32x4_t const candidate_runes_high = vld1q_u32(candidate_runes + 4);
                uint16x8_t const diagonal_vec = vld1q_u16(previous_row + (column - 1) * candidate_lanes_k);
                uint16x8_t const up_vec = vld1q_u16(previous_row + column * candidate_lanes_k);
                uint16x8_t const left_vec = vld1q_u16(current_row + (column - 1) * candidate_lanes_k);
                uint16x8_t const up_vertical_vec = vld1q_u16(vertical_row + column * candidate_lanes_k);

                // Compare both `u32` halves for rune equality, narrow each mask to `u16`, and recombine into one
                // full-width 8-lane `u16` match mask; `vbslq_u16` then selects `match` on equal lanes and `mismatch`
                // on the rest as the diagonal substitution penalty.
                uint16x4_t const match_low = vmovn_u32(vceqq_u32(query_rune_vec, candidate_runes_low));
                uint16x4_t const match_high = vmovn_u32(vceqq_u32(query_rune_vec, candidate_runes_high));
                uint16x8_t const equal_u16_vec = vcombine_u16(match_low, match_high);
                uint16x8_t const substitution_addend_vec = vbslq_u16(equal_u16_vec, match_vec, mismatch_vec);
                uint16x8_t const cost_if_substitution_vec = vaddq_u16(diagonal_vec, substitution_addend_vec);

                // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                uint16x8_t const vertical_vec = vminq_u16(vaddq_u16(up_vec, open_vec),
                                                          vaddq_u16(up_vertical_vec, extend_vec));
                horizontal_vec = vminq_u16(vaddq_u16(left_vec, open_vec), vaddq_u16(horizontal_vec, extend_vec));
                uint16x8_t const cost_if_gap_vec = vminq_u16(vertical_vec, horizontal_vec);

                uint16x8_t const cell_score_vec = vminq_u16(cost_if_substitution_vec, cost_if_gap_vec);
                vst1q_u16(vertical_row + column * candidate_lanes_k, vertical_vec);
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
 *  @brief Inter-sequence NEON walker: one @b rune (UTF-32) query against up to 4 rune candidates packed
 *         one-per-lane, unit-class @b affine-gap Levenshtein distance, 32-bit unsigned cells. Minimization only.
 *
 *  Width-twin of the 8-lane uniform `u16` affine rune walker, widened to `u32` cells so each `uint32x4_t` holds 4
 *  lanes. The 4 candidate runes occupy a single `uint32x4_t`, compared against the broadcast query rune with one
 *  `vceqq_u32` yielding a full-width mask that `vbslq_u32` uses to select `match` on equal lanes and `mismatch` on
 *  the rest, and the single gap term is the branchless Gotoh E/F tracks over `vminq_u32`:
 *      F[column] = min(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *      E         = min(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *      M[column] = min(M_diagonal + substitution, min(E, F))
 *  A large `discard_bias` is added to any track that cannot have been opened yet, keeping `min` from selecting it.
 *
 *  @note The cells are unsigned `u32`, so the kernel is only valid while every reachable score - plus the
 *      `discard_bias` headroom - stays below 4294967295; enforcing that bound is the caller's dispatch contract.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<rune_t, u32_t, uniform_substitution_costs_t, affine_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 4, void> {

    using char_t = rune_t;
    using score_t = u32_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 4;

    // The `u32` lane recurrence hardcodes `vminq_u32`; maximization would need a different blend.
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
     *  @param[in] candidates Transposed block of up to 4 rune candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Three row buffers carved from the byte span; each lane-vector lives at `row + column * 4`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        score_t *vertical_row = current_row + row_stride;

        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const open = static_cast<score_t>(gap_costs_.open);
        score_t const extend = static_cast<score_t>(gap_costs_.extend);
        uint32x4_t const match_vec = vdupq_n_u32(match_cost);
        uint32x4_t const mismatch_vec = vdupq_n_u32(mismatch_cost);
        uint32x4_t const open_vec = vdupq_n_u32(open);
        uint32x4_t const extend_vec = vdupq_n_u32(extend);
        // A magnitude above any in-range score so `min` never selects a track that has not been opened yet; the
        // caller's reach bound keeps real scores below it, and `discard_bias + extend` must not wrap `u32`.
        uint32x4_t const discard_bias_vec = vdupq_n_u32(static_cast<u32_t>(2000000000));

        // Row 0 (global): `M[0] = 0`; `M[column] = open + extend * (column - 1)`. The vertical `F` row cannot have
        // been entered from above at row 0, so it is seeded with the discarded magnitude added to the boundary.
        vst1q_u32(previous_row, vdupq_n_u32(0));
        vst1q_u32(vertical_row, discard_bias_vec);
        for (size_t column = 1; column <= longest_candidate; ++column) {
            uint32x4_t const boundary_vec = vdupq_n_u32(static_cast<u32_t>(open + extend * (u32_t)(column - 1)));
            vst1q_u32(previous_row + column * candidate_lanes_k, boundary_vec);
            vst1q_u32(vertical_row + column * candidate_lanes_k, vaddq_u32(discard_bias_vec, boundary_vec));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint32x4_t const query_rune_vec = vdupq_n_u32(static_cast<u32_t>(query[query_position - 1]));

            // First-column boundary of this row: `M = open + extend * (query_position - 1)`, and the rolling
            // horizontal `E` register reseeded with the discarded magnitude added to that left boundary.
            uint32x4_t const left_boundary_vec = vdupq_n_u32(
                static_cast<u32_t>(open + extend * (u32_t)(query_position - 1)));
            vst1q_u32(current_row, left_boundary_vec);
            uint32x4_t horizontal_vec = vaddq_u32(discard_bias_vec, left_boundary_vec);

            for (size_t column = 1; column <= longest_candidate; ++column) {
                // The 4 `u32` candidate runes for this column fill a single `uint32x4_t`.
                uint32x4_t const candidate_runes_vec = vld1q_u32((u32_t const *)candidates.position(column - 1));
                uint32x4_t const diagonal_vec = vld1q_u32(previous_row + (column - 1) * candidate_lanes_k);
                uint32x4_t const up_vec = vld1q_u32(previous_row + column * candidate_lanes_k);
                uint32x4_t const left_vec = vld1q_u32(current_row + (column - 1) * candidate_lanes_k);
                uint32x4_t const up_vertical_vec = vld1q_u32(vertical_row + column * candidate_lanes_k);

                // Compare the 4 candidate runes against the broadcast query rune so the mask is a full-width
                // `0xFFFFFFFF` (match) / `0x00000000` (mismatch); `vbslq_u32` then selects `match` on equal lanes and
                // `mismatch` on the rest as the diagonal substitution penalty.
                uint32x4_t const equal_u32_vec = vceqq_u32(query_rune_vec, candidate_runes_vec);
                uint32x4_t const substitution_addend_vec = vbslq_u32(equal_u32_vec, match_vec, mismatch_vec);
                uint32x4_t const cost_if_substitution_vec = vaddq_u32(diagonal_vec, substitution_addend_vec);

                // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                uint32x4_t const vertical_vec = vminq_u32(vaddq_u32(up_vec, open_vec),
                                                          vaddq_u32(up_vertical_vec, extend_vec));
                horizontal_vec = vminq_u32(vaddq_u32(left_vec, open_vec), vaddq_u32(horizontal_vec, extend_vec));
                uint32x4_t const cost_if_gap_vec = vminq_u32(vertical_vec, horizontal_vec);

                uint32x4_t const cell_score_vec = vminq_u32(cost_if_substitution_vec, cost_if_gap_vec);
                vst1q_u32(vertical_row + column * candidate_lanes_k, vertical_vec);
                vst1q_u32(current_row + column * candidate_lanes_k, cell_score_vec);
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

#pragma region Weighted Candidate Lane Walker

/**
 *  @brief Inter-sequence NEON walker: one query against up to 8 candidates packed one-per-lane, weighted
 *         Needleman-Wunsch (global) / Smith-Waterman (local) with 32-class substitution costs, 16-bit signed cells.
 *
 *  ONE body per score-width, templated over the gap model (linear vs @b affine) and the locality (global vs @b local)
 *  and resolved entirely with `if constexpr`, so a single hand-tuned NEON kernel covers all four weighted `i16` cases.
 *  The common skeleton is the 8-lane `i16` layout, the `classify16` candidate-class cache computed @b once before the
 *  row loop, and the `lookup16` substitution gather whose low 8 `i8` lanes sign-extend to 8 `i16` cells:
 *
 *  - @b Substitution: `cell_if_substitution = diagonal + substitution`, identical for every variant.
 *  - @b Gap (linear): `cell_if_gap = max(up, left) + gap` over signed `i16` cells with a (typically negative) gap.
 *  - @b Gap (affine): the branchless @b Gotoh recurrence with two extra score tracks - a deletion track `F` carried in
 *    a previous/current row pair and an insertion track `E` carried in a running per-lane register:
 *
 *        F = max(up_cell + open, F_up + extend)      // deletion track, vertical
 *        E = max(left_cell + open, E_left + extend)  // insertion track, horizontal
 *        cell_if_gap = max(E, F)
 *
 *    The boundary gap tracks are seeded with the discarded `open + extend + boundary` sentinel (mirroring the serial
 *    `init_gap`) so the first gap on either axis correctly pays `open`.
 *  - @b Global: the recurrence is `cell = max(cell_if_substitution, cell_if_gap)`; the result is each lane's own final
 *    column (ragged lengths mean different columns per lane).
 *  - @b Local: the cell clamps to zero `cell = max(0, cell_if_substitution, cell_if_gap)`, the boundary row/column are
 *    zero, and the result is a per-lane running maximum, updated only for columns within each lane's candidate length
 *    so a shorter candidate's zero-padded tail columns cannot inflate its score.
 *
 *  @note Signed `i16` cells; valid while every reachable score stays within the caller's contracted `i16` range
 *      (`[-32768, 32767]` global, `[0, 32767]` local).
 *  @note Requires Arm NEON CPUs.
 */
template <typename gap_costs_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, gap_costs_type_, objective_, locality_, sz_cap_neon_k, 8,
                             void> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 8;
    static constexpr bool is_affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    // The signed `i16` recurrence hardcodes `vmaxq_s16`; minimization would need a different blend.
    static_assert(
        objective_ == sz_maximize_score_k,
        "The weighted candidate-lane kernel only implements score " "maximization (Needleman-Wunsch / " "Smith-" "Water" "man)" ".");

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Scratch holds the score rows of `longest_candidate + 1` lane-vectors (8 `i16` cells each), plus one
     *      cached candidate-class lane-vector (8 `u8`) per candidate position. The linear gap needs two score rows;
     *      the affine gap needs two more deletion-track rows (the insertion track is a running per-lane register).
     */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        size_t const class_bytes = candidate_lanes_k * longest_candidate * sizeof(u8_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous score row
        amount += score_row_bytes; // current score row
        if constexpr (is_affine_k) {
            amount += score_row_bytes; // previous deletion-track row (F_up)
            amount += score_row_bytes; // current deletion-track row (F)
        }
        amount += class_bytes; // cached candidate classes
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

        // Carve the score rows (and, for affine, the deletion-track rows) from the byte span, then the class cache.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        score_t *previous_deletes = nullptr;
        score_t *current_deletes = nullptr;
        u8_t *candidate_classes;
        if constexpr (is_affine_k) {
            previous_deletes = current_row + row_stride;
            current_deletes = previous_deletes + row_stride;
            candidate_classes = reinterpret_cast<u8_t *>(current_deletes + row_stride);
        }
        else { candidate_classes = reinterpret_cast<u8_t *>(current_row + row_stride); }

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

        int16x8_t const zero_vec = vdupq_n_s16(0);
        // Gap-cost broadcasts: linear keeps a single `gap`, affine keeps `open` / `extend`. `gap` doubles as `open`
        // (with `extend` zero) so the boundary fills and the local-affine sentinel share one expression per variant.
        [[maybe_unused]] int16x8_t gap_vec {};
        [[maybe_unused]] int16x8_t open_vec {};
        [[maybe_unused]] int16x8_t extend_vec {};
        error_cost_t open = 0;
        error_cost_t extend = 0;
        if constexpr (is_affine_k) {
            open = gap_costs_.open;
            extend = gap_costs_.extend;
            open_vec = vdupq_n_s16(static_cast<i16_t>(open));
            extend_vec = vdupq_n_s16(static_cast<i16_t>(extend));
        }
        else {
            open = gap_costs_.open_or_extend;
            gap_vec = vdupq_n_s16(static_cast<i16_t>(open));
        }
        // The discarded gap-track sentinel for local affine: extending it never beats opening fresh.
        [[maybe_unused]] int16x8_t const sentinel_vec = vdupq_n_s16(static_cast<i16_t>(open + extend));

        // Per-lane candidate lengths gate the local running-maximum so a shorter candidate's padded columns are
        // excluded; the global latch reads each lane's own final column instead and needs neither.
        [[maybe_unused]] alignas(16) i16_t lane_lengths[candidate_lanes_k] = {0};
        [[maybe_unused]] int16x8_t lane_lengths_vec = zero_vec;
        [[maybe_unused]] int16x8_t running_max_vec = zero_vec;
        if constexpr (is_local_k) {
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                lane_lengths[lane_index] = static_cast<i16_t>(candidates.lengths[lane_index]);
            lane_lengths_vec = vld1q_s16(lane_lengths);
        }

        // Row 0 (boundary). Local: the boundary row and column are zero (an alignment may begin at any cell). Global:
        // the empty query prefix against every candidate prefix is an all-gap run.
        for (size_t column = 0; column <= longest_candidate; ++column) {
            if constexpr (is_local_k) {
                vst1q_s16(previous_row + column * candidate_lanes_k, zero_vec);
                if constexpr (is_affine_k) vst1q_s16(previous_deletes + column * candidate_lanes_k, sentinel_vec);
            }
            else if constexpr (is_affine_k) {
                // Column 0 is 0; column j (>= 1) costs `open + extend * (j - 1)`. The deletion track is seeded with
                // the discarded `open + extend + boundary` sentinel so the first vertical gap in row 1 pays `open`.
                i16_t const boundary = static_cast<i16_t>(column ? open + extend * (i16_t)(column - 1) : 0);
                vst1q_s16(previous_row + column * candidate_lanes_k, vdupq_n_s16(boundary));
                vst1q_s16(previous_deletes + column * candidate_lanes_k,
                          vdupq_n_s16(static_cast<i16_t>(open + extend + boundary)));
            }
            else {
                // Linear: a run of `gap * column`, identical across lanes (masked per-lane at latch time).
                vst1q_s16(previous_row + column * candidate_lanes_k,
                          vdupq_n_s16(static_cast<i16_t>(open * (i16_t)column)));
            }
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            // The query character is fixed for the whole row, so its class is scalar and broadcast across lanes.
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            uint8x16_t const query_class_vec = vdupq_n_u8(query_class);

            [[maybe_unused]] int16x8_t running_inserts_vec = zero_vec;
            // Column 0 (left boundary). Local: zero. Global linear: `gap * query_position`. Global affine: the
            // all-deletion cell `open + extend * (query_position - 1)`; the insertion track resets to the sentinel.
            if constexpr (is_local_k) {
                vst1q_s16(current_row, zero_vec);
                if constexpr (is_affine_k) {
                    vst1q_s16(current_deletes, sentinel_vec);
                    running_inserts_vec = sentinel_vec;
                }
            }
            else if constexpr (is_affine_k) {
                i16_t const row_boundary = static_cast<i16_t>(open + extend * (i16_t)(query_position - 1));
                vst1q_s16(current_row, vdupq_n_s16(row_boundary));
                vst1q_s16(current_deletes, vdupq_n_s16(static_cast<i16_t>(open + extend + row_boundary)));
                running_inserts_vec = vdupq_n_s16(static_cast<i16_t>(open + extend + row_boundary));
            }
            else { vst1q_s16(current_row, vdupq_n_s16(static_cast<i16_t>(open * (i16_t)query_position))); }

            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x8_t const candidate_classes_low = vld1_u8(candidate_classes + (column - 1) * candidate_lanes_k);
                uint8x16_t const candidate_classes_vec = vcombine_u8(candidate_classes_low, vdup_n_u8(0));
                int16x8_t const diagonal_vec = vld1q_s16(previous_row + (column - 1) * candidate_lanes_k);
                int16x8_t const up_vec = vld1q_s16(previous_row + column * candidate_lanes_k);
                int16x8_t const left_vec = vld1q_s16(current_row + (column - 1) * candidate_lanes_k);

                int16x8_t cost_if_gap_vec;
                if constexpr (is_affine_k) {
                    int16x8_t const up_deletes_vec = vld1q_s16(previous_deletes + column * candidate_lanes_k);
                    // Gotoh deletion track (vertical): open a gap from the cell above, or extend the running deletion.
                    int16x8_t const delete_vec = vmaxq_s16(vaddq_s16(up_vec, open_vec),
                                                           vaddq_s16(up_deletes_vec, extend_vec));
                    // Gotoh insertion track (horizontal): open a gap from the cell to the left, or extend the running one.
                    int16x8_t const insert_vec = vmaxq_s16(vaddq_s16(left_vec, open_vec),
                                                           vaddq_s16(running_inserts_vec, extend_vec));
                    running_inserts_vec = insert_vec;
                    vst1q_s16(current_deletes + column * candidate_lanes_k, delete_vec);
                    cost_if_gap_vec = vmaxq_s16(delete_vec, insert_vec);
                }
                else { cost_if_gap_vec = vaddq_s16(vmaxq_s16(up_vec, left_vec), gap_vec); }

                // Gather 16 substitution costs and sign-extend the low 8 `i8` lanes into 8 `i16` cells.
                int8x16_t const cost_i8_vec = lookup.lookup16(query_class_vec, candidate_classes_vec);
                int16x8_t const cost_i16_vec = vmovl_s8(vget_low_s8(cost_i8_vec));

                int16x8_t const cost_if_substitution_vec = vaddq_s16(diagonal_vec, cost_i16_vec);
                int16x8_t cell_score_vec = vmaxq_s16(cost_if_substitution_vec, cost_if_gap_vec);
                if constexpr (is_local_k) cell_score_vec = vmaxq_s16(zero_vec, cell_score_vec);
                vst1q_s16(current_row + column * candidate_lanes_k, cell_score_vec);

                if constexpr (is_local_k) {
                    // Fold into the running maximum only on lanes whose candidate reaches this column. `vcgtq_s16`'s
                    // `uint16x8_t` mask feeds `vbslq_s16` natively, keeping the new max on live lanes, the rest as-is.
                    uint16x8_t const column_live = vcgtq_s16(lane_lengths_vec,
                                                             vdupq_n_s16(static_cast<i16_t>(column - 1)));
                    running_max_vec = vbslq_s16(column_live, vmaxq_s16(running_max_vec, cell_score_vec),
                                                running_max_vec);
                }
            }
            trivial_swap(previous_row, current_row);
            if constexpr (is_affine_k) trivial_swap(previous_deletes, current_deletes);
        }

        if constexpr (is_local_k) {
            alignas(16) i16_t final_max[candidate_lanes_k];
            vst1q_s16(final_max, running_max_vec);
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
 *  @brief Inter-sequence NEON walker: one query against up to 4 candidates packed one-per-lane, weighted
 *         (32 x 32) substitution Needleman-Wunsch / Smith-Waterman with signed 32-bit cells. Maximization over
 *         `vmaxq_s32`.
 *
 *  Width-mirror of the 8-lane `i16` weighted walker, widened to `i32` cells so each `int32x4_t` holds 4 lanes. The
 *  candidate characters remain `char`/`u8`: only the low 4 of every loaded column carry live candidates, so the
 *  per-column class cache stores 4 `u8` classes and the per-cell substitution costs (gathered as `i8` by the shared
 *  table lookup) sign-extend through the `vmovl` chain into 4 `i32` cells. Halving the lane count to 4 doubles the
 *  representable score range to the signed `i32` window, so this serves candidates whose reachable scores exceed the
 *  `i16` contract while staying within `[-2147483648, 2147483647]` global (`[0, 2147483647]` local). Linear and affine
 *  gaps, global and local, all share this body exactly as the `i16` sibling does.
 *
 *  @note Signed `i32` cells; valid while every reachable score stays within the caller's contracted `i32` range.
 *  @note Requires Arm NEON CPUs.
 */
template <typename gap_costs_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct candidate_lane_walker<char, i32_t, error_costs_32x32_t, gap_costs_type_, objective_, locality_, sz_cap_neon_k, 4,
                             void> {

    using char_t = char;
    using score_t = i32_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 4;
    static constexpr bool is_affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    // The signed `i32` recurrence hardcodes `vmaxq_s32`; minimization would need a different blend.
    static_assert(
        objective_ == sz_maximize_score_k,
        "The weighted candidate-lane kernel only implements score " "maximization (Needleman-Wunsch / " "Smith-" "Water" "man)" ".");

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Scratch holds the score rows of `longest_candidate + 1` lane-vectors (4 `i32` cells each), plus one
     *      cached candidate-class lane-vector (4 `u8`) per candidate position. The linear gap needs two score rows;
     *      the affine gap needs two more deletion-track rows (the insertion track is a running per-lane register).
     */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        // One extra lane-vector pads the class cache so the per-column `vld1_u8` (8 bytes) on the final column,
        // which only consumes its low 4 lanes, never reads past the buffer end.
        size_t const class_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(u8_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous score row
        amount += score_row_bytes; // current score row
        if constexpr (is_affine_k) {
            amount += score_row_bytes; // previous deletion-track row (F_up)
            amount += score_row_bytes; // current deletion-track row (F)
        }
        amount += class_bytes; // cached candidate classes
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 4 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Carve the score rows (and, for affine, the deletion-track rows) from the byte span, then the class cache.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        score_t *previous_deletes = nullptr;
        score_t *current_deletes = nullptr;
        u8_t *candidate_classes;
        if constexpr (is_affine_k) {
            previous_deletes = current_row + row_stride;
            current_deletes = previous_deletes + row_stride;
            candidate_classes = reinterpret_cast<u8_t *>(current_deletes + row_stride);
        }
        else { candidate_classes = reinterpret_cast<u8_t *>(current_row + row_stride); }

        // The (32 x 32) cost matrix is symmetric across the two operand orderings here, so no transpose is needed:
        // the query class is the first operand and the candidate class the second, matching `class_substitution_costs`.
        substitution_lookup_neon_t lookup;
        lookup.reload_classes(substituter_.byte_to_class);
        lookup.reload_costs(substituter_.class_substitution_costs, false);

        // Pre-classify every candidate column once; the 4 candidate bytes per column are loop-invariant in rows.
        // Only the low 4 lanes carry live candidates: we classify a 16-lane vector and store its low 8 classes at the
        // 4-wide column stride. Successive columns overwrite the spilled high 4 lanes with their own low 4; the padded
        // class cache (see `scratch_space_needed`) absorbs the final column's spill.
        for (size_t column = 0; column < longest_candidate; ++column) {
            uint8x8_t const candidate_chars_low = vld1_u8((u8_t const *)candidates.position(column));
            uint8x16_t const candidate_chars_vec = vcombine_u8(candidate_chars_low, vdup_n_u8(0));
            uint8x16_t const candidate_classes_vec = lookup.classify16(candidate_chars_vec);
            vst1_u8(candidate_classes + column * candidate_lanes_k, vget_low_u8(candidate_classes_vec));
        }

        int32x4_t const zero_vec = vdupq_n_s32(0);
        // Gap-cost broadcasts: linear keeps a single `gap`, affine keeps `open` / `extend`. `gap` doubles as `open`
        // (with `extend` zero) so the boundary fills and the local-affine sentinel share one expression per variant.
        [[maybe_unused]] int32x4_t gap_vec {};
        [[maybe_unused]] int32x4_t open_vec {};
        [[maybe_unused]] int32x4_t extend_vec {};
        error_cost_t open = 0;
        error_cost_t extend = 0;
        if constexpr (is_affine_k) {
            open = gap_costs_.open;
            extend = gap_costs_.extend;
            open_vec = vdupq_n_s32(static_cast<i32_t>(open));
            extend_vec = vdupq_n_s32(static_cast<i32_t>(extend));
        }
        else {
            open = gap_costs_.open_or_extend;
            gap_vec = vdupq_n_s32(static_cast<i32_t>(open));
        }
        // The discarded gap-track sentinel for local affine: extending it never beats opening fresh.
        [[maybe_unused]] int32x4_t const sentinel_vec = vdupq_n_s32(static_cast<i32_t>(open + extend));

        // Per-lane candidate lengths gate the local running-maximum so a shorter candidate's padded columns are
        // excluded; the global latch reads each lane's own final column instead and needs neither.
        [[maybe_unused]] alignas(16) i32_t lane_lengths[candidate_lanes_k] = {0};
        [[maybe_unused]] int32x4_t lane_lengths_vec = zero_vec;
        [[maybe_unused]] int32x4_t running_max_vec = zero_vec;
        if constexpr (is_local_k) {
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                lane_lengths[lane_index] = static_cast<i32_t>(candidates.lengths[lane_index]);
            lane_lengths_vec = vld1q_s32(lane_lengths);
        }

        // Row 0 (boundary). Local: the boundary row and column are zero (an alignment may begin at any cell). Global:
        // the empty query prefix against every candidate prefix is an all-gap run.
        for (size_t column = 0; column <= longest_candidate; ++column) {
            if constexpr (is_local_k) {
                vst1q_s32(previous_row + column * candidate_lanes_k, zero_vec);
                if constexpr (is_affine_k) vst1q_s32(previous_deletes + column * candidate_lanes_k, sentinel_vec);
            }
            else if constexpr (is_affine_k) {
                // Column 0 is 0; column j (>= 1) costs `open + extend * (j - 1)`. The deletion track is seeded with
                // the discarded `open + extend + boundary` sentinel so the first vertical gap in row 1 pays `open`.
                i32_t const boundary = static_cast<i32_t>(column ? open + extend * (i32_t)(column - 1) : 0);
                vst1q_s32(previous_row + column * candidate_lanes_k, vdupq_n_s32(boundary));
                vst1q_s32(previous_deletes + column * candidate_lanes_k,
                          vdupq_n_s32(static_cast<i32_t>(open + extend + boundary)));
            }
            else {
                // Linear: a run of `gap * column`, identical across lanes (masked per-lane at latch time).
                vst1q_s32(previous_row + column * candidate_lanes_k,
                          vdupq_n_s32(static_cast<i32_t>(open * (i32_t)column)));
            }
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            // The query character is fixed for the whole row, so its class is scalar and broadcast across lanes.
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            uint8x16_t const query_class_vec = vdupq_n_u8(query_class);

            [[maybe_unused]] int32x4_t running_inserts_vec = zero_vec;
            // Column 0 (left boundary). Local: zero. Global linear: `gap * query_position`. Global affine: the
            // all-deletion cell `open + extend * (query_position - 1)`; the insertion track resets to the sentinel.
            if constexpr (is_local_k) {
                vst1q_s32(current_row, zero_vec);
                if constexpr (is_affine_k) {
                    vst1q_s32(current_deletes, sentinel_vec);
                    running_inserts_vec = sentinel_vec;
                }
            }
            else if constexpr (is_affine_k) {
                i32_t const row_boundary = static_cast<i32_t>(open + extend * (i32_t)(query_position - 1));
                vst1q_s32(current_row, vdupq_n_s32(row_boundary));
                vst1q_s32(current_deletes, vdupq_n_s32(static_cast<i32_t>(open + extend + row_boundary)));
                running_inserts_vec = vdupq_n_s32(static_cast<i32_t>(open + extend + row_boundary));
            }
            else { vst1q_s32(current_row, vdupq_n_s32(static_cast<i32_t>(open * (i32_t)query_position))); }

            for (size_t column = 1; column <= longest_candidate; ++column) {
                // Only the low 4 lanes carry live classes; the `vld1_u8` reads 8 bytes (a padded class lane-vector
                // backs the final column's over-read) and the `vmovl` chain discards the high 4 lanes.
                uint8x8_t const candidate_classes_low = vld1_u8(candidate_classes + (column - 1) * candidate_lanes_k);
                uint8x16_t const candidate_classes_vec = vcombine_u8(candidate_classes_low, vdup_n_u8(0));
                int32x4_t const diagonal_vec = vld1q_s32(previous_row + (column - 1) * candidate_lanes_k);
                int32x4_t const up_vec = vld1q_s32(previous_row + column * candidate_lanes_k);
                int32x4_t const left_vec = vld1q_s32(current_row + (column - 1) * candidate_lanes_k);

                int32x4_t cost_if_gap_vec;
                if constexpr (is_affine_k) {
                    int32x4_t const up_deletes_vec = vld1q_s32(previous_deletes + column * candidate_lanes_k);
                    // Gotoh deletion track (vertical): open a gap from the cell above, or extend the running deletion.
                    int32x4_t const delete_vec = vmaxq_s32(vaddq_s32(up_vec, open_vec),
                                                           vaddq_s32(up_deletes_vec, extend_vec));
                    // Gotoh insertion track (horizontal): open a gap from the cell to the left, or extend the running one.
                    int32x4_t const insert_vec = vmaxq_s32(vaddq_s32(left_vec, open_vec),
                                                           vaddq_s32(running_inserts_vec, extend_vec));
                    running_inserts_vec = insert_vec;
                    vst1q_s32(current_deletes + column * candidate_lanes_k, delete_vec);
                    cost_if_gap_vec = vmaxq_s32(delete_vec, insert_vec);
                }
                else { cost_if_gap_vec = vaddq_s32(vmaxq_s32(up_vec, left_vec), gap_vec); }

                // Gather 16 substitution costs and sign-extend the low 4 `i8` lanes through the `vmovl` chain into 4
                // `i32` cells (`i8` -> `i16` -> `i32`).
                int8x16_t const cost_i8_vec = lookup.lookup16(query_class_vec, candidate_classes_vec);
                int16x8_t const cost_i16_vec = vmovl_s8(vget_low_s8(cost_i8_vec));
                int32x4_t const cost_i32_vec = vmovl_s16(vget_low_s16(cost_i16_vec));

                int32x4_t const cost_if_substitution_vec = vaddq_s32(diagonal_vec, cost_i32_vec);
                int32x4_t cell_score_vec = vmaxq_s32(cost_if_substitution_vec, cost_if_gap_vec);
                if constexpr (is_local_k) cell_score_vec = vmaxq_s32(zero_vec, cell_score_vec);
                vst1q_s32(current_row + column * candidate_lanes_k, cell_score_vec);

                if constexpr (is_local_k) {
                    // Fold into the running maximum only on lanes whose candidate reaches this column. `vcgtq_s32`'s
                    // `uint32x4_t` mask feeds `vbslq_s32` natively, keeping the new max on live lanes, the rest as-is.
                    uint32x4_t const column_live = vcgtq_s32(lane_lengths_vec,
                                                             vdupq_n_s32(static_cast<i32_t>(column - 1)));
                    running_max_vec = vbslq_s32(column_live, vmaxq_s32(running_max_vec, cell_score_vec),
                                                running_max_vec);
                }
            }
            trivial_swap(previous_row, current_row);
            if constexpr (is_affine_k) trivial_swap(previous_deletes, current_deletes);
        }

        if constexpr (is_local_k) {
            alignas(16) i32_t final_max[candidate_lanes_k];
            vst1q_s32(final_max, running_max_vec);
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

#pragma endregion Weighted Candidate Lane Walker

/**
 *  @brief Inter-sequence NEON walker: one query against up to 8 candidates packed one-per-lane, unit-substitution
 *         @b affine-gap Levenshtein with unsigned 16-bit cells. Minimization over `vminq_u16`.
 *
 *  Affine sibling of the 8-lane uniform `u16` linear walker: the diagonal term keeps the `match`/`mismatch` blend
 *  (a `vceqq_u16` mask over the candidate bytes widened to `u16`, selecting `match` on equal lanes and `mismatch`
 *  on the rest), but the single gap term is replaced by the branchless Gotoh E/F tracks of the signed weighted
 *  affine walker, here over `vminq_u16` so the recurrence stays in non-negative distance space:
 *      F[column] = min(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *      E         = min(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *      M[column] = min(M_diagonal + substitution, min(E, F))
 *  The `F` track is materialized as a third scratch row indexed exactly like `M`; the `E` track only depends on the
 *  cell to its left, so it lives in a single rolling lane-register reseeded per row from the discarded boundary. A
 *  large `discard_bias` is added to any track that cannot have been opened yet, keeping `min` from selecting it.
 *
 *  @note The cells are unsigned `u16`, so the kernel is only valid while every reachable score - plus the
 *      `discard_bias` headroom - stays below 65535; enforcing that bound is the caller's dispatch contract.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, affine_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 8, void> {

    using char_t = char;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 8;

    // The `u16` lane recurrence hardcodes `vminq_u16`; maximization would need a different blend.
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
        uint16x8_t const match_vec = vdupq_n_u16(match_cost);
        uint16x8_t const mismatch_vec = vdupq_n_u16(mismatch_cost);
        uint16x8_t const open_vec = vdupq_n_u16(open);
        uint16x8_t const extend_vec = vdupq_n_u16(extend);
        // A magnitude above any in-range score so `min` never selects a track that has not been opened yet; the
        // caller's reach bound keeps real scores below it, and `discard_bias + extend` must not wrap `u16`.
        uint16x8_t const discard_bias_vec = vdupq_n_u16(static_cast<u16_t>(60000));

        // Row 0 (global): `M[0] = 0`; `M[column] = open + extend * (column - 1)`. The vertical `F` row cannot have
        // been entered from above at row 0, so it is seeded with the discarded magnitude added to the boundary.
        vst1q_u16(previous_row, vdupq_n_u16(0));
        vst1q_u16(vertical_row, discard_bias_vec);
        for (size_t column = 1; column <= longest_candidate; ++column) {
            uint16x8_t const boundary_vec = vdupq_n_u16(static_cast<u16_t>(open + extend * (u16_t)(column - 1)));
            vst1q_u16(previous_row + column * candidate_lanes_k, boundary_vec);
            vst1q_u16(vertical_row + column * candidate_lanes_k, vaddq_u16(discard_bias_vec, boundary_vec));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint16x8_t const query_char_vec = vdupq_n_u16(static_cast<u16_t>((u8_t)query[query_position - 1]));

            // First-column boundary of this row: `M = open + extend * (query_position - 1)`, and the rolling
            // horizontal `E` register reseeded with the discarded magnitude added to that left boundary.
            uint16x8_t const left_boundary_vec = vdupq_n_u16(
                static_cast<u16_t>(open + extend * (u16_t)(query_position - 1)));
            vst1q_u16(current_row, left_boundary_vec);
            uint16x8_t horizontal_vec = vaddq_u16(discard_bias_vec, left_boundary_vec);

            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x8_t const candidate_chars_u8_vec = vld1_u8((u8_t const *)candidates.position(column - 1));
                uint16x8_t const candidate_chars_vec = vmovl_u8(candidate_chars_u8_vec);
                uint16x8_t const diagonal_vec = vld1q_u16(previous_row + (column - 1) * candidate_lanes_k);
                uint16x8_t const up_vec = vld1q_u16(previous_row + column * candidate_lanes_k);
                uint16x8_t const left_vec = vld1q_u16(current_row + (column - 1) * candidate_lanes_k);
                uint16x8_t const up_vertical_vec = vld1q_u16(vertical_row + column * candidate_lanes_k);

                // Widen the 8 candidate bytes to `u16` and compare against the broadcast query character so the mask
                // is a full-width `0xFFFF` (match) / `0x0000` (mismatch); `vbslq_u16` then selects `match` on equal
                // lanes and `mismatch` on the rest as the diagonal substitution penalty.
                uint16x8_t const equal_u16_vec = vceqq_u16(query_char_vec, candidate_chars_vec);
                uint16x8_t const substitution_addend_vec = vbslq_u16(equal_u16_vec, match_vec, mismatch_vec);
                uint16x8_t const cost_if_substitution_vec = vaddq_u16(diagonal_vec, substitution_addend_vec);

                // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                uint16x8_t const vertical_vec = vminq_u16(vaddq_u16(up_vec, open_vec),
                                                          vaddq_u16(up_vertical_vec, extend_vec));
                horizontal_vec = vminq_u16(vaddq_u16(left_vec, open_vec), vaddq_u16(horizontal_vec, extend_vec));
                uint16x8_t const cost_if_gap_vec = vminq_u16(vertical_vec, horizontal_vec);

                uint16x8_t const cell_score_vec = vminq_u16(cost_if_substitution_vec, cost_if_gap_vec);
                vst1q_u16(vertical_row + column * candidate_lanes_k, vertical_vec);
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
 *  @brief Inter-sequence NEON walker: one query against up to 4 candidates packed one-per-lane, unit-substitution
 *         @b affine-gap Levenshtein with unsigned 32-bit cells. Minimization over `vminq_u32`.
 *
 *  Width-mirror of the 8-lane affine `u16` walker, widened to `u32` cells so each `uint32x4_t` holds 4 lanes. The
 *  diagonal term keeps the `match`/`mismatch` blend (a `vceqq_u32` mask over the 4 candidate bytes widened to `u32`,
 *  selecting `match` on equal lanes and `mismatch` on the rest), and the gap term keeps the branchless Gotoh E/F
 *  tracks over `vminq_u32` so the recurrence stays in non-negative distance space:
 *      F[column] = min(M_up + open,   F_up + extend)        // vertical track: a gap in the candidate
 *      E         = min(M_left + open, E_left + extend)       // horizontal track: a gap in the query
 *      M[column] = min(M_diagonal + substitution, min(E, F))
 *  The `F` track is materialized as a third scratch row indexed exactly like `M`; the `E` track only depends on the
 *  cell to its left, so it lives in a single rolling lane-register reseeded per row from the discarded boundary. A
 *  large `discard_bias` is added to any track that cannot have been opened yet, keeping `min` from selecting it.
 *  Halving the lane count to 4 quadruples the representable score range to the `u32` window, serving candidates whose
 *  reachable scores exceed the `u16` contract.
 *
 *  @note The cells are unsigned `u32`, so the kernel is only valid while every reachable score - plus the
 *      `discard_bias` headroom - stays below 4294967295; enforcing that bound is the caller's dispatch contract.
 *  @note Requires Arm NEON CPUs.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u32_t, uniform_substitution_costs_t, affine_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_neon_k, 4, void> {

    using char_t = char;
    using score_t = u32_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr size_t candidate_lanes_k = 4;

    // The `u32` lane recurrence hardcodes `vminq_u32`; maximization would need a different blend.
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
     *  @param[in] candidates Transposed block of up to 4 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Three row buffers carved from the byte span; each lane-vector lives at `row + column * 4`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        score_t *vertical_row = current_row + row_stride;

        score_t const match_cost = static_cast<score_t>(substituter_.match);
        score_t const mismatch_cost = static_cast<score_t>(substituter_.mismatch);
        score_t const open = static_cast<score_t>(gap_costs_.open);
        score_t const extend = static_cast<score_t>(gap_costs_.extend);
        uint32x4_t const match_vec = vdupq_n_u32(match_cost);
        uint32x4_t const mismatch_vec = vdupq_n_u32(mismatch_cost);
        uint32x4_t const open_vec = vdupq_n_u32(open);
        uint32x4_t const extend_vec = vdupq_n_u32(extend);
        // A magnitude above any in-range score so `min` never selects a track that has not been opened yet; the
        // caller's reach bound keeps real scores below it, and `discard_bias + extend` must not wrap `u32`.
        uint32x4_t const discard_bias_vec = vdupq_n_u32(static_cast<u32_t>(2000000000));

        // Row 0 (global): `M[0] = 0`; `M[column] = open + extend * (column - 1)`. The vertical `F` row cannot have
        // been entered from above at row 0, so it is seeded with the discarded magnitude added to the boundary.
        vst1q_u32(previous_row, vdupq_n_u32(0));
        vst1q_u32(vertical_row, discard_bias_vec);
        for (size_t column = 1; column <= longest_candidate; ++column) {
            uint32x4_t const boundary_vec = vdupq_n_u32(static_cast<u32_t>(open + extend * (u32_t)(column - 1)));
            vst1q_u32(previous_row + column * candidate_lanes_k, boundary_vec);
            vst1q_u32(vertical_row + column * candidate_lanes_k, vaddq_u32(discard_bias_vec, boundary_vec));
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            uint32x4_t const query_char_vec = vdupq_n_u32(static_cast<u32_t>((u8_t)query[query_position - 1]));

            // First-column boundary of this row: `M = open + extend * (query_position - 1)`, and the rolling
            // horizontal `E` register reseeded with the discarded magnitude added to that left boundary.
            uint32x4_t const left_boundary_vec = vdupq_n_u32(
                static_cast<u32_t>(open + extend * (u32_t)(query_position - 1)));
            vst1q_u32(current_row, left_boundary_vec);
            uint32x4_t horizontal_vec = vaddq_u32(discard_bias_vec, left_boundary_vec);

            for (size_t column = 1; column <= longest_candidate; ++column) {
                uint8x8_t const candidate_chars_u8_vec = vld1_u8((u8_t const *)candidates.position(column - 1));
                uint16x8_t const candidate_chars_u16_vec = vmovl_u8(candidate_chars_u8_vec);
                uint32x4_t const candidate_chars_vec = vmovl_u16(vget_low_u16(candidate_chars_u16_vec));
                uint32x4_t const diagonal_vec = vld1q_u32(previous_row + (column - 1) * candidate_lanes_k);
                uint32x4_t const up_vec = vld1q_u32(previous_row + column * candidate_lanes_k);
                uint32x4_t const left_vec = vld1q_u32(current_row + (column - 1) * candidate_lanes_k);
                uint32x4_t const up_vertical_vec = vld1q_u32(vertical_row + column * candidate_lanes_k);

                // Widen the 4 candidate bytes to `u32` and compare against the broadcast query character so the mask
                // is a full-width `0xFFFFFFFF` (match) / `0x00000000` (mismatch); `vbslq_u32` then selects `match` on
                // equal lanes and `mismatch` on the rest as the diagonal substitution penalty.
                uint32x4_t const equal_u32_vec = vceqq_u32(query_char_vec, candidate_chars_vec);
                uint32x4_t const substitution_addend_vec = vbslq_u32(equal_u32_vec, match_vec, mismatch_vec);
                uint32x4_t const cost_if_substitution_vec = vaddq_u32(diagonal_vec, substitution_addend_vec);

                // Gotoh tracks: vertical `F` from the cell above, horizontal `E` from the cell to the left.
                uint32x4_t const vertical_vec = vminq_u32(vaddq_u32(up_vec, open_vec),
                                                          vaddq_u32(up_vertical_vec, extend_vec));
                horizontal_vec = vminq_u32(vaddq_u32(left_vec, open_vec), vaddq_u32(horizontal_vec, extend_vec));
                uint32x4_t const cost_if_gap_vec = vminq_u32(vertical_vec, horizontal_vec);

                uint32x4_t const cell_score_vec = vminq_u32(cost_if_substitution_vec, cost_if_gap_vec);
                vst1q_u32(vertical_row + column * candidate_lanes_k, vertical_vec);
                vst1q_u32(current_row + column * candidate_lanes_k, cell_score_vec);
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
 *  @brief Batched byte-level @b Levenshtein distances on NEON, scoring two unit-cost pairs at once with the @b NEON
 *      Myers family - `distances_2x64_` (shorter <= 64), `distances_2x_multiword_<words_count>` (compile-time word
 *      count, shorter <= 512), and `distances_2x_multiword_large_` (runtime word count, shorter > 512) - and the
 *      anti-diagonal DP for everything else (non-unit costs, or a lone shorter side > 512). Cells are grouped by their
 *      exact `ceil(shorter / 64)` word bucket so each group loops one common word count.
 *
 *  Mirrors the Ice Lake `levenshtein_distances` cross-product engine's structure - live cells walked query-major,
 *  grouped into same-bucket blocks, scored by the bit-parallel Myers integers, and scattered into the strided result
 *  matrix (plus the mirror slot for symmetric self-similarity) - narrowed from Ice Lake's eight `u64` lanes to the
 *  two `uint64x2_t` lanes NEON carries. Non-unit substitution or gap costs route to the serial per-pair
 *  cross-product helpers.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 8;          // ? `u16` lanes for the non-unit candidate-lane walker.
    static constexpr size_t u16_reach_limit_k = 60000;      // ? `u16` headroom for the non-unit lane walker.
    static constexpr size_t u32_reach_limit_k = 1500000000; // ? `u32` headroom for the wide non-unit lane walker.

    using scoring_t = levenshtein_distance<char, gap_costs_t, capability_k>; // ? Per-pair DP fallback.
    using myers_t = levenshtein_distance_myers<char, capability_k>;          // ? NEON lockstep Myers.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_neon_k, (int)candidate_lanes_k,
                              void>; // ? NEON non-unit shared-query `u16` lanes.
    using lane_walker_wide_t = candidate_lane_walker<char, u32_t, uniform_substitution_costs_t, gap_costs_t,
                                                     sz_minimize_distance_k, sz_similarity_global_k, sz_cap_neon_k, 4,
                                                     void>; // ? NEON non-unit shared-query `u32` lanes.
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

    /** @brief Whether a `(query, candidate)` cell's worst-case distance stays inside the lane walker's `u16` headroom. */
    bool fits_u16_(size_t query_length, size_t candidate_length) const noexcept {
        size_t const magnitude = sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open_or_extend);
        return (query_length + candidate_length) * magnitude <= u16_reach_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case distance stays inside the wide walker's `u32` headroom. */
    bool fits_u32_(size_t query_length, size_t candidate_length) const noexcept {
        size_t const magnitude = sz_max_of_two((size_t)substituter_.mismatch, (size_t)gap_costs_.open_or_extend);
        return (query_length + candidate_length) * magnitude <= u32_reach_limit_k;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the Myers `match_masks` for the longest
     *      string, or the anti-diagonal DP fallback for the longest query x longest candidate (a real cell in both
     *      the full and symmetric grids, so this is a safe upper bound for every slice).
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
        size_t const myers_scratch = myers_t::match_masks_bytes_k;
        size_t dp_scratch = 0, twoxN_scratch = 0;
        size_t const shortest_longest = sz_min_of_two(longest_query, longest_candidate);
        if (queries.size() && candidates.size() && shortest_longest > 64) {
            // The 2xN kernels (templated and runtime) keep a per-lane multi-word `match_masks` of `match_masks_bytes_k *
            // words_bound`, where the word bound is `ceil(shorter / 64)` for the largest shorter side any group can
            // present. This applies to every shorter side above 64, not just the > 512 long tail.
            size_t const words_bound = divide_round_up<size_t>(shortest_longest, 64);
            twoxN_scratch = myers_t::match_masks_bytes_k * words_bound;
        }
        if (queries.size() && candidates.size() && shortest_longest > 512) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(sz_max_of_two(myers_scratch, dp_scratch), twoxN_scratch);
    }

#pragma region Cross Product Cell Addressing

    /**
     *  @brief A destination for one scored cell: the primary matrix slot plus an optional mirror slot. The Myers
     *      kernels assign `writer[group_local_index] = distance`, so the writer holds one of these per active lane
     *      and fans the score out to both slots on assignment.
     */
    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    /**
     *  @brief An indexable adapter handed to the Myers kernels so they can stay grouping-agnostic: a lane's
     *      group-local index selects its destination, and assigning a score writes the primary cell and, for
     *      symmetric self-similarity, the mirrored cell too.
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
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the lockstep NEON Myers family,
     *      falling back to the anti-diagonal DP for the long tail. Each cell scores `dist(query, candidate)` and
     *      writes it into the strided @p results matrix (plus the mirror slot for symmetric self-similarity). Cells
     *      are grouped query-major by their exact `ceil(shorter / 64)` word bucket so each group loops one common
     *      word count and `distances_2xN_<words_count>` can be selected at compile time.
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
        dummy_executor_t dummy;
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const shorter = sz_min_of_two(query.size(), candidate.size());

            // Empty cell: the only alignment is all-gaps, so the distance is `max(query, candidate)` length.
            if (shorter == 0) {
                cross_cell_destination_<value_t> const destination = destination_for(query_index, candidate_index);
                cross_cell_writer_<value_t> {&destination}[0] = sz_max_of_two(query.size(), candidate.size());
                ++cell_index;
                continue;
            }

            // Batched Myers: gather up to `lanes_k` consecutive live cells whose shorter side shares this seed cell's
            // exact `ceil(shorter / 64)` word bucket, so the whole group loops one common `words_count`. Per-lane
            // `top_bits` still handle differing exact lengths inside the bucket. Seeding first guarantees progress.
            size_t const seed_bucket = divide_round_up<size_t>(shorter, 64);
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
                if (next_shorter == 0 || divide_round_up<size_t>(next_shorter, 64) != seed_bucket) break;
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
            // Bucket 1 (shorter <= 64) takes the single-word kernel, 2..8 the compile-time multiword variant; longer
            // groups take the runtime sibling.
            status_t status = dispatch_word_bucket_<1, 8>(
                seed_bucket,
                [&](auto bucket) {
                    if constexpr (bucket.value == 1) return myers.distances_2x64_(group_pairs, group_writer, scratch);
                    else
                        return myers.template distances_2x_multiword_<bucket.value>(group_pairs, group_writer, scratch);
                },
                [&] { return myers.distances_2x_multiword_large_(group_pairs, group_writer, scratch); });
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

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `u16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_u16_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `u32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_u32_(query_length, candidate_length);
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
        if (!is_unit_cost_()) {
            lane_walker_narrow_t narrow {substituter_, gap_costs_};
            lane_walker_wide_t wide {substituter_, gap_costs_};
            scoring_t fallback {substituter_, gap_costs_};
            auto const fits_wide = fits_wide_policy_();
            if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                    narrow, wide, fallback, queries, candidates, fits_wide, specs));
                status != status_t::success_k)
                return status;
            return cross_product_candidate_lanes_range_(
                narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
            auto const fits_wide = fits_wide_policy_();
            if (status_t status = score_scratch_.try_resize(cross_product_candidate_lanes_scratch_(
                    narrow, wide, fallback, sequences, sequences, fits_wide, specs));
                status != status_t::success_k)
                return status;
            return cross_product_candidate_lanes_range_(
                narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
 *  @brief Batched byte-level @b affine-gap Levenshtein distances on NEON, packing up to 8 candidates of a shared
 *      query into the unsigned-`u16` affine `candidate_lane_walker` and falling back to the per-pair anti-diagonal
 *      Dynamic Programming scorer for the long tail (distances that escape `u16`).
 *
 *  Affine sibling of the NEON linear byte `levenshtein_distances` cross-product engine: identical query-major
 *  grouping, transpose, scatter, and mirror handling via the shared candidate-lane driver. There is no Myers fast
 *  path - the bit-parallel `+-1`-delta recurrence is linear-only - so @b every cost routes through the affine lane
 *  walker, and an empty `(query, candidate)` cell scores the single-gap-run `open + extend * (L - 1)`.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<affine_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 8;          // ? `u16` lanes for the affine candidate-lane walker.
    static constexpr size_t u16_reach_limit_k = 50000;      // ? `u16` headroom below the lane walker's discard bias.
    static constexpr size_t u32_reach_limit_k = 1500000000; // ? `u32` headroom for the wide affine lane walker.

    using scoring_t = levenshtein_distance<char, affine_gap_costs_t, sz_cap_serial_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, affine_gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_neon_k, (int)candidate_lanes_k,
                              void>; // ? NEON affine shared-query `u16` lanes.
    using lane_walker_wide_t = candidate_lane_walker<char, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                                                     sz_minimize_distance_k, sz_similarity_global_k, sz_cap_neon_k, 4,
                                                     void>; // ? NEON affine shared-query `u32` lanes.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    uniform_substitution_costs_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    levenshtein_distances(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances(uniform_substitution_costs_t subs, affine_gap_costs_t gaps,
                          allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief Whether a `(query, candidate)` cell's worst-case distance stays inside the lane walker's `u16` headroom. */
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
               u32_reach_limit_k;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case distance fits the `u16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_u16_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case distance fits the `u32` walker. */
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
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, queries, candidates, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, sequences, sequences, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
    static constexpr ssize_t score_range_limit_k = 30000;           // ? `i16` headroom for the narrow lane walker.
    static constexpr ssize_t score_range_limit_wide_k = 2000000000; // ? `i32` headroom for the wide lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, sz_caps_sn_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_neon_k, (int)candidate_lanes_k, void>; // ? NEON shared-query `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_neon_k, 4, void>; // ? NEON shared-query `i32` lanes.

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
    bool fits_i16_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the wide walker's `i32` headroom. */
    bool fits_i32_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_wide_k;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `i16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i16_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `i32` walker. */
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
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, queries, candidates, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, sequences, sequences, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
    static constexpr ssize_t score_range_limit_k = 30000;           // ? `i16` headroom for the narrow lane walker.
    static constexpr ssize_t score_range_limit_wide_k = 2000000000; // ? `i32` headroom for the wide lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, sz_caps_sn_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_neon_k, (int)candidate_lanes_k, void>; // ? NEON shared-query local `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_neon_k, 4, void>; // ? NEON shared-query local `i32` lanes.

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
    bool fits_i16_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the wide walker's `i32` headroom. */
    bool fits_i32_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_wide_k;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `i16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i16_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `i32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i32_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: a local-alignment empty cell scores zero. */
    auto empty_cell_policy_() const noexcept {
        return [](size_t query_length, size_t candidate_length) noexcept -> ssize_t {
            sz_unused_(query_length), sz_unused_(candidate_length);
            return 0;
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, queries, candidates, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, sequences, sequences, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
 *  @brief Batched byte-level @b weighted Needleman-Wunsch scores on NEON with an @b affine gap, packing up to 8
 *      candidates of a shared query into the signed-`i16` affine `candidate_lane_walker` and falling back to the
 *      per-pair anti-diagonal Dynamic Programming scorer for the long tail (scores that escape `i16`, or empty cells).
 *
 *  Affine sibling of the NEON linear-gap `needleman_wunsch_scores` cross-product engine: identical query-major
 *  grouping, transpose, and scatter, but it dispatches the affine (Gotoh E/F) lane walker, and an empty
 *  `(query, candidate)` cell scores `open + extend * (other_length - 1)` rather than `gap * other_length`.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, allocator_type_, capability_,
                               std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 8;
    static constexpr ssize_t score_range_limit_k = 30000;           // ? `i16` headroom for the narrow lane walker.
    static constexpr ssize_t score_range_limit_wide_k = 2000000000; // ? `i32` headroom for the wide lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, sz_caps_sn_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_neon_k, (int)candidate_lanes_k, void>; // ? NEON shared-query `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_neon_k, 4, void>; // ? NEON shared-query `i32` lanes.

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
    bool fits_i16_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the wide walker's `i32` headroom. */
    bool fits_i32_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_wide_k;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `i16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i16_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `i32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i32_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: an affine-gap empty cell scores `open + extend*(len-1)`. */
    auto empty_cell_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept -> ssize_t {
            ssize_t const other_length = (ssize_t)sz_max_of_two(query_length, candidate_length);
            return other_length ? (ssize_t)gap_costs_.open + (ssize_t)gap_costs_.extend * (other_length - 1) : 0;
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, queries, candidates, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, sequences, sequences, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
 *  @brief Batched byte-level @b weighted Smith-Waterman (local) scores on NEON with an @b affine gap, packing up to 8
 *      candidates of a shared query into the signed-`i16` local affine `candidate_lane_walker` and falling back to the
 *      per-pair anti-diagonal Dynamic Programming scorer for the long tail (scores that escape `i16`).
 *
 *  Local sibling of the NEON affine `needleman_wunsch_scores` cross-product engine: identical query-major grouping,
 *  transpose, and scatter, but it dispatches the @b local affine lane walker (zero-clamped Gotoh recurrence, score =
 *  the maximum cell of the matrix) and an empty `(query, candidate)` cell scores @b 0 (a local alignment may align
 *  nothing).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 8;
    static constexpr ssize_t score_range_limit_k = 30000;           // ? `i16` headroom for the narrow lane walker.
    static constexpr ssize_t score_range_limit_wide_k = 2000000000; // ? `i32` headroom for the wide lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, sz_caps_sn_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_neon_k, (int)candidate_lanes_k, void>; // ? NEON shared-query local `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_neon_k, 4, void>; // ? NEON shared-query local `i32` lanes.

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
    bool fits_i16_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the wide walker's `i32` headroom. */
    bool fits_i32_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_wide_k;
    }

#pragma region Public Cross Product Overloads

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `i16` walker. */
    auto fits_narrow_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i16_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the `i32` walker. */
    auto fits_wide_policy_() const noexcept {
        return [this](size_t query_length, size_t candidate_length) noexcept {
            return fits_i32_range_(query_length, candidate_length);
        };
    }

    /** @brief `(query_length, candidate_length) -> score`: a local-alignment empty cell scores zero. */
    auto empty_cell_policy_() const noexcept {
        return [](size_t query_length, size_t candidate_length) noexcept -> ssize_t {
            sz_unused_(query_length), sz_unused_(candidate_length);
            return 0;
        };
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        lane_walker_narrow_t narrow {substituter_, gap_costs_};
        lane_walker_wide_t wide {substituter_, gap_costs_};
        scoring_t fallback {substituter_, gap_costs_};
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, queries, candidates, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
            cross_live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
        auto const fits_wide = fits_wide_policy_();
        if (status_t status = score_scratch_.try_resize(
                cross_product_candidate_lanes_scratch_(narrow, wide, fallback, sequences, sequences, fits_wide, specs));
            status != status_t::success_k)
            return status;
        return cross_product_candidate_lanes_range_(
            narrow, wide, fallback, sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
            cross_live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
            fits_narrow_policy_(), fits_wide, empty_cell_policy_(), scratch_space_t(score_scratch_), specs);
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
 *  @brief Batched UTF-8 @b Levenshtein distances on NEON. For unit costs, transcodes both operands to @b runes
 *      (UTF-32) and batches consecutive cells whose shorter rune side shares the same `ceil(shorter / 64)` word bucket
 *      into the 2-lane rune `levenshtein_distance_myers` (the rune twin of the byte Myers driver), exactly mirroring
 *      the Ice Lake rune-Myers UTF-8 engine. The per-pair anti-diagonal rune Dynamic Programming scorer is the long
 *      tail / fallback (empty cells, invalid UTF-8, shorter > 4096 runes), and the same per-pair scorer also handles
 *      every non-unit / affine case.
 *
 *  Mirrors the NEON byte-level `levenshtein_distances` cross-product engine - live cells walked query-major, batched
 *  into same-bucket groups, transcoded into a reusable rune arena, scored, and scattered into the strided result
 *  matrix (plus the mirror slot for symmetric self-similarity) - but the alphabet is 32-bit code points instead of
 *  bytes, so the per-lane `match_masks` is the rune-keyed open-addressing hash instead of a dense 256-row table. A byte length
 *  is an upper bound on the rune count, so tier gating keys on byte lengths (no transcode needed to decide).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances_utf8<linear_gap_costs_t, allocator_type_, capability_,
                                  std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 8; // ? 8 candidate runes per block while distances fit `u16`.

    using scoring_t = levenshtein_distance_utf8<gap_costs_t, sz_cap_serial_k>; // ? Per-pair UTF-8 serial fallback.
    using myers_t = levenshtein_distance_myers<rune_t, capability_k>;          // ? NEON 2-lane rune Myers fast path.
    using lane_walker_t =
        candidate_lane_walker<rune_t, u16_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_neon_k, (int)candidate_lanes_k, void>;
    // The non-unit cost path routes the transcoded rune views through the shared width-tiered candidate-lane driver:
    // an 8-lane `u16` narrow walker plus a 4-lane `u32` wide walker, with the per-pair rune DP as the long-tail
    // fallback. Unit-cost linear stays on the rune Myers fast path (`score_range_`), untouched.
    using lane_walker_narrow_t = lane_walker_t; // ? NEON 8-lane `u16` non-unit rune shared query.
    using lane_walker_wide_t =
        candidate_lane_walker<rune_t, u32_t, uniform_substitution_costs_t, gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_neon_k, 4, void>; // ? 4-lane `u32` non-unit rune.
    // The driver's per-pair fallback receives @b rune views, so it is a rune-typed `levenshtein_distance`; the serial
    // capability covers every cell width, and this long-tail path is rare. It stays bit-exact with the serial oracle.
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
     *      sequence's rune span in @p views (pointing into @p arena). Returns `false` on invalid UTF-8 (the caller
     *      then routes the whole call through the per-pair UTF-8 serial fallback) or on allocation failure.
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

#pragma region Cross Product Cell Addressing

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

#pragma region Cross Product Scratch Sizing

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the shared-query rune buffer plus
     *      the transposed rune block plus the rune lane-walker arena for the longest candidate, or the per-pair rune DP
     *      fallback for the longest query x longest candidate. Byte lengths upper-bound the rune counts, so they size
     *      every region without a transcode.
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
        size_t const query_runes_bytes = round_up_to_multiple(sizeof(rune_t) * longest_query, specs.cache_line_width);
        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(rune_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs)
                                                        : 0;
        size_t const lane_path = query_runes_bytes + transpose_bytes + walker_scratch;
        // The rune-Myers fast path transcodes up to `myers_lanes_k` pairs' runes into one arena (a rune is at least one
        // byte, so byte length bounds rune count) and builds the 2-lane hash `match_masks` for the longest shorter side.
        size_t const myers_transcode_bytes = round_up_to_multiple(
            (size_t)myers_lanes_k * (longest_query + longest_candidate) * sizeof(rune_t), specs.cache_line_width);
        size_t const myers_match_masks_bytes = myers_t::scratch_bytes_for(
            sz_min_of_two(longest_query, longest_candidate));
        size_t const myers_path = myers_transcode_bytes + myers_match_masks_bytes;
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size()) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(sz_max_of_two(lane_path, myers_path), dp_scratch);
    }

#pragma endregion Cross Product Scratch Sizing

#pragma region Cross Product Scoring

    /** @brief Transcodes a UTF-8 byte span into a rune buffer; returns the rune count, or `invalid_utf8_k` on error. */
    static status_t transcode_(span<char const> source, rune_t *destination, size_t &rune_count) noexcept {
        rune_length_t rune_length {};
        rune_count = 0;
        for (size_t progress = 0; progress < source.size(); progress += rune_length, ++rune_count) {
            rune_length = sz_rune_decode_unchecked(source.data() + progress, destination + rune_count);
            if (rune_length == sz_rune_invalid_k) return status_t::invalid_utf8_k;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the 2-lane rune Myers - the
     *      unit-cost-linear UTF-8 fast path. Each cell is transcoded to runes, and consecutive cells whose shorter rune
     *      side shares the same `ceil(shorter / 64)` word bucket are batched into one rune-Myers call (single word for
     *      shorter <= 64, the compile-time `distances_2x_multiword_<W>` for buckets 2..8, the runtime sibling beyond),
     *      exactly the Ice Lake rune-Myers driver's grouping at a quarter of the lane width. Empty cells, invalid UTF-8,
     *      and lone shorter > 4096 cells fall back to the per-pair rune scorer. Writes each distance into the strided
     *      @p results matrix (plus the mirror slot for symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    SZ_NOINLINE status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                      results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                      size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        // Carve the slice's worst transcode area off the front of scratch (a rune is >= 1 byte, so byte length bounds
        // rune count) and leave the tail for the per-call 2-lane Myers `match_masks`. The global-worst sizing from
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
        auto const transcode_cell = [&](span<char const> query, span<char const> candidate, size_t &arena_used,
                                        span<rune_t const> &shorter_runes,
                                        span<rune_t const> &longer_runes) noexcept -> bool {
            size_t const query_offset = arena_used;
            size_t query_runes_count = 0;
            rune_length_t rune_length {};
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
            64; // ? `distances_2x_multiword_large_` covers shorter <= 4096 runes.
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
                        return myers.distances_2x64_(group_pairs, group_writer, match_masks_scratch);
                    else
                        return myers.template distances_2x_multiword_<bucket.value>(group_pairs, group_writer,
                                                                                    match_masks_scratch);
                },
                [&] { return myers.distances_2x_multiword_large_(group_pairs, group_writer, match_masks_scratch); });
            if (status == status_t::success_k) continue;
            // Defensive scratch-shortfall fallback: score every grouped pair through the per-pair UTF-8 rune DP over
            // its original cell (the `dp` scorer owns its own scratch via `dp_scratch_space`). Mirrors the Ice Lake
            // rune-Myers driver's group fallback; `worst_cell_scratch_` sizes the `match_masks` so this should not trigger.
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
 *  @brief Batched @b rune-level @b affine-gap Levenshtein distances on NEON, transcoding UTF-8 to UTF-32 once and
 *      packing up to 8 candidates of a shared query into the unsigned-`u16` affine rune `candidate_lane_walker` (or
 *      the 4-lane `u32` sibling), falling back to the per-pair rune anti-diagonal Dynamic Programming scorer for the
 *      long tail (rune lengths that escape the `u32` cell range, or candidates that cannot fill a lane block).
 *
 *  Affine sibling of the NEON linear UTF-8 `levenshtein_distances_utf8` cross-product engine. There is no Myers
 *  fast path - the bit-parallel `+-1`-delta recurrence is linear-only - so @b every cost routes through the affine
 *  rune lane driver, and an empty `(query, candidate)` cell scores the single-gap-run `open + extend * (L - 1)`. On
 *  invalid UTF-8 (or transcode alloc failure) the whole call routes through the per-pair UTF-8 serial scorer, which
 *  scores invalid bytes directly.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances_utf8<affine_gap_costs_t, allocator_type_, capability_,
                                  std::enable_if_t<(capability_ & sz_cap_neon_k) != 0>> {

    using char_t = char;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 8;     // ? `u16` lanes for the affine rune candidate-lane walker.
    static constexpr size_t u16_reach_limit_k = 50000; // ? `u16` headroom below the affine walker's discard bias.

    using scoring_t = levenshtein_distance_utf8<gap_costs_t, sz_cap_serial_k>; // ? Per-pair UTF-8 serial fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<rune_t, u16_t, uniform_substitution_costs_t, affine_gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_neon_k, (int)candidate_lanes_k,
                              void>; // ? NEON 8-lane `u16` affine rune shared query.
    using lane_walker_wide_t =
        candidate_lane_walker<rune_t, u32_t, uniform_substitution_costs_t, affine_gap_costs_t, sz_minimize_distance_k,
                              sz_similarity_global_k, sz_cap_neon_k, 4, void>;         // ? 4-lane `u32` affine rune.
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
