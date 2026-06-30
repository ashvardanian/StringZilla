/**
 *  @brief RISC-V Vector (RVV 1.0) string-similarity backend.
 *  @file include/stringzillas/similarities/rvv.hpp
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 *  @sa include/stringzillas/similarities/neon.hpp
 *
 *  @note Wave 1 foundation. Every engine in this header redirects to the scalar `sz_cap_serial_k`
 *        implementations, so `sz_caps_sr_k` (serial + RVV) is a fully dispatchable capability that runs
 *        at serial speed. The empty region markers below mark where the real vector-length-agnostic
 *        RVV kernels will land, region by region, in later waves - the build stays green throughout.
 */
#ifndef STRINGZILLAS_SIMILARITIES_RVV_HPP_
#define STRINGZILLAS_SIMILARITIES_RVV_HPP_

#include "stringzillas/similarities/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

/*  RVV implementation of the string similarity algorithms for 64-bit RISC-V CPUs.
 *  Until the vector-length-agnostic kernels land, the whole backend forwards to the scalar serial
 *  oracle: the per-pair Myers and the tiny-input Wagner-Fischer horizontal walker inherit the serial
 *  versions, and the diagonal `tile_scorer` cells redirect to the serial cell recurrence. The generic
 *  serial diagonal/horizontal walkers and the generic serial batch engines then drive everything, so
 *  `sz_caps_sr_k` is bit-exact with `sz_cap_serial_k`.
 */
#pragma region RVV Implementation
#if SZ_USE_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#pragma region Bit Parallel Myers

/**
 *  @brief RVV Myers/Hyyr\xc3\xb6 unit-cost Levenshtein for 64-bit RISC-V, scoring up to `VLEN/64 * LMUL` independent
 *      pairs at once with one register-resident 64-bit Myers integer per vector lane - the vector-length-agnostic
 *      generalization of the NEON two-lane and Ice Lake eight-lane byte Myers. Every lane is its own Myers integer;
 *      no carry, shift, or boundary mask ever crosses a lane, so the lane count is purely a runtime `vsetvl` quantity:
 *      - `distances_Nx64_`            - single-word Myers (shorter <= 64), one 64-bit integer per lane at LMUL m8;
 *      - `distances_Nx_multiword_<words_count_>` - multi-word Myers with a compile-time word count, covering shorter
 *        in `(64, 64 * words_count_]`, at LMUL m1 to bound register pressure from the per-word state;
 *      - `distances_Nx_multiword_large_` - the runtime-`words_count` sibling for the long tail (shorter > 512).
 *      The single-word kernel gathers each lane's `Eq` row with one `vluxei64` over the `match_masks` table (indexed
 *      `lane * 256 + symbol`); the multi-word kernels ripple the 65-bit `(Eq & VP) + VP + carry` add with the native
 *      `vadc` / `vmadc` carry instructions and shift bit63->bit0 across words with `vsrl ...,63` + `vsll ...,1` + `vor`.
 *      Variable per-lane lengths are frozen with a `vbool` active mask (`vmsgtu` of the longer length vs the position)
 *      that drives `vmerge` on the state and masked `vadd` / `vsub` on the per-lane score. Because RVV vector types are
 *      sizeless (no arrays), the multi-word per-word state lives in stack `u64` rows reloaded per word, and a group of
 *      up to `lanes_k` pairs is consumed in `vsetvl`-sized chunks so the same kernel is correct at every `VLEN`.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<char, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = char;
    using index_t = u32_t;
    // ? Compile-time grouping cap handed to the cross-product driver; the kernels consume a group in `vsetvl` chunks,
    // ? so this only bounds the stack arrays. 64 fully fills an m8 vector at `VLEN == 512` (the largest tested width).
    static constexpr index_t lanes_k = 64;
    static constexpr size_t match_masks_bytes_k = sizeof(u64_t) * lanes_k * 256; // ? 128 KB single-word `match_masks`.

    levenshtein_distance_myers() noexcept {}

    /** @brief Single-pair scratch sizing for the per-pair `levenshtein_distance` engine - delegated to the scalar
     *      serial Myers (one pair gains nothing from RVV batching; the lane kernels serve the cross-product). */
    auto layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        return levenshtein_distance_myers<char, sz_cap_serial_k> {}.layout(first, second, specs);
    }

    /** @brief Single-pair Myers (per-pair engine path) - delegated to the scalar serial Myers. */
    status_t operator()(span<char_t const> const &first, span<char_t const> const &second, size_t &result_ref,
                        scratch_space_t scratch_space) noexcept {
        return levenshtein_distance_myers<char, sz_cap_serial_k> {}(first, second, result_ref, scratch_space);
    }

    /**
     *  @brief Up to `lanes_k` independent single-word Myers distances, one per vector lane (each shorter side <= 64).
     *      @p scratch_space holds the `match_masks[lanes_k][256]` table (`match_masks_bytes_k`). The lockstep scan is
     *      the byte 64-bit Myers recurrence; the per-lane `Eq` row is gathered with one `vluxei64`.
     */
    template <typename results_writer_>
    status_t distances_Nx64_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                             scratch_space_t scratch_space) const noexcept {

        if (scratch_space.size() < match_masks_bytes_k) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data()); // ? Indexed `lane * 256 + symbol`.
        index_t const lanes = (index_t)pairs.lanes_count();
        u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        u64_t vertical_positive_init[lanes_k] = {0}, final_scores[lanes_k] = {0};

        for (index_t lane_index = 0; lane_index != lanes; ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            size_t const longer_length = pairs.longers[lane_index].size();
            char_t const *const shorter = pairs.shorters[lane_index].data();
            char_t const *const longer = pairs.longers[lane_index].data();
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[(size_t)lane_index * 256 + (u8_t)shorter[position]] = 0;
            for (size_t position = 0; position != longer_length; ++position)
                match_masks[(size_t)lane_index * 256 + (u8_t)longer[position]] = 0;
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[(size_t)lane_index * 256 + (u8_t)shorter[position]] |= (u64_t)1 << position;
            top_bits[lane_index] = (u64_t)1 << (shorter_length - 1);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = longer_length;
            vertical_positive_init[lane_index] = shorter_length == 64 ? ~(u64_t)0 : (((u64_t)1 << shorter_length) - 1);
        }

        // Consume the group in `vsetvl`-sized chunks; each chunk is an independent set of lanes scored end-to-end.
        for (index_t lane_base = 0; lane_base < lanes;) {
            size_t const vl = __riscv_vsetvl_e64m8(lanes - lane_base);
            size_t chunk_max_longer = 0;
            for (size_t i = 0; i != vl; ++i)
                chunk_max_longer = sz_max_of_two(chunk_max_longer, longer_lengths[lane_base + i]);

            vuint64m8_t const top_mask_u64m8 = __riscv_vle64_v_u64m8(top_bits + lane_base, vl);
            vuint64m8_t const longer_u64m8 = __riscv_vle64_v_u64m8(longer_lengths + lane_base, vl);
            vuint64m8_t const lane_global_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vl), (u64_t)lane_base, vl);
            vuint64m8_t const base_index_u64m8 = __riscv_vmul_vx_u64m8(lane_global_u64m8, 256, vl);
            vuint64m8_t vertical_positive_u64m8 = __riscv_vle64_v_u64m8(vertical_positive_init + lane_base, vl);
            vuint64m8_t vertical_negative_u64m8 = __riscv_vmv_v_x_u64m8(0, vl);
            vuint64m8_t score_u64m8 = __riscv_vle64_v_u64m8(shorter_lengths + lane_base, vl);

            for (size_t position = 0; position != chunk_max_longer; ++position) {
                vbool8_t const active_b8 = __riscv_vmsgtu_vx_u64m8_b8(longer_u64m8, (u64_t)position, vl);
                // Stage the current symbol per lane (0 for finished lanes, which `active_b8` freezes anyway), then
                // gather each lane's `Eq` bitmask row with a single indexed load over `match_masks`.
                u8_t symbols[lanes_k] = {0};
                for (size_t i = 0; i != vl; ++i)
                    if (position < longer_lengths[lane_base + i])
                        symbols[i] = (u8_t)pairs.longers[lane_base + i].data()[position];
                vuint64m8_t const symbol_u64m8 = __riscv_vzext_vf8_u64m8(__riscv_vle8_v_u8m1(symbols, vl), vl);
                vuint64m8_t const byte_offset_u64m8 = __riscv_vsll_vx_u64m8(
                    __riscv_vadd_vv_u64m8(base_index_u64m8, symbol_u64m8, vl), 3, vl);
                vuint64m8_t const equality_u64m8 = __riscv_vluxei64_v_u64m8(match_masks, byte_offset_u64m8, vl);

                vuint64m8_t const carry_in_u64m8 = __riscv_vor_vv_u64m8(equality_u64m8, vertical_negative_u64m8, vl);
                // Xh = (((Eq & VP) + VP) ^ VP) | Eq.
                vuint64m8_t const sum_u64m8 = __riscv_vadd_vv_u64m8(
                    __riscv_vand_vv_u64m8(equality_u64m8, vertical_positive_u64m8, vl), vertical_positive_u64m8, vl);
                vuint64m8_t const diagonal_u64m8 = __riscv_vor_vv_u64m8(
                    __riscv_vxor_vv_u64m8(sum_u64m8, vertical_positive_u64m8, vl), equality_u64m8, vl);
                // Ph = VN | ~(D | VP); Mh = VP & D.
                vuint64m8_t horizontal_positive_u64m8 = __riscv_vor_vv_u64m8(
                    vertical_negative_u64m8,
                    __riscv_vnot_v_u64m8(__riscv_vor_vv_u64m8(diagonal_u64m8, vertical_positive_u64m8, vl), vl), vl);
                vuint64m8_t horizontal_negative_u64m8 = __riscv_vand_vv_u64m8(vertical_positive_u64m8, diagonal_u64m8,
                                                                              vl);

                // score += active & ((Ph & top) != 0); score -= active & ((Mh & top) != 0).
                vbool8_t const add_b8 = __riscv_vmand_mm_b8(
                    active_b8,
                    __riscv_vmsne_vx_u64m8_b8(__riscv_vand_vv_u64m8(horizontal_positive_u64m8, top_mask_u64m8, vl), 0,
                                              vl),
                    vl);
                vbool8_t const sub_b8 = __riscv_vmand_mm_b8(
                    active_b8,
                    __riscv_vmsne_vx_u64m8_b8(__riscv_vand_vv_u64m8(horizontal_negative_u64m8, top_mask_u64m8, vl), 0,
                                              vl),
                    vl);
                score_u64m8 = __riscv_vadd_vx_u64m8_mu(add_b8, score_u64m8, score_u64m8, 1, vl);
                score_u64m8 = __riscv_vsub_vx_u64m8_mu(sub_b8, score_u64m8, score_u64m8, 1, vl);

                horizontal_positive_u64m8 = __riscv_vor_vx_u64m8(
                    __riscv_vsll_vx_u64m8(horizontal_positive_u64m8, 1, vl), 1, vl);
                horizontal_negative_u64m8 = __riscv_vsll_vx_u64m8(horizontal_negative_u64m8, 1, vl);
                // Pv' = Mh | ~(Xv | Ph) = Hn | ~(carry_in | Hp); applied only to active lanes.
                vuint64m8_t const next_positive_u64m8 = __riscv_vor_vv_u64m8(
                    horizontal_negative_u64m8,
                    __riscv_vnot_v_u64m8(__riscv_vor_vv_u64m8(carry_in_u64m8, horizontal_positive_u64m8, vl), vl), vl);
                vuint64m8_t const next_negative_u64m8 = __riscv_vand_vv_u64m8(horizontal_positive_u64m8, carry_in_u64m8,
                                                                              vl);
                vertical_positive_u64m8 = __riscv_vmerge_vvm_u64m8(vertical_positive_u64m8, next_positive_u64m8,
                                                                   active_b8, vl);
                vertical_negative_u64m8 = __riscv_vmerge_vvm_u64m8(vertical_negative_u64m8, next_negative_u64m8,
                                                                   active_b8, vl);
            }

            __riscv_vse64_v_u64m8(final_scores + lane_base, score_u64m8, vl);
            lane_base += (index_t)vl;
        }

        for (index_t lane_index = 0; lane_index != lanes; ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }

    /**
     *  @brief Up to `lanes_k` independent compile-time multi-word Myers distances, one pair per vector lane, covering
     *      shorter sides in `(64, 64 * words_count_]`. Each lane is one `words_count_`-word Myers integer; the per-word
     *      `vertical_positive` / `vertical_negative` state lives in stack `u64` rows (RVV vector types are sizeless, so
     *      they cannot be kept in a register array as on NEON) reloaded per word. Two carries cross words within a lane
     *      (never across lanes): the 65-bit `(Eq & VP) + VP + carry` ripple, tracked with the native `vadc` / `vmadc`,
     *      and the `horizontal_positive` / `horizontal_negative` bit63->bit0 shift. @p scratch_space holds the per-lane
     *      multi-word `match_masks` (`match_masks_bytes_k * words_count_`, entry `[lane * 256 * W + symbol * W + word]`).
     */
    template <size_t words_count_, typename results_writer_>
    status_t distances_Nx_multiword_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                     scratch_space_t scratch_space) const noexcept {
        return multiword_scan_(pairs, results, scratch_space, words_count_);
    }

    /**
     *  @brief The runtime-`words_count` sibling of `distances_Nx_multiword_<words_count_>` for the long tail (shorter >
     *      512); every lane in the group loops the same `ceil(max_shorter / 64)` word range. The math is identical.
     */
    template <typename results_writer_>
    status_t distances_Nx_multiword_large_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                           scratch_space_t scratch_space) const noexcept {
        size_t max_shorter = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            max_shorter = sz_max_of_two(max_shorter, pairs.shorters[lane_index].size());
        size_t const words_count = divide_round_up<size_t>(max_shorter, 64);
        if (words_count == 0 || words_count > stack_words_capacity_k) return status_t::bad_alloc_k;
        return multiword_scan_(pairs, results, scratch_space, words_count);
    }

  private:
    static constexpr size_t stack_words_capacity_k = 64; // ? Covers shorter sides up to 4096 on the stack.

    /** @brief Shared multi-word byte Myers scan (runtime @p words_count) backing both the compile-time and the large
     *      multi-word entry points; the per-word state lives in `vertical_*_state[word * lanes_k + lane]` rows. */
    template <typename results_writer_>
    status_t multiword_scan_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                             scratch_space_t scratch_space, size_t words_count) const noexcept {

        size_t const match_masks_words = (size_t)256 * words_count * lanes_k;
        if (scratch_space.size() < match_masks_words * sizeof(u64_t)) return status_t::bad_alloc_k;
        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data());
        for (size_t element = 0; element != match_masks_words; ++element) match_masks[element] = 0;

        index_t const lanes = (index_t)pairs.lanes_count();
        u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        u64_t final_scores[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != lanes; ++lane_index) {
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

        size_t const last_word = words_count - 1;
        // Per-word state rows, indexed `[word * lanes_k + local_lane]`; reset per chunk (chunks are independent).
        u64_t vertical_positive_state[lanes_k * stack_words_capacity_k];
        u64_t vertical_negative_state[lanes_k * stack_words_capacity_k];

        for (index_t lane_base = 0; lane_base < lanes;) {
            size_t const vl = __riscv_vsetvl_e64m1(lanes - lane_base);
            size_t chunk_max_longer = 0;
            for (size_t i = 0; i != vl; ++i)
                chunk_max_longer = sz_max_of_two(chunk_max_longer, longer_lengths[lane_base + i]);
            for (size_t word = 0; word != words_count; ++word) {
                __riscv_vse64_v_u64m1(vertical_positive_state + word * lanes_k, __riscv_vmv_v_x_u64m1(~(u64_t)0, vl),
                                      vl);
                __riscv_vse64_v_u64m1(vertical_negative_state + word * lanes_k, __riscv_vmv_v_x_u64m1(0, vl), vl);
            }

            vuint64m1_t const top_mask_u64m1 = __riscv_vle64_v_u64m1(top_bits + lane_base, vl);
            vuint64m1_t const longer_u64m1 = __riscv_vle64_v_u64m1(longer_lengths + lane_base, vl);
            vuint64m1_t const lane_global_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vl), (u64_t)lane_base, vl);
            // Each lane's table starts at `lane * 256 * words_count`; the current symbol's row is `+ symbol * words`.
            vuint64m1_t const lane_index_scaled_u64m1 = __riscv_vmul_vx_u64m1(lane_global_u64m1, 256, vl);
            vuint64m1_t score_u64m1 = __riscv_vle64_v_u64m1(shorter_lengths + lane_base, vl);

            for (size_t position = 0; position != chunk_max_longer; ++position) {
                vbool64_t const active_b64 = __riscv_vmsgtu_vx_u64m1_b64(longer_u64m1, (u64_t)position, vl);
                u8_t symbols[lanes_k] = {0};
                for (size_t i = 0; i != vl; ++i)
                    if (position < longer_lengths[lane_base + i])
                        symbols[i] = (u8_t)pairs.longers[lane_base + i].data()[position];
                vuint64m1_t const symbol_u64m1 = __riscv_vzext_vf8_u64m1(__riscv_vle8_v_u8mf8(symbols, vl), vl);
                // Row base in `u64` units: (lane * 256 + symbol) * words_count.
                vuint64m1_t const row_base_u64m1 = __riscv_vmul_vx_u64m1(
                    __riscv_vadd_vv_u64m1(lane_index_scaled_u64m1, symbol_u64m1, vl), words_count, vl);

                vbool64_t addition_carry_b64 = __riscv_vmclr_m_b64(vl); // ? 65-bit `(Eq & VP) + VP` ripple carry.
                vuint64m1_t horizontal_positive_carry_u64m1 = __riscv_vmv_v_x_u64m1(1, vl); // ? Word 0 seeds the 1.
                vuint64m1_t horizontal_negative_carry_u64m1 = __riscv_vmv_v_x_u64m1(0, vl);

                for (size_t word = 0; word != words_count; ++word) {
                    vuint64m1_t const byte_offset_u64m1 = __riscv_vsll_vx_u64m1(
                        __riscv_vadd_vx_u64m1(row_base_u64m1, (u64_t)word, vl), 3, vl);
                    vuint64m1_t const equality_u64m1 = __riscv_vluxei64_v_u64m1(match_masks, byte_offset_u64m1, vl);
                    vuint64m1_t const vertical_positive_u64m1 = __riscv_vle64_v_u64m1(
                        vertical_positive_state + word * lanes_k, vl);
                    vuint64m1_t const vertical_negative_u64m1 = __riscv_vle64_v_u64m1(
                        vertical_negative_state + word * lanes_k, vl);

                    // sum = (Eq & VP) + VP + addition_carry, with the native add-with-carry and its carry-out.
                    vuint64m1_t const summand_u64m1 = __riscv_vand_vv_u64m1(equality_u64m1, vertical_positive_u64m1,
                                                                            vl);
                    vuint64m1_t const sum_u64m1 = __riscv_vadc_vvm_u64m1(summand_u64m1, vertical_positive_u64m1,
                                                                         addition_carry_b64, vl);
                    addition_carry_b64 = __riscv_vmadc_vvm_u64m1_b64(summand_u64m1, vertical_positive_u64m1,
                                                                     addition_carry_b64, vl);

                    vuint64m1_t const carry_in_u64m1 = __riscv_vor_vv_u64m1(equality_u64m1, vertical_negative_u64m1,
                                                                            vl); // ? Eq | VN
                    // Xh = (sum ^ VP) | Eq | VN == (sum ^ VP) | carry_in.
                    vuint64m1_t const diagonal_u64m1 = __riscv_vor_vv_u64m1(
                        __riscv_vxor_vv_u64m1(sum_u64m1, vertical_positive_u64m1, vl), carry_in_u64m1, vl);
                    vuint64m1_t horizontal_positive_u64m1 = __riscv_vor_vv_u64m1(
                        vertical_negative_u64m1,
                        __riscv_vnot_v_u64m1(__riscv_vor_vv_u64m1(diagonal_u64m1, vertical_positive_u64m1, vl), vl),
                        vl);
                    vuint64m1_t horizontal_negative_u64m1 = __riscv_vand_vv_u64m1(vertical_positive_u64m1,
                                                                                  diagonal_u64m1, vl);

                    if (word == last_word) {
                        vbool64_t const add_b64 = __riscv_vmand_mm_b64(
                            active_b64,
                            __riscv_vmsne_vx_u64m1_b64(
                                __riscv_vand_vv_u64m1(horizontal_positive_u64m1, top_mask_u64m1, vl), 0, vl),
                            vl);
                        vbool64_t const sub_b64 = __riscv_vmand_mm_b64(
                            active_b64,
                            __riscv_vmsne_vx_u64m1_b64(
                                __riscv_vand_vv_u64m1(horizontal_negative_u64m1, top_mask_u64m1, vl), 0, vl),
                            vl);
                        score_u64m1 = __riscv_vadd_vx_u64m1_mu(add_b64, score_u64m1, score_u64m1, 1, vl);
                        score_u64m1 = __riscv_vsub_vx_u64m1_mu(sub_b64, score_u64m1, score_u64m1, 1, vl);
                    }

                    // Save each word's bit63 as the next word's bit0 carry, then shift up by one.
                    vuint64m1_t const next_positive_carry_u64m1 = __riscv_vsrl_vx_u64m1(horizontal_positive_u64m1, 63,
                                                                                        vl);
                    vuint64m1_t const next_negative_carry_u64m1 = __riscv_vsrl_vx_u64m1(horizontal_negative_u64m1, 63,
                                                                                        vl);
                    horizontal_positive_u64m1 = __riscv_vor_vv_u64m1(
                        __riscv_vsll_vx_u64m1(horizontal_positive_u64m1, 1, vl), horizontal_positive_carry_u64m1, vl);
                    horizontal_negative_u64m1 = __riscv_vor_vv_u64m1(
                        __riscv_vsll_vx_u64m1(horizontal_negative_u64m1, 1, vl), horizontal_negative_carry_u64m1, vl);
                    horizontal_positive_carry_u64m1 = next_positive_carry_u64m1;
                    horizontal_negative_carry_u64m1 = next_negative_carry_u64m1;

                    // Pv' = Hn | ~(carry_in | Hp); applied only to active lanes.
                    vuint64m1_t const next_positive_u64m1 = __riscv_vor_vv_u64m1(
                        horizontal_negative_u64m1,
                        __riscv_vnot_v_u64m1(__riscv_vor_vv_u64m1(carry_in_u64m1, horizontal_positive_u64m1, vl), vl),
                        vl);
                    vuint64m1_t const next_negative_u64m1 = __riscv_vand_vv_u64m1(horizontal_positive_u64m1,
                                                                                  carry_in_u64m1, vl);
                    __riscv_vse64_v_u64m1(
                        vertical_positive_state + word * lanes_k,
                        __riscv_vmerge_vvm_u64m1(vertical_positive_u64m1, next_positive_u64m1, active_b64, vl), vl);
                    __riscv_vse64_v_u64m1(
                        vertical_negative_state + word * lanes_k,
                        __riscv_vmerge_vvm_u64m1(vertical_negative_u64m1, next_negative_u64m1, active_b64, vl), vl);
                }
            }

            __riscv_vse64_v_u64m1(final_scores + lane_base, score_u64m1, vl);
            lane_base += (index_t)vl;
        }

        for (index_t lane_index = 0; lane_index != lanes; ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }
};

/**
 *  @brief RVV Myers/Hyyr\xc3\xb6 unit-cost @b rune (UTF-32) Levenshtein for 64-bit RISC-V - the rune twin of the byte
 *      `levenshtein_distance_myers`, scoring up to `VLEN/64 * LMUL` independent pairs per vector with one 64-bit Myers
 *      integer per lane. The scan is bit-for-bit identical to the byte family; only the `Eq` source differs. A rune
 *      pattern cannot index a dense 256-row table, so each lane builds a small @b open-addressing hash over its own
 *      pattern's distinct runes and per text position scalar-probes its hash to assemble that lane's `Eq` word(s),
 *      which are then staged into one vector (a text rune absent from a lane's pattern hits the lane's all-zero
 *      `absent_row`). The hash math is reused verbatim from the serial rune oracle so all three stay bit-exact.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<rune_t, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = rune_t;
    using index_t = u32_t;
    static constexpr index_t lanes_k = 64;
    static constexpr rune_t empty_slot_k = static_cast<rune_t>(0xFFFFFFFFu);

    levenshtein_distance_myers() noexcept {}

    static size_t words_count_for(size_t shorter_length) noexcept {
        return divide_round_up<size_t>(shorter_length, 64);
    }

    static index_t hash_capacity_for(size_t distinct_upper_bound) noexcept {
        size_t const slots_wanted = sz_max_of_two(2 * distinct_upper_bound, (size_t)1);
        return static_cast<index_t>(sz_size_bit_ceil(slots_wanted));
    }

    static index_t hash_rune(rune_t rune, index_t capacity) noexcept {
        u64_t const mixed = static_cast<u64_t>(static_cast<u32_t>(rune)) * 0x9E3779B97F4A7C15ull;
        return static_cast<index_t>((mixed >> 32) & static_cast<u64_t>(capacity - 1));
    }

    static size_t scratch_bytes_for(size_t max_shorter) noexcept {
        size_t const words_count = words_count_for(sz_max_of_two(max_shorter, (size_t)1));
        size_t const capacity = hash_capacity_for(sz_max_of_two(max_shorter, (size_t)1));
        size_t const slot_keys_bytes = sizeof(rune_t) * capacity * lanes_k;
        size_t const slot_masks_bytes = sizeof(u64_t) * capacity * words_count * lanes_k;
        size_t const absent_row_bytes = sizeof(u64_t) * words_count;
        return slot_keys_bytes + slot_masks_bytes + absent_row_bytes;
    }

    /** @brief Single-pair scratch sizing for the per-pair `levenshtein_distance` engine - delegated to the serial Myers. */
    auto layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        return levenshtein_distance_myers<rune_t, sz_cap_serial_k> {}.layout(first, second, specs);
    }

    /** @brief Single-pair rune Myers (per-pair engine path) - delegated to the serial Myers. */
    status_t operator()(span<char_t const> const &first, span<char_t const> const &second, size_t &result_ref,
                        scratch_space_t scratch_space) noexcept {
        return levenshtein_distance_myers<rune_t, sz_cap_serial_k> {}(first, second, result_ref, scratch_space);
    }

    /** @brief Carves and builds the per-lane hash `match_masks` out of @p scratch_space (mirrors the serial oracle). */
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

    /** @brief Probes lane @p lane's hash for text rune @p symbol, returning its `words_count` bitmask row or `absent_row`. */
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

    /** @brief Up to `lanes_k` independent single-word rune Myers distances (each shorter side <= 64 runes). */
    template <typename results_writer_>
    status_t distances_Nx64_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                             scratch_space_t scratch_space) const noexcept {
        return scan_(pairs, results, scratch_space, 1);
    }

    template <size_t words_count_, typename results_writer_>
    status_t distances_Nx_multiword_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                     scratch_space_t scratch_space) const noexcept {
        return scan_(pairs, results, scratch_space, words_count_);
    }

    template <typename results_writer_>
    status_t distances_Nx_multiword_large_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                                           scratch_space_t scratch_space) const noexcept {
        size_t max_shorter = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            max_shorter = sz_max_of_two(max_shorter, pairs.shorters[lane_index].size());
        size_t const words_count = words_count_for(sz_max_of_two(max_shorter, (size_t)1));
        if (words_count == 0 || words_count > stack_words_capacity_k) return status_t::bad_alloc_k;
        return scan_(pairs, results, scratch_space, words_count);
    }

  private:
    static constexpr size_t stack_words_capacity_k = 64; // ? Covers shorter sides up to 4096 runes on the stack.

    /** @brief Shared rune Myers scan (runtime @p words_count); the per-word state lives in stack `u64` rows. The `Eq`
     *      words are staged scalar per lane from the per-lane hash and loaded into one vector. */
    template <typename results_writer_>
    status_t scan_(lane_pairs_view<char_t> const &pairs, results_writer_ &results, scratch_space_t scratch_space,
                   size_t words_count) const noexcept {

        size_t max_shorter = 0;
        for (index_t lane_index = 0; lane_index != pairs.lanes_count(); ++lane_index)
            max_shorter = sz_max_of_two(max_shorter, pairs.shorters[lane_index].size());
        index_t const capacity = hash_capacity_for(sz_max_of_two(max_shorter, (size_t)1));
        rune_t *slot_keys = nullptr;
        u64_t *slot_masks = nullptr, *absent_row = nullptr;
        if (!build_lane_hashes_(pairs.shorters.data(), (index_t)pairs.lanes_count(), capacity, words_count,
                                scratch_space, slot_keys, slot_masks, absent_row))
            return status_t::bad_alloc_k;

        index_t const lanes = (index_t)pairs.lanes_count();
        u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        u64_t final_scores[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != lanes; ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            top_bits[lane_index] = (u64_t)1 << ((shorter_length - 1) & 63);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = pairs.longers[lane_index].size();
        }

        size_t const last_word = words_count - 1;
        u64_t vertical_positive_state[lanes_k * stack_words_capacity_k];
        u64_t vertical_negative_state[lanes_k * stack_words_capacity_k];

        for (index_t lane_base = 0; lane_base < lanes;) {
            size_t const vl = __riscv_vsetvl_e64m1(lanes - lane_base);
            size_t chunk_max_longer = 0;
            for (size_t i = 0; i != vl; ++i)
                chunk_max_longer = sz_max_of_two(chunk_max_longer, longer_lengths[lane_base + i]);
            for (size_t word = 0; word != words_count; ++word) {
                __riscv_vse64_v_u64m1(vertical_positive_state + word * lanes_k, __riscv_vmv_v_x_u64m1(~(u64_t)0, vl),
                                      vl);
                __riscv_vse64_v_u64m1(vertical_negative_state + word * lanes_k, __riscv_vmv_v_x_u64m1(0, vl), vl);
            }

            vuint64m1_t const top_mask_u64m1 = __riscv_vle64_v_u64m1(top_bits + lane_base, vl);
            vuint64m1_t const longer_u64m1 = __riscv_vle64_v_u64m1(longer_lengths + lane_base, vl);
            vuint64m1_t score_u64m1 = __riscv_vle64_v_u64m1(shorter_lengths + lane_base, vl);

            for (size_t position = 0; position != chunk_max_longer; ++position) {
                vbool64_t const active_b64 = __riscv_vmsgtu_vx_u64m1_b64(longer_u64m1, (u64_t)position, vl);
                // Per-lane hash probe of the current text rune -> the lane's `words_count` bitmask row (or `absent_row`).
                u64_t const *match_rows[lanes_k];
                for (size_t i = 0; i != vl; ++i) {
                    bool const lane_active = position < longer_lengths[lane_base + i];
                    rune_t const symbol = lane_active ? pairs.longers[lane_base + i].data()[position] : empty_slot_k;
                    match_rows[i] = lane_active ? lane_match_row_(slot_keys, slot_masks, absent_row, lane_base + i,
                                                                  capacity, words_count, symbol)
                                                : absent_row;
                }

                vbool64_t addition_carry_b64 = __riscv_vmclr_m_b64(vl);
                vuint64m1_t horizontal_positive_carry_u64m1 = __riscv_vmv_v_x_u64m1(1, vl);
                vuint64m1_t horizontal_negative_carry_u64m1 = __riscv_vmv_v_x_u64m1(0, vl);

                for (size_t word = 0; word != words_count; ++word) {
                    u64_t equality_words[lanes_k] = {0};
                    for (size_t i = 0; i != vl; ++i)
                        equality_words[i] = (position < longer_lengths[lane_base + i]) ? match_rows[i][word] : 0;
                    vuint64m1_t const equality_u64m1 = __riscv_vle64_v_u64m1(equality_words, vl);
                    vuint64m1_t const vertical_positive_u64m1 = __riscv_vle64_v_u64m1(
                        vertical_positive_state + word * lanes_k, vl);
                    vuint64m1_t const vertical_negative_u64m1 = __riscv_vle64_v_u64m1(
                        vertical_negative_state + word * lanes_k, vl);

                    vuint64m1_t const summand_u64m1 = __riscv_vand_vv_u64m1(equality_u64m1, vertical_positive_u64m1,
                                                                            vl);
                    vuint64m1_t const sum_u64m1 = __riscv_vadc_vvm_u64m1(summand_u64m1, vertical_positive_u64m1,
                                                                         addition_carry_b64, vl);
                    addition_carry_b64 = __riscv_vmadc_vvm_u64m1_b64(summand_u64m1, vertical_positive_u64m1,
                                                                     addition_carry_b64, vl);

                    vuint64m1_t const carry_in_u64m1 = __riscv_vor_vv_u64m1(equality_u64m1, vertical_negative_u64m1,
                                                                            vl);
                    vuint64m1_t const diagonal_u64m1 = __riscv_vor_vv_u64m1(
                        __riscv_vxor_vv_u64m1(sum_u64m1, vertical_positive_u64m1, vl), carry_in_u64m1, vl);
                    vuint64m1_t horizontal_positive_u64m1 = __riscv_vor_vv_u64m1(
                        vertical_negative_u64m1,
                        __riscv_vnot_v_u64m1(__riscv_vor_vv_u64m1(diagonal_u64m1, vertical_positive_u64m1, vl), vl),
                        vl);
                    vuint64m1_t horizontal_negative_u64m1 = __riscv_vand_vv_u64m1(vertical_positive_u64m1,
                                                                                  diagonal_u64m1, vl);

                    if (word == last_word) {
                        vbool64_t const add_b64 = __riscv_vmand_mm_b64(
                            active_b64,
                            __riscv_vmsne_vx_u64m1_b64(
                                __riscv_vand_vv_u64m1(horizontal_positive_u64m1, top_mask_u64m1, vl), 0, vl),
                            vl);
                        vbool64_t const sub_b64 = __riscv_vmand_mm_b64(
                            active_b64,
                            __riscv_vmsne_vx_u64m1_b64(
                                __riscv_vand_vv_u64m1(horizontal_negative_u64m1, top_mask_u64m1, vl), 0, vl),
                            vl);
                        score_u64m1 = __riscv_vadd_vx_u64m1_mu(add_b64, score_u64m1, score_u64m1, 1, vl);
                        score_u64m1 = __riscv_vsub_vx_u64m1_mu(sub_b64, score_u64m1, score_u64m1, 1, vl);
                    }

                    vuint64m1_t const next_positive_carry_u64m1 = __riscv_vsrl_vx_u64m1(horizontal_positive_u64m1, 63,
                                                                                        vl);
                    vuint64m1_t const next_negative_carry_u64m1 = __riscv_vsrl_vx_u64m1(horizontal_negative_u64m1, 63,
                                                                                        vl);
                    horizontal_positive_u64m1 = __riscv_vor_vv_u64m1(
                        __riscv_vsll_vx_u64m1(horizontal_positive_u64m1, 1, vl), horizontal_positive_carry_u64m1, vl);
                    horizontal_negative_u64m1 = __riscv_vor_vv_u64m1(
                        __riscv_vsll_vx_u64m1(horizontal_negative_u64m1, 1, vl), horizontal_negative_carry_u64m1, vl);
                    horizontal_positive_carry_u64m1 = next_positive_carry_u64m1;
                    horizontal_negative_carry_u64m1 = next_negative_carry_u64m1;

                    vuint64m1_t const next_positive_u64m1 = __riscv_vor_vv_u64m1(
                        horizontal_negative_u64m1,
                        __riscv_vnot_v_u64m1(__riscv_vor_vv_u64m1(carry_in_u64m1, horizontal_positive_u64m1, vl), vl),
                        vl);
                    vuint64m1_t const next_negative_u64m1 = __riscv_vand_vv_u64m1(horizontal_positive_u64m1,
                                                                                  carry_in_u64m1, vl);
                    __riscv_vse64_v_u64m1(
                        vertical_positive_state + word * lanes_k,
                        __riscv_vmerge_vvm_u64m1(vertical_positive_u64m1, next_positive_u64m1, active_b64, vl), vl);
                    __riscv_vse64_v_u64m1(
                        vertical_negative_state + word * lanes_k,
                        __riscv_vmerge_vvm_u64m1(vertical_negative_u64m1, next_negative_u64m1, active_b64, vl), vl);
                }
            }

            __riscv_vse64_v_u64m1(final_scores + lane_base, score_u64m1, vl);
            lane_base += (index_t)vl;
        }

        for (index_t lane_index = 0; lane_index != lanes; ++lane_index)
            results[pairs.positions[lane_index]] = (size_t)final_scores[lane_index];
        return status_t::success_k;
    }
};

#pragma endregion Bit Parallel Myers

/** @brief Redirects the RVV horizontal-walker specialization to the serial version. The Wagner-Fischer
 *         tiny-input path is not worth vectorizing, so it shares the serial implementation. */
template <typename char_type_, typename score_type_, typename substituter_type_, typename gap_costs_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                         sz_cap_rvv_k, void>
    : public horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                               sz_cap_serial_k, void> {

    using base_t = horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                                     sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

#pragma region Uniform Cost Levenshtein

/**
 *  @brief Redirects every RVV diagonal `tile_scorer` cell to the serial version. These four catch-alls
 *         (linear/affine x global/local, generic over the substituter and objective) cover both the
 *         uniform-cost Levenshtein cells and the class-based Needleman-Wunsch / Smith-Waterman cells, so
 *         the generic serial diagonal walker can drive the whole `sz_caps_sr_k` capability at serial speed.
 *         A later wave will replace the hot cell widths with specialized vector-length-agnostic kernels;
 *         these redirects stay as the fallback for the not-yet-vectorized combinations.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                         linear_gap_costs_t, objective_, sz_similarity_global_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                               linear_gap_costs_t, objective_, sz_similarity_global_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                         linear_gap_costs_t, objective_, sz_similarity_local_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                               linear_gap_costs_t, objective_, sz_similarity_local_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                         affine_gap_costs_t, objective_, sz_similarity_global_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                               affine_gap_costs_t, objective_, sz_similarity_global_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                         affine_gap_costs_t, objective_, sz_similarity_local_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                               affine_gap_costs_t, objective_, sz_similarity_local_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

#pragma endregion // Uniform Cost Levenshtein

#pragma region RVV Inter Sequence Candidate Lanes
// ! Future home of the vector-length-agnostic candidate-lane Levenshtein driver. Until then the generic
// ! serial batch engines score the cross-product per pair via the serial diagonal walker above.
#pragma endregion RVV Inter Sequence Candidate Lanes

#pragma region Weighted Candidate Lane Walker
// ! Future home of the class-based (Needleman-Wunsch / Smith-Waterman) candidate-lane walker.
#pragma endregion Weighted Candidate Lane Walker

#pragma region Cross Product

/**
 *  @brief Batched byte-level @b Levenshtein distances on RVV, scoring up to `lanes_k` unit-cost pairs at once with the
 *      RVV vector-length-agnostic Myers family - `distances_Nx64_` (shorter <= 64), `distances_Nx_multiword_<words>`
 *      (compile-time word count, shorter <= 512), and `distances_Nx_multiword_large_` (runtime word count, shorter >
 *      512) - and the anti-diagonal DP for a lone long tail. Cells are grouped by their exact `ceil(shorter / 64)` word
 *      bucket so each group loops one common word count; the lane count is whatever a `vsetvl` over the group yields.
 *      Mirrors the NEON / Ice Lake cross-product engines (live cells walked query-major, grouped, scored, scattered
 *      into the strided result matrix plus the mirror slot for symmetric self-similarity). Non-unit substitution or gap
 *      costs route to the serial per-pair cross-product helpers, which are bit-exact with the serial oracle.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;

    using scoring_t = levenshtein_distance<char, gap_costs_t, capability_k>; // ? Per-pair DP fallback (= serial).
    using myers_t = levenshtein_distance_myers<char, capability_k>;          // ? RVV lockstep Myers.
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

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the Myers `match_masks` for the
     *      longest string, or the anti-diagonal DP fallback for the longest query x longest candidate.
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
        size_t dp_scratch = 0, multiword_scratch = 0;
        size_t const shortest_longest = sz_min_of_two(longest_query, longest_candidate);
        if (queries.size() && candidates.size() && shortest_longest > 64) {
            size_t const words_bound = divide_round_up<size_t>(shortest_longest, 64);
            multiword_scratch = myers_t::match_masks_bytes_k * words_bound;
        }
        if (queries.size() && candidates.size() && shortest_longest > 512) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(sz_max_of_two(myers_scratch, dp_scratch), multiword_scratch);
    }

#pragma region Cross Product Cell Addressing

    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

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

    static size_t live_cells_count_(size_t queries_count, size_t candidates_count,
                                    cross_similarities_t cross_kind) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) return queries_count * (queries_count + 1) / 2;
        return queries_count * candidates_count;
    }

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
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the lockstep RVV Myers family,
     *      falling back to the anti-diagonal DP for the long tail. Cells are grouped query-major by their exact
     *      `ceil(shorter / 64)` word bucket so each group loops one common word count.
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    SZ_NOINLINE status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                      results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                      size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();
        scoring_t dp {substituter_, gap_costs_};

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

            if (shorter == 0) {
                cross_cell_destination_<value_t> const destination = destination_for(query_index, candidate_index);
                cross_cell_writer_<value_t> {&destination}[0] = sz_max_of_two(query.size(), candidate.size());
                ++cell_index;
                continue;
            }

            // Gather up to `lanes_k` consecutive live cells whose shorter side shares this seed's exact word bucket.
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
                    if constexpr (bucket.value == 1) return myers.distances_Nx64_(group_pairs, group_writer, scratch);
                    else
                        return myers.template distances_Nx_multiword_<bucket.value>(group_pairs, group_writer, scratch);
                },
                [&] { return myers.distances_Nx_multiword_large_(group_pairs, group_writer, scratch); });
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
            if (length == 0) return;
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

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
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
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
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
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
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
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                              cross_similarities_t::symmetric_k, score_scratch_,
                                              std::forward<executor_type_>(executor), specs);
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion Public Cross Product Overloads
};

/**
 *  @brief Batched UTF-8 @b rune-level @b Levenshtein distances on RVV, transcoding each cell to UTF-32 once and scoring
 *      up to `lanes_k` unit-cost pairs with the RVV rune Myers family, grouped by shorter rune word bucket. Empty
 *      cells, invalid UTF-8, and the > 4096-rune long tail fall back to the per-pair UTF-8 serial scorer. Non-unit
 *      substitution or gap costs route the whole call through the per-pair UTF-8 serial scorer (bit-exact with serial).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances_utf8<linear_gap_costs_t, allocator_type_, capability_,
                                  std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;

    using scoring_t = levenshtein_distance_utf8<gap_costs_t, sz_cap_serial_k>; // ? Per-pair UTF-8 serial fallback.
    using myers_t = levenshtein_distance_myers<rune_t, capability_k>;          // ? RVV rune Myers fast path.
    static constexpr index_t myers_lanes_k = myers_t::lanes_k;
    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    uniform_substitution_costs_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    levenshtein_distances_utf8(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances_utf8(uniform_substitution_costs_t subs, linear_gap_costs_t gaps,
                               allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    bool is_unit_cost_() const noexcept {
        return substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1;
    }

#pragma region Cross Product Cell Addressing

    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    static size_t live_cells_count_(size_t queries_count, size_t candidates_count,
                                    cross_similarities_t cross_kind) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) return queries_count * (queries_count + 1) / 2;
        return queries_count * candidates_count;
    }

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

    /**
     *  @brief Worst-case scratch for a single cell over the whole input: the rune-Myers transcode arena (up to
     *      `myers_lanes_k` pairs) plus the per-lane hash `match_masks` for the longest shorter side, or the per-pair
     *      rune DP fallback for the longest query x longest candidate (byte length bounds rune count).
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
        return sz_max_of_two(myers_path, dp_scratch);
    }

#pragma region Cross Product Scoring

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` with the RVV rune Myers - the unit-cost-linear UTF-8 fast
     *      path. Each cell is transcoded to runes; consecutive cells whose shorter rune side shares the same word
     *      bucket are batched into one rune-Myers call. Empty cells, invalid UTF-8, and lone shorter > 4096 cells fall
     *      back to the per-pair rune scorer.
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    SZ_NOINLINE status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                      results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                      size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

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

        static constexpr size_t stack_words_capacity_k = 64; // ? The rune Myers large kernel covers <= 4096 runes.
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);

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
                    arena_used = arena_before;
                    break;
                }
                if (next_shorter.size() == 0 || divide_round_up<size_t>(next_shorter.size(), 64) != seed_bucket) {
                    arena_used = arena_before;
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
            status_t status = dispatch_word_bucket_<1, 8>(
                seed_bucket,
                [&](auto bucket) {
                    if constexpr (bucket.value == 1)
                        return myers.distances_Nx64_(group_pairs, group_writer, match_masks_scratch);
                    else
                        return myers.template distances_Nx_multiword_<bucket.value>(group_pairs, group_writer,
                                                                                    match_masks_scratch);
                },
                [&] { return myers.distances_Nx_multiword_large_(group_pairs, group_writer, match_masks_scratch); });
            if (status == status_t::success_k) continue;
            // Defensive scratch-shortfall fallback: score every grouped pair through the per-pair UTF-8 rune DP.
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
            if (length == 0) return;
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

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
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
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
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
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
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
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                              cross_similarities_t::symmetric_k, score_scratch_,
                                              std::forward<executor_type_>(executor), specs);
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion Public Cross Product Overloads
};

#pragma endregion Cross Product

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif            // SZ_USE_RVV
#pragma endregion // RVV Implementation

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_RVV_HPP_
