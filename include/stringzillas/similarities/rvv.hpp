/**
 *  @brief RISC-V Vector (RVV 1.0) string-similarity backend.
 *  @file include/stringzillas/similarities/rvv.hpp
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 *  @sa include/stringzillas/similarities/neon.hpp
 *
 *  @note The vector-length-agnostic kernels have landed: the batch lane-per-pair byte/rune Myers, the Levenshtein
 *        cross-product engine, the anti-diagonal NW/SW and uniform-cost Levenshtein `tile_scorer`s, and the
 *        weighted (class-cost) NW/SW candidate-lane walkers all run real `vsetvl`-sized RVV kernels. The single-pair
 *        Myers path stays on the scalar serial oracle by design (a lone pair gains nothing from lane batching, the
 *        same contract Haswell and NEON use), and a few long-tail combinations still redirect to the serial cell
 *        recurrence - those redirects are documented inline where they remain.
 */
#ifndef STRINGZILLAS_SIMILARITIES_RVV_HPP_
#define STRINGZILLAS_SIMILARITIES_RVV_HPP_

#include "stringzillas/similarities/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

/*  RVV implementation of the string similarity algorithms for 64-bit RISC-V CPUs.
 *  The hot paths run real `vsetvl`-sized RVV kernels: the batch lane-per-pair Myers, the Levenshtein cross-product
 *  engine, the anti-diagonal NW/SW and uniform-cost Levenshtein `tile_scorer`s, and the weighted NW/SW candidate-lane
 *  walkers. The single-pair Myers engine and a handful of not-yet-vectorized diagonal cell combinations still
 *  redirect to the serial oracle, so `sz_caps_sr_k` stays bit-exact with `sz_cap_serial_k` everywhere it falls back.
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
     *      @p scratch_space holds the `match_masks[lanes_k][256]` table (`match_masks_bytes_k`) followed by a
     *      transposed-text buffer of at least `max_longer * lanes_k` bytes. The lockstep scan is the byte 64-bit
     *      Myers recurrence; the per-lane `Eq` row is gathered with one `vluxei64`.
     */
    template <typename results_writer_>
    status_t distances_Nx64_(lane_pairs_view<char_t> const &pairs, results_writer_ &results,
                             scratch_space_t scratch_space) const noexcept {

        index_t const lanes = (index_t)pairs.lanes_count();
        size_t max_longer = 0;
        for (index_t lane_index = 0; lane_index != lanes; ++lane_index)
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());
        if (scratch_space.size() < match_masks_bytes_k + max_longer * lanes_k) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data()); // ? Indexed `lane * 256 + symbol`.
        u8_t *const transposed_text = reinterpret_cast<u8_t *>(scratch_space.data() + match_masks_bytes_k);
        u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        u64_t vertical_positive_init[lanes_k] = {0}, final_scores[lanes_k] = {0};
        for (size_t position = 0; position != max_longer * lanes_k; ++position) transposed_text[position] = 0;

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
            for (size_t position = 0; position != longer_length; ++position)
                transposed_text[position * lanes_k + lane_index] = (u8_t)longer[position];
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
                // Load the current symbol per lane straight from the pre-transposed text (0 for finished lanes,
                // which `active_b8` freezes anyway), then gather each lane's `Eq` bitmask row with one `vluxei64`.
                vuint64m8_t const symbol_u64m8 = __riscv_vzext_vf8_u64m8(
                    __riscv_vle8_v_u8m1(transposed_text + position * lanes_k + lane_base, vl), vl);
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

        index_t const lanes = (index_t)pairs.lanes_count();
        size_t max_longer = 0;
        for (index_t lane_index = 0; lane_index != lanes; ++lane_index)
            max_longer = sz_max_of_two(max_longer, pairs.longers[lane_index].size());

        size_t const match_masks_words = (size_t)256 * words_count * lanes_k;
        size_t const match_masks_bytes = match_masks_words * sizeof(u64_t);
        if (scratch_space.size() < match_masks_bytes + max_longer * lanes_k) return status_t::bad_alloc_k;
        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data());
        for (size_t element = 0; element != match_masks_words; ++element) match_masks[element] = 0;
        u8_t *const transposed_text = reinterpret_cast<u8_t *>(scratch_space.data() + match_masks_bytes);
        for (size_t position = 0; position != max_longer * lanes_k; ++position) transposed_text[position] = 0;

        u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        u64_t final_scores[lanes_k] = {0};
        for (index_t lane_index = 0; lane_index != lanes; ++lane_index) {
            index_t const shorter_length = (index_t)pairs.shorters[lane_index].size();
            size_t const longer_length = pairs.longers[lane_index].size();
            char_t const *const shorter = pairs.shorters[lane_index].data();
            char_t const *const longer = pairs.longers[lane_index].data();
            u64_t *const lane_table = match_masks + (size_t)lane_index * 256 * words_count;
            for (index_t position = 0; position != shorter_length; ++position)
                lane_table[(size_t)(u8_t)shorter[position] * words_count + (position >> 6)] |= (u64_t)1
                                                                                               << (position & 63);
            top_bits[lane_index] = (u64_t)1 << ((shorter_length - 1) & 63);
            shorter_lengths[lane_index] = shorter_length;
            longer_lengths[lane_index] = longer_length;
            for (size_t position = 0; position != longer_length; ++position)
                transposed_text[position * lanes_k + lane_index] = (u8_t)longer[position];
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
                vuint64m1_t const symbol_u64m1 = __riscv_vzext_vf8_u64m1(
                    __riscv_vle8_v_u8mf8(transposed_text + position * lanes_k + lane_base, vl), vl);
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

#pragma region Class Cost Needleman Wunsch and Smith Waterman

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b global Needleman-Wunsch score with class-based
 *      substitution costs, over `i16_t` cells. Requires the RVV 1.0 Vector extension.
 *
 *      The (32 x 32) `class_substitution_costs` matrix is flattened into a resident 1024-entry byte
 *      table by `prepare`, transposed when the diagonal walker has swapped the shorter/longer operands.
 *      Every chunk gathers each side's class with one `vluxei8` over the resident 256-entry
 *      `byte_to_class` map, folds the two classes into one `first_class * 32 + second_class` index with a
 *      widening multiply-add, then gathers the matching cost with one `vluxei16` over the flattened
 *      table and sign-extends it into the cell width. The anti-diagonal is swept in `vsetvl`-sized chunks,
 *      so there is no scalar tail - the last chunk's `vl` is simply shorter than the rest.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    i8_t cost_matrix_1024_[error_costs_classes_count_k * error_costs_classes_count_k];

    void prepare(bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024_[first_class * error_costs_classes_count_k + second_class] =
                    transpose ? this->substituter_.class_substitution_costs[second_class][first_class]
                              : this->substituter_.class_substitution_costs[first_class][second_class];
        this->transpose_ = transpose;
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t *scores_new, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const *const byte_to_class = this->substituter_.byte_to_class;
        i16_t const gap = static_cast<i16_t>(this->gap_costs_.open_or_extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e16m4(length - progress);

            vuint8m2_t const first_byte_u8m2 = __riscv_vle8_v_u8m2(first_u8 + progress, vl);
            vuint8m2_t const second_byte_u8m2 = __riscv_vle8_v_u8m2(second_u8 + progress, vl);
            vuint8m2_t const first_class_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, first_byte_u8m2, vl);
            vuint8m2_t const second_class_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, second_byte_u8m2, vl);

            vuint16m4_t const first_class_scaled_u16m4 = __riscv_vwmulu_vx_u16m4(first_class_u8m2,
                                                                                 (u8_t)error_costs_classes_count_k, vl);
            vuint16m4_t const combined_index_u16m4 = __riscv_vwaddu_wv_u16m4(first_class_scaled_u16m4,
                                                                             second_class_u8m2, vl);
            vint8m2_t const cost_i8m2 = __riscv_vluxei16_v_i8m2(cost_matrix_1024_, combined_index_u16m4, vl);
            vint16m4_t const cost_i16m4 = __riscv_vsext_vf2_i16m4(cost_i8m2, vl);

            vint16m4_t const pre_substitution_i16m4 = __riscv_vle16_v_i16m4(scores_pre_substitution + progress, vl);
            vint16m4_t const pre_insertion_i16m4 = __riscv_vle16_v_i16m4(scores_pre_insertion + progress, vl);
            vint16m4_t const pre_deletion_i16m4 = __riscv_vle16_v_i16m4(scores_pre_deletion + progress, vl);
            vint16m4_t const if_substitution_i16m4 = __riscv_vadd_vv_i16m4(pre_substitution_i16m4, cost_i16m4, vl);
            vint16m4_t const if_gap_i16m4 = __riscv_vadd_vx_i16m4(
                __riscv_vmax_vv_i16m4(pre_insertion_i16m4, pre_deletion_i16m4, vl), gap, vl);
            vint16m4_t const cell_i16m4 = __riscv_vmax_vv_i16m4(if_substitution_i16m4, if_gap_i16m4, vl);
            __riscv_vse16_v_i16m4(scores_new + progress, cell_i16m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b local Smith-Waterman score with class-based
 *      substitution costs, over `i16_t` cells. Requires the RVV 1.0 Vector extension.
 *
 *      Identical recurrence to the global scorer above, but zero-clamps every cell with a signed
 *      `vmax_vx` against zero, and tracks the running best across the whole matrix with one
 *      `vredmax` reduction per chunk instead of a scalar walk.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    i8_t cost_matrix_1024_[error_costs_classes_count_k * error_costs_classes_count_k];

    void prepare(bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024_[first_class * error_costs_classes_count_k + second_class] =
                    transpose ? this->substituter_.class_substitution_costs[second_class][first_class]
                              : this->substituter_.class_substitution_costs[first_class][second_class];
        this->transpose_ = transpose;
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t *scores_new, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const *const byte_to_class = this->substituter_.byte_to_class;
        i16_t const gap = static_cast<i16_t>(this->gap_costs_.open_or_extend);
        i16_t running_best = this->best_score_;

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e16m4(length - progress);

            vuint8m2_t const first_byte_u8m2 = __riscv_vle8_v_u8m2(first_u8 + progress, vl);
            vuint8m2_t const second_byte_u8m2 = __riscv_vle8_v_u8m2(second_u8 + progress, vl);
            vuint8m2_t const first_class_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, first_byte_u8m2, vl);
            vuint8m2_t const second_class_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, second_byte_u8m2, vl);

            vuint16m4_t const first_class_scaled_u16m4 = __riscv_vwmulu_vx_u16m4(first_class_u8m2,
                                                                                 (u8_t)error_costs_classes_count_k, vl);
            vuint16m4_t const combined_index_u16m4 = __riscv_vwaddu_wv_u16m4(first_class_scaled_u16m4,
                                                                             second_class_u8m2, vl);
            vint8m2_t const cost_i8m2 = __riscv_vluxei16_v_i8m2(cost_matrix_1024_, combined_index_u16m4, vl);
            vint16m4_t const cost_i16m4 = __riscv_vsext_vf2_i16m4(cost_i8m2, vl);

            vint16m4_t const pre_substitution_i16m4 = __riscv_vle16_v_i16m4(scores_pre_substitution + progress, vl);
            vint16m4_t const pre_insertion_i16m4 = __riscv_vle16_v_i16m4(scores_pre_insertion + progress, vl);
            vint16m4_t const pre_deletion_i16m4 = __riscv_vle16_v_i16m4(scores_pre_deletion + progress, vl);
            vint16m4_t const if_substitution_i16m4 = __riscv_vadd_vv_i16m4(pre_substitution_i16m4, cost_i16m4, vl);
            vint16m4_t const if_gap_i16m4 = __riscv_vadd_vx_i16m4(
                __riscv_vmax_vv_i16m4(pre_insertion_i16m4, pre_deletion_i16m4, vl), gap, vl);
            vint16m4_t cell_i16m4 = __riscv_vmax_vv_i16m4(if_substitution_i16m4, if_gap_i16m4, vl);
            cell_i16m4 = __riscv_vmax_vx_i16m4(cell_i16m4, 0, vl);
            __riscv_vse16_v_i16m4(scores_new + progress, cell_i16m4, vl);

            vint16m1_t const running_best_seed_i16m1 = __riscv_vmv_s_x_i16m1(running_best, 1);
            vint16m1_t const reduced_i16m1 = __riscv_vredmax_vs_i16m4_i16m1(cell_i16m4, running_best_seed_i16m1, vl);
            running_best = __riscv_vmv_x_s_i16m1_i16(reduced_i16m1);

            progress += vl;
        }

        this->best_score_ = running_best;
    }
};

/**
 *  @brief Variant of the global diagonal class scorer over @b `i32_t` cells, for inputs whose scores exceed
 *      the 16-bit range. Mirrors the `i16_t` scorer, but the byte/class vectors keep the narrower `m1`
 *      grouping that matches an `i32m4` cell width, and the flattened cost byte is sign-extended directly
 *      into `i32` with one `vsext_vf4`. Requires the RVV 1.0 Vector extension.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    i8_t cost_matrix_1024_[error_costs_classes_count_k * error_costs_classes_count_k];

    void prepare(bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024_[first_class * error_costs_classes_count_k + second_class] =
                    transpose ? this->substituter_.class_substitution_costs[second_class][first_class]
                              : this->substituter_.class_substitution_costs[first_class][second_class];
        this->transpose_ = transpose;
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t *scores_new, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const *const byte_to_class = this->substituter_.byte_to_class;
        i32_t const gap = static_cast<i32_t>(this->gap_costs_.open_or_extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e32m4(length - progress);

            vuint8m1_t const first_byte_u8m1 = __riscv_vle8_v_u8m1(first_u8 + progress, vl);
            vuint8m1_t const second_byte_u8m1 = __riscv_vle8_v_u8m1(second_u8 + progress, vl);
            vuint8m1_t const first_class_u8m1 = __riscv_vluxei8_v_u8m1(byte_to_class, first_byte_u8m1, vl);
            vuint8m1_t const second_class_u8m1 = __riscv_vluxei8_v_u8m1(byte_to_class, second_byte_u8m1, vl);

            vuint16m2_t const first_class_scaled_u16m2 = __riscv_vwmulu_vx_u16m2(first_class_u8m1,
                                                                                 (u8_t)error_costs_classes_count_k, vl);
            vuint16m2_t const combined_index_u16m2 = __riscv_vwaddu_wv_u16m2(first_class_scaled_u16m2,
                                                                             second_class_u8m1, vl);
            vint8m1_t const cost_i8m1 = __riscv_vluxei16_v_i8m1(cost_matrix_1024_, combined_index_u16m2, vl);
            vint32m4_t const cost_i32m4 = __riscv_vsext_vf4_i32m4(cost_i8m1, vl);

            vint32m4_t const pre_substitution_i32m4 = __riscv_vle32_v_i32m4(scores_pre_substitution + progress, vl);
            vint32m4_t const pre_insertion_i32m4 = __riscv_vle32_v_i32m4(scores_pre_insertion + progress, vl);
            vint32m4_t const pre_deletion_i32m4 = __riscv_vle32_v_i32m4(scores_pre_deletion + progress, vl);
            vint32m4_t const if_substitution_i32m4 = __riscv_vadd_vv_i32m4(pre_substitution_i32m4, cost_i32m4, vl);
            vint32m4_t const if_gap_i32m4 = __riscv_vadd_vx_i32m4(
                __riscv_vmax_vv_i32m4(pre_insertion_i32m4, pre_deletion_i32m4, vl), gap, vl);
            vint32m4_t const cell_i32m4 = __riscv_vmax_vv_i32m4(if_substitution_i32m4, if_gap_i32m4, vl);
            __riscv_vse32_v_i32m4(scores_new + progress, cell_i32m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of the @b local diagonal class scorer over `i32_t` cells. Mirrors the global `i32_t`
 *      scorer above, but zero-clamps every cell and reduces the running best with `vredmax` per chunk.
 *      Requires the RVV 1.0 Vector extension.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    i8_t cost_matrix_1024_[error_costs_classes_count_k * error_costs_classes_count_k];

    void prepare(bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024_[first_class * error_costs_classes_count_k + second_class] =
                    transpose ? this->substituter_.class_substitution_costs[second_class][first_class]
                              : this->substituter_.class_substitution_costs[first_class][second_class];
        this->transpose_ = transpose;
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t *scores_new, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const *const byte_to_class = this->substituter_.byte_to_class;
        i32_t const gap = static_cast<i32_t>(this->gap_costs_.open_or_extend);
        i32_t running_best = this->best_score_;

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e32m4(length - progress);

            vuint8m1_t const first_byte_u8m1 = __riscv_vle8_v_u8m1(first_u8 + progress, vl);
            vuint8m1_t const second_byte_u8m1 = __riscv_vle8_v_u8m1(second_u8 + progress, vl);
            vuint8m1_t const first_class_u8m1 = __riscv_vluxei8_v_u8m1(byte_to_class, first_byte_u8m1, vl);
            vuint8m1_t const second_class_u8m1 = __riscv_vluxei8_v_u8m1(byte_to_class, second_byte_u8m1, vl);

            vuint16m2_t const first_class_scaled_u16m2 = __riscv_vwmulu_vx_u16m2(first_class_u8m1,
                                                                                 (u8_t)error_costs_classes_count_k, vl);
            vuint16m2_t const combined_index_u16m2 = __riscv_vwaddu_wv_u16m2(first_class_scaled_u16m2,
                                                                             second_class_u8m1, vl);
            vint8m1_t const cost_i8m1 = __riscv_vluxei16_v_i8m1(cost_matrix_1024_, combined_index_u16m2, vl);
            vint32m4_t const cost_i32m4 = __riscv_vsext_vf4_i32m4(cost_i8m1, vl);

            vint32m4_t const pre_substitution_i32m4 = __riscv_vle32_v_i32m4(scores_pre_substitution + progress, vl);
            vint32m4_t const pre_insertion_i32m4 = __riscv_vle32_v_i32m4(scores_pre_insertion + progress, vl);
            vint32m4_t const pre_deletion_i32m4 = __riscv_vle32_v_i32m4(scores_pre_deletion + progress, vl);
            vint32m4_t const if_substitution_i32m4 = __riscv_vadd_vv_i32m4(pre_substitution_i32m4, cost_i32m4, vl);
            vint32m4_t const if_gap_i32m4 = __riscv_vadd_vx_i32m4(
                __riscv_vmax_vv_i32m4(pre_insertion_i32m4, pre_deletion_i32m4, vl), gap, vl);
            vint32m4_t cell_i32m4 = __riscv_vmax_vv_i32m4(if_substitution_i32m4, if_gap_i32m4, vl);
            cell_i32m4 = __riscv_vmax_vx_i32m4(cell_i32m4, 0, vl);
            __riscv_vse32_v_i32m4(scores_new + progress, cell_i32m4, vl);

            vint32m1_t const running_best_seed_i32m1 = __riscv_vmv_s_x_i32m1(running_best, 1);
            vint32m1_t const reduced_i32m1 = __riscv_vredmax_vs_i32m4_i32m1(cell_i32m4, running_best_seed_i32m1, vl);
            running_best = __riscv_vmv_x_s_i32m1_i32(reduced_i32m1);

            progress += vl;
        }

        this->best_score_ = running_best;
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b global Needleman-Wunsch score with class-based
 *      substitution costs and @b affine gaps, over `i16_t` cells. Requires the RVV 1.0 Vector extension.
 *
 *      Same class-pair gather as the linear-gap scorer, but threads the separate insertion and deletion
 *      gap diagonals of the Gotoh recurrence (open vs extend) alongside the main score diagonal.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    i8_t cost_matrix_1024_[error_costs_classes_count_k * error_costs_classes_count_k];

    void prepare(bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024_[first_class * error_costs_classes_count_k + second_class] =
                    transpose ? this->substituter_.class_substitution_costs[second_class][first_class]
                              : this->substituter_.class_substitution_costs[first_class][second_class];
        this->transpose_ = transpose;
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
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const *const byte_to_class = this->substituter_.byte_to_class;
        i16_t const gap_open = static_cast<i16_t>(this->gap_costs_.open);
        i16_t const gap_extend = static_cast<i16_t>(this->gap_costs_.extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e16m4(length - progress);

            vuint8m2_t const first_byte_u8m2 = __riscv_vle8_v_u8m2(first_u8 + progress, vl);
            vuint8m2_t const second_byte_u8m2 = __riscv_vle8_v_u8m2(second_u8 + progress, vl);
            vuint8m2_t const first_class_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, first_byte_u8m2, vl);
            vuint8m2_t const second_class_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, second_byte_u8m2, vl);

            vuint16m4_t const first_class_scaled_u16m4 = __riscv_vwmulu_vx_u16m4(first_class_u8m2,
                                                                                 (u8_t)error_costs_classes_count_k, vl);
            vuint16m4_t const combined_index_u16m4 = __riscv_vwaddu_wv_u16m4(first_class_scaled_u16m4,
                                                                             second_class_u8m2, vl);
            vint8m2_t const cost_i8m2 = __riscv_vluxei16_v_i8m2(cost_matrix_1024_, combined_index_u16m4, vl);
            vint16m4_t const cost_i16m4 = __riscv_vsext_vf2_i16m4(cost_i8m2, vl);

            vint16m4_t const pre_substitution_i16m4 = __riscv_vle16_v_i16m4(scores_pre_substitution + progress, vl);
            vint16m4_t const pre_insertion_open_i16m4 = __riscv_vle16_v_i16m4(scores_pre_insertion + progress, vl);
            vint16m4_t const pre_deletion_open_i16m4 = __riscv_vle16_v_i16m4(scores_pre_deletion + progress, vl);
            vint16m4_t const running_insertion_i16m4 = __riscv_vle16_v_i16m4(scores_running_insertions + progress, vl);
            vint16m4_t const running_deletion_i16m4 = __riscv_vle16_v_i16m4(scores_running_deletions + progress, vl);

            vint16m4_t const if_insertion_i16m4 = __riscv_vmax_vv_i16m4(
                __riscv_vadd_vx_i16m4(pre_insertion_open_i16m4, gap_open, vl),
                __riscv_vadd_vx_i16m4(running_insertion_i16m4, gap_extend, vl), vl);
            vint16m4_t const if_deletion_i16m4 = __riscv_vmax_vv_i16m4(
                __riscv_vadd_vx_i16m4(pre_deletion_open_i16m4, gap_open, vl),
                __riscv_vadd_vx_i16m4(running_deletion_i16m4, gap_extend, vl), vl);
            vint16m4_t const if_substitution_i16m4 = __riscv_vadd_vv_i16m4(pre_substitution_i16m4, cost_i16m4, vl);
            vint16m4_t const cell_i16m4 = __riscv_vmax_vv_i16m4(
                __riscv_vmax_vv_i16m4(if_insertion_i16m4, if_deletion_i16m4, vl), if_substitution_i16m4, vl);

            __riscv_vse16_v_i16m4(scores_new + progress, cell_i16m4, vl);
            __riscv_vse16_v_i16m4(scores_new_insertions + progress, if_insertion_i16m4, vl);
            __riscv_vse16_v_i16m4(scores_new_deletions + progress, if_deletion_i16m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b local Smith-Waterman score with class-based
 *      substitution costs and @b affine gaps, over `i16_t` cells. Requires the RVV 1.0 Vector extension.
 *
 *      Identical to the global affine scorer above, but the zero-reset applies to @b only the
 *      substitution term (the insertion/deletion gap matrices stay unclamped), and the running best
 *      across the whole matrix is reduced per chunk with `vredmax`.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    i8_t cost_matrix_1024_[error_costs_classes_count_k * error_costs_classes_count_k];

    void prepare(bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024_[first_class * error_costs_classes_count_k + second_class] =
                    transpose ? this->substituter_.class_substitution_costs[second_class][first_class]
                              : this->substituter_.class_substitution_costs[first_class][second_class];
        this->transpose_ = transpose;
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
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const *const byte_to_class = this->substituter_.byte_to_class;
        i16_t const gap_open = static_cast<i16_t>(this->gap_costs_.open);
        i16_t const gap_extend = static_cast<i16_t>(this->gap_costs_.extend);
        i16_t running_best = this->best_score_;

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e16m4(length - progress);

            vuint8m2_t const first_byte_u8m2 = __riscv_vle8_v_u8m2(first_u8 + progress, vl);
            vuint8m2_t const second_byte_u8m2 = __riscv_vle8_v_u8m2(second_u8 + progress, vl);
            vuint8m2_t const first_class_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, first_byte_u8m2, vl);
            vuint8m2_t const second_class_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, second_byte_u8m2, vl);

            vuint16m4_t const first_class_scaled_u16m4 = __riscv_vwmulu_vx_u16m4(first_class_u8m2,
                                                                                 (u8_t)error_costs_classes_count_k, vl);
            vuint16m4_t const combined_index_u16m4 = __riscv_vwaddu_wv_u16m4(first_class_scaled_u16m4,
                                                                             second_class_u8m2, vl);
            vint8m2_t const cost_i8m2 = __riscv_vluxei16_v_i8m2(cost_matrix_1024_, combined_index_u16m4, vl);
            vint16m4_t const cost_i16m4 = __riscv_vsext_vf2_i16m4(cost_i8m2, vl);

            vint16m4_t const pre_substitution_i16m4 = __riscv_vle16_v_i16m4(scores_pre_substitution + progress, vl);
            vint16m4_t const pre_insertion_open_i16m4 = __riscv_vle16_v_i16m4(scores_pre_insertion + progress, vl);
            vint16m4_t const pre_deletion_open_i16m4 = __riscv_vle16_v_i16m4(scores_pre_deletion + progress, vl);
            vint16m4_t const running_insertion_i16m4 = __riscv_vle16_v_i16m4(scores_running_insertions + progress, vl);
            vint16m4_t const running_deletion_i16m4 = __riscv_vle16_v_i16m4(scores_running_deletions + progress, vl);

            vint16m4_t const if_insertion_i16m4 = __riscv_vmax_vv_i16m4(
                __riscv_vadd_vx_i16m4(pre_insertion_open_i16m4, gap_open, vl),
                __riscv_vadd_vx_i16m4(running_insertion_i16m4, gap_extend, vl), vl);
            vint16m4_t const if_deletion_i16m4 = __riscv_vmax_vv_i16m4(
                __riscv_vadd_vx_i16m4(pre_deletion_open_i16m4, gap_open, vl),
                __riscv_vadd_vx_i16m4(running_deletion_i16m4, gap_extend, vl), vl);
            // In Local Alignment for SW the zero-reset applies to @b only the substitution term; the
            // insertion/deletion gap matrices stay unclamped, exactly like the serial scorer.
            vint16m4_t const if_substitution_i16m4 = __riscv_vmax_vx_i16m4(
                __riscv_vadd_vv_i16m4(pre_substitution_i16m4, cost_i16m4, vl), 0, vl);
            vint16m4_t const cell_i16m4 = __riscv_vmax_vv_i16m4(
                __riscv_vmax_vv_i16m4(if_insertion_i16m4, if_deletion_i16m4, vl), if_substitution_i16m4, vl);

            __riscv_vse16_v_i16m4(scores_new + progress, cell_i16m4, vl);
            __riscv_vse16_v_i16m4(scores_new_insertions + progress, if_insertion_i16m4, vl);
            __riscv_vse16_v_i16m4(scores_new_deletions + progress, if_deletion_i16m4, vl);

            vint16m1_t const running_best_seed_i16m1 = __riscv_vmv_s_x_i16m1(running_best, 1);
            vint16m1_t const reduced_i16m1 = __riscv_vredmax_vs_i16m4_i16m1(cell_i16m4, running_best_seed_i16m1, vl);
            running_best = __riscv_vmv_x_s_i16m1_i16(reduced_i16m1);

            progress += vl;
        }

        this->best_score_ = running_best;
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b global Needleman-Wunsch score with class-based
 *      substitution costs and @b affine gaps, over `i32_t` cells. Mirrors the `i16_t` affine global
 *      scorer with the narrower `m1` byte/class grouping and a direct `vsext_vf4` widen of the cost byte.
 *      Requires the RVV 1.0 Vector extension.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    i8_t cost_matrix_1024_[error_costs_classes_count_k * error_costs_classes_count_k];

    void prepare(bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024_[first_class * error_costs_classes_count_k + second_class] =
                    transpose ? this->substituter_.class_substitution_costs[second_class][first_class]
                              : this->substituter_.class_substitution_costs[first_class][second_class];
        this->transpose_ = transpose;
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
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const *const byte_to_class = this->substituter_.byte_to_class;
        i32_t const gap_open = static_cast<i32_t>(this->gap_costs_.open);
        i32_t const gap_extend = static_cast<i32_t>(this->gap_costs_.extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e32m4(length - progress);

            vuint8m1_t const first_byte_u8m1 = __riscv_vle8_v_u8m1(first_u8 + progress, vl);
            vuint8m1_t const second_byte_u8m1 = __riscv_vle8_v_u8m1(second_u8 + progress, vl);
            vuint8m1_t const first_class_u8m1 = __riscv_vluxei8_v_u8m1(byte_to_class, first_byte_u8m1, vl);
            vuint8m1_t const second_class_u8m1 = __riscv_vluxei8_v_u8m1(byte_to_class, second_byte_u8m1, vl);

            vuint16m2_t const first_class_scaled_u16m2 = __riscv_vwmulu_vx_u16m2(first_class_u8m1,
                                                                                 (u8_t)error_costs_classes_count_k, vl);
            vuint16m2_t const combined_index_u16m2 = __riscv_vwaddu_wv_u16m2(first_class_scaled_u16m2,
                                                                             second_class_u8m1, vl);
            vint8m1_t const cost_i8m1 = __riscv_vluxei16_v_i8m1(cost_matrix_1024_, combined_index_u16m2, vl);
            vint32m4_t const cost_i32m4 = __riscv_vsext_vf4_i32m4(cost_i8m1, vl);

            vint32m4_t const pre_substitution_i32m4 = __riscv_vle32_v_i32m4(scores_pre_substitution + progress, vl);
            vint32m4_t const pre_insertion_open_i32m4 = __riscv_vle32_v_i32m4(scores_pre_insertion + progress, vl);
            vint32m4_t const pre_deletion_open_i32m4 = __riscv_vle32_v_i32m4(scores_pre_deletion + progress, vl);
            vint32m4_t const running_insertion_i32m4 = __riscv_vle32_v_i32m4(scores_running_insertions + progress, vl);
            vint32m4_t const running_deletion_i32m4 = __riscv_vle32_v_i32m4(scores_running_deletions + progress, vl);

            vint32m4_t const if_insertion_i32m4 = __riscv_vmax_vv_i32m4(
                __riscv_vadd_vx_i32m4(pre_insertion_open_i32m4, gap_open, vl),
                __riscv_vadd_vx_i32m4(running_insertion_i32m4, gap_extend, vl), vl);
            vint32m4_t const if_deletion_i32m4 = __riscv_vmax_vv_i32m4(
                __riscv_vadd_vx_i32m4(pre_deletion_open_i32m4, gap_open, vl),
                __riscv_vadd_vx_i32m4(running_deletion_i32m4, gap_extend, vl), vl);
            vint32m4_t const if_substitution_i32m4 = __riscv_vadd_vv_i32m4(pre_substitution_i32m4, cost_i32m4, vl);
            vint32m4_t const cell_i32m4 = __riscv_vmax_vv_i32m4(
                __riscv_vmax_vv_i32m4(if_insertion_i32m4, if_deletion_i32m4, vl), if_substitution_i32m4, vl);

            __riscv_vse32_v_i32m4(scores_new + progress, cell_i32m4, vl);
            __riscv_vse32_v_i32m4(scores_new_insertions + progress, if_insertion_i32m4, vl);
            __riscv_vse32_v_i32m4(scores_new_deletions + progress, if_deletion_i32m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b local Smith-Waterman score with class-based
 *      substitution costs and @b affine gaps, over `i32_t` cells. Mirrors the `i16_t` affine local
 *      scorer's zero-reset-on-substitution-only and `vredmax` running best, over the wider cell type.
 *      Requires the RVV 1.0 Vector extension.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    i8_t cost_matrix_1024_[error_costs_classes_count_k * error_costs_classes_count_k];

    void prepare(bool transpose) noexcept {
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024_[first_class * error_costs_classes_count_k + second_class] =
                    transpose ? this->substituter_.class_substitution_costs[second_class][first_class]
                              : this->substituter_.class_substitution_costs[first_class][second_class];
        this->transpose_ = transpose;
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
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const *const byte_to_class = this->substituter_.byte_to_class;
        i32_t const gap_open = static_cast<i32_t>(this->gap_costs_.open);
        i32_t const gap_extend = static_cast<i32_t>(this->gap_costs_.extend);
        i32_t running_best = this->best_score_;

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e32m4(length - progress);

            vuint8m1_t const first_byte_u8m1 = __riscv_vle8_v_u8m1(first_u8 + progress, vl);
            vuint8m1_t const second_byte_u8m1 = __riscv_vle8_v_u8m1(second_u8 + progress, vl);
            vuint8m1_t const first_class_u8m1 = __riscv_vluxei8_v_u8m1(byte_to_class, first_byte_u8m1, vl);
            vuint8m1_t const second_class_u8m1 = __riscv_vluxei8_v_u8m1(byte_to_class, second_byte_u8m1, vl);

            vuint16m2_t const first_class_scaled_u16m2 = __riscv_vwmulu_vx_u16m2(first_class_u8m1,
                                                                                 (u8_t)error_costs_classes_count_k, vl);
            vuint16m2_t const combined_index_u16m2 = __riscv_vwaddu_wv_u16m2(first_class_scaled_u16m2,
                                                                             second_class_u8m1, vl);
            vint8m1_t const cost_i8m1 = __riscv_vluxei16_v_i8m1(cost_matrix_1024_, combined_index_u16m2, vl);
            vint32m4_t const cost_i32m4 = __riscv_vsext_vf4_i32m4(cost_i8m1, vl);

            vint32m4_t const pre_substitution_i32m4 = __riscv_vle32_v_i32m4(scores_pre_substitution + progress, vl);
            vint32m4_t const pre_insertion_open_i32m4 = __riscv_vle32_v_i32m4(scores_pre_insertion + progress, vl);
            vint32m4_t const pre_deletion_open_i32m4 = __riscv_vle32_v_i32m4(scores_pre_deletion + progress, vl);
            vint32m4_t const running_insertion_i32m4 = __riscv_vle32_v_i32m4(scores_running_insertions + progress, vl);
            vint32m4_t const running_deletion_i32m4 = __riscv_vle32_v_i32m4(scores_running_deletions + progress, vl);

            vint32m4_t const if_insertion_i32m4 = __riscv_vmax_vv_i32m4(
                __riscv_vadd_vx_i32m4(pre_insertion_open_i32m4, gap_open, vl),
                __riscv_vadd_vx_i32m4(running_insertion_i32m4, gap_extend, vl), vl);
            vint32m4_t const if_deletion_i32m4 = __riscv_vmax_vv_i32m4(
                __riscv_vadd_vx_i32m4(pre_deletion_open_i32m4, gap_open, vl),
                __riscv_vadd_vx_i32m4(running_deletion_i32m4, gap_extend, vl), vl);
            // Zero-reset applies to @b only the substitution term, exactly like the serial scorer.
            vint32m4_t const if_substitution_i32m4 = __riscv_vmax_vx_i32m4(
                __riscv_vadd_vv_i32m4(pre_substitution_i32m4, cost_i32m4, vl), 0, vl);
            vint32m4_t const cell_i32m4 = __riscv_vmax_vv_i32m4(
                __riscv_vmax_vv_i32m4(if_insertion_i32m4, if_deletion_i32m4, vl), if_substitution_i32m4, vl);

            __riscv_vse32_v_i32m4(scores_new + progress, cell_i32m4, vl);
            __riscv_vse32_v_i32m4(scores_new_insertions + progress, if_insertion_i32m4, vl);
            __riscv_vse32_v_i32m4(scores_new_deletions + progress, if_deletion_i32m4, vl);

            vint32m1_t const running_best_seed_i32m1 = __riscv_vmv_s_x_i32m1(running_best, 1);
            vint32m1_t const reduced_i32m1 = __riscv_vredmax_vs_i32m4_i32m1(cell_i32m4, running_best_seed_i32m1, vl);
            running_best = __riscv_vmv_x_s_i32m1_i32(reduced_i32m1);

            progress += vl;
        }

        this->best_score_ = running_best;
    }
};

#pragma endregion Class Cost Needleman Wunsch and Smith Waterman

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

/**
 *  @brief RVV @b uniform-cost diagonal scorer - minimizes the Levenshtein distance over `u8_t` cells.
 *      Requires the RVV 1.0 Vector extension.
 *
 *      The (32 x 32) class lookup of the Needleman-Wunsch scorers above collapses to a single
 *      `vmseq` equality mask plus a `vmerge` select between the resident match/mismatch costs, and the
 *      objective is minimization, so the recurrence uses `vminu` with a positive gap. The text and the
 *      cell width share the same `m4` register group here, since both are 8-bit.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,           //
        u8_t const *scores_pre_deletion, u8_t *scores_new, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const match = (u8_t)this->substituter_.match;
        u8_t const mismatch = (u8_t)this->substituter_.mismatch;
        u8_t const gap = static_cast<u8_t>(this->gap_costs_.open_or_extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e8m4(length - progress);

            vuint8m4_t const first_u8m4 = __riscv_vle8_v_u8m4(first_u8 + progress, vl);
            vuint8m4_t const second_u8m4 = __riscv_vle8_v_u8m4(second_u8 + progress, vl);
            vbool2_t const match_b2 = __riscv_vmseq_vv_u8m4_b2(first_u8m4, second_u8m4, vl);
            vuint8m4_t const cost_u8m4 = __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(mismatch, vl), match, match_b2,
                                                                 vl);

            vuint8m4_t const pre_substitution_u8m4 = __riscv_vle8_v_u8m4(scores_pre_substitution + progress, vl);
            vuint8m4_t const pre_insertion_u8m4 = __riscv_vle8_v_u8m4(scores_pre_insertion + progress, vl);
            vuint8m4_t const pre_deletion_u8m4 = __riscv_vle8_v_u8m4(scores_pre_deletion + progress, vl);
            vuint8m4_t const if_substitution_u8m4 = __riscv_vadd_vv_u8m4(pre_substitution_u8m4, cost_u8m4, vl);
            vuint8m4_t const if_gap_u8m4 = __riscv_vadd_vx_u8m4(
                __riscv_vminu_vv_u8m4(pre_insertion_u8m4, pre_deletion_u8m4, vl), gap, vl);
            vuint8m4_t const cell_u8m4 = __riscv_vminu_vv_u8m4(if_substitution_u8m4, if_gap_u8m4, vl);
            __riscv_vse8_v_u8m4(scores_new + progress, cell_u8m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief RVV @b uniform-cost diagonal scorer - minimizes the Levenshtein distance over `u16_t` cells.
 *      Requires the RVV 1.0 Vector extension.
 *
 *      The equality mask is computed once over the `u8m2` byte group and reused directly for the `u16m4`
 *      cost `vmerge`, since both groups share the same vl and the same `SEW / LMUL` ratio - no widening
 *      pass over the mask is needed.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,         //
        u16_t const *scores_pre_deletion, u16_t *scores_new, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u16_t const match = (u16_t)this->substituter_.match;
        u16_t const mismatch = (u16_t)this->substituter_.mismatch;
        u16_t const gap = static_cast<u16_t>(this->gap_costs_.open_or_extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e16m4(length - progress);

            vuint8m2_t const first_u8m2 = __riscv_vle8_v_u8m2(first_u8 + progress, vl);
            vuint8m2_t const second_u8m2 = __riscv_vle8_v_u8m2(second_u8 + progress, vl);
            vbool4_t const match_b4 = __riscv_vmseq_vv_u8m2_b4(first_u8m2, second_u8m2, vl);
            vuint16m4_t const cost_u16m4 = __riscv_vmerge_vxm_u16m4(__riscv_vmv_v_x_u16m4(mismatch, vl), match,
                                                                    match_b4, vl);

            vuint16m4_t const pre_substitution_u16m4 = __riscv_vle16_v_u16m4(scores_pre_substitution + progress, vl);
            vuint16m4_t const pre_insertion_u16m4 = __riscv_vle16_v_u16m4(scores_pre_insertion + progress, vl);
            vuint16m4_t const pre_deletion_u16m4 = __riscv_vle16_v_u16m4(scores_pre_deletion + progress, vl);
            vuint16m4_t const if_substitution_u16m4 = __riscv_vadd_vv_u16m4(pre_substitution_u16m4, cost_u16m4, vl);
            vuint16m4_t const if_gap_u16m4 = __riscv_vadd_vx_u16m4(
                __riscv_vminu_vv_u16m4(pre_insertion_u16m4, pre_deletion_u16m4, vl), gap, vl);
            vuint16m4_t const cell_u16m4 = __riscv_vminu_vv_u16m4(if_substitution_u16m4, if_gap_u16m4, vl);
            __riscv_vse16_v_u16m4(scores_new + progress, cell_u16m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief RVV @b uniform-cost diagonal scorer - minimizes the Levenshtein distance over `u32_t` cells.
 *      Requires the RVV 1.0 Vector extension. Same recurrence as the `u16_t` scorer, with the `u8m1`
 *      byte group whose `SEW / LMUL` ratio matches the `u32m4` cell width.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,         //
        u32_t const *scores_pre_deletion, u32_t *scores_new, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u32_t const match = (u32_t)this->substituter_.match;
        u32_t const mismatch = (u32_t)this->substituter_.mismatch;
        u32_t const gap = static_cast<u32_t>(this->gap_costs_.open_or_extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e32m4(length - progress);

            vuint8m1_t const first_u8m1 = __riscv_vle8_v_u8m1(first_u8 + progress, vl);
            vuint8m1_t const second_u8m1 = __riscv_vle8_v_u8m1(second_u8 + progress, vl);
            vbool8_t const match_b8 = __riscv_vmseq_vv_u8m1_b8(first_u8m1, second_u8m1, vl);
            vuint32m4_t const cost_u32m4 = __riscv_vmerge_vxm_u32m4(__riscv_vmv_v_x_u32m4(mismatch, vl), match,
                                                                    match_b8, vl);

            vuint32m4_t const pre_substitution_u32m4 = __riscv_vle32_v_u32m4(scores_pre_substitution + progress, vl);
            vuint32m4_t const pre_insertion_u32m4 = __riscv_vle32_v_u32m4(scores_pre_insertion + progress, vl);
            vuint32m4_t const pre_deletion_u32m4 = __riscv_vle32_v_u32m4(scores_pre_deletion + progress, vl);
            vuint32m4_t const if_substitution_u32m4 = __riscv_vadd_vv_u32m4(pre_substitution_u32m4, cost_u32m4, vl);
            vuint32m4_t const if_gap_u32m4 = __riscv_vadd_vx_u32m4(
                __riscv_vminu_vv_u32m4(pre_insertion_u32m4, pre_deletion_u32m4, vl), gap, vl);
            vuint32m4_t const cell_u32m4 = __riscv_vminu_vv_u32m4(if_substitution_u32m4, if_gap_u32m4, vl);
            __riscv_vse32_v_u32m4(scores_new + progress, cell_u32m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief RVV @b affine-gap uniform-cost diagonal scorer - minimizes the Levenshtein distance over
 *      `u8_t` cells. Requires the RVV 1.0 Vector extension. Gotoh recurrence with separate insertion
 *      and deletion gap planes, all in `u8`.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,           //
        u8_t const *scores_pre_deletion, u8_t const *scores_running_insertions,          //
        u8_t const *scores_running_deletions, u8_t *scores_new,                          //
        u8_t *scores_new_insertions, u8_t *scores_new_deletions, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u8_t const match = (u8_t)this->substituter_.match;
        u8_t const mismatch = (u8_t)this->substituter_.mismatch;
        u8_t const gap_open = static_cast<u8_t>(this->gap_costs_.open);
        u8_t const gap_extend = static_cast<u8_t>(this->gap_costs_.extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e8m4(length - progress);

            vuint8m4_t const first_u8m4 = __riscv_vle8_v_u8m4(first_u8 + progress, vl);
            vuint8m4_t const second_u8m4 = __riscv_vle8_v_u8m4(second_u8 + progress, vl);
            vbool2_t const match_b2 = __riscv_vmseq_vv_u8m4_b2(first_u8m4, second_u8m4, vl);
            vuint8m4_t const cost_u8m4 = __riscv_vmerge_vxm_u8m4(__riscv_vmv_v_x_u8m4(mismatch, vl), match, match_b2,
                                                                 vl);

            vuint8m4_t const pre_substitution_u8m4 = __riscv_vle8_v_u8m4(scores_pre_substitution + progress, vl);
            vuint8m4_t const pre_insertion_open_u8m4 = __riscv_vle8_v_u8m4(scores_pre_insertion + progress, vl);
            vuint8m4_t const pre_deletion_open_u8m4 = __riscv_vle8_v_u8m4(scores_pre_deletion + progress, vl);
            vuint8m4_t const running_insertion_u8m4 = __riscv_vle8_v_u8m4(scores_running_insertions + progress, vl);
            vuint8m4_t const running_deletion_u8m4 = __riscv_vle8_v_u8m4(scores_running_deletions + progress, vl);

            vuint8m4_t const if_insertion_u8m4 = __riscv_vminu_vv_u8m4(
                __riscv_vadd_vx_u8m4(pre_insertion_open_u8m4, gap_open, vl),
                __riscv_vadd_vx_u8m4(running_insertion_u8m4, gap_extend, vl), vl);
            vuint8m4_t const if_deletion_u8m4 = __riscv_vminu_vv_u8m4(
                __riscv_vadd_vx_u8m4(pre_deletion_open_u8m4, gap_open, vl),
                __riscv_vadd_vx_u8m4(running_deletion_u8m4, gap_extend, vl), vl);
            vuint8m4_t const if_substitution_u8m4 = __riscv_vadd_vv_u8m4(pre_substitution_u8m4, cost_u8m4, vl);
            vuint8m4_t const cell_u8m4 = __riscv_vminu_vv_u8m4(
                __riscv_vminu_vv_u8m4(if_insertion_u8m4, if_deletion_u8m4, vl), if_substitution_u8m4, vl);

            __riscv_vse8_v_u8m4(scores_new + progress, cell_u8m4, vl);
            __riscv_vse8_v_u8m4(scores_new_insertions + progress, if_insertion_u8m4, vl);
            __riscv_vse8_v_u8m4(scores_new_deletions + progress, if_deletion_u8m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief RVV @b affine-gap uniform-cost diagonal scorer - minimizes the Levenshtein distance over
 *      `u16_t` cells. Requires the RVV 1.0 Vector extension. Gotoh recurrence with separate insertion
 *      and deletion gap planes.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,         //
        u16_t const *scores_pre_deletion, u16_t const *scores_running_insertions,        //
        u16_t const *scores_running_deletions, u16_t *scores_new,                        //
        u16_t *scores_new_insertions, u16_t *scores_new_deletions, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u16_t const match = (u16_t)this->substituter_.match;
        u16_t const mismatch = (u16_t)this->substituter_.mismatch;
        u16_t const gap_open = static_cast<u16_t>(this->gap_costs_.open);
        u16_t const gap_extend = static_cast<u16_t>(this->gap_costs_.extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e16m4(length - progress);

            vuint8m2_t const first_u8m2 = __riscv_vle8_v_u8m2(first_u8 + progress, vl);
            vuint8m2_t const second_u8m2 = __riscv_vle8_v_u8m2(second_u8 + progress, vl);
            vbool4_t const match_b4 = __riscv_vmseq_vv_u8m2_b4(first_u8m2, second_u8m2, vl);
            vuint16m4_t const cost_u16m4 = __riscv_vmerge_vxm_u16m4(__riscv_vmv_v_x_u16m4(mismatch, vl), match,
                                                                    match_b4, vl);

            vuint16m4_t const pre_substitution_u16m4 = __riscv_vle16_v_u16m4(scores_pre_substitution + progress, vl);
            vuint16m4_t const pre_insertion_open_u16m4 = __riscv_vle16_v_u16m4(scores_pre_insertion + progress, vl);
            vuint16m4_t const pre_deletion_open_u16m4 = __riscv_vle16_v_u16m4(scores_pre_deletion + progress, vl);
            vuint16m4_t const running_insertion_u16m4 = __riscv_vle16_v_u16m4(scores_running_insertions + progress, vl);
            vuint16m4_t const running_deletion_u16m4 = __riscv_vle16_v_u16m4(scores_running_deletions + progress, vl);

            vuint16m4_t const if_insertion_u16m4 = __riscv_vminu_vv_u16m4(
                __riscv_vadd_vx_u16m4(pre_insertion_open_u16m4, gap_open, vl),
                __riscv_vadd_vx_u16m4(running_insertion_u16m4, gap_extend, vl), vl);
            vuint16m4_t const if_deletion_u16m4 = __riscv_vminu_vv_u16m4(
                __riscv_vadd_vx_u16m4(pre_deletion_open_u16m4, gap_open, vl),
                __riscv_vadd_vx_u16m4(running_deletion_u16m4, gap_extend, vl), vl);
            vuint16m4_t const if_substitution_u16m4 = __riscv_vadd_vv_u16m4(pre_substitution_u16m4, cost_u16m4, vl);
            vuint16m4_t const cell_u16m4 = __riscv_vminu_vv_u16m4(
                __riscv_vminu_vv_u16m4(if_insertion_u16m4, if_deletion_u16m4, vl), if_substitution_u16m4, vl);

            __riscv_vse16_v_u16m4(scores_new + progress, cell_u16m4, vl);
            __riscv_vse16_v_u16m4(scores_new_insertions + progress, if_insertion_u16m4, vl);
            __riscv_vse16_v_u16m4(scores_new_deletions + progress, if_deletion_u16m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

/**
 *  @brief RVV @b affine-gap uniform-cost diagonal scorer - minimizes the Levenshtein distance over
 *      `u32_t` cells. Requires the RVV 1.0 Vector extension. Same Gotoh recurrence as the `u16_t`
 *      scorer, with the `u8m1` byte group whose `SEW / LMUL` ratio matches the `u32m4` cell width.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,         //
        u32_t const *scores_pre_deletion, u32_t const *scores_running_insertions,        //
        u32_t const *scores_running_deletions, u32_t *scores_new,                        //
        u32_t *scores_new_insertions, u32_t *scores_new_deletions, executor_type_ &&executor = {}) noexcept {
        sz_unused_(executor);

        u8_t const *const first_u8 = (u8_t const *)first_reversed_slice;
        u8_t const *const second_u8 = (u8_t const *)second_slice;
        u32_t const match = (u32_t)this->substituter_.match;
        u32_t const mismatch = (u32_t)this->substituter_.mismatch;
        u32_t const gap_open = static_cast<u32_t>(this->gap_costs_.open);
        u32_t const gap_extend = static_cast<u32_t>(this->gap_costs_.extend);

        for (size_t progress = 0; progress != length;) {
            size_t const vl = __riscv_vsetvl_e32m4(length - progress);

            vuint8m1_t const first_u8m1 = __riscv_vle8_v_u8m1(first_u8 + progress, vl);
            vuint8m1_t const second_u8m1 = __riscv_vle8_v_u8m1(second_u8 + progress, vl);
            vbool8_t const match_b8 = __riscv_vmseq_vv_u8m1_b8(first_u8m1, second_u8m1, vl);
            vuint32m4_t const cost_u32m4 = __riscv_vmerge_vxm_u32m4(__riscv_vmv_v_x_u32m4(mismatch, vl), match,
                                                                    match_b8, vl);

            vuint32m4_t const pre_substitution_u32m4 = __riscv_vle32_v_u32m4(scores_pre_substitution + progress, vl);
            vuint32m4_t const pre_insertion_open_u32m4 = __riscv_vle32_v_u32m4(scores_pre_insertion + progress, vl);
            vuint32m4_t const pre_deletion_open_u32m4 = __riscv_vle32_v_u32m4(scores_pre_deletion + progress, vl);
            vuint32m4_t const running_insertion_u32m4 = __riscv_vle32_v_u32m4(scores_running_insertions + progress, vl);
            vuint32m4_t const running_deletion_u32m4 = __riscv_vle32_v_u32m4(scores_running_deletions + progress, vl);

            vuint32m4_t const if_insertion_u32m4 = __riscv_vminu_vv_u32m4(
                __riscv_vadd_vx_u32m4(pre_insertion_open_u32m4, gap_open, vl),
                __riscv_vadd_vx_u32m4(running_insertion_u32m4, gap_extend, vl), vl);
            vuint32m4_t const if_deletion_u32m4 = __riscv_vminu_vv_u32m4(
                __riscv_vadd_vx_u32m4(pre_deletion_open_u32m4, gap_open, vl),
                __riscv_vadd_vx_u32m4(running_deletion_u32m4, gap_extend, vl), vl);
            vuint32m4_t const if_substitution_u32m4 = __riscv_vadd_vv_u32m4(pre_substitution_u32m4, cost_u32m4, vl);
            vuint32m4_t const cell_u32m4 = __riscv_vminu_vv_u32m4(
                __riscv_vminu_vv_u32m4(if_insertion_u32m4, if_deletion_u32m4, vl), if_substitution_u32m4, vl);

            __riscv_vse32_v_u32m4(scores_new + progress, cell_u32m4, vl);
            __riscv_vse32_v_u32m4(scores_new_insertions + progress, if_insertion_u32m4, vl);
            __riscv_vse32_v_u32m4(scores_new_deletions + progress, if_deletion_u32m4, vl);

            progress += vl;
        }

        this->last_score_ = scores_new[length - 1];
    }
};

#pragma endregion // Uniform Cost Levenshtein

#pragma region RVV Inter Sequence Candidate Lanes
// ! Future home of the lane-per-pair candidate-lane Levenshtein driver (mirrors the weighted NW/SW walker below,
// ! narrowed to the uniform-cost recurrence). Until then, non-unit-cost Levenshtein cross-products fall back to the
// ! `levenshtein_distances::cross_sequentially_`/`cross_in_parallel_` per-pair path, which still runs a real RVV
// ! anti-diagonal `tile_scorer` per pair (see "Uniform Cost Levenshtein" above) - just without the cross-pair lane
// ! batching that the candidate-lane walker would add for short, non-unit-cost batches.
#pragma endregion RVV Inter Sequence Candidate Lanes

#pragma region Weighted Candidate Lane Walker

/**
 *  @brief Inter-sequence RVV walker: one query against up to 64 candidates packed one-per-lane, weighted
 *      Needleman-Wunsch (global) / Smith-Waterman (local) with 32-class substitution costs, 16-bit signed cells.
 *
 *  Vector-length-agnostic twin of the NEON 8-lane `i16` weighted walker: one `__riscv_vsetvl_e16m8` call over the
 *  `candidate_lanes_k` cap binds every row's recurrence to one `vint16m8_t` register group - exactly 64 lanes at the
 *  smallest tested `VLEN == 128`, more at wider `VLEN` - so a single instruction stream covers every live candidate
 *  regardless of hardware width, with no inner chunking loop. ONE body per score width, templated over the gap model
 *  (linear vs @b affine) and the locality (global vs @b local) and resolved entirely with `if constexpr`, mirroring
 *  the NEON kernel's single-body design:
 *
 *  - Candidate classes are pre-gathered once per column with `vluxei8` over the resident `byte_to_class` map.
 *  - Every row folds the broadcast (scalar) query class with the cached per-lane candidate class into one
 *    `query_class * 32 + candidate_class` index via a zero-extend plus `vadd_vx`, gathers the matching `i8` cost
 *    from the resident 1024-entry table with `vluxei16`, and sign-extends it into the cell width with `vsext_vf2`.
 *  - @b Substitution: `cell_if_substitution = diagonal + substitution`, identical for every variant.
 *  - @b Gap (linear): `cell_if_gap = max(up, left) + gap` over signed `i16` cells with a (typically negative) gap.
 *  - @b Gap (affine): the branchless @b Gotoh recurrence with two extra score tracks - a deletion track `F` carried
 *    in a previous/current row pair and an insertion track `E` carried in a running per-lane register:
 *
 *        F = max(up_cell + open, F_up + extend)      // deletion track, vertical
 *        E = max(left_cell + open, E_left + extend)  // insertion track, horizontal
 *        cell_if_gap = max(E, F)
 *
 *    The boundary gap tracks are seeded with the discarded `open + extend + boundary` sentinel (mirroring the serial
 *    `init_gap`) so the first gap on either axis correctly pays `open`.
 *  - @b Global: the recurrence is `cell = max(cell_if_substitution, cell_if_gap)`; the result is each lane's own
 *    final column (ragged lengths mean different columns per lane).
 *  - @b Local: the cell clamps to zero `cell = max(0, cell_if_substitution, cell_if_gap)`, the boundary row/column
 *    are zero, and the result is a per-lane running maximum, folded only for columns within each lane's candidate
 *    length via a `vmsgt` mask and `vmerge` so a shorter candidate's zero-padded tail cannot inflate its score.
 *
 *  @note Signed `i16` cells; valid while every reachable score stays within the caller's contracted `i16` range
 *      (`[-32768, 32767]` global, `[0, 32767]` local).
 *  @note Requires the RVV 1.0 Vector extension.
 */
template <typename gap_costs_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_,
          sz_capability_t capability_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, gap_costs_type_, objective_, locality_, capability_, 64,
                             std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 64;
    static constexpr bool is_affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    // The signed `i16` recurrence hardcodes `vmax`; minimization would need a different blend.
    static_assert(
        objective_ == sz_maximize_score_k,
        "The weighted candidate-lane kernel only implements score " "maximization (Needleman-Wunsch / " "Smith-" "Water" "man)" ".");

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Scratch holds the score rows of `longest_candidate + 1` lane-vectors (64 `i16` cells each), plus one
     *      cached candidate-class lane-vector (64 `u8`) per candidate position. The linear gap needs two score rows;
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
     *  @param[in] candidates Transposed block of up to 64 candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);
        size_t const vl = __riscv_vsetvl_e16m8(candidate_lanes_k);

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
        u8_t const *const byte_to_class = substituter_.byte_to_class;
        i8_t cost_matrix_1024[error_costs_classes_count_k * error_costs_classes_count_k];
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024[first_class * error_costs_classes_count_k + second_class] =
                    substituter_.class_substitution_costs[first_class][second_class];

        // Pre-classify every candidate column once; the `candidate_lanes_k` candidate bytes per column are
        // loop-invariant in rows.
        for (size_t column = 0; column < longest_candidate; ++column) {
            vuint8m4_t const candidate_bytes_u8m4 = __riscv_vle8_v_u8m4((u8_t const *)candidates.position(column), vl);
            vuint8m4_t const candidate_classes_u8m4 = __riscv_vluxei8_v_u8m4(byte_to_class, candidate_bytes_u8m4, vl);
            __riscv_vse8_v_u8m4(candidate_classes + column * candidate_lanes_k, candidate_classes_u8m4, vl);
        }

        vint16m8_t const zero_i16m8 = __riscv_vmv_v_x_i16m8(0, vl);
        // Gap-cost broadcasts: linear keeps a single `gap`, affine keeps `open` / `extend`. `gap` doubles as `open`
        // (with `extend` zero) so the boundary fills and the local-affine sentinel share one expression per variant.
        [[maybe_unused]] vint16m8_t gap_i16m8 {};
        [[maybe_unused]] vint16m8_t open_i16m8 {};
        [[maybe_unused]] vint16m8_t extend_i16m8 {};
        error_cost_t open = 0;
        error_cost_t extend = 0;
        if constexpr (is_affine_k) {
            open = gap_costs_.open;
            extend = gap_costs_.extend;
            open_i16m8 = __riscv_vmv_v_x_i16m8(static_cast<i16_t>(open), vl);
            extend_i16m8 = __riscv_vmv_v_x_i16m8(static_cast<i16_t>(extend), vl);
        }
        else {
            open = gap_costs_.open_or_extend;
            gap_i16m8 = __riscv_vmv_v_x_i16m8(static_cast<i16_t>(open), vl);
        }
        // The discarded gap-track sentinel for local affine: extending it never beats opening fresh.
        [[maybe_unused]] vint16m8_t const sentinel_i16m8 = __riscv_vmv_v_x_i16m8(static_cast<i16_t>(open + extend), vl);

        // Per-lane candidate lengths gate the local running-maximum so a shorter candidate's padded columns are
        // excluded; the global latch reads each lane's own final column instead and needs neither.
        [[maybe_unused]] i16_t lane_lengths[candidate_lanes_k] = {0};
        [[maybe_unused]] vint16m8_t lane_lengths_i16m8 = zero_i16m8;
        [[maybe_unused]] vint16m8_t running_max_i16m8 = zero_i16m8;
        if constexpr (is_local_k) {
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                lane_lengths[lane_index] = static_cast<i16_t>(candidates.lengths[lane_index]);
            lane_lengths_i16m8 = __riscv_vle16_v_i16m8(lane_lengths, vl);
        }

        // Row 0 (boundary). Local: the boundary row and column are zero (an alignment may begin at any cell). Global:
        // the empty query prefix against every candidate prefix is an all-gap run.
        for (size_t column = 0; column <= longest_candidate; ++column) {
            if constexpr (is_local_k) {
                __riscv_vse16_v_i16m8(previous_row + column * candidate_lanes_k, zero_i16m8, vl);
                if constexpr (is_affine_k)
                    __riscv_vse16_v_i16m8(previous_deletes + column * candidate_lanes_k, sentinel_i16m8, vl);
            }
            else if constexpr (is_affine_k) {
                // Column 0 is 0; column j (>= 1) costs `open + extend * (j - 1)`. The deletion track is seeded with
                // the discarded `open + extend + boundary` sentinel so the first vertical gap in row 1 pays `open`.
                i16_t const boundary = static_cast<i16_t>(column ? open + extend * (i16_t)(column - 1) : 0);
                __riscv_vse16_v_i16m8(previous_row + column * candidate_lanes_k, __riscv_vmv_v_x_i16m8(boundary, vl),
                                      vl);
                __riscv_vse16_v_i16m8(previous_deletes + column * candidate_lanes_k,
                                      __riscv_vmv_v_x_i16m8(static_cast<i16_t>(open + extend + boundary), vl), vl);
            }
            else {
                // Linear: a run of `gap * column`, identical across lanes (masked per-lane at latch time).
                __riscv_vse16_v_i16m8(previous_row + column * candidate_lanes_k,
                                      __riscv_vmv_v_x_i16m8(static_cast<i16_t>(open * (i16_t)column), vl), vl);
            }
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            // The query character is fixed for the whole row, so its class is scalar and broadcast across lanes.
            u8_t const query_class = byte_to_class[(u8_t)query[query_position - 1]];
            // Hoisted once per row: the 32-byte class-cost row for this `query_class`, resident in a register group
            // sized to match `candidate_classes_u8m4` so every column folds it with one in-register `vrgather_vv`
            // instead of re-deriving the combined table index and re-gathering from memory per cell.
            vuint8m4_t const cost_row_u8m4 = __riscv_vle8_v_u8m4(
                reinterpret_cast<u8_t const *>(cost_matrix_1024 + (size_t)query_class * error_costs_classes_count_k),
                error_costs_classes_count_k);

            [[maybe_unused]] vint16m8_t running_inserts_i16m8 = zero_i16m8;
            // Column 0 (left boundary). Local: zero. Global linear: `gap * query_position`. Global affine: the
            // all-deletion cell `open + extend * (query_position - 1)`; the insertion track resets to the sentinel.
            if constexpr (is_local_k) {
                __riscv_vse16_v_i16m8(current_row, zero_i16m8, vl);
                if constexpr (is_affine_k) {
                    __riscv_vse16_v_i16m8(current_deletes, sentinel_i16m8, vl);
                    running_inserts_i16m8 = sentinel_i16m8;
                }
            }
            else if constexpr (is_affine_k) {
                i16_t const row_boundary = static_cast<i16_t>(open + extend * (i16_t)(query_position - 1));
                __riscv_vse16_v_i16m8(current_row, __riscv_vmv_v_x_i16m8(row_boundary, vl), vl);
                vint16m8_t const seed_i16m8 = __riscv_vmv_v_x_i16m8(static_cast<i16_t>(open + extend + row_boundary),
                                                                    vl);
                __riscv_vse16_v_i16m8(current_deletes, seed_i16m8, vl);
                running_inserts_i16m8 = seed_i16m8;
            }
            else {
                __riscv_vse16_v_i16m8(current_row,
                                      __riscv_vmv_v_x_i16m8(static_cast<i16_t>(open * (i16_t)query_position), vl), vl);
            }

            for (size_t column = 1; column <= longest_candidate; ++column) {
                vuint8m4_t const candidate_classes_u8m4 = __riscv_vle8_v_u8m4(
                    candidate_classes + (column - 1) * candidate_lanes_k, vl);
                vint16m8_t const diagonal_i16m8 = __riscv_vle16_v_i16m8(previous_row + (column - 1) * candidate_lanes_k,
                                                                        vl);
                vint16m8_t const up_i16m8 = __riscv_vle16_v_i16m8(previous_row + column * candidate_lanes_k, vl);
                vint16m8_t const left_i16m8 = __riscv_vle16_v_i16m8(current_row + (column - 1) * candidate_lanes_k, vl);

                vint16m8_t cost_if_gap_i16m8;
                if constexpr (is_affine_k) {
                    vint16m8_t const up_deletes_i16m8 = __riscv_vle16_v_i16m8(
                        previous_deletes + column * candidate_lanes_k, vl);
                    // Gotoh deletion track (vertical): open a gap from the cell above, or extend the running deletion.
                    vint16m8_t const delete_i16m8 = __riscv_vmax_vv_i16m8(
                        __riscv_vadd_vv_i16m8(up_i16m8, open_i16m8, vl),
                        __riscv_vadd_vv_i16m8(up_deletes_i16m8, extend_i16m8, vl), vl);
                    // Gotoh insertion track (horizontal): open a gap from the cell to the left, or extend the running one.
                    vint16m8_t const insert_i16m8 = __riscv_vmax_vv_i16m8(
                        __riscv_vadd_vv_i16m8(left_i16m8, open_i16m8, vl),
                        __riscv_vadd_vv_i16m8(running_inserts_i16m8, extend_i16m8, vl), vl);
                    running_inserts_i16m8 = insert_i16m8;
                    __riscv_vse16_v_i16m8(current_deletes + column * candidate_lanes_k, delete_i16m8, vl);
                    cost_if_gap_i16m8 = __riscv_vmax_vv_i16m8(delete_i16m8, insert_i16m8, vl);
                }
                else {
                    cost_if_gap_i16m8 = __riscv_vadd_vv_i16m8(__riscv_vmax_vv_i16m8(up_i16m8, left_i16m8, vl),
                                                              gap_i16m8, vl);
                }

                // Look up each lane's cost with one in-register `vrgather_vv` keyed on the per-lane candidate class
                // directly (the hoisted row is already class-indexed 0..31; no widen/add/memory-gather needed).
                vuint8m4_t const cost_u8m4 = __riscv_vrgather_vv_u8m4(cost_row_u8m4, candidate_classes_u8m4, vl);
                vint16m8_t const cost_i16m8 = __riscv_vsext_vf2_i16m8(__riscv_vreinterpret_v_u8m4_i8m4(cost_u8m4), vl);

                vint16m8_t const cost_if_substitution_i16m8 = __riscv_vadd_vv_i16m8(diagonal_i16m8, cost_i16m8, vl);
                vint16m8_t cell_score_i16m8 = __riscv_vmax_vv_i16m8(cost_if_substitution_i16m8, cost_if_gap_i16m8, vl);
                if constexpr (is_local_k) cell_score_i16m8 = __riscv_vmax_vx_i16m8(cell_score_i16m8, 0, vl);
                __riscv_vse16_v_i16m8(current_row + column * candidate_lanes_k, cell_score_i16m8, vl);

                if constexpr (is_local_k) {
                    // Fold into the running maximum only on lanes whose candidate reaches this column.
                    vbool2_t const column_live_b2 = __riscv_vmsgt_vx_i16m8_b2(lane_lengths_i16m8, (i16_t)(column - 1),
                                                                              vl);
                    vint16m8_t const candidate_max_i16m8 = __riscv_vmax_vv_i16m8(running_max_i16m8, cell_score_i16m8,
                                                                                 vl);
                    running_max_i16m8 = __riscv_vmerge_vvm_i16m8(running_max_i16m8, candidate_max_i16m8, column_live_b2,
                                                                 vl);
                }
            }
            trivial_swap(previous_row, current_row);
            if constexpr (is_affine_k) trivial_swap(previous_deletes, current_deletes);
        }

        if constexpr (is_local_k) {
            i16_t final_max[candidate_lanes_k];
            __riscv_vse16_v_i16m8(final_max, running_max_i16m8, vl);
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
 *  @brief Inter-sequence RVV walker: one query against up to 32 candidates packed one-per-lane, weighted
 *      (32 x 32) substitution Needleman-Wunsch / Smith-Waterman with signed 32-bit cells.
 *
 *  Width-mirror of the 64-lane `i16` weighted walker above, widened to `i32` cells so one `__riscv_vsetvl_e32m8`
 *  call binds 32 lanes at the smallest tested `VLEN == 128`. Halving the lane count quadruples the representable
 *  score range to the signed `i32` window, serving candidates whose reachable scores exceed the `i16` contract.
 *  Linear and affine gaps, global and local, all share this body exactly as the `i16` sibling does.
 *
 *  @note Signed `i32` cells; valid while every reachable score stays within the caller's contracted `i32` range.
 *  @note Requires the RVV 1.0 Vector extension.
 */
template <typename gap_costs_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_,
          sz_capability_t capability_>
struct candidate_lane_walker<char, i32_t, error_costs_32x32_t, gap_costs_type_, objective_, locality_, capability_, 32,
                             std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = char;
    using score_t = i32_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 32;
    static constexpr bool is_affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    static_assert(
        objective_ == sz_maximize_score_k,
        "The weighted candidate-lane kernel only implements score " "maximization (Needleman-Wunsch / " "Smith-" "Water" "man)" ".");

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds the score rows of `longest_candidate + 1` lane-vectors (32 `i32` cells each), plus one
     *      cached candidate-class lane-vector (32 `u8`) per candidate position; affine adds two deletion-track rows. */
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
        size_t const vl = __riscv_vsetvl_e32m8(candidate_lanes_k);

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

        u8_t const *const byte_to_class = substituter_.byte_to_class;
        i8_t cost_matrix_1024[error_costs_classes_count_k * error_costs_classes_count_k];
        for (size_t first_class = 0; first_class != error_costs_classes_count_k; ++first_class)
            for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                cost_matrix_1024[first_class * error_costs_classes_count_k + second_class] =
                    substituter_.class_substitution_costs[first_class][second_class];

        for (size_t column = 0; column < longest_candidate; ++column) {
            vuint8m2_t const candidate_bytes_u8m2 = __riscv_vle8_v_u8m2((u8_t const *)candidates.position(column), vl);
            vuint8m2_t const candidate_classes_u8m2 = __riscv_vluxei8_v_u8m2(byte_to_class, candidate_bytes_u8m2, vl);
            __riscv_vse8_v_u8m2(candidate_classes + column * candidate_lanes_k, candidate_classes_u8m2, vl);
        }

        vint32m8_t const zero_i32m8 = __riscv_vmv_v_x_i32m8(0, vl);
        [[maybe_unused]] vint32m8_t gap_i32m8 {};
        [[maybe_unused]] vint32m8_t open_i32m8 {};
        [[maybe_unused]] vint32m8_t extend_i32m8 {};
        error_cost_t open = 0;
        error_cost_t extend = 0;
        if constexpr (is_affine_k) {
            open = gap_costs_.open;
            extend = gap_costs_.extend;
            open_i32m8 = __riscv_vmv_v_x_i32m8(static_cast<i32_t>(open), vl);
            extend_i32m8 = __riscv_vmv_v_x_i32m8(static_cast<i32_t>(extend), vl);
        }
        else {
            open = gap_costs_.open_or_extend;
            gap_i32m8 = __riscv_vmv_v_x_i32m8(static_cast<i32_t>(open), vl);
        }
        [[maybe_unused]] vint32m8_t const sentinel_i32m8 = __riscv_vmv_v_x_i32m8(static_cast<i32_t>(open + extend), vl);

        [[maybe_unused]] i32_t lane_lengths[candidate_lanes_k] = {0};
        [[maybe_unused]] vint32m8_t lane_lengths_i32m8 = zero_i32m8;
        [[maybe_unused]] vint32m8_t running_max_i32m8 = zero_i32m8;
        if constexpr (is_local_k) {
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                lane_lengths[lane_index] = static_cast<i32_t>(candidates.lengths[lane_index]);
            lane_lengths_i32m8 = __riscv_vle32_v_i32m8(lane_lengths, vl);
        }

        for (size_t column = 0; column <= longest_candidate; ++column) {
            if constexpr (is_local_k) {
                __riscv_vse32_v_i32m8(previous_row + column * candidate_lanes_k, zero_i32m8, vl);
                if constexpr (is_affine_k)
                    __riscv_vse32_v_i32m8(previous_deletes + column * candidate_lanes_k, sentinel_i32m8, vl);
            }
            else if constexpr (is_affine_k) {
                i32_t const boundary = static_cast<i32_t>(column ? open + extend * (i32_t)(column - 1) : 0);
                __riscv_vse32_v_i32m8(previous_row + column * candidate_lanes_k, __riscv_vmv_v_x_i32m8(boundary, vl),
                                      vl);
                __riscv_vse32_v_i32m8(previous_deletes + column * candidate_lanes_k,
                                      __riscv_vmv_v_x_i32m8(static_cast<i32_t>(open + extend + boundary), vl), vl);
            }
            else {
                __riscv_vse32_v_i32m8(previous_row + column * candidate_lanes_k,
                                      __riscv_vmv_v_x_i32m8(static_cast<i32_t>(open * (i32_t)column), vl), vl);
            }
        }

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            u8_t const query_class = byte_to_class[(u8_t)query[query_position - 1]];
            // Hoisted once per row: the 32-byte class-cost row for this `query_class`, resident in a register group
            // sized to match `candidate_classes_u8m2` so every column folds it with one in-register `vrgather_vv`
            // instead of re-deriving the combined table index and re-gathering from memory per cell.
            vuint8m2_t const cost_row_u8m2 = __riscv_vle8_v_u8m2(
                reinterpret_cast<u8_t const *>(cost_matrix_1024 + (size_t)query_class * error_costs_classes_count_k),
                error_costs_classes_count_k);

            [[maybe_unused]] vint32m8_t running_inserts_i32m8 = zero_i32m8;
            if constexpr (is_local_k) {
                __riscv_vse32_v_i32m8(current_row, zero_i32m8, vl);
                if constexpr (is_affine_k) {
                    __riscv_vse32_v_i32m8(current_deletes, sentinel_i32m8, vl);
                    running_inserts_i32m8 = sentinel_i32m8;
                }
            }
            else if constexpr (is_affine_k) {
                i32_t const row_boundary = static_cast<i32_t>(open + extend * (i32_t)(query_position - 1));
                __riscv_vse32_v_i32m8(current_row, __riscv_vmv_v_x_i32m8(row_boundary, vl), vl);
                vint32m8_t const seed_i32m8 = __riscv_vmv_v_x_i32m8(static_cast<i32_t>(open + extend + row_boundary),
                                                                    vl);
                __riscv_vse32_v_i32m8(current_deletes, seed_i32m8, vl);
                running_inserts_i32m8 = seed_i32m8;
            }
            else {
                __riscv_vse32_v_i32m8(current_row,
                                      __riscv_vmv_v_x_i32m8(static_cast<i32_t>(open * (i32_t)query_position), vl), vl);
            }

            for (size_t column = 1; column <= longest_candidate; ++column) {
                vuint8m2_t const candidate_classes_u8m2 = __riscv_vle8_v_u8m2(
                    candidate_classes + (column - 1) * candidate_lanes_k, vl);
                vint32m8_t const diagonal_i32m8 = __riscv_vle32_v_i32m8(previous_row + (column - 1) * candidate_lanes_k,
                                                                        vl);
                vint32m8_t const up_i32m8 = __riscv_vle32_v_i32m8(previous_row + column * candidate_lanes_k, vl);
                vint32m8_t const left_i32m8 = __riscv_vle32_v_i32m8(current_row + (column - 1) * candidate_lanes_k, vl);

                vint32m8_t cost_if_gap_i32m8;
                if constexpr (is_affine_k) {
                    vint32m8_t const up_deletes_i32m8 = __riscv_vle32_v_i32m8(
                        previous_deletes + column * candidate_lanes_k, vl);
                    vint32m8_t const delete_i32m8 = __riscv_vmax_vv_i32m8(
                        __riscv_vadd_vv_i32m8(up_i32m8, open_i32m8, vl),
                        __riscv_vadd_vv_i32m8(up_deletes_i32m8, extend_i32m8, vl), vl);
                    vint32m8_t const insert_i32m8 = __riscv_vmax_vv_i32m8(
                        __riscv_vadd_vv_i32m8(left_i32m8, open_i32m8, vl),
                        __riscv_vadd_vv_i32m8(running_inserts_i32m8, extend_i32m8, vl), vl);
                    running_inserts_i32m8 = insert_i32m8;
                    __riscv_vse32_v_i32m8(current_deletes + column * candidate_lanes_k, delete_i32m8, vl);
                    cost_if_gap_i32m8 = __riscv_vmax_vv_i32m8(delete_i32m8, insert_i32m8, vl);
                }
                else {
                    cost_if_gap_i32m8 = __riscv_vadd_vv_i32m8(__riscv_vmax_vv_i32m8(up_i32m8, left_i32m8, vl),
                                                              gap_i32m8, vl);
                }

                // Look up each lane's cost with one in-register `vrgather_vv` keyed on the per-lane candidate class
                // directly (the hoisted row is already class-indexed 0..31; no widen/add/memory-gather needed).
                vuint8m2_t const cost_u8m2 = __riscv_vrgather_vv_u8m2(cost_row_u8m2, candidate_classes_u8m2, vl);
                vint32m8_t const cost_i32m8 = __riscv_vsext_vf4_i32m8(__riscv_vreinterpret_v_u8m2_i8m2(cost_u8m2), vl);

                vint32m8_t const cost_if_substitution_i32m8 = __riscv_vadd_vv_i32m8(diagonal_i32m8, cost_i32m8, vl);
                vint32m8_t cell_score_i32m8 = __riscv_vmax_vv_i32m8(cost_if_substitution_i32m8, cost_if_gap_i32m8, vl);
                if constexpr (is_local_k) cell_score_i32m8 = __riscv_vmax_vx_i32m8(cell_score_i32m8, 0, vl);
                __riscv_vse32_v_i32m8(current_row + column * candidate_lanes_k, cell_score_i32m8, vl);

                if constexpr (is_local_k) {
                    vbool4_t const column_live_b4 = __riscv_vmsgt_vx_i32m8_b4(lane_lengths_i32m8, (i32_t)(column - 1),
                                                                              vl);
                    vint32m8_t const candidate_max_i32m8 = __riscv_vmax_vv_i32m8(running_max_i32m8, cell_score_i32m8,
                                                                                 vl);
                    running_max_i32m8 = __riscv_vmerge_vvm_i32m8(running_max_i32m8, candidate_max_i32m8, column_live_b4,
                                                                 vl);
                }
            }
            trivial_swap(previous_row, current_row);
            if constexpr (is_affine_k) trivial_swap(previous_deletes, current_deletes);
        }

        if constexpr (is_local_k) {
            i32_t final_max[candidate_lanes_k];
            __riscv_vse32_v_i32m8(final_max, running_max_i32m8, vl);
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
                result_lanes[lane_index] = final_max[lane_index];
        }
        else {
            for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
                size_t const candidate_length = candidates.lengths[lane_index];
                result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
            }
        }
        return status_t::success_k;
    }
};

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
        // Both the single-word and multi-word lane scans also keep a `max_longer * lanes_k` transposed-text buffer
        // right after the `match_masks` table (the longest possible pair "longer" side bounds it across the group).
        size_t const longest_longer = sz_max_of_two(longest_query, longest_candidate);
        size_t const transposed_text_scratch = longest_longer * myers_t::lanes_k;
        size_t const myers_scratch = myers_t::match_masks_bytes_k + transposed_text_scratch;
        size_t dp_scratch = 0, multiword_scratch = 0;
        size_t const shortest_longest = sz_min_of_two(longest_query, longest_candidate);
        if (queries.size() && candidates.size() && shortest_longest > 64) {
            size_t const words_bound = divide_round_up<size_t>(shortest_longest, 64);
            multiword_scratch = myers_t::match_masks_bytes_k * words_bound + transposed_text_scratch;
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

    /** @brief Scores the cross-product in parallel: uneven per-cell costs ride the work-stealing scheduler. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    SZ_NOINLINE status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                         results_type_ &&results, cross_similarities_t cross_kind,
                                         executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        size_t const worker_scratch = worst_cell_scratch_(queries, candidates, specs);
        size_t const workers = sz_max_of_two(executor.threads_count(), (size_t)1);
        if (status_t status = score_scratch_.try_resize(worker_scratch * workers); status != status_t::success_k)
            return status;
        using prong_t = typename remove_cvref<executor_type_>::prong_t;
        std::atomic<status_t> error {status_t::success_k};
        executor.for_n_dynamic(cells_count, [&](prong_t prong) noexcept {
            scratch_space_t slice =
                scratch_space_t(score_scratch_).subspan(prong.thread * worker_scratch, worker_scratch);
            status_t status =
                score_range_(queries, candidates, results, cross_kind, prong.task, prong.task + 1, slice, specs);
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

    /** @brief Scores the cross-product in parallel: uneven per-cell costs ride the work-stealing scheduler. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    SZ_NOINLINE status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                         results_type_ &&results, cross_similarities_t cross_kind,
                                         executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        size_t const worker_scratch = worst_cell_scratch_(queries, candidates, specs);
        size_t const workers = sz_max_of_two(executor.threads_count(), (size_t)1);
        if (status_t status = score_scratch_.try_resize(worker_scratch * workers); status != status_t::success_k)
            return status;
        using prong_t = typename remove_cvref<executor_type_>::prong_t;
        std::atomic<status_t> error {status_t::success_k};
        executor.for_n_dynamic(cells_count, [&](prong_t prong) noexcept {
            scratch_space_t slice =
                scratch_space_t(score_scratch_).subspan(prong.thread * worker_scratch, worker_scratch);
            status_t status =
                score_range_(queries, candidates, results, cross_kind, prong.task, prong.task + 1, slice, specs);
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

#pragma region Class Cost Cross Product

/**
 *  @brief Batched byte-level @b weighted Needleman-Wunsch scores on RVV, packing up to 64 candidates of a shared
 *      query into the signed-`i16` `candidate_lane_walker` (32 candidates into the `i32` one for wider ranges) and
 *      falling back to the per-pair anti-diagonal Dynamic Programming scorer for the long tail (scores that escape
 *      `i16`/`i32`, or empty cells).
 *
 *  Mirrors the NEON / Ice Lake `needleman_wunsch_scores` cross-product engines, widened to the vector-length-agnostic
 *      64-/32-lane RVV walkers.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
                               std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 64;
    static constexpr ssize_t score_range_limit_k = 30000;           // ? `i16` headroom for the narrow lane walker.
    static constexpr ssize_t score_range_limit_wide_k = 2000000000; // ? `i32` headroom for the wide lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, capability_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              capability_k, (int)candidate_lanes_k, void>; // ? RVV shared-query `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              capability_k, 32, void>; // ? RVV shared-query `i32` lanes.

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
 *  @brief Batched byte-level @b weighted Smith-Waterman (local) scores on RVV, packing up to 64 candidates of a
 *      shared query into the signed-`i16` local `candidate_lane_walker` (32 into the `i32` one) and falling back to
 *      the per-pair anti-diagonal Dynamic Programming scorer for the long tail (scores that escape `i16`/`i32`).
 *
 *  Local sibling of the RVV `needleman_wunsch_scores` cross-product engine: identical query-major grouping,
 *      transpose, and scatter, but it dispatches the @b local lane walker (zero-clamped recurrence, score = the
 *      maximum cell of the matrix) and an empty `(query, candidate)` cell scores @b 0 (a local alignment may align
 *      nothing).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 64;
    static constexpr ssize_t score_range_limit_k = 30000;           // ? `i16` headroom for the narrow lane walker.
    static constexpr ssize_t score_range_limit_wide_k = 2000000000; // ? `i32` headroom for the wide lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, capability_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              capability_k, (int)candidate_lanes_k, void>; // ? RVV shared-query local `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              capability_k, 32, void>; // ? RVV shared-query local `i32` lanes.

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
 *  @brief Batched byte-level @b weighted Needleman-Wunsch scores on RVV with an @b affine gap, packing up to 64
 *      candidates of a shared query into the signed-`i16` affine `candidate_lane_walker` (32 into the `i32` one) and
 *      falling back to the per-pair anti-diagonal Dynamic Programming scorer for the long tail.
 *
 *  Affine sibling of the RVV linear-gap `needleman_wunsch_scores` cross-product engine: identical query-major
 *      grouping, transpose, and scatter, but it dispatches the affine (Gotoh E/F) lane walker, and an empty
 *      `(query, candidate)` cell scores `open + extend * (other_length - 1)` rather than `gap * other_length`.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, allocator_type_, capability_,
                               std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 64;
    static constexpr ssize_t score_range_limit_k = 30000;           // ? `i16` headroom for the narrow lane walker.
    static constexpr ssize_t score_range_limit_wide_k = 2000000000; // ? `i32` headroom for the wide lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, capability_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              capability_k, (int)candidate_lanes_k, void>; // ? RVV shared-query `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              capability_k, 32, void>; // ? RVV shared-query `i32` lanes.

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
 *  @brief Batched byte-level @b weighted Smith-Waterman (local) scores on RVV with an @b affine gap, packing up to 64
 *      candidates of a shared query into the signed-`i16` local affine `candidate_lane_walker` (32 into the `i32`
 *      one) and falling back to the per-pair anti-diagonal Dynamic Programming scorer for the long tail.
 *
 *  Local sibling of the RVV affine `needleman_wunsch_scores` cross-product engine: identical query-major grouping,
 *      transpose, and scatter, but it dispatches the @b local affine lane walker (zero-clamped Gotoh recurrence,
 *      score = the maximum cell of the matrix) and an empty `(query, candidate)` cell scores @b 0 (a local alignment
 *      may align nothing).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 64;
    static constexpr ssize_t score_range_limit_k = 30000;           // ? `i16` headroom for the narrow lane walker.
    static constexpr ssize_t score_range_limit_wide_k = 2000000000; // ? `i32` headroom for the wide lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, capability_k>; // ? Per-pair DP fallback.
    using lane_walker_narrow_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              capability_k, (int)candidate_lanes_k, void>; // ? RVV shared-query local `i16` lanes.
    using lane_walker_wide_t =
        candidate_lane_walker<char, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              capability_k, 32, void>; // ? RVV shared-query local `i32` lanes.

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

#pragma endregion Class Cost Cross Product

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
