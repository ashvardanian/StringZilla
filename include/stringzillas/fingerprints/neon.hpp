/**
 *  @brief Arm NEON (AArch64) rolling-hash fingerprint backend.
 *  @file include/stringzillas/fingerprints/neon.hpp
 *  @author Ash Vardanian
 *  @sa include/stringzillas/fingerprints/serial.hpp
 *  @sa include/stringzillas/fingerprints/skylake.hpp
 */
#ifndef STRINGZILLAS_FINGERPRINTS_NEON_HPP_
#define STRINGZILLAS_FINGERPRINTS_NEON_HPP_

#include "stringzillas/fingerprints/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

/*  NEON implementation of the floating-point rolling Min-Hashers for 64-bit Arm CPUs.
 *  Mirrors the Skylake backend's fused single-reduction roll, but a 128-bit Q register holds only
 *  two `f64_t` lanes, so it processes two hashes per register. Two NEON advantages over AVX-512
 *  show up here: `vrndmq_f64` is a real floor instruction, so the magic-number floor hack is gone,
 *  and the masked min/count updates use native compare masks + `vbslq` instead of mask registers.
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
 *  @brief Optimized rolling Min-Hashers built around floating-point numbers.
 *  In a single Q register we can store 2 `f64_t` values, so we can process 2 hashes per register.
 */
template <size_t dimensions_>
struct floating_rolling_hashers<sz_cap_neon_k, dimensions_, void> {

    using hasher_t = floating_rolling_hasher<f64_t>;
    using rolling_state_t = f64_t;
    using min_hash_t = u32_t;
    using min_count_t = u32_t;

    static constexpr size_t dimensions_k = dimensions_;
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr rolling_state_t skipped_rolling_hash_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t, dimensions_k>;
    using min_counts_span_t = span<min_count_t, dimensions_k>;

    static constexpr unsigned hashes_per_qreg_k = sizeof(u128_vec_t) / sizeof(rolling_state_t);
    static constexpr bool has_incomplete_tail_group_k = (dimensions_k % hashes_per_qreg_k) != 0;
    static constexpr size_t aligned_dimensions_k = has_incomplete_tail_group_k
                                                       ? (dimensions_k / hashes_per_qreg_k + 1) * hashes_per_qreg_k
                                                       : (dimensions_k);
    static constexpr unsigned groups_count_k = aligned_dimensions_k / hashes_per_qreg_k;

    // How many independent groups (Q registers) to interleave per pass. More groups hide more of the
    // Barrett dependency latency on the M5's wide FP units; 16 was the throughput sweet spot. Pick the
    // largest clean divisor up to 16 so the passes stay even (the `<1>` remainder loop is the slow path).
    static constexpr unsigned groups_per_pass_k = groups_count_k % 16 == 0  ? 16
                                                  : groups_count_k % 8 == 0 ? 8
                                                  : groups_count_k % 4 == 0 ? 4
                                                  : groups_count_k % 2 == 0 ? 2
                                                                            : 1;

    static_assert(dimensions_k <= 256, "Too many dimensions to keep on stack");

  private:
    rolling_state_t multipliers_[aligned_dimensions_k];
    rolling_state_t modulos_[aligned_dimensions_k];
    rolling_state_t inverse_modulos_[aligned_dimensions_k];
    rolling_state_t discarding_multipliers_[aligned_dimensions_k];
    size_t window_width_ = 0;

  public:
    constexpr size_t dimensions() const noexcept { return dimensions_k; }
    constexpr size_t window_width() const noexcept { return window_width_; }
    constexpr size_t window_width(size_t) const noexcept { return window_width_; }

    floating_rolling_hashers() noexcept {
        // Reset all variables to zeros
        for (auto &multiplier : multipliers_) multiplier = 0.0;
        for (auto &modulo : modulos_) modulo = 0.0;
        for (auto &inverse_modulo : inverse_modulos_) inverse_modulo = 0.0;
        for (auto &discarding_multiplier : discarding_multipliers_) discarding_multiplier = 0.0;
        window_width_ = 0;
    }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     *  @param[in] first_dimension_offset The offset for the first dimension within a larger fingerprint, typically 0.
     *  @param[in] seed Reproducibility seed; every value derives independent per-dimension multipliers and moduli.
     */
    SZ_NOINLINE status_t try_seed(size_t window_width, size_t alphabet_size = 256, size_t first_dimension_offset = 0,
                                  u64_t seed = default_seed_k) noexcept {
        for (size_t dim = 0; dim < dimensions_k; ++dim) {
            hasher_t hasher(window_width, alphabet_size, first_dimension_offset + dim, seed);
            multipliers_[dim] = hasher.multiplier();
            modulos_[dim] = hasher.modulo();
            inverse_modulos_[dim] = hasher.inverse_modulo();
            discarding_multipliers_[dim] = hasher.discarding_multiplier();
        }
        window_width_ = window_width;
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    SZ_NOINLINE void fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                                 min_counts_span_t min_counts) const noexcept {

        if (text.size() < window_width_) {
            for (auto &min_hash : min_hashes) min_hash = max_hash_k;
            for (auto &min_count : min_counts) min_count = 0;
            return;
        }

        rolling_state_t rolling_states[dimensions_k];
        rolling_state_t rolling_minimums[dimensions_k];
        for (size_t dim = 0; dim < dimensions_k; ++dim)
            rolling_states[dim] = 0, rolling_minimums[dim] = skipped_rolling_hash_k;
        fingerprint_chunk(text, &rolling_states[0], &rolling_minimums[0], min_hashes, min_counts);
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    SZ_NOINLINE status_t try_fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                                         min_counts_span_t min_counts) const noexcept {
        fingerprint(text, min_hashes, min_counts);
        return status_t::success_k;
    }

    /**
     *  @brief Underlying machinery of `fingerprint` that fills the states of the hashers.
     *  @param[in] text_chunk A chunk of text to update the @p `last_states` with.
     *  @param[inout] last_states The last computed floats for each hasher; start with @b zeroes.
     *  @param[inout] rolling_minimums The minimum floats for each hasher; start with @b `skipped_rolling_hash_k`.
     *  @param[out] min_hashes The @b optional output for minimum hashes, which are the final fingerprints.
     *  @param[out] min_counts The frequencies of @p `rolling_minimums` and optional @p `min_hashes` hashes.
     *  @param[in] passed_progress The offset of the received @p `text_chunk` in the whole text; defaults to 0.
     */
    SZ_NOINLINE void fingerprint_chunk(                       //
        span<byte_t const> text_chunk,                        //
        span<rolling_state_t, dimensions_k> last_states,      //
        span<rolling_state_t, dimensions_k> rolling_minimums, //
        min_hashes_span_t min_hashes,                         //
        min_counts_span_t min_counts,                         //
        size_t const passed_progress = 0) const noexcept {

        // Each group is a latency-bound chain of Barrett reductions, but groups are independent, so
        // several are interleaved per pass to keep the M5's FP pipes busy and hide that latency. The
        // last group is handled on its own when it is an incomplete (masked) tail.
        unsigned const complete_groups = groups_count_k - (has_incomplete_tail_group_k ? 1u : 0u);
        unsigned group_index = 0;
        for (; group_index + groups_per_pass_k <= complete_groups; group_index += groups_per_pass_k)
            roll_groups<groups_per_pass_k>(text_chunk, group_index, last_states, rolling_minimums, min_counts,
                                           passed_progress);
        for (; group_index < complete_groups; ++group_index)
            roll_groups<1>(text_chunk, group_index, last_states, rolling_minimums, min_counts, passed_progress);
        if (has_incomplete_tail_group_k)
            roll_tail_group(text_chunk, groups_count_k - 1, last_states, rolling_minimums, min_counts, passed_progress);

        // Finally, export the minimum hashes into the smaller representations
        if (min_hashes)
            for (size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t const &rolling_minimum = rolling_minimums[dim];
                min_hash_t &min_hash = min_hashes[dim];
                auto const rolling_minimum_as_uint = static_cast<u64_t>(rolling_minimum);
                min_hash = rolling_minimum == skipped_rolling_hash_k
                               ? max_hash_k // If the rolling minimum is not set, use the maximum hash value
                               : static_cast<min_hash_t>(rolling_minimum_as_uint & max_hash_k);
            }
    }

    /**
     *  @brief Computes many fingerprints in parallel for input @p texts via an @p executor.
     *  @param[in] texts The input texts to hash, typically a sequential container of UTF-8 encoded strings.
     *  @param[out] min_hashes_per_text The output fingerprints, an array of vectors of minimum hashes.
     *  @param[out] min_counts_per_text The output frequencies of @p `min_hashes_per_text` hashes.
     *  @param[in] executor The executor to use for parallel processing, defaults to a dummy executor.
     *  @param[in] specs The CPU specifications to use, defaults to an empty `cpu_specs_t`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <typename texts_type_, typename min_hashes_per_text_type_, typename min_counts_per_text_type_,
              typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    SZ_NOINLINE status_t operator()(texts_type_ const &texts, min_hashes_per_text_type_ &&min_hashes_per_text, //
                                    min_counts_per_text_type_ &&min_counts_per_text, executor_type_ &&executor = {},
                                    cpu_specs_t specs = {}) noexcept {
        return floating_rolling_hashers_in_parallel_(                     //
            *this, texts,                                                 //
            std::forward<min_hashes_per_text_type_>(min_hashes_per_text), //
            std::forward<min_counts_per_text_type_>(min_counts_per_text), //
            std::forward<executor_type_>(executor), specs);
    }

  private:
    SZ_INLINE float64x2_t barrett_mod(float64x2_t xs, float64x2_t modulos, float64x2_t inverse_modulos) const noexcept {

        // `vrndmq_f64` rounds toward minus infinity, i.e. it is a real `floor`, so the magic-number
        // floor the x86 ports need disappears here.
        float64x2_t qs = vrndmq_f64(vmulq_f64(xs, inverse_modulos));
        float64x2_t results = vfmsq_f64(xs, qs, modulos); // xs - qs * modulos

        // `vrndmq` is a native floor, so only the high-side fixup is kept (r ≥ modulo → r -= modulo).
        uint64x2_t overflow_mask = vcgeq_f64(results, modulos);
        results = vsubq_f64(results, vreinterpretq_f64_u64(vandq_u64(overflow_mask, vreinterpretq_u64_f64(modulos))));
        // `r < 0` fixup omitted: dead for the rolling-hash range (x < limit_k = 2^52).

#if SZ_DEBUG
        sz_u128_vec_t xs_vec, modulos_vec, results_vec;
        xs_vec.f64x2 = xs, modulos_vec.f64x2 = modulos, results_vec.f64x2 = results;
        sz_assert_(modulos_vec.f64s[0] == 0 ||
                   absolute_umod(xs_vec.f64s[0], modulos_vec.f64s[0]) == static_cast<u64_t>(results_vec.f64s[0]));
        sz_assert_(modulos_vec.f64s[1] == 0 ||
                   absolute_umod(xs_vec.f64s[1], modulos_vec.f64s[1]) == static_cast<u64_t>(results_vec.f64s[1]));
#endif

        return results;
    }

    /**
     *  @brief Rolls @p passes_ independent, complete Q-register groups through @p text_chunk at once.
     *  Each group owns its own Barrett chain; interleaving `passes_` of them hides that latency on the
     *  wide M5 FP units. `passes_` is a compile-time constant so the `for (pass)` loops fully unroll.
     */
    template <unsigned passes_>
    void roll_groups(                                              //
        span<byte_t const> text_chunk, unsigned const first_group, //
        span<rolling_state_t, dimensions_k> last_states,           //
        span<rolling_state_t, dimensions_k> rolling_minimums,      //
        span<min_count_t, dimensions_k> rolling_counts,            //
        size_t const passed_progress = 0) const noexcept {

        unsigned first_dims[passes_];
        float64x2_t states[passes_], minimums[passes_];
        uint64x2_t counts[passes_];
        float64x2_t multipliers[passes_], discarding[passes_], modulos[passes_], inverse_modulos[passes_];
        for (unsigned pass = 0; pass < passes_; ++pass) {
            unsigned const first_dim = (first_group + pass) * hashes_per_qreg_k;
            first_dims[pass] = first_dim;
            states[pass] = vld1q_f64(&last_states[first_dim]);
            minimums[pass] = vld1q_f64(&rolling_minimums[first_dim]);
            counts[pass] = vmovl_u32(vld1_u32(&rolling_counts[first_dim]));
            multipliers[pass] = vld1q_f64(&multipliers_[first_dim]);
            discarding[pass] = vld1q_f64(&discarding_multipliers_[first_dim]);
            modulos[pass] = vld1q_f64(&modulos_[first_dim]);
            inverse_modulos[pass] = vld1q_f64(&inverse_modulos_[first_dim]);
        }

        // Until we reach the `window_width_`, we don't need to discard any symbols
        size_t const prefix_length = (std::min)(text_chunk.size(), window_width_);
        size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            float64x2_t new_term_vec = vdupq_n_f64(static_cast<rolling_state_t>(text_chunk[new_char_offset]) + 1.0);
            for (unsigned pass = 0; pass < passes_; ++pass) {
                states[pass] = vfmaq_f64(new_term_vec, states[pass], multipliers[pass]);
                states[pass] = barrett_mod(states[pass], modulos[pass], inverse_modulos[pass]);
            }
        }

        // We now have our first minimum hashes
        uint64x2_t const ones_vec = vdupq_n_u64(1);
        if (new_char_offset == window_width_ && passed_progress < prefix_length)
            for (unsigned pass = 0; pass < passes_; ++pass) minimums[pass] = states[pass], counts[pass] = ones_vec;

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            float64x2_t new_term_vec = vdupq_n_f64(static_cast<rolling_state_t>(text_chunk[new_char_offset]) + 1.0);
            float64x2_t old_term_vec = vdupq_n_f64(
                static_cast<rolling_state_t>(text_chunk[new_char_offset - window_width_]) + 1.0);

            for (unsigned pass = 0; pass < passes_; ++pass) {
                // A single Barrett reduction handles both the discarded head and the incoming tail symbol.
                // `discarding_multipliers_` folds in the `multipliers_` and is non-negative, so the fused
                // intermediate stays within `[0, 2⁵²)` without underflowing zero before the reduction.
                states[pass] = vfmaq_f64(new_term_vec, states[pass], multipliers[pass]);
                states[pass] = vfmaq_f64(states[pass], discarding[pass], old_term_vec);
                states[pass] = barrett_mod(states[pass], modulos[pass], inverse_modulos[pass]);

                // To keep the right comparison mask, check out: https://stackoverflow.com/q/16988199
                uint64x2_t found_mask = vcleq_f64(states[pass], minimums[pass]);
                uint64x2_t discard_mask = vcgeq_f64(states[pass], minimums[pass]);
                minimums[pass] = vbslq_f64(found_mask, states[pass], minimums[pass]);

                // Branchless counts: zero them when a new strict minimum appears, then increment new & existing
                counts[pass] = vandq_u64(counts[pass], discard_mask);
                counts[pass] = vbslq_u64(found_mask, vaddq_u64(counts[pass], ones_vec), counts[pass]);
            }
        }

        for (unsigned pass = 0; pass < passes_; ++pass) {
            vst1q_f64(&last_states[first_dims[pass]], states[pass]);
            vst1q_f64(&rolling_minimums[first_dims[pass]], minimums[pass]);
            vst1_u32(&rolling_counts[first_dims[pass]], vmovn_u64(counts[pass]));
        }
    }

    /**
     *  @brief Rolls the final incomplete group, loading/storing only its `dimensions_k % 2` live lanes
     *  through a scalar staging buffer so the partial Q register never touches memory past the spans.
     */
    void roll_tail_group(                                          //
        span<byte_t const> text_chunk, unsigned const group_index, //
        span<rolling_state_t, dimensions_k> last_states,           //
        span<rolling_state_t, dimensions_k> rolling_minimums,      //
        span<min_count_t, dimensions_k> rolling_counts,            //
        size_t const passed_progress = 0) const noexcept {

        unsigned const first_dim = group_index * hashes_per_qreg_k;
        sz_u128_vec_t last_states_vec, rolling_minimums_vec, rolling_counts_vec;
        for (size_t word_index = 0; word_index < (dimensions_k - first_dim); ++word_index) {
            last_states_vec.f64s[word_index] = last_states[first_dim + word_index];
            rolling_minimums_vec.f64s[word_index] = rolling_minimums[first_dim + word_index];
            rolling_counts_vec.u64s[word_index] = rolling_counts[first_dim + word_index];
        }

        float64x2_t multipliers_vec = vld1q_f64(&multipliers_[first_dim]);
        float64x2_t discarding_multipliers_vec = vld1q_f64(&discarding_multipliers_[first_dim]);
        float64x2_t modulos_vec = vld1q_f64(&modulos_[first_dim]);
        float64x2_t inverse_modulos_vec = vld1q_f64(&inverse_modulos_[first_dim]);

        size_t const prefix_length = (std::min)(text_chunk.size(), window_width_);
        size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            float64x2_t new_term_vec = vdupq_n_f64(static_cast<rolling_state_t>(text_chunk[new_char_offset]) + 1.0);
            last_states_vec.f64x2 = vfmaq_f64(new_term_vec, last_states_vec.f64x2, multipliers_vec);
            last_states_vec.f64x2 = barrett_mod(last_states_vec.f64x2, modulos_vec, inverse_modulos_vec);
        }

        uint64x2_t const ones_vec = vdupq_n_u64(1);
        if (new_char_offset == window_width_ && passed_progress < prefix_length)
            rolling_minimums_vec.f64x2 = last_states_vec.f64x2, rolling_counts_vec.u64x2 = ones_vec;

        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            float64x2_t new_term_vec = vdupq_n_f64(static_cast<rolling_state_t>(text_chunk[new_char_offset]) + 1.0);
            float64x2_t old_term_vec = vdupq_n_f64(
                static_cast<rolling_state_t>(text_chunk[new_char_offset - window_width_]) + 1.0);

            last_states_vec.f64x2 = vfmaq_f64(new_term_vec, last_states_vec.f64x2, multipliers_vec);
            last_states_vec.f64x2 = vfmaq_f64(last_states_vec.f64x2, discarding_multipliers_vec, old_term_vec);
            last_states_vec.f64x2 = barrett_mod(last_states_vec.f64x2, modulos_vec, inverse_modulos_vec);

            uint64x2_t found_mask = vcleq_f64(last_states_vec.f64x2, rolling_minimums_vec.f64x2);
            uint64x2_t discard_mask = vcgeq_f64(last_states_vec.f64x2, rolling_minimums_vec.f64x2);
            rolling_minimums_vec.f64x2 = vbslq_f64(found_mask, last_states_vec.f64x2, rolling_minimums_vec.f64x2);
            rolling_counts_vec.u64x2 = vandq_u64(rolling_counts_vec.u64x2, discard_mask);
            rolling_counts_vec.u64x2 = vbslq_u64(found_mask, vaddq_u64(rolling_counts_vec.u64x2, ones_vec),
                                                 rolling_counts_vec.u64x2);
        }

        for (size_t word_index = 0; word_index < (dimensions_k - first_dim); ++word_index) {
            last_states[first_dim + word_index] = last_states_vec.f64s[word_index];
            rolling_minimums[first_dim + word_index] = rolling_minimums_vec.f64s[word_index];
            rolling_counts[first_dim + word_index] = static_cast<min_count_t>(rolling_counts_vec.u64s[word_index]);
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#pragma endregion NEON Implementation

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FINGERPRINTS_NEON_HPP_
