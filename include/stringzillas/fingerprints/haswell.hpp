/**
 *  @brief AVX2 (Haswell) rolling-hash fingerprint backend.
 *  @file include/stringzillas/fingerprints/haswell.hpp
 *  @author Ash Vardanian
 *  @sa include/stringzillas/fingerprints/serial.hpp
 */
#ifndef STRINGZILLAS_FINGERPRINTS_HASWELL_HPP_
#define STRINGZILLAS_FINGERPRINTS_HASWELL_HPP_

#include "stringzillas/fingerprints/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

/*  AVX2 implementation of the string hashing algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#endif

SZ_INLINE __m256d _mm256_floor_magic_pd(__m256d x) noexcept {
    // Magic number rounding approach for fast floor
    __m256d magic = _mm256_set1_pd(6755399441055744.0); // 2^52 + 2^51
    __m256d rounded = _mm256_sub_pd(_mm256_add_pd(x, magic), magic);

    // Handle negative numbers: if result > x, subtract 1
    __m256d neg_mask_pd = _mm256_cmp_pd(rounded, x, _CMP_GT_OQ);
    return _mm256_sub_pd(rounded, _mm256_and_pd(neg_mask_pd, _mm256_set1_pd(1.0)));
}

/**
 *  @brief Optimized rolling Min-Hashers built around floating-point numbers.
 *  In a single YMM register we can store 4 `f64_t` values, so we can process 4 hashes per register.
 */
template <size_t dimensions_>
struct floating_rolling_hashers<sz_cap_haswell_k, dimensions_, void> {

    using hasher_t = floating_rolling_hasher<f64_t>;
    using rolling_state_t = f64_t;
    using min_hash_t = u32_t;
    using min_count_t = u32_t;

    static constexpr size_t dimensions_k = dimensions_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr rolling_state_t skipped_rolling_hash_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t, dimensions_k>;
    using min_counts_span_t = span<min_count_t, dimensions_k>;

    static constexpr unsigned hashes_per_ymm_k = sizeof(u256_vec_t) / sizeof(rolling_state_t);
    static constexpr bool has_incomplete_tail_group_k = (dimensions_k % hashes_per_ymm_k) != 0;
    static constexpr size_t aligned_dimensions_k = has_incomplete_tail_group_k
                                                       ? (dimensions_k / hashes_per_ymm_k + 1) * hashes_per_ymm_k
                                                       : (dimensions_k);
    static constexpr unsigned groups_count_k = aligned_dimensions_k / hashes_per_ymm_k;

    static_assert(dimensions_k <= 256, "Too many dimensions to keep on stack");

  private:
    rolling_state_t multipliers_[aligned_dimensions_k];
    rolling_state_t modulos_[aligned_dimensions_k];
    rolling_state_t inverse_modulos_[aligned_dimensions_k];
    rolling_state_t negative_discarding_multipliers_[aligned_dimensions_k];
    size_t window_width_;

  public:
    constexpr size_t dimensions() const noexcept { return dimensions_k; }
    constexpr size_t window_width() const noexcept { return window_width_; }
    constexpr size_t window_width(size_t) const noexcept { return window_width_; }

    floating_rolling_hashers() noexcept {
        // Reset all variables to zeros
        for (auto &multiplier : multipliers_) multiplier = 0.0;
        for (auto &modulo : modulos_) modulo = 0.0;
        for (auto &inverse_modulo : inverse_modulos_) inverse_modulo = 0.0;
        for (auto &negative_discarding_multiplier : negative_discarding_multipliers_)
            negative_discarding_multiplier = 0.0;
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
            negative_discarding_multipliers_[dim] = hasher.negative_discarding_multiplier();
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
     *
     *  Unlike the `fingerprint` method, this function can be used in a @b rolling fashion, i.e., it can be called
     *  multiple times with different chunks of text, and it will update the states accordingly. In the end, it
     *  will anyways export the composing Count-Min-Sketch fingerprint into the @p `min_hashes` and @p `min_counts`,
     *  as its a relatively cheap operation.
     */
    SZ_NOINLINE void fingerprint_chunk(                       //
        span<byte_t const> text_chunk,                        //
        span<rolling_state_t, dimensions_k> last_states,      //
        span<rolling_state_t, dimensions_k> rolling_minimums, //
        min_hashes_span_t min_hashes,                         //
        min_counts_span_t min_counts,                         //
        size_t passed_progress = 0                            //
    ) const noexcept {

        for (unsigned group_index = 0; group_index < groups_count_k; ++group_index)
            roll_group(text_chunk, group_index, last_states, rolling_minimums, min_counts, passed_progress);

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
    SZ_INLINE __m256d barrett_mod(__m256d xs, __m256d modulos, __m256d inverse_modulos) const noexcept {
        __m256d qs = _mm256_floor_magic_pd(_mm256_mul_pd(xs, inverse_modulos));
        __m256d results = _mm256_fnmadd_pd(qs, modulos, xs);

        // Clamp into the [0, modulo) range.
        __m256d overflow_mask_pd = _mm256_cmp_pd(results, modulos, _CMP_GE_OQ);
        results = _mm256_sub_pd(results, _mm256_and_pd(overflow_mask_pd, modulos));
        __m256d negative_mask_pd = _mm256_cmp_pd(results, _mm256_setzero_pd(), _CMP_LT_OQ);
        results = _mm256_add_pd(results, _mm256_and_pd(negative_mask_pd, modulos));

        return results;
    }

    void roll_group(                                               //
        span<byte_t const> text_chunk, unsigned const group_index, //
        span<rolling_state_t, dimensions_k> last_states,           //
        span<rolling_state_t, dimensions_k> rolling_minimums,      //
        span<min_count_t, dimensions_k> rolling_counts,            //
        size_t const passed_progress = 0) const noexcept {

        unsigned const first_dim = group_index * hashes_per_ymm_k;

        // Register space for in-out variables
        u256_vec_t last_states_vec;
        u256_vec_t rolling_minimums_vec;
        u256_vec_t rolling_counts_vec; // ? Unlike other cases, use larger 64-bit counters simplify masking

        // Use scalar loads for the incomplete tail group
        if (has_incomplete_tail_group_k && group_index + 1 == groups_count_k) {
            for (size_t word_index = 0; word_index < (dimensions_k - first_dim); ++word_index) {
                last_states_vec.f64s[word_index] = last_states[first_dim + word_index];
                rolling_minimums_vec.f64s[word_index] = rolling_minimums[first_dim + word_index];
                rolling_counts_vec.u64s[word_index] = rolling_counts[first_dim + word_index];
            }
        }
        // Otherwise, everything is easy
        else {
            last_states_vec.ymm_pd = _mm256_loadu_pd(&last_states[first_dim]);
            rolling_minimums_vec.ymm_pd = _mm256_loadu_pd(&rolling_minimums[first_dim]);
            rolling_counts_vec.ymm = _mm256_cvtepu32_epi64(
                _mm_loadu_si128(reinterpret_cast<__m128i const *>(&rolling_counts[first_dim])));
        }

        // Temporary variables for the rolling state
        u256_vec_t multipliers_vec, negative_discarding_multipliers_vec, modulos_vec, inverse_modulos_vec;
        multipliers_vec.ymm_pd = _mm256_loadu_pd(&multipliers_[first_dim]);
        negative_discarding_multipliers_vec.ymm_pd = _mm256_loadu_pd(&negative_discarding_multipliers_[first_dim]);
        modulos_vec.ymm_pd = _mm256_loadu_pd(&modulos_[first_dim]);
        inverse_modulos_vec.ymm_pd = _mm256_loadu_pd(&inverse_modulos_[first_dim]);

        // Until we reach the `window_width_`, we don't need to discard any symbols and can keep the code simpler
        size_t const prefix_length = (std::min)(text_chunk.size(), window_width_);
        size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            __m256d new_term_ymm = _mm256_set1_pd(new_term);

            last_states_vec.ymm_pd = _mm256_fmadd_pd(last_states_vec.ymm_pd, multipliers_vec.ymm_pd, new_term_ymm);
            last_states_vec.ymm_pd = barrett_mod( //
                last_states_vec.ymm_pd,           //
                modulos_vec.ymm_pd,               //
                inverse_modulos_vec.ymm_pd);
        }

        // We now have our first minimum hashes
        __m256i const ones_ymm = _mm256_set1_epi64x(1);
        if (new_char_offset == window_width_ && passed_progress < prefix_length)
            rolling_minimums_vec.ymm_pd = last_states_vec.ymm_pd, rolling_counts_vec.ymm = ones_ymm;

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            byte_t const old_char = text_chunk[new_char_offset - window_width_];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            rolling_state_t const old_term = static_cast<rolling_state_t>(old_char) + 1.0;
            __m256d new_term_ymm = _mm256_set1_pd(new_term);
            __m256d old_term_ymm = _mm256_set1_pd(old_term);

            // Discard the old term
            last_states_vec.ymm_pd = _mm256_fmadd_pd(negative_discarding_multipliers_vec.ymm_pd, old_term_ymm,
                                                     last_states_vec.ymm_pd);
            last_states_vec.ymm_pd = barrett_mod( //
                last_states_vec.ymm_pd,           //
                modulos_vec.ymm_pd,               //
                inverse_modulos_vec.ymm_pd);

            // Add the new term
            last_states_vec.ymm_pd = _mm256_fmadd_pd(last_states_vec.ymm_pd, multipliers_vec.ymm_pd, new_term_ymm);
            last_states_vec.ymm_pd = barrett_mod( //
                last_states_vec.ymm_pd,           //
                modulos_vec.ymm_pd,               //
                inverse_modulos_vec.ymm_pd);

            // To keep the right comparison mask, check out: https://stackoverflow.com/q/16988199
            __m256d found_ymm = _mm256_cmp_pd(last_states_vec.ymm_pd, rolling_minimums_vec.ymm_pd, _CMP_LE_OQ);
            __m256d discard_ymm = _mm256_cmp_pd(last_states_vec.ymm_pd, rolling_minimums_vec.ymm_pd, _CMP_GE_OQ);
            rolling_minimums_vec.ymm_pd = _mm256_blendv_pd(rolling_minimums_vec.ymm_pd, last_states_vec.ymm_pd,
                                                           found_ymm);

            // A branchless way to update the counts
            // 1. Discard "min counts" to 0, if a new minimum is found
            // 2. Increment the counts for new & existing minimums
            rolling_counts_vec.ymm_pd = _mm256_blendv_pd(_mm256_setzero_pd(), rolling_counts_vec.ymm_pd, discard_ymm);
            rolling_counts_vec.ymm_pd = _mm256_blendv_pd(
                rolling_counts_vec.ymm_pd, _mm256_castsi256_pd(_mm256_add_epi64(rolling_counts_vec.ymm, ones_ymm)),
                found_ymm);
        }

        // Dump back the results from registers into our spans
        if (has_incomplete_tail_group_k && group_index + 1 == groups_count_k) {
            for (size_t word_index = 0; word_index < (dimensions_k - first_dim); ++word_index) {
                last_states[first_dim + word_index] = last_states_vec.f64s[word_index];
                rolling_minimums[first_dim + word_index] = rolling_minimums_vec.f64s[word_index];
                rolling_counts[first_dim + word_index] = static_cast<min_count_t>(rolling_counts_vec.u64s[word_index]);
            }
        }
        else {
            _mm256_storeu_pd(&last_states[first_dim], last_states_vec.ymm_pd);
            _mm256_storeu_pd(&rolling_minimums[first_dim], rolling_minimums_vec.ymm_pd);
            // AVX2-compatible replacement for `_mm256_cvtepi64_epi32`
            __m256i shuffled = _mm256_shuffle_epi32(rolling_counts_vec.ymm, _MM_SHUFFLE(2, 0, 2, 0));
            __m128i lo = _mm256_extracti128_si256(shuffled, 0);
            __m128i hi = _mm256_extracti128_si256(shuffled, 1);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(&rolling_counts[first_dim]), _mm_unpacklo_epi64(lo, hi));
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#pragma endregion Haswell Implementation

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FINGERPRINTS_HASWELL_HPP_
