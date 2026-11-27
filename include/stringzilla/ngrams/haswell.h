/**
 *  @file include/stringzilla/ngrams/haswell.h
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief AVX2 (Haswell) backend for N-grams: four prefix residues from one word and the residues
 *      of four N-grams at a time, filed into and probed against the serial tables and ranked into
 *      the serial sketch.
 *
 *  @sa c/stringzilla/ngrams.h
 */
#ifndef STRINGZILLA_NGRAMS_HASWELL_H_
#define STRINGZILLA_NGRAMS_HASWELL_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_HASWELL_

#include "stringzilla/ngrams/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "bmi", "bmi2", "popcnt")
#endif

#pragma region Haswell

/** Positions one YMM step advances the chain by: the four prefix residues ending in one word. */
enum { sz_ngrams_f64x4_positions_per_step_haswell_k = 4 };

/** Exact @c u32 → @c f64, since AVX2 offers only the signed conversion. */
STRINGZILLA_INLINE __m256d sz_ngrams_u32x4_to_f64x4_haswell_(__m128i values_u32x4) {
    sz_u256_vec_t signed_vec, wrapped_vec;
    signed_vec.ymm_pd = _mm256_cvtepi32_pd(values_u32x4);
    wrapped_vec.ymm_pd = _mm256_cmp_pd(signed_vec.ymm_pd, _mm256_setzero_pd(), _CMP_LT_OQ);
    return _mm256_add_pd(signed_vec.ymm_pd, _mm256_and_pd(wrapped_vec.ymm_pd, _mm256_set1_pd(4294967296.0)));
}

/** Exact @c f64 → @c u32 for values below 2^32, biasing around the signed conversion's ceiling. */
STRINGZILLA_INLINE __m128i sz_ngrams_f64x4_to_u32x4_haswell_(__m256d values_f64x4) {
    __m128i const biased_i32x4 = _mm256_cvttpd_epi32(_mm256_sub_pd(values_f64x4, _mm256_set1_pd(2147483648.0)));
    return _mm_xor_si128(biased_i32x4, _mm_set1_epi32((int)0x80000000u));
}

/**
 *  @brief (multiplier · multiplicand + addend) mod p, in [0, p), exact for every input below 2³²:
 *      the product's rounded head and exact tail reduce together, so 53 mantissa bits never bind.
 *  @see https://ashvardanian.com/posts/stringwars-on-gpus/#52-bit-math-for-bioinformatics
 */
STRINGZILLA_INLINE __m256d sz_ngrams_multiply_add_haswell_(__m256d multiplier_f64x4, __m256d multiplicand_f64x4,
                                                           __m256d addend_f64x4) {
    sz_u256_vec_t modulus_vec, reciprocal_vec, high_vec, low_vec, quotient_vec, folded_vec;
    sz_u256_vec_t second_vec, residue_vec, negative_vec;
    modulus_vec.ymm_pd = _mm256_set1_pd((sz_f64_t)sz_ngrams_modulus_k);
    reciprocal_vec.ymm_pd = _mm256_set1_pd(1.0 / (sz_f64_t)sz_ngrams_modulus_k);
    high_vec.ymm_pd = _mm256_mul_pd(multiplier_f64x4, multiplicand_f64x4);
    low_vec.ymm_pd = _mm256_fmsub_pd(multiplier_f64x4, multiplicand_f64x4, high_vec.ymm_pd);
    quotient_vec.ymm_pd = _mm256_round_pd(_mm256_mul_pd(high_vec.ymm_pd, reciprocal_vec.ymm_pd),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    folded_vec.ymm_pd = _mm256_add_pd(
        _mm256_add_pd(_mm256_fnmadd_pd(quotient_vec.ymm_pd, modulus_vec.ymm_pd, high_vec.ymm_pd), low_vec.ymm_pd),
        addend_f64x4);
    second_vec.ymm_pd = _mm256_round_pd(_mm256_mul_pd(folded_vec.ymm_pd, reciprocal_vec.ymm_pd),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    residue_vec.ymm_pd = _mm256_fnmadd_pd(second_vec.ymm_pd, modulus_vec.ymm_pd, folded_vec.ymm_pd);
    negative_vec.ymm_pd = _mm256_cmp_pd(residue_vec.ymm_pd, _mm256_setzero_pd(), _CMP_LT_OQ);
    return _mm256_add_pd(residue_vec.ymm_pd, _mm256_and_pd(negative_vec.ymm_pd, modulus_vec.ymm_pd));
}

/** Advances a chain over one step of bytes, writing the prefix residue past each. */
STRINGZILLA_INLINE sz_f64_t sz_ngrams_f64x4_prefix_step_haswell_(sz_f64_t prior, sz_cptr_t text, sz_f64_t *prefixes) {
    sz_u32_t const word = sz_u32_bytes_reverse(sz_u32_load(text).u32);
    __m128i const shifted_u32x4 = _mm_srlv_epi32(_mm_set1_epi32((int)word), _mm_setr_epi32(24, 16, 8, 0));
    sz_u256_vec_t chunks_vec, powers_vec, values_vec;
    chunks_vec.ymm_pd = sz_ngrams_u32x4_to_f64x4_haswell_(shifted_u32x4);
    powers_vec.ymm_pd = _mm256_loadu_pd(sz_ngrams_powers_of_256_k + 1);
    values_vec.ymm_pd = sz_ngrams_multiply_add_haswell_(_mm256_set1_pd(prior), powers_vec.ymm_pd, chunks_vec.ymm_pd);
    _mm256_storeu_pd(prefixes, values_vec.ymm_pd);
    return prefixes[3];
}

/** Advances a chain over the last @p count bytes, fewer than a step. */
STRINGZILLA_INLINE sz_f64_t sz_ngrams_f64x4_prefix_step_tail_haswell_(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                                      sz_f64_t *prefixes) {
    for (sz_size_t position = 0; position != count; ++position)
        prior = sz_ngrams_f64x1_prefix_step_serial_(prior, text + position, prefixes + position);
    return prior;
}

/** The residues of one step of N-grams, R(i, w) = P(i + w) − P(i) · bʷ mod p. */
STRINGZILLA_INLINE void sz_ngrams_f64x4_residue_step_haswell_(sz_f64_t const *starts, sz_f64_t const *ends,
                                                              sz_f64_t power, sz_u32_t *residues) {
    sz_u256_vec_t start_vec, end_vec, shifted_vec, difference_vec, negative_vec, residues_vec;
    start_vec.ymm_pd = _mm256_loadu_pd(starts);
    end_vec.ymm_pd = _mm256_loadu_pd(ends);
    shifted_vec.ymm_pd = sz_ngrams_multiply_add_haswell_(start_vec.ymm_pd, _mm256_set1_pd(power), _mm256_setzero_pd());
    difference_vec.ymm_pd = _mm256_sub_pd(end_vec.ymm_pd, shifted_vec.ymm_pd);
    negative_vec.ymm_pd = _mm256_cmp_pd(difference_vec.ymm_pd, _mm256_setzero_pd(), _CMP_LT_OQ);
    residues_vec.ymm_pd = _mm256_add_pd(
        difference_vec.ymm_pd, _mm256_and_pd(negative_vec.ymm_pd, _mm256_set1_pd((sz_f64_t)sz_ngrams_modulus_k)));
    _mm_storeu_si128((__m128i *)residues, sz_ngrams_f64x4_to_u32x4_haswell_(residues_vec.ymm_pd));
}

/** The residues of the last @p count N-grams, fewer than a step. */
STRINGZILLA_INLINE void sz_ngrams_f64x4_residue_step_tail_haswell_(sz_f64_t const *starts, sz_f64_t const *ends,
                                                                   sz_f64_t power, sz_size_t count,
                                                                   sz_u32_t *residues) {
    for (sz_size_t position = 0; position != count; ++position)
        sz_ngrams_f64x1_residue_step_serial_(starts + position, ends + position, power, residues + position);
}

/** Advances a chain from @p prior over @p count bytes of @p text, writing the prefix residue past
 *  each byte into @p prefixes, and answers the last. */
STRINGZILLA_INLINE sz_f64_t sz_ngrams_prefixes_haswell_(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                        sz_f64_t *prefixes) {
    sz_size_t position = 0;
    for (; position + sz_ngrams_f64x4_positions_per_step_haswell_k <= count;
         position += sz_ngrams_f64x4_positions_per_step_haswell_k)
        prior = sz_ngrams_f64x4_prefix_step_haswell_(prior, text + position, prefixes + position);
    return position != count ? sz_ngrams_f64x4_prefix_step_tail_haswell_(prior, text + position, count - position,
                                                                         prefixes + position)
                             : prior;
}

/** The residues of @p count N-grams of one width, from the prefix residues at their first bytes,
 *  @p starts, and past their last, @p ends. */
STRINGZILLA_INLINE void sz_ngrams_residues_span_haswell_(sz_f64_t const *starts, sz_f64_t const *ends, sz_f64_t power,
                                                         sz_size_t count, sz_u32_t *residues) {
    sz_size_t window = 0;
    for (; window + sz_ngrams_f64x4_positions_per_step_haswell_k <= count;
         window += sz_ngrams_f64x4_positions_per_step_haswell_k)
        sz_ngrams_f64x4_residue_step_haswell_(starts + window, ends + window, power, residues + window);
    if (window != count)
        sz_ngrams_f64x4_residue_step_tail_haswell_(starts + window, ends + window, power, count - window,
                                                   residues + window);
}

/** Mixes eight values at once, as @ref sz_ngrams_mix_ mixes one. */
STRINGZILLA_INLINE __m256i sz_ngrams_mix_u32x8_haswell_(__m256i values_u32x8) {
    values_u32x8 = _mm256_xor_si256(values_u32x8, _mm256_srli_epi32(values_u32x8, 16));
    values_u32x8 = _mm256_mullo_epi32(values_u32x8, _mm256_set1_epi32((int)0x7FEB352Du));
    values_u32x8 = _mm256_xor_si256(values_u32x8, _mm256_srli_epi32(values_u32x8, 15));
    values_u32x8 = _mm256_mullo_epi32(values_u32x8, _mm256_set1_epi32((int)0x846CA68Bu));
    return _mm256_xor_si256(values_u32x8, _mm256_srli_epi32(values_u32x8, 16));
}

/** Ranks @p count N-grams of one @p width by their @p residues into the @p slots least @p ranks,
 *  hashing eight N-grams a step. */
STRINGZILLA_INLINE void sz_ngrams_sketch_block_haswell_(sz_u32_t const *residues, sz_size_t count, sz_size_t width,
                                                        sz_u32_t *ranks, sz_size_t slots) {
    sz_u32_t slots_of[sz_ngrams_sketch_block_k], ranks_of[sz_ngrams_sketch_block_k];
    __m256i const width_u32x8 = _mm256_set1_epi32((int)width), slots_u32x8 = _mm256_set1_epi32((int)slots);
    __m256i const lanes_u32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (sz_size_t window = 0; window < count; window += 8) {
        __m256i const valid_u32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(count - window)), lanes_u32x8);
        __m256i const mixed_u32x8 = sz_ngrams_mix_u32x8_haswell_(
            _mm256_add_epi32(_mm256_maskload_epi32((int const *)(residues + window), valid_u32x8), width_u32x8));
        // The high words of the even and the odd lanes' products with the slot count.
        __m256i const even_u64x4 = _mm256_mul_epu32(mixed_u32x8, slots_u32x8);
        __m256i const odd_u64x4 = _mm256_mul_epu32(_mm256_srli_epi64(mixed_u32x8, 32), slots_u32x8);
        _mm256_maskstore_epi32((int *)(slots_of + window), valid_u32x8,
                               _mm256_blend_epi32(_mm256_srli_epi64(even_u64x4, 32), odd_u64x4, 0xAA));
        _mm256_maskstore_epi32(
            (int *)(ranks_of + window), valid_u32x8,
            sz_ngrams_mix_u32x8_haswell_(_mm256_xor_si256(mixed_u32x8, _mm256_set1_epi32((int)0x9E3779B9u))));
    }
    sz_ngrams_sketch_scatter_(slots_of, ranks_of, count, ranks);
}

/** Densifies the @p slots least @p ranks and writes each one's byte, eight slots a step: the
 *  nearest reached slot after each comes from @ref sz_ngrams_sketch_nexts_ a block at a time, and
 *  an unreached slot gathers its rank, so no step waits on the one before. */
STRINGZILLA_INLINE void sz_ngrams_sketch_close_haswell_(sz_u32_t *ranks, sz_size_t slots, sz_u8_t *sketch) {
    sz_size_t next;
    if (!sz_ngrams_sketch_lowest_(ranks, slots, &next)) {
        sz_ngrams_sketch_close_(ranks, slots, sketch);
        return;
    }
    sz_u32_t nexts[sz_ngrams_sketch_block_k];
    __m256i const empty_u32x8 = _mm256_set1_epi32(-1), slots_u32x8 = _mm256_set1_epi32((int)slots);
    __m256i const lanes_u32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i const sign_u32x8 = _mm256_set1_epi32((int)0x80000000u);
    // The low byte of each lane, gathered into the low eight bytes.
    __m256i const low_bytes_u8x32 = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4,
                                                     8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const lane_words_u32x8 = _mm256_setr_epi32(0, 4, 1, 1, 1, 1, 1, 1);
    for (sz_size_t high = slots; high != 0;) {
        sz_size_t const low = high > sz_ngrams_sketch_block_k ? high - sz_ngrams_sketch_block_k : 0;
        sz_ngrams_sketch_nexts_(ranks, low, high, &next, nexts);
        for (sz_size_t slot = low; slot < high; slot += 8) {
            sz_size_t const valid = sz_min_of_two(high - slot, (sz_size_t)8);
            __m256i const valid_u32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)valid), lanes_u32x8);
            __m256i const own_u32x8 = _mm256_maskload_epi32((int const *)(ranks + slot), valid_u32x8);
            __m256i const sources_u32x8 = _mm256_maskload_epi32((int const *)(nexts + (slot - low)), valid_u32x8);
            __m256i const slots_at_u32x8 = _mm256_add_epi32(_mm256_set1_epi32((int)slot), lanes_u32x8);
            __m256i const unreached_u32x8 = _mm256_and_si256(_mm256_cmpeq_epi32(own_u32x8, empty_u32x8), valid_u32x8);
            // A source below its slot lies past the wrap.
            __m256i const wrapped_u32x8 = _mm256_cmpgt_epi32(_mm256_xor_si256(slots_at_u32x8, sign_u32x8),
                                                             _mm256_xor_si256(sources_u32x8, sign_u32x8));
            __m256i const distances_u32x8 = _mm256_add_epi32(_mm256_sub_epi32(sources_u32x8, slots_at_u32x8),
                                                             _mm256_and_si256(wrapped_u32x8, slots_u32x8));
            __m256i const borrowed_u32x8 = _mm256_mask_i32gather_epi32(own_u32x8, (int const *)ranks, sources_u32x8,
                                                                       unreached_u32x8, 4);
            __m256i const dense_u32x8 = _mm256_blendv_epi8(
                own_u32x8, sz_ngrams_mix_u32x8_haswell_(_mm256_add_epi32(borrowed_u32x8, distances_u32x8)),
                unreached_u32x8);
            __m256i const bytes_u8x32 = _mm256_permutevar8x32_epi32(
                _mm256_shuffle_epi8(sz_ngrams_mix_u32x8_haswell_(dense_u32x8), low_bytes_u8x32), lane_words_u32x8);
            sz_u64_vec_t bytes_vec;
            bytes_vec.u64 = (sz_u64_t)_mm_cvtsi128_si64(_mm256_castsi256_si128(bytes_u8x32));
            for (sz_size_t lane = 0; lane != valid; ++lane) sketch[slot + lane] = bytes_vec.u8s[lane];
        }
        high = low;
    }
}

/** Ranks every window of @p text at the @p widths past @ref sz_ngrams_sketch_reach_k into @p ranks,
 *  two chains a width in lockstep, one at the N-grams' first bytes and one past their last. */
STRINGZILLA_INLINE void sz_ngrams_sketch_wide_haswell_(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                       sz_size_t widths_count, sz_f64_t const *powers, sz_u32_t *ranks,
                                                       sz_size_t slots) {
    sz_size_t const step = sz_ngrams_f64x4_positions_per_step_haswell_k;
    sz_f64_t starts[sz_ngrams_sketch_block_k + 1], ends[sz_ngrams_sketch_block_k];
    sz_u32_t residues[sz_ngrams_sketch_block_k];
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        if (width <= sz_ngrams_sketch_reach_k || width > length) continue;
        sz_size_t const ngrams = length - width + 1;
        sz_f64_t trailing = 0.0, leading = sz_ngrams_prefix_serial_(text, width - 1);
        for (sz_size_t first = 0, count; first != ngrams; first += count) {
            count = sz_min_of_two(ngrams - first, (sz_size_t)sz_ngrams_sketch_block_k);
            sz_cptr_t const first_bytes = text + first, last_bytes = text + first + width - 1;
            sz_size_t window = 0;
            starts[0] = trailing;
            for (; window + step <= count; window += step) {
                trailing = sz_ngrams_f64x4_prefix_step_haswell_(trailing, first_bytes + window, starts + window + 1);
                leading = sz_ngrams_f64x4_prefix_step_haswell_(leading, last_bytes + window, ends + window);
            }
            trailing = sz_ngrams_prefixes_haswell_(trailing, first_bytes + window, count - window, starts + window + 1);
            leading = sz_ngrams_prefixes_haswell_(leading, last_bytes + window, count - window, ends + window);
            sz_ngrams_residues_span_haswell_(starts, ends, powers[width_index], count, residues);
            sz_ngrams_sketch_block_haswell_(residues, count, width, ranks, slots);
        }
    }
}

/** Ranks the N-grams of one segment of @p text, whose prefix residues from its first start are at
 *  @p prefixes, at every width within @ref sz_ngrams_sketch_reach_k, through @p residues. */
STRINGZILLA_INLINE void sz_ngrams_sketch_segment_haswell_(sz_size_t first, sz_size_t length, sz_size_t const *widths,
                                                          sz_size_t widths_count, sz_f64_t const *powers,
                                                          sz_f64_t const *prefixes, sz_u32_t *residues, sz_u32_t *ranks,
                                                          sz_size_t slots) {
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        if (width > sz_ngrams_sketch_reach_k || first + width > length) continue;
        sz_size_t const count = sz_min_of_two(length - width + 1 - first, (sz_size_t)sz_ngrams_sketch_block_k);
        sz_ngrams_residues_span_haswell_(prefixes, prefixes + width, powers[width_index], count, residues);
        sz_ngrams_sketch_block_haswell_(residues, count, width, ranks, slots);
    }
}

#if STRINGZILLA_TARGET_HASWELL

STRINGZILLA_API void sz_ngrams_residues_haswell(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues) {
    prefixes[0] = 0.0;
    sz_ngrams_prefixes_haswell_(0.0, text, length, prefixes + 1);
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        if (!width || width > length) continue;
        sz_size_t const ngrams = length - width + 1;
        sz_ngrams_residues_span_haswell_(prefixes, prefixes + width, (sz_f64_t)sz_ngrams_power(width), ngrams,
                                         residues);
        residues += ngrams;
    }
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init_haswell(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                          sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                          sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_unused_(candidates_budget);
    sz_status_t const opened = sz_ngrams_engine_open_host_(queries, window_widths, window_widths_count, allocator,
                                                           engine);
    if (opened != sz_success_k) return opened;
    sz_f64_t *prefixes;
    sz_u32_t *residues;
    sz_status_t const grown = sz_ngrams_engine_query_scratch_(engine, stream, &prefixes, &residues);
    if (grown != sz_success_k) return grown;
    sz_string_view_t views[64];
    for (sz_size_t index = 0; index != engine->count; ++index) {
        if (index % 64 == 0) sz_strings_get_block_(queries, index, views);
        sz_ngrams_residues_haswell(views[index % 64].start, engine->lengths[index], engine->widths,
                                   engine->widths_count, prefixes, residues);
        sz_ngrams_engine_fill_row_(engine, index, residues);
    }
    engine->capability = sz_cap_haswell_k;
    return sz_success_k;
}

/** Hashes the candidates of @p round, four chains at a time, and scores each as @p round says. */
STRINGZILLA_INLINE void sz_ngrams_round_haswell_(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                 sz_size_t longest_candidate, sz_ngrams_round_t const *round) {
    sz_size_t const step = sz_ngrams_f64x4_positions_per_step_haswell_k;
    sz_size_t const chains = sz_ngrams_interleaved_chains_k;
    sz_size_t const chain_stride = longest_candidate + 1;
    sz_f64_t *const prefixes = (sz_f64_t *)engine->scratch;
    sz_u32_t *const residues = (sz_u32_t *)(prefixes + chain_stride * chains);

    // Four candidates' chains advance together, each carrying its own prior, so their latency-bound
    // steps overlap. A short last group repeats its first candidate in the spare chains, so every
    // group runs the same four-chain loop.
    for (sz_size_t first = 0; first < round->count; first += chains) {
        sz_size_t const interleaved = sz_min_of_two(round->count - first, chains);
        sz_cptr_t texts[sz_ngrams_interleaved_chains_k];
        sz_size_t lengths[sz_ngrams_interleaved_chains_k];
        sz_f64_t priors[sz_ngrams_interleaved_chains_k];
        sz_size_t indices[sz_ngrams_interleaved_chains_k];
        sz_string_view_t views[sz_ngrams_interleaved_chains_k];
        sz_size_t chain, shortest = STRINGZILLA_SIZE_MAX;
        for (chain = 0; chain != chains; ++chain)
            indices[chain] = round->order[first + (chain < interleaved ? chain : 0)];
        sz_strings_get_(candidates, indices, chains, views);
        for (chain = 0; chain != chains; ++chain) {
            texts[chain] = views[chain].start, lengths[chain] = views[chain].length;
            priors[chain] = prefixes[chain * chain_stride] = 0.0;
            shortest = sz_min_of_two(shortest, lengths[chain]);
        }
        sz_size_t walked = 0;
        for (; walked + step <= shortest; walked += step)
            for (chain = 0; chain != chains; ++chain)
                priors[chain] = sz_ngrams_f64x4_prefix_step_haswell_(priors[chain], texts[chain] + walked,
                                                                     prefixes + chain * chain_stride + walked + 1);

        // One candidate's residues at every width serve every query's table, so the hashing runs
        // once here and the probes run once per query over what it wrote.
        for (chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefixes = prefixes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_ngrams_prefixes_haswell_(priors[chain], texts[chain] + walked, length - walked,
                                        chain_prefixes + walked + 1);
            sz_u32_t *keys = residues;
            for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
                sz_size_t const width = engine->widths[width_index];
                sz_size_t const ngrams = width <= length ? length - width + 1 : 0;
                sz_ngrams_residues_span_haswell_(chain_prefixes, chain_prefixes + width,
                                                 (sz_f64_t)engine->powers[width_index], ngrams, keys);
                keys += ngrams;
            }
            sz_ngrams_round_score_(engine, round, indices[chain], residues, length);
        }
    }
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps_haswell(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                       sz_f32_t radius, sz_f32_t *shares, sz_size_t shares_query_stride,
                                                       sz_size_t shares_candidate_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_ngrams_round_t round;
    sz_size_t longest_candidate = 0;
    sz_status_t const prepared = sz_ngrams_overlaps_round_(
        engine, candidates, radius, STRINGZILLA_NULL, shares, shares_query_stride, shares_candidate_stride,
        sz_ngrams_interleaved_chains_k, stream, &round, &longest_candidate);
    if (prepared != sz_success_k) return prepared;
    sz_ngrams_round_haswell_(engine, candidates, longest_candidate, &round);
    return sz_success_k;
}

/** Counts what @ref sz_ngrams_overlaps_haswell shares: each query's distinct N-grams each candidate
 *  holds, at every width, @p counts_query_stride and @p counts_candidate_stride bytes apart. */
STRINGZILLA_INLINE sz_status_t sz_ngrams_overlap_counts_haswell_(sz_ngrams_engine_t *engine,
                                                                 sz_strings_view_t const *candidates, sz_f32_t radius,
                                                                 sz_u32_t *counts, sz_size_t counts_query_stride,
                                                                 sz_size_t counts_candidate_stride) {
    sz_ngrams_round_t round;
    sz_size_t longest_candidate = 0;
    sz_status_t const prepared = sz_ngrams_overlaps_round_(
        engine, candidates, radius, counts, STRINGZILLA_NULL, counts_query_stride, counts_candidate_stride,
        sz_ngrams_interleaved_chains_k, STRINGZILLA_NULL, &round, &longest_candidate);
    if (prepared != sz_success_k) return prepared;
    sz_ngrams_round_haswell_(engine, candidates, longest_candidate, &round);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_sketch_haswell(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                     sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots,
                                                     sz_u8_t *sketch) {
    sz_status_t const status = sz_ngrams_sketch_dimensions_(widths, widths_count, slots);
    if (status != sz_success_k) return status;
    for (sz_size_t slot = 0; slot != slots; ++slot) ranks[slot] = 0xFFFFFFFFu;
    sz_size_t const step = sz_ngrams_f64x4_positions_per_step_haswell_k;
    sz_size_t const chains = sz_ngrams_interleaved_chains_k, block = sz_ngrams_sketch_block_k;
    sz_f64_t powers[sz_ngrams_widths_max_k];
    sz_f64_t prefixes[sz_ngrams_interleaved_chains_k][(sz_size_t)sz_ngrams_sketch_block_k + sz_ngrams_sketch_reach_k];
    sz_u32_t residues[sz_ngrams_sketch_block_k];
    sz_size_t const reach = sz_ngrams_sketch_reach_(length, widths, widths_count, powers);

    // Four segments of window starts a round, each chain from zero at its segment's first start, so
    // every window within the reach is a difference of one chain and the chains step independently.
    for (sz_size_t base = 0; reach && base < length; base += chains * block) {
        sz_size_t firsts[sz_ngrams_interleaved_chains_k], covered[sz_ngrams_interleaved_chains_k];
        sz_f64_t priors[sz_ngrams_interleaved_chains_k];
        sz_size_t chain, shortest = STRINGZILLA_SIZE_MAX;
        for (chain = 0; chain != chains; ++chain) {
            firsts[chain] = sz_min_of_two(base + chain * block, length);
            covered[chain] = sz_min_of_two(block + reach - 1, length - firsts[chain]);
            priors[chain] = prefixes[chain][0] = 0.0;
            shortest = sz_min_of_two(shortest, covered[chain]);
        }
        sz_size_t walked = 0;
        for (; walked + step <= shortest; walked += step)
            for (chain = 0; chain != chains; ++chain)
                priors[chain] = sz_ngrams_f64x4_prefix_step_haswell_(priors[chain], text + firsts[chain] + walked,
                                                                     prefixes[chain] + walked + 1);
        for (chain = 0; chain != chains; ++chain) {
            sz_ngrams_prefixes_haswell_(priors[chain], text + firsts[chain] + walked, covered[chain] - walked,
                                        prefixes[chain] + walked + 1);
            sz_ngrams_sketch_segment_haswell_(firsts[chain], length, widths, widths_count, powers, prefixes[chain],
                                              residues, ranks, slots);
        }
    }
    sz_ngrams_sketch_wide_haswell_(text, length, widths, widths_count, powers, ranks, slots);
    sz_ngrams_sketch_close_haswell_(ranks, slots, sketch);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches_haswell(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                       sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                       sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_unused_(stream);
    sz_u32_t *ranks;
    sz_status_t const prepared = sz_ngrams_sketches_scratch_(engine, slots, sketches_stride, &ranks);
    if (prepared != sz_success_k) return prepared;
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < texts->count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(texts, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position)
            sz_ngrams_sketch_haswell(views[position].start, views[position].length, engine->widths,
                                     engine->widths_count, ranks, slots,
                                     sketches + (block_start + position) * sketches_stride);
    }
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_HASWELL

#pragma endregion Haswell

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_X8664_HASWELL_
#endif // STRINGZILLA_ARCH_X8664_

#endif // STRINGZILLA_NGRAMS_HASWELL_H_
