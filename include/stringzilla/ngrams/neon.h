/**
 *  @file include/stringzilla/ngrams/neon.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief NEON backend for N-grams: two prefix residues a step and the residues of two N-grams at a
 *      time, filed into and probed against the serial tables and ranked into the serial sketch.
 *
 *  @sa c/stringzilla/ngrams.h
 */
#ifndef STRINGZILLA_NGRAMS_NEON_H_
#define STRINGZILLA_NGRAMS_NEON_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEON_

#include "stringzilla/ngrams/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

#pragma region NEON

enum { sz_ngrams_f64x2_positions_per_step_neon_k = 2 };

/**
 *  @brief Advances the chain over two bytes, writing both prefix residues.
 *  @see https://ashvardanian.com/posts/stringwars-on-gpus/#52-bit-math-for-bioinformatics
 */
STRINGZILLA_INLINE sz_f64_t sz_ngrams_f64x2_prefix_step_neon_(sz_f64_t prior, sz_cptr_t text, sz_f64_t *prefixes) {
    sz_f64_t const chunks[2] = {(sz_f64_t)(sz_u8_t)text[0],
                                (sz_f64_t)((sz_u32_t)(sz_u8_t)text[0] * 256u + (sz_u8_t)text[1])};
    float64x2_t const modulus_f64x2 = vdupq_n_f64((sz_f64_t)sz_ngrams_modulus_k);
    // Power-of-two multipliers keep these integer products exact below 2^48.
    float64x2_t const product_f64x2 = vfmaq_f64(vld1q_f64(chunks), vdupq_n_f64(prior),
                                                vld1q_f64(sz_ngrams_powers_of_256_k + 1));
    float64x2_t const quotient_f64x2 = vrndnq_f64(vmulq_n_f64(product_f64x2, 1.0 / (sz_f64_t)sz_ngrams_modulus_k));
    float64x2_t const residue_f64x2 = vfmsq_f64(product_f64x2, quotient_f64x2, modulus_f64x2);
    vst1q_f64(prefixes,
              vaddq_f64(residue_f64x2, vreinterpretq_f64_u64(vandq_u64(vcltq_f64(residue_f64x2, vdupq_n_f64(0)),
                                                                       vreinterpretq_u64_f64(modulus_f64x2)))));
    return prefixes[1];
}

/** Advances the chain over the final zero or one byte. */
STRINGZILLA_INLINE sz_f64_t sz_ngrams_f64x2_prefix_step_tail_neon_(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                                   sz_f64_t *prefixes) {
    return count ? sz_ngrams_f64x1_prefix_step_serial_(prior, text, prefixes) : prior;
}

/** Extracts two N-gram residues from their start and end prefix residues. */
STRINGZILLA_INLINE void sz_ngrams_f64x2_residue_step_neon_(sz_f64_t const *start, sz_f64_t const *end, sz_f64_t power,
                                                           sz_u32_t *residues) {
    float64x2_t const modulus_f64x2 = vdupq_n_f64((sz_f64_t)sz_ngrams_modulus_k);
    float64x2_t const start_f64x2 = vld1q_f64(start), power_f64x2 = vdupq_n_f64(power);
    float64x2_t const high_f64x2 = vmulq_f64(start_f64x2, power_f64x2);
    float64x2_t const low_f64x2 = vfmaq_f64(vnegq_f64(high_f64x2), start_f64x2, power_f64x2);
    float64x2_t const quotient_f64x2 = vrndnq_f64(vmulq_n_f64(high_f64x2, 1.0 / (sz_f64_t)sz_ngrams_modulus_k));
    // Without an addend, one compensated reduction stays strictly between -p and p.
    float64x2_t const folded_f64x2 = vaddq_f64(vfmsq_f64(high_f64x2, quotient_f64x2, modulus_f64x2), low_f64x2);
    float64x2_t const shifted_f64x2 = vaddq_f64(
        folded_f64x2, vreinterpretq_f64_u64(
                          vandq_u64(vcltq_f64(folded_f64x2, vdupq_n_f64(0)), vreinterpretq_u64_f64(modulus_f64x2))));
    float64x2_t const difference_f64x2 = vsubq_f64(vld1q_f64(end), shifted_f64x2);
    float64x2_t const residue_f64x2 = vaddq_f64(
        difference_f64x2,
        vreinterpretq_f64_u64(vandq_u64(vcltq_f64(difference_f64x2, vdupq_n_f64(0)),
                                        vreinterpretq_u64_f64(vdupq_n_f64((sz_f64_t)sz_ngrams_modulus_k)))));
    vst1_u32(residues, vmovn_u64(vcvtq_u64_f64(residue_f64x2)));
}

/** Extracts the final zero or one N-gram residue. */
STRINGZILLA_INLINE void sz_ngrams_f64x2_residue_step_tail_neon_(sz_f64_t const *start, sz_f64_t const *end,
                                                                sz_f64_t power, sz_size_t count, sz_u32_t *residues) {
    if (count) sz_ngrams_f64x1_residue_step_serial_(start, end, power, residues);
}

/** Advances a chain from @p prior over @p count bytes of @p text, writing the prefix residue past
 *  each byte into @p prefixes, and answers the last. */
STRINGZILLA_INLINE sz_f64_t sz_ngrams_prefixes_neon_(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                     sz_f64_t *prefixes) {
    sz_size_t position = 0;
    for (; position + sz_ngrams_f64x2_positions_per_step_neon_k <= count;
         position += sz_ngrams_f64x2_positions_per_step_neon_k)
        prior = sz_ngrams_f64x2_prefix_step_neon_(prior, text + position, prefixes + position);
    return position != count
               ? sz_ngrams_f64x2_prefix_step_tail_neon_(prior, text + position, count - position, prefixes + position)
               : prior;
}

/** The residues of @p count N-grams of one width, from the prefix residues at their first bytes,
 *  @p starts, and past their last, @p ends. */
STRINGZILLA_INLINE void sz_ngrams_residues_span_neon_(sz_f64_t const *starts, sz_f64_t const *ends, sz_f64_t power,
                                                      sz_size_t count, sz_u32_t *residues) {
    sz_size_t window = 0;
    for (; window + sz_ngrams_f64x2_positions_per_step_neon_k <= count;
         window += sz_ngrams_f64x2_positions_per_step_neon_k)
        sz_ngrams_f64x2_residue_step_neon_(starts + window, ends + window, power, residues + window);
    if (window != count)
        sz_ngrams_f64x2_residue_step_tail_neon_(starts + window, ends + window, power, count - window,
                                                residues + window);
}

/** Mixes four values at once, as @ref sz_ngrams_mix_ mixes one. */
STRINGZILLA_INLINE uint32x4_t sz_ngrams_mix_u32x4_neon_(uint32x4_t values_u32x4) {
    values_u32x4 = veorq_u32(values_u32x4, vshrq_n_u32(values_u32x4, 16));
    values_u32x4 = vmulq_u32(values_u32x4, vdupq_n_u32(0x7FEB352Du));
    values_u32x4 = veorq_u32(values_u32x4, vshrq_n_u32(values_u32x4, 15));
    values_u32x4 = vmulq_u32(values_u32x4, vdupq_n_u32(0x846CA68Bu));
    return veorq_u32(values_u32x4, vshrq_n_u32(values_u32x4, 16));
}

/** Ranks @p count N-grams of one @p width by their @p residues into the @p slots least @p ranks,
 *  hashing four N-grams a step. */
STRINGZILLA_INLINE void sz_ngrams_sketch_block_neon_(sz_u32_t const *residues, sz_size_t count, sz_size_t width,
                                                     sz_u32_t *ranks, sz_size_t slots) {
    sz_u32_t slots_of[sz_ngrams_sketch_block_k], ranks_of[sz_ngrams_sketch_block_k];
    uint32x4_t const width_u32x4 = vdupq_n_u32((sz_u32_t)width);
    uint32x2_t const slots_u32x2 = vdup_n_u32((sz_u32_t)slots);
    sz_size_t window = 0;
    for (; window + 4 <= count; window += 4) {
        uint32x4_t const mixed_u32x4 = sz_ngrams_mix_u32x4_neon_(vaddq_u32(vld1q_u32(residues + window), width_u32x4));
        uint64x2_t const low_u64x2 = vmull_u32(vget_low_u32(mixed_u32x4), slots_u32x2);
        uint64x2_t const high_u64x2 = vmull_u32(vget_high_u32(mixed_u32x4), slots_u32x2);
        vst1q_u32(slots_of + window, vcombine_u32(vshrn_n_u64(low_u64x2, 32), vshrn_n_u64(high_u64x2, 32)));
        vst1q_u32(ranks_of + window, sz_ngrams_mix_u32x4_neon_(veorq_u32(mixed_u32x4, vdupq_n_u32(0x9E3779B9u))));
    }
    sz_ngrams_sketch_hash_serial_(residues + window, count - window, width, slots, slots_of + window,
                                  ranks_of + window);
    sz_ngrams_sketch_scatter_(slots_of, ranks_of, count, ranks);
}

/** Densifies the @p slots least @p ranks and writes each one's byte, four slots a step: the nearest
 *  reached slot after each comes from @ref sz_ngrams_sketch_nexts_ a block at a time, and an
 *  unreached slot borrows its rank by a load, so no step waits on the one before. */
STRINGZILLA_INLINE void sz_ngrams_sketch_close_neon_(sz_u32_t *ranks, sz_size_t slots, sz_u8_t *sketch) {
    sz_size_t next;
    if (!sz_ngrams_sketch_lowest_(ranks, slots, &next)) {
        sz_ngrams_sketch_close_(ranks, slots, sketch);
        return;
    }
    sz_u32_t nexts[sz_ngrams_sketch_block_k + 4], owns[4], borrowed[4];
    sz_u32_t const lanes[4] = {0, 1, 2, 3};
    uint32x4_t const empty_u32x4 = vdupq_n_u32(0xFFFFFFFFu), slots_u32x4 = vdupq_n_u32((sz_u32_t)slots);
    for (sz_size_t high = slots; high != 0;) {
        sz_size_t const low = high > sz_ngrams_sketch_block_k ? high - sz_ngrams_sketch_block_k : 0;
        sz_ngrams_sketch_nexts_(ranks, low, high, &next, nexts);
        for (sz_size_t slot = low; slot < high; slot += 4) {
            sz_size_t const valid = sz_min_of_two(high - slot, (sz_size_t)4);
            for (sz_size_t lane = 0; lane != 4; ++lane) {
                owns[lane] = lane < valid ? ranks[slot + lane] : 0;
                nexts[slot - low + lane] = lane < valid ? nexts[slot - low + lane] : (sz_u32_t)next;
                borrowed[lane] = ranks[nexts[slot - low + lane]];
            }
            uint32x4_t const own_u32x4 = vld1q_u32(owns), sources_u32x4 = vld1q_u32(nexts + (slot - low));
            uint32x4_t const slots_at_u32x4 = vaddq_u32(vdupq_n_u32((sz_u32_t)slot), vld1q_u32(lanes));
            // A source below its slot lies past the wrap.
            uint32x4_t const distances_u32x4 = vaddq_u32(
                vsubq_u32(sources_u32x4, slots_at_u32x4),
                vandq_u32(vcltq_u32(sources_u32x4, slots_at_u32x4), slots_u32x4));
            uint32x4_t const dense_u32x4 = vbslq_u32(
                vceqq_u32(own_u32x4, empty_u32x4),
                sz_ngrams_mix_u32x4_neon_(vaddq_u32(vld1q_u32(borrowed), distances_u32x4)), own_u32x4);
            uint8x8_t const bytes_u8x8 = vmovn_u16(
                vcombine_u16(vmovn_u32(sz_ngrams_mix_u32x4_neon_(dense_u32x4)), vdup_n_u16(0)));
            sz_u64_vec_t bytes_vec;
            bytes_vec.u64 = vget_lane_u64(vreinterpret_u64_u8(bytes_u8x8), 0);
            for (sz_size_t lane = 0; lane != valid; ++lane) sketch[slot + lane] = bytes_vec.u8s[lane];
        }
        high = low;
    }
}

/** Ranks every window of @p text at the @p widths past @ref sz_ngrams_sketch_reach_k into @p ranks,
 *  two chains a width in lockstep, one at the N-grams' first bytes and one past their last. */
STRINGZILLA_INLINE void sz_ngrams_sketch_wide_neon_(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                    sz_size_t widths_count, sz_f64_t const *powers, sz_u32_t *ranks,
                                                    sz_size_t slots) {
    sz_size_t const step = sz_ngrams_f64x2_positions_per_step_neon_k;
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
                trailing = sz_ngrams_f64x2_prefix_step_neon_(trailing, first_bytes + window, starts + window + 1);
                leading = sz_ngrams_f64x2_prefix_step_neon_(leading, last_bytes + window, ends + window);
            }
            trailing = sz_ngrams_prefixes_neon_(trailing, first_bytes + window, count - window, starts + window + 1);
            leading = sz_ngrams_prefixes_neon_(leading, last_bytes + window, count - window, ends + window);
            sz_ngrams_residues_span_neon_(starts, ends, powers[width_index], count, residues);
            sz_ngrams_sketch_block_neon_(residues, count, width, ranks, slots);
        }
    }
}

/** Ranks the N-grams of one segment of @p text, whose prefix residues from its first start are at
 *  @p prefixes, at every width within @ref sz_ngrams_sketch_reach_k, through @p residues. */
STRINGZILLA_INLINE void sz_ngrams_sketch_segment_neon_(sz_size_t first, sz_size_t length, sz_size_t const *widths,
                                                       sz_size_t widths_count, sz_f64_t const *powers,
                                                       sz_f64_t const *prefixes, sz_u32_t *residues, sz_u32_t *ranks,
                                                       sz_size_t slots) {
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        if (width > sz_ngrams_sketch_reach_k || first + width > length) continue;
        sz_size_t const count = sz_min_of_two(length - width + 1 - first, (sz_size_t)sz_ngrams_sketch_block_k);
        sz_ngrams_residues_span_neon_(prefixes, prefixes + width, powers[width_index], count, residues);
        sz_ngrams_sketch_block_neon_(residues, count, width, ranks, slots);
    }
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API void sz_ngrams_residues_neon(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                             sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues) {
    prefixes[0] = 0.0;
    sz_ngrams_prefixes_neon_(0.0, text, length, prefixes + 1);
    for (sz_size_t width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = widths[width_index];
        if (!width || width > length) continue;
        sz_size_t const ngrams = length - width + 1;
        sz_ngrams_residues_span_neon_(prefixes, prefixes + width, (sz_f64_t)sz_ngrams_power(width), ngrams, residues);
        residues += ngrams;
    }
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init_neon(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
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
        sz_ngrams_residues_neon(views[index % 64].start, engine->lengths[index], engine->widths, engine->widths_count,
                                prefixes, residues);
        sz_ngrams_engine_fill_row_(engine, index, residues);
    }
    engine->capability = sz_cap_neon_k;
    return sz_success_k;
}

/** Hashes the candidates of @p round, four chains at a time, and scores each as @p round says. */
STRINGZILLA_INLINE void sz_ngrams_round_neon_(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                              sz_size_t longest_candidate, sz_ngrams_round_t const *round) {
    sz_size_t const step = sz_ngrams_f64x2_positions_per_step_neon_k;
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
                priors[chain] = sz_ngrams_f64x2_prefix_step_neon_(priors[chain], texts[chain] + walked,
                                                                  prefixes + chain * chain_stride + walked + 1);

        // One candidate's residues at every width serve every query's table, so the hashing runs
        // once here and the probes run once per query over what it wrote.
        for (chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefixes = prefixes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_ngrams_prefixes_neon_(priors[chain], texts[chain] + walked, length - walked,
                                     chain_prefixes + walked + 1);
            sz_u32_t *keys = residues;
            for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
                sz_size_t const width = engine->widths[width_index];
                sz_size_t const ngrams = width <= length ? length - width + 1 : 0;
                sz_ngrams_residues_span_neon_(chain_prefixes, chain_prefixes + width,
                                              (sz_f64_t)engine->powers[width_index], ngrams, keys);
                keys += ngrams;
            }
            sz_ngrams_round_score_(engine, round, indices[chain], residues, length);
        }
    }
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps_neon(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                    sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                    sz_size_t matches_candidate_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_ngrams_round_t round;
    sz_size_t longest_candidate = 0;
    sz_status_t const prepared = sz_ngrams_overlaps_round_(engine, candidates, radius, matches, matches_query_stride,
                                                           matches_candidate_stride, sz_ngrams_interleaved_chains_k,
                                                           stream, &round, &longest_candidate);
    if (prepared != sz_success_k) return prepared;
    sz_ngrams_round_neon_(engine, candidates, longest_candidate, &round);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_sketch_neon(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                  sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots,
                                                  sz_u8_t *sketch) {
    sz_status_t const status = sz_ngrams_sketch_dimensions_(widths, widths_count, slots);
    if (status != sz_success_k) return status;
    for (sz_size_t slot = 0; slot != slots; ++slot) ranks[slot] = 0xFFFFFFFFu;
    sz_size_t const step = sz_ngrams_f64x2_positions_per_step_neon_k;
    sz_size_t const chains = sz_ngrams_interleaved_chains_k, block = sz_ngrams_sketch_block_k;
    sz_f64_t powers[sz_ngrams_widths_max_k];
    sz_f64_t prefixes[sz_ngrams_interleaved_chains_k][sz_ngrams_sketch_block_k + sz_ngrams_sketch_reach_k];
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
                priors[chain] = sz_ngrams_f64x2_prefix_step_neon_(priors[chain], text + firsts[chain] + walked,
                                                                  prefixes[chain] + walked + 1);
        for (chain = 0; chain != chains; ++chain) {
            sz_ngrams_prefixes_neon_(priors[chain], text + firsts[chain] + walked, covered[chain] - walked,
                                     prefixes[chain] + walked + 1);
            sz_ngrams_sketch_segment_neon_(firsts[chain], length, widths, widths_count, powers, prefixes[chain],
                                           residues, ranks, slots);
        }
    }
    sz_ngrams_sketch_wide_neon_(text, length, widths, widths_count, powers, ranks, slots);
    sz_ngrams_sketch_close_neon_(ranks, slots, sketch);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches_neon(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
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
            sz_ngrams_sketch_neon(views[position].start, views[position].length, engine->widths, engine->widths_count,
                                  ranks, slots, sketches + (block_start + position) * sketches_stride);
    }
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_NEON

#pragma endregion NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_ARM64_NEON_
#endif // STRINGZILLA_ARCH_ARM64_

#endif // STRINGZILLA_NGRAMS_NEON_H_
