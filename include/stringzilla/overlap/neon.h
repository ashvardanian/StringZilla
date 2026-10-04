/**
 *  @file include/stringzilla/overlap/neon.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief NEON prefix hashes and window overlap scores.
 */
#ifndef STRINGZILLA_OVERLAP_NEON_H_
#define STRINGZILLA_OVERLAP_NEON_H_
#include "stringzilla/types.h"
#include "stringzilla/overlap/serial.h"
#ifdef __cplusplus
extern "C" {
#endif
#if STRINGZILLA_ARCH_ARM64_NEON_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

enum { sz_overlap_f64x2_positions_per_step_neon_k = 2 };

/** Advances the chain over two bytes, writing both prefix hashes. */
STRINGZILLA_INLINE sz_f64_t sz_overlap_f64x2_prefix_hash_step_neon(sz_f64_t prior, sz_cptr_t text,
                                                                   sz_f64_t *prefix_hashes) {
    sz_f64_t const chunks[2] = {(sz_f64_t)(sz_u8_t)text[0],
                                (sz_f64_t)((sz_u32_t)(sz_u8_t)text[0] * 256u + (sz_u8_t)text[1])};
    float64x2_t const modulus_f64x2 = vdupq_n_f64((sz_f64_t)sz_overlap_modulus_k);
    // Power-of-two multipliers keep these integer products exact below 2^48.
    float64x2_t const product_f64x2 = vfmaq_f64(vld1q_f64(chunks), vdupq_n_f64(prior),
                                                vld1q_f64(sz_overlap_powers_of_256_k + 1));
    float64x2_t const quotient_f64x2 = vrndnq_f64(vmulq_n_f64(product_f64x2, 1.0 / (sz_f64_t)sz_overlap_modulus_k));
    float64x2_t const residue_f64x2 = vfmsq_f64(product_f64x2, quotient_f64x2, modulus_f64x2);
    vst1q_f64(prefix_hashes,
              vaddq_f64(residue_f64x2, vreinterpretq_f64_u64(vandq_u64(vcltq_f64(residue_f64x2, vdupq_n_f64(0)),
                                                                       vreinterpretq_u64_f64(modulus_f64x2)))));
    return prefix_hashes[1];
}

/** Advances the chain over the final zero or one byte. */
STRINGZILLA_INLINE sz_f64_t sz_overlap_f64x2_prefix_hash_step_tail_neon(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                                        sz_f64_t *prefix_hashes) {
    return count ? sz_overlap_f64x1_prefix_hash_step_serial_(prior, text, prefix_hashes) : prior;
}

/** Extracts two window hashes from their start and end prefix hashes. */
STRINGZILLA_INLINE void sz_overlap_f64x2_window_hash_step_neon(sz_f64_t const *start, sz_f64_t const *end,
                                                               sz_f64_t power, sz_u32_t *hashes) {
    float64x2_t const modulus_f64x2 = vdupq_n_f64((sz_f64_t)sz_overlap_modulus_k);
    float64x2_t const start_f64x2 = vld1q_f64(start), power_f64x2 = vdupq_n_f64(power);
    float64x2_t const high_f64x2 = vmulq_f64(start_f64x2, power_f64x2);
    float64x2_t const low_f64x2 = vfmaq_f64(vnegq_f64(high_f64x2), start_f64x2, power_f64x2);
    float64x2_t const quotient_f64x2 = vrndnq_f64(vmulq_n_f64(high_f64x2, 1.0 / (sz_f64_t)sz_overlap_modulus_k));
    // Without an addend, one compensated reduction stays strictly between -p and p.
    float64x2_t const folded_f64x2 = vaddq_f64(vfmsq_f64(high_f64x2, quotient_f64x2, modulus_f64x2), low_f64x2);
    float64x2_t const shifted_f64x2 = vaddq_f64(
        folded_f64x2, vreinterpretq_f64_u64(
                          vandq_u64(vcltq_f64(folded_f64x2, vdupq_n_f64(0)), vreinterpretq_u64_f64(modulus_f64x2))));
    float64x2_t const difference_f64x2 = vsubq_f64(vld1q_f64(end), shifted_f64x2);
    float64x2_t const residue_f64x2 = vaddq_f64(
        difference_f64x2,
        vreinterpretq_f64_u64(vandq_u64(vcltq_f64(difference_f64x2, vdupq_n_f64(0)),
                                        vreinterpretq_u64_f64(vdupq_n_f64((sz_f64_t)sz_overlap_modulus_k)))));
    vst1_u32(hashes, vmovn_u64(vcvtq_u64_f64(residue_f64x2)));
}

/** Extracts the final zero or one window hash. */
STRINGZILLA_INLINE void sz_overlap_f64x2_window_hash_step_tail_neon(sz_f64_t const *start, sz_f64_t const *end,
                                                                    sz_f64_t power, sz_size_t count, sz_u32_t *hashes) {
    if (count) sz_overlap_f64x1_window_hash_step_serial(start, end, power, hashes);
}

STRINGZILLA_INLINE uint32x4_t sz_overlap_reverse_neon_(uint32x4_t keys_u32x4) {
    return vextq_u32(vrev64q_u32(keys_u32x4), vrev64q_u32(keys_u32x4), 2);
}
STRINGZILLA_INLINE uint32x4_t sz_overlap_exchange_within_neon_(uint32x4_t keys_u32x4, sz_size_t distance) {
    uint32x4_t const partner_u32x4 = distance == 1 ? vrev64q_u32(keys_u32x4) : vextq_u32(keys_u32x4, keys_u32x4, 2);
    sz_u32_t const low_masks[2][4] = {{~0u, 0, ~0u, 0}, {~0u, ~0u, 0, 0}};
    return vbslq_u32(vld1q_u32(low_masks[distance - 1]), vminq_u32(keys_u32x4, partner_u32x4),
                     vmaxq_u32(keys_u32x4, partner_u32x4));
}

/** Sorts and deduplicates keys in the buffer sized by @ref sz_overlap_btree_sorted_capacity. */
STRINGZILLA_INLINE sz_size_t sz_overlap_u32x4_btree_sort_neon(sz_u32_t *keys, sz_size_t count) {
    sz_size_t const capacity = sz_overlap_btree_sorted_capacity_(count);
    for (sz_size_t i = count; i != capacity; ++i) keys[i] = sz_overlap_padding_key_k;
    for (sz_size_t i = 0; i != capacity; i += 4) {
        uint32x4_t keys_u32x4 = sz_overlap_exchange_within_neon_(vld1q_u32(keys + i), 1);
        sz_u32_t const mask[4] = {~0u, ~0u, 0, 0};
        uint32x4_t const partner_u32x4 = sz_overlap_reverse_neon_(keys_u32x4);
        keys_u32x4 = vbslq_u32(vld1q_u32(mask), vminq_u32(keys_u32x4, partner_u32x4),
                               vmaxq_u32(keys_u32x4, partner_u32x4));
        vst1q_u32(keys + i, sz_overlap_exchange_within_neon_(keys_u32x4, 1));
    }
    for (sz_size_t phase = 8; phase <= capacity; phase *= 2) {
        for (sz_size_t start = 0; start != capacity; start += phase)
            for (sz_size_t offset = 0; offset != phase / 2; offset += 4) {
                sz_u32_t *lower = keys + start + offset, *upper = keys + start + phase - 4 - offset;
                uint32x4_t const lower_u32x4 = vld1q_u32(lower),
                                 upper_u32x4 = sz_overlap_reverse_neon_(vld1q_u32(upper));
                vst1q_u32(lower, vminq_u32(lower_u32x4, upper_u32x4));
                vst1q_u32(upper, sz_overlap_reverse_neon_(vmaxq_u32(lower_u32x4, upper_u32x4)));
            }
        for (sz_size_t distance = phase / 4; distance >= 4; distance /= 2)
            for (sz_size_t start = 0; start != capacity; start += 2 * distance)
                for (sz_size_t offset = 0; offset != distance; offset += 4) {
                    sz_u32_t *lower = keys + start + offset, *upper = lower + distance;
                    uint32x4_t const lower_u32x4 = vld1q_u32(lower), upper_u32x4 = vld1q_u32(upper);
                    vst1q_u32(lower, vminq_u32(lower_u32x4, upper_u32x4));
                    vst1q_u32(upper, vmaxq_u32(lower_u32x4, upper_u32x4));
                }
        for (sz_size_t i = 0; i != capacity; i += 4)
            vst1q_u32(keys + i,
                      sz_overlap_exchange_within_neon_(sz_overlap_exchange_within_neon_(vld1q_u32(keys + i), 2), 1));
    }
    return sz_overlap_btree_unique_(keys, count);
}
STRINGZILLA_INLINE sz_size_t sz_overlap_branch_step_neon_(sz_u32_t const *node, int32x4_t key_i32x4) {
    uint32x4_t below_u32x4 = vdupq_n_u32(0);
    for (sz_size_t i = 0; i != sz_overlap_keys_per_node_k; i += 4)
        below_u32x4 = vsubq_u32(below_u32x4, vcgtq_s32(key_i32x4, vreinterpretq_s32_u32(vld1q_u32(node + i))));
    return vaddvq_u32(below_u32x4);
}
STRINGZILLA_INLINE sz_size_t sz_overlap_leaf_step_neon_(sz_u32_t const *node, int32x4_t key_i32x4) {
    uint32x4_t equal_u32x4 = vdupq_n_u32(0);
    for (sz_size_t i = 0; i != sz_overlap_keys_per_node_k; i += 4)
        equal_u32x4 = vorrq_u32(equal_u32x4, vceqq_s32(key_i32x4, vreinterpretq_s32_u32(vld1q_u32(node + i))));
    return vmaxvq_u32(equal_u32x4) != 0;
}

/** Counts how many of the candidate keys the tree holds. */
STRINGZILLA_INLINE sz_size_t sz_overlap_u32x4_btree_probe_neon(sz_overlap_btree_t const *btree, sz_u32_t const *keys,
                                                               sz_size_t count) {
    sz_size_t matches = 0;
    for (sz_size_t i = 0; i != count; ++i) {
        int32x4_t const key_i32x4 = vreinterpretq_s32_u32(vdupq_n_u32(keys[i] ^ sz_overlap_sign_flip_k));
        sz_size_t node = 0;
        for (sz_size_t level = 0; level + 1 != btree->levels; ++level)
            node = node * sz_overlap_branches_per_node_k +
                   sz_overlap_branch_step_neon_(
                       btree->nodes + (btree->level_bases[level] + node) * sz_overlap_keys_per_node_k, key_i32x4);
        matches += sz_overlap_leaf_step_neon_(
            btree->nodes + (btree->level_bases[btree->levels - 1] + node) * sz_overlap_keys_per_node_k, key_i32x4);
    }
    return matches;
}
#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_overlap_engine_init_neon(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_unused_(candidates_budget);
    sz_size_t const step = sz_overlap_f64x2_positions_per_step_neon_k;
    sz_allocator_t host;
    if (allocator) host = *allocator;
    else {
        sz_status_t const status = sz_allocator_init_heap(&host);
        if (status != sz_success_k) return status;
    }
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count, 0, &host, stream,
                                                       engine);
    if (opened != sz_success_k) return opened;
    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
    sz_status_t const grown = sz_overlap_engine_grow_(engine, (longest_query + 1) * sizeof(sz_f64_t), stream);
    if (grown != sz_success_k) {
        sz_overlap_engine_close_(engine, stream);
        return grown;
    }
    sz_u32_t *const nodes = (sz_u32_t *)engine->nodes;
    sz_u32_t *const keys_counts = (sz_u32_t *)engine->keys_counts;
    sz_f64_t *const chain = (sz_f64_t *)engine->scratch;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_cptr_t const text = queries->get_start(queries->handle, index);
        sz_size_t const length = engine->lengths[index];
        sz_u32_t *const arena = nodes + engine->nodes_offsets[index];
        chain[0] = 0.0;
        sz_f64_t prior = 0.0;
        sz_size_t position = 0;
        for (; position + step <= length; position += step)
            prior = sz_overlap_f64x2_prefix_hash_step_neon(prior, text + position, chain + position + 1);
        if (position != length)
            sz_overlap_f64x2_prefix_hash_step_tail_neon(prior, text + position, length - position,
                                                        chain + position + 1);
        sz_size_t written = 0;
        for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
            sz_size_t const width = engine->widths[width_index];
            if (!width || width > length) continue;
            sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
            sz_size_t const query_windows = length - width + 1;
            sz_size_t window = 0;
            for (; window + step <= query_windows; window += step)
                sz_overlap_f64x2_window_hash_step_neon(chain + window, chain + window + width, power,
                                                       arena + written + window);
            if (window != query_windows)
                sz_overlap_f64x2_window_hash_step_tail_neon(chain + window, chain + window + width, power,
                                                            query_windows - window, arena + written + window);
            written += query_windows;
        }
        sz_overlap_btree_t btree;
        sz_size_t const distinct = sz_overlap_u32x4_btree_sort_neon(arena, written);
        sz_overlap_btree_prepare_(arena, distinct, &btree);
        keys_counts[index] = (sz_u32_t)distinct;
    }

    engine->capability = sz_cap_neon_k;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_overlap_scores_neon(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                   sz_f32_t *scores, sz_size_t scores_query_stride,
                                                   sz_size_t scores_candidate_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, scores_query_stride,
                                                              scores_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;
    sz_size_t const step = sz_overlap_f64x2_positions_per_step_neon_k;
    sz_size_t const chains = sz_overlap_interleaved_chains_k;

    sz_size_t longest_candidate = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index) {
        sz_size_t const length = candidates->get_length(candidates->handle, index);
        if (length > longest_candidate) longest_candidate = length;
    }
    sz_status_t const grown = sz_overlap_engine_grow_(engine, sz_overlap_engine_round_bytes_(longest_candidate, chains),
                                                      stream);
    if (grown != sz_success_k) return grown;

    sz_size_t const chain_stride = longest_candidate + 1;
    sz_f64_t *const prefix_hashes = (sz_f64_t *)engine->scratch;
    sz_u32_t *const window_hashes = (sz_u32_t *)(prefix_hashes + chain_stride * chains);
    for (sz_size_t first = 0; first < candidates->count; first += chains) {
        sz_size_t const interleaved = candidates->count - first < chains ? candidates->count - first : chains;
        sz_cptr_t texts[sz_overlap_interleaved_chains_k];
        sz_size_t lengths[sz_overlap_interleaved_chains_k];
        sz_f64_t priors[sz_overlap_interleaved_chains_k];
        sz_size_t shortest = STRINGZILLA_SIZE_MAX;
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            texts[chain] = candidates->get_start(candidates->handle, first + chain);
            lengths[chain] = candidates->get_length(candidates->handle, first + chain);
            priors[chain] = prefix_hashes[chain * chain_stride] = 0.0;
            if (lengths[chain] < shortest) shortest = lengths[chain];
        }
        sz_size_t walked = 0;
        if (interleaved == chains)
            for (; walked + step <= shortest; walked += step)
                for (sz_size_t chain = 0; chain != chains; ++chain)
                    priors[chain] = sz_overlap_f64x2_prefix_hash_step_neon(
                        priors[chain], texts[chain] + walked, prefix_hashes + chain * chain_stride + walked + 1);
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_f64_t running = priors[chain];
            sz_size_t own = walked;
            for (; own + step <= lengths[chain]; own += step)
                running = sz_overlap_f64x2_prefix_hash_step_neon(running, texts[chain] + own,
                                                                 chain_prefix_hashes + own + 1);
            if (own != lengths[chain])
                sz_overlap_f64x2_prefix_hash_step_tail_neon(running, texts[chain] + own, lengths[chain] - own,
                                                            chain_prefix_hashes + own + 1);
        }
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            sz_f64_t const *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_f32_t *const candidate_scores = scores + (first + chain) * scores_candidate_stride;
            for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
                sz_size_t const width = engine->widths[width_index];
                sz_size_t const windows = width && width <= length ? length - width + 1 : 0;
                sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
                sz_size_t window = 0;
                for (; window + step <= windows; window += step)
                    sz_overlap_f64x2_window_hash_step_neon(chain_prefix_hashes + window,
                                                           chain_prefix_hashes + window + width, power,
                                                           window_hashes + window);
                if (window != windows)
                    sz_overlap_f64x2_window_hash_step_tail_neon(chain_prefix_hashes + window,
                                                                chain_prefix_hashes + window + width, power,
                                                                windows - window, window_hashes + window);
                for (sz_size_t query = 0; query != engine->count; ++query) {
                    sz_size_t const query_length = engine->lengths[query];
                    sz_f32_t *const slot = candidate_scores + query * scores_query_stride + width_index;
                    if (!windows || width > query_length) {
                        *slot = 0.0f;
                        continue;
                    }
                    sz_overlap_btree_t const btree = sz_overlap_engine_row_(engine, query);
                    sz_size_t const matches = sz_overlap_u32x4_btree_probe_neon(&btree, window_hashes, windows);
                    *slot = sz_overlap_share_(matches, windows, query_length - width + 1);
                }
            }
        }
    }
    return sz_success_k;
}

#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif
#ifdef __cplusplus
}
#endif
#endif
