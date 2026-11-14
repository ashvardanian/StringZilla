/**
 *  @file include/stringzilla/overlap/powervsx.h
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief IBM Power VSX prefix hashes and window overlap scores.
 *
 *  @sa c/stringzilla/overlap.h
 */
#ifndef STRINGZILLA_OVERLAP_POWERVSX_H_
#define STRINGZILLA_OVERLAP_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/overlap/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_PPC64_
#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

enum { sz_overlap_f64x2_positions_per_step_powervsx_k = 2 };

/** Advances the chain over two bytes, writing both prefix hashes. */
STRINGZILLA_INLINE sz_f64_t sz_overlap_f64x2_prefix_hash_step_powervsx(sz_f64_t prior, sz_cptr_t text,
                                                                       sz_f64_t *prefix_hashes) {
    __vector double const chunks_f64x2 = {(sz_f64_t)(sz_u8_t)text[0],
                                          (sz_f64_t)((sz_u32_t)(sz_u8_t)text[0] * 256u + (sz_u8_t)text[1])};
    __vector double const modulus_f64x2 = vec_splats((sz_f64_t)sz_overlap_modulus_k);
    // Power-of-two multipliers keep these integer products exact below 2^48.
    __vector double const product_f64x2 = vec_madd(vec_splats(prior), vec_xl(0, sz_overlap_powers_of_256_k + 1),
                                                   chunks_f64x2);
    __vector double const quotient_f64x2 = vec_rint(
        vec_mul(product_f64x2, vec_splats(1.0 / (sz_f64_t)sz_overlap_modulus_k)));
    __vector double const residue_f64x2 = vec_nmsub(quotient_f64x2, modulus_f64x2, product_f64x2);
    __vector double const reduced_f64x2 = vec_add(
        residue_f64x2, vec_and(modulus_f64x2, (__vector double)vec_cmplt(residue_f64x2, vec_splats(0.0))));
    vec_xst(reduced_f64x2, 0, prefix_hashes);
    return vec_extract(reduced_f64x2, 1);
}

/** Advances the chain over the final zero or one byte. */
STRINGZILLA_INLINE sz_f64_t sz_overlap_f64x2_prefix_hash_step_tail_powervsx(sz_f64_t prior, sz_cptr_t text,
                                                                            sz_size_t count, sz_f64_t *prefix_hashes) {
    return count ? sz_overlap_f64x1_prefix_hash_step_serial_(prior, text, prefix_hashes) : prior;
}

/** Extracts two window hashes from their start and end prefix hashes. */
STRINGZILLA_INLINE void sz_overlap_f64x2_window_hash_step_powervsx(sz_f64_t const *start, sz_f64_t const *end,
                                                                   sz_f64_t power, sz_u32_t *hashes) {
    __vector double const modulus_f64x2 = vec_splats((sz_f64_t)sz_overlap_modulus_k);
    __vector double const zero_f64x2 = vec_splats(0.0);
    __vector double const start_f64x2 = vec_xl(0, start), power_f64x2 = vec_splats(power);
    __vector double const high_f64x2 = vec_mul(start_f64x2, power_f64x2);
    __vector double const low_f64x2 = vec_msub(start_f64x2, power_f64x2, high_f64x2);
    __vector double const quotient_f64x2 = vec_rint(
        vec_mul(high_f64x2, vec_splats(1.0 / (sz_f64_t)sz_overlap_modulus_k)));
    // Without an addend, one compensated reduction stays strictly between -p and p.
    __vector double const folded_f64x2 = vec_add(vec_nmsub(quotient_f64x2, modulus_f64x2, high_f64x2), low_f64x2);
    __vector double const shifted_f64x2 = vec_add(
        folded_f64x2, vec_and(modulus_f64x2, (__vector double)vec_cmplt(folded_f64x2, zero_f64x2)));
    __vector double const difference_f64x2 = vec_sub(vec_xl(0, end), shifted_f64x2);
    __vector double const residue_f64x2 = vec_add(
        difference_f64x2, vec_and(modulus_f64x2, (__vector double)vec_cmplt(difference_f64x2, zero_f64x2)));
    __vector unsigned long long const residue_u64x2 = vec_unsigned(residue_f64x2);
    vec_xst_len((__vector unsigned char)vec_pack(residue_u64x2, residue_u64x2), (unsigned char *)hashes,
                2 * sizeof(sz_u32_t));
}

/** Extracts the final zero or one window hash. */
STRINGZILLA_INLINE void sz_overlap_f64x2_window_hash_step_tail_powervsx(sz_f64_t const *start, sz_f64_t const *end,
                                                                        sz_f64_t power, sz_size_t count,
                                                                        sz_u32_t *hashes) {
    if (count) sz_overlap_f64x1_window_hash_step_serial(start, end, power, hashes);
}

/** Orders each pair of keys @p distance lanes apart, the lower lane taking the smaller: a 32-bit
 *  rotation of each doubleword swaps neighbours, and a half rotation swaps the doublewords. */
STRINGZILLA_INLINE __vector unsigned int sz_overlap_exchange_within_powervsx_(__vector unsigned int keys_u32x4,
                                                                              sz_size_t distance) {
    __vector unsigned int const partner_u32x4 = distance == 1
                                                    ? (__vector unsigned int)vec_rl(
                                                          (__vector unsigned long long)keys_u32x4, vec_splats(32ull))
                                                    : vec_sld(keys_u32x4, keys_u32x4, 8);
    __vector unsigned int const low_mask_u32x4 = distance == 1 ? (__vector unsigned int) {~0u, 0u, ~0u, 0u}
                                                               : (__vector unsigned int) {~0u, ~0u, 0u, 0u};
    return vec_sel(vec_max(keys_u32x4, partner_u32x4), vec_min(keys_u32x4, partner_u32x4), low_mask_u32x4);
}

/** Sorts and deduplicates keys in the buffer sized by @ref sz_overlap_btree_sorted_capacity. */
STRINGZILLA_INLINE sz_size_t sz_overlap_u32x4_btree_sort_powervsx(sz_u32_t *keys, sz_size_t count) {
    sz_size_t const capacity = sz_overlap_btree_sorted_capacity_(count);
    for (sz_size_t i = count; i != capacity; ++i) keys[i] = sz_overlap_padding_key_k;
    __vector unsigned int const lower_half_u32x4 = {~0u, ~0u, 0u, 0u};
    for (sz_size_t i = 0; i != capacity; i += 4) {
        __vector unsigned int keys_u32x4 = sz_overlap_exchange_within_powervsx_(vec_xl(0, keys + i), 1);
        __vector unsigned int const partner_u32x4 = vec_reve(keys_u32x4);
        keys_u32x4 = vec_sel(vec_max(keys_u32x4, partner_u32x4), vec_min(keys_u32x4, partner_u32x4), lower_half_u32x4);
        vec_xst(sz_overlap_exchange_within_powervsx_(keys_u32x4, 1), 0, keys + i);
    }
    for (sz_size_t phase = 8; phase <= capacity; phase *= 2) {
        for (sz_size_t start = 0; start != capacity; start += phase)
            for (sz_size_t offset = 0; offset != phase / 2; offset += 4) {
                sz_u32_t *lower = keys + start + offset, *upper = keys + start + phase - 4 - offset;
                __vector unsigned int const lower_u32x4 = vec_xl(0, lower), upper_u32x4 = vec_reve(vec_xl(0, upper));
                vec_xst(vec_min(lower_u32x4, upper_u32x4), 0, lower);
                vec_xst(vec_reve(vec_max(lower_u32x4, upper_u32x4)), 0, upper);
            }
        for (sz_size_t distance = phase / 4; distance >= 4; distance /= 2)
            for (sz_size_t start = 0; start != capacity; start += 2 * distance)
                for (sz_size_t offset = 0; offset != distance; offset += 4) {
                    sz_u32_t *lower = keys + start + offset, *upper = lower + distance;
                    __vector unsigned int const lower_u32x4 = vec_xl(0, lower), upper_u32x4 = vec_xl(0, upper);
                    vec_xst(vec_min(lower_u32x4, upper_u32x4), 0, lower);
                    vec_xst(vec_max(lower_u32x4, upper_u32x4), 0, upper);
                }
        for (sz_size_t i = 0; i != capacity; i += 4)
            vec_xst(
                sz_overlap_exchange_within_powervsx_(sz_overlap_exchange_within_powervsx_(vec_xl(0, keys + i), 2), 1),
                0, keys + i);
    }
    return sz_overlap_btree_unique_(keys, count);
}

/** The child a branch @p node routes @p key_i32x4 to: its keys ascend, so the count below the key
 *  is the first lane not below it, which one POWER9 @c vclzlsbb finds over the packed compares. */
STRINGZILLA_INLINE sz_size_t sz_overlap_branch_step_powervsx_(sz_u32_t const *node, __vector signed int key_i32x4) {
    __vector unsigned int below_u32x4[sz_overlap_keys_per_node_k / 4];
    for (sz_size_t i = 0; i != sz_overlap_keys_per_node_k; i += 4)
        below_u32x4[i / 4] = (__vector unsigned int)vec_cmpgt(key_i32x4, (__vector signed int)vec_xl(0, node + i));
    __vector unsigned char const below_u8x16 = vec_pack(vec_pack(below_u32x4[0], below_u32x4[1]),
                                                        vec_pack(below_u32x4[2], below_u32x4[3]));
    return (sz_size_t)vec_cntlz_lsbb(vec_nor(below_u8x16, below_u8x16));
}

/** Whether a leaf @p node holds @p key_i32x4, by one predicate compare over the merged matches. */
STRINGZILLA_INLINE sz_size_t sz_overlap_leaf_step_powervsx_(sz_u32_t const *node, __vector signed int key_i32x4) {
    __vector unsigned int equal_u32x4 = vec_splats(0u);
    for (sz_size_t i = 0; i != sz_overlap_keys_per_node_k; i += 4)
        equal_u32x4 = vec_or(equal_u32x4,
                             (__vector unsigned int)vec_cmpeq(key_i32x4, (__vector signed int)vec_xl(0, node + i)));
    return vec_any_ne(equal_u32x4, vec_splats(0u));
}

/** Counts how many of the candidate keys the tree holds. */
STRINGZILLA_INLINE sz_size_t sz_overlap_u32x4_btree_probe_powervsx(sz_overlap_btree_t const *btree,
                                                                   sz_u32_t const *keys, sz_size_t count) {
    sz_size_t matches = 0;
    for (sz_size_t i = 0; i != count; ++i) {
        __vector signed int const key_i32x4 = vec_splats((signed int)(keys[i] ^ sz_overlap_sign_flip_k));
        sz_size_t node = 0;
        for (sz_size_t level = 0; level + 1 != btree->levels; ++level)
            node = node * sz_overlap_branches_per_node_k +
                   sz_overlap_branch_step_powervsx_(
                       btree->nodes + (btree->level_bases[level] + node) * sz_overlap_keys_per_node_k, key_i32x4);
        matches += sz_overlap_leaf_step_powervsx_(
            btree->nodes + (btree->level_bases[btree->levels - 1] + node) * sz_overlap_keys_per_node_k, key_i32x4);
    }
    return matches;
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_powervsx(sz_overlap_engine_t *engine,
                                                            sz_strings_view_t const *queries,
                                                            sz_size_t const *window_widths,
                                                            sz_size_t window_widths_count, sz_size_t candidates_budget,
                                                            sz_allocator_t *allocator, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_unused_(candidates_budget);
    sz_size_t const step = sz_overlap_f64x2_positions_per_step_powervsx_k;
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
    sz_string_view_t views[64];
    for (sz_size_t index = 0; index != engine->count; ++index) {
        if (index % 64 == 0) sz_strings_get_block_(queries, index, views);
        sz_cptr_t const text = views[index % 64].start;
        sz_size_t const length = engine->lengths[index];
        sz_u32_t *const arena = nodes + engine->nodes_offsets[index];
        chain[0] = 0.0;
        sz_f64_t prior = 0.0;
        sz_size_t position = 0;
        for (; position + step <= length; position += step)
            prior = sz_overlap_f64x2_prefix_hash_step_powervsx(prior, text + position, chain + position + 1);
        if (position != length)
            sz_overlap_f64x2_prefix_hash_step_tail_powervsx(prior, text + position, length - position,
                                                            chain + position + 1);
        sz_size_t written = 0;
        for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
            sz_size_t const width = engine->widths[width_index];
            if (!width || width > length) continue;
            sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
            sz_size_t const query_windows = length - width + 1;
            sz_size_t window = 0;
            for (; window + step <= query_windows; window += step)
                sz_overlap_f64x2_window_hash_step_powervsx(chain + window, chain + window + width, power,
                                                           arena + written + window);
            if (window != query_windows)
                sz_overlap_f64x2_window_hash_step_tail_powervsx(chain + window, chain + window + width, power,
                                                                query_windows - window, arena + written + window);
            written += query_windows;
        }
        sz_overlap_btree_t btree;
        sz_size_t const distinct = sz_overlap_u32x4_btree_sort_powervsx(arena, written);
        sz_overlap_btree_prepare_(arena, distinct, &btree);
        keys_counts[index] = (sz_u32_t)distinct;
    }

    engine->capability = sz_cap_powervsx_k;
    return sz_success_k;
}

/** Hashes the candidates of @p round, four chains at a time, and scores each as @p round says. */
STRINGZILLA_INLINE void sz_overlap_round_powervsx_(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                   sz_size_t longest_candidate, sz_overlap_round_t const *round) {
    sz_size_t const step = sz_overlap_f64x2_positions_per_step_powervsx_k;
    sz_size_t const chains = sz_overlap_interleaved_chains_k;
    sz_size_t const chain_stride = longest_candidate + 1;
    sz_f64_t *const prefix_hashes = (sz_f64_t *)engine->scratch;
    sz_u32_t *const window_hashes = (sz_u32_t *)(prefix_hashes + chain_stride * chains);

    // Four candidates' chains advance together, each carrying its own prior, so their latency-bound
    // steps overlap. A short last group repeats its first candidate in the spare chains, so every
    // group runs the same four-chain loop.
    for (sz_size_t first = 0; first < round->count; first += chains) {
        sz_size_t const interleaved = sz_min_of_two(round->count - first, chains);
        sz_cptr_t texts[sz_overlap_interleaved_chains_k];
        sz_size_t lengths[sz_overlap_interleaved_chains_k];
        sz_f64_t priors[sz_overlap_interleaved_chains_k];
        sz_size_t indices[sz_overlap_interleaved_chains_k];
        sz_string_view_t views[sz_overlap_interleaved_chains_k];
        sz_size_t chain, shortest = STRINGZILLA_SIZE_MAX;
        for (chain = 0; chain != chains; ++chain)
            indices[chain] = round->order[first + (chain < interleaved ? chain : 0)];
        sz_strings_get_(candidates, indices, chains, views);
        for (chain = 0; chain != chains; ++chain) {
            texts[chain] = views[chain].start, lengths[chain] = views[chain].length;
            priors[chain] = prefix_hashes[chain * chain_stride] = 0.0;
            shortest = sz_min_of_two(shortest, lengths[chain]);
        }
        sz_size_t walked = 0;
        for (; walked + step <= shortest; walked += step)
            for (chain = 0; chain != chains; ++chain)
                priors[chain] = sz_overlap_f64x2_prefix_hash_step_powervsx(
                    priors[chain], texts[chain] + walked, prefix_hashes + chain * chain_stride + walked + 1);

        // One candidate's window hashes at every width serve every query's tree, so the hashing
        // runs once here and the probes run once per query over what it wrote.
        for (chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_f64_t running = priors[chain];
            sz_size_t own = walked;
            for (; own + step <= length; own += step)
                running = sz_overlap_f64x2_prefix_hash_step_powervsx(running, texts[chain] + own,
                                                                     chain_prefix_hashes + own + 1);
            if (own != length)
                sz_overlap_f64x2_prefix_hash_step_tail_powervsx(running, texts[chain] + own, length - own,
                                                                chain_prefix_hashes + own + 1);
            sz_u32_t *keys = window_hashes;
            for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
                sz_size_t const width = engine->widths[width_index];
                sz_size_t const windows = width && width <= length ? length - width + 1 : 0;
                sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
                sz_size_t window = 0;
                for (; window + step <= windows; window += step)
                    sz_overlap_f64x2_window_hash_step_powervsx(
                        chain_prefix_hashes + window, chain_prefix_hashes + window + width, power, keys + window);
                if (window != windows)
                    sz_overlap_f64x2_window_hash_step_tail_powervsx(chain_prefix_hashes + window,
                                                                    chain_prefix_hashes + window + width, power,
                                                                    windows - window, keys + window);
                keys += windows;
            }
            sz_overlap_round_score_(engine, round, indices[chain], window_hashes, length,
                                    sz_overlap_u32x4_btree_probe_powervsx);
        }
    }
}

STRINGZILLA_API sz_status_t sz_overlap_scores_powervsx(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                       sz_f32_t radius, sz_u32_t *matches,
                                                       sz_size_t matches_query_stride,
                                                       sz_size_t matches_candidate_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_overlap_round_t round;
    sz_size_t longest_candidate = 0;
    sz_status_t const prepared = sz_overlap_scores_round_(engine, candidates, radius, matches, matches_query_stride,
                                                          matches_candidate_stride, sz_overlap_interleaved_chains_k,
                                                          stream, &round, &longest_candidate);
    if (prepared != sz_success_k) return prepared;
    sz_overlap_round_powervsx_(engine, candidates, longest_candidate, &round);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_POWERVSX
#endif // STRINGZILLA_ARCH_PPC64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_OVERLAP_POWERVSX_H_
