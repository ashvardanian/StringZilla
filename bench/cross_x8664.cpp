/**
 *  @file bench/cross_x8664.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel benchmarks - x86-64 family: Westmere, Goldmont, Haswell, Skylake, Ice Lake.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

/*  The Levenshtein and overlap steps and the pgram sorts are always inlined under their
 *  capability's target, so each adapter below compiles under the same target as the tier header it
 *  calls into, and the overlap steps and the sorts reach the portable drivers as plain calls,
 *  which a driver without that target can make. */

#if STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,bmi,bmi2,lzcnt,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "bmi", "bmi2", "lzcnt", "popcnt")
#endif

/** The serial step's eight lanes, four per YMM, so the two arms answer the same scores. */
struct levenshtein_step_from_haswell {
    static constexpr std::size_t groups_k = levenshtein_step_lanes_k /
                                            sz_levenshtein_u64x4_candidates_per_step_haswell_k;

    /** Query words every step walks. */
    std::size_t words = 0;

    /** Match masks the query points at. */
    std::vector<sz_u64_t> masks;

    /** @b [256] mask row each byte reads. */
    std::vector<sz_u8_t> byte_to_class;

    /** The query every lane is scored against. */
    sz_levenshtein_query_t query {};

    /** @b [positions,lanes] staged class ids. */
    std::vector<sz_u8_t> classes;

    /** @b [groups,words] Myers deltas. */
    std::vector<sz_levenshtein_u64x4_vertical_haswell_t> verticals;

    levenshtein_step_from_haswell(corpus_t const &corpus, std::size_t query_bytes)
        : masks(sz_levenshtein_query_mask_entries(query_bytes)), byte_to_class(sz_levenshtein_byte_classes_k) {
        std::string_view const text = levenshtein_query_token(corpus, query_bytes);
        if (sz_levenshtein_query_prepare(text.data(), text.size(), masks.data(), byte_to_class.data(), &query) !=
            sz_success_k)
            throw std::runtime_error("The query preparation failed.");
        words = sz_levenshtein_query_words(text.size());
        classes = levenshtein_staged_classes(corpus, byte_to_class.data(), levenshtein_step_lanes_k,
                                             levenshtein_step_positions_k);
        verticals.resize(groups_k * words);
    }

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        sz_levenshtein_u64x4_state_haswell_t states[groups_k];
        for (std::size_t group = 0; group != groups_k; ++group)
            sz_levenshtein_u64x4_init_haswell(&states[group], verticals.data() + group * words, words, &query);
        for (std::size_t position = 0; position != levenshtein_step_positions_k; ++position)
            for (std::size_t group = 0; group != groups_k; ++group)
                sz_levenshtein_u64x4_step_haswell(&states[group], verticals.data() + group * words, words, &query,
                                                  sz_levenshtein_u64x4_classes_u8_haswell(
                                                      classes.data() + position * levenshtein_step_lanes_k +
                                                      group * sz_levenshtein_u64x4_candidates_per_step_haswell_k));
        check_value_t mixed = 0;
        for (std::size_t group = 0; group != groups_k; ++group)
            for (std::size_t lane = 0; lane != sz_levenshtein_u64x4_candidates_per_step_haswell_k; ++lane)
                mixed = mixed * 31u + sz_levenshtein_u64x4_score_haswell(&states[group], lane);
        call_result_t result(levenshtein_step_positions_k * levenshtein_step_lanes_k, mixed,
                             levenshtein_step_positions_k * levenshtein_step_lanes_k * words);
        return result;
    }
};

sz_f64_t overlap_prefix_hash_step_haswell_(sz_f64_t prior, sz_cptr_t text, sz_f64_t *prefix_hashes) {
    return sz_overlap_f64x4_prefix_hash_step_haswell(prior, text, prefix_hashes);
}
sz_f64_t overlap_prefix_hash_step_tail_haswell_(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                sz_f64_t *prefix_hashes) {
    return sz_overlap_f64x4_prefix_hash_step_tail_haswell(prior, text, count, prefix_hashes);
}
void overlap_window_hash_step_haswell_(sz_f64_t const *prefix_hashes_at_start, sz_f64_t const *prefix_hashes_at_end,
                                       sz_f64_t window_power, sz_u32_t *window_hashes) {
    sz_overlap_f64x4_window_hash_step_haswell(prefix_hashes_at_start, prefix_hashes_at_end, window_power,
                                              window_hashes);
}
void overlap_window_hash_step_tail_haswell_(sz_f64_t const *prefix_hashes_at_start,
                                            sz_f64_t const *prefix_hashes_at_end, sz_f64_t window_power,
                                            sz_size_t count, sz_u32_t *window_hashes) {
    sz_overlap_f64x4_window_hash_step_tail_haswell(prefix_hashes_at_start, prefix_hashes_at_end, window_power, count,
                                                   window_hashes);
}
sz_size_t overlap_btree_sort_haswell_(sz_u32_t *keys, sz_size_t count) {
    return sz_overlap_u32x8_btree_sort_haswell(keys, count);
}
sz_size_t overlap_btree_probe_haswell_(sz_overlap_btree_t const *btree, sz_u32_t const *keys, sz_size_t count) {
    return sz_overlap_u32x8_btree_probe_haswell(btree, keys, count);
}
sz_status_t pgrams_sort_haswell_(sz_pgram_t *pgrams, sz_size_t count, sz_allocator_t *allocator,
                                 sz_sorted_idx_t *order) {
    return sz_pgrams_sort_haswell_(pgrams, count, allocator, order);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_SKYLAKE
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(                                                                       \
    __attribute__((target("avx,avx2,avx512f,avx512vl,avx512dq,avx512bw,bmi,bmi2,popcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx2,avx512f,avx512vl,avx512dq,avx512bw,bmi,bmi2,popcnt"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx2", "avx512f", "avx512vl", "avx512dq", "avx512bw", "bmi", "bmi2", "popcnt")
#endif

/** The serial step's eight lanes, eight per ZMM, so the two arms answer the same scores. */
struct levenshtein_step_from_skylake {

    /** Query words every step walks. */
    std::size_t words = 0;

    /** Match masks the query points at. */
    std::vector<sz_u64_t> masks;

    /** @b [256] mask row each byte reads. */
    std::vector<sz_u8_t> byte_to_class;

    /** The query every lane is scored against. */
    sz_levenshtein_query_t query {};

    /** @b [positions,lanes] staged class ids. */
    std::vector<sz_u8_t> classes;

    /** @b [words] Myers deltas of eight lanes. */
    std::vector<sz_levenshtein_u64x8_vertical_skylake_t> verticals;

    levenshtein_step_from_skylake(corpus_t const &corpus, std::size_t query_bytes)
        : masks(sz_levenshtein_query_mask_entries(query_bytes)), byte_to_class(sz_levenshtein_byte_classes_k) {
        std::string_view const text = levenshtein_query_token(corpus, query_bytes);
        if (sz_levenshtein_query_prepare(text.data(), text.size(), masks.data(), byte_to_class.data(), &query) !=
            sz_success_k)
            throw std::runtime_error("The query preparation failed.");
        words = sz_levenshtein_query_words(text.size());
        classes = levenshtein_staged_classes(corpus, byte_to_class.data(), levenshtein_step_lanes_k,
                                             levenshtein_step_positions_k);
        verticals.resize(words);
    }

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        sz_levenshtein_u64x8_state_skylake_t state;
        sz_levenshtein_u64x8_init_skylake(&state, verticals.data(), words, &query);
        for (std::size_t position = 0; position != levenshtein_step_positions_k; ++position)
            sz_levenshtein_u64x8_step_skylake(
                &state, verticals.data(), words, &query,
                sz_levenshtein_u64x8_classes_u8_skylake(classes.data() + position * levenshtein_step_lanes_k));
        check_value_t mixed = 0;
        for (std::size_t lane = 0; lane != levenshtein_step_lanes_k; ++lane)
            mixed = mixed * 31u + sz_levenshtein_u64x8_score_skylake(&state, lane);
        call_result_t result(levenshtein_step_positions_k * levenshtein_step_lanes_k, mixed,
                             levenshtein_step_positions_k * levenshtein_step_lanes_k * words);
        return result;
    }
};

sz_f64_t overlap_prefix_hash_step_skylake_(sz_f64_t prior, sz_cptr_t text, sz_f64_t *prefix_hashes) {
    return sz_overlap_f64x8_prefix_hash_step_skylake(prior, text, prefix_hashes);
}
sz_f64_t overlap_prefix_hash_step_tail_skylake_(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                sz_f64_t *prefix_hashes) {
    return sz_overlap_f64x8_prefix_hash_step_tail_skylake(prior, text, count, prefix_hashes);
}
void overlap_window_hash_step_skylake_(sz_f64_t const *prefix_hashes_at_start, sz_f64_t const *prefix_hashes_at_end,
                                       sz_f64_t window_power, sz_u32_t *window_hashes) {
    sz_overlap_f64x8_window_hash_step_skylake(prefix_hashes_at_start, prefix_hashes_at_end, window_power,
                                              window_hashes);
}
void overlap_window_hash_step_tail_skylake_(sz_f64_t const *prefix_hashes_at_start,
                                            sz_f64_t const *prefix_hashes_at_end, sz_f64_t window_power,
                                            sz_size_t count, sz_u32_t *window_hashes) {
    sz_overlap_f64x8_window_hash_step_tail_skylake(prefix_hashes_at_start, prefix_hashes_at_end, window_power, count,
                                                   window_hashes);
}
sz_size_t overlap_btree_sort_skylake_(sz_u32_t *keys, sz_size_t count) {
    return sz_overlap_u32x16_btree_sort_skylake(keys, count);
}
sz_size_t overlap_btree_probe_skylake_(sz_overlap_btree_t const *btree, sz_u32_t const *keys, sz_size_t count) {
    return sz_overlap_u32x16_btree_probe_skylake(btree, keys, count);
}
sz_status_t pgrams_sort_skylake_(sz_pgram_t *pgrams, sz_size_t count, sz_allocator_t *allocator,
                                 sz_sorted_idx_t *order) {
    return sz_pgrams_sort_skylake_(pgrams, count, allocator, order);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_ICELAKE
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(                                                                                        \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "lzcnt")
#endif

/**
 *  @brief Sixty-four byte lanes stepped per call, of which the first eight carry the serial arm's
 *      lanes, so the two answer the same scores while the timing covers every lane.
 *
 *  Only a query of at most eight symbols fits a byte lane, so the arm is registered at those
 *  lengths alone, and the deltas are folded into the scores on the cadence the sweep uses rather
 *  than on a per-position branch.
 */
struct levenshtein_step_from_icelake_narrow {
    static constexpr std::size_t lanes_k = sz_levenshtein_u8x64_candidates_per_step_icelake_k;
    static constexpr std::size_t positions_per_flush_k = sz_levenshtein_u8x64_positions_per_flush_icelake_k;

    /** Match masks the query points at. */
    std::vector<sz_u64_t> masks;

    /** @b [256] mask row each byte reads. */
    std::vector<sz_u8_t> byte_to_class;

    /** The query every lane is scored against. */
    sz_levenshtein_query_t query {};

    /** The same query as one mask byte per class. */
    sz_levenshtein_u8x64_query_icelake_t packed {};

    /** @b [positions,lanes] staged class ids. */
    std::vector<sz_u8_t> classes;

    /** @b [lanes] running distances the deltas fold into. */
    std::vector<sz_size_t> scores;

    levenshtein_step_from_icelake_narrow(corpus_t const &corpus, std::size_t query_bytes)
        : masks(sz_levenshtein_query_mask_entries(query_bytes)), byte_to_class(sz_levenshtein_byte_classes_k),
          scores(lanes_k) {
        std::string_view const text = levenshtein_query_token(corpus, query_bytes);
        if (sz_levenshtein_query_prepare(text.data(), text.size(), masks.data(), byte_to_class.data(), &query) !=
            sz_success_k)
            throw std::runtime_error("The query preparation failed.");
        sz_levenshtein_u8x64_pack_icelake(&query, &packed);
        classes = levenshtein_staged_classes(corpus, byte_to_class.data(), lanes_k, levenshtein_step_positions_k);
    }

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        sz_levenshtein_u8x64_state_icelake_t state;
        sz_levenshtein_u8x64_vertical_icelake_t vertical;
        sz_levenshtein_u8x64_init_icelake(&state, &vertical);
        for (std::size_t lane = 0; lane != lanes_k; ++lane) scores[lane] = query.length;
        for (std::size_t flushed = 0; flushed != levenshtein_step_positions_k; flushed += positions_per_flush_k) {
            for (std::size_t taken = 0; taken != positions_per_flush_k; ++taken)
                sz_levenshtein_u8x64_step_icelake(
                    &state, &vertical, &packed,
                    sz_levenshtein_u8x64_classes_u8_icelake(classes.data() + (flushed + taken) * lanes_k));
            sz_levenshtein_u8x64_flush_icelake(&state, scores.data());
        }
        check_value_t mixed = 0;
        for (std::size_t lane = 0; lane != levenshtein_step_lanes_k; ++lane) mixed = mixed * 31u + scores[lane];
        call_result_t result(levenshtein_step_positions_k * lanes_k, mixed, levenshtein_step_positions_k * lanes_k);
        return result;
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_ICELAKE

void bench_cross_x8664([[maybe_unused]] environment_t &env) {
#if STRINGZILLA_TARGET_WESTMERE
    if (section(env, "Cross Westmere", sz_cap_westmere_k)) {
        bench_find_kernels<sz_find_westmere, sz_rfind_westmere>(env, "westmere");
        bench_find_byte_kernels<sz_find_byte_westmere, sz_rfind_byte_westmere>(env, "westmere");
        bench_hash_kernels<sz_hash_westmere>(env, "westmere");
        bench_hash_multiseed_kernels<sz_hash_multiseed_westmere>(env, "westmere");
        bench_hash_stream_kernels<sz_hash_state_init_westmere, sz_hash_state_update_westmere,
                                  sz_hash_state_digest_westmere>(env, "westmere");
        bench_equal_kernels<sz_equal_westmere>(env, "westmere");
        bench_order_kernels<sz_order_westmere>(env, "westmere");
        bench_fill_random_kernels<sz_fill_random_westmere>(env, "westmere");
        bench_aes256_ctr_kernels<sz_aes256_key_init_westmere, sz_aes256_ctr_xor_westmere>(env, "westmere");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_westmere, sz_aes256_gcm_encrypt_westmere>(env, "westmere");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_westmere, sz_aes256_gcm_encryptor_init_westmere,
                                        sz_aes256_gcm_encryptor_update_westmere,
                                        sz_aes256_gcm_encryptor_digest_westmere>(env, "westmere");
        bench_sequence_intersect_kernels<sz_sequence_intersect_westmere>(env, "westmere");
    }
#endif // STRINGZILLA_TARGET_WESTMERE
#if STRINGZILLA_TARGET_GOLDMONT
    if (section(env, "Cross Goldmont", sz_cap_goldmont_k)) {
        bench_sha256_kernels<sz_sha256_state_init_goldmont, sz_sha256_state_update_goldmont,
                             sz_sha256_state_digest_goldmont>(env, "goldmont");
        bench_sha256_multistate_kernels<sz_sha256_multistate_update_goldmont, sz_sha256_multistate_digest_goldmont>(
            env, "goldmont");
    }
#endif // STRINGZILLA_TARGET_GOLDMONT
#if STRINGZILLA_TARGET_HASWELL
    if (section(env, "Cross Haswell", sz_cap_haswell_k)) {
        bench_find_kernels<sz_find_haswell, sz_rfind_haswell>(env, "haswell");
        bench_find_byte_kernels<sz_find_byte_haswell, sz_rfind_byte_haswell>(env, "haswell");
        bench_find_byteset_kernels<sz_find_byteset_haswell, sz_rfind_byteset_haswell>(env, "haswell");
        bench_utf8_count_kernels<sz_utf8_count_haswell>(env, "haswell");
        bench_utf8_seek_kernels<sz_utf8_seek_haswell>(env, "haswell");
        bench_utf8_decode_kernels<sz_utf8_decode_haswell>(env, "haswell");
        bench_utf8_newlines_kernels<sz_utf8_newlines_haswell>(env, "haswell");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_haswell>(env, "haswell");
        bench_utf8_delimiters_kernels<sz_utf8_delimiters_haswell>(env, "haswell");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_haswell>(env, "haswell");
        bench_utf8_graphemes_kernels<sz_utf8_graphemes_haswell>(env, "haswell");
        bench_utf8_sentences_kernels<sz_utf8_sentences_haswell>(env, "haswell");
        bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_haswell>(env, "haswell");
        bench_utf8_norm_kernels<sz_utf8_norm_haswell>(env, "haswell");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_haswell>(env, "haswell");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_haswell>(env, "haswell");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_haswell>(env, "haswell");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_haswell>(env, "haswell");
        bench_bytesum_kernels<sz_bytesum_haswell>(env, "haswell");
        bench_sha256_multistate_kernels<sz_sha256_multistate_update_haswell, sz_sha256_multistate_digest_haswell>(
            env, "haswell");
        bench_equal_kernels<sz_equal_haswell>(env, "haswell");
        bench_order_kernels<sz_order_haswell>(env, "haswell");
        bench_copy_kernels<sz_copy_haswell>(env, "haswell");
        bench_move_kernels<sz_move_haswell>(env, "haswell");
        bench_fill_kernels<sz_fill_haswell>(env, "haswell");
        bench_lookup_kernels<sz_lookup_haswell>(env, "haswell");
        bench_map_kernels<sz_order_haswell>(env, "haswell");
#if STRINGZILLA_TARGET_WESTMERE
        // No AVX2 hasher exists, so the fastest pairing takes a Westmere hash, which needs AES-NI.
        if (sz::device_t::cpu().capabilities_enabled().value & sz_cap_westmere_k)
            bench_unordered_map_kernels<sz_hash_westmere, sz_equal_haswell>(env, "westmere", "haswell");
#endif
        bench_sequence_argsort_kernels<sz_sequence_argsort_haswell, sz_sequence_argsort_uncased_haswell>(env,
                                                                                                         "haswell");
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_haswell, sz_levenshtein_distances_haswell>(
            env, "haswell", sz_levenshtein_bytes_k);
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_haswell, sz_levenshtein_distances_haswell>(
            env, "haswell", sz_levenshtein_runes_k);
        bench_overlap_scores_kernels<sz_overlap_engine_init_haswell, sz_overlap_scores_haswell>(env, "haswell");
        bench_substrings_kernels<sz_substrings_engine_init_haswell, sz_substrings_counts_haswell,
                                 sz_substrings_find_haswell, sz_substrings_replace_haswell,
                                 sz_substrings_bm25_scores_haswell>(env, "haswell");
#if STRINGZILLA_HEADER_ONLY
        bench_pgrams_sort_kernels<pgrams_sort_haswell_>(env, "haswell");
        bench_levenshtein_step_kernels<levenshtein_step_from_haswell>(env, "sz_levenshtein_u64x4_step_haswell");
        bench_overlap_step_kernels<sz_overlap_f64x4_positions_per_step_haswell_k, overlap_prefix_hash_step_haswell_,
                                   overlap_prefix_hash_step_tail_haswell_, overlap_window_hash_step_haswell_,
                                   overlap_window_hash_step_tail_haswell_, overlap_btree_sort_haswell_,
                                   overlap_btree_probe_haswell_>(env, "haswell");
#endif
    }
#endif // STRINGZILLA_TARGET_HASWELL
#if STRINGZILLA_TARGET_SKYLAKE
    if (section(env, "Cross Skylake", sz_cap_skylake_k)) {
        bench_find_kernels<sz_find_skylake, sz_rfind_skylake>(env, "skylake");
        bench_find_byte_kernels<sz_find_byte_skylake, sz_rfind_byte_skylake>(env, "skylake");
        bench_utf8_norm_kernels<sz_utf8_norm_skylake>(env, "skylake");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_skylake>(env, "skylake");
        bench_bytesum_kernels<sz_bytesum_skylake>(env, "skylake");
        bench_hash_kernels<sz_hash_skylake>(env, "skylake");
        bench_hash_stream_kernels<sz_hash_state_init_skylake, sz_hash_state_update_skylake,
                                  sz_hash_state_digest_skylake>(env, "skylake");
        bench_sha256_multistate_kernels<sz_sha256_multistate_update_skylake, sz_sha256_multistate_digest_skylake>(
            env, "skylake");
        bench_equal_kernels<sz_equal_skylake>(env, "skylake");
        bench_order_kernels<sz_order_skylake>(env, "skylake");
        bench_copy_kernels<sz_copy_skylake>(env, "skylake");
        bench_move_kernels<sz_move_skylake>(env, "skylake");
        bench_fill_kernels<sz_fill_skylake>(env, "skylake");
        bench_fill_random_kernels<sz_fill_random_skylake>(env, "skylake");
        bench_map_kernels<sz_order_skylake>(env, "skylake");
        bench_unordered_map_kernels<sz_hash_skylake, sz_equal_skylake>(env, "skylake", "skylake");
        bench_sequence_argsort_kernels<sz_sequence_argsort_skylake, sz_sequence_argsort_uncased_skylake>(env,
                                                                                                         "skylake");
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_skylake, sz_levenshtein_distances_skylake>(
            env, "skylake", sz_levenshtein_bytes_k);
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_skylake, sz_levenshtein_distances_skylake>(
            env, "skylake", sz_levenshtein_runes_k);
        bench_overlap_scores_kernels<sz_overlap_engine_init_skylake, sz_overlap_scores_skylake>(env, "skylake");
#if STRINGZILLA_HEADER_ONLY
        bench_pgrams_sort_kernels<pgrams_sort_skylake_>(env, "skylake");
        bench_levenshtein_step_kernels<levenshtein_step_from_skylake>(env, "sz_levenshtein_u64x8_step_skylake");
        bench_overlap_step_kernels<sz_overlap_f64x8_positions_per_step_skylake_k, overlap_prefix_hash_step_skylake_,
                                   overlap_prefix_hash_step_tail_skylake_, overlap_window_hash_step_skylake_,
                                   overlap_window_hash_step_tail_skylake_, overlap_btree_sort_skylake_,
                                   overlap_btree_probe_skylake_>(env, "skylake");
#endif
    }
#endif // STRINGZILLA_TARGET_SKYLAKE
#if STRINGZILLA_TARGET_ICELAKE
    if (section(env, "Cross Ice Lake", sz_cap_icelake_k)) {
        bench_find_byteset_kernels<sz_find_byteset_icelake, sz_rfind_byteset_icelake>(env, "icelake");
        bench_utf8_count_kernels<sz_utf8_count_icelake>(env, "icelake");
        bench_utf8_seek_kernels<sz_utf8_seek_icelake>(env, "icelake");
        bench_utf8_decode_kernels<sz_utf8_decode_icelake>(env, "icelake");
        bench_utf8_newlines_kernels<sz_utf8_newlines_icelake>(env, "icelake");
        bench_utf8_whitespaces_kernels<sz_utf8_whitespaces_icelake>(env, "icelake");
        bench_utf8_delimiters_kernels<sz_utf8_delimiters_icelake>(env, "icelake");
        bench_utf8_wordbreaks_kernels<sz_utf8_wordbreaks_icelake>(env, "icelake");
        bench_utf8_graphemes_kernels<sz_utf8_graphemes_icelake>(env, "icelake");
        bench_utf8_sentences_kernels<sz_utf8_sentences_icelake>(env, "icelake");
        bench_utf8_linebreaks_kernels<sz_utf8_linebreaks_icelake>(env, "icelake");
        bench_utf8_norm_kernels<sz_utf8_norm_icelake>(env, "icelake");
        bench_utf8_find_denormalized_kernels<sz_utf8_find_denormalized_icelake>(env, "icelake");
        bench_utf8_uncased_fold_kernels<sz_utf8_uncased_fold_icelake>(env, "icelake");
        bench_utf8_uncased_search_kernels<sz_utf8_uncased_search_icelake>(env, "icelake");
        bench_utf8_uncased_order_kernels<sz_utf8_uncased_order_icelake>(env, "icelake");
        bench_bytesum_kernels<sz_bytesum_icelake>(env, "icelake");
        bench_hash_kernels<sz_hash_icelake>(env, "icelake");
        bench_hash_multiseed_kernels<sz_hash_multiseed_icelake>(env, "icelake");
        bench_hash_stream_kernels<sz_hash_state_init_icelake, sz_hash_state_update_icelake,
                                  sz_hash_state_digest_icelake>(env, "icelake");
        bench_fill_random_kernels<sz_fill_random_icelake>(env, "icelake");
        bench_lookup_kernels<sz_lookup_icelake>(env, "icelake");
        bench_aes256_ctr_kernels<sz_aes256_key_init_icelake, sz_aes256_ctr_xor_icelake>(env, "icelake");
        bench_aes256_gcm_kernels<sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_encrypt_icelake>(env, "icelake");
        bench_aes256_gcm_stream_kernels<sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_encryptor_init_icelake,
                                        sz_aes256_gcm_encryptor_update_icelake, sz_aes256_gcm_encryptor_digest_icelake>(
            env, "icelake");
        bench_sequence_intersect_kernels<sz_sequence_intersect_icelake>(env, "icelake");
        bench_levenshtein_distances_kernels<sz_levenshtein_engine_init_icelake, sz_levenshtein_distances_icelake>(
            env, "icelake", sz_levenshtein_bytes_k);
        bench_substrings_kernels<sz_substrings_engine_init_icelake, sz_substrings_counts_icelake,
                                 sz_substrings_find_icelake, sz_substrings_replace_icelake,
                                 sz_substrings_bm25_scores_icelake>(env, "icelake");
#if STRINGZILLA_HEADER_ONLY
        bench_levenshtein_step_kernels<levenshtein_step_from_icelake_narrow>(env, "sz_levenshtein_u8x64_step_icelake",
                                                                             8);
#endif
    }
#endif // STRINGZILLA_TARGET_ICELAKE
}

} // namespace ashvardanian::stringzilla::bench
