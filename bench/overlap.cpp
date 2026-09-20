/**
 *  @file bench/overlap.cpp
 *  @brief Benchmarks for window overlap built from the `sz_overlap_*` step verbs.
 *         The program accepts a file path to a dataset, tokenizes it, prepares the leading tokens as the query
 *         B-tree, and scores every other token against it on every backend, validating SIMD backends against serial.
 *
 *  Compute-bound: the prefix hashes are one pass over a candidate and the window hashes one more, so a 64 MiB
 *  slice exercises every path.
 *
 *  Six arms are measured per backend, each reporting the windows it touched as `operations`, so the
 *  ops-per-second column reads as windows per second. The first three nest, so a stage's own cost is the
 *  difference between neighbouring arms:
 *  - `prefix_hashes` - the prefix hashes alone, one per byte, whatever the width;
 *  - `window_hashes` - the prefix hashes, then their differences at the derived width;
 *  - `window_lookups` - the prefix hashes, the window hashes, then the B-tree walk over every window hash;
 *  - `query_indexing` - the query's key sort and tree layout, once per call, the token ignored - at an 8 KiB query
 *    this is most of a round, so it stands on its own;
 *  - `score` - the one-to-one verb against one token: both chains interleaved, the query's sort and layout, one probe;
 *  - `scores` - the one-to-many verb against the next `STRINGWARS_BATCH` tokens, by default as many median tokens as
 *    fill a 32 KiB L1: the query prepared once, four chains interleaved.
 *
 *  Two query lengths run: the slice's median token length, and the byte count whose window hashes fill a 32 KiB L1.
 *  The window width is derived from the slice rather than fixed - `ceil(log2(query bytes · mean candidate bytes)
 *  / H2)`, with `H2` the byte collision entropy of the slice - and every arm's name carries it, as `:w6`.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or an integer [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names, e.g.
 *    `window_lookups.*skylake`.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_overlap_cpp20
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines build_release/stringzilla_bench_overlap_cpp20
 *  @endcode
 */
#include <cmath>   // `std::ceil`, `std::log2`
#include <cstring> // `std::memcpy`

#include <algorithm> // `std::copy`, `std::max`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::bench;

/** @brief The query whose window hashes fill the first-level cache, one @c u32 hash per byte. */
static std::size_t cache_resident_query_bytes(environment_t const &env) {
    return env.specs.l1_bytes / sizeof(sz_u32_t);
}

using overlap_prefix_hash_step_t = sz_f64_t (*)(sz_f64_t, sz_cptr_t, sz_f64_t *);
using overlap_prefix_hash_step_tail_t = sz_f64_t (*)(sz_f64_t, sz_cptr_t, sz_size_t, sz_f64_t *);
using overlap_window_hash_step_t = void (*)(sz_f64_t const *, sz_f64_t const *, sz_f64_t, sz_u32_t *);
using overlap_window_hash_step_tail_t = void (*)(sz_f64_t const *, sz_f64_t const *, sz_f64_t, sz_size_t, sz_u32_t *);
using overlap_btree_sort_t = sz_size_t (*)(sz_u32_t *, sz_size_t);
using overlap_btree_probe_t = sz_size_t (*)(sz_overlap_btree_t const *, sz_u32_t const *, sz_size_t);

/** @brief The chain over @p text, one prefix hash per byte after the empty one at @p prefix_hashes[0]. */
template <sz_size_t positions_per_step_, overlap_prefix_hash_step_t prefix_hash_step_,
          overlap_prefix_hash_step_tail_t prefix_hash_step_tail_>
static void overlap_prefix_hashes_(std::string_view text, sz_f64_t *prefix_hashes) {
    prefix_hashes[0] = 0.0;
    sz_f64_t prior = 0.0;
    std::size_t position = 0;
    for (; position + positions_per_step_ <= text.size(); position += positions_per_step_)
        prior = prefix_hash_step_(prior, text.data() + position, prefix_hashes + position + 1);
    if (position != text.size())
        prefix_hash_step_tail_(prior, text.data() + position, text.size() - position, prefix_hashes + position + 1);
}

/** @brief The prefix differences at @p width over a chain of @p bytes positions, answering the windows extracted. */
template <sz_size_t positions_per_step_, overlap_window_hash_step_t window_hash_step_,
          overlap_window_hash_step_tail_t window_hash_step_tail_>
static std::size_t overlap_window_hashes_(sz_f64_t const *prefix_hashes, std::size_t bytes, std::size_t width,
                                          sz_u32_t *window_hashes) {
    if (width > bytes) return 0;
    std::size_t const windows = bytes - width + 1;
    sz_f64_t const power = sz_overlap_window_power(width);
    std::size_t window = 0;
    for (; window + positions_per_step_ <= windows; window += positions_per_step_)
        window_hash_step_(prefix_hashes + window, prefix_hashes + window + width, power, window_hashes + window);
    if (window != windows)
        window_hash_step_tail_(prefix_hashes + window, prefix_hashes + window + width, power, windows - window,
                               window_hashes + window);
    return windows;
}

/** @brief The longest token in the slice, so every per-candidate arm sizes its scratch once. */
static std::size_t overlap_longest_token_(environment_t const &env) {
    std::size_t longest = 0;
    for (std::string_view const token : env.tokens) longest = std::max(longest, token.size());
    return longest;
}

/**
 *  @brief The window width at which a random query window and a random candidate window collide about once per query:
 *         @c ceil(log2(query_bytes · mean candidate bytes) / H2), with @c H2 the slice's byte collision entropy.
 */
static std::size_t overlap_width_(environment_t const &env, std::size_t query_bytes) {
    double counts[256] = {};
    for (char const byte : env.dataset) counts[static_cast<unsigned char>(byte)] += 1.0;
    double collisions = 0.0;
    for (double const count : counts) collisions += count * count;
    double const total = static_cast<double>(env.dataset.size());
    double const collision_entropy = -std::log2(collisions / (total * total));
    std::size_t token_bytes = 0;
    for (std::string_view const token : env.tokens) token_bytes += token.size();
    double const mean_candidate_bytes = static_cast<double>(token_bytes) / static_cast<double>(env.tokens.size());
    double const width = std::ceil(std::log2(static_cast<double>(query_bytes) * mean_candidate_bytes) /
                                   collision_entropy);
    return width > 1.0 ? static_cast<std::size_t>(width) : 1;
}

/** @brief One query length's fixed input: the leading tokens concatenated, and their raw window hashes at @c width. */
struct overlap_query_t {
    /** @brief The window width every arm extracts and scores at. */
    std::size_t width;
    /** @brief The dataset's leading tokens, concatenated up to the requested byte count. */
    std::string text;
    /** @brief The raw window hashes of @c text, what every sort and tree layout starts from. */
    std::vector<sz_u32_t> window_hashes;

    overlap_query_t(environment_t const &env, std::size_t query_bytes) : width(overlap_width_(env, query_bytes)) {
        for (std::string_view const token : env.tokens) {
            if (text.size() >= query_bytes) break;
            text.append(token);
        }
        std::vector<sz_f64_t> prefix_hashes(text.size() + 1);
        overlap_prefix_hashes_<sz_overlap_serial_f64x1_positions_per_step_k, sz_overlap_f64x1_prefix_hash_step_serial,
                               sz_overlap_f64x1_prefix_hash_step_tail_serial>(text, prefix_hashes.data());
        window_hashes.resize(text.size());
        window_hashes.resize(overlap_window_hashes_<sz_overlap_serial_f64x1_positions_per_step_k,
                                                    sz_overlap_f64x1_window_hash_step_serial,
                                                    sz_overlap_f64x1_window_hash_step_tail_serial>(
            prefix_hashes.data(), text.size(), width, window_hashes.data()));
    }
};

#pragma region Prefix Hashes

/** @brief The prefix hashes alone over one token, one per byte however many widths follow them. */
template <sz_size_t positions_per_step_, overlap_prefix_hash_step_t prefix_hash_step_,
          overlap_prefix_hash_step_tail_t prefix_hash_step_tail_>
struct prefix_hashes_from_sz {
    environment_t const &env;
    std::vector<sz_f64_t> prefix_hashes;

    explicit prefix_hashes_from_sz(environment_t const &env)
        : env(env), prefix_hashes(overlap_longest_token_(env) + 1) {}

    call_result_t operator()(std::size_t token_index) noexcept {
        std::string_view const text = env.tokens[token_index];
        overlap_prefix_hashes_<positions_per_step_, prefix_hash_step_, prefix_hash_step_tail_>(text,
                                                                                               prefix_hashes.data());
        return call_result_t(text.size(), static_cast<check_value_t>(prefix_hashes[text.size()]), text.size());
    }
};

/** @brief The chain on every backend, the accelerated arms logged against the serial one. */
static void bench_overlap_prefix_hashes(environment_t const &env, std::string const &suffix) {
    auto validator =
        prefix_hashes_from_sz<sz_overlap_serial_f64x1_positions_per_step_k, sz_overlap_f64x1_prefix_hash_step_serial,
                              sz_overlap_f64x1_prefix_hash_step_tail_serial> {env};
    bench_result_t base = bench_unary(env, "sz_overlap_prefix_hashes_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(
        env, "sz_overlap_prefix_hashes_haswell" + suffix, validator,
        prefix_hashes_from_sz<sz_overlap_haswell_f64x4_positions_per_step_k, sz_overlap_f64x4_prefix_hash_step_haswell,
                              sz_overlap_f64x4_prefix_hash_step_tail_haswell> {env})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(
        env, "sz_overlap_prefix_hashes_skylake" + suffix, validator,
        prefix_hashes_from_sz<sz_overlap_skylake_f64x8_positions_per_step_k, sz_overlap_f64x8_prefix_hash_step_skylake,
                              sz_overlap_f64x8_prefix_hash_step_tail_skylake> {env})
        .log(base);
#endif
}

#pragma endregion

#pragma region Window Hashes

/** @brief The chain, then the window hashing over it at the query's width. */
template <sz_size_t positions_per_step_, overlap_prefix_hash_step_t prefix_hash_step_,
          overlap_prefix_hash_step_tail_t prefix_hash_step_tail_, overlap_window_hash_step_t window_hash_step_,
          overlap_window_hash_step_tail_t window_hash_step_tail_>
struct window_hashes_from_sz {
    environment_t const &env;
    std::size_t width;
    std::vector<sz_f64_t> prefix_hashes;
    std::vector<sz_u32_t> window_hashes;

    window_hashes_from_sz(environment_t const &env, overlap_query_t const &query)
        : env(env), width(query.width), prefix_hashes(overlap_longest_token_(env) + 1),
          window_hashes(prefix_hashes.size()) {}

    call_result_t operator()(std::size_t token_index) noexcept {
        std::string_view const text = env.tokens[token_index];
        overlap_prefix_hashes_<positions_per_step_, prefix_hash_step_, prefix_hash_step_tail_>(text,
                                                                                               prefix_hashes.data());
        std::size_t const windows =
            overlap_window_hashes_<positions_per_step_, window_hash_step_, window_hash_step_tail_>(
                prefix_hashes.data(), text.size(), width, window_hashes.data());
        // Multiplied rather than summed, so two windows swapping hashes cannot cancel out.
        check_value_t mixed = 0;
        for (std::size_t window = 0; window != windows; ++window) mixed = mixed * 31u + window_hashes[window];
        return call_result_t(text.size(), mixed, windows);
    }
};

/** @brief The chain and the window hashes on every backend, the accelerated arms logged against the serial one. */
static void bench_overlap_window_hashes(environment_t const &env, overlap_query_t const &query,
                                        std::string const &suffix) {
    auto validator =
        window_hashes_from_sz<sz_overlap_serial_f64x1_positions_per_step_k, sz_overlap_f64x1_prefix_hash_step_serial,
                              sz_overlap_f64x1_prefix_hash_step_tail_serial, sz_overlap_f64x1_window_hash_step_serial,
                              sz_overlap_f64x1_window_hash_step_tail_serial> {env, query};
    bench_result_t base = bench_unary(env, "sz_overlap_window_hashes_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(
        env, "sz_overlap_window_hashes_haswell" + suffix, validator,
        window_hashes_from_sz<sz_overlap_haswell_f64x4_positions_per_step_k, sz_overlap_f64x4_prefix_hash_step_haswell,
                              sz_overlap_f64x4_prefix_hash_step_tail_haswell, sz_overlap_f64x4_window_hash_step_haswell,
                              sz_overlap_f64x4_window_hash_step_tail_haswell> {env, query})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(
        env, "sz_overlap_window_hashes_skylake" + suffix, validator,
        window_hashes_from_sz<sz_overlap_skylake_f64x8_positions_per_step_k, sz_overlap_f64x8_prefix_hash_step_skylake,
                              sz_overlap_f64x8_prefix_hash_step_tail_skylake, sz_overlap_f64x8_window_hash_step_skylake,
                              sz_overlap_f64x8_window_hash_step_tail_skylake> {env, query})
        .log(base);
#endif
}

#pragma endregion

#pragma region Window Lookups

/** @brief The chain, the window hashes, then the membership test of every one against the query's tree. */
template <sz_size_t positions_per_step_, overlap_prefix_hash_step_t prefix_hash_step_,
          overlap_prefix_hash_step_tail_t prefix_hash_step_tail_, overlap_window_hash_step_t window_hash_step_,
          overlap_window_hash_step_tail_t window_hash_step_tail_, overlap_btree_sort_t btree_sort_,
          overlap_btree_probe_t btree_probe_>
struct window_lookups_from_sz {
    environment_t const &env;
    std::size_t width;
    std::vector<sz_f64_t> prefix_hashes;
    std::vector<sz_u32_t> window_hashes;
    std::vector<sz_u32_t> nodes;
    sz_overlap_btree_t btree {};

    window_lookups_from_sz(environment_t const &env, overlap_query_t const &query)
        : env(env), width(query.width), prefix_hashes(overlap_longest_token_(env) + 1),
          window_hashes(prefix_hashes.size()), nodes(sz_overlap_btree_entries(query.window_hashes.size())) {
        std::copy(query.window_hashes.begin(), query.window_hashes.end(), nodes.begin());
        std::size_t const distinct = btree_sort_(nodes.data(), query.window_hashes.size());
        if (sz_overlap_btree_prepare(nodes.data(), distinct, &btree) != sz_success_k)
            throw std::runtime_error("The query B-tree could not be laid out.");
    }

    call_result_t operator()(std::size_t token_index) noexcept {
        std::string_view const text = env.tokens[token_index];
        overlap_prefix_hashes_<positions_per_step_, prefix_hash_step_, prefix_hash_step_tail_>(text,
                                                                                               prefix_hashes.data());
        std::size_t const windows =
            overlap_window_hashes_<positions_per_step_, window_hash_step_, window_hash_step_tail_>(
                prefix_hashes.data(), text.size(), width, window_hashes.data());
        return call_result_t(text.size(), btree_probe_(&btree, window_hashes.data(), windows), windows);
    }
};

/** @brief The chain, the window hashes and the lookups on every backend, the accelerated arms logged against the
 *         serial one. */
static void bench_overlap_window_lookups(environment_t const &env, overlap_query_t const &query,
                                         std::string const &suffix) {
    auto validator =
        window_lookups_from_sz<sz_overlap_serial_f64x1_positions_per_step_k, sz_overlap_f64x1_prefix_hash_step_serial,
                               sz_overlap_f64x1_prefix_hash_step_tail_serial, sz_overlap_f64x1_window_hash_step_serial,
                               sz_overlap_f64x1_window_hash_step_tail_serial, sz_overlap_u32x1_btree_sort_serial,
                               sz_overlap_u32x1_btree_probe_serial> {env, query};
    bench_result_t base = bench_unary(env, "sz_overlap_window_lookups_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(
        env, "sz_overlap_window_lookups_haswell" + suffix, validator,
        window_lookups_from_sz<sz_overlap_haswell_f64x4_positions_per_step_k, sz_overlap_f64x4_prefix_hash_step_haswell,
                               sz_overlap_f64x4_prefix_hash_step_tail_haswell,
                               sz_overlap_f64x4_window_hash_step_haswell,
                               sz_overlap_f64x4_window_hash_step_tail_haswell, sz_overlap_u32x8_btree_sort_haswell,
                               sz_overlap_u32x8_btree_probe_haswell> {env, query})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(
        env, "sz_overlap_window_lookups_skylake" + suffix, validator,
        window_lookups_from_sz<sz_overlap_skylake_f64x8_positions_per_step_k, sz_overlap_f64x8_prefix_hash_step_skylake,
                               sz_overlap_f64x8_prefix_hash_step_tail_skylake,
                               sz_overlap_f64x8_window_hash_step_skylake,
                               sz_overlap_f64x8_window_hash_step_tail_skylake, sz_overlap_u32x16_btree_sort_skylake,
                               sz_overlap_u32x16_btree_probe_skylake> {env, query})
        .log(base);
#endif
}

#pragma endregion

#pragma region Query Indexing

/** @brief The query's sort and tree layout alone, once per call from its raw window hashes; the token is ignored. */
template <overlap_btree_sort_t btree_sort_>
struct query_indexing_from_sz {
    overlap_query_t const &query;
    std::vector<sz_u32_t> nodes;
    sz_overlap_btree_t btree {};

    explicit query_indexing_from_sz(overlap_query_t const &query)
        : query(query), nodes(sz_overlap_btree_entries(query.window_hashes.size())) {}

    call_result_t operator()(std::size_t) {
        std::copy(query.window_hashes.begin(), query.window_hashes.end(), nodes.begin());
        std::size_t const distinct = btree_sort_(nodes.data(), query.window_hashes.size());
        if (sz_overlap_btree_prepare(nodes.data(), distinct, &btree) != sz_success_k)
            throw std::runtime_error("The query B-tree could not be laid out.");
        return call_result_t(query.window_hashes.size() * sizeof(sz_u32_t), distinct, query.window_hashes.size());
    }
};

/** @brief The query indexing on every backend, the accelerated arms logged against the serial one. */
static void bench_overlap_query_indexing(environment_t const &env, overlap_query_t const &query,
                                         std::string const &suffix) {
    auto validator = query_indexing_from_sz<sz_overlap_u32x1_btree_sort_serial> {query};
    bench_result_t base = bench_unary(env, "sz_overlap_query_indexing_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_overlap_query_indexing_haswell" + suffix, validator,
                query_indexing_from_sz<sz_overlap_u32x8_btree_sort_haswell> {query})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_overlap_query_indexing_skylake" + suffix, validator,
                query_indexing_from_sz<sz_overlap_u32x16_btree_sort_skylake> {query})
        .log(base);
#endif
}

#pragma endregion

#pragma region One to One

/** @brief The one-to-one verb at the query's width, one token per call. */
template <sz_overlap_score_t score_>
struct score_from_sz {
    environment_t const &env;
    overlap_query_t const &query;
    sz_memory_allocator_t alloc;

    score_from_sz(environment_t const &env, overlap_query_t const &query) : env(env), query(query) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        std::string_view const text = env.tokens[token_index];
        sz_size_t const width = query.width;
        sz_f32_t score = 0.0f;
        if (score_(query.text.data(), query.text.size(), text.data(), text.size(), &width, 1, &alloc, &score) !=
            sz_success_k)
            throw std::runtime_error("The one-to-one verb failed.");
        sz_u32_t bits = 0;
        std::memcpy(&bits, &score, sizeof(bits));
        std::size_t const windows = width <= text.size() ? text.size() - width + 1 : 0;
        return call_result_t(text.size(), bits, windows);
    }
};

/** @brief The one-to-one verb on every backend, the accelerated arms logged against the serial one. */
static void bench_overlap_score(environment_t const &env, overlap_query_t const &query, std::string const &suffix) {
    auto validator = score_from_sz<sz_overlap_score_serial> {env, query};
    bench_result_t base = bench_unary(env, "sz_overlap_score_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_overlap_score_haswell" + suffix, validator,
                score_from_sz<sz_overlap_score_haswell> {env, query})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_overlap_score_skylake" + suffix, validator,
                score_from_sz<sz_overlap_score_skylake> {env, query})
        .log(base);
#endif
}

#pragma endregion

#pragma region One to Many

/** @brief The one-to-many verb over the next @c candidates tokens, one call per iteration. */
template <sz_overlap_scores_t scores_>
struct scores_from_sz {
    environment_t const &env;
    overlap_query_t const &query;
    std::size_t candidates;
    sz_memory_allocator_t alloc;
    std::vector<sz_string_view_t> views;
    std::vector<sz_f32_t> scores;

    scores_from_sz(environment_t const &env, overlap_query_t const &query, std::size_t candidates)
        : env(env), query(query), candidates(candidates), views(candidates), scores(candidates) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        std::size_t bytes = 0, windows = 0;
        for (std::size_t candidate = 0; candidate != candidates; ++candidate) {
            std::string_view const text = env.tokens[(token_index + candidate) % env.tokens.size()];
            views[candidate] = {text.data(), text.size()};
            bytes += text.size();
            windows += query.width <= text.size() ? text.size() - query.width + 1 : 0;
        }
        sz_sequence_t sequence {};
        sz_sequence_from_string_views(views.data(), candidates, &sequence);
        sz_size_t const width = query.width;
        if (scores_(query.text.data(), query.text.size(), &sequence, &width, 1, &alloc, scores.data()) != sz_success_k)
            throw std::runtime_error("The one-to-many verb failed.");
        // Multiplied rather than summed, so two candidates swapping scores cannot cancel out.
        check_value_t mixed = 0;
        for (sz_f32_t const score : scores) {
            sz_u32_t bits = 0;
            std::memcpy(&bits, &score, sizeof(bits));
            mixed = mixed * 31u + bits;
        }
        call_result_t result(bytes, mixed, windows);
        result.inputs_processed = candidates;
        return result;
    }
};

/** @brief The one-to-many verb on every backend, the accelerated arms logged against the serial one. */
static void bench_overlap_scores(environment_t const &env, overlap_query_t const &query, std::size_t candidates,
                                 std::string const &suffix) {
    auto validator = scores_from_sz<sz_overlap_scores_serial> {env, query, candidates};
    bench_result_t base = bench_unary(env, "sz_overlap_scores_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_overlap_scores_haswell" + suffix, validator,
                scores_from_sz<sz_overlap_scores_haswell> {env, query, candidates})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_overlap_scores_skylake" + suffix, validator,
                scores_from_sz<sz_overlap_scores_skylake> {env, query, candidates})
        .log(base);
#endif
}

#pragma endregion

/** @brief Every arm at one query length, the width derived once from the slice and carried in every arm's name. */
static void bench_overlap_query(environment_t const &env, char const *query_name, std::size_t query_bytes,
                                std::size_t candidates) {
    overlap_query_t const query(env, query_bytes);
    std::string const suffix = std::string(":") + query_name + ":w" + std::to_string(query.width);
    bench_overlap_prefix_hashes(env, suffix);
    bench_overlap_window_hashes(env, query, suffix);
    bench_overlap_window_lookups(env, query, suffix);
    bench_overlap_query_indexing(env, query, suffix);
    bench_overlap_score(env, query, suffix);
    bench_overlap_scores(env, query, candidates, suffix);
}

int main(int argc, char const **argv) {
    install_test_signal_handlers();
    std::printf("Welcome to StringZilla!\n");
    if (auto code = log_environment(); code != 0) return code;

    // The arms throw on a failed status, so one bad call ends the run with its message rather than a crash.
    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment(argc, argv, "leipzig1M.txt", environment_t::tokenization_t::lines_k,
                                              compute_bound_slice_bytes_k);
        std::size_t const candidates = candidates_per_call(env);
        std::printf("Starting window overlap benchmarks...\n");
        bench_overlap_query(env, "short_query", median_token_bytes(env), candidates);
        bench_overlap_query(env, "long_query", cache_resident_query_bytes(env), candidates);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}
