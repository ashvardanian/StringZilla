/**
 *  @file bench/levenshtein.cpp
 *  @brief Benchmarks for Levenshtein edit distances under unit costs.
 *         The program accepts a file path to a dataset, tokenizes it, and benchmarks the one-to-one and
 *         one-to-many entries of every backend, validating the SIMD-accelerated backends against the serial one.
 *
 *  Compute-bound: Myers' algorithm costs one word-step per query word per candidate byte, so a 64 MiB slice
 *  exercises every path while each call samples only what it needs.
 *
 *  Three shapes are measured, byte-level and rune-level alike, every candidate at its own length:
 *  - `sz_levenshtein_distance_*` between a median-length query and its successor, on the serial entries alone,
 *    since the one-to-one entries have no other backend;
 *  - `sz_levenshtein_distances_*` from one query against the next `STRINGWARS_BATCH` tokens - by default as many
 *    median tokens as fill a 32 KiB L1 - at two query lengths, the slice's median and the 1024 bytes whose match
 *    masks fill that L1, on every compiled backend;
 *  - the exported building blocks one at a time, so the query's preparation, the staging of candidate bytes into
 *    class ids, and the word-steps over those ids each carry a number of their own.
 *
 *  Every arm's name carries its query length, as `:q117`.
 *  Throughput is reported as Cell Updates Per Second @b (CUPS): the query's length times the candidates' lengths.
 *  The building blocks count what each of them does instead: query bytes prepared, class ids staged, word-steps
 *  taken - so one arm's ops/s column is read beside the next one's rather than against a shared denominator.
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
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_levenshtein_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_release/stringzilla_bench_levenshtein_cpp20
 *  @endcode
 */
#include <stdexcept>   // `std::runtime_error`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <stringzilla/levenshtein.h> // `sz_levenshtein_*`

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::bench;

/** Candidates every step arm advances at once, so every tier answers the same eight scores in the same order. */
static constexpr std::size_t levenshtein_step_lanes_k = (std::size_t)sz_levenshtein_serial_u64x1_candidates_per_step_k *
                                                        sz_levenshtein_serial_u64x1_registers_per_position_k;

/** Positions one step arm walks per call, the transpose width the sweeps feed it from. */
static constexpr std::size_t levenshtein_step_positions_k = sz_levenshtein_positions_per_transpose_k;

/** @brief The first token long enough to fill a @p query_bytes query, clamped to it; the first token otherwise. */
static std::string_view levenshtein_query_token(environment_t const &env, std::size_t query_bytes) {
    for (token_view_t const token : env.tokens)
        if (token.size() >= query_bytes) return std::string_view(token.data(), query_bytes);
    token_view_t const shortest = env.tokens[0];
    return std::string_view(shortest.data(), shortest.size());
}

/** @brief Stages @p lanes tokens' first @p positions bytes transposed as class ids, class zero past a token's end. */
static std::vector<sz_u8_t> levenshtein_staged_classes(environment_t const &env, sz_u8_t const *byte_to_class,
                                                       std::size_t lanes, std::size_t positions) {
    std::vector<sz_u8_t> classes(positions * lanes, 0);
    for (std::size_t lane = 0; lane != lanes; ++lane) {
        token_view_t const token = env.tokens[lane % env.tokens.size()];
        std::size_t const filled = positions < token.size() ? positions : token.size();
        for (std::size_t position = 0; position != filled; ++position)
            classes[position * lanes + lane] = byte_to_class[(sz_u8_t)token[position]];
    }
    return classes;
}

#pragma region One to One

/** @brief One pair per call: a query clamped to @c query_bytes against its successor at its own length. */
template <sz_levenshtein_distance_t function_>
struct levenshtein_distance_from_sz {
    environment_t const &env;
    std::size_t query_bytes;
    sz_memory_allocator_t alloc;

    levenshtein_distance_from_sz(environment_t const &env, std::size_t query_bytes)
        : env(env), query_bytes(query_bytes) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        std::string_view const query = std::string_view(env.tokens[token_index]).substr(0, query_bytes);
        std::string_view const candidate = env.tokens[(token_index + 1) % env.tokens.size()];
        sz_size_t distance = 0;
        if (function_(query.data(), query.size(), candidate.data(), candidate.size(), &alloc, &distance) !=
            sz_success_k)
            throw std::runtime_error("The one-to-one entry failed.");
        return call_result_t(query.size() + candidate.size(), distance, query.size() * candidate.size());
    }
};

/** @brief The one-to-one entries on the serial walk alone, since one pair fills one candidate on every backend. */
void bench_levenshtein_one_to_one(environment_t const &env, std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    bench_unary(env, "sz_levenshtein_distance_serial" + suffix,
                levenshtein_distance_from_sz<sz_levenshtein_distance_serial> {env, query_bytes})
        .log();
    bench_unary(env, "sz_levenshtein_distance_utf8_serial" + suffix,
                levenshtein_distance_from_sz<sz_levenshtein_distance_utf8_serial> {env, query_bytes})
        .log();
}

#pragma endregion

#pragma region One to Many

/** @brief One query per call, clamped to @c query_bytes, against the next @c candidates tokens at their own length. */
template <sz_levenshtein_distances_t function_>
struct levenshtein_distances_from_sz {
    environment_t const &env;
    std::size_t query_bytes;
    std::size_t candidates;
    sz_memory_allocator_t alloc;
    std::vector<sz_string_view_t> views;
    std::vector<sz_size_t> distances;

    levenshtein_distances_from_sz(environment_t const &env, std::size_t query_bytes, std::size_t candidates)
        : env(env), query_bytes(query_bytes), candidates(candidates), views(candidates), distances(candidates) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        std::string_view const query = std::string_view(env.tokens[token_index]).substr(0, query_bytes);
        std::size_t bytes = 0, cells = 0;
        for (std::size_t candidate = 0; candidate != candidates; ++candidate) {
            std::string_view const token = env.tokens[(token_index + 1 + candidate) % env.tokens.size()];
            views[candidate] = {token.data(), token.size()};
            bytes += token.size(), cells += query.size() * token.size();
        }
        sz_sequence_t sequence;
        sz_sequence_from_string_views(views.data(), candidates, &sequence);
        if (function_(query.data(), query.size(), &sequence, &alloc, distances.data()) != sz_success_k)
            throw std::runtime_error("The one-to-many entry failed.");
        // Multiplied rather than summed, so two candidates swapping distances cannot cancel out.
        check_value_t mixed = 0;
        for (sz_size_t const distance : distances) mixed = mixed * 31u + distance;
        call_result_t result(bytes, mixed, cells);
        result.inputs_processed = candidates;
        return result;
    }
};

/** @brief The one-to-many entries at one query length, byte and rune level, every backend against its serial base. */
void bench_levenshtein_one_to_many(environment_t const &env, std::size_t query_bytes, std::size_t candidates) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    auto validator = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {env, query_bytes, candidates};
    bench_result_t base = bench_unary(env, "sz_levenshtein_distances_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_levenshtein_distances_haswell" + suffix, validator,
                levenshtein_distances_from_sz<sz_levenshtein_distances_haswell> {env, query_bytes, candidates})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_levenshtein_distances_skylake" + suffix, validator,
                levenshtein_distances_from_sz<sz_levenshtein_distances_skylake> {env, query_bytes, candidates})
        .log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_levenshtein_distances_icelake" + suffix, validator,
                levenshtein_distances_from_sz<sz_levenshtein_distances_icelake> {env, query_bytes, candidates})
        .log(base);
#endif
    // The rune-level entries decode every candidate byte, so their cost over the byte entries is the decoder's.
    auto validator_utf8 = levenshtein_distances_from_sz<sz_levenshtein_distances_utf8_serial> {env, query_bytes,
                                                                                               candidates};
    bench_result_t base_utf8 =
        bench_unary(env, "sz_levenshtein_distances_utf8_serial" + suffix, validator_utf8).log(base);
#if SZ_USE_HASWELL
    bench_unary(env, "sz_levenshtein_distances_utf8_haswell" + suffix, validator_utf8,
                levenshtein_distances_from_sz<sz_levenshtein_distances_utf8_haswell> {env, query_bytes, candidates})
        .log(base_utf8);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_levenshtein_distances_utf8_skylake" + suffix, validator_utf8,
                levenshtein_distances_from_sz<sz_levenshtein_distances_utf8_skylake> {env, query_bytes, candidates})
        .log(base_utf8);
#endif
}

#pragma endregion

#pragma region Query Preparation

/** @brief One query prepared per call, clamped to @c query_bytes: the fixed cost a sweep pays once per query. */
struct levenshtein_prepare_from_sz {
    environment_t const &env;                                      /**< The tokens the query is drawn from. */
    std::size_t query_bytes;                                       /**< Bytes the query is clamped to. */
    std::vector<sz_u64_t> masks;                                   /**< Match masks the preparation fills. */
    std::vector<sz_u8_t> byte_to_class;                            /**< @b [256] mask row each byte value reads. */
    std::vector<sz_levenshtein_u64x1_vertical_serial_t> verticals; /**< @b [words] one step's Myers deltas. */

    levenshtein_prepare_from_sz(environment_t const &env, std::size_t query_bytes)
        : env(env), query_bytes(query_bytes), masks(sz_levenshtein_query_mask_entries(query_bytes)),
          byte_to_class(sz_levenshtein_byte_classes_k), verticals(sz_levenshtein_query_words(query_bytes)) {}

    call_result_t operator()(std::size_t token_index) {
        std::string_view const query = std::string_view(env.tokens[token_index]).substr(0, query_bytes);
        sz_levenshtein_query_t prepared {};
        if (sz_levenshtein_query_prepare(query.data(), query.size(), masks.data(), byte_to_class.data(), &prepared) !=
            sz_success_k)
            throw std::runtime_error("The query preparation failed.");
        // One step over the fresh table, so its stores cannot be dropped as dead and the layout stays private.
        std::size_t const words = sz_levenshtein_query_words(query.size());
        sz_levenshtein_u64x1_state_serial_t state;
        sz_levenshtein_u64x1_init_serial(&state, verticals.data(), words, &prepared);
        sz_levenshtein_u64x1_step_serial(&state, verticals.data(), words, &prepared, byte_to_class[0]);
        return call_result_t(query.size(), sz_levenshtein_u64x1_score_serial(&state, 0), query.size());
    }
};

/** @brief The query preparation alone, at one query length, on the single entry the family exports for it. */
void bench_levenshtein_query_prepare(environment_t const &env, std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    bench_unary(env, "sz_levenshtein_query_prepare" + suffix, levenshtein_prepare_from_sz {env, query_bytes}).log();
}

#pragma endregion

#pragma region Steps

/**
 *  @brief One transpose of staged class ids stepped per call, eight candidates deep, the staging already done.
 *
 *  The word count is a run-time value here, as it is for every query past two words, and the classes and the
 *  query are fixed across calls, so what is timed is the recurrence and nothing around it.
 */
struct levenshtein_step_from_serial {
    std::size_t words = 0;                                         /**< Query words every step walks. */
    std::vector<sz_u64_t> masks;                                   /**< Match masks the query points at. */
    std::vector<sz_u8_t> byte_to_class;                            /**< @b [256] mask row each byte reads. */
    sz_levenshtein_query_t query {};                               /**< The query every lane is scored against. */
    std::vector<sz_u8_t> classes;                                  /**< @b [positions x lanes] staged class ids. */
    std::vector<sz_levenshtein_u64x1_vertical_serial_t> verticals; /**< @b [lanes x words] Myers deltas. */

    levenshtein_step_from_serial(environment_t const &env, std::size_t query_bytes)
        : masks(sz_levenshtein_query_mask_entries(query_bytes)), byte_to_class(sz_levenshtein_byte_classes_k) {
        std::string_view const text = levenshtein_query_token(env, query_bytes);
        if (sz_levenshtein_query_prepare(text.data(), text.size(), masks.data(), byte_to_class.data(), &query) !=
            sz_success_k)
            throw std::runtime_error("The query preparation failed.");
        words = sz_levenshtein_query_words(text.size());
        classes = levenshtein_staged_classes(env, byte_to_class.data(), levenshtein_step_lanes_k,
                                             levenshtein_step_positions_k);
        verticals.resize(levenshtein_step_lanes_k * words);
    }

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        sz_levenshtein_u64x1_state_serial_t states[levenshtein_step_lanes_k];
        for (std::size_t lane = 0; lane != levenshtein_step_lanes_k; ++lane)
            sz_levenshtein_u64x1_init_serial(&states[lane], verticals.data() + lane * words, words, &query);
        for (std::size_t position = 0; position != levenshtein_step_positions_k; ++position)
            for (std::size_t lane = 0; lane != levenshtein_step_lanes_k; ++lane)
                sz_levenshtein_u64x1_step_serial(&states[lane], verticals.data() + lane * words, words, &query,
                                                 classes[position * levenshtein_step_lanes_k + lane]);
        check_value_t mixed = 0;
        for (std::size_t lane = 0; lane != levenshtein_step_lanes_k; ++lane)
            mixed = mixed * 31u + sz_levenshtein_u64x1_score_serial(&states[lane], 0);
        call_result_t result(levenshtein_step_positions_k * levenshtein_step_lanes_k, mixed,
                             levenshtein_step_positions_k * levenshtein_step_lanes_k * words);
        result.inputs_processed = levenshtein_step_lanes_k;
        return result;
    }
};

#if SZ_USE_HASWELL

/** @brief The same eight lanes as the serial step, taken four per YMM, so the two arms answer the same scores. */
struct levenshtein_step_from_haswell {
    static constexpr std::size_t groups_k = levenshtein_step_lanes_k /
                                            sz_levenshtein_haswell_u64x4_candidates_per_step_k;

    std::size_t words = 0;                                          /**< Query words every step walks. */
    std::vector<sz_u64_t> masks;                                    /**< Match masks the query points at. */
    std::vector<sz_u8_t> byte_to_class;                             /**< @b [256] mask row each byte reads. */
    sz_levenshtein_query_t query {};                                /**< The query every lane is scored against. */
    std::vector<sz_u8_t> classes;                                   /**< @b [positions x lanes] staged class ids. */
    std::vector<sz_levenshtein_u64x4_vertical_haswell_t> verticals; /**< @b [groups x words] Myers deltas. */

    levenshtein_step_from_haswell(environment_t const &env, std::size_t query_bytes)
        : masks(sz_levenshtein_query_mask_entries(query_bytes)), byte_to_class(sz_levenshtein_byte_classes_k) {
        std::string_view const text = levenshtein_query_token(env, query_bytes);
        if (sz_levenshtein_query_prepare(text.data(), text.size(), masks.data(), byte_to_class.data(), &query) !=
            sz_success_k)
            throw std::runtime_error("The query preparation failed.");
        words = sz_levenshtein_query_words(text.size());
        classes = levenshtein_staged_classes(env, byte_to_class.data(), levenshtein_step_lanes_k,
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
                                                      group * sz_levenshtein_haswell_u64x4_candidates_per_step_k));
        check_value_t mixed = 0;
        for (std::size_t group = 0; group != groups_k; ++group)
            for (std::size_t lane = 0; lane != sz_levenshtein_haswell_u64x4_candidates_per_step_k; ++lane)
                mixed = mixed * 31u + sz_levenshtein_u64x4_score_haswell(&states[group], lane);
        call_result_t result(levenshtein_step_positions_k * levenshtein_step_lanes_k, mixed,
                             levenshtein_step_positions_k * levenshtein_step_lanes_k * words);
        result.inputs_processed = levenshtein_step_lanes_k;
        return result;
    }
};

#endif

#if SZ_USE_SKYLAKE

/** @brief The same eight lanes as the serial step, taken eight per ZMM, so the two arms answer the same scores. */
struct levenshtein_step_from_skylake {
    std::size_t words = 0;                                          /**< Query words every step walks. */
    std::vector<sz_u64_t> masks;                                    /**< Match masks the query points at. */
    std::vector<sz_u8_t> byte_to_class;                             /**< @b [256] mask row each byte reads. */
    sz_levenshtein_query_t query {};                                /**< The query every lane is scored against. */
    std::vector<sz_u8_t> classes;                                   /**< @b [positions x lanes] staged class ids. */
    std::vector<sz_levenshtein_u64x8_vertical_skylake_t> verticals; /**< @b [words] Myers deltas of eight lanes. */

    levenshtein_step_from_skylake(environment_t const &env, std::size_t query_bytes)
        : masks(sz_levenshtein_query_mask_entries(query_bytes)), byte_to_class(sz_levenshtein_byte_classes_k) {
        std::string_view const text = levenshtein_query_token(env, query_bytes);
        if (sz_levenshtein_query_prepare(text.data(), text.size(), masks.data(), byte_to_class.data(), &query) !=
            sz_success_k)
            throw std::runtime_error("The query preparation failed.");
        words = sz_levenshtein_query_words(text.size());
        classes = levenshtein_staged_classes(env, byte_to_class.data(), levenshtein_step_lanes_k,
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
        result.inputs_processed = levenshtein_step_lanes_k;
        return result;
    }
};

#endif

#if SZ_USE_ICELAKE

/**
 *  @brief Sixty-four byte lanes stepped per call, of which the first eight carry the serial arm's lanes, so the
 *      two answer the same scores while the timing covers every lane.
 *
 *  Only a query of at most eight symbols fits a byte lane, so the arm is registered at those lengths alone, and
 *  the deltas are folded into the scores on the cadence the sweep uses rather than on a per-position branch.
 */
struct levenshtein_step_from_icelake_narrow {
    static constexpr std::size_t lanes_k = sz_levenshtein_icelake_u8x64_candidates_per_step_k;
    static constexpr std::size_t positions_per_flush_k = sz_levenshtein_icelake_u8x64_positions_per_flush_k;

    std::vector<sz_u64_t> masks;                    /**< Match masks the query points at. */
    std::vector<sz_u8_t> byte_to_class;             /**< @b [256] mask row each byte reads. */
    sz_levenshtein_query_t query {};                /**< The query every lane is scored against. */
    sz_levenshtein_u8x64_query_icelake_t packed {}; /**< The same query as one mask byte per class. */
    std::vector<sz_u8_t> classes;                   /**< @b [positions x lanes] staged class ids. */
    std::vector<sz_size_t> scores;                  /**< @b [lanes] running distances the deltas fold into. */

    levenshtein_step_from_icelake_narrow(environment_t const &env, std::size_t query_bytes)
        : masks(sz_levenshtein_query_mask_entries(query_bytes)), byte_to_class(sz_levenshtein_byte_classes_k),
          scores(lanes_k) {
        std::string_view const text = levenshtein_query_token(env, query_bytes);
        if (sz_levenshtein_query_prepare(text.data(), text.size(), masks.data(), byte_to_class.data(), &query) !=
            sz_success_k)
            throw std::runtime_error("The query preparation failed.");
        sz_levenshtein_u8x64_pack_icelake(&query, &packed);
        classes = levenshtein_staged_classes(env, byte_to_class.data(), lanes_k, levenshtein_step_positions_k);
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
        result.inputs_processed = lanes_k;
        return result;
    }
};

#endif

/** @brief The word-step alone, every tier over the same eight lanes, checked against the serial recurrence. */
void bench_levenshtein_steps(environment_t const &env, std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    auto validator = levenshtein_step_from_serial {env, query_bytes};
    bench_result_t base = bench_unary(env, "sz_levenshtein_u64x1_step_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_levenshtein_u64x4_step_haswell" + suffix, validator,
                levenshtein_step_from_haswell {env, query_bytes})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_levenshtein_u64x8_step_skylake" + suffix, validator,
                levenshtein_step_from_skylake {env, query_bytes})
        .log(base);
#endif
#if SZ_USE_ICELAKE
    if (query_bytes <= 8)
        bench_unary(env, "sz_levenshtein_u8x64_step_icelake" + suffix, validator,
                    levenshtein_step_from_icelake_narrow {env, query_bytes})
            .log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers();
    std::printf("Welcome to StringZilla!\n");
    if (auto code = log_environment(); code != 0) return code;

    // The arms throw on a failed status, so one bad call ends the run with its message rather than a crash.
    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment(argc, argv, "xlsum.csv", environment_t::tokenization_t::lines_k);
        std::size_t const candidates = candidates_per_call(env);
        // The long arm is the 1024 bytes whose match masks fill a 32 KiB L1, where the multi-word regime starts.
        std::size_t const query_lengths[] = {median_token_bytes(env), 1024};
        std::printf("Starting Levenshtein benchmarks...\n");
        bench_levenshtein_one_to_one(env, median_token_bytes(env));
        for (std::size_t const query_bytes : query_lengths) {
            bench_levenshtein_one_to_many(env, query_bytes, candidates);
            bench_levenshtein_query_prepare(env, query_bytes);
            bench_levenshtein_steps(env, query_bytes);
        }
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}
