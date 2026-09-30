/**
 *  @file bench/overlap.cpp
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Benchmarks for window overlap built from the `sz_overlap_*` step verbs.
 *
 *  Prepares the leading tokens of the multilingual lines as the query B-tree, and scores every
 *  other token against it through the engine's verbs, which pick their kernel from the CPU's
 *  capabilities. Every capability's kernels, and the steps only the tier headers define, are timed
 *  against the serial ones by the `cross_<arch>.cpp` files.
 *
 *  Compute-bound: the prefix hashes are one pass over a candidate and the window hashes one more,
 *  so a 64 MiB slice exercises every path.
 *
 *  Five arms are measured per backend, each reporting the windows it touched as @c operations, so
 *  the ops-per-second column reads as windows per second. The first three nest, so a stage's own
 *  cost is the difference between neighbouring arms:
 *  - @c prefix_hashes - the prefix hashes alone, one per byte, whatever the width;
 *  - @c window_hashes - the prefix hashes, then their differences at the derived width;
 *  - @c window_lookups - the prefix hashes, the window hashes, then the B-tree walk over them all;
 *  - @c query_preparation - the query's key sort and tree layout, once per call, the token ignored
 *    - at an 8 KiB query this is most of a round, so it stands on its own;
 *  - @c scores - the engine's round against the next @c STRINGWARS_BATCH tokens, by default as many
 *    median tokens as fill a 32 KiB L1: the forest built once before the timing, four chains
 *    interleaved inside it.
 *
 *  The query is the slice's median token length. The window width is derived from the slice rather
 *  than fixed, and every arm's name carries it, as `:w6`. With H₂ the byte collision entropy of
 *  the slice, the width is ⌈log₂(query bytes × mean candidate bytes) / H₂⌉.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment
 *  variables are used:
 *  - `STRINGWARS_DATASET=path` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole
 *    file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or an integer
 *    [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_MAX_SECONDS=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names,
 *    e.g. `window_lookups.*skylake`.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_FILTER=overlap build_release/stringzilla_cpu_bench
 *  @endcode
 */
#include <string> // `std::string`

#include <fmt/format.h>

#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

/** The engine's init over the CPU's capabilities, in the shape of its init kernels. */
sz_status_t overlap_engine_init_cpu_(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                     sz_size_t const *window_widths, sz_size_t window_widths_count,
                                     sz_size_t candidates_budget, sz_size_t ordinal, sz_memory_allocator_t *allocator,
                                     void *stream) {
    return sz_overlap_engine_init(engine, queries, window_widths, window_widths_count, candidates_budget,
                                  sz::default_capabilities(), ordinal, allocator, stream);
}

} // namespace

void bench_overlap(corpora_t &corpora) {
    environment_t const &env = corpora.multilingual_lines();
    std::size_t const candidates = candidates_per_call(env);
    overlap_query_t const &query = overlap_median_query(env);
    fmt::println("Starting window overlap benchmarks...");
    bench_unary(env, "sz_overlap_scores:w" + std::to_string(query.width),
                scores_from_sz<overlap_engine_init_cpu_, sz_overlap_scores> {env, query, candidates})
        .log();
}
