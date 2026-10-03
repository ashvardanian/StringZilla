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
 *  so a 64 MB slice exercises every path.
 *
 *  Five arms are measured per backend, each reporting the windows it touched as @c operations, so
 *  the ops-per-second column reads as windows per second. The first three nest, so a stage's own
 *  cost is the difference between neighbouring arms:
 *  - @c prefix_hashes - the prefix hashes alone, one per byte, whatever the width;
 *  - @c window_hashes - the prefix hashes, then their differences at the derived width;
 *  - @c window_lookups - the prefix hashes, the window hashes, then the B-tree walk over them all;
 *  - @c query_preparation - the query's key sort and tree layout, once per call, the token ignored
 *    - at an 8 KB query this is most of a round, so it stands on its own;
 *  - @c scores - the engine's round against the next @c STRINGWARS_BATCH_PER_CORE tokens, by
 *    default as many median tokens as fill a 32 KB L1: the forest built once before the timing,
 *    four chains interleaved inside it.
 *
 *  The query is the slice's median token length. The window width is derived from the slice rather
 *  than fixed, and every arm's name carries it, as `:w6`. With H₂ the byte collision entropy of
 *  the slice, the width is ⌈log₂(query bytes × mean candidate bytes) / H₂⌉.
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

namespace ashvardanian::stringzilla::bench {

/** The engine's init over the CPU's capabilities, in the shape of its init kernels. */
sz_status_t overlap_engine_init_cpu_(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                     sz_size_t const *window_widths, sz_size_t window_widths_count,
                                     sz_size_t candidates_budget, sz_memory_allocator_t *allocator, void *stream) {
    return sz_overlap_engine_init(engine, queries, window_widths, window_widths_count, candidates_budget,
                                  sz::default_capabilities(), allocator, stream);
}

void bench_overlap(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_lines();
    std::size_t const candidates = candidates_per_call(env, corpus);
    overlap_query_t const query(corpus, median_token_bytes(corpus));
    fmt::println("Starting window overlap benchmarks...");
    print(bench_unary(env, corpus, "sz_overlap_scores:w" + std::to_string(query.width),
                      scores_from_sz<overlap_engine_init_cpu_, sz_overlap_scores> {corpus, query, candidates}));
}

} // namespace ashvardanian::stringzilla::bench
