/**
 *  @file bench/overlap.cpp
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Benchmarks prepared-query window overlap through the public engine API.
 *
 *  The query forest is prepared once before timing. Each round scores the next
 *  @c STRINGWARS_BATCH_PER_CORE candidates and reports windows per second.
 *  The cross files compare each capability's score kernel with the serial baseline.
 *
 *  The query is the slice's median token length. The window width is derived from the slice rather
 *  than fixed, and every arm's name carries it, as `:w6`. With H₂ the byte collision entropy of
 *  the slice, the width is ⌈log₂(query bytes × mean candidate bytes) / H₂⌉.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_FILTER=overlap build_release/stringzilla_bench
 *  @endcode
 */
#include <string> // `std::string`

#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

/** The engine's init over the CPU's capabilities, in the shape of its init kernels. */
sz_status_t overlap_engine_init_cpu_(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                     sz_size_t const *window_widths, sz_size_t window_widths_count,
                                     sz_size_t candidates_budget, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_overlap_engine_init(engine, queries, window_widths, window_widths_count, candidates_budget,
                                  sz::default_capabilities(), allocator, stream);
}

void bench_overlap(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_lines();
    std::size_t const candidates = candidates_per_call(env, corpus);
    overlap_query_t const query(corpus, median_token_bytes(corpus));
    fmt::println("Starting window overlap benchmarks...");
    print(bench_unary(env, corpus, "sz_overlap_scores:w" + std::to_string(query.width),
                      scores_from_sz {overlap_engine_init_cpu_, sz_overlap_scores, corpus, query, candidates}));
}

} // namespace ashvardanian::stringzilla::bench
