/**
 *  @file bench/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief GPU engine benchmarks - the Metal kernels against the CPU tiers this build carries, the
 *      twin of `cross_simt.cuh`.
 *
 *  Metal kernels reach only the device's unified memory, so the corpus is copied into it once, as a
 *  tape beside the scores, and every call scores that resident slice in place. The engine is built
 *  before the timing on both sides; only the round is timed, including the wait for the queue, as
 *  the CUDA twin times its stream synchronization.
 *
 *  Without a core count, Metal uses @c STRINGWARS_BATCH_PER_CORE candidates, or 2048.
 *
 *  @code{.sh}
 *  cmake --preset metal -D STRINGZILLA_BUILD_BENCH=ON
 *  cmake --build build_metal --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_metal/stringzilla_bench
 *  @endcode
 */
#include "cross.hpp"

#if STRINGZILLA_WITH_METAL
#include <algorithm> // `std::min`
#include <optional>  // `std::optional`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`, `std::to_string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "stringzilla/overlap.h" // `sz_overlap_*`

#endif

namespace ashvardanian::stringzilla::bench {

#if STRINGZILLA_WITH_METAL
static void bench_overlap_scores(environment_t const &env, corpus_t const &corpus, overlap_corpus_t &resident,
                                 std::size_t query_bytes, device_backend_t const &runtime) {
    std::size_t const width = overlap_width_(corpus, query_bytes);
    auto const query = std::string_view(corpus.tokens[0]).substr(0, query_bytes);
    std::string const suffix = ":w" + std::to_string(width);
    auto validator = scores_from_sz {sz_overlap_engine_init_serial,
                                     sz_overlap_scores_serial,
                                     query,
                                     width,
                                     resident.host_candidates(),
                                     resident.views};
    std::string const serial_name = "sz_overlap_scores_serial:metal:" + std::to_string(runtime.selected.ordinal()) +
                                    suffix,
                      metal_name = "sz_overlap_scores_metal:" + std::to_string(runtime.selected.ordinal()) + suffix;
    bool const serial_printed = env.settings.selects(serial_name);
    // ? Timed under the Metal row's name, so a filter naming only that row still times its baseline
    std::optional<row_t> base = bench_unary(env, corpus, serial_printed ? serial_name : metal_name, validator);
    if (base) base->name = serial_name;
    if (serial_printed) print(base);
    print(bench_unary(env, corpus, metal_name, validator,
                      scores_from_sz {sz_overlap_engine_init_metal, sz_overlap_scores_metal, query, width,
                                      resident.candidates.sequence(), resident.views, std::cref(runtime)}),
          baseline_of(base));
}

#endif

int bench_cross_metal(environment_t &env, std::size_t ordinal) {
#if STRINGZILLA_WITH_METAL
    stream_t queue(ordinal, sz_stream_init_metal, sz_stream_free_metal);
    device_backend_t runtime;
    runtime.selected = sz::device_t::make(sz::device_kind_t::metal_k, ordinal).value;
    runtime.capabilities = sz_cap_metal_k;
    runtime.stream = queue.handle;
    if (sz_allocator_init_unified_metal(&runtime.unified) != sz_success_k)
        throw std::runtime_error("The Metal allocator could not be initialized.");
    fmt::println("Device: metal:{}", ordinal);
    corpus_t const &corpus = env.corpora.multilingual_lines();
    std::size_t const count = std::min(corpus.tokens.size(), resident_candidates_per_call(env, runtime));
    std::vector<sz_string_view_t> views(count);
    for (std::size_t index = 0; index != count; ++index)
        views[index] = {corpus.tokens[index].data(), corpus.tokens[index].size()};
    overlap_corpus_t resident(views, runtime);
    fmt::println("Starting window overlap benchmarks over {} resident candidates...", resident.candidates.size());
    bench_overlap_scores(env, corpus, resident, median_token_bytes(corpus), runtime);
    return 0;
#else
    sz_unused_(env);
    sz_unused_(ordinal);
    return 0;
#endif
}

} // namespace ashvardanian::stringzilla::bench
