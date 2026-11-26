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

#if STRINGZILLA_ARCH_METAL_
#include <optional>  // `std::optional`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "stringzilla/ngrams.h" // `sz_ngrams_*`

#endif

namespace ashvardanian::stringzilla::bench {

#if STRINGZILLA_TARGET_METAL
static void bench_ngrams_overlaps(environment_t const &env, corpus_t const &corpus, overlap_corpus_t &resident,
                                  std::size_t query_bytes, device_backend_t const &runtime) {
    std::size_t const width = overlap_width_(corpus, query_bytes);
    auto const query = std::string_view(corpus.tokens[0]).substr(0, query_bytes);
    auto validator = overlaps_from_sz {sz_ngrams_engine_init_serial,
                                       sz_ngrams_overlaps_serial,
                                       query,
                                       width,
                                       resident.host_candidates(),
                                       resident.views};
    std::size_t const ordinal = runtime.selected.ordinal();
    std::string const metal_name = fmt::format("sz_ngrams_overlaps_metal:{}:w{}", ordinal, width);
    auto dispatched = overlaps_from_sz {ngrams_engine_init_cpu_,    sz_ngrams_overlaps, query, width,
                                        resident.host_candidates(), resident.views};
    std::optional<double> const base = bench_baseline(
        env, corpus, fmt::format("sz_ngrams_overlaps:metal:{}:w{}", ordinal, width), {metal_name}, dispatched);
    print(bench_unary(env, corpus, metal_name, validator,
                      overlaps_from_sz {sz_ngrams_engine_init_metal, sz_ngrams_overlaps_metal, query, width,
                                        resident.candidates.view().raw, resident.views, std::cref(runtime)}),
          base);
}

#endif

int bench_cross_metal(environment_t &env, std::size_t ordinal) {
#if STRINGZILLA_ARCH_METAL_
    stream_t queue(ordinal, sz_stream_init_metal, sz_stream_free_metal);
    device_backend_t runtime;
    runtime.selected = sz::device_t::make(sz::device_kind_t::metal_k, ordinal).value;
    runtime.capabilities = sz_cap_metal_k;
    runtime.stream = queue.handle;
    if (sz_allocator_init_unified_metal(&runtime.unified) != sz_success_k)
        throw std::runtime_error("The Metal allocator could not be initialized.");
    fmt::println("Device: metal:{}", ordinal);
#if STRINGZILLA_TARGET_METAL
    corpus_t const &corpus = env.corpora.multilingual_lines();
    std::size_t const count = min_of(corpus.tokens.size(), resident_candidates_per_call(env, runtime));
    std::vector<sz_string_view_t> views(count);
    for (std::size_t index = 0; index != count; ++index)
        views[index] = {corpus.tokens[index].data(), corpus.tokens[index].size()};
    overlap_corpus_t resident(views, runtime);
    fmt::println("Starting window overlap benchmarks over {} resident candidates...", resident.candidates.size());
    bench_ngrams_overlaps(env, corpus, resident, median_token_bytes(corpus), runtime);
#else
    sz_unused_(env);
#endif
    return 0;
#else
    sz_unused_(env);
    sz_unused_(ordinal);
    return 0;
#endif
}

} // namespace ashvardanian::stringzilla::bench
