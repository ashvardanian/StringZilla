/**
 *  @file bench/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief GPU engine benchmarks - the Metal kernels against the CPU tiers this build carries, the
 *      twin of `cross_simt.cuh`.
 *
 *  Metal kernels reach only the device's arena, so the corpus is copied into it once - texts, views
 *  and the scores - and every call scores that resident slice in place. The engine is built before
 *  the timing on both sides; only the round is timed, including the wait for the device, as the
 *  CUDA twin times its stream synchronization.
 *
 *  The environment variables are the ones `cross_simt.cuh` reads, with @c STRINGWARS_BATCH
 *  overriding the 65536 candidates a round scores by default.
 *
 *  @code{.sh}
 *  cmake --preset metal -D STRINGZILLA_BUILD_BENCH=ON
 *  cmake --build build_metal --target stringzilla_metal_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_metal/stringzilla_metal_bench
 *  @endcode
 */
#include <algorithm> // `std::min`
#include <cmath>     // `std::ceil`, `std::log2`
#include <cstring>   // `std::memcpy`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`, `std::to_string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include <stringzilla/overlap.h> // `sz_overlap_*`

#include "harness.hpp"

using namespace ashvardanian::stringzilla::bench;

/** The width the corpus's collision entropy picks for a query of @p query_bytes against a mean
 *  candidate of the corpus, as in `cross_simt.cuh`. */
static std::size_t overlap_width_(environment_t const &env, std::size_t query_bytes) {
    double counts[256] = {};
    for (char const byte : env.dataset) counts[static_cast<unsigned char>(byte)] += 1.0;
    double collisions = 0.0;
    for (double const count : counts) collisions += count * count;
    double const total = static_cast<double>(env.dataset.size());
    double const collision_entropy = -std::log2(collisions / (total * total));
    std::size_t token_bytes = 0;
    for (token_view_t const token : env.tokens) token_bytes += token.size();
    double const mean_candidate_bytes = static_cast<double>(token_bytes) / static_cast<double>(env.tokens.size());
    double const width = std::ceil(std::log2(static_cast<double>(query_bytes) * mean_candidate_bytes) /
                                   collision_entropy);
    return width > 1.0 ? static_cast<std::size_t>(width) : 1;
}

/** Candidates one round scores: @c STRINGWARS_BATCH, or 65536, within the corpus. */
static std::size_t overlap_metal_count_(environment_t const &env) {
    std::size_t const wanted = env.batch_sizes_override.empty() ? 65536 : env.batch_sizes_override.front();
    return std::min<std::size_t>(env.tokens.size(), wanted);
}

/** Arena bytes the resident corpus needs - texts, views and scores - plus room for the engine. */
static std::size_t overlap_metal_arena_bytes_(environment_t const &env) {
    std::size_t const count = overlap_metal_count_(env);
    std::size_t texts_bytes = 0;
    for (std::size_t index = 0; index != count; ++index) texts_bytes += env.tokens[index].size();
    return texts_bytes + count * (sizeof(sz_string_view_t) + sizeof(sz_f32_t)) + (64u << 20);
}

/** The corpus as @c device sees it: texts, views and one round's scores, all inside its arena. */
struct overlap_metal_corpus_t {
    sz_metal_device_t &device;
    sz_memory_allocator_t arena {};
    char *texts = nullptr;
    sz_string_view_t *views = nullptr;
    sz_f32_t *scores = nullptr;
    std::size_t count = 0, texts_bytes = 0;
    sz_sequence_t candidates {};

    /** Candidate bytes one round touches, which throughput divides by. */
    std::size_t bytes = 0;

    std::size_t windows_at(std::size_t width) const noexcept {
        std::size_t total = 0;
        for (std::size_t index = 0; index != count; ++index)
            total += width <= views[index].length ? views[index].length - width + 1 : 0;
        return total;
    }

    overlap_metal_corpus_t(environment_t const &env, sz_metal_device_t &device) : device(device) {
        count = overlap_metal_count_(env);
        for (std::size_t index = 0; index != count; ++index) texts_bytes += env.tokens[index].size();
        sz_memory_allocator_init_metal(&arena, &device);
        texts = static_cast<char *>(arena.allocate(texts_bytes ? texts_bytes : 1, arena.handle));
        views = static_cast<sz_string_view_t *>(arena.allocate(count * sizeof(sz_string_view_t), arena.handle));
        scores = static_cast<sz_f32_t *>(arena.allocate(count * sizeof(sz_f32_t), arena.handle));
        if (!texts || !views || !scores) throw std::runtime_error("The arena could not hold the corpus.");
        char *cursor = texts;
        for (std::size_t index = 0; index != count; ++index) {
            token_view_t const token = env.tokens[index];
            std::memcpy(cursor, token.data(), token.size());
            views[index].start = cursor, views[index].length = token.size();
            cursor += token.size();
        }
        bytes = texts_bytes;
        sz_sequence_from_string_views(views, count, &candidates);
    }
};

static std::string overlap_query_text_(environment_t const &env, std::size_t query_bytes) {
    token_view_t const whole = env.tokens[0];
    return std::string(whole.data(), std::min(whole.size(), query_bytes));
}

/** Mixes one round's scores into the value @c STRINGWARS_STRESS compares across arms. */
static check_value_t overlap_check_value_(sz_f32_t const *scores, std::size_t count) {
    check_value_t mixed = 0;
    for (std::size_t index = 0; index != count; ++index)
        mixed = mixed * 31u + (check_value_t)(scores[index] * 1048576.0f);
    return mixed;
}

/** Scores the resident corpus against the fixed query, entirely on the device. */
struct overlap_scores_from_metal {
    overlap_metal_corpus_t &corpus;
    std::size_t windows;
    std::string query;
    sz_overlap_engine_t engine {};

    overlap_scores_from_metal(environment_t const &env, overlap_metal_corpus_t &corpus, std::size_t query_bytes,
                              std::size_t width)
        : corpus(corpus), windows(corpus.windows_at(width)), query(overlap_query_text_(env, query_bytes)) {
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (sz_overlap_engine_init_metal(&engine, &queries, &scored_width, 1, corpus.count, 0, &corpus.arena,
                                         &corpus.device) != sz_success_k)
            throw std::runtime_error("The device forest could not be prepared.");
    }
    ~overlap_scores_from_metal() { sz_overlap_engine_free(&engine); }
    overlap_scores_from_metal(overlap_scores_from_metal const &) = delete;
    overlap_scores_from_metal &operator=(overlap_scores_from_metal const &) = delete;

    call_result_t operator()(std::size_t) {
        if (sz_overlap_scores_metal(&engine, &corpus.candidates, corpus.scores, corpus.count, 1, &corpus.device) !=
            sz_success_k)
            throw std::runtime_error("The GPU round failed.");
        if (sz_metal_device_synchronize(&corpus.device) != sz_success_k)
            throw std::runtime_error("The GPU round did not finish.");
        call_result_t result(corpus.bytes, overlap_check_value_(corpus.scores, corpus.count), windows);
        result.inputs_processed = corpus.count;
        return result;
    }
};

/** The same round on the CPU, over the very bytes the device reads. */
template <sz_kernel_overlap_engine_init_t init_, sz_kernel_overlap_scores_t scores_>
struct overlap_scores_from_sz {
    overlap_metal_corpus_t &corpus;
    std::size_t windows;
    std::string query;
    sz_memory_allocator_t allocator;
    std::vector<sz_f32_t> scores;
    sz_overlap_engine_t engine {};

    overlap_scores_from_sz(environment_t const &env, overlap_metal_corpus_t &corpus, std::size_t query_bytes,
                           std::size_t width)
        : corpus(corpus), windows(corpus.windows_at(width)), query(overlap_query_text_(env, query_bytes)),
          scores(corpus.count) {
        sz_memory_allocator_init_default(&allocator);
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (init_(&engine, &queries, &scored_width, 1, 0, 0, &allocator, nullptr) != sz_success_k)
            throw std::runtime_error("The host forest could not be prepared.");
    }
    ~overlap_scores_from_sz() { sz_overlap_engine_free(&engine); }
    overlap_scores_from_sz(overlap_scores_from_sz const &) = delete;
    overlap_scores_from_sz &operator=(overlap_scores_from_sz const &) = delete;

    call_result_t operator()(std::size_t) {
        if (scores_(&engine, &corpus.candidates, scores.data(), scores.size(), 1, nullptr) != sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        call_result_t result(corpus.bytes, overlap_check_value_(scores.data(), scores.size()), windows);
        result.inputs_processed = corpus.count;
        return result;
    }
};

static void bench_overlap_scores(environment_t const &env, overlap_metal_corpus_t &corpus, std::size_t query_bytes) {
    std::size_t const width = overlap_width_(env, query_bytes);
    std::string const suffix = ":w" + std::to_string(width);
    auto validator = overlap_scores_from_sz<sz_overlap_engine_init_serial, sz_overlap_scores_serial> {
        env, corpus, query_bytes, width};
    bench_result_t base = bench_unary(env, std::string("sz_overlap_scores_serial") + suffix, validator).log();
    bench_unary(env, std::string("sz_overlap_scores_metal") + suffix, validator,
                overlap_scores_from_metal {env, corpus, query_bytes, width})
        .log(base);
}

int main(int argc, char const **argv) {
    install_bench_signal_handlers();
    log_environment();
    print_bench_environment();

    sz_metal_device_t device {};
    try {
        fmt::println("Building up the environment...");
        environment_t env = build_environment(argc, argv, "xlsum.csv", environment_t::tokenization_t::lines_k);
        // ? Left zeroed, and so logged as no device, on a machine without a GPU
        sz_metal_device_init(0, overlap_metal_arena_bytes_(env), &device);
        if (!log_metal_device(device)) return 0;
        overlap_metal_corpus_t corpus(env, device);
        fmt::println("Starting window overlap benchmarks over {} resident candidates...", corpus.count);
        bench_overlap_scores(env, corpus, median_token_bytes(env));
    }
    catch (std::exception const &e) {
        sz_metal_device_free(&device);
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }
    sz_metal_device_free(&device);

    fmt::println("All benchmarks passed.");
    return 0;
}
