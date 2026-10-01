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
 *  A round scores @c STRINGWARS_BATCH_PER_CORE candidates per GPU core, or 2048 without it, the
 *  core count coming from IOKit, as Metal reports none.
 *
 *  @code{.sh}
 *  cmake --preset metal -D STRINGZILLA_BUILD_BENCH=ON
 *  cmake --build build_metal --target stringzilla_metal_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_metal/stringzilla_metal_bench
 *  @endcode
 */
#include <cmath>   // `std::ceil`, `std::log2`
#include <cstring> // `std::memcpy`

#include <dlfcn.h> // `dlopen`, `dlsym`

#include <algorithm> // `std::min`
#include <optional>  // `std::optional`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`, `std::to_string`
#include <vector>    // `std::vector`

#include <CoreFoundation/CoreFoundation.h> // `CFNumberGetValue`, `CFRelease`
#include <fmt/format.h>

#include <stringzilla/overlap.h> // `sz_overlap_*`

#include "harness.hpp"

namespace ashvardanian::stringzilla::bench {

/** The width the corpus's collision entropy picks for a query of @p query_bytes against a mean
 *  candidate of the corpus, as in `cross_simt.cuh`. */
static std::size_t overlap_width_(corpus_t const &corpus, std::size_t query_bytes) {
    double counts[256] = {};
    for (char const byte : corpus.dataset) counts[static_cast<unsigned char>(byte)] += 1.0;
    double collisions = 0.0;
    for (double const count : counts) collisions += count * count;
    double const total = static_cast<double>(corpus.dataset.size());
    double const collision_entropy = -std::log2(collisions / (total * total));
    std::size_t token_bytes = 0;
    for (token_view_t const token : corpus.tokens) token_bytes += token.size();
    double const mean_candidate_bytes = static_cast<double>(token_bytes) / static_cast<double>(corpus.tokens.size());
    double const width = std::ceil(std::log2(static_cast<double>(query_bytes) * mean_candidate_bytes) /
                                   collision_entropy);
    return width > 1.0 ? static_cast<std::size_t>(width) : 1;
}

/** The cores of the first Apple GPU, as IOKit's `gpu-core-count`, loaded at run time as the build
 *  does not link IOKit. A GPU without that entry, like a virtual machine's, counts as one core. */
static std::size_t metal_core_count_() {
    using matching_t = CFMutableDictionaryRef (*)(char const *);
    using service_t = unsigned (*)(unsigned, CFDictionaryRef);
    using property_t = CFTypeRef (*)(unsigned, CFStringRef, CFAllocatorRef, unsigned);
    using release_t = int (*)(unsigned);
    void *const iokit = dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", RTLD_LAZY);
    if (!iokit) return 1;
    auto const matching = reinterpret_cast<matching_t>(dlsym(iokit, "IOServiceMatching"));
    auto const service_of = reinterpret_cast<service_t>(dlsym(iokit, "IOServiceGetMatchingService"));
    auto const property_of = reinterpret_cast<property_t>(dlsym(iokit, "IORegistryEntryCreateCFProperty"));
    auto const release = reinterpret_cast<release_t>(dlsym(iokit, "IOObjectRelease"));
    unsigned const service = service_of(0, matching("AGXAccelerator"));
    CFTypeRef const property = service ? property_of(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0)
                                       : nullptr;
    int cores = 0;
    if (property) CFNumberGetValue(static_cast<CFNumberRef>(property), kCFNumberIntType, &cores), CFRelease(property);
    if (service) release(service);
    return cores > 0 ? static_cast<std::size_t>(cores) : 1;
}

/** Candidates one round scores: @c STRINGWARS_BATCH_PER_CORE, or 2048, per GPU core, within the
 *  corpus. */
static std::size_t overlap_metal_count_(environment_t const &env, corpus_t const &corpus) {
    std::size_t const per_core = env.settings.candidates_per_core.value_or(2048);
    return std::min<std::size_t>(corpus.tokens.size(), per_core * metal_core_count_());
}

/** Arena bytes the resident corpus needs - texts, views and scores - plus room for the engine. */
static std::size_t overlap_metal_arena_bytes_(environment_t const &env, corpus_t const &corpus) {
    std::size_t const count = overlap_metal_count_(env, corpus);
    std::size_t texts_bytes = 0;
    for (std::size_t index = 0; index != count; ++index) texts_bytes += corpus.tokens[index].size();
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

    overlap_metal_corpus_t(environment_t const &env, corpus_t const &corpus, sz_metal_device_t &device)
        : device(device) {
        count = overlap_metal_count_(env, corpus);
        for (std::size_t index = 0; index != count; ++index) texts_bytes += corpus.tokens[index].size();
        sz_memory_allocator_init_metal(&arena, &device);
        texts = static_cast<char *>(arena.allocate(texts_bytes ? texts_bytes : 1, arena.handle));
        views = static_cast<sz_string_view_t *>(arena.allocate(count * sizeof(sz_string_view_t), arena.handle));
        scores = static_cast<sz_f32_t *>(arena.allocate(count * sizeof(sz_f32_t), arena.handle));
        if (!texts || !views || !scores) throw std::runtime_error("The arena could not hold the corpus.");
        char *cursor = texts;
        for (std::size_t index = 0; index != count; ++index) {
            token_view_t const token = corpus.tokens[index];
            std::memcpy(cursor, token.data(), token.size());
            views[index].start = cursor, views[index].length = token.size();
            cursor += token.size();
        }
        bytes = texts_bytes;
        sz_sequence_from_string_views(views, count, &candidates);
    }
};

static std::string overlap_query_text_(corpus_t const &corpus, std::size_t query_bytes) {
    token_view_t const whole = corpus.tokens[0];
    return std::string(whole.data(), std::min(whole.size(), query_bytes));
}

/** Mixes one round's scores into the value @c STRINGZILLA_STRESS compares across arms. */
static check_value_t overlap_check_value_(sz_f32_t const *scores, std::size_t count) {
    check_value_t mixed = 0;
    for (std::size_t index = 0; index != count; ++index)
        mixed = mixed * 31u + (check_value_t)(scores[index] * 1048576.0f);
    return mixed;
}

/** Scores the resident corpus against the fixed query, entirely on the device. */
struct overlap_scores_from_metal {
    overlap_metal_corpus_t &resident;
    std::size_t windows;
    std::string query;
    sz_overlap_engine_t engine {};

    overlap_scores_from_metal(corpus_t const &corpus, overlap_metal_corpus_t &resident, std::size_t query_bytes,
                              std::size_t width)
        : resident(resident), windows(resident.windows_at(width)), query(overlap_query_text_(corpus, query_bytes)) {
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (sz_overlap_engine_init_metal(&engine, &queries, &scored_width, 1, resident.count, 0, &resident.arena,
                                         &resident.device) != sz_success_k)
            throw std::runtime_error("The device forest could not be prepared.");
    }
    ~overlap_scores_from_metal() { sz_overlap_engine_free(&engine); }
    overlap_scores_from_metal(overlap_scores_from_metal const &) = delete;
    overlap_scores_from_metal &operator=(overlap_scores_from_metal const &) = delete;

    call_result_t operator()(std::size_t) {
        if (sz_overlap_scores_metal(&engine, &resident.candidates, resident.scores, resident.count, 1,
                                    &resident.device) != sz_success_k)
            throw std::runtime_error("The GPU round failed.");
        if (sz_metal_device_synchronize(&resident.device) != sz_success_k)
            throw std::runtime_error("The GPU round did not finish.");
        return call_result_t(resident.bytes, overlap_check_value_(resident.scores, resident.count), windows);
    }
};

/** The same round on the CPU, over the very bytes the device reads. */
template <sz_kernel_overlap_engine_init_t init_, sz_kernel_overlap_scores_t scores_>
struct overlap_scores_from_sz {
    overlap_metal_corpus_t &resident;
    std::size_t windows;
    std::string query;
    sz_memory_allocator_t allocator;
    std::vector<sz_f32_t> scores;
    sz_overlap_engine_t engine {};

    overlap_scores_from_sz(corpus_t const &corpus, overlap_metal_corpus_t &resident, std::size_t query_bytes,
                           std::size_t width)
        : resident(resident), windows(resident.windows_at(width)), query(overlap_query_text_(corpus, query_bytes)),
          scores(resident.count) {
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
        if (scores_(&engine, &resident.candidates, scores.data(), scores.size(), 1, nullptr) != sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        return call_result_t(resident.bytes, overlap_check_value_(scores.data(), scores.size()), windows);
    }
};

static void bench_overlap_scores(environment_t const &env, corpus_t const &corpus, overlap_metal_corpus_t &resident,
                                 std::size_t query_bytes) {
    std::size_t const width = overlap_width_(corpus, query_bytes);
    std::string const suffix = ":w" + std::to_string(width);
    auto validator = overlap_scores_from_sz<sz_overlap_engine_init_serial, sz_overlap_scores_serial> {
        corpus, resident, query_bytes, width};
    std::string const serial_name = "sz_overlap_scores_serial" + suffix,
                      metal_name = "sz_overlap_scores_metal" + suffix;
    bool const serial_printed = env.settings.selects(serial_name);
    // ? Timed under the Metal row's name, so a filter naming only that row still times its baseline
    std::optional<row_t> base = bench_unary(env, corpus, serial_printed ? serial_name : metal_name, validator);
    if (base) base->name = serial_name;
    if (serial_printed) print(base);
    print(bench_unary(env, corpus, metal_name, validator,
                      overlap_scores_from_metal {corpus, resident, query_bytes, width}),
          baseline_of(base));
}

} // namespace ashvardanian::stringzilla::bench

using namespace ashvardanian::stringzilla::bench;

int main() {
    install_bench_signal_handlers();
    environment_t env {read_settings(), probe_machine()};
    print(env.machine);
    sz_metal_device_t device {};
    try {
        corpus_t const &corpus = env.corpora.multilingual_lines();
        // ? Left zeroed, and so printed as no device, on a machine without a GPU
        sz_metal_device_init(0, overlap_metal_arena_bytes_(env, corpus), &device);
        print(device);
        print(env.settings);
        if (!device.device) return 0;
        overlap_metal_corpus_t resident(env, corpus, device);
        fmt::println("Starting window overlap benchmarks over {} resident candidates...", resident.count);
        bench_overlap_scores(env, corpus, resident, median_token_bytes(corpus));
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
