/**
 *  @file bench/ngrams.cpp
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Benchmarks N-gram overlaps, N-gram residues and MinHash sketches through the public API.
 *
 *  The query tables are prepared once before timing. Each overlap round scores the next
 *  @c STRINGWARS_BATCH_PER_CORE candidates and reports N-grams per second, and the cross files
 *  compare each capability's overlap kernel with the serial baseline.
 *
 *  The query is the slice's median token length. The window width is derived from the slice rather
 *  than fixed, and every arm's name carries it, as <tt>:w6</tt>. With H₂ the byte collision entropy of
 *  the slice, the width is ⌈log₂(query bytes × mean candidate bytes) / H₂⌉.
 *
 *  Residue and sketch rounds take the next @c STRINGWARS_BATCH_PER_CORE lines over N-grams of five
 *  and eight bytes, on every CPU capability that has a kernel of its own, each held to the serial
 *  one, and report texts per second as "ops". Sketch rows carry their slot count, as <tt>:s512</tt>.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_FILTER=ngrams build_release/stringzilla_bench
 *  @endcode
 */
#include <span>      // `std::span`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

static constexpr sz_size_t ngrams_widths_k[] = {5, 8};

using ngrams_residues_t = void (*)(sz_cptr_t, sz_size_t, sz_size_t const *, sz_size_t, sz_f64_t *, sz_u32_t *);

/** One capability's residues kernel, timed only where this CPU runs it. */
struct ngrams_residues_kernel_t {
    sz_capability_t capability;
    ngrams_residues_t residues;
};

static constexpr ngrams_residues_kernel_t ngrams_residues_kernels_k[] = {
    {sz_cap_serial_k, sz_ngrams_residues_serial},
#if STRINGZILLA_TARGET_HASWELL
    {sz_cap_haswell_k, sz_ngrams_residues_haswell},
#endif
#if STRINGZILLA_TARGET_SKYLAKE
    {sz_cap_skylake_k, sz_ngrams_residues_skylake},
#endif
#if STRINGZILLA_TARGET_NEON
    {sz_cap_neon_k, sz_ngrams_residues_neon},
#endif
#if STRINGZILLA_TARGET_POWERVSX
    {sz_cap_powervsx_k, sz_ngrams_residues_powervsx},
#endif
};

/** Hashes every window of rolling batches of a corpus's tokens into residues with one kernel. */
struct ngrams_residues_callable_t {
    corpus_t const &corpus;
    ngrams_residues_t residues_kernel;
    std::size_t texts;
    std::vector<sz_f64_t> prefixes;
    std::vector<sz_u32_t> residues;

    ngrams_residues_callable_t(corpus_t const &corpus, ngrams_residues_t residues_kernel, std::size_t texts)
        : corpus(corpus), residues_kernel(residues_kernel), texts(texts) {
        std::size_t longest = 0;
        for (std::string_view const token : corpus.tokens) longest = max_of(longest, token.size());
        prefixes.resize(longest + 1);
        residues.resize(sz_ngrams_count(longest, ngrams_widths_k, 2));
    }

    call_result_t operator()(std::size_t token_index) {
        std::size_t bytes = 0;
        check_value_t mixed = 0;
        for (std::size_t text = 0; text != texts; ++text) {
            std::string_view const token = corpus.tokens[(token_index + text) % corpus.tokens.size()];
            residues_kernel(token.data(), token.size(), ngrams_widths_k, 2, prefixes.data(), residues.data());
            std::size_t const ngrams = sz_ngrams_count(token.size(), ngrams_widths_k, 2);
            for (sz_u32_t const residue : std::span(residues.data(), ngrams)) mixed = mixed * 31u + residue;
            bytes += token.size();
        }
        return call_result_t(bytes, mixed, texts);
    }
};

/** Sketches rolling batches of a corpus's tokens through a query-less engine on one capability. */
struct ngrams_sketches_callable_t {
    corpus_t const &corpus;
    std::size_t slots;
    std::vector<sz_string_view_t> views;
    std::vector<sz_u8_t> sketches;
    sz_ngrams_engine_t engine {};

    ngrams_sketches_callable_t(corpus_t const &corpus, sz_capability_t capabilities, std::size_t slots,
                               std::size_t texts)
        : corpus(corpus), slots(slots), views(texts), sketches(texts * slots) {
        sz_strings_view_t queries {};
        sz_strings_from_string_views(nullptr, 0, &queries);
        if (sz_ngrams_engine_init(&engine, &queries, ngrams_widths_k, 2, 0, capabilities, nullptr, nullptr) !=
            sz_success_k)
            throw std::runtime_error("The engine failed to initialize.");
    }
    ~ngrams_sketches_callable_t() noexcept { sz_ngrams_engine_free(&engine, nullptr); }
    ngrams_sketches_callable_t(ngrams_sketches_callable_t const &) = delete;
    ngrams_sketches_callable_t &operator=(ngrams_sketches_callable_t const &) = delete;

    call_result_t operator()(std::size_t token_index) {
        std::size_t bytes = 0;
        for (std::size_t text = 0; text != views.size(); ++text) {
            std::string_view const token = corpus.tokens[(token_index + text) % corpus.tokens.size()];
            views[text] = {token.data(), token.size()};
            bytes += token.size();
        }
        sz_strings_view_t texts {};
        sz_strings_from_string_views(views.data(), views.size(), &texts);
        if (sz_ngrams_sketches(&engine, &texts, slots, sketches.data(), slots, nullptr) != sz_success_k)
            throw std::runtime_error("The round failed.");
        check_value_t mixed = 0;
        for (sz_u8_t const slot : sketches) mixed = mixed * 31u + slot;
        return call_result_t(bytes, mixed, views.size());
    }
};

void bench_ngrams(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_lines();
    std::size_t const candidates = candidates_per_call(env, corpus);
    overlap_query_t const query(corpus, median_token_bytes(corpus));
    fmt::println("Starting N-gram overlap benchmarks...");
    print(bench_unary(env, corpus, "sz_ngrams_overlaps:w" + std::to_string(query.width),
                      overlaps_from_sz {ngrams_engine_init_cpu_, sz_ngrams_overlaps, corpus, query, candidates}));

    fmt::println("Starting N-gram residue benchmarks...");
    ngrams_residues_callable_t serial_residues {corpus, sz_ngrams_residues_serial, candidates};
    std::optional<double> const residues_baseline = bench_baseline(env, corpus, "sz_ngrams_residues_serial", {},
                                                                   serial_residues);
    for (ngrams_residues_kernel_t const &kernel : std::span(ngrams_residues_kernels_k).subspan(1)) {
        if (!(kernel.capability & cpu_capabilities_enabled())) continue;
        char name[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
        sz_capabilities_name(kernel.capability, name, sizeof(name));
        print(bench_unary(env, corpus, fmt::format("sz_ngrams_residues_{}", name), serial_residues,
                          ngrams_residues_callable_t {corpus, kernel.residues, candidates}),
              residues_baseline);
    }

    fmt::println("Starting MinHash sketch benchmarks...");
    for (std::size_t const slots : {128, 512, 2048}) {
        print(bench_unary(env, corpus, fmt::format("sz_ngrams_sketches:s{}", slots),
                          ngrams_sketches_callable_t {corpus, cpu_capabilities_enabled(), slots, candidates}));
        ngrams_sketches_callable_t serial {corpus, sz_cap_serial_k, slots, candidates};
        std::optional<double> const baseline = bench_baseline(
            env, corpus, fmt::format("sz_ngrams_sketches_serial:s{}", slots), {}, serial);
        for (sz_capability_t capability = sz_cap_serial_k << 1; capability & sz_cap_cpus_k; capability <<= 1) {
            sz_kernel_punned_t kernel = nullptr;
            sz_capability_t picked = 0;
            // A capability without a kernel of its own would time the one below it again.
            if (!(capability & cpu_capabilities_enabled()) ||
                sz_ngrams_find_kernel(sz_kernel_ngrams_sketches_k, capability, &kernel, &picked) != sz_success_k ||
                picked != capability)
                continue;
            char name[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
            sz_capabilities_name(capability, name, sizeof(name));
            print(bench_unary(env, corpus, fmt::format("sz_ngrams_sketches_{}:s{}", name, slots), serial,
                              ngrams_sketches_callable_t {corpus, capability, slots, candidates}),
                  baseline);
        }
    }
}

} // namespace ashvardanian::stringzilla::bench
