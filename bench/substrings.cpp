/**
 *  @file bench/substrings.cpp
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief Benchmarks multi-pattern search on the CPU: one compiled vocabulary against the corpus.
 *
 *  Times the engine's verbs over the multilingual lines, which pick their kernel from the CPU's
 *  capabilities. Every capability's kernels are timed against the serial ones by the
 *  `cross_<arch>.cpp` files, through the adapters in `cross.hpp`.
 *
 *  Memory-bound rather than compute-bound: the walk is one data-dependent load per byte, so what a
 *  row measures is how often that load hits a cache line the automaton already brought in.
 *  Vocabulary shape is what moves that, so every verb is measured twice - against the most frequent
 *  slice of the corpus vocabulary, where accepting states are common, and against the least
 *  frequent slice, where the walk sits near the root and almost never reports.
 *
 *  Compilation is timed separately, because a pipeline that recompiles per query is bound by that
 *  rather than by the walk, and because its cost scales with the vocabulary while every other row
 *  scales with the corpus. A compilation row reports needle bytes as its bytes and operations, and
 *  needles as its inputs; every other row reports the corpus bytes one call walks, and its
 *  haystacks as inputs.
 *
 *  There is no Standard row: the platform ships no multi-pattern search, so each verb is its own
 *  reference, and the kernels in the cross files are logged against the serial one.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_FILTER=substrings \
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  This file is the sibling of `substrings.cu`.
 */
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`

#include <fmt/format.h>

#include "cross.hpp" // `substrings_vocabularies`, `substrings_counts_from_sz`

namespace ashvardanian::stringzilla::bench {

#pragma region Compilation

/** Compiles the vocabulary from scratch: the cost a pipeline pays once rather than per haystack. */
struct substrings_build_from_sz {

    /** The needles to compile, and the sensitivity to compile them at. */
    substrings_dictionary_t const &dictionary;

    call_result_t operator()(std::size_t) const {
        sz_allocator_t allocator;
        sz_substrings_engine_t engine;
        sz_allocator_init_default(&allocator);
        if (sz_substrings_engine_init(&engine, &dictionary.needle_sequence, dictionary.sensitivity,
                                      sz_substrings_overlapping_k, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0,
                                      sz::default_capabilities(), &allocator, nullptr) != sz_success_k)
            throw std::runtime_error("The vocabulary would not compile.");
        check_value_t const mixed = (check_value_t)engine.state_count * 31u + engine.max_outputs_per_state;
        sz_substrings_engine_free(&engine, nullptr);
        call_result_t result(dictionary.needle_bytes, mixed, dictionary.needle_bytes);
        return result;
    }
};

#pragma endregion Compilation

#pragma region Verbs

/** One vocabulary slice, compiled, then walked by every verb under every policy it accepts. */
void bench_substrings_slice(environment_t const &env, corpus_t const &corpus, substrings_corpus_t const &staged,
                            substrings_vocabulary_t const &vocabulary) {
    substrings_dictionary_t const &dictionary = vocabulary.dictionary;
    std::string const &suffix = vocabulary.label;
    if (dictionary.needles.empty()) {
        fmt::println("Vocabulary {} is empty on this corpus, skipping it.", suffix.c_str());
        return;
    }
    {
        substrings_engine_t probe(dictionary, sz_substrings_overlapping_k, substrings_residency_t::host_k);
        fmt::println("Vocabulary {} holds {} needles over {} states, {} of them hot.", suffix.c_str(),
                     dictionary.needles.size(), probe.engine.state_count, probe.engine.hot_count);
    }

    print(bench_unary(env, corpus, "sz_substrings_engine_init" + suffix, substrings_build_from_sz {dictionary}));
    for (sz_substrings_overlap_policy_t const policy : substrings_policies_k) {
        substrings_engine_t engine(dictionary, policy, substrings_residency_t::host_k);
        std::string const cover = substrings_cover(vocabulary, policy);
        print(bench_unary(env, corpus, "sz_substrings_counts" + cover,
                          substrings_counts_from_sz<sz_substrings_counts> {engine, staged, staged.haystacks}));
        print(bench_unary(env, corpus, "sz_substrings_find" + cover,
                          substrings_find_from_sz<sz_substrings_find> {engine, staged, staged.haystacks}));
    }
    for (sz_substrings_overlap_policy_t const policy : substrings_leftmost_policies_k) {
        substrings_engine_t engine(dictionary, policy, substrings_residency_t::host_k);
        print(bench_unary(env, corpus, "sz_substrings_replace" + substrings_cover(vocabulary, policy),
                          substrings_replace_from_sz<sz_substrings_replace> {engine, staged, staged.haystacks,
                                                                             dictionary.replacements}));
    }
    substrings_engine_t engine(dictionary, sz_substrings_overlapping_k, substrings_residency_t::host_k);
    print(bench_unary(env, corpus, "sz_substrings_bm25_scores" + suffix,
                      substrings_bm25_from_sz<sz_substrings_bm25_scores> {engine, staged, staged.haystacks}));
}

#pragma endregion Verbs

void bench_substrings(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_lines();
    substrings_corpus_t const staged(corpus);
    fmt::println("Starting multi-pattern search benchmarks...");
    for (substrings_vocabulary_t const &vocabulary : substrings_vocabularies(env, corpus))
        bench_substrings_slice(env, corpus, staged, vocabulary);
}

} // namespace ashvardanian::stringzilla::bench
