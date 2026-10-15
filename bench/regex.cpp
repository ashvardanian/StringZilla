/**
 *  @file bench/regex.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Benchmarks multi-pattern regular expressions on the CPU: one compiled vocabulary over
 *      every line of the corpus.
 *
 *  Times the engine's verbs over the lines of the multilingual slice, which pick their kernel from
 *  the CPU's capabilities. Every capability's kernels are timed against the serial ones by the
 *  `cross_<arch>.cpp` files, through the adapters in `cross.hpp`.
 *
 *  The vocabularies are StringWars' three: literal keywords, everyday patterns like emails, URLs,
 *  dates, numbers and word-bounded keywords, and lookarounds. Each is counted, found and rewritten,
 *  and the year, month and day of every line's first ISO date are extracted.
 *
 *  Compilation is timed separately, because a pipeline that recompiles per query is bound by that
 *  rather than by the walk. A compilation row reports pattern bytes as its bytes and operations;
 *  every other row reports the corpus bytes one call walks.
 *
 *  There is no Standard row: `<regex>` searches one pattern at a time and spells no lookbehind, so
 *  each verb is its own reference, and the kernels in the cross files are logged against serial.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_FILTER=regex build_release/stringzilla_bench
 *  @endcode
 */
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`

#include <fmt/format.h>

#include "cross.hpp" // `regex_vocabularies_k`, `regex_counts_from_sz`

namespace ashvardanian::stringzilla::bench {

/** Compiles the vocabulary from scratch: the cost a pipeline pays once rather than per haystack. */
struct regex_build_from_sz_t {

    /** The patterns to compile. */
    vocabulary_t const &vocabulary;
    sz_capability_t capabilities;

    call_result_t operator()(std::size_t) const {
        sz_allocator_t allocator;
        sz_regex_engine_t engine;
        if (sz_allocator_init_heap(&allocator) != sz_success_k)
            throw std::runtime_error("The heap allocator could not be initialized.");
        sz_strings_view_t const patterns = vocabulary.strings();
        if (sz_regex_engine_init(&engine, &patterns, sz_cased_k, nullptr, capabilities, &allocator, nullptr) !=
            sz_success_k)
            throw std::runtime_error("The patterns would not compile.");
        check_value_t const mixed = (check_value_t)engine.forward_states_count * 31u + engine.reverse_states_count;
        sz_regex_engine_free(&engine, nullptr);
        std::size_t bytes = 0;
        for (sz_string_view_t const &pattern : vocabulary.texts) bytes += pattern.length;
        call_result_t result(bytes, mixed, bytes);
        return result;
    }
};

void bench_regex(environment_t &env) {
    sz_allocator_t allocator;
    if (sz_allocator_init_heap(&allocator) != sz_success_k)
        throw std::runtime_error("The heap allocator could not be initialized.");
    corpus_t const &corpus = env.corpora.multilingual_slice();
    substrings_corpus_t const staged(corpus);
    fmt::println("Starting regex benchmarks...");
    for (vocabulary_t const &vocabulary : regex_vocabularies_k) {
        if (!regex_selects(env, "", vocabulary)) continue;
        std::string const label = vocabulary.label;
        print(bench_unary(env, corpus, "sz_regex_engine_init" + label,
                          regex_build_from_sz_t {vocabulary, env.machine.enabled}));
        regex_engine_t engine(vocabulary, regex_engine_init_cpu_, allocator);
        print(bench_unary(env, corpus, "sz_regex_counts" + label,
                          regex_counts_from_sz {sz_regex_counts, engine, staged, staged.haystacks()}));
        print(bench_unary(env, corpus, "sz_regex_find" + label,
                          regex_find_from_sz {sz_regex_find, engine, staged, staged.haystacks()}));
        print(bench_unary(env, corpus, "sz_regex_replace" + label,
                          regex_replace_from_sz {sz_regex_replace, engine, staged, staged.haystacks(),
                                                 regex_replacements(vocabulary)}));
    }
    if (!regex_selects(env, "", regex_date_k)) return;
    regex_engine_t engine(regex_date_k, regex_engine_init_cpu_, allocator);
    print(bench_unary(env, corpus, std::string("sz_regex_extract") + regex_date_k.label,
                      regex_extract_from_sz {sz_regex_extract, engine, staged, staged.haystacks()}));
}

} // namespace ashvardanian::stringzilla::bench
