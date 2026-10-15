/**
 *  @file bench/subdocuments.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Benchmarks batched path queries on the CPU: one compiled vocabulary of paths against
 *      every document of the corpus.
 *
 *  Times the engine's verbs over one document per line, which pick their kernel from the CPU's
 *  capabilities. Every capability's kernels are timed against the serial ones by the
 *  `cross_<arch>.cpp` files, through the adapters in `cross.hpp`.
 *
 *  The documents are the JSON events of a GH Archive hour, queried with StringWars' JSON pointers'
 *  paths and with richer JSONPath queries holding a wildcard, a descendant segment and a filter. A
 *  `.xml` dataset, like a Stack Exchange `Posts.xml`, runs StringWars' XPath attributes over its
 *  `<row>` elements instead. A line that is not a whole document once trimmed, like the header of
 *  an XML dump or the line the read limit cuts, is left out, as StringWars leaves it out.
 *
 *  Every row reports the document bytes one call walks. There is no Standard row: the platform
 *  ships no JSON or XML parser, so each verb is its own reference, and the kernels in the cross
 *  files are logged against the serial one.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=gharchive/2024-01-01-15.json STRINGWARS_FILTER=subdocuments build_release/stringzilla_bench
 *  @endcode
 */
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`

#include <fmt/format.h>

#include "cross.hpp" // `subdocuments_vocabularies`, `subdocuments_counts_from_sz`

namespace ashvardanian::stringzilla::bench {

void bench_subdocuments(environment_t &env) {
    if (!subdocuments_selects(env, "")) return;
    sz_allocator_t allocator;
    if (sz_allocator_init_heap(&allocator) != sz_success_k)
        throw std::runtime_error("The heap allocator could not be initialized.");
    corpus_t const &corpus = env.corpora.documents();
    sz_subdocuments_format_t const format = subdocuments_format(corpus);
    substrings_corpus_t const staged(subdocuments_documents(corpus));
    fmt::println("Starting path query benchmarks over {} documents...", staged.views.size());
    for (vocabulary_t const &vocabulary : subdocuments_vocabularies(format)) {
        subdocuments_engine_t engine(vocabulary, format, subdocuments_engine_init_cpu_, allocator);
        std::string const label = vocabulary.label;
        print(bench_unary(env, corpus, "sz_subdocuments_counts" + label,
                          subdocuments_counts_from_sz {sz_subdocuments_counts, engine, staged, staged.haystacks()}));
        print(bench_unary(env, corpus, "sz_subdocuments_find" + label,
                          subdocuments_find_from_sz {sz_subdocuments_find, engine, staged, staged.haystacks()}));
    }
}

} // namespace ashvardanian::stringzilla::bench
