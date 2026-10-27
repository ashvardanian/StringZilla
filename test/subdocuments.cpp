/**
 *  @file test/subdocuments.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Path query tests: RFC 9535 and XPath known answers, malformed documents and paths, and
 *      the dispatched verbs checked against the serial ones.
 *
 *  The checks live in `cross.hpp`, where every CPU tier runs them by name; this file runs them over
 *  the dispatch points, plus what only a library build compiles: patterns the regex family refuses.
 */
#undef NDEBUG // ! Enable all assertions for testing

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <string> // Baseline
#include <vector> // `std::vector`

#include "stringzilla/stringzilla.h"   // Primary C API
#include "stringzilla/stringzilla.hpp" // `sz::default_capabilities`

#include "cross.hpp"   // `subdocuments_tier_t`, `check_subdocuments_unit_`, `check_subdocuments_safety_`
#include "harness.hpp" // `test_context_t`, `verify`

namespace ashvardanian::stringzilla::test {

#pragma region Helpers

/** The dispatched engine builder over the CPU capabilities, in the shape of its capability kernels,
 *  whose mask sits before the allocator. */
static sz_status_t subdocuments_engine_init_dispatched_(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *paths, sz_subdocuments_format_t format,
                                                        sz_subdocuments_options_t options, sz_size_t *malformed_index,
                                                        sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_subdocuments_engine_init(engine, paths, format, options, malformed_index, sz::default_capabilities(),
                                       allocator, stream);
}

/** The dispatched builder and the verbs that walk whatever capability it compiled for. */
static subdocuments_tier_t const subdocuments_dispatched {subdocuments_engine_init_dispatched_, sz_subdocuments_counts,
                                                          sz_subdocuments_find};

#pragma endregion Helpers

#pragma region Unit Cases

/** RFC 9535's bookstore and examples, XPath over an XML bookstore, refused paths and capacities. */
void test_subdocuments_unit() {
    check_subdocuments_unit_(subdocuments_dispatched);

    // Containment marks the documents counts finds a node in, past one block of 64 and into a
    // partial last byte, whose bits past the last document stay zero; `{}` stands in for an empty
    // document, which is malformed.
    {
        std::vector<std::string> documents(70, R"({"b": [1, 2]})");
        for (std::size_t index = 0; index < documents.size(); index += 3) documents[index] = R"({"a": {"c": 3}})";
        documents[5] = "{}";
        std::vector<std::string> const paths {"$.a.c", "$.d"};
        std::vector<sz_string_view_t> document_views;
        sz_strings_view_t const document_strings = strings_over_(documents, document_views);
        sz_subdocuments_engine_t engine;
        sz_size_t malformed_index = 0;
        verify(subdocuments_compile_(subdocuments_dispatched, &engine, paths, sz_subdocuments_json_k, &malformed_index,
                                     nullptr) == sz_success_k);
        std::vector<sz_size_t> counts(documents.size());
        std::vector<sz_u1x8_t> contained((documents.size() + 7) / 8, 0xFF);
        verify(sz_subdocuments_counts(&engine, &document_strings, counts.data(), 1, &malformed_index, nullptr) ==
               sz_success_k);
        verify(sz_subdocuments_contains(&engine, &document_strings, contained.data(), &malformed_index, nullptr) ==
               sz_success_k);
        for (std::size_t index = 0; index != documents.size(); ++index)
            verify((contained[index >> 3] >> (index & 7) & 1) == (counts[index] != 0));
        verify((contained.back() >> (documents.size() & 7)) == 0);

        // The walk stops at its first match, so a document broken past it is marked, not refused.
        std::vector<std::string> const broken {R"({"a": {"c": 3}, )"};
        std::vector<sz_string_view_t> broken_views;
        sz_strings_view_t const broken_strings = strings_over_(broken, broken_views);
        sz_u1x8_t broken_contained = 0;
        verify(sz_subdocuments_contains(&engine, &broken_strings, &broken_contained, &malformed_index, nullptr) ==
               sz_success_k);
        verify(broken_contained == 1);
        verify(sz_subdocuments_counts(&engine, &broken_strings, counts.data(), 1, &malformed_index, nullptr) ==
               sz_invalid_syntax_k);
        sz_subdocuments_engine_free(&engine, nullptr);
    }

    // RFC 8259 leaves repeated names open: by default each repeat is entered, under the option
    // only the first, while a wildcard enters every member either way.
    {
        std::vector<std::string> const documents {R"({"a":1,"a":2,"b":{"a":3,"a":4}})"};
        std::vector<sz_string_view_t> document_views;
        sz_strings_view_t const document_strings = strings_over_(documents, document_views);
        struct duplicate_case_t {
            char const *path;
            sz_size_t repeated, first;
        };
        for (duplicate_case_t const expected :
             {duplicate_case_t {"$.a", 2, 1}, duplicate_case_t {"$.b.a", 2, 1}, duplicate_case_t {"$.*", 3, 3}}) {
            for (sz_subdocuments_options_t const options :
                 {sz_subdocuments_options_none_k, sz_subdocuments_duplicate_names_first_k}) {
                std::vector<std::string> const paths {expected.path};
                std::vector<sz_string_view_t> path_views;
                sz_strings_view_t const path_strings = strings_over_(paths, path_views);
                sz_subdocuments_engine_t engine;
                sz_size_t malformed_index = 0, count = 0;
                verify(sz_subdocuments_engine_init(&engine, &path_strings, sz_subdocuments_json_k, options,
                                                   &malformed_index, sz::default_capabilities(), nullptr,
                                                   nullptr) == sz_success_k);
                verify(sz_subdocuments_counts(&engine, &document_strings, &count, 1, &malformed_index, nullptr) ==
                       sz_success_k);
                verify(count == (options == sz_subdocuments_options_none_k ? expected.repeated : expected.first));
                sz_subdocuments_engine_free(&engine, nullptr);
            }
        }
    }

    // A bare filter tests the root itself, beside an ordinary path in the same vocabulary.
    {
        sz_subdocuments_engine_t engine;
        sz_size_t malformed_index = 0;
        verify(subdocuments_compile_(subdocuments_dispatched, &engine, {"?@.a > 2", "$.c"}, sz_subdocuments_json_k,
                                     &malformed_index, nullptr) == sz_success_k);
        subdocuments_found_t const found = subdocuments_find_(
            subdocuments_dispatched, &engine, {R"({"a":1,"b":2})", R"({"a":5})", R"({"c":3})", "[1,2]"});
        verify(found.status == sz_success_k);
        verify(found.offsets == std::vector<sz_size_t>({0, 0, 1, 2, 2}));
        verify(found.matches[0].haystack_index == 1 && found.matches[0].needle_index == 0);
        verify(found.matches[0].byte_offset == 0 && found.matches[0].byte_length == 7);
        verify(found.matches[1].haystack_index == 2 && found.matches[1].needle_index == 1);
        verify(found.matches[1].byte_offset == 5 && found.matches[1].byte_length == 1);
        sz_subdocuments_engine_free(&engine, nullptr);
        verify(subdocuments_compile_status_(subdocuments_dispatched, {"?@.a,", "$.a"}, sz_subdocuments_json_k,
                                            malformed_index) == sz_invalid_syntax_k);
        verify(malformed_index == 0);
    }
}

/** What the verbs refuse, which is as much of the contract as what they accept. */
void test_subdocuments_safety(test_context_t &) {
    check_subdocuments_safety_(subdocuments_dispatched);

    // A pattern the regex family refuses refuses the path spelling it, whichever filter holds it.
    sz_size_t malformed_index = 0;
    verify(subdocuments_compile_status_(subdocuments_dispatched, {"$.a", "$[?@.b]", "$[?@.c || match(@.a, 'x[')]"},
                                        sz_subdocuments_json_k, malformed_index) == sz_invalid_syntax_k);
    verify(malformed_index == 2);
}

/** Random paths over random documents through the dispatched verbs, against the serial ones. */
void test_subdocuments_all(test_context_t &context) {
    check_subdocuments_equivalence_(context, subdocuments_dispatched);
}

#pragma endregion Unit Cases

} // namespace ashvardanian::stringzilla::test
