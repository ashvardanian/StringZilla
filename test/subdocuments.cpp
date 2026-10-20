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
#include "harness.hpp" // `refusing_allocator_`, `test_context_t`, `verify`

namespace ashvardanian::stringzilla::test {

#pragma region Helpers

/** The dispatched engine builder over the CPU capabilities, in the shape of its capability kernels,
 *  whose mask sits before the allocator. */
static sz_status_t subdocuments_engine_init_dispatched_(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *paths, sz_subdocuments_format_t format,
                                                        sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    return sz_subdocuments_engine_init(engine, paths, format, malformed_index, sz::default_capabilities(), allocator,
                                       stream);
}

/** The dispatched builder and the verbs that walk whatever capability it compiled for. */
static subdocuments_tier_t const subdocuments_dispatched {subdocuments_engine_init_dispatched_, sz_subdocuments_counts,
                                                          sz_subdocuments_find};

#pragma endregion Helpers

#pragma region Unit Cases

/** RFC 9535's bookstore and examples, XPath over an XML bookstore, refused paths and capacities. */
void test_subdocuments_unit() { check_subdocuments_unit_(subdocuments_dispatched); }

/** What the verbs refuse, which is as much of the contract as what they accept. */
void test_subdocuments_safety(test_context_t &) {
    check_subdocuments_safety_(subdocuments_dispatched);

    // A pattern the regex family refuses refuses the path spelling it, whichever filter holds it.
    sz_size_t malformed_index = 0;
    verify(subdocuments_compile_status_(subdocuments_dispatched, {"$.a", "$[?@.b]", "$[?@.c || match(@.a, 'x[')]"},
                                        sz_subdocuments_json_k, malformed_index) == sz_invalid_syntax_k);
    verify(malformed_index == 2);

    // Reject an oversized translated pattern before allocating or reading its text.
    sz_subdocuments_instruction_t instruction {};
    instruction.operation = sz_subdocuments_operation_match_k;
    instruction.string_length = 0x40000000u;
    sz_regex_engine_t regex {};
    sz_allocator_t refusing = refusing_allocator_();
    verify(sz_subdocuments_compile_patterns_(&regex, &instruction, 1, nullptr, 1, sz::default_capabilities(), &refusing,
                                             nullptr, &malformed_index) == sz_overflow_risk_k);
}

/** Random paths over random documents through the dispatched verbs, against the serial ones. */
void test_subdocuments_all(test_context_t &context) {
    check_subdocuments_equivalence_(context, subdocuments_dispatched);
}

#pragma endregion Unit Cases

} // namespace ashvardanian::stringzilla::test
