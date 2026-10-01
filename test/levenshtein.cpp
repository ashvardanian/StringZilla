/**
 *  @file test/levenshtein.cpp
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Levenshtein edit-distance tests: known answers, a textbook DP oracle, and the dispatched
 *      entry points against it.
 */
#undef NDEBUG // ! Enable all assertions for testing

/** The Visual C++ run-time library detects incorrect iterator use, and asserts and displays a
 *  dialog box at run time on Windows. */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Make sure to include the StringZilla headers before anything else, to intercept missing
 *  `#include` directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include "cross.hpp"   // `levenshtein_backend_t`, `check_levenshtein_unit_`, `check_levenshtein_equivalence_`
#include "harness.hpp" // `test_context_t`

namespace ashvardanian::stringzilla::test {

#pragma region Helpers

/** The dispatched engine builder over the CPU capabilities, in the shape of its capability kernels,
 *  whose mask sits before the ordinal rather than the stream. */
static sz_status_t levenshtein_engine_init_dispatched_(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                                       sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                       sz_memory_allocator_t *allocator, void *stream) {
    return sz_levenshtein_engine_init(engine, queries, symbol, sz::default_capabilities(), ordinal, allocator, stream);
}

/** The dispatched builder and the verb that scores whatever capability it prepared for. */
static levenshtein_backend_t const levenshtein_dispatched {"dispatched", levenshtein_engine_init_dispatched_,
                                                           sz_levenshtein_distances};

#pragma endregion Helpers

/** Known answers: the classic pairs, empties on either side, identity, the 64-symbol word boundary,
 *  and multi-byte runes, through the dispatched entry points. */
void test_levenshtein_unit() { check_levenshtein_unit_(levenshtein_dispatched); }

/** Degenerate inputs for the edit-distance family, asserting survival and the stated refusals.
 *  Answers are not the subject here: empties are accepted, a refused allocation is reported, and no
 *  failure writes an output. */
void test_levenshtein_safety(test_context_t &) { check_levenshtein_safety_(levenshtein_dispatched); }

/** Drives the oracle sweeps and the serial-versus-dispatched differential. */
void test_levenshtein_all(test_context_t &context) { check_levenshtein_equivalence_(context, levenshtein_dispatched); }

} // namespace ashvardanian::stringzilla::test
