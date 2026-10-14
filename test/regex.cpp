/**
 *  @file test/regex.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Regular expression tests: known answers and random vocabularies through the dispatched
 *      verbs, and what those verbs refuse.
 *
 *  The known answers in `cross.hpp` were settled against `regex-automata`, whose leftmost-first
 *  semantics the engine reproduces; random vocabularies hold the dispatched tier to the serial one.
 */
#undef NDEBUG // ! Enable all assertions for testing

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <cstdlib> // `std::malloc`, `std::free`

#include <string> // `std::string`
#include <vector> // `std::vector`

#include "stringzilla/stringzilla.h"   // Primary C API
#include "stringzilla/stringzilla.hpp" // `sz::default_capabilities`

#include "cross.hpp"   // `regex_tier_t`, `check_regex_unit_`, `check_regex_equivalence_`
#include "harness.hpp" // `handle_checked_heap_t`, `refusing_allocator_`, `test_context_t`, `verify`

namespace ashvardanian::stringzilla::test {

#pragma region Helpers

/** The dispatched engine builder over the CPU capabilities, shaped like its capability kernels. */
static sz_status_t regex_engine_init_dispatched_(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                 sz_case_sensitivity_t case_sensitivity, sz_size_t *malformed_index,
                                                 sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_regex_engine_init(engine, patterns, case_sensitivity, malformed_index, sz::default_capabilities(),
                                allocator, stream);
}

/** The dispatched builder and the verbs that walk whatever capability it prepared for. */
static regex_tier_t const regex_dispatched {regex_engine_init_dispatched_, sz_regex_counts, sz_regex_find};

/** The dispatched rewriting verbs. */
static regex_rewriting_t const regex_dispatched_rewriting {sz_regex_replace, sz_regex_extract};

/** An allocator refusing every request past its first @c grants, counting the blocks it holds. */
struct rationed_heap_t {
    std::size_t grants {};
    std::size_t live_allocations {};
    sz_allocator_t allocator {};

    explicit rationed_heap_t(std::size_t granted) noexcept : grants(granted) {
        allocator.allocate = +[](sz_size_t length, void *handle, sz_stream_t) -> void * {
            rationed_heap_t &self = *static_cast<rationed_heap_t *>(handle);
            if (!self.grants) return nullptr;
            void *const pointer = std::malloc(length ? length : 1);
            if (pointer) --self.grants, ++self.live_allocations;
            return pointer;
        };
        allocator.free = +[](void *pointer, sz_size_t, void *handle, sz_stream_t) {
            --static_cast<rationed_heap_t *>(handle)->live_allocations;
            std::free(pointer);
        };
        allocator.handle = this;
    }
    rationed_heap_t(rationed_heap_t const &) = delete;
    rationed_heap_t &operator=(rationed_heap_t const &) = delete;
};

#pragma endregion Helpers

#pragma region Unit Cases

void test_regex_unit() {
    check_regex_unit_(regex_dispatched, regex_dispatched_rewriting);

    // The password rules of the known answers with a nested fourth lookaround become columns of
    // both DFAs, and with a fifth leave both, finding the same match either way.
    using cases_t = std::vector<substrings_case_t>;
    std::vector<std::string> const rules {"^(?=.*\\d)(?=.*[a-z])(?=.*[A-Z](?!\\s)).{8,}\\z",
                                          "^(?=.*\\d)(?=.*[a-z])(?=.*[A-Z](?!\\s))(?!.*\\s).{8,}\\z"};
    for (std::size_t index = 0; index != rules.size(); ++index) {
        handle_checked_heap_t heap;
        std::vector<std::string> const patterns {rules[index]};
        std::vector<sz_string_view_t> views;
        sz_strings_view_t const strings = strings_over_(patterns, views);
        sz_regex_engine_t engine;
        verify(regex_dispatched.init(&engine, &strings, sz_cased_k, nullptr, &heap.allocator, nullptr) == sz_success_k);
        verify(engine.lookarounds_count == sz_regex_columned_lookarounds_k + index);
        verify((engine.forward_rows != nullptr) == (index == 0) && (engine.reverse_rows != nullptr) == (index == 0));
        verify(regex_matches_(regex_dispatched, &engine, {"Passw0rdX", "password1", "Sh0rt", "NODIGITS"}) ==
               cases_t({{0, 0, 0, 9}}));
        sz_regex_engine_free(&engine, nullptr);
        verify(heap.live_allocations == 0);
    }
}

/** What the verbs refuse, which is as much of the contract as what they accept. */
void test_regex_safety(test_context_t &) {
    std::vector<std::string> const patterns {"[a-z]+", "\\d+"};
    std::vector<std::string> const haystacks {"abc 123"};
    std::vector<sz_string_view_t> pattern_views, haystack_views;
    sz_strings_view_t const pattern_strings = strings_over_(patterns, pattern_views);
    sz_strings_view_t const haystack_strings = strings_over_(haystacks, haystack_views);

    // A malformed pattern is refused by index, leaving the engine untouched.
    {
        handle_checked_heap_t heap;
        std::vector<std::string> const malformed {"a", std::string("b\xC3", 2)};
        std::vector<sz_string_view_t> views;
        sz_strings_view_t const strings = strings_over_(malformed, views);
        sz_regex_engine_t engine {};
        sz_size_t index = STRINGZILLA_SIZE_MAX;
        verify(sz_regex_engine_init(&engine, &strings, sz_cased_k, &index, sz::default_capabilities(), &heap.allocator,
                                    nullptr) == sz_invalid_syntax_k);
        verify(index == 1 && engine.memory == nullptr);
        verify(heap.live_allocations == 0);
    }

    // Malformed haystacks are searched as they are: no class matches a malformed byte, and what is
    // well formed around it still matches.
    {
        handle_checked_heap_t heap;
        sz_regex_engine_t engine;
        verify(sz_regex_engine_init(&engine, &pattern_strings, sz_cased_k, nullptr, sz::default_capabilities(),
                                    &heap.allocator, nullptr) == sz_success_k);
        std::vector<std::string> const malformed {std::string("ab\xFF\xC3" "1\xE2\x82", 7), // `\xC3` ends before `1`
                                                  std::string("\x80\x80", 2)};
        using cases_t = std::vector<substrings_case_t>;
        verify(regex_matches_(regex_dispatched, &engine, malformed) == cases_t({{0, 0, 0, 2}, {0, 1, 4, 1}}));

        // A stride of zero addresses no entry.
        std::vector<sz_size_t> counts(1, STRINGZILLA_SIZE_MAX);
        verify(sz_regex_counts(&engine, &haystack_strings, counts.data(), 0, nullptr) == sz_unexpected_dimensions_k);
        verify(counts[0] == STRINGZILLA_SIZE_MAX);
        sz_regex_engine_free(&engine, nullptr);

        // An emptied engine compiles nothing to run.
        std::vector<sz_size_t> offsets(2, 0);
        verify(sz_regex_find_serial(&engine, &haystack_strings, nullptr, 0, offsets.data(), nullptr) ==
               sz_missing_kernel_k);
        verify(heap.live_allocations == 0);
    }

    // Rewriting needs one replacement per pattern, and extraction a pattern the vocabulary holds.
    {
        handle_checked_heap_t heap;
        sz_regex_engine_t engine;
        verify(sz_regex_engine_init(&engine, &pattern_strings, sz_cased_k, nullptr, sz::default_capabilities(),
                                    &heap.allocator, nullptr) == sz_success_k);
        std::vector<std::string> const one_replacement {"x"};
        std::vector<sz_string_view_t> replacement_views;
        sz_strings_view_t const replacement_strings = strings_over_(one_replacement, replacement_views);
        std::vector<sz_size_t> offsets(2, 0);
        verify(sz_regex_replace(&engine, &haystack_strings, &replacement_strings, nullptr, 0, offsets.data(),
                                nullptr) == sz_unexpected_dimensions_k);
        verify(sz_regex_extract(&engine, &haystack_strings, 2, nullptr, 0, offsets.data(), nullptr) ==
               sz_unexpected_dimensions_k);
        sz_regex_engine_free(&engine, nullptr);
        verify(heap.live_allocations == 0);
    }

    // A refused allocation is `sz_bad_alloc_k`, from the build and from a search's scratch alike.
    {
        sz_allocator_t refusing = refusing_allocator_();
        sz_regex_engine_t engine {};
        verify(sz_regex_engine_init(&engine, &pattern_strings, sz_cased_k, nullptr, sz::default_capabilities(),
                                    &refusing, nullptr) == sz_bad_alloc_k);
        verify(engine.memory == nullptr);
    }

    // Every allocation of the build refused in turn returns all the others, and a build granted
    // exactly what it takes leaves the search to refuse its scratch.
    for (std::size_t granted = 0;; ++granted) {
        rationed_heap_t rationed(granted);
        sz_regex_engine_t engine {};
        sz_status_t const status = sz_regex_engine_init(&engine, &pattern_strings, sz_cased_k, nullptr,
                                                        sz::default_capabilities(), &rationed.allocator, nullptr);
        if (status == sz_success_k) {
            std::vector<sz_size_t> offsets(2, 0);
            verify(sz_regex_find(&engine, &haystack_strings, nullptr, 0, offsets.data(), nullptr) == sz_bad_alloc_k);
            sz_regex_engine_free(&engine, nullptr);
            verify(rationed.live_allocations == 0);
            break;
        }
        verify(status == sz_bad_alloc_k);
        verify(engine.memory == nullptr && rationed.live_allocations == 0);
    }

    // The same sweep over a vocabulary whose start pass runs bit-parallel and whose end walks run
    // the PikeVM, over an uncased one with named groups and folds to several codepoints, and over
    // one with nested lookarounds.
    std::vector<std::string> const proximity {"\\b(?:error|warning)\\b.{0,8}\\b(?:disk|memory)\\b"};
    std::vector<std::string> const folded {"(?P<word>\\w+)\xC3\x9F", "[\xC3\x9F\xE1\xBA\x9E]x"};
    std::vector<std::string> const looking {"(?<=\\$)\\d+(?!\\.)", "\\b(?=\\w*(?<!x)y)\\w+"};
    for (std::vector<std::string> const *vocabulary : {&proximity, &folded, &looking})
        for (std::size_t granted = 0;; ++granted) {
            rationed_heap_t rationed(granted);
            sz_regex_engine_t engine {};
            sz_status_t const status = regex_compile_(*vocabulary, vocabulary == &folded ? sz_uncased_k : sz_cased_k, 0,
                                                      0, &rationed.allocator, &engine);
            if (status == sz_success_k) {
                sz_regex_engine_free(&engine, nullptr);
                verify(rationed.live_allocations == 0);
                break;
            }
            verify(status == sz_bad_alloc_k);
            verify(engine.memory == nullptr && rationed.live_allocations == 0);
        }
}

/** Random vocabularies over random haystacks through the dispatched verbs. */
void test_regex_all(test_context_t &context) {
    check_regex_equivalence_(context, regex_dispatched, regex_dispatched_rewriting);
}

#pragma endregion Unit Cases

} // namespace ashvardanian::stringzilla::test
