/**
 *  @file test/substrings.cpp
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief Multi-pattern search tests: known answers, a brute-force oracle, and the dispatched verbs
 *      checked against it.
 *
 *  The oracle in `cross.hpp` is a naive scan of every needle at every offset, which is what an
 *  Aho-Corasick automaton has to agree with by construction. Under case folding it folds both sides
 *  with the library's own @c sz_utf8_uncased_fold and searches the folded haystack, then snaps the
 *  span it found back onto whole source codepoints, testing the automaton rather than the fold.
 */
#undef NDEBUG // ! Enable all assertions for testing

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <cmath>   // `std::fabs`
#include <cstdlib> // `std::malloc`, `std::free`

#include <algorithm> // `std::sort`
#include <string>    // Baseline
#include <vector>    // `std::vector`

#include "cross.hpp"   // `substrings_tier_t`, `check_substrings_unit_`, `check_substrings_equivalence_`
#include "harness.hpp" // `random_string`, `refusing_allocator_`, `test_context_t`, `verify`

namespace ashvardanian::stringzilla::test {

#pragma region Helpers

/** The dispatched engine builder over the CPU capabilities, in the shape of its capability kernels,
 *  whose mask sits before the allocator. */
static sz_status_t substrings_engine_init_dispatched_(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                      sz_substrings_case_sensitivity_t case_sensitivity,
                                                      sz_substrings_overlap_policy_t overlap_policy,
                                                      sz_size_t hot_states, sz_size_t matches_budget,
                                                      sz_size_t haystacks_budget, sz_allocator_t *allocator,
                                                      void *stream) {
    return sz_substrings_engine_init(engine, needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                     haystacks_budget, sz::default_capabilities(), allocator, stream);
}

/** The dispatched builder and the verbs that walk whatever capability it prepared for. */
static substrings_tier_t const substrings_dispatched {substrings_engine_init_dispatched_, sz_substrings_counts,
                                                      sz_substrings_find, sz_substrings_replace,
                                                      sz_substrings_bm25_scores};

/** Compiles @p needles, so a refusal can be asserted on without naming a sequence of its own. */
static sz_status_t build_over_(std::vector<std::string> const &needles, sz_substrings_case_sensitivity_t sensitivity,
                               sz_allocator_t *allocator, sz_substrings_engine_t *engine) {
    std::vector<sz_string_view_t> views;
    sz_sequence_t const sequence = sequence_over_(needles, views);
    return sz_substrings_engine_init(engine, &sequence, sensitivity, sz_substrings_overlapping_k,
                                     STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0, sz::default_capabilities(),
                                     allocator, nullptr);
}

/** An allocator granting its first @c grants requests and refusing the rest, tallying the
 *  bytes it holds. */
struct rationed_allocator_t {

    /** Requests still to be granted before every later one is refused. */
    std::size_t grants {};

    /** Bytes handed out and not yet returned. */
    std::size_t bytes_held {};

    /** The C-side view, whose handle points back at this object. */
    sz_allocator_t allocator {};

    explicit rationed_allocator_t(std::size_t granted) noexcept : grants(granted) {
        allocator.allocate = +[](sz_size_t length, void *handle, void *) -> void * {
            rationed_allocator_t &self = *static_cast<rationed_allocator_t *>(handle);
            if (!self.grants) return nullptr;
            void *const pointer = std::malloc(length ? length : 1);
            if (pointer) --self.grants, self.bytes_held += length;
            return pointer;
        };
        allocator.free = +[](void *pointer, sz_size_t length, void *handle, void *) {
            static_cast<rationed_allocator_t *>(handle)->bytes_held -= length;
            std::free(pointer);
        };
        allocator.handle = this;
    }
    rationed_allocator_t(rationed_allocator_t const &) = delete;
    rationed_allocator_t &operator=(rationed_allocator_t const &) = delete;
};

/** Refuses the build's allocations one later each round, so every error path proves it frees
 *  what it took. */
static void check_build_refusals_(std::vector<std::string> const &needles,
                                  sz_substrings_case_sensitivity_t sensitivity) {
    for (std::size_t granted = 0;; ++granted) {
        rationed_allocator_t rationed(granted);
        sz_substrings_engine_t engine {};
        sz_status_t const status = build_over_(needles, sensitivity, &rationed.allocator, &engine);
        if (status == sz_success_k) {
            sz_substrings_engine_free(&engine, nullptr);
            verify(rationed.bytes_held == 0);
            return;
        }
        verify(status == sz_bad_alloc_k);
        verify(engine.memory == nullptr && "A refused build leaves the engine untouched");
        verify(rationed.bytes_held == 0 && "A refused build returns everything it took");
    }
}

#pragma endregion Helpers

#pragma region Unit Cases

/** The textbook cases, where a wrong failure link or a missed output run shows up by name. */
void test_substrings_unit() { check_substrings_unit_(substrings_dispatched); }

/** What the verbs refuse, which is as much of the contract as what they accept. */
void test_substrings_safety(test_context_t &context) {
    handle_checked_heap_t heap;
    sz_substrings_engine_t engine;
    std::vector<sz_string_view_t> views;

    // An empty needle would match at every position, so the whole vocabulary is refused rather than shifting
    // every later needle's reported index.
    {
        std::vector<std::string> const empty_needle {"ab", "", "cd"};
        verify(build_over_(empty_needle, sz_substrings_cased_k, &heap.allocator, &engine) ==
               sz_unexpected_dimensions_k);
    }

    // An empty vocabulary has no automaton to build.
    {
        std::vector<std::string> const no_needles;
        verify(build_over_(no_needles, sz_substrings_cased_k, &heap.allocator, &engine) == sz_unexpected_dimensions_k);
    }

    // A folded vocabulary needs well-formed UTF-8, since the walk resets on a malformed haystack byte and a
    // needle carrying one could never match.
    {
        std::vector<std::string> const malformed {std::string("ab\xFF", 3)};
        verify(build_over_(malformed, sz_substrings_uncased_k, &heap.allocator, &engine) == sz_invalid_utf8_k);
    }

    std::vector<std::string> const needles {"ab", "cd"};
    std::vector<std::string> const haystacks {"abcd"};
    std::vector<std::string> const replacements {"x", "y"};
    std::vector<sz_string_view_t> haystack_views, replacement_views;
    sz_sequence_t const needle_sequence = sequence_over_(needles, views);
    sz_sequence_t const haystack_sequence = sequence_over_(haystacks, haystack_views);
    sz_sequence_t const replacement_sequence = sequence_over_(replacements, replacement_views);
    verify(sz_substrings_engine_init(&engine, &needle_sequence, sz_substrings_cased_k, sz_substrings_overlapping_k,
                                     STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0, sz::default_capabilities(),
                                     &heap.allocator, nullptr) == sz_success_k);

    // A capacity that cannot hold the matches is not an error: the report names the true total and the
    // shortfall beside it, which is the one contract a device verb can also keep.
    {
        std::vector<sz_size_t> offsets(haystacks.size() + 1, STRINGZILLA_SIZE_MAX);
        verify(sz_substrings_find(&engine, &haystack_sequence, nullptr, 0, offsets.data(), nullptr) == sz_success_k);
        verify(engine.report->matches_emitted == 2);
        verify(engine.report->matches_stored == 0 && engine.report->shortfall == 2);
        verify(offsets[haystacks.size()] == 2);
    }

    // A substitution over matches that share bytes is not a function, so the overlapping policy is refused.
    {
        std::vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        verify(sz_substrings_replace(&engine, &haystack_sequence, &replacement_sequence, nullptr, 0, offsets.data(),
                                     nullptr) == sz_status_unknown_k);
    }

    // An output stride of zero cannot address one entry per haystack, whatever the haystack count.
    {
        std::vector<sz_size_t> counts(haystacks.size(), STRINGZILLA_SIZE_MAX);
        sz_substrings_bm25_t const unnormalized {1.2f, 0.0f, 0.0f};
        sz_f32_t const weights[] {1.0f, 2.0f};
        sz_f32_t score = -1;
        verify(sz_substrings_counts(&engine, &haystack_sequence, counts.data(), 0, nullptr) ==
               sz_unexpected_dimensions_k);
        verify(sz_substrings_bm25_scores(&engine, &haystack_sequence, nullptr, &unnormalized, weights, &score, 0,
                                         nullptr) == sz_unexpected_dimensions_k);
        verify(counts[0] == STRINGZILLA_SIZE_MAX && score == -1);
    }

    // BM25 needs one weight per needle, and a mean to normalize by whenever it normalizes at all.
    {
        sz_f32_t const weights[] {1.0f, 2.0f};
        sz_f32_t score = -1;
        sz_substrings_bm25_t const meanless {1.2f, 0.75f, 0.0f};
        sz_substrings_bm25_t const unnormalized {1.2f, 0.0f, 0.0f};
        verify(sz_substrings_bm25_scores(&engine, &haystack_sequence, nullptr, &unnormalized, nullptr, &score, 1,
                                         nullptr) == sz_unexpected_dimensions_k);
        verify(sz_substrings_bm25_scores(&engine, &haystack_sequence, nullptr, &meanless, weights, &score, 1,
                                         nullptr) == sz_unexpected_dimensions_k);
        verify(score == -1);
        // Without normalization the mean is never read: "ab" and "cd" once each, `tf·(k1+1)/(tf+k1)` = 1.
        verify(sz_substrings_bm25_scores(&engine, &haystack_sequence, nullptr, &unnormalized, weights, &score, 1,
                                         nullptr) == sz_success_k);
        verify(std::fabs(score - 3.0f) <= 1e-6f);
    }

    sz_substrings_engine_free(&engine, nullptr);

    // One replacement per needle, so a shorter list names no substitution for the needles it omits.
    {
        std::vector<std::string> const one_replacement {"x"};
        std::vector<sz_string_view_t> one_views;
        sz_sequence_t const one_sequence = sequence_over_(one_replacement, one_views);
        std::vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        sz_substrings_engine_t covering;
        verify(sz_substrings_engine_init(&covering, &needle_sequence, sz_substrings_cased_k,
                                         sz_substrings_leftmost_first_k, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0,
                                         sz::default_capabilities(), &heap.allocator, nullptr) == sz_success_k);
        verify(sz_substrings_replace(&covering, &haystack_sequence, &one_sequence, nullptr, 0, offsets.data(),
                                     nullptr) == sz_unexpected_dimensions_k);
        sz_substrings_engine_free(&covering, nullptr);
    }
    verify(heap.live_allocations == 0);

    // A refused allocation is reported as `sz_bad_alloc_k`, and construction is the only verb that allocates.
    {
        sz_allocator_t refusing = refusing_allocator_();
        sz_substrings_engine_t refused {};
        verify(build_over_(needles, sz_substrings_cased_k, &refusing, &refused) == sz_bad_alloc_k);
        verify(refused.memory == nullptr);
    }

    // Every allocation site of the build refused in turn, past the trie's and the arena's first growth.
    {
        std::vector<std::string> wide;
        for (std::size_t index = 0; index != 600; ++index)
            wide.push_back(random_string(context.generator, 3 + index % 6, "abcdefghijklmnopqrstuvwxyz"));
        std::sort(wide.begin(), wide.end());
        wide.erase(std::unique(wide.begin(), wide.end()), wide.end());
        check_build_refusals_(wide, sz_substrings_cased_k);
        check_build_refusals_(wide, sz_substrings_uncased_k);
    }
}

/** Random vocabularies over random corpora through the dispatched verbs, which is what reaches the
 *  packing search's fallbacks. */
void test_substrings_all(test_context_t &context) { check_substrings_equivalence_(context, substrings_dispatched); }

#pragma endregion Unit Cases

} // namespace ashvardanian::stringzilla::test
