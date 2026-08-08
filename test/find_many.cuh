/**
 *  @brief Extensive @b stress-testing suite for the StringZillas multi-pattern search engine (Aho-Corasick).
 *  @see Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file scripts/test_find_many.cuh
 *  @author Ash Vardanian
 *  @date June 16, 2026
 */
#include "stringzilla/utf8_uncased.h" // `sz_utf8_uncased_search`, the independent single-needle oracle

#include "stringzillas/find_many.hpp"

#if SZ_USE_CUDA
#include "stringzillas/find_many.cuh"
#endif

#if !SZ_IS_CPP17_
#error "This test requires C++17 or later."
#endif

#include <cstdio>  // `std::printf`, `std::fprintf`
#include <cstring> // `std::memcmp`, `std::memcpy`

#include <algorithm> // `std::sort`, `std::unique`
#include <limits>    // `std::numeric_limits`
#include <random>    // `std::mt19937`, `std::uniform_int_distribution`
#include <set>       // `std::set`
#include <string>    // `std::string`
#include <utility>   // `std::move`
#include <vector>    // `std::vector`

#include <stringzilla/stringzilla.h> // Primary C API

#include "stringzilla.hpp" // `verify`, `run_test`, `global_random_generator`, `fuzzy_config_t`
#include "utf8.hpp"        // `append_malformed_class_`, `utf8_random_segmentation_corpus_`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

using ashvardanian::stringzillas::dummy_executor_t;
using ashvardanian::stringzillas::find_many_cased_k;
using ashvardanian::stringzillas::forkunion_executor_t;
using ashvardanian::stringzillas::find_many_match_t;
using ashvardanian::stringzillas::find_many_case_sensitivity_t;
using ashvardanian::stringzillas::find_many_u16_dictionary_t;
using ashvardanian::stringzillas::find_many_u32_dictionary_t;
using ashvardanian::stringzillas::find_many_u32_parallel_t;
using ashvardanian::stringzillas::find_many_u32_serial_t;
using ashvardanian::stringzillas::find_many_uncased_k;

#if SZ_USE_CUDA
using ashvardanian::stringzillas::cuda_executor_t;
using ashvardanian::stringzillas::find_many_u32_cuda_t;
using ashvardanian::stringzillas::gpu_specs_fetch;
using ashvardanian::stringzillas::gpu_specs_t;
#endif

#pragma region Helpers

/** @brief Field-by-field ordering for `finalize`; the public match deliberately carries no comparators. */
inline bool find_many_match_less_(find_many_match_t const &left, find_many_match_t const &right) noexcept {
    if (left.haystack_index != right.haystack_index) return left.haystack_index < right.haystack_index;
    if (left.needle_index != right.needle_index) return left.needle_index < right.needle_index;
    if (left.byte_offset != right.byte_offset) return left.byte_offset < right.byte_offset;
    return left.byte_length < right.byte_length;
}
inline bool find_many_match_equal_(find_many_match_t const &left, find_many_match_t const &right) noexcept {
    return left.haystack_index == right.haystack_index && left.needle_index == right.needle_index &&
           left.byte_offset == right.byte_offset && left.byte_length == right.byte_length;
}

/**
 *  @brief Sorted, duplicate-checked set of every match one count-then-find pass reported, owning the pass's
 *         intermediate buffers so a sweep reuses one allocation instead of sizing a fresh pair per cell.
 *
 *  The intermediate `counts` and `matches` buffers are unified, since the CUDA engine writes them; the
 *  sorted set itself is host-only, and `clear` keeps its backing allocation.
 */
struct find_many_match_set_t {
    std::vector<find_many_match_t> matches;
    unified_vector<std::size_t> collected_counts;
    unified_vector<find_many_match_t> collected_matches;

    find_many_match_set_t() = default;

    /** @brief Builds an already-ordered expectation from a hand-written table, which can therefore spell its
     *         matches in whatever order reads best. */
    find_many_match_set_t(std::initializer_list<find_many_match_t> initial) {
        for (find_many_match_t const &match : initial) matches.push_back(match);
        finalize();
    }

    void clear() noexcept { matches.clear(); }
    void append(find_many_match_t match) { matches.push_back(match); }
    std::size_t size() const noexcept { return matches.size(); }
    bool empty() const noexcept { return matches.empty(); }
    auto begin() const noexcept { return matches.begin(); }
    auto end() const noexcept { return matches.end(); }

    /** @brief Orders the collected matches and fails on a repeat - the four fields identify a match uniquely,
     *         so a duplicate is always a bug. */
    void finalize() {
        std::sort(matches.begin(), matches.end(), find_many_match_less_);
        bool const has_duplicate = std::adjacent_find(matches.begin(), matches.end(), find_many_match_equal_) !=
                                   matches.end();
        verify(!has_duplicate && "Duplicate (haystack, needle, offset, length) reported twice");
    }

    /** @note Both sides must have been finalized since their last mutation. */
    bool operator==(find_many_match_set_t const &other) const noexcept {
        return matches.size() == other.matches.size() &&
               std::equal(matches.begin(), matches.end(), other.matches.begin(), find_many_match_equal_);
    }
    bool operator!=(find_many_match_set_t const &other) const noexcept { return !(*this == other); }
};

/**
 *  @brief Runs a full count-then-find pass through @p engine and reduces every match to its identity in
 *         @p out, whose own buffers hold the intermediates. Every differential check in this file is driven
 *         through this one public-API shape.
 *  @note `trailing_args_` forwards an executor (and optionally specs) to backends that want a specific one;
 *        an empty pack leaves each engine on its own defaults.
 */
template <typename engine_type_, typename haystacks_type_, typename... trailing_args_>
void collect_matches_into_(engine_type_ &engine, haystacks_type_ const &haystacks, find_many_match_set_t &out,
                           trailing_args_ &&...trailing_args) {
    out.collected_counts.assign(haystacks.size(), 0);
    std::size_t matches_total = 0;
    verify(engine.try_count(haystacks, span<std::size_t>(out.collected_counts.data(), out.collected_counts.size()),
                            matches_total, trailing_args...) == status_t::success_k);

    // Sized to exactly what counting promised, so a `try_find` overrun trips instead of writing into slack.
    out.collected_matches.assign(matches_total, find_many_match_t {});
    std::size_t matches_found = 0;
    verify(engine.try_find(haystacks,
                           span<find_many_match_t>(out.collected_matches.data(), out.collected_matches.size()),
                           matches_found, trailing_args...) == status_t::success_k);
    verify(matches_found == matches_total && "try_count and try_find disagree on the match count");

    out.clear();
    for (find_many_match_t const &match : out.collected_matches) out.append(match);
    out.finalize();
}

/** @brief A vocabulary of short random needles/haystack fragments over a small alphabet, so the resulting
 *         automaton is small enough to reason about by hand yet large enough to grow a real cold tier. */
inline std::vector<std::string> random_short_strings_(std::size_t count, int minimum_length, int maximum_length) {
    std::vector<std::string> result;
    result.reserve(count);
    auto &generator = global_random_generator();
    std::uniform_int_distribution<int> length_distribution(minimum_length, maximum_length);
    std::uniform_int_distribution<int> letter_distribution('a', 'j');
    for (std::size_t index = 0; index < count; ++index) {
        std::string value((std::size_t)length_distribution(generator), '\0');
        for (char &letter : value) letter = (char)letter_distribution(generator);
        result.push_back(std::move(value));
    }
    return result;
}

/**
 *  @brief Independent case-folded substring oracle: folds haystack and needle codepoint by codepoint via
 *         `sz_unicode_fold_codepoint_`, then slides the folded needle over the folded haystack, reporting
 *         every start position whose folded run matches - overlaps included.
 *
 *  Shares no machinery with the Aho-Corasick engine under test beyond the one folding table every backend
 *  reads. @sa `reference_uncased_find_` in `test/uncased.cpp`, which reports only the first match.
 */
inline std::vector<std::pair<std::size_t, std::size_t>> independent_uncased_matches_(span<char const> haystack,
                                                                                     span<char const> needle) {
    std::vector<sz_rune_t> needle_folded;
    for (char const *cursor = needle.begin(), *end = needle.end(); cursor != end;) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode(cursor, end, &rune);
        verify(consumed != sz_rune_invalid_k && "Independent oracle needles must be well-formed UTF-8");
        sz_rune_t folded[3];
        std::size_t const folded_count = sz_unicode_fold_codepoint_(rune, folded);
        for (std::size_t index = 0; index < folded_count; ++index) needle_folded.push_back(folded[index]);
        cursor += consumed;
    }

    std::vector<sz_rune_t> haystack_folded;
    std::vector<std::size_t> source_begin, source_end;
    // ? Only the first folded rune a codepoint contributes marks a legal match start; a codepoint whose own
    // full fold spans multiple runes - the sharp S folding to "s","s" - must not let a match begin on its
    // second rune, which has no byte of its own to start at.
    std::vector<bool> is_codepoint_start;
    for (char const *cursor = haystack.begin(), *end = haystack.end(); cursor != end;) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode(cursor, end, &rune);
        verify(consumed != sz_rune_invalid_k && "Independent oracle haystacks must be well-formed UTF-8");
        sz_rune_t folded[3];
        std::size_t const folded_count = sz_unicode_fold_codepoint_(rune, folded);
        std::size_t const codepoint_begin = (std::size_t)(cursor - haystack.begin());
        std::size_t const codepoint_end = codepoint_begin + (std::size_t)consumed;
        for (std::size_t index = 0; index < folded_count; ++index) {
            haystack_folded.push_back(folded[index]);
            source_begin.push_back(codepoint_begin);
            source_end.push_back(codepoint_end);
            is_codepoint_start.push_back(index == 0);
        }
        cursor += consumed;
    }

    std::vector<std::pair<std::size_t, std::size_t>> matches;
    std::size_t const needle_length = needle_folded.size();
    if (needle_length == 0) return matches;
    for (std::size_t start = 0; start + needle_length <= haystack_folded.size(); ++start) {
        if (!is_codepoint_start[start]) continue;
        bool equal = true;
        for (std::size_t index = 0; index < needle_length; ++index)
            if (haystack_folded[start + index] != needle_folded[index]) {
                equal = false;
                break;
            }
        if (!equal) continue;
        matches.emplace_back(source_begin[start], source_end[start + needle_length - 1] - source_begin[start]);
    }
    return matches;
}

/**
 *  @brief Ground-truth match keys for a whole vocabulary against a whole batch, by brute force per needle.
 *
 *  Shares nothing with the automaton under test: cased matching is a plain byte slide, and uncased defers to
 *  `independent_uncased_matches_`. A constant both the serial and the accelerated walk got wrong is still
 *  caught here, which comparing two backends against each other cannot do.
 */
inline void collect_independent_matches_(find_many_case_sensitivity_t sensitivity, arrow_strings_view_t needles,
                                         arrow_strings_view_t haystacks, find_many_match_set_t &out) {
    out.clear();
    for (std::size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
        span<char const> const haystack = haystacks[haystack_index];
        for (std::size_t needle_index = 0; needle_index < needles.size(); ++needle_index) {
            span<char const> const needle = needles[needle_index];
            if (needle.size() == 0 || needle.size() > haystack.size()) continue;

            if (sensitivity == find_many_uncased_k) {
                for (auto const &match : independent_uncased_matches_(haystack, needle))
                    out.append({haystack_index, needle_index, match.first, match.second});
                continue;
            }
            for (std::size_t offset = 0; offset + needle.size() <= haystack.size(); ++offset)
                if (std::memcmp(haystack.data() + offset, needle.data(), needle.size()) == 0)
                    out.append({haystack_index, needle_index, offset, needle.size()});
        }
    }
    out.finalize();
}

/** @brief Selects one closed-set adversarial needle vocabulary. */
enum class find_many_needle_generator_t {
    self_overlapping_k,     ///< "a", "aa", "aaa", … - every needle a suffix of the next, quadratic outputs.
    mutual_overlap_k,       ///< One needle's suffix is another's prefix - the textbook failure-link case.
    shared_prefix_fan_k,    ///< One long common prefix, diverging only at the final byte.
    sparse_wide_alphabet_k, ///< Only bytes 0x01 and 0xFF - worst case for the double-array packer.
    fold_expanding_k,       ///< Runs of "k", "ss", "ffl" - a 3:1 preimage byte expansion under folding.
    random_short_k,         ///< A fixed core plus deduplicated random needles - the control.
    count_k,                ///< One past the last generator, so the sweep can never drift from the list.
};

/** @brief Selects one closed-set adversarial haystack construction, built against the generated needles. */
enum class find_many_haystack_generator_t {
    planted_exact_k,        ///< Each needle embedded once inside disjoint-alphabet noise.
    planted_concatenated_k, ///< Needles back to back with no separator - forces overlapping matches.
    near_miss_k,            ///< Every needle with its final byte perturbed - a negative control.
    fold_preimage_k,        ///< Needles re-encoded through case variants that only folding can match.
    boundary_stress_k,      ///< Flush with the first byte, flush with the last, and self-repeats.
    count_k,                ///< One past the last generator, so the sweep can never drift from the list.
};

/** @brief Whether a generator's needles are well-formed UTF-8, and so can be case-folded at all. Only the
 *         double-array pressure vocabulary reaches for `0xFF`, which no UTF-8 sequence ever contains. */
constexpr bool find_many_needles_are_utf8_(find_many_needle_generator_t kind) noexcept {
    return kind != find_many_needle_generator_t::sparse_wide_alphabet_k;
}

/** @brief Whether a generator's automaton grows with the square of its needle count, so the driver can pick
 *         `scale_iterations_quadratic` over `scale_iterations` and keep the suite balanced. */
constexpr bool find_many_needles_grow_quadratically_(find_many_needle_generator_t kind) noexcept {
    return kind == find_many_needle_generator_t::self_overlapping_k;
}

constexpr char const *find_many_needle_generator_name(find_many_needle_generator_t kind) noexcept {
    switch (kind) {
    case find_many_needle_generator_t::self_overlapping_k: return "self_overlapping";
    case find_many_needle_generator_t::mutual_overlap_k: return "mutual_overlap";
    case find_many_needle_generator_t::shared_prefix_fan_k: return "shared_prefix_fan";
    case find_many_needle_generator_t::sparse_wide_alphabet_k: return "sparse_wide_alphabet";
    case find_many_needle_generator_t::fold_expanding_k: return "fold_expanding";
    case find_many_needle_generator_t::random_short_k: return "random_short";
    case find_many_needle_generator_t::count_k: break; // ? Never a real generator, only the sweep's bound
    }
    return "unknown";
}

constexpr char const *find_many_haystack_generator_name(find_many_haystack_generator_t kind) noexcept {
    switch (kind) {
    case find_many_haystack_generator_t::planted_exact_k: return "planted_exact";
    case find_many_haystack_generator_t::planted_concatenated_k: return "planted_concatenated";
    case find_many_haystack_generator_t::near_miss_k: return "near_miss";
    case find_many_haystack_generator_t::fold_preimage_k: return "fold_preimage";
    case find_many_haystack_generator_t::boundary_stress_k: return "boundary_stress";
    case find_many_haystack_generator_t::count_k: break; // ? Never a real generator, only the sweep's bound
    }
    return "unknown";
}

/**
 *  @brief Fills @p needles with one closed-set adversarial vocabulary, @p count deep where the kind scales.
 *
 *  Every case either appends straight into the tape or, for the one kind needing a whole-collection sort and
 *  deduplication, fills a local pool first.
 */
inline void generate_find_many_needles_(find_many_needle_generator_t kind, std::size_t count,
                                        arrow_strings_tape_t &needles) {
    needles.reset();
    std::string scratch;
    switch (kind) {

    case find_many_needle_generator_t::self_overlapping_k:
        // Every needle is a suffix of the next, so the deepest state carries `count` merged outputs and the
        // output pool grows with the square of the depth - the worst case for anything sizing that pool.
        for (std::size_t depth = 1; depth <= count; ++depth) {
            scratch.push_back('a');
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;

    case find_many_needle_generator_t::mutual_overlap_k:
        // One needle's suffix is another's prefix, so failure links traverse rather than collapse to the root.
        for (std::size_t index = 0; index < count; ++index) {
            char const first = (char)('a' + index % 6), second = (char)('a' + (index + 1) % 6);
            scratch.assign({first, second, first});
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
            scratch.assign({second, first});
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;

    case find_many_needle_generator_t::shared_prefix_fan_k:
        // A single deep trunk with `count` leaves, so almost every state has out-degree one and the
        // frequency ordering has nothing to separate.
        for (std::size_t index = 0; index < count; ++index) {
            scratch.assign(12, 'q');
            scratch.push_back((char)('a' + index % 26));
            scratch.push_back((char)('a' + (index / 26) % 26));
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;

    case find_many_needle_generator_t::sparse_wide_alphabet_k:
        // Two bytes at opposite ends of the alphabet, so every double-array base must reserve a 256-wide
        // window to hold two edges - maximum collision pressure for the packer.
        for (std::size_t pattern = 0; pattern < count; ++pattern) {
            scratch.clear();
            for (std::size_t depth = 0; depth < 8; ++depth) scratch.push_back((pattern >> depth) & 1 ? '\xFF' : '\x01');
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;

    case find_many_needle_generator_t::fold_expanding_k: {
        // Needles whose fold preimages span more bytes than the needle itself - "k" also matches the 3-byte
        // Kelvin sign, so `max_match_bytes` cannot be read off needle lengths.
        char const *const seeds[] = {"k", "ss", "ffl", "st", "a"};
        for (std::size_t index = 0; index < count; ++index) {
            char const *const seed = seeds[index % 5];
            scratch.clear();
            for (std::size_t repeat = 0; repeat <= index % 4; ++repeat) scratch.append(seed);
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;
    }

    case find_many_needle_generator_t::random_short_k: {
        // The one kind needing a whole-collection sort and deduplication before insertion, which a tape of
        // variable-length ranges cannot do in place.
        std::vector<std::string> pool {"he", "she", "his", "hers"};
        for (std::string &value : random_short_strings_(count, 3, 6)) pool.push_back(std::move(value));
        std::sort(pool.begin(), pool.end());
        pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
        verify(needles.try_assign(pool.data(), pool.data() + pool.size()) == status_t::success_k);
        return;
    }

    case find_many_needle_generator_t::count_k: break; // ? Never a real generator, only the sweep's bound
    }
}

/** @brief Fills @p haystacks with @p haystack_count adversarial haystacks built against @p needles. */
inline void generate_find_many_haystacks_(find_many_haystack_generator_t kind, arrow_strings_view_t needles,
                                          std::size_t haystack_count, arrow_strings_tape_t &haystacks) {
    haystacks.reset();
    if (needles.size() == 0) return;

    auto &generator = global_random_generator();
    std::uniform_int_distribution<int> noise_length(4, 24);
    std::uniform_int_distribution<int> noise_byte('m', 'z'); // ? Disjoint from every needle alphabet above.
    std::string scratch;

    for (std::size_t haystack_index = 0; haystack_index < haystack_count; ++haystack_index) {
        scratch.clear();
        std::size_t const first_needle = haystack_index % needles.size();
        switch (kind) {

        case find_many_haystack_generator_t::planted_exact_k: {
            span<char const> const needle = needles[first_needle];
            for (int index = 0; index < noise_length(generator); ++index)
                scratch.push_back((char)noise_byte(generator));
            scratch.append(needle.data(), needle.size());
            for (int index = 0; index < noise_length(generator); ++index)
                scratch.push_back((char)noise_byte(generator));
            break;
        }

        case find_many_haystack_generator_t::planted_concatenated_k:
            // No separator, so matches of different needles overlap and nest at the seams.
            for (std::size_t offset = 0; offset < 4 && offset < needles.size(); ++offset) {
                span<char const> const needle = needles[(first_needle + offset) % needles.size()];
                scratch.append(needle.data(), needle.size());
            }
            break;

        case find_many_haystack_generator_t::near_miss_k: {
            // Every planted needle has its final byte perturbed, so the walk descends deep and then fails.
            span<char const> const needle = needles[first_needle];
            scratch.append(needle.data(), needle.size());
            if (!scratch.empty()) scratch.back() = (char)(scratch.back() ^ 0x20);
            for (int index = 0; index < noise_length(generator); ++index)
                scratch.push_back((char)noise_byte(generator));
            break;
        }

        case find_many_haystack_generator_t::fold_preimage_k: {
            // Only an uncased dictionary can match these; a cased one must report nothing at all.
            span<char const> const needle = needles[first_needle];
            for (std::size_t index = 0; index < needle.size(); ++index) {
                char const letter = needle[index];
                if (letter == 'k' && index + 1 == needle.size()) { scratch.append("\xE2\x84\xAA"); } // ? Kelvin U+212A
                else if (letter >= 'a' && letter <= 'z') { scratch.push_back((char)(letter - 'a' + 'A')); }
                else { scratch.push_back(letter); }
            }
            break;
        }

        case find_many_haystack_generator_t::boundary_stress_k: {
            // Flush with the first byte and flush with the last, plus a self-repeat in between, so a match
            // lands on every edge a chunked or windowed walk could mishandle.
            span<char const> const needle = needles[first_needle];
            scratch.append(needle.data(), needle.size());
            scratch.append(needle.data(), needle.size());
            break;
        }

        case find_many_haystack_generator_t::count_k: break; // ? Never a real generator, only the sweep's bound
        }
        verify(haystacks.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
    }
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer vectors for `find_many`, pinned by hand rather than against a second backend.
 *
 *  Covers the textbook Aho-Corasick overlap example - "he" and "she" both completing at the same end offset
 *  over "ushers" - attribution of matches to the right haystack in a batch, and a self-overlapping
 *  vocabulary of {"a", "ab", "abc"} over "abcabc". Every match is reported, nested ones included; this is
 *  not leftmost-first matching.
 */
void test_find_many_unit() {
    std::printf("  - testing find_many known-answer vectors...\n");

    // "she" and "he" both complete the instant the scan reaches the shared 'e', so both are reported at the
    // same end offset; "his" never occurs in "ushers" and contributes nothing.
    {
        std::vector<std::string> const needle_strings {"he", "she", "his", "hers"};
        std::vector<std::string> const haystack_strings {"ushers"};
        arrow_strings_tape_t needles, haystacks;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);

        find_many_u32_serial_t engine;
        verify(engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);
        find_many_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

        find_many_match_set_t const expected {
            {0, 0, 2, 2}, // "he"   at byte [2, 4)
            {0, 1, 1, 3}, // "she"  at byte [1, 4) - same end offset as "he"
            {0, 3, 2, 4}, // "hers" at byte [2, 6)
        };
        verify(matches == expected && "Classic he/she/his/hers overlap example mismatched");
    }

    // Matches attribute to the right haystack in a batch, including a haystack with none at all.
    {
        std::vector<std::string> const needle_strings {"cat", "dog"};
        std::vector<std::string> const haystack_strings {"cats and dogs", "no pets here", "dogcat"};
        arrow_strings_tape_t needles, haystacks;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);

        find_many_u32_serial_t engine;
        verify(engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);
        find_many_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

        find_many_match_set_t const expected {
            {0, 0, 0, 3}, // "cat" in haystack 0 at byte [0, 3)
            {0, 1, 9, 3}, // "dog" in haystack 0 at byte [9, 12)
            {2, 1, 0, 3}, // "dog" in haystack 2 at byte [0, 3)
            {2, 0, 3, 3}, // "cat" in haystack 2 at byte [3, 6)
        };
        verify(matches == expected && "Batch haystack attribution mismatched");
    }

    // A vocabulary where every needle is a prefix of the next only makes sense once every overlapping and
    // nested match is reported - never leftmost-first, never longest-only.
    {
        std::vector<std::string> const needle_strings {"a", "ab", "abc"};
        std::vector<std::string> const haystack_strings {"abcabc"};
        arrow_strings_tape_t needles, haystacks;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);

        find_many_u32_serial_t engine;
        verify(engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);
        find_many_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

        find_many_match_set_t const expected {
            {0, 0, 0, 1}, {0, 1, 0, 2}, {0, 2, 0, 3}, // "a", "ab", "abc" at the first occurrence
            {0, 0, 3, 1}, {0, 1, 3, 2}, {0, 2, 3, 3}, // and again at the second
        };
        verify(matches == expected && "Nested-prefix overlap example mismatched");
    }

    // A multi-byte source codepoint immediately followed by a byte that itself continues a valid preimage
    // chain must produce exactly the one match the accepting codepoint completes - not an extra match
    // starting mid-codepoint, at the continuation byte, once the walk carries on past it.
    {
        std::vector<std::string> const needle_strings {"ss", "\xC3\x9F"};
        std::vector<std::string> const haystack_strings {"\xC3\x9Fsoft"};
        arrow_strings_tape_t needles, haystacks;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);

        find_many_u32_serial_t engine;
        verify(engine.try_build(needles.view(), find_many_uncased_k) == status_t::success_k);
        find_many_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

        find_many_match_set_t const expected {
            {0, 0, 0, 2}, // "ss" matches the sharp-S codepoint at byte [0, 2)
            {0, 1, 0, 2}, // "\xC3\x9F" matches itself at byte [0, 2)
        };
        verify(matches == expected && "Spurious match starting mid-codepoint after a multi-byte accept");
    }
}

#pragma endregion // Unit

#pragma region Uncased Conformance

/** @brief Builds a single-needle uncased dictionary and asserts every `must_match` haystack matches at least
 *         once, and every `must_not_match` haystack matches zero times. */
static void check_uncased_needle_matches_(char const *needle, std::vector<std::string> const &must_match,
                                          std::vector<std::string> const &must_not_match) {
    span<char const> const needle_span(needle, std::strlen(needle));
    std::vector<span<char const>> const needles {needle_span};

    find_many_u32_serial_t engine;
    verify(engine.try_build(needles, find_many_uncased_k) == status_t::success_k);

    for (std::string const &haystack : must_match) {
        std::vector<span<char const>> const haystacks {span<char const>(haystack.data(), haystack.size())};
        std::vector<std::size_t> counts(1, 0);
        std::size_t matches_total = 0;
        verify(engine.try_count(haystacks, span<std::size_t>(counts.data(), counts.size()), matches_total) ==
               status_t::success_k);
        verify(counts[0] >= 1 && "Needle must match this haystack under full case folding");
    }
    for (std::string const &haystack : must_not_match) {
        std::vector<span<char const>> const haystacks {span<char const>(haystack.data(), haystack.size())};
        std::vector<std::size_t> counts(1, 0);
        std::size_t matches_total = 0;
        verify(engine.try_count(haystacks, span<std::size_t>(counts.data(), counts.size()), matches_total) ==
               status_t::success_k);
        verify(counts[0] == 0 && "Needle must not match this haystack");
    }
}

/**
 *  @brief Full Unicode case-folding conformance table for `find_many_uncased_k`, then a differential against
 *         `independent_uncased_matches_`.
 *
 *  Every non-ASCII fixture is spelled as `\xHH` byte escapes, never as a raw literal or a `\u` universal
 *  character name - both have been silently re-encoded by tooling here before. A byte-level check against a
 *  numeric expected-bytes array guards the trickiest fixtures.
 */
void test_find_many_uncased() {
    std::printf("  - testing full Unicode case-folding conformance...\n");

    // Corruption guard: two load-bearing fixtures re-spelled as integer byte arrays, so a tool that silently
    // renormalizes the string literals below is caught here.
    {
        unsigned char const sharp_s_expected[] = {0xC3, 0x9F};
        verify(std::memcmp("\xC3\x9F", sharp_s_expected, 2) == 0 && "Sharp S literal corrupted");
        unsigned char const kelvin_sign_expected[] = {0xE2, 0x84, 0xAA};
        verify(std::memcmp("\xE2\x84\xAA", kelvin_sign_expected, 3) == 0 && "Kelvin sign literal corrupted");
    }

    // | Needle | Must match                                | Must NOT match |
    check_uncased_needle_matches_("ss", {"ss", "SS", "sS", "Ss", "\xC3\x9F", "\xE1\xBA\x9E"}, {"s"});
    check_uncased_needle_matches_("\xC3\x9F", {"\xC3\x9F", "\xE1\xBA\x9E", "ss", "SS", "sS", "Ss"}, {"s"});
    check_uncased_needle_matches_("K", {"K", "k", "temp\xE2\x84\xAAvalue"}, {});
    check_uncased_needle_matches_("\xC3\x85", {"\xC3\x85", "\xC3\xA5", "temp\xE2\x84\xABvalue"}, {"A\xCC\x8A"});
    check_uncased_needle_matches_("\xC4\xB0", {"i\xCC\x87"}, {"i", "I"});
    check_uncased_needle_matches_("\xC3\xA9", {"\xC3\xA9", "\xC3\x89"}, {"e\xCC\x81"});

    // A length-changing fold in the MIDDLE of a needle, not only at its end, so the byte-delta state keying
    // that reconverges variable-length preimages is actually exercised mid-walk, not just at acceptance.
    check_uncased_needle_matches_("wei\xC3\x9Frd", {"weissrd", "weiSSrd", "wei\xE1\xBA\x9Erd"}, {"weisrd", "weird"});

    // Differential: for a spread of needles and randomized valid UTF-8 haystacks, the full match set found by
    // `find_many` must equal the full match set found by the independent fold-and-scan oracle.
    std::printf("    - differential against the independent fold-and-scan oracle...\n");
    std::vector<std::string> const needle_pool {
        "ss", "\xC3\x9F", "K", "\xC3\x85", "\xC4\xB0", "\xC3\xA9", "wei\xC3\x9Frd", "the",
    };
    span<string_view const> const empty_motifs;
    auto &generator = global_random_generator();
    for (std::size_t iteration = 0; iteration < scale_iterations(60); ++iteration) {
        std::string const &needle = needle_pool[iteration % needle_pool.size()];
        std::string haystack;
        utf8_random_segmentation_corpus_(haystack, 96, utf8_corpus_flavor_t::valid_k, utf8_default_alphabet,
                                         empty_motifs, generator);
        haystack.append(needle); // ? Guarantees at least one hit most iterations, without excluding zero-hit ones.

        std::vector<span<char const>> const needles {span<char const>(needle.data(), needle.size())};
        find_many_u32_serial_t engine;
        verify(engine.try_build(needles, find_many_uncased_k) == status_t::success_k);
        std::vector<std::string> const haystack_strings {haystack};
        arrow_strings_tape_t haystacks;
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        find_many_match_set_t engine_matches;
        collect_matches_into_(engine, haystacks.view(), engine_matches);

        auto const oracle_matches = independent_uncased_matches_(haystacks.view()[0], needles[0]);
        find_many_match_set_t expected_matches;
        for (auto const &oracle_match : oracle_matches)
            expected_matches.append({0, 0, oracle_match.first, oracle_match.second});
        expected_matches.finalize();

        if (engine_matches != expected_matches) {
            std::fprintf(stderr, "Uncased differential mismatch for needle \"%s\": %zu vs %zu matches\n",
                         needle.c_str(), engine_matches.size(), expected_matches.size());
            for (find_many_match_t const &key : engine_matches)
                std::fprintf(stderr, "  engine offset=%zu length=%zu\n", key.byte_offset, key.byte_length);
            for (find_many_match_t const &key : expected_matches)
                std::fprintf(stderr, "  oracle offset=%zu length=%zu\n", key.byte_offset, key.byte_length);
            verify(false && "find_many disagrees with the independent fold-and-scan oracle");
        }
    }
}

#pragma endregion // Uncased Conformance

#pragma region Agreement

/**
 *  @brief A ONE-needle dictionary must agree exactly with the shipped `sz_utf8_uncased_search` over a fuzz
 *         corpus - that agreement is the entire semantic claim of `find_many_uncased_k`.
 *
 *  `find_many` reports every overlapping match, while `sz_utf8_uncased_search` reports only the first by
 *  start offset, so the comparison reduces `find_many`'s set to its own earliest-starting match.
 */
void test_find_many_agreement() {
    std::printf("  - testing one-needle agreement with sz_utf8_uncased_search...\n");

    std::vector<std::string> const needle_pool {
        "the", "quick", "STRASSE", "stra\xC3\x9F" "e", "ss", "K", "caf\xC3\xA9", "\xC3\x85ngstrom",
    };
    span<string_view const> const empty_motifs;
    auto &generator = global_random_generator();

    for (std::size_t iteration = 0; iteration < scale_iterations(200); ++iteration) {
        std::string const &needle = needle_pool[iteration % needle_pool.size()];

        std::string haystack;
        utf8_random_segmentation_corpus_(haystack, 80, utf8_corpus_flavor_t::valid_k, utf8_default_alphabet,
                                         empty_motifs, generator);
        if ((iteration & 1) == 0) haystack.append(needle); // ? Half the iterations guarantee a hit.

        std::vector<span<char const>> const needles {span<char const>(needle.data(), needle.size())};
        find_many_u32_serial_t engine;
        verify(engine.try_build(needles, find_many_uncased_k) == status_t::success_k);
        std::vector<std::string> const haystack_strings {haystack};
        arrow_strings_tape_t haystacks;
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        find_many_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

        bool engine_found = false;
        std::size_t engine_offset = 0, engine_length = 0;
        for (find_many_match_t const &match : matches)
            if (!engine_found || match.byte_offset < engine_offset) {
                engine_found = true;
                engine_offset = match.byte_offset;
                engine_length = match.byte_length;
            }

        sz_utf8_uncased_needle_metadata_t metadata = {};
        sz_size_t reference_length = 0;
        // The serial variant rather than the dispatched one: `SZ_API_COMPTIME`, so it needs no stringzilla
        // core library on this binary's link line, and it is the reference every backend is validated against.
        sz_cptr_t const reference_result = sz_utf8_uncased_search_serial(
            haystack.data(), haystack.size(), needle.data(), needle.size(), &metadata, &reference_length);
        bool const reference_found = reference_result != SZ_NULL_CHAR;
        std::size_t const reference_offset = reference_found ? (std::size_t)(reference_result - haystack.data()) : 0;

        bool const agrees = engine_found == reference_found &&
                            (!engine_found || (engine_offset == reference_offset && engine_length == reference_length));
        if (!agrees) {
            std::fprintf(                                                                           //
                stderr,                                                                             //
                "Agreement mismatch for needle \"%s\": find_many found=%d offset=%zu length=%zu | " //
                "sz_utf8_uncased_search found=%d offset=%zu length=%zu\n",                          //
                needle.c_str(), engine_found, engine_offset, engine_length, reference_found, reference_offset,
                reference_length);
            verify(false && "find_many and sz_utf8_uncased_search disagree");
        }
    }
}

#pragma endregion // Agreement

#pragma region Adversarial

/**
 *  @brief Everything one adversarial cell needs, constructed once by the sweep and refilled per cell.
 *
 *  No member is ever rebuilt: the generators `reset` the tapes they fill and every collector `clear`s the
 *  buffers it owns, so one backing allocation survives the whole sweep.
 */
struct find_many_adversarial_scratch_t {
    /** @brief Which generator pair produced the current cell, so a failing check can name its configuration
     *         instead of leaving it to be reconstructed from a step index. */
    find_many_needle_generator_t needle_kind = find_many_needle_generator_t::self_overlapping_k;
    find_many_haystack_generator_t haystack_kind = find_many_haystack_generator_t::planted_exact_k;

    arrow_strings_tape_t needles;
    arrow_strings_tape_t haystacks;
    find_many_match_set_t engine_keys, oracle_keys, variant_keys;
#if SZ_USE_CUDA
    find_many_match_set_t cuda_keys;
#endif
};

/** @brief Reports @p what went wrong and which generator pair produced the cell, mirroring how
 *         `edit_distance_log_mismatch` prefixes a similarity failure with the pair that caused it. */
void log_find_many_cell_mismatch_(find_many_adversarial_scratch_t const &scratch, char const *what) {
    std::fprintf(stderr, "%s on needles=%s haystacks=%s\n", what, find_many_needle_generator_name(scratch.needle_kind),
                 find_many_haystack_generator_name(scratch.haystack_kind));
}

/**
 *  @brief The indexing check: every match the serial engine reports must agree with brute force, needle,
 *         offset, length and haystack all at once. Runs on every cell.
 */
void check_find_many_against_oracle_(find_many_case_sensitivity_t sensitivity,
                                     find_many_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    find_many_u32_serial_t engine;
    verify(engine.try_build(needles_view, sensitivity) == status_t::success_k);
    collect_matches_into_(engine, haystacks_view, scratch.engine_keys);
    collect_independent_matches_(sensitivity, needles_view, haystacks_view, scratch.oracle_keys);

    if (scratch.engine_keys != scratch.oracle_keys) {
        log_find_many_cell_mismatch_(scratch, "Oracle disagreement");
        verify(false && "find_many disagrees with the brute-force vocabulary oracle");
    }
}

/** @brief Serial and fork-union-parallel must report the identical set on the same cell. */
void check_find_many_backends_agree_(find_many_case_sensitivity_t sensitivity,
                                     find_many_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    find_many_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles_view, sensitivity) == status_t::success_k);
    collect_matches_into_(serial_engine, haystacks_view, scratch.engine_keys);

    find_many_u32_parallel_t parallel_engine;
    verify(parallel_engine.try_build(needles_view, sensitivity) == status_t::success_k);
    collect_matches_into_(parallel_engine, haystacks_view, scratch.variant_keys, dummy_executor_t {});
    if (scratch.engine_keys != scratch.variant_keys) {
        log_find_many_cell_mismatch_(scratch, "Serial-vs-parallel divergence");
        verify(false && "Serial and parallel backends disagree");
    }
}

/**
 *  @brief A `u16` automaton over the same cell either agrees exactly or declines with `overflow_risk_k`.
 *
 *  Declining is a legitimate outcome - the narrower id space caps at 65534 states, and an adversarial
 *  vocabulary is built to reach ceilings - so the property is that it never silently truncates instead.
 */
void check_find_many_narrow_width_(find_many_case_sensitivity_t sensitivity, find_many_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    find_many_u16_dictionary_t narrow;
    narrow.case_sensitivity(sensitivity);
    for (std::size_t index = 0; index < needles_view.size(); ++index) {
        status_t const status = narrow.try_insert(needles_view[index]);
        if (status == status_t::overflow_risk_k) return;
        verify(status == status_t::success_k);
    }
    if (narrow.try_build() == status_t::overflow_risk_k) return;

    scratch.variant_keys.clear();
    for (std::size_t haystack_index = 0; haystack_index < haystacks_view.size(); ++haystack_index)
        narrow.find(haystacks_view[haystack_index],
                    [&](std::size_t needle_index, std::size_t match_offset, std::size_t match_length) noexcept {
                        scratch.variant_keys.append({haystack_index, needle_index, match_offset, match_length});
                        return true;
                    });
    scratch.variant_keys.finalize();

    collect_independent_matches_(sensitivity, needles_view, haystacks_view, scratch.oracle_keys);
    if (scratch.variant_keys != scratch.oracle_keys) {
        log_find_many_cell_mismatch_(scratch, "Narrow-width divergence");
        verify(false && "u16 automaton disagrees with the oracle");
    }
}

/**
 *  @brief The same dictionary with `hot_count` forced from fully cold to fully hot must report an identical
 *         set every time.
 *
 *  A small dictionary is entirely hot by default and never enters the cold double array at all, so a
 *  tier-boundary bug is otherwise invisible. Driven through `aho_corasick_dictionary::hot_count` directly,
 *  the engine wrapper having no hook to override the split before `try_build`.
 */
void check_find_many_tier_invariant_(find_many_case_sensitivity_t sensitivity,
                                     find_many_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    std::size_t raw_state_count = 0;
    {
        find_many_u32_dictionary_t probe;
        probe.case_sensitivity(sensitivity);
        for (std::size_t index = 0; index < needles_view.size(); ++index)
            verify(probe.try_insert(needles_view[index]) == status_t::success_k);
        raw_state_count = probe.count_states();
    }
    verify(raw_state_count > 0);

    std::size_t const hot_counts[] = {0, 1, raw_state_count / 2, raw_state_count};
    bool saw_cold_tier = false, saw_all_hot = false;
    for (std::size_t variant_index = 0; variant_index < 4; ++variant_index) {
        find_many_u32_dictionary_t dictionary;
        dictionary.case_sensitivity(sensitivity);
        for (std::size_t index = 0; index < needles_view.size(); ++index)
            verify(dictionary.try_insert(needles_view[index]) == status_t::success_k);
        dictionary.hot_count(hot_counts[variant_index]);
        verify(dictionary.try_build() == status_t::success_k);

        auto const automaton = dictionary.view();
        saw_cold_tier |= automaton.hot_count < automaton.state_count;
        saw_all_hot |= automaton.all_hot();

        scratch.variant_keys.clear();
        for (std::size_t haystack_index = 0; haystack_index < haystacks_view.size(); ++haystack_index)
            dictionary.find(haystacks_view[haystack_index],
                            [&](std::size_t needle_index, std::size_t match_offset, std::size_t match_length) noexcept {
                                scratch.variant_keys.append({haystack_index, needle_index, match_offset, match_length});
                                return true;
                            });
        scratch.variant_keys.finalize();

        if (variant_index == 0) { scratch.engine_keys = scratch.variant_keys; }
        else if (scratch.variant_keys != scratch.engine_keys) {
            log_find_many_cell_mismatch_(scratch, "Tier mismatch");
            std::fprintf(stderr, "  at hot_count=%zu (hot=%u, states=%u): %zu vs %zu matches\n",
                         hot_counts[variant_index], automaton.hot_count, automaton.state_count,
                         scratch.variant_keys.size(), scratch.engine_keys.size());
            verify(false && "Match set changed across the hot/cold tier boundary");
        }
    }
    verify(saw_cold_tier && "Tier sweep never exercised the cold tier - grow the vocabulary");
    verify(saw_all_hot && "Tier sweep never exercised the fully-hot path");
}

#if SZ_USE_CUDA
/** @brief The device backend must report the identical set for the same cell, through the very same call
 *         shape every host backend takes - only the memory the haystacks live in differs. */
void check_find_many_cuda_agrees_(find_many_case_sensitivity_t sensitivity, find_many_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    find_many_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles_view, sensitivity) == status_t::success_k);
    collect_matches_into_(serial_engine, haystacks_view, scratch.engine_keys);

    gpu_specs_t gpu_specs;
    verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
    cuda_executor_t executor;
    find_many_u32_cuda_t cuda_engine;
    verify(cuda_engine.try_build(needles_view, sensitivity, executor) == status_t::success_k);

    // One match type and one signature across backends: the device runs the very same collection body as
    // every CPU engine. The fixture tape is `unified_alloc`-backed under CUDA, so no copy exists anywhere.
    collect_matches_into_(cuda_engine, haystacks_view, scratch.cuda_keys, executor, gpu_specs);
    if (scratch.cuda_keys != scratch.engine_keys) {
        log_find_many_cell_mismatch_(scratch, "CUDA-vs-serial divergence");
        verify(false && "CUDA backend disagrees with the serial reference");
    }
}

/**
 *  @brief Pins the two sides of the device memory contract: scattered device-resident haystacks - several
 *         separate unified allocations rather than one packed tape - are searched correctly, and host-backed
 *         haystacks are refused with `device_memory_mismatch_k` rather than silently copied.
 */
void test_find_many_cuda_memory_contract() {
    std::printf("  - testing scattered unified haystacks and the host-memory refusal...\n");

    gpu_specs_t gpu_specs;
    verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
    cuda_executor_t executor;

    std::vector<std::string> const needle_strings {"he", "she", "his", "hers"};
    arrow_strings_tape_t needles;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    find_many_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);
    find_many_u32_cuda_t cuda_engine;
    verify(cuda_engine.try_build(needles.view(), find_many_cased_k, executor) == status_t::success_k);

    // Three haystacks in three separate unified allocations: the descriptors point wherever the caller's
    // memory happens to live, which is the whole point of not assuming a tape.
    std::vector<std::string> const texts {"ushers", "hishers", "she"};
    std::vector<unified_vector<char>> scattered_storage(texts.size());
    unified_vector<span<char const>> scattered_spans(texts.size());
    for (std::size_t index = 0; index < texts.size(); ++index) {
        scattered_storage[index].assign(texts[index].begin(), texts[index].end());
        scattered_spans[index] = {scattered_storage[index].data(), scattered_storage[index].size()};
    }
    span<span<char const> const> const scattered_view {scattered_spans.data(), scattered_spans.size()};

    arrow_strings_tape_t reference_haystacks;
    verify(reference_haystacks.try_assign(texts.data(), texts.data() + texts.size()) == status_t::success_k);
    find_many_match_set_t serial_keys, scattered_keys;
    collect_matches_into_(serial_engine, reference_haystacks.view(), serial_keys);
    collect_matches_into_(cuda_engine, scattered_view, scattered_keys, executor, gpu_specs);
    verify(!serial_keys.empty() && "The fixture must produce matches");
    verify(scattered_keys == serial_keys && "Scattered unified allocations must match the packed reference");

    // Host memory is refused, never copied: materialization is the caller's explicit choice, as in every
    // other domain. The strings above live on the host, so their spans are exactly the illegal input.
    std::vector<span<char const>> host_spans;
    for (std::string const &text : texts) host_spans.push_back({text.data(), text.size()});
    unified_vector<std::size_t> counts(host_spans.size());
    std::size_t matches_total = 0;
    auto const host_status = cuda_engine.try_count(host_spans, span<std::size_t>(counts.data(), counts.size()),
                                                   matches_total, executor, gpu_specs);
    verify(host_status == status_t::device_memory_mismatch_k && "Host input must be refused, not copied");
}
#endif

/**
 *  @brief Crosses every adversarial needle vocabulary with every adversarial haystack construction.
 *
 *  Walked by rotation rather than nested loops: `rotating_index` advances a phase each full turn, so
 *  crossing two rotations still reaches every pair, and `scale_iterations` decides how far into that product
 *  a run gets. Ground truth runs on every cell; the costlier properties rotate, keeping the per-cell price
 *  flat while the sweep as a whole still exercises each of them against each generator pair.
 */
void test_find_many_adversarial() {
    std::printf("  - testing adversarial needle x haystack cross-product...\n");
    std::size_t const needle_generators = (std::size_t)find_many_needle_generator_t::count_k;
    std::size_t const haystack_generators = (std::size_t)find_many_haystack_generator_t::count_k;
    std::size_t const cells = scale_iterations(needle_generators * haystack_generators);
    find_many_adversarial_scratch_t scratch; // ? Constructed once, refilled by every cell below.

    for (std::size_t step = 0; step < cells; ++step) {
        auto const needle_kind = (find_many_needle_generator_t)rotating_index(step, needle_generators);
        auto const haystack_kind = (find_many_haystack_generator_t)rotating_index(step, haystack_generators);
        // A vocabulary of arbitrary bytes can only be matched byte-exactly: folding rejects malformed UTF-8
        // by contract, and the brute-force oracle decodes its inputs too.
        auto const sensitivity = step % 2 && find_many_needles_are_utf8_(needle_kind) ? find_many_uncased_k
                                                                                      : find_many_cased_k;

        scratch.needle_kind = needle_kind;
        scratch.haystack_kind = haystack_kind;

        // A vocabulary whose output pool grows with the square of its depth takes the quadratic knob, so
        // doubling the multiplier doubles the work rather than quadrupling it.
        std::size_t const depth = find_many_needles_grow_quadratically_(needle_kind) ? scale_iterations_quadratic(10)
                                                                                     : scale_iterations(10);
        generate_find_many_needles_(needle_kind, depth, scratch.needles);
        generate_find_many_haystacks_(haystack_kind, scratch.needles.view(), 6, scratch.haystacks);

        check_find_many_against_oracle_(sensitivity, scratch);
        switch (rotating_index(step, 3)) {
        case 0: check_find_many_backends_agree_(sensitivity, scratch); break;
        case 1: check_find_many_narrow_width_(sensitivity, scratch); break;
        case 2: check_find_many_tier_invariant_(sensitivity, scratch); break;
        }
#if SZ_USE_CUDA
        check_find_many_cuda_agrees_(sensitivity, scratch);
#endif
    }
}

/**
 *  @brief The parallel engine's large-haystack path - every core slicing one haystack - must agree with the
 *         serial engine even when matches start on nearly every byte and straddle every slice boundary.
 *
 *  The default `cpu_specs_t` L2 threshold keeps the adversarial fixtures above on the one-core-per-haystack
 *  path, so this one forces the slicing with a threshold far below its own size and a real pool.
 */
void test_find_many_large_haystacks() {
    std::printf("  - testing the all-cores-on-one-large-haystack path...\n");

    forkunion_executor_t pool;
    verify(pool.try_spawn(4) == status_t::success_k);
    cpu_specs_t sliced_specs;
    sliced_specs.l2_bytes = 1024; // ? Far below the fixture, so `is_large_` takes the all-cores path

    // Overlapping needles over a periodic haystack: "ab" completes at every second byte, "abababab" spans
    // whole slice overlaps, so every boundary sees matches that begin before it and end after it.
    std::vector<std::string> const needle_strings {"ab", "aba", "bab", "abababab"};
    arrow_strings_tape_t needles;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    std::string periodic(scale_iterations(8) * 1024, '\0');
    for (std::size_t index = 0; index < periodic.size(); ++index) periodic[index] = index % 2 ? 'b' : 'a';
    std::vector<std::string> const haystack_strings {periodic};
    arrow_strings_tape_t haystacks;
    verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
           status_t::success_k);

    find_many_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);
    find_many_match_set_t serial_keys, parallel_keys;
    collect_matches_into_(serial_engine, haystacks.view(), serial_keys);

    find_many_u32_parallel_t parallel_engine;
    verify(parallel_engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);
    collect_matches_into_(parallel_engine, haystacks.view(), parallel_keys, pool, sliced_specs);

    verify(!serial_keys.empty() && "The fixture must produce matches");
    verify(parallel_keys == serial_keys && "The sliced parallel path disagrees with the serial reference");
}

#pragma endregion // Adversarial

#pragma region Construction

/**
 *  @brief Structural invariants of the compiled automaton, checked directly against the published
 *         `aho_corasick_view` rather than against another backend's output.
 */
void test_find_many_construction() {
    std::printf("  - testing structural invariants of the compiled automaton...\n");

    // Overlap policy, pinned as a count rather than left implicit: every occurrence of every needle is
    // reported, including the ones nested inside or overlapping a longer match. Over "abcabc", the
    // vocabulary {"a", "ab", "abc"} completes each of its three needles at each of two repetitions.
    {
        std::vector<std::string> const needle_strings {"a", "ab", "abc"};
        std::vector<std::string> const haystack_strings {"abcabc"};
        arrow_strings_tape_t needles, haystacks;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        find_many_u32_serial_t engine;
        verify(engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);
        find_many_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);
        verify(matches.size() == 6 && "All matches are reported - this is not leftmost-longest matching");
    }

    // An empty needle is rejected outright. Skipping it silently would consume no needle index, so every
    // later needle's reported `needle_index` would disagree with the caller's own array.
    {
        find_many_u32_dictionary_t dictionary;
        verify(dictionary.try_insert(span<char const> {}) == status_t::unexpected_dimensions_k);
        verify(dictionary.count_needles() == 0 && "A rejected needle must not consume an index");
        verify(dictionary.try_insert(span<char const> {"ab", 2}) == status_t::success_k);
        verify(dictionary.count_needles() == 1);
    }

    // Cold-tier double-array invariant: every slot a state claims sits within 256 of that state's own base,
    // and the failure chain from any cold state reaches the root. `hot_count` is forced to zero, so the cold
    // tier is genuinely exercised whatever the default hot tier would have been.
    {
        std::vector<std::string> const needle_strings = random_short_strings_(scale_iterations(300), 3, 7);
        arrow_strings_tape_t needles;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        find_many_u32_dictionary_t dictionary;
        dictionary.case_sensitivity(find_many_cased_k);
        for (span<char const> const &needle : needles.view())
            verify(dictionary.try_insert(needle) == status_t::success_k);
        dictionary.hot_count(0);
        verify(dictionary.try_build() == status_t::success_k);

        auto const automaton = dictionary.view();
        constexpr u32_t invalid_state_k = std::numeric_limits<u32_t>::max();
        std::size_t const cold_capacity = (std::size_t)automaton.state_count + 255;
        bool exercised_cold_tier = false;
        for (std::size_t slot = automaton.hot_count; slot < cold_capacity; ++slot) {
            u32_t const owner = automaton.check[slot];
            if (owner == invalid_state_k) continue;
            verify(owner < automaton.state_count && "Cold slot owned by an out-of-range state");
            std::size_t const base_of_owner = automaton.base[owner];
            verify(slot >= base_of_owner && slot - base_of_owner < 256 &&
                   "Cold slot's offset from its owner's base is not a valid byte");
            exercised_cold_tier = true;
        }
        verify(exercised_cold_tier && "Construction sweep never exercised the cold tier - grow the vocabulary");

        for (u32_t state = automaton.hot_count; state < automaton.state_count; ++state) {
            u32_t cursor = state;
            std::size_t hops = 0;
            while (cursor != automaton.root && hops <= automaton.state_count) {
                cursor = automaton.fail[cursor];
                ++hops;
            }
            verify(cursor == automaton.root && "Failure chain never reaches the root");
        }
    }

    // Uncased match spans land on UTF-8 codepoint boundaries at both ends.
    {
        std::vector<std::string> const needle_strings {"ss", "\xC3\x9F", "caf\xC3\xA9", "K"};
        arrow_strings_tape_t needles;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        find_many_u32_serial_t engine;
        verify(engine.try_build(needles.view(), find_many_uncased_k) == status_t::success_k);

        span<string_view const> const empty_motifs;
        auto &generator = global_random_generator();
        std::vector<std::string> haystack_storage;
        for (std::size_t index = 0; index < scale_iterations(20); ++index) {
            std::string corpus;
            utf8_random_segmentation_corpus_(corpus, 200, utf8_corpus_flavor_t::valid_k, utf8_default_alphabet,
                                             empty_motifs, generator);
            haystack_storage.push_back(std::move(corpus));
        }
        arrow_strings_tape_t haystacks;
        verify(haystacks.try_assign(haystack_storage.data(), haystack_storage.data() + haystack_storage.size()) ==
               status_t::success_k);
        find_many_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

        std::vector<std::size_t> counts(haystacks.view().size(), 0);
        std::size_t matches_total = 0;
        verify(engine.try_count(haystacks.view(), span<std::size_t>(counts.data(), counts.size()), matches_total) ==
               status_t::success_k);
        std::vector<find_many_match_t> raw_matches(matches_total);
        std::size_t matches_found = 0;
        verify(engine.try_find(haystacks.view(), span<find_many_match_t>(raw_matches.data(), raw_matches.size()),
                               matches_found) == status_t::success_k);

        for (find_many_match_t const &match : raw_matches) {
            // The match carries offsets only, so the bytes come from the haystack it names - the exact
            // recipe the match struct's own docstring prescribes.
            span<char const> const haystack = haystacks.view()[match.haystack_index];
            std::size_t const end = match.byte_offset + match.byte_length;
            bool const start_is_boundary = (haystack[match.byte_offset] & 0xC0) != 0x80;
            bool const end_is_boundary = end == haystack.size() || (haystack[end] & 0xC0) != 0x80;
            if (!start_is_boundary || !end_is_boundary)
                std::fprintf(stderr, "Boundary violation: needle_index=%zu byte_offset=%zu byte_length=%zu\n",
                             match.needle_index, match.byte_offset, match.byte_length);
            verify(start_is_boundary && "Uncased match starts mid-codepoint");
            verify(end_is_boundary && "Uncased match ends mid-codepoint");
        }
    }
}

#pragma endregion // Construction

#pragma region Safety

/**
 *  @brief The three degenerate output-buffer shapes: an empty batch, an undersized buffer, and an oversized
 *         one. The cases above always size `matches` to exactly the count `try_count` produced.
 */
void test_find_many_buffer_contracts() {
    std::printf("  - testing empty, undersized and oversized output buffers...\n");

    std::vector<std::string> const needle_strings {"he", "she", "his", "hers"};
    std::vector<std::string> const haystack_strings {"ushers", "hishers"};
    arrow_strings_tape_t needles, haystacks;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
           status_t::success_k);

    find_many_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);
    find_many_u32_parallel_t parallel_engine;
    verify(parallel_engine.try_build(needles.view(), find_many_cased_k) == status_t::success_k);

    std::vector<std::size_t> counts(haystacks.view().size(), 0);
    std::size_t required = 0;
    verify(serial_engine.try_count(haystacks.view(), span<std::size_t>(counts.data(), counts.size()), required) ==
           status_t::success_k);
    verify(required > 1 && "fixtures must produce several matches for the undersized case to mean anything");

    std::size_t matches_found = 0;

    // An empty batch must be accepted and write nothing, rather than dereferencing an unallocated offsets
    // array while seeding its first prefix-sum entry.
    {
        std::vector<span<char const>> const no_haystacks;
        span<find_many_match_t> const no_matches;
        verify(parallel_engine.try_find(no_haystacks, no_matches, matches_found) == status_t::success_k);
        verify(matches_found == 0);
    }

    // An undersized buffer must be refused before anything is written, on both backends, and the reported
    // count must stay zero so a caller cannot mistake a refusal for a partial success.
    {
        std::vector<find_many_match_t> too_small(required - 1);
        span<find_many_match_t> const too_small_view(too_small.data(), too_small.size());
        verify(serial_engine.try_find(haystacks.view(), too_small_view, matches_found) ==
               status_t::unexpected_dimensions_k);
        verify(matches_found == 0);
        verify(parallel_engine.try_find(haystacks.view(), too_small_view, matches_found) ==
               status_t::unexpected_dimensions_k);
        verify(matches_found == 0);
    }

    // An oversized buffer is legal: `matches` supplies capacity, and `matches_found` states how much of it
    // was actually used.
    {
        std::vector<find_many_match_t> too_large(required + 16);
        span<find_many_match_t> const too_large_view(too_large.data(), too_large.size());
        verify(serial_engine.try_find(haystacks.view(), too_large_view, matches_found) == status_t::success_k);
        verify(matches_found == required);
        verify(parallel_engine.try_find(haystacks.view(), too_large_view, matches_found) == status_t::success_k);
        verify(matches_found == required);
    }
}

void test_find_many_safety() {
    std::printf("  - testing malformed-needle rejection and malformed-haystack robustness...\n");

    auto &generator = global_random_generator();

    // `append_malformed_class_`'s pool mixes ill-formed byte sequences with noncharacters, which are
    // reserved codepoints but structurally well-formed UTF-8; only the former must be rejected. `sz_rune_decode`
    // decides which is which, so this tracks structural well-formedness rather than assuming every sample is
    // malformed.
    bool saw_rejection = false, saw_acceptance = false;
    for (std::size_t index = 0; index < scale_iterations(80); ++index) {
        std::string malformed;
        append_malformed_class_(malformed, generator);
        std::vector<span<char const>> const needles {span<char const>(malformed.data(), malformed.size())};

        find_many_u32_serial_t exact_engine;
        verify(exact_engine.try_build(needles, find_many_cased_k) == status_t::success_k &&
               "Exact mode must accept any byte sequence as a needle");

        bool structurally_valid = true;
        for (char const *cursor = malformed.data(), *end = cursor + malformed.size(); cursor != end;) {
            sz_rune_t rune;
            sz_rune_length_t const consumed = sz_rune_decode(cursor, end, &rune);
            if (consumed == sz_rune_invalid_k) {
                structurally_valid = false;
                break;
            }
            cursor += consumed;
        }

        find_many_u32_serial_t uncased_engine;
        status_t const uncased_status = uncased_engine.try_build(needles, find_many_uncased_k);
        status_t const expected_status = structurally_valid ? status_t::success_k : status_t::invalid_utf8_k;
        verify(uncased_status == expected_status &&
               "Uncased mode's acceptance must track structural UTF-8 well-formedness exactly");
        saw_rejection |= !structurally_valid;
        saw_acceptance |= structurally_valid;
    }
    verify(saw_rejection && "Safety sweep never produced a genuinely ill-formed needle");
    verify(saw_acceptance && "Safety sweep never produced a structurally well-formed needle");

    // Positive control: a well-formed UTF-8 needle is accepted under case folding.
    {
        std::vector<span<char const>> const needles {span<char const>("caf\xC3\xA9", 5)};
        find_many_u32_serial_t engine;
        verify(engine.try_build(needles, find_many_uncased_k) == status_t::success_k &&
               "A well-formed UTF-8 needle must be accepted under case folding");
    }

    // Malformed and mutated haystacks never crash or hang a well-formed uncased dictionary, and `try_count`
    // and `try_find` always agree on how many matches there were.
    {
        std::vector<std::string> const needle_strings {"ss", "\xC3\x9F", "the", "K"};
        arrow_strings_tape_t needles;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        find_many_u32_serial_t engine;
        verify(engine.try_build(needles.view(), find_many_uncased_k) == status_t::success_k);

        span<string_view const> const empty_motifs;
        for (std::size_t index = 0; index < scale_iterations(80); ++index) {
            std::string haystack;
            utf8_random_segmentation_corpus_(haystack, 128, utf8_corpus_flavor_t::malformed_k, utf8_default_alphabet,
                                             empty_motifs, generator);
            apply_mutation_passes_(haystack, generator);
            std::vector<span<char const>> const haystacks {span<char const>(haystack.data(), haystack.size())};

            std::vector<std::size_t> counts(1, 0);
            std::size_t matches_total = 0;
            verify(engine.try_count(haystacks, span<std::size_t>(counts.data(), counts.size()), matches_total) ==
                   status_t::success_k);
            std::vector<find_many_match_t> matches(matches_total);
            std::size_t matches_found = 0;
            verify(engine.try_find(haystacks, span<find_many_match_t>(matches.data(), matches.size()), matches_found) ==
                   status_t::success_k);
            verify(matches_found == counts[0] && "try_count and try_find disagree on match count");
        }
    }
}

#pragma endregion // Safety

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
