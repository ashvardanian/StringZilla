/**
 *  @brief Extensive @b stress-testing suite for the StringZillas multi-pattern search engine (Aho-Corasick).
 *  @see Stress-tests on real-world and synthetic data are integrated into the @b `bench/substrings.cpp` and
 *       @b `bench/substrings.cu` benchmarks.
 *
 *  @file test/substrings.cuh
 *  @author Ash Vardanian
 *  @date June 16, 2026
 */
#include "stringzilla/utf8_uncased.h" // `sz_utf8_uncased_search`, the independent single-needle oracle

#include "stringzillas/substrings.hpp"

#if SZ_USE_CUDA
#include "stringzillas/substrings.cuh"
#endif

#if !SZ_IS_CPP17_
#error "This test requires C++17 or later."
#endif

#include <cmath>   // `std::fabs`
#include <cstdio>  // `std::printf`, `std::fprintf`
#include <cstring> // `std::memcmp`, `std::memcpy`

#include <algorithm> // `std::sort`, `std::unique`
#include <limits>    // `std::numeric_limits`
#include <map>       // `std::map`
#include <random>    // `std::mt19937`, `std::uniform_int_distribution`
#include <set>       // `std::set`
#include <string>    // `std::string`
#include <utility>   // `std::move`
#include <vector>    // `std::vector`

#include <stringzilla/stringzilla.h> // Primary C API

#include "stringzilla.hpp" // `verify`, `run_test`, `global_random_generator`, `fuzzy_config_t`
#include "utf8.hpp"        // `malformed_classes_`, `utf8_random_segmentation_corpus_`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

using ashvardanian::stringzillas::dummy_executor_t;
using ashvardanian::stringzillas::substrings_cased_k;
using ashvardanian::stringzillas::forkunion_executor_t;
using ashvardanian::stringzillas::substrings_bm25_t;
using ashvardanian::stringzillas::substrings_leftmost_first_k;
using ashvardanian::stringzillas::substrings_leftmost_longest_k;
using ashvardanian::stringzillas::substrings_match_t;
using ashvardanian::stringzillas::substrings_overlap_policy_t;
using ashvardanian::stringzillas::substrings_overlapping_k;
using ashvardanian::stringzillas::substrings_case_sensitivity_t;
using ashvardanian::stringzillas::substrings_parallel_t;
using ashvardanian::stringzillas::substrings_serial_t;
using ashvardanian::stringzillas::substrings_state_width_t;
using ashvardanian::stringzillas::substrings_u16_dictionary_t;
using ashvardanian::stringzillas::substrings_u32_dictionary_t;
using ashvardanian::stringzillas::substrings_uncased_k;

#if SZ_USE_CUDA
using ashvardanian::stringzillas::cuda_executor_t;
using ashvardanian::stringzillas::substrings_cuda_t;
using ashvardanian::stringzillas::gpu_specs_fetch;
using ashvardanian::stringzillas::gpu_specs_t;
#endif

#pragma region Helpers

/** @brief Field-by-field ordering for `finalize`; the public match deliberately carries no comparators. */
inline bool substrings_match_less_(substrings_match_t const &left, substrings_match_t const &right) noexcept {
    if (left.haystack_index != right.haystack_index) return left.haystack_index < right.haystack_index;
    if (left.needle_index != right.needle_index) return left.needle_index < right.needle_index;
    if (left.byte_offset != right.byte_offset) return left.byte_offset < right.byte_offset;
    return left.byte_length < right.byte_length;
}
/** @brief Position-only ordering for the cover's touch check; ignores needle index and length. */
inline bool substrings_match_by_position_less_(substrings_match_t const &left,
                                               substrings_match_t const &right) noexcept {
    if (left.haystack_index != right.haystack_index) return left.haystack_index < right.haystack_index;
    return left.byte_offset < right.byte_offset;
}
inline bool substrings_match_equal_(substrings_match_t const &left, substrings_match_t const &right) noexcept {
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
struct substrings_match_set_t {
    std::vector<substrings_match_t> matches;
    unified_vector<std::size_t> collected_counts;
    unified_vector<substrings_match_t> collected_matches;

    substrings_match_set_t() = default;

    /** @brief Builds an already-ordered expectation from a hand-written table, which can therefore spell its
     *         matches in whatever order reads best. */
    substrings_match_set_t(std::initializer_list<substrings_match_t> initial) {
        for (substrings_match_t const &match : initial) matches.push_back(match);
        finalize();
    }

    void clear() noexcept { matches.clear(); }
    void append(substrings_match_t match) { matches.push_back(match); }
    std::size_t size() const noexcept { return matches.size(); }
    bool empty() const noexcept { return matches.empty(); }
    auto begin() const noexcept { return matches.begin(); }
    auto end() const noexcept { return matches.end(); }

    /** @brief Orders the collected matches and fails on a repeat - the four fields identify a match uniquely,
     *         so a duplicate is always a bug. */
    void finalize() {
        std::sort(matches.begin(), matches.end(), substrings_match_less_);
        bool const has_duplicate = std::adjacent_find(matches.begin(), matches.end(), substrings_match_equal_) !=
                                   matches.end();
        verify(!has_duplicate && "Duplicate (haystack, needle, offset, length) reported twice");
    }

    /** @note Both sides must have been finalized since their last mutation. */
    bool operator==(substrings_match_set_t const &other) const noexcept {
        return matches.size() == other.matches.size() &&
               std::equal(matches.begin(), matches.end(), other.matches.begin(), substrings_match_equal_);
    }
    bool operator!=(substrings_match_set_t const &other) const noexcept { return !(*this == other); }
};

/**
 *  @brief Runs a full count-then-find pass through @p engine under @p overlap_policy and reduces every match
 *         to its identity in @p out, whose own buffers hold the intermediates. Every differential check in
 *         this file is driven through this one public-API shape.
 *  @note `trailing_args_` forwards an executor (and optionally specs) to backends that want a specific one;
 *        an empty pack leaves each engine on its own defaults.
 */
template <typename engine_type_, typename haystacks_type_, typename... trailing_args_>
void collect_matches_under_(engine_type_ &engine, haystacks_type_ const &haystacks,
                            substrings_overlap_policy_t overlap_policy, substrings_match_set_t &out,
                            trailing_args_ &&...trailing_args) {
    out.collected_counts.assign(haystacks.size(), 0);
    std::size_t matches_total = 0;
    span<std::size_t> const counts {out.collected_counts.data(), out.collected_counts.size()};

    verify(engine.try_count(haystacks, overlap_policy, counts, matches_total, trailing_args...) == status_t::success_k);

    // Sized to exactly what counting promised, so a `try_find` overrun trips instead of writing into slack.
    out.collected_matches.assign(matches_total, substrings_match_t {});
    std::size_t matches_found = 0;
    span<substrings_match_t> const matches {out.collected_matches.data(), out.collected_matches.size()};
    verify(engine.try_find(haystacks, overlap_policy, matches, matches_found, trailing_args...) == status_t::success_k);
    verify(matches_found == matches_total && "try_count and try_find disagree on the match count");

    out.clear();
    for (substrings_match_t const &match : out.collected_matches) out.append(match);
    out.finalize();
}

/** @brief `collect_matches_under_` for the overlapping policy, which is what most checks compare against. */
template <typename engine_type_, typename haystacks_type_, typename... trailing_args_>
void collect_overlapping_matches_into_(engine_type_ &engine, haystacks_type_ const &haystacks,
                                       substrings_match_set_t &out, trailing_args_ &&...trailing_args) {
    collect_matches_under_(engine, haystacks, substrings_overlapping_k, out,
                           std::forward<trailing_args_>(trailing_args)...);
}

/** @brief A vocabulary of short random needles/haystack fragments over a ten-letter alphabet, so the
 *         resulting automaton is small enough to reason about by hand yet large enough to grow a real cold
 *         tier. Wraps the shared `randomize_strings` in the by-value shape these fixtures read better with. */
inline std::vector<std::string> random_short_strings_(std::size_t count, std::size_t minimum_length,
                                                      std::size_t maximum_length) {
    std::vector<std::string> result;
    randomize_strings({"abcdefghij", count, minimum_length, maximum_length}, result);
    return result;
}

/** @brief Random haystacks that each carry one of @p needles, so the corpus still reaches the match paths
 *         once `SZ_TESTS_MULTIPLIER` shrinks it - ten letters spell the whole vocabulary, which leaves a
 *         chance hit vanishingly rare at small counts, and a fixture comparing engines over no matches at
 *         all compares nothing. Both rotations advance a phase each turn, so the needles spread across the
 *         haystacks and land inside them, where covers and neighbouring matches mean something, rather than
 *         all at the tail. @p needles must not be empty. */
inline std::vector<std::string> random_haystacks_with_needles_(std::vector<std::string> const &needles,
                                                               std::size_t count, std::size_t minimum_length,
                                                               std::size_t maximum_length) {
    std::vector<std::string> result = random_short_strings_(count, minimum_length, maximum_length);
    for (std::size_t index = 0; index < result.size(); ++index) {
        std::string const &needle = needles[rotating_index(index, needles.size())];
        result[index].insert(rotating_index(index, result[index].size() + 1), needle);
    }
    return result;
}

/** @brief One de-duplicated match span in `independent_uncased_matches_`, keyed by source byte offsets. */
struct uncased_match_span_t {
    std::size_t byte_begin = 0;
    std::size_t byte_end = 0;

    bool operator<(uncased_match_span_t const &other) const {
        return byte_begin != other.byte_begin ? byte_begin < other.byte_begin : byte_end < other.byte_end;
    }
};

/**
 *  @brief Independent case-folded substring oracle: folds haystack and needle codepoint by codepoint via
 *         `sz_unicode_fold_codepoint_`, then slides the folded needle over the folded haystack, reporting
 *         every distinct source span whose folded run matches - overlaps included.
 *
 *  Shares no machinery with the Aho-Corasick engine under test beyond the one folding table every backend
 *  reads. @sa `reference_uncased_find_` in `test/uncased.cpp`, which reports only the first match.
 */
inline std::vector<span<char const>> independent_uncased_matches_(span<char const> haystack, span<char const> needle) {
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
        }
        cursor += consumed;
    }

    // A match is any contiguous run of the folded haystack, including one that begins or ends part-way
    // through an expansion - "s" does match inside the sharp S. Neither end has a byte of its own there, so
    // both snap outward to the codepoint that produced them, and two runs that snap to one span are one
    // match. This is the same rule `sz_utf8_uncased_search` follows.
    std::set<uncased_match_span_t> spans;
    std::vector<span<char const>> matches;
    std::size_t const needle_length = needle_folded.size();
    if (needle_length == 0) return matches;
    for (std::size_t start = 0; start + needle_length <= haystack_folded.size(); ++start) {
        bool equal = true;
        for (std::size_t index = 0; index < needle_length; ++index)
            if (haystack_folded[start + index] != needle_folded[index]) {
                equal = false;
                break;
            }
        if (!equal) continue;
        std::size_t const from = source_begin[start], to = source_end[start + needle_length - 1];
        if (!spans.emplace(uncased_match_span_t {from, to}).second) continue;
        matches.emplace_back(haystack.data() + from, to - from);
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
inline void collect_independent_matches_(substrings_case_sensitivity_t sensitivity, arrow_strings_view_t needles,
                                         arrow_strings_view_t haystacks, substrings_match_set_t &out) {
    out.clear();
    for (std::size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
        span<char const> const haystack = haystacks[haystack_index];
        for (std::size_t needle_index = 0; needle_index < needles.size(); ++needle_index) {
            span<char const> const needle = needles[needle_index];
            // No byte-length pre-filter for uncased matching: a needle longer than the haystack can still
            // match, because folding expands haystack codepoints - the 3-byte U+1FC7 folds to 6 bytes, so
            // 4 of them legitimately carry a 24-byte needle. The cased loop's own bound handles oversize.
            if (needle.size() == 0) continue;

            if (sensitivity == substrings_uncased_k) {
                for (auto const &match : independent_uncased_matches_(haystack, needle))
                    out.append(
                        {haystack_index, needle_index, (std::size_t)(match.data() - haystack.data()), match.size()});
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
enum class substrings_needle_generator_t {
    /** "a", "aa", "aaa", … - every needle a suffix of the next, quadratic outputs. */
    self_overlapping_k,
    /** One needle's suffix is another's prefix - the textbook failure-link case. */
    mutual_overlap_k,
    /** One long common prefix, diverging only at the final byte. */
    shared_prefix_fan_k,
    /** Only bytes 0x01 and 0xFF - worst case for the double-array packer. */
    sparse_wide_alphabet_k,
    /** Wide fold images sampled from the shipped tables - multi-rune expansions under folding. */
    fold_expanding_k,
    /** A fixed core plus deduplicated random needles - the control. */
    random_short_k,
    /** Needles of 1-3 runes sampled from the fold-preimage tables' image side. */
    preimage_sampled_k,
    /** One rune per UTF-8 width class 1-4, so a single needle spans every decode path. */
    mixed_width_k,
    /** One past the last generator, so the sweep can never drift from the list. */
    count_k,
};

/** @brief Selects one byte-agnostic haystack skeleton; the planted bytes come from the transform. */
enum class substrings_placement_t {
    /** Noise, one transformed needle, noise. */
    planted_in_noise_k,
    /** Up to 4 consecutive needles' variants back to back - forces overlapping matches. */
    concatenated_k,
    /** The variant twice back to back, flush with the first byte and flush with the last. */
    boundary_flush_k,
    /** One ~16 KB haystack with variants planted every ~1 KB - pairs with the sliced-specs cells. */
    straddling_k,
    /** The variant interrupted by one byte no fold can bridge - the planting must never match. */
    torn_codepoint_k,
    /** One past the last placement, so the sweep can never drift from the list. */
    count_k,
};

/** @brief Selects how a planted needle is re-spelled, decoupled from where it lands. */
enum class substrings_transform_t {
    /** The needle verbatim - preserved under both modes. */
    identity_k,
    /** Table-driven inverse folding - preserved uncased, defeated cased when any byte changed. */
    fold_preimage_k,
    /** `^0x20` on every ASCII letter - preserved uncased, defeated cased when any letter existed. */
    case_flip_ascii_k,
    /** `^0x01` on the last ASCII byte - defeated under BOTH modes, the true negative control. */
    byte_perturb_k,
    /** The final codepoint dropped - defeated under both modes. */
    truncated_tail_k,
    /** One past the last transform, so the sweep can never drift from the list. */
    count_k,
};

/** @brief What a planting demands of the engine: the exact match record present, or absent. */
enum class substrings_planting_effect_t {
    preserved_k,
    defeated_k,
};

/** @brief One planted variant's ground truth: the exact match record and what the engine owes it. */
struct substrings_planting_t {
    substrings_match_t match;
    substrings_planting_effect_t effect;
};

/** @brief Whether a generator's needles are well-formed UTF-8, and so can be case-folded at all. Only the
 *         double-array pressure vocabulary reaches for `0xFF`, which no UTF-8 sequence ever contains. */
constexpr bool substrings_needles_are_utf8_(substrings_needle_generator_t kind) noexcept {
    return kind != substrings_needle_generator_t::sparse_wide_alphabet_k;
}

/** @brief Whether a generator's automaton grows with the square of its needle count, so the driver can pick
 *         `scale_iterations_quadratic` over `scale_iterations` and keep the suite balanced. */
constexpr bool substrings_needles_grow_quadratically_(substrings_needle_generator_t kind) noexcept {
    return kind == substrings_needle_generator_t::self_overlapping_k;
}

constexpr char const *substrings_needle_generator_name(substrings_needle_generator_t kind) noexcept {
    switch (kind) {
    case substrings_needle_generator_t::self_overlapping_k: return "self_overlapping";
    case substrings_needle_generator_t::mutual_overlap_k: return "mutual_overlap";
    case substrings_needle_generator_t::shared_prefix_fan_k: return "shared_prefix_fan";
    case substrings_needle_generator_t::sparse_wide_alphabet_k: return "sparse_wide_alphabet";
    case substrings_needle_generator_t::fold_expanding_k: return "fold_expanding";
    case substrings_needle_generator_t::random_short_k: return "random_short";
    case substrings_needle_generator_t::preimage_sampled_k: return "preimage_sampled";
    case substrings_needle_generator_t::mixed_width_k: return "mixed_width";
    case substrings_needle_generator_t::count_k: break; // ? Never a real generator, only the sweep's bound
    }
    return "unknown";
}

constexpr char const *substrings_placement_name(substrings_placement_t placement) noexcept {
    switch (placement) {
    case substrings_placement_t::planted_in_noise_k: return "planted_in_noise";
    case substrings_placement_t::concatenated_k: return "concatenated";
    case substrings_placement_t::boundary_flush_k: return "boundary_flush";
    case substrings_placement_t::straddling_k: return "straddling";
    case substrings_placement_t::torn_codepoint_k: return "torn_codepoint";
    case substrings_placement_t::count_k: break; // ? Never a real placement, only the sweep's bound
    }
    return "unknown";
}

constexpr char const *substrings_transform_name(substrings_transform_t transform) noexcept {
    switch (transform) {
    case substrings_transform_t::identity_k: return "identity";
    case substrings_transform_t::fold_preimage_k: return "fold_preimage";
    case substrings_transform_t::case_flip_ascii_k: return "case_flip_ascii";
    case substrings_transform_t::byte_perturb_k: return "byte_perturb";
    case substrings_transform_t::truncated_tail_k: return "truncated_tail";
    case substrings_transform_t::count_k: break; // ? Never a real transform, only the sweep's bound
    }
    return "unknown";
}

/** @brief Sorts and deduplicates a locally built vocabulary pool, then assigns it into the tape. */
inline void assign_deduplicated_(std::vector<std::string> &pool, arrow_strings_tape_t &needles) {
    std::sort(pool.begin(), pool.end());
    pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
    verify(needles.try_assign(pool.data(), pool.data() + pool.size()) == status_t::success_k);
}

/** @brief Appends @p rune to @p out as 1-4 UTF-8 bytes; sampled fold runes must always re-encode. */
inline void append_rune_utf8_(sz_rune_t rune, std::string &out) {
    sz_u8_t encoded[4];
    sz_rune_length_t const encoded_length = sz_rune_encode(rune, encoded);
    verify(encoded_length != sz_rune_invalid_k && "Sampled fold rune must re-encode");
    out.append((char const *)encoded, (std::size_t)encoded_length);
}

/**
 *  @brief Which source codepoints fold onto each image, built once by folding every codepoint forward.
 *
 *  The engine has no use for this direction - it folds needles and haystacks the same way and compares - so
 *  the index is the test's own, derived from `sz_unicode_fold_codepoint_` rather than from a shipped table.
 *  Deriving it here is also what keeps the fixtures honest: they cannot drift from the fold under test.
 */
struct fold_preimage_index_t {
    /** @brief Single-rune images, ascending, so a range of them can be binary-searched. */
    std::vector<sz_rune_t> narrow_images;
    /** @brief Multi-rune images, each zero-padded to three runes. */
    std::vector<std::array<sz_rune_t, 3>> wide_images;
    /** @brief Sources per image, keyed by the image's runes. */
    std::map<std::vector<sz_rune_t>, std::vector<sz_rune_t>> sources_of_image;
};

/** @brief The one index, folded on first use and shared by every generator that needs a preimage. */
inline fold_preimage_index_t const &fold_preimage_index_() {
    static fold_preimage_index_t const index = [] {
        fold_preimage_index_t built;
        for (sz_rune_t rune = 0; rune <= 0x10FFFF; ++rune) {
            if (rune >= 0xD800 && rune <= 0xDFFF) continue; // ? Surrogates are not codepoints on their own
            sz_rune_t images[3];
            std::size_t const runes = sz_unicode_fold_codepoint_(rune, images);
            if (runes == 1 && images[0] == rune) continue; // ? Folds to itself, so it is nobody's preimage
            built.sources_of_image[std::vector<sz_rune_t>(images, images + runes)].push_back(rune);
        }
        for (auto const &[image, sources] : built.sources_of_image) {
            if (image.size() == 1) {
                built.narrow_images.push_back(image[0]);
                continue;
            }
            std::array<sz_rune_t, 3> padded {};
            for (std::size_t index = 0; index < image.size(); ++index) padded[index] = image[index];
            built.wide_images.push_back(padded);
        }
        std::sort(built.narrow_images.begin(), built.narrow_images.end());
        return built;
    }();
    return index;
}

/** @brief Source codepoints whose full fold is exactly @p image, or an empty span when it folds to itself. */
inline span<sz_rune_t const> fold_preimage_of_runes_(span<sz_rune_t const> image) {
    auto const &sources = fold_preimage_index_().sources_of_image;
    auto const found = sources.find(std::vector<sz_rune_t>(image.begin(), image.end()));
    if (found == sources.end()) return {};
    return {found->second.data(), found->second.size()};
}

/** @brief Source codepoints whose full fold is the single rune @p image. */
inline span<sz_rune_t const> fold_preimage_of_rune_(sz_rune_t image) { return fold_preimage_of_runes_({&image, 1}); }

/** @brief One fold-space rune drawn uniformly from the narrow preimage table's images within
 *         `[first_rune, last_rune)`; the image array is sorted, so the range is a binary-searched slab. */
inline sz_rune_t sample_fold_image_rune_(sz_rune_t first_rune, sz_rune_t last_rune) {
    std::vector<sz_rune_t> const &images = fold_preimage_index_().narrow_images;
    auto const slab_begin = std::lower_bound(images.begin(), images.end(), first_rune);
    auto const slab_end = std::lower_bound(images.begin(), images.end(), last_rune);
    verify(slab_begin != slab_end && "The requested rune range holds no fold images");
    std::uniform_int_distribution<std::size_t> slab_position(0, (std::size_t)(slab_end - slab_begin) - 1);
    std::size_t const chosen_position = slab_position(global_random_generator());
    return slab_begin[chosen_position];
}

/**
 *  @brief Appends @p needle re-encoded through fold space with every image swapped for one genuine
 *         preimage - wide multi-rune images first - so only case folding can reconcile the result.
 *         Returns the number of swapped images; runes without preimages pass through verbatim.
 */
inline std::size_t append_fold_preimage_variant_(span<char const> needle, std::string &out) {
    std::vector<sz_rune_t> folded;
    for (char const *cursor = needle.begin(), *end = needle.end(); cursor != end;) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode(cursor, end, &rune);
        verify(consumed != sz_rune_invalid_k && "Fold-preimage variants need well-formed UTF-8 needles");
        sz_rune_t folded_runes[3];
        std::size_t const folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (std::size_t index = 0; index < folded_count; ++index) folded.push_back(folded_runes[index]);
        cursor += consumed;
    }

    auto &generator = global_random_generator();
    std::size_t swapped_images = 0;
    for (std::size_t position = 0; position < folded.size();) {
        // Wide images first, longest window first, so multi-rune expansions like the sharp S appear.
        std::size_t const remaining_runes = folded.size() - position;
        span<sz_rune_t const> preimages;
        std::size_t consumed_runes = 1;
        if (remaining_runes >= 3) //
            preimages = fold_preimage_of_runes_({folded.data() + position, 3}), consumed_runes = 3;
        if (preimages.size() == 0 && remaining_runes >= 2)
            preimages = fold_preimage_of_runes_({folded.data() + position, 2}), consumed_runes = 2;
        if (preimages.size() == 0) //
            preimages = fold_preimage_of_rune_(folded[position]), consumed_runes = 1;
        if (preimages.size() == 0) {
            append_rune_utf8_(folded[position], out);
            position += 1;
            continue;
        }
        std::uniform_int_distribution<std::size_t> preimage_position(0, preimages.size() - 1);
        std::size_t const chosen_position = preimage_position(generator);
        append_rune_utf8_(preimages[chosen_position], out);
        ++swapped_images;
        position += consumed_runes;
    }
    return swapped_images;
}

/** @brief Perturbs the last ASCII byte with `^0x01`, appending the variant; a needle without a single
 *         ASCII byte is left un-appended. @return whether the perturbation applied. */
inline bool try_perturb_last_ascii_byte_(span<char const> needle, std::string &out) {
    std::size_t ascii_position = needle.size();
    for (std::size_t index = needle.size(); index > 0; --index)
        if ((unsigned char)needle[index - 1] < 0x80) {
            ascii_position = index - 1;
            break;
        }
    if (ascii_position == needle.size()) return false;
    std::size_t const variant_begin = out.size();
    out.append(needle.data(), needle.size());
    out[variant_begin + ascii_position] = (char)(out[variant_begin + ascii_position] ^ 0x01);
    return true;
}

/** @brief Drops the needle's final codepoint by walking back over continuation bytes - on a non-UTF-8
 *         vocabulary the walk stops immediately, dropping one byte; a single-codepoint needle refuses. */
inline bool try_truncate_needle_tail_(span<char const> needle, std::string &out) {
    if (needle.size() < 2) return false;
    std::size_t truncated_size = needle.size() - 1;
    while (truncated_size > 0 && ((unsigned char)needle[truncated_size] & 0xC0) == 0x80) --truncated_size;
    if (truncated_size == 0) return false;
    out.append(needle.data(), truncated_size);
    return true;
}

/**
 *  @brief Appends one transform's variant of @p needle to @p out, answering what the planting demands of
 *         the engine under @p sensitivity. Transforms that cannot apply fall through a terminating ladder:
 *         a needle with no ASCII byte sends the perturbation to tail truncation, a single-codepoint needle
 *         sends truncation to the perturbation, and the identity ends every path.
 */
inline substrings_planting_effect_t append_transformed_needle_( //
    substrings_transform_t transform, substrings_case_sensitivity_t sensitivity,
    substrings_needle_generator_t needle_kind, span<char const> needle, std::string &out) {

    if (transform == substrings_transform_t::fold_preimage_k && !substrings_needles_are_utf8_(needle_kind))
        transform = substrings_transform_t::case_flip_ascii_k; // ? Folding needs decodable needles.

    std::size_t const variant_begin = out.size();
    switch (transform) {
    case substrings_transform_t::identity_k: break;

    case substrings_transform_t::fold_preimage_k: {
        [[maybe_unused]] std::size_t const swapped_images = append_fold_preimage_variant_(needle, out);
        // A swap may reproduce the original spelling - the sharp S is its own preimage - so the cased
        // verdict compares bytes, not swap counts.
        bool const changed = out.size() - variant_begin != needle.size() ||
                             std::memcmp(out.data() + variant_begin, needle.data(), needle.size()) != 0;
        if (sensitivity == substrings_uncased_k || !changed) return substrings_planting_effect_t::preserved_k;
        return substrings_planting_effect_t::defeated_k;
    }

    case substrings_transform_t::case_flip_ascii_k: {
        bool any_letter_flipped = false;
        for (std::size_t index = 0; index < needle.size(); ++index) {
            char const letter = needle[index];
            bool const is_ascii_letter = (letter >= 'a' && letter <= 'z') || (letter >= 'A' && letter <= 'Z');
            any_letter_flipped |= is_ascii_letter;
            out.push_back(is_ascii_letter ? (char)(letter ^ 0x20) : letter);
        }
        if (sensitivity == substrings_uncased_k || !any_letter_flipped)
            return substrings_planting_effect_t::preserved_k;
        return substrings_planting_effect_t::defeated_k;
    }

    case substrings_transform_t::byte_perturb_k: {
        if (try_perturb_last_ascii_byte_(needle, out)) return substrings_planting_effect_t::defeated_k;
        if (try_truncate_needle_tail_(needle, out)) return substrings_planting_effect_t::defeated_k;
        break;
    }

    case substrings_transform_t::truncated_tail_k: {
        if (try_truncate_needle_tail_(needle, out)) return substrings_planting_effect_t::defeated_k;
        if (try_perturb_last_ascii_byte_(needle, out)) return substrings_planting_effect_t::defeated_k;
        break;
    }

    case substrings_transform_t::count_k: break; // ? Never a real transform, only the sweep's bound
    }
    out.append(needle.data(), needle.size());
    return substrings_planting_effect_t::preserved_k;
}

/**
 *  @brief Fills @p needles with one closed-set adversarial vocabulary, @p count deep where the kind scales.
 *
 *  Every case either appends straight into the tape or, for the one kind needing a whole-collection sort and
 *  deduplication, fills a local pool first.
 */
inline void generate_substrings_needles_(substrings_needle_generator_t kind, std::size_t count,
                                         arrow_strings_tape_t &needles) {
    needles.reset();
    std::string scratch;
    switch (kind) {

    case substrings_needle_generator_t::self_overlapping_k:
        // Every needle is a suffix of the next, so the deepest state carries `count` merged outputs and the
        // output pool grows with the square of the depth - the worst case for anything sizing that pool.
        for (std::size_t depth = 1; depth <= count; ++depth) {
            scratch.push_back('a');
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;

    case substrings_needle_generator_t::mutual_overlap_k:
        // One needle's suffix is another's prefix, so failure links traverse rather than collapse to the root.
        for (std::size_t index = 0; index < count; ++index) {
            char const first = (char)('a' + index % 6), second = (char)('a' + (index + 1) % 6);
            scratch.assign({first, second, first});
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
            scratch.assign({second, first});
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;

    case substrings_needle_generator_t::shared_prefix_fan_k:
        // A single deep trunk with `count` leaves, so almost every state has out-degree one and the
        // frequency ordering has nothing to separate.
        for (std::size_t index = 0; index < count; ++index) {
            scratch.assign(12, 'q');
            scratch.push_back((char)('a' + index % 26));
            scratch.push_back((char)('a' + (index / 26) % 26));
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;

    case substrings_needle_generator_t::sparse_wide_alphabet_k:
        // Two bytes at opposite ends of the alphabet, so every double-array base must reserve a 256-wide
        // window to hold two edges - maximum collision pressure for the packer.
        for (std::size_t pattern = 0; pattern < count; ++pattern) {
            scratch.clear();
            for (std::size_t depth = 0; depth < 8; ++depth) scratch.push_back((pattern >> depth) & 1 ? '\xFF' : '\x01');
            verify(needles.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        }
        return;

    case substrings_needle_generator_t::fold_expanding_k: {
        // Needles whose fold preimages span more bytes than the needle itself - "k" also matches the 3-byte
        // Kelvin sign, so `max_source_match_bytes` cannot be read off needle lengths. The Kelvin anchor stays
        // pinned; every following needle is a wide image repeated 1-4 times, so 2:1 and 3:1 byte expansions
        // from every script appear organically rather than from a hand-picked seed list.
        std::vector<std::string> pool {"k"};
        auto &generator = global_random_generator();
        std::vector<std::array<sz_rune_t, 3>> const &wide_images = fold_preimage_index_().wide_images;
        std::uniform_int_distribution<std::size_t> row_position(0, wide_images.size() - 1);
        for (std::size_t index = 0; index < count; ++index) {
            std::array<sz_rune_t, 3> const &image = wide_images[row_position(generator)];
            scratch.clear();
            for (std::size_t repeat = 0; repeat <= index % 4; ++repeat)
                for (std::size_t rune_index = 0; rune_index < 3 && image[rune_index] != 0; ++rune_index)
                    append_rune_utf8_(image[rune_index], scratch);
            pool.push_back(scratch);
        }
        assign_deduplicated_(pool, needles);
        return;
    }

    case substrings_needle_generator_t::random_short_k: {
        // A fixed textbook core plus random needles, deduplicated as one pool.
        std::vector<std::string> pool {"he", "she", "his", "hers"};
        for (std::string &value : random_short_strings_(count, 3, 6)) pool.push_back(std::move(value));
        assign_deduplicated_(pool, needles);
        return;
    }

    case substrings_needle_generator_t::preimage_sampled_k: {
        // 1-3 fold-space runes per needle, sampled from the narrow table's image side, so Greek, Cyrillic,
        // Cherokee, and astral fold pairs appear by construction rather than from hand-picked literals.
        std::vector<std::string> pool;
        auto &generator = global_random_generator();
        std::uniform_int_distribution<std::size_t> rune_count_distribution(1, 3);
        for (std::size_t index = 0; index < count; ++index) {
            scratch.clear();
            std::size_t const rune_count = rune_count_distribution(generator);
            for (std::size_t rune_index = 0; rune_index < rune_count; ++rune_index)
                append_rune_utf8_(sample_fold_image_rune_(0, 0x110000), scratch);
            pool.push_back(scratch);
        }
        assign_deduplicated_(pool, needles);
        return;
    }

    case substrings_needle_generator_t::mixed_width_k: {
        // One rune per UTF-8 width class - ASCII, 2-byte, 3-byte, 4-byte - so every needle drags the walk
        // through every decode width, and match spans never equal codepoint counts.
        std::vector<std::string> pool;
        auto &generator = global_random_generator();
        std::uniform_int_distribution<int> ascii_letter('a', 'z');
        for (std::size_t index = 0; index < count; ++index) {
            scratch.clear();
            scratch.push_back((char)ascii_letter(generator));
            append_rune_utf8_(sample_fold_image_rune_(0x80, 0x800), scratch);
            append_rune_utf8_(sample_fold_image_rune_(0x800, 0x10000), scratch);
            append_rune_utf8_(sample_fold_image_rune_(0x10000, 0x110000), scratch);
            pool.push_back(scratch);
        }
        assign_deduplicated_(pool, needles);
        return;
    }

    case substrings_needle_generator_t::count_k: break; // ? Never a real generator, only the sweep's bound
    }
}

/**
 *  @brief Fills @p haystacks with @p haystack_count skeletons of @p placement, planting each needle's
 *         @p transform variant and recording every planting's ground truth into @p plantings.
 */
inline void generate_substrings_placements_(substrings_placement_t placement, substrings_transform_t transform,
                                            substrings_case_sensitivity_t sensitivity,
                                            substrings_needle_generator_t needle_kind, arrow_strings_view_t needles,
                                            std::size_t haystack_count, arrow_strings_tape_t &haystacks,
                                            std::vector<substrings_planting_t> &plantings) {
    haystacks.reset();
    plantings.clear();
    if (needles.size() == 0) return;

    auto &generator = global_random_generator();
    std::uniform_int_distribution<int> noise_length(4, 24);
    std::uniform_int_distribution<int> noise_byte('m', 'z'); // ? Disjoint from every needle alphabet above.
    // Cased cells sprinkle malformed UTF-8 into the noise - lone continuation bytes - since the byte-exact
    // oracle is total over arbitrary bytes, while the uncased oracle demands well-formed haystacks.
    std::uniform_int_distribution<int> malformed_byte(0x80, 0xBF);
    std::uniform_int_distribution<int> malformed_gate(0, 7);
    std::string scratch;

    auto append_noise = [&] {
        for (int index = 0, length = noise_length(generator); index < length; ++index)
            if (sensitivity == substrings_cased_k && malformed_gate(generator) == 0)
                scratch.push_back((char)malformed_byte(generator));
            else scratch.push_back((char)noise_byte(generator));
    };
    auto plant = [&](std::size_t haystack_index, std::size_t needle_index) {
        std::size_t const offset = scratch.size();
        substrings_planting_effect_t const effect = append_transformed_needle_(transform, sensitivity, needle_kind,
                                                                               needles[needle_index], scratch);
        plantings.push_back({{haystack_index, needle_index, offset, scratch.size() - offset}, effect});
    };
    auto tear_last_planting = [&] {
        // One byte no fold can bridge - '0' appears in no vocabulary and in no letter's fold - lands inside
        // the variant, so the planting must never match. The uncased oracle demands well-formed haystacks,
        // so there the tear retreats to a codepoint boundary; single-codepoint variants stay whole.
        substrings_planting_t &planting = plantings.back();
        if (planting.match.byte_length < 2) return;
        std::size_t tear_offset = planting.match.byte_offset + planting.match.byte_length / 2;
        if (sensitivity == substrings_uncased_k)
            while (tear_offset > planting.match.byte_offset && ((unsigned char)scratch[tear_offset] & 0xC0) == 0x80)
                --tear_offset;
        if (tear_offset == planting.match.byte_offset) return;
        scratch.insert(scratch.begin() + (std::ptrdiff_t)tear_offset, '0');
        planting.match.byte_length += 1;
        planting.effect = substrings_planting_effect_t::defeated_k;
    };
    auto append_haystack = [&] {
        verify(haystacks.try_append({scratch.data(), scratch.size()}) == status_t::success_k);
        scratch.clear();
    };

    switch (placement) {

    case substrings_placement_t::planted_in_noise_k:
        for (std::size_t haystack_index = 0; haystack_index < haystack_count; ++haystack_index) {
            append_noise();
            plant(haystack_index, haystack_index % needles.size());
            append_noise();
            append_haystack();
        }
        return;

    case substrings_placement_t::concatenated_k:
        // No separator, so matches of different needles' variants overlap and nest at the seams.
        for (std::size_t haystack_index = 0; haystack_index < haystack_count; ++haystack_index) {
            for (std::size_t offset = 0; offset < 4 && offset < needles.size(); ++offset)
                plant(haystack_index, (haystack_index + offset) % needles.size());
            append_haystack();
        }
        return;

    case substrings_placement_t::boundary_flush_k:
        // Flush with the first byte and flush with the last, plus a self-repeat in between, so a match
        // lands on every edge a chunked or windowed walk could mishandle.
        for (std::size_t haystack_index = 0; haystack_index < haystack_count; ++haystack_index) {
            plant(haystack_index, haystack_index % needles.size());
            plant(haystack_index, haystack_index % needles.size());
            append_haystack();
        }
        return;

    case substrings_placement_t::straddling_k: {
        // One large haystack for the all-cores-on-one-haystack path: variants planted at a fixed stride,
        // so under the sliced specs the matches keep crossing slice seams.
        std::size_t const haystack_bytes = 16 * 1024;
        std::size_t const stride_bytes = 1024;
        std::size_t planted_count = 0;
        while (scratch.size() < haystack_bytes) {
            std::size_t const next_plant_offset = (std::min)(scratch.size() + stride_bytes, haystack_bytes);
            while (scratch.size() < next_plant_offset) append_noise();
            plant(0, planted_count % needles.size());
            ++planted_count;
        }
        append_haystack();
        return;
    }

    case substrings_placement_t::torn_codepoint_k:
        for (std::size_t haystack_index = 0; haystack_index < haystack_count; ++haystack_index) {
            append_noise();
            plant(haystack_index, haystack_index % needles.size());
            tear_last_planting();
            append_noise();
            append_haystack();
        }
        return;

    case substrings_placement_t::count_k: break; // ? Never a real placement, only the sweep's bound
    }
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer vectors for `substrings`, pinned by hand rather than against a second backend.
 *
 *  Covers the textbook Aho-Corasick overlap example - "he" and "she" both completing at the same end offset
 *  over "ushers" - attribution of matches to the right haystack in a batch, and a self-overlapping
 *  vocabulary of {"a", "ab", "abc"} over "abcabc". Every match is reported, nested ones included; this is
 *  not leftmost-first matching.
 */
void test_substrings_unit() {
    std::printf("  - testing substrings known-answer vectors...\n");

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

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_overlapping_matches_into_(engine, haystacks.view(), matches);

        substrings_match_set_t const expected {
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

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_overlapping_matches_into_(engine, haystacks.view(), matches);

        substrings_match_set_t const expected {
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

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_overlapping_matches_into_(engine, haystacks.view(), matches);

        substrings_match_set_t const expected {
            {0, 0, 0, 1}, {0, 1, 0, 2}, {0, 2, 0, 3}, // "a", "ab", "abc" at the first occurrence
            {0, 0, 3, 1}, {0, 1, 3, 2}, {0, 2, 3, 3}, // and again at the second
        };
        verify(matches == expected && "Nested-prefix overlap example mismatched");
    }

    // A match may begin part-way through an expansion, and then reports the whole codepoint it began inside.
    // Folding "\xC3\x9Fsoft" gives "sssoft", so "ss" matches both the sharp S alone and the run crossing from
    // its second "s" into the literal one - the second snapping outward to byte 0 and running to byte 3.
    {
        std::vector<std::string> const needle_strings {"ss", "\xC3\x9F"};
        std::vector<std::string> const haystack_strings {"\xC3\x9Fsoft"};
        arrow_strings_tape_t needles, haystacks;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_uncased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_overlapping_matches_into_(engine, haystacks.view(), matches);

        substrings_match_set_t const expected {
            {0, 0, 0, 2}, // "ss" matches the sharp-S codepoint at byte [0, 2)
            {0, 1, 0, 2}, // "\xC3\x9F" matches itself at byte [0, 2)
            {0, 0, 0, 3}, // "ss" again, from the sharp S's second "s" into the literal one
            {0, 1, 0, 3}, // and "\xC3\x9F" folds to the same "ss", so it matches that run too
        };
        verify(matches == expected && "Mid-expansion matches must snap outward to the codepoint they begin in");
    }

    // Needles whose fold self-overlaps across a mixed-width fold image: the folded stream lines up in more
    // ways than the source bytes do, so each case pins both which windows match and which source span each
    // one snaps to. `U+212A` is the Kelvin sign (folds to `k`), `U+017F` the long s (folds to `s`).
    struct uncased_case_t {
        std::vector<std::string> needles;
        std::string haystack;
        substrings_match_set_t expected;
        char const *label;
    };
    uncased_case_t const uncased_cases[] = {
        // The ASCII runs join their escapes directly, which is safe only because `k` and `s` are not hex
        // digits. A case whose codepoint is followed by one of `a`-`f` must break the literal, or `\x9F`
        // before an `a` would parse as the single overlong escape `\x9Fa`.
        {{"kk"}, "\xE2\x84\xAAkk", {{0, 0, 0, 4}, {0, 0, 3, 2}}, "kk over Kelvin-k-k"},
        {{"ss"}, "\xC5\xBFss", {{0, 0, 0, 3}, {0, 0, 2, 2}}, "ss over long-s-s"},
        // Both windows of four folded s-runes. Over "ss\xC3\x9Fs" each one starts on a codepoint; over
        // "\xC3\x9Fsss" the second starts on the sharp S's *second* "s", so it snaps back to byte 0 and runs
        // to the end.
        {{"ssss"}, "ss\xC3\x9Fs", {{0, 0, 0, 4}, {0, 0, 1, 4}}, "ssss over s-s-sharp-s"},
        {{"ssss"}, "\xC3\x9Fsss", {{0, 0, 0, 4}, {0, 0, 0, 5}}, "ssss over sharp-s-s-s"},
        // Nested needles: a wrong failure chain on one needle corrupts the other's reported matches too.
        {{"k", "kk"}, "\xE2\x84\xAAk", {{0, 0, 0, 3}, {0, 1, 0, 4}, {0, 0, 3, 1}}, "nested k/kk over Kelvin-k"},
    };
    for (uncased_case_t const &probe : uncased_cases) {
        arrow_strings_tape_t needles, haystacks;
        std::vector<std::string> const haystack_strings {probe.haystack};
        verify(needles.try_assign(probe.needles.data(), probe.needles.data() + probe.needles.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + 1) == status_t::success_k);

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_uncased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_overlapping_matches_into_(engine, haystacks.view(), matches);
        if (matches != probe.expected) std::fprintf(stderr, "Reconvergence hazard: %s\n", probe.label);
        verify(matches == probe.expected && "Reconvergence hazard produced the wrong match set");
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

    substrings_serial_t engine;
    verify(engine.try_index(needles, substrings_uncased_k) == status_t::success_k);

    for (std::string const &haystack : must_match) {
        std::vector<span<char const>> const haystacks {span<char const>(haystack.data(), haystack.size())};
        std::vector<std::size_t> counts(1, 0);
        std::size_t matches_total = 0;
        verify(engine.try_count(haystacks, substrings_overlapping_k, span<std::size_t>(counts.data(), counts.size()),
                                matches_total) == status_t::success_k);
        verify(counts[0] >= 1 && "Needle must match this haystack under full case folding");
    }
    for (std::string const &haystack : must_not_match) {
        std::vector<span<char const>> const haystacks {span<char const>(haystack.data(), haystack.size())};
        std::vector<std::size_t> counts(1, 0);
        std::size_t matches_total = 0;
        verify(engine.try_count(haystacks, substrings_overlapping_k, span<std::size_t>(counts.data(), counts.size()),
                                matches_total) == status_t::success_k);
        verify(counts[0] == 0 && "Needle must not match this haystack");
    }
}

/**
 *  @brief Full Unicode case-folding conformance table for `substrings_uncased_k`, then a differential against
 *         `independent_uncased_matches_`.
 *
 *  Every non-ASCII fixture is spelled as `\xHH` byte escapes, never as a raw literal or a `\u` universal
 *  character name - both have been silently re-encoded by tooling here before. A byte-level check against a
 *  numeric expected-bytes array guards the trickiest fixtures.
 */
void test_substrings_uncased_unit() {
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
}

#pragma endregion // Uncased Conformance

#pragma region Agreement

/**
 *  @brief A ONE-needle dictionary must agree exactly with the shipped `sz_utf8_uncased_search` over a fuzz
 *         corpus - that agreement is the entire semantic claim of `substrings_uncased_k`.
 *
 *  `substrings` reports every overlapping match, while `sz_utf8_uncased_search` reports only the first by
 *  start offset, so the comparison reduces `substrings`'s set to its own earliest-starting match.
 */
void test_substrings_uncased_equivalence() {
    std::printf("  - testing one-needle agreement with sz_utf8_uncased_search...\n");

    // Differential: for a spread of needles and randomized valid UTF-8 haystacks, the full match set found by
    // `substrings` must equal the full match set found by the independent fold-and-scan oracle.
    {
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
            substrings_serial_t engine;
            verify(engine.try_index(needles, substrings_uncased_k) == status_t::success_k);
            std::vector<std::string> const haystack_strings {haystack};
            arrow_strings_tape_t haystacks;
            verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
                   status_t::success_k);
            substrings_match_set_t engine_matches;
            collect_overlapping_matches_into_(engine, haystacks.view(), engine_matches);

            span<char const> const haystack_view = haystacks[0];
            auto const oracle_matches = independent_uncased_matches_(haystack_view, needles[0]);
            substrings_match_set_t expected_matches;
            for (auto const &oracle_match : oracle_matches)
                expected_matches.append(
                    {0, 0, (std::size_t)(oracle_match.data() - haystack_view.data()), oracle_match.size()});
            expected_matches.finalize();

            if (engine_matches != expected_matches) {
                std::fprintf(stderr, "Uncased differential mismatch for needle \"%s\": %zu vs %zu matches\n",
                             needle.c_str(), engine_matches.size(), expected_matches.size());
                for (substrings_match_t const &key : engine_matches)
                    std::fprintf(stderr, "  engine offset=%zu length=%zu\n", key.byte_offset, key.byte_length);
                for (substrings_match_t const &key : expected_matches)
                    std::fprintf(stderr, "  oracle offset=%zu length=%zu\n", key.byte_offset, key.byte_length);
                verify(false && "substrings disagrees with the independent fold-and-scan oracle");
            }
        }
    }

    std::vector<std::string> needle_pool {
        "the",
        "quick",
        "STRASSE",
        "stra\xC3\x9F" "e", // ? Split so "\x9Fe" cannot parse as one escape.
        "ss",
        "K",
        "caf\xC3\xA9",
        "\xC3\x85ngstrom",
    };
    // Beyond the pinned anchors, needles sampled from the fold tables, so the agreement claim covers every
    // script with fold pairs rather than the hand-picked list: preimage re-spellings of each anchor, then
    // fold-space runes drawn straight from the narrow table's image side.
    for (std::size_t anchor_index = 0, anchors = needle_pool.size(); anchor_index < anchors; ++anchor_index) {
        std::string variant;
        span<char const> const anchor {needle_pool[anchor_index].data(), needle_pool[anchor_index].size()};
        [[maybe_unused]] std::size_t const swapped_images = append_fold_preimage_variant_(anchor, variant);
        needle_pool.push_back(std::move(variant));
    }
    for (std::size_t sample_index = 0; sample_index < 8; ++sample_index) {
        std::string sampled;
        append_rune_utf8_(sample_fold_image_rune_(0, 0x110000), sampled);
        append_rune_utf8_(sample_fold_image_rune_(0, 0x110000), sampled);
        needle_pool.push_back(std::move(sampled));
    }
    span<string_view const> const empty_motifs;
    auto &generator = global_random_generator();

    for (std::size_t iteration = 0; iteration < scale_iterations(200); ++iteration) {
        std::string const &needle = needle_pool[iteration % needle_pool.size()];

        std::string haystack;
        utf8_random_segmentation_corpus_(haystack, 80, utf8_corpus_flavor_t::valid_k, utf8_default_alphabet,
                                         empty_motifs, generator);
        if ((iteration & 1) == 0) haystack.append(needle); // ? Half the iterations guarantee a hit.

        std::vector<span<char const>> const needles {span<char const>(needle.data(), needle.size())};
        substrings_serial_t engine;
        verify(engine.try_index(needles, substrings_uncased_k) == status_t::success_k);
        std::vector<std::string> const haystack_strings {haystack};
        arrow_strings_tape_t haystacks;
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        substrings_match_set_t matches;
        collect_overlapping_matches_into_(engine, haystacks.view(), matches);

        bool engine_found = false;
        std::size_t engine_offset = 0, engine_length = 0;
        for (substrings_match_t const &match : matches)
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
            std::fprintf(                                                                            //
                stderr,                                                                              //
                "Agreement mismatch for needle \"%s\": substrings found=%d offset=%zu length=%zu | " //
                "sz_utf8_uncased_search found=%d offset=%zu length=%zu\n",                           //
                needle.c_str(), engine_found, engine_offset, engine_length, reference_found, reference_offset,
                reference_length);
            verify(false && "substrings and sz_utf8_uncased_search disagree");
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
struct substrings_adversarial_scratch_t {
    /** @brief Which axis triple produced the current cell, so a failing check can name its configuration
     *         instead of leaving it to be reconstructed from a step index. */
    substrings_needle_generator_t needle_kind = substrings_needle_generator_t::self_overlapping_k;
    substrings_placement_t placement = substrings_placement_t::planted_in_noise_k;
    substrings_transform_t transform = substrings_transform_t::identity_k;

    /** @brief Default on most cells; every third cell shrinks `l2_bytes` so straddling haystacks slice. */
    cpu_specs_t specs;

    arrow_strings_tape_t needles;
    arrow_strings_tape_t haystacks;
    std::vector<substrings_planting_t> plantings;
    substrings_match_set_t engine_keys, oracle_keys, variant_keys;
#if SZ_USE_CUDA
    substrings_match_set_t cuda_keys;
#endif
};

/** @brief Reports @p what went wrong and which axis triple produced the cell, mirroring how
 *         `edit_distance_log_mismatch` prefixes a similarity failure with the pair that caused it. */
void log_substrings_cell_mismatch_(substrings_adversarial_scratch_t const &scratch, char const *what) {
    std::fprintf(stderr, "%s on needles=%s placement=%s transform=%s\n", what,
                 substrings_needle_generator_name(scratch.needle_kind), substrings_placement_name(scratch.placement),
                 substrings_transform_name(scratch.transform));
}

/**
 *  @brief The indexing check: every match the serial engine reports must agree with brute force, needle,
 *         offset, length and haystack all at once. Runs on every cell.
 */
void check_substrings_against_oracle_(substrings_case_sensitivity_t sensitivity,
                                      substrings_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    substrings_serial_t engine;
    verify(engine.try_index(needles_view, sensitivity) == status_t::success_k);
    collect_overlapping_matches_into_(engine, haystacks_view, scratch.engine_keys);
    collect_independent_matches_(sensitivity, needles_view, haystacks_view, scratch.oracle_keys);

    if (scratch.engine_keys != scratch.oracle_keys) {
        log_substrings_cell_mismatch_(scratch, "Oracle disagreement");
        verify(false && "substrings disagrees with the brute-force vocabulary oracle");
    }
}

/**
 *  @brief Construction-time truth: every preserved planting must appear among the engine's keys as its
 *         exact record, every defeated one must not. A third witness beside the oracle - the generator
 *         knows where it planted what, so a misconception the engine and oracle share cannot survive it.
 *         Runs on every cell, right after the oracle check fills `engine_keys`.
 */
void check_substrings_declared_effects_(substrings_adversarial_scratch_t &scratch) {
    for (substrings_planting_t const &planting : scratch.plantings) {
        bool const present = std::binary_search(scratch.engine_keys.matches.begin(), scratch.engine_keys.matches.end(),
                                                planting.match, substrings_match_less_);
        bool const expected = planting.effect == substrings_planting_effect_t::preserved_k;
        if (present == expected) continue;
        log_substrings_cell_mismatch_(scratch, expected ? "Preserved planting missing" : "Defeated planting matched");
        std::fprintf(stderr, "  haystack=%zu needle=%zu offset=%zu length=%zu\n", planting.match.haystack_index,
                     planting.match.needle_index, planting.match.byte_offset, planting.match.byte_length);
        verify(false && "The engine's matches contradict a planting's declared effect");
    }
}

/** @brief Serial and fork-union-parallel must report the identical set on the same cell; the parallel
 *         engine takes the real pool and the cell's specs, so sliced-L2 cells genuinely slice. */
void check_substrings_backends_agree_(substrings_case_sensitivity_t sensitivity,
                                      substrings_adversarial_scratch_t &scratch, forkunion_executor_t &pool) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    substrings_serial_t serial_engine;
    verify(serial_engine.try_index(needles_view, sensitivity) == status_t::success_k);
    collect_overlapping_matches_into_(serial_engine, haystacks_view, scratch.engine_keys);

    substrings_parallel_t parallel_engine;
    verify(parallel_engine.try_index(needles_view, sensitivity) == status_t::success_k);
    collect_overlapping_matches_into_(parallel_engine, haystacks_view, scratch.variant_keys, pool, scratch.specs);
    if (scratch.engine_keys != scratch.variant_keys) {
        log_substrings_cell_mismatch_(scratch, "Serial-vs-parallel divergence");
        verify(false && "Serial and parallel backends disagree");
    }
}

/**
 *  @brief A `u16` automaton over the same cell either agrees exactly or declines with `overflow_risk_k`.
 *
 *  Declining is a legitimate outcome - the narrower id space caps at 65534 states, and an adversarial
 *  vocabulary is built to reach ceilings - so the property is that it never silently truncates instead.
 *  Reports whether it got as far as comparing, so the sweep can refuse to pass on declines alone.
 */
bool check_substrings_narrow_width_(substrings_case_sensitivity_t sensitivity,
                                    substrings_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    substrings_u16_dictionary_t narrow;
    narrow.case_sensitivity(sensitivity);
    for (std::size_t index = 0; index < needles_view.size(); ++index) {
        status_t const status = narrow.try_insert(needles_view[index]);
        if (status == status_t::overflow_risk_k) return false;
        verify(status == status_t::success_k);
    }
    status_t const build_status = narrow.try_build();
    if (build_status == status_t::overflow_risk_k) return false;
    verify(build_status == status_t::success_k);

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
        log_substrings_cell_mismatch_(scratch, "Narrow-width divergence");
        verify(false && "u16 automaton disagrees with the oracle");
    }
    return true;
}

/**
 *  @brief The same dictionary with `hot_count` forced from fully cold to fully hot must report an identical
 *         set every time.
 *
 *  A small dictionary is entirely hot by default and never enters the cold double array at all, so a
 *  tier-boundary bug is otherwise invisible. Driven through `aho_corasick_dictionary::hot_count` directly,
 *  the engine wrapper having no hook to override the split before `try_build`.
 */
void check_substrings_tier_invariant_(substrings_case_sensitivity_t sensitivity,
                                      substrings_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    std::size_t raw_state_count = 0;
    {
        substrings_u32_dictionary_t probe;
        probe.case_sensitivity(sensitivity);
        for (std::size_t index = 0; index < needles_view.size(); ++index)
            verify(probe.try_insert(needles_view[index]) == status_t::success_k);
        // Build it: uncased reconvergence splits states during the build, so the published state count -
        // the true "all states hot" target for the sweep below - is only known after `try_build`.
        verify(probe.try_build() == status_t::success_k);
        raw_state_count = probe.count_states();
    }
    verify(raw_state_count > 0);

    std::size_t const hot_counts[] = {0, 1, raw_state_count / 2, raw_state_count};
    bool saw_cold_tier = false, saw_all_hot = false;
    for (std::size_t variant_index = 0; variant_index < 4; ++variant_index) {
        substrings_u32_dictionary_t dictionary;
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
            log_substrings_cell_mismatch_(scratch, "Tier mismatch");
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
/**
 *  @brief Reaches the scalar rune helpers from device code, so a helper that stops being reachable stops
 *         this file from compiling.
 *
 *  Uncased matching walks a folded cursor and that cursor decodes a rune per step. The helpers it calls
 *  name no execution space of their own: `SZ_HELPER_AUTO` marks them `constexpr`, and
 *  `--expt-relaxed-constexpr` is what lets a kernel reach a host `constexpr` function at all. Downgrade one
 *  to `SZ_HELPER_INLINE` and it becomes host-only, yet `nvcc` still resolves the call from inside the
 *  cursor without a diagnostic - every multi-byte codepoint then decodes as a run of malformed single
 *  bytes, so an uncased search matches nothing outside ASCII on the device while every host backend agrees
 *  with the oracle and the whole suite passes.
 *
 *  A direct call from a `__global__` function is what turns that silence into a build error, and a build
 *  error is the only form this can be caught in without a GPU: the device backend is compiled on every
 *  CUDA runner and executed on none of them, so `check_substrings_cuda_agrees_` below never gets to run.
 */
__global__ void reach_rune_helpers_on_device_(sz_cptr_t utf8, sz_size_t length, sz_rune_t *rune_out,
                                              sz_rune_length_t *length_out, sz_u8_t *encoded_out) {
    sz_rune_t rune = 0;
    *length_out = sz_rune_decode(utf8, utf8 + length, &rune);
    *rune_out = rune;
    sz_rune_encode(rune, encoded_out);
}

/** @brief The device backend must report the identical set for the same cell, through the very same call
 *         shape every host backend takes - only the memory the haystacks live in differs. */
void check_substrings_cuda_agrees_(substrings_case_sensitivity_t sensitivity,
                                   substrings_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    substrings_serial_t serial_engine;
    verify(serial_engine.try_index(needles_view, sensitivity) == status_t::success_k);
    collect_overlapping_matches_into_(serial_engine, haystacks_view, scratch.engine_keys);

    gpu_specs_t gpu_specs;
    verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
    cuda_executor_t executor;
    substrings_cuda_t cuda_engine;
    verify(cuda_engine.try_index(needles_view, sensitivity, executor, gpu_specs) == status_t::success_k);

    // One match type and one signature across backends: the device runs the very same collection body as
    // every CPU engine. The fixture tape is `unified_alloc`-backed under CUDA, so no copy exists anywhere.
    collect_overlapping_matches_into_(cuda_engine, haystacks_view, scratch.cuda_keys, executor, gpu_specs);
    if (scratch.cuda_keys != scratch.engine_keys) {
        log_substrings_cell_mismatch_(scratch, "CUDA-vs-serial divergence");
        verify(false && "CUDA backend disagrees with the serial reference");
    }

    // The leftmost covers are a sequential greedy over each haystack, which the device resolves after the
    // walk, over the matches it emitted, in segments no match reaches across. Both policies are checked,
    // since they differ only in which match wins a start and that choice propagates.
    substrings_overlap_policy_t const covers[] = {substrings_leftmost_first_k, substrings_leftmost_longest_k};
    for (substrings_overlap_policy_t const policy : covers) {
        collect_matches_under_(serial_engine, haystacks_view, policy, scratch.engine_keys);
        collect_matches_under_(cuda_engine, haystacks_view, policy, scratch.cuda_keys, executor, gpu_specs);
        if (scratch.cuda_keys != scratch.engine_keys) {
            log_substrings_cell_mismatch_(scratch, policy == substrings_leftmost_first_k
                                                       ? "CUDA-vs-serial divergence under leftmost-first"
                                                       : "CUDA-vs-serial divergence under leftmost-longest");
            verify(false && "CUDA leftmost cover disagrees with the serial reference");
        }
    }
}

#endif // SZ_USE_CUDA

/**
 *  @brief Pins the two sides of the device memory contract: scattered device-resident haystacks - several
 *         separate unified allocations rather than one packed tape - are searched correctly, and host-backed
 *         haystacks are refused with `device_memory_mismatch_k` rather than silently copied.
 */
void test_substrings_cuda_memory_safety() {
    std::printf("  - testing unified, host, pinned and device memory against the contract...\n");
#if SZ_USE_CUDA

    gpu_specs_t gpu_specs;
    verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
    cuda_executor_t executor;

    std::vector<std::string> const needle_strings {"he", "she", "his", "hers"};
    arrow_strings_tape_t needles;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    substrings_serial_t serial_engine;
    verify(serial_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
    substrings_cuda_t cuda_engine;
    verify(cuda_engine.try_index(needles.view(), substrings_cased_k, executor, gpu_specs) == status_t::success_k);

    // Three haystacks in three separate unified allocations: the descriptors point wherever the caller's
    // memory happens to live, which is the whole point of not assuming a tape.
    std::vector<std::string> const texts {"ushers", "hishers", "she"};
    unified_texts_t const scattered {texts};
    span<span<char const> const> const scattered_view = scattered.view();

    arrow_strings_tape_t reference_haystacks;
    verify(reference_haystacks.try_assign(texts.data(), texts.data() + texts.size()) == status_t::success_k);
    substrings_match_set_t serial_keys, scattered_keys;
    collect_overlapping_matches_into_(serial_engine, reference_haystacks.view(), serial_keys);
    collect_overlapping_matches_into_(cuda_engine, scattered_view, scattered_keys, executor, gpu_specs);
    verify(!serial_keys.empty() && "The fixture must produce matches");
    verify(scattered_keys == serial_keys && "Scattered unified allocations must match the packed reference");

    // Host memory is refused, never copied: materialization is the caller's explicit choice, as in every
    // other domain. The strings above live on the host, so their spans are exactly the illegal input.
    std::vector<span<char const>> host_spans;
    for (std::string const &text : texts) host_spans.push_back({text.data(), text.size()});
    unified_vector<std::size_t> counts(host_spans.size());
    std::size_t matches_total = 0;
    auto const host_status = cuda_engine.try_count(host_spans, substrings_overlapping_k,
                                                   span<std::size_t>(counts.data(), counts.size()), matches_total,
                                                   executor, gpu_specs);
    verify(host_status == status_t::device_memory_mismatch_k && "Host input must be refused, not copied");

    // Host OUTPUTS are refused just as firmly, and every verb probes its own. The haystacks below are the
    // unified ones, so the only illegal thing in each call is where the results were asked to land.
    {
        std::vector<std::size_t> host_counts(scattered_view.size());
        std::size_t total = 0;
        verify(cuda_engine.try_count(scattered_view, substrings_overlapping_k,
                                     span<std::size_t>(host_counts.data(), host_counts.size()), total, executor,
                                     gpu_specs) == status_t::device_memory_mismatch_k &&
               "A host counts array must be refused");

        std::vector<substrings_match_t> host_matches(16);
        std::size_t found = 0;
        verify(cuda_engine.try_find(scattered_view, substrings_overlapping_k,
                                    span<substrings_match_t>(host_matches.data(), host_matches.size()), found, executor,
                                    gpu_specs) == status_t::device_memory_mismatch_k &&
               "A host matches array must be refused");

        unified_vector<float> weights(needle_strings.size(), 1.0f);
        std::vector<float> host_scores(scattered_view.size());
        substrings_bm25_t parameters;
        parameters.average_document_length = 8.0f;
        verify(cuda_engine.try_score_bm25(scattered_view, span<float const>(), parameters,
                                          {weights.data(), weights.size()},
                                          span<float>(host_scores.data(), host_scores.size()), executor,
                                          gpu_specs) == status_t::device_memory_mismatch_k &&
               "A host scores array must be refused");
    }

    // Page-locked host memory is a third kind, and the driver reports it as host - so it is refused too,
    // which is the refusal a caller is least likely to predict.
    {
        pinned_vector<std::size_t> pinned_counts(scattered_view.size());
        std::size_t total = 0;
        verify(cuda_engine.try_count(scattered_view, substrings_overlapping_k,
                                     span<std::size_t>(pinned_counts.data(), pinned_counts.size()), total, executor,
                                     gpu_specs) == status_t::device_memory_mismatch_k &&
               "Page-locked host memory must be refused, like any other host memory");
    }

    // Plain device memory is accepted, which unified memory alone would not prove: every host loop that once
    // filled these arrays would have faulted on a pointer the host cannot touch.
    {
        device_vector<std::size_t> device_counts;
        verify(device_counts.try_resize_uninitialized(scattered_view.size()) == status_t::success_k);
        std::size_t total = 0;
        verify(cuda_engine.try_count(scattered_view, substrings_overlapping_k,
                                     span<std::size_t>(device_counts.data(), device_counts.size()), total, executor,
                                     gpu_specs) == status_t::success_k &&
               "Plain device memory must be accepted, not just unified");

        std::vector<std::size_t> drained(device_counts.size());
        verify(copy_device_to_host(device_counts, span<std::size_t>(drained.data(), drained.size())) == CUDA_SUCCESS &&
               "Draining the device counts must succeed");
        std::size_t drained_total = 0;
        for (std::size_t const count : drained) drained_total += count;
        verify(drained_total == total && "Counts written to device memory must sum to the reported total");
    }
#endif // SZ_USE_CUDA
}

/**
 *  @brief Crosses every needle vocabulary with every placement skeleton and every needle transform.
 *
 *  Walked by rotation rather than nested loops: `rotating_index` advances a phase each full turn, so
 *  crossing the rotations still reaches every combination, and `scale_iterations` decides how far into that
 *  product a run gets. Ground truth and the declared planting effects run on every cell; the costlier
 *  properties rotate, keeping the per-cell price flat while the sweep as a whole still exercises each of
 *  them against each axis triple.
 */
void test_substrings_adversarial_equivalence() {
    std::printf("  - testing adversarial needle x placement x transform cross-product...\n");
    std::size_t const needle_generators = (std::size_t)substrings_needle_generator_t::count_k;
    std::size_t const placements = (std::size_t)substrings_placement_t::count_k;
    std::size_t const transforms = (std::size_t)substrings_transform_t::count_k;
    std::size_t const cells = scale_iterations(needle_generators * placements * transforms);
    substrings_adversarial_scratch_t scratch; // ? Constructed once, refilled by every cell below.
    bool saw_narrow_build = false;            // ? A sweep of nothing but declines would prove nothing.

    // One real pool, so sliced-specs cells exercise the all-cores-on-one-haystack path; a dummy executor
    // would run `for_slices` as a single slice and never slice at all.
    forkunion_executor_t pool;
    verify(pool.try_spawn(4) == status_t::success_k);

    for (std::size_t step = 0; step < cells; ++step) {
        auto const needle_kind = (substrings_needle_generator_t)rotating_index(step, needle_generators);
        auto const placement = (substrings_placement_t)rotating_index(step, placements);
        auto const transform = (substrings_transform_t)rotating_index(step, transforms);
        // A vocabulary of arbitrary bytes can only be matched byte-exactly: folding rejects malformed UTF-8
        // by contract, and the brute-force oracle decodes its inputs too.
        auto const sensitivity = step % 2 && substrings_needles_are_utf8_(needle_kind) ? substrings_uncased_k
                                                                                       : substrings_cased_k;

        scratch.needle_kind = needle_kind;
        scratch.placement = placement;
        scratch.transform = transform;
        scratch.specs = cpu_specs_t {};
        if (step % 3 == 0) scratch.specs.l2_bytes = 1024; // ? Far below the fixtures, forcing haystack slicing.

        // A vocabulary whose output pool grows with the square of its depth takes the quadratic knob, so
        // doubling the multiplier doubles the work rather than quadrupling it.
        std::size_t const depth = substrings_needles_grow_quadratically_(needle_kind) ? scale_iterations_quadratic(10)
                                                                                      : scale_iterations(10);
        generate_substrings_needles_(needle_kind, depth, scratch.needles);
        generate_substrings_placements_(placement, transform, sensitivity, needle_kind, scratch.needles.view(), 6,
                                        scratch.haystacks, scratch.plantings);

        check_substrings_against_oracle_(sensitivity, scratch);
        check_substrings_declared_effects_(scratch);
        switch (rotating_index(step, 3)) {
        case 0: check_substrings_backends_agree_(sensitivity, scratch, pool); break;
        case 1: saw_narrow_build |= check_substrings_narrow_width_(sensitivity, scratch); break;
        case 2: check_substrings_tier_invariant_(sensitivity, scratch); break;
        }
#if SZ_USE_CUDA
        check_substrings_cuda_agrees_(sensitivity, scratch);
#endif
    }

    verify(saw_narrow_build && "Narrow-width sweep never built a u16 automaton - shrink the vocabulary");
}

/**
 *  @brief The parallel engine's large-haystack path - every core slicing one haystack - must agree with the
 *         serial engine even when matches start on nearly every byte and straddle every slice boundary.
 *
 *  The default `cpu_specs_t` L2 threshold keeps the adversarial fixtures above on the one-core-per-haystack
 *  path, so this one forces the slicing with a threshold far below its own size and a real pool.
 */
void test_substrings_large_haystacks_equivalence() {
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

    substrings_serial_t serial_engine;
    verify(serial_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
    substrings_match_set_t serial_keys, parallel_keys;
    collect_overlapping_matches_into_(serial_engine, haystacks.view(), serial_keys);

    substrings_parallel_t parallel_engine;
    verify(parallel_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
    collect_overlapping_matches_into_(parallel_engine, haystacks.view(), parallel_keys, pool, sliced_specs);

    verify(!serial_keys.empty() && "The fixture must produce matches");
    verify(parallel_keys == serial_keys && "The sliced parallel path disagrees with the serial reference");

    // The same split, folded. Several of the four cuts land mid-codepoint on this fixture, which a folded
    // walk cannot restart on, so the slices snap back to a codepoint start before folding. Attribution by
    // start position makes an unsnapped cut safe on its own - no match can begin inside the codepoint a cut
    // lands in - so this pins parallel and serial agreement over multi-byte content rather than the snap.
    {
        std::vector<std::string> const folded_needles {"ss", "\xC3\x9F", "k", "\xE2\x84\xAA"};
        arrow_strings_tape_t uncased_needles;
        verify(uncased_needles.try_assign(folded_needles.data(), folded_needles.data() + folded_needles.size()) ==
               status_t::success_k);

        std::string multibyte;
        char const *const cycle[] = {"\xC3\x9F", "\xC5\xBF", "\xE2\x84\xAA", "s", "k"};
        for (std::size_t index = 0; index < scale_iterations(4) * 1024; ++index) multibyte += cycle[index % 5];
        std::vector<std::string> const uncased_haystack_strings {multibyte};
        arrow_strings_tape_t uncased_haystacks;
        verify(uncased_haystacks.try_assign(uncased_haystack_strings.data(), uncased_haystack_strings.data() + 1) ==
               status_t::success_k);

        substrings_serial_t uncased_serial;
        verify(uncased_serial.try_index(uncased_needles.view(), substrings_uncased_k) == status_t::success_k);
        substrings_match_set_t uncased_serial_keys, uncased_parallel_keys;
        collect_overlapping_matches_into_(uncased_serial, uncased_haystacks.view(), uncased_serial_keys);

        substrings_parallel_t uncased_parallel;
        verify(uncased_parallel.try_index(uncased_needles.view(), substrings_uncased_k) == status_t::success_k);
        collect_overlapping_matches_into_(uncased_parallel, uncased_haystacks.view(), uncased_parallel_keys, pool,
                                          sliced_specs);

        verify(!uncased_serial_keys.empty() && "The folded fixture must produce matches");
        verify(uncased_parallel_keys == uncased_serial_keys &&
               "The sliced parallel path disagrees with the serial reference on folded multi-byte content");
    }
}

#pragma endregion // Adversarial

#pragma region Construction

/**
 *  @brief Structural invariants of the compiled automaton, checked directly against the published
 *         `aho_corasick_view` rather than against another backend's output.
 */
void test_substrings_construction_equivalence() {
    std::printf("  - testing structural invariants of the compiled automaton...\n");

    // Re-indexing replaces the needle set outright. An engine is tiered for the machine that indexed it, so
    // being able to index again is what lets one engine be re-pointed at a new vocabulary - or re-tiered for
    // another device - instead of being destroyed and rebuilt.
    {
        std::vector<std::string> const first_strings {"alpha", "beta"};
        std::vector<std::string> const second_strings {"gamma", "delta", "epsilon"};
        std::vector<std::string> const haystack_strings {"alpha beta gamma delta epsilon"};
        arrow_strings_tape_t first, second, haystacks;
        verify(first.try_assign(first_strings.data(), first_strings.data() + first_strings.size()) ==
               status_t::success_k);
        verify(second.try_assign(second_strings.data(), second_strings.data() + second_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);

        substrings_serial_t reused;
        verify(reused.try_index(first.view(), substrings_cased_k) == status_t::success_k);
        verify(reused.count_needles() == first_strings.size());
        substrings_match_set_t first_matches;
        collect_overlapping_matches_into_(reused, haystacks.view(), first_matches);

        verify(reused.try_index(second.view(), substrings_cased_k) == status_t::success_k);
        verify(reused.count_needles() == second_strings.size() &&
               "Re-indexing must replace the needle set, not extend it");
        substrings_match_set_t second_matches;
        collect_overlapping_matches_into_(reused, haystacks.view(), second_matches);

        // A freshly built engine over the same vocabulary is the oracle: re-indexing must leave nothing of
        // the first automaton behind, in the dictionary or in the scratch sized against it.
        substrings_serial_t fresh;
        verify(fresh.try_index(second.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t fresh_matches;
        collect_overlapping_matches_into_(fresh, haystacks.view(), fresh_matches);
        verify(second_matches == fresh_matches && "A re-indexed engine must match a freshly indexed one");
        verify(second_matches != first_matches && "The fixture must distinguish the two vocabularies");
    }

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
        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_overlapping_matches_into_(engine, haystacks.view(), matches);
        verify(matches.size() == 6 && "All matches are reported - this is not leftmost-longest matching");
    }

    // An empty needle is rejected outright. Skipping it silently would consume no needle index, so every
    // later needle's reported `needle_index` would disagree with the caller's own array.
    {
        substrings_u32_dictionary_t dictionary;
        verify(dictionary.try_insert(span<char const> {}) == status_t::unexpected_dimensions_k);
        verify(dictionary.count_needles() == 0 && "A rejected needle must not consume an index");
        verify(dictionary.try_insert(span<char const> {"ab", 2}) == status_t::success_k);
        verify(dictionary.count_needles() == 1);
    }

    // The doubling family: a run of `s` folds ambiguously, since each `s` is also half a sharp-S. Folding the
    // stream leaves one path per needle, so the run is a plain chain of its own folded length and every id
    // width has room for it, whatever `state_id_t` is.
    {
        std::string const long_run(11, 's');
        substrings_u16_dictionary_t narrow;
        narrow.case_sensitivity(substrings_uncased_k);
        verify(narrow.try_insert({long_run.data(), long_run.size()}) == status_t::success_k);
        verify(narrow.try_build() == status_t::success_k && "The doubling family is linear once the stream folds");
        verify(narrow.count_states() == long_run.size() + 1 && "One state per folded byte, plus the root");

        substrings_u32_dictionary_t wide;
        wide.case_sensitivity(substrings_uncased_k);
        verify(wide.try_insert({long_run.data(), long_run.size()}) == status_t::success_k);
        verify(wide.try_build() == status_t::success_k);
        verify(wide.count_states() == narrow.count_states() && "The id width cannot change the automaton's shape");

        // Both widths still find the sharp S spelling of the same run, which is what makes it the hard case.
        std::string const spelled = "\xC3\x9F\xC3\x9F\xC3\x9F\xC3\x9F\xC3\x9Fs";
        std::size_t found = 0;
        wide.find({spelled.data(), spelled.size()},
                  [&](std::size_t, std::size_t, std::size_t) { return ++found, true; });
        verify(found != 0 && "Eleven folded s-runes are spelled by five sharp-S codepoints and one s");
    }

    // Narrowing an automaton that does fit. The engine derives once at the wider id and keeps the narrower
    // dictionary when its ceilings hold, so what is checked here is that it actually does: halving the row
    // width is the point - a `u16` row is 512 bytes where `u32`'s is 1024. That the narrowed arrays answer
    // identically to the wide ones is `check_substrings_narrow_width_`'s job, against the brute-force oracle.
    {
        std::vector<std::string> const needle_strings = random_short_strings_(scale_iterations(200), 3, 7);
        std::vector<std::string> const haystack_strings = random_haystacks_with_needles_(needle_strings,
                                                                                         scale_iterations(64), 16, 96);
        arrow_strings_tape_t needles, haystacks;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);

        substrings_serial_t narrowed;
        verify(narrowed.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        verify(narrowed.state_width() == substrings_state_width_t::u16_k &&
               "A few hundred short needles fit the narrow id, so the engine must have kept it");

        // The wide derivation the narrowing copied from, rebuilt here, must describe the same automaton.
        substrings_u32_dictionary_t wide;
        wide.case_sensitivity(substrings_cased_k);
        for (span<char const> const &needle : needles.view()) verify(wide.try_insert(needle) == status_t::success_k);
        verify(wide.try_build() == status_t::success_k);
        verify(narrowed.count_states() == wide.count_states());
        verify(narrowed.count_needles() == wide.count_needles());

        substrings_match_set_t narrowed_matches;
        collect_overlapping_matches_into_(narrowed, haystacks.view(), narrowed_matches);
        verify(narrowed_matches.collected_matches.size() != 0 && "The fixture must produce matches to compare");
    }

    // Cold-tier double-array invariant: every slot a state claims sits within 256 of that state's own base,
    // and the failure chain from any cold state reaches the root. `hot_count` is forced to zero, so the cold
    // tier is genuinely exercised whatever the default hot tier would have been.
    {
        std::vector<std::string> const needle_strings = random_short_strings_(scale_iterations(300), 3, 7);
        arrow_strings_tape_t needles;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        substrings_u32_dictionary_t dictionary;
        dictionary.case_sensitivity(substrings_cased_k);
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
        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_uncased_k) == status_t::success_k);

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
        substrings_match_set_t matches;
        collect_overlapping_matches_into_(engine, haystacks.view(), matches);

        std::vector<std::size_t> counts(haystacks.view().size(), 0);
        std::size_t matches_total = 0;
        verify(engine.try_count(haystacks.view(), substrings_overlapping_k,
                                span<std::size_t>(counts.data(), counts.size()), matches_total) == status_t::success_k);
        std::vector<substrings_match_t> raw_matches(matches_total);
        std::size_t matches_found = 0;
        verify(engine.try_find(haystacks.view(), substrings_overlapping_k,
                               span<substrings_match_t>(raw_matches.data(), raw_matches.size()),
                               matches_found) == status_t::success_k);

        for (substrings_match_t const &match : raw_matches) {
            // The match carries offsets only, so the bytes come from the haystack it names - the exact
            // recipe the match struct's own docstring prescribes.
            span<char const> const haystack = haystacks[match.haystack_index];
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

#pragma region Matching

/**
 *  @brief The leftmost covers are a subset of the overlapping matches, and they share no bytes.
 *
 *  Stated against the overlapping walk rather than a second implementation: every match a cover reports must
 *  be a match the exhaustive walk also found, and no two of them may touch.
 */
void test_substrings_cover_equivalence() {
    std::printf("  - testing that a leftmost cover is a non-overlapping subset of every match...\n");

    std::vector<std::string> const needle_strings = random_short_strings_(scale_iterations(150), 2, 5);
    std::vector<std::string> const haystack_strings = random_haystacks_with_needles_(needle_strings,
                                                                                     scale_iterations(40), 16, 64);
    arrow_strings_tape_t needles, haystacks;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
           status_t::success_k);

    substrings_serial_t engine;
    verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
    substrings_match_set_t overlapping;
    collect_matches_under_(engine, haystacks.view(), substrings_overlapping_k, overlapping);

    substrings_overlap_policy_t const leftmost_policies[2] = {substrings_leftmost_longest_k,
                                                              substrings_leftmost_first_k};
    for (substrings_overlap_policy_t const policy : leftmost_policies) {
        substrings_match_set_t cover;
        collect_matches_under_(engine, haystacks.view(), policy, cover);

        for (substrings_match_t const &match : cover)
            verify(std::binary_search(overlapping.matches.begin(), overlapping.matches.end(), match,
                                      substrings_match_less_) &&
                   "A cover may only report matches the exhaustive walk also found");

        // The set orders by needle before offset, so it has to be re-sorted by position before neighbours
        // mean anything - only then does "no two neighbours touch" imply "no two matches touch".
        std::vector<substrings_match_t> by_position = cover.matches;
        std::sort(by_position.begin(), by_position.end(), substrings_match_by_position_less_);
        for (std::size_t index = 1; index < by_position.size(); ++index) {
            substrings_match_t const &earlier = by_position[index - 1];
            substrings_match_t const &later = by_position[index];
            if (earlier.haystack_index != later.haystack_index) continue;
            verify(earlier.byte_offset + earlier.byte_length <= later.byte_offset &&
                   "Two matches of one cover shared a byte");
        }
    }
}

#pragma endregion // Matching

#pragma region Rewriting

/**
 *  @brief The flat tape and per-haystack offsets a single rewrite pass produced.
 *
 *  Both members are unified, because a CUDA engine writes them from the device and refuses host memory.
 */
struct substrings_rewrite_tape_t {
    unified_vector<char> text;
    unified_vector<std::size_t> offsets;

    span<char const> operator[](std::size_t haystack_index) const noexcept {
        return {text.data() + offsets[haystack_index], offsets[haystack_index + 1] - offsets[haystack_index]};
    }
};

/** @brief Whether @p rewritten 's whole tape holds @p expected, compared as bytes. */
inline bool tape_equals_(substrings_rewrite_tape_t const &rewritten, std::string const &expected) noexcept {
    return rewritten.text.size() == expected.size() &&
           std::memcmp(rewritten.text.data(), expected.data(), expected.size()) == 0;
}

/** @brief Whether @p rewritten holds @p expected at @p haystack_index, compared as bytes. */
inline bool rewritten_equals_(substrings_rewrite_tape_t const &rewritten, std::size_t haystack_index,
                              std::string const &expected) noexcept {
    span<char const> const produced = rewritten[haystack_index];
    return produced.size() == expected.size() && std::memcmp(produced.data(), expected.data(), expected.size()) == 0;
}

/**
 *  @brief Rewrites @p haystacks through @p engine under @p overlap_policy, returning the flat result tape.
 *
 *  Sizes the tape with a zero-capacity call first, which is the same size query `try_find` offers, so the
 *  rewrite itself always runs against a buffer that is exactly big enough.
 */
template <typename engine_type_, typename haystacks_type_, typename... trailing_args_>
substrings_rewrite_tape_t rewrite_all_(engine_type_ &engine, haystacks_type_ const &haystacks,
                                       substrings_overlap_policy_t overlap_policy,
                                       arrow_strings_tape_t const &replacements, trailing_args_ &&...trailing) {

    substrings_rewrite_tape_t rewritten;
    rewritten.offsets.assign(haystacks.size() + 1, 0);
    span<std::size_t> const offsets_view(rewritten.offsets.data(), rewritten.offsets.size());
    std::size_t needed = 0;
    status_t const sized = engine.try_replace(haystacks, overlap_policy, replacements.view(), span<char>(),
                                              offsets_view, needed, trailing...);
    verify((needed == 0 ? sized == status_t::success_k : sized == status_t::unexpected_dimensions_k) &&
           "A zero-capacity rewrite must refuse and name the size it wanted");

    rewritten.text.assign(needed, '\0');
    std::size_t written = 0;
    verify(engine.try_replace(haystacks, overlap_policy, replacements.view(), span<char>(rewritten.text.data(), needed),
                              offsets_view, written, trailing...) == status_t::success_k);
    verify(written == needed && "The sizing pass and the writing pass must agree");
    verify(rewritten.offsets[haystacks.size()] == written && "The terminator must close the last haystack");
    return rewritten;
}

/**
 *  @brief Rewriting under both leftmost policies, against the oracle a rewrite carries for free.
 *
 *  Replacing every needle with itself must reproduce the input byte for byte, whatever the vocabulary and
 *  whichever cover resolved it - a property that needs no second implementation to check against.
 */
void test_substrings_rewriting_equivalence() {
    std::printf("  - testing rewriting against the self-replacement oracle...\n");

    substrings_overlap_policy_t const leftmost_policies[2] = {substrings_leftmost_longest_k,
                                                              substrings_leftmost_first_k};

    // Replacing each needle with itself is the identity, so any deviation is the rewrite's own bug.
    {
        std::vector<std::string> const needle_strings = random_short_strings_(scale_iterations(200), 2, 6);
        std::vector<std::string> const haystack_strings = random_haystacks_with_needles_(needle_strings,
                                                                                         scale_iterations(50), 16, 64);
        arrow_strings_tape_t needles, haystacks, replacements;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        verify(replacements.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        for (substrings_overlap_policy_t const policy : leftmost_policies) {
            substrings_rewrite_tape_t const rewritten = rewrite_all_(engine, haystacks.view(), policy, replacements);
            verify(rewritten.offsets.size() == haystack_strings.size() + 1 && "One boundary per haystack, plus one");
            for (std::size_t index = 0; index < haystack_strings.size(); ++index)
                verify(rewritten_equals_(rewritten, index, haystack_strings[index]) &&
                       "Replacing every needle with itself must reproduce the input");
        }
    }

    // An empty replacement deletes, and a replacement shorter than its needle shrinks the output - the sizing
    // pass accumulates removals and insertions apart, so neither can wrap the unsigned total.
    {
        std::vector<std::string> const needle_strings {"cat", "dog"};
        std::vector<std::string> const haystack_strings {"cats and dogs", "dogcat", "no pets here"};
        std::vector<std::string> const replacement_strings {"", "z"};
        arrow_strings_tape_t needles, haystacks, replacements;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        verify(replacements.try_assign(replacement_strings.data(),
                                       replacement_strings.data() + replacement_strings.size()) == status_t::success_k);

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_rewrite_tape_t const rewritten = rewrite_all_(engine, haystacks.view(),
                                                                 substrings_leftmost_longest_k, replacements);
        verify(rewritten_equals_(rewritten, 0, "s and zs") && "An empty replacement deletes and a shorter one shrinks");
        verify(rewritten_equals_(rewritten, 1, "z") && "Both needles rewrite in one haystack");
        verify(rewritten_equals_(rewritten, 2, haystack_strings[2]) &&
               "A haystack with no match passes through untouched");
    }

    // A longer needle shadows a shorter one under `leftmost_longest`, while `leftmost_first` takes whichever
    // needle was listed first - the same distinction the matching policies draw, carried into the rewrite.
    {
        std::vector<std::string> const needle_strings {"cat", "catalog"};
        std::vector<std::string> const haystack_strings {"catalog"};
        std::vector<std::string> const replacement_strings {"feline", "directory"};
        arrow_strings_tape_t needles, haystacks, replacements;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        verify(replacements.try_assign(replacement_strings.data(),
                                       replacement_strings.data() + replacement_strings.size()) == status_t::success_k);

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        verify(rewritten_equals_(rewrite_all_(engine, haystacks.view(), substrings_leftmost_longest_k, replacements), 0,
                                 "directory") &&
               "The longest needle shadows the shorter one it contains");
        verify(rewritten_equals_(rewrite_all_(engine, haystacks.view(), substrings_leftmost_first_k, replacements), 0,
                                 "felinealog") &&
               "The lower needle index wins however long its rival");
    }

    // An overlapping rewrite is not a function - two matches sharing a byte have no single answer - so the
    // policy is refused rather than resolved to some arbitrary winner.
    {
        std::vector<std::string> const needle_strings {"ab"};
        std::vector<std::string> const haystack_strings {"abab"};
        arrow_strings_tape_t needles, haystacks, replacements;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        verify(replacements.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);

        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        std::vector<std::size_t> offsets(haystacks.view().size() + 1, 0);
        std::size_t written = 0;
        verify(engine.try_replace(haystacks.view(), substrings_overlapping_k, replacements.view(), span<char>(),
                                  span<std::size_t>(offsets.data(), offsets.size()), written) != status_t::success_k);
    }

    // A haystack no core's cache holds is split, each core resolving its own stretch of the cover from a
    // restart no match spans. The identity oracle holds across the cuts, and so does the serial engine's tape.
    {
        std::vector<std::string> const needle_strings {"cat", "at", "concat", "the"};
        std::vector<std::string> const replacement_strings {"[CAT]", "@", "<<CONCAT>>", ""};
        std::vector<std::string> long_strings;
        for (std::size_t index = 0; index < 5; ++index) {
            std::string text;
            while (text.size() < 96u * 1024u) text += "the cat sat on a mat concatenating cats at ";
            long_strings.push_back(std::move(text));
        }

        arrow_strings_tape_t needles, haystacks, replacements, identity;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(long_strings.data(), long_strings.data() + long_strings.size()) ==
               status_t::success_k);
        verify(replacements.try_assign(replacement_strings.data(),
                                       replacement_strings.data() + replacement_strings.size()) == status_t::success_k);
        verify(identity.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);

        substrings_serial_t serial_engine;
        substrings_parallel_t parallel_engine;
        verify(serial_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        verify(parallel_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        forkunion_executor_t pool;
        verify(pool.try_spawn(4) == status_t::success_k);
        cpu_specs_t split_specs;
        split_specs.l2_bytes = 4096; // ? Small enough that every haystack above is split across all cores.

        for (substrings_overlap_policy_t policy : leftmost_policies) {
            substrings_rewrite_tape_t const unsplit = rewrite_all_(serial_engine, haystacks.view(), policy,
                                                                   replacements);
            substrings_rewrite_tape_t const split = rewrite_all_(parallel_engine, haystacks.view(), policy,
                                                                 replacements, pool, split_specs);
            verify(split.offsets == unsplit.offsets && "A split rewrite must land on the same boundaries");
            verify(split.text == unsplit.text && "A split rewrite must produce the same bytes");

            substrings_rewrite_tape_t const unchanged = rewrite_all_(parallel_engine, haystacks.view(), policy,
                                                                     identity, pool, split_specs);
            std::string packed;
            for (std::string const &text : long_strings) packed += text;
            verify(tape_equals_(unchanged, packed) && "Self-replacement must survive every slice boundary");
        }
    }
}

#pragma endregion // Rewriting

#pragma region Scoring

/**
 *  @brief BM25 against scores worked out by hand, plus the bit-stability the header promises.
 *
 *  Term frequencies are raw overlapping counts, so a needle nested in another still contributes every one of
 *  its own occurrences - which is what classic BM25 scores and what a leftmost cover would have suppressed.
 */
void test_substrings_scoring_unit() {
    std::printf("  - testing BM25 scores against hand-computed values...\n");

    std::vector<std::string> const needle_strings {"cat", "dog"};
    std::vector<std::string> const haystack_strings {"catcat", "dog", "nothing"};
    arrow_strings_tape_t needles, haystacks;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
           status_t::success_k);

    substrings_serial_t engine;
    verify(engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);

    substrings_bm25_t const parameters {.term_frequency_saturation = 1.2f,
                                        .length_normalization = 0.75f,
                                        .average_document_length = 6.0f};
    std::vector<float> const weights {1.0f, 2.0f};
    std::vector<float> scores(haystack_strings.size(), 0.0f);
    span<float const> const weights_view(weights.data(), weights.size());
    span<float> const scores_view(scores.data(), scores.size());

    verify(engine.try_score_bm25(haystacks.view(), span<float const>(), parameters, weights_view, scores_view) ==
           status_t::success_k);

    // "catcat" is 6 bytes against a 6-byte mean, so the length term is exactly one and "cat" twice scores
    //   1.0 * 2 * (1.2 + 1) / (2 + 1.2 * 1) = 4.4 / 3.2
    verify(std::fabs(scores[0] - 4.4f / 3.2f) < 1e-5f && "Two occurrences must saturate, not double");

    // "dog" is 3 bytes against the same mean, so the length term is 1 - 0.75 + 0.75 * 3 / 6 = 0.625, and the
    // needle's own weight of two scales the whole term: 2.0 * 1 * 2.2 / (1 + 1.2 * 0.625).
    verify(std::fabs(scores[1] - 2.0f * 2.2f / (1.0f + 1.2f * 0.625f)) < 1e-5f &&
           "A weight scales its needle's whole contribution");
    verify(scores[2] == 0.0f && "A haystack no needle hits scores exactly zero");

    // Bit-stability: the same call on the same engine must reproduce every score exactly, not merely closely.
    std::vector<float> const first_scores = scores;
    for (std::size_t repeat = 0; repeat < 4; ++repeat) {
        std::fill(scores.begin(), scores.end(), 0.0f);
        verify(engine.try_score_bm25(haystacks.view(), span<float const>(), parameters, weights_view, scores_view) ==
               status_t::success_k);
        verify(scores == first_scores && "Scores must be bit-identical across runs of one backend");
    }

    // An empty `document_lengths` means byte lengths, which the caller can also state outright.
    {
        std::vector<float> byte_lengths;
        for (std::string const &haystack : haystack_strings) byte_lengths.push_back((float)haystack.size());
        std::vector<float> explicit_scores(haystack_strings.size(), 0.0f);
        verify(engine.try_score_bm25(
                   haystacks.view(), span<float const>(byte_lengths.data(), byte_lengths.size()), parameters,
                   weights_view, span<float>(explicit_scores.data(), explicit_scores.size())) == status_t::success_k);
        verify(explicit_scores == first_scores && "Byte lengths stated outright must score identically");
    }

    // A zero weight removes its needle from the ranking without removing it from the automaton.
    {
        std::vector<float> const muted {0.0f, 0.0f};
        std::vector<float> muted_scores(haystack_strings.size(), 1.0f);
        verify(engine.try_score_bm25(haystacks.view(), span<float const>(), parameters,
                                     span<float const>(muted.data(), muted.size()),
                                     span<float>(muted_scores.data(), muted_scores.size())) == status_t::success_k);
        for (float const score : muted_scores) verify(score == 0.0f && "Zero weights must score zero");
    }

    // The parallel backend splits a haystack no core's cache holds, merging per-core tallies. Frequencies are
    // integers and the merged row reduces in the same ascending order, so the scores stay bit-identical.
    {
        std::vector<std::string> long_strings;
        for (std::size_t index = 0; index < 6; ++index) {
            std::string text;
            while (text.size() < 64u * 1024u) text += "cat dog concatenate at the ";
            long_strings.push_back(std::move(text));
        }
        arrow_strings_tape_t long_haystacks;
        verify(long_haystacks.try_assign(long_strings.data(), long_strings.data() + long_strings.size()) ==
               status_t::success_k);

        substrings_parallel_t parallel_engine;
        verify(parallel_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        forkunion_executor_t pool;
        verify(pool.try_spawn(4) == status_t::success_k);
        cpu_specs_t split_specs;
        split_specs.l2_bytes = 4096; // ? Small enough that every haystack above is split across all cores.

        std::vector<float> serial_long(long_strings.size(), 0.0f), parallel_long(long_strings.size(), 0.0f);
        verify(engine.try_score_bm25(long_haystacks.view(), span<float const>(), parameters, weights_view,
                                     span<float>(serial_long.data(), serial_long.size())) == status_t::success_k);
        verify(parallel_engine.try_score_bm25(long_haystacks.view(), span<float const>(), parameters, weights_view,
                                              span<float>(parallel_long.data(), parallel_long.size()), pool,
                                              split_specs) == status_t::success_k);
        verify(parallel_long == serial_long && "A split haystack must score exactly as an unsplit one");
    }
}

/**
 *  @brief BM25 over dictionaries wide enough to change how both backends hold their counters.
 *
 *  The CPU orders the needles a document hit in passes covering the widest needle index, so a two-needle
 *  fixture drives only one pass. The GPU indexes its table straight by needle while the dictionary fits it,
 *  hashes once it does not, and spills to a per-block row beyond that. All three fail silently, since a
 *  needle counted twice scores as two small counts and `substrings_bm25_term` is concave.
 *
 *  Frequencies are known by construction: the needles are fixed width and space separated, so a match can
 *  only begin where a token does, and the text's own recipe is the oracle.
 */
void test_substrings_scoring_wide_equivalence() {
    std::printf("  - testing BM25 across the direct, hashed and overflow regimes...\n");

    // Fixed-width needles, so none is a substring of another and a haystack's distinct count is exactly the
    // number of tokens it was built from - which is what lets the regime be chosen rather than hoped for.
    auto const needle_at = [](std::size_t index) {
        std::string text = "w00000";
        for (std::size_t digit = 0, value = index; digit < 5; ++digit, value /= 10)
            text[5 - digit] = (char)('0' + (value % 10));
        return text;
    };

    // Below one slot per needle the GPU indexes its table directly; above it, hashed. The widths also carry
    // the CPU ordering past one radix pass, which a needle index under 256 would never reach.
    for (std::size_t needle_count : {std::size_t {4000}, std::size_t {20000}}) {
        std::vector<std::string> needle_strings;
        for (std::size_t index = 0; index < needle_count; ++index) needle_strings.push_back(needle_at(index));
        arrow_strings_tape_t needles;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);

        // One haystack per distinct-count, the widest touching every needle, so a 20,000-needle dictionary
        // overruns an 8,192-slot table while a 4,000-needle one never can. The recipe is kept as the oracle.
        std::vector<std::string> texts;
        std::vector<std::vector<std::uint32_t>> expected_frequencies;
        for (std::size_t distinct : {needle_count, needle_count / 2, std::size_t {17}, std::size_t {0}}) {
            std::string text;
            std::vector<std::uint32_t> frequencies(needle_count, 0u);
            for (std::size_t index = 0; index < distinct; ++index) {
                text += needle_at(index), text += ' ', ++frequencies[index];
                if (index % 3 == 0) text += needle_at(index), text += ' ', ++frequencies[index];
            }
            texts.push_back(std::move(text));
            expected_frequencies.push_back(std::move(frequencies));
        }

        arrow_strings_tape_t host_haystacks;
        verify(host_haystacks.try_assign(texts.data(), texts.data() + texts.size()) == status_t::success_k);

        substrings_serial_t serial_engine;
        verify(serial_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);

        substrings_bm25_t const parameters {.term_frequency_saturation = 1.2f,
                                            .length_normalization = 0.75f,
                                            .average_document_length = 4096.0f};
        // Weights are read by the scoring kernel, so on a CUDA scope they must live where it can reach them.
        unified_vector<float> weights(needle_count);
        for (std::size_t index = 0; index < needle_count; ++index) weights[index] = 1.0f + (float)(index % 7);
        span<float const> const weights_view {weights.data(), weights.size()};

        std::vector<float> host_scores(texts.size(), 0.0f);
        verify(serial_engine.try_score_bm25(host_haystacks.view(), span<float const>(), parameters, weights_view,
                                            span<float>(host_scores.data(), host_scores.size())) ==
               status_t::success_k);

        // The oracle sums the recipe's own frequencies ascending by needle, in `f32`, which is the order the
        // header publishes - so this is an equality rather than a tolerance, and it is what pins the ordering
        // the engine reaches by a different route.
        for (std::size_t index = 0; index < texts.size(); ++index) {
            float const document_length = (float)texts[index].size();
            float expected = 0;
            for (std::size_t needle = 0; needle < needle_count; ++needle) {
                std::uint32_t const frequency = expected_frequencies[index][needle];
                if (frequency == 0u) continue;
                expected += weights[needle] * substrings_bm25_term(parameters, (float)frequency, document_length);
            }
            verify(host_scores[index] == expected && "Serial scores must match an ascending-order oracle exactly");
        }

#if SZ_USE_CUDA
        gpu_specs_t gpu_specs;
        verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
        cuda_executor_t executor;

        // A kernel cannot reach the caller's host memory, so the same texts are staged unified for the device.
        unified_texts_t const staged {texts};
        span<span<char const> const> const haystacks_view = staged.view();

        substrings_cuda_t cuda_engine;
        verify(cuda_engine.try_index(needles.view(), substrings_cased_k, executor, gpu_specs) == status_t::success_k);

        unified_vector<float> device_scores(texts.size(), 0.0f);
        verify(cuda_engine
                   .try_score_bm25(haystacks_view, span<float const>(), parameters, weights_view,
                                   span<float>(device_scores.data(), device_scores.size()), executor, gpu_specs)
                   .status == status_t::success_k);

        for (std::size_t index = 0; index < texts.size(); ++index) {
            verify(std::isfinite(device_scores[index]) && "A score must be a number in every regime");
            verify(std::fabs(device_scores[index] - host_scores[index]) <=
                       1e-4f * std::fabs(host_scores[index]) + 1e-6f &&
                   "Device scores must agree with the serial engine at every dictionary width");
        }

        // One needle weighted alone, so a count split between the table and the overflow row reads as
        // `5 * term(1)` against `term(5)` - which the relative comparison above would hide at this width.
        {
            std::size_t const watched = needle_count - 1;
            std::string text;
            for (std::size_t index = 0; index < needle_count; ++index) text += needle_at(index) + " ";
            for (std::size_t repeat = 0; repeat < 4; ++repeat) text += needle_at(watched) + " ";

            unified_texts_t const watched_staged {{text}};

            unified_vector<float> lone_weights(needle_count, 0.0f);
            lone_weights[watched] = 1.0f;
            unified_vector<float> lone_score(1, 0.0f);
            verify(cuda_engine
                       .try_score_bm25(watched_staged.view(), span<float const>(), parameters,
                                       span<float const>(lone_weights.data(), lone_weights.size()),
                                       span<float>(lone_score.data(), lone_score.size()), executor, gpu_specs)
                       .status == status_t::success_k);

            float const normalized = 1.0f - parameters.length_normalization +
                                     parameters.length_normalization * (float)text.size() /
                                         parameters.average_document_length;
            float const expected = 5.0f * (parameters.term_frequency_saturation + 1.0f) /
                                   (5.0f + parameters.term_frequency_saturation * normalized);
            verify(std::fabs(lone_score[0] - expected) <= 1e-5f * expected &&
                   "Five occurrences of one needle must sum into one slot, not split across two");
        }

        // A hashed table is filled by racing lanes, so the seating order differs run to run; fixed-point
        // accumulation is what makes the total independent of it.
        for (std::size_t repeat = 0; repeat < 3; ++repeat) {
            unified_vector<float> repeated(texts.size(), 0.0f);
            verify(cuda_engine
                       .try_score_bm25(haystacks_view, span<float const>(), parameters, weights_view,
                                       span<float>(repeated.data(), repeated.size()), executor, gpu_specs)
                       .status == status_t::success_k);
            for (std::size_t index = 0; index < texts.size(); ++index)
                verify(repeated[index] == device_scores[index] && "Device scores must repeat bit for bit");
        }
#endif // SZ_USE_CUDA
    }

    // Full length normalization against an empty haystack leaves the closed form at `0/0`. Nothing is counted,
    // so no term is ever evaluated, and the score is a plain zero rather than a NaN that would poison a rank.
    {
        std::vector<std::string> const needle_strings {"cat", "dog"};
        std::vector<std::string> const haystack_strings {""};
        arrow_strings_tape_t needles, haystacks;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);

        substrings_bm25_t const degenerate {.term_frequency_saturation = 1.2f,
                                            .length_normalization = 1.0f,
                                            .average_document_length = 6.0f};
        unified_vector<float> const weights(2, 1.0f);
        span<float const> const weights_view {weights.data(), weights.size()};

        substrings_serial_t serial_engine;
        verify(serial_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
        std::vector<float> host_scores(1, 1.0f);
        verify(serial_engine.try_score_bm25(haystacks.view(), span<float const>(), degenerate, weights_view,
                                            span<float>(host_scores.data(), host_scores.size())) ==
               status_t::success_k);
        verify(host_scores[0] == 0.0f && "An empty haystack scores zero, not a NaN");

#if SZ_USE_CUDA
        gpu_specs_t gpu_specs;
        verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
        cuda_executor_t executor;

        unified_texts_t const empty_staged {{std::string {}}};

        substrings_cuda_t cuda_engine;
        verify(cuda_engine.try_index(needles.view(), substrings_cased_k, executor, gpu_specs) == status_t::success_k);
        unified_vector<float> device_scores(1, 1.0f);
        verify(cuda_engine
                   .try_score_bm25(empty_staged.view(), span<float const>(), degenerate, weights_view,
                                   span<float>(device_scores.data(), device_scores.size()), executor, gpu_specs)
                   .status == status_t::success_k);
        verify(device_scores[0] == 0.0f && "An empty haystack scores zero on the device too");
#endif // SZ_USE_CUDA
    }
}

/**
 *  @brief Rewriting and scoring on the GPU, against the serial engine and against the identity oracle.
 *
 *  The oracle carries the rewrite's own proof - replacing every needle with itself must reproduce the input
 *  byte for byte - so a device kernel is checked without a second device implementation to check it against.
 *  Scores are compared within a tolerance across backends and bit for bit against the device itself, which is
 *  the pair of promises the header makes - see the reduction note beside the comparison below.
 */
void test_substrings_cuda_equivalence() {
    std::printf("  - testing CUDA rewriting and BM25 against the serial engine...\n");
#if SZ_USE_CUDA

    gpu_specs_t gpu_specs;
    verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
    cuda_executor_t executor;

    std::vector<std::string> const needle_strings {"cat", "at", "concat", "dog", "the"};
    std::vector<std::string> const replacement_strings {"[CAT]", "@", "<<CONCAT>>", "", "THE"};
    arrow_strings_tape_t needles, replacements, identity;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    verify(replacements.try_assign(replacement_strings.data(),
                                   replacement_strings.data() + replacement_strings.size()) == status_t::success_k);
    verify(identity.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);

    std::vector<std::string> texts;
    for (std::size_t index = 0; index < scale_iterations(24); ++index)
        texts.push_back("the cat sat on a mat, concatenating cats at the dog " + std::to_string(index));

    // Unified, because a kernel cannot reach the caller's host memory - the contract every CUDA entry keeps.
    unified_texts_t const staged {texts};
    span<span<char const> const> const haystacks_view = staged.view();
    arrow_strings_tape_t reference_haystacks;
    verify(reference_haystacks.try_assign(texts.data(), texts.data() + texts.size()) == status_t::success_k);

    for (substrings_case_sensitivity_t sensitivity : {substrings_cased_k, substrings_uncased_k}) {
        substrings_serial_t serial_engine;
        substrings_cuda_t cuda_engine;
        verify(serial_engine.try_index(needles.view(), sensitivity) == status_t::success_k);
        verify(cuda_engine.try_index(needles.view(), sensitivity, executor, gpu_specs) == status_t::success_k);

        for (substrings_overlap_policy_t policy : {substrings_leftmost_longest_k, substrings_leftmost_first_k}) {
            substrings_rewrite_tape_t const on_host = rewrite_all_(serial_engine, reference_haystacks.view(), policy,
                                                                   replacements);
            substrings_rewrite_tape_t const on_device = rewrite_all_(cuda_engine, haystacks_view, policy, replacements,
                                                                     executor, gpu_specs);
            verify(on_device.offsets == on_host.offsets && "Rewritten boundaries must agree with the serial engine");
            verify(on_device.text == on_host.text && "Rewritten bytes must agree with the serial engine");

            // Replacing every needle with itself is the identity, whatever the cover resolved to.
            substrings_rewrite_tape_t const unchanged = rewrite_all_(cuda_engine, haystacks_view, policy, identity,
                                                                     executor, gpu_specs);
            std::string packed;
            for (std::string const &text : texts) packed += text;
            verify(tape_equals_(unchanged, packed) && "Replacing each needle with itself must reproduce the input");
        }

        unified_vector<float> weights(needle_strings.size(), 1.5f);
        unified_vector<float> device_scores(texts.size());
        std::vector<float> host_scores(texts.size());
        substrings_bm25_t parameters;
        parameters.average_document_length = 48.0f;
        verify(serial_engine.try_score_bm25(reference_haystacks.view(), span<float const>(), parameters,
                                            {weights.data(), weights.size()},
                                            {host_scores.data(), host_scores.size()}) == status_t::success_k);
        verify(cuda_engine.try_score_bm25(
                   haystacks_view, span<float const>(), parameters, {weights.data(), weights.size()},
                   {device_scores.data(), device_scores.size()}, executor, gpu_specs) == status_t::success_k);
        // Bit-stability is promised per backend, not across two of them: both reduce in ascending needle
        // order, but `nvcc` contracts `score + weight * term` into an FMA where the host compiler need not,
        // which moves the last ulp. So the backends are compared numerically, and the device is compared
        // against itself for the exact reproducibility the header does promise.
        for (std::size_t index = 0; index < texts.size(); ++index)
            verify(std::fabs(device_scores[index] - host_scores[index]) <=
                       1e-5f * std::fabs(host_scores[index]) + 1e-6f &&
                   "Device scores must agree with the serial engine");

        unified_vector<float> repeated(texts.size());
        for (std::size_t repeat = 0; repeat < 3; ++repeat) {
            std::fill(repeated.begin(), repeated.end(), 0.0f);
            verify(cuda_engine.try_score_bm25(haystacks_view, span<float const>(), parameters,
                                              {weights.data(), weights.size()}, {repeated.data(), repeated.size()},
                                              executor, gpu_specs) == status_t::success_k);
            for (std::size_t index = 0; index < texts.size(); ++index)
                verify(repeated[index] == device_scores[index] &&
                       "Scores must be bit-identical across runs of one backend");
        }
    }
#endif // SZ_USE_CUDA
}

#pragma endregion // Scoring

#pragma region Safety

/**
 *  @brief The three degenerate output-buffer shapes: an empty batch, an undersized buffer, and an oversized
 *         one. The cases above always size `matches` to exactly the count `try_count` produced.
 */
void test_substrings_buffer_safety() {
    std::printf("  - testing empty, undersized and oversized output buffers...\n");

    std::vector<std::string> const needle_strings {"he", "she", "his", "hers"};
    std::vector<std::string> const haystack_strings {"ushers", "hishers"};
    arrow_strings_tape_t needles, haystacks;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
           status_t::success_k);

    substrings_serial_t serial_engine;
    verify(serial_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);
    substrings_parallel_t parallel_engine;
    verify(parallel_engine.try_index(needles.view(), substrings_cased_k) == status_t::success_k);

    std::vector<std::size_t> counts(haystacks.view().size(), 0);
    std::size_t required = 0;
    verify(serial_engine.try_count(haystacks.view(), substrings_overlapping_k,
                                   span<std::size_t>(counts.data(), counts.size()), required) == status_t::success_k);
    verify(required > 1 && "fixtures must produce several matches for the undersized case to mean anything");

    std::size_t matches_found = 0;

    // An empty batch must be accepted and write nothing, rather than dereferencing an unallocated offsets
    // array while seeding its first prefix-sum entry.
    {
        std::vector<span<char const>> const no_haystacks;
        span<substrings_match_t> const no_matches;
        verify(parallel_engine.try_find(no_haystacks, substrings_overlapping_k, no_matches, matches_found) ==
               status_t::success_k);
        verify(matches_found == 0);
    }

    // An undersized buffer must be refused before anything is written, on both backends, and the reported
    // count must name the capacity the call wanted. The status already distinguishes a refusal from a
    // success, so reporting the need costs no ambiguity and saves the caller a counting pass.
    {
        std::vector<substrings_match_t> too_small(required - 1);
        span<substrings_match_t> const too_small_view(too_small.data(), too_small.size());
        verify(serial_engine.try_find(haystacks.view(), substrings_overlapping_k, too_small_view, matches_found) ==
               status_t::unexpected_dimensions_k);
        verify(matches_found == required);
        verify(parallel_engine.try_find(haystacks.view(), substrings_overlapping_k, too_small_view, matches_found) ==
               status_t::unexpected_dimensions_k);
        verify(matches_found == required);
    }

    // The same refusal with no buffer at all is the canonical size query: one call, no allocation, and the
    // need comes back in `matches_found`.
    {
        span<substrings_match_t> const no_capacity;
        verify(serial_engine.try_find(haystacks.view(), substrings_overlapping_k, no_capacity, matches_found) ==
               status_t::unexpected_dimensions_k);
        verify(matches_found == required);
        verify(parallel_engine.try_find(haystacks.view(), substrings_overlapping_k, no_capacity, matches_found) ==
               status_t::unexpected_dimensions_k);
        verify(matches_found == required);
    }

    // An oversized buffer is legal: `matches` supplies capacity, and `matches_found` states how much of it
    // was actually used.
    {
        std::vector<substrings_match_t> too_large(required + 16);
        span<substrings_match_t> const too_large_view(too_large.data(), too_large.size());
        verify(serial_engine.try_find(haystacks.view(), substrings_overlapping_k, too_large_view, matches_found) ==
               status_t::success_k);
        verify(matches_found == required);
        verify(parallel_engine.try_find(haystacks.view(), substrings_overlapping_k, too_large_view, matches_found) ==
               status_t::success_k);
        verify(matches_found == required);
    }
}

/**
 *  @brief Malformed-needle rejection over every malformed-input class - exact mode accepts any byte sequence,
 *         uncased mode accepts a needle exactly when it is structurally well-formed UTF-8 - then a sweep of
 *         malformed, mutated haystacks against a well-formed dictionary, where `try_count` and `try_find`
 *         must agree on the match count without crashing or hanging.
 */
void test_substrings_safety() {
    std::printf("  - testing malformed-needle rejection and malformed-haystack robustness...\n");

    auto &generator = global_random_generator();

    // The malformed pool mixes ill-formed byte sequences with noncharacters, which are reserved codepoints but
    // structurally well-formed UTF-8; only the former must be rejected. `sz_rune_decode` decides which is which,
    // so this tracks structural well-formedness rather than assuming every sample is malformed. The sweep walks
    // every class rather than sampling, so both outcomes are reached at any `SZ_TESTS_MULTIPLIER`.
    sz::span<char const *const> const malformed_pool = malformed_classes_();
    bool saw_rejection = false, saw_acceptance = false;
    for (std::size_t index = 0; index < malformed_pool.size(); ++index) {
        std::string const malformed {malformed_pool[index]};
        std::vector<span<char const>> const needles {span<char const>(malformed.data(), malformed.size())};

        substrings_serial_t exact_engine;
        verify(exact_engine.try_index(needles, substrings_cased_k) == status_t::success_k &&
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

        substrings_serial_t uncased_engine;
        status_t const uncased_status = uncased_engine.try_index(needles, substrings_uncased_k);
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
        substrings_serial_t engine;
        verify(engine.try_index(needles, substrings_uncased_k) == status_t::success_k &&
               "A well-formed UTF-8 needle must be accepted under case folding");
    }

    // Malformed and mutated haystacks never crash or hang a well-formed uncased dictionary, and `try_count`
    // and `try_find` always agree on how many matches there were.
    {
        std::vector<std::string> const needle_strings {"ss", "\xC3\x9F", "the", "K"};
        arrow_strings_tape_t needles;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        substrings_serial_t engine;
        verify(engine.try_index(needles.view(), substrings_uncased_k) == status_t::success_k);

        span<string_view const> const empty_motifs;
        for (std::size_t index = 0; index < scale_iterations(80); ++index) {
            std::string haystack;
            utf8_random_segmentation_corpus_(haystack, 128, utf8_corpus_flavor_t::malformed_k, utf8_default_alphabet,
                                             empty_motifs, generator);
            apply_mutation_passes_(haystack, generator);
            std::vector<span<char const>> const haystacks {span<char const>(haystack.data(), haystack.size())};

            std::vector<std::size_t> counts(1, 0);
            std::size_t matches_total = 0;
            verify(engine.try_count(haystacks, substrings_overlapping_k,
                                    span<std::size_t>(counts.data(), counts.size()),
                                    matches_total) == status_t::success_k);
            std::vector<substrings_match_t> matches(matches_total);
            std::size_t matches_found = 0;
            verify(engine.try_find(haystacks, substrings_overlapping_k,
                                   span<substrings_match_t>(matches.data(), matches.size()),
                                   matches_found) == status_t::success_k);
            verify(matches_found == counts[0] && "try_count and try_find disagree on match count");
        }
    }
}

#pragma endregion // Safety

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
