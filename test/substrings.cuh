/**
 *  @brief Extensive @b stress-testing suite for the StringZillas multi-pattern search engine (Aho-Corasick).
 *  @see Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file scripts/test_substrings.cuh
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
#include "utf8.hpp"        // `malformed_classes_`, `utf8_random_segmentation_corpus_`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

using ashvardanian::stringzillas::dummy_executor_t;
using ashvardanian::stringzillas::fold_preimage_of_rune;
using ashvardanian::stringzillas::fold_preimage_of_runes;
using ashvardanian::stringzillas::substrings_cased_k;
using ashvardanian::stringzillas::forkunion_executor_t;
using ashvardanian::stringzillas::substrings_match_t;
using ashvardanian::stringzillas::substrings_case_sensitivity_t;
using ashvardanian::stringzillas::substrings_u16_dictionary_t;
using ashvardanian::stringzillas::substrings_u32_dictionary_t;
using ashvardanian::stringzillas::substrings_u32_parallel_t;
using ashvardanian::stringzillas::substrings_u32_serial_t;
using ashvardanian::stringzillas::substrings_uncased_k;

#if SZ_USE_CUDA
using ashvardanian::stringzillas::cuda_executor_t;
using ashvardanian::stringzillas::substrings_u32_cuda_t;
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
 *  @brief Runs a full count-then-find pass through @p engine and reduces every match to its identity in
 *         @p out, whose own buffers hold the intermediates. Every differential check in this file is driven
 *         through this one public-API shape.
 *  @note `trailing_args_` forwards an executor (and optionally specs) to backends that want a specific one;
 *        an empty pack leaves each engine on its own defaults.
 */
template <typename engine_type_, typename haystacks_type_, typename... trailing_args_>
void collect_matches_into_(engine_type_ &engine, haystacks_type_ const &haystacks, substrings_match_set_t &out,
                           trailing_args_ &&...trailing_args) {
    out.collected_counts.assign(haystacks.size(), 0);
    std::size_t matches_total = 0;
    verify(engine.try_count(haystacks, span<std::size_t>(out.collected_counts.data(), out.collected_counts.size()),
                            matches_total, trailing_args...) == status_t::success_k);

    // Sized to exactly what counting promised, so a `try_find` overrun trips instead of writing into slack.
    out.collected_matches.assign(matches_total, substrings_match_t {});
    std::size_t matches_found = 0;
    verify(engine.try_find(haystacks,
                           span<substrings_match_t>(out.collected_matches.data(), out.collected_matches.size()),
                           matches_found, trailing_args...) == status_t::success_k);
    verify(matches_found == matches_total && "try_count and try_find disagree on the match count");

    out.clear();
    for (substrings_match_t const &match : out.collected_matches) out.append(match);
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

    std::vector<span<char const>> matches;
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
        matches.emplace_back(haystack.data() + source_begin[start],
                             source_end[start + needle_length - 1] - source_begin[start]);
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
            if (needle.size() == 0 || needle.size() > haystack.size()) continue;

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

/** @brief One fold-space rune drawn uniformly from the narrow preimage table's images within
 *         `[first_rune, last_rune)`; the image array is sorted, so the range is a binary-searched slab. */
inline sz_rune_t sample_fold_image_rune_(sz_rune_t first_rune, sz_rune_t last_rune) {
    sz_u32_t const *const images_begin = szs_fold_preimage_narrow_images_;
    sz_u32_t const *const images_end = images_begin + szs_fold_preimage_narrow_count_k;
    sz_u32_t const *const slab_begin = std::lower_bound(images_begin, images_end, (sz_u32_t)first_rune);
    sz_u32_t const *const slab_end = std::lower_bound(images_begin, images_end, (sz_u32_t)last_rune);
    verify(slab_begin != slab_end && "The requested rune range holds no fold images");
    std::uniform_int_distribution<std::size_t> slab_position(0, (std::size_t)(slab_end - slab_begin) - 1);
    std::size_t const chosen_position = slab_position(global_random_generator());
    return (sz_rune_t)slab_begin[chosen_position];
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
            preimages = fold_preimage_of_runes({folded.data() + position, 3}), consumed_runes = 3;
        if (preimages.size() == 0 && remaining_runes >= 2)
            preimages = fold_preimage_of_runes({folded.data() + position, 2}), consumed_runes = 2;
        if (preimages.size() == 0) //
            preimages = fold_preimage_of_rune(folded[position]), consumed_runes = 1;
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
        // Kelvin sign, so `max_match_bytes` cannot be read off needle lengths. The Kelvin anchor stays
        // pinned; every following needle is a wide-table image repeated 1-4 times, so 2:1 and 3:1 byte
        // expansions from every script appear organically rather than from a hand-picked seed list.
        std::vector<std::string> pool {"k"};
        auto &generator = global_random_generator();
        std::uniform_int_distribution<std::size_t> row_position(0, szs_fold_preimage_wide_count_k - 1);
        for (std::size_t index = 0; index < count; ++index) {
            std::size_t const chosen_row = row_position(generator);
            sz_u32_t const *const image = szs_fold_preimage_wide_images_[chosen_row];
            scratch.clear();
            for (std::size_t repeat = 0; repeat <= index % 4; ++repeat)
                for (std::size_t rune_index = 0; rune_index < 3 && image[rune_index] != 0; ++rune_index)
                    append_rune_utf8_((sz_rune_t)image[rune_index], scratch);
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

        substrings_u32_serial_t engine;
        verify(engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

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

        substrings_u32_serial_t engine;
        verify(engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

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

        substrings_u32_serial_t engine;
        verify(engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

        substrings_match_set_t const expected {
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

        substrings_u32_serial_t engine;
        verify(engine.try_build(needles.view(), substrings_uncased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

        substrings_match_set_t const expected {
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

    substrings_u32_serial_t engine;
    verify(engine.try_build(needles, substrings_uncased_k) == status_t::success_k);

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
 *  @brief Full Unicode case-folding conformance table for `substrings_uncased_k`, then a differential against
 *         `independent_uncased_matches_`.
 *
 *  Every non-ASCII fixture is spelled as `\xHH` byte escapes, never as a raw literal or a `\u` universal
 *  character name - both have been silently re-encoded by tooling here before. A byte-level check against a
 *  numeric expected-bytes array guards the trickiest fixtures.
 */
void test_substrings_uncased() {
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
    // `substrings` must equal the full match set found by the independent fold-and-scan oracle.
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
        substrings_u32_serial_t engine;
        verify(engine.try_build(needles, substrings_uncased_k) == status_t::success_k);
        std::vector<std::string> const haystack_strings {haystack};
        arrow_strings_tape_t haystacks;
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        substrings_match_set_t engine_matches;
        collect_matches_into_(engine, haystacks.view(), engine_matches);

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

#pragma endregion // Uncased Conformance

#pragma region Agreement

/**
 *  @brief A ONE-needle dictionary must agree exactly with the shipped `sz_utf8_uncased_search` over a fuzz
 *         corpus - that agreement is the entire semantic claim of `substrings_uncased_k`.
 *
 *  `substrings` reports every overlapping match, while `sz_utf8_uncased_search` reports only the first by
 *  start offset, so the comparison reduces `substrings`'s set to its own earliest-starting match.
 */
void test_substrings_agreement() {
    std::printf("  - testing one-needle agreement with sz_utf8_uncased_search...\n");

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
        substrings_u32_serial_t engine;
        verify(engine.try_build(needles, substrings_uncased_k) == status_t::success_k);
        std::vector<std::string> const haystack_strings {haystack};
        arrow_strings_tape_t haystacks;
        verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
               status_t::success_k);
        substrings_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);

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

    substrings_u32_serial_t engine;
    verify(engine.try_build(needles_view, sensitivity) == status_t::success_k);
    collect_matches_into_(engine, haystacks_view, scratch.engine_keys);
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

    substrings_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles_view, sensitivity) == status_t::success_k);
    collect_matches_into_(serial_engine, haystacks_view, scratch.engine_keys);

    substrings_u32_parallel_t parallel_engine;
    verify(parallel_engine.try_build(needles_view, sensitivity) == status_t::success_k);
    collect_matches_into_(parallel_engine, haystacks_view, scratch.variant_keys, pool, scratch.specs);
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
 */
void check_substrings_narrow_width_(substrings_case_sensitivity_t sensitivity,
                                    substrings_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    substrings_u16_dictionary_t narrow;
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
        log_substrings_cell_mismatch_(scratch, "Narrow-width divergence");
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
/** @brief The device backend must report the identical set for the same cell, through the very same call
 *         shape every host backend takes - only the memory the haystacks live in differs. */
void check_substrings_cuda_agrees_(substrings_case_sensitivity_t sensitivity,
                                   substrings_adversarial_scratch_t &scratch) {
    arrow_strings_view_t const needles_view = scratch.needles.view();
    arrow_strings_view_t const haystacks_view = scratch.haystacks.view();

    substrings_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles_view, sensitivity) == status_t::success_k);
    collect_matches_into_(serial_engine, haystacks_view, scratch.engine_keys);

    gpu_specs_t gpu_specs;
    verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
    cuda_executor_t executor;
    substrings_u32_cuda_t cuda_engine;
    verify(cuda_engine.try_build(needles_view, sensitivity, executor) == status_t::success_k);

    // One match type and one signature across backends: the device runs the very same collection body as
    // every CPU engine. The fixture tape is `unified_alloc`-backed under CUDA, so no copy exists anywhere.
    collect_matches_into_(cuda_engine, haystacks_view, scratch.cuda_keys, executor, gpu_specs);
    if (scratch.cuda_keys != scratch.engine_keys) {
        log_substrings_cell_mismatch_(scratch, "CUDA-vs-serial divergence");
        verify(false && "CUDA backend disagrees with the serial reference");
    }
}

/**
 *  @brief Pins the two sides of the device memory contract: scattered device-resident haystacks - several
 *         separate unified allocations rather than one packed tape - are searched correctly, and host-backed
 *         haystacks are refused with `device_memory_mismatch_k` rather than silently copied.
 */
void test_substrings_cuda_memory_contract() {
    std::printf("  - testing scattered unified haystacks and the host-memory refusal...\n");

    gpu_specs_t gpu_specs;
    verify(gpu_specs_fetch(gpu_specs) == status_t::success_k);
    cuda_executor_t executor;

    std::vector<std::string> const needle_strings {"he", "she", "his", "hers"};
    arrow_strings_tape_t needles;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    substrings_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);
    substrings_u32_cuda_t cuda_engine;
    verify(cuda_engine.try_build(needles.view(), substrings_cased_k, executor) == status_t::success_k);

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
    substrings_match_set_t serial_keys, scattered_keys;
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
 *  @brief Crosses every needle vocabulary with every placement skeleton and every needle transform.
 *
 *  Walked by rotation rather than nested loops: `rotating_index` advances a phase each full turn, so
 *  crossing the rotations still reaches every combination, and `scale_iterations` decides how far into that
 *  product a run gets. Ground truth and the declared planting effects run on every cell; the costlier
 *  properties rotate, keeping the per-cell price flat while the sweep as a whole still exercises each of
 *  them against each axis triple.
 */
void test_substrings_adversarial() {
    std::printf("  - testing adversarial needle x placement x transform cross-product...\n");
    std::size_t const needle_generators = (std::size_t)substrings_needle_generator_t::count_k;
    std::size_t const placements = (std::size_t)substrings_placement_t::count_k;
    std::size_t const transforms = (std::size_t)substrings_transform_t::count_k;
    std::size_t const cells = scale_iterations(needle_generators * placements * transforms);
    substrings_adversarial_scratch_t scratch; // ? Constructed once, refilled by every cell below.

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
        case 1: check_substrings_narrow_width_(sensitivity, scratch); break;
        case 2: check_substrings_tier_invariant_(sensitivity, scratch); break;
        }
#if SZ_USE_CUDA
        check_substrings_cuda_agrees_(sensitivity, scratch);
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
void test_substrings_large_haystacks() {
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

    substrings_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);
    substrings_match_set_t serial_keys, parallel_keys;
    collect_matches_into_(serial_engine, haystacks.view(), serial_keys);

    substrings_u32_parallel_t parallel_engine;
    verify(parallel_engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);
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
void test_substrings_construction() {
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
        substrings_u32_serial_t engine;
        verify(engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);
        substrings_match_set_t matches;
        collect_matches_into_(engine, haystacks.view(), matches);
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
        substrings_u32_serial_t engine;
        verify(engine.try_build(needles.view(), substrings_uncased_k) == status_t::success_k);

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
        collect_matches_into_(engine, haystacks.view(), matches);

        std::vector<std::size_t> counts(haystacks.view().size(), 0);
        std::size_t matches_total = 0;
        verify(engine.try_count(haystacks.view(), span<std::size_t>(counts.data(), counts.size()), matches_total) ==
               status_t::success_k);
        std::vector<substrings_match_t> raw_matches(matches_total);
        std::size_t matches_found = 0;
        verify(engine.try_find(haystacks.view(), span<substrings_match_t>(raw_matches.data(), raw_matches.size()),
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

#pragma region Safety

/**
 *  @brief The three degenerate output-buffer shapes: an empty batch, an undersized buffer, and an oversized
 *         one. The cases above always size `matches` to exactly the count `try_count` produced.
 */
void test_substrings_buffer_contracts() {
    std::printf("  - testing empty, undersized and oversized output buffers...\n");

    std::vector<std::string> const needle_strings {"he", "she", "his", "hers"};
    std::vector<std::string> const haystack_strings {"ushers", "hishers"};
    arrow_strings_tape_t needles, haystacks;
    verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
           status_t::success_k);
    verify(haystacks.try_assign(haystack_strings.data(), haystack_strings.data() + haystack_strings.size()) ==
           status_t::success_k);

    substrings_u32_serial_t serial_engine;
    verify(serial_engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);
    substrings_u32_parallel_t parallel_engine;
    verify(parallel_engine.try_build(needles.view(), substrings_cased_k) == status_t::success_k);

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
        span<substrings_match_t> const no_matches;
        verify(parallel_engine.try_find(no_haystacks, no_matches, matches_found) == status_t::success_k);
        verify(matches_found == 0);
    }

    // An undersized buffer must be refused before anything is written, on both backends, and the reported
    // count must stay zero so a caller cannot mistake a refusal for a partial success.
    {
        std::vector<substrings_match_t> too_small(required - 1);
        span<substrings_match_t> const too_small_view(too_small.data(), too_small.size());
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
        std::vector<substrings_match_t> too_large(required + 16);
        span<substrings_match_t> const too_large_view(too_large.data(), too_large.size());
        verify(serial_engine.try_find(haystacks.view(), too_large_view, matches_found) == status_t::success_k);
        verify(matches_found == required);
        verify(parallel_engine.try_find(haystacks.view(), too_large_view, matches_found) == status_t::success_k);
        verify(matches_found == required);
    }
}

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

        substrings_u32_serial_t exact_engine;
        verify(exact_engine.try_build(needles, substrings_cased_k) == status_t::success_k &&
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

        substrings_u32_serial_t uncased_engine;
        status_t const uncased_status = uncased_engine.try_build(needles, substrings_uncased_k);
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
        substrings_u32_serial_t engine;
        verify(engine.try_build(needles, substrings_uncased_k) == status_t::success_k &&
               "A well-formed UTF-8 needle must be accepted under case folding");
    }

    // Malformed and mutated haystacks never crash or hang a well-formed uncased dictionary, and `try_count`
    // and `try_find` always agree on how many matches there were.
    {
        std::vector<std::string> const needle_strings {"ss", "\xC3\x9F", "the", "K"};
        arrow_strings_tape_t needles;
        verify(needles.try_assign(needle_strings.data(), needle_strings.data() + needle_strings.size()) ==
               status_t::success_k);
        substrings_u32_serial_t engine;
        verify(engine.try_build(needles.view(), substrings_uncased_k) == status_t::success_k);

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
            std::vector<substrings_match_t> matches(matches_total);
            std::size_t matches_found = 0;
            verify(engine.try_find(haystacks, span<substrings_match_t>(matches.data(), matches.size()),
                                   matches_found) == status_t::success_k);
            verify(matches_found == counts[0] && "try_count and try_find disagree on match count");
        }
    }
}

#pragma endregion // Safety

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
