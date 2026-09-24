/**
 *  @brief  Multi-pattern search tests: known answers, a brute-force oracle, and the three verbs against it.
 *  @file   test/substrings.cpp
 *  @author Ash Vardanian
 *  @date   September 20, 2026
 *
 *  The oracle is a naive scan of every needle at every offset, which is what an Aho-Corasick automaton has
 *  to agree with by construction. Under case folding the oracle folds both sides with the library's own
 *  `sz_utf8_uncased_fold` and searches the folded haystack, then snaps the span it found back onto whole
 *  source codepoints - so it tests the automaton rather than re-deriving the fold.
 */
#undef NDEBUG // ! Enable all assertions for testing

#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <cctype>  // `std::toupper`
#include <cmath>   // `std::fabs`
#include <cstdlib> // `std::malloc`, `std::free`
#include <cstring> // `std::memcmp`

#include <algorithm> // `std::sort`
#include <string>    // Baseline
#include <vector>    // `std::vector`

#include "stringzilla.hpp" // `global_random_generator`, `random_string`, `refusing_allocator_`, `verify`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

#pragma region Helpers

/** @brief One match as the oracle and the backends both spell it, so a differential is one comparison. */
struct substrings_case_t {
    std::size_t haystack_index;
    std::size_t needle_index;
    std::size_t byte_offset;
    std::size_t byte_length;

    bool operator==(substrings_case_t const &other) const noexcept {
        return haystack_index == other.haystack_index && needle_index == other.needle_index &&
               byte_offset == other.byte_offset && byte_length == other.byte_length;
    }
    bool operator<(substrings_case_t const &other) const noexcept {
        if (haystack_index != other.haystack_index) return haystack_index < other.haystack_index;
        if (byte_offset != other.byte_offset) return byte_offset < other.byte_offset;
        if (byte_length != other.byte_length) return byte_length < other.byte_length;
        return needle_index < other.needle_index;
    }
};

/** @brief Binds a sequence over a vector of strings, which is what every verb here takes. */
static sz_sequence_t sequence_over_(std::vector<std::string> const &strings, std::vector<sz_string_view_t> &views) {
    views.resize(strings.size());
    for (std::size_t index = 0; index != strings.size(); ++index)
        views[index].start = strings[index].data(), views[index].length = strings[index].size();
    sz_sequence_t sequence;
    sz_sequence_from_string_views(views.data(), views.size(), &sequence);
    return sequence;
}

/** @brief Folds @p text with the library's own folder, which is the stream an uncased automaton walks. */
static std::string folded_(std::string const &text) {
    std::string folded(text.size() * 4 + 4, '\0');
    std::size_t const written = sz_utf8_uncased_fold(text.data(), text.size(), &folded[0]);
    folded.resize(written);
    return folded;
}

/**
 *  @brief Where the source codepoint behind each folded byte starts and ends.
 *
 *  A folded match snaps outward onto whole codepoints, so a folded span `[first, last)` becomes the source
 *  span `[starts[first], ends[last - 1])`. Snapping the end onto the producing codepoint's own start
 *  instead would collapse a match ending inside an expansion to nothing.
 */
struct folded_origins_t {
    std::vector<std::size_t> starts; /**< Source offset the codepoint behind each folded byte begins at. */
    std::vector<std::size_t> ends;   /**< Source offset just past that codepoint. */
};

static folded_origins_t folded_origins_(std::string const &text) {
    folded_origins_t origins;
    std::size_t offset = 0;
    while (offset < text.size()) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode(text.data() + offset, text.data() + text.size(), &rune);
        std::size_t const source_length = consumed == sz_rune_invalid_k ? 1 : (std::size_t)consumed;
        std::string const image = folded_(text.substr(offset, source_length));
        for (std::size_t byte = 0; byte != image.size(); ++byte)
            origins.starts.push_back(offset), origins.ends.push_back(offset + source_length);
        offset += source_length;
    }
    return origins;
}

/** @brief Every match of every needle at every offset, which is what the automaton must agree with. */
static std::vector<substrings_case_t> oracle_overlapping_(std::vector<std::string> const &haystacks,
                                                          std::vector<std::string> const &needles,
                                                          sz_substrings_case_sensitivity_t sensitivity) {
    std::vector<substrings_case_t> found;
    for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
        std::string const &haystack = haystacks[haystack_index];
        if (sensitivity == sz_substrings_cased_k) {
            for (std::size_t needle_index = 0; needle_index != needles.size(); ++needle_index) {
                std::string const &needle = needles[needle_index];
                if (needle.empty() || needle.size() > haystack.size()) continue;
                for (std::size_t offset = 0; offset + needle.size() <= haystack.size(); ++offset)
                    if (std::memcmp(haystack.data() + offset, needle.data(), needle.size()) == 0)
                        found.push_back({haystack_index, needle_index, offset, needle.size()});
            }
            continue;
        }

        // The automaton walks folded bytes, so the oracle does too, and reports the source span the folded
        // one snaps outward onto. Distinct folded offsets can snap onto one source span, which is the
        // repeat the walk collapses, so the oracle collapses it here as well.
        std::string const folded_haystack = folded_(haystack);
        folded_origins_t const origins = folded_origins_(haystack);
        verify(origins.starts.size() == folded_haystack.size());
        for (std::size_t needle_index = 0; needle_index != needles.size(); ++needle_index) {
            std::string const folded_needle = folded_(needles[needle_index]);
            if (folded_needle.empty() || folded_needle.size() > folded_haystack.size()) continue;
            std::size_t previous_offset = haystack.size() + 1, previous_length = 0;
            for (std::size_t offset = 0; offset + folded_needle.size() <= folded_haystack.size(); ++offset) {
                if (std::memcmp(folded_haystack.data() + offset, folded_needle.data(), folded_needle.size()) != 0)
                    continue;
                std::size_t const source_offset = origins.starts[offset];
                std::size_t const source_end = origins.ends[offset + folded_needle.size() - 1];
                if (source_offset == previous_offset && source_end - source_offset == previous_length) continue;
                previous_offset = source_offset, previous_length = source_end - source_offset;
                found.push_back({haystack_index, needle_index, source_offset, previous_length});
            }
        }
    }
    std::sort(found.begin(), found.end());
    return found;
}

/** @brief The greedy cover the leftmost policies name, taken over the oracle's own overlapping matches. */
static std::vector<substrings_case_t> oracle_leftmost_(std::vector<substrings_case_t> const &overlapping,
                                                       std::size_t haystacks_count,
                                                       sz_substrings_overlap_policy_t policy) {
    std::vector<substrings_case_t> kept;
    for (std::size_t haystack_index = 0; haystack_index != haystacks_count; ++haystack_index) {
        std::size_t cursor = 0;
        for (;;) {
            substrings_case_t const *chosen = nullptr;
            for (substrings_case_t const &candidate : overlapping) {
                if (candidate.haystack_index != haystack_index || candidate.byte_offset < cursor) continue;
                if (!chosen) {
                    chosen = &candidate;
                    continue;
                }
                if (candidate.byte_offset != chosen->byte_offset) {
                    if (candidate.byte_offset < chosen->byte_offset) chosen = &candidate;
                    continue;
                }
                if (policy == sz_substrings_leftmost_longest_k && candidate.byte_length != chosen->byte_length) {
                    if (candidate.byte_length > chosen->byte_length) chosen = &candidate;
                    continue;
                }
                if (candidate.needle_index < chosen->needle_index) chosen = &candidate;
            }
            if (!chosen) break;
            kept.push_back(*chosen);
            cursor = chosen->byte_offset + chosen->byte_length;
        }
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}

/** @brief The rewrite that cover implies, spliced by the oracle rather than by the backend. */
static std::string oracle_rewrite_(std::string const &haystack, std::size_t haystack_index,
                                   std::vector<substrings_case_t> const &cover,
                                   std::vector<std::string> const &replacements) {
    std::string rewritten;
    std::size_t cursor = 0;
    for (substrings_case_t const &match : cover) {
        if (match.haystack_index != haystack_index) continue;
        rewritten.append(haystack, cursor, match.byte_offset - cursor);
        rewritten.append(replacements[match.needle_index]);
        cursor = match.byte_offset + match.byte_length;
    }
    rewritten.append(haystack, cursor, haystack.size() - cursor);
    return rewritten;
}

/** @brief The verbs one CPU tier exports, so every compiled tier meets the oracle, not only the dispatched one. */
struct substrings_tier_t {
    decltype(&sz_substrings_counts_serial) counts;
    decltype(&sz_substrings_find_serial) find;
    decltype(&sz_substrings_replace_serial) replace;
    decltype(&sz_substrings_bm25_scores_serial) bm25_scores;
};

static std::vector<substrings_tier_t> substrings_tiers_() {
    std::vector<substrings_tier_t> tiers {{&sz_substrings_counts_serial, &sz_substrings_find_serial,
                                           &sz_substrings_replace_serial, &sz_substrings_bm25_scores_serial}};
#if SZ_USE_HASWELL
    tiers.push_back({&sz_substrings_counts_haswell, &sz_substrings_find_haswell, &sz_substrings_replace_haswell,
                     &sz_substrings_bm25_scores_haswell});
#endif
#if SZ_USE_ICELAKE
    tiers.push_back({&sz_substrings_counts_icelake, &sz_substrings_find_icelake, &sz_substrings_replace_icelake,
                     &sz_substrings_bm25_scores_icelake});
#endif
#if SZ_USE_NEON
    tiers.push_back({&sz_substrings_counts_neon, &sz_substrings_find_neon, &sz_substrings_replace_neon,
                     &sz_substrings_bm25_scores_neon});
#endif
    return tiers;
}

/** @brief BM25 over the oracle's own overlapping matches, summed in double precision in any order. */
static std::vector<double> oracle_bm25_(std::vector<substrings_case_t> const &overlapping,
                                        std::vector<std::string> const &haystacks, std::size_t needles_count,
                                        std::vector<sz_f32_t> const &document_lengths,
                                        sz_substrings_bm25_t const &parameters, std::vector<sz_f32_t> const &weights) {
    std::vector<std::vector<std::size_t>> frequencies(haystacks.size(), std::vector<std::size_t>(needles_count));
    for (substrings_case_t const &match : overlapping) ++frequencies[match.haystack_index][match.needle_index];
    std::vector<double> scores(haystacks.size(), 0);
    for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
        double const length = document_lengths.empty() ? (double)haystacks[haystack_index].size()
                                                       : (double)document_lengths[haystack_index];
        double const normalization = parameters.length_normalization;
        double const saturation = parameters.term_frequency_saturation;
        double const norm = normalization > 0
                                ? 1 - normalization + normalization * length / parameters.average_document_length
                                : 1;
        for (std::size_t needle_index = 0; needle_index != needles_count; ++needle_index) {
            double const frequency = (double)frequencies[haystack_index][needle_index];
            if (frequency)
                scores[haystack_index] += weights[needle_index] * frequency * (saturation + 1) /
                                          (frequency + saturation * norm);
        }
    }
    return scores;
}

/** @brief The dispatched BM25 against the oracle, by byte lengths and by caller-given ones, with and without
 *         length normalization. */
static void check_bm25_(substrings_tier_t const &tier, sz_substrings_engine_t *engine,
                        sz_sequence_t const *haystack_sequence, std::vector<std::string> const &haystacks,
                        std::vector<substrings_case_t> const &overlapping) {
    std::size_t const needles_count = engine->needles_count;
    std::vector<sz_f32_t> weights(needles_count), given_lengths(haystacks.size());
    for (std::size_t index = 0; index != needles_count; ++index) weights[index] = 0.5f + (float)(index % 7) * 0.25f;
    double bytes_total = 0, given_total = 0;
    for (std::size_t index = 0; index != haystacks.size(); ++index) {
        given_lengths[index] = (sz_f32_t)(1 + index % 5);
        bytes_total += haystacks[index].size(), given_total += given_lengths[index];
    }

    for (std::vector<sz_f32_t> const &lengths : {std::vector<sz_f32_t>(), given_lengths}) {
        double const average = (lengths.empty() ? bytes_total : given_total) / haystacks.size();
        sz_substrings_bm25_t const normalized {1.2f, 0.75f, (sz_f32_t)(average > 0 ? average : 1)};
        sz_substrings_bm25_t const unnormalized {1.2f, 0.0f, 0.0f};
        for (sz_substrings_bm25_t const &parameters : {normalized, unnormalized}) {
            std::vector<double> const expected = oracle_bm25_(overlapping, haystacks, needles_count, lengths,
                                                              parameters, weights);
            std::vector<sz_f32_t> scores(haystacks.size(), -1);
            verify(tier.bm25_scores(engine, haystack_sequence, lengths.empty() ? nullptr : lengths.data(), &parameters,
                                    weights.data(), scores.data(), 1) == sz_success_k);
            for (std::size_t index = 0; index != haystacks.size(); ++index)
                verify(std::fabs(scores[index] - expected[index]) <= 1e-5 * std::max(1.0, std::fabs(expected[index])));
        }
    }
}

/** @brief Compiles @p needles, so a refusal can be asserted on without naming a sequence of its own. */
static sz_status_t build_over_(std::vector<std::string> const &needles, sz_substrings_case_sensitivity_t sensitivity,
                               sz_memory_allocator_t *alloc, sz_substrings_engine_t *engine) {
    std::vector<sz_string_view_t> views;
    sz_sequence_t const sequence = sequence_over_(needles, views);
    return sz_substrings_engine_init_cpu(&sequence, sensitivity, sz_substrings_overlapping_k,
                                         SZ_SUBSTRINGS_HOT_STATES_AUTO, 0, alloc, engine);
}

/** An allocator granting its first @c grants requests and refusing the rest, tallying the bytes it holds. */
struct rationed_allocator_t {
    std::size_t grants {};              /**< Requests still to be granted before every later one is refused. */
    std::size_t bytes_held {};          /**< Bytes handed out and not yet returned. */
    sz_memory_allocator_t allocator {}; /**< The C-side view, whose handle points back at this object. */

    explicit rationed_allocator_t(std::size_t granted) noexcept : grants(granted) {
        allocator.allocate = +[](sz_size_t length, void *handle) -> void * {
            rationed_allocator_t &self = *static_cast<rationed_allocator_t *>(handle);
            if (!self.grants) return nullptr;
            void *const pointer = std::malloc(length ? length : 1);
            if (pointer) --self.grants, self.bytes_held += length;
            return pointer;
        };
        allocator.free = +[](void *pointer, sz_size_t length, void *handle) {
            static_cast<rationed_allocator_t *>(handle)->bytes_held -= length;
            std::free(pointer);
        };
        allocator.handle = this;
    }
    rationed_allocator_t(rationed_allocator_t const &) = delete;
    rationed_allocator_t &operator=(rationed_allocator_t const &) = delete;
};

/** Refuses the build's allocations one later each round, so every error path proves it frees what it took. */
static void check_build_refusals_(std::vector<std::string> const &needles,
                                  sz_substrings_case_sensitivity_t sensitivity) {
    for (std::size_t granted = 0;; ++granted) {
        rationed_allocator_t rationed(granted);
        sz_substrings_engine_t engine {};
        sz_status_t const status = build_over_(needles, sensitivity, &rationed.allocator, &engine);
        if (status == sz_success_k) {
            sz_substrings_engine_free(&engine);
            verify(rationed.bytes_held == 0);
            return;
        }
        verify(status == sz_bad_alloc_k);
        verify(engine.memory == nullptr && "A refused build leaves the engine untouched");
        verify(rationed.bytes_held == 0 && "A refused build returns everything it took");
    }
}

/** @brief Reads every match the engine reports, sizing the array from the report the sizing call leaves. */
static std::vector<substrings_case_t> backend_find_(substrings_tier_t const &tier, sz_substrings_engine_t *engine,
                                                    sz_sequence_t const *haystacks) {
    std::vector<sz_size_t> offsets(haystacks->count + 1, 0);
    verify(tier.find(engine, haystacks, nullptr, 0, offsets.data()) == sz_success_k);
    sz_size_t const total = engine->report->matches_emitted;
    verify(offsets[haystacks->count] == total);
    verify(engine->report->shortfall == total);

    std::vector<sz_substrings_match_t> matches(total);
    verify(tier.find(engine, haystacks, matches.data(), matches.size(), offsets.data()) == sz_success_k);
    verify(engine->report->matches_stored == total && engine->report->shortfall == 0);

    std::vector<substrings_case_t> reported(total);
    for (std::size_t index = 0; index != total; ++index)
        reported[index] = {matches[index].haystack_index, matches[index].needle_index, matches[index].byte_offset,
                           matches[index].byte_length};
    std::sort(reported.begin(), reported.end());
    return reported;
}

/** @brief One vocabulary against one corpus under one policy, compared with the oracle on all three verbs. */
static void check_corpus_(std::vector<std::string> const &haystacks, std::vector<std::string> const &needles,
                          sz_substrings_case_sensitivity_t sensitivity, sz_substrings_overlap_policy_t policy,
                          std::size_t hot_states = SZ_SUBSTRINGS_HOT_STATES_AUTO) {
    sz_memory_allocator_t alloc;
    sz_memory_allocator_init_default(&alloc);
    std::vector<sz_string_view_t> haystack_views, needle_views, replacement_views;
    sz_sequence_t const haystack_sequence = sequence_over_(haystacks, haystack_views);
    sz_sequence_t const needle_sequence = sequence_over_(needles, needle_views);

    sz_substrings_engine_t engine;
    verify(sz_substrings_engine_init_cpu(&needle_sequence, sensitivity, policy, hot_states, 0, &alloc, &engine) ==
           sz_success_k);
    verify(engine.needles_count == needles.size());
    verify(engine.root == 0);

    std::vector<substrings_case_t> const overlapping = oracle_overlapping_(haystacks, needles, sensitivity);
    std::vector<substrings_case_t> const expected = policy == sz_substrings_overlapping_k
                                                        ? overlapping
                                                        : oracle_leftmost_(overlapping, haystacks.size(), policy);

    for (substrings_tier_t const &tier : substrings_tiers_()) {
        // Every match, located the same way the oracle locates it.
        std::vector<substrings_case_t> const reported = backend_find_(tier, &engine, &haystack_sequence);
        verify(reported.size() == expected.size());
        for (std::size_t index = 0; index != reported.size(); ++index) verify(reported[index] == expected[index]);

        // The per-haystack counts, which a separate walk answers and so can disagree with the matches.
        std::vector<sz_size_t> counts(haystacks.size(), 0);
        verify(tier.counts(&engine, &haystack_sequence, counts.data(), 1) == sz_success_k);
        for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
            std::size_t owned = 0;
            for (substrings_case_t const &match : expected) owned += match.haystack_index == haystack_index;
            verify(counts[haystack_index] == owned);
        }

        // BM25 reads raw overlapping frequencies, so it is checked once per corpus rather than per policy.
        if (policy == sz_substrings_overlapping_k)
            check_bm25_(tier, &engine, &haystack_sequence, haystacks, overlapping);

        // The rewrite, which only a cover admits.
        if (policy != sz_substrings_overlapping_k) {
            std::vector<std::string> replacements;
            for (std::size_t needle_index = 0; needle_index != needles.size(); ++needle_index)
                replacements.push_back(needle_index % 3 == 0 ? std::string()
                                                             : "<" + std::to_string(needle_index) + ">");
            sz_sequence_t const replacement_sequence = sequence_over_(replacements, replacement_views);

            std::vector<sz_size_t> offsets(haystacks.size() + 1, 0);
            verify(tier.replace(&engine, &haystack_sequence, &replacement_sequence, nullptr, 0, offsets.data()) ==
                   sz_success_k);
            verify(engine.report->tape_bytes == offsets[haystacks.size()]);

            std::vector<char> tape(offsets[haystacks.size()]);
            verify(tier.replace(&engine, &haystack_sequence, &replacement_sequence,
                                tape.empty() ? nullptr : tape.data(), tape.size(), offsets.data()) == sz_success_k);
            verify(engine.report->shortfall == 0);
            for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
                std::string const rewritten = oracle_rewrite_(haystacks[haystack_index], haystack_index, expected,
                                                              replacements);
                std::size_t const first = offsets[haystack_index], last = offsets[haystack_index + 1];
                verify(last - first == rewritten.size());
                verify(rewritten.empty() || std::memcmp(tape.data() + first, rewritten.data(), rewritten.size()) == 0);
            }
        }
    }

    sz_substrings_engine_free(&engine);
}

/** @brief The same corpus under every policy, so one call covers a vocabulary's whole behaviour. */
static void check_policies_(std::vector<std::string> const &haystacks, std::vector<std::string> const &needles,
                            sz_substrings_case_sensitivity_t sensitivity,
                            std::size_t hot_states = SZ_SUBSTRINGS_HOT_STATES_AUTO) {
    check_corpus_(haystacks, needles, sensitivity, sz_substrings_overlapping_k, hot_states);
    check_corpus_(haystacks, needles, sensitivity, sz_substrings_leftmost_longest_k, hot_states);
    check_corpus_(haystacks, needles, sensitivity, sz_substrings_leftmost_first_k, hot_states);
}

/**
 *  @brief The same corpus at every tier split, which is the only way the cold tier is reached at all.
 *
 *  A default build keeps thousands of states hot, so every vocabulary small enough to compare against a
 *  brute-force oracle is entirely hot and the double array is never read. Sweeping the split walks the same
 *  answers through the dense rows, through a mixed automaton, and through an all-cold one.
 */
static void check_tier_splits_(std::vector<std::string> const &haystacks, std::vector<std::string> const &needles,
                               sz_substrings_case_sensitivity_t sensitivity) {
    for (std::size_t hot_states : {(std::size_t)0, (std::size_t)1, (std::size_t)2, (std::size_t)4, (std::size_t)8,
                                   (std::size_t)32, SZ_SUBSTRINGS_HOT_STATES_AUTO})
        check_policies_(haystacks, needles, sensitivity, hot_states);
}

#pragma endregion Helpers

#pragma region Unit Cases

/** @brief The textbook cases, where a wrong failure link or a missed output run shows up by name. */
void test_substrings_unit() {
    // The canonical Aho-Corasick vocabulary: nested suffixes, so every state carries an inherited run.
    check_policies_({"ushers"}, {"he", "she", "his", "hers"}, sz_substrings_cased_k);

    // A needle that is a suffix of another, over text that spells both at once.
    check_policies_({"abcd", "abcde", "zzz"}, {"bc", "abcd", "cd", "d"}, sz_substrings_cased_k);

    // One needle repeated, which is what exercises the leftmost cursor rather than the ranking.
    check_policies_({"aaaaaaaa"}, {"aa", "aaa"}, sz_substrings_cased_k);

    // A vocabulary no haystack hits, so the cover drains nothing and the ring stays zero.
    check_policies_({"the quick brown fox", ""}, {"zebra", "quetzal"}, sz_substrings_cased_k);

    // Bytes outside ASCII, byte-exact, including a lead byte a folded walk would have resynchronized on.
    check_policies_({std::string("\xC3\xA9\xE2\x82\xAC\xFF\xFE", 7)}, {std::string("\xFF\xFE", 2), "\xC3\xA9"},
                    sz_substrings_cased_k);

    // Case folding, where the needle and the haystack agree only after both are folded.
    check_policies_({"Hello World", "HELLO", "hello"}, {"hello", "WORLD"}, sz_substrings_uncased_k);

    // The sharp S folds to two runes, so one source codepoint spans two folded bytes and a match can end
    // at either of them - the repeat the walk collapses.
    check_policies_({"Straße", "STRASSE", "strasse"}, {"strasse", "sse", "s"}, sz_substrings_uncased_k);

    // The Kelvin sign folds to one ASCII byte, contracting three source bytes into one folded one.
    check_policies_({"\xE2\x84\xAA elvin", "kelvin"}, {"k", "kelvin"}, sz_substrings_uncased_k);

    // Malformed UTF-8 in the haystack, which an uncased walk resynchronizes past one byte at a time.
    check_policies_({std::string("ab\xFF" "cd", 5), // "\xFFc" is one escape
                     "abcd"},
                    {"ab", "cd"}, sz_substrings_uncased_k);

    // A chain trie split one state into the hot tier, walked over a byte no needle spells. A cold state's
    // own slot sits inside its children's addressing window, so a slot recorded as its own owner answers
    // that state's probe as an edge, and the walk stays deep where it should have fallen to the root.
    check_tier_splits_({std::string("aaaa\x01" "aaaa", 9), // "\x01a" is one escape
                        "aaaaaaaa"},
                       {"a", "aa", "aaa", "aaaa"}, sz_substrings_cased_k);

    // Every tier split over the canonical vocabulary: a cold state's probe can land on a slot a hot
    // state's child owns, so a mixed automaton is its own case rather than a shade of the two extremes.
    check_tier_splits_({"ushers", "she sells seashells"}, {"he", "she", "his", "hers"}, sz_substrings_cased_k);
    check_tier_splits_({"abcabcabc", "aaaa"}, {"a", "ab", "abc", "bc", "c"}, sz_substrings_cased_k);
    check_tier_splits_({"Straße", "STRASSE"}, {"strasse", "sse", "s"}, sz_substrings_uncased_k);

    // Uppercase needles in a byte-exact vocabulary, which the folded vocabularies' column aliasing must not touch.
    check_policies_({"AbabAB aBAb", "ABAB"}, {"Ab", "ab", "AB", "bA"}, sz_substrings_cased_k);

    // A single-byte vocabulary, whose leftmost ring is narrower than one bitmap word.
    check_policies_({"aabbaab", "b"}, {"a", "b"}, sz_substrings_cased_k);

    // A folded needle past 21 bytes: its source span triples, so the ring spans several bitmap words.
    check_policies_({"The Quick Brown Fox Jumps Over The Lazy Dog, the quick brown fox jumps over the lazy dog"},
                    {"quick brown fox jumps over", "the lazy dog", "fox"}, sz_substrings_uncased_k);

    // An all-cold automaton, so every step probes the double array and chases failure links.
    {
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_default(&alloc);
        std::vector<std::string> const needles {"he", "she", "his", "hers"};
        std::vector<sz_string_view_t> views;
        sz_sequence_t const sequence = sequence_over_(needles, views);
        sz_substrings_engine_t engine;
        verify(sz_substrings_engine_init_cpu(&sequence, sz_substrings_cased_k, sz_substrings_overlapping_k, 0, 0,
                                             &alloc, &engine) == sz_success_k);
        verify(engine.hot_count == 0);
        sz_substrings_engine_free(&engine);
    }
}

/** @brief What the verbs refuse, which is as much of the contract as what they accept. */
void test_substrings_safety() {
    sz_memory_allocator_t alloc;
    sz_memory_allocator_init_default(&alloc);
    sz_substrings_engine_t engine;
    std::vector<sz_string_view_t> views;

    // An empty needle would match at every position, so the whole vocabulary is refused rather than shifting
    // every later needle's reported index.
    {
        std::vector<std::string> const empty_needle {"ab", "", "cd"};
        verify(build_over_(empty_needle, sz_substrings_cased_k, &alloc, &engine) == sz_unexpected_dimensions_k);
    }

    // An empty vocabulary has no automaton to build.
    {
        std::vector<std::string> const no_needles;
        verify(build_over_(no_needles, sz_substrings_cased_k, &alloc, &engine) == sz_unexpected_dimensions_k);
    }

    // A folded vocabulary needs well-formed UTF-8, since the walk resets on a malformed haystack byte and a
    // needle carrying one could never match.
    {
        std::vector<std::string> const malformed {std::string("ab\xFF", 3)};
        verify(build_over_(malformed, sz_substrings_uncased_k, &alloc, &engine) == sz_invalid_utf8_k);
    }

    std::vector<std::string> const needles {"ab", "cd"};
    std::vector<std::string> const haystacks {"abcd"};
    std::vector<std::string> const replacements {"x", "y"};
    std::vector<sz_string_view_t> haystack_views, replacement_views;
    sz_sequence_t const needle_sequence = sequence_over_(needles, views);
    sz_sequence_t const haystack_sequence = sequence_over_(haystacks, haystack_views);
    sz_sequence_t const replacement_sequence = sequence_over_(replacements, replacement_views);
    verify(sz_substrings_engine_init_cpu(&needle_sequence, sz_substrings_cased_k, sz_substrings_overlapping_k,
                                         SZ_SUBSTRINGS_HOT_STATES_AUTO, 0, &alloc, &engine) == sz_success_k);

    // A capacity that cannot hold the matches is not an error: the report names the true total and the
    // shortfall beside it, which is the one contract a device verb can also keep.
    {
        std::vector<sz_size_t> offsets(haystacks.size() + 1, SZ_SIZE_MAX);
        verify(sz_substrings_find(&engine, &haystack_sequence, nullptr, 0, offsets.data()) == sz_success_k);
        verify(engine.report->matches_emitted == 2);
        verify(engine.report->matches_stored == 0 && engine.report->shortfall == 2);
        verify(offsets[haystacks.size()] == 2);
    }

    // A substitution over matches that share bytes is not a function, so the overlapping policy is refused.
    {
        std::vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        verify(sz_substrings_replace(&engine, &haystack_sequence, &replacement_sequence, nullptr, 0, offsets.data()) ==
               sz_status_unknown_k);
    }

    // An output stride of zero cannot address one entry per haystack, whatever the haystack count.
    {
        std::vector<sz_size_t> counts(haystacks.size(), SZ_SIZE_MAX);
        sz_substrings_bm25_t const unnormalized {1.2f, 0.0f, 0.0f};
        sz_f32_t const weights[] {1.0f, 2.0f};
        sz_f32_t score = -1;
        verify(sz_substrings_counts(&engine, &haystack_sequence, counts.data(), 0) == sz_unexpected_dimensions_k);
        verify(sz_substrings_bm25_scores(&engine, &haystack_sequence, nullptr, &unnormalized, weights, &score, 0) ==
               sz_unexpected_dimensions_k);
        verify(counts[0] == SZ_SIZE_MAX && score == -1);
    }

    // BM25 needs one weight per needle, and a mean to normalize by whenever it normalizes at all.
    {
        sz_f32_t const weights[] {1.0f, 2.0f};
        sz_f32_t score = -1;
        sz_substrings_bm25_t const meanless {1.2f, 0.75f, 0.0f};
        sz_substrings_bm25_t const unnormalized {1.2f, 0.0f, 0.0f};
        verify(sz_substrings_bm25_scores(&engine, &haystack_sequence, nullptr, &unnormalized, nullptr, &score, 1) ==
               sz_unexpected_dimensions_k);
        verify(sz_substrings_bm25_scores(&engine, &haystack_sequence, nullptr, &meanless, weights, &score, 1) ==
               sz_unexpected_dimensions_k);
        verify(score == -1);
        // Without normalization the mean is never read: "ab" and "cd" once each, `tf·(k1+1)/(tf+k1)` = 1.
        verify(sz_substrings_bm25_scores(&engine, &haystack_sequence, nullptr, &unnormalized, weights, &score, 1) ==
               sz_success_k);
        verify(std::fabs(score - 3.0f) <= 1e-6f);
    }

    sz_substrings_engine_free(&engine);

    // One replacement per needle, so a shorter list names no substitution for the needles it omits.
    {
        std::vector<std::string> const one_replacement {"x"};
        std::vector<sz_string_view_t> one_views;
        sz_sequence_t const one_sequence = sequence_over_(one_replacement, one_views);
        std::vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        sz_substrings_engine_t covering;
        verify(sz_substrings_engine_init_cpu(&needle_sequence, sz_substrings_cased_k, sz_substrings_leftmost_first_k,
                                             SZ_SUBSTRINGS_HOT_STATES_AUTO, 0, &alloc, &covering) == sz_success_k);
        verify(sz_substrings_replace(&covering, &haystack_sequence, &one_sequence, nullptr, 0, offsets.data()) ==
               sz_unexpected_dimensions_k);
        sz_substrings_engine_free(&covering);
    }

    // A refused allocation is reported as `sz_bad_alloc_k`, and construction is the only verb that allocates.
    {
        sz_memory_allocator_t refusing = refusing_allocator_();
        sz_substrings_engine_t refused {};
        verify(build_over_(needles, sz_substrings_cased_k, &refusing, &refused) == sz_bad_alloc_k);
        verify(refused.memory == nullptr);
    }

    // Every allocation site of the build refused in turn, past the trie's and the arena's first growth.
    {
        std::vector<std::string> wide;
        for (std::size_t index = 0; index != 600; ++index)
            wide.push_back(random_string(3 + index % 6, "abcdefghijklmnopqrstuvwxyz", 26));
        std::sort(wide.begin(), wide.end());
        wide.erase(std::unique(wide.begin(), wide.end()), wide.end());
        check_build_refusals_(wide, sz_substrings_cased_k);
        check_build_refusals_(wide, sz_substrings_uncased_k);
    }
}

/** @brief Random vocabularies over random corpora, which is what reaches the packing search's fallbacks. */
void test_substrings_all() {
    char const *const alphabets[] = {"ab", "abcdefgh", "abcdefghijklmnopqrstuvwxyz"};
    std::size_t const rounds = scale_iterations(24);

    for (std::size_t round = 0; round != rounds; ++round) {
        char const *const alphabet = alphabets[round % 3];
        std::size_t const cardinality = std::strlen(alphabet);
        std::size_t const needles_count = 1 + (round * 7) % 24;
        std::size_t const haystacks_count = 1 + (round * 5) % 9;

        std::vector<std::string> needles;
        for (std::size_t index = 0; index != needles_count; ++index)
            needles.push_back(random_string(1 + (index * 3 + round) % 7, alphabet, cardinality));
        // A vocabulary may not repeat a needle under folding either, and a duplicate would make two needle
        // indices report the same span - which the oracle's own ordering could not distinguish.
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());

        std::vector<std::string> haystacks;
        for (std::size_t index = 0; index != haystacks_count; ++index)
            haystacks.push_back(random_string((index * 11 + round * 3) % 200, alphabet, cardinality));

        check_policies_(haystacks, needles, sz_substrings_cased_k);
        check_policies_(haystacks, needles, sz_substrings_uncased_k);
    }

    // Haystacks spanning several ordered rounds, so leftmost covers stitch windows and rounds together, and
    // folded vocabularies meet single-byte and multi-byte text alike.
    for (std::size_t round = 0; round != scale_iterations(6); ++round) {
        char const *const alphabet = alphabets[round % 3];
        std::size_t const cardinality = std::strlen(alphabet);
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 1 + (round * 5) % 17; ++index)
            needles.push_back(random_string(1 + (index + round) % 9, alphabet, cardinality));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 3; ++index)
            haystacks.push_back(random_string(2048 * (1 + index) + round * 131, alphabet, cardinality));
        std::string mixed = haystacks.back();
        mixed.insert(mixed.size() / 3, "Straße ÄÖÜ \xE2\x84\xAA");
        haystacks.push_back(mixed);
        std::string uppercased = haystacks.front();
        for (char &character : uppercased) character = (char)std::toupper((unsigned char)character);
        haystacks.push_back(uppercased);
        check_policies_(haystacks, needles, sz_substrings_cased_k);
        check_policies_(haystacks, needles, sz_substrings_uncased_k);
    }

    // Needles opening on letters the text rarely holds, so fewer than one byte in eight leaves the root and the
    // tiers skip between live bytes rather than stepping every one.
    for (std::size_t round = 0; round != scale_iterations(4); ++round) {
        std::vector<std::string> const needles {"zebra", "quartz", "qu", "zz", "quiz", "z"};
        std::vector<std::string> haystacks;
        for (std::size_t index = 0; index != 4; ++index) {
            std::string haystack = random_string(3000 + index * 777 + round * 13, "abcdefghijklmnoprstuvwxy", 24);
            for (std::size_t insert = 0; insert != 20; ++insert)
                haystack.insert((insert * 997 + round) % haystack.size(), needles[insert % needles.size()]);
            haystacks.push_back(haystack);
        }
        haystacks.push_back("");
        haystacks.push_back("zzzz");
        check_policies_(haystacks, needles, sz_substrings_cased_k);
        check_policies_(haystacks, needles, sz_substrings_uncased_k);
    }

    // Small alphabets, where hot rows are only a few classes wide: nucleotides with bytes no needle spells
    // mixed in, a two-letter alphabet, and every tier split over them.
    for (std::size_t round = 0; round != scale_iterations(4); ++round) {
        std::vector<std::string> nucleotides, bits, haystacks, binary_haystacks;
        for (std::size_t index = 0; index != 20 + round * 30; ++index)
            nucleotides.push_back(random_string(3 + (index + round) % 14, "ACGT", 4));
        for (std::size_t index = 0; index != 12; ++index) bits.push_back(random_string(1 + index % 9, "\x00\x01", 2));
        for (std::vector<std::string> *vocabulary : {&nucleotides, &bits}) {
            std::sort(vocabulary->begin(), vocabulary->end());
            vocabulary->erase(std::unique(vocabulary->begin(), vocabulary->end()), vocabulary->end());
        }
        for (std::size_t index = 0; index != 3; ++index) {
            std::string haystack = random_string(1500 + index * 2111 + round * 7, "ACGT", 4);
            for (std::size_t offset = 97; offset < haystack.size(); offset += 389)
                haystack[offset] = "N\n>"[offset % 3];
            haystacks.push_back(haystack);
            binary_haystacks.push_back(random_string(2000 + index * 333, "\x00\x01", 2));
        }
        haystacks.push_back("ACGTNNACGT\nACG");
        check_policies_(haystacks, nucleotides, sz_substrings_cased_k);
        check_policies_(binary_haystacks, bits, sz_substrings_cased_k);
        if (round == 0) check_tier_splits_(haystacks, nucleotides, sz_substrings_cased_k);
    }

    // Every byte value spelled by some needle, so no class is shared and a hot row spans all 256 columns.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t byte = 0; byte != 256; ++byte)
            needles.push_back(std::string(1, (char)byte) + std::string(1, (char)((byte * 7 + 3) & 0xFF)));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 3; ++index) {
            std::string haystack(3000 + index * 101, '\0');
            for (std::size_t offset = 0; offset != haystack.size(); ++offset)
                haystack[offset] = (char)((offset * 31 + index * 17 + (offset * offset) % 13) & 0xFF);
            haystacks.push_back(haystack);
        }
        check_policies_(haystacks, needles, sz_substrings_cased_k);
    }

    // A vocabulary wide enough to push the double array past its first growth, and deep enough that the
    // failure chains it packs are longer than one edge.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 400; ++index)
            needles.push_back(random_string(2 + index % 9, "abcdefghijklmnopqrstuvwxyz", 26));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 8; ++index)
            haystacks.push_back(random_string(500 + index * 37, "abcdefghijklmnopqrstuvwxyz", 26));
        check_policies_(haystacks, needles, sz_substrings_cased_k);
    }
}

#pragma endregion Unit Cases
