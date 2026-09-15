/**
 *  @brief  Levenshtein edit-distance tests: known answers, a textbook DP oracle, and every backend against it.
 *  @file   test/levenshtein.cpp
 *  @author Ash Vardanian
 *  @date January 4, 2024
 */
#undef NDEBUG // ! Enable all assertions for testing

/**
 *  The Visual C++ run-time library detects incorrect iterator use,
 *  and asserts and displays a dialog box at run time on Windows.
 */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

/**
 *  ! Overload the following with caution.
 *  ! Those parameters must never be explicitly set during releases,
 *  ! but they come handy during development, if you want to validate
 *  ! different ISA-specific implementations.

 #define SZ_USE_HASWELL 0
 #define SZ_USE_ICELAKE 0
 */
#define SZ_USE_MISALIGNED_LOADS 0
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/**
 *  Make sure to include the StringZilla headers before anything else,
 *  to intercept missing `#include` directives and other issues.
 */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <cstdio> // `std::printf`

#include <algorithm> // `std::min`
#include <random>    // `std::uniform_int_distribution`
#include <string>    // Baseline
#include <vector>    // `std::vector`

#include "stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;

#pragma region Helpers

/** @brief Textbook O(n·m) Levenshtein over any symbol sequence, the oracle every backend is measured against. */
template <typename symbols_type_>
static std::size_t levenshtein_reference_(symbols_type_ const &first, symbols_type_ const &second) {
    std::vector<std::size_t> previous(second.size() + 1), current(second.size() + 1);
    for (std::size_t second_position = 0; second_position <= second.size(); ++second_position)
        previous[second_position] = second_position;
    for (std::size_t first_position = 1; first_position <= first.size(); ++first_position) {
        current[0] = first_position;
        for (std::size_t second_position = 1; second_position <= second.size(); ++second_position) {
            std::size_t const substitution = previous[second_position - 1] +
                                             (first[first_position - 1] != second[second_position - 1]);
            current[second_position] = std::min(
                {previous[second_position] + 1, current[second_position - 1] + 1, substitution});
        }
        std::swap(previous, current);
    }
    return previous[second.size()];
}

/** @brief The rune sequence the UTF-8 entries score, decoded here on the rune codec alone so the oracle shares nothing
 *         with the kernel: an ill-formed byte is one @c U+FFFD. */
static std::u32string levenshtein_runes_(std::string const &text) {
    std::u32string runes;
    for (std::size_t position = 0; position < text.size();) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode(text.data() + position, text.data() + text.size(), &rune);
        if (consumed == sz_rune_invalid_k) runes.push_back(sz_rune_replacement_k), ++position;
        else runes.push_back(rune), position += consumed;
    }
    return runes;
}

/** @brief One backend's one-to-many entries; the one-to-one entries have no backends, every one is the serial walk. */
struct levenshtein_backend_t {
    char const *name;
    sz_levenshtein_distances_t distances;
    sz_levenshtein_distances_t distances_utf8;
};

static levenshtein_backend_t const levenshtein_backends[] = {
    {"serial", sz_levenshtein_distances_serial, sz_levenshtein_distances_utf8_serial},
#if SZ_USE_HASWELL
    {"haswell", sz_levenshtein_distances_haswell, sz_levenshtein_distances_utf8_haswell},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_levenshtein_distances_icelake, sz_levenshtein_distances_utf8_icelake},
#endif
};

/** @brief Runs one query against @p candidates through @p distances, asserting each answer matches @p expected. */
static void check_levenshtein_distances_(std::string const &query, std::vector<std::string> const &candidates,
                                         std::vector<sz_size_t> const &expected, sz_levenshtein_distances_t distances) {
    sz_memory_allocator_t alloc;
    sz_memory_allocator_init_default(&alloc);
    sz_sequence_t const sequence = sequence_from_(candidates);
    std::vector<sz_size_t> computed(candidates.size(), SZ_SIZE_MAX);
    verify(distances(query.data(), query.size(), &sequence, &alloc, computed.data()) == sz_success_k);
    verify(computed == expected);
}

/** @brief Runs every pair through @p distance in both directions, asserting each answer matches @p expected. */
static void check_levenshtein_distance_(std::string const &query, std::vector<std::string> const &candidates,
                                        std::vector<sz_size_t> const &expected, sz_levenshtein_distance_t distance) {
    sz_memory_allocator_t alloc;
    sz_memory_allocator_init_default(&alloc);
    for (std::size_t index = 0; index != candidates.size(); ++index) {
        sz_size_t forward = 0, backward = 0;
        verify(distance(query.data(), query.size(), candidates[index].data(), candidates[index].size(), &alloc,
                        &forward) == sz_success_k);
        verify(distance(candidates[index].data(), candidates[index].size(), query.data(), query.size(), &alloc,
                        &backward) == sz_success_k);
        verify(forward == expected[index] && backward == expected[index]);
    }
}

/** @brief Runs one query against @p candidates through every backend and the dispatched entries, byte-level and
 *         rune-level alike, each against its own oracle. */
static void check_levenshtein_case_(std::string const &query, std::vector<std::string> const &candidates) {
    std::vector<sz_size_t> expected_bytes, expected_runes;
    std::u32string const query_runes = levenshtein_runes_(query);
    for (std::string const &candidate : candidates) {
        expected_bytes.push_back(levenshtein_reference_(query, candidate));
        expected_runes.push_back(levenshtein_reference_(query_runes, levenshtein_runes_(candidate)));
    }
    check_levenshtein_distance_(query, candidates, expected_bytes, sz_levenshtein_distance);
    check_levenshtein_distance_(query, candidates, expected_bytes, sz_levenshtein_distance_serial);
    check_levenshtein_distance_(query, candidates, expected_runes, sz_levenshtein_distance_utf8);
    check_levenshtein_distance_(query, candidates, expected_runes, sz_levenshtein_distance_utf8_serial);
    check_levenshtein_distances_(query, candidates, expected_bytes, sz_levenshtein_distances);
    check_levenshtein_distances_(query, candidates, expected_runes, sz_levenshtein_distances_utf8);
    for (levenshtein_backend_t const &backend : levenshtein_backends) {
        check_levenshtein_distances_(query, candidates, expected_bytes, backend.distances);
        check_levenshtein_distances_(query, candidates, expected_runes, backend.distances_utf8);
    }
}

#pragma endregion // Helpers

#pragma region Unit

/** @brief Known-answer cases: the classic pairs, empties on either side, and identity. */
void test_levenshtein_unit() {
    check_levenshtein_case_("kitten", {"sitting", "kitten", "", "k", "kittens", "mitten"});
    check_levenshtein_case_("flaw", {"lawn", "flaw", "flaws", "law"});
    check_levenshtein_case_("", {"", "a", "abc", std::string(300, 'x')});
    check_levenshtein_case_(std::string(64, 'a'), {std::string(64, 'a'), std::string(65, 'a'), std::string(63, 'a'),
                                                   std::string(64, 'b'), ""});
    check_levenshtein_case_(std::string(65, 'a'), {std::string(65, 'a'), std::string(64, 'a'), std::string(130, 'a')});
    // Multi-byte runes: one rune edit costs several byte edits, and an ill-formed byte is one rune.
    check_levenshtein_case_("héllo", {"hello", "héllo", "h\xC3llo", "hé", "日本語"});
    check_levenshtein_case_("日本語", {"日本", "日本語です", "本", "", "\xFF\xFE"});
    check_levenshtein_case_("\xE2\x82", {"€", "\xE2\x82\xAC", "ab"});
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief Differential test against the oracle across every backend: query and candidate lengths sweep every
 *         64-symbol word boundary and reach past the register-resident word tiers, on a two-letter alphabet
 *         that forces matches, on a multi-byte rune alphabet, and on full bytes.
 */
void test_levenshtein_equivalence() {
    std::size_t const lengths[] = {0, 1, 2, 5, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300, 511, 512, 513, 640, 1000};
    char const binary_alphabet[] = "ab";
    std::vector<std::string> const rune_alphabet = {"a", "é", "日", "€", "𝄞"};
    std::mt19937 &generator = global_random_generator();
    for (std::size_t query_length : lengths) {
        std::vector<std::string> candidates;
        for (std::size_t candidate_length : lengths)
            candidates.push_back(random_string(candidate_length, binary_alphabet, 2));
        std::string const query = random_string(query_length, binary_alphabet, 2);
        candidates.push_back(query);
        candidates.push_back(query.substr(0, query_length / 2));
        check_levenshtein_case_(query, candidates);
    }
    for (std::size_t query_runes : {0, 1, 63, 64, 65, 129, 300, 640, 1000}) {
        std::vector<std::string> candidates;
        for (std::size_t candidate_runes : {0, 1, 64, 65, 200, 256, 512, 513})
            candidates.push_back(random_string(candidate_runes, rune_alphabet));
        std::string const query = random_string(query_runes, rune_alphabet);
        candidates.push_back(query);
        check_levenshtein_case_(query, candidates);
    }
    // Three hundred consecutive CJK runes: more classes than a byte holds, and two pages of the rune table.
    std::string wide_query;
    for (sz_rune_t rune = 0x4E00; rune != 0x4E00 + 300; ++rune) {
        wide_query += static_cast<char>(0xE0 | (rune >> 12));
        wide_query += static_cast<char>(0x80 | ((rune >> 6) & 0x3F));
        wide_query += static_cast<char>(0x80 | (rune & 0x3F));
    }
    check_levenshtein_case_(wide_query, {wide_query, wide_query.substr(0, 150), wide_query.substr(300), "abc", ""});
    check_levenshtein_case_("abc", {});
    for (std::size_t round = 0; round != scale_iterations(8); ++round) {
        std::size_t const query_length = std::uniform_int_distribution<std::size_t>(1, 700)(generator);
        std::string query(query_length, 0);
        randomize_string(query.data(), query.size());
        std::vector<std::string> candidates;
        for (std::size_t index = 0; index != 40; ++index) {
            std::string candidate(std::uniform_int_distribution<std::size_t>(0, 900)(generator), 0);
            randomize_string(candidate.data(), candidate.size());
            candidates.push_back(candidate);
        }
        check_levenshtein_case_(query, candidates);
    }
}

#pragma endregion // Equivalence

#pragma region Safety

/** @brief A refusing allocator: every entry answers the allocation failure instead of touching the outputs. */
void test_levenshtein_safety() {
    sz_memory_allocator_t refusing;
    refusing.allocate = +[](sz_size_t, void *) -> void * { return nullptr; };
    refusing.free = +[](void *, sz_size_t, void *) {};
    refusing.handle = nullptr;
    std::vector<std::string> const refused_candidates = {"sitting", "kitten"};
    sz_sequence_t const refused_sequence = sequence_from_(refused_candidates);
    sz_size_t refused[2] = {SZ_SIZE_MAX, SZ_SIZE_MAX};
    for (levenshtein_backend_t const &backend : levenshtein_backends) {
        verify(backend.distances("kitten", 6, &refused_sequence, &refusing, refused) == sz_bad_alloc_k);
        verify(backend.distances_utf8("kitten", 6, &refused_sequence, &refusing, refused) == sz_bad_alloc_k);
    }
    verify(sz_levenshtein_distance_serial("kitten", 6, "sitting", 7, &refusing, refused) == sz_bad_alloc_k);
    verify(sz_levenshtein_distance_utf8_serial("kitten", 6, "sitting", 7, &refusing, refused) == sz_bad_alloc_k);
    verify(refused[0] == SZ_SIZE_MAX && refused[1] == SZ_SIZE_MAX);
}

#pragma endregion // Safety
