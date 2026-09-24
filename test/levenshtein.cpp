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

#include <cstdio>  // `std::printf`, `std::fprintf`
#include <cstring> // `std::strcmp`

#include <algorithm> // `std::min`
#include <random>    // `std::uniform_int_distribution`
#include <string>    // Baseline
#include <vector>    // `std::vector`

#include "stringzilla.hpp" // `global_random_generator`, `random_string`, `refusing_allocator_`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

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

/** @brief One backend's cross-product verb; a batch is prepared by one host builder whatever tier scores it. */
struct levenshtein_backend_t {
    char const *name;                     /**< The row's spelling, for @ref fail_backend_ and the log. */
    sz_levenshtein_distances_t distances; /**< The verb, over bytes or over runes as the engine was built. */
};

/** @brief Every cross-product backend compiled into this translation unit, dispatched first. */
static levenshtein_backend_t const levenshtein_backends[] = {
    {"dispatched", sz_levenshtein_distances},
    {"serial", sz_levenshtein_distances_serial},
#if SZ_USE_HASWELL
    {"haswell", sz_levenshtein_distances_haswell},
#endif
#if SZ_USE_SKYLAKE
    {"skylake", sz_levenshtein_distances_skylake},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_levenshtein_distances_icelake},
#endif
};

/** @brief One prepared batch, released with the scope that named it. */
struct levenshtein_engine_t {
    sz_levenshtein_engine_t engine {};

    levenshtein_engine_t(sz_sequence_t const &queries, sz_levenshtein_symbol_t symbol) {
        verify(sz_levenshtein_engine_init_cpu(&queries, symbol, nullptr, &engine) == sz_success_k);
    }
    ~levenshtein_engine_t() { sz_levenshtein_engine_free(&engine); }
    levenshtein_engine_t(levenshtein_engine_t const &) = delete;
    levenshtein_engine_t &operator=(levenshtein_engine_t const &) = delete;
};

/** @brief Runs @p queries against @p candidates through @p distances of @p name, asserting the matrix matches
 *         @p expected, which is @b [queries, candidates] row-major. */
static void check_levenshtein_distances_(char const *name, std::vector<std::string> const &queries,
                                         std::vector<std::string> const &candidates,
                                         std::vector<sz_size_t> const &expected, sz_levenshtein_symbol_t symbol,
                                         sz_levenshtein_distances_t distances) {
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const candidate_sequence = sequence_from_(candidates);
    levenshtein_engine_t prepared(query_sequence, symbol);
    std::vector<sz_size_t> computed(queries.size() * candidates.size(), SZ_SIZE_MAX);
    if (distances(&prepared.engine, &candidate_sequence, computed.data(), candidates.size()) != sz_success_k)
        fail_backend_(name, "the cross-product verb refused a well-formed batch");
    if (computed != expected) fail_backend_(name, "the cross-product distances differ from the expected answers");
}

/** @brief Runs @p queries against @p candidates through every backend row, byte-level and rune-level alike,
 *         against the given answers. */
static void check_levenshtein_expected_(std::vector<std::string> const &queries,
                                        std::vector<std::string> const &candidates,
                                        std::vector<sz_size_t> const &expected_bytes,
                                        std::vector<sz_size_t> const &expected_runes) {
    for (levenshtein_backend_t const &backend : levenshtein_backends) {
        check_levenshtein_distances_(backend.name, queries, candidates, expected_bytes, sz_levenshtein_bytes_k,
                                     backend.distances);
        check_levenshtein_distances_(backend.name, queries, candidates, expected_runes, sz_levenshtein_runes_k,
                                     backend.distances);
    }
}

/** @brief Runs @p queries against @p candidates through every entry, each against its own oracle. */
static void check_levenshtein_case_(std::vector<std::string> const &queries,
                                    std::vector<std::string> const &candidates) {
    std::vector<sz_size_t> expected_bytes, expected_runes;
    for (std::string const &query : queries) {
        std::u32string const query_runes = levenshtein_runes_(query);
        for (std::string const &candidate : candidates) {
            expected_bytes.push_back(levenshtein_reference_(query, candidate));
            expected_runes.push_back(levenshtein_reference_(query_runes, levenshtein_runes_(candidate)));
        }
    }
    check_levenshtein_expected_(queries, candidates, expected_bytes, expected_runes);
}

/** @brief One backend's answers against the serial backend's, bit for bit, over random batches on a two-letter
 *         and on a multi-byte rune alphabet, with several queries in flight so the query axis is exercised. */
static void check_levenshtein_equivalence_(levenshtein_backend_t const &reference,
                                           levenshtein_backend_t const &candidate, std::size_t rounds) {
    std::mt19937 &generator = global_random_generator();
    std::vector<std::string> candidates, queries;
    std::vector<sz_size_t> from_reference, from_candidate;
    for (std::size_t round = 0; round != rounds; ++round)
        for (char const *alphabet : {"ab", "aé日€𝄞"}) {
            randomize_strings(fuzzy_config_t(alphabet, 16, 0, 900), candidates);
            queries.clear();
            for (std::size_t query = 0; query != 5; ++query) {
                std::size_t const query_length = std::uniform_int_distribution<std::size_t>(1, 700)(generator);
                queries.push_back(random_string(query_length, alphabet_characters(alphabet)));
            }
            sz_sequence_t const query_sequence = sequence_from_(queries);
            sz_sequence_t const candidate_sequence = sequence_from_(candidates);
            std::size_t const cells = queries.size() * candidates.size();

            for (sz_levenshtein_symbol_t const symbol : {sz_levenshtein_bytes_k, sz_levenshtein_runes_k}) {
                levenshtein_engine_t prepared(query_sequence, symbol);
                from_reference.assign(cells, SZ_SIZE_MAX), from_candidate.assign(cells, SZ_SIZE_MAX);
                verify(reference.distances(&prepared.engine, &candidate_sequence, from_reference.data(),
                                           candidates.size()) == sz_success_k);
                if (candidate.distances(&prepared.engine, &candidate_sequence, from_candidate.data(),
                                        candidates.size()) != sz_success_k)
                    fail_backend_(candidate.name, "a batch serial accepted was refused");
                if (from_reference != from_candidate) fail_backend_(candidate.name, "distances disagreed with serial");
            }
        }
}

#pragma endregion // Helpers

#pragma region Unit

/** @brief One candidate and the literal distances it must score against the query, in bytes and in runes. */
struct levenshtein_known_t {
    std::string candidate;
    sz_size_t bytes;
    sz_size_t runes;
};

/** @brief Runs one query against its known candidates through every backend row. */
static void check_levenshtein_unit_(std::string const &query, std::vector<levenshtein_known_t> const &known) {
    std::vector<std::string> candidates;
    std::vector<sz_size_t> expected_bytes, expected_runes;
    for (levenshtein_known_t const &entry : known) {
        candidates.push_back(entry.candidate);
        expected_bytes.push_back(entry.bytes);
        expected_runes.push_back(entry.runes);
    }
    check_levenshtein_expected_({query}, candidates, expected_bytes, expected_runes);
}

/** @brief Known answers: the classic pairs, empties on either side, identity, the 64-symbol word boundary, and
 *         multi-byte runes, through every backend row. */
void test_levenshtein_unit() {
    std::printf("  - testing edit-distance known-answer vectors...\n");
    check_levenshtein_unit_(
        "kitten", {{"sitting", 3, 3}, {"kitten", 0, 0}, {"", 6, 6}, {"k", 5, 5}, {"kittens", 1, 1}, {"mitten", 1, 1}});
    check_levenshtein_unit_("flaw", {{"lawn", 2, 2}, {"flaw", 0, 0}, {"flaws", 1, 1}, {"law", 1, 1}});
    check_levenshtein_unit_("", {{"", 0, 0}, {"a", 1, 1}, {"abc", 3, 3}, {std::string(300, 'x'), 300, 300}});
    check_levenshtein_unit_(std::string(64, 'a'), {{std::string(64, 'a'), 0, 0},
                                                   {std::string(65, 'a'), 1, 1},
                                                   {std::string(63, 'a'), 1, 1},
                                                   {std::string(64, 'b'), 64, 64},
                                                   {"", 64, 64}});
    check_levenshtein_unit_(
        std::string(65, 'a'),
        {{std::string(65, 'a'), 0, 0}, {std::string(64, 'a'), 1, 1}, {std::string(130, 'a'), 65, 65}});
    // Multi-byte runes: one rune edit costs several byte edits, and an ill-formed byte is one rune.
    check_levenshtein_unit_("héllo",
                            {{"hello", 2, 1}, {"héllo", 0, 0}, {"h\xC3llo", 1, 1}, {"hé", 3, 3}, {"日本語", 9, 5}});
    check_levenshtein_unit_("日本語",
                            {{"日本", 3, 1}, {"日本語です", 6, 2}, {"本", 6, 2}, {"", 9, 3}, {"\xFF\xFE", 9, 3}});
    check_levenshtein_unit_("\xE2\x82", {{"€", 1, 2}, {"\xE2\x82\xAC", 1, 2}, {"ab", 2, 2}});

    // Several queries in one batch, so the rows of the matrix are what the sweep wrote and not one row repeated.
    check_levenshtein_case_({"kitten", "sitting", "", std::string(70, 'a')},
                            {"kitten", "sitting", "", "kit", std::string(70, 'b')});
}

#pragma endregion // Unit

#pragma region Safety

/** @brief The cross-product verb of @p backend on degenerate batches: empties on either side, both, and none
 *         survive, and a stride too narrow for one row is refused without touching the outputs. */
static void check_levenshtein_safety_(levenshtein_backend_t const &backend) {
    std::vector<std::string> const queries = {"kitten", ""};
    std::vector<std::string> const words = {"sitting", "kitten"};
    std::vector<std::string> const empty = {""};
    std::vector<std::string> const none;
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const words_sequence = sequence_from_(words);
    sz_sequence_t const empty_sequence = sequence_from_(empty);
    sz_sequence_t const none_sequence = sequence_from_(none);
    for (sz_levenshtein_symbol_t const symbol : {sz_levenshtein_bytes_k, sz_levenshtein_runes_k}) {
        levenshtein_engine_t prepared(query_sequence, symbol);
        sz_size_t answers[4] = {SZ_SIZE_MAX, SZ_SIZE_MAX, SZ_SIZE_MAX, SZ_SIZE_MAX};
        if (backend.distances(&prepared.engine, &words_sequence, answers, 2) != sz_success_k)
            fail_backend_(backend.name, "a batch holding an empty query was refused");
        if (backend.distances(&prepared.engine, &empty_sequence, answers, 1) != sz_success_k)
            fail_backend_(backend.name, "an empty candidate was refused");
        if (backend.distances(&prepared.engine, &none_sequence, answers, 0) != sz_success_k)
            fail_backend_(backend.name, "an empty batch of candidates was refused");
        sz_size_t narrow[4] = {SZ_SIZE_MAX, SZ_SIZE_MAX, SZ_SIZE_MAX, SZ_SIZE_MAX};
        if (backend.distances(&prepared.engine, &words_sequence, narrow, 1) != sz_unexpected_dimensions_k)
            fail_backend_(backend.name, "a stride too narrow for one row was not refused");
        for (sz_size_t const written : narrow)
            if (written != SZ_SIZE_MAX) fail_backend_(backend.name, "a refused round still wrote the distances");
    }
}

/** @brief The host builder reporting a refused allocation rather than handing back a half-built engine. */
static void check_levenshtein_build_safety_() {
    sz_memory_allocator_t refusing = refusing_allocator_();
    std::vector<std::string> const queries = {"kitten", "sitting"};
    sz_sequence_t const query_sequence = sequence_from_(queries);
    for (sz_levenshtein_symbol_t const symbol : {sz_levenshtein_bytes_k, sz_levenshtein_runes_k}) {
        sz_levenshtein_engine_t engine {};
        if (sz_levenshtein_engine_init_cpu(&query_sequence, symbol, &refusing, &engine) != sz_bad_alloc_k)
            fail_backend_("dispatched", "the builder did not report the refused allocation");
        if (engine.memory != nullptr) fail_backend_("dispatched", "a refused build still kept a block");
    }
}

/**
 *  @brief Degenerate inputs for the edit-distance family, asserting survival and the stated refusals.
 *         Answers are not the subject here: empties are accepted, a refused allocation is reported, and no failure
 *         writes an output.
 */
void test_levenshtein_safety() {
    std::printf("  - testing degenerate inputs and refused allocations of the edit-distance kernels...\n");
    for (levenshtein_backend_t const &backend : levenshtein_backends) check_levenshtein_safety_(backend);
    check_levenshtein_build_safety_();
}

#pragma endregion // Safety

#pragma region Drivers

/**
 *  @brief Drives the oracle sweeps and the serial-versus-SIMD differential across every backend compiled here: query
 *         and candidate lengths sweep every 64-symbol word boundary and reach past the register-resident word tiers,
 *         on a two-letter alphabet that forces matches, on a multi-byte rune alphabet, and on full bytes.
 */
void test_levenshtein_all() {
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
        check_levenshtein_case_({query}, candidates);
    }
    for (std::size_t query_runes : {0, 1, 63, 64, 65, 129, 300, 640, 1000}) {
        std::vector<std::string> candidates;
        for (std::size_t candidate_runes : {0, 1, 64, 65, 200, 256, 512, 513})
            candidates.push_back(random_string(candidate_runes, rune_alphabet));
        std::string const query = random_string(query_runes, rune_alphabet);
        candidates.push_back(query);
        check_levenshtein_case_({query}, candidates);
    }
    // Three hundred consecutive CJK runes: more classes than a byte holds, and two pages of the rune table.
    std::string wide_query;
    for (sz_rune_t rune = 0x4E00; rune != 0x4E00 + 300; ++rune) {
        wide_query += static_cast<char>(0xE0 | (rune >> 12));
        wide_query += static_cast<char>(0x80 | ((rune >> 6) & 0x3F));
        wide_query += static_cast<char>(0x80 | (rune & 0x3F));
    }
    check_levenshtein_case_({wide_query},
                            {wide_query, wide_query.substr(0, 150), wide_query.substr(300), "abc", ""});
    check_levenshtein_case_({"abc"}, {});
    for (std::size_t round = 0; round != scale_iterations(8); ++round) {
        std::vector<std::string> queries;
        for (std::size_t index = 0; index != 3; ++index) {
            std::size_t const query_length = std::uniform_int_distribution<std::size_t>(1, 700)(generator);
            queries.push_back(std::string(query_length, '\0'));
            randomize_string(&queries.back()[0], queries.back().size());
        }
        std::vector<std::string> candidates;
        for (std::size_t index = 0; index != 40; ++index) {
            std::string candidate(std::uniform_int_distribution<std::size_t>(0, 900)(generator), '\0');
            randomize_string(&candidate[0], candidate.size());
            candidates.push_back(candidate);
        }
        check_levenshtein_case_(queries, candidates);
    }

    // Serial is the reference for everything, itself included.
    levenshtein_backend_t const &reference = backend_named_(levenshtein_backends, "serial");
    for (levenshtein_backend_t const &candidate : levenshtein_backends)
        check_levenshtein_equivalence_(reference, candidate, scale_iterations(8));
}

#pragma endregion // Drivers
