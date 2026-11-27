/**
 *  @file test/cross_device.hpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine checks - the device backend and dispatch scenario CUDA, ROCm and Metal share.
 */
#pragma once
#ifndef STRINGZILLA_TEST_CROSS_DEVICE_HPP_
#define STRINGZILLA_TEST_CROSS_DEVICE_HPP_

#include <cmath>   // `std::fabs`, `std::exp`, `std::log`
#include <cstddef> // `std::size_t`
#include <cstring> // `std::memset`

#include <algorithm>   // `std::sort`, `std::copy`
#include <array>       // `std::array`
#include <random>      // `std::uniform_real_distribution`
#include <type_traits> // `std::is_null_pointer_v`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <vector>      // `std::vector`

#include "stringzilla/stringzilla.h"     // Primary C API
#include "stringzilla/strings.hpp"       // `strings<>::view`
#include "stringzilla/strings_verbs.hpp" // `sz::gather`

#include "harness.hpp" // `capability_gate_t`, `pinned_vector`, `expect_true`
#include "cross.hpp"   // `check_uncased_fold_equivalence_`, `for_each_adversarial_utf8_input_`

#if STRINGZILLA_ARCH_CUDA_
#include "stringzilla/cuda.cuh" // `cudaDeviceGetAttribute`, `cudaStream_t`
#endif
#if STRINGZILLA_ARCH_ROCM_
#include "stringzilla/rocm.cuh" // `hipDeviceProp_t`, `hipStream_t`
#endif

namespace ashvardanian::stringzilla::test {

/** One vendor's engine kernels, or the dispatch points in their place, which take the same
 *  arguments: a device engine records its capability, so a verb reaches that vendor's kernel. */
template <typename fold_type_, typename norm_type_, typename sha256_update_type_, typename sha256_digest_type_>
struct device_kernels {

    /** The spelling @ref fail_backend_ and the test names carry. */
    char const *name;
    device_backend_t const &runtime;

    sz_kernel_levenshtein_distances_t levenshtein_distances;
    sz_kernel_ngrams_overlaps_t ngrams_overlaps;
    sz_kernel_ngrams_sketches_t ngrams_sketches;

    /** The three @c contains, each null where the vendor has none. */
    sz_kernel_substrings_contains_t substrings_contains;
    sz_kernel_substrings_counts_t substrings_counts;
    sz_kernel_substrings_find_t substrings_find;
    sz_kernel_substrings_replace_t substrings_replace;
    sz_kernel_substrings_bm25_scores_t substrings_bm25_scores;
    sz_kernel_regex_contains_t regex_contains;
    sz_kernel_regex_counts_t regex_counts;
    sz_kernel_regex_find_t regex_find;
    sz_kernel_regex_replace_t regex_replace;
    sz_kernel_regex_extract_t regex_extract;
    sz_kernel_subdocuments_contains_t subdocuments_contains;
    sz_kernel_subdocuments_counts_t subdocuments_counts;
    sz_kernel_subdocuments_find_t subdocuments_find;
    fold_type_ utf8_uncased_fold;
    norm_type_ utf8_norm;

    /** Null where the vendor has no multi-state SHA-256, as Metal has none. */
    sha256_update_type_ sha256_multistate_update;
    sha256_digest_type_ sha256_multistate_digest;
};

/** Binds the dispatch point to the device's enabled capabilities. */
template <auto best_>
inline auto gpu_best(sz_capability_t capabilities) {
    return [capabilities](auto... arguments) noexcept { return call_best<best_>(capabilities, arguments...); };
}

/** Joins @p stream, or the default one, which is what every device verb leaves the caller to do. */
inline void join_(device_backend_t const &runtime, sz_stream_t stream) {
    expect_equal(sz_stream_synchronize_best(runtime.capabilities, stream), sz_success_k);
}

/** @p strings as one tape in pinned memory, gathered on the host, which the serial oracle walks and
 *  the device gathers its own copy from. */
inline pinned_strings_t staged_(device_backend_t const &runtime, sz_strings_view_t const &strings) {
    auto [gathered, status] = sz::gather(strings_view_t(strings), pinned_alloc_<char>(runtime));
    expect_equal(status, status_t::success_k);
    return std::move(gathered);
}
inline pinned_strings_t staged_(device_backend_t const &runtime, std::vector<std::string> const &texts) {
    return staged_(runtime, strings_from_(texts));
}
inline pinned_strings_t staged_(device_backend_t const &runtime, std::span<sz_string_view_t const> views) {
    return staged_(runtime, strings_from_(views));
}

/** The device's own copy of @p pinned, gathered there by its kernels into the backend's arena,
 *  which its verbs read until the case rewinds the arena. */
inline sz_strings_view_t uploaded_(device_backend_t const &runtime, sz_strings_view_t const &pinned) {
    sz_allocator_t policy = device_arena_policy_(*runtime.arena);
    sz_strings_span_t span {};
    expect_equal(
        sz_strings_to_tape_best(&pinned, nullptr, pinned.count, &policy, &span, runtime.capabilities, runtime.stream),
        sz_success_k);
    sz_strings_view_t device;
    sz_strings_from_span(&span, &device);
    return device;
}

/** @p body, handed the case's generator when it takes one, rewinding the backend's arena once it
 *  returns, with every tape it carved joined. */
template <typename body_type_>
auto rewound_(device_backend_t const &runtime, body_type_ body) {
    return [&runtime, body]([[maybe_unused]] split_mix_generator_t &generator) {
        if constexpr (std::is_invocable_v<body_type_ const &, split_mix_generator_t &>) body(generator);
        else body();
        runtime.arena->consumed = 0;
    };
}

/** A device @p contains call's bitmap against @p expected, the serial one, byte for byte: ending
 *  where its buffer ends, so a write past it trips the sanitizers, starting there and one byte in,
 *  so both alignments of its head run, and filled with ones, so an unwritten tail bit shows. */
inline void check_contains_(device_backend_t const &runtime, std::vector<sz_u1x8_t> const &expected,
                            auto const &contains) {
    for (std::size_t const offset : {0, 1}) {
        pinned_vector<sz_u1x8_t> buffer(offset + expected.size(), 0xFF, pinned_alloc_<sz_u1x8_t>(runtime));
        expect_equal(contains(buffer.data() + offset), sz_success_k);
        join_(runtime, runtime.stream);
        expect_true(std::equal(buffer.begin() + offset, buffer.end(), expected.begin()));
        expect_true(std::all_of(buffer.begin(), buffer.begin() + offset, [](sz_u1x8_t byte) { return byte == 0xFF; }));
    }
}

/** How the candidate lengths of a skewed batch differ by orders of magnitude. */
enum class device_skew_t {

    /** One candidate of three mebibytes, wherever the draw puts it, among a hundred thousand of at
     *  most thirty-two bytes. */
    lone_long_k,

    /** A hundred and twenty-eight drawn log-uniform up to a mebibyte. */
    log_uniform_k,
};

/** Candidate lengths of one @p skew. A round that hands one thread one candidate costs its longest,
 *  so these are the batches a balanced round exists for. */
inline std::vector<std::size_t> device_skewed_lengths_(split_mix_generator_t &generator, device_skew_t skew) {
    if (skew == device_skew_t::lone_long_k) {
        std::vector<std::size_t> lengths(100000);
        for (std::size_t &length : lengths) length = generator() % 33;
        lengths[generator() % lengths.size()] = (std::size_t)3 << 20;
        return lengths;
    }
    std::vector<std::size_t> lengths(128);
    std::uniform_real_distribution<double> exponent(0.0, std::log((double)((std::size_t)1 << 20)));
    for (std::size_t &length : lengths) length = (std::size_t)std::exp(exponent(generator));
    return lengths;
}

/** Pads the last of @p texts with spaces until their bytes end three past a word boundary, so a
 *  load of the word holding the last byte runs past the payload. */
inline void end_off_word_(std::vector<std::string> &texts) {
    std::size_t bytes = 0;
    for (std::string const &text : texts) bytes += text.size();
    for (; bytes % 4 != 3; ++bytes) texts.back() += ' ';
}

/** One view per string of @p tape, for an accessor over its bytes or a reordered copy of it. */
inline std::vector<sz_string_view_t> views_of_(pinned_strings_t const &tape) {
    std::vector<sz_string_view_t> views;
    for (string_view_t const text : tape) views.push_back({text.data(), text.size()});
    return views;
}

#pragma region Levenshtein Helpers

/** What a corpus symbol is: a byte the tables class directly, or a rune the page table classes. */
enum class levenshtein_device_alphabet_t { bytes_k, runes_k };

/** The alphabet enumerator an engine is prepared over, matching the corpus alphabet of a check. */
inline sz_levenshtein_symbol_t levenshtein_device_symbol_(levenshtein_device_alphabet_t alphabet) {
    return alphabet == levenshtein_device_alphabet_t::runes_k ? sz_levenshtein_runes_k : sz_levenshtein_bytes_k;
}

/** Corpus symbols: every byte value, or runes across four scripts and all three encoded widths. */
inline std::vector<std::string> levenshtein_device_symbols_(levenshtein_device_alphabet_t alphabet) {
    std::vector<std::string> symbols;
    if (alphabet == levenshtein_device_alphabet_t::bytes_k) {
        for (unsigned value = 0; value != 256; ++value) symbols.push_back(std::string(1, (char)value));
        return symbols;
    }
    char const *const runes[] = {"a", "z",  "é",  "ß",  "α",          "ω",          "ж",
                                 "я", "中", "漢", "字", "\U0001f600", "\U0001f30d", "\U0001d11e"};
    for (char const *const rune : runes) symbols.push_back(std::string(rune));
    return symbols;
}

/**
 *  @brief Near-duplicates of the queries: the candidates in a pinned tape and their device copy,
 *      and the matrix.
 *
 *  The candidates are edits of the queries, so the distances are small and every word of the Myers
 *  state moves. The serial reference walks the pinned candidates and every device round reads their
 *  device copy. Edits land on whole symbols, so a rune corpus stays well-formed and the engine is
 *  told its exact symbol count.
 */
struct levenshtein_device_corpus_t {

    /** The queries, which an engine's builder reads on the host. */
    std::vector<std::string> queries;

    /** The candidates, which the serial reference walks. */
    pinned_strings_t pinned;

    /** The candidates gathered on the device, which every device round reads. */
    sz_strings_view_t device {};

    /** @b [queries, candidates], written by whichever round ran. */
    pinned_vector<sz_size_t> distances;

    sz_strings_view_t query_strings() const { return strings_from_(queries); }

    levenshtein_device_corpus_t(device_backend_t const &runtime, std::size_t count, std::size_t query_symbols,
                                std::size_t query_count, levenshtein_device_alphabet_t alphabet)
        : distances(query_count * count, pinned_alloc_<sz_size_t>(runtime)) {
        std::vector<std::string> const symbols = levenshtein_device_symbols_(alphabet);
        std::vector<std::string> const drawn = fill_queries_(symbols, query_symbols, query_count);
        std::vector<std::string> candidates(count), edited;
        for (std::size_t index = 0; index != count; ++index) {
            edited = drawn;
            for (std::size_t edit = 0, edits = index % 17; edit != edits && edited.size() > 1; ++edit) {
                std::size_t const at = (index * 31 + edit * 7) % edited.size();
                if (edit % 2) { edited.erase(edited.begin() + (std::ptrdiff_t)at); }
                else { edited[at] = symbols[(edit + index) % symbols.size()]; }
            }
            for (std::string const &symbol : edited) candidates[index] += symbol;
        }
        end_off_word_(candidates);
        stage_(runtime, candidates);
    }

    /** Candidates of @p candidate_symbols symbols each, one in four of them drawn from the queries'
     *  own, so a distance moves well below the longer side's length. */
    levenshtein_device_corpus_t(device_backend_t const &runtime, split_mix_generator_t &generator,
                                std::vector<std::size_t> const &candidate_symbols, std::size_t query_symbols,
                                std::size_t query_count, levenshtein_device_alphabet_t alphabet)
        : distances(query_count * candidate_symbols.size(), pinned_alloc_<sz_size_t>(runtime)) {
        std::vector<std::string> const symbols = levenshtein_device_symbols_(alphabet);
        std::vector<std::string> const drawn = fill_queries_(symbols, query_symbols, query_count);
        std::vector<std::string> candidates(candidate_symbols.size());
        for (std::size_t index = 0; index != candidate_symbols.size(); ++index)
            for (std::size_t symbol = 0; symbol != candidate_symbols[index]; ++symbol)
                candidates[index] += generator() % 4 == 0 ? drawn[generator() % drawn.size()]
                                                          : symbols[generator() % symbols.size()];
        stage_(runtime, candidates);
    }

    /** Fills @p query_count queries, every one as long as @p query_symbols and every one
     *  different, and returns the symbols they were edited from. */
    std::vector<std::string> fill_queries_(std::vector<std::string> const &symbols, std::size_t query_symbols,
                                           std::size_t query_count) {
        std::vector<std::string> drawn(query_symbols);
        for (std::size_t symbol = 0; symbol != query_symbols; ++symbol)
            drawn[symbol] = symbols[(symbol * 37 + 11) % symbols.size()];
        // Every query is the same length, so one batch still spans one rung, and every one differs.
        queries.assign(query_count, std::string());
        for (std::size_t query = 0; query != query_count; ++query) {
            std::vector<std::string> edited = drawn;
            for (std::size_t symbol = query; symbol < edited.size(); symbol += query_count + 1)
                edited[symbol] = symbols[(symbol + query * 5 + 3) % symbols.size()];
            for (std::string const &symbol : edited) queries[query] += symbol;
        }
        return drawn;
    }

    /** Stages @p candidates in the pinned tape and gathers the device copy from it. */
    void stage_(device_backend_t const &runtime, std::vector<std::string> const &candidates) {
        pinned = staged_(runtime, candidates);
        device = uploaded_(runtime, pinned.view().raw);
    }

    /** Candidates one round scores, which is also the row stride of the matrix it writes. */
    std::size_t count() const { return pinned.size(); }
};

/** The serial backend's answers for the same corpus, read off the pinned tape the device copied,
 *  into @p expected, one per distance the corpus holds. */
inline void levenshtein_serial_reference_(levenshtein_device_corpus_t const &corpus, sz_levenshtein_symbol_t symbol,
                                          std::span<sz_size_t> expected) {
    sz_strings_view_t const query_strings = corpus.query_strings();
    handle_checked_heap_t heap;
    sz_levenshtein_engine_t engine {};
    expect_equal(sz_levenshtein_engine_init_serial(&engine, &query_strings, symbol, &heap.allocator, nullptr),
                 sz_success_k);
    std::memset(expected.data(), 0, expected.size_bytes());
    sz_strings_view_t const candidates = corpus.pinned.view().raw;
    expect_equal(sz_levenshtein_distances_serial(&engine, &candidates, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                                 expected.data(), corpus.count() * sizeof(sz_size_t), nullptr),
                 sz_success_k);
    sz_levenshtein_engine_free(&engine, nullptr);
    expect_equal(heap.live_allocations, 0);
}

#pragma endregion Levenshtein Helpers

#pragma region Levenshtein Expectations

/** Symbol-pairs one sweep scores before it drops the widest batch, so the serial reference stays
 *  quadratic in time but bounded in it as the rungs widen. */
enum { levenshtein_device_sweep_budget_k = 4u * 1024u * 1024u };

/** Queries one sweep prepares at a time, so @c grid.y carries an axis wider than one per rung. */
enum { levenshtein_device_sweep_queries_k = 3 };

/**
 *  @brief Every query length the sweeps walk, either side of each boundary a rung is cut at.
 *
 *  The narrow rungs share one register between candidates, four ways through eight symbols and two
 *  ways through sixteen, so the symbols either side of 8 and 16. The threaded rung cuts at every
 *  word through its sixteenth, where the warped rung takes over, so the bytes either side of 64,
 *  128 and 1024. The warped rung hands its lanes one more word each at every thirty-second word
 *  past that, so the bytes either side of every multiple of 2048 up to the ceiling itself.
 */
inline constexpr std::size_t levenshtein_device_query_symbols_k[] = {
    1,    2,    4,    8,    9,    16,   17,   63,   64,    65,    127,   128,   129,   1023,  1024,  1025,  2047,  2048,
    2049, 4095, 4096, 4097, 6144, 6145, 8192, 8193, 10240, 10241, 12288, 12289, 14336, 14337, 16383, 16384, 16385,
};

#pragma endregion Levenshtein Expectations

#pragma region Levenshtein Checks

/** One backend's distances against serial's, on a corpus that never leaves the device. */
inline void check_levenshtein_device_equivalence_(auto const &backend, levenshtein_device_alphabet_t alphabet,
                                                  sz_levenshtein_symbol_t symbol) {
    std::vector<sz_size_t> expected;
    for (std::size_t const query_symbols : levenshtein_device_query_symbols_k)
        for (std::size_t count : {1u, 129u, 2048u}) {
            std::size_t const queries = levenshtein_device_sweep_queries_k;
            if (count * query_symbols * queries > (std::size_t)levenshtein_device_sweep_budget_k) continue;
            levenshtein_device_corpus_t corpus(backend.runtime, count, query_symbols, queries, alphabet);
            sz_strings_view_t const query_strings = corpus.query_strings();
            sz_levenshtein_engine_t engine {};
            sz_status_t const prepared = sz_levenshtein_engine_init(&engine, &query_strings, symbol,
                                                                    backend.runtime.capabilities, STRINGZILLA_NULL,
                                                                    backend.runtime.stream);
            if (prepared != sz_success_k)
                fail_backend_(backend.name, "a device batch drew a status the rungs do not imply");
            if (prepared != sz_success_k) continue;
            sz_strings_view_t const candidates = corpus.device;
            expected.resize(corpus.distances.size());
            levenshtein_serial_reference_(corpus, symbol, expected);
            // Unbounded, then half the query's length, which leaves candidates on both sides.
            for (sz_size_t const bound : {STRINGZILLA_SIZE_MAX, (sz_size_t)(query_symbols / 2)}) {
                sz_status_t const produced = backend.levenshtein_distances(
                    &engine, &candidates, nullptr, nullptr, bound, corpus.distances.data(),
                    corpus.count() * sizeof(sz_size_t), backend.runtime.stream);
                join_(backend.runtime, backend.runtime.stream);
                if (produced != sz_success_k) fail_backend_(backend.name, "a device-resident round was refused");
                for (std::size_t index = 0; index != expected.size(); ++index)
                    if (corpus.distances[index] != (expected[index] > bound ? bound + 1 : expected[index]))
                        fail_backend_(backend.name, "a distance differs from serial under its bound");
            }
            // The planes the rounds read live in the engine's block, released after the joins.
            sz_levenshtein_engine_free(&engine, backend.runtime.stream);
        }
}

/** Short queries over a wide batch with varied candidate lengths and alignments. */
inline void check_levenshtein_device_short_queries_(split_mix_generator_t &generator, auto const &backend) {
    enum { arena_bytes_k = 256, longest_candidate_k = 33, offsets_k = 97 };
    handle_checked_heap_t heap;

    std::size_t const count = 4096;
    for (std::size_t const query_symbols : {(std::size_t)8, (std::size_t)16}) {

        // Every third byte is one of the query's own, so a candidate draws both classed
        // and absent symbols.
        std::string query(query_symbols, '\0'), arena(arena_bytes_k, '\0');
        randomize_string(generator, query);
        randomize_string(generator, arena);
        for (std::size_t index = 0; index < arena.size(); index += 3) arena[index] = query[index % query.size()];

        std::vector<sz_string_view_t> query_views(1), views(count);
        pinned_vector<sz_size_t> distances(count, pinned_alloc_<sz_size_t>(backend.runtime));
        query_views[0].start = query.data(), query_views[0].length = query.size();
        for (std::size_t index = 0; index != count; ++index)
            views[index].start = arena.data() + index % offsets_k, views[index].length = index % longest_candidate_k;
        sz_strings_view_t const queries = strings_from_(query_views);
        sz_strings_view_t host_candidates {};
        sz_strings_from_string_views(views.data(), views.size(), &host_candidates);
        pinned_strings_t const staged = staged_(backend.runtime, host_candidates);
        sz_strings_view_t const candidate_strings = uploaded_(backend.runtime, staged.view().raw);

        sz_levenshtein_engine_t engine {};
        expect_equal(sz_levenshtein_engine_init(&engine, &queries, sz_levenshtein_bytes_k, backend.runtime.capabilities,
                                                STRINGZILLA_NULL, backend.runtime.stream),
                     sz_success_k);
        sz_status_t const produced = backend.levenshtein_distances(&engine, &candidate_strings, nullptr, nullptr,
                                                                   STRINGZILLA_SIZE_MAX, distances.data(),
                                                                   count * sizeof(sz_size_t), backend.runtime.stream);
        // The planes the round reads live in the engine's block, so the join comes before
        // it is released.
        join_(backend.runtime, backend.runtime.stream);
        sz_levenshtein_engine_free(&engine, backend.runtime.stream);
        if (produced != sz_success_k) fail_backend_(backend.name, "a short-query batch was refused");

        sz_levenshtein_engine_t host_engine {};
        expect_equal(
            sz_levenshtein_engine_init_serial(&host_engine, &queries, sz_levenshtein_bytes_k, &heap.allocator, nullptr),
            sz_success_k);
        std::vector<sz_size_t> expected(count);
        expect_equal(
            sz_levenshtein_distances_serial(&host_engine, &host_candidates, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                            expected.data(), count * sizeof(sz_size_t), nullptr),
            sz_success_k);
        sz_levenshtein_engine_free(&host_engine, nullptr);
        for (std::size_t index = 0; index != count; ++index)
            if (distances[index] != expected[index])
                fail_backend_(backend.name, "a short-query distance differs from serial");
    }
    expect_equal(heap.live_allocations, 0);
}

/** Symbol-words a skewed batch scores at one query width before that width is skipped, so the
 *  serial reference stays within seconds. */
enum { levenshtein_device_skewed_budget_k = 256u * 1024u * 1024u };

/** Skewed batches against serial, at query widths the threaded rung and the warped one take. */
inline void check_levenshtein_device_skewed_(split_mix_generator_t &generator, auto const &backend,
                                             levenshtein_device_alphabet_t alphabet) {
    sz_levenshtein_symbol_t const symbol = levenshtein_device_symbol_(alphabet);
    std::vector<sz_size_t> expected;
    for (device_skew_t const skew : {device_skew_t::lone_long_k, device_skew_t::log_uniform_k}) {
        std::vector<std::size_t> const lengths = device_skewed_lengths_(generator, skew);
        std::size_t total = 0;
        for (std::size_t const length : lengths) total += length;
        for (std::size_t const query_symbols : {(std::size_t)40, (std::size_t)600, (std::size_t)1500}) {
            if (total * sz_levenshtein_query_words(query_symbols) > (std::size_t)levenshtein_device_skewed_budget_k)
                continue;
            levenshtein_device_corpus_t corpus(backend.runtime, generator, lengths, query_symbols, 2, alphabet);
            sz_strings_view_t const query_strings = corpus.query_strings();
            sz_levenshtein_engine_t engine {};
            expect_equal(sz_levenshtein_engine_init(&engine, &query_strings, symbol, backend.runtime.capabilities,
                                                    STRINGZILLA_NULL, backend.runtime.stream),
                         sz_success_k);
            sz_strings_view_t const candidates = corpus.device;
            sz_status_t const produced = backend.levenshtein_distances(
                &engine, &candidates, nullptr, nullptr, STRINGZILLA_SIZE_MAX, corpus.distances.data(),
                corpus.count() * sizeof(sz_size_t), backend.runtime.stream);
            join_(backend.runtime, backend.runtime.stream);
            sz_levenshtein_engine_free(&engine, backend.runtime.stream);
            if (produced != sz_success_k) fail_backend_(backend.name, "a skewed batch was refused");

            expected.resize(corpus.distances.size());
            levenshtein_serial_reference_(corpus, symbol, expected);
            for (std::size_t index = 0; index != expected.size(); ++index)
                if (corpus.distances[index] != expected[index])
                    fail_backend_(backend.name, "a skewed batch's distance differs from serial");
        }
    }
}

inline void check_levenshtein_device_mixed_(auto const &backend) {
    std::string unicode;
    for (std::size_t index = 0; index != 16641; ++index) unicode += "é";
    std::array<std::string, 9> const texts {std::string(65, 'b'),    "",
                                            std::string(32769, 'c'), "k\xFFtten",
                                            std::string(16384, 'a'), std::string(63, 'd'),
                                            std::string(32768, 'e'), unicode,
                                            std::string(64, 'f')};
    std::array<sz_string_view_t, texts.size()> views;
    for (std::size_t index = 0; index != texts.size(); ++index)
        views[index] = {texts[index].data(), texts[index].size()};
    sz_strings_view_t queries {};
    sz_strings_from_string_views(views.data(), views.size(), &queries);
    std::array<std::string, 19> candidates_text {"x",
                                                 std::string(16385, 'a'),
                                                 "",
                                                 "kitten",
                                                 "é",
                                                 "f",
                                                 std::string(65, 'b'),
                                                 "\xFF",
                                                 std::string(32768, 'e'),
                                                 "dd",
                                                 "abc",
                                                 std::string(64, 'f'),
                                                 "ef",
                                                 std::string(16384, 'a'),
                                                 "éé",
                                                 std::string(63, 'd'),
                                                 std::string(32769, 'c'),
                                                 "c",
                                                 "ba"};
    std::array<sz_string_view_t, candidates_text.size()> candidate_views;
    std::size_t const stride = candidates_text.size() + 3;
    pinned_vector<sz_size_t> distances(texts.size() * stride, STRINGZILLA_SIZE_MAX,
                                       pinned_alloc_<sz_size_t>(backend.runtime));
    for (sz_levenshtein_symbol_t symbol : {sz_levenshtein_bytes_k, sz_levenshtein_runes_k}) {
        handle_checked_heap_t heap;
        sz_levenshtein_engine_t host {}, device {};
        expect_equal(sz_levenshtein_engine_init_serial(&host, &queries, symbol, &heap.allocator, nullptr),
                     sz_success_k);
        expect_equal(sz_levenshtein_engine_init(&device, &queries, symbol, backend.runtime.capabilities, nullptr,
                                                backend.runtime.stream),
                     sz_success_k);
        std::vector<sz_size_t> expected(distances.size(), STRINGZILLA_SIZE_MAX);
        for (bool reverse : {false, true}) {
            for (std::size_t index = 0; index != candidate_views.size(); ++index) {
                std::string const &text = candidates_text[reverse ? candidate_views.size() - 1 - index : index];
                candidate_views[index] = {text.data(), text.size()};
            }
            sz_strings_view_t reference {};
            sz_strings_from_string_views(candidate_views.data(), candidate_views.size(), &reference);
            pinned_strings_t const staged = staged_(backend.runtime, reference);
            sz_strings_view_t const candidates = uploaded_(backend.runtime, staged.view().raw);
            std::fill(distances.begin(), distances.end(), STRINGZILLA_SIZE_MAX);
            if (!reverse)
                expect_equal(sz_levenshtein_distances_serial(&host, &reference, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                                             expected.data(), stride * sizeof(sz_size_t), nullptr),
                             sz_success_k);
            else
                for (std::size_t row = 0; row != texts.size(); ++row)
                    std::reverse(expected.begin() + row * stride,
                                 expected.begin() + row * stride + candidate_views.size());
            expect_equal(
                backend.levenshtein_distances(&device, &candidates, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                              distances.data(), stride * sizeof(sz_size_t), backend.runtime.stream),
                sz_success_k);
            join_(backend.runtime, backend.runtime.stream);
            expect_true(std::equal(distances.begin(), distances.end(), expected.begin()));
        }
        sz_levenshtein_engine_free(&device, backend.runtime.stream);
        sz_levenshtein_engine_free(&host, nullptr);
        expect_equal(heap.live_allocations, 0);
    }
    {
        std::array<std::string, 40> texts;
        std::array<sz_string_view_t, 40> query_views, candidate_views;
        for (std::size_t index = 0; index != texts.size(); ++index) {
            texts[index].assign(17000 + index, 'a');
            query_views[index] = {texts[index].data(), texts[index].size()};
        }
        std::reverse_copy(query_views.begin(), query_views.end(), candidate_views.begin());
        sz_strings_view_t queries {};
        sz_strings_from_string_views(query_views.data(), query_views.size(), &queries);
        pinned_strings_t const staged = staged_(backend.runtime, candidate_views);
        sz_strings_view_t const candidates = uploaded_(backend.runtime, staged.view().raw);
        std::size_t const stride = 43;
        pinned_vector<sz_size_t> distances(texts.size() * stride, STRINGZILLA_SIZE_MAX,
                                           pinned_alloc_<sz_size_t>(backend.runtime));
        sz_levenshtein_engine_t engine {};
        expect_equal(sz_levenshtein_engine_init(&engine, &queries, sz_levenshtein_bytes_k, backend.runtime.capabilities,
                                                nullptr, backend.runtime.stream),
                     sz_success_k);
        expect_equal(
            backend.levenshtein_distances(&engine, &candidates, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                          distances.data(), stride * sizeof(sz_size_t), backend.runtime.stream),
            sz_success_k);
        join_(backend.runtime, backend.runtime.stream);
        for (std::size_t row = 0; row != texts.size(); ++row) {
            for (std::size_t column = 0; column != candidate_views.size(); ++column) {
                std::size_t const query_length = query_views[row].length,
                                  candidate_length = candidate_views[column].length;
                expect_equal(distances[row * stride + column], query_length > candidate_length
                                                                   ? query_length - candidate_length
                                                                   : candidate_length - query_length);
            }
            for (std::size_t column = candidate_views.size(); column != stride; ++column)
                expect_equal(distances[row * stride + column], STRINGZILLA_SIZE_MAX);
        }
        sz_levenshtein_engine_free(&engine, backend.runtime.stream);
    }
}

/** One backend refusing host memory, accessors and outputs alike, rather than staging it. */
inline void check_levenshtein_device_memory_safety_(split_mix_generator_t &generator, auto const &backend) {
    levenshtein_device_corpus_t corpus(backend.runtime, 8, 200, 1, levenshtein_device_alphabet_t::bytes_k);
    std::fill(corpus.distances.begin(), corpus.distances.end(), STRINGZILLA_SIZE_MAX);
    sz_strings_view_t const query_strings = corpus.query_strings();
    sz_levenshtein_engine_t engine {};
    expect_equal(sz_levenshtein_engine_init(&engine, &query_strings, sz_levenshtein_bytes_k,
                                            backend.runtime.capabilities, STRINGZILLA_NULL, backend.runtime.stream),
                 sz_success_k);
    std::string text(64, '\0');
    randomize_string(generator, text);
    std::array<sz_string_view_t, 1> const host_views {sz_string_view_t {text.data(), text.size()}};
    sz_strings_view_t host_candidates {};
    sz_strings_from_string_views(host_views.data(), host_views.size(), &host_candidates);
    std::vector<sz_size_t> host_distances(corpus.distances.size(), STRINGZILLA_SIZE_MAX);
    sz_strings_view_t const candidates = corpus.device;
    expect_equal(backend.levenshtein_distances(&engine, &candidates, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                               host_distances.data(), corpus.count() * sizeof(sz_size_t),
                                               backend.runtime.stream),
                 sz_device_memory_mismatch_k);
    expect_equal(backend.levenshtein_distances(&engine, &host_candidates, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                               corpus.distances.data(), corpus.count() * sizeof(sz_size_t),
                                               backend.runtime.stream),
                 sz_device_memory_mismatch_k);
    expect_equal(backend.levenshtein_distances(&engine, &candidates, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                               corpus.distances.data(), (corpus.count() - 1) * sizeof(sz_size_t),
                                               backend.runtime.stream),
                 sz_unexpected_dimensions_k);
    // An accessor is host code, so it is refused even over bytes the device reaches.
    std::vector<sz_string_view_t> const resident_views = views_of_(corpus.pinned);
    sz_strings_view_t const resident_accessor = strings_from_(resident_views);
    expect_equal(backend.levenshtein_distances(&engine, &resident_accessor, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                               corpus.distances.data(), corpus.count() * sizeof(sz_size_t),
                                               backend.runtime.stream),
                 sz_device_memory_mismatch_k);
    join_(backend.runtime, backend.runtime.stream);
    sz_levenshtein_engine_free(&engine, backend.runtime.stream);
    for (sz_size_t const untouched : host_distances) expect_equal(untouched, STRINGZILLA_SIZE_MAX);
    for (sz_size_t const untouched : corpus.distances) expect_equal(untouched, STRINGZILLA_SIZE_MAX);
}

/** Two threads scoring on two queues of one device at once, each joining only its own: encoders
 *  live per call, and the device's shared state sits under one lock. */
inline void check_levenshtein_device_threads_(auto const &backend) {
    sz_stream_t const queue = backend.runtime.stream;
    stream_t const second(backend.runtime.selected.ordinal, backend.runtime.init, backend.runtime.free);
    levenshtein_device_corpus_t corpus(backend.runtime, 3, 16385, 2, levenshtein_device_alphabet_t::bytes_k);
    sz_strings_view_t const queries = corpus.query_strings(), candidates = corpus.device;
    sz_levenshtein_engine_t engine {};
    expect_equal(sz_levenshtein_engine_init(&engine, &queries, sz_levenshtein_bytes_k, backend.runtime.capabilities,
                                            nullptr, queue),
                 sz_success_k);
    join_(backend.runtime, queue);
    std::vector<sz_size_t> expected(corpus.distances.size());
    levenshtein_serial_reference_(corpus, sz_levenshtein_bytes_k, expected);
    pinned_vector<sz_size_t> other_distances(corpus.distances.size(), STRINGZILLA_SIZE_MAX,
                                             pinned_alloc_<sz_size_t>(backend.runtime));
    std::vector<sz_string_view_t> reversed_views = views_of_(corpus.pinned);
    std::reverse(reversed_views.begin(), reversed_views.end());
    pinned_strings_t const reversed_staged = staged_(backend.runtime, reversed_views);
    sz_strings_view_t const reversed = uploaded_(backend.runtime, reversed_staged.view().raw);
    std::vector<sz_size_t> reversed_expected = expected;
    for (std::size_t row = 0; row != corpus.queries.size(); ++row)
        std::reverse(reversed_expected.begin() + row * corpus.count(),
                     reversed_expected.begin() + (row + 1) * corpus.count());
    join_(backend.runtime, queue);
    auto const round = [&](sz_stream_t stream, sz_strings_view_t const &inputs, sz_size_t *output,
                           std::vector<sz_size_t> const &reference) {
        expect_equal(sz_levenshtein_distances(&engine, &inputs, nullptr, nullptr, STRINGZILLA_SIZE_MAX, output,
                                              corpus.count() * sizeof(sz_size_t), stream),
                     sz_success_k);
        join_(backend.runtime, stream);
        if (!std::equal(reference.begin(), reference.end(), output))
            fail_backend_("dispatched", "concurrent scoring of one engine differs from serial");
    };
    std::thread other([&] { round(second.handle, reversed, other_distances.data(), reversed_expected); });
    round(queue, candidates, corpus.distances.data(), expected);
    other.join();
    sz_levenshtein_engine_free(&engine, queue);
}

/** Releasing a queued input retains it until the stream finishes reading it: the pinned tape, as
 *  the arena's tapes are never released one by one. */
inline void check_levenshtein_device_deferred_free_(auto const &backend) {
    sz_stream_t const queue = backend.runtime.stream;
    levenshtein_device_corpus_t corpus(backend.runtime, 129, 1024, levenshtein_device_sweep_queries_k,
                                       levenshtein_device_alphabet_t::bytes_k);
    sz_strings_view_t const query_strings = corpus.query_strings();
    std::vector<sz_size_t> expected(corpus.distances.size());
    levenshtein_serial_reference_(corpus, sz_levenshtein_bytes_k, expected);
    sz_levenshtein_engine_t engine {};
    expect_equal(sz_levenshtein_engine_init(&engine, &query_strings, sz_levenshtein_bytes_k,
                                            backend.runtime.capabilities, STRINGZILLA_NULL, queue),
                 sz_success_k);
    sz_strings_view_t const freed = corpus.pinned.view().raw;
    expect_equal(sz_levenshtein_distances(&engine, &freed, nullptr, nullptr, STRINGZILLA_SIZE_MAX,
                                          corpus.distances.data(), corpus.count() * sizeof(sz_size_t), queue),
                 sz_success_k);
    corpus.pinned = pinned_strings_t {};
    expect_equal(sz_stream_synchronize_best(backend.runtime.capabilities, queue), sz_success_k);
    for (std::size_t index = 0; index != expected.size(); ++index)
        if (corpus.distances[index] != expected[index])
            fail_backend_("dispatched", "a round whose tape was freed under it differs from serial");
    sz_levenshtein_engine_free(&engine, queue);
}

#pragma endregion Levenshtein Checks

#pragma region Ngrams Helpers

/**
 *  @brief Candidates in a pinned tape and their device copy, and room for one round's counts.
 *
 *  The serial reference walks the pinned candidates and every device round reads their device
 *  copy. The queries stay plain host strings, because an engine's builder reads them on the host
 *  and no kernel ever sees them.
 */
struct ngrams_overlaps_corpus_t {

    /** The texts whose N-grams fill the tables. */
    std::vector<std::string> queries;

    /** The candidates, which the serial reference walks. */
    pinned_strings_t pinned;

    /** The candidates gathered on the device, which every device round reads. */
    sz_strings_view_t device {};

    /** @b [queries,candidates,widths] shares, written by whichever verb ran. */
    pinned_vector<sz_f32_t> shares;

    /** Host accessors, which is what an engine's builder calls. */
    sz_strings_view_t query_strings() const { return strings_from_(queries); }

    ngrams_overlaps_corpus_t(device_backend_t const &runtime, split_mix_generator_t &generator,
                             std::size_t queries_count, std::size_t count, std::size_t query_length,
                             std::size_t widths_count)
        : shares(queries_count * count * widths_count, pinned_alloc_<sz_f32_t>(runtime)) {
        fill_queries_(generator, queries_count, query_length);
        std::vector<std::string> candidates(count);
        for (std::size_t index = 0; index != count; ++index) {
            candidates[index].resize(1 + (index * 37) % 900);
            randomize_string(generator, candidates[index]);
        }
        end_off_word_(candidates);
        stage_(runtime, candidates);
    }

    /** Candidates of @p lengths bytes, spliced from spans of the queries between runs of random
     *  bytes, so N-grams match across wherever a kernel cuts a candidate. */
    ngrams_overlaps_corpus_t(device_backend_t const &runtime, split_mix_generator_t &generator,
                             std::size_t queries_count, std::vector<std::size_t> const &lengths,
                             std::size_t query_length, std::size_t widths_count)
        : shares(queries_count * lengths.size() * widths_count, pinned_alloc_<sz_f32_t>(runtime)) {
        fill_queries_(generator, queries_count, query_length);
        std::size_t total = 0;
        for (std::size_t const length : lengths) total += length;
        std::string arena(total, '\0');
        randomize_string(generator, arena);
        for (std::size_t at = 0; at < total;) {
            std::string const &query = queries[generator() % queries.size()];
            std::size_t const span = min_of(min_of((std::size_t)(1 + generator() % 128), query.size()), total - at);
            std::size_t const from = generator() % (query.size() - span + 1);
            std::copy(query.begin() + (std::ptrdiff_t)from, query.begin() + (std::ptrdiff_t)(from + span),
                      arena.begin() + (std::ptrdiff_t)at);
            at += span + generator() % 9;
        }
        std::vector<std::string> candidates(lengths.size());
        for (std::size_t index = 0, written = 0; index != lengths.size(); written += lengths[index++])
            candidates[index] = arena.substr(written, lengths[index]);
        stage_(runtime, candidates);
    }

    /** Draws the query texts, each a different length below @p query_length. */
    void fill_queries_(split_mix_generator_t &generator, std::size_t queries_count, std::size_t query_length) {
        for (std::size_t index = 0; index != queries_count; ++index) {
            std::string text(query_length ? query_length - index % query_length : 0, '\0');
            randomize_string(generator, text);
            queries.push_back(text);
        }
    }

    /** Stages @p candidates in the pinned tape and gathers the device copy from it. */
    void stage_(device_backend_t const &runtime, std::vector<std::string> const &candidates) {
        pinned = staged_(runtime, candidates);
        device = uploaded_(runtime, pinned.view().raw);
    }

    std::size_t candidate_stride() const noexcept {
        return shares.size() / (queries.size() * pinned.size()) * sizeof(sz_f32_t);
    }
    std::size_t query_stride() const noexcept { return pinned.size() * candidate_stride(); }
};

/** The serial backend's answers for the same corpus, read off the pinned tape the device copied,
 *  into @p expected, one per share the corpus holds. */
inline void ngrams_overlaps_serial_reference_(ngrams_overlaps_corpus_t const &corpus, std::span<sz_size_t const> widths,
                                              sz_f32_t radius, std::span<sz_f32_t> expected) {
    sz_strings_view_t const query_strings = corpus.query_strings();
    handle_checked_heap_t heap;
    sz_ngrams_engine_t engine {};
    expect_equal(sz_ngrams_engine_init_serial(&engine, &query_strings, widths.data(), widths.size(), 0, &heap.allocator,
                                              nullptr),
                 sz_success_k);
    std::memset(expected.data(), 0xFF, expected.size_bytes());
    sz_strings_view_t const candidates = corpus.pinned.view().raw;
    expect_equal(sz_ngrams_overlaps_serial(&engine, &candidates, radius, expected.data(), corpus.query_stride(),
                                           corpus.candidate_stride(), nullptr),
                 sz_success_k);
    sz_ngrams_engine_free(&engine, nullptr);
    expect_equal(heap.live_allocations, 0);
}

#pragma endregion Ngrams Helpers

#pragma region Ngrams Overlaps Checks

/** One backend's shares against serial's, bit for bit, on a corpus that never leaves the device,
 *  unbounded and at radii that keep every pair, some, and only identical ones. */
inline void check_ngrams_overlaps_device_equivalence_(split_mix_generator_t &generator, auto const &backend) {
    std::array<sz_size_t, 3> const widths {4, 6, 8};
    std::vector<sz_f32_t> expected;

    for (std::size_t queries_count : {1u, 3u})
        for (std::size_t count : {1u, 129u, 4096u})
            for (std::size_t query_length : {12u, 777u}) {
                ngrams_overlaps_corpus_t corpus(backend.runtime, generator, queries_count, count, query_length,
                                                widths.size());
                sz_strings_view_t const query_strings = corpus.query_strings();
                sz_ngrams_engine_t engine {};
                if (sz_ngrams_engine_init(&engine, &query_strings, widths.data(), widths.size(), 0,
                                          backend.runtime.capabilities, STRINGZILLA_NULL,
                                          backend.runtime.stream) != sz_success_k)
                    fail_backend_(backend.name, "a device engine was refused for a well-formed batch");
                sz_strings_view_t const candidates = corpus.device;
                expected.resize(corpus.shares.size());
                for (sz_f32_t const radius : {1.0f, 0.0f, 0.5f}) {
                    ngrams_overlaps_serial_reference_(corpus, widths, radius, expected);
                    if (backend.ngrams_overlaps(&engine, &candidates, radius, corpus.shares.data(),
                                                corpus.query_stride(), corpus.candidate_stride(),
                                                backend.runtime.stream) != sz_success_k)
                        fail_backend_(backend.name, "a device-resident batch was refused");
                    join_(backend.runtime, backend.runtime.stream);
                    if (std::memcmp(corpus.shares.data(), expected.data(), expected.size() * sizeof(sz_f32_t)) != 0)
                        fail_backend_(backend.name, "a share differs from serial");
                }
                sz_ngrams_engine_free(&engine, backend.runtime.stream);
            }
}

/** Skewed batches against serial, at narrow widths and at widths past 32 bytes, which most of the
 *  lone-long batch's candidates are too short to hold. */
inline void check_ngrams_overlaps_device_skewed_(split_mix_generator_t &generator, auto const &backend) {
    std::array<sz_size_t, 4> const widths {3, 8, 33, 100};
    std::vector<sz_f32_t> expected;
    for (device_skew_t const skew : {device_skew_t::lone_long_k, device_skew_t::log_uniform_k}) {
        std::vector<std::size_t> const lengths = device_skewed_lengths_(generator, skew);
        ngrams_overlaps_corpus_t corpus(backend.runtime, generator, 2, lengths, 777, widths.size());
        sz_strings_view_t const query_strings = corpus.query_strings();
        expected.resize(corpus.shares.size());
        ngrams_overlaps_serial_reference_(corpus, widths, 1.0f, expected);
        // The lone-long batch carries more candidates than Metal's default round budget.
        sz_ngrams_engine_t engine {};
        if (sz_ngrams_engine_init(&engine, &query_strings, widths.data(), widths.size(), lengths.size(),
                                  backend.runtime.capabilities, STRINGZILLA_NULL,
                                  backend.runtime.stream) != sz_success_k)
            fail_backend_(backend.name, "a device engine was refused for a well-formed batch");
        sz_strings_view_t const candidates = corpus.device;
        if (backend.ngrams_overlaps(&engine, &candidates, 1.0f, corpus.shares.data(), corpus.query_stride(),
                                    corpus.candidate_stride(), backend.runtime.stream) != sz_success_k)
            fail_backend_(backend.name, "a skewed batch was refused");
        join_(backend.runtime, backend.runtime.stream);
        sz_ngrams_engine_free(&engine, backend.runtime.stream);
        if (std::memcmp(corpus.shares.data(), expected.data(), expected.size() * sizeof(sz_f32_t)) != 0)
            fail_backend_(backend.name, "a skewed batch's share differs from serial");
    }
}

/** One backend refusing host memory no kernel can address, rather than reading a bad pointer. */
inline void check_ngrams_overlaps_device_memory_safety_(split_mix_generator_t &generator, auto const &backend) {
    std::array<sz_size_t, 2> const widths {4, 6};
    ngrams_overlaps_corpus_t corpus(backend.runtime, generator, 1, 8, 333, widths.size());
    sz_f32_t const unwritten = std::numeric_limits<sz_f32_t>::quiet_NaN();
    std::fill(corpus.shares.begin(), corpus.shares.end(), unwritten);
    sz_strings_view_t const query_strings = corpus.query_strings();
    sz_ngrams_engine_t engine {};
    if (sz_ngrams_engine_init(&engine, &query_strings, widths.data(), widths.size(), 0, backend.runtime.capabilities,
                              STRINGZILLA_NULL, backend.runtime.stream) != sz_success_k)
        fail_backend_(backend.name, "a device engine was refused for a well-formed batch");

    // The corpus's own tapes live in pinned and device memory, so host strings need plain storage.
    std::string text(64, '\0');
    randomize_string(generator, text);
    std::array<sz_string_view_t, 1> const host_views {sz_string_view_t {text.data(), text.size()}};
    sz_strings_view_t host_candidates {};
    sz_strings_from_string_views(host_views.data(), host_views.size(), &host_candidates);

    std::vector<sz_f32_t> host_shares(corpus.shares.size(), unwritten);
    sz_strings_view_t const candidates = corpus.device;
    if (backend.ngrams_overlaps(&engine, &candidates, 1.0f, host_shares.data(), corpus.query_stride(),
                                corpus.candidate_stride(), backend.runtime.stream) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host shares were not refused");
    if (backend.ngrams_overlaps(&engine, &host_candidates, 1.0f, corpus.shares.data(), corpus.query_stride(),
                                corpus.candidate_stride(), backend.runtime.stream) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host-resident candidates were not refused");
    // An accessor is host code, so it is refused even over bytes the device reaches.
    std::vector<sz_string_view_t> const resident_views = views_of_(corpus.pinned);
    sz_strings_view_t const resident_accessor = strings_from_(resident_views);
    expect_equal(backend.ngrams_overlaps(&engine, &resident_accessor, 1.0f, corpus.shares.data(), corpus.query_stride(),
                                         corpus.candidate_stride(), backend.runtime.stream),
                 sz_device_memory_mismatch_k);
    // Metal reserves per-candidate storage; CUDA and ROCm ignore this budget.
    if (backend.runtime.selected.backend == sz::device_kind_t::metal_k) {
        sz_ngrams_engine_t narrow {};
        expect_equal(
            sz_ngrams_engine_init(&narrow, &query_strings, widths.data(), widths.size(), corpus.pinned.size() - 1,
                                  backend.runtime.capabilities, STRINGZILLA_NULL, backend.runtime.stream),
            sz_success_k);
        expect_equal(backend.ngrams_overlaps(&narrow, &candidates, 1.0f, corpus.shares.data(), corpus.query_stride(),
                                             corpus.candidate_stride(), backend.runtime.stream),
                     sz_unexpected_dimensions_k);
        sz_ngrams_engine_free(&narrow, backend.runtime.stream);
    }
    join_(backend.runtime, backend.runtime.stream);
    for (sz_f32_t const &untouched : corpus.shares)
        expect_equal(std::memcmp(&untouched, &unwritten, sizeof(unwritten)), 0);
    for (sz_f32_t const &untouched : host_shares)
        if (std::memcmp(&untouched, &unwritten, sizeof(unwritten)) != 0)
            fail_backend_(backend.name, "a refused call still wrote a share");
    sz_ngrams_engine_free(&engine, backend.runtime.stream);
}

/** Known answers past 32 bytes, at a text's own length and one past it, over more widths than a
 *  Metal walk rolls together: a text holds every window of its own and of its first half. */
inline void check_ngrams_overlaps_device_width_safety_(split_mix_generator_t &generator, auto const &backend) {
    std::array<sz_size_t, 17> const widths {1, 2, 3, 5, 8, 13, 21, 31, 32, 33, 64, 100, 128, 129, 200, 256, 257};

    // Both candidates are the query's own bytes, whole and halved.
    ngrams_overlaps_corpus_t corpus(backend.runtime, generator, 1, 2, 256, widths.size());
    sz_strings_view_t const query_strings = corpus.query_strings();
    std::string const &query = corpus.queries.front();
    std::vector<std::string> const candidates {query, query.substr(0, query.size() / 2)};
    corpus.stage_(backend.runtime, candidates);

    sz_ngrams_engine_t engine {};
    if (sz_ngrams_engine_init(&engine, &query_strings, widths.data(), widths.size(), 0, backend.runtime.capabilities,
                              STRINGZILLA_NULL, backend.runtime.stream) != sz_success_k)
        fail_backend_(backend.name, "a device engine refused widths the host takes");
    sz_strings_view_t const itself_strings = corpus.device;
    if (backend.ngrams_overlaps(&engine, &itself_strings, 1.0f, corpus.shares.data(), corpus.query_stride(),
                                corpus.candidate_stride(), backend.runtime.stream) != sz_success_k)
        fail_backend_(backend.name, "a device-resident batch was refused");
    join_(backend.runtime, backend.runtime.stream);
    sz_ngrams_engine_free(&engine, backend.runtime.stream);
    for (std::size_t candidate = 0; candidate != 2; ++candidate)
        for (std::size_t index = 0; index != widths.size(); ++index)
            if (corpus.shares[candidate * widths.size() + index] !=
                ngrams_reference_overlap_(query, candidates[candidate], widths[index]))
                fail_backend_(backend.name, "a text and its half do not overlap it as the oracle says");
}

/** Widths past every query, as @c ngrams_past_query_case_ lays them out, count as the oracle says:
 *  a window opening on zero bytes at such a width matches nothing. */
inline void check_ngrams_overlaps_device_past_query_(split_mix_generator_t &generator, auto const &backend) {
    ngrams_case_t const past_query = ngrams_past_query_case_();
    std::vector<sz_size_t> const widths(past_query.widths.begin(), past_query.widths.end());
    ngrams_overlaps_corpus_t corpus(backend.runtime, generator, past_query.queries.size(), past_query.candidates.size(),
                                    1, widths.size());
    corpus.queries = past_query.queries;
    corpus.stage_(backend.runtime, past_query.candidates);
    sz_strings_view_t const query_strings = corpus.query_strings();
    sz_ngrams_engine_t engine {};
    if (sz_ngrams_engine_init(&engine, &query_strings, widths.data(), widths.size(), 0, backend.runtime.capabilities,
                              STRINGZILLA_NULL, backend.runtime.stream) != sz_success_k)
        fail_backend_(backend.name, "a device engine refused widths the host takes");
    if (backend.ngrams_overlaps(&engine, &corpus.device, 1.0f, corpus.shares.data(), corpus.query_stride(),
                                corpus.candidate_stride(), backend.runtime.stream) != sz_success_k)
        fail_backend_(backend.name, "a device-resident batch was refused");
    join_(backend.runtime, backend.runtime.stream);
    sz_ngrams_engine_free(&engine, backend.runtime.stream);
    for (std::size_t query = 0; query != corpus.queries.size(); ++query)
        for (std::size_t candidate = 0; candidate != past_query.candidates.size(); ++candidate)
            for (std::size_t index = 0; index != widths.size(); ++index)
                if (corpus.shares[(query * past_query.candidates.size() + candidate) * widths.size() + index] !=
                    ngrams_reference_overlap_(corpus.queries[query], past_query.candidates[candidate], widths[index]))
                    fail_backend_(backend.name, "a width past the query counted a window");
}

/** Two streams score one prepared engine and produce the same answers after synchronization. */
inline void check_ngrams_overlaps_device_streams_(split_mix_generator_t &generator, auto const &backend) {
    std::array<sz_size_t, 3> const widths {4, 6, 8};
    ngrams_overlaps_corpus_t corpus(backend.runtime, generator, 8, 4096, 777, widths.size());
    sz_strings_view_t const query_strings = corpus.query_strings();
    std::vector<sz_f32_t> expected(corpus.shares.size());
    ngrams_overlaps_serial_reference_(corpus, widths, 1.0f, expected);
    pinned_vector<sz_f32_t> second(corpus.shares.size(), -1.0f, pinned_alloc_<sz_f32_t>(backend.runtime));
    stream_t const first_stream(backend.runtime.selected.ordinal, backend.runtime.init, backend.runtime.free);
    stream_t const second_stream(backend.runtime.selected.ordinal, backend.runtime.init, backend.runtime.free);

    sz_ngrams_engine_t engine {};
    expect_equal(sz_ngrams_engine_init(&engine, &query_strings, widths.data(), widths.size(), 0,
                                       backend.runtime.capabilities, STRINGZILLA_NULL, backend.runtime.stream),
                 sz_success_k);
    sz_strings_view_t const candidates = corpus.device;
    expect_equal(sz_ngrams_overlaps(&engine, &candidates, 1.0f, corpus.shares.data(), corpus.query_stride(),
                                    corpus.candidate_stride(), first_stream.handle),
                 sz_success_k);
    expect_equal(sz_ngrams_overlaps(&engine, &candidates, 1.0f, second.data(), corpus.query_stride(),
                                    corpus.candidate_stride(), second_stream.handle),
                 sz_success_k);
    join_(backend.runtime, first_stream.handle);
    join_(backend.runtime, second_stream.handle);
    sz_ngrams_engine_free(&engine, backend.runtime.stream);
    std::size_t const bytes = expected.size() * sizeof(sz_f32_t);
    if (std::memcmp(corpus.shares.data(), expected.data(), bytes) != 0 ||
        std::memcmp(second.data(), expected.data(), bytes) != 0)
        fail_backend_("dispatched", "two streams scoring one engine disagreed with serial");
}

#pragma endregion Ngrams Overlaps Checks

#pragma region Ngrams Sketches Checks

/** Texts of mixed alphabets for a device round, in a pinned tape the serial reference walks and in
 *  the device's own copy, with room for sketches of @c slots bytes, @c stride bytes apart. */
struct ngrams_sketches_corpus_t {

    /** The texts, which the serial reference walks. */
    pinned_strings_t pinned;

    /** The texts gathered on the device, which every device round reads. */
    sz_strings_view_t device {};

    /** One sketch per text, over a fill of 0x55. */
    pinned_vector<sz_u8_t> sketches;

    /** Bytes of one sketch. */
    std::size_t slots;

    /** Bytes from one sketch to the next, three past the slots. */
    std::size_t stride;

    ngrams_sketches_corpus_t(device_backend_t const &runtime, split_mix_generator_t &generator,
                             std::vector<std::size_t> const &lengths, std::size_t slots = 512)
        : sketches(lengths.size() * (slots + 3), 0x55, pinned_alloc_<sz_u8_t>(runtime)), slots(slots),
          stride(slots + 3) {
        std::vector<std::string> texts(lengths.size());
        for (std::size_t index = 0; index != lengths.size(); ++index) {
            if (index % 3 == 0) texts[index].resize(lengths[index]), randomize_string(generator, texts[index]);
            else if (index % 3 == 1) texts[index] = random_string(generator, lengths[index], "ab");
            else texts[index] = random_string(generator, lengths[index], alphabet_characters("aZéЖ€𝄞"));
        }
        end_off_word_(texts);
        pinned = staged_(runtime, texts);
        device = uploaded_(runtime, pinned.view().raw);
    }

    /** The serial kernels' sketches of the pinned texts at @p widths, laid out as the device's. */
    std::vector<sz_u8_t> serial_reference(std::vector<sz_size_t> const &widths) const {
        handle_checked_heap_t heap;
        sz_ngrams_engine_t engine {};
        expect_equal(
            sz_ngrams_engine_init_serial(&engine, nullptr, widths.data(), widths.size(), 0, &heap.allocator, nullptr),
            sz_success_k);
        std::vector<sz_u8_t> expected(sketches.size(), 0x55);
        sz_strings_view_t const texts = pinned.view().raw;
        expect_equal(sz_ngrams_sketches_serial(&engine, &texts, slots, expected.data(), stride, nullptr), sz_success_k);
        sz_ngrams_engine_free(&engine, nullptr);
        expect_equal(heap.live_allocations, 0);
        return expected;
    }
};

/** Text lengths of the sketch rounds: one per text, every one short of a warp's words to several
 *  times them. */
inline std::vector<std::size_t> ngrams_sketches_lengths_(std::size_t count) {
    std::vector<std::size_t> lengths(count);
    for (std::size_t index = 0; index != count; ++index) lengths[index] = (index * 37) % 211;
    return lengths;
}

/** One backend's sketches of @p corpus at @p widths against serial's, every byte of the strided
 *  output included, so a lane writing past its slots shows. */
inline void check_ngrams_sketches_device_round_(auto const &backend, ngrams_sketches_corpus_t &corpus,
                                                std::vector<sz_size_t> const &widths) {
    std::vector<sz_u8_t> const expected = corpus.serial_reference(widths);
    sz_ngrams_engine_t engine {};
    if (sz_ngrams_engine_init(&engine, nullptr, widths.data(), widths.size(), 0, backend.runtime.capabilities,
                              STRINGZILLA_NULL, backend.runtime.stream) != sz_success_k)
        fail_backend_(backend.name, "a sketching engine was refused for well-formed widths");
    if (backend.ngrams_sketches(&engine, &corpus.device, corpus.slots, corpus.sketches.data(), corpus.stride,
                                backend.runtime.stream) != sz_success_k)
        fail_backend_(backend.name, "a device-resident batch was refused");
    join_(backend.runtime, backend.runtime.stream);
    sz_ngrams_engine_free(&engine, backend.runtime.stream);
    if (!std::equal(expected.begin(), expected.end(), corpus.sketches.begin()))
        fail_backend_(backend.name, "a sketch differs from serial");
}

/** One backend's sketches against serial's, of texts that never leave the device, at one width and
 *  at many, in batches of one warp, of several and of many blocks, at slot counts inside one lane's
 *  share, across lanes and past a warp's 32 a pass, and over skewed batches whose long texts spread
 *  their N-grams over every lane. */
inline void check_ngrams_sketches_device_equivalence_(split_mix_generator_t &generator, auto const &backend) {
    for (std::vector<sz_size_t> const &widths : std::vector<std::vector<sz_size_t>> {{5, 8}, {1}, {3, 70, 7, 2}})
        for (std::size_t count : {1u, 129u, 1000u}) {
            ngrams_sketches_corpus_t corpus(backend.runtime, generator, ngrams_sketches_lengths_(count));
            check_ngrams_sketches_device_round_(backend, corpus, widths);
        }
    for (std::size_t slots : {1u, 3u, 4u, 17u, 64u, 513u, 1024u}) {
        ngrams_sketches_corpus_t corpus(backend.runtime, generator, ngrams_sketches_lengths_(129), slots);
        check_ngrams_sketches_device_round_(backend, corpus, {3, 70, 7, 2});
    }
    ngrams_sketches_corpus_t skewed(backend.runtime, generator,
                                    device_skewed_lengths_(generator, device_skew_t::log_uniform_k), 513);
    check_ngrams_sketches_device_round_(backend, skewed, {5, 8, 33});
}

/** One backend refusing host memory no kernel can address, a stride under the slots, no slots,
 *  ranks past what the device holds per text and widths it cannot take, rather than reading a bad
 *  pointer or writing one sketch into its neighbour's. */
inline void check_ngrams_sketches_device_memory_safety_(split_mix_generator_t &generator, auto const &backend) {
    ngrams_sketches_corpus_t corpus(backend.runtime, generator, ngrams_sketches_lengths_(8));
    std::vector<sz_size_t> const widths = {5, 8}, many(sz_ngrams_widths_max_k + 1, 3), zero = {3, 0};
    sz_ngrams_engine_t engine {};
    expect_equal(sz_ngrams_engine_init(&engine, nullptr, widths.data(), widths.size(), 0, backend.runtime.capabilities,
                                       STRINGZILLA_NULL, backend.runtime.stream),
                 sz_success_k);

    std::vector<sz_u8_t> host_sketches(corpus.sketches.size(), 0x55);
    if (backend.ngrams_sketches(&engine, &corpus.device, corpus.slots, host_sketches.data(), corpus.stride,
                                backend.runtime.stream) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host sketches were not refused");
    // An accessor is host code, so it is refused even over bytes the device reaches.
    std::vector<sz_string_view_t> const resident_views = views_of_(corpus.pinned);
    sz_strings_view_t const resident_accessor = strings_from_(resident_views);
    if (backend.ngrams_sketches(&engine, &resident_accessor, corpus.slots, corpus.sketches.data(), corpus.stride,
                                backend.runtime.stream) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "an accessor over the texts was not refused");
    if (backend.ngrams_sketches(&engine, &corpus.device, 0, corpus.sketches.data(), corpus.stride,
                                backend.runtime.stream) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a sketch of no slots was not refused");
    if (backend.ngrams_sketches(&engine, &corpus.device, corpus.slots, corpus.sketches.data(), corpus.slots - 1,
                                backend.runtime.stream) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a stride under the slots was not refused");
    // No device holds a gibibyte of ranks per text on chip.
    sz_size_t const vast = (sz_size_t)1 << 28;
    if (backend.ngrams_sketches(&engine, &corpus.device, vast, corpus.sketches.data(), vast, backend.runtime.stream) !=
        sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "ranks past the device's per-text memory were not refused");
    sz_ngrams_engine_free(&engine, backend.runtime.stream);
    for (std::vector<sz_size_t> const *refused : {&many, &zero}) {
        sz_ngrams_engine_t narrow {};
        if (sz_ngrams_engine_init(&narrow, nullptr, refused->data(), refused->size(), 0, backend.runtime.capabilities,
                                  STRINGZILLA_NULL, backend.runtime.stream) != sz_unexpected_dimensions_k)
            fail_backend_(backend.name, "widths past the bound or of zero were not refused");
    }
    join_(backend.runtime, backend.runtime.stream);
    for (sz_u8_t const untouched : corpus.sketches) expect_equal(untouched, 0x55);
    for (sz_u8_t const untouched : host_sketches)
        if (untouched != 0x55) fail_backend_(backend.name, "a refused call still wrote a sketch");
}

#pragma endregion Ngrams Sketches Checks

#pragma region Substrings Helpers

/** Haystacks in a pinned tape, which the serial oracle walks, and their device copy, which every
 *  device verb reads. */
struct device_corpus_t {

    /** The haystacks, which the serial oracle walks. */
    pinned_strings_t pinned;

    /** The haystacks gathered on the device, which every device verb reads. */
    sz_strings_view_t device {};

    device_corpus_t(device_backend_t const &runtime, std::vector<std::string> const &haystacks)
        : pinned(staged_(runtime, haystacks)), device(uploaded_(runtime, pinned.view().raw)) {}
};

/**
 *  @brief One vocabulary compiled twice under one policy: for the host and for @c stream 's device.
 *
 *  The policy sizes the arena, so it belongs to the engine rather than to a call, and comparing two
 *  tiers under one policy means holding two engines rather than one object read two ways.
 */
struct substrings_device_engines_t {

    /** What the serial oracle's engine is built from. */
    handle_checked_heap_t heap;

    /** The serial oracle's engine, in plain host memory. */
    sz_substrings_engine_t host {};

    /** The device's engine, arena and report included, in pinned memory, as the host builds the
     *  vocabulary in place and reads the report after a join. */
    sz_substrings_engine_t device {};
    sz_stream_t stream;

    substrings_device_engines_t(pinned_strings_t const &vocabulary, sz_case_sensitivity_t sensitivity,
                                sz_substrings_overlap_policy_t policy, auto const &backend)
        : stream(backend.runtime.stream) {
        sz_strings_view_t const needles = vocabulary.view().raw;
        sz_allocator_t pinned = backend.runtime.pinned;
        expect_equal(
            sz_substrings_engine_init_serial(&host, &needles, sensitivity, policy,
                                             STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0, &heap.allocator, nullptr),
            sz_success_k);
        expect_equal(
            sz_substrings_engine_init(&device, &needles, sensitivity, policy, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0,
                                      0, backend.runtime.capabilities, &pinned, backend.runtime.stream),
            sz_success_k);
    }
    substrings_device_engines_t(substrings_device_engines_t const &) = delete;
    substrings_device_engines_t &operator=(substrings_device_engines_t const &) = delete;
    ~substrings_device_engines_t() noexcept { reset(); }
    void reset() noexcept {
        sz_substrings_engine_free(&device, stream);
        sz_substrings_engine_free(&host, nullptr);
    }
};

/** One match, ordered so two backends' reports compare as sequences rather than as multisets. */
struct substrings_device_case_t {

    /** Which haystack of the collection this match was found in. */
    sz_size_t haystack_index {};

    /** Which needle of the vocabulary matched. */
    sz_size_t needle_index {};

    /** Where the match starts inside that haystack. */
    sz_size_t byte_offset {};

    /** Haystack bytes the match spans. */
    sz_size_t byte_length {};

    bool operator==(substrings_device_case_t const &other) const noexcept {
        return haystack_index == other.haystack_index && needle_index == other.needle_index &&
               byte_offset == other.byte_offset && byte_length == other.byte_length;
    }
    bool operator<(substrings_device_case_t const &other) const noexcept {
        if (haystack_index != other.haystack_index) return haystack_index < other.haystack_index;
        if (byte_offset != other.byte_offset) return byte_offset < other.byte_offset;
        if (byte_length != other.byte_length) return byte_length < other.byte_length;
        return needle_index < other.needle_index;
    }
};

/** Sorts one verb's match array into the shape two backends compare position by position. */
inline std::vector<substrings_device_case_t> substrings_device_matches_sorted_(sz_match_t const *matches,
                                                                               sz_size_t count) {
    std::vector<substrings_device_case_t> reported(count);
    for (std::size_t index = 0; index != count; ++index)
        reported[index] = {matches[index].haystack_index, matches[index].needle_index, matches[index].byte_offset,
                           matches[index].byte_length};
    std::sort(reported.begin(), reported.end());
    return reported;
}

/** Every match the serial tier reports, sized from the report its own sizing call leaves. */
inline std::vector<substrings_device_case_t> substrings_device_matches_serial_(sz_substrings_engine_t *engine,
                                                                               sz_strings_view_t const *haystacks) {
    std::vector<sz_size_t> offsets(haystacks->count + 1, 0);
    expect_equal(sz_substrings_find_serial(engine, haystacks, nullptr, 0, offsets.data(), nullptr), sz_success_k);
    std::vector<sz_match_t> matches(engine->report->matches_emitted);
    expect_equal(sz_substrings_find_serial(engine, haystacks, matches.data(), matches.size(), offsets.data(), nullptr),
                 sz_success_k);
    expect_equal(engine->report->shortfall, 0);
    return substrings_device_matches_sorted_(matches.data(), matches.size());
}

/** Every match the device tier reports, joined per call since no verb joins for the caller. */
inline std::vector<substrings_device_case_t> device_matches_(auto const &backend, sz_substrings_engine_t *engine,
                                                             sz_strings_view_t const *haystacks) {
    pinned_vector<sz_size_t> offsets(haystacks->count + 1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
    expect_equal(backend.substrings_find(engine, haystacks, nullptr, 0, offsets.data(), backend.runtime.stream),
                 sz_success_k);
    join_(backend.runtime, backend.runtime.stream);
    // A cover thins the matches after the sizing walk, so the boundaries name the survivors and the
    // report's emitted count names what the walk found before them.
    sz_size_t const total = offsets[haystacks->count];
    expect_equal(engine->report->matches_stored + engine->report->shortfall, total);

    pinned_vector<sz_match_t> matches(total, pinned_alloc_<sz_match_t>(backend.runtime));
    expect_equal(backend.substrings_find(engine, haystacks, matches.data(), matches.size(), offsets.data(),
                                         backend.runtime.stream),
                 sz_success_k);
    join_(backend.runtime, backend.runtime.stream);
    expect_true(engine->report->matches_stored == total && engine->report->shortfall == 0);
    return substrings_device_matches_sorted_(matches.data(), total);
}

/** The device's matches, counts and rewrite against serial's, over one corpus and policy. */
inline void check_against_serial_(auto const &backend, device_corpus_t &corpus, pinned_strings_t const &vocabulary,
                                  std::vector<std::string> const &haystacks, sz_case_sensitivity_t sensitivity,
                                  sz_substrings_overlap_policy_t policy) {
    substrings_device_engines_t engines(vocabulary, sensitivity, policy, backend);
    sz_strings_view_t const pinned_tape = corpus.pinned.view().raw, device_tape = corpus.device;
    std::vector<substrings_device_case_t> const expected = substrings_device_matches_serial_(&engines.host,
                                                                                             &pinned_tape);
    std::vector<substrings_device_case_t> const reported = device_matches_(backend, &engines.device, &device_tape);
    sz_size_t const device_total = reported.size();
    if (expect_equal(reported.size(), expected.size()))
        for (std::size_t index = 0; index != expected.size(); ++index) expect_equal(reported[index], expected[index]);

    // The counts come from the boundaries rather than from the match list, so they can
    // disagree with it.
    pinned_vector<sz_size_t> device_counts(haystacks.size(), 0, pinned_alloc_<sz_size_t>(backend.runtime));
    std::vector<sz_size_t> serial_counts(haystacks.size(), 0);
    expect_equal(
        sz_substrings_counts_serial(&engines.host, &pinned_tape, serial_counts.data(), sizeof(sz_size_t), nullptr),
        sz_success_k);
    expect_equal(backend.substrings_counts(&engines.device, &device_tape, device_counts.data(), sizeof(sz_size_t),
                                           backend.runtime.stream),
                 sz_success_k);
    join_(backend.runtime, backend.runtime.stream);
    for (std::size_t index = 0; index != haystacks.size(); ++index)
        expect_equal(device_counts[index], serial_counts[index]);

    // The dispatched verb picks the same kernel by the engine's recorded capability, so both agree.
    pinned_vector<sz_size_t> dispatched_counts(haystacks.size(), 0, pinned_alloc_<sz_size_t>(backend.runtime));
    expect_equal(sz_substrings_counts(&engines.device, &device_tape, dispatched_counts.data(), sizeof(sz_size_t),
                                      backend.runtime.stream),
                 sz_success_k);
    join_(backend.runtime, backend.runtime.stream);
    for (std::size_t index = 0; index != haystacks.size(); ++index)
        expect_equal(dispatched_counts[index], device_counts[index]);
    {
        sz_size_t counted = 0;
        for (sz_size_t const count : device_counts) counted += count;
        expect_equal(counted, device_total, "The counts and the matches come from one pass");
    }

    // Contains stops each chunk at its first match, under every policy.
    if (backend.substrings_contains) {
        std::vector<sz_u1x8_t> serial_contained(sz_size_divide_round_up(haystacks.size(), 8), 0);
        expect_equal(sz_substrings_contains_serial(&engines.host, &pinned_tape, serial_contained.data(), nullptr),
                     sz_success_k);
        check_contains_(backend.runtime, serial_contained, [&](sz_u1x8_t *contained) {
            return backend.substrings_contains(&engines.device, &device_tape, contained, backend.runtime.stream);
        });
        check_contains_(backend.runtime, serial_contained, [&](sz_u1x8_t *contained) {
            return sz_substrings_contains(&engines.device, &device_tape, contained, backend.runtime.stream);
        });
    }

    if (policy == sz_substrings_overlapping_k) return;

    // The rewrite, whose device path owns a block scan and a tiled copy the serial one has
    // no analogue for.
    std::vector<std::string> replacements;
    for (std::size_t index = 0; index != vocabulary.size(); ++index)
        replacements.push_back(index % 3 == 0 ? std::string() : "<" + std::to_string(index) + ">");
    device_corpus_t replacement_corpus(backend.runtime, replacements);

    pinned_vector<sz_size_t> device_offsets(haystacks.size() + 1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
    std::vector<sz_size_t> serial_offsets(haystacks.size() + 1, 0);
    sz_strings_view_t const pinned_replacements = replacement_corpus.pinned.view().raw,
                            device_replacements = replacement_corpus.device;
    expect_equal(backend.substrings_replace(&engines.device, &device_tape, &device_replacements, nullptr, 0,
                                            device_offsets.data(), backend.runtime.stream),
                 sz_success_k);
    join_(backend.runtime, backend.runtime.stream);
    expect_equal(sz_substrings_replace_serial(&engines.host, &pinned_tape, &pinned_replacements, nullptr, 0,
                                              serial_offsets.data(), nullptr),
                 sz_success_k);
    for (std::size_t index = 0; index != haystacks.size() + 1; ++index)
        expect_equal(device_offsets[index], serial_offsets[index]);

    sz_size_t const rewritten = serial_offsets[haystacks.size()];
    pinned_vector<char> device_target(rewritten, pinned_alloc_<char>(backend.runtime));
    std::vector<char> serial_target(rewritten);
    expect_equal(backend.substrings_replace(&engines.device, &device_tape, &device_replacements,
                                            device_target.empty() ? nullptr : device_target.data(), rewritten,
                                            device_offsets.data(), backend.runtime.stream),
                 sz_success_k);
    join_(backend.runtime, backend.runtime.stream);
    expect_equal(engines.device.report->shortfall, 0);
    expect_equal(sz_substrings_replace_serial(&engines.host, &pinned_tape, &pinned_replacements,
                                              serial_target.empty() ? nullptr : serial_target.data(), rewritten,
                                              serial_offsets.data(), nullptr),
                 sz_success_k);
    for (std::size_t index = 0; index != rewritten; ++index) expect_equal(device_target[index], serial_target[index]);
    engines.reset();
    expect_equal(engines.heap.live_allocations, 0);
}

/**
 *  @brief The device's BM25 against the serial tier's, by byte lengths and by caller-given ones.
 *
 *  The device sums in fixed point and the host in ascending needle order, so the two agree to
 *  rounding rather than bit for bit.
 */
inline void check_bm25_against_serial_(auto const &backend, device_corpus_t &corpus, pinned_strings_t const &vocabulary,
                                       std::vector<std::string> const &haystacks, sz_case_sensitivity_t sensitivity) {
    substrings_device_engines_t engines(vocabulary, sensitivity, sz_substrings_overlapping_k, backend);
    std::size_t const needles_count = engines.host.needles_count;
    pinned_vector<sz_f32_t> weights(needles_count, pinned_alloc_<sz_f32_t>(backend.runtime)),
        given_lengths(haystacks.size(), pinned_alloc_<sz_f32_t>(backend.runtime));
    for (std::size_t index = 0; index != needles_count; ++index) weights[index] = 0.5f + (float)(index % 7) * 0.25f;
    double bytes_total = 0, given_total = 0;
    for (std::size_t index = 0; index != haystacks.size(); ++index) {
        given_lengths[index] = (sz_f32_t)(1 + index % 5);
        bytes_total += haystacks[index].size(), given_total += given_lengths[index];
    }

    for (sz_f32_t const *lengths : {(sz_f32_t const *)nullptr, (sz_f32_t const *)given_lengths.data()}) {
        double const average = (lengths ? given_total : bytes_total) / haystacks.size();
        sz_substrings_bm25_t const parameters {1.2f, 0.75f, (sz_f32_t)(average > 0 ? average : 1)};
        std::vector<sz_f32_t> serial_scores(haystacks.size(), -1);
        pinned_vector<sz_f32_t> device_scores(haystacks.size(), -1, pinned_alloc_<sz_f32_t>(backend.runtime));
        sz_strings_view_t const pinned_tape = corpus.pinned.view().raw, device_tape = corpus.device;
        expect_equal(sz_substrings_bm25_scores_serial(&engines.host, &pinned_tape, lengths, &parameters, weights.data(),
                                                      serial_scores.data(), sizeof(sz_f32_t), nullptr),
                     sz_success_k);
        expect_equal(backend.substrings_bm25_scores(&engines.device, &device_tape, lengths, &parameters, weights.data(),
                                                    device_scores.data(), sizeof(sz_f32_t), backend.runtime.stream),
                     sz_success_k);
        join_(backend.runtime, backend.runtime.stream);
        for (std::size_t index = 0; index != haystacks.size(); ++index) {
            double const tolerance = 1e-5 * max_of(1.0, std::fabs((double)serial_scores[index]));
            expect_near(device_scores[index], serial_scores[index], tolerance);
        }
    }
    engines.reset();
    expect_equal(engines.heap.live_allocations, 0);
}

/** One vocabulary against one corpus under every policy. */
inline void check_policies_(auto const &backend, std::vector<std::string> const &haystacks,
                            std::vector<std::string> const &needles, sz_case_sensitivity_t sensitivity) {
    device_corpus_t corpus(backend.runtime, haystacks);
    pinned_strings_t const vocabulary = staged_(backend.runtime, needles);
    check_against_serial_(backend, corpus, vocabulary, haystacks, sensitivity, sz_substrings_overlapping_k);
    check_against_serial_(backend, corpus, vocabulary, haystacks, sensitivity, sz_substrings_leftmost_longest_k);
    check_against_serial_(backend, corpus, vocabulary, haystacks, sensitivity, sz_substrings_leftmost_first_k);
    check_bm25_against_serial_(backend, corpus, vocabulary, haystacks, sensitivity);
}

#pragma endregion Substrings Helpers

#pragma region Substrings Checks

/** The chunk-boundary cases a single-chain host walk cannot express. */
inline void check_substrings_device_unit_(auto const &backend) {
    // The device walk is chunked and the host walk is not, so a match spanning a chunk boundary is
    // the one thing this can get wrong that the CPU suite cannot see.
    check_policies_(backend, {"ushers"}, {"he", "she", "his", "hers"}, sz_cased_k);
    check_policies_(backend, {"Straße", "STRASSE"}, {"strasse", "sse", "s"}, sz_uncased_k);

    // The Kelvin sign folds to one ASCII byte, contracting three source bytes into one folded one.
    check_policies_(backend, {"\xE2\x84\xAA elvin", "kelvin"}, {"k", "kelvin"}, sz_uncased_k);

    // Malformed UTF-8 in the haystack, which an uncased walk resynchronizes past byte by byte.
    check_policies_(backend,
                    {std::string("ab\xFF" "cd", 5), // "\xFFc" is one escape
                     "abcd"},
                    {"ab", "cd"}, sz_uncased_k);

    // 45 haystacks run past one ballot and end mid-byte, so the bitmap's tail bits count too.
    std::vector<std::string> batch(45, "us");
    for (std::size_t index = 1; index < batch.size(); index += 3) batch[index] = "ushers";
    check_policies_(backend, batch, {"he", "she", "his", "hers"}, sz_cased_k);
}

/** Random corpora wide enough that the planner cuts several chunks per haystack. */
inline void check_substrings_device_equivalence_(split_mix_generator_t &generator, auto const &backend) {
    std::string folded_edges;
    std::string const folded_tile = std::string(63, 'x') + "ßſKﬃStraße .";
    for (unsigned repeat = 0; repeat != 128; ++repeat) folded_edges += folded_tile;
    check_policies_(backend, {"sßſKﬃStraße", std::string("s\xc0\x80SS", 5), folded_edges},
                    {"s", "ss", "k", "ffi", "strasse"}, sz_uncased_k);

    char const *const alphabets[] = {"ab", "abcdefgh", "abcdefghijklmnopqrstuvwxyz"};
    std::size_t const rounds = scaled_iterations(8);

    for (std::size_t round = 0; round != rounds; ++round) {
        char const *const alphabet = alphabets[round % 3];
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 1 + (round * 7) % 24; ++index)
            needles.push_back(random_string(generator, 1 + (index * 3 + round) % 7, alphabet));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        // Long enough that the planner cuts several chunks out of one haystack, which is the
        // regime the warm-up exists for; a corpus of short haystacks never crosses a chunk
        // boundary at all.
        for (std::size_t index = 0; index != 1 + (round * 5) % 9; ++index)
            haystacks.push_back(random_string(generator, 4096 + index * 977, alphabet));
        std::string mixed = haystacks.back();
        mixed.insert(mixed.size() / 3, "Straße ÄÖÜ \xE2\x84\xAA");
        haystacks.push_back(mixed);
        std::string uppercased = haystacks.front();
        for (char &character : uppercased) character = (char)std::toupper((unsigned char)character);
        haystacks.push_back(uppercased);
        end_off_word_(haystacks);
        check_policies_(backend, haystacks, needles, sz_cased_k);
        check_policies_(backend, haystacks, needles, sz_uncased_k);
    }

    // A vocabulary wider than a block's tally, so needles hash into shared slots and
    // spill past them.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 6000; ++index)
            needles.push_back(random_string(generator, 2 + index % 4, "abcdefghijklmnopqrstuvwxyz"));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 64; ++index)
            haystacks.push_back(random_string(generator, 256 + index * 97, "abcdefghijklmnopqrstuvwxyz"));
        // Short documents past a residency wave, so the grid holds many blocks, each with its
        // own overflow row.
        for (std::size_t index = 0; index != 4096; ++index)
            haystacks.push_back(random_string(generator, 16 + index % 97, "abcdefghijklmnopqrstuvwxyz"));
        // A haystack spelling every needle seats more of them than a tally holds, twice in
        // one threadgroup's run.
        std::string every_needle;
        for (std::string const &needle : needles) every_needle += needle + "#";
        haystacks[0] = every_needle;
        haystacks.insert(haystacks.begin() + 64, every_needle);
        device_corpus_t corpus(backend.runtime, haystacks);
        pinned_strings_t const vocabulary = staged_(backend.runtime, needles);
        expect_true(vocabulary.size() > 4096);
        check_bm25_against_serial_(backend, corpus, vocabulary, haystacks, sz_cased_k);
    }

    // Documents long enough for a cluster of blocks to walk each of them together, among short ones
    // a block scores alone, under a tally of a slot per needle and under a hashed one.
    {
        std::vector<std::string> haystacks;
        for (std::size_t index = 0; index != 20; ++index)
            haystacks.push_back(random_string(generator, index % 7 == 3 ? (std::size_t)3 << 20 : 512 + index * 31,
                                              "abcdefghijklmnopqrstuvwxyz"));
        device_corpus_t corpus(backend.runtime, haystacks);
        for (std::size_t const needles_count : {(std::size_t)300, (std::size_t)6000}) {
            std::vector<std::string> needles;
            for (std::size_t index = 0; index != needles_count; ++index)
                needles.push_back(random_string(generator, 2 + index % 4, "abcdefghijklmnopqrstuvwxyz"));
            std::sort(needles.begin(), needles.end());
            needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
            pinned_strings_t const vocabulary = staged_(backend.runtime, needles);
            check_bm25_against_serial_(backend, corpus, vocabulary, haystacks, sz_cased_k);
        }
    }

    // Nucleotides, where the whole hot tier is five columns wide and fits a block's shared memory.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 300; ++index)
            needles.push_back(random_string(generator, 4 + index % 13, "ACGT"));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 6; ++index) {
            std::string haystack = random_string(generator, 20000 + index * 4099, "ACGT");
            for (std::size_t offset = 61; offset < haystack.size(); offset += 997) haystack[offset] = 'N';
            haystacks.push_back(haystack);
        }
        check_policies_(backend, haystacks, needles, sz_cased_k);
    }

    // One haystack far wider than a chunk, so the boundary case is hit many times over in
    // a single walk.
    {
        std::vector<std::string> const needles {"the", "there", "here", "her", "he"};
        std::vector<std::string> haystacks {random_string(generator, 1 << 20, "the ")};
        // The rewrite spans many tiles.
        check_policies_(backend, haystacks, needles, sz_cased_k);
    }
}

/** What the device verbs refuse, and what the report says when an output cannot hold the answer. */
inline void check_substrings_device_safety_(auto const &backend) {
    std::vector<std::string> const needles {"ab", "cd"};
    std::vector<std::string> const haystacks {"abcdabcd"};
    device_corpus_t corpus(backend.runtime, haystacks);
    pinned_strings_t const vocabulary = staged_(backend.runtime, needles);
    substrings_device_engines_t engines(vocabulary, sz_cased_k, sz_substrings_overlapping_k, backend);

    // A device verb refuses host memory rather than reading it from a kernel, whatever
    // else is resident.
    {
        std::vector<sz_size_t> host_counts(haystacks.size(), 0);
        sz_strings_view_t const device_tape = corpus.device;
        expect_equal(backend.substrings_counts(&engines.device, &device_tape, host_counts.data(), sizeof(sz_size_t),
                                               backend.runtime.stream),
                     sz_device_memory_mismatch_k);
    }

    // A host bitmap is refused too, and an empty batch writes none.
    if (backend.substrings_contains) {
        std::vector<sz_u1x8_t> host_contained(1, 0);
        sz_strings_view_t const device_tape = corpus.device;
        sz_strings_view_t const empty {};
        expect_equal(
            backend.substrings_contains(&engines.device, &device_tape, host_contained.data(), backend.runtime.stream),
            sz_device_memory_mismatch_k);
        expect_equal(backend.substrings_contains(&engines.device, &empty, nullptr, backend.runtime.stream),
                     sz_success_k);
    }

    // An output stride of zero cannot address one entry per haystack, whatever the haystack count.
    {
        pinned_vector<sz_size_t> counts(haystacks.size(), STRINGZILLA_SIZE_MAX,
                                        pinned_alloc_<sz_size_t>(backend.runtime));
        sz_strings_view_t const device_tape = corpus.device;
        expect_equal(backend.substrings_counts(&engines.device, &device_tape, counts.data(), 0, backend.runtime.stream),
                     sz_unexpected_dimensions_k);
        expect_equal(counts[0], STRINGZILLA_SIZE_MAX);
    }

    // The counts a device walk answers with, which the same corpus answers on the host.
    {
        pinned_vector<sz_size_t> counts(haystacks.size(), 0, pinned_alloc_<sz_size_t>(backend.runtime));
        sz_strings_view_t const device_tape = corpus.device;
        expect_equal(backend.substrings_counts(&engines.device, &device_tape, counts.data(), sizeof(sz_size_t),
                                               backend.runtime.stream),
                     sz_success_k);
        join_(backend.runtime, backend.runtime.stream);
        expect_equal(counts[0], 4);
    }

    // A capacity that cannot hold the matches is not an error: the report names the true total and
    // the shortfall beside it, and the output is left untouched rather than truncated.
    {
        pinned_vector<sz_size_t> offsets(haystacks.size() + 1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
        sz_strings_view_t const device_tape = corpus.device;
        expect_equal(
            backend.substrings_find(&engines.device, &device_tape, nullptr, 0, offsets.data(), backend.runtime.stream),
            sz_success_k);
        join_(backend.runtime, backend.runtime.stream);
        expect_equal(engines.device.report->matches_emitted, 4);
        expect_true(engines.device.report->matches_stored == 0 && engines.device.report->shortfall == 4);
    }

    // A substitution over matches that share bytes is not a function, so the overlapping policy is
    // refused before anything is launched.
    {
        std::vector<std::string> const replacements {"x", "y"};
        device_corpus_t replacement_corpus(backend.runtime, replacements);
        pinned_vector<sz_size_t> offsets(haystacks.size() + 1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
        sz_strings_view_t const device_tape = corpus.device;
        sz_strings_view_t const device_replacements = replacement_corpus.device;
        expect_equal(backend.substrings_replace(&engines.device, &device_tape, &device_replacements, nullptr, 0,
                                                offsets.data(), backend.runtime.stream),
                     sz_status_unknown_k);
    }

    // A vocabulary a round cannot emit within its budget leaves every later kernel retired, so the
    // outputs keep whatever they held and the report alone says the round did not fit.
    {
        sz_strings_view_t const needles = vocabulary.view().raw;
        sz_allocator_t pinned = backend.runtime.pinned;
        sz_substrings_engine_t tiny {};
        pinned_vector<sz_size_t> offsets(haystacks.size() + 1, STRINGZILLA_SIZE_MAX,
                                         pinned_alloc_<sz_size_t>(backend.runtime));
        expect_equal(sz_substrings_engine_init(&tiny, &needles, sz_cased_k, sz_substrings_leftmost_first_k,
                                               STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 1, 0,
                                               backend.runtime.capabilities, &pinned, backend.runtime.stream),
                     sz_success_k);
        sz_strings_view_t const device_tape = corpus.device;
        expect_equal(backend.substrings_find(&tiny, &device_tape, nullptr, 0, offsets.data(), backend.runtime.stream),
                     sz_success_k);
        join_(backend.runtime, backend.runtime.stream);
        expect_true(tiny.report->matches_emitted == 4 && tiny.report->shortfall == 3);
        expect_equal(offsets[haystacks.size()], 0, "A retired offsets kernel leaves the boundaries zeroed");
        sz_substrings_engine_free(&tiny, backend.runtime.stream);
    }

    // The arena is sized for a haystacks budget when the engine is built, so a round carrying more
    // is refused rather than grown under a stream that may still be reading it.
    {
        device_corpus_t pair(backend.runtime, std::vector<std::string> {"abcd", "abcd"});
        sz_strings_view_t const needles = vocabulary.view().raw;
        sz_allocator_t pinned = backend.runtime.pinned;
        sz_substrings_engine_t narrow {};
        pinned_vector<sz_size_t> offsets(3, 0, pinned_alloc_<sz_size_t>(backend.runtime));
        expect_equal(sz_substrings_engine_init(&narrow, &needles, sz_cased_k, sz_substrings_overlapping_k,
                                               STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 1,
                                               backend.runtime.capabilities, &pinned, backend.runtime.stream),
                     sz_success_k);
        sz_strings_view_t const pair_tape = pair.device;
        expect_equal(backend.substrings_find(&narrow, &pair_tape, nullptr, 0, offsets.data(), backend.runtime.stream),
                     sz_unexpected_dimensions_k);
        sz_substrings_engine_free(&narrow, backend.runtime.stream);
    }
    engines.reset();
    expect_equal(engines.heap.live_allocations, 0);
}

#pragma endregion Substrings Checks

#pragma region Regex Checks

/** One vocabulary of patterns compiled twice under one case sensitivity:
 *  for the host and for @c stream 's device. */
struct regex_device_engines_t {

    /** What the serial oracle's engine is built from. */
    handle_checked_heap_t heap;

    /** The serial oracle's engine, in plain host memory. */
    sz_regex_engine_t host {};

    /** The device's engine, in pinned memory, as the host builds it in place. */
    sz_regex_engine_t device {};
    sz_stream_t stream;

    regex_device_engines_t(pinned_strings_t const &vocabulary, auto const &backend, sz_case_sensitivity_t sensitivity)
        : stream(backend.runtime.stream) {
        sz_strings_view_t const patterns = vocabulary.view().raw;
        sz_allocator_t pinned = backend.runtime.pinned;
        expect_equal(sz_regex_engine_init_serial(&host, &patterns, sensitivity, nullptr, &heap.allocator, nullptr),
                     sz_success_k);
        expect_equal(sz_regex_engine_init(&device, &patterns, sensitivity, nullptr, backend.runtime.capabilities,
                                          &pinned, backend.runtime.stream),
                     sz_success_k);
    }
    regex_device_engines_t(regex_device_engines_t const &) = delete;
    regex_device_engines_t &operator=(regex_device_engines_t const &) = delete;
    ~regex_device_engines_t() noexcept { reset(); }
    void reset() noexcept {
        sz_regex_engine_free(&device, stream);
        sz_regex_engine_free(&host, nullptr);
    }
};

/** Whether two located matches, or two rows of groups, are one. */
inline bool regex_same_match_(sz_match_t const &device, sz_match_t const &serial) {
    return device.haystack_index == serial.haystack_index && device.needle_index == serial.needle_index &&
           device.byte_offset == serial.byte_offset && device.byte_length == serial.byte_length;
}

/** The device's boundaries, matches, counts, rewrites and rows of groups against serial's, in
 *  order, under capacities holding them all and fewer, which leave the rest unwritten. */
inline void check_regex_against_serial_(auto const &backend, std::vector<std::string> const &haystacks,
                                        std::vector<std::string> const &patterns,
                                        sz_case_sensitivity_t sensitivity = sz_cased_k) {
    device_corpus_t corpus(backend.runtime, haystacks);
    pinned_strings_t const vocabulary = staged_(backend.runtime, patterns);
    regex_device_engines_t engines(vocabulary, backend, sensitivity);
    sz_strings_view_t const pinned_tape = corpus.pinned.view().raw, device_tape = corpus.device;
    sz_stream_t const stream = backend.runtime.stream;
    std::size_t const count = haystacks.size();

    std::vector<sz_size_t> serial_offsets(count + 1, 0);
    expect_equal(sz_regex_find_serial(&engines.host, &pinned_tape, nullptr, 0, serial_offsets.data(), nullptr),
                 sz_success_k);
    std::size_t const total = serial_offsets[count];
    std::vector<sz_match_t> serial_matches(total);
    expect_equal(
        sz_regex_find_serial(&engines.host, &pinned_tape, serial_matches.data(), total, serial_offsets.data(), nullptr),
        sz_success_k);

    sz_match_t const untouched {STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX};
    for (std::size_t const capacity : {total, total / 2}) {
        pinned_vector<sz_size_t> offsets(count + 1, STRINGZILLA_SIZE_MAX, pinned_alloc_<sz_size_t>(backend.runtime));
        pinned_vector<sz_match_t> matches(total + 1, untouched, pinned_alloc_<sz_match_t>(backend.runtime));
        expect_equal(
            backend.regex_find(&engines.device, &device_tape, matches.data(), capacity, offsets.data(), stream),
            sz_success_k);
        join_(backend.runtime, stream);
        for (std::size_t index = 0; index <= count; ++index) expect_equal(offsets[index], serial_offsets[index]);
        for (std::size_t index = 0; index != capacity; ++index)
            expect_true(regex_same_match_(matches[index], serial_matches[index]));
        for (std::size_t index = capacity; index <= total; ++index)
            expect_equal(matches[index].haystack_index, STRINGZILLA_SIZE_MAX);
    }

    // The dispatched verb runs the kernel of the capability the engine recorded, so both agree.
    pinned_vector<sz_size_t> device_counts(count, 0, pinned_alloc_<sz_size_t>(backend.runtime)),
        dispatched_counts(count, 0, pinned_alloc_<sz_size_t>(backend.runtime));
    expect_equal(backend.regex_counts(&engines.device, &device_tape, device_counts.data(), sizeof(sz_size_t), stream),
                 sz_success_k);
    expect_equal(sz_regex_counts(&engines.device, &device_tape, dispatched_counts.data(), sizeof(sz_size_t), stream),
                 sz_success_k);
    join_(backend.runtime, stream);
    for (std::size_t index = 0; index != count; ++index)
        expect_true(device_counts[index] == serial_offsets[index + 1] - serial_offsets[index] &&
                    dispatched_counts[index] == device_counts[index]);
    if (backend.regex_contains) {
        std::vector<sz_u1x8_t> serial_contained(sz_size_divide_round_up(count, 8), 0);
        expect_equal(sz_regex_contains_serial(&engines.host, &pinned_tape, serial_contained.data(), nullptr),
                     sz_success_k);
        check_contains_(backend.runtime, serial_contained, [&](sz_u1x8_t *contained) {
            return backend.regex_contains(&engines.device, &device_tape, contained, stream);
        });
        check_contains_(backend.runtime, serial_contained, [&](sz_u1x8_t *contained) {
            return sz_regex_contains(&engines.device, &device_tape, contained, stream);
        });
    }

    // Every match rewritten, the groups a replacement names read from its own span: sized first
    // with no target, as a caller sizes one, then under targets holding all of it and a third.
    std::vector<std::string> templates;
    for (std::size_t index = 0; index != patterns.size(); ++index)
        templates.push_back(index % 3 == 0 ? "<$0|$1>" : index % 3 == 1 ? "${2}~$$" : "${rest}$1x");
    device_corpus_t replacements(backend.runtime, templates);
    sz_strings_view_t const pinned_replacements = replacements.pinned.view().raw,
                            device_replacements = replacements.device;
    std::vector<sz_size_t> serial_rewritten(count + 1, 0);
    expect_equal(sz_regex_replace_serial(&engines.host, &pinned_tape, &pinned_replacements, nullptr, 0,
                                         serial_rewritten.data(), nullptr),
                 sz_success_k);
    std::size_t const bytes = serial_rewritten[count];
    std::string serial_target(bytes, '\0');
    expect_equal(sz_regex_replace_serial(&engines.host, &pinned_tape, &pinned_replacements, serial_target.data(), bytes,
                                         serial_rewritten.data(), nullptr),
                 sz_success_k);
    for (std::size_t const capacity : {std::size_t(0), bytes, bytes / 3}) {
        pinned_vector<sz_size_t> offsets(count + 1, STRINGZILLA_SIZE_MAX, pinned_alloc_<sz_size_t>(backend.runtime));
        pinned_vector<char> target(bytes + 1, '\x01', pinned_alloc_<char>(backend.runtime));
        expect_equal(backend.regex_replace(&engines.device, &device_tape, &device_replacements,
                                           capacity ? target.data() : nullptr, capacity, offsets.data(), stream),
                     sz_success_k);
        join_(backend.runtime, stream);
        for (std::size_t index = 0; index <= count; ++index) expect_equal(offsets[index], serial_rewritten[index]);
        expect_true(std::equal(target.begin(), target.begin() + capacity, serial_target.begin()));
        expect_true(std::all_of(target.begin() + capacity, target.end(), [](char byte) { return byte == '\x01'; }));
    }

    // Every pattern's groups in each haystack's first match of it,
    // under capacities holding all rows and half.
    for (std::size_t pattern = 0; pattern != patterns.size(); ++pattern) {
        std::vector<sz_size_t> serial_rows_offsets(count + 1, 0);
        expect_equal(sz_regex_extract_serial(&engines.host, &pinned_tape, pattern, nullptr, 0,
                                             serial_rows_offsets.data(), nullptr),
                     sz_success_k);
        std::size_t const rows_count = serial_rows_offsets[count];
        std::vector<sz_match_t> serial_rows(rows_count);
        expect_equal(sz_regex_extract_serial(&engines.host, &pinned_tape, pattern, serial_rows.data(), rows_count,
                                             serial_rows_offsets.data(), nullptr),
                     sz_success_k);
        for (std::size_t const capacity : {rows_count, rows_count / 2}) {
            pinned_vector<sz_size_t> offsets(count + 1, STRINGZILLA_SIZE_MAX,
                                             pinned_alloc_<sz_size_t>(backend.runtime));
            pinned_vector<sz_match_t> rows(rows_count + 1, untouched, pinned_alloc_<sz_match_t>(backend.runtime));
            expect_equal(backend.regex_extract(&engines.device, &device_tape, pattern, rows.data(), capacity,
                                               offsets.data(), stream),
                         sz_success_k);
            join_(backend.runtime, stream);
            for (std::size_t index = 0; index <= count; ++index)
                expect_equal(offsets[index], serial_rows_offsets[index]);
            for (std::size_t index = 0; index != capacity; ++index)
                expect_true(regex_same_match_(rows[index], serial_rows[index]));
            for (std::size_t index = capacity; index <= rows_count; ++index)
                expect_equal(rows[index].haystack_index, STRINGZILLA_SIZE_MAX);
        }
    }
    engines.reset();
    expect_equal(engines.heap.live_allocations, 0);
}

/** Priority across patterns, empty matches, anchors and word boundaries, walks wider than a chunk,
 *  folded walks, the fallbacks of vocabularies past both state budgets, and lookarounds. */
inline void check_regex_device_unit_(auto const &backend) {
    check_regex_against_serial_(backend, {"", "ab", "Hello World 42 \"quoted\" x", "line\nnext\n\n"},
                                {"\\b[A-Z][a-z]+\\b", "[0-9]+", "\"[^\"]*\""});
    check_regex_against_serial_(backend, {"Hello World", "x Zeta9 99", ""},
                                {"(?<first>[A-Z])(?<rest>[a-z]+)", "([0-9])([0-9])?", "(?<rest>\\w)\\w"});

    // A folded walk snaps a start or an end inside one codepoint's folds outward, so `ß` meets "ss"
    // and the KELVIN SIGN meets "k", and rewrites and groups copy the source bytes.
    std::vector<std::string> const folded {"Stra\xC3\x9F" "e STRASSE strasse",   // "\x9Fe" is one escape
                                           "\xE2\x84\xAA" "elvin kelvin KELVIN", // "\xAAe" is one escape
                                           "\xC3\x9F\xC3\x9F ss SS", "", "\xD0\x96\xD0\xB6 \xC3\x89\xC3\xA9 \x80"};
    check_regex_against_serial_(backend, folded, {"stra(ss|\xC3\x9F)e", "\\bk([a-z]+)", "(s)+"}, sz_uncased_k);

    // Vocabularies past both state budgets walk position sets for starts and a PikeVM for ends.
    std::string const gaps = "ab cab a\nb ca " + std::string(300, 'c') + " a" + std::string(40, ' ') + "a";
    for (sz_case_sensitivity_t const sensitivity : {sz_cased_k, sz_uncased_k})
        check_regex_against_serial_(backend,
                                    {"ab0cdefghijklmnopq7", "a b c a\nab  ca",
                                     "Stra\xC3\x9F" "e x9", // "\x9Fe" is one escape
                                     gaps},
                                    {"[a-z].{16}[0-9]", "(?:[a-c ]{1,14}.{0,13}\\s)*(a)"}, sensitivity);

    // Empty matches between bytes, never inside a codepoint, nor where the previous match ended.
    check_regex_against_serial_(backend,
                                {"", "a", "baaab", "\xD0\xB6" "a\xD0\xB6", "\xFF\x80"}, // "\xB6a" is one escape
                                {"a*", "b|"});
    check_regex_against_serial_(backend, {"one\ntwo\n", "x", "  word  "}, {"(?m)^\\w+$", "\\b", "\\z"});

    // 45 haystacks run past one ballot and end mid-byte, so the bitmap's tail bits count too.
    std::vector<std::string> batch(45, "none");
    for (std::size_t index = 2; index < batch.size(); index += 3) batch[index] = "x" + std::to_string(index);
    check_regex_against_serial_(backend, batch, {"x([0-9]+)"});

    // A rewrite shrinking long matches over many scan tiles drifts below zero, which the scans
    // of the rewritten boundaries carry as exactly as they carry counts.
    std::string shrinking;
    for (std::size_t index = 0; index != 2000; ++index) shrinking += "government minister president police ";
    check_regex_against_serial_(backend, {shrinking, "", shrinking.substr(0, 999)}, {"x", "[a-z]{6,}"});

    // A quoted span much wider than a chunk leaves end walks reaching chunk edges, and a long line
    // keeps the start-finding walk from collapsing within one.
    std::string const quoted = "\"" + std::string(9000, 'x') + "\" tail \"" + std::string(70, 'y') + "\"";
    std::string const line = std::string(5000, 'a') + " error " + std::string(5000, 'b') + "\n" + std::string(300, 'c');
    check_regex_against_serial_(backend, {quoted, line}, {"\"[^\"]*\"", ".*error.*", "[a-c]{3}"});

    // Lookarounds read the table their passes fill: nested, negated under line anchors, over folds,
    // and with contexts open across many chunks, a lookbehind's after its quote and a lookahead's
    // before its own.
    check_regex_against_serial_(backend,
                                {"cost $42 and $7.50 or 3", "key: value\nother:x", "abc xbc bc  c", "price:$5", ""},
                                {"(?<=\\$)\\d+", "\\w+(?=:)", "(?<!x)bc", "\\s+(?!\\S)"});
    check_regex_against_serial_(backend, {"abc bbc bc xc", "ab bc", "bbbc"}, {"(?<=(?<!a)b)c", "(?=b(?!c))b"});
    check_regex_against_serial_(backend, {"one foo\ntwo\nthree food\n", "foo", ""}, {"(?m)^(?!.*foo).*$"});
    check_regex_against_serial_(backend,
                                {"Stra\xC3\x9F" "e STRASSE strasse", // "\x9Fe" is one escape
                                 "\xE2\x84\xAA" "elvin kelvin",      // "\xAAe" is one escape
                                 "\xC3\x9Fs SSs ss"},
                                {"(?<=stra)(ss|\xC3\x9F)e", "k(?=elvin)", "(?<=\xC3\x9F)s"}, sz_uncased_k);
    std::string const spans = "\"" + std::string(4000, 'x') + "q" + std::string(5000, 'y') + "\" q \"" +
                              std::string(70, 'q') + "\" " + std::string(3000, 'q');
    for (sz_case_sensitivity_t const sensitivity : {sz_cased_k, sz_uncased_k})
        check_regex_against_serial_(backend, {spans, line, quoted},
                                    {"(?<=\"[^\"]*)q+", "x(?=[^\"]*\")", "(?<=error.*)b{3}", "(?<!\"[^\"]*)y"},
                                    sensitivity);

    // Literal prefixes mark candidates without the start-finding walk: keywords every 61 bytes take
    // every offset into a chunk and cross tiles, and one its haystack's end cuts short is no match.
    std::string keywords(9000, '.');
    for (std::size_t offset = 1; offset + 6 < keywords.size(); offset += 61)
        keywords.replace(offset, 6, offset % 3 ? "quokka" : "zebras");
    check_regex_against_serial_(backend, {"quo", "kka", "quokk", "aquokkaa", keywords, "zebrazebras quokkquokka", ""},
                                {"quokka", "zebras?"});

    // One-byte lookarounds read the bytes beside each slot and the multi-byte bodies walk DFAs, at
    // haystack edges and at every offset into a chunk, five of them as columns of both DFAs.
    std::string seams;
    for (std::size_t index = 0; index != 400; ++index) seams += index % 7 ? "Ab " : "$12% CD Ef Gh abc";
    check_regex_against_serial_(
        backend, {"$5", "7%", "AB", "Ab Cd", "Ab Cd ", "Ab C", seams, "abc", "c", ""},
        {"(?<=\\$)[0-9]+", "[0-9]+(?=%)", "[A-Z][a-z]+(?= [A-Z][a-z]+)", "(?<![A-Za-z])[A-Z]{2,}(?![a-z])"});
    check_regex_against_serial_(backend, {"abc", "c", "bc", "ab", seams}, {"(?<=ab)c", "(?<!ab)c", "b(?=c\\s)"});
}

/** Random vocabularies over corpora of short haystacks among long ones, and a corpus past a segment
 *  of every vendor, whose carry hands the next segment where a straddling haystack resumes. */
inline void check_regex_device_equivalence_(split_mix_generator_t &generator, auto const &backend) {
    static char const *const pool[] = {
        "a+",    "ab|ba",  "[abc]{2,4}", "\\bab\\b", "a*",   "(?:ab)+c?",          "[^a ]+", "\"[^\"]*\"",
        "^.*a$", "(?m)^b", "a.{0,6}b",   "\\d+",     "c\\s", "\xD0\xB6+|\xC3\xA9", ".",      "(a|ab)(c|bcd)"};
    std::vector<std::vector<std::string>> const alphabets {
        {"a", "b"},
        {"a", "b", "c", " ", "\n", "\""},                            // " " is one literal
        {"a", "b", "0", " ", "\xD0\xB6", "\xC3\xA9", "\n", "\x80"}}; // " " is one literal
    for (std::size_t round = 0; round != scaled_iterations(16); ++round) {
        std::vector<std::string> patterns, haystacks;
        for (std::size_t index = 0; index != 1 + round % 4; ++index)
            patterns.push_back(pool[generator() % (sizeof(pool) / sizeof(pool[0]))]);
        std::vector<std::string> const &alphabet = alphabets[round % alphabets.size()];
        for (std::size_t index = 0; index != 1 + round % 40; ++index)
            haystacks.push_back(random_string(generator, generator() % 64, alphabet));
        for (std::size_t index = 0; index != round % 3; ++index)
            haystacks.push_back(random_string(generator, 4096 + generator() % 60000, alphabet));
        end_off_word_(haystacks);
        check_regex_against_serial_(backend, haystacks, patterns);
    }

    // Segments span a power of two of slots on every vendor, 2^22 on Metal and 2^24 on an H100, a
    // slot per byte and one per haystack ahead of its bytes. At every power from 2^20 on, a quoted
    // line alternately ends exactly at that edge or crosses it, so any vendor's first two edges
    // see both, each followed by an empty match where it ends, which the carry from the segment
    // before must refuse.
    std::string const prefix(100, 'a');
    std::string haystack = random_string(generator, ((std::size_t)1 << 26) + 4096, "abcdefgh ");
    for (std::size_t offset = 997; offset < haystack.size(); offset += 10007) haystack[offset] = '7';
    std::size_t const first_slot = prefix.size() + 1;
    for (std::size_t power = 20; power <= 26; ++power) {
        std::size_t const edge = ((std::size_t)1 << power) - first_slot;
        if (power % 2) haystack[edge - 40] = '"', haystack[edge + 40] = '"', haystack[edge + 41] = '\n';
        else haystack[edge - 50] = '"', haystack[edge - 2] = '"', haystack[edge - 1] = '\n';
    }
    check_regex_against_serial_(backend, {prefix, haystack, "\"x\""}, {"\"[^\"]*\"\n?", "(?m)^", "[0-9]+"});
}

/**
 *  @brief One unbalanced batch per kind of engine: a haystack of a mebibyte per unit of
 *      @c STRINGZILLA_SCALE between thousands of empty and few-byte ones.
 *
 *  Half of that haystack is one quoted span that the start-finding walk, the end walks and the
 *  lookaround passes all carry across thousands of chunks. The kinds are a cased DFA, an
 *  uncased one, position sets for the starts, a PikeVM for the ends, lookarounds ahead of the
 *  span, inside it, and before digits as DFA columns, and five of them, more than the columns
 *  hold, on positions and the PikeVM. The serial walks that check them cost most of the time,
 *  so the haystack crosses a segment only from a scale of 8 on Metal and 32 on an H100. A last
 *  batch holds a haystack of 9 MiB per unit of scale, whose end walks outlive the lookaround
 *  table's window of two segments and slide past it: on Metal at the default scale, on an H100
 *  from a scale of 4.
 */
inline void check_regex_device_unbalanced_(split_mix_generator_t &generator, auto const &backend) {
    std::vector<std::string> const words {"lorem",
                                          "ipsum",
                                          " ",
                                          " ",
                                          "\n",
                                          "Hello",
                                          "World",
                                          "42",
                                          "2026",
                                          "x",
                                          "q",
                                          "abc",
                                          "y_z",
                                          "Strasse",
                                          "stra\xC3\x9F" "e",      // "\x9Fe" is one escape
                                          "\xE2\x84\xAA" "elvin"}; // "\xAAe" is one escape
    std::size_t const length = scaled_iterations((std::size_t)1 << 20) + 4096;
    std::string haystack;
    while (haystack.size() < length) haystack += words[generator() % words.size()];
    haystack[haystack.size() / 4] = '"', haystack[haystack.size() / 4 * 3] = '"';
    std::vector<std::string> haystacks;
    for (std::size_t index = 0; index != 4096; ++index) {
        if (index == 2048) haystacks.push_back(haystack);
        haystacks.push_back(random_string(generator, generator() % 8, "aq\"x1 "));
    }
    check_regex_against_serial_(backend, haystacks, {"\"[^\"]*\"", "[0-9]+", "\\b[A-Z][a-z]+\\b"});
    check_regex_against_serial_(backend, haystacks, {"\"[^\"]*\"", "stra(ss|\xC3\x9F)e", "\\bk[a-z]+"}, sz_uncased_k);
    check_regex_against_serial_(backend, haystacks, {"\"[^\"]*\"", "x.{16}[0-9]"});
    check_regex_against_serial_(backend, haystacks, {"\"[^\"]*\"", "(?:[a-c ]{1,14}.{0,13}\\s)*a"});
    check_regex_against_serial_(backend, haystacks, {"(?<=\"[^\"]*)q", "x(?=[^\"]*\")", "\\d+(?!\\d)"});
    check_regex_against_serial_(backend, haystacks,
                                {"(?<=\"[^\"]*)q", "x(?=[^\"]*\")", "\\d+(?!\\d)", "(?<!\\w)[A-Z]", "z(?=\\s)"});

    // The first `q` opens a walk that no byte kills, matching up to digits spread through the
    // haystack and one near its end, so it pauses at the window's end and resumes in the next one;
    // whitespace runs test a lookahead too.
    std::string sliding = random_string(generator, scaled_iterations((std::size_t)9 << 20) + 4096, "abcdefgh \n");
    sliding[100] = 'q', sliding[sliding.size() - 7] = '9';
    for (std::size_t offset = 1 << 16; offset < sliding.size(); offset += 1 << 16)
        sliding[offset] = static_cast<char>('0' + offset % 7);
    std::vector<std::string> sliding_batch;
    for (std::size_t index = 0; index != 1024; ++index) {
        if (index == 512) sliding_batch.push_back(sliding);
        sliding_batch.push_back(random_string(generator, generator() % 8, "q1 \n"));
    }
    check_regex_against_serial_(backend, sliding_batch, {"(?s)q.*(?<=\\d)", "\\s+(?!\\S)"});
}

/** What the device verbs refuse: outputs a kernel cannot reach, a zero stride, replacements or a
 *  pattern the vocabulary lacks, and groups of a match wider than the lookaround table's window. */
inline void check_regex_device_safety_(auto const &backend) {
    sz_stream_t const stream = backend.runtime.stream;
    device_corpus_t corpus(backend.runtime, {"abcdabcd"});
    pinned_strings_t const vocabulary = staged_(backend.runtime, {"ab", "cd"});
    regex_device_engines_t engines(vocabulary, backend, sz_cased_k);
    sz_strings_view_t const device_tape = corpus.device;

    std::vector<sz_size_t> host_counts(1, 0);
    expect_equal(backend.regex_counts(&engines.device, &device_tape, host_counts.data(), sizeof(sz_size_t), stream),
                 sz_device_memory_mismatch_k);
    if (backend.regex_contains) {
        std::vector<sz_u1x8_t> host_contained(1, 0);
        sz_strings_view_t const empty {};
        expect_equal(backend.regex_contains(&engines.device, &device_tape, host_contained.data(), stream),
                     sz_device_memory_mismatch_k);
        expect_equal(backend.regex_contains(&engines.device, &empty, nullptr, stream), sz_success_k);
    }
    pinned_vector<sz_size_t> counts(1, STRINGZILLA_SIZE_MAX, pinned_alloc_<sz_size_t>(backend.runtime));
    expect_equal(backend.regex_counts(&engines.device, &device_tape, counts.data(), 0, stream),
                 sz_unexpected_dimensions_k);
    expect_equal(counts[0], STRINGZILLA_SIZE_MAX);

    // One replacement per pattern, and a vocabulary pattern to extract, or nothing is launched.
    device_corpus_t replacements(backend.runtime, {"x"});
    sz_strings_view_t const device_replacements = replacements.device;
    pinned_vector<sz_size_t> offsets(2, STRINGZILLA_SIZE_MAX, pinned_alloc_<sz_size_t>(backend.runtime));
    expect_equal(
        backend.regex_replace(&engines.device, &device_tape, &device_replacements, nullptr, 0, offsets.data(), stream),
        sz_unexpected_dimensions_k);
    expect_equal(backend.regex_extract(&engines.device, &device_tape, 2, nullptr, 0, offsets.data(), stream),
                 sz_unexpected_dimensions_k);
    expect_true(offsets[0] == STRINGZILLA_SIZE_MAX && offsets[1] == STRINGZILLA_SIZE_MAX);
    engines.reset();
    expect_equal(engines.heap.live_allocations, 0);

    // A group walk reads the table from the match's start to its end,
    // past the window of every vendor's segments.
    std::string spanning(((std::size_t)1 << 26) + 4096, 'a');
    spanning[100] = 'q', spanning[spanning.size() - 10] = '7';
    device_corpus_t spanning_corpus(backend.runtime, {spanning});
    pinned_strings_t const spanning_vocabulary = staged_(backend.runtime, {"(?s)q(.*)(?<=\\d)"});
    regex_device_engines_t spanning_engines(spanning_vocabulary, backend, sz_cased_k);
    sz_strings_view_t const spanning_tape = spanning_corpus.device;
    expect_equal(backend.regex_extract(&spanning_engines.device, &spanning_tape, 0, nullptr, 0, offsets.data(), stream),
                 sz_unexpected_dimensions_k);
    spanning_engines.reset();
    expect_equal(spanning_engines.heap.live_allocations, 0);
}

#pragma endregion Regex Checks

#pragma region Subdocuments Checks

/** One vocabulary of paths compiled twice: for the host and for @c stream 's device. */
struct subdocuments_device_engines_t {

    /** What the serial oracle's engine is built from. */
    handle_checked_heap_t heap;

    /** The serial oracle's engine, in plain host memory. */
    sz_subdocuments_engine_t host {};

    /** The device's engine, in pinned memory, as the host builds it and reads it on every call. */
    sz_subdocuments_engine_t device {};
    sz_stream_t stream;

    subdocuments_device_engines_t(pinned_strings_t const &vocabulary, auto const &backend,
                                  sz_subdocuments_format_t format)
        : stream(backend.runtime.stream) {
        sz_strings_view_t const paths = vocabulary.view().raw;
        sz_allocator_t pinned = backend.runtime.pinned;
        expect_equal(sz_subdocuments_engine_init_serial(&host, &paths, format, sz_subdocuments_options_none_k, nullptr,
                                                        &heap.allocator, nullptr),
                     sz_success_k);
        expect_equal(sz_subdocuments_engine_init(&device, &paths, format, sz_subdocuments_options_none_k, nullptr,
                                                 backend.runtime.capabilities, &pinned, backend.runtime.stream),
                     sz_success_k);
    }
    subdocuments_device_engines_t(subdocuments_device_engines_t const &) = delete;
    subdocuments_device_engines_t &operator=(subdocuments_device_engines_t const &) = delete;
    ~subdocuments_device_engines_t() noexcept { reset(); }
    void reset() noexcept {
        sz_subdocuments_engine_free(&device, stream);
        sz_subdocuments_engine_free(&host, nullptr);
    }
};

/** The device's boundaries, matches and counts against serial's, every match in order, under a
 *  capacity holding them all and one holding half; or, for a malformed corpus, the same refusal. */
inline void check_subdocuments_against_serial_(auto const &backend, std::vector<std::string> const &documents,
                                               std::vector<std::string> const &paths,
                                               sz_subdocuments_format_t format = sz_subdocuments_json_k) {
    device_corpus_t corpus(backend.runtime, documents);
    pinned_strings_t const vocabulary = staged_(backend.runtime, paths);
    subdocuments_device_engines_t engines(vocabulary, backend, format);
    sz_strings_view_t const pinned_tape = corpus.pinned.view().raw, device_tape = corpus.device;
    sz_stream_t const stream = backend.runtime.stream;
    std::size_t const count = documents.size();

    std::vector<sz_size_t> serial_offsets(count + 1, 0);
    sz_size_t serial_malformed = STRINGZILLA_SIZE_MAX, device_malformed = STRINGZILLA_SIZE_MAX;
    sz_status_t const serial_status = sz_subdocuments_find_serial(&engines.host, &pinned_tape, nullptr, 0,
                                                                  serial_offsets.data(), &serial_malformed, nullptr);
    if (serial_status != sz_success_k) {
        pinned_vector<sz_size_t> offsets(count + 1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
        expect_equal(serial_status, sz_invalid_syntax_k);
        expect_equal(backend.subdocuments_find(&engines.device, &device_tape, nullptr, 0, offsets.data(),
                                               &device_malformed, stream),
                     sz_invalid_syntax_k);
        expect_equal(device_malformed, serial_malformed);
        // Device walks run whole, so contains refuses what counts refuses, even past a first match.
        if (backend.subdocuments_contains) {
            pinned_vector<sz_u1x8_t> contained(sz_size_divide_round_up(count, 8), 0,
                                               pinned_alloc_<sz_u1x8_t>(backend.runtime));
            device_malformed = STRINGZILLA_SIZE_MAX;
            expect_equal(backend.subdocuments_contains(&engines.device, &device_tape, contained.data(),
                                                       &device_malformed, stream),
                         sz_invalid_syntax_k);
            expect_equal(device_malformed, serial_malformed);
        }
        return;
    }
    std::size_t const total = serial_offsets[count];
    std::vector<sz_match_t> serial_matches(total);
    expect_equal(sz_subdocuments_find_serial(&engines.host, &pinned_tape, serial_matches.data(), total,
                                             serial_offsets.data(), &serial_malformed, nullptr),
                 sz_success_k);

    sz_match_t const untouched {STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX};
    for (std::size_t const capacity : {total, total / 2}) {
        pinned_vector<sz_size_t> offsets(count + 1, STRINGZILLA_SIZE_MAX, pinned_alloc_<sz_size_t>(backend.runtime));
        pinned_vector<sz_match_t> matches(total + 1, untouched, pinned_alloc_<sz_match_t>(backend.runtime));
        expect_equal(backend.subdocuments_find(&engines.device, &device_tape, matches.data(), capacity, offsets.data(),
                                               &device_malformed, stream),
                     sz_success_k);
        join_(backend.runtime, stream);
        for (std::size_t index = 0; index <= count; ++index) expect_equal(offsets[index], serial_offsets[index]);
        for (std::size_t index = 0; index != capacity; ++index)
            expect_true(matches[index].haystack_index == serial_matches[index].haystack_index &&
                        matches[index].needle_index == serial_matches[index].needle_index &&
                        matches[index].byte_offset == serial_matches[index].byte_offset &&
                        matches[index].byte_length == serial_matches[index].byte_length);
        for (std::size_t index = capacity; index <= total; ++index)
            expect_equal(matches[index].haystack_index, STRINGZILLA_SIZE_MAX);
    }

    // The dispatched verb runs the kernel of the capability the engine recorded, so both agree.
    pinned_vector<sz_size_t> device_counts(count, 0, pinned_alloc_<sz_size_t>(backend.runtime)),
        dispatched_counts(count, 0, pinned_alloc_<sz_size_t>(backend.runtime));
    expect_equal(backend.subdocuments_counts(&engines.device, &device_tape, device_counts.data(), sizeof(sz_size_t),
                                             &device_malformed, stream),
                 sz_success_k);
    expect_equal(sz_subdocuments_counts(&engines.device, &device_tape, dispatched_counts.data(), sizeof(sz_size_t),
                                        &device_malformed, stream),
                 sz_success_k);
    join_(backend.runtime, stream);
    for (std::size_t index = 0; index != count; ++index)
        expect_true(device_counts[index] == serial_offsets[index + 1] - serial_offsets[index] &&
                    dispatched_counts[index] == device_counts[index]);
    if (backend.subdocuments_contains) {
        std::vector<sz_u1x8_t> serial_contained(sz_size_divide_round_up(count, 8), 0);
        expect_equal(sz_subdocuments_contains_serial(&engines.host, &pinned_tape, serial_contained.data(),
                                                     &serial_malformed, nullptr),
                     sz_success_k);
        check_contains_(backend.runtime, serial_contained, [&](sz_u1x8_t *contained) {
            return backend.subdocuments_contains(&engines.device, &device_tape, contained, &device_malformed, stream);
        });
        check_contains_(backend.runtime, serial_contained, [&](sz_u1x8_t *contained) {
            return sz_subdocuments_contains(&engines.device, &device_tape, contained, &device_malformed, stream);
        });
    }
    engines.reset();
    expect_equal(engines.heap.live_allocations, 0);
}

/** One random JSON value nesting at most @p depth levels, over few keys, so paths select often. */
inline std::string subdocuments_device_value_(split_mix_generator_t &generator, std::size_t depth) {
    static char const *const keys[] = {"a", "b", "c", "name", "x y", "\xC3\xA9", "a\\u0062", "\\\"q"};
    static char const *const scalars[] = {"1", "-2.5e3", "0", "true", "false", "null", "\"s\"", "\"q\\\"u\"", "\"\""};
    static char const *const blanks[] = {"", " ", "\n", "\t "}; // " " is one literal
    std::size_t const kind = depth ? generator() % 4 : 3;
    std::string text;
    if (kind == 3) return scalars[generator() % (sizeof(scalars) / sizeof(scalars[0]))];
    text += kind == 1 ? '[' : '{';
    for (std::size_t index = 0, members = generator() % 5; index != members; ++index) {
        if (index) text += ',';
        text += blanks[generator() % 4];
        if (kind != 1) text += "\"" + std::string(keys[generator() % (sizeof(keys) / sizeof(keys[0]))]) + "\":";
        text += subdocuments_device_value_(generator, depth - 1);
    }
    text += kind == 1 ? ']' : '}';
    return text;
}

/** Known corners: names, wildcards, descendants, indices and slices, escaped keys, slices an
 *  array's length settles, filters, XML paths, documents needing more frames than a thread's first
 *  stack holds, and malformed documents, which fail both tiers at the same index. */
inline void check_subdocuments_device_unit_(auto const &backend) {
    std::string const store =
        R"({"store": {"book": [{"author": "R", "price": 9}, {"author": "W"}, {"author": "M", "isbn": 0}], "bike": 1}})";
    check_subdocuments_against_serial_(backend, {store, "[]", "{}", "7", "{\"author\": [1, {\"author\": 2}]}"},
                                       {"$.store.book[*].author", "$..author", "$.store.*", "$..book[2]",
                                        "$..book[0:2]", "$..*", "$['store']['bike', 'book']"});
    check_subdocuments_against_serial_(backend, {"{\"a\\u0062\": 1, \"ab\": 2, \"\\u00e9\": [3]}"},
                                       {"$.ab", "$['\\u00e9'][0]"});

    // 45 documents run past one ballot and end mid-byte, so the bitmap's tail bits count too.
    std::vector<std::string> batch(45, "{\"b\": 2}");
    for (std::size_t index = 0; index < batch.size(); index += 3) batch[index] = "{\"a\": [1]}";
    check_subdocuments_against_serial_(backend, batch, {"$.a[0]"});

    // Negative indices and backward slices keep their array's nodes pending until it closes.
    check_subdocuments_against_serial_(
        backend, {store, "[1, [2, 3], [4, [5, 6, 7]]]", "[]"},
        {"$..book[-1]", "$..book[::-1].author", "$..[-2]", "$..book[-2:]", "$[-1:0:-1]"});

    // Filters decide at the close of the node they filter,
    // `search()` walking the regex engine's DFA there.
    check_subdocuments_against_serial_(
        backend, {store, R"([{"a": 1}, {"a": "q"}, "e", 3])"},
        {"$.store.book[?@.price < 10].author", "$..book[?@.isbn]", "$..[?@.author == 'W']", "$..[?length(@) > 1]",
         "$..[?count(@.*) == 2]", "$..[?value(@.a) == 1]", "$..[?search(@, 'e')]", "$..[?@.price && !@.isbn]"});

    // XML: names, wildcards, attributes, text around CDATA, sibling positions, namespaced names.
    std::string const library = std::string(R"(<?xml version="1.0"?><!-- c --><library xmlns:x="urn:x" id="L1">)") +
                                R"(<book id="b1"><title>R</title></book><book id="b2"><title>W<![CDATA[a<u>]]>e)" +
                                R"(</title><x:note x:kind="k">n</x:note></book><shelf><book id="b3"/></shelf>)" +
                                "tail</library>";
    check_subdocuments_against_serial_(backend, {library, "<a/>"},
                                       {"/library/book/title", "//book", "/library/book[2]/title/text()", "//@id",
                                        "/library/*/@*", "/library/text()", "//x:note/@x:kind", "//book[1]"},
                                       sz_subdocuments_xml_k);

    // Forty levels, past the first stacks, so the deep documents walk again with deeper ones:
    // plain, under a pending run and a filter at every depth, and as XML elements.
    std::string const deep = std::string(40, '[') + "1" + std::string(40, ']');
    check_subdocuments_against_serial_(backend, {deep, "[[1]]", deep + ' '}, {"$..*", "$..[0]"});
    check_subdocuments_against_serial_(backend, {deep, "[[1]]", deep}, {"$..[-1]", "$..[?@ == 1]", "$..[?@[?@ == 1]]"});
    std::string deep_markup;
    for (std::size_t level = 0; level != 40; ++level) deep_markup += "<a id=\"" + std::to_string(level) + "\">t";
    for (std::size_t level = 0; level != 40; ++level) deep_markup += "</a>";
    check_subdocuments_against_serial_(backend, {deep_markup, "<a><a/></a>"}, {"//a", "//a/text()", "//a[1]/@id"},
                                       sz_subdocuments_xml_k);

    // A warp walks keys, colons, values and escapes a lane each: every one lands on every lane of
    // its batch and on both sides of a window's edge, keys leave their live sets to the next batch,
    // and escaped keys, element positions and nesting past the stacks carry across batches.
    std::vector<std::string> shifted;
    std::vector<std::string> const pointers = {"$.type", "$.actor.login", "$.repo.name", "$[33].type"};
    for (std::size_t filler = 990; filler != 1060; ++filler)
        shifted.push_back("{\"pad\": \"" + std::string(filler, 'f') + "\\\\\\\"\", \"type\": \"x\", \"actor\": " +
                          "{\"login\": \"y\", \"id\": [1, {\"a\": 2}], \"n\": 12345678}, \"repo\": {\"name\": \"z\"}}");
    for (std::size_t length = 1020; length != 1030; ++length)
        shifted.push_back("{\"type\":\"" + std::string(length - 11, 'f') + "\"}");
    for (std::size_t members = 0; members != 40; ++members) {
        std::string text = "{";
        for (std::size_t index = 0; index != members; ++index) text += "\"k\": [1], \"n\": 2, ";
        shifted.push_back(text + "\"type\": [3], \"actor\": {\"login\": 4}, \"ty\\u0070e\": 5}");
    }
    std::string array = "[", nest, closers;
    for (std::size_t index = 0; index != 80; ++index)
        array += (index ? ", " : "") +
                 (index % 3 ? std::to_string(index) : "{\"type\": [" + std::to_string(index) + "]}");
    shifted.push_back(array + "]");
    for (std::size_t level = 0; level != 40; ++level) nest += "{\"actor\": [", closers += "]}";
    shifted.push_back(nest + "{\"login\": 1}" + closers);
    check_subdocuments_against_serial_(backend, shifted, pointers);
    check_subdocuments_against_serial_(backend, shifted, {"$..login", "$..[1]", "$..type[0]"});
    // Malformed in a walked container, after the root, or inside a skipped one, which only its
    // brackets are counted in, so both tiers accept it.
    for (std::string const malformed :
         {"{\"actor\": {\"login\" 1}}", "{\"type\": \"x\"} 2", "{\"type\": \"x\"}}", "{\"type\": \"x\"]",
          "{\"type\": \"x\" \"actor\": 1}", "{\"type\": \"x\",}", "{\"type\"}", "{\"type\": tru}", "{\"type\": \"x",
          "{\"actor\": {\"login\": 1}", "{\"actor\": {\"login\": 1]}", "{, \"type\": 1}", "{\"type\": 1,, \"a\": 2}",
          "{\"type\"::1}", "{\"ty\\qe\": 1}", "{\"a\": [1 2], \"type\": 1}", "{\"a\": {\"b\": [}], \"type\": 1}",
          "{\"a\": {\"\\q\": 1}, \"type\": 1}"})
        for (std::size_t filler : {0, 1000, 1010, 1017, 1023})
            check_subdocuments_against_serial_(
                backend,
                {"{}", filler ? "{\"pad\": \"" + std::string(filler, 'f') + "\", " + malformed.substr(1) : malformed},
                pointers);
    // A `/` outside strings fails a document even inside a skipped container, on either side of a
    // window's edge and in a document long enough for the device to walk in chunks, while slashes
    // in strings pass.
    for (std::size_t filler : {0, 1000, 1017, 1023, 2050, 140000})
        for (char const *between : {", /**/ ", ", // ]\n", ", \"/*\", "})
            check_subdocuments_against_serial_(
                backend, {"{}", "{\"a\": [\"" + std::string(filler, 'f') + "\"" + between + "1], \"type\": 1}"},
                pointers);

    // Unbalanced, unterminated and stray bytes, each after a well-formed document.
    check_subdocuments_against_serial_(backend, {"{}", "{\"a\": [1, 2}"}, {"$..*"});
    check_subdocuments_against_serial_(backend, {"[1]", "[1]", "{\"a\": \"open"}, {"$.a"});
    check_subdocuments_against_serial_(backend, {"[1] x"}, {"$[0]"});
    check_subdocuments_against_serial_(backend, {"<a/>", "<a><b></a>"}, {"//a"}, sz_subdocuments_xml_k);
    check_subdocuments_against_serial_(backend, {"<a/>", "<!DOCTYPE a [<!ENTITY e 'x'>]><a/>"}, {"//a"},
                                       sz_subdocuments_xml_k);
}

/** A JSON document of tens of megabytes, which the device walks over chunks: a wide array of
 *  scalars, a wide array of objects with escaped keys and strings, a chain 100'000 levels deep, and
 *  strings whose runs of backslashes cross chunks. */
inline std::string subdocuments_device_large_(split_mix_generator_t &generator) {
    std::string text = "{\"wide\": [", closers;
    for (std::size_t index = 0; index != 3000000; ++index) text += (index ? "," : "") + std::to_string(index % 997);
    text += "], \"objects\": [";
    for (std::size_t index = 0; index != 150000; ++index)
        text += (index ? ", {\"a\": " : "{\"a\": ") + std::to_string(index) + ", \"b\": [" + std::to_string(index % 7) +
                ", \"q\\\"\\\\\"], \"name\": \"n" + std::to_string(index % 13) + "\", \"a\\u0062\": true}";
    text += "], \"deep\": ";
    for (std::size_t level = 0; level != 100000; ++level) {
        text += level % 3 ? "[" : "{\"x\": ";
        closers += level % 3 ? ']' : '}';
    }
    text += "{\"a\": 1, \"x\": [1, 2, 3]}";
    text.append(closers.rbegin(), closers.rend());
    text += ", \"strings\": [";
    for (std::size_t index = 0; index != 200; ++index)
        text += (index ? ", \"" : "\"") + std::string(2 * (generator() % 400), '\\') + "\\\"e\"";
    return text + "]}";
}

/** One unbalanced batch: a document of tens of megabytes between two of a few megabytes, amid
 *  thousands of empty, scalar and few-byte ones, under names, wildcards, descendants, positive and
 *  negative indices, backward slices and unions; then the same batch with a comma missing inside an
 *  array the paths enter, which fails at the middle document's index while the large documents
 *  around it still walk in one pipeline. */
inline void check_subdocuments_device_unbalanced_(split_mix_generator_t &generator, auto const &backend) {
    static char const *const tiny[] = {"{}", "[]", "7", "\"s\"", "null", "true", "[1]", "{\"a\": 1}", "-0.5e3"};
    std::vector<std::string> const paths = {"$.wide[-1]",     "$.wide[::-1000000]",
                                            "$.objects[*].a", "$..a",
                                            "$..b[-1]",       "$.objects[5:1:-2]['a', 'name']",
                                            "$..x[-1]",       "$['wide', 'strings'][0]",
                                            "$..ab",          "$.objects[-1:-4:-1].b[-2]",
                                            "$.deep..a",      "$.*"};
    std::string const large = subdocuments_device_large_(generator);
    std::string wide = "{\"wide\": [", objects = "[";
    for (std::size_t index = 0; index != 400000; ++index) wide += (index ? "," : "") + std::to_string(index % 991);
    for (std::size_t index = 0; index != 60000; ++index)
        objects += (index ? ", {\"a\": " : "{\"a\": ") + std::to_string(index) + ", \"x\": [" +
                   std::to_string(index % 5) + "]}";
    wide += "], \"a\": 1}", objects += "]";
    std::vector<std::string> documents;
    std::size_t middle = 0;
    for (std::size_t index = 0; index != 4000; ++index) {
        if (index == 1000) documents.push_back(wide);
        if (index == 2000) middle = documents.size(), documents.push_back(large);
        if (index == 3000) documents.push_back(objects);
        documents.push_back(index % 3 ? std::string(tiny[index % 9]) : subdocuments_device_value_(generator, 2));
    }
    check_subdocuments_against_serial_(backend, documents, paths);
    documents[middle][large.find(',')] = ' ';
    check_subdocuments_against_serial_(backend, documents, paths);
}

/** Random documents of every size, under random vocabularies of JSON paths, slices and filters
 *  among them, and one unbalanced batch. */
inline void check_subdocuments_device_equivalence_(split_mix_generator_t &generator, auto const &backend) {
    static char const *const pool[] = {"$.a",      "$.a.b",     "$..b",        "$.*",         "$[*]",     "$..[0]",
                                       "$.a[1:3]", "$..c[::2]", "$['a', 'b']", "$..*",        "$.name.*", "$..['x y']",
                                       "$[-1]",    "$..[::-2]", "$..[?@.a]",   "$..[?@ == 1]"};
    for (std::size_t round = 0; round != scaled_iterations(16); ++round) {
        std::vector<std::string> paths, documents;
        for (std::size_t index = 0; index != 1 + round % 4; ++index)
            paths.push_back(pool[generator() % (sizeof(pool) / sizeof(pool[0]))]);
        for (std::size_t index = 0; index != 1 + round % 64; ++index)
            documents.push_back(subdocuments_device_value_(generator, 1 + round % 6));
        for (std::size_t index = 0; index != round % 3; ++index) {
            std::string array = "[";
            for (std::size_t element = 0; element != 2000; ++element)
                array += (element ? "," : "") + subdocuments_device_value_(generator, 4);
            documents.push_back(array + "]");
        }
        end_off_word_(documents);
        check_subdocuments_against_serial_(backend, documents, paths);
    }
    check_subdocuments_device_unbalanced_(generator, backend);
}

/** What the device verbs refuse: outputs a kernel cannot reach, and a zero stride. */
inline void check_subdocuments_device_safety_(auto const &backend) {
    sz_stream_t const stream = backend.runtime.stream;
    device_corpus_t corpus(backend.runtime, {"{\"a\": 1}"});
    pinned_strings_t const vocabulary = staged_(backend.runtime, {"$.a"});
    subdocuments_device_engines_t engines(vocabulary, backend, sz_subdocuments_json_k);
    sz_strings_view_t const device_tape = corpus.device;
    sz_size_t malformed = STRINGZILLA_SIZE_MAX;
    std::vector<sz_size_t> host_counts(1, 0);
    expect_equal(backend.subdocuments_counts(&engines.device, &device_tape, host_counts.data(), sizeof(sz_size_t),
                                             &malformed, stream),
                 sz_device_memory_mismatch_k);
    if (backend.subdocuments_contains) {
        std::vector<sz_u1x8_t> host_contained(1, 0);
        sz_strings_view_t const empty {};
        expect_equal(
            backend.subdocuments_contains(&engines.device, &device_tape, host_contained.data(), &malformed, stream),
            sz_device_memory_mismatch_k);
        expect_equal(backend.subdocuments_contains(&engines.device, &empty, nullptr, &malformed, stream), sz_success_k);
    }
    pinned_vector<sz_size_t> counts(1, STRINGZILLA_SIZE_MAX, pinned_alloc_<sz_size_t>(backend.runtime));
    expect_equal(backend.subdocuments_counts(&engines.device, &device_tape, counts.data(), 0, &malformed, stream),
                 sz_unexpected_dimensions_k);
    expect_equal(counts[0], STRINGZILLA_SIZE_MAX);
    engines.reset();
    expect_equal(engines.heap.live_allocations, 0);
}

#pragma endregion Subdocuments Checks

#pragma region UTF8 Uncased Checks

/** @p fold called the way a CPU fold is, its text, output and length staged where the device
 *  reaches them and the stream joined, so the CPU fold checks drive it unchanged. */
inline auto utf8_uncased_fold_staged_(auto const &backend) {
    return [&backend](sz_cptr_t source, sz_size_t length, sz_ptr_t target, sz_size_t *target_length,
                      sz_stream_t) -> sz_status_t {
        pinned_vector<char> staged_source(source, source + length, pinned_alloc_<char>(backend.runtime)),
            staged_target(length * 3 + 4, char(0x5A), pinned_alloc_<char>(backend.runtime));
        pinned_vector<sz_size_t> staged_length(1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
        sz_status_t const status = backend.utf8_uncased_fold(staged_source.data(), length, staged_target.data(),
                                                             staged_length.data(), backend.runtime.stream);
        if (status != sz_success_k) return status;
        join_(backend.runtime, backend.runtime.stream);
        expect_true(staged_length[0] <= length * 3);
        expect_true(std::all_of(staged_target.begin() + length * 3, staged_target.end(),
                                [](char byte) { return byte == char(0x5A); }));
        std::copy(staged_target.begin(), staged_target.begin() + staged_length[0], target);
        *target_length = staged_length[0];
        return status;
    };
}

/** The device fold against the serial one, over the battery every CPU fold is held to. */
inline void check_utf8_uncased_device_equivalence_(split_mix_generator_t &generator, auto const &backend) {
    check_uncased_fold_equivalence_(generator, sz_utf8_uncased_fold_serial, utf8_uncased_fold_staged_(backend), 4000,
                                    scaled_iterations(1200));
}

/** The malformed inputs every CPU fold survives, which the device folds byte for byte as serial
 *  does, and memory the device cannot reach, which it refuses rather than reading. */
inline void check_utf8_uncased_device_safety_(split_mix_generator_t &generator, auto const &backend) {
    auto const staged = utf8_uncased_fold_staged_(backend);
    std::vector<char> expected, produced;
    auto const compare = [&](char const *input, std::size_t length) {
        expected.resize(length * 3 + 4), produced.resize(length * 3 + 4);
        sz_size_t const expected_length = kernel_result<sz_size_t>(sz_utf8_uncased_fold_serial, input, length,
                                                                   expected.data());
        sz_size_t const produced_length = kernel_result<sz_size_t>(staged, input, length, produced.data());
        if (produced_length != expected_length ||
            !std::equal(expected.begin(), expected.begin() + expected_length, produced.begin()))
            fail_backend_(backend.name, "a text folds differently from serial");
    };
    for_each_adversarial_utf8_input_(generator, scaled_iterations(1000), compare);
    for (std::size_t length : {std::size_t(16383), std::size_t(16384), std::size_t(16385)}) {
        std::string input(length, 'A');
        input.replace(length - 2, 2, "\xC3\x9F", 2);
        compare(input.data(), input.size());
    }
    std::string malformed(65539, char(0x80));
    malformed += "\xC3\x9F\xF0\x9F";
    compare(malformed.data(), malformed.size());

    std::string const host_source = "HELLO";
    pinned_vector<char> source(host_source.begin(), host_source.end(), pinned_alloc_<char>(backend.runtime)),
        target(host_source.size() * 3, pinned_alloc_<char>(backend.runtime));
    pinned_vector<sz_size_t> target_length(1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
    std::vector<char> host_target(host_source.size() * 3);
    sz_size_t host_length = 0;
    if (backend.utf8_uncased_fold(host_source.data(), host_source.size(), target.data(), target_length.data(),
                                  backend.runtime.stream) != sz_device_memory_mismatch_k ||
        backend.utf8_uncased_fold(source.data(), source.size(), host_target.data(), target_length.data(),
                                  backend.runtime.stream) != sz_device_memory_mismatch_k ||
        backend.utf8_uncased_fold(source.data(), source.size(), target.data(), &host_length, backend.runtime.stream) !=
            sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "a fold of memory the device cannot reach was not refused");
}

#pragma endregion UTF8 Uncased Checks

#pragma region UTF8 Norm Checks

/** @p norm called the way a CPU normalizer is, its text, output and length staged where the device
 *  reaches them and the stream joined, so the CPU normalization checks drive it unchanged. */
inline auto utf8_norm_staged_(auto const &backend) {
    return [&backend](sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t target,
                      sz_size_t *target_length, sz_stream_t) -> sz_status_t {
        pinned_vector<char> staged_source(source, source + length, pinned_alloc_<char>(backend.runtime)),
            staged_target(length * 18 + 4, char(0x5A), pinned_alloc_<char>(backend.runtime));
        pinned_vector<sz_size_t> staged_length(1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
        sz_status_t const status = backend.utf8_norm(staged_source.data(), length, form, staged_target.data(),
                                                     staged_length.data(), backend.runtime.stream);
        if (status != sz_success_k) return status;
        join_(backend.runtime, backend.runtime.stream);
        expect_true(staged_length[0] <= length * 18);
        expect_true(std::all_of(staged_target.begin() + length * 18, staged_target.end(),
                                [](char byte) { return byte == char(0x5A); }));
        std::copy(staged_target.begin(), staged_target.begin() + staged_length[0], target);
        *target_length = staged_length[0];
        return status;
    };
}

/** The device normalizer against the serial one over the shared Unicode corpus. */
inline void check_utf8_norm_device_equivalence_(split_mix_generator_t &generator, auto const &backend) {
    check_utf8_normalize_equivalence_(generator, utf8_norm_staged_(backend));
}

/** Adversarial inputs and bulk boundaries against serial, plus inaccessible-memory refusals. */
inline void check_utf8_norm_device_safety_(split_mix_generator_t &generator, auto const &backend) {
    static sz_normal_form_t const norm_forms[4] = {sz_normal_form_nfd_k, sz_normal_form_nfc_k, sz_normal_form_nfkd_k,
                                                   sz_normal_form_nfkc_k};
    auto const staged = utf8_norm_staged_(backend);
    std::vector<char> expected, produced;
    auto const compare = [&](char const *input, std::size_t length) {
        expected.resize(length * 18 + 18), produced.resize(length * 18 + 18);
        for (sz_normal_form_t form : norm_forms) {
            sz_size_t const expected_length = kernel_result<sz_size_t>(sz_utf8_norm_serial, input, length, form,
                                                                       expected.data());
            sz_size_t const produced_length = kernel_result<sz_size_t>(staged, input, length, form, produced.data());
            if (produced_length != expected_length ||
                !std::equal(expected.begin(), expected.begin() + expected_length, produced.begin()))
                fail_backend_(backend.name, "a text normalizes differently from serial");
        }
    };
    for_each_adversarial_utf8_input_(generator, scaled_iterations(1000), [&](char const *input, std::size_t length) {
        if (sz_utf8_find_malformed(input, length) == STRINGZILLA_NULL_CHAR) compare(input, length);
    });
    std::string hangul(17 * 1024 - 3, 'x');
    hangul += "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8";
    hangul.append(1024, 'y');
    compare(hangul.data(), hangul.size());
    std::string marks = "a";
    for (std::size_t index = 0; index != 70000; ++index) marks += index % 2 ? "\xCC\x80" : "\xCC\x95";
    marks += "\xFF";
    marks += "e\xCC\x81";
    compare(marks.data(), marks.size());
    std::string malformed(65539, char(0x80));
    malformed += "e\xCC\x81\xF0\x9F";
    compare(malformed.data(), malformed.size());

    std::string const host_source = "caf\xC3\xA9";
    pinned_vector<char> source(host_source.begin(), host_source.end(), pinned_alloc_<char>(backend.runtime)),
        target(host_source.size() * 18, pinned_alloc_<char>(backend.runtime));
    pinned_vector<sz_size_t> target_length(1, 0, pinned_alloc_<sz_size_t>(backend.runtime));
    std::vector<char> host_target(host_source.size() * 18);
    sz_size_t host_length = 0;
    if (backend.utf8_norm(host_source.data(), host_source.size(), sz_normal_form_nfd_k, target.data(),
                          target_length.data(), backend.runtime.stream) != sz_device_memory_mismatch_k ||
        backend.utf8_norm(source.data(), source.size(), sz_normal_form_nfd_k, host_target.data(), target_length.data(),
                          backend.runtime.stream) != sz_device_memory_mismatch_k ||
        backend.utf8_norm(source.data(), source.size(), sz_normal_form_nfd_k, target.data(), &host_length,
                          backend.runtime.stream) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "a normalization of memory the device cannot reach was not refused");
}

#pragma endregion UTF8 Norm Checks

#pragma region Strings Checks

/** The same strings where a device reaches them, in both layouts a device verb reads: a tape of
 *  32-bit offsets the device gathers from a pinned one, and a hand-built pinned tape of 64-bit
 *  ones, which the device reads over the bus, as the library gathers no wide offsets. */
class strings_device_tapes_t {

    /** The 32-bit tape, gathered on the host into pinned blocks. */
    pinned_strings_t narrow_pinned_;

    /** The 32-bit tape, gathered by the device from @c narrow_pinned_ into the backend's arena. */
    sz_strings_view_t narrow_device_ {};

    /** The bytes of every string of the 64-bit tape, back to back. */
    pinned_vector<char> wide_payload_;

    /** Offsets into @c wide_payload_, one per string and one past the end of the last string. */
    pinned_vector<sz_u64_t> wide_offsets_;

  public:
    strings_device_tapes_t(device_backend_t const &runtime, std::vector<std::string> const &strings)
        : narrow_pinned_(staged_(runtime, strings)), narrow_device_(uploaded_(runtime, narrow_pinned_.view().raw)),
          wide_payload_(pinned_alloc_<char>(runtime)), wide_offsets_(1, 0, pinned_alloc_<sz_u64_t>(runtime)) {
        for (std::string const &string : strings) {
            wide_payload_.insert(wide_payload_.end(), string.begin(), string.end());
            wide_offsets_.push_back(wide_payload_.size());
        }
    }

    /** Both layouts, by name. */
    std::array<named_strings_t, 2> all() const noexcept {
        strings_view_t const wide(wide_payload_.data(), wide_offsets_.data(), wide_offsets_.size() - 1);
        return {{{"a 32-bit tape", narrow_device_}, {"a 64-bit tape", wide.raw}}};
    }
};

/** Writes @p strings into pinned rows of widths around a vector's from both device layouts, in
 *  order and through @p order, demanding serial's bytes, stride gaps and longest length. */
inline void check_strings_device_rows_(device_backend_t const &runtime, strings_backend_t const &kernels,
                                       std::vector<std::string> const &strings, pinned_vector<sz_size_t> const &order) {
    strings_layouts_t const host(strings);
    sz_strings_view_t const host_view = host.u32_tape();
    strings_device_tapes_t const tapes(runtime, strings);
    sz_size_t const *const selections[] = {nullptr, order.data()};
    for (named_strings_t const &layout : tapes.all())
        for (sz_size_t const *selection : selections)
            for (sz_size_t const width : {0, 1, 7, 33, 64}) {
                sz_size_t const count = selection ? order.size() : strings.size(), stride = width + 3;
                std::vector<char> expected(count * stride, '#');
                pinned_vector<char> written(count * stride, '#', pinned_alloc_<char>(runtime));
                pinned_vector<sz_size_t> written_longest(1, 0, pinned_alloc_<sz_size_t>(runtime));
                sz_strings_rows_t const expected_rows {expected.data(), stride, width};
                sz_strings_rows_t const written_rows {written.data(), stride, width};
                sz_size_t expected_longest = 0;
                expect_equal(
                    sz_strings_to_rows_serial(&host_view, selection, count, &expected_rows, &expected_longest, nullptr),
                    sz_success_k);
                expect_equal(kernels.to_rows(&layout.view, selection, count, &written_rows, written_longest.data(),
                                             runtime.stream),
                             sz_success_k);
                join_(runtime, runtime.stream);
                if (!std::equal(expected.begin(), expected.end(), written.begin()) ||
                    written_longest[0] != expected_longest)
                    fail_backend_(kernels.name, fmt::format("rows {} wide from {} {} differ from serial", width,
                                                            layout.name, selection ? "reordered" : "in order")
                                                    .c_str());
            }
}

/** Runs one verb of @p kernels and of serial over @p strings, the device reading both its layouts,
 *  in order and through @p order, into a zeroed span and in place, demanding the same outcome. */
inline void check_strings_device_verb_(device_backend_t const &runtime, strings_backend_t const &kernels,
                                       strings_verb_t verb, std::vector<std::string> const &strings,
                                       pinned_vector<sz_size_t> const &order) {
    sized_heap_t heap;
    sz_allocator_t pinned = runtime.pinned;
    strings_layouts_t const host(strings);
    sz_strings_view_t const host_view = host.u32_tape();
    strings_device_tapes_t const tapes(runtime, strings);
    sz_size_t const *const selections[] = {nullptr, order.data()};
    for (sz_size_t const *selection : selections) {
        sz_size_t const count = selection ? order.size() : strings.size();
        sz_strings_span_t reference {};
        strings_outcome_t const expected = strings_outcome_(strings_serial_backend_(), verb, host_view, selection,
                                                            count, &heap.allocator, reference);
        sz_strings_span_free(&reference, &heap.allocator, nullptr);
        auto const demand = [&](char const *layout, char const *span, strings_outcome_t const &outcome) {
            if (outcome == expected) return;
            fmt::println(stderr, "{} from {} {} into {}: {} instead of {}", strings_verb_name_(verb), layout,
                         selection ? "reordered" : "in order", span, outcome, expected);
            fail_backend_(kernels.name, "a device strings kernel disagreed with serial");
        };

        for (named_strings_t const &layout : tapes.all()) {
            sz_strings_span_t fresh {};
            demand(layout.name, "a zeroed span",
                   strings_outcome_(kernels, verb, layout.view, selection, count, &pinned, fresh, &runtime));
            sz_strings_span_free(&fresh, &pinned, runtime.stream);
        }

        sz_strings_span_t aliased {};
        expect_equal(
            sz_strings_to_tape_best(&host_view, nullptr, strings.size(), &pinned, &aliased, sz_cap_serial_k, nullptr),
            sz_success_k);
        sz_strings_view_t own;
        sz_strings_from_span(&aliased, &own);
        demand("the tape it writes", "itself",
               strings_outcome_(kernels, verb, own, selection, count, &pinned, aliased, &runtime));
        sz_strings_span_free(&aliased, &pinned, runtime.stream);
    }
    expect_true(heap.mismatches == 0 && heap.blocks.empty());
}

/**
 *  @brief Holds one vendor's strings kernels to serial over device-resident tapes: gathering, a
 *      lookup, rows, every escape kind both ways, and every normalization form.
 *
 *  The collections are empty, all empty strings past a block of 64, and random mixes of
 *  escape-worthy bytes, letters and marks to normalize, empty strings and ill-formed UTF-8.
 *  Unescaping reads both what serial escaped, which succeeds, and the raw strings, which are
 *  malformed somewhere for most kinds.
 */
inline void check_strings_device_equivalence_(split_mix_generator_t &generator, device_backend_t const &runtime,
                                              strings_backend_t const &kernels) {
    std::vector<std::string> characters = alphabet_characters(
        "aZ09 ,;&#%+~-._/<>'\"\\\n\r\t\x01\x1f\x7f\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80\xC2\x85\xE2\x80\xAE");
    // Ill-formed bytes, then marks, a syllable, jamo, a ligature, a fraction and an ignorable.
    for (char const *character :
         {"\xFF", "\xC0", "\x80", "\xED\xA0", "\xCC\x81", "\xCC\x96", "\xC3\x9F", "\xEA\xB0\x80", "\xE1\x84\x80",
          "\xE1\x85\xA1", "\xEF\xAC\x81", "\xC2\xBD", "\xE2\x84\xAB", "\xC2\xAD"})
        characters.push_back(character);
    characters.push_back(std::string(1, '\0'));

    std::vector<std::vector<std::string>> collections {{}, std::vector<std::string>(70)};
    std::uniform_int_distribution<std::size_t> count_distribution(1, 300), length_distribution(0, 40);
    for (std::size_t iteration = 0; iteration != scaled_iterations(2); ++iteration) {
        std::vector<std::string> strings(count_distribution(generator));
        for (std::string &string : strings) {
            std::size_t const length = length_distribution(generator);
            string = random_string(generator, length == 40 ? 300 : length, characters);
        }
        collections.push_back(std::move(strings));
    }

    sized_heap_t heap;
    for (std::vector<std::string> const &strings : collections) {
        // Up to twice as many indices as strings, repeating some and skipping others, where the
        // device reads them.
        pinned_vector<sz_size_t> order(pinned_alloc_<sz_size_t>(runtime));
        if (!strings.empty()) {
            order.resize(1 + generator() % (2 * strings.size()));
            for (sz_size_t &index : order) index = generator() % strings.size();
        }
        check_strings_device_rows_(runtime, kernels, strings, order);
        for (strings_verb_t const verb : strings_verbs_()) {
            check_strings_device_verb_(runtime, kernels, verb, strings, order);
            if (verb.kernel != sz_kernel_strings_unescape_k) continue;
            strings_layouts_t const layouts(strings);
            sz_strings_span_t escaped {};
            expect_equal(strings_outcome_(strings_serial_backend_(), {sz_kernel_strings_escape_k, verb.kind},
                                          layouts.u32_tape(), nullptr, strings.size(), &heap.allocator, escaped)
                             .status,
                         sz_success_k);
            check_strings_device_verb_(runtime, kernels, verb, strings_in_(escaped), order);
            sz_strings_span_free(&escaped, &heap.allocator, nullptr);
        }
    }
    expect_true(heap.mismatches == 0 && heap.blocks.empty());
}

/** Every device verb refuses an accessor, host code even over bytes the device reaches, and an
 *  output span whose blocks the device cannot reach, leaving that span as it was. */
inline void check_strings_device_safety_(device_backend_t const &runtime, strings_backend_t const &kernels) {
    std::vector<std::string> const strings {"kitten", "", "sitting"};
    strings_device_tapes_t const tapes(runtime, strings);
    sz_strings_view_t const accessor = strings_from_(strings);
    sz_allocator_t pinned = runtime.pinned;
    for (strings_verb_t const verb : strings_verbs_()) {
        sz_strings_span_t untouched {};
        expect_equal(
            strings_outcome_(kernels, verb, accessor, nullptr, strings.size(), &pinned, untouched, &runtime).status,
            sz_device_memory_mismatch_k);
        expect_equal(untouched.count, 0);
    }
    pinned_vector<char> rows_data(3 * 8, '#', pinned_alloc_<char>(runtime));
    pinned_vector<sz_size_t> longest(1, 0, pinned_alloc_<sz_size_t>(runtime));
    sz_strings_rows_t const rows {rows_data.data(), 8, 8};
    expect_equal(kernels.to_rows(&accessor, nullptr, strings.size(), &rows, longest.data(), runtime.stream),
                 sz_device_memory_mismatch_k);

    sized_heap_t heap;
    std::vector<std::string> const reused {"a reused span", "", "of other sizes"};
    sz_strings_span_t host_span = span_from_(reused, sz_strings_u32_tape_k, heap.allocator);
    for (named_strings_t const &layout : tapes.all())
        expect_equal(kernels.to_tape(&layout.view, nullptr, strings.size(), &pinned, &host_span, runtime.stream),
                     sz_device_memory_mismatch_k);
    join_(runtime, runtime.stream);
    expect_equal(strings_in_(host_span), reused);
    expect_true(std::all_of(rows_data.begin(), rows_data.end(), [](char byte) { return byte == '#'; }));
    sz_strings_span_free(&host_span, &heap.allocator, nullptr);
    expect_true(heap.mismatches == 0 && heap.blocks.empty());
}

#pragma endregion Strings Checks

#pragma region Drivers

/** Registers the Levenshtein checks of one vendor's kernels, of one tier's, or of the dispatch
 *  points, behind @p gate . */
inline void check_device_levenshtein_(capability_gate_t const &gate, auto const &backend) {
    std::string const suffix = backend.name;
    run_test_case(("check_levenshtein_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_levenshtein_device_equivalence_(backend, levenshtein_device_alphabet_t::bytes_k,
                                                            sz_levenshtein_bytes_k);
                      check_levenshtein_device_equivalence_(backend, levenshtein_device_alphabet_t::runes_k,
                                                            sz_levenshtein_runes_k);
                      check_levenshtein_device_equivalence_(backend, levenshtein_device_alphabet_t::bytes_k,
                                                            sz_levenshtein_runes_k);
                      check_levenshtein_device_short_queries_(generator, backend);
                      check_levenshtein_device_mixed_(backend);
                  }));
    run_test_case(("check_levenshtein_device_skewed_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_levenshtein_device_skewed_(generator, backend, levenshtein_device_alphabet_t::bytes_k);
                      check_levenshtein_device_skewed_(generator, backend, levenshtein_device_alphabet_t::runes_k);
                  }));
    run_test_case(("check_levenshtein_device_memory_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_levenshtein_device_memory_safety_(generator, backend);
                  }));
}

/** Registers the N-grams checks of one vendor's kernels, of one tier's, or of the dispatch
 *  points, behind @p gate . */
inline void check_device_ngrams_(capability_gate_t const &gate, auto const &backend) {
    std::string const suffix = backend.name;
    run_test_case(("check_ngrams_overlaps_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_ngrams_overlaps_device_equivalence_(generator, backend);
                  }));
    run_test_case(("check_ngrams_overlaps_device_skewed_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_ngrams_overlaps_device_skewed_(generator, backend);
                  }));
    run_test_case(("check_ngrams_overlaps_device_memory_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_ngrams_overlaps_device_memory_safety_(generator, backend);
                      check_ngrams_overlaps_device_width_safety_(generator, backend);
                      check_ngrams_overlaps_device_past_query_(generator, backend);
                  }));
    run_test_case(("check_ngrams_sketches_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_ngrams_sketches_device_equivalence_(generator, backend);
                  }));
    run_test_case(("check_ngrams_sketches_device_memory_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_ngrams_sketches_device_memory_safety_(generator, backend);
                  }));
}

/** Registers the substrings checks of one vendor's kernels, of one tier's, or of the dispatch
 *  points, behind @p gate . */
inline void check_device_substrings_(capability_gate_t const &gate, auto const &backend) {
    std::string const suffix = backend.name;
    run_test_case(("check_substrings_device_unit_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&] { check_substrings_device_unit_(backend); }));
    run_test_case(("check_substrings_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_substrings_device_equivalence_(generator, backend);
                  }));
    run_test_case(("check_substrings_device_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&] { check_substrings_device_safety_(backend); }));
}

/** Registers the regex checks of one vendor's kernels, or of the dispatch points, behind
 *  @p gate. */
inline void check_device_regex_(capability_gate_t const &gate, auto const &backend) {
    std::string const suffix = backend.name;
    run_test_case(("check_regex_device_unit_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&] { check_regex_device_unit_(backend); }));
    run_test_case(("check_regex_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_regex_device_equivalence_(generator, backend);
                  }));
    run_test_case(("check_regex_device_unbalanced_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_regex_device_unbalanced_(generator, backend);
                  }));
    run_test_case(("check_regex_device_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&] { check_regex_device_safety_(backend); }));
}

/** Registers the subdocuments checks of a vendor's kernels, or of dispatch points, behind
 *  @p gate. */
inline void check_device_subdocuments_(capability_gate_t const &gate, auto const &backend) {
    std::string const suffix = backend.name;
    run_test_case(("check_subdocuments_device_unit_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&] { check_subdocuments_device_unit_(backend); }));
    run_test_case(("check_subdocuments_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_subdocuments_device_equivalence_(generator, backend);
                  }));
    run_test_case(("check_subdocuments_device_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&] { check_subdocuments_device_safety_(backend); }));
}

/** Registers the strings checks of one vendor's kernels behind @p gate , which no tier narrows. */
inline void check_device_strings_(capability_gate_t const &gate, device_backend_t const &runtime,
                                  strings_backend_t const &kernels) {
    std::string const suffix = kernels.name;
    run_test_case(("check_strings_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(runtime, [&](split_mix_generator_t &generator) {
                      check_strings_device_equivalence_(generator, runtime, kernels);
                  }));
    run_test_case(("check_strings_device_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(runtime, [&] { check_strings_device_safety_(runtime, kernels); }));
}

/** Multi-state SHA-256 on the device against serial, with half the lanes carrying a buffered prefix
 *  from an earlier call, so the head, body and tail paths all run. */
inline void check_sha256_multistate_device_equivalence_(split_mix_generator_t &generator, auto const &backend) {
    device_backend_t const &runtime = backend.runtime;
    sz_size_t const lanes_count = (sz_size_t)scaled_iterations(4096);
    fuzzy_config_t config;
    config.batch_size = (std::size_t)lanes_count;
    config.min_string_length = 0;
    config.max_string_length = 4 * STRINGZILLA_SHA256_BLOCK_LENGTH + 3;
    std::vector<std::string> messages, prefixes;
    randomize_strings(generator, config, messages);
    config.max_string_length = STRINGZILLA_SHA256_BLOCK_LENGTH - 1;
    randomize_strings(generator, config, prefixes);

    std::vector<sz_string_view_t> views(lanes_count);
    for (sz_size_t lane = 0; lane != lanes_count; ++lane) views[lane] = {messages[lane].data(), messages[lane].size()};
    pinned_strings_t const staged = staged_(runtime, views);
    sz_strings_view_t const texts_strings = uploaded_(runtime, staged.view().raw);
    pinned_vector<sz_sha256_state_t> states(lanes_count, pinned_alloc_<sz_sha256_state_t>(runtime));
    pinned_vector<sz_u8_t> digests(lanes_count * STRINGZILLA_SHA256_DIGEST_LENGTH, pinned_alloc_<sz_u8_t>(runtime));
    std::vector<sz_sha256_state_t> expected(lanes_count);
    std::vector<sz_u8_t> expected_digests(lanes_count * STRINGZILLA_SHA256_DIGEST_LENGTH);
    for (sz_size_t lane = 0; lane != lanes_count; ++lane) {
        expect_equal(sz_sha256_state_init_serial(&expected[lane], nullptr), sz_success_k);
        if (lane % 2)
            expect_equal(
                sz_sha256_state_update_serial(&expected[lane], prefixes[lane].data(), prefixes[lane].size(), nullptr),
                sz_success_k);
        states[lane] = expected[lane];
    }

    expect_equal(backend.sha256_multistate_update(states.data(), &texts_strings, runtime.stream), sz_success_k);
    expect_equal(backend.sha256_multistate_digest(states.data(), lanes_count, digests.data(), runtime.stream),
                 sz_success_k);
    expect_equal(sz_stream_synchronize_best(runtime.capabilities, runtime.stream), sz_success_k);
    for (sz_size_t lane = 0; lane != lanes_count; ++lane)
        expect_equal(
            sz_sha256_state_update_serial(&expected[lane], messages[lane].data(), messages[lane].size(), nullptr),
            sz_success_k);
    expect_equal(sz_sha256_multistate_digest_serial(expected.data(), lanes_count, expected_digests.data(), nullptr),
                 sz_success_k);
    expect_equal(std::memcmp(digests.data(), expected_digests.data(), expected_digests.size()), 0,
                 "Device multi-state SHA-256 disagreed with serial");
}

/** An empty tape is a no-op, and strings the device cannot address are refused, not read. */
inline void check_sha256_multistate_device_safety_(auto const &backend) {
    sz_stream_t const stream = backend.runtime.stream;
    sz_strings_view_t const empty {};
    expect_equal(backend.sha256_multistate_update(nullptr, &empty, stream), sz_success_k);
    expect_equal(backend.sha256_multistate_digest(nullptr, 0, nullptr, stream), sz_success_k);

    std::array<sz_string_view_t, 2> const views {sz_string_view_t {"abc", 3}, {"", 0}};
    sz_strings_view_t host {};
    sz_strings_from_string_views(views.data(), views.size(), &host);
    std::array<sz_sha256_state_t, 2> states {};
    std::array<sz_u8_t, 2 * STRINGZILLA_SHA256_DIGEST_LENGTH> digests {};
    expect_equal(backend.sha256_multistate_update(states.data(), &host, stream), sz_device_memory_mismatch_k);
    expect_equal(backend.sha256_multistate_digest(states.data(), states.size(), digests.data(), stream),
                 sz_device_memory_mismatch_k);
}

/** Registers every check of one vendor's kernels, or of the dispatch points, behind @p gate . */
inline void check_device_kernels_(capability_gate_t const &gate, auto const &backend) {
    std::string const suffix = backend.name;
    check_device_levenshtein_(gate, backend);
    check_device_ngrams_(gate, backend);
    check_device_substrings_(gate, backend);
    check_device_regex_(gate, backend);
    check_device_subdocuments_(gate, backend);
    run_test_case(("check_utf8_uncased_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_utf8_uncased_device_equivalence_(generator, backend);
                  }));
    run_test_case(("check_utf8_uncased_device_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_utf8_uncased_device_safety_(generator, backend);
                  }));
    run_test_case(("check_utf8_norm_device_equivalence_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_utf8_norm_device_equivalence_(generator, backend);
                  }));
    run_test_case(("check_utf8_norm_device_safety_(" + suffix + ")").c_str(), gate,
                  rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                      check_utf8_norm_device_safety_(generator, backend);
                  }));
    if constexpr (!std::is_null_pointer_v<decltype(backend.sha256_multistate_update)>) {
        run_test_case(("check_sha256_multistate_device_equivalence_(" + suffix + ")").c_str(), gate,
                      rewound_(backend.runtime, [&](split_mix_generator_t &generator) {
                          check_sha256_multistate_device_equivalence_(generator, backend);
                      }));
        run_test_case(("check_sha256_multistate_device_safety_(" + suffix + ")").c_str(), gate,
                      rewound_(backend.runtime, [&] { check_sha256_multistate_device_safety_(backend); }));
    }
}

/** The dispatching entry points, over the capabilities of the device the checks launch on, and the
 *  refusals and asynchrony only a dispatch point's engine init promises. The SHA-256 pair is null
 *  where the vendor has no multi-state SHA-256, as in its own @ref device_kernels. */
inline void test_cross_dispatch_device(device_backend_t const &runtime, auto sha256_multistate_update,
                                       auto sha256_multistate_digest) {
    sz_capability_t const capabilities = runtime.capabilities;
    // Only a group that has the kernel gets a dispatch point.
    auto const offered = [capabilities](sz_kernel_kind_t kind) {
        sz_kernel_punned_t kernel = nullptr;
        sz_capability_t capability = 0;
        return sz_find_kernel_punned(kind, capabilities, &kernel, &capability) == sz_success_k;
    };
    device_kernels const dispatched {"dispatched",
                                     runtime,
                                     sz_levenshtein_distances,
                                     sz_ngrams_overlaps,
                                     sz_ngrams_sketches,
                                     offered(sz_kernel_substrings_contains_k) ? sz_substrings_contains : nullptr,
                                     sz_substrings_counts,
                                     sz_substrings_find,
                                     sz_substrings_replace,
                                     sz_substrings_bm25_scores,
                                     offered(sz_kernel_regex_contains_k) ? sz_regex_contains : nullptr,
                                     sz_regex_counts,
                                     sz_regex_find,
                                     sz_regex_replace,
                                     sz_regex_extract,
                                     offered(sz_kernel_subdocuments_contains_k) ? sz_subdocuments_contains : nullptr,
                                     sz_subdocuments_counts,
                                     sz_subdocuments_find,
                                     gpu_best<sz_utf8_uncased_fold_best>(capabilities),
                                     gpu_best<sz_utf8_norm_best>(capabilities),
                                     sha256_multistate_update,
                                     sha256_multistate_digest};
    capability_gate_t const gate(runtime.capabilities, runtime.capabilities);
    check_device_kernels_(gate, dispatched);
    run_test_case("check_levenshtein_device_threads_(dispatched)", gate,
                  rewound_(runtime, [&] { check_levenshtein_device_threads_(dispatched); }));
    run_test_case("check_levenshtein_device_deferred_free_(dispatched)", gate,
                  rewound_(runtime, [&] { check_levenshtein_device_deferred_free_(dispatched); }));
    run_test_case("check_ngrams_overlaps_device_streams_(dispatched)", gate,
                  rewound_(runtime, [&](split_mix_generator_t &generator) {
                      check_ngrams_overlaps_device_streams_(generator, dispatched);
                  }));
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test

#endif // STRINGZILLA_TEST_CROSS_DEVICE_HPP_
