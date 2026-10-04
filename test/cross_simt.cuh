/**
 *  @file test/cross_simt.cuh
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine checks - the device backend and dispatch scenario CUDA and ROCm share.
 *
 *  Included by one translation unit per binary, `cross_cuda.cu` or `cross_rocm.hip`, which names
 *  its vendor's kernels in a @c simt_backend_t. The checks reach the runtime only through the
 *  library and the vendor helpers `harness.hpp` picks, so both vendors run them unchanged.
 *
 *  These are the cases a host translation unit cannot express: device-reachable memory, a
 *  device-bound sequence, a caller's own stream, and the refusals that keep a host pointer from
 *  reaching a kernel as an address. The serial kernels are the oracle, as the CPU suites already
 *  measure them against brute force, so what is open here is whether a device round agrees.
 *
 *  Levenshtein query lengths are swept from a table whose status at each length is derived from
 *  the widest rung that length's alphabet reaches, so a length past it is expected to be refused
 *  and a raised constant turns that row into an answer without a line changing here.
 */
#pragma once
#ifndef STRINGZILLA_TEST_CROSS_SIMT_CUH
#define STRINGZILLA_TEST_CROSS_SIMT_CUH

#include <cmath>   // `std::fabs`, `std::exp`, `std::log`
#include <cstddef> // `std::size_t`

#include <algorithm> // `std::sort`, `std::copy`, `std::min`
#include <array>     // `std::array`
#include <random>    // `std::mt19937`, `std::uniform_real_distribution`
#include <span>      // `std::span`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp" // `cross_section_t`, `unified_vector`, `verify`
#include "cross.hpp"   // `check_uncased_fold_equivalence_`, `for_each_adversarial_utf8_input_`

namespace ashvardanian::stringzilla::test {

/** One vendor's engine kernels, or the dispatch points in their place, which take the same
 *  arguments: a device engine records its capability, so a verb reaches that vendor's kernel. */
struct simt_backend_t {

    /** The spelling @ref fail_backend_ and the test names carry. */
    char const *name;

    sz_kernel_levenshtein_distances_t levenshtein_distances;
    sz_kernel_levenshtein_distance_tiled_t levenshtein_distance_tiled;
    sz_kernel_overlap_scores_t overlap_scores;
    sz_kernel_substrings_counts_t substrings_counts;
    sz_kernel_substrings_find_t substrings_find;
    sz_kernel_substrings_replace_t substrings_replace;
    sz_kernel_substrings_bm25_scores_t substrings_bm25_scores;
    sz_kernel_utf8_uncased_fold_t utf8_uncased_fold;
    sz_kernel_utf8_norm_t utf8_norm;
};

/** The dispatch point @p best_ in the shape of its capability kernels, over the capabilities of the
 *  device the checks launch on: callable like them, and convertible to their function pointers. */
template <auto best_>
inline constexpr auto gpu_best =
    [](auto... arguments) noexcept { return call_best<best_>(gpu_capabilities(), arguments...); };

/** Joins @p stream, or the default one, which is what every device verb leaves the caller to do. */
static void join_(void *stream = nullptr) {
    verify(sz_stream_synchronize_best(gpu_capabilities(), stream) == sz_success_k);
}

/** How the candidate lengths of a skewed batch differ by orders of magnitude. */
enum class simt_skew_t {

    /** One candidate of three mebibytes, wherever the draw puts it, among a hundred thousand of at
     *  most thirty-two bytes. */
    lone_long_k,

    /** A hundred and twenty-eight drawn log-uniform up to a mebibyte. */
    log_uniform_k,
};

/** Candidate lengths of one @p skew. A round that hands one thread one candidate costs its longest,
 *  so these are the batches a balanced round exists for. */
static std::vector<std::size_t> simt_skewed_lengths_(std::mt19937 &generator, simt_skew_t skew) {
    if (skew == simt_skew_t::lone_long_k) {
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

#pragma region Levenshtein Helpers

/** What a corpus symbol is: a byte the tables class directly, or a rune the page table classes. */
enum class levenshtein_simt_alphabet_t { bytes_k, runes_k };

/** The alphabet enumerator an engine is prepared over, matching the corpus alphabet of a check. */
static sz_levenshtein_symbol_t levenshtein_simt_symbol_(levenshtein_simt_alphabet_t alphabet) {
    return alphabet == levenshtein_simt_alphabet_t::runes_k ? sz_levenshtein_runes_k : sz_levenshtein_bytes_k;
}

/** Corpus symbols: every byte value, or runes across four scripts and all three encoded widths. */
static std::vector<std::string> levenshtein_simt_symbols_(levenshtein_simt_alphabet_t alphabet) {
    std::vector<std::string> symbols;
    if (alphabet == levenshtein_simt_alphabet_t::bytes_k) {
        for (unsigned value = 0; value != 256; ++value) symbols.push_back(std::string(1, (char)value));
        return symbols;
    }
    char const *const runes[] = {"a", "z",  "é",  "ß",  "α",          "ω",          "ж",
                                 "я", "中", "漢", "字", "\U0001f600", "\U0001f30d", "\U0001d11e"};
    for (char const *const rune : runes) symbols.push_back(std::string(rune));
    return symbols;
}

/**
 *  @brief A corpus both sides address: an arena of near-duplicates, its views, and the matrix.
 *
 *  The candidates are edits of the queries, so the distances are small and every word of the Myers
 *  state moves. Unified storage is readable from the host, so the serial reference runs against
 *  these very bytes and the device builder reads them where they already live. Edits land on whole
 *  symbols, so a rune corpus stays well-formed and the engine is told its exact symbol count.
 */
struct levenshtein_simt_corpus_t {

    /** Every query's bytes, back to back. */
    unified_vector<char> query_arena;

    /** One view per query; its size is the query count. */
    unified_vector<sz_string_view_t> query_views;

    /** Every candidate's bytes, back to back. */
    unified_vector<char> arena;

    /** One view per candidate; its size is the candidate count. */
    unified_vector<sz_string_view_t> views;

    /** @b [queries, candidates], written by whichever round ran. */
    unified_vector<sz_size_t> distances;

    /** Host accessors over the queries, as an init requires. */
    sz_sequence_t queries {};

    /** The candidates copied into one tape, which every device round reads. */
    gpu_tape_t device_tape;

    /** The tape's accessors, which a kernel calls, as the scoring verb requires. */
    sz_sequence_t device_candidates {};

    /** Accessors the serial reference calls, over the same views. */
    sz_sequence_t host_candidates {};

    levenshtein_simt_corpus_t(std::size_t count, std::size_t query_symbols, std::size_t query_count,
                              levenshtein_simt_alphabet_t alphabet)
        : query_views(query_count), views(count), distances(query_count * count) {
        std::vector<std::string> const symbols = levenshtein_simt_symbols_(alphabet);
        std::vector<std::string> const drawn = fill_queries_(symbols, query_symbols);
        arena.reserve(count * query_symbols * 4);
        std::vector<std::string> edited;
        for (std::size_t index = 0; index != count; ++index) {
            edited = drawn;
            for (std::size_t edit = 0, edits = index % 17; edit != edits && edited.size() > 1; ++edit) {
                std::size_t const at = (index * 31 + edit * 7) % edited.size();
                if (edit % 2) { edited.erase(edited.begin() + (std::ptrdiff_t)at); }
                else { edited[at] = symbols[(edit + index) % symbols.size()]; }
            }
            std::size_t length = 0;
            for (std::string const &symbol : edited) length += symbol.size();
            views[index].length = length;
            for (std::string const &symbol : edited) arena.insert(arena.end(), symbol.begin(), symbol.end());
        }
        bind_();
    }

    /** Candidates of @p candidate_symbols symbols each, one in four of them drawn from the queries'
     *  own, so a distance moves well below the longer side's length. */
    levenshtein_simt_corpus_t(std::mt19937 &generator, std::vector<std::size_t> const &candidate_symbols,
                              std::size_t query_symbols, std::size_t query_count, levenshtein_simt_alphabet_t alphabet)
        : query_views(query_count), views(candidate_symbols.size()), distances(query_count * candidate_symbols.size()) {
        std::vector<std::string> const symbols = levenshtein_simt_symbols_(alphabet);
        std::vector<std::string> const drawn = fill_queries_(symbols, query_symbols);
        std::size_t total = 0;
        for (std::size_t const count : candidate_symbols) total += count;
        arena.reserve(total * 4);
        for (std::size_t index = 0; index != candidate_symbols.size(); ++index) {
            std::size_t const before = arena.size();
            for (std::size_t symbol = 0; symbol != candidate_symbols[index]; ++symbol) {
                std::string const &picked = generator() % 4 == 0 ? drawn[generator() % drawn.size()]
                                                                 : symbols[generator() % symbols.size()];
                arena.insert(arena.end(), picked.begin(), picked.end());
            }
            views[index].length = arena.size() - before;
        }
        bind_();
    }

    /** Fills the queries, every one as long as @p query_symbols and every one different, and
     *  returns the symbols they were edited from. */
    std::vector<std::string> fill_queries_(std::vector<std::string> const &symbols, std::size_t query_symbols) {
        std::vector<std::string> drawn(query_symbols);
        for (std::size_t symbol = 0; symbol != query_symbols; ++symbol)
            drawn[symbol] = symbols[(symbol * 37 + 11) % symbols.size()];
        // Every query is the same length, so one batch still spans one rung, and every one differs.
        for (std::size_t query = 0; query != query_views.size(); ++query) {
            std::vector<std::string> edited = drawn;
            for (std::size_t symbol = query; symbol < edited.size(); symbol += query_views.size() + 1)
                edited[symbol] = symbols[(symbol + query * 5 + 3) % symbols.size()];
            std::size_t length = 0;
            for (std::string const &symbol : edited) length += symbol.size();
            query_views[query].length = length;
            for (std::string const &symbol : edited)
                query_arena.insert(query_arena.end(), symbol.begin(), symbol.end());
        }
        return drawn;
    }

    /** Points the views into the arenas and binds the sequences, once neither arena grows. */
    void bind_() {
        std::size_t written = 0;
        for (sz_string_view_t &view : query_views) view.start = query_arena.data() + written, written += view.length;
        written = 0;
        for (sz_string_view_t &view : views) view.start = arena.data() + written, written += view.length;
        sz_sequence_from_string_views(query_views.data(), query_views.size(), &queries);
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
        device_tape.copy(host_candidates);
        device_candidates = device_tape.sequence;
    }

    /** Candidates one round scores, which is also the row stride of the matrix it writes. */
    std::size_t count() const { return views.size(); }
};

/** The serial backend's answers for the same corpus, read off the very bytes the device reads. */
static std::vector<sz_size_t> levenshtein_serial_reference_(levenshtein_simt_corpus_t const &corpus,
                                                            sz_levenshtein_symbol_t symbol) {
    handle_checked_heap_t heap;
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init_serial(&engine, &corpus.queries, symbol, &heap.allocator, nullptr) ==
           sz_success_k);
    std::vector<sz_size_t> expected(corpus.distances.size());
    verify(sz_levenshtein_distances_serial(&engine, &corpus.host_candidates, expected.data(), corpus.count(),
                                           nullptr) == sz_success_k);
    sz_levenshtein_engine_free(&engine, nullptr);
    verify(heap.live_allocations == 0);
    return expected;
}

#pragma endregion Levenshtein Helpers

#pragma region Levenshtein Expectations

/**
 *  @brief The status a query of @p query_symbols symbols draws, as the widest rung decides it.
 *
 *  Both alphabets reach the warped rung, whose thirty-two lanes hold
 *  @ref sz_levenshtein_gpu_words_max_k words between them; a rune query gets there by riding its
 *  class down the lanes rather than by indexing the candidate.
 */
static constexpr sz_status_t levenshtein_simt_expected_status_(std::size_t query_symbols) {
    return query_symbols <= (std::size_t)sz_levenshtein_gpu_words_max_k * 64 ? sz_success_k
                                                                             : sz_unexpected_dimensions_k;
}

/** Symbol-pairs one sweep scores before it drops the widest batch, so the serial reference stays
 *  quadratic in time but bounded in it as the rungs widen. */
enum { levenshtein_simt_sweep_budget_k = 4u * 1024u * 1024u };

/** Queries one sweep prepares at a time, so @c grid.y carries an axis wider than one per rung. */
enum { levenshtein_simt_sweep_queries_k = 3 };

/**
 *  @brief Every query length the sweeps walk, either side of each boundary a rung is cut at.
 *
 *  The narrow rungs share one register between candidates, four ways through eight symbols and two
 *  ways through sixteen, so the symbols either side of 8 and 16. The threaded rung cuts at every
 *  word through its sixteenth, where the warped rung takes over, so the bytes either side of 64,
 *  128 and 1024. The warped rung hands its lanes one more word each at every thirty-second word
 *  past that, so the bytes either side of every multiple of 2048 up to the ceiling itself.
 */
static constexpr std::size_t levenshtein_simt_query_symbols_k[] = {
    1,    2,    4,    8,    9,    16,   17,   63,   64,    65,    127,   128,   129,   1023,  1024,  1025,  2047,  2048,
    2049, 4095, 4096, 4097, 6144, 6145, 8192, 8193, 10240, 10241, 12288, 12289, 14336, 14337, 16383, 16384, 16385,
};

#pragma endregion Levenshtein Expectations

#pragma region Levenshtein Checks

/** One backend's distances against serial's, on a corpus that never leaves the device. */
static void check_levenshtein_simt_equivalence_(char const *name, levenshtein_simt_alphabet_t alphabet,
                                                sz_kernel_levenshtein_distances_t device) {
    sz_levenshtein_symbol_t const symbol = levenshtein_simt_symbol_(alphabet);
    for (std::size_t const query_symbols : levenshtein_simt_query_symbols_k)
        for (std::size_t count : {1u, 129u, 2048u}) {
            std::size_t const queries = levenshtein_simt_sweep_queries_k;
            if (count * query_symbols * queries > (std::size_t)levenshtein_simt_sweep_budget_k) continue;
            levenshtein_simt_corpus_t corpus(count, query_symbols, queries, alphabet);

            sz_levenshtein_engine_t engine {};
            sz_status_t const prepared = sz_levenshtein_engine_init(
                &engine, &corpus.queries, symbol, gpu_capabilities(), STRINGZILLA_NULL, STRINGZILLA_NULL);
            if (prepared != levenshtein_simt_expected_status_(query_symbols))
                fail_backend_(name, "a device batch drew a status the rungs do not imply");
            if (prepared != sz_success_k) continue;
            sz_status_t const produced = device(&engine, &corpus.device_candidates, corpus.distances.data(),
                                                corpus.count(), STRINGZILLA_NULL);
            // The planes the round reads live in the engine's block, so the join comes before
            // it is released.
            join_();
            sz_levenshtein_engine_free(&engine, nullptr);
            if (produced != sz_success_k) fail_backend_(name, "a device-resident round was refused");

            std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus, symbol);
            for (std::size_t index = 0; index != expected.size(); ++index)
                if (corpus.distances[index] != expected[index]) fail_backend_(name, "a distance differs from serial");
        }
}

/**
 *  @brief The narrow rungs against serial, on a batch wide enough for the dispatch to reach them.
 *
 *  Four candidates share a byte-laned register and two share a short-laned one, so their grid is
 *  that many times narrower than the threaded rung's over one batch, and the dispatch keeps them
 *  for batches that still fill the device. That width is past anything the length sweep builds,
 *  which is why they are checked here, over candidates that run past the eight bytes one refill
 *  hands a lane and start at every alignment.
 */
static void check_levenshtein_simt_narrow_lanes_(std::mt19937 &generator, char const *name,
                                                 sz_kernel_levenshtein_distances_t device) {
    enum { arena_bytes_k = 256, longest_candidate_k = 33, offsets_k = 97 };
    handle_checked_heap_t heap;

    std::size_t const count = gpu_multiprocessors() * gpu_threads_per_multiprocessor() *
                              sz_levenshtein_gpu_lanes_waves_min_k;
    if (count == 0) return;
    for (std::size_t const query_symbols : {(std::size_t)8, (std::size_t)16}) {

        // Every third byte is one of the query's own, so a candidate draws both classed
        // and absent symbols.
        unified_vector<char> query(query_symbols), arena(arena_bytes_k);
        randomize_string(generator, query);
        randomize_string(generator, arena);
        for (std::size_t index = 0; index < arena.size(); index += 3) arena[index] = query[index % query.size()];

        unified_vector<sz_string_view_t> query_views(1), views(count);
        unified_vector<sz_size_t> distances(count);
        query_views[0].start = query.data(), query_views[0].length = query.size();
        for (std::size_t index = 0; index != count; ++index)
            views[index].start = arena.data() + index % offsets_k, views[index].length = index % longest_candidate_k;
        sz_sequence_t queries {}, host_candidates {};
        sz_sequence_from_string_views(query_views.data(), query_views.size(), &queries);
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
        gpu_tape_t candidates;
        candidates.copy(host_candidates);

        sz_levenshtein_engine_t engine {};
        verify(sz_levenshtein_engine_init(&engine, &queries, sz_levenshtein_bytes_k, gpu_capabilities(),
                                          STRINGZILLA_NULL, STRINGZILLA_NULL) == sz_success_k);
        sz_status_t const produced = device(&engine, &candidates.sequence, distances.data(), count, STRINGZILLA_NULL);
        // The planes the round reads live in the engine's block, so the join comes before
        // it is released.
        join_();
        sz_levenshtein_engine_free(&engine, nullptr);
        if (produced != sz_success_k) fail_backend_(name, "a batch wide enough for the narrow rungs was refused");

        sz_levenshtein_engine_t host_engine {};
        verify(sz_levenshtein_engine_init_serial(&host_engine, &queries, sz_levenshtein_bytes_k, &heap.allocator,
                                                 nullptr) == sz_success_k);
        std::vector<sz_size_t> expected(count);
        verify(sz_levenshtein_distances_serial(&host_engine, &host_candidates, expected.data(), count, nullptr) ==
               sz_success_k);
        sz_levenshtein_engine_free(&host_engine, nullptr);
        for (std::size_t index = 0; index != count; ++index)
            if (distances[index] != expected[index])
                fail_backend_(name, "a narrow rung's distance differs from serial");
    }
    verify(heap.live_allocations == 0);
}

/** Symbol-words a skewed batch scores at one query width before that width is skipped, so the
 *  serial reference stays within seconds. */
enum { levenshtein_simt_skewed_budget_k = 256u * 1024u * 1024u };

/** Skewed batches against serial, at query widths the threaded rung and the warped one take. */
static void check_levenshtein_simt_skewed_(std::mt19937 &generator, char const *name,
                                           levenshtein_simt_alphabet_t alphabet,
                                           sz_kernel_levenshtein_distances_t device) {
    sz_levenshtein_symbol_t const symbol = levenshtein_simt_symbol_(alphabet);
    for (simt_skew_t const skew : {simt_skew_t::lone_long_k, simt_skew_t::log_uniform_k}) {
        std::vector<std::size_t> const lengths = simt_skewed_lengths_(generator, skew);
        std::size_t total = 0;
        for (std::size_t const length : lengths) total += length;
        for (std::size_t const query_symbols : {(std::size_t)40, (std::size_t)600, (std::size_t)1500}) {
            if (total * sz_levenshtein_query_words(query_symbols) > (std::size_t)levenshtein_simt_skewed_budget_k)
                continue;
            levenshtein_simt_corpus_t corpus(generator, lengths, query_symbols, 2, alphabet);
            sz_levenshtein_engine_t engine {};
            verify(sz_levenshtein_engine_init(&engine, &corpus.queries, symbol, gpu_capabilities(), STRINGZILLA_NULL,
                                              STRINGZILLA_NULL) == sz_success_k);
            sz_status_t const produced = device(&engine, &corpus.device_candidates, corpus.distances.data(),
                                                corpus.count(), STRINGZILLA_NULL);
            join_();
            sz_levenshtein_engine_free(&engine, nullptr);
            if (produced != sz_success_k) fail_backend_(name, "a skewed batch was refused");

            std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus, symbol);
            for (std::size_t index = 0; index != expected.size(); ++index)
                if (corpus.distances[index] != expected[index])
                    fail_backend_(name, "a skewed batch's distance differs from serial");
        }
    }
}

/** The tiled wavefront against serial's answer at the same lengths, on the one pair it takes. */
static void check_levenshtein_simt_tiled_(char const *name, sz_kernel_levenshtein_distance_tiled_t tiled) {
    handle_checked_heap_t heap;
    unified_vector<sz_size_t> distance(1, 0);

    for (std::size_t const query_symbols : levenshtein_simt_query_symbols_k) {
        levenshtein_simt_corpus_t corpus(1, query_symbols, 1, levenshtein_simt_alphabet_t::bytes_k);
        sz_string_view_t const query = corpus.query_views[0], candidate = corpus.views[0];
        unified_vector<char> scratch(sz_levenshtein_distance_tiled_scratch_bytes(query.length, candidate.length));
        if (tiled(query.start, query.length, candidate.start, candidate.length, scratch.data(), distance.data(),
                  STRINGZILLA_NULL) != sz_success_k)
            fail_backend_(name, "the wavefront refused a device-resident pair");
        join_();

        sz_levenshtein_engine_t engine {};
        verify(sz_levenshtein_engine_init_serial(&engine, &corpus.queries, sz_levenshtein_bytes_k, &heap.allocator,
                                                 nullptr) == sz_success_k);
        sz_size_t expected = 0;
        verify(sz_levenshtein_distances_serial(&engine, &corpus.host_candidates, &expected, 1, nullptr) ==
               sz_success_k);
        sz_levenshtein_engine_free(&engine, nullptr);
        if (distance[0] != expected) fail_backend_(name, "a wavefront distance differs from serial");
    }

    // An empty text has nothing to tile, so its distance is the other's length, stored by a kernel
    // of its own.
    levenshtein_simt_corpus_t corpus(1, 100, 1, levenshtein_simt_alphabet_t::bytes_k);
    sz_string_view_t const query = corpus.query_views[0];
    if (tiled(query.start, query.length, query.start, 0, STRINGZILLA_NULL, distance.data(), STRINGZILLA_NULL) !=
        sz_success_k)
        fail_backend_(name, "the wavefront refused an empty text");
    join_();
    if (distance[0] != query.length) fail_backend_(name, "an empty text's distance is not the other's length");
    verify(heap.live_allocations == 0);
}

/** One backend refusing host memory, sequence handle and outputs alike, rather than staging it. */
static void check_levenshtein_simt_memory_safety_(std::mt19937 &generator, simt_backend_t const &backend) {
    std::string arena(660, '\0');
    randomize_string(generator, arena);
    std::array<sz_string_view_t, 4> const views {sz_string_view_t {arena.data(), 1},
                                                 {arena.data() + 1, 60},
                                                 {arena.data() + 61, 199},
                                                 {arena.data() + 260, 400}};
    sz_sequence_t candidates {};
    sz_sequence_from_string_views(views.data(), views.size(), &candidates);

    levenshtein_simt_corpus_t corpus(1, 200, 1, levenshtein_simt_alphabet_t::bytes_k);
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init(&engine, &corpus.queries, sz_levenshtein_bytes_k, gpu_capabilities(),
                                      STRINGZILLA_NULL, STRINGZILLA_NULL) == sz_success_k);
    std::vector<sz_size_t> produced(views.size());
    if (backend.levenshtein_distances(&engine, &candidates, produced.data(), views.size(), STRINGZILLA_NULL) !=
        sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host memory reached a kernel as an address");
    sz_levenshtein_engine_free(&engine, nullptr);
}

/** The builder refusing a query past what the widest rung holds, in bytes and again in runes. */
static void check_levenshtein_simt_query_safety_() {
    enum { symbols_k = sz_levenshtein_gpu_words_max_k * 64 + 1 };
    for (levenshtein_simt_alphabet_t const alphabet :
         {levenshtein_simt_alphabet_t::bytes_k, levenshtein_simt_alphabet_t::runes_k}) {
        levenshtein_simt_corpus_t corpus(1, symbols_k, 1, alphabet);
        sz_levenshtein_engine_t engine {};
        if (sz_levenshtein_engine_init(&engine, &corpus.queries, levenshtein_simt_symbol_(alphabet), gpu_capabilities(),
                                       STRINGZILLA_NULL, STRINGZILLA_NULL) != sz_unexpected_dimensions_k)
            fail_backend_("dispatched", "a query past the verticals was not refused");
        if (engine.memory != STRINGZILLA_NULL) fail_backend_("dispatched", "a refused build still kept a block");
    }
}

/** An empty query has no last word to read a score off, so a batch holding one is refused. */
static void check_levenshtein_simt_empty_query_safety_() {
    unified_vector<sz_string_view_t> query_views(1);
    query_views[0].start = STRINGZILLA_NULL, query_views[0].length = 0;
    sz_sequence_t queries {};
    sz_sequence_from_string_views(query_views.data(), query_views.size(), &queries);
    sz_levenshtein_engine_t engine {};
    if (sz_levenshtein_engine_init(&engine, &queries, sz_levenshtein_bytes_k, gpu_capabilities(), STRINGZILLA_NULL,
                                   STRINGZILLA_NULL) != sz_unexpected_dimensions_k)
        fail_backend_("dispatched", "an empty query was not refused");
}

/** Device work queued ahead of the verb, so a stream draining after it returns is no accident. */
enum { levenshtein_simt_filler_bytes_k = 128u * 1024u * 1024u, levenshtein_simt_filler_passes_k = 16 };

/** Candidates and query symbols the asynchrony check scores, enough to outlast one launch alone. */
enum { levenshtein_simt_scheduled_candidates_k = 2048, levenshtein_simt_scheduled_symbols_k = 1024 };

/**
 *  @brief The scoring verb returning before its round completes, on a stream the caller owns.
 *
 *  Work is queued on that stream first, so it is busy when the verb is called: a verb that joined
 *  would drain the filler too, and the query right after it would report the stream idle. Nothing
 *  is asserted about the init, which is allowed to join. The distances are checked once the stream
 *  does drain, so the round that ran asynchronously is the round the equivalence sweep would run.
 */
static void check_levenshtein_simt_scheduled_asynchrony_() {
    levenshtein_simt_corpus_t corpus(levenshtein_simt_scheduled_candidates_k, levenshtein_simt_scheduled_symbols_k,
                                     levenshtein_simt_sweep_queries_k, levenshtein_simt_alphabet_t::bytes_k);

    gpu_stream_t const stream;
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init(&engine, &corpus.queries, sz_levenshtein_bytes_k, gpu_capabilities(),
                                      STRINGZILLA_NULL, STRINGZILLA_NULL) == sz_success_k);

    void *const filler = gpu_allocate_device(levenshtein_simt_filler_bytes_k, nullptr, stream.handle);
    verify(filler != nullptr);
    for (int pass = 0; pass != levenshtein_simt_filler_passes_k; ++pass)
        verify(gpu_fill(filler, levenshtein_simt_filler_bytes_k, (sz_u8_t)pass, stream.handle) == sz_success_k);

    verify(sz_levenshtein_distances(&engine, &corpus.device_candidates, corpus.distances.data(), corpus.count(),
                                    stream.handle) == sz_success_k);
    if (!gpu_stream_query(stream.handle))
        fail_backend_("dispatched", "the scoring verb drained the stream it was handed");

    join_(stream.handle);
    gpu_free_device(filler, levenshtein_simt_filler_bytes_k, nullptr, stream.handle);
    sz_levenshtein_engine_free(&engine, nullptr);

    std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus, sz_levenshtein_bytes_k);
    for (std::size_t index = 0; index != expected.size(); ++index)
        if (corpus.distances[index] != expected[index])
            fail_backend_("dispatched", "a scheduled round's distance differs from serial");
}

#pragma endregion Levenshtein Checks

#pragma region Overlap Helpers

/**
 *  @brief A corpus both sides address: one arena, its views, and room for one round's scores.
 *
 *  Unified storage is readable from the host, so the serial reference runs against these very bytes
 *  rather than a second copy of them, and the two candidate sequences differ only in whose
 *  accessors they carry. The queries stay plain host strings, because an engine's builder reads
 *  them on the host and no kernel ever sees them.
 */
struct overlap_simt_corpus_t {

    /** The texts whose windows are sorted into the forest. */
    std::vector<std::string> queries;

    /** Every candidate's bytes, back to back. */
    unified_vector<char> arena;

    /** One view per candidate; its size is the candidate count. */
    unified_vector<sz_string_view_t> views;

    /** @b [queries,candidates,widths], written by whichever verb ran. */
    unified_vector<sz_f32_t> scores;

    /** Host accessors, which is what an engine's builder calls. */
    sz_sequence_t query_sequence {};

    /** The candidates copied into one tape, which every device round reads. */
    gpu_tape_t device_tape;

    /** The tape's accessors, which a kernel calls, as the scoring verb requires. */
    sz_sequence_t device_candidates {};

    /** Accessors the serial reference calls, over the same views. */
    sz_sequence_t host_candidates {};

    overlap_simt_corpus_t(std::mt19937 &generator, std::size_t queries_count, std::size_t count,
                          std::size_t query_length, std::size_t widths_count)
        : views(count), scores(queries_count * count * widths_count) {
        fill_queries_(generator, queries_count, query_length);
        for (std::size_t index = 0; index != count; ++index) arena.resize(arena.size() + 1 + (index * 37) % 900);
        randomize_string(generator, arena);
        std::vector<std::size_t> lengths(count);
        for (std::size_t index = 0; index != count; ++index) lengths[index] = 1 + (index * 37) % 900;
        bind_(lengths);
    }

    /** Candidates of @p lengths bytes, spliced from spans of the queries between runs of random
     *  bytes, so windows match across wherever a kernel cuts a candidate. */
    overlap_simt_corpus_t(std::mt19937 &generator, std::size_t queries_count, std::vector<std::size_t> const &lengths,
                          std::size_t query_length, std::size_t widths_count)
        : views(lengths.size()), scores(queries_count * lengths.size() * widths_count) {
        fill_queries_(generator, queries_count, query_length);
        std::size_t total = 0;
        for (std::size_t const length : lengths) total += length;
        arena.resize(total);
        randomize_string(generator, arena);
        for (std::size_t at = 0; at < total;) {
            std::string const &query = queries[generator() % queries.size()];
            std::size_t const span = std::min({(std::size_t)(1 + generator() % 48), query.size(), total - at});
            std::size_t const from = generator() % (query.size() - span + 1);
            std::copy(query.begin() + (std::ptrdiff_t)from, query.begin() + (std::ptrdiff_t)(from + span),
                      arena.begin() + (std::ptrdiff_t)at);
            at += span + generator() % 9;
        }
        bind_(lengths);
    }

    /** Draws the query texts, each a different length below @p query_length. */
    void fill_queries_(std::mt19937 &generator, std::size_t queries_count, std::size_t query_length) {
        for (std::size_t index = 0; index != queries_count; ++index) {
            std::string text(query_length ? query_length - index % query_length : 0, '\0');
            randomize_string(generator, text);
            queries.push_back(text);
        }
    }

    /** Cuts the arena into candidates of @p lengths bytes and binds the sequences over them. */
    void bind_(std::vector<std::size_t> const &lengths) {
        std::size_t written = 0;
        for (std::size_t index = 0; index != lengths.size(); ++index) {
            views[index].start = arena.data() + written, views[index].length = lengths[index];
            written += lengths[index];
        }
        query_sequence = sequence_from_(queries);
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
        device_tape.copy(host_candidates);
        device_candidates = device_tape.sequence;
    }

    std::size_t candidate_stride() const noexcept { return scores.size() / (queries.size() * views.size()); }
    std::size_t query_stride() const noexcept { return views.size() * candidate_stride(); }
};

/** The serial backend's answers for the same corpus, read off the very bytes the device reads. */
static std::vector<sz_f32_t> overlap_serial_reference_(overlap_simt_corpus_t const &corpus,
                                                       std::span<sz_size_t const> widths) {
    handle_checked_heap_t heap;
    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init_serial(&engine, &corpus.query_sequence, widths.data(), widths.size(), 0,
                                         &heap.allocator, nullptr) == sz_success_k);
    std::vector<sz_f32_t> expected(corpus.scores.size(), -1.0f);
    verify(sz_overlap_scores_serial(&engine, &corpus.host_candidates, expected.data(), corpus.query_stride(),
                                    corpus.candidate_stride(), nullptr) == sz_success_k);
    sz_overlap_engine_free(&engine, nullptr);
    verify(heap.live_allocations == 0);
    return expected;
}

#pragma endregion Overlap Helpers

#pragma region Overlap Checks

/** One backend's scores against serial's, on a corpus that never leaves the device. */
static void check_overlap_simt_equivalence_(std::mt19937 &generator, simt_backend_t const &backend) {
    std::array<sz_size_t, 3> const widths {4, 6, 8};

    for (std::size_t queries_count : {1u, 3u})
        for (std::size_t count : {1u, 129u, 4096u})
            for (std::size_t query_length : {12u, 777u}) {
                overlap_simt_corpus_t corpus(generator, queries_count, count, query_length, widths.size());
                std::vector<sz_f32_t> const expected = overlap_serial_reference_(corpus,
                                                                                 {widths.data(), widths.size()});
                sz_overlap_engine_t engine {};
                if (sz_overlap_engine_init(&engine, &corpus.query_sequence, widths.data(), widths.size(), 0,
                                           gpu_capabilities(), STRINGZILLA_NULL, STRINGZILLA_NULL) != sz_success_k)
                    fail_backend_(backend.name, "a device engine was refused for a well-formed batch");
                if (backend.overlap_scores(&engine, &corpus.device_candidates, corpus.scores.data(),
                                           corpus.query_stride(), corpus.candidate_stride(),
                                           STRINGZILLA_NULL) != sz_success_k)
                    fail_backend_(backend.name, "a device-resident batch was refused");
                join_();
                sz_overlap_engine_free(&engine, nullptr);
                for (std::size_t slot = 0; slot != expected.size(); ++slot)
                    if (corpus.scores[slot] != expected[slot])
                        fail_backend_(backend.name, "a share differs from serial");
            }
}

/** Skewed batches against serial, at the narrowest window and at the widest the device takes. */
static void check_overlap_simt_skewed_(std::mt19937 &generator, simt_backend_t const &backend) {
    std::array<sz_size_t, 3> const widths {3, 8, sz_overlap_gpu_widest_window_k};
    for (simt_skew_t const skew : {simt_skew_t::lone_long_k, simt_skew_t::log_uniform_k}) {
        std::vector<std::size_t> const lengths = simt_skewed_lengths_(generator, skew);
        overlap_simt_corpus_t corpus(generator, 2, lengths, 777, widths.size());
        std::vector<sz_f32_t> const expected = overlap_serial_reference_(corpus, {widths.data(), widths.size()});
        sz_overlap_engine_t engine {};
        if (sz_overlap_engine_init(&engine, &corpus.query_sequence, widths.data(), widths.size(), 0, gpu_capabilities(),
                                   STRINGZILLA_NULL, STRINGZILLA_NULL) != sz_success_k)
            fail_backend_(backend.name, "a device engine was refused for a well-formed batch");
        if (backend.overlap_scores(&engine, &corpus.device_candidates, corpus.scores.data(), corpus.query_stride(),
                                   corpus.candidate_stride(), STRINGZILLA_NULL) != sz_success_k)
            fail_backend_(backend.name, "a skewed batch was refused");
        join_();
        sz_overlap_engine_free(&engine, nullptr);
        for (std::size_t slot = 0; slot != expected.size(); ++slot)
            if (corpus.scores[slot] != expected[slot])
                fail_backend_(backend.name, "a skewed batch's share differs from serial");
    }
}

/** One backend refusing host memory no kernel can address, rather than reading a bad pointer. */
static void check_overlap_simt_memory_safety_(std::mt19937 &generator, simt_backend_t const &backend) {
    std::array<sz_size_t, 2> const widths {4, 6};
    overlap_simt_corpus_t corpus(generator, 1, 8, 333, widths.size());
    sz_overlap_engine_t engine {};
    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, widths.data(), widths.size(), 0, gpu_capabilities(),
                               STRINGZILLA_NULL, STRINGZILLA_NULL) != sz_success_k)
        fail_backend_(backend.name, "a device engine was refused for a well-formed batch");

    // The corpus's own views are unified, so a host-resident sequence needs its own plain storage.
    std::string text(64, '\0');
    randomize_string(generator, text);
    std::array<sz_string_view_t, 1> const host_views {sz_string_view_t {text.data(), text.size()}};
    sz_sequence_t host_candidates {};
    sz_sequence_from_string_views(host_views.data(), host_views.size(), &host_candidates);

    std::vector<sz_f32_t> host_scores(corpus.scores.size(), -1.0f);
    if (backend.overlap_scores(&engine, &corpus.device_candidates, host_scores.data(), corpus.query_stride(),
                               corpus.candidate_stride(), STRINGZILLA_NULL) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host scores were not refused");
    if (backend.overlap_scores(&engine, &host_candidates, corpus.scores.data(), corpus.query_stride(),
                               corpus.candidate_stride(), STRINGZILLA_NULL) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "a host-resident sequence handle was not refused");
    for (sz_f32_t const untouched : host_scores)
        if (untouched != -1.0f) fail_backend_(backend.name, "a refused call still wrote a score");
    sz_overlap_engine_free(&engine, nullptr);
}

/** The widest window a device engine takes, and the refusals one step past either bound. */
static void check_overlap_simt_width_safety_(std::mt19937 &generator, simt_backend_t const &backend) {
    std::array<sz_size_t, 1> const widest {sz_overlap_gpu_widest_window_k};
    std::array<sz_size_t, 1> const past {sz_overlap_gpu_widest_window_k + 1};
    std::array<sz_size_t, sz_overlap_gpu_widths_max_k + 1> too_many {};
    for (std::size_t index = 0; index != too_many.size(); ++index) too_many[index] = index + 1;

    // The one candidate is the query's own bytes in unified storage, so the text scores itself.
    overlap_simt_corpus_t corpus(generator, 1, 1, 256, widest.size());
    corpus.arena.assign(corpus.queries.front().begin(), corpus.queries.front().end());
    corpus.views[0].start = corpus.arena.data(), corpus.views[0].length = corpus.arena.size();
    gpu_tape_t itself;
    itself.copy(corpus.host_candidates);

    sz_overlap_engine_t engine {};
    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, past.data(), past.size(), 0, gpu_capabilities(),
                               STRINGZILLA_NULL, STRINGZILLA_NULL) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a width past the ring was not refused");
    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, too_many.data(), too_many.size(), 0, gpu_capabilities(),
                               STRINGZILLA_NULL, STRINGZILLA_NULL) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "more widths than the registers hold were not refused");
    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, widest.data(), widest.size(), 0, gpu_capabilities(),
                               STRINGZILLA_NULL, STRINGZILLA_NULL) != sz_success_k)
        fail_backend_(backend.name, "the widest window the ring holds was refused");
    if (backend.overlap_scores(&engine, &itself.sequence, corpus.scores.data(), corpus.query_stride(),
                               corpus.candidate_stride(), STRINGZILLA_NULL) != sz_success_k)
        fail_backend_(backend.name, "a device-resident batch was refused");
    join_();
    sz_overlap_engine_free(&engine, nullptr);
    if (corpus.scores[0] != 1.0f) fail_backend_(backend.name, "a text does not fully overlap itself");
}

/** The scoring verb enqueues on the stream it is handed and returns, so a round big enough to
 *  outlive the call is still running after it, and the engine keeps no stream, so a second one
 *  scores it at once. */
static void check_overlap_simt_asynchrony_(std::mt19937 &generator) {
    std::array<sz_size_t, 3> const widths {4, 6, 8};
    overlap_simt_corpus_t corpus(generator, 8, 4096, 777, widths.size());
    std::vector<sz_f32_t> const expected = overlap_serial_reference_(corpus, {widths.data(), widths.size()});
    unified_vector<sz_f32_t> second(corpus.scores.size(), -1.0f);
    gpu_stream_t const first_stream, second_stream;

    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init(&engine, &corpus.query_sequence, widths.data(), widths.size(), 0, gpu_capabilities(),
                                  STRINGZILLA_NULL, STRINGZILLA_NULL) == sz_success_k);
    verify(sz_overlap_scores(&engine, &corpus.device_candidates, corpus.scores.data(), corpus.query_stride(),
                             corpus.candidate_stride(), first_stream.handle) == sz_success_k);
    verify(sz_overlap_scores(&engine, &corpus.device_candidates, second.data(), corpus.query_stride(),
                             corpus.candidate_stride(), second_stream.handle) == sz_success_k);
    verify(gpu_stream_query(first_stream.handle) && "the scoring verb joined the stream it enqueued on");
    join_(first_stream.handle);
    join_(second_stream.handle);
    sz_overlap_engine_free(&engine, nullptr);
    for (std::size_t slot = 0; slot != expected.size(); ++slot)
        if (corpus.scores[slot] != expected[slot] || second[slot] != expected[slot])
            fail_backend_("dispatched", "two streams scoring one engine disagreed with serial");
}

#pragma endregion Overlap Checks

#pragma region Substrings Helpers

/**
 *  @brief A corpus both sides address: one arena of haystacks and the views into it.
 *
 *  Unified storage is readable from the host, so the serial reference runs against these very bytes
 *  and the device verbs accept them without anything being staged.
 */
struct substrings_simt_corpus_t {

    /** Every haystack's bytes, back to back. */
    unified_vector<char> arena;

    /** One view per haystack; its size is the haystack count. */
    unified_vector<sz_string_view_t> views;

    /** The haystacks copied into one tape, which every device verb reads. */
    gpu_tape_t device_tape;

    /** The tape's accessors, which a kernel calls, as the device verbs require. */
    sz_sequence_t device_haystacks {};

    /** Accessors the serial reference calls, over the same views. */
    sz_sequence_t host_haystacks {};

    substrings_simt_corpus_t(std::vector<std::string> const &haystacks) : views(haystacks.size()) {
        for (std::size_t index = 0; index != haystacks.size(); ++index) {
            views[index].length = haystacks[index].size();
            arena.insert(arena.end(), haystacks[index].begin(), haystacks[index].end());
        }
        // The arena's address is only final once it has stopped growing, so the starts
        // are filled afterwards.
        std::size_t written = 0;
        for (sz_string_view_t &view : views) view.start = arena.data() + written, written += view.length;
        sz_sequence_from_string_views(views.data(), views.size(), &host_haystacks);
        device_tape.copy(host_haystacks);
        device_haystacks = device_tape.sequence;
    }
};

/** A vocabulary both sides address, in memory a kernel can read. */
struct substrings_simt_vocabulary_t {

    /** Every needle's bytes, back to back. */
    unified_vector<char> arena;

    /** One view per needle. */
    unified_vector<sz_string_view_t> views;

    /** Hands back memory both the host and the device address. */
    sz_allocator_t unified {};

    /** Host accessors, which is all the builder needs. */
    sz_sequence_t needles {};

    substrings_simt_vocabulary_t(std::vector<std::string> const &strings) : views(strings.size()) {
        for (std::size_t index = 0; index != strings.size(); ++index) {
            views[index].length = strings[index].size();
            arena.insert(arena.end(), strings[index].begin(), strings[index].end());
        }
        std::size_t written = 0;
        for (sz_string_view_t &view : views) view.start = arena.data() + written, written += view.length;
        sz_sequence_from_string_views(views.data(), views.size(), &needles);
        verify(sz_allocator_init_unified_best(&unified, gpu_capabilities()) == sz_success_k);
    }
    substrings_simt_vocabulary_t(substrings_simt_vocabulary_t const &) = delete;
    substrings_simt_vocabulary_t &operator=(substrings_simt_vocabulary_t const &) = delete;
};

/**
 *  @brief One vocabulary compiled twice under one policy: for the host and for @c stream 's device.
 *
 *  The policy sizes the arena, so it belongs to the engine rather than to a call, and comparing two
 *  tiers under one policy means holding two engines rather than one object read two ways.
 */
struct substrings_simt_engines_t {

    /** What the serial oracle's engine is built from. */
    handle_checked_heap_t heap;

    /** The serial oracle's engine, in plain host memory. */
    sz_substrings_engine_t host {};

    /** The device's engine, arena and report included. */
    sz_substrings_engine_t device {};

    substrings_simt_engines_t(substrings_simt_vocabulary_t &vocabulary, sz_substrings_case_sensitivity_t sensitivity,
                              sz_substrings_overlap_policy_t policy) {
        verify(sz_substrings_engine_init_serial(&host, &vocabulary.needles, sensitivity, policy,
                                                STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0, &heap.allocator,
                                                nullptr) == sz_success_k);
        verify(sz_substrings_engine_init(&device, &vocabulary.needles, sensitivity, policy,
                                         STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0, gpu_capabilities(),
                                         &vocabulary.unified, nullptr) == sz_success_k);
        verify(gpu_memory_reaches(device.memory) && "A device engine's block is one a kernel addresses");
        verify(!gpu_memory_reaches(host.memory) && "A host engine's block is not");
    }
    substrings_simt_engines_t(substrings_simt_engines_t const &) = delete;
    substrings_simt_engines_t &operator=(substrings_simt_engines_t const &) = delete;
    ~substrings_simt_engines_t() noexcept {
        sz_substrings_engine_free(&device, nullptr);
        sz_substrings_engine_free(&host, nullptr);
        verify(heap.live_allocations == 0);
    }
};

/** One match, ordered so two backends' reports compare as sequences rather than as multisets. */
struct substrings_simt_case_t {

    /** Which haystack of the sequence this match was found in. */
    sz_size_t haystack_index {};

    /** Which needle of the vocabulary matched. */
    sz_size_t needle_index {};

    /** Where the match starts inside that haystack. */
    sz_size_t byte_offset {};

    /** Haystack bytes the match spans. */
    sz_size_t byte_length {};

    bool operator==(substrings_simt_case_t const &other) const noexcept {
        return haystack_index == other.haystack_index && needle_index == other.needle_index &&
               byte_offset == other.byte_offset && byte_length == other.byte_length;
    }
    bool operator<(substrings_simt_case_t const &other) const noexcept {
        if (haystack_index != other.haystack_index) return haystack_index < other.haystack_index;
        if (byte_offset != other.byte_offset) return byte_offset < other.byte_offset;
        if (byte_length != other.byte_length) return byte_length < other.byte_length;
        return needle_index < other.needle_index;
    }
};

/** Sorts one verb's match array into the shape two backends compare position by position. */
static std::vector<substrings_simt_case_t> sorted_(sz_substrings_match_t const *matches, sz_size_t count) {
    std::vector<substrings_simt_case_t> reported(count);
    for (std::size_t index = 0; index != count; ++index)
        reported[index] = {matches[index].haystack_index, matches[index].needle_index, matches[index].byte_offset,
                           matches[index].byte_length};
    std::sort(reported.begin(), reported.end());
    return reported;
}

/** Every match the serial tier reports, sized from the report its own sizing call leaves. */
static std::vector<substrings_simt_case_t> serial_matches_(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks) {
    std::vector<sz_size_t> offsets(haystacks->count + 1, 0);
    verify(sz_substrings_find_serial(engine, haystacks, nullptr, 0, offsets.data(), nullptr) == sz_success_k);
    std::vector<sz_substrings_match_t> matches(engine->report->matches_emitted);
    verify(sz_substrings_find_serial(engine, haystacks, matches.data(), matches.size(), offsets.data(), nullptr) ==
           sz_success_k);
    verify(engine->report->shortfall == 0);
    return sorted_(matches.data(), matches.size());
}

/** Every match the device tier reports, joined per call since no verb joins for the caller. */
static std::vector<substrings_simt_case_t> device_matches_(simt_backend_t const &backend,
                                                           sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks) {
    unified_vector<sz_size_t> offsets(haystacks->count + 1, 0);
    verify(backend.substrings_find(engine, haystacks, nullptr, 0, offsets.data(), STRINGZILLA_NULL) == sz_success_k);
    join_();
    // A cover thins the matches after the sizing walk, so the boundaries name the survivors and the
    // report's emitted count names what the walk found before them.
    sz_size_t const total = offsets[haystacks->count];
    verify(engine->report->matches_stored + engine->report->shortfall == total);

    unified_vector<sz_substrings_match_t> matches(total);
    verify(backend.substrings_find(engine, haystacks, matches.data(), matches.size(), offsets.data(),
                                   STRINGZILLA_NULL) == sz_success_k);
    join_();
    verify(engine->report->matches_stored == total && engine->report->shortfall == 0);
    return sorted_(matches.data(), total);
}

/** That every reported match is real, and that no two of them share a byte of one haystack. */
static void verify_is_a_cover_(std::vector<substrings_simt_case_t> const &reported,
                               std::vector<substrings_simt_case_t> const &every_match) {
    // Both lists arrive sorted, so the subset test is one merge rather than a scan
    // per reported match.
    std::size_t candidate = 0;
    for (substrings_simt_case_t const &match : reported) {
        while (candidate != every_match.size() && every_match[candidate] < match) ++candidate;
        verify(candidate != every_match.size() && every_match[candidate] == match &&
               "A cover reports only matches the overlapping walk found");
    }
    for (std::size_t index = 1; index < reported.size(); ++index) {
        substrings_simt_case_t const &previous = reported[index - 1], &current = reported[index];
        if (previous.haystack_index != current.haystack_index) continue;
        verify(current.byte_offset >= previous.byte_offset + previous.byte_length &&
               "A cover's matches share no bytes");
    }
    // A cover leaves a match out only for overlapping a kept one, which an empty or thinned
    // report would not.
    std::size_t kept = 0;
    for (substrings_simt_case_t const &match : every_match) {
        while (kept != reported.size() &&
               (reported[kept].haystack_index < match.haystack_index ||
                (reported[kept].haystack_index == match.haystack_index &&
                 reported[kept].byte_offset + reported[kept].byte_length <= match.byte_offset)))
            ++kept;
        verify(kept != reported.size() && reported[kept].haystack_index == match.haystack_index &&
               reported[kept].byte_offset < match.byte_offset + match.byte_length &&
               "A cover leaves out only matches it overlaps");
    }
}

/**
 *  @brief The device's matches, counts and rewrite against serial's, over one corpus and policy.
 *  @param[in] fidelity Whether the device's cover must equal the serial one or just be valid.
 *
 *  A dense vocabulary leaves no gap between matches, so one run of mutually-reaching matches spans
 *  a whole haystack and the device falls back to emitted order rather than running a quadratic
 *  greedy over it.
 */
static void check_against_serial_(simt_backend_t const &backend, substrings_simt_corpus_t &corpus,
                                  substrings_simt_vocabulary_t &vocabulary, std::vector<std::string> const &haystacks,
                                  sz_substrings_case_sensitivity_t sensitivity, sz_substrings_overlap_policy_t policy,
                                  sz_substrings_cover_fidelity_t fidelity = sz_substrings_cover_exact_k) {
    substrings_simt_engines_t engines(vocabulary, sensitivity, policy);
    std::vector<substrings_simt_case_t> const expected = serial_matches_(&engines.host, &corpus.host_haystacks);
    std::vector<substrings_simt_case_t> const reported = device_matches_(backend, &engines.device,
                                                                         &corpus.device_haystacks);
    sz_size_t const device_total = reported.size();
    if (fidelity == sz_substrings_cover_exact_k) {
        verify(reported.size() == expected.size());
        for (std::size_t index = 0; index != expected.size(); ++index) verify(reported[index] == expected[index]);
    }
    else {
        // Every match the overlapping walk found, which is what a cover may draw from.
        substrings_simt_engines_t overlapping(vocabulary, sensitivity, sz_substrings_overlapping_k);
        verify_is_a_cover_(reported, serial_matches_(&overlapping.host, &corpus.host_haystacks));
    }

    // The counts come from the boundaries rather than from the match list, so they can
    // disagree with it.
    unified_vector<sz_size_t> device_counts(haystacks.size(), 0);
    std::vector<sz_size_t> serial_counts(haystacks.size(), 0);
    verify(sz_substrings_counts_serial(&engines.host, &corpus.host_haystacks, serial_counts.data(), 1, nullptr) ==
           sz_success_k);
    verify(backend.substrings_counts(&engines.device, &corpus.device_haystacks, device_counts.data(), 1,
                                     STRINGZILLA_NULL) == sz_success_k);
    join_();
    for (std::size_t index = 0; index != haystacks.size(); ++index)
        if (fidelity == sz_substrings_cover_exact_k) verify(device_counts[index] == serial_counts[index]);

    // The dispatched verb picks the same kernel by the engine's recorded capability, so both agree.
    unified_vector<sz_size_t> dispatched_counts(haystacks.size(), 0);
    verify(sz_substrings_counts(&engines.device, &corpus.device_haystacks, dispatched_counts.data(), 1,
                                STRINGZILLA_NULL) == sz_success_k);
    join_();
    for (std::size_t index = 0; index != haystacks.size(); ++index)
        verify(dispatched_counts[index] == device_counts[index]);
    {
        sz_size_t counted = 0;
        for (sz_size_t const count : device_counts) counted += count;
        verify(counted == device_total && "The counts and the matches come from one pass");
    }

    if (policy == sz_substrings_overlapping_k) return;
    if (fidelity != sz_substrings_cover_exact_k) return;

    // The rewrite, whose device path owns a block scan and a tiled copy the serial one has
    // no analogue for.
    std::vector<std::string> replacements;
    for (std::size_t index = 0; index != vocabulary.views.size(); ++index)
        replacements.push_back(index % 3 == 0 ? std::string() : "<" + std::to_string(index) + ">");
    substrings_simt_corpus_t replacement_corpus(replacements);

    unified_vector<sz_size_t> device_offsets(haystacks.size() + 1, 0);
    std::vector<sz_size_t> serial_offsets(haystacks.size() + 1, 0);
    verify(backend.substrings_replace(&engines.device, &corpus.device_haystacks, &replacement_corpus.device_haystacks,
                                      nullptr, 0, device_offsets.data(), STRINGZILLA_NULL) == sz_success_k);
    join_();
    verify(sz_substrings_replace_serial(&engines.host, &corpus.host_haystacks, &replacement_corpus.host_haystacks,
                                        nullptr, 0, serial_offsets.data(), nullptr) == sz_success_k);
    for (std::size_t index = 0; index != haystacks.size() + 1; ++index)
        verify(device_offsets[index] == serial_offsets[index]);

    sz_size_t const rewritten = serial_offsets[haystacks.size()];
    unified_vector<char> device_tape(rewritten);
    std::vector<char> serial_tape(rewritten);
    verify(backend.substrings_replace(&engines.device, &corpus.device_haystacks, &replacement_corpus.device_haystacks,
                                      device_tape.empty() ? nullptr : device_tape.data(), rewritten,
                                      device_offsets.data(), STRINGZILLA_NULL) == sz_success_k);
    join_();
    verify(engines.device.report->shortfall == 0);
    verify(sz_substrings_replace_serial(&engines.host, &corpus.host_haystacks, &replacement_corpus.host_haystacks,
                                        serial_tape.empty() ? nullptr : serial_tape.data(), rewritten,
                                        serial_offsets.data(), nullptr) == sz_success_k);
    for (std::size_t index = 0; index != rewritten; ++index) verify(device_tape[index] == serial_tape[index]);
}

/**
 *  @brief The device's BM25 against the serial tier's, by byte lengths and by caller-given ones.
 *
 *  The device sums in fixed point and the host in ascending needle order, so the two agree to
 *  rounding rather than bit for bit.
 */
static void check_bm25_against_serial_(simt_backend_t const &backend, substrings_simt_corpus_t &corpus,
                                       substrings_simt_vocabulary_t &vocabulary,
                                       std::vector<std::string> const &haystacks,
                                       sz_substrings_case_sensitivity_t sensitivity) {
    substrings_simt_engines_t engines(vocabulary, sensitivity, sz_substrings_overlapping_k);
    std::size_t const needles_count = engines.host.needles_count;
    unified_vector<sz_f32_t> weights(needles_count), given_lengths(haystacks.size());
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
        unified_vector<sz_f32_t> device_scores(haystacks.size(), -1);
        verify(sz_substrings_bm25_scores_serial(&engines.host, &corpus.host_haystacks, lengths, &parameters,
                                                weights.data(), serial_scores.data(), 1, nullptr) == sz_success_k);
        verify(backend.substrings_bm25_scores(&engines.device, &corpus.device_haystacks, lengths, &parameters,
                                              weights.data(), device_scores.data(), 1,
                                              STRINGZILLA_NULL) == sz_success_k);
        join_();
        for (std::size_t index = 0; index != haystacks.size(); ++index) {
            double const tolerance = 1e-5 * std::max(1.0, std::fabs((double)serial_scores[index]));
            verify(std::fabs(device_scores[index] - serial_scores[index]) <= tolerance);
        }
    }
}

/** One vocabulary against one corpus under every policy. */
static void check_policies_(simt_backend_t const &backend, std::vector<std::string> const &haystacks,
                            std::vector<std::string> const &needles, sz_substrings_case_sensitivity_t sensitivity,
                            sz_substrings_cover_fidelity_t fidelity = sz_substrings_cover_exact_k) {
    substrings_simt_corpus_t corpus(haystacks);
    substrings_simt_vocabulary_t vocabulary(needles);
    // An overlapping walk reports every match whatever the density, so it is
    // always compared exactly.
    check_against_serial_(backend, corpus, vocabulary, haystacks, sensitivity, sz_substrings_overlapping_k);
    check_against_serial_(backend, corpus, vocabulary, haystacks, sensitivity, sz_substrings_leftmost_longest_k,
                          fidelity);
    check_against_serial_(backend, corpus, vocabulary, haystacks, sensitivity, sz_substrings_leftmost_first_k,
                          fidelity);
    check_bm25_against_serial_(backend, corpus, vocabulary, haystacks, sensitivity);
}

#pragma endregion Substrings Helpers

#pragma region Substrings Checks

/** The chunk-boundary cases a single-chain host walk cannot express. */
static void test_substrings_simt_unit(simt_backend_t const &backend) {
    // The device walk is chunked and the host walk is not, so a match spanning a chunk boundary is
    // the one thing this can get wrong that the CPU suite cannot see.
    check_policies_(backend, {"ushers"}, {"he", "she", "his", "hers"}, sz_substrings_cased_k);
    check_policies_(backend, {"Straße", "STRASSE"}, {"strasse", "sse", "s"}, sz_substrings_uncased_k);

    // The Kelvin sign folds to one ASCII byte, contracting three source bytes into one folded one.
    check_policies_(backend, {"\xE2\x84\xAA elvin", "kelvin"}, {"k", "kelvin"}, sz_substrings_uncased_k);

    // Malformed UTF-8 in the haystack, which an uncased walk resynchronizes past byte by byte.
    check_policies_(backend,
                    {std::string("ab\xFF" "cd", 5), // "\xFFc" is one escape
                     "abcd"},
                    {"ab", "cd"}, sz_substrings_uncased_k);
}

/** Random corpora wide enough that the planner cuts several chunks per haystack. */
static void test_substrings_simt_equivalence(test_context_t &context, simt_backend_t const &backend) {
    std::mt19937 &generator = context.generator;
    char const *const alphabets[] = {"ab", "abcdefgh", "abcdefghijklmnopqrstuvwxyz"};
    std::size_t const rounds = context.iterations(8);

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
        check_policies_(backend, haystacks, needles, sz_substrings_cased_k, sz_substrings_cover_approximate_k);
        check_policies_(backend, haystacks, needles, sz_substrings_uncased_k, sz_substrings_cover_approximate_k);
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
        substrings_simt_corpus_t corpus(haystacks);
        substrings_simt_vocabulary_t vocabulary(needles);
        verify(vocabulary.views.size() > 4096);
        check_bm25_against_serial_(backend, corpus, vocabulary, haystacks, sz_substrings_cased_k);
    }

    // Documents long enough for a cluster of blocks to walk each of them together, among short ones
    // a block scores alone, under a tally of a slot per needle and under a hashed one.
    {
        std::vector<std::string> haystacks;
        for (std::size_t index = 0; index != 20; ++index)
            haystacks.push_back(random_string(generator, index % 7 == 3 ? (std::size_t)3 << 20 : 512 + index * 31,
                                              "abcdefghijklmnopqrstuvwxyz"));
        substrings_simt_corpus_t corpus(haystacks);
        for (std::size_t const needles_count : {(std::size_t)300, (std::size_t)6000}) {
            std::vector<std::string> needles;
            for (std::size_t index = 0; index != needles_count; ++index)
                needles.push_back(random_string(generator, 2 + index % 4, "abcdefghijklmnopqrstuvwxyz"));
            std::sort(needles.begin(), needles.end());
            needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
            substrings_simt_vocabulary_t vocabulary(needles);
            check_bm25_against_serial_(backend, corpus, vocabulary, haystacks, sz_substrings_cased_k);
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
        check_policies_(backend, haystacks, needles, sz_substrings_cased_k, sz_substrings_cover_approximate_k);
    }

    // One haystack far wider than a chunk, so the boundary case is hit many times over in
    // a single walk.
    {
        std::vector<std::string> const needles {"the", "there", "here", "her", "he"};
        std::vector<std::string> haystacks {random_string(generator, 1 << 20, "the ")};
        // No needle reaches across another's end, so the cover is exact and the rewrite
        // spans many tiles.
        check_policies_(backend, haystacks, needles, sz_substrings_cased_k);
    }
}

/** What the device verbs refuse, and what the report says when an output cannot hold the answer. */
static void test_substrings_simt_safety(simt_backend_t const &backend) {
    std::vector<std::string> const needles {"ab", "cd"};
    std::vector<std::string> const haystacks {"abcdabcd"};
    substrings_simt_corpus_t corpus(haystacks);
    substrings_simt_vocabulary_t vocabulary(needles);
    substrings_simt_engines_t engines(vocabulary, sz_substrings_cased_k, sz_substrings_overlapping_k);

    // A device verb refuses host memory rather than reading it from a kernel, whatever
    // else is resident.
    {
        std::vector<sz_size_t> host_counts(haystacks.size(), 0);
        verify(backend.substrings_counts(&engines.device, &corpus.device_haystacks, host_counts.data(), 1,
                                         STRINGZILLA_NULL) == sz_device_memory_mismatch_k);
    }

    // An output stride of zero cannot address one entry per haystack, whatever the haystack count.
    {
        unified_vector<sz_size_t> counts(haystacks.size(), STRINGZILLA_SIZE_MAX);
        verify(backend.substrings_counts(&engines.device, &corpus.device_haystacks, counts.data(), 0,
                                         STRINGZILLA_NULL) == sz_unexpected_dimensions_k);
        verify(counts[0] == STRINGZILLA_SIZE_MAX);
    }

    // The counts a device walk answers with, which the same corpus answers on the host.
    {
        unified_vector<sz_size_t> counts(haystacks.size(), 0);
        verify(backend.substrings_counts(&engines.device, &corpus.device_haystacks, counts.data(), 1,
                                         STRINGZILLA_NULL) == sz_success_k);
        join_();
        verify(counts[0] == 4);
    }

    // A capacity that cannot hold the matches is not an error: the report names the true total and
    // the shortfall beside it, and the output is left untouched rather than truncated.
    {
        unified_vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        verify(backend.substrings_find(&engines.device, &corpus.device_haystacks, nullptr, 0, offsets.data(),
                                       STRINGZILLA_NULL) == sz_success_k);
        join_();
        verify(engines.device.report->matches_emitted == 4);
        verify(engines.device.report->matches_stored == 0 && engines.device.report->shortfall == 4);
    }

    // A substitution over matches that share bytes is not a function, so the overlapping policy is
    // refused before anything is launched.
    {
        std::vector<std::string> const replacements {"x", "y"};
        substrings_simt_corpus_t replacement_corpus(replacements);
        unified_vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        verify(backend.substrings_replace(&engines.device, &corpus.device_haystacks,
                                          &replacement_corpus.device_haystacks, nullptr, 0, offsets.data(),
                                          STRINGZILLA_NULL) == sz_status_unknown_k);
    }

    // A vocabulary a round cannot emit within its budget leaves every later kernel retired, so the
    // outputs keep whatever they held and the report alone says the round did not fit.
    {
        sz_substrings_engine_t tiny {};
        unified_vector<sz_size_t> offsets(haystacks.size() + 1, STRINGZILLA_SIZE_MAX);
        verify(sz_substrings_engine_init(&tiny, &vocabulary.needles, sz_substrings_cased_k,
                                         sz_substrings_leftmost_first_k, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 1, 0,
                                         gpu_capabilities(), &vocabulary.unified, nullptr) == sz_success_k);
        verify(backend.substrings_find(&tiny, &corpus.device_haystacks, nullptr, 0, offsets.data(), STRINGZILLA_NULL) ==
               sz_success_k);
        join_();
        verify(tiny.report->matches_emitted == 4 && tiny.report->shortfall == 3);
        verify(offsets[haystacks.size()] == 0 && "A retired offsets kernel leaves the boundaries zeroed");
        sz_substrings_engine_free(&tiny, nullptr);
    }

    // The arena is sized for a haystacks budget when the engine is built, so a round carrying more
    // is refused rather than grown under a stream that may still be reading it.
    {
        substrings_simt_corpus_t pair(std::vector<std::string> {"abcd", "abcd"});
        sz_substrings_engine_t narrow {};
        unified_vector<sz_size_t> offsets(3, 0);
        verify(sz_substrings_engine_init(&narrow, &vocabulary.needles, sz_substrings_cased_k,
                                         sz_substrings_overlapping_k, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 1,
                                         gpu_capabilities(), &vocabulary.unified, nullptr) == sz_success_k);
        verify(backend.substrings_find(&narrow, &pair.device_haystacks, nullptr, 0, offsets.data(), STRINGZILLA_NULL) ==
               sz_unexpected_dimensions_k);
        sz_substrings_engine_free(&narrow, nullptr);
    }
}

#pragma endregion Substrings Checks

#pragma region UTF8 Uncased Checks

/** @p fold called the way a CPU fold is, its text, output and length staged where the device
 *  reaches them and the stream joined, so the CPU fold checks drive it unchanged. */
static auto utf8_uncased_fold_staged_(sz_kernel_utf8_uncased_fold_t fold) {
    return [fold](sz_cptr_t source, sz_size_t length, sz_ptr_t target, sz_size_t *target_length,
                  void *stream) -> sz_status_t {
        unified_vector<char> staged_source(source, source + length), staged_target(length * 3 + 4);
        unified_vector<sz_size_t> staged_length(1, 0);
        sz_status_t const status = fold(staged_source.data(), length, staged_target.data(), staged_length.data(),
                                        stream);
        if (status != sz_success_k) return status;
        join_();
        std::copy(staged_target.begin(), staged_target.begin() + staged_length[0], target);
        *target_length = staged_length[0];
        return status;
    };
}

/** The device fold against the serial one, over the battery every CPU fold is held to. */
static void check_utf8_uncased_simt_equivalence_(test_context_t &context, simt_backend_t const &backend) {
    check_uncased_fold_equivalence_(context, sz_utf8_uncased_fold_serial,
                                    utf8_uncased_fold_staged_(backend.utf8_uncased_fold), 4000,
                                    context.iterations(1200));
}

/** The malformed inputs every CPU fold survives, which the device folds byte for byte as serial
 *  does, and memory the device cannot reach, which it refuses rather than reading. */
static void check_utf8_uncased_simt_safety_(test_context_t &context, simt_backend_t const &backend) {
    auto const staged = utf8_uncased_fold_staged_(backend.utf8_uncased_fold);
    std::vector<char> expected, produced;
    for_each_adversarial_utf8_input_(context, context.iterations(1000), [&](char const *input, std::size_t length) {
        expected.resize(length * 3 + 4), produced.resize(length * 3 + 4);
        sz_size_t const expected_length = kernel_result<sz_size_t>(sz_utf8_uncased_fold_serial, input, length,
                                                                   expected.data());
        sz_size_t const produced_length = kernel_result<sz_size_t>(staged, input, length, produced.data());
        if (produced_length != expected_length ||
            !std::equal(expected.begin(), expected.begin() + expected_length, produced.begin()))
            fail_backend_(backend.name, "a malformed text folds differently from serial");
    });

    std::string const host_source = "HELLO";
    unified_vector<char> source(host_source.begin(), host_source.end()), target(host_source.size() * 3);
    unified_vector<sz_size_t> target_length(1, 0);
    std::vector<char> host_target(host_source.size() * 3);
    sz_size_t host_length = 0;
    if (backend.utf8_uncased_fold(host_source.data(), host_source.size(), target.data(), target_length.data(),
                                  STRINGZILLA_NULL) != sz_device_memory_mismatch_k ||
        backend.utf8_uncased_fold(source.data(), source.size(), host_target.data(), target_length.data(),
                                  STRINGZILLA_NULL) != sz_device_memory_mismatch_k ||
        backend.utf8_uncased_fold(source.data(), source.size(), target.data(), &host_length, STRINGZILLA_NULL) !=
            sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "a fold of memory the device cannot reach was not refused");
}

#pragma endregion UTF8 Uncased Checks

#pragma region UTF8 Norm Checks

/** @p norm called the way a CPU normalizer is, its text, output and length staged where the device
 *  reaches them and the stream joined, so the CPU normalization checks drive it unchanged. */
static auto utf8_norm_staged_(sz_kernel_utf8_norm_t norm) {
    return [norm](sz_cptr_t source, sz_size_t length, sz_normal_form_t form, sz_ptr_t target, sz_size_t *target_length,
                  void *stream) -> sz_status_t {
        unified_vector<char> staged_source(source, source + length), staged_target(length * 18 + 18);
        unified_vector<sz_size_t> staged_length(1, 0);
        sz_status_t const status = norm(staged_source.data(), length, form, staged_target.data(), staged_length.data(),
                                        stream);
        if (status != sz_success_k) return status;
        join_();
        std::copy(staged_target.begin(), staged_target.begin() + staged_length[0], target);
        *target_length = staged_length[0];
        return status;
    };
}

/** The device normalizer against the serial one, over the battery every CPU normalizer is held to;
 *  the device has no violation finder, so the serial one stands in for it. */
static void check_utf8_norm_simt_equivalence_(test_context_t &context, simt_backend_t const &backend) {
    struct {
        decltype(utf8_norm_staged_(nullptr)) norm;
        sz_kernel_utf8_find_denormalized_t find_denormalized;
    } const kernels {utf8_norm_staged_(backend.utf8_norm), sz_utf8_find_denormalized_serial};
    check_utf8_norm_equivalence_(context, kernels);
}

/** The well-formed adversarial inputs every CPU normalizer faces, which the device normalizes byte
 *  for byte as serial does in every form, and memory the device cannot reach, which it refuses. */
static void check_utf8_norm_simt_safety_(test_context_t &context, simt_backend_t const &backend) {
    static sz_normal_form_t const norm_forms[4] = {sz_normal_form_nfd_k, sz_normal_form_nfc_k, sz_normal_form_nfkd_k,
                                                   sz_normal_form_nfkc_k};
    auto const staged = utf8_norm_staged_(backend.utf8_norm);
    std::vector<char> expected, produced;
    for_each_adversarial_utf8_input_(context, context.iterations(1000), [&](char const *input, std::size_t length) {
        if (sz_utf8_find_malformed(input, (sz_size_t)length) != STRINGZILLA_NULL_CHAR) return;
        expected.resize(length * 18 + 18), produced.resize(length * 18 + 18);
        for (sz_normal_form_t form : norm_forms) {
            sz_size_t const expected_length = kernel_result<sz_size_t>(sz_utf8_norm_serial, input, length, form,
                                                                       expected.data());
            sz_size_t const produced_length = kernel_result<sz_size_t>(staged, input, length, form, produced.data());
            if (produced_length != expected_length ||
                !std::equal(expected.begin(), expected.begin() + expected_length, produced.begin()))
                fail_backend_(backend.name, "a text normalizes differently from serial");
        }
    });

    std::string const host_source = "caf\xC3\xA9";
    unified_vector<char> source(host_source.begin(), host_source.end()), target(host_source.size() * 18);
    unified_vector<sz_size_t> target_length(1, 0);
    std::vector<char> host_target(host_source.size() * 18);
    sz_size_t host_length = 0;
    if (backend.utf8_norm(host_source.data(), host_source.size(), sz_normal_form_nfd_k, target.data(),
                          target_length.data(), STRINGZILLA_NULL) != sz_device_memory_mismatch_k ||
        backend.utf8_norm(source.data(), source.size(), sz_normal_form_nfd_k, host_target.data(), target_length.data(),
                          STRINGZILLA_NULL) != sz_device_memory_mismatch_k ||
        backend.utf8_norm(source.data(), source.size(), sz_normal_form_nfd_k, target.data(), &host_length,
                          STRINGZILLA_NULL) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "a normalization of memory the device cannot reach was not refused");
}

#pragma endregion UTF8 Norm Checks

#pragma region Drivers

/** Registers every check of one vendor's kernels, or of the dispatch points, in @p check. */
inline void check_simt_backend_(cross_section_t &check, simt_backend_t const &backend) {
    std::string const suffix = backend.name;
    check("test_levenshtein_equivalence_" + suffix, [&](test_context_t &context) {
        check_levenshtein_simt_equivalence_(backend.name, levenshtein_simt_alphabet_t::bytes_k,
                                            backend.levenshtein_distances);
        check_levenshtein_simt_equivalence_(backend.name, levenshtein_simt_alphabet_t::runes_k,
                                            backend.levenshtein_distances);
        check_levenshtein_simt_narrow_lanes_(context.generator, backend.name, backend.levenshtein_distances);
        check_levenshtein_simt_tiled_(backend.name, backend.levenshtein_distance_tiled);
    });
    check("test_levenshtein_skewed_" + suffix, [&](test_context_t &context) {
        check_levenshtein_simt_skewed_(context.generator, backend.name, levenshtein_simt_alphabet_t::bytes_k,
                                       backend.levenshtein_distances);
        check_levenshtein_simt_skewed_(context.generator, backend.name, levenshtein_simt_alphabet_t::runes_k,
                                       backend.levenshtein_distances);
    });
    check("test_levenshtein_safety_" + suffix,
          [&](test_context_t &context) { check_levenshtein_simt_memory_safety_(context.generator, backend); });
    check("test_overlap_equivalence_" + suffix,
          [&](test_context_t &context) { check_overlap_simt_equivalence_(context.generator, backend); });
    check("test_overlap_skewed_" + suffix,
          [&](test_context_t &context) { check_overlap_simt_skewed_(context.generator, backend); });
    check("test_overlap_safety_" + suffix, [&](test_context_t &context) {
        check_overlap_simt_memory_safety_(context.generator, backend);
        check_overlap_simt_width_safety_(context.generator, backend);
    });
    check("test_substrings_unit_" + suffix, [&] { test_substrings_simt_unit(backend); });
    check("test_substrings_equivalence_" + suffix,
          [&](test_context_t &context) { test_substrings_simt_equivalence(context, backend); });
    check("test_substrings_safety_" + suffix, [&] { test_substrings_simt_safety(backend); });
    check("test_utf8_uncased_equivalence_" + suffix,
          [&](test_context_t &context) { check_utf8_uncased_simt_equivalence_(context, backend); });
    check("test_utf8_uncased_safety_" + suffix,
          [&](test_context_t &context) { check_utf8_uncased_simt_safety_(context, backend); });
    check("test_utf8_norm_equivalence_" + suffix,
          [&](test_context_t &context) { check_utf8_norm_simt_equivalence_(context, backend); });
    check("test_utf8_norm_safety_" + suffix,
          [&](test_context_t &context) { check_utf8_norm_simt_safety_(context, backend); });
}

/** A copy lands every string in one unified tape, and copying that tape again only re-points it. */
static void check_sequence_copy_simt_() {
    std::array<sz_string_view_t, 3> const views {sz_string_view_t {"kitten", 6}, {"", 0}, {"sitting", 7}};
    sz_sequence_t host {};
    sz_sequence_from_string_views(views.data(), views.size(), &host);
    gpu_tape_t tape;
    tape.copy(host);
    verify(tape.bytes == 4 * sizeof(sz_u64_t) + 13 && gpu_memory_reaches(tape.sequence.handle));
    for (std::size_t index = 0; index != views.size(); ++index) {
        sz_string_view_t const copied {sz_sequence_tape_start(tape.sequence.handle, index),
                                       sz_sequence_tape_length(tape.sequence.handle, index)};
        verify(std::string_view(copied.start, copied.length) ==
               std::string_view(views[index].start, views[index].length));
    }
    sz_sequence_t again {};
    sz_size_t again_bytes = 1;
    verify(sz_sequence_realloc_best(&again, &tape.sequence, &tape.unified, &again_bytes, gpu_capabilities(), nullptr) ==
           sz_success_k);
    verify(again_bytes == 0 && again.handle == tape.sequence.handle && again.get_start == tape.sequence.get_start);
}

/** The dispatching entry points, over the capabilities of the device the checks launch on, and the
 *  refusals and asynchrony only a dispatch point's engine init promises. */
std::size_t test_cross_dispatch(environment_t const &env) {
    simt_backend_t const dispatched {"dispatched",
                                     sz_levenshtein_distances,
                                     gpu_best<sz_levenshtein_distance_tiled_best>,
                                     sz_overlap_scores,
                                     sz_substrings_counts,
                                     sz_substrings_find,
                                     sz_substrings_replace,
                                     sz_substrings_bm25_scores,
                                     gpu_best<sz_utf8_uncased_fold_best>,
                                     gpu_best<sz_utf8_norm_best>};
    cross_section_t check(env);
    check.detected = gpu_capabilities();
    check.section("Cross Dispatch", gpu_baseline_k);
    check_simt_backend_(check, dispatched);
    check("test_sequence_copy_dispatched", [] { check_sequence_copy_simt_(); });
    check("test_levenshtein_refusals_dispatched", [] {
        check_levenshtein_simt_query_safety_();
        check_levenshtein_simt_empty_query_safety_();
    });
    check("test_levenshtein_asynchrony_dispatched", [] { check_levenshtein_simt_scheduled_asynchrony_(); });
    check("test_overlap_asynchrony_dispatched",
          [](test_context_t &context) { check_overlap_simt_asynchrony_(context.generator); });
    return check.failures;
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test

#endif // STRINGZILLA_TEST_CROSS_SIMT_CUH
