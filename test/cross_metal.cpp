/**
 *  @file test/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief GPU engine checks - the Metal kernels by name, and the dispatching entry points, the twin
 *      of `cross_simt.cuh`.
 *
 *  Covers each Metal engine against serial's answers bit for bit, the registry contract its verbs
 *  keep, the bounds the widest rung and the per-thread ring impose, the chunk boundaries a
 *  single-chain host walk never crosses, the asynchrony the verbs promise, two threads on two
 *  queues, and a free deferred behind a running round.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <cmath>   // `std::fabs`
#include <cstddef> // `std::size_t`

#include <algorithm>   // `std::sort`
#include <array>       // `std::array`
#include <string>      // `std::string`
#include <thread>      // `std::thread`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp" // `cross_section_t`, `handle_checked_heap_t`, `verify`

namespace ashvardanian::stringzilla::test {

/** What Metal device 0 enables, the device whose queue @c main opens, verified to exist. */
static sz_capability_t metal_capabilities_() {
    sz_capability_t capabilities = 0;
    verify(sz_metal_capabilities_enabled(0, &capabilities) == sz_success_k);
    return capabilities;
}

template <typename value_type_>
using metal_unified_alloc = unified_alloc<value_type_, sz_cap_metal_k>;

template <typename value_type_>
using metal_vector = std::vector<value_type_, metal_unified_alloc<value_type_>>;

/** Host views copied into one tape in unified memory, which a kernel reads and the serial reference
 *  calls through the same accessors. */
struct metal_tape_t {
    void *queue;
    sz_sequence_t sequence {};
    sz_size_t bytes = 0;

    explicit metal_tape_t(void *queue) noexcept : queue(queue) {}
    metal_tape_t(metal_tape_t const &) = delete;
    metal_tape_t &operator=(metal_tape_t const &) = delete;
    ~metal_tape_t() noexcept { reset(); }

    void copy(std::vector<sz_string_view_t> const &views) {
        reset();
        sz_sequence_t host {};
        sz_sequence_from_string_views(views.data(), views.size(), &host);
        sz_allocator_t unified;
        sz_allocator_init_unified_metal(&unified);
        verify(sz_sequence_realloc_metal(&sequence, &host, &unified, &bytes, queue) == sz_success_k);
    }
    void reset() noexcept {
        if (!bytes) return;
        sz_allocator_t unified;
        sz_allocator_init_unified_metal(&unified);
        unified.free(const_cast<void *>(sequence.handle), bytes, unified.handle, queue);
        bytes = 0;
    }
};

/** Accessors one byte off the tape they sit on stand for any other layout, which a kernel reading
 *  the tape directly would score wrongly. */
static sz_sequence_t foreign_tape_(sz_sequence_t const &tape) {
    sz_sequence_t foreign = tape;
    foreign.get_start = [](void const *handle, sz_size_t index) -> sz_cptr_t {
        sz_u64_t const *offsets = static_cast<sz_u64_t const *>(handle);
        return static_cast<sz_cptr_t>(handle) + offsets[index] + 1;
    };
    foreign.get_length = [](void const *handle, sz_size_t index) -> sz_size_t {
        sz_u64_t const *offsets = static_cast<sz_u64_t const *>(handle);
        return offsets[index + 1] - offsets[index] - 1;
    };
    return foreign;
}

/** The Metal engine kernels, or the dispatch points in their place, which take the same arguments:
 *  a device engine records its capability, so a verb reaches the Metal kernel. */
struct metal_backend_t {

    /** The spelling @ref fail_backend_ and the test names carry. */
    char const *name;

    sz_kernel_levenshtein_distances_t levenshtein_distances;
    sz_kernel_overlap_scores_t overlap_scores;
    sz_kernel_substrings_counts_t substrings_counts;
    sz_kernel_substrings_find_t substrings_find;
    sz_kernel_substrings_replace_t substrings_replace;
    sz_kernel_substrings_bm25_scores_t substrings_bm25_scores;
};

#pragma region Levenshtein Helpers

/** What a corpus symbol is: a byte the tables class directly, or a rune the page table classes. */
enum class levenshtein_metal_alphabet_t { bytes_k, runes_k };

/** The alphabet enumerator an engine is prepared over, matching the corpus alphabet of a check. */
static sz_levenshtein_symbol_t levenshtein_metal_symbol_(levenshtein_metal_alphabet_t alphabet) {
    return alphabet == levenshtein_metal_alphabet_t::runes_k ? sz_levenshtein_runes_k : sz_levenshtein_bytes_k;
}

/** Corpus symbols: every byte value, or runes across four scripts and all three encoded widths. */
static std::vector<std::string> levenshtein_metal_symbols_(levenshtein_metal_alphabet_t alphabet) {
    std::vector<std::string> symbols;
    if (alphabet == levenshtein_metal_alphabet_t::bytes_k) {
        for (unsigned value = 0; value != 256; ++value) symbols.push_back(std::string(1, (char)value));
        return symbols;
    }
    char const *const runes[] = {"a", "z",  "é",  "ß",  "α",          "ω",          "ж",
                                 "я", "中", "漢", "字", "\U0001f600", "\U0001f30d", "\U0001d11e"};
    for (char const *const rune : runes) symbols.push_back(std::string(rune));
    return symbols;
}

/**
 *  @brief A corpus both sides address: near-duplicates of the queries as a tape in unified memory,
 *      and the matrix.
 *
 *  The queries are read on the host alone, so they stay in host memory. The candidates are edits of
 *  the queries, so the distances are small and every word of the Myers state moves, and edits land
 *  on whole symbols, so a rune corpus stays well-formed.
 */
struct levenshtein_metal_corpus_t {

    /** Every query's bytes, back to back. */
    std::vector<char> query_arena;

    /** One view per query; its size is the query count. */
    std::vector<sz_string_view_t> query_views;

    /** Every candidate's bytes, back to back, on the host. */
    std::vector<char> candidate_arena;

    /** One view per candidate; its size is the candidate count. */
    std::vector<sz_string_view_t> candidate_views;

    /** @b [queries, candidates], written by whichever round ran. */
    metal_vector<sz_size_t> distances;

    /** Host accessors over the queries, as an init requires. */
    sz_sequence_t queries {};

    /** The candidates' tape, which the device reads directly and the serial reference calls. */
    metal_tape_t candidates;

    levenshtein_metal_corpus_t(void *queue, std::size_t count, std::size_t query_symbols, std::size_t query_count,
                               levenshtein_metal_alphabet_t alphabet)
        : query_views(query_count), candidate_views(count),
          distances(query_count * count, STRINGZILLA_SIZE_MAX, metal_unified_alloc<sz_size_t>(queue)),
          candidates(queue) {
        std::vector<std::string> const symbols = levenshtein_metal_symbols_(alphabet);
        std::vector<std::string> drawn(query_symbols);
        for (std::size_t symbol = 0; symbol != query_symbols; ++symbol)
            drawn[symbol] = symbols[(symbol * 37 + 11) % symbols.size()];

        // Every query is the same length, so one batch spans one rung, and every one
        // of them differs.
        for (std::size_t query = 0; query != query_count; ++query) {
            std::vector<std::string> edited = drawn;
            for (std::size_t symbol = query; symbol < edited.size(); symbol += query_count + 1)
                edited[symbol] = symbols[(symbol + query * 5 + 3) % symbols.size()];
            std::size_t length = 0;
            for (std::string const &symbol : edited) length += symbol.size();
            query_views[query].length = length;
            for (std::string const &symbol : edited)
                query_arena.insert(query_arena.end(), symbol.begin(), symbol.end());
        }

        std::vector<std::string> edited;
        for (std::size_t index = 0; index != count; ++index) {
            // The second candidate is empty, which the warped rung reaches without taking
            // a single step.
            edited = index == 1 ? std::vector<std::string>() : drawn;
            for (std::size_t edit = 0, edits = index % 17; edit != edits && edited.size() > 1; ++edit) {
                std::size_t const at = (index * 31 + edit * 7) % edited.size();
                if (edit % 2) { edited.erase(edited.begin() + (std::ptrdiff_t)at); }
                else { edited[at] = symbols[(edit + index) % symbols.size()]; }
            }
            std::size_t length = 0;
            for (std::string const &symbol : edited) length += symbol.size();
            candidate_views[index].length = length;
            for (std::string const &symbol : edited)
                candidate_arena.insert(candidate_arena.end(), symbol.begin(), symbol.end());
        }

        // An arena's address is final only once it stops growing, so the starts
        // are filled afterwards.
        std::size_t written = 0;
        for (sz_string_view_t &view : query_views) view.start = query_arena.data() + written, written += view.length;
        written = 0;
        for (sz_string_view_t &view : candidate_views)
            view.start = candidate_arena.data() + written, written += view.length;
        sz_sequence_from_string_views(query_views.data(), query_views.size(), &queries);
        candidates.copy(candidate_views);
    }

    /** Candidates one round scores, which is also the row stride of the matrix it writes. */
    std::size_t count() const { return candidate_views.size(); }
};

/** The serial backend's answers for the same corpus, read off the very tape the device reads. */
static std::vector<sz_size_t> levenshtein_serial_reference_(levenshtein_metal_corpus_t const &corpus,
                                                            sz_levenshtein_symbol_t symbol) {
    handle_checked_heap_t heap;
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init_serial(&engine, &corpus.queries, symbol, &heap.allocator, nullptr) ==
           sz_success_k);
    std::vector<sz_size_t> expected(corpus.distances.size());
    verify(sz_levenshtein_distances_serial(&engine, &corpus.candidates.sequence, expected.data(), corpus.count(),
                                           nullptr) == sz_success_k);
    sz_levenshtein_engine_free(&engine, nullptr);
    verify(heap.live_allocations == 0);
    return expected;
}

#pragma endregion Levenshtein Helpers

#pragma region Levenshtein Expectations

/** The status a query of @p query_symbols symbols draws: the widest rung's thirty-two lanes hold
 *  @ref sz_levenshtein_gpu_words_max_k words between them, over bytes and runes alike. */
static constexpr sz_status_t levenshtein_metal_expected_status_(std::size_t query_symbols) {
    return query_symbols <= (std::size_t)sz_levenshtein_gpu_words_max_k * 64 ? sz_success_k
                                                                             : sz_unexpected_dimensions_k;
}

/** Symbol-pairs one sweep scores before it drops the widest batch, so the serial reference stays
 *  quadratic in time but bounded in it as the rungs widen. */
enum { levenshtein_metal_sweep_budget_k = 4u * 1024u * 1024u };

/** Queries one sweep prepares at a time, so the grid's row axis carries more than one. */
enum { levenshtein_metal_sweep_queries_k = 3 };

/** Every query length the sweeps walk, either side of each boundary a rung is cut at: every
 *  word through the threaded rung's fifteenth, each its own entry point, and every
 *  thirty-second word past the warped rung's crossing, where its lanes take one more word each,
 *  up to the ceiling itself. */
static constexpr std::size_t levenshtein_metal_query_symbols_k[] = {
    1,    2,    8,    63,   64,    65,    127,   128,   129,   192,   193,   256,   257,   320,  321,
    384,  385,  448,  449,  512,   513,   576,   577,   640,   641,   704,   705,   768,   769,  832,
    833,  896,  897,  959,  960,   961,   1023,  1024,  1025,  2047,  2048,  2049,  4095,  4096, 4097,
    6144, 6145, 8192, 8193, 10240, 10241, 12288, 12289, 14336, 14337, 16383, 16384, 16385,
};

#pragma endregion Levenshtein Expectations

#pragma region Levenshtein Checks

/**
 *  @brief One backend's distances against serial's, on corpora that never leave unified memory.
 *
 *  The corpus alphabet and the engine's symbol are separate, so a byte corpus read as runes feeds
 *  the rune decoder stray continuations, truncated sequences and bytes no rune may start with.
 */
static void check_levenshtein_metal_equivalence_(void *queue, levenshtein_metal_alphabet_t alphabet,
                                                 sz_levenshtein_symbol_t symbol, metal_backend_t const &backend) {
    for (std::size_t const query_symbols : levenshtein_metal_query_symbols_k)
        for (std::size_t count : {1u, 129u, 2048u}) {
            std::size_t const queries = levenshtein_metal_sweep_queries_k;
            if (count * query_symbols * queries > (std::size_t)levenshtein_metal_sweep_budget_k) continue;
            levenshtein_metal_corpus_t corpus(queue, count, query_symbols, queries, alphabet);
            std::size_t longest = 0;
            for (sz_string_view_t const &query : corpus.query_views)
                longest = std::max<std::size_t>(longest, symbol == sz_levenshtein_runes_k
                                                             ? sz_levenshtein_utf8_runes(query.start, query.length)
                                                             : query.length);

            sz_levenshtein_engine_t engine {};
            sz_status_t const prepared = sz_levenshtein_engine_init(&engine, &corpus.queries, symbol,
                                                                    metal_capabilities_(), STRINGZILLA_NULL, queue);
            if (prepared != levenshtein_metal_expected_status_(longest))
                fail_backend_(backend.name, "a device batch drew a status the rungs do not imply");
            if (prepared != sz_success_k) continue;
            sz_status_t const produced = backend.levenshtein_distances(&engine, &corpus.candidates.sequence,
                                                                       corpus.distances.data(), corpus.count(), queue);
            verify(sz_stream_synchronize_metal(queue) == sz_success_k);
            sz_levenshtein_engine_free(&engine, queue);
            if (produced != sz_success_k) fail_backend_(backend.name, "a unified-memory round was refused");

            std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus, symbol);
            for (std::size_t index = 0; index != expected.size(); ++index)
                if (corpus.distances[index] != expected[index])
                    fail_backend_(backend.name, "a distance differs from serial");
        }
}

/** One backend refusing unregistered memory, and sequences it cannot read, before encoding. */
static void check_levenshtein_metal_memory_safety_(void *queue, std::mt19937 &generator,
                                                   metal_backend_t const &backend) {
    levenshtein_metal_corpus_t corpus(queue, 8, 200, 1, levenshtein_metal_alphabet_t::bytes_k);
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init(&engine, &corpus.queries, sz_levenshtein_bytes_k, metal_capabilities_(),
                                      STRINGZILLA_NULL, queue) == sz_success_k);

    std::string text(64, '\0');
    randomize_string(generator, text);
    std::array<sz_string_view_t, 1> const host_views {sz_string_view_t {text.data(), text.size()}};
    sz_sequence_t host_candidates {};
    sz_sequence_from_string_views(host_views.data(), host_views.size(), &host_candidates);
    std::vector<sz_size_t> host_distances(corpus.distances.size(), STRINGZILLA_SIZE_MAX);
    if (backend.levenshtein_distances(&engine, &corpus.candidates.sequence, host_distances.data(), corpus.count(),
                                      queue) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host distances were not refused");
    if (backend.levenshtein_distances(&engine, &host_candidates, corpus.distances.data(), corpus.count(), queue) !=
        sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "a host-resident view array was not refused");
    if (backend.levenshtein_distances(&engine, &corpus.candidates.sequence, corpus.distances.data(), corpus.count() - 1,
                                      queue) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a stride under the candidates was not refused");
    sz_sequence_t const foreign = foreign_tape_(corpus.candidates.sequence);
    if (backend.levenshtein_distances(&engine, &foreign, corpus.distances.data(), corpus.count(), queue) !=
        sz_device_code_mismatch_k)
        fail_backend_(backend.name, "a sequence with foreign accessors was not refused");
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    sz_levenshtein_engine_free(&engine, queue);
    for (sz_size_t const untouched : host_distances)
        if (untouched != STRINGZILLA_SIZE_MAX) fail_backend_(backend.name, "a refused call still wrote a distance");
    for (sz_size_t const untouched : corpus.distances)
        if (untouched != STRINGZILLA_SIZE_MAX) fail_backend_(backend.name, "a refused call still wrote a distance");
}

/** The builder refusing a query past what the widest rung holds, in bytes and again in runes, and
 *  a batch holding an empty query, which has no last word to read a score off. */
static void check_levenshtein_metal_query_safety_(void *queue) {
    enum { symbols_k = sz_levenshtein_gpu_words_max_k * 64 + 1 };
    for (levenshtein_metal_alphabet_t const alphabet :
         {levenshtein_metal_alphabet_t::bytes_k, levenshtein_metal_alphabet_t::runes_k}) {
        levenshtein_metal_corpus_t corpus(queue, 1, symbols_k, 1, alphabet);
        sz_levenshtein_engine_t engine {};
        if (sz_levenshtein_engine_init(&engine, &corpus.queries, levenshtein_metal_symbol_(alphabet),
                                       metal_capabilities_(), STRINGZILLA_NULL, queue) != sz_unexpected_dimensions_k)
            fail_backend_("dispatched", "a query past the verticals was not refused");
        if (engine.memory != STRINGZILLA_NULL) fail_backend_("dispatched", "a refused build still kept a block");
    }

    std::array<sz_string_view_t, 1> const empty_views {sz_string_view_t {STRINGZILLA_NULL, 0}};
    sz_sequence_t empty {};
    sz_sequence_from_string_views(empty_views.data(), empty_views.size(), &empty);
    sz_levenshtein_engine_t engine {};
    if (sz_levenshtein_engine_init(&engine, &empty, sz_levenshtein_bytes_k, metal_capabilities_(), STRINGZILLA_NULL,
                                   queue) != sz_unexpected_dimensions_k)
        fail_backend_("dispatched", "an empty query was not refused");

    // The wavefront waits on neighbouring threadgroups, which Apple GPUs never
    // promise to co-schedule, so Metal lists no kernel for it.
    if (sz_levenshtein_distance_tiled_best("abc", 3, "abd", 3, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                           metal_capabilities_(), queue) != sz_missing_kernel_k)
        fail_backend_("dispatched", "the tiled wavefront did not refuse a Metal device");
}

/** The scoring verb commits and returns, so a round big enough to outlive the call is still
 *  running after it, and lands once the queue is synchronized. */
static void check_levenshtein_metal_asynchrony_(void *queue) {
    levenshtein_metal_corpus_t corpus(queue, 2048, 1024, levenshtein_metal_sweep_queries_k,
                                      levenshtein_metal_alphabet_t::bytes_k);
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init(&engine, &corpus.queries, sz_levenshtein_bytes_k, metal_capabilities_(),
                                      STRINGZILLA_NULL, queue) == sz_success_k);
    verify(sz_levenshtein_distances(&engine, &corpus.candidates.sequence, corpus.distances.data(), corpus.count(),
                                    queue) == sz_success_k);
    std::size_t landed_before = 0;
    for (sz_size_t const distance : corpus.distances) landed_before += distance != STRINGZILLA_SIZE_MAX;
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    sz_levenshtein_engine_free(&engine, queue);
    if (landed_before == corpus.distances.size())
        fmt::println(stderr, "    note: the round finished before the verb returned; asynchrony was not observable");

    std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus, sz_levenshtein_bytes_k);
    for (std::size_t index = 0; index != expected.size(); ++index)
        if (corpus.distances[index] != expected[index])
            fail_backend_("dispatched", "an asynchronous round's distance differs from serial");
}

/** Two threads scoring on two queues of one device at once, each joining only its own: encoders
 *  live per call, and the device's shared state sits under one lock. */
static void check_levenshtein_metal_threads_(void *queue) {
    void *second = nullptr;
    verify(sz_metal_stream_init(0, &second) == sz_success_k);
    auto const round = [](void *stream) {
        levenshtein_metal_corpus_t corpus(stream, 2048, 300, levenshtein_metal_sweep_queries_k,
                                          levenshtein_metal_alphabet_t::bytes_k);
        sz_levenshtein_engine_t engine {};
        verify(sz_levenshtein_engine_init(&engine, &corpus.queries, sz_levenshtein_bytes_k, metal_capabilities_(),
                                          STRINGZILLA_NULL, stream) == sz_success_k);
        verify(sz_levenshtein_distances(&engine, &corpus.candidates.sequence, corpus.distances.data(), corpus.count(),
                                        stream) == sz_success_k);
        verify(sz_stream_synchronize_metal(stream) == sz_success_k);
        sz_levenshtein_engine_free(&engine, stream);
        std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus, sz_levenshtein_bytes_k);
        for (std::size_t index = 0; index != expected.size(); ++index)
            if (corpus.distances[index] != expected[index])
                fail_backend_("dispatched", "a round beside another thread's differs from serial");
    };
    std::thread other(round, second);
    round(queue);
    other.join();
    verify(sz_metal_stream_free(second) == sz_success_k);
}

/** A tape freed while a round still reads it lives until its queue is joined, and is gone from the
 *  registry after, so a later round over it is refused rather than reading freed memory. */
static void check_levenshtein_metal_deferred_free_(void *queue) {
    levenshtein_metal_corpus_t corpus(queue, 2048, 1024, levenshtein_metal_sweep_queries_k,
                                      levenshtein_metal_alphabet_t::bytes_k);
    std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus, sz_levenshtein_bytes_k);
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init(&engine, &corpus.queries, sz_levenshtein_bytes_k, metal_capabilities_(),
                                      STRINGZILLA_NULL, queue) == sz_success_k);
    sz_sequence_t const freed = corpus.candidates.sequence;
    verify(sz_levenshtein_distances(&engine, &freed, corpus.distances.data(), corpus.count(), queue) == sz_success_k);
    corpus.candidates.reset();
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    for (std::size_t index = 0; index != expected.size(); ++index)
        if (corpus.distances[index] != expected[index])
            fail_backend_("dispatched", "a round whose tape was freed under it differs from serial");
    if (sz_levenshtein_distances(&engine, &freed, corpus.distances.data(), corpus.count(), queue) !=
        sz_device_memory_mismatch_k)
        fail_backend_("dispatched", "a tape released after the join was still reachable");
    sz_levenshtein_engine_free(&engine, queue);
}

#pragma endregion Levenshtein Checks

#pragma region Overlap Helpers

/**
 *  @brief A corpus in unified memory: its texts as a tape and room for one round's scores.
 *
 *  One tape serves both the device and the serial reference, which reads the kernel's very bytes.
 */
struct overlap_metal_corpus_t {
    std::vector<std::string> queries;
    std::vector<char> texts;
    std::vector<sz_string_view_t> views;
    metal_vector<sz_f32_t> scores;
    sz_sequence_t query_sequence {};
    metal_tape_t candidates;

    overlap_metal_corpus_t(void *queue, std::mt19937 &generator, std::size_t queries_count, std::size_t count,
                           std::size_t query_length, std::size_t widths_count)
        : views(count), scores(queries_count * count * widths_count, -1.0f, metal_unified_alloc<sz_f32_t>(queue)),
          candidates(queue) {
        for (std::size_t index = 0; index != queries_count; ++index) {
            std::string text(query_length ? query_length - index % query_length : 0, '\0');
            randomize_string(generator, text);
            queries.push_back(text);
        }
        std::size_t total = 0;
        for (std::size_t index = 0; index != count; ++index) total += 1 + (index * 37) % 900;
        texts.resize(total);
        randomize_string(generator, texts);
        std::size_t written = 0;
        for (std::size_t index = 0; index != count; ++index) {
            std::size_t const length = 1 + (index * 37) % 900;
            views[index].start = texts.data() + written, views[index].length = length;
            written += length;
        }
        query_sequence = sequence_from_(queries);
        candidates.copy(views);
    }
    std::size_t candidate_stride() const noexcept { return scores.size() / (queries.size() * views.size()); }
    std::size_t query_stride() const noexcept { return views.size() * candidate_stride(); }
};

/** The serial backend's answers for the same corpus, read off the very tape the device reads. */
static std::vector<sz_f32_t> overlap_serial_reference_(overlap_metal_corpus_t const &corpus,
                                                       sz::span<sz_size_t const> widths) {
    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init_serial(&engine, &corpus.query_sequence, widths.data(), widths.size(), 0,
                                         STRINGZILLA_NULL, nullptr) == sz_success_k);
    std::vector<sz_f32_t> expected(corpus.scores.size(), -1.0f);
    verify(sz_overlap_scores_serial(&engine, &corpus.candidates.sequence, expected.data(), corpus.query_stride(),
                                    corpus.candidate_stride(), nullptr) == sz_success_k);
    sz_overlap_engine_free(&engine, nullptr);
    return expected;
}

#pragma endregion Overlap Helpers

#pragma region Overlap Checks

/** One backend's scores against serial's, bit for bit, over widths from narrowest to widest. */
static void check_overlap_metal_equivalence_(void *queue, std::mt19937 &generator, metal_backend_t const &backend) {
    std::array<std::array<sz_size_t, 3>, 2> const width_sets {{{4, 6, 8}, {1, 17, 31}}};
    for (auto const &widths : width_sets)
        for (std::size_t queries_count : {1u, 3u})
            for (std::size_t count : {1u, 129u, 4096u})
                for (std::size_t query_length : {12u, 777u}) {
                    overlap_metal_corpus_t corpus(queue, generator, queries_count, count, query_length, widths.size());
                    std::vector<sz_f32_t> const expected = overlap_serial_reference_(corpus,
                                                                                     {widths.data(), widths.size()});
                    sz_overlap_engine_t engine {};
                    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, widths.data(), widths.size(), count,
                                               metal_capabilities_(), STRINGZILLA_NULL, queue) != sz_success_k)
                        fail_backend_(backend.name, "a device engine was refused for a well-formed batch");
                    if (backend.overlap_scores(&engine, &corpus.candidates.sequence, corpus.scores.data(),
                                               corpus.query_stride(), corpus.candidate_stride(), queue) != sz_success_k)
                        fail_backend_(backend.name, "a unified-memory batch was refused");
                    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
                    sz_overlap_engine_free(&engine, queue);
                    for (std::size_t slot = 0; slot != expected.size(); ++slot)
                        if (corpus.scores[slot] != expected[slot])
                            fail_backend_(backend.name, "a share differs from serial");
                }
}

/** One backend refusing unregistered memory, and sequences it cannot read, before encoding. */
static void check_overlap_metal_memory_safety_(void *queue, std::mt19937 &generator, metal_backend_t const &backend) {
    std::array<sz_size_t, 2> const widths {4, 6};
    overlap_metal_corpus_t corpus(queue, generator, 1, 8, 333, widths.size());
    sz_overlap_engine_t engine {};
    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, widths.data(), widths.size(), 0, metal_capabilities_(),
                               STRINGZILLA_NULL, queue) != sz_success_k)
        fail_backend_(backend.name, "a device engine was refused for a well-formed batch");

    std::string text(64, '\0');
    randomize_string(generator, text);
    std::array<sz_string_view_t, 1> const host_views {sz_string_view_t {text.data(), text.size()}};
    sz_sequence_t host_candidates {};
    sz_sequence_from_string_views(host_views.data(), host_views.size(), &host_candidates);
    std::vector<sz_f32_t> host_scores(corpus.scores.size(), -1.0f);
    if (backend.overlap_scores(&engine, &corpus.candidates.sequence, host_scores.data(), corpus.query_stride(),
                               corpus.candidate_stride(), queue) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host scores were not refused");
    if (backend.overlap_scores(&engine, &host_candidates, corpus.scores.data(), corpus.query_stride(),
                               corpus.candidate_stride(), queue) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "a host-resident view array was not refused");
    sz_sequence_t const foreign = foreign_tape_(corpus.candidates.sequence);
    if (backend.overlap_scores(&engine, &foreign, corpus.scores.data(), corpus.query_stride(),
                               corpus.candidate_stride(), queue) != sz_device_code_mismatch_k)
        fail_backend_(backend.name, "a sequence with foreign accessors was not refused");

    // The round block was sized for a candidates budget when the engine was built, so a round past
    // it is refused rather than grown under a queue that may still be reading it.
    sz_overlap_engine_t narrow {};
    verify(sz_overlap_engine_init(&narrow, &corpus.query_sequence, widths.data(), widths.size(),
                                  corpus.views.size() - 1, metal_capabilities_(), STRINGZILLA_NULL,
                                  queue) == sz_success_k);
    if (backend.overlap_scores(&narrow, &corpus.candidates.sequence, corpus.scores.data(), corpus.query_stride(),
                               corpus.candidate_stride(), queue) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a round past the candidates budget was not refused");
    sz_overlap_engine_free(&narrow, queue);
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    for (sz_f32_t const untouched : host_scores)
        if (untouched != -1.0f) fail_backend_(backend.name, "a refused call still wrote a score");
    for (sz_f32_t const untouched : corpus.scores)
        if (untouched != -1.0f) fail_backend_(backend.name, "a refused call still wrote a score");
    sz_overlap_engine_free(&engine, queue);
}

/** The widest window the per-thread ring holds, and the refusals one step past either bound. */
static void check_overlap_metal_width_safety_(void *queue, std::mt19937 &generator, metal_backend_t const &backend) {
    std::array<sz_size_t, 1> const widest {sz_overlap_gpu_widest_window_k};
    std::array<sz_size_t, 1> const past {sz_overlap_gpu_widest_window_k + 1};
    std::array<sz_size_t, sz_overlap_gpu_widths_max_k + 1> too_many {};
    for (std::size_t index = 0; index != too_many.size(); ++index) too_many[index] = index + 1;

    // The one candidate is the query's own bytes, so the text scores against itself.
    overlap_metal_corpus_t corpus(queue, generator, 1, 1, 256, widest.size());
    corpus.texts.assign(corpus.queries.front().begin(), corpus.queries.front().end());
    corpus.views[0].start = corpus.texts.data(), corpus.views[0].length = corpus.texts.size();
    corpus.candidates.copy(corpus.views);
    sz_overlap_engine_t engine {};
    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, past.data(), past.size(), 0, metal_capabilities_(),
                               STRINGZILLA_NULL, queue) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "a width past the ring was not refused");
    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, too_many.data(), too_many.size(), 0,
                               metal_capabilities_(), STRINGZILLA_NULL, queue) != sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "more widths than a thread holds were not refused");
    if (sz_overlap_engine_init(&engine, &corpus.query_sequence, widest.data(), widest.size(), 0, metal_capabilities_(),
                               STRINGZILLA_NULL, queue) != sz_success_k)
        fail_backend_(backend.name, "the widest window the ring holds was refused");
    if (backend.overlap_scores(&engine, &corpus.candidates.sequence, corpus.scores.data(), corpus.query_stride(),
                               corpus.candidate_stride(), queue) != sz_success_k)
        fail_backend_(backend.name, "a unified-memory batch was refused");
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    sz_overlap_engine_free(&engine, queue);
    if (corpus.scores[0] != 1.0f) fail_backend_(backend.name, "a text does not fully overlap itself");
}

/** The scoring verb commits and returns, so a round big enough to outlive the call is still
 *  running after it, and lands once the queue is synchronized. */
static void check_overlap_metal_asynchrony_(void *queue, std::mt19937 &generator) {
    std::array<sz_size_t, 3> const widths {4, 6, 8};
    overlap_metal_corpus_t corpus(queue, generator, 8, 4096, 777, widths.size());
    std::vector<sz_f32_t> const expected = overlap_serial_reference_(corpus, {widths.data(), widths.size()});
    sz_overlap_engine_t engine {};
    verify(sz_overlap_engine_init(&engine, &corpus.query_sequence, widths.data(), widths.size(), corpus.views.size(),
                                  metal_capabilities_(), STRINGZILLA_NULL, queue) == sz_success_k);
    verify(sz_overlap_scores(&engine, &corpus.candidates.sequence, corpus.scores.data(), corpus.query_stride(),
                             corpus.candidate_stride(), queue) == sz_success_k);
    std::size_t landed_before = 0;
    for (sz_f32_t const score : corpus.scores) landed_before += score != -1.0f;
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    sz_overlap_engine_free(&engine, queue);
    if (landed_before == corpus.scores.size())
        fmt::println(stderr, "    note: the round finished before the verb returned; asynchrony was not observable");
    for (std::size_t slot = 0; slot != expected.size(); ++slot)
        if (corpus.scores[slot] != expected[slot]) fail_backend_("dispatched", "an asynchronous round differs");
}

#pragma endregion Overlap Checks

#pragma region Substrings Helpers

/** Matches one Metal round may emit in these checks, which keeps several engines
 *  on one device. */
enum { substrings_metal_matches_budget_k = 1 << 18 };

/** Haystacks one Metal round may carry in these checks. */
enum { substrings_metal_haystacks_budget_k = 1 << 13 };

/** Texts both sides address: their bytes and views on the host, and the tape the device reads,
 *  which the serial reference calls as well. */
struct substrings_metal_texts_t {

    /** Every text's bytes, back to back. */
    std::vector<char> arena;

    /** One view per text. */
    std::vector<sz_string_view_t> views;

    /** The texts' tape in unified memory. */
    metal_tape_t tape;

    substrings_metal_texts_t(void *queue, std::vector<std::string> const &texts) : views(texts.size()), tape(queue) {
        for (std::size_t index = 0; index != texts.size(); ++index) {
            views[index].length = texts[index].size();
            arena.insert(arena.end(), texts[index].begin(), texts[index].end());
        }
        // The arena's address is final only once it stops growing, so the starts
        // are filled afterwards.
        std::size_t written = 0;
        for (sz_string_view_t &view : views) view.start = arena.data() + written, written += view.length;
        tape.copy(views);
    }
};

/** One vocabulary compiled twice under one policy: for the host and for the device. */
struct substrings_metal_engines_t {

    /** What the serial oracle's engine is built from. */
    handle_checked_heap_t heap;

    /** The serial oracle's engine, in plain host memory. */
    sz_substrings_engine_t host {};

    /** The device's engine, round block and report included. */
    sz_substrings_engine_t device {};

    /** The queue the device engine was built on and is freed on. */
    void *queue;

    substrings_metal_engines_t(void *queue, sz_sequence_t const &needles, sz_substrings_overlap_policy_t policy)
        : queue(queue) {
        verify(sz_substrings_engine_init_serial(&host, &needles, sz_substrings_cased_k, policy,
                                                STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0, &heap.allocator,
                                                nullptr) == sz_success_k);
        verify(sz_substrings_engine_init(&device, &needles, sz_substrings_cased_k, policy,
                                         STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, substrings_metal_matches_budget_k,
                                         substrings_metal_haystacks_budget_k, metal_capabilities_(), STRINGZILLA_NULL,
                                         queue) == sz_success_k);
    }
    substrings_metal_engines_t(substrings_metal_engines_t const &) = delete;
    substrings_metal_engines_t &operator=(substrings_metal_engines_t const &) = delete;
    ~substrings_metal_engines_t() noexcept {
        sz_substrings_engine_free(&device, queue);
        sz_substrings_engine_free(&host, nullptr);
        verify(heap.live_allocations == 0);
    }
};

/** One match, ordered so two backends' reports compare as sequences rather than as multisets. */
struct substrings_metal_case_t {
    sz_size_t haystack_index {}, needle_index {}, byte_offset {}, byte_length {};

    bool operator==(substrings_metal_case_t const &other) const noexcept {
        return haystack_index == other.haystack_index && needle_index == other.needle_index &&
               byte_offset == other.byte_offset && byte_length == other.byte_length;
    }
    bool operator<(substrings_metal_case_t const &other) const noexcept {
        if (haystack_index != other.haystack_index) return haystack_index < other.haystack_index;
        if (byte_offset != other.byte_offset) return byte_offset < other.byte_offset;
        if (byte_length != other.byte_length) return byte_length < other.byte_length;
        return needle_index < other.needle_index;
    }
};

/** Sorts one verb's match array into the shape two backends compare position by position. */
static std::vector<substrings_metal_case_t> sorted_(sz_substrings_match_t const *matches, sz_size_t count) {
    std::vector<substrings_metal_case_t> reported(count);
    for (std::size_t index = 0; index != count; ++index)
        reported[index] = {matches[index].haystack_index, matches[index].needle_index, matches[index].byte_offset,
                           matches[index].byte_length};
    std::sort(reported.begin(), reported.end());
    return reported;
}

/** Every match the serial tier reports, sized from the report its own sizing call leaves. */
static std::vector<substrings_metal_case_t> serial_matches_(sz_substrings_engine_t *engine,
                                                            sz_sequence_t const *haystacks) {
    std::vector<sz_size_t> offsets(haystacks->count + 1, 0);
    verify(sz_substrings_find_serial(engine, haystacks, nullptr, 0, offsets.data(), nullptr) == sz_success_k);
    std::vector<sz_substrings_match_t> matches(engine->report->matches_emitted);
    verify(sz_substrings_find_serial(engine, haystacks, matches.data(), matches.size(), offsets.data(), nullptr) ==
           sz_success_k);
    verify(engine->report->shortfall == 0);
    return sorted_(matches.data(), matches.size());
}

/** Every match the device reports, synchronized per call since no verb waits for the caller. */
static std::vector<substrings_metal_case_t> device_matches_(void *queue, metal_backend_t const &backend,
                                                            sz_substrings_engine_t *engine,
                                                            sz_sequence_t const *haystacks) {
    metal_vector<sz_size_t> offsets(haystacks->count + 1, 0, metal_unified_alloc<sz_size_t>(queue));
    verify(backend.substrings_find(engine, haystacks, nullptr, 0, offsets.data(), queue) == sz_success_k);
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    sz_size_t const total = offsets[haystacks->count];
    verify(engine->report->matches_stored + engine->report->shortfall == total);

    metal_vector<sz_substrings_match_t> matches(total, metal_unified_alloc<sz_substrings_match_t>(queue));
    verify(backend.substrings_find(engine, haystacks, matches.data(), matches.size(), offsets.data(), queue) ==
           sz_success_k);
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    verify(engine->report->matches_stored == total && engine->report->shortfall == 0);
    return sorted_(matches.data(), total);
}

/** That every reported match is real, and that no two of them share a byte of one haystack. */
static void verify_is_a_cover_(std::vector<substrings_metal_case_t> const &reported,
                               std::vector<substrings_metal_case_t> const &every_match) {
    std::size_t candidate = 0;
    for (substrings_metal_case_t const &match : reported) {
        while (candidate != every_match.size() && every_match[candidate] < match) ++candidate;
        verify(candidate != every_match.size() && every_match[candidate] == match &&
               "A cover reports only matches the overlapping walk found");
    }
    for (std::size_t index = 1; index < reported.size(); ++index) {
        substrings_metal_case_t const &previous = reported[index - 1], &current = reported[index];
        if (previous.haystack_index != current.haystack_index) continue;
        verify(current.byte_offset >= previous.byte_offset + previous.byte_length &&
               "A cover's matches share no bytes");
    }
    // A cover leaves a match out only for overlapping a kept one, which an empty or thinned
    // report would not.
    std::size_t kept = 0;
    for (substrings_metal_case_t const &match : every_match) {
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

/** The device's matches, counts and rewrite against serial's, over one corpus and policy. */
static void check_against_serial_(void *queue, metal_backend_t const &backend, substrings_metal_texts_t &corpus,
                                  substrings_metal_texts_t &vocabulary, sz_substrings_overlap_policy_t policy,
                                  sz_substrings_cover_fidelity_t fidelity) {
    std::size_t const haystacks_count = corpus.views.size();
    sz_sequence_t const *const haystacks = &corpus.tape.sequence;
    substrings_metal_engines_t engines(queue, vocabulary.tape.sequence, policy);
    std::vector<substrings_metal_case_t> const expected = serial_matches_(&engines.host, haystacks);
    std::vector<substrings_metal_case_t> const reported = device_matches_(queue, backend, &engines.device, haystacks);
    if (fidelity == sz_substrings_cover_exact_k) {
        verify(reported.size() == expected.size());
        for (std::size_t index = 0; index != expected.size(); ++index) verify(reported[index] == expected[index]);
    }
    else {
        substrings_metal_engines_t overlapping(queue, vocabulary.tape.sequence, sz_substrings_overlapping_k);
        verify_is_a_cover_(reported, serial_matches_(&overlapping.host, haystacks));
    }

    // The counts come from the boundaries rather than from the match list, so they can
    // disagree with it.
    metal_vector<sz_size_t> device_counts(haystacks_count, 0, metal_unified_alloc<sz_size_t>(queue));
    std::vector<sz_size_t> serial_counts(haystacks_count, 0);
    verify(sz_substrings_counts_serial(&engines.host, haystacks, serial_counts.data(), 1, nullptr) == sz_success_k);
    verify(backend.substrings_counts(&engines.device, haystacks, device_counts.data(), 1, queue) == sz_success_k);
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    sz_size_t counted = 0;
    for (std::size_t index = 0; index != haystacks_count; ++index) {
        if (fidelity == sz_substrings_cover_exact_k) verify(device_counts[index] == serial_counts[index]);
        counted += device_counts[index];
    }
    verify(counted == reported.size() && "The counts and the matches come from one pass");
    if (policy == sz_substrings_overlapping_k || fidelity != sz_substrings_cover_exact_k) return;

    // The rewrite, whose device path owns a threadgroup scan and a tiled copy the serial one has
    // no analogue for.
    std::vector<std::string> replacement_texts;
    for (std::size_t index = 0; index != vocabulary.views.size(); ++index)
        replacement_texts.push_back(index % 3 == 0 ? std::string() : "<" + std::to_string(index) + ">");
    substrings_metal_texts_t replacements(queue, replacement_texts);
    metal_vector<sz_size_t> device_offsets(haystacks_count + 1, 0, metal_unified_alloc<sz_size_t>(queue));
    std::vector<sz_size_t> serial_offsets(haystacks_count + 1, 0);
    verify(backend.substrings_replace(&engines.device, haystacks, &replacements.tape.sequence, nullptr, 0,
                                      device_offsets.data(), queue) == sz_success_k);
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    verify(sz_substrings_replace_serial(&engines.host, haystacks, &replacements.tape.sequence, nullptr, 0,
                                        serial_offsets.data(), nullptr) == sz_success_k);
    for (std::size_t index = 0; index != haystacks_count + 1; ++index)
        verify(device_offsets[index] == serial_offsets[index]);

    sz_size_t const rewritten = serial_offsets[haystacks_count];
    metal_vector<char> device_tape(rewritten + 1, 0, metal_unified_alloc<char>(queue));
    std::vector<char> serial_tape(rewritten + 1, 0);
    verify(backend.substrings_replace(&engines.device, haystacks, &replacements.tape.sequence, device_tape.data(),
                                      rewritten, device_offsets.data(), queue) == sz_success_k);
    verify(sz_stream_synchronize_metal(queue) == sz_success_k);
    verify(engines.device.report->shortfall == 0);
    verify(sz_substrings_replace_serial(&engines.host, haystacks, &replacements.tape.sequence, serial_tape.data(),
                                        rewritten, serial_offsets.data(), nullptr) == sz_success_k);
    for (std::size_t index = 0; index != rewritten; ++index) verify(device_tape[index] == serial_tape[index]);
}

/** The device's BM25 against serial's, by byte lengths and by caller-given ones: the device sums in
 *  @c f32 terms and fixed point, so the two agree to rounding rather than bit for bit. */
static void check_bm25_against_serial_(void *queue, metal_backend_t const &backend, substrings_metal_texts_t &corpus,
                                       substrings_metal_texts_t &vocabulary) {
    std::size_t const haystacks_count = corpus.views.size();
    sz_sequence_t const *const haystacks = &corpus.tape.sequence;
    substrings_metal_engines_t engines(queue, vocabulary.tape.sequence, sz_substrings_overlapping_k);
    std::size_t const needles_count = engines.host.needles_count;
    metal_vector<sz_f32_t> weights(needles_count, 0, metal_unified_alloc<sz_f32_t>(queue));
    metal_vector<sz_f32_t> given_lengths(haystacks_count, 0, metal_unified_alloc<sz_f32_t>(queue));
    for (std::size_t index = 0; index != needles_count; ++index) weights[index] = 0.5f + (float)(index % 7) * 0.25f;
    double bytes_total = 0, given_total = 0;
    for (std::size_t index = 0; index != haystacks_count; ++index) {
        given_lengths[index] = (sz_f32_t)(1 + index % 5);
        bytes_total += corpus.views[index].length, given_total += given_lengths[index];
    }

    for (sz_f32_t const *lengths : {(sz_f32_t const *)nullptr, (sz_f32_t const *)given_lengths.data()}) {
        double const average = (lengths ? given_total : bytes_total) / haystacks_count;
        sz_substrings_bm25_t const parameters {1.2f, 0.75f, (sz_f32_t)(average > 0 ? average : 1)};
        std::vector<sz_f32_t> serial_scores(haystacks_count, -1);
        metal_vector<sz_f32_t> device_scores(haystacks_count, -1, metal_unified_alloc<sz_f32_t>(queue));
        verify(sz_substrings_bm25_scores_serial(&engines.host, haystacks, lengths, &parameters, weights.data(),
                                                serial_scores.data(), 1, nullptr) == sz_success_k);
        verify(backend.substrings_bm25_scores(&engines.device, haystacks, lengths, &parameters, weights.data(),
                                              device_scores.data(), 1, queue) == sz_success_k);
        verify(sz_stream_synchronize_metal(queue) == sz_success_k);
        for (std::size_t index = 0; index != haystacks_count; ++index) {
            double const tolerance = 1e-5 * std::max(1.0, std::fabs((double)serial_scores[index]));
            verify(std::fabs(device_scores[index] - serial_scores[index]) <= tolerance);
        }
    }
}

/** One vocabulary against one corpus under every policy, and scored. */
static void check_policies_(void *queue, metal_backend_t const &backend, std::vector<std::string> const &haystacks,
                            std::vector<std::string> const &needles,
                            sz_substrings_cover_fidelity_t fidelity = sz_substrings_cover_exact_k) {
    substrings_metal_texts_t corpus(queue, haystacks), vocabulary(queue, needles);
    check_against_serial_(queue, backend, corpus, vocabulary, sz_substrings_overlapping_k, sz_substrings_cover_exact_k);
    check_against_serial_(queue, backend, corpus, vocabulary, sz_substrings_leftmost_longest_k, fidelity);
    check_against_serial_(queue, backend, corpus, vocabulary, sz_substrings_leftmost_first_k, fidelity);
    check_bm25_against_serial_(queue, backend, corpus, vocabulary);
}

#pragma endregion Substrings Helpers

#pragma region Substrings Checks

/** The chunk-boundary cases a single-chain host walk cannot express, and random corpora wide
 *  enough that the planner cuts several chunks per haystack, on @p queue. */
static void test_substrings_metal_equivalence(test_context_t &context, void *queue, metal_backend_t const &backend) {
    std::mt19937 &generator = context.generator;
    check_policies_(queue, backend, {"ushers"}, {"he", "she", "his", "hers"});

    char const *const alphabets[] = {"ab", "abcdefgh", "abcdefghijklmnopqrstuvwxyz"};
    std::size_t const rounds = context.iterations(8);
    for (std::size_t round = 0; round != rounds; ++round) {
        char const *const alphabet = alphabets[round % 3];
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 1 + (round * 7) % 24; ++index)
            needles.push_back(random_string(generator, 1 + (index * 3 + round) % 7, alphabet));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 1 + (round * 5) % 9; ++index)
            haystacks.push_back(random_string(generator, 4096 + index * 977, alphabet));
        check_policies_(queue, backend, haystacks, needles, sz_substrings_cover_approximate_k);
    }

    // A vocabulary wider than a threadgroup's tally, so needles hash into its slots and
    // spill past them.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 6000; ++index)
            needles.push_back(random_string(generator, 2 + index % 4, "abcdefghijklmnopqrstuvwxyz"));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 64; ++index)
            haystacks.push_back(random_string(generator, 256 + index * 97, "abcdefghijklmnopqrstuvwxyz"));
        for (std::size_t index = 0; index != 4096; ++index)
            haystacks.push_back(random_string(generator, 16 + index % 97, "abcdefghijklmnopqrstuvwxyz"));
        // A haystack spelling every needle seats more of them than a tally holds, twice in
        // one threadgroup's run.
        std::string every_needle;
        for (std::string const &needle : needles) every_needle += needle + "#";
        haystacks[0] = every_needle;
        haystacks.insert(haystacks.begin() + 64, every_needle);
        substrings_metal_texts_t corpus(queue, haystacks), vocabulary(queue, needles);
        verify(vocabulary.views.size() > 2048);
        check_bm25_against_serial_(queue, backend, corpus, vocabulary);
    }

    // One haystack far wider than a chunk, so the boundary case is hit many times over in a
    // single walk. No needle reaches across another's end, so the cover is exact and the rewrite
    // spans many tiles.
    check_policies_(queue, backend, {random_string(generator, 1 << 20, "the ")}, {"the", "there", "here", "her", "he"});
}

/** What the device verbs refuse, and what the report says when an output cannot hold the answer. */
static void test_substrings_metal_safety(void *queue, metal_backend_t const &backend) {
    substrings_metal_texts_t corpus(queue, {"abcdabcd"}), vocabulary(queue, {"ab", "cd"});
    substrings_metal_engines_t engines(queue, vocabulary.tape.sequence, sz_substrings_overlapping_k);
    sz_sequence_t const *const haystacks = &corpus.tape.sequence;

    // Folding walks the host tiers' Unicode tables, which have no Metal port, so it is
    // refused at init.
    {
        sz_substrings_engine_t uncased {};
        verify(sz_substrings_engine_init(&uncased, &vocabulary.tape.sequence, sz_substrings_uncased_k,
                                         sz_substrings_overlapping_k, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0,
                                         metal_capabilities_(), STRINGZILLA_NULL, queue) == sz_device_code_mismatch_k);
        verify(uncased.memory == STRINGZILLA_NULL);
    }

    // A device verb refuses memory outside the registry rather than reading it from a kernel.
    {
        std::vector<sz_size_t> host_counts(1, 0);
        verify(backend.substrings_counts(&engines.device, haystacks, host_counts.data(), 1, queue) ==
               sz_device_memory_mismatch_k);
    }

    // An output stride of zero cannot address one entry per haystack, whatever the haystack count.
    {
        metal_vector<sz_size_t> counts(1, STRINGZILLA_SIZE_MAX, metal_unified_alloc<sz_size_t>(queue));
        verify(backend.substrings_counts(&engines.device, haystacks, counts.data(), 0, queue) ==
               sz_unexpected_dimensions_k);
        verify(counts[0] == STRINGZILLA_SIZE_MAX);
    }

    // A capacity that cannot hold the matches is not an error: the report names the true total and
    // the shortfall beside it.
    {
        metal_vector<sz_size_t> offsets(2, 0, metal_unified_alloc<sz_size_t>(queue));
        verify(backend.substrings_find(&engines.device, haystacks, nullptr, 0, offsets.data(), queue) == sz_success_k);
        verify(sz_stream_synchronize_metal(queue) == sz_success_k);
        verify(engines.device.report->matches_emitted == 4);
        verify(engines.device.report->matches_stored == 0 && engines.device.report->shortfall == 4);
    }

    // A substitution over matches that share bytes is not a function, so the overlapping
    // policy is refused.
    {
        substrings_metal_texts_t replacements(queue, {"x", "y"});
        metal_vector<sz_size_t> offsets(2, 0, metal_unified_alloc<sz_size_t>(queue));
        verify(backend.substrings_replace(&engines.device, haystacks, &replacements.tape.sequence, nullptr, 0,
                                          offsets.data(), queue) == sz_status_unknown_k);
    }

    // A round past its match budget leaves every later kernel retired, so the report alone says it
    // did not fit.
    {
        sz_substrings_engine_t tiny {};
        metal_vector<sz_size_t> offsets(2, STRINGZILLA_SIZE_MAX, metal_unified_alloc<sz_size_t>(queue));
        verify(sz_substrings_engine_init(&tiny, &vocabulary.tape.sequence, sz_substrings_cased_k,
                                         sz_substrings_leftmost_first_k, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 1, 0,
                                         metal_capabilities_(), STRINGZILLA_NULL, queue) == sz_success_k);
        verify(backend.substrings_find(&tiny, haystacks, nullptr, 0, offsets.data(), queue) == sz_success_k);
        verify(sz_stream_synchronize_metal(queue) == sz_success_k);
        verify(tiny.report->matches_emitted == 4 && tiny.report->shortfall == 3);
        verify(offsets[1] == 0 && "A retired offsets kernel leaves the boundaries zeroed");
        sz_substrings_engine_free(&tiny, queue);
    }

    // The round block is sized for a haystacks budget at init, so a round carrying more
    // is refused untouched.
    {
        substrings_metal_texts_t pair(queue, {"abcd", "abcd"});
        sz_substrings_engine_t narrow {};
        metal_vector<sz_size_t> offsets(3, STRINGZILLA_SIZE_MAX, metal_unified_alloc<sz_size_t>(queue));
        verify(sz_substrings_engine_init(&narrow, &vocabulary.tape.sequence, sz_substrings_cased_k,
                                         sz_substrings_overlapping_k, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 1,
                                         metal_capabilities_(), STRINGZILLA_NULL, queue) == sz_success_k);
        verify(backend.substrings_find(&narrow, &pair.tape.sequence, nullptr, 0, offsets.data(), queue) ==
               sz_unexpected_dimensions_k);
        verify(sz_stream_synchronize_metal(queue) == sz_success_k);
        verify(offsets[0] == STRINGZILLA_SIZE_MAX && "A refused round touched the caller's boundaries");
        sz_substrings_engine_free(&narrow, queue);
    }
}

#pragma endregion Substrings Checks

#pragma region Drivers

/** Registers every check of the Metal kernels, or of the dispatch points, in @p check. */
static void check_metal_backend_(cross_section_t &check, void *queue, metal_backend_t const &backend) {
    std::string const suffix = backend.name;
    check("test_levenshtein_equivalence_" + suffix, [&] {
        check_levenshtein_metal_equivalence_(queue, levenshtein_metal_alphabet_t::bytes_k, sz_levenshtein_bytes_k,
                                             backend);
        check_levenshtein_metal_equivalence_(queue, levenshtein_metal_alphabet_t::runes_k, sz_levenshtein_runes_k,
                                             backend);
        check_levenshtein_metal_equivalence_(queue, levenshtein_metal_alphabet_t::bytes_k, sz_levenshtein_runes_k,
                                             backend);
    });
    check("test_levenshtein_safety_" + suffix,
          [&](test_context_t &context) { check_levenshtein_metal_memory_safety_(queue, context.generator, backend); });
    check("test_overlap_equivalence_" + suffix,
          [&](test_context_t &context) { check_overlap_metal_equivalence_(queue, context.generator, backend); });
    check("test_overlap_safety_" + suffix, [&](test_context_t &context) {
        check_overlap_metal_memory_safety_(queue, context.generator, backend);
        check_overlap_metal_width_safety_(queue, context.generator, backend);
    });
    check("test_substrings_equivalence_" + suffix,
          [&](test_context_t &context) { test_substrings_metal_equivalence(context, queue, backend); });
    check("test_substrings_safety_" + suffix, [&] { test_substrings_metal_safety(queue, backend); });
}

std::size_t test_cross_metal(environment_t const &env, void *queue) {
    metal_backend_t const metal {
        "metal",
        sz_levenshtein_distances_metal,
        sz_overlap_scores_metal,
        sz_substrings_counts_metal,
        sz_substrings_find_metal,
        sz_substrings_replace_metal,
        sz_substrings_bm25_scores_metal,
    };
    cross_section_t check(env);
    check.detected = metal_capabilities_();
    check.section("Cross Metal", sz_cap_metal_k);
    check("test_unified_alloc_metal", [&] {
        metal_unified_alloc<sz_size_t> allocator(queue);
        metal_unified_alloc<char> rebound(allocator);
        verify(rebound == metal_unified_alloc<char>(queue));
        verify(allocator != metal_unified_alloc<sz_size_t>());
        metal_vector<sz_size_t> source(2, 42, allocator);
        metal_vector<sz_size_t> target(1, 0, metal_unified_alloc<sz_size_t>());
        target = std::move(source);
        verify(target.get_allocator() == allocator);
        verify(target.size() == 2 && target[0] == 42 && target[1] == 42);
    });
    check("test_sequence_realloc_metal", [&] {
        sz_u64_t host_offsets[] = {2 * sizeof(sz_u64_t), 2 * sizeof(sz_u64_t)};
        sz_sequence_t host {};
        host.handle = host_offsets, host.count = 1;
        host.get_start = sz_sequence_tape_start, host.get_length = sz_sequence_tape_length;
        sz_allocator_t unified;
        verify(sz_allocator_init_unified_metal(&unified) == sz_success_k);
        sz_sequence_t tape {}, borrowed {};
        sz_size_t bytes = 0, borrowed_bytes = 1;
        verify(sz_sequence_realloc_metal(&tape, &host, &unified, &bytes, queue) == sz_success_k);
        verify(bytes == sizeof(host_offsets) && tape.handle != host.handle);
        verify(sz_sequence_realloc_metal(&borrowed, &tape, &unified, &borrowed_bytes, queue) == sz_success_k);
        verify(borrowed.handle == tape.handle && !borrowed_bytes);
        unified.free(const_cast<void *>(tape.handle), bytes, unified.handle, queue);
    });
    check_metal_backend_(check, queue, metal);
    return check.failures;
}

/** The dispatching entry points on @p queue, and the refusals, asynchrony, threading and deferred
 *  frees only a dispatch point's engine init promises. */
std::size_t test_cross_dispatch(environment_t const &env, void *queue) {
    metal_backend_t const dispatched {
        "dispatched",       sz_levenshtein_distances, sz_overlap_scores,         sz_substrings_counts,
        sz_substrings_find, sz_substrings_replace,    sz_substrings_bm25_scores,
    };
    cross_section_t check(env);
    check.detected = metal_capabilities_();
    check.section("Cross Dispatch", sz_cap_metal_k);
    check_metal_backend_(check, queue, dispatched);
    check("test_levenshtein_refusals_dispatched", [&] { check_levenshtein_metal_query_safety_(queue); });
    check("test_levenshtein_asynchrony_dispatched", [&] { check_levenshtein_metal_asynchrony_(queue); });
    check("test_levenshtein_threads_dispatched", [&] { check_levenshtein_metal_threads_(queue); });
    check("test_levenshtein_deferred_free_dispatched", [&] { check_levenshtein_metal_deferred_free_(queue); });
    check("test_overlap_asynchrony_dispatched",
          [&](test_context_t &context) { check_overlap_metal_asynchrony_(queue, context.generator); });
    return check.failures;
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
