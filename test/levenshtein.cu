/**
 *  @file test/levenshtein.cu
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Levenshtein distances on the GPU.
 *
 *  Covers the device engine against serial's answers at every word boundary a rung is cut at, the
 *  memory contract the verbs keep, and the query bound the verticals impose.
 *
 *  @c test/levenshtein.cpp defines @c test_levenshtein_all and @c test_levenshtein_safety over the
 *  CPU backend table, and this file defines them over the CUDA one. No target links both -
 *  @c stringzilla_test_cpp20 takes the first and @c stringzilla_test_cu20 the second - a CMake
 *  invariant rather than a language one.
 *
 *  The sibling @c test/levenshtein.cpp drives the step primitives and the CPU backends; nothing
 *  here repeats that. These are the cases a host translation unit cannot express: device-reachable
 *  memory, a device-bound sequence, a caller's own stream, and the refusals that keep a host
 *  pointer from reaching a kernel as an address.
 *
 *  Query lengths are swept from a table whose status at each length is derived from the widest rung
 *  that length's alphabet reaches, so a length past it is expected to be refused and a raised
 *  constant turns that row into an answer without a line changing here.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <cstddef> // `std::size_t`

#include <array>  // `std::array`
#include <string> // `std::string`
#include <vector> // `std::vector`

#include <fmt/format.h>

#include <stringzilla/levenshtein.h> // `sz_levenshtein_*`
#include <stringzilla/stringzilla.h> // Primary C API

#include "stringzilla.hpp" // `randomize_string`, `verify`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

#pragma region Helpers

/** What a corpus symbol is: a byte the tables class directly, or a rune the page table classes. */
enum class levenshtein_cuda_alphabet_t { bytes_k, runes_k };

/** The alphabet enumerator an engine is prepared over, matching the corpus alphabet of a check. */
static sz_levenshtein_symbol_t levenshtein_cuda_symbol_(levenshtein_cuda_alphabet_t alphabet) {
    return alphabet == levenshtein_cuda_alphabet_t::runes_k ? sz_levenshtein_runes_k : sz_levenshtein_bytes_k;
}

/** Corpus symbols: every byte value, or runes across four scripts and all three encoded widths. */
static std::vector<std::string> levenshtein_cuda_symbols_(levenshtein_cuda_alphabet_t alphabet) {
    std::vector<std::string> symbols;
    if (alphabet == levenshtein_cuda_alphabet_t::bytes_k) {
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
struct levenshtein_cuda_corpus_t {

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

    /** Accessors a kernel calls, as the scoring verb requires. */
    sz_sequence_t device_candidates {};

    /** Accessors the serial reference calls, over the same views. */
    sz_sequence_t host_candidates {};

    levenshtein_cuda_corpus_t(std::size_t count, std::size_t query_symbols, std::size_t query_count,
                              levenshtein_cuda_alphabet_t alphabet)
        : query_views(query_count), views(count), distances(query_count * count) {
        std::vector<std::string> const symbols = levenshtein_cuda_symbols_(alphabet);
        std::vector<std::string> drawn(query_symbols);
        for (std::size_t symbol = 0; symbol != query_symbols; ++symbol)
            drawn[symbol] = symbols[(symbol * 37 + 11) % symbols.size()];

        // Every query is the same length, so one batch still spans one rung, and every one of them differs.
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

        // An arena's address is only final once it has stopped growing, so the starts are filled afterwards.
        std::size_t written = 0;
        for (sz_string_view_t &view : query_views) view.start = query_arena.data() + written, written += view.length;
        written = 0;
        for (sz_string_view_t &view : views) view.start = arena.data() + written, written += view.length;
        sz_sequence_from_string_views(query_views.data(), query_views.size(), &queries);
        verify(sz_sequence_from_string_views_cuda(views.data(), views.size(), &device_candidates) == sz_success_k);
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
    }

    /** Candidates one round scores, which is also the row stride of the matrix it writes. */
    std::size_t count() const { return views.size(); }
};

/** The serial backend's answers for the same corpus, read off the very bytes the device reads. */
static std::vector<sz_size_t> levenshtein_serial_reference_(levenshtein_cuda_corpus_t const &corpus,
                                                            sz_levenshtein_symbol_t symbol) {
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init_cpu(&corpus.queries, symbol, nullptr, &engine) == sz_success_k);
    std::vector<sz_size_t> expected(corpus.distances.size());
    verify(sz_levenshtein_distances_serial(&engine, &corpus.host_candidates, expected.data(), corpus.count()) ==
           sz_success_k);
    sz_levenshtein_engine_free(&engine);
    return expected;
}

#pragma endregion Helpers

#pragma region Backends

/** One CUDA backend's cross-product verb, and the device it needs. */
struct levenshtein_cuda_backend_t {

    /** The row's spelling, for @ref fail_backend_ and the log. */
    char const *name;

    /** The tier bits @ref sz_capabilities must carry for this row to run. */
    sz_capability_t required;

    /** The verb, over bytes or over runes as the engine was built. */
    sz_levenshtein_distances_t distances;
};

/**
 *  @brief Every CUDA backend compiled into this translation unit, dispatched first.
 *
 *  Unlike the CPU tables, whose rows a `#if` selects, every CUDA tier is compiled into one fatbin
 *  and chosen at run time - so a row states the tier bits it needs and the drivers skip it on a
 *  device that lacks them.
 */
static levenshtein_cuda_backend_t const levenshtein_cuda_backends[] = {
    {"dispatched", sz_cap_cuda_k, sz_levenshtein_distances},
    {"cuda", sz_cap_cuda_k, sz_levenshtein_distances_cuda},
};

#pragma endregion Backends

#pragma region Expectations

/**
 *  @brief The status a query of @p query_symbols symbols draws, as the widest rung decides it.
 *
 *  Both alphabets reach the warped rung, whose thirty-two lanes hold
 *  @ref sz_levenshtein_cuda_words_max_k words between them; a rune query gets there by riding its
 *  class down the lanes rather than by indexing the candidate.
 */
static constexpr sz_status_t levenshtein_cuda_expected_status_(std::size_t query_symbols) {
    return query_symbols <= (std::size_t)sz_levenshtein_cuda_words_max_k * 64 ? sz_success_k
                                                                              : sz_unexpected_dimensions_k;
}

/** Symbol-pairs one sweep scores before it drops the widest batch, so the serial reference stays
 *  quadratic in time but bounded in it as the rungs widen. */
enum { levenshtein_cuda_sweep_budget_k = 4u * 1024u * 1024u };

/** Queries one sweep prepares at a time, so @c grid.y carries an axis wider than one per rung. */
enum { levenshtein_cuda_sweep_queries_k = 3 };

/**
 *  @brief Every query length the sweeps walk, either side of each boundary a rung is cut at.
 *
 *  The narrow rungs share one register between candidates, four ways through eight symbols and two
 *  ways through sixteen, so the symbols either side of 8 and 16. The threaded rung cuts at every
 *  word through its sixteenth, where the warped rung takes over, so the bytes either side of 64,
 *  128 and 1024. The warped rung hands its lanes one more word each at every thirty-second word
 *  past that, so the bytes either side of every multiple of 2048 up to the ceiling itself.
 */
static constexpr std::size_t levenshtein_cuda_query_symbols_k[] = {
    1,    2,    4,    8,    9,    16,   17,   63,   64,    65,    127,   128,   129,   1023,  1024,  1025,  2047,  2048,
    2049, 4095, 4096, 4097, 6144, 6145, 8192, 8193, 10240, 10241, 12288, 12289, 14336, 14337, 16383, 16384, 16385,
};

#pragma endregion Expectations

#pragma region Checks

/** One backend's distances against serial's, on a corpus that never leaves the device. */
static void check_levenshtein_cuda_equivalence_(char const *name, levenshtein_cuda_alphabet_t alphabet,
                                                sz_levenshtein_distances_t device) {
    sz_levenshtein_symbol_t const symbol = levenshtein_cuda_symbol_(alphabet);
    for (std::size_t const query_symbols : levenshtein_cuda_query_symbols_k)
        for (std::size_t count : {1u, 129u, 2048u}) {
            std::size_t const queries = levenshtein_cuda_sweep_queries_k;
            if (count * query_symbols * queries > (std::size_t)levenshtein_cuda_sweep_budget_k) continue;
            levenshtein_cuda_corpus_t corpus(count, query_symbols, queries, alphabet);

            sz_levenshtein_engine_t engine {};
            sz_status_t const prepared = sz_levenshtein_engine_init_gpu(&corpus.queries, symbol, SZ_NULL, SZ_NULL,
                                                                        &engine);
            if (prepared != levenshtein_cuda_expected_status_(query_symbols))
                fail_backend_(name, "a device batch drew a status the rungs do not imply");
            if (prepared != sz_success_k) continue;
            sz_status_t const produced = device(&engine, &corpus.device_candidates, corpus.distances.data(),
                                                corpus.count());
            // The planes the round reads live in the engine's block, so the join comes before it is released.
            verify(cudaStreamSynchronize((cudaStream_t)SZ_NULL) == cudaSuccess);
            sz_levenshtein_engine_free(&engine);
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
static void check_levenshtein_cuda_narrow_lanes_(char const *name, sz_levenshtein_distances_t device) {
    enum { arena_bytes_k = 256, longest_candidate_k = 33, offsets_k = 97 };
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);

    std::size_t const count = sz_levenshtein_cuda_lanes_candidates_min_();
    if (count == SZ_SIZE_MAX || count == 0) return;
    for (std::size_t const query_symbols : {(std::size_t)8, (std::size_t)16}) {

        // Every third byte is one of the query's own, so a candidate draws both classed and absent symbols.
        unified_vector<char> query(query_symbols), arena(arena_bytes_k);
        randomize_string(query.data(), query.size());
        randomize_string(arena.data(), arena.size());
        for (std::size_t index = 0; index < arena.size(); index += 3) arena[index] = query[index % query.size()];

        unified_vector<sz_string_view_t> query_views(1), views(count);
        unified_vector<sz_size_t> distances(count);
        query_views[0].start = query.data(), query_views[0].length = query.size();
        for (std::size_t index = 0; index != count; ++index)
            views[index].start = arena.data() + index % offsets_k, views[index].length = index % longest_candidate_k;
        sz_sequence_t queries {}, candidates {};
        sz_sequence_from_string_views(query_views.data(), query_views.size(), &queries);
        verify(sz_sequence_from_string_views_cuda(views.data(), views.size(), &candidates) == sz_success_k);

        sz_levenshtein_engine_t engine {};
        verify(sz_levenshtein_engine_init_gpu(&queries, sz_levenshtein_bytes_k, SZ_NULL, SZ_NULL, &engine) ==
               sz_success_k);
        sz_status_t const produced = device(&engine, &candidates, distances.data(), count);
        // The planes the round reads live in the engine's block, so the join comes before it is released.
        verify(cudaStreamSynchronize((cudaStream_t)SZ_NULL) == cudaSuccess);
        sz_levenshtein_engine_free(&engine);
        if (produced != sz_success_k) fail_backend_(name, "a batch wide enough for the narrow rungs was refused");

        sz_sequence_t host_candidates {};
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
        sz_levenshtein_engine_t host_engine {};
        verify(sz_levenshtein_engine_init_cpu(&queries, sz_levenshtein_bytes_k, &host, &host_engine) == sz_success_k);
        std::vector<sz_size_t> expected(count);
        verify(sz_levenshtein_distances_serial(&host_engine, &host_candidates, expected.data(), count) == sz_success_k);
        sz_levenshtein_engine_free(&host_engine);
        for (std::size_t index = 0; index != count; ++index)
            if (distances[index] != expected[index])
                fail_backend_(name, "a narrow rung's distance differs from serial");
    }
}

/** The tiled wavefront against serial's answer at the same lengths, on the one pair it takes. */
static void check_levenshtein_cuda_tiled_() {
    sz_memory_allocator_t unified, host;
    sz_memory_allocator_init_unified(&unified, SZ_NULL);
    sz_memory_allocator_init_default(&host);

    for (std::size_t const query_symbols : levenshtein_cuda_query_symbols_k) {
        levenshtein_cuda_corpus_t corpus(1, query_symbols, 1, levenshtein_cuda_alphabet_t::bytes_k);
        sz_string_view_t const query = corpus.query_views[0], candidate = corpus.views[0];
        sz_size_t distance = 0;
        if (sz_levenshtein_distance_tiled_cuda(query.start, query.length, candidate.start, candidate.length, &unified,
                                               &distance, SZ_NULL) != sz_success_k)
            fail_backend_("cuda", "the wavefront refused a device-resident pair");

        sz_levenshtein_engine_t engine {};
        verify(sz_levenshtein_engine_init_cpu(&corpus.queries, sz_levenshtein_bytes_k, &host, &engine) == sz_success_k);
        sz_size_t expected = 0;
        verify(sz_levenshtein_distances_serial(&engine, &corpus.host_candidates, &expected, 1) == sz_success_k);
        sz_levenshtein_engine_free(&engine);
        if (distance != expected) fail_backend_("cuda", "a wavefront distance differs from serial");
    }
}

/** One backend refusing host memory, sequence handle and outputs alike, rather than staging it. */
static void check_levenshtein_cuda_memory_safety_(levenshtein_cuda_backend_t const &backend) {
    std::string arena(660, '\0');
    randomize_string(&arena[0], arena.size());
    std::array<sz_string_view_t, 4> const views {sz_string_view_t {arena.data(), 1},
                                                 {arena.data() + 1, 60},
                                                 {arena.data() + 61, 199},
                                                 {arena.data() + 260, 400}};
    sz_sequence_t candidates {};
    sz_sequence_from_string_views(views.data(), views.size(), &candidates);

    levenshtein_cuda_corpus_t corpus(1, 200, 1, levenshtein_cuda_alphabet_t::bytes_k);
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init_gpu(&corpus.queries, sz_levenshtein_bytes_k, SZ_NULL, SZ_NULL, &engine) ==
           sz_success_k);
    std::vector<sz_size_t> produced(views.size());
    if (backend.distances(&engine, &candidates, produced.data(), views.size()) != sz_device_memory_mismatch_k)
        fail_backend_(backend.name, "host memory reached a kernel as an address");
    sz_levenshtein_engine_free(&engine);
}

/** The builder refusing a query past what the widest rung holds, in bytes and again in runes. */
static void check_levenshtein_cuda_query_safety_() {
    enum { symbols_k = sz_levenshtein_cuda_words_max_k * 64 + 1 };
    for (levenshtein_cuda_alphabet_t const alphabet :
         {levenshtein_cuda_alphabet_t::bytes_k, levenshtein_cuda_alphabet_t::runes_k}) {
        levenshtein_cuda_corpus_t corpus(1, symbols_k, 1, alphabet);
        sz_levenshtein_engine_t engine {};
        if (sz_levenshtein_engine_init_gpu(&corpus.queries, levenshtein_cuda_symbol_(alphabet), SZ_NULL, SZ_NULL,
                                           &engine) != sz_unexpected_dimensions_k)
            fail_backend_("cuda", "a query past the verticals was not refused");
        if (engine.memory != SZ_NULL) fail_backend_("cuda", "a refused build still kept a block");
    }
}

/** An empty query has no last word to read a score off, so a batch holding one is refused. */
static void check_levenshtein_cuda_empty_query_safety_() {
    unified_vector<sz_string_view_t> query_views(1);
    query_views[0].start = SZ_NULL, query_views[0].length = 0;
    sz_sequence_t queries {};
    sz_sequence_from_string_views(query_views.data(), query_views.size(), &queries);
    sz_levenshtein_engine_t engine {};
    if (sz_levenshtein_engine_init_gpu(&queries, sz_levenshtein_bytes_k, SZ_NULL, SZ_NULL, &engine) !=
        sz_unexpected_dimensions_k)
        fail_backend_("cuda", "an empty query was not refused");
}

/** Device work queued ahead of the verb, so a stream draining after it returns is no accident. */
enum { levenshtein_cuda_filler_bytes_k = 128u * 1024u * 1024u, levenshtein_cuda_filler_passes_k = 16 };

/** Candidates and query symbols the asynchrony check scores, enough to outlast one launch alone. */
enum { levenshtein_cuda_scheduled_candidates_k = 2048, levenshtein_cuda_scheduled_symbols_k = 1024 };

/**
 *  @brief The scoring verb returning before its round completes, on a stream the caller owns.
 *
 *  Work is queued on that stream first, so it is busy when the verb is called: a verb that joined
 *  would drain the filler too, and the query right after it would report the stream idle. Nothing
 *  is asserted about the init, which is allowed to join. The distances are checked once the stream
 *  does drain, so the round that ran asynchronously is the round the equivalence sweep would run.
 */
static void check_levenshtein_cuda_scheduled_asynchrony_() {
    levenshtein_cuda_corpus_t corpus(levenshtein_cuda_scheduled_candidates_k, levenshtein_cuda_scheduled_symbols_k,
                                     levenshtein_cuda_sweep_queries_k, levenshtein_cuda_alphabet_t::bytes_k);

    cudaStream_t stream = SZ_NULL;
    verify(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    sz_levenshtein_engine_t engine {};
    verify(sz_levenshtein_engine_init_gpu(&corpus.queries, sz_levenshtein_bytes_k, SZ_NULL, stream, &engine) ==
           sz_success_k);

    void *filler = SZ_NULL;
    verify(cudaMallocAsync(&filler, levenshtein_cuda_filler_bytes_k, stream) == cudaSuccess);
    for (int pass = 0; pass != levenshtein_cuda_filler_passes_k; ++pass)
        verify(cudaMemsetAsync(filler, pass, levenshtein_cuda_filler_bytes_k, stream) == cudaSuccess);

    verify(sz_levenshtein_distances(&engine, &corpus.device_candidates, corpus.distances.data(), corpus.count()) ==
           sz_success_k);
    if (cudaStreamQuery(stream) != cudaErrorNotReady)
        fail_backend_("cuda", "the scoring verb drained the stream it was handed");

    verify(cudaFreeAsync(filler, stream) == cudaSuccess);
    verify(cudaStreamSynchronize(stream) == cudaSuccess);
    verify(cudaStreamDestroy(stream) == cudaSuccess);
    sz_levenshtein_engine_free(&engine);

    std::vector<sz_size_t> const expected = levenshtein_serial_reference_(corpus, sz_levenshtein_bytes_k);
    for (std::size_t index = 0; index != expected.size(); ++index)
        if (corpus.distances[index] != expected[index])
            fail_backend_("cuda", "a scheduled round's distance differs from serial");
}

#pragma endregion Checks

#pragma region Drivers

/** Every CUDA backend this device carries, against serial, on generated corpora at rung edges. */
void test_levenshtein_all() {
    fmt::println("  - testing {} CUDA query lengths against serial over bytes and runes, refused past {} words",
                 sizeof(levenshtein_cuda_query_symbols_k) / sizeof(std::size_t), (int)sz_levenshtein_cuda_words_max_k);
    for (levenshtein_cuda_backend_t const &backend : levenshtein_cuda_backends) {
        if ((sz_capabilities() & backend.required) != backend.required) continue;
        check_levenshtein_cuda_equivalence_(backend.name, levenshtein_cuda_alphabet_t::bytes_k, backend.distances);
        check_levenshtein_cuda_equivalence_(backend.name, levenshtein_cuda_alphabet_t::runes_k, backend.distances);
        check_levenshtein_cuda_narrow_lanes_(backend.name, backend.distances);
    }
    if ((sz_capabilities() & sz_cap_cuda_k) == sz_cap_cuda_k) {
        check_levenshtein_cuda_tiled_();
        check_levenshtein_cuda_scheduled_asynchrony_();
    }
}

/** Degenerate inputs, stated refusals, and the bound each alphabet's widest rung imposes. */
void test_levenshtein_safety() {
    fmt::println("  - testing degenerate inputs and refused batches of the CUDA edit-distance kernels...");
    for (levenshtein_cuda_backend_t const &backend : levenshtein_cuda_backends) {
        if ((sz_capabilities() & backend.required) != backend.required) continue;
        check_levenshtein_cuda_memory_safety_(backend);
    }
    check_levenshtein_cuda_query_safety_();
    check_levenshtein_cuda_empty_query_safety_();
}

#pragma endregion Drivers
