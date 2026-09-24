/**
 *  @brief  Multi-pattern search on the GPU: the device backend against serial's answers, and the memory
 *          and budget contracts the device verbs keep.
 *  @file   test/substrings.cu
 *  @author Ash Vardanian
 *  @date   September 20, 2026
 *
 *  @c test/substrings.cpp defines @c test_substrings_all and @c test_substrings_safety over the CPU backend,
 *  and this file defines them over the CUDA one. No target links both - @c stringzilla_test_cpp20 takes the
 *  first and @c stringzilla_test_cu20 the second - a CMake invariant rather than a language one.
 *
 *  The serial tier is the oracle rather than a second brute force: the CPU file already measures it against
 *  one, so what is open here is whether a chunked, warmed-up device walk reports the same matches as a
 *  single-chain host walk - which is the whole of what the chunking can get wrong.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <cmath>   // `std::fabs`
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::printf`
#include <cstring> // `std::strlen`

#include <algorithm> // `std::sort`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <stringzilla/stringzilla.h> // Primary C API
#include <stringzilla/substrings.h>  // `sz_substrings_*`

#include "stringzilla.hpp" // `random_string`, `scale_iterations`, `unified_vector`, `verify`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

#pragma region Helpers

/**
 *  @brief A corpus both sides address: one arena of haystacks and the views into it.
 *
 *  Unified storage is readable from the host, so the serial reference runs against these very bytes and the
 *  device verbs accept them without anything being staged.
 */
struct substrings_cuda_corpus_t {
    unified_vector<char> arena;             /**< Every haystack's bytes, back to back. */
    unified_vector<sz_string_view_t> views; /**< One view per haystack; its size is the haystack count. */
    sz_sequence_t device_haystacks {};      /**< Accessors a kernel calls, as the device verbs require. */
    sz_sequence_t host_haystacks {};        /**< Accessors the serial reference calls, over the same views. */

    substrings_cuda_corpus_t(std::vector<std::string> const &haystacks) : views(haystacks.size()) {
        for (std::size_t index = 0; index != haystacks.size(); ++index) {
            views[index].length = haystacks[index].size();
            arena.insert(arena.end(), haystacks[index].begin(), haystacks[index].end());
        }
        // The arena's address is only final once it has stopped growing, so the starts are filled afterwards.
        std::size_t written = 0;
        for (sz_string_view_t &view : views) view.start = arena.data() + written, written += view.length;
        verify(sz_sequence_from_string_views_cuda(views.data(), views.size(), &device_haystacks) == sz_success_k);
        sz_sequence_from_string_views(views.data(), views.size(), &host_haystacks);
    }
};

/** @brief A vocabulary both sides address, in memory a kernel can read. */
struct substrings_cuda_vocabulary_t {
    unified_vector<char> arena;             /**< Every needle's bytes, back to back. */
    unified_vector<sz_string_view_t> views; /**< One view per needle. */
    sz_memory_allocator_t unified {};       /**< Hands back memory both the host and the device address. */
    sz_sequence_t needles {};               /**< Host accessors, which is all the builder needs. */

    substrings_cuda_vocabulary_t(std::vector<std::string> const &strings) : views(strings.size()) {
        for (std::size_t index = 0; index != strings.size(); ++index) {
            views[index].length = strings[index].size();
            arena.insert(arena.end(), strings[index].begin(), strings[index].end());
        }
        std::size_t written = 0;
        for (sz_string_view_t &view : views) view.start = arena.data() + written, written += view.length;
        sz_sequence_from_string_views(views.data(), views.size(), &needles);
        sz_memory_allocator_init_unified(&unified, SZ_NULL);
    }
    substrings_cuda_vocabulary_t(substrings_cuda_vocabulary_t const &) = delete;
    substrings_cuda_vocabulary_t &operator=(substrings_cuda_vocabulary_t const &) = delete;
};

/**
 *  @brief One vocabulary compiled twice under one policy: once for the host, once for @c stream 's device.
 *
 *  The policy sizes the arena, so it belongs to the engine rather than to a call, and comparing two tiers
 *  under one policy means holding two engines rather than one object two verbs read differently.
 */
struct substrings_cuda_engines_t {
    sz_substrings_engine_t host {};   /**< The serial oracle's engine, in plain host memory. */
    sz_substrings_engine_t device {}; /**< The device's engine, arena and report included. */

    substrings_cuda_engines_t(substrings_cuda_vocabulary_t &vocabulary,
                              sz_substrings_case_sensitivity_t sensitivity,
                              sz_substrings_overlap_policy_t policy) {
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);
        verify(sz_substrings_engine_init_cpu(&vocabulary.needles, sensitivity, policy, SZ_SUBSTRINGS_HOT_STATES_AUTO,
                                             0, &allocator, &host) == sz_success_k);
        verify(sz_substrings_engine_init_gpu(&vocabulary.needles, sensitivity, policy, SZ_SUBSTRINGS_HOT_STATES_AUTO,
                                             0, &vocabulary.unified, nullptr, &device) == sz_success_k);
        verify(sz_memory_reaches_device(device.memory) && "A device engine's block is one a kernel addresses");
        verify(!sz_memory_reaches_device(host.memory) && "A host engine's block is not");
    }
    substrings_cuda_engines_t(substrings_cuda_engines_t const &) = delete;
    substrings_cuda_engines_t &operator=(substrings_cuda_engines_t const &) = delete;
    ~substrings_cuda_engines_t() noexcept {
        sz_substrings_engine_free(&device);
        sz_substrings_engine_free(&host);
    }
};

/** @brief Joins the default stream, which is what every device verb leaves the caller to do. */
static void join_() { verify(cudaStreamSynchronize(nullptr) == cudaSuccess); }

/** @brief One match, ordered so two backends' reports compare as sequences rather than as multisets. */
struct substrings_cuda_case_t {
    sz_size_t haystack_index {}; /**< Which haystack of the sequence this match was found in. */
    sz_size_t needle_index {};   /**< Which needle of the vocabulary matched. */
    sz_size_t byte_offset {};    /**< Where the match starts inside that haystack. */
    sz_size_t byte_length {};    /**< Haystack bytes the match spans. */

    bool operator==(substrings_cuda_case_t const &other) const noexcept {
        return haystack_index == other.haystack_index && needle_index == other.needle_index &&
               byte_offset == other.byte_offset && byte_length == other.byte_length;
    }
    bool operator<(substrings_cuda_case_t const &other) const noexcept {
        if (haystack_index != other.haystack_index) return haystack_index < other.haystack_index;
        if (byte_offset != other.byte_offset) return byte_offset < other.byte_offset;
        if (byte_length != other.byte_length) return byte_length < other.byte_length;
        return needle_index < other.needle_index;
    }
};

/** @brief Sorts one verb's match array into the shape two backends compare position by position. */
static std::vector<substrings_cuda_case_t> sorted_(sz_substrings_match_t const *matches, sz_size_t count) {
    std::vector<substrings_cuda_case_t> reported(count);
    for (std::size_t index = 0; index != count; ++index)
        reported[index] = {matches[index].haystack_index, matches[index].needle_index, matches[index].byte_offset,
                           matches[index].byte_length};
    std::sort(reported.begin(), reported.end());
    return reported;
}

/** @brief Every match the serial tier reports, sized from the report its own sizing call leaves. */
static std::vector<substrings_cuda_case_t> serial_matches_(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks) {
    std::vector<sz_size_t> offsets(haystacks->count + 1, 0);
    verify(sz_substrings_find_serial(engine, haystacks, nullptr, 0, offsets.data()) == sz_success_k);
    std::vector<sz_substrings_match_t> matches(engine->report->matches_emitted);
    verify(sz_substrings_find_serial(engine, haystacks, matches.data(), matches.size(), offsets.data()) ==
           sz_success_k);
    verify(engine->report->shortfall == 0);
    return sorted_(matches.data(), matches.size());
}

/** @brief Every match the device tier reports, joined once per call because no verb joins for the caller. */
static std::vector<substrings_cuda_case_t> device_matches_(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks) {
    unified_vector<sz_size_t> offsets(haystacks->count + 1, 0);
    verify(sz_substrings_find_cuda(engine, haystacks, nullptr, 0, offsets.data()) == sz_success_k);
    join_();
    // A cover thins the matches after the sizing walk, so the boundaries name the survivors and the report's
    // emitted count names what the walk found before them.
    sz_size_t const total = offsets[haystacks->count];
    verify(engine->report->matches_stored + engine->report->shortfall == total);

    unified_vector<sz_substrings_match_t> matches(total);
    verify(sz_substrings_find_cuda(engine, haystacks, matches.data(), matches.size(), offsets.data()) ==
           sz_success_k);
    join_();
    verify(engine->report->matches_stored == total && engine->report->shortfall == 0);
    return sorted_(matches.data(), total);
}

/** @brief That every reported match is real, and that no two of them share a byte of one haystack. */
static void verify_is_a_cover_(std::vector<substrings_cuda_case_t> const &reported,
                               std::vector<substrings_cuda_case_t> const &every_match) {
    // Both lists arrive sorted, so the subset test is one merge rather than a scan per reported match.
    std::size_t candidate = 0;
    for (substrings_cuda_case_t const &match : reported) {
        while (candidate != every_match.size() && every_match[candidate] < match) ++candidate;
        verify(candidate != every_match.size() && every_match[candidate] == match &&
               "A cover reports only matches the overlapping walk found");
    }
    for (std::size_t index = 1; index < reported.size(); ++index) {
        substrings_cuda_case_t const &previous = reported[index - 1], &current = reported[index];
        if (previous.haystack_index != current.haystack_index) continue;
        verify(current.byte_offset >= previous.byte_offset + previous.byte_length &&
               "A cover's matches share no bytes");
    }
}

/**
 *  @brief The device's matches, counts and rewrite against the serial tier's, over one corpus and policy.
 *  @param[in] fidelity Whether the device's cover must equal the serial one, or merely be a valid cover.
 *
 *  A dense vocabulary leaves no gap between matches, so one run of mutually-reaching matches spans a whole
 *  haystack and the device falls back to emitted order rather than running a quadratic greedy over it.
 */
static void check_against_serial_(substrings_cuda_corpus_t &corpus, substrings_cuda_vocabulary_t &vocabulary,
                                  std::vector<std::string> const &haystacks,
                                  sz_substrings_case_sensitivity_t sensitivity,
                                  sz_substrings_overlap_policy_t policy,
                                  sz_substrings_cover_fidelity_t fidelity = sz_substrings_cover_exact_k) {
    substrings_cuda_engines_t engines(vocabulary, sensitivity, policy);
    std::vector<substrings_cuda_case_t> const expected = serial_matches_(&engines.host, &corpus.host_haystacks);
    std::vector<substrings_cuda_case_t> const reported = device_matches_(&engines.device, &corpus.device_haystacks);
    sz_size_t const device_total = reported.size();
    if (fidelity == sz_substrings_cover_exact_k) {
        verify(reported.size() == expected.size());
        for (std::size_t index = 0; index != expected.size(); ++index) verify(reported[index] == expected[index]);
    }
    else {
        // Every match the overlapping walk found, which is what a cover may draw from.
        substrings_cuda_engines_t overlapping(vocabulary, sensitivity, sz_substrings_overlapping_k);
        verify_is_a_cover_(reported, serial_matches_(&overlapping.host, &corpus.host_haystacks));
    }

    // The counts come from the boundaries rather than from the match list, so they can disagree with it.
    unified_vector<sz_size_t> device_counts(haystacks.size(), 0);
    std::vector<sz_size_t> serial_counts(haystacks.size(), 0);
    verify(sz_substrings_counts_serial(&engines.host, &corpus.host_haystacks, serial_counts.data(), 1) ==
           sz_success_k);
    verify(sz_substrings_counts_cuda(&engines.device, &corpus.device_haystacks, device_counts.data(), 1) ==
           sz_success_k);
    join_();
    for (std::size_t index = 0; index != haystacks.size(); ++index)
        if (fidelity == sz_substrings_cover_exact_k) verify(device_counts[index] == serial_counts[index]);
    {
        sz_size_t counted = 0;
        for (sz_size_t const count : device_counts) counted += count;
        verify(counted == device_total && "The counts and the matches come from one pass");
    }

    if (policy == sz_substrings_overlapping_k) return;
    if (fidelity != sz_substrings_cover_exact_k) return;

    // The rewrite, whose device path owns a block scan and a tiled copy the serial one has no analogue for.
    std::vector<std::string> replacements;
    for (std::size_t index = 0; index != vocabulary.views.size(); ++index)
        replacements.push_back(index % 3 == 0 ? std::string() : "<" + std::to_string(index) + ">");
    substrings_cuda_corpus_t replacement_corpus(replacements);

    unified_vector<sz_size_t> device_offsets(haystacks.size() + 1, 0);
    std::vector<sz_size_t> serial_offsets(haystacks.size() + 1, 0);
    verify(sz_substrings_replace_cuda(&engines.device, &corpus.device_haystacks,
                                      &replacement_corpus.device_haystacks, nullptr, 0,
                                      device_offsets.data()) == sz_success_k);
    join_();
    verify(sz_substrings_replace_serial(&engines.host, &corpus.host_haystacks, &replacement_corpus.host_haystacks,
                                        nullptr, 0, serial_offsets.data()) == sz_success_k);
    for (std::size_t index = 0; index != haystacks.size() + 1; ++index)
        verify(device_offsets[index] == serial_offsets[index]);

    sz_size_t const rewritten = serial_offsets[haystacks.size()];
    unified_vector<char> device_tape(rewritten);
    std::vector<char> serial_tape(rewritten);
    verify(sz_substrings_replace_cuda(&engines.device, &corpus.device_haystacks,
                                      &replacement_corpus.device_haystacks,
                                      device_tape.empty() ? nullptr : device_tape.data(), rewritten,
                                      device_offsets.data()) == sz_success_k);
    join_();
    verify(engines.device.report->shortfall == 0);
    verify(sz_substrings_replace_serial(&engines.host, &corpus.host_haystacks, &replacement_corpus.host_haystacks,
                                        serial_tape.empty() ? nullptr : serial_tape.data(), rewritten,
                                        serial_offsets.data()) == sz_success_k);
    for (std::size_t index = 0; index != rewritten; ++index) verify(device_tape[index] == serial_tape[index]);
}

/**
 *  @brief The device's BM25 against the serial tier's, by byte lengths and by caller-given ones.
 *
 *  The device sums in fixed point and the host in ascending needle order, so the two agree to rounding
 *  rather than bit for bit.
 */
static void check_bm25_against_serial_(substrings_cuda_corpus_t &corpus, substrings_cuda_vocabulary_t &vocabulary,
                                       std::vector<std::string> const &haystacks,
                                       sz_substrings_case_sensitivity_t sensitivity) {
    substrings_cuda_engines_t engines(vocabulary, sensitivity, sz_substrings_overlapping_k);
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
                                                weights.data(), serial_scores.data(), 1) == sz_success_k);
        verify(sz_substrings_bm25_scores_cuda(&engines.device, &corpus.device_haystacks, lengths, &parameters,
                                              weights.data(), device_scores.data(), 1) == sz_success_k);
        join_();
        for (std::size_t index = 0; index != haystacks.size(); ++index) {
            double const tolerance = 1e-5 * std::max(1.0, std::fabs((double)serial_scores[index]));
            verify(std::fabs(device_scores[index] - serial_scores[index]) <= tolerance);
        }
    }
}

/** @brief One vocabulary against one corpus under every policy. */
static void check_policies_(std::vector<std::string> const &haystacks, std::vector<std::string> const &needles,
                            sz_substrings_case_sensitivity_t sensitivity,
                            sz_substrings_cover_fidelity_t fidelity = sz_substrings_cover_exact_k) {
    substrings_cuda_corpus_t corpus(haystacks);
    substrings_cuda_vocabulary_t vocabulary(needles);
    // An overlapping walk reports every match whatever the density, so it is always compared exactly.
    check_against_serial_(corpus, vocabulary, haystacks, sensitivity, sz_substrings_overlapping_k);
    check_against_serial_(corpus, vocabulary, haystacks, sensitivity, sz_substrings_leftmost_longest_k, fidelity);
    check_against_serial_(corpus, vocabulary, haystacks, sensitivity, sz_substrings_leftmost_first_k, fidelity);
    check_bm25_against_serial_(corpus, vocabulary, haystacks, sensitivity);
}

#pragma endregion Helpers

#pragma region Unit Cases

/** The chunk-boundary cases a single-chain host walk cannot express. */
void test_substrings_unit() {
    // The device walk is chunked and the host walk is not, so a match spanning a chunk boundary is the one
    // thing this can get wrong that the CPU suite cannot see.
    check_policies_({"ushers"}, {"he", "she", "his", "hers"}, sz_substrings_cased_k);
    check_policies_({"Straße", "STRASSE"}, {"strasse", "sse", "s"}, sz_substrings_uncased_k);
}

/** Random corpora wide enough that the planner cuts several chunks per haystack. */
void test_substrings_all() {
    char const *const alphabets[] = {"ab", "abcdefgh", "abcdefghijklmnopqrstuvwxyz"};
    std::size_t const rounds = scale_iterations(8);

    for (std::size_t round = 0; round != rounds; ++round) {
        char const *const alphabet = alphabets[round % 3];
        std::size_t const cardinality = std::strlen(alphabet);
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 1 + (round * 7) % 24; ++index)
            needles.push_back(random_string(1 + (index * 3 + round) % 7, alphabet, cardinality));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        // Long enough that the planner cuts several chunks out of one haystack, which is the regime the
        // warm-up exists for; a corpus of short haystacks never crosses a chunk boundary at all.
        for (std::size_t index = 0; index != 1 + (round * 5) % 9; ++index)
            haystacks.push_back(random_string(4096 + index * 977, alphabet, cardinality));
        check_policies_(haystacks, needles, sz_substrings_cased_k, sz_substrings_cover_approximate_k);
    }

    // A vocabulary wider than a block's tally, so needles hash into shared slots and spill past them.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 6000; ++index)
            needles.push_back(random_string(2 + index % 4, "abcdefghijklmnopqrstuvwxyz", 26));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 64; ++index)
            haystacks.push_back(random_string(256 + index * 97, "abcdefghijklmnopqrstuvwxyz", 26));
        // Short documents past a residency wave, so the grid holds many blocks, each with its own overflow row.
        for (std::size_t index = 0; index != 4096; ++index)
            haystacks.push_back(random_string(16 + index % 97, "abcdefghijklmnopqrstuvwxyz", 26));
        substrings_cuda_corpus_t corpus(haystacks);
        substrings_cuda_vocabulary_t vocabulary(needles);
        verify(vocabulary.views.size() > 4096);
        check_bm25_against_serial_(corpus, vocabulary, haystacks, sz_substrings_cased_k);
    }

    // Nucleotides, where the whole hot tier is five columns wide and fits a block's shared memory.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 300; ++index) needles.push_back(random_string(4 + index % 13, "ACGT", 4));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 6; ++index) {
            std::string haystack = random_string(20000 + index * 4099, "ACGT", 4);
            for (std::size_t offset = 61; offset < haystack.size(); offset += 997) haystack[offset] = 'N';
            haystacks.push_back(haystack);
        }
        check_policies_(haystacks, needles, sz_substrings_cased_k, sz_substrings_cover_approximate_k);
    }

    // One haystack far wider than a chunk, so the boundary case is hit many times over in a single walk.
    {
        std::vector<std::string> const needles {"the", "there", "here", "her", "he"};
        std::vector<std::string> haystacks {random_string(1 << 20, "the ", 4)};
        check_policies_(haystacks, needles, sz_substrings_cased_k, sz_substrings_cover_approximate_k);
    }
}

/** What the device verbs refuse, and what the report says when an output could not hold the answer. */
void test_substrings_safety() {
    std::vector<std::string> const needles {"ab", "cd"};
    std::vector<std::string> const haystacks {"abcdabcd"};
    substrings_cuda_corpus_t corpus(haystacks);
    substrings_cuda_vocabulary_t vocabulary(needles);
    substrings_cuda_engines_t engines(vocabulary, sz_substrings_cased_k, sz_substrings_overlapping_k);

    // A device verb refuses host memory rather than reading it from a kernel, whatever else is resident.
    {
        std::vector<sz_size_t> host_counts(haystacks.size(), 0);
        verify(sz_substrings_counts_cuda(&engines.device, &corpus.device_haystacks, host_counts.data(), 1) ==
               sz_device_memory_mismatch_k);
    }

    // An output stride of zero cannot address one entry per haystack, whatever the haystack count.
    {
        unified_vector<sz_size_t> counts(haystacks.size(), SZ_SIZE_MAX);
        verify(sz_substrings_counts_cuda(&engines.device, &corpus.device_haystacks, counts.data(), 0) ==
               sz_unexpected_dimensions_k);
        verify(counts[0] == SZ_SIZE_MAX);
    }

    // The counts a device walk answers with, which the same corpus answers on the host.
    {
        unified_vector<sz_size_t> counts(haystacks.size(), 0);
        verify(sz_substrings_counts_cuda(&engines.device, &corpus.device_haystacks, counts.data(), 1) ==
               sz_success_k);
        join_();
        verify(counts[0] == 4);
    }

    // A capacity that cannot hold the matches is not an error: the report names the true total and the
    // shortfall beside it, and the output is left untouched rather than truncated.
    {
        unified_vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        verify(sz_substrings_find_cuda(&engines.device, &corpus.device_haystacks, nullptr, 0, offsets.data()) ==
               sz_success_k);
        join_();
        verify(engines.device.report->matches_emitted == 4);
        verify(engines.device.report->matches_stored == 0 && engines.device.report->shortfall == 4);
    }

    // A substitution over matches that share bytes is not a function, so the overlapping policy is refused
    // before anything is launched.
    {
        std::vector<std::string> const replacements {"x", "y"};
        substrings_cuda_corpus_t replacement_corpus(replacements);
        unified_vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        verify(sz_substrings_replace_cuda(&engines.device, &corpus.device_haystacks,
                                          &replacement_corpus.device_haystacks, nullptr, 0,
                                          offsets.data()) == sz_status_unknown_k);
    }

    // A vocabulary a round cannot emit within its budget leaves every later kernel retired, so the outputs
    // keep whatever they held and the report alone says the round did not fit.
    {
        sz_substrings_engine_t tiny {};
        unified_vector<sz_size_t> offsets(haystacks.size() + 1, SZ_SIZE_MAX);
        verify(sz_substrings_engine_init_gpu(&vocabulary.needles, sz_substrings_cased_k,
                                             sz_substrings_leftmost_first_k, SZ_SUBSTRINGS_HOT_STATES_AUTO, 1,
                                             &vocabulary.unified, nullptr, &tiny) == sz_success_k);
        verify(sz_substrings_find_cuda(&tiny, &corpus.device_haystacks, nullptr, 0, offsets.data()) == sz_success_k);
        join_();
        verify(tiny.report->matches_emitted == 4 && tiny.report->shortfall == 3);
        verify(offsets[haystacks.size()] == 0 && "A retired offsets kernel leaves the boundaries zeroed");
        sz_substrings_engine_free(&tiny);
    }
}

#pragma endregion Unit Cases
