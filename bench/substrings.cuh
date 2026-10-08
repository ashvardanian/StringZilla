/**
 *  @file bench/substrings.cuh
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief Shared code for CPU and GPU multi-pattern search benchmarks: the vocabulary, the corpus,
 *      and the arms.
 */
#pragma once

#include <algorithm>   // `std::sort`, `std::min`, `std::max`
#include <limits>      // `std::numeric_limits`
#include <numeric>     // `std::iota`
#include <optional>    // `std::optional`
#include <random>      // `std::mt19937_64`
#include <set>         // `std::set`
#include <stdexcept>   // `std::runtime_error`
#include <string>      // `std::string`, `std::to_string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <stringzilla/substrings.h> // `sz_substrings_*`

#include "harness.hpp"

namespace ashvardanian::stringzilla::bench {

#pragma region Vocabulary

/** Which slice of the corpus vocabulary a dictionary was compiled from. */
enum class substrings_slice_t {

    /** The most common one percent, where nearly every byte enters an accepting state. */
    frequent_k,

    /** The least common one percent, where the walk sits near the root and rarely reports. */
    rare_k,

    /** Substrings of the corpus itself, which exist for any alphabet, with words or without. */
    sampled_k,
};

/** The suffix every row of one dictionary carries: its slice, then its case sensitivity. */
static std::string substrings_label(substrings_slice_t slice, sz_substrings_case_sensitivity_t sensitivity) {
    char const *const name = slice == substrings_slice_t::frequent_k ? ":frequent"
                             : slice == substrings_slice_t::rare_k   ? ":rare"
                                                                     : ":sampled";
    return std::string(name) + (sensitivity == sz_substrings_uncased_k ? ":uncased" : ":cased");
}

/** The suffix a row carries for the overlap policy it ran under. */
static char const *substrings_policy_name(sz_substrings_overlap_policy_t policy) {
    switch (policy) {
    case sz_substrings_overlapping_k: return ":overlapping";
    case sz_substrings_leftmost_longest_k: return ":longest";
    case sz_substrings_leftmost_first_k: return ":first";
    }
    return ":unknown";
}

/** Every overlap policy, all of which counting and reporting accept. */
static sz_substrings_overlap_policy_t const substrings_policies_k[] = {
    sz_substrings_overlapping_k, sz_substrings_leftmost_longest_k, sz_substrings_leftmost_first_k};

/** The leftmost policies, the only ones a substitution accepts. */
static sz_substrings_overlap_policy_t const substrings_leftmost_policies_k[] = {sz_substrings_leftmost_longest_k,
                                                                                sz_substrings_leftmost_first_k};

/** Whether the filter keeps any row over the slice @p label, every row named after its verb, then
 *  @p kit, like @c _haswell, then the slice and the policy, so an unwanted slice is never drawn. */
inline bool substrings_selects(environment_t const &env, std::string_view kit, std::string const &label) {
    auto const selects = [&](char const *verb, char const *policy) {
        return env.settings.selects(fmt::format("sz_substrings_{}{}{}{}", verb, kit, label, policy));
    };
    for (sz_substrings_overlap_policy_t const policy : substrings_policies_k)
        if (selects("counts", substrings_policy_name(policy)) || selects("find", substrings_policy_name(policy)))
            return true;
    for (sz_substrings_overlap_policy_t const policy : substrings_leftmost_policies_k)
        if (selects("replace", substrings_policy_name(policy))) return true;
    return selects("bm25_scores", "") || selects("engine_init", "");
}

/** Shortest word admitted: anything under three bytes matches at nearly every position and measures
 *  the reporting path rather than the automaton. */
static constexpr std::size_t substrings_min_word_bytes_k = 3;

/** Longest word admitted: longer whitespace-cut tokens are unsegmented CJK runs or URLs. */
static constexpr std::size_t substrings_max_word_bytes_k = 32;

/** Fraction of the frequency-ordered vocabulary discarded from the top as stopwords. */
static constexpr double substrings_frequent_cutoff_k = 0.01;

/** Occurrences a term needs to be admitted; a hapax is mostly OCR noise no haystack meets twice. */
static constexpr std::size_t substrings_minimum_occurrences_k = 2;

/** Needles a sampled slice draws, and the span of their lengths in bytes. */
static constexpr std::size_t substrings_sampled_needles_k = 1000;
static constexpr std::size_t substrings_sampled_min_bytes_k = 4, substrings_sampled_max_bytes_k = 16;

/** Distinct substrings of haystacks at seeded random positions, so any corpus yields needles. */
static std::vector<std::string> substrings_sampled(environment_t const &env, corpus_t const &corpus) {
    std::mt19937_64 generator(env.settings.seed.value);
    std::set<std::string> distinct;
    std::size_t attempts = 0;
    while (distinct.size() != substrings_sampled_needles_k && attempts++ != 64 * substrings_sampled_needles_k) {
        token_view_t const token = corpus.tokens[generator() % corpus.tokens.size()];
        std::size_t const length = substrings_sampled_min_bytes_k +
                                   generator() % (substrings_sampled_max_bytes_k - substrings_sampled_min_bytes_k + 1);
        if (token.size() < length) continue;
        distinct.emplace(token.data() + generator() % (token.size() - length + 1), length);
    }
    return {distinct.begin(), distinct.end()};
}

/**
 *  @brief The dataset's distinct words seen at least @c substrings_minimum_occurrences_k times,
 *      most frequent first and by content within a count, which every word slice is a window of.
 *
 *  Drawn from the dataset's words whatever tokenization shapes the haystacks, so a line search and
 *  a word search draw needles from the same vocabulary. Sorting every word is most of a draw, so a
 *  caller ranks once and cuts every word slice it needs from the same ranking.
 */
struct substrings_ranking_t {

    /** The kept words in rank order, viewing the corpus. */
    std::vector<std::string_view> words;

    /** Distinct words of any count, hapax included, which the frequent cutoff is a fraction of. */
    std::size_t distinct_total = 0;

    explicit substrings_ranking_t(corpus_t const &corpus) {
        std::vector<std::string_view> occurrences;
        {
            tokens_t const tokenized = tokenize(corpus.dataset);
            occurrences.reserve(tokenized.size());
            for (token_view_t const word : tokenized)
                if (word.size() >= substrings_min_word_bytes_k && word.size() <= substrings_max_word_bytes_k)
                    occurrences.push_back(word);
        }
        std::sort(occurrences.begin(), occurrences.end());
        std::vector<std::size_t> frequencies;
        for (std::size_t position = 0; position != occurrences.size();) {
            std::size_t run = position + 1;
            while (run != occurrences.size() && occurrences[run] == occurrences[position]) ++run;
            ++distinct_total;
            if (run - position >= substrings_minimum_occurrences_k)
                words.push_back(occurrences[position]), frequencies.push_back(run - position);
            position = run;
        }

        // Frequency first, then content, so the ranking is deterministic across runs and platforms.
        std::vector<std::size_t> ranks(words.size());
        std::iota(ranks.begin(), ranks.end(), (std::size_t)0);
        std::sort(ranks.begin(), ranks.end(), [&](std::size_t left, std::size_t right) {
            if (frequencies[left] != frequencies[right]) return frequencies[left] > frequencies[right];
            return words[left] < words[right];
        });
        std::vector<std::string_view> ranked(words.size());
        for (std::size_t rank = 0; rank != ranks.size(); ++rank) ranked[rank] = words[ranks[rank]];
        words = std::move(ranked);
    }
};

/**
 *  @brief One percent of the post-cutoff @p ranking, taken from the end @p slice names.
 *
 *  Both noisy ends are removed before any slice is taken. The most frequent one percent are
 *  stopwords that every haystack holds, which would measure the reporting path rather than the
 *  automaton; terms occurring once are noise no haystack reaches twice. The top cutoff is a
 *  fraction of @b all distinct terms, hapax included, so the slice stays where it is whenever the
 *  hapax filter moves.
 */
static std::vector<std::string> substrings_vocabulary(substrings_ranking_t const &ranking, substrings_slice_t slice) {
    // ? A corpus without repeated words, such as nucleotides, has no word slice.
    if (ranking.words.empty()) return {};
    std::size_t const dropped = std::min<std::size_t>(
        (std::size_t)((double)ranking.distinct_total * substrings_frequent_cutoff_k), ranking.words.size());
    std::size_t const available = ranking.words.size() - dropped;
    std::size_t const wanted = std::max<std::size_t>(available / 100, 1);
    std::size_t const first = slice == substrings_slice_t::frequent_k ? dropped : dropped + available - wanted;
    return {ranking.words.begin() + first, ranking.words.begin() + first + wanted};
}

/** The needles of @p slice: sampled from @p corpus, or cut from its word @p ranking, which the
 *  first word slice a caller asks for builds and the rest reuse. */
static std::vector<std::string> substrings_needles(environment_t const &env, corpus_t const &corpus,
                                                   std::optional<substrings_ranking_t> &ranking,
                                                   substrings_slice_t slice) {
    if (slice == substrings_slice_t::sampled_k) return substrings_sampled(env, corpus);
    if (!ranking) ranking.emplace(corpus);
    return substrings_vocabulary(*ranking, slice);
}

#pragma endregion Vocabulary

#pragma region Inputs

/** One vocabulary slice, and one replacement per needle for a rewrite to substitute. */
struct substrings_dictionary_t {

    /** The words this dictionary was compiled from. */
    std::vector<std::string> needles;

    /** Every replacement back to back, where a kernel reads. */
    unified_vector<char> replacement_bytes;

    /** One per needle, alternating with an empty deletion. */
    std::vector<sz_string_view_t> replacement_views;

    sz_substrings_case_sensitivity_t sensitivity;

    substrings_dictionary_t(std::vector<std::string> words, sz_substrings_case_sensitivity_t sensitivity,
                            sz_allocator_t const &memory, sz_stream_t stream = nullptr)
        : needles(std::move(words)), replacement_bytes(unified_alloc<char>(memory, stream)),
          replacement_views(needles.size()), sensitivity(sensitivity) {
        for (std::size_t index = 0; index != needles.size(); ++index) {
            std::string const replacement = index % 2 ? std::string() : "<" + std::to_string(index) + ">";
            replacement_bytes.insert(replacement_bytes.end(), replacement.begin(), replacement.end());
            replacement_views[index].length = replacement.size();
        }
        // The arena's address is final only once it has stopped growing.
        std::size_t written = 0;
        for (sz_string_view_t &view : replacement_views) {
            view.start = replacement_bytes.data() + written;
            written += view.length;
        }
    }
    sz_sequence_t needle_sequence() const noexcept {
        return {needles.data(), needles.size(),
                [](void const *handle, sz_size_t index) -> sz_cptr_t {
                    auto const *strings = static_cast<std::string const *>(handle);
                    return strings[index].data();
                },
                [](void const *handle, sz_size_t index) -> sz_size_t {
                    auto const *strings = static_cast<std::string const *>(handle);
                    return strings[index].size();
                }};
    }
    sz_sequence_t replacements() const noexcept {
        sz_sequence_t result {};
        sz_sequence_from_string_views(replacement_views.data(), replacement_views.size(), &result);
        return result;
    }
    std::size_t needle_bytes() const noexcept {
        std::size_t total = 0;
        for (std::string const &needle : needles) total += needle.size();
        return total;
    }

    substrings_dictionary_t(substrings_dictionary_t const &) = delete;
    substrings_dictionary_t &operator=(substrings_dictionary_t const &) = delete;
};

struct substrings_engine_t {
    sz_substrings_engine_t engine {};
    sz_stream_t stream = nullptr;

    template <typename init_type_>
    substrings_engine_t(substrings_dictionary_t const &dictionary, sz_substrings_overlap_policy_t policy,
                        init_type_ init, sz_allocator_t allocator, sz_size_t matches_budget = 0,
                        sz_size_t haystacks_budget = 0, sz_stream_t stream = nullptr)
        : stream(stream) {
        sz_sequence_t const needles = dictionary.needle_sequence();
        if (init(&engine, &needles, dictionary.sensitivity, policy, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO,
                 matches_budget, haystacks_budget, &allocator, stream) != sz_success_k)
            throw std::runtime_error("The vocabulary would not compile.");
    }
    substrings_engine_t(substrings_engine_t const &) = delete;
    substrings_engine_t &operator=(substrings_engine_t const &) = delete;
    ~substrings_engine_t() noexcept { sz_substrings_engine_free(&engine, stream); }
    void join() const {
        if (engine.capability & sz_cap_gpus_k)
            if (sz_stream_synchronize_best(engine.capability, stream) != sz_success_k)
                throw std::runtime_error("The substring stream would not synchronize.");
    }
};

/** Host views of the corpus, copied into a resident tape by device benchmarks. */
struct substrings_corpus_t {

    /** One view per haystack, in corpus order. */
    std::vector<sz_string_view_t> views;

    /** Views over the leading @p limit tokens of @p corpus, or over all of them. */
    explicit substrings_corpus_t(corpus_t const &corpus, std::size_t limit = std::numeric_limits<std::size_t>::max())
        : views(std::min(corpus.tokens.size(), limit)) {
        for (std::size_t index = 0; index != views.size(); ++index) {
            token_view_t const token = corpus.tokens[index];
            views[index] = {token.data(), token.size()};
        }
    }

    /** Views over @p spans, whose text storage must outlive this corpus. */
    explicit substrings_corpus_t(std::vector<sz_string_view_t> const &spans) : views(spans.size()) {
        for (std::size_t index = 0; index != spans.size(); ++index) views[index] = spans[index];
    }

    sz_sequence_t haystacks() const noexcept {
        sz_sequence_t result {};
        sz_sequence_from_string_views(views.data(), views.size(), &result);
        return result;
    }

    std::size_t bytes() const noexcept {
        std::size_t total = 0;
        for (sz_string_view_t const &view : views) total += view.length;
        return total;
    }

    /** What one round over every haystack reports: its bytes as both throughput and operations. */
    call_result_t round(check_value_t check_value) const {
        std::size_t const total = bytes();
        call_result_t result(total, check_value, total);
        return result;
    }
};

#pragma endregion Inputs

#pragma region Arms

/** Counts every match in the whole corpus, one call per round, into counts it owns. */
template <typename function_type_>
struct substrings_counts_from_sz {

    /** The compiled vocabulary this arm walks, policy included. */
    substrings_engine_t &engine;
    function_type_ counts_kernel;

    /** The haystacks one round walks. */
    substrings_corpus_t const &corpus;

    /** Borrowed view of the strings read by this arm. */
    sz_sequence_t haystacks;

    /** @b [haystacks], what a round fills, on a device with memory of its own. */
    device_vector<sz_size_t> device_counts;

    /** @b [haystacks], what the host checks. */
    pinned_vector<sz_size_t> counts;
    std::optional<std::reference_wrapper<device_backend_t const>> runtime;

    substrings_counts_from_sz(function_type_ kernel, substrings_engine_t &engine, substrings_corpus_t const &corpus,
                              sz_sequence_t const &haystacks,
                              std::optional<std::reference_wrapper<device_backend_t const>> runtime = {})
        : engine(engine), counts_kernel(kernel), corpus(corpus), haystacks(haystacks), runtime(runtime) {
        bool const separate = runtime && runtime->get().separate();
        counts = pinned_vector<sz_size_t>(
            corpus.views.size(), 0,
            pinned_alloc<sz_size_t>(separate ? runtime->get().pinned : engine.engine.allocator, engine.stream));
        if (!separate) return;
        device_counts = device_vector<sz_size_t>(device_alloc<sz_size_t>(runtime->get().device, engine.stream));
        if (device_counts.resize_uninitialized(corpus.views.size()) != sz::status_t::success_k)
            throw std::runtime_error("The device would not hold the counts.");
    }

    call_result_t operator()(std::size_t) {
        if (counts_kernel(&engine.engine, &haystacks, device_counts.size() == 0 ? counts.data() : device_counts.data(), 1,
                          engine.stream) != sz_success_k)
            throw std::runtime_error("The counting round failed.");
        if (device_counts.size() != 0 &&
            copy_device_to_host(device_counts, std::span<sz_size_t>(counts), runtime->get()) != sz_success_k)
            throw std::runtime_error("The counts would not come back.");
        engine.join();
        check_value_t mixed = 0;
        for (sz_size_t const count : counts) mixed = mixed * 31u + (check_value_t)count;
        return corpus.round(mixed);
    }
};

/** Reports every match in the whole corpus, into an array its own backend sized. */
template <typename function_type_>
struct substrings_find_from_sz {

    /** The compiled vocabulary this arm walks, policy included. */
    substrings_engine_t &engine;
    function_type_ find_kernel;

    /** The haystacks one round walks. */
    substrings_corpus_t const &corpus;

    /** Borrowed view of the strings read by this arm. */
    sz_sequence_t haystacks;

    /** @b [haystacks + 1] boundaries into @c matches. */
    unified_vector<sz_size_t> offsets;

    /** Sized by a size query, so a round never grows it. */
    unified_vector<sz_substrings_match_t> matches;
    bool sized = false;

    substrings_find_from_sz(function_type_ kernel, substrings_engine_t &engine, substrings_corpus_t const &corpus,
                            sz_sequence_t const &haystacks)
        : engine(engine), find_kernel(kernel), corpus(corpus), haystacks(haystacks),
          offsets(corpus.views.size() + 1, unified_alloc<sz_size_t>(engine.engine.allocator, engine.stream)),
          matches(unified_alloc<sz_substrings_match_t>(engine.engine.allocator, engine.stream)) {}

    /** The size query, a whole walk, run once the filter keeps the row. */
    void preprocess() {
        if (sized) return;
        if (find_kernel(&engine.engine, &haystacks, nullptr, 0, offsets.data(), engine.stream) != sz_success_k)
            throw std::runtime_error("The reporting round could not be sized.");
        engine.join();
        // The boundaries name the survivors, which a cover thins below what the sizing walk emitted.
        matches.resize(offsets[corpus.views.size()]);
        sized = true;
    }

    call_result_t operator()(std::size_t) {
        if (find_kernel(&engine.engine, &haystacks, matches.data(), matches.size(), offsets.data(), engine.stream) !=
            sz_success_k)
            throw std::runtime_error("The reporting round failed.");
        engine.join();
        return corpus.round((check_value_t)engine.engine.report->matches_stored);
    }
};

/** Rewrites the whole corpus onto one tape its own backend sized. */
template <typename function_type_>
struct substrings_replace_from_sz {

    /** The compiled vocabulary this arm walks, policy included. */
    substrings_engine_t &engine;
    function_type_ replace_kernel;

    /** The haystacks one round walks. */
    substrings_corpus_t const &corpus;

    /** Borrowed view of the strings read by this arm. */
    sz_sequence_t haystacks;

    /** Accessors over the replacements, of the same kind. */
    sz_sequence_t replacements;

    /** @b [haystacks + 1] rewritten boundaries, the last the total. */
    unified_vector<sz_size_t> offsets;

    /** Sized by a size query, so a round never grows it. */
    unified_vector<char> tape;
    bool sized = false;

    substrings_replace_from_sz(function_type_ kernel, substrings_engine_t &engine, substrings_corpus_t const &corpus,
                               sz_sequence_t const &haystacks, sz_sequence_t const &replacements)
        : engine(engine), replace_kernel(kernel), corpus(corpus), haystacks(haystacks), replacements(replacements),
          offsets(corpus.views.size() + 1, unified_alloc<sz_size_t>(engine.engine.allocator, engine.stream)),
          tape(unified_alloc<char>(engine.engine.allocator, engine.stream)) {}

    /** The size query, a whole walk, run once the filter keeps the row. */
    void preprocess() {
        if (sized) return;
        if (replace_kernel(&engine.engine, &haystacks, &replacements, nullptr, 0, offsets.data(), engine.stream) !=
            sz_success_k)
            throw std::runtime_error("The rewriting round could not be sized.");
        engine.join();
        tape.resize(engine.engine.report->target_length);
        sized = true;
    }

    call_result_t operator()(std::size_t) {
        if (replace_kernel(&engine.engine, &haystacks, &replacements, tape.data(), tape.size(), offsets.data(),
                           engine.stream) != sz_success_k)
            throw std::runtime_error("The rewriting round failed.");
        engine.join();
        return corpus.round((check_value_t)engine.engine.report->target_length);
    }
};

/** Scores every haystack against the whole vocabulary as one BM25 query, into scores it owns. */
template <typename function_type_>
struct substrings_bm25_from_sz {

    /** The compiled vocabulary, which is the query. */
    substrings_engine_t &engine;
    function_type_ scores_kernel;

    /** The haystacks one round scores. */
    substrings_corpus_t const &corpus;

    /** Borrowed view of the strings read by this arm. */
    sz_sequence_t haystacks;

    /** The customary @c k1 and @c b, over the corpus's mean bytes. */
    sz_substrings_bm25_t parameters;

    /** @b [needles], all one: a weight scales a term, not the walk. */
    unified_vector<sz_f32_t> weights;

    /** @b [haystacks], what a round fills, on a device with memory of its own. */
    device_vector<sz_f32_t> device_scores;

    /** @b [haystacks], what the host checks. */
    pinned_vector<sz_f32_t> scores;
    std::optional<std::reference_wrapper<device_backend_t const>> runtime;

    substrings_bm25_from_sz(function_type_ kernel, substrings_engine_t &engine, substrings_corpus_t const &corpus,
                            sz_sequence_t const &haystacks,
                            std::optional<std::reference_wrapper<device_backend_t const>> runtime = {})
        : engine(engine), scores_kernel(kernel), corpus(corpus), haystacks(haystacks), parameters {1.2f, 0.75f, 0},
          weights(engine.engine.needles_count, 1.0f, unified_alloc<sz_f32_t>(engine.engine.allocator, engine.stream)),
          runtime(runtime) {
        std::size_t const bytes = corpus.bytes();
        parameters.average_document_length = (sz_f32_t)bytes / (sz_f32_t)std::max<std::size_t>(corpus.views.size(), 1);
        bool const separate = runtime && runtime->get().separate();
        scores = pinned_vector<sz_f32_t>(
            corpus.views.size(), 0.0f,
            pinned_alloc<sz_f32_t>(separate ? runtime->get().pinned : engine.engine.allocator, engine.stream));
        if (!separate) return;
        device_scores = device_vector<sz_f32_t>(device_alloc<sz_f32_t>(runtime->get().device, engine.stream));
        if (device_scores.resize_uninitialized(corpus.views.size()) != sz::status_t::success_k)
            throw std::runtime_error("The device would not hold the scores.");
    }

    call_result_t operator()(std::size_t) {
        if (scores_kernel(&engine.engine, &haystacks, nullptr, &parameters, weights.data(),
                          device_scores.size() == 0 ? scores.data() : device_scores.data(), 1, engine.stream) != sz_success_k)
            throw std::runtime_error("The scoring round failed.");
        if (device_scores.size() != 0 &&
            copy_device_to_host(device_scores, std::span<sz_f32_t>(scores), runtime->get()) != sz_success_k)
            throw std::runtime_error("The scores would not come back.");
        engine.join();
        // Backends round their sums differently, so the check is which haystacks scored rather than how much.
        check_value_t scored = 0;
        for (sz_f32_t const score : scores) scored += score > 0;
        return corpus.round(scored);
    }
};

#pragma endregion Arms

} // namespace ashvardanian::stringzilla::bench
