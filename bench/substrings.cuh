/**
 *  @file bench/substrings.cuh
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief Shared code for CPU and GPU multi-pattern search benchmarks: the vocabulary, the corpus,
 *      and the arms.
 */
#include <algorithm>   // `std::sort`, `std::min`, `std::max`
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
static std::vector<std::string> substrings_sampled(environment_t const &env) {
    std::mt19937_64 generator(env.seed);
    std::set<std::string> distinct;
    std::size_t attempts = 0;
    while (distinct.size() != substrings_sampled_needles_k && attempts++ != 64 * substrings_sampled_needles_k) {
        token_view_t const token = env.tokens[generator() % env.tokens.size()];
        std::size_t const length = substrings_sampled_min_bytes_k +
                                   generator() % (substrings_sampled_max_bytes_k - substrings_sampled_min_bytes_k + 1);
        if (token.size() < length) continue;
        distinct.emplace(token.data() + generator() % (token.size() - length + 1), length);
    }
    return {distinct.begin(), distinct.end()};
}

/**
 *  @brief One percent of the post-cutoff vocabulary, taken from one end of the frequency ranking.
 *
 *  Both noisy ends are removed before any slice is taken. The most frequent one percent are
 *  stopwords that every haystack holds, which would measure the reporting path rather than the
 *  automaton; terms occurring once are noise no haystack reaches twice. The top cutoff is a
 *  fraction of @b all distinct terms, hapax included, so the slice stays where it is whenever the
 *  hapax filter moves.
 *
 *  Drawn from the dataset's words whatever tokenization shapes the haystacks, so a line search and
 *  a word search draw needles from the same vocabulary.
 */
static std::vector<std::string> substrings_vocabulary(environment_t const &env, substrings_slice_t slice) {
    if (slice == substrings_slice_t::sampled_k) return substrings_sampled(env);
    tokens_t const tokenized = tokenize(env.dataset);
    std::vector<std::string_view> words;
    std::vector<std::string> distinct;
    std::vector<std::size_t> frequencies;
    std::size_t distinct_total = 0;

    words.reserve(tokenized.size());
    for (auto const &word : tokenized) {
        if (word.size() < substrings_min_word_bytes_k || word.size() > substrings_max_word_bytes_k) continue;
        words.emplace_back(word.data(), word.size());
    }
    std::sort(words.begin(), words.end());
    for (std::size_t index = 0; index != words.size();) {
        std::size_t run = index;
        while (run != words.size() && words[run] == words[index]) ++run;
        ++distinct_total;
        if (run - index >= substrings_minimum_occurrences_k)
            distinct.emplace_back(words[index]), frequencies.push_back(run - index);
        index = run;
    }
    if (distinct.empty()) return {}; // ? A corpus without repeated words, such as nucleotides, has no word slice.

    // Frequency first, then content, so the ranking is deterministic across runs and platforms.
    std::vector<std::size_t> order(distinct.size());
    for (std::size_t index = 0; index != order.size(); ++index) order[index] = index;
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        if (frequencies[left] != frequencies[right]) return frequencies[left] > frequencies[right];
        return distinct[left] < distinct[right];
    });

    std::size_t const dropped = std::min<std::size_t>(
        (std::size_t)((double)distinct_total * substrings_frequent_cutoff_k), order.size());
    std::size_t const available = order.size() - dropped;
    std::size_t const wanted = std::max<std::size_t>(available / 100, 1);
    std::size_t const first = slice == substrings_slice_t::frequent_k ? dropped : dropped + available - wanted;
    std::vector<std::string> vocabulary;
    for (std::size_t rank = first; rank != first + wanted; ++rank) vocabulary.push_back(distinct[order[rank]]);
    return vocabulary;
}

#pragma endregion Vocabulary

#pragma region Inputs

/** One vocabulary slice, and one replacement per needle for a rewrite to substitute. */
struct substrings_dictionary_t {

    /** The words this dictionary was compiled from. */
    std::vector<std::string> needles;

    /** Views over @c needles, which only the builder reads. */
    std::vector<sz_string_view_t> needle_views;

    /** Every replacement back to back, where a kernel reads. */
    unified_vector<char> replacement_bytes;

    /** One per needle, alternating with an empty deletion. */
    unified_vector<sz_string_view_t> replacement_views;

    /** Host accessors over @c needle_views. */
    sz_sequence_t needle_sequence {};

    /** Host accessors over @c replacement_views. */
    sz_sequence_t replacements {};

    /** What every engine over this vocabulary is built from. */
    sz_memory_allocator_t allocator;

    /** Whether both sides fold before they meet. */
    sz_substrings_case_sensitivity_t sensitivity;

    /** Needle bytes, which a compilation row is scaled by. */
    std::size_t needle_bytes = 0;

    substrings_dictionary_t(environment_t const &env, substrings_slice_t slice,
                            sz_substrings_case_sensitivity_t sensitivity, sz_memory_allocator_t const &memory)
        : needles(substrings_vocabulary(env, slice)), needle_views(needles.size()), replacement_views(needles.size()),
          allocator(memory), sensitivity(sensitivity) {
        for (std::size_t index = 0; index != needles.size(); ++index) {
            std::string const replacement = index % 2 ? std::string() : "<" + std::to_string(index) + ">";
            replacement_bytes.insert(replacement_bytes.end(), replacement.begin(), replacement.end());
            replacement_views[index].length = replacement.size();
            needle_views[index] = {needles[index].data(), needles[index].size()};
            needle_bytes += needles[index].size();
        }
        // The arena's address is final only once it has stopped growing.
        std::size_t written = 0;
        for (sz_string_view_t &view : replacement_views)
            view.start = replacement_bytes.data() + written, written += view.length;
        sz_sequence_from_string_views(needle_views.data(), needle_views.size(), &needle_sequence);
        sz_sequence_from_string_views(replacement_views.data(), replacement_views.size(), &replacements);
    }
    substrings_dictionary_t(substrings_dictionary_t const &) = delete;
    substrings_dictionary_t &operator=(substrings_dictionary_t const &) = delete;
};

/** Where an engine's blocks live, which is also which of the two inits builds it. */
enum class substrings_residency_t {

    /** Plain host memory, walked by the CPU tiers. */
    host_k = 0,

    /** Memory a kernel addresses, with the round's arena beside it. */
    device_k = 1,
};

/** Compiles @p dictionary into a host engine, which every CPU tier's verbs then read. */
inline sz_status_t substrings_init_host(substrings_dictionary_t const &dictionary,
                                        sz_substrings_overlap_policy_t policy, sz_substrings_engine_t &engine) {
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    return sz_substrings_engine_init_cpu(&dictionary.needle_sequence, dictionary.sensitivity, policy,
                                         STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, &host, &engine);
}

#if STRINGZILLA_TARGET_CUDA

/** Compiles @p dictionary where a kernel reads it, through the dictionary's unified allocator. */
inline sz_status_t substrings_init_device(substrings_dictionary_t const &dictionary,
                                          sz_substrings_overlap_policy_t policy, sz_substrings_engine_t &engine) {
    sz_memory_allocator_t allocator = dictionary.allocator;
    return sz_substrings_engine_init_gpu(&dictionary.needle_sequence, dictionary.sensitivity, policy,
                                         STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, &allocator, nullptr, &engine);
}

/** Joins the default stream if a device engine could have enqueued on it; host engines never do. */
inline void substrings_join(sz_substrings_engine_t const &engine) {
    if (engine.capability & sz_caps_cuda_k) cudaStreamSynchronize(nullptr);
}
#else

/** A build with no device runtime cannot compile a device engine at all. */
inline sz_status_t substrings_init_device(substrings_dictionary_t const &dictionary,
                                          sz_substrings_overlap_policy_t policy, sz_substrings_engine_t &engine) {
    sz_unused_(dictionary), sz_unused_(policy), sz_unused_(engine);
    return sz_device_code_mismatch_k;
}

/** A build with no device runtime has nothing to join, as every engine it builds is a host one. */
inline void substrings_join(sz_substrings_engine_t const &engine) { sz_unused_(engine); }
#endif

/**
 *  @brief One vocabulary compiled under one policy, owned for as long as the arms that read it.
 *
 *  The policy sizes the arena, so it belongs to construction rather than to a call, and an arm
 *  therefore holds an engine rather than a vocabulary plus a policy argument.
 */
struct substrings_engine_t {

    /** The compiled vocabulary, its arena and its report. */
    sz_substrings_engine_t engine {};

    substrings_engine_t(substrings_dictionary_t const &dictionary, sz_substrings_overlap_policy_t policy,
                        substrings_residency_t residency) {
        sz_status_t const status = residency == substrings_residency_t::host_k
                                       ? substrings_init_host(dictionary, policy, engine)
                                       : substrings_init_device(dictionary, policy, engine);
        if (status != sz_success_k) throw std::runtime_error("The vocabulary would not compile.");
    }
    substrings_engine_t(substrings_engine_t const &) = delete;
    substrings_engine_t &operator=(substrings_engine_t const &) = delete;
    ~substrings_engine_t() noexcept { sz_substrings_engine_free(&engine); }
};

/** The corpus as one sequence of haystacks, over views the device reaches in a device build. */
struct substrings_corpus_t {

    /** One view per haystack, in corpus order. */
    unified_vector<sz_string_view_t> views;

    /** Host accessors over @c views. */
    sz_sequence_t haystacks {};

    /** Haystack bytes one round walks. */
    std::size_t bytes = 0;

    explicit substrings_corpus_t(environment_t const &env) : views(env.tokens.size()) {
        for (std::size_t index = 0; index != env.tokens.size(); ++index) {
            token_view_t const token = env.tokens[index];
            views[index] = {token.data(), token.size()};
            bytes += token.size();
        }
        sz_sequence_from_string_views(views.data(), views.size(), &haystacks);
    }

    /** What one round over every haystack reports: its bytes as both throughput and operations. */
    call_result_t round(check_value_t check_value) const {
        call_result_t result(bytes, check_value, bytes);
        result.inputs_processed = views.size();
        return result;
    }
};

#pragma endregion Inputs

#pragma region Arms

/** Counts every match in the whole corpus, one call per round, into counts it owns. */
template <sz_substrings_counts_t counts_>
struct substrings_counts_from_sz {

    /** The compiled vocabulary this arm walks, policy included. */
    sz_substrings_engine_t *engine;

    /** The haystacks one round walks. */
    substrings_corpus_t const &corpus;

    /** Host accessors for a CPU arm, device ones for a CUDA arm. */
    sz_sequence_t haystacks;

    /** @b [haystacks], what a round fills. */
    unified_vector<sz_size_t> counts;

    substrings_counts_from_sz(substrings_engine_t &engine, substrings_corpus_t const &corpus,
                              sz_sequence_t const &haystacks)
        : engine(&engine.engine), corpus(corpus), haystacks(haystacks), counts(corpus.views.size()) {}

    call_result_t operator()(std::size_t) {
        if (counts_(engine, &haystacks, counts.data(), 1) != sz_success_k)
            throw std::runtime_error("The counting round failed.");
        substrings_join(*engine);
        check_value_t mixed = 0;
        for (sz_size_t const count : counts) mixed = mixed * 31u + (check_value_t)count;
        return corpus.round(mixed);
    }
};

/** Reports every match in the whole corpus, into an array its own backend sized. */
template <sz_substrings_find_t find_>
struct substrings_find_from_sz {

    /** The compiled vocabulary this arm walks, policy included. */
    sz_substrings_engine_t *engine;

    /** The haystacks one round walks. */
    substrings_corpus_t const &corpus;

    /** Host accessors for a CPU arm, device ones for a CUDA arm. */
    sz_sequence_t haystacks;

    /** @b [haystacks + 1] boundaries into @c matches. */
    unified_vector<sz_size_t> offsets;

    /** Sized by a size query, so a round never grows it. */
    unified_vector<sz_substrings_match_t> matches;

    substrings_find_from_sz(substrings_engine_t &engine, substrings_corpus_t const &corpus,
                            sz_sequence_t const &haystacks)
        : engine(&engine.engine), corpus(corpus), haystacks(haystacks), offsets(corpus.views.size() + 1) {
        if (find_(this->engine, &haystacks, nullptr, 0, offsets.data()) != sz_success_k)
            throw std::runtime_error("The reporting round could not be sized.");
        substrings_join(*this->engine);
        // The boundaries name the survivors, which a cover thins below what the sizing walk emitted.
        matches.resize(offsets[corpus.views.size()]);
    }

    call_result_t operator()(std::size_t) {
        if (find_(engine, &haystacks, matches.data(), matches.size(), offsets.data()) != sz_success_k)
            throw std::runtime_error("The reporting round failed.");
        substrings_join(*engine);
        return corpus.round((check_value_t)engine->report->matches_stored);
    }
};

/** Rewrites the whole corpus onto one tape its own backend sized. */
template <sz_substrings_replace_t replace_>
struct substrings_replace_from_sz {

    /** The compiled vocabulary this arm walks, policy included. */
    sz_substrings_engine_t *engine;

    /** The haystacks one round walks. */
    substrings_corpus_t const &corpus;

    /** Host accessors for a CPU arm, device ones for a CUDA arm. */
    sz_sequence_t haystacks;

    /** Accessors over the replacements, of the same kind. */
    sz_sequence_t replacements;

    /** @b [haystacks + 1] rewritten boundaries, the last the total. */
    unified_vector<sz_size_t> offsets;

    /** Sized by a size query, so a round never grows it. */
    unified_vector<char> tape;

    substrings_replace_from_sz(substrings_engine_t &engine, substrings_corpus_t const &corpus,
                               sz_sequence_t const &haystacks, sz_sequence_t const &replacements)
        : engine(&engine.engine), corpus(corpus), haystacks(haystacks), replacements(replacements),
          offsets(corpus.views.size() + 1) {
        if (replace_(this->engine, &haystacks, &replacements, nullptr, 0, offsets.data()) != sz_success_k)
            throw std::runtime_error("The rewriting round could not be sized.");
        substrings_join(*this->engine);
        tape.resize(this->engine->report->tape_bytes);
    }

    call_result_t operator()(std::size_t) {
        if (replace_(engine, &haystacks, &replacements, tape.data(), tape.size(), offsets.data()) != sz_success_k)
            throw std::runtime_error("The rewriting round failed.");
        substrings_join(*engine);
        return corpus.round((check_value_t)engine->report->tape_bytes);
    }
};

/** Scores every haystack against the whole vocabulary as one BM25 query, into scores it owns. */
template <sz_substrings_bm25_scores_t scores_>
struct substrings_bm25_from_sz {

    /** The compiled vocabulary, which is the query. */
    sz_substrings_engine_t *engine;

    /** The haystacks one round scores. */
    substrings_corpus_t const &corpus;

    /** Host accessors for a CPU arm, device ones for a CUDA arm. */
    sz_sequence_t haystacks;

    /** The customary @c k1 and @c b, over the corpus's mean bytes. */
    sz_substrings_bm25_t parameters;

    /** @b [needles], all one: a weight scales a term, not the walk. */
    unified_vector<sz_f32_t> weights;

    /** @b [haystacks], what a round fills. */
    unified_vector<sz_f32_t> scores;

    substrings_bm25_from_sz(substrings_engine_t &engine, substrings_corpus_t const &corpus,
                            sz_sequence_t const &haystacks)
        : engine(&engine.engine), corpus(corpus), haystacks(haystacks),
          parameters {1.2f, 0.75f, (sz_f32_t)corpus.bytes / (sz_f32_t)std::max<std::size_t>(corpus.views.size(), 1)},
          weights(this->engine->needles_count, 1.0f), scores(corpus.views.size()) {}

    call_result_t operator()(std::size_t) {
        if (scores_(engine, &haystacks, nullptr, &parameters, weights.data(), scores.data(), 1) != sz_success_k)
            throw std::runtime_error("The scoring round failed.");
        substrings_join(*engine);
        // Backends round their sums differently, so the check is which haystacks scored rather than how much.
        check_value_t scored = 0;
        for (sz_f32_t const score : scores) scored += score > 0;
        return corpus.round(scored);
    }
};

#pragma endregion Arms

} // namespace ashvardanian::stringzilla::bench
