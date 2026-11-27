/**
 *  @file cpp/stringzilla/substrings.hpp
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief Multi-pattern substring search: one compiled Aho-Corasick engine over many haystacks.
 */
#ifndef STRINGZILLA_SUBSTRINGS_HPP_
#define STRINGZILLA_SUBSTRINGS_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/substrings.h" // `sz_substrings_engine_init`, `sz_substrings_counts`

#include "stringzilla/strings.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A vocabulary of needles compiled once, then streamed over any number of haystack batches.
 *  @sa sz_substrings_engine_init, sz_substrings_contains, sz_substrings_counts, sz_substrings_find,
 *      sz_substrings_replace, sz_substrings_bm25_scores
 */
class substrings_engine_t {
    sz_substrings_engine_t raw_ {};

  public:
    /** Holds nothing, so every verb refuses. */
    substrings_engine_t() noexcept = default;
    substrings_engine_t(substrings_engine_t &&other) noexcept : raw_(std::exchange(other.raw_, {})) {}
    substrings_engine_t &operator=(substrings_engine_t &&other) noexcept {
        std::swap(raw_, other.raw_);
        return *this;
    }
    ~substrings_engine_t() noexcept { sz_substrings_engine_free(&raw_, nullptr); }

    /**
     *  @brief Compiles @p needles for the best capability of @p capabilities.
     *  @param[in] needles The vocabulary, none of it empty.
     *  @param[in] case_sensitivity Whether both sides are folded before they meet, or kept as is.
     *  @param[in] overlap_policy The cover every round runs.
     *  @param[in] hot_states States kept in the dense hot rows, or
     *      @ref STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO to fill a fixed byte budget.
     *  @param[in] matches_budget Matches one device round may emit, zero asking for a default.
     *  @param[in] haystacks_budget Haystacks one device round may carry, zero asking for a default.
     *  @param[in] allocator Source of the engine's blocks, kept until it is destroyed, or null for
     *      the unified allocator of @p capabilities.
     *  @param[in] capabilities Capabilities to pick from; zero compiles for the serial kernels,
     *      which header-only builds also have.
     *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
     */
    static expected<substrings_engine_t> make(strings_view_t needles, sz_case_sensitivity_t case_sensitivity,
                                              sz_substrings_overlap_policy_t overlap_policy, std::size_t hot_states,
                                              std::size_t matches_budget, std::size_t haystacks_budget,
                                              sz_allocator_t *allocator = nullptr,
                                              sz_capability_t capabilities = default_capabilities(),
                                              sz_stream_t stream = nullptr) noexcept {
        expected<substrings_engine_t> made {{}, status_t::success_k};
        made.status = static_cast<status_t>(
            sz_substrings_engine_init(&made.value.raw_, &needles.raw, case_sensitivity, overlap_policy, hot_states,
                                      matches_budget, haystacks_budget, capabilities, allocator, stream));
        return made;
    }

    /** Needles the vocabulary holds, which bounds every reported @c needle_index. */
    std::size_t needles_count() const noexcept { return raw_.needles_count; }

    /** The cover every round runs, fixed when the engine was compiled. */
    sz_substrings_overlap_policy_t overlap_policy() const noexcept { return raw_.overlap_policy; }

    /** What the last round found, read once its stream is joined; zeros for an empty engine. */
    sz_substrings_report_t report() const noexcept { return raw_.report ? *raw_.report : sz_substrings_report_t {}; }

    /** The C engine, for a caller that runs one kernel of its own choice over it. */
    sz_substrings_engine_t &raw() noexcept { return raw_; }

    /**
     *  @brief Marks which haystacks hold any match, one bit each, the lowest bit first.
     *  @param[in] haystacks The texts to search.
     *  @param[out] contained At least one bit per haystack.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p contained holds fewer bits than haystacks.
     */
    status_t contains(strings_view_t haystacks, std::span<sz_u1x8_t> contained, sz_stream_t stream = nullptr) noexcept {
        if (contained.size() < haystacks.size() / 8 + (haystacks.size() % 8 != 0))
            return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(sz_substrings_contains(&raw_, &haystacks.raw, contained.data(), stream));
    }

    /**
     *  @brief Counts every haystack's matches, haystack @c h at @c h * @p counts_stride bytes in.
     *  @param[in] haystacks The texts to search.
     *  @param[out] counts The per-haystack counts.
     *  @param[in] counts_stride Bytes from one haystack's count to the next.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p counts is shorter than the strided column.
     */
    status_t counts(strings_view_t haystacks, std::span<std::size_t> counts, std::size_t counts_stride,
                    sz_stream_t stream = nullptr) noexcept {
        if (counts.size_bytes() < strided_bytes_(haystacks.size(), counts_stride, sizeof(std::size_t)))
            return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(sz_substrings_counts(&raw_, &haystacks.raw, counts.data(), counts_stride, stream));
    }

    /**
     *  @brief Reports every match, ascending by haystack, as many as @p matches holds.
     *  @param[in] haystacks The texts to search.
     *  @param[out] matches Room for the matches; an empty span sizes only.
     *  @param[out] matches_offsets One boundary per haystack plus one, the last the total, filled
     *      whether or not the matches fit.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p matches_offsets holds no more boundaries than haystacks.
     */
    status_t find(strings_view_t haystacks, std::span<sz_match_t> matches, std::span<std::size_t> matches_offsets,
                  sz_stream_t stream = nullptr) noexcept {
        if (matches_offsets.size() <= haystacks.size()) return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(
            sz_substrings_find(&raw_, &haystacks.raw, matches.data(), matches.size(), matches_offsets.data(), stream));
    }

    /**
     *  @brief Rewrites every haystack with one replacement per needle, into one @p target.
     *  @param[in] haystacks The texts to rewrite.
     *  @param[in] replacements One replacement per needle; an empty one deletes.
     *  @param[out] target Room for the rewrite; an empty span sizes only.
     *  @param[out] offsets One rewritten boundary per haystack plus one, the last the total, filled
     *      whether or not @p target held the rewrite.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p offsets holds no more boundaries than haystacks.
     */
    status_t replace(strings_view_t haystacks, strings_view_t replacements, std::span<char> target,
                     std::span<std::size_t> offsets, sz_stream_t stream = nullptr) noexcept {
        if (offsets.size() <= haystacks.size()) return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(sz_substrings_replace(&raw_, &haystacks.raw, &replacements.raw, target.data(),
                                                           target.size(), offsets.data(), stream));
    }

    /**
     *  @brief Scores every haystack against the vocabulary as one BM25 query, haystack @c h at
     *      @c h * @p scores_stride bytes into @p scores.
     *  @param[in] haystacks The documents to score.
     *  @param[in] document_lengths One length per haystack to normalize by, or empty for each
     *      haystack's byte length.
     *  @param[in] parameters BM25's continuous parameters.
     *  @param[in] needle_weights One weight per needle.
     *  @param[out] scores The per-haystack scores.
     *  @param[in] scores_stride Bytes from one haystack's score to the next.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k A non-empty @p document_lengths, @p needle_weights or @p scores
     *    shorter than its axis.
     */
    status_t bm25_scores(strings_view_t haystacks, std::span<float const> document_lengths,
                         sz_substrings_bm25_t const &parameters, std::span<float const> needle_weights,
                         std::span<float> scores, std::size_t scores_stride, sz_stream_t stream = nullptr) noexcept {
        if (!document_lengths.empty() && document_lengths.size() < haystacks.size())
            return status_t::unexpected_dimensions_k;
        if (needle_weights.size() < raw_.needles_count) return status_t::unexpected_dimensions_k;
        if (scores.size_bytes() < strided_bytes_(haystacks.size(), scores_stride, sizeof(float)))
            return status_t::unexpected_dimensions_k;
        float const *lengths = document_lengths.empty() ? nullptr : document_lengths.data();
        return static_cast<status_t>(sz_substrings_bm25_scores(
            &raw_, &haystacks.raw, lengths, &parameters, needle_weights.data(), scores.data(), scores_stride, stream));
    }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SUBSTRINGS_HPP_
