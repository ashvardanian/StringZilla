/**
 *  @file cpp/stringzilla/regex.hpp
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief Multi-pattern regular expressions: one compiled engine over many haystacks.
 */
#ifndef STRINGZILLA_REGEX_HPP_
#define STRINGZILLA_REGEX_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/regex.h" // `sz_regex_engine_init`, `sz_regex_counts`

#include "stringzilla/strings.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A vocabulary of patterns compiled once, then matched over any number of haystack batches.
 *  @sa sz_regex_engine_init, sz_regex_contains, sz_regex_counts,
 *      sz_regex_find, sz_regex_replace, sz_regex_extract
 */
class regex_engine_t {
    sz_regex_engine_t raw_ {};

  public:
    /** Holds nothing, so every verb refuses. */
    regex_engine_t() noexcept = default;
    regex_engine_t(regex_engine_t &&other) noexcept : raw_(std::exchange(other.raw_, {})) {}
    regex_engine_t &operator=(regex_engine_t &&other) noexcept {
        std::swap(raw_, other.raw_);
        return *this;
    }
    ~regex_engine_t() noexcept { sz_regex_engine_free(&raw_, nullptr); }

    /**
     *  @brief Compiles @p patterns for the best capability of @p capabilities.
     *  @param[in] patterns The vocabulary; the lower index wins a tie at one position.
     *  @param[in] case_sensitivity Whether patterns match bytes as written, or fold both sides.
     *  @param[out] malformed_index The first refused pattern, written on @c invalid_syntax_k.
     *  @param[in] allocator Source of the engine's blocks, kept until it is destroyed, or null for
     *      the unified allocator of @p capabilities.
     *  @param[in] capabilities Capabilities to pick from; zero compiles for the serial kernels,
     *      which header-only builds also have.
     *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
     */
    static expected<regex_engine_t> make(strings_view_t patterns, sz_case_sensitivity_t case_sensitivity,
                                         std::size_t &malformed_index, sz_allocator_t *allocator = nullptr,
                                         sz_capability_t capabilities = default_capabilities(),
                                         sz_stream_t stream = nullptr) noexcept {
        expected<regex_engine_t> made {{}, status_t::success_k};
        made.status = static_cast<status_t>(sz_regex_engine_init(&made.value.raw_, &patterns.raw, case_sensitivity,
                                                                 &malformed_index, capabilities, allocator, stream));
        return made;
    }

    /** Patterns the vocabulary holds, which bounds every reported @c needle_index. */
    std::size_t patterns_count() const noexcept { return raw_.patterns_count; }

    /** The C engine, for a caller that runs one kernel of its own choice over it. */
    sz_regex_engine_t &raw() noexcept { return raw_; }

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
        return static_cast<status_t>(sz_regex_contains(&raw_, &haystacks.raw, contained.data(), stream));
    }

    /**
     *  @brief Counts every haystack's leftmost-first matches, haystack @c h starting
     *      @c h * @p counts_stride bytes into @p counts.
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
        return static_cast<status_t>(sz_regex_counts(&raw_, &haystacks.raw, counts.data(), counts_stride, stream));
    }

    /**
     *  @brief Reports every leftmost-first match, ascending by haystack, then by start, as many as
     *      @p matches holds.
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
            sz_regex_find(&raw_, &haystacks.raw, matches.data(), matches.size(), matches_offsets.data(), stream));
    }

    /**
     *  @brief Rewrites every haystack, each match by its pattern's replacement, into one @p target.
     *  @param[in] haystacks The texts to rewrite.
     *  @param[in] replacements One replacement per pattern, naming groups as <tt>$N</tt>.
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
        return static_cast<status_t>(sz_regex_replace(&raw_, &haystacks.raw, &replacements.raw, target.data(),
                                                      target.size(), offsets.data(), stream));
    }

    /**
     *  @brief Locates the groups of each haystack's first match of @p pattern_index, a row for the
     *      whole match and one per group that took part, as many as @p groups holds.
     *  @param[in] haystacks The texts to search.
     *  @param[in] pattern_index The pattern whose groups to locate.
     *  @param[out] groups Room for the rows; an empty span sizes only.
     *  @param[out] groups_offsets One boundary per haystack plus one, the last the total, filled
     *      whether or not the rows fit.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p groups_offsets holds no more boundaries than haystacks.
     */
    status_t extract(strings_view_t haystacks, std::size_t pattern_index, std::span<sz_match_t> groups,
                     std::span<std::size_t> groups_offsets, sz_stream_t stream = nullptr) noexcept {
        if (groups_offsets.size() <= haystacks.size()) return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(sz_regex_extract(&raw_, &haystacks.raw, pattern_index, groups.data(),
                                                      groups.size(), groups_offsets.data(), stream));
    }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_REGEX_HPP_
