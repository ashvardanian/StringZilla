/**
 *  @file cpp/stringzilla/subdocuments.hpp
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief JSON and XML path queries: one compiled engine over many documents.
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_HPP_
#define STRINGZILLA_SUBDOCUMENTS_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/subdocuments.h" // `sz_subdocuments_engine_init`, `sz_subdocuments_counts`

#include "stringzilla/strings.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A vocabulary of JSON or XML paths compiled once, then walked over document batches.
 *  @sa sz_subdocuments_engine_init, sz_subdocuments_contains, sz_subdocuments_counts,
 *      sz_subdocuments_find
 */
class subdocuments_engine_t {
    sz_subdocuments_engine_t raw_ {};

  public:
    /** Holds nothing, so every verb refuses. */
    subdocuments_engine_t() noexcept = default;
    subdocuments_engine_t(subdocuments_engine_t &&other) noexcept : raw_(std::exchange(other.raw_, {})) {}
    subdocuments_engine_t &operator=(subdocuments_engine_t &&other) noexcept {
        std::swap(raw_, other.raw_);
        return *this;
    }
    ~subdocuments_engine_t() noexcept { sz_subdocuments_engine_free(&raw_, nullptr); }

    /**
     *  @brief Compiles @p paths for the best capability of @p capabilities.
     *  @param[in] paths The vocabulary, one RFC 9535 JSON path or XPath 1.0 path per string.
     *  @param[in] format The grammar of the documents every verb reads.
     *  @param[in] options How JSON walks read what RFC 8259 leaves open.
     *  @param[out] malformed_index The first path breaking the grammar or the subset, on failure.
     *  @param[in] allocator Source of the engine's blocks, kept until it is destroyed, or null for
     *      the unified allocator of @p capabilities.
     *  @param[in] capabilities Capabilities to pick from; zero compiles for the serial kernels,
     *      which header-only builds also have.
     *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
     */
    static expected<subdocuments_engine_t> make(strings_view_t paths, sz_subdocuments_format_t format,
                                                sz_subdocuments_options_t options, std::size_t &malformed_index,
                                                sz_allocator_t *allocator = nullptr,
                                                sz_capability_t capabilities = default_capabilities(),
                                                sz_stream_t stream = nullptr) noexcept {
        expected<subdocuments_engine_t> made {{}, status_t::success_k};
        made.status = static_cast<status_t>(sz_subdocuments_engine_init(
            &made.value.raw_, &paths.raw, format, options, &malformed_index, capabilities, allocator, stream));
        return made;
    }

    /** Paths the vocabulary holds, which bounds every reported @c needle_index. */
    std::size_t paths_count() const noexcept { return raw_.paths_count; }

    /** The grammar every document of every verb follows. */
    sz_subdocuments_format_t format() const noexcept { return raw_.format; }

    /** The C engine, for a caller that runs one kernel of its own choice over it. */
    sz_subdocuments_engine_t &raw() noexcept { return raw_; }

    /**
     *  @brief Marks which documents hold a node some path selects, one bit each, lowest first.
     *  @param[in] documents One document per string.
     *  @param[out] contained At least one bit per document.
     *  @param[out] malformed_index The first malformed document, on @c invalid_syntax_k.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p contained holds fewer bits than documents.
     */
    status_t contains(strings_view_t documents, std::span<sz_u1x8_t> contained, std::size_t &malformed_index,
                      sz_stream_t stream = nullptr) noexcept {
        if (contained.size() < documents.size() / 8 + (documents.size() % 8 != 0))
            return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(
            sz_subdocuments_contains(&raw_, &documents.raw, contained.data(), &malformed_index, stream));
    }

    /**
     *  @brief Counts every document's selected nodes, once per path, document @c d starting
     *      @c d * @p counts_stride bytes into @p counts.
     *  @param[in] documents One document per string.
     *  @param[out] counts The per-document counts.
     *  @param[in] counts_stride Bytes from one document's count to the next.
     *  @param[out] malformed_index The first malformed document, on @c invalid_syntax_k.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p counts is shorter than the strided column.
     */
    status_t counts(strings_view_t documents, std::span<std::size_t> counts, std::size_t counts_stride,
                    std::size_t &malformed_index, sz_stream_t stream = nullptr) noexcept {
        if (counts.size_bytes() < strided_bytes_(documents.size(), counts_stride, sizeof(std::size_t)))
            return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(
            sz_subdocuments_counts(&raw_, &documents.raw, counts.data(), counts_stride, &malformed_index, stream));
    }

    /**
     *  @brief Reports every selected node by document, start and path, as many as @p matches holds.
     *  @param[in] documents One document per string.
     *  @param[out] matches Room for the matches; an empty span sizes only.
     *  @param[out] matches_offsets One boundary per document plus one, the last the total, filled
     *      whether or not the matches fit.
     *  @param[out] malformed_index The first malformed document, on @c invalid_syntax_k.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p matches_offsets holds no more boundaries than documents.
     */
    status_t find(strings_view_t documents, std::span<sz_match_t> matches, std::span<std::size_t> matches_offsets,
                  std::size_t &malformed_index, sz_stream_t stream = nullptr) noexcept {
        if (matches_offsets.size() <= documents.size()) return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(sz_subdocuments_find(&raw_, &documents.raw, matches.data(), matches.size(),
                                                          matches_offsets.data(), &malformed_index, stream));
    }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SUBDOCUMENTS_HPP_
