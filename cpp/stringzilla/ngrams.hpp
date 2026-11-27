/**
 *  @file cpp/stringzilla/ngrams.hpp
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief N-gram overlaps of prepared queries with batches of candidates, and MinHash sketches.
 */
#ifndef STRINGZILLA_NGRAMS_HPP_
#define STRINGZILLA_NGRAMS_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include <cstdint> // `std::uint8_t`

#include "stringzilla/ngrams.h" // `sz_ngrams_engine_init`, `sz_ngrams_overlaps`, `sz_ngrams_sketches`

#include "stringzilla/strings.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief Queries filed at one or more N-gram widths, overlapped with any number of candidate
 *      batches, and sketching any batch of texts at those widths.
 *  @sa sz_ngrams_engine_init, sz_ngrams_overlaps, sz_ngrams_sketches
 */
class ngrams_engine_t {
    sz_ngrams_engine_t raw_ {};

  public:
    /** Holds nothing, so every verb refuses. */
    ngrams_engine_t() noexcept = default;
    ngrams_engine_t(ngrams_engine_t &&other) noexcept : raw_(std::exchange(other.raw_, {})) {}
    ngrams_engine_t &operator=(ngrams_engine_t &&other) noexcept {
        std::swap(raw_, other.raw_);
        return *this;
    }
    ~ngrams_engine_t() noexcept { sz_ngrams_engine_free(&raw_, nullptr); }

    /**
     *  @brief Files @p queries at every width in @p window_widths with the best of @p capabilities.
     *  @param[in] queries The texts whose N-grams are filed, or none to only sketch.
     *  @param[in] window_widths Window widths in bytes, each at least one, the last axis of
     *      every overlap output.
     *  @param[in] candidates_budget Candidates one round may carry where a backend keeps state per
     *      candidate, zero asking for its default.
     *  @param[in] allocator Source of the engine's blocks, kept until it is destroyed, or null for
     *      the unified allocator of @p capabilities.
     *  @param[in] capabilities Capabilities to pick from.
     *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
     */
    static expected<ngrams_engine_t> make(strings_view_t queries, std::span<std::size_t const> window_widths,
                                          std::size_t candidates_budget, sz_allocator_t *allocator = nullptr,
                                          sz_capability_t capabilities = default_capabilities(),
                                          sz_stream_t stream = nullptr) noexcept {
        expected<ngrams_engine_t> made {{}, status_t::success_k};
        made.status = static_cast<status_t>(sz_ngrams_engine_init(&made.value.raw_, &queries.raw, window_widths.data(),
                                                                  window_widths.size(), candidates_budget, capabilities,
                                                                  allocator, stream));
        return made;
    }

    /** Queries filed, the first axis of every overlap output, or zero to only sketch. */
    std::size_t queries_count() const noexcept { return raw_.count; }

    /** Window widths, the last axis of every overlap output. */
    std::size_t widths_count() const noexcept { return raw_.widths_count; }

    /**
     *  @brief Shares of every query's distinct N-grams each candidate holds, per width, into a
     *      @b [queries, candidates, widths] block with a unit-strided width axis.
     *  @param[in] candidates The texts probed against the queries.
     *  @param[in] radius The share of a query's distinct N-grams a pair may miss and stay exact.
     *  @param[out] shares The block, device-reachable on a device.
     *  @param[in] shares_query_stride Bytes from one query's plane to the next.
     *  @param[in] shares_candidate_stride Bytes from one candidate's row to the next.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p shares is shorter than the block.
     */
    status_t overlaps(strings_view_t candidates, float radius, std::span<float> shares, std::size_t shares_query_stride,
                      std::size_t shares_candidate_stride, sz_stream_t stream = nullptr) noexcept {
        std::size_t const row_bytes = strided_bytes_(raw_.widths_count, sizeof(float), sizeof(float));
        std::size_t const plane_bytes = strided_bytes_(candidates.size(), shares_candidate_stride, row_bytes);
        if (shares.size_bytes() < strided_bytes_(raw_.count, shares_query_stride, plane_bytes))
            return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(sz_ngrams_overlaps(&raw_, &candidates.raw, radius, shares.data(),
                                                        shares_query_stride, shares_candidate_stride, stream));
    }

    /**
     *  @brief One MinHash sketch of @p slots bytes per text of @p texts, at the engine's widths.
     *  @param[in] texts The texts to sketch.
     *  @param[in] slots Bytes of one sketch.
     *  @param[out] sketches The sketches, device-reachable on a device.
     *  @param[in] sketches_stride Bytes from one sketch to the next.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k @p sketches is shorter than the sketches.
     */
    status_t sketches(strings_view_t texts, std::size_t slots, std::span<std::uint8_t> sketches,
                      std::size_t sketches_stride, sz_stream_t stream = nullptr) noexcept {
        if (sketches.size_bytes() < strided_bytes_(texts.size(), sketches_stride, slots))
            return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(
            sz_ngrams_sketches(&raw_, &texts.raw, slots, sketches.data(), sketches_stride, stream));
    }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_NGRAMS_HPP_
