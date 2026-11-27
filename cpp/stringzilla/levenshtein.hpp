/**
 *  @file cpp/stringzilla/levenshtein.hpp
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief Levenshtein edit distances from a prepared batch of queries to batches of candidates.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_HPP_
#define STRINGZILLA_LEVENSHTEIN_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/levenshtein.h" // `sz_levenshtein_engine_init`, `sz_levenshtein_distances`

#include "stringzilla/strings.hpp"

#include "stringzilla/devices.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief A prepared batch of queries, scored against any number of candidate batches.
 *  @sa sz_levenshtein_engine_init, sz_levenshtein_distances
 */
class levenshtein_engine_t {
    sz_levenshtein_engine_t raw_ {};

  public:
    /** Holds nothing, so every verb refuses. */
    levenshtein_engine_t() noexcept = default;
    levenshtein_engine_t(levenshtein_engine_t &&other) noexcept : raw_(std::exchange(other.raw_, {})) {}
    levenshtein_engine_t &operator=(levenshtein_engine_t &&other) noexcept {
        std::swap(raw_, other.raw_);
        return *this;
    }
    ~levenshtein_engine_t() noexcept { sz_levenshtein_engine_free(&raw_, nullptr); }

    /**
     *  @brief Prepares @p queries for the best capability of @p capabilities.
     *  @param[in] queries The patterns every candidate is scored against.
     *  @param[in] symbol Whether a distance counts bytes or UTF-8 runes.
     *  @param[in] allocator Source of the engine's blocks, kept until it is destroyed, or null for
     *      the unified allocator of @p capabilities.
     *  @param[in] capabilities Capabilities to pick from.
     *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
     */
    static expected<levenshtein_engine_t> make(strings_view_t queries, sz_levenshtein_symbol_t symbol,
                                               sz_allocator_t *allocator = nullptr,
                                               sz_capability_t capabilities = default_capabilities(),
                                               sz_stream_t stream = nullptr) noexcept {
        expected<levenshtein_engine_t> made {{}, status_t::success_k};
        made.status = static_cast<status_t>(
            sz_levenshtein_engine_init(&made.value.raw_, &queries.raw, symbol, capabilities, allocator, stream));
        return made;
    }

    /** Queries the batch holds, which is the first axis of every output. */
    std::size_t queries_count() const noexcept { return raw_.count; }

    /** The alphabet the batch was prepared over. */
    sz_levenshtein_symbol_t symbol() const noexcept { return raw_.symbol; }

    /** The C engine, for a caller that runs one kernel of its own choice over it. */
    sz_levenshtein_engine_t &raw() noexcept { return raw_; }

    /**
     *  @brief Distances into a @b [queries, candidates] block, rows a stride apart.
     *  @param[in] candidates The texts every query is scored against.
     *  @param[in] bound The largest distance reported exactly; a candidate past it reports
     *      @p bound + 1. @c SIZE_MAX keeps every distance exact.
     *  @param[out] distances The block, device-reachable on a device.
     *  @param[in] distances_stride Bytes from one query's row to the next.
     *  @param[in] stream Null on the CPU, or the stream of the engine's device.
     *
     *  @par Statuses:
     *  - @c unexpected_dimensions_k A stride under one row, or @p distances shorter than the block.
     */
    status_t distances(strings_view_t candidates, std::size_t bound, std::span<std::size_t> distances,
                       std::size_t distances_stride, sz_stream_t stream = nullptr) noexcept {
        std::size_t const row_bytes = strided_bytes_(candidates.size(), sizeof(std::size_t), sizeof(std::size_t));
        if (distances_stride < row_bytes) return status_t::unexpected_dimensions_k;
        if (distances.size_bytes() < strided_bytes_(raw_.count, distances_stride, row_bytes))
            return status_t::unexpected_dimensions_k;
        return static_cast<status_t>(sz_levenshtein_distances(&raw_, &candidates.raw, nullptr, nullptr, bound,
                                                              distances.data(), distances_stride, stream));
    }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_LEVENSHTEIN_HPP_
