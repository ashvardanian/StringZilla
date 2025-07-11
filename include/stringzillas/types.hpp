/**
 *  @brief  Shared definitions for the StringCuZilla C++ library.
 *  @file   types.hpp
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLAS_TYPES_HPP_
#define STRINGZILLAS_TYPES_HPP_

#include <thread> // `std::thread::hardware_concurrency`

#include "stringzilla/types.hpp"

namespace ashvardanian {
namespace stringzillas {

using namespace ashvardanian::stringzilla;

enum bytes_per_cell_t : uint {
    zero_bytes_per_cell_k = 0,
    one_byte_per_cell_k = 1,
    two_bytes_per_cell_k = 2,
    four_bytes_per_cell_k = 4,
    eight_bytes_per_cell_k = 8,
};

struct dummy_executor_t {

    constexpr size_t threads_count() const noexcept { return 1; }

    /**
     *  @brief  Calls the @p function for each index from 0 to @p (n) in such
     *          a way that consecutive elements are likely to be processed by
     *          the same thread.
     */
    template <typename function_type_>
    inline void for_n(size_t n, function_type_ &&function) const noexcept {
        for (size_t i = 0; i < n; ++i) function(i);
    }

    /**
     *  @brief  Calls the @p function on each thread propagating a 2 indices
     *          to the function. The first index is the start of the range
     *          and the second index is the exclusive end of the range to be
     *          handled by a particular thread.
     */
    template <typename function_type_>
    inline void for_slices(size_t n, function_type_ &&function) const noexcept {
        function(0, n);
    }

    /**
     *  @brief  Calls the @p function for each index from 0 to @p (n) expecting
     *          that individual invocations can have drastically different duration,
     *          so each thread eagerly processes the next index in the range.
     */
    template <typename function_type_>
    inline void for_n_dynamic(size_t n, function_type_ &&function) const noexcept {
        for (size_t i = 0; i < n; ++i) function(i);
    }

    /**
     *  @brief  Executes a function in parallel on the current and all worker threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     */
    template <typename function_type_>
    void for_threads(function_type_ &&function) noexcept {
        function(0);
    }
};

template <typename executor_type_>
concept executor_like = requires(executor_type_ executor) {
#if !defined(__NVCC__) && 0
    { executor.threads_count() } -> std::same_as<size_t>;
    {
        executor.for_n(0u, [](size_t) {})
    };
    {
        executor.for_slices(0u, [](size_t, size_t) {})
    };
    {
        executor.for_n_dynamic(0u, [](size_t) {})
    };
    {
        executor.for_threads([](size_t) {})
    };
#else
    sizeof(executor) > 0;
#endif
};

struct openmp_executor_t {

    /**
     *  @brief  Calls the @p function for each index from 0 to @p (n) in such
     *          a way that consecutive elements are likely to be processed by
     *          the same thread.
     */
    template <typename function_type_>
    inline void for_n(size_t n, function_type_ &&function) const noexcept {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) function(i);
    }

    /**
     *  @brief  Calls the @p function on each thread propagating a 2 indices
     *          to the function. The first index is the start of the range
     *          and the second index is the exclusive end of the range to be
     *          handled by a particular thread.
     */
    template <typename function_type_>
    inline void for_slices(size_t n, function_type_ &&function) const noexcept {
        // OpenMP won't use more threads than the number of available cores
        // and by using STL to query that number, we avoid the need to link
        // against OpenMP libraries.
        size_t const total_threads = std::thread::hardware_concurrency();
        size_t const chunk_size = divide_round_up(n, total_threads);
#pragma omp parallel for schedule(static, 1)
        for (size_t i = 0; i < total_threads; ++i) {
            size_t const start = i * chunk_size;
            size_t const end = std::min(start + chunk_size, n);
            function(start, end);
        }
    }

    /**
     *  @brief  Calls the @p function for each index from 0 to @p (n) expecting
     *          that individual invocations can have drastically different duration,
     *          so each thread eagerly processes the next index in the range.
     */
    template <typename function_type_>
    inline void for_n_dynamic(size_t n, function_type_ &&function) const noexcept {
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < n; ++i) function(i);
    }

    /**
     *  @brief  Executes a function in parallel on the current and all worker threads.
     *  @param[in] function The callback, receiving the thread index as an argument.
     */
    template <typename function_type_>
    void for_threads(function_type_ const &function) noexcept {
        // ! Using the `omp_get_thread_num()` would force us to include the OpenMP headers
        // ! and link to the right symbols, which is not always possible.
        std::atomic<size_t> atomic_thread_index = 0;
#pragma omp parallel
        {
            size_t const thread_index = atomic_thread_index.fetch_add(1, std::memory_order_relaxed);
            function(thread_index);
        }
    }

    inline size_t threads_count() const noexcept {
        // ! Using the `omp_get_num_threads()` would force us to include the OpenMP headers
        // ! and link to the right symbols, which is not always possible.
        std::atomic<size_t> atomic_thread_index = 0;
#pragma omp parallel
        { atomic_thread_index.fetch_add(1, std::memory_order_relaxed); }
        return atomic_thread_index.load(std::memory_order_relaxed);
    }
};

#if !defined(__NVCC__)
static_assert(executor_like<dummy_executor_t>);
static_assert(executor_like<openmp_executor_t>);
// static_assert(!executor_like<int>);
#endif

template <typename continuous_type_>
concept continuous_like = requires(continuous_type_ container) {
    { container.data() } -> std::same_as<typename continuous_type_::value_type *>;
    { container.size() } -> std::convertible_to<size_t>;
};

static_assert(continuous_like<span<char>>);
static_assert(!continuous_like<int>);

/**
 *  @brief  A function that takes a range of elements and a @p callback function and groups the elements
 *          that @p equality function considers equal. Analogous to `std::ranges::group_by`.
 *  @return The number of groups formed.
 */
template <typename begin_iterator_type_, typename end_iterator_type_, typename equality_type_,
          typename slice_callback_type_>
size_t group_by(begin_iterator_type_ const begin, end_iterator_type_ const end, equality_type_ &&equality,
                slice_callback_type_ &&slice_callback) {

    auto const size = std::distance(begin, end);
    auto slice_start = begin;
    size_t group_count = 0;

    while (slice_start != end) {
        // Find the end of the current group by advancing `slice_end`
        auto slice_end = slice_start + 1;
        while (slice_end != end && equality(*slice_start, *slice_end)) ++slice_end;
        slice_callback(slice_start, slice_end);
        group_count++;
        // Move `slice_start` to the beginning of the next potential group
        slice_start = slice_end;
    }

    return group_count;
}

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_TYPES_HPP_
