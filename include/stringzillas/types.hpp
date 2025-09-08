/**
 *  @brief  Shared definitions for the StringZillas C++ library.
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

enum bytes_per_cell_t : unsigned {
    zero_bytes_per_cell_k = 0,
    one_byte_per_cell_k = 1,
    two_bytes_per_cell_k = 2,
    four_bytes_per_cell_k = 4,
    eight_bytes_per_cell_k = 8,
};

struct dummy_mutex_t {
    constexpr void lock() noexcept {}
    constexpr void unlock() noexcept {}
};

/**
 *  @brief  Simple RAII lock guard analog to `std::lock_guard` for C++11 compatibility.
 *          Automatically locks the mutex on construction and unlocks on destruction.
 */
template <typename mutex_type_>
class lock_guard {
    mutex_type_ &mutex_;

  public:
    explicit lock_guard(mutex_type_ &mutex) noexcept : mutex_(mutex) { mutex_.lock(); }
    ~lock_guard() noexcept { mutex_.unlock(); }

    lock_guard(lock_guard &&) = delete;
    lock_guard(lock_guard const &) = delete;
    lock_guard &operator=(lock_guard &&) = delete;
    lock_guard &operator=(lock_guard const &) = delete;
};

struct dummy_prong_t {
    std::size_t task = 0;
    std::size_t thread = 0;

    operator std::size_t() const noexcept { return task; }
};

/**
 *  @brief  C++17-compatible equivalent of std::remove_cvref (which was added in C++20).
 *          Removes const, volatile, and reference qualifiers from a type.
 */
template <typename type_>
using remove_cvref = typename std::remove_cv<typename std::remove_reference<type_>::type>::type;

struct dummy_executor_t {
    using prong_t = dummy_prong_t;

    constexpr size_t threads_count() const noexcept { return 1; }
    constexpr dummy_mutex_t make_mutex() const noexcept { return {}; }

    /**
     *  @brief  Calls the @p function for each index from 0 to @p (n) in such
     *          a way that consecutive elements are likely to be processed by
     *          the same thread.
     */
    template <typename function_type_>
    inline void for_n(size_t n, function_type_ &&function) const noexcept {
        for (size_t i = 0; i < n; ++i) function(dummy_prong_t {i, 0});
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
        for (size_t i = 0; i < n; ++i) function(dummy_prong_t {i, 0});
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

#if SZ_HAS_CONCEPTS_

template <typename executor_type_>
concept executor_like = requires(executor_type_ executor) {
    { executor.threads_count() } -> std::convertible_to<size_t>;
    typename executor_type_::prong_t;
    executor.for_n(0u, [](typename executor_type_::prong_t) {});
    executor.for_slices(0u, [](typename executor_type_::prong_t, size_t) {});
    executor.for_n_dynamic(0u, [](typename executor_type_::prong_t) {});
    executor.for_threads([](size_t) {});
};

template <typename results_type_>
concept indexed_results_like = requires(results_type_ results, size_t i) {
    { results[i] };
};

#endif

/** @brief Type trait to extract the value type from indexed results. */
template <typename results_type_>
struct indexed_results_type {
    using clean_type = typename std::remove_reference<results_type_>::type;
    using type = typename clean_type::value_type;
};

template <typename value_type_>
struct indexed_results_type<value_type_ *> {
    using type = value_type_;
};

template <typename value_type_>
struct indexed_results_type<value_type_ *&> {
    using type = value_type_;
};

/**
 *  @brief An example of an executor that uses OpenMP for parallel execution.
 *  @note Fork Union is preferred over this for library builds, but this is useful for users already leveraging OpenMP.
 */
struct openmp_executor_t {
    using prong_t = std::size_t;

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

#if SZ_HAS_CONCEPTS_
static_assert(executor_like<dummy_executor_t>);
static_assert(executor_like<openmp_executor_t>);
static_assert(!executor_like<int>);

template <typename continuous_type_>
concept continuous_like = requires(continuous_type_ container) {
    { container.data() } -> std::same_as<typename continuous_type_::value_type *>;
    { container.size() } -> std::convertible_to<size_t>;
};

static_assert(continuous_like<span<char>>);
static_assert(!continuous_like<int>);
#endif

/**
 *  @brief  A function that takes a range of elements and a @p callback function and groups the elements
 *          that @p equality function considers equal. Analogous to `std::ranges::group_by`.
 *  @return The number of groups formed.
 */
template <typename begin_iterator_type_, typename end_iterator_type_, typename equality_type_,
          typename slice_callback_type_>
size_t group_by(begin_iterator_type_ const begin, end_iterator_type_ const end, equality_type_ &&equality,
                slice_callback_type_ &&slice_callback) {

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

/**
 *  @brief  Safer alternative to `std::vector`, that avoids exceptions, copy constructors,
 *          and provides alternative `try_push_back` and `try_reserve` for faulty memory allocations.
 */
template <typename value_type_, typename allocator_type_>
class safe_vector {
  public:
    using value_type = value_type_;
    using size_type = std::size_t;
    using allocator_type = allocator_type_;

    using allocator_traits = std::allocator_traits<allocator_type>;
    using allocated_type = typename allocator_traits::value_type;
    static_assert(sizeof(value_type) == sizeof(allocated_type),
                  "Allocator value type must be the same size as the vector value type");
    static_assert(allocator_traits::propagate_on_container_move_assignment::value,
                  "Allocator must propagate on move assignment, otherwise the move assignment won't be `noexcept`.");

  private:
    value_type *data_;
    size_type size_;
    size_type capacity_;
    allocator_type alloc_;

  public:
    safe_vector() noexcept : data_(nullptr), size_(0), capacity_(0), alloc_() {}
    safe_vector(allocator_type alloc) noexcept : data_(nullptr), size_(0), capacity_(0), alloc_(alloc) {}
    ~safe_vector() noexcept { reset(); }

    void clear() noexcept {
        if constexpr (!std::is_trivially_destructible<value_type>::value)
            for (size_type i = 0; i < size_; ++i) data_[i].~value_type();
        size_ = 0;
    }

    void reset() noexcept {
        clear();
        if (data_) alloc_.deallocate((allocated_type *)data_, capacity_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    /** @warning Use `try_assign` instead to handle out-of-memory failures. */
    safe_vector(safe_vector const &other) = delete;
    /** @warning Use `try_assign` instead to handle out-of-memory failures. */
    safe_vector &operator=(safe_vector const &other) = delete;

    safe_vector(safe_vector &&other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_), alloc_(std::move(other.alloc_)) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    safe_vector &operator=(safe_vector &&other) noexcept {
        if (this != &other) {
            clear();
            if (data_) alloc_.deallocate((allocated_type *)data_, capacity_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            alloc_ = std::move(other.alloc_);
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    status_t try_assign(span<value_type const> const other) noexcept {
        reset();

        if (other.size() == 0) return status_t::success_k; // Nothing to do :)

        // Allocate exact needed capacity
        size_type new_cap = other.size();
        allocated_type *raw = allocator_traits::allocate(alloc_, new_cap);
        if (!raw) return status_t::bad_alloc_k;
        data_ = reinterpret_cast<value_type *>(raw);
        capacity_ = new_cap;

        // Copy‐construct each element
        if constexpr (!std::is_trivially_constructible<value_type>::value)
            for (size_type i = 0; i < other.size(); ++i) new (data_ + i) value_type(other[i]);
        else
            for (size_type i = 0; i < other.size(); ++i) data_[i] = other[i];
        size_ = other.size();
        return status_t::success_k;
    }

    template <typename other_allocator_type_ = allocator_type>
    status_t try_assign(safe_vector<value_type, other_allocator_type_> const &other) noexcept {
        if constexpr (allocator_traits::propagate_on_container_copy_assignment::value) alloc_ = other.alloc_;
        return try_assign(span<value_type>(other.data(), other.size()));
    }

    status_t try_reserve(size_type new_cap) noexcept {
        if (new_cap <= capacity_) return status_t::success_k;
        value_type *new_data = (value_type *)alloc_.allocate(new_cap);
        if (!new_data) return status_t::bad_alloc_k;
        for (size_type i = 0; i < size_; ++i) {
            new (new_data + i) value_type(std::move(data_[i]));
            if constexpr (!std::is_trivially_destructible<value_type>::value) data_[i].~value_type();
        }
        if (data_) alloc_.deallocate((allocated_type *)data_, capacity_);
        data_ = new_data;
        capacity_ = new_cap;
        return status_t::success_k;
    }

    status_t try_resize(size_type new_size) noexcept {
        if (new_size > capacity_ && try_reserve(new_size) != status_t::success_k) return status_t::bad_alloc_k;

        if (new_size > size_) {
            if constexpr (!std::is_trivially_constructible<value_type>::value)
                for (size_type i = size_; i < new_size; ++i) new (data_ + i) value_type();
        }
        else if (new_size < size_) {
            if constexpr (!std::is_trivially_destructible<value_type>::value)
                for (size_type i = new_size; i < size_; ++i) data_[i].~value_type();
        }

        size_ = new_size;
        return status_t::success_k;
    }

    status_t try_push_back(value_type const &val) noexcept {
        if (size_ == capacity_) {
            size_type new_cap = capacity_ ? capacity_ * 2 : 1;
            if (try_reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        new (data_ + size_) value_type(val);
        ++size_;
        return status_t::success_k;
    }

    status_t try_push_back(value_type &&val) noexcept {
        if (size_ == capacity_) {
            size_type new_cap = capacity_ ? capacity_ * 2 : 1;
            if (try_reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        new (data_ + size_) value_type(std::move(val));
        ++size_;
        return status_t::success_k;
    }

    status_t try_append(span<value_type const> source) noexcept {
        size_type needed = size_ + source.size();
        if (needed > capacity_) {
            size_type new_cap = capacity_ ? capacity_ : 1;
            while (new_cap < needed) new_cap *= 2;
            if (try_reserve(new_cap) != status_t::success_k) return status_t::bad_alloc_k;
        }
        for (size_type i = 0; i < source.size(); ++i) new (data_ + size_ + i) value_type(source[i]);
        size_ = needed;
        return status_t::success_k;
    }

    value_type *begin() noexcept { return data_; }
    value_type const *begin() const noexcept { return data_; }
    value_type *end() noexcept { return data_ + size_; }
    value_type const *end() const noexcept { return data_ + size_; }
    value_type &operator[](size_type i) noexcept { return data_[i]; }
    value_type const &operator[](size_type i) const noexcept { return data_[i]; }
    value_type *data() noexcept { return data_; }
    value_type const *data() const noexcept { return data_; }
    value_type &front() noexcept { return data_[0]; }
    value_type const &front() const noexcept { return data_[0]; }
    value_type &back() noexcept { return data_[size_ - 1]; }
    value_type const &back() const noexcept { return data_[size_ - 1]; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    operator span<value_type>() noexcept { return {data_, size_}; }
    operator span<value_type const>() const noexcept { return {data_, size_}; }
};

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_TYPES_HPP_
