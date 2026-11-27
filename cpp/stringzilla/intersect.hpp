/**
 *  @file cpp/stringzilla/intersect.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Positions of the identical strings in two deduplicated containers.
 */
#ifndef STRINGZILLA_INTERSECT_HPP_
#define STRINGZILLA_INTERSECT_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include <cstdint> // `std::uint64_t`

#include "stringzilla/intersect.h" // `sz_intersect_best`

#include "stringzilla/sort.hpp"

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief Locates the positions of identically valued elements in 2 deduplicated string arrays.
 *  @sa sz_intersect_best
 *
 *  @param[in] first_container The first array of string-like elements.
 *  @param[in] first_extractor Extracts the string from each object of the first array.
 *  @param[in] second_container The second array of string-like elements.
 *  @param[in] second_extractor Extracts the string from each object of the second array.
 *  @param[in] seed Randomizes the internal hash table to resist adversarial inputs.
 *  @param[out] first_positions The caller-owned output span of indices from the first array.
 *  @param[out] second_positions The caller-owned output span of indices from the second array.
 *      Each span must fit at least <tt>min(first.size(), second.size())</tt> entries.
 *  @param[in] allocator Serves the scratch memory of the hash table.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @return The number of matched pairs, valid in the first entries of each output span, with
 *      @c unexpected_dimensions_k if a span is too short, or the status of the kernel.
 */
template <typename first_container_, typename second_container_, typename first_extractor_, typename second_extractor_,
          typename allocator_type_ = std::allocator<char>>
expected<std::size_t> intersect(                                                          //
    first_container_ const &first_container, first_extractor_ const &first_extractor,     //
    second_container_ const &second_container, second_extractor_ const &second_extractor, //
    std::uint64_t seed,                                                                   //
    std::span<sz_size_t> first_positions, std::span<sz_size_t> second_positions,          //
    allocator_type_ allocator = {},                                                       //
    sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {

    std::size_t const max_count = sz_min_of_two(first_container.size(), second_container.size());
    if (first_positions.size() < max_count || second_positions.size() < max_count)
        return {0, status_t::unexpected_dimensions_k};

    // Pack the arguments into a single structure to reference it from the callback.
    using first_t = strings_args_<first_container_, first_extractor_>;
    using second_t = strings_args_<second_container_, second_extractor_>;
    first_t first_args {first_container, first_extractor};
    second_t second_args {second_container, second_extractor};
    sz_strings_view_t const first_strings = first_args.view(), second_strings = second_args.view();

    sz_size_t intersection_count = 0;
    using sz_alloc_type = sz_allocator_t;
    typename std::allocator_traits<allocator_type_>::template rebind_alloc<char> bytes_allocator(allocator);
    status_t status = with_allocator_(bytes_allocator, [&](sz_alloc_type &c_allocator) {
        return sz_intersect_best(&first_strings, &second_strings, &c_allocator, static_cast<sz_u64_t>(seed),
                                 &intersection_count, first_positions.data(), second_positions.data(), capabilities,
                                 stream);
    });
    return {intersection_count, status};
}

/** The positions of the identical elements in both arrays, as @c intersect finds them. */
template <typename allocator_type_ = std::allocator<char>>
struct intersect_result {
    sorted_indices<allocator_type_> first_offsets;
    sorted_indices<allocator_type_> second_offsets;
};

using intersect_result_t = intersect_result<>;

/**
 *  @brief Locates identical elements in two arrays.
 *  @param[in] allocator Serves the returned indices and the scratch memory of the hash table.
 *  @return Two arrays of indices, mapping the elements of the first and the second array that
 *      hold identical values, with @c bad_alloc_k if an allocation failed.
 */
template <typename first_type_, typename second_type_, a_string_extractor<first_type_> first_extractor_,
          a_string_extractor<second_type_> second_extractor_, typename allocator_type_ = std::allocator<char>>
expected<intersect_result<allocator_type_>> intersect(                     //
    first_type_ const &first, first_extractor_ const &first_extractor,     //
    second_type_ const &second, second_extractor_ const &second_extractor, //
    std::uint64_t seed = 0, allocator_type_ allocator = {},                //
    sz_capability_t capabilities = default_capabilities(), sz_stream_t stream = nullptr) noexcept {

    std::size_t const max_count = sz_min_of_two(first.size(), second.size());
    expected<intersect_result<allocator_type_>> result {
        {sorted_indices<allocator_type_>(allocator), sorted_indices<allocator_type_>(allocator)},
        status_t::success_k};
    intersect_result<allocator_type_> &offsets = result.value;
    result.status = resize_or_bad_alloc_(offsets.first_offsets, max_count);
    if (succeeded(result.status)) result.status = resize_or_bad_alloc_(offsets.second_offsets, max_count);
    if (failed(result.status)) return result;

    auto [matched_count, status] = intersect( //
        first, first_extractor, second, second_extractor, seed, {offsets.first_offsets.data(), max_count},
        {offsets.second_offsets.data(), max_count}, allocator, capabilities, stream);
    result.status = status;
    offsets.first_offsets.resize(matched_count);
    offsets.second_offsets.resize(matched_count);
    return result;
}

/** @overload */
template <typename first_type_, typename second_type_, typename allocator_type_ = std::allocator<char>>
expected<intersect_result<allocator_type_>> intersect(first_type_ const &first, second_type_ const &second,
                                                      std::uint64_t seed = 0, allocator_type_ allocator = {},
                                                      sz_capability_t capabilities = default_capabilities(),
                                                      sz_stream_t stream = nullptr) noexcept {
    using first_string_type = typename first_type_::value_type;
    using second_string_type = typename second_type_::value_type;
    static_assert( //
        std::is_convertible<first_string_type, string_view_t>::value, "The type must be convertible to string_view_t.");
    static_assert( //
        std::is_convertible<second_string_type, string_view_t>::value,
        "The type must be convertible to string_view_t.");
    return intersect(
        first, [](first_string_type const &s) -> string_view_t { return s; },   //
        second, [](second_string_type const &s) -> string_view_t { return s; }, //
        seed, allocator, capabilities, stream);
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_INTERSECT_HPP_
