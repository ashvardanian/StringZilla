/**
 *  @file c/stringzillas/levenshtein_serial.cpp
 *  @brief Serial single-pair scorer instantiations for parallel Levenshtein distances.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include <fork_union.hpp>                // `ashvardanian::fork_union::basic_pool_t`
#include "stringzillas/similarities.hpp" // Engines + single-pair scorers

namespace ashvardanian {
namespace stringzillas {
namespace fu = ashvardanian::fork_union;

template status_t levenshtein_distance<char, linear_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const>, span<char const>, size_t &, scratch_space_t, dummy_executor_t &, cpu_specs_t const &) const;
template status_t levenshtein_distance<char, linear_gap_costs_t, sz_cap_serial_k>::operator()<fu::basic_pool_t>(
    span<char const>, span<char const>, size_t &, scratch_space_t, fu::basic_pool_t &, cpu_specs_t const &) const;
template status_t levenshtein_distance<char, affine_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const>, span<char const>, size_t &, scratch_space_t, dummy_executor_t &, cpu_specs_t const &) const;
template status_t levenshtein_distance<char, affine_gap_costs_t, sz_cap_serial_k>::operator()<fu::basic_pool_t>(
    span<char const>, span<char const>, size_t &, scratch_space_t, fu::basic_pool_t &, cpu_specs_t const &) const;
template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const>, span<char const>, size_t &, scratch_space_t, dummy_executor_t &, cpu_specs_t const &) const;
template status_t levenshtein_distance_utf8<linear_gap_costs_t, sz_cap_serial_k>::operator()<fu::basic_pool_t>(
    span<char const>, span<char const>, size_t &, scratch_space_t, fu::basic_pool_t &, cpu_specs_t const &) const;
template status_t levenshtein_distance_utf8<affine_gap_costs_t, sz_cap_serial_k>::operator()<dummy_executor_t>(
    span<char const>, span<char const>, size_t &, scratch_space_t, dummy_executor_t &, cpu_specs_t const &) const;
template status_t levenshtein_distance_utf8<affine_gap_costs_t, sz_cap_serial_k>::operator()<fu::basic_pool_t>(
    span<char const>, span<char const>, size_t &, scratch_space_t, fu::basic_pool_t &, cpu_specs_t const &) const;

} // namespace stringzillas
} // namespace ashvardanian
