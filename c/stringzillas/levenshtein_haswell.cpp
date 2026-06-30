/**
 *  @file c/stringzillas/levenshtein_haswell.cpp
 *  @brief Haswell (AVX2) single-pair scorer instantiations for parallel Levenshtein distances.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include <fork_union.hpp>                // `ashvardanian::fork_union::basic_pool_t`
#include "stringzillas/similarities.hpp" // Engines + single-pair scorers

#if SZ_USE_HASWELL
namespace ashvardanian {
namespace stringzillas {
namespace fu = ashvardanian::fork_union;

template status_t                                                                           //
levenshtein_distance<char, linear_gap_costs_t, sz_caps_sh_k>::operator()<dummy_executor_t>( //
    span<char const>, span<char const>, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                           //
levenshtein_distance<char, linear_gap_costs_t, sz_caps_sh_k>::operator()<fu::basic_pool_t>( //
    span<char const>, span<char const>, size_t &, scratch_space_t, fu::basic_pool_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                           //
levenshtein_distance<char, affine_gap_costs_t, sz_caps_sh_k>::operator()<dummy_executor_t>( //
    span<char const>, span<char const>, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                           //
levenshtein_distance<char, affine_gap_costs_t, sz_caps_sh_k>::operator()<fu::basic_pool_t>( //
    span<char const>, span<char const>, size_t &, scratch_space_t, fu::basic_pool_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                          //
levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sh_k>::operator()<dummy_executor_t>( //
    span<char const>, span<char const>, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                          //
levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sh_k>::operator()<fu::basic_pool_t>( //
    span<char const>, span<char const>, size_t &, scratch_space_t, fu::basic_pool_t &,
    cpu_specs_t const &) const noexcept;

} // namespace stringzillas
} // namespace ashvardanian
#endif // SZ_USE_HASWELL
