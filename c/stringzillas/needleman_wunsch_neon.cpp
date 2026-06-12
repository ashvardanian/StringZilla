/**
 *  @file c/stringzillas/needleman_wunsch_neon.cpp
 *  @brief NEON (AArch64) single-pair scorer instantiations for parallel Needleman-Wunsch global alignment.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include <fork_union.hpp>                // `ashvardanian::fork_union::basic_pool_t`
#include "stringzillas/similarities.hpp" // Engines + single-pair scorers

#if SZ_USE_NEON
namespace ashvardanian {
namespace stringzillas {
namespace fu = ashvardanian::fork_union;

template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>(
    span<char const>, span<char const>, ssize_t &, scratch_space_t, dummy_executor_t &, cpu_specs_t const &) const;
template status_t
needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sn_k>::operator()<fu::basic_pool_t>(
    span<char const>, span<char const>, ssize_t &, scratch_space_t, fu::basic_pool_t &, cpu_specs_t const &) const;
template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>(
    span<char const>, span<char const>, ssize_t &, scratch_space_t, dummy_executor_t &, cpu_specs_t const &) const;
template status_t
needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sn_k>::operator()<fu::basic_pool_t>(
    span<char const>, span<char const>, ssize_t &, scratch_space_t, fu::basic_pool_t &, cpu_specs_t const &) const;

} // namespace stringzillas
} // namespace ashvardanian
#endif // SZ_USE_NEON
