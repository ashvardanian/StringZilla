/**
 *  @file c/stringzillas/smith_waterman_icelake.cpp
 *  @brief Ice Lake (AVX-512) single-pair scorer instantiations for parallel Smith-Waterman local alignment.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include <fork_union.hpp>                // `ashvardanian::fork_union::basic_pool_t`
#include "stringzillas/similarities.hpp" // Engines + single-pair scorers

#if SZ_USE_ICELAKE
namespace ashvardanian {
namespace stringzillas {
namespace fu = ashvardanian::fork_union;

template status_t //
smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t,
                     sz_caps_sil_k>::operator()<dummy_executor_t>( //
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t //
smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t,
                     sz_caps_sil_k>::operator()<fu::basic_pool_t>( //
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, fu::basic_pool_t &,
    cpu_specs_t const &) const noexcept;
template status_t //
smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t,
                     sz_caps_sil_k>::operator()<dummy_executor_t>( //
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t //
smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t,
                     sz_caps_sil_k>::operator()<fu::basic_pool_t>( //
    span<char const> const &, span<char const> const &, ssize_t &, scratch_space_t, fu::basic_pool_t &,
    cpu_specs_t const &) const noexcept;

} // namespace stringzillas
} // namespace ashvardanian
#endif // SZ_USE_ICELAKE
