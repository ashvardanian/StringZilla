/**
 *  @file c/stringzillas/levenshtein_neon.cpp
 *  @brief NEON (AArch64) single-pair scorer instantiations for parallel Levenshtein distances.
 *  @author Ash Vardanian
 */
#include "stringzillas/similarities.hpp" // Engines + single-pair scorers

#if SZ_USE_NEON
namespace ashvardanian {
namespace stringzillas {

template status_t                                                                           //
levenshtein_distance<char, linear_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>( //
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                           //
levenshtein_distance<char, linear_gap_costs_t, sz_caps_sn_k>::operator()<forkunion_executor_t>( //
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                           //
levenshtein_distance<char, affine_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>( //
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                           //
levenshtein_distance<char, affine_gap_costs_t, sz_caps_sn_k>::operator()<forkunion_executor_t>( //
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;
// UTF-8 NEON is linear-only, mirroring the Ice Lake provider (the affine rune variant is not yet specialized).
template status_t                                                                          //
levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sn_k>::operator()<dummy_executor_t>( //
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, dummy_executor_t &,
    cpu_specs_t const &) const noexcept;
template status_t                                                                          //
levenshtein_distance_utf8<linear_gap_costs_t, sz_caps_sn_k>::operator()<forkunion_executor_t>( //
    span<char const> const &, span<char const> const &, size_t &, scratch_space_t, forkunion_executor_t &,
    cpu_specs_t const &) const noexcept;

} // namespace stringzillas
} // namespace ashvardanian
#endif // SZ_USE_NEON
