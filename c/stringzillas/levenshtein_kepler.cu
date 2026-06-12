/**
 *  @file c/stringzillas/levenshtein_kepler.cu
 *  @brief Kepler-tier instantiations for parallel Levenshtein distances.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include "stringzillas/similarities.cuh"

#if SZ_USE_KEPLER
namespace ashvardanian {
namespace stringzillas {
template struct levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_caps_ck_k>;
template struct levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_caps_ck_k>;
} // namespace stringzillas
} // namespace ashvardanian
#endif // SZ_USE_KEPLER
