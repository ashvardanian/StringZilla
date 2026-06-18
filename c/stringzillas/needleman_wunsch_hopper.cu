/**
 *  @file c/stringzillas/needleman_wunsch_hopper.cu
 *  @brief Hopper-tier instantiations for parallel Needleman-Wunsch global alignment scores.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include "stringzillas/similarities.cuh"

#if SZ_USE_HOPPER
namespace ashvardanian {
namespace stringzillas {
template struct needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
template struct needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
} // namespace stringzillas
} // namespace ashvardanian
#endif // SZ_USE_HOPPER
