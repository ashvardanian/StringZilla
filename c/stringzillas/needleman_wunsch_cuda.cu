/**
 *  @file c/stringzillas/needleman_wunsch_cuda.cu
 *  @brief Base-CUDA-tier instantiations for parallel Needleman-Wunsch global alignment scores.
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#include "stringzillas/similarities.cuh"

namespace ashvardanian {
namespace stringzillas {
template struct needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
template struct needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
} // namespace stringzillas
} // namespace ashvardanian
