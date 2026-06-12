/**
 *  @brief CUDA-accelerated string similarity utilities - GPU backends hub.
 *  @file include/stringzillas/similarities.cuh
 *  @author Ash Vardanian
 *
 *  Unlike the CPU backend, which also has single-pair similarity scores, the CUDA backend focuses @b only on
 *  batch-processing of large collections of strings, generally, assigning a single warp to each string pair:
 *
 *  - `sz::levenshtein_distances` & `sz::levenshtein_distances_utf8` for Levenshtein edit-distances.
 *  - `sz::needleman_wunsch_score` for weighted Needleman-Wunsch global alignment scores.
 *  - `sz::smith_waterman_score` for weighted Smith-Waterman local alignment scores.
 *
 *  This is a thin hub. It first pulls in the CPU hub (for the ISA-agnostic template core in
 *  `similarities/serial.hpp`), then the per-GPU-tier specialization headers:
 *
 *  - `similarities/cuda.cuh`   - base SIMT tier for any CUDA-capable GPU.
 *  - `similarities/kepler.cuh` - Kepler-generation packed 8/16-bit SIMD specializations.
 *  - `similarities/hopper.cuh` - Hopper-generation DPX min/max specializations.
 *
 *  @sa similarities.hpp
 *  @sa similarities/serial.hpp
 */
#ifndef STRINGZILLAS_SIMILARITIES_CUH_
#define STRINGZILLAS_SIMILARITIES_CUH_

#include "stringzillas/similarities.hpp" // ISA-agnostic template core (via similarities/serial.hpp) + CPU backends

#include "stringzillas/similarities/cuda.cuh"   // Base CUDA SIMT tier
#include "stringzillas/similarities/kepler.cuh" // Kepler specializations (#if SZ_USE_KEPLER)
#include "stringzillas/similarities/hopper.cuh" // Hopper specializations (#if SZ_USE_HOPPER)

// The per-tier `.cu` providers emit each engine's kernels once; here we only `extern` them so consumers link
// rather than recompile. Mirror the providers exactly (NW global, SW local, Kepler only for Levenshtein).
#if SZ_DYNAMIC_DISPATCH && SZ_USE_CUDA
namespace ashvardanian {
namespace stringzillas {

extern template struct levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
extern template struct levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
#if SZ_USE_KEPLER
extern template struct levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_caps_ck_k>;
extern template struct levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_caps_ck_k>;
#endif
#if SZ_USE_HOPPER
extern template struct levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
extern template struct levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
#endif

// Needleman-Wunsch (global) and Smith-Waterman (local) share the `cuda_weighted_scores` base, which owns the heavy
// out-of-class `run_trampoline_`; the `needleman_wunsch_scores`/`smith_waterman_scores` wrappers only inherit it, so
// externing the base is what keeps the kernels out of the consumer TUs.
extern template struct cuda_weighted_scores<linear_gap_costs_t, ualloc_t, sz_similarity_global_k, sz_cap_cuda_k>;
extern template struct cuda_weighted_scores<affine_gap_costs_t, ualloc_t, sz_similarity_global_k, sz_cap_cuda_k>;
extern template struct cuda_weighted_scores<linear_gap_costs_t, ualloc_t, sz_similarity_local_k, sz_cap_cuda_k>;
extern template struct cuda_weighted_scores<affine_gap_costs_t, ualloc_t, sz_similarity_local_k, sz_cap_cuda_k>;
#if SZ_USE_HOPPER
extern template struct cuda_weighted_scores<linear_gap_costs_t, ualloc_t, sz_similarity_global_k, sz_caps_ckh_k>;
extern template struct cuda_weighted_scores<affine_gap_costs_t, ualloc_t, sz_similarity_global_k, sz_caps_ckh_k>;
extern template struct cuda_weighted_scores<linear_gap_costs_t, ualloc_t, sz_similarity_local_k, sz_caps_ckh_k>;
extern template struct cuda_weighted_scores<affine_gap_costs_t, ualloc_t, sz_similarity_local_k, sz_caps_ckh_k>;
#endif

} // namespace stringzillas
} // namespace ashvardanian
#endif // SZ_DYNAMIC_DISPATCH && SZ_USE_CUDA

#endif // STRINGZILLAS_SIMILARITIES_CUH_
