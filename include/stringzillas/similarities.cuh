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

#endif // STRINGZILLAS_SIMILARITIES_CUH_
