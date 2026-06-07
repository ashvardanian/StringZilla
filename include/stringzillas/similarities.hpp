/**
 *  @brief Parallel string similarity scores in C++ - CPU backends hub.
 *  @file include/stringzillas/similarities.hpp
 *  @author Ash Vardanian
 *
 *  Includes core APIs, defined as the following template objects:
 *
 *  - `sz::levenshtein_distance` & `sz::levenshtein_distance_utf8` for Levenshtein edit-scores.
 *  - `sz::needleman_wunsch_score` for weighted Needleman-Wunsch @b (NW) global alignment.
 *  - `sz::smith_waterman_score` for weighted Smith-Waterman @b (SW) local alignment.
 *
 *  Also includes their batch-capable and parallel versions:
 *
 *  - `sz::levenshtein_distances` & `sz::levenshtein_distances_utf8` for Levenshtein edit-scores.
 *  - `sz::needleman_wunsch_scores` for weighted Needleman-Wunsch global alignment.
 *  - `sz::smith_waterman_scores` for weighted Smith-Waterman local alignment.
 *
 *  This is a thin hub aggregating the per-backend headers. The ISA-agnostic template core and
 *  the serial backend aliases live in `similarities/serial.hpp`; CPU SIMD specializations live
 *  in their own per-ISA files. GPU backends are aggregated by `similarities.cuh`.
 *
 *  @sa similarities/serial.hpp
 *  @sa similarities/icelake.hpp
 */
#ifndef STRINGZILLAS_SIMILARITIES_HPP_
#define STRINGZILLAS_SIMILARITIES_HPP_

#include "stringzillas/similarities/serial.hpp"  // ISA-agnostic template core + serial aliases
#include "stringzillas/similarities/icelake.hpp" // AVX-512 (Ice Lake) specializations

#endif // STRINGZILLAS_SIMILARITIES_HPP_
