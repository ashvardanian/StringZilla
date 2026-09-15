/**
 *  @brief  Multi-pattern exact and case-folded substring search for CUDA GPUs.
 *  @file   include/stringzillas/substrings.cuh
 *  @author Ash Vardanian
 *  @sa     include/stringzillas/substrings.hpp for the CPU backends.
 *
 *  This is a thin hub aggregating the GPU backend on top of the CPU one, since the dictionary is built on the
 *  host and only its published view is uploaded.
 *
 *  A thread walks one chunk of the tape and the hot tier is staged into shared memory per block.
 *  `substrings/cuda.cuh` documents the chunking and staging rules.
 */
#ifndef STRINGZILLAS_SUBSTRINGS_CUH_
#define STRINGZILLAS_SUBSTRINGS_CUH_

#include "stringzillas/substrings.hpp" // Host-side dictionary, built before any upload

#include "stringzillas/substrings/cuda.cuh" // GPU engine over the uploaded view

#endif // STRINGZILLAS_SUBSTRINGS_CUH_
