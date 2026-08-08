/**
 *  @brief  Multi-pattern exact and case-folded substring search for CUDA GPUs.
 *  @file   include/stringzillas/find_many.cuh
 *  @author Ash Vardanian
 *  @sa     include/stringzillas/find_many.hpp for the CPU backends.
 *
 *  This is a thin hub aggregating the GPU backend on top of the CPU one, since the dictionary is built on the
 *  host and only its published view is uploaded.
 *
 *  A thread walks one chunk of the tape and the hot tier is staged into shared memory per block.
 *  `find_many/cuda.cuh` documents the chunking and staging rules.
 */
#ifndef STRINGZILLAS_FIND_MANY_CUH_
#define STRINGZILLAS_FIND_MANY_CUH_

#include "stringzillas/find_many.hpp" // Host-side dictionary, built before any upload

#include "stringzillas/find_many/cuda.cuh" // GPU engine over the uploaded view

#endif // STRINGZILLAS_FIND_MANY_CUH_
