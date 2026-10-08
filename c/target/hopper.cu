/**
 *  @file c/target/hopper.cu
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief The @c hopper kernels and their exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_CUDA
#define STRINGZILLA_TARGET_CUDA 0
#undef STRINGZILLA_TARGET_BLACKWELL
#define STRINGZILLA_TARGET_BLACKWELL 0
#include "stringzilla/substrings.h"

#include "stringzilla/substrings/hopper.cuh"
