/**
 *  @file c/target/blackwell.cu
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief The @c blackwell kernels and their exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_CUDA
#define STRINGZILLA_TARGET_CUDA 0
#undef STRINGZILLA_TARGET_HOPPER
#define STRINGZILLA_TARGET_HOPPER 0
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"

#include "stringzilla/levenshtein/blackwell.cuh"
#include "stringzilla/overlap/blackwell.cuh"
