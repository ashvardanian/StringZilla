/**
 *  @file src/capabilities/ampere.cu
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief The @c ampere kernels and their exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_CUDA
#define STRINGZILLA_TARGET_CUDA 0
#undef STRINGZILLA_TARGET_HOPPER
#define STRINGZILLA_TARGET_HOPPER 0
#undef STRINGZILLA_TARGET_BLACKWELL
#define STRINGZILLA_TARGET_BLACKWELL 0
#include "stringzilla/levenshtein.h"
#include "stringzilla/ngrams.h"
#include "stringzilla/substrings.h"
#include "stringzilla/regex.h"
#include "stringzilla/subdocuments.h"

#include "stringzilla/levenshtein/ampere.cuh"
#include "stringzilla/ngrams/ampere.cuh"
#include "stringzilla/substrings/ampere.cuh"
#include "stringzilla/regex/ampere.cuh"
#include "stringzilla/subdocuments/ampere.cuh"

/** The kernel @c sz_capabilities_compiled_cuda asks the runtime to load. */
extern "C" void const *sz_probe_ampere_(void) { return (void const *)&sz_ngrams_overlaps_ampere_kernel_; }
