/**
 *  @file src/capabilities/cdna3.hip
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief The @c cdna3 kernels and their exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_ROCM
#define STRINGZILLA_TARGET_ROCM 0
#undef STRINGZILLA_TARGET_CDNA4
#define STRINGZILLA_TARGET_CDNA4 0
#undef STRINGZILLA_TARGET_CDNA5
#define STRINGZILLA_TARGET_CDNA5 0
#include "stringzilla/levenshtein.h"
#include "stringzilla/ngrams.h"
#include "stringzilla/substrings.h"
#include "stringzilla/subdocuments.h"

#include "stringzilla/levenshtein/cdna3.cuh"
#include "stringzilla/ngrams/cdna3.cuh"
#include "stringzilla/substrings/cdna3.cuh"
#include "stringzilla/subdocuments/cdna3.cuh"

/** The kernel @c sz_capabilities_compiled_rocm asks the runtime to load. */
extern "C" void const *sz_probe_cdna3_(void) { return (void const *)&sz_ngrams_overlaps_cdna3_kernel_; }
