/**
 *  @file src/capabilities/cdna4.hip
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief The @c cdna4 kernels and their exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_ROCM
#define STRINGZILLA_TARGET_ROCM 0
#undef STRINGZILLA_TARGET_CDNA3
#define STRINGZILLA_TARGET_CDNA3 0
#undef STRINGZILLA_TARGET_CDNA5
#define STRINGZILLA_TARGET_CDNA5 0
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"
#include "stringzilla/subdocuments.h"

#include "stringzilla/levenshtein/cdna4.cuh"
#include "stringzilla/overlap/cdna4.cuh"
#include "stringzilla/substrings/cdna4.cuh"
#include "stringzilla/subdocuments/cdna4.cuh"

/** The kernel @c sz_capabilities_compiled_rocm asks the runtime to load. */
extern "C" void const *sz_probe_cdna4_(void) { return (void const *)&sz_subdocuments_scan_cdna4_kernel_; }
