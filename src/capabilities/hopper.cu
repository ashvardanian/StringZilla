/**
 *  @file src/capabilities/hopper.cu
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief The @c hopper kernels and their exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_CUDA
#define STRINGZILLA_TARGET_CUDA 0
#undef STRINGZILLA_TARGET_AMPERE
#define STRINGZILLA_TARGET_AMPERE 0
#undef STRINGZILLA_TARGET_BLACKWELL
#define STRINGZILLA_TARGET_BLACKWELL 0
#include "stringzilla/substrings.h"
#include "stringzilla/regex.h"
#include "stringzilla/subdocuments.h"

#include "stringzilla/substrings/hopper.cuh"
#include "stringzilla/regex/hopper.cuh"
#include "stringzilla/subdocuments/hopper.cuh"

/** The kernel @c sz_capabilities_compiled_cuda asks the runtime to load. */
extern "C" void const *sz_probe_hopper_(void) { return (void const *)&sz_substrings_scan_hopper_kernel_; }
