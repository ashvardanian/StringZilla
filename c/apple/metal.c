/**
 *  @file c/apple/metal.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The engines' @c metal kernels and the Metal device exports, defined once for the library.
 */
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"

#include "stringzilla/levenshtein/simt.h"
#include "stringzilla/overlap/simt.h"
#include "stringzilla/substrings/simt.h"

/*  `build.rs` compiles every `.c` under `c/`, so without Metal this unit defines nothing. */
#if STRINGZILLA_WITH_METAL

STRINGZILLA_API sz_status_t sz_metal_count_devices(sz_size_t *count) {
    *count = sz_metal_count_devices_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_metal_capabilities_detected(sz_size_t device, sz_capability_t *capabilities) {
    return sz_metal_capabilities_detected_(device, capabilities);
}

STRINGZILLA_API sz_status_t sz_metal_device_init(sz_size_t ordinal, sz_size_t arena_bytes, sz_metal_device_t *device) {
    return sz_metal_device_init_(ordinal, arena_bytes, device);
}

STRINGZILLA_API sz_status_t sz_metal_device_synchronize(sz_metal_device_t *device) {
    return sz_metal_device_synchronize_(device);
}

STRINGZILLA_API void sz_metal_device_free(sz_metal_device_t *device) { sz_metal_device_free_(device); }

#endif // STRINGZILLA_WITH_METAL
