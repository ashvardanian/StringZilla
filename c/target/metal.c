/**
 *  @file c/target/metal.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The engines' @c metal kernels and the Metal runtime exports, defined once for the library.
 */
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"

#include "stringzilla/levenshtein/metal.h"
#include "stringzilla/overlap/metal.h"
#include "stringzilla/substrings/metal.h"

/*  `build.rs` compiles every `.c` under `c/`, so without Metal this unit defines nothing. */
#if STRINGZILLA_WITH_METAL

STRINGZILLA_API sz_status_t sz_metal_count_devices(sz_size_t *count) {
    *count = sz_metal_count_devices_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_metal_capabilities_detected(sz_size_t ordinal, sz_capability_t *capabilities) {
    return sz_metal_capabilities_detected_(ordinal, capabilities);
}

STRINGZILLA_API sz_status_t sz_metal_stream_init(sz_size_t ordinal, void **stream) {
    return sz_metal_stream_init_(ordinal, stream);
}

STRINGZILLA_API sz_status_t sz_metal_stream_free(void *stream) { return sz_metal_stream_free_(stream); }

STRINGZILLA_API sz_status_t sz_memory_allocator_init_unified_metal(sz_memory_allocator_t *allocator) {
    return sz_memory_allocator_init_unified_metal_(allocator);
}

STRINGZILLA_API sz_status_t sz_sequence_copy_metal(sz_sequence_t *target, sz_sequence_t const *source,
                                                   sz_memory_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                   void *stream) {
    return sz_sequence_copy_metal_(target, source, allocator, allocated_bytes, stream);
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_metal(void *stream) { return sz_stream_synchronize_metal_(stream); }

#endif // STRINGZILLA_WITH_METAL
