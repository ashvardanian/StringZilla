/**
 *  @file c/target/metal.c
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The engines' @c metal kernels and the Metal runtime exports, defined once for the library.
 */
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"
#include "stringzilla/utf8_norm.h"
#include "stringzilla/utf8_uncased_fold.h"

#include "stringzilla/levenshtein/metal.h"
#include "stringzilla/overlap/metal.h"
#include "stringzilla/substrings/metal.h"
#include "stringzilla/utf8_norm/metal.h"
#include "stringzilla/utf8_uncased_fold/metal.h"

/*  `build.rs` compiles every `.c` under `c/`, so without Metal this unit defines nothing. */
#if STRINGZILLA_WITH_METAL

STRINGZILLA_API sz_metal_context_t *sz_metal_contexts_(os_unfair_lock_t *contexts_lock) {
    static sz_metal_context_t contexts[sz_metal_contexts_max_k];
    static os_unfair_lock lock;
    *contexts_lock = &lock;
    return contexts;
}

STRINGZILLA_API sz_status_t sz_device_count_metal(sz_size_t *count) {
    *count = sz_device_count_metal_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    return sz_capabilities_detected_metal_(ordinal, capabilities);
}

STRINGZILLA_API sz_status_t sz_stream_init_metal(sz_size_t ordinal, sz_stream_t *stream) {
    return sz_stream_init_metal_(ordinal, stream);
}

STRINGZILLA_API sz_status_t sz_stream_free_metal(sz_stream_t stream) { return sz_stream_free_metal_(stream); }

STRINGZILLA_API sz_status_t sz_allocator_init_unified_metal(sz_allocator_t *allocator) {
    return sz_allocator_init_unified_metal_(allocator);
}

STRINGZILLA_API sz_status_t sz_sequence_realloc_metal(sz_sequence_t *target, sz_sequence_t const *source,
                                                      sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                      sz_stream_t stream) {
    return sz_sequence_realloc_metal_(target, source, allocator, allocated_bytes, stream);
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_metal(sz_stream_t stream) {
    return sz_stream_synchronize_metal_(stream);
}

#endif // STRINGZILLA_WITH_METAL
