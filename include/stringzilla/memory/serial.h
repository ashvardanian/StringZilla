/**
 *  @file include/stringzilla/memory/serial.h
 *  @author Ash Vardanian
 *  @date January 1, 2024
 *  @brief Serial backend for hardware-accelerated memory operations.
 *
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_SERIAL_H_
#define STRINGZILLA_MEMORY_SERIAL_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

STRINGZILLA_INLINE void sz_lookup_serial_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                          char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);
    sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
    sz_u8_t const *source_u8 = (sz_u8_t const *)source;
    sz_u8_t *target_u8 = (sz_u8_t *)target;
    sz_u8_t const *source_end = source_u8 + length;
    for (; source_u8 != source_end; ++source_u8, ++target_u8) *target_u8 = lut_u8[*source_u8];
}

STRINGZILLA_INLINE void sz_fill_serial_(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    sz_ptr_t end = target + length;
    // Dealing with short strings, a single sequential pass would be faster.
    // If the size is larger than 2 words, then at least 1 of them will be aligned.
    // But just one aligned word may not be worth SWAR.
    if (length < STRINGZILLA_SWAR_THRESHOLD)
        while (target != end) *(target++) = value;

    // In case of long strings, skip unaligned bytes, and then fill the rest in 64-bit chunks.
    else {
        sz_u64_t value64 = (sz_u64_t)value * 0x0101010101010101ull;
        while ((sz_size_t)target & 7ull) *(target++) = value;
        while (target + 8 <= end) *(sz_u64_t *)target = value64, target += 8;
        while (target != end) *(target++) = value;
    }
}

STRINGZILLA_INLINE void sz_copy_serial_(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
#if STRINGZILLA_ALLOW_MISALIGNED_LOADS
    while (length >= 8) *(sz_u64_t *)target = *(sz_u64_t const *)source, target += 8, source += 8, length -= 8;
#endif
    while (length--) *(target++) = *(source++);
}

STRINGZILLA_INLINE void sz_move_serial_(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // Implementing `memmove` is trickier, than `memcpy`, as the ranges may overlap.
    // Existing implementations often have two passes, in normal and reversed order,
    // depending on the relation of `target` and `source` addresses.
    // https://student.cs.uwaterloo.ca/~cs350/common/os161-src-html/doxygen/html/memmove_8c_source.html
    // https://marmota.medium.com/c-language-making-memmove-def8792bb8d5
    //
    // We can use the `memcpy` like left-to-right pass if we know that the `target` is before `source`.
    // Or if we know that they don't intersect! In that case the traversal order is irrelevant,
    // but older CPUs may predict and fetch forward-passes better.
    if (target < source || target >= source + length) {
#if STRINGZILLA_ALLOW_MISALIGNED_LOADS
        while (length >= 8) *(sz_u64_t *)target = *(sz_u64_t const *)(source), target += 8, source += 8, length -= 8;
#endif
        while (length--) *(target++) = *(source++);
    }
    else {
        // Jump to the end and walk backwards.
        target += length, source += length;
#if STRINGZILLA_ALLOW_MISALIGNED_LOADS
        while (length >= 8) *(sz_u64_t *)(target -= 8) = *(sz_u64_t const *)(source -= 8), length -= 8;
#endif
        while (length--) *(--target) = *(--source);
    }
}

/** Writes @p source into a fresh tape from @p allocator and points @p target at it through the host
 *  tape accessors, which every group's copy starts from and a device one then replaces. */
STRINGZILLA_INLINE sz_status_t sz_sequence_realloc_serial_(sz_sequence_t *target, sz_sequence_t const *source,
                                                           sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                           sz_stream_t stream) {
    sz_size_t const count = source->count;
    if (count >= STRINGZILLA_SIZE_MAX / sizeof(sz_u64_t)) return sz_bad_alloc_k;
    sz_size_t bytes = (count + 1) * sizeof(sz_u64_t);
    for (sz_size_t index = 0; index != count; ++index) {
        sz_size_t const length = source->get_length(source->handle, index);
        if (length > STRINGZILLA_SIZE_MAX - bytes) return sz_bad_alloc_k;
        bytes += length;
    }
    sz_u64_t *const offsets = (sz_u64_t *)allocator->allocate(bytes, allocator->handle, stream);
    if (!offsets) return sz_bad_alloc_k;
    offsets[0] = (count + 1) * sizeof(sz_u64_t);
    for (sz_size_t index = 0; index != count; ++index) {
        sz_size_t const length = source->get_length(source->handle, index);
        sz_copy_serial_((sz_ptr_t)offsets + offsets[index], source->get_start(source->handle, index), length);
        offsets[index + 1] = offsets[index] + length;
    }
    target->handle = offsets, target->count = count;
    target->get_start = sz_sequence_tape_start, target->get_length = sz_sequence_tape_length;
    *allocated_bytes = bytes;
    return sz_success_k;
}

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_lookup_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                             char const lut[sz_at_least_(256)], sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_lookup_serial_(target, source, length, lut);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_fill_serial(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_fill_serial_(target, length, value);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_copy_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_copy_serial_(target, source, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_move_serial(sz_ptr_t target, sz_cptr_t source, sz_size_t length, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_move_serial_(target, source, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_serial(sz_allocator_t *allocator) {
    sz_allocator_init_default(allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sequence_realloc_serial(sz_sequence_t *target, sz_sequence_t const *source,
                                                       sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                       sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    if (source->get_start != sz_sequence_tape_start)
        return sz_sequence_realloc_serial_(target, source, allocator, allocated_bytes, stream);
    *target = *source;
    *allocated_bytes = 0;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_serial(sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SERIAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_SERIAL_H_
