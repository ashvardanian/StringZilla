/**
 *  @file include/stringzilla/small_string.h
 *  @author Ash Vardanian
 *  @date December 30, 2023
 *  @brief Small String Optimization implemented as a C 99 structure.
 *
 *  Includes core APIs:
 *
 *  - @c sz_string_init
 *  - @c sz_string_init_length
 *  - @c sz_string_free
 *
 *  Accessing the underlying string:
 *
 *  - @c sz_string_is_on_stack
 *  - @c sz_string_unpack
 *  - @c sz_string_range
 *  - @c sz_string_equal
 *  - @c sz_string_order
 *
 *  Modifying the string:
 *
 *  - @c sz_string_reserve
 *  - @c sz_string_expand
 *  - @c sz_string_erase
 *  - @c sz_string_shrink_to_fit
 */
#ifndef STRINGZILLA_SMALL_STRING_H_
#define STRINGZILLA_SMALL_STRING_H_

#include "stringzilla/types.h"

#include "stringzilla/find.h"   // `sz_equal`
#include "stringzilla/memory.h" // `sz_copy`, `sz_move`, `sz_fill`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core Structure

/** The number of bytes a stack-allocated string can hold, including the @c STRINGZILLA_NULL
 *  terminator. It can't be changed from outside, yet an `#error` is avoided, as it may
 *  already be set. */
#ifdef STRINGZILLA_STRING_INTERNAL_SPACE
#undef STRINGZILLA_STRING_INTERNAL_SPACE
#endif
#define STRINGZILLA_STRING_INTERNAL_SPACE (sizeof(sz_size_t) * 3 - 1) // 3 pointers minus one byte for an 8-bit length

/**
 *  @brief Tiny memory-owning string structure with a Small String Optimization (SSO).
 *
 *  Differs in layout from Folly, Clang, GCC, and probably most other implementations. It's designed
 *  to avoid any branches on read-only operations, and can store up to 22 characters on stack on
 *  64-bit machines, followed by the @c STRINGZILLA_NULL terminator.
 *
 *  @section small_string_changing_length Changing Length
 *
 *  One nice thing about this design, is that you can, in many cases, change the length of the
 *  string without any branches, invoking a `+=` or `-=` on the 64-bit @c length field. If the
 *  string is on heap, the solution is obvious. If it's on stack, inplace decrement wouldn't affect
 *  the top bytes of the string, only changing the last byte containing the length.
 */
typedef union sz_string_t {

#if !STRINGZILLA_ARCH_BIG_ENDIAN_

    struct external {
        sz_ptr_t start;
        sz_size_t length;
        sz_size_t space;
        sz_size_t padding;
    } external;

    struct internal {
        sz_ptr_t start;
        sz_u8_t length;
        char chars[STRINGZILLA_STRING_INTERNAL_SPACE];
    } internal;

#else

    struct external {
        sz_ptr_t start;
        sz_size_t space;
        sz_size_t padding;
        sz_size_t length;
    } external;

    struct internal {
        sz_ptr_t start;
        char chars[STRINGZILLA_STRING_INTERNAL_SPACE];
        sz_u8_t length;
    } internal;

#endif

    sz_size_t words[4];

} sz_string_t;

/*  Verify structure layout for branchless length extraction. On little-endian targets the 8-bit
 *  internal.length overlaps the LSB of external.length, at offset 8 on 64-bit and 4 on 32-bit. On
 *  big-endian 64-bit, internal.length is at offset 31 and external.length at 24; on big-endian
 *  32-bit, at 15 and 12. */
#if STRINGZILLA_WITH_LIBC // `offsetof` comes from `stddef.h`, which is part of the C standard library.
sz_static_assert_(offsetof(sz_string_t, internal.start) == offsetof(sz_string_t, external.start),
                  Alignment_confusion_between_internal_and_external_storage);
#if !STRINGZILLA_ARCH_BIG_ENDIAN_
#if STRINGZILLA_ARCH_64BIT_
sz_static_assert_(offsetof(sz_string_t, internal.length) == 8, Internal_length_offset_mismatch_on_little_endian_64);
sz_static_assert_(offsetof(sz_string_t, external.length) == 8, External_length_offset_mismatch_on_little_endian_64);
#else
sz_static_assert_(offsetof(sz_string_t, internal.length) == 4, Internal_length_offset_mismatch_on_little_endian_32);
sz_static_assert_(offsetof(sz_string_t, external.length) == 4, External_length_offset_mismatch_on_little_endian_32);
#endif
#else // STRINGZILLA_ARCH_BIG_ENDIAN_
#if STRINGZILLA_ARCH_64BIT_
sz_static_assert_(offsetof(sz_string_t, internal.length) == 31, Internal_length_offset_mismatch_on_big_endian_64);
sz_static_assert_(offsetof(sz_string_t, external.length) == 24, External_length_offset_mismatch_on_big_endian_64);
#else
sz_static_assert_(offsetof(sz_string_t, internal.length) == 15, Internal_length_offset_mismatch_on_big_endian_32);
sz_static_assert_(offsetof(sz_string_t, external.length) == 12, External_length_offset_mismatch_on_big_endian_32);
#endif
#endif
#endif

#pragma endregion Core Structure

#pragma region Core API

/** Initializes a string class instance to an empty value. */
STRINGZILLA_API_COMPTIME void sz_string_init(sz_string_t *string);

/** Checks if @p string is stored inside of the instance itself, rather than on the heap. */
STRINGZILLA_API_COMPTIME sz_bool_t sz_string_is_on_stack(sz_string_t const *string);

/**
 *  @brief Unpacks the opaque instance of a string class into its components, best used only in
 *      read-only operations.
 *
 *  @param[in] string String to unpack.
 *  @param[out] start Pointer to the start of the string.
 *  @param[out] length Number of bytes in the string, before the @c STRINGZILLA_NULL character.
 *  @param[out] space Number of bytes allocated on heap or stack, including the
 *      @c STRINGZILLA_NULL byte.
 *  @param[out] is_external Whether the string is on the heap, or fits within the @p string itself.
 */
STRINGZILLA_API_COMPTIME void sz_string_unpack( //
    sz_string_t const *string, sz_ptr_t *start, sz_size_t *length, sz_size_t *space, sz_bool_t *is_external);

/**
 *  @brief Unpacks only the start and length of the string, best used in read-only operations.
 *
 *  @param[in] string String to unpack.
 *  @param[out] start Pointer to the start of the string.
 *  @param[out] length Number of bytes in the string, before the @c STRINGZILLA_NULL character.
 */
STRINGZILLA_API_COMPTIME void sz_string_range(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length);

/** Returns the length of the string in a branchless manner. */
STRINGZILLA_API_COMPTIME sz_size_t sz_string_length(sz_string_t const *string);

/**
 *  @brief Constructs a string of a given @p length with noisy contents. Use the returned character
 *      pointer to populate the string.
 *
 *  @param[out] string String to initialize.
 *  @param[in] length Number of bytes in the string, before the @c STRINGZILLA_NULL character.
 *  @param[in] allocator Memory allocator to use for the allocation.
 *  @return @c STRINGZILLA_NULL if the operation failed, pointer to the start of
 *      the string otherwise.
 */
STRINGZILLA_API_COMPTIME sz_ptr_t sz_string_init_length(sz_string_t *string, sz_size_t length,
                                                        sz_memory_allocator_t *allocator);

/**
 *  @brief Grows the available memory capacity, without changing the contents or the length.
 *
 *  This is beneficial, if several insertions are expected, and we want to minimize allocations.
 *  Reserving less than the current capacity is a harmless no-op, like @c std::string::reserve.
 *
 *  @param[inout] string String to grow.
 *  @param[in] new_capacity The number of characters to reserve space for, including existing ones.
 *  @param[in] allocator Memory allocator to use for the allocation.
 *  @return @c STRINGZILLA_NULL if the operation failed, pointer to the new start of
 *      the string otherwise.
 */
STRINGZILLA_API_COMPTIME sz_ptr_t sz_string_reserve(sz_string_t *string, sz_size_t new_capacity,
                                                    sz_memory_allocator_t *allocator);

/**
 *  @brief Grows the string by adding an uninitialized region of @p added_length at the given
 *      @p offset.
 *
 *  Would often be used in conjunction with one or more @c sz_copy calls to populate the allocated
 *  region. Similar to @c sz_string_reserve, but changes the length of the @p string.
 *
 *  @param[inout] string String to grow.
 *  @param[in] offset Offset of the first byte to reserve space for, capped at the length.
 *  @param[in] added_length The number of new characters to reserve space for.
 *  @param[in] allocator Memory allocator to use for the allocation.
 *  @return @c STRINGZILLA_NULL if the operation failed, pointer to the new start of
 *      the string otherwise.
 */
STRINGZILLA_API_COMPTIME sz_ptr_t sz_string_expand( //
    sz_string_t *string, sz_size_t offset, sz_size_t added_length, sz_memory_allocator_t *allocator);

/**
 *  @brief Removes a range from a string. Changes the length, but not the capacity. Performs no
 *      allocations or deallocations and can't fail.
 *
 *  @param[inout] string String to clean.
 *  @param[in] offset Offset of the first byte to remove; at or past the length, nothing is removed.
 *  @param[in] length Number of bytes to remove. Out-of-bound ranges will be capped.
 *  @return Number of bytes removed, zero if @p offset ≥ the string length.
 */
STRINGZILLA_API_COMPTIME sz_size_t sz_string_erase(sz_string_t *string, sz_size_t offset, sz_size_t length);

/**
 *  @brief Shrinks the string to fit the current length, if it's allocated on the heap. It's the
 *      reverse operation of @ref sz_string_reserve.
 *
 *  @param[inout] string String to shrink.
 *  @param[in] allocator Memory allocator to use for the allocation.
 *  @return Whether the operation was successful. The only failures can come from the allocator, and
 *      leave the string unchanged.
 */
STRINGZILLA_API_COMPTIME sz_ptr_t sz_string_shrink_to_fit(sz_string_t *string, sz_memory_allocator_t *allocator);

/** Frees the string, if it's allocated on the heap, and resets its state either way. */
STRINGZILLA_API_COMPTIME void sz_string_free(sz_string_t *string, sz_memory_allocator_t *allocator);

#pragma endregion

#pragma region Serial Implementation

STRINGZILLA_API_COMPTIME sz_bool_t sz_string_is_on_stack(sz_string_t const *string) {
    // It doesn't matter if it's on stack or heap, the pointer location is the same.
    return (sz_bool_t)((sz_cptr_t)string->internal.start == (sz_cptr_t)&string->internal.chars[0]);
}

STRINGZILLA_API_COMPTIME void sz_string_range(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length) {
    sz_size_t is_small = (sz_cptr_t)string->internal.start == (sz_cptr_t)&string->internal.chars[0];
    sz_size_t is_big_mask = is_small - (sz_size_t)1;
    *start = string->external.start; // It doesn't matter if it's on stack or heap, the pointer location is the same.
    // If the string is small, use branch-less approach to mask-out the top bytes of the length.
    *length = string->external.length & ((sz_size_t)0xFF | is_big_mask);
}

STRINGZILLA_API_COMPTIME sz_size_t sz_string_length(sz_string_t const *string) {
    sz_size_t is_small = (sz_cptr_t)string->internal.start == (sz_cptr_t)&string->internal.chars[0];
    sz_size_t is_big_mask = is_small - (sz_size_t)1;
    return string->external.length & ((sz_size_t)0xFF | is_big_mask);
}

STRINGZILLA_API_COMPTIME void sz_string_unpack( //
    sz_string_t const *string, sz_ptr_t *start, sz_size_t *length, sz_size_t *space, sz_bool_t *is_external) {
    sz_size_t is_small = (sz_cptr_t)string->internal.start == (sz_cptr_t)&string->internal.chars[0];
    sz_size_t is_big_mask = is_small - (sz_size_t)1;
    *start = string->external.start; // It doesn't matter if it's on stack or heap, the pointer location is the same.
    // If the string is small, use branch-less approach to mask-out the top bytes of the length.
    *length = string->external.length & ((sz_size_t)0xFF | is_big_mask);
    // In case the string is small, the `is_small - (sz_size_t)1` will become all-ones mask.
    *space = sz_u64_blend(STRINGZILLA_STRING_INTERNAL_SPACE, string->external.space, is_big_mask);
    *is_external = (sz_bool_t)!is_small;
}

STRINGZILLA_API_COMPTIME sz_bool_t sz_string_equal(sz_string_t const *a, sz_string_t const *b) {
    // Fast path for self-comparison
    if (a == b) return sz_true_k;
    // Tempting to say that the external.length is bitwise the same even if it includes
    // some bytes of the on-stack payload, but we don't at this writing maintain that invariant.
    // (An on-stack string includes noise bytes in the high-order bits of external.length. So do this
    // the hard/correct way.

#if STRINGZILLA_ALLOW_MISALIGNED_LOADS
    // Dealing with StringZilla strings, we know that the `start` pointer always points
    // to a word at least 8 bytes long. Therefore, we can compare the first 8 bytes at once.

#endif
    // Alternatively, fall back to byte-by-byte comparison.
    sz_ptr_t a_start, b_start;
    sz_size_t a_length, b_length;
    sz_string_range(a, &a_start, &a_length);
    sz_string_range(b, &b_start, &b_length);
    return (sz_bool_t)(a_length == b_length && sz_equal(a_start, b_start, b_length));
}

STRINGZILLA_API_COMPTIME sz_ordering_t sz_string_order(sz_string_t const *a, sz_string_t const *b) {
#if STRINGZILLA_ALLOW_MISALIGNED_LOADS
    // Dealing with StringZilla strings, we know that the `start` pointer always points
    // to a word at least 8 bytes long. Therefore, we can compare the first 8 bytes at once.

#endif
    // Alternatively, fall back to byte-by-byte comparison.
    sz_ptr_t a_start, b_start;
    sz_size_t a_length, b_length;
    sz_string_range(a, &a_start, &a_length);
    sz_string_range(b, &b_start, &b_length);
    return sz_order(a_start, a_length, b_start, b_length);
}

STRINGZILLA_API_COMPTIME void sz_string_init(sz_string_t *string) {
    sz_assert_(string && "String can't be STRINGZILLA_NULL.");

    // Only 8 + 1 + 1 need to be initialized.
    string->internal.start = &string->internal.chars[0];
    // But for safety let's initialize the entire structure to zeros.
    // string->internal.chars[0] = 0;
    // string->internal.length = 0;
    string->words[1] = 0;
    string->words[2] = 0;
    string->words[3] = 0;
}

STRINGZILLA_API_COMPTIME sz_ptr_t sz_string_init_length(sz_string_t *string, sz_size_t length,
                                                        sz_memory_allocator_t *allocator) {
    sz_size_t space_needed = length + 1; // space for trailing \0
    sz_assert_(string && allocator && "String and allocator can't be STRINGZILLA_NULL.");
    // Initialize the string to zeros for safety.
    string->words[1] = 0;
    string->words[2] = 0;
    string->words[3] = 0;
    // If we are lucky, no memory allocations will be needed.
    if (space_needed <= STRINGZILLA_STRING_INTERNAL_SPACE) {
        string->internal.start = &string->internal.chars[0];
        string->internal.length = (sz_u8_t)length;
    }
    else {
        // If we are not lucky, we need to allocate memory.
        string->external.start = (sz_ptr_t)allocator->allocate(space_needed, allocator->handle);
        if (!string->external.start) return STRINGZILLA_NULL_CHAR;
        string->external.length = length;
        string->external.space = space_needed;
    }
    string->external.start[length] = 0;
    return string->external.start;
}

STRINGZILLA_API_COMPTIME sz_ptr_t sz_string_reserve(sz_string_t *string, sz_size_t new_capacity,
                                                    sz_memory_allocator_t *allocator) {

    sz_assert_(string && allocator && "Strings and allocators can't be STRINGZILLA_NULL.");

    sz_size_t new_space = new_capacity + 1;
    if (new_space <= STRINGZILLA_STRING_INTERNAL_SPACE) return string->external.start;

    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_external);
    sz_assert_(string_length < string_space && string_start[string_length] == 0 &&
               "A string keeps its terminator inside its space, which a byte-copied stack string does not");
    // Shrinking is a no-op, matching `std::string::reserve` semantics. Without this check a smaller
    // `new_capacity` would allocate a smaller buffer and then overflow it with the old contents.
    if (new_space <= string_space) return string->external.start;

    sz_ptr_t new_start = (sz_ptr_t)allocator->allocate(new_space, allocator->handle);
    if (!new_start) return STRINGZILLA_NULL_CHAR;

    sz_copy(new_start, string_start, string_length + 1); // ? The terminator moves with the contents
    string->external.start = new_start;
    string->external.space = new_space;
    string->external.padding = 0;
    string->external.length = string_length;

    // Deallocate the old string.
    if (string_is_external) allocator->free(string_start, string_space, allocator->handle);
    return string->external.start;
}

STRINGZILLA_API_COMPTIME sz_ptr_t sz_string_shrink_to_fit(sz_string_t *string, sz_memory_allocator_t *allocator) {

    sz_assert_(string && allocator && "Strings and allocators can't be STRINGZILLA_NULL.");

    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_external);

    // We may already be space-optimal, and in that case we don't need to do anything.
    sz_size_t new_space = string_length + 1;
    if (string_space == new_space || !string_is_external) return string->external.start;

    sz_ptr_t new_start = (sz_ptr_t)allocator->allocate(new_space, allocator->handle);
    if (!new_start) return STRINGZILLA_NULL_CHAR;

    sz_copy(new_start, string_start, string_length + 1); // ? The terminator moves with the contents
    string->external.start = new_start;
    string->external.space = new_space;
    string->external.padding = 0;
    string->external.length = string_length;

    // Deallocate the old string.
    if (string_is_external) allocator->free(string_start, string_space, allocator->handle);
    return string->external.start;
}

STRINGZILLA_API_COMPTIME sz_ptr_t sz_string_expand( //
    sz_string_t *string, sz_size_t offset, sz_size_t added_length, sz_memory_allocator_t *allocator) {

    sz_assert_(string && allocator && "String and allocator can't be STRINGZILLA_NULL.");

    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_external);
    sz_assert_(string_length < string_space && string_start[string_length] == 0 &&
               "A string keeps its terminator inside its space, which a byte-copied stack string does not");

    // The user intended to extend the string.
    offset = sz_min_of_two(offset, string_length);

    // Guard against integer overflow in size calculation.
    if (added_length > STRINGZILLA_SSIZE_MAX - string_length - 1) return STRINGZILLA_NULL_CHAR;

    // If we are lucky, no memory allocations will be needed.
    if (string_length + added_length < string_space) {
        sz_move(string_start + offset + added_length, string_start + offset, string_length - offset);
        string_start[string_length + added_length] = 0;
        // Even if the string is on the stack, the `+=` won't affect the tail of the string.
        string->external.length += added_length;
    }
    // If we are not lucky, we need to allocate more memory.
    else {
        // A growing string's first heap buffer is one cache line, so a few appends do not walk the allocator.
        sz_size_t next_planned_size = sz_max_of_two(STRINGZILLA_CACHE_LINE_BYTES, string_space * (sz_size_t)2);
        sz_size_t min_needed_space = sz_size_bit_ceil(offset + string_length + added_length + 1);
        sz_size_t new_space = sz_max_of_two(min_needed_space, next_planned_size);
        string_start = sz_string_reserve(string, new_space - 1, allocator);
        if (!string_start) return STRINGZILLA_NULL_CHAR;

        // Copy into the new buffer.
        sz_move(string_start + offset + added_length, string_start + offset, string_length - offset);
        string_start[string_length + added_length] = 0;
        string->external.length = string_length + added_length;
    }

    return string_start;
}

STRINGZILLA_API_COMPTIME sz_size_t sz_string_erase(sz_string_t *string, sz_size_t offset, sz_size_t length) {

    sz_assert_(string && "String can't be STRINGZILLA_NULL.");

    sz_ptr_t string_start;
    sz_size_t string_length;
    sz_size_t string_space;
    sz_bool_t string_is_external;
    sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_external);
    sz_assert_(string_length < string_space && string_start[string_length] == 0 &&
               "A string keeps its terminator inside its space, which a byte-copied stack string does not");

    // Normalize the offset, it can't be larger than the length.
    offset = sz_min_of_two(offset, string_length);

    // We shouldn't normalize the length, to avoid overflowing on
    // `offset + length >= string_length`, if receiving `length == STRINGZILLA_SIZE_MAX`. After
    // following expression the `length` will contain exactly the delta between original and final
    // length of this `string`.
    length = sz_min_of_two(length, string_length - offset);

    // There are 2 common cases, that wouldn't even require a `memmove`:
    //      1.  Erasing the entire contents of the string.
    //          In that case `length` argument will be equal or greater than `length` member.
    //      2.  Removing the tail of the string with something like `string.pop_back()` in C++.
    //
    // In both of those, regardless of the location of the string - stack or heap,
    // the erasing is as easy as setting the length to the offset.
    // In every other case, we must `memmove` the tail of the string to the left.
    if (offset + length < string_length)
        sz_move(string_start + offset, string_start + offset + length, string_length - offset - length);

    // The `string->external.length = offset` assignment would discard last characters
    // of the on-the-stack string, but inplace subtraction would work.
    string->external.length -= length;
    string_start[string_length - length] = 0;
    return length;
}

STRINGZILLA_API_COMPTIME void sz_string_free(sz_string_t *string, sz_memory_allocator_t *allocator) {
    if (!sz_string_is_on_stack(string))
        allocator->free(string->external.start, string->external.space, allocator->handle);
    sz_string_init(string);
}

#pragma endregion Serial Implementation

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SMALL_STRING_H_
