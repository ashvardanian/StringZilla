/**
 *  @brief  Experimental kernels for StringZilla.
 *  @file   experimental.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_EXPERIMENTAL_H_
#define STRINGZILLA_EXPERIMENTAL_H_

#include "stringzilla.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 8-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_8bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                        sz_size_t n_length) {
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *h_end = h_unsigned + h_length;

    // Here is our baseline:
    //
    //      sz_u8_t running_match = 0xFF;
    //      sz_u8_t character_position_masks[256];
    //      for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFF; }
    //      for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    //      for (sz_size_t i = 0; i < h_length; ++i) {
    //          running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
    //          if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    //      }
    //
    // On very short patterns, however, every tiny condition may have a huge affect on performance.
    // 1. Let's replace byte-level intialization of `character_position_masks` with 64-bit ops.
    // 2. Let's combine the first `n_length - 1` passes of the last loop into the previous loop.
    typedef sz_u8_t offset_mask_t;

    // Initialize the possible offset masks.
    // Even using 8-byte `wide_masks` words, this would require 64 iterations to populate 256 bytes.
    union {
        offset_mask_t masks[256];
        sz_u64_t wide_masks[sizeof(offset_mask_t) * 256 / sizeof(sz_u64_t)];
    } character_positions;
    for (sz_size_t i = 0; i != sizeof(offset_mask_t) * 256 / sizeof(sz_u64_t); ++i) {
        character_positions.wide_masks[i] = 0xFFFFFFFFFFFFFFFFull;
    }

    // Populate the mask with possible positions for each character.
    for (sz_size_t i = 0; i != n_length; ++i) { character_positions.masks[n_unsigned[i]] &= ~((offset_mask_t)1 << i); }

    // The "running match" for the serial algorithm should be at least as wide as the `offset_mask_t`.
    // But on modern systems larger integers may work better.
    offset_mask_t running_match, final_match = 1;
    running_match = ~(running_match ^ running_match); //< Initialize with all-ones
    final_match <<= n_length - 1;

    for (; (h_unsigned != h_end) + ((running_match & final_match) != 0) == 2; ++h_unsigned) {
        running_match = (running_match << 1) | character_positions.masks[h_unsigned[0]];
    }
    return ((running_match & final_match) == 0) ? (sz_cptr_t)(h_unsigned - n_length) : NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 8-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_rfind_bitap_upto_8bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u8_t running_match = 0xFF;
    sz_u8_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 16-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_16bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u16_t running_match = 0xFFFF;
    sz_u16_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 16-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_rfind_bitap_upto_16bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                          sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u16_t running_match = 0xFFFF;
    sz_u16_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 32-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_32bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u32_t running_match = 0xFFFFFFFF;
    sz_u32_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 32-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_rfind_bitap_upto_32bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                          sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u32_t running_match = 0xFFFFFFFF;
    sz_u32_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFF; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1u << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns up to @b 64-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                         sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for exact matching of patterns up to @b 64-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_rfind_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                          sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for approximate matching of patterns up to @b 64-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bounded_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                                 sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[i]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[i]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + i - n_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algorithm for approximate matching of patterns up to @b 64-bytes long in @b reverse order.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
SZ_INTERNAL sz_cptr_t _sz_find_bounded_last_bitap_upto_64bytes_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n,
                                                                      sz_size_t n_length) {
    sz_u8_t const *h_unsigned = (sz_u8_t const *)h;
    sz_u8_t const *n_unsigned = (sz_u8_t const *)n;
    sz_u64_t running_match = 0xFFFFFFFFFFFFFFFFull;
    sz_u64_t character_position_masks[256];
    for (sz_size_t i = 0; i != 256; ++i) { character_position_masks[i] = 0xFFFFFFFFFFFFFFFFull; }
    for (sz_size_t i = 0; i < n_length; ++i) { character_position_masks[n_unsigned[n_length - i - 1]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < h_length; ++i) {
        running_match = (running_match << 1) | character_position_masks[h_unsigned[h_length - i - 1]];
        if ((running_match & (1ull << (n_length - 1))) == 0) { return h + h_length - i - 1; }
    }

    return NULL;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLA_EXPERIMENTAL_H_