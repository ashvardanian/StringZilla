/**
 *  @file include/stringzilla/utf8_uncased_fold/simt.cuh
 *  @author Ash Vardanian
 *  @date October 2, 2026
 *  @brief The kernel CUDA and ROCm share for UTF-8 case folding: a text cut into tiles, every
 *      thread folding one stretch of a tile twice, once to measure it and once to write it where
 *      the tiles before it end.
 *
 *  The fold is the serial tier's, held as tables rather than as its ladder of comparisons: a
 *  warp's lanes fold different letters, and each switch of the ladder becomes an indirect branch
 *  they take one target at a time, while the tables cost every lane the same two dependent loads.
 *  The tables are generated from @c sz_unicode_fold_codepoint_, and the device tests fold every
 *  codepoint against it.
 *
 *  A tile learns where its output begins from the tiles before it, which publish their folded sizes
 *  in turn through the caller's own length slot, as the @b Chained @b Tiles of `types.cuh` do, so a
 *  round needs no scratch beyond the slot it reports into.
 *
 *  Written in C, as every `.cuh` in this library is. Only device code lives here; each vendor
 *  launches the kernel from its own host side, in `cuda.cuh` and `rocm.cuh` beside this file.
 *
 *  @sa include/stringzilla/utf8_uncased_fold.h
 *  @sa include/stringzilla/utf8_uncased_fold/cuda.cuh
 *  @sa include/stringzilla/utf8_uncased_fold/rocm.cuh
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_SIMT_CUH_
#define STRINGZILLA_UTF8_UNCASED_FOLD_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/utf8_uncased_fold/serial.h"

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#include "stringzilla/utf8_uncased_fold/tables.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Fold Tables

/** One codepoint's folded UTF-8, carried in registers rather than in a buffer. */
typedef struct sz_utf8_folded_image_t {

    /** The folded bytes, the first in the lowest byte. */
    sz_u64_t bytes;

    /** How many folded bytes there are, from one to six. */
    sz_u8_t length;

    /** Bit @c index marks the byte at @c index as the last of a folded rune. */
    sz_u8_t rune_ends;
} sz_utf8_folded_image_t;

/** The @p length UTF-8 bytes of a well-formed @p rune, the first in the lowest byte. */
STRINGZILLA_DEVICE sz_u32_t sz_rune_image_simt_(sz_rune_t rune, sz_rune_length_t length) {
    sz_u32_t const low = 0x80u | (rune & 0x3Fu), middle = 0x80u | (rune >> 6 & 0x3Fu),
                   high = 0x80u | (rune >> 12 & 0x3Fu);
    switch (length) {
    case sz_rune_1byte_k: return rune;
    case sz_rune_2bytes_k: return (0xC0u | rune >> 6) | low << 8;
    case sz_rune_3bytes_k: return (0xE0u | rune >> 12) | middle << 8 | low << 16;
    default: return (0xF0u | rune >> 18) | high << 8 | middle << 16 | low << 24;
    }
}

/** The fold of a well-formed @p rune that encodes in @p length bytes, from two dependent loads. */
STRINGZILLA_DEVICE sz_utf8_folded_image_t sz_unicode_fold_image_simt_(sz_rune_t rune, sz_rune_length_t length) {
    sz_utf8_folded_image_t image;
    sz_size_t const page = rune >> sz_unicode_fold_page_bits_simt_k;
    sz_u32_t const entry = page < sizeof(sz_unicode_fold_pages_simt_)
                               ? sz_unicode_fold_images_simt_[(sz_size_t)sz_unicode_fold_pages_simt_[page]
                                                                  << sz_unicode_fold_page_bits_simt_k |
                                                              (rune & ((1u << sz_unicode_fold_page_bits_simt_k) - 1))]
                               : 0;
    if ((entry & 0xC0u) == 0x80u) {
        sz_u64_t const expansion = sz_unicode_fold_expansions_simt_[entry >> 8];
        image.bytes = expansion & 0xFFFFFFFFFFFFull;
        image.length = (sz_u8_t)(expansion >> 48);
        image.rune_ends = (sz_u8_t)(expansion >> 56);
        return image;
    }
    if (entry) {
        sz_u8_t const lead = (sz_u8_t)entry;
        image.bytes = entry;
        image.length = (sz_u8_t)(1 + (lead >= 0xC0u) + (lead >= 0xE0u) + (lead >= 0xF0u));
    }
    else {
        image.bytes = sz_rune_image_simt_(rune, length);
        image.length = (sz_u8_t)length;
    }
    image.rune_ends = (sz_u8_t)(1u << (image.length - 1));
    return image;
}

/** Folds the codepoint at @p text into @p image and returns the source bytes it spans; a byte
 *  beginning no codepoint folds to itself and ends no rune, which is how a walk tells it apart. */
STRINGZILLA_DEVICE sz_size_t sz_utf8_fold_next_simt_(sz_u8_t const *text, sz_u8_t const *end,
                                                     sz_utf8_folded_image_t *image) {
    sz_u8_t const lead = *text;
    sz_rune_t rune;
    sz_rune_length_t rune_length;
    if (lead < 0x80) {
        image->bytes = sz_ascii_fold_(lead), image->length = 1, image->rune_ends = 1;
        return 1;
    }
    rune_length = sz_rune_decode((sz_cptr_t)text, (sz_cptr_t)end, &rune);
    if (rune_length == sz_rune_invalid_k) {
        image->bytes = lead, image->length = 1, image->rune_ends = 0;
        return 1;
    }
    // Nothing under this lead byte folds, so the codepoint's own bytes are its folded image.
    if (sz_utf8_lead_may_fold_(lead)) *image = sz_unicode_fold_image_simt_(rune, rune_length);
    else
        image->bytes = sz_rune_image_simt_(rune, rune_length), image->length = (sz_u8_t)rune_length,
        image->rune_ends = (sz_u8_t)(1u << (rune_length - 1));
    return rune_length;
}

#pragma endregion Fold Tables

#pragma region Fold Kernel

/** Threads one block folds a tile with, which every vendor's launch passes as its block size. */
enum { sz_utf8_uncased_fold_threads_simt_k = 256 };

/** Folds @p source from @p begin to @p end into @p target, or only measures it when @p target is
 *  null, decoding against the whole text's @p length as the serial walk does. */
STRINGZILLA_DEVICE sz_size_t sz_utf8_uncased_fold_span_simt_(sz_u8_t const *source, sz_size_t length, sz_size_t begin,
                                                             sz_size_t end, sz_u8_t *target) {
    sz_size_t folded = 0, index;
    sz_utf8_folded_image_t image;
    while (begin < end) {
        begin += sz_utf8_fold_next_simt_(source + begin, source + length, &image);
        if (target)
            for (index = 0; index != image.length; ++index)
                target[folded + index] = (sz_u8_t)(image.bytes >> (8 * index));
        folded += image.length;
    }
    return folded;
}

#if STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM

/** Folds one tile of @p source, @p tile_bytes wide, of the @p tiles a round cuts it into, measuring
 *  each thread's stretch first and writing it once the chain says where the tile begins. */
static __global__ void sz_utf8_uncased_fold_simt_kernel_(sz_u8_t const *source, sz_size_t length, sz_size_t tile_bytes,
                                                         sz_size_t tiles, sz_u8_t *target, sz_size_t *target_length) {
    __shared__ sz_size_t shared[sz_utf8_uncased_fold_threads_simt_k];
    sz_size_t const ticket = sz_chain_ticket_simt_(target_length);
    sz_size_t const tile_begin = ticket * tile_bytes, tile_end = sz_min_of_two(tile_begin + tile_bytes, length);
    sz_size_t const thread_bytes = sz_size_divide_round_up(tile_end - tile_begin, (sz_size_t)blockDim.x);
    sz_size_t const begin = sz_chain_utf8_cut_simt_(source, length,
                                                    sz_min_of_two(tile_begin + threadIdx.x * thread_bytes, tile_end));
    sz_size_t const end = sz_chain_utf8_cut_simt_(
        source, length, sz_min_of_two(tile_begin + (threadIdx.x + 1) * thread_bytes, tile_end));
    sz_size_t tile_total;
    sz_size_t const preceding = sz_block_scan_simt_(
        sz_utf8_uncased_fold_span_simt_(source, length, begin, end, STRINGZILLA_NULL), shared, &tile_total);
    sz_size_t const tile_offset = sz_chain_offset_simt_(target_length, ticket, tiles, tile_total);
    sz_utf8_uncased_fold_span_simt_(source, length, begin, end, target + tile_offset + preceding);
}

#endif // STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM

#pragma endregion Fold Kernel

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_UTF8_UNCASED_FOLD_SIMT_CUH_
