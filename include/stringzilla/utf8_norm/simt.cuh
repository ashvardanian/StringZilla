/**
 *  @file include/stringzilla/utf8_norm/simt.cuh
 *  @author Ash Vardanian
 *  @date October 2, 2026
 *  @brief CUDA and ROCm backend for Unicode normalization: a text cut into tiles at safe
 *      boundaries, every thread normalizing one stretch twice, once to measure it and once to
 *      write it where the tiles before it end.
 *
 *  The tables are the serial tier's own: every call copies them into the device's copies below
 *  before its kernel reads them, so `tables.h` stays their only definition, and the lookups read
 *  them as plain global loads. Two calls on different streams write the same bytes into them.
 *
 *  A stretch begins and ends on a safe boundary, a starter whose quick-check is Yes or a malformed
 *  byte, which is where the serial engine splits its dirty regions too, so stretches normalize
 *  independently. Within one, the clean runs are copied and only the dirty ones decompose, reorder
 *  and compose, as in @c sz_utf8_norm_engine_. A tile learns where its output begins through the
 *  @b Chained @b Tiles of `types.cuh`.
 *
 *  Written in C, as every `.cuh` in this library is.
 *
 *  @sa include/stringzilla/utf8_norm.h
 */
#ifndef STRINGZILLA_UTF8_NORM_SIMT_CUH_
#define STRINGZILLA_UTF8_NORM_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/utf8_norm/serial.h"

#if STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
extern "C" {
#endif

/*  The device's copy of every table of `tables.h` a normalizing lookup reads, sized from the table
 *  itself and refilled by every call before its kernel runs. */
#pragma region Device Tables

static __device__ sz_u16_t sz_utf8_norm_stage1_simt_[sizeof(sz_utf8_norm_stage1_) / sizeof(sz_u16_t)];
static __device__ sz_u16_t sz_utf8_norm_stage2_simt_[sizeof(sz_utf8_norm_stage2_) / sizeof(sz_u16_t)];
static __device__ sz_u16_t sz_utf8_norm_stage3_simt_[sizeof(sz_utf8_norm_stage3_) / sizeof(sz_u16_t)];
static __device__ sz_utf8_norm_props_t
    sz_utf8_norm_props_simt_[sizeof(sz_utf8_norm_props_) / sizeof(sz_utf8_norm_props_t)];
static __device__ sz_utf8_norm_decomp_t
    sz_utf8_norm_decomp_simt_[sizeof(sz_utf8_norm_decomp_) / sizeof(sz_utf8_norm_decomp_t)];
static __device__ sz_u16_t sz_utf8_norm_pool_simt_[sizeof(sz_utf8_norm_pool_) / sizeof(sz_u16_t)];
static __device__ sz_rune_t sz_utf8_norm_pool_astral_simt_[sizeof(sz_utf8_norm_pool_astral_) / sizeof(sz_rune_t)];
static __device__ sz_utf8_norm_compose_starter_t
    sz_utf8_norm_compose_starters_simt_[sizeof(sz_utf8_norm_compose_starters_) /
                                        sizeof(sz_utf8_norm_compose_starter_t)];
static __device__ sz_u16_t sz_utf8_norm_compose_partner_simt_[sizeof(sz_utf8_norm_compose_partner_) / sizeof(sz_u16_t)];
static __device__ sz_rune_t sz_utf8_norm_compose_value_simt_[sizeof(sz_utf8_norm_compose_value_) / sizeof(sz_rune_t)];

/** Copies every table the device lookups read into the device's copies, in order on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_utf8_norm_upload_simt_(void *stream) {
    sz_status_t status = sz_copy_to_symbol_simt_(sz_utf8_norm_stage1_simt_, sz_utf8_norm_stage1_,
                                                 sizeof(sz_utf8_norm_stage1_), stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_stage2_simt_, sz_utf8_norm_stage2_, sizeof(sz_utf8_norm_stage2_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_stage3_simt_, sz_utf8_norm_stage3_, sizeof(sz_utf8_norm_stage3_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_props_simt_, sz_utf8_norm_props_, sizeof(sz_utf8_norm_props_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_decomp_simt_, sz_utf8_norm_decomp_, sizeof(sz_utf8_norm_decomp_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_pool_simt_, sz_utf8_norm_pool_, sizeof(sz_utf8_norm_pool_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_pool_astral_simt_, sz_utf8_norm_pool_astral_,
                                         sizeof(sz_utf8_norm_pool_astral_), stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_compose_starters_simt_, sz_utf8_norm_compose_starters_,
                                         sizeof(sz_utf8_norm_compose_starters_), stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_compose_partner_simt_, sz_utf8_norm_compose_partner_,
                                         sizeof(sz_utf8_norm_compose_partner_), stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_simt_(sz_utf8_norm_compose_value_simt_, sz_utf8_norm_compose_value_,
                                         sizeof(sz_utf8_norm_compose_value_), stream);
    return status;
}

#pragma endregion Device Tables

/*  The serial tier's table-reading steps, spelled again over the device's copies; everything that
 *  reads no table, like the canonical ordering, is the serial tier's own. */
#pragma region Device Lookups

/** The properties of @p codepoint, @c sz_utf8_norm_lookup_ over the device's copies. */
STRINGZILLA_DEVICE sz_utf8_norm_props_t sz_utf8_norm_lookup_simt_(sz_rune_t codepoint) {
    sz_size_t index = 0;
    if (codepoint < sz_utf8_norm_table_max_k) {
        sz_size_t const leaf = codepoint >> sz_utf8_norm_low_bits_k;
        sz_size_t const middle = sz_utf8_norm_stage1_simt_[leaf >> sz_utf8_norm_mid_bits_k];
        sz_size_t const block =
            sz_utf8_norm_stage2_simt_[middle * sz_utf8_norm_mid_k + (leaf & sz_utf8_norm_mid_mask_k)];
        index = sz_utf8_norm_stage3_simt_[block * sz_utf8_norm_low_k + (codepoint & sz_utf8_norm_low_mask_k)];
    }
    return sz_utf8_norm_props_simt_[index];
}

/** Whether @p codepoint is a Hangul syllable, which decomposes algorithmically. */
STRINGZILLA_DEVICE sz_bool_t sz_utf8_norm_hangul_simt_(sz_rune_t codepoint) {
    return (sz_bool_t)(codepoint >= sz_utf8_norm_hangul_s_base_k &&
                       codepoint < sz_utf8_norm_hangul_s_base_k + sz_utf8_norm_hangul_s_count_k);
}

/** Whether a split before @p codepoint, of @p properties, is safe for @p form, as
 *  @c sz_utf8_norm_is_safe_boundary_ answers. */
STRINGZILLA_DEVICE sz_bool_t sz_utf8_norm_boundary_simt_(sz_rune_t codepoint, sz_utf8_norm_props_t properties,
                                                         sz_normal_form_t form) {
    if (properties.canonical_combining_class != 0) return sz_false_k;
    switch (form) {
    case sz_normal_form_nfc_k: return (sz_bool_t)((properties.quick_check & 3) == 0);
    case sz_normal_form_nfkc_k: return (sz_bool_t)((properties.quick_check & 12) == 0);
    case sz_normal_form_nfd_k: return (sz_bool_t)(!properties.nfd && !sz_utf8_norm_hangul_simt_(codepoint));
    default: return (sz_bool_t)(!properties.nfkd && !sz_utf8_norm_hangul_simt_(codepoint));
    }
}

/** Whether @p codepoint, of @p properties, may change under @p form, as
 *  @c sz_utf8_norm_classify_serial_ flags it. */
STRINGZILLA_DEVICE sz_bool_t sz_utf8_norm_dirty_simt_(sz_rune_t codepoint, sz_utf8_norm_props_t properties,
                                                      sz_normal_form_t form) {
    if (properties.canonical_combining_class != 0) return sz_true_k;
    switch (form) {
    case sz_normal_form_nfc_k: return (sz_bool_t)((properties.quick_check & 3) != 0);
    case sz_normal_form_nfkc_k: return (sz_bool_t)((properties.quick_check & 12) != 0);
    case sz_normal_form_nfd_k: return (sz_bool_t)(properties.nfd || sz_utf8_norm_hangul_simt_(codepoint));
    default: return (sz_bool_t)(properties.nfkd || sz_utf8_norm_hangul_simt_(codepoint));
    }
}

/** @c sz_utf8_norm_decompose_rune_ over the device's copies. */
STRINGZILLA_DEVICE sz_size_t sz_utf8_norm_decompose_rune_simt_(sz_rune_t codepoint, sz_bool_t compat, sz_rune_t *out,
                                                               sz_u8_t *out_canonical_combining_class) {
    if (sz_utf8_norm_hangul_simt_(codepoint)) {
        sz_u32_t const syllable = codepoint - sz_utf8_norm_hangul_s_base_k;
        sz_u32_t const trailing = syllable % sz_utf8_norm_hangul_t_count_k;
        out[0] = sz_utf8_norm_hangul_l_base_k + syllable / sz_utf8_norm_hangul_n_count_k;
        out[1] = sz_utf8_norm_hangul_v_base_k +
                 (syllable % sz_utf8_norm_hangul_n_count_k) / sz_utf8_norm_hangul_t_count_k;
        out_canonical_combining_class[0] = 0, out_canonical_combining_class[1] = 0;
        if (!trailing) return 2;
        out[2] = sz_utf8_norm_hangul_t_base_k + trailing, out_canonical_combining_class[2] = 0;
        return 3;
    }
    sz_utf8_norm_props_t const properties = sz_utf8_norm_lookup_simt_(codepoint);
    sz_u16_t const index = compat ? properties.nfkd : properties.nfd;
    if (index == 0) {
        out[0] = codepoint, out_canonical_combining_class[0] = properties.canonical_combining_class;
        return 1;
    }
    sz_utf8_norm_decomp_t const decomposition = sz_utf8_norm_decomp_simt_[index];
    for (sz_size_t i = 0; i != decomposition.length; ++i) {
        sz_u16_t const value = sz_utf8_norm_pool_simt_[decomposition.offset + i];
        sz_rune_t const rune = value < sz_utf8_norm_pool_astral_k
                                   ? (sz_rune_t)value
                                   : sz_utf8_norm_pool_astral_simt_[value - sz_utf8_norm_pool_astral_k];
        out[i] = rune, out_canonical_combining_class[i] = sz_utf8_norm_lookup_simt_(rune).canonical_combining_class;
    }
    return decomposition.length;
}

/** @c sz_utf8_norm_compose_pair_ over the device's copies. */
STRINGZILLA_DEVICE sz_rune_t sz_utf8_norm_compose_pair_simt_(sz_rune_t a, sz_rune_t b) {
    if (a >= sz_utf8_norm_hangul_l_base_k && a < sz_utf8_norm_hangul_l_base_k + sz_utf8_norm_hangul_l_count_k &&
        b >= sz_utf8_norm_hangul_v_base_k && b < sz_utf8_norm_hangul_v_base_k + sz_utf8_norm_hangul_v_count_k)
        return sz_utf8_norm_hangul_s_base_k + ((a - sz_utf8_norm_hangul_l_base_k) * sz_utf8_norm_hangul_v_count_k +
                                               (b - sz_utf8_norm_hangul_v_base_k)) *
                                                  sz_utf8_norm_hangul_t_count_k;
    if (sz_utf8_norm_hangul_simt_(a) && (a - sz_utf8_norm_hangul_s_base_k) % sz_utf8_norm_hangul_t_count_k == 0 &&
        b > sz_utf8_norm_hangul_t_base_k && b < sz_utf8_norm_hangul_t_base_k + sz_utf8_norm_hangul_t_count_k)
        return a + (b - sz_utf8_norm_hangul_t_base_k);

    sz_utf8_norm_props_t const properties_a = sz_utf8_norm_lookup_simt_(a), properties_b = sz_utf8_norm_lookup_simt_(b);
    if (properties_a.starter == 0xFFFF || properties_b.partner == 0xFFFF) return 0;
    sz_utf8_norm_compose_starter_t const starter = sz_utf8_norm_compose_starters_simt_[properties_a.starter];
    sz_size_t low = starter.offset, high = (sz_size_t)starter.offset + starter.count;
    while (low < high) {
        sz_size_t const middle = low + ((high - low) >> 1);
        if (sz_utf8_norm_compose_partner_simt_[middle] < properties_b.partner) low = middle + 1;
        else high = middle;
    }
    if (low < (sz_size_t)starter.offset + starter.count &&
        sz_utf8_norm_compose_partner_simt_[low] == properties_b.partner)
        return sz_utf8_norm_compose_value_simt_[low];
    return 0;
}

#pragma endregion Device Lookups

#pragma region Normalize Kernel

enum {

    /** Threads one block normalizes a tile with. */
    sz_utf8_norm_threads_simt_k = 256,

    /** Bytes one thread normalizes at the least, which keeps a short text to few tiles. */
    sz_utf8_norm_thread_bytes_simt_k = 64,
};

/** Appends @p rune's UTF-8 at @p written bytes into @p target, or only counts it when @p target is
 *  null, and returns its length. */
STRINGZILLA_DEVICE sz_size_t sz_utf8_norm_emit_simt_(sz_u8_t *target, sz_size_t written, sz_rune_t rune) {
    if (target) return (sz_size_t)sz_rune_encode(rune, target + written);
    return 1 + (rune > 0x7F) + (rune > 0x7FF) + (rune > 0xFFFF);
}

/** @c sz_utf8_norm_flush_ over the device's copies: orders, optionally composes, and emits one
 *  segment, returning the bytes it took. */
STRINGZILLA_DEVICE sz_size_t sz_utf8_norm_flush_simt_(sz_rune_t *runes, sz_u8_t *canonical_combining_classes,
                                                      sz_size_t count, sz_bool_t compose, sz_u8_t *target,
                                                      sz_size_t written) {
    sz_size_t const before = written;
    if (count == 0) return 0;
    sz_utf8_norm_canonical_order_(runes, canonical_combining_classes, count);
    if (compose && canonical_combining_classes[0] == 0) {
        sz_rune_t starter = runes[0];
        sz_size_t produced = 1;
        int last_canonical_combining_class = 0;
        for (sz_size_t i = 1; i < count; ++i) {
            sz_rune_t const codepoint = runes[i];
            sz_u8_t const canonical_combining_class = canonical_combining_classes[i];
            if (last_canonical_combining_class < (int)canonical_combining_class ||
                last_canonical_combining_class == 0) {
                sz_rune_t const composed = sz_utf8_norm_compose_pair_simt_(starter, codepoint);
                if (composed) {
                    starter = composed, runes[0] = composed;
                    continue;
                }
            }
            last_canonical_combining_class = canonical_combining_class;
            runes[produced] = codepoint, canonical_combining_classes[produced] = canonical_combining_class;
            ++produced;
        }
        count = produced;
    }
    for (sz_size_t i = 0; i < count; ++i) written += sz_utf8_norm_emit_simt_(target, written, runes[i]);
    return written - before;
}

/** @c sz_utf8_norm_run_ over the device's copies: normalizes @p length bytes of @p source at
 *  @p written bytes into @p target, or only measures them, returning the bytes they took. */
STRINGZILLA_DEVICE sz_size_t sz_utf8_norm_run_simt_(sz_u8_t const *source, sz_size_t length, sz_normal_form_t form,
                                                    sz_u8_t *target, sz_size_t written) {
    sz_bool_t const compat = (sz_bool_t)(form == sz_normal_form_nfkd_k || form == sz_normal_form_nfkc_k);
    sz_bool_t const compose = (sz_bool_t)(form == sz_normal_form_nfc_k || form == sz_normal_form_nfkc_k);
    sz_u8_t const *position = source, *const end = source + length;
    sz_rune_t segment[sz_utf8_norm_seg_cap_k], decomposed[sz_utf8_norm_decomp_max_k];
    sz_u8_t segment_classes[sz_utf8_norm_seg_cap_k], decomposed_classes[sz_utf8_norm_decomp_max_k];
    sz_size_t segment_length = 0, parts, part;
    sz_size_t const before = written;
    while (position < end) {
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)position, (sz_cptr_t)end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            // A malformed byte is an opaque barrier, emitted verbatim between the segments.
            written += sz_utf8_norm_flush_simt_(segment, segment_classes, segment_length, compose, target, written);
            segment_length = 0;
            if (target) target[written] = *position;
            ++written, ++position;
            continue;
        }
        position += rune_length;
        parts = sz_utf8_norm_decompose_rune_simt_(rune, compat, decomposed, decomposed_classes);
        for (part = 0; part != parts; ++part) {
            if (decomposed_classes[part] == 0) {
                if (compose && segment_length == 1 && segment_classes[0] == 0) {
                    sz_rune_t const merged = sz_utf8_norm_compose_pair_simt_(segment[0], decomposed[part]);
                    if (merged) {
                        segment[0] = merged;
                        continue;
                    }
                }
                written += sz_utf8_norm_flush_simt_(segment, segment_classes, segment_length, compose, target, written);
                segment[0] = decomposed[part], segment_classes[0] = 0, segment_length = 1;
            }
            else {
                if (segment_length >= sz_utf8_norm_seg_cap_k) {
                    written += sz_utf8_norm_flush_simt_(segment, segment_classes, segment_length, compose, target,
                                                        written);
                    segment_length = 0;
                }
                segment[segment_length] = decomposed[part], segment_classes[segment_length] = decomposed_classes[part];
                ++segment_length;
            }
        }
    }
    written += sz_utf8_norm_flush_simt_(segment, segment_classes, segment_length, compose, target, written);
    return written - before;
}

/** Bytes the codepoint or malformed byte at @p position takes, and whether a split before it is
 *  safe for @p form, decoding against the whole text's @p length. */
STRINGZILLA_DEVICE sz_size_t sz_utf8_norm_step_simt_(sz_u8_t const *text, sz_size_t length, sz_size_t position,
                                                     sz_normal_form_t form, sz_bool_t *boundary) {
    sz_rune_t rune;
    sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)text + position, (sz_cptr_t)text + length, &rune);
    if (rune_length == sz_rune_invalid_k) {
        *boundary = sz_true_k;
        return 1;
    }
    *boundary = sz_utf8_norm_boundary_simt_(rune, sz_utf8_norm_lookup_simt_(rune), form);
    return rune_length;
}

/** The first safe boundary for @p form at or after @p position, which is where both neighbours of
 *  a cut agree to split. */
STRINGZILLA_DEVICE sz_size_t sz_utf8_norm_cut_simt_(sz_u8_t const *text, sz_size_t length, sz_size_t position,
                                                    sz_normal_form_t form) {
    sz_bool_t boundary = sz_false_k;
    position = sz_chain_utf8_cut_simt_(text, length, position);
    while (position != 0 && position < length) {
        sz_size_t const step = sz_utf8_norm_step_simt_(text, length, position, form, &boundary);
        if (boundary) break;
        position += step;
    }
    return position;
}

/**
 *  @brief Normalizes @p text from @p begin to @p end, both safe boundaries, into @p target, or only
 *      measures it when @p target is null; @c sz_utf8_norm_engine_ over that stretch.
 *
 *  Walking forward, the last safe boundary seen is where a dirty codepoint's region begins,
 *  which is where the serial engine's backward search for it stops, and the region ends at the
 *  next safe boundary after it.
 */
STRINGZILLA_DEVICE sz_size_t sz_utf8_norm_span_simt_(sz_u8_t const *text, sz_size_t length, sz_size_t begin,
                                                     sz_size_t end, sz_normal_form_t form, sz_u8_t *target) {
    sz_size_t written = 0, clean_from = begin, segment = begin, position = begin, tail, index;
    sz_bool_t boundary;
    while (position < end) {
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)text + position, (sz_cptr_t)text + length,
                                                            &rune);
        sz_utf8_norm_props_t properties;
        if (rune_length == sz_rune_invalid_k) {
            segment = position++;
            continue;
        }
        properties = sz_utf8_norm_lookup_simt_(rune);
        if (sz_utf8_norm_boundary_simt_(rune, properties, form)) segment = position;
        if (!sz_utf8_norm_dirty_simt_(rune, properties, form)) {
            position += rune_length;
            continue;
        }
        tail = position + rune_length;
        while (tail < end) {
            sz_size_t const step = sz_utf8_norm_step_simt_(text, length, tail, form, &boundary);
            if (boundary) break;
            tail += step;
        }
        if (target)
            for (index = clean_from; index != segment; ++index) target[written + index - clean_from] = text[index];
        written += segment - clean_from;
        written += sz_utf8_norm_run_simt_(text + segment, tail - segment, form, target, written);
        clean_from = segment = position = tail;
    }
    if (target)
        for (index = clean_from; index != end; ++index) target[written + index - clean_from] = text[index];
    return written + (end - clean_from);
}

/** Normalizes one tile of @p text, @p tile_bytes wide, of the @p tiles a round cuts it into,
 *  measuring each thread's stretch first, then writing it once the chain says where it begins. */
static __global__ void sz_utf8_norm_simt_kernel_(sz_u8_t const *text, sz_size_t length, sz_normal_form_t form,
                                                 sz_size_t tile_bytes, sz_size_t tiles, sz_u8_t *target,
                                                 sz_size_t *target_length) {
    __shared__ sz_size_t shared[sz_utf8_norm_threads_simt_k];
    sz_size_t const ticket = sz_chain_ticket_simt_(target_length);
    sz_size_t const tile_begin = ticket * tile_bytes, tile_end = sz_min_of_two(tile_begin + tile_bytes, length);
    sz_size_t const thread_bytes = sz_size_divide_round_up(tile_end - tile_begin, (sz_size_t)blockDim.x);
    sz_size_t const begin = sz_utf8_norm_cut_simt_(
        text, length, sz_min_of_two(tile_begin + threadIdx.x * thread_bytes, tile_end), form);
    sz_size_t const end = sz_utf8_norm_cut_simt_(
        text, length, sz_min_of_two(tile_begin + (threadIdx.x + 1) * thread_bytes, tile_end), form);
    sz_size_t tile_total;
    sz_size_t const preceding = sz_block_scan_simt_(
        sz_utf8_norm_span_simt_(text, length, begin, end, form, STRINGZILLA_NULL), shared, &tile_total);
    sz_size_t const tile_offset = sz_chain_offset_simt_(target_length, ticket, tiles, tile_total);
    sz_utf8_norm_span_simt_(text, length, begin, end, form, target + tile_offset + preceding);
}

/**
 *  @brief Normalizes @p source into @p target on the caller's current device, the length landing in
 *      @p target_length once @p stream is joined.
 *  @return @c sz_success_k once enqueued, @c sz_unexpected_dimensions_k for a text whose
 *      normalization the length slot cannot count, or @c sz_device_memory_mismatch_k when a buffer
 *      or the slot is not memory the device reaches.
 *  @note Allocates nothing and joins nothing; the tables travel on @p stream ahead of the kernel.
 */
STRINGZILLA_INLINE sz_status_t sz_utf8_norm_simt_(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                                  sz_ptr_t target, sz_size_t *target_length, void *stream) {
    sz_u8_t const *launch_source = (sz_u8_t const *)source;
    sz_u8_t *launch_target = (sz_u8_t *)target;
    sz_size_t *launch_target_length = target_length;
    sz_size_t launch_length = source_length, tile_bytes, tiles;
    sz_normal_form_t launch_form = form;
    void *arguments[7];
    dim3 grid, block;
    sz_status_t status;
    if ((sz_u64_t)source_length > ((sz_u64_t)1 << sz_chain_chained_shift_k) / sz_utf8_norm_decomp_max_k)
        return sz_unexpected_dimensions_k;
    if (!sz_memory_reaches_simt_(target_length)) return sz_device_memory_mismatch_k;
    if (source_length && (!sz_memory_reaches_simt_(source) || !sz_memory_reaches_simt_(target)))
        return sz_device_memory_mismatch_k;
    status = sz_fill_simt_(target_length, sizeof(sz_size_t), 0, stream);
    if (status != sz_success_k || !source_length) return status;
    status = sz_utf8_norm_upload_simt_(stream);
    if (status != sz_success_k) return status;

    tiles = sz_chain_tiles_simt_(
        source_length, (sz_size_t)sz_utf8_norm_threads_simt_k * sz_utf8_norm_thread_bytes_simt_k, &tile_bytes);
    grid.x = (unsigned)tiles, grid.y = 1, grid.z = 1;
    block.x = sz_utf8_norm_threads_simt_k, block.y = 1, block.z = 1;
    arguments[0] = &launch_source, arguments[1] = &launch_length, arguments[2] = &launch_form;
    arguments[3] = &tile_bytes, arguments[4] = &tiles, arguments[5] = &launch_target;
    arguments[6] = &launch_target_length;
    return sz_launch_simt_((void const *)sz_utf8_norm_simt_kernel_, grid, block, arguments, 0, stream);
}

STRINGZILLA_INLINE sz_status_t sz_utf8_norm_scoped_simt_(sz_cptr_t source, sz_size_t source_length,
                                                         sz_normal_form_t form, sz_ptr_t target,
                                                         sz_size_t *target_length, void *stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_simt_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_utf8_norm_simt_(source, source_length, form, target, target_length, stream);
    sz_device_leave_simt_(caller);
    return status;
}

#pragma endregion Normalize Kernel

#if STRINGZILLA_TARGET_CUDA

STRINGZILLA_API sz_status_t sz_utf8_norm_cuda(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                              sz_ptr_t target, sz_size_t *target_length, void *stream) {
    return sz_utf8_norm_scoped_simt_(source, source_length, form, target, target_length, stream);
}

#endif // STRINGZILLA_TARGET_CUDA

#if STRINGZILLA_TARGET_ROCM

STRINGZILLA_API sz_status_t sz_utf8_norm_rocm(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                              sz_ptr_t target, sz_size_t *target_length, void *stream) {
    return sz_utf8_norm_scoped_simt_(source, source_length, form, target, target_length, stream);
}

#endif // STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM
#endif // STRINGZILLA_UTF8_NORM_SIMT_CUH_
