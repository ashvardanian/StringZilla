/**
 *  @file include/stringzilla/utf8_uncased_fold/metal.metal
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief Lossless Unicode folding and ordered single-threadgroup tiles on Metal.
 */
struct sz_utf8_folded_image_metal_t {
    ulong bytes;
    uchar length, rune_ends;
};

inline uint sz_utf8_fold_next_metal_(device uchar const *text, device uchar const *end,
                                     thread sz_utf8_folded_image_metal_t &image) {
    uint const lead = *text;
    if (lead < 0x80) {
        image.bytes = lead + (uint(lead - 'A') <= 25u ? 0x20u : 0u);
        image.length = 1, image.rune_ends = 1;
        return 1;
    }
    uint rune;
    uint const consumed = sz_utf8_decode_rune_metal_(text, ulong(end - text), 0, rune);
    if (!consumed) {
        image.bytes = lead, image.length = 1, image.rune_ends = 0;
        return 1;
    }
    uint const page = rune >> sz_unicode_fold_page_bits_simt_k;
    uint const entry =
        page < sizeof(sz_unicode_fold_pages_simt_)
            ? sz_unicode_fold_images_simt_[uint(sz_unicode_fold_pages_simt_[page]) << sz_unicode_fold_page_bits_simt_k |
                                           (rune & ((1u << sz_unicode_fold_page_bits_simt_k) - 1))]
            : 0;
    if ((entry & 0xC0u) == 0x80u) {
        ulong const expansion = sz_unicode_fold_expansions_simt_[entry >> 8];
        image.bytes = expansion & 0xFFFFFFFFFFFFul;
        image.length = uchar(expansion >> 48), image.rune_ends = uchar(expansion >> 56);
    }
    else if (entry) {
        uint const folded_lead = uchar(entry);
        image.bytes = entry;
        image.length = uchar(1 + (folded_lead >= 0xC0u) + (folded_lead >= 0xE0u) + (folded_lead >= 0xF0u));
        image.rune_ends = uchar(1u << (image.length - 1));
    }
    else {
        image.bytes = 0;
        for (uint index = 0; index != consumed; ++index) image.bytes |= ulong(text[index]) << (8 * index);
        image.length = uchar(consumed), image.rune_ends = uchar(1u << (consumed - 1));
    }
    return consumed;
}

inline ulong sz_utf8_fold_cut_metal_(device uchar const *text, ulong length, ulong position) {
    for (uint step = 0; step != 3 && position && position < length && (text[position] & 0xC0u) == 0x80u; ++step)
        ++position;
    return position;
}

inline uint sz_utf8_uncased_fold_span_metal_(device uchar const *source, ulong length, ulong begin, ulong end,
                                             device uchar *target) {
    uint folded = 0;
    sz_utf8_folded_image_metal_t image;
    while (begin < end) {
        begin += sz_utf8_fold_next_metal_(source + begin, source + length, image);
        if (target)
            for (uint index = 0; index != image.length; ++index)
                target[folded + index] = uchar(image.bytes >> (8 * index));
        folded += image.length;
    }
    return folded;
}

struct sz_utf8_uncased_fold_arguments_metal_t {
    ulong length, begin, end;
};

kernel void sz_utf8_uncased_fold_clear_metal_kernel_(device ulong *target_length [[buffer(0)]]) { *target_length = 0; }

kernel void sz_utf8_uncased_fold_metal_kernel_(device uchar const *source [[buffer(0)]],
                                               device uchar *target [[buffer(1)]],
                                               device ulong *target_length [[buffer(2)]],
                                               constant sz_utf8_uncased_fold_arguments_metal_t &args [[buffer(3)]],
                                               uint lane [[thread_index_in_threadgroup]],
                                               uint simd_width [[threads_per_simdgroup]]) {
    threadgroup uint totals[256];
    threadgroup ulong base;
    ulong const thread_bytes = sz_size_divide_round_up_metal_(args.end - args.begin, 256ul);
    ulong const begin = sz_utf8_fold_cut_metal_(source, args.length, min(args.begin + lane * thread_bytes, args.end));
    ulong const end = sz_utf8_fold_cut_metal_(source, args.length,
                                              min(args.begin + (lane + 1) * thread_bytes, args.end));
    uint const measured = sz_utf8_uncased_fold_span_metal_(source, args.length, begin, end, nullptr);
    uint const within = lane % simd_width, group = lane / simd_width, groups = 256 / simd_width;
    uint const preceding = simd_prefix_exclusive_sum(measured);
    uint const subtotal = simd_sum(measured);
    if (within + 1 == simd_width) totals[group] = subtotal;
    if (!lane) base = *target_length;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0, total = 0;
    for (uint index = 0; index != groups; ++index) {
        if (index < group) before += totals[index];
        total += totals[index];
    }
    sz_utf8_uncased_fold_span_metal_(source, args.length, begin, end, target + base + before + preceding);
    if (!lane) *target_length = base + total;
}
