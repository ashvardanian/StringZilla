/**
 *  @file include/stringzilla/utf8_norm/metal.metal
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief Metal Unicode normalization with shared decomposition and composition tables.
 */
enum { sz_normal_form_nfd_k = 0, sz_normal_form_nfc_k = 1, sz_normal_form_nfkd_k = 2, sz_normal_form_nfkc_k = 3 };
enum {
    sz_utf8_norm_table_max_k = 0x30000,
    sz_utf8_norm_low_bits_k = 3,
    sz_utf8_norm_low_k = 1 << sz_utf8_norm_low_bits_k,
    sz_utf8_norm_low_mask_k = sz_utf8_norm_low_k - 1,
    sz_utf8_norm_mid_bits_k = 5,
    sz_utf8_norm_mid_k = 1 << sz_utf8_norm_mid_bits_k,
    sz_utf8_norm_mid_mask_k = sz_utf8_norm_mid_k - 1,
    sz_utf8_norm_pool_astral_k = 0xD800,
};

enum {

    sz_utf8_norm_seg_cap_k = 256,

    sz_utf8_norm_decomp_max_k = 18
};

enum {
    sz_utf8_norm_hangul_s_base_k = 0xAC00,
    sz_utf8_norm_hangul_l_base_k = 0x1100,
    sz_utf8_norm_hangul_v_base_k = 0x1161,
    sz_utf8_norm_hangul_t_base_k = 0x11A7,
    sz_utf8_norm_hangul_l_count_k = 19,
    sz_utf8_norm_hangul_v_count_k = 21,
    sz_utf8_norm_hangul_t_count_k = 28,
    sz_utf8_norm_hangul_n_count_k = sz_utf8_norm_hangul_v_count_k * sz_utf8_norm_hangul_t_count_k,
    sz_utf8_norm_hangul_s_count_k = sz_utf8_norm_hangul_l_count_k * sz_utf8_norm_hangul_n_count_k,
};

inline void sz_utf8_norm_canonical_order_metal_(thread uint *runes, thread uchar *canonical_combining_classes,
                                                ulong count) {
    for (ulong i = 1; i < count; ++i) {
        uint rune = runes[i];
        uchar canonical_combining_class = canonical_combining_classes[i];
        if (canonical_combining_class == 0) continue;

        ulong j = i;
        while (j > 0 && canonical_combining_classes[j - 1] > canonical_combining_class) --j;
        for (ulong k = i; k > j; --k)
            runes[k] = runes[k - 1], canonical_combining_classes[k] = canonical_combining_classes[k - 1];
        runes[j] = rune;
        canonical_combining_classes[j] = canonical_combining_class;
    }
}

inline ulong sz_utf8_norm_encode_metal_(uint rune, device uchar *target) {
    if (rune < 0x80) {
        target[0] = (uchar)rune;
        return 1;
    }
    if (rune < 0x800) {
        target[0] = (uchar)(0xC0 | (rune >> 6));
        target[1] = (uchar)(0x80 | (rune & 63));
        return 2;
    }
    if (rune < 0x10000) {
        target[0] = (uchar)(0xE0 | (rune >> 12));
        target[1] = (uchar)(0x80 | ((rune >> 6) & 63));
        target[2] = (uchar)(0x80 | (rune & 63));
        return 3;
    }
    target[0] = (uchar)(0xF0 | (rune >> 18));
    target[1] = (uchar)(0x80 | ((rune >> 12) & 63));
    target[2] = (uchar)(0x80 | ((rune >> 6) & 63));
    target[3] = (uchar)(0x80 | (rune & 63));
    return 4;
}

inline sz_utf8_norm_props_t sz_utf8_norm_lookup_metal_(uint codepoint) {
    ulong index = 0;
    if (codepoint < sz_utf8_norm_table_max_k) {
        ulong const leaf = codepoint >> sz_utf8_norm_low_bits_k;
        ulong const middle = sz_utf8_norm_stage1_[leaf >> sz_utf8_norm_mid_bits_k];
        ulong const block = sz_utf8_norm_stage2_[middle * sz_utf8_norm_mid_k + (leaf & sz_utf8_norm_mid_mask_k)];
        index = sz_utf8_norm_stage3_[block * sz_utf8_norm_low_k + (codepoint & sz_utf8_norm_low_mask_k)];
    }
    return sz_utf8_norm_props_[index];
}

inline bool sz_utf8_norm_hangul_metal_(uint codepoint) {
    return (codepoint >= sz_utf8_norm_hangul_s_base_k &&
            codepoint < sz_utf8_norm_hangul_s_base_k + sz_utf8_norm_hangul_s_count_k);
}

inline bool sz_utf8_norm_dirty_metal_(uint codepoint, sz_utf8_norm_props_t properties, uint form) {
    if (properties.canonical_combining_class != 0) return true;
    switch (form) {
    case sz_normal_form_nfc_k: return ((properties.quick_check & 3) != 0);
    case sz_normal_form_nfkc_k: return ((properties.quick_check & 12) != 0);
    case sz_normal_form_nfd_k: return (properties.nfd || sz_utf8_norm_hangul_metal_(codepoint));
    default: return (properties.nfkd || sz_utf8_norm_hangul_metal_(codepoint));
    }
}

inline ulong sz_utf8_norm_decompose_rune_metal_(uint codepoint, uint form, thread uint *out,
                                                thread uchar *out_canonical_combining_class) {
    bool const compat = form == sz_normal_form_nfkd_k || form == sz_normal_form_nfkc_k;
    if (sz_utf8_norm_hangul_metal_(codepoint)) {
        uint const syllable = codepoint - sz_utf8_norm_hangul_s_base_k;
        uint const trailing = syllable % sz_utf8_norm_hangul_t_count_k;
        out[0] = sz_utf8_norm_hangul_l_base_k + syllable / sz_utf8_norm_hangul_n_count_k;
        out[1] = sz_utf8_norm_hangul_v_base_k +
                 (syllable % sz_utf8_norm_hangul_n_count_k) / sz_utf8_norm_hangul_t_count_k;
        out_canonical_combining_class[0] = 0, out_canonical_combining_class[1] = 0;
        if (!trailing) return 2;
        out[2] = sz_utf8_norm_hangul_t_base_k + trailing, out_canonical_combining_class[2] = 0;
        return 3;
    }
    sz_utf8_norm_props_t const properties = sz_utf8_norm_lookup_metal_(codepoint);
    ushort const index = compat ? properties.nfkd : properties.nfd;
    if (index == 0) {
        out[0] = codepoint, out_canonical_combining_class[0] = properties.canonical_combining_class;
        return 1;
    }
    sz_utf8_norm_decomp_t const decomposition = sz_utf8_norm_decomp_[index];
    for (ulong i = 0; i != decomposition.length; ++i) {
        ushort const value = sz_utf8_norm_pool_[decomposition.offset + i];
        uint const rune = value < sz_utf8_norm_pool_astral_k
                              ? (uint)value
                              : sz_utf8_norm_pool_astral_[value - sz_utf8_norm_pool_astral_k];
        out[i] = rune, out_canonical_combining_class[i] = sz_utf8_norm_lookup_metal_(rune).canonical_combining_class;
    }
    return decomposition.length;
}

inline uint sz_utf8_norm_compose_pair_metal_(uint a, uint b) {
    if (a >= sz_utf8_norm_hangul_l_base_k && a < sz_utf8_norm_hangul_l_base_k + sz_utf8_norm_hangul_l_count_k &&
        b >= sz_utf8_norm_hangul_v_base_k && b < sz_utf8_norm_hangul_v_base_k + sz_utf8_norm_hangul_v_count_k)
        return sz_utf8_norm_hangul_s_base_k + ((a - sz_utf8_norm_hangul_l_base_k) * sz_utf8_norm_hangul_v_count_k +
                                               (b - sz_utf8_norm_hangul_v_base_k)) *
                                                  sz_utf8_norm_hangul_t_count_k;
    if (sz_utf8_norm_hangul_metal_(a) && (a - sz_utf8_norm_hangul_s_base_k) % sz_utf8_norm_hangul_t_count_k == 0 &&
        b > sz_utf8_norm_hangul_t_base_k && b < sz_utf8_norm_hangul_t_base_k + sz_utf8_norm_hangul_t_count_k)
        return a + (b - sz_utf8_norm_hangul_t_base_k);

    sz_utf8_norm_props_t const properties_a = sz_utf8_norm_lookup_metal_(a),
                               properties_b = sz_utf8_norm_lookup_metal_(b);
    if (properties_a.starter == 0xFFFF || properties_b.partner == 0xFFFF) return 0;
    sz_utf8_norm_compose_starter_t const starter = sz_utf8_norm_compose_starters_[properties_a.starter];
    ulong low = starter.offset, high = (ulong)starter.offset + starter.count;
    while (low < high) {
        ulong const middle = low + ((high - low) >> 1);
        if (sz_utf8_norm_compose_partner_[middle] < properties_b.partner) low = middle + 1;
        else high = middle;
    }
    if (low < (ulong)starter.offset + starter.count && sz_utf8_norm_compose_partner_[low] == properties_b.partner)
        return sz_utf8_norm_compose_value_[low];
    return 0;
}

inline ulong sz_utf8_norm_emit_metal_(device uchar *target, ulong written, uint rune) {
    if (target) return (ulong)sz_utf8_norm_encode_metal_(rune, target + written);
    return 1 + (rune > 0x7F) + (rune > 0x7FF) + (rune > 0xFFFF);
}

inline ulong sz_utf8_norm_flush_metal_(thread uint *runes, thread uchar *canonical_combining_classes, ulong count,
                                       uint form, device uchar *target, ulong written) {
    bool const compose = form == sz_normal_form_nfc_k || form == sz_normal_form_nfkc_k;
    ulong const before = written;
    if (count == 0) return 0;
    sz_utf8_norm_canonical_order_metal_(runes, canonical_combining_classes, count);
    if (compose && canonical_combining_classes[0] == 0) {
        uint starter = runes[0];
        ulong produced = 1;
        int last_canonical_combining_class = 0;
        for (ulong i = 1; i < count; ++i) {
            uint const codepoint = runes[i];
            uchar const canonical_combining_class = canonical_combining_classes[i];
            if (last_canonical_combining_class < (int)canonical_combining_class ||
                last_canonical_combining_class == 0) {
                uint const composed = sz_utf8_norm_compose_pair_metal_(starter, codepoint);
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
    for (ulong i = 0; i < count; ++i) written += sz_utf8_norm_emit_metal_(target, written, runes[i]);
    return written - before;
}

inline ulong sz_utf8_norm_run_metal_(device uchar const *source, ulong length, uint form, device uchar *target,
                                     ulong written) {
    bool const compose = (form == sz_normal_form_nfc_k || form == sz_normal_form_nfkc_k);
    device uchar const *position = source, *const end = source + length;
    uint segment[sz_utf8_norm_seg_cap_k], decomposed[sz_utf8_norm_decomp_max_k];
    uchar segment_classes[sz_utf8_norm_seg_cap_k], decomposed_classes[sz_utf8_norm_decomp_max_k];
    ulong segment_length = 0, parts, part;
    ulong const before = written;
    while (position < end) {
        uint rune;
        uint const rune_length = sz_utf8_decode_rune_metal_(position, (ulong)(end - position), 0, rune);
        if (rune_length == 0) {
            // A malformed byte is an opaque barrier, emitted verbatim between the segments.
            written += sz_utf8_norm_flush_metal_(segment, segment_classes, segment_length, form, target, written);
            segment_length = 0;
            if (target) target[written] = *position;
            ++written, ++position;
            continue;
        }
        position += rune_length;
        parts = sz_utf8_norm_decompose_rune_metal_(rune, form, decomposed, decomposed_classes);
        for (part = 0; part != parts; ++part) {
            if (decomposed_classes[part] == 0) {
                if (compose && segment_length == 1 && segment_classes[0] == 0) {
                    uint const merged = sz_utf8_norm_compose_pair_metal_(segment[0], decomposed[part]);
                    if (merged) {
                        segment[0] = merged;
                        continue;
                    }
                }
                written += sz_utf8_norm_flush_metal_(segment, segment_classes, segment_length, form, target, written);
                segment[0] = decomposed[part], segment_classes[0] = 0, segment_length = 1;
            }
            else {
                if (segment_length >= sz_utf8_norm_seg_cap_k) {
                    written += sz_utf8_norm_flush_metal_(segment, segment_classes, segment_length, form, target,
                                                         written);
                    segment_length = 0;
                }
                segment[segment_length] = decomposed[part], segment_classes[segment_length] = decomposed_classes[part];
                ++segment_length;
            }
        }
    }
    written += sz_utf8_norm_flush_metal_(segment, segment_classes, segment_length, form, target, written);
    return written - before;
}

inline ulong sz_utf8_norm_step_metal_(device uchar const *text, ulong length, ulong position, uint form,
                                      thread bool *boundary) {
    uint rune;
    uint const rune_length = sz_utf8_decode_rune_metal_(text, length, position, rune);
    if (rune_length == 0) {
        *boundary = true;
        return 1;
    }
    *boundary = !sz_utf8_norm_dirty_metal_(rune, sz_utf8_norm_lookup_metal_(rune), form);
    return rune_length;
}

inline ulong sz_utf8_norm_cut_metal_(device uchar const *text, ulong length, ulong position, ulong limit, uint form) {
    if (!position) return 0;
    for (uint step = 0; step != 3 && position < length && (text[position] & 0xC0u) == 0x80u; ++step) ++position;
    while (position < limit) {
        bool boundary;
        ulong const step = sz_utf8_norm_step_metal_(text, length, position, form, &boundary);
        if (boundary) return position;
        position += step;
    }
    return limit;
}

inline ulong sz_utf8_norm_span_metal_(device uchar const *text, ulong length, ulong begin, ulong end, uint form,
                                      device uchar *target) {
    ulong written = 0, clean_from = begin, segment = begin, position = begin, tail, index;
    bool boundary;
    while (position < end) {
        uint rune;
        uint const rune_length = sz_utf8_decode_rune_metal_(text, length, position, rune);
        sz_utf8_norm_props_t properties;
        if (rune_length == 0) {
            segment = position++;
            continue;
        }
        properties = sz_utf8_norm_lookup_metal_(rune);
        if (!sz_utf8_norm_dirty_metal_(rune, properties, form)) {
            segment = position;
            position += rune_length;
            continue;
        }
        tail = position + rune_length;
        while (tail < end) {
            ulong const step = sz_utf8_norm_step_metal_(text, length, tail, form, &boundary);
            if (boundary) break;
            tail += step;
        }
        if (target)
            for (index = clean_from; index != segment; ++index) target[written + index - clean_from] = text[index];
        written += segment - clean_from;
        written += sz_utf8_norm_run_metal_(text + segment, tail - segment, form, target, written);
        clean_from = segment = position = tail;
    }
    if (target)
        for (index = clean_from; index != end; ++index) target[written + index - clean_from] = text[index];
    return written + (end - clean_from);
}

struct sz_utf8_norm_arguments_metal_t {
    ulong length, form, chunks;
};
kernel void sz_utf8_norm_metal_kernel_(device uchar const *text [[buffer(0)]], device uchar *target [[buffer(1)]],
                                       device ulong *target_length [[buffer(2)]],
                                       constant sz_utf8_norm_arguments_metal_t &arguments [[buffer(3)]],
                                       uint lane [[thread_index_in_threadgroup]],
                                       uint threads [[threads_per_threadgroup]]) {
    threadgroup ulong offsets[256];
    ulong const length = arguments.length;
    ulong const quotient = length / threads, remainder = length % threads;
    ulong const begin = sz_utf8_norm_cut_metal_(text, length, lane * quotient + min((ulong)lane, remainder), length,
                                                (uint)arguments.form);
    ulong const end = sz_utf8_norm_cut_metal_(text, length, (lane + 1) * quotient + min((ulong)(lane + 1), remainder),
                                              length, (uint)arguments.form);
    offsets[lane] = sz_utf8_norm_span_metal_(text, length, begin, end, (uint)arguments.form, nullptr);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) {
        ulong total = 0;
        for (uint i = 0; i < threads; ++i) {
            ulong const bytes = offsets[i];
            offsets[i] = total;
            total += bytes;
        }
        *target_length = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sz_utf8_norm_span_metal_(text, length, begin, end, (uint)arguments.form, target + offsets[lane]);
}

struct sz_utf8_norm_span_metal_t {
    ulong begin, end, offset;
};

inline ulong sz_utf8_norm_scan_metal_(ulong value, uint lane, threadgroup ulong *totals) {
    uint const low = uint(value);
    uint const low_prefix = simd_prefix_exclusive_sum(low);
    uint const carries = simd_prefix_exclusive_sum(uint(low_prefix + low < low_prefix));
    uint const high_prefix = simd_prefix_exclusive_sum(uint(value >> 32));
    ulong const prefix = (ulong(high_prefix + carries) << 32) | low_prefix;
    if ((lane & 31) == 31) totals[lane / 32] = prefix + value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong preceding = 0;
    for (uint group = 0; group < lane / 32; ++group) preceding += totals[group];
    return preceding + prefix;
}

kernel void sz_utf8_norm_count_metal_kernel_(device uchar const *text [[buffer(0)]],
                                             device sz_utf8_norm_span_metal_t *spans [[buffer(3)]],
                                             device ulong *blocks [[buffer(4)]],
                                             constant sz_utf8_norm_arguments_metal_t &arguments [[buffer(5)]],
                                             uint group [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_threadgroup]]) {
    threadgroup ulong totals[1];
    ulong const index = ulong(group) * 32 + lane;
    ulong begin = arguments.length, end = arguments.length, bytes = 0;
    if (index < arguments.chunks) {
        ulong const first = index * 64, limit = min(first + 64, arguments.length);
        begin = sz_utf8_norm_cut_metal_(text, arguments.length, first, limit, uint(arguments.form));
        if (begin < limit) {
            end = sz_utf8_norm_cut_metal_(text, arguments.length, limit, arguments.length, uint(arguments.form));
            bytes = sz_utf8_norm_span_metal_(text, arguments.length, begin, end, uint(arguments.form), nullptr);
        }
        else begin = end;
    }
    ulong const offset = sz_utf8_norm_scan_metal_(bytes, lane, totals);
    if (index < arguments.chunks) spans[index] = {begin, end, offset};
    if (lane == 31) blocks[group] = offset + bytes;
}

kernel void sz_utf8_norm_offsets_metal_kernel_(device ulong *target_length [[buffer(2)]],
                                               device ulong *blocks [[buffer(4)]],
                                               constant sz_utf8_norm_arguments_metal_t &arguments [[buffer(5)]],
                                               uint lane [[thread_index_in_threadgroup]]) {
    threadgroup ulong totals[8];
    ulong const count = sz_size_divide_round_up_metal_<ulong>(arguments.chunks, 32);
    ulong const chunk = sz_size_divide_round_up_metal_<ulong>(count, 256);
    ulong const begin = min(ulong(lane) * chunk, count), end = min(begin + chunk, count);
    ulong sum = 0;
    for (ulong index = begin; index < end; ++index) {
        ulong const bytes = blocks[index];
        blocks[index] = sum;
        sum += bytes;
    }
    ulong const offset = sz_utf8_norm_scan_metal_(sum, lane, totals);
    for (ulong index = begin; index < end; ++index) blocks[index] += offset;
    if (lane == 255) *target_length = offset + sum;
}

kernel void sz_utf8_norm_write_metal_kernel_(device uchar const *text [[buffer(0)]], device uchar *target [[buffer(1)]],
                                             device sz_utf8_norm_span_metal_t const *spans [[buffer(3)]],
                                             device ulong const *blocks [[buffer(4)]],
                                             constant sz_utf8_norm_arguments_metal_t &arguments [[buffer(5)]],
                                             uint group [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_threadgroup]]) {
    ulong const index = ulong(group) * 32 + lane;
    if (index >= arguments.chunks) return;
    sz_utf8_norm_span_metal_t const span = spans[index];
    sz_utf8_norm_span_metal_(text, arguments.length, span.begin, span.end, uint(arguments.form),
                             target + blocks[group] + span.offset);
}
