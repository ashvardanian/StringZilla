/**
 *  @file include/stringzilla/levenshtein/metal.metal
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Levenshtein distances on Apple GPUs of Metal family 7, M1 and newer: Myers' bit-parallel
 *      recurrence over a batch of prepared queries, one candidate per thread for a narrow query and
 *      one per simdgroup for a wide one.
 *
 *  @sa include/stringzilla/levenshtein/metal.h, which embeds and launches this source
 *  @sa include/stringzilla/levenshtein/simt.cuh, the CUDA sibling this mirrors step for step
 *  @sa include/stringzilla/types.metal, the prelude compiled ahead of this source
 *
 *  MSL cannot include the C headers, whose pointers carry no address space, so the three helpers
 *  this needs - the engine's row, the rune decoder and the Myers step - are mirrored here, and the
 *  tests hold every distance equal to serial's.
 *
 *  The engine's planes are host addresses inside its block, turned into the kernel's own by
 *  @ref sz_reach_metal_ against the block's host address, while the candidates' tape and the
 *  distances are bound directly. A word count is a template argument, as it is a literal on CUDA,
 *  so the verticals stay in registers; the host instantiates one entry point per word count a
 *  batch reaches.
 */

/** Unicode as 256-rune pages, as @c sz_levenshtein_utf8_pages_k. */
constant uint sz_levenshtein_utf8_pages_metal_k = 0x110000 / 256;

/** Byte values, each owning a row of a byte query's class map. */
constant uint sz_levenshtein_byte_classes_metal_k = 256;

/** Words a class row is padded to, as @c sz_levenshtein_words_stride_k. */
constant ulong sz_levenshtein_words_stride_metal_k = 32;

/** Lanes a warped candidate is spread across, which is a simdgroup, as
 *  @c sz_levenshtein_gpu_warp_lanes_k. */
constant uint sz_levenshtein_gpu_warp_lanes_metal_k = 32;

/** The launch record the host copies in: host addresses inside the engine's block, and the round's
 *  shape. */
struct sz_levenshtein_arguments_metal_t {
    ulong engine_host;
    ulong masks, masks_offsets, symbol_to_class, lengths, order;
    ulong candidates_count, distances_stride;
};

/** One prepared query, as @c sz_levenshtein_engine_row_ materializes it. */
struct sz_levenshtein_query_metal_t {
    device ulong const *masks;
    device uchar const *byte_to_class;
    device ushort const *page_rows;
    device uint const *class_rows;
    ulong stride, length;
};

/** Query @p index of the batch, its class map read as bytes or as a rune page table. */
template <bool runes_>
inline sz_levenshtein_query_metal_t sz_levenshtein_row_metal_(device uchar *engine,
                                                              constant sz_levenshtein_arguments_metal_t &arguments,
                                                              ulong index) {
    ulong const engine_host = arguments.engine_host;
    device ulong const *offsets = sz_reach_metal_<ulong>(engine, engine_host, arguments.masks_offsets);
    device uint const *lengths = sz_reach_metal_<uint>(engine, engine_host, arguments.lengths);
    device uchar const *classes = sz_reach_metal_<uchar>(engine, engine_host, arguments.symbol_to_class);
    sz_levenshtein_query_metal_t query;
    query.masks = sz_reach_metal_<ulong>(engine, engine_host, arguments.masks) + offsets[index];
    query.length = lengths[index];
    query.stride = ((query.length + 63) / 64 + sz_levenshtein_words_stride_metal_k - 1) &
                   ~(sz_levenshtein_words_stride_metal_k - 1);
    query.byte_to_class = classes + index * sz_levenshtein_byte_classes_metal_k;
    if (runes_) {
        // The rune alphabet keeps one page table per query, its offset heading the class map.
        device ulong const *pages_offsets = (device ulong const *)classes;
        query.page_rows = (device ushort const *)(classes + pages_offsets[index]);
        query.class_rows = (device uint const *)(query.page_rows + sz_levenshtein_utf8_pages_metal_k);
    }
    return query;
}

/** The rune at @p position, advancing it, as @c sz_utf8_next_rune_ decodes: one @c U+FFFD per
 *  ill-formed byte. */
inline uint sz_levenshtein_next_rune_metal_(device uchar const *text, ulong length, thread ulong &position) {
    ulong const available = length - position;
    uint const lead = text[position];
    uint rune = 0xFFFD;
    ulong consumed = 1;
    if (lead < 0x80) rune = lead;
    else if (lead >= 0xC2 && lead <= 0xF4) {
        ulong const needed = lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
        if (available >= needed) {
            uint const second = text[position + 1];
            bool formed = (second & 0xC0) == 0x80;
            if (lead == 0xE0) formed = formed && second >= 0xA0;
            if (lead == 0xED) formed = formed && second < 0xA0;
            if (lead == 0xF0) formed = formed && second >= 0x90;
            if (lead == 0xF4) formed = formed && second < 0x90;
            for (ulong index = 2; index < needed; ++index) formed = formed && (text[position + index] & 0xC0) == 0x80;
            if (formed) {
                if (needed == 2) rune = (lead & 0x1F) << 6 | (second & 0x3F);
                else if (needed == 3) rune = (lead & 0x0F) << 12 | (second & 0x3F) << 6 | (text[position + 2] & 0x3F);
                else
                    rune = (lead & 0x07) << 18 | (second & 0x3F) << 12 | (text[position + 2] & 0x3F) << 6 |
                           (text[position + 3] & 0x3F);
                consumed = needed;
            }
        }
    }
    position += consumed;
    return rune;
}

/** The class of @p rune under a rune query, zero for a rune the query lacks. */
inline ulong sz_levenshtein_rune_class_metal_(thread sz_levenshtein_query_metal_t const &query, uint rune) {
    return query.class_rows[(ulong)query.page_rows[rune >> 8] * 256 + (rune & 255)];
}

/** One Myers step over @c words_ verticals against the @p masks row of the stepped symbol's class,
 *  the score moving on the last word, as @c sz_levenshtein_u64x1_step_serial steps. */
template <uint words_>
inline void sz_levenshtein_step_metal_(thread ulong *positive, thread ulong *negative, thread ulong &score,
                                       device ulong const *masks, ulong last_symbol_bit) {
    ulong positive_carry = 1, negative_carry = 0;
#pragma unroll
    for (uint word = 0; word != words_; ++word) {
        ulong const equality = masks[word];
        ulong const vertical_carry = equality | negative[word];
        ulong const matched = equality | negative_carry;
        ulong const diagonal = (((matched & positive[word]) + positive[word]) ^ positive[word]) | matched;
        ulong horizontal_positive = negative[word] | ~(diagonal | positive[word]);
        ulong horizontal_negative = positive[word] & diagonal;
        if (word + 1 == words_) {
            score += (horizontal_positive & last_symbol_bit) != 0;
            score -= (horizontal_negative & last_symbol_bit) != 0;
        }
        ulong const next_positive_carry = horizontal_positive >> 63;
        ulong const next_negative_carry = horizontal_negative >> 63;
        horizontal_positive = (horizontal_positive << 1) | positive_carry;
        horizontal_negative = (horizontal_negative << 1) | negative_carry;
        positive_carry = next_positive_carry, negative_carry = next_negative_carry;
        positive[word] = horizontal_negative | ~(vertical_carry | horizontal_positive);
        negative[word] = horizontal_positive & vertical_carry;
    }
}

/** One candidate per thread against one query per grid row, the query's @c words_ verticals in
 *  the thread's registers, as @c sz_levenshtein_sweep_simt_ runs on CUDA. */
template <uint words_, bool runes_>
kernel void sz_levenshtein_threaded_metal_kernel_(device uchar *engine [[buffer(0)]],
                                                  device ulong const *candidates [[buffer(1)]],
                                                  device ulong *distances [[buffer(2)]],
                                                  constant sz_levenshtein_arguments_metal_t &arguments [[buffer(3)]],
                                                  uint2 position [[thread_position_in_grid]]) {
    ulong const candidate = position.x;
    if (candidate >= arguments.candidates_count) return;
    device uint const *order = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.order);
    ulong const query_index = order[position.y];
    sz_levenshtein_query_metal_t const query = sz_levenshtein_row_metal_<runes_>(engine, arguments, query_index);
    sz_sequence_tape_metal_t const tape = {candidates};
    device uchar const *text = sz_sequence_tape_start_metal_(tape, candidate);
    ulong const text_length = sz_sequence_tape_length_metal_(tape, candidate);

    ulong positive[words_], negative[words_];
    for (uint word = 0; word != words_; ++word) positive[word] = ~0ul, negative[word] = 0;
    ulong score = query.length;
    ulong const last_symbol_bit = 1ul << ((query.length - 1) & 63);
    if (runes_) {
        for (ulong cursor = 0; cursor < text_length;) {
            uint const rune = sz_levenshtein_next_rune_metal_(text, text_length, cursor);
            sz_levenshtein_step_metal_<words_>(
                positive, negative, score, query.masks + sz_levenshtein_rune_class_metal_(query, rune) * query.stride,
                last_symbol_bit);
        }
    }
    else {
        for (ulong cursor = 0; cursor != text_length; ++cursor)
            sz_levenshtein_step_metal_<words_>(positive, negative, score,
                                               query.masks + (ulong)query.byte_to_class[text[cursor]] * query.stride,
                                               last_symbol_bit);
    }
    distances[query_index * arguments.distances_stride + candidate] = score;
}

/**
 *  @brief One candidate per simdgroup, @c words_per_lane_ words to a lane, skewed by one symbol per
 *      lane so the carry between words walks one lane per step, as
 *      @c sz_levenshtein_warp_sweep_simt_ and its rune twin run on CUDA.
 *
 *  Over bytes every lane indexes the candidate itself. Over runes lane zero decodes one rune per
 *  step and hands its class up the lanes beside the carry, the class plus one, so zero marks a step
 *  past the candidate's end and the simdgroup runs while a live lane still holds a rune. Words past
 *  the query's last are stepped rather than branched around, since the recurrence only carries
 *  upward, and the distance is the candidate's length plus the verticals' deltas over live bits.
 */
template <uint words_per_lane_, bool runes_>
kernel void sz_levenshtein_warped_metal_kernel_(
    device uchar *engine [[buffer(0)]], device ulong const *candidates [[buffer(1)]],
    device ulong *distances [[buffer(2)]], constant sz_levenshtein_arguments_metal_t &arguments [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    ulong const candidate = (ulong)group.x * simdgroups + simdgroup;
    // Simdgroup uniform, so the shuffles below still see every lane.
    if (candidate >= arguments.candidates_count) return;
    device uint const *order = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.order);
    ulong const query_index = order[group.y];
    sz_levenshtein_query_metal_t const query = sz_levenshtein_row_metal_<runes_>(engine, arguments, query_index);
    sz_sequence_tape_metal_t const tape = {candidates};
    device uchar const *text = sz_sequence_tape_start_metal_(tape, candidate);
    ulong const text_length = sz_sequence_tape_length_metal_(tape, candidate);
    ulong const words = sz_size_divide_round_up_metal_<ulong>(query.length, 64);
    ulong const live_lanes = sz_size_divide_round_up_metal_<ulong>(words, words_per_lane_);
    ulong const first_word = (ulong)lane * words_per_lane_;
    device ulong const *lane_masks = query.masks + first_word;

    ulong positive[words_per_lane_], negative[words_per_lane_];
    for (uint word = 0; word != words_per_lane_; ++word) positive[word] = ~0ul, negative[word] = 0;

    // What a lane hands the one above: the horizontal positive in bit zero, the negative
    // in bit one.
    uint carry = 0, held = 0;
    ulong cursor = 0, symbols = 0;
    ulong const steps = runes_ ? ~0ul : text_length ? text_length + live_lanes - 1 : 0;
    for (ulong step = 0; step != steps; ++step) {
        uint const received = simd_shuffle_up(carry, 1);
        bool stepping;
        ulong class_id = 0;
        if (runes_) {
            uint decoded = 0;
            if (lane == 0 && cursor < text_length) {
                uint const rune = sz_levenshtein_next_rune_metal_(text, text_length, cursor);
                decoded = (uint)sz_levenshtein_rune_class_metal_(query, rune) + 1, ++symbols;
            }
            uint const inherited = simd_shuffle_up(held, 1);
            held = lane == 0 ? decoded : inherited;
            stepping = held != 0 && lane < live_lanes;
            if (!simd_any(stepping)) break;
            class_id = held - 1;
        }
        else {
            ulong const position = step >= lane ? step - lane : 0;
            stepping = lane < live_lanes && step >= lane && position < text_length;
            // Every lane loads, clamped: behind `stepping`, the load reads zero on all lanes
            // but the first.
            class_id = query.byte_to_class[text[min(position, text_length - 1)]];
        }
        if (!stepping) continue;

        device ulong const *masks = lane_masks + class_id * query.stride;
        // The row above the query's first word is one edit higher than the cell left of it, which
        // is lane zero's carry at every symbol rather than anything a shuffle brings.
        ulong positive_carry = lane == 0 ? 1 : (ulong)(received & 1u);
        ulong negative_carry = lane == 0 ? 0 : (ulong)((received >> 1) & 1u);
#pragma unroll
        for (uint word = 0; word != words_per_lane_; ++word) {
            ulong const equality = masks[word];
            ulong const vertical_carry = equality | negative[word];
            ulong const matched = equality | negative_carry;
            ulong const diagonal = (((matched & positive[word]) + positive[word]) ^ positive[word]) | matched;
            ulong horizontal_positive = negative[word] | ~(diagonal | positive[word]);
            ulong horizontal_negative = positive[word] & diagonal;
            ulong const next_positive_carry = horizontal_positive >> 63;
            ulong const next_negative_carry = horizontal_negative >> 63;
            horizontal_positive = (horizontal_positive << 1) | positive_carry;
            horizontal_negative = (horizontal_negative << 1) | negative_carry;
            positive_carry = next_positive_carry, negative_carry = next_negative_carry;
            positive[word] = horizontal_negative | ~(vertical_carry | horizontal_positive);
            negative[word] = horizontal_positive & vertical_carry;
        }
        carry = (uint)positive_carry | ((uint)negative_carry << 1);
    }

    ulong const last_symbol_bit = 1ul << ((query.length - 1) & 63);
    ulong const last_word_live = last_symbol_bit | (last_symbol_bit - 1);
    int deltas = 0;
    for (uint word = 0; word != words_per_lane_; ++word) {
        ulong const index = first_word + word;
        ulong const live = index + 1 < words ? ~0ul : index + 1 == words ? last_word_live : 0;
        deltas += (int)popcount(positive[word] & live) - (int)popcount(negative[word] & live);
    }
    deltas = simd_sum(deltas);
    ulong const length = runes_ ? symbols : text_length;
    if (lane == 0) distances[query_index * arguments.distances_stride + candidate] = (ulong)((long)length + deltas);
}

/*  One threaded entry point per word count below the warped rung, over bytes then runes, named as
 *  the host's tables spell it. */
template [[host_name("sz_levenshtein_distances_w1_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<1, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w2_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<2, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w3_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<3, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w4_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<4, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w5_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<5, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w6_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<6, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w7_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<7, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w8_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<8, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w9_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<9, false>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w10_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<10, false>(device uchar *, device ulong const *, device ulong *,
                                                 constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w11_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<11, false>(device uchar *, device ulong const *, device ulong *,
                                                 constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w12_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<12, false>(device uchar *, device ulong const *, device ulong *,
                                                 constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w13_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<13, false>(device uchar *, device ulong const *, device ulong *,
                                                 constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w14_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<14, false>(device uchar *, device ulong const *, device ulong *,
                                                 constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_w15_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<15, false>(device uchar *, device ulong const *, device ulong *,
                                                 constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w1_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<1, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w2_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<2, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w3_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<3, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w4_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<4, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w5_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<5, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w6_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<6, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w7_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<7, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w8_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<8, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w9_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<9, true>(device uchar *, device ulong const *, device ulong *,
                                               constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w10_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<10, true>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w11_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<11, true>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w12_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<12, true>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w13_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<13, true>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w14_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<14, true>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);
template [[host_name("sz_levenshtein_distances_utf8_w15_metal_kernel_")]] kernel void
sz_levenshtein_threaded_metal_kernel_<15, true>(device uchar *, device ulong const *, device ulong *,
                                                constant sz_levenshtein_arguments_metal_t &, uint2);

/*  One warped entry point per words-per-lane above the crossing, over bytes then runes, named as
 *  the host's tables spell it. */
template [[host_name("sz_levenshtein_distances_k1_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<1, false>(device uchar *, device ulong const *, device ulong *,
                                              constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_k2_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<2, false>(device uchar *, device ulong const *, device ulong *,
                                              constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_k3_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<3, false>(device uchar *, device ulong const *, device ulong *,
                                              constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_k4_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<4, false>(device uchar *, device ulong const *, device ulong *,
                                              constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_k5_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<5, false>(device uchar *, device ulong const *, device ulong *,
                                              constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_k6_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<6, false>(device uchar *, device ulong const *, device ulong *,
                                              constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_k7_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<7, false>(device uchar *, device ulong const *, device ulong *,
                                              constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_k8_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<8, false>(device uchar *, device ulong const *, device ulong *,
                                              constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_utf8_k1_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<1, true>(device uchar *, device ulong const *, device ulong *,
                                             constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_utf8_k2_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<2, true>(device uchar *, device ulong const *, device ulong *,
                                             constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_utf8_k3_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<3, true>(device uchar *, device ulong const *, device ulong *,
                                             constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_utf8_k4_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<4, true>(device uchar *, device ulong const *, device ulong *,
                                             constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_utf8_k5_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<5, true>(device uchar *, device ulong const *, device ulong *,
                                             constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_utf8_k6_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<6, true>(device uchar *, device ulong const *, device ulong *,
                                             constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_utf8_k7_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<7, true>(device uchar *, device ulong const *, device ulong *,
                                             constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
template [[host_name("sz_levenshtein_distances_utf8_k8_metal_kernel_")]] kernel void
sz_levenshtein_warped_metal_kernel_<8, true>(device uchar *, device ulong const *, device ulong *,
                                             constant sz_levenshtein_arguments_metal_t &, uint2, uint, uint, uint);
