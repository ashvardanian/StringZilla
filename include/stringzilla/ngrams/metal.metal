/**
 *  @file include/stringzilla/ngrams/metal.metal
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief N-grams on Apple GPUs of Metal family 7, M1 and newer: the N-gram residue's roll, the
 *      overlap walk of a simdgroup per candidate over one prepared query's table, and the sketch of
 *      a simdgroup per text over ranks in threadgroup memory.
 *
 *  Nothing here needs more than Apple7: 32-bit atomics, simdgroup votes and sums and 32 KB of
 *  threadgroup memory. A later family that earns its own kernels gets its own file beside this one.
 *
 *  @sa include/stringzilla/ngrams/metal.h, which embeds and launches this source
 *  @sa include/stringzilla/ngrams/simt.cuh, the CUDA sibling this mirrors step for step
 *  @sa include/stringzilla/types.metal, the prelude compiled ahead of this source
 *
 *  The CUDA kernels reach the serial helpers through @c --expt-relaxed-constexpr; MSL cannot
 *  include the C headers, whose pointers carry no address space, so the slot layout, the probe and
 *  the sketch's mixer are mirrored here, and the tests hold every count and sketch to serial's.
 *
 *  Apple GPUs have no FP64 for the host tiers' @c f64 chain and no 64-bit divider either, so a
 *  remainder over @c ulong compiles to a long emulated sequence. Every residue is reduced with
 *  Montgomery products instead, which need only the native 32-bit @c mul and @c mulhi, and every
 *  walk rolls its residues a byte at a time.
 *
 *  @see https://ashvardanian.com/posts/stringwars-on-gpus/#52-bit-math-for-bioinformatics
 *
 *  The tables point inside their block by host address, turned into the kernel's own by
 *  @ref sz_reach_metal_ against the block's host address, while the texts' tape, the shares and
 *  the sketches are bound directly. A round is one dispatch.
 */

/** The 32-bit prime every N-gram residue is reduced by, as @c sz_ngrams_modulus_k. */
constant uint sz_ngrams_modulus_metal_k = 4026525731u;

/** −p⁻¹ mod 2³², R² mod p and 256 × R mod p, for Montgomery products with R = 2³². */
constant uint sz_ngrams_modulus_inverse_metal_k = 574879861u;
constant uint sz_ngrams_montgomery_square_metal_k = 3066829947u;
constant uint sz_ngrams_byte_shift_metal_k = 270103213u;

/** A slot's layout, as the @c sz_ngrams_slot_*_k constants lay it: the residue in the high word,
 *  then the window's first byte in 24 bits, its width's index in 6, then the repeat bit, and last
 *  the occupied bit. */
constant uint sz_ngrams_slot_first_shift_metal_k = 8, sz_ngrams_slot_width_shift_metal_k = 2;
constant uint sz_ngrams_slot_repeat_metal_k = 2, sz_ngrams_slot_occupied_metal_k = 1;

/** The bits two slots of one window share, as @c sz_ngrams_slot_key_mask_k. */
constant ulong sz_ngrams_slot_key_mask_metal_k = 0xFFFFFFFF000000FDul;

/** Widths one overlap walk rolls together, their residues and counters held per lane; an engine
 *  with more walks each stretch of text once per group of them. */
constant uint sz_ngrams_widths_per_walk_metal_k = 8;

/** Widths one engine holds at most, as @c sz_ngrams_widths_max_k. */
constant uint sz_ngrams_widths_max_metal_k = 64;

/** The overlaps launch record the host copies in: host addresses inside the tables' block, and the
 *  round's shape. */
struct sz_ngrams_overlaps_arguments_metal_t {
    ulong engine_host;
    ulong slots, slots_offsets, filters, distinct_counts, widths, powers, lengths;
    ulong queries_count, widths_count, candidates_count;
    ulong shares_query_stride, shares_candidate_stride, staged_slots_count, hit_words_per_simdgroup, candidates_layout,
        misses_bound;
};

/** The sketches launch record, laid out as the host's record of the same name. */
struct sz_ngrams_sketches_arguments_metal_t {
    ulong texts_count, slots, sketches_stride, texts_layout, widths_count;
    uint widths[sz_ngrams_widths_max_metal_k];
};

#pragma region Residues

/**
 *  @brief a × b ÷ R mod p for a, b < p: with @p b in Montgomery form, as b × R mod p, a × b mod p.
 *
 *  The low halves of a × b and m × p sum to zero mod 2³² by the choice of m, so they carry exactly
 *  when the low half of a × b is nonzero, and the high halves plus that carry stay under 2p.
 */
inline uint sz_ngrams_montgomery_metal_(uint a, uint b) {
    uint const low = a * b, high = mulhi(a, b);
    uint const correction = mulhi(low * sz_ngrams_modulus_inverse_metal_k, sz_ngrams_modulus_metal_k);
    ulong const sum = (ulong)high + correction + (low != 0);
    return (uint)(sum >= sz_ngrams_modulus_metal_k ? sum - sz_ngrams_modulus_metal_k : sum);
}

/** @p power, 256 to a width mod p, lifted into Montgomery form, so its product with a byte comes
 *  back plain. */
inline uint sz_ngrams_lift_metal_(uint power) {
    return sz_ngrams_montgomery_metal_(power, sz_ngrams_montgomery_square_metal_k);
}

/** One byte's step of a residue, <tt>H' = H * 256 + incoming - outgoing * 256^w</tt>, with
 *  @p lifted_power being 256^w in Montgomery form. */
inline uint sz_ngrams_roll_lifted_metal_(uint residue, uint incoming, uint outgoing, uint lifted_power) {
    // Under p + 256, which still fits 32 bits.
    uint const shifted = sz_ngrams_montgomery_metal_(residue, sz_ngrams_byte_shift_metal_k) + incoming;
    uint const leaving = sz_ngrams_montgomery_metal_(outgoing, lifted_power);
    // Wrapping is exact here: the true difference is under p + 256, below 2³².
    uint rolled = shifted - leaving + (shifted < leaving ? sz_ngrams_modulus_metal_k : 0u);
    if (rolled >= sz_ngrams_modulus_metal_k) rolled -= sz_ngrams_modulus_metal_k;
    return rolled;
}

/**
 *  @brief The residue of the window one byte on from the window of @p residue : @p outgoing leaves,
 *      the rest shifts a byte up and @p incoming enters, as @c sz_ngrams_roll_simt rolls it.
 *
 *  A window being primed rolls in its bytes with a zero @p outgoing, from a zero residue. The lift
 *  of @p power depends on it alone, so a loop rolling one width lifts it once.
 *
 *  @param[in] power 256 to the window's width, mod p, as @c sz_ngrams_power.
 */
inline uint sz_ngrams_roll_metal(uint residue, uint incoming, uint outgoing, uint power) {
    return sz_ngrams_roll_lifted_metal_(residue, incoming, outgoing, sz_ngrams_lift_metal_(power));
}

/** 256 to @p width, mod p, as @c sz_ngrams_power: by squaring in Montgomery form, from one's
 *  form, R mod p. */
inline uint sz_ngrams_power_metal_(uint width) {
    uint power = sz_ngrams_montgomery_metal_(1, sz_ngrams_montgomery_square_metal_k);
    uint base = sz_ngrams_byte_shift_metal_k;
    for (uint exponent = width; exponent; exponent >>= 1) {
        if (exponent & 1) power = sz_ngrams_montgomery_metal_(power, base);
        base = sz_ngrams_montgomery_metal_(base, base);
    }
    return sz_ngrams_montgomery_metal_(power, 1);
}

/** N-grams a text of @p length bytes holds, summed over @p count @p widths, as
 *  @c sz_ngrams_count counts them: a width of zero or past the text holds none. */
inline ulong sz_ngrams_count_metal(ulong length, device ulong const *widths, ulong count) {
    ulong ngrams = 0;
    for (ulong index = 0; index != count; ++index)
        if (widths[index] && widths[index] <= length) ngrams += length - widths[index] + 1;
    return ngrams;
}

#pragma endregion Residues

#pragma region Prepared Queries

/** The bit of a table's miss filter that a window of @p residue sets, in a table of @p mask plus
 *  one slots, as @c sz_ngrams_filter_bit_; its bits past the low three name the probe's slot. */
inline uint sz_ngrams_filter_bit_metal_(uint residue, uint mask) {
    return (uint)(((ulong)(residue * 2654435761u) * ((ulong)mask + 1)) >> 29);
}

/** The slot of @p slots_count @p slots holding @p residue at width @p width_index, or zero when the
 *  query lacks that window, the slots in either memory space and the @p filter on the device. */
template <typename slots_type_>
inline ulong sz_ngrams_slot_probe_metal_(slots_type_ slots, device uint const *filter, uint slots_count, uint residue,
                                         uint width_index) {
    ulong const wanted = ((ulong)residue << 32) | (width_index << sz_ngrams_slot_width_shift_metal_k) |
                         sz_ngrams_slot_occupied_metal_k;
    uint const mask = slots_count - 1, bit = sz_ngrams_filter_bit_metal_(residue, mask);
    if (!((filter[bit / 32] >> (bit % 32)) & 1)) return 0;
    for (uint slot = bit >> 3;; slot = (slot + 1) & mask) {
        ulong const entry = slots[slot];
        if (!entry || ((entry ^ wanted) & sz_ngrams_slot_key_mask_metal_k) == 0) return entry;
    }
}

/** The first query byte of the window a nonzero @p slot holds, as @c sz_ngrams_slot_first. */
inline uint sz_ngrams_slot_first_metal(ulong slot) {
    return (uint)(slot >> sz_ngrams_slot_first_shift_metal_k) & 0xFFFFFFu;
}

/** Whether the query holds the window a nonzero @p slot holds more than once, as
 *  @c sz_ngrams_slot_repeated. */
inline bool sz_ngrams_slot_repeated_metal(ulong slot) { return (slot & sz_ngrams_slot_repeat_metal_k) != 0; }

/** One query prepared where a walk scores it, inside a span its caller owns, as
 *  @c sz_ngrams_query_t lays it out: the table of its distinct N-grams at every width, and what a
 *  candidate's share needs beside it. Trivially copyable, so a walk may broadcast it. */
struct sz_ngrams_query_metal_t {

    /** Slots <tt>residue:32 | first:24 | width:6 | repeat:1 | occupied:1</tt>, zero empty, at the
     *  head of the span. */
    device ulong const *slots;

    /** A power of two, at least twice @c ngrams and at least 64. */
    ulong slots_count;

    /** The miss filter after the slots, eight bits a slot, as @c sz_ngrams_query_t keeps it. */
    device uint const *filter;

    /** The @b [widths_count] window widths, in bytes, read where the caller keeps them. */
    device ulong const *widths;

    /** The @b [widths_count] powers of 256 at each width, mod p, at the tail of the span. */
    device uint const *powers;

    /** Entries in @c widths and @c powers. */
    ulong widths_count;

    /** The query's N-grams summed over every width, repeats included, which is the number of bits
     *  a candidate's hits span. */
    ulong ngrams;

    /** The query's distinct N-grams, the share's denominator. */
    ulong distinct;

    /** The query's bytes, so a width past them probes nothing. */
    ulong length;
};

/** The slot of @p query holding the window of @p residue at width @p width_index, or zero when the
 *  query lacks that window, as @c sz_ngrams_query_probe finds it. */
inline ulong sz_ngrams_query_probe_metal(thread sz_ngrams_query_metal_t const &query, uint residue, ulong width_index) {
    return sz_ngrams_slot_probe_metal_(query.slots, query.filter, (uint)query.slots_count, residue, (uint)width_index);
}

/** Whether a lane has a window to file into the table. */
enum sz_ngrams_lane_metal_t { sz_ngrams_lane_idle_metal_k = 0, sz_ngrams_lane_filing_metal_k = 1 };

/**
 *  @brief Files each lane's window into the table over @p halves, every lane of the simdgroup
 *      taking part, those without a window included.
 *
 *  Apple7 has no 64-bit compare-and-swap, so a slot is claimed through its low word and its residue
 *  stored after, and the lanes go in rounds split by barriers: every lane reads its slot, then a
 *  lane that read it empty claims it, one that read its own window marks it repeated keeping the
 *  smaller first byte, and any other moves on. A read never meets a half-written slot, so the table
 *  holds each window once, with the host's first byte and repeat bit, in any order of the lanes.
 *
 *  @param[in,out] halves The slots as 32-bit words, low word first.
 *  @param[in,out] filter The table's miss filter, which gets each filed window's bit.
 *  @param[in] state Whether this lane has a window to file.
 *  @return One when this lane's window was new to the table.
 */
inline uint sz_ngrams_slot_file_metal_(device atomic_uint *halves, device atomic_uint *filter, uint slots_count,
                                       uint residue, uint width_index, uint first, sz_ngrams_lane_metal_t state,
                                       sz_group_metal_t group) {
    bool filing = state == sz_ngrams_lane_filing_metal_k;
    uint const low = (first << sz_ngrams_slot_first_shift_metal_k) |
                     (width_index << sz_ngrams_slot_width_shift_metal_k) | sz_ngrams_slot_occupied_metal_k;
    uint const key_mask = (uint)sz_ngrams_slot_key_mask_metal_k, mask = slots_count - 1;
    uint const bit = sz_ngrams_filter_bit_metal_(residue, mask);
    if (filing) atomic_fetch_or_explicit(filter + bit / 32, 1u << (bit % 32), memory_order_relaxed);
    uint slot = bit >> 3, claimed = 0;
    while (group.any(filing)) {
        uint const seen_low = filing ? atomic_load_explicit(halves + 2 * slot, memory_order_relaxed) : 0u;
        uint const seen_residue = filing ? atomic_load_explicit(halves + 2 * slot + 1, memory_order_relaxed) : 0u;
        group.sync();
        if (filing && !seen_low) {
            // A lane losing the claim to another rereads the slot, whole by then, next round.
            uint expected = 0;
            if (atomic_compare_exchange_weak_explicit(halves + 2 * slot, &expected, low, memory_order_relaxed,
                                                      memory_order_relaxed)) {
                atomic_store_explicit(halves + 2 * slot + 1, residue, memory_order_relaxed);
                claimed = 1, filing = false;
            }
        }
        else if (filing && seen_residue == residue && ((seen_low ^ low) & key_mask) == 0) {
            atomic_fetch_min_explicit(halves + 2 * slot, low | sz_ngrams_slot_repeat_metal_k, memory_order_relaxed);
            atomic_fetch_or_explicit(halves + 2 * slot, sz_ngrams_slot_repeat_metal_k, memory_order_relaxed);
            filing = false;
        }
        else if (filing) slot = (slot + 1) & mask;
        group.sync();
    }
    return claimed;
}

/**
 *  @brief Prepares one query into a simdgroup's @p span, every lane of the simdgroup taking part,
 *      as @c sz_ngrams_query_prepare_simt prepares it.
 *
 *  Each lane rolls its own run of every width's N-grams, from the same residues the serial tier
 *  computes, and the simdgroup files them a window per lane at a time through
 *  @ref sz_ngrams_slot_file_metal_. The simdgroup is synchronized on entry, as the span may still
 *  serve the previous query, and on return.
 *
 *  @param[in] text The query, read by plain loads, at most @c sz_ngrams_query_bytes_max_k bytes.
 *  @param[in] widths The @b [widths_count] window widths, kept where every walk reads them, at most
 *      @c sz_ngrams_widths_max_k of them.
 *  @param[out] span Eight-byte aligned, @c sz_ngrams_query_span_bytes long: the slots, their
 *      miss filter, then a power per width.
 *  @param[out] query Points into @p span and at @p widths, complete in every lane.
 *  @param[in] lane The calling lane, below 32.
 */
inline void sz_ngrams_query_prepare_metal(device uchar const *text, ulong length, device ulong const *widths,
                                          ulong widths_count, device uchar *span, thread sz_ngrams_query_metal_t &query,
                                          uint lane) {
    uint const lanes = 32;
    sz_group_metal_t const group = {lanes, lane};
    ulong const ngrams = sz_ngrams_count_metal(length, widths, widths_count);
    ulong slots_count = 64;
    while (slots_count < 2 * ngrams) slots_count *= 2;
    device ulong *const slots = (device ulong *)span;
    device atomic_uint *const halves = (device atomic_uint *)span;
    device atomic_uint *const filter = (device atomic_uint *)(slots + slots_count);
    device uint *const powers = (device uint *)(slots + slots_count) + slots_count / 4;
    group.sync();
    // The slots' halves and the filter's words lie back to back.
    for (ulong word = lane; word < 2 * slots_count + slots_count / 4; word += lanes)
        atomic_store_explicit(halves + word, 0u, memory_order_relaxed);
    for (ulong index = lane; index < widths_count; index += lanes)
        powers[index] = sz_ngrams_power_metal_((uint)widths[index]);
    group.sync();

    // Each lane rolls a run of one width's N-grams, priming its residue on the run's first window,
    // and every lane steps as many endings as the first, so the filing rounds meet at each one.
    uint distinct = 0;
    for (ulong index = 0; index != widths_count; ++index) {
        ulong const width = widths[index];
        if (!width || width > length) continue;
        uint const lifted_power = sz_ngrams_lift_metal_(powers[index]);
        ulong const width_ngrams = length - width + 1;
        ulong const run = sz_size_divide_round_up_metal_<ulong>(width_ngrams, lanes);
        ulong const first = min((ulong)lane * run, width_ngrams), last = min(first + run, width_ngrams);
        ulong const endings = first != last ? last - first + width - 1 : 0;
        uint residue = 0;
        for (ulong ending = 1; ending < run + width; ++ending) {
            bool const rolled = ending <= endings;
            if (rolled) {
                uint const outgoing = ending > width ? text[first + ending - width - 1] : 0u;
                residue = sz_ngrams_roll_lifted_metal_(residue, text[first + ending - 1], outgoing, lifted_power);
            }
            distinct += sz_ngrams_slot_file_metal_(
                halves, filter, (uint)slots_count, residue, (uint)index, (uint)(first + ending - width),
                rolled && ending >= width ? sz_ngrams_lane_filing_metal_k : sz_ngrams_lane_idle_metal_k, group);
        }
    }
    distinct = simd_sum(distinct);
    group.sync();
    query.slots = slots;
    query.slots_count = slots_count;
    query.filter = (device uint const *)filter;
    query.widths = widths;
    query.powers = powers;
    query.widths_count = widths_count;
    query.ngrams = ngrams;
    query.distinct = distinct;
    query.length = length;
}

#pragma endregion Prepared Queries

#pragma region Overlaps Walk

/** Whether @p predicate holds on any lane of this lane's aligned group of @p lanes, read as
 *  nonzero, which other groups of the simdgroup may have left: the votes meet by halves, never past
 *  the group, as @c sz_group_any_cuda_ votes. */
inline bool sz_ngrams_group_any_metal_(int predicate, uint lanes) {
    if (lanes == 32) return simd_any(predicate != 0);
    uint votes = predicate != 0 ? 1u : 0u;
    for (uint distance = 1; distance < lanes; distance *= 2) votes |= simd_shuffle_xor(votes, ushort(distance));
    return votes != 0;
}

/** The sum of @p value over this lane's aligned group of @p lanes, in every lane of the group, as
 *  @c sz_ngrams_lanes_sum_simt_ sums it: in one step for the simdgroup, by halves otherwise. */
inline uint sz_ngrams_lanes_sum_metal_(uint value, uint lanes) {
    if (lanes == 32) return simd_sum(value);
    for (uint distance = lanes / 2; distance != 0; distance >>= 1) value += simd_shuffle_xor(value, ushort(distance));
    return value;
}

/**
 *  @brief Rolls each width's residue through the @p walk_bytes endings of @p walk, on one lane, and
 *      sets the hit bit of every window the query holds, counting per width the bits it set first.
 *
 *  Each width rolls its own residue a byte at a time, the byte leaving the window read again from
 *  the text rather than kept, so the residues stay in registers. The first @p warm_bytes endings
 *  only prime the residues, and a window not yet full is never probed.
 *
 *  @param[in] first_width The index of the walk's first width among the query's.
 *  @param[in] widths One entry per width of the walk, zero for one past the query.
 *  @param[in] lifted_powers One per width, 256 to the width lifted into Montgomery form.
 *  @param[in] bases Each width's first bit among the hit words, which is the query's N-grams at
 *      the widths before it.
 *  @param[in,out] residues One entry per width, up to @c sz_ngrams_widths_per_walk_metal_k.
 *  @param[in,out] counts One entry per width, up to @c sz_ngrams_widths_per_walk_metal_k.
 */
template <typename slots_type_, typename hit_words_type_>
inline void sz_ngrams_roll_walk_metal_(device uchar const *walk, ulong walk_bytes, ulong warm_bytes, slots_type_ slots,
                                       device uint const *filter, uint slots_count, uint first_width,
                                       thread uint const *widths, thread uint const *lifted_powers,
                                       thread uint const *bases, uint widths_count, hit_words_type_ hit_words,
                                       thread uint *residues, thread uint *counts) {
    for (ulong ending = 1; ending <= walk_bytes; ++ending) {
        uint const incoming = walk[ending - 1];
#pragma unroll
        for (uint index = 0; index != sz_ngrams_widths_per_walk_metal_k; ++index) {
            if (index >= widths_count) break;
            ulong const width = widths[index];
            if (!width) continue;
            uint const outgoing = ending > width ? walk[ending - width - 1] : 0u;
            residues[index] = sz_ngrams_roll_lifted_metal_(residues[index], incoming, outgoing, lifted_powers[index]);
            if (ending <= warm_bytes || width > ending) continue;
            ulong const entry = sz_ngrams_slot_probe_metal_(slots, filter, slots_count, residues[index],
                                                            first_width + index);
            if (!entry) continue;
            uint const bit = bases[index] + sz_ngrams_slot_first_metal(entry);
            uint const mask = 1u << (bit & 31);
            counts[index] += (atomic_fetch_or_explicit(hit_words + (bit >> 5), mask, memory_order_relaxed) & mask) == 0;
        }
    }
}

/**
 *  @brief Counts the distinct N-grams of one prepared query that one candidate holds, at every
 *      width, every lane of an aligned group of @p lanes taking part, as
 *      @c sz_ngrams_candidate_walk_simt_ counts them.
 *
 *  Each lane of the group rolls an even run of the candidate's bytes, primed from the widest
 *  window's bytes before it, and a hit sets the bit of the query window's first byte in
 *  @p hit_words, so a window either side repeats counts once. The words go back to zero before the
 *  group returns, ready for its next candidate, and the vote on what the clears read closes the
 *  walk, so no lane starts another candidate before every word is zero.
 *
 *  @param[in] slots The query's table, @c query.slots or a staged copy of it.
 *  @param[in,out] hit_words The group's own <tt>(query.ngrams+31)/32</tt> words, zero on entry
 *      and on return.
 *  @param[in] lane The calling lane within its group, below @p lanes.
 *  @param[in] lanes The group's lanes, one of 1, 2, 4, 8, 16 and 32, aligned within the simdgroup.
 *  @param[in] distinct_counts The query's distinct N-grams at each width, null when @p shares is.
 *  @param[out] shares The pair's @b [widths_count] shares of @p distinct_counts, written by lane
 *      zero, or null when only the count's sum is wanted.
 *  @return The count over every width, in every lane of the group.
 */
template <typename slots_type_, typename hit_words_type_>
inline ulong sz_ngrams_candidate_walk_metal_(thread sz_ngrams_query_metal_t const &query, slots_type_ slots,
                                             device uchar const *text, ulong length, hit_words_type_ hit_words,
                                             uint lane, uint lanes, device uint const *distinct_counts,
                                             device float *shares) {
    uint const per_walk = sz_ngrams_widths_per_walk_metal_k;
    ulong const run = sz_size_divide_round_up_metal_<ulong>(length, (ulong)lanes);
    ulong const run_begin = min((ulong)lane * run, length);
    ulong const run_end = min(run_begin + run, length);
    uint held = 0, base = 0;
    for (uint first_width = 0; first_width < query.widths_count; first_width += per_walk) {
        device ulong const *const group_widths = query.widths + first_width;
        uint const group_count = min((uint)query.widths_count - first_width, per_walk);
        ulong warm_up = 0;
        for (uint index = 0; index != group_count; ++index) warm_up = max(warm_up, (ulong)group_widths[index]);
        warm_up = warm_up ? warm_up - 1 : 0;
        ulong const walk_begin = run_begin > warm_up ? run_begin - warm_up : 0;
        ulong const walk_bytes = run_end != run_begin ? run_end - walk_begin : 0;

        // A width past the query rolls as zero, which probes nothing and holds no bits.
        uint live_widths[sz_ngrams_widths_per_walk_metal_k], lifted_powers[sz_ngrams_widths_per_walk_metal_k];
        uint bases[sz_ngrams_widths_per_walk_metal_k];
        uint residues[sz_ngrams_widths_per_walk_metal_k], found[sz_ngrams_widths_per_walk_metal_k];
#pragma unroll
        for (uint index = 0; index != per_walk; ++index) {
            bool const live = index < group_count && group_widths[index] && group_widths[index] <= query.length;
            live_widths[index] = live ? (uint)group_widths[index] : 0u;
            lifted_powers[index] = live ? sz_ngrams_lift_metal_(query.powers[first_width + index]) : 0u;
            bases[index] = base;
            base += live ? (uint)(query.length - group_widths[index] + 1) : 0u;
            residues[index] = 0, found[index] = 0;
        }
        sz_ngrams_roll_walk_metal_(text + walk_begin, walk_bytes, run_begin - walk_begin, slots, query.filter,
                                   (uint)query.slots_count, first_width, live_widths, lifted_powers, bases, group_count,
                                   hit_words, residues, found);
#pragma unroll
        for (uint index = 0; index != per_walk; ++index) {
            if (index >= group_count) break;
            held += found[index];
            if (!shares) continue;
            uint const total = sz_ngrams_lanes_sum_metal_(found[index], lanes);
            uint const distinct = distinct_counts[first_width + index];
            if (lane == 0)
                shares[first_width + index] = distinct ? precise::divide((float)total, (float)distinct) : 0.0f;
        }
    }

    // Every lane's hits returned before it adds them here, so no word is cleared under a pending
    // hit of another lane.
    held = sz_ngrams_lanes_sum_metal_(held, lanes);
    uint cleared = 0;
    for (ulong word = lane; word < sz_size_divide_round_up_metal_<ulong>(query.ngrams, 32); word += lanes)
        cleared |= atomic_exchange_explicit(hit_words + word, 0u, memory_order_relaxed);
    return sz_ngrams_group_any_metal_(cleared != 0, lanes) ? held : 0;
}

/**
 *  @brief Counts the distinct N-grams of a prepared query that one candidate holds, over every
 *      width, every lane of an aligned group of @p lanes taking part, as
 *      @c sz_ngrams_candidate_matches_simt counts them.
 *
 *  The share the candidate lacks is the query's @c distinct less this count, over @c distinct.
 *
 *  @param[in] query Prepared by @ref sz_ngrams_query_prepare_metal, its span read through plain
 *      loads, so the same kernel may have written it.
 *  @param[in,out] hit_words The group's own <tt>(query.ngrams+31)/32</tt> words, zero on entry
 *      and on return; groups scoring at once need words of their own.
 *  @param[in] lane The calling lane within its group, below @p lanes.
 *  @param[in] lanes The group's lanes, one of 1, 2, 4, 8, 16 and 32, aligned within the simdgroup.
 *  @return The same count in every lane of the group.
 */
inline ulong sz_ngrams_candidate_matches_metal(thread sz_ngrams_query_metal_t const &query, device uchar const *text,
                                               ulong length, device atomic_uint *hit_words, uint lane, uint lanes) {
    return sz_ngrams_candidate_walk_metal_(query, query.slots, text, length, hit_words, lane, lanes,
                                           (device uint const *)nullptr, (device float *)nullptr);
}

/**
 *  @brief One threadgroup per tile of candidates and per prepared query, each simdgroup scoring a
 *      candidate of the tile at a time, the query's table staged when the launch asked for it.
 *
 *  Threadgroup memory holds the staged table, then each simdgroup's hit words, which stay zero
 *  between candidates. A pair is one call of @ref sz_ngrams_candidate_walk_metal_, dropped to
 *  zeros once its misses pass the bound of the query's distinct N-grams, or before a byte is read
 *  when its lengths alone pass it. The barriers around the staging are collective, so no thread
 *  leaves the query loop before the others.
 */
kernel void sz_ngrams_overlaps_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar const *candidates [[buffer(1)]],
    device uchar const *candidates_offsets [[buffer(2)]], device float *shares [[buffer(3)]],
    constant sz_ngrams_overlaps_arguments_metal_t &arguments [[buffer(4)]],
    threadgroup ulong *staged_slots [[threadgroup(0)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint simdgroups [[simdgroups_per_threadgroup]]) {
    sz_strings_tape_metal_t const tape = {candidates, candidates_offsets, arguments.candidates_layout};
    device ulong const *slots = sz_reach_metal_<ulong>(engine, arguments.engine_host, arguments.slots);
    device ulong const *slots_offsets = sz_reach_metal_<ulong>(engine, arguments.engine_host, arguments.slots_offsets);
    device uint const *filters = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.filters);
    device uint const *distinct_counts = sz_reach_metal_<uint>(engine, arguments.engine_host,
                                                               arguments.distinct_counts);
    device ulong const *widths = sz_reach_metal_<ulong>(engine, arguments.engine_host, arguments.widths);
    device uint const *powers = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.powers);
    device uint const *lengths = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.lengths);
    ulong const widths_count = arguments.widths_count;
    threadgroup atomic_uint *const hit_words = (threadgroup atomic_uint *)(staged_slots +
                                                                           arguments.staged_slots_count) +
                                               simdgroup * arguments.hit_words_per_simdgroup;
    for (ulong word = lane; word < arguments.hit_words_per_simdgroup; word += 32)
        atomic_store_explicit(hit_words + word, 0u, memory_order_relaxed);
    ulong const tile_first = (ulong)group.x * group_size.x;
    ulong const tile_count = min((ulong)group_size.x, arguments.candidates_count - tile_first);

    for (ulong query_index = group.y; query_index < arguments.queries_count; query_index += groups.y) {
        device ulong const *const query_slots = slots + slots_offsets[query_index];
        ulong const slots_count = slots_offsets[query_index + 1] - slots_offsets[query_index];
        if (arguments.staged_slots_count)
            for (ulong slot = thread_index; slot < slots_count; slot += group_size.x)
                staged_slots[slot] = query_slots[slot];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sz_ngrams_query_metal_t query;
        query.slots = query_slots, query.slots_count = slots_count;
        query.filter = filters + slots_offsets[query_index] / 4;
        query.widths = widths, query.powers = powers, query.widths_count = widths_count;
        query.length = lengths[query_index];
        query.ngrams = sz_ngrams_count_metal(query.length, widths, widths_count);
        query.distinct = 0;
        for (ulong index = 0; index != widths_count; ++index)
            query.distinct += distinct_counts[query_index * widths_count + index];
        ulong const bound = (arguments.misses_bound * query.distinct) >> 24;
        for (ulong candidate = simdgroup; candidate < tile_count; candidate += simdgroups) {
            device uchar const *const text = sz_strings_start_metal_(tape, tile_first + candidate);
            ulong const length = sz_strings_length_metal_(tape, tile_first + candidate);
            device float *const row = shares + query_index * arguments.shares_query_stride +
                                      (tile_first + candidate) * arguments.shares_candidate_stride;
            device uint const *const query_distinct_counts = distinct_counts + query_index * widths_count;
            ulong reachable = 0, index;
            for (index = 0; index != widths_count; ++index) {
                ulong const width = widths[index];
                reachable += width && width <= length && width <= query.length ? length - width + 1 : 0;
            }
            ulong held = 0;
            bool const reaches = query.distinct - min(query.distinct, reachable) <= bound;
            if (reaches && arguments.staged_slots_count)
                held = sz_ngrams_candidate_walk_metal_(query, (threadgroup ulong const *)staged_slots, text, length,
                                                       hit_words, lane, 32, query_distinct_counts, row);
            else if (reaches)
                held = sz_ngrams_candidate_walk_metal_(query, query_slots, text, length, hit_words, lane, 32,
                                                       query_distinct_counts, row);
            // Lane zero wrote the shares, so lane zero alone overwrites a pair past its budget.
            bool const zeroes = lane == 0 && (!reaches || query.distinct - held > bound);
            for (index = 0; zeroes && index != widths_count; ++index) row[index] = 0.0f;
        }
        // The next query restages the table these walks still read.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#pragma endregion Overlaps Walk

#pragma region Sketches

/** Mixes 32 bits, as @c sz_ngrams_mix_. */
inline uint sz_ngrams_mix_metal_(uint value) {
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    return value ^ (value >> 16);
}

/** Unequal bytes of two words, four slots of two sketches, as @c sz_ngrams_sketch_word_distance. */
inline uint sz_ngrams_sketch_word_distance_metal(uint a_word, uint b_word) {
    uint const differing = a_word ^ b_word;
    uint const raised = (((differing & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | differing) & 0x80808080u;
    return ((raised >> 7) * 0x01010101u) >> 24;
}

/**
 *  @brief The @p slots bytes of the sketch of one text, as @c sz_ngrams_sketch_simt writes it;
 *      every lane of one simdgroup calls it for the same text.
 *
 *  Each lane rolls an even run of every width's N-grams and lowers each window's slot by an atomic
 *  minimum on @p ranks, then fills a run of slots: a slot no window reached borrows the rank of the
 *  nearest reached slot after it, wrapping past the last, mixed with its distance. The ranks are
 *  read and never overwritten past the minimums, so no lane waits on another's fill.
 *
 *  @param[in] widths Bytes a window spans at each of @p widths_count widths, each at least one.
 *  @param[in,out] ranks The simdgroup's own @p slots words, in threadgroup or device memory.
 *  @param[in] slots Bytes of the sketch, at least one.
 *  @param[out] sketch The @p slots bytes, which the lanes write between them.
 *  @param[in] lane The caller's lane in its simdgroup, taken modulo 32.
 */
template <typename widths_type_, typename ranks_type_>
inline void sz_ngrams_sketch_metal(device uchar const *text, ulong length, widths_type_ widths, ulong widths_count,
                                   ranks_type_ ranks, ulong slots, device uchar *sketch, uint lane) {
    uint const lanes = 32;
    sz_group_metal_t const group = {lanes, lane & 31u};
    ulong slot;
    // The ranks may still serve the previous text's fills until every lane gets here.
    group.sync();
    for (slot = group.lane; slot < slots; slot += lanes) atomic_store_explicit(ranks + slot, ~0u, memory_order_relaxed);
    group.sync();

    for (ulong index = 0; index != widths_count; ++index) {
        ulong const width = widths[index];
        if (!width || width > length) continue;
        uint const lifted_power = sz_ngrams_lift_metal_(sz_ngrams_power_metal_((uint)width));
        ulong const ngrams = length - width + 1;
        ulong const run = sz_size_divide_round_up_metal_<ulong>(ngrams, lanes);
        ulong const first = min((ulong)group.lane * run, ngrams), last = min(first + run, ngrams);
        if (first == last) continue;
        uint residue = 0;
        for (ulong position = first; position < first + width - 1; ++position)
            residue = sz_ngrams_roll_lifted_metal_(residue, text[position], 0u, lifted_power);
        for (ulong window = first; window < last; ++window) {
            uint const outgoing = window != first ? text[window - 1] : 0u;
            residue = sz_ngrams_roll_lifted_metal_(residue, text[window + width - 1], outgoing, lifted_power);
            uint const mixed = sz_ngrams_mix_metal_(residue + (uint)width);
            ulong const target = ((ulong)mixed * slots) >> 32;
            atomic_fetch_min_explicit(ranks + target, sz_ngrams_mix_metal_(mixed ^ 0x9E3779B9u), memory_order_relaxed);
        }
    }
    group.sync();

    // Each lane fills its run of slots downwards from the nearest reached slot past it, found once.
    ulong const run = sz_size_divide_round_up_metal_<ulong>(slots, lanes);
    ulong const begin = min((ulong)group.lane * run, slots), end = min(begin + run, slots);
    ulong next = slots;
    for (ulong step = 0, at = end; begin != end && step < slots; ++step, ++at) {
        if (at == slots) at = 0;
        if (atomic_load_explicit(ranks + at, memory_order_relaxed) != ~0u) {
            next = at;
            break;
        }
    }
    for (slot = end; slot > begin; --slot) {
        uint rank = atomic_load_explicit(ranks + slot - 1, memory_order_relaxed);
        if (rank != ~0u) next = slot - 1;
        else if (next != slots)
            rank = sz_ngrams_mix_metal_(atomic_load_explicit(ranks + next, memory_order_relaxed) +
                                        (uint)(next >= slot ? next - (slot - 1) : next + slots - (slot - 1)));
        sketch[slot - 1] = (uchar)sz_ngrams_mix_metal_(rank);
    }
}

/**
 *  @brief One simdgroup per text, each with its own @p slots ranks in threadgroup memory, writing
 *      the text's sketch.
 *
 *  A simdgroup takes every text a grid's worth of simdgroups apart, so a batch past one grid
 *  dimension is still covered by the threadgroups it has.
 */
kernel void sz_ngrams_sketches_metal_kernel_(
    device uchar const *texts [[buffer(0)]], device uchar const *texts_offsets [[buffer(1)]],
    device uchar *sketches [[buffer(2)]], constant sz_ngrams_sketches_arguments_metal_t &arguments [[buffer(3)]],
    threadgroup atomic_uint *ranks [[threadgroup(0)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint simdgroups [[simdgroups_per_threadgroup]]) {
    sz_strings_tape_metal_t const tape = {texts, texts_offsets, arguments.texts_layout};
    threadgroup atomic_uint *const own_ranks = ranks + simdgroup * arguments.slots;
    ulong const stride = (ulong)groups.x * simdgroups;
    for (ulong index = (ulong)group.x * simdgroups + simdgroup; index < arguments.texts_count; index += stride)
        sz_ngrams_sketch_metal(sz_strings_start_metal_(tape, index), sz_strings_length_metal_(tape, index),
                               arguments.widths, arguments.widths_count, own_ranks, arguments.slots,
                               sketches + index * arguments.sketches_stride, lane);
}

#pragma endregion Sketches
