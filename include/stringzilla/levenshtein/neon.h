/**
 *  @file include/stringzilla/levenshtein/neon.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief NEON bit-vector edit distances over bytes and UTF-8 runes.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_NEON_H_
#define STRINGZILLA_LEVENSHTEIN_NEON_H_
#include "stringzilla/types.h"
#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/utf8_runes/neon.h" // `sz_utf8_decode_neon_`
#ifdef __cplusplus
extern "C" {
#endif
#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEON_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

typedef struct sz_levenshtein_u64x2_state_neon_t {
    sz_u128_vec_t scores_vec;
} sz_levenshtein_u64x2_state_neon_t;
typedef struct sz_levenshtein_u64x2_vertical_neon_t {
    sz_u128_vec_t positive_vec, negative_vec;
} sz_levenshtein_u64x2_vertical_neon_t;

STRINGZILLA_INLINE void sz_levenshtein_u64x2_init_neon_(sz_levenshtein_u64x2_state_neon_t *state,
                                                        sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                        sz_size_t words, sz_levenshtein_query_t const *query) {
    state->scores_vec.u64x2 = vdupq_n_u64(query->length);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.u64x2 = vdupq_n_u64(~(sz_u64_t)0);
        verticals[word].negative_vec.u64x2 = vdupq_n_u64(0);
    }
}

STRINGZILLA_INLINE void sz_levenshtein_u64x2_steps_neon_(sz_levenshtein_u64x2_state_neon_t *states,
                                                         sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                         sz_size_t words, sz_levenshtein_query_t const *query,
                                                         sz_u64_t const *const *rows, sz_size_t registers) {
    uint64x2_t positive_carry_u64x2[4], negative_carry_u64x2[4];
    for (sz_size_t reg = 0; reg != registers; ++reg) {
        positive_carry_u64x2[reg] = vdupq_n_u64(1);
        negative_carry_u64x2[reg] = vdupq_n_u64(0);
    }
    uint64x2_t const last_symbol_bit_u64x2 = vdupq_n_u64(sz_levenshtein_last_symbol_bit_(query->length));
    int64x2_t const last_symbol_shift_i64x2 = vdupq_n_s64(-(sz_i64_t)sz_levenshtein_last_symbol_shift_(query->length));
    for (sz_size_t word = 0; word != words; ++word) {
        for (sz_size_t reg = 0; reg != registers; ++reg) {
            sz_levenshtein_u64x2_state_neon_t *state = states + reg;
            sz_levenshtein_u64x2_vertical_neon_t *vertical = verticals + reg * words + word;
            sz_u64_t const masks[2] = {rows[reg * 2][word * query->stride], rows[reg * 2 + 1][word * query->stride]};
            uint64x2_t const equality_u64x2 = vld1q_u64(masks);
            uint64x2_t const vertical_carry_u64x2 = vorrq_u64(equality_u64x2, vertical->negative_vec.u64x2);
            uint64x2_t const matched_u64x2 = vorrq_u64(equality_u64x2, negative_carry_u64x2[reg]);
            uint64x2_t const sum_u64x2 = vaddq_u64(vandq_u64(matched_u64x2, vertical->positive_vec.u64x2),
                                                   vertical->positive_vec.u64x2);
            uint64x2_t const diagonal_u64x2 = vorrq_u64(veorq_u64(sum_u64x2, vertical->positive_vec.u64x2),
                                                        matched_u64x2);
            uint64x2_t horizontal_positive_u64x2 = vorrq_u64(
                vertical->negative_vec.u64x2, vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(
                                                  vorrq_u64(diagonal_u64x2, vertical->positive_vec.u64x2)))));
            uint64x2_t horizontal_negative_u64x2 = vandq_u64(vertical->positive_vec.u64x2, diagonal_u64x2);
            if (word + 1 == words) {
                state->scores_vec.u64x2 = vaddq_u64(
                    state->scores_vec.u64x2,
                    vshlq_u64(vandq_u64(horizontal_positive_u64x2, last_symbol_bit_u64x2), last_symbol_shift_i64x2));
                state->scores_vec.u64x2 = vsubq_u64(
                    state->scores_vec.u64x2,
                    vshlq_u64(vandq_u64(horizontal_negative_u64x2, last_symbol_bit_u64x2), last_symbol_shift_i64x2));
            }
            uint64x2_t const next_positive_u64x2 = vshrq_n_u64(horizontal_positive_u64x2, 63),
                             next_negative_u64x2 = vshrq_n_u64(horizontal_negative_u64x2, 63);
            horizontal_positive_u64x2 = vorrq_u64(vshlq_n_u64(horizontal_positive_u64x2, 1), positive_carry_u64x2[reg]);
            horizontal_negative_u64x2 = vorrq_u64(vshlq_n_u64(horizontal_negative_u64x2, 1), negative_carry_u64x2[reg]);
            positive_carry_u64x2[reg] = next_positive_u64x2, negative_carry_u64x2[reg] = next_negative_u64x2;
            vertical->positive_vec.u64x2 = vorrq_u64(horizontal_negative_u64x2,
                                                     vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(
                                                         vorrq_u64(vertical_carry_u64x2, horizontal_positive_u64x2)))));
            vertical->negative_vec.u64x2 = vandq_u64(horizontal_positive_u64x2, vertical_carry_u64x2);
        }
    }
}

STRINGZILLA_INLINE void sz_levenshtein_u64x2_step_neon_(sz_levenshtein_u64x2_state_neon_t *state,
                                                        sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                        sz_size_t words, sz_levenshtein_query_t const *query,
                                                        sz_u128_vec_t classes_vec) {
    sz_u64_t const *rows[2] = {query->masks + vgetq_lane_u64(classes_vec.u64x2, 0),
                               query->masks + vgetq_lane_u64(classes_vec.u64x2, 1)};
    sz_levenshtein_u64x2_steps_neon_(state, verticals, words, query, rows, 1);
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x2_score_neon_(sz_levenshtein_u64x2_state_neon_t const *state,
                                                              sz_size_t candidate) {
    sz_u64_t scores[2];
    vst1q_u64(scores, state->scores_vec.u64x2);
    return scores[candidate];
}

/** Initializes two candidates' Myers states for a prepared query. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_init_neon(sz_levenshtein_u64x2_state_neon_t *state,
                                                       sz_levenshtein_u64x2_vertical_neon_t *verticals, sz_size_t words,
                                                       sz_levenshtein_query_t const *query) {
    sz_levenshtein_u64x2_init_neon_(state, verticals, words, query);
}

/** Advances two candidates by their query class IDs. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_step_neon(sz_levenshtein_u64x2_state_neon_t *state,
                                                       sz_levenshtein_u64x2_vertical_neon_t *verticals, sz_size_t words,
                                                       sz_levenshtein_query_t const *query, sz_u128_vec_t classes_vec) {
    sz_levenshtein_u64x2_step_neon_(state, verticals, words, query, classes_vec);
}

/** Tests whether either unfinished candidate can still reach the radius. */
STRINGZILLA_INLINE sz_bool_t sz_levenshtein_u64x2_any_active_neon(sz_levenshtein_u64x2_state_neon_t const *state,
                                                                  sz_u128_vec_t symbol_counts_vec, sz_size_t position,
                                                                  sz_ssize_t radius) {
    uint64x2_t const positions_u64x2 = vdupq_n_u64(position);
    int64x2_t const floor_i64x2 = vreinterpretq_s64_u64(
        vsubq_u64(vaddq_u64(state->scores_vec.u64x2, positions_u64x2), symbol_counts_vec.u64x2));
    uint64x2_t const active_u64x2 = vandq_u64(vcleq_s64(floor_i64x2, vdupq_n_s64(radius)),
                                              vcgtq_u64(symbol_counts_vec.u64x2, positions_u64x2));
    return (vgetq_lane_u64(active_u64x2, 0) | vgetq_lane_u64(active_u64x2, 1)) != 0 ? sz_true_k : sz_false_k;
}

/** Reads a candidate's running score at its text's end. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x2_score_neon(sz_levenshtein_u64x2_state_neon_t const *state,
                                                             sz_size_t candidate) {
    return sz_levenshtein_u64x2_score_neon_(state, candidate);
}

STRINGZILLA_INLINE sz_u128_vec_t sz_levenshtein_u64x2_classes_u8_neon_(sz_u8_t const *classes) {
    sz_u128_vec_t classes_vec;
    classes_vec.u64x2 = vcombine_u64(vcreate_u64(classes[0]), vcreate_u64(classes[1]));
    return classes_vec;
}

/** Two byte-wide class ids, as the byte transpose emits them, widened to row indices. */
STRINGZILLA_INLINE sz_u128_vec_t sz_levenshtein_u64x2_classes_u8_neon(sz_u8_t const *classes) {
    return sz_levenshtein_u64x2_classes_u8_neon_(classes);
}

STRINGZILLA_INLINE sz_u128_vec_t sz_levenshtein_u64x2_classes_u32_neon_(sz_u32_t const *classes) {
    sz_u128_vec_t classes_vec;
    classes_vec.u64x2 = vmovl_u32(vld1_u32(classes));
    return classes_vec;
}

/** Two four-byte class ids, as the UTF-8 transpose emits them, widened to row indices. */
STRINGZILLA_INLINE sz_u128_vec_t sz_levenshtein_u64x2_classes_u32_neon(sz_u32_t const *classes) {
    return sz_levenshtein_u64x2_classes_u32_neon_(classes);
}

/** Sixteen bytes of a @p length -byte @p text from @p cursor on, zeros past its end, never read
 *  outside it: a text of sixteen bytes or more loads the sixteen ending at its own end once the
 *  cursor nears it and shuffles out the ones passed; a shorter text reads two clamped octets. */
STRINGZILLA_INLINE uint8x16_t sz_levenshtein_sixteen_neon_(sz_cptr_t text, sz_size_t length, sz_size_t cursor) {
    if (length < 16)
        return vcombine_u8(vcreate_u8(sz_levenshtein_octet_(text, length, cursor)),
                           vcreate_u8(sz_levenshtein_octet_(text, length, cursor + 8)));
    sz_size_t const start = sz_min_of_two(cursor, length - 16);
    sz_size_t const passed = sz_min_of_two(cursor, length) - start;
    // Byte `i` takes byte `i + passed`, and an index past fifteen reads zero.
    uint8x16_t const indices_u8x16 = vaddq_u8(
        vcombine_u8(vcreate_u8(0x0706050403020100ull), vcreate_u8(0x0F0E0D0C0B0A0908ull)), vdupq_n_u8((sz_u8_t)passed));
    return vqtbl1q_u8(vld1q_u8((sz_u8_t const *)(text + start)), indices_u8x16);
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x8_transpose_neon_(sz_levenshtein_query_t const *query,
                                                                 sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                 sz_size_t candidates, sz_size_t *cursors,
                                                                 sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                 sz_size_t positions, void *transpose_classes) {
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    sz_u8_t const *const byte_to_class = query->byte_to_class;
    sz_u8_t *const classes = (sz_u8_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != 8; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    // Four 64-entry lookups class a byte, each answering zero outside its quarter of the values.
    uint8x16x4_t const quarters_u8x16x4[4] = {vld1q_u8_x4(byte_to_class), vld1q_u8_x4(byte_to_class + 64),
                                              vld1q_u8_x4(byte_to_class + 128), vld1q_u8_x4(byte_to_class + 192)};
    for (sz_size_t position = 0; position < filled; position += 16) {
        uint8x16_t lanes_u8x16[8];
        for (sz_size_t candidate = 0; candidate != 8; ++candidate) {
            uint8x16_t const bytes_u8x16 = sz_levenshtein_sixteen_neon_(
                texts[candidate], (sz_size_t)byte_counts[candidate], cursors[candidate] + position);
            lanes_u8x16[candidate] = vorrq_u8(
                vorrq_u8(vqtbl4q_u8(quarters_u8x16x4[0], bytes_u8x16),
                         vqtbl4q_u8(quarters_u8x16x4[1], veorq_u8(bytes_u8x16, vdupq_n_u8(0x40)))),
                vorrq_u8(vqtbl4q_u8(quarters_u8x16x4[2], veorq_u8(bytes_u8x16, vdupq_n_u8(0x80))),
                         vqtbl4q_u8(quarters_u8x16x4[3], veorq_u8(bytes_u8x16, vdupq_n_u8(0xC0)))));
        }
        // Three rounds of transposes at 8, 16 and 32 bits: register `q` ends with position `q` in
        // its low half and position `q + 8` in its high half, all eight lanes in each.
        uint8x16_t pairs_u8x16[8];
        for (sz_size_t pair = 0; pair != 4; ++pair) {
            pairs_u8x16[pair] = vtrn1q_u8(lanes_u8x16[2 * pair], lanes_u8x16[2 * pair + 1]);
            pairs_u8x16[pair + 4] = vtrn2q_u8(lanes_u8x16[2 * pair], lanes_u8x16[2 * pair + 1]);
        }
        uint16x8_t quads_u16x8[8];
        for (sz_size_t parity = 0; parity != 2; ++parity)
            for (sz_size_t half = 0; half != 2; ++half) {
                uint16x8_t const first_u16x8 = vreinterpretq_u16_u8(pairs_u8x16[parity * 4 + half * 2]);
                uint16x8_t const second_u16x8 = vreinterpretq_u16_u8(pairs_u8x16[parity * 4 + half * 2 + 1]);
                quads_u16x8[parity * 4 + half] = vtrn1q_u16(first_u16x8, second_u16x8);
                quads_u16x8[parity * 4 + half + 2] = vtrn2q_u16(first_u16x8, second_u16x8);
            }
        uint8x16_t rows_u8x16[8];
        for (sz_size_t offset = 0; offset != 4; ++offset) {
            sz_size_t const parity = offset & 1, pair = offset >> 1;
            uint32x4_t const lower_u32x4 = vreinterpretq_u32_u16(quads_u16x8[parity * 4 + pair * 2]);
            uint32x4_t const upper_u32x4 = vreinterpretq_u32_u16(quads_u16x8[parity * 4 + pair * 2 + 1]);
            rows_u8x16[offset] = vreinterpretq_u8_u32(vtrn1q_u32(lower_u32x4, upper_u32x4));
            rows_u8x16[offset + 4] = vreinterpretq_u8_u32(vtrn2q_u32(lower_u32x4, upper_u32x4));
        }
        // A last chunk past `positions` stores only the rows the buffer holds.
        sz_size_t const rows = sz_min_of_two(positions - position, (sz_size_t)16);
        for (sz_size_t row = 0; row != rows; ++row)
            vst1_u8(classes + (position + row) * 8,
                    row < 8 ? vget_low_u8(rows_u8x16[row]) : vget_high_u8(rows_u8x16[row - 8]));
    }
    for (sz_size_t candidate = 0; candidate != 8; ++candidate)
        cursors[candidate] += sz_min_of_two(filled, (sz_size_t)byte_counts[candidate] - cursors[candidate]);
    return filled;
}

/** The byte transpose for eight candidates, two per register across four registers: sixteen
 *  positions per eight loads, each clamped to its own candidate's end, classed by four table
 *  lookups and transposed by three rounds of @c TRN; emits the @c sz_u8_t class of every byte. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x8_transpose_neon(sz_levenshtein_query_t const *query,
                                                                sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                sz_size_t candidates, sz_size_t *cursors,
                                                                sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                sz_size_t positions, void *transpose_classes) {
    return sz_levenshtein_u8x8_transpose_neon_(query, texts, byte_counts, candidates, cursors, symbol_counts,
                                               transpose_start, positions, transpose_classes);
}

enum { sz_levenshtein_u64x2_candidates_per_step_neon_k = 2, sz_levenshtein_u64x2_registers_per_position_neon_k = 4 };

/** Sweeps eight candidates, two per register, through every transpose; @p words and @p width are
 *  constants, keeping short queries' verticals in registers and the class loads unbranched. Each
 *  transpose is consumed in runs that step the registers holding unread candidates from one event
 *  to the next: a candidate's text ending, where its score is read and the stepped registers may
 *  shrink, or a freeze test, which ends the sweep once every unread candidate is past @p radius,
 *  after which the unread ones read their running scores where they stand. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_sweep_neon_(sz_levenshtein_query_t const *shared_query,
                                                         sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                         sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                         sz_levenshtein_transpose_t_ transpose,
                                                         sz_levenshtein_classes_width_t width, sz_ssize_t radius,
                                                         sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                         sz_size_t words, sz_size_t *distances) {
    enum {
        registers_k = sz_levenshtein_u64x2_registers_per_position_neon_k,
        candidates_per_position_k = 2 * registers_k,
        positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k
    };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_u64_t unread = ((sz_u64_t)1 << sweep_count) - 1;
    // Registers through the highest one holding an unread candidate, all stepped by every run.
    sz_size_t registers = (sweep_count + 1) / 2;
    sz_levenshtein_u64x2_state_neon_t states[registers_k];
    for (sz_size_t reg = 0; reg != registers_k; ++reg)
        sz_levenshtein_u64x2_init_neon_(&states[reg], verticals + reg * words, words, query);
    sz_u32_t transpose_classes[positions_per_transpose_k][candidates_per_position_k];
    sz_u8_t const *const transpose_bytes = (sz_u8_t const *)&transpose_classes[0][0];
    sz_size_t freeze_position = sz_levenshtein_freeze_start_(radius);
    for (sz_size_t transpose_start = 0, filled = positions_per_transpose_k; filled == positions_per_transpose_k;
         transpose_start += filled) {
        filled = transpose(query, texts, byte_counts, candidates_per_position_k, cursors, symbol_counts,
                           transpose_start, positions_per_transpose_k, &transpose_classes[0][0]);
        sz_u64_t counts[candidates_per_position_k];
        for (sz_size_t candidate = 0; candidate != candidates_per_position_k; ++candidate)
            counts[candidate] = symbol_counts[candidate];
        sz_levenshtein_deadline_t deadline = sz_levenshtein_deadline_(unread, counts);
        for (sz_size_t position = 0;;) {
            sz_size_t const event = sz_min_of_two(deadline.position, freeze_position);
            sz_size_t const run_end = sz_min_of_two(event - transpose_start, filled);
            for (; position != run_end; ++position) {
                sz_u64_t const *rows[candidates_per_position_k];
                for (sz_size_t candidate = 0; candidate != registers * 2; ++candidate)
                    rows[candidate] = query->masks +
                                      (width == sz_levenshtein_classes_u8_k
                                           ? transpose_bytes[position * candidates_per_position_k + candidate]
                                           : transpose_classes[position][candidate]);
                sz_levenshtein_u64x2_steps_neon_(states, verticals, words, query, rows, registers);
            }
            if (position == filled) break;
            if (event == deadline.position) {
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)sz_u64_ctz_neon_(ending);
                    distances[candidate] = sz_levenshtein_u64x2_score_neon_(&states[candidate / 2], candidate % 2);
                }
                unread &= ~deadline.retiring;
                registers = unread ? (sz_size_t)(65 - sz_u64_clz_neon_(unread)) / 2 : 0;
                deadline = sz_levenshtein_deadline_(unread, counts);
            }
            if (event == freeze_position) {
                sz_size_t active = 0;
                for (sz_size_t reg = 0; reg != registers; ++reg) {
                    sz_u128_vec_t counts_vec;
                    counts_vec.u64x2 = vld1q_u64(counts + reg * 2);
                    active += sz_levenshtein_u64x2_any_active_neon(&states[reg], counts_vec, event, radius) ==
                              sz_true_k;
                }
                // A zero fill ends the transposes, as when every candidate runs dry.
                if (active == 0) {
                    filled = 0;
                    break;
                }
                freeze_position += sz_levenshtein_freeze_positions_k;
            }
        }
    }
    // Unread candidates end where the transposes ran dry, or froze and read their running score.
    for (; unread; unread &= unread - 1) {
        sz_size_t const candidate = (sz_size_t)sz_u64_ctz_neon_(unread);
        distances[candidate] = sz_levenshtein_u64x2_score_neon_(&states[candidate / 2], candidate % 2);
    }
}

/** Streams every candidate through a prepared @p query, eight at a time in @p order, with
 *  @p transpose emitting their classes at @p width, each distance under @p bound; @p verticals
 *  holds enough for a runtime word count. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_distances_neon_(
    sz_levenshtein_query_t const *query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_levenshtein_u64x2_vertical_neon_t *verticals, sz_size_t *distances) {
    enum { candidates_per_position_k = 2 * sz_levenshtein_u64x2_registers_per_position_neon_k };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_levenshtein_u64x2_vertical_neon_t resident_verticals[sz_levenshtein_u64x2_registers_per_position_neon_k * 2];
    for (sz_size_t next_rank = 0; next_rank != candidates->count;) {
        sz_cptr_t texts[candidates_per_position_k];
        sz_u64_t byte_counts[candidates_per_position_k], sweep_symbol_counts[candidates_per_position_k];
        sz_size_t indices[candidates_per_position_k], sweep_distances[candidates_per_position_k];
        sz_size_t const sweep_count = sz_levenshtein_sweep_seed_(candidates, symbol_counts, order, query, bound,
                                                                 &next_rank, candidates_per_position_k, indices, texts,
                                                                 byte_counts, sweep_symbol_counts, distances);
        if (!sweep_count) break;
        if (words == 1)
            sz_levenshtein_u64x2_sweep_neon_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                             width, radius, resident_verticals, 1, sweep_distances);
        else if (words == 2)
            sz_levenshtein_u64x2_sweep_neon_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                             width, radius, resident_verticals, 2, sweep_distances);
        else
            sz_levenshtein_u64x2_sweep_neon_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                             width, radius, verticals, words, sweep_distances);
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
            distances[indices[candidate]] = sz_levenshtein_bounded_(sweep_distances[candidate], bound);
    }
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_neon(sz_levenshtein_engine_t *engine,
                                                            sz_strings_view_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                            sz_stream_t stream) {
    return sz_levenshtein_engine_init_cpu_(engine, queries, symbol, sz_cap_neon_k, allocator, stream);
}

/** Scores every candidate of a round against @p engine 's query @p index on NEON. */
STRINGZILLA_INLINE void sz_levenshtein_run_neon_(sz_levenshtein_engine_t const *engine, sz_size_t index,
                                                 sz_strings_view_t const *scored, sz_size_t const *symbol_counts,
                                                 sz_size_t const *order, sz_size_t bound,
                                                 sz_levenshtein_u64x2_vertical_neon_t *verticals,
                                                 sz_size_t *distances) {
    if (engine->lengths[index] == 0) {
        sz_levenshtein_engine_empty_(scored, symbol_counts, bound, distances);
        return;
    }
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
    // Each alphabet passes its transpose and class width as literals, which the sweeps fold.
    if (engine->symbol == sz_levenshtein_bytes_k)
        sz_levenshtein_u64x2_distances_neon_(&query, scored, symbol_counts, order, sz_levenshtein_u8x8_transpose_neon_,
                                             sz_levenshtein_classes_u8_k, bound, verticals, distances);
    else
        sz_levenshtein_u64x2_distances_neon_(&query, scored, symbol_counts, order, sz_levenshtein_transpose_runes_,
                                             sz_levenshtein_classes_u32_k, bound, verticals, distances);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_neon(sz_levenshtein_engine_t *engine,
                                                          sz_strings_view_t const *candidates,
                                                          sz_size_t const *symbol_counts, sz_size_t const *order,
                                                          sz_size_t bound, sz_size_t *distances,
                                                          sz_size_t distances_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_assert_((engine->capability & sz_cap_cpus_k) != 0 &&
               "A host tier never scores a device-prepared engine, whose head only its GPU tier reads");
    enum { registers_k = sz_levenshtein_u64x2_registers_per_position_neon_k };
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    sz_size_t const verticals_bytes = sz_levenshtein_engine_verticals_bytes_(
        registers_k, sz_levenshtein_engine_words_max_(engine), sizeof(sz_levenshtein_u64x2_vertical_neon_t));
    sz_strings_view_t scored;
    sz_size_t const *candidate_order;
    sz_status_t const prepared = sz_levenshtein_engine_round_(
        engine, candidates, order, verticals_bytes, sz_utf8_decode_neon_, &scored, &symbol_counts, &candidate_order);
    if (prepared != sz_success_k) return prepared;
    sz_levenshtein_u64x2_vertical_neon_t *const verticals =
        (sz_levenshtein_u64x2_vertical_neon_t *)sz_levenshtein_engine_verticals_(engine);
    for (sz_size_t index = 0; index != engine->count; ++index)
        sz_levenshtein_run_neon_(engine, index, &scored, symbol_counts, candidate_order, bound, verticals,
                                 distances + index * distances_stride);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_NEON_
#endif // STRINGZILLA_ARCH_ARM64_
#ifdef __cplusplus
}
#endif
#endif
