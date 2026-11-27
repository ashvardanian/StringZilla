/**
 *  @file include/stringzilla/levenshtein/powervsx.h
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief IBM Power VSX bit-vector edit distances over bytes and UTF-8 runes.
 *
 *  @sa c/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_POWERVSX_H_
#define STRINGZILLA_LEVENSHTEIN_POWERVSX_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_PPC64_
#if STRINGZILLA_ARCH_PPC64_POWERVSX_

#include "stringzilla/levenshtein/serial.h"
#include "stringzilla/utf8_runes/powervsx.h" // `sz_utf8_decode_powervsx_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

enum {
    sz_levenshtein_u64x2_candidates_per_step_powervsx_k = 2,
    sz_levenshtein_u64x2_registers_per_position_powervsx_k = 4
};

typedef struct sz_levenshtein_u64x2_state_powervsx_t {
    sz_u128_vec_t scores_vec;
} sz_levenshtein_u64x2_state_powervsx_t;
typedef struct sz_levenshtein_u64x2_vertical_powervsx_t {
    sz_u128_vec_t positive_vec, negative_vec;
} sz_levenshtein_u64x2_vertical_powervsx_t;

STRINGZILLA_INLINE void sz_levenshtein_u64x2_init_powervsx_(sz_levenshtein_u64x2_state_powervsx_t *state,
                                                            sz_levenshtein_u64x2_vertical_powervsx_t *verticals,
                                                            sz_size_t words, sz_levenshtein_query_t const *query) {
    state->scores_vec.vsx_u64 = vec_splats((unsigned long long)query->length);
    for (sz_size_t word = 0; word != words; ++word) {
        verticals[word].positive_vec.vsx_u64 = vec_splats(~0ull);
        verticals[word].negative_vec.vsx_u64 = vec_splats(0ull);
    }
}

STRINGZILLA_INLINE void sz_levenshtein_u64x2_steps_powervsx_(sz_levenshtein_u64x2_state_powervsx_t *states,
                                                             sz_levenshtein_u64x2_vertical_powervsx_t *verticals,
                                                             sz_size_t words, sz_levenshtein_query_t const *query,
                                                             sz_u64_t const *const *rows, sz_size_t registers) {
    __vector unsigned long long positive_carry_u64x2[sz_levenshtein_u64x2_registers_per_position_powervsx_k],
        negative_carry_u64x2[sz_levenshtein_u64x2_registers_per_position_powervsx_k];
    __vector unsigned long long const one_u64x2 = vec_splats(1ull), top_bit_u64x2 = vec_splats(63ull);
    for (sz_size_t reg = 0; reg != registers; ++reg) {
        positive_carry_u64x2[reg] = one_u64x2;
        negative_carry_u64x2[reg] = vec_splats(0ull);
    }
    __vector unsigned long long const last_symbol_bit_u64x2 = vec_splats(
        (unsigned long long)sz_levenshtein_last_symbol_bit_(query->length));
    __vector unsigned long long const last_symbol_shift_u64x2 = vec_splats(
        (unsigned long long)sz_levenshtein_last_symbol_shift_(query->length));
    for (sz_size_t word = 0; word != words; ++word) {
        for (sz_size_t reg = 0; reg != registers; ++reg) {
            sz_levenshtein_u64x2_state_powervsx_t *state = states + reg;
            sz_levenshtein_u64x2_vertical_powervsx_t *vertical = verticals + reg * words + word;
            __vector unsigned long long const positive_u64x2 = vertical->positive_vec.vsx_u64;
            // Two general-purpose words join one register in a single POWER9 `mtvsrdd`.
            __vector unsigned long long const equality_u64x2 = {rows[reg * 2][word * query->stride],
                                                                rows[reg * 2 + 1][word * query->stride]};
            __vector unsigned long long const vertical_carry_u64x2 = vec_or(equality_u64x2,
                                                                            vertical->negative_vec.vsx_u64);
            __vector unsigned long long const matched_u64x2 = vec_or(equality_u64x2, negative_carry_u64x2[reg]);
            __vector unsigned long long const sum_u64x2 = vec_add(vec_and(matched_u64x2, positive_u64x2),
                                                                  positive_u64x2);
            __vector unsigned long long const diagonal_u64x2 = vec_or(vec_xor(sum_u64x2, positive_u64x2),
                                                                      matched_u64x2);
            __vector unsigned long long horizontal_positive_u64x2 = vec_or(vertical->negative_vec.vsx_u64,
                                                                           vec_nor(diagonal_u64x2, positive_u64x2));
            __vector unsigned long long horizontal_negative_u64x2 = vec_and(positive_u64x2, diagonal_u64x2);
            if (word + 1 == words) {
                state->scores_vec.vsx_u64 = vec_add(
                    state->scores_vec.vsx_u64,
                    vec_sr(vec_and(horizontal_positive_u64x2, last_symbol_bit_u64x2), last_symbol_shift_u64x2));
                state->scores_vec.vsx_u64 = vec_sub(
                    state->scores_vec.vsx_u64,
                    vec_sr(vec_and(horizontal_negative_u64x2, last_symbol_bit_u64x2), last_symbol_shift_u64x2));
            }
            __vector unsigned long long const next_positive_u64x2 = vec_sr(horizontal_positive_u64x2, top_bit_u64x2),
                                              next_negative_u64x2 = vec_sr(horizontal_negative_u64x2, top_bit_u64x2);
            horizontal_positive_u64x2 = vec_or(vec_sl(horizontal_positive_u64x2, one_u64x2), positive_carry_u64x2[reg]);
            horizontal_negative_u64x2 = vec_or(vec_sl(horizontal_negative_u64x2, one_u64x2), negative_carry_u64x2[reg]);
            positive_carry_u64x2[reg] = next_positive_u64x2, negative_carry_u64x2[reg] = next_negative_u64x2;
            vertical->positive_vec.vsx_u64 = vec_or(horizontal_negative_u64x2,
                                                    vec_nor(vertical_carry_u64x2, horizontal_positive_u64x2));
            vertical->negative_vec.vsx_u64 = vec_and(horizontal_positive_u64x2, vertical_carry_u64x2);
        }
    }
}

STRINGZILLA_INLINE void sz_levenshtein_u64x2_step_powervsx_(sz_levenshtein_u64x2_state_powervsx_t *state,
                                                            sz_levenshtein_u64x2_vertical_powervsx_t *verticals,
                                                            sz_size_t words, sz_levenshtein_query_t const *query,
                                                            sz_u128_vec_t classes_vec) {
    sz_u64_t const *rows[2] = {query->masks + vec_extract(classes_vec.vsx_u64, 0),
                               query->masks + vec_extract(classes_vec.vsx_u64, 1)};
    sz_levenshtein_u64x2_steps_powervsx_(state, verticals, words, query, rows, 1);
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x2_score_powervsx_(sz_levenshtein_u64x2_state_powervsx_t const *state,
                                                                  sz_size_t candidate) {
    return (sz_size_t)vec_extract(state->scores_vec.vsx_u64, (int)candidate);
}

/** Initializes two candidates' Myers states for a prepared query. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_init_powervsx(sz_levenshtein_u64x2_state_powervsx_t *state,
                                                           sz_levenshtein_u64x2_vertical_powervsx_t *verticals,
                                                           sz_size_t words, sz_levenshtein_query_t const *query) {
    sz_levenshtein_u64x2_init_powervsx_(state, verticals, words, query);
}

/** Advances two candidates by their query class IDs. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_step_powervsx(sz_levenshtein_u64x2_state_powervsx_t *state,
                                                           sz_levenshtein_u64x2_vertical_powervsx_t *verticals,
                                                           sz_size_t words, sz_levenshtein_query_t const *query,
                                                           sz_u128_vec_t classes_vec) {
    sz_levenshtein_u64x2_step_powervsx_(state, verticals, words, query, classes_vec);
}

/** Tests whether either unfinished candidate can still reach the radius. */
STRINGZILLA_INLINE sz_bool_t sz_levenshtein_u64x2_any_active_powervsx(
    sz_levenshtein_u64x2_state_powervsx_t const *state, sz_u128_vec_t symbol_counts_vec, sz_size_t position,
    sz_ssize_t radius) {
    __vector unsigned long long const positions_u64x2 = vec_splats((unsigned long long)position);
    __vector signed long long const floor_i64x2 = (__vector signed long long)vec_sub(
        vec_add(state->scores_vec.vsx_u64, positions_u64x2), symbol_counts_vec.vsx_u64);
    __vector unsigned long long const active_u64x2 = vec_and(
        (__vector unsigned long long)vec_cmple(floor_i64x2, vec_splats((signed long long)radius)),
        (__vector unsigned long long)vec_cmpgt(symbol_counts_vec.vsx_u64, positions_u64x2));
    return vec_any_ne(active_u64x2, vec_splats(0ull)) ? sz_true_k : sz_false_k;
}

/** Reads a candidate's running score at its text's end. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u64x2_score_powervsx(sz_levenshtein_u64x2_state_powervsx_t const *state,
                                                                 sz_size_t candidate) {
    return sz_levenshtein_u64x2_score_powervsx_(state, candidate);
}

STRINGZILLA_INLINE sz_u128_vec_t sz_levenshtein_u64x2_classes_u8_powervsx_(sz_u8_t const *classes) {
    sz_u128_vec_t classes_vec;
    classes_vec.vsx_u64 = (__vector unsigned long long) {classes[0], classes[1]};
    return classes_vec;
}

/** Two byte-wide class ids, as the byte transpose emits them, widened to row indices. */
STRINGZILLA_INLINE sz_u128_vec_t sz_levenshtein_u64x2_classes_u8_powervsx(sz_u8_t const *classes) {
    return sz_levenshtein_u64x2_classes_u8_powervsx_(classes);
}

STRINGZILLA_INLINE sz_u128_vec_t sz_levenshtein_u64x2_classes_u32_powervsx_(sz_u32_t const *classes) {
    sz_u128_vec_t classes_vec;
    classes_vec.vsx_u64 = (__vector unsigned long long) {classes[0], classes[1]};
    return classes_vec;
}

/** Two four-byte class ids, as the UTF-8 transpose emits them, widened to row indices. */
STRINGZILLA_INLINE sz_u128_vec_t sz_levenshtein_u64x2_classes_u32_powervsx(sz_u32_t const *classes) {
    return sz_levenshtein_u64x2_classes_u32_powervsx_(classes);
}

/** Sixteen bytes of a @p length -byte @p text from @p cursor on, zeros past its end, never read
 *  outside it: one length-limited @c lxvl load, so neither a text's end nor a short text needs a
 *  path of its own. */
STRINGZILLA_INLINE __vector unsigned char sz_levenshtein_sixteen_powervsx_(sz_cptr_t text, sz_size_t length,
                                                                           sz_size_t cursor) {
    sz_size_t const start = sz_min_of_two(cursor, length);
    return vec_xl_len((unsigned char *)(text + start), sz_min_of_two(length - start, (sz_size_t)16));
}

/** Each byte's entry of the 256-entry table held in @p table_u8x16 's sixteen registers: eight
 *  two-register permutes look up the low five bits, and three rounds of selects on the top three
 *  bits pick among them. */
STRINGZILLA_INLINE __vector unsigned char sz_levenshtein_lookup_powervsx_(__vector unsigned char const *table_u8x16,
                                                                          __vector unsigned char bytes_u8x16) {
    __vector unsigned char const seven_u8x16 = vec_splats((unsigned char)7);
    __vector unsigned char eighths_u8x16[8];
    for (sz_size_t eighth = 0; eighth != 8; ++eighth)
        eighths_u8x16[eighth] = vec_perm(table_u8x16[2 * eighth], table_u8x16[2 * eighth + 1], bytes_u8x16);
    // Shifting a bit into the sign and back arithmetically fills its byte, ready for a select.
    for (sz_size_t bit = 5, count = 8; bit != 8; ++bit, count /= 2) {
        __vector unsigned char const selector_u8x16 = (__vector unsigned char)vec_sra(
            (__vector signed char)vec_sl(bytes_u8x16, vec_splats((unsigned char)(7 - bit))), seven_u8x16);
        for (sz_size_t half = 0; half != count / 2; ++half)
            eighths_u8x16[half] = vec_sel(eighths_u8x16[2 * half], eighths_u8x16[2 * half + 1], selector_u8x16);
    }
    return eighths_u8x16[0];
}

STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x8_transpose_powervsx_(sz_levenshtein_query_t const *query,
                                                                     sz_cptr_t const *texts,
                                                                     sz_u64_t const *byte_counts, sz_size_t candidates,
                                                                     sz_size_t *cursors, sz_u64_t *symbol_counts,
                                                                     sz_size_t transpose_start, sz_size_t positions,
                                                                     void *transpose_classes) {
    sz_unused_(symbol_counts), sz_unused_(transpose_start), sz_unused_(candidates);
    sz_u8_t *const classes = (sz_u8_t *)transpose_classes;
    sz_size_t filled = 0;
    for (sz_size_t candidate = 0; candidate != 8; ++candidate)
        filled = sz_max_of_two(filled,
                               sz_min_of_two(positions, (sz_size_t)byte_counts[candidate] - cursors[candidate]));
    __vector unsigned char table_u8x16[16];
    for (sz_size_t sixteenth = 0; sixteenth != 16; ++sixteenth)
        table_u8x16[sixteenth] = vec_xl((long)(sixteenth * 16), query->byte_to_class);
    for (sz_size_t position = 0; position < filled; position += 16) {
        __vector unsigned char lanes_u8x16[8];
        for (sz_size_t candidate = 0; candidate != 8; ++candidate)
            lanes_u8x16[candidate] = sz_levenshtein_lookup_powervsx_(
                table_u8x16, sz_levenshtein_sixteen_powervsx_(texts[candidate], (sz_size_t)byte_counts[candidate],
                                                              cursors[candidate] + position));
        // Three rounds of merges at 8, 16 and 32 bits leave every register two consecutive
        // positions, all eight candidates of each, in memory order.
        __vector unsigned short couples_u16x8[8];
        for (sz_size_t couple = 0; couple != 4; ++couple) {
            couples_u16x8[couple] = (__vector unsigned short)vec_mergeh(lanes_u8x16[2 * couple],
                                                                        lanes_u8x16[2 * couple + 1]);
            couples_u16x8[couple + 4] = (__vector unsigned short)vec_mergel(lanes_u8x16[2 * couple],
                                                                            lanes_u8x16[2 * couple + 1]);
        }
        __vector unsigned int quads_u32x4[8];
        for (sz_size_t half = 0; half != 2; ++half)
            for (sz_size_t quad = 0; quad != 2; ++quad) {
                __vector unsigned short const first_u16x8 = couples_u16x8[half * 4 + quad * 2];
                __vector unsigned short const second_u16x8 = couples_u16x8[half * 4 + quad * 2 + 1];
                quads_u32x4[half * 4 + quad] = (__vector unsigned int)vec_mergeh(first_u16x8, second_u16x8);
                quads_u32x4[half * 4 + quad + 2] = (__vector unsigned int)vec_mergel(first_u16x8, second_u16x8);
            }
        // A last chunk past `positions` stores only the bytes of the rows the buffer holds.
        sz_size_t const row_bytes = sz_min_of_two(positions - position, (sz_size_t)16) * 8;
        for (sz_size_t group = 0; group != 4; ++group) {
            __vector unsigned int const lower_u32x4 = quads_u32x4[2 * group], upper_u32x4 = quads_u32x4[2 * group + 1];
            __vector unsigned char const pairs_u8x16[2] = {
                (__vector unsigned char)vec_mergeh(lower_u32x4, upper_u32x4),
                (__vector unsigned char)vec_mergel(lower_u32x4, upper_u32x4)};
            for (sz_size_t pair = 0; pair != 2; ++pair) {
                sz_size_t const offset = (group * 2 + pair) * 16;
                vec_xst_len(pairs_u8x16[pair], classes + position * 8 + offset,
                            sz_min_of_two(row_bytes - sz_min_of_two(row_bytes, offset), (sz_size_t)16));
            }
        }
    }
    for (sz_size_t candidate = 0; candidate != 8; ++candidate)
        cursors[candidate] += sz_min_of_two(filled, (sz_size_t)byte_counts[candidate] - cursors[candidate]);
    return filled;
}

/** The byte transpose for eight candidates, two per register across four registers: sixteen
 *  positions per eight length-limited loads, each clamped to its own candidate's end, classed by
 *  eight permutes and seven selects and transposed by three rounds of merges; emits the @c sz_u8_t
 *  class of every byte. */
STRINGZILLA_INLINE sz_size_t sz_levenshtein_u8x8_transpose_powervsx(sz_levenshtein_query_t const *query,
                                                                    sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                                    sz_size_t candidates, sz_size_t *cursors,
                                                                    sz_u64_t *symbol_counts, sz_size_t transpose_start,
                                                                    sz_size_t positions, void *transpose_classes) {
    return sz_levenshtein_u8x8_transpose_powervsx_(query, texts, byte_counts, candidates, cursors, symbol_counts,
                                                   transpose_start, positions, transpose_classes);
}

/** Sweeps eight candidates, two per register, through every transpose; @p words and @p width are
 *  constants, keeping short queries' verticals in registers and the class loads unbranched. Each
 *  transpose is consumed in runs that step the registers holding unread candidates from one event
 *  to the next: a candidate's text ending, where its score is read and the stepped registers may
 *  shrink, or a freeze test, which ends the sweep once every unread candidate is past @p radius,
 *  after which the unread ones read their running scores where they stand. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_sweep_powervsx_(sz_levenshtein_query_t const *shared_query,
                                                             sz_cptr_t const *texts, sz_u64_t const *byte_counts,
                                                             sz_u64_t *symbol_counts, sz_size_t sweep_count,
                                                             sz_levenshtein_transpose_t_ transpose,
                                                             sz_levenshtein_classes_width_t width, sz_ssize_t radius,
                                                             sz_levenshtein_u64x2_vertical_powervsx_t *verticals,
                                                             sz_size_t words, sz_size_t *distances) {
    enum {
        registers_k = sz_levenshtein_u64x2_registers_per_position_powervsx_k,
        candidates_per_position_k = sz_levenshtein_u64x2_candidates_per_step_powervsx_k * registers_k,
        positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k
    };
    sz_levenshtein_query_t const local_query = *shared_query;
    sz_levenshtein_query_t const *const query = &local_query;
    sz_size_t cursors[candidates_per_position_k] = {0};
    sz_u64_t unread = sz_u64_mask_until_serial_(sweep_count);
    // Registers through the highest one holding an unread candidate, all stepped by every run.
    sz_size_t registers = (sweep_count + 1) / 2;
    sz_levenshtein_u64x2_state_powervsx_t states[registers_k];
    for (sz_size_t reg = 0; reg != registers_k; ++reg)
        sz_levenshtein_u64x2_init_powervsx_(&states[reg], verticals + reg * words, words, query);
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
                sz_levenshtein_u64x2_steps_powervsx_(states, verticals, words, query, rows, registers);
            }
            if (position == filled) break;
            if (event == deadline.position) {
                for (sz_u64_t ending = deadline.retiring; ending; ending &= ending - 1) {
                    sz_size_t const candidate = (sz_size_t)sz_u64_ctz(ending);
                    distances[candidate] = sz_levenshtein_u64x2_score_powervsx_(&states[candidate / 2], candidate % 2);
                }
                unread &= ~deadline.retiring;
                registers = unread ? (sz_size_t)(65 - sz_u64_clz(unread)) / 2 : 0;
                deadline = sz_levenshtein_deadline_(unread, counts);
            }
            if (event == freeze_position) {
                sz_size_t active = 0;
                for (sz_size_t reg = 0; reg != registers; ++reg) {
                    sz_u128_vec_t counts_vec;
                    counts_vec.vsx_u64 = (__vector unsigned long long) {counts[reg * 2], counts[reg * 2 + 1]};
                    active += sz_levenshtein_u64x2_any_active_powervsx(&states[reg], counts_vec, event, radius) ==
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
        sz_size_t const candidate = (sz_size_t)sz_u64_ctz(unread);
        distances[candidate] = sz_levenshtein_u64x2_score_powervsx_(&states[candidate / 2], candidate % 2);
    }
}

/** Streams every candidate through a prepared @p query, eight at a time in @p order, with
 *  @p transpose emitting their classes at @p width, each distance under @p bound; @p verticals
 *  holds enough for a runtime word count. */
STRINGZILLA_INLINE void sz_levenshtein_u64x2_distances_powervsx_(
    sz_levenshtein_query_t const *query, sz_strings_view_t const *candidates, sz_size_t const *symbol_counts,
    sz_size_t const *order, sz_levenshtein_transpose_t_ transpose, sz_levenshtein_classes_width_t width,
    sz_size_t bound, sz_levenshtein_u64x2_vertical_powervsx_t *verticals, sz_size_t *distances) {
    enum {
        registers_k = sz_levenshtein_u64x2_registers_per_position_powervsx_k,
        candidates_per_position_k = sz_levenshtein_u64x2_candidates_per_step_powervsx_k * registers_k
    };
    sz_size_t const words = sz_levenshtein_query_words(query->length);
    sz_ssize_t const radius = sz_levenshtein_radius_(bound);
    sz_levenshtein_u64x2_vertical_powervsx_t resident_verticals[registers_k * 2];
    for (sz_size_t next_rank = 0; next_rank != candidates->count;) {
        sz_cptr_t texts[candidates_per_position_k];
        sz_u64_t byte_counts[candidates_per_position_k], sweep_symbol_counts[candidates_per_position_k];
        sz_size_t indices[candidates_per_position_k], sweep_distances[candidates_per_position_k];
        sz_size_t const sweep_count = sz_levenshtein_sweep_seed_(candidates, symbol_counts, order, query, bound,
                                                                 &next_rank, candidates_per_position_k, indices, texts,
                                                                 byte_counts, sweep_symbol_counts, distances);
        if (!sweep_count) break;
        if (words == 1)
            sz_levenshtein_u64x2_sweep_powervsx_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                 width, radius, resident_verticals, 1, sweep_distances);
        else if (words == 2)
            sz_levenshtein_u64x2_sweep_powervsx_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                 width, radius, resident_verticals, 2, sweep_distances);
        else
            sz_levenshtein_u64x2_sweep_powervsx_(query, texts, byte_counts, sweep_symbol_counts, sweep_count, transpose,
                                                 width, radius, verticals, words, sweep_distances);
        for (sz_size_t candidate = 0; candidate != sweep_count; ++candidate)
            distances[indices[candidate]] = sz_levenshtein_bounded_(sweep_distances[candidate], bound);
    }
}

#if STRINGZILLA_TARGET_POWERVSX

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_powervsx(sz_levenshtein_engine_t *engine,
                                                                sz_strings_view_t const *queries,
                                                                sz_levenshtein_symbol_t symbol,
                                                                sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_levenshtein_engine_init_cpu_(engine, queries, symbol, sz_cap_powervsx_k, allocator, stream);
}

/** Scores every candidate of a round against @p engine 's query @p index on Power VSX. */
STRINGZILLA_INLINE void sz_levenshtein_run_powervsx_(sz_levenshtein_engine_t const *engine, sz_size_t index,
                                                     sz_strings_view_t const *scored, sz_size_t const *symbol_counts,
                                                     sz_size_t const *order, sz_size_t bound,
                                                     sz_levenshtein_u64x2_vertical_powervsx_t *verticals,
                                                     sz_size_t *distances) {
    if (engine->lengths[index] == 0) {
        sz_levenshtein_engine_empty_(scored, symbol_counts, bound, distances);
        return;
    }
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(engine, index);
    // Each alphabet passes its transpose and class width as literals, which the sweeps fold.
    if (engine->symbol == sz_levenshtein_bytes_k)
        sz_levenshtein_u64x2_distances_powervsx_(&query, scored, symbol_counts, order,
                                                 sz_levenshtein_u8x8_transpose_powervsx_, sz_levenshtein_classes_u8_k,
                                                 bound, verticals, distances);
    else
        sz_levenshtein_u64x2_distances_powervsx_(&query, scored, symbol_counts, order, sz_levenshtein_transpose_runes_,
                                                 sz_levenshtein_classes_u32_k, bound, verticals, distances);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_powervsx(sz_levenshtein_engine_t *engine,
                                                              sz_strings_view_t const *candidates,
                                                              sz_size_t const *symbol_counts, sz_size_t const *order,
                                                              sz_size_t bound, sz_size_t *distances,
                                                              sz_size_t distances_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_assert_((engine->capability & sz_cap_cpus_k) != 0 &&
               "A host tier never scores a device-prepared engine, whose head only its GPU tier reads");
    sz_size_t distances_stride_entries;
    sz_status_t const strided = sz_stride_entries_(distances_stride, sizeof(sz_size_t), candidates->count,
                                                   &distances_stride_entries);
    if (strided != sz_success_k) return strided;
    sz_size_t const verticals_bytes = sz_levenshtein_engine_verticals_bytes_(
        sz_levenshtein_u64x2_registers_per_position_powervsx_k, sz_levenshtein_engine_words_max_(engine),
        sizeof(sz_levenshtein_u64x2_vertical_powervsx_t));
    sz_strings_view_t scored;
    sz_size_t const *candidate_order;
    sz_status_t const prepared = sz_levenshtein_engine_round_(engine, candidates, order, verticals_bytes,
                                                              sz_utf8_decode_powervsx_, &scored, &symbol_counts,
                                                              &candidate_order);
    if (prepared != sz_success_k) return prepared;
    sz_levenshtein_u64x2_vertical_powervsx_t *const verticals =
        (sz_levenshtein_u64x2_vertical_powervsx_t *)sz_levenshtein_engine_verticals_(engine);
    for (sz_size_t index = 0; index != engine->count; ++index)
        sz_levenshtein_run_powervsx_(engine, index, &scored, symbol_counts, candidate_order, bound, verticals,
                                     distances + index * distances_stride_entries);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_POWERVSX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_PPC64_POWERVSX_
#endif // STRINGZILLA_ARCH_PPC64_

#endif // STRINGZILLA_LEVENSHTEIN_POWERVSX_H_
