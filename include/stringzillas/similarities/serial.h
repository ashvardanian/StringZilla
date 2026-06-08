/**
 *  @file include/stringzillas/similarities/serial.h
 *  @brief Plain-C serial similarity kernels (Levenshtein, Needleman-Wunsch, Smith-Waterman).
 *  @author Ash Vardanian
 *
 *  Serial backend for the four pairwise alignment operations, plus the shared ISA-agnostic cell math the
 *  SIMD & GPU backends reuse (folded in from the former `similarity_kernels.h`). Two explicit anti-diagonal
 *  bodies (linear & affine gaps) compute in `sz_i32_t` cells over `sz_u32_t` elements; objective (min/max),
 *  locality (global/local) and the uniform-vs-LUT substitution model are loop-invariant runtime fields. The
 *  per-op entries widen their byte/rune inputs to `sz_u32_t`, then run one body via a single per-pair loop
 *  over the device scope. Public results are u32 (distance) / i32 (score) widened into the caller's
 *  `sz_size_t`/`sz_ssize_t` slot. Narrow u8/u16 cells live only in the SIMD backends, where lane count pays.
 */
#ifndef STRINGZILLAS_SIMILARITIES_SERIAL_H_
#define STRINGZILLAS_SIMILARITIES_SERIAL_H_

#include "stringzillas/types.h"        // Engine PODs, `szs_sequence_view_t`, similarity enums
#include "stringzillas/stringzillas.h" // `sz_sequence_t`, `sz_sequence_u32tape_t`, `sz_sequence_u64tape_t`
#include "stringzillas/parallel.h"     // `szs_parallel_for_`: serial-or-fork_union per-pair loop

#include "stringzilla/types.h" // `sz_rune_t`, `sz_rune_length_t`, `sz_min_of_two`, `sz_max_of_two`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region - Shared Cell Math

/** @brief Pick min (minimize) or max (maximize) of two signed 32-bit cells per the objective. */
static inline sz_i32_t sz_pick_best_i32(sz_i32_t a, sz_i32_t b, sz_similarity_objective_t obj) {
    if (obj == sz_maximize_score_k) return a > b ? a : b;
    return a < b ? a : b;
}

/** @brief Pick min (minimize) or max (maximize) of two unsigned 32-bit cells per the objective. */
static inline sz_u32_t sz_pick_best_u32(sz_u32_t a, sz_u32_t b, sz_similarity_objective_t obj) {
    if (obj == sz_maximize_score_k) return a > b ? a : b;
    return a < b ? a : b;
}

/**
 *  @brief One linear-gap DP cell: best of substitution, insertion-gap, deletion-gap.
 *  @param pre_sub Diagonal predecessor cell (match/mismatch lands here + @p sub_cost).
 *  @param pre_ins,pre_del The two neighbor cells reached via a single linear gap of cost @p gap.
 *  @param is_local Smith-Waterman clamp: on maximize, never let a cell drop below 0.
 */
static inline sz_i32_t sz_cell_linear_i32(                //
    sz_i32_t pre_sub, sz_i32_t pre_ins, sz_i32_t pre_del, //
    sz_error_cost_t sub_cost, sz_error_cost_t gap,        //
    sz_similarity_objective_t obj, int is_local) {
    sz_i32_t best = pre_sub + (sz_i32_t)sub_cost;
    best = sz_pick_best_i32(best, pre_ins + (sz_i32_t)gap, obj);
    best = sz_pick_best_i32(best, pre_del + (sz_i32_t)gap, obj);
    if (is_local && obj == sz_maximize_score_k && best < 0) best = 0;
    return best;
}

/**
 *  @brief One affine-gap DP cell (Gotoh): updates the insertion & deletion gap tracks and the main cell.
 *  @param pre_sub Diagonal predecessor of the main matrix.
 *  @param up_main,up_ins Main & insertion-track cells of the upper neighbor (opening vs extending an insertion).
 *  @param left_main,left_del Main & deletion-track cells of the left neighbor (opening vs extending a deletion).
 *  @param out_ins,out_del Written with the new insertion/deletion-track values for this cell.
 *  @return The new main-matrix value for this cell.
 */
static inline sz_i32_t sz_cell_affine_i32(                                          //
    sz_i32_t pre_sub, sz_i32_t up_main, sz_i32_t up_ins,                            //
    sz_i32_t left_main, sz_i32_t left_del,                                          //
    sz_error_cost_t sub_cost, sz_error_cost_t gap_open, sz_error_cost_t gap_extend, //
    sz_similarity_objective_t obj, int is_local,                                    //
    sz_i32_t *out_ins, sz_i32_t *out_del) {
    sz_i32_t ins = sz_pick_best_i32(up_main + (sz_i32_t)gap_open, up_ins + (sz_i32_t)gap_extend, obj);
    sz_i32_t del = sz_pick_best_i32(left_main + (sz_i32_t)gap_open, left_del + (sz_i32_t)gap_extend, obj);
    sz_i32_t best = pre_sub + (sz_i32_t)sub_cost;
    best = sz_pick_best_i32(best, ins, obj);
    best = sz_pick_best_i32(best, del, obj);
    if (is_local && obj == sz_maximize_score_k && best < 0) best = 0;
    *out_ins = ins;
    *out_del = del;
    return best;
}

/** @brief Boundary value for the linear-gap main matrix at distance @p index from the origin. */
static inline sz_i32_t sz_init_score_linear(sz_size_t index, sz_error_cost_t gap, int is_local) {
    if (is_local) return 0;
    return (sz_i32_t)index * (sz_i32_t)gap;
}

/** @brief Boundary value for the affine-gap main matrix at distance @p index (open then extend). */
static inline sz_i32_t sz_init_score_affine(sz_size_t index, sz_error_cost_t gap_open, sz_error_cost_t gap_extend,
                                            int is_local) {
    if (is_local || index == 0) return 0;
    return (sz_i32_t)gap_open + (sz_i32_t)(index - 1) * (sz_i32_t)gap_extend;
}

/** @brief Boundary value for an affine gap track (kept saturating-low so it never wins on the edge). */
static inline sz_i32_t sz_init_gap_affine(sz_size_t index, sz_error_cost_t gap_open, sz_error_cost_t gap_extend,
                                          int is_local) {
    if (is_local) return 0;
    if (index == 0) return (sz_i32_t)gap_open;
    return (sz_i32_t)gap_open + (sz_i32_t)index * (sz_i32_t)gap_extend;
}

/** @brief Triangle / band / triangle anti-diagonal geometry of an (shorter+1) x (longer+1) DP grid. */
typedef struct sz_antidiagonal_geometry_t {
    sz_size_t shorter_dim;         ///< shorter_length + 1.
    sz_size_t longer_dim;          ///< longer_length + 1.
    sz_size_t diagonals_count;     ///< shorter_dim + longer_dim - 1.
    sz_size_t max_diagonal_length; ///< Length of the longest (middle) anti-diagonal == shorter_dim.
} sz_antidiagonal_geometry_t;

/** @brief Build the anti-diagonal geometry from the two string lengths (in characters/runes). */
static inline sz_antidiagonal_geometry_t sz_antidiagonal_geometry(sz_size_t shorter_length, sz_size_t longer_length) {
    sz_antidiagonal_geometry_t geom;
    geom.shorter_dim = shorter_length + 1;
    geom.longer_dim = longer_length + 1;
    geom.diagonals_count = geom.shorter_dim + geom.longer_dim - 1;
    geom.max_diagonal_length = geom.shorter_dim;
    return geom;
}

/** @brief The cells covered by one anti-diagonal: its length and the inclusive grid-row span. */
typedef struct sz_antidiagonal_slice_t {
    sz_size_t length;    ///< Number of cells on this anti-diagonal.
    sz_size_t row_first; ///< Grid row of the first (top-most) cell on this anti-diagonal.
    sz_size_t row_last;  ///< Grid row of the last (bottom-most) cell on this anti-diagonal.
} sz_antidiagonal_slice_t;

/** @brief Slice geometry for anti-diagonal @p diag_index in [0, diagonals_count) of @p geom. */
static inline sz_antidiagonal_slice_t sz_antidiagonal_slice(sz_antidiagonal_geometry_t geom, sz_size_t diag_index) {
    sz_antidiagonal_slice_t slice;
    // The first cell's row is the largest of 0 and (diag_index - (longer_dim - 1)).
    sz_size_t row_first = diag_index + 1 > geom.longer_dim ? diag_index + 1 - geom.longer_dim : 0;
    // The last cell's row is the smallest of diag_index and (shorter_dim - 1).
    sz_size_t row_last = diag_index < geom.shorter_dim ? diag_index : geom.shorter_dim - 1;
    slice.row_first = row_first;
    slice.row_last = row_last;
    slice.length = row_last - row_first + 1;
    return slice;
}

/** @brief 3-buffer ping-pong: previous-previous <- previous <- current <- (old previous-previous). */
static inline void sz_rotate3_ptr(void **previous_previous, void **previous, void **current) {
    void *spare = *previous_previous;
    *previous_previous = *previous;
    *previous = *current;
    *current = spare;
}

/** @brief Cell width selected for a DP pass, in bytes (0 => no work, strings empty). */
typedef enum sz_bytes_per_cell_t {
    sz_zero_bytes_per_cell_k = 0,
    sz_one_byte_per_cell_k = 1,
    sz_two_bytes_per_cell_k = 2,
    sz_four_bytes_per_cell_k = 4,
    sz_eight_bytes_per_cell_k = 8,
} sz_bytes_per_cell_t;

/** @brief Scratch-memory layout & sizing for one DP pass; lifted from `similarity_memory_requirements`. */
typedef struct sz_dp_memory_t {
    sz_size_t max_diagonal_length;      ///< Longest anti-diagonal == min(len_a,len_b) + 1.
    sz_bytes_per_cell_t bytes_per_cell; ///< Width chosen so the largest reachable cell value fits.
    sz_size_t bytes_per_diagonal;       ///< One diagonal buffer, rounded up to @p register_width.
    sz_size_t total;                    ///< Total scratch bytes (diagonals + the two char copies).
} sz_dp_memory_t;

/** @brief Round @p value up to the next multiple of @p multiple (>= 1). */
static inline sz_size_t sz_round_up_to_multiple(sz_size_t value, sz_size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

/**
 *  @brief Compute DP scratch requirements; mirrors `serial.hpp::similarity_memory_requirements`.
 *  @param a_len,b_len String lengths in characters/runes.
 *  @param gaps Linear vs affine (selects 3 vs 7 diagonal buffers).
 *  @param sub_magnitude,gap_magnitude Absolute maximum change a single cell can absorb.
 *  @param bytes_per_char 1 for ASCII/bytes, 4 for UTF-32 runes.
 *  @param register_width Diagonal alignment: 4 for CUDA, 64 for AVX-512, etc.
 *  @param is_signed Non-zero for score families (i16/i32/i64), zero for distance families (u8..u64).
 */
static inline sz_dp_memory_t sz_dp_memory(                       //
    sz_size_t a_len, sz_size_t b_len, sz_similarity_gaps_t gaps, //
    sz_size_t sub_magnitude, sz_size_t gap_magnitude,            //
    sz_size_t bytes_per_char, sz_size_t register_width, int is_signed) {
    sz_dp_memory_t out;
    out.max_diagonal_length = 0;
    out.bytes_per_cell = sz_zero_bytes_per_cell_k;
    out.bytes_per_diagonal = 0;
    out.total = 0;

    sz_size_t shorter_length = sz_min_of_two(a_len, b_len);
    if (shorter_length == 0) return out; // Empty input needs no DP scratch.

    sz_size_t longer_length = sz_max_of_two(a_len, b_len);
    out.max_diagonal_length = shorter_length + 1;

    sz_size_t magnitude = sz_max_of_two(sub_magnitude, gap_magnitude);
    sz_size_t max_cell_value = (longer_length + 1) * magnitude;
    if (!is_signed)
        out.bytes_per_cell = max_cell_value < 256           ? sz_one_byte_per_cell_k
                             : max_cell_value < 65536       ? sz_two_bytes_per_cell_k
                             : max_cell_value < 4294967296u ? sz_four_bytes_per_cell_k
                                                            : sz_eight_bytes_per_cell_k;
    else
        out.bytes_per_cell = max_cell_value < 127           ? sz_one_byte_per_cell_k
                             : max_cell_value < 32767       ? sz_two_bytes_per_cell_k
                             : max_cell_value < 2147483647u ? sz_four_bytes_per_cell_k
                                                            : sz_eight_bytes_per_cell_k;

    out.bytes_per_diagonal = sz_round_up_to_multiple(out.max_diagonal_length * (sz_size_t)out.bytes_per_cell,
                                                     register_width);

    // Linear gaps need 3 diagonal buffers; affine needs 3 main + 2x2 gap-track buffers = 7.
    sz_size_t diagonals_count = gaps == sz_gaps_linear_k ? 3 : 7;
    sz_size_t first_length_bytes = sz_round_up_to_multiple(a_len * bytes_per_char, register_width);
    sz_size_t second_length_bytes = sz_round_up_to_multiple(b_len * bytes_per_char, register_width);
    out.total = diagonals_count * out.bytes_per_diagonal + first_length_bytes + second_length_bytes;
    return out;
}

/** @brief Map a raw byte to its substitution-class index via the LUT's `byte_to_class` remap. */
static inline sz_u8_t sz_subs_class_of_byte(sz_substitution_costs_t const *costs, sz_u8_t byte) {
    return costs->byte_to_class[byte];
}

/** @brief Substitution cost between two class indices (already resolved via `sz_subs_class_of_byte`). */
static inline sz_error_cost_t sz_subs_cost(sz_substitution_costs_t const *costs, sz_u8_t class_a, sz_u8_t class_b) {
    return costs->costs[class_a][class_b];
}

/**
 *  @brief Decode a UTF-8 byte string into UTF-32 runes; transcode prologue for the rune-level kernels.
 *  @param utf8,utf8_length The encoded input bytes.
 *  @param runes Output buffer with capacity for at least @p utf8_length runes.
 *  @param runes_count Receives the decoded rune count.
 *  @return Non-zero on success, zero on invalid UTF-8 (matching `sz_rune_parse`'s `sz_utf8_invalid_k`).
 */
static inline int sz_utf8_to_runes(sz_cptr_t utf8, sz_size_t utf8_length, sz_rune_t *runes, sz_size_t *runes_count) {
    sz_size_t progress_utf8 = 0, progress_runes = 0;
    while (progress_utf8 < utf8_length) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(utf8 + progress_utf8, &rune, &rune_length);
        if (rune_length == sz_utf8_invalid_k) return 0;
        runes[progress_runes] = rune;
        progress_utf8 += rune_length;
        ++progress_runes;
    }
    *runes_count = progress_runes;
    return 1;
}

#pragma endregion

#pragma region - Shared Cost Context

typedef struct szs_cell_costs_t {
    int is_uniform;                              // Non-zero => use match/mismatch; else the LUT.
    sz_error_cost_t match;                       // Uniform-cost match (Levenshtein, typically 0).
    sz_error_cost_t mismatch;                    // Uniform-cost mismatch (Levenshtein, typically 1).
    sz_substitution_costs_t const *substitution; // Class-based LUT (NW/SW); NULL for uniform.
    sz_error_cost_t gap_open;                    // Gap-open cost.
    sz_error_cost_t gap_extend;                  // Gap-extend cost (== open for linear gaps).
    sz_similarity_objective_t objective;         // Minimize (Levenshtein) vs maximize (NW/SW).
    int is_local;                                // Smith-Waterman clamp-to-zero on maximize.
} szs_cell_costs_t;

static inline sz_error_cost_t szs_cell_subs_cost_(szs_cell_costs_t const *costs, sz_u32_t a, sz_u32_t b) {
    if (costs->is_uniform) return a == b ? costs->match : costs->mismatch;
    sz_u8_t class_a = sz_subs_class_of_byte(costs->substitution, (sz_u8_t)a);
    sz_u8_t class_b = sz_subs_class_of_byte(costs->substitution, (sz_u8_t)b);
    return sz_subs_cost(costs->substitution, class_a, class_b);
}

#pragma endregion

#pragma region - Diagonal Megakernels (i32 cells, u32 elements)

/*  One explicit anti-diagonal sweep for linear gaps: upper-left triangle, central band, bottom-right
 *  triangle, with the recurrence inlined via `sz_cell_linear_i32` and the 3-buffer ping-pong via
 *  `sz_rotate3_ptr`. Order of operations matches the reference Wagner-Fischer DP exactly.
 */
SZ_INTERNAL sz_status_t szs_diagonal_linear_serial_(sz_u32_t const *first, sz_size_t first_length,
                                                    sz_u32_t const *second, sz_size_t second_length,
                                                    szs_cell_costs_t const *costs, sz_memory_allocator_t *alloc,
                                                    sz_i32_t *result) {
    sz_error_cost_t const gap = costs->gap_open;
    if (first_length == 0 || second_length == 0) {
        sz_i32_t value = 0;
        if (!costs->is_local) {
            if (first_length != 0) value = (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)first_length);
            else if (second_length != 0) value = (sz_i32_t)((sz_i64_t)gap * (sz_i64_t)second_length);
        }
        *result = value;
        return sz_success_k;
    }

    sz_u32_t const *shorter = first;
    sz_u32_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_u32_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const buffer_length = sizeof(sz_i32_t) * max_diagonal_length * 3 + shorter_length * sizeof(sz_u32_t);
    sz_i32_t *buffer = (sz_i32_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_i32_t *previous_scores = buffer;
    sz_i32_t *current_scores = previous_scores + max_diagonal_length;
    sz_i32_t *next_scores = current_scores + max_diagonal_length;
    sz_u32_t *shorter_reversed = (sz_u32_t *)(next_scores + max_diagonal_length);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = sz_init_score_linear(0, gap, costs->is_local);
    current_scores[0] = sz_init_score_linear(1, gap, costs->is_local);
    current_scores[1] = sz_init_score_linear(1, gap, costs->is_local);
    sz_i32_t last_score = current_scores[1];
    sz_i32_t best_score = 0;
    sz_size_t next_diagonal_index = 2;

    // Upper-left triangle: each diagonal grows by one cell.
    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_u32_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        for (sz_size_t index = 0; index < interior_length; ++index) {
            sz_error_cost_t sub = szs_cell_subs_cost_(costs, first_reversed[index], longer[index]);
            sz_i32_t cell = sz_cell_linear_i32(previous_scores[index], current_scores[index], current_scores[index + 1],
                                               sub, gap, costs->objective, costs->is_local);
            next_scores[index + 1] = cell;
            best_score = sz_pick_best_i32(best_score, cell, costs->objective);
        }
        next_scores[0] = sz_init_score_linear(next_diagonal_index, gap, costs->is_local);
        next_scores[next_diagonal_length - 1] = sz_init_score_linear(next_diagonal_index, gap, costs->is_local);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
    }

    // Central band: every diagonal spans the full shorter dimension.
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_u32_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u32_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        for (sz_size_t index = 0; index < interior_length; ++index) {
            sz_error_cost_t sub = szs_cell_subs_cost_(costs, first_reversed[index], second_slice[index]);
            sz_i32_t cell = sz_cell_linear_i32(previous_scores[index], current_scores[index], current_scores[index + 1],
                                               sub, gap, costs->objective, costs->is_local);
            next_scores[index] = cell;
            best_score = sz_pick_best_i32(best_score, cell, costs->objective);
        }
        next_scores[next_diagonal_length - 1] = sz_init_score_linear(next_diagonal_index, gap, costs->is_local);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    // Bottom-right triangle: each diagonal shrinks by one cell.
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u32_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u32_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        for (sz_size_t index = 0; index < next_diagonal_length; ++index) {
            sz_error_cost_t sub = szs_cell_subs_cost_(costs, first_reversed[index], second_slice[index]);
            sz_i32_t cell = sz_cell_linear_i32(previous_scores[index], current_scores[index], current_scores[index + 1],
                                               sub, gap, costs->objective, costs->is_local);
            next_scores[index] = cell;
            best_score = sz_pick_best_i32(best_score, cell, costs->objective);
        }
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        previous_scores++;
    }

    *result = costs->is_local ? best_score : last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

/*  Affine-gap counterpart: seven diagonals (scores + running inserts/deletes) instead of three, with
 *  the recurrence inlined via `sz_cell_affine_i32` which also emits the next insert/delete cells.
 */
SZ_INTERNAL sz_status_t szs_diagonal_affine_serial_(sz_u32_t const *first, sz_size_t first_length,
                                                    sz_u32_t const *second, sz_size_t second_length,
                                                    szs_cell_costs_t const *costs, sz_memory_allocator_t *alloc,
                                                    sz_i32_t *result) {
    sz_error_cost_t const open = costs->gap_open, extend = costs->gap_extend;
    if (first_length == 0 || second_length == 0) {
        sz_i32_t value = 0;
        if (!costs->is_local) {
            if (first_length != 0) value = (sz_i32_t)((sz_i64_t)open + (sz_i64_t)extend * (sz_i64_t)(first_length - 1));
            else if (second_length != 0)
                value = (sz_i32_t)((sz_i64_t)open + (sz_i64_t)extend * (sz_i64_t)(second_length - 1));
        }
        *result = value;
        return sz_success_k;
    }

    sz_u32_t const *shorter = first;
    sz_u32_t const *longer = second;
    sz_size_t shorter_length = first_length, longer_length = second_length;
    if (shorter_length > longer_length) {
        sz_u32_t const *swap_pointer = shorter;
        shorter = longer, longer = swap_pointer;
        sz_size_t swap_length = shorter_length;
        shorter_length = longer_length, longer_length = swap_length;
    }

    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;
    sz_size_t const buffer_length = sizeof(sz_i32_t) * max_diagonal_length * 7 + shorter_length * sizeof(sz_u32_t);
    sz_i32_t *buffer = (sz_i32_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return sz_bad_alloc_k;

    sz_i32_t *previous_scores = buffer;
    sz_i32_t *current_scores = previous_scores + max_diagonal_length;
    sz_i32_t *next_scores = current_scores + max_diagonal_length;
    sz_i32_t *current_inserts = next_scores + max_diagonal_length;
    sz_i32_t *next_inserts = current_inserts + max_diagonal_length;
    sz_i32_t *current_deletes = next_inserts + max_diagonal_length;
    sz_i32_t *next_deletes = current_deletes + max_diagonal_length;
    sz_u32_t *shorter_reversed = (sz_u32_t *)(next_deletes + max_diagonal_length);
    for (sz_size_t index = 0; index != shorter_length; ++index)
        shorter_reversed[index] = shorter[shorter_length - 1 - index];

    previous_scores[0] = sz_init_score_affine(0, open, extend, costs->is_local);
    current_scores[0] = sz_init_score_affine(1, open, extend, costs->is_local);
    current_scores[1] = sz_init_score_affine(1, open, extend, costs->is_local);
    current_inserts[0] = sz_init_gap_affine(1, open, extend, costs->is_local);
    current_deletes[1] = sz_init_gap_affine(1, open, extend, costs->is_local);
    sz_i32_t last_score = current_scores[1];
    sz_i32_t best_score = 0;
    sz_size_t next_diagonal_index = 2;

    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        sz_u32_t const *first_reversed = shorter_reversed + shorter_length - next_diagonal_index + 1;
        sz_size_t const interior_length = next_diagonal_length - 2;
        for (sz_size_t index = 0; index < interior_length; ++index) {
            sz_error_cost_t sub = szs_cell_subs_cost_(costs, first_reversed[index], longer[index]);
            sz_i32_t out_insert, out_delete;
            sz_i32_t cell = sz_cell_affine_i32(previous_scores[index], current_scores[index], current_inserts[index],
                                               current_scores[index + 1], current_deletes[index + 1], sub, open, extend,
                                               costs->objective, costs->is_local, &out_insert, &out_delete);
            next_scores[index + 1] = cell;
            next_inserts[index + 1] = out_insert;
            next_deletes[index + 1] = out_delete;
            best_score = sz_pick_best_i32(best_score, cell, costs->objective);
        }
        next_scores[0] = sz_init_score_affine(next_diagonal_index, open, extend, costs->is_local);
        next_scores[next_diagonal_length - 1] = sz_init_score_affine(next_diagonal_index, open, extend,
                                                                     costs->is_local);
        next_inserts[0] = sz_init_gap_affine(next_diagonal_index, open, extend, costs->is_local);
        next_deletes[next_diagonal_length - 1] = sz_init_gap_affine(next_diagonal_index, open, extend, costs->is_local);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_i32_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
    }

    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        sz_u32_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u32_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        sz_size_t const interior_length = next_diagonal_length - 1;
        for (sz_size_t index = 0; index < interior_length; ++index) {
            sz_error_cost_t sub = szs_cell_subs_cost_(costs, first_reversed[index], second_slice[index]);
            sz_i32_t out_insert, out_delete;
            sz_i32_t cell = sz_cell_affine_i32(previous_scores[index], current_scores[index], current_inserts[index],
                                               current_scores[index + 1], current_deletes[index + 1], sub, open, extend,
                                               costs->objective, costs->is_local, &out_insert, &out_delete);
            next_scores[index] = cell;
            next_inserts[index] = out_insert;
            next_deletes[index] = out_delete;
            best_score = sz_pick_best_i32(best_score, cell, costs->objective);
        }
        next_scores[next_diagonal_length - 1] = sz_init_score_affine(next_diagonal_index, open, extend,
                                                                     costs->is_local);
        next_deletes[next_diagonal_length - 1] = sz_init_gap_affine(next_diagonal_index, open, extend, costs->is_local);
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_i32_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
        for (sz_size_t index = 0; index + 1 < max_diagonal_length; ++index)
            previous_scores[index] = previous_scores[index + 1];
    }

    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        sz_u32_t const *first_reversed = shorter_reversed + shorter_length - shorter_dim + 1;
        sz_u32_t const *second_slice = longer + next_diagonal_index - shorter_dim;
        for (sz_size_t index = 0; index < next_diagonal_length; ++index) {
            sz_error_cost_t sub = szs_cell_subs_cost_(costs, first_reversed[index], second_slice[index]);
            sz_i32_t out_insert, out_delete;
            sz_i32_t cell = sz_cell_affine_i32(previous_scores[index], current_scores[index], current_inserts[index],
                                               current_scores[index + 1], current_deletes[index + 1], sub, open, extend,
                                               costs->objective, costs->is_local, &out_insert, &out_delete);
            next_scores[index] = cell;
            next_inserts[index] = out_insert;
            next_deletes[index] = out_delete;
            best_score = sz_pick_best_i32(best_score, cell, costs->objective);
        }
        last_score = next_scores[next_diagonal_length - 1];
        sz_rotate3_ptr((void **)&previous_scores, (void **)&current_scores, (void **)&next_scores);
        sz_i32_t *swap = current_inserts;
        current_inserts = next_inserts, next_inserts = swap;
        swap = current_deletes, current_deletes = next_deletes, next_deletes = swap;
        previous_scores++;
    }

    *result = costs->is_local ? best_score : last_score;
    alloc->free(buffer, buffer_length, alloc->handle);
    return sz_success_k;
}

#pragma endregion

#pragma region - Input Decoding

typedef struct szs_pair_spans_t {
    sz_cptr_t a_ptr;
    sz_size_t a_len;
    sz_cptr_t b_ptr;
    sz_size_t b_len;
} szs_pair_spans_t;

static inline sz_size_t szs_inputs_count_(szs_sequence_view_t const *view) {
    switch (view->kind) {
    case szs_inputs_sequence_k: return view->first.sequence->count;
    case szs_inputs_u32tape_k: return view->first.u32tape->count;
    case szs_inputs_u64tape_k: return view->first.u64tape->count;
    default: return 0;
    }
}

static inline szs_pair_spans_t szs_inputs_pair_(szs_sequence_view_t const *view, sz_size_t i) {
    szs_pair_spans_t out;
    switch (view->kind) {
    case szs_inputs_sequence_k: {
        sz_sequence_t const *sa = view->first.sequence;
        sz_sequence_t const *sb = view->second.sequence;
        out.a_ptr = sa->get_start(sa->handle, i);
        out.a_len = sa->get_length(sa->handle, i);
        out.b_ptr = sb->get_start(sb->handle, i);
        out.b_len = sb->get_length(sb->handle, i);
        break;
    }
    case szs_inputs_u32tape_k: {
        sz_sequence_u32tape_t const *ta = view->first.u32tape;
        sz_sequence_u32tape_t const *tb = view->second.u32tape;
        out.a_ptr = ta->data + ta->offsets[i];
        out.a_len = ta->offsets[i + 1] - ta->offsets[i];
        out.b_ptr = tb->data + tb->offsets[i];
        out.b_len = tb->offsets[i + 1] - tb->offsets[i];
        break;
    }
    case szs_inputs_u64tape_k: {
        sz_sequence_u64tape_t const *ta = view->first.u64tape;
        sz_sequence_u64tape_t const *tb = view->second.u64tape;
        out.a_ptr = ta->data + ta->offsets[i];
        out.a_len = ta->offsets[i + 1] - ta->offsets[i];
        out.b_ptr = tb->data + tb->offsets[i];
        out.b_len = tb->offsets[i + 1] - tb->offsets[i];
        break;
    }
    default: out.a_ptr = 0, out.a_len = 0, out.b_ptr = 0, out.b_len = 0; break;
    }
    return out;
}

static inline sz_memory_allocator_t *szs_resolve_alloc_(sz_memory_allocator_t *engine_alloc,
                                                        sz_memory_allocator_t *fallback) {
    if (engine_alloc && engine_alloc->allocate) return engine_alloc;
    sz_memory_allocator_init_default(fallback);
    return fallback;
}

#pragma endregion

#pragma region - Per-Op Dispatch Entries

/*  One per-pair body for all four operations: decode the pair (bytes or runes), widen to `sz_u32_t`, run one
 *  diagonal megakernel, then widen the i32 result into the caller's signed/unsigned output slot. The
 *  fork_union callback ABI is the single sanctioned `void*` boundary; everything else is typed.
 */
typedef struct szs_pairwise_ctx_t {
    szs_sequence_view_t inputs;   // Typed input collections + kind.
    szs_cell_costs_t costs;       // Resolved cost model + objective/local flags.
    sz_memory_allocator_t *alloc; // Per-call scratch allocator.
    int decode_runes;             // Non-zero => UTF-8 decode (rune-level Levenshtein).
    int linear;                   // Non-zero => linear-gap recurrence, else affine.
    int results_signed;           // 0 => write `sz_size_t` distance, 1 => write `sz_ssize_t` score.
    void *results;                // Base of the caller's output plane.
    sz_size_t results_stride;     // Byte stride between consecutive outputs.
    char const **error;           // Optional error-message out-pointer.
} szs_pairwise_ctx_t;

static sz_status_t szs_pairwise_body_(void *ctx_punned, sz_size_t i) {
    szs_pairwise_ctx_t *ctx = (szs_pairwise_ctx_t *)ctx_punned;
    sz_memory_allocator_t *alloc = ctx->alloc;
    szs_pair_spans_t pair = szs_inputs_pair_(&ctx->inputs, i);

    // Widen both inputs to `sz_u32_t` elements (bytes zero-extend; UTF-8 decodes to runes).
    sz_size_t const widen_capacity = pair.a_len + pair.b_len;
    sz_size_t const alloc_bytes = (widen_capacity ? widen_capacity : 1) * sizeof(sz_u32_t);
    sz_u32_t *elements = (sz_u32_t *)alloc->allocate(alloc_bytes, alloc->handle);
    if (!elements) {
        if (ctx->error) *ctx->error = "Pairwise element buffer allocation failed";
        return sz_bad_alloc_k;
    }
    sz_u32_t *a_elements = elements;
    sz_u32_t *b_elements = elements + pair.a_len;
    sz_size_t a_count = 0, b_count = 0;
    if (ctx->decode_runes) {
        if (!sz_utf8_to_runes(pair.a_ptr, pair.a_len, a_elements, &a_count) ||
            !sz_utf8_to_runes(pair.b_ptr, pair.b_len, b_elements, &b_count)) {
            alloc->free(elements, alloc_bytes, alloc->handle);
            if (ctx->error) *ctx->error = "Invalid UTF-8 input";
            return sz_invalid_utf8_k;
        }
    }
    else {
        sz_u8_t const *ap = (sz_u8_t const *)pair.a_ptr, *bp = (sz_u8_t const *)pair.b_ptr;
        for (sz_size_t k = 0; k < pair.a_len; ++k) a_elements[k] = ap[k];
        for (sz_size_t k = 0; k < pair.b_len; ++k) b_elements[k] = bp[k];
        a_count = pair.a_len, b_count = pair.b_len;
    }

    sz_i32_t score = 0;
    sz_status_t status =
        ctx->linear ? szs_diagonal_linear_serial_(a_elements, a_count, b_elements, b_count, &ctx->costs, alloc, &score)
                    : szs_diagonal_affine_serial_(a_elements, a_count, b_elements, b_count, &ctx->costs, alloc, &score);
    alloc->free(elements, alloc_bytes, alloc->handle);
    if (status != sz_success_k) {
        if (ctx->error) *ctx->error = "Pairwise serial kernel failed";
        return status;
    }
    if (ctx->results_signed) *(sz_ssize_t *)((sz_u8_t *)ctx->results + i * ctx->results_stride) = (sz_ssize_t)score;
    else *(sz_size_t *)((sz_u8_t *)ctx->results + i * ctx->results_stride) = (sz_size_t)(sz_u32_t)score;
    return sz_success_k;
}

/*  Levenshtein presets: uniform match/mismatch, minimize distance, global, unsigned result. An affine engine
 *  whose open == extend collapses to the linear recurrence. The byte and UTF-8 entries differ only in
 *  `decode_runes`. */
SZ_INTERNAL void szs_levenshtein_preset_(szs_pairwise_ctx_t *ctx, szs_levenshtein_engine_t const *engine,
                                         int decode_runes) {
    int const is_affine = engine->gap_mode == sz_gaps_affine_k;
    int const affine_collapses = is_affine && engine->gaps.open == engine->gaps.extend;
    ctx->costs.is_uniform = 1;
    ctx->costs.match = engine->substitution.match;
    ctx->costs.mismatch = engine->substitution.mismatch;
    ctx->costs.substitution = SZ_NULL;
    ctx->costs.gap_open = engine->gaps.open;
    ctx->costs.gap_extend = affine_collapses ? engine->gaps.open : engine->gaps.extend;
    ctx->costs.objective = sz_minimize_distance_k;
    ctx->costs.is_local = 0;
    ctx->linear = !is_affine || affine_collapses;
    ctx->decode_runes = decode_runes;
    ctx->results_signed = 0;
}

SZ_INTERNAL sz_status_t szs_levenshtein_serial(szs_levenshtein_engine_t const *engine, szs_device_scope_impl_t *device,
                                               szs_sequence_view_t inputs, sz_size_t *results, sz_size_t results_stride,
                                               char const **error) {
    sz_memory_allocator_t fallback_alloc;
    szs_pairwise_ctx_t ctx;
    ctx.inputs = inputs;
    ctx.alloc = szs_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc, &fallback_alloc);
    szs_levenshtein_preset_(&ctx, engine, /*decode_runes*/ 0);
    ctx.results = results, ctx.results_stride = results_stride, ctx.error = error;
    return szs_parallel_for_(device, szs_inputs_count_(&inputs), &szs_pairwise_body_, &ctx);
}

SZ_INTERNAL sz_status_t szs_levenshtein_utf8_serial(szs_levenshtein_engine_t const *engine,
                                                    szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                    sz_size_t *results, sz_size_t results_stride, char const **error) {
    sz_memory_allocator_t fallback_alloc;
    szs_pairwise_ctx_t ctx;
    ctx.inputs = inputs;
    ctx.alloc = szs_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc, &fallback_alloc);
    szs_levenshtein_preset_(&ctx, engine, /*decode_runes*/ 1);
    ctx.results = results, ctx.results_stride = results_stride, ctx.error = error;
    return szs_parallel_for_(device, szs_inputs_count_(&inputs), &szs_pairwise_body_, &ctx);
}

/*  Alignment presets: class-based LUT, maximize score, signed result. NW is global (is_local=0), SW local
 *  (is_local=1). NW & SW carry distinct engine types but share field layout, so one filler reads the common
 *  cost fields from either via the shared prefix. */
SZ_INTERNAL void szs_alignment_preset_(szs_pairwise_ctx_t *ctx, sz_substitution_costs_t const *substitution,
                                       szs_gap_costs_t gaps, sz_similarity_gaps_t gap_mode, int is_local) {
    ctx->costs.is_uniform = 0;
    ctx->costs.match = 0, ctx->costs.mismatch = 0;
    ctx->costs.substitution = substitution;
    ctx->costs.gap_open = gaps.open;
    ctx->costs.gap_extend = gaps.extend;
    ctx->costs.objective = sz_maximize_score_k;
    ctx->costs.is_local = is_local;
    ctx->linear = gap_mode != sz_gaps_affine_k; // NW/SW do not auto-collapse affine to linear.
    ctx->decode_runes = 0;
    ctx->results_signed = 1;
}

SZ_INTERNAL sz_status_t szs_needleman_wunsch_serial(szs_needleman_wunsch_engine_t const *engine,
                                                    szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                    sz_ssize_t *results, sz_size_t results_stride, char const **error) {
    sz_memory_allocator_t fallback_alloc;
    szs_pairwise_ctx_t ctx;
    ctx.inputs = inputs;
    ctx.alloc = szs_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc, &fallback_alloc);
    szs_alignment_preset_(&ctx, &engine->substitution, engine->gaps, engine->gap_mode, /*is_local*/ 0);
    ctx.results = results, ctx.results_stride = results_stride, ctx.error = error;
    return szs_parallel_for_(device, szs_inputs_count_(&inputs), &szs_pairwise_body_, &ctx);
}

SZ_INTERNAL sz_status_t szs_smith_waterman_serial(szs_smith_waterman_engine_t const *engine,
                                                  szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                  sz_ssize_t *results, sz_size_t results_stride, char const **error) {
    sz_memory_allocator_t fallback_alloc;
    szs_pairwise_ctx_t ctx;
    ctx.inputs = inputs;
    ctx.alloc = szs_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc, &fallback_alloc);
    szs_alignment_preset_(&ctx, &engine->substitution, engine->gaps, engine->gap_mode, /*is_local*/ 1);
    ctx.results = results, ctx.results_stride = results_stride, ctx.error = error;
    return szs_parallel_for_(device, szs_inputs_count_(&inputs), &szs_pairwise_body_, &ctx);
}

#pragma endregion

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLAS_SIMILARITIES_SERIAL_H_
