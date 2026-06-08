/**
 *  @file include/stringzillas/fingerprints/serial.h
 *  @brief Plain-C serial rolling-hash Min-Hash / Count-Min fingerprint kernels.
 *  @author Ash Vardanian
 *
 *  The serial backend for the fingerprint operation, plus the shared floating Rabin-Karp / Barrett math the
 *  SIMD backends reuse (folded in from the former `fingerprint_kernels.h`). Fingerprints have NO
 *  gap/width/locality/objective axes; the only real kernel boundary is capability.
 *
 *  The per-backend SWEEP is a single megakernel because the SIMD specializations vectorize across the
 *  dimensions axis (a memory-layout change). One engine carries multiple window widths, each owning a
 *  contiguous slice of the total dimensions; the dispatch entry walks those slices, seeding each with the
 *  matching `first_dimension_offset` so every dimension gets a distinct multiplier.
 */
#ifndef STRINGZILLAS_FINGERPRINTS_SERIAL_H_
#define STRINGZILLAS_FINGERPRINTS_SERIAL_H_

#include "stringzillas/types.h"        // Engine PODs, `szs_sequence_view_t`
#include "stringzillas/stringzillas.h" // `sz_sequence_t`, `sz_sequence_u32tape_t`, `sz_sequence_u64tape_t`
#include "stringzillas/parallel.h"     // `szs_parallel_for_`: serial-or-fork_union per-text loop

#include "stringzilla/types.h" // `sz_f64_t`, `sz_u32_t`, `sz_u64_t`, `sz_size_t`, `sz_byte_t`

#include <math.h> // `floor`, `fma`, `fmod`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region - Floating Rabin-Karp Constants

/** @brief The largest integer exactly representable as an IEEE-754 f64 (2^52 - 1). */
#define SZ_FH_LIMIT_K 4503599627370495.0
/** @brief Default alphabet cardinality - the 256 possible values of a single byte. */
#define SZ_FH_DEFAULT_ALPHABET_SIZE_K 256u
/** @brief Default modulo base: largest prime under `limit/1000 - 257`. */
#define SZ_FH_DEFAULT_MODULO_BASE_K 4503599626977u
/** @brief Sentinel for a rolling minimum that never captured (window never filled). */
#define SZ_FH_SKIPPED_ROLLING_HASH_K 1.7976931348623157e308 // `DBL_MAX`
/** @brief All-ones 32-bit Min-Hash sentinel for short inputs / skipped minimums. */
#define SZ_FH_MAX_HASH_K 0xFFFFFFFFu

/**
 *  @brief Per-dimension rolling-hash coefficients, mirroring `floating_rolling_hasher<f64_t>`.
 *  @note `negative_discarding_multiplier` is already negated, ready for the discarding FMA.
 */
typedef struct sz_fh_dim_t {
    sz_f64_t multiplier;                     ///< Polynomial base for this dimension.
    sz_f64_t modulo;                         ///< Reduction modulo for this dimension.
    sz_f64_t inverse_modulo;                 ///< `1.0 / modulo`, precomputed for Barrett reduction.
    sz_f64_t negative_discarding_multiplier; ///< `-(multiplier^(window_width-1) mod modulo)`.
} sz_fh_dim_t;

#pragma endregion

#pragma region - Floating Rabin-Karp Seeding & Barrett Reduction

/**
 *  @brief Seeds one dimension's coefficients exactly like the `floating_rolling_hasher<f64_t>` ctor.
 *  @param window_width Width of the rolling window (must be > 1).
 *  @param multiplier_int Integer multiplier (alphabet_size + first_dimension_offset + dim).
 *  @param modulo_int Integer modulo base (typically `SZ_FH_DEFAULT_MODULO_BASE_K`).
 *
 *  Reproduces the ctor's `negative_discarding_multiplier_` accumulation with `std::fmod`:
 *      for (i+1 < window_width): acc = fmod(acc * multiplier, modulo); then acc = -acc.
 */
SZ_INTERNAL sz_fh_dim_t sz_fh_seed_dim_(sz_size_t window_width, sz_u64_t multiplier_int, sz_u64_t modulo_int) {
    sz_fh_dim_t dim;
    sz_f64_t const multiplier = (sz_f64_t)multiplier_int;
    sz_f64_t const modulo = (sz_f64_t)modulo_int;
    sz_f64_t discarding = 1.0;
    sz_size_t step_index;
    dim.multiplier = multiplier;
    dim.modulo = modulo;
    dim.inverse_modulo = 1.0 / modulo;
    for (step_index = 0; step_index + 1 < window_width; ++step_index)
        discarding = fmod(discarding * multiplier, modulo);
    dim.negative_discarding_multiplier = -discarding;
    return dim;
}

/**
 *  @brief Scalar Barrett reduction matching `floating_rolling_hashers<serial>::barrett_mod`.
 *
 *  Uses `std::floor`-based quotient, a plain `x - q*modulo` (NOT a fused FMA), and predicate-scaled
 *  clamp adds so the serial C kernel reproduces the C++ serial sweep bit-for-bit.
 */
SZ_INTERNAL sz_f64_t sz_fh_barrett_mod(sz_f64_t x, sz_f64_t modulo, sz_f64_t inverse_modulo) {
    sz_f64_t const q = floor(x * inverse_modulo);
    sz_f64_t result = x - q * modulo;
    result += modulo * (sz_f64_t)(result < 0.0);
    result -= modulo * (sz_f64_t)(result >= modulo);
    return result;
}

/** @brief Add-head term step: `barrett_mod(fma(state, multiplier, new_term))`, matching the serial sweep. */
SZ_INTERNAL sz_f64_t sz_fh_push(sz_f64_t state, sz_fh_dim_t const *dim, sz_f64_t new_term) {
    sz_f64_t pushed = fma(state, dim->multiplier, new_term);
    return sz_fh_barrett_mod(pushed, dim->modulo, dim->inverse_modulo);
}

/**
 *  @brief Two-step roll: discard the tail term, reduce, then add the new head term, reduce.
 *  @note Matches the serial sweep's exact operation order: discard-then-mod, add-then-mod.
 */
SZ_INTERNAL sz_f64_t sz_fh_roll(sz_f64_t state, sz_fh_dim_t const *dim, sz_f64_t old_term, sz_f64_t new_term) {
    sz_f64_t without_old = fma(dim->negative_discarding_multiplier, old_term, state);
    without_old = sz_fh_barrett_mod(without_old, dim->modulo, dim->inverse_modulo);
    {
        sz_f64_t with_new = fma(without_old, dim->multiplier, new_term);
        return sz_fh_barrett_mod(with_new, dim->modulo, dim->inverse_modulo);
    }
}

#pragma endregion

#pragma region - Serial Sweep Megakernel

SZ_INTERNAL void szs_fingerprints_sweep_serial_(          //
    sz_byte_t const *text, sz_size_t text_length,         //
    sz_fh_dim_t const *dims, sz_size_t dimensions,        //
    sz_size_t window_width,                               //
    sz_f64_t *rolling_states, sz_f64_t *rolling_minimums, //
    sz_u32_t *min_hashes, sz_u32_t *min_counts) {

    sz_size_t dim;
    sz_size_t new_char_offset = 0;
    sz_size_t const prefix_length = text_length < window_width ? text_length : window_width;

    // Short-input shortcut, exactly like the C++ `fingerprint` early exit.
    if (text_length < window_width) {
        for (dim = 0; dim < dimensions; ++dim) min_hashes[dim] = SZ_FH_MAX_HASH_K;
        for (dim = 0; dim < dimensions; ++dim) min_counts[dim] = 0;
        return;
    }

    for (dim = 0; dim < dimensions; ++dim)
        rolling_states[dim] = 0.0, rolling_minimums[dim] = SZ_FH_SKIPPED_ROLLING_HASH_K;

    // Branching prefix: until the window fills, we only push the new head.
    for (; new_char_offset < prefix_length; ++new_char_offset) {
        sz_byte_t const new_char = text[new_char_offset];
        sz_f64_t const new_term = (sz_f64_t)new_char + 1.0;
        for (dim = 0; dim < dimensions; ++dim)
            rolling_states[dim] = sz_fh_push(rolling_states[dim], &dims[dim], new_term);
    }

    // First minimum hashes captured once the window is exactly full.
    if (new_char_offset == window_width)
        for (dim = 0; dim < dimensions; ++dim) rolling_minimums[dim] = rolling_states[dim], min_counts[dim] = 1;

    // Branchless central loop: roll past the window width.
    for (; new_char_offset < text_length; ++new_char_offset) {
        sz_byte_t const new_char = text[new_char_offset];
        sz_byte_t const old_char = text[new_char_offset - window_width];
        sz_f64_t const new_term = (sz_f64_t)new_char + 1.0;
        sz_f64_t const old_term = (sz_f64_t)old_char + 1.0;
        for (dim = 0; dim < dimensions; ++dim) {
            sz_f64_t const state = sz_fh_roll(rolling_states[dim], &dims[dim], old_term, new_term);
            sz_f64_t const rolling_minimum = rolling_minimums[dim];
            rolling_states[dim] = state;
            // Branchless Count-Min: discard count to 0 on a strictly new minimum, increment on new & old minimums.
            min_counts[dim] *= (sz_u32_t)(state >= rolling_minimum);
            min_counts[dim] += (sz_u32_t)(state <= rolling_minimum);
            rolling_minimums[dim] = state < rolling_minimum ? state : rolling_minimum;
        }
    }

    // Digest the rolling minimums into the 32-bit Min-Hash plane.
    for (dim = 0; dim < dimensions; ++dim) {
        sz_f64_t const rolling_minimum = rolling_minimums[dim];
        sz_u64_t const rolling_minimum_as_uint = (sz_u64_t)rolling_minimum;
        min_hashes[dim] = rolling_minimum == SZ_FH_SKIPPED_ROLLING_HASH_K
                              ? SZ_FH_MAX_HASH_K
                              : (sz_u32_t)(rolling_minimum_as_uint & SZ_FH_MAX_HASH_K);
    }
}

#pragma endregion

#pragma region - Input Decoding

typedef struct szs_text_span_t {
    sz_cptr_t ptr;
    sz_size_t len;
} szs_text_span_t;

static inline sz_size_t szs_fingerprints_count_(szs_sequence_view_t const *view) {
    switch (view->kind) {
    case szs_inputs_sequence_k: return view->first.sequence->count;
    case szs_inputs_u32tape_k: return view->first.u32tape->count;
    case szs_inputs_u64tape_k: return view->first.u64tape->count;
    default: return 0;
    }
}

static inline szs_text_span_t szs_fingerprints_text_(szs_sequence_view_t const *view, sz_size_t i) {
    szs_text_span_t out;
    switch (view->kind) {
    case szs_inputs_sequence_k: {
        sz_sequence_t const *s = view->first.sequence;
        out.ptr = s->get_start(s->handle, i);
        out.len = s->get_length(s->handle, i);
        break;
    }
    case szs_inputs_u32tape_k: {
        sz_sequence_u32tape_t const *t = view->first.u32tape;
        out.ptr = t->data + t->offsets[i];
        out.len = t->offsets[i + 1] - t->offsets[i];
        break;
    }
    case szs_inputs_u64tape_k: {
        sz_sequence_u64tape_t const *t = view->first.u64tape;
        out.ptr = t->data + t->offsets[i];
        out.len = t->offsets[i + 1] - t->offsets[i];
        break;
    }
    default: out.ptr = 0, out.len = 0; break;
    }
    return out;
}

static inline sz_memory_allocator_t *szs_fingerprints_resolve_alloc_(sz_memory_allocator_t *engine_alloc,
                                                                     sz_memory_allocator_t *fallback) {
    if (engine_alloc && engine_alloc->allocate) return engine_alloc;
    sz_memory_allocator_init_default(fallback);
    return fallback;
}

#pragma endregion

#pragma region - Per-Backend Dispatch Helpers

SZ_INTERNAL void szs_fingerprints_group_geometry_( //
    sz_size_t group_index, sz_size_t dimensions,   //
    sz_size_t window_widths_count,                 //
    sz_size_t *out_first_dim, sz_size_t *out_group_dims) {
    sz_size_t const base = dimensions / window_widths_count;
    sz_size_t const remainder = dimensions % window_widths_count;
    sz_size_t first_dim = group_index * base + (group_index < remainder ? group_index : remainder);
    sz_size_t group_dims = base + (group_index < remainder ? 1 : 0);
    *out_first_dim = first_dim;
    *out_group_dims = group_dims;
}

SZ_INTERNAL void szs_fingerprints_seed_group_(                    //
    sz_fh_dim_t *dims, sz_size_t first_dim, sz_size_t group_dims, //
    sz_size_t window_width, sz_size_t alphabet_size) {
    sz_size_t dim;
    for (dim = 0; dim < group_dims; ++dim)
        dims[dim] = sz_fh_seed_dim_(window_width, (sz_u64_t)(alphabet_size + first_dim + dim),
                                    SZ_FH_DEFAULT_MODULO_BASE_K);
}

#pragma endregion

#pragma region - Serial Entry

// Per-window-width sweep signature shared by serial/haswell/skylake backends.
typedef void (*szs_fingerprints_sweep_fn_t)(sz_byte_t const *text, sz_size_t text_len, sz_fh_dim_t const *dims,
                                            sz_size_t group_dims, sz_size_t window_width, sz_f64_t *rolling_states,
                                            sz_f64_t *rolling_minimums, sz_u32_t *out_hashes, sz_u32_t *out_counts);

typedef struct szs_fingerprints_ctx_t {
    szs_fingerprints_engine_t const *eng;
    szs_sequence_view_t inputs;
    sz_memory_allocator_t *alloc;
    sz_size_t dimensions;
    sz_size_t window_widths_count;
    sz_size_t alphabet_size;
    szs_fingerprints_sweep_fn_t sweep;
    sz_u32_t *min_hashes;
    sz_size_t min_hashes_stride;
    sz_u32_t *min_counts;
    sz_size_t min_counts_stride;
    char const **error;
} szs_fingerprints_ctx_t;

static sz_status_t szs_fingerprints_body_(void *ctx_punned, sz_size_t text_index) {
    szs_fingerprints_ctx_t *ctx = (szs_fingerprints_ctx_t *)ctx_punned;
    sz_memory_allocator_t *alloc = ctx->alloc;
    sz_size_t const dimensions = ctx->dimensions;
    sz_size_t const window_widths_count = ctx->window_widths_count;
    sz_size_t const alphabet_size = ctx->alphabet_size;

    sz_size_t const dims_bytes = dimensions * sizeof(sz_fh_dim_t);
    sz_size_t const states_bytes = dimensions * sizeof(sz_f64_t);
    sz_size_t const scratch_bytes = dims_bytes + 2 * states_bytes;
    sz_byte_t *scratch = (sz_byte_t *)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) {
        if (ctx->error) *ctx->error = "Fingerprint scratch allocation failed";
        return sz_bad_alloc_k;
    }
    {
        sz_fh_dim_t *dims = (sz_fh_dim_t *)scratch;
        sz_f64_t *rolling_states = (sz_f64_t *)(scratch + dims_bytes);
        sz_f64_t *rolling_minimums = rolling_states + dimensions;
        szs_text_span_t const text = szs_fingerprints_text_(&ctx->inputs, text_index);
        sz_u32_t *text_hashes = (sz_u32_t *)((sz_byte_t *)ctx->min_hashes + text_index * ctx->min_hashes_stride);
        sz_u32_t *text_counts = (sz_u32_t *)((sz_byte_t *)ctx->min_counts + text_index * ctx->min_counts_stride);
        sz_size_t group_index;
        for (group_index = 0; group_index < window_widths_count; ++group_index) {
            sz_size_t first_dim, group_dims;
            szs_fingerprints_group_geometry_(group_index, dimensions, window_widths_count, &first_dim, &group_dims);
            if (group_dims == 0) continue;
            szs_fingerprints_seed_group_(dims, first_dim, group_dims, ctx->eng->window_widths[group_index],
                                         alphabet_size);
            ctx->sweep((sz_byte_t const *)text.ptr, text.len, dims, group_dims, ctx->eng->window_widths[group_index],
                       rolling_states, rolling_minimums, text_hashes + first_dim, text_counts + first_dim);
        }
    }
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return sz_success_k;
}

static inline szs_fingerprints_ctx_t szs_fingerprints_make_ctx_( //
    szs_fingerprints_engine_t const *eng, szs_sequence_view_t inputs, sz_memory_allocator_t *alloc,
    szs_fingerprints_sweep_fn_t sweep, sz_u32_t *min_hashes, sz_size_t min_hashes_stride, sz_u32_t *min_counts,
    sz_size_t min_counts_stride, char const **error) {
    szs_fingerprints_ctx_t ctx;
    ctx.eng = eng;
    ctx.inputs = inputs;
    ctx.alloc = alloc;
    ctx.dimensions = eng->dimensions;
    ctx.window_widths_count = eng->window_widths_count;
    ctx.alphabet_size = eng->alphabet_size ? eng->alphabet_size : SZ_FH_DEFAULT_ALPHABET_SIZE_K;
    ctx.sweep = sweep;
    ctx.min_hashes = min_hashes;
    ctx.min_hashes_stride = min_hashes_stride;
    ctx.min_counts = min_counts;
    ctx.min_counts_stride = min_counts_stride;
    ctx.error = error;
    return ctx;
}

SZ_INTERNAL sz_status_t szs_fingerprints_serial(                            //
    szs_fingerprints_engine_t const *engine, szs_device_scope_impl_t *device, //
    szs_sequence_view_t inputs,                                             //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                      //
    sz_u32_t *min_counts, sz_size_t min_counts_stride,                      //
    char const **error) {

    sz_memory_allocator_t fallback_alloc;
    sz_memory_allocator_t *alloc = szs_fingerprints_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc,
                                                                   &fallback_alloc);
    sz_size_t const count = szs_fingerprints_count_(&inputs);
    szs_fingerprints_ctx_t ctx = szs_fingerprints_make_ctx_(engine, inputs, alloc, &szs_fingerprints_sweep_serial_,
                                                            min_hashes, min_hashes_stride, min_counts,
                                                            min_counts_stride, error);
    return szs_parallel_for_(device, count, &szs_fingerprints_body_, &ctx);
}

#pragma endregion

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLAS_FINGERPRINTS_SERIAL_H_
