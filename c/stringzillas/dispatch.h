/**
 *  @file c/stringzillas/dispatch.h
 *  @brief Shared dispatch table for the StringZillas similarity & fingerprint shims.
 *  @author Ash Vardanian
 *
 *  Mirrors `c/stringzilla/dispatch.h`: the compiled StringZillas library is split into one
 *  translation unit per domain (`similarities.c`, `fingerprints.c`), each filling its slice of the
 *  shared `szs_dispatch_table` via `szs_dispatch_<op>_update_` and defining the `SZ_DYNAMIC` public
 *  wrappers that call through the table. The thin `runtime.c` owns the table definition & one-time
 *  init. One table slot per op covers both gap modes and all widths; the width branch lives inside
 *  each entry (it is a per-call, data-dependent runtime choice from string lengths x cost magnitude).
 */
#ifndef STRINGZILLAS_DISPATCH_H_
#define STRINGZILLAS_DISPATCH_H_

// Reuse the C core's dispatch machinery verbatim: this sets `SZ_DYNAMIC_DISPATCH`, defines
// `SZ_DISPATCH_INTERNAL`, and pulls in `<stringzilla/types.h>`. We only add the StringZillas engine PODs
// and the parallel dispatch table on top.
#include "../stringzilla/dispatch.h"
#include "stringzillas/types.h" // `szs_*` engine/scope PODs, `szs_inputs_kind_t`

#ifdef __cplusplus
extern "C" {
#endif

/*  Each entry receives the concrete engine POD, the device-scope POD, the typed input view, the results
 *  plane(s), and an optional error-message out-pointer. Engine & scope are separate pointers exactly as in
 *  the public `szs_*` C API.
 */

/** @brief Levenshtein distance entry (min/global); results are unsigned `sz_size_t`. */
typedef sz_status_t (*szs_levenshtein_fn_t)(          //
    szs_levenshtein_engine_t const *engine,           //
    szs_device_scope_impl_t *device,                  //
    szs_sequence_view_t inputs,                       //
    sz_size_t *results, sz_size_t results_stride,     //
    char const **error);

/** @brief UTF-8 Levenshtein distance entry (min/global); identical shape, rune-level. */
typedef sz_status_t (*szs_levenshtein_utf8_fn_t)(     //
    szs_levenshtein_engine_t const *engine,           //
    szs_device_scope_impl_t *device,                  //
    szs_sequence_view_t inputs,                       //
    sz_size_t *results, sz_size_t results_stride,     //
    char const **error);

/** @brief Needleman-Wunsch global alignment entry (max/global); results are signed `sz_ssize_t`. */
typedef sz_status_t (*szs_needleman_wunsch_fn_t)(     //
    szs_needleman_wunsch_engine_t const *engine,      //
    szs_device_scope_impl_t *device,                  //
    szs_sequence_view_t inputs,                       //
    sz_ssize_t *results, sz_size_t results_stride,    //
    char const **error);

/** @brief Smith-Waterman local alignment entry (max/local); results are signed `sz_ssize_t`. */
typedef sz_status_t (*szs_smith_waterman_fn_t)(       //
    szs_smith_waterman_engine_t const *engine,        //
    szs_device_scope_impl_t *device,                  //
    szs_sequence_view_t inputs,                       //
    sz_ssize_t *results, sz_size_t results_stride,    //
    char const **error);

/**
 *  @brief Fingerprint entry; single input collection (the view's @c first), two output planes.
 *  @note Named `szs_fingerprints_fn_t` to avoid clashing with the public `szs_fingerprints_t` handle.
 */
typedef sz_status_t (*szs_fingerprints_fn_t)(          //
    szs_fingerprints_engine_t const *engine,           //
    szs_device_scope_impl_t *device,                   //
    szs_sequence_view_t inputs,                        //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride, //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, //
    char const **error);

/**
 *  @brief The global "virtual table" of supported backends, defined in `runtime.c`
 *         and populated by the per-op updaters below.
 */
typedef struct szs_implementations_t {
    szs_levenshtein_fn_t levenshtein;           ///< min/global.
    szs_levenshtein_utf8_fn_t levenshtein_utf8; ///< min/global (rune-level).
    szs_needleman_wunsch_fn_t needleman_wunsch; ///< max/global.
    szs_smith_waterman_fn_t smith_waterman;     ///< max/local.
    szs_fingerprints_fn_t fingerprints;         ///< Min-Hash + Count-Min-Sketch.
#if SZ_USE_CUDA
    // The loaded CUmodule/CUfunctions live in the host launcher's own state; here we only record which GPU
    // tier was selected so the public wrappers can route default/gpu scopes to the GPU entries.
    sz_capability_t gpu_tier; ///< Selected GPU tier: cuda | kepler | hopper.
#endif
} szs_implementations_t;

extern SZ_DISPATCH_INTERNAL szs_implementations_t szs_dispatch_table;

/*  Each updater fills only its own slot(s), defaulting to the serial backend and then overriding for
 *  the most capable enabled SIMD/GPU generation matching @p caps (same cascade as `c/stringzilla/find.c`).
 */
SZ_DISPATCH_INTERNAL void szs_dispatch_levenshtein_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void szs_dispatch_levenshtein_utf8_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void szs_dispatch_needleman_wunsch_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void szs_dispatch_smith_waterman_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void szs_dispatch_fingerprints_update_(sz_capability_t caps);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLAS_DISPATCH_H_
