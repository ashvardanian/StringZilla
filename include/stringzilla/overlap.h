/**
 *  @brief Window overlap between a query and a batch of candidates, at one or more n-gram widths.
 *  @file include/stringzilla/overlap.h
 *  @author Ash Vardanian
 *
 *  A window is a fixed-width byte n-gram, and the overlap of two texts at that width is the count of one's
 *  windows that occur in the other, over whichever of the two has more. The query is hashed once into one
 *  B-tree over every width's window hashes and every candidate then streams through it, hashing its own
 *  windows and probing. Nothing is stored per candidate, so the widths may be chosen per corpus, or per pair.
 *
 *  Window hashes come from a prefix difference, @c H(i,w) = P(i+w) - P(i)·b^w mod p, which costs one
 *  modular multiply-add at any width and any offset; widths need not form a doubling chain. The modulus is
 *  a 32-bit prime chosen for collision weight, so a window hash is full-width.
 *
 *  @section overlap_api Public API
 *
 *  - @ref sz_overlap_score → one query against one candidate, one score per width;
 *  - @ref sz_overlap_scores → one query against many candidates, one score per candidate per width.
 *
 *  Every register-level step behind these - the prefix chain, the window hashing, the key sort and the B-tree
 *  probe - is public per backend in @c overlap/serial.h and its SIMD backends, named by positions per step -
 *  @c sz_overlap_f64x4_prefix_hash_step_haswell and so on - so a caller driving its own loops binds those steps
 *  and skips these verbs entirely. The tree layout is the same on every backend, so any of them may probe it.
 */
#ifndef STRINGZILLA_OVERLAP_H_
#define STRINGZILLA_OVERLAP_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Window overlap of @p query with every candidate, one score per candidate per width.
 *
 *  Every width's query window hashes share one B-tree, so the sort runs once per query and the probe depth grows
 *  only logarithmically with the widths added; a candidate window hash carries its own width, so a cross-width
 *  coincidence costs @c 2^-32.
 *
 *  @param[in] query The text whose windows are sorted into the B-tree.
 *  @param[in] query_length Number of bytes in the query.
 *  @param[in] candidates The collection of texts whose windows are probed against the query's.
 *  @param[in] window_widths Window widths in bytes; a width past a text scores zero for that pair.
 *  @param[in] window_widths_count Number of widths; each candidate's scores follow that order.
 *  @param[in] alloc Where one round's scratch comes from; freed before returning.
 *  @param[out] scores The @b [candidates,window_widths] shares, each in @c [0,1], row-major.
 *
 *  @retval sz_success_k if every score was computed.
 *  @retval sz_unexpected_dimensions_k for zero widths.
 *  @retval sz_bad_alloc_k when the round's scratch cannot be taken.
 *  @pre The @p scores array must fit at least @c candidates->count × @p window_widths_count entries.
 *  @note Asymmetric: a candidate's window occurrences count against the query's distinct windows, so swapping
 *      sides changes the score when either repeats a window.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_overlap_scores_serial, sz_overlap_scores_haswell, sz_overlap_scores_skylake
 */
SZ_API_RUNTIME sz_status_t sz_overlap_scores(sz_cptr_t query, sz_size_t query_length, sz_sequence_t const *candidates,
                                             sz_size_t const *window_widths, sz_size_t window_widths_count,
                                             sz_memory_allocator_t *alloc, sz_f32_t *scores);

/**
 *  @brief Window overlap of @p query and @p candidate, one score per width.
 *
 *  @param[in] query The text whose windows are sorted into the B-tree.
 *  @param[in] query_length Number of bytes in the query.
 *  @param[in] candidate The text whose windows are probed against the query's.
 *  @param[in] candidate_length Number of bytes in the candidate.
 *  @param[in] window_widths Window widths in bytes; a width past either text scores zero.
 *  @param[in] window_widths_count Number of widths; the scores follow that order.
 *  @param[in] alloc Where one round's scratch comes from; freed before returning.
 *  @param[out] scores The @b [window_widths] shares, each in @c [0,1], in the order @p window_widths lists them.
 *
 *  @retval sz_success_k if every score was computed.
 *  @retval sz_unexpected_dimensions_k for zero widths.
 *  @retval sz_bad_alloc_k when the round's scratch cannot be taken.
 *  @note Asymmetric: a candidate's window occurrences count against the query's distinct windows, so swapping
 *      sides changes the score when either repeats a window.
 *  @sa sz_overlap_scores
 */
SZ_API_RUNTIME sz_status_t sz_overlap_score(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                            sz_size_t candidate_length, sz_size_t const *window_widths,
                                            sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                            sz_f32_t *scores);

/** @copydoc sz_overlap_scores */
SZ_API_COMPTIME sz_status_t sz_overlap_scores_serial(sz_cptr_t query, sz_size_t query_length,
                                                     sz_sequence_t const *candidates, sz_size_t const *window_widths,
                                                     sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                     sz_f32_t *scores);

#if SZ_USE_HASWELL
/** @copydoc sz_overlap_scores */
SZ_API_COMPTIME sz_status_t sz_overlap_scores_haswell(sz_cptr_t query, sz_size_t query_length,
                                                      sz_sequence_t const *candidates, sz_size_t const *window_widths,
                                                      sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                      sz_f32_t *scores);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_overlap_scores */
SZ_API_COMPTIME sz_status_t sz_overlap_scores_skylake(sz_cptr_t query, sz_size_t query_length,
                                                      sz_sequence_t const *candidates, sz_size_t const *window_widths,
                                                      sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                      sz_f32_t *scores);
#endif

#if SZ_USE_CUDA
/**
 *  @copydoc sz_overlap_scores
 *  @note Takes the CPU backends' arguments exactly, so it binds wherever they do, and asks nothing of the caller
 *      about where its memory lives: whatever the device cannot reach is staged into memory it can, scored, and
 *      read back before returning. A caller already holding its texts and scores on the device pays none of that
 *      - the round then runs in place - so the staging is a fallback rather than a step.
 *  @sa sz_overlap_scores_scheduled_cuda to run on a chosen stream, where staging is refused rather than performed.
 */
SZ_API_COMPTIME sz_status_t sz_overlap_scores_cuda(sz_cptr_t query, sz_size_t query_length,
                                                   sz_sequence_t const *candidates, sz_size_t const *window_widths,
                                                   sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                   sz_f32_t *scores);

/**
 *  @brief Window overlap of @p query with every candidate on one GPU, scheduled on the caller's stream.
 *
 *  Nothing crosses the bus: @p candidates ' texts are read where they already are, and @p scores and whatever
 *  @p alloc hands back are written the same way, so the round costs one launch rather than a round trip. Host
 *  memory is refused rather than copied - page-locked host memory counts as host memory here.
 *
 *  @pre @p candidates carries @b device accessors, as @ref sz_sequence_from_string_views_cuda binds them, because
 *      the kernel is what calls them - one call per candidate, uniform across the warp. Only the handle can be
 *      checked from this side, so host accessors reach the device as an invalid address rather than a status.
 *
 *  @param[in] stream A @c cudaStream_t the caller owns and keeps, or zero for the current device's default one.
 *      The round is synchronous either way - @p scores is filled before returning - but only this stream is
 *      waited on, so the caller's other work on the device keeps running.
 *  @param[in] alloc Where the tree and the per-candidate arrays come from; must itself reach the device, and is
 *      freed before returning.
 *  @param[out] scores The @b [candidates,window_widths] shares, each in @c [0,1], row-major.
 *
 *  @retval sz_unexpected_dimensions_k for no widths, more than @ref sz_overlap_cuda_widths_max_k of them, or one
 *      past @ref sz_overlap_cuda_widest_window_k, which is the per-thread ring's compile-time bound.
 *  @retval sz_device_memory_mismatch_k when the scores, the first candidate, or the scratch is host memory.
 *  @retval sz_device_code_mismatch_k when the launch itself fails, the stream reporting it at the join.
 *  @sa sz_overlap_scores_cuda for the same round on the default stream.
 */
SZ_API_COMPTIME sz_status_t sz_overlap_scores_scheduled_cuda(
    sz_cptr_t query, sz_size_t query_length, sz_sequence_t const *candidates, sz_size_t const *window_widths,
    sz_size_t window_widths_count, sz_memory_allocator_t *alloc, sz_f32_t *scores, void *stream);
#endif

/** @copydoc sz_overlap_score */
SZ_API_COMPTIME sz_status_t sz_overlap_score_serial(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                                    sz_size_t candidate_length, sz_size_t const *window_widths,
                                                    sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                    sz_f32_t *scores);

#if SZ_USE_HASWELL
/** @copydoc sz_overlap_score */
SZ_API_COMPTIME sz_status_t sz_overlap_score_haswell(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                                     sz_size_t candidate_length, sz_size_t const *window_widths,
                                                     sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                     sz_f32_t *scores);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_overlap_score */
SZ_API_COMPTIME sz_status_t sz_overlap_score_skylake(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                                     sz_size_t candidate_length, sz_size_t const *window_widths,
                                                     sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                     sz_f32_t *scores);
#endif

#if SZ_USE_CUDA
/** @copydoc sz_overlap_score */
SZ_API_COMPTIME sz_status_t sz_overlap_score_cuda(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                                  sz_size_t candidate_length, sz_size_t const *window_widths,
                                                  sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                  sz_f32_t *scores);
#endif

#pragma endregion Core API

#include "stringzilla/overlap/serial.h"
#include "stringzilla/overlap/haswell.h"
#include "stringzilla/overlap/skylake.h"
#include "stringzilla/overlap/cuda.cuh"

/*  Pick the right implementation for the window overlap algorithms.
 *  To override this behavior and precompile all backends - set @c SZ_DYNAMIC_DISPATCH to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_status_t sz_overlap_scores(sz_cptr_t query, sz_size_t query_length, sz_sequence_t const *candidates,
                                             sz_size_t const *window_widths, sz_size_t window_widths_count,
                                             sz_memory_allocator_t *alloc, sz_f32_t *scores) {
#if SZ_USE_CUDA
    return sz_overlap_scores_cuda(query, query_length, candidates, window_widths, window_widths_count, alloc, scores);
#elif SZ_USE_SKYLAKE
    return sz_overlap_scores_skylake(query, query_length, candidates, window_widths, window_widths_count, alloc,
                                     scores);
#elif SZ_USE_HASWELL
    return sz_overlap_scores_haswell(query, query_length, candidates, window_widths, window_widths_count, alloc,
                                     scores);
#else
    return sz_overlap_scores_serial(query, query_length, candidates, window_widths, window_widths_count, alloc, scores);
#endif
}

SZ_API_RUNTIME sz_status_t sz_overlap_score(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                            sz_size_t candidate_length, sz_size_t const *window_widths,
                                            sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                            sz_f32_t *scores) {
#if SZ_USE_CUDA
    return sz_overlap_score_cuda(query, query_length, candidate, candidate_length, window_widths, window_widths_count,
                                 alloc, scores);
#elif SZ_USE_SKYLAKE
    return sz_overlap_score_skylake(query, query_length, candidate, candidate_length, window_widths,
                                    window_widths_count, alloc, scores);
#elif SZ_USE_HASWELL
    return sz_overlap_score_haswell(query, query_length, candidate, candidate_length, window_widths,
                                    window_widths_count, alloc, scores);
#else
    return sz_overlap_score_serial(query, query_length, candidate, candidate_length, window_widths, window_widths_count,
                                   alloc, scores);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_OVERLAP_H_
