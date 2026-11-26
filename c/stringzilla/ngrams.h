/**
 *  @file c/stringzilla/ngrams.h
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief N-gram overlaps of prepared queries with batches of candidates, and MinHash sketches of
 *      batches of texts, at one or more N-gram widths.
 *
 *  An N-gram is a run of a fixed number of bytes, hashed to a residue under one 32-bit prime. A
 *  window's residue is a prefix difference, R(i, w) = P(i + w) − P(i) × bʷ mod p, one modular
 *  multiply-add at any width and any offset, so widths need not form a doubling chain. The modulus
 *  is a 32-bit prime chosen for collision weight, so a residue is full-width. Every backend reduces
 *  the same integers, so a residue, an overlap and a sketch are the same on every capability.
 *
 *  @section ngrams_overlaps Overlaps
 *
 *  The overlap of a query with a candidate at a width is the share of the query's distinct N-grams
 *  that the candidate holds. Every query is filed once into one table of its distinct N-grams at
 *  every width, and every candidate streams through the tables, hashing its own N-grams and
 *  probing. Nothing is stored per candidate, so the candidates may change round to round while the
 *  queries and the widths stay.
 *
 *  @section ngrams_sketches Sketches
 *
 *  A sketch is one MinHash sketch per text, summarizing the set of its N-grams at the engine's
 *  widths in as many one-byte slots as the caller asks. The share of positions where two sketches
 *  differ estimates one minus the Jaccard similarity of their N-gram sets, and more slots give a
 *  finer estimate. Only sketches of one slot count compare. The construction and the estimate's
 *  error are laid out in the family's README, beside the kernels.
 *
 *  @section ngrams_api Public API
 *
 *  - @ref sz_ngrams_engine_init → prepares a batch of queries, or none for an engine that only
 *    sketches, for the best capability of a mask, on the host or on one device;
 *  - @ref sz_ngrams_overlaps → the @b [queries,candidates,widths] counts of one round;
 *  - @ref sz_ngrams_sketches → the sketch of every text of a batch, ranked in the engine's scratch;
 *  - @ref sz_ngrams_engine_free → returns the engine's blocks;
 *  - @ref sz_ngrams_find_kernel → the kernel any of them would run, for a caller that keeps it.
 *
 *  For kits driving their own loops, per CPU capability: @c sz_ngrams_residues_serial and its twins
 *  write every window's residue, and @c sz_ngrams_sketch_serial and its twins sketch one text with
 *  ranks the caller owns, so any number of threads sketch at once. A host engine's query tables are
 *  read through @ref sz_ngrams_engine_query, @ref sz_ngrams_query_probe, @ref sz_ngrams_slot_first
 *  and @ref sz_ngrams_slot_repeated.
 */
#ifndef STRINGZILLA_NGRAMS_H_
#define STRINGZILLA_NGRAMS_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`
#include "stringzilla/metal.h"
#include "stringzilla/ngrams/serial.h" // `sz_ngrams_engine_t`, `sz_ngrams_query_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Prepares @p queries into one block from @p allocator, for the best capability of
 *      @p capabilities that has an init kernel.
 *
 *  Every width's N-grams of one query share that query's table, so it is filled once per batch
 *  rather than once per round, and a probe stays a few slots long for any number of widths.
 *
 *  @param[out] engine Left untouched unless the call succeeds.
 *  @param[in] queries The texts whose N-grams are filed into the tables, read on the host, each
 *      at most @ref sz_ngrams_query_bytes_max_k bytes; null or empty for a sketch-only engine.
 *  @param[in] window_widths Window widths in bytes, each at least one; a width past a text holds no
 *      N-grams of it.
 *  @param[in] window_widths_count Number of widths, at most @ref sz_ngrams_widths_max_k, which is
 *      the last axis of every overlap output.
 *  @param[in] candidates_budget Candidates one round may carry, for a backend whose rounds keep
 *      state per candidate, zero asking for its default; one that keeps none ignores it.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu or
 *      @c sz_capabilities_enabled_metal report; its group picks the CPU or a GPU vendor.
 *  @param[in] allocator Source of the engine's blocks, or @c STRINGZILLA_NULL for
 *      @ref sz_allocator_init_unified_best of @p capabilities; stored by value, so
 *      @ref sz_ngrams_engine_free needs no allocator of its own.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, also naming the device:
 *      a @c cudaStream_t, a @c hipStream_t, or an @c id<MTLCommandQueue>; null for the default.
 *
 *  @note May join @p stream; no other verb ever does.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k No capability of the mask has an init.
 *  - @c sz_unexpected_dimensions_k Zero widths, too many, a width of zero, or a query too long; on
 *    a device, a query whose hit words outgrow one warp's shared memory.
 *  - @c sz_device_memory_mismatch_k On a device, memory or a @p stream it cannot use.
 */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                  sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                  sz_size_t candidates_budget, sz_capability_t capabilities,
                                                  sz_allocator_t *allocator, sz_stream_t stream);

/** Returns @p engine 's blocks to the allocator that built them, once the work queued on @p stream
 *  is done with them, and leaves it empty. */
STRINGZILLA_API void sz_ngrams_engine_free(sz_ngrams_engine_t *engine, sz_stream_t stream);

/**
 *  @brief Counts, per width, the distinct N-grams of every prepared query that every candidate
 *      holds, by the kernel of the capability @p engine was prepared for, exact within @p radius.
 *
 *  A count over the query's distinct N-grams at that width is the overlap, the share of the query
 *  the candidate holds. A batch of queries prepared once scores any number of candidate batches,
 *  and a host scratch block only grows, so a batch within the longest seen allocates nothing. A
 *  pair's misses are the query's distinct N-grams the candidate lacks, summed over every width,
 *  and its distance is those misses over the query's distinct N-grams in all. A pair within
 *  @p radius is counted exactly; a pair past it counts zero at every width, on every backend, or
 *  exactly when within rounding of the share, less than 2⁻²⁴ past it. A candidate with too few
 *  N-grams to come within that share is dropped from the two lengths alone, before a byte is read.
 *
 *  @param[in] engine The prepared tables; a host scratch block grows here and is never shrunk. A
 *      device engine keeps no stream and its rounds keep no state, so any number of streams or
 *      queues may score it at once.
 *  @param[in] candidates The texts whose N-grams are probed against the tables, on the residency
 *      the engine was built for: any layout on the CPU, and on a device a tape
 *      @ref sz_strings_to_tape_best wrote into a unified allocator.
 *  @param[in] radius The share of the query's distinct N-grams a pair may miss, in [0, 1]; 1 or
 *      more keeps every count exact.
 *  @param[out] matches Counts with a unit-strided width axis, laid out as shown below.
 *  @param[in] matches_query_stride Entries from one query's plane to the next, at least the
 *      candidates times @p matches_candidate_stride.
 *  @param[in] matches_candidate_stride Entries between candidate rows, at least the widths the
 *      engine was prepared at.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, which also names the device
 *      the round runs on; null for the default.
 *
 *  @note Asymmetric: a candidate holding the whole query counts every distinct query window
 *      however much more it holds, so swapping sides changes the count.
 *  @note Never joins, so the caller joins @p stream before reading @p matches.
 *
 *  The count of query q, candidate c and width w sits at:
 *
 *  @verbatim
 *  matches[q * matches_query_stride + c * matches_candidate_stride + w]
 *  @endverbatim
 *
 *  @par Statuses:
 *  - @c sz_unexpected_dimensions_k Either stride is under the axis it spans.
 *  - @c sz_missing_kernel_k An empty engine.
 *  - @c sz_device_memory_mismatch_k On a device, an engine, an output, an accessor, a tape or a
 *    stream the device of @p stream cannot use.
 */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                               sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                               sz_size_t matches_candidate_stride, sz_stream_t stream);

/**
 *  @brief One MinHash sketch per text of @p texts, summarizing the set of its N-grams at the widths
 *      of @p engine, by the kernel of the capability @p engine was prepared for.
 *
 *  The share of positions where two sketches differ estimates one minus the Jaccard similarity of
 *  their N-gram sets, and more slots give a finer estimate. Only same-size sketches compare.
 *
 *  @param[in] engine Any engine, queries or none; its scratch grows to 4 bytes a slot and is never
 *      shrunk, so a slot count within the widest seen allocates nothing.
 *  @param[in] texts The texts, on the residency the engine was built for: any layout on the CPU,
 *      and on a device a tape @ref sz_strings_to_tape_best wrote into a unified allocator.
 *  @param[in] slots Bytes of one sketch, at least one and below 2³².
 *  @param[out] sketches One sketch per text; sketch @c i starts at
 *      <tt>sketches + i * sketches_stride</tt>.
 *  @param[in] sketches_stride Bytes from one sketch to the next, at least @p slots.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, also naming the device.
 *
 *  @note On a device, never joins.
 *
 *  @par Statuses:
 *  - @c sz_unexpected_dimensions_k No slots, too many, or a stride under them.
 *  - @c sz_missing_kernel_k An empty engine.
 *  - @c sz_device_memory_mismatch_k On a device, an output, a tape or a stream the device of
 *    @p stream cannot use.
 */
STRINGZILLA_API sz_status_t sz_ngrams_sketches(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                               sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                               sz_stream_t stream);

/**
 *  @brief The residue of every window of one text at every width, width after width in position
 *      order, as @ref sz_ngrams_count counts them, on the serial kernels.
 *
 *  @param[in] text The text's first byte.
 *  @param[in] length Bytes of the text.
 *  @param[in] widths Window widths in bytes; a width of zero or past the text holds no N-grams.
 *  @param[in] widths_count Number of widths.
 *  @param[out] prefixes Working space of @p length + 1 prefix residues, left holding them.
 *  @param[out] residues The N-grams' residues, @ref sz_ngrams_count of them.
 *
 *  @note Keeps no state and allocates nothing, so any number of threads may call it at once.
 */
STRINGZILLA_API void sz_ngrams_residues_serial(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                               sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues);

/**
 *  @brief The MinHash sketch of one text, summarizing the set of its N-grams at every width in
 *      @p slots bytes, on the serial kernels.
 *
 *  @param[in] text The text's first byte.
 *  @param[in] length Bytes of the text.
 *  @param[in] widths Window widths in bytes, each at least one; a width past the text yields no
 *      N-grams of it.
 *  @param[in] widths_count Number of widths, at most @ref sz_ngrams_widths_max_k.
 *  @param[out] ranks Working space of @p slots ranks, 4 bytes a slot, overwritten.
 *  @param[in] slots Bytes of the sketch, at least one and below 2³².
 *  @param[out] sketch The @p slots bytes.
 *
 *  @return @c sz_success_k, or @c sz_unexpected_dimensions_k for no slots or too many, too many
 *      widths or a width of zero.
 *  @note Keeps no state and allocates nothing, so any number of threads may call it at once.
 */
STRINGZILLA_API sz_status_t sz_ngrams_sketch_serial(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                    sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots,
                                                    sz_u8_t *sketch);

/**
 *  @brief Prepares @p queries into one block from @p allocator, for the serial kernels.
 *  @param[in] allocator Source of the engine's blocks, or @c STRINGZILLA_NULL for the heap.
 *  @sa sz_ngrams_engine_init
 */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_serial(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                         sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_serial(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                      sz_f32_t radius, sz_u32_t *matches,
                                                      sz_size_t matches_query_stride,
                                                      sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_serial(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                      sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                      sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_ngrams_residues_serial */
STRINGZILLA_API void sz_ngrams_residues_haswell(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues);
/** @copydoc sz_ngrams_sketch_serial */
STRINGZILLA_API sz_status_t sz_ngrams_sketch_haswell(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                     sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots,
                                                     sz_u8_t *sketch);
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_haswell(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                          sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                          sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                          sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_haswell(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                       sz_f32_t radius, sz_u32_t *matches,
                                                       sz_size_t matches_query_stride,
                                                       sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_haswell(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                       sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                       sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SKYLAKE
/** @copydoc sz_ngrams_residues_serial */
STRINGZILLA_API void sz_ngrams_residues_skylake(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues);
/** @copydoc sz_ngrams_sketch_serial */
STRINGZILLA_API sz_status_t sz_ngrams_sketch_skylake(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                     sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots,
                                                     sz_u8_t *sketch);
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_skylake(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                          sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                          sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                          sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_skylake(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                       sz_f32_t radius, sz_u32_t *matches,
                                                       sz_size_t matches_query_stride,
                                                       sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_skylake(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                       sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                       sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_ngrams_residues_serial */
STRINGZILLA_API void sz_ngrams_residues_neon(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                             sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues);
/** @copydoc sz_ngrams_sketch_serial */
STRINGZILLA_API sz_status_t sz_ngrams_sketch_neon(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                  sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots,
                                                  sz_u8_t *sketch);
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_neon(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                       sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                       sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                       sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_neon(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                    sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                    sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_neon(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                    sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_ngrams_residues_serial */
STRINGZILLA_API void sz_ngrams_residues_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                 sz_size_t widths_count, sz_f64_t *prefixes, sz_u32_t *residues);
/** @copydoc sz_ngrams_sketch_serial */
STRINGZILLA_API sz_status_t sz_ngrams_sketch_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                      sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots,
                                                      sz_u8_t *sketch);
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_powervsx(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                           sz_size_t const *window_widths,
                                                           sz_size_t window_widths_count, sz_size_t candidates_budget,
                                                           sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_powervsx(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                        sz_f32_t radius, sz_u32_t *matches,
                                                        sz_size_t matches_query_stride,
                                                        sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_powervsx(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                        sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                        sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_cuda(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                       sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                       sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                       sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_cuda(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                    sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                    sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_cuda(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                    sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_AMPERE
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_ampere(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                         sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_ampere(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                      sz_f32_t radius, sz_u32_t *matches,
                                                      sz_size_t matches_query_stride,
                                                      sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_ampere(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                      sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                      sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_BLACKWELL
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_blackwell(sz_ngrams_engine_t *engine,
                                                            sz_strings_view_t const *queries,
                                                            sz_size_t const *window_widths,
                                                            sz_size_t window_widths_count, sz_size_t candidates_budget,
                                                            sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_blackwell(sz_ngrams_engine_t *engine,
                                                         sz_strings_view_t const *candidates, sz_f32_t radius,
                                                         sz_u32_t *matches, sz_size_t matches_query_stride,
                                                         sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_blackwell(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                         sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                         sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_rocm(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                       sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                       sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                       sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_rocm(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                    sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                    sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_rocm(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                    sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CDNA3
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_cdna3(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_cdna3(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                     sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                     sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_cdna3(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                     sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CDNA4
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_cdna4(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_cdna4(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                     sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                     sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_cdna4(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                     sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CDNA5
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_cdna5(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_cdna5(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                     sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                     sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_cdna5(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                     sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                     sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL
/** @copydoc sz_ngrams_engine_init */
STRINGZILLA_API sz_status_t sz_ngrams_engine_init_metal(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream);
/** @copydoc sz_ngrams_overlaps */
STRINGZILLA_API sz_status_t sz_ngrams_overlaps_metal(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                     sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                     sz_size_t matches_candidate_stride, sz_stream_t stream);
/** @copydoc sz_ngrams_sketches */
STRINGZILLA_API sz_status_t sz_ngrams_sketches_metal(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                     sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                     sz_stream_t stream);
#endif

/**
 *  @brief Finds the N-grams kernel of @p kind, from the best of @p capabilities.
 *  @param[in] kind @c sz_kernel_ngrams_engine_init_k, @c sz_kernel_ngrams_overlaps_k or
 *      @c sz_kernel_ngrams_sketches_k.
 *  @param[in] capabilities The mask to pick the best capability from.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k
 *  - @c sz_missing_library_k Header-only.
 */
STRINGZILLA_API sz_status_t sz_ngrams_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/ngrams/haswell.h"
#include "stringzilla/ngrams/skylake.h"
#include "stringzilla/ngrams/neon.h"
#include "stringzilla/ngrams/powervsx.h"
#include "stringzilla/ngrams/cuda.cuh"
#include "stringzilla/ngrams/blackwell.cuh"
#include "stringzilla/ngrams/rocm.cuh"
#include "stringzilla/ngrams/metal.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_ngrams_engine_init(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                  sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                  sz_size_t candidates_budget, sz_capability_t capabilities,
                                                  sz_allocator_t *allocator, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(queries), sz_unused_(window_widths), sz_unused_(window_widths_count),
        sz_unused_(candidates_budget), sz_unused_(capabilities), sz_unused_(allocator), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API void sz_ngrams_engine_free(sz_ngrams_engine_t *engine, sz_stream_t stream) {
    sz_ngrams_engine_close_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                               sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                               sz_size_t matches_candidate_stride, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(candidates), sz_unused_(radius), sz_unused_(matches),
        sz_unused_(matches_query_stride), sz_unused_(matches_candidate_stride), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                               sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                               sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(texts), sz_unused_(slots), sz_unused_(sketches), sz_unused_(sketches_stride),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_NGRAMS_H_
