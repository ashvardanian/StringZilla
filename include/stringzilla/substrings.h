/**
 *  @file include/stringzilla/substrings.h
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief Multi-pattern substring search: one compiled Aho-Corasick engine over many haystacks.
 *
 *  A vocabulary of needles compiles once into a byte-level automaton, and every haystack then
 *  streams through it in a single pass, whatever the needle count. The automaton is two-tiered: a
 *  dense goto-completed hot table for the states text keeps returning to, and a double array for
 *  the rest, because a trie's visit counts are skewed however large the vocabulary grows.
 *
 *  Case folding lives in the stream, not the automaton. A needle is folded once at insertion and
 *  inserted byte for byte, and an uncased haystack is folded as the walk consumes it, so both sides
 *  meet in one canonical byte stream and the trie stays a tree with single-valued failure links.
 *
 *  @section substrings_api Public API
 *
 *  - @ref sz_substrings_engine_init → compiles a vocabulary for the best capability of a mask, on
 *    the host or where one device's kernels reach it;
 *  - @ref sz_substrings_counts → how many matches each haystack holds;
 *  - @ref sz_substrings_find → every match, located by haystack, needle and byte span;
 *  - @ref sz_substrings_replace → the haystacks rewritten with one replacement per needle;
 *  - @ref sz_substrings_bm25_scores → one BM25 score per haystack, the vocabulary being the query;
 *  - @ref sz_substrings_find_kernel → the kernel any of them would run, for a caller that keeps it.
 *
 *  Each verb runs the kernel of the capability its engine was compiled for, on the host or enqueued
 *  on a stream of that engine's device. The engine and its types live in `substrings/serial.h`.
 *
 *  The engine is a plain struct of flat arrays over two owned blocks, so a caller driving its own
 *  loops binds @ref sz_substrings_step and walks it directly, and a device backend passes it to a
 *  kernel by value.
 */
#ifndef STRINGZILLA_SUBSTRINGS_H_
#define STRINGZILLA_SUBSTRINGS_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`
#include "stringzilla/metal.h"
#include "stringzilla/substrings/serial.h" // `sz_substrings_engine_t`, `sz_substrings_match_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Compiles @p needles for the best capability of @p capabilities that has an init kernel,
 *      and keeps the best copy kernel of the same mask for host rewrites.
 *
 *  @param[out] engine Left untouched unless the call succeeds.
 *  @param[in] needles The vocabulary; an empty needle is refused rather than skipped, as it would
 *      match at every position and dropping it would shift every later needle's reported index.
 *      Read on the host.
 *  @param[in] case_sensitivity Whether both sides are folded before they meet, or compared as is.
 *  @param[in] overlap_policy The cover every round runs, as the arena is sized for that one alone.
 *  @param[in] hot_states States to keep in the dense hot rows, or
 *      @ref STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO to fill a fixed byte budget, which holds more
 *      states the fewer classes the vocabulary spells.
 *  @param[in] matches_budget Matches one round may emit, read by a device tier only; a host tier
 *      walks straight into the caller's output and ignores it. Zero asks for a tier-chosen default.
 *  @param[in] haystacks_budget Haystacks one round may carry, read by a device tier only, zero
 *      asking for its default. Its arena is sized for it here, so no compute verb allocates, and a
 *      round carrying more is refused with @c sz_unexpected_dimensions_k.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu or
 *      @c sz_capabilities_enabled_cuda report; its group picks the CPU or a GPU vendor.
 *  @param[in] allocator Where both blocks come from, or @c STRINGZILLA_NULL for
 *      @ref sz_allocator_init_unified_best of @p capabilities. Stored by value, so
 *      @ref sz_substrings_engine_free needs none and cannot get the wrong one; on a device it must
 *      be host-writable, as the host writes the automaton there.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, also naming the device:
 *      a @c cudaStream_t, a @c hipStream_t, or an @c id<MTLCommandQueue>; null for the default.
 *  @return @c sz_success_k once the vocabulary compiled; @c sz_missing_kernel_k when no capability
 *      of the mask has an init; @c sz_bad_alloc_k if memory allocation failed;
 *      @c sz_overflow_risk_k if the vocabulary exceeds a 32-bit state id; @c sz_invalid_utf8_k for
 *      a malformed needle under @ref sz_substrings_uncased_k; @c sz_unexpected_dimensions_k if the
 *      vocabulary or one of its needles was empty; or on a device @c sz_device_memory_mismatch_k
 *      for a block or a @p stream it cannot use, @c sz_missing_gpu_k when it doesn't answer, and
 *      @c sz_device_code_mismatch_k on Metal for @ref sz_substrings_uncased_k, whose folding tables
 *      have no Metal port.
 *  @note May join @p stream; no compute verb ever does.
 */
STRINGZILLA_API sz_status_t sz_substrings_engine_init(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                      sz_substrings_case_sensitivity_t case_sensitivity,
                                                      sz_substrings_overlap_policy_t overlap_policy,
                                                      sz_size_t hot_states, sz_size_t matches_budget,
                                                      sz_size_t haystacks_budget, sz_capability_t capabilities,
                                                      sz_allocator_t *allocator, sz_stream_t stream);

/** Returns both of the engine's blocks to the allocator that built them, once the work queued on
 *  @p stream is done with them, emptying @p engine. */
STRINGZILLA_API void sz_substrings_engine_free(sz_substrings_engine_t *engine, sz_stream_t stream);

/**
 *  @brief Counts the matches of every needle in every haystack, one count per haystack.
 *
 *  Every verb runs the kernel of the capability @p engine was compiled for. A device one enqueues
 *  on @p stream and returns; every round of one engine runs out of its one arena and writes its one
 *  report, so the caller orders them - one stream, or events between two - and reads the report
 *  only after joining the round that wrote it.
 *
 *  @param[in] engine A compiled vocabulary, whose @c overlap_policy decides whether matches sharing
 *      bytes are all counted or thinned to a leftmost run.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[out] counts The per-haystack counts, haystack h at `counts[h * stride]`.
 *  @param[in] counts_stride Entries from one haystack's count to the next, at least one, so a
 *      strided call writes one column of a @b [haystacks, vocabularies] feature matrix.
 *  @param[in] stream Null on the CPU. On a GPU, the stream to queue on, which also names the device
 *      the round runs on; null for the default.
 *  @return @c sz_success_k once the haystacks were counted, or on a device the counting enqueued;
 *      @c sz_unexpected_dimensions_k if @p counts_stride is zero; @c sz_missing_kernel_k for an
 *      empty engine; or @c sz_device_memory_mismatch_k if a device got an argument or a @p stream
 *      it cannot use.
 */
STRINGZILLA_API sz_status_t sz_substrings_counts(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                 sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);

/**
 *  @brief Reports every match of every needle in every haystack.
 *
 *  @param[in] engine A compiled vocabulary.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[out] matches Room for @p matches_capacity matches, ascending by haystack;
 *      @c STRINGZILLA_NULL together with a zero capacity makes the call a pure size query.
 *  @param[in] matches_capacity Entries @p matches holds.
 *  @param[out] matches_offsets One boundary per haystack into @p matches plus one, the last being
 *      the total; filled whether or not the matches fit, which is what sizes the next call.
 *  @param[in] stream As @ref sz_substrings_counts takes it.
 *  @return @c sz_success_k once every match was reported, or on a device the reporting enqueued;
 *      @c sz_missing_kernel_k for an empty engine; or @c sz_device_memory_mismatch_k if a device
 *      got an unaddressable argument.
 *
 *  A capacity too small is not an error: the report's @c matches_emitted names the true total and
 *  @c shortfall names what did not fit, so a sizing call and a filling call need no walk between.
 */
STRINGZILLA_API sz_status_t sz_substrings_find(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                               sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream);

/**
 *  @brief Rewrites every haystack, substituting one replacement per needle, into one @p target.
 *
 *  @param[in] engine A vocabulary compiled under a leftmost policy;
 *      @ref sz_substrings_overlapping_k is refused, since a substitution over matches that share
 *      bytes is not a function.
 *  @param[in] haystacks The texts to rewrite; their bytes are read in place and never copied.
 *  @param[in] replacements One replacement per needle, indexed by needle; an empty one deletes.
 *  @param[out] target Room for @p target_capacity bytes, or @c STRINGZILLA_NULL with zero capacity
 *      to size only.
 *  @param[in] target_capacity Bytes @p target holds.
 *  @param[out] offsets One rewritten boundary per haystack plus one, the last being the total;
 *      filled whether or not the target held the result, which is what sizes the next call.
 *  @param[in] stream As @ref sz_substrings_counts takes it.
 *  @return @c sz_success_k once every haystack was rewritten, or on a device the rewrite enqueued;
 *      @c sz_unexpected_dimensions_k if @p replacements lacks one entry per needle;
 *      @c sz_status_unknown_k if the policy leaves no cover; @c sz_missing_kernel_k for an empty
 *      engine; or @c sz_device_memory_mismatch_k for an argument no device kernel can address.
 *
 *  A target too small is not an error: the report's @c target_length names the bytes the rewrite
 *  needs and @c shortfall names what did not fit, leaving the contents of @p target unspecified.
 */
STRINGZILLA_API sz_status_t sz_substrings_replace(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                  sz_sequence_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream);

/**
 *  @brief Scores every haystack against the vocabulary as one BM25 query, one score per haystack.
 *
 *  The vocabulary is the query: @p needle_weights holds each needle's IDF or boost. Term
 *  frequencies are raw overlapping counts, since a leftmost cover would suppress genuine
 *  occurrences of a needle nested in another, so the engine's own policy does not apply here. A CPU
 *  backend sums in ascending needle order and a CUDA backend in fixed-point integers, so each is
 *  bit-stable across its own runs, and the two agree.
 *
 *  @param[in] engine A compiled vocabulary.
 *  @param[in] haystacks The documents to score; their bytes are read in place and never copied.
 *  @param[in] document_lengths One length per haystack to normalize by, in the unit of
 *      @c average_document_length, or @c STRINGZILLA_NULL to use each haystack's byte length.
 *  @param[in] parameters BM25's continuous parameters.
 *  @param[in] needle_weights One weight per needle of the engine.
 *  @param[out] scores The per-haystack scores, haystack h at `scores[h * stride]`.
 *  @param[in] scores_stride Entries from one haystack's score to the next, at least one, so a
 *      strided call writes one column of a @b [haystacks, vocabularies] feature matrix.
 *  @param[in] stream As @ref sz_substrings_counts takes it.
 *  @return @c sz_success_k once every haystack was scored, or on a device the scoring enqueued;
 *      @c sz_unexpected_dimensions_k for invalid arguments, as below; @c sz_missing_kernel_k for an
 *      empty engine; or @c sz_device_memory_mismatch_k for an argument a device cannot address.
 *
 *  The arguments are invalid when @p needle_weights is @c STRINGZILLA_NULL, @p scores_stride is
 *  zero, or @c length_normalization is positive while @c average_document_length is not.
 */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_f32_t const *document_lengths,
                                                      sz_substrings_bm25_t const *parameters,
                                                      sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                      sz_size_t scores_stride, sz_stream_t stream);

/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_serial(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_serial(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_serial(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                      sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_serial(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                         sz_sequence_t const *replacements, sz_ptr_t target,
                                                         sz_size_t target_capacity, sz_size_t *offsets,
                                                         sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_serial(sz_substrings_engine_t *engine,
                                                             sz_sequence_t const *haystacks,
                                                             sz_f32_t const *document_lengths,
                                                             sz_substrings_bm25_t const *parameters,
                                                             sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                             sz_size_t scores_stride, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_haswell(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_haswell(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                         sz_size_t *counts, sz_size_t counts_stride,
                                                         sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_haswell(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                       sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_haswell(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_sequence_t const *replacements, sz_ptr_t target,
                                                          sz_size_t target_capacity, sz_size_t *offsets,
                                                          sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_haswell(sz_substrings_engine_t *engine,
                                                              sz_sequence_t const *haystacks,
                                                              sz_f32_t const *document_lengths,
                                                              sz_substrings_bm25_t const *parameters,
                                                              sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                              sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_icelake(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_icelake(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                         sz_size_t *counts, sz_size_t counts_stride,
                                                         sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_icelake(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                       sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_icelake(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_sequence_t const *replacements, sz_ptr_t target,
                                                          sz_size_t target_capacity, sz_size_t *offsets,
                                                          sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_icelake(sz_substrings_engine_t *engine,
                                                              sz_sequence_t const *haystacks,
                                                              sz_f32_t const *document_lengths,
                                                              sz_substrings_bm25_t const *parameters,
                                                              sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                              sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_neon(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                           sz_substrings_case_sensitivity_t case_sensitivity,
                                                           sz_substrings_overlap_policy_t overlap_policy,
                                                           sz_size_t hot_states, sz_size_t matches_budget,
                                                           sz_size_t haystacks_budget, sz_allocator_t *allocator,
                                                           sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_neon(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_neon(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                    sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                    sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_neon(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements, sz_ptr_t target,
                                                       sz_size_t target_capacity, sz_size_t *offsets,
                                                       sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_neon(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                           sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_rvv(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                          sz_substrings_case_sensitivity_t case_sensitivity,
                                                          sz_substrings_overlap_policy_t overlap_policy,
                                                          sz_size_t hot_states, sz_size_t matches_budget,
                                                          sz_size_t haystacks_budget, sz_allocator_t *allocator,
                                                          sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_rvv(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_rvv(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                   sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                   sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_rvv(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_sequence_t const *replacements, sz_ptr_t target,
                                                      sz_size_t target_capacity, sz_size_t *offsets,
                                                      sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_rvv(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_f32_t const *document_lengths,
                                                          sz_substrings_bm25_t const *parameters,
                                                          sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                          sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_v128(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                           sz_substrings_case_sensitivity_t case_sensitivity,
                                                           sz_substrings_overlap_policy_t overlap_policy,
                                                           sz_size_t hot_states, sz_size_t matches_budget,
                                                           sz_size_t haystacks_budget, sz_allocator_t *allocator,
                                                           sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_v128(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_v128(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                    sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                    sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_v128(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements, sz_ptr_t target,
                                                       sz_size_t target_capacity, sz_size_t *offsets,
                                                       sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_v128(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                           sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_loongsonasx(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_loongsonasx(sz_substrings_engine_t *engine,
                                                             sz_sequence_t const *haystacks, sz_size_t *counts,
                                                             sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_loongsonasx(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                           sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_loongsonasx(sz_substrings_engine_t *engine,
                                                              sz_sequence_t const *haystacks,
                                                              sz_sequence_t const *replacements, sz_ptr_t target,
                                                              sz_size_t target_capacity, sz_size_t *offsets,
                                                              sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_loongsonasx(sz_substrings_engine_t *engine,
                                                                  sz_sequence_t const *haystacks,
                                                                  sz_f32_t const *document_lengths,
                                                                  sz_substrings_bm25_t const *parameters,
                                                                  sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                                  sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_powervsx(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_powervsx(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks, sz_size_t *counts,
                                                          sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_powervsx(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                        sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_powervsx(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_sequence_t const *replacements, sz_ptr_t target,
                                                           sz_size_t target_capacity, sz_size_t *offsets,
                                                           sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_powervsx(sz_substrings_engine_t *engine,
                                                               sz_sequence_t const *haystacks,
                                                               sz_f32_t const *document_lengths,
                                                               sz_substrings_bm25_t const *parameters,
                                                               sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                               sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                           sz_substrings_case_sensitivity_t case_sensitivity,
                                                           sz_substrings_overlap_policy_t overlap_policy,
                                                           sz_size_t hot_states, sz_size_t matches_budget,
                                                           sz_size_t haystacks_budget, sz_allocator_t *allocator,
                                                           sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                    sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                    sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements, sz_ptr_t target,
                                                       sz_size_t target_capacity, sz_size_t *offsets,
                                                       sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_cuda(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                           sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_HOPPER
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_hopper(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_hopper(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_hopper(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                      sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_hopper(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                         sz_sequence_t const *replacements, sz_ptr_t target,
                                                         sz_size_t target_capacity, sz_size_t *offsets,
                                                         sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_hopper(sz_substrings_engine_t *engine,
                                                             sz_sequence_t const *haystacks,
                                                             sz_f32_t const *document_lengths,
                                                             sz_substrings_bm25_t const *parameters,
                                                             sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                             sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_rocm(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                           sz_substrings_case_sensitivity_t case_sensitivity,
                                                           sz_substrings_overlap_policy_t overlap_policy,
                                                           sz_size_t hot_states, sz_size_t matches_budget,
                                                           sz_size_t haystacks_budget, sz_allocator_t *allocator,
                                                           sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_rocm(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_rocm(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                    sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                    sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_rocm(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements, sz_ptr_t target,
                                                       sz_size_t target_capacity, sz_size_t *offsets,
                                                       sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_rocm(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                           sz_size_t scores_stride, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_METAL
/** @copydoc sz_substrings_engine_init */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_metal(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream);
/** @copydoc sz_substrings_counts */
STRINGZILLA_API sz_status_t sz_substrings_counts_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream);
/** @copydoc sz_substrings_find */
STRINGZILLA_API sz_status_t sz_substrings_find_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                     sz_size_t *matches_offsets, sz_stream_t stream);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API sz_status_t sz_substrings_replace_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_sequence_t const *replacements, sz_ptr_t target,
                                                        sz_size_t target_capacity, sz_size_t *offsets,
                                                        sz_stream_t stream);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_metal(sz_substrings_engine_t *engine,
                                                            sz_sequence_t const *haystacks,
                                                            sz_f32_t const *document_lengths,
                                                            sz_substrings_bm25_t const *parameters,
                                                            sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                            sz_size_t scores_stride, sz_stream_t stream);
#endif

/**
 *  @brief Finds the substrings kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_substrings_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                      sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/substrings/haswell.h"
#include "stringzilla/substrings/icelake.h"
#include "stringzilla/substrings/neon.h"
#include "stringzilla/substrings/rvv.h"
#include "stringzilla/substrings/v128.h"
#include "stringzilla/substrings/loongsonasx.h"
#include "stringzilla/substrings/powervsx.h"
#include "stringzilla/substrings/cuda.cuh"
#include "stringzilla/substrings/hopper.cuh"
#include "stringzilla/substrings/rocm.cuh"
#include "stringzilla/substrings/metal.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_substrings_engine_init(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                      sz_substrings_case_sensitivity_t case_sensitivity,
                                                      sz_substrings_overlap_policy_t overlap_policy,
                                                      sz_size_t hot_states, sz_size_t matches_budget,
                                                      sz_size_t haystacks_budget, sz_capability_t capabilities,
                                                      sz_allocator_t *allocator, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(needles), sz_unused_(case_sensitivity), sz_unused_(overlap_policy),
        sz_unused_(hot_states), sz_unused_(matches_budget), sz_unused_(haystacks_budget), sz_unused_(capabilities),
        sz_unused_(allocator), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API void sz_substrings_engine_free(sz_substrings_engine_t *engine, sz_stream_t stream) {
    sz_substrings_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                 sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(haystacks), sz_unused_(counts), sz_unused_(counts_stride), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_substrings_find(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                               sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(haystacks), sz_unused_(matches), sz_unused_(matches_capacity),
        sz_unused_(matches_offsets), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_substrings_replace(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                  sz_sequence_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(haystacks), sz_unused_(replacements), sz_unused_(target),
        sz_unused_(target_capacity), sz_unused_(offsets), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_f32_t const *document_lengths,
                                                      sz_substrings_bm25_t const *parameters,
                                                      sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                      sz_size_t scores_stride, sz_stream_t stream) {
    sz_unused_(engine), sz_unused_(haystacks), sz_unused_(document_lengths), sz_unused_(parameters),
        sz_unused_(needle_weights), sz_unused_(scores), sz_unused_(scores_stride), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_substrings_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                      sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SUBSTRINGS_H_
