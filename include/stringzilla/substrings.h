/**
 *  @brief Multi-pattern substring search: one compiled Aho-Corasick automaton over many haystacks.
 *  @file include/stringzilla/substrings.h
 *  @author Ash Vardanian
 *
 *  A vocabulary of needles compiles once into a byte-level automaton, and every haystack then streams
 *  through it in a single pass, whatever the needle count. The automaton is two-tiered: a dense
 *  goto-completed hot table for the states text keeps returning to, and a double array for the rest,
 *  because a trie's visit counts are skewed however large the vocabulary grows.
 *
 *  Case folding lives in the stream rather than in the automaton. A needle is folded once at insertion
 *  and inserted byte for byte, and an uncased haystack is folded as the walk consumes it, so both sides
 *  meet in one canonical byte stream and the trie stays a tree with single-valued failure links.
 *
 *  @section substrings_api Public API
 *
 *  - @ref sz_substrings_build → compiles a vocabulary into an automaton the other verbs read;
 *  - @ref sz_substrings_counts → how many matches each haystack holds;
 *  - @ref sz_substrings_find → every match, located by haystack, needle and byte span;
 *  - @ref sz_substrings_replace → the haystacks rewritten with one replacement per needle;
 *  - @ref sz_substrings_bm25_scores → one BM25 score per haystack, the vocabulary being the query.
 *
 *  The automaton is a plain struct of flat arrays over one owned block, so a caller driving its own loops
 *  binds @ref sz_substrings_step and walks it directly, and a device backend passes it to a kernel by value.
 */
#ifndef STRINGZILLA_SUBSTRINGS_H_
#define STRINGZILLA_SUBSTRINGS_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/** Whether a vocabulary matches needles byte-for-byte, or folds both sides to a shared case first. */
typedef enum sz_substrings_case_sensitivity_t {
    /** Byte-exact matching; needles may be arbitrary bytes, including malformed UTF-8. */
    sz_substrings_cased_k = 0,
    /** Full Unicode case folding as `CaseFolding.txt` defines it; needles must be valid UTF-8. */
    sz_substrings_uncased_k = 1,
} sz_substrings_case_sensitivity_t;


/**
 *  @brief One needle ending on one state, as the automaton stores it.
 *
 *  The automaton walks @b folded bytes, so this length is the folded one - the needle's own, identical for
 *  every match of that needle. The @b source span it corresponds to is not: needle "k" matches both the
 *  1-byte "k" and the 3-byte Kelvin sign U+212A. Recovering that span is the walk's job.
 */
typedef struct sz_substrings_output_t {
    /** Which needle ends here, as an index into the sequence the automaton was built from. */
    sz_u32_t needle_index;
    /** Folded bytes this match spans; a walk traverses one edge per byte, so it fits a state id. */
    sz_u32_t folded_match_bytes;
} sz_substrings_output_t;

/**
 *  @brief One reported match, locating it by haystack, by needle, and by byte span.
 *
 *  Under case folding a needle's own byte length is not the length of every match, so the span is carried
 *  per match rather than looked up from the needle.
 */
typedef struct sz_substrings_match_t {
    /** Which haystack of the sequence this match was found in. */
    sz_size_t haystack_index;
    /** Which needle of the vocabulary matched. */
    sz_size_t needle_index;
    /** Where the match starts inside that haystack, in its own bytes. */
    sz_size_t byte_offset;
    /** Haystack bytes the match spans, which folding can make differ from the needle's own length. */
    sz_size_t byte_length;
} sz_substrings_match_t;

/**
 *  @brief One start position's incumbent match while a leftmost policy is still deciding it.
 *
 *  Separate from @ref sz_substrings_output_t because the units differ: an output carries the folded length
 *  the automaton walked, this carries the @b source span that length resolved to.
 */
typedef struct sz_substrings_pending_start_t {
    /** Which needle currently claims this start. */
    sz_u32_t needle_index;
    /** Haystack bytes that match spans; zero means no match has claimed the start yet. */
    sz_u32_t source_match_bytes;
} sz_substrings_pending_start_t;

/**
 *  @brief A compiled vocabulary: flat arrays over one owned block, trivially copyable into a kernel.
 *
 *  Transitions are split into two tiers by how often a state is visited. Text keeps resetting the walk
 *  toward the root, so a small set of states absorbs most byte steps whatever the dictionary size, and the
 *  tiers are sized to that skew rather than to the automaton as a whole.
 *
 *  The @b hot tier is a dense goto-completed table, one row of 256 targets per state, so a step is a single
 *  load with no branch and no failure chasing. The @b cold tier is a double array: @c base and @c check
 *  encode transitions as address arithmetic plus an ownership test, and @c fail restores the failure links
 *  that goto completion would otherwise have folded away. States are numbered so the hot ones come first,
 *  which makes the tier test @c state @c < @c hot_count with no lookup.
 */
typedef struct sz_substrings_automaton_t {
    /** Hot tier: @b [hot_count * 256] goto-completed targets, row-major, shallow states first. */
    sz_u32_t const *hot_rows;
    /** Cold tier: transition target for @c state on @c byte is @c base[state] @c + @c byte, if owned. */
    sz_u32_t const *base;
    /** Cold tier: owner of each slot, so a collision reads as a missing edge rather than a wrong one. */
    sz_u32_t const *check;
    /** Cold tier: failure link, followed when @c check denies ownership. */
    sz_u32_t const *fail;
    /** One bit per slot: whether any needle ends there, so a non-matching byte touches no count. */
    sz_u32_t const *accepts_words;
    /** Matches ending at each state, flattened and already merged along failure chains at build time. */
    sz_substrings_output_t const *outputs;
    /** Outputs per slot, in the published numbering. */
    sz_u32_t const *outputs_counts;
    /** Exclusive prefix sum of @c outputs_counts; a nested-suffix vocabulary drives the pool to O(states^2),
     *  so these stay pointer-wide where the counts do not. */
    sz_size_t const *outputs_offsets;
    /** Length of @c outputs, so a consumer never rescans the CSR to recover it. */
    sz_size_t outputs_total;
    /** Slots every cold-tier array holds, which is @c state_count plus the alphabet's address headroom. */
    sz_size_t slots_count;
    /** States @c [0, @c hot_count) live in @c hot_rows; the rest live in the double array. */
    sz_u32_t hot_count;
    /** Published state ceiling; a packed child's id is address arithmetic, so it exceeds the trie's own. */
    sz_u32_t state_count;
    /** The root's published id, which is always zero. */
    sz_u32_t root;
    /** Needles the vocabulary holds, which bounds every reported @c needle_index. */
    sz_u32_t needles_count;
    /** Most @b haystack bytes one match can span, which every slice, halo and warm-up is sized from. */
    sz_u32_t max_source_match_bytes;
    /** Fewest haystack bytes one match can span; the mirror bound. */
    sz_u32_t min_source_match_bytes;
    /** Most merged outputs any single state carries, so a consumer can bound one pass's match count. */
    sz_u32_t max_outputs_per_state;
    /** Whether a walk folds the haystack as it consumes it, or steps it byte for byte. */
    sz_substrings_case_sensitivity_t case_sensitivity;
    /** The one block every pointer above addresses, which @ref sz_substrings_automaton_free returns. */
    void *memory;
    /** Bytes of that block, which the allocator's @c free is handed back. */
    sz_size_t memory_bytes;
} sz_substrings_automaton_t;

/**
 *  @brief How faithfully a backend's leftmost cover reproduces the serial one.
 *
 *  A CPU backend settles a cover during the walk and always reports the leftmost one. A CUDA backend
 *  settles it afterwards, one run of mutually-reaching matches per thread, and a greedy over such a run is
 *  quadratic in it - so past a bound it accepts in the order the walk emitted instead. That is the same
 *  cover whenever match starts ascend with their ends, and a coarser one when they do not, which happens
 *  when a vocabulary is dense enough that no gap ever separates two matches.
 *
 *  Both answers are covers: every match is real and no two share a byte. Only the choice among rivals
 *  differs, so a caller that needs the leftmost cover exactly asks a CPU backend for it.
 */
typedef enum sz_substrings_cover_fidelity_t {
    /** The leftmost cover itself, which every CPU backend reports. */
    sz_substrings_cover_exact_k = 0,
    /** A valid cover that may differ among rivals, which a CUDA backend reports on a dense vocabulary. */
    sz_substrings_cover_approximate_k = 1,
} sz_substrings_cover_fidelity_t;

/** BM25's continuous parameters. */
typedef struct sz_substrings_bm25_t {
    /** The literature's `k1`: how slowly repeated occurrences stop adding score; 1.2 is customary. */
    sz_f32_t term_frequency_saturation;
    /** The literature's `b`, in [0, 1]: 0 ignores document length, 1 normalizes it fully; 0.75 is customary. */
    sz_f32_t length_normalization;
    /** The corpus-wide mean document length, in the unit of the lengths scored; read only when
     *  `length_normalization` is positive. */
    sz_f32_t average_document_length;
} sz_substrings_bm25_t;

/** @ref sz_substrings_build's "size the hot tier yourself" argument, so zero stays a real all-cold request. */
#define SZ_SUBSTRINGS_HOT_STATES_AUTO (SZ_SIZE_MAX)

/**
 *  @brief Compiles @p needles into an automaton the matching verbs read.
 *
 *  @param[in] needles The vocabulary; an empty needle is refused rather than skipped, since it would match
 *             at every position and dropping it would shift every later needle's reported index.
 *  @param[in] case_sensitivity Whether both sides are folded before they meet, or compared byte for byte.
 *  @param[in] hot_states States to keep in the dense hot rows, or @ref SZ_SUBSTRINGS_HOT_STATES_AUTO to
 *             size the tier from the vocabulary itself.
 *  @param[in] alloc Where the builder's scratch and the automaton's one block come from; never @c SZ_NULL.
 *             The scratch is freed before returning, the block by @ref sz_substrings_automaton_free with this
 *             same allocator. The strict CUDA verbs read the block in place, so it must then reach the device.
 *  @param[out] automaton Left untouched unless the call succeeds.
 *
 *  @retval sz_success_k The vocabulary compiled.
 *  @retval sz_bad_alloc_k Memory allocation failed.
 *  @retval sz_overflow_risk_k The vocabulary exceeds what a 32-bit state id can address.
 *  @retval sz_invalid_utf8_k Under @ref sz_substrings_uncased_k, a needle was not well-formed UTF-8.
 *  @retval sz_unexpected_dimensions_k The vocabulary was empty, or one of its needles was.
 *  @sa sz_substrings_automaton_free
 */
SZ_API_COMPTIME sz_status_t sz_substrings_build(sz_sequence_t const *needles,
                                                sz_substrings_case_sensitivity_t case_sensitivity,
                                                sz_size_t hot_states, sz_memory_allocator_t *alloc,
                                                sz_substrings_automaton_t *automaton);

/** Returns the automaton's block to @p alloc, the allocator that built it and never @c SZ_NULL, and leaves
 *  @p automaton empty. */
SZ_API_COMPTIME void sz_substrings_automaton_free(sz_substrings_automaton_t *automaton,
                                                  sz_memory_allocator_t *alloc);

/**
 *  @brief Counts the matches of every needle in every haystack, one count per haystack.
 *
 *  @param[in] automaton A vocabulary compiled by @ref sz_substrings_build.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[in] overlap_policy Whether matches sharing bytes are all reported, or thinned to a leftmost run.
 *  @param[in] alloc Where the call's scratch comes from, all of it freed before returning; never @c SZ_NULL.
 *             A CPU backend takes the leftmost ring from it, a CUDA backend its device-side arrays when the
 *             scratch and every argument reach the device, and otherwise the host buffer it stages texts through.
 *  @param[out] counts The @b [haystacks->count] per-haystack match counts.
 *
 *  @retval sz_success_k The haystacks were counted.
 *  @retval sz_bad_alloc_k The scratch could not be allocated.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_substrings_counts_serial
 */
SZ_API_RUNTIME sz_status_t sz_substrings_counts(sz_substrings_automaton_t const *automaton,
                                                sz_sequence_t const *haystacks,
                                                sz_substrings_overlap_policy_t overlap_policy,
                                                sz_memory_allocator_t *alloc, sz_size_t *counts);

/**
 *  @brief Reports every match of every needle in every haystack.
 *
 *  @param[in] automaton A vocabulary compiled by @ref sz_substrings_build.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[in] overlap_policy Whether matches sharing bytes are all reported, or thinned to a leftmost run.
 *  @param[in] alloc Where the call's scratch comes from, all of it freed before returning; never @c SZ_NULL.
 *             A CPU backend takes the leftmost ring from it, a CUDA backend its device-side arrays when the
 *             scratch and every argument reach the device, and otherwise the host buffer it stages texts through.
 *  @param[out] matches Room for @p matches_capacity matches, ascending by haystack; may be @c SZ_NULL
 *              together with a zero capacity, which makes the call a pure size query.
 *  @param[in] matches_capacity Entries @p matches holds.
 *  @param[out] matches_found Matches the haystacks hold, which is the true total whether or not they fit.
 *
 *  @retval sz_success_k Every match was reported.
 *  @retval sz_unexpected_dimensions_k More matches exist than @p matches_capacity holds; the first
 *          @p matches_capacity of them are written and @p matches_found names the true total, so one
 *          sizing call and one filling call need no second walk between them.
 *  @retval sz_bad_alloc_k The scratch could not be allocated.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_substrings_find_serial
 */
SZ_API_RUNTIME sz_status_t sz_substrings_find(sz_substrings_automaton_t const *automaton,
                                              sz_sequence_t const *haystacks,
                                              sz_substrings_overlap_policy_t overlap_policy,
                                              sz_memory_allocator_t *alloc, sz_substrings_match_t *matches,
                                              sz_size_t matches_capacity, sz_size_t *matches_found);

/**
 *  @brief Rewrites every haystack, substituting one replacement per needle, onto one output tape.
 *
 *  @param[in] automaton A vocabulary compiled by @ref sz_substrings_build.
 *  @param[in] haystacks The texts to rewrite; their bytes are read in place and never copied.
 *  @param[in] replacements One replacement per needle, indexed by needle; an empty one deletes the match.
 *  @param[in] overlap_policy A leftmost policy; @ref sz_substrings_overlapping_k is refused, since a
 *             substitution over matches that share bytes is not a function.
 *  @param[in] alloc Where the call's scratch comes from, all of it freed before returning; never @c SZ_NULL.
 *             A CPU backend takes the leftmost ring from it, a CUDA backend its device-side arrays when the
 *             scratch and every argument reach the device, and otherwise the host buffer it stages texts through.
 *  @param[out] tape Room for @p tape_capacity bytes, or @c SZ_NULL with a zero capacity to size only.
 *  @param[in] tape_capacity Bytes @p tape holds.
 *  @param[out] offsets The @b [haystacks->count + 1] rewritten boundaries, the last being the total; these
 *              are filled whether or not the tape held the result, which is what sizes the next call.
 *
 *  @retval sz_success_k Every haystack was rewritten.
 *  @retval sz_unexpected_dimensions_k @p replacements does not hold one entry per needle, or the rewrite
 *          needs more than @p tape_capacity bytes - in which case @p offsets names how many and @p tape's
 *          contents are unspecified, since a backend sizes and splices in one walk.
 *  @retval sz_status_unknown_k @p overlap_policy leaves no cover to substitute.
 *  @retval sz_bad_alloc_k The scratch could not be allocated.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_substrings_replace_serial
 */
SZ_API_RUNTIME sz_status_t sz_substrings_replace(sz_substrings_automaton_t const *automaton,
                                                 sz_sequence_t const *haystacks, sz_sequence_t const *replacements,
                                                 sz_substrings_overlap_policy_t overlap_policy,
                                                 sz_memory_allocator_t *alloc, sz_ptr_t tape,
                                                 sz_size_t tape_capacity, sz_size_t *offsets);

/**
 *  @brief Scores every haystack against the vocabulary as one BM25 query, one score per haystack.
 *
 *  The vocabulary is the query: @p needle_weights holds each needle's IDF or boost. Term frequencies are raw
 *  overlapping counts, since a leftmost cover would suppress genuine occurrences of a needle nested in
 *  another, so no overlap policy applies. A CPU backend sums in ascending needle order and a CUDA backend in
 *  fixed-point integers, so each is bit-stable across its own runs, and the two agree numerically.
 *
 *  @param[in] automaton A vocabulary compiled by @ref sz_substrings_build.
 *  @param[in] haystacks The documents to score; their bytes are read in place and never copied.
 *  @param[in] document_lengths The @b [haystacks->count] lengths to normalize by, in any unit consistent with
 *             @c average_document_length, or @c SZ_NULL to use each haystack's byte length.
 *  @param[in] parameters BM25's continuous parameters.
 *  @param[in] needle_weights The @b [automaton->needles_count] per-needle weights.
 *  @param[in] alloc Where the call's scratch comes from, all of it freed before returning; never @c SZ_NULL.
 *             A CPU backend takes its per-needle counters from it, a CUDA backend its overflow counters when
 *             the vocabulary outgrows a block's table, and otherwise the host buffer it stages texts through.
 *  @param[out] scores The @b [haystacks->count] scores.
 *
 *  @retval sz_success_k Every haystack was scored.
 *  @retval sz_unexpected_dimensions_k @p needle_weights is @c SZ_NULL, or @c length_normalization is positive
 *          while @c average_document_length is not, leaving no mean to normalize by.
 *  @retval sz_bad_alloc_k The scratch could not be allocated.
 *  @note Selects the fastest implementation at compile- or run-time based on @c SZ_DYNAMIC_DISPATCH.
 *  @sa sz_substrings_bm25_scores_serial
 */
SZ_API_RUNTIME sz_status_t sz_substrings_bm25_scores(sz_substrings_automaton_t const *automaton,
                                                     sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
                                                     sz_substrings_bm25_t const *parameters,
                                                     sz_f32_t const *needle_weights, sz_memory_allocator_t *alloc,
                                                     sz_f32_t *scores);

/** @copydoc sz_substrings_counts */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_serial(sz_substrings_automaton_t const *automaton,
                                                        sz_sequence_t const *haystacks,
                                                        sz_substrings_overlap_policy_t overlap_policy,
                                                        sz_memory_allocator_t *alloc, sz_size_t *counts);

/** @copydoc sz_substrings_find */
SZ_API_COMPTIME sz_status_t sz_substrings_find_serial(sz_substrings_automaton_t const *automaton,
                                                      sz_sequence_t const *haystacks,
                                                      sz_substrings_overlap_policy_t overlap_policy,
                                                      sz_memory_allocator_t *alloc, sz_substrings_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_found);

/** @copydoc sz_substrings_replace */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_serial(sz_substrings_automaton_t const *automaton,
                                                         sz_sequence_t const *haystacks,
                                                         sz_sequence_t const *replacements,
                                                         sz_substrings_overlap_policy_t overlap_policy,
                                                         sz_memory_allocator_t *alloc, sz_ptr_t tape,
                                                         sz_size_t tape_capacity, sz_size_t *offsets);

/** @copydoc sz_substrings_bm25_scores */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_serial(sz_substrings_automaton_t const *automaton,
                                                             sz_sequence_t const *haystacks,
                                                             sz_f32_t const *document_lengths,
                                                             sz_substrings_bm25_t const *parameters,
                                                             sz_f32_t const *needle_weights,
                                                             sz_memory_allocator_t *alloc, sz_f32_t *scores);

#if SZ_USE_CUDA

/**
 *  @copydoc sz_substrings_counts
 *
 *  Stages whatever the device cannot already reach, so a caller holding host memory still gets an answer.
 *  @sa sz_substrings_counts_scheduled_cuda for the strict verb that refuses instead of staging.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_cuda(sz_substrings_automaton_t const *automaton,
                                                      sz_sequence_t const *haystacks,
                                                      sz_substrings_overlap_policy_t overlap_policy,
                                                      sz_memory_allocator_t *alloc, sz_size_t *counts);

/**
 *  @copydoc sz_substrings_find
 *  Stages whatever the device cannot already reach, so a caller holding host memory still gets an answer.
 *  @sa sz_substrings_find_scheduled_cuda for the strict verb that refuses instead of staging.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_find_cuda(sz_substrings_automaton_t const *automaton,
                                                    sz_sequence_t const *haystacks,
                                                    sz_substrings_overlap_policy_t overlap_policy,
                                                    sz_memory_allocator_t *alloc, sz_substrings_match_t *matches,
                                                    sz_size_t matches_capacity, sz_size_t *matches_found);

/**
 *  @copydoc sz_substrings_replace
 *  Stages whatever the device cannot already reach, so a caller holding host memory still gets an answer.
 *  @sa sz_substrings_replace_scheduled_cuda for the strict verb that refuses instead of staging.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_cuda(sz_substrings_automaton_t const *automaton,
                                                       sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements,
                                                       sz_substrings_overlap_policy_t overlap_policy,
                                                       sz_memory_allocator_t *alloc, sz_ptr_t tape,
                                                       sz_size_t tape_capacity, sz_size_t *offsets);

/**
 *  @copydoc sz_substrings_bm25_scores
 *  Stages whatever the device cannot already reach, so a caller holding host memory still gets an answer.
 *  @sa sz_substrings_bm25_scores_scheduled_cuda for the strict verb that refuses instead of staging.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_cuda(sz_substrings_automaton_t const *automaton,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights,
                                                           sz_memory_allocator_t *alloc, sz_f32_t *scores);

/**
 *  @brief Counts on a device that already holds every argument, on @p stream, without staging anything.
 *  @param[in] stream A @c cudaStream_t, or @c SZ_NULL for the default stream.
 *  @retval sz_device_memory_mismatch_k Some argument, or the scratch @p alloc hands back, is not
 *          device-reachable; nothing was launched.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_scheduled_cuda(sz_substrings_automaton_t const *automaton,
                                                                sz_sequence_t const *haystacks,
                                                                sz_substrings_overlap_policy_t overlap_policy,
                                                                sz_memory_allocator_t *alloc, sz_size_t *counts,
                                                                void *stream);

/**
 *  @copydoc sz_substrings_find
 *  @param[in] stream A @c cudaStream_t, or @c SZ_NULL for the default stream.
 *  @retval sz_device_memory_mismatch_k Some argument, or the scratch @p alloc hands back, is not
 *          device-reachable; nothing was launched.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_find_scheduled_cuda(sz_substrings_automaton_t const *automaton,
                                                              sz_sequence_t const *haystacks,
                                                              sz_substrings_overlap_policy_t overlap_policy,
                                                              sz_memory_allocator_t *alloc,
                                                              sz_substrings_match_t *matches,
                                                              sz_size_t matches_capacity, sz_size_t *matches_found,
                                                              void *stream);

/**
 *  @copydoc sz_substrings_replace
 *  @param[in] stream A @c cudaStream_t, or @c SZ_NULL for the default stream.
 *  @retval sz_device_memory_mismatch_k Some argument, or the scratch @p alloc hands back, is not
 *          device-reachable; nothing was launched.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_scheduled_cuda(sz_substrings_automaton_t const *automaton,
                                                                 sz_sequence_t const *haystacks,
                                                                 sz_sequence_t const *replacements,
                                                                 sz_substrings_overlap_policy_t overlap_policy,
                                                                 sz_memory_allocator_t *alloc, sz_ptr_t tape,
                                                                 sz_size_t tape_capacity, sz_size_t *offsets,
                                                                 void *stream);

/**
 *  @copydoc sz_substrings_bm25_scores
 *  @param[in] stream A @c cudaStream_t, or @c SZ_NULL for the default stream.
 *  @retval sz_device_memory_mismatch_k Some argument, or the scratch @p alloc hands back, is not
 *          device-reachable; nothing was launched.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_scheduled_cuda(sz_substrings_automaton_t const *automaton,
                                                                     sz_sequence_t const *haystacks,
                                                                     sz_f32_t const *document_lengths,
                                                                     sz_substrings_bm25_t const *parameters,
                                                                     sz_f32_t const *needle_weights,
                                                                     sz_memory_allocator_t *alloc, sz_f32_t *scores,
                                                                     void *stream);

#endif

#pragma endregion Core API

#include "stringzilla/substrings/serial.h"
#include "stringzilla/substrings/cuda.cuh"

/*  Pick the right implementation for the multi-pattern search algorithms.
 *  To override this behavior and precompile all backends - set @c SZ_DYNAMIC_DISPATCH to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_status_t sz_substrings_counts(sz_substrings_automaton_t const *automaton,
                                                sz_sequence_t const *haystacks,
                                                sz_substrings_overlap_policy_t overlap_policy,
                                                sz_memory_allocator_t *alloc, sz_size_t *counts) {
#if SZ_USE_CUDA
    return sz_substrings_counts_cuda(automaton, haystacks, overlap_policy, alloc, counts);
#else
    return sz_substrings_counts_serial(automaton, haystacks, overlap_policy, alloc, counts);
#endif
}

SZ_API_RUNTIME sz_status_t sz_substrings_find(sz_substrings_automaton_t const *automaton,
                                              sz_sequence_t const *haystacks,
                                              sz_substrings_overlap_policy_t overlap_policy,
                                              sz_memory_allocator_t *alloc, sz_substrings_match_t *matches,
                                              sz_size_t matches_capacity, sz_size_t *matches_found) {
#if SZ_USE_CUDA
    return sz_substrings_find_cuda(automaton, haystacks, overlap_policy, alloc, matches, matches_capacity,
                                   matches_found);
#else
    return sz_substrings_find_serial(automaton, haystacks, overlap_policy, alloc, matches, matches_capacity,
                                     matches_found);
#endif
}

SZ_API_RUNTIME sz_status_t sz_substrings_replace(sz_substrings_automaton_t const *automaton,
                                                 sz_sequence_t const *haystacks, sz_sequence_t const *replacements,
                                                 sz_substrings_overlap_policy_t overlap_policy,
                                                 sz_memory_allocator_t *alloc, sz_ptr_t tape,
                                                 sz_size_t tape_capacity, sz_size_t *offsets) {
#if SZ_USE_CUDA
    return sz_substrings_replace_cuda(automaton, haystacks, replacements, overlap_policy, alloc, tape, tape_capacity,
                                      offsets);
#else
    return sz_substrings_replace_serial(automaton, haystacks, replacements, overlap_policy, alloc, tape, tape_capacity,
                                        offsets);
#endif
}

SZ_API_RUNTIME sz_status_t sz_substrings_bm25_scores(sz_substrings_automaton_t const *automaton,
                                                     sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
                                                     sz_substrings_bm25_t const *parameters,
                                                     sz_f32_t const *needle_weights, sz_memory_allocator_t *alloc,
                                                     sz_f32_t *scores) {
#if SZ_USE_CUDA
    return sz_substrings_bm25_scores_cuda(automaton, haystacks, document_lengths, parameters, needle_weights, alloc,
                                          scores);
#else
    return sz_substrings_bm25_scores_serial(automaton, haystacks, document_lengths, parameters, needle_weights, alloc,
                                            scores);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SUBSTRINGS_H_
