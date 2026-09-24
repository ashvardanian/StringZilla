/**
 *  @brief Multi-pattern substring search: one compiled Aho-Corasick engine over many haystacks.
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
 *  - @ref sz_substrings_engine_init_cpu → compiles a vocabulary the host verbs read;
 *  - @ref sz_substrings_engine_init_gpu → compiles the same vocabulary where a kernel can reach it;
 *  - @ref sz_substrings_counts → how many matches each haystack holds;
 *  - @ref sz_substrings_find → every match, located by haystack, needle and byte span;
 *  - @ref sz_substrings_replace → the haystacks rewritten with one replacement per needle;
 *  - @ref sz_substrings_bm25_scores → one BM25 score per haystack, the vocabulary being the query.
 *
 *  The engine is a plain struct of flat arrays over two owned blocks, so a caller driving its own loops
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
 *  @brief One needle ending on one state, as the engine stores it.
 *
 *  The engine walks @b folded bytes, so this length is the folded one - the needle's own, identical for
 *  every match of that needle. The @b source span it corresponds to is not: needle "k" matches both the
 *  1-byte "k" and the 3-byte Kelvin sign U+212A. Recovering that span is the walk's job.
 */
typedef struct sz_substrings_output_t {
    /** Which needle ends here, as an index into the sequence the engine was built from. */
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
 *  @brief What a round found, written by the device and read after the caller's own join.
 *
 *  A device verb enqueues and returns, so nothing it discovers can reach the caller through a status code.
 *  The sizing walk always runs, so @c matches_emitted is the truth whatever the caller's output could hold,
 *  and a nonzero @c shortfall is the one signal that the output is incomplete rather than wrong. Under a
 *  leftmost policy a cover thins the matches after that walk, so for a round that stayed inside its budget
 *  the matches a caller could have read are @c matches_stored plus @c shortfall, which is also what the last
 *  boundary of a @ref sz_substrings_find names; a round that outran its budget ran no cover at all, and its
 *  @c shortfall counts the matches the budget could not hold.
 */
typedef struct sz_substrings_report_t {
    /** Matches the sizing walk found, which is the truth whatever the output held. */
    sz_size_t matches_emitted;
    /** Matches written out, which is @c matches_emitted clipped at the capacity. */
    sz_size_t matches_stored;
    /** Bytes a rewrite needs, which is the truth whatever the tape held. */
    sz_size_t tape_bytes;
    /** Matches or bytes the round could not hold, zero when everything fit. */
    sz_size_t shortfall;
} sz_substrings_report_t;

/**
 *  @brief The compiled vocabulary, the policy it was sized for, and the round's arena, in one lifetime.
 *
 *  Transitions are split into two tiers by how often a state is visited. Text keeps resetting the walk
 *  toward the root, so a small set of states absorbs most byte steps whatever the dictionary size, and the
 *  tiers are sized to that skew rather than to the vocabulary as a whole.
 *
 *  The @b hot tier is a dense goto-completed table, one row per state with one target per byte @b class, so
 *  a step is a single load with no branch and no failure chasing. A class is a byte some needle spells, or
 *  the one shared class of every byte none does, so a row is as wide as the vocabulary's own alphabet: five
 *  targets for nucleotides, a few hundred for multilingual text, and never more than 256. The @b cold tier
 *  is a double array: @c base and @c check encode transitions as address arithmetic plus an ownership test,
 *  and @c fail restores the failure links that goto completion would otherwise have folded away. States are
 *  numbered so the hot ones come first, which makes the tier test @c state @c < @c hot_count with no lookup.
 *
 *  Both blocks are built by @c _init_cpu or @c _init_gpu and live until @ref sz_substrings_engine_free, so a
 *  compute verb allocates nothing and a device backend has no host clone to stage.
 */
typedef struct sz_substrings_engine_t {
    /** Hot tier: @b [hot_count * classes_count] goto-completed targets, row-major, shallow states first. */
    sz_u32_t const *hot_rows;
    /** Each byte's column in @c hot_rows: its own for a byte some needle spells, one shared for all others. */
    sz_u8_t const *byte_to_class;
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
    /** Columns of a hot row, at most 256. */
    sz_u32_t classes_count;
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
    /** Bytes that move a walk off the root, which a byte search can skip to while the walk sits there. */
    sz_byteset_t root_live;

    /** The policy the arena was sized for, and the only one it runs. */
    sz_substrings_overlap_policy_t overlap_policy;
    /** Matches one round may emit; past it every later kernel retires. */
    sz_size_t matches_budget;
    /** Chunks one round may cut the haystacks into, fixed here, not discovered. */
    sz_size_t chunk_budget;
    /** Device-resident counts every verb writes and no verb joins to read. */
    sz_substrings_report_t *report;
    /** The tier @c _init_* resolved, and the only one that may match with it. */
    sz_capability_t capability;
    /** What built both blocks below. */
    sz_memory_allocator_t alloc;
    /** The automaton's block, fixed for the engine's life. */
    void *memory;
    /** Bytes of that block. */
    sz_size_t memory_bytes;
    /** The round's arena: ring, chunks, emitted, keep, scan scratch. */
    void *scratch;
    /** Bytes of that block. */
    sz_size_t scratch_bytes;
} sz_substrings_engine_t;

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

/** The @c hot_states argument's "size the hot tier yourself" value, so zero stays a real all-cold request. */
#define SZ_SUBSTRINGS_HOT_STATES_AUTO (SZ_SIZE_MAX)

/**
 *  @brief Compiles @p needles into an engine the matching verbs read, on the host.
 *
 *  @param[in] needles The vocabulary; an empty needle is refused rather than skipped, since it would match
 *             at every position and dropping it would shift every later needle's reported index.
 *  @param[in] case_sensitivity Whether both sides are folded before they meet, or compared byte for byte.
 *  @param[in] overlap_policy The cover every round runs, since the arena is sized for one and holds no other.
 *  @param[in] hot_states States to keep in the dense hot rows, or @ref SZ_SUBSTRINGS_HOT_STATES_AUTO to
 *             fill a fixed byte budget, which holds more states the fewer classes the vocabulary spells.
 *  @param[in] matches_budget Matches one round may emit, read by a device tier only; a host tier walks
 *             straight into the caller's output and ignores it. Zero asks for a tier-chosen default.
 *  @param[in] alloc Where both blocks come from, or @c SZ_NULL for the default host allocator. Stored by
 *             value, so @ref sz_substrings_engine_free needs none and cannot be handed the wrong one.
 *  @param[out] engine Left untouched unless the call succeeds.
 *
 *  @retval sz_success_k The vocabulary compiled.
 *  @retval sz_bad_alloc_k Memory allocation failed.
 *  @retval sz_overflow_risk_k The vocabulary exceeds what a 32-bit state id can address.
 *  @retval sz_invalid_utf8_k Under @ref sz_substrings_uncased_k, a needle was not well-formed UTF-8.
 *  @retval sz_unexpected_dimensions_k The vocabulary was empty, or one of its needles was.
 *  @sa sz_substrings_engine_free
 */
SZ_API_RUNTIME sz_status_t sz_substrings_engine_init_cpu(sz_sequence_t const *needles,
                                                         sz_substrings_case_sensitivity_t case_sensitivity,
                                                         sz_substrings_overlap_policy_t overlap_policy,
                                                         sz_size_t hot_states, sz_size_t matches_budget,
                                                         sz_memory_allocator_t *alloc,
                                                         sz_substrings_engine_t *engine);

/**
 *  @brief Compiles @p needles into an engine on @p stream 's device, arena included.
 *
 *  @param[in] alloc Unified and bound to @p stream 's device, or @c SZ_NULL to have one derived from it; the
 *             host builder writes the automaton in place, so a device-only block cannot serve here.
 *  @param[in] stream A @c cudaStream_t, or @c SZ_NULL for the default stream.
 *  @copydetails sz_substrings_engine_init_cpu
 *  @retval sz_device_memory_mismatch_k The block @p alloc handed back does not reach the device.
 *  @note May join @p stream; no compute verb ever does.
 */
SZ_API_RUNTIME sz_status_t sz_substrings_engine_init_gpu(sz_sequence_t const *needles,
                                                         sz_substrings_case_sensitivity_t case_sensitivity,
                                                         sz_substrings_overlap_policy_t overlap_policy,
                                                         sz_size_t hot_states, sz_size_t matches_budget,
                                                         sz_memory_allocator_t *alloc,
                                                         void *stream, sz_substrings_engine_t *engine);

/** Returns both of the engine's blocks to the allocator they were built with, and leaves @p engine empty. */
SZ_API_RUNTIME void sz_substrings_engine_free(sz_substrings_engine_t *engine);

/**
 *  @brief Counts the matches of every needle in every haystack, one count per haystack.
 *
 *  @param[in] engine A vocabulary compiled by @c _init_cpu or @c _init_gpu, whose @c overlap_policy decides
 *             whether matches sharing bytes are all counted or thinned to a leftmost run.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[out] counts The @b [haystacks->count] per-haystack counts, haystack @c h at @c counts[h*stride].
 *  @param[in] counts_stride Entries from one haystack's count to the next, at least one, so a strided call
 *             writes one column of a @b [haystacks, vocabularies] feature matrix.
 *
 *  @retval sz_success_k The haystacks were counted, or on a device tier the counting was enqueued.
 *  @retval sz_unexpected_dimensions_k @p counts_stride is zero.
 *  @retval sz_device_memory_mismatch_k A device tier was handed an argument no kernel can address.
 *  @note Reads @c engine->capability to pick the table, then the slot for the tier @c _init_* resolved.
 *  @sa sz_substrings_counts_serial
 */
SZ_API_RUNTIME sz_status_t sz_substrings_counts(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                sz_size_t *counts, sz_size_t counts_stride);

/**
 *  @brief Reports every match of every needle in every haystack.
 *
 *  @param[in] engine A vocabulary compiled by @c _init_cpu or @c _init_gpu.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[out] matches Room for @p matches_capacity matches, ascending by haystack; may be @c SZ_NULL
 *              together with a zero capacity, which makes the call a pure size query.
 *  @param[in] matches_capacity Entries @p matches holds.
 *  @param[out] matches_offsets The @b [haystacks->count + 1] boundaries into @p matches, the last being the
 *              total; filled whether or not the matches fit, which is what sizes the next call.
 *
 *  @retval sz_success_k Every match was reported, or on a device tier the reporting was enqueued.
 *  @retval sz_device_memory_mismatch_k A device tier was handed an argument no kernel can address.
 *  @note A capacity too small is not an error: @c engine->report->matches_emitted names the true total and
 *        @c shortfall names what did not fit, so one sizing call and one filling call need no walk between.
 *  @sa sz_substrings_find_serial
 */
SZ_API_RUNTIME sz_status_t sz_substrings_find(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                              sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                              sz_size_t *matches_offsets);

/**
 *  @brief Rewrites every haystack, substituting one replacement per needle, onto one output tape.
 *
 *  @param[in] engine A vocabulary compiled by @c _init_cpu or @c _init_gpu, under a leftmost policy;
 *             @ref sz_substrings_overlapping_k is refused, since a substitution over matches that share
 *             bytes is not a function.
 *  @param[in] haystacks The texts to rewrite; their bytes are read in place and never copied.
 *  @param[in] replacements One replacement per needle, indexed by needle; an empty one deletes the match.
 *  @param[out] tape Room for @p tape_capacity bytes, or @c SZ_NULL with a zero capacity to size only.
 *  @param[in] tape_capacity Bytes @p tape holds.
 *  @param[out] offsets The @b [haystacks->count + 1] rewritten boundaries, the last being the total; these
 *              are filled whether or not the tape held the result, which is what sizes the next call.
 *
 *  @retval sz_success_k Every haystack was rewritten, or on a device tier the rewrite was enqueued.
 *  @retval sz_unexpected_dimensions_k @p replacements does not hold one entry per needle.
 *  @retval sz_status_unknown_k The engine's policy leaves no cover to substitute.
 *  @retval sz_device_memory_mismatch_k A device tier was handed an argument no kernel can address.
 *  @note A tape too small is not an error: @c engine->report->tape_bytes names the bytes the rewrite needs
 *        and @c shortfall names what did not fit, while @p tape 's contents are then unspecified.
 *  @sa sz_substrings_replace_serial
 */
SZ_API_RUNTIME sz_status_t sz_substrings_replace(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                 sz_sequence_t const *replacements, sz_ptr_t tape,
                                                 sz_size_t tape_capacity, sz_size_t *offsets);

/**
 *  @brief Scores every haystack against the vocabulary as one BM25 query, one score per haystack.
 *
 *  The vocabulary is the query: @p needle_weights holds each needle's IDF or boost. Term frequencies are raw
 *  overlapping counts, since a leftmost cover would suppress genuine occurrences of a needle nested in
 *  another, so the engine's own policy does not apply here. A CPU backend sums in ascending needle order and
 *  a CUDA backend in fixed-point integers, so each is bit-stable across its own runs, and the two agree.
 *
 *  @param[in] engine A vocabulary compiled by @c _init_cpu or @c _init_gpu.
 *  @param[in] haystacks The documents to score; their bytes are read in place and never copied.
 *  @param[in] document_lengths The @b [haystacks->count] lengths to normalize by, in any unit consistent with
 *             @c average_document_length, or @c SZ_NULL to use each haystack's byte length.
 *  @param[in] parameters BM25's continuous parameters.
 *  @param[in] needle_weights The @b [engine->needles_count] per-needle weights.
 *  @param[out] scores The @b [haystacks->count] scores, haystack @c h at @c scores[h*stride].
 *  @param[in] scores_stride Entries from one haystack's score to the next, at least one, so a strided call
 *             writes one column of a @b [haystacks, vocabularies] feature matrix.
 *
 *  @retval sz_success_k Every haystack was scored, or on a device tier the scoring was enqueued.
 *  @retval sz_unexpected_dimensions_k @p needle_weights is @c SZ_NULL, @p scores_stride is zero, or
 *          @c length_normalization is positive while @c average_document_length is not.
 *  @retval sz_device_memory_mismatch_k A device tier was handed an argument no kernel can address.
 *  @sa sz_substrings_bm25_scores_serial
 */
SZ_API_RUNTIME sz_status_t sz_substrings_bm25_scores(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_f32_t const *document_lengths,
                                                     sz_substrings_bm25_t const *parameters,
                                                     sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                     sz_size_t scores_stride);

/** @copydoc sz_substrings_counts */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_serial(sz_substrings_engine_t *engine,
                                                        sz_sequence_t const *haystacks, sz_size_t *counts,
                                                        sz_size_t counts_stride);

/** @copydoc sz_substrings_find */
SZ_API_COMPTIME sz_status_t sz_substrings_find_serial(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                      sz_size_t *matches_offsets);

/** @copydoc sz_substrings_replace */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_serial(sz_substrings_engine_t *engine,
                                                         sz_sequence_t const *haystacks,
                                                         sz_sequence_t const *replacements, sz_ptr_t tape,
                                                         sz_size_t tape_capacity, sz_size_t *offsets);

/** @copydoc sz_substrings_bm25_scores */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_serial(sz_substrings_engine_t *engine,
                                                             sz_sequence_t const *haystacks,
                                                             sz_f32_t const *document_lengths,
                                                             sz_substrings_bm25_t const *parameters,
                                                             sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                             sz_size_t scores_stride);

#if SZ_USE_HASWELL
/** @copydoc sz_substrings_counts */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_haswell(sz_substrings_engine_t *engine,
                                                         sz_sequence_t const *haystacks, sz_size_t *counts,
                                                         sz_size_t counts_stride);
/** @copydoc sz_substrings_find */
SZ_API_COMPTIME sz_status_t sz_substrings_find_haswell(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                       sz_size_t *matches_offsets);
/** @copydoc sz_substrings_replace */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_haswell(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_sequence_t const *replacements, sz_ptr_t tape,
                                                          sz_size_t tape_capacity, sz_size_t *offsets);
/** @copydoc sz_substrings_bm25_scores */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_haswell(sz_substrings_engine_t *engine,
                                                              sz_sequence_t const *haystacks,
                                                              sz_f32_t const *document_lengths,
                                                              sz_substrings_bm25_t const *parameters,
                                                              sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                              sz_size_t scores_stride);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_substrings_counts */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_icelake(sz_substrings_engine_t *engine,
                                                         sz_sequence_t const *haystacks, sz_size_t *counts,
                                                         sz_size_t counts_stride);
/** @copydoc sz_substrings_find */
SZ_API_COMPTIME sz_status_t sz_substrings_find_icelake(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                       sz_size_t *matches_offsets);
/** @copydoc sz_substrings_replace */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_icelake(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_sequence_t const *replacements, sz_ptr_t tape,
                                                          sz_size_t tape_capacity, sz_size_t *offsets);
/** @copydoc sz_substrings_bm25_scores */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_icelake(sz_substrings_engine_t *engine,
                                                              sz_sequence_t const *haystacks,
                                                              sz_f32_t const *document_lengths,
                                                              sz_substrings_bm25_t const *parameters,
                                                              sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                              sz_size_t scores_stride);
#endif

#if SZ_USE_NEON
/** @copydoc sz_substrings_counts */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_neon(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_size_t *counts, sz_size_t counts_stride);
/** @copydoc sz_substrings_find */
SZ_API_COMPTIME sz_status_t sz_substrings_find_neon(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                    sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                    sz_size_t *matches_offsets);
/** @copydoc sz_substrings_replace */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_neon(sz_substrings_engine_t *engine,
                                                       sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements, sz_ptr_t tape,
                                                       sz_size_t tape_capacity, sz_size_t *offsets);
/** @copydoc sz_substrings_bm25_scores */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_neon(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                           sz_size_t scores_stride);
#endif

#if SZ_USE_CUDA

/**
 *  @brief Compiles @p needles where a kernel can reach them, and sizes the round's arena from the budgets.
 *  @copydetails sz_substrings_engine_init_gpu
 */
SZ_API_COMPTIME sz_status_t sz_substrings_engine_init_cuda(sz_sequence_t const *needles,
                                                           sz_substrings_case_sensitivity_t case_sensitivity,
                                                           sz_substrings_overlap_policy_t overlap_policy,
                                                           sz_size_t hot_states, sz_size_t matches_budget,
                                                           sz_memory_allocator_t *alloc,
                                                           void *stream, sz_substrings_engine_t *engine);

/** @copydoc sz_substrings_counts */
SZ_API_COMPTIME sz_status_t sz_substrings_counts_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_size_t *counts, sz_size_t counts_stride);

/** @copydoc sz_substrings_find */
SZ_API_COMPTIME sz_status_t sz_substrings_find_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                    sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                    sz_size_t *matches_offsets);

/** @copydoc sz_substrings_replace */
SZ_API_COMPTIME sz_status_t sz_substrings_replace_cuda(sz_substrings_engine_t *engine,
                                                       sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements, sz_ptr_t tape,
                                                       sz_size_t tape_capacity, sz_size_t *offsets);

/** @copydoc sz_substrings_bm25_scores */
SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_cuda(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                           sz_size_t scores_stride);

#endif

#pragma endregion Core API

#include "stringzilla/substrings/serial.h"
#include "stringzilla/substrings/haswell.h"
#include "stringzilla/substrings/icelake.h"
#include "stringzilla/substrings/neon.h"
#include "stringzilla/substrings/cuda.cuh"

/*  Pick the right implementation for the multi-pattern search algorithms.
 *  To override this behavior and precompile all backends - set @c SZ_DYNAMIC_DISPATCH to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_status_t sz_substrings_engine_init_cpu(sz_sequence_t const *needles,
                                                         sz_substrings_case_sensitivity_t case_sensitivity,
                                                         sz_substrings_overlap_policy_t overlap_policy,
                                                         sz_size_t hot_states, sz_size_t matches_budget,
                                                         sz_memory_allocator_t *alloc,
                                                         sz_substrings_engine_t *engine) {
#if SZ_USE_ICELAKE
    sz_capability_t const capability = sz_cap_icelake_k;
#elif SZ_USE_HASWELL
    sz_capability_t const capability = sz_cap_haswell_k;
#elif SZ_USE_NEON
    sz_capability_t const capability = sz_cap_neon_k;
#else
    sz_capability_t const capability = sz_cap_serial_k;
#endif
    return sz_substrings_engine_build_(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                       capability, alloc, engine);
}

#if SZ_USE_CUDA
SZ_API_RUNTIME sz_status_t sz_substrings_engine_init_gpu(sz_sequence_t const *needles,
                                                         sz_substrings_case_sensitivity_t case_sensitivity,
                                                         sz_substrings_overlap_policy_t overlap_policy,
                                                         sz_size_t hot_states, sz_size_t matches_budget,
                                                         sz_memory_allocator_t *alloc,
                                                         void *stream, sz_substrings_engine_t *engine) {
    return sz_substrings_engine_init_cuda(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                       alloc, stream, engine);
}
#endif

SZ_API_RUNTIME void sz_substrings_engine_free(sz_substrings_engine_t *engine) { sz_substrings_engine_free_(engine); }

SZ_API_RUNTIME sz_status_t sz_substrings_counts(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                sz_size_t *counts, sz_size_t counts_stride) {
#if SZ_USE_CUDA
    if (engine->capability & sz_caps_cuda_k) return sz_substrings_counts_cuda(engine, haystacks, counts, counts_stride);
#endif
#if SZ_USE_ICELAKE
    return sz_substrings_counts_icelake(engine, haystacks, counts, counts_stride);
#elif SZ_USE_HASWELL
    return sz_substrings_counts_haswell(engine, haystacks, counts, counts_stride);
#elif SZ_USE_NEON
    return sz_substrings_counts_neon(engine, haystacks, counts, counts_stride);
#else
    return sz_substrings_counts_serial(engine, haystacks, counts, counts_stride);
#endif
}

SZ_API_RUNTIME sz_status_t sz_substrings_find(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                              sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                              sz_size_t *matches_offsets) {
#if SZ_USE_CUDA
    if (engine->capability & sz_caps_cuda_k)
        return sz_substrings_find_cuda(engine, haystacks, matches, matches_capacity, matches_offsets);
#endif
#if SZ_USE_ICELAKE
    return sz_substrings_find_icelake(engine, haystacks, matches, matches_capacity, matches_offsets);
#elif SZ_USE_HASWELL
    return sz_substrings_find_haswell(engine, haystacks, matches, matches_capacity, matches_offsets);
#elif SZ_USE_NEON
    return sz_substrings_find_neon(engine, haystacks, matches, matches_capacity, matches_offsets);
#else
    return sz_substrings_find_serial(engine, haystacks, matches, matches_capacity, matches_offsets);
#endif
}

SZ_API_RUNTIME sz_status_t sz_substrings_replace(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                 sz_sequence_t const *replacements, sz_ptr_t tape,
                                                 sz_size_t tape_capacity, sz_size_t *offsets) {
#if SZ_USE_CUDA
    if (engine->capability & sz_caps_cuda_k)
        return sz_substrings_replace_cuda(engine, haystacks, replacements, tape, tape_capacity, offsets);
#endif
#if SZ_USE_ICELAKE
    return sz_substrings_replace_icelake(engine, haystacks, replacements, tape, tape_capacity, offsets);
#elif SZ_USE_HASWELL
    return sz_substrings_replace_haswell(engine, haystacks, replacements, tape, tape_capacity, offsets);
#elif SZ_USE_NEON
    return sz_substrings_replace_neon(engine, haystacks, replacements, tape, tape_capacity, offsets);
#else
    return sz_substrings_replace_serial(engine, haystacks, replacements, tape, tape_capacity, offsets);
#endif
}

SZ_API_RUNTIME sz_status_t sz_substrings_bm25_scores(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_f32_t const *document_lengths,
                                                     sz_substrings_bm25_t const *parameters,
                                                     sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                     sz_size_t scores_stride) {
#if SZ_USE_CUDA
    if (engine->capability & sz_caps_cuda_k)
        return sz_substrings_bm25_scores_cuda(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                              scores_stride);
#endif
#if SZ_USE_ICELAKE
    return sz_substrings_bm25_scores_icelake(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                             scores_stride);
#elif SZ_USE_HASWELL
    return sz_substrings_bm25_scores_haswell(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                             scores_stride);
#elif SZ_USE_NEON
    return sz_substrings_bm25_scores_neon(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                          scores_stride);
#else
    return sz_substrings_bm25_scores_serial(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                            scores_stride);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SUBSTRINGS_H_
