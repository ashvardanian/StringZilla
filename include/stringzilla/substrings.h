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
 *  - @ref sz_substrings_engine_init_cpu → compiles a vocabulary the host verbs read;
 *  - @ref sz_substrings_engine_init_gpu → compiles the same vocabulary where a kernel can reach it;
 *  - @ref sz_substrings_counts → how many matches each haystack holds;
 *  - @ref sz_substrings_find → every match, located by haystack, needle and byte span;
 *  - @ref sz_substrings_replace → the haystacks rewritten with one replacement per needle;
 *  - @ref sz_substrings_bm25_scores → one BM25 score per haystack, the vocabulary being the query.
 *
 *  The engine is a plain struct of flat arrays over two owned blocks, so a caller driving its own
 *  loops binds @ref sz_substrings_step and walks it directly, and a device backend passes it to a
 *  kernel by value.
 */
#ifndef STRINGZILLA_SUBSTRINGS_H_
#define STRINGZILLA_SUBSTRINGS_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/** Whether a vocabulary matches needles byte-for-byte, or folds both sides to one case first. */
typedef enum sz_substrings_case_sensitivity_t {

    /** Byte-exact matching; needles may be arbitrary bytes, including malformed UTF-8. */
    sz_substrings_cased_k = 0,

    /** Full Unicode case folding as `CaseFolding.txt` defines it; needles must be valid UTF-8. */
    sz_substrings_uncased_k = 1,
} sz_substrings_case_sensitivity_t;

/**
 *  @brief One needle ending on one state, as the engine stores it.
 *
 *  The engine walks @b folded bytes, so this length is the folded one - the needle's own, identical
 *  for every match of that needle. The @b source span it corresponds to is not: needle "k" matches
 *  both the 1-byte "k" and the 3-byte Kelvin sign U+212A. Recovering that span is the walk's job.
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
 *  Under case folding a needle's own byte length is not the length of every match, so the span is
 *  carried per match rather than looked up from the needle.
 */
typedef struct sz_substrings_match_t {

    /** Which haystack of the sequence this match was found in. */
    sz_size_t haystack_index;

    /** Which needle of the vocabulary matched. */
    sz_size_t needle_index;

    /** Where the match starts inside that haystack, in its own bytes. */
    sz_size_t byte_offset;

    /** Haystack bytes the match spans, which folding can make differ from the needle's length. */
    sz_size_t byte_length;
} sz_substrings_match_t;

/**
 *  @brief One start position's incumbent match while a leftmost policy is still deciding it.
 *
 *  Separate from @ref sz_substrings_output_t because the units differ: an output carries the folded
 *  length the automaton walked, this carries the @b source span that length resolved to.
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
 *  A device verb enqueues and returns, so nothing it discovers can reach the caller through a
 *  status code. The sizing walk always runs, so @c matches_emitted is the truth whatever the
 *  caller's output could hold, and a nonzero @c shortfall is the one signal that the output is
 *  incomplete rather than wrong. Under a leftmost policy a cover thins the matches after that walk,
 *  so for a round that stayed inside its budget the matches a caller could have read are
 *  @c matches_stored plus @c shortfall, which is also what the last boundary of a
 *  @ref sz_substrings_find names; a round that outran its budget ran no cover at all, and its
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
 *  @brief The compiled vocabulary, its sizing policy, and the round's arena, in one lifetime.
 *
 *  Transitions are split into two tiers by how often a state is visited. Text keeps resetting the
 *  walk toward the root, so a small set of states absorbs most byte steps whatever the dictionary
 *  size, and the tiers are sized to that skew rather than to the vocabulary as a whole.
 *
 *  The @b hot tier is a dense goto-completed table, one row per state with one target per byte
 *  @b class, so a step is a single load with no branch and no failure chasing. A class is a byte
 *  some needle spells, or the one shared class of every byte none does, so a row is as wide as the
 *  vocabulary's own alphabet: five targets for nucleotides, a few hundred for multilingual text,
 *  and never more than 256. The @b cold tier is a double array: @c base and @c check encode
 *  transitions as address arithmetic plus an ownership test, and @c fail restores the failure links
 *  that goto completion would otherwise have folded away. States are numbered so the hot ones come
 *  first, which makes the tier test state < hot_count with no lookup.
 *
 *  Both blocks are built by @c _init_cpu or @c _init_gpu and live until
 *  @ref sz_substrings_engine_free, so a compute verb allocates nothing and a device backend has no
 *  host clone to stage.
 */
typedef struct sz_substrings_engine_t {

    /** Hot tier: hot_count × classes_count goto-completed targets, row-major, shallow first. */
    sz_u32_t const *hot_rows;

    /** Each byte's column in @c hot_rows: its own if a needle spells it, else one shared column. */
    sz_u8_t const *byte_to_class;

    /** Cold tier: transition target for a state on a byte is base[state] + byte, if owned. */
    sz_u32_t const *base;

    /** Cold tier: owner of each slot, so a collision reads as a missing edge, not a wrong one. */
    sz_u32_t const *check;

    /** Cold tier: failure link, followed when @c check denies ownership. */
    sz_u32_t const *fail;

    /** One bit per slot: whether any needle ends there, so a non-matching byte touches no count. */
    sz_u32_t const *accepts_words;

    /** Matches ending at each state, flattened and merged along failure chains at build time. */
    sz_substrings_output_t const *outputs;

    /** Outputs per slot, in the published numbering. */
    sz_u32_t const *outputs_counts;

    /** Exclusive prefix sum of @c outputs_counts; a nested-suffix vocabulary drives the pool to
     *  O(states²), so these stay pointer-wide where the counts do not. */
    sz_size_t const *outputs_offsets;

    /** Length of @c outputs, so a consumer never rescans the CSR to recover it. */
    sz_size_t outputs_total;

    /** Slots every cold-tier array holds: @c state_count plus the alphabet's address headroom. */
    sz_size_t slots_count;

    /** States below @c hot_count live in @c hot_rows; the rest live in the double array. */
    sz_u32_t hot_count;

    /** Columns of a hot row, at most 256. */
    sz_u32_t classes_count;

    /** Published state ceiling, above the trie's, as a packed child's id is address arithmetic. */
    sz_u32_t state_count;

    /** The root's published id, which is always zero. */
    sz_u32_t root;

    /** Needles the vocabulary holds, which bounds every reported @c needle_index. */
    sz_u32_t needles_count;

    /** Most @b haystack bytes one match can span, sizing every slice, halo and warm-up. */
    sz_u32_t max_source_match_bytes;

    /** Fewest haystack bytes one match can span; the mirror bound. */
    sz_u32_t min_source_match_bytes;

    /** Most merged outputs any state carries, so a consumer can bound one pass's match count. */
    sz_u32_t max_outputs_per_state;

    /** Whether a walk folds the haystack as it consumes it, or steps it byte for byte. */
    sz_substrings_case_sensitivity_t case_sensitivity;

    /** Bytes that move a walk off the root, which a byte search can skip to while it sits there. */
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
 *  A CPU backend settles a cover during the walk and always reports the leftmost one. A CUDA
 *  backend settles it afterwards, one run of mutually-reaching matches per thread, and a greedy
 *  over such a run is quadratic in it - so past a bound it accepts in the order the walk emitted
 *  instead. That is the same cover whenever match starts ascend with their ends, and a coarser one
 *  when they do not, which happens when a vocabulary is so dense that no gap ever separates any two
 *  of its matches.
 *
 *  Both answers are covers: every match is real and no two share a byte. Only the choice among
 *  rivals differs, so a caller that needs the leftmost cover exactly asks a CPU backend for it.
 */
typedef enum sz_substrings_cover_fidelity_t {

    /** The leftmost cover itself, which every CPU backend reports. */
    sz_substrings_cover_exact_k = 0,

    /** A valid cover that may differ among rivals, as CUDA reports on dense vocabularies. */
    sz_substrings_cover_approximate_k = 1,
} sz_substrings_cover_fidelity_t;

/** BM25's continuous parameters. */
typedef struct sz_substrings_bm25_t {

    /** The literature's k₁: how slowly repeated occurrences stop adding score; 1.2 is customary. */
    sz_f32_t term_frequency_saturation;

    /** The literature's b, in [0, 1]: 0 ignores document length and 1 normalizes it fully, while
     *  0.75 is the customary choice. */
    sz_f32_t length_normalization;

    /** The corpus-wide mean document length, in the unit of the lengths scored; read only when
     *  @c length_normalization is positive. */
    sz_f32_t average_document_length;
} sz_substrings_bm25_t;

/** The @c hot_states value that sizes the hot tier itself, so zero stays a genuine all-cold ask. */
#define STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO (STRINGZILLA_SIZE_MAX)

/**
 *  @brief Compiles @p needles into an engine the matching verbs read, on the host.
 *
 *  @param[in] needles The vocabulary; an empty needle is refused rather than skipped, as it would
 *      match at every position and dropping it would shift every later needle's reported index.
 *  @param[in] case_sensitivity Whether both sides are folded before they meet, or compared as is.
 *  @param[in] overlap_policy The cover every round runs, as the arena is sized for that one alone.
 *  @param[in] hot_states States to keep in the dense hot rows, or
 *      @ref STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO to fill a fixed byte budget, which holds more
 *      states the fewer classes the vocabulary spells.
 *  @param[in] matches_budget Matches one round may emit, read by a device tier only; a host tier
 *      walks straight into the caller's output and ignores it. Zero asks for a tier-chosen default.
 *  @param[in] alloc Where both blocks come from, or @c STRINGZILLA_NULL for the default host
 *      allocator. Stored by value, so @ref sz_substrings_engine_free needs none and cannot get
 *      the wrong one.
 *  @param[out] engine Left untouched unless the call succeeds.
 *  @return @c sz_success_k once the vocabulary compiled, @c sz_bad_alloc_k if memory allocation
 *      failed, @c sz_overflow_risk_k if the vocabulary exceeds a 32-bit state id,
 *      @c sz_invalid_utf8_k for a malformed needle under @ref sz_substrings_uncased_k, or
 *      @c sz_unexpected_dimensions_k if the vocabulary or one of its needles was empty.
 *  @sa sz_substrings_engine_free
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_engine_init_cpu(sz_sequence_t const *needles,
                                                                  sz_substrings_case_sensitivity_t case_sensitivity,
                                                                  sz_substrings_overlap_policy_t overlap_policy,
                                                                  sz_size_t hot_states, sz_size_t matches_budget,
                                                                  sz_memory_allocator_t *alloc,
                                                                  sz_substrings_engine_t *engine);

/**
 *  @brief Compiles @p needles into an engine on @p stream 's device, arena included.
 *
 *  @param[in] alloc Unified and bound to @p stream 's device, or @c STRINGZILLA_NULL to derive
 *      one from it; the host builder writes the automaton in place, so a device-only block
 *      cannot serve here.
 *  @param[in] stream A @c cudaStream_t, or @c STRINGZILLA_NULL for the default stream.
 *  @note Also returns @c sz_device_memory_mismatch_k when the block @p alloc handed back does not
 *      reach the device.
 *  @note May join @p stream; no compute verb ever does.
 *
 *  @copydetails sz_substrings_engine_init_cpu
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_engine_init_gpu(sz_sequence_t const *needles,
                                                                  sz_substrings_case_sensitivity_t case_sensitivity,
                                                                  sz_substrings_overlap_policy_t overlap_policy,
                                                                  sz_size_t hot_states, sz_size_t matches_budget,
                                                                  sz_memory_allocator_t *alloc, void *stream,
                                                                  sz_substrings_engine_t *engine);

/** Returns both of the engine's blocks to the allocator that built them, emptying @p engine. */
STRINGZILLA_API_RUNTIME void sz_substrings_engine_free(sz_substrings_engine_t *engine);

/**
 *  @brief Counts the matches of every needle in every haystack, one count per haystack.
 *
 *  @param[in] engine A vocabulary compiled by @c _init_cpu or @c _init_gpu, whose @c overlap_policy
 *      decides whether matches sharing bytes are all counted or thinned to a leftmost run.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[out] counts The per-haystack counts, haystack h at `counts[h * stride]`.
 *  @param[in] counts_stride Entries from one haystack's count to the next, at least one, so a
 *      strided call writes one column of a @b [haystacks, vocabularies] feature matrix.
 *  @return @c sz_success_k once the haystacks were counted, or on a device tier the counting
 *      enqueued; @c sz_unexpected_dimensions_k if @p counts_stride is zero; or
 *      @c sz_device_memory_mismatch_k if a device tier got an argument no kernel can address.
 *  @note Reads the engine's capability to pick the table, then the slot of the tier init resolved.
 *  @sa sz_substrings_counts_serial
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_counts(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                         sz_size_t *counts, sz_size_t counts_stride);

/**
 *  @brief Reports every match of every needle in every haystack.
 *
 *  @param[in] engine A vocabulary compiled by @c _init_cpu or @c _init_gpu.
 *  @param[in] haystacks The texts to search; their bytes are read in place and never copied.
 *  @param[out] matches Room for @p matches_capacity matches, ascending by haystack;
 *      @c STRINGZILLA_NULL together with a zero capacity makes the call a pure size query.
 *  @param[in] matches_capacity Entries @p matches holds.
 *  @param[out] matches_offsets One boundary per haystack into @p matches plus one, the last being
 *      the total; filled whether or not the matches fit, which is what sizes the next call.
 *  @return @c sz_success_k once every match was reported, or on a device tier the reporting
 *      enqueued, or @c sz_device_memory_mismatch_k if a device tier got an unaddressable argument.
 *
 *  @sa sz_substrings_find_serial
 *
 *  A capacity too small is not an error: the report's @c matches_emitted names the true total and
 *  @c shortfall names what did not fit, so a sizing call and a filling call need no walk between.
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_find(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                       sz_size_t *matches_offsets);

/**
 *  @brief Rewrites every haystack, substituting one replacement per needle, onto one output tape.
 *
 *  @param[in] engine A vocabulary compiled by @c _init_cpu or @c _init_gpu under a leftmost policy;
 *      @ref sz_substrings_overlapping_k is refused, since a substitution over matches that share
 *      bytes is not a function.
 *  @param[in] haystacks The texts to rewrite; their bytes are read in place and never copied.
 *  @param[in] replacements One replacement per needle, indexed by needle; an empty one deletes.
 *  @param[out] tape Room for @p tape_capacity bytes, or @c STRINGZILLA_NULL with zero capacity
 *      to size only.
 *  @param[in] tape_capacity Bytes @p tape holds.
 *  @param[out] offsets One rewritten boundary per haystack plus one, the last being the total;
 *      filled whether or not the tape held the result, which is what sizes the next call.
 *  @return @c sz_success_k once every haystack was rewritten, or on a device tier the rewrite
 *      enqueued; @c sz_unexpected_dimensions_k if @p replacements lacks one entry per needle;
 *      @c sz_status_unknown_k if the policy leaves no cover; or @c sz_device_memory_mismatch_k.
 *
 *  @sa sz_substrings_replace_serial
 *
 *  A tape too small is not an error: the report's @c tape_bytes names the bytes the rewrite needs
 *  and @c shortfall names what did not fit, while the contents of @p tape are then unspecified. A
 *  device tier returns @c sz_device_memory_mismatch_k for an argument no kernel can address.
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_replace(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_sequence_t const *replacements, sz_ptr_t tape,
                                                          sz_size_t tape_capacity, sz_size_t *offsets);

/**
 *  @brief Scores every haystack against the vocabulary as one BM25 query, one score per haystack.
 *
 *  The vocabulary is the query: @p needle_weights holds each needle's IDF or boost. Term
 *  frequencies are raw overlapping counts, since a leftmost cover would suppress genuine
 *  occurrences of a needle nested in another, so the engine's own policy does not apply here. A CPU
 *  backend sums in ascending needle order and a CUDA backend in fixed-point integers, so each is
 *  bit-stable across its own runs, and the two agree.
 *
 *  @param[in] engine A vocabulary compiled by @c _init_cpu or @c _init_gpu.
 *  @param[in] haystacks The documents to score; their bytes are read in place and never copied.
 *  @param[in] document_lengths One length per haystack to normalize by, in the unit of
 *      @c average_document_length, or @c STRINGZILLA_NULL to use each haystack's byte length.
 *  @param[in] parameters BM25's continuous parameters.
 *  @param[in] needle_weights One weight per needle of the engine.
 *  @param[out] scores The per-haystack scores, haystack h at `scores[h * stride]`.
 *  @param[in] scores_stride Entries from one haystack's score to the next, at least one, so a
 *      strided call writes one column of a @b [haystacks, vocabularies] feature matrix.
 *  @return @c sz_success_k once every haystack was scored, or on a device tier the scoring
 *      enqueued; @c sz_unexpected_dimensions_k for invalid arguments, as below; or
 *      @c sz_device_memory_mismatch_k if a device tier got an argument no kernel can address.
 *  @sa sz_substrings_bm25_scores_serial
 *
 *  The arguments are invalid when @p needle_weights is @c STRINGZILLA_NULL, @p scores_stride is
 *  zero, or @c length_normalization is positive while @c average_document_length is not.
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_bm25_scores(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride);

/** @copydoc sz_substrings_counts */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_counts_serial(sz_substrings_engine_t *engine,
                                                                 sz_sequence_t const *haystacks, sz_size_t *counts,
                                                                 sz_size_t counts_stride);

/** @copydoc sz_substrings_find */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_find_serial(sz_substrings_engine_t *engine,
                                                               sz_sequence_t const *haystacks,
                                                               sz_substrings_match_t *matches,
                                                               sz_size_t matches_capacity, sz_size_t *matches_offsets);

/** @copydoc sz_substrings_replace */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_replace_serial(sz_substrings_engine_t *engine,
                                                                  sz_sequence_t const *haystacks,
                                                                  sz_sequence_t const *replacements, sz_ptr_t tape,
                                                                  sz_size_t tape_capacity, sz_size_t *offsets);

/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_bm25_scores_serial(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_substrings_counts */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_counts_haswell(sz_substrings_engine_t *engine,
                                                                  sz_sequence_t const *haystacks, sz_size_t *counts,
                                                                  sz_size_t counts_stride);
/** @copydoc sz_substrings_find */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_find_haswell(sz_substrings_engine_t *engine,
                                                                sz_sequence_t const *haystacks,
                                                                sz_substrings_match_t *matches,
                                                                sz_size_t matches_capacity, sz_size_t *matches_offsets);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_replace_haswell(sz_substrings_engine_t *engine,
                                                                   sz_sequence_t const *haystacks,
                                                                   sz_sequence_t const *replacements, sz_ptr_t tape,
                                                                   sz_size_t tape_capacity, sz_size_t *offsets);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_bm25_scores_haswell(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_substrings_counts */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_counts_icelake(sz_substrings_engine_t *engine,
                                                                  sz_sequence_t const *haystacks, sz_size_t *counts,
                                                                  sz_size_t counts_stride);
/** @copydoc sz_substrings_find */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_find_icelake(sz_substrings_engine_t *engine,
                                                                sz_sequence_t const *haystacks,
                                                                sz_substrings_match_t *matches,
                                                                sz_size_t matches_capacity, sz_size_t *matches_offsets);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_replace_icelake(sz_substrings_engine_t *engine,
                                                                   sz_sequence_t const *haystacks,
                                                                   sz_sequence_t const *replacements, sz_ptr_t tape,
                                                                   sz_size_t tape_capacity, sz_size_t *offsets);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_bm25_scores_icelake(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_substrings_counts */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_counts_neon(sz_substrings_engine_t *engine,
                                                               sz_sequence_t const *haystacks, sz_size_t *counts,
                                                               sz_size_t counts_stride);
/** @copydoc sz_substrings_find */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_find_neon(sz_substrings_engine_t *engine,
                                                             sz_sequence_t const *haystacks,
                                                             sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                             sz_size_t *matches_offsets);
/** @copydoc sz_substrings_replace */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_replace_neon(sz_substrings_engine_t *engine,
                                                                sz_sequence_t const *haystacks,
                                                                sz_sequence_t const *replacements, sz_ptr_t tape,
                                                                sz_size_t tape_capacity, sz_size_t *offsets);
/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_bm25_scores_neon(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride);
#endif

#if STRINGZILLA_TARGET_CUDA

/**
 *  @brief Compiles @p needles where a kernel can reach them, sizing the round's arena from budgets.
 *
 *  @copydetails sz_substrings_engine_init_gpu
 */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_engine_init_cuda(sz_sequence_t const *needles,
                                                                    sz_substrings_case_sensitivity_t case_sensitivity,
                                                                    sz_substrings_overlap_policy_t overlap_policy,
                                                                    sz_size_t hot_states, sz_size_t matches_budget,
                                                                    sz_memory_allocator_t *alloc, void *stream,
                                                                    sz_substrings_engine_t *engine);

/** @copydoc sz_substrings_counts */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_counts_cuda(sz_substrings_engine_t *engine,
                                                               sz_sequence_t const *haystacks, sz_size_t *counts,
                                                               sz_size_t counts_stride);

/** @copydoc sz_substrings_find */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_find_cuda(sz_substrings_engine_t *engine,
                                                             sz_sequence_t const *haystacks,
                                                             sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                             sz_size_t *matches_offsets);

/** @copydoc sz_substrings_replace */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_replace_cuda(sz_substrings_engine_t *engine,
                                                                sz_sequence_t const *haystacks,
                                                                sz_sequence_t const *replacements, sz_ptr_t tape,
                                                                sz_size_t tape_capacity, sz_size_t *offsets);

/** @copydoc sz_substrings_bm25_scores */
STRINGZILLA_API_COMPTIME sz_status_t sz_substrings_bm25_scores_cuda(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride);

#endif

#pragma endregion Core API

#include "stringzilla/substrings/serial.h"
#include "stringzilla/substrings/haswell.h"
#include "stringzilla/substrings/icelake.h"
#include "stringzilla/substrings/neon.h"
#include "stringzilla/substrings/cuda.cuh"

/*  Pick the right implementation for the multi-pattern search algorithms. To override this behavior
 *  and precompile all backends - set @c STRINGZILLA_RUNTIME_DISPATCH to 1. */
#pragma region Compile Time Dispatching
#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_engine_init_cpu(sz_sequence_t const *needles,
                                                                  sz_substrings_case_sensitivity_t case_sensitivity,
                                                                  sz_substrings_overlap_policy_t overlap_policy,
                                                                  sz_size_t hot_states, sz_size_t matches_budget,
                                                                  sz_memory_allocator_t *alloc,
                                                                  sz_substrings_engine_t *engine) {
#if STRINGZILLA_TARGET_ICELAKE
    sz_capability_t const capability = sz_cap_icelake_k;
#elif STRINGZILLA_TARGET_HASWELL
    sz_capability_t const capability = sz_cap_haswell_k;
#elif STRINGZILLA_TARGET_NEON
    sz_capability_t const capability = sz_cap_neon_k;
#else
    sz_capability_t const capability = sz_cap_serial_k;
#endif
    return sz_substrings_engine_build_(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                       capability, alloc, engine);
}

#if STRINGZILLA_TARGET_CUDA
STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_engine_init_gpu(sz_sequence_t const *needles,
                                                                  sz_substrings_case_sensitivity_t case_sensitivity,
                                                                  sz_substrings_overlap_policy_t overlap_policy,
                                                                  sz_size_t hot_states, sz_size_t matches_budget,
                                                                  sz_memory_allocator_t *alloc, void *stream,
                                                                  sz_substrings_engine_t *engine) {
    return sz_substrings_engine_init_cuda(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                       alloc, stream, engine);
}
#endif

STRINGZILLA_API_RUNTIME void sz_substrings_engine_free(sz_substrings_engine_t *engine) {
    sz_substrings_engine_free_(engine);
}

STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_counts(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                         sz_size_t *counts, sz_size_t counts_stride) {
#if STRINGZILLA_TARGET_CUDA
    if (engine->capability & sz_caps_cuda_k) return sz_substrings_counts_cuda(engine, haystacks, counts, counts_stride);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    return sz_substrings_counts_icelake(engine, haystacks, counts, counts_stride);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_substrings_counts_haswell(engine, haystacks, counts, counts_stride);
#elif STRINGZILLA_TARGET_NEON
    return sz_substrings_counts_neon(engine, haystacks, counts, counts_stride);
#else
    return sz_substrings_counts_serial(engine, haystacks, counts, counts_stride);
#endif
}

STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_find(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                       sz_size_t *matches_offsets) {
#if STRINGZILLA_TARGET_CUDA
    if (engine->capability & sz_caps_cuda_k)
        return sz_substrings_find_cuda(engine, haystacks, matches, matches_capacity, matches_offsets);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    return sz_substrings_find_icelake(engine, haystacks, matches, matches_capacity, matches_offsets);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_substrings_find_haswell(engine, haystacks, matches, matches_capacity, matches_offsets);
#elif STRINGZILLA_TARGET_NEON
    return sz_substrings_find_neon(engine, haystacks, matches, matches_capacity, matches_offsets);
#else
    return sz_substrings_find_serial(engine, haystacks, matches, matches_capacity, matches_offsets);
#endif
}

STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_replace(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_sequence_t const *replacements, sz_ptr_t tape,
                                                          sz_size_t tape_capacity, sz_size_t *offsets) {
#if STRINGZILLA_TARGET_CUDA
    if (engine->capability & sz_caps_cuda_k)
        return sz_substrings_replace_cuda(engine, haystacks, replacements, tape, tape_capacity, offsets);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    return sz_substrings_replace_icelake(engine, haystacks, replacements, tape, tape_capacity, offsets);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_substrings_replace_haswell(engine, haystacks, replacements, tape, tape_capacity, offsets);
#elif STRINGZILLA_TARGET_NEON
    return sz_substrings_replace_neon(engine, haystacks, replacements, tape, tape_capacity, offsets);
#else
    return sz_substrings_replace_serial(engine, haystacks, replacements, tape, tape_capacity, offsets);
#endif
}

STRINGZILLA_API_RUNTIME sz_status_t sz_substrings_bm25_scores(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride) {
#if STRINGZILLA_TARGET_CUDA
    if (engine->capability & sz_caps_cuda_k)
        return sz_substrings_bm25_scores_cuda(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                              scores_stride);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    return sz_substrings_bm25_scores_icelake(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                             scores_stride);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_substrings_bm25_scores_haswell(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                             scores_stride);
#elif STRINGZILLA_TARGET_NEON
    return sz_substrings_bm25_scores_neon(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                          scores_stride);
#else
    return sz_substrings_bm25_scores_serial(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                            scores_stride);
#endif
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SUBSTRINGS_H_
