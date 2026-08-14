/**
 *  @brief  Hardware-accelerated multi-pattern exact and case-folded substring search (serial backend).
 *  @file   include/stringzillas/substrings/serial.hpp
 *  @author Ash Vardanian
 *
 *  Implements the Aho-Corasick automaton at the core of the multi-pattern search engine:
 *
 *  - `aho_corasick_dictionary` builds the trie, derives the walking automaton (below), splits the result
 *    into a shallow-first, out-degree-ordered hot tier and a double-array cold tier, and exposes CSR-flattened
 *    match outputs through the shared `aho_corasick_view` contract. Construction allocates a constant
 *    number of flat buffers rather than one per state, and none of them outlives `try_build`.
 *  - `substrings` wraps the dictionary behind a `try_build` / `try_count` / `try_find` triple, with a
 *    single-threaded serial specialization and a two-level parallel one - one haystack per core below the
 *    L2 size, all cores on one haystack above it.
 *
 *  Every other StringZillas engine exposes a single `operator()`; the triple exists here because a
 *  dictionary is compiled once and reused across many later calls, and counting is useful in its own right.
 *
 *  Case folding lives in the @b stream, not in the automaton. A needle is folded once at build time and
 *  inserted byte for byte, exactly like a cased one, so the trie stays a tree and Aho-Corasick failure links
 *  stay single-valued. A haystack is folded one codepoint at a time as the walk consumes it - never into a
 *  buffer - by a cursor built on the same folded iterator the single-pattern engine walks with.
 *
 *  Inverting the fold onto the needle instead cannot express a match beginning or ending part-way through an
 *  expansion, and cannot be repaired: an automaton whose alphabet is raw haystack bytes and which must report
 *  match spans is exponential in needle length, since a self-overlapping needle's span encodes the byte width
 *  of every spelling behind it.
 *
 *  Folded byte offsets equal source byte offsets everywhere except across the 139 codepoints whose fold
 *  breaks a codepoint boundary, so a match's source span is plain arithmetic unless it touches one of those.
 */
#ifndef STRINGZILLAS_SUBSTRINGS_SERIAL_HPP_
#define STRINGZILLAS_SUBSTRINGS_SERIAL_HPP_

#include "stringzilla/types.hpp"                  // `status_t::status_t`
#include "stringzilla/memory.h"                   // `sz_copy`
#include "stringzilla/utf8_runes/serial.h"        // `sz_rune_decode`, `sz_rune_encode`
#include "stringzilla/utf8_uncased.h"             // `sz_utf8_folded_reverse_iter_t`, via its own backends
#include "stringzilla/utf8_uncased_fold/serial.h" // `sz_unicode_fold_codepoint_`, `sz_ascii_fold_`
#include "stringzillas/types.hpp"                 // `dummy_executor_t`

#include <forkunion/types.hpp> // `indexed_split_t` - the balanced range split every executor uses

#include <limits>      // `std::numeric_limits` for numeric types
#include <memory>      // `std::allocator_traits` to re-bind the allocator
#include <type_traits> // `std::enable_if_t` for meta-programming
#include <variant>     // `std::variant` holds the automaton at whichever state-id width it fits

namespace ashvardanian {
namespace stringzillas {

namespace fu = ashvardanian::forkunion;

// Per-symbol: a using-directive re-exports our `memcpy` and nvcc then finds the call ambiguous.
using ashvardanian::stringzilla::byte_t;
using ashvardanian::stringzilla::dummy_alloc_t;
using ashvardanian::stringzilla::size_t;
using ashvardanian::stringzilla::span;
using ashvardanian::stringzilla::to_bytes_view;

#pragma region Vocabulary

/** @brief Whether a dictionary matches needles byte-for-byte or folds both sides to a shared case first. */
enum substrings_case_sensitivity_t {
    /** @brief Byte-exact matching; needles may be arbitrary bytes. */
    substrings_cased_k,
    /** @brief Full Unicode case folding; needles must be valid UTF-8. */
    substrings_uncased_k,
};

/** @brief How matches that share bytes resolve: reported in full, or thinned to a leftmost run. */
enum substrings_overlap_policy_t {
    /** @brief Every match of every needle, including ones that share bytes and ones nested in others. */
    substrings_overlapping_k,
    /** @brief Matches sharing no bytes: earliest start, then longest span, then lower needle index. */
    substrings_leftmost_longest_k,
    /** @brief Matches sharing no bytes: earliest start, then lower needle index, however long the rival. */
    substrings_leftmost_first_k,
};

/**
 *  @brief  One reported match: which needle matched, and how long it is in the bytes the automaton walked.
 *
 *  The automaton walks @b folded bytes, so this length is the folded one - the needle's own, identical for
 *  every match of that needle. The @b source span it corresponds to is not: needle "k" matches both the
 *  1-byte "k" and the 3-byte Kelvin sign U+212A. Recovering that span is the walk's job, not this struct's.
 */
template <typename state_id_type_>
struct substrings_output {
    using state_id_t = state_id_type_;

    state_id_t needle_index {};
    /** @brief Folded bytes this match spans; a walk traverses one edge per byte, so it fits a state id. */
    state_id_t folded_match_bytes {};
};

/**
 *  @brief One reported match, locating it by haystack, by needle, and by byte span - the one match shape
 *         every backend emits, layout-identical to the C ABI's `szs_substrings_match_t`.
 *
 *  Under case folding a needle's own byte length is not the length of every match - needle "k" matches both
 *  the 1-byte "k" and the 3-byte Kelvin sign - so the span is carried per match. The matched bytes, when
 *  needed, are `to_bytes_view(haystacks[match.haystack_index]).subspan(byte_offset, byte_length)`.
 */
struct substrings_match_t {
    size_t haystack_index {};
    size_t needle_index {};
    size_t byte_offset {};
    size_t byte_length {};

    /** @brief All four fields, since two matches agreeing on three of them are still different matches. */
    friend bool operator==(substrings_match_t const &first, substrings_match_t const &second) noexcept {
        return first.haystack_index == second.haystack_index && first.needle_index == second.needle_index &&
               first.byte_offset == second.byte_offset && first.byte_length == second.byte_length;
    }
    friend bool operator!=(substrings_match_t const &first, substrings_match_t const &second) noexcept {
        return !(first == second);
    }
};

/**
 *  @brief  One start position's incumbent match while a leftmost policy is still deciding it.
 *
 *  Separate from `substrings_output` because the units differ: an output carries the folded length the
 *  automaton walked, a pending start carries the @b source span that length resolved to. Sharing one struct
 *  would let a folded length be compared against a source one, which the two leftmost policies would then
 *  silently resolve wrong.
 */
struct substrings_pending_start {
    u32_t needle_index {};
    /** @brief Haystack bytes this match spans; zero means no match has claimed that start yet. */
    u32_t source_match_bytes {};
};

/**
 *  @brief  Whether @p challenger outranks @p incumbent among matches sharing one start position.
 *  @param[in] incumbent A zero `source_match_bytes` means no match has claimed that start yet.
 *
 *  The only place either leftmost policy is consulted, and it runs once per discovered match rather than
 *  once per byte, so the policy stays an ordinary argument. Lengths compared here are source bytes, since
 *  a needle that folds shorter can still outspan a rival whose folded form is longer.
 */
constexpr bool substrings_leftmost_wins(          //
    substrings_pending_start const &challenger,   //
    substrings_pending_start const &incumbent,    //
    substrings_overlap_policy_t policy) noexcept {

    if (incumbent.source_match_bytes == 0) return true;
    if (policy == substrings_leftmost_longest_k && challenger.source_match_bytes != incumbent.source_match_bytes)
        return challenger.source_match_bytes > incumbent.source_match_bytes;
    return challenger.needle_index < incumbent.needle_index;
}

/**
 *  @brief How many starts a leftmost walk can hold undecided, given the longest match in the dictionary.
 *  @param[in] max_source_match_bytes The longest match in @b source bytes, which is what the slots are keyed
 *             by; passing the folded bound instead would undersize the ring wherever a fold contracts.
 *
 *  A start can no longer be outbid once the walk is that many bytes past it, so that many slots suffice.
 *  Rounded up to a power of two, which turns the walk's slot lookup into a mask instead of a division by a
 *  value only known at runtime.
 */
inline size_t substrings_pending_starts_width(size_t max_source_match_bytes) noexcept {
    return sz_size_bit_ceil(sz_max_of_two(max_source_match_bytes, (size_t)1));
}

/** @brief BM25's continuous parameters. */
struct substrings_bm25_t {
    /** @brief The literature's `k1`: how slowly repeated occurrences stop adding score. */
    f32_t term_frequency_saturation = 1.2f;
    /** @brief The literature's `b`, in [0, 1]: 0 ignores document length, 1 normalizes fully. */
    f32_t length_normalization = 0.75f;
    /** @brief Corpus-wide mean document length; zero disables length normalization. */
    f32_t average_document_length = 0.0f;
};

/** @brief One needle's saturated contribution, before its weight. */
constexpr f32_t substrings_bm25_term(substrings_bm25_t const &parameters, f32_t term_frequency,
                                     f32_t document_length) noexcept {
    // A zero mean length has no normalizer to divide by, so the length term collapses to one.
    f32_t const normalized_length = parameters.average_document_length > 0.0f
                                        ? 1.0f - parameters.length_normalization +
                                              parameters.length_normalization * document_length /
                                                  parameters.average_document_length
                                        : 1.0f;
    return term_frequency * (parameters.term_frequency_saturation + 1.0f) /
           (term_frequency + parameters.term_frequency_saturation * normalized_length);
}

#pragma endregion Vocabulary

#pragma region Published View

/** @brief Number of columns in a hot-tier row, one per possible input byte. */
static constexpr size_t substrings_alphabet_size_k = 256;

/**
 *  @brief One goto-completed row of @p rows, whichever memory space holds them.
 *  @tparam index_type_ `size_t` for the global tier; `small_size_t` for a shared-memory staged prefix, whose
 *          bound makes the narrower multiply safe. Both operands convert first, so the alphabet constant
 *          cannot widen the product back.
 */
template <typename index_type_ = size_t, typename state_id_type_>
constexpr span<state_id_type_ const, substrings_alphabet_size_k> hot_row_of( //
    span<state_id_type_ const> rows, state_id_type_ state) noexcept {
    static_assert(sizeof(index_type_) >= sizeof(state_id_type_),
                  "The row index must be at least as wide as the state id, so this conversion never narrows");
    index_type_ const first_cell = static_cast<index_type_>(state) *
                                   static_cast<index_type_>(substrings_alphabet_size_k);
    sz_assert_(static_cast<size_t>(first_cell) + substrings_alphabet_size_k <= rows.size());
    return {rows.data() + first_cell};
}

/**
 *  @brief  Immutable, trivially-copyable view of a built dictionary, safe to pass to a CUDA kernel by value.
 *
 *  Owns nothing. Every pointer refers to storage held by the `aho_corasick_dictionary` that produced the
 *  view, which must outlive it. A device-side view points at device memory holding the same layout.
 *
 *  Transitions are split into two tiers by how often a state is visited. Text keeps resetting the walk toward
 *  the root, so a small set of states absorbs most byte steps whatever the dictionary size, and the tiers are
 *  sized to that skew rather than to the automaton as a whole.
 *
 *  The @b hot tier is a dense goto-completed table, one row of 256 targets per state. A step is a single load
 *  with no branch and no failure chasing. The @b cold tier is a double array: `base` and `check` encode
 *  transitions as address arithmetic plus an ownership test, and `fail` restores the failure links that
 *  goto-completion would otherwise have folded away. Completing the cold tier the same way would require
 *  every one of a state's 256 slots to be free, which is a dense row again, so its failure links stay live.
 *
 *  States are numbered so the hot ones come first, making the tier test `state < hot_count` with no lookup.
 */
template <typename state_id_type_>
struct aho_corasick_view {
    using state_id_t = state_id_type_;
    using output_t = substrings_output<state_id_t>;

    /** @brief Hot tier: `hot_count * 256` goto-completed targets, row-major, shallow states first and each
     *         depth band ordered by out-degree - a build-time proxy for how often text visits a state. */
    state_id_t const *hot_rows {};

    /** @brief Cold tier: transition target for `state` on `byte` is `base[state] + byte`, if owned. */
    state_id_t const *base {};
    /** @brief Cold tier: owner of each slot, so a collision reads as a missing edge rather than a wrong one. */
    state_id_t const *check {};
    /** @brief Cold tier: failure link, followed when `check` denies ownership. */
    state_id_t const *fail {};

    /**
     *  @brief Matches ending at each state, flattened; already merged along failure chains at build time.
     *
     *  Offsets are `size_t` rather than a narrower width because a state's outputs are its own plus its
     *  failure state's whole run, so a nested-suffix dictionary drives the pool to O(states squared).
     */
    output_t const *outputs {};
    state_id_t const *outputs_counts {};
    size_t const *outputs_offsets {};
    /** @brief Length of `outputs`, so a consumer never has to rescan the CSR to recover it. */
    size_t outputs_total {};

    /** @brief States `[0, hot_count)` live in `hot_rows`; the rest live in the double array. */
    state_id_t hot_count {};
    state_id_t state_count {};
    state_id_t root {};

    /** @brief Longest needle, in the folded bytes the automaton walks. */
    state_id_t max_folded_match_bytes {};

    /** @brief Shortest needle, in folded bytes; bounds how densely a rewrite can fire. */
    state_id_t min_folded_match_bytes {};

    /**
     *  @brief Most @b haystack bytes one match can span, which is what every slice, halo and warm-up needs.
     *
     *  A fold can contract three source bytes into one - the Kelvin sign into `k` - so a folded length says
     *  nothing directly about how far back into the haystack a match reaches. Sizing a window from
     *  `max_folded_match_bytes` instead would silently drop matches straddling a boundary.
     */
    state_id_t max_source_match_bytes {};

    /** @brief Fewest haystack bytes one match can span; the mirror bound. */
    state_id_t min_source_match_bytes {};

    /** @brief Whether a walk folds the haystack as it consumes it, or steps it byte for byte. */
    substrings_case_sensitivity_t case_sensitivity {substrings_cased_k};

    /**
     *  @brief Most merged outputs any single state carries, so a consumer can bound one pass's match count:
     *         `n` bytes report at most `n * max_outputs_per_state` matches. A nested-suffix vocabulary puts
     *         every shorter needle on the deepest state's run, and that is the worst case.
     */
    state_id_t max_outputs_per_state {};

    /** @brief Whether the cold tier is empty, so every step takes the branch-free hot path. */
    constexpr bool all_hot() const noexcept { return state_count <= hot_count; }

    /** @brief The whole hot tier as one span, so a row lookup can bounds-check itself. */
    constexpr span<state_id_t const> all_hot_rows() const noexcept {
        return {hot_rows, hot_count * substrings_alphabet_size_k};
    }

    /** @brief One goto-completed row, whose width is the alphabet and therefore known at compile time. */
    constexpr span<state_id_t const, substrings_alphabet_size_k> hot_row(state_id_t state) const noexcept {
        return hot_row_of(all_hot_rows(), state);
    }
};

#pragma endregion Published View

#pragma region Transition

/**
 *  @brief  Advances one state by one byte. The single definition shared by every CPU and GPU kernel.
 *
 *  Hot states resolve in one load. Cold states probe the double array and, when the slot is owned by
 *  somebody else, hop to the failure link and retry the same byte. The root is total - every one of its
 *  slots resolves, self-looping where the trie has no edge - which is what terminates the retry loop.
 *
 *  `constexpr` rather than host-device annotated, so CUDA kernels reach it through
 *  @b `--expt-relaxed-constexpr`, which every build path already passes.
 */
template <typename state_id_type_>
constexpr state_id_type_ aho_corasick_step( //
    aho_corasick_view<state_id_type_> const &view, state_id_type_ state, u8_t byte) noexcept {

    for (;;) {
        if (state < view.hot_count) return view.hot_row(state)[byte];
        // The probe index stays wide: `base + byte` can exceed the id ceiling on a slot this state does not
        // own, and narrowing first would wrap onto a slot `check` might accept. The arrays carry
        // `alphabet_size_k - 1` slots of headroom past the last state for exactly this reach.
        size_t const candidate = (size_t)view.base[state] + byte;
        if (view.check[candidate] == state) return (state_id_type_)candidate;
        if (state == view.root) return view.root;
        state = view.fail[state];
    }
}

/**
 *  @brief  Advances @p state by one byte and reports how many needles end on the new state.
 *
 *  The pair every walk repeats: the transition, then the output count that decides whether the walk stops
 *  to enumerate matches. Sharing it keeps `find`, `count`, and the per-core counters reading one array.
 */
template <typename state_id_type_>
constexpr state_id_type_ aho_corasick_step_counting( //
    aho_corasick_view<state_id_type_> const &view, state_id_type_ &state, u8_t byte) noexcept {

    state = aho_corasick_step(view, state, byte);
    return view.outputs_counts[state];
}

#pragma endregion Transition

#pragma region Folding Filter

/** @brief Most folded bytes one source codepoint can produce: three runes of three bytes each. */
static constexpr size_t substrings_folded_image_max_k = 9;

/** @brief One folded byte, and everything the walk needs about the codepoint it came from. */
struct substrings_folded_byte_t {
    u8_t byte {};
    /** @brief Whether this byte ends a folded rune; a needle is valid UTF-8, so only there can a match end. */
    bool rune_end {};
    /** @brief Whether this codepoint's fold leaves a folded byte at an offset no source byte owns. */
    bool breaks_boundary {};
    /** @brief Whether the source byte began no well-formed codepoint, so the walk must resynchronize. */
    bool malformed {};
    /** @brief Offset just past the source codepoint; every folded byte of it reports the same end. */
    size_t codepoint_end {};
    /** @brief Folded bytes of this codepoint still to come, which a backward walk has to step over first. */
    u8_t trailing {};
    /** @brief Folded bytes back to this codepoint's previous rune end, zero at its first. */
    u8_t shift {};
};

/**
 *  @brief  Streams a haystack as folded bytes, one source codepoint at a time and never into a buffer.
 *
 *  The automaton's alphabet is bytes while `sz_utf8_folded_iter_t` yields runes, so one codepoint's runes
 *  are drained and re-encoded into a nine-byte image, then handed out a byte at a time. The ASCII fast path,
 *  the expansion buffering and the one-byte malformed resynchronization all remain the iterator's.
 */
struct substrings_folded_cursor_t {
    sz_utf8_folded_iter_t runes {};
    cptr_t origin {};
    u8_t image[substrings_folded_image_max_k] {};
    /** @brief Bit `index` marks the byte at `index` as ending a folded rune. */
    u16_t rune_end_mask {};
    u8_t image_length {};
    u8_t image_index {};
    u8_t previous_rune_end {};
    size_t codepoint_end {};
    bool breaks_boundary {};
    bool malformed {};
};

SZ_HELPER_AUTO void substrings_folded_cursor_init(substrings_folded_cursor_t &cursor,
                                                  span<byte_t const> haystack) noexcept {
    sz_utf8_folded_iter_init_(&cursor.runes, (cptr_t)haystack.data(), haystack.size());
    cursor.origin = (cptr_t)haystack.data();
    cursor.image_length = 0;
    cursor.image_index = 0;
}

/** @brief Next folded byte, or false once the haystack is spent. */
SZ_HELPER_AUTO bool substrings_folded_cursor_next(substrings_folded_cursor_t &cursor,
                                                  substrings_folded_byte_t &folded) noexcept {

    if (cursor.image_index == cursor.image_length) {
        // ASCII is its own codepoint and folds with one add, so it never decodes and never consults a table.
        if (cursor.runes.ptr < cursor.runes.end && (sz_u8_t)*cursor.runes.ptr < 0x80) {
            cursor.image[0] = (u8_t)sz_ascii_fold_((sz_u8_t)*cursor.runes.ptr);
            cursor.image_length = 1;
            cursor.rune_end_mask = 1;
            cursor.runes.codepoint_begin = cursor.runes.ptr;
            cursor.runes.codepoint_length = 1;
            ++cursor.runes.ptr;
            cursor.breaks_boundary = false;
            cursor.malformed = false;
        }
        else {
            // Nothing under this lead byte folds, so the codepoint's own bytes already are its folded image:
            // no rune is assembled, no fold ladder walked, no re-encode. This is every CJK, Arabic, Hebrew,
            // Devanagari and emoji sequence, and it is what holds them at byte-exact speed.
            rune_t rune;
            rune_length_t verbatim = sz_rune_invalid_k;
            if (cursor.runes.ptr < cursor.runes.end && !sz_utf8_lead_may_fold_((sz_u8_t)*cursor.runes.ptr))
                verbatim = sz_rune_decode(cursor.runes.ptr, cursor.runes.end, &rune);

            if (verbatim != sz_rune_invalid_k) {
                for (size_t index = 0; index < (size_t)verbatim; ++index)
                    cursor.image[index] = (u8_t)cursor.runes.ptr[index];
                cursor.image_length = (u8_t)verbatim;
                cursor.rune_end_mask = (u16_t)(1u << (verbatim - 1));
                cursor.runes.codepoint_begin = cursor.runes.ptr;
                cursor.runes.codepoint_length = (size_t)verbatim;
                cursor.runes.ptr += verbatim;
                cursor.malformed = false;
                cursor.breaks_boundary = false;
            }
            else {
                if (!sz_utf8_folded_iter_next_(&cursor.runes, &rune)) return false;

                // A malformed byte arrives tagged, outside the scalar range `sz_rune_encode` accepts, so it
                // passes through as the literal byte it was.
                cursor.malformed = (rune & 0x80000000u) != 0;
                cursor.image_length = 0;
                cursor.rune_end_mask = 0;
                size_t rune_count = 0;
                for (;;) {
                    if (cursor.malformed) cursor.image[cursor.image_length++] = (u8_t)(rune & 0xFF);
                    else cursor.image_length += (u8_t)sz_rune_encode(rune, cursor.image + cursor.image_length);
                    cursor.rune_end_mask |= (u16_t)(1u << (cursor.image_length - 1));
                    ++rune_count;
                    // More pending runes belong to this codepoint; the iterator refills only once they are spent.
                    if (cursor.runes.pending_idx >= cursor.runes.pending_count) break;
                    sz_utf8_folded_iter_next_(&cursor.runes, &rune);
                }
                cursor.breaks_boundary = rune_count != 1 ||
                                         (size_t)cursor.image_length != cursor.runes.codepoint_length;
            }
        }

        cursor.codepoint_end = (size_t)(cursor.runes.codepoint_begin - cursor.origin) + cursor.runes.codepoint_length;
        cursor.image_index = 0;
        cursor.previous_rune_end = 0;
    }

    u8_t const index = cursor.image_index++;
    folded.byte = cursor.image[index];
    folded.rune_end = (cursor.rune_end_mask >> index) & 1u;
    folded.breaks_boundary = cursor.breaks_boundary;
    folded.malformed = cursor.malformed;
    folded.codepoint_end = cursor.codepoint_end;
    folded.trailing = (u8_t)(cursor.image_length - cursor.image_index);
    folded.shift = (u8_t)(cursor.image_index - cursor.previous_rune_end);
    if (folded.rune_end) {
        // A codepoint's first rune end has nothing before it to repeat, which a zero shift names.
        if (cursor.previous_rune_end == 0) folded.shift = 0;
        cursor.previous_rune_end = cursor.image_index;
    }
    return true;
}

/**
 *  @brief  Where a match resolved by walking backwards from the codepoint it ends in landed.
 *  @see    `substrings_folded_span` for the cheap path that skips the walk entirely.
 *
 *  `repeats` marks a span an earlier rune end of that same codepoint already reported: needle "s" ends at
 *  both runes the sharp S folds to, and both spans are the whole codepoint. Equal spans are what makes a
 *  repeat, not equal content - `"ss"` ends at both runes of the second sharp S in `"ßß"` too, and those are
 *  two genuinely different spans.
 */
struct substrings_resolved_match_t {
    size_t source_offset {};
    bool repeats {};
};

/**
 *  @brief  Resolves a match's source start, and whether it repeats an earlier rune end's span, in one pass.
 *  @param[in] source_end End of the codepoint the match ends in; every rune end of it shares this.
 *  @param[in] trailing Folded bytes of that codepoint sitting past the match's end.
 *  @param[in] shift Folded bytes back to the previous rune end of the same codepoint, zero at the first.
 *
 *  Reached only when the match starts at or before the last boundary-breaking codepoint, which is the only
 *  place either question is open: a boundary-preserving match starts on a lead byte, so distinct starts sit
 *  in distinct codepoints and no span can repeat. Two windows one length apart hold the same bytes exactly
 *  when the folded stream is periodic with that period, so the repeat test rides a `shift`-sized ring that
 *  one codepoint's image bounds, and shares the single backward walk with the start it recovers.
 */
SZ_HELPER_AUTO substrings_resolved_match_t substrings_resolve_match(span<byte_t const> haystack, size_t source_end,
                                                                    size_t trailing, size_t folded_match_bytes,
                                                                    size_t shift) noexcept {

    sz_utf8_folded_reverse_iter_t iterator;
    sz_utf8_folded_reverse_iter_init_(&iterator, (cptr_t)haystack.data(), (cptr_t)(haystack.data() + source_end));

    substrings_resolved_match_t resolved;
    resolved.source_offset = source_end;
    resolved.repeats = false;

    u8_t pending[4];
    size_t pending_count = 0;
    cptr_t codepoint_begin = (cptr_t)(haystack.data() + source_end);
    u8_t ring[substrings_folded_image_max_k] = {};
    cptr_t start_here = nullptr, start_earlier = nullptr;
    size_t const wanted = shift != 0 ? folded_match_bytes + shift : folded_match_bytes;
    bool periodic = shift != 0;

    for (size_t stepped = 0; stepped < trailing + wanted; ++stepped) {
        if (pending_count == 0) {
            rune_t image;
            if (!sz_utf8_folded_reverse_iter_prev_(&iterator, &image)) break;
            // A malformed byte arrives tagged above the encodable range, so it passes through as itself.
            if (image & 0x80000000u) pending[0] = (u8_t)(image & 0xFF), pending_count = 1;
            else pending_count = (size_t)sz_rune_encode(image, pending);
            // The cursor sits on the codepoint's own start as soon as any of its runes is yielded, which is
            // the outward snap a match beginning mid-expansion needs.
            codepoint_begin = iterator.ptr;
        }
        u8_t const byte = pending[--pending_count];
        if (stepped < trailing) continue; // ? Folded bytes of the ending codepoint past the match's own end

        size_t const taken = stepped - trailing + 1;
        if (shift != 0) {
            if (taken > shift && ring[(taken - shift) % substrings_folded_image_max_k] != byte) periodic = false;
            ring[taken % substrings_folded_image_max_k] = byte;
        }
        if (taken == folded_match_bytes) start_here = codepoint_begin;
        if (taken == wanted) start_earlier = codepoint_begin;
        if (!periodic && taken >= folded_match_bytes) break;
    }

    if (start_here != nullptr) resolved.source_offset = (size_t)(start_here - (cptr_t)haystack.data());
    resolved.repeats = periodic && start_here != nullptr && start_here == start_earlier;
    return resolved;
}

/**
 *  @brief  Resolves the source span of one output at the rune end @p step stands on.
 *  @param[in] folded Folded bytes consumed so far, the ending offset the output's length is taken back from.
 *  @param[in] last_break_folded_end Folded end of the last codepoint whose fold broke a boundary.
 *
 *  A match starting at or after the last break lies where folded and source offsets still agree, so its start
 *  is one subtraction; anything earlier pays the backward walk. Every walk resolves matches this way, so the
 *  cheap test and the expensive fallback stay one decision rather than four copies of one.
 */
SZ_HELPER_AUTO substrings_resolved_match_t substrings_folded_span(span<byte_t const> haystack,
                                                                  substrings_folded_byte_t const &step, size_t folded,
                                                                  size_t last_break_folded_end,
                                                                  size_t folded_match_bytes) noexcept {
    if (folded - folded_match_bytes >= last_break_folded_end) return {step.codepoint_end - folded_match_bytes, false};
    return substrings_resolve_match(haystack, step.codepoint_end, step.trailing, folded_match_bytes, step.shift);
}

#pragma endregion Folding Filter

#pragma region Engine

/**
 *  @brief  Multi-pattern search engine: one compiled dictionary applied to many haystacks in a single pass.
 *
 *  Declared without a body so every backend supplies its own specialization, guarded on @p capability_ -
 *  the shape `levenshtein_distances` and the other StringZillas engines already use. The state-id width is
 *  not a parameter here: it follows from the needle set, so `try_build` settles it and stores whichever
 *  automaton won, the way the similarity engines choose a cell width from the inputs.
 */
template <typename allocator_type_ = dummy_alloc_t, sz_capability_t capability_ = sz_cap_serial_k,
          typename enable_ = void>
struct substrings;

/** @brief Which state-id width a built automaton settled on. @sa `substrings::state_width`. */
enum class substrings_state_width_t : bool { u16_k, u32_k };

#pragma endregion Engine

#pragma region Dictionary

/**
 *  @brief Two-tier @b byte-level Aho-Corasick dictionary for multi-pattern exact and case-folded substring
 *         search: a dense goto-completed hot tier plus a double-array cold tier.
 *  @tparam state_id_type_ The type of the state ID. Default is `u32_t`; `u16_t` fits dictionaries under
 *          65534 states into half the row width.
 *  @tparam allocator_type_ The type of the allocator. Default is `dummy_alloc_t`.
 *
 *  Holds no STL container and throws nothing, reporting through `status_t` and `try_`-prefixed functions.
 *  Construction allocates a constant number of flat buffers rather than one per state, none of which
 *  survives `try_build`, and no phase ever materializes a dense row per state.
 *
 *  `hot_rows_` is a plain dense 256-wide goto-completed table for the `hot_count_` most-visited states -
 *  raw state IDs, no byte-class compression and no premultiplication, so a lookup is
 *  `hot_rows_[state * 256 + byte]`. Every other state lives in the double array `base_` / `check_` /
 *  `fail_`, exactly as `aho_corasick_view` publishes it.
 */
template <typename state_id_type_ = u32_t, typename allocator_type_ = dummy_alloc_t>
struct aho_corasick_dictionary {

    using state_id_t = state_id_type_;
    using allocator_t = allocator_type_;
    using output_t = substrings_output<state_id_t>;
    using pending_start_t = substrings_pending_start;
    static_assert(std::is_unsigned<state_id_t>::value, "State ID should be unsigned");

    static constexpr size_t alphabet_size_k = 256;
    /**
     *  @brief The one ceiling of the automaton, shared by everything stored in `state_id_t` cells - states,
     *         needles, edges, slot capacity, match byte-lengths, and merged output runs. Each is checked for
     *         `overflow_risk_k` at its own growth site, so a `u16` dictionary caps needles at 65535 bytes.
     */
    static constexpr state_id_t invalid_state_k = std::numeric_limits<state_id_t>::max();
    /** @brief `hot_count_`'s "not chosen yet" state, so `hot_count(0)` stays a real all-cold request. */
    static constexpr size_t derive_hot_count_k = std::numeric_limits<size_t>::max();

    /**
     *  @brief Narrows @p value to a state id, asserting in debug that the pool ceiling still holds.
     *
     *  Only for values whose ceiling `allocate_raw_state_` already enforced at insertion time. @b Not for
     *  caller-controlled sizes - those report `overflow_risk_k` in release builds too.
     */
    static constexpr state_id_t state_id_of_(size_t value) noexcept {
        sz_assert_(value <= static_cast<size_t>(invalid_state_k) && "State id does not fit state_id_t");
        return static_cast<state_id_t>(value);
    }

    /** @brief One literal trie edge, appended as it is created and counting-sorted by `parent` at build. */
    struct trie_edge_t {
        state_id_t parent;
        state_id_t child;
        u8_t byte;
    };

    /** @brief One trie edge again, narrowed for the build-time CSR where `parent` is the row index. */
    struct csr_edge_t {
        state_id_t child;
        u8_t byte;
    };

    /** @brief One pending match at a raw state, threaded into that state's own list by `output_run_t`. */
    struct pending_output_t {
        state_id_t needle_index;
        state_id_t folded_match_bytes;
        size_t next;
    };

    /** @brief Per spelling state: the matches ending on it, shared by every walking state that spells it. */
    struct output_run_t {
        /** @brief Head of this state's own match list, most recent first. */
        size_t own_head = SZ_SIZE_MAX;
        /** @brief Matches ending exactly on this state, before failure-chain inheritance. */
        state_id_t own_count = 0;
    };

    /**
     *  @brief Per trie state: its failure link, its published slot, and its merged output run.
     *
     *  One entry per raw state, addressed by that state's own id. Insertion builds a tree, so a state has
     *  exactly one spelling and therefore exactly one failure link.
     */
    struct trie_state_t {
        /** @brief The failure state, always strictly shallower, so depth order finishes it first. */
        state_id_t failure_state = 0;
        /** @brief Published double-array slot for this state. */
        state_id_t published_id = invalid_state_k;
        /** @brief Into `outputs_`: own matches followed by the failure state's whole run. */
        size_t total_offset = 0;
        state_id_t total_count = 0;
    };

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using state_id_allocator_t = typename allocator_traits_t::template rebind_alloc<state_id_t>;
    using output_allocator_t = typename allocator_traits_t::template rebind_alloc<output_t>;
    using edge_allocator_t = typename allocator_traits_t::template rebind_alloc<trie_edge_t>;
    using pending_output_allocator_t = typename allocator_traits_t::template rebind_alloc<pending_output_t>;
    using output_run_allocator_t = typename allocator_traits_t::template rebind_alloc<output_run_t>;
    using trie_state_allocator_t = typename allocator_traits_t::template rebind_alloc<trie_state_t>;
    using csr_edge_allocator_t = typename allocator_traits_t::template rebind_alloc<csr_edge_t>;
    using word_allocator_t = typename allocator_traits_t::template rebind_alloc<u64_t>;
    using byte_allocator_t = typename allocator_traits_t::template rebind_alloc<std::byte>;
    /** @brief Rebinds to `size_t`, for the edge index, `outputs_counts`, and `outputs_offsets` alike. */
    using offset_allocator_t = typename allocator_traits_t::template rebind_alloc<size_t>;

    /** @brief Every literal trie edge, in creation order; `compact_edges_into_csr_` consumes it. */
    safe_vector<trie_edge_t, edge_allocator_t> edges_;
    /** @brief Open-addressed indices into `edges_`, giving insertion and the failure chase an O(1) lookup
     *         of `(parent, byte)` without storing a key of its own. Released alongside `edges_`. */
    safe_vector<size_t, offset_allocator_t> edge_index_;
    /** @brief Matches ending at each raw state, before failure-chain inheritance merges them. */
    safe_vector<pending_output_t, pending_output_allocator_t> own_outputs_;
    /** @brief One entry per raw state; grows with the state pool and survives into the output pass. */
    safe_vector<output_run_t, output_run_allocator_t> output_runs_;
    /** @brief The needle under construction, folded once into canonical bytes; uncased mode only. */
    safe_vector<byte_t, byte_allocator_t> folded_needle_;

    /** @brief One block carved by `build_layout_` into the buffers whose size is fixed once insertion ends. */
    safe_vector<std::byte, byte_allocator_t> build_scratch_;

    /** @brief One entry per raw trie state, addressed by that state's own id. */
    safe_vector<trie_state_t, trie_state_allocator_t> trie_states_;
    /** @brief Trie states in depth-band, out-degree-descending order: the input to the published numbering.
     *         Insertion order is not depth order, so this permutation - not the id itself - is what a band
     *         is a contiguous range of. */
    safe_vector<state_id_t, state_id_allocator_t> trie_order_;
    /** @brief Scratch the band sort permutes through, since a band's states are not a contiguous id range. */
    safe_vector<state_id_t, state_id_allocator_t> trie_order_scratch_;
    /** @brief The root's goto-completed row, dense over the alphabet, so a failure chase ends in one lookup
     *         rather than a scan of the root's whole edge list. */
    safe_vector<state_id_t, state_id_allocator_t> trie_root_row_;

    /** @brief Published slot -> walking state; grows with the double array, so it lives outside the layout. */
    safe_vector<state_id_t, state_id_allocator_t> old_of_final_;
    /** @brief One bit per double-array slot; the only record of what the packing search has claimed. */
    safe_vector<u64_t, word_allocator_t> occupied_bits_;
    /** @brief Lowest slot that could still be free; packing only fills forward, so it never moves back. */
    size_t lowest_free_cursor_ = 0;

    /** @brief Hot tier: `hot_count_ * alphabet_size_k` goto-completed targets, row-major, shallow states
     *         first and each depth band ordered by out-degree descending. */
    safe_vector<state_id_t, state_id_allocator_t> hot_rows_;
    /** @brief Cold tier: transition target for `state` on `byte` is `base_[state] + byte`, if `check_` confirms
     *         ownership. Sized `count_states_ + (alphabet_size_k - 1)`, since a child's ID is address
     *         arithmetic and can exceed the real state count; only entries `>= hot_count_` are meaningful. */
    safe_vector<state_id_t, state_id_allocator_t> base_;
    /** @brief Cold tier: owner of each double-array slot; a mismatch means "no such edge", not "wrong edge".
     *         Same length as `base_`. */
    safe_vector<state_id_t, state_id_allocator_t> check_;
    /** @brief Cold tier: failure link, followed when `check_` denies ownership. Same length as `base_`. */
    safe_vector<state_id_t, state_id_allocator_t> fail_;
    /** @brief CSR-flattened match outputs, addressed through `outputs_offsets_` / `outputs_counts_`. */
    safe_vector<output_t, output_allocator_t> outputs_;
    /** @brief Number of outputs per state, in the published band-ordered numbering, same length as `base_`. */
    safe_vector<state_id_t, state_id_allocator_t> outputs_counts_;
    /** @brief Exclusive prefix sum of `outputs_counts_`, same length as `base_`. */
    safe_vector<size_t, offset_allocator_t> outputs_offsets_;

    size_t count_states_ = 0;
    size_t count_needles_ = 0;
    state_id_t max_folded_match_bytes_ = 0;
    state_id_t min_folded_match_bytes_ = 0;
    /** @brief Worst-case haystack span of one match; a fold contracting 3 bytes into 1 is what widens it. */
    state_id_t max_source_match_bytes_ = 0;
    state_id_t min_source_match_bytes_ = 0;
    state_id_t max_outputs_per_state_ = 0;
    substrings_case_sensitivity_t case_sensitivity_ = substrings_cased_k;

    /** @brief States `[0, hot_count_)` live in `hot_rows_`; `derive_hot_count_k` asks `try_build` to size
     *         the tier from `cpu_specs_t` instead, which `hot_count` overrides with any explicit value. */
    size_t hot_count_ = derive_hot_count_k;
    /** @brief The root's ID in the published numbering; always `0`, since the root is the unique
     *         shallowest state and therefore always sorts first. */
    state_id_t root_ = 0;

    allocator_t alloc_;

#pragma region Construction Helpers

    /** @brief Scrambles a `(parent, byte)` pair into a starting probe; the index stores no key of its own. */
    static size_t probe_of_(state_id_t parent, u8_t byte) noexcept {
        u64_t const mixed = ((((u64_t)parent << 8) | byte) + 1) * 0x9E3779B97F4A7C15ull;
        return mixed >> 32;
    }

    /** @brief Rebuilds `edge_index_` at @p new_capacity slots, reinserting every edge already in `edges_`. */
    status_t rehash_edge_index_(size_t new_capacity) noexcept {
        if (edge_index_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t slot = 0; slot < new_capacity; ++slot) edge_index_[slot] = SZ_SIZE_MAX;
        size_t const mask = new_capacity - 1;
        for (size_t edge = 0; edge < edges_.size(); ++edge) {
            trie_edge_t const &record = edges_[edge];
            size_t slot = probe_of_(record.parent, record.byte) & mask;
            while (edge_index_[slot] != SZ_SIZE_MAX) slot = (slot + 1) & mask;
            edge_index_[slot] = edge;
        }
        return status_t::success_k;
    }

    /** @brief Index into `edges_` of the `(parent, byte)` edge, or the invalid sentinel when absent. */
    size_t find_edge_(state_id_t parent, u8_t byte) const noexcept {
        size_t const mask = edge_index_.size() - 1;
        for (size_t slot = probe_of_(parent, byte) & mask;; slot = (slot + 1) & mask) {
            size_t const edge = edge_index_[slot];
            if (edge == SZ_SIZE_MAX) return edge;
            trie_edge_t const &record = edges_[edge];
            if (record.parent == parent && record.byte == byte) return edge;
        }
    }

    /** @brief Records a `(parent, byte) -> child` edge that `find_edge_` has just reported missing. */
    status_t add_edge_(state_id_t parent, u8_t byte, state_id_t child) noexcept {
        // Keep the load factor under one half, so linear probing stays a couple of slots deep.
        if ((edges_.size() + 1) * 2 >= edge_index_.size())
            if (rehash_edge_index_(sz_size_bit_ceil(edge_index_.size() + 1) * 2) != status_t::success_k)
                return status_t::bad_alloc_k;
        if (edges_.try_push_back(trie_edge_t {parent, child, byte}) != status_t::success_k)
            return status_t::bad_alloc_k;

        size_t const mask = edge_index_.size() - 1;
        size_t slot = probe_of_(parent, byte) & mask;
        while (edge_index_[slot] != SZ_SIZE_MAX) slot = (slot + 1) & mask;
        edge_index_[slot] = (edges_.size() - 1);
        return status_t::success_k;
    }

    status_t allocate_raw_state_(state_id_t &new_state) noexcept {
        if (count_states_ >= (size_t)invalid_state_k) return status_t::overflow_risk_k;
        // `try_reserve` allocates exactly what is asked, so a bare `try_resize(count + 1)` per state would
        // re-move the whole array on every allocation - quadratic in the state count. Reserving in
        // power-of-two steps keeps the growth amortized-linear; the reserve is a no-op once capacity holds.
        if (output_runs_.try_reserve(sz_size_bit_ceil(count_states_ + 1)) != status_t::success_k)
            return status_t::bad_alloc_k;
        if (output_runs_.try_resize(count_states_ + 1) != status_t::success_k) return status_t::bad_alloc_k;
        new_state = static_cast<state_id_t>(count_states_);
        ++count_states_;
        return status_t::success_k;
    }

    status_t ensure_root_() noexcept {
        if (edge_index_.size() == 0 && rehash_edge_index_(1024) != status_t::success_k) return status_t::bad_alloc_k;
        if (count_states_ != 0) return status_t::success_k;
        state_id_t root;
        status_t const status = allocate_raw_state_(root);
        sz_assert_(status != status_t::success_k || root == 0);
        return status;
    }

    /**
     *  @brief Appends a match ending at @p state, onto that state's own list.
     *
     *  A match traverses one edge per byte along a trie path whose positions strictly increase, so it can
     *  never span more bytes than the automaton has states - which is why the length rides `state_id_t`.
     */
    status_t add_output_(state_id_t state, state_id_t needle_index, size_t folded_match_bytes) noexcept {
        // Caller-controlled length, so the ceiling is a real status in every build rather than an assert.
        if (folded_match_bytes > static_cast<size_t>(invalid_state_k)) return status_t::overflow_risk_k;
        output_run_t &run = output_runs_[state];
        state_id_t const folded_narrow = static_cast<state_id_t>(folded_match_bytes);
        if (own_outputs_.try_push_back(pending_output_t {needle_index, folded_narrow, run.own_head}) !=
            status_t::success_k)
            return status_t::bad_alloc_k;
        run.own_head = (own_outputs_.size() - 1);
        ++run.own_count;
        max_folded_match_bytes_ = sz_max_of_two(max_folded_match_bytes_, folded_narrow);
        min_folded_match_bytes_ = min_folded_match_bytes_ ? sz_min_of_two(min_folded_match_bytes_, folded_narrow)
                                                          : folded_narrow;

        // One folded byte can stand for up to `sz_utf8_fold_max_contraction_k` source bytes, and one source
        // byte for up to `sz_utf8_fold_max_expansion_k` folded ones, so a folded length brackets rather than
        // fixes the source span. Cased needles fold to themselves, so their bounds stay exact.
        size_t const contraction = case_sensitivity_ == substrings_uncased_k ? (size_t)sz_utf8_fold_max_contraction_k
                                                                             : (size_t)1;
        size_t const expansion = case_sensitivity_ == substrings_uncased_k ? (size_t)sz_utf8_fold_max_expansion_k
                                                                           : (size_t)1;
        size_t const source_ceiling = folded_match_bytes * contraction;
        size_t const source_floor = (folded_match_bytes + expansion - 1) / expansion;
        if (source_ceiling > static_cast<size_t>(invalid_state_k)) return status_t::overflow_risk_k;
        max_source_match_bytes_ = sz_max_of_two(max_source_match_bytes_, static_cast<state_id_t>(source_ceiling));
        min_source_match_bytes_ = min_source_match_bytes_
                                      ? sz_min_of_two(min_source_match_bytes_, static_cast<state_id_t>(source_floor))
                                      : static_cast<state_id_t>(source_floor);
        return status_t::success_k;
    }

    /** @brief Canonical UTF-8 length of @p rune, as the encoder itself reports it. */
    static rune_length_t utf8_length_of_rune_(rune_t rune) noexcept {
        u8_t scratch[4];
        return sz_rune_encode(rune, scratch);
    }

    /** @brief Follows @p parent's @p byte edge, minting a state and wiring the edge when it is missing. */
    status_t follow_or_create_(state_id_t parent, u8_t byte, state_id_t &child) noexcept {
        size_t const edge = find_edge_(parent, byte);
        if (edge != SZ_SIZE_MAX) {
            child = edges_[edge].child;
            return status_t::success_k;
        }
        status_t const status = allocate_raw_state_(child);
        if (status != status_t::success_k) return status;
        return add_edge_(parent, byte, child);
    }

    /**
     *  @brief Byte-exact trie insertion: standard follow-or-create walk, one state per byte consumed.
     */
    status_t try_insert_cased_(span<byte_t const> needle, state_id_t needle_index) noexcept {
        status_t status = ensure_root_();
        if (status != status_t::success_k) return status;

        state_id_t current_state = 0;
        for (size_t offset = 0; offset < needle.size(); ++offset) {
            status = follow_or_create_(current_state, needle[offset], current_state);
            if (status != status_t::success_k) return status;
        }

        return add_output_(current_state, needle_index, needle.size());
    }

    /** @brief Decodes and fully folds @p needle into `folded_needle_`, in canonical UTF-8 bytes. */
    status_t fold_needle_(span<byte_t const> needle) noexcept {
        folded_needle_.clear();
        byte_t const *cursor = needle.begin();
        byte_t const *const needle_end = needle.end();
        while (cursor != needle_end) {
            rune_t rune;
            rune_length_t const consumed = sz_rune_decode(reinterpret_cast<cptr_t>(cursor),
                                                          reinterpret_cast<cptr_t>(needle_end), &rune);
            if (consumed == sz_rune_invalid_k) return status_t::invalid_utf8_k;
            rune_t images[3];
            size_t const runes = sz_unicode_fold_codepoint_(rune, images);
            for (size_t index = 0; index < runes; ++index) {
                u8_t encoded[4];
                rune_length_t const encoded_length = sz_rune_encode(images[index], encoded);
                for (size_t byte = 0; byte < (size_t)encoded_length; ++byte)
                    if (folded_needle_.try_push_back((byte_t)encoded[byte]) != status_t::success_k)
                        return status_t::bad_alloc_k;
            }
            cursor += consumed;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Case-folded trie insertion: fold the needle once, then insert those bytes literally.
     *
     *  The haystack is folded as the walk consumes it, so both sides meet in one canonical byte stream and
     *  the trie has nothing case-specific left in it - which is what keeps it a tree, and its failure links
     *  single-valued. A needle that folds to the same bytes as an earlier one simply shares its path.
     */
    status_t try_insert_uncased_(span<byte_t const> needle, state_id_t needle_index) noexcept {
        status_t status = ensure_root_();
        if (status != status_t::success_k) return status;

        // Reject malformed UTF-8 outright: the walk resets to the root on a malformed haystack byte, so a
        // needle carrying one could never match, and accepting it would only hide the caller's mistake.
        status = fold_needle_(needle);
        if (status != status_t::success_k) return status;
        if (folded_needle_.size() == 0) return status_t::success_k;

        return try_insert_cased_({folded_needle_.data(), folded_needle_.size()}, needle_index);
    }

#pragma endregion Construction Helpers

#pragma region Build Phases

    /**
     *  @brief Byte offsets of the spelling-automaton CSR, the only construction buffers whose size is fixed
     *         once insertion ends. Everything the splitting pass produces is sized by the walking-state
     *         count, which it only learns as it runs, so those buffers are growable members instead.
     */
    struct layout_t {
        size_t edges = 0, edge_offsets = 0, total = 0;
    };

    layout_t build_layout_(cpu_specs_t const &specs) const noexcept {
        scratch_amount_t amount {specs.cache_line_width};
        layout_t layout;
        layout.edges = amount, amount += edges_.size() * sizeof(csr_edge_t);
        layout.edge_offsets = amount, amount += (count_states_ + 1) * sizeof(state_id_t);
        layout.total = amount;
        return layout;
    }

    csr_edge_t *edges_at_(layout_t const &layout) noexcept {
        return (csr_edge_t *)(build_scratch_.data() + layout.edges);
    }
    state_id_t *edge_offsets_at_(layout_t const &layout) noexcept {
        return (state_id_t *)(build_scratch_.data() + layout.edge_offsets);
    }

    /**
     *  @brief Turns the insertion-time edge pool into a state-major CSR - one counting sort, no comparisons.
     *  @note Byte order inside a row is left alone; the packing search anchors on a `sz_byteset_t` child mask.
     */
    void compact_edges_into_csr_(layout_t const &layout) noexcept {
        state_id_t *const offsets = edge_offsets_at_(layout);
        csr_edge_t *const rows = edges_at_(layout);
        for (size_t state = 0; state <= count_states_; ++state) offsets[state] = 0;
        for (size_t edge = 0; edge < edges_.size(); ++edge) ++offsets[(size_t)edges_[edge].parent + 1];
        for (size_t state = 0; state < count_states_; ++state) offsets[state + 1] += offsets[state];
        // Scatter with `offsets[parent]` doubling as that row's fill cursor, which leaves every entry
        // holding its row's END; one backward shift turns them into starts again.
        for (size_t edge = 0; edge < edges_.size(); ++edge) {
            trie_edge_t const &record = edges_[edge];
            rows[offsets[record.parent]++] = csr_edge_t {record.child, record.byte};
        }
        for (size_t state = count_states_; state > 0; --state) offsets[state] = offsets[state - 1];
        offsets[0] = 0;
    }

    /**
     *  @brief Reorders one depth band of `trie_order_` by out-degree descending, so shallow high-fan-out
     *         states - the ones text keeps returning to - land in the hot tier regardless of dictionary
     *         content. A counting sort over a bounded degree.
     *  @note A band is the position range `[band_first, band_last)` within `trie_order_`, not an id range:
     *        insertion numbers states in needle order, so depth order lives only in this permutation.
     */
    void order_band_by_out_degree_(state_id_t const *offsets, size_t band_first, size_t band_last) noexcept {
        if (band_last - band_first < 2) return;

        state_id_t histogram[alphabet_size_k + 1] = {};
        for (size_t index = band_first; index < band_last; ++index) {
            state_id_t const state = trie_order_[index];
            ++histogram[offsets[state + 1] - offsets[state]];
        }
        // Suffix-summed, so the highest degree claims the lowest positions in the band.
        state_id_t running = state_id_of_(band_first);
        for (size_t degree = alphabet_size_k + 1; degree-- > 0;) {
            state_id_t const count = histogram[degree];
            histogram[degree] = running;
            running += count;
        }
        // Scattered through scratch rather than in place: the source is the band itself, so writing a
        // position before reading it would overwrite a state still waiting to be placed.
        for (size_t index = band_first; index < band_last; ++index) trie_order_scratch_[index] = trie_order_[index];
        for (size_t index = band_first; index < band_last; ++index) {
            state_id_t const state = trie_order_scratch_[index];
            trie_order_[histogram[offsets[state + 1] - offsets[state]]++] = state;
        }
    }

    /** @brief Fills the dense root row from the root's edges, defaulting every other byte to the
     *         self-looping root, so a failure chase that falls all the way back resolves in one lookup. */
    status_t fill_root_row_(state_id_t const *offsets, csr_edge_t const *rows) noexcept {
        if (trie_root_row_.try_resize(alphabet_size_k) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t byte = 0; byte < alphabet_size_k; ++byte) trie_root_row_[byte] = 0;
        for (size_t edge = offsets[0]; edge < offsets[1]; ++edge) trie_root_row_[rows[edge].byte] = rows[edge].child;
        return status_t::success_k;
    }

    /** @brief Child of @p state on @p byte among its literal edges, or `invalid_state_k` if none. */
    state_id_t find_trie_edge_(state_id_t const *offsets, csr_edge_t const *rows, state_id_t state,
                               u8_t byte) const noexcept {
        for (size_t edge = offsets[state]; edge < offsets[state + 1]; ++edge)
            if (rows[edge].byte == byte) return rows[edge].child;
        return invalid_state_k;
    }

    /** @brief Goto-completed target for @p state on @p byte; the root answers from its dense row. */
    state_id_t chase_trie_(state_id_t const *offsets, csr_edge_t const *rows, state_id_t state,
                           u8_t byte) const noexcept {
        for (state_id_t current = state;;) {
            if (current == 0) return trie_root_row_[byte];
            state_id_t const child = find_trie_edge_(offsets, rows, current, byte);
            if (child != invalid_state_k) return child;
            current = trie_states_[current].failure_state;
        }
    }

    /**
     *  @brief Assigns every state's failure link and lays `trie_order_` out depth band by depth band.
     *
     *  The classic Aho-Corasick construction: a child's failure link is found by chasing its parent's, and a
     *  failure link is always strictly shallower, so one shallow-to-deep pass finishes with no fixpoint. The
     *  trie is a tree, so each state is reached by exactly one edge and this visits each exactly once - which
     *  is why the state count is final at insertion and nothing is minted here.
     */
    status_t build_failure_links_(layout_t const &layout) noexcept {
        state_id_t const *const offsets = edge_offsets_at_(layout);
        csr_edge_t const *const rows = edges_at_(layout);

        if (trie_states_.try_resize(count_states_) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t state = 0; state < count_states_; ++state) trie_states_[state] = trie_state_t {};
        if (trie_order_.try_resize(count_states_) != status_t::success_k) return status_t::bad_alloc_k;
        if (trie_order_scratch_.try_resize(count_states_) != status_t::success_k) return status_t::bad_alloc_k;

        trie_order_[0] = 0; // ? The root fails to itself, which `trie_state_t` already defaults to.
        size_t discovered = 1;
        for (size_t band_first = 0, band_last = 1; band_first != band_last;) {
            for (size_t index = band_first; index < band_last; ++index) {
                state_id_t const parent = trie_order_[index];
                state_id_t const parent_failure = trie_states_[parent].failure_state;
                for (size_t edge = offsets[parent]; edge < offsets[parent + 1]; ++edge) {
                    state_id_t const child = rows[edge].child;
                    // A depth-one state fails to the root; anything deeper chases its parent's failure link.
                    trie_states_[child].failure_state =
                        parent == 0 ? 0 : chase_trie_(offsets, rows, parent_failure, rows[edge].byte);
                    trie_order_[discovered++] = child;
                }
                // The root's own row has to be dense before any chase consults it, and the root is the only
                // state in the first band, so filling it here is still ahead of every lookup.
                if (parent == 0) {
                    status_t const status = fill_root_row_(offsets, rows);
                    if (status != status_t::success_k) return status;
                }
            }
            order_band_by_out_degree_(offsets, band_first, band_last);
            band_first = band_last, band_last = discovered;
        }
        // Every state is reachable from the root by construction, so a short walk means the trie is malformed.
        sz_assert_(discovered == count_states_ && "A tree trie reaches every state exactly once");
        return status_t::success_k;
    }

    /** @brief Grows every slot-indexed array to hold @p minimum slots. */
    status_t ensure_slot_capacity_(size_t minimum) noexcept {
        if (minimum <= check_.size()) return status_t::success_k;
        size_t const old_capacity = check_.size();
        size_t const new_capacity = sz_size_bit_ceil(minimum);
        if (new_capacity >= (size_t)invalid_state_k) return status_t::overflow_risk_k;

        if (base_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;
        if (check_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;
        if (old_of_final_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;

        // Four words of headroom so a 256-bit feasibility window never reads past the end. `try_resize`
        // skips construction for trivially-constructible types, so every new word is explicitly cleared -
        // the bitmap is the only record of what is claimed, and a stale set bit would hide a free slot.
        size_t const old_words = occupied_bits_.size();
        size_t const new_words = (new_capacity >> 6) + 8;
        if (occupied_bits_.try_resize(new_words) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t word = old_words; word < new_words; ++word) occupied_bits_[word] = 0;

        for (size_t slot = old_capacity; slot < new_capacity; ++slot) {
            base_[slot] = 0;
            check_[slot] = invalid_state_k;
            old_of_final_[slot] = invalid_state_k;
        }
        return status_t::success_k;
    }

    /** @brief Marks @p slot taken; the bitmap is the only record of what is free. */
    void claim_slot_(size_t slot) noexcept { occupied_bits_[slot >> 6] |= (u64_t)1 << (slot & 63); }

    /**
     *  @brief First free slot at or after @p from, growing the arena when the search runs off the end.
     *
     *  Packing only ever fills forward, so `lowest_free_cursor_` never moves back and the whole walk across
     *  one build is amortized linear in the slot count.
     */
    status_t next_free_slot_(size_t from, size_t &found) noexcept {
        for (size_t word = from >> 6;; ++word) {
            if ((word << 6) >= check_.size()) {
                status_t const status = ensure_slot_capacity_(check_.size() * 2);
                if (status != status_t::success_k) return status;
            }
            u64_t vacancies = ~occupied_bits_[word];
            // Slots before `from` are not candidates, even when the word says they are free.
            if (word == (from >> 6)) vacancies &= ~(u64_t)0 << (from & 63);
            if (vacancies == 0) continue;
            found = (word << 6) + (size_t)sz_u64_ctz(vacancies);
            return status_t::success_k;
        }
    }

    /** @brief True when every byte set in @p wanted lands on a currently-free slot at @p base. */
    bool slots_are_free_(size_t base, sz_byteset_t const &wanted) const noexcept {
        size_t const word = base >> 6, shift = base & 63;
        for (size_t quarter = 0; quarter < 4; ++quarter) {
            u64_t const low = occupied_bits_[word + quarter];
            u64_t const high = occupied_bits_[word + quarter + 1];
            u64_t const window = shift == 0 ? low : (low >> shift) | (high << (64 - shift));
            if (window & wanted._u64s[quarter]) return false;
        }
        return true;
    }

    /**
     *  @brief Assigns every live state its published ID: the hot tier takes `[0, hot_count_)` in frequency
     *         order, and the rest are packed into the double array by a free list walked in the same
     *         depth-ascending order, so a state's literal parent always resolves before the state itself.
     *
     *  Every state has exactly one parent edge, so every state is placed exactly once and each published slot
     *  names a distinct state.
     */
    status_t pack_cold_tier_(state_id_t const *offsets, csr_edge_t const *rows) noexcept {
        // Every state is fresh with `published_id == invalid_state_k`, which the packing reads as
        // "not placed yet", so no reset loop is needed.
        lowest_free_cursor_ = hot_count_; // ? Hot states own the low IDs outright.
        status_t status = ensure_slot_capacity_(hot_count_ + alphabet_size_k);
        if (status != status_t::success_k) return status;

        for (size_t hot_index = 0; hot_index < hot_count_; ++hot_index) {
            state_id_t const state = trie_order_[hot_index];
            trie_states_[state].published_id = state_id_of_(hot_index);
            old_of_final_[hot_index] = state;
            claim_slot_(hot_index);
        }

        if (hot_count_ == 0) { // ? The root has no parent to assign it a cold ID.
            size_t assigned;
            status = next_free_slot_(lowest_free_cursor_, assigned);
            if (status != status_t::success_k) return status;
            claim_slot_(assigned);
            check_[assigned] = state_id_of_(assigned);
            old_of_final_[assigned] = 0;
            trie_states_[0].published_id = state_id_of_(assigned);
            lowest_free_cursor_ = assigned + 1;
        }

        for (size_t index = 0; index < count_states_; ++index) {
            state_id_t const parent = trie_order_[index];
            status = index < hot_count_ ? pack_hot_children_(offsets, rows, parent)
                                        : pack_cold_children_(offsets, rows, parent);
            if (status != status_t::success_k) return status;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Places a hot parent's children on whatever free slots come next.
     *
     *  `hot_rows_` addresses every child unconditionally, so nothing has to verify ownership through
     *  `check_` for them and they need no shared base. A child some other parent already placed keeps the
     *  slot it has, since both routes resolve to the same target through the completed row.
     */
    status_t pack_hot_children_(state_id_t const *offsets, csr_edge_t const *rows, state_id_t parent) noexcept {
        for (size_t edge = offsets[parent]; edge < offsets[parent + 1]; ++edge) {
            state_id_t const child = rows[edge].child;
            if (trie_states_[child].published_id != invalid_state_k) continue;
            size_t assigned;
            status_t const status = next_free_slot_(lowest_free_cursor_, assigned);
            if (status != status_t::success_k) return status;
            claim_slot_(assigned);
            check_[assigned] = state_id_of_(assigned);
            old_of_final_[assigned] = child;
            trie_states_[child].published_id = state_id_of_(assigned);
            lowest_free_cursor_ = assigned + 1;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Places a cold parent's children on one shared base, so `base_[parent] + byte` addresses each.
     *
     *  Candidates are scanned out of the occupancy bitmap anchored on the parent's smallest child byte, and
     *  the whole row is tested at once rather than probed child by child.
     */
    status_t pack_cold_children_(state_id_t const *offsets, csr_edge_t const *rows, state_id_t parent) noexcept {
        if (offsets[parent] == offsets[parent + 1]) return status_t::success_k;

        safe_array<state_id_t, alphabet_size_k> child_of_byte;
        sz_byteset_t child_mask;
        sz_byteset_init(&child_mask);
        for (size_t edge = offsets[parent]; edge < offsets[parent + 1]; ++edge) {
            sz_byteset_add_u8(&child_mask, rows[edge].byte);
            child_of_byte[rows[edge].byte] = rows[edge].child;
        }

        u8_t anchor_byte = 0;
        for (size_t quarter = 0; quarter < 4; ++quarter)
            if (child_mask._u64s[quarter]) {
                anchor_byte = (u8_t)(quarter * 64 + sz_u64_ctz(child_mask._u64s[quarter]));
                break;
            }

        for (size_t candidate = lowest_free_cursor_;;) {
            status_t status = next_free_slot_(candidate, candidate);
            if (status != status_t::success_k) return status;
            if (candidate < anchor_byte) {
                ++candidate;
                continue;
            }
            size_t const candidate_base = candidate - anchor_byte;
            status = ensure_slot_capacity_(candidate_base + alphabet_size_k);
            if (status != status_t::success_k) return status;
            if (!slots_are_free_(candidate_base, child_mask)) {
                ++candidate;
                continue;
            }

            state_id_t const parent_final = trie_states_[parent].published_id;
            base_[parent_final] = state_id_of_(candidate_base);
            for (size_t quarter = 0; quarter < 4; ++quarter)
                for (u64_t bits = child_mask._u64s[quarter]; bits; bits &= bits - 1) {
                    u8_t const byte = (u8_t)(quarter * 64 + sz_u64_ctz(bits));
                    size_t const slot = candidate_base + byte;
                    state_id_t const child = child_of_byte[byte];
                    claim_slot_(slot);
                    check_[slot] = parent_final;
                    old_of_final_[slot] = child;
                    // A tree gives every state exactly one parent edge, so no state is ever claimed twice and
                    // no slot is a relay for another's identity.
                    sz_assert_(trie_states_[child].published_id == invalid_state_k && "One parent edge per state");
                    trie_states_[child].published_id = state_id_of_(slot);
                }
            return status_t::success_k;
        }
    }

    /**
     *  @brief Sizes and fills the published output pool in one pass each.
     *
     *  A state's matches are its own plus its failure state's complete set, and BFS order guarantees the
     *  failure state is finished first, so the pool is written once at exactly the right size.
     */
    status_t size_and_fill_outputs_() noexcept {
        size_t running = 0;
        for (size_t index = 0; index < count_states_; ++index) {
            state_id_t const state = trie_order_[index];
            trie_state_t &entry = trie_states_[state];
            output_run_t const &own = output_runs_[state];
            // A state's matches are its own plus its failure state's whole run. Read on every byte step, so
            // the total rides `state_id_t` and the ceiling is refused here rather than assumed. Depth-band
            // order finished the failure state first, so its total is already set.
            size_t const total = static_cast<size_t>(own.own_count) +
                                 (state == 0 ? size_t {0}
                                             : static_cast<size_t>(trie_states_[entry.failure_state].total_count));
            if (total > static_cast<size_t>(invalid_state_k)) return status_t::overflow_risk_k;
            entry.total_count = static_cast<state_id_t>(total);
            entry.total_offset = running;
            running += entry.total_count;
        }

        if (outputs_.try_resize(running) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t index = 0; index < count_states_; ++index) {
            state_id_t const state = trie_order_[index];
            trie_state_t const &entry = trie_states_[state];
            output_run_t const &own = output_runs_[state];
            output_t *const destination = outputs_.data() + entry.total_offset;

            // The per-state list is most-recent-first, so filling it backwards restores insertion order.
            size_t written = own.own_count;
            for (size_t walk = own.own_head; walk != SZ_SIZE_MAX; walk = own_outputs_[walk].next) {
                pending_output_t const &pending = own_outputs_[walk];
                destination[--written] = output_t {pending.needle_index, pending.folded_match_bytes};
            }

            if (state == 0) continue;
            trie_state_t const &inherited = trie_states_[entry.failure_state];
            for (size_t position = 0; position < inherited.total_count; ++position)
                destination[own.own_count + position] = outputs_[inherited.total_offset + position];
        }
        return status_t::success_k;
    }

    /**
     *  @brief Materializes the hot tier's goto-completed rows by inheritance.
     *
     *  Goto completion means `goto(state, byte) == goto(fail(state), byte)` wherever `state` has no literal
     *  edge on `byte`. The depth-primary ordering puts `fail(state)` at a strictly smaller index, so a hot
     *  state's failure state is always hot and always already materialized - one row copy plus one store per
     *  literal edge, instead of a failure chase per cell.
     */
    status_t materialize_hot_rows_(state_id_t const *offsets, csr_edge_t const *rows) noexcept {
        if (hot_rows_.try_resize(hot_count_ * alphabet_size_k) != status_t::success_k) return status_t::bad_alloc_k;
        if (hot_count_ == 0) return status_t::success_k;

        state_id_t *const root_row = hot_rows_.data();
        for (size_t byte = 0; byte < alphabet_size_k; ++byte) root_row[byte] = root_;
        for (size_t edge = offsets[0]; edge < offsets[1]; ++edge)
            root_row[rows[edge].byte] = trie_states_[rows[edge].child].published_id;

        for (size_t hot_index = 1; hot_index < hot_count_; ++hot_index) {
            state_id_t const state = trie_order_[hot_index];
            // A failure state is strictly shallower, so depth-primary order places it earlier and its row is
            // already final. A real return rather than an assert: violated in a release build this would read
            // an unwritten row and bake wrong transitions into the published automaton, silently.
            size_t const inherited_index = (size_t)trie_states_[trie_states_[state].failure_state].published_id;
            if (inherited_index >= hot_index) return status_t::unexpected_dimensions_k;
            state_id_t const *const inherited = hot_rows_.data() + inherited_index * alphabet_size_k;
            state_id_t *const row = hot_rows_.data() + hot_index * alphabet_size_k;
            for (size_t byte = 0; byte < alphabet_size_k; ++byte) row[byte] = inherited[byte];
            for (size_t edge = offsets[state]; edge < offsets[state + 1]; ++edge)
                row[rows[edge].byte] = trie_states_[rows[edge].child].published_id;
        }
        return status_t::success_k;
    }

    /** @brief Fills `fail_`, `outputs_counts_`, and `outputs_offsets_` over the published slot range. */
    status_t publish_(size_t cold_capacity_published) noexcept {

        if (fail_.try_resize(cold_capacity_published) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_counts_.try_resize(cold_capacity_published) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_offsets_.try_resize(cold_capacity_published) != status_t::success_k) return status_t::bad_alloc_k;

        for (size_t slot = 0; slot < cold_capacity_published; ++slot) {
            state_id_t const raw_state = slot < old_of_final_.size() ? old_of_final_[slot] : invalid_state_k;
            if (raw_state == invalid_state_k) {
                outputs_counts_[slot] = 0, outputs_offsets_[slot] = 0;
                if (slot >= hot_count_) fail_[slot] = root_;
                continue;
            }
            trie_state_t const &entry = trie_states_[raw_state];
            outputs_counts_[slot] = entry.total_count;
            outputs_offsets_[slot] = entry.total_offset;
            max_outputs_per_state_ = sz_max_of_two(max_outputs_per_state_, entry.total_count);
            if (slot < hot_count_) continue;
            // Each state owns exactly one slot, so a slot's failure link is simply its state's, published.
            sz_assert_(entry.published_id == state_id_of_(slot) && "One published slot per state");
            fail_[slot] = trie_states_[entry.failure_state].published_id;
        }
        return status_t::success_k;
    }

#pragma endregion Build Phases

  public:
    aho_corasick_dictionary() = default;
    ~aho_corasick_dictionary() noexcept { reset(); }

    explicit aho_corasick_dictionary(allocator_t alloc) noexcept
        : edges_(alloc), edge_index_(alloc), own_outputs_(alloc), output_runs_(alloc), folded_needle_(alloc),
          build_scratch_(alloc), trie_states_(alloc), trie_order_(alloc), trie_order_scratch_(alloc),
          trie_root_row_(alloc), old_of_final_(alloc), occupied_bits_(alloc), hot_rows_(alloc), base_(alloc),
          check_(alloc), fail_(alloc), outputs_(alloc), outputs_counts_(alloc), outputs_offsets_(alloc), alloc_(alloc) {
    }

    aho_corasick_dictionary(aho_corasick_dictionary &&) noexcept = default;
    aho_corasick_dictionary &operator=(aho_corasick_dictionary &&) noexcept = default;
    aho_corasick_dictionary(aho_corasick_dictionary const &) = delete;
    aho_corasick_dictionary &operator=(aho_corasick_dictionary const &) = delete;

    /** @brief Frees every buffer construction needs and matching never touches. */
    void release_construction_scratch_() noexcept {
        edges_.reset();
        edge_index_.reset();
        own_outputs_.reset();
        output_runs_.reset();
        folded_needle_.reset();
        build_scratch_.reset();
        trie_states_.reset();
        trie_order_.reset();
        trie_order_scratch_.reset();
        trie_root_row_.reset();
        old_of_final_.reset();
        occupied_bits_.reset();
        lowest_free_cursor_ = 0;
    }

    void reset() noexcept {
        release_construction_scratch_();
        hot_rows_.reset();
        base_.reset();
        check_.reset();
        fail_.reset();
        outputs_.reset();
        outputs_counts_.reset();
        outputs_offsets_.reset();
        count_states_ = 0;
        count_needles_ = 0;
        max_folded_match_bytes_ = 0;
        min_folded_match_bytes_ = 0;
        max_source_match_bytes_ = 0;
        min_source_match_bytes_ = 0;
        max_outputs_per_state_ = 0;
        case_sensitivity_ = substrings_cased_k;
        hot_count_ = derive_hot_count_k;
        root_ = 0;
    }

    /** @brief Selects byte-exact or case-folded matching; must be called before the first `try_insert`. */
    void case_sensitivity(substrings_case_sensitivity_t desired) noexcept {
        sz_assert_(count_needles_ == 0 && "Case sensitivity can't change once needles have been inserted");
        case_sensitivity_ = desired;
    }
    substrings_case_sensitivity_t case_sensitivity() const noexcept { return case_sensitivity_; }

    /** @brief Forces the hot-tier size instead of deriving it from `cpu_specs_t` in `try_build`. */
    void hot_count(size_t desired) noexcept { hot_count_ = desired; }

    size_t count_states() const noexcept { return count_states_; }
    size_t count_needles() const noexcept { return count_needles_; }
    state_id_t max_folded_match_bytes() const noexcept { return max_folded_match_bytes_; }
    state_id_t min_folded_match_bytes() const noexcept { return min_folded_match_bytes_; }
    state_id_t max_source_match_bytes() const noexcept { return max_source_match_bytes_; }
    state_id_t min_source_match_bytes() const noexcept { return min_source_match_bytes_; }
    size_t hot_count() const noexcept { return hot_count_; }
    allocator_t const &allocator() const noexcept { return alloc_; }

    /** @brief Bytes held by both transition tiers together, hot rows plus the double array. */
    size_t transitions_bytes() const noexcept {
        return (hot_rows_.size() + base_.size() + check_.size() + fail_.size()) * sizeof(state_id_t);
    }

    /**
     *  @brief Adds a single @p needle to the vocabulary, assigning it a unique, insertion-order needle ID.
     *  @note Can't be called after `try_build`. Can't be called from multiple threads at the same time.
     *  @retval `status_t::success_k` The needle was successfully added.
     *  @retval `status_t::bad_alloc_k` Memory allocation failed.
     *  @retval `status_t::overflow_risk_k` Too many needles or states for the current state ID type.
     *  @retval `status_t::invalid_utf8_k` In `substrings_uncased_k` mode, the needle was not valid UTF-8.
     *  @retval `status_t::unexpected_dimensions_k` The needle was empty.
     *
     *  An empty needle is rejected rather than skipped: it would match at every one of `haystack_length + 1`
     *  positions, and dropping it silently would shift every later needle's reported `needle_index`.
     */
    status_t try_insert(span<byte_t const> needle) noexcept {
        if (needle.size() == 0) return status_t::unexpected_dimensions_k;
        if (count_needles_ >= (size_t)invalid_state_k) return status_t::overflow_risk_k;

        state_id_t const needle_index = static_cast<state_id_t>(count_needles_);
        status_t const status = case_sensitivity_ == substrings_uncased_k ? try_insert_uncased_(needle, needle_index)
                                                                          : try_insert_cased_(needle, needle_index);
        if (status != status_t::success_k) return status;
        ++count_needles_;
        return status_t::success_k;
    }

    status_t try_insert(span<char const> needle) noexcept { return try_insert(needle.template cast<byte_t const>()); }

    /**
     *  @brief Constructs the automaton from the vocabulary. Can only be called @b once.
     *  @param[in] specs Sizes the hot tier from the host's last-level cache, unless `hot_count` forced it.
     *
     *  Seven phases, each named below: the edge pool becomes the spelling CSR, the splitting pass derives the
     *  walking automaton one depth band at a time while ordering each band by out-degree, the hot/cold split
     *  falls out of that ordering, the double array packs the rest, and the outputs and hot rows materialize.
     */
    status_t try_build(cpu_specs_t const &specs = {}) noexcept {
        status_t status = ensure_root_();
        if (status != status_t::success_k) return status;

        // Uncased preimages add edges into pre-existing states, so the edge pool is not bounded by the state
        // pool - yet the CSR row offsets below store edge ordinals in `state_id_t` cells.
        if (edges_.size() > (size_t)invalid_state_k) return status_t::overflow_risk_k;

        // Every sub-buffer is written before it is read, so zeroing the block first would be pure waste.
        layout_t const layout = build_layout_(specs);
        if (build_scratch_.try_resize_uninitialized(layout.total) != status_t::success_k) return status_t::bad_alloc_k;
        compact_edges_into_csr_(layout);

        // The insertion pools have no reader past compaction; the CSR in `build_scratch_` carries every edge,
        // and every phase below walks it rather than `find_edge_`.
        edges_.reset();
        edge_index_.reset();

        state_id_t const *const offsets = edge_offsets_at_(layout);
        csr_edge_t const *const rows = edges_at_(layout);

        status = build_failure_links_(layout);
        if (status != status_t::success_k) return status;

        // Hot rows are shared, read-mostly, and re-entered on nearly every byte, so they're sized against
        // the last-level cache rather than a private L2 slice.
        if (hot_count_ == derive_hot_count_k) hot_count_ = specs.l3_bytes / (alphabet_size_k * sizeof(state_id_t));
        hot_count_ = sz_min_of_two(hot_count_, count_states_);

        status = pack_cold_tier_(offsets, rows);
        if (status != status_t::success_k) return status;

        root_ = trie_states_[0].published_id;
        sz_assert_(root_ == 0 && "The root is the unique shallowest state, so it always sorts first");

        // The exclusive published bound: not `hot_count_` plus the cold-state count, since a packed child's
        // ID is address arithmetic and can skip past slots no state ever ended up owning.
        size_t state_count_published = hot_count_;
        for (size_t state = 0; state < count_states_; ++state)
            state_count_published = sz_max_of_two(state_count_published, (size_t)trie_states_[state].published_id + 1);
        size_t const cold_capacity_published = state_count_published + (alphabet_size_k - 1);

        // `base_` and `check_` were written in place by the packing above; widening them here only extends
        // the address-arithmetic headroom a `base_[state] + byte` lookup can reach.
        status = ensure_slot_capacity_(cold_capacity_published);
        if (status != status_t::success_k) return status;

        status = size_and_fill_outputs_();
        if (status != status_t::success_k) return status;
        status = materialize_hot_rows_(offsets, rows);
        if (status != status_t::success_k) return status;
        status = publish_(cold_capacity_published);
        if (status != status_t::success_k) return status;

        count_states_ = state_count_published;
        release_construction_scratch_();
        return status_t::success_k;
    }

    /**
     *  @brief Adopts an already-built @p wider dictionary, narrowing every published array to this width.
     *
     *  The walking automaton is derived once at the widest id and narrowed here, so a vocabulary that fits a
     *  smaller id pays no second derivation - only a copy of the published arrays, whose rows then halve.
     *  A `u16` row is 512 bytes against `u32`'s 1024, so twice the automaton stays cache-resident, which is
     *  what the tier split is sized against in the first place.
     *
     *  Reads @p wider through its public view and accessors alone, so the two widths need no friendship.
     *  @retval `status_t::overflow_risk_k` Some published value exceeds this width; @p wider stays usable.
     *  @retval `status_t::bad_alloc_k` Memory allocation failed.
     */
    template <typename wider_id_type_, typename wider_allocator_type_>
    status_t try_build(aho_corasick_dictionary<wider_id_type_, wider_allocator_type_> const &wider) noexcept {
        static_assert(sizeof(state_id_t) <= sizeof(wider_id_type_), "This overload only ever narrows");
        auto const source = wider.view();

        // Every ceiling this width imposes, tested before a single element is copied. Slot ids reach past the
        // state count by the alphabet's headroom, since a packed child's id is address arithmetic.
        size_t const slots_published = (size_t)source.state_count + (alphabet_size_k - 1);
        if (slots_published > (size_t)invalid_state_k) return status_t::overflow_risk_k;
        if (wider.count_needles() > (size_t)invalid_state_k) return status_t::overflow_risk_k;
        if ((size_t)source.max_source_match_bytes > (size_t)invalid_state_k) return status_t::overflow_risk_k;
        if ((size_t)source.max_outputs_per_state > (size_t)invalid_state_k) return status_t::overflow_risk_k;

        size_t const hot_cells = (size_t)source.hot_count * alphabet_size_k;
        if (hot_rows_.try_resize(hot_cells) != status_t::success_k) return status_t::bad_alloc_k;
        if (base_.try_resize(slots_published) != status_t::success_k) return status_t::bad_alloc_k;
        if (check_.try_resize(slots_published) != status_t::success_k) return status_t::bad_alloc_k;
        if (fail_.try_resize(slots_published) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_.try_resize(source.outputs_total) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_counts_.try_resize(slots_published) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_offsets_.try_resize(slots_published) != status_t::success_k) return status_t::bad_alloc_k;

        for (size_t cell = 0; cell < hot_cells; ++cell)
            hot_rows_[cell] = static_cast<state_id_t>(source.hot_rows[cell]);
        for (size_t slot = 0; slot < slots_published; ++slot) {
            base_[slot] = static_cast<state_id_t>(source.base[slot]);
            check_[slot] = static_cast<state_id_t>(source.check[slot]);
            fail_[slot] = static_cast<state_id_t>(source.fail[slot]);
            outputs_counts_[slot] = static_cast<state_id_t>(source.outputs_counts[slot]);
            outputs_offsets_[slot] = source.outputs_offsets[slot]; // ? Indexes a pool, so it stays `size_t`
        }
        for (size_t output = 0; output < source.outputs_total; ++output)
            outputs_[output] = output_t {static_cast<state_id_t>(source.outputs[output].needle_index),
                                         static_cast<state_id_t>(source.outputs[output].folded_match_bytes)};

        count_states_ = source.state_count;
        count_needles_ = wider.count_needles();
        hot_count_ = source.hot_count;
        root_ = static_cast<state_id_t>(source.root);
        max_folded_match_bytes_ = static_cast<state_id_t>(source.max_folded_match_bytes);
        min_folded_match_bytes_ = static_cast<state_id_t>(source.min_folded_match_bytes);
        max_source_match_bytes_ = static_cast<state_id_t>(source.max_source_match_bytes);
        min_source_match_bytes_ = static_cast<state_id_t>(source.min_source_match_bytes);
        max_outputs_per_state_ = static_cast<state_id_t>(source.max_outputs_per_state);
        case_sensitivity_ = wider.case_sensitivity();
        return status_t::success_k;
    }

#pragma region Published View

    using view_t = aho_corasick_view<state_id_t>;

    /**
     *  @brief Immutable, trivially-copyable view of this dictionary, safe to pass to a CUDA kernel by value.
     *  @note Every pointer refers to storage owned by `*this`, which must outlive the view.
     */
    view_t view() const noexcept {
        view_t result;
        result.hot_rows = hot_rows_.data();
        result.base = base_.data();
        result.check = check_.data();
        result.fail = fail_.data();
        result.outputs = outputs_.data();
        result.outputs_counts = outputs_counts_.data();
        result.outputs_offsets = outputs_offsets_.data();
        result.outputs_total = outputs_.size();
        result.hot_count = state_id_of_(hot_count_);
        result.state_count = state_id_of_(count_states_);
        result.root = root_;
        result.max_folded_match_bytes = max_folded_match_bytes_;
        result.min_folded_match_bytes = min_folded_match_bytes_;
        result.max_source_match_bytes = max_source_match_bytes_;
        result.min_source_match_bytes = min_source_match_bytes_;
        result.max_outputs_per_state = max_outputs_per_state_;
        result.case_sensitivity = case_sensitivity_;
        return result;
    }

#pragma endregion Published View

#pragma region Matching

    /**
     *  @brief Finds all occurrences of all needles in the @p haystack, byte for byte.
     *  @note This is the serial reference oracle: obvious correctness over speed.
     *  @param[in] callback Invoked as `callback(needle_index, match_offset, match_length)` with offsets
     *             relative to the span handed in, returning `true` to continue.
     */
    template <typename callback_type_>
    void find_cased_(span<byte_t const> haystack, callback_type_ &&callback) const noexcept {
        view_t const automaton = view();
        state_id_t current_state = automaton.root;
        for (size_t offset = 0; offset < haystack.size(); ++offset) {
            u8_t const byte = haystack[offset];
            size_t const output_count = aho_corasick_step_counting(automaton, current_state, byte);
            if (output_count == 0) continue;
            size_t const output_offset = automaton.outputs_offsets[current_state];

            for (size_t index = 0; index < output_count; ++index) {
                output_t const &output = automaton.outputs[output_offset + index];
                size_t const match_length = output.folded_match_bytes;
                // Tested by addition rather than by subtracting the length from the position: the walk
                // always restarts at the root at this span's own start, so a match can never reach behind
                // it, but a subtraction would wrap and read as in-bounds if that ever stopped holding.
                if (offset + 1 < match_length) continue;
                if (!callback((size_t)output.needle_index, offset + 1 - match_length, match_length)) return;
            }
        }
    }

    /**
     *  @brief Finds all occurrences of all needles in the @p haystack, folding it one codepoint at a time.
     *  @param[in] callback Invoked as `callback(needle_index, match_offset, match_length)` with offsets
     *             relative to the span handed in, returning `true` to continue.
     *
     *  Only a byte ending a folded rune can end a match, so a reported end is always a whole codepoint's.
     */
    template <typename callback_type_>
    void find_uncased_(span<byte_t const> haystack, callback_type_ &&callback) const noexcept {
        view_t const automaton = view();
        state_id_t current_state = automaton.root;

        substrings_folded_cursor_t cursor;
        substrings_folded_cursor_init(cursor, haystack);

        size_t folded = 0, last_break_folded_end = 0;
        substrings_folded_byte_t step;
        while (substrings_folded_cursor_next(cursor, step)) {
            ++folded;
            if (step.malformed) {
                // A malformed byte can never sit inside a match, so the walk drops back to the root and
                // resynchronizes one byte at a time, exactly as the folded iterators do.
                current_state = automaton.root;
                continue;
            }

            current_state = aho_corasick_step(automaton, current_state, step.byte);
            if (!step.rune_end) continue;
            // Claimed before this rune end reports: a match ending inside a boundary-breaking codepoint
            // starts inside it too, and subtracting its folded length would land mid-codepoint.
            if (step.breaks_boundary) last_break_folded_end = folded + step.trailing;

            size_t const output_count = automaton.outputs_counts[current_state];
            if (output_count == 0) continue;
            size_t const output_offset = automaton.outputs_offsets[current_state];
            for (size_t index = 0; index < output_count; ++index) {
                output_t const &output = automaton.outputs[output_offset + index];
                size_t const folded_length = output.folded_match_bytes;
                if (folded < folded_length) continue;

                substrings_resolved_match_t const resolved = substrings_folded_span(
                    haystack, step, folded, last_break_folded_end, folded_length);
                if (resolved.repeats) continue;
                size_t const match_offset = resolved.source_offset;
                if (!callback((size_t)output.needle_index, match_offset, step.codepoint_end - match_offset)) return;
            }
        }
    }

    /** @brief Finds all occurrences of all needles in the @p haystack, folding it when the mode asks. */
    template <typename callback_type_>
    void find(span<byte_t const> haystack, callback_type_ &&callback) const noexcept {
        if (case_sensitivity_ == substrings_uncased_k)
            return find_uncased_(haystack, std::forward<callback_type_>(callback));
        find_cased_(haystack, std::forward<callback_type_>(callback));
    }

    template <typename callback_type_>
    void find(span<char const> haystack, callback_type_ &&callback) const noexcept {
        find(haystack.template cast<byte_t const>(), std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Counts the number of occurrences of all the needles in the @p haystack.
     *  @return The number of potentially-overlapping occurrences.
     */
    size_t count(span<byte_t const> haystack) const noexcept {
        // The folded walk reports at folded-rune ends rather than at every byte, and collapses spans that
        // resolve to one, so a byte-per-step tally would not agree with what `find` emits. Counting through
        // the same walk is what keeps `try_find`'s count-then-write pass consistent.
        if (case_sensitivity_ == substrings_uncased_k) {
            size_t total = 0;
            find_uncased_(haystack, [&](size_t, size_t, size_t) noexcept { return ++total, true; });
            return total;
        }

        view_t const automaton = view();
        size_t total = 0;
        state_id_t current_state = automaton.root;
        // One 4-byte load feeds four transitions - the state chain stays strictly serial, and `sz_u32_load`
        // absorbs misalignment itself, so only a tail loop remains.
        size_t offset = 0;
        for (; offset + 4 <= haystack.size(); offset += 4) {
            sz_u32_vec_t const quad = sz_u32_load((sz_cptr_t)(haystack.data() + offset));
            total += aho_corasick_step_counting(automaton, current_state, quad.u8s[0]);
            total += aho_corasick_step_counting(automaton, current_state, quad.u8s[1]);
            total += aho_corasick_step_counting(automaton, current_state, quad.u8s[2]);
            total += aho_corasick_step_counting(automaton, current_state, quad.u8s[3]);
        }
        for (; offset < haystack.size(); ++offset)
            total += aho_corasick_step_counting(automaton, current_state, haystack[offset]);
        return total;
    }

    /**
     *  @brief Emits the matches of @p haystack that share no bytes, one per accepted start position.
     *  @param[in] pending_starts Scratch of `substrings_pending_starts_width` entries, at least one;
     *             contents on entry are ignored.
     *  @param[in] callback Invoked as `callback(needle_index, match_offset, match_length)` in ascending
     *             start order, returning `true` to continue.
     *
     *  Matches surface at their end, so the earliest start is not the first seen: over "abcd" against
     *  {"bc", "abcd"}, "bc" completes first and "abcd" starts before it. A start settles only once the walk
     *  is `max_source_match_bytes` past it, which is what the pending starts hold.
     */
    template <typename callback_type_>
    void find_leftmost(span<byte_t const> haystack, span<pending_start_t> pending_starts,
                       substrings_overlap_policy_t policy, callback_type_ &&callback) const noexcept {

        sz_assert_(policy != substrings_overlapping_k && "Overlapping matches are reported through `find`");
        size_t const width = pending_starts.size();
        sz_assert_(width >= substrings_pending_starts_width(max_source_match_bytes_));
        sz_assert_((width & (width - 1)) == 0 && "A power-of-two width is what turns the lookup into a mask");
        size_t const mask = width - 1;
        for (size_t slot_index = 0; slot_index < width; ++slot_index) pending_starts[slot_index] = {};

        size_t cursor = 0, settled = 0;
        bool keep_going = true;
        auto const accept_start = [&](size_t start) noexcept {
            pending_start_t &slot = pending_starts[start & mask];
            if (slot.source_match_bytes != 0 && start >= cursor) {
                keep_going = callback((size_t)slot.needle_index, start, (size_t)slot.source_match_bytes);
                cursor = start + slot.source_match_bytes;
            }
            slot = {};
        };

        // `find` reports in non-decreasing end order, so the starts drain from the end each match reaches -
        // no second walk, and the cursor test waits until a start can no longer be outbid.
        size_t undrained_end = 0;
        find(haystack, [&](size_t needle_index, size_t start, size_t length) noexcept {
            size_t const settles_before = start + length > width ? start + length - width : 0;
            for (; settled < settles_before && keep_going; ++settled) accept_start(settled);
            if (!keep_going) return false;
            pending_start_t const challenger {(u32_t)needle_index, (u32_t)length};
            pending_start_t &slot = pending_starts[start & mask];
            if (substrings_leftmost_wins(challenger, slot, policy)) slot = challenger;
            undrained_end = sz_max_of_two(undrained_end, start + 1);
            return true;
        });

        // Draining stops at the last start any match claimed rather than at the haystack's end: a slot is
        // only ever occupied by a start a match reported, and visiting positions past the final one would
        // both waste a pass over the whole haystack and re-report a stale slot at a position it never
        // matched. A haystack no needle hits leaves this at zero and drains nothing at all.
        for (; settled < undrained_end && keep_going; ++settled) accept_start(settled);
    }

    template <typename callback_type_>
    void find_leftmost(span<char const> haystack, span<pending_start_t> pending_starts,
                       substrings_overlap_policy_t policy, callback_type_ &&callback) const noexcept {
        find_leftmost(haystack.template cast<byte_t const>(), pending_starts, policy,
                      std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Every match of @p haystack, in the order @p policy reports them.
     *  @param[in] pending_starts Scratch the leftmost policies settle starts in; unused when they overlap.
     *
     *  This is the one place the policy picks a walk. Engines forward their argument here rather than
     *  branching on it themselves, so a new backend implements this pair and nothing else.
     */
    template <typename callback_type_>
    void visit(span<byte_t const> haystack, substrings_overlap_policy_t policy, span<pending_start_t> pending_starts,
               callback_type_ &&callback) const noexcept {
        if (policy == substrings_overlapping_k) return find(haystack, std::forward<callback_type_>(callback));
        find_leftmost(haystack, pending_starts, policy, std::forward<callback_type_>(callback));
    }

    /**
     *  @brief How many matches `visit` would report, without enumerating them where it need not.
     *  @param[in] pending_starts Scratch the leftmost policies settle starts in; unused when they overlap.
     */
    size_t count(span<byte_t const> haystack, substrings_overlap_policy_t policy,
                 span<pending_start_t> pending_starts) const noexcept {
        // The overlapping tally has a dedicated four-byte-load walk that never enumerates an output run.
        if (policy == substrings_overlapping_k) return count(haystack);
        size_t total = 0;
        find_leftmost(haystack, pending_starts, policy, [&](size_t, size_t, size_t) noexcept {
            ++total;
            return true;
        });
        return total;
    }

#pragma endregion Matching
};

using substrings_u16_dictionary_t = aho_corasick_dictionary<u16_t, std::allocator<char>>;
using substrings_u32_dictionary_t = aho_corasick_dictionary<u32_t, std::allocator<char>>;

#pragma endregion Dictionary

#pragma region Rewriting

/** @brief Bytes @p haystack becomes once every match is swapped for its needle's replacement. */
template <typename dictionary_type_, typename replacements_type_>
size_t substrings_rewritten_size(dictionary_type_ const &dictionary, span<byte_t const> haystack,
                                 span<typename dictionary_type_::pending_start_t> pending_starts,
                                 substrings_overlap_policy_t policy, replacements_type_ const &replacements) noexcept {
    size_t removed = 0, added = 0;
    dictionary.find_leftmost(haystack, pending_starts, policy,
                             [&](size_t needle_index, size_t, size_t length) noexcept {
                                 removed += length;
                                 added += to_bytes_view(replacements[needle_index]).size();
                                 return true;
                             });
    // Accumulated apart rather than netted per match, so a shrinking rewrite never wraps the unsigned sum.
    return haystack.size() - removed + added;
}

/** @brief Writes the rewritten @p haystack at @p output, whose room the caller has already reserved. */
template <typename dictionary_type_, typename replacements_type_>
void substrings_rewrite(dictionary_type_ const &dictionary, span<byte_t const> haystack,
                        span<typename dictionary_type_::pending_start_t> pending_starts,
                        substrings_overlap_policy_t policy, replacements_type_ const &replacements,
                        char *output) noexcept {
    size_t cursor = 0;
    dictionary.find_leftmost(haystack, pending_starts, policy,
                             [&](size_t needle_index, size_t offset, size_t length) noexcept {
                                 span<byte_t const> const replacement = to_bytes_view(replacements[needle_index]);
                                 sz_copy(output, (sz_cptr_t)(haystack.data() + cursor), offset - cursor);
                                 output += offset - cursor;
                                 sz_copy(output, (sz_cptr_t)replacement.data(), replacement.size());
                                 output += replacement.size();
                                 cursor = offset + length;
                                 return true;
                             });
    sz_copy(output, (sz_cptr_t)(haystack.data() + cursor), haystack.size() - cursor);
}

/** @brief Rejects a rewrite under a policy that leaves no non-overlapping cover to substitute. */
inline status_t substrings_check_rewritable(substrings_overlap_policy_t policy) noexcept {
    return policy == substrings_overlapping_k ? status_t::unknown_k : status_t::success_k;
}

/** @brief Turns per-haystack sizes into `size() - 1` boundaries plus a terminator, in place. */
inline void substrings_sizes_into_offsets(span<size_t> offsets, size_t &total) noexcept {
    total = 0;
    for (size_t index = 0; index + 1 < offsets.size(); ++index) {
        size_t const size = offsets[index];
        offsets[index] = total;
        total += size;
    }
    offsets[offsets.size() - 1] = total;
}

#pragma endregion Rewriting

#pragma region Scoring

/** @brief Whether a caller supplied its own length for a document, or wants the haystack's byte count. */
enum class substrings_document_length_t : bool {
    /** @brief Take @p document_length as given, in whatever unit the pipeline normalizes by. */
    given_k,
    /** @brief Ignore @p document_length and measure the haystack in bytes, the only unit this engine owns. */
    haystack_bytes_k,
};

/**
 *  @brief Counts every occurrence of every needle in @p haystack into @p frequencies.
 *  @param[in,out] touched_count Grows by the needles this call hits for the first time.
 *
 *  Split from the scoring below so several cores can tally slices of one haystack into their own rows and
 *  merge afterwards, which is the only way a single long document reaches more than one core.
 */
template <typename dictionary_type_>
void substrings_bm25_tally(dictionary_type_ const &dictionary, span<byte_t const> haystack, span<u32_t> frequencies,
                           span<u32_t> touched, size_t &touched_count) noexcept {
    dictionary.find(haystack, [&](size_t needle_index, size_t, size_t) noexcept {
        if (frequencies[needle_index]++ == 0) touched[touched_count++] = (u32_t)needle_index;
        return true;
    });
}

/**
 *  @brief Scores one tallied document, leaving @p frequencies zeroed again for whoever runs next.
 *  @param[in] touched The needles the document hit, so the reset skips the rest of the vocabulary.
 */
inline f32_t substrings_bm25_reduce(span<f32_t const> needle_weights, substrings_bm25_t parameters,
                                    f32_t document_length, span<u32_t> frequencies, span<u32_t> touched,
                                    size_t touched_count) noexcept {

    // Float addition is not associative, so the summation order is part of the answer, and the order this
    // engine publishes is ascending by needle. A walk touches needles in whatever order the haystack spells
    // them, so they are ordered here rather than summed as they arrived - an insertion sort, because this
    // runs over the needles one document hit, not over the whole vocabulary.
    for (size_t slot = 1; slot < touched_count; ++slot) {
        u32_t const needle_index = touched[slot];
        size_t earlier = slot;
        for (; earlier > 0 && touched[earlier - 1] > needle_index; --earlier) touched[earlier] = touched[earlier - 1];
        touched[earlier] = needle_index;
    }

    f32_t score = 0;
    for (size_t slot = 0; slot < touched_count; ++slot) {
        size_t const needle_index = touched[slot];
        score += needle_weights[needle_index] *
                 substrings_bm25_term(parameters, (f32_t)frequencies[needle_index], document_length);
        frequencies[needle_index] = 0;
    }
    return score;
}

/**
 *  @brief One haystack's BM25 score, leaving @p frequencies zeroed again for whoever runs next.
 *  @param[in] length_source Whether @p document_length is the caller's own or should be measured here.
 *  @param[in] frequencies One counter per needle, zero on entry and on return.
 *  @param[in] touched Room for the needles this haystack hits, so the reset skips the rest of the vocabulary.
 */
template <typename dictionary_type_>
f32_t substrings_bm25_score(dictionary_type_ const &dictionary, span<byte_t const> haystack,
                            substrings_document_length_t length_source, f32_t document_length,
                            substrings_bm25_t parameters, span<f32_t const> needle_weights, span<u32_t> frequencies,
                            span<u32_t> touched) noexcept {
    size_t touched_count = 0;
    substrings_bm25_tally(dictionary, haystack, frequencies, touched, touched_count);
    if (length_source == substrings_document_length_t::haystack_bytes_k) document_length = (f32_t)haystack.size();
    return substrings_bm25_reduce(needle_weights, parameters, document_length, frequencies, touched, touched_count);
}

#pragma endregion Scoring

#pragma region Primary API

/**
 *  @brief Aho-Corasick-based @b single-threaded multi-pattern exact/case-folded substring search.
 *  @tparam state_id_type_ The type of the state ID.
 *  @tparam allocator_type_ The type of the allocator.
 *  @tparam capability_ Matches `sz_cap_serial_k` and any other capability that isn't parallel or CUDA; the
 *          two-level parallel specialization below claims `sz_caps_sp_k` specifically.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct substrings<allocator_type_, capability_,
                  std::enable_if_t<(capability_ & (sz_cap_parallel_k | sz_cap_cuda_k)) == 0>> {
    using allocator_t = allocator_type_;
    using narrow_dictionary_t = aho_corasick_dictionary<u16_t, allocator_t>;
    using wide_dictionary_t = aho_corasick_dictionary<u32_t, allocator_t>;
    using match_t = substrings_match_t;
    using pending_start_t = substrings_pending_start;
    static constexpr sz_capability_t capability_k = capability_;

    using size_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;
    using pending_start_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<pending_start_t>;
    using u32_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u32_t>;

    explicit substrings(allocator_t alloc = allocator_t()) noexcept
        : alloc_(alloc), dict_(std::in_place_type_t<wide_dictionary_t>(), alloc) {}
    void reset() noexcept {
        std::visit([](auto &dict) noexcept { dict.reset(); }, dict_);
    }

    /** @brief The state-id width `try_build` settled on, which the needle set alone decides. */
    substrings_state_width_t state_width() const noexcept {
        return std::holds_alternative<narrow_dictionary_t>(dict_) ? substrings_state_width_t::u16_k
                                                                  : substrings_state_width_t::u32_k;
    }
    size_t count_needles() const noexcept {
        return std::visit([](auto const &dict) noexcept { return dict.count_needles(); }, dict_);
    }
    size_t count_states() const noexcept {
        return std::visit([](auto const &dict) noexcept { return dict.count_states(); }, dict_);
    }
    size_t max_source_match_bytes() const noexcept {
        return std::visit([](auto const &dict) noexcept { return (size_t)dict.max_source_match_bytes(); }, dict_);
    }
    size_t min_source_match_bytes() const noexcept {
        return std::visit([](auto const &dict) noexcept { return (size_t)dict.min_source_match_bytes(); }, dict_);
    }
    size_t hot_count() const noexcept {
        return std::visit([](auto const &dict) noexcept { return dict.hot_count(); }, dict_);
    }
    substrings_case_sensitivity_t case_sensitivity() const noexcept {
        return std::visit([](auto const &dict) noexcept { return dict.case_sensitivity(); }, dict_);
    }

    /**
     *  @brief Runs @p callable against the automaton at whichever state-id width it settled on.
     *
     *  One dispatch per call rather than per byte: the walks it hands the dictionary to stay monomorphic,
     *  so this costs a branch where a width-templated engine cost an instantiation.
     */
    template <typename callable_type_>
    auto visit_dictionary(callable_type_ &&callable) const noexcept {
        return std::visit(std::forward<callable_type_>(callable), dict_);
    }

    /**
     *  @brief Indexes all of the @p needles strings into the FSM, at whichever state id it ends up fitting.
     *
     *  Construction runs wide, because a dictionary's state count is only known once it is built; the narrowing
     *  attempt is then itself the ceiling test, and only `overflow_risk_k` means "does not fit".
     *  @param[in] executor Taken for one shape across every entry point; construction stays on the calling
     *             thread, as building the FSM is not generally a bottleneck next to walking it.
     *  @param[in] specs Sizes the hot tier from the host's last-level cache.
     *  @note Replaces any previously indexed needle set: the automaton is rebuilt from scratch and the old one
     *        released, so an engine can be re-indexed for a different vocabulary or a different machine.
     *  @sa `aho_corasick_dictionary::try_insert` for the status codes this forwards.
     */
    template <typename needles_type_, typename executor_type_ = dummy_executor_t>
    status_t try_index(needles_type_ &&needles, substrings_case_sensitivity_t case_sensitivity = substrings_cased_k,
                       executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) noexcept {
        sz_unused_(executor);
        wide_dictionary_t wide(alloc_);
        wide.case_sensitivity(case_sensitivity);
        for (auto const &needle : needles) {
            status_t const status = wide.try_insert(to_bytes_view(needle));
            if (status != status_t::success_k) return status;
        }
        if (status_t const built = wide.try_build(specs); built != status_t::success_k) return built;

        narrow_dictionary_t narrow(alloc_);
        status_t const narrowed = narrow.try_build(wide);
        if (narrowed == status_t::success_k) {
            dict_.template emplace<narrow_dictionary_t>(std::move(narrow));
            return status_t::success_k;
        }
        if (narrowed != status_t::overflow_risk_k) return narrowed;
        dict_.template emplace<wide_dictionary_t>(std::move(wide));
        return status_t::success_k;
    }

    /**
     *  @brief Occurrences of all needles in each of the @p haystacks, for filtering and ranking.
     *  @param[in] overlap_policy Whether every overlapping match counts, or only the leftmost ones.
     *  @param[out] matches_total Sum of @p counts_per_haystack, which is what sizes a later `try_find` buffer.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
    status_t try_count(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                       span<size_t> counts_per_haystack, size_t &matches_total, executor_type_ &&executor = {},
                       cpu_specs_t const &specs = {}) noexcept {
        // A single-threaded walk has no split to schedule and no cache size to compare against.
        sz_unused_(executor);
        sz_unused_(specs);
        sz_assert_(counts_per_haystack.size() == haystacks.size());
        if (status_t const reserved = try_reserve_pending_starts_(overlap_policy); reserved != status_t::success_k)
            return reserved;

        matches_total = 0;
        std::visit(
            [&](auto const &dict) noexcept {
                for (size_t index = 0; index < counts_per_haystack.size(); ++index)
                    matches_total += counts_per_haystack[index] =
                        dict.count(to_bytes_view(haystacks[index]), overlap_policy, pending_starts_span_());
            },
            dict_);
        return status_t::success_k;
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks, resolved under @p overlap_policy.
     *  @param[out] matches_found Matches written, or - when @p matches is too small - the count that would be,
     *              so `matches.size() == 0` is a size query rather than a wasted call.
     *  @retval `status_t::unexpected_dimensions_k` @p matches is too small; nothing is written in that case.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
    status_t try_find(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                      span<substrings_match_t> matches, size_t &matches_found, executor_type_ &&executor = {},
                      cpu_specs_t const &specs = {}) noexcept {

        // Counting first is what makes the capacity refusable before any write.
        matches_found = 0;
        if (counts_per_haystack_.try_resize(haystacks.size()) != status_t::success_k) return status_t::bad_alloc_k;
        span<size_t> const counts_per_haystack {counts_per_haystack_.data(), haystacks.size()};
        if (status_t const status = try_count(haystacks, overlap_policy, counts_per_haystack, matches_found, executor,
                                              specs);
            status != status_t::success_k)
            return status;

        // The count survives the refusal, so a caller that brought no buffer still learns what to allocate.
        if (matches_found > matches.size()) return status_t::unexpected_dimensions_k;

        size_t count_written = 0;
        std::visit(
            [&](auto const &dict) noexcept {
                for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index)
                    dict.visit(to_bytes_view(haystacks[haystack_index]), overlap_policy, pending_starts_span_(),
                               [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                                   matches[count_written] = {haystack_index, needle_index, match_offset, match_length};
                                   count_written++;
                                   return true;
                               });
            },
            dict_);
        sz_assert_(count_written == matches_found);
        return status_t::success_k;
    }

    /**
     *  @brief Rewrites every haystack into one tape, substituting each match with its needle's replacement.
     *  @param[in] overlap_policy Must name a leftmost policy; an overlapping rewrite is not a function.
     *  @param[in] replacements One per needle, inserted verbatim.
     *  @param[out] output_offsets Rewritten boundaries, `haystacks.size() + 1` entries; always filled.
     *  @param[out] output_bytes_written Bytes written, or - when @p output_bytes is short - the size needed.
     *  @retval `status_t::unexpected_dimensions_k` @p output_bytes is too small, and nothing was written.
     */
    template <typename haystacks_type_, typename replacements_type_, typename executor_type_ = dummy_executor_t>
    status_t try_replace(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                         replacements_type_ const &replacements, span<char> output_bytes, span<size_t> output_offsets,
                         size_t &output_bytes_written, executor_type_ &&executor = {},
                         cpu_specs_t const &specs = {}) noexcept {
        sz_unused_(executor), sz_unused_(specs);
        sz_assert_(output_offsets.size() == haystacks.size() + 1);
        output_bytes_written = 0;
        if (status_t const rewritable = substrings_check_rewritable(overlap_policy); rewritable != status_t::success_k)
            return rewritable;
        if (status_t const reserved = try_reserve_pending_starts_(overlap_policy); reserved != status_t::success_k)
            return reserved;

        // Sizes land in the offsets array and become boundaries in place, so this needs no scratch of its own.
        std::visit(
            [&](auto const &dict) noexcept {
                for (size_t index = 0; index < haystacks.size(); ++index)
                    output_offsets[index] = substrings_rewritten_size(dict, to_bytes_view(haystacks[index]),
                                                                      pending_starts_span_(), overlap_policy,
                                                                      replacements);
            },
            dict_);
        substrings_sizes_into_offsets(output_offsets, output_bytes_written);
        if (output_bytes_written > output_bytes.size()) return status_t::unexpected_dimensions_k;

        std::visit(
            [&](auto const &dict) noexcept {
                for (size_t index = 0; index < haystacks.size(); ++index)
                    substrings_rewrite(dict, to_bytes_view(haystacks[index]), pending_starts_span_(), overlap_policy,
                                       replacements, output_bytes.data() + output_offsets[index]);
            },
            dict_);
        return status_t::success_k;
    }

    /**
     *  @brief Scores every haystack against the compiled needle set in one walk.
     *  @param[in] document_lengths One per haystack; an empty span uses byte lengths.
     *  @param[in] needle_weights One IDF or boost per needle.
     *  @param[out] scores One per haystack.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
    status_t try_score_bm25(haystacks_type_ const &haystacks, span<f32_t const> document_lengths,
                            substrings_bm25_t parameters, span<f32_t const> needle_weights, span<f32_t> scores,
                            executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) noexcept {
        sz_unused_(executor), sz_unused_(specs);
        sz_assert_(scores.size() == haystacks.size());
        sz_assert_(needle_weights.size() == count_needles());
        sz_assert_(document_lengths.size() == 0 || document_lengths.size() == haystacks.size());

        if (status_t const reserved = try_reserve_frequencies_(); reserved != status_t::success_k) return reserved;
        span<u32_t> const frequencies {frequencies_.data(), frequencies_.size()};
        span<u32_t> const touched {touched_needles_.data(), touched_needles_.size()};
        substrings_document_length_t const length_source = document_lengths.size()
                                                               ? substrings_document_length_t::given_k
                                                               : substrings_document_length_t::haystack_bytes_k;
        std::visit(
            [&](auto const &dict) noexcept {
                for (size_t index = 0; index < haystacks.size(); ++index)
                    scores[index] = substrings_bm25_score(dict, to_bytes_view(haystacks[index]), length_source,
                                                          document_lengths.size() ? document_lengths[index] : 0.0f,
                                                          parameters, needle_weights, frequencies, touched);
            },
            dict_);
        return status_t::success_k;
    }

  private:
    allocator_t alloc_ {};
    std::variant<narrow_dictionary_t, wide_dictionary_t> dict_;

    /** @brief Grow-only per-call scratch, reused across calls; concurrent calls on one engine are unsafe. */
    safe_vector<size_t, size_allocator_t> counts_per_haystack_ {};
    /** @brief The undecided starts a leftmost walk keeps; stays empty until a leftmost policy asks for it. */
    safe_vector<pending_start_t, pending_start_allocator_t> pending_starts_ {};

    /** @brief Sizes the pending-start scratch to the built dictionary, a no-op for the overlapping policy. */
    status_t try_reserve_pending_starts_(substrings_overlap_policy_t policy) noexcept {
        if (policy == substrings_overlapping_k) return status_t::success_k;
        size_t const width = substrings_pending_starts_width(max_source_match_bytes());
        return pending_starts_.size() == width ? status_t::success_k : pending_starts_.try_resize(width);
    }

    span<pending_start_t> pending_starts_span_() noexcept { return {pending_starts_.data(), pending_starts_.size()}; }

    /** @brief Sizes the frequency counters to the dictionary, leaving every one of them at zero. */
    status_t try_reserve_frequencies_() noexcept {
        size_t const needles = count_needles();
        if (frequencies_.size() == needles) return status_t::success_k;
        // `try_resize` leaves trivial types uninitialized, and the counters must start - and stay - zero.
        if (frequencies_.try_resize(needles) != status_t::success_k ||
            touched_needles_.try_resize(needles) != status_t::success_k)
            return status_t::bad_alloc_k;
        for (size_t index = 0; index < needles; ++index) frequencies_[index] = 0;
        return status_t::success_k;
    }

    /** @brief How often each needle hit the haystack being scored, and which needles those were. */
    safe_vector<u32_t, u32_allocator_t> frequencies_ {};
    safe_vector<u32_t, u32_allocator_t> touched_needles_ {};
};

#pragma endregion Primary API

#pragma region Parallel Backend

/**
 *  @brief  Aho-Corasick-based @b multi-threaded multi-pattern exact/case-folded substring search.
 *  @note   Construction of the FSM is not parallelized, as it is not generally a bottleneck.
 *
 *  Two levels of parallelism: one core per input below the L2 size, all cores on one input above it.
 *
 *  Matches straddling two threads' slices are the hard part. Rather than firing a callback concurrently
 *  and leaving the user to synchronize, this counts each slice first and then writes the slices in
 *  parallel into disjoint output ranges, so neither a mutex nor an atomic sits on the write path.
 */
template <typename allocator_type_, typename enable_>
struct substrings<allocator_type_, sz_caps_sp_k, enable_> {

    using allocator_t = allocator_type_;
    using narrow_dictionary_t = aho_corasick_dictionary<u16_t, allocator_t>;
    using wide_dictionary_t = aho_corasick_dictionary<u32_t, allocator_t>;
    using match_t = substrings_match_t;
    using pending_start_t = substrings_pending_start;
    static constexpr sz_capability_t capability_k = sz_caps_sp_k;

    using size_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;
    using pending_start_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<pending_start_t>;
    using u32_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u32_t>;

    explicit substrings(allocator_t alloc = allocator_t()) noexcept
        : alloc_(alloc), dict_(std::in_place_type_t<wide_dictionary_t>(), alloc) {}
    void reset() noexcept {
        std::visit([](auto &dict) noexcept { dict.reset(); }, dict_);
    }

    /** @brief The state-id width `try_build` settled on, which the needle set alone decides. */
    substrings_state_width_t state_width() const noexcept {
        return std::holds_alternative<narrow_dictionary_t>(dict_) ? substrings_state_width_t::u16_k
                                                                  : substrings_state_width_t::u32_k;
    }
    size_t count_needles() const noexcept {
        return std::visit([](auto const &dict) noexcept { return dict.count_needles(); }, dict_);
    }
    size_t count_states() const noexcept {
        return std::visit([](auto const &dict) noexcept { return dict.count_states(); }, dict_);
    }
    size_t max_source_match_bytes() const noexcept {
        return std::visit([](auto const &dict) noexcept { return (size_t)dict.max_source_match_bytes(); }, dict_);
    }
    size_t min_source_match_bytes() const noexcept {
        return std::visit([](auto const &dict) noexcept { return (size_t)dict.min_source_match_bytes(); }, dict_);
    }
    size_t hot_count() const noexcept {
        return std::visit([](auto const &dict) noexcept { return dict.hot_count(); }, dict_);
    }
    substrings_case_sensitivity_t case_sensitivity() const noexcept {
        return std::visit([](auto const &dict) noexcept { return dict.case_sensitivity(); }, dict_);
    }

    /**
     *  @brief Runs @p callable against the automaton at whichever state-id width it settled on.
     *
     *  One dispatch per call rather than per byte: the walks it hands the dictionary to stay monomorphic,
     *  so this costs a branch where a width-templated engine cost an instantiation.
     */
    template <typename callable_type_>
    auto visit_dictionary(callable_type_ &&callable) const noexcept {
        return std::visit(std::forward<callable_type_>(callable), dict_);
    }

    /**
     *  @brief Indexes all of the @p needles strings into the FSM, at whichever state id it ends up fitting.
     *
     *  Construction runs wide, because a dictionary's state count is only known once it is built; the narrowing
     *  attempt is then itself the ceiling test, and only `overflow_risk_k` means "does not fit".
     *  @param[in] executor Taken for one shape across every entry point; construction stays on the calling
     *             thread, as building the FSM is not generally a bottleneck next to walking it.
     *  @param[in] specs Sizes the hot tier from the host's last-level cache.
     *  @note Replaces any previously indexed needle set: the automaton is rebuilt from scratch and the old one
     *        released, so an engine can be re-indexed for a different vocabulary or a different machine.
     *  @sa `aho_corasick_dictionary::try_insert` for the status codes this forwards.
     */
    template <typename needles_type_, typename executor_type_ = dummy_executor_t>
    status_t try_index(needles_type_ &&needles, substrings_case_sensitivity_t case_sensitivity = substrings_cased_k,
                       executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) noexcept {
        sz_unused_(executor);
        wide_dictionary_t wide(alloc_);
        wide.case_sensitivity(case_sensitivity);
        for (auto const &needle : needles) {
            status_t const status = wide.try_insert(to_bytes_view(needle));
            if (status != status_t::success_k) return status;
        }
        if (status_t const built = wide.try_build(specs); built != status_t::success_k) return built;

        narrow_dictionary_t narrow(alloc_);
        status_t const narrowed = narrow.try_build(wide);
        if (narrowed == status_t::success_k) {
            dict_.template emplace<narrow_dictionary_t>(std::move(narrow));
            return status_t::success_k;
        }
        if (narrowed != status_t::overflow_risk_k) return narrowed;
        dict_.template emplace<wide_dictionary_t>(std::move(wide));
        return status_t::success_k;
    }

    /**
     *  @brief Occurrences of all needles in each of the @p haystacks, for filtering and ranking.
     *  @param[out] matches_total Sum of @p counts_per_haystack, which is what sizes a later `try_find` buffer.
     *  @param[in] specs Picks the threading strategy per haystack, by comparing its size against the L2.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t try_count(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                       span<size_t> counts_per_haystack, size_t &matches_total, executor_type_ &&executor = {},
                       cpu_specs_t const &specs = {}) noexcept {
        sz_assert_(counts_per_haystack.size() == haystacks.size());
        if (overlap_policy != substrings_overlapping_k)
            return count_all_leftmost_(haystacks, overlap_policy, counts_per_haystack, matches_total, executor);
        return count_all_overlapping_(haystacks, counts_per_haystack, matches_total, executor, specs);
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks, resolved under @p overlap_policy.
     *  @param[out] matches_found Matches written, or - when @p matches is too small - the count that would be.
     *  @retval `status_t::unexpected_dimensions_k` @p matches is too small; nothing is written in that case.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t try_find(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                      span<substrings_match_t> matches, size_t &matches_found, executor_type_ &&executor = {},
                      cpu_specs_t const &specs = {}) noexcept {
        if (overlap_policy != substrings_overlapping_k)
            return find_all_leftmost_(haystacks, overlap_policy, matches, matches_found, executor);
        return find_all_overlapping_(haystacks, matches, matches_found, executor, specs);
    }

    /**
     *  @brief Rewrites every haystack into one tape, substituting each match with its needle's replacement.
     *  @param[in] overlap_policy Must name a leftmost policy; an overlapping rewrite is not a function.
     *  @param[in] replacements One per needle, inserted verbatim.
     *  @param[out] output_offsets Rewritten boundaries, `haystacks.size() + 1` entries; always filled.
     *  @param[out] output_bytes_written Bytes written, or - when @p output_bytes is short - the size needed.
     *  @retval `status_t::unexpected_dimensions_k` @p output_bytes is too small, and nothing was written.
     */
    template <typename haystacks_type_, typename replacements_type_, typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t try_replace(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                         replacements_type_ const &replacements, span<char> output_bytes, span<size_t> output_offsets,
                         size_t &output_bytes_written, executor_type_ &&executor = {},
                         cpu_specs_t const &specs = {}) noexcept {
        sz_assert_(output_offsets.size() == haystacks.size() + 1);
        output_bytes_written = 0;
        if (status_t const rewritable = substrings_check_rewritable(overlap_policy); rewritable != status_t::success_k)
            return rewritable;
        size_t const cores = executor.threads_count();
        // One share row per large haystack, so the sizing pass's cover survives into the writing pass.
        size_t large_total = 0;
        for (size_t index = 0; index < haystacks.size(); ++index)
            if (is_large_(to_bytes_view(haystacks[index]).size(), specs)) ++large_total;
        if (status_t const reserved = try_reserve_pending_starts_(cores); reserved != status_t::success_k)
            return reserved;
        if (status_t const reserved = try_reserve_rewrite_shares_(cores, large_total); reserved != status_t::success_k)
            return reserved;

        using prong_t = typename std::decay<executor_type_>::type::prong_t;

        // Sizes land in the offsets array and become boundaries in place, so this needs no scratch of its own.
        visit_dictionary([&](auto const &dict) noexcept {
            executor.for_n_dynamic(haystacks.size(), [&](prong_t prong) noexcept {
                span<byte_t const> const haystack = to_bytes_view(haystacks[prong.task]);
                if (is_large_(haystack.size(), specs)) return;
                output_offsets[prong.task] = substrings_rewritten_size(dict, haystack, pending_starts_of_(prong.thread),
                                                                       overlap_policy, replacements);
            });
        });
        size_t large_index = 0;
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
            span<byte_t const> const haystack = to_bytes_view(haystacks[haystack_index]);
            if (!is_large_(haystack.size(), specs)) continue; // ? Already sized above.
            output_offsets[haystack_index] =
                size_one_large_(haystack, overlap_policy, replacements, executor, shares_of_large_(large_index, cores));
            ++large_index;
        }
        substrings_sizes_into_offsets(output_offsets, output_bytes_written);
        if (output_bytes_written > output_bytes.size()) return status_t::unexpected_dimensions_k;

        visit_dictionary([&](auto const &dict) noexcept {
            executor.for_n_dynamic(haystacks.size(), [&](prong_t prong) noexcept {
                span<byte_t const> const haystack = to_bytes_view(haystacks[prong.task]);
                if (is_large_(haystack.size(), specs)) return;
                substrings_rewrite(dict, haystack, pending_starts_of_(prong.thread), overlap_policy, replacements,
                                   output_bytes.data() + output_offsets[prong.task]);
            });
        });
        large_index = 0;
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
            span<byte_t const> const haystack = to_bytes_view(haystacks[haystack_index]);
            if (!is_large_(haystack.size(), specs)) continue;
            span<rewrite_share_t> const shares = shares_of_large_(large_index, cores);
            executor.for_threads([&](size_t core_index) noexcept {
                rewrite_share_t const &share = shares[core_index];
                if (share.output_bytes == 0) return;
                rewrite_share_(haystack, share, overlap_policy, replacements, pending_starts_of_(core_index),
                               spanned_of_(core_index),
                               output_bytes.data() + output_offsets[haystack_index] + share.output_offset);
            });
            ++large_index;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Scores every haystack against the compiled needle set in one walk.
     *  @param[in] document_lengths One per haystack; an empty span uses byte lengths.
     *  @param[in] needle_weights One IDF or boost per needle.
     *  @param[out] scores One per haystack.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t try_score_bm25(haystacks_type_ const &haystacks, span<f32_t const> document_lengths,
                            substrings_bm25_t parameters, span<f32_t const> needle_weights, span<f32_t> scores,
                            executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) noexcept {
        sz_assert_(scores.size() == haystacks.size());
        sz_assert_(needle_weights.size() == count_needles());
        sz_assert_(document_lengths.size() == 0 || document_lengths.size() == haystacks.size());

        if (status_t const reserved = try_reserve_frequencies_(executor.threads_count());
            reserved != status_t::success_k)
            return reserved;
        // Carries how many needles each core touched when a long haystack is split across all of them.
        if (counts_per_core_.try_resize(executor.threads_count()) != status_t::success_k) return status_t::bad_alloc_k;

        substrings_document_length_t const length_source = document_lengths.size()
                                                               ? substrings_document_length_t::given_k
                                                               : substrings_document_length_t::haystack_bytes_k;
        using prong_t = typename std::decay<executor_type_>::type::prong_t;

        // A haystack per core, for everything that fits a core's cache.
        visit_dictionary([&](auto const &dict) noexcept {
            executor.for_n_dynamic(haystacks.size(), [&](prong_t prong) noexcept {
                if (is_large_(to_bytes_view(haystacks[prong.task]).size(), specs)) return;
                size_t const first = prong.thread * needles_per_core_;
                scores[prong.task] = substrings_bm25_score(
                    dict, to_bytes_view(haystacks[prong.task]), length_source,
                    document_lengths.size() ? document_lengths[prong.task] : 0.0f, parameters, needle_weights,
                    {frequencies_.data() + first, needles_per_core_},
                    {touched_needles_.data() + first, needles_per_core_});
            });
        });

        // Every core on each long haystack: frequencies are order-independent, so the rows merge by addition.
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
            span<byte_t const> const haystack = to_bytes_view(haystacks[haystack_index]);
            if (!is_large_(haystack.size(), specs)) continue; // ? Already scored above.
            scores[haystack_index] = score_one_large_(haystack, length_source,
                                                      document_lengths.size() ? document_lengths[haystack_index] : 0.0f,
                                                      parameters, needle_weights, executor);
        }
        return status_t::success_k;
    }

  private:
    /**
     *  @brief Counts every overlapping match, splitting one haystack across cores once it outgrows the L2.
     *  @param[in] specs Picks the threading strategy per haystack, by comparing its size against the L2.
     */
    template <typename haystacks_type_, typename executor_type_>
    status_t count_all_overlapping_(haystacks_type_ const &haystacks, span<size_t> counts_per_haystack,
                                    size_t &matches_total, executor_type_ &executor,
                                    cpu_specs_t const &specs) noexcept {

        matches_total = 0;

        using haystack_t = typename haystacks_type_::value_type;
        static_assert(std::is_trivially_copyable<haystack_t>::value,
                      "The haystack should be trivially copyable for higher compatibility.");

        // On small strings, individually compute the counts.
        visit_dictionary([&](auto const &dict) noexcept {
            executor.for_n_dynamic(counts_per_haystack.size(), [&](size_t haystack_index) noexcept {
                haystack_t const &haystack = haystacks[haystack_index];
                if (is_large_(haystack.size(), specs)) return;
                counts_per_haystack[haystack_index] = dict.count(to_bytes_view(haystack));
            });
        });

        // On longer strings, throw all cores on each haystack.
        if (counts_per_core_.try_resize(executor.threads_count()) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t haystack_index = 0; haystack_index < counts_per_haystack.size(); ++haystack_index) {
            haystack_t const &haystack = haystacks[haystack_index];
            if (!is_large_(haystack.size(), specs)) continue; // ? Already processed above.
            auto const haystack_bytes = to_bytes_view(haystack);

            count_matches_per_core_(haystack_bytes, executor, counts_per_core_);
            size_t total = 0;
            for (size_t core_index = 0; core_index < counts_per_core_.size(); ++core_index)
                total += counts_per_core_[core_index];
            counts_per_haystack[haystack_index] = total;
        }

        for (size_t haystack_index = 0; haystack_index < counts_per_haystack.size(); ++haystack_index)
            matches_total += counts_per_haystack[haystack_index];
        return status_t::success_k;
    }

    /**
     *  @brief Locates every overlapping match, splitting one haystack across cores once it outgrows the L2.
     *  @param[out] matches_found Matches written, in ascending haystack order.
     */
    template <typename haystacks_type_, typename executor_type_>
    status_t find_all_overlapping_(haystacks_type_ const &haystacks, span<substrings_match_t> matches,
                                   size_t &matches_found, executor_type_ &executor, cpu_specs_t const &specs) noexcept {

        using haystack_t = typename haystacks_type_::value_type;

        matches_found = 0;
        if (haystacks.size() == 0) return status_t::success_k;
        size_t const cores_total = executor.threads_count();

        // Counting here rather than through `try_count` is what keeps every large haystack to two walks
        // instead of three: `try_count` may attribute a straddling match to the core it ends on, while the
        // scatter reserves each core's slot by where a match starts, so its split cannot be reused.
        size_t large_total = 0;
        for (size_t index = 0; index < haystacks.size(); ++index)
            if (is_large_(haystacks[index].size(), specs)) ++large_total;

        if (counts_per_haystack_.try_resize(haystacks.size()) != status_t::success_k ||
            offsets_per_haystack_.try_resize(haystacks.size()) != status_t::success_k ||
            counts_per_core_per_large_.try_resize(large_total * cores_total) != status_t::success_k)
            return status_t::bad_alloc_k;

        visit_dictionary([&](auto const &dict) noexcept {
            executor.for_n_dynamic(haystacks.size(), [&](size_t haystack_index) noexcept {
                haystack_t const &haystack = haystacks[haystack_index];
                if (is_large_(haystack.size(), specs)) return;
                counts_per_haystack_[haystack_index] = dict.count(to_bytes_view(haystack));
            });
        });

        size_t large_index = 0;
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
            haystack_t const &haystack = haystacks[haystack_index];
            if (!is_large_(haystack.size(), specs)) continue;
            auto const haystack_bytes = to_bytes_view(haystack);

            span<size_t> const counts_per_core {counts_per_core_per_large_.data() + large_index * cores_total,
                                                cores_total};
            count_matches_per_core_by_start_(haystack_bytes, executor, counts_per_core);
            size_t total = 0;
            for (size_t core_index = 0; core_index < cores_total; ++core_index)
                counts_per_core[core_index] = (total += counts_per_core[core_index]);
            counts_per_haystack_[haystack_index] = total;
            ++large_index;
        }

        status_t const prologue = prefix_and_check_(counts_per_haystack_, offsets_per_haystack_, matches.size(),
                                                    matches_found);
        if (prologue != status_t::success_k) return prologue;

        scatter_matches_of_small_(haystacks, counts_per_haystack_, offsets_per_haystack_, matches, executor, specs);

        large_index = 0;
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
            haystack_t const &haystack = haystacks[haystack_index];
            if (!is_large_(haystack.size(), specs)) continue;
            auto const haystack_bytes = to_bytes_view(haystack);

            span<size_t const> const counts_per_core {counts_per_core_per_large_.data() + large_index * cores_total,
                                                      cores_total};
            scatter_matches_of_one_large_(haystack_bytes, haystack_index, offsets_per_haystack_[haystack_index],
                                          counts_per_core, matches, executor);
            ++large_index;
        }

        return status_t::success_k;
    }

    /** @brief Counts a leftmost walk's matches; the recurrence keeps one core per haystack, whatever its size. */
    template <typename haystacks_type_, typename executor_type_>
    status_t count_all_leftmost_(haystacks_type_ const &haystacks, substrings_overlap_policy_t policy,
                                 span<size_t> counts_per_haystack, size_t &matches_total,
                                 executor_type_ &executor) noexcept {

        matches_total = 0;
        if (status_t const reserved = try_reserve_pending_starts_(executor.threads_count());
            reserved != status_t::success_k)
            return reserved;

        using prong_t = typename std::decay<executor_type_>::type::prong_t;
        visit_dictionary([&](auto const &dict) noexcept {
            executor.for_n_dynamic(counts_per_haystack.size(), [&](prong_t prong) noexcept {
                size_t total = 0;
                dict.find_leftmost(to_bytes_view(haystacks[prong.task]), pending_starts_of_(prong.thread), policy,
                                   [&](size_t, size_t, size_t) noexcept {
                                       ++total;
                                       return true;
                                   });
                counts_per_haystack[prong.task] = total;
            });
        });

        for (size_t haystack_index = 0; haystack_index < counts_per_haystack.size(); ++haystack_index)
            matches_total += counts_per_haystack[haystack_index];
        return status_t::success_k;
    }

    /** @brief Locates a leftmost walk's matches, counting into offsets so the scatter needs no atomics. */
    template <typename haystacks_type_, typename executor_type_>
    status_t find_all_leftmost_(haystacks_type_ const &haystacks, substrings_overlap_policy_t policy,
                                span<substrings_match_t> matches, size_t &matches_found,
                                executor_type_ &executor) noexcept {

        matches_found = 0;
        if (haystacks.size() == 0) return status_t::success_k;
        if (counts_per_haystack_.try_resize(haystacks.size()) != status_t::success_k ||
            offsets_per_haystack_.try_resize(haystacks.size()) != status_t::success_k)
            return status_t::bad_alloc_k;

        span<size_t> const counts_per_haystack {counts_per_haystack_.data(), haystacks.size()};
        size_t counted_total = 0;
        if (status_t const counted = count_all_leftmost_(haystacks, policy, counts_per_haystack, counted_total,
                                                         executor);
            counted != status_t::success_k)
            return counted;

        status_t const prologue = prefix_and_check_(counts_per_haystack_, offsets_per_haystack_, matches.size(),
                                                    matches_found);
        if (prologue != status_t::success_k) return prologue;

        using prong_t = typename std::decay<executor_type_>::type::prong_t;
        visit_dictionary([&](auto const &dict) noexcept {
            executor.for_n_dynamic(haystacks.size(), [&](prong_t prong) noexcept {
                size_t written = 0;
                size_t const base_offset = offsets_per_haystack_[prong.task];
                dict.find_leftmost(to_bytes_view(haystacks[prong.task]), pending_starts_of_(prong.thread), policy,
                                   [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                                       matches[base_offset + written] = {prong.task, needle_index, match_offset,
                                                                         match_length};
                                       ++written;
                                       return true;
                                   });
                sz_assert_(written == counts_per_haystack_[prong.task]);
            });
        });

        return status_t::success_k;
    }

    /** @brief Sizes one pending-start row per core, laid end to end so a core's slice needs no allocation. */
    status_t try_reserve_pending_starts_(size_t cores_total) noexcept {
        size_t const width = substrings_pending_starts_width(max_source_match_bytes());
        if (pending_starts_width_ == width && pending_starts_.size() == width * cores_total) return status_t::success_k;
        if (pending_starts_.try_resize(width * cores_total) != status_t::success_k) return status_t::bad_alloc_k;
        pending_starts_width_ = width;
        return status_t::success_k;
    }

    span<pending_start_t> pending_starts_of_(size_t core_index) noexcept {
        return {pending_starts_.data() + core_index * pending_starts_width_, pending_starts_width_};
    }

    /** @brief One core's share of a long haystack's cover: the source it owns, and what it rewrites into. */
    struct rewrite_share_t {
        /** @brief Ownership bounds, by match @b start: this core substitutes the matches starting in here.
         *         Both passes test against these, so they can never disagree about who owns a match. */
        size_t slice_begin {};
        size_t slice_end {};
        /** @brief First source byte this core writes; the end of whatever match crossed into its slice. */
        size_t source_begin {};
        /** @brief One past the last source byte, which a match crossing out of the slice can push forward. */
        size_t source_end {};
        size_t output_bytes {};
        /** @brief Where this core writes, relative to its haystack's own base in the output tape. */
        size_t output_offset {};
    };

    /**
     *  @brief Sizes one share record per core per large haystack, and one coverage row per core.
     *
     *  The shares are per haystack because the sizing pass settles them and the writing pass reads them back;
     *  the coverage row is per core only, being scratch one walk consumes before the next reuses it. The two
     *  therefore grow on different triggers, and testing them under one condition would skip a resize whenever
     *  only the other moved.
     *
     *  The coverage row is as wide as the restart search's window - four times the longest match - since that
     *  is the span a difference array has to mark before a position can be judged unspanned.
     */
    status_t try_reserve_rewrite_shares_(size_t cores_total, size_t large_total) noexcept {
        size_t const shares_wanted = cores_total * large_total;
        if (rewrite_shares_.size() != shares_wanted &&
            rewrite_shares_.try_resize(shares_wanted) != status_t::success_k)
            return status_t::bad_alloc_k;
        size_t const width = max_source_match_bytes() * 4 + 1;
        if (spanned_width_ != width || spanned_.size() != width * cores_total) {
            if (spanned_.try_resize(width * cores_total) != status_t::success_k) return status_t::bad_alloc_k;
            spanned_width_ = width;
        }
        return status_t::success_k;
    }

    /** @brief One large haystack's row of per-core shares, laid end to end like every other per-core scratch. */
    span<rewrite_share_t> shares_of_large_(size_t large_index, size_t cores_total) noexcept {
        return {rewrite_shares_.data() + large_index * cores_total, cores_total};
    }

    span<i32_t> spanned_of_(size_t core_index) noexcept {
        return {spanned_.data() + core_index * spanned_width_, spanned_width_};
    }

    /** @brief Sizes one frequency row per core, leaving every counter at zero. */
    status_t try_reserve_frequencies_(size_t cores_total) noexcept {
        needles_per_core_ = count_needles();
        size_t const wanted = cores_total * needles_per_core_;
        if (frequencies_.size() == wanted) return status_t::success_k;
        // `try_resize` leaves trivial types uninitialized, and the counters must start - and stay - zero.
        if (frequencies_.try_resize(wanted) != status_t::success_k ||
            touched_needles_.try_resize(wanted) != status_t::success_k)
            return status_t::bad_alloc_k;
        for (size_t index = 0; index < wanted; ++index) frequencies_[index] = 0;
        return status_t::success_k;
    }

  private:
    allocator_t alloc_ {};
    /** @brief The compiled automaton, at whichever state-id width `try_build` found it fits. */
    std::variant<narrow_dictionary_t, wide_dictionary_t> dict_;

    /** @brief Grow-only per-call scratch, reused across calls; concurrent calls on one engine are unsafe. */
    safe_vector<size_t, size_allocator_t> counts_per_core_ {};
    safe_vector<size_t, size_allocator_t> counts_per_haystack_ {};
    /** @brief One pending-start row per core, laid end to end so a core's slice needs no separate allocation. */
    safe_vector<pending_start_t, pending_start_allocator_t> pending_starts_ {};
    size_t pending_starts_width_ {};
    /** @brief One frequency row and one touched-needle row per core, laid out the same way. */
    safe_vector<u32_t, u32_allocator_t> frequencies_ {};
    safe_vector<u32_t, u32_allocator_t> touched_needles_ {};
    size_t needles_per_core_ {};
    /** @brief One share per core per large haystack, settled while sizing and read back while writing, so the
     *         cover is resolved once and both passes agree on who owns a match straddling a slice boundary. */
    safe_vector<rewrite_share_t, typename std::allocator_traits<allocator_t>::template rebind_alloc<rewrite_share_t>>
        rewrite_shares_ {};
    /** @brief One coverage row per core, the difference array the restart search marks matches into. */
    safe_vector<i32_t, typename std::allocator_traits<allocator_t>::template rebind_alloc<i32_t>> spanned_ {};
    size_t spanned_width_ {};
    safe_vector<size_t, size_allocator_t> offsets_per_haystack_ {};
    safe_vector<size_t, size_allocator_t> counts_per_core_per_large_ {};

    /** @brief Whether a haystack is big enough to deserve every core, rather than one core of its own. */
    static bool is_large_(size_t haystack_bytes, cpu_specs_t const &specs) noexcept {
        return haystack_bytes > specs.l2_bytes;
    }

    /**
     *  @brief Turns per-haystack @p counts into the exclusive prefix @p offsets, refusing a short output.
     *
     *  Nothing downstream bounds its writes against the output, so the whole call is refused up front
     *  rather than half-written.
     */
    static status_t prefix_and_check_(span<size_t const> counts, span<size_t> offsets, size_t matches_capacity,
                                      size_t &matches_total) noexcept {
        offsets[0] = 0;
        for (size_t index = 1; index < counts.size(); ++index) offsets[index] = offsets[index - 1] + counts[index - 1];
        matches_total = offsets[counts.size() - 1] + counts[counts.size() - 1];
        if (matches_total <= matches_capacity) return status_t::success_k;
        // The total survives the refusal, so a caller that brought no buffer still learns what to allocate.
        return status_t::unexpected_dimensions_k;
    }

    /**
     *  @brief Fills @p counts_per_core for one haystack, attributing a straddling match to the core it
     *         @b starts on - the rule the scatter reserves slots by, unlike `count_matches_per_core_`.
     */
    template <typename executor_type_>
    void count_matches_per_core_by_start_(span<byte_t const> haystack, executor_type_ &executor,
                                          span<size_t> counts_per_core) const noexcept {
        fu::indexed_split_t const optimal_split {haystack.size(), counts_per_core.size()};
        executor.for_threads([&](size_t core_index) noexcept {
            counts_per_core[core_index] = count_matches_in_one_part(
                haystack, snapped_subrange_(haystack, optimal_split, core_index));
        });
    }

    /**
     *  @brief The last position at or before @p limit that no match spans, so a cover can restart there.
     *
     *  A greedy cursor never runs past the largest end among the matches before it, so at a position no match
     *  spans the cursor is exactly that position, whatever came earlier. That is what lets one core resolve
     *  its own slice of a leftmost cover without knowing what the core before it decided.
     *
     *  Restarts are dense in real text, so a window of a few times the longest match almost always holds one;
     *  a slice that finds none falls back to the haystack's start, where the cursor is zero by definition.
     *
     *  @param[in] spanned Scratch as wide as the searched window, holding a difference array of coverage.
     */
    size_t restart_before_(span<byte_t const> haystack, size_t limit, span<i32_t> spanned) const noexcept {
        size_t const longest = max_source_match_bytes();
        size_t const search_begin = limit >= longest * 3 ? limit - longest * 3 : 0;
        if (search_begin == 0 || longest == 0) return 0;

        // A match spanning a judged position starts within `longest` of it, so judging only past that much of
        // the window guarantees every such match was seen; anything starting earlier is invisible here.
        size_t const judge_begin = search_begin + longest;
        size_t const walk_end = sz_min_of_two(limit + longest, haystack.size());
        size_t const window = walk_end - search_begin;
        if (window + 1 > spanned.size()) return 0;
        for (size_t index = 0; index <= window; ++index) spanned[index] = 0;

        // A position is spanned when it lies strictly inside a match, so each match marks `(start, end)` and
        // the running sum below reads zero exactly where nothing reaches across.
        visit_dictionary([&](auto const &dict) noexcept {
            dict.find({haystack.data() + search_begin, window},
                      [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                          sz_unused_(needle_index);
                          size_t const start = search_begin + match_offset + 1,
                                       end = search_begin + match_offset + match_length;
                          size_t const from = sz_max_of_two(start, search_begin);
                          size_t const to = sz_min_of_two(end, walk_end);
                          if (from < to) ++spanned[from - search_begin], --spanned[to - search_begin];
                          return true;
                      });
        });

        size_t restart = 0;
        i32_t reaching = 0;
        for (size_t position = search_begin; position <= limit && position < walk_end; ++position) {
            reaching += spanned[position - search_begin];
            if (position >= judge_begin && reaching == 0) restart = position;
        }
        return restart;
    }

    /**
     *  @brief What one core owns of a long haystack's rewrite, resolved without talking to its neighbours.
     *
     *  Ownership is by match @b start, so the core holding a match writes all of it even when it ends past
     *  the slice, and the next core starts after it. Both cores derive that boundary from the same cover, so
     *  they agree without a handshake.
     */
    template <typename replacements_type_>
    rewrite_share_t rewrite_share_of_core_(span<byte_t const> haystack, size_t slice_begin, size_t slice_end,
                                           substrings_overlap_policy_t policy, replacements_type_ const &replacements,
                                           span<pending_start_t> pending_starts, span<i32_t> spanned) noexcept {

        size_t const restart = restart_before_(haystack, slice_begin, spanned);
        rewrite_share_t share;
        share.slice_begin = slice_begin;
        share.slice_end = slice_end;
        share.source_begin = slice_begin;
        share.source_end = slice_end;
        size_t removed = 0, added = 0;

        visit_dictionary([&](auto const &dict) noexcept {
            dict.find_leftmost({haystack.data() + restart, haystack.size() - restart}, pending_starts, policy,
                               [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                                   size_t const start = restart + match_offset, end = start + match_length;
                                   // A match starting before this slice belongs to an earlier core, and pushes
                                   // this one's first byte past whatever it consumed.
                                   if (start < slice_begin) {
                                       share.source_begin = sz_max_of_two(share.source_begin, end);
                                       return true;
                                   }
                                   if (start >= slice_end) return false; // ? The next core's, and every one after.
                                   removed += match_length;
                                   added += to_bytes_view(replacements[needle_index]).size();
                                   share.source_end = sz_max_of_two(share.source_end, end);
                                   return true;
                               });
        });

        share.source_begin = sz_min_of_two(share.source_begin, share.source_end);
        share.output_bytes = (share.source_end - share.source_begin) - removed + added;
        return share;
    }

    /**
     *  @brief Settles every core's share of one long haystack and returns the bytes they rewrite to.
     *
     *  The shares survive into the writing pass, so the cover is resolved once rather than twice - which also
     *  keeps the two passes from disagreeing about who owns a match that straddles a slice boundary.
     */
    template <typename replacements_type_, typename executor_type_>
    size_t size_one_large_(span<byte_t const> haystack, substrings_overlap_policy_t policy,
                           replacements_type_ const &replacements, executor_type_ &executor,
                           span<rewrite_share_t> shares) noexcept {

        size_t const cores = executor.threads_count();
        fu::indexed_split_t const optimal_split {haystack.size(), cores};
        executor.for_threads([&](size_t core_index) noexcept {
            fu::indexed_range_t const slice = snapped_subrange_(haystack, optimal_split, core_index);
            shares[core_index] = rewrite_share_of_core_(haystack, slice.first, slice.first + slice.count, policy,
                                                        replacements, pending_starts_of_(core_index),
                                                        spanned_of_(core_index));
        });

        size_t total = 0;
        for (size_t core_index = 0; core_index < cores; ++core_index) {
            shares[core_index].output_offset = total;
            total += shares[core_index].output_bytes;
        }
        return total;
    }

    /** @brief Writes one core's share, spliced exactly as `substrings_rewrite` does for a whole haystack. */
    template <typename replacements_type_>
    void rewrite_share_(span<byte_t const> haystack, rewrite_share_t const &share, substrings_overlap_policy_t policy,
                        replacements_type_ const &replacements, span<pending_start_t> pending_starts,
                        span<i32_t> spanned, char *output) noexcept {

        size_t const restart = restart_before_(haystack, share.slice_begin, spanned);
        size_t copied_through = share.source_begin;
        visit_dictionary([&](auto const &dict) noexcept {
            dict.find_leftmost({haystack.data() + restart, haystack.size() - restart}, pending_starts, policy,
                               [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                                   size_t const start = restart + match_offset;
                                   // The same bounds the sizing pass used, so the two passes substitute exactly
                                   // the same matches - anything else would write bytes nobody accounted for.
                                   if (start < share.slice_begin) return true;
                                   if (start >= share.slice_end) return false;
                                   sz_copy((sz_ptr_t)output, (sz_cptr_t)(haystack.data() + copied_through),
                                           start - copied_through);
                                   output += start - copied_through;
                                   span<byte_t const> const replacement = to_bytes_view(replacements[needle_index]);
                                   sz_copy((sz_ptr_t)output, (sz_cptr_t)replacement.data(), replacement.size());
                                   output += replacement.size();
                                   copied_through = start + match_length;
                                   return true;
                               });
        });
        sz_copy((sz_ptr_t)output, (sz_cptr_t)(haystack.data() + copied_through), share.source_end - copied_through);
    }

    /**
     *  @brief Scores one haystack too long for a single core, tallying its slices in parallel.
     *
     *  Each core walks its own slice preceded by a warm-up of the longest match, so a match straddling a cut
     *  is still spelled, and claims it only when it @b ends inside the slice - the same one-owner rule
     *  `count_matches_per_core_` uses. Frequencies are integers, so the rows merge by plain addition and the
     *  merged row scores exactly as a single-core tally would.
     */
    template <typename executor_type_>
    f32_t score_one_large_(span<byte_t const> haystack, substrings_document_length_t length_source,
                           f32_t document_length, substrings_bm25_t parameters, span<f32_t const> needle_weights,
                           executor_type_ &executor) noexcept {

        size_t const cores = executor.threads_count();
        fu::indexed_split_t const optimal_split {haystack.size(), cores};
        size_t const longest = max_source_match_bytes();

        executor.for_threads([&](size_t core_index) noexcept {
            fu::indexed_range_t const slice = snapped_subrange_(haystack, optimal_split, core_index);
            size_t const slice_begin = slice.first, slice_end = slice.first + slice.count;
            size_t const walk_begin = slice_begin >= longest ? slice_begin - longest : 0;
            span<byte_t const> const walked {haystack.data() + walk_begin, slice_end - walk_begin};

            size_t const first = core_index * needles_per_core_;
            span<u32_t> const frequencies {frequencies_.data() + first, needles_per_core_};
            span<u32_t> const touched {touched_needles_.data() + first, needles_per_core_};
            size_t touched_count = 0;
            visit_dictionary([&](auto const &dict) noexcept {
                dict.find(walked, [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                    // A match belongs to the core its end falls in, so the warm-up never double-counts.
                    if (walk_begin + match_offset + match_length <= slice_begin) return true;
                    if (frequencies[needle_index]++ == 0) touched[touched_count++] = (u32_t)needle_index;
                    return true;
                });
            });
            counts_per_core_[core_index] = touched_count;
        });

        // Merge into the first core's row, in ascending needle order so the sum is the one the header names.
        span<u32_t> const merged {frequencies_.data(), needles_per_core_};
        span<u32_t> const merged_touched {touched_needles_.data(), needles_per_core_};
        size_t merged_count = counts_per_core_[0];
        for (size_t core_index = 1; core_index < cores; ++core_index) {
            size_t const first = core_index * needles_per_core_;
            for (size_t slot = 0; slot < counts_per_core_[core_index]; ++slot) {
                u32_t const needle_index = touched_needles_[first + slot];
                if (merged[needle_index] == 0) merged_touched[merged_count++] = needle_index;
                merged[needle_index] += frequencies_[first + needle_index];
                frequencies_[first + needle_index] = 0;
            }
        }

        if (length_source == substrings_document_length_t::haystack_bytes_k) document_length = (f32_t)haystack.size();
        return substrings_bm25_reduce(needle_weights, parameters, document_length, merged, merged_touched,
                                      merged_count);
    }

    /** @brief Writes every small haystack's matches, one core per haystack into disjoint output ranges. */
    template <typename haystacks_type_, typename executor_type_>
    void scatter_matches_of_small_(haystacks_type_ const &haystacks, span<size_t const> counts,
                                   span<size_t const> offsets, span<substrings_match_t> matches,
                                   executor_type_ &executor, cpu_specs_t const &specs) const noexcept {

        using haystack_t = typename haystacks_type_::value_type;
        sz_unused_(counts);

        visit_dictionary([&](auto const &dict) noexcept {
            executor.for_n_dynamic(offsets.size(), [&](size_t haystack_index) noexcept {
                haystack_t const &haystack = haystacks[haystack_index];
                auto const haystack_bytes = to_bytes_view(haystack);
                if (is_large_(haystack_bytes.size(), specs)) return;

                size_t matches_found = 0;
                dict.find(haystack_bytes, [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                    matches[offsets[haystack_index] + matches_found] = {haystack_index, needle_index, match_offset,
                                                                        match_length};
                    ++matches_found;
                    return true;
                });
                sz_assert_(counts[haystack_index] == matches_found);
            });
        });
    }

    /**
     *  @brief Writes one large haystack's matches, each core into the slot its own prefix sum reserves.
     *  @param[in] counts_per_core Inclusive prefix sums of each core's match count, attributed by start.
     */
    template <typename executor_type_>
    void scatter_matches_of_one_large_(span<byte_t const> haystack, size_t haystack_index, size_t base_offset,
                                       span<size_t const> counts_per_core, span<substrings_match_t> matches,
                                       executor_type_ &executor) const noexcept {

        fu::indexed_split_t const optimal_split {haystack.size(), counts_per_core.size()};
        executor.for_threads([&](size_t core_index) noexcept {
            size_t const count_matches_before_this_core = core_index ? counts_per_core[core_index - 1] : 0;
            size_t const count_matches_expected_on_this_core = counts_per_core[core_index] -
                                                               count_matches_before_this_core;

            fu::indexed_range_t const optimal_subrange = snapped_subrange_(haystack, optimal_split, core_index);
            byte_t const *optimal_begin = haystack.begin() + optimal_subrange.first;
            byte_t const *const optimal_end = optimal_begin + optimal_subrange.count;
            // An empty dictionary reports zero, where `optimal_end + 0 - 1` would step past the end.
            size_t const longest = max_source_match_bytes();
            size_t const overlap_bytes = longest > 0 ? longest - 1 : 0;
            byte_t const *const overlapping_end = sz_min_of_two(optimal_end + overlap_bytes, haystack.end());

            // Offsets arrive relative to the slice the dictionary was handed, not to the whole haystack.
            size_t const slice_offset_in_haystack = (size_t)(optimal_begin - haystack.begin());
            size_t const owned_bytes = optimal_subrange.count;
            size_t count_matches_found_on_this_core = 0;
            visit_dictionary([&](auto const &dict) noexcept {
                dict.find({optimal_begin, overlapping_end},
                          [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                              bool const belongs_to_this_core = match_offset < owned_bytes;
                              if (!belongs_to_this_core) return true;
                              matches[base_offset + count_matches_before_this_core +
                                      count_matches_found_on_this_core] = {haystack_index, needle_index,
                                                                           slice_offset_in_haystack + match_offset,
                                                                           match_length};
                              count_matches_found_on_this_core++;
                              return true;
                          });
            });
            sz_assert_(count_matches_found_on_this_core == count_matches_expected_on_this_core);
        });
    }

    /**
     *  @brief One core's slice, with both ends pulled back to codepoint starts whenever the walk folds.
     *
     *  Attribution alone already makes an unsnapped cut safe: a match starts on a lead byte, so no match can
     *  start inside the codepoint a cut lands in, and that codepoint's own matches start before the cut and
     *  belong to the previous core, which reaches them through its overlap. Snapping keeps that argument from
     *  having to be made - the walk begins on a real codepoint - and saves the resynchronization it would
     *  otherwise spend treating a continuation byte as malformed. Both ends go through the same snap and one
     *  core's end is the next one's start, so the slices stay an exact partition however the boundaries move.
     */
    fu::indexed_range_t snapped_subrange_(span<byte_t const> haystack, fu::indexed_split_t const &split,
                                          size_t core_index) const noexcept {
        fu::indexed_range_t subrange = split[core_index];
        if (case_sensitivity() != substrings_uncased_k) return subrange;
        size_t const begin = sz_utf8_rune_start_at_((cptr_t)haystack.data(), haystack.size(), subrange.first);
        size_t const end = sz_utf8_rune_start_at_((cptr_t)haystack.data(), haystack.size(),
                                                  subrange.first + subrange.count);
        subrange.first = begin;
        subrange.count = end - begin; // ? Zero when a whole slice fell inside one codepoint
        return subrange;
    }

    /**
     *  @brief Fills @p counts_per_core for one haystack, picking the cheaper of the two counting strategies.
     *
     *  The two strategies attribute a straddling match to different cores - `count_matches_in_one_part` to
     *  the core the match starts on, `count_short_matches_in_one_part` to the core it ends on - so only a
     *  caller that sums across cores may choose freely.
     */
    template <typename executor_type_>
    void count_matches_per_core_(span<byte_t const> haystack, executor_type_ &executor,
                                 span<size_t> counts_per_core) const noexcept {
        fu::indexed_split_t const optimal_split {haystack.size(), counts_per_core.size()};
        // The short-match strategy steps raw haystack bytes through the automaton, which only spells a match
        // when the dictionary is byte-exact; a folded walk has to go through `find`.
        bool const longest_match_fits_on_one_core = optimal_split.smallest_size() >= max_source_match_bytes() &&
                                                    case_sensitivity() == substrings_cased_k;

        if (!longest_match_fits_on_one_core)
            executor.for_threads([&](size_t core_index) noexcept {
                counts_per_core[core_index] = count_matches_in_one_part(
                    haystack, snapped_subrange_(haystack, optimal_split, core_index));
            });
        else
            executor.for_threads([&](size_t core_index) noexcept {
                size_t matches_in_prefix = 0;
                size_t const matches_in_part = count_short_matches_in_one_part(haystack, optimal_split[core_index],
                                                                               matches_in_prefix);
                counts_per_core[core_index] = matches_in_part - non_zero_if<size_t>(matches_in_prefix, core_index > 0);
            });
    }

    /**
     *  @brief  Helper method implementing the core logic of the parallel `try_count` and part of `try_find`.
     *  @return Number of matches that @b begin in this core's slice and may end in another core's slice.
     */
    size_t count_matches_in_one_part(span<byte_t const> haystack,
                                     fu::indexed_range_t const optimal_subrange) const noexcept {

        size_t const max_source_match_bytes = sz_min_of_two(this->max_source_match_bytes(), haystack.size());

        byte_t const *optimal_begin = haystack.begin() + optimal_subrange.first;
        byte_t const *const optimal_end = optimal_begin + optimal_subrange.count;

        size_t const count_matches_non_overlapping =
            visit_dictionary([&](auto const &dict) noexcept { return dict.count({optimal_begin, optimal_end}); });

        byte_t const *overlapping_start;
        byte_t const *overlapping_end;
        if (optimal_begin + max_source_match_bytes >= optimal_end) {
            overlapping_start = optimal_begin;
            overlapping_end = sz_min_of_two(optimal_end + max_source_match_bytes, haystack.end());
        }
        else {
            overlapping_start = sz_max_of_two(optimal_end - max_source_match_bytes + 1, optimal_begin);
            overlapping_end = sz_min_of_two(optimal_end + max_source_match_bytes - 1, haystack.end());
        }

        // Both branches place `overlapping_start` at or after `optimal_begin`, so offsets relative to it need
        // no lower-bound test.
        size_t const slice_end_offset = (size_t)(optimal_end - overlapping_start);
        size_t count_matches_overlapping = 0;
        visit_dictionary([&](auto const &dict) noexcept {
            dict.find({overlapping_start, overlapping_end},
                      [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                          sz_unused_(needle_index);
                          bool const belongs_to_this_core =                   //
                              match_offset < slice_end_offset &&              // ? Starts before this slice ends.
                              match_offset + match_length > slice_end_offset; // ? Ends beyond this slice.
                          count_matches_overlapping += belongs_to_this_core;
                          return true;
                      });
        });

        return count_matches_non_overlapping + count_matches_overlapping;
    }

    /**
     *  @brief  More optimized alternative to `count_matches_in_one_part`, assuming the longest match fits
     *          within a single core's slice, so a match can only spill into 2 core regions at most.
     *  @param[out] matches_in_prefix Matches ending within the first `max_source_match_bytes` of this core's slice,
     *              which the preceding core has already counted as its own overlapping tail.
     *  @return Total matches ending anywhere in this core's slice or its overlapping tail.
     */
    size_t count_short_matches_in_one_part(span<byte_t const> haystack, fu::indexed_range_t const optimal_subrange,
                                           size_t &matches_in_prefix) const noexcept {

        // One dispatch per core per haystack, outside the walk, so the transition chain below stays monomorphic.
        return std::visit(
            [&](auto const &dict) noexcept -> size_t {
                using walked_state_id_t = typename std::decay<decltype(dict)>::type::state_id_t;
                auto const automaton = dict.view();
                sz_assert_(automaton.case_sensitivity == substrings_cased_k &&
                           "A folded walk cannot step raw haystack bytes; the uncased path counts through `find`");
                size_t const max_source_match_bytes = (size_t)dict.max_source_match_bytes();
                byte_t const *optimal_begin = haystack.begin() + optimal_subrange.first;
                byte_t const *const optimal_end = optimal_begin + optimal_subrange.count;
                byte_t const *const prefix_end = sz_min_of_two(optimal_begin + max_source_match_bytes, haystack.end());
                byte_t const *const overlapping_end =
                    sz_min_of_two(optimal_end + max_source_match_bytes, haystack.end());

                size_t matches_in_part = 0;
                matches_in_prefix = 0;
                walked_state_id_t current_state = automaton.root;
                // The prefix window spans at most `max_source_match_bytes` bytes and is the only region needing the
                // per-byte attribution test, so it walks scalar; `prefix_end` never exceeds `overlapping_end`,
                // both being clamped by the same haystack end.
                for (; optimal_begin != prefix_end; ++optimal_begin) {
                    walked_state_id_t const output_count =
                        aho_corasick_step_counting(automaton, current_state, *optimal_begin);
                    matches_in_part += output_count;
                    matches_in_prefix += output_count;
                }
                // One 4-byte load feeds four transitions - the state chain stays strictly serial, and `sz_u32_load`
                // absorbs misalignment itself, so only a tail loop remains.
                for (; optimal_begin + 4 <= overlapping_end; optimal_begin += 4) {
                    sz_u32_vec_t const quad = sz_u32_load((sz_cptr_t)optimal_begin);
                    matches_in_part += aho_corasick_step_counting(automaton, current_state, quad.u8s[0]);
                    matches_in_part += aho_corasick_step_counting(automaton, current_state, quad.u8s[1]);
                    matches_in_part += aho_corasick_step_counting(automaton, current_state, quad.u8s[2]);
                    matches_in_part += aho_corasick_step_counting(automaton, current_state, quad.u8s[3]);
                }
                for (; optimal_begin != overlapping_end; ++optimal_begin)
                    matches_in_part += aho_corasick_step_counting(automaton, current_state, *optimal_begin);

                return matches_in_part;
            },
            dict_);
    }
};

using substrings_serial_t = substrings<std::allocator<char>, sz_cap_serial_k>;
using substrings_parallel_t = substrings<std::allocator<char>, sz_caps_sp_k>;

#pragma endregion Parallel Backend

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SUBSTRINGS_SERIAL_HPP_
