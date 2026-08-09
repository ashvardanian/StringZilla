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
 *  Case-folding works in two layers. The @b spelling automaton keys states by `(folded position, byte
 *  delta)`, so every spelling of a fold reconverges onto one state instead of enumerating a path each. That
 *  makes it a DAG, on which Aho-Corasick failure links are not single-valued - a state two spellings reach
 *  at different byte widths needs two failure continuations. The published @b walking automaton therefore
 *  splits each spelling state by its `(spelling state, failure state)` pair, the coarsest split that keeps
 *  failure links single-valued. Pathological doubling needles - "ssssss" case-insensitively - have a
 *  provably exponential automaton and are declined with `overflow_risk_k` once the walking-state count would
 *  overflow the id type, rather than built wrong.
 */
#ifndef STRINGZILLAS_SUBSTRINGS_SERIAL_HPP_
#define STRINGZILLAS_SUBSTRINGS_SERIAL_HPP_

#include "stringzilla/types.hpp"                  // `status_t::status_t`
#include "stringzilla/utf8_runes/serial.h"        // `sz_rune_decode`, `sz_rune_encode`
#include "stringzilla/utf8_uncased_fold/serial.h" // `sz_unicode_fold_codepoint_`
#include "stringzillas/substrings/tables.h"       // `szs_fold_preimage_narrow_images_`
#include "stringzillas/types.hpp"                 // `dummy_executor_t`

#include <forkunion/types.hpp> // `indexed_split_t` - the balanced range split every executor uses

#include <limits>      // `std::numeric_limits` for numeric types
#include <memory>      // `std::allocator_traits` to re-bind the allocator
#include <type_traits> // `std::enable_if_t` for meta-programming

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

/**
 *  @brief  One reported match: which needle matched, and how many haystack bytes the match actually spans.
 *
 *  Under case folding, a needle's @b own byte length is not the length of every match: needle "k" matches
 *  both the 1-byte "k" and the 3-byte Kelvin sign U+212A, so the span has to be carried per match, not
 *  looked up from the needle.
 */
template <typename state_id_type_>
struct substrings_output {
    using state_id_t = state_id_type_;

    state_id_t needle_index {};
    /** @brief Haystack bytes this match spans; a walk traverses one edge per byte, so it fits a state id. */
    state_id_t match_bytes {};
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
};

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

    /** @brief Longest match any needle can produce, in bytes. Sets the overlap when slicing a haystack. */
    state_id_t max_match_bytes {};

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

#pragma region Engine

/**
 *  @brief  Multi-pattern search engine: one compiled dictionary applied to many haystacks in a single pass.
 *
 *  Declared without a body so every backend supplies its own specialization, guarded on @p capability_ -
 *  the shape `levenshtein_distances` and the other StringZillas engines already use.
 */
template <typename state_id_type_ = u32_t, typename allocator_type_ = dummy_alloc_t,
          sz_capability_t capability_ = sz_cap_serial_k, typename enable_ = void>
struct substrings;

#pragma endregion Engine

#pragma region Dictionary

#pragma region Fold Preimage Reverse Index

/**
 *  @brief Scratch capacity for one preimage lookup: the identity codepoint plus the widest fan-out.
 *
 *  A window wider than one rune contributes no identity, so the narrow case bounds both.
 */
static constexpr size_t fold_preimage_max_sources_k = 1 + szs_fold_preimage_max_fanout_k;

/** @brief Source codepoints whose full fold is the single rune @p image, empty when it folds to itself. */
inline span<rune_t const> fold_preimage_of_rune(rune_t image) noexcept {
    size_t low = 0, high = szs_fold_preimage_narrow_count_k;
    while (low < high) {
        size_t const middle = low + (high - low) / 2;
        if (szs_fold_preimage_narrow_images_[middle] < image) low = middle + 1;
        else high = middle;
    }
    if (low == szs_fold_preimage_narrow_count_k || szs_fold_preimage_narrow_images_[low] != image) return {};
    u32_t const begin = szs_fold_preimage_narrow_offsets_[low];
    return {szs_fold_preimage_narrow_sources_ + begin, szs_fold_preimage_narrow_offsets_[low + 1] - begin};
}

/**
 *  @brief First multi-rune image whose leading rune is at least @p leading.
 *
 *  Only 73 images span more than one rune, so this same bound answers both "can this rune begin a wide
 *  image at all" - the answer is no when the entry it lands on leads with a different rune - and "where
 *  does its group start", which is why no separate filter is needed.
 */
inline size_t fold_preimage_wide_lower_bound(rune_t leading) noexcept {
    size_t low = 0, high = szs_fold_preimage_wide_count_k;
    while (low < high) {
        size_t const middle = low + (high - low) / 2;
        if (szs_fold_preimage_wide_images_[middle][0] < leading) low = middle + 1;
        else high = middle;
    }
    return low;
}

/** @brief Source codepoints whose full fold is exactly the runes of @p image, for `image.size() > 1`. */
inline span<rune_t const> fold_preimage_of_runes(span<rune_t const> image) noexcept {
    rune_t const second = image[1];
    rune_t const third = image.size() >= 3 ? image[2] : 0;
    for (size_t index = fold_preimage_wide_lower_bound(image[0]);
         index < szs_fold_preimage_wide_count_k && szs_fold_preimage_wide_images_[index][0] == image[0]; ++index) {
        if (szs_fold_preimage_wide_images_[index][1] != second) continue;
        if (szs_fold_preimage_wide_images_[index][2] != third) continue;
        u32_t const begin = szs_fold_preimage_wide_offsets_[index];
        return {szs_fold_preimage_wide_sources_ + begin, szs_fold_preimage_wide_offsets_[index + 1] - begin};
    }
    return {};
}

#pragma endregion Fold Preimage Reverse Index

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

    /**
     *  @brief How a chain slot was reached, which is part of its identity rather than a passing property.
     *
     *  An atomic arrival consumes its source bytes as one unit - sharp S folding to "ss" - so no haystack
     *  position exists partway through it. Merging it with a stepped arrival at the same delta would hand
     *  it a failure link computed for a resuming position it does not have.
     */
    enum class rune_arrival_t : u8_t {
        stepped_k = 0,
        atomic_k = 1,
    };

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
        state_id_t match_bytes;
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
     *  @brief One published state: a spelling state paired with the failure state its arrival demands.
     *
     *  Ids are assigned in discovery order, so a depth band is a contiguous id range and every failure id is
     *  smaller than its own state's id.
     */
    struct walking_state_t {
        /** @brief The spelling state this one spells: supplies its row bytes, own matches, and out-degree. */
        state_id_t spelling_state = 0;
        /** @brief The failure walking state; half of this state's identity, stored at creation. */
        state_id_t failure_state = 0;
        /** @brief Published double-array slot for this state; the old per-state `final_id`. */
        state_id_t published_id = invalid_state_k;
        /** @brief Into `outputs_`: own matches followed by the failure state's whole run. */
        size_t total_offset = 0;
        state_id_t total_count = 0;
    };

    /** @brief One reachable `(byte delta, raw state)` pair at a given folded position of one needle's chain. */
    struct chain_slot_t {
        /**
         *  @brief Source bytes consumed minus the folded spelling's own canonical bytes.
         *
         *  Its consumer is `match_bytes`, published as `state_id_t`, so a wider accumulator would only
         *  represent matches the view cannot describe.
         */
        i32_t delta;
        state_id_t state;
        /** @brief Next slot at the same folded position; the position itself holds the head. */
        size_t next;
        rune_arrival_t arrival;
    };

    /** @brief One folded rune of the needle under construction, plus the slots reachable at that position. */
    struct folded_position_t {
        rune_t rune;
        /** @brief Canonical encoded length, which is why no prefix-sum array is needed. */
        rune_length_t utf8_length;
        size_t chain_head = SZ_SIZE_MAX;
    };

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using state_id_allocator_t = typename allocator_traits_t::template rebind_alloc<state_id_t>;
    using output_allocator_t = typename allocator_traits_t::template rebind_alloc<output_t>;
    using edge_allocator_t = typename allocator_traits_t::template rebind_alloc<trie_edge_t>;
    using pending_output_allocator_t = typename allocator_traits_t::template rebind_alloc<pending_output_t>;
    using output_run_allocator_t = typename allocator_traits_t::template rebind_alloc<output_run_t>;
    using walking_state_allocator_t = typename allocator_traits_t::template rebind_alloc<walking_state_t>;
    using csr_edge_allocator_t = typename allocator_traits_t::template rebind_alloc<csr_edge_t>;
    using folded_position_allocator_t = typename allocator_traits_t::template rebind_alloc<folded_position_t>;
    using chain_slot_allocator_t = typename allocator_traits_t::template rebind_alloc<chain_slot_t>;
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
    /** @brief The needle under construction, one entry per folded rune; uncased mode only. */
    safe_vector<folded_position_t, folded_position_allocator_t> folded_positions_;
    /** @brief Pooled chain slots for that needle, threaded per position by `folded_position_t::chain_head`. */
    safe_vector<chain_slot_t, chain_slot_allocator_t> chain_slots_;

    /** @brief One block carved by `build_layout_` into the buffers whose size is fixed once insertion ends. */
    safe_vector<std::byte, byte_allocator_t> build_scratch_;

    /** @brief Every walking state, in discovery order; index 0 is the root paired with itself. Sized by the
     *         walking count, which the splitting pass only learns as it runs, so like `old_of_final_` it
     *         lives outside the fixed layout. */
    safe_vector<walking_state_t, walking_state_allocator_t> walking_states_;
    /** @brief Walking-state CSR row starts, `walking_states_.size() + 1` entries. */
    safe_vector<size_t, offset_allocator_t> walking_offsets_;
    /** @brief Walking-state CSR rows: one `(child, byte)` per edge, same shape as the spelling CSR so the
     *         packing and hot-row phases read it through the same code. */
    safe_vector<csr_edge_t, csr_edge_allocator_t> walking_rows_;
    /** @brief Walking states in depth-band, out-degree-descending order: the input to the published numbering. */
    safe_vector<state_id_t, state_id_allocator_t> walking_order_;
    /** @brief The root's goto-completed row, dense over the alphabet, so a failure chase ends in one lookup
     *         rather than a scan of the root's whole edge list. */
    safe_vector<state_id_t, state_id_allocator_t> walking_root_row_;
    /** @brief Open-addressed `(spelling, failure) -> walking` memo, so a repeated pair reuses its state in
     *         one probe; `SZ_SIZE_MAX` marks an empty slot, exactly as `edge_index_` does for edges. */
    safe_vector<size_t, offset_allocator_t> walking_index_;

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
    state_id_t max_match_bytes_ = 0;
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
    status_t add_output_(state_id_t state, state_id_t needle_index, size_t match_bytes) noexcept {
        // Caller-controlled length, so the ceiling is a real status in every build rather than an assert.
        if (match_bytes > static_cast<size_t>(invalid_state_k)) return status_t::overflow_risk_k;
        output_run_t &run = output_runs_[state];
        state_id_t const match_bytes_narrow = static_cast<state_id_t>(match_bytes);
        if (own_outputs_.try_push_back(pending_output_t {needle_index, match_bytes_narrow, run.own_head}) !=
            status_t::success_k)
            return status_t::bad_alloc_k;
        run.own_head = (own_outputs_.size() - 1);
        ++run.own_count;
        max_match_bytes_ = sz_max_of_two(max_match_bytes_, match_bytes_narrow);
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

    /**
     *  @brief Walks the interior bytes of `encode_utf8(codepoint)` from @p source_state, stopping one byte
     *         short: @p parent_of_last_byte owns the codepoint's trailing byte and @p last_byte is that byte.
     *
     *  Stopping short lets the caller choose the final target before any state ID is spent.
     */
    status_t walk_byte_path_(state_id_t source_state, rune_t codepoint, state_id_t &parent_of_last_byte,
                             u8_t &last_byte) noexcept {
        u8_t encoded[4];
        rune_length_t const length = sz_rune_encode(codepoint, encoded);

        state_id_t current_state = source_state;
        for (size_t index = 0; index + 1 < (size_t)length; ++index) {
            status_t const status = follow_or_create_(current_state, encoded[index], current_state);
            if (status != status_t::success_k) return status;
        }

        parent_of_last_byte = current_state;
        last_byte = encoded[length - 1];
        return status_t::success_k;
    }

    /** @brief Decodes and fully folds @p needle into `folded_positions_`, one entry per folded rune. */
    status_t fold_needle_(span<byte_t const> needle) noexcept {
        folded_positions_.clear();
        byte_t const *cursor = needle.begin();
        byte_t const *const needle_end = needle.end();
        while (cursor != needle_end) {
            rune_t rune;
            rune_length_t const consumed = sz_rune_decode(reinterpret_cast<cptr_t>(cursor),
                                                          reinterpret_cast<cptr_t>(needle_end), &rune);
            if (consumed == sz_rune_invalid_k) return status_t::invalid_utf8_k;
            rune_t folded[3];
            size_t const folded_count = sz_unicode_fold_codepoint_(rune, folded);
            for (size_t index = 0; index < folded_count; ++index) {
                folded_position_t entry;
                entry.rune = folded[index];
                entry.utf8_length = utf8_length_of_rune_(folded[index]);
                if (folded_positions_.try_push_back(entry) != status_t::success_k) return status_t::bad_alloc_k;
            }
            cursor += consumed;
        }
        return status_t::success_k;
    }

    /** @brief The slot already reachable at @p position with this key, or the invalid sentinel. */
    size_t find_slot_(size_t position, i32_t delta, rune_arrival_t arrival) const noexcept {
        for (size_t walk = folded_positions_[position].chain_head; walk != SZ_SIZE_MAX;
             walk = chain_slots_[walk].next) {
            chain_slot_t const &slot = chain_slots_[walk];
            if (slot.delta == delta && slot.arrival == arrival) return walk;
        }
        return SZ_SIZE_MAX;
    }

    /** @brief Gathers the preimage codepoints of the @p window runes starting at @p position. */
    size_t collect_preimages_(size_t position, size_t window,
                              safe_array<rune_t, fold_preimage_max_sources_k> &sources) const noexcept {
        rune_t const leading = folded_positions_[position].rune;
        size_t count = 0;
        span<rune_t const> run;
        if (window == 1) {
            // Every codepoint folds to itself unless the table says otherwise, and the table lists only
            // non-identity folds, so the identity spelling is added here rather than looked up.
            sources[count++] = leading;
            run = fold_preimage_of_rune(leading);
        }
        else {
            // Zeroed, so a two-rune window's unused third lane matches the trailing zero the wide table
            // stores for two-rune images rather than holding whatever was on the stack.
            rune_t window_runes[3] = {};
            for (size_t within = 0; within < window; ++within)
                window_runes[within] = folded_positions_[position + within].rune;
            run = fold_preimage_of_runes({window_runes, window});
        }
        for (size_t within = 0; within < run.size() && count < fold_preimage_max_sources_k; ++within)
            sources[count++] = run[within];
        return count;
    }

    /**
     *  @brief Resolves one `(source slot, window)` pair: every codepoint whose fold is the window's runes
     *         gets a byte path from the source slot's state, all converging on one successor slot.
     */
    status_t insert_window_(size_t position, size_t window, size_t source) noexcept {
        size_t canonical_span = 0;
        for (size_t within = 0; within < window; ++within)
            canonical_span += (size_t)folded_positions_[position + within].utf8_length;

        safe_array<rune_t, fold_preimage_max_sources_k> preimage_sources;
        size_t const preimage_count = collect_preimages_(position, window, preimage_sources);

        // A window wider than one rune consumes its source codepoint atomically, and that taints every slot
        // reached through it from here on, even if a later single-rune hop would land back on a delta a
        // purely stepped chain also reaches.
        rune_arrival_t const arrival = chain_slots_[source].arrival == rune_arrival_t::atomic_k || window > 1
                                           ? rune_arrival_t::atomic_k
                                           : rune_arrival_t::stepped_k;

        for (size_t index = 0; index < preimage_count; ++index) {
            rune_t const source_codepoint = preimage_sources[index];
            size_t const encoded_length = (size_t)utf8_length_of_rune_(source_codepoint);
            i32_t const child_delta = chain_slots_[source].delta + (i32_t)encoded_length - (i32_t)canonical_span;

            size_t const matching_slot = find_slot_(position + window, child_delta, arrival);
            bool const slot_exists = matching_slot != SZ_SIZE_MAX;

            // Probe the byte path before committing to a target: an earlier needle sharing this exact
            // `(source state, source codepoint)` pair may already own the transition, in which case its
            // target is reused outright and no state ID is spent on this step.
            state_id_t parent_of_last_byte;
            u8_t last_byte;
            status_t status = walk_byte_path_(chain_slots_[source].state, source_codepoint, parent_of_last_byte,
                                              last_byte);
            if (status != status_t::success_k) return status;
            size_t const existing_edge = find_edge_(parent_of_last_byte, last_byte);

            state_id_t actual_target;
            if (existing_edge != SZ_SIZE_MAX) { actual_target = edges_[existing_edge].child; }
            else {
                if (slot_exists) { actual_target = chain_slots_[matching_slot].state; }
                else {
                    status = allocate_raw_state_(actual_target);
                    if (status != status_t::success_k) return status;
                }
                status = add_edge_(parent_of_last_byte, last_byte, actual_target);
                if (status != status_t::success_k) return status;
            }

            if (slot_exists) {
                // The splitting pass relies on one target per `(state, byte)`, so the repoint is checked.
                sz_assert_(chain_slots_[matching_slot].state == actual_target &&
                           "Two preimages of one window from one state must share their target");
                chain_slots_[matching_slot].state = actual_target;
            }
            else {
                folded_position_t &target_position = folded_positions_[position + window];
                chain_slot_t const fresh {child_delta, actual_target, target_position.chain_head, arrival};
                if (chain_slots_.try_push_back(fresh) != status_t::success_k) return status_t::bad_alloc_k;
                target_position.chain_head = (chain_slots_.size() - 1);
            }
        }
        return status_t::success_k;
    }

    /**
     *  @brief Case-folded trie insertion: one state per `(folded position, byte delta, arrival)` triple,
     *         with every codepoint whose fold matches a 1-3 rune window reconverging onto the shared
     *         successor for that window - except across the split `rune_arrival_t` documents, which stays
     *         two states even at matching deltas.
     *
     *  A step from position `p` only ever writes positions `p + 1 .. p + 3`, so a position's own chain is
     *  already complete when the loop reaches it and can be walked in place rather than snapshotted.
     */
    status_t try_insert_uncased_(span<byte_t const> needle, state_id_t needle_index) noexcept {
        status_t status = ensure_root_();
        if (status != status_t::success_k) return status;

        // Reject malformed UTF-8 outright: a raw continuation byte could otherwise pick up a failure link
        // mid-codepoint and report a spurious match.
        status = fold_needle_(needle);
        if (status != status_t::success_k) return status;

        size_t const folded_length = folded_positions_.size();
        if (folded_length == 0) return status_t::success_k;

        // One extra position holds the terminal slots, and every chain starts empty but the root's.
        folded_position_t terminal_position {};
        if (folded_positions_.try_push_back(terminal_position) != status_t::success_k) return status_t::bad_alloc_k;
        chain_slots_.clear();
        chain_slot_t const root_slot {0, 0, SZ_SIZE_MAX, rune_arrival_t::stepped_k};
        if (chain_slots_.try_push_back(root_slot) != status_t::success_k) return status_t::bad_alloc_k;
        folded_positions_[0].chain_head = 0;

        size_t canonical_prefix = 0; // ? Bytes the folded spelling itself spends before `position`.
        for (size_t position = 0; position < folded_length; ++position) {
            size_t const reachable_window = sz_min_of_two((size_t)3, folded_length - position);
            // Only 73 of the 1524 fold images span more than one rune, so a leading rune that cannot begin
            // one skips both wider windows outright.
            size_t const wide_bound = fold_preimage_wide_lower_bound(folded_positions_[position].rune);
            bool const may_lead_wide = wide_bound < szs_fold_preimage_wide_count_k &&
                                       szs_fold_preimage_wide_images_[wide_bound][0] ==
                                           folded_positions_[position].rune;
            size_t const max_window = may_lead_wide ? reachable_window : 1;

            for (size_t source = folded_positions_[position].chain_head; source != SZ_SIZE_MAX;
                 source = chain_slots_[source].next)
                for (size_t window = 1; window <= max_window; ++window) {
                    status = insert_window_(position, window, source);
                    if (status != status_t::success_k) return status;
                }

            canonical_prefix += (size_t)folded_positions_[position].utf8_length;
        }

        for (size_t walk = folded_positions_[folded_length].chain_head; walk != SZ_SIZE_MAX;
             walk = chain_slots_[walk].next) {
            chain_slot_t const &slot = chain_slots_[walk];
            status = add_output_(slot.state, needle_index, canonical_prefix + (size_t)slot.delta);
            if (status != status_t::success_k) return status;
        }
        return status_t::success_k;
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
     *  @brief Reorders one depth band of `walking_order_` by out-degree descending, so shallow high-fan-out
     *         states - the ones text keeps returning to - land in the hot tier regardless of dictionary
     *         content. A counting sort over a bounded degree.
     *  @note A band is the contiguous walking-id range `[band_first, band_last)`, so the states being sorted
     *        are exactly those ids; the scatter reads no `walking_order_` entry and writes it in place.
     */
    void order_band_by_out_degree_(size_t band_first, size_t band_last) noexcept {
        if (band_last - band_first < 2) return;

        // Positions into `walking_order_`, riding the state id like every other ordinal.
        state_id_t histogram[alphabet_size_k + 1] = {};
        for (size_t state = band_first; state < band_last; ++state)
            ++histogram[walking_offsets_[state + 1] - walking_offsets_[state]];
        // Suffix-summed, so the highest degree claims the lowest positions in the band.
        state_id_t running = state_id_of_(band_first);
        for (size_t degree = alphabet_size_k + 1; degree-- > 0;) {
            state_id_t const count = histogram[degree];
            histogram[degree] = running;
            running += count;
        }
        for (size_t state = band_first; state < band_last; ++state)
            walking_order_[histogram[walking_offsets_[state + 1] - walking_offsets_[state]]++] = state_id_of_(state);
    }

    /** @brief Fills the dense root row from the root's walking edges, defaulting every other byte to the
     *         self-looping root, so a failure chase that falls all the way back resolves in one lookup. */
    status_t fill_root_row_() noexcept {
        if (walking_root_row_.try_resize(alphabet_size_k) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t byte = 0; byte < alphabet_size_k; ++byte) walking_root_row_[byte] = 0;
        for (size_t edge = walking_offsets_[0]; edge < walking_offsets_[1]; ++edge)
            walking_root_row_[walking_rows_[edge].byte] = walking_rows_[edge].child;
        return status_t::success_k;
    }

    /** @brief Child of @p walking_state on @p byte among its literal edges, or `invalid_state_k` if none. */
    state_id_t find_walking_edge_(state_id_t walking_state, u8_t byte) const noexcept {
        for (size_t edge = walking_offsets_[walking_state]; edge < walking_offsets_[walking_state + 1]; ++edge)
            if (walking_rows_[edge].byte == byte) return walking_rows_[edge].child;
        return invalid_state_k;
    }

    /** @brief Goto-completed target for @p walking_state on @p byte; the root answers from its dense row. */
    state_id_t chase_walking_(state_id_t walking_state, u8_t byte) const noexcept {
        for (state_id_t current = walking_state;;) {
            if (current == 0) return walking_root_row_[byte];
            state_id_t const child = find_walking_edge_(current, byte);
            if (child != invalid_state_k) return child;
            current = walking_states_[current].failure_state;
        }
    }

    /** @brief Scrambles a `(spelling, failure)` pair into a starting probe; the memo stores no key of its own. */
    static size_t walking_probe_(state_id_t spelling_state, state_id_t failure_state) noexcept {
        u64_t const mixed = (((u64_t)spelling_state << 32) ^ (u64_t)failure_state) * 0x9E3779B97F4A7C15ull;
        return (size_t)(mixed >> 32);
    }

    /** @brief Rebuilds `walking_index_` at @p new_capacity slots, reinserting every walking state's pair. */
    status_t rehash_walking_index_(size_t new_capacity) noexcept {
        if (walking_index_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t slot = 0; slot < new_capacity; ++slot) walking_index_[slot] = SZ_SIZE_MAX;
        size_t const mask = new_capacity - 1;
        for (size_t walking = 0; walking < walking_states_.size(); ++walking) {
            size_t slot =
                walking_probe_(walking_states_[walking].spelling_state, walking_states_[walking].failure_state) & mask;
            while (walking_index_[slot] != SZ_SIZE_MAX) slot = (slot + 1) & mask;
            walking_index_[slot] = walking;
        }
        return status_t::success_k;
    }

    /** @brief Appends a fresh walking state, seeding its identity position in `walking_order_` so a one-state
     *         band sorts to itself. */
    status_t append_walking_state_(state_id_t spelling_state, state_id_t failure_state,
                                   state_id_t &walking_state) noexcept {
        state_id_t const id = state_id_of_(walking_states_.size());
        walking_state_t fresh;
        fresh.spelling_state = spelling_state;
        fresh.failure_state = failure_state;
        if (walking_states_.try_push_back(fresh) != status_t::success_k) return status_t::bad_alloc_k;
        if (walking_order_.try_push_back(id) != status_t::success_k) return status_t::bad_alloc_k;
        walking_state = id;
        return status_t::success_k;
    }

    /** @brief The walking state for @p spelling_state with failure @p failure_state, reused from the memo or
     *         minted. The only ceiling is `invalid_state_k`, the same the edge count already hits. */
    status_t walking_state_for_(state_id_t spelling_state, state_id_t failure_state,
                                state_id_t &walking_state) noexcept {
        // Grow the memo before it passes 3/4 load, so probe chains stay short.
        if ((walking_states_.size() + 1) * 4 >= walking_index_.size() * 3) {
            size_t const grown = walking_index_.size() < 1024 ? 1024 : walking_index_.size() * 2;
            if (rehash_walking_index_(grown) != status_t::success_k) return status_t::bad_alloc_k;
        }
        size_t const mask = walking_index_.size() - 1;
        size_t slot = walking_probe_(spelling_state, failure_state) & mask;
        for (; walking_index_[slot] != SZ_SIZE_MAX; slot = (slot + 1) & mask) {
            walking_state_t const &candidate = walking_states_[walking_index_[slot]];
            if (candidate.spelling_state == spelling_state && candidate.failure_state == failure_state)
                return walking_state = state_id_of_(walking_index_[slot]), status_t::success_k;
        }
        if (walking_states_.size() >= (size_t)invalid_state_k) return status_t::overflow_risk_k;
        state_id_t fresh;
        status_t const status = append_walking_state_(spelling_state, failure_state, fresh);
        if (status != status_t::success_k) return status;
        walking_index_[slot] = fresh;
        walking_state = fresh;
        return status_t::success_k;
    }

    /**
     *  @brief Derives the walking automaton from the spelling CSR, splitting each spelling state by its
     *         `(spelling, failure)` pair. A failure link is always shallower, so one shallow-to-deep pass
     *         finishes; @p live_walking_count returns the state count.
     */
    status_t build_walking_automaton_(layout_t const &layout, size_t &live_walking_count) noexcept {
        state_id_t const *const spelling_offsets = edge_offsets_at_(layout);
        csr_edge_t const *const spelling_rows = edges_at_(layout);

        if (walking_offsets_.try_push_back(0) != status_t::success_k) return status_t::bad_alloc_k;
        state_id_t root_walking;
        status_t status = append_walking_state_(0, 0, root_walking); // ? The root pairs with itself.
        if (status != status_t::success_k) return status;

        for (state_id_t band_first = 0, band_last = 1; band_first != band_last;) {
            for (state_id_t parent = band_first; parent < band_last; ++parent) {
                // Copied out: `walking_state_for_` reallocates `walking_states_`, so a held reference dangles.
                state_id_t const parent_spelling = walking_states_[parent].spelling_state;
                state_id_t const parent_failure = walking_states_[parent].failure_state;
                for (state_id_t edge = spelling_offsets[parent_spelling]; edge < spelling_offsets[parent_spelling + 1];
                     ++edge) {
                    u8_t const byte = spelling_rows[edge].byte;
                    // A depth-one state fails to the root; anything deeper chases its parent's failure link.
                    state_id_t const child_failure = parent == 0 ? 0 : chase_walking_(parent_failure, byte);
                    state_id_t child_walking;
                    status = walking_state_for_(spelling_rows[edge].child, child_failure, child_walking);
                    if (status != status_t::success_k) return status;
                    if (walking_rows_.try_push_back(csr_edge_t {child_walking, byte}) != status_t::success_k)
                        return status_t::bad_alloc_k;
                }
                if (walking_offsets_.try_push_back(walking_rows_.size()) != status_t::success_k)
                    return status_t::bad_alloc_k;
                if (parent == 0 && (status = fill_root_row_()) != status_t::success_k) return status;
            }
            order_band_by_out_degree_(band_first, band_last);
            band_first = band_last, band_last = state_id_of_(walking_states_.size());
        }
        live_walking_count = walking_states_.size();
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
     *  Case folding lets several preimage bytes converge on one shared successor, and a double array can
     *  encode only one of them as the state's real slot. Every other converging byte still gets a slot that
     *  `check_` verifies, but that slot is a relay - `publish_` recognizes it as a slot whose raw state's
     *  real slot is elsewhere, and points its failure link straight there.
     */
    status_t pack_cold_tier_(size_t live_walking_count) noexcept {
        // Every walking state is fresh with `published_id == invalid_state_k`, which the packing reads as
        // "not placed yet", so no reset loop is needed.
        lowest_free_cursor_ = hot_count_; // ? Hot states own the low IDs outright.
        status_t status = ensure_slot_capacity_(hot_count_ + alphabet_size_k);
        if (status != status_t::success_k) return status;

        for (size_t hot_index = 0; hot_index < hot_count_; ++hot_index) {
            state_id_t const state = walking_order_[hot_index];
            walking_states_[state].published_id = state_id_of_(hot_index);
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
            walking_states_[0].published_id = state_id_of_(assigned);
            lowest_free_cursor_ = assigned + 1;
        }

        for (size_t index = 0; index < live_walking_count; ++index) {
            state_id_t const parent = walking_order_[index];
            status = index < hot_count_ ? pack_hot_children_(parent) : pack_cold_children_(parent);
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
    status_t pack_hot_children_(state_id_t parent) noexcept {
        for (size_t edge = walking_offsets_[parent]; edge < walking_offsets_[parent + 1]; ++edge) {
            state_id_t const child = walking_rows_[edge].child;
            if (walking_states_[child].published_id != invalid_state_k) continue;
            size_t assigned;
            status_t const status = next_free_slot_(lowest_free_cursor_, assigned);
            if (status != status_t::success_k) return status;
            claim_slot_(assigned);
            check_[assigned] = state_id_of_(assigned);
            old_of_final_[assigned] = child;
            walking_states_[child].published_id = state_id_of_(assigned);
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
    status_t pack_cold_children_(state_id_t parent) noexcept {
        if (walking_offsets_[parent] == walking_offsets_[parent + 1]) return status_t::success_k;

        safe_array<state_id_t, alphabet_size_k> child_of_byte;
        sz_byteset_t child_mask;
        sz_byteset_init(&child_mask);
        for (size_t edge = walking_offsets_[parent]; edge < walking_offsets_[parent + 1]; ++edge) {
            sz_byteset_add_u8(&child_mask, walking_rows_[edge].byte);
            child_of_byte[walking_rows_[edge].byte] = walking_rows_[edge].child;
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

            state_id_t const parent_final = walking_states_[parent].published_id;
            base_[parent_final] = state_id_of_(candidate_base);
            for (size_t quarter = 0; quarter < 4; ++quarter)
                for (u64_t bits = child_mask._u64s[quarter]; bits; bits &= bits - 1) {
                    u8_t const byte = (u8_t)(quarter * 64 + sz_u64_ctz(bits));
                    size_t const slot = candidate_base + byte;
                    state_id_t const child = child_of_byte[byte];
                    claim_slot_(slot);
                    check_[slot] = parent_final;
                    old_of_final_[slot] = child;
                    // The first slot to claim a walking state owns it; later ones become relays.
                    if (walking_states_[child].published_id == invalid_state_k)
                        walking_states_[child].published_id = state_id_of_(slot);
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
    status_t size_and_fill_outputs_(size_t live_walking_count) noexcept {
        size_t running = 0;
        for (size_t index = 0; index < live_walking_count; ++index) {
            state_id_t const state = walking_order_[index];
            walking_state_t &walking = walking_states_[state];
            output_run_t const &spelling = output_runs_[walking.spelling_state];
            // A state's matches are its own plus its failure state's whole run. Read on every byte step, so
            // the total rides `state_id_t` and the ceiling is refused here rather than assumed. Depth-band
            // order finished the failure state first, so its total is already set.
            size_t const total =
                static_cast<size_t>(spelling.own_count) +
                (state == 0 ? size_t {0} : static_cast<size_t>(walking_states_[walking.failure_state].total_count));
            if (total > static_cast<size_t>(invalid_state_k)) return status_t::overflow_risk_k;
            walking.total_count = static_cast<state_id_t>(total);
            walking.total_offset = running;
            running += walking.total_count;
        }

        if (outputs_.try_resize(running) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t index = 0; index < live_walking_count; ++index) {
            state_id_t const state = walking_order_[index];
            walking_state_t const &walking = walking_states_[state];
            output_run_t const &spelling = output_runs_[walking.spelling_state];
            output_t *const destination = outputs_.data() + walking.total_offset;

            // The per-spelling list is most-recent-first, so filling it backwards restores insertion order.
            size_t written = spelling.own_count;
            for (size_t walk = spelling.own_head; walk != SZ_SIZE_MAX; walk = own_outputs_[walk].next) {
                pending_output_t const &pending = own_outputs_[walk];
                destination[--written] = output_t {pending.needle_index, pending.match_bytes};
            }

            if (state == 0) continue;
            walking_state_t const &inherited = walking_states_[walking.failure_state];
            for (size_t position = 0; position < inherited.total_count; ++position)
                destination[spelling.own_count + position] = outputs_[inherited.total_offset + position];
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
    status_t materialize_hot_rows_() noexcept {
        if (hot_rows_.try_resize(hot_count_ * alphabet_size_k) != status_t::success_k) return status_t::bad_alloc_k;
        if (hot_count_ == 0) return status_t::success_k;

        state_id_t *const root_row = hot_rows_.data();
        for (size_t byte = 0; byte < alphabet_size_k; ++byte) root_row[byte] = root_;
        for (size_t edge = walking_offsets_[0]; edge < walking_offsets_[1]; ++edge)
            root_row[walking_rows_[edge].byte] = walking_states_[walking_rows_[edge].child].published_id;

        for (size_t hot_index = 1; hot_index < hot_count_; ++hot_index) {
            state_id_t const state = walking_order_[hot_index];
            // A failure state is strictly shallower, so depth-primary order places it earlier and its row is
            // already final. A real return rather than an assert: violated in a release build this would read
            // an unwritten row and bake wrong transitions into the published automaton, silently.
            size_t const inherited_index = (size_t)walking_states_[walking_states_[state].failure_state].published_id;
            if (inherited_index >= hot_index) return status_t::unexpected_dimensions_k;
            state_id_t const *const inherited = hot_rows_.data() + inherited_index * alphabet_size_k;
            state_id_t *const row = hot_rows_.data() + hot_index * alphabet_size_k;
            for (size_t byte = 0; byte < alphabet_size_k; ++byte) row[byte] = inherited[byte];
            for (size_t edge = walking_offsets_[state]; edge < walking_offsets_[state + 1]; ++edge)
                row[walking_rows_[edge].byte] = walking_states_[walking_rows_[edge].child].published_id;
        }
        return status_t::success_k;
    }

    /** @brief Fills `fail_`, `outputs_counts_`, and `outputs_offsets_` over the published slot range. */
    status_t publish_(size_t cold_capacity_published) noexcept {

        if (fail_.try_resize(cold_capacity_published) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_counts_.try_resize(cold_capacity_published) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_offsets_.try_resize(cold_capacity_published) != status_t::success_k) return status_t::bad_alloc_k;

        for (size_t slot = 0; slot < cold_capacity_published; ++slot) {
            state_id_t const raw_walking = slot < old_of_final_.size() ? old_of_final_[slot] : invalid_state_k;
            if (raw_walking == invalid_state_k) {
                outputs_counts_[slot] = 0, outputs_offsets_[slot] = 0;
                if (slot >= hot_count_) fail_[slot] = root_;
                continue;
            }
            // A relay slot shares its walking state's run outright, so nothing is duplicated in `outputs_`.
            walking_state_t const &walking = walking_states_[raw_walking];
            outputs_counts_[slot] = walking.total_count;
            outputs_offsets_[slot] = walking.total_offset;
            max_outputs_per_state_ = sz_max_of_two(max_outputs_per_state_, walking.total_count);
            if (slot < hot_count_) continue;
            // A relay names a walking state whose real slot is elsewhere; one extra failure hop lands on
            // that state's own children, which the relay itself does not own.
            state_id_t const real_slot = walking.published_id;
            fail_[slot] = real_slot != state_id_of_(slot) ? real_slot
                                                          : walking_states_[walking.failure_state].published_id;
        }
        return status_t::success_k;
    }

#pragma endregion Build Phases

  public:
    aho_corasick_dictionary() = default;
    ~aho_corasick_dictionary() noexcept { reset(); }

    explicit aho_corasick_dictionary(allocator_t alloc) noexcept
        : edges_(alloc), edge_index_(alloc), own_outputs_(alloc), output_runs_(alloc), folded_positions_(alloc),
          chain_slots_(alloc), build_scratch_(alloc), walking_states_(alloc), walking_offsets_(alloc),
          walking_rows_(alloc), walking_order_(alloc), walking_root_row_(alloc), walking_index_(alloc),
          old_of_final_(alloc), occupied_bits_(alloc), hot_rows_(alloc), base_(alloc), check_(alloc), fail_(alloc),
          outputs_(alloc), outputs_counts_(alloc), outputs_offsets_(alloc), alloc_(alloc) {}

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
        folded_positions_.reset();
        chain_slots_.reset();
        build_scratch_.reset();
        walking_states_.reset();
        walking_offsets_.reset();
        walking_rows_.reset();
        walking_order_.reset();
        walking_root_row_.reset();
        walking_index_.reset();
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
        max_match_bytes_ = 0;
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
    state_id_t max_match_bytes() const noexcept { return max_match_bytes_; }
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

        // The insertion pools have no reader past compaction; the spelling CSR in `build_scratch_` carries
        // every edge, and the splitting pass below walks it rather than `find_edge_`.
        edges_.reset();
        edge_index_.reset();

        size_t live_walking_count = 0;
        status = build_walking_automaton_(layout, live_walking_count);
        if (status != status_t::success_k) return status;

        // The spelling CSR has no reader past the splitting pass; the walking CSR carries every edge now.
        build_scratch_.reset();

        // Hot rows are shared, read-mostly, and re-entered on nearly every byte, so they're sized against
        // the last-level cache rather than a private L2 slice.
        if (hot_count_ == derive_hot_count_k) hot_count_ = specs.l3_bytes / (alphabet_size_k * sizeof(state_id_t));
        hot_count_ = sz_min_of_two(hot_count_, live_walking_count);

        status = pack_cold_tier_(live_walking_count);
        if (status != status_t::success_k) return status;

        root_ = walking_states_[0].published_id;
        sz_assert_(root_ == 0 && "The root is the unique shallowest state, so it always sorts first");

        // The exclusive published bound: not `hot_count_` plus the cold-state count, since a packed child's
        // ID is address arithmetic and can skip past slots no state ever ended up owning.
        size_t state_count_published = hot_count_;
        for (size_t walking = 0; walking < live_walking_count; ++walking)
            state_count_published = sz_max_of_two(state_count_published, (size_t)walking_states_[walking].published_id + 1);
        size_t const cold_capacity_published = state_count_published + (alphabet_size_k - 1);

        // `base_` and `check_` were written in place by the packing above; widening them here only extends
        // the address-arithmetic headroom a `base_[state] + byte` lookup can reach.
        status = ensure_slot_capacity_(cold_capacity_published);
        if (status != status_t::success_k) return status;

        status = size_and_fill_outputs_(live_walking_count);
        if (status != status_t::success_k) return status;
        status = materialize_hot_rows_();
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
        if ((size_t)source.max_match_bytes > (size_t)invalid_state_k) return status_t::overflow_risk_k;
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
                                          static_cast<state_id_t>(source.outputs[output].match_bytes)};

        count_states_ = source.state_count;
        count_needles_ = wider.count_needles();
        hot_count_ = source.hot_count;
        root_ = static_cast<state_id_t>(source.root);
        max_match_bytes_ = static_cast<state_id_t>(source.max_match_bytes);
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
        result.max_match_bytes = max_match_bytes_;
        result.max_outputs_per_state = max_outputs_per_state_;
        return result;
    }

#pragma endregion Published View

#pragma region Matching

    /**
     *  @brief Finds all occurrences of all needles in the @p haystack.
     *  @note This is the serial reference oracle: obvious correctness over speed.
     *  @param[in] callback Invoked as `callback(needle_index, match_offset, match_length)` with offsets
     *             relative to the span handed in, returning `true` to continue.
     */
    template <typename callback_type_>
    void find(span<byte_t const> haystack, callback_type_ &&callback) const noexcept {
        view_t const automaton = view();
        state_id_t current_state = automaton.root;
        for (size_t offset = 0; offset < haystack.size(); ++offset) {
            u8_t const byte = haystack[offset];
            size_t const output_count = aho_corasick_step_counting(automaton, current_state, byte);
            if (output_count == 0) continue;
            size_t const output_offset = automaton.outputs_offsets[current_state];

            for (size_t index = 0; index < output_count; ++index) {
                output_t const &output = automaton.outputs[output_offset + index];
                size_t const match_length = output.match_bytes;
                // Tested by addition rather than by subtracting the length from the position: the walk
                // always restarts at the root at this span's own start, so a match can never reach behind
                // it, but a subtraction would wrap and read as in-bounds if that ever stopped holding.
                if (offset + 1 < match_length) continue;
                if (!callback((size_t)output.needle_index, offset + 1 - match_length, match_length)) return;
            }
        }
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

#pragma endregion Matching
};

using substrings_u16_dictionary_t = aho_corasick_dictionary<u16_t, std::allocator<char>>;
using substrings_u32_dictionary_t = aho_corasick_dictionary<u32_t, std::allocator<char>>;

#pragma endregion Dictionary

#pragma region Primary API

/**
 *  @brief Aho-Corasick-based @b single-threaded multi-pattern exact/case-folded substring search.
 *  @tparam state_id_type_ The type of the state ID.
 *  @tparam allocator_type_ The type of the allocator.
 *  @tparam capability_ Matches `sz_cap_serial_k` and any other capability that isn't parallel or CUDA; the
 *          two-level parallel specialization below claims `sz_caps_sp_k` specifically.
 */
template <typename state_id_type_, typename allocator_type_, sz_capability_t capability_>
struct substrings<state_id_type_, allocator_type_, capability_,
                  std::enable_if_t<(capability_ & (sz_cap_parallel_k | sz_cap_cuda_k)) == 0>> {
    using dictionary_t = aho_corasick_dictionary<state_id_type_, allocator_type_>;
    using state_id_t = typename dictionary_t::state_id_t;
    using allocator_t = typename dictionary_t::allocator_t;
    using match_t = substrings_match_t;
    static constexpr sz_capability_t capability_k = capability_;

    using size_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;

    explicit substrings(allocator_t alloc = allocator_t()) noexcept : dict_(alloc) {}
    void reset() noexcept { dict_.reset(); }
    dictionary_t const &dictionary() const noexcept { return dict_; }

    /**
     *  @brief Indexes all of the @p needles strings into the FSM.
     *  @param[in] specs Sizes the hot tier from the host's last-level cache.
     *  @note Before reusing, please `reset` the FSM.
     *  @sa `aho_corasick_dictionary::try_insert` for the status codes this forwards.
     */
    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles, substrings_case_sensitivity_t case_sensitivity = substrings_cased_k,
                       cpu_specs_t const &specs = {}) noexcept {
        dict_.case_sensitivity(case_sensitivity);
        for (auto const &needle : needles) {
            status_t const status = dict_.try_insert(to_bytes_view(needle));
            if (status != status_t::success_k) return status;
        }
        return dict_.try_build(specs);
    }

    /**
     *  @brief Adopts an already-built @p wider automaton at this engine's narrower state id.
     *
     *  Named by the dictionary rather than the engine that owns it: the narrowing needs nothing else, and
     *  the concrete parameter type is what keeps this overload from competing with the needles one above.
     *  @sa `aho_corasick_dictionary::try_build` for the narrowing contract and its status codes.
     */
    template <typename wider_id_type_, typename wider_allocator_type_>
    status_t try_build(aho_corasick_dictionary<wider_id_type_, wider_allocator_type_> const &wider) noexcept {
        return dict_.try_build(wider);
    }

    /**
     *  @brief Occurrences of all needles in each of the @p haystacks, for filtering and ranking.
     *  @param[out] matches_total Sum of @p counts_per_haystack, which is what sizes a later `try_find` buffer.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
    status_t try_count(haystacks_type_ const &haystacks, span<size_t> counts_per_haystack, size_t &matches_total,
                       executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) const noexcept {
        // A single-threaded walk has no split to schedule and no cache size to compare against.
        sz_unused_(executor);
        sz_unused_(specs);
        sz_assert_(counts_per_haystack.size() == haystacks.size());
        matches_total = 0;
        for (size_t index = 0; index < counts_per_haystack.size(); ++index)
            matches_total += counts_per_haystack[index] = dict_.count(to_bytes_view(haystacks[index]));
        return status_t::success_k;
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks.
     *  @param[out] matches_found Matches written, or - when @p matches is too small - the count that would be,
     *              so `matches.size() == 0` is a size query rather than a wasted call.
     *  @retval `status_t::unexpected_dimensions_k` @p matches is too small; nothing is written in that case.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
    status_t try_find(haystacks_type_ const &haystacks, span<substrings_match_t> matches, size_t &matches_found,
                      executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) noexcept {

        // Counting first is what makes the capacity refusable before any write.
        matches_found = 0;
        if (counts_per_haystack_.try_resize(haystacks.size()) != status_t::success_k) return status_t::bad_alloc_k;
        span<size_t> const counts_per_haystack {counts_per_haystack_.data(), haystacks.size()};
        if (status_t const status = try_count(haystacks, counts_per_haystack, matches_found, executor, specs);
            status != status_t::success_k)
            return status;

        // The count survives the refusal, so a caller that brought no buffer still learns what to allocate.
        if (matches_found > matches.size()) return status_t::unexpected_dimensions_k;

        size_t count_written = 0;
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index)
            dict_.find(to_bytes_view(haystacks[haystack_index]),
                       [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                           matches[count_written] = {haystack_index, needle_index, match_offset, match_length};
                           count_written++;
                           return true;
                       });
        sz_assert_(count_written == matches_found);
        return status_t::success_k;
    }

  private:
    dictionary_t dict_;

    /** @brief Grow-only per-call scratch, reused across calls; concurrent calls on one engine are unsafe. */
    safe_vector<size_t, size_allocator_t> counts_per_haystack_ {};
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
template <typename state_id_type_, typename allocator_type_, typename enable_>
struct substrings<state_id_type_, allocator_type_, sz_caps_sp_k, enable_> {

    using dictionary_t = aho_corasick_dictionary<state_id_type_, allocator_type_>;
    using state_id_t = typename dictionary_t::state_id_t;
    using allocator_t = typename dictionary_t::allocator_t;
    using match_t = substrings_match_t;
    static constexpr sz_capability_t capability_k = sz_caps_sp_k;

    using size_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;

    explicit substrings(allocator_t alloc = allocator_t()) noexcept : dict_(alloc) {}
    void reset() noexcept { dict_.reset(); }
    dictionary_t const &dictionary() const noexcept { return dict_; }

    /**
     *  @brief Indexes all of the @p needles strings into the FSM.
     *  @param[in] specs Sizes the hot tier from the host's last-level cache.
     *  @note Before reusing, please `reset` the FSM.
     *  @sa `aho_corasick_dictionary::try_insert` for the status codes this forwards.
     */
    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles, substrings_case_sensitivity_t case_sensitivity = substrings_cased_k,
                       cpu_specs_t const &specs = {}) noexcept {
        dict_.case_sensitivity(case_sensitivity);
        for (auto const &needle : needles) {
            status_t const status = dict_.try_insert(to_bytes_view(needle));
            if (status != status_t::success_k) return status;
        }
        return dict_.try_build(specs);
    }

    /**
     *  @brief Adopts an already-built @p wider automaton at this engine's narrower state id.
     *
     *  Named by the dictionary rather than the engine that owns it: the narrowing needs nothing else, and
     *  the concrete parameter type is what keeps this overload from competing with the needles one above.
     *  @sa `aho_corasick_dictionary::try_build` for the narrowing contract and its status codes.
     */
    template <typename wider_id_type_, typename wider_allocator_type_>
    status_t try_build(aho_corasick_dictionary<wider_id_type_, wider_allocator_type_> const &wider) noexcept {
        return dict_.try_build(wider);
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
    status_t try_count(haystacks_type_ const &haystacks, span<size_t> counts_per_haystack, size_t &matches_total,
                       executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) noexcept {

        sz_assert_(counts_per_haystack.size() == haystacks.size());
        matches_total = 0;

        using haystack_t = typename haystacks_type_::value_type;
        static_assert(std::is_trivially_copyable<haystack_t>::value,
                      "The haystack should be trivially copyable for higher compatibility.");

        // On small strings, individually compute the counts.
        executor.for_n_dynamic(counts_per_haystack.size(), [&](size_t haystack_index) noexcept {
            haystack_t const &haystack = haystacks[haystack_index];
            if (is_large_(haystack.size(), specs)) return;
            counts_per_haystack[haystack_index] = dict_.count(to_bytes_view(haystack));
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
     *  @brief Finds all occurrences of all needles in all the @p haystacks.
     *  @param[out] matches_found Matches written, in ascending haystack order.
     *  @retval `status_t::unexpected_dimensions_k` @p matches is too small; nothing is written in that case.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t try_find(haystacks_type_ const &haystacks, span<substrings_match_t> matches, size_t &matches_found,
                      executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) noexcept {

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

        executor.for_n_dynamic(haystacks.size(), [&](size_t haystack_index) noexcept {
            haystack_t const &haystack = haystacks[haystack_index];
            if (is_large_(haystack.size(), specs)) return;
            counts_per_haystack_[haystack_index] = dict_.count(to_bytes_view(haystack));
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

  private:
    dictionary_t dict_;

    /** @brief Grow-only per-call scratch, reused across calls; concurrent calls on one engine are unsafe. */
    safe_vector<size_t, size_allocator_t> counts_per_core_ {};
    safe_vector<size_t, size_allocator_t> counts_per_haystack_ {};
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
            counts_per_core[core_index] = count_matches_in_one_part(haystack, optimal_split[core_index]);
        });
    }

    /** @brief Writes every small haystack's matches, one core per haystack into disjoint output ranges. */
    template <typename haystacks_type_, typename executor_type_>
    void scatter_matches_of_small_(haystacks_type_ const &haystacks, span<size_t const> counts,
                                   span<size_t const> offsets, span<substrings_match_t> matches,
                                   executor_type_ &executor, cpu_specs_t const &specs) const noexcept {

        using haystack_t = typename haystacks_type_::value_type;
        sz_unused_(counts);

        executor.for_n_dynamic(offsets.size(), [&](size_t haystack_index) noexcept {
            haystack_t const &haystack = haystacks[haystack_index];
            auto const haystack_bytes = to_bytes_view(haystack);
            if (is_large_(haystack_bytes.size(), specs)) return;

            size_t matches_found = 0;
            dict_.find(haystack_bytes, [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                matches[offsets[haystack_index] + matches_found] = {haystack_index, needle_index, match_offset,
                                                                    match_length};
                ++matches_found;
                return true;
            });
            sz_assert_(counts[haystack_index] == matches_found);
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

            fu::indexed_range_t const optimal_subrange = optimal_split[core_index];
            byte_t const *optimal_begin = haystack.begin() + optimal_subrange.first;
            byte_t const *const optimal_end = optimal_begin + optimal_subrange.count;
            // An empty dictionary reports zero, where `optimal_end + 0 - 1` would step past the end.
            size_t const overlap_bytes = dict_.max_match_bytes() > 0 ? (size_t)dict_.max_match_bytes() - 1 : 0;
            byte_t const *const overlapping_end = sz_min_of_two(optimal_end + overlap_bytes, haystack.end());

            // Offsets arrive relative to the slice the dictionary was handed, not to the whole haystack.
            size_t const slice_offset_in_haystack = (size_t)(optimal_begin - haystack.begin());
            size_t const owned_bytes = optimal_subrange.count;
            size_t count_matches_found_on_this_core = 0;
            dict_.find({optimal_begin, overlapping_end},
                       [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                           bool const belongs_to_this_core = match_offset < owned_bytes;
                           if (!belongs_to_this_core) return true;
                           matches[base_offset + count_matches_before_this_core + count_matches_found_on_this_core] = {
                               haystack_index, needle_index, slice_offset_in_haystack + match_offset, match_length};
                           count_matches_found_on_this_core++;
                           return true;
                       });
            sz_assert_(count_matches_found_on_this_core == count_matches_expected_on_this_core);
        });
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
        bool const longest_match_fits_on_one_core = optimal_split.smallest_size() >= dict_.max_match_bytes();

        if (!longest_match_fits_on_one_core)
            executor.for_threads([&](size_t core_index) noexcept {
                counts_per_core[core_index] = count_matches_in_one_part(haystack, optimal_split[core_index]);
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

        size_t const max_match_bytes = sz_min_of_two(dict_.max_match_bytes(), haystack.size());

        byte_t const *optimal_begin = haystack.begin() + optimal_subrange.first;
        byte_t const *const optimal_end = optimal_begin + optimal_subrange.count;

        size_t const count_matches_non_overlapping = dict_.count({optimal_begin, optimal_end});

        byte_t const *overlapping_start;
        byte_t const *overlapping_end;
        if (optimal_begin + max_match_bytes >= optimal_end) {
            overlapping_start = optimal_begin;
            overlapping_end = sz_min_of_two(optimal_end + max_match_bytes, haystack.end());
        }
        else {
            overlapping_start = sz_max_of_two(optimal_end - max_match_bytes + 1, optimal_begin);
            overlapping_end = sz_min_of_two(optimal_end + max_match_bytes - 1, haystack.end());
        }

        // Both branches place `overlapping_start` at or after `optimal_begin`, so offsets relative to it need
        // no lower-bound test.
        size_t const slice_end_offset = (size_t)(optimal_end - overlapping_start);
        size_t count_matches_overlapping = 0;
        dict_.find({overlapping_start, overlapping_end},
                   [&](size_t needle_index, size_t match_offset, size_t match_length) noexcept {
                       sz_unused_(needle_index);
                       bool const belongs_to_this_core =                   //
                           match_offset < slice_end_offset &&              // ? Starts before this slice ends.
                           match_offset + match_length > slice_end_offset; // ? Ends beyond this slice.
                       count_matches_overlapping += belongs_to_this_core;
                       return true;
                   });

        return count_matches_non_overlapping + count_matches_overlapping;
    }

    /**
     *  @brief  More optimized alternative to `count_matches_in_one_part`, assuming the longest match fits
     *          within a single core's slice, so a match can only spill into 2 core regions at most.
     *  @param[out] matches_in_prefix Matches ending within the first `max_match_bytes` of this core's slice,
     *              which the preceding core has already counted as its own overlapping tail.
     *  @return Total matches ending anywhere in this core's slice or its overlapping tail.
     */
    size_t count_short_matches_in_one_part(span<byte_t const> haystack, fu::indexed_range_t const optimal_subrange,
                                           size_t &matches_in_prefix) const noexcept {

        typename dictionary_t::view_t const automaton = dict_.view();
        size_t const max_match_bytes = dict_.max_match_bytes();
        byte_t const *optimal_begin = haystack.begin() + optimal_subrange.first;
        byte_t const *const optimal_end = optimal_begin + optimal_subrange.count;
        byte_t const *const prefix_end = sz_min_of_two(optimal_begin + max_match_bytes, haystack.end());
        byte_t const *const overlapping_end = sz_min_of_two(optimal_end + max_match_bytes, haystack.end());

        size_t matches_in_part = 0;
        matches_in_prefix = 0;
        state_id_t current_state = automaton.root;
        // The prefix window spans at most `max_match_bytes` bytes and is the only region needing the
        // per-byte attribution test, so it walks scalar; `prefix_end` never exceeds `overlapping_end`,
        // both being clamped by the same haystack end.
        for (; optimal_begin != prefix_end; ++optimal_begin) {
            state_id_t const output_count = aho_corasick_step_counting(automaton, current_state, *optimal_begin);
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
    }
};

using substrings_u16_serial_t = substrings<u16_t, std::allocator<char>, sz_cap_serial_k>;
using substrings_u16_parallel_t = substrings<u16_t, std::allocator<char>, sz_caps_sp_k>;
using substrings_u32_serial_t = substrings<u32_t, std::allocator<char>, sz_cap_serial_k>;
using substrings_u32_parallel_t = substrings<u32_t, std::allocator<char>, sz_caps_sp_k>;

#pragma endregion Parallel Backend

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SUBSTRINGS_SERIAL_HPP_
