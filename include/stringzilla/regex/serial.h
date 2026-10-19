/**
 *  @file include/stringzilla/regex/serial.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Serial backend for regular expressions: the parser, the Thompson NFA, both subset
 *      constructions, priming, and the windowed walks every other CPU tier reuses.
 *
 *  A vocabulary compiles into two automata over one set of byte classes. The start-finding one is
 *  the patterns reversed and unanchored, keeping every match: walked right to left, it marks every
 *  position some match starts at. The end-finding one is the patterns anchored and leftmost-first:
 *  walked from one start until it dies, its last accepting state names the match's end and pattern.
 *  A search marks one window's starts, then follows the chain the sequential search takes: the
 *  first start at or past the cursor, its end, and the cursor moved there. Where every match starts
 *  with a few literal bytes, a search for them replaces the marking.
 *
 *  Both are built the way `regex-automata` builds its dense DFAs, so leftmost-first priority,
 *  look-around and empty matches come out exactly as Rust's @c regex reports them: the NFA follows
 *  its Thompson compiler, and a DFA state is an ordered set of NFA states, cut after the first
 *  match under leftmost-first, that reports a match one byte late so look-ahead can resolve.
 *
 *  A start-finding DFA past @ref sz_regex_reverse_budget_k states, which bounded gaps like
 *  `.{0,40}` reach, gives way to a bit-parallel automaton over the same reversed NFA: one bit per
 *  class occurrence, consuming whole codepoints, stepped by shifts and grouped follow sets. An
 *  end-finding DFA past @ref sz_regex_forward_budget_k gives way to a PikeVM over the forward NFA,
 *  which the engine keeps either way, as groups are read from it.
 *
 *  A lookaround compiles its body into the forward NFA as a sub-vocabulary of its own, and its look
 *  state tests one bit of a table every walk fills first: per haystack position, the lookarounds
 *  that hold there, found by a forward pass of the lookbehind bodies, unanchored and keeping every
 *  match, and a backward pass of the reversed lookahead bodies. Up to six lookarounds become DFA
 *  columns: each set of them that may hold at a boundary takes a block of the row, which a step
 *  picks from the table; a vocabulary with more walks positions and the PikeVM.
 *
 *  @sa include/stringzilla/regex.h
 */
#ifndef STRINGZILLA_REGEX_SERIAL_H_
#define STRINGZILLA_REGEX_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`, `sz_cap_serial_k`

#include "stringzilla/compare/serial.h"           // `sz_equal_serial_`
#include "stringzilla/find/serial.h"              // `sz_rfind_byteset_serial_`
#include "stringzilla/utf8_runes/serial.h"        // `sz_rune_decode`, `sz_rune_encode`
#include "stringzilla/utf8_uncased_fold/serial.h" // `sz_unicode_fold_codepoint_`, the folded iterators

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Engine

/** The kinds of NFA states. */
typedef enum sz_regex_nfa_kind_t {

    /** Consumes one byte along every link whose range holds it; with no links it never matches. */
    sz_regex_nfa_ranges_k = 0,

    /** Moves along every link without consuming, in link order. */
    sz_regex_nfa_union_k = 1,

    /** Moves along every link without consuming; links are prepended, so later patches go first. */
    sz_regex_nfa_union_lazy_k = 2,

    /** Moves to @c target without consuming. */
    sz_regex_nfa_empty_k = 3,

    /** Moves to @c target without consuming where its assertion holds. */
    sz_regex_nfa_look_k = 4,

    /** A pattern matched; @c target is the pattern's index. */
    sz_regex_nfa_match_k = 5,
} sz_regex_nfa_kind_t;

/** One NFA state. */
typedef struct sz_regex_nfa_state_t {

    /** What the state does. */
    sz_regex_nfa_kind_t kind;

    /** An assertion's bit. */
    sz_u32_t look;

    /** The next state of an empty or look state, or a match state's pattern. */
    sz_u32_t target;

    /** The capture slot an empty state writes as a walk passes it, twice its group plus one where
     *  the group closes, or @ref sz_regex_none_k. */
    sz_u32_t slot;

    /** The first outgoing link of a ranges or union state, or @ref sz_regex_none_k. */
    sz_u32_t first_link;

    /** The last outgoing link, which an appended link follows. */
    sz_u32_t last_link;
} sz_regex_nfa_state_t;

/** How a lookaround reads its body around the position it is tested at. */
typedef enum sz_regex_lookaround_kind_t {

    /** `(?=…)`: the body matches from the position on. */
    sz_regex_lookahead_k = 0,

    /** `(?!…)`: the body matches nowhere from the position on. */
    sz_regex_lookahead_negated_k = 1,

    /** `(?<=…)`: the body matches up to the position. */
    sz_regex_lookbehind_k = 2,

    /** `(?<!…)`: the body matches nowhere up to the position. */
    sz_regex_lookbehind_negated_k = 3,
} sz_regex_lookaround_kind_t;

/** One lookaround: its body, compiled into the engine's NFA as a sub-vocabulary of its own, and how
 *  it reads it. */
typedef struct sz_regex_lookaround_t {

    /** Where a walk of the body enters @c nfa_states, compiled forward for a lookbehind and
     *  reversed for a lookahead; while the vocabulary builds, the root of its syntax tree. */
    sz_u32_t start;

    /** How the lookaround reads its body: odd kinds are negated, and those from
     *  @c sz_regex_lookbehind_k look behind. */
    sz_regex_lookaround_kind_t kind;

    /** One past the highest level among the lookarounds of its body, so levels computed in
     *  ascending order find every bit a body tests known. */
    sz_u32_t level;
} sz_regex_lookaround_t;

/** One lookaround body's DFA, unanchored and keeping every match, which fills the table of where
 *  its lookaround holds a byte per step instead of the PikeVM. */
typedef struct sz_regex_body_t {

    /** Rows of `classes_count + 1` next states, the last column taken at the far edge of a
     *  haystack, ids being row offsets; null for a body over other lookarounds, of one byte, or
     *  past its budget. */
    sz_u32_t const *rows;

    /** The state a pass starts from at the haystack's edge it walks from. */
    sz_u32_t start;

    /** Ids at or above this one accept, which the step over the unit after a match of the body
     *  lands on. */
    sz_u32_t accepts_begin;
} sz_regex_body_t;

/** One outgoing edge, chained per state. */
typedef struct sz_regex_nfa_link_t {

    /** The state the link leads to. */
    sz_u32_t target;

    /** The owner's next link, or @ref sz_regex_none_k. */
    sz_u32_t next;

    /** The lowest byte a ranges state's link consumes. */
    sz_u8_t low;

    /** The highest byte a ranges state's link consumes. */
    sz_u8_t high;
} sz_regex_nfa_link_t;

/** Bytes from a candidate start that the literal prefilter tests. */
enum { sz_regex_prefixes_depth_k = 3 };

/** Paths of byte sets the literal prefilter tells apart, one bucket bit each. */
enum { sz_regex_prefixes_paths_k = 8 };

/**
 *  @brief The paths of byte sets the first bytes of every match follow, as the end-finding DFA's
 *      first steps allow them under any lookaround, each path one bucket bit.
 *
 *  A search skips to the next position whose bytes keep some bucket at every offset, and the end
 *  walk from there settles whether a match starts there, so a superset of the paths stays exact.
 *  A path shorter than the deepest holds every byte past its end. Vector tiers test a byte's two
 *  nibbles apart, which holds a superset of its bucket's bytes where a path's set has several.
 */
typedef struct sz_regex_prefixes_t {

    /** Per offset from a candidate start, per byte, the buckets whose paths hold the byte there. */
    sz_u8_t buckets[sz_regex_prefixes_depth_k][256];

    /** Per offset, the buckets holding some byte of each low nibble, then of each high nibble;
     *  every bucket at offsets past @c depth. */
    sz_u8_t nibbles[sz_regex_prefixes_depth_k][32];

    /** Offsets of the deepest path. */
    sz_u32_t depth;
} sz_regex_prefixes_t;

/**
 *  @brief A compiled vocabulary of regular expressions: two automata over one byte-class map.
 *
 *  Both automata are dense tables of `classes_count + 1` columns per block, the last column taken
 *  at a haystack's edge, and one block per set of lookarounds that may hold at the boundary a step
 *  crosses, as @ref sz_regex_stride_ and @ref sz_regex_looks_block_ lay them out. A state id is its
 *  row's offset, so a step is one load with no multiply; the dead state is zero, and accepting
 *  states are numbered last, so acceptance is one compare.
 *
 *  A start-finding DFA that would pass its state budget is replaced by a position automaton, with
 *  @c reverse_rows null: a set of @c positions_words words over the classes of the reversed
 *  patterns, stepped over whole codepoints. A boundary kind is the kind of the unit left of a
 *  boundary times four plus the kind right of it, each zero at the haystack's edge, one for a line
 *  feed, two for a word byte and three otherwise; a mask of them is a 16-bit word.
 *
 *  A lookaround is a look state testing bit `sz_regex_look_lookaround_k << i` of its index @c i,
 *  which a table of the walk holds per haystack position, as @c lookarounds describes; a DFA step
 *  reads it to pick its block, except within one codepoint's folds, which take the first block.
 *
 *  Both blocks are owned and live until @ref sz_regex_engine_free; the round's scratch grows on
 *  demand and is never shrunk.
 */
typedef struct sz_regex_engine_t {

    /** Each byte's column in both automata; bytes share one only if no pattern tells them apart. */
    sz_u8_t const *byte_to_class;

    /** Start-finding automaton: the reversed patterns, unanchored, keeping every match, walked
     *  right to left. Rows of blocks of `classes_count + 1` next states, the last column taken at
     *  the start of a haystack, ids being row offsets. A state at or above @c reverse_accepts_begin
     *  after the byte at @c i starts a match at `i + 1`; that step takes the block of the
     *  lookarounds holding at `i + 1`. */
    sz_u32_t const *reverse_rows;

    /** End-finding automaton: the patterns anchored and leftmost-first, walked left to right from
     *  one start and laid out the same, a step over the byte at @c i taking the block of the
     *  lookarounds holding at @c i. The last accepting state before it dies names the match's end
     *  and pattern. Null past its state budget or past the lookarounds its capability takes as
     *  columns, where end walks run a PikeVM over @c nfa_states. */
    sz_u32_t const *forward_rows;

    /** Per forward accepting state, in id order, the pattern leftmost-first priority picks. */
    sz_u32_t const *forward_patterns;

    /** Per forward state, in id order, the bytes that move it elsewhere under any block, which an
     *  end walk searches ahead for where the state stays put on all but a few printable ASCII
     *  bytes; every byte for other states. */
    sz_byteset_t const *forward_escapes;

    /** The paths every match's first bytes follow, which a search skips to instead of marking
     *  starts; null where they would pass too many positions, where some match is empty, for an
     *  uncased engine, and where end walks run the PikeVM. */
    sz_regex_prefixes_t const *prefixes;

    /** Forward start by the byte before a match: none, a line feed, a word byte, or another. */
    sz_u32_t forward_starts[4];

    /** Reverse start by the byte after a haystack: none, a line feed, a word byte, or another. */
    sz_u32_t reverse_starts[4];

    /** Forward ids at or above this one accept, so acceptance is a compare rather than a load. */
    sz_u32_t forward_accepts_begin;

    /** Reverse ids at or above this one mark a start. */
    sz_u32_t reverse_accepts_begin;

    /** Columns of a block before its haystack-edge column. */
    sz_u32_t classes_count;

    /** Rows of the end-finding automaton. */
    sz_u32_t forward_states_count;

    /** Rows of the start-finding automaton, all of which priming walks at once. */
    sz_u32_t reverse_states_count;

    /** Patterns the vocabulary holds, which bounds every reported @c needle_index. */
    sz_u32_t patterns_count;

    /** Bytes that move the start pass off its idle states, which a reverse byte-set search skips
     *  to: for the DFA, the starts after a line feed, a word byte and another byte, which any other
     *  byte maps into one another under every block; for positions, the loop alone, or every byte
     *  when the loop alone can match. */
    sz_byteset_t reverse_live;

    /** Words of one position set of the bit-parallel start pass, which replaces @c reverse_rows
     *  past their budget. */
    sz_u32_t positions_words;

    /** Records in @c positions_shifts: a signed distance word, then the positions moving by it
     *  on every unit. */
    sz_u32_t positions_shifts_count;

    /** Records in @c positions_groups: a mask of boundary kinds, the source positions, then the
     *  targets they enter. */
    sz_u32_t positions_groups_count;

    /** Pairs in @c positions_ranges. */
    sz_u32_t positions_ranges_count;

    /** Per unit class, the positions able to consume it; the loop and match positions are
     *  in every row. */
    sz_u64_t const *positions_reach;

    /** The unconditional follows that a constant distance between positions covers,
     *  one record per distance. */
    sz_u64_t const *positions_shifts;

    /** Every other follow, its sources grouped by the targets they share under one condition: a
     *  mask of boundary kinds in the low 16 bits of the record's first word, then the look bits of
     *  the lookarounds that must hold. */
    sz_u64_t const *positions_groups;

    /** Every position a walk can hold, which priming's upper run starts from. */
    sz_u64_t const *positions_all;

    /** Per boundary kind, the positions the always-live loop enters, itself among them,
     *  @c positions_words each. */
    sz_u64_t const *positions_initial;

    /** Each ASCII codepoint's unit class. */
    sz_u32_t const *positions_ascii;

    /** Past ASCII, ascending pairs of a codepoint range's first codepoint and its unit class. */
    sz_u32_t const *positions_ranges;

    /** The Thompson NFA the end-finding automaton was built from, which a PikeVM walks where
     *  @c forward_rows is null and which groups are read from; a capture is an empty state
     *  carrying a slot. */
    sz_regex_nfa_state_t const *nfa_states;

    /** Every link of @c nfa_states, chained per state. */
    sz_regex_nfa_link_t const *nfa_links;

    /** States of @c nfa_states. */
    sz_u32_t nfa_states_count;

    /** Links of @c nfa_links. */
    sz_u32_t nfa_links_count;

    /** Where a walk of every pattern at once enters the NFA, the patterns as alternatives
     *  in index order. */
    sz_u32_t nfa_start;

    /** Per pattern, where a walk of that pattern alone enters the NFA. */
    sz_u32_t const *patterns_starts;

    /** Per pattern, its first capturing group among all groups numbered pattern after pattern,
     *  then their total; group @c g of a pattern, counting from one, writes slots `2g` and
     *  `2g + 1` of a walk. */
    sz_u32_t const *patterns_groups;

    /** Per lookaround, in the order of the look bits naming them, its body's entry into
     *  @c nfa_states and its kind. */
    sz_regex_lookaround_t const *lookarounds;

    /** Per lookaround, its body's DFA, which fills the table where it is built; null without
     *  lookarounds. */
    sz_regex_body_t const *bodies;

    /** The lookarounds whose bodies match one byte of a set and assert nothing, so the byte beside
     *  a position alone decides whether they hold; then per byte, those holding right after it;
     *  then per byte, those holding right before it. Null where no body is one byte, as in an
     *  uncased engine, whose lookarounds hold between whole codepoints only. */
    sz_u32_t const *looks_bytes;

    /** Lookarounds the vocabulary holds, nested ones included; up to
     *  @c sz_regex_columned_lookarounds_k of them widen the rows of both DFAs to a block per set,
     *  and more leave walks on positions and the PikeVM. */
    sz_u32_t lookarounds_count;

    /** Every capturing group's name, in the order @c patterns_groups numbers them,
     *  empty for an unnamed group. */
    sz_strings_view_t groups_names;

    /** Whether walks fold the haystack as they consume it, or step it byte for byte. */
    sz_case_sensitivity_t case_sensitivity;

    /** The capability that compiled the vocabulary, whose kernels every round then runs. */
    sz_capability_t capability;

    /** What built both blocks below. */
    sz_allocator_t allocator;

    /** Both automata and their side arrays, fixed for the engine's life. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** A round's start bits, priming sets and window edges,
     *  grown on demand like @c levenshtein's. */
    void *scratch;

    /** Bytes of that block. */
    sz_size_t scratch_bytes;
} sz_regex_engine_t;

#pragma endregion Engine

#pragma region Syntax

/** The longest window one start pass marks before the cursor chain consumes it. A haystack splits
 *  into equal windows at most this long, a whole number of batches of
 *  @c sz_regex_chains_serial_k windows each. */
enum { sz_regex_window_serial_k = 262144 };

/** The shortest window worth a chain of its own where every window primes its entry, as uncased and
 *  position walks do; a haystack too short for a batch of them is one window. */
enum { sz_regex_window_minimum_k = 4096 };

/** The shortest window worth a chain of its own where a cased DFA batch guesses its entries,
 *  priming one per batch. */
enum { sz_regex_window_guessed_k = 128 };

/** Start-finding DFA states past which the start pass runs on the bit-parallel
 *  position automaton instead. */
enum { sz_regex_reverse_budget_k = 16384 };

/** End-finding DFA states past which end walks run a PikeVM over the forward NFA instead, holding
 *  compilation near 50 ms where a gap under an unbounded repetition would take seconds. */
enum { sz_regex_forward_budget_k = 16384 };

/** Printable ASCII bytes an end-finding state may leave on and still be searched through
 *  rather than stepped. */
enum { sz_regex_escapes_limit_k = 8 };

/** Distances between positions that the start pass moves by shifts rather than by groups. */
enum { sz_regex_shifts_limit_k = 8 };

/** Every boundary kind: a bit per kind of the unit left of a boundary times four, plus the kind
 *  right of it. */
enum { sz_regex_kinds_all_k = 0xFFFF };

/** A link, state or repetition bound that is absent. */
static sz_u32_t const sz_regex_none_k = 0xFFFFFFFFu;

/** The deepest group nesting a pattern may use, as `regex-syntax` allows. */
enum { sz_regex_depth_limit_k = 250 };

/** The largest product of nested counted repetitions, as RE2 allows. */
enum { sz_regex_repeats_limit_k = 1000 };

/** The input unit past a haystack's last byte or before its first, after the 256 byte values. */
enum { sz_regex_end_of_input_k = 256 };

/** Look-around assertions, each a bit of one set. */
typedef enum sz_regex_look_t {

    /** `\A`, and `^` outside `(?m)`. */
    sz_regex_look_text_start_k = 1,

    /** `\z`, and `$` outside `(?m)`. */
    sz_regex_look_text_end_k = 2,

    /** `^` under `(?m)`: the haystack's start or just after a line feed. */
    sz_regex_look_line_start_k = 4,

    /** `$` under `(?m)`: the haystack's end or just before a line feed. */
    sz_regex_look_line_end_k = 8,

    /** `\b`: one side is an ASCII word byte and the other is not. */
    sz_regex_look_word_boundary_k = 16,

    /** `\B`: both sides are ASCII word bytes, or neither is. */
    sz_regex_look_word_inside_k = 32,

    /** The first lookaround's bit; lookaround @c i of a vocabulary holds bit
     *  `sz_regex_look_lookaround_k << i`. */
    sz_regex_look_lookaround_k = 64,
} sz_regex_look_t;

/** Lookarounds one vocabulary may hold, as many as the look bits past the assertions. */
enum { sz_regex_lookarounds_limit_k = 26 };

/** Lookarounds the DFAs test through their columns, where a wider row costs only cache, as the
 *  halved budgets keep the tables' bytes; a vocabulary with more walks positions and the PikeVM. */
enum { sz_regex_columned_lookarounds_k = 6 };

/** Words of @ref sz_regex_engine_t::looks_bytes: the mask of one-byte lookarounds, then two rows of
 *  256. */
enum { sz_regex_singles_k = 513 };

/** Cells of a row of either DFA: a column per class and the haystack-edge column, a block of them
 *  per set of the lookarounds that may hold at the boundary a step crosses. */
STRINGZILLA_CONSTEXPR sz_size_t sz_regex_stride_(sz_u32_t classes_count, sz_u32_t lookarounds_count) {
    return (sz_size_t)(classes_count + 1) << lookarounds_count;
}

/** The first column of the block a lookaround table entry's @p looks select. */
STRINGZILLA_CONSTEXPR sz_size_t sz_regex_looks_block_(sz_u32_t classes_count, sz_u32_t lookarounds_count,
                                                      sz_u32_t looks) {
    return (sz_size_t)(looks / sz_regex_look_lookaround_k & ((1u << lookarounds_count) - 1)) * (classes_count + 1);
}

/** The flags a group scopes, each a bit of one set. */
typedef enum sz_regex_flags_t {

    /** `(?s)`: the dot also matches a line feed. */
    sz_regex_flag_dot_all_k = 1,

    /** `(?m)`: `^` and `$` also match around line feeds. */
    sz_regex_flag_multi_line_k = 2,

    /** `(?U)`: quantifiers are lazy unless marked with `?`. */
    sz_regex_flag_swap_greed_k = 4,
} sz_regex_flags_t;

/** How wide the matches of a subexpression can be, which decides how a repetition
 *  of it compiles. */
typedef enum sz_regex_width_t {

    /** Every match spans at least one byte, or there is none. */
    sz_regex_width_positive_k = 0,

    /** Some matches can be empty and some cannot. */
    sz_regex_width_mixed_k = 1,

    /** Every match is empty, as for assertions and empty groups. */
    sz_regex_width_zero_k = 2,
} sz_regex_width_t;

/** Which repetitions a quantifier tries first. */
typedef enum sz_regex_greed_t {

    /** The most repetitions first. */
    sz_regex_greedy_k = 0,

    /** The fewest repetitions first. */
    sz_regex_lazy_k = 1,
} sz_regex_greed_t;

/** The kinds of syntax tree nodes. */
typedef enum sz_regex_node_kind_t {

    /** Matches the empty string. */
    sz_regex_node_empty_k = 0,

    /** One codepoint out of a set of ranges; a literal is a class of one. */
    sz_regex_node_class_k = 1,

    /** A look-around assertion. */
    sz_regex_node_look_k = 2,

    /** A bounded or unbounded repetition of one child. */
    sz_regex_node_repeat_k = 3,

    /** Children one after another. */
    sz_regex_node_concat_k = 4,

    /** Children as alternatives, earlier ones preferred. */
    sz_regex_node_alternation_k = 5,

    /** A capturing group around one child, which the forward NFA brackets with capture states. */
    sz_regex_node_group_k = 6,
} sz_regex_node_kind_t;

/** An inclusive range of codepoints. */
typedef struct sz_regex_range_t {

    /** The lowest codepoint of the range. */
    sz_u32_t first;

    /** The highest codepoint of the range. */
    sz_u32_t last;
} sz_regex_range_t;

/** One node of a pattern's syntax tree, all nodes of all patterns sharing one array. */
typedef struct sz_regex_node_t {

    /** What the node matches. */
    sz_regex_node_kind_t kind;

    /** A repetition's preference. */
    sz_regex_greed_t greed;

    /** The widths of this node's matches, computed as it is built. */
    sz_regex_width_t width;

    /** An assertion's bit. */
    sz_u32_t look;

    /** A class's first range, a sequence's first entry in @c children, or a repetition's or a
     *  group's child node. */
    sz_u32_t first;

    /** A class's ranges, a sequence's children, or a group's number within its pattern,
     *  counting from one. */
    sz_u32_t count;

    /** A repetition's least count. */
    sz_u32_t minimum;

    /** A repetition's greatest count, or @ref sz_regex_none_k when unbounded. */
    sz_u32_t maximum;

    /** The largest product of counted repetitions nested here, which RE2 caps. */
    sz_u32_t repeats;
} sz_regex_node_t;

/** One scan over a pattern's bytes, which are valid UTF-8 by the time it starts. */
typedef struct sz_regex_parser_t {

    /** The pattern's bytes. */
    sz_u8_t const *text;

    /** Bytes of the pattern. */
    sz_size_t length;

    /** The next byte to read. */
    sz_size_t position;
} sz_regex_parser_t;

#pragma endregion Syntax

#pragma region Automata

/** A Thompson NFA over bytes, in one direction. */
typedef struct sz_regex_nfa_t {

    /** Every state, and how many there are and fit. */
    sz_regex_nfa_state_t *states;
    sz_size_t states_count;
    sz_size_t states_capacity;

    /** Every link, chained per owning state, and how many there are and fit. */
    sz_regex_nfa_link_t *links;
    sz_size_t links_count;
    sz_size_t links_capacity;

    /** Where every walk enters. */
    sz_u32_t start;

    /** Every assertion some look state tests. */
    sz_u32_t looks;

    /** The assertions reachable from @c start without consuming, which are all a start
     *  state can see. */
    sz_u32_t prefix_looks;
} sz_regex_nfa_t;

/** Which way a subexpression compiles. */
typedef enum sz_regex_direction_t {

    /** Bytes in reading order, as the end-finding automaton consumes them. */
    sz_regex_forward_k = 0,

    /** Bytes last to first and assertions mirrored, as the start-finding
     *  automaton consumes them. */
    sz_regex_reverse_k = 1,
} sz_regex_direction_t;

/** What a match does to the threads of lower priority. */
typedef enum sz_regex_match_kind_t {

    /** Leftmost-first: every thread after the first match in priority order is cut. */
    sz_regex_leftmost_first_k = 0,

    /** Every thread survives a match, which the start-finding automaton needs. */
    sz_regex_all_matches_k = 1,
} sz_regex_match_kind_t;

/** One compiled subexpression: where it is entered, and the state its exit is patched through. */
typedef struct sz_regex_fragment_t {

    /** The state the subexpression is entered through. */
    sz_u32_t start;

    /** The state whose exit the next subexpression is patched into. */
    sz_u32_t end;
} sz_regex_fragment_t;

/** An NFA state set that keeps insertion order, cleared in constant time. */
typedef struct sz_regex_set_t {

    /** The members in insertion order. */
    sz_u32_t *dense;

    /** Per NFA state, its slot in @c dense if it is a member. */
    sz_u32_t *sparse;

    /** Members in @c dense. */
    sz_size_t count;
} sz_regex_set_t;

/**
 *  @brief A subset construction in progress.
 *
 *  A state's key is its flags word, its matched pattern, its NFA state count, then those states in
 *  priority order. The flags word holds the match bit, the "after a word byte" bit, the assertions
 *  known to hold from bit 8, and the assertions its states test from bit 16.
 */
typedef struct sz_regex_dfa_t {

    /** Every state's key, back to back. */
    sz_u32_t *keys;
    sz_size_t keys_count;
    sz_size_t keys_capacity;

    /** Per state, where its key starts. */
    sz_size_t *offsets;
    sz_size_t offsets_capacity;

    /** Per state, a row of next states as @ref sz_regex_stride_ sizes it,
     *  as indices until publication. */
    sz_u32_t *rows;
    sz_size_t rows_capacity;

    /** Open-addressed table of states by key, each slot a state index plus one, zero when free. */
    sz_u32_t *slots;
    sz_size_t slots_count;

    /** States found so far, the dead one first. */
    sz_size_t states_count;

    /** Per byte kind before the walk, the start state's index. */
    sz_u32_t starts[4];

    /** States past which the construction gives up with @c sz_overflow_risk_k. */
    sz_size_t states_budget;
} sz_regex_dfa_t;

/** Words of a DFA key before its NFA states. */
enum { sz_regex_key_header_k = 3 };

/**
 *  @brief One position of the bit-parallel start pass: a class compiled into the reversed NFA,
 *      which it enters at one ranges state and leaves through one empty state whichever
 *      codepoint it consumed.
 *
 *  Position zero is the unanchored loop, which consumes any unit, and position one is the match,
 *  which a step enters where a match starts; neither has a class.
 */
typedef struct sz_regex_position_t {

    /** The ranges state every UTF-8 path of the class leaves from. */
    sz_u32_t entry;

    /** The empty state every path reaches. */
    sz_u32_t exit;

    /** The class node, or @ref sz_regex_none_k. */
    sz_u32_t node;
} sz_regex_position_t;

/** One follow edge between positions, under the boundary kinds and lookarounds the assertions
 *  it crosses need. */
typedef struct sz_regex_follow_t {

    /** The position consumed first, right to left. */
    sz_u32_t from;

    /** The position consumed next. */
    sz_u32_t to;

    /** The boundary kinds the edge can be crossed at. */
    sz_u32_t holds;

    /** The lookarounds it crosses, as look bits, every one of which must hold where
     *  it is crossed. */
    sz_u32_t looks;
} sz_regex_follow_t;

/** The construction state one build threads through its phases, none of which survives the call. */
typedef struct sz_regex_builder_t {

    /** Syntax trees of every pattern, each node after its children. */
    sz_regex_node_t *nodes;
    sz_size_t nodes_count;
    sz_size_t nodes_capacity;

    /** Codepoint ranges of every class, sorted and disjoint per class. */
    sz_regex_range_t *ranges;
    sz_size_t ranges_count;
    sz_size_t ranges_capacity;

    /** Children of every sequence, contiguous per sequence. */
    sz_u32_t *children;
    sz_size_t children_count;
    sz_size_t children_capacity;

    /** Children of the sequences still being parsed, a stack each finished one pops. */
    sz_u32_t *pending;
    sz_size_t pending_count;
    sz_size_t pending_capacity;

    /** Capturing groups of the pattern being parsed, in order, as offset and length pairs of their
     *  names into it, zero long for an unnamed group. */
    sz_size_t *names;
    sz_size_t names_count;
    sz_size_t names_capacity;

    /** Whether patterns parse into their case folds, so that they match the folded
     *  haystacks walks step. */
    sz_case_sensitivity_t case_sensitivity;

    /** Every lookaround of every pattern, numbered as its body closes, and how many there
     *  are and fit. */
    sz_regex_lookaround_t *lookarounds;
    sz_size_t lookarounds_count;
    sz_size_t lookarounds_capacity;

    /** Folds of the class being parsed that span several codepoints, as a count and three
     *  codepoints each, and how many words there are and fit. */
    sz_u32_t *expansions;
    sz_size_t expansions_count;
    sz_size_t expansions_capacity;

    /** Each pattern's root node, @b [patterns_count], the first of three arrays in one block. */
    sz_u32_t *roots;
    sz_size_t patterns_count;

    /** Each pattern's entry into the forward NFA. */
    sz_u32_t *patterns_starts;

    /** Each pattern's first capturing group among all patterns' groups, then their total. */
    sz_u32_t *patterns_groups;

    /** Every capturing group's name, back to back, and how many bytes there are and fit. */
    sz_u8_t *groups_names;
    sz_size_t groups_names_count;
    sz_size_t groups_names_capacity;

    /** Where each group's name ends in @c groups_names, after a leading zero, and how many entries
     *  there are and fit. */
    sz_u32_t *groups_ends;
    sz_size_t groups_ends_count;
    sz_size_t groups_ends_capacity;

    /** The patterns compiled left to right, anchored, and right to left behind
     *  an unanchored prefix. */
    sz_regex_nfa_t forward;
    sz_regex_nfa_t reverse;

    /** Two ordered sets for closures and a stack for their depth-first walk. */
    sz_regex_set_t sets[2];
    sz_u32_t *stack;
    sz_size_t set_capacity;
    sz_size_t stack_capacity;

    /** One DFA key being formed, @c set_capacity plus the header long. */
    sz_u32_t *key;

    /** The two subset constructions. */
    sz_regex_dfa_t forward_dfa;
    sz_regex_dfa_t reverse_dfa;

    /** Per DFA state, its published row offset, sized for the larger automaton. */
    sz_u32_t *published;
    sz_size_t published_capacity;

    /** Per lookaround, three words: where its body's DFA starts in @c bodies_rows plus one, its
     *  start state, and its first accepting state, zeros for a body of one byte or one the PikeVM
     *  walks; then the @c sz_regex_singles_k words of @ref sz_regex_engine_t::looks_bytes. */
    sz_u32_t *bodies;

    /** Rows of every body's DFA, `classes_count + 1` columns each, and how many words there are
     *  and fit. */
    sz_u32_t *bodies_rows;
    sz_size_t bodies_rows_count, bodies_rows_capacity;

    /** Positions of the start pass: the loop, the match, then every class of the reversed NFA
     *  in walk order. */
    sz_regex_position_t *positions;
    sz_size_t positions_count;
    sz_size_t positions_capacity;

    /** Follow edges between positions, one per pair, grouped by source. */
    sz_regex_follow_t *follows;
    sz_size_t follows_count;
    sz_size_t follows_capacity;

    /** Words of one position set, zero unless the start pass runs bit-parallel. */
    sz_size_t words;

    /** Reach rows of the unit classes, @c words each, and how many classes there are and fit. */
    sz_u64_t *reach;
    sz_size_t unit_classes_count;
    sz_size_t reach_capacity;

    /** Shift records, then group records, back to back in words, and how many words there
     *  are and fit. */
    sz_u64_t *records;
    sz_size_t records_count;
    sz_size_t records_capacity;
    sz_size_t shifts_count;
    sz_size_t groups_count;

    /** Every position a walk can hold, @c words long, then per boundary kind the positions
     *  the loop enters. */
    sz_u64_t *all;

    /** Each ASCII codepoint's unit class. */
    sz_u32_t ascii[128];

    /** Past ASCII, pairs of a range's first codepoint and its unit class, and how many words there
     *  are and fit. */
    sz_u32_t *ranges_classes;
    sz_size_t ranges_classes_count;
    sz_size_t ranges_classes_capacity;

    /** Each byte's column, the columns' count, and the first byte of each column. */
    sz_u8_t byte_to_class[256];
    sz_size_t classes_count;
    sz_u8_t representatives[256];

    /** The allocator every buffer above came from, and the stream it allocates on. */
    sz_allocator_t *allocator;
    sz_stream_t stream;
} sz_regex_builder_t;

#pragma endregion Automata

#pragma region Parsing

/** Whether @p byte is an ASCII word byte, which is what `\b` and `\B` test. */
STRINGZILLA_CONSTEXPR sz_bool_t sz_regex_is_word_(sz_u32_t byte) {
    return (sz_bool_t)((byte >= '0' && byte <= '9') || (byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
                       byte == '_');
}

/** The boundary kind of the unit a byte begins or ends: one for a line feed, two for a word byte,
 *  three for another. */
STRINGZILLA_CONSTEXPR sz_size_t sz_regex_byte_kind_(sz_u8_t byte) {
    return byte == '\n' ? 1 : sz_regex_is_word_(byte) ? 2 : 3;
}

/** Hands every buffer the build took back to its allocator. */
STRINGZILLA_INLINE void sz_regex_builder_free_(sz_regex_builder_t *builder) {
    sz_allocator_t *const allocator = builder->allocator;
    sz_stream_t const stream = builder->stream;
    sz_regex_nfa_t *const nfas[2] = {&builder->forward, &builder->reverse};
    sz_regex_dfa_t *const dfas[2] = {&builder->forward_dfa, &builder->reverse_dfa};
    if (builder->nodes)
        allocator->free(builder->nodes, builder->nodes_capacity * sizeof(sz_regex_node_t), allocator->handle, stream);
    if (builder->ranges)
        allocator->free(builder->ranges, builder->ranges_capacity * sizeof(sz_regex_range_t), allocator->handle,
                        stream);
    if (builder->children)
        allocator->free(builder->children, builder->children_capacity * sizeof(sz_u32_t), allocator->handle, stream);
    if (builder->pending)
        allocator->free(builder->pending, builder->pending_capacity * sizeof(sz_u32_t), allocator->handle, stream);
    if (builder->names)
        allocator->free(builder->names, builder->names_capacity * sizeof(sz_size_t), allocator->handle, stream);
    if (builder->expansions)
        allocator->free(builder->expansions, builder->expansions_capacity * sizeof(sz_u32_t), allocator->handle,
                        stream);
    if (builder->lookarounds)
        allocator->free(builder->lookarounds, builder->lookarounds_capacity * sizeof(sz_regex_lookaround_t),
                        allocator->handle, stream);
    if (builder->roots)
        allocator->free(builder->roots, (3 * builder->patterns_count + 1) * sizeof(sz_u32_t), allocator->handle,
                        stream);
    if (builder->groups_names)
        allocator->free(builder->groups_names, builder->groups_names_capacity, allocator->handle, stream);
    if (builder->groups_ends)
        allocator->free(builder->groups_ends, builder->groups_ends_capacity * sizeof(sz_u32_t), allocator->handle,
                        stream);
    for (sz_size_t index = 0; index != 2; ++index) {
        sz_regex_nfa_t *const nfa = nfas[index];
        sz_regex_dfa_t *const dfa = dfas[index];
        if (nfa->states)
            allocator->free(nfa->states, nfa->states_capacity * sizeof(sz_regex_nfa_state_t), allocator->handle,
                            stream);
        if (nfa->links)
            allocator->free(nfa->links, nfa->links_capacity * sizeof(sz_regex_nfa_link_t), allocator->handle, stream);
        if (dfa->keys) allocator->free(dfa->keys, dfa->keys_capacity * sizeof(sz_u32_t), allocator->handle, stream);
        if (dfa->offsets)
            allocator->free(dfa->offsets, dfa->offsets_capacity * sizeof(sz_size_t), allocator->handle, stream);
        if (dfa->rows) allocator->free(dfa->rows, dfa->rows_capacity * sizeof(sz_u32_t), allocator->handle, stream);
        if (dfa->slots) allocator->free(dfa->slots, dfa->slots_count * sizeof(sz_u32_t), allocator->handle, stream);
        if (builder->sets[index].dense)
            allocator->free(builder->sets[index].dense, builder->set_capacity * sizeof(sz_u32_t), allocator->handle,
                            stream);
        if (builder->sets[index].sparse)
            allocator->free(builder->sets[index].sparse, builder->set_capacity * sizeof(sz_u32_t), allocator->handle,
                            stream);
    }
    if (builder->stack)
        allocator->free(builder->stack, builder->stack_capacity * sizeof(sz_u32_t), allocator->handle, stream);
    if (builder->key)
        allocator->free(builder->key, (builder->set_capacity + sz_regex_key_header_k) * sizeof(sz_u32_t),
                        allocator->handle, stream);
    if (builder->published)
        allocator->free(builder->published, builder->published_capacity * sizeof(sz_u32_t), allocator->handle, stream);
    if (builder->bodies)
        allocator->free(builder->bodies, (3 * builder->lookarounds_count + sz_regex_singles_k) * sizeof(sz_u32_t),
                        allocator->handle, stream);
    if (builder->bodies_rows)
        allocator->free(builder->bodies_rows, builder->bodies_rows_capacity * sizeof(sz_u32_t), allocator->handle,
                        stream);
    if (builder->positions)
        allocator->free(builder->positions, builder->positions_capacity * sizeof(sz_regex_position_t),
                        allocator->handle, stream);
    if (builder->follows)
        allocator->free(builder->follows, builder->follows_capacity * sizeof(sz_regex_follow_t), allocator->handle,
                        stream);
    if (builder->reach)
        allocator->free(builder->reach, builder->reach_capacity * sizeof(sz_u64_t), allocator->handle, stream);
    if (builder->records)
        allocator->free(builder->records, builder->records_capacity * sizeof(sz_u64_t), allocator->handle, stream);
    if (builder->all) allocator->free(builder->all, 17 * builder->words * sizeof(sz_u64_t), allocator->handle, stream);
    if (builder->ranges_classes)
        allocator->free(builder->ranges_classes, builder->ranges_classes_capacity * sizeof(sz_u32_t), allocator->handle,
                        stream);
}

/**
 *  @brief Grows an array to hold @p needed elements, doubling, and returns where it now lives.
 *  @return The array, moved or not, or null when the allocation failed, leaving the old
 *      one in place.
 */
STRINGZILLA_INLINE void *sz_regex_grow_(sz_regex_builder_t const *builder, void *data, sz_size_t *capacity,
                                        sz_size_t needed, sz_size_t element_bytes) {
    sz_allocator_t *const allocator = builder->allocator;
    sz_size_t grown_capacity = *capacity ? *capacity : 64;
    void *grown;
    if (needed <= *capacity) return data;
    while (grown_capacity < needed) grown_capacity *= 2;
    grown = allocator->allocate(grown_capacity * element_bytes, allocator->handle, builder->stream);
    if (!grown) return STRINGZILLA_NULL;
    if (data) {
        memcpy(grown, data, *capacity * element_bytes);
        allocator->free(data, *capacity * element_bytes, allocator->handle, builder->stream);
    }
    *capacity = grown_capacity;
    return grown;
}

/** Appends one codepoint range to the class being built at the end of @c ranges. */
STRINGZILLA_INLINE sz_status_t sz_regex_add_range_(sz_regex_builder_t *builder, sz_u32_t first, sz_u32_t last) {
    sz_regex_range_t *const ranges = (sz_regex_range_t *)sz_regex_grow_(
        builder, builder->ranges, &builder->ranges_capacity, builder->ranges_count + 1, sizeof(sz_regex_range_t));
    if (!ranges) return sz_bad_alloc_k;
    builder->ranges = ranges;
    ranges[builder->ranges_count].first = first, ranges[builder->ranges_count].last = last;
    ++builder->ranges_count;
    return sz_success_k;
}

/** Records a fold to several codepoints as an alternative of the class being parsed, once however
 *  many members fold to it. */
STRINGZILLA_INLINE sz_status_t sz_regex_add_expansion_(sz_regex_builder_t *builder, sz_rune_t const *folds,
                                                       sz_size_t count) {
    sz_u32_t record[4] = {(sz_u32_t)count, folds[0], folds[1], count > 2 ? folds[2] : 0};
    sz_u32_t *expansions;
    for (sz_size_t existing = 0; existing != builder->expansions_count; existing += 4)
        if (sz_equal_serial_((sz_cptr_t)(builder->expansions + existing), (sz_cptr_t)record, sizeof(record)))
            return sz_success_k;
    expansions = (sz_u32_t *)sz_regex_grow_(builder, builder->expansions, &builder->expansions_capacity,
                                            builder->expansions_count + 4, sizeof(sz_u32_t));
    if (!expansions) return sz_bad_alloc_k;
    builder->expansions = expansions;
    memcpy(expansions + builder->expansions_count, record, sizeof(record));
    builder->expansions_count += 4;
    return sz_success_k;
}

/**
 *  @brief Appends the codepoints `[first, last]` a pattern spells to the class whose ranges start
 *      at @p begin: as written when cased, else their case folds, a fold to several codepoints
 *      pending as an alternative.
 *
 *  Codepoints under a lead byte that folds nothing are their own folds, so their runs join whole,
 *  and a fold right after the class's last range extends it.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_add_members_(sz_regex_builder_t *builder, sz_size_t begin, sz_u32_t first,
                                                     sz_u32_t last) {
    sz_status_t status = sz_success_k;
    if (builder->case_sensitivity == sz_cased_k) return sz_regex_add_range_(builder, first, last);
    for (sz_u32_t codepoint = first; status == sz_success_k; ++codepoint) {
        sz_u8_t lead[4] = {0};
        sz_rune_t folds[3];
        sz_u32_t run_last = codepoint;
        sz_size_t folds_count = 1;
        sz_rune_encode(codepoint, lead);
        folds[0] = codepoint;
        if (!sz_utf8_lead_may_fold_(lead[0])) {
            run_last |= codepoint < 0x80 ? 0u : codepoint < 0x800 ? 0x3Fu : codepoint < 0x10000 ? 0xFFFu : 0x3FFFFu;
            if (run_last > last) run_last = last;
        }
        else folds_count = sz_unicode_fold_codepoint_(codepoint, folds);
        if (folds_count != 1) status = sz_regex_add_expansion_(builder, folds, folds_count);
        else if (builder->ranges_count > begin && builder->ranges[builder->ranges_count - 1].last + 1 == folds[0])
            builder->ranges[builder->ranges_count - 1].last = folds[0] + (run_last - codepoint);
        else status = sz_regex_add_range_(builder, folds[0], folds[0] + (run_last - codepoint));
        if (run_last == last) break;
        codepoint = run_last;
    }
    return status;
}

/** Appends the members a table spells as pairs of bytes, like "09AZ" for digits and capitals. */
STRINGZILLA_INLINE sz_status_t sz_regex_add_pairs_(sz_regex_builder_t *builder, sz_size_t begin, char const *pairs,
                                                   sz_size_t length) {
    for (sz_size_t index = 0; index + 1 < length; index += 2) {
        sz_status_t const status = sz_regex_add_members_(builder, begin, (sz_u8_t)pairs[index],
                                                         (sz_u8_t)pairs[index + 1]);
        if (status != sz_success_k) return status;
    }
    return sz_success_k;
}

/** Sorts and merges the ranges from @p begin to the end of @c ranges in place. */
STRINGZILLA_INLINE void sz_regex_normalize_ranges_(sz_regex_builder_t *builder, sz_size_t begin) {
    sz_regex_range_t *const ranges = builder->ranges;
    sz_size_t const end = builder->ranges_count;
    sz_size_t merged = begin;
    for (sz_size_t index = begin + 1; index < end; ++index) {
        sz_regex_range_t const range = ranges[index];
        sz_size_t slot = index;
        while (slot > begin && ranges[slot - 1].first > range.first) ranges[slot] = ranges[slot - 1], --slot;
        ranges[slot] = range;
    }
    for (sz_size_t index = begin; index < end; ++index) {
        if (merged > begin && ranges[index].first <= ranges[merged - 1].last + 1) {
            if (ranges[index].last > ranges[merged - 1].last) ranges[merged - 1].last = ranges[index].last;
            continue;
        }
        ranges[merged++] = ranges[index];
    }
    builder->ranges_count = merged;
}

/** Replaces the normalized ranges from @p begin with their complement within the codepoints. */
STRINGZILLA_INLINE sz_status_t sz_regex_negate_ranges_(sz_regex_builder_t *builder, sz_size_t begin) {
    sz_size_t const end = builder->ranges_count;
    sz_u32_t next = 0;
    sz_status_t status;
    for (sz_size_t index = begin; index < end; ++index) {
        sz_regex_range_t const range = builder->ranges[index];
        if (range.first > next && (status = sz_regex_add_range_(builder, next, range.first - 1)) != sz_success_k)
            return status;
        next = range.last + 1;
    }
    if (next <= 0x10FFFF && (status = sz_regex_add_range_(builder, next, 0x10FFFF)) != sz_success_k) return status;
    memmove(builder->ranges + begin, builder->ranges + end, (builder->ranges_count - end) * sizeof(sz_regex_range_t));
    builder->ranges_count = begin + (builder->ranges_count - end);
    return sz_success_k;
}

/** Appends a node, answering its index. */
STRINGZILLA_INLINE sz_status_t sz_regex_add_node_(sz_regex_builder_t *builder, sz_regex_node_kind_t kind,
                                                  sz_regex_width_t width, sz_u32_t *index) {
    sz_regex_node_t *const nodes = (sz_regex_node_t *)sz_regex_grow_(builder, builder->nodes, &builder->nodes_capacity,
                                                                     builder->nodes_count + 1, sizeof(sz_regex_node_t));
    sz_regex_node_t *node;
    if (!nodes) return sz_bad_alloc_k;
    builder->nodes = nodes;
    node = nodes + builder->nodes_count;
    node->kind = kind, node->greed = sz_regex_greedy_k, node->width = width, node->look = 0;
    node->first = 0, node->count = 0, node->minimum = 0, node->maximum = 0, node->repeats = 1;
    *index = (sz_u32_t)builder->nodes_count++;
    return sz_success_k;
}

/** Pushes a finished child onto the stack of the sequence being parsed. */
STRINGZILLA_INLINE sz_status_t sz_regex_push_pending_(sz_regex_builder_t *builder, sz_u32_t node) {
    sz_u32_t *const pending = (sz_u32_t *)sz_regex_grow_(builder, builder->pending, &builder->pending_capacity,
                                                         builder->pending_count + 1, sizeof(sz_u32_t));
    if (!pending) return sz_bad_alloc_k;
    builder->pending = pending;
    pending[builder->pending_count++] = node;
    return sz_success_k;
}

/**
 *  @brief Pops the children pushed since @p base into one sequence node.
 *
 *  A sequence of one child is that child, and an empty concatenation is the empty node, as
 *  `regex-syntax` simplifies them.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_add_sequence_(sz_regex_builder_t *builder, sz_regex_node_kind_t kind,
                                                      sz_size_t base, sz_u32_t *index) {
    sz_size_t const count = builder->pending_count - base;
    sz_regex_width_t width;
    sz_u32_t repeats = 1;
    sz_u32_t *children;
    sz_status_t status;
    if (count == 0) return sz_regex_add_node_(builder, sz_regex_node_empty_k, sz_regex_width_zero_k, index);
    if (count == 1) {
        *index = builder->pending[base];
        builder->pending_count = base;
        return sz_success_k;
    }
    children = (sz_u32_t *)sz_regex_grow_(builder, builder->children, &builder->children_capacity,
                                          builder->children_count + count, sizeof(sz_u32_t));
    if (!children) return sz_bad_alloc_k;
    builder->children = children;
    {
        sz_size_t zero = 0, positive = 0;
        for (sz_size_t offset = 0; offset != count; ++offset) {
            sz_regex_node_t const *child = builder->nodes + builder->pending[base + offset];
            children[builder->children_count + offset] = builder->pending[base + offset];
            zero += child->width == sz_regex_width_zero_k;
            positive += child->width == sz_regex_width_positive_k;
            if (child->repeats > repeats) repeats = child->repeats;
        }
        if (zero == count) width = sz_regex_width_zero_k;
        else if (kind == sz_regex_node_concat_k) width = positive ? sz_regex_width_positive_k : sz_regex_width_mixed_k;
        else width = positive == count ? sz_regex_width_positive_k : sz_regex_width_mixed_k;
    }
    status = sz_regex_add_node_(builder, kind, width, index);
    if (status != sz_success_k) return status;
    builder->nodes[*index].first = (sz_u32_t)builder->children_count;
    builder->nodes[*index].count = (sz_u32_t)count;
    builder->nodes[*index].repeats = repeats;
    builder->children_count += count;
    builder->pending_count = base;
    return sz_success_k;
}

/**
 *  @brief Turns the ranges from @p begin to the end of @c ranges into one normalized class node.
 *
 *  The folds of an uncased class that span several codepoints follow it as alternatives, each a
 *  sequence of single codepoints, so `[sß]` tries @c s before @c ss; a class left with no ranges is
 *  dropped from the alternation. Out of line, as it recurses for those folds and has six callers.
 */
STRINGZILLA_OUTLINED_ sz_status_t sz_regex_add_class_(sz_regex_builder_t *builder, sz_size_t begin, sz_u32_t *index) {
    sz_size_t const base = builder->pending_count, expansions_count = builder->expansions_count;
    sz_status_t status;
    sz_regex_normalize_ranges_(builder, begin);
    builder->expansions_count = 0;
    if (builder->ranges_count != begin || !expansions_count) {
        status = sz_regex_add_node_(builder, sz_regex_node_class_k, sz_regex_width_positive_k, index);
        if (status != sz_success_k) return status;
        builder->nodes[*index].first = (sz_u32_t)begin;
        builder->nodes[*index].count = (sz_u32_t)(builder->ranges_count - begin);
        if (!expansions_count) return sz_success_k;
        if ((status = sz_regex_push_pending_(builder, *index)) != sz_success_k) return status;
    }
    for (sz_size_t expansion = 0; expansion != expansions_count; expansion += 4) {
        sz_size_t const sequence_base = builder->pending_count;
        sz_u32_t node;
        for (sz_u32_t rune = 0; rune != builder->expansions[expansion]; ++rune) {
            sz_size_t const rune_begin = builder->ranges_count;
            sz_u32_t const codepoint = builder->expansions[expansion + 1 + rune];
            if ((status = sz_regex_add_range_(builder, codepoint, codepoint)) != sz_success_k ||
                (status = sz_regex_add_class_(builder, rune_begin, &node)) != sz_success_k ||
                (status = sz_regex_push_pending_(builder, node)) != sz_success_k)
                return status;
        }
        if ((status = sz_regex_add_sequence_(builder, sz_regex_node_concat_k, sequence_base, &node)) != sz_success_k ||
            (status = sz_regex_push_pending_(builder, node)) != sz_success_k)
            return status;
    }
    return sz_regex_add_sequence_(builder, sz_regex_node_alternation_k, base, index);
}

/**
 *  @brief Wraps @p child into a repetition, simplified as `regex-syntax` does.
 *
 *  A child that only matches empty repeats at most once, `{0}` is the empty node, and `{1}` is the
 *  child itself, which then carries @p repeats.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_add_repeat_(sz_regex_builder_t *builder, sz_u32_t child, sz_u32_t minimum,
                                                    sz_u32_t maximum, sz_regex_greed_t greed, sz_u32_t repeats,
                                                    sz_u32_t *index) {
    sz_regex_node_t const sub = builder->nodes[child];
    sz_regex_width_t width;
    sz_status_t status;
    if (sub.width == sz_regex_width_zero_k) {
        if (minimum > 1) minimum = 1;
        if (maximum == sz_regex_none_k || maximum > 1) maximum = 1;
    }
    if (minimum == 0 && maximum == 0) {
        status = sz_regex_add_node_(builder, sz_regex_node_empty_k, sz_regex_width_zero_k, index);
        if (status == sz_success_k) builder->nodes[*index].repeats = repeats;
        return status;
    }
    if (minimum == 1 && maximum == 1) {
        builder->nodes[child].repeats = repeats;
        *index = child;
        return sz_success_k;
    }
    if (sub.width == sz_regex_width_zero_k) width = sz_regex_width_zero_k;
    else if (minimum == 0) width = sz_regex_width_mixed_k;
    else width = sub.width;
    status = sz_regex_add_node_(builder, sz_regex_node_repeat_k, width, index);
    if (status != sz_success_k) return status;
    builder->nodes[*index].greed = greed;
    builder->nodes[*index].first = child;
    builder->nodes[*index].minimum = minimum;
    builder->nodes[*index].maximum = maximum;
    builder->nodes[*index].repeats = repeats;
    return sz_success_k;
}

/** The ASCII digit or hexadecimal value of @p byte, or 16 for neither. */
STRINGZILLA_CONSTEXPR sz_u32_t sz_regex_hex_value_(sz_u8_t byte) {
    return byte >= '0' && byte <= '9'   ? (sz_u32_t)(byte - '0')
           : byte >= 'a' && byte <= 'f' ? (sz_u32_t)(byte - 'a' + 10)
           : byte >= 'A' && byte <= 'F' ? (sz_u32_t)(byte - 'A' + 10)
                                        : 16u;
}

/** Decodes the codepoint at the parser's position and steps past it; the pattern is valid UTF-8. */
STRINGZILLA_INLINE sz_u32_t sz_regex_next_rune_(sz_regex_parser_t *parser) {
    sz_rune_t rune = 0;
    sz_cptr_t const text = (sz_cptr_t)parser->text;
    sz_rune_length_t const length = sz_rune_decode(text + parser->position, text + parser->length, &rune);
    parser->position += length ? (sz_size_t)length : 1;
    return rune;
}

/** Appends the ranges of a Perl class letter, @c d, @c w or @c s in either case,
 *  as RE2's ASCII sets. */
STRINGZILLA_INLINE sz_status_t sz_regex_add_perl_(sz_regex_builder_t *builder, sz_u8_t letter) {
    sz_size_t const begin = builder->ranges_count;
    sz_u8_t const lower = (sz_u8_t)(letter | 0x20);
    sz_status_t status = lower == 'd'   ? sz_regex_add_pairs_(builder, begin, "09", 2)
                         : lower == 'w' ? sz_regex_add_pairs_(builder, begin, "09AZ__az", 8)
                                        : sz_regex_add_pairs_(builder, begin, "\t\n\f\r  ", 6);
    if (status != sz_success_k || letter == lower) return status;
    sz_regex_normalize_ranges_(builder, begin);
    return sz_regex_negate_ranges_(builder, begin);
}

/**
 *  @brief Reads a character escape, the part after the backslash shared inside and outside classes.
 *  @param[out] rune The codepoint it names, when it names one.
 *  @return @c sz_success_k for a codepoint, @c sz_status_unknown_k for anything else, or
 *      @c sz_invalid_syntax_k for a malformed `\x`.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_parse_character_escape_(sz_regex_parser_t *parser, sz_u32_t *rune) {
    sz_u8_t const letter = parser->text[parser->position];
    sz_u32_t value = 0;
    switch (letter) {
    case 'n': *rune = '\n'; break;
    case 't': *rune = '\t'; break;
    case 'r': *rune = '\r'; break;
    case 'f': *rune = '\f'; break;
    case 'v': *rune = '\v'; break;
    case 'a': *rune = '\a'; break;
    case 'x':
        ++parser->position;
        if (parser->position < parser->length && parser->text[parser->position] == '{') {
            sz_size_t digits = 0;
            ++parser->position;
            while (parser->position < parser->length && parser->text[parser->position] != '}') {
                sz_u32_t const digit = sz_regex_hex_value_(parser->text[parser->position]);
                if (digit == 16 || ++digits > 8) return sz_invalid_syntax_k;
                value = value * 16 + digit, ++parser->position;
            }
            if (parser->position == parser->length || digits == 0) return sz_invalid_syntax_k;
        }
        else {
            for (sz_size_t digits = 0; digits != 2; ++digits, ++parser->position) {
                sz_u32_t const digit = parser->position < parser->length
                                           ? sz_regex_hex_value_(parser->text[parser->position])
                                           : 16u;
                if (digit == 16) return sz_invalid_syntax_k;
                value = value * 16 + digit;
            }
            --parser->position;
        }
        if (value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) return sz_invalid_syntax_k;
        *rune = value;
        break;
    default:
        // Punctuation escapes to itself; letters, digits and the `\<` `\>` boundaries are refused
        // by the caller.
        if (letter >= 0x80 || (letter >= '0' && letter <= '9') || (letter >= 'A' && letter <= 'Z') ||
            (letter >= 'a' && letter <= 'z') || letter == '<' || letter == '>')
            return sz_status_unknown_k;
        *rune = letter;
        break;
    }
    ++parser->position;
    return sz_success_k;
}

/** Appends the ranges of a POSIX class at `[:`, which `[[:alpha:]]` and `[[:^alpha:]]` spell. */
STRINGZILLA_INLINE sz_status_t sz_regex_parse_posix_(sz_regex_builder_t *builder, sz_regex_parser_t *parser) {
    static char const *const names[14] = {"alnum", "alpha", "ascii", "blank", "cntrl", "digit", "graph",
                                          "lower", "print", "punct", "space", "upper", "word",  "xdigit"};
    static char const *const pairs[14] = {"09AZaz", "AZaz", "\x00\x7F", "\t\t  ", "\x00\x1F\x7F\x7F",
                                          "09",     "!~",   "az",       " ~",     "!/:@[`{~",
                                          "\t\r  ", "AZ",   "09AZ__az", "09AFaf"};
    static sz_u8_t const pairs_lengths[14] = {6, 4, 2, 4, 4, 2, 2, 2, 2, 8, 4, 2, 8, 6};
    static sz_u8_t const names_lengths[14] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 6};
    sz_size_t const begin = builder->ranges_count;
    sz_size_t name_begin, name_end;
    sz_bool_t negated;
    sz_status_t status;
    parser->position += 2;
    negated = (sz_bool_t)(parser->position < parser->length && parser->text[parser->position] == '^');
    parser->position += negated;
    name_begin = parser->position;
    while (parser->position < parser->length && parser->text[parser->position] >= 'a' &&
           parser->text[parser->position] <= 'z')
        ++parser->position;
    name_end = parser->position;
    if (parser->position + 1 >= parser->length || parser->text[parser->position] != ':' ||
        parser->text[parser->position + 1] != ']')
        return sz_invalid_syntax_k;
    parser->position += 2;
    for (sz_size_t index = 0; index != 14; ++index) {
        sz_size_t const name_length = names_lengths[index];
        if (name_length != name_end - name_begin ||
            !sz_equal_serial_(names[index], (sz_cptr_t)(parser->text + name_begin), name_length))
            continue;
        status = sz_regex_add_pairs_(builder, begin, pairs[index], pairs_lengths[index]);
        if (status != sz_success_k || !negated) return status;
        sz_regex_normalize_ranges_(builder, begin);
        return sz_regex_negate_ranges_(builder, begin);
    }
    return sz_invalid_syntax_k;
}

/**
 *  @brief Reads one class item that must name a single codepoint: a literal or a character escape.
 *  @return @c sz_success_k, or @c sz_status_unknown_k for a Perl class, which only the
 *      caller can take.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_parse_class_rune_(sz_regex_parser_t *parser, sz_u32_t *rune) {
    sz_status_t status;
    if (parser->text[parser->position] != '\\') {
        // Nested classes and `&&`, `--`, `~~` set operators mean something else to `regex-syntax`.
        sz_u8_t const byte = parser->text[parser->position];
        sz_bool_t const doubled = (sz_bool_t)(parser->position + 1 < parser->length &&
                                              parser->text[parser->position + 1] == byte &&
                                              (byte == '&' || byte == '-' || byte == '~'));
        if (byte == '[' || doubled) return sz_invalid_syntax_k;
        *rune = sz_regex_next_rune_(parser);
        return sz_success_k;
    }
    ++parser->position;
    if (parser->position == parser->length) return sz_invalid_syntax_k;
    status = sz_regex_parse_character_escape_(parser, rune);
    if (status != sz_status_unknown_k) return status;
    switch (parser->text[parser->position]) {
    case 'd':
    case 'D':
    case 'w':
    case 'W':
    case 's':
    case 'S': return sz_status_unknown_k;
    default: return sz_invalid_syntax_k;
    }
}

/** Parses a bracketed class from its `[` into one class node. */
STRINGZILLA_INLINE sz_status_t sz_regex_parse_bracket_(sz_regex_builder_t *builder, sz_regex_parser_t *parser,
                                                       sz_u32_t *index) {
    sz_size_t const begin = builder->ranges_count;
    sz_bool_t negated, first = sz_true_k;
    sz_status_t status;
    ++parser->position;
    negated = (sz_bool_t)(parser->position < parser->length && parser->text[parser->position] == '^');
    parser->position += negated;
    for (;;) {
        sz_u32_t low, high;
        if (parser->position == parser->length) return sz_invalid_syntax_k;
        if (parser->text[parser->position] == ']' && !first) {
            ++parser->position;
            break;
        }
        first = sz_false_k;
        if (parser->text[parser->position] == '[' && parser->position + 1 < parser->length &&
            parser->text[parser->position + 1] == ':') {
            status = sz_regex_parse_posix_(builder, parser);
            if (status != sz_success_k) return status;
            continue;
        }
        status = sz_regex_parse_class_rune_(parser, &low);
        if (status == sz_status_unknown_k) {
            status = sz_regex_add_perl_(builder, parser->text[parser->position]);
            ++parser->position;
            if (status != sz_success_k) return status;
            if (parser->position + 1 < parser->length && parser->text[parser->position] == '-' &&
                parser->text[parser->position + 1] != ']')
                return sz_invalid_syntax_k;
            continue;
        }
        if (status != sz_success_k) return status;
        high = low;
        if (parser->position + 1 < parser->length && parser->text[parser->position] == '-' &&
            parser->text[parser->position + 1] != ']') {
            ++parser->position;
            status = sz_regex_parse_class_rune_(parser, &high);
            if (status != sz_success_k || high < low) return sz_invalid_syntax_k;
            if (parser->position + 1 < parser->length && parser->text[parser->position] == '-' &&
                parser->text[parser->position + 1] != ']')
                return sz_invalid_syntax_k;
        }
        status = sz_regex_add_members_(builder, begin, low, high);
        if (status != sz_success_k) return status;
    }
    sz_regex_normalize_ranges_(builder, begin);
    if (negated) {
        builder->expansions_count = 0;
        if ((status = sz_regex_negate_ranges_(builder, begin)) != sz_success_k) return status;
    }
    return sz_regex_add_class_(builder, begin, index);
}

/** Parses a decimal repetition count, refusing anything past RE2's largest. */
STRINGZILLA_INLINE sz_status_t sz_regex_parse_count_(sz_regex_parser_t *parser, sz_u32_t *count) {
    sz_size_t const begin = parser->position;
    sz_u32_t value = 0;
    while (parser->position < parser->length && parser->text[parser->position] >= '0' &&
           parser->text[parser->position] <= '9') {
        value = value * 10 + (sz_u32_t)(parser->text[parser->position++] - '0');
        if (value > sz_regex_repeats_limit_k) return sz_invalid_syntax_k;
    }
    *count = value;
    return parser->position == begin ? sz_invalid_syntax_k : sz_success_k;
}

STRINGZILLA_OUTLINED_ sz_status_t sz_regex_parse_alternation_(sz_regex_builder_t *builder, sz_regex_parser_t *parser,
                                                              sz_u32_t *flags, sz_u32_t depth, sz_u32_t *index);

/** Parses a lookaround from the `=`, `!` or `<` after its `(?` into a look node testing the
 *  lookaround's bit, refusing a capturing group in its body, out of which no walk reads groups. */
STRINGZILLA_OUTLINED_ sz_status_t sz_regex_parse_lookaround_(sz_regex_builder_t *builder, sz_regex_parser_t *parser,
                                                             sz_u32_t flags, sz_u32_t depth, sz_u32_t *index) {
    sz_size_t const names_count = builder->names_count, nested = builder->lookarounds_count;
    sz_bool_t const behind = (sz_bool_t)(parser->text[parser->position] == '<');
    sz_regex_lookaround_t *lookarounds;
    sz_regex_lookaround_kind_t kind;
    sz_u32_t body, level = 1;
    sz_status_t status;
    parser->position += behind;
    kind = (sz_regex_lookaround_kind_t)((behind ? sz_regex_lookbehind_k : sz_regex_lookahead_k) +
                                        (parser->text[parser->position] == '!'));
    ++parser->position;
    if (depth + 1 > sz_regex_depth_limit_k) return sz_invalid_syntax_k;
    status = sz_regex_parse_alternation_(builder, parser, &flags, depth + 1, &body);
    if (status != sz_success_k) return status;
    if (parser->position == parser->length || parser->text[parser->position] != ')' ||
        builder->names_count != names_count || builder->lookarounds_count == sz_regex_lookarounds_limit_k)
        return sz_invalid_syntax_k;
    ++parser->position;
    lookarounds = (sz_regex_lookaround_t *)sz_regex_grow_(builder, builder->lookarounds, &builder->lookarounds_capacity,
                                                          builder->lookarounds_count + 1,
                                                          sizeof(sz_regex_lookaround_t));
    if (!lookarounds) return sz_bad_alloc_k;
    builder->lookarounds = lookarounds;
    for (sz_size_t inner = nested; inner != builder->lookarounds_count; ++inner)
        if (lookarounds[inner].level >= level) level = lookarounds[inner].level + 1;
    lookarounds[builder->lookarounds_count].start = body, lookarounds[builder->lookarounds_count].kind = kind;
    lookarounds[builder->lookarounds_count].level = level;
    status = sz_regex_add_node_(builder, sz_regex_node_look_k, sz_regex_width_zero_k, index);
    if (status == sz_success_k)
        builder->nodes[*index].look = (sz_u32_t)sz_regex_look_lookaround_k << builder->lookarounds_count++;
    return status;
}

/**
 *  @brief Parses a group from its `(`: capturing, non-capturing, named, a flag group,
 *      or a lookaround.
 *  @param[out] index The group's node, or @ref sz_regex_none_k for a flag group, which only
 *      changes @p flags for the rest of the enclosing group.
 *
 *  Capturing groups, named or not, are numbered from one by their opening parenthesis.
 */
STRINGZILLA_OUTLINED_ sz_status_t sz_regex_parse_group_(sz_regex_builder_t *builder, sz_regex_parser_t *parser,
                                                        sz_u32_t *flags, sz_u32_t depth, sz_u32_t *index) {
    sz_u8_t const *const text = parser->text;
    sz_u32_t scoped = *flags, group = 0;
    sz_size_t name_begin = ++parser->position, name_length = 0;
    sz_status_t status;
    *index = sz_regex_none_k;
    if (parser->position < parser->length && text[parser->position] == '?') {
        ++parser->position;
        if (parser->position == parser->length) return sz_invalid_syntax_k;
        if (text[parser->position] == '=' || text[parser->position] == '!' ||
            (text[parser->position] == '<' && parser->position + 1 < parser->length &&
             (text[parser->position + 1] == '=' || text[parser->position + 1] == '!')))
            return sz_regex_parse_lookaround_(builder, parser, scoped, depth, index);
        if (text[parser->position] == 'P' || text[parser->position] == '<') {
            if (text[parser->position] == 'P') ++parser->position;
            if (parser->position == parser->length || text[parser->position] != '<') return sz_invalid_syntax_k;
            name_begin = ++parser->position;
            while (parser->position < parser->length && text[parser->position] != '>') {
                sz_u8_t const byte = text[parser->position];
                sz_bool_t const letter = (sz_bool_t)((byte | 0x20) >= 'a' && (byte | 0x20) <= 'z');
                sz_bool_t const digit = (sz_bool_t)(byte >= '0' && byte <= '9');
                if (!(letter || byte == '_' || (digit && parser->position != name_begin))) return sz_invalid_syntax_k;
                ++parser->position;
            }
            if (parser->position == parser->length || parser->position == name_begin) return sz_invalid_syntax_k;
            name_length = parser->position - name_begin;
            for (sz_size_t name = 0; name != builder->names_count; name += 2)
                if (builder->names[name + 1] == name_length &&
                    sz_equal_serial_((sz_cptr_t)(text + builder->names[name]), (sz_cptr_t)(text + name_begin),
                                     name_length))
                    return sz_invalid_syntax_k;
            ++parser->position;
        }
        else if (text[parser->position] == ':') {
            ++parser->position;
            group = sz_regex_none_k;
        }
        else {
            sz_u32_t seen = 0;
            sz_bool_t negating = sz_false_k, after_negation = sz_false_k;
            for (;; ++parser->position) {
                sz_u32_t flag;
                if (parser->position == parser->length) return sz_invalid_syntax_k;
                switch (text[parser->position]) {
                case 's': flag = sz_regex_flag_dot_all_k; break;
                case 'm': flag = sz_regex_flag_multi_line_k; break;
                case 'U': flag = sz_regex_flag_swap_greed_k; break;
                case '-':
                    if (negating) return sz_invalid_syntax_k;
                    negating = sz_true_k;
                    continue;
                case ':':
                case ')': flag = 0; break;
                default: return sz_invalid_syntax_k;
                }
                if (!flag) break;
                if (seen & flag) return sz_invalid_syntax_k;
                seen |= flag, after_negation = negating;
                scoped = negating ? scoped & ~flag : scoped | flag;
            }
            if (!seen || (negating && !after_negation)) return sz_invalid_syntax_k;
            if (text[parser->position++] == ')') {
                *flags = scoped;
                return sz_success_k;
            }
            group = sz_regex_none_k;
        }
    }
    if (group == 0) {
        sz_size_t *const names = (sz_size_t *)sz_regex_grow_(builder, builder->names, &builder->names_capacity,
                                                             builder->names_count + 2, sizeof(sz_size_t));
        if (!names) return sz_bad_alloc_k;
        builder->names = names;
        names[builder->names_count++] = name_begin;
        names[builder->names_count++] = name_length;
        group = (sz_u32_t)(builder->names_count / 2);
    }
    if (depth + 1 > sz_regex_depth_limit_k) return sz_invalid_syntax_k;
    status = sz_regex_parse_alternation_(builder, parser, &scoped, depth + 1, index);
    if (status != sz_success_k) return status;
    if (parser->position == parser->length || text[parser->position] != ')') return sz_invalid_syntax_k;
    ++parser->position;
    if (group == sz_regex_none_k) return sz_success_k;
    {
        sz_u32_t const child = *index;
        status = sz_regex_add_node_(builder, sz_regex_node_group_k, builder->nodes[child].width, index);
        if (status != sz_success_k) return status;
        builder->nodes[*index].first = child, builder->nodes[*index].count = group;
        builder->nodes[*index].repeats = builder->nodes[child].repeats;
    }
    return sz_success_k;
}

/** Parses an escape outside a class from its backslash: a codepoint, a Perl class,
 *  or an assertion. */
STRINGZILLA_INLINE sz_status_t sz_regex_parse_escape_(sz_regex_builder_t *builder, sz_regex_parser_t *parser,
                                                      sz_u32_t *index) {
    sz_size_t const begin = builder->ranges_count;
    sz_u32_t rune, look = 0;
    sz_status_t status;
    ++parser->position;
    if (parser->position == parser->length) return sz_invalid_syntax_k;
    status = sz_regex_parse_character_escape_(parser, &rune);
    if (status == sz_success_k) {
        status = sz_regex_add_members_(builder, begin, rune, rune);
        return status == sz_success_k ? sz_regex_add_class_(builder, begin, index) : status;
    }
    if (status != sz_status_unknown_k) return status;
    switch (parser->text[parser->position]) {
    case 'd':
    case 'D':
    case 'w':
    case 'W':
    case 's':
    case 'S':
        status = sz_regex_add_perl_(builder, parser->text[parser->position++]);
        return status == sz_success_k ? sz_regex_add_class_(builder, begin, index) : status;
    case 'b': look = sz_regex_look_word_boundary_k; break;
    case 'B': look = sz_regex_look_word_inside_k; break;
    case 'A': look = sz_regex_look_text_start_k; break;
    case 'z': look = sz_regex_look_text_end_k; break;
    default: return sz_invalid_syntax_k;
    }
    ++parser->position;
    // `regex-syntax` reads `\b{start}` and its kin where RE2 reads a repetition.
    if (parser->position < parser->length && parser->text[parser->position] == '{' &&
        (look == sz_regex_look_word_boundary_k || look == sz_regex_look_word_inside_k))
        return sz_invalid_syntax_k;
    status = sz_regex_add_node_(builder, sz_regex_node_look_k, sz_regex_width_zero_k, index);
    if (status == sz_success_k) builder->nodes[*index].look = look;
    return status;
}

/** Parses the quantifiers after an item, refusing a second one,
 *  which RE2 and `regex-syntax` read apart. */
STRINGZILLA_INLINE sz_status_t sz_regex_parse_quantifier_(sz_regex_builder_t *builder, sz_regex_parser_t *parser,
                                                          sz_u32_t flags, sz_u32_t *index) {
    sz_u8_t const *const text = parser->text;
    sz_u32_t minimum, maximum, repeats = builder->nodes[*index].repeats;
    sz_regex_greed_t greed;
    sz_status_t status;
    if (parser->position == parser->length) return sz_success_k;
    switch (text[parser->position]) {
    case '*': minimum = 0, maximum = sz_regex_none_k; break;
    case '+': minimum = 1, maximum = sz_regex_none_k; break;
    case '?': minimum = 0, maximum = 1; break;
    case '{':
        ++parser->position;
        if ((status = sz_regex_parse_count_(parser, &minimum)) != sz_success_k) return status;
        maximum = minimum;
        if (parser->position < parser->length && text[parser->position] == ',') {
            ++parser->position;
            if (parser->position < parser->length && text[parser->position] == '}') maximum = sz_regex_none_k;
            else if ((status = sz_regex_parse_count_(parser, &maximum)) != sz_success_k) return status;
        }
        if (parser->position == parser->length || text[parser->position] != '}' || maximum < minimum)
            return sz_invalid_syntax_k;
        // RE2 caps the product of nested counted repetitions, counting an unbounded one's minimum.
        if (maximum != sz_regex_none_k ? maximum : minimum) {
            if ((sz_u64_t)repeats * (maximum != sz_regex_none_k ? maximum : minimum) > sz_regex_repeats_limit_k)
                return sz_invalid_syntax_k;
            repeats *= maximum != sz_regex_none_k ? maximum : minimum;
        }
        break;
    default: return sz_success_k;
    }
    ++parser->position;
    greed = (flags & sz_regex_flag_swap_greed_k) ? sz_regex_lazy_k : sz_regex_greedy_k;
    if (parser->position < parser->length && text[parser->position] == '?') {
        greed = greed == sz_regex_lazy_k ? sz_regex_greedy_k : sz_regex_lazy_k;
        ++parser->position;
    }
    if (parser->position < parser->length && (text[parser->position] == '*' || text[parser->position] == '+' ||
                                              text[parser->position] == '?' || text[parser->position] == '{'))
        return sz_invalid_syntax_k;
    return sz_regex_add_repeat_(builder, *index, minimum, maximum, greed, repeats, index);
}

/** Parses items up to the next `|` or `)` into one concatenation. */
STRINGZILLA_OUTLINED_ sz_status_t sz_regex_parse_concat_(sz_regex_builder_t *builder, sz_regex_parser_t *parser,
                                                         sz_u32_t *flags, sz_u32_t depth, sz_u32_t *index) {
    sz_size_t const base = builder->pending_count;
    sz_u8_t const *const text = parser->text;
    sz_status_t status;
    while (parser->position < parser->length && text[parser->position] != '|' && text[parser->position] != ')') {
        sz_size_t const begin = builder->ranges_count;
        sz_u32_t item = sz_regex_none_k, rune;
        switch (text[parser->position]) {
        case '(': status = sz_regex_parse_group_(builder, parser, flags, depth, &item); break;
        case '[': status = sz_regex_parse_bracket_(builder, parser, &item); break;
        case '\\': status = sz_regex_parse_escape_(builder, parser, &item); break;
        case '*':
        case '+':
        case '?':
        case '{': status = sz_invalid_syntax_k; break;
        case '.':
            ++parser->position;
            status = (*flags & sz_regex_flag_dot_all_k) ? sz_regex_add_range_(builder, 0, 0x10FFFF)
                                                        : sz_regex_add_range_(builder, 0, 0x09);
            if (status == sz_success_k && !(*flags & sz_regex_flag_dot_all_k))
                status = sz_regex_add_range_(builder, 0x0B, 0x10FFFF);
            if (status == sz_success_k) status = sz_regex_add_class_(builder, begin, &item);
            break;
        case '^':
        case '$': {
            sz_bool_t const multi_line = (sz_bool_t)((*flags & sz_regex_flag_multi_line_k) != 0);
            sz_u32_t const look = text[parser->position] == '^'
                                      ? (multi_line ? sz_regex_look_line_start_k : sz_regex_look_text_start_k)
                                      : (multi_line ? sz_regex_look_line_end_k : sz_regex_look_text_end_k);
            ++parser->position;
            status = sz_regex_add_node_(builder, sz_regex_node_look_k, sz_regex_width_zero_k, &item);
            if (status == sz_success_k) builder->nodes[item].look = look;
            break;
        }
        default:
            rune = sz_regex_next_rune_(parser);
            status = sz_regex_add_members_(builder, begin, rune, rune);
            if (status == sz_success_k) status = sz_regex_add_class_(builder, begin, &item);
            break;
        }
        if (status != sz_success_k) return status;
        if (item == sz_regex_none_k) {
            if (parser->position < parser->length && (text[parser->position] == '*' || text[parser->position] == '+' ||
                                                      text[parser->position] == '?' || text[parser->position] == '{'))
                return sz_invalid_syntax_k;
            continue;
        }
        if ((status = sz_regex_parse_quantifier_(builder, parser, *flags, &item)) != sz_success_k) return status;
        if ((status = sz_regex_push_pending_(builder, item)) != sz_success_k) return status;
    }
    return sz_regex_add_sequence_(builder, sz_regex_node_concat_k, base, index);
}

STRINGZILLA_OUTLINED_ sz_status_t sz_regex_parse_alternation_(sz_regex_builder_t *builder, sz_regex_parser_t *parser,
                                                              sz_u32_t *flags, sz_u32_t depth, sz_u32_t *index) {
    sz_size_t const base = builder->pending_count;
    for (;;) {
        sz_u32_t branch;
        sz_status_t status = sz_regex_parse_concat_(builder, parser, flags, depth, &branch);
        if (status == sz_success_k) status = sz_regex_push_pending_(builder, branch);
        if (status != sz_success_k) return status;
        if (parser->position == parser->length || parser->text[parser->position] != '|') break;
        ++parser->position;
    }
    return sz_regex_add_sequence_(builder, sz_regex_node_alternation_k, base, index);
}

/** Parses one whole pattern into a syntax tree, refusing malformed UTF-8 and unbalanced groups. */
STRINGZILLA_INLINE sz_status_t sz_regex_parse_pattern_(sz_regex_builder_t *builder, sz_cptr_t pattern, sz_size_t length,
                                                       sz_u32_t *root) {
    sz_regex_parser_t parser;
    sz_u32_t flags = 0;
    sz_status_t status;
    for (sz_size_t position = 0; position < length;) {
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode(pattern + position, pattern + length, &rune);
        if (rune_length == sz_rune_invalid_k) return sz_invalid_syntax_k;
        position += (sz_size_t)rune_length;
    }
    parser.text = (sz_u8_t const *)pattern, parser.length = length, parser.position = 0;
    builder->names_count = 0;
    status = sz_regex_parse_alternation_(builder, &parser, &flags, 0, root);
    if (status == sz_success_k && parser.position != length) status = sz_invalid_syntax_k;
    return status;
}

#pragma endregion Parsing

#pragma region Compilation

/** Inserts @p state into @p set unless it is there already, answering whether it was new. */
STRINGZILLA_INLINE sz_bool_t sz_regex_set_insert_(sz_regex_set_t *set, sz_u32_t state) {
    sz_u32_t const slot = set->sparse[state];
    if (slot < set->count && set->dense[slot] == state) return sz_false_k;
    set->sparse[state] = (sz_u32_t)set->count, set->dense[set->count++] = state;
    return sz_true_k;
}

/** Appends one NFA state with no links and no target yet. */
STRINGZILLA_INLINE sz_status_t sz_regex_nfa_add_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa,
                                                 sz_regex_nfa_kind_t kind, sz_u32_t *state) {
    sz_regex_nfa_state_t *states;
    if (nfa->states_count + 1 >= sz_regex_none_k) return sz_overflow_risk_k;
    states = (sz_regex_nfa_state_t *)sz_regex_grow_(builder, nfa->states, &nfa->states_capacity, nfa->states_count + 1,
                                                    sizeof(sz_regex_nfa_state_t));
    if (!states) return sz_bad_alloc_k;
    nfa->states = states;
    states[nfa->states_count].kind = kind, states[nfa->states_count].look = 0;
    states[nfa->states_count].target = sz_regex_none_k, states[nfa->states_count].slot = sz_regex_none_k;
    states[nfa->states_count].first_link = sz_regex_none_k, states[nfa->states_count].last_link = sz_regex_none_k;
    *state = (sz_u32_t)nfa->states_count++;
    return sz_success_k;
}

/** Adds a link from @p from, appended, or prepended for a lazy union. */
STRINGZILLA_INLINE sz_status_t sz_regex_nfa_link_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa, sz_u32_t from,
                                                  sz_u32_t target, sz_u8_t low, sz_u8_t high) {
    sz_regex_nfa_link_t *links;
    sz_regex_nfa_state_t *state;
    sz_u32_t link;
    if (nfa->links_count + 1 >= sz_regex_none_k) return sz_overflow_risk_k;
    links = (sz_regex_nfa_link_t *)sz_regex_grow_(builder, nfa->links, &nfa->links_capacity, nfa->links_count + 1,
                                                  sizeof(sz_regex_nfa_link_t));
    if (!links) return sz_bad_alloc_k;
    nfa->links = links;
    link = (sz_u32_t)nfa->links_count++;
    state = nfa->states + from;
    links[link].target = target, links[link].low = low, links[link].high = high;
    if (state->kind == sz_regex_nfa_union_lazy_k) {
        links[link].next = state->first_link;
        state->first_link = link;
        if (state->last_link == sz_regex_none_k) state->last_link = link;
    }
    else {
        links[link].next = sz_regex_none_k;
        if (state->last_link == sz_regex_none_k) state->first_link = link;
        else links[state->last_link].next = link;
        state->last_link = link;
    }
    return sz_success_k;
}

/** Points the exit of @p from at @p to: an empty or look state's target,
 *  or one more union alternative. */
STRINGZILLA_INLINE sz_status_t sz_regex_nfa_patch_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa, sz_u32_t from,
                                                   sz_u32_t to) {
    sz_regex_nfa_kind_t const kind = nfa->states[from].kind;
    if (kind == sz_regex_nfa_empty_k || kind == sz_regex_nfa_look_k) nfa->states[from].target = to;
    else if (kind == sz_regex_nfa_union_k || kind == sz_regex_nfa_union_lazy_k)
        return sz_regex_nfa_link_(builder, nfa, from, to, 0, 0);
    return sz_success_k;
}

/** Splits one codepoint range into UTF-8 byte-range sequences,
 *  as `regex-syntax`'s @c Utf8Sequences does. */
STRINGZILLA_INLINE sz_size_t sz_regex_utf8_sequences_(sz_u32_t first, sz_u32_t last, sz_u8_t lows[][4],
                                                      sz_u8_t highs[][4], sz_u8_t *lengths) {
    sz_u32_t stack_first[16], stack_last[16];
    sz_size_t stack_count = 0, count = 0;
    stack_first[stack_count] = first, stack_last[stack_count++] = last;
    while (stack_count) {
        sz_u32_t start = stack_first[--stack_count], end = stack_last[stack_count];
        for (;;) {
            sz_u32_t split = 0;
            if (start < 0xE000 && end > 0xD7FF) {
                stack_first[stack_count] = 0xE000, stack_last[stack_count++] = end;
                end = 0xD7FF;
                continue;
            }
            if (start > end) break;
            if (start <= 0x7F && 0x7F < end) split = 0x7F;
            else if (start <= 0x7FF && 0x7FF < end) split = 0x7FF;
            else if (start <= 0xFFFF && 0xFFFF < end) split = 0xFFFF;
            if (split) {
                stack_first[stack_count] = split + 1, stack_last[stack_count++] = end;
                end = split;
                continue;
            }
            if (end <= 0x7F) {
                lows[count][0] = (sz_u8_t)start, highs[count][0] = (sz_u8_t)end, lengths[count++] = 1;
                break;
            }
            for (sz_u32_t shift = 6; shift <= 18 && !split; shift += 6) {
                sz_u32_t const mask = (1u << shift) - 1;
                if ((start & ~mask) == (end & ~mask)) continue;
                if (start & mask) {
                    stack_first[stack_count] = (start | mask) + 1, stack_last[stack_count++] = end;
                    end = start | mask, split = 1;
                }
                else if ((end & mask) != mask) {
                    stack_first[stack_count] = end & ~mask, stack_last[stack_count++] = end;
                    end = (end & ~mask) - 1, split = 1;
                }
            }
            if (split) continue;
            lengths[count] = (sz_u8_t)sz_rune_encode(start, lows[count]);
            sz_rune_encode(end, highs[count]);
            ++count;
            break;
        }
    }
    return count;
}

/**
 *  @brief Compiles a class into a ranges state whose byte paths spell each member's UTF-8 encoding.
 *
 *  Paths share their tails through a per-class cache keyed by range and target, so the bytes after
 *  a lead byte lead to one state whichever lead byte it was. Reversed, a path spells the encoding
 *  last byte first.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_compile_class_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa,
                                                       sz_regex_node_t const *node, sz_regex_direction_t direction,
                                                       sz_regex_fragment_t *fragment) {
    sz_u8_t lows[32][4], highs[32][4], lengths[32];
    sz_u32_t root, end, cache_begin;
    sz_status_t status;
    if ((status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &end)) != sz_success_k) return status;
    if ((status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_ranges_k, &root)) != sz_success_k) return status;
    cache_begin = root + 1;
    for (sz_u32_t range = 0; range != node->count; ++range) {
        sz_regex_range_t const span = builder->ranges[node->first + range];
        sz_size_t const sequences = sz_regex_utf8_sequences_(span.first, span.last, lows, highs, lengths);
        for (sz_size_t sequence = 0; sequence != sequences; ++sequence) {
            sz_size_t const length = lengths[sequence];
            sz_u32_t target = end;
            for (sz_size_t step = length - 1; step != 0; --step) {
                sz_size_t const byte = direction == sz_regex_forward_k ? step : length - 1 - step;
                sz_u8_t const low = lows[sequence][byte], high = highs[sequence][byte];
                sz_u32_t cached = sz_regex_none_k;
                for (sz_u32_t candidate = cache_begin; candidate < nfa->states_count && cached == sz_regex_none_k;
                     ++candidate) {
                    sz_regex_nfa_link_t const *link = nfa->links + nfa->states[candidate].first_link;
                    if (link->target == target && link->low == low && link->high == high) cached = candidate;
                }
                if (cached == sz_regex_none_k) {
                    if ((status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_ranges_k, &cached)) != sz_success_k ||
                        (status = sz_regex_nfa_link_(builder, nfa, cached, target, low, high)) != sz_success_k)
                        return status;
                }
                target = cached;
            }
            {
                sz_size_t const byte = direction == sz_regex_forward_k ? 0 : length - 1;
                status = sz_regex_nfa_link_(builder, nfa, root, target, lows[sequence][byte], highs[sequence][byte]);
                if (status != sz_success_k) return status;
            }
        }
    }
    fragment->start = root, fragment->end = end;
    return sz_success_k;
}

/** Adds a union whose alternatives are tried in the order @p greed asks for. */
STRINGZILLA_INLINE sz_status_t sz_regex_nfa_union_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa,
                                                   sz_regex_greed_t greed, sz_u32_t *state) {
    return sz_regex_nfa_add_(builder, nfa, greed == sz_regex_lazy_k ? sz_regex_nfa_union_lazy_k : sz_regex_nfa_union_k,
                             state);
}

/** Appends a position of the start pass, entered at @p entry and left at @p exit. */
STRINGZILLA_INLINE sz_status_t sz_regex_add_position_(sz_regex_builder_t *builder, sz_u32_t entry, sz_u32_t exit,
                                                      sz_u32_t node) {
    sz_regex_position_t *const positions = (sz_regex_position_t *)sz_regex_grow_(
        builder, builder->positions, &builder->positions_capacity, builder->positions_count + 1,
        sizeof(sz_regex_position_t));
    if (!positions) return sz_bad_alloc_k;
    builder->positions = positions;
    positions[builder->positions_count].entry = entry, positions[builder->positions_count].exit = exit;
    positions[builder->positions_count].node = node;
    ++builder->positions_count;
    return sz_success_k;
}

STRINGZILLA_OUTLINED_ sz_status_t sz_regex_compile_node_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa,
                                                         sz_u32_t index, sz_regex_direction_t direction,
                                                         sz_regex_fragment_t *fragment);

/** Compiles @p count copies of a node one after another, or an empty state for none. */
STRINGZILLA_OUTLINED_ sz_status_t sz_regex_compile_copies_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa,
                                                           sz_u32_t index, sz_u32_t count,
                                                           sz_regex_direction_t direction,
                                                           sz_regex_fragment_t *fragment) {
    sz_status_t status;
    if (count == 0) {
        status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &fragment->start);
        fragment->end = fragment->start;
        return status;
    }
    if ((status = sz_regex_compile_node_(builder, nfa, index, direction, fragment)) != sz_success_k) return status;
    for (sz_u32_t copy = 1; copy != count; ++copy) {
        sz_regex_fragment_t next;
        if ((status = sz_regex_compile_node_(builder, nfa, index, direction, &next)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, fragment->end, next.start)) != sz_success_k)
            return status;
        fragment->end = next.end;
    }
    return sz_success_k;
}

/**
 *  @brief Compiles a repetition the way `regex-automata`'s Thompson compiler does, which fixes how
 *      leftmost-first priority treats empty iterations.
 *
 *  An unbounded repetition of a child that can match empty compiles as `(x+)?` rather than a loop
 *  around @c x, and a bounded one as a chain of optional copies sharing one exit.
 */
STRINGZILLA_OUTLINED_ sz_status_t sz_regex_compile_repeat_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa,
                                                           sz_regex_node_t const *node, sz_regex_direction_t direction,
                                                           sz_regex_fragment_t *fragment) {
    sz_u32_t const child = node->first, minimum = node->minimum, maximum = node->maximum;
    sz_regex_greed_t const greed = node->greed;
    sz_regex_fragment_t body;
    sz_u32_t loop, exit;
    sz_status_t status;
    if (minimum == 0 && maximum == 1) {
        if ((status = sz_regex_nfa_union_(builder, nfa, greed, &loop)) != sz_success_k ||
            (status = sz_regex_compile_node_(builder, nfa, child, direction, &body)) != sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &exit)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, body.start)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, exit)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, body.end, exit)) != sz_success_k)
            return status;
        fragment->start = loop, fragment->end = exit;
        return sz_success_k;
    }
    if (maximum == sz_regex_none_k && minimum == 0) {
        if (builder->nodes[child].width == sz_regex_width_positive_k) {
            if ((status = sz_regex_nfa_union_(builder, nfa, greed, &loop)) != sz_success_k ||
                (status = sz_regex_compile_node_(builder, nfa, child, direction, &body)) != sz_success_k ||
                (status = sz_regex_nfa_patch_(builder, nfa, loop, body.start)) != sz_success_k ||
                (status = sz_regex_nfa_patch_(builder, nfa, body.end, loop)) != sz_success_k)
                return status;
            fragment->start = loop, fragment->end = loop;
            return sz_success_k;
        }
        sz_u32_t question;
        if ((status = sz_regex_compile_node_(builder, nfa, child, direction, &body)) != sz_success_k ||
            (status = sz_regex_nfa_union_(builder, nfa, greed, &loop)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, body.end, loop)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, body.start)) != sz_success_k ||
            (status = sz_regex_nfa_union_(builder, nfa, greed, &question)) != sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &exit)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, question, body.start)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, question, exit)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, exit)) != sz_success_k)
            return status;
        fragment->start = question, fragment->end = exit;
        return sz_success_k;
    }
    if (maximum == sz_regex_none_k) {
        sz_regex_fragment_t prefix;
        if (minimum > 1 &&
            (status = sz_regex_compile_copies_(builder, nfa, child, minimum - 1, direction, &prefix)) != sz_success_k)
            return status;
        if ((status = sz_regex_compile_node_(builder, nfa, child, direction, &body)) != sz_success_k ||
            (status = sz_regex_nfa_union_(builder, nfa, greed, &loop)) != sz_success_k ||
            (minimum > 1 && (status = sz_regex_nfa_patch_(builder, nfa, prefix.end, body.start)) != sz_success_k) ||
            (status = sz_regex_nfa_patch_(builder, nfa, body.end, loop)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, body.start)) != sz_success_k)
            return status;
        fragment->start = minimum > 1 ? prefix.start : body.start, fragment->end = loop;
        return sz_success_k;
    }
    if ((status = sz_regex_compile_copies_(builder, nfa, child, minimum, direction, fragment)) != sz_success_k ||
        minimum == maximum)
        return status;
    if ((status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &exit)) != sz_success_k) return status;
    for (sz_u32_t copy = minimum; copy != maximum; ++copy) {
        if ((status = sz_regex_nfa_union_(builder, nfa, greed, &loop)) != sz_success_k ||
            (status = sz_regex_compile_node_(builder, nfa, child, direction, &body)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, fragment->end, loop)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, body.start)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, exit)) != sz_success_k)
            return status;
        fragment->end = body.end;
    }
    status = sz_regex_nfa_patch_(builder, nfa, fragment->end, exit);
    fragment->end = exit;
    return status;
}

/** The same assertion seen from the other side, which a reversed pattern tests. */
STRINGZILLA_CONSTEXPR sz_u32_t sz_regex_reverse_look_(sz_u32_t look) {
    return look == sz_regex_look_text_start_k   ? (sz_u32_t)sz_regex_look_text_end_k
           : look == sz_regex_look_text_end_k   ? (sz_u32_t)sz_regex_look_text_start_k
           : look == sz_regex_look_line_start_k ? (sz_u32_t)sz_regex_look_line_end_k
           : look == sz_regex_look_line_end_k   ? (sz_u32_t)sz_regex_look_line_start_k
                                                : look;
}

/** Compiles one syntax tree node into NFA states, children in reverse order for
 *  a reversed pattern. */
STRINGZILLA_OUTLINED_ sz_status_t sz_regex_compile_node_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa,
                                                         sz_u32_t index, sz_regex_direction_t direction,
                                                         sz_regex_fragment_t *fragment) {
    sz_regex_node_t const node = builder->nodes[index];
    sz_status_t status;
    switch (node.kind) {
    case sz_regex_node_empty_k:
        status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &fragment->start);
        fragment->end = fragment->start;
        return status;
    case sz_regex_node_class_k:
        status = sz_regex_compile_class_(builder, nfa, &node, direction, fragment);
        if (status != sz_success_k || direction == sz_regex_forward_k) return status;
        return sz_regex_add_position_(builder, fragment->start, fragment->end, index);
    case sz_regex_node_look_k:
        status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_look_k, &fragment->start);
        if (status != sz_success_k) return status;
        nfa->states[fragment->start].look = direction == sz_regex_forward_k ? node.look
                                                                            : sz_regex_reverse_look_(node.look);
        nfa->looks |= nfa->states[fragment->start].look;
        fragment->end = fragment->start;
        return sz_success_k;
    case sz_regex_node_repeat_k: return sz_regex_compile_repeat_(builder, nfa, &node, direction, fragment);
    case sz_regex_node_group_k: {
        sz_regex_fragment_t body;
        sz_u32_t open, close;
        if (direction == sz_regex_reverse_k)
            return sz_regex_compile_node_(builder, nfa, node.first, direction, fragment);
        if ((status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &open)) != sz_success_k ||
            (status = sz_regex_compile_node_(builder, nfa, node.first, direction, &body)) != sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &close)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, open, body.start)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, body.end, close)) != sz_success_k)
            return status;
        nfa->states[open].slot = 2 * node.count, nfa->states[close].slot = 2 * node.count + 1;
        fragment->start = open, fragment->end = close;
        return sz_success_k;
    }
    case sz_regex_node_concat_k:
        for (sz_u32_t offset = 0; offset != node.count; ++offset) {
            sz_u32_t const position = direction == sz_regex_forward_k ? offset : node.count - 1 - offset;
            sz_regex_fragment_t next;
            status = sz_regex_compile_node_(builder, nfa, builder->children[node.first + position], direction, &next);
            if (status != sz_success_k) return status;
            if (offset == 0) *fragment = next;
            else if ((status = sz_regex_nfa_patch_(builder, nfa, fragment->end, next.start)) != sz_success_k)
                return status;
            fragment->end = next.end;
        }
        return sz_success_k;
    default: {
        sz_u32_t branches, exit;
        if ((status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_union_k, &branches)) != sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &exit)) != sz_success_k)
            return status;
        for (sz_u32_t offset = 0; offset != node.count; ++offset) {
            sz_regex_fragment_t next;
            if ((status = sz_regex_compile_node_(builder, nfa, builder->children[node.first + offset], direction,
                                                 &next)) != sz_success_k ||
                (status = sz_regex_nfa_patch_(builder, nfa, branches, next.start)) != sz_success_k ||
                (status = sz_regex_nfa_patch_(builder, nfa, next.end, exit)) != sz_success_k)
                return status;
        }
        fragment->start = branches, fragment->end = exit;
        return sz_success_k;
    }
    }
}

/**
 *  @brief Compiles every pattern into one NFA, as alternatives in index order, each ending in its
 *      own match state; the reversed one is entered through a lazy loop over any byte.
 *
 *  Compiling the reversed one also records the start pass's positions: the loop, the match,
 *  then every class.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_compile_patterns_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa,
                                                          sz_regex_direction_t direction) {
    sz_u32_t branches = sz_regex_none_k;
    sz_status_t status;
    if (direction == sz_regex_reverse_k &&
        ((status = sz_regex_add_position_(builder, sz_regex_none_k, sz_regex_none_k, sz_regex_none_k)) !=
             sz_success_k ||
         (status = sz_regex_add_position_(builder, sz_regex_none_k, sz_regex_none_k, sz_regex_none_k)) != sz_success_k))
        return status;
    if (builder->patterns_count > 1 &&
        (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_union_k, &branches)) != sz_success_k)
        return status;
    for (sz_size_t pattern = 0; pattern != builder->patterns_count; ++pattern) {
        sz_regex_fragment_t body;
        sz_u32_t match;
        if ((status = sz_regex_compile_node_(builder, nfa, builder->roots[pattern], direction, &body)) !=
                sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_match_k, &match)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, body.end, match)) != sz_success_k)
            return status;
        nfa->states[match].target = (sz_u32_t)pattern;
        if (direction == sz_regex_forward_k) builder->patterns_starts[pattern] = body.start;
        if (branches == sz_regex_none_k) nfa->start = body.start;
        else if ((status = sz_regex_nfa_patch_(builder, nfa, branches, body.start)) != sz_success_k) return status;
    }
    if (branches != sz_regex_none_k) nfa->start = branches;
    if (direction == sz_regex_reverse_k) {
        sz_u32_t loop, any, after;
        if ((status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_union_lazy_k, &loop)) != sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_empty_k, &after)) != sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_ranges_k, &any)) != sz_success_k ||
            (status = sz_regex_nfa_link_(builder, nfa, any, after, 0x00, 0xFF)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, any)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, after, loop)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, nfa->start)) != sz_success_k)
            return status;
        nfa->start = loop;
        builder->positions[0].entry = any, builder->positions[0].exit = after;
    }
    return sz_success_k;
}

/**
 *  @brief Compiles every lookaround's body into the forward NFA after the patterns, a lookbehind's
 *      as written and a lookahead's reversed, each ending in a match state naming its lookaround.
 *
 *  A reversed body records no positions, as the start pass reads its lookaround's bit rather
 *  than its classes.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_compile_lookarounds_(sz_regex_builder_t *builder) {
    sz_regex_nfa_t *const nfa = &builder->forward;
    sz_size_t const positions_count = builder->positions_count;
    sz_status_t status = sz_success_k;
    for (sz_size_t index = 0; index != builder->lookarounds_count && status == sz_success_k; ++index) {
        sz_regex_lookaround_t *const lookaround = builder->lookarounds + index;
        sz_regex_direction_t const direction = lookaround->kind >= sz_regex_lookbehind_k ? sz_regex_forward_k
                                                                                         : sz_regex_reverse_k;
        sz_regex_fragment_t body;
        sz_u32_t match;
        if ((status = sz_regex_compile_node_(builder, nfa, lookaround->start, direction, &body)) != sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_match_k, &match)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, body.end, match)) != sz_success_k)
            break;
        nfa->states[match].target = (sz_u32_t)index;
        lookaround->start = body.start;
    }
    builder->positions_count = positions_count;
    return status;
}

/** Collects into @c prefix_looks the assertions reachable from the start without consuming,
 *  crossing every one. */
STRINGZILLA_INLINE void sz_regex_nfa_prefix_(sz_regex_builder_t *builder, sz_regex_nfa_t *nfa) {
    sz_regex_set_t *const visited = &builder->sets[0];
    sz_u32_t *const stack = builder->stack;
    sz_size_t stack_count = 0;
    visited->count = 0, nfa->prefix_looks = 0;
    stack[stack_count++] = nfa->start;
    while (stack_count) {
        sz_u32_t const state = stack[--stack_count];
        sz_regex_nfa_state_t const *node = nfa->states + state;
        if (state == sz_regex_none_k || !sz_regex_set_insert_(visited, state)) continue;
        if (node->kind == sz_regex_nfa_look_k) nfa->prefix_looks |= node->look, stack[stack_count++] = node->target;
        else if (node->kind == sz_regex_nfa_empty_k) stack[stack_count++] = node->target;
        else if (node->kind == sz_regex_nfa_union_k || node->kind == sz_regex_nfa_union_lazy_k)
            for (sz_u32_t link = node->first_link; link != sz_regex_none_k; link = nfa->links[link].next)
                stack[stack_count++] = nfa->links[link].target;
    }
}

/** Gives bytes one column unless something tells them apart: a range edge of some link, a line feed
 *  beside line anchors, or the word-byte edges beside word boundaries. */
STRINGZILLA_INLINE void sz_regex_builder_classify_(sz_regex_builder_t *builder) {
    sz_regex_nfa_t const *const nfa = &builder->forward;
    sz_u8_t boundary[256];
    sz_size_t column = 0;
    memset(boundary, 0, sizeof(boundary));
    for (sz_size_t link = 0; link != nfa->links_count; ++link) {
        if (nfa->links[link].low > 0) boundary[nfa->links[link].low - 1] = 1;
        boundary[nfa->links[link].high] = 1;
    }
    if (nfa->looks & (sz_regex_look_line_start_k | sz_regex_look_line_end_k))
        boundary['\n' - 1] = boundary[(sz_u8_t)'\n'] = 1;
    if (nfa->looks & (sz_regex_look_word_boundary_k | sz_regex_look_word_inside_k))
        for (sz_size_t byte = 0; byte != 255; ++byte)
            if (sz_regex_is_word_((sz_u32_t)byte) != sz_regex_is_word_((sz_u32_t)byte + 1)) boundary[byte] = 1;
    for (sz_size_t byte = 0; byte != 256; ++byte) {
        builder->byte_to_class[byte] = (sz_u8_t)column;
        if (byte != 255 && boundary[byte]) ++column;
    }
    builder->classes_count = column + 1;
    for (sz_size_t byte = 256; byte != 0; --byte)
        builder->representatives[builder->byte_to_class[byte - 1]] = (sz_u8_t)(byte - 1);
}

#pragma endregion Compilation

#pragma region Determinization

/**
 *  @brief Adds every state reachable from @p start without consuming into @p set, depth first in
 *      priority order, crossing only the assertions in @p have.
 *
 *  A state already in the set ends its path, which is what makes leftmost-first priority the
 *  first path's. An unmet assertion stays in the set, so a later step can cross it.
 */
STRINGZILLA_INLINE void sz_regex_closure_(sz_regex_nfa_t const *nfa, sz_u32_t start, sz_u32_t have, sz_regex_set_t *set,
                                          sz_u32_t *stack) {
    sz_size_t stack_count = 0;
    stack[stack_count++] = start;
    while (stack_count) {
        sz_u32_t state = stack[--stack_count];
        while (state != sz_regex_none_k && sz_regex_set_insert_(set, state)) {
            sz_regex_nfa_state_t const *node = nfa->states + state;
            sz_u32_t link;
            sz_size_t base;
            if (node->kind == sz_regex_nfa_ranges_k || node->kind == sz_regex_nfa_match_k) break;
            if (node->kind == sz_regex_nfa_look_k && !(have & node->look)) break;
            if (node->kind == sz_regex_nfa_look_k || node->kind == sz_regex_nfa_empty_k) {
                state = node->target;
                continue;
            }
            link = node->first_link;
            if (link == sz_regex_none_k) break;
            state = nfa->links[link].target;
            base = stack_count;
            for (link = nfa->links[link].next; link != sz_regex_none_k; link = nfa->links[link].next)
                stack[stack_count++] = nfa->links[link].target;
            for (sz_size_t low = base, high = stack_count; low + 1 < high; ++low, --high) {
                sz_u32_t const swapped = stack[low];
                stack[low] = stack[high - 1], stack[high - 1] = swapped;
            }
        }
    }
}

/** Writes the key of the DFA state an ordered NFA set forms, dropping pass-through states, and
 *  returns its length; assertions known to hold are kept only while some state still tests one. */
STRINGZILLA_INLINE sz_size_t sz_regex_dfa_key_(sz_regex_nfa_t const *nfa, sz_regex_set_t const *set, sz_u32_t matched,
                                               sz_u32_t from_word, sz_u32_t have, sz_u32_t pattern, sz_u32_t *key) {
    sz_size_t count = 0;
    sz_u32_t need = 0;
    for (sz_size_t index = 0; index != set->count; ++index) {
        sz_u32_t const state = set->dense[index];
        sz_regex_nfa_state_t const *node = nfa->states + state;
        if (node->kind == sz_regex_nfa_empty_k) continue;
        if (node->kind == sz_regex_nfa_look_k) need |= node->look;
        key[sz_regex_key_header_k + count++] = state;
    }
    if (!need) have = 0;
    key[0] = matched | from_word << 1 | have << 8 | need << 16;
    key[1] = pattern;
    key[2] = (sz_u32_t)count;
    return sz_regex_key_header_k + count;
}

/** Mixes a key into a hash for the state table. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_regex_key_hash_(sz_u32_t const *key, sz_size_t length) {
    sz_u64_t hash = 0x9E3779B97F4A7C15ull ^ length;
    for (sz_size_t index = 0; index != length; ++index) {
        hash = (hash ^ key[index]) * 0xFF51AFD7ED558CCDull;
        hash ^= hash >> 29;
    }
    return hash;
}

/** Finds the state of @p key, adding it with an unfilled row when it is new. */
STRINGZILLA_INLINE sz_status_t sz_regex_dfa_intern_(sz_regex_builder_t *builder, sz_regex_dfa_t *dfa,
                                                    sz_u32_t const *key, sz_size_t length, sz_u32_t *state) {
    sz_allocator_t *const allocator = builder->allocator;
    sz_size_t const stride = sz_regex_stride_((sz_u32_t)builder->classes_count, (sz_u32_t)builder->lookarounds_count);
    sz_size_t slot, mask;
    if ((dfa->states_count + 1) * 2 > dfa->slots_count) {
        sz_size_t const grown_count = dfa->slots_count ? dfa->slots_count * 2 : 1024;
        sz_u32_t *const grown = (sz_u32_t *)allocator->allocate(grown_count * sizeof(sz_u32_t), allocator->handle,
                                                                builder->stream);
        if (!grown) return sz_bad_alloc_k;
        memset(grown, 0, grown_count * sizeof(sz_u32_t));
        for (sz_size_t existing = 0; existing != dfa->states_count; ++existing) {
            sz_u32_t const *existing_key = dfa->keys + dfa->offsets[existing];
            slot = sz_regex_key_hash_(existing_key, sz_regex_key_header_k + existing_key[2]) & (grown_count - 1);
            while (grown[slot]) slot = (slot + 1) & (grown_count - 1);
            grown[slot] = (sz_u32_t)existing + 1;
        }
        if (dfa->slots)
            allocator->free(dfa->slots, dfa->slots_count * sizeof(sz_u32_t), allocator->handle, builder->stream);
        dfa->slots = grown, dfa->slots_count = grown_count;
    }
    mask = dfa->slots_count - 1;
    slot = sz_regex_key_hash_(key, length) & mask;
    while (dfa->slots[slot]) {
        sz_u32_t const candidate = dfa->slots[slot] - 1;
        sz_u32_t const *candidate_key = dfa->keys + dfa->offsets[candidate];
        if (sz_regex_key_header_k + candidate_key[2] == length &&
            sz_equal_serial_((sz_cptr_t)candidate_key, (sz_cptr_t)key, length * sizeof(sz_u32_t))) {
            *state = candidate;
            return sz_success_k;
        }
        slot = (slot + 1) & mask;
    }
    // Every id must stay a 32-bit row offset, the published automaton's one ceiling.
    if (dfa->states_count == dfa->states_budget || (dfa->states_count + 1) * stride > 0xFFFFFFFFull)
        return sz_overflow_risk_k;
    {
        sz_u32_t *const keys = (sz_u32_t *)sz_regex_grow_(builder, dfa->keys, &dfa->keys_capacity,
                                                          dfa->keys_count + length, sizeof(sz_u32_t));
        sz_size_t *offsets;
        sz_u32_t *rows;
        if (!keys) return sz_bad_alloc_k;
        dfa->keys = keys;
        offsets = (sz_size_t *)sz_regex_grow_(builder, dfa->offsets, &dfa->offsets_capacity, dfa->states_count + 1,
                                              sizeof(sz_size_t));
        if (!offsets) return sz_bad_alloc_k;
        dfa->offsets = offsets;
        rows = (sz_u32_t *)sz_regex_grow_(builder, dfa->rows, &dfa->rows_capacity, (dfa->states_count + 1) * stride,
                                          sizeof(sz_u32_t));
        if (!rows) return sz_bad_alloc_k;
        dfa->rows = rows;
    }
    memcpy(dfa->keys + dfa->keys_count, key, length * sizeof(sz_u32_t));
    dfa->offsets[dfa->states_count] = dfa->keys_count;
    dfa->keys_count += length;
    dfa->slots[slot] = (sz_u32_t)dfa->states_count + 1;
    *state = (sz_u32_t)dfa->states_count++;
    return sz_success_k;
}

/**
 *  @brief Forms the key of the state @p index moves to on @p unit, a byte or the end of input,
 *      into `builder->key`, porting `regex-automata`'s @c determinize::next.
 *
 *  Assertions that look ahead are settled first, by re-running the closure from every state once
 *  @p unit or the lookarounds of @p looks, which hold at the boundary before @p unit, prove more of
 *  them. A match state among the current ones makes the next state accept, one unit late; under
 *  leftmost-first, the states after the first match are dropped.
 */
STRINGZILLA_INLINE sz_size_t sz_regex_dfa_step_(sz_regex_builder_t *builder, sz_regex_nfa_t const *nfa,
                                                sz_regex_dfa_t const *dfa, sz_size_t index, sz_u32_t unit,
                                                sz_u32_t looks, sz_regex_match_kind_t match_kind) {
    sz_u32_t const *const state_key = dfa->keys + dfa->offsets[index];
    sz_u32_t const flags = state_key[0], have_before = (flags >> 8) & 0xFF, need = flags >> 16;
    sz_u32_t const unit_is_word = unit != sz_regex_end_of_input_k && sz_regex_is_word_(unit);
    sz_regex_set_t *current = &builder->sets[0], *following = &builder->sets[1];
    sz_u32_t have_after = 0, matched = 0, pattern = 0;
    current->count = 0;
    for (sz_u32_t offset = 0; offset != state_key[2]; ++offset)
        sz_regex_set_insert_(current, state_key[sz_regex_key_header_k + offset]);
    if (need) {
        sz_u32_t have = have_before | looks;
        if (unit == sz_regex_end_of_input_k) have |= sz_regex_look_text_end_k | sz_regex_look_line_end_k;
        else if (unit == '\n') have |= sz_regex_look_line_end_k;
        have |= ((flags >> 1) & 1) == unit_is_word ? sz_regex_look_word_inside_k : sz_regex_look_word_boundary_k;
        if (have & ~have_before & need) {
            sz_regex_set_t *const swapped = current;
            following->count = 0;
            for (sz_size_t offset = 0; offset != current->count; ++offset)
                sz_regex_closure_(nfa, current->dense[offset], have, following, builder->stack);
            current = following, following = swapped;
        }
    }
    if ((nfa->looks & (sz_regex_look_line_start_k | sz_regex_look_line_end_k)) && unit == '\n')
        have_after |= sz_regex_look_line_start_k;
    following->count = 0;
    for (sz_size_t offset = 0; offset != current->count; ++offset) {
        sz_regex_nfa_state_t const *node = nfa->states + current->dense[offset];
        if (node->kind == sz_regex_nfa_match_k) {
            if (!matched) pattern = node->target;
            matched = 1;
            if (match_kind == sz_regex_leftmost_first_k) break;
            continue;
        }
        if (node->kind != sz_regex_nfa_ranges_k || unit == sz_regex_end_of_input_k) continue;
        for (sz_u32_t link = node->first_link; link != sz_regex_none_k; link = nfa->links[link].next)
            if (nfa->links[link].low <= unit && unit <= nfa->links[link].high)
                sz_regex_closure_(nfa, nfa->links[link].target, have_after, following, builder->stack);
    }
    return sz_regex_dfa_key_(nfa, following, matched,
                             following->count && unit_is_word &&
                                 (nfa->looks & (sz_regex_look_word_boundary_k | sz_regex_look_word_inside_k)),
                             have_after, pattern, builder->key);
}

/**
 *  @brief Builds the DFA of @p nfa by subset construction, from the dead state and four start
 *      states, one per kind of byte before the walk: none, a line feed, a word byte, another.
 *
 *  A start state knows only the assertions some prefix of the NFA tests, so kinds that no prefix
 *  tells apart share one state. Each block of a row steps under its own set of lookarounds, and a
 *  state testing no assertion copies its first block into the rest.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_determinize_(sz_regex_builder_t *builder, sz_regex_nfa_t const *nfa,
                                                     sz_regex_dfa_t *dfa, sz_regex_match_kind_t match_kind) {
    sz_size_t const classes_count = builder->classes_count;
    sz_size_t const stride = sz_regex_stride_((sz_u32_t)classes_count, (sz_u32_t)builder->lookarounds_count);
    sz_u32_t const prefix = nfa->prefix_looks;
    sz_u32_t *const key = builder->key;
    sz_u32_t state;
    sz_status_t status;
    key[0] = 0, key[1] = 0, key[2] = 0;
    if ((status = sz_regex_dfa_intern_(builder, dfa, key, sz_regex_key_header_k, &state)) != sz_success_k)
        return status;
    for (sz_size_t kind = 0; kind != 4; ++kind) {
        sz_u32_t have = 0, from_word = 0;
        if (kind == 0) have = prefix & (sz_regex_look_text_start_k | sz_regex_look_line_start_k);
        else if (kind == 1) have = prefix & sz_regex_look_line_start_k;
        else if (kind == 2) from_word = (prefix & (sz_regex_look_word_boundary_k | sz_regex_look_word_inside_k)) != 0;
        builder->sets[1].count = 0;
        sz_regex_closure_(nfa, nfa->start, have, &builder->sets[1], builder->stack);
        status = sz_regex_dfa_intern_(builder, dfa, key,
                                      sz_regex_dfa_key_(nfa, &builder->sets[1], 0, from_word, have, 0, key),
                                      &dfa->starts[kind]);
        if (status != sz_success_k) return status;
    }
    for (sz_size_t index = 0; index < dfa->states_count; ++index)
        for (sz_size_t column = 0; column != stride; ++column) {
            sz_size_t const unit_column = column % (classes_count + 1);
            sz_u32_t const unit = unit_column != classes_count ? builder->representatives[unit_column]
                                                               : (sz_u32_t)sz_regex_end_of_input_k;
            sz_u32_t const looks = (sz_u32_t)(column / (classes_count + 1)) * sz_regex_look_lookaround_k;
            sz_size_t length;
            if (looks && !(dfa->keys[dfa->offsets[index]] >> 16)) {
                dfa->rows[index * stride + column] = dfa->rows[index * stride + unit_column];
                continue;
            }
            length = sz_regex_dfa_step_(builder, nfa, dfa, index, unit, looks, match_kind);
            if ((status = sz_regex_dfa_intern_(builder, dfa, key, length, &state)) != sz_success_k) return status;
            dfa->rows[index * stride + column] = state;
        }
    return sz_success_k;
}

/**
 *  @brief Renumbers a finished DFA into published rows: dead first, then the states that do not
 *      accept, then those that do, each id its row's offset.
 *  @return How many states accept.
 */
STRINGZILLA_INLINE sz_size_t sz_regex_publish_rows_(sz_regex_builder_t *builder, sz_regex_dfa_t const *dfa,
                                                    sz_u32_t *rows, sz_u32_t *patterns, sz_u32_t *starts) {
    sz_size_t const stride = sz_regex_stride_((sz_u32_t)builder->classes_count, (sz_u32_t)builder->lookarounds_count);
    sz_u32_t *const positions = builder->published;
    sz_size_t accepting = 0, position = 0;
    for (sz_size_t index = 0; index != dfa->states_count; ++index) accepting += dfa->keys[dfa->offsets[index]] & 1;
    for (sz_size_t index = 0; index != dfa->states_count; ++index) {
        sz_u32_t const *key = dfa->keys + dfa->offsets[index];
        if (key[0] & 1) {
            sz_size_t const rank = dfa->states_count - accepting + position;
            if (patterns) patterns[position] = key[1];
            positions[index] = (sz_u32_t)(rank * stride);
            ++position;
        }
    }
    position = 0;
    for (sz_size_t index = 0; index != dfa->states_count; ++index)
        if (!(dfa->keys[dfa->offsets[index]] & 1)) positions[index] = (sz_u32_t)(position++ * stride);
    for (sz_size_t index = 0; index != dfa->states_count; ++index) {
        sz_u32_t *const row = rows + positions[index];
        sz_u32_t const *source = dfa->rows + index * stride;
        for (sz_size_t column = 0; column != stride; ++column) row[column] = positions[source[column]];
    }
    for (sz_size_t kind = 0; kind != 4; ++kind) starts[kind] = positions[dfa->starts[kind]];
    return accepting;
}

/** States past which a body's DFA is not checked for matching one byte, as one that does needs a
 *  handful. */
enum { sz_regex_single_states_k = 16 };

/**
 *  @brief Whether a body's published DFA matches exactly one byte of a set and asserts nothing, so
 *      a match ends right past each byte of the set; per class, whether the set holds it goes to
 *      @p held.
 *
 *  Every state a walk reaches over bytes must land on an accepting one exactly when the step before
 *  took a class of the set, whatever unit follows, the edge of the haystack included, and the start
 *  state, before any byte, never. States entered over the edge column end the walk, so no check
 *  reads their rows.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_body_single_(sz_u32_t const *rows, sz_u32_t start, sz_u32_t accepts,
                                                   sz_size_t columns, sz_size_t states_count, sz_u8_t *held) {
    sz_u32_t reached[sz_regex_single_states_k];
    sz_size_t reached_count = 1;
    if (states_count > sz_regex_single_states_k) return sz_false_k;
    for (sz_size_t column = 0; column != columns; ++column)
        if (rows[start + column] >= accepts) return sz_false_k;
    for (sz_size_t unit_class = 0; unit_class + 1 != columns; ++unit_class)
        held[unit_class] = (sz_u8_t)(rows[rows[start + unit_class]] >= accepts);
    reached[0] = start;
    for (sz_size_t visited = 0; visited != reached_count; ++visited)
        for (sz_size_t unit_class = 0; unit_class + 1 != columns; ++unit_class) {
            sz_u32_t const next = rows[reached[visited] + unit_class];
            sz_size_t known = 0;
            for (sz_size_t column = 0; column != columns; ++column)
                if ((rows[next + column] >= accepts) != held[unit_class]) return sz_false_k;
            while (known != reached_count && reached[known] != next) ++known;
            if (known == reached_count) reached[reached_count++] = next;
        }
    return sz_true_k;
}

/**
 *  @brief Builds the DFA of every lookaround body that nests no other, unanchored and keeping every
 *      match as the start-finding one is, into @c bodies_rows, numbered the way published rows are.
 *
 *  Each body is entered through a loop over any byte, appended to the forward NFA for the build and
 *  dropped after it, so the published NFA still ends with the bodies. Its rows hold one block, as
 *  no lookaround nests in it, and a body past the start-finding budget stays on the PikeVM. A cased
 *  body of one byte from a set keeps no rows, as the byte beside a position decides it.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_build_bodies_(sz_regex_builder_t *builder) {
    sz_allocator_t *const allocator = builder->allocator;
    sz_regex_nfa_t *const nfa = &builder->forward;
    sz_size_t const count = builder->lookarounds_count, columns = builder->classes_count + 1;
    sz_size_t const states_count = nfa->states_count, links_count = nfa->links_count;
    sz_status_t status = sz_success_k;
    if (!count) return sz_success_k;
    builder->bodies = (sz_u32_t *)allocator->allocate((3 * count + sz_regex_singles_k) * sizeof(sz_u32_t),
                                                      allocator->handle, builder->stream);
    if (!builder->bodies) return sz_bad_alloc_k;
    memset(builder->bodies, 0, (3 * count + sz_regex_singles_k) * sizeof(sz_u32_t));
    // The strides below follow the count, and no body steps under a lookaround.
    builder->lookarounds_count = 0;
    for (sz_size_t index = 0; index != count && status == sz_success_k; ++index) {
        sz_regex_dfa_t dfa;
        sz_regex_nfa_t view;
        sz_u32_t loop, any, *ranks = STRINGZILLA_NULL, *rows;
        sz_size_t accepting = 0, accepted = 0, rejected = 0;
        if (builder->lookarounds[index].level != 1) continue;
        nfa->states_count = states_count, nfa->links_count = links_count;
        if ((status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_union_k, &loop)) != sz_success_k ||
            (status = sz_regex_nfa_add_(builder, nfa, sz_regex_nfa_ranges_k, &any)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, builder->lookarounds[index].start)) != sz_success_k ||
            (status = sz_regex_nfa_patch_(builder, nfa, loop, any)) != sz_success_k ||
            (status = sz_regex_nfa_link_(builder, nfa, any, loop, 0x00, 0xFF)) != sz_success_k)
            break;
        view = *nfa, view.start = loop;
        sz_regex_nfa_prefix_(builder, &view);
        memset(&dfa, 0, sizeof(dfa));
        dfa.states_budget = sz_regex_reverse_budget_k;
        status = sz_regex_determinize_(builder, &view, &dfa, sz_regex_all_matches_k);
        if (status == sz_success_k) {
            ranks = (sz_u32_t *)allocator->allocate(dfa.states_count * sizeof(sz_u32_t), allocator->handle,
                                                    builder->stream);
            rows = (sz_u32_t *)sz_regex_grow_(builder, builder->bodies_rows, &builder->bodies_rows_capacity,
                                              builder->bodies_rows_count + dfa.states_count * columns,
                                              sizeof(sz_u32_t));
            if (rows) builder->bodies_rows = rows;
            if (!ranks || !rows) status = sz_bad_alloc_k;
        }
        if (status == sz_success_k) {
            sz_u32_t *const published = builder->bodies_rows + builder->bodies_rows_count;
            for (sz_size_t state = 0; state != dfa.states_count; ++state) accepting += dfa.keys[dfa.offsets[state]] & 1;
            for (sz_size_t state = 0; state != dfa.states_count; ++state) {
                sz_size_t const rank = dfa.keys[dfa.offsets[state]] & 1 ? dfa.states_count - accepting + accepted++
                                                                        : rejected++;
                ranks[state] = (sz_u32_t)(rank * columns);
            }
            sz_u32_t const start = ranks[dfa.starts[0]], accepts = (sz_u32_t)((dfa.states_count - accepting) * columns);
            sz_u8_t held[256];
            for (sz_size_t state = 0; state != dfa.states_count; ++state)
                for (sz_size_t column = 0; column != columns; ++column)
                    published[ranks[state] + column] = ranks[dfa.rows[state * columns + column]];
            if (builder->case_sensitivity == sz_cased_k &&
                sz_regex_body_single_(published, start, accepts, columns, dfa.states_count, held)) {
                sz_u32_t *const singles = builder->bodies + 3 * count;
                sz_u32_t const bit = (sz_u32_t)sz_regex_look_lookaround_k << index;
                sz_size_t const side = builder->lookarounds[index].kind >= sz_regex_lookbehind_k ? 1 : 257;
                singles[0] |= bit;
                for (sz_size_t byte = 0; byte != 256; ++byte)
                    if (held[builder->byte_to_class[byte]]) singles[side + byte] |= bit;
            }
            else {
                builder->bodies[3 * index] = (sz_u32_t)builder->bodies_rows_count + 1;
                builder->bodies[3 * index + 1] = start, builder->bodies[3 * index + 2] = accepts;
                builder->bodies_rows_count += dfa.states_count * columns;
            }
        }
        if (status == sz_overflow_risk_k) status = sz_success_k;
        if (ranks) allocator->free(ranks, dfa.states_count * sizeof(sz_u32_t), allocator->handle, builder->stream);
        if (dfa.keys)
            allocator->free(dfa.keys, dfa.keys_capacity * sizeof(sz_u32_t), allocator->handle, builder->stream);
        if (dfa.offsets)
            allocator->free(dfa.offsets, dfa.offsets_capacity * sizeof(sz_size_t), allocator->handle, builder->stream);
        if (dfa.rows)
            allocator->free(dfa.rows, dfa.rows_capacity * sizeof(sz_u32_t), allocator->handle, builder->stream);
        if (dfa.slots)
            allocator->free(dfa.slots, dfa.slots_count * sizeof(sz_u32_t), allocator->handle, builder->stream);
    }
    builder->lookarounds_count = count;
    nfa->states_count = states_count, nfa->links_count = links_count;
    return status;
}

#pragma endregion Determinization

#pragma region Positions

/** The boundary kinds at which the assertion @p look holds,
 *  as a mask of @c sz_regex_kinds_all_k bits. */
STRINGZILLA_INLINE sz_u32_t sz_regex_look_holds_(sz_u32_t look) {
    sz_u32_t holds = 0;
    for (sz_u32_t left = 0; left != 4; ++left)
        for (sz_u32_t right = 0; right != 4; ++right) {
            sz_u32_t const word_left = left == 2, word_right = right == 2;
            sz_u32_t held;
            switch (look) {
            case sz_regex_look_text_start_k: held = left == 0; break;
            case sz_regex_look_text_end_k: held = right == 0; break;
            case sz_regex_look_line_start_k: held = left <= 1; break;
            case sz_regex_look_line_end_k: held = right <= 1; break;
            case sz_regex_look_word_boundary_k: held = word_left != word_right; break;
            default: held = word_left == word_right; break;
            }
            holds |= held << (left * 4 + right);
        }
    return holds;
}

/** Pushes @p state onto a closure's stack under the kinds of @p holds no earlier path of the same
 *  closure brought, so no state is pushed more than sixteen times; the high half of each @p seen
 *  word names its closure. */
STRINGZILLA_INLINE sz_status_t sz_regex_follow_push_(sz_regex_builder_t *builder, sz_u64_t *seen, sz_u64_t closure,
                                                     sz_u32_t **stack, sz_size_t *stack_count,
                                                     sz_size_t *stack_capacity, sz_u32_t state, sz_u32_t holds) {
    sz_u32_t const known = seen[state] >> 32 == closure ? (sz_u32_t)seen[state] : 0;
    sz_u32_t *grown;
    holds &= ~known;
    if (!holds) return sz_success_k;
    seen[state] = closure << 32 | known | holds;
    grown = (sz_u32_t *)sz_regex_grow_(builder, *stack, stack_capacity, *stack_count + 2, sizeof(sz_u32_t));
    if (!grown) return sz_bad_alloc_k;
    *stack = grown;
    grown[(*stack_count)++] = state, grown[(*stack_count)++] = holds;
    return sz_success_k;
}

/**
 *  @brief Collects the start pass's follow edges from the reversed NFA: from each position's exit,
 *      every position and the match reachable without consuming, under the boundary kinds the
 *      assertions crossed allow and with the lookarounds crossed.
 *
 *  A reversed look state tests the mirrored assertion, so a crossed one is mirrored back, as
 *  boundary kinds name their units in reading order. A path crossing a lookaround its closure has
 *  not crossed waits for a closure of its own, one per set of lookarounds, so each closure keeps
 *  one set of kinds per state. Edges come out grouped by source, sources ascending.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_collect_follows_(sz_regex_builder_t *builder) {
    sz_regex_nfa_t const *const nfa = &builder->reverse;
    sz_allocator_t *const allocator = builder->allocator;
    sz_size_t const states_count = nfa->states_count, positions_count = builder->positions_count;
    sz_size_t const block_bytes = states_count * (sizeof(sz_u64_t) + sizeof(sz_u32_t)) +
                                  2 * positions_count * sizeof(sz_u32_t);
    sz_u64_t *const seen = (sz_u64_t *)allocator->allocate(block_bytes, allocator->handle, builder->stream);
    sz_u32_t *position_of, *holds_of, *targets;
    sz_u32_t *stack = STRINGZILLA_NULL, *deferred = STRINGZILLA_NULL;
    sz_size_t stack_capacity = 0, deferred_capacity = 0;
    sz_u64_t closure = 0;
    sz_status_t status = sz_success_k;
    if (!seen) return sz_bad_alloc_k;
    position_of = (sz_u32_t *)(seen + states_count);
    holds_of = position_of + states_count, targets = holds_of + positions_count;
    memset(seen, 0, states_count * sizeof(sz_u64_t));
    memset(position_of, 0xFF, states_count * sizeof(sz_u32_t));
    memset(holds_of, 0, positions_count * sizeof(sz_u32_t));
    for (sz_size_t position = 0; position != positions_count; ++position)
        if (builder->positions[position].entry != sz_regex_none_k)
            position_of[builder->positions[position].entry] = (sz_u32_t)position;
    for (sz_size_t position = 0; position != positions_count && status == sz_success_k; ++position) {
        sz_size_t stack_count = 0, deferred_count = 0;
        sz_u32_t looks = 0;
        if (builder->positions[position].exit == sz_regex_none_k) continue;
        status = sz_regex_follow_push_(builder, seen, ++closure, &stack, &stack_count, &stack_capacity,
                                       builder->positions[position].exit, sz_regex_kinds_all_k);
        while (status == sz_success_k) {
            sz_size_t targets_count = 0, kept = 0;
            while (stack_count && status == sz_success_k) {
                sz_u32_t const holds = stack[--stack_count], state = stack[--stack_count];
                sz_regex_nfa_state_t const *node = nfa->states + state;
                sz_u32_t target = sz_regex_none_k;
                if (node->kind == sz_regex_nfa_ranges_k) target = position_of[state];
                else if (node->kind == sz_regex_nfa_match_k) target = 1;
                else if (node->kind == sz_regex_nfa_look_k && (node->look & ~looks) >= sz_regex_look_lookaround_k) {
                    sz_u32_t *const grown = (sz_u32_t *)sz_regex_grow_(builder, deferred, &deferred_capacity,
                                                                       3 * deferred_count + 3, sizeof(sz_u32_t));
                    if (!grown) status = sz_bad_alloc_k;
                    else {
                        deferred = grown;
                        deferred[3 * deferred_count] = node->target, deferred[3 * deferred_count + 1] = holds;
                        deferred[3 * deferred_count++ + 2] = looks | node->look;
                    }
                }
                else if (node->kind == sz_regex_nfa_empty_k || node->kind == sz_regex_nfa_look_k) {
                    sz_u32_t const crossed = node->kind == sz_regex_nfa_empty_k ||
                                                     node->look >= sz_regex_look_lookaround_k
                                                 ? holds
                                                 : holds & sz_regex_look_holds_(sz_regex_reverse_look_(node->look));
                    if (crossed)
                        status = sz_regex_follow_push_(builder, seen, closure, &stack, &stack_count, &stack_capacity,
                                                       node->target, crossed);
                }
                else
                    for (sz_u32_t link = node->first_link; link != sz_regex_none_k && status == sz_success_k;
                         link = nfa->links[link].next)
                        status = sz_regex_follow_push_(builder, seen, closure, &stack, &stack_count, &stack_capacity,
                                                       nfa->links[link].target, holds);
                if (target == sz_regex_none_k) continue;
                if (!holds_of[target]) targets[targets_count++] = target;
                holds_of[target] |= holds;
            }
            for (sz_size_t index = 0; index != targets_count && status == sz_success_k; ++index) {
                sz_regex_follow_t *const follows = (sz_regex_follow_t *)sz_regex_grow_(
                    builder, builder->follows, &builder->follows_capacity, builder->follows_count + 1,
                    sizeof(sz_regex_follow_t));
                if (!follows) {
                    status = sz_bad_alloc_k;
                    break;
                }
                builder->follows = follows;
                follows[builder->follows_count].from = (sz_u32_t)position,
                follows[builder->follows_count].to = targets[index];
                follows[builder->follows_count].holds = holds_of[targets[index]];
                follows[builder->follows_count].looks = looks;
                ++builder->follows_count;
            }
            for (sz_size_t index = 0; index != targets_count; ++index) holds_of[targets[index]] = 0;
            if (!deferred_count || status != sz_success_k) break;
            // The next closure takes every waiting path across the same lookarounds
            // as the last one waiting.
            looks = deferred[3 * deferred_count - 1], ++closure;
            for (sz_size_t index = 0; index != deferred_count && status == sz_success_k; ++index) {
                sz_u32_t const state = deferred[3 * index], holds = deferred[3 * index + 1];
                if (deferred[3 * index + 2] == looks)
                    status = sz_regex_follow_push_(builder, seen, closure, &stack, &stack_count, &stack_capacity, state,
                                                   holds);
                else
                    deferred[3 * kept] = state, deferred[3 * kept + 1] = holds,
                                 deferred[3 * kept + 2] = deferred[3 * index + 2], ++kept;
            }
            deferred_count = kept;
        }
    }
    if (stack) allocator->free(stack, stack_capacity * sizeof(sz_u32_t), allocator->handle, builder->stream);
    if (deferred) allocator->free(deferred, deferred_capacity * sizeof(sz_u32_t), allocator->handle, builder->stream);
    allocator->free(seen, block_bytes, allocator->handle, builder->stream);
    return status;
}

/** Inserts @p point into the ascending, distinct @p points unless it is there already. */
STRINGZILLA_INLINE sz_status_t sz_regex_insert_point_(sz_regex_builder_t *builder, sz_u32_t **points,
                                                      sz_size_t *points_count, sz_size_t *points_capacity,
                                                      sz_u32_t point) {
    sz_u32_t const *const sorted = *points;
    sz_size_t low = 0, high = *points_count;
    sz_u32_t *grown;
    while (low < high) {
        sz_size_t const middle = (low + high) / 2;
        if (sorted[middle] < point) low = middle + 1;
        else high = middle;
    }
    if (low < *points_count && sorted[low] == point) return sz_success_k;
    grown = (sz_u32_t *)sz_regex_grow_(builder, *points, points_capacity, *points_count + 1, sizeof(sz_u32_t));
    if (!grown) return sz_bad_alloc_k;
    *points = grown;
    memmove(grown + low + 1, grown + low, (*points_count - low) * sizeof(sz_u32_t));
    grown[low] = point, ++*points_count;
    return sz_success_k;
}

/** Mixes a run of words into a hash for an open-addressed table. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_regex_words_hash_(sz_u64_t const *words, sz_size_t count, sz_u64_t seed) {
    sz_u64_t hash = 0x9E3779B97F4A7C15ull ^ seed;
    for (sz_size_t index = 0; index != count; ++index) {
        hash = (hash ^ words[index]) * 0xFF51AFD7ED558CCDull;
        hash ^= hash >> 29;
    }
    return hash;
}

/**
 *  @brief Splits the codepoints into the unit classes no position tells apart, and gives each
 *      class its reach row.
 *
 *  Class zero holds the codepoints no position consumes and every malformed byte, class one is
 *  the haystack's start, which only the match position survives, and the rest follow in order of
 *  their first codepoint.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_partition_codepoints_(sz_regex_builder_t *builder) {
    sz_allocator_t *const allocator = builder->allocator;
    sz_size_t const words = builder->words, positions_count = builder->positions_count;
    sz_u32_t *points = STRINGZILLA_NULL, *slots = STRINGZILLA_NULL;
    sz_u64_t *signatures = STRINGZILLA_NULL;
    sz_size_t points_count = 0, points_capacity = 0, intervals = 0, slots_count = 1, classes = 2;
    sz_status_t status = sz_regex_insert_point_(builder, &points, &points_count, &points_capacity, 0);
    if (status == sz_success_k)
        status = sz_regex_insert_point_(builder, &points, &points_count, &points_capacity, 0x80);
    if (status == sz_success_k)
        status = sz_regex_insert_point_(builder, &points, &points_count, &points_capacity, 0x110000);
    for (sz_size_t position = 2; position < positions_count && status == sz_success_k; ++position) {
        sz_regex_node_t const *node = builder->nodes + builder->positions[position].node;
        for (sz_u32_t range = 0; range != node->count && status == sz_success_k; ++range) {
            sz_regex_range_t const span = builder->ranges[node->first + range];
            status = sz_regex_insert_point_(builder, &points, &points_count, &points_capacity, span.first);
            if (status == sz_success_k)
                status = sz_regex_insert_point_(builder, &points, &points_count, &points_capacity, span.last + 1);
        }
    }
    if (status == sz_success_k) {
        sz_u64_t *reach;
        intervals = points_count - 1;
        while (slots_count < 2 * intervals) slots_count *= 2;
        signatures = (sz_u64_t *)allocator->allocate(intervals * words * sizeof(sz_u64_t), allocator->handle,
                                                     builder->stream);
        slots = (sz_u32_t *)allocator->allocate(slots_count * sizeof(sz_u32_t), allocator->handle, builder->stream);
        reach = (sz_u64_t *)sz_regex_grow_(builder, builder->reach, &builder->reach_capacity, (2 + intervals) * words,
                                           sizeof(sz_u64_t));
        if (reach) builder->reach = reach;
        if (!signatures || !slots || !reach) status = sz_bad_alloc_k;
    }
    if (status == sz_success_k) {
        sz_u64_t *const reach = builder->reach;
        memset(signatures, 0, intervals * words * sizeof(sz_u64_t));
        memset(slots, 0, slots_count * sizeof(sz_u32_t));
        memset(reach, 0, 2 * words * sizeof(sz_u64_t));
        reach[0] = 3, reach[words] = 2;
        for (sz_size_t position = 2; position < positions_count; ++position) {
            sz_regex_node_t const *node = builder->nodes + builder->positions[position].node;
            for (sz_u32_t range = 0; range != node->count; ++range) {
                sz_regex_range_t const span = builder->ranges[node->first + range];
                sz_size_t low = 0, high = points_count;
                while (low < high) {
                    sz_size_t const middle = (low + high) / 2;
                    if (points[middle] < span.first) low = middle + 1;
                    else high = middle;
                }
                for (; points[low] <= span.last; ++low)
                    signatures[low * words + position / 64] |= 1ull << (position % 64);
            }
        }
        builder->ranges_classes_count = 0;
        for (sz_size_t interval = 0; interval != intervals && status == sz_success_k; ++interval) {
            sz_u64_t const *signature = signatures + interval * words;
            sz_u64_t any = 0;
            sz_u32_t unit_class = 0;
            for (sz_size_t word = 0; word != words; ++word) any |= signature[word];
            if (any) {
                sz_size_t slot = sz_regex_words_hash_(signature, words, 0) & (slots_count - 1);
                for (; slots[slot]; slot = (slot + 1) & (slots_count - 1)) {
                    sz_u64_t const *row = reach + slots[slot] * words;
                    sz_size_t word = 1;
                    if ((row[0] & ~3ull) != signature[0]) continue;
                    while (word != words && row[word] == signature[word]) ++word;
                    if (word == words) break;
                }
                if (!slots[slot]) {
                    memcpy(reach + classes * words, signature, words * sizeof(sz_u64_t));
                    reach[classes * words] |= 3;
                    slots[slot] = (sz_u32_t)classes++;
                }
                unit_class = slots[slot];
            }
            if (points[interval] < 0x80)
                for (sz_u32_t codepoint = points[interval]; codepoint != points[interval + 1]; ++codepoint)
                    builder->ascii[codepoint] = unit_class;
            else if (!builder->ranges_classes_count ||
                     builder->ranges_classes[builder->ranges_classes_count - 1] != unit_class) {
                sz_u32_t *const pairs = (sz_u32_t *)sz_regex_grow_(builder, builder->ranges_classes,
                                                                   &builder->ranges_classes_capacity,
                                                                   builder->ranges_classes_count + 2, sizeof(sz_u32_t));
                if (!pairs) status = sz_bad_alloc_k;
                else {
                    builder->ranges_classes = pairs;
                    pairs[builder->ranges_classes_count++] = points[interval];
                    pairs[builder->ranges_classes_count++] = unit_class;
                }
            }
        }
        builder->unit_classes_count = classes;
    }
    if (points) allocator->free(points, points_capacity * sizeof(sz_u32_t), allocator->handle, builder->stream);
    if (signatures)
        allocator->free(signatures, intervals * words * sizeof(sz_u64_t), allocator->handle, builder->stream);
    if (slots) allocator->free(slots, slots_count * sizeof(sz_u32_t), allocator->handle, builder->stream);
    return status;
}

/**
 *  @brief Turns the follow edges into the start pass's records, and collects every
 *      position a walk can hold.
 *
 *  The loop is live at every unit, so its edges across no lookaround become a table of targets by
 *  boundary kind. The distances most common between the other unconditional edges become shifts,
 *  each moving at once every position with such an edge. Every other edge joins a group: the
 *  sources whose remaining targets under one condition are the same, which a step tests once and
 *  enters together. A condition is a mask of boundary kinds in its low 16 bits, then the look bits
 *  of the lookarounds that must hold.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_build_records_(sz_regex_builder_t *builder) {
    sz_allocator_t *const allocator = builder->allocator;
    sz_size_t const words = builder->words, positions_count = builder->positions_count;
    sz_size_t const follows_count = builder->follows_count, record_words = 1 + 2 * words;
    sz_size_t const distances_count = 2 * positions_count - 1;
    sz_regex_follow_t const *const follows = builder->follows;
    sz_size_t slots_count = 1, shifts_words, block_bytes;
    sz_i64_t distances[sz_regex_shifts_limit_k];
    sz_u64_t *targets;
    sz_u32_t *counts, *slots, *queue;
    sz_u8_t *covered;
    sz_status_t status = sz_success_k;
    while (slots_count < 2 * (follows_count + 1)) slots_count *= 2;
    block_bytes = words * sizeof(sz_u64_t) + (distances_count + slots_count + positions_count) * sizeof(sz_u32_t) +
                  follows_count;
    targets = (sz_u64_t *)allocator->allocate(block_bytes, allocator->handle, builder->stream);
    builder->all = (sz_u64_t *)allocator->allocate(17 * words * sizeof(sz_u64_t), allocator->handle, builder->stream);
    if (!targets || !builder->all) {
        if (targets) allocator->free(targets, block_bytes, allocator->handle, builder->stream);
        return sz_bad_alloc_k;
    }
    counts = (sz_u32_t *)(targets + words), slots = counts + distances_count, queue = slots + slots_count;
    covered = (sz_u8_t *)(queue + positions_count);
    memset(counts, 0, (distances_count + slots_count) * sizeof(sz_u32_t));
    memset(covered, 0, follows_count);
    memset(builder->all, 0, 17 * words * sizeof(sz_u64_t));
    for (sz_size_t index = 0; index != follows_count; ++index) {
        sz_u32_t const to = follows[index].to;
        if (follows[index].looks) continue;
        if (follows[index].from == 0) {
            for (sz_size_t kind = 0; kind != 16; ++kind)
                if (follows[index].holds >> kind & 1) builder->all[(1 + kind) * words + to / 64] |= 1ull << (to % 64);
            covered[index] = 1;
        }
        else if (follows[index].holds == sz_regex_kinds_all_k) ++counts[to + positions_count - 1 - follows[index].from];
    }
    builder->shifts_count = 0;
    // A shift costs a step as much as a group does, so a distance only coincidences share stays
    // in the groups.
    for (sz_u32_t most = 0; builder->shifts_count != sz_regex_shifts_limit_k;) {
        sz_size_t best = 0;
        for (sz_size_t index = 1; index != distances_count; ++index)
            if (counts[index] > counts[best]) best = index;
        if (counts[best] < 2 || counts[best] * 4 < most) break;
        if (!most) most = counts[best];
        counts[best] = 0;
        distances[builder->shifts_count++] = (sz_i64_t)best - (sz_i64_t)(positions_count - 1);
    }
    shifts_words = builder->shifts_count * (1 + words);
    builder->records = (sz_u64_t *)sz_regex_grow_(builder, builder->records, &builder->records_capacity, shifts_words,
                                                  sizeof(sz_u64_t));
    if (!builder->records && shifts_words) status = sz_bad_alloc_k;
    if (status == sz_success_k) {
        if (shifts_words) memset(builder->records, 0, shifts_words * sizeof(sz_u64_t));
        for (sz_size_t shift = 0; shift != builder->shifts_count; ++shift)
            builder->records[shift * (1 + words)] = (sz_u64_t)distances[shift];
        for (sz_size_t index = 0; index != follows_count; ++index)
            for (sz_size_t shift = 0; shift != builder->shifts_count; ++shift) {
                sz_u32_t const from = follows[index].from;
                if (covered[index] || follows[index].holds != sz_regex_kinds_all_k || follows[index].looks ||
                    (sz_i64_t)follows[index].to - (sz_i64_t)from != distances[shift])
                    continue;
                builder->records[shift * (1 + words) + 1 + from / 64] |= 1ull << (from % 64);
                covered[index] = 1;
            }
        builder->records_count = shifts_words, builder->groups_count = 0;
    }
    for (sz_size_t index = 0; index != follows_count && status == sz_success_k; ++index) {
        sz_u32_t const from = follows[index].from;
        sz_u64_t const condition = follows[index].holds | (sz_u64_t)follows[index].looks << 16;
        sz_size_t slot;
        if (covered[index]) continue;
        memset(targets, 0, words * sizeof(sz_u64_t));
        for (sz_size_t other = index; other != follows_count && follows[other].from == from; ++other)
            if (!covered[other] && (follows[other].holds | (sz_u64_t)follows[other].looks << 16) == condition)
                targets[follows[other].to / 64] |= 1ull << (follows[other].to % 64), covered[other] = 1;
        slot = sz_regex_words_hash_(targets, words, condition) & (slots_count - 1);
        for (; slots[slot]; slot = (slot + 1) & (slots_count - 1)) {
            sz_u64_t const *record = builder->records + shifts_words + (slots[slot] - 1) * record_words;
            if (record[0] == condition &&
                sz_equal_serial_((sz_cptr_t)(record + 1 + words), (sz_cptr_t)targets, words * sizeof(sz_u64_t)))
                break;
        }
        if (!slots[slot]) {
            sz_u64_t *const records = (sz_u64_t *)sz_regex_grow_(builder, builder->records, &builder->records_capacity,
                                                                 builder->records_count + record_words,
                                                                 sizeof(sz_u64_t));
            if (!records) {
                status = sz_bad_alloc_k;
                break;
            }
            builder->records = records;
            records[builder->records_count] = condition;
            memset(records + builder->records_count + 1, 0, words * sizeof(sz_u64_t));
            memcpy(records + builder->records_count + 1 + words, targets, words * sizeof(sz_u64_t));
            builder->records_count += record_words;
            slots[slot] = (sz_u32_t)++builder->groups_count;
        }
        builder->records[shifts_words + (slots[slot] - 1) * record_words + 1 + from / 64] |= 1ull << (from % 64);
    }
    if (status == sz_success_k) {
        sz_size_t queue_begin = 0, queue_end = 1;
        builder->all[0] = 1, queue[0] = 0;
        while (queue_begin != queue_end) {
            sz_u32_t const from = queue[queue_begin++];
            sz_size_t low = 0, high = follows_count;
            while (low < high) {
                sz_size_t const middle = (low + high) / 2;
                if (follows[middle].from < from) low = middle + 1;
                else high = middle;
            }
            for (; low != follows_count && follows[low].from == from; ++low) {
                sz_u32_t const to = follows[low].to;
                if (builder->all[to / 64] >> (to % 64) & 1) continue;
                builder->all[to / 64] |= 1ull << (to % 64), queue[queue_end++] = to;
            }
        }
    }
    allocator->free(targets, block_bytes, allocator->handle, builder->stream);
    return status;
}

#pragma endregion Positions

#pragma region Building

STRINGZILLA_INLINE void sz_regex_engine_free_(sz_regex_engine_t *engine, sz_stream_t stream) {
    sz_allocator_t *const allocator = &engine->allocator;
    if (engine->memory) allocator->free(engine->memory, engine->memory_bytes, allocator->handle, stream);
    if (engine->scratch) allocator->free(engine->scratch, engine->scratch_bytes, allocator->handle, stream);
    engine->memory = STRINGZILLA_NULL, engine->memory_bytes = 0;
    engine->scratch = STRINGZILLA_NULL, engine->scratch_bytes = 0;
    engine->byte_to_class = STRINGZILLA_NULL, engine->reverse_rows = STRINGZILLA_NULL;
    engine->forward_rows = STRINGZILLA_NULL, engine->forward_patterns = STRINGZILLA_NULL;
    engine->forward_escapes = STRINGZILLA_NULL, engine->prefixes = STRINGZILLA_NULL;
    engine->classes_count = 0, engine->forward_states_count = 0, engine->reverse_states_count = 0;
    engine->patterns_count = 0, engine->capability = sz_cap_serial_k;
    engine->positions_words = 0, engine->positions_shifts_count = 0, engine->positions_groups_count = 0;
    engine->positions_ranges_count = 0, engine->positions_reach = STRINGZILLA_NULL;
    engine->positions_shifts = STRINGZILLA_NULL, engine->positions_groups = STRINGZILLA_NULL;
    engine->positions_all = STRINGZILLA_NULL, engine->positions_initial = STRINGZILLA_NULL;
    engine->positions_ascii = STRINGZILLA_NULL;
    engine->positions_ranges = STRINGZILLA_NULL;
    engine->nfa_states = STRINGZILLA_NULL, engine->nfa_links = STRINGZILLA_NULL;
    engine->nfa_states_count = 0, engine->nfa_links_count = 0, engine->nfa_start = 0;
    engine->patterns_starts = STRINGZILLA_NULL, engine->patterns_groups = STRINGZILLA_NULL;
    engine->lookarounds = STRINGZILLA_NULL, engine->bodies = STRINGZILLA_NULL, engine->lookarounds_count = 0;
    engine->looks_bytes = STRINGZILLA_NULL;
    engine->groups_names.layout = sz_strings_u32_tape_k, engine->groups_names.count = 0;
    engine->groups_names.data.u32_tape.data = STRINGZILLA_NULL,
    engine->groups_names.data.u32_tape.offsets = STRINGZILLA_NULL;
}

/** Parses every pattern, refusing the first malformed one by index,
 *  and numbers and names its capturing groups. */
STRINGZILLA_INLINE sz_status_t sz_regex_builder_parse_(sz_regex_builder_t *builder, sz_strings_view_t const *patterns,
                                                       sz_size_t *malformed_index) {
    sz_string_view_t views[64];
    builder->groups_ends = (sz_u32_t *)sz_regex_grow_(builder, STRINGZILLA_NULL, &builder->groups_ends_capacity, 1,
                                                      sizeof(sz_u32_t));
    if (!builder->groups_ends) return sz_bad_alloc_k;
    builder->groups_ends[0] = 0, builder->groups_ends_count = 1, builder->patterns_groups[0] = 0;
    for (sz_size_t pattern = 0; pattern != patterns->count; ++pattern) {
        sz_status_t status;
        if (pattern % 64 == 0) sz_strings_get_block_(patterns, pattern, views);
        status = sz_regex_parse_pattern_(builder, views[pattern % 64].start, views[pattern % 64].length,
                                         builder->roots + pattern);
        if (status == sz_invalid_syntax_k && malformed_index) *malformed_index = pattern;
        if (status != sz_success_k) return status;
        builder->patterns_groups[pattern + 1] = builder->patterns_groups[pattern] +
                                                (sz_u32_t)(builder->names_count / 2);
        for (sz_size_t name = 0; name != builder->names_count; name += 2) {
            sz_size_t const length = builder->names[name + 1];
            sz_u32_t *const ends = (sz_u32_t *)sz_regex_grow_(builder, builder->groups_ends,
                                                              &builder->groups_ends_capacity,
                                                              builder->groups_ends_count + 1, sizeof(sz_u32_t));
            if (!ends) return sz_bad_alloc_k;
            builder->groups_ends = ends;
            if (length) {
                sz_u8_t *const names = (sz_u8_t *)sz_regex_grow_(builder, builder->groups_names,
                                                                 &builder->groups_names_capacity,
                                                                 builder->groups_names_count + length, 1);
                if (!names) return sz_bad_alloc_k;
                builder->groups_names = names;
                memcpy(names + builder->groups_names_count, views[pattern % 64].start + builder->names[name], length);
                builder->groups_names_count += length;
            }
            ends[builder->groups_ends_count++] = (sz_u32_t)builder->groups_names_count;
        }
    }
    return sz_success_k;
}

/** Sizes the closure scratch for the larger NFA, the forward one with room for the loops its
 *  lookaround bodies' DFAs enter through, zeroing the sparse halves so every read is defined. */
STRINGZILLA_INLINE sz_status_t sz_regex_builder_scratch_(sz_regex_builder_t *builder) {
    sz_allocator_t *const allocator = builder->allocator;
    sz_size_t const states = sz_max_of_two(builder->forward.states_count + 2 * builder->lookarounds_count,
                                           builder->reverse.states_count);
    sz_size_t const links = sz_max_of_two(builder->forward.links_count + 3 * builder->lookarounds_count,
                                          builder->reverse.links_count);
    builder->set_capacity = states + 1;
    builder->stack_capacity = states + links + 1;
    for (sz_size_t index = 0; index != 2; ++index) {
        builder->sets[index].dense = (sz_u32_t *)allocator->allocate(builder->set_capacity * sizeof(sz_u32_t),
                                                                     allocator->handle, builder->stream);
        builder->sets[index].sparse = (sz_u32_t *)allocator->allocate(builder->set_capacity * sizeof(sz_u32_t),
                                                                      allocator->handle, builder->stream);
        if (!builder->sets[index].dense || !builder->sets[index].sparse) return sz_bad_alloc_k;
        memset(builder->sets[index].sparse, 0, builder->set_capacity * sizeof(sz_u32_t));
        builder->sets[index].count = 0;
    }
    builder->stack = (sz_u32_t *)allocator->allocate(builder->stack_capacity * sizeof(sz_u32_t), allocator->handle,
                                                     builder->stream);
    builder->key = (sz_u32_t *)allocator->allocate((builder->set_capacity + sz_regex_key_header_k) * sizeof(sz_u32_t),
                                                   allocator->handle, builder->stream);
    return builder->stack && builder->key ? sz_success_k : sz_bad_alloc_k;
}

/** DFA states one path of the literal prefilter's derivation may hold before it gives up. */
enum { sz_regex_prefixes_states_k = 16 };

/** A path the literal prefilter's derivation follows: the end-finding states it holds, the classes
 *  it took, and whether a match can end where it stands. */
typedef struct sz_regex_prefix_path_t {
    sz_u32_t states[sz_regex_prefixes_states_k];
    sz_u32_t states_count;
    sz_u8_t classes[sz_regex_prefixes_depth_k];
    sz_u32_t length;
    sz_bool_t ended;
} sz_regex_prefix_path_t;

/**
 *  @brief Derives from the end-finding DFA the paths of classes the first bytes of every match
 *      follow, into @p prefixes, answering whether a search should skip to them.
 *
 *  Paths grow class by class from every start state, under every lookaround block, until a match
 *  can end where they stand, they reach @c sz_regex_prefixes_depth_k classes, or the next step
 *  would hold more than @c sz_regex_prefixes_paths_k paths. A match that can be empty leaves none.
 *  The paths must also pass few positions, judged with every byte at least a sixteenth as frequent
 *  as any, as text is: two literal bytes pass, one or a wide class does not.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_prefixes_derive_(sz_regex_engine_t const *engine, sz_regex_prefixes_t *prefixes) {
    sz_u32_t const *const rows = engine->forward_rows;
    sz_size_t const classes_count = engine->classes_count, columns = classes_count + 1;
    sz_size_t const stride = sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_regex_prefix_path_t paths[2][sz_regex_prefixes_paths_k];
    sz_regex_prefix_path_t *current = paths[0], *next = paths[1];
    sz_size_t current_count = 1, depth = 0, estimate = 0, class_bytes[256];
    current[0].states_count = 0, current[0].length = 0, current[0].ended = sz_false_k;
    for (sz_size_t kind = 0; kind != 4; ++kind) {
        sz_u32_t const start = engine->forward_starts[kind];
        sz_size_t held = 0;
        while (held != current[0].states_count && current[0].states[held] != start) ++held;
        if (start && held == current[0].states_count) current[0].states[current[0].states_count++] = start;
    }
    for (;;) {
        sz_size_t next_count = 0;
        sz_bool_t crowded = sz_false_k;
        for (sz_size_t path = 0; path != current_count; ++path)
            for (sz_size_t held = 0; held != current[path].states_count && !current[path].ended; ++held)
                for (sz_size_t column = 0; column != stride; ++column)
                    if (rows[current[path].states[held] + column] >= engine->forward_accepts_begin) {
                        if (depth == 0) return sz_false_k;
                        current[path].ended = sz_true_k;
                        break;
                    }
        if (depth == sz_regex_prefixes_depth_k) break;
        for (sz_size_t path = 0; path != current_count && !crowded; ++path) {
            if (current[path].ended) {
                if (next_count == sz_regex_prefixes_paths_k) crowded = sz_true_k;
                else next[next_count++] = current[path];
                continue;
            }
            for (sz_size_t unit_class = 0; unit_class != classes_count && !crowded; ++unit_class) {
                sz_regex_prefix_path_t *const grown = next + next_count;
                if (next_count == sz_regex_prefixes_paths_k) {
                    // A step that dies everywhere takes no bucket, so only a live one crowds.
                    for (sz_size_t held = 0; held != current[path].states_count && !crowded; ++held)
                        for (sz_size_t block = 0; block < stride && !crowded; block += columns)
                            crowded = (sz_bool_t)(rows[current[path].states[held] + block + unit_class] != 0);
                    continue;
                }
                grown->states_count = 0;
                for (sz_size_t held = 0; held != current[path].states_count && !crowded; ++held)
                    for (sz_size_t block = 0; block < stride && !crowded; block += columns) {
                        sz_u32_t const target = rows[current[path].states[held] + block + unit_class];
                        sz_size_t known = 0;
                        while (known != grown->states_count && grown->states[known] != target) ++known;
                        if (!target || known != grown->states_count) continue;
                        if (grown->states_count == sz_regex_prefixes_states_k) crowded = sz_true_k;
                        else grown->states[grown->states_count++] = target;
                    }
                if (!grown->states_count || crowded) continue;
                memcpy(grown->classes, current[path].classes, sizeof(grown->classes));
                grown->classes[depth] = (sz_u8_t)unit_class;
                grown->length = (sz_u32_t)depth + 1, grown->ended = sz_false_k;
                ++next_count;
            }
        }
        if (crowded) break;
        {
            sz_regex_prefix_path_t *const swapped = current;
            current = next, next = swapped, current_count = next_count, ++depth;
        }
    }
    if (depth == 0) return sz_false_k;
    memset(prefixes, 0, sizeof(*prefixes));
    memset(class_bytes, 0, sizeof(class_bytes));
    for (sz_size_t byte = 0; byte != 256; ++byte) ++class_bytes[engine->byte_to_class[byte]];
    for (sz_size_t path = 0; path != current_count; ++path) {
        sz_u8_t const bucket = (sz_u8_t)(1u << path);
        sz_size_t rate = 1;
        for (sz_size_t offset = 0; offset != sz_regex_prefixes_depth_k; ++offset) {
            sz_bool_t const bound = (sz_bool_t)(offset < current[path].length);
            rate *= bound ? sz_max_of_two(class_bytes[current[path].classes[offset]], (sz_size_t)16) : 256;
            for (sz_size_t byte = 0; byte != 256 && offset < depth; ++byte)
                if (!bound || engine->byte_to_class[byte] == current[path].classes[offset])
                    prefixes->buckets[offset][byte] |= bucket;
        }
        estimate += rate;
    }
    // The rates are in units of one in the cube of the alphabet; past one position in 128, end
    // walks from false candidates would cost more than the start pass they replace.
    if (estimate > ((sz_size_t)1 << 24) / 128) return sz_false_k;
    for (sz_size_t offset = 0; offset != sz_regex_prefixes_depth_k; ++offset)
        for (sz_size_t byte = 0; byte != 256; ++byte) {
            sz_u8_t const buckets = offset < depth ? prefixes->buckets[offset][byte] : (sz_u8_t)0xFF;
            prefixes->nibbles[offset][byte & 15] |= buckets, prefixes->nibbles[offset][16 + (byte >> 4)] |= buckets;
        }
    prefixes->depth = (sz_u32_t)depth;
    return sz_true_k;
}

/** Publishes both automata into the engine's one block: the start pass's records when it runs on
 *  positions, forward rows unless a PikeVM replaces them, reverse rows when it does not, forward
 *  patterns, codepoint classes, the forward NFA with its lookarounds and every pattern's entry and
 *  groups, the byte-to-class map, then the group names. */
STRINGZILLA_INLINE sz_status_t sz_regex_builder_publish_(sz_regex_builder_t *builder, sz_regex_engine_t *engine) {
    sz_allocator_t *const allocator = builder->allocator;
    sz_size_t const unit_columns = builder->classes_count + 1, words = builder->words;
    sz_size_t const stride = sz_regex_stride_((sz_u32_t)builder->classes_count, (sz_u32_t)builder->lookarounds_count);
    sz_size_t const reverse_states = words ? 0 : builder->reverse_dfa.states_count;
    sz_size_t const forward_cells = builder->forward_dfa.states_count * stride, reverse_cells = reverse_states * stride;
    sz_size_t const reach_words = builder->unit_classes_count * words;
    sz_size_t const program_words = words ? reach_words + builder->records_count + 17 * words : 0;
    sz_size_t const ascii_count = words ? 128 : 0, ranges_count = words ? builder->ranges_classes_count : 0;
    sz_size_t const nfa_states = builder->forward.states_count, nfa_links = builder->forward.links_count;
    sz_size_t const patterns_count = builder->patterns_count, groups_count = builder->groups_ends_count - 1;
    sz_size_t const lookarounds_count = builder->lookarounds_count;
    sz_size_t forward_accepting = 0, reverse_accepting = 0, total;
    sz_regex_lookaround_t *lookarounds;
    sz_u64_t *program;
    sz_u32_t *forward_rows, *reverse_rows, *forward_patterns, *ascii, *patterns_starts, *patterns_groups, *groups_ends;
    sz_regex_nfa_state_t *states;
    sz_regex_nfa_link_t *links;
    sz_byteset_t *escapes;
    sz_regex_body_t *bodies;
    sz_regex_prefixes_t *prefixes;
    sz_u32_t *bodies_rows, *looks_bytes;
    sz_u8_t *byte_to_class;
    for (sz_size_t index = 0; index != builder->forward_dfa.states_count; ++index)
        forward_accepting += builder->forward_dfa.keys[builder->forward_dfa.offsets[index]] & 1;
    builder->published_capacity = sz_max_of_two(sz_max_of_two(builder->forward_dfa.states_count, reverse_states), 1);
    builder->published = (sz_u32_t *)allocator->allocate(builder->published_capacity * sizeof(sz_u32_t),
                                                         allocator->handle, builder->stream);
    if (!builder->published) return sz_bad_alloc_k;
    total = program_words * sizeof(sz_u64_t) + builder->forward_dfa.states_count * sizeof(sz_byteset_t) +
            lookarounds_count * sizeof(sz_regex_body_t) + sizeof(sz_regex_prefixes_t) +
            (builder->bodies_rows_count + (lookarounds_count ? sz_regex_singles_k : 0) + forward_cells + reverse_cells +
             forward_accepting + ascii_count + ranges_count) *
                sizeof(sz_u32_t) +
            nfa_states * sizeof(sz_regex_nfa_state_t) + nfa_links * sizeof(sz_regex_nfa_link_t) +
            lookarounds_count * sizeof(sz_regex_lookaround_t) +
            (2 * patterns_count + groups_count + 2) * sizeof(sz_u32_t) + 256 + builder->groups_names_count;
    program = (sz_u64_t *)allocator->allocate(total, allocator->handle, builder->stream);
    if (!program) return sz_bad_alloc_k;
    escapes = (sz_byteset_t *)(program + program_words);
    bodies = (sz_regex_body_t *)(escapes + builder->forward_dfa.states_count);
    prefixes = (sz_regex_prefixes_t *)(bodies + lookarounds_count);
    forward_rows = (sz_u32_t *)(prefixes + 1);
    reverse_rows = forward_rows + forward_cells;
    bodies_rows = reverse_rows + reverse_cells;
    looks_bytes = bodies_rows + builder->bodies_rows_count;
    forward_patterns = looks_bytes + (lookarounds_count ? sz_regex_singles_k : 0);
    ascii = forward_patterns + forward_accepting;
    states = (sz_regex_nfa_state_t *)(ascii + ascii_count + ranges_count);
    links = (sz_regex_nfa_link_t *)(states + nfa_states);
    lookarounds = (sz_regex_lookaround_t *)(links + nfa_links);
    patterns_starts = (sz_u32_t *)(lookarounds + lookarounds_count);
    patterns_groups = patterns_starts + patterns_count;
    groups_ends = patterns_groups + patterns_count + 1;
    byte_to_class = (sz_u8_t *)(groups_ends + groups_count + 1);
    memcpy(byte_to_class, builder->byte_to_class, 256);
    memcpy(states, builder->forward.states, nfa_states * sizeof(sz_regex_nfa_state_t));
    memcpy(links, builder->forward.links, nfa_links * sizeof(sz_regex_nfa_link_t));
    if (lookarounds_count) memcpy(lookarounds, builder->lookarounds, lookarounds_count * sizeof(sz_regex_lookaround_t));
    if (builder->bodies_rows_count)
        memcpy(bodies_rows, builder->bodies_rows, builder->bodies_rows_count * sizeof(sz_u32_t));
    if (lookarounds_count)
        memcpy(looks_bytes, builder->bodies + 3 * lookarounds_count, sz_regex_singles_k * sizeof(sz_u32_t));
    for (sz_size_t index = 0; index != lookarounds_count; ++index) {
        sz_u32_t const *const body = builder->bodies + 3 * index;
        bodies[index].rows = body[0] ? bodies_rows + body[0] - 1 : STRINGZILLA_NULL;
        bodies[index].start = body[1], bodies[index].accepts_begin = body[2];
    }
    memcpy(patterns_starts, builder->patterns_starts, patterns_count * sizeof(sz_u32_t));
    memcpy(patterns_groups, builder->patterns_groups, (patterns_count + 1) * sizeof(sz_u32_t));
    memcpy(groups_ends, builder->groups_ends, (groups_count + 1) * sizeof(sz_u32_t));
    if (builder->groups_names_count) memcpy(byte_to_class + 256, builder->groups_names, builder->groups_names_count);
    if (forward_cells)
        sz_regex_publish_rows_(builder, &builder->forward_dfa, forward_rows, forward_patterns, engine->forward_starts);
    else memset(engine->forward_starts, 0, sizeof(engine->forward_starts));
    for (sz_size_t state = 0; state != builder->forward_dfa.states_count; ++state) {
        sz_u32_t const *const row = forward_rows + state * stride;
        sz_size_t printable = 0;
        sz_byteset_init(escapes + state);
        for (sz_size_t byte = 0; byte != 256; ++byte) {
            sz_size_t block = 0;
            while (block != stride && row[block + byte_to_class[byte]] == state * stride) block += unit_columns;
            if (block != stride)
                sz_byteset_add_u8(escapes + state, (sz_u8_t)byte), printable += byte >= 0x20 && byte < 0x7F;
        }
        if (state == 0 || printable > sz_regex_escapes_limit_k)
            sz_byteset_init(escapes + state), sz_byteset_invert(escapes + state);
    }
    if (words) {
        memcpy(program, builder->reach, reach_words * sizeof(sz_u64_t));
        memcpy(program + reach_words, builder->records, builder->records_count * sizeof(sz_u64_t));
        memcpy(program + reach_words + builder->records_count, builder->all, 17 * words * sizeof(sz_u64_t));
        memcpy(ascii, builder->ascii, sizeof(builder->ascii));
        memcpy(ascii + ascii_count, builder->ranges_classes, ranges_count * sizeof(sz_u32_t));
        memset(engine->reverse_starts, 0, sizeof(engine->reverse_starts));
    }
    else
        reverse_accepting = sz_regex_publish_rows_(builder, &builder->reverse_dfa, reverse_rows, STRINGZILLA_NULL,
                                                   engine->reverse_starts);
    engine->forward_rows = forward_cells ? forward_rows : STRINGZILLA_NULL;
    engine->reverse_rows = reverse_cells ? reverse_rows : STRINGZILLA_NULL;
    engine->nfa_states = states, engine->nfa_links = links;
    engine->nfa_states_count = (sz_u32_t)nfa_states, engine->nfa_links_count = (sz_u32_t)nfa_links;
    engine->nfa_start = builder->forward.start;
    engine->patterns_starts = patterns_starts, engine->patterns_groups = patterns_groups;
    engine->lookarounds = lookarounds_count ? lookarounds : STRINGZILLA_NULL;
    engine->bodies = lookarounds_count ? bodies : STRINGZILLA_NULL;
    engine->looks_bytes = lookarounds_count && looks_bytes[0] ? looks_bytes : STRINGZILLA_NULL;
    engine->lookarounds_count = (sz_u32_t)lookarounds_count;
    engine->groups_names.layout = sz_strings_u32_tape_k, engine->groups_names.count = groups_count;
    engine->groups_names.data.u32_tape.data = (sz_cptr_t)(byte_to_class + 256);
    engine->groups_names.data.u32_tape.offsets = groups_ends;
    engine->forward_patterns = forward_patterns, engine->byte_to_class = byte_to_class;
    engine->forward_escapes = escapes;
    engine->classes_count = (sz_u32_t)builder->classes_count;
    engine->forward_states_count = (sz_u32_t)builder->forward_dfa.states_count;
    engine->reverse_states_count = (sz_u32_t)reverse_states;
    engine->forward_accepts_begin = (sz_u32_t)((builder->forward_dfa.states_count - forward_accepting) * stride);
    engine->reverse_accepts_begin = (sz_u32_t)((reverse_states - reverse_accepting) * stride);
    engine->positions_words = (sz_u32_t)words;
    engine->positions_shifts_count = (sz_u32_t)builder->shifts_count;
    engine->positions_groups_count = (sz_u32_t)builder->groups_count;
    engine->positions_ranges_count = (sz_u32_t)(ranges_count / 2);
    engine->positions_reach = words ? program : STRINGZILLA_NULL;
    engine->positions_shifts = words ? program + reach_words : STRINGZILLA_NULL;
    engine->positions_groups = words ? program + reach_words + builder->shifts_count * (1 + words) : STRINGZILLA_NULL;
    engine->positions_all = words ? program + reach_words + builder->records_count : STRINGZILLA_NULL;
    engine->positions_initial = words ? engine->positions_all + words : STRINGZILLA_NULL;
    engine->positions_ascii = words ? ascii : STRINGZILLA_NULL;
    engine->positions_ranges = words ? ascii + ascii_count : STRINGZILLA_NULL;
    engine->memory = program, engine->memory_bytes = total;
    engine->prefixes = forward_cells && builder->case_sensitivity == sz_cased_k &&
                               sz_regex_prefixes_derive_(engine, prefixes)
                           ? prefixes
                           : STRINGZILLA_NULL;
    sz_byteset_init(&engine->reverse_live);
    if (!words)
        for (sz_size_t byte = 0; byte != 256; ++byte) {
            sz_u32_t const after = engine->reverse_starts[sz_regex_byte_kind_((sz_u8_t)byte)];
            for (sz_size_t kind = 1; kind != 4; ++kind)
                for (sz_size_t block = 0; block != stride; block += unit_columns)
                    if (reverse_rows[engine->reverse_starts[kind] + block + byte_to_class[byte]] != after)
                        sz_byteset_add_u8(&engine->reverse_live, (sz_u8_t)byte);
        }
    else {
        sz_u64_t const *const initial = builder->all + words;
        sz_u64_t const *const groups = builder->records + builder->shifts_count * (1 + words);
        sz_u64_t matched = 0;
        for (sz_size_t kind = 0; kind != 16; ++kind) matched |= initial[kind * words] & 2;
        // The loop's follows across lookarounds sit in groups of its own, which it enters
        // wherever those hold.
        for (sz_size_t group = 0; group != builder->groups_count; ++group)
            if (groups[group * (1 + 2 * words) + 1] & 1) matched |= groups[group * (1 + 2 * words) + 1 + words] & 2;
        for (sz_size_t unit_class = 2; unit_class != builder->unit_classes_count && !matched; ++unit_class) {
            sz_u64_t const *const reach = builder->reach + unit_class * words;
            sz_u64_t entered = 0;
            for (sz_size_t kind = 0; kind != 16; ++kind)
                for (sz_size_t word = 0; word != words; ++word)
                    entered |= initial[kind * words + word] & reach[word] & (word ? ~0ull : ~3ull);
            for (sz_size_t group = 0; group != builder->groups_count; ++group)
                if (groups[group * (1 + 2 * words) + 1] & 1)
                    for (sz_size_t word = 0; word != words; ++word)
                        entered |= groups[group * (1 + 2 * words) + 1 + words + word] & reach[word] &
                                   (word ? ~0ull : ~3ull);
            if (!entered) continue;
            for (sz_size_t byte = 0; byte != 128; ++byte)
                if (builder->ascii[byte] == unit_class) sz_byteset_add_u8(&engine->reverse_live, (sz_u8_t)byte);
            // A live codepoint past ASCII makes every non-ASCII byte live, so a skip always lands
            // on a unit's end.
            for (sz_size_t pair = 0; pair < ranges_count; pair += 2)
                if (builder->ranges_classes[pair + 1] == unit_class)
                    for (sz_size_t byte = 0x80; byte != 256; ++byte)
                        sz_byteset_add_u8(&engine->reverse_live, (sz_u8_t)byte);
        }
        if (matched) sz_byteset_invert(&engine->reverse_live);
    }
    // Uncased walks step the folds of the bytes a skip lands between, and any byte past ASCII can
    // fold into a live one.
    if (builder->case_sensitivity != sz_cased_k)
        for (sz_size_t byte = 0; byte != 256; ++byte)
            if (byte >= 0x80 || sz_byteset_contains_u8(&engine->reverse_live, sz_ascii_fold_((sz_u8_t)byte)))
                sz_byteset_add_u8(&engine->reverse_live, (sz_u8_t)byte);
    return sz_success_k;
}

/**
 *  @brief Compiles @p patterns into @p engine, leaving the round's scratch to grow on demand.
 *
 *  Each pattern parses into a syntax tree; all trees compile into a forward NFA and a reversed one;
 *  one byte-class partition serves both; and two subset constructions turn them into the
 *  end-finding and the start-finding automata. A start-finding construction that passes
 *  @p reverse_budget states gives way to the bit-parallel position automaton of the same reversed
 *  NFA, and an end-finding one that passes @p forward_budget to a PikeVM over the forward NFA,
 *  which the engine keeps either way.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_engine_compile_(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                        sz_case_sensitivity_t case_sensitivity,
                                                        sz_size_t *malformed_index, sz_capability_t capability,
                                                        sz_allocator_t const *requested_allocator,
                                                        sz_size_t reverse_budget, sz_size_t forward_budget,
                                                        sz_stream_t stream) {
    sz_regex_builder_t builder;
    sz_allocator_t resolved;
    sz_status_t status;
    if (requested_allocator) resolved = *requested_allocator;
    else if ((status = sz_allocator_init_heap(&resolved)) != sz_success_k) return status;
    if (patterns->count == 0) {
        if (malformed_index) *malformed_index = 0;
        return sz_invalid_syntax_k;
    }
    if (patterns->count >= sz_regex_none_k) return sz_overflow_risk_k;
    memset(&builder, 0, sizeof(builder));
    builder.allocator = &resolved, builder.stream = stream;
    builder.patterns_count = patterns->count, builder.case_sensitivity = case_sensitivity;
    builder.roots = (sz_u32_t *)resolved.allocate((3 * patterns->count + 1) * sizeof(sz_u32_t), resolved.handle,
                                                  stream);
    status = builder.roots ? sz_success_k : sz_bad_alloc_k;
    if (status == sz_success_k) {
        builder.patterns_starts = builder.roots + patterns->count;
        builder.patterns_groups = builder.patterns_starts + patterns->count;
        status = sz_regex_builder_parse_(&builder, patterns, malformed_index);
    }
    if (status == sz_success_k) status = sz_regex_compile_patterns_(&builder, &builder.forward, sz_regex_forward_k);
    if (status == sz_success_k) status = sz_regex_compile_patterns_(&builder, &builder.reverse, sz_regex_reverse_k);
    if (status == sz_success_k) status = sz_regex_compile_lookarounds_(&builder);
    if (status == sz_success_k) status = sz_regex_builder_scratch_(&builder);
    if (status == sz_success_k) {
        sz_regex_nfa_prefix_(&builder, &builder.forward);
        sz_regex_nfa_prefix_(&builder, &builder.reverse);
        sz_regex_builder_classify_(&builder);
        // Each set of lookarounds that may hold takes a block of columns, so the budgets shrink to
        // keep the tables' bytes.
        if (builder.lookarounds_count > sz_regex_columned_lookarounds_k) forward_budget = 0, reverse_budget = 0;
        else forward_budget >>= builder.lookarounds_count, reverse_budget >>= builder.lookarounds_count;
        builder.forward_dfa.states_budget = forward_budget, builder.reverse_dfa.states_budget = reverse_budget;
        status = sz_regex_determinize_(&builder, &builder.forward, &builder.forward_dfa, sz_regex_leftmost_first_k);
        if (status == sz_overflow_risk_k) builder.forward_dfa.states_count = 0, status = sz_success_k;
        if (status == sz_success_k) {
            status = sz_regex_determinize_(&builder, &builder.reverse, &builder.reverse_dfa, sz_regex_all_matches_k);
            if (status == sz_overflow_risk_k) {
                builder.words = (builder.positions_count + 63) / 64;
                status = sz_regex_collect_follows_(&builder);
                if (status == sz_success_k) status = sz_regex_partition_codepoints_(&builder);
                if (status == sz_success_k) status = sz_regex_build_records_(&builder);
            }
        }
        if (status == sz_success_k) status = sz_regex_build_bodies_(&builder);
        if (status == sz_success_k) status = sz_regex_builder_publish_(&builder, engine);
    }
    sz_regex_builder_free_(&builder);
    if (status != sz_success_k) return status;
    engine->patterns_count = (sz_u32_t)patterns->count;
    engine->case_sensitivity = case_sensitivity;
    engine->capability = capability;
    engine->allocator = resolved;
    engine->scratch = STRINGZILLA_NULL, engine->scratch_bytes = 0;
    return sz_success_k;
}

#pragma endregion Building

#pragma region Matching

/** Windows whose start passes one batch walks at once, so the steps of their chains overlap. */
enum { sz_regex_chains_serial_k = 8 };

/** Position sets walked in lockstep, fewer than DFA chains, as each step holds many more values. */
enum { sz_regex_position_chains_serial_k = 4 };

/** Bytes between the positions where end walks leave their state for later walks to meet. */
enum { sz_regex_checkpoint_spacing_k = 16 };

/** Checkpoints kept, the latest ones by position, so a walk meets one within that
 *  many spacings back. */
enum { sz_regex_checkpoints_k = 256 };

/** Where an end walk crossed a checkpoint, the state it held there, and the match it ended with. */
typedef struct sz_regex_checkpoint_t {

    /** The checkpoint's position in the haystack. */
    sz_size_t position;

    /** The haystack's walk the checkpoint belongs to, so one left by another
     *  haystack never matches. */
    sz_size_t serial;

    /** Where that walk's match ended, if at or past @c position, or @c STRINGZILLA_SIZE_MAX. */
    sz_size_t end;

    /** The end-finding state the walk held at @c position. */
    sz_u32_t state;

    /** The pattern of that match. */
    sz_u32_t pattern;
} sz_regex_checkpoint_t;

/** A PikeVM over the forward NFA: the threads one position holds, in priority order,
 *  and those its step enters. */
typedef struct sz_regex_pike_t {

    /** The position's threads by NFA state, in priority order, and how many there are. */
    sz_u32_t *dense;
    sz_size_t count;

    /** Per NFA state, its index in @c dense while the position holds it. */
    sz_u32_t *sparse;

    /** The states the last step entered, in priority order, repeats included,
     *  and how many there are. */
    sz_u32_t *entered;
    sz_size_t entered_count;

    /** Capture slots, @c slots_count each: per thread of @c dense, per state of @c entered, of the
     *  path being closed, and of the last match; an unset slot holds @c STRINGZILLA_SIZE_MAX. */
    sz_size_t *dense_slots;
    sz_size_t *entered_slots;
    sz_size_t *current;
    sz_size_t *matched;
    sz_size_t slots_count;

    /** The closure's stack of pairs: a state to explore, or a slot past the states' count to
     *  restore, then its value. */
    sz_size_t *frames;

    /** Where the last match ends, its pattern, and whether the last step found it. */
    sz_size_t end;
    sz_u32_t pattern;
    sz_bool_t found;
} sz_regex_pike_t;

/** The scratch one haystack's walk carves out of the engine's round block,
 *  and the windows it marked. */
typedef struct sz_regex_walk_t {

    /** One bit per position of each window of the marked batch, from its first position through one
     *  past its last, @c window_words words per window. */
    sz_u64_t *starts;

    /** Per window, the start pass's state at its first position once known, zero while unknown: a
     *  DFA state in one word, or a position set in @c positions_words, the haystack's end last. */
    sz_u64_t *edges;

    /** Position sets: two per chain, then priming's lower and upper runs and the set
     *  they step into. */
    sz_u64_t *sets;

    /** Every window's first position moved forward to a unit boundary,
     *  then the haystack's length. */
    sz_size_t *bounds;

    /** The checkpoints of end walks, each at the slot of its position. */
    sz_regex_checkpoint_t *checkpoints;

    /** The DFA states priming still holds. */
    sz_u32_t *primed;

    /** The states they step to, deduplicated. */
    sz_u32_t *stepped;

    /** Per DFA state, whether @c stepped holds it already. */
    sz_u8_t *seen;

    /** Per haystack position, the lookarounds holding there as look bits,
     *  or null for a vocabulary without any. */
    sz_u32_t *looks;

    /** Windows the haystack splits into. */
    sz_size_t windows;

    /** Bytes of every window but the last before its first position moves to a unit boundary. */
    sz_size_t window_bytes;

    /** Words of start bits per window, with room for the position past its last and a
     *  boundary's extra bytes. */
    sz_size_t window_words;

    /** The window the cursor was last in. */
    sz_size_t window;

    /** The first window of the marked batch. */
    sz_size_t batch;

    /** Windows in the marked batch, zero before the first. */
    sz_size_t batch_count;

    /** This walk's number among all walks over the scratch, which tags its checkpoints. */
    sz_size_t serial;

    /** What the skipping batches cost in bytes of a lockstep pass, and the bytes they spanned,
     *  which decide whether skipping still pays. */
    sz_size_t skip_cost;
    sz_size_t skip_spanned;

    /** The tier's reverse byte-set search, which skips to the next byte off the idle state. */
    sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *);

    /** The tier's forward byte-set search, which an end walk skips with to the next byte off its
     *  state, or none where a byte loop would be no faster than stepping, as on the serial tier. */
    sz_cptr_t (*find_escape)(sz_cptr_t, sz_size_t, sz_byteset_t const *);

    /** The tier's search for the prefixes of the engine, which replaces the start pass where they
     *  exist. */
    sz_cptr_t (*find_prefixes)(sz_cptr_t, sz_size_t, sz_regex_prefixes_t const *);

    /** The tier's fill of the lookaround table from the bytes beside each position, as
     *  @ref sz_regex_fill_looks_serial_ does. */
    void (*fill_looks)(sz_cptr_t, sz_size_t, sz_u32_t const *, sz_u32_t, sz_u32_t *);

    /** The PikeVM that end walks run without forward rows and groups are read with, empty unless
     *  one of them is. */
    sz_regex_pike_t pike;
} sz_regex_walk_t;

/** Finds the first position of @p text whose bytes keep some bucket of @p prefixes at every offset
 *  inside it, a byte at a time. */
STRINGZILLA_INLINE sz_cptr_t sz_regex_find_prefixes_serial_(sz_cptr_t text, sz_size_t length,
                                                            sz_regex_prefixes_t const *prefixes) {
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_size_t const depth = prefixes->depth;
    for (sz_size_t position = 0; position != length; ++position) {
        sz_u8_t buckets = prefixes->buckets[0][bytes[position]];
        for (sz_size_t offset = 1; buckets && offset != depth && position + offset != length; ++offset)
            buckets &= prefixes->buckets[offset][bytes[position + offset]];
        if (buckets) return text + position;
    }
    return STRINGZILLA_NULL;
}

/**
 *  @brief Fills the lookaround table of @p text with the one-byte lookarounds holding at every
 *      position, by the tables of @p looks_bytes, a byte at a time, and @p flipped XORed in.
 *
 *  A position holds a lookbehind of one byte by the byte before it and a lookahead by the byte at
 *  it; neither side holds past an edge of the haystack.
 */
STRINGZILLA_INLINE void sz_regex_fill_looks_serial_(sz_cptr_t text, sz_size_t length, sz_u32_t const *looks_bytes,
                                                    sz_u32_t flipped, sz_u32_t *looks) {
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u32_t const *const after = looks_bytes + 1, *const before = after + 256;
    sz_u32_t previous = 0;
    for (sz_size_t position = 0; position != length; ++position)
        looks[position] = (previous | before[bytes[position]]) ^ flipped, previous = after[bytes[position]];
    looks[length] = previous ^ flipped;
}

/** The first unit boundary at or past @p position: a byte that does not continue a codepoint
 *  begun before it. */
STRINGZILLA_INLINE sz_size_t sz_regex_boundary_(sz_cptr_t haystack, sz_size_t length, sz_size_t position) {
    sz_size_t back = 1;
    sz_rune_t rune;
    sz_rune_length_t rune_length;
    if (position == length || !sz_utf8_is_continuation_((sz_u8_t)haystack[position])) return position;
    while (back != 4 && back <= position && sz_utf8_is_continuation_((sz_u8_t)haystack[position - back])) ++back;
    if (back == 4 || back > position) return position;
    rune_length = sz_rune_decode(haystack + position - back, haystack + length, &rune);
    return (sz_size_t)rune_length > back ? position - back + (sz_size_t)rune_length : position;
}

/**
 *  @brief Whether so few bytes of `[begin, end)` leave the start pass's idle states that skipping
 *      between them beats walking in lockstep, judged by 64 evenly spaced samples, as in
 *      @c substrings.
 *
 *  A DFA skips once fewer than one in 46 is live, as each costs a search and a few slow steps, or
 *  one in 20 where the tier's searches are @p vectorized, passing a block per compare; a position
 *  walk skips once fewer than one in six is live, as its lockstep steps are slow too.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_live_sparse_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t begin,
                                                   sz_size_t end, sz_bool_t vectorized) {
    sz_size_t const samples = 64, stride = end - begin > samples ? (end - begin) / samples : 1;
    sz_size_t const ratio = engine->positions_words ? 6 : vectorized ? 20 : 46;
    sz_size_t position, taken = 0, live = 0;
    for (position = begin; position < end && taken != samples; position += stride, ++taken)
        live += sz_byteset_contains_u8(&engine->reverse_live, (sz_u8_t)haystack[position]);
    return (sz_bool_t)(live * ratio < taken);
}

/** One unit of a haystack as an uncased walk steps it: the case folds of one codepoint,
 *  or one malformed byte. */
typedef struct sz_regex_image_t {

    /** The folded runes back to back in UTF-8, a malformed byte as itself; nine bytes at most. */
    sz_u8_t bytes[12];

    /** The folded runes, a malformed byte tagged as @c sz_utf8_folded_iter_t tags it. */
    sz_rune_t runes[3];

    /** Where each rune's bytes start. */
    sz_u8_t rune_starts[3];

    /** Runes of the image. */
    sz_u8_t runes_count;

    /** Bytes of the image. */
    sz_u8_t length;
} sz_regex_image_t;

/** Appends one folded rune to @p image. */
STRINGZILLA_INLINE void sz_regex_image_push_(sz_regex_image_t *image, sz_rune_t rune) {
    image->runes[image->runes_count] = rune, image->rune_starts[image->runes_count++] = image->length;
    if (rune & 0x80000000u) image->bytes[image->length++] = (sz_u8_t)rune;
    else image->length += (sz_u8_t)sz_rune_encode(rune, image->bytes + image->length);
}

/** Folds the unit starting at @p position, which precedes @p length,
 *  answering where the unit ends. */
STRINGZILLA_INLINE sz_size_t sz_regex_image_after_(sz_cptr_t haystack, sz_size_t length, sz_size_t position,
                                                   sz_regex_image_t *image) {
    sz_utf8_folded_iter_t runes;
    sz_rune_t rune = 0; // Never read unset: `position` precedes `length`, so every step yields a rune
    sz_utf8_folded_iter_init_(&runes, haystack + position, length - position);
    image->runes_count = 0, image->length = 0;
    do {
        sz_utf8_folded_iter_next_(&runes, &rune);
        sz_regex_image_push_(image, rune);
    } while (runes.pending_index < runes.pending_count);
    return position + runes.codepoint_length;
}

/** Folds the unit ending at @p end, which is past zero, answering where the unit begins. */
STRINGZILLA_INLINE sz_size_t sz_regex_image_before_(sz_cptr_t haystack, sz_size_t end, sz_regex_image_t *image) {
    sz_utf8_folded_reverse_iter_t runes;
    sz_rune_t reversed[3];
    sz_size_t count = 0;
    sz_utf8_folded_reverse_iter_init_(&runes, haystack, haystack + end);
    do sz_utf8_folded_reverse_iter_prev_(&runes, reversed + count++);
    while (runes.pending_index < runes.pending_count);
    image->runes_count = 0, image->length = 0;
    while (count) sz_regex_image_push_(image, reversed[--count]);
    return (sz_size_t)(runes.ptr - haystack);
}

/** The boundary kind of a folded rune: one for a line feed, two for an ASCII word byte,
 *  three for anything else. */
STRINGZILLA_CONSTEXPR sz_size_t sz_regex_rune_kind_(sz_rune_t rune) {
    return rune < 0x80 ? sz_regex_byte_kind_((sz_u8_t)rune) : 3;
}

STRINGZILLA_OUTLINED_ void sz_regex_looks_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t length,
                                                  sz_regex_walk_t *walk);

/**
 *  @brief Grows the engine's round block for @p haystack and carves @p walk out of it, its windows
 *      aligned to units.
 *
 *  The block starts with a count of the walks it served, which numbers each walk's checkpoints,
 *  then the checkpoints, both at offsets no haystack moves, so every checkpoint left there
 *  carries an older number; growing the block clears them. A PikeVM keeping @p slots_count
 *  capture slots per thread is carved too where end walks need it, slots are asked for, or a
 *  vocabulary with lookarounds runs it to fill the table of where they hold, which happens here,
 *  before any walk reads it.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_walk_scratch_(
    sz_regex_engine_t *engine, sz_cptr_t haystack, sz_size_t length,
    sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_escape)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_prefixes)(sz_cptr_t, sz_size_t, sz_regex_prefixes_t const *),
    void (*fill_looks)(sz_cptr_t, sz_size_t, sz_u32_t const *, sz_u32_t, sz_u32_t *), sz_size_t slots_count,
    sz_regex_walk_t *walk) {
    sz_size_t const batch_bytes = (sz_size_t)sz_regex_chains_serial_k * sz_regex_window_serial_k;
    sz_size_t const batches = sz_size_divide_round_up(length, batch_bytes);
    sz_bool_t const guessed = (sz_bool_t)(engine->case_sensitivity == sz_cased_k && !engine->positions_words);
    // Where prefixes replace the start pass, no window is marked; a haystack too short for windows
    // that prime splits into guessed ones only to walk in lockstep, which sparse live bytes leave
    // to one skipping walk.
    sz_size_t const chained =
        !engine->prefixes &&
        (length >= (sz_size_t)sz_regex_chains_serial_k * sz_regex_window_minimum_k ||
         (guessed && length >= (sz_size_t)sz_regex_chains_serial_k * sz_regex_window_guessed_k &&
          !sz_regex_live_sparse_(engine, haystack, 0, length, (sz_bool_t)(find_escape != STRINGZILLA_NULL))));
    sz_size_t const window_bytes = chained ? sz_size_divide_round_up(length, batches * sz_regex_chains_serial_k)
                                           : length + 1;
    sz_size_t const windows = chained ? sz_size_divide_round_up(length, window_bytes) : 1;
    sz_size_t const words = engine->positions_words, stride = words ? words : 1;
    sz_size_t const priming = windows > 1 && !words ? engine->reverse_states_count : 0;
    sz_size_t const window_words = (chained ? window_bytes : length) / 64 + 2;
    sz_size_t const words_count = 1 + (chained ? sz_regex_chains_serial_k : 1) * window_words + (windows + 1) * stride +
                                  (2 * sz_regex_chains_serial_k + 3) * words;
    sz_size_t const pike_states = !engine->forward_rows || slots_count || engine->lookarounds_count
                                      ? engine->nfa_states_count
                                      : 0;
    sz_size_t const pike_entered = pike_states ? engine->nfa_links_count + 1 : 0;
    sz_size_t const pike_slots = (pike_states + pike_entered + 2) * slots_count + 2 * (pike_states + pike_entered);
    sz_size_t const looks_count = engine->lookarounds_count ? length + 1 : 0;
    sz_size_t const bytes = words_count * sizeof(sz_u64_t) + (windows + 1 + pike_slots) * sizeof(sz_size_t) +
                            sz_regex_checkpoints_k * sizeof(sz_regex_checkpoint_t) +
                            (2 * priming + 2 * pike_states + pike_entered + looks_count) * sizeof(sz_u32_t) + priming;
    sz_size_t *served;
    if (engine->scratch_bytes < bytes) {
        void *const grown = engine->allocator.allocate(bytes, engine->allocator.handle, STRINGZILLA_NULL);
        if (!grown) return sz_bad_alloc_k;
        if (engine->scratch)
            engine->allocator.free(engine->scratch, engine->scratch_bytes, engine->allocator.handle, STRINGZILLA_NULL);
        engine->scratch = grown, engine->scratch_bytes = bytes;
        memset(grown, 0, bytes);
    }
    served = (sz_size_t *)engine->scratch;
    walk->checkpoints = (sz_regex_checkpoint_t *)((sz_u64_t *)engine->scratch + 1);
    walk->starts = (sz_u64_t *)(walk->checkpoints + sz_regex_checkpoints_k);
    walk->edges = walk->starts + (chained ? sz_regex_chains_serial_k : 1) * window_words;
    walk->sets = walk->edges + (windows + 1) * stride;
    walk->bounds = (sz_size_t *)(walk->sets + (2 * sz_regex_chains_serial_k + 3) * words);
    walk->pike.dense_slots = walk->bounds + windows + 1;
    walk->pike.entered_slots = walk->pike.dense_slots + pike_states * slots_count;
    walk->pike.current = walk->pike.entered_slots + pike_entered * slots_count;
    walk->pike.matched = walk->pike.current + slots_count;
    walk->pike.frames = walk->pike.matched + slots_count;
    walk->primed = (sz_u32_t *)(walk->pike.frames + 2 * (pike_states + pike_entered));
    walk->stepped = walk->primed + priming;
    walk->pike.dense = walk->stepped + priming;
    walk->pike.sparse = walk->pike.dense + pike_states;
    walk->pike.entered = walk->pike.sparse + pike_states;
    walk->looks = looks_count ? walk->pike.entered + pike_entered : STRINGZILLA_NULL;
    walk->seen = (sz_u8_t *)(walk->pike.entered + pike_entered + looks_count);
    walk->pike.slots_count = slots_count;
    walk->windows = windows, walk->window_bytes = window_bytes, walk->window_words = window_words, walk->window = 0;
    walk->batch = 0, walk->batch_count = 0, walk->serial = ++*served;
    walk->skip_cost = 0, walk->skip_spanned = 0, walk->rfind_live = rfind_live, walk->find_escape = find_escape;
    walk->find_prefixes = find_prefixes, walk->fill_looks = fill_looks;
    memset(walk->edges, 0, (windows + 1) * stride * sizeof(sz_u64_t));
    memset(walk->seen, 0, priming);
    walk->bounds[0] = 0, walk->bounds[windows] = length;
    for (sz_size_t window = 1; window < windows; ++window)
        walk->bounds[window] = sz_regex_boundary_(haystack, length, window * window_bytes);
    if (looks_count) sz_regex_looks_serial_(engine, haystack, length, walk);
    return sz_success_k;
}

/**
 *  @brief Finds the unit ending at @p end: an ASCII byte, a whole codepoint, or one malformed byte.
 *  @param[out] unit_class Its unit class, zero for a malformed byte.
 *  @return Where it begins.
 */
STRINGZILLA_INLINE sz_size_t sz_regex_unit_before_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t end,
                                                   sz_size_t *unit_class) {
    sz_u8_t const last = (sz_u8_t)haystack[end - 1];
    sz_size_t back = 2, low = 0, high = engine->positions_ranges_count;
    sz_rune_t rune = 0;
    if (last < 0x80) {
        *unit_class = engine->positions_ascii[last];
        return end - 1;
    }
    *unit_class = 0;
    if (!sz_utf8_is_continuation_(last)) return end - 1;
    while (back != 5 && back <= end && sz_utf8_is_continuation_((sz_u8_t)haystack[end - back])) ++back;
    if (back == 5 || back > end || (sz_size_t)sz_rune_decode(haystack + end - back, haystack + end, &rune) != back)
        return end - 1;
    while (high - low > 1) {
        sz_size_t const middle = (low + high) / 2;
        if (engine->positions_ranges[middle * 2] <= rune) low = middle;
        else high = middle;
    }
    *unit_class = engine->positions_ranges[low * 2 + 1];
    return end - back;
}

/** The unit class of a folded rune, zero for a malformed byte. */
STRINGZILLA_INLINE sz_size_t sz_regex_rune_class_(sz_regex_engine_t const *engine, sz_rune_t rune) {
    sz_size_t low = 0, high = engine->positions_ranges_count;
    if (rune < 0x80) return engine->positions_ascii[rune];
    if (rune & 0x80000000u) return 0;
    while (high - low > 1) {
        sz_size_t const middle = (low + high) / 2;
        if (engine->positions_ranges[middle * 2] <= rune) low = middle;
        else high = middle;
    }
    return engine->positions_ranges[low * 2 + 1];
}

/** The lookarounds holding at @p position of a walk's table, none without a table. */
STRINGZILLA_INLINE sz_u32_t sz_regex_looks_at_(sz_u32_t const *looks, sz_size_t position) {
    return looks ? looks[position] : 0;
}

/** The first column of the DFA block the lookarounds holding at @p position select, zero without a
 *  table, so a walk passed a literal null steps with no loads beyond the row. */
STRINGZILLA_INLINE sz_u32_t sz_regex_block_at_(sz_regex_engine_t const *engine, sz_u32_t const *looks,
                                               sz_size_t position) {
    return looks ? (sz_u32_t)sz_regex_looks_block_(engine->classes_count, engine->lookarounds_count, looks[position])
                 : 0;
}

/** Steps the position set @p state over one unit of @p unit_class at a boundary of @p kind where
 *  the lookarounds of @p looks hold, into @p next; callers pass @p words as a literal where they
 *  can, which unrolls every loop over it. */
STRINGZILLA_INLINE void sz_regex_positions_step_(sz_regex_engine_t const *engine, sz_size_t words,
                                                 sz_u64_t const *state, sz_u64_t *next, sz_size_t unit_class,
                                                 sz_size_t kind, sz_u32_t looks) {
    sz_u64_t const *record = engine->positions_shifts;
    sz_u64_t const *const reach = engine->positions_reach + unit_class * words;
    sz_u64_t const *const initial = engine->positions_initial + kind * words;
    sz_size_t word;
    for (word = 0; word != words; ++word) next[word] = initial[word];
    for (sz_size_t shift = 0; shift != engine->positions_shifts_count; ++shift, record += 1 + words) {
        sz_i64_t const distance = (sz_i64_t)record[0];
        sz_size_t const span = (sz_size_t)(distance < 0 ? -distance : distance), offset = span / 64, bits = span % 64;
        sz_size_t const carry = 64 - bits;
        sz_u64_t const *const mask = record + 1;
        for (word = offset; word < words; ++word) {
            sz_size_t const low = word - offset;
            if (distance >= 0) {
                next[word] |= (state[low] & mask[low]) << bits;
                if (bits && low) next[word] |= (state[low - 1] & mask[low - 1]) >> carry;
            }
            else {
                next[low] |= (state[word] & mask[word]) >> bits;
                if (bits && word + 1 != words) next[low] |= (state[word + 1] & mask[word + 1]) << carry;
            }
        }
    }
    record = engine->positions_groups;
    for (sz_size_t group = 0; group != engine->positions_groups_count; ++group, record += 1 + 2 * words) {
        sz_u64_t any = 0, enabled;
        for (word = 0; word != words; ++word) any |= state[word] & record[1 + word];
        enabled = (sz_u64_t)0 -
                  (sz_u64_t)((any != 0) & (record[0] >> kind & 1) & ((record[0] >> 16 & ~(sz_u64_t)looks) == 0));
        for (word = 0; word != words; ++word) next[word] |= record[1 + words + word] & enabled;
    }
    for (word = 0; word != words; ++word) next[word] &= reach[word];
}

/** @ref sz_regex_positions_step_ for a one-word set, which stays in a register. */
STRINGZILLA_INLINE sz_u64_t sz_regex_positions_step_one_(sz_regex_engine_t const *engine, sz_u64_t state,
                                                         sz_size_t unit_class, sz_size_t kind, sz_u32_t looks) {
    sz_u64_t next = engine->positions_initial[kind];
    sz_u64_t const *record = engine->positions_shifts;
    for (sz_size_t shift = 0; shift != engine->positions_shifts_count; ++shift, record += 2) {
        sz_i64_t const distance = (sz_i64_t)record[0];
        next |= distance >= 0 ? (state & record[1]) << distance : (state & record[1]) >> -distance;
    }
    record = engine->positions_groups;
    for (sz_size_t group = 0; group != engine->positions_groups_count; ++group, record += 3)
        next |= record[2] & ((sz_u64_t)0 - (sz_u64_t)(((state & record[1]) != 0) & (record[0] >> kind & 1) &
                                                      ((record[0] >> 16 & ~(sz_u64_t)looks) == 0)));
    return next & engine->positions_reach[unit_class];
}

/**
 *  @brief Finds the start-finding state at the first position of @p window without walking from the
 *      haystack's end, and records it there.
 *
 *  Every state walks the bytes right of the edge toward it together, deduplicated, until one state
 *  remains: whichever state the walk from the end held there, it is now that one. A distance that
 *  ends uncollapsed is doubled; one that reaches the end starts from the true start instead. Every
 *  window edge passed once collapsed is recorded, so no later window primes over the same bytes.
 */
STRINGZILLA_INLINE sz_u32_t sz_regex_prime_serial_(sz_regex_engine_t const *engine, sz_u8_t const *bytes,
                                                   sz_size_t length, sz_size_t window, sz_regex_walk_t *walk) {
    sz_u32_t const *const rows = engine->reverse_rows;
    sz_u8_t const *const byte_to_class = engine->byte_to_class;
    sz_u32_t const stride = (sz_u32_t)sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_u32_t const states_count = engine->reverse_states_count;
    sz_size_t const edge = walk->bounds[window];
    for (sz_size_t distance = 32;; distance *= 2) {
        sz_size_t const far_edge = length - edge > distance ? edge + distance : length;
        sz_size_t position = far_edge, primed_count, record = walk->windows;
        sz_u32_t *primed = walk->primed, *stepped = walk->stepped;
        if (far_edge == length) primed[0] = engine->reverse_starts[0], primed_count = 1;
        else {
            for (sz_u32_t state = 1; state != states_count; ++state) primed[state - 1] = state * stride;
            primed_count = states_count - 1;
        }
        while (primed_count > 1 && position > edge) {
            sz_u32_t const column = sz_regex_block_at_(engine, walk->looks, position) +
                                    byte_to_class[bytes[position - 1]];
            sz_size_t stepped_count = 0;
            for (sz_size_t index = 0; index != primed_count; ++index) {
                sz_u32_t const next = rows[primed[index] + column];
                // The true walk never dies inside a haystack, so a state that does is not the one.
                if (next == 0 || walk->seen[next / stride]) continue;
                walk->seen[next / stride] = 1, stepped[stepped_count++] = next;
            }
            for (sz_size_t index = 0; index != stepped_count; ++index) walk->seen[stepped[index] / stride] = 0;
            primed_count = stepped_count, --position;
            {
                sz_u32_t *const swapped = primed;
                primed = stepped, stepped = swapped;
            }
        }
        if (primed_count == 1) {
            sz_u32_t state = primed[0];
            for (;;) {
                while (walk->bounds[record] > position) --record;
                if (walk->bounds[record] == position) walk->edges[record] = state;
                if (position == edge) return state;
                state = rows[state + sz_regex_block_at_(engine, walk->looks, position) +
                             byte_to_class[bytes[position - 1]]];
                --position;
            }
        }
    }
}

/** @ref sz_regex_prime_serial_ over the folds of whole units, for an uncased engine. */
STRINGZILLA_INLINE sz_u32_t sz_regex_prime_folded_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                   sz_size_t length, sz_size_t window, sz_regex_walk_t *walk) {
    sz_u32_t const *const rows = engine->reverse_rows;
    sz_u8_t const *const byte_to_class = engine->byte_to_class;
    sz_u32_t const stride = (sz_u32_t)sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_u32_t const states_count = engine->reverse_states_count;
    sz_size_t const edge = walk->bounds[window];
    for (sz_size_t distance = 32;; distance *= 2) {
        sz_size_t const far_edge = length - edge > distance ? sz_regex_boundary_(haystack, length, edge + distance)
                                                            : length;
        sz_size_t position = far_edge, primed_count, record = walk->windows;
        sz_u32_t *primed = walk->primed, *stepped = walk->stepped, block;
        sz_regex_image_t image;
        if (far_edge == length) primed[0] = engine->reverse_starts[0], primed_count = 1;
        else {
            for (sz_u32_t state = 1; state != states_count; ++state) primed[state - 1] = state * stride;
            primed_count = states_count - 1;
        }
        while (primed_count > 1 && position > edge) {
            block = sz_regex_block_at_(engine, walk->looks, position);
            position = sz_regex_image_before_(haystack, position, &image);
            for (sz_size_t byte = image.length; byte--; block = 0) {
                sz_u32_t const column = block + byte_to_class[image.bytes[byte]];
                sz_size_t stepped_count = 0;
                for (sz_size_t index = 0; index != primed_count; ++index) {
                    sz_u32_t const next = rows[primed[index] + column];
                    if (next == 0 || walk->seen[next / stride]) continue;
                    walk->seen[next / stride] = 1, stepped[stepped_count++] = next;
                }
                for (sz_size_t index = 0; index != stepped_count; ++index) walk->seen[stepped[index] / stride] = 0;
                primed_count = stepped_count;
                {
                    sz_u32_t *const swapped = primed;
                    primed = stepped, stepped = swapped;
                }
            }
        }
        if (primed_count == 1) {
            sz_u32_t state = primed[0];
            for (;;) {
                while (walk->bounds[record] > position) --record;
                if (walk->bounds[record] == position) walk->edges[record] = state;
                if (position == edge) return state;
                block = sz_regex_block_at_(engine, walk->looks, position);
                position = sz_regex_image_before_(haystack, position, &image);
                for (sz_size_t byte = image.length; byte--; block = 0)
                    state = rows[state + block + byte_to_class[image.bytes[byte]]];
            }
        }
    }
}

/**
 *  @brief Finds the position set at the first position of @p window without walking from the
 *      haystack's end, and records it there.
 *
 *  A step is monotone in its set, so a run from the loop alone and a run from every position a walk
 *  can hold bound the true set from both sides, and where they meet it is known. A distance that
 *  ends apart is doubled, one that reaches the end walks from the true start instead, and every
 *  window edge passed once the set is known is recorded.
 */
STRINGZILLA_INLINE sz_u64_t const *sz_regex_prime_positions_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                                    sz_size_t length, sz_size_t window,
                                                                    sz_regex_walk_t *walk) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_size_t const words = engine->positions_words, edge = walk->bounds[window];
    for (sz_size_t distance = 32;; distance *= 2) {
        sz_size_t const far_edge = length - edge > distance ? sz_regex_boundary_(haystack, length, edge + distance)
                                                            : length;
        sz_size_t position = far_edge, record = walk->windows,
                  right_kind = far_edge == length ? 0 : sz_regex_byte_kind_(bytes[far_edge]);
        sz_u64_t *lower = walk->sets + 2 * sz_regex_chains_serial_k * words, *upper = lower + words,
                 *next = upper + words;
        sz_bool_t known = (sz_bool_t)(far_edge == length);
        memset(lower, 0, words * sizeof(sz_u64_t));
        lower[0] = 1;
        memcpy(upper, engine->positions_all, words * sizeof(sz_u64_t));
        while (position > edge) {
            sz_size_t unit_class, left_kind = sz_regex_byte_kind_(bytes[position - 1]);
            sz_size_t const begin = sz_regex_unit_before_(engine, haystack, position, &unit_class);
            sz_u32_t const looks = sz_regex_looks_at_(walk->looks, position);
            sz_u64_t *swapped;
            sz_regex_positions_step_(engine, words, lower, next, unit_class, left_kind * 4 + right_kind, looks);
            swapped = lower, lower = next, next = swapped;
            if (!known) {
                sz_regex_positions_step_(engine, words, upper, next, unit_class, left_kind * 4 + right_kind, looks);
                swapped = upper, upper = next, next = swapped;
                known = sz_equal_serial_((sz_cptr_t)lower, (sz_cptr_t)upper, words * sizeof(sz_u64_t));
            }
            right_kind = left_kind, position = begin;
            while (walk->bounds[record] > position) --record;
            if (known && walk->bounds[record] == position)
                memcpy(walk->edges + record * words, lower, words * sizeof(sz_u64_t));
        }
        if (known) return walk->edges + window * words;
    }
}

/** @ref sz_regex_prime_positions_serial_ over folded runes, for an uncased engine. */
STRINGZILLA_INLINE sz_u64_t const *sz_regex_prime_positions_folded_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                                    sz_size_t length, sz_size_t window,
                                                                    sz_regex_walk_t *walk) {
    sz_size_t const words = engine->positions_words, edge = walk->bounds[window];
    for (sz_size_t distance = 32;; distance *= 2) {
        sz_size_t const far_edge = length - edge > distance ? sz_regex_boundary_(haystack, length, edge + distance)
                                                            : length;
        sz_size_t position = far_edge, record = walk->windows, right_kind = 0;
        sz_u64_t *lower = walk->sets + 2 * sz_regex_chains_serial_k * words, *upper = lower + words,
                 *next = upper + words;
        sz_bool_t known = (sz_bool_t)(far_edge == length);
        sz_regex_image_t image;
        if (far_edge != length) {
            sz_regex_image_after_(haystack, length, far_edge, &image);
            right_kind = sz_regex_rune_kind_(image.runes[0]);
        }
        memset(lower, 0, words * sizeof(sz_u64_t));
        lower[0] = 1;
        memcpy(upper, engine->positions_all, words * sizeof(sz_u64_t));
        while (position > edge) {
            sz_u32_t const unit_looks = sz_regex_looks_at_(walk->looks, position);
            position = sz_regex_image_before_(haystack, position, &image);
            for (sz_size_t rune = image.runes_count; rune--;) {
                sz_size_t const unit_class = sz_regex_rune_class_(engine, image.runes[rune]);
                sz_size_t const left_kind = sz_regex_rune_kind_(image.runes[rune]);
                sz_u32_t const looks = rune + 1 == image.runes_count ? unit_looks : 0;
                sz_u64_t *swapped;
                sz_regex_positions_step_(engine, words, lower, next, unit_class, left_kind * 4 + right_kind, looks);
                swapped = lower, lower = next, next = swapped;
                if (!known) {
                    sz_regex_positions_step_(engine, words, upper, next, unit_class, left_kind * 4 + right_kind, looks);
                    swapped = upper, upper = next, next = swapped;
                    known = sz_equal_serial_((sz_cptr_t)lower, (sz_cptr_t)upper, words * sizeof(sz_u64_t));
                }
                right_kind = left_kind;
            }
            while (walk->bounds[record] > position) --record;
            if (known && walk->bounds[record] == position)
                memcpy(walk->edges + record * words, lower, words * sizeof(sz_u64_t));
        }
        if (known) return walk->edges + window * words;
    }
}

/**
 *  @brief Marks every position in `[low, edge]` that some match starts at, walking right to left
 *      from the start-finding state at @p edge; the bit of position @c p is `p - base`, ORed in.
 *  @param[out] final The state after the byte before @p low, or after the haystack's first byte
 *      where @p low is zero.
 *  @return What it cost in bytes of a lockstep pass, which steps one byte about three times as fast
 *      and covers forty bytes in the time of one byte-set search.
 *
 *  While the walk sits on an idle start state, the reverse byte-set search jumps to the next byte
 *  that leaves them, as nothing between can start a match; the walk lands on the start for the kind
 *  of the last byte it skipped. Steps read @p looks for their blocks, which callers pass as a
 *  literal null for a vocabulary without lookarounds.
 */
STRINGZILLA_INLINE sz_size_t sz_regex_mark_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t low,
                                                   sz_size_t edge, sz_size_t base, sz_u32_t state,
                                                   sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
                                                   sz_u32_t const *looks, sz_u64_t *starts, sz_u32_t *final) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_u32_t const *const rows = engine->reverse_rows, *const idles = engine->reverse_starts;
    sz_u8_t const *const byte_to_class = engine->byte_to_class;
    sz_u32_t const accepts = engine->reverse_accepts_begin;
    sz_size_t const stop = low ? low - 1 : 0;
    sz_size_t position = edge, cost = 0;
    while (position > stop) {
        if ((state == idles[1] || state == idles[2] || state == idles[3]) &&
            !sz_byteset_contains_u8(&engine->reverse_live, bytes[position - 1])) {
            sz_cptr_t const live = rfind_live(haystack + stop, position - stop, &engine->reverse_live);
            cost += 40;
            position = live ? (sz_size_t)(live - haystack) + 1 : stop;
            state = idles[sz_regex_byte_kind_(bytes[position])];
            if (!live) break;
        }
        state = rows[state + sz_regex_block_at_(engine, looks, position) + byte_to_class[bytes[position - 1]]];
        --position, cost += 3;
        if (state >= accepts) starts[(position + 1 - base) / 64] |= 1ull << ((position + 1 - base) % 64);
    }
    if (low == 0 && rows[state + sz_regex_block_at_(engine, looks, 0) + engine->classes_count] >= accepts)
        starts[0] |= 1;
    *final = state;
    return cost;
}

/**
 *  @brief Marks every unit boundary in `[low, edge]` that some match starts at, stepping
 *      position sets right to left from @p entry at @p edge; the bit of position @c p is
 *      `p - base`, ORed in.
 *  @return What it cost in bytes of a lockstep pass, which steps one unit in about the time of one
 *      byte and covers four bytes in the time of one byte-set search.
 *
 *  A match starts at a boundary when the step over the unit before it enters the match position, so
 *  the start of the haystack takes one more step, over the edge class. While only the loop is live,
 *  the reverse byte-set search jumps to the next byte that enters another position.
 */
STRINGZILLA_INLINE sz_size_t sz_regex_mark_positions_serial_(sz_regex_engine_t const *engine, sz_size_t words,
                                                             sz_cptr_t haystack, sz_size_t length, sz_size_t low,
                                                             sz_size_t edge, sz_size_t base, sz_u64_t const *entry,
                                                             sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t,
                                                                                     sz_byteset_t const *),
                                                             sz_u32_t const *looks, sz_u64_t *sets, sz_u64_t *starts) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_size_t const stop = low ? low - 1 : 0;
    sz_size_t position = edge, right_kind = edge == length ? 0 : sz_regex_byte_kind_(bytes[edge]), cost = 0;
    sz_u64_t *state = sets, *next = sets + words;
    memcpy(state, entry, words * sizeof(sz_u64_t));
    while (position != 0 && position >= low) {
        sz_size_t unit_class, left_kind = sz_regex_byte_kind_(bytes[position - 1]), begin;
        sz_u64_t *const swapped = state;
        sz_u64_t others = state[0] ^ 1;
        for (sz_size_t word = 1; word != words; ++word) others |= state[word];
        if (!others && !sz_byteset_contains_u8(&engine->reverse_live, bytes[position - 1])) {
            sz_cptr_t const live = rfind_live(haystack + stop, position - stop, &engine->reverse_live);
            cost += 4;
            position = live ? (sz_size_t)(live - haystack) + 1 : 0;
            right_kind = position == length ? 0 : sz_regex_byte_kind_(bytes[position]);
            if (!live) break;
            left_kind = sz_regex_byte_kind_(bytes[position - 1]);
        }
        begin = sz_regex_unit_before_(engine, haystack, position, &unit_class);
        sz_regex_positions_step_(engine, words, state, next, unit_class, left_kind * 4 + right_kind,
                                 sz_regex_looks_at_(looks, position));
        state = next, next = swapped, ++cost;
        starts[(position - base) / 64] |= (state[0] >> 1 & 1) << ((position - base) % 64);
        right_kind = left_kind, position = begin;
    }
    if (position == 0 && low == 0) {
        sz_regex_positions_step_(engine, words, state, next, 1, right_kind, sz_regex_looks_at_(looks, 0));
        starts[0] |= next[0] >> 1 & 1;
    }
    return cost;
}

/**
 *  @brief @ref sz_regex_mark_serial_ over the folds of whole units, for an uncased engine.
 *
 *  A start inside the folds of one codepoint snaps outward to the codepoint's own start. Every byte
 *  past ASCII is live, so a skip only passes bytes whose folds are themselves.
 */
STRINGZILLA_INLINE void sz_regex_mark_folded_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t low,
                                              sz_size_t edge, sz_size_t base, sz_u32_t state,
                                              sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
                                              sz_u32_t const *looks, sz_u64_t *starts, sz_u64_t *low_edge) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_u32_t const *const rows = engine->reverse_rows, *const idles = engine->reverse_starts;
    sz_u8_t const *const byte_to_class = engine->byte_to_class;
    sz_u32_t const accepts = engine->reverse_accepts_begin;
    sz_size_t const stop = low ? low - 1 : 0;
    sz_size_t position = edge;
    sz_regex_image_t image;
    if (position == low) *low_edge = state;
    while (position > stop) {
        sz_size_t begin;
        sz_u32_t block;
        if ((state == idles[1] || state == idles[2] || state == idles[3]) &&
            !sz_byteset_contains_u8(&engine->reverse_live, bytes[position - 1])) {
            sz_cptr_t const live = rfind_live(haystack + stop, position - stop, &engine->reverse_live);
            if (!live) {
                if (position > low) *low_edge = idles[sz_regex_byte_kind_(bytes[low])];
                state = idles[sz_regex_byte_kind_(bytes[stop])];
                break;
            }
            position = (sz_size_t)(live - haystack) + 1;
            state = idles[sz_regex_byte_kind_(bytes[position])];
            if (position == low) *low_edge = state;
        }
        block = sz_regex_block_at_(engine, looks, position);
        begin = sz_regex_image_before_(haystack, position, &image);
        for (sz_size_t byte = image.length; byte--; block = 0) {
            sz_size_t const start = byte + 1 == image.length ? position : begin;
            state = rows[state + block + byte_to_class[image.bytes[byte]]];
            if (state >= accepts && start >= low) starts[(start - base) / 64] |= 1ull << ((start - base) % 64);
        }
        position = begin;
        if (position == low) *low_edge = state;
    }
    if (low == 0 && rows[state + sz_regex_block_at_(engine, looks, 0) + engine->classes_count] >= accepts)
        starts[0] |= 1;
}

/** @ref sz_regex_mark_positions_serial_ over folded runes, for an uncased engine; a start inside
 *  the folds of one codepoint snaps outward to the codepoint's own start. */
STRINGZILLA_INLINE void sz_regex_mark_positions_folded_(
    sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t length, sz_size_t low, sz_size_t edge,
    sz_size_t base, sz_u64_t const *entry, sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_u32_t const *looks, sz_u64_t *sets, sz_u64_t *starts, sz_u64_t *low_edge) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_size_t const words = engine->positions_words, stop = low ? low - 1 : 0;
    sz_size_t position = edge, right_kind = 0;
    sz_u64_t *state = sets, *next = sets + words;
    sz_regex_image_t image;
    if (edge != length) {
        sz_regex_image_after_(haystack, length, edge, &image);
        right_kind = sz_regex_rune_kind_(image.runes[0]);
    }
    memcpy(state, entry, words * sizeof(sz_u64_t));
    if (position == low) memcpy(low_edge, state, words * sizeof(sz_u64_t));
    while (position != 0 && position >= low) {
        sz_size_t begin;
        sz_u64_t others = state[0] ^ 1;
        for (sz_size_t word = 1; word != words; ++word) others |= state[word];
        if (!others && !sz_byteset_contains_u8(&engine->reverse_live, bytes[position - 1])) {
            sz_cptr_t const live = rfind_live(haystack + stop, position - stop, &engine->reverse_live);
            if (!live) {
                if (position > low) memcpy(low_edge, state, words * sizeof(sz_u64_t));
                position = 0, right_kind = sz_regex_byte_kind_(bytes[0]);
                break;
            }
            position = (sz_size_t)(live - haystack) + 1;
            right_kind = position == length ? 0 : sz_regex_byte_kind_(bytes[position]);
            if (position == low) memcpy(low_edge, state, words * sizeof(sz_u64_t));
        }
        begin = sz_regex_image_before_(haystack, position, &image);
        for (sz_size_t rune = image.runes_count; rune--;) {
            sz_size_t const start = rune + 1 == image.runes_count ? position : begin;
            sz_size_t const left_kind = sz_regex_rune_kind_(image.runes[rune]);
            sz_u64_t *const swapped = state;
            sz_regex_positions_step_(engine, words, state, next, sz_regex_rune_class_(engine, image.runes[rune]),
                                     left_kind * 4 + right_kind,
                                     rune + 1 == image.runes_count ? sz_regex_looks_at_(looks, position) : 0);
            state = next, next = swapped;
            if (start >= low) starts[(start - base) / 64] |= (state[0] >> 1 & 1) << ((start - base) % 64);
            right_kind = left_kind;
        }
        position = begin;
        if (position == low) memcpy(low_edge, state, words * sizeof(sz_u64_t));
    }
    if (position == 0 && low == 0) {
        sz_regex_positions_step_(engine, words, state, next, 1, right_kind, sz_regex_looks_at_(looks, 0));
        starts[0] |= next[0] >> 1 & 1;
    }
}

/** The start pass's DFA state at the first position of @p window, primed unless known. */
STRINGZILLA_INLINE sz_u32_t sz_regex_entry_serial_(sz_regex_engine_t const *engine, sz_u8_t const *bytes,
                                                   sz_size_t length, sz_size_t window, sz_regex_walk_t *walk) {
    if (walk->bounds[window] == length) return engine->reverse_starts[0];
    if (walk->edges[window]) return (sz_u32_t)walk->edges[window];
    return engine->case_sensitivity == sz_cased_k
               ? sz_regex_prime_serial_(engine, bytes, length, window, walk)
               : sz_regex_prime_folded_(engine, (sz_cptr_t)bytes, length, window, walk);
}

/** The start pass's position set at the first position of @p window, primed unless known. */
STRINGZILLA_INLINE sz_u64_t const *sz_regex_entry_positions_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                                    sz_size_t length, sz_size_t window,
                                                                    sz_regex_walk_t *walk) {
    sz_size_t const words = engine->positions_words;
    sz_u64_t *const known = walk->edges + window * words;
    if (walk->bounds[window] == length) {
        memset(known, 0, words * sizeof(sz_u64_t));
        known[0] = 1;
        return known;
    }
    if (known[0] & 1) return known;
    return engine->case_sensitivity == sz_cased_k
               ? sz_regex_prime_positions_serial_(engine, haystack, length, window, walk)
               : sz_regex_prime_positions_folded_(engine, haystack, length, window, walk);
}

/**
 *  @brief Marks the starts of @c sz_regex_chains_serial_k windows at once, one DFA chain per window
 *      walking right to left from its own entry state, so the chains' loads overlap.
 *
 *  Each chain gathers its start bits in a register, 64 steps at a time, and stores them at its
 *  window's offset; windows differ in length by a few bytes, which the chains then finish one by
 *  one, each leaving its last state in @p finals. Steps read @p looks for their blocks, which
 *  callers pass as a literal null for a vocabulary without lookarounds.
 */
STRINGZILLA_INLINE void sz_regex_mark_chains_serial_(
    sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t const *bounds, sz_u32_t const *entries,
    sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *), sz_u32_t const *looks, sz_size_t window_words,
    sz_u64_t *starts, sz_u32_t *finals) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_u32_t const *const rows = engine->reverse_rows;
    sz_u8_t const *const byte_to_class = engine->byte_to_class;
    sz_u32_t const accepts = engine->reverse_accepts_begin;
    sz_u32_t states[sz_regex_chains_serial_k];
    sz_size_t steps = STRINGZILLA_SIZE_MAX, chain;
    for (chain = 0; chain != sz_regex_chains_serial_k; ++chain)
        states[chain] = entries[chain], steps = sz_min_of_two(steps, bounds[chain + 1] - bounds[chain]);
    steps -= steps % 64;
    for (sz_size_t block = 0; block != steps; block += 64) {
        sz_u64_t gathered[sz_regex_chains_serial_k];
        for (chain = 0; chain != sz_regex_chains_serial_k; ++chain) gathered[chain] = 0;
        for (sz_size_t step = 0; step != 64; ++step)
            for (chain = 0; chain != sz_regex_chains_serial_k; ++chain) {
                sz_size_t const position = bounds[chain + 1] - block - step;
                states[chain] = rows[states[chain] + sz_regex_block_at_(engine, looks, position) +
                                     byte_to_class[bytes[position - 1]]];
                gathered[chain] = gathered[chain] << 1 | (sz_u64_t)(states[chain] >= accepts);
            }
        for (chain = 0; chain != sz_regex_chains_serial_k; ++chain) {
            // The last step's bit is the lowest position, 63 below the block's first.
            sz_size_t const lowest = bounds[chain + 1] - bounds[chain] - block - 63;
            sz_size_t const shift = lowest % 64, carry = 64 - shift;
            sz_u64_t *const bits = starts + chain * window_words;
            bits[lowest / 64] |= gathered[chain] << shift;
            if (shift) bits[lowest / 64 + 1] |= gathered[chain] >> carry;
        }
    }
    for (chain = 0; chain != sz_regex_chains_serial_k; ++chain)
        sz_regex_mark_serial_(engine, haystack, bounds[chain], bounds[chain + 1] - steps, bounds[chain], states[chain],
                              rfind_live, looks, starts + chain * window_words, finals + chain);
}

/**
 *  @brief Marks the starts of @c sz_regex_position_chains_serial_k windows at once, one position
 *      set per window stepping right to left from its own primed set, so the chains' steps overlap.
 *
 *  Units span one to four bytes, so the chains advance together for a quarter of the shortest
 *  remainder at a time, then finish one by one.
 */
STRINGZILLA_INLINE void sz_regex_mark_chains_positions_serial_(
    sz_regex_engine_t const *engine, sz_size_t words, sz_cptr_t haystack, sz_size_t length, sz_size_t const *bounds,
    sz_u64_t const *const *entries, sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_u32_t const *looks, sz_size_t window_words, sz_u64_t *sets, sz_u64_t *starts) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_u64_t *states[sz_regex_position_chains_serial_k], *nexts[sz_regex_position_chains_serial_k],
        singles[sz_regex_position_chains_serial_k];
    sz_size_t positions[sz_regex_position_chains_serial_k], right_kinds[sz_regex_position_chains_serial_k], chain;
    for (chain = 0; chain != sz_regex_position_chains_serial_k; ++chain) {
        states[chain] = sets + 2 * chain * words, nexts[chain] = states[chain] + words;
        memcpy(states[chain], entries[chain], words * sizeof(sz_u64_t));
        positions[chain] = bounds[chain + 1];
        right_kinds[chain] = positions[chain] == length ? 0 : sz_regex_byte_kind_(bytes[positions[chain]]);
    }
    for (chain = 0; chain != sz_regex_position_chains_serial_k; ++chain) singles[chain] = states[chain][0];
    for (;;) {
        sz_size_t rounds = STRINGZILLA_SIZE_MAX;
        for (chain = 0; chain != sz_regex_position_chains_serial_k; ++chain)
            rounds = sz_min_of_two(rounds, (positions[chain] - bounds[chain]) / 4);
        if (!rounds) break;
        while (rounds--)
            for (chain = 0; chain != sz_regex_position_chains_serial_k; ++chain) {
                sz_size_t const position = positions[chain], offset = position - bounds[chain];
                sz_size_t unit_class, left_kind = sz_regex_byte_kind_(bytes[position - 1]), matched;
                sz_size_t const kind = left_kind * 4 + right_kinds[chain];
                sz_u32_t const held = sz_regex_looks_at_(looks, position);
                positions[chain] = sz_regex_unit_before_(engine, haystack, position, &unit_class);
                if (words == 1)
                    singles[chain] = sz_regex_positions_step_one_(engine, singles[chain], unit_class, kind, held),
                    matched = singles[chain] >> 1 & 1;
                else {
                    sz_u64_t *const swapped = states[chain];
                    sz_regex_positions_step_(engine, words, states[chain], nexts[chain], unit_class, kind, held);
                    states[chain] = nexts[chain], nexts[chain] = swapped, matched = states[chain][0] >> 1 & 1;
                }
                starts[chain * window_words + offset / 64] |= (sz_u64_t)matched << (offset % 64);
                right_kinds[chain] = left_kind;
            }
    }
    if (words == 1)
        for (chain = 0; chain != sz_regex_position_chains_serial_k; ++chain) states[chain][0] = singles[chain];
    // A chain standing on its window's first position still steps once more, to mark a start there.
    for (chain = 0; chain != sz_regex_position_chains_serial_k; ++chain)
        sz_regex_mark_positions_serial_(engine, words, haystack, length, bounds[chain], positions[chain], bounds[chain],
                                        states[chain], rfind_live, looks, sets + 2 * sz_regex_chains_serial_k * words,
                                        starts + chain * window_words);
}

/**
 *  @brief Whether the next batch should skip from the idle states with the tier's byte-set search
 *      rather than walk its windows in lockstep.
 *
 *  Before any batch skipped, the samples of @ref sz_regex_live_sparse_ decide. A rare live byte
 *  can still hold a walk off its idle states for long, like the opening quote of `"[^"]*"`, so
 *  once batches skipped, skipping goes on only while it costs less than a lockstep pass would
 *  have cost over the bytes it spanned.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_skipping_pays_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                     sz_regex_walk_t const *walk, sz_size_t begin, sz_size_t end) {
    if (walk->skip_spanned) return (sz_bool_t)(walk->skip_cost < walk->skip_spanned);
    return sz_regex_live_sparse_(engine, haystack, begin, end, (sz_bool_t)(walk->find_escape != STRINGZILLA_NULL));
}

/** Bytes right of a window's edge the start pass walks from an idle state to guess the state it
 *  enters the window with. */
enum { sz_regex_mark_warmup_k = 16 };

/**
 *  @brief Guesses the start-finding state at the right edge of each of the first @p count windows
 *      from @p bounds, walking in lockstep from an idle state over the bytes just right of it.
 *
 *  A guess is the true state once the walk from the haystack's end would have reached an idle state
 *  within those bytes; a walk starting at the haystack's end starts from its true state.
 */
STRINGZILLA_INLINE void sz_regex_mark_guesses_serial_(sz_regex_engine_t const *engine, sz_u8_t const *bytes,
                                                      sz_size_t length, sz_size_t const *bounds, sz_size_t count,
                                                      sz_u32_t const *looks, sz_u32_t *guesses) {
    sz_u32_t const *const rows = engine->reverse_rows;
    sz_size_t positions[sz_regex_chains_serial_k];
    for (sz_size_t chain = 0; chain != count; ++chain) {
        sz_size_t const far_edge = sz_min_of_two(bounds[chain + 1] + sz_regex_mark_warmup_k, length);
        positions[chain] = far_edge;
        guesses[chain] = engine->reverse_starts[far_edge == length ? 0 : sz_regex_byte_kind_(bytes[far_edge])];
    }
    for (sz_size_t step = 0; step != sz_regex_mark_warmup_k; ++step)
        for (sz_size_t chain = 0; chain != count; ++chain) {
            sz_size_t const position = positions[chain];
            if (position == bounds[chain + 1]) continue;
            guesses[chain] = rows[guesses[chain] + sz_regex_block_at_(engine, looks, position) +
                                  engine->byte_to_class[bytes[position - 1]]];
            positions[chain] = position - 1;
        }
}

/**
 *  @brief Checks the guessed entry of every window of a batch but the last against the state the
 *      window right of it ended on, right to left, correcting the start bits where they differ.
 *
 *  A window's first step marks its right edge, which only the window right of it is searched for,
 *  so the guess is compared one step in. Where it differs, both states walk on until they meet, and
 *  the bits of the positions where only one of them accepts flip; a pair that never meets leaves
 *  the true state as the window's last, and the first window's edge bit follows it.
 */
STRINGZILLA_INLINE void sz_regex_mark_settle_serial_(sz_regex_engine_t const *engine, sz_u8_t const *bytes,
                                                     sz_size_t const *bounds, sz_size_t count, sz_u32_t const *guesses,
                                                     sz_u32_t const *looks, sz_size_t window_words, sz_u64_t *starts,
                                                     sz_u32_t *finals) {
    sz_u32_t const *const rows = engine->reverse_rows;
    sz_u8_t const *const byte_to_class = engine->byte_to_class;
    sz_u32_t const accepts = engine->reverse_accepts_begin;
    for (sz_size_t chain = count - 1; chain-- != 0;) {
        sz_size_t const low = bounds[chain], edge = bounds[chain + 1], stop = low ? low - 1 : 0;
        sz_u64_t *const bits = starts + chain * window_words;
        sz_u32_t truth = finals[chain + 1];
        sz_u32_t guess =
            rows[guesses[chain] + sz_regex_block_at_(engine, looks, edge) + byte_to_class[bytes[edge - 1]]];
        sz_size_t position = edge - 1;
        if (truth == guess) continue;
        while (truth != guess && position > stop) {
            sz_u32_t const column = sz_regex_block_at_(engine, looks, position) + byte_to_class[bytes[position - 1]];
            truth = rows[truth + column], guess = rows[guess + column];
            if ((truth >= accepts) != (guess >= accepts))
                bits[(position - low) / 64] ^= 1ull << ((position - low) % 64);
            --position;
        }
        if (truth == guess) continue;
        finals[chain] = truth;
        if (low == 0)
            bits[0] = (bits[0] & ~1ull) |
                      (sz_u64_t)(rows[truth + sz_regex_block_at_(engine, looks, 0) + engine->classes_count] >= accepts);
    }
}

/**
 *  @brief Marks the starts of a cased DFA batch for @ref sz_regex_mark_batch_serial_, priming only
 *      its last window's entry.
 *
 *  Skipping, it walks right to left as one walk, each window entered with the state the one right
 *  of it ended on; in lockstep, it guesses the other entries and settles them once every window is
 *  marked. Outlined, so that no tier's target widens the chains into gathers.
 */
STRINGZILLA_OUTLINED_ void sz_regex_mark_states_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                        sz_size_t length, sz_regex_walk_t *walk, sz_size_t first,
                                                        sz_size_t count, sz_bool_t skipping, sz_bool_t trying) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    sz_size_t const *const bounds = walk->bounds + first;
    sz_u32_t states[sz_regex_chains_serial_k], finals[sz_regex_chains_serial_k];
    states[count - 1] = sz_regex_entry_serial_(engine, bytes, length, first + count, walk);
    for (sz_size_t chain = count; skipping && chain-- != 0;) {
        sz_bool_t const last = (sz_bool_t)(chain + 1 == count);
        sz_size_t const edge = last ? bounds[chain + 1] : bounds[chain + 1] - 1;
        sz_u32_t const entry = last ? states[chain] : finals[chain + 1];
        sz_u64_t *const starts = walk->starts + chain * walk->window_words;
        sz_size_t const cost = walk->looks
                                   ? sz_regex_mark_serial_(engine, haystack, bounds[chain], edge, bounds[chain], entry,
                                                           walk->rfind_live, walk->looks, starts, finals + chain)
                                   : sz_regex_mark_serial_(engine, haystack, bounds[chain], edge, bounds[chain], entry,
                                                           walk->rfind_live, STRINGZILLA_NULL, starts, finals + chain);
        if (count == 1) continue;
        walk->skip_cost += cost, walk->skip_spanned += bounds[chain + 1] - bounds[chain];
        if (trying && last) skipping = (sz_bool_t)(walk->skip_cost < walk->skip_spanned);
    }
    if (skipping) return;
    sz_regex_mark_guesses_serial_(engine, bytes, length, bounds, count - 1, walk->looks, states);
    if (walk->looks)
        sz_regex_mark_chains_serial_(engine, haystack, bounds, states, walk->rfind_live, walk->looks,
                                     walk->window_words, walk->starts, finals);
    else
        sz_regex_mark_chains_serial_(engine, haystack, bounds, states, walk->rfind_live, STRINGZILLA_NULL,
                                     walk->window_words, walk->starts, finals);
    sz_regex_mark_settle_serial_(engine, bytes, bounds, count, states, walk->looks, walk->window_words, walk->starts,
                                 finals);
}

/**
 *  @brief Marks the starts of the windows from @p first: a batch of @c sz_regex_chains_serial_k in
 *      lockstep or skipping one window after another, or one window alone in a haystack too short
 *      for a batch.
 *
 *  The first batch that samples well enough to skip only tries one window, then decides again from
 *  what that cost; a window marked twice keeps its bits, as both passes OR the same ones in.
 *  Position sets of one and two words step with their word loops unrolled, and DFA walks without
 *  lookarounds step with no table loads.
 */
STRINGZILLA_INLINE void sz_regex_mark_batch_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                    sz_size_t length, sz_regex_walk_t *walk, sz_size_t first) {
    sz_size_t const words = engine->positions_words, *const bounds = walk->bounds + first;
    sz_size_t const count = walk->windows - first >= sz_regex_chains_serial_k ? sz_regex_chains_serial_k : 1;
    sz_u64_t *const single = walk->sets + 2 * sz_regex_chains_serial_k * words;
    sz_u64_t const *sets[sz_regex_chains_serial_k];
    sz_bool_t skipping = (sz_bool_t)(count == 1 ||
                                     sz_regex_skipping_pays_(engine, haystack, walk, bounds[0], bounds[count]));
    sz_bool_t const trying = (sz_bool_t)(skipping && count != 1 && !walk->skip_spanned);
    memset(walk->starts, 0, count * walk->window_words * sizeof(sz_u64_t));
    walk->batch = first, walk->batch_count = count;
    if (engine->case_sensitivity != sz_cased_k) {
        for (sz_size_t chain = count; chain-- != 0;) {
            sz_u64_t *const starts = walk->starts + chain * walk->window_words;
            if (words)
                sz_regex_mark_positions_folded_(
                    engine, haystack, length, bounds[chain], bounds[chain + 1], bounds[chain],
                    sz_regex_entry_positions_serial_(engine, haystack, length, first + chain + 1, walk),
                    walk->rfind_live, walk->looks, single, starts, walk->edges + (first + chain) * words);
            else
                sz_regex_mark_folded_(
                    engine, haystack, bounds[chain], bounds[chain + 1], bounds[chain],
                    sz_regex_entry_serial_(engine, (sz_u8_t const *)haystack, length, first + chain + 1, walk),
                    walk->rfind_live, walk->looks, starts, walk->edges + first + chain);
        }
        return;
    }
    if (!words) {
        sz_regex_mark_states_serial_(engine, haystack, length, walk, first, count, skipping, trying);
        return;
    }
    for (sz_size_t chain = 0; chain != count; ++chain)
        sets[chain] = sz_regex_entry_positions_serial_(engine, haystack, length, first + chain + 1, walk);
    for (sz_size_t chain = 0; skipping && chain != count; ++chain) {
        sz_u64_t *const starts = walk->starts + chain * walk->window_words;
        sz_size_t const cost = words == 1
                                   ? sz_regex_mark_positions_serial_(engine, 1, haystack, length, bounds[chain],
                                                                     bounds[chain + 1], bounds[chain], sets[chain],
                                                                     walk->rfind_live, walk->looks, single, starts)
                                   : sz_regex_mark_positions_serial_(engine, words, haystack, length, bounds[chain],
                                                                     bounds[chain + 1], bounds[chain], sets[chain],
                                                                     walk->rfind_live, walk->looks, single, starts);
        if (count == 1) continue;
        walk->skip_cost += cost, walk->skip_spanned += bounds[chain + 1] - bounds[chain];
        if (trying && chain == 0) skipping = (sz_bool_t)(walk->skip_cost < walk->skip_spanned);
    }
    if (skipping) return;
    for (sz_size_t chain = 0; chain != count; chain += sz_regex_position_chains_serial_k) {
        sz_u64_t *const starts = walk->starts + chain * walk->window_words;
        if (words == 1)
            sz_regex_mark_chains_positions_serial_(engine, 1, haystack, length, bounds + chain, sets + chain,
                                                   walk->rfind_live, walk->looks, walk->window_words, walk->sets,
                                                   starts);
        else if (words == 2)
            sz_regex_mark_chains_positions_serial_(engine, 2, haystack, length, bounds + chain, sets + chain,
                                                   walk->rfind_live, walk->looks, walk->window_words, walk->sets,
                                                   starts);
        else
            sz_regex_mark_chains_positions_serial_(engine, words, haystack, length, bounds + chain, sets + chain,
                                                   walk->rfind_live, walk->looks, walk->window_words, walk->sets,
                                                   starts);
    }
}

/** The first marked position in `[from, last]`, or @c STRINGZILLA_SIZE_MAX. */
STRINGZILLA_INLINE sz_size_t sz_regex_next_start_(sz_u64_t const *starts, sz_size_t base, sz_size_t from,
                                                  sz_size_t last) {
    sz_size_t const end = last - base;
    sz_size_t word = (from - base) / 64;
    sz_u64_t bits = starts[word] & ~sz_u64_mask_until_serial_((from - base) % 64);
    for (;;) {
        if (bits) {
            sz_size_t const found = word * 64 + (sz_size_t)sz_u64_ctz(bits);
            return found <= end ? base + found : STRINGZILLA_SIZE_MAX;
        }
        if ((word + 1) * 64 > end) return STRINGZILLA_SIZE_MAX;
        bits = starts[++word];
    }
}

/** At a checkpoint, the earlier walk that held @p state there, or none once this walk left its
 *  own state there. */
STRINGZILLA_INLINE sz_regex_checkpoint_t const *sz_regex_checkpoint_meet_(sz_regex_walk_t *walk, sz_size_t position,
                                                                          sz_u32_t state, sz_size_t *first_checkpoint) {
    sz_regex_checkpoint_t *const checkpoint = walk->checkpoints +
                                              position / sz_regex_checkpoint_spacing_k % sz_regex_checkpoints_k;
    if (checkpoint->serial == walk->serial && checkpoint->position == position && checkpoint->state == state)
        return checkpoint;
    checkpoint->serial = walk->serial, checkpoint->position = position, checkpoint->state = state;
    if (*first_checkpoint == STRINGZILLA_SIZE_MAX) *first_checkpoint = position;
    return STRINGZILLA_NULL;
}

/** Tells every checkpoint a walk left in `[first, bound)` the match the walk ended with,
 *  valid only past it. */
STRINGZILLA_INLINE void sz_regex_checkpoints_settle_(sz_regex_walk_t *walk, sz_size_t first, sz_size_t bound,
                                                     sz_size_t end, sz_u32_t pattern) {
    for (sz_size_t crossed = first; crossed < bound; crossed += sz_regex_checkpoint_spacing_k) {
        sz_regex_checkpoint_t *const checkpoint = walk->checkpoints +
                                                  crossed / sz_regex_checkpoint_spacing_k % sz_regex_checkpoints_k;
        if (checkpoint->serial != walk->serial || checkpoint->position != crossed) continue;
        checkpoint->end = end != STRINGZILLA_SIZE_MAX && end >= crossed ? end : STRINGZILLA_SIZE_MAX;
        checkpoint->pattern = pattern;
    }
}

/**
 *  @brief Walks the end-finding automaton from @p start until it dies, keeping its
 *      last accepting state.
 *
 *  At every checkpoint it crosses, a walk leaves its state; a later walk holding the same state
 *  there walks the same bytes on, so it stops and takes that walk's match if it ends past the
 *  checkpoint, or its own before it. That bounds the bytes walked when many starts share a long
 *  look-ahead, like every @c a of a line before `a[^\n]*z|a` gives up on the @c z. At a checkpoint
 *  the walk also searches ahead for the next byte that moves its state, where the state stays put
 *  on the byte at hand, as inside `"[^"]*"`. Steps read @p looks for their blocks, which callers
 *  pass as a literal null for a vocabulary without lookarounds.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_walk_end_serial_(sz_regex_engine_t const *engine, sz_u8_t const *bytes,
                                                       sz_size_t length, sz_size_t start, sz_u32_t const *looks,
                                                       sz_regex_walk_t *walk, sz_size_t *end, sz_u32_t *pattern) {
    sz_u32_t const *const rows = engine->forward_rows;
    sz_u8_t const *const byte_to_class = engine->byte_to_class;
    sz_u32_t const accepts = engine->forward_accepts_begin;
    sz_u32_t const stride = (sz_u32_t)sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_regex_checkpoint_t const *met = STRINGZILLA_NULL;
    sz_u32_t state = engine->forward_starts[start ? sz_regex_byte_kind_(bytes[start - 1]) : 0], last = 0;
    sz_size_t position = start, last_end = STRINGZILLA_SIZE_MAX, first_checkpoint = STRINGZILLA_SIZE_MAX;
    sz_u32_t last_pattern = 0;
    while (position != length && state != 0) {
        if (position % sz_regex_checkpoint_spacing_k == 0) {
            if ((met = sz_regex_checkpoint_meet_(walk, position, state, &first_checkpoint))) break;
            if (walk->find_escape &&
                !sz_byteset_contains_u8(engine->forward_escapes + state / stride, bytes[position])) {
                sz_cptr_t const escape = walk->find_escape((sz_cptr_t)bytes + position, length - position,
                                                           engine->forward_escapes + state / stride);
                sz_size_t const reached = escape ? (sz_size_t)(escape - (sz_cptr_t)bytes) : length;
                if (state >= accepts) last = state, last_end = reached - 1;
                position = reached;
                continue;
            }
        }
        state = rows[state + sz_regex_block_at_(engine, looks, position) + byte_to_class[bytes[position]]];
        if (state >= accepts) last = state, last_end = position;
        ++position;
    }
    if (!met && state != 0 &&
        (state = rows[state + sz_regex_block_at_(engine, looks, length) + engine->classes_count]) >= accepts)
        last = state, last_end = length;
    if (last) last_pattern = engine->forward_patterns[(last - accepts) / stride];
    if (met && met->end != STRINGZILLA_SIZE_MAX) last_end = met->end, last_pattern = met->pattern;
    sz_regex_checkpoints_settle_(walk, first_checkpoint, position, last_end, last_pattern);
    if (last_end == STRINGZILLA_SIZE_MAX) return sz_false_k;
    *end = last_end, *pattern = last_pattern;
    return sz_true_k;
}

/**
 *  @brief @ref sz_regex_walk_end_serial_ over folded bytes, from @p offset bytes into the folds of
 *      the unit at @p start, for an uncased engine.
 *  @return Whether a match ends there, an empty one counting only at the unit's own start.
 *
 *  A match ending inside the folds of one codepoint snaps outward to the codepoint's end.
 *  Checkpoints are left at the unit boundaries that fall on their spacing, and only steps from
 *  those boundaries read lookaround blocks.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_walk_end_folded_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                       sz_size_t length, sz_size_t start, sz_size_t offset,
                                                       sz_regex_walk_t *walk, sz_size_t *end, sz_u32_t *pattern) {
    sz_u32_t const *const rows = engine->forward_rows;
    sz_u8_t const *const byte_to_class = engine->byte_to_class;
    sz_u32_t const accepts = engine->forward_accepts_begin;
    sz_u32_t const stride = (sz_u32_t)sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_regex_checkpoint_t const *met = STRINGZILLA_NULL;
    sz_regex_image_t image;
    sz_size_t position = start, unit_end = start, byte = offset, kind = 0, consumed = 0, last_consumed = 0;
    sz_size_t last_end = STRINGZILLA_SIZE_MAX, first_checkpoint = STRINGZILLA_SIZE_MAX;
    sz_u32_t state, last = 0, last_pattern = 0;
    if (start != length) unit_end = sz_regex_image_after_(haystack, length, start, &image);
    if (offset) kind = sz_regex_byte_kind_(image.bytes[offset - 1]);
    else if (start) {
        sz_regex_image_t previous;
        sz_regex_image_before_(haystack, start, &previous);
        kind = sz_regex_byte_kind_(previous.bytes[previous.length - 1]);
    }
    state = engine->forward_starts[kind];
    while (position != length && state != 0) {
        if (byte == 0 && position % sz_regex_checkpoint_spacing_k == 0 &&
            (met = sz_regex_checkpoint_meet_(walk, position, state, &first_checkpoint)))
            break;
        state = rows[state + (byte ? 0 : sz_regex_block_at_(engine, walk->looks, position)) +
                     byte_to_class[image.bytes[byte]]];
        if (state >= accepts) last = state, last_end = byte ? unit_end : position, last_consumed = consumed;
        ++consumed;
        if (++byte != image.length) continue;
        position = unit_end, byte = 0;
        if (position != length) unit_end = sz_regex_image_after_(haystack, length, position, &image);
    }
    if (!met && state != 0 &&
        (state = rows[state + sz_regex_block_at_(engine, walk->looks, length) + engine->classes_count]) >= accepts)
        last = state, last_end = length, last_consumed = consumed;
    if (last) last_pattern = engine->forward_patterns[(last - accepts) / stride];
    if (met && met->end != STRINGZILLA_SIZE_MAX) last_end = met->end, last_pattern = met->pattern, last_consumed = 1;
    // A walk dying inside a unit stops at the unit's start, where it may have left
    // a checkpoint too.
    sz_regex_checkpoints_settle_(walk, first_checkpoint, position + 1, last_end, last_pattern);
    if (last_end == STRINGZILLA_SIZE_MAX || (offset && !last_consumed)) return sz_false_k;
    *end = last_end, *pattern = last_pattern;
    return sz_true_k;
}

/** The unit at @p position as walks step it, before @p length: the byte itself when cased,
 *  else its folds. */
STRINGZILLA_INLINE sz_size_t sz_regex_image_at_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t length,
                                                sz_size_t position, sz_regex_image_t *image) {
    if (engine->case_sensitivity != sz_cased_k) return sz_regex_image_after_(haystack, length, position, image);
    image->bytes[0] = (sz_u8_t)haystack[position], image->length = 1;
    image->runes_count = 1, image->rune_starts[0] = 0;
    return position + 1;
}

/** The unit ending at @p end, which is past zero, as walks step it: the byte itself when cased,
 *  else its folds. */
STRINGZILLA_INLINE sz_size_t sz_regex_image_ending_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t end,
                                                    sz_regex_image_t *image) {
    if (engine->case_sensitivity != sz_cased_k) return sz_regex_image_before_(haystack, end, image);
    image->bytes[0] = (sz_u8_t)haystack[end - 1], image->length = 1;
    image->runes_count = 1, image->rune_starts[0] = 0;
    return end - 1;
}

/** The assertions that hold between two units, bytes or @c sz_regex_end_of_input_k at
 *  a haystack's edge. */
STRINGZILLA_INLINE sz_u32_t sz_regex_looks_between_(sz_u32_t before, sz_u32_t after) {
    sz_u32_t have = sz_regex_is_word_(before) == sz_regex_is_word_(after) ? (sz_u32_t)sz_regex_look_word_inside_k
                                                                          : (sz_u32_t)sz_regex_look_word_boundary_k;
    if (before == sz_regex_end_of_input_k) have |= sz_regex_look_text_start_k | sz_regex_look_line_start_k;
    else if (before == '\n') have |= sz_regex_look_line_start_k;
    if (after == sz_regex_end_of_input_k) have |= sz_regex_look_text_end_k | sz_regex_look_line_end_k;
    else if (after == '\n') have |= sz_regex_look_line_end_k;
    return have;
}

/**
 *  @brief Adds to the position's threads every state @p state reaches without consuming, depth
 *      first in priority order, crossing the assertions in @p have; a capture state writes @p open
 *      or @p close into the path's slots.
 *
 *  A state the position holds already ends its path, which keeps the first path's priority and
 *  slots, as `regex-automata`'s PikeVM does.
 */
STRINGZILLA_INLINE void sz_regex_pike_close_(sz_regex_engine_t const *engine, sz_regex_pike_t *pike, sz_u32_t state,
                                             sz_size_t const *slots, sz_u32_t have, sz_size_t open, sz_size_t close) {
    sz_regex_nfa_state_t const *const states = engine->nfa_states;
    sz_regex_nfa_link_t const *const links = engine->nfa_links;
    sz_size_t const states_count = engine->nfa_states_count, slots_count = pike->slots_count;
    sz_size_t *const frames = pike->frames, *const current = pike->current;
    sz_size_t top = 0;
    memcpy(current, slots, slots_count * sizeof(sz_size_t));
    frames[top++] = state, frames[top++] = 0;
    while (top) {
        sz_size_t const value = frames[--top], tag = frames[--top];
        if (tag >= states_count) {
            current[tag - states_count] = value;
            continue;
        }
        for (state = (sz_u32_t)tag;;) {
            sz_regex_nfa_state_t const *const node = states + state;
            sz_u32_t const held = pike->sparse[state];
            sz_u32_t link;
            sz_size_t base;
            if (held < pike->count && pike->dense[held] == state) break;
            pike->sparse[state] = (sz_u32_t)pike->count, pike->dense[pike->count] = state;
            if (node->kind == sz_regex_nfa_ranges_k || node->kind == sz_regex_nfa_match_k)
                memcpy(pike->dense_slots + pike->count * slots_count, current, slots_count * sizeof(sz_size_t));
            ++pike->count;
            if (node->kind == sz_regex_nfa_ranges_k || node->kind == sz_regex_nfa_match_k) break;
            if (node->kind == sz_regex_nfa_look_k && !(have & node->look)) break;
            if (node->kind == sz_regex_nfa_empty_k && node->slot < slots_count) {
                frames[top++] = states_count + node->slot, frames[top++] = current[node->slot];
                current[node->slot] = node->slot & 1 ? close : open;
            }
            if (node->kind == sz_regex_nfa_look_k || node->kind == sz_regex_nfa_empty_k) {
                state = node->target;
                continue;
            }
            link = node->first_link;
            if (link == sz_regex_none_k) break;
            base = top;
            for (sz_u32_t other = links[link].next; other != sz_regex_none_k; other = links[other].next)
                frames[top++] = links[other].target, frames[top++] = 0;
            // Alternatives pushed in priority order pop in reverse, so the pairs are flipped.
            for (sz_size_t low = base, high = top; low + 2 < high; low += 2, high -= 2) {
                sz_size_t const swapped = frames[low];
                frames[low] = frames[high - 2], frames[high - 2] = swapped;
            }
            state = links[link].target;
        }
    }
}

/**
 *  @brief Runs the PikeVM at one position: closes the threads the last step entered, keeps the
 *      first match among them and cuts every thread after it, then steps the rest over @p after.
 *  @param[in] looks The lookarounds holding at the position, as look bits.
 *  @param[in] open,close What a capture state writes here; they differ only inside the folds
 *      of one codepoint.
 *  @return Whether any thread entered the next position.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_pike_step_(sz_regex_engine_t const *engine, sz_regex_pike_t *pike,
                                                 sz_u32_t before, sz_u32_t after, sz_u32_t looks, sz_size_t open,
                                                 sz_size_t close) {
    sz_regex_nfa_state_t const *const states = engine->nfa_states;
    sz_regex_nfa_link_t const *const links = engine->nfa_links;
    sz_size_t const slots_count = pike->slots_count;
    sz_u32_t const have = sz_regex_looks_between_(before, after) | looks;
    pike->count = 0, pike->found = sz_false_k;
    for (sz_size_t index = 0; index != pike->entered_count; ++index)
        sz_regex_pike_close_(engine, pike, pike->entered[index], pike->entered_slots + index * slots_count, have, open,
                             close);
    pike->entered_count = 0;
    for (sz_size_t index = 0; index != pike->count; ++index) {
        sz_regex_nfa_state_t const *const node = states + pike->dense[index];
        if (node->kind == sz_regex_nfa_match_k) {
            pike->end = close, pike->pattern = node->target, pike->found = sz_true_k;
            memcpy(pike->matched, pike->dense_slots + index * slots_count, slots_count * sizeof(sz_size_t));
            break;
        }
        if (node->kind != sz_regex_nfa_ranges_k || after == sz_regex_end_of_input_k) continue;
        for (sz_u32_t link = node->first_link; link != sz_regex_none_k; link = links[link].next) {
            if (after < links[link].low || after > links[link].high) continue;
            pike->entered[pike->entered_count] = links[link].target;
            memcpy(pike->entered_slots + pike->entered_count * slots_count, pike->dense_slots + index * slots_count,
                   slots_count * sizeof(sz_size_t));
            ++pike->entered_count;
        }
    }
    return (sz_bool_t)(pike->entered_count != 0);
}

/**
 *  @brief Runs a PikeVM from the NFA state @p entry at @p offset bytes into the unit at @p start,
 *      until its threads die or it passes the unit boundary @p bound, keeping the last match, which
 *      leftmost-first priority leaves.
 *  @return Whether a match ends there, an empty one counting only at the unit's own start.
 *
 *  End walks run it over the whole remainder where the engine has no forward rows; groups run it
 *  over a match's own span. Units are bytes for a cased engine and the folds of codepoints for an
 *  uncased one, where a slot or an end inside the folds of one codepoint snaps outward to it. A
 *  step on a haystack position reads @p looks there.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_walk_pike_(sz_regex_engine_t const *engine, sz_regex_pike_t *pike,
                                                 sz_cptr_t haystack, sz_size_t length, sz_size_t start,
                                                 sz_size_t offset, sz_u32_t entry, sz_size_t bound,
                                                 sz_u32_t const *looks, sz_size_t *end, sz_u32_t *pattern) {
    sz_regex_image_t image;
    sz_size_t position = start, unit_end = start, byte = offset, last_step = 0;
    sz_bool_t matched = sz_false_k;
    sz_u32_t before = sz_regex_end_of_input_k;
    if (start != length) unit_end = sz_regex_image_at_(engine, haystack, length, start, &image);
    if (offset) before = image.bytes[offset - 1];
    else if (start && engine->case_sensitivity == sz_cased_k) before = (sz_u8_t)haystack[start - 1];
    else if (start) {
        sz_regex_image_t previous;
        sz_regex_image_before_(haystack, start, &previous);
        before = previous.bytes[previous.length - 1];
    }
    pike->entered[0] = entry, pike->entered_count = 1;
    memset(pike->entered_slots, 0xFF, pike->slots_count * sizeof(sz_size_t));
    memset(pike->matched, 0xFF, pike->slots_count * sizeof(sz_size_t));
    for (sz_size_t step = 0;; ++step) {
        sz_u32_t const after = position == length ? (sz_u32_t)sz_regex_end_of_input_k : image.bytes[byte];
        sz_u32_t const held = byte ? 0 : sz_regex_looks_at_(looks, position);
        sz_bool_t const alive = sz_regex_pike_step_(engine, pike, before, after, held, position,
                                                    byte ? unit_end : position);
        if (pike->found) matched = sz_true_k, last_step = step;
        if (!alive || (position == bound && byte == 0)) break;
        before = after;
        if (++byte != image.length) continue;
        position = unit_end, byte = 0;
        if (position != length) unit_end = sz_regex_image_at_(engine, haystack, length, position, &image);
    }
    if (!matched || (offset && last_step == 0)) return sz_false_k;
    *end = pike->end, *pattern = pike->pattern;
    return sz_true_k;
}

/**
 *  @brief Steps the threads of the lookaround bodies of @p members over one boundary, keeping every
 *      match: closes the states the last unit entered and every member's body starting here under
 *      @p have, then steps them over @p after.
 *  @return The look bits of the members whose bodies match here.
 */
STRINGZILLA_INLINE sz_u32_t sz_regex_looks_step_(sz_regex_engine_t const *engine, sz_regex_pike_t *pike,
                                                 sz_u32_t members, sz_u32_t have, sz_u32_t after) {
    sz_regex_nfa_state_t const *const states = engine->nfa_states;
    sz_regex_nfa_link_t const *const links = engine->nfa_links;
    sz_u32_t matched = 0;
    pike->count = 0;
    for (sz_size_t index = 0; index != pike->entered_count; ++index)
        sz_regex_pike_close_(engine, pike, pike->entered[index], pike->matched, have, 0, 0);
    for (sz_u32_t lookaround = 0; lookaround != engine->lookarounds_count; ++lookaround)
        if (members & (sz_u32_t)sz_regex_look_lookaround_k << lookaround)
            sz_regex_pike_close_(engine, pike, engine->lookarounds[lookaround].start, pike->matched, have, 0, 0);
    pike->entered_count = 0;
    for (sz_size_t index = 0; index != pike->count; ++index) {
        sz_regex_nfa_state_t const *const node = states + pike->dense[index];
        if (node->kind == sz_regex_nfa_match_k) matched |= (sz_u32_t)sz_regex_look_lookaround_k << node->target;
        if (node->kind != sz_regex_nfa_ranges_k || after == sz_regex_end_of_input_k) continue;
        for (sz_u32_t link = node->first_link; link != sz_regex_none_k; link = links[link].next)
            if (links[link].low <= after && after <= links[link].high)
                pike->entered[pike->entered_count++] = links[link].target;
    }
    return matched;
}

/** Steps a chain of a body pass takes from the body's start, over the bytes just before its
 *  stretch, to guess the state it enters the stretch with. */
enum { sz_regex_body_warmup_k = 16 };

/** The shortest stretch of a haystack worth a body chain of its own. */
enum { sz_regex_body_stretch_k = 128 };

/** Body chains walked in lockstep, fewer than the start pass's, as each keeps its gathered bits
 *  in a register too and eight spill. */
enum { sz_regex_body_chains_k = 4 };

/**
 *  @brief Steps one body's DFA over a cased haystack in @c sz_regex_body_chains_k chains, each over
 *      its own stretch, so their dependent loads overlap, flipping @p bit where it accepts.
 *  @return The state after the last byte.
 *
 *  Step @c i reads the byte at @c i and marks position @c i when walking @p behind, and reads the
 *  byte before position `length - i` and marks that position otherwise. Each chain but the first
 *  guesses the state it enters its stretch with by walking from the start over the bytes just
 *  before it. Once all are done, each in turn checks its guess against the state the chain before
 *  it ended on, and where the two differ walks both on until they meet, flipping each position
 *  where exactly one of them accepts.
 */
STRINGZILLA_INLINE sz_u32_t sz_regex_body_chains_(sz_regex_body_t const *body, sz_u8_t const *byte_to_class,
                                                  sz_u8_t const *bytes, sz_size_t length, sz_bool_t behind,
                                                  sz_u32_t bit, sz_u32_t *looks) {
    sz_u32_t const *const rows = body->rows;
    sz_u32_t const accepts = body->accepts_begin;
    sz_size_t const chains = length >= (sz_size_t)sz_regex_body_chains_k * sz_regex_body_stretch_k
                                 ? (sz_size_t)sz_regex_body_chains_k
                                 : 1;
    sz_size_t firsts[sz_regex_body_chains_k + 1], steps = 0, chain, step;
    sz_u32_t states[sz_regex_body_chains_k], guesses[sz_regex_body_chains_k];
    for (chain = 0; chain != chains; ++chain) firsts[chain] = chain * length / chains, states[chain] = body->start;
    firsts[chains] = length;
    if (chains == sz_regex_body_chains_k) {
        sz_size_t const warmup = sz_min_of_two((sz_size_t)sz_regex_body_warmup_k, firsts[1]);
        for (step = 0; step != warmup; ++step)
            for (chain = 1; chain != sz_regex_body_chains_k; ++chain) {
                sz_size_t const index = firsts[chain] - warmup + step;
                states[chain] = rows[states[chain] + byte_to_class[bytes[behind ? index : length - 1 - index]]];
            }
        steps = firsts[1];
        for (chain = 1; chain != sz_regex_body_chains_k; ++chain)
            steps = sz_min_of_two(steps, firsts[chain + 1] - firsts[chain]);
        steps -= steps % 64;
    }
    for (chain = 0; chain != chains; ++chain) guesses[chain] = states[chain];
    for (sz_size_t block = 0; block != steps; block += 64) {
        sz_u64_t gathered[sz_regex_body_chains_k];
        for (chain = 0; chain != sz_regex_body_chains_k; ++chain) gathered[chain] = 0;
        for (step = 0; step != 64; ++step)
            for (chain = 0; chain != sz_regex_body_chains_k; ++chain) {
                sz_size_t const index = firsts[chain] + block + step;
                states[chain] = rows[states[chain] + byte_to_class[bytes[behind ? index : length - 1 - index]]];
                gathered[chain] = gathered[chain] << 1 | (sz_u64_t)(states[chain] >= accepts);
            }
        // Accepting steps are rare, and the last step's bit is the lowest.
        for (chain = 0; chain != sz_regex_body_chains_k; ++chain)
            for (sz_u64_t bits = gathered[chain]; bits; bits &= bits - 1) {
                sz_size_t const index = firsts[chain] + block + 63 - (sz_size_t)sz_u64_ctz(bits);
                looks[behind ? index : length - index] ^= bit;
            }
    }
    for (chain = 0; chain != chains; ++chain)
        for (step = firsts[chain] + steps; step != firsts[chain + 1]; ++step) {
            states[chain] = rows[states[chain] + byte_to_class[bytes[behind ? step : length - 1 - step]]];
            if (states[chain] >= accepts) looks[behind ? step : length - step] ^= bit;
        }
    for (chain = 1; chain != chains; ++chain) {
        sz_u32_t truth = states[chain - 1], guess = guesses[chain];
        for (step = firsts[chain]; truth != guess && step != firsts[chain + 1]; ++step) {
            sz_u8_t const unit_class = byte_to_class[bytes[behind ? step : length - 1 - step]];
            truth = rows[truth + unit_class], guess = rows[guess + unit_class];
            if ((truth >= accepts) != (guess >= accepts)) looks[behind ? step : length - step] ^= bit;
        }
        if (truth != guess) states[chain] = truth;
    }
    return states[chains - 1];
}

/**
 *  @brief Flips the bits of @p members where their lookaround bodies match, stepping their DFAs one
 *      after another over the haystack, forward for lookbehinds and backward for lookaheads, as
 *      @ref sz_regex_looks_serial_ steps the PikeVM.
 *
 *  A step landing on an accepting state marks the boundary it crossed, one unit late as both
 *  automata report, so the haystack's far edge takes the edge column. Steps inside the folds of one
 *  codepoint mark the codepoint's edge they snap out to.
 */
STRINGZILLA_INLINE void sz_regex_looks_bodies_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t length,
                                               sz_u32_t members, sz_bool_t behind, sz_u32_t *looks) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack, *const byte_to_class = engine->byte_to_class;
    sz_size_t const edge = engine->classes_count, stop = behind ? length : 0;
    sz_regex_image_t image;
    for (sz_u32_t index = 0; index != engine->lookarounds_count; ++index) {
        sz_u32_t const bit = (sz_u32_t)sz_regex_look_lookaround_k << index;
        sz_regex_body_t const *const body = engine->bodies + index;
        sz_u32_t const *const rows = body->rows, accepts = body->accepts_begin;
        sz_u32_t state = body->start, pending = 0;
        sz_size_t position = behind ? 0 : length;
        if (!(members & bit)) continue;
        if (engine->case_sensitivity == sz_cased_k)
            state = behind ? sz_regex_body_chains_(body, byte_to_class, bytes, length, sz_true_k, bit, looks)
                           : sz_regex_body_chains_(body, byte_to_class, bytes, length, sz_false_k, bit, looks);
        else
            while (position != stop) {
                sz_size_t const next = behind ? sz_regex_image_after_(haystack, length, position, &image)
                                              : sz_regex_image_before_(haystack, position, &image);
                for (sz_size_t byte = 0; byte != image.length; ++byte) {
                    sz_u32_t held;
                    state = rows[state + byte_to_class[image.bytes[behind ? byte : image.length - 1 - byte]]];
                    held = state >= accepts ? bit : 0;
                    if (byte) pending |= held;
                    else looks[position] ^= held | pending, pending = 0;
                }
                position = next;
            }
        looks[stop] ^= (rows[state + edge] >= accepts ? bit : 0) | pending;
    }
}

/**
 *  @brief Fills the walk's table of the lookarounds holding at every position, level by level, as
 *      the bodies of a level test the bits of the levels below.
 *
 *  Per level, a forward pass runs every lookbehind body as one unanchored automaton keeping every
 *  match and marks where they end, and a backward pass runs the reversed lookahead bodies alike and
 *  marks where they start; a negated lookaround holds where its body marked nothing. The tier's
 *  fill first writes the one-byte lookarounds and marks every negated one a DFA walks as holding;
 *  bodies with a DFA of their own then flip their bits where they match, in chains over a cased
 *  haystack, and the rest step a PikeVM. Steps inside the folds of one codepoint see no lookaround
 *  hold, and a body ending or starting there marks the codepoint's edge it snaps out to.
 */
STRINGZILLA_OUTLINED_ void sz_regex_looks_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t length,
                                                  sz_regex_walk_t *walk) {
    sz_regex_pike_t *const pike = &walk->pike;
    sz_u32_t *const looks = walk->looks;
    sz_size_t const slots_count = pike->slots_count;
    sz_u32_t const singles = engine->looks_bytes ? engine->looks_bytes[0] : 0;
    sz_u32_t levels = 0, flipped = 0;
    sz_regex_image_t image;
    for (sz_size_t index = 0; index != engine->lookarounds_count; ++index) {
        sz_u32_t const bit = (sz_u32_t)sz_regex_look_lookaround_k << index;
        levels = sz_max_of_two(levels, engine->lookarounds[index].level);
        // Negated lookarounds a byte decides or a DFA walks hold unless their bodies flip them.
        if (engine->lookarounds[index].kind & 1 && (singles & bit || engine->bodies[index].rows)) flipped |= bit;
    }
    if (singles) walk->fill_looks(haystack, length, engine->looks_bytes, flipped, looks);
    else if (flipped)
        for (sz_size_t position = 0; position != length + 1; ++position) looks[position] = flipped;
    else memset(looks, 0, (length + 1) * sizeof(sz_u32_t));
    // The bodies hold no groups, so their threads carry no slots.
    pike->slots_count = 0;
    for (sz_u32_t pass = 0; pass != 2 * levels; ++pass) {
        sz_u32_t const behind = pass % 2 == 0, level = pass / 2 + 1;
        sz_size_t const stop = behind ? length : 0;
        sz_size_t position = behind ? 0 : length;
        sz_u32_t members = 0, negated = 0, walked = 0, before = sz_regex_end_of_input_k, pending = 0;
        for (sz_u32_t index = 0; index != engine->lookarounds_count; ++index) {
            sz_regex_lookaround_t const lookaround = engine->lookarounds[index];
            if (lookaround.level != level || (sz_u32_t)(lookaround.kind >= sz_regex_lookbehind_k) != behind) continue;
            members |= (sz_u32_t)sz_regex_look_lookaround_k << index;
            if (lookaround.kind & 1) negated |= (sz_u32_t)sz_regex_look_lookaround_k << index;
            if (engine->bodies[index].rows) walked |= (sz_u32_t)sz_regex_look_lookaround_k << index;
        }
        if (walked) sz_regex_looks_bodies_(engine, haystack, length, walked, (sz_bool_t)behind, looks);
        members &= ~walked & ~singles, negated &= ~walked & ~singles;
        if (!members) continue;
        pike->entered_count = 0;
        for (;;) {
            sz_size_t next = position;
            sz_u32_t after = sz_regex_end_of_input_k, matched;
            if (position != stop) {
                next = behind ? sz_regex_image_at_(engine, haystack, length, position, &image)
                              : sz_regex_image_ending_(engine, haystack, position, &image);
                after = image.bytes[behind ? 0 : image.length - 1];
            }
            matched = sz_regex_looks_step_(engine, pike, members,
                                           sz_regex_looks_between_(before, after) | looks[position], after);
            looks[position] |= ((matched | pending) ^ negated) & members, pending = 0;
            if (position == stop) break;
            for (sz_size_t step = 1; step < image.length; ++step) {
                before = after, after = image.bytes[behind ? step : image.length - 1 - step];
                pending |= sz_regex_looks_step_(engine, pike, members, sz_regex_looks_between_(before, after), after);
            }
            before = after, position = next;
        }
    }
    pike->slots_count = slots_count;
}

/**
 *  @brief Finds the leftmost-first match starting at or past @p from, as `regex-automata` does.
 *  @param[out] offset Folded bytes into its first unit the match starts at, zero unless uncased.
 *
 *  Windows are marked a batch at a time, from the window the cursor first reaches. An empty
 *  match before a UTF-8 continuation byte is dropped and the search resumes one byte further;
 *  only an empty match can start there, as no class matches it. An uncased start may lie between
 *  the runes one codepoint folds to, so each of them is tried in turn, and an empty match there
 *  is dropped too.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_search_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                     sz_size_t length, sz_regex_walk_t *walk, sz_size_t from,
                                                     sz_size_t *start, sz_size_t *end, sz_u32_t *pattern,
                                                     sz_size_t *offset) {
    sz_u8_t const *const bytes = (sz_u8_t const *)haystack;
    for (;;) {
        sz_size_t candidate = STRINGZILLA_SIZE_MAX;
        sz_regex_image_t image;
        // No match is empty where prefixes exist, so none starts at the haystack's end.
        if (engine->prefixes) {
            sz_cptr_t const found = from < length
                                        ? walk->find_prefixes(haystack + from, length - from, engine->prefixes)
                                        : STRINGZILLA_NULL;
            if (found) candidate = (sz_size_t)(found - haystack);
        }
        else
            while (from <= length) {
                sz_size_t window = walk->window, base, edge;
                if (from < walk->bounds[window] || (from >= walk->bounds[window + 1] && window + 1 != walk->windows)) {
                    window = sz_min_of_two(from / walk->window_bytes, walk->windows - 1);
                    if (from < walk->bounds[window]) --window;
                    walk->window = window;
                }
                base = walk->bounds[window], edge = walk->bounds[window + 1];
                if (window < walk->batch || window >= walk->batch + walk->batch_count)
                    sz_regex_mark_batch_serial_(engine, haystack, length, walk, window);
                candidate = sz_regex_next_start_(walk->starts + (window - walk->batch) * walk->window_words, base, from,
                                                 edge == length ? edge : edge - 1);
                if (candidate != STRINGZILLA_SIZE_MAX || edge == length) break;
                from = edge;
            }
        if (candidate == STRINGZILLA_SIZE_MAX) return sz_false_k;
        image.runes_count = 1, image.rune_starts[0] = 0;
        if (engine->case_sensitivity != sz_cased_k && candidate != length)
            sz_regex_image_after_(haystack, length, candidate, &image);
        for (sz_size_t rune = 0; rune != image.runes_count; ++rune) {
            sz_bool_t const found =
                !engine->forward_rows
                    ? sz_regex_walk_pike_(engine, &walk->pike, haystack, length, candidate, image.rune_starts[rune],
                                          engine->nfa_start, length, walk->looks, end, pattern)
                : engine->case_sensitivity != sz_cased_k
                    ? sz_regex_walk_end_folded_(engine, haystack, length, candidate, image.rune_starts[rune], walk, end,
                                                pattern)
                : walk->looks
                    ? sz_regex_walk_end_serial_(engine, bytes, length, candidate, walk->looks, walk, end, pattern)
                    : sz_regex_walk_end_serial_(engine, bytes, length, candidate, STRINGZILLA_NULL, walk, end, pattern);
            if (!found || (*end == candidate && candidate < length && (bytes[candidate] & 0xC0) == 0x80)) continue;
            *start = candidate, *offset = image.rune_starts[rune];
            return sz_true_k;
        }
        from = candidate + 1;
    }
}

/**
 *  @brief Finds the match @c find_iter yields after @p found matches of one haystack, the last
 *      ending at @p cursor.
 *
 *  The search resumes at the last match's end, and an empty match there is refused in favour of the
 *  search one byte further, so matches share no bytes and none repeats.
 */
STRINGZILLA_INLINE sz_bool_t sz_regex_next_(sz_regex_engine_t const *engine, sz_cptr_t haystack, sz_size_t length,
                                            sz_regex_walk_t *walk, sz_size_t cursor, sz_size_t found, sz_size_t *start,
                                            sz_size_t *end, sz_u32_t *pattern, sz_size_t *offset) {
    if (!sz_regex_search_serial_(engine, haystack, length, walk, cursor, start, end, pattern, offset))
        return sz_false_k;
    if (!found || *start != *end || *end != cursor) return sz_true_k;
    return sz_regex_search_serial_(engine, haystack, length, walk, cursor + 1, start, end, pattern, offset);
}

/**
 *  @brief Counts and optionally writes every match of one haystack,
 *      in the order @c find_iter yields.
 *  @param[inout] matches_count Matches of earlier haystacks, advanced by this one's; only those
 *      below @p matches_capacity are written.
 *  @return This haystack's matches.
 */
STRINGZILLA_INLINE sz_size_t sz_regex_walk_serial_(sz_regex_engine_t const *engine, sz_cptr_t haystack,
                                                   sz_size_t length, sz_regex_walk_t *walk, sz_size_t haystack_index,
                                                   sz_match_t *matches, sz_size_t matches_capacity,
                                                   sz_size_t *matches_count) {
    sz_size_t cursor = 0, found = 0, start, end, offset;
    sz_u32_t pattern;
    while (sz_regex_next_(engine, haystack, length, walk, cursor, found, &start, &end, &pattern, &offset)) {
        if (matches && *matches_count < matches_capacity) {
            sz_match_t *const match = matches + *matches_count;
            match->haystack_index = haystack_index, match->needle_index = pattern;
            match->byte_offset = start, match->byte_length = end - start;
        }
        ++*matches_count, ++found;
        cursor = end;
    }
    return found;
}

/** Counts the matches of every haystack, skipping with @p rfind_live,
 *  which is every CPU tier's counting verb. */
STRINGZILLA_INLINE sz_status_t sz_regex_counts_with_(
    sz_regex_engine_t *engine, sz_strings_view_t const *haystacks, sz_size_t *counts, sz_size_t counts_stride,
    sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_escape)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_prefixes)(sz_cptr_t, sz_size_t, sz_regex_prefixes_t const *),
    void (*fill_looks)(sz_cptr_t, sz_size_t, sz_u32_t const *, sz_u32_t, sz_u32_t *)) {
    sz_string_view_t views[64];
    sz_size_t matches_count = 0;
    if (counts_stride == 0) return sz_unexpected_dimensions_k;
    if (!engine->memory) return sz_missing_kernel_k;
    for (sz_size_t haystack_index = 0; haystack_index != haystacks->count; ++haystack_index) {
        sz_regex_walk_t walk;
        sz_status_t status;
        if (haystack_index % 64 == 0) sz_strings_get_block_(haystacks, haystack_index, views);
        status = sz_regex_walk_scratch_(engine, views[haystack_index % 64].start, views[haystack_index % 64].length,
                                        rfind_live, find_escape, find_prefixes, fill_looks, 0, &walk);
        if (status != sz_success_k) return status;
        counts[haystack_index * counts_stride] = sz_regex_walk_serial_(
            engine, views[haystack_index % 64].start, views[haystack_index % 64].length, &walk, haystack_index,
            STRINGZILLA_NULL, 0, &matches_count);
    }
    return sz_success_k;
}

/** Reports the matches of every haystack, skipping with @p rfind_live,
 *  which is every CPU tier's finding verb. */
STRINGZILLA_INLINE sz_status_t sz_regex_find_with_(
    sz_regex_engine_t *engine, sz_strings_view_t const *haystacks, sz_match_t *matches, sz_size_t matches_capacity,
    sz_size_t *matches_offsets, sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_escape)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_prefixes)(sz_cptr_t, sz_size_t, sz_regex_prefixes_t const *),
    void (*fill_looks)(sz_cptr_t, sz_size_t, sz_u32_t const *, sz_u32_t, sz_u32_t *)) {
    sz_string_view_t views[64];
    sz_size_t matches_count = 0;
    if (!engine->memory) return sz_missing_kernel_k;
    for (sz_size_t haystack_index = 0; haystack_index != haystacks->count; ++haystack_index) {
        sz_regex_walk_t walk;
        sz_status_t status;
        if (haystack_index % 64 == 0) sz_strings_get_block_(haystacks, haystack_index, views);
        status = sz_regex_walk_scratch_(engine, views[haystack_index % 64].start, views[haystack_index % 64].length,
                                        rfind_live, find_escape, find_prefixes, fill_looks, 0, &walk);
        if (status != sz_success_k) return status;
        matches_offsets[haystack_index] = matches_count;
        sz_regex_walk_serial_(engine, views[haystack_index % 64].start, views[haystack_index % 64].length, &walk,
                              haystack_index, matches, matches_capacity, &matches_count);
    }
    matches_offsets[haystacks->count] = matches_count;
    return sz_success_k;
}

#pragma endregion Matching

#pragma region Rewriting

/** Copies what of @p count bytes at @p source fits below @p capacity, at @p at of @p target. */
STRINGZILLA_INLINE void sz_regex_append_(sz_ptr_t target, sz_size_t capacity, sz_size_t at, sz_cptr_t source,
                                         sz_size_t count) {
    if (at < capacity) memcpy(target + at, source, sz_min_of_two(count, capacity - at));
}

/** The groups the pattern with the most of them has, which sizes every walk's capture slots. */
STRINGZILLA_INLINE sz_size_t sz_regex_groups_most_(sz_regex_engine_t const *engine) {
    sz_size_t most = 0;
    for (sz_size_t pattern = 0; pattern != engine->patterns_count; ++pattern)
        most = sz_max_of_two(most,
                             (sz_size_t)(engine->patterns_groups[pattern + 1] - engine->patterns_groups[pattern]));
    return most;
}

/**
 *  @brief Appends one match's replacement to @p target at @p written.
 *  @return The bytes the replacement takes, whether or not they fit below @p capacity.
 *
 *  The replacement appends its bytes as written, `$$` as one dollar, and `$N`, `${N}` or `${name}`
 *  as the text of that group of the match's pattern, `$0` the whole match; a group that took no
 *  part or does not exist is empty, and a dollar starting no reference stands for itself.
 *
 *  Groups come from a PikeVM over the match's own span, run once per match and only for a reference
 *  that needs it.
 */
STRINGZILLA_INLINE sz_size_t sz_regex_expand_(sz_regex_engine_t const *engine, sz_regex_walk_t *walk,
                                              sz_cptr_t haystack, sz_size_t length, sz_string_view_t replacement,
                                              sz_size_t start, sz_size_t offset, sz_size_t end, sz_u32_t pattern,
                                              sz_ptr_t target, sz_size_t capacity, sz_size_t written) {
    sz_cptr_t const text = replacement.start, names = engine->groups_names.data.u32_tape.data;
    sz_u32_t const *const names_offsets = engine->groups_names.data.u32_tape.offsets;
    sz_size_t const first_group = engine->patterns_groups[pattern];
    sz_size_t const groups = engine->patterns_groups[pattern + 1] - first_group;
    sz_size_t cursor = 0, appended = 0;
    sz_bool_t walked = sz_false_k;
    while (cursor != replacement.length) {
        sz_size_t next = cursor + 1, group = STRINGZILLA_SIZE_MAX, number = 0, copied = 1;
        sz_cptr_t source = text + cursor;
        sz_bool_t referenced = sz_false_k;
        if (text[cursor] == '$' && next != replacement.length) {
            sz_size_t name_begin = next, name_end = next;
            if (text[next] == '$') source = text + next++;
            else if (text[next] == '{') {
                name_begin = name_end = next + 1;
                while (name_end != replacement.length && text[name_end] != '}') ++name_end;
                if (name_end != replacement.length && name_end != name_begin)
                    referenced = sz_true_k, next = name_end + 1;
            }
            else {
                while (name_end != replacement.length && text[name_end] >= '0' && text[name_end] <= '9') ++name_end;
                if (name_end != name_begin) referenced = sz_true_k, next = name_end;
            }
            if (referenced) {
                sz_size_t digit = name_begin;
                for (; digit != name_end && text[digit] >= '0' && text[digit] <= '9'; ++digit)
                    number = number > groups ? number : number * 10 + (sz_size_t)(text[digit] - '0');
                if (digit == name_end && number <= groups) group = number;
                for (sz_size_t named = 0; digit != name_end && named != groups; ++named) {
                    sz_size_t const name_offset = names_offsets[first_group + named];
                    if (names_offsets[first_group + named + 1] - name_offset == name_end - name_begin &&
                        sz_equal_serial_(names + name_offset, text + name_begin, name_end - name_begin))
                        group = named + 1;
                }
            }
        }
        while (!referenced && next != replacement.length && text[next] != '$') ++next, ++copied;
        if (referenced) {
            source = haystack + start, copied = group == 0 ? end - start : 0;
            if (group != 0 && group != STRINGZILLA_SIZE_MAX) {
                sz_size_t found_end;
                sz_u32_t found_pattern;
                if (!walked)
                    sz_regex_walk_pike_(engine, &walk->pike, haystack, length, start, offset,
                                        engine->patterns_starts[pattern], end, walk->looks, &found_end, &found_pattern);
                walked = sz_true_k;
                if (walk->pike.matched[2 * group] != STRINGZILLA_SIZE_MAX &&
                    walk->pike.matched[2 * group + 1] != STRINGZILLA_SIZE_MAX)
                    source = haystack + walk->pike.matched[2 * group],
                    copied = walk->pike.matched[2 * group + 1] - walk->pike.matched[2 * group];
            }
        }
        sz_regex_append_(target, capacity, written + appended, source, copied);
        appended += copied, cursor = next;
    }
    return appended;
}

/** Rewrites every haystack, skipping with @p rfind_live,
 *  which is every CPU tier's replacing verb. */
STRINGZILLA_INLINE sz_status_t sz_regex_replace_with_(
    sz_regex_engine_t *engine, sz_strings_view_t const *haystacks, sz_strings_view_t const *replacements,
    sz_ptr_t target, sz_size_t target_capacity, sz_size_t *offsets,
    sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_escape)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_prefixes)(sz_cptr_t, sz_size_t, sz_regex_prefixes_t const *),
    void (*fill_looks)(sz_cptr_t, sz_size_t, sz_u32_t const *, sz_u32_t, sz_u32_t *)) {
    sz_string_view_t views[64];
    sz_size_t written = 0, slots_count;
    if (!engine->memory) return sz_missing_kernel_k;
    if (replacements->count != engine->patterns_count) return sz_unexpected_dimensions_k;
    slots_count = 2 * sz_regex_groups_most_(engine) + 2;
    for (sz_size_t haystack_index = 0; haystack_index != haystacks->count; ++haystack_index) {
        sz_regex_walk_t walk;
        sz_size_t cursor = 0, found = 0, start, end, offset, length;
        sz_cptr_t haystack;
        sz_u32_t pattern;
        sz_status_t status;
        if (haystack_index % 64 == 0) sz_strings_get_block_(haystacks, haystack_index, views);
        haystack = views[haystack_index % 64].start, length = views[haystack_index % 64].length;
        status = sz_regex_walk_scratch_(engine, haystack, length, rfind_live, find_escape, find_prefixes, fill_looks,
                                        slots_count, &walk);
        if (status != sz_success_k) return status;
        offsets[haystack_index] = written;
        while (sz_regex_next_(engine, haystack, length, &walk, cursor, found, &start, &end, &pattern, &offset)) {
            sz_size_t const index = pattern;
            sz_string_view_t replacement;
            sz_regex_append_(target, target_capacity, written, haystack + cursor, start - cursor);
            written += start - cursor;
            sz_strings_get_(replacements, &index, 1, &replacement);
            written += sz_regex_expand_(engine, &walk, haystack, length, replacement, start, offset, end, pattern,
                                        target, target_capacity, written);
            cursor = end, ++found;
        }
        sz_regex_append_(target, target_capacity, written, haystack + cursor, length - cursor);
        written += length - cursor;
    }
    offsets[haystacks->count] = written;
    return sz_success_k;
}

/**
 *  @brief Locates the groups of every haystack's first match of @p pattern_index, skipping with
 *      @p rfind_live, which is every CPU tier's extracting verb.
 *
 *  One walk per haystack finds the match, then a PikeVM over its own span reads its groups; rows
 *  past @p groups_capacity are counted and not written, as @c sz_regex_find_with_ counts matches.
 */
STRINGZILLA_INLINE sz_status_t sz_regex_extract_with_(
    sz_regex_engine_t *engine, sz_strings_view_t const *haystacks, sz_size_t pattern_index, sz_match_t *groups,
    sz_size_t groups_capacity, sz_size_t *groups_offsets,
    sz_cptr_t (*rfind_live)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_escape)(sz_cptr_t, sz_size_t, sz_byteset_t const *),
    sz_cptr_t (*find_prefixes)(sz_cptr_t, sz_size_t, sz_regex_prefixes_t const *),
    void (*fill_looks)(sz_cptr_t, sz_size_t, sz_u32_t const *, sz_u32_t, sz_u32_t *)) {
    sz_string_view_t views[64];
    sz_size_t groups_count, rows_count = 0;
    if (!engine->memory) return sz_missing_kernel_k;
    if (pattern_index >= engine->patterns_count) return sz_unexpected_dimensions_k;
    groups_count = engine->patterns_groups[pattern_index + 1] - engine->patterns_groups[pattern_index];
    for (sz_size_t haystack_index = 0; haystack_index != haystacks->count; ++haystack_index) {
        sz_regex_walk_t walk;
        sz_size_t cursor = 0, found = 0, start = 0, end = 0, offset = 0, length;
        sz_cptr_t haystack;
        sz_u32_t pattern = 0;
        sz_bool_t matched = sz_false_k;
        sz_status_t status;
        if (haystack_index % 64 == 0) sz_strings_get_block_(haystacks, haystack_index, views);
        haystack = views[haystack_index % 64].start, length = views[haystack_index % 64].length;
        status = sz_regex_walk_scratch_(engine, haystack, length, rfind_live, find_escape, find_prefixes, fill_looks,
                                        2 * groups_count + 2, &walk);
        if (status != sz_success_k) return status;
        groups_offsets[haystack_index] = rows_count;
        while (!matched &&
               sz_regex_next_(engine, haystack, length, &walk, cursor, found, &start, &end, &pattern, &offset))
            matched = (sz_bool_t)(pattern == pattern_index), cursor = end, ++found;
        if (!matched) continue;
        if (groups_count)
            sz_regex_walk_pike_(engine, &walk.pike, haystack, length, start, offset,
                                engine->patterns_starts[pattern_index], end, walk.looks, &cursor, &pattern);
        for (sz_size_t group = 0; group <= groups_count; ++group) {
            sz_size_t const *const slots = walk.pike.matched + 2 * group;
            if (group && (slots[0] == STRINGZILLA_SIZE_MAX || slots[1] == STRINGZILLA_SIZE_MAX)) continue;
            if (rows_count < groups_capacity) {
                sz_match_t *const row = groups + rows_count;
                row->haystack_index = haystack_index, row->needle_index = group;
                row->byte_offset = group ? slots[0] : start;
                row->byte_length = group ? slots[1] - slots[0] : end - start;
            }
            ++rows_count;
        }
    }
    groups_offsets[haystacks->count] = rows_count;
    return sz_success_k;
}

#pragma endregion Rewriting

#pragma region Serial Backends

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_regex_engine_init_serial(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                        sz_case_sensitivity_t case_sensitivity,
                                                        sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_engine_compile_(engine, patterns, case_sensitivity, malformed_index, sz_cap_serial_k, allocator,
                                    sz_regex_reverse_budget_k, sz_regex_forward_budget_k, stream);
}

STRINGZILLA_API sz_status_t sz_regex_counts_serial(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_counts_with_(engine, haystacks, counts, counts_stride, &sz_rfind_byteset_serial_, STRINGZILLA_NULL,
                                 &sz_regex_find_prefixes_serial_, &sz_regex_fill_looks_serial_);
}

STRINGZILLA_API sz_status_t sz_regex_find_serial(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                 sz_size_t *matches_offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_find_with_(engine, haystacks, matches, matches_capacity, matches_offsets, &sz_rfind_byteset_serial_,
                               STRINGZILLA_NULL, &sz_regex_find_prefixes_serial_, &sz_regex_fill_looks_serial_);
}

STRINGZILLA_API sz_status_t sz_regex_replace_serial(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_strings_view_t const *replacements, sz_ptr_t target,
                                                    sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_replace_with_(engine, haystacks, replacements, target, target_capacity, offsets,
                                  &sz_rfind_byteset_serial_, STRINGZILLA_NULL, &sz_regex_find_prefixes_serial_,
                                  &sz_regex_fill_looks_serial_);
}

STRINGZILLA_API sz_status_t sz_regex_extract_serial(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_size_t pattern_index, sz_match_t *groups,
                                                    sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                    sz_stream_t stream) {
    if (stream) return sz_device_memory_mismatch_k;
    return sz_regex_extract_with_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                  &sz_rfind_byteset_serial_, STRINGZILLA_NULL, &sz_regex_find_prefixes_serial_,
                                  &sz_regex_fill_looks_serial_);
}

#endif // STRINGZILLA_TARGET_SERIAL

#pragma endregion Serial Backends

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_REGEX_SERIAL_H_
