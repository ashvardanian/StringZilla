/**
 *  @file include/stringzilla/subdocuments/serial.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Serial backend for path queries: the path compiler, the block classifier, and the walk
 *      every other CPU tier reuses with a classifier of its own.
 *
 *  A path compiles into steps, one per selector, and a document is walked once with a bitset of
 *  live steps per open container, so every path advances together and no tree is ever built. The
 *  walk reads only the structural bytes of 64-byte blocks, and steps over a container no live step
 *  can enter by counting its brackets a block at a time.
 *
 *  @sa include/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_SERIAL_H_
#define STRINGZILLA_SUBDOCUMENTS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h"      // `sz_capability_t`
#include "stringzilla/regex.h"             // `sz_regex_engine_t`, `sz_regex_counts`
#include "stringzilla/find/serial.h"       // `sz_find_byte_serial_`
#include "stringzilla/utf8_runes/serial.h" // `sz_rune_encode`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Engine

/** What a filter instruction does to the values its program holds, evaluated in postfix order. */
typedef enum sz_subdocuments_operation_t {

    /** Pushes a literal, @c operand being its @c sz_subdocuments_kind_t. */
    sz_subdocuments_operation_literal_k = 0,

    /** Pushes whether the query ending at step @c operand selects any node: an existence test. */
    sz_subdocuments_operation_exists_k = 1,

    /** Pushes the value of the one node the query ending at step @c operand selects, or Nothing for
     *  none or several: a singular query, or `value()`. */
    sz_subdocuments_operation_value_k = 2,

    /** Pushes how many nodes the query ending at step @c operand selects: `count()`. */
    sz_subdocuments_operation_count_k = 3,

    /** Replaces a value with its `length()`. */
    sz_subdocuments_operation_length_k = 4,

    /** Replaces a value with whether pattern @c operand matches all of it: `match()`. */
    sz_subdocuments_operation_match_k = 5,

    /** Replaces a value with whether pattern @c operand matches some of it: `search()`. */
    sz_subdocuments_operation_search_k = 6,

    /** Replaces two values with their comparison by @c operand, a
     *  @c sz_subdocuments_comparison_t. */
    sz_subdocuments_operation_compare_k = 7,

    /** Negates a logical value. */
    sz_subdocuments_operation_not_k = 8,

    /** Replaces two logical values with their conjunction. */
    sz_subdocuments_operation_and_k = 9,

    /** Replaces two logical values with their disjunction. */
    sz_subdocuments_operation_or_k = 10,
} sz_subdocuments_operation_t;

/** The kind of a value a filter program holds, RFC 9535's Nothing standing for no value at all. */
typedef enum sz_subdocuments_kind_t {
    sz_subdocuments_kind_nothing_k = 0,
    sz_subdocuments_kind_logical_k = 1,
    sz_subdocuments_kind_null_k = 2,
    sz_subdocuments_kind_false_k = 3,
    sz_subdocuments_kind_true_k = 4,
    sz_subdocuments_kind_number_k = 5,
    sz_subdocuments_kind_string_k = 6,
    sz_subdocuments_kind_array_k = 7,
    sz_subdocuments_kind_object_k = 8,
} sz_subdocuments_kind_t;

/** The comparison operators of RFC 9535 filters. */
typedef enum sz_subdocuments_comparison_t {
    sz_subdocuments_comparison_equal_k = 0,
    sz_subdocuments_comparison_not_equal_k = 1,
    sz_subdocuments_comparison_less_k = 2,
    sz_subdocuments_comparison_less_equal_k = 3,
    sz_subdocuments_comparison_greater_k = 4,
    sz_subdocuments_comparison_greater_equal_k = 5,
} sz_subdocuments_comparison_t;

/** One instruction of a filter program. */
typedef struct sz_subdocuments_instruction_t {

    /** What it does, a @c sz_subdocuments_operation_t. */
    sz_u32_t operation;

    /** A step, a pattern, a comparison or a literal's kind, as @c operation reads it. */
    sz_u32_t operand;

    /** A number literal's value. */
    sz_f64_t number;

    /** Where a string literal or a `match()` or `search()` pattern starts in @c strings_data. */
    sz_u32_t string_offset;

    /** Its unescaped bytes. */
    sz_u32_t string_length;
} sz_subdocuments_instruction_t;

/** The filters of one bracketed selection, joined by disjunction, and the program deciding them. */
typedef struct sz_subdocuments_filter_t {

    /** The step whose selection they decide. */
    sz_u32_t step;

    /** The program's first instruction in @c instructions. */
    sz_u32_t instructions_begin;

    /** One past its last. */
    sz_u32_t instructions_end;
} sz_subdocuments_filter_t;

/**
 *  @brief The compiled paths: one bit-parallel automaton over keys and positions, with no table.
 *
 *  A step is one segment of one path, numbered path after path, the first step of every path
 *  being its root; "live" is a bitset of steps. Entering a member, an element, an attribute or a
 *  text node advances every live step that is not final to the next one, keeps those whose
 *  selectors accept the child's key, name, kind or position, and carries the steps a descendant
 *  segment keeps live at every depth below.
 */
typedef struct sz_subdocuments_engine_t {

    /** Bytes of every key, element or attribute name the paths spell, back to back, to
     *  confirm a hash hit. */
    sz_cptr_t names_data;

    /** Offsets into @c names_data, one per name and one past the last. */
    sz_u32_t const *names_offsets;

    /** Open-addressed name hash table, each slot a name index or @c sz_subdocuments_no_name_k. */
    sz_u32_t const *names_slots;

    /** Per name, the steps whose selector is that name, @c words_count words each. */
    sz_u64_t const *names_steps;

    /** Steps any key or index enters: wildcards. */
    sz_u64_t const *wildcard_steps;

    /** Steps a descendant segment keeps live at every depth below: each one
     *  precedes a `..` selector. */
    sz_u64_t const *descendant_steps;

    /** Steps with a slice the array's length settles: negative indices or bounds, or a step
     *  under -1. An array such a step may enter opens a pending run, settled when it closes. */
    sz_u64_t const *relative_steps;

    /** XPath steps selecting attributes, `@name` and `@*`, which select no element. */
    sz_u64_t const *attribute_steps;

    /** XPath steps selecting text nodes, `text()`. */
    sz_u64_t const *text_steps;

    /** Steps with a filter selector, whose selection of a node waits for that node's close. */
    sz_u64_t const *filter_steps;

    /** Per filter, the root steps of its queries, @c words_count words each: the
     *  steps `@` stands for. */
    sz_u64_t const *filters_roots;

    /** Every filter, in the order compiled. */
    sz_subdocuments_filter_t const *filters;

    /** Every filter's program, back to back. */
    sz_subdocuments_instruction_t const *instructions;

    /** Bytes of every string literal and pattern the filters spell, unescaped. */
    sz_cptr_t strings_data;

    /** The last step of every path and filter query, which never advances. */
    sz_u64_t const *final_steps;

    /** The first step of every path, live at the document's root; a filter query's first step is
     *  live only at the nodes its filter decides. */
    sz_u64_t const *root_steps;

    /** The array positions of every index and slice selector, step after step. */
    sz_subdocuments_slice_t const *slices;

    /** Per step, where its slices start in @c slices, and one past the last step's. */
    sz_u32_t const *slices_offsets;

    /** Per step, the path it belongs to, reported as @c needle_index. */
    sz_u32_t const *steps_paths;

    /** Steps across all paths. */
    sz_u32_t steps_count;

    /** 64-bit words per step bitset. */
    sz_u32_t words_count;

    /** Slots of @c names_slots, a power of two, or zero when no step selects by name. */
    sz_u32_t names_slots_count;

    /** Paths the vocabulary holds; the queries of filters take steps after
     *  theirs, and report nothing. */
    sz_u32_t paths_count;

    /** Filters the paths hold. */
    sz_u32_t filters_count;

    /** Patterns of `match()` and `search()`, each compiled behind a tag naming it, so
     *  that one engine answers for each alone: a tested string carries the tag of its
     *  pattern. Empty without any. */
    sz_regex_engine_t regex;

    /** Patterns @c regex holds, zero when it holds none and was never compiled. */
    sz_u32_t patterns_count;

    /** The document grammar and the path language that goes with it. */
    sz_subdocuments_format_t format;

    /** The capability that compiled the paths. */
    sz_capability_t capability;

    /** What built both blocks below. */
    sz_allocator_t allocator;

    /** The compiled paths. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** The host depth stack of live sets and the nodes of a pending run, grown on demand like the
     *  round scratch of @c levenshtein. */
    void *scratch;

    /** Bytes of that block. */
    sz_size_t scratch_bytes;
} sz_subdocuments_engine_t;

/** The name index of an empty slot of @c names_slots. */
static sz_u32_t const sz_subdocuments_no_name_k = 0xFFFFFFFFu;

/** The largest index or slice bound a path may spell, RFC 9535's I-JSON ceiling of 2⁵³ - 1. */
static sz_i64_t const sz_subdocuments_index_max_k = 9007199254740991ll;

/** A slice bound past either end of every array, standing for an omitted one. */
static sz_i64_t const sz_subdocuments_unbounded_k = 9007199254740992ll;

/** The deepest nesting of filters, parentheses and calls a path may spell, which bounds the
 *  recursion of the compiler and the values a filter program holds at once. */
enum { sz_subdocuments_filter_depth_k = 32 };

/** Bytes the compiled paths are aligned and padded to, so no page of theirs, which the host reads
 *  every call, also holds a buffer a device writes. */
enum { sz_subdocuments_page_bytes_k = 1 << 16 };

#pragma endregion Engine

#pragma region Paths

/** Reads four hexadecimal digits, or returns a value past U+10FFFF if one is not a digit. */
STRINGZILLA_INLINE sz_u32_t sz_subdocuments_hex_unit_(sz_cptr_t digits) {
    sz_u32_t unit = 0;
    for (sz_size_t index = 0; index != 4; ++index) {
        sz_u8_t const digit = (sz_u8_t)digits[index], lowered = (sz_u8_t)(digit | 0x20);
        if (digit >= '0' && digit <= '9') unit = unit << 4 | (sz_u32_t)(digit - '0');
        else if (lowered >= 'a' && lowered <= 'f') unit = unit << 4 | (sz_u32_t)(lowered - 'a' + 10);
        else return 0xFFFFFFFFu;
    }
    return unit;
}

/**
 *  @brief Unescapes a string body delimited by @p quote, under RFC 8259 for keys and RFC 9535 for
 *      name selectors, whose only difference is which quote may be escaped.
 *  @return The unescaped length, of which only the first @p capacity bytes are written, or
 *      @c STRINGZILLA_SIZE_MAX for a raw control byte, an unknown escape or an unpaired surrogate.
 */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_unescape_(sz_cptr_t source, sz_size_t length, sz_u8_t quote,
                                                       sz_ptr_t target, sz_size_t capacity) {
    sz_size_t read = 0, written = 0;
    while (read != length) {
        sz_u8_t encoded[4];
        sz_size_t encoded_length = 1, index;
        sz_u8_t byte = (sz_u8_t)source[read++];
        if (byte < 0x20) return STRINGZILLA_SIZE_MAX;
        if (byte == '\\') {
            if (read == length) return STRINGZILLA_SIZE_MAX;
            byte = (sz_u8_t)source[read++];
            switch (byte) {
            case 'b': byte = '\b'; break;
            case 'f': byte = '\f'; break;
            case 'n': byte = '\n'; break;
            case 'r': byte = '\r'; break;
            case 't': byte = '\t'; break;
            case '/':
            case '\\': break;
            case 'u': {
                sz_u32_t rune;
                if (length - read < 4) return STRINGZILLA_SIZE_MAX;
                rune = sz_subdocuments_hex_unit_(source + read);
                read += 4;
                if (rune >= 0xDC00 && rune <= 0xDFFF) return STRINGZILLA_SIZE_MAX;
                if (rune >= 0xD800 && rune <= 0xDBFF) {
                    sz_u32_t low;
                    if (length - read < 6 || source[read] != '\\' || source[read + 1] != 'u')
                        return STRINGZILLA_SIZE_MAX;
                    low = sz_subdocuments_hex_unit_(source + read + 2);
                    if (low < 0xDC00 || low > 0xDFFF) return STRINGZILLA_SIZE_MAX;
                    read += 6;
                    rune = 0x10000 + ((rune - 0xD800) << 10) + (low - 0xDC00);
                }
                encoded_length = (sz_size_t)sz_rune_encode(rune, encoded);
                if (encoded_length == 0) return STRINGZILLA_SIZE_MAX;
                byte = encoded[0];
                break;
            }
            default:
                if (byte != quote) return STRINGZILLA_SIZE_MAX;
                break;
            }
        }
        if (encoded_length == 1) encoded[0] = byte;
        for (index = 0; index != encoded_length; ++index, ++written)
            if (written < capacity) target[written] = (char)encoded[index];
    }
    return written;
}

/** A word of @p bytes in memory order, its first byte the lowest. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_word_load_(sz_cptr_t bytes) {
    sz_u64_t const word = sz_u64_load(bytes).u64;
    return STRINGZILLA_ARCH_BIG_ENDIAN_ ? sz_u64_bytes_reverse(word) : word;
}

/** The @p tail_bytes, at most seven, at @p tail as one word, its first byte the lowest, read whole
 *  while @p readable bytes from @p tail on allow it. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_tail_word_(sz_cptr_t tail, sz_size_t tail_bytes, sz_size_t readable) {
    sz_u64_t word = 0;
    sz_size_t index;
    if (readable >= sizeof(sz_u64_t))
        return sz_subdocuments_word_load_(tail) & sz_u64_mask_until_serial_(tail_bytes * 8);
    for (index = tail_bytes; index-- != 0;) word = word << 8 | (sz_u8_t)tail[index];
    return word;
}

/**
 *  @brief The slot of the name table holding @p key, or the empty slot it would take.
 *
 *  Keys are short and many, so the hash takes one multiply and shift per 8 bytes, and both the hash
 *  and the comparisons read a word at a time; the names are padded past their last word.
 *
 *  @param[in] readable Bytes from @p key on that may be read, at least @p key_length.
 */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_name_slot_(sz_cptr_t names_data, sz_u32_t const *names_offsets,
                                                        sz_u32_t const *names_slots, sz_size_t slots_count,
                                                        sz_cptr_t key, sz_size_t key_length, sz_size_t readable) {
    sz_size_t const mask = slots_count - 1, words = key_length / 8, tail_bytes = key_length % 8;
    sz_u64_t const tail_word = sz_subdocuments_tail_word_(key + words * 8, tail_bytes, readable - words * 8);
    sz_u64_t hash = key_length * 0x9E3779B97F4A7C15ull;
    sz_size_t index, slot;
    for (index = 0; index != words; ++index) {
        hash = (hash ^ sz_u64_load(key + index * 8).u64) * 0xFF51AFD7ED558CCDull;
        hash ^= hash >> 32;
    }
    // The tail enters the hash with its first byte the highest.
    hash = (hash ^ (tail_bytes ? sz_u64_bytes_reverse(tail_word) >> (8 * (8 - tail_bytes)) : 0)) *
           0xC4CEB9FE1A85EC53ull;
    for (slot = (sz_size_t)(hash ^ hash >> 32) & mask;; slot = (slot + 1) & mask) {
        sz_u32_t const name = names_slots[slot];
        sz_cptr_t name_bytes;
        sz_u64_t differences;
        if (name == sz_subdocuments_no_name_k) return slot;
        if (names_offsets[name + 1] - names_offsets[name] != key_length) continue;
        name_bytes = names_data + names_offsets[name];
        differences = sz_subdocuments_tail_word_(name_bytes + words * 8, tail_bytes, sizeof(sz_u64_t)) ^ tail_word;
        for (index = 0; index != words; ++index)
            differences |= sz_u64_load(key + index * 8).u64 ^ sz_u64_load(name_bytes + index * 8).u64;
        if (!differences) return slot;
    }
}

/** Running totals of the path compiler: sizes when counting, next free entries when publishing. */
typedef struct sz_subdocuments_builder_t {

    /** Steps so far, the last of which takes the next selector. */
    sz_size_t steps_count;

    /** Name selectors so far, repeats included, which bounds the distinct names. */
    sz_size_t name_selectors_count;

    /** Index and slice selectors so far. */
    sz_size_t slices_count;

    /** Distinct names so far, counted on the publishing pass only. */
    sz_size_t names_count;

    /** Bytes of @c names_data so far: escaped while counting, which bounds
     *  the unescaped ones published. */
    sz_size_t names_bytes;

    /** Filters so far, the disjoined filters of one bracketed selection counting once. */
    sz_size_t filters_count;

    /** Filter instructions so far. */
    sz_size_t instructions_count;

    /** Patterns of `match()` and `search()` so far. */
    sz_size_t patterns_count;

    /** Bytes of @c strings_data so far, bounded by the escaped ones while counting. */
    sz_size_t strings_bytes;

    /** Words of each step bitset, settled by the counting pass. */
    sz_size_t words_count;

    /** Slots of the name table, settled by the counting pass. */
    sz_size_t names_slots_count;

    /** Name bytes being published, null on the counting pass like every array below. */
    sz_ptr_t names_data;

    /** Name offsets being published. */
    sz_u32_t *names_offsets;

    /** Name table being published. */
    sz_u32_t *names_slots;

    /** Per-name step bitsets being published. */
    sz_u64_t *names_steps;

    /** Wildcard steps being published. */
    sz_u64_t *wildcard_steps;

    /** Descendant steps being published. */
    sz_u64_t *descendant_steps;

    /** Steps with a slice counted from the array's end, being published. */
    sz_u64_t *relative_steps;

    /** Attribute steps being published. */
    sz_u64_t *attribute_steps;

    /** Text steps being published. */
    sz_u64_t *text_steps;

    /** Filter steps being published. */
    sz_u64_t *filter_steps;

    /** Per-filter query roots being published. */
    sz_u64_t *filters_roots;

    /** Final steps being published. */
    sz_u64_t *final_steps;

    /** Root steps being published. */
    sz_u64_t *root_steps;

    /** Slices being published. */
    sz_subdocuments_slice_t *slices;

    /** Filters being published. */
    sz_subdocuments_filter_t *filters;

    /** Filter instructions being published. */
    sz_subdocuments_instruction_t *instructions;

    /** String literal and pattern bytes being published. */
    sz_ptr_t strings_data;

    /** Per-step slice offsets being published. */
    sz_u32_t *slices_offsets;

    /** Per-step paths being published. */
    sz_u32_t *steps_paths;
} sz_subdocuments_builder_t;

/** Starts a step of path @p path_index, which a descendant segment's selectors make the step after
 *  one that stays live at every depth below. */
STRINGZILLA_INLINE void sz_subdocuments_builder_step_(sz_subdocuments_builder_t *builder, sz_bool_t descendant,
                                                      sz_size_t path_index) {
    sz_size_t const step = builder->steps_count++;
    if (!builder->steps_paths) return;
    builder->steps_paths[step] = (sz_u32_t)path_index;
    builder->slices_offsets[step + 1] = (sz_u32_t)builder->slices_count;
    if (descendant == sz_true_k) builder->descendant_steps[(step - 1) >> 6] |= (sz_u64_t)1 << ((step - 1) & 63);
}

/** Adds a name selector, @p name being the body of a literal quoted by @p quote,
 *  to the last step. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_builder_name_(sz_subdocuments_builder_t *builder, sz_cptr_t name,
                                                             sz_size_t name_length, sz_u8_t quote) {
    sz_size_t const step = builder->steps_count - 1;
    sz_size_t slot;
    sz_ptr_t const target = builder->names_data ? builder->names_data + builder->names_bytes : STRINGZILLA_NULL;
    sz_size_t const unescaped_length = sz_subdocuments_unescape_(name, name_length, quote, target,
                                                                 target ? name_length : 0);
    sz_u32_t name_index;
    if (unescaped_length == STRINGZILLA_SIZE_MAX) return sz_invalid_syntax_k;
    ++builder->name_selectors_count;
    if (!target) {
        builder->names_bytes += name_length;
        return sz_success_k;
    }
    // The block holds a word past the last name, which @ref sz_subdocuments_engine_compile_ sizes.
    slot = sz_subdocuments_name_slot_(builder->names_data, builder->names_offsets, builder->names_slots,
                                      builder->names_slots_count, target, unescaped_length,
                                      unescaped_length + sizeof(sz_u64_t));
    name_index = builder->names_slots[slot];
    if (name_index == sz_subdocuments_no_name_k) {
        name_index = (sz_u32_t)builder->names_count++;
        builder->names_bytes += unescaped_length;
        builder->names_offsets[name_index + 1] = (sz_u32_t)builder->names_bytes;
        builder->names_slots[slot] = name_index;
    }
    builder->names_steps[name_index * builder->words_count + (step >> 6)] |= (sz_u64_t)1 << (step & 63);
    return sz_success_k;
}

/** Adds a wildcard selector to the last step. */
STRINGZILLA_INLINE void sz_subdocuments_builder_wildcard_(sz_subdocuments_builder_t *builder) {
    sz_size_t const step = builder->steps_count - 1;
    if (builder->wildcard_steps) builder->wildcard_steps[step >> 6] |= (sz_u64_t)1 << (step & 63);
}

/** Makes the last step select attributes instead of elements. */
STRINGZILLA_INLINE void sz_subdocuments_builder_attribute_(sz_subdocuments_builder_t *builder) {
    sz_size_t const step = builder->steps_count - 1;
    if (builder->attribute_steps) builder->attribute_steps[step >> 6] |= (sz_u64_t)1 << (step & 63);
}

/** Makes the last step select text nodes. */
STRINGZILLA_INLINE void sz_subdocuments_builder_text_(sz_subdocuments_builder_t *builder) {
    sz_size_t const step = builder->steps_count - 1;
    if (builder->text_steps) builder->text_steps[step >> 6] |= (sz_u64_t)1 << (step & 63);
}

/** Adds an index or slice selector to the last step. */
STRINGZILLA_INLINE void sz_subdocuments_builder_slice_(sz_subdocuments_builder_t *builder,
                                                       sz_subdocuments_slice_t slice) {
    sz_size_t const step = builder->steps_count - 1;
    // Positions counted from the end, or a stride anchored there, wait for the array's length.
    sz_bool_t const relative = slice.step != 0 && (slice.step < -1 || slice.begin < 0 ||
                                                   (slice.end < 0 && slice.end != -sz_subdocuments_unbounded_k))
                                   ? sz_true_k
                                   : sz_false_k;
    sz_size_t const index = builder->slices_count++;
    if (!builder->slices) return;
    builder->slices[index] = slice;
    builder->slices_offsets[step + 1] = (sz_u32_t)builder->slices_count;
    if (relative == sz_true_k) builder->relative_steps[step >> 6] |= (sz_u64_t)1 << (step & 63);
}

/** Adds a filter selector to the last step, whose program comes later. */
STRINGZILLA_INLINE void sz_subdocuments_builder_filter_(sz_subdocuments_builder_t *builder) {
    sz_size_t const step = builder->steps_count - 1;
    if (builder->filter_steps) builder->filter_steps[step >> 6] |= (sz_u64_t)1 << (step & 63);
}

/** Stores the string literal @p body quoted by @p quote, returning where its unescaped bytes start
 *  and how many there are, or @c sz_invalid_syntax_k for a broken escape. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_builder_string_(sz_subdocuments_builder_t *builder, sz_cptr_t body,
                                                               sz_size_t length, sz_u8_t quote, sz_u32_t *offset,
                                                               sz_u32_t *unescaped_length) {
    sz_ptr_t const target = builder->strings_data ? builder->strings_data + builder->strings_bytes : STRINGZILLA_NULL;
    sz_size_t const unescaped = sz_subdocuments_unescape_(body, length, quote, target, target ? length : 0);
    if (unescaped == STRINGZILLA_SIZE_MAX) return sz_invalid_syntax_k;
    *offset = (sz_u32_t)builder->strings_bytes, *unescaped_length = (sz_u32_t)unescaped;
    builder->strings_bytes += target ? unescaped : length;
    return sz_success_k;
}

/** The first byte of @p path at or after @p position that is not RFC 9535 blank space. */
STRINGZILLA_CONSTEXPR sz_size_t sz_subdocuments_skip_blanks_(sz_cptr_t path, sz_size_t length, sz_size_t position) {
    while (position != length &&
           (path[position] == ' ' || path[position] == '\t' || path[position] == '\n' || path[position] == '\r'))
        ++position;
    return position;
}

/** Bytes a member name shorthand may start with: letters, the underscore, every non-ASCII byte. */
STRINGZILLA_CONSTEXPR sz_bool_t sz_subdocuments_is_name_first_(sz_u8_t byte) {
    sz_u8_t const lowered = (sz_u8_t)(byte | 0x20);
    return (lowered >= 'a' && lowered <= 'z') || byte == '_' || byte >= 0x80 ? sz_true_k : sz_false_k;
}

/** The end of the string literal opened by the quote at @p path [position], one past its closing
 *  quote, or zero if it never closes. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_skip_literal_(sz_cptr_t path, sz_size_t length, sz_size_t position) {
    sz_u8_t const quote = (sz_u8_t)path[position++];
    while (position < length && (sz_u8_t)path[position] != quote) position += path[position] == '\\' ? 2 : 1;
    return position < length ? position + 1 : 0;
}

/**
 *  @brief Reads an RFC 9535 integer at @p path [*position]: an optional minus, no leading zeros, no
 *      `-0`, and a magnitude of at most 2⁵³ - 1.
 *  @return Whether one was there, leaving @p position where it was otherwise.
 */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_parse_integer_(sz_cptr_t path, sz_size_t length, sz_size_t *position,
                                                            sz_i64_t *value) {
    sz_size_t cursor = *position;
    sz_bool_t const negative = cursor != length && path[cursor] == '-' ? sz_true_k : sz_false_k;
    sz_i64_t parsed = 0;
    cursor += negative == sz_true_k;
    if (cursor == length || path[cursor] < '0' || path[cursor] > '9') return sz_false_k;
    if (path[cursor] == '0') {
        if (negative == sz_true_k) return sz_false_k;
        ++cursor;
    }
    else
        for (; cursor != length && path[cursor] >= '0' && path[cursor] <= '9'; ++cursor) {
            parsed = parsed * 10 + (sz_i64_t)(path[cursor] - '0');
            if (parsed > sz_subdocuments_index_max_k) return sz_false_k;
        }
    if (cursor != length && path[cursor] >= '0' && path[cursor] <= '9') return sz_false_k;
    *position = cursor, *value = negative == sz_true_k ? -parsed : parsed;
    return sz_true_k;
}

/**
 *  @brief Reads a JSON number at @p text [*position], RFC 9535's number literal too, into @p value.
 *
 *  A mantissa of up to 19 digits scaled by one exact power of ten is correctly rounded whenever it
 *  fits 53 bits and the power is at most 10²², which every equal spelling of a number shares.
 *
 *  @return Whether one was there, leaving @p position where it was otherwise.
 */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_parse_number_(sz_cptr_t text, sz_size_t length, sz_size_t *position,
                                                           sz_f64_t *value) {
    sz_size_t cursor = *position, digits;
    sz_bool_t const negative = cursor != length && text[cursor] == '-' ? sz_true_k : sz_false_k;
    sz_u64_t mantissa = 0;
    sz_i64_t exponent = 0, magnitude = 0;
    sz_f64_t scale = 1, result;
    cursor += negative == sz_true_k;
    if (cursor == length || text[cursor] < '0' || text[cursor] > '9') return sz_false_k;
    if (text[cursor] == '0') ++cursor;
    else
        for (; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (mantissa < 1000000000000000000ull) mantissa = mantissa * 10 + (sz_u64_t)(text[cursor] - '0');
            else ++exponent;
    if (cursor != length && text[cursor] == '.') {
        for (digits = ++cursor; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (mantissa < 1000000000000000000ull)
                mantissa = mantissa * 10 + (sz_u64_t)(text[cursor] - '0'), --exponent;
        if (cursor == digits) return sz_false_k;
    }
    if (cursor != length && (text[cursor] | 0x20) == 'e') {
        sz_bool_t negative_exponent;
        ++cursor;
        negative_exponent = cursor != length && text[cursor] == '-' ? sz_true_k : sz_false_k;
        if (cursor != length && (text[cursor] == '+' || text[cursor] == '-')) ++cursor;
        for (digits = cursor; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (magnitude < 100000) magnitude = magnitude * 10 + (text[cursor] - '0');
        if (cursor == digits) return sz_false_k;
        exponent += negative_exponent == sz_true_k ? -magnitude : magnitude;
    }
    for (magnitude = exponent < 0 ? -exponent : exponent; magnitude > 0 && scale < 1e308; --magnitude) scale *= 10;
    result = exponent < 0 ? (sz_f64_t)mantissa / scale : (sz_f64_t)mantissa * scale;
    *position = cursor, *value = negative == sz_true_k ? -result : result;
    return sz_true_k;
}

/** Parses an index or slice selector at @p path [*position] into @p slice, with omitted bounds set
 *  past either end of every array as the step's sign decides, and @p index set for an index. */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_parse_slice_(sz_cptr_t path, sz_size_t length, sz_size_t *position,
                                                          sz_subdocuments_slice_t *slice, sz_bool_t *index) {
    sz_size_t cursor = *position;
    sz_i64_t begin = 0, end = 0, step = 1;
    sz_bool_t const has_begin = sz_subdocuments_parse_integer_(path, length, &cursor, &begin);
    sz_bool_t has_end;
    cursor = sz_subdocuments_skip_blanks_(path, length, cursor);
    *index = cursor == length || path[cursor] != ':' ? sz_true_k : sz_false_k;
    if (*index == sz_true_k) {
        if (has_begin == sz_false_k) return sz_false_k;
        // An index is a one-position slice, open-ended for the last element.
        slice->begin = begin, slice->end = begin == -1 ? sz_subdocuments_unbounded_k : begin + 1, slice->step = 1;
        *position = cursor;
        return sz_true_k;
    }
    cursor = sz_subdocuments_skip_blanks_(path, length, cursor + 1);
    has_end = sz_subdocuments_parse_integer_(path, length, &cursor, &end);
    cursor = sz_subdocuments_skip_blanks_(path, length, cursor);
    if (cursor != length && path[cursor] == ':') {
        cursor = sz_subdocuments_skip_blanks_(path, length, cursor + 1);
        // RFC 9535 accepts `[::0]`, which selects nothing, as a zero step already does.
        if (cursor != length && path[cursor] != ']' && path[cursor] != ',' &&
            sz_subdocuments_parse_integer_(path, length, &cursor, &step) == sz_false_k)
            return sz_false_k;
    }
    slice->step = step;
    slice->begin = has_begin == sz_true_k ? begin : step >= 0 ? 0 : sz_subdocuments_unbounded_k;
    slice->end = has_end == sz_true_k ? end : step >= 0 ? sz_subdocuments_unbounded_k : -sz_subdocuments_unbounded_k;
    *position = cursor;
    return sz_true_k;
}

/** Where a path's parser stands, shared by its segments and its filters' expressions. */
typedef struct sz_subdocuments_parser_t {

    /** The builder taking the steps, filters and instructions. */
    sz_subdocuments_builder_t *builder;

    /** The path's bytes. */
    sz_cptr_t path;

    /** Bytes the path holds. */
    sz_size_t length;

    /** The next byte to read. */
    sz_size_t position;

    /** The path every step belongs to, its filters' queries included. */
    sz_size_t path_index;

    /** Filters, parentheses and calls the parser is inside, at most
     *  @c sz_subdocuments_filter_depth_k. */
    sz_size_t depth;

    /** Values the program being emitted holds at this point, at most
     *  @c sz_subdocuments_filter_depth_k. */
    sz_size_t stack_depth;

    /** The filter whose program is being emitted. */
    sz_size_t filter;

    /** The second pass over an expression, which compiles the filters inside its
     *  queries and emits nothing. */
    sz_bool_t nesting;

    /** On the second pass, the root step of the next query. */
    sz_size_t next_root;
} sz_subdocuments_parser_t;

/** Appends one instruction to the filter program being emitted, keeping its stack within bounds. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_emit_(sz_subdocuments_parser_t *parser,
                                                     sz_subdocuments_operation_t operation, sz_size_t operand,
                                                     sz_f64_t number, sz_u32_t string_offset, sz_u32_t string_length) {
    sz_subdocuments_builder_t *const builder = parser->builder;
    sz_size_t const index = builder->instructions_count;
    if (parser->nesting == sz_true_k) return sz_success_k;
    if (operation <= sz_subdocuments_operation_count_k) ++parser->stack_depth;
    else if (operation == sz_subdocuments_operation_compare_k || operation >= sz_subdocuments_operation_and_k)
        --parser->stack_depth;
    if (parser->stack_depth > sz_subdocuments_filter_depth_k) return sz_invalid_syntax_k;
    ++builder->instructions_count;
    if (!builder->instructions) return sz_success_k;
    builder->instructions[index].operation = (sz_u32_t)operation;
    builder->instructions[index].operand = (sz_u32_t)operand;
    builder->instructions[index].number = number;
    builder->instructions[index].string_offset = string_offset;
    builder->instructions[index].string_length = string_length;
    return sz_success_k;
}

/** What a primary of a filter expression is, which settles where it may stand. */
typedef enum sz_subdocuments_primary_t {

    /** A literal, already emitted: a comparable. */
    sz_subdocuments_primary_literal_k = 0,

    /** A query, not emitted yet: an existence test, the argument of `count()` or `value()`, or a
     *  comparable if singular. */
    sz_subdocuments_primary_query_k = 1,

    /** A singular query, not emitted yet. */
    sz_subdocuments_primary_singular_k = 2,

    /** A call of `length()` or `count()`, already emitted: a comparable that is
     *  a number or Nothing. */
    sz_subdocuments_primary_number_k = 3,

    /** A call of `value()`, already emitted: a comparable. */
    sz_subdocuments_primary_value_k = 4,

    /** A call of `match()` or `search()`, already emitted: a test. */
    sz_subdocuments_primary_logical_k = 5,
} sz_subdocuments_primary_t;

STRINGZILLA_INLINE sz_status_t sz_subdocuments_parse_segments_(sz_subdocuments_parser_t *parser, sz_size_t root,
                                                               sz_bool_t filtering, sz_size_t *segments_count,
                                                               sz_bool_t *singular);
STRINGZILLA_OUTLINED_ sz_status_t sz_subdocuments_parse_or_(sz_subdocuments_parser_t *parser);

/** Emits the instruction reading a query that ends at @p final_step as
 *  @p operation, unless already emitted. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_emit_query_(sz_subdocuments_parser_t *parser,
                                                           sz_subdocuments_primary_t primary,
                                                           sz_subdocuments_operation_t operation,
                                                           sz_size_t final_step) {
    if (primary != sz_subdocuments_primary_query_k && primary != sz_subdocuments_primary_singular_k)
        return sz_success_k;
    return sz_subdocuments_emit_(parser, operation, final_step, 0, 0, 0);
}

/**
 *  @brief Parses one primary of a filter expression: a relative query, a literal, or a call of
 *      `length()`, `count()`, `value()`, `match()` or `search()`, checking each argument's type.
 *
 *  A query's own steps are numbered on the first pass, its root joining the filter's roots; on the
 *  second pass its segments compile the filters nested in it.
 *
 *  @param[out] final_step For a query, its last step, whose live nodes are what it selects.
 */
STRINGZILLA_OUTLINED_ sz_status_t sz_subdocuments_parse_primary_(sz_subdocuments_parser_t *parser,
                                                                 sz_subdocuments_primary_t *primary,
                                                                 sz_size_t *final_step) {
    sz_subdocuments_builder_t *const builder = parser->builder;
    sz_cptr_t const path = parser->path;
    sz_size_t const length = parser->length;
    sz_size_t position = sz_subdocuments_skip_blanks_(path, length, parser->position), word_end;
    sz_status_t status;
    if (position == length) return sz_invalid_syntax_k;

    if (path[position] == '@') {
        sz_size_t root, segments_count;
        sz_bool_t singular;
        parser->position = position + 1;
        if (parser->nesting == sz_false_k) {
            root = builder->steps_count;
            sz_subdocuments_builder_step_(builder, sz_false_k, parser->path_index);
            status = sz_subdocuments_parse_segments_(parser, root, sz_false_k, &segments_count, &singular);
            if (status != sz_success_k) return status;
            if (builder->final_steps) {
                builder->final_steps[(root + segments_count) >> 6] |= (sz_u64_t)1 << ((root + segments_count) & 63);
                builder->filters_roots[parser->filter * builder->words_count + (root >> 6)] |= (sz_u64_t)1
                                                                                               << (root & 63);
            }
        }
        else {
            root = parser->next_root;
            status = sz_subdocuments_parse_segments_(parser, root, sz_true_k, &segments_count, &singular);
            if (status != sz_success_k) return status;
            parser->next_root = root + 1 + segments_count;
        }
        *final_step = root + segments_count;
        *primary = singular == sz_true_k ? sz_subdocuments_primary_singular_k : sz_subdocuments_primary_query_k;
        return sz_success_k;
    }
    // Absolute queries would need the document's root while the filter's candidate is still open.
    if (path[position] == '$') return sz_invalid_syntax_k;

    if (path[position] == '\'' || path[position] == '"') {
        sz_size_t const end = sz_subdocuments_skip_literal_(path, length, position);
        sz_u32_t offset = 0, unescaped_length = 0;
        if (end == 0) return sz_invalid_syntax_k;
        if (parser->nesting == sz_false_k) {
            status = sz_subdocuments_builder_string_(builder, path + position + 1, end - position - 2,
                                                     (sz_u8_t)path[position], &offset, &unescaped_length);
            if (status != sz_success_k) return status;
        }
        parser->position = end, *primary = sz_subdocuments_primary_literal_k;
        return sz_subdocuments_emit_(parser, sz_subdocuments_operation_literal_k, sz_subdocuments_kind_string_k, 0,
                                     offset, unescaped_length);
    }
    if (path[position] == '-' || (path[position] >= '0' && path[position] <= '9')) {
        sz_f64_t number;
        sz_size_t cursor = position;
        // RFC 9535 numbers are JSON's, whose leading zero takes no more digits.
        if (sz_subdocuments_parse_number_(path, length, &cursor, &number) == sz_false_k) return sz_invalid_syntax_k;
        parser->position = cursor, *primary = sz_subdocuments_primary_literal_k;
        return sz_subdocuments_emit_(parser, sz_subdocuments_operation_literal_k, sz_subdocuments_kind_number_k, number,
                                     0, 0);
    }

    for (word_end = position; word_end != length && path[word_end] >= 'a' && path[word_end] <= 'z';) ++word_end;
    {
        sz_cptr_t const word = path + position;
        sz_size_t const word_length = word_end - position;
        sz_subdocuments_kind_t const kind =
            word_length == 4 && word[0] == 't' && word[1] == 'r' && word[2] == 'u' && word[3] == 'e'
                ? sz_subdocuments_kind_true_k
            : word_length == 5 && word[0] == 'f' && word[1] == 'a' && word[2] == 'l' && word[3] == 's' && word[4] == 'e'
                ? sz_subdocuments_kind_false_k
            : word_length == 4 && word[0] == 'n' && word[1] == 'u' && word[2] == 'l' && word[3] == 'l'
                ? sz_subdocuments_kind_null_k
                : sz_subdocuments_kind_nothing_k;
        if (kind != sz_subdocuments_kind_nothing_k) {
            parser->position = word_end, *primary = sz_subdocuments_primary_literal_k;
            return sz_subdocuments_emit_(parser, sz_subdocuments_operation_literal_k, kind, 0, 0, 0);
        }
    }
    if (word_end == length || path[word_end] != '(' || parser->depth >= sz_subdocuments_filter_depth_k)
        return sz_invalid_syntax_k;

    {
        sz_size_t const name_length = word_end - position;
        sz_cptr_t const name = path + position;
        sz_bool_t const is_length = name_length == 6 && name[0] == 'l' && name[1] == 'e' && name[2] == 'n' &&
                                            name[3] == 'g' && name[4] == 't' && name[5] == 'h'
                                        ? sz_true_k
                                        : sz_false_k;
        sz_bool_t const is_count = name_length == 5 && name[0] == 'c' && name[1] == 'o' && name[2] == 'u' &&
                                           name[3] == 'n' && name[4] == 't'
                                       ? sz_true_k
                                       : sz_false_k;
        sz_bool_t const is_value = name_length == 5 && name[0] == 'v' && name[1] == 'a' && name[2] == 'l' &&
                                           name[3] == 'u' && name[4] == 'e'
                                       ? sz_true_k
                                       : sz_false_k;
        sz_bool_t const is_match = name_length == 5 && name[0] == 'm' && name[1] == 'a' && name[2] == 't' &&
                                           name[3] == 'c' && name[4] == 'h'
                                       ? sz_true_k
                                       : sz_false_k;
        sz_bool_t const is_search = name_length == 6 && name[0] == 's' && name[1] == 'e' && name[2] == 'a' &&
                                            name[3] == 'r' && name[4] == 'c' && name[5] == 'h'
                                        ? sz_true_k
                                        : sz_false_k;
        sz_subdocuments_primary_t argument;
        sz_size_t argument_step = 0;
        if (!is_length && !is_count && !is_value && !is_match && !is_search) return sz_invalid_syntax_k;
        ++parser->depth;
        parser->position = word_end + 1;
        status = sz_subdocuments_parse_primary_(parser, &argument, &argument_step);
        if (status != sz_success_k) return status;
        // `count()` and `value()` take any query, the rest a value: a literal, a singular query
        // or a value call.
        if (is_count || is_value) {
            if (argument != sz_subdocuments_primary_query_k && argument != sz_subdocuments_primary_singular_k)
                return sz_invalid_syntax_k;
            status = sz_subdocuments_emit_query_(
                parser, argument, is_count ? sz_subdocuments_operation_count_k : sz_subdocuments_operation_value_k,
                argument_step);
        }
        else {
            if (argument == sz_subdocuments_primary_query_k || argument == sz_subdocuments_primary_logical_k)
                return sz_invalid_syntax_k;
            status = sz_subdocuments_emit_query_(parser, argument, sz_subdocuments_operation_value_k, argument_step);
        }
        if (status != sz_success_k) return status;
        position = sz_subdocuments_skip_blanks_(path, length, parser->position);
        if (is_match || is_search) {
            sz_size_t end;
            sz_u32_t offset = 0, unescaped_length = 0;
            // The pattern must be a string literal, as the engine compiles every pattern once.
            if (position == length || path[position] != ',') return sz_invalid_syntax_k;
            position = sz_subdocuments_skip_blanks_(path, length, position + 1);
            if (position == length || (path[position] != '\'' && path[position] != '"')) return sz_invalid_syntax_k;
            end = sz_subdocuments_skip_literal_(path, length, position);
            if (end == 0) return sz_invalid_syntax_k;
            if (parser->nesting == sz_false_k) {
                status = sz_subdocuments_builder_string_(builder, path + position + 1, end - position - 2,
                                                         (sz_u8_t)path[position], &offset, &unescaped_length);
                if (status != sz_success_k) return status;
            }
            status = sz_subdocuments_emit_(
                parser, is_match ? sz_subdocuments_operation_match_k : sz_subdocuments_operation_search_k,
                parser->nesting == sz_false_k ? builder->patterns_count++ : 0, 0, offset, unescaped_length);
            if (status != sz_success_k) return status;
            position = sz_subdocuments_skip_blanks_(path, length, end);
            *primary = sz_subdocuments_primary_logical_k;
        }
        else if (is_length) {
            status = sz_subdocuments_emit_(parser, sz_subdocuments_operation_length_k, 0, 0, 0, 0);
            if (status != sz_success_k) return status;
            *primary = sz_subdocuments_primary_number_k;
        }
        else *primary = is_count ? sz_subdocuments_primary_number_k : sz_subdocuments_primary_value_k;
        if (position == length || path[position] != ')') return sz_invalid_syntax_k;
        parser->position = position + 1;
        --parser->depth;
        return sz_success_k;
    }
}

/** Parses a basic expression: a parenthesized one, a comparison, or a test, either of the last two
 *  negated by `!` only when it is a test. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_parse_basic_(sz_subdocuments_parser_t *parser) {
    sz_cptr_t const path = parser->path;
    sz_size_t const length = parser->length;
    sz_size_t position = sz_subdocuments_skip_blanks_(path, length, parser->position), final_step = 0;
    sz_bool_t negated = sz_false_k;
    sz_subdocuments_primary_t left, right;
    sz_status_t status;
    if (position != length && path[position] == '!')
        negated = sz_true_k, position = sz_subdocuments_skip_blanks_(path, length, position + 1);
    if (position != length && path[position] == '(') {
        if (parser->depth >= sz_subdocuments_filter_depth_k) return sz_invalid_syntax_k;
        ++parser->depth, parser->position = position + 1;
        status = sz_subdocuments_parse_or_(parser);
        if (status != sz_success_k) return status;
        position = sz_subdocuments_skip_blanks_(path, length, parser->position);
        if (position == length || path[position] != ')') return sz_invalid_syntax_k;
        --parser->depth, parser->position = position + 1;
        return negated ? sz_subdocuments_emit_(parser, sz_subdocuments_operation_not_k, 0, 0, 0, 0) : sz_success_k;
    }
    parser->position = position;
    status = sz_subdocuments_parse_primary_(parser, &left, &final_step);
    if (status != sz_success_k) return status;
    position = sz_subdocuments_skip_blanks_(path, length, parser->position);

    if (negated == sz_false_k && length - position >= 1 &&
        (path[position] == '<' || path[position] == '>' ||
         (length - position >= 2 && (path[position] == '=' || path[position] == '!') && path[position + 1] == '='))) {
        sz_subdocuments_comparison_t comparison;
        sz_size_t right_step = 0;
        if (path[position] == '=') comparison = sz_subdocuments_comparison_equal_k, position += 2;
        else if (path[position] == '!') comparison = sz_subdocuments_comparison_not_equal_k, position += 2;
        else if (length - position >= 2 && path[position + 1] == '=')
            comparison = path[position] == '<' ? sz_subdocuments_comparison_less_equal_k
                                               : sz_subdocuments_comparison_greater_equal_k,
            position += 2;
        else
            comparison = path[position] == '<' ? sz_subdocuments_comparison_less_k
                                               : sz_subdocuments_comparison_greater_k,
            ++position;
        // One side is a literal or a number, so no comparison needs two
        // document values' deep equality.
        if (left == sz_subdocuments_primary_query_k || left == sz_subdocuments_primary_logical_k)
            return sz_invalid_syntax_k;
        status = sz_subdocuments_emit_query_(parser, left, sz_subdocuments_operation_value_k, final_step);
        if (status != sz_success_k) return status;
        parser->position = position;
        status = sz_subdocuments_parse_primary_(parser, &right, &right_step);
        if (status != sz_success_k) return status;
        if (right == sz_subdocuments_primary_query_k || right == sz_subdocuments_primary_logical_k)
            return sz_invalid_syntax_k;
        if (left != sz_subdocuments_primary_literal_k && left != sz_subdocuments_primary_number_k &&
            right != sz_subdocuments_primary_literal_k && right != sz_subdocuments_primary_number_k)
            return sz_invalid_syntax_k;
        status = sz_subdocuments_emit_query_(parser, right, sz_subdocuments_operation_value_k, right_step);
        if (status != sz_success_k) return status;
        return sz_subdocuments_emit_(parser, sz_subdocuments_operation_compare_k, comparison, 0, 0, 0);
    }

    if (left == sz_subdocuments_primary_query_k || left == sz_subdocuments_primary_singular_k)
        status = sz_subdocuments_emit_query_(parser, left, sz_subdocuments_operation_exists_k, final_step);
    else if (left != sz_subdocuments_primary_logical_k) return sz_invalid_syntax_k;
    if (status != sz_success_k) return status;
    return negated ? sz_subdocuments_emit_(parser, sz_subdocuments_operation_not_k, 0, 0, 0, 0) : sz_success_k;
}

/** Parses a conjunction of basic expressions. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_parse_and_(sz_subdocuments_parser_t *parser) {
    sz_status_t status = sz_subdocuments_parse_basic_(parser);
    while (status == sz_success_k) {
        sz_size_t const position = sz_subdocuments_skip_blanks_(parser->path, parser->length, parser->position);
        if (parser->length - position < 2 || parser->path[position] != '&' || parser->path[position + 1] != '&') break;
        parser->position = position + 2;
        status = sz_subdocuments_parse_basic_(parser);
        if (status == sz_success_k) status = sz_subdocuments_emit_(parser, sz_subdocuments_operation_and_k, 0, 0, 0, 0);
    }
    return status;
}

/** Parses a disjunction of conjunctions: a whole logical expression. Every recursion of the filter
 *  grammar passes through here, so it stays out of line, as GCC 15 crashed inlining that cycle. */
STRINGZILLA_OUTLINED_ sz_status_t sz_subdocuments_parse_or_(sz_subdocuments_parser_t *parser) {
    sz_status_t status = sz_subdocuments_parse_and_(parser);
    while (status == sz_success_k) {
        sz_size_t const position = sz_subdocuments_skip_blanks_(parser->path, parser->length, parser->position);
        if (parser->length - position < 2 || parser->path[position] != '|' || parser->path[position + 1] != '|') break;
        parser->position = position + 2;
        status = sz_subdocuments_parse_and_(parser);
        if (status == sz_success_k) status = sz_subdocuments_emit_(parser, sz_subdocuments_operation_or_k, 0, 0, 0, 0);
    }
    return status;
}

/**
 *  @brief Compiles the filter selector whose expression starts at `parser->position`, for @p step.
 *
 *  The first pass emits the program and numbers the steps of its queries, which follow every step
 *  numbered so far; the second compiles the filters nested in those queries, so every program and
 *  every query's steps stay contiguous. Two filters of one bracketed selection are two filters of
 *  one step, whose results are joined when they run.
 */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_parse_filter_(sz_subdocuments_parser_t *parser, sz_size_t step) {
    sz_subdocuments_builder_t *const builder = parser->builder;
    sz_subdocuments_parser_t expression = *parser;
    sz_size_t const filter = builder->filters_count++;
    sz_status_t status;
    if (parser->depth >= sz_subdocuments_filter_depth_k) return sz_invalid_syntax_k;
    if (builder->filters)
        builder->filters[filter].step = (sz_u32_t)step,
        builder->filters[filter].instructions_begin = (sz_u32_t)builder->instructions_count;
    expression.depth = parser->depth + 1, expression.stack_depth = 0, expression.filter = filter;
    expression.nesting = sz_false_k, expression.next_root = builder->steps_count;
    status = sz_subdocuments_parse_or_(&expression);
    if (status != sz_success_k) return status;
    if (builder->filters) builder->filters[filter].instructions_end = (sz_u32_t)builder->instructions_count;

    expression.position = parser->position, expression.nesting = sz_true_k;
    status = sz_subdocuments_parse_or_(&expression);
    parser->position = expression.position;
    return status;
}

/** The end of a filter selector's expression starting at @p position: the first `,` or `]` outside
 *  parentheses, brackets and string literals, or @p length. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_skip_filter_(sz_cptr_t path, sz_size_t length, sz_size_t position) {
    sz_size_t depth = 0;
    while (position != length) {
        sz_u8_t const byte = (sz_u8_t)path[position];
        if (byte == '\'' || byte == '"') {
            position = sz_subdocuments_skip_literal_(path, length, position);
            if (position == 0) return length;
        }
        else if (depth == 0 && (byte == ',' || byte == ']')) return position;
        else depth += byte == '(' || byte == '[', depth -= depth && (byte == ')' || byte == ']'), ++position;
    }
    return length;
}

/**
 *  @brief Parses the segments after a path's `$` or a filter query's `@`, from `parser->position`
 *      up to blank space no segment follows.
 *
 *  Numbering, it adds one step per segment and skips filter bodies; @p filtering, it adds nothing
 *  and compiles each filter selector for its segment's step, `root + segment`. Two passes keep
 *  every path's own steps contiguous, whatever its filters hold.
 *
 *  @param[out] segments_count Segments read.
 *  @param[out] singular Whether every segment is a child segment of one name or one index.
 */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_parse_segments_(sz_subdocuments_parser_t *parser, sz_size_t root,
                                                               sz_bool_t filtering, sz_size_t *segments_count,
                                                               sz_bool_t *singular) {
    sz_subdocuments_builder_t *const builder = parser->builder;
    sz_cptr_t const path = parser->path;
    sz_size_t const length = parser->length;
    sz_size_t count = 0;
    sz_status_t status;
    *singular = sz_true_k;
    for (;;) {
        sz_bool_t descendant = sz_false_k;
        sz_size_t position = sz_subdocuments_skip_blanks_(path, length, parser->position), selectors = 0;
        if (position == length || (path[position] != '.' && path[position] != '[')) break;
        if (path[position] == '.') {
            ++position;
            if (position != length && path[position] == '.') descendant = sz_true_k, ++position;
            if (position == length) return sz_invalid_syntax_k;
            if (descendant == sz_false_k || path[position] != '[') {
                ++count;
                if (filtering == sz_false_k) sz_subdocuments_builder_step_(builder, descendant, parser->path_index);
                if (descendant == sz_true_k || path[position] == '*') *singular = sz_false_k;
                if (path[position] == '*') {
                    if (filtering == sz_false_k) sz_subdocuments_builder_wildcard_(builder);
                    parser->position = position + 1;
                    continue;
                }
                if (sz_subdocuments_is_name_first_((sz_u8_t)path[position]) == sz_false_k) return sz_invalid_syntax_k;
                {
                    sz_cptr_t const name = path + position;
                    while (position != length &&
                           (sz_subdocuments_is_name_first_((sz_u8_t)path[position]) == sz_true_k ||
                            (path[position] >= '0' && path[position] <= '9')))
                        ++position;
                    if (filtering == sz_false_k &&
                        sz_subdocuments_builder_name_(builder, name, (sz_size_t)(path + position - name), 0) !=
                            sz_success_k)
                        return sz_invalid_syntax_k;
                }
                parser->position = position;
                continue;
            }
        }

        ++count;
        if (filtering == sz_false_k) sz_subdocuments_builder_step_(builder, descendant, parser->path_index);
        for (++position;; ++position, ++selectors) {
            position = sz_subdocuments_skip_blanks_(path, length, position);
            if (position == length) return sz_invalid_syntax_k;
            if (path[position] == '\'' || path[position] == '"') {
                sz_size_t const end = sz_subdocuments_skip_literal_(path, length, position);
                if (end == 0) return sz_invalid_syntax_k;
                if (filtering == sz_false_k &&
                    sz_subdocuments_builder_name_(builder, path + position + 1, end - position - 2,
                                                  (sz_u8_t)path[position]) != sz_success_k)
                    return sz_invalid_syntax_k;
                position = end;
            }
            else if (path[position] == '*') {
                if (filtering == sz_false_k) sz_subdocuments_builder_wildcard_(builder);
                *singular = sz_false_k, ++position;
            }
            else if (path[position] == '?') {
                *singular = sz_false_k;
                if (filtering == sz_false_k) {
                    sz_subdocuments_builder_filter_(builder);
                    position = sz_subdocuments_skip_filter_(path, length, position + 1);
                }
                else {
                    parser->position = position + 1;
                    status = sz_subdocuments_parse_filter_(parser, root + count);
                    if (status != sz_success_k) return status;
                    position = parser->position;
                }
            }
            else {
                sz_subdocuments_slice_t slice;
                sz_bool_t index;
                if (sz_subdocuments_parse_slice_(path, length, &position, &slice, &index) == sz_false_k)
                    return sz_invalid_syntax_k;
                if (index == sz_false_k) *singular = sz_false_k;
                if (filtering == sz_false_k) sz_subdocuments_builder_slice_(builder, slice);
            }
            position = sz_subdocuments_skip_blanks_(path, length, position);
            if (position == length) return sz_invalid_syntax_k;
            if (path[position] == ']') break;
            if (path[position] != ',') return sz_invalid_syntax_k;
        }
        if (selectors != 0 || descendant == sz_true_k) *singular = sz_false_k;
        parser->position = position + 1;
    }
    *segments_count = count;
    return sz_success_k;
}

/**
 *  @brief Parses one RFC 9535 path into @p builder 's steps, one per segment, a bracketed segment's
 *      selectors all joining its step, then compiles its filters.
 *  @return @c sz_invalid_syntax_k for anything outside the grammar or the supported subset:
 *      absolute queries inside filters, comparisons of two document values, and patterns given
 *      otherwise than as a string literal.
 */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_parse_path_(sz_cptr_t path, sz_size_t length, sz_size_t path_index,
                                                           sz_subdocuments_builder_t *builder) {
    sz_subdocuments_parser_t parser;
    sz_size_t const root = builder->steps_count;
    sz_size_t segments_count;
    sz_bool_t singular;
    sz_status_t status;
    if (length == 0 || path[0] != '$') return sz_invalid_syntax_k;
    parser.builder = builder, parser.path = path, parser.length = length, parser.position = 1;
    parser.path_index = path_index, parser.depth = 0, parser.stack_depth = 0, parser.filter = 0;
    parser.nesting = sz_false_k, parser.next_root = 0;
    sz_subdocuments_builder_step_(builder, sz_false_k, path_index);
    if (builder->root_steps) builder->root_steps[root >> 6] |= (sz_u64_t)1 << (root & 63);
    status = sz_subdocuments_parse_segments_(&parser, root, sz_false_k, &segments_count, &singular);
    // Blank space only separates segments, so none may trail the last one.
    if (status != sz_success_k || parser.position != length) return sz_invalid_syntax_k;
    if (builder->final_steps)
        builder->final_steps[(root + segments_count) >> 6] |= (sz_u64_t)1 << ((root + segments_count) & 63);
    parser.position = 1;
    return sz_subdocuments_parse_segments_(&parser, root, sz_true_k, &segments_count, &singular);
}

/** Bytes an XML name may hold, as XPath name tests spell them: letters, digits, underscores, `-`,
 *  `.`, the `:` of a QName prefix, and every non-ASCII byte. */
STRINGZILLA_CONSTEXPR sz_bool_t sz_subdocuments_is_xml_name_(sz_u8_t byte) {
    sz_u8_t const lowered = (sz_u8_t)(byte | 0x20);
    return (lowered >= 'a' && lowered <= 'z') || (byte >= '0' && byte <= ':') || byte == '_' || byte == '-' ||
                   byte == '.' || byte >= 0x80
               ? sz_true_k
               : sz_false_k;
}

/**
 *  @brief Parses one XPath 1.0 location path along forward axes into @p builder 's steps: an
 *      absolute path of `/` and `//` steps, each a name test, `*`, `@name`, `@*` or `text()`, with
 *      an optional position `[n]`, and no blank space.
 *  @return @c sz_invalid_syntax_k for anything else: relative paths, reverse axes, other
 *      predicates, and a bare `/`, whose document node is no element, attribute or text to report.
 */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_parse_xpath_(sz_cptr_t path, sz_size_t length, sz_size_t path_index,
                                                            sz_subdocuments_builder_t *builder) {
    sz_size_t position = 0;
    sz_size_t const root = builder->steps_count;
    if (length == 0 || path[0] != '/') return sz_invalid_syntax_k;
    sz_subdocuments_builder_step_(builder, sz_false_k, path_index);
    if (builder->root_steps) builder->root_steps[root >> 6] |= (sz_u64_t)1 << (root & 63);
    while (position != length) {
        sz_bool_t descendant = sz_false_k, attribute = sz_false_k;
        if (path[position] != '/') return sz_invalid_syntax_k;
        if (++position != length && path[position] == '/') descendant = sz_true_k, ++position;
        if (position == length) return sz_invalid_syntax_k;
        sz_subdocuments_builder_step_(builder, descendant, path_index);
        if (path[position] == '@') attribute = sz_true_k, ++position, sz_subdocuments_builder_attribute_(builder);
        if (position != length && path[position] == '*') sz_subdocuments_builder_wildcard_(builder), ++position;
        else {
            sz_cptr_t const name = path + position;
            sz_u8_t const first = position != length ? (sz_u8_t)path[position] : 0;
            if (first == 0 || (first >= '0' && first <= '9') || first == '-' || first == '.' || first == ':' ||
                sz_subdocuments_is_xml_name_(first) == sz_false_k)
                return sz_invalid_syntax_k;
            sz_size_t colons = 0;
            for (; position != length && sz_subdocuments_is_xml_name_((sz_u8_t)path[position]) == sz_true_k; ++position)
                colons += path[position] == ':';
            // A QName has at most one colon, inside it, which also keeps out axes like `child::`.
            if (colons > 1 || path[position - 1] == ':') return sz_invalid_syntax_k;
            if (attribute == sz_false_k && length - position >= 2 && path[position] == '(' &&
                path[position + 1] == ')' && path + position - name == 4 && name[0] == 't' && name[1] == 'e' &&
                name[2] == 'x' && name[3] == 't')
                sz_subdocuments_builder_text_(builder), position += 2;
            else if (sz_subdocuments_builder_name_(builder, name, (sz_size_t)(path + position - name), 0) !=
                     sz_success_k)
                return sz_invalid_syntax_k;
        }
        if (position != length && path[position] == '[') {
            sz_subdocuments_slice_t slice;
            sz_i64_t index;
            ++position;
            if (sz_subdocuments_parse_integer_(path, length, &position, &index) == sz_false_k || index < 0 ||
                position == length || path[position] != ']')
                return sz_invalid_syntax_k;
            ++position;
            // XPath counts from one, and `[0]` selects nothing, as a zero step does.
            slice.begin = index - 1, slice.end = index, slice.step = index != 0;
            sz_subdocuments_builder_slice_(builder, slice);
        }
    }
    if (builder->final_steps)
        builder->final_steps[(builder->steps_count - 1) >> 6] |= (sz_u64_t)1 << ((builder->steps_count - 1) & 63);
    return sz_success_k;
}

/**
 *  @brief Writes I-Regexp @p pattern in the regex family's syntax behind the tag of pattern
 *      @p index, a @p whole one matching all of a tested string and a partial one any part of it.
 *
 *  I-Regexp's `.` excludes line feeds and carriage returns; its `^` and `$` are plain characters.
 *
 *  @return Bytes written, at most `7 * length + 48`.
 */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_translate_pattern_(sz_cptr_t pattern, sz_size_t length, sz_size_t index,
                                                                sz_bool_t whole, sz_ptr_t target) {
    static char const prefix[] = "\\A\\x01", separator[] = "\\x02", partial[] = "(?s:.)*?", dot[] = "[^\\n\\r]";
    sz_size_t written = 0, read, power = 1;
    sz_bool_t in_class = sz_false_k;
    for (read = 0; prefix[read]; ++read) target[written++] = prefix[read];
    for (read = index; read >= 10; read /= 10) power *= 10;
    for (; power; power /= 10) target[written++] = (char)('0' + index / power % 10);
    for (read = 0; separator[read]; ++read) target[written++] = separator[read];
    if (whole == sz_false_k)
        for (read = 0; partial[read]; ++read) target[written++] = partial[read];
    target[written++] = '(', target[written++] = '?', target[written++] = ':';
    for (read = 0; read != length; ++read) {
        char const byte = pattern[read];
        if (byte == '\\' && read + 1 != length) {
            target[written++] = byte, target[written++] = pattern[++read];
            continue;
        }
        if (in_class == sz_true_k || byte == '[') in_class = byte != ']' ? sz_true_k : sz_false_k;
        else if (byte == '.') {
            for (sz_size_t copied = 0; dot[copied]; ++copied) target[written++] = dot[copied];
            continue;
        }
        else if (byte == '^' || byte == '$') target[written++] = '\\';
        target[written++] = byte;
    }
    target[written++] = ')';
    if (whole == sz_true_k) target[written++] = '\\', target[written++] = 'z';
    return written;
}

/**
 *  @brief Compiles the pattern of every `match()` and `search()` of @p instructions into @p regex,
 *      each behind its tag, for the capabilities in @p capabilities.
 *  @param[out] malformed_pattern The refused pattern, when @c sz_invalid_syntax_k is returned.
 */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_compile_patterns_(sz_regex_engine_t *regex,
                                                                 sz_subdocuments_instruction_t const *instructions,
                                                                 sz_size_t instructions_count, sz_cptr_t strings_data,
                                                                 sz_size_t patterns_count, sz_capability_t capabilities,
                                                                 sz_allocator_t *allocator, sz_stream_t stream,
                                                                 sz_size_t *malformed_pattern) {
    sz_size_t bytes = (patterns_count + 1) * sizeof(sz_u32_t), written = 0, index;
    sz_strings_view_t view;
    sz_u32_t *offsets;
    sz_ptr_t data;
    sz_status_t status;
    for (index = 0; index != instructions_count; ++index)
        if (instructions[index].operation == sz_subdocuments_operation_match_k ||
            instructions[index].operation == sz_subdocuments_operation_search_k)
            bytes += 7 * instructions[index].string_length + 48;
    if (bytes > 0xFFFFFFFFull) return sz_overflow_risk_k;
    offsets = (sz_u32_t *)allocator->allocate(bytes, allocator->handle, stream);
    if (!offsets) return sz_bad_alloc_k;
    data = (sz_ptr_t)(offsets + patterns_count + 1);
    offsets[0] = 0;
    for (index = 0; index != instructions_count; ++index) {
        sz_subdocuments_instruction_t const *const instruction = &instructions[index];
        if (instruction->operation != sz_subdocuments_operation_match_k &&
            instruction->operation != sz_subdocuments_operation_search_k)
            continue;
        written += sz_subdocuments_translate_pattern_(
            strings_data + instruction->string_offset, instruction->string_length, instruction->operand,
            instruction->operation == sz_subdocuments_operation_match_k ? sz_true_k : sz_false_k, data + written);
        offsets[instruction->operand + 1] = (sz_u32_t)written;
    }
    view.layout = sz_strings_u32_tape_k, view.count = patterns_count;
    view.data.u32_tape.data = data, view.data.u32_tape.offsets = offsets;
    status = sz_regex_engine_init(regex, &view, sz_cased_k, malformed_pattern, capabilities, allocator, stream);
    allocator->free(offsets, bytes, allocator->handle, stream);
    return status;
}

/** Zeroes @p engine 's arrays and returns both its blocks to the allocator that built them. */
STRINGZILLA_INLINE void sz_subdocuments_engine_free_(sz_subdocuments_engine_t *engine, sz_stream_t stream) {
    sz_allocator_t *const allocator = &engine->allocator;
    if (engine->memory) allocator->free(engine->memory, engine->memory_bytes, allocator->handle, stream);
    if (engine->scratch) allocator->free(engine->scratch, engine->scratch_bytes, allocator->handle, stream);
    if (engine->patterns_count) sz_regex_engine_free(&engine->regex, stream);
    engine->memory = STRINGZILLA_NULL, engine->memory_bytes = 0;
    engine->scratch = STRINGZILLA_NULL, engine->scratch_bytes = 0;
    engine->names_data = STRINGZILLA_NULL, engine->names_offsets = STRINGZILLA_NULL;
    engine->names_slots = STRINGZILLA_NULL, engine->names_steps = STRINGZILLA_NULL;
    engine->wildcard_steps = STRINGZILLA_NULL, engine->descendant_steps = STRINGZILLA_NULL;
    engine->relative_steps = STRINGZILLA_NULL, engine->attribute_steps = STRINGZILLA_NULL;
    engine->text_steps = STRINGZILLA_NULL, engine->filter_steps = STRINGZILLA_NULL;
    engine->filters_roots = STRINGZILLA_NULL, engine->filters = STRINGZILLA_NULL;
    engine->instructions = STRINGZILLA_NULL, engine->strings_data = STRINGZILLA_NULL;
    engine->final_steps = STRINGZILLA_NULL, engine->root_steps = STRINGZILLA_NULL;
    engine->slices = STRINGZILLA_NULL, engine->slices_offsets = STRINGZILLA_NULL;
    engine->steps_paths = STRINGZILLA_NULL;
    engine->steps_count = 0, engine->words_count = 0, engine->names_slots_count = 0, engine->paths_count = 0;
    engine->filters_count = 0, engine->patterns_count = 0, engine->capability = 0;
}

/**
 *  @brief Compiles @p paths into one block, parsing them twice: once to size it, once to fill it.
 *
 *  The block is laid out widest element first: the step bitsets, the slices, the 32-bit arrays,
 *  and the name bytes last, so no array needs padding, over whole pages of
 *  @c sz_subdocuments_page_bytes_k from the block's first page boundary on.
 */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_engine_compile_(sz_subdocuments_engine_t *engine,
                                                               sz_strings_view_t const *paths,
                                                               sz_subdocuments_format_t format,
                                                               sz_size_t *malformed_index, sz_capability_t capability,
                                                               sz_allocator_t const *requested_allocator,
                                                               sz_stream_t stream) {
    sz_subdocuments_builder_t builder;
    sz_allocator_t allocator;
    sz_string_view_t views[64];
    sz_size_t path_index, names_bound, steps_bound, slices_bound, filters_bound, instructions_bound, bitset_words;
    sz_size_t total_bytes, page_bytes, index, filter;
    sz_regex_engine_t regex;
    sz_status_t status;
    sz_u8_t *block;
    sz_status_t (*const parse)(sz_cptr_t, sz_size_t, sz_size_t, sz_subdocuments_builder_t *) =
        format == sz_subdocuments_xml_k ? &sz_subdocuments_parse_xpath_ : &sz_subdocuments_parse_path_;

    if (format != sz_subdocuments_json_k && format != sz_subdocuments_xml_k) return sz_missing_kernel_k;
    if (paths->count == 0) return sz_unexpected_dimensions_k;
    if (requested_allocator) allocator = *requested_allocator;
    else if ((status = sz_allocator_init_heap(&allocator)) != sz_success_k) return status;

    builder.steps_count = 0, builder.name_selectors_count = 0, builder.slices_count = 0, builder.names_count = 0;
    builder.names_bytes = 0, builder.filters_count = 0, builder.instructions_count = 0, builder.patterns_count = 0;
    builder.strings_bytes = 0, builder.words_count = 0, builder.names_slots_count = 0;
    builder.names_data = STRINGZILLA_NULL, builder.names_offsets = STRINGZILLA_NULL;
    builder.names_slots = STRINGZILLA_NULL, builder.names_steps = STRINGZILLA_NULL;
    builder.wildcard_steps = STRINGZILLA_NULL, builder.descendant_steps = STRINGZILLA_NULL;
    builder.relative_steps = STRINGZILLA_NULL, builder.attribute_steps = STRINGZILLA_NULL;
    builder.text_steps = STRINGZILLA_NULL, builder.filter_steps = STRINGZILLA_NULL;
    builder.filters_roots = STRINGZILLA_NULL, builder.final_steps = STRINGZILLA_NULL;
    builder.root_steps = STRINGZILLA_NULL;
    builder.filters = STRINGZILLA_NULL, builder.instructions = STRINGZILLA_NULL,
    builder.strings_data = STRINGZILLA_NULL;
    builder.slices = STRINGZILLA_NULL, builder.slices_offsets = STRINGZILLA_NULL,
    builder.steps_paths = STRINGZILLA_NULL;
    for (path_index = 0; path_index != paths->count; ++path_index) {
        if (path_index % 64 == 0) sz_strings_get_block_(paths, path_index, views);
        status = parse(views[path_index % 64].start, views[path_index % 64].length, path_index, &builder);
        if (status != sz_success_k) {
            *malformed_index = path_index;
            return status;
        }
    }
    if (builder.steps_count > 0xFFFFFFFFull || builder.slices_count > 0xFFFFFFFFull ||
        builder.names_bytes > 0xFFFFFFFFull || builder.instructions_count > 0xFFFFFFFFull ||
        builder.strings_bytes > 0xFFFFFFFFull)
        return sz_overflow_risk_k;

    names_bound = builder.name_selectors_count, steps_bound = builder.steps_count, slices_bound = builder.slices_count;
    filters_bound = builder.filters_count, instructions_bound = builder.instructions_count;
    builder.words_count = sz_size_divide_round_up(steps_bound, 64);
    builder.names_slots_count = names_bound ? sz_size_bit_ceil(names_bound * 2) : 0;
    bitset_words = (names_bound + 8 + filters_bound) * builder.words_count;
    total_bytes = bitset_words * sizeof(sz_u64_t) + slices_bound * sizeof(sz_subdocuments_slice_t) +
                  instructions_bound * sizeof(sz_subdocuments_instruction_t) +
                  (2 * steps_bound + 1 + names_bound + 1 + builder.names_slots_count) * sizeof(sz_u32_t) +
                  filters_bound * sizeof(sz_subdocuments_filter_t) + builder.names_bytes + builder.strings_bytes +
                  sizeof(sz_u64_t); // A name's last word is read whole
    page_bytes = sz_subdocuments_page_bytes_k;
    total_bytes = (sz_size_divide_round_up(total_bytes, page_bytes) + 1) * page_bytes;
    block = (sz_u8_t *)allocator.allocate(total_bytes, allocator.handle, stream);
    if (!block) return sz_bad_alloc_k;

    builder.names_steps = (sz_u64_t *)(block + (page_bytes - (sz_size_t)block % page_bytes) % page_bytes);
    builder.wildcard_steps = builder.names_steps + names_bound * builder.words_count;
    builder.descendant_steps = builder.wildcard_steps + builder.words_count;
    builder.relative_steps = builder.descendant_steps + builder.words_count;
    builder.attribute_steps = builder.relative_steps + builder.words_count;
    builder.text_steps = builder.attribute_steps + builder.words_count;
    builder.filter_steps = builder.text_steps + builder.words_count;
    builder.final_steps = builder.filter_steps + builder.words_count;
    builder.root_steps = builder.final_steps + builder.words_count;
    builder.filters_roots = builder.root_steps + builder.words_count;
    builder.slices = (sz_subdocuments_slice_t *)(builder.filters_roots + filters_bound * builder.words_count);
    builder.instructions = (sz_subdocuments_instruction_t *)(builder.slices + slices_bound);
    builder.steps_paths = (sz_u32_t *)(builder.instructions + instructions_bound);
    builder.slices_offsets = builder.steps_paths + steps_bound;
    builder.names_offsets = builder.slices_offsets + steps_bound + 1;
    builder.names_slots = builder.names_offsets + names_bound + 1;
    builder.filters = (sz_subdocuments_filter_t *)(builder.names_slots + builder.names_slots_count);
    builder.names_data = (sz_ptr_t)(builder.filters + filters_bound);
    builder.strings_data = builder.names_data + builder.names_bytes;
    for (index = 0; index != bitset_words; ++index) builder.names_steps[index] = 0;
    for (index = 0; index != builder.names_slots_count; ++index) builder.names_slots[index] = sz_subdocuments_no_name_k;
    builder.names_offsets[0] = 0, builder.slices_offsets[0] = 0;
    builder.steps_count = 0, builder.name_selectors_count = 0, builder.slices_count = 0, builder.names_bytes = 0;
    builder.filters_count = 0, builder.instructions_count = 0, builder.patterns_count = 0, builder.strings_bytes = 0;
    for (path_index = 0; path_index != paths->count; ++path_index) {
        if (path_index % 64 == 0) sz_strings_get_block_(paths, path_index, views);
        parse(views[path_index % 64].start, views[path_index % 64].length, path_index, &builder);
    }

    if (builder.patterns_count) {
        sz_size_t malformed_pattern = 0;
        status = sz_subdocuments_compile_patterns_(
            &regex, builder.instructions, builder.instructions_count, builder.strings_data, builder.patterns_count,
            capability | (capability - 1), &allocator, stream, &malformed_pattern);
        if (status != sz_success_k) {
            // Blame the path whose filter spells the refused pattern.
            for (index = 0; status == sz_invalid_syntax_k && index != builder.instructions_count; ++index)
                if ((builder.instructions[index].operation == sz_subdocuments_operation_match_k ||
                     builder.instructions[index].operation == sz_subdocuments_operation_search_k) &&
                    builder.instructions[index].operand == malformed_pattern)
                    break;
            for (filter = 0; status == sz_invalid_syntax_k && filter != builder.filters_count; ++filter)
                if (builder.filters[filter].instructions_end > index) {
                    *malformed_index = builder.steps_paths[builder.filters[filter].step];
                    break;
                }
            allocator.free(block, total_bytes, allocator.handle, stream);
            return status;
        }
        engine->regex = regex;
    }

    engine->names_data = builder.names_data;
    engine->names_offsets = builder.names_offsets;
    engine->names_slots = builder.names_slots;
    engine->names_steps = builder.names_steps;
    engine->wildcard_steps = builder.wildcard_steps;
    engine->descendant_steps = builder.descendant_steps;
    engine->relative_steps = builder.relative_steps;
    engine->attribute_steps = builder.attribute_steps;
    engine->text_steps = builder.text_steps;
    engine->filter_steps = builder.filter_steps;
    engine->filters_roots = builder.filters_roots;
    engine->filters = builder.filters;
    engine->instructions = builder.instructions;
    engine->strings_data = builder.strings_data;
    engine->final_steps = builder.final_steps;
    engine->root_steps = builder.root_steps;
    engine->slices = builder.slices;
    engine->slices_offsets = builder.slices_offsets;
    engine->steps_paths = builder.steps_paths;
    engine->steps_count = (sz_u32_t)steps_bound;
    engine->words_count = (sz_u32_t)builder.words_count;
    engine->names_slots_count = (sz_u32_t)builder.names_slots_count;
    engine->paths_count = (sz_u32_t)paths->count;
    engine->filters_count = (sz_u32_t)filters_bound;
    engine->patterns_count = (sz_u32_t)builder.patterns_count;
    engine->format = format;
    engine->capability = capability;
    engine->allocator = allocator;
    engine->memory = block;
    engine->memory_bytes = total_bytes;
    engine->scratch = STRINGZILLA_NULL;
    engine->scratch_bytes = 0;
    return sz_success_k;
}

#pragma endregion Paths

#pragma region Blocks

/** The masks a tier's classifier makes for one 64-byte block, bit @c i standing for byte @c i. */
typedef struct sz_subdocuments_block_t {

    /** `{` and `[` outside strings. */
    sz_u64_t opens;

    /** `}` and `]` outside strings. */
    sz_u64_t closes;

    /** Every byte the walk stops at: brackets, colons and commas outside strings, unescaped quotes,
     *  and the first byte of every scalar. */
    sz_u64_t structurals;

    /** Bytes of scalars: outside strings, and neither blank, a quote, a bracket, a colon nor a
     *  comma. */
    sz_u64_t scalars;

    /** Backslashes, inside strings or not, so a key the block holds is known to need no
     *  unescaping. */
    sz_u64_t backslashes;
} sz_subdocuments_block_t;

/** What one block's classification hands the next. */
typedef struct sz_subdocuments_carry_t {

    /** One when the block ended in an odd run of backslashes, escaping the
     *  next block's first byte. */
    sz_u64_t escaped;

    /** All ones when the block ended inside a string. */
    sz_u64_t in_string;

    /** One when the block ended inside a scalar, so the next block's first byte starts none. */
    sz_u64_t scalar;

    /** Nonzero once any block held a `/` outside strings, which no JSON document does. */
    sz_u64_t slashes;
} sz_subdocuments_carry_t;

/** Class bits by low nibble, for the SIMD tiers' classifiers, which AND it with the entry of the
 *  high nibble: `{[` 0x01, `}]` 0x02, `:` 0x04, `,` 0x08, `"` 0x10, `\` 0x20, space 0x40, and tab,
 *  line feed and carriage return 0x80. Every class is a product of nibble sets, so no other byte
 *  gets a bit, while `/` alone gets both the comma's and the space's, which the settle tests. */
static sz_u8_t const sz_subdocuments_low_nibble_classes_k[16] = {0x40, 0,    0x10, 0,    0,    0,    0, 0,
                                                                 0,    0x80, 0x84, 0x01, 0x28, 0x82, 0, 0x48};

/** Class bits by high nibble, the other half of @c sz_subdocuments_low_nibble_classes_k. */
static sz_u8_t const sz_subdocuments_high_nibble_classes_k[16] = {0x80, 0, 0x58, 0x04, 0, 0x23, 0, 0x03,
                                                                  0,    0, 0,    0,    0, 0,    0, 0};

/** Signature of a tier's classifier: 64 readable bytes in, their masks out. A tier's skim shares
 *  it, setting only the brackets, and a tier without one passes its classifier as the skim. */
typedef sz_subdocuments_block_t (*sz_subdocuments_classify_t)(sz_u8_t const *, sz_subdocuments_carry_t *);

/** Each bit of @p bits XORed with every bit below it, which turns unescaped quotes into the bytes
 *  inside strings; the tiers with a carry-less multiply replace it. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_subdocuments_prefix_xor_serial_(sz_u64_t bits) {
    bits ^= bits << 1, bits ^= bits << 2, bits ^= bits << 4;
    bits ^= bits << 8, bits ^= bits << 16, bits ^= bits << 32;
    return bits;
}

/** The bytes of a block inside strings, from its raw @p quotes, which lose the escaped ones, and
 *  @p backslashes, through @p prefix_xor; updates the escape and string carries. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_in_string_(sz_u64_t *quotes, sz_u64_t backslashes,
                                                       sz_u64_t (*prefix_xor)(sz_u64_t),
                                                       sz_subdocuments_carry_t *carry) {
    sz_u64_t escaped, in_string;
    // A byte is escaped when an odd run of backslashes precedes it, simdjson's rule.
    if (!backslashes) escaped = carry->escaped, carry->escaped = 0;
    else {
        sz_u64_t const odd_bits = 0xAAAAAAAAAAAAAAAAull;
        sz_u64_t const potential_escape = backslashes & ~carry->escaped;
        sz_u64_t const maybe_escaped = potential_escape << 1;
        sz_u64_t const escape_and_terminal = ((maybe_escaped | odd_bits) - potential_escape) ^ odd_bits;
        escaped = escape_and_terminal ^ (backslashes | carry->escaped);
        carry->escaped = (escape_and_terminal & backslashes) >> 63;
    }
    *quotes &= ~escaped;
    in_string = prefix_xor(*quotes) ^ carry->in_string;
    carry->in_string = 0 - (in_string >> 63);
    return in_string;
}

/** Turns a block's raw byte masks into the walk's: drops escaped quotes, clears everything inside
 *  strings with @p prefix_xor, and marks the first byte of every scalar. A `/` arrives in both
 *  @p separators and @p blanks, the one byte that does, and the carry records those outside
 *  strings, which fail the document however the walk reads them. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_block_settle_with_(sz_u64_t quotes, sz_u64_t backslashes,
                                                                              sz_u64_t opens, sz_u64_t closes,
                                                                              sz_u64_t separators, sz_u64_t blanks,
                                                                              sz_u64_t (*prefix_xor)(sz_u64_t),
                                                                              sz_subdocuments_carry_t *carry) {
    sz_u64_t const in_string = sz_subdocuments_in_string_(&quotes, backslashes, prefix_xor, carry);
    sz_u64_t scalars;
    sz_subdocuments_block_t block;
    block.backslashes = backslashes;
    carry->slashes |= separators & blanks & ~in_string;

    opens &= ~in_string, closes &= ~in_string, separators &= ~in_string;
    scalars = ~(in_string | quotes | blanks | opens | closes | separators);
    block.opens = opens, block.closes = closes;
    block.structurals = opens | closes | separators | quotes | (scalars & ~(scalars << 1 | carry->scalar));
    block.scalars = scalars;
    carry->scalar = scalars >> 63;
    return block;
}

/** @ref sz_subdocuments_block_settle_with_ over the shift ladder, where every tier without a
 *  carry-less multiply ends its classifier. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_block_settle_(sz_u64_t quotes, sz_u64_t backslashes,
                                                                         sz_u64_t opens, sz_u64_t closes,
                                                                         sz_u64_t separators, sz_u64_t blanks,
                                                                         sz_subdocuments_carry_t *carry) {
    return sz_subdocuments_block_settle_with_(quotes, backslashes, opens, closes, separators, blanks,
                                              &sz_subdocuments_prefix_xor_serial_, carry);
}

/** One bit per byte of @p top_bits, set where that byte's top bit is, as a movemask would. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_subdocuments_gather_bytes_serial_(sz_u64_t top_bits) {
    return ((top_bits >> 7) * 0x0102040810204080ull) >> 56;
}

/** Classifies a block eight bytes at a time with SWAR compares, as every tier's masks look. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_serial_(sz_u8_t const *text,
                                                                            sz_subdocuments_carry_t *carry) {
    sz_u64_t quotes = 0, backslashes = 0, opens = 0, closes = 0, separators = 0, blanks = 0;
    sz_u64_vec_t quote_vec, backslash_vec, open_vec, close_vec, colon_vec, comma_vec, space_vec, tab_vec, line_feed_vec,
        carriage_return_vec, slash_vec;
    quote_vec.u64 = 0x2222222222222222ull, backslash_vec.u64 = 0x5C5C5C5C5C5C5C5Cull;
    open_vec.u64 = 0x7B7B7B7B7B7B7B7Bull, close_vec.u64 = 0x7D7D7D7D7D7D7D7Dull;
    colon_vec.u64 = 0x3A3A3A3A3A3A3A3Aull, comma_vec.u64 = 0x2C2C2C2C2C2C2C2Cull;
    space_vec.u64 = 0x2020202020202020ull, tab_vec.u64 = 0x0909090909090909ull;
    line_feed_vec.u64 = 0x0A0A0A0A0A0A0A0Aull, carriage_return_vec.u64 = 0x0D0D0D0D0D0D0D0Dull;
    slash_vec.u64 = 0x2F2F2F2F2F2F2F2Full;
    for (sz_size_t word_index = 0; word_index != 8; ++word_index) {
        sz_u64_vec_t text_vec = sz_u64_load((sz_cptr_t)text + word_index * 8), folded_vec;
        sz_size_t const shift = word_index * 8;
        sz_u64_t const slash_bits = sz_u64_each_byte_equal_(text_vec, slash_vec).u64;
        // `[` and `]` differ from `{` and `}` only in the 0x20 bit, and no other byte folds there.
        folded_vec.u64 = text_vec.u64 | 0x2020202020202020ull;
        quotes |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(text_vec, quote_vec).u64) << shift;
        backslashes |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(text_vec, backslash_vec).u64)
                       << shift;
        opens |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(folded_vec, open_vec).u64) << shift;
        closes |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(folded_vec, close_vec).u64) << shift;
        separators |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(text_vec, colon_vec).u64 |
                                                           sz_u64_each_byte_equal_(text_vec, comma_vec).u64 |
                                                           slash_bits)
                      << shift;
        blanks |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(text_vec, space_vec).u64 |
                                                       sz_u64_each_byte_equal_(text_vec, tab_vec).u64 |
                                                       sz_u64_each_byte_equal_(text_vec, line_feed_vec).u64 |
                                                       sz_u64_each_byte_equal_(text_vec, carriage_return_vec).u64 |
                                                       slash_bits)
                  << shift;
    }
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, separators, blanks, carry);
}

/**
 *  @brief Turns a block's raw quotes, backslashes, brackets and slashes into the brackets outside
 *      strings, all a skipped container's inner blocks need, through @p prefix_xor.
 *
 *  The carries are those the full settle leaves, but for the scalar one, which is cleared: it only
 *  marks the first byte of the next block, and a skip classifies again, in full, the block it ends
 *  in, clearing its structurals up to the closing bracket.
 */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_brackets_settle_with_(sz_u64_t quotes, sz_u64_t backslashes,
                                                                                 sz_u64_t opens, sz_u64_t closes,
                                                                                 sz_u64_t slashes,
                                                                                 sz_u64_t (*prefix_xor)(sz_u64_t),
                                                                                 sz_subdocuments_carry_t *carry) {
    sz_u64_t const in_string = sz_subdocuments_in_string_(&quotes, backslashes, prefix_xor, carry);
    sz_subdocuments_block_t block;
    carry->scalar = 0;
    carry->slashes |= slashes & ~in_string;
    block.opens = opens & ~in_string, block.closes = closes & ~in_string;
    block.structurals = 0, block.scalars = 0, block.backslashes = 0;
    return block;
}

/** The brackets outside strings of a block a skip passes whole, as
 *  @ref sz_subdocuments_classify_serial_ finds them, with five compares per word where the full
 *  classifier takes eleven. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_skim_serial_(sz_u8_t const *text,
                                                                        sz_subdocuments_carry_t *carry) {
    sz_u64_t quotes = 0, backslashes = 0, opens = 0, closes = 0, slashes = 0;
    sz_u64_vec_t quote_vec, backslash_vec, open_vec, close_vec, slash_vec;
    quote_vec.u64 = 0x2222222222222222ull, backslash_vec.u64 = 0x5C5C5C5C5C5C5C5Cull;
    open_vec.u64 = 0x7B7B7B7B7B7B7B7Bull, close_vec.u64 = 0x7D7D7D7D7D7D7D7Dull;
    slash_vec.u64 = 0x2F2F2F2F2F2F2F2Full;
    for (sz_size_t word_index = 0; word_index != 8; ++word_index) {
        sz_u64_vec_t text_vec = sz_u64_load((sz_cptr_t)text + word_index * 8), folded_vec;
        sz_size_t const shift = word_index * 8;
        folded_vec.u64 = text_vec.u64 | 0x2020202020202020ull;
        quotes |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(text_vec, quote_vec).u64) << shift;
        backslashes |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(text_vec, backslash_vec).u64)
                       << shift;
        opens |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(folded_vec, open_vec).u64) << shift;
        closes |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(folded_vec, close_vec).u64) << shift;
        slashes |= sz_subdocuments_gather_bytes_serial_(sz_u64_each_byte_equal_(text_vec, slash_vec).u64) << shift;
    }
    return sz_subdocuments_brackets_settle_with_(quotes, backslashes, opens, closes, slashes,
                                                 &sz_subdocuments_prefix_xor_serial_, carry);
}

#pragma endregion Blocks

#pragma region Walk

/** A document's position in the walk: the classified block and the structurals left in it. */
typedef struct sz_subdocuments_cursor_t {

    /** The document's bytes. */
    sz_u8_t const *text;

    /** Bytes the document holds. */
    sz_size_t length;

    /** Where the classified block starts. */
    sz_size_t block_offset;

    /** The classified block's masks. */
    sz_subdocuments_block_t block;

    /** Structurals of that block the walk has not visited yet. */
    sz_u64_t pending;

    /** What the next block's classification inherits. */
    sz_subdocuments_carry_t carry;
} sz_subdocuments_cursor_t;

/** Classifies the block at @p offset, padding a partial last one with blanks. */
STRINGZILLA_INLINE void sz_subdocuments_cursor_load_(sz_subdocuments_cursor_t *cursor,
                                                     sz_subdocuments_classify_t classify, sz_size_t offset) {
    sz_size_t const available = cursor->length - offset;
    cursor->block_offset = offset;
    if (available >= 64) cursor->block = classify(cursor->text + offset, &cursor->carry);
    else {
        sz_u8_t padded[64];
        sz_size_t index;
        for (index = 0; index != available; ++index) padded[index] = cursor->text[offset + index];
        for (; index != 64; ++index) padded[index] = ' ';
        cursor->block = classify(padded, &cursor->carry);
    }
    cursor->pending = cursor->block.structurals;
}

/** Moves to the next structural byte, or reports the end of the document. */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_cursor_next_(sz_subdocuments_cursor_t *cursor,
                                                          sz_subdocuments_classify_t classify, sz_size_t *position) {
    while (!cursor->pending) {
        if (cursor->length - cursor->block_offset <= 64) return sz_false_k;
        sz_subdocuments_cursor_load_(cursor, classify, cursor->block_offset + 64);
    }
    *position = cursor->block_offset + (sz_size_t)sz_u64_ctz(cursor->pending);
    cursor->pending &= cursor->pending - 1;
    return sz_true_k;
}

/**
 *  @brief Moves past the bracket closing the container just entered, skipping what it holds.
 *
 *  A block whose closes cannot outnumber the open brackets only shifts the depth by its two
 *  popcounts; only the block where the depth may reach zero is walked bit by bit. The whole blocks
 *  before it are only skimmed for their brackets, and the one the depth may reach zero in is
 *  classified again, in full, from the carries it started with. Brackets are counted by kind-blind
 *  depth, so a skipped subtree is validated no further than its strings and slashes.
 */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_cursor_skip_(sz_subdocuments_cursor_t *cursor,
                                                          sz_subdocuments_classify_t classify,
                                                          sz_subdocuments_classify_t skim, sz_size_t *position) {
    sz_size_t depth = 1;
    for (;;) {
        sz_u64_t const opens = cursor->block.opens & cursor->pending, closes = cursor->block.closes & cursor->pending;
        sz_size_t const closes_count = (sz_size_t)sz_u64_popcount(closes);
        if (depth > closes_count) {
            depth = depth - closes_count + (sz_size_t)sz_u64_popcount(opens);
            while (cursor->length - cursor->block_offset >= 128) {
                sz_subdocuments_carry_t const carry = cursor->carry;
                sz_subdocuments_block_t const brackets = skim(cursor->text + cursor->block_offset + 64, &cursor->carry);
                sz_size_t const skimmed_closes = (sz_size_t)sz_u64_popcount(brackets.closes);
                if (depth <= skimmed_closes) {
                    cursor->carry = carry;
                    break;
                }
                depth = depth - skimmed_closes + (sz_size_t)sz_u64_popcount(brackets.opens);
                cursor->block_offset += 64;
            }
        }
        else
            for (sz_u64_t brackets = opens | closes; brackets;) {
                sz_u64_t const bit = brackets & (0 - brackets);
                brackets ^= bit;
                if (opens & bit) ++depth;
                else if (--depth == 0) {
                    *position = cursor->block_offset + (sz_size_t)sz_u64_ctz(bit);
                    cursor->pending &= ~(bit | (bit - 1));
                    return sz_true_k;
                }
            }
        if (cursor->length - cursor->block_offset <= 64) return sz_false_k;
        sz_subdocuments_cursor_load_(cursor, classify, cursor->block_offset + 64);
    }
}

/** A nonzero byte for every byte of @p word that is not an ASCII digit. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_subdocuments_non_digits_(sz_u64_t word) {
    return ((word & 0xF0F0F0F0F0F0F0F0ull) ^ 0x3030303030303030ull) |
           (((word & 0x0F0F0F0F0F0F0F0Full) + 0x0606060606060606ull) & 0x1010101010101010ull);
}

/**
 *  @brief @p run, the bytes from @p scalar to the next blank or structural byte, or zero unless
 *      they spell a number, @c true, @c false or @c null.
 *
 *  An integer of at most 16 bytes, most numbers, is checked by two word tests for digits, with no
 *  loop per byte, while @p readable bytes from @p scalar on allow reading both words.
 */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_scalar_run_length_(sz_u8_t const *scalar, sz_size_t run,
                                                                sz_size_t readable) {
    sz_size_t cursor = 0, digits;
    if (run <= 16 && readable >= 16) {
        sz_size_t const sign = scalar[0] == '-', first_bytes = sz_min_of_two(run, (sz_size_t)8);
        sz_u64_t const first = sz_subdocuments_non_digits_(sz_subdocuments_word_load_((sz_cptr_t)scalar)) &
                               sz_u64_mask_until_serial_(first_bytes * 8) & ~((sz_u64_t)sign * 0xFF);
        sz_u64_t const second = sz_subdocuments_non_digits_(sz_subdocuments_word_load_((sz_cptr_t)scalar + 8)) &
                                sz_u64_mask_until_serial_((run - first_bytes) * 8);
        // Only a lone zero may start with one.
        if (!(first | second) && run != sign) return scalar[sign] == '0' && run != sign + 1 ? 0 : run;
    }
    if (run == 4 && scalar[0] == 't' && scalar[1] == 'r' && scalar[2] == 'u' && scalar[3] == 'e') return run;
    if (run == 4 && scalar[0] == 'n' && scalar[1] == 'u' && scalar[2] == 'l' && scalar[3] == 'l') return run;
    if (run == 5 && scalar[0] == 'f' && scalar[1] == 'a' && scalar[2] == 'l' && scalar[3] == 's' && scalar[4] == 'e')
        return run;
    if (cursor != run && scalar[cursor] == '-') ++cursor;
    if (cursor == run || scalar[cursor] < '0' || scalar[cursor] > '9') return 0;
    if (scalar[cursor] == '0') ++cursor;
    else
        while (cursor != run && scalar[cursor] >= '0' && scalar[cursor] <= '9') ++cursor;
    if (cursor != run && scalar[cursor] == '.') {
        for (digits = ++cursor; cursor != run && scalar[cursor] >= '0' && scalar[cursor] <= '9';) ++cursor;
        if (cursor == digits) return 0;
    }
    if (cursor != run && (scalar[cursor] | 0x20) == 'e') {
        ++cursor;
        if (cursor != run && (scalar[cursor] == '+' || scalar[cursor] == '-')) ++cursor;
        for (digits = cursor; cursor != run && scalar[cursor] >= '0' && scalar[cursor] <= '9';) ++cursor;
        if (cursor == digits) return 0;
    }
    return cursor == run ? run : 0;
}

/** Bytes of the scalar at @p position, or zero unless it is a number, @c true,
 *  @c false or @c null. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_scalar_length_(sz_u8_t const *text, sz_size_t position, sz_size_t length) {
    sz_size_t run = 0;
    for (; position + run != length; ++run) {
        sz_u8_t const byte = text[position + run];
        if (byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' || byte == ',' || byte == ':' || byte == '"' ||
            (byte | 0x20) == '{' || (byte | 0x20) == '}')
            break;
    }
    return sz_subdocuments_scalar_run_length_(text + position, run, length - position);
}

/** Where a walk stands in the grammar, which settles what the next structural byte may be. */
typedef enum sz_subdocuments_expect_t {

    /** A value: the document's root, a member's after its colon, or an element after a comma. */
    sz_subdocuments_expect_value_k = 0,

    /** An array's first element, or its closing bracket. */
    sz_subdocuments_expect_value_or_close_k = 1,

    /** A member's key after a comma. */
    sz_subdocuments_expect_key_k = 2,

    /** An object's first key, or its closing brace. */
    sz_subdocuments_expect_key_or_close_k = 3,

    /** The colon after a key. */
    sz_subdocuments_expect_colon_k = 4,

    /** A comma or the closing bracket after a value inside a container. */
    sz_subdocuments_expect_comma_or_close_k = 5,

    /** Nothing but blank space, after the root value. */
    sz_subdocuments_expect_end_k = 6,
} sz_subdocuments_expect_t;

/** The slot index of no slot: the parent of the root value. */
static sz_size_t const sz_subdocuments_no_slot_k = STRINGZILLA_SIZE_MAX;

/**
 *  @brief One slot of the depth stack, its live set following it in @c words_count words: an open
 *      JSON container, or inside a pending run any node the run may report, or an open XML element,
 *      whose positional counters follow its live set.
 *
 *  Outside a pending run the stack holds the open containers only, one slot per depth. Inside one
 *  no slot is freed until the run's array closes: every container entered and every node a
 *  tentative live set reports is kept, in document order, for the replay. An XML element uses only
 *  @c parent, @c byte_offset and the matches range.
 */
typedef struct sz_subdocuments_frame_t {

    /** The slot of the container holding this node, or
     *  @c sz_subdocuments_no_slot_k for the root. */
    sz_size_t parent;

    /** The node's position for an element; its key's name index, or
     *  @c sz_subdocuments_no_name_k, for a member. */
    sz_size_t selector;

    /** The next element's position for an array, which its close leaves as the element count. */
    sz_size_t next_position;

    /** Where the node starts. */
    sz_size_t byte_offset;

    /** Bytes the node spans, settled when it ends. */
    sz_size_t byte_length;

    /** The first match this container reported outside a pending run. */
    sz_size_t matches_begin;

    /** One past the last match it reported, whose lengths settle when it closes. */
    sz_size_t matches_end;

    /** The byte that closes it, `}` or `]`, or zero for a node that holds
     *  nothing the walk entered. */
    sz_size_t closer;

    /** Inside a pending run, one past the last slot of the node's subtree,
     *  settled before the replay. */
    sz_size_t subtree_end;
} sz_subdocuments_frame_t;

/** 64-bit words a frame's header takes before its live set. */
STRINGZILLA_CONSTEXPR sz_size_t sz_subdocuments_frame_header_words_(void) {
    return sz_size_divide_round_up(sizeof(sz_subdocuments_frame_t), sizeof(sz_u64_t));
}

/** Grows @p engine 's depth stack to at least @p bytes, keeping the frames it holds. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_engine_scratch_(sz_subdocuments_engine_t *engine, sz_size_t bytes) {
    sz_u64_t *grown, *const kept = (sz_u64_t *)engine->scratch;
    sz_size_t index;
    if (engine->scratch_bytes >= bytes) return sz_success_k;
    bytes = sz_max_of_two(bytes, engine->scratch_bytes * 2);
    grown = (sz_u64_t *)engine->allocator.allocate(bytes, engine->allocator.handle, STRINGZILLA_NULL);
    if (!grown) return sz_bad_alloc_k;
    for (index = 0; index != engine->scratch_bytes / sizeof(sz_u64_t); ++index) grown[index] = kept[index];
    if (kept) engine->allocator.free(kept, engine->scratch_bytes, engine->allocator.handle, STRINGZILLA_NULL);
    engine->scratch = grown, engine->scratch_bytes = bytes;
    return sz_success_k;
}

/** Bytes the walk unescapes a key or a compared string into: past the longest name, as a longer key
 *  matches none, and past the longest string literal, as a string ordering against one needs at
 *  most one byte more of the other. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_key_bytes_(sz_subdocuments_engine_t const *engine) {
    sz_size_t longest = 0, slot, index;
    for (slot = 0; slot != engine->names_slots_count; ++slot) {
        sz_u32_t const name = engine->names_slots[slot];
        if (name != sz_subdocuments_no_name_k)
            longest = sz_max_of_two(longest,
                                    (sz_size_t)(engine->names_offsets[name + 1] - engine->names_offsets[name]));
    }
    for (index = 0; engine->filters_count && index != engine->filters[engine->filters_count - 1].instructions_end;
         ++index)
        if (engine->instructions[index].operation == sz_subdocuments_operation_literal_k &&
            engine->instructions[index].operand == sz_subdocuments_kind_string_k)
            longest = sz_max_of_two(longest, (sz_size_t)engine->instructions[index].string_length + 1);
    return sz_size_divide_round_up(longest, sizeof(sz_u64_t)) * sizeof(sz_u64_t);
}

/** The lengths of the names, bit @c n set for a name of @c n bytes and bit 63 for every longer
 *  one. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_names_lengths_(sz_subdocuments_engine_t const *engine) {
    sz_u64_t lengths = 0;
    sz_size_t slot;
    for (slot = 0; slot != engine->names_slots_count; ++slot) {
        sz_u32_t const name = engine->names_slots[slot];
        if (name != sz_subdocuments_no_name_k)
            lengths |= (sz_u64_t)1 << sz_min_of_two(
                           (sz_size_t)(engine->names_offsets[name + 1] - engine->names_offsets[name]), (sz_size_t)63);
    }
    return lengths;
}

/** Whether @p slice selects element @p position of an array of @p length elements, as RFC 9535
 *  normalizes negative bounds and clamps both to the array. */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_slice_selects_(sz_subdocuments_slice_t const *slice, sz_i64_t position,
                                                            sz_i64_t length) {
    sz_i64_t lower, upper;
    if (slice->step > 0) {
        lower = slice->begin >= 0 ? sz_min_of_two(slice->begin, length) : sz_max_of_two(length + slice->begin, 0);
        upper = slice->end >= 0 ? sz_min_of_two(slice->end, length) : sz_max_of_two(length + slice->end, 0);
        return position >= lower && position < upper && (position - lower) % slice->step == 0 ? sz_true_k : sz_false_k;
    }
    if (slice->step < 0) {
        upper = slice->begin >= 0 ? sz_min_of_two(slice->begin, length - 1) : sz_max_of_two(length + slice->begin, -1);
        lower = slice->end >= 0 ? sz_min_of_two(slice->end, length - 1) : sz_max_of_two(length + slice->end, -1);
        return position > lower && position <= upper && (upper - position) % -slice->step == 0 ? sz_true_k : sz_false_k;
    }
    return sz_false_k;
}

/**
 *  @brief The live set of an array's element: the steps advancing into it that select @p position,
 *      and those a descendant segment carries.
 *  @param[in] words_count Words per step bitset, as the engine holds them, which a walk specialized
 *      for one word folds.
 *  @param[in] length The array's element count, or zero while it is open, when a step in
 *      @c relative_steps selects tentatively; every other slice reads the same for any length past
 *      @p position.
 */
STRINGZILLA_INLINE void sz_subdocuments_element_live_(sz_subdocuments_engine_t const *engine, sz_size_t words_count,
                                                      sz_u64_t const *parent_live, sz_size_t position, sz_size_t length,
                                                      sz_u64_t *live) {
    sz_i64_t const known_length = length ? (sz_i64_t)length : (sz_i64_t)position + 1;
    sz_u64_t carried = 0;
    sz_size_t word;
    for (word = 0; word != words_count; ++word) {
        sz_u64_t const advancing = parent_live[word] & ~engine->final_steps[word];
        sz_u64_t const candidates = advancing << 1 | carried;
        sz_u64_t const tentative = length ? 0 : candidates & engine->relative_steps[word];
        sz_u64_t positional = candidates & ~engine->wildcard_steps[word] & ~tentative;
        sz_u64_t selected = (candidates & engine->wildcard_steps[word]) | tentative;
        carried = advancing >> 63;
        for (; positional; positional &= positional - 1) {
            sz_size_t const step = word * 64 + (sz_size_t)sz_u64_ctz(positional);
            sz_u32_t slice;
            for (slice = engine->slices_offsets[step]; slice != engine->slices_offsets[step + 1]; ++slice)
                if (sz_subdocuments_slice_selects_(&engine->slices[slice], (sz_i64_t)position, known_length) ==
                    sz_true_k) {
                    selected |= positional & (0 - positional);
                    break;
                }
        }
        live[word] = selected | (parent_live[word] & engine->descendant_steps[word]);
    }
}

/** Whether a member of an object whose live set is @p parent_live needs its key looked up: whether
 *  a step advancing into it selects by anything but a wildcard. @p words_count is the engine's. */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_needs_name_(sz_subdocuments_engine_t const *engine, sz_size_t words_count,
                                                         sz_u64_t const *parent_live) {
    sz_u64_t carried = 0, needed = 0;
    sz_size_t word;
    for (word = 0; word != words_count; ++word) {
        sz_u64_t const advancing = parent_live[word] & ~engine->final_steps[word];
        needed |= (advancing << 1 | carried) & ~engine->wildcard_steps[word];
        carried = advancing >> 63;
    }
    return needed ? sz_true_k : sz_false_k;
}

/** The live set of an object's member: the steps advancing into it whose names include its key's
 *  @p name, the wildcards, and those a descendant segment carries, over @p words_count words. */
STRINGZILLA_INLINE void sz_subdocuments_member_live_(sz_subdocuments_engine_t const *engine, sz_size_t words_count,
                                                     sz_u64_t const *parent_live, sz_size_t name, sz_u64_t *live) {
    sz_u64_t carried = 0;
    sz_size_t word;
    for (word = 0; word != words_count; ++word) {
        sz_u64_t const advancing = parent_live[word] & ~engine->final_steps[word];
        sz_u64_t const named = name != sz_subdocuments_no_name_k ? engine->names_steps[name * words_count + word] : 0;
        live[word] = ((advancing << 1 | carried) & (named | engine->wildcard_steps[word])) |
                     (parent_live[word] & engine->descendant_steps[word]);
        carried = advancing >> 63;
    }
}

/** Reports a node once per final step of its @p live set, by path, writing
 *  those below @p matches_capacity. @p words_count is the engine's. */
STRINGZILLA_INLINE void sz_subdocuments_report_(sz_subdocuments_engine_t const *engine, sz_size_t words_count,
                                                sz_u64_t const *live, sz_size_t document_index, sz_size_t byte_offset,
                                                sz_size_t byte_length, sz_match_t *matches, sz_size_t matches_capacity,
                                                sz_size_t *emitted) {
    sz_size_t word;
    for (word = 0; word != words_count; ++word)
        for (sz_u64_t reported = live[word] & engine->final_steps[word]; reported; reported &= reported - 1) {
            if (*emitted < matches_capacity) {
                sz_match_t *const match = &matches[*emitted];
                match->haystack_index = document_index;
                match->needle_index = engine->steps_paths[word * 64 + (sz_size_t)sz_u64_ctz(reported)];
                match->byte_offset = byte_offset;
                match->byte_length = byte_length;
            }
            ++*emitted;
        }
}

/** Settles the lengths of the matches `[begin, end)` that fit, all starting at one byte. */
STRINGZILLA_INLINE void sz_subdocuments_close_matches_(sz_match_t *matches, sz_size_t matches_capacity, sz_size_t begin,
                                                       sz_size_t end, sz_size_t end_offset) {
    for (end = sz_min_of_two(end, matches_capacity); begin < end; ++begin)
        matches[begin].byte_length = end_offset - matches[begin].byte_offset;
}

/** Whether no step is live in @p live: then none is below it either, so a
 *  replay skips the subtree. */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_dead_(sz_u64_t const *live, sz_size_t words_count) {
    sz_u64_t any = 0;
    sz_size_t word;
    for (word = 0; word != words_count; ++word) any |= live[word];
    return any ? sz_false_k : sz_true_k;
}

/** Makes every filter step advancing from @p parent_live into a node select it tentatively, and its
 *  filters' query roots live at the node, so the walk keeps every node a filter may read.
 *  @p words_count is the engine's. */
STRINGZILLA_INLINE void sz_subdocuments_filters_tentative_(sz_subdocuments_engine_t const *engine,
                                                           sz_size_t words_count, sz_u64_t const *parent_live,
                                                           sz_u64_t *live) {
    sz_u64_t carried = 0, any = 0;
    sz_size_t word, filter;
    for (word = 0; word != words_count; ++word) {
        sz_u64_t const advancing = parent_live[word] & ~engine->final_steps[word];
        sz_u64_t const filtered = (advancing << 1 | carried) & engine->filter_steps[word];
        live[word] |= filtered, any |= filtered, carried = advancing >> 63;
    }
    if (!any) return;
    for (filter = 0; filter != engine->filters_count; ++filter) {
        sz_size_t const step = engine->filters[filter].step;
        if (live[step >> 6] >> (step & 63) & 1)
            for (word = 0; word != words_count; ++word)
                live[word] |= engine->filters_roots[filter * words_count + word];
    }
}

/** A value a filter program holds. */
typedef struct sz_subdocuments_value_t {

    /** Its kind, Nothing for none at all. */
    sz_subdocuments_kind_t kind;

    /** A number, or a logical value as one or zero. */
    sz_f64_t number;

    /** A string's body, or a container's bytes. */
    sz_cptr_t bytes;

    /** Bytes of @c bytes. */
    sz_size_t length;

    /** Whether a string's body is a document's, still holding JSON escapes, or a literal's. */
    sz_bool_t escaped;
} sz_subdocuments_value_t;

/** The value of the document node spanning @p length bytes at @p bytes,
 *  which the walk validated. */
STRINGZILLA_INLINE sz_subdocuments_value_t sz_subdocuments_node_value_(sz_cptr_t bytes, sz_size_t length) {
    sz_subdocuments_value_t value;
    sz_size_t position = 0;
    value.number = 0, value.bytes = bytes, value.length = length, value.escaped = sz_false_k;
    switch (bytes[0]) {
    case '"':
        value.kind = sz_subdocuments_kind_string_k, ++value.bytes, value.length -= 2, value.escaped = sz_true_k;
        break;
    case '{': value.kind = sz_subdocuments_kind_object_k; break;
    case '[': value.kind = sz_subdocuments_kind_array_k; break;
    case 't': value.kind = sz_subdocuments_kind_true_k; break;
    case 'f': value.kind = sz_subdocuments_kind_false_k; break;
    case 'n': value.kind = sz_subdocuments_kind_null_k; break;
    default:
        value.kind = sz_subdocuments_kind_number_k;
        sz_subdocuments_parse_number_(bytes, length, &position, &value.number);
        break;
    }
    return value;
}

/** Orders two strings by codepoints, a document's one unescaped only as far as @p capacity
 *  bytes of @p buffer, which holds one more than the longest literal; both may be
 *  literals, never two documents'. */
STRINGZILLA_INLINE int sz_subdocuments_order_strings_(sz_subdocuments_value_t const *left,
                                                      sz_subdocuments_value_t const *right, sz_ptr_t buffer,
                                                      sz_size_t capacity) {
    sz_cptr_t left_bytes = left->bytes, right_bytes = right->bytes;
    sz_size_t left_length = left->length, right_length = right->length, index, shared;
    // UTF-8 orders as its codepoints do, so the unescaped bytes compare directly.
    if (left->escaped == sz_true_k)
        left_length = sz_subdocuments_unescape_(left_bytes, left_length, '"', buffer, capacity), left_bytes = buffer;
    if (right->escaped == sz_true_k)
        right_length = sz_subdocuments_unescape_(right_bytes, right_length, '"', buffer, capacity),
        right_bytes = buffer;
    shared = sz_min_of_two(left_length, right_length);
    for (index = 0; index != shared; ++index)
        if (left_bytes[index] != right_bytes[index])
            return (sz_u8_t)left_bytes[index] < (sz_u8_t)right_bytes[index] ? -1 : 1;
    return left_length == right_length ? 0 : left_length < right_length ? -1 : 1;
}

/** RFC 9535's comparison of @p left with @p right by @p comparison, where a document's string never
 *  meets another's and a container never meets one of equal kind. */
STRINGZILLA_INLINE sz_bool_t sz_subdocuments_compare_(sz_subdocuments_value_t const *left,
                                                      sz_subdocuments_value_t const *right,
                                                      sz_subdocuments_comparison_t comparison, sz_ptr_t buffer,
                                                      sz_size_t capacity) {
    sz_bool_t equal = sz_false_k, less = sz_false_k, greater = sz_false_k;
    if (left->kind == right->kind) switch (left->kind) {
        case sz_subdocuments_kind_number_k:
            equal = left->number == right->number ? sz_true_k : sz_false_k;
            less = left->number < right->number ? sz_true_k : sz_false_k;
            greater = left->number > right->number ? sz_true_k : sz_false_k;
            break;
        case sz_subdocuments_kind_string_k: {
            int const order = sz_subdocuments_order_strings_(left, right, buffer, capacity);
            equal = order == 0 ? sz_true_k : sz_false_k, less = order < 0 ? sz_true_k : sz_false_k;
            greater = order > 0 ? sz_true_k : sz_false_k;
            break;
        }
        default: equal = sz_true_k; break;
        }
    switch (comparison) {
    case sz_subdocuments_comparison_equal_k: return equal;
    case sz_subdocuments_comparison_not_equal_k: return equal == sz_true_k ? sz_false_k : sz_true_k;
    case sz_subdocuments_comparison_less_k: return less;
    case sz_subdocuments_comparison_less_equal_k:
        return less == sz_true_k || equal == sz_true_k ? sz_true_k : sz_false_k;
    case sz_subdocuments_comparison_greater_k: return greater;
    default: return greater == sz_true_k || equal == sz_true_k ? sz_true_k : sz_false_k;
    }
}

/** RFC 9535's `length()`: a string's codepoints, a container's elements or members, or Nothing. */
STRINGZILLA_INLINE sz_subdocuments_value_t sz_subdocuments_length_(sz_subdocuments_value_t const *value) {
    sz_subdocuments_value_t result;
    sz_size_t count = 0, index;
    sz_u32_t unit;
    result.kind = sz_subdocuments_kind_number_k, result.bytes = STRINGZILLA_NULL, result.length = 0;
    result.escaped = sz_false_k;
    if (value->kind == sz_subdocuments_kind_string_k)
        for (index = 0; index < value->length; ++index) {
            sz_u8_t const byte = (sz_u8_t)value->bytes[index];
            if (byte == '\\' && value->escaped == sz_true_k) {
                ++count;
                if (++index + 4 >= value->length || value->bytes[index] != 'u') continue;
                // A high surrogate's escape and its pair's spell one codepoint.
                unit = sz_subdocuments_hex_unit_(value->bytes + index + 1), index += 4;
                if (unit >= 0xD800 && unit <= 0xDBFF) index += 6;
            }
            else count += (byte & 0xC0) != 0x80;
        }
    else if (value->kind == sz_subdocuments_kind_array_k || value->kind == sz_subdocuments_kind_object_k) {
        sz_size_t depth = 0;
        sz_bool_t content = sz_false_k;
        for (index = 1; index + 1 < value->length; ++index) {
            sz_u8_t const byte = (sz_u8_t)value->bytes[index];
            if (byte == '"') {
                for (++index; index + 1 < value->length && value->bytes[index] != '"'; ++index)
                    index += value->bytes[index] == '\\';
                content = sz_true_k;
            }
            else if (byte == '[' || byte == '{') ++depth, content = sz_true_k;
            else if (byte == ']' || byte == '}') --depth;
            else if (byte == ',' && depth == 0) ++count;
            else if (byte != ' ' && byte != '\t' && byte != '\n' && byte != '\r') content = sz_true_k;
        }
        count += content == sz_true_k;
    }
    else result.kind = sz_subdocuments_kind_nothing_k;
    result.number = (sz_f64_t)count;
    return result;
}

/** Whether pattern @p pattern of @p engine 's regex matches the string @p value, as compiled:
 *  all of it for `match()`, some of it for `search()`; any other value matches nothing. The
 *  string is copied behind the pattern's tag into @p buffer, which the walk sizes past its
 *  document and the longest tag. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_regex_test_(sz_subdocuments_engine_t *engine,
                                                           sz_subdocuments_value_t const *value, sz_size_t pattern,
                                                           sz_ptr_t buffer, sz_bool_t *matched) {
    sz_size_t tagged = 0, power, count = 0;
    sz_string_view_t haystack;
    sz_strings_view_t haystacks;
    sz_status_t status;
    *matched = sz_false_k;
    if (value->kind != sz_subdocuments_kind_string_k) return sz_success_k;
    buffer[tagged++] = '\x01';
    for (power = 1; pattern / power >= 10;) power *= 10;
    for (; power; power /= 10) buffer[tagged++] = (char)('0' + pattern / power % 10);
    buffer[tagged++] = '\x02';
    if (value->escaped == sz_false_k)
        for (power = 0; power != value->length; ++power) buffer[tagged++] = value->bytes[power];
    else {
        power = sz_subdocuments_unescape_(value->bytes, value->length, '"', buffer + tagged, value->length);
        // A broken escape in a string no walk validated matches nothing.
        if (power == STRINGZILLA_SIZE_MAX) return sz_success_k;
        tagged += power;
    }
    haystack.start = buffer, haystack.length = tagged;
    sz_strings_from_string_views(&haystack, 1, &haystacks);
    status = sz_regex_counts(&engine->regex, &haystacks, &count, 1, STRINGZILLA_NULL);
    *matched = count ? sz_true_k : sz_false_k;
    return status;
}

/** Runs filter @p filter for the node in slot @p node, over the slots `[node, end)` its subtree
 *  holds, whose live sets a replay from the filter's query roots left: a query selects the nodes
 *  where its final step is live. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_filter_holds_(sz_subdocuments_engine_t *engine, sz_u8_t const *text,
                                                             sz_u64_t const *stack, sz_size_t node, sz_size_t end,
                                                             sz_subdocuments_filter_t const *filter, sz_ptr_t buffer,
                                                             sz_size_t capacity, sz_bool_t *holds) {
    sz_size_t const header_words = sz_subdocuments_frame_header_words_(), words_count = engine->words_count;
    sz_size_t const frame_words = header_words + words_count;
    sz_subdocuments_value_t values[sz_subdocuments_filter_depth_k];
    sz_size_t depth = 0, index;
    sz_status_t status = sz_success_k;
    values[0].number = 0;
    for (index = filter->instructions_begin; index != filter->instructions_end && status == sz_success_k; ++index) {
        sz_subdocuments_instruction_t const *const instruction = &engine->instructions[index];
        sz_subdocuments_value_t *const top = &values[depth];
        switch (instruction->operation) {
        case sz_subdocuments_operation_literal_k:
            top->kind = (sz_subdocuments_kind_t)instruction->operand, top->number = instruction->number;
            top->bytes = engine->strings_data + instruction->string_offset, top->length = instruction->string_length;
            top->escaped = sz_false_k, ++depth;
            break;
        case sz_subdocuments_operation_exists_k:
        case sz_subdocuments_operation_value_k:
        case sz_subdocuments_operation_count_k: {
            sz_size_t const step = instruction->operand;
            sz_size_t found = 0, first = 0, slot;
            // A dead slot's subtree was not replayed and holds stale live sets.
            for (slot = node; slot < end; ++slot)
                if (sz_subdocuments_dead_(stack + slot * frame_words + header_words, words_count) == sz_true_k)
                    slot = ((sz_subdocuments_frame_t const *)(stack + slot * frame_words))->subtree_end - 1;
                else if (stack[slot * frame_words + header_words + (step >> 6)] >> (step & 63) & 1)
                    first = found++ ? first : slot;
            top->bytes = STRINGZILLA_NULL, top->length = 0, top->escaped = sz_false_k;
            if (instruction->operation == sz_subdocuments_operation_value_k) {
                sz_subdocuments_frame_t const *const frame = (sz_subdocuments_frame_t const *)(stack +
                                                                                               first * frame_words);
                if (found == 1)
                    *top = sz_subdocuments_node_value_((sz_cptr_t)text + frame->byte_offset, frame->byte_length);
                else top->kind = sz_subdocuments_kind_nothing_k;
            }
            else if (instruction->operation == sz_subdocuments_operation_count_k)
                top->kind = sz_subdocuments_kind_number_k, top->number = (sz_f64_t)found;
            else top->kind = sz_subdocuments_kind_logical_k, top->number = found != 0;
            ++depth;
            break;
        }
        case sz_subdocuments_operation_length_k: values[depth - 1] = sz_subdocuments_length_(&values[depth - 1]); break;
        case sz_subdocuments_operation_match_k:
        case sz_subdocuments_operation_search_k: {
            sz_bool_t matched;
            status = sz_subdocuments_regex_test_(engine, &values[depth - 1], instruction->operand, buffer, &matched);
            values[depth - 1].kind = sz_subdocuments_kind_logical_k, values[depth - 1].number = matched == sz_true_k;
            break;
        }
        case sz_subdocuments_operation_compare_k:
            --depth;
            values[depth - 1].number = sz_subdocuments_compare_(&values[depth - 1], &values[depth],
                                                                (sz_subdocuments_comparison_t)instruction->operand,
                                                                buffer, capacity) == sz_true_k;
            values[depth - 1].kind = sz_subdocuments_kind_logical_k;
            break;
        case sz_subdocuments_operation_not_k: values[depth - 1].number = values[depth - 1].number == 0; break;
        case sz_subdocuments_operation_and_k:
            --depth, values[depth - 1].number = values[depth - 1].number != 0 && values[depth].number != 0;
            break;
        default: --depth, values[depth - 1].number = values[depth - 1].number != 0 || values[depth].number != 0; break;
        }
    }
    *holds = values[0].number != 0 ? sz_true_k : sz_false_k;
    return status;
}

/**
 *  @brief Settles slots `[begin, end)` of a closed pending run, each from its parent's live set.
 *
 *  Before a node's own live set, every filter of a step advancing into it runs: a replay of the
 *  node's subtree from the filter's query roots, which overwrites the live sets there before this
 *  replay recomputes them, and nests for the filters inside those queries. The main replay reports
 *  the matches; a filter's reports nothing, its queries' final steps marking what they select.
 *
 *  @param[in] spare A free slot past every kept one, holding this level's filter results; deeper
 *      levels take the next ones.
 *  @param[inout] emitted Matches so far, or null for a filter's replay.
 */
STRINGZILLA_C_INLINE_ sz_status_t sz_subdocuments_replay_(sz_subdocuments_engine_t *engine, sz_u8_t const *text,
                                                          sz_u64_t *stack, sz_size_t begin, sz_size_t end,
                                                          sz_size_t spare, sz_ptr_t buffer, sz_size_t capacity,
                                                          sz_size_t document_index, sz_match_t *matches,
                                                          sz_size_t matches_capacity, sz_size_t *emitted) {
    sz_size_t const header_words = sz_subdocuments_frame_header_words_(), words_count = engine->words_count;
    sz_size_t const frame_words = header_words + words_count;
    sz_u64_t *const results = stack + spare * frame_words;
    sz_size_t slot, word, filter;
    for (slot = begin; slot != end; ++slot) {
        sz_subdocuments_frame_t const *const frame = (sz_subdocuments_frame_t const *)(stack + slot * frame_words);
        sz_subdocuments_frame_t const *const parent = (sz_subdocuments_frame_t const *)(stack +
                                                                                        frame->parent * frame_words);
        sz_u64_t *const live = stack + slot * frame_words + header_words;
        sz_u64_t const *const parent_live = (sz_u64_t const *)parent + header_words;
        sz_u64_t carried = 0, filtered = 0;
        for (word = 0; word != words_count; ++word) {
            sz_u64_t const advancing = parent_live[word] & ~engine->final_steps[word];
            filtered |= (advancing << 1 | carried) & engine->filter_steps[word], carried = advancing >> 63;
            results[word] = 0;
        }
        if (filtered) {
            sz_size_t const subtree_end = frame->subtree_end;
            // A filter step advances into the node when the step before it is live and not final.
            for (filter = 0; filter != engine->filters_count; ++filter) {
                sz_subdocuments_filter_t const *const candidate = &engine->filters[filter];
                sz_size_t const step = candidate->step, before = step - 1;
                sz_bool_t holds = sz_false_k;
                sz_status_t status;
                if (!((parent_live[before >> 6] & ~engine->final_steps[before >> 6]) >> (before & 63) & 1) ||
                    results[step >> 6] >> (step & 63) & 1)
                    continue;
                for (word = 0; word != words_count; ++word)
                    live[word] = engine->filters_roots[filter * words_count + word];
                status = sz_subdocuments_replay_(engine, text, stack, slot + 1, subtree_end, spare + 1, buffer,
                                                 capacity, document_index, matches, matches_capacity,
                                                 (sz_size_t *)STRINGZILLA_NULL);
                if (status == sz_success_k)
                    status = sz_subdocuments_filter_holds_(engine, text, stack, slot, subtree_end, candidate, buffer,
                                                           capacity, &holds);
                if (status != sz_success_k) return status;
                if (holds == sz_true_k) results[step >> 6] |= (sz_u64_t)1 << (step & 63);
            }
        }
        if (parent->closer == ']')
            sz_subdocuments_element_live_(engine, words_count, parent_live, frame->selector, parent->next_position,
                                          live);
        else sz_subdocuments_member_live_(engine, words_count, parent_live, frame->selector, live);
        if (filtered)
            for (word = 0; word != words_count; ++word) live[word] |= results[word];
        if (emitted)
            sz_subdocuments_report_(engine, words_count, live, document_index, frame->byte_offset, frame->byte_length,
                                    matches, matches_capacity, emitted);
        if (sz_subdocuments_dead_(live, words_count) == sz_true_k) slot = frame->subtree_end - 1;
    }
    return sz_success_k;
}

/**
 *  @brief Walks one document, reporting every node a path selects, in document order.
 *
 *  The live set of the value the walk is about to meet sits in the first free slot, @c top: the
 *  root steps at the start, a member's set from its key, an element's from its position. A
 *  container whose children a waiting step may select, counted from an array's end or decided by a
 *  filter, opens a pending run, where every node is walked with a tentative live set, a superset of
 *  the true one, and kept until the replay.
 *
 *  @param[in] skim The tier's skim for the blocks a skipped container holds whole, or its
 *      classifier where the tier has none.
 *  @param[in] words_count Words per step bitset, as the engine holds them, a constant where the
 *      batch specializes the walk for one word, so no loop over words remains.
 *  @param[in] names_lengths The lengths of the names, as @ref sz_subdocuments_names_lengths_ sets
 *      them for the batch.
 *  @param[inout] matches_count Matches reported before this document, then after it; only those
 *      below @p matches_capacity are written.
 *  @return @c sz_success_k, @c sz_invalid_syntax_k for a malformed document, or @c sz_bad_alloc_k
 *      when the depth stack cannot grow.
 */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_walk_(sz_subdocuments_engine_t *engine,
                                                     sz_subdocuments_classify_t classify,
                                                     sz_subdocuments_classify_t skim, sz_size_t words_count,
                                                     sz_cptr_t document, sz_size_t length, sz_size_t document_index,
                                                     sz_size_t key_bytes, sz_u64_t names_lengths, sz_match_t *matches,
                                                     sz_size_t matches_capacity, sz_size_t *matches_count) {
    sz_u64_t const *const final_steps = engine->final_steps, *const relative_steps = engine->relative_steps;
    sz_size_t const header_words = sz_subdocuments_frame_header_words_();
    sz_size_t const frame_words = header_words + words_count, key_words = key_bytes / sizeof(sz_u64_t);
    sz_u8_t const *const text = (sz_u8_t const *)document;
    sz_subdocuments_expect_t expect = sz_subdocuments_expect_value_k;
    sz_size_t top = 0, current = sz_subdocuments_no_slot_k, run = sz_subdocuments_no_slot_k;
    sz_size_t position, closing, word, emitted = *matches_count;
    sz_subdocuments_cursor_t cursor;
    sz_u64_t *stack, *live, carried;
    sz_status_t status = sz_subdocuments_engine_scratch_(engine, (key_words + 2 * frame_words) * sizeof(sz_u64_t));
    if (status != sz_success_k) return status;
    stack = (sz_u64_t *)engine->scratch + key_words;

    live = stack + header_words;
    for (word = 0; word != words_count; ++word) live[word] = engine->root_steps[word];

    cursor.text = text, cursor.length = length;
    cursor.carry.escaped = 0, cursor.carry.in_string = 0, cursor.carry.scalar = 0, cursor.carry.slashes = 0;
    sz_subdocuments_cursor_load_(&cursor, classify, 0);
    while (sz_subdocuments_cursor_next_(&cursor, classify, &position) == sz_true_k) {
        sz_u8_t const byte = text[position];
        sz_subdocuments_frame_t *const parent = current != sz_subdocuments_no_slot_k
                                                    ? (sz_subdocuments_frame_t *)(stack + current * frame_words)
                                                    : (sz_subdocuments_frame_t *)STRINGZILLA_NULL;
        sz_u64_t const *const parent_live = parent ? (sz_u64_t const *)parent + header_words : STRINGZILLA_NULL;
        sz_subdocuments_frame_t *const frame = (sz_subdocuments_frame_t *)(stack + top * frame_words);
        sz_size_t matches_begin, end;
        sz_u64_t reported = 0, continuing = 0;
        live = (sz_u64_t *)frame + header_words;

        if (byte == ':') {
            if (expect != sz_subdocuments_expect_colon_k) return sz_invalid_syntax_k;
            expect = sz_subdocuments_expect_value_k;
            continue;
        }
        if (byte == ',') {
            if (expect != sz_subdocuments_expect_comma_or_close_k) return sz_invalid_syntax_k;
            expect = parent->closer == '}' ? sz_subdocuments_expect_key_k : sz_subdocuments_expect_value_k;
            continue;
        }
        if (byte == '}' || byte == ']') {
            if (!parent || parent->closer != byte) return sz_invalid_syntax_k;
            if (expect != sz_subdocuments_expect_comma_or_close_k &&
                expect !=
                    (byte == '}' ? sz_subdocuments_expect_key_or_close_k : sz_subdocuments_expect_value_or_close_k))
                return sz_invalid_syntax_k;
            parent->byte_length = position + 1 - parent->byte_offset;
            // The replay may grow the stack, so the grandparent is read while `parent`
            // still points into it.
            closing = parent->parent;
            if (run == sz_subdocuments_no_slot_k || run == current) {
                sz_subdocuments_close_matches_(matches, matches_capacity, parent->matches_begin, parent->matches_end,
                                               position + 1);
                if (run == current) {
                    sz_size_t slot;
                    // Filters replay at one spare slot per nesting level past the kept ones.
                    status = sz_subdocuments_engine_scratch_(
                        engine, (key_words + (top + engine->filters_count + 2) * frame_words) * sizeof(sz_u64_t));
                    if (status != sz_success_k) return status;
                    stack = (sz_u64_t *)engine->scratch + key_words;
                    // Kept slots are in document order, so a subtree ends one
                    // past its last descendant.
                    for (slot = run + 1; slot != top; ++slot)
                        ((sz_subdocuments_frame_t *)(stack + slot * frame_words))->subtree_end = slot + 1;
                    for (slot = top - 1; slot > run + 1; --slot) {
                        sz_subdocuments_frame_t const *const node =
                            (sz_subdocuments_frame_t const *)(stack + slot * frame_words);
                        sz_subdocuments_frame_t *const holder = (sz_subdocuments_frame_t *)(stack +
                                                                                            node->parent * frame_words);
                        if (node->parent != run)
                            holder->subtree_end = sz_max_of_two(holder->subtree_end, node->subtree_end);
                    }
                    status = sz_subdocuments_replay_(engine, text, stack, run + 1, top, top, (sz_ptr_t)engine->scratch,
                                                     key_bytes, document_index, matches, matches_capacity, &emitted);
                    if (status != sz_success_k) return status;
                    run = sz_subdocuments_no_slot_k;
                }
                top = current;
            }
            current = closing;
            expect = current != sz_subdocuments_no_slot_k ? sz_subdocuments_expect_comma_or_close_k
                                                          : sz_subdocuments_expect_end_k;
            continue;
        }

        if (byte == '"' &&
            (expect == sz_subdocuments_expect_key_k || expect == sz_subdocuments_expect_key_or_close_k)) {
            sz_cptr_t key = document + position + 1;
            sz_size_t key_length, name = sz_subdocuments_no_name_k, index;
            if (sz_subdocuments_cursor_next_(&cursor, classify, &closing) == sz_false_k) return sz_invalid_syntax_k;
            key_length = closing - position - 1;
            if (engine->names_slots_count &&
                sz_subdocuments_needs_name_(engine, words_count, parent_live) == sz_true_k) {
                // A key the block holds whole has its backslashes in the block's mask.
                sz_bool_t escaped = sz_false_k;
                if (position + 1 >= cursor.block_offset)
                    escaped = (cursor.block.backslashes >> (position + 1 - cursor.block_offset) &
                               sz_u64_mask_until_serial_(key_length))
                                  ? sz_true_k
                                  : sz_false_k;
                else
                    for (index = 0; index != key_length && escaped == sz_false_k; ++index)
                        escaped = key[index] == '\\' ? sz_true_k : sz_false_k;
                // Most keys share no name's length, and need no hash.
                if (escaped == sz_false_k) {
                    if (names_lengths >> sz_min_of_two(key_length, (sz_size_t)63) & 1)
                        name = engine->names_slots[sz_subdocuments_name_slot_(
                            engine->names_data, engine->names_offsets, engine->names_slots, engine->names_slots_count,
                            key, key_length, length - position - 1)];
                }
                else {
                    sz_ptr_t const unescaped = (sz_ptr_t)engine->scratch;
                    key_length = sz_subdocuments_unescape_(key, key_length, '"', unescaped, key_bytes);
                    if (key_length == STRINGZILLA_SIZE_MAX) return sz_invalid_syntax_k;
                    if (key_length <= key_bytes)
                        name = engine->names_slots[sz_subdocuments_name_slot_(
                            engine->names_data, engine->names_offsets, engine->names_slots, engine->names_slots_count,
                            unescaped, key_length, key_bytes)];
                }
            }
            sz_subdocuments_member_live_(engine, words_count, parent_live, name, live);
            if (engine->filters_count) sz_subdocuments_filters_tentative_(engine, words_count, parent_live, live);
            frame->selector = name;
            expect = sz_subdocuments_expect_colon_k;
            continue;
        }

        if (expect != sz_subdocuments_expect_value_k && expect != sz_subdocuments_expect_value_or_close_k)
            return sz_invalid_syntax_k;
        if (parent && parent->closer == ']') {
            frame->selector = parent->next_position++;
            sz_subdocuments_element_live_(engine, words_count, parent_live, frame->selector, 0, live);
            if (engine->filters_count) sz_subdocuments_filters_tentative_(engine, words_count, parent_live, live);
        }
        for (word = 0; word != words_count; ++word)
            reported |= live[word] & final_steps[word], continuing |= live[word] & ~final_steps[word];
        matches_begin = emitted;
        if (run == sz_subdocuments_no_slot_k)
            sz_subdocuments_report_(engine, words_count, live, document_index, position, 0, matches, matches_capacity,
                                    &emitted);
        if ((byte == '{' || byte == '[') && continuing) {
            frame->parent = current, frame->byte_offset = position;
            frame->next_position = 0, frame->matches_begin = matches_begin, frame->matches_end = emitted;
            frame->closer = byte == '{' ? '}' : ']';
            // A container opens a pending run when a step that waits may advance
            // into what it holds.
            if (run == sz_subdocuments_no_slot_k) {
                sz_u64_t waiting = 0;
                for (word = 0, carried = 0; word != words_count; ++word) {
                    sz_u64_t const advancing = live[word] & ~final_steps[word];
                    waiting |= (advancing << 1 | carried) &
                               ((byte == '[' ? relative_steps[word] : 0) | engine->filter_steps[word]);
                    carried = advancing >> 63;
                }
                if (waiting) run = top;
            }
            current = top++;
            status = sz_subdocuments_engine_scratch_(engine, (key_words + (top + 1) * frame_words) * sizeof(sz_u64_t));
            if (status != sz_success_k) return status;
            stack = (sz_u64_t *)engine->scratch + key_words;
            expect = byte == '{' ? sz_subdocuments_expect_key_or_close_k : sz_subdocuments_expect_value_or_close_k;
            continue;
        }
        if (byte == '{' || byte == '[') {
            if (sz_subdocuments_cursor_skip_(&cursor, classify, skim, &closing) == sz_false_k)
                return sz_invalid_syntax_k;
            end = closing + 1;
        }
        else if (byte == '"') {
            if (sz_subdocuments_cursor_next_(&cursor, classify, &closing) == sz_false_k) return sz_invalid_syntax_k;
            end = closing + 1;
        }
        else {
            // A scalar inside the classified block ends at the first byte its mask leaves out.
            sz_u64_t const after = ~cursor.block.scalars >> (position - cursor.block_offset);
            end = position + (after ? sz_subdocuments_scalar_run_length_(text + position, (sz_size_t)sz_u64_ctz(after),
                                                                         length - position)
                                    : sz_subdocuments_scalar_length_(text, position, length));
            if (end == position) return sz_invalid_syntax_k;
        }
        if (run == sz_subdocuments_no_slot_k)
            sz_subdocuments_close_matches_(matches, matches_capacity, matches_begin, emitted, end);
        else if (reported) {
            frame->parent = current, frame->byte_offset = position, frame->byte_length = end - position;
            frame->closer = 0;
            ++top;
            status = sz_subdocuments_engine_scratch_(engine, (key_words + (top + 1) * frame_words) * sizeof(sz_u64_t));
            if (status != sz_success_k) return status;
            stack = (sz_u64_t *)engine->scratch + key_words;
        }
        expect = current != sz_subdocuments_no_slot_k ? sz_subdocuments_expect_comma_or_close_k
                                                      : sz_subdocuments_expect_end_k;
    }
    if (current != sz_subdocuments_no_slot_k || expect != sz_subdocuments_expect_end_k || cursor.carry.slashes)
        return sz_invalid_syntax_k;
    *matches_count = emitted;
    return sz_success_k;
}

#pragma endregion Walk

#pragma region Markup

/** Signature of a tier's byte search, which the XML lexer runs to the next byte its state needs:
 *  `<` in character data, the closing quote of an attribute value, `>` in comments and the like. */
typedef sz_cptr_t (*sz_subdocuments_find_byte_t)(sz_cptr_t, sz_size_t, sz_cptr_t);

/** The first @p byte at or after @p from, or @p length. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_markup_find_(sz_u8_t const *text, sz_size_t length,
                                                          sz_subdocuments_find_byte_t find_byte, sz_size_t from,
                                                          sz_u8_t byte) {
    sz_cptr_t const found = find_byte((sz_cptr_t)text + from, length - from, (sz_cptr_t)&byte);
    return found ? (sz_size_t)(found - (sz_cptr_t)text) : length;
}

/** The `>` ending a comment, CDATA section or processing instruction whose body starts at
 *  @p from: the first preceded by @p terminator_length copies of @p terminator inside the
 *  body, or @p length. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_markup_end_(sz_u8_t const *text, sz_size_t length,
                                                         sz_subdocuments_find_byte_t find_byte, sz_size_t from,
                                                         sz_u8_t terminator, sz_size_t terminator_length) {
    sz_size_t candidate = from;
    for (;; ++candidate) {
        candidate = sz_subdocuments_markup_find_(text, length, find_byte, candidate, '>');
        if (candidate == length) return candidate;
        if (candidate >= from + terminator_length && text[candidate - 1] == terminator &&
            text[candidate - terminator_length] == terminator)
            return candidate;
    }
}

/** The first byte at or after @p position that is not XML blank space. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_markup_skip_blanks_(sz_u8_t const *text, sz_size_t length,
                                                                 sz_size_t position) {
    while (position != length &&
           (text[position] == ' ' || text[position] == '\t' || text[position] == '\n' || text[position] == '\r'))
        ++position;
    return position;
}

/** The first byte at or after @p position that cannot continue an XML name. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_markup_skip_name_(sz_u8_t const *text, sz_size_t length,
                                                               sz_size_t position) {
    while (position != length && sz_subdocuments_is_xml_name_(text[position]) == sz_true_k) ++position;
    return position;
}

/** What an XPath step meets: an element, an attribute or a text node. */
typedef enum sz_subdocuments_node_t {

    /** An element, which name tests and `*` select. */
    sz_subdocuments_node_element_k = 0,

    /** An attribute, which `@name` and `@*` select. */
    sz_subdocuments_node_attribute_k = 1,

    /** A text node, which `text()` selects. */
    sz_subdocuments_node_text_k = 2,
} sz_subdocuments_node_t;

/**
 *  @brief The live set of an XML node from its parent element's: the steps advancing into it whose
 *      test it passes, kept by their positions among the siblings passing the same test, plus for
 *      an element the steps a descendant segment carries.
 *  @param[inout] counters The parent's count of children passing each positional step's test, by
 *      its slice index, as a step holds at most one position.
 */
STRINGZILLA_INLINE void sz_subdocuments_node_live_(sz_subdocuments_engine_t const *engine, sz_u64_t const *parent_live,
                                                   sz_u32_t *counters, sz_subdocuments_node_t kind, sz_size_t name,
                                                   sz_u64_t *live) {
    sz_u64_t carried = 0;
    sz_size_t word;
    for (word = 0; word != engine->words_count; ++word) {
        sz_u64_t const advancing = parent_live[word] & ~engine->final_steps[word];
        sz_u64_t const candidates = advancing << 1 | carried;
        sz_u64_t const named = name != sz_subdocuments_no_name_k
                                   ? engine->names_steps[name * engine->words_count + word]
                                   : 0;
        sz_u64_t tested = kind == sz_subdocuments_node_text_k ? candidates & engine->text_steps[word]
                          : kind == sz_subdocuments_node_attribute_k
                              ? candidates & (named | engine->wildcard_steps[word]) & engine->attribute_steps[word]
                              : candidates & (named | engine->wildcard_steps[word]) & ~engine->attribute_steps[word];
        carried = advancing >> 63;
        for (sz_u64_t counted = tested; counted; counted &= counted - 1) {
            sz_size_t const step = word * 64 + (sz_size_t)sz_u64_ctz(counted);
            sz_u32_t const slice = engine->slices_offsets[step];
            if (slice == engine->slices_offsets[step + 1]) continue;
            if (sz_subdocuments_slice_selects_(&engine->slices[slice], counters[slice],
                                               (sz_i64_t)counters[slice] + 1) == sz_false_k)
                tested &= ~(counted & (0 - counted));
            ++counters[slice];
        }
        live[word] = tested |
                     (kind == sz_subdocuments_node_element_k ? parent_live[word] & engine->descendant_steps[word] : 0);
    }
}

/** The name index of the @p name_length bytes at @p name, or @c sz_subdocuments_no_name_k. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_markup_name_(sz_subdocuments_engine_t const *engine, sz_cptr_t name,
                                                          sz_size_t name_length, sz_size_t key_bytes) {
    if (!engine->names_slots_count || name_length > key_bytes) return sz_subdocuments_no_name_k;
    return engine
        ->names_slots[sz_subdocuments_name_slot_(engine->names_data, engine->names_offsets, engine->names_slots,
                                                 engine->names_slots_count, name, name_length, name_length)];
}

/**
 *  @brief Walks one XML document, reporting every element, attribute and text node a path selects,
 *      in document order.
 *
 *  Slot 0 is the document node, whose child is the root element; every open element a live step can
 *  advance from takes a slot, its live set and its positional counters following the header. XML
 *  lexing needs no state across blocks, so instead of classifying every byte, the lexer runs the
 *  tier's byte search to the one byte its state needs, `<` in character data, the closing quote in
 *  an attribute value, and `>` in comments, CDATA sections and processing instructions, and reads
 *  names byte by byte. An element no live step advances from is lexed to its end tag without a
 *  slot, so what it holds is not validated beyond its nesting.
 *
 *  @return @c sz_success_k, @c sz_invalid_syntax_k for a malformed document or a DOCTYPE with an
 *      internal subset, or @c sz_bad_alloc_k when the depth stack cannot grow.
 */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_walk_markup_(sz_subdocuments_engine_t *engine,
                                                            sz_subdocuments_find_byte_t find_byte, sz_cptr_t document,
                                                            sz_size_t length, sz_size_t document_index,
                                                            sz_size_t key_bytes, sz_match_t *matches,
                                                            sz_size_t matches_capacity, sz_size_t *matches_count) {
    sz_u64_t const *const final_steps = engine->final_steps;
    sz_size_t const words_count = engine->words_count, header_words = sz_subdocuments_frame_header_words_();
    sz_size_t const counter_words = (engine->slices_offsets[engine->steps_count] + 1) / 2;
    sz_size_t const frame_words = header_words + words_count + counter_words, key_words = key_bytes / sizeof(sz_u64_t);
    sz_u8_t const *const text = (sz_u8_t const *)document;
    sz_size_t top = 1, current = 0, position = 0, text_begin = 0, skipped_depth = 0, skipped_begin = 0;
    sz_size_t skipped_end = 0, skipped_opening = 0, opening, tag_end, word, emitted = *matches_count;
    sz_bool_t rooted = sz_false_k;
    sz_subdocuments_frame_t *frame;
    sz_u64_t *stack, *live;
    // Past the document node, one slot for the next element and one for an attribute's live set.
    sz_status_t status = sz_subdocuments_engine_scratch_(engine, (key_words + 3 * frame_words) * sizeof(sz_u64_t));
    if (status != sz_success_k) return status;
    stack = (sz_u64_t *)engine->scratch + key_words;

    frame = (sz_subdocuments_frame_t *)stack, frame->parent = sz_subdocuments_no_slot_k;
    live = stack + header_words;
    for (word = 0; word != words_count; ++word) live[word] = engine->root_steps[word];
    for (word = 0; word != counter_words; ++word) live[words_count + word] = 0;

    for (;;) {
        sz_subdocuments_frame_t *const parent = (sz_subdocuments_frame_t *)(stack + current * frame_words);
        sz_u64_t const *const parent_live = (sz_u64_t const *)parent + header_words;
        sz_u32_t *const parent_counters = (sz_u32_t *)(parent_live + words_count);
        sz_u8_t kind;
        opening = sz_subdocuments_markup_find_(text, length, find_byte, position, '<');
        // Outside the root element, character data may only be blank space.
        if (current == 0 && skipped_depth == 0 &&
            sz_subdocuments_markup_skip_blanks_(text, opening, position) != opening)
            return sz_invalid_syntax_k;
        if (opening == length) break;
        if (length - opening < 2) return sz_invalid_syntax_k;
        kind = text[opening + 1];

        // A CDATA section is character data, so the text node around it goes on.
        if (kind == '!' && length - opening >= 9 && text[opening + 2] == '[' && text[opening + 3] == 'C' &&
            text[opening + 4] == 'D' && text[opening + 5] == 'A' && text[opening + 6] == 'T' &&
            text[opening + 7] == 'A' && text[opening + 8] == '[') {
            if (current == 0 && skipped_depth == 0) return sz_invalid_syntax_k;
            tag_end = sz_subdocuments_markup_end_(text, length, find_byte, opening + 9, ']', 2);
            if (tag_end == length) return sz_invalid_syntax_k;
            position = tag_end + 1;
            continue;
        }
        // Every other markup ends a text node, which is any character data or CDATA between markup.
        if (current != 0 && skipped_depth == 0 && opening != text_begin) {
            frame = (sz_subdocuments_frame_t *)(stack + top * frame_words), live = (sz_u64_t *)frame + header_words;
            sz_subdocuments_node_live_(engine, parent_live, parent_counters, sz_subdocuments_node_text_k,
                                       sz_subdocuments_no_name_k, live);
            sz_subdocuments_report_(engine, words_count, live, document_index, text_begin, opening - text_begin,
                                    matches, matches_capacity, &emitted);
        }

        if (kind == '!' && length - opening >= 4 && text[opening + 2] == '-' && text[opening + 3] == '-') {
            tag_end = sz_subdocuments_markup_end_(text, length, find_byte, opening + 4, '-', 2);
            if (tag_end == length) return sz_invalid_syntax_k;
        }
        else if (kind == '?') {
            tag_end = sz_subdocuments_markup_end_(text, length, find_byte, opening + 2, '?', 1);
            if (tag_end == length) return sz_invalid_syntax_k;
        }
        else if (kind == '!') {
            // An internal subset could declare entities that no walk expands.
            sz_size_t const name_end = sz_subdocuments_markup_skip_name_(text, length, opening + 2);
            sz_size_t index;
            tag_end = sz_subdocuments_markup_find_(text, length, find_byte, name_end, '>');
            if (current != 0 || rooted == sz_true_k || name_end - opening != 9 || text[opening + 2] != 'D' ||
                text[opening + 3] != 'O' || text[opening + 4] != 'C' || text[opening + 5] != 'T' ||
                text[opening + 6] != 'Y' || text[opening + 7] != 'P' || text[opening + 8] != 'E' || tag_end == length)
                return sz_invalid_syntax_k;
            for (index = name_end; index != tag_end; ++index)
                if (text[index] == '[') return sz_invalid_syntax_k;
        }
        else if (kind == '/') {
            sz_size_t const name_end = sz_subdocuments_markup_skip_name_(text, length, opening + 2);
            sz_size_t const name_length = name_end - opening - 2;
            sz_size_t open_name, index;
            tag_end = sz_subdocuments_markup_skip_blanks_(text, length, name_end);
            if (tag_end == length || text[tag_end] != '>' || name_length == 0) return sz_invalid_syntax_k;
            // Inside a skipped element only nesting counts; the element's own end
            // tag repeats its name.
            if (skipped_depth > 1) --skipped_depth;
            else {
                if (skipped_depth == 0 && current == 0) return sz_invalid_syntax_k;
                open_name = (skipped_depth ? skipped_opening : parent->byte_offset) + 1;
                if (sz_subdocuments_markup_skip_name_(text, length, open_name) - open_name != name_length)
                    return sz_invalid_syntax_k;
                for (index = 0; index != name_length; ++index)
                    if (text[open_name + index] != text[opening + 2 + index]) return sz_invalid_syntax_k;
                if (skipped_depth)
                    sz_subdocuments_close_matches_(matches, matches_capacity, skipped_begin, skipped_end, tag_end + 1),
                        skipped_depth = 0;
                else {
                    sz_subdocuments_close_matches_(matches, matches_capacity, parent->matches_begin,
                                                   parent->matches_end, tag_end + 1);
                    top = current, current = parent->parent;
                }
            }
        }
        else {
            sz_size_t const name_end = sz_subdocuments_markup_skip_name_(text, length, opening + 1);
            sz_size_t matches_begin = emitted, attribute = name_end;
            sz_u64_t continuing = 0;
            sz_u32_t *counters = STRINGZILLA_NULL;
            sz_bool_t empty;
            if (name_end == opening + 1) return sz_invalid_syntax_k;
            if (skipped_depth == 0) {
                if (current == 0 && rooted == sz_true_k) return sz_invalid_syntax_k;
                rooted = sz_true_k;
                frame = (sz_subdocuments_frame_t *)(stack + top * frame_words), live = (sz_u64_t *)frame + header_words;
                counters = (sz_u32_t *)(live + words_count);
                sz_subdocuments_node_live_(engine, parent_live, parent_counters, sz_subdocuments_node_element_k,
                                           sz_subdocuments_needs_name_(engine, words_count, parent_live) == sz_true_k
                                               ? sz_subdocuments_markup_name_(engine, (sz_cptr_t)text + opening + 1,
                                                                              name_end - opening - 1, key_bytes)
                                               : sz_subdocuments_no_name_k,
                                           live);
                sz_subdocuments_report_(engine, words_count, live, document_index, opening, 0, matches,
                                        matches_capacity, &emitted);
                for (word = 0; word != words_count; ++word) continuing |= live[word] & ~final_steps[word];
                for (word = 0; word != counter_words; ++word) live[words_count + word] = 0;
                frame->parent = current, frame->byte_offset = opening;
                frame->matches_begin = matches_begin, frame->matches_end = emitted;
            }
            // Attributes are read even where none is selected, as their values may hold any `>`.
            for (;;) {
                sz_size_t value_begin, value_end, attribute_end;
                sz_u8_t quote;
                attribute = sz_subdocuments_markup_skip_blanks_(text, length, attribute);
                if (attribute == length) return sz_invalid_syntax_k;
                if (text[attribute] == '>') {
                    empty = sz_false_k, tag_end = attribute;
                    break;
                }
                if (text[attribute] == '/') {
                    if (attribute + 1 == length || text[attribute + 1] != '>') return sz_invalid_syntax_k;
                    empty = sz_true_k, tag_end = attribute + 1;
                    break;
                }
                attribute_end = sz_subdocuments_markup_skip_name_(text, length, attribute);
                value_begin = sz_subdocuments_markup_skip_blanks_(text, length, attribute_end);
                if (attribute_end == attribute || value_begin == length || text[value_begin] != '=')
                    return sz_invalid_syntax_k;
                value_begin = sz_subdocuments_markup_skip_blanks_(text, length, value_begin + 1);
                if (value_begin == length || (text[value_begin] != '"' && text[value_begin] != '\''))
                    return sz_invalid_syntax_k;
                quote = text[value_begin++];
                value_end = sz_subdocuments_markup_find_(text, length, find_byte, value_begin, quote);
                if (value_end == length) return sz_invalid_syntax_k;
                // Namespace declarations are no attributes in XPath's data model.
                if (continuing &&
                    !(attribute_end - attribute >= 5 && text[attribute] == 'x' && text[attribute + 1] == 'm' &&
                      text[attribute + 2] == 'l' && text[attribute + 3] == 'n' && text[attribute + 4] == 's' &&
                      (attribute_end - attribute == 5 || text[attribute + 5] == ':'))) {
                    sz_u64_t *const attribute_live = live + words_count + counter_words;
                    sz_subdocuments_node_live_(engine, live, counters, sz_subdocuments_node_attribute_k,
                                               sz_subdocuments_needs_name_(engine, words_count, live) == sz_true_k
                                                   ? sz_subdocuments_markup_name_(engine, (sz_cptr_t)text + attribute,
                                                                                  attribute_end - attribute, key_bytes)
                                                   : sz_subdocuments_no_name_k,
                                               attribute_live);
                    sz_subdocuments_report_(engine, words_count, attribute_live, document_index, value_begin,
                                            value_end - value_begin, matches, matches_capacity, &emitted);
                }
                attribute = value_end + 1;
            }
            if (skipped_depth) skipped_depth += empty == sz_false_k;
            else if (empty == sz_true_k)
                sz_subdocuments_close_matches_(matches, matches_capacity, matches_begin, frame->matches_end,
                                               tag_end + 1);
            else if (continuing) {
                current = top++;
                status = sz_subdocuments_engine_scratch_(engine,
                                                         (key_words + (top + 2) * frame_words) * sizeof(sz_u64_t));
                if (status != sz_success_k) return status;
                stack = (sz_u64_t *)engine->scratch + key_words;
            }
            else
                skipped_depth = 1, skipped_opening = opening, skipped_begin = matches_begin,
                skipped_end = frame->matches_end;
        }
        position = tag_end + 1, text_begin = position;
    }
    if (current != 0 || skipped_depth || rooted == sz_false_k) return sz_invalid_syntax_k;
    *matches_count = emitted;
    return sz_success_k;
}

#pragma endregion Markup

#pragma region Batches

/** Bytes of the key buffer for a document of @p length bytes: @p key_bytes, or past the document
 *  and the longest pattern tag when `match()` or `search()` may copy one of its strings there. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_buffer_bytes_(sz_subdocuments_engine_t const *engine, sz_size_t key_bytes,
                                                           sz_size_t length) {
    if (!engine->patterns_count) return key_bytes;
    return sz_max_of_two(key_bytes, sz_size_divide_round_up(length + 24, sizeof(sz_u64_t)) * sizeof(sz_u64_t));
}

/** Per-document counts through a tier's classifiers, which is every CPU tier's counting verb. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_counts_with_(sz_subdocuments_engine_t *engine,
                                                            sz_subdocuments_classify_t classify,
                                                            sz_subdocuments_classify_t skim,
                                                            sz_subdocuments_find_byte_t find_byte,
                                                            sz_strings_view_t const *documents, sz_size_t *counts,
                                                            sz_size_t counts_stride, sz_size_t *malformed_index) {
    sz_size_t const key_bytes = sz_subdocuments_key_bytes_(engine);
    sz_u64_t const names_lengths = sz_subdocuments_names_lengths_(engine);
    sz_string_view_t views[64];
    sz_size_t document_index;
    if (!engine->steps_count) return sz_missing_kernel_k;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    for (document_index = 0; document_index != documents->count; ++document_index) {
        sz_size_t counted = 0;
        sz_status_t status;
        if (document_index % 64 == 0) sz_strings_get_block_(documents, document_index, views);
        {
            sz_cptr_t const document = views[document_index % 64].start;
            sz_size_t const length = views[document_index % 64].length;
            sz_size_t const buffer_bytes = sz_subdocuments_buffer_bytes_(engine, key_bytes, length);
            // Most vocabularies fit their steps in one word, where the walk loops over none.
            if (engine->format == sz_subdocuments_xml_k)
                status = sz_subdocuments_walk_markup_(engine, find_byte, document, length, document_index, key_bytes,
                                                      (sz_match_t *)STRINGZILLA_NULL, 0, &counted);
            else if (engine->words_count == 1)
                status = sz_subdocuments_walk_(engine, classify, skim, 1, document, length, document_index,
                                               buffer_bytes, names_lengths, (sz_match_t *)STRINGZILLA_NULL, 0,
                                               &counted);
            else
                status = sz_subdocuments_walk_(engine, classify, skim, engine->words_count, document, length,
                                               document_index, buffer_bytes, names_lengths,
                                               (sz_match_t *)STRINGZILLA_NULL, 0, &counted);
        }
        if (status == sz_invalid_syntax_k) *malformed_index = document_index;
        if (status != sz_success_k) return status;
        counts[document_index * counts_stride] = counted;
    }
    return sz_success_k;
}

/** Every match through a tier's classifiers, which is every CPU tier's locating verb. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_find_with_(
    sz_subdocuments_engine_t *engine, sz_subdocuments_classify_t classify, sz_subdocuments_classify_t skim,
    sz_subdocuments_find_byte_t find_byte, sz_strings_view_t const *documents, sz_match_t *matches,
    sz_size_t matches_capacity, sz_size_t *matches_offsets, sz_size_t *malformed_index) {
    sz_size_t const key_bytes = sz_subdocuments_key_bytes_(engine);
    sz_u64_t const names_lengths = sz_subdocuments_names_lengths_(engine);
    sz_string_view_t views[64];
    sz_size_t document_index, emitted = 0;
    if (!engine->steps_count) return sz_missing_kernel_k;
    for (document_index = 0; document_index != documents->count; ++document_index) {
        sz_status_t status;
        if (document_index % 64 == 0) sz_strings_get_block_(documents, document_index, views);
        matches_offsets[document_index] = emitted;
        {
            sz_cptr_t const document = views[document_index % 64].start;
            sz_size_t const length = views[document_index % 64].length;
            sz_size_t const buffer_bytes = sz_subdocuments_buffer_bytes_(engine, key_bytes, length);
            // Most vocabularies fit their steps in one word, where the walk loops over none.
            if (engine->format == sz_subdocuments_xml_k)
                status = sz_subdocuments_walk_markup_(engine, find_byte, document, length, document_index, key_bytes,
                                                      matches, matches_capacity, &emitted);
            else if (engine->words_count == 1)
                status = sz_subdocuments_walk_(engine, classify, skim, 1, document, length, document_index,
                                               buffer_bytes, names_lengths, matches, matches_capacity, &emitted);
            else
                status = sz_subdocuments_walk_(engine, classify, skim, engine->words_count, document, length,
                                               document_index, buffer_bytes, names_lengths, matches, matches_capacity,
                                               &emitted);
        }
        if (status == sz_invalid_syntax_k) *malformed_index = document_index;
        if (status != sz_success_k) return status;
    }
    matches_offsets[documents->count] = emitted;
    return sz_success_k;
}

#pragma endregion Batches

#pragma region Serial Backends

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_serial(sz_subdocuments_engine_t *engine,
                                                               sz_strings_view_t const *paths,
                                                               sz_subdocuments_format_t format,
                                                               sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                               sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_serial_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_serial(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_size_t *counts,
                                                          sz_size_t counts_stride, sz_size_t *malformed_index,
                                                          sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_serial_, &sz_subdocuments_skim_serial_,
                                        &sz_find_byte_serial_, documents, counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_serial(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_match_t *matches,
                                                        sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                        sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_serial_, &sz_subdocuments_skim_serial_,
                                      &sz_find_byte_serial_, documents, matches, matches_capacity, matches_offsets,
                                      malformed_index);
}

#endif // STRINGZILLA_TARGET_SERIAL

#pragma endregion Serial Backends

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_SUBDOCUMENTS_SERIAL_H_
