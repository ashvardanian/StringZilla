/**
 *  @file include/stringzilla/subdocuments/metal.metal
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Path queries on Apple GPUs of Metal family 7, M1 and newer: one thread walks one document
 *      the way the serial tier does, its depth stack in the round block.
 *
 *  @sa include/stringzilla/subdocuments/metal.h, which embeds and launches this source
 *  @sa include/stringzilla/subdocuments/simt.cuh, the CUDA sibling this mirrors kernel for kernel
 *  @sa include/stringzilla/types.metal, the prelude compiled ahead of this source
 *
 *  The engine's block, its regex engine's block and the round block point inside themselves by host
 *  address, turned into the kernel's own by @ref sz_reach_metal_ against each block's host address.
 *  The scan between counting and storing runs on the host, as Metal's blocks are all shared and the
 *  counting walks read back anyway. MSL has no @c double, so filters hold numbers as binary64 bit
 *  patterns, rounded in integer arithmetic exactly as the serial tier's @c double rounds them.
 */

/** One frame's header, laid out as @c sz_subdocuments_frame_t, its live set following it. */
struct sz_subdocuments_frame_metal_t {
    ulong parent, selector, next_position, byte_offset, byte_length, matches_begin, matches_end, closer, subtree_end;
};

/** Words of a frame ahead of its live set. */
constant ulong sz_subdocuments_frame_header_words_metal_k = sizeof(sz_subdocuments_frame_metal_t) / sizeof(ulong);

/** The name index of an empty slot of the name table. */
constant uint sz_subdocuments_no_name_metal_k = 0xFFFFFFFFu;

/** The slot of no slot, the parent of the root value; also a replay level's
 *  filter before its first. */
constant ulong sz_subdocuments_no_slot_metal_k = ~0ul;

/** The document format of XML, numbered as @c sz_subdocuments_xml_k. */
constant uint sz_subdocuments_xml_metal_k = 1;

/** Frames past which a document the cooperative walk takes goes to it, as
 *  @c sz_subdocuments_deep_frames_simt_k. */
constant ulong sz_subdocuments_deep_frames_metal_k = 1ul << 16;

/** Bytes of each chunk a large document is cut into, as @c sz_subdocuments_chunk_bytes_simt_k. */
constant ulong sz_subdocuments_chunk_bytes_metal_k = 256;

/** Chunks the listed documents span together at most, as @c sz_subdocuments_large_chunks_simt_k. */
constant ulong sz_subdocuments_large_chunks_metal_k = 1ul << 24;

/** The values a filter program holds at most, as @c sz_subdocuments_filter_depth_k bounds them. */
enum { sz_subdocuments_filter_depth_metal_k = 32 };

/** How a walk ends, numbered as the statuses @c sz_subdocuments_walk_simt_ returns. */
enum sz_subdocuments_ending_metal_t : int {
    sz_subdocuments_walked_metal_k = 0,
    sz_subdocuments_too_deep_metal_k = -10,
    sz_subdocuments_malformed_metal_k = -22,
};

/** Where a walk stands in the grammar, numbered as @c sz_subdocuments_expect_t. */
enum sz_subdocuments_expect_metal_t : uint {
    sz_subdocuments_expect_value_metal_k = 0,
    sz_subdocuments_expect_value_or_close_metal_k = 1,
    sz_subdocuments_expect_key_metal_k = 2,
    sz_subdocuments_expect_key_or_close_metal_k = 3,
    sz_subdocuments_expect_colon_metal_k = 4,
    sz_subdocuments_expect_comma_or_close_metal_k = 5,
    sz_subdocuments_expect_end_metal_k = 6,
};

/** What a filter instruction does, numbered as @c sz_subdocuments_operation_t. */
enum sz_subdocuments_operation_metal_t : uint {
    sz_subdocuments_operation_literal_metal_k = 0,
    sz_subdocuments_operation_exists_metal_k = 1,
    sz_subdocuments_operation_value_metal_k = 2,
    sz_subdocuments_operation_count_metal_k = 3,
    sz_subdocuments_operation_length_metal_k = 4,
    sz_subdocuments_operation_match_metal_k = 5,
    sz_subdocuments_operation_search_metal_k = 6,
    sz_subdocuments_operation_compare_metal_k = 7,
    sz_subdocuments_operation_not_metal_k = 8,
    sz_subdocuments_operation_and_metal_k = 9,
    sz_subdocuments_operation_or_metal_k = 10,
};

/** The kind of a value a filter program holds, numbered as @c sz_subdocuments_kind_t. */
enum sz_subdocuments_kind_metal_t : uint {
    sz_subdocuments_kind_nothing_metal_k = 0,
    sz_subdocuments_kind_logical_metal_k = 1,
    sz_subdocuments_kind_null_metal_k = 2,
    sz_subdocuments_kind_false_metal_k = 3,
    sz_subdocuments_kind_true_metal_k = 4,
    sz_subdocuments_kind_number_metal_k = 5,
    sz_subdocuments_kind_string_metal_k = 6,
    sz_subdocuments_kind_array_metal_k = 7,
    sz_subdocuments_kind_object_metal_k = 8,
};

/** The comparisons of filters, numbered as @c sz_subdocuments_comparison_t. */
enum sz_subdocuments_comparison_metal_t : uint {
    sz_subdocuments_comparison_equal_metal_k = 0,
    sz_subdocuments_comparison_not_equal_metal_k = 1,
    sz_subdocuments_comparison_less_metal_k = 2,
    sz_subdocuments_comparison_less_equal_metal_k = 3,
    sz_subdocuments_comparison_greater_metal_k = 4,
    sz_subdocuments_comparison_greater_equal_metal_k = 5,
};

/** What an XPath step meets, numbered as @c sz_subdocuments_node_t. */
enum sz_subdocuments_node_metal_t : uint {
    sz_subdocuments_node_element_metal_k = 0,
    sz_subdocuments_node_attribute_metal_k = 1,
    sz_subdocuments_node_text_metal_k = 2,
};

/** One match, laid out as @c sz_match_t. */
struct sz_match_metal_t {
    ulong haystack_index, needle_index, byte_offset, byte_length;
};

/** The array positions one selector picks, laid out as @c sz_subdocuments_slice_t. */
struct sz_subdocuments_slice_metal_t {
    long begin, end, step;
};

/** The filters of one bracketed selection, laid out as @c sz_subdocuments_filter_t. */
struct sz_subdocuments_filter_metal_t {
    uint step, instructions_begin, instructions_end;
};

/** One instruction of a filter program, laid out as @c sz_subdocuments_instruction_t, the
 *  number as its bits. */
struct sz_subdocuments_instruction_metal_t {
    uint operation, operand;
    ulong number;
    uint string_offset, string_length;
};

/** A value a filter program holds, as @c sz_subdocuments_value_t: a number as its binary64 bits, a
 *  logical value as one or zero. */
struct sz_subdocuments_value_metal_t {
    uint kind;
    bool escaped;
    ulong number, length;
    device uchar const *bytes;
};

/** Where one level of a replay stands, in the header of its spare frame, as
 *  @c sz_subdocuments_level_simt_t. */
struct sz_subdocuments_level_metal_t {
    ulong slot, end, filter;
};

/** The cooperative walk of the listed large documents as the host lays it out, host addresses
 *  inside the round block and its shape, as @c sz_subdocuments_large_simt_t keeps them. */
struct sz_subdocuments_large_metal_t {
    ulong firsts, malformed, transfers, tokens, opens, depths, extremes, sums, matches, tiles, scanned;
    ulong starts, positions, openers, closers, minima, parents, elements, jumps;
    ulong keys, kinds, offsets, selectors, lives, carried;
    ulong listed, bound, chunks, containers_count, threads, key_bytes, levels;
    ulong minima_offsets[8];
    ulong scanned_count, scan_tiles, round;
    uint compose, storing, carried_buffer, descendants;
};

/** The launch record the host copies in: host addresses inside the three blocks,
 *  and the launch's shape. */
struct sz_subdocuments_arguments_metal_t {
    ulong engine_host, scratch_host, regex_host;
    ulong names_data, names_offsets, names_slots, names_steps, wildcard_steps, descendant_steps, relative_steps;
    ulong attribute_steps, text_steps, filter_steps, filters_roots, filters, instructions, strings_data, final_steps;
    ulong root_steps, slices, slices_offsets, steps_paths, byte_to_class, forward_rows;
    ulong header, counted, deep, keys, stacks, large_list;
    ulong documents_count, documents_layout, counts_stride, matches_capacity, items, threads, frames, frame_words;
    ulong key_bytes, large_bytes;
    uint words_count, names_slots_count, filters_count, format, listed, storing;
    uint forward_start, forward_accepts_begin, classes_count;
    sz_subdocuments_large_metal_t large;
};

/** The compiled paths as a walk reads them, with the forward DFA of their regex engine. */
struct sz_subdocuments_engine_metal_t {
    device uchar const *names_data, *strings_data, *byte_to_class;
    device uint const *names_offsets, *names_slots, *slices_offsets, *steps_paths, *forward_rows;
    device ulong const *names_steps, *wildcard_steps, *descendant_steps, *relative_steps, *attribute_steps;
    device ulong const *text_steps, *filter_steps, *filters_roots, *final_steps, *root_steps;
    device sz_subdocuments_slice_metal_t const *slices;
    device sz_subdocuments_filter_metal_t const *filters;
    device sz_subdocuments_instruction_metal_t const *instructions;
    uint words_count, names_slots_count, filters_count, format;
    uint forward_start, forward_accepts_begin, classes_count;
};

/** The masks of one block, and what its classification hands the next, as the
 *  serial tier keeps them. */
struct sz_subdocuments_cursor_metal_t {
    device uchar const *text;
    ulong length, block_offset;
    ulong opens, closes, pending;
    ulong escaped, in_string, scalar;
};

/** Classifies the block at @p offset, padding a partial last one with blanks, as
 *  @c sz_subdocuments_load_simt_ does. */
inline void sz_subdocuments_load_metal_(thread sz_subdocuments_cursor_metal_t &cursor, ulong offset) {
    ulong const available = min(cursor.length - offset, 64ul);
    ulong quotes = 0, backslashes = 0, opens = 0, closes = 0, separators = 0, blanks = 0;
    ulong escaped, in_string, scalars;
    for (ulong index = 0; index != 64; ++index) {
        uchar const byte = index < available ? cursor.text[offset + index] : ' ';
        uchar const folded = byte | 0x20;
        ulong const bit = 1ul << index;
        if (byte == '"') quotes |= bit;
        if (byte == '\\') backslashes |= bit;
        if (folded == '{') opens |= bit;
        if (folded == '}') closes |= bit;
        if (byte == ':' || byte == ',') separators |= bit;
        if (byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r') blanks |= bit;
    }
    if (!backslashes) escaped = cursor.escaped, cursor.escaped = 0;
    else {
        ulong const odd_bits = 0xAAAAAAAAAAAAAAAAul;
        ulong const potential_escape = backslashes & ~cursor.escaped;
        ulong const escape_and_terminal = (((potential_escape << 1) | odd_bits) - potential_escape) ^ odd_bits;
        escaped = escape_and_terminal ^ (backslashes | cursor.escaped);
        cursor.escaped = (escape_and_terminal & backslashes) >> 63;
    }
    quotes &= ~escaped;
    in_string = quotes;
    in_string ^= in_string << 1, in_string ^= in_string << 2, in_string ^= in_string << 4;
    in_string ^= in_string << 8, in_string ^= in_string << 16, in_string ^= in_string << 32;
    in_string ^= cursor.in_string;
    cursor.in_string = 0 - (in_string >> 63);
    opens &= ~in_string, closes &= ~in_string, separators &= ~in_string;
    scalars = ~(in_string | quotes | blanks | opens | closes | separators);
    cursor.block_offset = offset, cursor.opens = opens, cursor.closes = closes;
    cursor.pending = opens | closes | separators | quotes | (scalars & ~(scalars << 1 | cursor.scalar));
    cursor.scalar = scalars >> 63;
}

/** Moves to the next structural byte, or reports the end of the document. */
inline bool sz_subdocuments_next_metal_(thread sz_subdocuments_cursor_metal_t &cursor, thread ulong &position) {
    while (!cursor.pending) {
        if (cursor.length - cursor.block_offset <= 64) return false;
        sz_subdocuments_load_metal_(cursor, cursor.block_offset + 64);
    }
    position = cursor.block_offset + ctz(cursor.pending);
    cursor.pending &= cursor.pending - 1;
    return true;
}

/** Moves past the bracket closing the container just entered, counting brackets kind-blind. */
inline bool sz_subdocuments_skip_metal_(thread sz_subdocuments_cursor_metal_t &cursor, thread ulong &position) {
    ulong depth = 1;
    for (;;) {
        ulong const opens = cursor.opens & cursor.pending, closes = cursor.closes & cursor.pending;
        ulong const closes_count = popcount(closes);
        if (depth > closes_count) depth = depth - closes_count + popcount(opens);
        else
            for (ulong brackets = opens | closes; brackets;) {
                ulong const bit = brackets & (0 - brackets);
                brackets ^= bit;
                if (opens & bit) ++depth;
                else if (--depth == 0) {
                    position = cursor.block_offset + ctz(bit);
                    cursor.pending &= ~(bit | (bit - 1));
                    return true;
                }
            }
        if (cursor.length - cursor.block_offset <= 64) return false;
        sz_subdocuments_load_metal_(cursor, cursor.block_offset + 64);
    }
}

/** Bytes of the scalar at @p position, or zero unless it is a number, @c true,
 *  @c false or @c null. */
inline ulong sz_subdocuments_scalar_length_metal_(device uchar const *text, ulong position, ulong length) {
    device uchar const *const scalar = text + position;
    ulong run = 0, cursor = 0, digits;
    for (; position + run != length; ++run) {
        uchar const byte = scalar[run];
        if (byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' || byte == ',' || byte == ':' || byte == '"' ||
            (byte | 0x20) == '{' || (byte | 0x20) == '}')
            break;
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

/** Four hexadecimal digits, or a value past U+10FFFF if one is not a digit. */
inline uint sz_subdocuments_hex_unit_metal_(device uchar const *digits) {
    uint unit = 0;
    for (uint index = 0; index != 4; ++index) {
        uchar const digit = digits[index], lowered = digit | 0x20;
        if (digit >= '0' && digit <= '9') unit = unit << 4 | uint(digit - '0');
        else if (lowered >= 'a' && lowered <= 'f') unit = unit << 4 | uint(lowered - 'a' + 10);
        else return 0xFFFFFFFFu;
    }
    return unit;
}

/**
 *  @brief Decodes the unit of a `"`-quoted string body at @p source [read], a byte or one escape,
 *      into @p encoded, as @c sz_subdocuments_unescape_unit_simt_ does.
 *  @return Bytes it encodes to, or zero for a malformed one.
 */
inline ulong sz_subdocuments_unescape_unit_metal_(device uchar const *source, ulong length, thread ulong &read,
                                                  thread uchar *encoded) {
    uchar byte = source[read++];
    if (byte < 0x20) return 0;
    if (byte != '\\') return encoded[0] = byte, 1;
    if (read == length) return 0;
    byte = source[read++];
    if (byte == 'b') return encoded[0] = '\b', 1;
    if (byte == 'f') return encoded[0] = '\f', 1;
    if (byte == 'n') return encoded[0] = '\n', 1;
    if (byte == 'r') return encoded[0] = '\r', 1;
    if (byte == 't') return encoded[0] = '\t', 1;
    if (byte != 'u') return byte == '/' || byte == '\\' || byte == '"' ? (encoded[0] = byte, 1) : 0;
    if (length - read < 4) return 0;
    uint rune = sz_subdocuments_hex_unit_metal_(source + read);
    read += 4;
    if (rune > 0xFFFF || (rune >= 0xDC00 && rune <= 0xDFFF)) return 0;
    if (rune >= 0xD800 && rune <= 0xDBFF) {
        if (length - read < 6 || source[read] != '\\' || source[read + 1] != 'u') return 0;
        uint const low = sz_subdocuments_hex_unit_metal_(source + read + 2);
        if (low < 0xDC00 || low > 0xDFFF) return 0;
        read += 6;
        rune = 0x10000 + ((rune - 0xD800) << 10) + (low - 0xDC00);
    }
    if (rune < 0x80) return encoded[0] = uchar(rune), 1;
    if (rune < 0x800) return encoded[0] = uchar(0xC0 | rune >> 6), encoded[1] = uchar(0x80 | (rune & 0x3F)), 2;
    if (rune < 0x10000)
        return encoded[0] = uchar(0xE0 | rune >> 12), encoded[1] = uchar(0x80 | (rune >> 6 & 0x3F)),
               encoded[2] = uchar(0x80 | (rune & 0x3F)), 3;
    return encoded[0] = uchar(0xF0 | rune >> 18), encoded[1] = uchar(0x80 | (rune >> 12 & 0x3F)),
           encoded[2] = uchar(0x80 | (rune >> 6 & 0x3F)), encoded[3] = uchar(0x80 | (rune & 0x3F)), 4;
}

/**
 *  @brief Unescapes a `"`-quoted string body into @p target,
 *      writing only its first @p capacity bytes.
 *  @return The unescaped length, or `~0` for a malformed escape.
 */
inline ulong sz_subdocuments_unescape_metal_(device uchar const *source, ulong length, device uchar *target,
                                             ulong capacity) {
    ulong read = 0, written = 0;
    uchar encoded[4];
    while (read != length) {
        ulong const encoded_length = sz_subdocuments_unescape_unit_metal_(source, length, read, encoded);
        if (!encoded_length) return ~0ul;
        for (ulong index = 0; index != encoded_length; ++index, ++written)
            if (written < capacity) target[written] = encoded[index];
    }
    return written;
}

/** The name a key spells, or @c sz_subdocuments_no_name_metal_k, by the hash and probing of
 *  @c sz_subdocuments_name_slot_ ; keys are read a byte at a time, as nothing aligns them. */
inline ulong sz_subdocuments_name_metal_(thread sz_subdocuments_engine_metal_t const &engine, device uchar const *key,
                                         ulong key_length) {
    ulong const mask = engine.names_slots_count - 1;
    ulong hash = key_length * 0x9E3779B97F4A7C15ul, tail = 0, index = 0;
    for (; index + 8 <= key_length; index += 8) {
        ulong word = 0;
        for (ulong byte = 8; byte-- != 0;) word = word << 8 | key[index + byte];
        hash = (hash ^ word) * 0xFF51AFD7ED558CCDul;
        hash ^= hash >> 32;
    }
    for (; index != key_length; ++index) tail = tail << 8 | key[index];
    hash = (hash ^ tail) * 0xC4CEB9FE1A85EC53ul;
    for (ulong slot = (hash ^ hash >> 32) & mask;; slot = (slot + 1) & mask) {
        uint const name = engine.names_slots[slot];
        if (name == sz_subdocuments_no_name_metal_k) return name;
        if (engine.names_offsets[name + 1] - engine.names_offsets[name] != key_length) continue;
        for (index = 0; index != key_length && engine.names_data[engine.names_offsets[name] + index] == key[index];)
            ++index;
        if (index == key_length) return name;
    }
}

/** Whether @p slice selects element @p position of an array of @p length elements. */
inline bool sz_subdocuments_slice_selects_metal_(sz_subdocuments_slice_metal_t slice, long position, long length) {
    if (slice.step > 0) {
        long const lower = slice.begin >= 0 ? min(slice.begin, length) : max(length + slice.begin, 0l);
        long const upper = slice.end >= 0 ? min(slice.end, length) : max(length + slice.end, 0l);
        return position >= lower && position < upper && (position - lower) % slice.step == 0;
    }
    if (slice.step < 0) {
        long const upper = slice.begin >= 0 ? min(slice.begin, length - 1) : max(length + slice.begin, -1l);
        long const lower = slice.end >= 0 ? min(slice.end, length - 1) : max(length + slice.end, -1l);
        return position > lower && position <= upper && (upper - position) % -slice.step == 0;
    }
    return false;
}

/** Which child of a container a live set advances into. */
enum sz_subdocuments_child_metal_t : uint {
    sz_subdocuments_child_member_metal_k = 0,
    sz_subdocuments_child_element_metal_k = 1,
};

/** Word @p word of the live set of a child of a node live in @p live, joined with @p carried unless
 *  null: as @p child says, a member named @p selector or an element at position @p selector, as
 *  @c sz_subdocuments_child_word_simt_ settles it. */
inline ulong sz_subdocuments_child_word_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                               device ulong const *live, device ulong const *carried, uint word,
                                               thread ulong &carry, sz_subdocuments_child_metal_t child, ulong selector,
                                               ulong length) {
    ulong const parent = live[word] | (carried ? carried[word] : 0);
    ulong const advancing = parent & ~engine.final_steps[word];
    ulong const candidates = advancing << 1 | carry;
    ulong selected;
    carry = advancing >> 63;
    if (child == sz_subdocuments_child_member_metal_k)
        selected = candidates & ((selector != sz_subdocuments_no_name_metal_k
                                      ? engine.names_steps[selector * engine.words_count + word]
                                      : 0) |
                                 engine.wildcard_steps[word]);
    else {
        long const known_length = length ? long(length) : long(selector) + 1;
        ulong const tentative = length ? 0 : candidates & engine.relative_steps[word];
        ulong positional = candidates & ~engine.wildcard_steps[word] & ~tentative;
        selected = (candidates & engine.wildcard_steps[word]) | tentative;
        for (; positional; positional &= positional - 1) {
            ulong const step = word * 64 + ctz(positional);
            for (uint slice = engine.slices_offsets[step]; slice != engine.slices_offsets[step + 1]; ++slice)
                if (sz_subdocuments_slice_selects_metal_(engine.slices[slice], long(selector), known_length)) {
                    selected |= positional & (0 - positional);
                    break;
                }
        }
    }
    return selected | (parent & engine.descendant_steps[word]);
}

/** The live set of an array's element, tentatively for the steps its length settles while
 *  @p length is zero. */
inline void sz_subdocuments_element_live_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                                device ulong const *parent_live, ulong position, ulong length,
                                                device ulong *live) {
    ulong carry = 0;
    for (uint word = 0; word != engine.words_count; ++word)
        live[word] = sz_subdocuments_child_word_metal_(engine, parent_live, nullptr, word, carry,
                                                       sz_subdocuments_child_element_metal_k, position, length);
}

/** Whether a member of an object live in @p parent_live, joined with @p carried unless null, needs
 *  its key looked up. */
inline bool sz_subdocuments_needs_name_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                              device ulong const *parent_live, device ulong const *carried) {
    ulong carry = 0, needed = 0;
    for (uint word = 0; word != engine.words_count; ++word) {
        ulong const advancing = (parent_live[word] | (carried ? carried[word] : 0)) & ~engine.final_steps[word];
        needed |= (advancing << 1 | carry) & ~engine.wildcard_steps[word];
        carry = advancing >> 63;
    }
    return needed != 0;
}

/** The live set of an object's member. */
inline void sz_subdocuments_member_live_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                               device ulong const *parent_live, ulong name, device ulong *live) {
    ulong carry = 0;
    for (uint word = 0; word != engine.words_count; ++word)
        live[word] = sz_subdocuments_child_word_metal_(engine, parent_live, nullptr, word, carry,
                                                       sz_subdocuments_child_member_metal_k, name, 0);
}

/** Makes every filter step advancing into a node select it tentatively, with
 *  its filters' query roots. */
inline void sz_subdocuments_filters_tentative_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                                     device ulong const *parent_live, device ulong *live) {
    uint const words = engine.words_count;
    ulong carried = 0, any = 0;
    for (uint word = 0; word != words; ++word) {
        ulong const advancing = parent_live[word] & ~engine.final_steps[word];
        ulong const filtered = (advancing << 1 | carried) & engine.filter_steps[word];
        live[word] |= filtered, any |= filtered, carried = advancing >> 63;
    }
    if (!any) return;
    for (uint filter = 0; filter != engine.filters_count; ++filter) {
        uint const step = engine.filters[filter].step;
        if (live[step >> 6] >> (step & 63) & 1)
            for (uint word = 0; word != words; ++word) live[word] |= engine.filters_roots[filter * words + word];
    }
}

/** Reports a node once per final step of its @p live set. */
inline void sz_subdocuments_report_metal_(thread sz_subdocuments_engine_metal_t const &engine, device ulong const *live,
                                          ulong document, ulong byte_offset, ulong byte_length,
                                          device sz_match_metal_t *matches, ulong matches_capacity,
                                          thread ulong &emitted) {
    for (uint word = 0; word != engine.words_count; ++word)
        for (ulong reported = live[word] & engine.final_steps[word]; reported; reported &= reported - 1) {
            if (emitted < matches_capacity)
                matches[emitted] = {document, engine.steps_paths[word * 64 + ctz(reported)], byte_offset, byte_length};
            ++emitted;
        }
}

/** Settles the lengths of the matches `[begin, end)` that fit, all starting at one byte. */
inline void sz_subdocuments_close_metal_(device sz_match_metal_t *matches, ulong matches_capacity, ulong begin,
                                         ulong end, ulong end_offset) {
    for (end = min(end, matches_capacity); begin < end; ++begin)
        matches[begin].byte_length = end_offset - matches[begin].byte_offset;
}

/** Whether no step is live in @p live, so a replay skips the subtree below. */
inline bool sz_subdocuments_dead_metal_(device ulong const *live, uint words) {
    ulong any = 0;
    for (uint word = 0; word != words; ++word) any |= live[word];
    return any == 0;
}

/** Whether nonzero bits were dropped past a significand, the sticky bit of IEEE 754 rounding. */
enum sz_subdocuments_f64_tail_metal_t : uint {
    sz_subdocuments_f64_exact_metal_k = 0,
    sz_subdocuments_f64_inexact_metal_k = 1,
};

/** The binary64 bits of significand × 2^exponent for a nonzero @p significand, past which @p tail
 *  says whether nonzero bits were dropped, rounded to nearest even, to a subnormal or to infinity
 *  as IEEE 754 rounds. */
inline ulong sz_subdocuments_f64_pack_metal_(ulong significand, int exponent, sz_subdocuments_f64_tail_metal_t tail) {
    int const lead = 63 - int(clz(significand));
    int biased = exponent + lead + 1023, shift = lead - 52;
    if (biased < 1) shift += 1 - biased, biased = 1;
    if (biased > 2046) return 0x7FF0000000000000ul;
    if (shift >= 64) return 0;
    // The exponent field is added one short, so a mantissa carrying
    // into bit 52 or 53 carries into it.
    ulong bits = (ulong(biased - 1) << 52) + (shift <= 0 ? significand << -shift : significand >> shift);
    if (shift > 0) {
        ulong const dropped = significand & ((1ul << shift) - 1), halfway = 1ul << (shift - 1);
        bits += dropped > halfway ||
                (dropped == halfway && (tail == sz_subdocuments_f64_inexact_metal_k || (bits & 1)));
    }
    return min(bits, 0x7FF0000000000000ul);
}

/** The binary64 bits of @p value. */
inline ulong sz_subdocuments_f64_from_u64_metal_(ulong value) {
    return value ? sz_subdocuments_f64_pack_metal_(value, 0, sz_subdocuments_f64_exact_metal_k) : 0;
}

/** The product of two non-negative binary64 values, NaN for zero times infinity. */
inline ulong sz_subdocuments_f64_multiply_metal_(ulong left, ulong right) {
    ulong const infinity = 0x7FF0000000000000ul, fraction = 0xFFFFFFFFFFFFFul;
    if (left == infinity || right == infinity) return left && right ? infinity : 0x7FF8000000000000ul;
    if (!left || !right) return 0;
    ulong const left_mantissa = (left & fraction) | (left >> 52 ? fraction + 1 : 0);
    ulong const right_mantissa = (right & fraction) | (right >> 52 ? fraction + 1 : 0);
    int const exponent = max(int(left >> 52), 1) + max(int(right >> 52), 1) - 2150;
    ulong const high = mulhi(left_mantissa, right_mantissa), low = left_mantissa * right_mantissa;
    // The 106-bit product's top 64 bits, the 42 below folded into the sticky bit.
    return sz_subdocuments_f64_pack_metal_(
        high << 22 | low >> 42, exponent + 42,
        (low << 22) != 0 ? sz_subdocuments_f64_inexact_metal_k : sz_subdocuments_f64_exact_metal_k);
}

/** The quotient of a finite non-negative binary64 value by one at least one, finite or infinite. */
inline ulong sz_subdocuments_f64_divide_metal_(ulong left, ulong right) {
    ulong const fraction = 0xFFFFFFFFFFFFFul;
    if (!left || right == 0x7FF0000000000000ul) return 0;
    ulong const left_mantissa = (left & fraction) | (left >> 52 ? fraction + 1 : 0);
    ulong const right_mantissa = (right & fraction) | (fraction + 1);
    ulong remainder = left_mantissa, quotient = 0;
    // Long division, a bit at a time, to left_mantissa × 2⁶³ / right_mantissa, below 2⁶⁴.
    for (uint bit = 0; bit != 64; ++bit) {
        quotient = quotient << 1 | (remainder >= right_mantissa);
        if (remainder >= right_mantissa) remainder -= right_mantissa;
        remainder <<= 1;
    }
    return sz_subdocuments_f64_pack_metal_(
        quotient, max(int(left >> 52), 1) - int(right >> 52) - 63,
        remainder != 0 ? sz_subdocuments_f64_inexact_metal_k : sz_subdocuments_f64_exact_metal_k);
}

/** Orders two binary64 values as IEEE 754 compares them: -1, 0 or 1, or 2 when either is NaN. */
inline int sz_subdocuments_f64_order_metal_(ulong left, ulong right) {
    ulong const magnitude = 0x7FFFFFFFFFFFFFFFul, infinity = 0x7FF0000000000000ul, sign = ~magnitude;
    if ((left & magnitude) > infinity || (right & magnitude) > infinity) return 2;
    if (!((left | right) & magnitude)) return 0;
    // Flipping a negative's bits and setting a positive's sign orders the patterns as their values.
    ulong const left_key = left & sign ? ~left : left | sign, right_key = right & sign ? ~right : right | sign;
    return left_key < right_key ? -1 : left_key > right_key ? 1 : 0;
}

/** The bits of the number a validated JSON number spells over @p length bytes, rounded as
 *  @c sz_subdocuments_parse_number_ rounds it. */
inline ulong sz_subdocuments_number_metal_(device uchar const *text, ulong length) {
    ulong cursor = text[0] == '-', mantissa = 0, scale = 0x3FF0000000000000ul;
    long exponent = 0, magnitude = 0;
    if (text[cursor] == '0') ++cursor;
    else
        for (; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (mantissa < 1000000000000000000ul) mantissa = mantissa * 10 + ulong(text[cursor] - '0');
            else ++exponent;
    if (cursor != length && text[cursor] == '.')
        for (++cursor; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (mantissa < 1000000000000000000ul) mantissa = mantissa * 10 + ulong(text[cursor] - '0'), --exponent;
    if (cursor != length && (text[cursor] | 0x20) == 'e') {
        bool const negative_exponent = ++cursor != length && text[cursor] == '-';
        if (cursor != length && (text[cursor] == '+' || text[cursor] == '-')) ++cursor;
        for (; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (magnitude < 100000) magnitude = magnitude * 10 + (text[cursor] - '0');
        exponent += negative_exponent ? -magnitude : magnitude;
    }
    // The serial tier's `scale < 1e308` and `scale *= 10`, on the patterns of both numbers.
    for (magnitude = exponent < 0 ? -exponent : exponent; magnitude > 0 && scale < 0x7FE1CCF385EBC8A0ul; --magnitude)
        scale = sz_subdocuments_f64_multiply_metal_(scale, 0x4024000000000000ul);
    ulong const result = exponent < 0
                             ? sz_subdocuments_f64_divide_metal_(sz_subdocuments_f64_from_u64_metal_(mantissa), scale)
                             : sz_subdocuments_f64_multiply_metal_(sz_subdocuments_f64_from_u64_metal_(mantissa),
                                                                   scale);
    return text[0] == '-' ? result ^ (1ul << 63) : result;
}

/** The value of the node spanning @p length bytes at @p bytes. */
inline sz_subdocuments_value_metal_t sz_subdocuments_node_value_metal_(device uchar const *bytes, ulong length) {
    sz_subdocuments_value_metal_t value = {sz_subdocuments_kind_number_metal_k, false, 0, length, bytes};
    if (bytes[0] == '"')
        value.kind = sz_subdocuments_kind_string_metal_k, ++value.bytes, value.length -= 2, value.escaped = true;
    else if (bytes[0] == '{') value.kind = sz_subdocuments_kind_object_metal_k;
    else if (bytes[0] == '[') value.kind = sz_subdocuments_kind_array_metal_k;
    else if (bytes[0] == 't') value.kind = sz_subdocuments_kind_true_metal_k;
    else if (bytes[0] == 'f') value.kind = sz_subdocuments_kind_false_metal_k;
    else if (bytes[0] == 'n') value.kind = sz_subdocuments_kind_null_metal_k;
    else value.number = sz_subdocuments_number_metal_(bytes, length);
    return value;
}

/** Orders two strings by codepoints, a document's one unescaped into @p buffer. */
inline int sz_subdocuments_order_strings_metal_(thread sz_subdocuments_value_metal_t const &left,
                                                thread sz_subdocuments_value_metal_t const &right, device uchar *buffer,
                                                ulong capacity) {
    device uchar const *left_bytes = left.bytes, *right_bytes = right.bytes;
    ulong left_length = left.length, right_length = right.length;
    if (left.escaped)
        left_length = sz_subdocuments_unescape_metal_(left_bytes, left_length, buffer, capacity), left_bytes = buffer;
    if (right.escaped)
        right_length = sz_subdocuments_unescape_metal_(right_bytes, right_length, buffer, capacity),
        right_bytes = buffer;
    ulong const shared = min(left_length, right_length);
    for (ulong index = 0; index != shared; ++index)
        if (left_bytes[index] != right_bytes[index]) return left_bytes[index] < right_bytes[index] ? -1 : 1;
    return left_length == right_length ? 0 : left_length < right_length ? -1 : 1;
}

/** RFC 9535's comparison of @p left with @p right by @p comparison. */
inline bool sz_subdocuments_compare_metal_(thread sz_subdocuments_value_metal_t const &left,
                                           thread sz_subdocuments_value_metal_t const &right, uint comparison,
                                           device uchar *buffer, ulong capacity) {
    int order = 2;
    if (left.kind == right.kind && left.kind == sz_subdocuments_kind_number_metal_k)
        order = sz_subdocuments_f64_order_metal_(left.number, right.number);
    else if (left.kind == right.kind && left.kind == sz_subdocuments_kind_string_metal_k)
        order = sz_subdocuments_order_strings_metal_(left, right, buffer, capacity);
    else if (left.kind == right.kind) order = 0;
    if (comparison == sz_subdocuments_comparison_equal_metal_k) return order == 0;
    if (comparison == sz_subdocuments_comparison_not_equal_metal_k) return order != 0;
    if (comparison == sz_subdocuments_comparison_less_metal_k) return order == -1;
    if (comparison == sz_subdocuments_comparison_less_equal_metal_k) return order == -1 || order == 0;
    if (comparison == sz_subdocuments_comparison_greater_metal_k) return order == 1;
    return order == 1 || order == 0;
}

/** RFC 9535's `length()`: a string's codepoints, a container's elements or members, or Nothing. */
inline sz_subdocuments_value_metal_t sz_subdocuments_length_metal_(thread sz_subdocuments_value_metal_t const &value) {
    device uchar const *const bytes = value.bytes;
    sz_subdocuments_value_metal_t result = {sz_subdocuments_kind_number_metal_k, false, 0, 0, nullptr};
    ulong count = 0, index;
    if (value.kind == sz_subdocuments_kind_string_metal_k)
        for (index = 0; index < value.length; ++index) {
            if (bytes[index] == '\\' && value.escaped) {
                ++count;
                if (++index + 4 >= value.length || bytes[index] != 'u') continue;
                // A high surrogate's escape and its pair's spell one codepoint.
                uint const unit = sz_subdocuments_hex_unit_metal_(bytes + index + 1);
                index += 4;
                if (unit >= 0xD800 && unit <= 0xDBFF) index += 6;
            }
            else count += (bytes[index] & 0xC0) != 0x80;
        }
    else if (value.kind == sz_subdocuments_kind_array_metal_k || value.kind == sz_subdocuments_kind_object_metal_k) {
        ulong depth = 0;
        bool content = false;
        for (index = 1; index + 1 < value.length; ++index) {
            uchar const byte = bytes[index];
            if (byte == '"') {
                for (++index; index + 1 < value.length && bytes[index] != '"'; ++index) index += bytes[index] == '\\';
                content = true;
            }
            else if (byte == '[' || byte == '{') ++depth, content = true;
            else if (byte == ']' || byte == '}') --depth;
            else if (byte == ',' && depth == 0) ++count;
            else if (byte != ' ' && byte != '\t' && byte != '\n' && byte != '\r') content = true;
        }
        count += content;
    }
    else result.kind = sz_subdocuments_kind_nothing_metal_k;
    result.number = sz_subdocuments_f64_from_u64_metal_(count);
    return result;
}

/** Whether pattern @p pattern matches the string @p value, streaming the pattern's tag and the
 *  unescaped string through the forward DFA, as @c sz_subdocuments_regex_test_simt_ does. */
inline bool sz_subdocuments_regex_test_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                              thread sz_subdocuments_value_metal_t const &value, ulong pattern) {
    device uint const *const rows = engine.forward_rows;
    device uchar const *const classes = engine.byte_to_class;
    ulong power = 1, read = 0;
    uchar encoded[4];
    if (value.kind != sz_subdocuments_kind_string_metal_k) return false;
    // A broken escape in a string no walk validated matches nothing, wherever the match would end.
    if (value.escaped && sz_subdocuments_unescape_metal_(value.bytes, value.length, nullptr, 0) == ~0ul) return false;
    uint state = rows[engine.forward_start + classes[1]];
    while (pattern / power >= 10) power *= 10;
    for (; power; power /= 10) state = rows[state + classes['0' + pattern / power % 10]];
    state = rows[state + classes[2]];
    while (read != value.length) {
        ulong const encoded_length = value.escaped ? sz_subdocuments_unescape_unit_metal_(value.bytes, value.length,
                                                                                          read, encoded)
                                                   : (encoded[0] = value.bytes[read++], 1ul);
        for (ulong index = 0; index != encoded_length; ++index) {
            state = rows[state + classes[encoded[index]]];
            if (state >= engine.forward_accepts_begin) return true;
            if (!state) return false;
        }
    }
    return rows[state + engine.classes_count] >= engine.forward_accepts_begin;
}

/** Runs filter @p filter for the node in slot @p node over the slots
 *  `[node, end)` of its subtree. */
inline bool sz_subdocuments_filter_holds_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                                device uchar const *text, device ulong const *stack, ulong frame_words,
                                                ulong node, ulong end, sz_subdocuments_filter_metal_t filter,
                                                device uchar *buffer, ulong capacity) {
    ulong const header_words = sz_subdocuments_frame_header_words_metal_k;
    sz_subdocuments_value_metal_t values[sz_subdocuments_filter_depth_metal_k];
    uint depth = 0;
    for (uint index = filter.instructions_begin; index != filter.instructions_end; ++index) {
        sz_subdocuments_instruction_metal_t const instruction = engine.instructions[index];
        uint const operation = instruction.operation;
        thread sz_subdocuments_value_metal_t &top = values[depth];
        if (operation == sz_subdocuments_operation_literal_metal_k) {
            top.kind = instruction.operand, top.number = instruction.number, top.escaped = false;
            top.bytes = engine.strings_data + instruction.string_offset, top.length = instruction.string_length;
            ++depth;
        }
        else if (operation == sz_subdocuments_operation_exists_metal_k ||
                 operation == sz_subdocuments_operation_value_metal_k ||
                 operation == sz_subdocuments_operation_count_metal_k) {
            uint const step = instruction.operand;
            ulong found = 0, first = 0;
            // A dead slot's subtree was not replayed and holds stale live sets.
            for (ulong slot = node; slot < end; ++slot)
                if (sz_subdocuments_dead_metal_(stack + slot * frame_words + header_words, engine.words_count))
                    slot = ((device sz_subdocuments_frame_metal_t const *)(stack + slot * frame_words))->subtree_end -
                           1;
                else if (stack[slot * frame_words + header_words + (step >> 6)] >> (step & 63) & 1)
                    first = found++ ? first : slot;
            top.bytes = nullptr, top.length = 0, top.escaped = false;
            if (operation == sz_subdocuments_operation_value_metal_k) {
                device sz_subdocuments_frame_metal_t const *const frame =
                    (device sz_subdocuments_frame_metal_t const *)(stack + first * frame_words);
                if (found == 1) top = sz_subdocuments_node_value_metal_(text + frame->byte_offset, frame->byte_length);
                else top.kind = sz_subdocuments_kind_nothing_metal_k;
            }
            else if (operation == sz_subdocuments_operation_count_metal_k)
                top.kind = sz_subdocuments_kind_number_metal_k, top.number = sz_subdocuments_f64_from_u64_metal_(found);
            else top.kind = sz_subdocuments_kind_logical_metal_k, top.number = found != 0;
            ++depth;
        }
        else if (operation == sz_subdocuments_operation_length_metal_k)
            values[depth - 1] = sz_subdocuments_length_metal_(values[depth - 1]);
        else if (operation == sz_subdocuments_operation_match_metal_k ||
                 operation == sz_subdocuments_operation_search_metal_k) {
            bool const matched = sz_subdocuments_regex_test_metal_(engine, values[depth - 1], instruction.operand);
            values[depth - 1].kind = sz_subdocuments_kind_logical_metal_k, values[depth - 1].number = matched;
        }
        else if (operation == sz_subdocuments_operation_compare_metal_k) {
            --depth;
            values[depth - 1].number = sz_subdocuments_compare_metal_(values[depth - 1], values[depth],
                                                                      instruction.operand, buffer, capacity);
            values[depth - 1].kind = sz_subdocuments_kind_logical_metal_k;
        }
        else if (operation == sz_subdocuments_operation_not_metal_k)
            values[depth - 1].number = values[depth - 1].number == 0;
        else if (operation == sz_subdocuments_operation_and_metal_k)
            --depth, values[depth - 1].number = values[depth - 1].number != 0 && values[depth].number != 0;
        else --depth, values[depth - 1].number = values[depth - 1].number != 0 || values[depth].number != 0;
    }
    return values[0].number != 0;
}

/** Settles the slots `[begin, end)` of a closed pending run, each filter's replay of a node's
 *  subtree one level up, as @c sz_subdocuments_replay_simt_ does. */
inline void sz_subdocuments_replay_metal_(thread sz_subdocuments_engine_metal_t const &engine, device uchar const *text,
                                          device ulong *stack, ulong frame_words, ulong begin, ulong end, ulong spare,
                                          device uchar *buffer, ulong capacity, ulong document,
                                          device sz_match_metal_t *matches, ulong matches_capacity,
                                          thread ulong &emitted) {
    ulong const header_words = sz_subdocuments_frame_header_words_metal_k;
    uint const words = engine.words_count;
    device sz_subdocuments_level_metal_t *state = (device sz_subdocuments_level_metal_t *)(stack + spare * frame_words);
    ulong level = 0;
    state->slot = begin, state->end = end, state->filter = sz_subdocuments_no_slot_metal_k;
    for (;;) {
        device ulong *const results = (device ulong *)state + header_words;
        ulong const slot = state->slot;
        device sz_subdocuments_frame_metal_t const *const frame =
            (device sz_subdocuments_frame_metal_t const *)(stack + slot * frame_words);
        if (slot == state->end) {
            if (!level) return;
            // A filter's replay ended, so the level below decides that filter at its
            // node and moves on.
            state = (device sz_subdocuments_level_metal_t *)(stack + (spare + --level) * frame_words);
            device ulong *const level_results = (device ulong *)state + header_words;
            sz_subdocuments_filter_metal_t const filter = engine.filters[state->filter++];
            device sz_subdocuments_frame_metal_t const *const node =
                (device sz_subdocuments_frame_metal_t const *)(stack + state->slot * frame_words);
            if (sz_subdocuments_filter_holds_metal_(engine, text, stack, frame_words, state->slot, node->subtree_end,
                                                    filter, buffer, capacity))
                level_results[filter.step >> 6] |= 1ul << (filter.step & 63);
            continue;
        }
        device sz_subdocuments_frame_metal_t const *const parent =
            (device sz_subdocuments_frame_metal_t const *)(stack + frame->parent * frame_words);
        device ulong const *const parent_live = (device ulong const *)parent + header_words;
        device ulong *const live = stack + slot * frame_words + header_words;
        if (state->filter == sz_subdocuments_no_slot_metal_k) {
            for (uint word = 0; word != words; ++word) results[word] = 0;
            state->filter = 0;
        }
        // A filter step advances into the node when the step before it is live and not final.
        for (; state->filter != engine.filters_count; ++state->filter) {
            uint const step = engine.filters[state->filter].step, before = step - 1;
            if ((parent_live[before >> 6] & ~engine.final_steps[before >> 6]) >> (before & 63) & 1 &&
                !(results[step >> 6] >> (step & 63) & 1))
                break;
        }
        if (state->filter != engine.filters_count) {
            for (uint word = 0; word != words; ++word) live[word] = engine.filters_roots[state->filter * words + word];
            state = (device sz_subdocuments_level_metal_t *)(stack + (spare + ++level) * frame_words);
            state->slot = slot + 1, state->end = frame->subtree_end, state->filter = sz_subdocuments_no_slot_metal_k;
            continue;
        }
        if (parent->closer == ']')
            sz_subdocuments_element_live_metal_(engine, parent_live, frame->selector, parent->next_position, live);
        else sz_subdocuments_member_live_metal_(engine, parent_live, frame->selector, live);
        for (uint word = 0; word != words; ++word) live[word] |= results[word];
        if (!level)
            sz_subdocuments_report_metal_(engine, live, document, frame->byte_offset, frame->byte_length, matches,
                                          matches_capacity, emitted);
        state->slot = sz_subdocuments_dead_metal_(live, words) ? frame->subtree_end : slot + 1;
        state->filter = sz_subdocuments_no_slot_metal_k;
    }
}

/** Walks one JSON document as @c sz_subdocuments_walk_simt_ does, emitting its matches
 *  from @p emitted on. */
inline int sz_subdocuments_walk_metal_(thread sz_subdocuments_engine_metal_t const &engine, device uchar const *text,
                                       ulong length, ulong document, device ulong *stack, ulong frames,
                                       ulong frame_words, device uchar *key, ulong key_bytes,
                                       device sz_match_metal_t *matches, ulong matches_capacity,
                                       thread ulong &emitted) {
    ulong const header_words = sz_subdocuments_frame_header_words_metal_k;
    uint const words = engine.words_count;
    uint expect = sz_subdocuments_expect_value_metal_k;
    ulong top = 0, current = sz_subdocuments_no_slot_metal_k, run = sz_subdocuments_no_slot_metal_k;
    ulong position, closing;
    sz_subdocuments_cursor_metal_t cursor;
    for (uint word = 0; word != words; ++word) stack[header_words + word] = engine.root_steps[word];
    cursor.text = text, cursor.length = length, cursor.escaped = 0, cursor.in_string = 0, cursor.scalar = 0;
    sz_subdocuments_load_metal_(cursor, 0);

    while (sz_subdocuments_next_metal_(cursor, position)) {
        uchar const byte = text[position];
        device sz_subdocuments_frame_metal_t *const parent =
            current != sz_subdocuments_no_slot_metal_k
                ? (device sz_subdocuments_frame_metal_t *)(stack + current * frame_words)
                : (device sz_subdocuments_frame_metal_t *)nullptr;
        device ulong const *const parent_live = parent ? (device ulong const *)parent + header_words : nullptr;
        device sz_subdocuments_frame_metal_t *const frame = (device sz_subdocuments_frame_metal_t *)(stack +
                                                                                                     top * frame_words);
        device ulong *const live = (device ulong *)frame + header_words;
        ulong reported = 0, continuing = 0, end;

        if (byte == ':') {
            if (expect != sz_subdocuments_expect_colon_metal_k) return sz_subdocuments_malformed_metal_k;
            expect = sz_subdocuments_expect_value_metal_k;
            continue;
        }
        if (byte == ',') {
            if (expect != sz_subdocuments_expect_comma_or_close_metal_k) return sz_subdocuments_malformed_metal_k;
            expect = parent->closer == '}' ? sz_subdocuments_expect_key_metal_k : sz_subdocuments_expect_value_metal_k;
            continue;
        }
        if (byte == '}' || byte == ']') {
            if (!parent || parent->closer != byte) return sz_subdocuments_malformed_metal_k;
            if (expect != sz_subdocuments_expect_comma_or_close_metal_k &&
                expect != (byte == '}' ? sz_subdocuments_expect_key_or_close_metal_k
                                       : sz_subdocuments_expect_value_or_close_metal_k))
                return sz_subdocuments_malformed_metal_k;
            parent->byte_length = position + 1 - parent->byte_offset;
            closing = parent->parent;
            if (run == sz_subdocuments_no_slot_metal_k || run == current) {
                sz_subdocuments_close_metal_(matches, matches_capacity, parent->matches_begin, parent->matches_end,
                                             position + 1);
                if (run == current) {
                    // Filters replay at one spare slot per nesting level past the kept ones.
                    if (top + engine.filters_count + 2 > frames) return sz_subdocuments_too_deep_metal_k;
                    // Kept slots are in document order, so a subtree ends one
                    // past its last descendant.
                    for (ulong slot = run + 1; slot != top; ++slot)
                        ((device sz_subdocuments_frame_metal_t *)(stack + slot * frame_words))->subtree_end = slot + 1;
                    for (ulong slot = top - 1; slot > run + 1; --slot) {
                        device sz_subdocuments_frame_metal_t const *const node =
                            (device sz_subdocuments_frame_metal_t const *)(stack + slot * frame_words);
                        device sz_subdocuments_frame_metal_t *const holder =
                            (device sz_subdocuments_frame_metal_t *)(stack + node->parent * frame_words);
                        if (node->parent != run) holder->subtree_end = max(holder->subtree_end, node->subtree_end);
                    }
                    sz_subdocuments_replay_metal_(engine, text, stack, frame_words, run + 1, top, top, key, key_bytes,
                                                  document, matches, matches_capacity, emitted);
                    run = sz_subdocuments_no_slot_metal_k;
                }
                top = current;
            }
            current = closing;
            expect = current != sz_subdocuments_no_slot_metal_k ? sz_subdocuments_expect_comma_or_close_metal_k
                                                                : sz_subdocuments_expect_end_metal_k;
            continue;
        }

        if (byte == '"' &&
            (expect == sz_subdocuments_expect_key_metal_k || expect == sz_subdocuments_expect_key_or_close_metal_k)) {
            device uchar const *name_bytes = text + position + 1;
            ulong name = sz_subdocuments_no_name_metal_k;
            if (!sz_subdocuments_next_metal_(cursor, closing)) return sz_subdocuments_malformed_metal_k;
            ulong name_length = closing - position - 1;
            if (engine.names_slots_count && sz_subdocuments_needs_name_metal_(engine, parent_live, nullptr)) {
                ulong index = 0;
                while (index != name_length && name_bytes[index] != '\\') ++index;
                if (index != name_length) {
                    name_length = sz_subdocuments_unescape_metal_(name_bytes, name_length, key, key_bytes);
                    if (name_length == ~0ul) return sz_subdocuments_malformed_metal_k;
                    name_bytes = key;
                }
                if (name_length <= key_bytes) name = sz_subdocuments_name_metal_(engine, name_bytes, name_length);
            }
            sz_subdocuments_member_live_metal_(engine, parent_live, name, live);
            if (engine.filters_count) sz_subdocuments_filters_tentative_metal_(engine, parent_live, live);
            frame->selector = name;
            expect = sz_subdocuments_expect_colon_metal_k;
            continue;
        }

        if (expect != sz_subdocuments_expect_value_metal_k && expect != sz_subdocuments_expect_value_or_close_metal_k)
            return sz_subdocuments_malformed_metal_k;
        if (parent && parent->closer == ']') {
            frame->selector = parent->next_position++;
            sz_subdocuments_element_live_metal_(engine, parent_live, frame->selector, 0, live);
            if (engine.filters_count) sz_subdocuments_filters_tentative_metal_(engine, parent_live, live);
        }
        for (uint word = 0; word != words; ++word)
            reported |= live[word] & engine.final_steps[word], continuing |= live[word] & ~engine.final_steps[word];
        ulong const matches_begin = emitted;
        if (run == sz_subdocuments_no_slot_metal_k)
            sz_subdocuments_report_metal_(engine, live, document, position, 0, matches, matches_capacity, emitted);
        if ((byte == '{' || byte == '[') && continuing) {
            frame->parent = current, frame->byte_offset = position;
            frame->next_position = 0, frame->matches_begin = matches_begin, frame->matches_end = emitted;
            frame->closer = byte == '{' ? '}' : ']';
            // A container opens a pending run when a step that waits may advance
            // into what it holds.
            if (run == sz_subdocuments_no_slot_metal_k) {
                ulong waiting = 0, carried = 0;
                for (uint word = 0; word != words; ++word) {
                    ulong const advancing = live[word] & ~engine.final_steps[word];
                    waiting |= (advancing << 1 | carried) &
                               ((byte == '[' ? engine.relative_steps[word] : 0) | engine.filter_steps[word]);
                    carried = advancing >> 63;
                }
                if (waiting) run = top;
            }
            current = top++;
            // The next value's live set takes the slot past the last, so that one must fit as well.
            if (top + 1 > frames) return sz_subdocuments_too_deep_metal_k;
            expect = byte == '{' ? sz_subdocuments_expect_key_or_close_metal_k
                                 : sz_subdocuments_expect_value_or_close_metal_k;
            continue;
        }
        if (byte == '{' || byte == '[') {
            if (!sz_subdocuments_skip_metal_(cursor, closing)) return sz_subdocuments_malformed_metal_k;
            end = closing + 1;
        }
        else if (byte == '"') {
            if (!sz_subdocuments_next_metal_(cursor, closing)) return sz_subdocuments_malformed_metal_k;
            end = closing + 1;
        }
        else {
            end = position + sz_subdocuments_scalar_length_metal_(text, position, length);
            if (end == position) return sz_subdocuments_malformed_metal_k;
        }
        if (run == sz_subdocuments_no_slot_metal_k)
            sz_subdocuments_close_metal_(matches, matches_capacity, matches_begin, emitted, end);
        else if (reported) {
            frame->parent = current, frame->byte_offset = position, frame->byte_length = end - position;
            frame->closer = 0;
            if (++top + 1 > frames) return sz_subdocuments_too_deep_metal_k;
        }
        expect = current != sz_subdocuments_no_slot_metal_k ? sz_subdocuments_expect_comma_or_close_metal_k
                                                            : sz_subdocuments_expect_end_metal_k;
    }
    return current != sz_subdocuments_no_slot_metal_k || expect != sz_subdocuments_expect_end_metal_k
               ? sz_subdocuments_malformed_metal_k
               : sz_subdocuments_walked_metal_k;
}

/** The first @p byte at or after @p from, or @p from past the last. */
inline ulong sz_subdocuments_markup_find_metal_(device uchar const *text, ulong length, ulong from, uchar byte) {
    while (from < length && text[from] != byte) ++from;
    return from;
}

/** The `>` ending a comment, CDATA section or processing instruction whose body starts at
 *  @p from, or @p length. */
inline ulong sz_subdocuments_markup_end_metal_(device uchar const *text, ulong length, ulong from, uchar terminator,
                                               ulong terminator_length) {
    for (ulong candidate = from;; ++candidate) {
        candidate = sz_subdocuments_markup_find_metal_(text, length, candidate, '>');
        if (candidate >= length) return length;
        if (candidate >= from + terminator_length && text[candidate - 1] == terminator &&
            text[candidate - terminator_length] == terminator)
            return candidate;
    }
}

/** The first byte at or after @p position that is not XML blank space. */
inline ulong sz_subdocuments_markup_skip_blanks_metal_(device uchar const *text, ulong length, ulong position) {
    while (position != length &&
           (text[position] == ' ' || text[position] == '\t' || text[position] == '\n' || text[position] == '\r'))
        ++position;
    return position;
}

/** The first byte at or after @p position that cannot continue an XML name. */
inline ulong sz_subdocuments_markup_skip_name_metal_(device uchar const *text, ulong length, ulong position) {
    for (; position != length; ++position) {
        uchar const byte = text[position], lowered = byte | 0x20;
        if (!((lowered >= 'a' && lowered <= 'z') || (byte >= '0' && byte <= ':') || byte == '_' || byte == '-' ||
              byte == '.' || byte >= 0x80))
            break;
    }
    return position;
}

/** The live set of an XML node from its parent element's, counting positional
 *  steps in @p counters. */
inline void sz_subdocuments_node_live_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                             device ulong const *parent_live, device uint *counters, uint kind,
                                             ulong name, device ulong *live) {
    ulong carried = 0;
    for (uint word = 0; word != engine.words_count; ++word) {
        ulong const advancing = parent_live[word] & ~engine.final_steps[word];
        ulong const candidates = advancing << 1 | carried;
        ulong const named = name != sz_subdocuments_no_name_metal_k
                                ? engine.names_steps[name * engine.words_count + word]
                                : 0;
        ulong tested = kind == sz_subdocuments_node_text_metal_k ? candidates & engine.text_steps[word]
                       : kind == sz_subdocuments_node_attribute_metal_k
                           ? candidates & (named | engine.wildcard_steps[word]) & engine.attribute_steps[word]
                           : candidates & (named | engine.wildcard_steps[word]) & ~engine.attribute_steps[word];
        carried = advancing >> 63;
        for (ulong counted = tested; counted; counted &= counted - 1) {
            ulong const step = word * 64 + ctz(counted);
            uint const slice = engine.slices_offsets[step];
            if (slice == engine.slices_offsets[step + 1]) continue;
            if (!sz_subdocuments_slice_selects_metal_(engine.slices[slice], long(counters[slice]),
                                                      long(counters[slice]) + 1))
                tested &= ~(counted & (0 - counted));
            ++counters[slice];
        }
        live[word] = tested |
                     (kind == sz_subdocuments_node_element_metal_k ? parent_live[word] & engine.descendant_steps[word]
                                                                   : 0);
    }
}

/** The name index of the @p name_length bytes at @p name, or @c sz_subdocuments_no_name_metal_k,
 *  for a walk that needs it. */
inline ulong sz_subdocuments_markup_name_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                                device ulong const *parent_live, device uchar const *name,
                                                ulong name_length, ulong key_bytes) {
    if (!engine.names_slots_count || name_length > key_bytes ||
        !sz_subdocuments_needs_name_metal_(engine, parent_live, nullptr))
        return sz_subdocuments_no_name_metal_k;
    return sz_subdocuments_name_metal_(engine, name, name_length);
}

/** Walks one XML document as @c sz_subdocuments_walk_markup_simt_ does, emitting its matches
 *  from @p emitted on. */
inline int sz_subdocuments_walk_markup_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                              device uchar const *text, ulong length, ulong document,
                                              device ulong *stack, ulong frames, ulong frame_words, ulong key_bytes,
                                              device sz_match_metal_t *matches, ulong matches_capacity,
                                              thread ulong &emitted) {
    ulong const header_words = sz_subdocuments_frame_header_words_metal_k;
    uint const words = engine.words_count;
    ulong const counter_words = frame_words - header_words - words;
    ulong top = 1, current = 0, position = 0, text_begin = 0, skipped_depth = 0, skipped_begin = 0;
    ulong skipped_end = 0, skipped_opening = 0, opening, tag_end;
    bool rooted = false;
    device sz_subdocuments_frame_metal_t *frame = (device sz_subdocuments_frame_metal_t *)stack;
    device ulong *live = stack + header_words;
    frame->parent = sz_subdocuments_no_slot_metal_k;
    for (uint word = 0; word != words; ++word) live[word] = engine.root_steps[word];
    for (ulong word = 0; word != counter_words; ++word) live[words + word] = 0;

    for (;;) {
        device sz_subdocuments_frame_metal_t *const parent =
            (device sz_subdocuments_frame_metal_t *)(stack + current * frame_words);
        device ulong const *const parent_live = (device ulong const *)parent + header_words;
        device uint *const parent_counters = (device uint *)(parent_live + words);
        opening = sz_subdocuments_markup_find_metal_(text, length, position, '<');
        // Outside the root element, character data may only be blank space.
        if (current == 0 && skipped_depth == 0 &&
            sz_subdocuments_markup_skip_blanks_metal_(text, opening, position) != opening)
            return sz_subdocuments_malformed_metal_k;
        if (opening == length) break;
        if (length - opening < 2) return sz_subdocuments_malformed_metal_k;
        uchar const kind = text[opening + 1];

        // A CDATA section is character data, so the text node around it goes on.
        if (kind == '!' && length - opening >= 9 && text[opening + 2] == '[' && text[opening + 3] == 'C' &&
            text[opening + 4] == 'D' && text[opening + 5] == 'A' && text[opening + 6] == 'T' &&
            text[opening + 7] == 'A' && text[opening + 8] == '[') {
            if (current == 0 && skipped_depth == 0) return sz_subdocuments_malformed_metal_k;
            tag_end = sz_subdocuments_markup_end_metal_(text, length, opening + 9, ']', 2);
            if (tag_end == length) return sz_subdocuments_malformed_metal_k;
            position = tag_end + 1;
            continue;
        }
        // Every other markup ends a text node, which is any character data or CDATA between markup.
        if (current != 0 && skipped_depth == 0 && opening != text_begin) {
            frame = (device sz_subdocuments_frame_metal_t *)(stack + top * frame_words);
            live = (device ulong *)frame + header_words;
            sz_subdocuments_node_live_metal_(engine, parent_live, parent_counters, sz_subdocuments_node_text_metal_k,
                                             sz_subdocuments_no_name_metal_k, live);
            sz_subdocuments_report_metal_(engine, live, document, text_begin, opening - text_begin, matches,
                                          matches_capacity, emitted);
        }

        if (kind == '!' && length - opening >= 4 && text[opening + 2] == '-' && text[opening + 3] == '-') {
            tag_end = sz_subdocuments_markup_end_metal_(text, length, opening + 4, '-', 2);
            if (tag_end == length) return sz_subdocuments_malformed_metal_k;
        }
        else if (kind == '?') {
            tag_end = sz_subdocuments_markup_end_metal_(text, length, opening + 2, '?', 1);
            if (tag_end == length) return sz_subdocuments_malformed_metal_k;
        }
        else if (kind == '!') {
            // An internal subset could declare entities that no walk expands.
            ulong const name_end = sz_subdocuments_markup_skip_name_metal_(text, length, opening + 2);
            tag_end = sz_subdocuments_markup_find_metal_(text, length, name_end, '>');
            if (current != 0 || rooted || name_end - opening != 9 || text[opening + 2] != 'D' ||
                text[opening + 3] != 'O' || text[opening + 4] != 'C' || text[opening + 5] != 'T' ||
                text[opening + 6] != 'Y' || text[opening + 7] != 'P' || text[opening + 8] != 'E' || tag_end == length)
                return sz_subdocuments_malformed_metal_k;
            for (ulong index = name_end; index != tag_end; ++index)
                if (text[index] == '[') return sz_subdocuments_malformed_metal_k;
        }
        else if (kind == '/') {
            ulong const name_end = sz_subdocuments_markup_skip_name_metal_(text, length, opening + 2);
            ulong const name_length = name_end - opening - 2;
            tag_end = sz_subdocuments_markup_skip_blanks_metal_(text, length, name_end);
            if (tag_end == length || text[tag_end] != '>' || name_length == 0) return sz_subdocuments_malformed_metal_k;
            // Inside a skipped element only nesting counts; the element's own end
            // tag repeats its name.
            if (skipped_depth > 1) --skipped_depth;
            else {
                if (skipped_depth == 0 && current == 0) return sz_subdocuments_malformed_metal_k;
                ulong const open_name = (skipped_depth ? skipped_opening : parent->byte_offset) + 1;
                if (sz_subdocuments_markup_skip_name_metal_(text, length, open_name) - open_name != name_length)
                    return sz_subdocuments_malformed_metal_k;
                for (ulong index = 0; index != name_length; ++index)
                    if (text[open_name + index] != text[opening + 2 + index]) return sz_subdocuments_malformed_metal_k;
                if (skipped_depth)
                    sz_subdocuments_close_metal_(matches, matches_capacity, skipped_begin, skipped_end, tag_end + 1),
                        skipped_depth = 0;
                else {
                    sz_subdocuments_close_metal_(matches, matches_capacity, parent->matches_begin, parent->matches_end,
                                                 tag_end + 1);
                    top = current, current = parent->parent;
                }
            }
        }
        else {
            ulong const name_end = sz_subdocuments_markup_skip_name_metal_(text, length, opening + 1);
            ulong const matches_begin = emitted;
            ulong attribute = name_end, continuing = 0;
            device uint *counters = nullptr;
            bool empty;
            if (name_end == opening + 1) return sz_subdocuments_malformed_metal_k;
            if (skipped_depth == 0) {
                if (current == 0 && rooted) return sz_subdocuments_malformed_metal_k;
                rooted = true;
                frame = (device sz_subdocuments_frame_metal_t *)(stack + top * frame_words);
                live = (device ulong *)frame + header_words;
                counters = (device uint *)(live + words);
                sz_subdocuments_node_live_metal_(
                    engine, parent_live, parent_counters, sz_subdocuments_node_element_metal_k,
                    sz_subdocuments_markup_name_metal_(engine, parent_live, text + opening + 1, name_end - opening - 1,
                                                       key_bytes),
                    live);
                sz_subdocuments_report_metal_(engine, live, document, opening, 0, matches, matches_capacity, emitted);
                for (uint word = 0; word != words; ++word) continuing |= live[word] & ~engine.final_steps[word];
                for (ulong word = 0; word != counter_words; ++word) live[words + word] = 0;
                frame->parent = current, frame->byte_offset = opening;
                frame->matches_begin = matches_begin, frame->matches_end = emitted;
            }
            // Attributes are read even where none is selected, as their values may hold any `>`.
            for (;;) {
                attribute = sz_subdocuments_markup_skip_blanks_metal_(text, length, attribute);
                if (attribute == length) return sz_subdocuments_malformed_metal_k;
                if (text[attribute] == '>') {
                    empty = false, tag_end = attribute;
                    break;
                }
                if (text[attribute] == '/') {
                    if (attribute + 1 == length || text[attribute + 1] != '>') return sz_subdocuments_malformed_metal_k;
                    empty = true, tag_end = attribute + 1;
                    break;
                }
                ulong const attribute_end = sz_subdocuments_markup_skip_name_metal_(text, length, attribute);
                ulong value_begin = sz_subdocuments_markup_skip_blanks_metal_(text, length, attribute_end);
                if (attribute_end == attribute || value_begin == length || text[value_begin] != '=')
                    return sz_subdocuments_malformed_metal_k;
                value_begin = sz_subdocuments_markup_skip_blanks_metal_(text, length, value_begin + 1);
                if (value_begin == length || (text[value_begin] != '"' && text[value_begin] != '\''))
                    return sz_subdocuments_malformed_metal_k;
                uchar const quote = text[value_begin++];
                ulong const value_end = sz_subdocuments_markup_find_metal_(text, length, value_begin, quote);
                if (value_end == length) return sz_subdocuments_malformed_metal_k;
                // Namespace declarations are no attributes in XPath's data model.
                if (continuing &&
                    !(attribute_end - attribute >= 5 && text[attribute] == 'x' && text[attribute + 1] == 'm' &&
                      text[attribute + 2] == 'l' && text[attribute + 3] == 'n' && text[attribute + 4] == 's' &&
                      (attribute_end - attribute == 5 || text[attribute + 5] == ':'))) {
                    device ulong *const attribute_live = live + words + counter_words;
                    sz_subdocuments_node_live_metal_(
                        engine, live, counters, sz_subdocuments_node_attribute_metal_k,
                        sz_subdocuments_markup_name_metal_(engine, live, text + attribute, attribute_end - attribute,
                                                           key_bytes),
                        attribute_live);
                    sz_subdocuments_report_metal_(engine, attribute_live, document, value_begin,
                                                  value_end - value_begin, matches, matches_capacity, emitted);
                }
                attribute = value_end + 1;
            }
            if (skipped_depth) skipped_depth += !empty;
            else if (empty)
                sz_subdocuments_close_metal_(matches, matches_capacity, matches_begin, frame->matches_end, tag_end + 1);
            else if (continuing) {
                current = top++;
                // The next element and an attribute's live set take the two slots past the last.
                if (top + 2 > frames) return sz_subdocuments_too_deep_metal_k;
            }
            else
                skipped_depth = 1, skipped_opening = opening, skipped_begin = matches_begin,
                skipped_end = frame->matches_end;
        }
        position = tag_end + 1, text_begin = position;
    }
    return current != 0 || skipped_depth || !rooted ? sz_subdocuments_malformed_metal_k
                                                    : sz_subdocuments_walked_metal_k;
}

/** Frames a walk of the @p length bytes at @p text needs at most, as
 *  @c sz_subdocuments_frames_simt_ bounds them. */
inline ulong sz_subdocuments_frames_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                           device uchar const *text, ulong length) {
    ulong depth = 0, deepest = 0, structurals = 0, position = 0, relative = 0;
    sz_subdocuments_cursor_metal_t cursor;
    if (engine.format == sz_subdocuments_xml_metal_k) {
        for (; (position = sz_subdocuments_markup_find_metal_(text, length, position, '<')) + 1 < length; ++position) {
            uchar const kind = text[position + 1];
            if (kind == '!' && length - position >= 9 && text[position + 2] == '[' && text[position + 3] == 'C' &&
                text[position + 4] == 'D' && text[position + 5] == 'A' && text[position + 6] == 'T' &&
                text[position + 7] == 'A' && text[position + 8] == '[')
                position = sz_subdocuments_markup_end_metal_(text, length, position + 9, ']', 2);
            else if (kind == '!' && length - position >= 4 && text[position + 2] == '-' && text[position + 3] == '-')
                position = sz_subdocuments_markup_end_metal_(text, length, position + 4, '-', 2);
            else if (kind == '?') position = sz_subdocuments_markup_end_metal_(text, length, position + 2, '?', 1);
            else if (kind == '!' || kind == '/') {
                depth -= kind == '/' && depth;
                position = sz_subdocuments_markup_find_metal_(text, length, position, '>');
            }
            else {
                // Quoted attribute values may hold any `>`.
                for (++position; position < length && text[position] != '>'; ++position)
                    if (text[position] == '"' || text[position] == '\'')
                        position = sz_subdocuments_markup_find_metal_(text, length, position + 1, text[position]);
                if (position < length && text[position - 1] != '/') deepest = max(deepest, ++depth);
            }
        }
        return deepest + 3;
    }
    cursor.text = text, cursor.length = length, cursor.escaped = 0, cursor.in_string = 0, cursor.scalar = 0;
    for (; position < length; position += 64) {
        sz_subdocuments_load_metal_(cursor, position);
        structurals += popcount(cursor.pending);
        for (ulong brackets = cursor.opens | cursor.closes; brackets; brackets &= brackets - 1)
            if (cursor.opens >> ctz(brackets) & 1) deepest = max(deepest, ++depth);
            else if (depth) --depth;
    }
    for (uint word = 0; word != engine.words_count; ++word) relative |= engine.relative_steps[word];
    return engine.filters_count || relative ? structurals + engine.filters_count + 2 : deepest + 1;
}

/** Lists @p document of @p length bytes for the cooperative walk or, measuring its frames when
 *  @p frames is zero, for the second walk, through the header words @p header, as
 *  @c sz_subdocuments_list_simt_ does. */
inline void sz_subdocuments_list_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                        constant sz_subdocuments_arguments_metal_t &arguments,
                                        device atomic_uint *header, device ulong *large_list, device uint *deep,
                                        device ulong *counted, device ulong *counts, device uchar const *text,
                                        ulong document, ulong length, ulong frames) {
    uint const chunks = uint(sz_size_divide_round_up_metal_<ulong>(length, sz_subdocuments_chunk_bytes_metal_k));
    if (arguments.large_bytes != ~0ul && length <= 0xFFFFFFFFul &&
        (length > arguments.large_bytes || frames > sz_subdocuments_deep_frames_metal_k)) {
        if (ulong(atomic_fetch_add_explicit(header + 4, chunks, memory_order_relaxed)) + chunks <=
            sz_subdocuments_large_chunks_metal_k) {
            uint const slot = atomic_fetch_add_explicit(header + 3, 1u, memory_order_relaxed);
            if (counted) counted[document] = 0;
            else counts[document * arguments.counts_stride] = 0;
            large_list[2 * slot] = document, large_list[2 * slot + 1] = length;
            return;
        }
        atomic_fetch_sub_explicit(header + 4, chunks, memory_order_relaxed);
    }
    if (!frames) frames = sz_subdocuments_frames_metal_(engine, text, length);
    deep[atomic_fetch_add_explicit(header + 1, 1u, memory_order_relaxed)] = uint(document);
    atomic_fetch_max_explicit(header + 2, uint(frames), memory_order_relaxed);
}

/** The compiled paths at the kernel's own addresses, from the host's ones in @p arguments. */
inline sz_subdocuments_engine_metal_t sz_subdocuments_engine_reach_metal_(
    device uchar *engine_block, device uchar *regex_block, constant sz_subdocuments_arguments_metal_t &arguments) {
    ulong const host = arguments.engine_host, regex_host = arguments.regex_host;
    sz_subdocuments_engine_metal_t engine;
    engine.names_data = sz_reach_metal_<uchar const>(engine_block, host, arguments.names_data);
    engine.strings_data = sz_reach_metal_<uchar const>(engine_block, host, arguments.strings_data);
    engine.names_offsets = sz_reach_metal_<uint const>(engine_block, host, arguments.names_offsets);
    engine.names_slots = sz_reach_metal_<uint const>(engine_block, host, arguments.names_slots);
    engine.slices_offsets = sz_reach_metal_<uint const>(engine_block, host, arguments.slices_offsets);
    engine.steps_paths = sz_reach_metal_<uint const>(engine_block, host, arguments.steps_paths);
    engine.names_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.names_steps);
    engine.wildcard_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.wildcard_steps);
    engine.descendant_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.descendant_steps);
    engine.relative_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.relative_steps);
    engine.attribute_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.attribute_steps);
    engine.text_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.text_steps);
    engine.filter_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.filter_steps);
    engine.filters_roots = sz_reach_metal_<ulong const>(engine_block, host, arguments.filters_roots);
    engine.final_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.final_steps);
    engine.root_steps = sz_reach_metal_<ulong const>(engine_block, host, arguments.root_steps);
    engine.slices = sz_reach_metal_<sz_subdocuments_slice_metal_t const>(engine_block, host, arguments.slices);
    engine.filters = sz_reach_metal_<sz_subdocuments_filter_metal_t const>(engine_block, host, arguments.filters);
    engine.instructions = sz_reach_metal_<sz_subdocuments_instruction_metal_t const>(engine_block, host,
                                                                                     arguments.instructions);
    engine.byte_to_class = sz_reach_metal_<uchar const>(regex_block, regex_host, arguments.byte_to_class);
    engine.forward_rows = sz_reach_metal_<uint const>(regex_block, regex_host, arguments.forward_rows);
    engine.words_count = arguments.words_count, engine.names_slots_count = arguments.names_slots_count;
    engine.filters_count = arguments.filters_count, engine.format = arguments.format;
    engine.forward_start = arguments.forward_start, engine.forward_accepts_begin = arguments.forward_accepts_begin;
    engine.classes_count = arguments.classes_count;
    return engine;
}

/** Walks every document, or every listed one, one thread each: counting, or storing at the
 *  offsets the counts were scanned into, as @c sz_subdocuments_walk_simt_kernel_ does,
 *  leaving a document past @c large_bytes to the cooperative walk while it has room, and to
 *  the second walk past that. */
kernel void sz_subdocuments_walk_metal_kernel_(
    device uchar *engine_block [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *documents [[buffer(2)]], device uchar const *documents_offsets [[buffer(3)]],
    device sz_match_metal_t *matches [[buffer(4)]], device ulong *matches_offsets [[buffer(5)]],
    device ulong *counts [[buffer(6)]], device uchar *regex_block [[buffer(7)]],
    constant sz_subdocuments_arguments_metal_t &arguments [[buffer(8)]], uint position [[thread_position_in_grid]]) {
    ulong const scratch_host = arguments.scratch_host;
    sz_subdocuments_engine_metal_t const engine = sz_subdocuments_engine_reach_metal_(engine_block, regex_block,
                                                                                      arguments);
    sz_strings_tape_metal_t const tape = {documents, documents_offsets, arguments.documents_layout};
    device atomic_uint *const header = sz_reach_metal_<atomic_uint>(scratch, scratch_host, arguments.header);
    device ulong *const large_list = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.large_list);
    device ulong *const counted = arguments.counted ? sz_reach_metal_<ulong>(scratch, scratch_host, arguments.counted)
                                                    : (device ulong *)nullptr;
    device uint *const deep = sz_reach_metal_<uint>(scratch, scratch_host, arguments.deep);
    device ulong *const stack = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.stacks) +
                                position * arguments.frames * arguments.frame_words;
    device uchar *const key = sz_reach_metal_<uchar>(scratch, scratch_host, arguments.keys) +
                              position * arguments.key_bytes;
    ulong const matches_capacity = arguments.storing ? arguments.matches_capacity : 0;

    // Only the threads the stacks were sized for walk, whatever the grid rounds up to.
    for (ulong item = position; position < arguments.threads && item < arguments.items; item += arguments.threads) {
        ulong const document = arguments.listed ? ulong(deep[item]) : item;
        device uchar const *const text = sz_strings_start_metal_(tape, document);
        ulong const length = sz_strings_length_metal_(tape, document);
        ulong emitted = arguments.storing ? counted[document] : 0;
        bool const large = !arguments.listed && length > arguments.large_bytes && length <= 0xFFFFFFFFul;
        if (arguments.storing) {
            matches_offsets[document] = emitted;
            if (document + 1 == arguments.documents_count)
                matches_offsets[arguments.documents_count] = counted[arguments.documents_count];
            if (emitted >= matches_capacity || large) continue;
        }
        else if (large) {
            sz_subdocuments_list_metal_(engine, arguments, header, large_list, deep, counted, counts, text, document,
                                        length, 0);
            continue;
        }
        int const ending = engine.format == sz_subdocuments_xml_metal_k
                               ? sz_subdocuments_walk_markup_metal_(
                                     engine, text, length, document, stack, arguments.frames, arguments.frame_words,
                                     arguments.key_bytes, matches, matches_capacity, emitted)
                               : sz_subdocuments_walk_metal_(engine, text, length, document, stack, arguments.frames,
                                                             arguments.frame_words, key, arguments.key_bytes, matches,
                                                             matches_capacity, emitted);
        if (arguments.storing) continue;
        if (ending == sz_subdocuments_walked_metal_k) {
            if (counted) counted[document] = emitted;
            else counts[document * arguments.counts_stride] = emitted;
        }
        else if (ending == sz_subdocuments_malformed_metal_k)
            atomic_fetch_min_explicit(header, uint(document), memory_order_relaxed);
        else if (!arguments.listed)
            sz_subdocuments_list_metal_(engine, arguments, header, large_list, deep, counted, counts, text, document,
                                        length, sz_subdocuments_frames_metal_(engine, text, length));
    }
}

/** Entries of one level of depth minima each entry of the level above summarizes. */
constant ulong sz_subdocuments_minima_fanout_metal_k = 32;

/** Threads every threadgroup runs, which the scans size their threadgroup memory by. */
enum { sz_subdocuments_threads_per_group_metal_k = 256 };

/** What a structural byte of a large document is, numbered as @c sz_subdocuments_token_t. */
enum sz_subdocuments_token_metal_t : uchar {
    sz_subdocuments_token_object_metal_k = 0,
    sz_subdocuments_token_array_metal_k = 1,
    sz_subdocuments_token_object_end_metal_k = 2,
    sz_subdocuments_token_array_end_metal_k = 3,
    sz_subdocuments_token_colon_metal_k = 4,
    sz_subdocuments_token_comma_metal_k = 5,
    sz_subdocuments_token_quote_metal_k = 6,
    sz_subdocuments_token_quote_end_metal_k = 7,
    sz_subdocuments_token_scalar_metal_k = 8,
};

/** The container of none, and the parent left for the pyramid to find, as the
 *  SIMT walk numbers them. */
constant uint sz_subdocuments_no_container_metal_k = 0xFFFFFFFFu;
constant uint sz_subdocuments_outer_container_metal_k = 0xFFFFFFFEu;

/** The cooperative walk of the listed large documents at the kernel's own addresses,
 *  with the list itself. */
struct sz_subdocuments_view_metal_t {
    device ulong *list, *firsts, *transfers, *tokens, *opens, *depths, *extremes, *sums, *matches, *tiles, *scanned;
    device uint *malformed, *starts, *positions, *openers, *closers, *minima, *parents, *elements, *jumps;
    device uchar *keys, *kinds;
    device ulong *offsets, *selectors, *lives, *carried;
};

/** Where a chunk lies, as @c sz_subdocuments_place_simt_t keeps it. */
struct sz_subdocuments_place_metal_t {
    ulong slot, document, length, first, next;
};

/** The arrays of @p arguments ' cooperative walk at the kernel's own addresses
 *  inside the round block. */
inline sz_subdocuments_view_metal_t sz_subdocuments_view_metal_(device uchar *scratch,
                                                                constant sz_subdocuments_arguments_metal_t &arguments) {
    ulong const host = arguments.scratch_host;
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t view;
    view.list = sz_reach_metal_<ulong>(scratch, host, arguments.large_list);
    view.firsts = sz_reach_metal_<ulong>(scratch, host, large.firsts);
    view.malformed = sz_reach_metal_<uint>(scratch, host, large.malformed);
    view.transfers = sz_reach_metal_<ulong>(scratch, host, large.transfers);
    view.tokens = sz_reach_metal_<ulong>(scratch, host, large.tokens);
    view.opens = sz_reach_metal_<ulong>(scratch, host, large.opens);
    view.depths = sz_reach_metal_<ulong>(scratch, host, large.depths);
    view.extremes = sz_reach_metal_<ulong>(scratch, host, large.extremes);
    view.sums = sz_reach_metal_<ulong>(scratch, host, large.sums);
    view.matches = sz_reach_metal_<ulong>(scratch, host, large.matches);
    view.tiles = sz_reach_metal_<ulong>(scratch, host, large.tiles);
    view.scanned = sz_reach_metal_<ulong>(scratch, host, large.scanned);
    view.starts = sz_reach_metal_<uint>(scratch, host, large.starts);
    view.positions = sz_reach_metal_<uint>(scratch, host, large.positions);
    view.openers = sz_reach_metal_<uint>(scratch, host, large.openers);
    view.closers = sz_reach_metal_<uint>(scratch, host, large.closers);
    view.minima = sz_reach_metal_<uint>(scratch, host, large.minima);
    view.parents = sz_reach_metal_<uint>(scratch, host, large.parents);
    view.elements = sz_reach_metal_<uint>(scratch, host, large.elements);
    view.jumps = sz_reach_metal_<uint>(scratch, host, large.jumps);
    view.keys = sz_reach_metal_<uchar>(scratch, host, large.keys);
    view.kinds = sz_reach_metal_<uchar>(scratch, host, large.kinds);
    view.offsets = sz_reach_metal_<ulong>(scratch, host, large.offsets);
    view.selectors = sz_reach_metal_<ulong>(scratch, host, large.selectors);
    view.lives = sz_reach_metal_<ulong>(scratch, host, large.lives);
    view.carried = sz_reach_metal_<ulong>(scratch, host, large.carried);
    return view;
}

/** The effect of chunk @p earlier then chunk @p later on the carries, or the sum of two counts
 *  unless @p compose is set, as @c sz_subdocuments_combine_simt_ composes them, an effect starting
 *  a document ignoring those before. */
inline ulong sz_subdocuments_combine_metal_(ulong earlier, ulong later, uint compose) {
    if (!compose) return earlier + later;
    if (!(earlier & 1) || (later & 0x80)) return later;
    if (!(later & 1)) return earlier;
    ulong composed = 1 | (earlier & 0x80);
    for (ulong escape = 0; escape != 2; ++escape) {
        ulong const middle = earlier >> (1 + escape) & 1;
        composed |= (later >> (1 + middle) & 1) << (1 + escape);
        composed |= ((earlier >> (3 + escape) ^ later >> (3 + middle)) & 1) << (3 + escape);
        composed |= (later >> (5 + middle) & 1) << (5 + escape);
    }
    return composed;
}

/** The threadgroup's exclusive scan of @p value under @ref sz_subdocuments_combine_metal_, its
 *  total in @p total. */
inline ulong sz_subdocuments_block_scan_metal_(ulong value, threadgroup ulong *shared, uint compose, uint lane,
                                               uint width, thread ulong &total) {
    shared[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1; offset < width; offset *= 2) {
        ulong const earlier = lane >= offset ? shared[lane - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lane] = sz_subdocuments_combine_metal_(earlier, shared[lane], compose);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong const exclusive = lane ? shared[lane - 1] : 0;
    total = shared[width - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup); // ! The caller reuses `shared` for the next tile.
    return exclusive;
}

/** The listed document holding item @p index, numbered per chunk by @p numbered or, when that is
 *  null, chunk @p index itself, as @c sz_subdocuments_slot_simt_ finds it. */
inline ulong sz_subdocuments_slot_metal_(thread sz_subdocuments_view_metal_t const &view,
                                         constant sz_subdocuments_large_metal_t &large, device ulong const *numbered,
                                         ulong index) {
    ulong low = 0, high = large.listed;
    while (low != high) {
        ulong const middle = high - (high - low) / 2, first = view.firsts[middle];
        if ((numbered ? numbered[first] : first) <= index) low = middle;
        else high = middle - 1;
    }
    return low;
}

/** Where chunk @p chunk lies, as @c sz_subdocuments_place_simt_ finds it. */
inline sz_subdocuments_place_metal_t sz_subdocuments_place_metal_(thread sz_subdocuments_view_metal_t const &view,
                                                                  constant sz_subdocuments_large_metal_t &large,
                                                                  ulong chunk) {
    ulong const slot = sz_subdocuments_slot_metal_(view, large, nullptr, chunk);
    bool const listed = slot != large.listed;
    sz_subdocuments_place_metal_t place;
    place.slot = slot, place.first = view.firsts[slot];
    place.next = listed ? view.firsts[slot + 1] : large.chunks;
    place.document = listed ? view.list[2 * slot] : 0;
    place.length = listed ? view.list[2 * slot + 1] : 0;
    return place;
}

/** Starts @p cursor at chunk @p chunk of the document at @p place, @p text, with the carries the
 *  scanned effects of its chunks before it leave. */
inline void sz_subdocuments_chunk_cursor_metal_(thread sz_subdocuments_view_metal_t const &view,
                                                thread sz_subdocuments_place_metal_t const &place,
                                                device uchar const *text, ulong chunk,
                                                thread sz_subdocuments_cursor_metal_t &cursor) {
    ulong const before = chunk != place.first ? view.transfers[chunk] : 0, in_string = before >> 3 & 1;
    cursor.text = text, cursor.length = place.length;
    cursor.escaped = before >> 1 & 1, cursor.in_string = 0 - in_string;
    cursor.scalar = (before >> 5 & 1) & (in_string ^ 1);
}

/** Whether the last byte of the chunk `[start, end)` belongs to a scalar outside strings, as
 *  @c sz_subdocuments_scalar_end_simt_ decides it. */
inline ulong sz_subdocuments_scalar_end_metal_(device uchar const *text, ulong start, ulong end, ulong escape) {
    uchar const byte = text[end - 1];
    ulong run = 0;
    if (byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' || byte == ':' || byte == ',' ||
        (byte | 0x20) == '{' || (byte | 0x20) == '}')
        return 0;
    if (byte != '"') return 1;
    while (end - 1 - run != start && text[end - 2 - run] == '\\') ++run;
    return (run + (end - 1 - run == start ? escape : 0)) & 1;
}

/** The last container before @p index shallower than @p depth, or
 *  @c sz_subdocuments_no_container_metal_k, over the pyramid of minima, as
 *  @c sz_subdocuments_shallower_simt_ finds it. */
inline uint sz_subdocuments_shallower_metal_(thread sz_subdocuments_view_metal_t const &view,
                                             constant sz_subdocuments_large_metal_t &large, ulong index, uint depth) {
    ulong const fanout = sz_subdocuments_minima_fanout_metal_k;
    ulong level = 0, position = index, found = 0;
    bool hit = false;
    for (; !hit; ++level) {
        device uint const *const values = view.minima + large.minima_offsets[level];
        ulong const first = position - position % fanout;
        for (found = position; found != first && !hit;) hit = values[--found] < depth;
        if (!hit && !first) return sz_subdocuments_no_container_metal_k;
        position = first / fanout;
    }
    for (--level; level; --level) {
        device uint const *const values = view.minima + large.minima_offsets[level - 1];
        ulong child = min(found * fanout + fanout, large.minima_offsets[level] - large.minima_offsets[level - 1]);
        do --child;
        while (values[child] >= depth);
        found = child;
    }
    return uint(found);
}

/** Whether token @p token of a document whose tokens end at @p end is a value its container counts:
 *  an opening bracket, a scalar, or a string no colon follows. */
inline bool sz_subdocuments_token_value_metal_(thread sz_subdocuments_view_metal_t const &view, ulong token,
                                               ulong end) {
    uchar const kind = view.kinds[token];
    if (kind == sz_subdocuments_token_quote_metal_k)
        return token + 2 >= end || view.kinds[token + 2] != sz_subdocuments_token_colon_metal_k;
    return kind <= sz_subdocuments_token_array_metal_k || kind == sz_subdocuments_token_scalar_metal_k;
}

/** Live set @p mode of @p container, zero tentative and one exact, and the descendant steps its
 *  ancestors carry into it, null without descendant segments. */
inline device ulong *sz_subdocuments_live_metal_(thread sz_subdocuments_view_metal_t const &view,
                                                 constant sz_subdocuments_large_metal_t &large, ulong words, ulong mode,
                                                 ulong container) {
    return view.lives + (mode * large.containers_count + container) * words;
}

inline device ulong const *sz_subdocuments_carried_metal_(thread sz_subdocuments_view_metal_t const &view,
                                                          constant sz_subdocuments_large_metal_t &large, ulong words,
                                                          ulong mode, ulong container) {
    if (!large.descendants) return nullptr;
    return view.carried + ((large.carried_buffer * 2 + mode) * large.containers_count + container) * words;
}

/** Whether any step live in @p live, joined with @p carried unless null, advances into children. */
inline bool sz_subdocuments_continues_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                             device ulong const *live, device ulong const *carried) {
    ulong continuing = 0;
    for (uint word = 0; word != engine.words_count; ++word)
        continuing |= (live[word] | (carried ? carried[word] : 0)) & ~engine.final_steps[word];
    return continuing != 0;
}

/** The name of the key of the member whose value is token @p value, as
 *  @c sz_subdocuments_member_name_simt_ finds it. */
inline ulong sz_subdocuments_member_name_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                                thread sz_subdocuments_view_metal_t const &view,
                                                constant sz_subdocuments_large_metal_t &large, device uchar const *text,
                                                ulong value, device uchar *key) {
    // A member's value follows its key's two quotes and a colon.
    if (!engine.names_slots_count || value < 3 || view.kinds[value - 1] != sz_subdocuments_token_colon_metal_k ||
        view.kinds[value - 3] != sz_subdocuments_token_quote_metal_k)
        return sz_subdocuments_no_name_metal_k;
    device uchar const *bytes = text + view.positions[value - 3] + 1;
    ulong length = view.positions[value - 2] - view.positions[value - 3] - 1, index = 0;
    while (index != length && bytes[index] != '\\') ++index;
    if (index != length) {
        length = sz_subdocuments_unescape_metal_(bytes, length, key, large.key_bytes);
        if (length == ~0ul) return sz_subdocuments_no_name_metal_k;
        bytes = key;
    }
    return length <= large.key_bytes ? sz_subdocuments_name_metal_(engine, bytes, length)
                                     : sz_subdocuments_no_name_metal_k;
}

/** What the walk expects at token @p token inside container @p current of a document whose tokens
 *  begin at @p begin: whatever may follow the token before. */
inline uint sz_subdocuments_token_expect_metal_(thread sz_subdocuments_view_metal_t const &view, ulong begin,
                                                ulong token, uint current) {
    if (token == begin) return sz_subdocuments_expect_value_metal_k;
    uchar const previous = view.kinds[token - 1];
    if (previous == sz_subdocuments_token_object_metal_k) return sz_subdocuments_expect_key_or_close_metal_k;
    if (previous == sz_subdocuments_token_array_metal_k) return sz_subdocuments_expect_value_or_close_metal_k;
    if (previous == sz_subdocuments_token_colon_metal_k) return sz_subdocuments_expect_value_metal_k;
    if (previous == sz_subdocuments_token_comma_metal_k)
        return current != sz_subdocuments_no_container_metal_k &&
                       view.kinds[view.openers[current]] == sz_subdocuments_token_object_metal_k
                   ? sz_subdocuments_expect_key_metal_k
                   : sz_subdocuments_expect_value_metal_k;
    // A string a colon follows is a key, so the colon completes it, and anything
    // else follows a value.
    if (previous == sz_subdocuments_token_quote_end_metal_k && view.kinds[token] == sz_subdocuments_token_colon_metal_k)
        return sz_subdocuments_expect_colon_metal_k;
    return current != sz_subdocuments_no_container_metal_k ? sz_subdocuments_expect_comma_or_close_metal_k
                                                           : sz_subdocuments_expect_end_metal_k;
}

/** Whether the walk accepts token @p token inside container @p current of the document of
 *  @p length bytes at @p text whose tokens are `[begin, end)`, as
 *  @c sz_subdocuments_token_valid_simt_ decides it. */
inline bool sz_subdocuments_token_valid_metal_(thread sz_subdocuments_engine_metal_t const &engine,
                                               thread sz_subdocuments_view_metal_t const &view,
                                               constant sz_subdocuments_large_metal_t &large, device uchar const *text,
                                               ulong length, ulong begin, ulong end, ulong token, uint current) {
    uint const expect = sz_subdocuments_token_expect_metal_(view, begin, token, current);
    uchar const kind = view.kinds[token];
    ulong const position = view.positions[token];
    bool const value = expect == sz_subdocuments_expect_value_metal_k ||
                       expect == sz_subdocuments_expect_value_or_close_metal_k;
    if (kind == sz_subdocuments_token_colon_metal_k) return expect == sz_subdocuments_expect_colon_metal_k;
    if (kind == sz_subdocuments_token_comma_metal_k) return expect == sz_subdocuments_expect_comma_or_close_metal_k;
    if (kind == sz_subdocuments_token_object_end_metal_k || kind == sz_subdocuments_token_array_end_metal_k)
        return current != sz_subdocuments_no_container_metal_k && view.kinds[view.openers[current]] + 2 == kind &&
               (expect == sz_subdocuments_expect_comma_or_close_metal_k ||
                expect == (kind == sz_subdocuments_token_object_end_metal_k
                               ? sz_subdocuments_expect_key_or_close_metal_k
                               : sz_subdocuments_expect_value_or_close_metal_k));
    if (kind == sz_subdocuments_token_scalar_metal_k)
        return value && sz_subdocuments_scalar_length_metal_(text, position, length) != 0;
    if (kind != sz_subdocuments_token_quote_metal_k || sz_subdocuments_token_value_metal_(view, token, end))
        return value;
    if (expect != sz_subdocuments_expect_key_metal_k && expect != sz_subdocuments_expect_key_or_close_metal_k)
        return false;
    // A key the walk looks up must unescape.
    if (!engine.names_slots_count ||
        !sz_subdocuments_needs_name_metal_(engine,
                                           sz_subdocuments_live_metal_(view, large, engine.words_count, 0, current),
                                           sz_subdocuments_carried_metal_(view, large, engine.words_count, 0, current)))
        return true;
    return sz_subdocuments_unescape_metal_(text + position + 1, view.positions[token + 1] - position - 1, nullptr, 0) !=
           ~0ul;
}

/** Bytes the value at token @p token of the document of @p length bytes at @p text spans, as
 *  @c sz_subdocuments_token_length_simt_ measures them. */
inline ulong sz_subdocuments_token_length_metal_(thread sz_subdocuments_view_metal_t const &view,
                                                 device uchar const *text, ulong length, ulong token, ulong container) {
    uchar const kind = view.kinds[token];
    ulong const position = view.positions[token];
    if (kind <= sz_subdocuments_token_array_metal_k) return view.positions[view.closers[container]] + 1 - position;
    if (kind == sz_subdocuments_token_quote_metal_k) return view.positions[token + 1] + 1 - position;
    return sz_subdocuments_scalar_length_metal_(text, position, length);
}

/** Counts each listed document's chunks, none for one at or past @c bound, and clears its verdict,
 *  as @c sz_subdocuments_large_lengths_simt_kernel_ does. */
kernel void sz_subdocuments_large_lengths_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                        constant sz_subdocuments_arguments_metal_t &arguments
                                                        [[buffer(8)]],
                                                        uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    for (ulong slot = position; slot <= large.listed; slot += large.threads) {
        bool const kept = slot != large.listed && view.list[2 * slot] < large.bound;
        view.firsts[slot] = kept ? sz_size_divide_round_up_metal_<ulong>(view.list[2 * slot + 1],
                                                                         sz_subdocuments_chunk_bytes_metal_k)
                                 : 0;
        if (slot != large.listed) view.malformed[slot] = 0;
    }
}

/** Records each chunk's effect on the carries for either escape carry it starts with, marking the
 *  first of each document, as @c sz_subdocuments_large_summary_simt_kernel_ does. */
kernel void sz_subdocuments_large_summary_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                        device uchar const *documents [[buffer(2)]],
                                                        device uchar const *documents_offsets [[buffer(3)]],
                                                        constant sz_subdocuments_arguments_metal_t &arguments
                                                        [[buffer(8)]],
                                                        uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    sz_strings_tape_metal_t const tape = {documents, documents_offsets, arguments.documents_layout};
    for (ulong chunk = position; chunk <= large.chunks; chunk += large.threads) {
        sz_subdocuments_place_metal_t const place = sz_subdocuments_place_metal_(view, large, chunk);
        if (place.slot == large.listed) {
            view.transfers[chunk] = 0;
            continue;
        }
        device uchar const *const text = sz_strings_start_metal_(tape, place.document);
        ulong const start = (chunk - place.first) * sz_subdocuments_chunk_bytes_metal_k;
        ulong const end = min(start + sz_subdocuments_chunk_bytes_metal_k, place.length);
        ulong transfer = chunk == place.first ? 0x81 : 1;
        sz_subdocuments_cursor_metal_t cursor;
        cursor.text = text, cursor.length = place.length, cursor.scalar = 0;
        for (ulong escape = 0; escape != 2; ++escape) {
            cursor.escaped = escape, cursor.in_string = 0;
            for (ulong offset = start; offset < end; offset += 64) sz_subdocuments_load_metal_(cursor, offset);
            transfer |= (cursor.escaped & 1) << (1 + escape) | (cursor.in_string & 1) << (3 + escape) |
                        sz_subdocuments_scalar_end_metal_(text, start, end, escape) << (5 + escape);
        }
        view.transfers[chunk] = transfer;
    }
}

/** Reduces each threadgroup's own tile of @c scanned into @c tiles, the first of
 *  a scan's three dispatches. */
kernel void sz_subdocuments_large_reduce_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                       constant sz_subdocuments_arguments_metal_t &arguments
                                                       [[buffer(8)]],
                                                       uint group [[threadgroup_position_in_grid]],
                                                       uint lane [[thread_index_in_threadgroup]],
                                                       uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_subdocuments_threads_per_group_metal_k];
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    ulong const count = large.scanned_count, per_tile = sz_size_divide_round_up_metal_<ulong>(count, large.scan_tiles);
    ulong const begin = min(ulong(group) * per_tile, count), end = min(begin + per_tile, count);
    ulong running = 0, total;
    for (ulong first = begin; first < end; first += width) {
        ulong const index = first + lane;
        sz_subdocuments_block_scan_metal_(index < end ? view.scanned[index] : 0, shared, large.compose, lane, width,
                                          total);
        running = sz_subdocuments_combine_metal_(running, total, large.compose);
    }
    if (lane == 0) view.tiles[group] = running;
}

/** Scans @c tiles in place on one threadgroup, the second of a scan's three dispatches. */
kernel void sz_subdocuments_large_carry_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                      constant sz_subdocuments_arguments_metal_t &arguments
                                                      [[buffer(8)]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_subdocuments_threads_per_group_metal_k];
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    ulong running = 0, total;
    for (ulong first = 0; first < large.scan_tiles; first += width) {
        ulong const index = first + lane;
        ulong const exclusive = sz_subdocuments_block_scan_metal_(index < large.scan_tiles ? view.tiles[index] : 0,
                                                                  shared, large.compose, lane, width, total);
        if (index < large.scan_tiles)
            view.tiles[index] = sz_subdocuments_combine_metal_(running, exclusive, large.compose);
        running = sz_subdocuments_combine_metal_(running, total, large.compose);
    }
}

/** Scans each threadgroup's own tile of @c scanned in place from the carry before it, the
 *  last of three dispatches. */
kernel void sz_subdocuments_large_apply_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                      constant sz_subdocuments_arguments_metal_t &arguments
                                                      [[buffer(8)]],
                                                      uint group [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_subdocuments_threads_per_group_metal_k];
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    ulong const count = large.scanned_count, per_tile = sz_size_divide_round_up_metal_<ulong>(count, large.scan_tiles);
    ulong const begin = min(ulong(group) * per_tile, count), end = min(begin + per_tile, count);
    ulong running = view.tiles[group], total;
    for (ulong first = begin; first < end; first += width) {
        ulong const index = first + lane;
        ulong const exclusive = sz_subdocuments_block_scan_metal_(index < end ? view.scanned[index] : 0, shared,
                                                                  large.compose, lane, width, total);
        if (index < end) view.scanned[index] = sz_subdocuments_combine_metal_(running, exclusive, large.compose);
        running = sz_subdocuments_combine_metal_(running, total, large.compose);
    }
}

/** Counts each chunk's tokens and opening brackets, the depth it ends at, and how far its
 *  depth dips and rises. */
kernel void sz_subdocuments_large_count_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                      device uchar const *documents [[buffer(2)]],
                                                      device uchar const *documents_offsets [[buffer(3)]],
                                                      constant sz_subdocuments_arguments_metal_t &arguments
                                                      [[buffer(8)]],
                                                      uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    sz_strings_tape_metal_t const tape = {documents, documents_offsets, arguments.documents_layout};
    for (ulong chunk = position; chunk <= large.chunks; chunk += large.threads) {
        sz_subdocuments_place_metal_t const place = sz_subdocuments_place_metal_(view, large, chunk);
        if (place.slot == large.listed) {
            view.tokens[chunk] = 0, view.opens[chunk] = 0, view.depths[chunk] = 0, view.extremes[chunk] = 0;
            continue;
        }
        device uchar const *const text = sz_strings_start_metal_(tape, place.document);
        ulong const start = (chunk - place.first) * sz_subdocuments_chunk_bytes_metal_k;
        ulong const end = min(start + sz_subdocuments_chunk_bytes_metal_k, place.length);
        long depth = 0, lowest = 0, highest = 0;
        ulong tokens = 0, opens = 0;
        sz_subdocuments_cursor_metal_t cursor;
        sz_subdocuments_chunk_cursor_metal_(view, place, text, chunk, cursor);
        for (ulong offset = start; offset < end; offset += 64) {
            sz_subdocuments_load_metal_(cursor, offset);
            tokens += popcount(cursor.pending);
            for (ulong brackets = cursor.opens | cursor.closes; brackets; brackets &= brackets - 1)
                if (cursor.opens >> ctz(brackets) & 1) ++opens, ++depth, highest = max(highest, depth);
                else --depth, lowest = min(lowest, depth);
        }
        view.tokens[chunk] = tokens, view.opens[chunk] = opens, view.depths[chunk] = ulong(depth);
        view.extremes[chunk] = ulong(-lowest) | ulong(highest) << 32;
    }
}

/** Refuses a document whose depth dips below zero, counted from its first chunk, ends above it, or
 *  ends inside a string, as @c sz_subdocuments_large_check_simt_kernel_ does. */
kernel void sz_subdocuments_large_check_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                      constant sz_subdocuments_arguments_metal_t &arguments
                                                      [[buffer(8)]],
                                                      uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    device atomic_uint *const header = sz_reach_metal_<atomic_uint>(scratch, arguments.scratch_host, arguments.header);
    for (ulong chunk = position; chunk < large.chunks; chunk += large.threads) {
        sz_subdocuments_place_metal_t const place = sz_subdocuments_place_metal_(view, large, chunk);
        if (place.slot == large.listed) continue;
        ulong const base = view.depths[place.first];
        long const depth = long(view.depths[chunk] - base);
        bool malformed = depth - long(view.extremes[chunk] & 0xFFFFFFFFul) < 0;
        if (chunk == place.first)
            malformed |= view.depths[place.next] != base || (view.transfers[place.next] >> 3 & 1) != 0;
        if (malformed) {
            view.malformed[place.slot] = 1;
            atomic_fetch_min_explicit(header, uint(place.document), memory_order_relaxed);
        }
    }
}

/** Drops the chunks of every refused document and publishes the deepest any kept document nests, as
 *  @c sz_subdocuments_large_drop_simt_kernel_ does. */
kernel void sz_subdocuments_large_drop_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                     constant sz_subdocuments_arguments_metal_t &arguments
                                                     [[buffer(8)]],
                                                     uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    device atomic_uint *const header = sz_reach_metal_<atomic_uint>(scratch, arguments.scratch_host, arguments.header);
    for (ulong chunk = position; chunk < large.chunks; chunk += large.threads) {
        ulong const slot = sz_subdocuments_slot_metal_(view, large, nullptr, chunk);
        if (slot == large.listed) continue;
        if (view.malformed[slot]) {
            view.tokens[chunk] = 0, view.opens[chunk] = 0;
            continue;
        }
        long const highest = long(view.depths[chunk] - view.depths[view.firsts[slot]] + (view.extremes[chunk] >> 32));
        if (highest > 0) atomic_fetch_max_explicit(header + 5, uint(highest), memory_order_relaxed);
    }
}

/** Writes each chunk's tokens and the containers it opens, at depths counted from its document's
 *  first chunk, as @c sz_subdocuments_large_tokens_simt_kernel_ does. */
kernel void sz_subdocuments_large_tokens_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                       device uchar const *documents [[buffer(2)]],
                                                       device uchar const *documents_offsets [[buffer(3)]],
                                                       constant sz_subdocuments_arguments_metal_t &arguments
                                                       [[buffer(8)]],
                                                       uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    sz_strings_tape_metal_t const tape = {documents, documents_offsets, arguments.documents_layout};
    for (ulong chunk = position; chunk < large.chunks; chunk += large.threads) {
        ulong token = view.tokens[chunk], container = view.opens[chunk];
        // A dropped document's chunks own no tokens to write.
        if (token == view.tokens[chunk + 1]) continue;
        sz_subdocuments_place_metal_t const place = sz_subdocuments_place_metal_(view, large, chunk);
        device uchar const *const text = sz_strings_start_metal_(tape, place.document);
        ulong const start = (chunk - place.first) * sz_subdocuments_chunk_bytes_metal_k;
        ulong const end = min(start + sz_subdocuments_chunk_bytes_metal_k, place.length);
        uint depth = uint(view.depths[chunk] - view.depths[place.first]);
        uint current = depth ? sz_subdocuments_outer_container_metal_k : sz_subdocuments_no_container_metal_k;
        sz_subdocuments_cursor_metal_t cursor;
        sz_subdocuments_chunk_cursor_metal_(view, place, text, chunk, cursor);
        bool inside = cursor.in_string != 0;
        for (ulong offset = start; offset < end; offset += 64) {
            sz_subdocuments_load_metal_(cursor, offset);
            for (ulong pending = cursor.pending; pending; pending &= pending - 1, ++token) {
                ulong const at = offset + ctz(pending);
                uchar const byte = text[at];
                uchar kind;
                if (byte == '"')
                    kind = inside ? sz_subdocuments_token_quote_end_metal_k : sz_subdocuments_token_quote_metal_k,
                    inside = !inside;
                else if ((byte | 0x20) == '{') {
                    kind = byte == '{' ? sz_subdocuments_token_object_metal_k : sz_subdocuments_token_array_metal_k;
                    view.openers[container] = uint(token), view.minima[container] = depth++;
                    view.parents[container] = current, view.elements[container] = 0;
                    current = uint(container++);
                }
                else if ((byte | 0x20) == '}') {
                    kind = byte == '}' ? sz_subdocuments_token_object_end_metal_k
                                       : sz_subdocuments_token_array_end_metal_k;
                    --depth;
                    if (current < sz_subdocuments_outer_container_metal_k) current = view.parents[current];
                    else
                        current = depth ? sz_subdocuments_outer_container_metal_k
                                        : sz_subdocuments_no_container_metal_k;
                }
                else if (byte == ':') kind = sz_subdocuments_token_colon_metal_k;
                else if (byte == ',') kind = sz_subdocuments_token_comma_metal_k;
                else kind = sz_subdocuments_token_scalar_metal_k;
                view.positions[token] = uint(at), view.kinds[token] = kind;
            }
        }
    }
}

/** Builds level @c round of the depth minima from the level below it. */
kernel void sz_subdocuments_large_minima_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                       constant sz_subdocuments_arguments_metal_t &arguments
                                                       [[buffer(8)]],
                                                       uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    ulong const below = large.minima_offsets[large.round - 1], above = large.minima_offsets[large.round];
    ulong const count = large.minima_offsets[large.round + 1] - above;
    for (ulong index = position; index < count; index += large.threads) {
        ulong const first = below + index * sz_subdocuments_minima_fanout_metal_k;
        ulong const last = min(first + sz_subdocuments_minima_fanout_metal_k, above);
        uint least = 0xFFFFFFFFu;
        for (ulong entry = first; entry != last; ++entry) least = min(least, view.minima[entry]);
        view.minima[above + index] = least;
    }
}

/** Finds the container each chunk starts inside and the parents its chunk left open, as
 *  @c sz_subdocuments_large_parents_simt_kernel_ does. */
kernel void sz_subdocuments_large_parents_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                        constant sz_subdocuments_arguments_metal_t &arguments
                                                        [[buffer(8)]],
                                                        uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    for (ulong chunk = position; chunk < large.chunks; chunk += large.threads) {
        sz_subdocuments_place_metal_t const place = sz_subdocuments_place_metal_(view, large, chunk);
        ulong const first = view.opens[chunk], last = view.opens[chunk + 1];
        uint const depth = uint(view.depths[chunk] - view.depths[place.first]);
        uint cached_depth = 0, cached = sz_subdocuments_no_container_metal_k;
        view.starts[chunk] = depth ? sz_subdocuments_shallower_metal_(view, large, first, depth)
                                   : sz_subdocuments_no_container_metal_k;
        for (ulong container = first; container != last; ++container) {
            if (view.parents[container] != sz_subdocuments_outer_container_metal_k) continue;
            if (view.minima[container] != cached_depth)
                cached_depth = view.minima[container],
                cached = sz_subdocuments_shallower_metal_(view, large, first, cached_depth);
            view.parents[container] = cached;
        }
    }
}

/** Records where each container closes and counts the values directly inside it, as
 *  @c sz_subdocuments_large_closes_simt_kernel_ does. */
kernel void sz_subdocuments_large_closes_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                       constant sz_subdocuments_arguments_metal_t &arguments
                                                       [[buffer(8)]],
                                                       uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    device atomic_uint *const elements = (device atomic_uint *)view.elements;
    for (ulong chunk = position; chunk < large.chunks; chunk += large.threads) {
        ulong const last = view.tokens[chunk + 1];
        ulong const end = view.tokens[sz_subdocuments_place_metal_(view, large, chunk).next];
        ulong container = view.opens[chunk];
        uint current = view.starts[chunk], values = 0;
        for (ulong token = view.tokens[chunk]; token != last; ++token) {
            uchar const kind = view.kinds[token];
            values += sz_subdocuments_token_value_metal_(view, token, end);
            if (kind > sz_subdocuments_token_array_end_metal_k) continue;
            if (current != sz_subdocuments_no_container_metal_k && values)
                atomic_fetch_add_explicit(elements + current, values, memory_order_relaxed);
            values = 0;
            if (kind <= sz_subdocuments_token_array_metal_k) current = uint(container++);
            else if (current != sz_subdocuments_no_container_metal_k)
                view.closers[current] = uint(token), current = view.parents[current];
        }
        if (current != sz_subdocuments_no_container_metal_k && values)
            atomic_fetch_add_explicit(elements + current, values, memory_order_relaxed);
    }
}

/** Sums each chunk's values less the values of the containers it closes. */
kernel void sz_subdocuments_large_sums_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                     constant sz_subdocuments_arguments_metal_t &arguments
                                                     [[buffer(8)]],
                                                     uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    for (ulong chunk = position; chunk <= large.chunks; chunk += large.threads) {
        ulong const last = chunk == large.chunks ? 0 : view.tokens[chunk + 1];
        ulong const end = view.tokens[sz_subdocuments_place_metal_(view, large, chunk).next];
        ulong container = view.opens[chunk], sum = 0;
        uint current = chunk == large.chunks ? sz_subdocuments_no_container_metal_k : view.starts[chunk];
        for (ulong token = chunk == large.chunks ? 0 : view.tokens[chunk]; token != last; ++token) {
            uchar const kind = view.kinds[token];
            sum += sz_subdocuments_token_value_metal_(view, token, end);
            if (kind <= sz_subdocuments_token_array_metal_k) current = uint(container++);
            else if (kind <= sz_subdocuments_token_array_end_metal_k)
                sum -= view.elements[current], current = view.parents[current];
        }
        view.sums[chunk] = sum;
    }
}

/** Records the scanned count just past each container's opening bracket. */
kernel void sz_subdocuments_large_offsets_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                        constant sz_subdocuments_arguments_metal_t &arguments
                                                        [[buffer(8)]],
                                                        uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    for (ulong chunk = position; chunk < large.chunks; chunk += large.threads) {
        ulong const last = view.tokens[chunk + 1];
        ulong const end = view.tokens[sz_subdocuments_place_metal_(view, large, chunk).next];
        ulong container = view.opens[chunk], running = view.sums[chunk];
        uint current = view.starts[chunk];
        for (ulong token = view.tokens[chunk]; token != last; ++token) {
            uchar const kind = view.kinds[token];
            running += sz_subdocuments_token_value_metal_(view, token, end);
            if (kind <= sz_subdocuments_token_array_metal_k)
                view.offsets[container] = running, current = uint(container++);
            else if (kind <= sz_subdocuments_token_array_end_metal_k)
                running -= view.elements[current], current = view.parents[current];
        }
    }
}

/** Settles each container's selector and starts its live sets, as
 *  @c sz_subdocuments_large_selectors_simt_kernel_ does. */
kernel void sz_subdocuments_large_selectors_metal_kernel_(
    device uchar *engine_block [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *documents [[buffer(2)]], device uchar const *documents_offsets [[buffer(3)]],
    device uchar *regex_block [[buffer(7)]], constant sz_subdocuments_arguments_metal_t &arguments [[buffer(8)]],
    uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_engine_metal_t const engine = sz_subdocuments_engine_reach_metal_(engine_block, regex_block,
                                                                                      arguments);
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    sz_strings_tape_metal_t const tape = {documents, documents_offsets, arguments.documents_layout};
    device uchar *const key = view.keys + position * large.key_bytes;
    ulong const words = engine.words_count;
    for (ulong container = position; container < large.containers_count; container += large.threads) {
        uint const parent = view.parents[container];
        ulong selector = 0;
        if (parent != sz_subdocuments_no_container_metal_k &&
            view.kinds[view.openers[parent]] == sz_subdocuments_token_object_metal_k) {
            ulong const slot = sz_subdocuments_slot_metal_(view, large, view.opens, container);
            device uchar const *const text = sz_strings_start_metal_(tape, view.list[2 * slot]);
            selector = sz_subdocuments_member_name_metal_(engine, view, large, text, view.openers[container], key);
        }
        else if (parent != sz_subdocuments_no_container_metal_k)
            selector = view.offsets[container] - 1 - view.offsets[parent];
        view.selectors[container] = selector;
        for (ulong word = 0; word != words; ++word) {
            ulong const root = parent == sz_subdocuments_no_container_metal_k ? engine.root_steps[word] : 0;
            view.lives[container * words + word] = root;
            view.lives[(large.containers_count + container) * words + word] = root;
        }
    }
}

/** Gathers into each container the descendant steps live at its ancestors, doubling how far up per
 *  round, as @c sz_subdocuments_large_jump_simt_kernel_ does. */
kernel void sz_subdocuments_large_jump_metal_kernel_(device uchar *engine_block [[buffer(0)]],
                                                     device uchar *scratch [[buffer(1)]],
                                                     device uchar *regex_block [[buffer(7)]],
                                                     constant sz_subdocuments_arguments_metal_t &arguments
                                                     [[buffer(8)]],
                                                     uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_engine_metal_t const engine = sz_subdocuments_engine_reach_metal_(engine_block, regex_block,
                                                                                      arguments);
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    ulong const words = engine.words_count, containers = large.containers_count;
    ulong const source = (large.round + 1) & 1, target = large.round & 1;
    for (ulong container = position; container < containers; container += large.threads) {
        uint const jump = large.round ? view.jumps[source * containers + container] : view.parents[container];
        view.jumps[target * containers + container] = large.round && jump != sz_subdocuments_no_container_metal_k
                                                          ? view.jumps[source * containers + jump]
                                                          : jump;
        for (ulong mode = 0; mode != 2; ++mode) {
            device ulong *const gathered = view.carried + ((target * 2 + mode) * containers + container) * words;
            device ulong const *const own = view.carried + ((source * 2 + mode) * containers + container) * words;
            device ulong const *const above = jump == sz_subdocuments_no_container_metal_k ? nullptr
                                              : large.round
                                                  ? view.carried + ((source * 2 + mode) * containers + jump) * words
                                                  : view.lives + (mode * containers + jump) * words;
            for (ulong word = 0; word != words; ++word)
                gathered[word] = (large.round ? own[word] : 0) |
                                 (above ? above[word] & (large.round ? ~0ul : engine.descendant_steps[word]) : 0);
        }
    }
}

/** Advances each container's live sets from its parent's in place, as
 *  @c sz_subdocuments_large_advance_simt_kernel_ does. */
kernel void sz_subdocuments_large_advance_metal_kernel_(device uchar *engine_block [[buffer(0)]],
                                                        device uchar *scratch [[buffer(1)]],
                                                        device uchar *regex_block [[buffer(7)]],
                                                        constant sz_subdocuments_arguments_metal_t &arguments
                                                        [[buffer(8)]],
                                                        uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_engine_metal_t const engine = sz_subdocuments_engine_reach_metal_(engine_block, regex_block,
                                                                                      arguments);
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    ulong const words = engine.words_count;
    for (ulong container = position; container < large.containers_count; container += large.threads) {
        uint const parent = view.parents[container];
        if (parent == sz_subdocuments_no_container_metal_k) continue;
        sz_subdocuments_child_metal_t const child = view.kinds[view.openers[parent]] ==
                                                            sz_subdocuments_token_array_metal_k
                                                        ? sz_subdocuments_child_element_metal_k
                                                        : sz_subdocuments_child_member_metal_k;
        for (ulong mode = 0; mode != 2; ++mode) {
            device ulong *const live = sz_subdocuments_live_metal_(view, large, words, mode, container);
            ulong carry = 0;
            for (uint word = 0; word != words; ++word)
                live[word] = sz_subdocuments_child_word_metal_(
                    engine, sz_subdocuments_live_metal_(view, large, words, mode, parent),
                    sz_subdocuments_carried_metal_(view, large, words, mode, parent), word, carry, child,
                    view.selectors[container], mode ? view.elements[parent] : 0);
        }
    }
}

/** Walks each chunk's tokens in order, counting matches and refusing its document where the serial
 *  walk would or when it holds no token, or writing the matches from where its document's first
 *  match lands when storing, as @c sz_subdocuments_large_walk_simt_kernel_ does. */
kernel void sz_subdocuments_large_walk_metal_kernel_(
    device uchar *engine_block [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *documents [[buffer(2)]], device uchar const *documents_offsets [[buffer(3)]],
    device sz_match_metal_t *matches [[buffer(4)]], device uchar *regex_block [[buffer(7)]],
    constant sz_subdocuments_arguments_metal_t &arguments [[buffer(8)]], uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_engine_metal_t const engine = sz_subdocuments_engine_reach_metal_(engine_block, regex_block,
                                                                                      arguments);
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    sz_strings_tape_metal_t const tape = {documents, documents_offsets, arguments.documents_layout};
    device uchar *const key = view.keys + position * large.key_bytes;
    device atomic_uint *const header = sz_reach_metal_<atomic_uint>(scratch, arguments.scratch_host, arguments.header);
    device ulong const *const counted = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.counted);
    ulong const words = engine.words_count, capacity = large.storing ? arguments.matches_capacity : 0;
    for (ulong chunk = position; chunk <= large.chunks; chunk += large.threads) {
        sz_subdocuments_place_metal_t const place = sz_subdocuments_place_metal_(view, large, chunk);
        device uchar const *const text = sz_strings_start_metal_(tape, place.document);
        ulong const begin = view.tokens[place.first], end = view.tokens[place.next];
        ulong const last = chunk == large.chunks ? 0 : view.tokens[chunk + 1];
        ulong container = view.opens[chunk], found = 0, running = view.sums[chunk];
        ulong emitted = large.storing ? counted[place.document] + view.matches[chunk] - view.matches[place.first] : 0;
        uint current = chunk == large.chunks ? sz_subdocuments_no_container_metal_k : view.starts[chunk];
        bool malformed = !large.storing && place.slot != large.listed && chunk == place.first && begin == end;
        for (ulong token = chunk == large.chunks ? 0 : view.tokens[chunk]; token != last; ++token) {
            uchar const kind = view.kinds[token];
            bool const value = sz_subdocuments_token_value_metal_(view, token, end);
            if (kind == sz_subdocuments_token_quote_end_metal_k) continue;
            if (!large.storing &&
                (current == sz_subdocuments_no_container_metal_k ||
                 sz_subdocuments_continues_metal_(engine, sz_subdocuments_live_metal_(view, large, words, 0, current),
                                                  sz_subdocuments_carried_metal_(view, large, words, 0, current))))
                malformed |= !sz_subdocuments_token_valid_metal_(engine, view, large, text, place.length, begin, end,
                                                                 token, current);
            if (value) {
                // A container's live set is settled; any other value's is its
                // parent's advanced into it.
                bool const rooted = current == sz_subdocuments_no_container_metal_k;
                bool const element = !rooted &&
                                     view.kinds[view.openers[current]] == sz_subdocuments_token_array_metal_k;
                device ulong const *const parent = rooted ? engine.root_steps
                                                          : sz_subdocuments_live_metal_(view, large, words, 1, current);
                device ulong const *const parent_carried =
                    rooted ? nullptr : sz_subdocuments_carried_metal_(view, large, words, 1, current);
                device ulong const *const own = sz_subdocuments_live_metal_(view, large, words, 1, container);
                device ulong const *const own_carried = sz_subdocuments_carried_metal_(view, large, words, 1,
                                                                                       container);
                ulong selector = element ? running - view.offsets[current] : ulong(sz_subdocuments_no_name_metal_k);
                ulong byte_length = ~0ul, carry = 0;
                if (!rooted && !element && kind > sz_subdocuments_token_array_metal_k &&
                    sz_subdocuments_needs_name_metal_(engine, parent, parent_carried))
                    selector = sz_subdocuments_member_name_metal_(engine, view, large, text, token, key);
                for (uint word = 0; word != words; ++word) {
                    ulong const live = kind <= sz_subdocuments_token_array_metal_k
                                           ? own[word] | (own_carried ? own_carried[word] : 0)
                                       : rooted ? parent[word]
                                                : sz_subdocuments_child_word_metal_(
                                                      engine, parent, parent_carried, word, carry,
                                                      element ? sz_subdocuments_child_element_metal_k
                                                              : sz_subdocuments_child_member_metal_k,
                                                      selector, element ? view.elements[current] : 0);
                    ulong reported = live & engine.final_steps[word];
                    if (!large.storing) {
                        found += popcount(reported);
                        continue;
                    }
                    if (reported && byte_length == ~0ul)
                        byte_length = sz_subdocuments_token_length_metal_(view, text, place.length, token, container);
                    for (; reported; reported &= reported - 1, ++emitted)
                        if (emitted < capacity)
                            matches[emitted] = {place.document, engine.steps_paths[word * 64 + ctz(reported)],
                                                view.positions[token], byte_length};
                }
            }
            if (kind <= sz_subdocuments_token_array_metal_k) current = uint(container++);
            else if (kind <= sz_subdocuments_token_array_end_metal_k)
                running -= view.elements[current], current = view.parents[current];
            running += value;
        }
        if (!large.storing) view.matches[chunk] = found;
        if (malformed) atomic_fetch_min_explicit(header, uint(place.document), memory_order_relaxed);
    }
}

/** Writes each listed document's matches, the scanned matches of its chunks, into its count, as
 *  @c sz_subdocuments_large_finish_simt_kernel_ does. */
kernel void sz_subdocuments_large_finish_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                       device ulong *counts [[buffer(6)]],
                                                       constant sz_subdocuments_arguments_metal_t &arguments
                                                       [[buffer(8)]],
                                                       uint position [[thread_position_in_grid]]) {
    constant sz_subdocuments_large_metal_t &large = arguments.large;
    sz_subdocuments_view_metal_t const view = sz_subdocuments_view_metal_(scratch, arguments);
    device ulong *const counted = arguments.counted
                                      ? sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.counted)
                                      : (device ulong *)nullptr;
    for (ulong slot = position; slot < large.listed; slot += large.threads) {
        ulong const document = view.list[2 * slot];
        ulong const found = view.matches[view.firsts[slot + 1]] - view.matches[view.firsts[slot]];
        if (counted) counted[document] = found;
        else counts[document * arguments.counts_stride] = found;
    }
}
