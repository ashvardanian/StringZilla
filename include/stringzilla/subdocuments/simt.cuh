/**
 *  @file include/stringzilla/subdocuments/simt.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The kernels CUDA and ROCm share for path queries: one thread, or one warp for JSON, walks
 *      one document the way the serial tier does, its depth stack in the round block.
 *
 *  Both walks are the serial ones. JSON classifies every 64-byte block into masks, skips a
 *  container no live step continues into by counting its brackets, and keeps the nodes of a pending
 *  run until the array or filtered node holding them closes and the replay settles them. A warp
 *  walking a JSON document classifies 1,024 bytes at once, 32 to a lane, and its lanes then take
 *  the structural bytes one each, matching each with its container by depth, so the checks, the key
 *  lookups and the live sets of a batch settle side by side. XML searches byte by byte for the one
 *  byte its lexer needs next. The replay runs the filters of a node one level up in its spare
 *  frames rather than recursing, and `match()` and `search()` walk the forward DFA of the regex
 *  engine the paths embed over the tested string in the same thread.
 *
 *  A thread's stack holds a fixed number of frames. A document needing more is measured instead and
 *  listed, and a second launch walks the listed ones on fewer threads with stacks as deep as the
 *  most any of them needs, so every document is walked exactly however deep it nests.
 *
 *  Under paths without filters, once any JSON document of a batch passes both
 *  @c sz_subdocuments_large_bytes_simt_k and the share of the batch's bytes
 *  @c sz_subdocuments_large_walkers_simt_k walkers take, every one past that share is listed too,
 *  and every listed one is walked over 256-byte chunks by the whole device in one pipeline, each
 *  chunk finding its document by binary search over a scan of their lengths. Each chunk's effect on
 *  the escape, quote and scalar carries composes, a document's first chunk resetting them, so one
 *  scan settles where every chunk starts; scans of opens and of opens minus closes number the
 *  containers and give their depths, each document's counted from its own first chunk. A document
 *  no walk accepts by its depths or quotes is dropped before its tokens are numbered. Each
 *  container's parent is the last shallower one before it, found over a pyramid of depth minima,
 *  which reaches its document's root before any earlier document, and element positions are a scan
 *  of values from which each closed container takes its own back. Entering a child maps a live set
 *  by a function that distributes over union, so the steps a descendant segment carries are an OR
 *  over ancestors, which pointer jumping gathers in O(log depth) launches, once per step of the
 *  longest path. The tokens are then checked wherever the serial walk would enter, which with
 *  tentative live sets for the steps an array's length settles is exactly where it validates, and
 *  matched in document order.
 *
 *  Each vendor's host side, in `subdocuments/cuda.cuh` and `subdocuments/rocm.cuh`,
 *  sizes and launches these.
 *
 *  @sa include/stringzilla/subdocuments.h
 *  @sa include/stringzilla/subdocuments/metal.metal, which these mirror kernel for kernel
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_SIMT_CUH_
#define STRINGZILLA_SUBDOCUMENTS_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/subdocuments.h" // `sz_subdocuments_engine_t`, `sz_subdocuments_frame_t`

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Shapes

/** Threads every kernel here launches with. */
enum { sz_subdocuments_threads_per_block_simt_k = 256 };

/** Frames a thread's stack holds in the first launch; past them a document waits for the second. */
enum { sz_subdocuments_frames_simt_k = 16 };

/** Frames past which a document the cooperative walk takes goes to it rather than to the second
 *  launch, where one stack this deep would crowd the others out of their share. */
enum { sz_subdocuments_deep_frames_simt_k = 1 << 16 };

/** Bytes the second launch's stacks may grow to, so the listed documents walk side by side. */
enum { sz_subdocuments_deep_bytes_simt_k = 1 << 30 };

/** Matches each walker of a JSON find stages while its first launch counts, which storing then
 *  moves into place rather than walking those documents again. */
enum { sz_subdocuments_staged_matches_simt_k = 128 };

/** Words of a frame ahead of its live set, laid out as the serial @c sz_subdocuments_frame_t. */
enum { sz_subdocuments_frame_header_words_simt_k = sizeof(sz_subdocuments_frame_t) / sizeof(sz_u64_t) };

/** Bytes a JSON document under paths without filters must pass for its batch to take the
 *  cooperative walk at all, as a warp may walk a shorter one sooner than its launches end. */
enum { sz_subdocuments_large_bytes_simt_k = 1 << 17 };

/** Walkers whose share of a batch's bytes a document must pass to walk over chunks once the batch
 *  takes the cooperative walk, as its one warp would still be walking after the others end. */
enum { sz_subdocuments_large_walkers_simt_k = 4 };

/** Bytes of each chunk a large document is cut into, four blocks of 64, which
 *  bounds every thread's work. */
enum { sz_subdocuments_chunk_bytes_simt_k = 256 };

/** Chunks the listed documents span together at most, 4 GiB of text, so 32-bit
 *  numbers address their tokens. */
enum { sz_subdocuments_large_chunks_simt_k = 1 << 24 };

/** Entries of one level of depth minima each entry of the level above summarizes. */
enum { sz_subdocuments_minima_fanout_simt_k = 32 };

/** Levels of depth minima at most, the containers' own depths the first, which
 *  32-bit counts never pass. */
enum { sz_subdocuments_minima_levels_simt_k = 7 };

/** Tiles a cooperative scan cuts its values into at most, one block each. */
enum { sz_subdocuments_scan_tiles_simt_k = 1024 };

/** What a structural byte of a large document is, each kept as one token in document order. */
typedef enum sz_subdocuments_token_t {
    sz_subdocuments_token_object_k = 0,
    sz_subdocuments_token_array_k = 1,
    sz_subdocuments_token_object_end_k = 2,
    sz_subdocuments_token_array_end_k = 3,
    sz_subdocuments_token_colon_k = 4,
    sz_subdocuments_token_comma_k = 5,
    sz_subdocuments_token_quote_k = 6,
    sz_subdocuments_token_quote_end_k = 7,
    sz_subdocuments_token_scalar_k = 8,
} sz_subdocuments_token_t;

/** The container of none: the parent of a document's root value. */
static sz_u32_t const sz_subdocuments_no_container_simt_k = 0xFFFFFFFFu;

/** The parent of a container opened directly inside one an earlier chunk opened, until
 *  the pyramid finds it. */
static sz_u32_t const sz_subdocuments_outer_container_simt_k = 0xFFFFFFFEu;

/** What the host reads back after a walk. */
typedef struct sz_subdocuments_header_simt_t {

    /** The first malformed document, or the documents' count when none is. */
    sz_u64_t first_malformed;

    /** Documents listed for the second launch. */
    sz_u64_t deep_count;

    /** The most frames any of them needs. */
    sz_u64_t frames;

    /** The slot the matches scan chains its tiles through. */
    sz_u64_t chain;

    /** Documents listed for the cooperative walk, each as its index and its length. */
    sz_u64_t large_count;

    /** Chunks the listed documents span together. */
    sz_u64_t large_chunks;

    /** The deepest any byte of them nests, which bounds the rounds of pointer jumping. */
    sz_u64_t deepest;

    /** Bytes past which a document walks over chunks, none until the routing launch lowers it. */
    sz_u64_t large_bytes;
} sz_subdocuments_header_simt_t;

/** What one launch reads and writes, every pointer one the device addresses. */
typedef struct sz_subdocuments_round_simt_t {
    sz_subdocuments_header_simt_t *header;
    sz_u64_t *stacks, *keys, *deep, *large, *staged;
    sz_size_t *counted, *counts, *matches_offsets;
    sz_match_t *matches, *staging;
    sz_size_t counts_stride, matches_capacity, items, threads, frames, frame_words, key_bytes, large_bytes;
    sz_size_t large_shares, staging_capacity;
    sz_u32_t listed, storing;
} sz_subdocuments_round_simt_t;

/**
 *  @brief The cooperative walk of the listed large documents,
 *      every pointer one the device addresses.
 *
 *  It holds, per listed document and one past the last, its first chunk and whether it is known
 *  malformed; per chunk and one past the last, the values the scans turn into where each chunk
 *  starts; per token its byte and kind; per container its tokens, depth, parent, element count,
 *  position, selector and live sets.
 *
 *  Tokens and containers are numbered across the documents, each document's in its own contiguous
 *  range. Live sets are tentative, as the serial walk enters with them, then exact, as it reports
 *  with them, each @c words words per container; the carried ones gather the descendant steps of
 *  ancestors in two buffers that pointer jumping alternates between, as do the jumps.
 */
typedef struct sz_subdocuments_large_simt_t {
    sz_u64_t *list, *firsts, *transfers, *tokens, *opens, *depths, *extremes, *sums, *matches, *tiles, *scanned;
    sz_u32_t *malformed, *starts, *positions, *openers, *closers, *minima, *parents, *elements, *jumps;
    sz_u8_t *keys, *kinds;
    sz_u64_t *offsets, *selectors, *lives, *carried;
    sz_size_t listed, bound, chunks, containers_count, threads, key_bytes, levels;
    sz_size_t minima_offsets[sz_subdocuments_minima_levels_simt_k + 1];
    sz_size_t scanned_count, scan_tiles, round;
    sz_u32_t compose, storing, carried_buffer, descendants;
} sz_subdocuments_large_simt_t;

/** Where a chunk lies: the listed document holding it, as its slot in the list and its index in
 *  the batch, that document's length, and its chunks `[first, next)`; a chunk past every listed
 *  one has slot @c listed. */
typedef struct sz_subdocuments_place_simt_t {
    sz_size_t slot, document, length, first, next;
} sz_subdocuments_place_simt_t;

/** Takes @p bytes past @p offset, each array 64-byte aligned, and returns where they start. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_take_simt_(sz_size_t *offset, sz_size_t bytes) {
    sz_size_t const taken = *offset;
    *offset += sz_size_divide_round_up(bytes, 64) * 64;
    return taken;
}

/**
 *  @brief Points @p large into @p region for @p listed documents over @p chunks chunks: the
 *      documents' and the chunks' values, and, once their scans counted them, @p tokens tokens
 *      and @p containers containers.
 *  @return Bytes the region spans.
 */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_large_layout_simt_(sz_subdocuments_engine_t const *engine, sz_ptr_t region,
                                                                sz_size_t listed, sz_size_t chunks, sz_size_t threads,
                                                                sz_size_t key_bytes, sz_size_t tokens,
                                                                sz_size_t containers,
                                                                sz_subdocuments_large_simt_t *large) {
    sz_size_t const values = (chunks + 1) * sizeof(sz_u64_t), words = engine->words_count;
    sz_size_t offset = 0, level = 1, size = containers, minima = containers, word;
    sz_u64_t descendants = 0;
    for (word = 0; word != words; ++word) descendants |= engine->descendant_steps[word];
    large->firsts = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, (listed + 1) * sizeof(sz_u64_t)));
    large->malformed = (sz_u32_t *)(region + sz_subdocuments_take_simt_(&offset, listed * sizeof(sz_u32_t)));
    large->transfers = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, values));
    large->tokens = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, values));
    large->opens = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, values));
    large->depths = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, values));
    large->extremes = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, values));
    large->sums = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, values));
    large->matches = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, values));
    large->tiles = (sz_u64_t *)(region + sz_subdocuments_take_simt_(
                                             &offset, sz_subdocuments_scan_tiles_simt_k * sizeof(sz_u64_t)));
    large->starts = (sz_u32_t *)(region + sz_subdocuments_take_simt_(&offset, chunks * sizeof(sz_u32_t)));
    large->keys = (sz_u8_t *)(region + sz_subdocuments_take_simt_(&offset, threads * key_bytes));
    large->positions = (sz_u32_t *)(region + sz_subdocuments_take_simt_(&offset, tokens * sizeof(sz_u32_t)));
    large->kinds = (sz_u8_t *)(region + sz_subdocuments_take_simt_(&offset, tokens));
    large->openers = (sz_u32_t *)(region + sz_subdocuments_take_simt_(&offset, containers * sizeof(sz_u32_t)));
    large->closers = (sz_u32_t *)(region + sz_subdocuments_take_simt_(&offset, containers * sizeof(sz_u32_t)));
    large->parents = (sz_u32_t *)(region + sz_subdocuments_take_simt_(&offset, containers * sizeof(sz_u32_t)));
    large->elements = (sz_u32_t *)(region + sz_subdocuments_take_simt_(&offset, containers * sizeof(sz_u32_t)));
    // Every level of minima holds one entry per `fanout` of the one below, up to a level a
    // scan crosses at once.
    large->minima_offsets[0] = 0;
    for (; size > sz_subdocuments_minima_fanout_simt_k; ++level) {
        size = sz_size_divide_round_up(size, sz_subdocuments_minima_fanout_simt_k);
        large->minima_offsets[level] = minima, minima += size;
    }
    large->minima_offsets[level] = minima, large->levels = level;
    large->minima = (sz_u32_t *)(region + sz_subdocuments_take_simt_(&offset, minima * sizeof(sz_u32_t)));
    large->offsets = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, containers * sizeof(sz_u64_t)));
    large->selectors = (sz_u64_t *)(region + sz_subdocuments_take_simt_(&offset, containers * sizeof(sz_u64_t)));
    large->lives = (sz_u64_t *)(region +
                                sz_subdocuments_take_simt_(&offset, 2 * containers * words * sizeof(sz_u64_t)));
    // Without descendant segments every live set is its parent's advanced, and nothing is gathered.
    large->descendants = descendants != 0;
    large->jumps = (sz_u32_t *)(region + sz_subdocuments_take_simt_(
                                             &offset, descendants ? 2 * containers * sizeof(sz_u32_t) : 0));
    large->carried = (sz_u64_t *)(region + sz_subdocuments_take_simt_(
                                               &offset, descendants ? 4 * containers * words * sizeof(sz_u64_t) : 0));
    large->listed = listed, large->chunks = chunks, large->containers_count = containers, large->threads = threads;
    large->key_bytes = key_bytes;
    return offset;
}

/** Steps of the longest path, which bounds how many times live sets advance before they settle. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_longest_path_simt_(sz_subdocuments_engine_t const *engine) {
    sz_size_t longest = 0, first = 0;
    for (sz_size_t step = 0; step != engine->steps_count; ++step) {
        if (engine->root_steps[step >> 6] >> (step & 63) & 1) first = step;
        if (engine->final_steps[step >> 6] >> (step & 63) & 1) longest = sz_max_of_two(longest, step - first + 1);
    }
    return longest;
}

/** Words of one frame of @p engine 's walks: the header, the live set, and in XML the positional
 *  counters, two to a word, as the serial markup walk lays them out. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_frame_words_simt_(sz_subdocuments_engine_t const *engine) {
    sz_size_t const counters = engine->format == sz_subdocuments_xml_k
                                   ? (engine->slices_offsets[engine->steps_count] + 1) / 2
                                   : 0;
    return sz_subdocuments_frame_header_words_simt_k + engine->words_count + counters;
}

#pragma endregion Shapes

#pragma region Device Helpers

/** The index of the lowest set bit of a nonzero @p word. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_ctz_simt_(sz_u64_t word) {
    return (sz_size_t)__ffsll((long long)word) - 1;
}

/** The masks of one block, and what its classification hands the next, as the serial tier keeps
 *  them; a warp walking one document together also keeps its window's masks, 32 bytes to a lane,
 *  and hands the carries on from window to window. */
typedef struct sz_subdocuments_cursor_simt_t {
    sz_u8_t const *text;
    sz_size_t length, block_offset, window_end;
    sz_u64_t opens, closes, pending;
    sz_u64_t escaped, in_string, scalar, slashes;
    sz_u32_t lane_opens, lane_closes, lane_pending;
} sz_subdocuments_cursor_simt_t;

/** The class masks of 32 bytes, one bit per byte. */
typedef struct sz_subdocuments_classes_simt_t {
    sz_u32_t quotes, backslashes, opens, closes, separators, blanks, slashes;
} sz_subdocuments_classes_simt_t;

/** A vendor's lane steps for the 32 lanes of a warp walking one JSON document together. */
typedef struct sz_subdocuments_warp_simt_t {
    sz_u32_t (*ballot)(int);
    sz_u32_t (*shuffle)(sz_u32_t, unsigned);
    sz_u32_t (*shuffle_up)(sz_u32_t, unsigned);
} sz_subdocuments_warp_simt_t;

/** The top bit of every byte of @p word equal to the byte @p repeated repeats, with no carry
 *  between bytes. */
STRINGZILLA_DEVICE sz_u32_t sz_subdocuments_bytes_equal_simt_(sz_u32_t word, sz_u32_t repeated) {
    sz_u32_t const difference = word ^ repeated;
    return ~(((difference & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | difference) & 0x80808080u;
}

/** The top bits of the four bytes of @p flags gathered into a nibble, the first byte's lowest. */
STRINGZILLA_DEVICE sz_u64_t sz_subdocuments_nibble_simt_(sz_u32_t flags) {
    return (sz_u64_t)((flags * 0x00204081u) >> 28);
}

/** Classifies the 32 bytes at @p text [offset], those past @p length blank, four at a time from
 *  aligned words funnel-shifted into place. */
STRINGZILLA_DEVICE sz_subdocuments_classes_simt_t sz_subdocuments_classify_simt_(sz_u8_t const *text, sz_size_t length,
                                                                                 sz_size_t offset) {
    sz_u8_t const *const bytes = text + offset;
    unsigned const available = offset < length ? (unsigned)sz_min_of_two(length - offset, (sz_size_t)32) : 0;
    unsigned const misalignment = (unsigned)((sz_size_t)bytes & 3);
    sz_u32_t const *const aligned = (sz_u32_t const *)(bytes - misalignment);
    sz_subdocuments_classes_simt_t classes;
    // An aligned word holding one of the bytes lies in a page the device may read whole.
    sz_u32_t previous = available ? __ldg(aligned) : 0;
    classes.quotes = 0, classes.backslashes = 0, classes.opens = 0, classes.closes = 0, classes.separators = 0;
    classes.blanks = 0, classes.slashes = 0;
#pragma unroll
    for (unsigned index = 0; index != 8; ++index) {
        sz_u32_t const next = 4 * index + 4 < misalignment + available ? __ldg(aligned + index + 1) : 0;
        sz_u32_t const padding = 4 * index + 4 <= available ? 0u
                                 : 4 * index >= available   ? 0xFFFFFFFFu
                                                            : 0xFFFFFFFFu << (8 * (available - 4 * index));
        sz_u32_t const word = (__funnelshift_r(previous, next, misalignment * 8) & ~padding) | (0x20202020u & padding);
        sz_u32_t const folded = word | 0x20202020u;
        unsigned const shift = 4 * index;
        classes.quotes |= sz_subdocuments_nibble_simt_(sz_subdocuments_bytes_equal_simt_(word, 0x22222222u)) << shift;
        classes.backslashes |= sz_subdocuments_nibble_simt_(sz_subdocuments_bytes_equal_simt_(word, 0x5C5C5C5Cu))
                               << shift;
        classes.opens |= sz_subdocuments_nibble_simt_(sz_subdocuments_bytes_equal_simt_(folded, 0x7B7B7B7Bu)) << shift;
        classes.closes |= sz_subdocuments_nibble_simt_(sz_subdocuments_bytes_equal_simt_(folded, 0x7D7D7D7Du)) << shift;
        classes.separators |= sz_subdocuments_nibble_simt_(sz_subdocuments_bytes_equal_simt_(word, 0x3A3A3A3Au) |
                                                           sz_subdocuments_bytes_equal_simt_(word, 0x2C2C2C2Cu))
                              << shift;
        // Tab and carriage return are the bytes that 0x04 raises to 0x0D.
        classes.blanks |= sz_subdocuments_nibble_simt_(
                              sz_subdocuments_bytes_equal_simt_(word, 0x20202020u) |
                              sz_subdocuments_bytes_equal_simt_(word, 0x0A0A0A0Au) |
                              sz_subdocuments_bytes_equal_simt_(word | 0x04040404u, 0x0D0D0D0Du))
                          << shift;
        classes.slashes |= sz_subdocuments_nibble_simt_(sz_subdocuments_bytes_equal_simt_(word, 0x2F2F2F2Fu)) << shift;
        previous = next;
    }
    return classes;
}

/** Classifies the block at @p offset, padding a partial last one with blanks, as
 *  @c sz_subdocuments_classify_serial_ and @c sz_subdocuments_block_settle_ do. */
STRINGZILLA_DEVICE void sz_subdocuments_load_simt_(sz_subdocuments_cursor_simt_t *cursor, sz_size_t offset) {
    sz_subdocuments_classes_simt_t const low = sz_subdocuments_classify_simt_(cursor->text, cursor->length, offset);
    sz_subdocuments_classes_simt_t const high = sz_subdocuments_classify_simt_(cursor->text, cursor->length,
                                                                               offset + 32);
    sz_u64_t quotes = low.quotes | (sz_u64_t)high.quotes << 32;
    sz_u64_t const backslashes = low.backslashes | (sz_u64_t)high.backslashes << 32;
    sz_u64_t opens = low.opens | (sz_u64_t)high.opens << 32, closes = low.closes | (sz_u64_t)high.closes << 32;
    sz_u64_t separators = low.separators | (sz_u64_t)high.separators << 32;
    sz_u64_t const blanks = low.blanks | (sz_u64_t)high.blanks << 32;
    sz_u64_t const slashes = low.slashes | (sz_u64_t)high.slashes << 32;
    sz_u64_t escaped, in_string, scalars;
    if (!backslashes) escaped = cursor->escaped, cursor->escaped = 0;
    else {
        sz_u64_t const odd_bits = 0xAAAAAAAAAAAAAAAAull;
        sz_u64_t const potential_escape = backslashes & ~cursor->escaped;
        sz_u64_t const escape_and_terminal = (((potential_escape << 1) | odd_bits) - potential_escape) ^ odd_bits;
        escaped = escape_and_terminal ^ (backslashes | cursor->escaped);
        cursor->escaped = (escape_and_terminal & backslashes) >> 63;
    }
    quotes &= ~escaped;
    in_string = quotes;
    in_string ^= in_string << 1, in_string ^= in_string << 2, in_string ^= in_string << 4;
    in_string ^= in_string << 8, in_string ^= in_string << 16, in_string ^= in_string << 32;
    in_string ^= cursor->in_string;
    cursor->in_string = 0 - (in_string >> 63);
    cursor->slashes |= slashes & ~in_string;
    opens &= ~in_string, closes &= ~in_string, separators &= ~in_string;
    scalars = ~(in_string | quotes | blanks | opens | closes | separators);
    cursor->block_offset = offset, cursor->opens = opens, cursor->closes = closes;
    cursor->pending = opens | closes | separators | quotes | (scalars & ~(scalars << 1 | cursor->scalar));
    cursor->scalar = scalars >> 63;
}

/**
 *  @brief Classifies the 1,024 bytes from @p window, 32 to a lane of the warp walking the document,
 *      every lane calling it at once, so each lane keeps its own bytes' masks.
 *
 *  A lane holding backslashes alone hands on the escape it receives, and any other the parity of
 *  its last run, so one ballot finds every lane's escape; a ballot of quote parities tells each
 *  lane whether it starts inside a string, and a shuffle hands on the scalar carry. The window's
 *  own carries come from its last lanes, as a block's do from its last bytes.
 */
STRINGZILLA_DEVICE void sz_subdocuments_window_simt_(sz_subdocuments_cursor_simt_t *cursor, sz_size_t window,
                                                     sz_u32_t (*ballot)(int), sz_u32_t (*shuffle)(sz_u32_t, unsigned)) {
    unsigned const lane = threadIdx.x & 31u;
    sz_u32_t const below = (1u << lane) - 1u, odd_bits = 0xAAAAAAAAu;
    sz_subdocuments_classes_simt_t const classes = sz_subdocuments_classify_simt_(cursor->text, cursor->length,
                                                                                  window + 32 * lane);
    sz_u32_t const backslashes = classes.backslashes;
    sz_u32_t const settled = (((backslashes << 1) | odd_bits) - backslashes) ^ odd_bits;
    sz_u32_t const blocking = ~ballot(backslashes == 0xFFFFFFFFu), handing = ballot((settled & backslashes) >> 31);
    sz_u32_t const escape_in = blocking & below ? handing >> (31 - __clz(blocking & below)) & 1u
                                                : (sz_u32_t)cursor->escaped;
    sz_u32_t const potential_escape = backslashes & ~escape_in;
    sz_u32_t const escape_and_terminal = (((potential_escape << 1) | odd_bits) - potential_escape) ^ odd_bits;
    sz_u32_t const quotes = classes.quotes & ~(escape_and_terminal ^ (backslashes | escape_in));
    sz_u32_t const odd_lanes = ballot(__popc(quotes) & 1);
    sz_u32_t in_string = quotes, opens, closes, separators, scalars, scalar_before;
    in_string ^= in_string << 1, in_string ^= in_string << 2, in_string ^= in_string << 4;
    in_string ^= in_string << 8, in_string ^= in_string << 16;
    in_string ^= 0u - (((sz_u32_t)__popc(odd_lanes & below) ^ (sz_u32_t)cursor->in_string) & 1u);
    opens = classes.opens & ~in_string, closes = classes.closes & ~in_string;
    separators = classes.separators & ~in_string;
    scalars = ~(in_string | quotes | classes.blanks | opens | closes | separators);
    cursor->slashes |= ballot((classes.slashes & ~in_string) != 0);
    cursor->lane_opens = opens, cursor->lane_closes = closes;
    scalar_before = shuffle(scalars >> 31, (lane + 31) & 31u);
    cursor->lane_pending = opens | closes | separators | quotes |
                           (scalars & ~(scalars << 1 | (lane ? scalar_before : (sz_u32_t)cursor->scalar)));
    cursor->escaped = blocking ? handing >> (31 - __clz(blocking)) & 1u : cursor->escaped;
    cursor->in_string = 0 - (sz_u64_t)(((sz_u32_t)__popc(odd_lanes) ^ (sz_u32_t)cursor->in_string) & 1u);
    cursor->scalar = shuffle(scalars >> 31, 31);
    cursor->window_end = window + 1024;
}

/** The bits of the 32 bytes from @p base at or past @p from. */
STRINGZILLA_DEVICE sz_u32_t sz_subdocuments_from_mask_simt_(sz_size_t base, sz_size_t from) {
    return from <= base ? 0xFFFFFFFFu : from - base >= 32 ? 0u : 0xFFFFFFFFu << (from - base);
}

/** The position of the set bit of @p bits with @p rank set bits below it. */
STRINGZILLA_DEVICE unsigned sz_subdocuments_select_simt_(sz_u32_t bits, unsigned rank) {
    unsigned position = 0;
    for (unsigned width = 16; width; width >>= 1) {
        unsigned const lower = (unsigned)__popc(bits & ((1u << width) - 1u));
        if (rank >= lower) rank -= lower, bits >>= width, position += width;
    }
    return position;
}

/** The sum of @p value over this lane and every lane below it. */
STRINGZILLA_DEVICE sz_u32_t sz_subdocuments_lanes_sum_simt_(sz_subdocuments_warp_simt_t const *warp, sz_u32_t value) {
    unsigned const lane = threadIdx.x & 31u;
    for (unsigned delta = 1; delta != 32; delta <<= 1) {
        sz_u32_t const below = warp->shuffle_up(value, delta);
        if (lane >= delta) value += below;
    }
    return value;
}

/** The least of @p value over this lane and every lane below it. */
STRINGZILLA_DEVICE int sz_subdocuments_lanes_min_simt_(sz_subdocuments_warp_simt_t const *warp, int value) {
    unsigned const lane = threadIdx.x & 31u;
    for (unsigned delta = 1; delta != 32; delta <<= 1) {
        int const below = (int)warp->shuffle_up((sz_u32_t)value, delta);
        if (lane >= delta && below < value) value = below;
    }
    return value;
}

/** Classifies the window past the cursor's last, every lane at once, or returns false past the
 *  document. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_advance_simt_(sz_subdocuments_cursor_simt_t *cursor,
                                                           sz_subdocuments_warp_simt_t const *warp) {
    if (cursor->window_end >= cursor->length) return sz_false_k;
    sz_subdocuments_window_simt_(cursor, cursor->window_end, warp->ballot, warp->shuffle);
    return sz_true_k;
}

/** The first structural byte at or past @p from, classifying windows as it passes them, or
 *  @c STRINGZILLA_SIZE_MAX past the document; every lane calls it at once. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_next_simt_(sz_subdocuments_cursor_simt_t *cursor, sz_size_t from,
                                                        sz_subdocuments_warp_simt_t const *warp) {
    unsigned const lane = threadIdx.x & 31u;
    for (;; from = cursor->window_end) {
        sz_u32_t pending, holding;
        if (from >= cursor->window_end && !sz_subdocuments_advance_simt_(cursor, warp)) return STRINGZILLA_SIZE_MAX;
        pending = cursor->lane_pending & sz_subdocuments_from_mask_simt_(cursor->window_end - 1024 + 32 * lane, from);
        holding = warp->ballot(pending != 0);
        if (holding) {
            unsigned const first = (unsigned)__ffs((int)holding) - 1;
            return cursor->window_end - 1024 + 32 * first + (unsigned)__ffs((int)warp->shuffle(pending, first)) - 1;
        }
    }
}

/**
 *  @brief The bracket closing the container whose opening bracket lies just before @p from, or
 *      @c STRINGZILLA_SIZE_MAX past the document, counting brackets kind-blind a window at a time.
 *
 *  Each lane counts its bytes' brackets and the lowest depth they reach, a sum over the lanes
 *  places every lane's first byte, and the first lane whose bytes close the container finds where.
 */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_skip_simt_(sz_subdocuments_cursor_simt_t *cursor, sz_size_t from,
                                                        sz_subdocuments_warp_simt_t const *warp) {
    unsigned const lane = threadIdx.x & 31u;
    sz_i64_t depth = 1;
    for (;; from = cursor->window_end) {
        sz_size_t window;
        sz_u32_t mask, opens, closes, brackets, closing, net, before_lane;
        int running = 0, lowest = 0;
        sz_i64_t before;
        if (from >= cursor->window_end && !sz_subdocuments_advance_simt_(cursor, warp)) return STRINGZILLA_SIZE_MAX;
        window = cursor->window_end - 1024;
        mask = sz_subdocuments_from_mask_simt_(window + 32 * lane, from);
        opens = cursor->lane_opens & mask, closes = cursor->lane_closes & mask;
        for (brackets = closes ? opens | closes : 0; brackets; brackets &= brackets - 1)
            running += opens & brackets & (0u - brackets) ? 1 : -1, lowest = running < lowest ? running : lowest;
        net = (sz_u32_t)(__popc(opens) - __popc(closes));
        before_lane = sz_subdocuments_lanes_sum_simt_(warp, net) - net;
        before = depth + (sz_i64_t)(int)before_lane;
        closing = warp->ballot(before + lowest <= 0);
        if (closing) {
            unsigned const first = (unsigned)__ffs((int)closing) - 1;
            sz_u32_t found = 0;
            for (brackets = lane == first ? opens | closes : 0; brackets; brackets &= brackets - 1) {
                if (opens & brackets & (0u - brackets)) ++before;
                else if (--before == 0) {
                    found = 32 * lane + (unsigned)__ffs((int)brackets) - 1;
                    break;
                }
            }
            return window + warp->shuffle(found, first);
        }
        depth += (sz_i64_t)(int)warp->shuffle(before_lane + net, 31);
    }
}

/** Bytes of the scalar at @p position, or zero unless it is a number, @c true, @c false or @c null,
 *  as @c sz_subdocuments_scalar_length_ reads it. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_scalar_length_simt_(sz_u8_t const *text, sz_size_t position,
                                                                 sz_size_t length) {
    sz_u8_t const *const scalar = text + position;
    unsigned const misalignment = (unsigned)((sz_size_t)scalar & 3);
    sz_u32_t const *const aligned = (sz_u32_t const *)(scalar - misalignment);
    sz_size_t const available = length - position;
    sz_size_t run = 0, cursor = 0, digits, chunk;
    // An aligned word holding a byte of the document lies in a page the device may read whole.
    sz_u32_t previous = __ldg(aligned), first = 0, others = 0;
    // The run ends at the first blank, separator, quote or bracket, four bytes at a time, noting
    // whether any byte of it is not a digit.
    for (chunk = 0;; ++chunk) {
        sz_u32_t const next = 4 * chunk + 4 - misalignment < available ? __ldg(aligned + chunk + 1) : 0;
        sz_u32_t const word = __funnelshift_r(previous, next, misalignment * 8), folded = word | 0x20202020u;
        sz_size_t const left = available - 4 * chunk;
        sz_u32_t const outside = left >= 4 ? 0u : 0x80808080u << (8 * left);
        sz_u32_t const stops = sz_subdocuments_bytes_equal_simt_(word, 0x20202020u) |
                               sz_subdocuments_bytes_equal_simt_(word, 0x0A0A0A0Au) |
                               sz_subdocuments_bytes_equal_simt_(word | 0x04040404u, 0x0D0D0D0Du) |
                               sz_subdocuments_bytes_equal_simt_(word, 0x2C2C2C2Cu) |
                               sz_subdocuments_bytes_equal_simt_(word, 0x3A3A3A3Au) |
                               sz_subdocuments_bytes_equal_simt_(word, 0x22222222u) |
                               sz_subdocuments_bytes_equal_simt_(folded, 0x7B7B7B7Bu) |
                               sz_subdocuments_bytes_equal_simt_(folded, 0x7D7D7D7Du) | outside;
        unsigned const stop = stops ? (unsigned)(__ffs((int)stops) - 1) / 8 : 4;
        sz_u32_t const shifted = word ^ 0x30303030u;
        others |= ((shifted & 0x7F7F7F7Fu) + 0x76767676u | shifted) & 0x80808080u &
                  (stop == 4 ? 0xFFFFFFFFu : (1u << (8 * stop)) - 1u);
        if (!chunk) first = word;
        run = 4 * chunk + stop;
        if (stop != 4) break;
        previous = next;
    }
    if (run == 4 && first == 0x65757274u) return run;
    if (run == 4 && first == 0x6C6C756Eu) return run;
    if (run == 5 && first == 0x736C6166u && scalar[4] == 'e') return run;
    if (run && !others) return run == 1 || (first & 0xFFu) != '0' ? run : 0;
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
STRINGZILLA_DEVICE sz_u32_t sz_subdocuments_hex_unit_simt_(sz_u8_t const *digits) {
    sz_u32_t unit = 0;
    for (sz_size_t index = 0; index != 4; ++index) {
        sz_u8_t const digit = digits[index], lowered = (sz_u8_t)(digit | 0x20);
        if (digit >= '0' && digit <= '9') unit = unit << 4 | (sz_u32_t)(digit - '0');
        else if (lowered >= 'a' && lowered <= 'f') unit = unit << 4 | (sz_u32_t)(lowered - 'a' + 10);
        else return 0xFFFFFFFFu;
    }
    return unit;
}

/**
 *  @brief Decodes the unit of a `"`-quoted string body at @p source [*read], a byte or one escape,
 *      into @p encoded as @c sz_subdocuments_unescape_ does, moving @p read past it.
 *  @return Bytes it encodes to, or zero for a raw control byte, an unknown
 *      escape or an unpaired surrogate.
 */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_unescape_unit_simt_(sz_u8_t const *source, sz_size_t length,
                                                                 sz_size_t *read, sz_u8_t *encoded) {
    sz_u8_t byte = source[(*read)++];
    sz_u32_t rune, low;
    if (byte < 0x20) return 0;
    if (byte != '\\') return encoded[0] = byte, 1;
    if (*read == length) return 0;
    byte = source[(*read)++];
    if (byte == 'b') return encoded[0] = '\b', 1;
    if (byte == 'f') return encoded[0] = '\f', 1;
    if (byte == 'n') return encoded[0] = '\n', 1;
    if (byte == 'r') return encoded[0] = '\r', 1;
    if (byte == 't') return encoded[0] = '\t', 1;
    if (byte != 'u') return byte == '/' || byte == '\\' || byte == '"' ? (encoded[0] = byte, 1) : 0;
    if (length - *read < 4) return 0;
    rune = sz_subdocuments_hex_unit_simt_(source + *read);
    *read += 4;
    if (rune > 0xFFFF || (rune >= 0xDC00 && rune <= 0xDFFF)) return 0;
    if (rune >= 0xD800 && rune <= 0xDBFF) {
        if (length - *read < 6 || source[*read] != '\\' || source[*read + 1] != 'u') return 0;
        low = sz_subdocuments_hex_unit_simt_(source + *read + 2);
        if (low < 0xDC00 || low > 0xDFFF) return 0;
        *read += 6;
        rune = 0x10000 + ((rune - 0xD800) << 10) + (low - 0xDC00);
    }
    if (rune < 0x80) return encoded[0] = (sz_u8_t)rune, 1;
    if (rune < 0x800) return encoded[0] = (sz_u8_t)(0xC0 | rune >> 6), encoded[1] = (sz_u8_t)(0x80 | (rune & 0x3F)), 2;
    if (rune < 0x10000)
        return encoded[0] = (sz_u8_t)(0xE0 | rune >> 12), encoded[1] = (sz_u8_t)(0x80 | (rune >> 6 & 0x3F)),
               encoded[2] = (sz_u8_t)(0x80 | (rune & 0x3F)), 3;
    return encoded[0] = (sz_u8_t)(0xF0 | rune >> 18), encoded[1] = (sz_u8_t)(0x80 | (rune >> 12 & 0x3F)),
           encoded[2] = (sz_u8_t)(0x80 | (rune >> 6 & 0x3F)), encoded[3] = (sz_u8_t)(0x80 | (rune & 0x3F)), 4;
}

/**
 *  @brief Unescapes a `"`-quoted string body, writing only the first @p capacity bytes.
 *  @return The unescaped length, or @c STRINGZILLA_SIZE_MAX for a malformed escape.
 */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_unescape_simt_(sz_u8_t const *source, sz_size_t length, sz_u8_t *target,
                                                            sz_size_t capacity) {
    sz_size_t read = 0, written = 0;
    sz_u8_t encoded[4];
    while (read != length) {
        sz_size_t const encoded_length = sz_subdocuments_unescape_unit_simt_(source, length, &read, encoded);
        if (!encoded_length) return STRINGZILLA_SIZE_MAX;
        for (sz_size_t index = 0; index != encoded_length; ++index, ++written)
            if (written < capacity) target[written] = encoded[index];
    }
    return written;
}

/** The name a key spells, or @c sz_subdocuments_no_name_k, by the hash and probing of
 *  @c sz_subdocuments_name_slot_ ; keys are read a byte at a time, as nothing aligns them. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_name_simt_(sz_subdocuments_engine_t const *engine, sz_u8_t const *key,
                                                        sz_size_t key_length) {
    sz_size_t const mask = engine->names_slots_count - 1;
    sz_u64_t hash = key_length * 0x9E3779B97F4A7C15ull, tail = 0;
    sz_size_t index = 0, slot;
    for (; index + 8 <= key_length; index += 8) {
        sz_u64_t word = 0;
        for (sz_size_t byte = 8; byte-- != 0;) word = word << 8 | key[index + byte];
        hash = (hash ^ word) * 0xFF51AFD7ED558CCDull;
        hash ^= hash >> 32;
    }
    for (; index != key_length; ++index) tail = tail << 8 | key[index];
    hash = (hash ^ tail) * 0xC4CEB9FE1A85EC53ull;
    for (slot = (sz_size_t)(hash ^ hash >> 32) & mask;; slot = (slot + 1) & mask) {
        sz_u32_t const name = engine->names_slots[slot];
        sz_size_t name_length;
        if (name == sz_subdocuments_no_name_k) return name;
        name_length = engine->names_offsets[name + 1] - engine->names_offsets[name];
        if (name_length != key_length) continue;
        for (index = 0;
             index != key_length && (sz_u8_t)engine->names_data[engine->names_offsets[name] + index] == key[index];)
            ++index;
        if (index == key_length) return name;
    }
}

/**
 *  @brief The name the @p length bytes of a document at @p key spell, as
 *      @ref sz_subdocuments_name_simt_ finds it for a key of at most @p key_bytes bytes,
 *      reading them as aligned words up to the closing quote that follows them.
 *  @return The name, @c sz_subdocuments_no_name_k, or @c STRINGZILLA_SIZE_MAX for a key holding a
 *      backslash, which only its unescaped bytes name.
 */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_key_name_simt_(sz_subdocuments_engine_t const *engine, sz_u8_t const *key,
                                                            sz_size_t length, sz_size_t key_bytes) {
    unsigned const misalignment = (unsigned)((sz_size_t)key & 3);
    sz_u32_t const *const aligned = (sz_u32_t const *)(key - misalignment);
    sz_size_t const mask = engine->names_slots_count - 1;
    sz_u64_t hash = length * 0x9E3779B97F4A7C15ull, word = 0;
    sz_u32_t previous = __ldg(aligned), backslashes = 0;
    sz_size_t chunk, slot, index, tail_bytes = length & 7;
    for (chunk = 0; 4 * chunk < length; ++chunk) {
        // An aligned word holding a byte up to the closing quote lies in the document.
        sz_u32_t const next = 4 * chunk + 4 - misalignment <= length ? __ldg(aligned + chunk + 1) : 0;
        sz_u32_t const left = (sz_u32_t)sz_min_of_two(length - 4 * chunk, (sz_size_t)4);
        sz_u32_t const kept = left == 4 ? 0xFFFFFFFFu : (1u << (8 * left)) - 1u;
        sz_u32_t const bytes = __funnelshift_r(previous, next, misalignment * 8) & kept;
        backslashes |= sz_subdocuments_bytes_equal_simt_(bytes, 0x5C5C5C5Cu) & kept;
        word |= (sz_u64_t)bytes << (chunk & 1 ? 32 : 0);
        if (chunk & 1 && 4 * chunk + 4 <= length)
            hash = (hash ^ word) * 0xFF51AFD7ED558CCDull, hash ^= hash >> 32, word = 0;
        previous = next;
    }
    if (backslashes) return STRINGZILLA_SIZE_MAX;
    if (length > key_bytes) return sz_subdocuments_no_name_k;
    // The bytes past the last whole word hash first byte highest.
    if (tail_bytes)
        word = ((sz_u64_t)__byte_perm((sz_u32_t)word, 0, 0x0123) << 32 |
                __byte_perm((sz_u32_t)(word >> 32), 0, 0x0123)) >>
               (8 * (8 - tail_bytes));
    hash = (hash ^ word) * 0xC4CEB9FE1A85EC53ull;
    for (slot = (sz_size_t)(hash ^ hash >> 32) & mask;; slot = (slot + 1) & mask) {
        sz_u32_t const name = engine->names_slots[slot];
        if (name == sz_subdocuments_no_name_k) return name;
        if (engine->names_offsets[name + 1] - engine->names_offsets[name] != length) continue;
        for (index = 0;
             index != length && (sz_u8_t)engine->names_data[engine->names_offsets[name] + index] == key[index];)
            ++index;
        if (index == length) return name;
    }
}

/** Whether @p slice selects element @p position of an array of @p length elements, as
 *  @c sz_subdocuments_slice_selects_ normalizes and clamps the bounds. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_slice_selects_simt_(sz_subdocuments_slice_t const *slice,
                                                                 sz_i64_t position, sz_i64_t length) {
    sz_i64_t lower, upper;
    if (slice->step > 0) {
        lower = slice->begin >= 0 ? sz_min_of_two(slice->begin, length)
                                  : sz_max_of_two(length + slice->begin, (sz_i64_t)0);
        upper = slice->end >= 0 ? sz_min_of_two(slice->end, length) : sz_max_of_two(length + slice->end, (sz_i64_t)0);
        return (sz_bool_t)(position >= lower && position < upper && (position - lower) % slice->step == 0);
    }
    if (slice->step < 0) {
        upper = slice->begin >= 0 ? sz_min_of_two(slice->begin, length - 1)
                                  : sz_max_of_two(length + slice->begin, (sz_i64_t)-1);
        lower = slice->end >= 0 ? sz_min_of_two(slice->end, length - 1)
                                : sz_max_of_two(length + slice->end, (sz_i64_t)-1);
        return (sz_bool_t)(position > lower && position <= upper && (upper - position) % -slice->step == 0);
    }
    return sz_false_k;
}

/**
 *  @brief Word @p word of the live set of a child of a node live in @p live, joined with @p carried
 *      unless null: an object's member named @p selector, or, for an @p element, an array's element
 *      at position @p selector.
 *
 *  Both settle as @c sz_subdocuments_member_live_ and @c sz_subdocuments_element_live_ settle them.
 *
 *  @param[inout] carry The top advancing bit of the word before, zero for the first word.
 *  @param[in] length The array's element count, or zero while it is open, when the steps
 *      it settles select tentatively.
 */
STRINGZILLA_DEVICE sz_u64_t sz_subdocuments_child_word_simt_(sz_subdocuments_engine_t const *engine,
                                                             sz_u64_t const *live, sz_u64_t const *carried,
                                                             sz_size_t word, sz_u64_t *carry, sz_bool_t element,
                                                             sz_size_t selector, sz_size_t length) {
    sz_u64_t const parent = live[word] | (carried ? carried[word] : 0);
    sz_u64_t const advancing = parent & ~engine->final_steps[word];
    sz_u64_t const candidates = advancing << 1 | *carry;
    sz_u64_t selected;
    *carry = advancing >> 63;
    if (!element)
        selected = candidates &
                   ((selector != sz_subdocuments_no_name_k ? engine->names_steps[selector * engine->words_count + word]
                                                           : 0) |
                    engine->wildcard_steps[word]);
    else {
        sz_i64_t const known_length = length ? (sz_i64_t)length : (sz_i64_t)selector + 1;
        sz_u64_t const tentative = length ? 0 : candidates & engine->relative_steps[word];
        sz_u64_t positional = candidates & ~engine->wildcard_steps[word] & ~tentative;
        selected = (candidates & engine->wildcard_steps[word]) | tentative;
        for (; positional; positional &= positional - 1) {
            sz_size_t const step = word * 64 + sz_subdocuments_ctz_simt_(positional);
            for (sz_u32_t slice = engine->slices_offsets[step]; slice != engine->slices_offsets[step + 1]; ++slice)
                if (sz_subdocuments_slice_selects_simt_(&engine->slices[slice], (sz_i64_t)selector, known_length)) {
                    selected |= positional & (0 - positional);
                    break;
                }
        }
    }
    return selected | (parent & engine->descendant_steps[word]);
}

/** The live set of an array's element as @c sz_subdocuments_element_live_ settles it, tentatively
 *  for the steps an array's length settles while @p length is zero. */
STRINGZILLA_DEVICE void sz_subdocuments_element_live_simt_(sz_subdocuments_engine_t const *engine,
                                                           sz_u64_t const *parent_live, sz_size_t position,
                                                           sz_size_t length, sz_u64_t *live) {
    sz_u64_t carry = 0;
    for (sz_size_t word = 0; word != engine->words_count; ++word)
        live[word] = sz_subdocuments_child_word_simt_(engine, parent_live, (sz_u64_t const *)STRINGZILLA_NULL, word,
                                                      &carry, sz_true_k, position, length);
}

/** Whether a member of an object live in @p parent_live, joined with @p carried unless null, needs
 *  its key looked up. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_needs_name_simt_(sz_subdocuments_engine_t const *engine,
                                                              sz_u64_t const *parent_live, sz_u64_t const *carried) {
    sz_u64_t carry = 0, needed = 0;
    for (sz_size_t word = 0; word != engine->words_count; ++word) {
        sz_u64_t const advancing = (parent_live[word] | (carried ? carried[word] : 0)) & ~engine->final_steps[word];
        needed |= (advancing << 1 | carry) & ~engine->wildcard_steps[word];
        carry = advancing >> 63;
    }
    return (sz_bool_t)(needed != 0);
}

/** The live set of an object's member as @c sz_subdocuments_member_live_ settles it. */
STRINGZILLA_DEVICE void sz_subdocuments_member_live_simt_(sz_subdocuments_engine_t const *engine,
                                                          sz_u64_t const *parent_live, sz_size_t name, sz_u64_t *live) {
    sz_u64_t carry = 0;
    for (sz_size_t word = 0; word != engine->words_count; ++word)
        live[word] = sz_subdocuments_child_word_simt_(engine, parent_live, (sz_u64_t const *)STRINGZILLA_NULL, word,
                                                      &carry, sz_false_k, name, 0);
}

/** Makes every filter step advancing into a node select it tentatively, with its filters' query
 *  roots, as @c sz_subdocuments_filters_tentative_ does. */
STRINGZILLA_DEVICE void sz_subdocuments_filters_tentative_simt_(sz_subdocuments_engine_t const *engine,
                                                                sz_u64_t const *parent_live, sz_u64_t *live) {
    sz_size_t const words = engine->words_count;
    sz_u64_t carried = 0, any = 0;
    sz_size_t word, filter;
    for (word = 0; word != words; ++word) {
        sz_u64_t const advancing = parent_live[word] & ~engine->final_steps[word];
        sz_u64_t const filtered = (advancing << 1 | carried) & engine->filter_steps[word];
        live[word] |= filtered, any |= filtered, carried = advancing >> 63;
    }
    if (!any) return;
    for (filter = 0; filter != engine->filters_count; ++filter) {
        sz_size_t const step = engine->filters[filter].step;
        if (live[step >> 6] >> (step & 63) & 1)
            for (word = 0; word != words; ++word) live[word] |= engine->filters_roots[filter * words + word];
    }
}

/** Reports a node once per final step of its @p live set, as @c sz_subdocuments_report_ does. */
STRINGZILLA_DEVICE void sz_subdocuments_report_simt_(sz_subdocuments_engine_t const *engine, sz_u64_t const *live,
                                                     sz_size_t document, sz_size_t byte_offset, sz_size_t byte_length,
                                                     sz_match_t *matches, sz_size_t matches_capacity,
                                                     sz_size_t *emitted) {
    for (sz_size_t word = 0; word != engine->words_count; ++word)
        for (sz_u64_t reported = live[word] & engine->final_steps[word]; reported; reported &= reported - 1) {
            if (*emitted < matches_capacity) {
                sz_match_t *const match = &matches[*emitted];
                match->haystack_index = document;
                match->needle_index = engine->steps_paths[word * 64 + sz_subdocuments_ctz_simt_(reported)];
                match->byte_offset = byte_offset, match->byte_length = byte_length;
            }
            ++*emitted;
        }
}

/** Settles the lengths of the matches `[begin, end)` that fit, all starting at one byte. */
STRINGZILLA_DEVICE void sz_subdocuments_close_simt_(sz_match_t *matches, sz_size_t matches_capacity, sz_size_t begin,
                                                    sz_size_t end, sz_size_t end_offset) {
    for (end = sz_min_of_two(end, matches_capacity); begin < end; ++begin)
        matches[begin].byte_length = end_offset - matches[begin].byte_offset;
}

/** Settles the lengths of the matches @p frame 's container reported that fit, closing at
 *  @p end_offset, from the frame every lane wrote rather than the matches one lane may have. */
STRINGZILLA_DEVICE void sz_subdocuments_close_frame_simt_(sz_match_t *matches, sz_size_t matches_capacity,
                                                          sz_subdocuments_frame_t const *frame, sz_size_t end_offset) {
    sz_size_t const end = sz_min_of_two(frame->matches_end, matches_capacity);
    for (sz_size_t index = frame->matches_begin; index < end; ++index)
        matches[index].byte_length = end_offset - frame->byte_offset;
}

/** Whether no step is live in @p live, so a replay skips the subtree below. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_dead_simt_(sz_u64_t const *live, sz_size_t words) {
    sz_u64_t any = 0;
    for (sz_size_t word = 0; word != words; ++word) any |= live[word];
    return (sz_bool_t)(any == 0);
}

/** The number a validated JSON number spells over @p length bytes, rounded as
 *  @c sz_subdocuments_parse_number_ rounds it. */
STRINGZILLA_DEVICE sz_f64_t sz_subdocuments_number_simt_(sz_u8_t const *text, sz_size_t length) {
    sz_size_t cursor = text[0] == '-';
    sz_u64_t mantissa = 0;
    sz_i64_t exponent = 0, magnitude = 0;
    sz_f64_t scale = 1, result;
    if (text[cursor] == '0') ++cursor;
    else
        for (; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (mantissa < 1000000000000000000ull) mantissa = mantissa * 10 + (sz_u64_t)(text[cursor] - '0');
            else ++exponent;
    if (cursor != length && text[cursor] == '.')
        for (++cursor; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (mantissa < 1000000000000000000ull)
                mantissa = mantissa * 10 + (sz_u64_t)(text[cursor] - '0'), --exponent;
    if (cursor != length && (text[cursor] | 0x20) == 'e') {
        sz_bool_t const negative_exponent = (sz_bool_t)(++cursor != length && text[cursor] == '-');
        if (cursor != length && (text[cursor] == '+' || text[cursor] == '-')) ++cursor;
        for (; cursor != length && text[cursor] >= '0' && text[cursor] <= '9'; ++cursor)
            if (magnitude < 100000) magnitude = magnitude * 10 + (text[cursor] - '0');
        exponent += negative_exponent ? -magnitude : magnitude;
    }
    for (magnitude = exponent < 0 ? -exponent : exponent; magnitude > 0 && scale < 1e308; --magnitude) scale *= 10;
    result = exponent < 0 ? (sz_f64_t)mantissa / scale : (sz_f64_t)mantissa * scale;
    return text[0] == '-' ? -result : result;
}

/** The value of the node spanning @p length bytes at @p bytes, as
 *  @c sz_subdocuments_node_value_ reads it. */
STRINGZILLA_DEVICE sz_subdocuments_value_t sz_subdocuments_node_value_simt_(sz_u8_t const *bytes, sz_size_t length) {
    sz_subdocuments_value_t value;
    value.number = 0, value.bytes = (sz_cptr_t)bytes, value.length = length, value.escaped = sz_false_k;
    if (bytes[0] == '"')
        value.kind = sz_subdocuments_kind_string_k, ++value.bytes, value.length -= 2, value.escaped = sz_true_k;
    else if (bytes[0] == '{') value.kind = sz_subdocuments_kind_object_k;
    else if (bytes[0] == '[') value.kind = sz_subdocuments_kind_array_k;
    else if (bytes[0] == 't') value.kind = sz_subdocuments_kind_true_k;
    else if (bytes[0] == 'f') value.kind = sz_subdocuments_kind_false_k;
    else if (bytes[0] == 'n') value.kind = sz_subdocuments_kind_null_k;
    else value.kind = sz_subdocuments_kind_number_k, value.number = sz_subdocuments_number_simt_(bytes, length);
    return value;
}

/** Orders two strings as @c sz_subdocuments_order_strings_ does, a document's one
 *  unescaped into @p buffer. */
STRINGZILLA_DEVICE int sz_subdocuments_order_strings_simt_(sz_subdocuments_value_t const *left,
                                                           sz_subdocuments_value_t const *right, sz_u8_t *buffer,
                                                           sz_size_t capacity) {
    sz_u8_t const *left_bytes = (sz_u8_t const *)left->bytes, *right_bytes = (sz_u8_t const *)right->bytes;
    sz_size_t left_length = left->length, right_length = right->length, index, shared;
    if (left->escaped)
        left_length = sz_subdocuments_unescape_simt_(left_bytes, left_length, buffer, capacity), left_bytes = buffer;
    if (right->escaped)
        right_length = sz_subdocuments_unescape_simt_(right_bytes, right_length, buffer, capacity),
        right_bytes = buffer;
    shared = sz_min_of_two(left_length, right_length);
    for (index = 0; index != shared; ++index)
        if (left_bytes[index] != right_bytes[index]) return left_bytes[index] < right_bytes[index] ? -1 : 1;
    return left_length == right_length ? 0 : left_length < right_length ? -1 : 1;
}

/** RFC 9535's comparison of @p left with @p right by @p comparison, as
 *  @c sz_subdocuments_compare_ decides it. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_compare_simt_(sz_subdocuments_value_t const *left,
                                                           sz_subdocuments_value_t const *right, sz_u32_t comparison,
                                                           sz_u8_t *buffer, sz_size_t capacity) {
    sz_bool_t equal = sz_false_k, less = sz_false_k, greater = sz_false_k;
    if (left->kind == right->kind && left->kind == sz_subdocuments_kind_number_k)
        equal = (sz_bool_t)(left->number == right->number), less = (sz_bool_t)(left->number < right->number),
        greater = (sz_bool_t)(left->number > right->number);
    else if (left->kind == right->kind && left->kind == sz_subdocuments_kind_string_k) {
        int const order = sz_subdocuments_order_strings_simt_(left, right, buffer, capacity);
        equal = (sz_bool_t)(order == 0), less = (sz_bool_t)(order < 0), greater = (sz_bool_t)(order > 0);
    }
    else if (left->kind == right->kind) equal = sz_true_k;
    if (comparison == sz_subdocuments_comparison_equal_k) return equal;
    if (comparison == sz_subdocuments_comparison_not_equal_k) return (sz_bool_t)!equal;
    if (comparison == sz_subdocuments_comparison_less_k) return less;
    if (comparison == sz_subdocuments_comparison_less_equal_k) return (sz_bool_t)(less || equal);
    if (comparison == sz_subdocuments_comparison_greater_k) return greater;
    return (sz_bool_t)(greater || equal);
}

/** RFC 9535's `length()`, as @c sz_subdocuments_length_ counts it. */
STRINGZILLA_DEVICE sz_subdocuments_value_t sz_subdocuments_length_simt_(sz_subdocuments_value_t const *value) {
    sz_u8_t const *const bytes = (sz_u8_t const *)value->bytes;
    sz_subdocuments_value_t result;
    sz_size_t count = 0, index;
    result.kind = sz_subdocuments_kind_number_k, result.bytes = STRINGZILLA_NULL, result.length = 0;
    result.escaped = sz_false_k;
    if (value->kind == sz_subdocuments_kind_string_k)
        for (index = 0; index < value->length; ++index) {
            if (bytes[index] == '\\' && value->escaped) {
                ++count;
                if (++index + 4 >= value->length || bytes[index] != 'u') continue;
                // A high surrogate's escape and its pair's spell one codepoint.
                sz_u32_t const unit = sz_subdocuments_hex_unit_simt_(bytes + index + 1);
                index += 4;
                if (unit >= 0xD800 && unit <= 0xDBFF) index += 6;
            }
            else count += (bytes[index] & 0xC0) != 0x80;
        }
    else if (value->kind == sz_subdocuments_kind_array_k || value->kind == sz_subdocuments_kind_object_k) {
        sz_size_t depth = 0;
        sz_bool_t content = sz_false_k;
        for (index = 1; index + 1 < value->length; ++index) {
            sz_u8_t const byte = bytes[index];
            if (byte == '"') {
                for (++index; index + 1 < value->length && bytes[index] != '"'; ++index) index += bytes[index] == '\\';
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

/**
 *  @brief Whether pattern @p pattern matches the string @p value, as @c sz_subdocuments_regex_test_
 *      decides it, streaming the pattern's tag and the unescaped string through the forward DFA of
 *      @p regex in place.
 *
 *  Every tagged pattern is anchored by `\A`, so the one walk from the start answers, and the first
 *  accepting state settles it, acceptance trailing the match's end by one byte.
 */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_regex_test_simt_(sz_regex_engine_t const *regex,
                                                              sz_subdocuments_value_t const *value, sz_size_t pattern) {
    sz_u8_t const *const bytes = (sz_u8_t const *)value->bytes;
    sz_u32_t const *const rows = regex->forward_rows;
    sz_u32_t const accepts = regex->forward_accepts_begin;
    sz_u32_t state = rows[regex->forward_starts[0] + regex->byte_to_class[1]];
    sz_size_t power = 1, read = 0;
    sz_u8_t encoded[4];
    if (value->kind != sz_subdocuments_kind_string_k) return sz_false_k;
    // A broken escape in a string no walk validated matches nothing, wherever the match would end.
    if (value->escaped && sz_subdocuments_unescape_simt_(bytes, value->length, encoded, 0) == STRINGZILLA_SIZE_MAX)
        return sz_false_k;
    while (pattern / power >= 10) power *= 10;
    for (; power; power /= 10) state = rows[state + regex->byte_to_class['0' + pattern / power % 10]];
    state = rows[state + regex->byte_to_class[2]];
    while (read != value->length) {
        sz_size_t const encoded_length = value->escaped
                                             ? sz_subdocuments_unescape_unit_simt_(bytes, value->length, &read, encoded)
                                             : (encoded[0] = bytes[read++], 1);
        for (sz_size_t index = 0; index != encoded_length; ++index) {
            state = rows[state + regex->byte_to_class[encoded[index]]];
            if (state >= accepts) return sz_true_k;
            if (!state) return sz_false_k;
        }
    }
    return (sz_bool_t)(rows[state + regex->classes_count] >= accepts);
}

/** Runs filter @p filter for the node in slot @p node over the slots `[node, end)` of its subtree,
 *  as @c sz_subdocuments_filter_holds_ does. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_filter_holds_simt_(sz_subdocuments_engine_t const *engine,
                                                                sz_u8_t const *text, sz_u64_t const *stack,
                                                                sz_size_t frame_words, sz_size_t node, sz_size_t end,
                                                                sz_subdocuments_filter_t const *filter, sz_u8_t *buffer,
                                                                sz_size_t capacity) {
    sz_size_t const header_words = sz_subdocuments_frame_header_words_simt_k, words = engine->words_count;
    sz_subdocuments_value_t values[sz_subdocuments_filter_depth_k];
    sz_size_t depth = 0;
    for (sz_size_t index = filter->instructions_begin; index != filter->instructions_end; ++index) {
        sz_subdocuments_instruction_t const *const instruction = &engine->instructions[index];
        sz_subdocuments_value_t *const top = &values[depth];
        sz_u32_t const operation = instruction->operation;
        if (operation == sz_subdocuments_operation_literal_k) {
            top->kind = (sz_subdocuments_kind_t)instruction->operand, top->number = instruction->number;
            top->bytes = engine->strings_data + instruction->string_offset, top->length = instruction->string_length;
            top->escaped = sz_false_k, ++depth;
        }
        else if (operation == sz_subdocuments_operation_exists_k || operation == sz_subdocuments_operation_value_k ||
                 operation == sz_subdocuments_operation_count_k) {
            sz_size_t const step = instruction->operand;
            sz_size_t found = 0, first = 0, slot;
            // A dead slot's subtree was not replayed and holds stale live sets.
            for (slot = node; slot < end; ++slot)
                if (sz_subdocuments_dead_simt_(stack + slot * frame_words + header_words, words))
                    slot = ((sz_subdocuments_frame_t const *)(stack + slot * frame_words))->subtree_end - 1;
                else if (stack[slot * frame_words + header_words + (step >> 6)] >> (step & 63) & 1)
                    first = found++ ? first : slot;
            top->bytes = STRINGZILLA_NULL, top->length = 0, top->escaped = sz_false_k;
            if (operation == sz_subdocuments_operation_value_k) {
                sz_subdocuments_frame_t const *const frame = (sz_subdocuments_frame_t const *)(stack +
                                                                                               first * frame_words);
                if (found == 1) *top = sz_subdocuments_node_value_simt_(text + frame->byte_offset, frame->byte_length);
                else top->kind = sz_subdocuments_kind_nothing_k;
            }
            else if (operation == sz_subdocuments_operation_count_k)
                top->kind = sz_subdocuments_kind_number_k, top->number = (sz_f64_t)found;
            else top->kind = sz_subdocuments_kind_logical_k, top->number = found != 0;
            ++depth;
        }
        else if (operation == sz_subdocuments_operation_length_k)
            values[depth - 1] = sz_subdocuments_length_simt_(&values[depth - 1]);
        else if (operation == sz_subdocuments_operation_match_k || operation == sz_subdocuments_operation_search_k) {
            sz_bool_t const matched = sz_subdocuments_regex_test_simt_(&engine->regex, &values[depth - 1],
                                                                       instruction->operand);
            values[depth - 1].kind = sz_subdocuments_kind_logical_k, values[depth - 1].number = matched;
        }
        else if (operation == sz_subdocuments_operation_compare_k) {
            --depth;
            values[depth - 1].number = sz_subdocuments_compare_simt_(&values[depth - 1], &values[depth],
                                                                     instruction->operand, buffer, capacity);
            values[depth - 1].kind = sz_subdocuments_kind_logical_k;
        }
        else if (operation == sz_subdocuments_operation_not_k) values[depth - 1].number = values[depth - 1].number == 0;
        else if (operation == sz_subdocuments_operation_and_k)
            --depth, values[depth - 1].number = values[depth - 1].number != 0 && values[depth].number != 0;
        else --depth, values[depth - 1].number = values[depth - 1].number != 0 || values[depth].number != 0;
    }
    return (sz_bool_t)(values[0].number != 0);
}

/** Where one level of a replay stands, in the header of its spare frame: the slot it
 *  settles next, one past the last, and the filter it decides there,
 *  @c STRINGZILLA_SIZE_MAX before the slot's first. */
typedef struct sz_subdocuments_level_simt_t {
    sz_size_t slot, end, filter;
} sz_subdocuments_level_simt_t;

/**
 *  @brief Settles the slots `[begin, end)` of a closed pending run as @c sz_subdocuments_replay_
 *      does, each filter's replay of a node's subtree running one level up rather than recursing.
 *  @param[in] spare The first free slot past every kept one; level @c l keeps its place
 *      in the header of slot `spare + l` and its filter results in that slot's live set.
 *      Only level zero reports.
 */
STRINGZILLA_DEVICE void sz_subdocuments_replay_simt_(sz_subdocuments_engine_t const *engine, sz_u8_t const *text,
                                                     sz_u64_t *stack, sz_size_t frame_words, sz_size_t begin,
                                                     sz_size_t end, sz_size_t spare, sz_u8_t *buffer,
                                                     sz_size_t capacity, sz_size_t document, sz_match_t *matches,
                                                     sz_size_t matches_capacity, sz_size_t *emitted) {
    sz_size_t const header_words = sz_subdocuments_frame_header_words_simt_k, words = engine->words_count;
    sz_subdocuments_level_simt_t *state = (sz_subdocuments_level_simt_t *)(stack + spare * frame_words);
    sz_size_t level = 0, word;
    state->slot = begin, state->end = end, state->filter = STRINGZILLA_SIZE_MAX;
    for (;;) {
        sz_u64_t *const results = (sz_u64_t *)state + header_words;
        sz_size_t const slot = state->slot;
        sz_subdocuments_frame_t const *const frame = (sz_subdocuments_frame_t const *)(stack + slot * frame_words);
        sz_subdocuments_frame_t const *parent;
        sz_u64_t const *parent_live;
        sz_u64_t *live;
        if (slot == state->end) {
            sz_subdocuments_filter_t const *filter;
            sz_subdocuments_frame_t const *node;
            sz_u64_t *level_results;
            if (!level) return;
            // A filter's replay ended, so the level below decides that filter at its
            // node and moves on.
            state = (sz_subdocuments_level_simt_t *)(stack + (spare + --level) * frame_words);
            level_results = (sz_u64_t *)state + header_words;
            filter = &engine->filters[state->filter++];
            node = (sz_subdocuments_frame_t const *)(stack + state->slot * frame_words);
            if (sz_subdocuments_filter_holds_simt_(engine, text, stack, frame_words, state->slot, node->subtree_end,
                                                   filter, buffer, capacity))
                level_results[filter->step >> 6] |= (sz_u64_t)1 << (filter->step & 63);
            continue;
        }
        parent = (sz_subdocuments_frame_t const *)(stack + frame->parent * frame_words);
        parent_live = (sz_u64_t const *)parent + header_words;
        live = stack + slot * frame_words + header_words;
        if (state->filter == STRINGZILLA_SIZE_MAX) {
            for (word = 0; word != words; ++word) results[word] = 0;
            state->filter = 0;
        }
        // A filter step advances into the node when the step before it is live and not final.
        for (; state->filter != engine->filters_count; ++state->filter) {
            sz_size_t const step = engine->filters[state->filter].step, before = step - 1;
            if ((parent_live[before >> 6] & ~engine->final_steps[before >> 6]) >> (before & 63) & 1 &&
                !(results[step >> 6] >> (step & 63) & 1))
                break;
        }
        if (state->filter != engine->filters_count) {
            for (word = 0; word != words; ++word) live[word] = engine->filters_roots[state->filter * words + word];
            state = (sz_subdocuments_level_simt_t *)(stack + (spare + ++level) * frame_words);
            state->slot = slot + 1, state->end = frame->subtree_end, state->filter = STRINGZILLA_SIZE_MAX;
            continue;
        }
        if (parent->closer == ']')
            sz_subdocuments_element_live_simt_(engine, parent_live, frame->selector, parent->next_position, live);
        else sz_subdocuments_member_live_simt_(engine, parent_live, frame->selector, live);
        for (word = 0; word != words; ++word) live[word] |= results[word];
        if (!level)
            sz_subdocuments_report_simt_(engine, live, document, frame->byte_offset, frame->byte_length, matches,
                                         matches_capacity, emitted);
        state->slot = sz_subdocuments_dead_simt_(live, words) ? frame->subtree_end : slot + 1;
        state->filter = STRINGZILLA_SIZE_MAX;
    }
}

/**
 *  @brief Walks the next structural bytes of the window a byte to a lane, as
 *      @ref sz_subdocuments_walk_simt_ steps through them one at a time, while every live set fits
 *      a single word.
 *
 *  Each lane takes the byte left whose rank is its own: a sum of opens minus closes over the lanes
 *  gives each byte its depth, and the lowest depth before it tells whether a container opened in
 *  this batch holds it. Every byte none holds is checked against what the byte before it leaves
 *  expected, with keys looked up and live sets settled in their own lanes; one that such a
 *  container holds is skipped with it, as no live step enters it, and that container's close ends
 *  its value. The batch commits every byte before the first it cannot settle alone: a container to
 *  enter, entered last; one whose close lies past the batch, after which the warp skips to that
 *  close; and a key to unescape, a container opening a run or the close of the pending one, left to
 *  the walk's own step. Inside a pending run nothing reports: each value a path selects and each
 *  container entered keeps the next slot in order, as the walk keeps them for the replay, and each
 *  container's slot is found by following the parents up from the innermost one.
 *
 *  @param[inout] from The first byte not walked yet, left in place when nothing was committed.
 *  @param[inout] current The innermost open container's slot, never @c sz_subdocuments_no_slot_k ;
 *      @c sz_subdocuments_no_slot_k once the root closes.
 *  @param[inout] top Inside a pending run, the first slot past the kept ones; read otherwise.
 *  @param[in] run The pending run's container, or @c sz_subdocuments_no_slot_k .
 *  @return The first failure among the committed bytes, else @c sz_success_k.
 */
STRINGZILLA_DEVICE sz_status_t sz_subdocuments_batch_simt_(
    sz_subdocuments_engine_t const *engine, sz_u8_t const *text, sz_size_t length, sz_size_t document, sz_u64_t *stack,
    sz_size_t frames, sz_size_t frame_words, sz_size_t key_bytes, sz_match_t *matches, sz_size_t matches_capacity,
    sz_size_t *emitted, sz_subdocuments_cursor_simt_t *cursor, sz_subdocuments_warp_simt_t const *warp, sz_size_t *from,
    sz_size_t *current, sz_size_t *top, sz_size_t run, sz_subdocuments_expect_t *expect) {
    unsigned const lane = threadIdx.x & 31u;
    sz_u32_t const below = (1u << lane) - 1u, above = lane == 31 ? 0u : ~0u << (lane + 1);
    sz_size_t const header_words = sz_subdocuments_frame_header_words_simt_k;
    sz_u64_t const final_steps = engine->final_steps[0];
    sz_bool_t const pending_run = (sz_bool_t)(run != sz_subdocuments_no_slot_k);
    sz_size_t window, position, end = 0, name = sz_subdocuments_no_name_k, closer = 0;
    sz_u32_t pending, inclusive, total, bits, offset, quotes, openings, elements, closes, since, ends, committed;
    sz_u32_t after, expected, stops, failures, reports, reports_through, keys, kept, chain = (sz_u32_t)*current;
    sz_u64_t owner_live = 0, live = 0, reported, continuing, waiting;
    unsigned source = 0, tokens, step, commit, last;
    int delta, depth_after, depth_before, lowest, lowest_through;
    sz_i64_t slot, kept_slot;
    sz_bool_t valid, quote, opening, walked, is_open, is_close, is_scalar, keyed, is_value, malformed;
    sz_bool_t entering, unfinished, keeping, unescape = sz_false_k, enters, skips;
    sz_subdocuments_frame_t *owner;
    sz_u8_t byte;

    // Each lane takes the structural byte at or past the first unwalked one whose rank is its own:
    // the lane holding it, then its bit.
    for (;; *from = cursor->window_end) {
        if (*from >= cursor->window_end && !sz_subdocuments_advance_simt_(cursor, warp)) return sz_success_k;
        window = cursor->window_end - 1024;
        pending = cursor->lane_pending & sz_subdocuments_from_mask_simt_(window + 32 * lane, *from);
        inclusive = sz_subdocuments_lanes_sum_simt_(warp, (sz_u32_t)__popc(pending));
        total = warp->shuffle(inclusive, 31);
        if (total) break;
    }
    for (step = 16; step; step >>= 1)
        if (warp->shuffle(inclusive, source + step - 1) <= lane) source += step;
    bits = warp->shuffle(pending, source);
    offset = 32 * source +
             sz_subdocuments_select_simt_(bits, lane - (warp->shuffle(inclusive, source) - (sz_u32_t)__popc(bits)));
    position = window + offset;
    tokens = total < 32 ? total : 32;
    byte = lane < tokens ? text[position] : (sz_u8_t)' ';

    // Quotes pair up in order, and a string closing past the batch waits for the next one.
    quotes = warp->ballot(lane < tokens && byte == '"');
    quote = (sz_bool_t)(quotes >> lane & 1u);
    opening = (sz_bool_t)(quote && !(__popc(quotes & below) & 1));
    openings = warp->ballot(opening);
    if (openings >> (tokens - 1) & 1u && (total > 32 || cursor->window_end < length)) --tokens;
    if (!tokens) return sz_success_k;
    valid = (sz_bool_t)(lane < tokens);

    // Depths relative to the innermost open container, and how many of this batch's containers hold
    // a byte, which leaves it to be walked only when none does.
    delta = valid ? ((byte | 0x20) == '{') - ((byte | 0x20) == '}') : 0;
    depth_after = (int)sz_subdocuments_lanes_sum_simt_(warp, (sz_u32_t)delta);
    depth_before = depth_after - delta;
    lowest_through = sz_subdocuments_lanes_min_simt_(warp, depth_after);
    lowest = (int)warp->shuffle_up((sz_u32_t)lowest_through, 1);
    lowest = lane && lowest < 0 ? lowest : 0;
    walked = (sz_bool_t)(valid && depth_before == lowest);
    is_open = (sz_bool_t)(delta > 0), is_close = (sz_bool_t)(delta < 0);
    is_scalar = (sz_bool_t)(valid && !quote && !delta && byte != ':' && byte != ',');
    // Outside a run every open container sits a slot past its parent; inside one, each lane follows
    // the parents up as many levels as its index.
    if (pending_run)
        for (int level = 1, levels = -(int)warp->shuffle((sz_u32_t)lowest_through, tokens - 1); level <= levels;
             ++level)
            if (lane >= (unsigned)level && chain != 0xFFFFFFFFu)
                chain = (sz_u32_t)((sz_subdocuments_frame_t const *)(stack + (sz_size_t)chain * frame_words))->parent;
    slot = pending_run ? (sz_i64_t)(int)warp->shuffle(chain, (unsigned)-depth_before & 31u)
                       : (sz_i64_t)*current + depth_before;
    owner = (sz_subdocuments_frame_t *)(stack + (sz_size_t)(walked && slot >= 0 ? slot : 0) * frame_words);
    if (walked && slot >= 0) {
        sz_u64_t const *const owner_words = (sz_u64_t const *)owner;
        closer = owner->closer, owner_live = owner_words[header_words];
    }

    // What each byte leaves expected, a closing quote's by whether its string was a key.
    after = sz_subdocuments_expect_comma_or_close_k;
    if (byte == ':') after = sz_subdocuments_expect_value_k;
    else if (byte == ',') after = closer == '}' ? sz_subdocuments_expect_key_k : sz_subdocuments_expect_value_k;
    else if (walked && is_close) after = pending_run || slot > 0 ? after : sz_subdocuments_expect_end_k;
    else if (quote && !opening) after = sz_subdocuments_expect_colon_k;
    expected = warp->shuffle_up(after, 1);
    expected = lane ? expected : (sz_u32_t)*expect;
    keyed = (sz_bool_t)(opening && (expected == sz_subdocuments_expect_key_k ||
                                    expected == sz_subdocuments_expect_key_or_close_k));
    if (warp->shuffle_up(keyed, 1) && quote && !opening) after = sz_subdocuments_expect_colon_k;
    else if (quote && !opening) after = sz_subdocuments_expect_comma_or_close_k;
    expected = warp->shuffle_up(after, 1);
    expected = lane ? expected : (sz_u32_t)*expect;

    // Each byte checked as the walk's step for it checks it.
    is_value = (sz_bool_t)(walked && (is_open || is_scalar || (opening && !keyed)));
    malformed = (sz_bool_t)(slot < 0);
    if (byte == ':') malformed = (sz_bool_t)(malformed || expected != sz_subdocuments_expect_colon_k);
    else if (byte == ',') malformed = (sz_bool_t)(malformed || expected != sz_subdocuments_expect_comma_or_close_k);
    else if (is_close)
        malformed = (sz_bool_t)(malformed || closer != byte ||
                                (expected != sz_subdocuments_expect_comma_or_close_k &&
                                 expected != (byte == '}' ? sz_subdocuments_expect_key_or_close_k
                                                          : sz_subdocuments_expect_value_or_close_k)));
    else if (is_value)
        malformed = (sz_bool_t)(malformed || (expected != sz_subdocuments_expect_value_k &&
                                              expected != sz_subdocuments_expect_value_or_close_k));
    // A string still open at the document's end has no closing quote.
    if (opening && lane + 1 == tokens && !(total > 32 || cursor->window_end < length)) malformed = sz_true_k;
    malformed = (sz_bool_t)(walked && malformed);

    // Keys look up their names, and live sets settle in their own lanes.
    {
        sz_u32_t const closing = warp->shuffle(offset, (lane + 1) & 31u);
        sz_u64_t carry = 0;
        if (opening) end = window + closing + 1;
        if (walked && keyed && !malformed) {
            sz_u8_t const *const name_bytes = text + position + 1;
            sz_size_t const name_length = window + closing - position - 1;
            if (engine->names_slots_count && ((owner_live & ~final_steps) << 1 & ~engine->wildcard_steps[0]) != 0) {
                name = sz_subdocuments_key_name_simt_(engine, name_bytes, name_length, key_bytes);
                if (name == STRINGZILLA_SIZE_MAX) unescape = sz_true_k, name = sz_subdocuments_no_name_k;
            }
            live = sz_subdocuments_child_word_simt_(engine, &owner_live, (sz_u64_t const *)STRINGZILLA_NULL, 0, &carry,
                                                    sz_false_k, name, 0);
        }
    }
    elements = warp->ballot(is_value && closer == ']');
    closes = warp->ballot(walked && is_close);
    since = closes & below ? ~0u << (31 - __clz((int)(closes & below))) << 1 : ~0u;
    if (is_value && closer == ']') {
        sz_u64_t carry = 0;
        name = owner->next_position + (sz_size_t)__popc(elements & below & since);
        live = sz_subdocuments_child_word_simt_(engine, &owner_live, (sz_u64_t const *)STRINGZILLA_NULL, 0, &carry,
                                                sz_true_k, name, 0);
    }
    if (engine->filters_count && walked && (keyed || (is_value && closer == ']'))) {
        sz_u64_t const filtered = (owner_live & ~final_steps) << 1 & engine->filter_steps[0];
        live |= filtered;
        for (sz_size_t filter = 0; filtered && filter != engine->filters_count; ++filter)
            if (live >> engine->filters[filter].step & 1) live |= engine->filters_roots[filter];
    }
    // A member's value takes the live set its key settled three lanes before, or, for a key before
    // the batch, the one it left in the slot past the last kept one.
    {
        unsigned const key_lane = (lane - 3) & 31u;
        sz_u64_t const key_live = warp->shuffle((sz_u32_t)live, key_lane) |
                                  (sz_u64_t)warp->shuffle((sz_u32_t)(live >> 32), key_lane) << 32;
        sz_size_t const key_name = warp->shuffle((sz_u32_t)name, key_lane);
        if (is_value && closer == '}' && lane >= 3) live = key_live, name = key_name;
        else if (is_value && closer == '}') {
            sz_u64_t const *const held = stack + *top * frame_words;
            live = held[header_words], name = ((sz_subdocuments_frame_t const *)held)->selector;
        }
    }

    // Values report, or inside a run keep a slot each in order, and containers are entered,
    // skipped to their close, or left to the walk.
    reported = is_value ? live & final_steps : 0;
    continuing = is_value ? live & ~final_steps : 0;
    entering = (sz_bool_t)(is_open && continuing);
    waiting = entering && !pending_run
                  ? continuing << 1 & ((byte == '[' ? engine->relative_steps[0] : 0) | engine->filter_steps[0])
                  : 0;
    keeping = (sz_bool_t)(pending_run ? entering || reported : entering);
    kept = warp->ballot(keeping);
    kept_slot = pending_run ? (sz_i64_t)(*top + (sz_size_t)__popc(kept & below)) : slot + 1;
    ends = warp->ballot(valid && is_close && depth_before == lowest + 1);
    {
        sz_u32_t const ends_after = ends & above;
        sz_u32_t const closing = warp->shuffle(offset, ends_after ? (unsigned)__ffs((int)ends_after) - 1 : lane);
        unfinished = (sz_bool_t)(is_value && is_open && !continuing && !ends_after);
        if (is_value && is_open && !continuing) end = window + closing + 1;
    }
    if (is_value && is_scalar) {
        end = position + sz_subdocuments_scalar_length_simt_(text, position, length);
        malformed = (sz_bool_t)(malformed || end == position);
    }
    reports = pending_run ? 0 : (sz_u32_t)__popcll(reported);
    reports_through = sz_subdocuments_lanes_sum_simt_(warp, reports);

    // The batch commits up to its first stop; the first failure before it fails the walk.
    commit = tokens;
    // In a run, its own close replays it, and no batch closes more containers than a lane can name.
    stops = warp->ballot(
        walked && (unescape || waiting || (pending_run && ((is_close && slot == (sz_i64_t)run) || depth_after < -31))));
    if (stops) commit = sz_min_of_two(commit, (unsigned)__ffs((int)stops) - 1);
    stops = warp->ballot(walked && ((entering && !waiting) || unfinished));
    if (stops) commit = sz_min_of_two(commit, (unsigned)__ffs((int)stops));
    if (!commit) return sz_success_k;
    committed = commit == 32 ? ~0u : (1u << commit) - 1u;
    failures = warp->ballot(malformed || (keeping && kept_slot + 2 > (sz_i64_t)frames)) & committed;
    if (failures)
        return warp->shuffle(malformed, (unsigned)__ffs((int)failures) - 1) ? sz_invalid_syntax_k : sz_bad_alloc_k;

    // Matches go in place, and the containers open before the batch that it closes settle theirs.
    if (committed >> lane & 1u) {
        sz_size_t index = *emitted + reports_through - reports;
        for (sz_u64_t left = pending_run ? 0 : reported; left; left &= left - 1, ++index)
            if (index < matches_capacity) {
                sz_match_t *const match = &matches[index];
                match->haystack_index = document;
                match->needle_index = engine->steps_paths[sz_subdocuments_ctz_simt_(left)];
                match->byte_offset = position;
                if (!entering && !unfinished) match->byte_length = end - position;
            }
        if (walked && is_close && !pending_run)
            sz_subdocuments_close_frame_simt_(matches, matches_capacity, owner, position + 1);
    }

    // Every lane writes the same frames, so each reads its own writes back.
    last = commit - 1;
    enters = (sz_bool_t)warp->shuffle(entering, last), skips = (sz_bool_t)warp->shuffle(unfinished, last);
    {
        int const level = (int)warp->shuffle((sz_u32_t)(enters || skips ? depth_before : depth_after), last);
        sz_i64_t const open = pending_run ? (sz_i64_t)(int)warp->shuffle(chain, (unsigned)-level & 31u)
                                          : (sz_i64_t)*current + level;
        sz_size_t const last_position = window + warp->shuffle(offset, last);
        sz_size_t const last_reports = warp->shuffle(reports, last);
        sz_size_t const reports_committed = warp->shuffle(reports_through, last);
        sz_u32_t const closes_committed = closes & committed, kept_committed = kept & committed;
        sz_u32_t const tail = closes_committed ? ~0u << (31 - __clz((int)closes_committed)) << 1 : ~0u;
        sz_u32_t const appended = (sz_u32_t)__popc(elements & committed & tail);
        sz_u32_t const last_after = warp->shuffle(after, last);
        sz_u8_t const last_byte = (sz_u8_t)warp->shuffle(byte, last);
        sz_u32_t left;
        // In a run, the containers the batch closes keep spans and element counts for the replay.
        for (left = pending_run ? closes_committed : 0; left; left &= left - 1) {
            unsigned const closing = (unsigned)__ffs((int)left) - 1;
            sz_u32_t const before_closing = closes_committed & ((1u << closing) - 1u);
            sz_u32_t const from_lane = before_closing ? ~0u << (31 - __clz((int)before_closing)) << 1 : ~0u;
            sz_subdocuments_frame_t *const frame =
                (sz_subdocuments_frame_t *)(stack + warp->shuffle((sz_u32_t)slot, closing) * frame_words);
            frame->next_position += (sz_size_t)__popc(elements & from_lane & ((1u << closing) - 1u));
            frame->byte_length = window + warp->shuffle(offset, closing) + 1 - frame->byte_offset;
        }
        // Inside a run, every value a path selects keeps a frame for the replay.
        for (left = pending_run ? kept_committed & ~warp->ballot(entering) & ~warp->ballot(unfinished) : 0; left;
             left &= left - 1) {
            unsigned const keeper = (unsigned)__ffs((int)left) - 1;
            sz_u64_t *const slot_words = stack + (sz_size_t)(*top + (sz_size_t)__popc(kept & ((1u << keeper) - 1u))) *
                                                     frame_words;
            sz_subdocuments_frame_t *const frame = (sz_subdocuments_frame_t *)slot_words;
            sz_size_t const begin = window + warp->shuffle(offset, keeper);
            frame->parent = warp->shuffle((sz_u32_t)slot, keeper), frame->byte_offset = begin;
            frame->byte_length = warp->shuffle((sz_u32_t)(end - position), keeper) |
                                 (sz_size_t)warp->shuffle((sz_u32_t)((end - position) >> 32), keeper) << 32;
            frame->closer = 0;
            frame->selector = warp->shuffle((sz_u32_t)name, keeper) |
                              (sz_size_t)warp->shuffle((sz_u32_t)((sz_u64_t)name >> 32), keeper) << 32;
            slot_words[header_words] = warp->shuffle((sz_u32_t)live, keeper) |
                                       (sz_u64_t)warp->shuffle((sz_u32_t)(live >> 32), keeper) << 32;
        }
        if (appended && open >= 0) ((sz_subdocuments_frame_t *)(stack + open * frame_words))->next_position += appended;
        if (pending_run) *top += (sz_size_t)__popc(kept_committed);
        // A key whose value lies past the batch leaves its live set where that value will read it.
        keys = warp->ballot(walked && keyed && lane < commit && lane + 3 >= commit);
        if (keys) {
            unsigned const key_lane = (unsigned)__ffs((int)keys) - 1;
            sz_u64_t *const slot_words =
                stack +
                (pending_run ? *top : (sz_size_t)(int)warp->shuffle((sz_u32_t)slot, key_lane) + 1) * frame_words;
            slot_words[header_words] = warp->shuffle((sz_u32_t)live, key_lane) |
                                       (sz_u64_t)warp->shuffle((sz_u32_t)(live >> 32), key_lane) << 32;
            ((sz_subdocuments_frame_t *)slot_words)->selector = warp->shuffle((sz_u32_t)name, key_lane);
        }
        if (enters) {
            sz_size_t const pushed = (sz_size_t)(int)warp->shuffle((sz_u32_t)kept_slot, last);
            sz_u64_t *const slot_words = stack + pushed * frame_words;
            sz_subdocuments_frame_t *const frame = (sz_subdocuments_frame_t *)slot_words;
            frame->parent = (sz_size_t)open, frame->byte_offset = last_position, frame->next_position = 0;
            frame->matches_begin = *emitted + reports_committed - last_reports;
            frame->matches_end = *emitted + reports_committed;
            frame->closer = last_byte == '{' ? '}' : ']';
            frame->selector = warp->shuffle((sz_u32_t)name, last) |
                              (sz_size_t)warp->shuffle((sz_u32_t)((sz_u64_t)name >> 32), last) << 32;
            slot_words[header_words] = warp->shuffle((sz_u32_t)live, last) |
                                       (sz_u64_t)warp->shuffle((sz_u32_t)(live >> 32), last) << 32;
            *current = pushed;
            *expect = last_byte == '{' ? sz_subdocuments_expect_key_or_close_k
                                       : sz_subdocuments_expect_value_or_close_k;
        }
        else if (skips) *current = (sz_size_t)open, *expect = sz_subdocuments_expect_comma_or_close_k;
        else
            *current = open >= 0 ? (sz_size_t)open : sz_subdocuments_no_slot_k,
            *expect = (sz_subdocuments_expect_t)last_after;
        *emitted += reports_committed;
        *from = last_position + 1;
    }
    if (skips) {
        sz_size_t const closing = sz_subdocuments_skip_simt_(cursor, *from, warp);
        if (closing == STRINGZILLA_SIZE_MAX) return sz_invalid_syntax_k;
        if (lane == last)
            for (sz_size_t index = *emitted - reports, ends_at = sz_min_of_two(*emitted, matches_capacity);
                 index < ends_at; ++index)
                matches[index].byte_length = closing + 1 - position;
        // A container a run keeps, skipped past the batch, takes its frame once its close is known.
        if (pending_run && kept >> last & 1u) {
            sz_u64_t *const slot_words = stack + (*top - 1) * frame_words;
            sz_subdocuments_frame_t *const frame = (sz_subdocuments_frame_t *)slot_words;
            frame->parent = *current, frame->byte_offset = *from - 1;
            frame->byte_length = closing + 1 - frame->byte_offset;
            frame->closer = 0;
            frame->selector = warp->shuffle((sz_u32_t)name, last) |
                              (sz_size_t)warp->shuffle((sz_u32_t)((sz_u64_t)name >> 32), last) << 32;
            slot_words[header_words] = warp->shuffle((sz_u32_t)live, last) |
                                       (sz_u64_t)warp->shuffle((sz_u32_t)(live >> 32), last) << 32;
        }
        *from = closing + 1;
    }
    return sz_success_k;
}

/**
 *  @brief Walks one JSON document as @c sz_subdocuments_walk_ does, emitting its matches from
 *      @p emitted on, those below @p matches_capacity written, every lane of a warp at once.
 *
 *  While live sets take one word, @ref sz_subdocuments_batch_simt_ walks the bytes a lane each; the
 *  rest, and whatever a batch leaves, take one step at a time on every lane alike, finding bytes
 *  and skipping containers a window at a time.
 *
 *  @param[in] stack Room for @p frames frames of @p frame_words words each.
 *  @param[in] key Room for @p key_bytes bytes, where keys and compared strings are unescaped.
 *  @return @c sz_success_k, @c sz_invalid_syntax_k for a malformed document, or
 *      @c sz_bad_alloc_k when it needs more than @p frames frames, as the serial walk does when
 *      its stack cannot grow.
 */
STRINGZILLA_DEVICE sz_status_t sz_subdocuments_walk_simt_(sz_subdocuments_engine_t const *engine, sz_u8_t const *text,
                                                          sz_size_t length, sz_size_t document, sz_u64_t *stack,
                                                          sz_size_t frames, sz_size_t frame_words, sz_u8_t *key,
                                                          sz_size_t key_bytes, sz_match_t *matches,
                                                          sz_size_t matches_capacity, sz_size_t *emitted,
                                                          sz_subdocuments_warp_simt_t const *warp) {
    sz_u64_t const *const final_steps = engine->final_steps;
    sz_size_t const words = engine->words_count, header_words = sz_subdocuments_frame_header_words_simt_k;
    sz_subdocuments_expect_t expect = sz_subdocuments_expect_value_k;
    sz_size_t top = 0, current = sz_subdocuments_no_slot_k, run = sz_subdocuments_no_slot_k, from = 0;
    sz_size_t position, closing, word;
    sz_subdocuments_cursor_simt_t cursor;
    for (word = 0; word != words; ++word) stack[header_words + word] = engine->root_steps[word];
    cursor.text = text, cursor.length = length, cursor.escaped = 0, cursor.in_string = 0, cursor.scalar = 0;
    cursor.slashes = 0, cursor.window_end = 0;

    for (;;) {
        if (words == 1 && current != sz_subdocuments_no_slot_k) {
            sz_size_t const walked = from;
            sz_status_t const status = sz_subdocuments_batch_simt_(
                engine, text, length, document, stack, frames, frame_words, key_bytes, matches, matches_capacity,
                emitted, &cursor, warp, &from, &current, &top, run, &expect);
            if (status != sz_success_k) return status;
            if (run == sz_subdocuments_no_slot_k) top = current != sz_subdocuments_no_slot_k ? current + 1 : 0;
            if (from != walked) continue;
        }
        if ((position = sz_subdocuments_next_simt_(&cursor, from, warp)) == STRINGZILLA_SIZE_MAX) break;
        from = position + 1;
        sz_u8_t const byte = text[position];
        sz_subdocuments_frame_t *const parent = current != sz_subdocuments_no_slot_k
                                                    ? (sz_subdocuments_frame_t *)(stack + current * frame_words)
                                                    : (sz_subdocuments_frame_t *)STRINGZILLA_NULL;
        sz_u64_t const *const parent_live = parent ? (sz_u64_t const *)parent + header_words
                                                   : (sz_u64_t const *)STRINGZILLA_NULL;
        sz_subdocuments_frame_t *const frame = (sz_subdocuments_frame_t *)(stack + top * frame_words);
        sz_u64_t *const live = (sz_u64_t *)frame + header_words;
        sz_u64_t reported = 0, continuing = 0, carried;
        sz_size_t matches_begin, end;

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
            closing = parent->parent;
            if (run == sz_subdocuments_no_slot_k || run == current) {
                sz_subdocuments_close_frame_simt_(matches, matches_capacity, parent, position + 1);
                if (run == current) {
                    sz_size_t slot;
                    // Filters replay at one spare slot per nesting level past the kept ones.
                    if (top + engine->filters_count + 2 > frames) return sz_bad_alloc_k;
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
                    sz_subdocuments_replay_simt_(engine, text, stack, frame_words, run + 1, top, top, key, key_bytes,
                                                 document, matches, matches_capacity, emitted);
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
            sz_u8_t const *const name_bytes = text + position + 1;
            sz_size_t name_length, name = sz_subdocuments_no_name_k;
            if ((closing = sz_subdocuments_next_simt_(&cursor, from, warp)) == STRINGZILLA_SIZE_MAX)
                return sz_invalid_syntax_k;
            from = closing + 1;
            name_length = closing - position - 1;
            if (engine->names_slots_count &&
                sz_subdocuments_needs_name_simt_(engine, parent_live, (sz_u64_t const *)STRINGZILLA_NULL) &&
                (name = sz_subdocuments_key_name_simt_(engine, name_bytes, name_length, key_bytes)) ==
                    STRINGZILLA_SIZE_MAX) {
                name_length = sz_subdocuments_unescape_simt_(name_bytes, name_length, key, key_bytes);
                if (name_length == STRINGZILLA_SIZE_MAX) return sz_invalid_syntax_k;
                name = name_length <= key_bytes ? sz_subdocuments_name_simt_(engine, key, name_length)
                                                : sz_subdocuments_no_name_k;
            }
            sz_subdocuments_member_live_simt_(engine, parent_live, name, live);
            if (engine->filters_count) sz_subdocuments_filters_tentative_simt_(engine, parent_live, live);
            frame->selector = name;
            expect = sz_subdocuments_expect_colon_k;
            continue;
        }

        if (expect != sz_subdocuments_expect_value_k && expect != sz_subdocuments_expect_value_or_close_k)
            return sz_invalid_syntax_k;
        if (parent && parent->closer == ']') {
            frame->selector = parent->next_position++;
            sz_subdocuments_element_live_simt_(engine, parent_live, frame->selector, 0, live);
            if (engine->filters_count) sz_subdocuments_filters_tentative_simt_(engine, parent_live, live);
        }
        for (word = 0; word != words; ++word)
            reported |= live[word] & final_steps[word], continuing |= live[word] & ~final_steps[word];
        matches_begin = *emitted;
        if (run == sz_subdocuments_no_slot_k)
            sz_subdocuments_report_simt_(engine, live, document, position, 0, matches, matches_capacity, emitted);
        if ((byte == '{' || byte == '[') && continuing) {
            frame->parent = current, frame->byte_offset = position;
            frame->next_position = 0, frame->matches_begin = matches_begin, frame->matches_end = *emitted;
            frame->closer = byte == '{' ? '}' : ']';
            // A container opens a pending run when a step that waits may advance
            // into what it holds.
            if (run == sz_subdocuments_no_slot_k) {
                sz_u64_t waiting = 0;
                for (word = 0, carried = 0; word != words; ++word) {
                    sz_u64_t const advancing = live[word] & ~final_steps[word];
                    waiting |= (advancing << 1 | carried) &
                               ((byte == '[' ? engine->relative_steps[word] : 0) | engine->filter_steps[word]);
                    carried = advancing >> 63;
                }
                if (waiting) run = top;
            }
            current = top++;
            // The next value's live set takes the slot past the last, so that one must fit as well.
            if (top + 1 > frames) return sz_bad_alloc_k;
            expect = byte == '{' ? sz_subdocuments_expect_key_or_close_k : sz_subdocuments_expect_value_or_close_k;
            continue;
        }
        if (byte == '{' || byte == '[' || byte == '"') {
            closing = byte == '"' ? sz_subdocuments_next_simt_(&cursor, from, warp)
                                  : sz_subdocuments_skip_simt_(&cursor, from, warp);
            if (closing == STRINGZILLA_SIZE_MAX) return sz_invalid_syntax_k;
            end = from = closing + 1;
        }
        else {
            end = position + sz_subdocuments_scalar_length_simt_(text, position, length);
            if (end == position) return sz_invalid_syntax_k;
        }
        if (run == sz_subdocuments_no_slot_k)
            sz_subdocuments_close_simt_(matches, matches_capacity, matches_begin, *emitted, end);
        else if (reported) {
            frame->parent = current, frame->byte_offset = position, frame->byte_length = end - position;
            frame->closer = 0;
            if (++top + 1 > frames) return sz_bad_alloc_k;
        }
        expect = current != sz_subdocuments_no_slot_k ? sz_subdocuments_expect_comma_or_close_k
                                                      : sz_subdocuments_expect_end_k;
    }
    return current != sz_subdocuments_no_slot_k || expect != sz_subdocuments_expect_end_k || cursor.slashes
               ? sz_invalid_syntax_k
               : sz_success_k;
}

/** The first @p byte at or after @p from, or @p from past the last, read four bytes at a time. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_markup_find_simt_(sz_u8_t const *text, sz_size_t length, sz_size_t from,
                                                               sz_u8_t byte) {
    sz_u32_t const repeated = byte * 0x01010101u;
    if (from >= length) return from;
    do {
        sz_u8_t const *const at = text + from;
        unsigned const misalignment = (unsigned)((sz_size_t)at & 3);
        // An aligned word holding a byte of the text lies in a page the device may read whole.
        sz_u32_t const word = __ldg((sz_u32_t const *)(at - misalignment)) >> (8 * misalignment);
        sz_u32_t const found = sz_subdocuments_bytes_equal_simt_(word, repeated) & (0x80808080u >> (8 * misalignment));
        if (found) return sz_min_of_two(from + (sz_size_t)((__ffs((int)found) - 1) >> 3), length);
        from += 4 - misalignment;
    } while (from < length);
    return length;
}

/** The `>` ending a comment, CDATA section or processing instruction whose body starts at @p from,
 *  as @c sz_subdocuments_markup_end_ finds it, or @p length. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_markup_end_simt_(sz_u8_t const *text, sz_size_t length, sz_size_t from,
                                                              sz_u8_t terminator, sz_size_t terminator_length) {
    for (sz_size_t candidate = from;; ++candidate) {
        candidate = sz_subdocuments_markup_find_simt_(text, length, candidate, '>');
        if (candidate >= length) return length;
        if (candidate >= from + terminator_length && text[candidate - 1] == terminator &&
            text[candidate - terminator_length] == terminator)
            return candidate;
    }
}

/** The first byte at or after @p position that is not XML blank space. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_markup_skip_blanks_simt_(sz_u8_t const *text, sz_size_t length,
                                                                      sz_size_t position) {
    while (position != length &&
           (text[position] == ' ' || text[position] == '\t' || text[position] == '\n' || text[position] == '\r'))
        ++position;
    return position;
}

/** The first byte at or after @p position that cannot continue an XML name, as
 *  @c sz_subdocuments_is_xml_name_ tells them apart. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_markup_skip_name_simt_(sz_u8_t const *text, sz_size_t length,
                                                                    sz_size_t position) {
    for (; position != length; ++position) {
        sz_u8_t const byte = text[position], lowered = (sz_u8_t)(byte | 0x20);
        if (!((lowered >= 'a' && lowered <= 'z') || (byte >= '0' && byte <= ':') || byte == '_' || byte == '-' ||
              byte == '.' || byte >= 0x80))
            break;
    }
    return position;
}

/** The live set of an XML node from its parent element's, as
 *  @c sz_subdocuments_node_live_ settles it. */
STRINGZILLA_DEVICE void sz_subdocuments_node_live_simt_(sz_subdocuments_engine_t const *engine,
                                                        sz_u64_t const *parent_live, sz_u32_t *counters,
                                                        sz_subdocuments_node_t kind, sz_size_t name, sz_u64_t *live) {
    sz_u64_t carried = 0;
    for (sz_size_t word = 0; word != engine->words_count; ++word) {
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
            sz_size_t const step = word * 64 + sz_subdocuments_ctz_simt_(counted);
            sz_u32_t const slice = engine->slices_offsets[step];
            if (slice == engine->slices_offsets[step + 1]) continue;
            if (!sz_subdocuments_slice_selects_simt_(&engine->slices[slice], counters[slice],
                                                     (sz_i64_t)counters[slice] + 1))
                tested &= ~(counted & (0 - counted));
            ++counters[slice];
        }
        live[word] = tested |
                     (kind == sz_subdocuments_node_element_k ? parent_live[word] & engine->descendant_steps[word] : 0);
    }
}

/** The name index of the @p name_length bytes at @p name, or @c sz_subdocuments_no_name_k, for a
 *  walk that needs it, as @c sz_subdocuments_markup_name_ looks it up. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_markup_name_simt_(sz_subdocuments_engine_t const *engine,
                                                               sz_u64_t const *parent_live, sz_u8_t const *name,
                                                               sz_size_t name_length, sz_size_t key_bytes) {
    if (!engine->names_slots_count || name_length > key_bytes ||
        !sz_subdocuments_needs_name_simt_(engine, parent_live, (sz_u64_t const *)STRINGZILLA_NULL))
        return sz_subdocuments_no_name_k;
    return sz_subdocuments_name_simt_(engine, name, name_length);
}

/**
 *  @brief Walks one XML document as @c sz_subdocuments_walk_markup_ does, emitting its matches from
 *      @p emitted on, those below @p matches_capacity written.
 *  @return @c sz_success_k, @c sz_invalid_syntax_k for a malformed document, or @c sz_bad_alloc_k
 *      when it needs more than @p frames frames.
 */
STRINGZILLA_DEVICE sz_status_t sz_subdocuments_walk_markup_simt_(sz_subdocuments_engine_t const *engine,
                                                                 sz_u8_t const *text, sz_size_t length,
                                                                 sz_size_t document, sz_u64_t *stack, sz_size_t frames,
                                                                 sz_size_t frame_words, sz_size_t key_bytes,
                                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                                 sz_size_t *emitted) {
    sz_u64_t const *const final_steps = engine->final_steps;
    sz_size_t const words = engine->words_count, header_words = sz_subdocuments_frame_header_words_simt_k;
    sz_size_t const counter_words = frame_words - header_words - words;
    sz_size_t top = 1, current = 0, position = 0, text_begin = 0, skipped_depth = 0, skipped_begin = 0;
    sz_size_t skipped_end = 0, skipped_opening = 0, opening, tag_end, word;
    sz_bool_t rooted = sz_false_k;
    sz_subdocuments_frame_t *frame = (sz_subdocuments_frame_t *)stack;
    sz_u64_t *live = stack + header_words;
    frame->parent = sz_subdocuments_no_slot_k;
    for (word = 0; word != words; ++word) live[word] = engine->root_steps[word];
    for (word = 0; word != counter_words; ++word) live[words + word] = 0;

    for (;;) {
        sz_subdocuments_frame_t *const parent = (sz_subdocuments_frame_t *)(stack + current * frame_words);
        sz_u64_t const *const parent_live = (sz_u64_t const *)parent + header_words;
        sz_u32_t *const parent_counters = (sz_u32_t *)(parent_live + words);
        sz_u8_t kind;
        opening = sz_subdocuments_markup_find_simt_(text, length, position, '<');
        // Outside the root element, character data may only be blank space.
        if (current == 0 && skipped_depth == 0 &&
            sz_subdocuments_markup_skip_blanks_simt_(text, opening, position) != opening)
            return sz_invalid_syntax_k;
        if (opening == length) break;
        if (length - opening < 2) return sz_invalid_syntax_k;
        kind = text[opening + 1];

        // A CDATA section is character data, so the text node around it goes on.
        if (kind == '!' && length - opening >= 9 && text[opening + 2] == '[' && text[opening + 3] == 'C' &&
            text[opening + 4] == 'D' && text[opening + 5] == 'A' && text[opening + 6] == 'T' &&
            text[opening + 7] == 'A' && text[opening + 8] == '[') {
            if (current == 0 && skipped_depth == 0) return sz_invalid_syntax_k;
            tag_end = sz_subdocuments_markup_end_simt_(text, length, opening + 9, ']', 2);
            if (tag_end == length) return sz_invalid_syntax_k;
            position = tag_end + 1;
            continue;
        }
        // Every other markup ends a text node, which is any character data or CDATA between markup.
        if (current != 0 && skipped_depth == 0 && opening != text_begin) {
            frame = (sz_subdocuments_frame_t *)(stack + top * frame_words), live = (sz_u64_t *)frame + header_words;
            sz_subdocuments_node_live_simt_(engine, parent_live, parent_counters, sz_subdocuments_node_text_k,
                                            sz_subdocuments_no_name_k, live);
            sz_subdocuments_report_simt_(engine, live, document, text_begin, opening - text_begin, matches,
                                         matches_capacity, emitted);
        }

        if (kind == '!' && length - opening >= 4 && text[opening + 2] == '-' && text[opening + 3] == '-') {
            tag_end = sz_subdocuments_markup_end_simt_(text, length, opening + 4, '-', 2);
            if (tag_end == length) return sz_invalid_syntax_k;
        }
        else if (kind == '?') {
            tag_end = sz_subdocuments_markup_end_simt_(text, length, opening + 2, '?', 1);
            if (tag_end == length) return sz_invalid_syntax_k;
        }
        else if (kind == '!') {
            // An internal subset could declare entities that no walk expands.
            sz_size_t const name_end = sz_subdocuments_markup_skip_name_simt_(text, length, opening + 2);
            sz_size_t index;
            tag_end = sz_subdocuments_markup_find_simt_(text, length, name_end, '>');
            if (current != 0 || rooted || name_end - opening != 9 || text[opening + 2] != 'D' ||
                text[opening + 3] != 'O' || text[opening + 4] != 'C' || text[opening + 5] != 'T' ||
                text[opening + 6] != 'Y' || text[opening + 7] != 'P' || text[opening + 8] != 'E' || tag_end == length)
                return sz_invalid_syntax_k;
            for (index = name_end; index != tag_end; ++index)
                if (text[index] == '[') return sz_invalid_syntax_k;
        }
        else if (kind == '/') {
            sz_size_t const name_end = sz_subdocuments_markup_skip_name_simt_(text, length, opening + 2);
            sz_size_t const name_length = name_end - opening - 2;
            sz_size_t open_name, index;
            tag_end = sz_subdocuments_markup_skip_blanks_simt_(text, length, name_end);
            if (tag_end == length || text[tag_end] != '>' || name_length == 0) return sz_invalid_syntax_k;
            // Inside a skipped element only nesting counts; the element's own end
            // tag repeats its name.
            if (skipped_depth > 1) --skipped_depth;
            else {
                if (skipped_depth == 0 && current == 0) return sz_invalid_syntax_k;
                open_name = (skipped_depth ? skipped_opening : parent->byte_offset) + 1;
                if (sz_subdocuments_markup_skip_name_simt_(text, length, open_name) - open_name != name_length)
                    return sz_invalid_syntax_k;
                for (index = 0; index != name_length; ++index)
                    if (text[open_name + index] != text[opening + 2 + index]) return sz_invalid_syntax_k;
                if (skipped_depth)
                    sz_subdocuments_close_simt_(matches, matches_capacity, skipped_begin, skipped_end, tag_end + 1),
                        skipped_depth = 0;
                else {
                    sz_subdocuments_close_simt_(matches, matches_capacity, parent->matches_begin, parent->matches_end,
                                                tag_end + 1);
                    top = current, current = parent->parent;
                }
            }
        }
        else {
            sz_size_t const name_end = sz_subdocuments_markup_skip_name_simt_(text, length, opening + 1);
            sz_size_t matches_begin = *emitted, attribute = name_end;
            sz_u64_t continuing = 0;
            sz_u32_t *counters = (sz_u32_t *)STRINGZILLA_NULL;
            sz_bool_t empty;
            if (name_end == opening + 1) return sz_invalid_syntax_k;
            if (skipped_depth == 0) {
                if (current == 0 && rooted) return sz_invalid_syntax_k;
                rooted = sz_true_k;
                frame = (sz_subdocuments_frame_t *)(stack + top * frame_words), live = (sz_u64_t *)frame + header_words;
                counters = (sz_u32_t *)(live + words);
                sz_subdocuments_node_live_simt_(
                    engine, parent_live, parent_counters, sz_subdocuments_node_element_k,
                    sz_subdocuments_markup_name_simt_(engine, parent_live, text + opening + 1, name_end - opening - 1,
                                                      key_bytes),
                    live);
                sz_subdocuments_report_simt_(engine, live, document, opening, 0, matches, matches_capacity, emitted);
                for (word = 0; word != words; ++word) continuing |= live[word] & ~final_steps[word];
                for (word = 0; word != counter_words; ++word) live[words + word] = 0;
                frame->parent = current, frame->byte_offset = opening;
                frame->matches_begin = matches_begin, frame->matches_end = *emitted;
            }
            // Attributes are read even where none is selected, as their values may hold any `>`.
            for (;;) {
                sz_size_t value_begin, value_end, attribute_end;
                sz_u8_t quote;
                attribute = sz_subdocuments_markup_skip_blanks_simt_(text, length, attribute);
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
                attribute_end = sz_subdocuments_markup_skip_name_simt_(text, length, attribute);
                value_begin = sz_subdocuments_markup_skip_blanks_simt_(text, length, attribute_end);
                if (attribute_end == attribute || value_begin == length || text[value_begin] != '=')
                    return sz_invalid_syntax_k;
                value_begin = sz_subdocuments_markup_skip_blanks_simt_(text, length, value_begin + 1);
                if (value_begin == length || (text[value_begin] != '"' && text[value_begin] != '\''))
                    return sz_invalid_syntax_k;
                quote = text[value_begin++];
                value_end = sz_subdocuments_markup_find_simt_(text, length, value_begin, quote);
                if (value_end == length) return sz_invalid_syntax_k;
                // Namespace declarations are no attributes in XPath's data model.
                if (continuing &&
                    !(attribute_end - attribute >= 5 && text[attribute] == 'x' && text[attribute + 1] == 'm' &&
                      text[attribute + 2] == 'l' && text[attribute + 3] == 'n' && text[attribute + 4] == 's' &&
                      (attribute_end - attribute == 5 || text[attribute + 5] == ':'))) {
                    sz_u64_t *const attribute_live = live + words + counter_words;
                    sz_subdocuments_node_live_simt_(
                        engine, live, counters, sz_subdocuments_node_attribute_k,
                        sz_subdocuments_markup_name_simt_(engine, live, text + attribute, attribute_end - attribute,
                                                          key_bytes),
                        attribute_live);
                    sz_subdocuments_report_simt_(engine, attribute_live, document, value_begin, value_end - value_begin,
                                                 matches, matches_capacity, emitted);
                }
                attribute = value_end + 1;
            }
            if (skipped_depth) skipped_depth += !empty;
            else if (empty)
                sz_subdocuments_close_simt_(matches, matches_capacity, matches_begin, frame->matches_end, tag_end + 1);
            else if (continuing) {
                current = top++;
                // The next element and an attribute's live set take the two slots past the last.
                if (top + 2 > frames) return sz_bad_alloc_k;
            }
            else
                skipped_depth = 1, skipped_opening = opening, skipped_begin = matches_begin,
                skipped_end = frame->matches_end;
        }
        position = tag_end + 1, text_begin = position;
    }
    return current != 0 || skipped_depth || !rooted ? sz_invalid_syntax_k : sz_success_k;
}

/**
 *  @brief Frames a walk of the @p length bytes at @p text needs at most: past the nesting of its
 *      containers or elements, and where a pending run may open, one per value the run keeps, which
 *      its structural bytes bound.
 *
 *  XML is lexed as the markup walk lexes it, so a malformed document is measured as far as
 *  the walk reads it.
 */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_frames_simt_(sz_subdocuments_engine_t const *engine, sz_u8_t const *text,
                                                          sz_size_t length) {
    sz_size_t depth = 0, deepest = 0, structurals = 0, position = 0;
    sz_u64_t relative = 0;
    sz_subdocuments_cursor_simt_t cursor;
    if (engine->format == sz_subdocuments_xml_k) {
        for (; (position = sz_subdocuments_markup_find_simt_(text, length, position, '<')) + 1 < length; ++position) {
            sz_u8_t const kind = text[position + 1];
            if (kind == '!' && length - position >= 9 && text[position + 2] == '[' && text[position + 3] == 'C' &&
                text[position + 4] == 'D' && text[position + 5] == 'A' && text[position + 6] == 'T' &&
                text[position + 7] == 'A' && text[position + 8] == '[')
                position = sz_subdocuments_markup_end_simt_(text, length, position + 9, ']', 2);
            else if (kind == '!' && length - position >= 4 && text[position + 2] == '-' && text[position + 3] == '-')
                position = sz_subdocuments_markup_end_simt_(text, length, position + 4, '-', 2);
            else if (kind == '?') position = sz_subdocuments_markup_end_simt_(text, length, position + 2, '?', 1);
            else if (kind == '!' || kind == '/') {
                depth -= kind == '/' && depth;
                position = sz_subdocuments_markup_find_simt_(text, length, position, '>');
            }
            else {
                // Quoted attribute values may hold any `>`.
                for (++position; position < length && text[position] != '>'; ++position)
                    if (text[position] == '"' || text[position] == '\'')
                        position = sz_subdocuments_markup_find_simt_(text, length, position + 1, text[position]);
                if (position < length && text[position - 1] != '/') deepest = sz_max_of_two(deepest, ++depth);
            }
        }
        return deepest + 3;
    }
    cursor.text = text, cursor.length = length, cursor.escaped = 0, cursor.in_string = 0, cursor.scalar = 0;
    cursor.slashes = 0;
    for (; position < length; position += 64) {
        sz_subdocuments_load_simt_(&cursor, position);
        structurals += (sz_size_t)__popcll(cursor.pending);
        for (sz_u64_t brackets = cursor.opens | cursor.closes; brackets; brackets &= brackets - 1)
            if (cursor.opens >> sz_subdocuments_ctz_simt_(brackets) & 1) deepest = sz_max_of_two(deepest, ++depth);
            else if (depth) --depth;
    }
    for (sz_size_t word = 0; word != engine->words_count; ++word) relative |= engine->relative_steps[word];
    return engine->filters_count || relative ? structurals + engine->filters_count + 2 : deepest + 1;
}

/** Lists @p document of @p length bytes for the cooperative walk, emptying its count, when it is
 *  past @p large_bytes or needs more than @c sz_subdocuments_deep_frames_simt_k of its @p frames,
 *  and the listed chunks stay within @c sz_subdocuments_large_chunks_simt_k; otherwise for the
 *  second launch, measuring its frames when @p frames is zero. */
STRINGZILLA_DEVICE void sz_subdocuments_list_simt_(sz_subdocuments_engine_t const *engine,
                                                   sz_subdocuments_round_simt_t const *round, sz_u8_t const *text,
                                                   sz_size_t document, sz_size_t length, sz_size_t frames,
                                                   sz_size_t large_bytes) {
    unsigned long long const chunks = sz_size_divide_round_up(length, sz_subdocuments_chunk_bytes_simt_k);
    unsigned long long *const listed_chunks = (unsigned long long *)&round->header->large_chunks;
    if (round->large_bytes != STRINGZILLA_SIZE_MAX && length <= 0xFFFFFFFFull &&
        (length > large_bytes || frames > sz_subdocuments_deep_frames_simt_k)) {
        if (atomicAdd(listed_chunks, chunks) + chunks <= sz_subdocuments_large_chunks_simt_k) {
            sz_size_t const slot = (sz_size_t)atomicAdd((unsigned long long *)&round->header->large_count, 1ull);
            if (round->counted) round->counted[document] = 0;
            else round->counts[document * round->counts_stride] = 0;
            round->large[2 * slot] = document, round->large[2 * slot + 1] = length;
            return;
        }
        atomicAdd(listed_chunks, 0ull - chunks);
    }
    if (!frames) frames = sz_subdocuments_frames_simt_(engine, text, length);
    round->deep[atomicAdd((unsigned long long *)&round->header->deep_count, 1ull)] = document;
    atomicMax((unsigned long long *)&round->header->frames, (unsigned long long)frames);
}

#pragma endregion Device Helpers

#pragma region Kernels

/**
 *  @brief Walks every document, or every listed one, as walker @p walker of a launch's walkers,
 *      each a thread or a warp: counting, or storing at the offsets the counts scanned into.
 *
 *  JSON documents walk on the lanes of a warp, with @p warp its vendor's lane steps, and XML ones
 *  on a thread, with @p warp null.
 *
 *  A counting walk writes each count, and lists a document needing more frames than the walker's
 *  stack holds for the second launch, measuring how many; only lane zero of the @p lanes walking
 *  one document lists or refuses it, and every lane writes the same values. A storing walk writes
 *  each document's boundary and, when any of its matches fits, the matches; one cut short by its
 *  stack is stored again by the second launch. With @c staging_capacity set, the first launch's
 *  counting walk also writes each document's matches into the walker's own staging slots while they
 *  fit, marking where in @c staged, and its storing walk moves those instead of walking. A document
 *  past the header's @c large_bytes, or under the same paths one needing a stack past
 *  @c sz_subdocuments_deep_frames_simt_k frames, is left to the cooperative walk: counting lists it
 *  and empties its count, and storing, which runs out of frames early, leaves its matches to be
 *  written over. A large one the cooperative walk has no room for goes to the second launch, which
 *  alone stores it.
 */
STRINGZILLA_DEVICE void sz_subdocuments_walk_documents_simt_(sz_subdocuments_engine_t const *engine,
                                                             sz_strings_view_t const *documents,
                                                             sz_subdocuments_round_simt_t const *round,
                                                             sz_size_t walker, sz_size_t lane, sz_size_t lanes,
                                                             sz_subdocuments_warp_simt_t const *warp) {
    sz_u64_t *const stack = round->stacks + walker * round->frames * round->frame_words;
    sz_u8_t *const key = (sz_u8_t *)round->keys + walker * round->key_bytes;
    sz_size_t const large_bytes = (sz_size_t)round->header->large_bytes;
    sz_bool_t const leader = (sz_bool_t)(lane == 0), staging = (sz_bool_t)(round->staging_capacity && !round->listed);
    sz_u32_t staged = 0;
    // Only the walkers the stacks were sized for walk, whatever the grid rounds up to.
    for (sz_size_t item = walker; walker < round->threads && item < round->items; item += round->threads) {
        sz_size_t const document = round->listed ? (sz_size_t)round->deep[item] : item;
        sz_u8_t const *const text = (sz_u8_t const *)sz_strings_start_simt_(*documents, document);
        sz_size_t const length = sz_strings_length_simt_(*documents, document);
        sz_match_t *const matches = round->storing ? round->matches
                                    : staging      ? round->staging + walker * round->staging_capacity + staged
                                                   : (sz_match_t *)STRINGZILLA_NULL;
        sz_size_t const matches_capacity = round->storing ? round->matches_capacity
                                           : staging      ? round->staging_capacity - staged
                                                          : 0;
        sz_size_t emitted = round->storing ? round->counted[document] : 0;
        sz_bool_t const large = (sz_bool_t)(!round->listed && length > large_bytes && length <= 0xFFFFFFFFull);
        sz_status_t status;
        if (round->storing) {
            round->matches_offsets[document] = emitted;
            if (document + 1 == documents->count)
                round->matches_offsets[documents->count] = round->counted[documents->count];
            if (emitted >= round->matches_capacity || large) continue;
            if (staging && round->staged[document] != STRINGZILLA_SIZE_MAX) {
                sz_match_t const *const source = round->staging + round->staged[document];
                sz_size_t const kept = sz_min_of_two(round->counted[document + 1], round->matches_capacity) - emitted;
                for (sz_size_t index = lane; index < kept; index += lanes)
                    round->matches[emitted + index] = source[index];
                continue;
            }
        }
        else if (large) {
            if (staging) round->staged[document] = STRINGZILLA_SIZE_MAX;
            if (leader) sz_subdocuments_list_simt_(engine, round, text, document, length, 0, large_bytes);
            continue;
        }
        status = !warp ? sz_subdocuments_walk_markup_simt_(engine, text, length, document, stack, round->frames,
                                                           round->frame_words, round->key_bytes, matches,
                                                           matches_capacity, &emitted)
                       : sz_subdocuments_walk_simt_(engine, text, length, document, stack, round->frames,
                                                    round->frame_words, key, round->key_bytes, matches,
                                                    matches_capacity, &emitted, warp);
        if (round->storing) continue;
        if (staging) {
            sz_bool_t const kept = (sz_bool_t)(status == sz_success_k && emitted <= round->staging_capacity - staged);
            round->staged[document] = kept ? walker * round->staging_capacity + staged : STRINGZILLA_SIZE_MAX;
            staged += kept ? (sz_u32_t)emitted : 0u;
        }
        if (status == sz_success_k) {
            if (round->counted) round->counted[document] = emitted;
            else round->counts[document * round->counts_stride] = emitted;
        }
        else if (!leader) continue;
        else if (status == sz_invalid_syntax_k)
            atomicMin((unsigned long long *)&round->header->first_malformed, (unsigned long long)document);
        else if (!round->listed)
            sz_subdocuments_list_simt_(engine, round, text, document, length,
                                       sz_subdocuments_frames_simt_(engine, text, length), large_bytes);
    }
}

/** Lowers the header's @c large_bytes to one of @c large_shares of the batch's bytes once any
 *  document passes both that share and the round's @c large_bytes, so the documents past their
 *  share walk over chunks only in a batch that takes the cooperative walk anyway. */
static __global__ void sz_subdocuments_route_simt_kernel_(sz_strings_view_t documents,
                                                          sz_subdocuments_round_simt_t round) {
    sz_size_t const batch = (sz_size_t)(sz_strings_start_simt_(documents, documents.count) -
                                        sz_strings_start_simt_(documents, 0));
    sz_size_t const share = batch / round.large_shares, floor = sz_max_of_two(round.large_bytes, share);
    for (sz_size_t document = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; document < documents.count;
         document += (sz_size_t)gridDim.x * blockDim.x) {
        sz_size_t const length = sz_strings_length_simt_(documents, document);
        if (length > floor && length <= 0xFFFFFFFFull) round.header->large_bytes = share;
    }
}

/** @ref sz_subdocuments_walk_documents_simt_ with a thread walking each XML document. */
static __global__ void sz_subdocuments_walk_simt_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
                                                         sz_subdocuments_round_simt_t round) {
    sz_subdocuments_walk_documents_simt_(&engine, &documents, &round, (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x,
                                         0, 1, (sz_subdocuments_warp_simt_t const *)STRINGZILLA_NULL);
}

/** Scans one ticket's own tile of @p values in place, seeded by the totals the tiles before
 *  it chained through @p chain, so the whole scan is one launch, as
 *  @c sz_regex_scan_simt_kernel_ does. */
static __global__ void sz_subdocuments_scan_simt_kernel_(sz_size_t *values, sz_size_t count,
                                                         sz_size_t elements_per_tile, sz_size_t *chain) {
    __shared__ sz_size_t shared[sz_subdocuments_threads_per_block_simt_k];
    sz_size_t const tiles = sz_size_divide_round_up(count, elements_per_tile);
    sz_size_t tile, begin, end, running = 0, first;
    if (blockIdx.x >= tiles) return;
    tile = sz_chain_ticket_simt_(chain);
    begin = tile * elements_per_tile;
    end = sz_min_of_two(begin + elements_per_tile, count);
    for (first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t total = 0;
        sz_block_scan_simt_(index < end ? values[index] : 0, shared, &total);
        running += total;
        __syncthreads();
    }
    running = sz_chain_offset_simt_(chain, tile, tiles, running);
    for (first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t const value = index < end ? values[index] : 0;
        sz_size_t total = 0;
        sz_size_t const exclusive = sz_block_scan_simt_(value, shared, &total);
        if (index < end) values[index] = running + exclusive;
        running += total;
        __syncthreads();
    }
}

#pragma endregion Kernels

#pragma region Large Documents

/**
 *  @brief The effect of chunk @p earlier then chunk @p later on the escape, quote and scalar
 *      carries, or, unless @p compose is set, the sum of two counts.
 *
 *  An effect keeps, for each escape carry its chunk may start with, the escape carry it leaves
 *  at bits 1 and 2, the parity of the quotes it holds at bits 3 and 4, and whether its last
 *  byte belongs to a scalar outside strings at bits 5 and 6. Bit zero marks an effect, so zero
 *  is the identity of both operations, and bit 7 one that starts a document, which ignores
 *  every effect before it.
 */
STRINGZILLA_DEVICE sz_u64_t sz_subdocuments_combine_simt_(sz_u64_t earlier, sz_u64_t later, sz_u32_t compose) {
    sz_u64_t composed = 1 | (earlier & 0x80);
    if (!compose) return earlier + later;
    if (!(earlier & 1) || (later & 0x80)) return later;
    if (!(later & 1)) return earlier;
    for (sz_u64_t escape = 0; escape != 2; ++escape) {
        sz_u64_t const middle = earlier >> (1 + escape) & 1;
        composed |= (later >> (1 + middle) & 1) << (1 + escape);
        composed |= ((earlier >> (3 + escape) ^ later >> (3 + middle)) & 1) << (3 + escape);
        composed |= (later >> (5 + middle) & 1) << (5 + escape);
    }
    return composed;
}

/** The block's exclusive scan of @p value under @ref sz_subdocuments_combine_simt_, its total
 *  left in @p total. */
STRINGZILLA_DEVICE sz_u64_t sz_subdocuments_block_scan_simt_(sz_u64_t value, sz_u64_t *shared, sz_u32_t compose,
                                                             sz_u64_t *total) {
    unsigned const lane = threadIdx.x;
    sz_u64_t exclusive;
    shared[lane] = value;
    __syncthreads();
    for (unsigned offset = 1; offset < blockDim.x; offset *= 2) {
        sz_u64_t const earlier = lane >= offset ? shared[lane - offset] : 0;
        __syncthreads();
        shared[lane] = sz_subdocuments_combine_simt_(earlier, shared[lane], compose);
        __syncthreads();
    }
    exclusive = lane ? shared[lane - 1] : 0;
    *total = shared[blockDim.x - 1];
    __syncthreads(); // ! The caller reuses `shared` for the next tile.
    return exclusive;
}

/** The listed document holding item @p index, numbered per chunk by @p numbered or, when that
 *  is null, chunk @p index itself: the last whose first chunk numbers at most @p index,
 *  @c listed past every one. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_slot_simt_(sz_subdocuments_large_simt_t const *large,
                                                        sz_u64_t const *numbered, sz_size_t index) {
    sz_size_t low = 0, high = large->listed;
    while (low != high) {
        sz_size_t const middle = high - (high - low) / 2, first = (sz_size_t)large->firsts[middle];
        if ((numbered ? (sz_size_t)numbered[first] : first) <= index) low = middle;
        else high = middle - 1;
    }
    return low;
}

/** Where chunk @p chunk lies; past every listed document, in the batch's first
 *  document with no length. */
STRINGZILLA_DEVICE sz_subdocuments_place_simt_t sz_subdocuments_place_simt_(sz_subdocuments_large_simt_t const *large,
                                                                            sz_size_t chunk) {
    sz_size_t const slot = sz_subdocuments_slot_simt_(large, (sz_u64_t const *)STRINGZILLA_NULL, chunk);
    sz_bool_t const listed = (sz_bool_t)(slot != large->listed);
    sz_subdocuments_place_simt_t place;
    place.slot = slot, place.first = (sz_size_t)large->firsts[slot];
    place.next = listed ? (sz_size_t)large->firsts[slot + 1] : large->chunks;
    place.document = listed ? (sz_size_t)large->list[2 * slot] : 0;
    place.length = listed ? (sz_size_t)large->list[2 * slot + 1] : 0;
    return place;
}

/** Starts @p cursor at chunk @p chunk of the document at @p place, @p text, with the carries the
 *  scanned effects of its chunks before it leave. */
STRINGZILLA_DEVICE void sz_subdocuments_chunk_cursor_simt_(sz_subdocuments_large_simt_t const *large,
                                                           sz_subdocuments_place_simt_t const *place,
                                                           sz_u8_t const *text, sz_size_t chunk,
                                                           sz_subdocuments_cursor_simt_t *cursor) {
    sz_u64_t const before = chunk != place->first ? large->transfers[chunk] : 0, in_string = before >> 3 & 1;
    cursor->text = text, cursor->length = place->length;
    cursor->escaped = before >> 1 & 1, cursor->in_string = 0 - in_string;
    cursor->scalar = (before >> 5 & 1) & (in_string ^ 1), cursor->slashes = 0;
}

/** Whether the last byte of the chunk `[start, end)` belongs to a scalar outside strings, its
 *  first byte escaped when @p escape is set: every byte but blank space, brackets, separators
 *  and unescaped quotes does. */
STRINGZILLA_DEVICE sz_u64_t sz_subdocuments_scalar_end_simt_(sz_u8_t const *text, sz_size_t start, sz_size_t end,
                                                             sz_u64_t escape) {
    sz_u8_t const byte = text[end - 1];
    sz_size_t run = 0;
    if (byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' || byte == ':' || byte == ',' ||
        (byte | 0x20) == '{' || (byte | 0x20) == '}')
        return 0;
    if (byte != '"') return 1;
    while (end - 1 - run != start && text[end - 2 - run] == '\\') ++run;
    return (run + (end - 1 - run == start ? escape : 0)) & 1;
}

/** The last container before @p index shallower than @p depth, or
 *  @c sz_subdocuments_no_container_simt_k: up the levels of minima until one holds a shallower
 *  entry before it, then down to the last such container. */
STRINGZILLA_DEVICE sz_u32_t sz_subdocuments_shallower_simt_(sz_subdocuments_large_simt_t const *large, sz_size_t index,
                                                            sz_u32_t depth) {
    sz_size_t const fanout = sz_subdocuments_minima_fanout_simt_k;
    sz_size_t level = 0, position = index, found = 0;
    sz_bool_t hit = sz_false_k;
    for (; !hit; ++level) {
        sz_u32_t const *const values = large->minima + large->minima_offsets[level];
        sz_size_t const first = position - position % fanout;
        for (found = position; found != first && !hit;) hit = (sz_bool_t)(values[--found] < depth);
        if (!hit && !first) return sz_subdocuments_no_container_simt_k;
        position = first / fanout;
    }
    for (--level; level; --level) {
        sz_u32_t const *const values = large->minima + large->minima_offsets[level - 1];
        sz_size_t const below = large->minima_offsets[level] - large->minima_offsets[level - 1];
        sz_size_t child = sz_min_of_two(found * fanout + fanout, below);
        do --child;
        while (values[child] >= depth);
        found = child;
    }
    return (sz_u32_t)found;
}

/** Whether token @p token of a document whose tokens end at @p end is a value its container counts:
 *  an opening bracket, a scalar, or a string no colon follows, as one always follows a key. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_token_value_simt_(sz_subdocuments_large_simt_t const *large,
                                                               sz_size_t token, sz_size_t end) {
    sz_u8_t const kind = large->kinds[token];
    if (kind == sz_subdocuments_token_quote_k)
        return (sz_bool_t)(token + 2 >= end || large->kinds[token + 2] != sz_subdocuments_token_colon_k);
    return (sz_bool_t)(kind <= sz_subdocuments_token_array_k || kind == sz_subdocuments_token_scalar_k);
}

/** Live set @p mode of @p container, zero tentative and one exact, and the descendant steps its
 *  ancestors carry into it, null without descendant segments. */
STRINGZILLA_DEVICE sz_u64_t const *sz_subdocuments_live_simt_(sz_subdocuments_large_simt_t const *large,
                                                              sz_size_t words, sz_size_t mode, sz_size_t container) {
    return large->lives + (mode * large->containers_count + container) * words;
}

STRINGZILLA_DEVICE sz_u64_t const *sz_subdocuments_carried_simt_(sz_subdocuments_large_simt_t const *large,
                                                                 sz_size_t words, sz_size_t mode, sz_size_t container) {
    return large->descendants
               ? large->carried + ((large->carried_buffer * 2 + mode) * large->containers_count + container) * words
               : (sz_u64_t const *)STRINGZILLA_NULL;
}

/** Whether any step live in @p live, joined with @p carried unless null, advances into children. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_continues_simt_(sz_subdocuments_engine_t const *engine,
                                                             sz_u64_t const *live, sz_u64_t const *carried) {
    sz_u64_t continuing = 0;
    for (sz_size_t word = 0; word != engine->words_count; ++word)
        continuing |= (live[word] | (carried ? carried[word] : 0)) & ~engine->final_steps[word];
    return (sz_bool_t)(continuing != 0);
}

/** The name of the key of the member whose value is token @p value, or
 *  @c sz_subdocuments_no_name_k past every name and for a malformed escape, which only the check
 *  of an entered object refuses. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_member_name_simt_(sz_subdocuments_engine_t const *engine,
                                                               sz_subdocuments_large_simt_t const *large,
                                                               sz_u8_t const *text, sz_size_t value, sz_u8_t *key) {
    sz_u8_t const *bytes;
    sz_size_t length, index;
    // A member's value follows its key's two quotes and a colon.
    if (!engine->names_slots_count || value < 3 || large->kinds[value - 1] != sz_subdocuments_token_colon_k ||
        large->kinds[value - 3] != sz_subdocuments_token_quote_k)
        return sz_subdocuments_no_name_k;
    bytes = text + large->positions[value - 3] + 1;
    length = large->positions[value - 2] - large->positions[value - 3] - 1;
    for (index = 0; index != length && bytes[index] != '\\';) ++index;
    if (index != length) {
        length = sz_subdocuments_unescape_simt_(bytes, length, key, large->key_bytes);
        if (length == STRINGZILLA_SIZE_MAX) return sz_subdocuments_no_name_k;
        bytes = key;
    }
    return length <= large->key_bytes ? sz_subdocuments_name_simt_(engine, bytes, length) : sz_subdocuments_no_name_k;
}

/** What the walk expects at token @p token inside container @p current of a document whose tokens
 *  begin at @p begin: whatever may follow the token before. */
STRINGZILLA_DEVICE sz_subdocuments_expect_t sz_subdocuments_token_expect_simt_(
    sz_subdocuments_large_simt_t const *large, sz_size_t begin, sz_size_t token, sz_u32_t current) {
    sz_u8_t previous;
    if (token == begin) return sz_subdocuments_expect_value_k;
    previous = large->kinds[token - 1];
    if (previous == sz_subdocuments_token_object_k) return sz_subdocuments_expect_key_or_close_k;
    if (previous == sz_subdocuments_token_array_k) return sz_subdocuments_expect_value_or_close_k;
    if (previous == sz_subdocuments_token_colon_k) return sz_subdocuments_expect_value_k;
    if (previous == sz_subdocuments_token_comma_k)
        return current != sz_subdocuments_no_container_simt_k &&
                       large->kinds[large->openers[current]] == sz_subdocuments_token_object_k
                   ? sz_subdocuments_expect_key_k
                   : sz_subdocuments_expect_value_k;
    // A string a colon follows is a key, so the colon completes it, and anything
    // else follows a value.
    if (previous == sz_subdocuments_token_quote_end_k && large->kinds[token] == sz_subdocuments_token_colon_k)
        return sz_subdocuments_expect_colon_k;
    return current != sz_subdocuments_no_container_simt_k ? sz_subdocuments_expect_comma_or_close_k
                                                          : sz_subdocuments_expect_end_k;
}

/** Whether the walk accepts token @p token inside container @p current of the document of
 *  @p length bytes at @p text whose tokens are `[begin, end)`, as @c sz_subdocuments_walk_
 *  checks every structural byte of a container it enters: a key in the wrong place fails here,
 *  as its colon would there. */
STRINGZILLA_DEVICE sz_bool_t sz_subdocuments_token_valid_simt_(sz_subdocuments_engine_t const *engine,
                                                               sz_subdocuments_large_simt_t const *large,
                                                               sz_u8_t const *text, sz_size_t length, sz_size_t begin,
                                                               sz_size_t end, sz_size_t token, sz_u32_t current) {
    sz_subdocuments_expect_t const expect = sz_subdocuments_token_expect_simt_(large, begin, token, current);
    sz_u8_t const kind = large->kinds[token];
    sz_size_t const position = large->positions[token], words = engine->words_count;
    sz_bool_t const value = (sz_bool_t)(expect == sz_subdocuments_expect_value_k ||
                                        expect == sz_subdocuments_expect_value_or_close_k);
    if (kind == sz_subdocuments_token_colon_k) return (sz_bool_t)(expect == sz_subdocuments_expect_colon_k);
    if (kind == sz_subdocuments_token_comma_k) return (sz_bool_t)(expect == sz_subdocuments_expect_comma_or_close_k);
    if (kind == sz_subdocuments_token_object_end_k || kind == sz_subdocuments_token_array_end_k)
        return (sz_bool_t)(current != sz_subdocuments_no_container_simt_k &&
                           large->kinds[large->openers[current]] + 2 == kind &&
                           (expect == sz_subdocuments_expect_comma_or_close_k ||
                            expect == (kind == sz_subdocuments_token_object_end_k
                                           ? sz_subdocuments_expect_key_or_close_k
                                           : sz_subdocuments_expect_value_or_close_k)));
    if (kind == sz_subdocuments_token_scalar_k)
        return (sz_bool_t)(value && sz_subdocuments_scalar_length_simt_(text, position, length) != 0);
    if (kind != sz_subdocuments_token_quote_k || sz_subdocuments_token_value_simt_(large, token, end)) return value;
    if (expect != sz_subdocuments_expect_key_k && expect != sz_subdocuments_expect_key_or_close_k) return sz_false_k;
    // A key the walk looks up must unescape.
    if (!engine->names_slots_count ||
        !sz_subdocuments_needs_name_simt_(engine, sz_subdocuments_live_simt_(large, words, 0, current),
                                          sz_subdocuments_carried_simt_(large, words, 0, current)))
        return sz_true_k;
    return (sz_bool_t)(sz_subdocuments_unescape_simt_(text + position + 1, large->positions[token + 1] - position - 1,
                                                      (sz_u8_t *)STRINGZILLA_NULL, 0) != STRINGZILLA_SIZE_MAX);
}

/** Bytes the value at token @p token of the document of @p length bytes at @p text spans:
 *  through its closing quote, through the bracket that closes container @p container it opens,
 *  or across its scalar. */
STRINGZILLA_DEVICE sz_size_t sz_subdocuments_token_length_simt_(sz_subdocuments_large_simt_t const *large,
                                                                sz_u8_t const *text, sz_size_t length, sz_size_t token,
                                                                sz_size_t container) {
    sz_u8_t const kind = large->kinds[token];
    sz_size_t const position = large->positions[token];
    if (kind <= sz_subdocuments_token_array_k) return large->positions[large->closers[container]] + 1 - position;
    if (kind == sz_subdocuments_token_quote_k) return large->positions[token + 1] + 1 - position;
    return sz_subdocuments_scalar_length_simt_(text, position, length);
}

/** Counts each listed document's chunks into its slot, none for one at or past @c bound,
 *  and clears its verdict. */
static __global__ void sz_subdocuments_large_lengths_simt_kernel_(sz_subdocuments_large_simt_t large) {
    for (sz_size_t slot = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; slot <= large.listed;
         slot += large.threads) {
        sz_bool_t const kept = (sz_bool_t)(slot != large.listed && large.list[2 * slot] < large.bound);
        large.firsts[slot] =
            kept ? sz_size_divide_round_up((sz_size_t)large.list[2 * slot + 1], sz_subdocuments_chunk_bytes_simt_k) : 0;
        if (slot != large.listed) large.malformed[slot] = 0;
    }
}

/** Records each chunk's effect on the escape, quote and scalar carries, for either escape carry it
 *  starts with, marking the first of each document. */
static __global__ void sz_subdocuments_large_summary_simt_kernel_(sz_strings_view_t documents,
                                                                  sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk <= large.chunks;
         chunk += large.threads) {
        sz_subdocuments_place_simt_t const place = sz_subdocuments_place_simt_(&large, chunk);
        sz_u8_t const *const text = (sz_u8_t const *)sz_strings_start_simt_(documents, place.document);
        sz_size_t const start = (chunk - place.first) * sz_subdocuments_chunk_bytes_simt_k;
        sz_size_t const end = sz_min_of_two(start + sz_subdocuments_chunk_bytes_simt_k, place.length);
        sz_u64_t transfer = chunk == place.first ? 0x81 : 1;
        sz_subdocuments_cursor_simt_t cursor;
        if (place.slot == large.listed) {
            large.transfers[chunk] = 0;
            continue;
        }
        cursor.text = text, cursor.length = place.length, cursor.scalar = 0, cursor.slashes = 0;
        for (sz_u64_t escape = 0; escape != 2; ++escape) {
            cursor.escaped = escape, cursor.in_string = 0;
            for (sz_size_t offset = start; offset < end; offset += 64) sz_subdocuments_load_simt_(&cursor, offset);
            transfer |= (cursor.escaped & 1) << (1 + escape) | (cursor.in_string & 1) << (3 + escape) |
                        sz_subdocuments_scalar_end_simt_(text, start, end, escape) << (5 + escape);
        }
        large.transfers[chunk] = transfer;
    }
}

/** Reduces each block's own tile of @c scanned into @c tiles, the first of
 *  a scan's three launches. */
static __global__ void sz_subdocuments_large_reduce_simt_kernel_(sz_subdocuments_large_simt_t large) {
    __shared__ sz_u64_t shared[sz_subdocuments_threads_per_block_simt_k];
    sz_size_t const count = large.scanned_count, per_tile = sz_size_divide_round_up(count, large.scan_tiles);
    sz_size_t const begin = sz_min_of_two((sz_size_t)blockIdx.x * per_tile, count);
    sz_size_t const end = sz_min_of_two(begin + per_tile, count);
    sz_u64_t running = 0, total;
    for (sz_size_t first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_subdocuments_block_scan_simt_(index < end ? large.scanned[index] : 0, shared, large.compose, &total);
        running = sz_subdocuments_combine_simt_(running, total, large.compose);
    }
    if (threadIdx.x == 0) large.tiles[blockIdx.x] = running;
}

/** Scans @c tiles in place on one block, the second of a scan's three launches. */
static __global__ void sz_subdocuments_large_carry_simt_kernel_(sz_subdocuments_large_simt_t large) {
    __shared__ sz_u64_t shared[sz_subdocuments_threads_per_block_simt_k];
    sz_u64_t running = 0, total;
    for (sz_size_t first = 0; first < large.scan_tiles; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_u64_t const exclusive = sz_subdocuments_block_scan_simt_(index < large.scan_tiles ? large.tiles[index] : 0,
                                                                    shared, large.compose, &total);
        if (index < large.scan_tiles)
            large.tiles[index] = sz_subdocuments_combine_simt_(running, exclusive, large.compose);
        running = sz_subdocuments_combine_simt_(running, total, large.compose);
    }
}

/** Scans each block's own tile of @c scanned in place from the carry before it, the
 *  last of three launches. */
static __global__ void sz_subdocuments_large_apply_simt_kernel_(sz_subdocuments_large_simt_t large) {
    __shared__ sz_u64_t shared[sz_subdocuments_threads_per_block_simt_k];
    sz_size_t const count = large.scanned_count, per_tile = sz_size_divide_round_up(count, large.scan_tiles);
    sz_size_t const begin = sz_min_of_two((sz_size_t)blockIdx.x * per_tile, count);
    sz_size_t const end = sz_min_of_two(begin + per_tile, count);
    sz_u64_t running = large.tiles[blockIdx.x], total;
    for (sz_size_t first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_u64_t const exclusive = sz_subdocuments_block_scan_simt_(index < end ? large.scanned[index] : 0, shared,
                                                                    large.compose, &total);
        if (index < end) large.scanned[index] = sz_subdocuments_combine_simt_(running, exclusive, large.compose);
        running = sz_subdocuments_combine_simt_(running, total, large.compose);
    }
}

/** Counts each chunk's tokens and opening brackets, the depth it ends at, and how far its
 *  depth dips and rises, marking its document malformed for a `/` outside strings. */
static __global__ void sz_subdocuments_large_count_simt_kernel_(sz_strings_view_t documents,
                                                                sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk <= large.chunks;
         chunk += large.threads) {
        sz_subdocuments_place_simt_t const place = sz_subdocuments_place_simt_(&large, chunk);
        sz_u8_t const *const text = (sz_u8_t const *)sz_strings_start_simt_(documents, place.document);
        sz_size_t const start = (chunk - place.first) * sz_subdocuments_chunk_bytes_simt_k;
        sz_size_t const end = sz_min_of_two(start + sz_subdocuments_chunk_bytes_simt_k, place.length);
        sz_i64_t depth = 0, lowest = 0, highest = 0;
        sz_u64_t tokens = 0, opens = 0;
        sz_subdocuments_cursor_simt_t cursor;
        if (place.slot == large.listed) {
            large.tokens[chunk] = 0, large.opens[chunk] = 0, large.depths[chunk] = 0, large.extremes[chunk] = 0;
            continue;
        }
        sz_subdocuments_chunk_cursor_simt_(&large, &place, text, chunk, &cursor);
        for (sz_size_t offset = start; offset < end; offset += 64) {
            sz_subdocuments_load_simt_(&cursor, offset);
            tokens += (sz_u64_t)__popcll(cursor.pending);
            for (sz_u64_t brackets = cursor.opens | cursor.closes; brackets; brackets &= brackets - 1)
                if (cursor.opens >> sz_subdocuments_ctz_simt_(brackets) & 1)
                    ++opens, ++depth, highest = sz_max_of_two(highest, depth);
                else --depth, lowest = sz_min_of_two(lowest, depth);
        }
        large.tokens[chunk] = tokens, large.opens[chunk] = opens, large.depths[chunk] = (sz_u64_t)depth;
        large.extremes[chunk] = (sz_u64_t)-lowest | (sz_u64_t)highest << 32;
        if (cursor.slashes) large.malformed[place.slot] = 1;
    }
}

/** Refuses a document whose depth dips below zero, counted from its first chunk, ends above it,
 *  ends inside a string, or that the count marked, as no walk accepts one. */
static __global__ void sz_subdocuments_large_check_simt_kernel_(sz_subdocuments_round_simt_t round,
                                                                sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk < large.chunks;
         chunk += large.threads) {
        sz_subdocuments_place_simt_t const place = sz_subdocuments_place_simt_(&large, chunk);
        sz_u64_t const base = large.depths[place.first];
        sz_i64_t const depth = (sz_i64_t)(large.depths[chunk] - base);
        sz_u32_t malformed = depth - (sz_i64_t)(large.extremes[chunk] & 0xFFFFFFFFull) < 0;
        if (place.slot == large.listed) continue;
        if (chunk == place.first)
            malformed |= large.depths[place.next] != base || (large.transfers[place.next] >> 3 & 1) != 0 ||
                         large.malformed[place.slot];
        if (malformed) {
            large.malformed[place.slot] = 1;
            atomicMin((unsigned long long *)&round.header->first_malformed, (unsigned long long)place.document);
        }
    }
}

/** Drops the chunks of every refused document, so no token or container is numbered in one, and
 *  publishes the deepest any kept document nests. */
static __global__ void sz_subdocuments_large_drop_simt_kernel_(sz_subdocuments_round_simt_t round,
                                                               sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk < large.chunks;
         chunk += large.threads) {
        sz_size_t const slot = sz_subdocuments_slot_simt_(&large, (sz_u64_t const *)STRINGZILLA_NULL, chunk);
        sz_i64_t highest;
        if (slot == large.listed) continue;
        if (large.malformed[slot]) {
            large.tokens[chunk] = 0, large.opens[chunk] = 0;
            continue;
        }
        highest = (sz_i64_t)(large.depths[chunk] - large.depths[large.firsts[slot]] + (large.extremes[chunk] >> 32));
        if (highest > 0) atomicMax((unsigned long long *)&round.header->deepest, (unsigned long long)highest);
    }
}

/** Writes each chunk's tokens, and for every container it opens the token, the depth counted from
 *  its document's first chunk, an empty element count, and the parent when the chunk opened that
 *  too, @c sz_subdocuments_outer_container_simt_k when an earlier one did. */
static __global__ void sz_subdocuments_large_tokens_simt_kernel_(sz_strings_view_t documents,
                                                                 sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk < large.chunks;
         chunk += large.threads) {
        sz_subdocuments_place_simt_t const place = sz_subdocuments_place_simt_(&large, chunk);
        sz_u8_t const *const text = (sz_u8_t const *)sz_strings_start_simt_(documents, place.document);
        sz_size_t const start = (chunk - place.first) * sz_subdocuments_chunk_bytes_simt_k;
        sz_size_t const end = sz_min_of_two(start + sz_subdocuments_chunk_bytes_simt_k, place.length);
        sz_size_t token = large.tokens[chunk], container = large.opens[chunk];
        sz_u32_t depth = (sz_u32_t)(large.depths[chunk] - large.depths[place.first]);
        sz_u32_t current = depth ? sz_subdocuments_outer_container_simt_k : sz_subdocuments_no_container_simt_k;
        sz_subdocuments_cursor_simt_t cursor;
        sz_u32_t inside;
        // A dropped document's chunks own no tokens to write.
        if (token == large.tokens[chunk + 1]) continue;
        sz_subdocuments_chunk_cursor_simt_(&large, &place, text, chunk, &cursor);
        inside = cursor.in_string != 0;
        for (sz_size_t offset = start; offset < end; offset += 64) {
            sz_subdocuments_load_simt_(&cursor, offset);
            for (sz_u64_t pending = cursor.pending; pending; pending &= pending - 1, ++token) {
                sz_size_t const position = offset + sz_subdocuments_ctz_simt_(pending);
                sz_u8_t const byte = text[position];
                sz_u8_t kind;
                if (byte == '"')
                    kind = inside ? sz_subdocuments_token_quote_end_k : sz_subdocuments_token_quote_k, inside ^= 1;
                else if ((byte | 0x20) == '{') {
                    kind = byte == '{' ? sz_subdocuments_token_object_k : sz_subdocuments_token_array_k;
                    large.openers[container] = (sz_u32_t)token, large.minima[container] = depth++;
                    large.parents[container] = current, large.elements[container] = 0;
                    current = (sz_u32_t)container++;
                }
                else if ((byte | 0x20) == '}') {
                    kind = byte == '}' ? sz_subdocuments_token_object_end_k : sz_subdocuments_token_array_end_k;
                    --depth;
                    if (current < sz_subdocuments_outer_container_simt_k) current = large.parents[current];
                    else current = depth ? sz_subdocuments_outer_container_simt_k : sz_subdocuments_no_container_simt_k;
                }
                else if (byte == ':') kind = sz_subdocuments_token_colon_k;
                else if (byte == ',') kind = sz_subdocuments_token_comma_k;
                else kind = sz_subdocuments_token_scalar_k;
                large.positions[token] = (sz_u32_t)position, large.kinds[token] = kind;
            }
        }
    }
}

/** Builds level @c round of the depth minima, each entry the least of @c fanout entries
 *  of the level below. */
static __global__ void sz_subdocuments_large_minima_simt_kernel_(sz_subdocuments_large_simt_t large) {
    sz_size_t const below = large.minima_offsets[large.round - 1], above = large.minima_offsets[large.round];
    sz_size_t const count = large.minima_offsets[large.round + 1] - above;
    for (sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += large.threads) {
        sz_size_t const first = below + index * sz_subdocuments_minima_fanout_simt_k;
        sz_size_t const last = sz_min_of_two(first + sz_subdocuments_minima_fanout_simt_k, above);
        sz_u32_t least = 0xFFFFFFFFu;
        for (sz_size_t entry = first; entry != last; ++entry) least = sz_min_of_two(least, large.minima[entry]);
        large.minima[above + index] = least;
    }
}

/** Finds the container each chunk starts inside, and the parents its chunk left open, once per
 *  depth, as every container a chunk opens directly inside an earlier chunk's at one depth shares
 *  that parent. A kept document's depths start at zero at its root, so no search passes its root
 *  into an earlier document. */
static __global__ void sz_subdocuments_large_parents_simt_kernel_(sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk < large.chunks;
         chunk += large.threads) {
        sz_subdocuments_place_simt_t const place = sz_subdocuments_place_simt_(&large, chunk);
        sz_size_t const first = large.opens[chunk], last = large.opens[chunk + 1];
        sz_u32_t const depth = (sz_u32_t)(large.depths[chunk] - large.depths[place.first]);
        sz_u32_t cached_depth = 0, cached = sz_subdocuments_no_container_simt_k;
        large.starts[chunk] = depth ? sz_subdocuments_shallower_simt_(&large, first, depth)
                                    : sz_subdocuments_no_container_simt_k;
        for (sz_size_t container = first; container != last; ++container) {
            if (large.parents[container] != sz_subdocuments_outer_container_simt_k) continue;
            if (large.minima[container] != cached_depth)
                cached_depth = large.minima[container],
                cached = sz_subdocuments_shallower_simt_(&large, first, cached_depth);
            large.parents[container] = cached;
        }
    }
}

/** Records where each container closes, and counts the values directly inside it, one atomic per
 *  container a chunk touches. */
static __global__ void sz_subdocuments_large_closes_simt_kernel_(sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk < large.chunks;
         chunk += large.threads) {
        sz_size_t const last = large.tokens[chunk + 1];
        sz_size_t const end = large.tokens[sz_subdocuments_place_simt_(&large, chunk).next];
        sz_size_t container = large.opens[chunk];
        sz_u32_t current = large.starts[chunk], values = 0;
        for (sz_size_t token = large.tokens[chunk]; token != last; ++token) {
            sz_u8_t const kind = large.kinds[token];
            values += sz_subdocuments_token_value_simt_(&large, token, end);
            if (kind > sz_subdocuments_token_array_end_k) continue;
            if (current != sz_subdocuments_no_container_simt_k && values) atomicAdd(&large.elements[current], values);
            values = 0;
            if (kind <= sz_subdocuments_token_array_k) current = (sz_u32_t)container++;
            else if (current != sz_subdocuments_no_container_simt_k)
                large.closers[current] = (sz_u32_t)token, current = large.parents[current];
        }
        if (current != sz_subdocuments_no_container_simt_k && values) atomicAdd(&large.elements[current], values);
    }
}

/** Sums each chunk's values less the values of the containers it closes, so a scan of the sums
 *  counts, before any token, the values of every container still open there. */
static __global__ void sz_subdocuments_large_sums_simt_kernel_(sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk <= large.chunks;
         chunk += large.threads) {
        sz_size_t const last = chunk == large.chunks ? 0 : large.tokens[chunk + 1];
        sz_size_t const end = large.tokens[sz_subdocuments_place_simt_(&large, chunk).next];
        sz_size_t container = large.opens[chunk];
        sz_u32_t current = chunk == large.chunks ? sz_subdocuments_no_container_simt_k : large.starts[chunk];
        sz_u64_t sum = 0;
        for (sz_size_t token = chunk == large.chunks ? 0 : large.tokens[chunk]; token != last; ++token) {
            sz_u8_t const kind = large.kinds[token];
            sum += sz_subdocuments_token_value_simt_(&large, token, end);
            if (kind <= sz_subdocuments_token_array_k) current = (sz_u32_t)container++;
            else if (kind <= sz_subdocuments_token_array_end_k)
                sum -= large.elements[current], current = large.parents[current];
        }
        large.sums[chunk] = sum;
    }
}

/** Records the scanned count just past each container's opening bracket, from which
 *  its elements' positions count. */
static __global__ void sz_subdocuments_large_offsets_simt_kernel_(sz_subdocuments_large_simt_t large) {
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk < large.chunks;
         chunk += large.threads) {
        sz_size_t const last = large.tokens[chunk + 1];
        sz_size_t const end = large.tokens[sz_subdocuments_place_simt_(&large, chunk).next];
        sz_size_t container = large.opens[chunk];
        sz_u32_t current = large.starts[chunk];
        sz_u64_t running = large.sums[chunk];
        for (sz_size_t token = large.tokens[chunk]; token != last; ++token) {
            sz_u8_t const kind = large.kinds[token];
            running += sz_subdocuments_token_value_simt_(&large, token, end);
            if (kind <= sz_subdocuments_token_array_k)
                large.offsets[container] = running, current = (sz_u32_t)container++;
            else if (kind <= sz_subdocuments_token_array_end_k)
                running -= large.elements[current], current = large.parents[current];
        }
    }
}

/** Settles each container's selector, its key's name in an object or its position in an array, and
 *  starts the live sets: the root steps at the root, nothing elsewhere. */
static __global__ void sz_subdocuments_large_selectors_simt_kernel_(sz_subdocuments_engine_t engine,
                                                                    sz_strings_view_t documents,
                                                                    sz_subdocuments_large_simt_t large) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x, words = engine.words_count;
    sz_u8_t *const key = large.keys + thread * large.key_bytes;
    for (sz_size_t container = thread; container < large.containers_count; container += large.threads) {
        sz_u32_t const parent = large.parents[container];
        sz_size_t selector = 0;
        if (parent != sz_subdocuments_no_container_simt_k &&
            large.kinds[large.openers[parent]] == sz_subdocuments_token_object_k) {
            sz_size_t const slot = sz_subdocuments_slot_simt_(&large, large.opens, container);
            sz_u8_t const *const text = (sz_u8_t const *)sz_strings_start_simt_(documents, large.list[2 * slot]);
            selector = sz_subdocuments_member_name_simt_(&engine, &large, text, large.openers[container], key);
        }
        else if (parent != sz_subdocuments_no_container_simt_k)
            selector = (sz_size_t)(large.offsets[container] - 1 - large.offsets[parent]);
        large.selectors[container] = selector;
        for (sz_size_t word = 0; word != words; ++word) {
            sz_u64_t const root = parent == sz_subdocuments_no_container_simt_k ? engine.root_steps[word] : 0;
            large.lives[container * words + word] = root;
            large.lives[(large.containers_count + container) * words + word] = root;
        }
    }
}

/** Gathers into each container the descendant steps live at its ancestors: round zero
 *  takes its parent's, and every later round doubles how far up each one reaches, from one
 *  buffer into the other. */
static __global__ void sz_subdocuments_large_jump_simt_kernel_(sz_subdocuments_engine_t engine,
                                                               sz_subdocuments_large_simt_t large) {
    sz_size_t const words = engine.words_count, containers = large.containers_count;
    sz_size_t const source = (large.round + 1) & 1, target = large.round & 1;
    for (sz_size_t container = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; container < containers;
         container += large.threads) {
        sz_u32_t const jump = large.round ? large.jumps[source * containers + container] : large.parents[container];
        large.jumps[target * containers + container] = large.round && jump != sz_subdocuments_no_container_simt_k
                                                           ? large.jumps[source * containers + jump]
                                                           : jump;
        for (sz_size_t mode = 0; mode != 2; ++mode) {
            sz_u64_t *const gathered = large.carried + ((target * 2 + mode) * containers + container) * words;
            sz_u64_t const *const own = large.carried + ((source * 2 + mode) * containers + container) * words;
            sz_u64_t const *const above = jump == sz_subdocuments_no_container_simt_k
                                              ? (sz_u64_t const *)STRINGZILLA_NULL
                                          : large.round
                                              ? large.carried + ((source * 2 + mode) * containers + jump) * words
                                              : large.lives + (mode * containers + jump) * words;
            for (sz_size_t word = 0; word != words; ++word)
                gathered[word] = (large.round ? own[word] : 0) |
                                 (above ? above[word] & (large.round ? ~0ull : engine.descendant_steps[word]) : 0);
        }
    }
}

/** Advances each container's live sets from its parent's, joined with what its ancestors carry, in
 *  place: a set read before or after its update only grows towards the settled one. */
static __global__ void sz_subdocuments_large_advance_simt_kernel_(sz_subdocuments_engine_t engine,
                                                                  sz_subdocuments_large_simt_t large) {
    sz_size_t const words = engine.words_count;
    for (sz_size_t container = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; container < large.containers_count;
         container += large.threads) {
        sz_u32_t const parent = large.parents[container];
        sz_bool_t element;
        if (parent == sz_subdocuments_no_container_simt_k) continue;
        element = (sz_bool_t)(large.kinds[large.openers[parent]] == sz_subdocuments_token_array_k);
        for (sz_size_t mode = 0; mode != 2; ++mode) {
            sz_u64_t *const live = large.lives + (mode * large.containers_count + container) * words;
            sz_u64_t carry = 0;
            for (sz_size_t word = 0; word != words; ++word)
                live[word] = sz_subdocuments_child_word_simt_(
                    &engine, sz_subdocuments_live_simt_(&large, words, mode, parent),
                    sz_subdocuments_carried_simt_(&large, words, mode, parent), word, &carry, element,
                    large.selectors[container], mode ? large.elements[parent] : 0);
        }
    }
}

/** Walks each chunk's tokens in order, counting into its chunk's slot the matches of every value
 *  and refusing its document at any token the serial walk refuses or when it holds none, or, when
 *  storing, writing the matches from where its document's first match lands. */
static __global__ void sz_subdocuments_large_walk_simt_kernel_(sz_subdocuments_engine_t engine,
                                                               sz_strings_view_t documents,
                                                               sz_subdocuments_round_simt_t round,
                                                               sz_subdocuments_large_simt_t large) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x, words = engine.words_count;
    sz_u8_t *const key = large.keys + thread * large.key_bytes;
    for (sz_size_t chunk = thread; chunk <= large.chunks; chunk += large.threads) {
        sz_subdocuments_place_simt_t const place = sz_subdocuments_place_simt_(&large, chunk);
        sz_u8_t const *const text = (sz_u8_t const *)sz_strings_start_simt_(documents, place.document);
        sz_size_t const begin = large.tokens[place.first], end = large.tokens[place.next];
        sz_size_t const last = chunk == large.chunks ? 0 : large.tokens[chunk + 1];
        sz_size_t container = large.opens[chunk], counted = 0;
        sz_size_t emitted = large.storing
                                ? round.counted[place.document] + large.matches[chunk] - large.matches[place.first]
                                : 0;
        sz_u32_t current = chunk == large.chunks ? sz_subdocuments_no_container_simt_k : large.starts[chunk];
        sz_u32_t malformed = !large.storing && place.slot != large.listed && chunk == place.first && begin == end;
        sz_u64_t running = large.sums[chunk];
        for (sz_size_t token = chunk == large.chunks ? 0 : large.tokens[chunk]; token != last; ++token) {
            sz_u8_t const kind = large.kinds[token];
            sz_bool_t const value = sz_subdocuments_token_value_simt_(&large, token, end);
            if (kind == sz_subdocuments_token_quote_end_k) continue;
            if (!large.storing &&
                (current == sz_subdocuments_no_container_simt_k ||
                 sz_subdocuments_continues_simt_(&engine, sz_subdocuments_live_simt_(&large, words, 0, current),
                                                 sz_subdocuments_carried_simt_(&large, words, 0, current))))
                malformed |= !sz_subdocuments_token_valid_simt_(&engine, &large, text, place.length, begin, end, token,
                                                                current);
            if (value) {
                // A container's live set is settled; any other value's is its
                // parent's advanced into it.
                sz_bool_t const rooted = (sz_bool_t)(current == sz_subdocuments_no_container_simt_k);
                sz_bool_t const element = (sz_bool_t)(!rooted && large.kinds[large.openers[current]] ==
                                                                     sz_subdocuments_token_array_k);
                sz_u64_t const *const parent = rooted ? engine.root_steps
                                                      : sz_subdocuments_live_simt_(&large, words, 1, current);
                sz_u64_t const *const parent_carried = rooted
                                                           ? (sz_u64_t const *)STRINGZILLA_NULL
                                                           : sz_subdocuments_carried_simt_(&large, words, 1, current);
                sz_u64_t const *const own = sz_subdocuments_live_simt_(&large, words, 1, container);
                sz_u64_t const *const own_carried = sz_subdocuments_carried_simt_(&large, words, 1, container);
                sz_size_t selector = element ? (sz_size_t)(running - large.offsets[current])
                                             : sz_subdocuments_no_name_k;
                sz_size_t byte_length = STRINGZILLA_SIZE_MAX;
                sz_u64_t carry = 0;
                if (!rooted && !element && kind > sz_subdocuments_token_array_k &&
                    sz_subdocuments_needs_name_simt_(&engine, parent, parent_carried))
                    selector = sz_subdocuments_member_name_simt_(&engine, &large, text, token, key);
                for (sz_size_t word = 0; word != words; ++word) {
                    sz_u64_t const live = kind <= sz_subdocuments_token_array_k
                                              ? own[word] | (own_carried ? own_carried[word] : 0)
                                          : rooted ? parent[word]
                                                   : sz_subdocuments_child_word_simt_(
                                                         &engine, parent, parent_carried, word, &carry, element,
                                                         selector, element ? large.elements[current] : 0);
                    sz_u64_t reported = live & engine.final_steps[word];
                    if (!large.storing) {
                        counted += (sz_size_t)__popcll(reported);
                        continue;
                    }
                    if (reported && byte_length == STRINGZILLA_SIZE_MAX)
                        byte_length = sz_subdocuments_token_length_simt_(&large, text, place.length, token, container);
                    for (; reported; reported &= reported - 1, ++emitted)
                        if (emitted < round.matches_capacity) {
                            sz_match_t *const match = &round.matches[emitted];
                            match->haystack_index = place.document;
                            match->needle_index = engine.steps_paths[word * 64 + sz_subdocuments_ctz_simt_(reported)];
                            match->byte_offset = large.positions[token], match->byte_length = byte_length;
                        }
                }
            }
            if (kind <= sz_subdocuments_token_array_k) current = (sz_u32_t)container++;
            else if (kind <= sz_subdocuments_token_array_end_k)
                running -= large.elements[current], current = large.parents[current];
            running += value;
        }
        if (!large.storing) large.matches[chunk] = counted;
        if (malformed)
            atomicMin((unsigned long long *)&round.header->first_malformed, (unsigned long long)place.document);
    }
}

/** Writes each listed document's matches, the scanned matches of its chunks, into its count. */
static __global__ void sz_subdocuments_large_finish_simt_kernel_(sz_subdocuments_round_simt_t round,
                                                                 sz_subdocuments_large_simt_t large) {
    for (sz_size_t slot = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; slot < large.listed;
         slot += large.threads) {
        sz_size_t const document = (sz_size_t)large.list[2 * slot];
        sz_size_t const found = (sz_size_t)(large.matches[large.firsts[slot + 1]] - large.matches[large.firsts[slot]]);
        if (round.counted) round.counted[document] = found;
        else round.counts[document * round.counts_stride] = found;
    }
}

#pragma endregion Large Documents

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_SUBDOCUMENTS_SIMT_CUH_
