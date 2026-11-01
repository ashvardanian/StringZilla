/**
 *  @file include/stringzilla/subdocuments/kernels.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief StringZilla's own launch bodies over the walks of <tt>subdocuments/simt.cuh</tt>, which products
 *      never include, as only <tt>subdocuments/cuda.cuh</tt> and <tt>subdocuments/rocm.cuh</tt> launch them.
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_KERNELS_CUH_
#define STRINGZILLA_SUBDOCUMENTS_KERNELS_CUH_

#include "stringzilla/subdocuments/simt.cuh"

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

/*  StringZilla's own kernels and the bodies they share, which read @c blockIdx, declare
 *  @c __shared__, publish through atomics and drive the multi-launch rounds each vendor's host side
 *  sizes; the walks of <tt>subdocuments/simt.cuh</tt> do the walking. */
#pragma region Kernels

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

/**
 *  @brief Walks every document, or every listed one, as walker @p walker of a launch's walkers,
 *      each a thread or a group of lanes: counting, or storing where the counts scanned to.
 *
 *  JSON documents walk on the lanes of @p group, its vendor's lane steps, and XML ones on a thread,
 *  with @p group null.
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
                                                             sz_simt_group_t const *group) {
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
        status = !group ? sz_subdocuments_walk_markup_simt_(engine, text, length, document, stack, round->frames,
                                                            round->frame_words, round->key_bytes, matches,
                                                            matches_capacity, &emitted)
                        : sz_subdocuments_walk_simt_(engine, text, length, document, stack, round->frames,
                                                     round->frame_words, key, round->key_bytes, matches,
                                                     matches_capacity, &emitted, group);
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

/** Marks which documents a counting walk found a match in, a lane each, every group of 32 lanes
 *  storing what @p ballot gathers as four bytes of @p contained. */
STRINGZILLA_DEVICE void sz_subdocuments_contains_simt_(sz_size_t const *counted, sz_u1x8_t *contained, sz_size_t count,
                                                       sz_u32_t (*ballot)(int)) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x, lanes = sz_size_divide_round_up(count, 32) * 32;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < lanes; index += stride) {
        sz_u32_t const bits = ballot(index < count && counted[index] != 0);
        if (index % 32 == 0) sz_u1x8_store_simt_(contained, count, index, bits);
    }
}

/** Lowers the header's @c large_bytes to one of @c large_shares of the batch's bytes once any
 *  document passes both that share and the round's @c large_bytes, so the documents past their
 *  share walk over chunks only in a batch that takes the cooperative walk anyway. */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_route_simt_kernel_(sz_strings_view_t documents, sz_subdocuments_round_simt_t round) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_walk_simt_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
                                      sz_subdocuments_round_simt_t round) {
    sz_subdocuments_walk_documents_simt_(&engine, &documents, &round, (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x,
                                         0, 1, (sz_simt_group_t const *)STRINGZILLA_NULL);
}

/** Scans one ticket's own tile of @p values in place, seeded by the totals the tiles before
 *  it chained through @p chain, so the whole scan is one launch, as
 *  @c sz_regex_scan_simt_kernel_ does. */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_scan_simt_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile,
                                      sz_size_t *chain) {
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

/** Counts each listed document's chunks into its slot, none for one at or past @c bound,
 *  and clears its verdict. */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_lengths_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_summary_simt_kernel_(sz_strings_view_t documents, sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_reduce_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_carry_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_apply_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
 *  depth dips and rises, marking its document malformed for a <tt>/</tt> outside strings. */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_count_simt_kernel_(sz_strings_view_t documents, sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_check_simt_kernel_(sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_drop_simt_kernel_(sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_tokens_simt_kernel_(sz_strings_view_t documents, sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_minima_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_parents_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_closes_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_sums_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_offsets_simt_kernel_(sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_selectors_simt_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_jump_simt_kernel_(sz_subdocuments_engine_t engine, sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_advance_simt_kernel_(sz_subdocuments_engine_t engine, sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_walk_simt_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
                                            sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
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
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_finish_simt_kernel_(sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
    for (sz_size_t slot = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; slot < large.listed;
         slot += large.threads) {
        sz_size_t const document = (sz_size_t)large.list[2 * slot];
        sz_size_t const found = (sz_size_t)(large.matches[large.firsts[slot + 1]] - large.matches[large.firsts[slot]]);
        if (round.counted) round.counted[document] = found;
        else round.counts[document * round.counts_stride] = found;
    }
}

#pragma endregion Kernels

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_SUBDOCUMENTS_KERNELS_CUH_
