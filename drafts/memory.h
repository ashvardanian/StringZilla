

SZ_PUBLIC void sz_copy_sve(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    sz_size_t vec_len = svcntb(); // Vector length in bytes

    // Arm Neoverse V2 cores in Graviton 4, for example, come with 256 KB of L1 data cache per core,
    // and 8 MB of L2 cache per core. Moreover, the L1 cache is fully associative.
    // With two strings, we may consider the overal workload huge, if each exceeds 1 MB in length.
    int const is_huge = length >= 1ull * 1024ull * 1024ull;

    // When the buffer is small, there isn't much to innovate.
    if (length <= vec_len) {
        // Small buffer case: use mask to handle small writes
        svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)length);
        svuint8_t data = svld1_u8(mask, (sz_u8_t *)source);
        svst1_u8(mask, (sz_u8_t *)target, data);
    }
    // For gigantic buffers exceeding L1 cache, use a sophisticated approach with non-temporal operations.
    // We load 5 consecutive (potentially misaligned) vectors, use svext to extract 4 aligned vectors,
    // then store them with non-temporal writes. This approach:
    // 1. Minimizes cache pollution for large transfers
    // 2. Achieves aligned stores for maximum throughput
    // 3. Amortizes the cost of misalignment correction across multiple stores
    else if (is_huge) {
        // Align target to vector boundary
        sz_size_t head_length = (vec_len - ((sz_size_t)target % vec_len)) % vec_len;
        sz_size_t tail_length = (sz_size_t)(target + length) % vec_len;
        sz_size_t body_length = length - head_length - tail_length;

        // Handle unaligned head
        if (head_length) {
            svbool_t head_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)head_length);
            svuint8_t head_data = svld1_u8(head_mask, (sz_u8_t *)source);
            svst1_u8(head_mask, (sz_u8_t *)target, head_data);
            source += head_length;
            target += head_length;
        }

        // Calculate misalignment of source relative to vector boundary
        sz_size_t source_misalignment = (sz_size_t)source % vec_len;

        // Main loop: load 5 vectors non-temporally, realign to 4, store 4 aligned non-temporally
        if (source_misalignment == 0) {
            // Fast path: source is also aligned, no realignment needed
            for (; body_length >= vec_len * 4;
                 source += vec_len * 4, target += vec_len * 4, body_length -= vec_len * 4) {
                svuint8_t v0 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)source);
                svuint8_t v1 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)(source + vec_len));
                svuint8_t v2 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)(source + vec_len * 2));
                svuint8_t v3 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)(source + vec_len * 3));
                svstnt1_u8(svptrue_b8(), (sz_u8_t *)target, v0);
                svstnt1_u8(svptrue_b8(), (sz_u8_t *)(target + vec_len), v1);
                svstnt1_u8(svptrue_b8(), (sz_u8_t *)(target + vec_len * 2), v2);
                svstnt1_u8(svptrue_b8(), (sz_u8_t *)(target + vec_len * 3), v3);
            }
        }
        else {
            // Slow path: source is misaligned
            // For pure SVE (without SVE2), we use svtbl on single vectors combined with bitwise OR
            // to achieve the same effect as svtbl2 would give us.
            //
            // Strategy: For each aligned output, we need bytes from two consecutive input vectors.
            // We'll create two sets of indices:
            // - indices_low: extracts bytes [misalignment..vec_len-1] from the first vector
            // - indices_high: extracts bytes [0..misalignment-1] from the second vector
            // Then OR them together to get the final aligned result.

            svuint8_t indices_low =
                svindex_u8(source_misalignment, 1);    // [misalignment, misalignment+1, ..., vec_len-1, wrap...]
            svuint8_t indices_high = svindex_u8(0, 1); // [0, 1, 2, ..., vec_len-1]

            // Create predicate mask for which bytes come from low vector
            // Bytes at positions [0..vec_len-misalignment-1] come from low vector (tail of first input)
            // Bytes at positions [vec_len-misalignment..vec_len-1] come from high vector (head of second input)
            svbool_t use_low = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)(vec_len - source_misalignment));

            for (; body_length >= vec_len * 4;
                 source += vec_len * 4, target += vec_len * 4, body_length -= vec_len * 4) {
                // Load 5 consecutive vectors to cover 4 aligned vectors worth of data
                svuint8_t v0 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)source);
                svuint8_t v1 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)(source + vec_len));
                svuint8_t v2 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)(source + vec_len * 2));
                svuint8_t v3 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)(source + vec_len * 3));
                svuint8_t v4 = svldnt1_u8(svptrue_b8(), (sz_u8_t const *)(source + vec_len * 4));

                // Extract aligned chunks using svtbl on single vectors and blend
                // For each output, we table-lookup from two inputs and blend based on the mask
                svuint8_t low0 = svtbl_u8(v0, indices_low);
                svuint8_t high0 = svtbl_u8(v1, indices_high);
                svuint8_t aligned0 = svsel_u8(use_low, low0, high0);

                svuint8_t low1 = svtbl_u8(v1, indices_low);
                svuint8_t high1 = svtbl_u8(v2, indices_high);
                svuint8_t aligned1 = svsel_u8(use_low, low1, high1);

                svuint8_t low2 = svtbl_u8(v2, indices_low);
                svuint8_t high2 = svtbl_u8(v3, indices_high);
                svuint8_t aligned2 = svsel_u8(use_low, low2, high2);

                svuint8_t low3 = svtbl_u8(v3, indices_low);
                svuint8_t high3 = svtbl_u8(v4, indices_high);
                svuint8_t aligned3 = svsel_u8(use_low, low3, high3);

                // Store 4 aligned vectors non-temporally
                svstnt1_u8(svptrue_b8(), (sz_u8_t *)target, aligned0);
                svstnt1_u8(svptrue_b8(), (sz_u8_t *)(target + vec_len), aligned1);
                svstnt1_u8(svptrue_b8(), (sz_u8_t *)(target + vec_len * 2), aligned2);
                svstnt1_u8(svptrue_b8(), (sz_u8_t *)(target + vec_len * 3), aligned3);
            }
        }

        // Handle remaining body with regular loads/stores
        for (; body_length >= vec_len; source += vec_len, target += vec_len, body_length -= vec_len) {
            svuint8_t data = svld1_u8(svptrue_b8(), (sz_u8_t *)source);
            svst1_u8(svptrue_b8(), (sz_u8_t *)target, data);
        }

        // Handle unaligned tail
        if (tail_length) {
            svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)tail_length);
            svuint8_t tail_data = svld1_u8(tail_mask, (sz_u8_t *)source);
            svst1_u8(tail_mask, (sz_u8_t *)target, tail_data);
        }
    }
    // For medium-sized buffers, use bidirectional traversal without non-temporal operations
    else {
        // Calculating head, body, and tail sizes depends on the `vec_len`,
        // but it's runtime constant, and the modulo operation is expensive!
        // Instead we use the fact, that it's always a multiple of 128 bits or 16 bytes.
        sz_size_t head_length = 16 - ((sz_size_t)target % 16);
        sz_size_t tail_length = (sz_size_t)(target + length) % 16;
        sz_size_t body_length = length - head_length - tail_length;

        // Handle unaligned parts
        svbool_t head_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)head_length);
        svuint8_t head_data = svld1_u8(head_mask, (sz_u8_t *)source);
        svst1_u8(head_mask, (sz_u8_t *)target, head_data);
        svbool_t tail_mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)tail_length);
        svuint8_t tail_data = svld1_u8(tail_mask, (sz_u8_t *)source + head_length + body_length);
        svst1_u8(tail_mask, (sz_u8_t *)target + head_length + body_length, tail_data);
        target += head_length;
        source += head_length;

        // Aligned body loop, walking in two directions
        for (; body_length >= vec_len * 2; target += vec_len, source += vec_len, body_length -= vec_len * 2) {
            svuint8_t forward_data = svld1_u8(svptrue_b8(), (sz_u8_t *)source);
            svuint8_t backward_data = svld1_u8(svptrue_b8(), (sz_u8_t *)source + body_length - vec_len);
            svst1_u8(svptrue_b8(), (sz_u8_t *)target, forward_data);
            svst1_u8(svptrue_b8(), (sz_u8_t *)target + body_length - vec_len, backward_data);
        }
        // Up to (vec_len * 2 - 1) bytes of data may be left in the body,
        // so we can unroll the last two optional loop iterations.
        if (body_length > vec_len) {
            svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)body_length);
            svuint8_t data = svld1_u8(mask, (sz_u8_t *)source);
            svst1_u8(mask, (sz_u8_t *)target, data);
            body_length -= vec_len;
            source += vec_len;
            target += vec_len;
        }
        if (body_length) {
            svbool_t mask = svwhilelt_b8((sz_u64_t)0ull, (sz_u64_t)body_length);
            svuint8_t data = svld1_u8(mask, (sz_u8_t *)source);
            svst1_u8(mask, (sz_u8_t *)target, data);
        }
    }
}
