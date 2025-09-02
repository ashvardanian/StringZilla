

/**
 *  @brief  Perform a compare–exchange (compare–swap) on two 8‑lane vectors,
 *          updating both the keys and their associated offsets.
 *
 *  @param  keys         Pointer to a __m512i containing 8 keys.
 *  @param  offsets      Pointer to a __m512i containing 8 offsets.
 *  @param  perm         Permutation vector (as __m512i) that maps each lane
 *                       to its “partner” in the compare–exchange.
 *  @param  fixed_mask   An 8‑bit immediate mask (as __mmask8) that indicates,
 *                       for each pair, which lane is designated as the “upper”
 *                       element. For that lane the max is chosen, while for the
 *                       complementary (“lower”) lane the min is chosen.
 *
 *  This helper function “mirrors” the scalar operation:
 *
 *      if (keys[i] > keys[j]) {
 *          swap(keys[i], keys[j]);
 *          swap(offsets[i], offsets[j]);
 *      }
 *
 *  for each pair (i,j) defined by the permutation vector.
 *
 *  The keys are updated by computing the unsigned min and max between each
 *  element and its partner, and then blending them into the designated positions
 *  using the fixed_mask. In order to update the offsets in a stable manner,
 *  we first compute the partner offsets (using the same permutation), then for each
 *  pair we choose:
 *
 *      - For the lane designated as lower (mask bit = 0):
 *            if (orig_key <= partner_key) then keep self’s offset,
 *            else take the partner’s offset.
 *
 *      - For the lane designated as upper (mask bit = 1):
 *            if (orig_key > partner_key) then keep self’s offset,
 *            else take the partner’s offset.
 *
 *  This ensures that if keys are equal (thus stable), no swap is done.
 */
SZ_INTERNAL void cswap_argsort_avx512(__m512i *pgrams, __m512i *offsets, __m512i perm, __mmask8 fixed_mask) {
    // Save original pgrams and offsets for condition computation.
    __m512i orig_pgrams = *pgrams;
    __m512i orig_offsets = *offsets;

    // Compute partner vectors using the permutation vector.
    __m512i partner_pgrams = _mm512_permutexvar_epi64(perm, orig_pgrams);
    __m512i partner_offsets = _mm512_permutexvar_epi64(perm, orig_offsets);

    // Compute new pgrams: for each pair, choose the unsigned min for the lower lane
    // and the unsigned max for the upper lane.
    __m512i pgrams_min = _mm512_min_epu64(orig_pgrams, partner_pgrams);
    __m512i pgrams_max = _mm512_max_epu64(orig_pgrams, partner_pgrams);
    *pgrams = _mm512_mask_blend_epi64(fixed_mask, pgrams_min, pgrams_max);

    // For offsets, we want to mimic the swap decision used for pgrams.
    // For each pair (i,j) (with i < j), if orig_pgrams[i] <= partner_pgrams[i] then
    // the lower key came from the current lane (i) and the upper from the partner (j);
    // otherwise the lower key came from the partner.
    __mmask8 lower_cond =
        _mm512_cmp_epu64_mask(orig_pgrams, partner_pgrams, _MM_CMPINT_LE); // true if no swap needed for lower lane.
    __mmask8 upper_cond =
        _mm512_cmp_epu64_mask(orig_pgrams, partner_pgrams, _MM_CMPINT_GT); // true if swap needed for upper lane.

    // Compute offsets for lower positions (fixed_mask bit = 0):
    //   If lower_cond is true, then the current lane’s offset is correct;
    //   otherwise, use the partner’s offset.
    __m512i offsets_lower = _mm512_mask_blend_epi64(lower_cond, partner_offsets, orig_offsets);

    // Compute offsets for upper positions (fixed_mask bit = 1):
    //   If upper_cond is true, then keep the current lane’s offset;
    //   otherwise, use the partner’s offset.
    __m512i offsets_upper = _mm512_mask_blend_epi64(upper_cond, orig_offsets, partner_offsets);

    // Combine the two sets: for lanes designated as lower (mask bit = 0) use offsets_lower;
    // for lanes designated as upper (mask bit = 1) use offsets_upper.
    *offsets = _mm512_mask_blend_epi64(fixed_mask, offsets_lower, offsets_upper);

    // Validate the sorting network.
    if (SZ_DEBUG) {
        sz_pgram_t pgrams_array[8];
        sz_sorted_idx_t offsets_array[8];
        _mm512_storeu_si512(pgrams_array, *pgrams);
        _mm512_storeu_si512(offsets_array, *offsets);
        for (sz_size_t i = 1; i < 8; ++i)
            sz_assert_(pgrams_array[i - 1] <= pgrams_array[i] &&
                       "The sorting network must sort the pgrams in ascending order.");
    }
}

SZ_PUBLIC void sz_sequence_argsort_ice_recursively_(                    //
    sz_sequence_t const *const collection,                              //
    sz_pgram_t *const global_pgrams, sz_size_t *const global_order,     //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence, //
    sz_size_t const start_character) {

    // Prepare the new range of windows
    sz_sequence_argsort_serial_export_next_pgrams_(collection, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character);

    // We can implement a form of a Radix sort here, that will count the number of elements with
    // a certain bit set. The naive approach may require too many loops over data. A more "vectorized"
    // approach would be to maintain a histogram for several bits at once. For 4 bits we will
    // need 2^4 = 16 counters.
    sz_size_t histogram[16] = {0};
    for (sz_size_t byte_in_window = 0; byte_in_window != sizeof(sz_pgram_t); ++byte_in_window) {
        // First sort based on the low nibble of each byte.
        for (sz_size_t i = start_in_sequence; i < end_in_sequence; ++i) {
            sz_size_t const byte = (global_pgrams[i] >> (byte_in_window * 8)) & 0xFF;
            ++histogram[byte];
        }
        sz_size_t offset = start_in_sequence;
        for (sz_size_t i = 0; i != 16; ++i) {
            sz_size_t const count = histogram[i];
            histogram[i] = offset;
            offset += count;
        }
        for (sz_size_t i = start_in_sequence; i < end_in_sequence; ++i) {
            sz_size_t const byte = (global_pgrams[i] >> (byte_in_window * 8)) & 0xFF;
            global_order[histogram[byte]] = i;
            ++histogram[byte];
        }
    }
}
