/**
 *  @file include/stringzilla/strings/icelake.h
 *  @author Ash Vardanian
 *  @date October 11, 2024
 *  @brief AVX-512 VBMI backend for collections of strings on Ice Lake and newer x86 CPUs.
 *
 *  @sa include/stringzilla/strings.h
 */
#ifndef STRINGZILLA_STRINGS_ICELAKE_H_
#define STRINGZILLA_STRINGS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/find/icelake.h" // `sz_find_byteset_icelake_`
#include "stringzilla/utf8_norm.h"    // `sz_utf8_norm_icelake`, `sz_utf8_find_denormalized_icelake`
#include "stringzilla/strings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_TARGET_ICELAKE
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(                                                                                         \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,popcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                 \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,popcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "popcnt")
#endif

STRINGZILLA_INLINE void sz_strings_lookup_bytes_icelake_(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                                         char const lut[sz_at_least_(256)]) {
    sz_assert_no_overlap_(target, length, source, length);

    // Loading the table into registers only pays off once at least 3 cache lines are mapped.
    if (length <= 128) {
        sz_u8_t const *lut_u8 = (sz_u8_t const *)lut;
        sz_u8_t const *source_u8 = (sz_u8_t const *)source;
        sz_u8_t *target_u8 = (sz_u8_t *)target;
        for (sz_size_t byte_index = 0; byte_index < length; ++byte_index)
            target_u8[byte_index] = lut_u8[source_u8[byte_index]];
        return;
    }

    // Masked stores cover the unaligned head and tail, so every body store is cache-line aligned.
    sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
    sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
    __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
    __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);

    // We use VPERMI2B (`_mm512_permutex2var_epi8`) to perform 256-entry lookups efficiently.
    // VPERMI2B uses bit 6 of each index to select between two 64-byte tables, allowing us to
    // cover 128 entries per instruction (2 instructions for all 256 entries).
    //
    // For the high-bit (bit 7) selection, we use VPMOVB2M (`_mm512_movepi8_mask`) which extracts
    // the sign bit of each byte directly to a mask register. This goes to port 0 on Intel,
    // avoiding the port 5 bottleneck that VPTESTMB would cause.
    sz_u512_vec_t lut_0_to_63_vec, lut_64_to_127_vec, lut_128_to_191_vec, lut_192_to_255_vec;
    lut_0_to_63_vec.zmm = _mm512_loadu_si512((lut));
    lut_64_to_127_vec.zmm = _mm512_loadu_si512((lut + 64));
    lut_128_to_191_vec.zmm = _mm512_loadu_si512((lut + 128));
    lut_192_to_255_vec.zmm = _mm512_loadu_si512((lut + 192));

    __mmask64 high_bit_mask_m64;
    sz_u512_vec_t source_vec, low_half_vec, high_half_vec, result_vec;

    // Handling the head.
    if (head_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(head_mask_m64, source);
        // VPERMI2B: bit 6 selects between the two tables, bits 0-5 index within each
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        // VPMOVB2M moves bit 7 of each byte into a mask on port 0, sparing the busy port 5
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_mask_storeu_epi8(target, head_mask_m64, result_vec.zmm);
        source += head_length, target += head_length, length -= head_length;
    }

    // The body in 64-byte chunks, aligned to cache lines of `target`.
    while (length >= 64) {
        source_vec.zmm = _mm512_loadu_si512(source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_store_si512(target, result_vec.zmm); //! Aligned store, our main weapon!
        source += 64, target += 64, length -= 64;
    }

    // Handling the tail.
    if (tail_length) {
        source_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask_m64, source);
        low_half_vec.zmm = _mm512_permutex2var_epi8(lut_0_to_63_vec.zmm, source_vec.zmm, lut_64_to_127_vec.zmm);
        high_half_vec.zmm = _mm512_permutex2var_epi8(lut_128_to_191_vec.zmm, source_vec.zmm, lut_192_to_255_vec.zmm);
        high_bit_mask_m64 = _mm512_movepi8_mask(source_vec.zmm);
        result_vec.zmm = _mm512_mask_blend_epi8(high_bit_mask_m64, low_half_vec.zmm, high_half_vec.zmm);
        _mm512_mask_storeu_epi8(target, tail_mask_m64, result_vec.zmm);
        source += tail_length, target += tail_length, length -= tail_length;
    }
}

/** The bytes of @p text_u8x64 in the byteset @p set_u8x64 holds twice over, as bits: @c vpermb
 *  reads only the low 6 bits of an index, so `c >> 3` picks the set's byte and @c c its bit. */
STRINGZILLA_INLINE sz_u64_t sz_strings_classify_icelake_(__m512i text_u8x64, __m512i set_u8x64) {
    __m512i const bits_u8x64 = _mm512_set1_epi64((long long)0x8040201008040201ull);
    __m512i const bytes_u8x64 = _mm512_permutexvar_epi8(_mm512_srli_epi16(text_u8x64, 3), set_u8x64);
    return _cvtmask64_u64(_mm512_test_epi8_mask(bytes_u8x64, _mm512_permutexvar_epi8(text_u8x64, bits_u8x64)));
}

/** The bytes of line breaks among the 64 at @p source into @p removed, and the first byte of each
 *  break into @p ends. @p previous holds the CR, NEL and LS-or-PS masks of the block before and
 *  takes this one's; the 2 bytes after the block must be readable, as longer breaks look ahead. */
STRINGZILLA_INLINE void sz_strings_lines_classify_icelake_(sz_u8_t const *source, sz_u64_t previous[3],
                                                           sz_u64_t *removed, sz_u64_t *ends) {
    __m512i const text_u8x64 = _mm512_loadu_si512(source);
    // LF, VT, FF and CR run from 0x0A, and FS, GS and RS from 0x1C.
    sz_u64_t const singles = _cvtmask64_u64(
        _kor_mask64(_mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, _mm512_set1_epi8(0x0A)), _mm512_set1_epi8(4)),
                    _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, _mm512_set1_epi8(0x1C)), _mm512_set1_epi8(3))));
    sz_u64_t const returns = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8('\r')));
    sz_u64_t const feeds = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8('\n')));
    sz_u64_t const feeds_after_returns = feeds & (returns << 1 | previous[0] >> 63);
    sz_u64_t nels = 0, separators = 0;
    // NEL starts with C2 and LS and PS with E2, so blocks below C2 hold no multi-byte break.
    if (_cvtmask64_u64(_mm512_cmpge_epu8_mask(text_u8x64, _mm512_set1_epi8((char)0xC2)))) {
        __m512i const next1_u8x64 = _mm512_loadu_si512(source + 1), next2_u8x64 = _mm512_loadu_si512(source + 2);
        nels = _cvtmask64_u64(_kand_mask64(_mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xC2)),
                                           _mm512_cmpeq_epi8_mask(next1_u8x64, _mm512_set1_epi8((char)0x85))));
        // LS and PS end in A8 and A9, which differ in the lowest bit alone.
        separators = _cvtmask64_u64(_kand_mask64(
            _kand_mask64(_mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE2)),
                         _mm512_cmpeq_epi8_mask(next1_u8x64, _mm512_set1_epi8((char)0x80))),
            _mm512_cmpeq_epi8_mask(_mm512_or_si512(next2_u8x64, _mm512_set1_epi8(1)), _mm512_set1_epi8((char)0xA9))));
    }
    sz_u64_t const carried = previous[1] >> 63 | previous[2] >> 62 | previous[2] >> 63;
    *removed = singles | nels | nels << 1 | separators | separators << 1 | separators << 2 | carried;
    *ends = (singles & ~feeds_after_returns) | nels | separators;
    previous[0] = returns, previous[1] = nels, previous[2] = separators;
}

/** Appends the @p lanes bytes at @p source but its @p removed ones at @p target with one exact
 *  store, and sets the end offset, @p written plus the bytes kept before it, of each part a lane of
 *  @p ends ends. A removed lane ending no part directly follows one that does, so rewriting its
 *  slot is harmless. */
STRINGZILLA_INLINE void sz_strings_split_emit_icelake_(sz_u8_t const *source, sz_size_t lanes, sz_u64_t removed,
                                                       sz_u64_t ends, sz_u8_t *target, sz_size_t written,
                                                       sz_strings_span_t *output, sz_size_t *ended) {
    sz_u64_t const kept = ~removed & _cvtmask64_u64(sz_u64_mask_until_(lanes));
    _mm512_mask_storeu_epi8(target, sz_u64_mask_until_((sz_size_t)_mm_popcnt_u64(kept)),
                            _mm512_maskz_compress_epi8(_cvtu64_mask64(kept), _mm512_loadu_si512(source)));
    sz_size_t skipped = 0;
    for (sz_u64_t rest = removed; rest; rest &= rest - 1, ++skipped) {
        sz_size_t const lane = (sz_size_t)_tzcnt_u64(rest);
        *ended += ends >> lane & 1;
        sz_strings_offset_set_(output, *ended, written + lane - skipped);
    }
}

/** Copies @p length bytes of @p source to @p target: whole 64-byte blocks, then one masked. */
STRINGZILLA_INLINE void sz_strings_join_piece_icelake_(sz_u8_t *target, sz_u8_t const *source, sz_size_t length) {
    for (; length > 64; target += 64, source += 64, length -= 64)
        _mm512_storeu_si512(target, _mm512_loadu_si512(source));
    __mmask64 const tail_m64 = sz_u64_mask_until_(length);
    _mm512_mask_storeu_epi8(target, tail_m64, _mm512_maskz_loadu_epi8(tail_m64, source));
}

/** The Ice Lake stages. */
STRINGZILLA_INLINE sz_strings_walks_t sz_strings_walks_icelake_(void) {
    sz_strings_walks_t walks;
    walks.find_byteset = &sz_find_byteset_icelake_;
    walks.lookup_bytes = &sz_strings_lookup_bytes_icelake_;
    return walks;
}

/** The bytes escaping adds for each of the 64 bytes of @p text_u8x64, from 0 to 5: JSON, HTML or
 *  URL escapes. */
STRINGZILLA_INLINE __m512i sz_strings_escape_extras_icelake_(sz_escape_kind_t kind, __m512i text_u8x64) {
    // Every byte outside `A-Z a-z 0-9 - . _ ~`
    static sz_byteset_t const url_set = {{0xFC009FFFFFFFFFFFull, 0xB800000178000001ull, ~0ull, ~0ull}};
    // `\u00XX` for the controls, but `\b`, `\t`, `\n`, `\f` and `\r`
    static sz_u8_t const json_controls[32] = {5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 5, 1, 1, 5, 5,
                                              5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    // `&quot;` at 0x22, `&amp;` and `&#39;` at 0x26 and 0x27, `&lt;` and `&gt;` at 0x3C and 0x3E
    static sz_u8_t const html_extras[64] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                                            0, 0, 5, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, //
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0};
    if (kind == sz_escape_url_k) {
        __m512i const set_u8x64 = _mm512_broadcast_i64x4(_mm256_loadu_si256((__m256i const *)url_set._u8s));
        return _mm512_maskz_mov_epi8(_cvtu64_mask64(sz_strings_classify_icelake_(text_u8x64, set_u8x64)),
                                     _mm512_set1_epi8(2));
    }
    if (kind == sz_escape_html_k)
        return _mm512_maskz_permutexvar_epi8(_mm512_cmplt_epu8_mask(text_u8x64, _mm512_set1_epi8(64)), text_u8x64,
                                             _mm512_loadu_si512(html_extras));
    __mmask64 const controls_m64 = _mm512_cmplt_epu8_mask(text_u8x64, _mm512_set1_epi8(32));
    __mmask64 const quoted_m64 = _kor_mask64(_mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8('"')),
                                             _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8('\\')));
    __m512i const controls_u8x64 = _mm512_broadcast_i64x4(_mm256_loadu_si256((__m256i const *)json_controls));
    return _mm512_mask_mov_epi8(_mm512_maskz_permutexvar_epi8(controls_m64, text_u8x64, controls_u8x64), quoted_m64,
                                _mm512_set1_epi8(1));
}

/** The sum of the bytes of @p extras_u8x64 in the lanes of @p lanes_m64. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_sum_icelake_(__m512i extras_u8x64, __mmask64 lanes_m64) {
    __m512i const lanes_u8x64 = _mm512_maskz_mov_epi8(lanes_m64, extras_u8x64);
    return (sz_size_t)_mm512_reduce_add_epi64(_mm512_sad_epu8(lanes_u8x64, _mm512_setzero_si512()));
}

/** Percent-encodes the @p lanes bytes of @p text_u8x16, those of @p escaped as `%` and the digits
 *  of @p high_u8x16 and @p low_u8x16, into @p target exactly, returning the bytes written. Each
 *  byte takes a 4-byte slot of itself and its digits, and keeps 1 or 3 of them. */
STRINGZILLA_INLINE sz_size_t sz_strings_escape_url_slots_icelake_(__m128i text_u8x16, __m128i high_u8x16,
                                                                  __m128i low_u8x16, sz_u64_t escaped, sz_size_t lanes,
                                                                  sz_u8_t *target) {
    __m128i const first_u8x16 = _mm_mask_blend_epi8((__mmask16)escaped, text_u8x16, _mm_set1_epi8('%'));
    __m512i const slots_u32x16 = _mm512_or_si512(
        _mm512_cvtepu8_epi32(first_u8x16), _mm512_or_si512(_mm512_slli_epi32(_mm512_cvtepu8_epi32(high_u8x16), 8),
                                                           _mm512_slli_epi32(_mm512_cvtepu8_epi32(low_u8x16), 16)));
    sz_u64_t const kept = _pdep_u64(_cvtmask64_u64(sz_u64_mask_until_(lanes)), 0x1111111111111111ull) |
                          _pdep_u64(escaped, 0x2222222222222222ull) | _pdep_u64(escaped, 0x4444444444444444ull);
    sz_size_t const written = (sz_size_t)_mm_popcnt_u64(kept);
    _mm512_mask_storeu_epi8(target, sz_u64_mask_until_(written),
                            _mm512_maskz_compress_epi8(_cvtu64_mask64(kept), slots_u32x16));
    return written;
}

/** Percent-encodes the @p lanes bytes of @p text_u8x64, those of @p escaped as `%XY`, into
 *  @p target exactly, 16 bytes at a time. */
STRINGZILLA_INLINE void sz_strings_escape_url_emit_icelake_(__m512i text_u8x64, sz_u64_t escaped, sz_size_t lanes,
                                                            sz_u8_t *target) {
    static sz_u8_t const digits[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
    __m512i const digits_u8x64 = _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i const *)digits));
    __m512i const high_u8x64 = _mm512_permutexvar_epi8(
        _mm512_and_si512(_mm512_srli_epi16(text_u8x64, 4), _mm512_set1_epi8(15)), digits_u8x64);
    __m512i const low_u8x64 = _mm512_permutexvar_epi8(_mm512_and_si512(text_u8x64, _mm512_set1_epi8(15)), digits_u8x64);
    sz_size_t const lanes_0 = sz_min_of_two(lanes, (sz_size_t)16);
    sz_size_t const lanes_1 = sz_min_of_two(lanes - lanes_0, (sz_size_t)16);
    sz_size_t const lanes_2 = sz_min_of_two(lanes - lanes_0 - lanes_1, (sz_size_t)16);
    sz_size_t const lanes_3 = lanes - lanes_0 - lanes_1 - lanes_2;
    target += sz_strings_escape_url_slots_icelake_(
        _mm512_extracti32x4_epi32(text_u8x64, 0), _mm512_extracti32x4_epi32(high_u8x64, 0),
        _mm512_extracti32x4_epi32(low_u8x64, 0), escaped & 0xFFFF, lanes_0, target);
    target += sz_strings_escape_url_slots_icelake_(
        _mm512_extracti32x4_epi32(text_u8x64, 1), _mm512_extracti32x4_epi32(high_u8x64, 1),
        _mm512_extracti32x4_epi32(low_u8x64, 1), escaped >> 16 & 0xFFFF, lanes_1, target);
    target += sz_strings_escape_url_slots_icelake_(
        _mm512_extracti32x4_epi32(text_u8x64, 2), _mm512_extracti32x4_epi32(high_u8x64, 2),
        _mm512_extracti32x4_epi32(low_u8x64, 2), escaped >> 32 & 0xFFFF, lanes_2, target);
    sz_strings_escape_url_slots_icelake_(_mm512_extracti32x4_epi32(text_u8x64, 3),
                                         _mm512_extracti32x4_epi32(high_u8x64, 3),
                                         _mm512_extracti32x4_epi32(low_u8x64, 3), escaped >> 48, lanes_3, target);
}

/** Escapes the @p lanes bytes of @p text_u8x64 for JSON or HTML into @p target exactly, those of
 *  @p escaped as their sequences: compressing the lanes from an escape on shifts the escape and the
 *  bytes after it down to lane 0, so every byte comes from the register. */
STRINGZILLA_INLINE void sz_strings_escape_emit_icelake_(sz_escape_kind_t kind, __m512i text_u8x64, sz_u64_t escaped,
                                                        sz_size_t lanes, sz_u8_t *target) {
    // The bytes before the first escape, or all of them.
    _mm512_mask_storeu_epi8(target, sz_u64_mask_until_(sz_min_of_two((sz_size_t)_tzcnt_u64(escaped), lanes)),
                            text_u8x64);
    sz_size_t extra = 0;
    for (sz_u64_t rest = escaped; rest; rest &= rest - 1) {
        sz_size_t const lane = (sz_size_t)_tzcnt_u64(rest);
        sz_u64_t const later = rest & (rest - 1);
        sz_size_t const next = later ? (sz_size_t)_tzcnt_u64(later) : lanes;
        __m512i const shifted_u8x64 = _mm512_maskz_compress_epi8(
            _kandn_mask64(sz_u64_mask_until_(lane), sz_u64_mask_until_(lanes)), text_u8x64);
        sz_u8_t *const output = target + lane + extra;
        sz_size_t const before = extra;
        sz_u64_t const sequence = sz_strings_escape_sequence_(
            kind, (sz_u8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(shifted_u8x64)), &extra);
        sz_size_t const sequence_length = extra - before + 1;
        _mm_mask_storeu_epi8(output, sz_u16_mask_until_(sequence_length), _mm_cvtsi64_si128((long long)sequence));
        // The bytes up to the next escape follow, lane 0 of the shifted ones being the escape.
        _mm512_mask_storeu_epi8(output + sequence_length - 1, _kshiftli_mask64(sz_u64_mask_until_(next - lane - 1), 1),
                                shifted_u8x64);
    }
}

/**
 *  @brief Escapes the @p count strings of a tape, read in its own order as one run of bytes, for a
 *      @p kind whose bytes escape alone: JSON, HTML or URL escapes.
 *
 *  Every store is exact and every byte comes from the block's register, so in place the output of
 *  a block ends before the next block to be read.
 */
STRINGZILLA_INLINE sz_status_t sz_strings_escape_run_icelake_(sz_strings_view_t const *strings, sz_size_t count,
                                                              sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                              sz_strings_span_t *tape) {
    sz_size_t const start = sz_strings_view_offset_(strings, 0);
    sz_size_t const length = sz_strings_view_offset_(strings, count) - start;
    sz_cptr_t text = sz_strings_view_data_(strings) + start;
    // The last block reads a copy padded with a byte no kind escapes.
    sz_size_t const copied_from = length & ~(sz_size_t)63;
    sz_u8_t tail[64];
    memset(tail, 'a', sizeof(tail));
    if (length != copied_from) memcpy(tail, text + copied_from, length - copied_from);

    sz_size_t payload = length;
    for (sz_size_t block = 0; block < length; block += 64) {
        __m512i const text_u8x64 = _mm512_loadu_si512(block < copied_from ? (sz_u8_t const *)text + block : tail);
        payload += sz_strings_escape_sum_icelake_(sz_strings_escape_extras_icelake_(kind, text_u8x64),
                                                  _cvtu64_mask64(~(sz_u64_t)0));
    }

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_escapes_in_place_(strings, tape, count, payload)
                                   ? sz_strings_escape_grow_(tape, payload, allocator, &output, &text)
                                   : sz_strings_acquire_(strings, tape, count, payload, allocator, &output,
                                                         STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_size_t written = 0, row = 0;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_size_t const lanes = sz_min_of_two(length - block, (sz_size_t)64);
        __m512i const text_u8x64 = _mm512_loadu_si512(block < copied_from ? bytes + block : tail);
        __m512i const extras_u8x64 = sz_strings_escape_extras_icelake_(kind, text_u8x64);
        sz_u64_t const escaped = _cvtmask64_u64(_mm512_test_epi8_mask(extras_u8x64, extras_u8x64));
        if (kind == sz_escape_url_k) sz_strings_escape_url_emit_icelake_(text_u8x64, escaped, lanes, data + written);
        else sz_strings_escape_emit_icelake_(kind, text_u8x64, escaped, lanes, data + written);
        for (sz_size_t end; row != count && (end = sz_strings_view_offset_(strings, row + 1) - start) < block + 64;
             ++row)
            sz_strings_offset_set_(
                &output, row + 1,
                written + end - block + sz_strings_escape_sum_icelake_(extras_u8x64, sz_u64_mask_until_(end - block)));
        written += lanes + sz_strings_escape_sum_icelake_(extras_u8x64, _cvtu64_mask64(~(sz_u64_t)0));
    }
    for (; row != count; ++row) sz_strings_offset_set_(&output, row + 1, payload);
    sz_strings_offset_set_(&output, 0, 0);
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

/** The hexadecimal digit pairs of the 32 bytes of @p text_u8x32, from the 16 @p digits_u8x64
 *  starts with: the word of each byte holds its high nibble low and its low nibble high. */
STRINGZILLA_INLINE __m512i sz_strings_hex_pairs_icelake_(__m256i text_u8x32, __m512i digits_u8x64) {
    __m512i const words_u16x32 = _mm512_cvtepu8_epi16(text_u8x32);
    __m512i const nibbles_u8x64 = _mm512_or_si512(
        _mm512_srli_epi16(words_u16x32, 4),
        _mm512_slli_epi16(_mm512_and_si512(words_u16x32, _mm512_set1_epi16(15)), 8));
    return _mm512_permutexvar_epi8(nibbles_u8x64, digits_u8x64);
}

/** Escapes the @p count strings of a tape, read in its own order as one run of bytes, as
 *  hexadecimal digit pairs: every 32 bytes are read before their 64 are written, which in place
 *  end at the next unread byte at the latest. */
STRINGZILLA_INLINE sz_status_t sz_strings_escape_hex_icelake_(sz_strings_view_t const *strings, sz_size_t count,
                                                              sz_allocator_t *allocator, sz_strings_span_t *tape) {
    static sz_u8_t const digits[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    sz_size_t const start = sz_strings_view_offset_(strings, 0);
    sz_size_t const length = sz_strings_view_offset_(strings, count) - start, payload = length * 2;
    sz_cptr_t text = sz_strings_view_data_(strings) + start;
    sz_strings_span_t output;
    sz_status_t const status = sz_strings_escapes_in_place_(strings, tape, count, payload)
                                   ? sz_strings_escape_grow_(tape, payload, allocator, &output, &text)
                                   : sz_strings_acquire_(strings, tape, count, payload, allocator, &output,
                                                         STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    __m512i const digits_u8x64 = _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i const *)digits));
    sz_size_t offset = 0;
    for (; offset + 32 <= length; offset += 32)
        _mm512_storeu_si512(
            data + offset * 2,
            sz_strings_hex_pairs_icelake_(_mm256_loadu_si256((__m256i const *)(bytes + offset)), digits_u8x64));
    sz_size_t const rest = length - offset;
    __m256i const rest_u8x32 = _mm256_maskz_loadu_epi8(sz_u32_mask_until_(rest), bytes + offset);
    _mm512_mask_storeu_epi8(data + offset * 2, sz_u64_mask_until_(rest * 2),
                            sz_strings_hex_pairs_icelake_(rest_u8x32, digits_u8x64));
    for (sz_size_t row = 0; row <= count; ++row)
        sz_strings_offset_set_(&output, row, (sz_strings_view_offset_(strings, row) - start) * 2);
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

/** The lanes of @p text_u8x64 that are hexadecimal digits of either case. */
STRINGZILLA_INLINE sz_u64_t sz_strings_hex_digits_icelake_(__m512i text_u8x64) {
    __m512i const decimal_u8x64 = _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8('0'));
    __m512i const letter_u8x64 = _mm512_sub_epi8(_mm512_or_si512(text_u8x64, _mm512_set1_epi8(0x20)),
                                                 _mm512_set1_epi8('a'));
    return _cvtmask64_u64(_kor_mask64(_mm512_cmplt_epu8_mask(decimal_u8x64, _mm512_set1_epi8(10)),
                                      _mm512_cmplt_epu8_mask(letter_u8x64, _mm512_set1_epi8(6))));
}

/** The values of the hexadecimal digits of either case in @p text_u8x64, 4 bits in every lane. */
STRINGZILLA_INLINE __m512i sz_strings_hex_values_icelake_(__m512i text_u8x64) {
    __m512i const decimal_u8x64 = _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8('0'));
    __m512i const letter_u8x64 = _mm512_sub_epi8(_mm512_or_si512(text_u8x64, _mm512_set1_epi8(0x20)),
                                                 _mm512_set1_epi8('a' - 10));
    __m512i const values_u8x64 = _mm512_mask_blend_epi8(_mm512_cmplt_epu8_mask(decimal_u8x64, _mm512_set1_epi8(10)),
                                                        letter_u8x64, decimal_u8x64);
    return _mm512_and_si512(values_u8x64, _mm512_set1_epi8(15));
}

/**
 *  @brief Unescapes the @p count percent-encoded strings of a tape, read in its own order as one
 *      run of bytes, into fresh blocks: each block decodes its escapes in place and compresses
 *      away their digits.
 *  @return @c sz_invalid_utf8_k with nothing written when a `%` lacks two hexadecimal digits after
 *      it in its string, for the driver to name that string.
 */
STRINGZILLA_INLINE sz_status_t sz_strings_unescape_url_run_icelake_(sz_strings_view_t const *strings, sz_size_t count,
                                                                    sz_allocator_t *allocator,
                                                                    sz_strings_span_t *tape) {
    sz_size_t const start = sz_strings_view_offset_(strings, 0);
    sz_size_t const length = sz_strings_view_offset_(strings, count) - start;
    sz_u8_t const *const text = (sz_u8_t const *)sz_strings_view_data_(strings) + start;
    // Blocks within 66 bytes of the end read a zeroed copy, as decoding one reads 2 bytes past it.
    sz_size_t const copied_from = length < 66 ? 0 : ((length - 66) / 64 + 1) * 64;
    sz_u8_t tail[192] = {0};
    if (length != copied_from) memcpy(tail, text + copied_from, length - copied_from);

    sz_u64_t malformed = 0, owed = 0;
    sz_size_t percents_count = 0, row = 0;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? text + block : tail + (block - copied_from);
        __m512i const text_u8x64 = _mm512_loadu_si512(source);
        sz_u64_t const percents = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8('%')));
        sz_u64_t const needed = percents << 1 | percents << 2 | owed;
        malformed |= needed & ~sz_strings_hex_digits_icelake_(text_u8x64);
        // A string ending where a digit is due cuts its escape short.
        for (sz_size_t end; row != count && (end = sz_strings_view_offset_(strings, row + 1) - start) < block + 64;
             ++row)
            malformed |= needed >> (end - block) & 1;
        owed = percents >> 63 | percents >> 62, percents_count += (sz_size_t)_mm_popcnt_u64(percents);
    }
    if (malformed | owed) return sz_invalid_utf8_k;

    sz_size_t const payload = length - 2 * percents_count;
    sz_strings_span_t output;
    sz_status_t const status = sz_strings_acquire_(strings, tape, count, payload, allocator, &output, STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_size_t written = 0;
    owed = 0, percents_count = 0, row = 0;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? text + block : tail + (block - copied_from);
        __m512i const text_u8x64 = _mm512_loadu_si512(source);
        __m512i const high_u8x64 = sz_strings_hex_values_icelake_(_mm512_loadu_si512(source + 1));
        __m512i const low_u8x64 = sz_strings_hex_values_icelake_(_mm512_loadu_si512(source + 2));
        __mmask64 const percents_m64 = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8('%'));
        sz_u64_t const percents = _cvtmask64_u64(percents_m64);
        __m512i const decoded_u8x64 = _mm512_mask_blend_epi8(
            percents_m64, text_u8x64, _mm512_or_si512(_mm512_slli_epi16(high_u8x64, 4), low_u8x64));
        sz_u64_t const valid = _cvtmask64_u64(sz_u64_clamp_mask_until_(length - block));
        sz_u64_t const kept = ~(percents << 1 | percents << 2 | owed) & valid;
        sz_size_t const kept_count = (sz_size_t)_mm_popcnt_u64(kept);
        _mm512_mask_storeu_epi8(data + written, sz_u64_mask_until_(kept_count),
                                _mm512_maskz_compress_epi8(_cvtu64_mask64(kept), decoded_u8x64));
        for (sz_size_t end; row != count && (end = sz_strings_view_offset_(strings, row + 1) - start) < block + 64;
             ++row) {
            sz_u64_t const before = percents & _cvtmask64_u64(sz_u64_mask_until_(end - block));
            sz_strings_offset_set_(&output, row + 1, end - 2 * (percents_count + (sz_size_t)_mm_popcnt_u64(before)));
        }
        owed = percents >> 63 | percents >> 62, percents_count += (sz_size_t)_mm_popcnt_u64(percents);
        written += kept_count;
    }
    for (; row != count; ++row) sz_strings_offset_set_(&output, row + 1, payload);
    sz_strings_offset_set_(&output, 0, 0);
    sz_strings_commit_(tape, &output, allocator, STRINGZILLA_NULL);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_strings_to_tape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_allocator_t *allocator,
                                                       sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_tape_with_(strings, order, count, allocator, tape);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_strings_rows_t const *rows,
                                                       sz_size_t *longest, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_to_rows_with_(strings, order, count, rows, longest);
}

STRINGZILLA_API sz_status_t sz_strings_escape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // A tape read in its own order escapes as one run of bytes where each byte escapes alone.
    sz_bool_t const run = (sz_bool_t)(!order && strings->layout != sz_strings_accessor_k);
    if (run && kind == sz_escape_hex_k) return sz_strings_escape_hex_icelake_(strings, count, allocator, tape);
    if (run && kind == sz_escape_json_k)
        return sz_strings_escape_run_icelake_(strings, count, sz_escape_json_k, allocator, tape);
    if (run && kind == sz_escape_html_k)
        return sz_strings_escape_run_icelake_(strings, count, sz_escape_html_k, allocator, tape);
    if (run && kind == sz_escape_url_k)
        return sz_strings_escape_run_icelake_(strings, count, sz_escape_url_k, allocator, tape);
    sz_strings_walks_t const walks = sz_strings_walks_icelake_();
    return sz_strings_escape_with_(strings, order, count, kind, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_escape_kind_t kind,
                                                        sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                        sz_size_t *malformed_index, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // A run holding a malformed string fails, and the driver then names that string.
    sz_bool_t const run = (sz_bool_t)(!order && strings->layout != sz_strings_accessor_k && kind == sz_escape_url_k);
    sz_status_t const status = run ? sz_strings_unescape_url_run_icelake_(strings, count, allocator, tape)
                                   : sz_invalid_utf8_k;
    if (status != sz_invalid_utf8_k) return status;
    sz_strings_walks_t const walks = sz_strings_walks_icelake_();
    return sz_strings_unescape_with_(strings, order, count, kind, allocator, tape, malformed_index, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, char const lut[sz_at_least_(256)],
                                                      sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                      sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_strings_walks_t const walks = sz_strings_walks_icelake_();
    return sz_strings_lookup_with_(strings, order, count, lut, allocator, tape, &walks);
}

STRINGZILLA_API sz_status_t sz_strings_normalize_icelake(sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_normal_form_t form,
                                                         sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                         sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_strings_normalize_with_(strings, order, count, form, allocator, tape, &sz_utf8_norm_icelake);
}

STRINGZILLA_API sz_status_t sz_strings_split_icelake(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                     sz_split_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    __m512i const set_u8x64 = _mm512_broadcast_i64x4(_mm256_loadu_si256((__m256i const *)delimiters->_u8s));
    sz_u64_t const runs = kind == sz_split_runs_k ? ~(sz_u64_t)0 : 0;
    // The last block reads from a zeroed copy, as no load may pass the text.
    sz_size_t const copied_from = length & ~(sz_size_t)63;
    sz_u8_t tail[64] = {0};
    if (length != copied_from) memcpy(tail, bytes + copied_from, length - copied_from);

    sz_size_t removed_count = 0, ends_count = 0;
    sz_u64_t previous_removed = 1;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? bytes + block : tail;
        sz_size_t const lanes = sz_min_of_two(length - block, (sz_size_t)64);
        sz_u64_t const removed = sz_strings_classify_icelake_(_mm512_loadu_si512(source), set_u8x64) &
                                 _cvtmask64_u64(sz_u64_mask_until_(lanes));
        removed_count += (sz_size_t)_mm_popcnt_u64(removed);
        ends_count += (sz_size_t)_mm_popcnt_u64(removed & ~((removed << 1 | previous_removed) & runs));
        previous_removed = removed >> (lanes - 1) & 1;
    }
    sz_size_t const payload = length - removed_count;
    sz_size_t const parts = runs ? ends_count + (previous_removed ^ 1) : removed_count + 1;

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_split_acquire_(text, length, tape, parts, payload, allocator, &output);
    if (status != sz_success_k) return status;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_size_t written = 0, ended = 0;
    previous_removed = 1;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? bytes + block : tail;
        sz_size_t const lanes = sz_min_of_two(length - block, (sz_size_t)64);
        sz_u64_t const removed = sz_strings_classify_icelake_(_mm512_loadu_si512(source), set_u8x64) &
                                 _cvtmask64_u64(sz_u64_mask_until_(lanes));
        sz_u64_t const ends = removed & ~((removed << 1 | previous_removed) & runs);
        sz_strings_split_emit_icelake_(source, lanes, removed, ends, data + written, written, &output, &ended);
        written += lanes - (sz_size_t)_mm_popcnt_u64(removed), previous_removed = removed >> 63;
    }
    sz_strings_split_commit_(tape, &output, payload, allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_strings_split_lines_icelake(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                           sz_strings_span_t *tape, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    // Blocks within 66 bytes of the end read from a zeroed copy, as breaks look 2 bytes ahead.
    sz_size_t const copied_from = length < 66 ? 0 : ((length - 66) / 64 + 1) * 64;
    sz_u8_t tail[192] = {0};
    if (length != copied_from) memcpy(tail, bytes + copied_from, length - copied_from);

    sz_size_t removed_count = 0, ends_count = 0;
    sz_u64_t previous[3] = {0, 0, 0}, last_removed = 1, removed, ends;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? bytes + block : tail + (block - copied_from);
        sz_size_t const lanes = sz_min_of_two(length - block, (sz_size_t)64);
        sz_u64_t const valid = _cvtmask64_u64(sz_u64_mask_until_(lanes));
        sz_strings_lines_classify_icelake_(source, previous, &removed, &ends);
        removed &= valid, ends &= valid;
        removed_count += (sz_size_t)_mm_popcnt_u64(removed), ends_count += (sz_size_t)_mm_popcnt_u64(ends);
        last_removed = removed >> (lanes - 1) & 1;
    }
    sz_size_t const payload = length - removed_count, parts = ends_count + (last_removed ^ 1);

    sz_strings_span_t output;
    sz_status_t const status = sz_strings_split_acquire_(text, length, tape, parts, payload, allocator, &output);
    if (status != sz_success_k) return status;
    sz_u8_t *const data = (sz_u8_t *)sz_strings_payload_(&output);
    sz_size_t written = 0, ended = 0;
    previous[0] = previous[1] = previous[2] = 0;
    for (sz_size_t block = 0; block < length; block += 64) {
        sz_u8_t const *const source = block < copied_from ? bytes + block : tail + (block - copied_from);
        sz_size_t const lanes = sz_min_of_two(length - block, (sz_size_t)64);
        sz_u64_t const valid = _cvtmask64_u64(sz_u64_mask_until_(lanes));
        sz_strings_lines_classify_icelake_(source, previous, &removed, &ends);
        removed &= valid, ends &= valid;
        sz_strings_split_emit_icelake_(source, lanes, removed, ends, data + written, written, &output, &ended);
        written += lanes - (sz_size_t)_mm_popcnt_u64(removed);
    }
    sz_strings_split_commit_(tape, &output, payload, allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_strings_join_icelake(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                    sz_size_t rows, sz_string_view_t const *frames,
                                                    sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                    sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    if (!output) return sz_strings_join_measure_(columns, columns_count, rows, frames, between, length);
    // Accessors join cell by cell, while masked loads read frames of any length in place.
    sz_size_t by_cell = 0;
    for (sz_size_t column = 0; column != columns_count && !by_cell; ++column)
        by_cell = columns[column].layout == sz_strings_accessor_k;
    if (by_cell) return sz_strings_join_rows_serial_(columns, columns_count, rows, frames, between, output);

    sz_u8_t *target = (sz_u8_t *)output;
    for (sz_size_t row = 0; row != rows; ++row) {
        for (sz_size_t column = 0; column != columns_count; ++column) {
            sz_strings_join_piece_icelake_(target, (sz_u8_t const *)frames[column].start, frames[column].length);
            target += frames[column].length;
            sz_size_t const start = sz_strings_view_offset_(columns + column, row);
            sz_size_t const cell = sz_strings_view_offset_(columns + column, row + 1) - start;
            sz_strings_join_piece_icelake_(target, (sz_u8_t const *)sz_strings_view_data_(columns + column) + start,
                                           cell);
            target += cell;
        }
        sz_size_t const separator = row + 1 != rows ? between.length : 0;
        sz_strings_join_piece_icelake_(target, (sz_u8_t const *)frames[columns_count].start,
                                       frames[columns_count].length);
        target += frames[columns_count].length;
        sz_strings_join_piece_icelake_(target, (sz_u8_t const *)between.start, separator);
        target += separator;
    }
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_ICELAKE
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_STRINGS_ICELAKE_H_
