/**
 *  @brief AVX-512 backend for hardware-accelerated memory operations on Skylake and newer x86 CPUs.
 *  @file include/stringzilla/memory/skylake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/memory.h
 */
#ifndef STRINGZILLA_MEMORY_SKYLAKE_H_
#define STRINGZILLA_MEMORY_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

SZ_PUBLIC void sz_fill_skylake(sz_ptr_t target, sz_size_t length, sz_u8_t value) {
    __m512i value_vec = _mm512_set1_epi8(value);
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "stores".
    //
    //    for (; length >= 64; target += 64, length -= 64) _mm512_storeu_si512(target, value_vec);
    //    _mm512_mask_storeu_epi8(target, sz_u64_mask_until_(length), value_vec);
    //
    // When the buffer is small, there isn't much to innovate.
    if (length <= 64) {
        __mmask64 mask = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask, value_vec);
    }
    // When the buffer is over 64 bytes, it's guaranteed to touch at least two cache lines - the head and tail,
    // and may include more cache-lines in-between. Knowing this, we can avoid expensive unaligned stores
    // by computing 2 masks - for the head and tail, using masked stores for the head and tail, and unmasked
    // for the body.
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 64.
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask, value_vec);
        for (target += head_length; body_length >= 64; target += 64, body_length -= 64)
            _mm512_store_si512(target, value_vec);
        _mm512_mask_storeu_epi8(target, tail_mask, value_vec);
    }
}

SZ_PUBLIC void sz_copy_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "stores" and "loads".
    //
    //    for (; length >= 64; target += 64, source += 64, length -= 64)
    //        _mm512_storeu_si512(target, _mm512_loadu_si512(source));
    //    __mmask64 mask = sz_u64_mask_until_(length);
    //    _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    //
    // A typical AWS Sapphire Rapids instance can have 48 KB x 2 blocks of L1 data cache per core,
    // 2 MB x 2 blocks of L2 cache per core, and one shared 60 MB buffer of L3 cache.
    // With two strings, we may consider the overal workload huge, if each exceeds 1 MB in length.
    int const is_huge = length >= 1ull * 1024ull * 1024ull;

    // When the buffer is small, there isn't much to innovate.
    if (length <= 64) {
        __mmask64 mask = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    }
    // When dealing with larger arrays, the optimization is not as simple as with the `sz_fill_skylake` function,
    // as both buffers may be unaligned. If we are lucky and the requested operation is some huge page transfer,
    // we can use aligned loads and stores, and the performance will be great.
    else if ((sz_size_t)target % 64 == 0 && (sz_size_t)source % 64 == 0 && !is_huge) {
        for (; length >= 64; target += 64, source += 64, length -= 64)
            _mm512_store_si512(target, _mm512_load_si512(source));
        // At this point the length is guaranteed to be under 64.
        __mmask64 mask = sz_u64_mask_until_(length);
        // Aligned load and stores would work too, but it's not defined.
        _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    }
    // The trickiest case is when both `source` and `target` are not aligned.
    // In such and simpler cases we can copy enough bytes into `target` to reach its cacheline boundary,
    // and then combine unaligned loads with aligned stores.
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 64.
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
        for (target += head_length, source += head_length; body_length >= 64;
             target += 64, source += 64, body_length -= 64)
            _mm512_store_si512(target, _mm512_loadu_si512(source)); // Unaligned load, but aligned store!
        _mm512_mask_storeu_epi8(target, tail_mask, _mm512_maskz_loadu_epi8(tail_mask, source));
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    //
    //      1. Moving in both directions to maximize the throughput, when fetching from multiple
    //         memory pages. Also helps with cache set-associativity issues, as we won't always
    //         be fetching the same entries in the lookup table.
    //      2. Using non-temporal stores to avoid polluting the cache.
    //      3. Prefetching the next cache line, to avoid stalling the CPU. This generally useless
    //         for predictable patterns, so disregard this advice.
    //
    // Bidirectional traversal adds about 10%, accelerating from 11 GB/s to 12 GB/s.
    // Using "streaming stores" boosts us from 12 GB/s to 19 GB/s.
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);
        _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
        _mm512_mask_storeu_epi8(target + head_length + body_length, tail_mask,
                                _mm512_maskz_loadu_epi8(tail_mask, source + head_length + body_length));

        // Now in the main loop, we can use non-temporal loads and stores,
        // performing the operation in both directions.
        for (target += head_length, source += head_length; //
             body_length >= 128;                           //
             target += 64, source += 64, body_length -= 128) {
            _mm512_stream_si512((__m512i *)(target), _mm512_loadu_si512(source));
            _mm512_stream_si512((__m512i *)(target + body_length - 64), _mm512_loadu_si512(source + body_length - 64));
        }
        if (body_length >= 64) _mm512_stream_si512((__m512i *)target, _mm512_loadu_si512(source));
        // Non-temporal stores are weakly ordered; fence before returning so a later reader or an
        // overlapping copy cannot observe stale data (glibc's non-temporal `memcpy` fences the same way).
        _mm_sfence();
    }
}

SZ_PUBLIC void sz_move_skylake(sz_ptr_t target, sz_cptr_t source, sz_size_t length) {
    if (target == source) return; // Don't be silly, don't move the data if it's already there.

    // On very short buffers, that are one cache line in width or less, we don't need any loops.
    // We can also avoid any data-dependencies between iterations, assuming we have 32 registers
    // to pre-load the data, before writing it back.
    if (length <= 64) {
        __mmask64 mask = sz_u64_mask_until_(length);
        _mm512_mask_storeu_epi8(target, mask, _mm512_maskz_loadu_epi8(mask, source));
    }
    else if (length <= 128) {
        sz_size_t last_length = length - 64;
        __mmask64 mask = sz_u64_mask_until_(last_length);
        __m512i source0 = _mm512_loadu_epi8(source);
        __m512i source1 = _mm512_maskz_loadu_epi8(mask, source + 64);
        _mm512_storeu_epi8(target, source0);
        _mm512_mask_storeu_epi8(target + 64, mask, source1);
    }
    else if (length <= 192) {
        sz_size_t last_length = length - 128;
        __mmask64 mask = sz_u64_mask_until_(last_length);
        __m512i source0 = _mm512_loadu_epi8(source);
        __m512i source1 = _mm512_loadu_epi8(source + 64);
        __m512i source2 = _mm512_maskz_loadu_epi8(mask, source + 128);
        _mm512_storeu_epi8(target, source0);
        _mm512_storeu_epi8(target + 64, source1);
        _mm512_mask_storeu_epi8(target + 128, mask, source2);
    }
    else if (length <= 256) {
        sz_size_t last_length = length - 192;
        __mmask64 mask = sz_u64_mask_until_(last_length);
        __m512i source0 = _mm512_loadu_epi8(source);
        __m512i source1 = _mm512_loadu_epi8(source + 64);
        __m512i source2 = _mm512_loadu_epi8(source + 128);
        __m512i source3 = _mm512_maskz_loadu_epi8(mask, source + 192);
        _mm512_storeu_epi8(target, source0);
        _mm512_storeu_epi8(target + 64, source1);
        _mm512_storeu_epi8(target + 128, source2);
        _mm512_mask_storeu_epi8(target + 192, mask, source3);
    }

    // If the regions don't overlap at all, just use "copy" and save some brain cells thinking about corner cases.
    else if (target + length < source || target >= source + length) { sz_copy_skylake(target, source, length); }

    // When the buffer is over 64 bytes, it's guaranteed to touch at least two cache lines - the head and tail,
    // and may include more cache-lines in-between. Knowing this, we can avoid expensive unaligned stores
    // by computing 2 masks - for the head and tail, using masked stores for the head and tail, and unmasked
    // for the body.
    else {
        sz_size_t head_length = (64 - ((sz_size_t)target % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(target + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length;   // Multiple of 64.
        __mmask64 head_mask = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask = sz_u64_mask_until_(tail_length);

        // The absolute most common case of using "moves" is shifting the data within a continuous buffer
        // when adding a removing some values in it. In such cases, a typical shift is by 1, 2, 4, 8, 16,
        // or 32 bytes, rarely larger. For small shifts, under the size of the ZMM register, we can use shuffles.
        //
        // Remember:
        //      - if we are shifting data left, that we are traversing to the right.
        //      - if we are shifting data right, that we are traversing to the left.
        int const left_to_right_traversal = source > target;

        // Now we guarantee, that the relative shift within registers is from 1 to 63 bytes and the output is aligned.
        // Hopefully, we need to shift more than two ZMM registers, so we could consider `valignr` instruction.
        // Sadly, using `_mm512_alignr_epi8` doesn't make sense, as it operates at a 128-bit granularity.
        //
        //      - `_mm256_alignr_epi8` shifts entire 256-bit register, but we need many of them.
        //      - `_mm512_alignr_epi32` shifts 512-bit chunks, but only if the `shift` is a multiple of 4 bytes.
        //      - `_mm512_alignr_epi64` shifts 512-bit chunks by 8 bytes.
        //
        // All of those have a latency of 1 cycle, and the shift amount must be an immediate value!
        // For 1-byte-shift granularity, the `_mm512_permutex2var_epi8` has a latency of 6 and needs VBMI!
        // The most efficient and broadly compatible alternative could be to use a combination of align and shuffle.
        // A similar approach was outlined in "Byte-wise alignr in AVX512F" by Wojciech Muła.
        // http://0x80.pl/notesen/2016-10-16-avx512-byte-alignr.html
        //
        // That solution, is extremely mouthful, assuming we need compile time constants for the shift amount.
        // A cleaner one, with a latency of 3 cycles, is to use `_mm512_permutexvar_epi8` or
        // `_mm512_mask_permutexvar_epi8`, which can be seen as combination of a cross-register shuffle and blend,
        // and is available with VBMI. That solution is still noticeably slower than AVX2.
        //
        // The GLibC implementation also uses non-temporal stores for larger buffers, we don't.
        // https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/multiarch/memmove-avx512-no-vzeroupper.S.html
        if (left_to_right_traversal) {
            // Head, body, and tail.
            _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
            for (target += head_length, source += head_length; body_length >= 64;
                 target += 64, source += 64, body_length -= 64)
                _mm512_store_si512(target, _mm512_loadu_si512(source));
            _mm512_mask_storeu_epi8(target, tail_mask, _mm512_maskz_loadu_epi8(tail_mask, source));
        }
        else {
            // Tail, body, and head.
            _mm512_mask_storeu_epi8(target + head_length + body_length, tail_mask,
                                    _mm512_maskz_loadu_epi8(tail_mask, source + head_length + body_length));
            for (; body_length >= 64; body_length -= 64)
                _mm512_store_si512(target + head_length + body_length - 64,
                                   _mm512_loadu_si512(source + head_length + body_length - 64));
            _mm512_mask_storeu_epi8(target, head_mask, _mm512_maskz_loadu_epi8(head_mask, source));
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SKYLAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_MEMORY_SKYLAKE_H_
