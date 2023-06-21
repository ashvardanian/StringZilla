#pragma once
#include <stdint.h> // `byte_t`
#include <stddef.h> // `size_t`

#if !defined(__APPLE__)
#include <omp.h> // pragmas
#endif
#if defined(__AVX2__)
#include <x86intrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <limits>    // `std::numeric_limits`
#include <algorithm> // `std::search`

#pragma once
#if defined(__clang__)
#define stringzilla_compiler_is_gcc_m 0
#define stringzilla_compiler_is_clang_m 1
#define stringzilla_compiler_is_msvc_m 0
#define stringzilla_compiler_is_llvm_m 0
#define stringzilla_compiler_is_intel_m 0
#elif defined(__GNUC__) || defined(__GNUG__)
#define stringzilla_compiler_is_gcc_m 1
#define stringzilla_compiler_is_clang_m 0
#define stringzilla_compiler_is_msvc_m 0
#define stringzilla_compiler_is_llvm_m 0
#define stringzilla_compiler_is_intel_m 0
#elif defined(_MSC_VER)
#define stringzilla_compiler_is_gcc_m 0
#define stringzilla_compiler_is_clang_m 0
#define stringzilla_compiler_is_msvc_m 1
#define stringzilla_compiler_is_llvm_m 0
#define stringzilla_compiler_is_intel_m 0
#elif defined(__llvm__)
#define stringzilla_compiler_is_gcc_m 0
#define stringzilla_compiler_is_clang_m 0
#define stringzilla_compiler_is_msvc_m 0
#define stringzilla_compiler_is_llvm_m 1
#define stringzilla_compiler_is_intel_m 0
#elif defined(__INTEL_COMPILER)
#define stringzilla_compiler_is_gcc_m 0
#define stringzilla_compiler_is_clang_m 0
#define stringzilla_compiler_is_msvc_m 0
#define stringzilla_compiler_is_llvm_m 0
#define stringzilla_compiler_is_intel_m 1
#endif

namespace av::stringzilla {

    using byte_t = uint8_t;
    using byte8x_t = uint32_t;

    template <typename at = std::size_t>
    inline at divide_round_up(at x, at divisor) {
        return (x + (divisor - 1)) / divisor;
    }

    struct span_t {
        byte_t const *data_ = nullptr;
        size_t len_ = 0;

        inline size_t size() const noexcept { return len_; }
        inline byte_t const *begin() const noexcept { return data_; }
        inline byte_t const *end() const noexcept { return data_ + len_; }
        inline span_t after_n(size_t offset) const noexcept {
            return (offset < len_) ? span_t {data_ + offset, len_ - offset} : span_t {};
        }
        inline span_t before_n(size_t tail) const noexcept {
            return (tail < len_) ? span_t {data_ + len_ - tail, len_ - tail} : span_t {};
        }
    };

    /**
     *  @brief This is a faster alternative to `strncmp(a, b, len_) == 0`.
     */
    template <typename int_at>
    inline bool are_equal(int_at const *a, int_at const *b, size_t len_) noexcept {
        int_at const *const a_end = a + len_;
        for (; a != a_end && *a == *b; a++, b++)
            ;
        return a_end == a;
    }

    /**
     *  @brief  Iterates through every match with a callback.
     *  @return Total number of matches.
     */
    template <typename engine_at, typename callback_at>
    size_t find_all(span_t haystack, span_t needle, bool overlaps, engine_at &&engine, callback_at &&callback) {

        size_t match = 0;
        size_t progress = 0;
        size_t count_matches = 0;

        if (overlaps)
            while ((match = engine.next_offset(haystack.after_n(progress), needle)) != (haystack.size() - progress))
                callback(progress + match), count_matches++, progress += match + 1;
        else
            while ((match = engine.next_offset(haystack.after_n(progress), needle)) != (haystack.size() - progress))
                callback(progress + match), count_matches++, progress += match + needle.len_;

        return count_matches;
    }

    struct stl_t {

        size_t count(span_t haystack, byte_t needle) const noexcept {
            return std::count(haystack.begin(), haystack.end(), needle);
        }

        size_t next_offset(span_t haystack, byte_t needle) const noexcept {
            return std::find(haystack.begin(), haystack.end(), needle) - haystack.begin();
        }

        size_t count(span_t haystack, span_t needle, bool overlaps = false) const noexcept {
            return find_all(haystack, needle, overlaps, *this, [](size_t) {});
        }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {
            return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end()) - haystack.begin();
        }
    };

    /**
     *  @brief A naive subtring matching algorithm with O(|haystack|*|needle|) comparisons.
     *  Matching performance fluctuates between 200 MB/s and 2 GB/s.
     */
    struct naive_t {

        size_t count(span_t haystack, byte_t needle) const noexcept {
            size_t result = 0;
            for (byte_t byte : haystack)
                result += byte == needle;
            return result;
        }

        size_t next_offset(span_t haystack, byte_t needle) const noexcept {
            for (byte_t const &byte : haystack)
                if (byte == needle)
                    return &byte - haystack.data_;
            return haystack.len_;
        }

        size_t count(span_t haystack, span_t needle, bool overlap = false) const noexcept {

            if (haystack.len_ < needle.len_)
                return 0;

            size_t result = 0;
            if (!overlap)

                for (size_t off = 0; off <= haystack.len_ - needle.len_;)
                    if (are_equal(haystack.data_ + off, needle.data_, needle.len_))
                        off += needle.len_, result++;
                    else
                        off++;

            else
                for (size_t off = 0; off <= haystack.len_ - needle.len_; off++)
                    result += are_equal(haystack.data_ + off, needle.data_, needle.len_);

            return result;
        }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {

            if (haystack.len_ < needle.len_)
                return haystack.len_;

            for (size_t off = 0; off + needle.len_ <= haystack.len_; off++) {
                if (are_equal(haystack.data_ + off, needle.data_, needle.len_))
                    return off;
            }

            return haystack.len_;
        }
    };

    /**
     *  @brief Modified version inspired by Rabin-Karp algorithm.
     *  Matching performance fluctuates between 1 GB/s and 3,5 GB/s.
     *
     *  Similar to Rabin-Karp Algorithm, instead of comparing variable length
     *  strings - we can compare some fixed size fingerprints, which can make
     *  the number of nested loops smaller. But preprocessing text to generate
     *  hashes is very expensive.
     *  Instead - we compare the first 4 bytes of the `needle` to every 4 byte
     *  substring in the `haystack`. If those match - compare the rest.
     */
    struct prefixed_t {

        size_t count(span_t h, byte_t n) const noexcept { return naive_t {}.count(h, n); }
        size_t next_offset(span_t h, byte_t n) const noexcept { return naive_t {}.next_offset(h, n); }
        size_t count(span_t h, span_t n, bool o) const noexcept { return naive_t {}.count(h, n, o); }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {

            if (needle.len_ < 5)
                return naive_t {}.next_offset(haystack, needle);

            // Precomputed constants.
            byte_t const *h_ptr = haystack.data_;
            byte_t const *const h_end = haystack.end();
            size_t const n_suffix_len = needle.len_ - 4;
            byte8x_t const n_prefix = *reinterpret_cast<byte8x_t const *>(needle.data_);
            byte_t const *n_suffix_ptr = needle.data_ + 4;

            for (; h_ptr + needle.len_ <= h_end; h_ptr++) {
                if (n_prefix == *reinterpret_cast<byte8x_t const *>(h_ptr))
                    if (are_equal(h_ptr + 4, n_suffix_ptr, n_suffix_len))
                        return h_ptr - haystack.data_;
            }

            return haystack.len_;
        }
    };

    struct prefixed_autovec_t {

        size_t count(span_t h, byte_t n) const noexcept { return naive_t {}.count(h, n); }
        size_t next_offset(span_t h, byte_t n) const noexcept { return naive_t {}.next_offset(h, n); }
        size_t count(span_t h, span_t n, bool o) const noexcept { return naive_t {}.count(h, n, o); }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {

            if (needle.len_ < 5)
                return naive_t {}.next_offset(haystack, needle);

            byte_t const *h_ptr = haystack.data_;
            byte_t const *const h_end = haystack.end();
            byte8x_t const n_prefix = *reinterpret_cast<byte8x_t const *>(needle.data_);

            for (; (h_ptr + needle.len_ + 32) <= h_end; h_ptr += 32) {

                int count_matches = 0;

#if stringzilla_compiler_is_clang_m
#pragma clang loop vectorize(enable)
                for (size_t i = 0; i < 32; i++)
                    count_matches += (n_prefix == *reinterpret_cast<byte8x_t const *>(h_ptr + i));
#elif stringzilla_compiler_is_intel_m
#pragma vector always
#pragma ivdep
                for (size_t i = 0; i < 32; i++)
                    count_matches += (n_prefix == *reinterpret_cast<byte8x_t const *>(h_ptr + i));
#elif stringzilla_compiler_is_gcc_m
#pragma GCC ivdep
                for (size_t i = 0; i < 32; i++)
                    count_matches += (n_prefix == *reinterpret_cast<byte8x_t const *>(h_ptr + i));
#else
#pragma omp for simd reduction(+ : count_matches)
                for (size_t i = 0; i < 32; i++)
                    count_matches += (n_prefix == *reinterpret_cast<byte8x_t const *>(h_ptr + i));
#endif

                if (count_matches) {
                    for (size_t i = 0; i < 32; i++) {
                        if (are_equal(h_ptr + i, needle.data_, needle.len_))
                            return i + (h_ptr - haystack.data_);
                    }
                }
            }

            // Don't forget the last (up to 35) characters.
            size_t tail_start = h_ptr - haystack.data_;
            size_t tail_match = prefixed_t {}.next_offset(haystack.after_n(tail_start), needle);
            return tail_match + tail_start;
        }
    };

#if defined(__AVX2__)

    /**
     *  @brief A SIMD vectorized version for AVX2 instruction set.
     *  Matching performance is ~ 9 GB/s.
     *
     *  This version processes 32 `haystack` substrings per iteration,
     *  so the number of instructions is only:
     *  + 4 loads
     *  + 4 comparisons
     *  + 3 bitwise ORs
     *  + 1 masking
     *  for every 32 consecutive substrings.
     */
    struct prefixed_avx2_t {

        size_t count(span_t h, byte_t n) const noexcept { return naive_t {}.count(h, n); }
        size_t next_offset(span_t h, byte_t n) const noexcept { return naive_t {}.next_offset(h, n); }
        size_t count(span_t h, span_t n, bool o) const noexcept { return naive_t {}.count(h, n, o); }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {

            if (needle.len_ < 5)
                return naive_t {}.next_offset(haystack, needle);

            byte_t const *const h_end = haystack.end();
            __m256i const n_prefix = _mm256_set1_epi32(*(byte8x_t const *)(needle.data_));

            byte_t const *h_ptr = haystack.data_;
            for (; (h_ptr + needle.len_ + 32) <= h_end; h_ptr += 32) {

                __m256i h0 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr)), n_prefix);
                __m256i h1 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 1)), n_prefix);
                __m256i h2 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 2)), n_prefix);
                __m256i h3 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 3)), n_prefix);
                __m256i h_any = _mm256_or_si256(_mm256_or_si256(h0, h1), _mm256_or_si256(h2, h3));
                int mask = _mm256_movemask_epi8(h_any);

                if (mask) {
                    for (size_t i = 0; i < 32; i++) {
                        if (are_equal(h_ptr + i, needle.data_, needle.len_))
                            return i + (h_ptr - haystack.data_);
                    }
                }
            }

            // Don't forget the last (up to 35) characters.
            size_t tail_start = h_ptr - haystack.data_;
            size_t tail_match = prefixed_t {}.next_offset(haystack.after_n(tail_start), needle);
            return tail_match + tail_start;
        }
    };

    /**
     *  @brief Speculative SIMD version for AVX2 instruction set.
     *  Matching performance is ~ 12 GB/s.
     *
     *  Up to 40% of performance in modern CPUs comes from speculative
     *  out-of-order execution. The `prefixed_avx2_t` version has
     *  4 explicit local memory  barries: 3 ORs and 1 IF branch.
     *  This has only 1 IF branch in the main loop.
     */
    struct speculative_avx2_t {

        size_t count(span_t h, byte_t n) const noexcept { return naive_t {}.count(h, n); }
        size_t next_offset(span_t h, byte_t n) const noexcept { return naive_t {}.next_offset(h, n); }
        size_t count(span_t h, span_t n, bool o) const noexcept { return naive_t {}.count(h, n, o); }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {

            if (needle.len_ < 5)
                return naive_t {}.next_offset(haystack, needle);

            // Precomputed constants.
            byte_t const *const h_end = haystack.end();
            __m256i const n_prefix = _mm256_set1_epi32(*(byte8x_t const *)(needle.data_));

            // Top level for-loop changes dramatically.
            // In sequentail computing model for 32 offsets we would do:
            //  + 32 comparions.
            //  + 32 branches.
            // In vectorized computations models:
            //  + 4 vectorized comparisons.
            //  + 4 movemasks.
            //  + 3 bitwise ANDs.
            //  + 1 heavy (but very unlikely) branch.
            byte_t const *h_ptr = haystack.data_;
            for (; (h_ptr + needle.len_ + 32) <= h_end; h_ptr += 32) {

                __m256i h0_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr));
                int masks0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h0_prefixes, n_prefix));
                __m256i h1_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 1));
                int masks1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h1_prefixes, n_prefix));
                __m256i h2_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 2));
                int masks2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h2_prefixes, n_prefix));
                __m256i h3_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 3));
                int masks3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h3_prefixes, n_prefix));

                if (masks0 | masks1 | masks2 | masks3) {
                    for (size_t i = 0; i < 32; i++) {
                        if (are_equal(h_ptr + i, needle.data_, needle.len_))
                            return i + (h_ptr - haystack.data_);
                    }
                }
            }

            // Don't forget the last (up to 35) characters.
            size_t tail_start = h_ptr - haystack.data_;
            size_t tail_match = prefixed_t {}.next_offset(haystack.after_n(tail_start), needle);
            return tail_match + tail_start;
        }
    };

    /**
     *  @brief A hybrid of `prefixed_avx2_t` and `speculative_avx2_t`.
     *  It demonstrates the current inability of scheduler to optimize
     *  the execution flow better, than a human.
     */
    struct hybrid_avx2_t {

        size_t count(span_t h, byte_t n) const noexcept { return naive_t {}.count(h, n); }
        size_t next_offset(span_t h, byte_t n) const noexcept { return naive_t {}.next_offset(h, n); }
        size_t count(span_t h, span_t n, bool o) const noexcept { return naive_t {}.count(h, n, o); }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {

            if (needle.len_ < 5)
                return naive_t {}.next_offset(haystack, needle);

            byte_t const *const h_end = haystack.end();
            __m256i const n_prefix = _mm256_set1_epi32(*(byte8x_t const *)(needle.data_));

            byte_t const *h_ptr = haystack.data_;
            for (; (h_ptr + needle.len_ + 64) <= h_end; h_ptr += 64) {

                __m256i h0 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr)), n_prefix);
                __m256i h1 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 1)), n_prefix);
                __m256i h2 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 2)), n_prefix);
                __m256i h3 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 3)), n_prefix);
                int mask03 = _mm256_movemask_epi8(_mm256_or_si256(_mm256_or_si256(h0, h1), _mm256_or_si256(h2, h3)));

                __m256i h4 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 32)), n_prefix);
                __m256i h5 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 33)), n_prefix);
                __m256i h6 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 34)), n_prefix);
                __m256i h7 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 35)), n_prefix);
                int mask47 = _mm256_movemask_epi8(_mm256_or_si256(_mm256_or_si256(h4, h5), _mm256_or_si256(h6, h7)));

                if (mask03 | mask47) {
                    for (size_t i = 0; i < 64; i++) {
                        if (are_equal(h_ptr + i, needle.data_, needle.len_))
                            return i + (h_ptr - haystack.data_);
                    }
                }
            }

            // Don't forget the last (up to 67) characters.
            size_t tail_start = h_ptr - haystack.data_;
            size_t tail_match = prefixed_t {}.next_offset(haystack.after_n(tail_start), needle);
            return tail_match + tail_start;
        }
    };

#endif

#if defined(__AVX512F__)

    struct speculative_avx512_t {

        size_t count(span_t h, byte_t n) const noexcept { return naive_t {}.count(h, n); }
        size_t next_offset(span_t h, byte_t n) const noexcept { return naive_t {}.next_offset(h, n); }
        size_t count(span_t h, span_t n, bool o) const noexcept { return naive_t {}.count(h, n, o); }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {

            if (needle.len_ < 5)
                return naive_t {}.next_offset(haystack, needle);

            // Precomputed constants.
            byte_t const *const h_end = haystack.end();
            __m512i const n_prefix = _mm512_set1_epi32(*(byte8x_t const *)(needle.data_));

            byte_t const *h_ptr = haystack.data_;
            for (; (h_ptr + needle.len_ + 64) <= h_end; h_ptr += 64) {

                __m512i h0_prefixes = _mm512_loadu_si512((__m512i const *)(h_ptr));
                int masks0 = _mm512_cmpeq_epi32_mask(h0_prefixes, n_prefix);
                __m512i h1_prefixes = _mm512_loadu_si512((__m512i const *)(h_ptr + 1));
                int masks1 = _mm512_cmpeq_epi32_mask(h1_prefixes, n_prefix);
                __m512i h2_prefixes = _mm512_loadu_si512((__m512i const *)(h_ptr + 2));
                int masks2 = _mm512_cmpeq_epi32_mask(h2_prefixes, n_prefix);
                __m512i h3_prefixes = _mm512_loadu_si512((__m512i const *)(h_ptr + 3));
                int masks3 = _mm512_cmpeq_epi32_mask(h3_prefixes, n_prefix);

                if (masks0 | masks1 | masks2 | masks3) {
                    for (size_t i = 0; i < 64; i++) {
                        if (are_equal(h_ptr + i, needle.data_, needle.len_))
                            return i + (h_ptr - haystack.data_);
                    }
                }
            }

            // Don't forget the last (up to 64+3=67) characters.
            size_t tail_start = h_ptr - haystack.data_;
            size_t tail_match = prefixed_t {}.next_offset(haystack.after_n(tail_start), needle);
            return tail_match + tail_start;
        }
    };

#endif

#if defined(__ARM_NEON)
    /**
     *  @brief 128-bit implementation for ARM Neon.
     *
     *  https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/
     *  https://developer.arm.com/documentation/dui0473/m/neon-programming/neon-data_-types
     *  https://developer.arm.com/documentation/dui0473/m/neon-programming/neon-vectors
     *  https://blog.cloudflare.com/neon-is-the-new-black/
     */
    struct speculative_neon_t {

        size_t count(span_t h, byte_t n) const noexcept {
            // The plan is simple, skim through the misaligned part of the string.
            byte_t const *aligned_start = (byte_t const *)(divide_round_up<size_t>((uintptr_t)h.data_, 16) * 16);
            size_t misaligned_len = std::min(static_cast<size_t>(aligned_start - h.data_), h.len_);
            size_t result = naive_t {}.count({h.data_, misaligned_len}, n);

            if (h.len_ < misaligned_len)
                return result;

            // Count matches in the aligned part.
            byte_t const *h_ptr = aligned_start;
            byte_t const *const h_end = h.end();
            uint8x16_t n_vector = vld1q_dup_u8((uint8_t const *)&n);
            for (; (h_ptr + 16) <= h_end; h_ptr += 16) {
                uint8x16_t masks = vceqq_u8(vld1q_u8((uint8_t const *)h_ptr), n_vector);
                uint64x2_t masks64x2 = vreinterpretq_u64_u8(masks);
                result += __builtin_popcountll(vgetq_lane_u64(masks64x2, 0)) / 8;
                result += __builtin_popcountll(vgetq_lane_u64(masks64x2, 1)) / 8;
            }

            // Count matches in the misaligned tail.
            size_t tail_len = h_end - h_ptr;
            result += naive_t {}.count({h_ptr, tail_len}, n);
            return result;
        }

        size_t next_offset(span_t h, byte_t n) const noexcept { return naive_t {}.next_offset(h, n); }

        size_t count(span_t h, span_t n, bool o) const noexcept {
            return find_all(h, n, o, *this, [](size_t) {});
        }

        size_t next_offset(span_t haystack, span_t needle) const noexcept {

            if (needle.len_ < 4)
                return naive_t {}.next_offset(haystack, needle);

            // Precomputed constants.
            byte_t const *const h_end = haystack.end();
            uint32x4_t const n_prefix = vld1q_dup_u32((byte8x_t const *)(needle.data_));

            byte_t const *h_ptr = haystack.data_;
            for (; (h_ptr + needle.len_ + 16) <= h_end; h_ptr += 16) {

                uint32x4_t masks0 = vceqq_u32(vld1q_u32((byte8x_t const *)(h_ptr)), n_prefix);
                uint32x4_t masks1 = vceqq_u32(vld1q_u32((byte8x_t const *)(h_ptr + 1)), n_prefix);
                uint32x4_t masks2 = vceqq_u32(vld1q_u32((byte8x_t const *)(h_ptr + 2)), n_prefix);
                uint32x4_t masks3 = vceqq_u32(vld1q_u32((byte8x_t const *)(h_ptr + 3)), n_prefix);

                // Extracting matches from masks:
                // vmaxvq_u32 (only a64)
                // vgetq_lane_u32 (all)
                // vorrq_u32 (all)
                uint32x4_t masks = vorrq_u32(vorrq_u32(masks0, masks1), vorrq_u32(masks2, masks3));
                uint64x2_t masks64x2 = vreinterpretq_u64_u32(masks);
                bool has_match = vgetq_lane_u64(masks64x2, 0) | vgetq_lane_u64(masks64x2, 1);

                if (has_match) {
                    for (size_t i = 0; i < 16; i++) {
                        if (are_equal(h_ptr + i, needle.data_, needle.len_))
                            return i + (h_ptr - haystack.data_);
                    }
                }
            }

            // Don't forget the last (up to 16+3=19) characters.
            size_t tail_start = h_ptr - haystack.data_;
            size_t tail_match = prefixed_t {}.next_offset(haystack.after_n(tail_start), needle);
            return tail_match + tail_start;
        }
    };

#endif

} // namespace av::stringzilla
