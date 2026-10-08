/**
 *  @file test/cross.hpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel-level checks, shared by the family suites and the per-architecture cross files.
 *
 *  Every check takes its kernels by pointer and holds them to known answers, or to the serial
 *  kernel of the same operation. The family files run them over the dispatch points, and each
 *  `cross_<arch>.cpp` over its architecture's kernels by name, one section per capability.
 *
 *  Nothing here may reach a dispatch point or the C++ layer of `stringzilla.hpp`, because the
 *  header-only test runs these checks, where every dispatch point is a stub.
 */
#pragma once
#ifndef STRINGZILLA_TEST_CROSS_HPP
#define STRINGZILLA_TEST_CROSS_HPP

#include <stringzilla/stringzilla.h>

#include <cctype>  // `std::toupper`
#include <cmath>   // `std::fabs`
#include <cstdint> // `std::uint64_t`
#include <cstring> // `std::memcmp`, `std::memset`

#include <algorithm>   // `std::min`, `std::max`
#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits`
#include <random>      // `std::mt19937`, `std::uniform_int_distribution`
#include <set>         // `std::set`, the window-overlap oracle and intersection pairs
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include "harness.hpp"
#include "utf8.hpp"

namespace ashvardanian::stringzilla::test {

#pragma region Compare

/** Lengths every comparison and byte-scan size ladder switches at, plus one either side of each. */
inline constexpr sz_size_t backend_ladder_lengths_[] = {1,  7,  8,  9,  15,  16,  17,  31, 32,
                                                        33, 63, 64, 65, 127, 128, 129, 255};

/** One backend's comparison kernels: a capability's, or the dispatch points in their shape. */
struct compare_backend_t {
    char const *name;
    sz_kernel_equal_t equal = nullptr;
    sz_kernel_order_t order = nullptr;
};

/**
 *  @brief Holds one comparison backend to hand-verifiable pairs, then sweeps it across the lengths
 *      its size ladder switches at.
 *
 *  Every backend delegates to serial below its vector width, so the three-byte literals never reach
 *  the vectorized body at all. The overlapping-load boundaries at 8, 16, 32 and 64 are where these
 *  ladders break, and a mismatch is placed at the front, the middle and the last byte of each
 *  length to catch a tail that reads one byte too few or one too many.
 */
inline void check_compare_unit_(compare_backend_t const &backend) {
    sz_kernel_equal_t const equal = backend.equal;
    sz_kernel_order_t const order = backend.order;
    verify(kernel_result<sz_ordering_t>(order, "abc", 3, "abc", 3) == sz_equal_k);   // Equal strings
    verify(kernel_result<sz_ordering_t>(order, "abc", 3, "abd", 3) == sz_less_k);    // Differ in the last byte
    verify(kernel_result<sz_ordering_t>(order, "abd", 3, "abc", 3) == sz_greater_k); // Differ in the last byte
    verify(kernel_result<sz_ordering_t>(order, "ab", 2, "abc", 3) == sz_less_k);     // Prefix orders before
    verify(kernel_result<sz_ordering_t>(order, "abc", 3, "ab", 2) == sz_greater_k);  // Longer orders after
    verify(kernel_result<sz_bool_t>(equal, "abc", "abc", 3) == sz_true_k);           // Identical bytes
    verify(kernel_result<sz_bool_t>(equal, "abc", "abd", 3) == sz_false_k);          // Differing bytes

    std::string left, right;
    for (sz_size_t const length : span_over(backend_ladder_lengths_)) {
        left.assign((std::size_t)length, 'a');

        right = left;
        if (kernel_result<sz_bool_t>(equal, left.data(), right.data(), length) != sz_true_k) {
            fmt::println(stderr, "{}: equal() denied identical {}-byte inputs", backend.name, (std::size_t)length);
            verify(false && "Comparison backend must accept identical inputs at every ladder length");
        }
        verify(kernel_result<sz_ordering_t>(order, left.data(), length, right.data(), length) == sz_equal_k &&
               "Comparison backend must order two identical inputs as equal");

        sz_size_t const positions[] = {0, length / 2, length - 1};
        for (sz_size_t const position : span_over(positions)) {
            right = left, right[(std::size_t)position] = 'b'; // ? 'b' sorts after 'a', so `left` is the lesser
            if (kernel_result<sz_bool_t>(equal, left.data(), right.data(), length) != sz_false_k) {
                fmt::println(stderr, "{}: equal() missed a difference at byte {} of {}", backend.name,
                             (std::size_t)position, (std::size_t)length);
                verify(false && "Comparison backend must see a single differing byte at every ladder length");
            }
            verify(kernel_result<sz_ordering_t>(order, left.data(), length, right.data(), length) == sz_less_k &&
                   "Comparison backend must order the byte-decreased input as lesser");
            verify(kernel_result<sz_ordering_t>(order, right.data(), length, left.data(), length) == sz_greater_k &&
                   "Comparison backend must order the byte-increased input as greater");
        }
    }
}

#pragma endregion Compare

#pragma region Memory

/** The copy, move and fill kernels of one capability, or the dispatch points in their shape. */
struct memory_backend_t {
    char const *name;
    sz_kernel_copy_t copy = nullptr;
    sz_kernel_move_t move = nullptr;
    sz_kernel_fill_t fill = nullptr;
};

/** One backend's byte-lookup kernel: a capability's, or the dispatch point in its shape. */
struct lookup_backend_t {
    char const *name;
    sz_kernel_lookup_t lookup = nullptr;
};

/**
 *  @brief Runs one movement backend through hand-verifiable known-answer vectors.
 *
 *  Guard bytes past @c length catch stray writes, and @c move shifts a buffer into itself.
 */
inline void check_memory_unit_(memory_backend_t const &backend) {

    // `copy` duplicates a known buffer byte-for-byte. We over-allocate the target so a stray write
    // past `length` is visible as a corrupted guard byte.
    {
        char const source[] = "The quick brown fox"; // 19 bytes + terminator
        sz_size_t const length = (sz_size_t)(sizeof(source) - 1);
        char target[sizeof(source) + 1];
        std::memset(target, '#', sizeof(target));
        verify(backend.copy(target, source, length, nullptr) == sz_success_k);
        verify(std::memcmp(target, source, length) == 0 && "Copy backend diverged from the known-answer source");
        verify(target[length] == '#' && "Copy backend wrote past the requested length");
    }

    // `move` handles overlapping regions: shifting "abcdef" into itself by two yields "cdef" first.
    {
        char const expected[] = "cdef"; // After moving "cdef" (offset 2, 4 bytes) to offset 0
        char buffer[] = "abcdef";
        verify(backend.move(buffer, buffer + 2, 4, nullptr) == sz_success_k);
        verify(std::memcmp(buffer, expected, 4) == 0 && "Move backend produced wrong bytes for overlapping shift");
    }

    // `fill` writes a known byte across a known span, leaving a guard byte untouched.
    {
        char const expected[] = "*****"; // Five asterisks
        char target[5 + 1];
        std::memset(target, '#', sizeof(target));
        verify(backend.fill(target, 5, (sz_u8_t)'*', nullptr) == sz_success_k);
        verify(std::memcmp(target, expected, 5) == 0 && "Fill backend produced wrong bytes for the known pattern");
        verify(target[5] == '#' && "Fill backend wrote past the requested length");
    }
}

/**
 *  @brief Runs one byte-lookup backend through a hand-verifiable known-answer vector.
 *
 *  The upper-casing table maps "Hello, World!" to "HELLO, WORLD!" while leaving punctuation and
 *  digits intact; a guard byte past @c length catches stray writes.
 */
inline void check_lookup_unit_(lookup_backend_t const &backend) {
    // An ASCII upper-casing table built locally, so the known answer has an external ground truth.
    char upper_table[256];
    for (sz_size_t byte_value = 0; byte_value != 256; ++byte_value) {
        char const character = (char)(unsigned char)byte_value;
        upper_table[byte_value] = (character >= 'a' && character <= 'z') ? (char)(character - 'a' + 'A') : character;
    }
    char const source[] = "Hello, World!"; // 13 bytes
    char const expected[] = "HELLO, WORLD!";
    sz_size_t const length = (sz_size_t)(sizeof(source) - 1);
    char target[sizeof(source) + 1];
    std::memset(target, '#', sizeof(target));
    verify(backend.lookup(target, source, length, upper_table, nullptr) == sz_success_k);
    verify(std::memcmp(target, expected, length) == 0 && "Lookup backend diverged from the known-answer upper-casing");
    verify(target[length] == '#' && "Lookup backend wrote past the requested length");
}

/** A representative spread of lengths covering 0, tiny, the SWAR/SIMD-width neighborhood, and
 *  larger, so a kernel's head/body/tail handling is exercised on every backend. */
inline std::vector<sz_size_t> memory_equivalence_lengths() noexcept {
    return {0,  1,  2,  3,  7,  8,  9,   15,  16,  17,  31,  32,  33,   47,
            48, 63, 64, 65, 95, 96, 127, 128, 129, 255, 256, 257, 1024, 4096};
}

/**
 *  @brief Copies, moves and fills buffers with a @p candidate backend, comparing against serial.
 *
 *  Runs over @c for_each_cacheline_offset_ so the destination and source buffers are exercised at
 *  every sub-cache-line alignment, across the representative length set, with embedded-NUL content
 *  and overlapping @c move regions, so a misaligned head or tail bug on any backend is caught
 *  against the reference.
 */
inline void check_memory_equivalence_(test_context_t &context, memory_backend_t const &candidate) {

    memory_backend_t const reference {"serial", sz_copy_serial, sz_move_serial, sz_fill_serial};
    sz_size_t const inputs = (sz_size_t)context.iterations(2);
    std::vector<sz_size_t> const lengths = memory_equivalence_lengths();
    sz_size_t const max_length = lengths.back();

    for (sz_size_t length : lengths) {
        for (sz_size_t input = 0; input != inputs; ++input) {

            // A randomized source with embedded NULs - the byte primitives must stay length-driven.
            // It is read from a cache-line-shifted span, so the load alignment varies too.
            std::vector<char> source_storage(length + sz_default_alignment_k, '\0');
            sz_cptr_t const source = source_storage.data() + (input % sz_default_alignment_k);
            if (length) randomize_string(context.generator, {const_cast<char *>(source), length});

            // `copy` and `fill`: place the destination at every sub-cache-line alignment, comparing
            // the candidate output against a serial reference run at the same alignment.
            sz_u8_t const fill_value = (sz_u8_t)(0xA5u ^ (sz_u8_t)length);
            for_each_cacheline_offset_(max_length, [&](sz_ptr_t target, std::size_t) {
                std::vector<char> reference_output(length, '\0');
                verify(reference.copy(reference_output.data(), source, length, nullptr) == sz_success_k);
                verify(candidate.copy(target, source, length, nullptr) == sz_success_k);
                if (length)
                    verify(std::memcmp(reference_output.data(), target, length) == 0 &&
                           "Candidate copy backend diverged from the serial reference");

                verify(reference.fill(reference_output.data(), length, fill_value, nullptr) == sz_success_k);
                verify(candidate.fill(target, length, fill_value, nullptr) == sz_success_k);
                if (length)
                    verify(std::memcmp(reference_output.data(), target, length) == 0 &&
                           "Candidate fill backend diverged from the serial reference");
            });

            // `move` with overlapping regions: shift the source pattern within one buffer by a
            // small offset, both forwards and backwards, at every alignment of the buffer.
            for (sz_size_t shift : {(sz_size_t)1, (sz_size_t)7, (sz_size_t)16}) {
                if (length <= shift) continue;
                sz_size_t const moved = length - shift;
                for_each_cacheline_offset_(max_length + shift, [&](sz_ptr_t buffer, std::size_t) {
                    std::vector<char> reference_buffer(length + shift, '\0');

                    // Forward overlap: destination ahead of the source.
                    std::memcpy(buffer, source, length);
                    std::memcpy(reference_buffer.data(), source, length);
                    verify(candidate.move(buffer + shift, buffer, moved, nullptr) == sz_success_k);
                    verify(reference.move(reference_buffer.data() + shift, reference_buffer.data(), moved, nullptr) ==
                           sz_success_k);
                    verify(std::memcmp(buffer, reference_buffer.data(), length) == 0 &&
                           "Candidate move backend diverged from reference on forward overlap");

                    // Backward overlap: destination behind the source.
                    std::memcpy(buffer, source, length);
                    std::memcpy(reference_buffer.data(), source, length);
                    verify(candidate.move(buffer, buffer + shift, moved, nullptr) == sz_success_k);
                    verify(reference.move(reference_buffer.data(), reference_buffer.data() + shift, moved, nullptr) ==
                           sz_success_k);
                    verify(std::memcmp(buffer, reference_buffer.data(), length) == 0 &&
                           "Candidate move backend diverged from reference on backward overlap");
                });
            }
        }
    }
}

/**
 *  @brief Applies byte-lookup tables with a @p candidate backend, comparing against serial.
 *
 *  Runs over @c for_each_cacheline_offset_ so the destination and source buffers are exercised at
 *  every sub-cache-line alignment, across the representative length set, against the upper, lower
 *  and ASCII tables.
 */
inline void check_lookup_equivalence_(test_context_t &context, lookup_backend_t const &candidate) {

    lookup_backend_t const reference {"serial", sz_lookup_serial};
    sz_size_t const inputs = (sz_size_t)context.iterations(2);
    char upper_table[256], lower_table[256], ascii_table[256];
    sz_lookup_init_upper(upper_table);
    sz_lookup_init_lower(lower_table);
    sz_lookup_init_ascii(ascii_table);
    char const *const tables[] = {upper_table, lower_table, ascii_table};

    std::vector<sz_size_t> const lengths = memory_equivalence_lengths();
    sz_size_t const max_length = lengths.back();

    for (char const *table : tables)
        for (sz_size_t length : lengths) {
            for (sz_size_t input = 0; input != inputs; ++input) {

                std::vector<char> source_storage(length + sz_default_alignment_k, '\0');
                sz_cptr_t const source = source_storage.data() + (input % sz_default_alignment_k);
                if (length) randomize_string(context.generator, {const_cast<char *>(source), length});

                for_each_cacheline_offset_(max_length, [&](sz_ptr_t target, std::size_t) {
                    std::vector<char> reference_output(length, '\0');
                    verify(reference.lookup(reference_output.data(), source, length, table, nullptr) == sz_success_k);
                    verify(candidate.lookup(target, source, length, table, nullptr) == sz_success_k);
                    if (length)
                        verify(std::memcmp(reference_output.data(), target, length) == 0 &&
                               "Candidate lookup output diverged from reference for this lookup table");
                });
            }
        }
}

/** Runs one movement backend through adversarial inputs guarded by canary bytes, asserting no
 *  out-of-bounds write occurs (the canaries stay intact) and the operation does not crash. */
inline void check_memory_safety_(memory_backend_t const &backend) {

    // Zero-length: copy/move/fill must touch nothing, including NULL targets.
    verify(backend.copy(nullptr, nullptr, 0, nullptr) == sz_success_k);
    verify(backend.move(nullptr, nullptr, 0, nullptr) == sz_success_k);
    verify(backend.fill(nullptr, 0, (sz_u8_t)'!', nullptr) == sz_success_k);

    // A canary-guarded destination: writes outside [0, length) corrupt a guard byte.
    for (std::size_t length : {(std::size_t)1, (std::size_t)8, (std::size_t)64, (std::size_t)257})
        with_guarded_buffer_(length, [&](sz_ptr_t destination, std::size_t usable_length) {
            std::vector<char> source(usable_length, (char)0xC3);
            verify(backend.copy(destination, source.data(), usable_length, nullptr) == sz_success_k);
            verify(backend.fill(destination, usable_length, (sz_u8_t)0x7E, nullptr) == sz_success_k);
            verify(backend.move(destination, source.data(), usable_length, nullptr) == sz_success_k); // Disjoint
        });

    // Overlapping move inside one canary-guarded buffer, plus embedded-NUL content. The usable
    // window spans `length + shift`, so both shifted overlaps stay inside the guards.
    {
        std::size_t const length = 257;
        std::size_t const shift = 16;
        with_guarded_buffer_(length + shift, [&](sz_ptr_t buffer, std::size_t) {
            for (std::size_t byte = 0; byte != length; ++byte) buffer[byte] = (char)((byte % 2) ? (byte & 0xFF) : 0);
            verify(backend.move(buffer + shift, buffer, length, nullptr) == sz_success_k); // Forward overlap
            verify(backend.move(buffer, buffer + shift, length, nullptr) == sz_success_k); // Backward overlap
        });
    }
}

/** Runs one lookup backend through adversarial inputs guarded by canary bytes, asserting no
 *  out-of-bounds write occurs (the canaries stay intact) and the operation does not crash. */
inline void check_lookup_safety_(lookup_backend_t const &backend) {

    char upper_table[256], lower_table[256], ascii_table[256];
    sz_lookup_init_upper(upper_table);
    sz_lookup_init_lower(lower_table);
    sz_lookup_init_ascii(ascii_table);
    char const *const tables[] = {upper_table, lower_table, ascii_table};

    for (char const *table : tables) {
        verify(backend.lookup(nullptr, nullptr, 0, table, nullptr) == sz_success_k); // Zero-length touches nothing

        for (std::size_t length : {(std::size_t)1, (std::size_t)8, (std::size_t)64, (std::size_t)257})
            with_guarded_buffer_(length, [&](sz_ptr_t destination, std::size_t usable_length) {
                std::vector<char> source(usable_length, '\0');
                for (std::size_t byte = 0; byte != usable_length; ++byte)
                    source[byte] = (char)((byte % 3) ? 'a' + (byte % 26) : 0);
                verify(backend.lookup(destination, source.data(), usable_length, table, nullptr) == sz_success_k);
            });
    }
}

#pragma endregion Memory

#pragma region Find

/** The search kernels of one capability or of the dispatch points, null for a verb it lacks. */
struct find_backend_t {
    char const *name;
    sz_kernel_find_t find = nullptr;
    sz_kernel_find_t rfind = nullptr;
    sz_kernel_find_byte_t find_byte = nullptr;
    sz_kernel_find_byte_t rfind_byte = nullptr;
    sz_kernel_find_byteset_t find_byteset = nullptr;
    sz_kernel_find_byteset_t rfind_byteset = nullptr;
};

/**
 *  @brief Drives one byte-scanner pair across the size ladder, with a match at either end or none.
 *
 *  The reverse scanners are the risky half: each backend reaches "last match" differently -
 *  predicate reversal, gather-reversal, or a leading-zero count with a hand-written offset
 *  correction - and all of it is index arithmetic that an eleven-byte literal never exercises.
 */
inline void check_find_byte_ladder_(find_backend_t const &backend) {
    std::string haystack;
    for (sz_size_t const length : span_over(backend_ladder_lengths_)) {
        haystack.assign((std::size_t)length, 'a');
        verify(kernel_result<sz_cptr_t>(backend.find_byte, haystack.data(), length, "z") == STRINGZILLA_NULL_CHAR &&
               "Forward byte scan must miss an absent byte");
        verify(kernel_result<sz_cptr_t>(backend.rfind_byte, haystack.data(), length, "z") == STRINGZILLA_NULL_CHAR &&
               "Reverse byte scan must miss an absent byte");

        sz_size_t const positions[] = {0, length / 2, length - 1};
        for (sz_size_t const position : span_over(positions)) {
            haystack.assign((std::size_t)length, 'a');
            haystack[(std::size_t)position] = 'z';
            sz_cptr_t const expected = haystack.data() + position;
            if (kernel_result<sz_cptr_t>(backend.find_byte, haystack.data(), length, "z") != expected) {
                fmt::println(stderr, "{}: find_byte missed the byte at {} of {}", backend.name, (std::size_t)position,
                             (std::size_t)length);
                verify(false && "Forward byte scan must find a lone match at every ladder length");
            }
            if (kernel_result<sz_cptr_t>(backend.rfind_byte, haystack.data(), length, "z") != expected) {
                fmt::println(stderr, "{}: rfind_byte missed the byte at {} of {}", backend.name, (std::size_t)position,
                             (std::size_t)length);
                verify(false && "Reverse byte scan must find a lone match at every ladder length");
            }
        }

        // Both ends occupied, so forward and reverse must disagree about which one they report.
        haystack.assign((std::size_t)length, 'a');
        haystack[0] = 'z', haystack[(std::size_t)length - 1] = 'z';
        verify(kernel_result<sz_cptr_t>(backend.find_byte, haystack.data(), length, "z") == haystack.data() &&
               "Forward byte scan must report the first of two matches");
        verify(kernel_result<sz_cptr_t>(backend.rfind_byte, haystack.data(), length, "z") ==
                   haystack.data() + length - 1 &&
               "Reverse byte scan must report the last of two matches");
    }
}

/**
 *  @brief Holds one search backend to known answers in "hello world", then sweeps its byte scanners
 *      across the size ladder, since "hello world" only ever reaches a backend's scalar tail.
 *
 *  The substring "o" occurs at offsets 4 and 7, the needle "wor" at 6, and "xyz" nowhere; the
 *  byte 'l' at offsets 2, 3 and 9; the vowels first at 1 and last at 7, and no digit at all.
 */
inline void check_find_unit_(find_backend_t const &backend) {
    char const *hello = "hello world";
    sz_size_t const hello_length = (sz_size_t)std::strlen(hello); // 11 bytes

    if (backend.find) {
        verify(kernel_result<sz_cptr_t>(backend.find, hello, hello_length, "o", 1) == hello + 4);
        verify(kernel_result<sz_cptr_t>(backend.find, hello, hello_length, "wor", 3) == hello + 6);
        verify(kernel_result<sz_cptr_t>(backend.find, hello, hello_length, "xyz", 3) == STRINGZILLA_NULL_CHAR);
    }
    if (backend.rfind) {
        verify(kernel_result<sz_cptr_t>(backend.rfind, hello, hello_length, "o", 1) == hello + 7);
        verify(kernel_result<sz_cptr_t>(backend.rfind, hello, hello_length, "wor", 3) == hello + 6);
        verify(kernel_result<sz_cptr_t>(backend.rfind, hello, hello_length, "xyz", 3) == STRINGZILLA_NULL_CHAR);
    }
    if (backend.find_byte && backend.rfind_byte) {
        verify(kernel_result<sz_cptr_t>(backend.find_byte, hello, hello_length, "l") == hello + 2);
        verify(kernel_result<sz_cptr_t>(backend.rfind_byte, hello, hello_length, "l") == hello + 9);
        verify(kernel_result<sz_cptr_t>(backend.find_byte, hello, hello_length, "z") == STRINGZILLA_NULL_CHAR);
        verify(kernel_result<sz_cptr_t>(backend.rfind_byte, hello, hello_length, "z") == STRINGZILLA_NULL_CHAR);
        check_find_byte_ladder_(backend);
    }

    sz_byteset_t vowels, digits;
    sz_byteset_init(&vowels);
    for (char vowel : std::string_view("aeiou")) sz_byteset_add(&vowels, vowel);
    sz_byteset_init(&digits);
    sz_byteset_add(&digits, '0');
    sz_byteset_add(&digits, '9');
    if (backend.find_byteset) {
        verify(kernel_result<sz_cptr_t>(backend.find_byteset, hello, hello_length, &vowels) == hello + 1);
        verify(kernel_result<sz_cptr_t>(backend.find_byteset, hello, hello_length, &digits) == STRINGZILLA_NULL_CHAR);
    }
    if (backend.rfind_byteset) {
        verify(kernel_result<sz_cptr_t>(backend.rfind_byteset, hello, hello_length, &vowels) == hello + 7);
        verify(kernel_result<sz_cptr_t>(backend.rfind_byteset, hello, hello_length, &digits) == STRINGZILLA_NULL_CHAR);
    }
}

/**
 *  @brief Cross-checks two substring-search kernels of the @b same operation, both forward or both
 *      backward, on random and hand-picked edge-case inputs.
 *
 *  The @p candidate must resolve every needle to the identical pointer the @p reference does, or
 *  both must return @c STRINGZILLA_NULL. Each haystack is replayed at every sub-cacheline alignment
 *  via @c for_each_cacheline_offset_, so a needle straddling a 64-byte boundary is exercised.
 */
inline void check_find_search_equivalence_(test_context_t &context, char const *name, sz_kernel_find_t reference,
                                           sz_kernel_find_t candidate) {

    // Replays one haystack/needle pair at every intra-cacheline alignment and compares the kernels.
    auto compare_on = [&](std::string const &haystack_pattern, std::string const &needle) {
        for_each_cacheline_offset_(
            haystack_pattern.size(), [&](sz_ptr_t haystack, [[maybe_unused]] std::size_t offset) {
                std::memcpy(haystack, haystack_pattern.data(), haystack_pattern.size());
                sz_size_t const haystack_length = (sz_size_t)haystack_pattern.size();
                sz_size_t const needle_length = (sz_size_t)needle.size();

                sz_cptr_t const result_reference = kernel_result<sz_cptr_t>(reference, haystack, haystack_length,
                                                                            needle.data(), needle_length);
                sz_cptr_t const result_candidate = kernel_result<sz_cptr_t>(candidate, haystack, haystack_length,
                                                                            needle.data(), needle_length);
                if (result_reference != result_candidate) {
                    fmt::println(stderr,
                                 "serial vs {}: substring search disagreed on a {}-byte needle in a {}-byte haystack",
                                 name, (std::size_t)needle_length, (std::size_t)haystack_length);
                    verify(false && "Candidate backend must resolve every needle to the same offset as the reference");
                }
            });
    };

    // Hand-picked edge cases: empty needle, not-found, needle at start, needle at end, needle equal
    // to the haystack, repeated occurrences, and an embedded NUL byte.
    compare_on("hello world", "");                                 // Empty needle
    compare_on("hello world", "xyz");                              // Not found
    compare_on("hello world", "hello");                            // Needle at the start
    compare_on("hello world", "world");                            // Needle at the end
    compare_on("hello world", "hello world");                      // Needle equals the haystack
    compare_on("abababab", "ab");                                  // Repeated occurrences
    compare_on(std::string("a\0bc\0a", 6), std::string("\0a", 2)); // Embedded NUL byte

    // Random haystacks and needles of assorted lengths.
    for (sz_size_t iteration = 0; iteration != context.iterations(200); ++iteration) {
        std::size_t const haystack_length = std::uniform_int_distribution<std::size_t>(0, 200)(context.generator);
        std::size_t const needle_length = std::uniform_int_distribution<std::size_t>(
            0, haystack_length + 4)(context.generator);
        // A small alphabet makes spurious and overlapping matches likely, stressing the kernels.
        std::string const haystack = random_string(context.generator, haystack_length, "abc");
        std::string const needle = random_string(context.generator, needle_length, "abc");
        compare_on(haystack, needle);
    }
}

/**
 *  @brief Cross-checks two byteset-search kernels of the @b same operation, both forward or both
 *      backward, on random and hand-picked edge-case inputs.
 *
 *  The @p candidate must resolve every byteset to the identical pointer the @p reference does, or
 *  both must return @c STRINGZILLA_NULL. Each haystack is replayed at every sub-cacheline alignment
 *  via @c for_each_cacheline_offset_, so a match straddling a 64-byte boundary is always exercised.
 */
inline void check_byteset_equivalence_(test_context_t &context, char const *name, sz_kernel_find_byteset_t reference,
                                       sz_kernel_find_byteset_t candidate) {

    // The byteset of ASCII vowels, used for the hand-picked structured cases.
    sz_byteset_t vowels;
    sz_byteset_init(&vowels);
    for (char vowel : std::string_view("aeiou")) sz_byteset_add(&vowels, vowel);

    // Replays one haystack at every intra-cacheline alignment and compares the kernels.
    auto compare_on = [&](std::string const &haystack_pattern, sz_byteset_t const &byteset) {
        for_each_cacheline_offset_(haystack_pattern.size(), [&](sz_ptr_t haystack,
                                                                [[maybe_unused]] std::size_t offset) {
            std::memcpy(haystack, haystack_pattern.data(), haystack_pattern.size());
            sz_size_t const haystack_length = (sz_size_t)haystack_pattern.size();

            sz_cptr_t const result_reference = kernel_result<sz_cptr_t>(reference, haystack, haystack_length, &byteset);
            sz_cptr_t const result_candidate = kernel_result<sz_cptr_t>(candidate, haystack, haystack_length, &byteset);
            if (result_reference != result_candidate) {
                fmt::println(stderr, "serial vs {}: byteset search disagreed on a {}-byte haystack", name,
                             (std::size_t)haystack_length);
                verify(false && "Candidate backend must resolve every byteset to the same offset as the reference");
            }
        });
    };

    // Hand-picked edge cases: empty haystack, no member present, member at start, member at end,
    // all members, repeated members, and an embedded NUL byte.
    compare_on("", vowels);                         // Empty haystack
    compare_on("xyz wrld", vowels);                 // No member present
    compare_on("apple", vowels);                    // Member at the start
    compare_on("xyzo", vowels);                     // Member at the end
    compare_on("aeiou", vowels);                    // Every byte is a member
    compare_on("aaeeii", vowels);                   // Repeated members
    compare_on(std::string("x\0aey\0", 6), vowels); // Embedded NUL byte

    // Random haystacks of assorted lengths against a random byteset.
    for (sz_size_t iteration = 0; iteration != context.iterations(200); ++iteration) {
        sz_byteset_t random_byteset;
        sz_byteset_init(&random_byteset);
        std::size_t const members = std::uniform_int_distribution<std::size_t>(0, 16)(context.generator);
        for (std::size_t member = 0; member != members; ++member)
            sz_byteset_add(&random_byteset, (sz_u8_t)std::uniform_int_distribution<int>(0, 255)(context.generator));

        std::size_t const haystack_length = std::uniform_int_distribution<std::size_t>(0, 200)(context.generator);
        std::string haystack(haystack_length, '\0');
        randomize_string(context.generator, haystack);
        compare_on(haystack, random_byteset);
    }
}

/** Holds every search kernel of the @p candidate backend to the serial kernel of the same verb. */
inline void check_find_equivalence_(test_context_t &context, find_backend_t const &candidate) {
    if (candidate.find) check_find_search_equivalence_(context, candidate.name, sz_find_serial, candidate.find);
    if (candidate.rfind) check_find_search_equivalence_(context, candidate.name, sz_rfind_serial, candidate.rfind);
    if (candidate.find_byteset)
        check_byteset_equivalence_(context, candidate.name, sz_find_byteset_serial, candidate.find_byteset);
    if (candidate.rfind_byteset)
        check_byteset_equivalence_(context, candidate.name, sz_rfind_byteset_serial, candidate.rfind_byteset);
}

/**
 *  @brief Degenerate and boundary shapes for one search backend, asserting survival and bounds.
 *
 *  Answers are not the subject here - a needle longer than its haystack, a zero length, or an empty
 *  byteset each has one defensible reply, and what matters is that the backend gives it without
 *  reading a byte it was not handed. The scanners run at every sub-cache-line alignment, since a
 *  misaligned tail is where an overlapping load reaches past the end.
 */
inline void check_find_safety_(find_backend_t const &backend) {
    char const *body = "the quick brown fox";
    sz_size_t const body_length = (sz_size_t)std::strlen(body);

    if (backend.find_byte && backend.rfind_byte) {
        if (kernel_result<sz_cptr_t>(backend.find_byte, body, 0, "a") != STRINGZILLA_NULL_CHAR) {
            fmt::println(stderr, "{}: find_byte reported a match in a zero-length haystack", backend.name);
            verify(false && "A zero-length haystack holds no byte to find");
        }
        if (kernel_result<sz_cptr_t>(backend.rfind_byte, body, 0, "a") != STRINGZILLA_NULL_CHAR) {
            fmt::println(stderr, "{}: rfind_byte reported a match in a zero-length haystack", backend.name);
            verify(false && "A zero-length haystack holds no byte to find");
        }
    }
    if (backend.find) {
        verify(kernel_result<sz_cptr_t>(backend.find, body, 0, "a", 1) == STRINGZILLA_NULL_CHAR);
        verify(kernel_result<sz_cptr_t>(backend.find, body, 3, body, body_length) == STRINGZILLA_NULL_CHAR);
    }
    if (backend.rfind) {
        verify(kernel_result<sz_cptr_t>(backend.rfind, body, 0, "a", 1) == STRINGZILLA_NULL_CHAR);
        verify(kernel_result<sz_cptr_t>(backend.rfind, body, 3, body, body_length) == STRINGZILLA_NULL_CHAR);
    }

    // An empty byteset matches nothing; a full one matches the first and last byte.
    sz_byteset_t empty_set, full_set;
    sz_byteset_init(&empty_set);
    sz_byteset_init(&full_set);
    for (int byte_value = 0; byte_value != 256; ++byte_value) sz_byteset_add_u8(&full_set, (sz_u8_t)byte_value);
    if (backend.find_byteset) {
        verify(kernel_result<sz_cptr_t>(backend.find_byteset, body, body_length, &empty_set) == STRINGZILLA_NULL_CHAR);
        verify(kernel_result<sz_cptr_t>(backend.find_byteset, body, body_length, &full_set) == body);
        verify(kernel_result<sz_cptr_t>(backend.find_byteset, body, 0, &full_set) == STRINGZILLA_NULL_CHAR);
    }
    if (backend.rfind_byteset) {
        verify(kernel_result<sz_cptr_t>(backend.rfind_byteset, body, body_length, &empty_set) == STRINGZILLA_NULL_CHAR);
        verify(kernel_result<sz_cptr_t>(backend.rfind_byteset, body, body_length, &full_set) == body + body_length - 1);
    }

    // Every scanner at every sub-cache-line alignment, over a buffer without a match, then with one
    // at the end.
    for (sz_size_t length : span_over(backend_ladder_lengths_)) {
        for_each_cacheline_offset_((std::size_t)length, [&](sz_ptr_t buffer, std::size_t) {
            std::memset(buffer, 'a', (std::size_t)length);
            if (backend.find_byte)
                verify(kernel_result<sz_cptr_t>(backend.find_byte, buffer, length, "z") == STRINGZILLA_NULL_CHAR);
            if (backend.rfind_byte)
                verify(kernel_result<sz_cptr_t>(backend.rfind_byte, buffer, length, "z") == STRINGZILLA_NULL_CHAR);
            if (backend.find)
                verify(kernel_result<sz_cptr_t>(backend.find, buffer, length, "zz", 2) == STRINGZILLA_NULL_CHAR);
            buffer[length - 1] = 'z';
            if (backend.find_byte)
                verify(kernel_result<sz_cptr_t>(backend.find_byte, buffer, length, "z") == buffer + length - 1);
            if (backend.rfind_byte)
                verify(kernel_result<sz_cptr_t>(backend.rfind_byte, buffer, length, "z") == buffer + length - 1);
        });
    }
}

#pragma endregion Find

#pragma region Hash

/** Parses a 64-character lowercase-hex SHA256 digest into 32 bytes. */
inline void sha256_digest_from_hex_(char const *hex, sz_u8_t (&digest)[STRINGZILLA_SHA256_DIGEST_LENGTH]) {
    auto nibble = [](char character) -> sz_u8_t {
        if (character >= '0' && character <= '9') return (sz_u8_t)(character - '0');
        return (sz_u8_t)(character - 'a' + 10);
    };
    for (std::size_t byte_index = 0; byte_index != STRINGZILLA_SHA256_DIGEST_LENGTH; ++byte_index)
        digest[byte_index] = (sz_u8_t)((nibble(hex[byte_index * 2]) << 4) | nibble(hex[byte_index * 2 + 1]));
}

/** A message paired with the digest it must produce, for known-answer testing. */
struct known_sha256_t {
    char const *message;
    char const *digest_hex;
};

/** Seven bytes whose fourth is NUL, which a kernel stopping at a terminator would cut to three. */
inline constexpr char interior_nul_[] = "abc\x00" // "\x00d" would be one escape
                                        "def";

/** The three canonical FIPS 180-4 vectors: empty, "abc", and the 56-byte two-block message. */
inline constexpr known_sha256_t known_sha256_vectors_[] = {
    {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},    //
    {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}, //
    {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",                 //
     "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
};

/** One hashing backend's one-shot and streaming kernels. */
struct hash_backend_t {
    sz_kernel_hash_t hash_kernel;
    sz_kernel_hash_state_init_t init_kernel;
    sz_kernel_hash_state_update_t update_kernel;
    sz_kernel_hash_state_digest_t digest_kernel;

    sz_u64_t operator()(sz_cptr_t text, sz_size_t length, sz_u64_t seed) const {
        return kernel_result<sz_u64_t>(hash_kernel, text, length, seed);
    }
    void init(sz_hash_state_t *state, sz_u64_t seed) const {
        verify(init_kernel(state, seed, nullptr) == sz_success_k);
    }
    void update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) const {
        verify(update_kernel(state, text, length, nullptr) == sz_success_k);
    }
    sz_u64_t digest(sz_hash_state_t const *state) const { return kernel_result<sz_u64_t>(digest_kernel, state); }
};

/** One backend's batch multi-seed hash, beside the single-seed hash it must reduce to. */
struct hash_multiseed_backend_t {
    sz_kernel_hash_multiseed_t multiseed_kernel;
    sz_kernel_hash_t hash_kernel;

    void multiseed(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds, sz_size_t seed_count,
                   sz_u64_t *output) const {
        verify(multiseed_kernel(text, length, seeds, seed_count, output, nullptr) == sz_success_k);
    }
    sz_u64_t hash_one(sz_cptr_t text, sz_size_t length, sz_u64_t seed) const {
        return kernel_result<sz_u64_t>(hash_kernel, text, length, seed);
    }
};

/** One SHA256 backend's init, update and digest kernels. */
struct sha256_backend_t {
    sz_kernel_sha256_state_init_t init_kernel;
    sz_kernel_sha256_state_update_t update_kernel;
    sz_kernel_sha256_state_digest_t digest_kernel;

    void init(sz_sha256_state_t *state) const { verify(init_kernel(state, nullptr) == sz_success_k); }
    void update(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length) const {
        verify(update_kernel(state, text, length, nullptr) == sz_success_k);
    }
    void digest(sz_sha256_state_t *state, sz_u8_t *output) const {
        verify(digest_kernel(state, output, nullptr) == sz_success_k);
    }
};

/** One multi-state SHA256 backend's batch update and digest kernels. */
struct sha256_multistate_backend_t {
    sz_kernel_sha256_multistate_update_t update_kernel;
    sz_kernel_sha256_multistate_digest_t digest_kernel;

    void update(sz_sha256_state_t *states, sz_sequence_t const *texts) const {
        verify(update_kernel(states, texts, nullptr) == sz_success_k);
    }
    void digest(sz_sha256_state_t const *states, sz_size_t states_count, sz_u8_t *digests) const {
        verify(digest_kernel(states, states_count, digests, nullptr) == sz_success_k);
    }
};

/** The serial hashing kernels, every other backend's reference. */
inline hash_backend_t hash_serial_backend_() noexcept {
    return {sz_hash_serial, sz_hash_state_init_serial, sz_hash_state_update_serial, sz_hash_state_digest_serial};
}

/** Runs one SHA256 backend over @p message and asserts the expected digest. */
inline void check_sha256_digest_(sha256_backend_t const &backend, std::string const &message,
                                 char const *expected_hex) {
    sz_sha256_state_t state;
    sz_u8_t produced[STRINGZILLA_SHA256_DIGEST_LENGTH], expected[STRINGZILLA_SHA256_DIGEST_LENGTH];
    sha256_digest_from_hex_(expected_hex, expected);
    backend.init(&state);
    backend.update(&state, message.data(), (sz_size_t)message.size());
    backend.digest(&state, produced);
    verify(std::memcmp(produced, expected, STRINGZILLA_SHA256_DIGEST_LENGTH) == 0);
}

/** Holds one SHA256 backend to the FIPS 180-4 vectors and to every byte of @c interior_nul_. */
inline void check_sha256_unit_(sha256_backend_t const &backend) {
    for (known_sha256_t const &vector : span_over(known_sha256_vectors_))
        check_sha256_digest_(backend, vector.message, vector.digest_hex);
    check_sha256_digest_(backend, std::string(interior_nul_, 7),
                         "516a5e926ce20c5f4d80f00e1a01abdf14986def6588d6abeed9fce090bc660c");
}

/** Holds one multi-state SHA256 backend to the FIPS 180-4 vectors hashed as one batch, so it is
 *  pinned to known digests rather than only to another implementation of itself. */
inline void check_sha256_multistate_unit_(sha256_multistate_backend_t const &backend) {
    std::size_t const vectors_count = span_over(known_sha256_vectors_).size();
    std::vector<sz_sha256_state_t> states(vectors_count);
    std::vector<sz_u8_t> produced(vectors_count * STRINGZILLA_SHA256_DIGEST_LENGTH);
    std::vector<sz_string_view_t> messages(vectors_count);
    sz_u8_t expected[STRINGZILLA_SHA256_DIGEST_LENGTH];
    for (std::size_t lane_index = 0; lane_index != vectors_count; ++lane_index) {
        verify(sz_sha256_state_init_serial(&states[lane_index], nullptr) == sz_success_k);
        messages[lane_index].start = known_sha256_vectors_[lane_index].message;
        messages[lane_index].length = (sz_size_t)std::strlen(known_sha256_vectors_[lane_index].message);
    }
    sz_sequence_t texts;
    sz_sequence_from_string_views(messages.data(), vectors_count, &texts);
    backend.update(states.data(), &texts);
    backend.digest(states.data(), (sz_size_t)vectors_count, produced.data());
    for (std::size_t lane_index = 0; lane_index != vectors_count; ++lane_index) {
        sha256_digest_from_hex_(known_sha256_vectors_[lane_index].digest_hex, expected);
        verify(std::memcmp(&produced[lane_index * STRINGZILLA_SHA256_DIGEST_LENGTH], expected,
                           STRINGZILLA_SHA256_DIGEST_LENGTH) == 0 &&
               "Multi-state digest disagreed with the known-answer digest for this lane");
    }
}

/** Holds one byte sum to known totals: "abc" sums to 0x126, an interior NUL is read past, and the
 *  empty input sums to zero. */
inline void check_bytesum_unit_(sz_kernel_bytesum_t bytesum) {
    verify(kernel_result<sz_u64_t>(bytesum, "abc", 3) == 0x126u);
    verify(kernel_result<sz_u64_t>(bytesum, interior_nul_, 7) == 0x255u);
    verify(kernel_result<sz_u64_t>(bytesum, "", 0) == 0);
}

/** Holds one hash to what a dropped seed or a short read would break: the seed reaches it at a
 *  short and a multi-word length, it reads past an interior NUL, and empty input hashes stably. */
inline void check_hash_unit_(sz_kernel_hash_t hash) {
    verify(kernel_result<sz_u64_t>(hash, "abc", 3, 100) != kernel_result<sz_u64_t>(hash, "abc", 3, 200));
    verify(kernel_result<sz_u64_t>(hash, "abcdefgh", 8, 0) != kernel_result<sz_u64_t>(hash, "abcdefgh", 8, 7));
    verify(kernel_result<sz_u64_t>(hash, interior_nul_, 7, 0) != kernel_result<sz_u64_t>(hash, interior_nul_, 3, 0));
    verify(kernel_result<sz_u64_t>(hash, "", 0, 0) == kernel_result<sz_u64_t>(hash, "", 0, 0));
}

/**
 *  @brief Cross-checks a byte-summing backend against serial across lengths and alignments.
 *
 *  The wide kernels split a buffer into an unaligned head, an aligned body, and a tail, so sweeping
 *  cache-line offsets is what reaches the head and tail paths at all. The AVX-512 tiers
 *  additionally switch to non-temporal loads and bidirectional traversal past a megabyte, which
 *  only one oversized input reaches.
 */
inline void check_bytesum_equivalence_(test_context_t &context, sz_kernel_bytesum_t candidate) {
    std::mt19937 &generator = context.generator;
    sz_size_t const inputs = (sz_size_t)context.iterations_quadratic(200);
    auto reference_sum = [](sz_cptr_t text, sz_size_t length) {
        return kernel_result<sz_u64_t>(sz_bytesum_serial, text, length);
    };
    auto candidate_sum = [&](sz_cptr_t text, sz_size_t length) {
        return kernel_result<sz_u64_t>(candidate, text, length);
    };

    // A sum of bytes is order-independent, so a run of one repeated byte must total `length * byte`
    // on any backend. This invariant holds without consulting the reference at all.
    std::vector<std::size_t> const uniform_lengths = {0, 1, 63, 64, 65, 4096};
    for (auto length : uniform_lengths) {
        std::string const uniform(length, static_cast<char>(0xA5));
        verify(candidate_sum(uniform.data(), static_cast<sz_size_t>(length)) ==
                   static_cast<sz_u64_t>(length) * 0xA5ull &&
               "Byte sum of a uniform buffer must equal length times the repeated byte");
    }

    // The fixed ladder covers the sub-register, register, cache-line, and multi-block tiers; each
    // length is walked across cache-line offsets so the head and tail paths see every misalignment.
    std::vector<std::size_t> const lengths = {1, 11, 23, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000};
    for (auto length : lengths)
        for_each_cacheline_offset_(length, [&](sz_ptr_t pointer, [[maybe_unused]] std::size_t offset) {
            randomize_string(generator, {pointer, length});
            verify(reference_sum(pointer, static_cast<sz_size_t>(length)) ==
                       candidate_sum(pointer, static_cast<sz_size_t>(length)) &&
                   "Byte sum backend disagreed with the reference at this length and cache-line offset");
        });

    // Beyond the ladder, fuzz a contiguous run of random lengths at a single alignment.
    std::string text;
    for (sz_size_t length = 0; length != inputs; ++length) {
        text.resize(length);
        randomize_string(generator, text);
        verify(reference_sum(text.data(), length) == candidate_sum(text.data(), length) &&
               "Byte sum backend disagreed with the reference at this fuzzed length");
    }

    // One oversized input, since the Skylake and Ice Lake kernels take a different branch past a
    // megabyte. The trailing bytes keep the buffer off a page boundary so the head and tail still
    // have work to do.
    std::string huge(1024ull * 1024ull + 129ull, '\0');
    randomize_string(generator, huge);
    verify(reference_sum(huge.data(), (sz_size_t)huge.size()) == candidate_sum(huge.data(), (sz_size_t)huge.size()) &&
           "Byte sum backend disagreed with the reference on the oversized, past-a-megabyte input");
}

/**
 *  @brief Hashes strings through a candidate backend and serial, one-shot and streamed, expecting
 *      identical digests and identical states.
 *
 *  The test covers increasingly long and complex strings, starting with "abcabc..." repetitions and
 *  progressing towards corner cases like empty strings, all-zero inputs, zero seeds, and so on.
 */
inline void check_hash_equivalence_(test_context_t &context, hash_backend_t const &candidate) {
    std::mt19937 &generator = context.generator;
    sz_size_t const inputs = (sz_size_t)context.iterations_quadratic(200);
    hash_backend_t const reference = hash_serial_backend_();

    auto test_on_seed = [&](std::string const &text, sz_u64_t seed) {
        // Compute the entire hash at once, expecting the same output
        sz_u64_t result_base = reference(text.data(), text.size(), seed);
        sz_u64_t result_simd = candidate(text.data(), text.size(), seed);
        verify(result_base == result_simd && "Hash backend disagreed with the reference on the one-shot digest");

        // Compare incremental hashing across platforms
        sz_hash_state_t state_base, state_simd;
        reference.init(&state_base, seed);
        candidate.init(&state_simd, seed);
        verify(sz_hash_state_equal(&state_base, &state_base) == sz_true_k); // Self-equality
        verify(sz_hash_state_equal(&state_simd, &state_simd) == sz_true_k); // Self-equality
        verify(sz_hash_state_equal(&state_base, &state_simd) == sz_true_k); // Same across platforms

        // Let's also create an intentionally misaligned version of the state,
        // assuming some of the SIMD instructions may require alignment.
        sz_align_(64) char state_misaligned_buffer[sizeof(sz_hash_state_t) + 1];
        sz_hash_state_t &state_misaligned = *reinterpret_cast<sz_hash_state_t *>(state_misaligned_buffer + 1);
        candidate.init(&state_misaligned, seed);
        verify(sz_hash_state_equal(&state_base, &state_misaligned) == sz_true_k);

        // Try breaking those strings into arbitrary chunks, expecting the same output in the
        // streaming mode. The length of each chunk and the number of chunks will be determined with
        // a coin toss.
        iterate_in_random_slices(generator, text, [&](std::string slice) {
            reference.update(&state_base, slice.data(), slice.size());
            candidate.update(&state_simd, slice.data(), slice.size());
            verify(sz_hash_state_equal(&state_base, &state_simd) == sz_true_k);

            candidate.update(&state_misaligned, slice.data(), slice.size());
            verify(sz_hash_state_equal(&state_base, &state_misaligned) == sz_true_k);

            result_base = reference.digest(&state_base);
            result_simd = candidate.digest(&state_simd);
            verify(result_base == result_simd && "Hash backend disagreed with the reference after a streamed slice");
            sz_u64_t result_misaligned = candidate.digest(&state_misaligned);
            verify(result_base == result_misaligned && "Misaligned hash state disagreed with the aligned digest");
        });
    };

    // Let's try different-length strings repeating a "abc" pattern:
    std::vector<sz_u64_t> seeds = {
        0u,
        42u,                                  //
        std::numeric_limits<sz_u32_t>::max(), //
        std::numeric_limits<sz_u64_t>::max(), //
    };
    // A fixed repeat ladder: `inputs` already carries the multiplier, so scaling both compounds.
    for (auto seed : seeds)
        for (std::size_t copies = 1; copies != 100; ++copies) //
            test_on_seed(repeat("abc", copies), seed);

    // Let's try truly random inputs of different lengths, placing each input at every
    // sub-cache-line offset so serial-vs-ISA agreement is checked across all alignments the SIMD
    // kernels may hit.
    for (sz_size_t length = 0; length != inputs; ++length) {
        for_each_cacheline_offset_(length, [&](sz_ptr_t pointer, [[maybe_unused]] std::size_t offset) {
            randomize_string(generator, {pointer, length});
            std::string text(pointer, length);
            for (auto seed : seeds) test_on_seed(text, seed);
        });
    }
}

/**
 *  @brief Verifies a backend's batch multi-seed output equals a loop of its own single-seed hashes,
 *      across many lengths and seed counts, covering the 4-lane tail handling.
 *
 *  This is a single-backend self-consistency check: the candidate's @c multiseed must agree with
 *  its own @c hash_one for every seed, so a wrong shared constant in both is still caught against
 *  the per-seed reduction rather than a sibling backend. Every length is hashed by every seeded
 *  kernel, so the length count is the whole budget.
 */
inline void check_hash_multiseed_equivalence_(test_context_t &context, hash_multiseed_backend_t const &candidate) {
    std::mt19937 &generator = context.generator;
    sz_size_t const inputs = (sz_size_t)context.iterations_quadratic(512);

    // Enough seeds to exercise full 4-wide groups plus every 1..3-seed tail remainder.
    std::vector<sz_u64_t> seeds = {0u,
                                   1u,
                                   42u,
                                   314159u,
                                   std::numeric_limits<sz_u32_t>::max(),
                                   std::numeric_limits<sz_u64_t>::max(),
                                   7u,
                                   8u,
                                   9u,
                                   10u,
                                   11u,
                                   12u,
                                   13u,
                                   14u,
                                   15u,
                                   16u,
                                   17u};

    auto check = [&](std::string const &text) {
        for (std::size_t seed_count = 0; seed_count <= seeds.size(); ++seed_count) {
            // One guard slot past the end catches any write beyond `seed_count`.
            std::vector<sz_u64_t> output(seed_count + 1, 0xDEADBEEFDEADBEEFull);
            candidate.multiseed(text.data(), text.size(), seeds.data(), seed_count, output.data());
            for (std::size_t index = 0; index < seed_count; ++index)
                verify(output[index] == candidate.hash_one(text.data(), text.size(), seeds[index]) &&
                       "Multi-seed output disagreed with the single-seed hash at this seed index");
            verify(output[seed_count] == 0xDEADBEEFDEADBEEFull); // No overwrite past `seed_count`
        }
    };

    // Cover the minimal (<= 64 byte) ladder boundaries and the wide path, well past the 64-byte
    // tier and into a few kilobytes so the shared-input load over multiple blocks is exercised too.
    for (std::size_t length = 0; length != inputs; ++length) {
        std::string text(length, '\0');
        randomize_string(generator, text);
        check(text);
    }
}

/** Tests a Pseudo-Random Number Generator (PRNG), ensuring that the same nonce produces exactly the
 *  same output from the candidate as from serial. */
inline void check_fill_random_equivalence_(test_context_t &context, sz_kernel_fill_random_t candidate) {
    sz_size_t const inputs = (sz_size_t)context.iterations_quadratic(200);

    auto test_on_nonce = [&](std::size_t length, sz_u64_t nonce) {
        std::string text_base(length, '\0');
        std::string text_simd(length, '\0');
        verify(sz_fill_random_serial(&text_base[0], static_cast<sz_size_t>(length), nonce, nullptr) == sz_success_k);
        verify(candidate(&text_simd[0], static_cast<sz_size_t>(length), nonce, nullptr) == sz_success_k);
        verify(text_base == text_simd && "PRNG backend disagreed with the reference for this nonce and length");
    };

    // Boundary nonces are always exercised, including the 0 and max extremes:
    std::vector<sz_u64_t> nonces = {
        0u,
        42u,                                  //
        std::numeric_limits<sz_u32_t>::max(), //
        std::numeric_limits<sz_u64_t>::max(), //
    };

    // The fixed structured lengths cover the sub-cache-line, cache-line, and multi-block tiers;
    // every nonce is checked against all of them at multiplier 1.0.
    std::vector<std::size_t> lengths = {1, 11, 23, 37, 40, 51, 64, 128, 1000};
    for (auto nonce : nonces)
        for (auto length : lengths) //
            test_on_nonce(length, nonce);

    // Beyond the structured ladder, fuzz a contiguous run of random lengths.
    for (sz_size_t length = 0; length != inputs; ++length)
        for (auto nonce : nonces) //
            test_on_nonce(length, nonce);
}

/** Cross-checks a SHA256 backend against serial on random inputs, one-shot and incremental. The
 *  known-answer FIPS 180-4 vectors live in @c check_sha256_unit_. */
inline void check_sha256_equivalence_(test_context_t &context, sha256_backend_t const &candidate) {
    std::mt19937 &generator = context.generator;
    sz_size_t const inputs = (sz_size_t)context.iterations_quadratic(256);
    sha256_backend_t const reference {sz_sha256_state_init_serial, sz_sha256_state_update_serial,
                                      sz_sha256_state_digest_serial};

    // Test random inputs of various lengths
    for (sz_size_t length = 0; length <= inputs; ++length) {
        std::string random_text(length, '\0');
        randomize_string(generator, random_text);

        sz_sha256_state_t state_base, state_simd;
        sz_u8_t digest_base_result[STRINGZILLA_SHA256_DIGEST_LENGTH],
            digest_simd_result[STRINGZILLA_SHA256_DIGEST_LENGTH];

        // One-shot hashing
        reference.init(&state_base);
        candidate.init(&state_simd);
        reference.update(&state_base, random_text.data(), length);
        candidate.update(&state_simd, random_text.data(), length);
        reference.digest(&state_base, digest_base_result);
        candidate.digest(&state_simd, digest_simd_result);
        verify(std::memcmp(digest_base_result, digest_simd_result, STRINGZILLA_SHA256_DIGEST_LENGTH) == 0 &&
               "SHA256 backend disagreed with the reference on the one-shot digest at this length");

        // Incremental hashing with random chunks
        reference.init(&state_base);
        candidate.init(&state_simd);
        iterate_in_random_slices(generator, random_text, [&](std::string slice) {
            reference.update(&state_base, slice.data(), slice.size());
            candidate.update(&state_simd, slice.data(), slice.size());
        });
        reference.digest(&state_base, digest_base_result);
        candidate.digest(&state_simd, digest_simd_result);
        verify(std::memcmp(digest_base_result, digest_simd_result, STRINGZILLA_SHA256_DIGEST_LENGTH) == 0 &&
               "SHA256 backend disagreed with the reference on the incrementally streamed digest");
    }
}

/**
 *  @brief Compares a multi-state SHA256 backend with serial over batches of random-shaped messages.
 *
 *  Sweeps every lane count up to the bound, so batches that fall one lane short of a vector width
 *  take a different path through the kernel than batches that fill it. Each batch is fed twice:
 *  once in a single call, then again split into random per-lane slices, which is what carries a
 *  partial block across calls. Both digest buffers keep a guard lane past the end, since a batched
 *  kernel that miscounts lanes would otherwise corrupt the caller's memory silently.
 *
 *  Messages reach several blocks rather than the one a 64-byte bound would give, because lanes
 *  retire from the wide loop independently: a batch where every lane owns the same number of blocks
 *  never exercises the retirement order, the countdown, or the rule that parks a finished lane's
 *  cursor on its last full block. Each batch carries one message per lane, so the work is quadratic
 *  in the bound.
 */
inline void check_sha256_multistate_equivalence_(test_context_t &context,
                                                 sha256_multistate_backend_t const &candidate) {
    std::mt19937 &generator = context.generator;
    sz_size_t const inputs = (sz_size_t)context.iterations_quadratic(64);
    sha256_multistate_backend_t const reference {sz_sha256_multistate_update_serial,
                                                 sz_sha256_multistate_digest_serial};

    for (sz_size_t lanes_count = 0; lanes_count <= inputs; ++lanes_count) {
        std::vector<std::string> messages;
        fuzzy_config_t config;
        config.batch_size = (std::size_t)lanes_count;
        config.min_string_length = 0;
        config.max_string_length = (std::size_t)inputs * STRINGZILLA_SHA256_BLOCK_LENGTH / 8;
        randomize_strings(generator, config, messages);

        std::vector<sz_sha256_state_t> reference_states(lanes_count ? lanes_count : 1);
        std::vector<sz_sha256_state_t> candidate_states(lanes_count ? lanes_count : 1);
        std::vector<sz_u8_t> reference_digests((lanes_count + 1) * STRINGZILLA_SHA256_DIGEST_LENGTH, 0xA5);
        std::vector<sz_u8_t> candidate_digests((lanes_count + 1) * STRINGZILLA_SHA256_DIGEST_LENGTH, 0xA5);
        auto init_lanes = [&] {
            for (std::size_t lane_index = 0; lane_index != messages.size(); ++lane_index) {
                verify(sz_sha256_state_init_serial(&reference_states[lane_index], nullptr) == sz_success_k);
                verify(sz_sha256_state_init_serial(&candidate_states[lane_index], nullptr) == sz_success_k);
            }
        };
        init_lanes();

        // One-shot: every lane consumes its whole message in a single call
        sz_sequence_t const texts = sequence_from_(messages);
        reference.update(reference_states.data(), &texts);
        candidate.update(candidate_states.data(), &texts);
        reference.digest(reference_states.data(), lanes_count, reference_digests.data());
        candidate.digest(candidate_states.data(), lanes_count, candidate_digests.data());
        verify(std::memcmp(reference_digests.data(), candidate_digests.data(),
                           lanes_count * STRINGZILLA_SHA256_DIGEST_LENGTH) == 0 &&
               "Multi-state backend disagreed with the reference on the one-shot batch digest");

        // Incremental: the same messages, cut into random per-lane slices across several calls
        init_lanes();
        // The slices are windows into the messages the caller already holds, not copies, so the
        // kernels see a cursor advancing through one stable buffer, as a real caller streams.
        std::vector<std::size_t> offsets(messages.size(), 0);
        std::vector<sz_string_view_t> slices(messages.size());
        std::size_t remaining_lanes = messages.size();
        while (remaining_lanes != 0) {
            remaining_lanes = 0;
            for (std::size_t lane_index = 0; lane_index != messages.size(); ++lane_index) {
                std::size_t const left = messages[lane_index].size() - offsets[lane_index];
                std::uniform_int_distribution<std::size_t> slice_length_distribution(0, left);
                std::size_t const take = slice_length_distribution(generator);
                slices[lane_index].start = messages[lane_index].data() + offsets[lane_index];
                slices[lane_index].length = (sz_size_t)take;
                offsets[lane_index] += take;
                if (offsets[lane_index] != messages[lane_index].size()) ++remaining_lanes;
            }
            sz_sequence_t slice_texts;
            sz_sequence_from_string_views(slices.data(), slices.size(), &slice_texts);
            reference.update(reference_states.data(), &slice_texts);
            candidate.update(candidate_states.data(), &slice_texts);
        }
        reference.digest(reference_states.data(), lanes_count, reference_digests.data());
        candidate.digest(candidate_states.data(), lanes_count, candidate_digests.data());
        verify(std::memcmp(reference_digests.data(), candidate_digests.data(),
                           lanes_count * STRINGZILLA_SHA256_DIGEST_LENGTH) == 0 &&
               "Multi-state backend disagreed with the reference on the incrementally sliced batch digest");

        for (std::size_t guard_index = 0; guard_index != STRINGZILLA_SHA256_DIGEST_LENGTH;
             ++guard_index) // No overwrite past the last lane
            verify(candidate_digests[lanes_count * STRINGZILLA_SHA256_DIGEST_LENGTH + guard_index] == 0xA5);

        // Buffered head: each lane parks a different partial block, then the next call completes
        // some of them and not others. Random slicing reaches this only by luck, but it is the one
        // path where lanes compress out of a buffer rather than out of the caller's bytes, under a
        // mask of its own.
        init_lanes();
        for (std::size_t pass_index = 0; pass_index != 2; ++pass_index) {
            for (std::size_t lane_index = 0; lane_index != messages.size(); ++lane_index) {
                std::size_t const parked = 1 + lane_index % (STRINGZILLA_SHA256_BLOCK_LENGTH - 1);
                std::size_t const first = parked < messages[lane_index].size() ? parked : messages[lane_index].size();
                std::size_t const offset = pass_index == 0 ? 0 : first;
                std::size_t const length = pass_index == 0 ? first : messages[lane_index].size() - first;
                slices[lane_index].start = messages[lane_index].data() + offset;
                slices[lane_index].length = (sz_size_t)length;
            }
            sz_sequence_t parked_texts;
            sz_sequence_from_string_views(slices.data(), slices.size(), &parked_texts);
            reference.update(reference_states.data(), &parked_texts);
            candidate.update(candidate_states.data(), &parked_texts);
        }
        reference.digest(reference_states.data(), lanes_count, reference_digests.data());
        candidate.digest(candidate_states.data(), lanes_count, candidate_digests.data());
        verify(std::memcmp(reference_digests.data(), candidate_digests.data(),
                           lanes_count * STRINGZILLA_SHA256_DIGEST_LENGTH) == 0 &&
               "Multi-state backend disagreed with the reference on the buffered-head batch digest");
    }
}

#pragma endregion Hash

#pragma region Cipher

/** Decodes a hexadecimal literal into bytes, returning how many were written. */
inline std::size_t bytes_from_hex_(char const *hex, sz_u8_t *output) noexcept {
    auto nibble = [](char character) -> sz_u8_t {
        return (sz_u8_t)(character <= '9' ? character - '0' : (character | 32) - 'a' + 10);
    };
    std::size_t written = 0;
    for (; hex[written * 2] != '\0' && hex[written * 2 + 1] != '\0'; ++written)
        output[written] = (sz_u8_t)((nibble(hex[written * 2]) << 4) | nibble(hex[written * 2 + 1]));
    return written;
}

/** One published Galois/counter mode vector, spelled out rather than derived from any backend. */
struct known_gcm_t {
    char const *key_hex;
    char const *nonce_hex;
    char const *associated_hex;
    char const *plaintext_hex;
    char const *ciphertext_hex;
    char const *tag_hex;
};

/**
 *  @brief Cases 13 through 16 of McGrew and Viega's Galois/counter mode note, the ones NIST's own
 *      validation suite is built from.
 *
 *  Values longer than a line are split into adjacent literals at 32-byte boundaries, so the
 *  formatter has no reason to re-break them somewhere less readable.
 */
inline constexpr known_gcm_t known_gcm_vectors_[] = {
    // Empty message, empty associated data.
    {"0000000000000000000000000000000000000000000000000000000000000000", //
     "000000000000000000000000",                                         //
     "",                                                                 //
     "",                                                                 //
     "",                                                                 //
     "530f8afbc74536b9a963b4f1c4cb738b"},
    // One zero block, empty associated data.
    {"0000000000000000000000000000000000000000000000000000000000000000", //
     "000000000000000000000000",                                         //
     "",                                                                 //
     "00000000000000000000000000000000",                                 //
     "cea7403d4d606b6e074ec5d3baf39d18",                                 //
     "d0d1c8a799996bf0265b98b5d48ab919"},
    // Four whole blocks, empty associated data.
    {"feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308", //
     "cafebabefacedbaddecaf888",                                         //
     "",                                                                 //
     "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"  //
     "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255", //
     "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"  //
     "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad", //
     "b094dac5d93471bdec1a502270e3cc6c"},
    // A partial trailing block plus associated data, which exercises both zero pads.
    {"feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308", //
     "cafebabefacedbaddecaf888",                                         //
     "feedfacedeadbeeffeedfacedeadbeefabaddad2",                         //
     "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"  //
     "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",         //
     "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"  //
     "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",         //
     "76fc6ece0f4e1768cddf8853bb2d551b"},
};

/**
 *  @brief NIST SP 800-38A F.5.5, the AES-256 counter mode encryption vector.
 *
 *  That vector names a full 128-bit initial counter block incremented across its whole width, while
 *  @c sz_aes256_ctr_xor_best builds its counter from a twelve-byte nonce and a 32-bit block index
 *  starting at zero. The two coincide exactly when the stream is entered at the block index the
 *  published counter spells out, which is what the byte offset reaches, and the low 32 bits carry
 *  nowhere across four blocks.
 */
struct known_ctr_t {
    char const *key_hex;
    char const *nonce_hex;
    sz_u64_t byte_offset;
    char const *plaintext_hex;
    char const *ciphertext_hex;
};

inline constexpr known_ctr_t known_ctr_vectors_[] = {
    {"603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", //
     "f0f1f2f3f4f5f6f7f8f9fafb",                                         //
     (sz_u64_t)0xFCFDFEFFull * STRINGZILLA_AES_BLOCK_LENGTH,             //
     "6bc1bee22e409f96e93d7e117393172a"                                  //
     "ae2d8a571e03ac9c9eb76fac45af8e51"                                  //
     "30c81c46a35ce411e5fbc1191a0a52ef"                                  //
     "f69f2445df4f9b17ad2b417be66c3710",                                 //
     "601ec313775789a5b7a7f504bbf3d228"                                  //
     "f443e3ca4d62b59aca84e990cacaf5c5"                                  //
     "2b0930daa23de94ce87017ba2d84988d"                                  //
     "dfc9c58db67aada613c2dd08457941a6"},
};

/** One counter-mode backend, named so a failing check can say which one disagreed. */
struct ctr_backend_t {
    char const *name;
    sz_kernel_aes256_key_init_t key_init;
    sz_kernel_aes256_ctr_xor_t xor_bytes;
};

/** One authenticated backend, one-shot and streaming kernels alike. */
struct gcm_backend_t {
    char const *name;
    sz_kernel_aes256_gcm_key_init_t key_init;
    sz_kernel_aes256_gcm_encrypt_t encrypt;
    sz_kernel_aes256_gcm_decrypt_t decrypt;
    sz_kernel_aes256_gcm_encryptor_init_t sealer_init;
    sz_kernel_aes256_gcm_encryptor_associate_t sealer_associate;
    sz_kernel_aes256_gcm_encryptor_update_t sealer_update;
    sz_kernel_aes256_gcm_encryptor_digest_t sealer_digest;
    sz_kernel_aes256_gcm_decryptor_init_t opener_init;
    sz_kernel_aes256_gcm_decryptor_associate_t opener_associate;
    sz_kernel_aes256_gcm_decryptor_update_unverified_t opener_update;
    sz_kernel_aes256_gcm_decryptor_verify_t opener_verify;
};

/** The serial counter-mode kernels, every other backend's reference. */
inline ctr_backend_t ctr_serial_backend_() noexcept {
    return {"serial", sz_aes256_key_init_serial, sz_aes256_ctr_xor_serial};
}

/** The serial authenticated kernels, every other backend's reference. */
inline gcm_backend_t gcm_serial_backend_() noexcept {
    return {"serial",
            sz_aes256_gcm_key_init_serial,
            sz_aes256_gcm_encrypt_serial,
            sz_aes256_gcm_decrypt_serial,
            sz_aes256_gcm_encryptor_init_serial,
            sz_aes256_gcm_encryptor_associate_serial,
            sz_aes256_gcm_encryptor_update_serial,
            sz_aes256_gcm_encryptor_digest_serial,
            sz_aes256_gcm_decryptor_init_serial,
            sz_aes256_gcm_decryptor_associate_serial,
            sz_aes256_gcm_decryptor_update_unverified_serial,
            sz_aes256_gcm_decryptor_verify_serial};
}

/** Checks one counter-mode backend against a published vector, seeking to its block index. */
inline void check_ctr_unit_(ctr_backend_t const &backend, known_ctr_t const &vector) {
    sz_u8_t secret[32], nonce[12];
    std::vector<sz_u8_t> plaintext(64), expected(64), produced(64), recovered(64);
    bytes_from_hex_(vector.key_hex, secret);
    bytes_from_hex_(vector.nonce_hex, nonce);
    std::size_t const length = bytes_from_hex_(vector.plaintext_hex, plaintext.data());
    bytes_from_hex_(vector.ciphertext_hex, expected.data());

    sz_aes256_key_t key;
    verify(backend.key_init(&key, secret, nullptr) == sz_success_k);
    verify(backend.xor_bytes(&key, nonce, vector.byte_offset, (sz_cptr_t)plaintext.data(), (sz_size_t)length,
                             (sz_ptr_t)produced.data(), nullptr) == sz_success_k);
    if (std::memcmp(produced.data(), expected.data(), length) != 0)
        fail_backend_(backend.name, "counter mode ciphertext differs from NIST SP 800-38A F.5.5");

    // Counter mode is its own inverse, so the same call must walk the vector back.
    verify(backend.xor_bytes(&key, nonce, vector.byte_offset, (sz_cptr_t)produced.data(), (sz_size_t)length,
                             (sz_ptr_t)recovered.data(), nullptr) == sz_success_k);
    if (std::memcmp(recovered.data(), plaintext.data(), length) != 0)
        fail_backend_(backend.name, "counter mode did not recover its own plaintext");
}

/** Checks one authenticated backend on a published vector, both directions and a forged tag. */
inline void check_gcm_unit_(gcm_backend_t const &backend, known_gcm_t const &vector) {
    sz_u8_t secret[32], nonce[12], associated[64], expected_tag[16], produced_tag[16];
    std::vector<sz_u8_t> plaintext(256), expected(256), produced(256), recovered(256);
    bytes_from_hex_(vector.key_hex, secret);
    bytes_from_hex_(vector.nonce_hex, nonce);
    std::size_t const associated_length = bytes_from_hex_(vector.associated_hex, associated);
    std::size_t const length = bytes_from_hex_(vector.plaintext_hex, plaintext.data());
    bytes_from_hex_(vector.ciphertext_hex, expected.data());
    bytes_from_hex_(vector.tag_hex, expected_tag);

    sz_aes256_gcm_key_t key;
    verify(backend.key_init(&key, secret, nullptr) == sz_success_k);
    verify(backend.encrypt(&key, nonce, (sz_cptr_t)associated, (sz_size_t)associated_length,
                           (sz_cptr_t)plaintext.data(), (sz_size_t)length, (sz_ptr_t)produced.data(), produced_tag,
                           nullptr) == sz_success_k);
    if (std::memcmp(produced.data(), expected.data(), length) != 0)
        fail_backend_(backend.name, "authenticated ciphertext differs from the published vector");
    if (std::memcmp(produced_tag, expected_tag, 16) != 0)
        fail_backend_(backend.name, "authentication tag differs from the published vector");

    // The same vector must decrypt back, and a single flipped tag bit must be refused.
    if (backend.decrypt(&key, nonce, (sz_cptr_t)associated, (sz_size_t)associated_length, (sz_cptr_t)produced.data(),
                        (sz_size_t)length, (sz_ptr_t)recovered.data(), produced_tag, nullptr) != sz_success_k)
        fail_backend_(backend.name, "a genuine tag was refused");
    if (std::memcmp(recovered.data(), plaintext.data(), length) != 0)
        fail_backend_(backend.name, "decryption did not recover the published plaintext");

    produced_tag[0] ^= 0x01;
    for (std::size_t index = 0; index != recovered.size(); ++index) recovered[index] = 0xA5;
    if (backend.decrypt(&key, nonce, (sz_cptr_t)associated, (sz_size_t)associated_length, (sz_cptr_t)produced.data(),
                        (sz_size_t)length, (sz_ptr_t)recovered.data(), produced_tag,
                        nullptr) != sz_authentication_failed_k)
        fail_backend_(backend.name, "a forged tag was accepted");
    // A caller who drops the status must not find forged plaintext waiting in the buffer.
    for (std::size_t index = 0; index != length; ++index)
        if (recovered[index] != 0) fail_backend_(backend.name, "forged plaintext survived a failed authentication");
}

/**
 *  @brief Holds one backend's counter-mode and authenticated kernels to literal expectations only.
 *
 *  The counter-mode vector is NIST SP 800-38A F.5.5 and the four authenticated vectors are cases 13
 *  through 16 of McGrew and Viega's Galois/counter mode note, the ones NIST's own validation suite
 *  is built from. Nothing here is derived by calling another backend, so no shared mistake hides.
 */
inline void check_cipher_unit_(ctr_backend_t const &ctr, gcm_backend_t const &gcm) {
    for (known_ctr_t const &vector : span_over(known_ctr_vectors_)) check_ctr_unit_(ctr, vector);
    for (known_gcm_t const &vector : span_over(known_gcm_vectors_)) check_gcm_unit_(gcm, vector);
}

/**
 *  @brief Cross-checks a counter-mode backend against a reference across lengths and seek offsets.
 *
 *  Seeking is the whole reason counter mode is exposed separately, so every offset is compared
 *  against the same bytes taken from a from-zero encryption, not just the reference backend.
 */
inline void check_ctr_equivalence_(std::mt19937 &generator, ctr_backend_t const &reference,
                                   ctr_backend_t const &candidate, sz_size_t inputs) {
    sz_u8_t secret[32], nonce[12];
    for (std::size_t index = 0; index != 32; ++index) secret[index] = (sz_u8_t)(index * 7 + 1);
    for (std::size_t index = 0; index != 12; ++index) nonce[index] = (sz_u8_t)(index * 5 + 2);

    sz_aes256_key_t reference_key, candidate_key;
    verify(reference.key_init(&reference_key, secret, nullptr) == sz_success_k);
    verify(candidate.key_init(&candidate_key, secret, nullptr) == sz_success_k);
    if (std::memcmp(&reference_key, &candidate_key, sizeof(reference_key)) != 0)
        fail_backend_(candidate.name, "expanded a different round-key schedule than serial");

    std::string text, from_reference, from_candidate;
    for (sz_size_t length = 0; length <= inputs; ++length) {
        text.resize(length), from_reference.resize(length), from_candidate.resize(length);
        randomize_string(generator, text);
        verify(reference.xor_bytes(&reference_key, nonce, 0, text.data(), length, &from_reference[0], nullptr) ==
               sz_success_k);
        verify(candidate.xor_bytes(&candidate_key, nonce, 0, text.data(), length, &from_candidate[0], nullptr) ==
               sz_success_k);
        if (from_reference != from_candidate) fail_backend_(candidate.name, "counter mode disagreed with serial");
    }

    // Every offset must land on the same keystream the whole-stream encryption used.
    std::size_t const span = 1024;
    std::string whole(span, '\0'), sliced;
    randomize_string(generator, whole);
    std::string whole_out(span, '\0');
    verify(reference.xor_bytes(&reference_key, nonce, 0, whole.data(), span, &whole_out[0], nullptr) == sz_success_k);
    for (std::size_t offset = 0; offset <= 200; ++offset) {
        sliced.assign(span - offset, '\0');
        verify(candidate.xor_bytes(&candidate_key, nonce, offset, whole.data() + offset, span - offset, &sliced[0],
                                   nullptr) == sz_success_k);
        if (std::memcmp(sliced.data(), whole_out.data() + offset, span - offset) != 0)
            fail_backend_(candidate.name, "seeking into the keystream landed on different bytes");

        // Passing one pointer for both sides is what lets a caller transform a buffer without a
        // copy. The seeked entry is the interesting one: its head block is the byte the kernel is
        // likeliest to overwrite before it has finished reading.
        std::string aliased(whole, offset, span - offset);
        verify(candidate.xor_bytes(&candidate_key, nonce, offset, aliased.data(), span - offset, &aliased[0],
                                   nullptr) == sz_success_k);
        if (std::memcmp(aliased.data(), whole_out.data() + offset, span - offset) != 0)
            fail_backend_(candidate.name, "counter mode in place differs from out-of-place");
    }
}

/**
 *  @brief Cross-checks an authenticated backend against a reference, one-shot and in chunks.
 *  @param[in] inputs The longest message fuzzed, inclusive.
 *
 *  Also asserts the aliasing permission the header grants, which holds independently of any
 *  reference: passing one pointer for both sides reaches the bytes and the tag two pointers would,
 *  and a rejected tag still clears the buffer it was handed.
 */
inline void check_gcm_equivalence_(std::mt19937 &generator, gcm_backend_t const &reference,
                                   gcm_backend_t const &candidate, sz_size_t inputs) {
    sz_u8_t secret[32], nonce[12], reference_tag[16], candidate_tag[16];
    for (std::size_t index = 0; index != 32; ++index) secret[index] = (sz_u8_t)(index * 3 + 5);
    for (std::size_t index = 0; index != 12; ++index) nonce[index] = (sz_u8_t)(index + 9);

    sz_aes256_gcm_key_t reference_key, candidate_key;
    verify(reference.key_init(&reference_key, secret, nullptr) == sz_success_k);
    verify(candidate.key_init(&candidate_key, secret, nullptr) == sz_success_k);
    if (std::memcmp(&reference_key, &candidate_key, sizeof(reference_key)) != 0)
        fail_backend_(candidate.name, "expanded a different schedule or different subkey powers than serial");

    std::vector<std::size_t> const associated_lengths = {0, 1, 15, 16, 17, 40};
    std::string text, from_reference, from_candidate, associated;

    for (sz_size_t length = 0; length <= inputs; ++length) {
        text.resize(length), from_reference.resize(length), from_candidate.resize(length);
        randomize_string(generator, text);
        std::size_t const associated_length = associated_lengths[length % associated_lengths.size()];
        associated.resize(associated_length);
        randomize_string(generator, associated);

        verify(reference.encrypt(&reference_key, nonce, associated.data(), (sz_size_t)associated_length, text.data(),
                                 length, &from_reference[0], reference_tag, nullptr) == sz_success_k);
        verify(candidate.encrypt(&candidate_key, nonce, associated.data(), (sz_size_t)associated_length, text.data(),
                                 length, &from_candidate[0], candidate_tag, nullptr) == sz_success_k);
        if (from_reference != from_candidate) fail_backend_(candidate.name, "ciphertext disagreed with serial");
        if (std::memcmp(reference_tag, candidate_tag, 16) != 0)
            fail_backend_(candidate.name, "authentication tag disagreed with serial");

        // Decryption must recover the plaintext and report success on the genuine tag.
        std::string recovered(length, '\0');
        if (candidate.decrypt(&candidate_key, nonce, associated.data(), (sz_size_t)associated_length,
                              from_candidate.data(), length, &recovered[0], candidate_tag, nullptr) != sz_success_k)
            fail_backend_(candidate.name, "refused a tag it had just produced");
        if (recovered != text) fail_backend_(candidate.name, "decryption did not recover its own plaintext");

        // Sealing and opening inside the buffer a record arrived in is what lets a caller skip a
        // copy per message, so one pointer for both sides must reach the bytes two pointers would.
        std::string aliased = text;
        sz_u8_t aliased_tag[16];
        verify(candidate.encrypt(&candidate_key, nonce, associated.data(), (sz_size_t)associated_length, aliased.data(),
                                 length, &aliased[0], aliased_tag, nullptr) == sz_success_k);
        if (aliased != from_candidate) fail_backend_(candidate.name, "in-place sealing differs from out-of-place");
        if (std::memcmp(aliased_tag, candidate_tag, 16) != 0)
            fail_backend_(candidate.name, "in-place sealing produced a different tag than out-of-place");

        if (candidate.decrypt(&candidate_key, nonce, associated.data(), (sz_size_t)associated_length, aliased.data(),
                              length, &aliased[0], candidate_tag, nullptr) != sz_success_k)
            fail_backend_(candidate.name, "in-place opening refused a tag it had just produced");
        if (aliased != text) fail_backend_(candidate.name, "in-place opening recovered the wrong plaintext");

        // A rejected tag clears the buffer even when that buffer is the ciphertext itself, so a
        // caller who drops the status cannot act on forged data it opened in place.
        aliased = from_candidate;
        aliased_tag[0] = (sz_u8_t)(candidate_tag[0] ^ 0x01);
        if (candidate.decrypt(&candidate_key, nonce, associated.data(), (sz_size_t)associated_length, aliased.data(),
                              length, &aliased[0], aliased_tag, nullptr) != sz_authentication_failed_k)
            fail_backend_(candidate.name, "in-place opening accepted a forged tag");
        for (std::size_t index = 0; index != length; ++index)
            if (aliased[index] != 0) fail_backend_(candidate.name, "forged plaintext survived opening in place");
    }

    // A chunk boundary must be invisible: the keystream block, the hash block, and the
    // associated-data tail all straddle it, and the associated data is chunked on its own rhythm
    // rather than the text's.
    std::size_t const streamed_length = 512, streamed_associated_length = 37;
    text.resize(streamed_length);
    randomize_string(generator, text);
    associated.resize(streamed_associated_length);
    randomize_string(generator, associated);
    from_reference.assign(streamed_length, '\0');
    verify(reference.encrypt(&reference_key, nonce, associated.data(), (sz_size_t)streamed_associated_length,
                             text.data(), streamed_length, &from_reference[0], reference_tag, nullptr) == sz_success_k);
    for (std::size_t chunk = 1; chunk <= 40; ++chunk) {
        std::string streamed(streamed_length, '\0');
        sz_aes256_gcm_encryptor_t encryptor;
        verify(candidate.sealer_init(&encryptor, &candidate_key, nonce, nullptr) == sz_success_k);
        for (std::size_t offset = 0; offset < streamed_associated_length; offset += chunk) {
            std::size_t const taken = streamed_associated_length - offset < chunk ? streamed_associated_length - offset
                                                                                  : chunk;
            verify(candidate.sealer_associate(&encryptor, associated.data() + offset, (sz_size_t)taken, nullptr) ==
                   sz_success_k);
        }
        for (std::size_t offset = 0; offset < streamed_length; offset += chunk) {
            std::size_t const taken = streamed_length - offset < chunk ? streamed_length - offset : chunk;
            verify(candidate.sealer_update(&encryptor, text.data() + offset, (sz_size_t)taken, &streamed[offset],
                                           nullptr) == sz_success_k);
        }
        verify(candidate.sealer_digest(&encryptor, candidate_tag, nullptr) == sz_success_k);
        if (streamed != from_reference) fail_backend_(candidate.name, "chunked sealing differs from one-shot");
        if (std::memcmp(reference_tag, candidate_tag, 16) != 0)
            fail_backend_(candidate.name, "chunked sealing produced a different tag than one-shot");

        // The opposite direction is a separate type, so it needs its own pass over the chunking.
        std::string reopened(streamed_length, '\0');
        sz_aes256_gcm_decryptor_t decryptor;
        verify(candidate.opener_init(&decryptor, &candidate_key, nonce, nullptr) == sz_success_k);
        for (std::size_t offset = 0; offset < streamed_associated_length; offset += chunk) {
            std::size_t const taken = streamed_associated_length - offset < chunk ? streamed_associated_length - offset
                                                                                  : chunk;
            verify(candidate.opener_associate(&decryptor, associated.data() + offset, (sz_size_t)taken, nullptr) ==
                   sz_success_k);
        }
        for (std::size_t offset = 0; offset < streamed_length; offset += chunk) {
            std::size_t const taken = streamed_length - offset < chunk ? streamed_length - offset : chunk;
            verify(candidate.opener_update(&decryptor, from_reference.data() + offset, (sz_size_t)taken,
                                           &reopened[offset], nullptr) == sz_success_k);
        }
        if (reopened != text) fail_backend_(candidate.name, "chunked opening did not recover the plaintext");
        if (candidate.opener_verify(&decryptor, reference_tag, nullptr) != sz_success_k)
            fail_backend_(candidate.name, "chunked opening refused a genuine tag");

        // The streaming entries carry the same aliasing permission, and their mid-chunk resume path
        // is where a kernel is likeliest to read a byte it has already overwritten.
        std::string aliased = text;
        sz_aes256_gcm_encryptor_t aliasing_encryptor;
        verify(candidate.sealer_init(&aliasing_encryptor, &candidate_key, nonce, nullptr) == sz_success_k);
        verify(candidate.sealer_associate(&aliasing_encryptor, associated.data(), (sz_size_t)streamed_associated_length,
                                          nullptr) == sz_success_k);
        for (std::size_t offset = 0; offset < streamed_length; offset += chunk) {
            std::size_t const taken = streamed_length - offset < chunk ? streamed_length - offset : chunk;
            verify(candidate.sealer_update(&aliasing_encryptor, aliased.data() + offset, (sz_size_t)taken,
                                           &aliased[offset], nullptr) == sz_success_k);
        }
        verify(candidate.sealer_digest(&aliasing_encryptor, candidate_tag, nullptr) == sz_success_k);
        if (aliased != from_reference) fail_backend_(candidate.name, "chunked sealing in place differs from one-shot");
        if (std::memcmp(reference_tag, candidate_tag, 16) != 0)
            fail_backend_(candidate.name, "chunked sealing in place produced a different tag");

        sz_aes256_gcm_decryptor_t aliasing_decryptor;
        verify(candidate.opener_init(&aliasing_decryptor, &candidate_key, nonce, nullptr) == sz_success_k);
        verify(candidate.opener_associate(&aliasing_decryptor, associated.data(), (sz_size_t)streamed_associated_length,
                                          nullptr) == sz_success_k);
        for (std::size_t offset = 0; offset < streamed_length; offset += chunk) {
            std::size_t const taken = streamed_length - offset < chunk ? streamed_length - offset : chunk;
            verify(candidate.opener_update(&aliasing_decryptor, aliased.data() + offset, (sz_size_t)taken,
                                           &aliased[offset], nullptr) == sz_success_k);
        }
        if (aliased != text) fail_backend_(candidate.name, "chunked opening in place did not recover the plaintext");
        if (candidate.opener_verify(&aliasing_decryptor, reference_tag, nullptr) != sz_success_k)
            fail_backend_(candidate.name, "chunked opening in place refused a genuine tag");
    }
}

/**
 *  @brief Holds one capability's counter-mode and authenticated kernels to serial's, byte for byte.
 *
 *  Serial is the reference for everything, itself included: running it against itself catches a
 *  streaming path that disagrees with its own one-shot kernel. Each length sweeps a fresh buffer,
 *  so the work grows with the square of the count.
 */
inline void check_cipher_equivalence_(test_context_t &context, ctr_backend_t const &ctr, gcm_backend_t const &gcm) {
    sz_size_t const inputs = (sz_size_t)context.iterations_quadratic(160);
    check_ctr_equivalence_(context.generator, ctr_serial_backend_(), ctr, inputs);
    check_gcm_equivalence_(context.generator, gcm_serial_backend_(), gcm, inputs);
}

#pragma endregion Cipher

#pragma region Sort

/** One backend's byte and uncased sequence arg-sort kernels. */
struct sort_backend_t {
    char const *name;
    sz_kernel_sequence_argsort_t argsort;
    sz_kernel_sequence_argsort_t argsort_uncased;
};

/** Runs one arg-sort kernel over @c sequence, asserting the permutation matches @c expected. */
inline void check_argsort_order_(sz_kernel_sequence_argsort_t argsort, sz_sequence_t const *sequence,
                                 std::vector<sz_sorted_idx_t> const &expected) {
    std::vector<sz_sorted_idx_t> order(expected.size());
    handle_checked_heap_t heap;
    verify(argsort(sequence, 0, sz_false_k, &heap.allocator, order.data(), nullptr) == sz_success_k &&
           "Kernel call failed");
    verify(heap.live_allocations == 0);
    verify(order == expected);
}

/** Holds one backend to known orders: {"banana","apple","cherry"} sorts bytewise to {1, 0, 2}, and
 *  {"Banana","apple"} case-folds to {"banana","apple"}, ordering them {1, 0}. */
inline void check_sort_unit_(sort_backend_t const &backend) {
    std::vector<std::string> const fruits = {"banana", "apple", "cherry"};
    sz_sequence_t const fruits_sequence = sequence_from_(fruits);
    check_argsort_order_(backend.argsort, &fruits_sequence, {1u, 0u, 2u});

    std::vector<std::string> const words = {"Banana", "apple"};
    sz_sequence_t const words_sequence = sequence_from_(words);
    check_argsort_order_(backend.argsort_uncased, &words_sequence, {1u, 0u});
}

/**
 *  @brief Degenerate sequences for one backend's arg-sorts, whose output must stay a permutation.
 *
 *  An empty sequence, a single element, and one where every string is identical each have a
 *  defensible answer, and what is asserted here is the shape of the reply rather than its order:
 *  the output must be a permutation of the input indices, every index present exactly once. An
 *  all-equal input is the one that catches a comparator returning a strict order where it should
 *  report a tie, since any ordering of it looks sorted and only the permutation property fails.
 */
inline void check_sort_safety_(sort_backend_t const &backend) {
    using strs_t = std::vector<std::string>;
    auto check_is_permutation_ = [&](sz_kernel_sequence_argsort_t argsort, strs_t const &input) {
        sz_sequence_t const sequence = sequence_from_(input);
        std::vector<sz_sorted_idx_t> order(input.size());
        handle_checked_heap_t heap;
        verify(argsort(&sequence, 0, sz_false_k, &heap.allocator, order.data(), nullptr) == sz_success_k &&
               "Kernel call failed");
        verify(heap.live_allocations == 0);
        std::vector<bool> seen(input.size(), false);
        for (sz_sorted_idx_t const index : order) {
            if ((std::size_t)index >= input.size() || seen[(std::size_t)index]) {
                fmt::println(stderr, "{}: argsort produced {} for a {}-element input", backend.name,
                             (std::size_t)index >= input.size() ? "an out-of-range index" : "a repeated index",
                             input.size());
                verify(false && "A sort's output must be a permutation of the input indices");
            }
            seen[(std::size_t)index] = true;
        }
    };

    strs_t degenerate_inputs[] = {
        strs_t {},                                  // Empty sequence
        strs_t {"only"},                            // One element
        strs_t(17, "same"),                         // All equal, so only the shape can fail
        strs_t(129, "same"),                        // Past the insertion-sort cutover, still equal
        strs_t {"", "a", "", "aa", "a", "", "aaa"}, // Empty strings and prefixes, tied by length
    };
    // Differing only past an embedded NUL, which a length-truncating comparison would call equal.
    strs_t embedded;
    embedded.push_back(std::string("a\0b", 3));
    embedded.push_back(std::string("a\0a", 3));

    for (sz_kernel_sequence_argsort_t argsort : {backend.argsort, backend.argsort_uncased}) {
        for (strs_t const &input : span_over(degenerate_inputs)) check_is_permutation_(argsort, input);
        check_is_permutation_(argsort, embedded);
    }
}

/**
 *  @brief Demands a candidate sort backend produce results identical to serial.
 *
 *  Both the byte and uncased arg-sorts are @b stable, so for any input the permutation is unique -
 *  the candidate and reference @c order arrays must match exactly across the ascending, descending,
 *  and top-K modes. Four repetitions at scale 1.0, one per top-K mode.
 */
inline void check_sort_equivalence_(test_context_t &context, sort_backend_t const &candidate) {
    std::size_t const repetition_count = context.iterations(4);
    sort_backend_t const reference {"serial", sz_sequence_argsort_serial, sz_sequence_argsort_uncased_serial};

    using strs_t = std::vector<std::string>;
    std::mt19937 &generator = context.generator;
    handle_checked_heap_t heap;

    // Each repetition draws fresh random datasets and covers one top-K mode, so a larger count
    // widens the fuzzing coverage rather than enlarging a fixed dataset. The datasets span the
    // vectorized block, the scalar tail, and the slack-region boundaries, and the largest crosses
    // into the large-input partitioning path.
    for (std::size_t repetition = 0; repetition < repetition_count; ++repetition) {
        std::vector<strs_t> datasets;
        for (std::size_t full_count : {33u, 64u, 100u, 1000u, 5000u, 100000u}) {
            // Only the repetition count rides the multiplier; scaling the sizes too would make the
            // dial multiplicative, so a 10x run would cost a hundredfold. Never below 33, so the
            // QuickSort partition rather than the insertion-sort fallback keeps running.
            std::size_t const count = std::max<std::size_t>(33, full_count);
            // Short strings over a tiny alphabet, so many exact duplicates fill the equal region.
            strs_t fixed_dups;
            for (std::size_t i = 0; i < count; ++i) fixed_dups.push_back(random_string(generator, i % 5, "ab"));
            datasets.push_back(fixed_dups);
            strs_t varied; // Longer, common-prefix strings => deep pgram recursion.
            for (std::size_t i = 0; i < count; ++i) varied.push_back(random_string(generator, 6 + i % 40, "abc"));
            datasets.push_back(varied);
        }
        { // Deterministic mixed-case / multi-script set so the uncased path sees real folds.
            char const *seed[] = {"Apple",   "apple",  "BANANA", "banana", "Straße",
                                  "STRASSE", "Привет", "ПРИВЕТ", "Ab",     "aB"};
            strs_t mixed;
            for (std::size_t r = 0; r < 50; ++r)
                for (char const *word : seed) mixed.push_back(word);
            std::shuffle(mixed.begin(), mixed.end(), generator);
            datasets.push_back(mixed);
        }

        for (strs_t const &dataset : datasets) {
            std::size_t const count = dataset.size();
            sz_sequence_t const sequence = sequence_from_(dataset);

            std::vector<sz_sorted_idx_t> order_reference(count), order_candidate(count);
            sz_size_t const top_modes[] = {0, 1, (sz_size_t)(count / 3), (sz_size_t)count};
            sz_size_t const top = top_modes[repetition % 4];
            for (sz_bool_t reverse : {sz_false_k, sz_true_k}) {
                std::size_t const head = (top != 0 && top < count) ? top : count;

                // Byte arg-sort is stable, so the permutations must match exactly over the prefix.
                verify(reference.argsort(&sequence, top, reverse, &heap.allocator, order_reference.data(), nullptr) ==
                       sz_success_k);
                verify(candidate.argsort(&sequence, top, reverse, &heap.allocator, order_candidate.data(), nullptr) ==
                       sz_success_k);
                for (std::size_t i = 0; i < head; ++i)
                    verify(order_reference[i] == order_candidate[i] && "SIMD byte arg-sort disagrees with serial");

                // Uncased arg-sort: also stable, same exact-match requirement.
                verify(reference.argsort_uncased(&sequence, top, reverse, &heap.allocator, order_reference.data(),
                                                 nullptr) == sz_success_k);
                verify(candidate.argsort_uncased(&sequence, top, reverse, &heap.allocator, order_candidate.data(),
                                                 nullptr) == sz_success_k);
                for (std::size_t i = 0; i < head; ++i)
                    verify(order_reference[i] == order_candidate[i] && "SIMD uncased arg-sort disagrees with serial");
            }
        }
    }
    verify(heap.live_allocations == 0);
}

#pragma endregion Sort

#pragma region Intersect

/** One matched pair from an intersection: @c first_index into the first sequence, @c second_index
 *  into the second. */
struct intersect_match_t {
    std::size_t first_index;
    std::size_t second_index;

    bool operator<(intersect_match_t const &other) const noexcept {
        return first_index != other.first_index ? first_index < other.first_index : second_index < other.second_index;
    }
    bool operator==(intersect_match_t const &other) const noexcept {
        return first_index == other.first_index && second_index == other.second_index;
    }
};

/** Runs one intersect kernel over both inputs and compares the matched index pairs as a set, since
 *  their order is unspecified. */
inline void check_intersect_pairs_(sz_kernel_sequence_intersect_t intersect, std::vector<std::string> const &first,
                                   std::vector<std::string> const &second,
                                   std::set<intersect_match_t> const &expected_pairs) {
    sz_sequence_t const first_sequence = sequence_from_(first);
    sz_sequence_t const second_sequence = sequence_from_(second);
    sz_size_t const capacity = std::min(first.size(), second.size());
    std::vector<sz_sorted_idx_t> first_positions(capacity), second_positions(capacity);
    sz_size_t intersection_count = 0;
    handle_checked_heap_t heap;
    verify(intersect(&first_sequence, &second_sequence, &heap.allocator, 0u, &intersection_count, //
                     first_positions.data(), second_positions.data(), nullptr) == sz_success_k &&
           "Kernel call failed");
    verify(heap.live_allocations == 0);
    verify(intersection_count == expected_pairs.size() && "Kernel reported the wrong intersection size");
    std::set<intersect_match_t> produced;
    for (sz_size_t index = 0; index != intersection_count; ++index)
        produced.insert({(std::size_t)first_positions[index], (std::size_t)second_positions[index]});
    verify(produced == expected_pairs);
}

/** Holds one intersect kernel to known pairs: a shared pair of fruits, an identity, a reversal, and
 *  two sets sharing a prefix. */
inline void check_intersect_unit_(sz_kernel_sequence_intersect_t intersect) {
    using strs_t = std::vector<std::string>;
    // Banana sits at (1, 2) and cherry at (2, 0).
    check_intersect_pairs_(intersect, strs_t {"apple", "banana", "cherry"}, strs_t {"cherry", "date", "banana"},
                           {{1u, 2u}, {2u, 0u}});
    strs_t const abcd {"a", "b", "c", "d"}, dcba {"d", "c", "b", "a"}, abs {"a", "b", "s"};
    check_intersect_pairs_(intersect, abcd, abcd, {{0u, 0u}, {1u, 1u}, {2u, 2u}, {3u, 3u}});
    check_intersect_pairs_(intersect, abcd, dcba, {{0u, 3u}, {1u, 2u}, {2u, 1u}, {3u, 0u}});
    check_intersect_pairs_(intersect, abcd, abs, {{0u, 0u}, {1u, 1u}});
}

#pragma endregion Intersect

#pragma region Levenshtein

/** Textbook O(n · m) Levenshtein over any symbol sequence, the oracle every backend is held to. */
template <typename symbols_type_>
std::size_t levenshtein_reference_(symbols_type_ const &first, symbols_type_ const &second) {
    std::vector<std::size_t> previous(second.size() + 1), current(second.size() + 1);
    for (std::size_t second_position = 0; second_position <= second.size(); ++second_position)
        previous[second_position] = second_position;
    for (std::size_t first_position = 1; first_position <= first.size(); ++first_position) {
        current[0] = first_position;
        for (std::size_t second_position = 1; second_position <= second.size(); ++second_position) {
            std::size_t const substitution = previous[second_position - 1] +
                                             (first[first_position - 1] != second[second_position - 1]);
            current[second_position] = std::min(
                {previous[second_position] + 1, current[second_position - 1] + 1, substitution});
        }
        std::swap(previous, current);
    }
    return previous[second.size()];
}

/** The rune sequence the UTF-8 entries score, decoded here on the rune codec alone so the oracle
 *  shares nothing with the kernel: an ill-formed byte is one @c U+FFFD. */
inline std::u32string levenshtein_runes_(std::string const &text) {
    std::u32string runes;
    for (std::size_t position = 0; position < text.size();) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode(text.data() + position, text.data() + text.size(), &rune);
        if (consumed == sz_rune_invalid_k) runes.push_back(sz_rune_replacement_k), ++position;
        else runes.push_back(rune), position += consumed;
    }
    return runes;
}

/** One backend's engine builder and the cross-product verb it then scores. */
struct levenshtein_backend_t {

    /** The row's spelling, for @ref fail_backend_ and the log. */
    char const *name;

    /** Prepares a batch of queries, over bytes or over runes. */
    sz_kernel_levenshtein_engine_init_t init;

    /** The verb, over bytes or over runes as the engine was built. */
    sz_kernel_levenshtein_distances_t distances;
};

/** One prepared batch, released with the scope that named it. */
struct levenshtein_engine_t {
    handle_checked_heap_t heap;
    sz_levenshtein_engine_t engine {};

    levenshtein_engine_t(levenshtein_backend_t const &backend, sz_sequence_t const &queries,
                         sz_levenshtein_symbol_t symbol) {
        if (backend.init(&engine, &queries, symbol, &heap.allocator, nullptr) != sz_success_k)
            fail_backend_(backend.name, "the builder refused a well-formed batch of queries");
    }
    ~levenshtein_engine_t() {
        sz_levenshtein_engine_free(&engine, nullptr);
        verify(heap.live_allocations == 0);
    }
    levenshtein_engine_t(levenshtein_engine_t const &) = delete;
    levenshtein_engine_t &operator=(levenshtein_engine_t const &) = delete;
};

/** Runs @p queries against @p candidates through @p backend, asserting the matrix matches
 *  @p expected, which is @b [queries, candidates] row-major. */
inline void check_levenshtein_distances_(levenshtein_backend_t const &backend, std::vector<std::string> const &queries,
                                         std::vector<std::string> const &candidates,
                                         std::vector<sz_size_t> const &expected, sz_levenshtein_symbol_t symbol) {
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const candidate_sequence = sequence_from_(candidates);
    levenshtein_engine_t prepared(backend, query_sequence, symbol);
    std::vector<sz_size_t> computed(queries.size() * candidates.size(), STRINGZILLA_SIZE_MAX);
    if (backend.distances(&prepared.engine, &candidate_sequence, computed.data(), candidates.size(), nullptr) !=
        sz_success_k)
        fail_backend_(backend.name, "the cross-product verb refused a well-formed batch");
    if (computed != expected)
        fail_backend_(backend.name, "the cross-product distances differ from the expected answers");
}

/** Runs @p queries against @p candidates through @p backend, byte-level and rune-level alike,
 *  against the given answers. */
inline void check_levenshtein_expected_(levenshtein_backend_t const &backend, std::vector<std::string> const &queries,
                                        std::vector<std::string> const &candidates,
                                        std::vector<sz_size_t> const &expected_bytes,
                                        std::vector<sz_size_t> const &expected_runes) {
    check_levenshtein_distances_(backend, queries, candidates, expected_bytes, sz_levenshtein_bytes_k);
    check_levenshtein_distances_(backend, queries, candidates, expected_runes, sz_levenshtein_runes_k);
}

/** Runs @p queries against @p candidates through @p backend, each symbol kind by its oracle. */
inline void check_levenshtein_case_(levenshtein_backend_t const &backend, std::vector<std::string> const &queries,
                                    std::vector<std::string> const &candidates) {
    std::vector<sz_size_t> expected_bytes, expected_runes;
    for (std::string const &query : queries) {
        std::u32string const query_runes = levenshtein_runes_(query);
        for (std::string const &candidate : candidates) {
            expected_bytes.push_back(levenshtein_reference_(query, candidate));
            expected_runes.push_back(levenshtein_reference_(query_runes, levenshtein_runes_(candidate)));
        }
    }
    check_levenshtein_expected_(backend, queries, candidates, expected_bytes, expected_runes);
}

/** One candidate and the literal distances it must score against the query, in bytes and runes. */
struct levenshtein_known_t {
    std::string candidate;
    sz_size_t bytes;
    sz_size_t runes;
};

/** Runs one query against its known candidates through @p backend. */
inline void check_levenshtein_known_(levenshtein_backend_t const &backend, std::string const &query,
                                     std::vector<levenshtein_known_t> const &known) {
    std::vector<std::string> candidates;
    std::vector<sz_size_t> expected_bytes, expected_runes;
    for (levenshtein_known_t const &entry : known) {
        candidates.push_back(entry.candidate);
        expected_bytes.push_back(entry.bytes);
        expected_runes.push_back(entry.runes);
    }
    check_levenshtein_expected_(backend, {query}, candidates, expected_bytes, expected_runes);
}

/** Known answers: the classic pairs, empties on either side, identity, the 64-symbol word boundary,
 *  and multi-byte runes. */
inline void check_levenshtein_unit_(levenshtein_backend_t const &backend) {
    check_levenshtein_known_(
        backend, "kitten",
        {{"sitting", 3, 3}, {"kitten", 0, 0}, {"", 6, 6}, {"k", 5, 5}, {"kittens", 1, 1}, {"mitten", 1, 1}});
    check_levenshtein_known_(backend, "flaw", {{"lawn", 2, 2}, {"flaw", 0, 0}, {"flaws", 1, 1}, {"law", 1, 1}});
    check_levenshtein_known_(backend, "", {{"", 0, 0}, {"a", 1, 1}, {"abc", 3, 3}, {std::string(300, 'x'), 300, 300}});
    check_levenshtein_known_(backend, std::string(64, 'a'),
                             {{std::string(64, 'a'), 0, 0},
                              {std::string(65, 'a'), 1, 1},
                              {std::string(63, 'a'), 1, 1},
                              {std::string(64, 'b'), 64, 64},
                              {"", 64, 64}});
    check_levenshtein_known_(
        backend, std::string(65, 'a'),
        {{std::string(65, 'a'), 0, 0}, {std::string(64, 'a'), 1, 1}, {std::string(130, 'a'), 65, 65}});
    // Multi-byte runes: one rune edit costs several byte edits, and an ill-formed byte is one rune.
    check_levenshtein_known_(backend, "héllo",
                             {{"hello", 2, 1}, {"héllo", 0, 0}, {"h\xC3llo", 1, 1}, {"hé", 3, 3}, {"日本語", 9, 5}});
    check_levenshtein_known_(backend, "日本語",
                             {{"日本", 3, 1}, {"日本語です", 6, 2}, {"本", 6, 2}, {"", 9, 3}, {"\xFF\xFE", 9, 3}});
    check_levenshtein_known_(backend, "\xE2\x82", {{"€", 1, 2}, {"\xE2\x82\xAC", 1, 2}, {"ab", 2, 2}});

    // Several queries in one batch, so each matrix row is what the sweep wrote, not a repeat.
    check_levenshtein_case_(backend, {"kitten", "sitting", "", std::string(70, 'a')},
                            {"kitten", "sitting", "", "kit", std::string(70, 'b')});
}

/** The cross-product verb of @p backend on degenerate batches: empties on either side, both, and
 *  none survive, and a stride too narrow for one row is refused without touching the outputs. Its
 *  builder reports a refused allocation rather than handing back a half-built engine. */
inline void check_levenshtein_safety_(levenshtein_backend_t const &backend) {
    std::vector<std::string> const queries = {"kitten", ""};
    std::vector<std::string> const words = {"sitting", "kitten"};
    std::vector<std::string> const empty = {""};
    std::vector<std::string> const none;
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const words_sequence = sequence_from_(words);
    sz_sequence_t const empty_sequence = sequence_from_(empty);
    sz_sequence_t const none_sequence = sequence_from_(none);
    for (sz_levenshtein_symbol_t const symbol : {sz_levenshtein_bytes_k, sz_levenshtein_runes_k}) {
        levenshtein_engine_t prepared(backend, query_sequence, symbol);
        sz_size_t answers[4] = {STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX};
        if (backend.distances(&prepared.engine, &words_sequence, answers, 2, nullptr) != sz_success_k)
            fail_backend_(backend.name, "a batch holding an empty query was refused");
        if (backend.distances(&prepared.engine, &empty_sequence, answers, 1, nullptr) != sz_success_k)
            fail_backend_(backend.name, "an empty candidate was refused");
        if (backend.distances(&prepared.engine, &none_sequence, answers, 0, nullptr) != sz_success_k)
            fail_backend_(backend.name, "an empty batch of candidates was refused");
        sz_size_t narrow[4] = {STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX, STRINGZILLA_SIZE_MAX};
        if (backend.distances(&prepared.engine, &words_sequence, narrow, 1, nullptr) != sz_unexpected_dimensions_k)
            fail_backend_(backend.name, "a stride too narrow for one row was not refused");
        for (sz_size_t const written : narrow)
            if (written != STRINGZILLA_SIZE_MAX)
                fail_backend_(backend.name, "a refused round still wrote the distances");
    }

    sz_allocator_t refusing = refusing_allocator_();
    std::vector<std::string> const refused_queries = {"kitten", "sitting"};
    sz_sequence_t const refused_sequence = sequence_from_(refused_queries);
    for (sz_levenshtein_symbol_t const symbol : {sz_levenshtein_bytes_k, sz_levenshtein_runes_k}) {
        sz_levenshtein_engine_t engine {};
        if (backend.init(&engine, &refused_sequence, symbol, &refusing, nullptr) != sz_bad_alloc_k)
            fail_backend_(backend.name, "the builder did not report the refused allocation");
        if (engine.memory != nullptr) fail_backend_(backend.name, "a refused build still kept a block");
    }
}

/**
 *  @brief One backend against the textbook oracle over generated corpora, then against serial.
 *
 *  Query and candidate lengths sweep every 64-symbol word boundary and reach past the
 *  register-resident word tiers, on a two-letter alphabet that forces matches, on a multi-byte rune
 *  alphabet, and on full bytes. Serial is the reference for everything, itself included.
 */
inline void check_levenshtein_equivalence_(test_context_t &context, levenshtein_backend_t const &backend) {
    std::size_t const lengths[] = {0, 1, 2, 5, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300, 511, 512, 513, 640, 1000};
    char const binary_alphabet[] = "ab";
    std::vector<std::string> const rune_alphabet = {"a", "é", "日", "€", "𝄞"};
    std::mt19937 &generator = context.generator;
    for (std::size_t query_length : lengths) {
        std::vector<std::string> candidates;
        for (std::size_t candidate_length : lengths)
            candidates.push_back(random_string(generator, candidate_length, binary_alphabet));
        std::string const query = random_string(generator, query_length, binary_alphabet);
        candidates.push_back(query);
        candidates.push_back(query.substr(0, query_length / 2));
        check_levenshtein_case_(backend, {query}, candidates);
    }
    for (std::size_t query_runes : {0, 1, 63, 64, 65, 129, 300, 640, 1000}) {
        std::vector<std::string> candidates;
        for (std::size_t candidate_runes : {0, 1, 64, 65, 200, 256, 512, 513})
            candidates.push_back(random_string(generator, candidate_runes, rune_alphabet));
        std::string const query = random_string(generator, query_runes, rune_alphabet);
        candidates.push_back(query);
        check_levenshtein_case_(backend, {query}, candidates);
    }
    // Three hundred consecutive CJK runes: more classes than a byte holds, over two table pages.
    std::string wide_query;
    for (sz_rune_t rune = 0x4E00; rune != 0x4E00 + 300; ++rune) {
        wide_query += static_cast<char>(0xE0 | (rune >> 12));
        wide_query += static_cast<char>(0x80 | ((rune >> 6) & 0x3F));
        wide_query += static_cast<char>(0x80 | (rune & 0x3F));
    }
    check_levenshtein_case_(backend, {wide_query},
                            {wide_query, wide_query.substr(0, 150), wide_query.substr(300), "abc", ""});
    check_levenshtein_case_(backend, {"abc"}, {});
    for (std::size_t round = 0; round != context.iterations(8); ++round) {
        std::vector<std::string> queries;
        for (std::size_t index = 0; index != 3; ++index) {
            std::size_t const query_length = std::uniform_int_distribution<std::size_t>(1, 700)(generator);
            queries.push_back(std::string(query_length, '\0'));
            randomize_string(generator, queries.back());
        }
        std::vector<std::string> candidates;
        for (std::size_t index = 0; index != 40; ++index) {
            std::string candidate(std::uniform_int_distribution<std::size_t>(0, 900)(generator), '\0');
            randomize_string(generator, candidate);
            candidates.push_back(candidate);
        }
        check_levenshtein_case_(backend, queries, candidates);
    }

    // The engine is the candidate's, and serial scores the same batch, over random batches on a
    // two-letter and a multi-byte rune alphabet, with several queries in flight to cover that axis.
    std::vector<std::string> candidates, queries;
    std::vector<sz_size_t> from_reference, from_candidate;
    for (std::size_t round = 0; round != context.iterations(8); ++round)
        for (char const *alphabet : {"ab", "aé日€𝄞"}) {
            randomize_strings(generator, fuzzy_config_t(alphabet, 16, 0, 900), candidates);
            queries.clear();
            for (std::size_t query = 0; query != 5; ++query) {
                std::size_t const query_length = std::uniform_int_distribution<std::size_t>(1, 700)(generator);
                queries.push_back(random_string(generator, query_length, alphabet_characters(alphabet)));
            }
            sz_sequence_t const query_sequence = sequence_from_(queries);
            sz_sequence_t const candidate_sequence = sequence_from_(candidates);
            std::size_t const cells = queries.size() * candidates.size();

            for (sz_levenshtein_symbol_t const symbol : {sz_levenshtein_bytes_k, sz_levenshtein_runes_k}) {
                levenshtein_engine_t prepared(backend, query_sequence, symbol);
                from_reference.assign(cells, STRINGZILLA_SIZE_MAX), from_candidate.assign(cells, STRINGZILLA_SIZE_MAX);
                verify(sz_levenshtein_distances_serial(&prepared.engine, &candidate_sequence, from_reference.data(),
                                                       candidates.size(), nullptr) == sz_success_k);
                if (backend.distances(&prepared.engine, &candidate_sequence, from_candidate.data(), candidates.size(),
                                      nullptr) != sz_success_k)
                    fail_backend_(backend.name, "a batch serial accepted was refused");
                if (from_reference != from_candidate) fail_backend_(backend.name, "distances disagreed with serial");
            }
        }
}

#pragma endregion Levenshtein

#pragma region Overlap

/** One backend's engine constructor and the round it then scores. */
struct overlap_backend_t {
    char const *name;
    sz_kernel_overlap_engine_init_t init;
    sz_kernel_overlap_scores_t scores;
};

/** The share of @p candidate windows among the distinct @p query windows, through @c std::set. */
inline sz_f32_t overlap_reference_score_(std::string const &query, std::string const &candidate, std::size_t width) {
    std::size_t const query_windows = width <= query.size() ? query.size() - width + 1 : 0;
    std::size_t const candidate_windows = width <= candidate.size() ? candidate.size() - width + 1 : 0;
    std::set<std::string> present;
    for (std::size_t window = 0; window != query_windows; ++window) present.insert(query.substr(window, width));
    std::size_t matches = 0;
    for (std::size_t window = 0; window != candidate_windows; ++window)
        matches += present.count(candidate.substr(window, width));
    std::size_t const longer = std::max(query_windows, candidate_windows);
    return longer ? static_cast<sz_f32_t>(static_cast<double>(matches) / static_cast<double>(longer)) : 0.0f;
}

inline std::vector<sz_f32_t> overlap_tensor_(overlap_backend_t const &backend, std::vector<std::string> const &queries,
                                             std::vector<std::string> const &candidates,
                                             std::vector<std::size_t> const &widths) {
    handle_checked_heap_t heap;
    sz_sequence_t const query_sequence = sequence_from_(queries);
    sz_sequence_t const candidate_sequence = sequence_from_(candidates);
    sz_overlap_engine_t engine {};
    if (backend.init(&engine, &query_sequence, widths.data(), widths.size(), 0, &heap.allocator, nullptr) !=
        sz_success_k)
        fail_backend_(backend.name, "the engine refused a well-formed batch of queries");
    std::vector<sz_f32_t> scores(queries.size() * candidates.size() * widths.size(), -1.0f);
    if (backend.scores(&engine, &candidate_sequence, scores.data(), candidates.size() * widths.size(), widths.size(),
                       nullptr) != sz_success_k)
        fail_backend_(backend.name, "the engine refused a well-formed batch of candidates");
    sz_overlap_engine_free(&engine, nullptr);
    verify(heap.live_allocations == 0);
    return scores;
}

/** Runs @p queries against @p candidates through @p backend at @p widths, asserting each share
 *  matches the oracle to within one rounding. */
inline void check_overlap_scores_(overlap_backend_t const &backend, std::vector<std::string> const &queries,
                                  std::vector<std::string> const &candidates, std::vector<std::size_t> const &widths) {
    std::vector<sz_f32_t> const computed = overlap_tensor_(backend, queries, candidates, widths);
    std::size_t const candidate_stride = widths.size(), query_stride = candidates.size() * widths.size();
    for (std::size_t query = 0; query != queries.size(); ++query)
        for (std::size_t candidate = 0; candidate != candidates.size(); ++candidate)
            for (std::size_t width_index = 0; width_index != widths.size(); ++width_index) {
                sz_f32_t const expected = overlap_reference_score_(queries[query], candidates[candidate],
                                                                   widths[width_index]);
                sz_f32_t const produced = computed[query * query_stride + candidate * candidate_stride + width_index];
                if (std::fabs(produced - expected) > 1e-6f)
                    fail_backend_(backend.name, "a share differs from the std::set oracle by more than one rounding");
            }
}

/**
 *  @brief One backend's engine against the @c std::set oracle and against serial, bit for bit.
 *
 *  One engine over every query must also match one engine per query, since batched and lone rows
 *  walk different forests. Serial is the reference for everything, itself included.
 */
inline void check_overlap_equivalence_(test_context_t &context, overlap_backend_t const &backend) {
    std::mt19937 &generator = context.generator;
    std::vector<std::size_t> const widths = {1, 3, 4, 6, 8, 11, 32};
    for (std::size_t round = 0; round != context.iterations(8); ++round) {
        std::size_t const query_length = std::uniform_int_distribution<std::size_t>(0, 700)(generator);
        std::string query(query_length, '\0');
        randomize_string(generator, query);

        // Full bytes rarely repeat a window past width three, so half the candidates are cut from
        // the query itself and the rest come from a two-letter alphabet where every width repeats.
        std::vector<std::string> candidates;
        randomize_strings(generator, fuzzy_config_t("ab", 12, 0, 900), candidates);
        candidates.push_back(query);
        candidates.push_back(query.substr(query_length / 3));
        candidates.push_back(query + query);
        std::vector<std::string> const queries = {query, random_string(generator, query_length, "ab"), std::string()};
        check_overlap_scores_(backend, queries, candidates, widths);

        std::vector<sz_f32_t> const batched = overlap_tensor_(backend, queries, candidates, widths);
        std::size_t const query_stride = candidates.size() * widths.size();
        for (std::size_t index = 0; index != queries.size(); ++index) {
            std::vector<std::string> const alone = {queries[index]};
            std::vector<sz_f32_t> const lone = overlap_tensor_(backend, alone, candidates, widths);
            if (std::memcmp(batched.data() + index * query_stride, lone.data(), query_stride * sizeof(sz_f32_t)) != 0)
                fail_backend_(backend.name, "a query's row differs between a batched engine and its own");
        }
    }

    overlap_backend_t const reference {"serial", sz_overlap_engine_init_serial, sz_overlap_scores_serial};
    std::vector<std::string> candidates;
    for (std::size_t round = 0; round != context.iterations(8); ++round) {
        std::string query(std::uniform_int_distribution<std::size_t>(0, 700)(generator), '\0');
        randomize_string(generator, query);
        randomize_strings(generator, fuzzy_config_t("ab", 12, 0, 900), candidates);
        candidates.push_back(query);
        candidates.push_back(query.substr(query.size() / 3));
        std::vector<std::string> const queries = {query, query.substr(query.size() / 2), std::string()};

        std::vector<sz_f32_t> const from_reference = overlap_tensor_(reference, queries, candidates, widths);
        std::vector<sz_f32_t> const from_candidate = overlap_tensor_(backend, queries, candidates, widths);
        if (std::memcmp(from_reference.data(), from_candidate.data(), from_reference.size() * sizeof(sz_f32_t)) != 0)
            fail_backend_(backend.name, "the engine's scores disagreed with serial");
    }
}

/** One backend's engine on degenerate inputs: empties on either side, both, and none survive, as
 *  does a candidate narrower than the width; zero widths, a refused allocation and a stride under
 *  its own axis are reported without touching the outputs. */
inline void check_overlap_safety_(overlap_backend_t const &backend) {
    handle_checked_heap_t heap;
    sz_allocator_t refusing = refusing_allocator_();
    std::vector<std::string> const words = {"sitting", "kitten"};
    std::vector<std::string> const kitten = {"kitten"};
    std::vector<std::string> const empty = {""};
    std::vector<std::string> const narrow = {"ab"};
    std::vector<std::string> const none;
    std::vector<std::size_t> const widths = {3, 8};

    check_overlap_scores_(backend, empty, words, widths);
    check_overlap_scores_(backend, kitten, empty, widths);
    check_overlap_scores_(backend, empty, empty, widths);
    check_overlap_scores_(backend, kitten, none, widths);
    check_overlap_scores_(backend, none, words, widths);
    check_overlap_scores_(backend, kitten, narrow, widths);

    sz_sequence_t const kitten_sequence = sequence_from_(kitten);
    sz_sequence_t const words_sequence = sequence_from_(words);
    sz_overlap_engine_t engine {};
    if (backend.init(&engine, &kitten_sequence, widths.data(), 0, 0, &heap.allocator, nullptr) !=
        sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "the engine accepted zero widths");
    if (backend.init(&engine, &kitten_sequence, widths.data(), widths.size(), 0, &refusing, nullptr) != sz_bad_alloc_k)
        fail_backend_(backend.name, "the engine did not report the refused allocation");
    if (backend.init(&engine, &kitten_sequence, widths.data(), widths.size(), 0, &heap.allocator, nullptr) !=
        sz_success_k)
        fail_backend_(backend.name, "the engine refused a well-formed batch of queries");

    sz_f32_t refused[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    if (backend.scores(&engine, &words_sequence, refused, words.size() * widths.size(), widths.size() - 1, nullptr) !=
        sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "the engine accepted a candidate stride under its widths");
    if (backend.scores(&engine, &words_sequence, refused, words.size() * widths.size() - 1, widths.size(), nullptr) !=
        sz_unexpected_dimensions_k)
        fail_backend_(backend.name, "the engine accepted a query stride under its candidates");
    for (sz_f32_t const untouched : refused)
        if (untouched != -1.0f) fail_backend_(backend.name, "a refused call still wrote a score");
    sz_overlap_engine_free(&engine, nullptr);
    verify(heap.live_allocations == 0);
}

#pragma endregion Overlap

#pragma region Substrings

/** One match as the oracle and the backends both spell it, so a differential is one comparison. */
struct substrings_case_t {
    std::size_t haystack_index;
    std::size_t needle_index;
    std::size_t byte_offset;
    std::size_t byte_length;

    bool operator==(substrings_case_t const &other) const noexcept {
        return haystack_index == other.haystack_index && needle_index == other.needle_index &&
               byte_offset == other.byte_offset && byte_length == other.byte_length;
    }
    bool operator<(substrings_case_t const &other) const noexcept {
        if (haystack_index != other.haystack_index) return haystack_index < other.haystack_index;
        if (byte_offset != other.byte_offset) return byte_offset < other.byte_offset;
        if (byte_length != other.byte_length) return byte_length < other.byte_length;
        return needle_index < other.needle_index;
    }
};

/** One tier's engine builder and the verbs it then answers, so every compiled tier meets the
 *  oracle, not only the dispatched one. */
struct substrings_tier_t {
    sz_kernel_substrings_engine_init_t init;
    sz_kernel_substrings_counts_t counts;
    sz_kernel_substrings_find_t find;
    sz_kernel_substrings_replace_t replace;
    sz_kernel_substrings_bm25_scores_t bm25_scores;
};

/** Binds a sequence over a vector of strings, which is what every verb here takes. */
inline sz_sequence_t sequence_over_(std::vector<std::string> const &strings, std::vector<sz_string_view_t> &views) {
    views.resize(strings.size());
    for (std::size_t index = 0; index != strings.size(); ++index)
        views[index].start = strings[index].data(), views[index].length = strings[index].size();
    sz_sequence_t sequence;
    sz_sequence_from_string_views(views.data(), views.size(), &sequence);
    return sequence;
}

/** Folds @p text with the serial folder, which is the stream an uncased automaton walks. */
inline std::string folded_(std::string const &text) {
    std::string folded(text.size() * 4 + 4, '\0');
    std::size_t const written = kernel_result<sz_size_t>(sz_utf8_uncased_fold_serial, text.data(), text.size(),
                                                         &folded[0]);
    folded.resize(written);
    return folded;
}

/**
 *  @brief Where the source codepoint behind each folded byte starts and ends.
 *
 *  A folded match snaps outward onto whole codepoints, so a folded span `[first, last)` becomes the
 *  source span `[starts[first], ends[last - 1])`. Snapping the end onto the producing codepoint's
 *  own start instead would collapse a match ending inside an expansion to nothing.
 */
struct folded_origins_t {

    /** Source offset the codepoint behind each folded byte begins at. */
    std::vector<std::size_t> starts;

    /** Source offset just past that codepoint. */
    std::vector<std::size_t> ends;
};

inline folded_origins_t folded_origins_(std::string const &text) {
    folded_origins_t origins;
    std::size_t offset = 0;
    while (offset < text.size()) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode(text.data() + offset, text.data() + text.size(), &rune);
        std::size_t const source_length = consumed == sz_rune_invalid_k ? 1 : (std::size_t)consumed;
        std::string const image = folded_(text.substr(offset, source_length));
        for (std::size_t byte = 0; byte != image.size(); ++byte)
            origins.starts.push_back(offset), origins.ends.push_back(offset + source_length);
        offset += source_length;
    }
    return origins;
}

/** Every match of every needle at every offset, which is what the automaton must agree with. */
inline std::vector<substrings_case_t> oracle_overlapping_(std::vector<std::string> const &haystacks,
                                                          std::vector<std::string> const &needles,
                                                          sz_substrings_case_sensitivity_t sensitivity) {
    std::vector<substrings_case_t> found;
    for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
        std::string const &haystack = haystacks[haystack_index];
        if (sensitivity == sz_substrings_cased_k) {
            for (std::size_t needle_index = 0; needle_index != needles.size(); ++needle_index) {
                std::string const &needle = needles[needle_index];
                if (needle.empty() || needle.size() > haystack.size()) continue;
                for (std::size_t offset = 0; offset + needle.size() <= haystack.size(); ++offset)
                    if (std::memcmp(haystack.data() + offset, needle.data(), needle.size()) == 0)
                        found.push_back({haystack_index, needle_index, offset, needle.size()});
            }
            continue;
        }

        // The automaton walks folded bytes, so the oracle does too, and reports the source span the
        // folded one snaps outward onto. Distinct folded offsets can snap onto one source span,
        // which is the repeat the walk collapses, so the oracle collapses it here as well.
        std::string const folded_haystack = folded_(haystack);
        folded_origins_t const origins = folded_origins_(haystack);
        verify(origins.starts.size() == folded_haystack.size());
        for (std::size_t needle_index = 0; needle_index != needles.size(); ++needle_index) {
            std::string const folded_needle = folded_(needles[needle_index]);
            if (folded_needle.empty() || folded_needle.size() > folded_haystack.size()) continue;
            std::size_t previous_offset = haystack.size() + 1, previous_length = 0;
            for (std::size_t offset = 0; offset + folded_needle.size() <= folded_haystack.size(); ++offset) {
                if (std::memcmp(folded_haystack.data() + offset, folded_needle.data(), folded_needle.size()) != 0)
                    continue;
                std::size_t const source_offset = origins.starts[offset];
                std::size_t const source_end = origins.ends[offset + folded_needle.size() - 1];
                if (source_offset == previous_offset && source_end - source_offset == previous_length) continue;
                previous_offset = source_offset, previous_length = source_end - source_offset;
                found.push_back({haystack_index, needle_index, source_offset, previous_length});
            }
        }
    }
    std::sort(found.begin(), found.end());
    return found;
}

/** The greedy cover the leftmost policies name, taken over the oracle's own overlapping matches. */
inline std::vector<substrings_case_t> oracle_leftmost_(std::vector<substrings_case_t> const &overlapping,
                                                       std::size_t haystacks_count,
                                                       sz_substrings_overlap_policy_t policy) {
    std::vector<substrings_case_t> kept;
    for (std::size_t haystack_index = 0; haystack_index != haystacks_count; ++haystack_index) {
        std::size_t cursor = 0;
        for (;;) {
            substrings_case_t const *chosen = nullptr;
            for (substrings_case_t const &candidate : overlapping) {
                if (candidate.haystack_index != haystack_index || candidate.byte_offset < cursor) continue;
                if (!chosen) {
                    chosen = &candidate;
                    continue;
                }
                if (candidate.byte_offset != chosen->byte_offset) {
                    if (candidate.byte_offset < chosen->byte_offset) chosen = &candidate;
                    continue;
                }
                if (policy == sz_substrings_leftmost_longest_k && candidate.byte_length != chosen->byte_length) {
                    if (candidate.byte_length > chosen->byte_length) chosen = &candidate;
                    continue;
                }
                if (candidate.needle_index < chosen->needle_index) chosen = &candidate;
            }
            if (!chosen) break;
            kept.push_back(*chosen);
            cursor = chosen->byte_offset + chosen->byte_length;
        }
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}

/** The rewrite that cover implies, spliced by the oracle rather than by the backend. */
inline std::string oracle_rewrite_(std::string const &haystack, std::size_t haystack_index,
                                   std::vector<substrings_case_t> const &cover,
                                   std::vector<std::string> const &replacements) {
    std::string rewritten;
    std::size_t cursor = 0;
    for (substrings_case_t const &match : cover) {
        if (match.haystack_index != haystack_index) continue;
        rewritten.append(haystack, cursor, match.byte_offset - cursor);
        rewritten.append(replacements[match.needle_index]);
        cursor = match.byte_offset + match.byte_length;
    }
    rewritten.append(haystack, cursor, haystack.size() - cursor);
    return rewritten;
}

/** BM25 over the oracle's own overlapping matches, summed in double precision in any order. */
inline std::vector<double> oracle_bm25_(std::vector<substrings_case_t> const &overlapping,
                                        std::vector<std::string> const &haystacks, std::size_t needles_count,
                                        std::vector<sz_f32_t> const &document_lengths,
                                        sz_substrings_bm25_t const &parameters, std::vector<sz_f32_t> const &weights) {
    std::vector<std::vector<std::size_t>> frequencies(haystacks.size(), std::vector<std::size_t>(needles_count));
    for (substrings_case_t const &match : overlapping) ++frequencies[match.haystack_index][match.needle_index];
    std::vector<double> scores(haystacks.size(), 0);
    for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
        double const length = document_lengths.empty() ? (double)haystacks[haystack_index].size()
                                                       : (double)document_lengths[haystack_index];
        double const normalization = parameters.length_normalization;
        double const saturation = parameters.term_frequency_saturation;
        double const norm = normalization > 0
                                ? 1 - normalization + normalization * length / parameters.average_document_length
                                : 1;
        for (std::size_t needle_index = 0; needle_index != needles_count; ++needle_index) {
            double const frequency = (double)frequencies[haystack_index][needle_index];
            if (frequency)
                scores[haystack_index] += weights[needle_index] * frequency * (saturation + 1) /
                                          (frequency + saturation * norm);
        }
    }
    return scores;
}

/** One tier's BM25 against the oracle, by byte lengths and by caller-given ones, with and without
 *  length normalization. */
inline void check_bm25_(substrings_tier_t const &tier, sz_substrings_engine_t *engine,
                        sz_sequence_t const *haystack_sequence, std::vector<std::string> const &haystacks,
                        std::vector<substrings_case_t> const &overlapping) {
    std::size_t const needles_count = engine->needles_count;
    std::vector<sz_f32_t> weights(needles_count), given_lengths(haystacks.size());
    for (std::size_t index = 0; index != needles_count; ++index) weights[index] = 0.5f + (float)(index % 7) * 0.25f;
    double bytes_total = 0, given_total = 0;
    for (std::size_t index = 0; index != haystacks.size(); ++index) {
        given_lengths[index] = (sz_f32_t)(1 + index % 5);
        bytes_total += haystacks[index].size(), given_total += given_lengths[index];
    }

    for (std::vector<sz_f32_t> const &lengths : {std::vector<sz_f32_t>(), given_lengths}) {
        double const average = (lengths.empty() ? bytes_total : given_total) / haystacks.size();
        sz_substrings_bm25_t const normalized {1.2f, 0.75f, (sz_f32_t)(average > 0 ? average : 1)};
        sz_substrings_bm25_t const unnormalized {1.2f, 0.0f, 0.0f};
        for (sz_substrings_bm25_t const &parameters : {normalized, unnormalized}) {
            std::vector<double> const expected = oracle_bm25_(overlapping, haystacks, needles_count, lengths,
                                                              parameters, weights);
            std::vector<sz_f32_t> scores(haystacks.size(), -1);
            verify(tier.bm25_scores(engine, haystack_sequence, lengths.empty() ? nullptr : lengths.data(), &parameters,
                                    weights.data(), scores.data(), 1, nullptr) == sz_success_k);
            for (std::size_t index = 0; index != haystacks.size(); ++index)
                verify(std::fabs(scores[index] - expected[index]) <= 1e-5 * std::max(1.0, std::fabs(expected[index])));
        }
    }
}

/** Reads every match the engine reports, sizing the array by the report the sizing call leaves. */
inline std::vector<substrings_case_t> backend_find_(substrings_tier_t const &tier, sz_substrings_engine_t *engine,
                                                    sz_sequence_t const *haystacks) {
    std::vector<sz_size_t> offsets(haystacks->count + 1, 0);
    verify(tier.find(engine, haystacks, nullptr, 0, offsets.data(), nullptr) == sz_success_k);
    sz_size_t const total = engine->report->matches_emitted;
    verify(offsets[haystacks->count] == total);
    verify(engine->report->shortfall == total);

    std::vector<sz_substrings_match_t> matches(total);
    verify(tier.find(engine, haystacks, matches.data(), matches.size(), offsets.data(), nullptr) == sz_success_k);
    verify(engine->report->matches_stored == total && engine->report->shortfall == 0);

    std::vector<substrings_case_t> reported(total);
    for (std::size_t index = 0; index != total; ++index)
        reported[index] = {matches[index].haystack_index, matches[index].needle_index, matches[index].byte_offset,
                           matches[index].byte_length};
    std::sort(reported.begin(), reported.end());
    return reported;
}

/** One vocabulary against one corpus under one policy, built and walked by @p tier, and compared
 *  with the oracle on every verb. */
inline void check_corpus_(substrings_tier_t const &tier, std::vector<std::string> const &haystacks,
                          std::vector<std::string> const &needles, sz_substrings_case_sensitivity_t sensitivity,
                          sz_substrings_overlap_policy_t policy,
                          std::size_t hot_states = STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO) {
    handle_checked_heap_t heap;
    std::vector<sz_string_view_t> haystack_views, needle_views, replacement_views;
    sz_sequence_t const haystack_sequence = sequence_over_(haystacks, haystack_views);
    sz_sequence_t const needle_sequence = sequence_over_(needles, needle_views);

    sz_substrings_engine_t engine;
    verify(tier.init(&engine, &needle_sequence, sensitivity, policy, hot_states, 0, 0, &heap.allocator, nullptr) ==
           sz_success_k);
    verify(engine.needles_count == needles.size());
    verify(engine.root == 0);

    std::vector<substrings_case_t> const overlapping = oracle_overlapping_(haystacks, needles, sensitivity);
    std::vector<substrings_case_t> const expected = policy == sz_substrings_overlapping_k
                                                        ? overlapping
                                                        : oracle_leftmost_(overlapping, haystacks.size(), policy);

    // Every match, located the same way the oracle locates it.
    std::vector<substrings_case_t> const reported = backend_find_(tier, &engine, &haystack_sequence);
    verify(reported.size() == expected.size());
    for (std::size_t index = 0; index != reported.size(); ++index) verify(reported[index] == expected[index]);

    // The per-haystack counts, which a separate walk answers and so can disagree with the matches.
    std::vector<sz_size_t> counts(haystacks.size(), 0);
    verify(tier.counts(&engine, &haystack_sequence, counts.data(), 1, nullptr) == sz_success_k);
    for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
        std::size_t owned = 0;
        for (substrings_case_t const &match : expected) owned += match.haystack_index == haystack_index;
        verify(counts[haystack_index] == owned);
    }

    // BM25 reads raw overlapping frequencies, so it is checked once per corpus, not per policy.
    if (policy == sz_substrings_overlapping_k) check_bm25_(tier, &engine, &haystack_sequence, haystacks, overlapping);

    // The rewrite, which only a cover admits.
    if (policy != sz_substrings_overlapping_k) {
        std::vector<std::string> replacements;
        for (std::size_t needle_index = 0; needle_index != needles.size(); ++needle_index)
            replacements.push_back(needle_index % 3 == 0 ? std::string() : "<" + std::to_string(needle_index) + ">");
        sz_sequence_t const replacement_sequence = sequence_over_(replacements, replacement_views);

        std::vector<sz_size_t> offsets(haystacks.size() + 1, 0);
        verify(tier.replace(&engine, &haystack_sequence, &replacement_sequence, nullptr, 0, offsets.data(), nullptr) ==
               sz_success_k);
        verify(engine.report->target_length == offsets[haystacks.size()]);

        std::vector<char> tape(offsets[haystacks.size()]);
        verify(tier.replace(&engine, &haystack_sequence, &replacement_sequence, tape.empty() ? nullptr : tape.data(),
                            tape.size(), offsets.data(), nullptr) == sz_success_k);
        verify(engine.report->shortfall == 0);
        for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
            std::string const rewritten = oracle_rewrite_(haystacks[haystack_index], haystack_index, expected,
                                                          replacements);
            std::size_t const first = offsets[haystack_index], last = offsets[haystack_index + 1];
            verify(last - first == rewritten.size());
            verify(rewritten.empty() || std::memcmp(tape.data() + first, rewritten.data(), rewritten.size()) == 0);
        }
    }

    sz_substrings_engine_free(&engine, nullptr);
    verify(heap.live_allocations == 0);
}

/** The same corpus under every policy, so one call covers a vocabulary's whole behaviour. */
inline void check_policies_(substrings_tier_t const &tier, std::vector<std::string> const &haystacks,
                            std::vector<std::string> const &needles, sz_substrings_case_sensitivity_t sensitivity,
                            std::size_t hot_states = STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO) {
    check_corpus_(tier, haystacks, needles, sensitivity, sz_substrings_overlapping_k, hot_states);
    check_corpus_(tier, haystacks, needles, sensitivity, sz_substrings_leftmost_longest_k, hot_states);
    check_corpus_(tier, haystacks, needles, sensitivity, sz_substrings_leftmost_first_k, hot_states);
}

/**
 *  @brief The same corpus at every tier split, the only way the cold tier is reached at all.
 *
 *  A default build keeps thousands of states hot, so every vocabulary small enough to compare
 *  against a brute-force oracle is entirely hot and the double array is never read. Sweeping the
 *  split walks the same answers through the dense rows, a mixed automaton, and an all-cold one.
 */
inline void check_tier_splits_(substrings_tier_t const &tier, std::vector<std::string> const &haystacks,
                               std::vector<std::string> const &needles, sz_substrings_case_sensitivity_t sensitivity) {
    for (std::size_t hot_states : {(std::size_t)0, (std::size_t)1, (std::size_t)2, (std::size_t)4, (std::size_t)8,
                                   (std::size_t)32, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO})
        check_policies_(tier, haystacks, needles, sensitivity, hot_states);
}

/** The textbook cases, where a wrong failure link or a missed output run shows up by name. */
inline void check_substrings_unit_(substrings_tier_t const &tier) {
    // The canonical Aho-Corasick vocabulary: nested suffixes, so every state inherits a run.
    check_policies_(tier, {"ushers"}, {"he", "she", "his", "hers"}, sz_substrings_cased_k);

    // A needle that is a suffix of another, over text that spells both at once.
    check_policies_(tier, {"abcd", "abcde", "zzz"}, {"bc", "abcd", "cd", "d"}, sz_substrings_cased_k);

    // One needle repeated, which is what exercises the leftmost cursor rather than the ranking.
    check_policies_(tier, {"aaaaaaaa"}, {"aa", "aaa"}, sz_substrings_cased_k);

    // A vocabulary no haystack hits, so the cover drains nothing and the ring stays zero.
    check_policies_(tier, {"the quick brown fox", ""}, {"zebra", "quetzal"}, sz_substrings_cased_k);

    // Bytes outside ASCII, byte-exact, with a lead byte a folded walk would have resynchronized on.
    check_policies_(tier, {std::string("\xC3\xA9\xE2\x82\xAC\xFF\xFE", 7)}, {std::string("\xFF\xFE", 2), "\xC3\xA9"},
                    sz_substrings_cased_k);

    // Case folding, where the needle and the haystack agree only after both are folded.
    check_policies_(tier, {"Hello World", "HELLO", "hello"}, {"hello", "WORLD"}, sz_substrings_uncased_k);

    // The sharp S folds to two runes, so one source codepoint spans two folded bytes and a match
    // can end at either of them - the repeat the walk collapses.
    check_policies_(tier, {"Straße", "STRASSE", "strasse"}, {"strasse", "sse", "s"}, sz_substrings_uncased_k);

    // The Kelvin sign folds to one ASCII byte, contracting three source bytes into one folded one.
    check_policies_(tier, {"\xE2\x84\xAA elvin", "kelvin"}, {"k", "kelvin"}, sz_substrings_uncased_k);

    // Malformed UTF-8 in the haystack, which an uncased walk resynchronizes past byte by byte.
    check_policies_(tier,
                    {std::string("ab\xFF" "cd", 5), // "\xFFc" is one escape
                     "abcd"},
                    {"ab", "cd"}, sz_substrings_uncased_k);

    // A chain trie split one state into the hot tier, walked over a byte no needle spells. A cold
    // state's own slot sits inside its children's addressing window, so a slot recorded as its own
    // owner answers that state's probe as an edge, and the walk stays deep where it should have
    // fallen to the root.
    check_tier_splits_(tier,
                       {std::string("aaaa\x01" "aaaa", 9), // "\x01a" is one escape
                        "aaaaaaaa"},
                       {"a", "aa", "aaa", "aaaa"}, sz_substrings_cased_k);

    // Every tier split over the canonical vocabulary: a cold state's probe can land on a slot a hot
    // state's child owns, so a mixed automaton is its own case, not a shade of the two extremes.
    check_tier_splits_(tier, {"ushers", "she sells seashells"}, {"he", "she", "his", "hers"}, sz_substrings_cased_k);
    check_tier_splits_(tier, {"abcabcabc", "aaaa"}, {"a", "ab", "abc", "bc", "c"}, sz_substrings_cased_k);
    check_tier_splits_(tier, {"Straße", "STRASSE"}, {"strasse", "sse", "s"}, sz_substrings_uncased_k);

    // Uppercase needles in a byte-exact vocabulary, which the folded vocabularies' column aliasing
    // must not touch.
    check_policies_(tier, {"AbabAB aBAb", "ABAB"}, {"Ab", "ab", "AB", "bA"}, sz_substrings_cased_k);

    // A single-byte vocabulary, whose leftmost ring is narrower than one bitmap word.
    check_policies_(tier, {"aabbaab", "b"}, {"a", "b"}, sz_substrings_cased_k);

    // A folded needle past 21 bytes: its source span triples, and its ring spans several words.
    check_policies_(tier, {"The Quick Brown Fox Jumps Over The Lazy Dog, the quick brown fox jumps over the lazy dog"},
                    {"quick brown fox jumps over", "the lazy dog", "fox"}, sz_substrings_uncased_k);

    // An all-cold automaton, so every step probes the double array and chases failure links.
    handle_checked_heap_t heap;
    std::vector<std::string> const needles {"he", "she", "his", "hers"};
    std::vector<sz_string_view_t> views;
    sz_sequence_t const sequence = sequence_over_(needles, views);
    sz_substrings_engine_t engine;
    verify(tier.init(&engine, &sequence, sz_substrings_cased_k, sz_substrings_overlapping_k, 0, 0, 0, &heap.allocator,
                     nullptr) == sz_success_k);
    verify(engine.hot_count == 0);
    sz_substrings_engine_free(&engine, nullptr);
    verify(heap.live_allocations == 0);
}

/** Random vocabularies over random corpora, which reach the packing search's fallbacks. */
inline void check_substrings_equivalence_(test_context_t &context, substrings_tier_t const &tier) {
    std::mt19937 &generator = context.generator;
    char const *const alphabets[] = {"ab", "abcdefgh", "abcdefghijklmnopqrstuvwxyz"};
    std::size_t const rounds = context.iterations(24);

    for (std::size_t round = 0; round != rounds; ++round) {
        char const *const alphabet = alphabets[round % 3];
        std::size_t const needles_count = 1 + (round * 7) % 24;
        std::size_t const haystacks_count = 1 + (round * 5) % 9;

        std::vector<std::string> needles;
        for (std::size_t index = 0; index != needles_count; ++index)
            needles.push_back(random_string(generator, 1 + (index * 3 + round) % 7, alphabet));
        // A vocabulary may not repeat a needle under folding either, and a duplicate would make two
        // needle indices report the same span, which the oracle's own order could not tell apart.
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());

        std::vector<std::string> haystacks;
        for (std::size_t index = 0; index != haystacks_count; ++index)
            haystacks.push_back(random_string(generator, (index * 11 + round * 3) % 200, alphabet));

        check_policies_(tier, haystacks, needles, sz_substrings_cased_k);
        check_policies_(tier, haystacks, needles, sz_substrings_uncased_k);
    }

    // Haystacks spanning several ordered rounds, so leftmost covers stitch windows and rounds
    // together, and folded vocabularies meet single-byte and multi-byte text alike.
    for (std::size_t round = 0; round != context.iterations(6); ++round) {
        char const *const alphabet = alphabets[round % 3];
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 1 + (round * 5) % 17; ++index)
            needles.push_back(random_string(generator, 1 + (index + round) % 9, alphabet));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 3; ++index)
            haystacks.push_back(random_string(generator, 2048 * (1 + index) + round * 131, alphabet));
        std::string mixed = haystacks.back();
        mixed.insert(mixed.size() / 3, "Straße ÄÖÜ \xE2\x84\xAA");
        haystacks.push_back(mixed);
        std::string uppercased = haystacks.front();
        for (char &character : uppercased) character = (char)std::toupper((unsigned char)character);
        haystacks.push_back(uppercased);
        check_policies_(tier, haystacks, needles, sz_substrings_cased_k);
        check_policies_(tier, haystacks, needles, sz_substrings_uncased_k);
    }

    // Needles opening on letters the text rarely holds, so fewer than one byte in eight leaves the
    // root and the tiers skip between live bytes rather than stepping every one.
    for (std::size_t round = 0; round != context.iterations(4); ++round) {
        std::vector<std::string> const needles {"zebra", "quartz", "qu", "zz", "quiz", "z"};
        std::vector<std::string> haystacks;
        for (std::size_t index = 0; index != 4; ++index) {
            std::string haystack = random_string(generator, 3000 + index * 777 + round * 13,
                                                 "abcdefghijklmnoprstuvwxy");
            for (std::size_t insert = 0; insert != 20; ++insert)
                haystack.insert((insert * 997 + round) % haystack.size(), needles[insert % needles.size()]);
            haystacks.push_back(haystack);
        }
        haystacks.push_back("");
        haystacks.push_back("zzzz");
        check_policies_(tier, haystacks, needles, sz_substrings_cased_k);
        check_policies_(tier, haystacks, needles, sz_substrings_uncased_k);
    }

    // Small alphabets, where hot rows are only a few classes wide: nucleotides with bytes no needle
    // spells mixed in, a two-letter alphabet, and every tier split over them.
    for (std::size_t round = 0; round != context.iterations(4); ++round) {
        std::vector<std::string> nucleotides, bits, haystacks, binary_haystacks;
        for (std::size_t index = 0; index != 20 + round * 30; ++index)
            nucleotides.push_back(random_string(generator, 3 + (index + round) % 14, "ACGT"));
        for (std::size_t index = 0; index != 12; ++index)
            bits.push_back(random_string(generator, 1 + index % 9, std::string_view("\x00\x01", 2)));
        for (std::vector<std::string> *vocabulary : {&nucleotides, &bits}) {
            std::sort(vocabulary->begin(), vocabulary->end());
            vocabulary->erase(std::unique(vocabulary->begin(), vocabulary->end()), vocabulary->end());
        }
        for (std::size_t index = 0; index != 3; ++index) {
            std::string haystack = random_string(generator, 1500 + index * 2111 + round * 7, "ACGT");
            for (std::size_t offset = 97; offset < haystack.size(); offset += 389)
                haystack[offset] = "N\n>"[offset % 3];
            haystacks.push_back(haystack);
            binary_haystacks.push_back(random_string(generator, 2000 + index * 333, std::string_view("\x00\x01", 2)));
        }
        haystacks.push_back("ACGTNNACGT\nACG");
        check_policies_(tier, haystacks, nucleotides, sz_substrings_cased_k);
        check_policies_(tier, binary_haystacks, bits, sz_substrings_cased_k);
        if (round == 0) check_tier_splits_(tier, haystacks, nucleotides, sz_substrings_cased_k);
    }

    // Every byte value starts some needle, so no class is shared and a hot row spans 256 columns.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t byte = 0; byte != 256; ++byte)
            needles.push_back(std::string(1, (char)byte) + std::string(1, (char)((byte * 7 + 3) & 0xFF)));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 3; ++index) {
            std::string haystack(3000 + index * 101, '\0');
            for (std::size_t offset = 0; offset != haystack.size(); ++offset)
                haystack[offset] = (char)((offset * 31 + index * 17 + (offset * offset) % 13) & 0xFF);
            haystacks.push_back(haystack);
        }
        check_policies_(tier, haystacks, needles, sz_substrings_cased_k);
    }

    // A vocabulary wide enough to push the double array past its first growth, and deep enough that
    // the failure chains it packs are longer than one edge.
    {
        std::vector<std::string> needles, haystacks;
        for (std::size_t index = 0; index != 400; ++index)
            needles.push_back(random_string(generator, 2 + index % 9, "abcdefghijklmnopqrstuvwxyz"));
        std::sort(needles.begin(), needles.end());
        needles.erase(std::unique(needles.begin(), needles.end()), needles.end());
        for (std::size_t index = 0; index != 8; ++index)
            haystacks.push_back(random_string(generator, 500 + index * 37, "abcdefghijklmnopqrstuvwxyz"));
        check_policies_(tier, haystacks, needles, sz_substrings_cased_k);
    }
}

#pragma endregion Substrings

#pragma region UTF8 Runes

/** Repeats one UTF-8 encoded codepoint @p repeats times, giving a run of a single byte-width. */
inline std::string uniform_utf8_run_(char const *encoded_rune, std::size_t repeats) {
    std::string text;
    text.reserve(std::strlen(encoded_rune) * repeats);
    for (std::size_t repeat = 0; repeat != repeats; ++repeat) text += encoded_rune;
    return text;
}

/**
 *  @brief Streams @c unpack over the entire @p text, collecting every decoded rune.
 *
 *  Mirrors the documented streaming contract: each call decodes a prefix of the remaining bytes and
 *  reports how many runes it produced and how many bytes it consumed; every call must advance, so
 *  the loop terminates.
 *
 *  A small @p chunk_capacity forces the capacity-limited resume path, where the decoder fills the
 *  buffer, returns mid-input, and resumes after the consumed bytes.
 */
inline void collect_unpacked_runes_(sz_kernel_utf8_decode_t unpack, sz_cptr_t text, sz_size_t length,
                                    sz_size_t chunk_capacity, std::vector<sz_rune_t> &out) {
    out.clear();
    verify(chunk_capacity <= 64u); // The stack buffer below
    sz_size_t offset = 0;
    while (offset < length) {
        sz_rune_t chunk_runes[64];
        sz_size_t chunk_count = 0;
        sz_size_t const consumed = kernel_result<sz_size_t>(unpack, text + offset, length - offset, chunk_runes,
                                                            chunk_capacity, &chunk_count);
        verify(consumed > 0); // Must advance, otherwise the streaming loop would never terminate
        offset += consumed;
        // Fill-or-drain: a call that stops short of the end must have filled the capacity, so an
        // iterator decodes once per buffer whatever the script width, never once per register.
        verify((offset == length || chunk_count == chunk_capacity) && "Unpack stopped before filling capacity");
        for (sz_size_t index = 0; index != chunk_count; ++index) out.push_back(chunk_runes[index]);
    }
}

/** One backend's UTF-8 codepoint kernels: a capability's, or the dispatch points in their shape.
 *  The streaming decoder is null for a capability without one, like V128 Relaxed. */
struct utf8_runes_backend_t {
    char const *name;
    sz_kernel_utf8_count_t count;
    sz_kernel_utf8_seek_t seek;
    sz_kernel_utf8_decode_t decode;
};

/**
 *  @brief Holds one codepoint backend to the mixed-width anchor: its count, the byte offset of
 *      every codepoint, and the decoded runes at every caller capacity.
 *
 *  Mirrors @c check_sha256_unit_: driven once per backend, so a wrong constant shared by the
 *  serial-vs-SIMD agreement tests is still caught against an external ground truth.
 */
inline void check_utf8_runes_unit_(utf8_runes_backend_t const &backend) {
    // `a` (1 byte), U+00DF (2 bytes) and U+4E2D (3 bytes): 6 bytes, 3 codepoints.
    char const text[] = "a\xC3\x9F\xE4\xB8\xAD";
    sz_size_t const length = (sz_size_t)(sizeof(text) - 1);
    sz_rune_t const expected_runes[] = {0x61u, 0xDFu, 0x4E2Du};
    sz_size_t const expected_count = 3u;

    // Codepoint count is the external ground truth - not derived from a sibling kernel.
    verify(kernel_result<sz_size_t>(backend.count, text, length) == expected_count);

    // `sz_utf8_seek_best`: codepoint starts land at the byte offset of each rune; the
    // one-past-the-end index returns the NUL sentinel.
    sz_size_t byte_offset = 0;
    for (sz_size_t rune_index = 0; rune_index != expected_count; ++rune_index) {
        verify(kernel_result<sz_cptr_t>(backend.seek, text, length, rune_index) == text + byte_offset);
        sz_u8_t bytes[4];
        byte_offset += (sz_size_t)sz_rune_encode(expected_runes[rune_index], bytes);
    }
    verify(kernel_result<sz_cptr_t>(backend.seek, text, length, expected_count) ==
           STRINGZILLA_NULL_CHAR); // Beyond the last codepoint

    // `sz_utf8_decode_best`: streaming the decoder must reproduce exactly the expected runes at
    // every caller capacity, the tiny ones landing mid-rune-sequence on each resume.
    if (!backend.decode) return;

    // The end of the text ends the input, so a truncated final sequence decodes to one U+FFFD.
    sz_rune_t tail_runes[4];
    sz_size_t tail_count = 0;
    verify(kernel_result<sz_size_t>(backend.decode, "a\xE2\x82", 3, tail_runes, 4, &tail_count) == 3u);
    verify(tail_count == 2 && tail_runes[0] == 'a' && tail_runes[1] == 0xFFFDu);

    sz_size_t const capacities[] = {1u, 2u, 3u, 16u, 64u};
    std::vector<sz_rune_t> decoded;
    for (sz_size_t capacity : capacities) {
        collect_unpacked_runes_(backend.decode, text, length, capacity, decoded);
        verify(decoded.size() == expected_count);
        for (sz_size_t index = 0; index != expected_count; ++index) verify(decoded[index] == expected_runes[index]);
    }
}

/**
 *  @brief Cross-checks one codepoint backend against the serial kernels on random, well-formed
 *      inputs: chunk-unpacked runes, nth-codepoint byte offsets, and counts.
 *
 *  The known-answer anchor lives in @c check_utf8_runes_unit_; this differential is the only
 *  coverage that exercises the SIMD decode and seek kernels beyond it. Each input is checked at
 *  every sub-cache-line offset, over five decoder capacities.
 */
inline void check_utf8_runes_equivalence_(test_context_t &context, utf8_runes_backend_t const &candidate) {
    std::mt19937 &generator = context.generator;
    std::vector<sz_rune_t> runes_serial, runes_candidate;
    std::vector<sz_cptr_t> offsets_serial;

    auto check = [&](sz_cptr_t data, sz_size_t length) {
        sz_size_t const count_reference = kernel_result<sz_size_t>(sz_utf8_count_serial, data, length);

        // `sz_utf8_seek_best`: sweep every codepoint index plus a couple past the end.
        offsets_serial.clear();
        for (sz_size_t rune_index = 0; rune_index <= count_reference + 2u; ++rune_index)
            offsets_serial.push_back(kernel_result<sz_cptr_t>(sz_utf8_seek_serial, data, length, rune_index));

        collect_unpacked_runes_(sz_utf8_decode_serial, data, length, 64u, runes_serial);
        verify(runes_serial.size() == count_reference);

        verify(kernel_result<sz_size_t>(candidate.count, data, length) == count_reference);
        for (sz_size_t rune_index = 0; rune_index <= count_reference + 2u; ++rune_index)
            verify(kernel_result<sz_cptr_t>(candidate.seek, data, length, rune_index) == offsets_serial[rune_index]);
        if (!candidate.decode) return;

        // `sz_utf8_decode_best`: streaming both backends must decode identical runes, and the
        // candidate must decode the same sequence at every caller capacity - a capacity below the
        // input's rune count exercises the fill-return-resume path no whole-input capacity reaches.
        sz_size_t const capacities[] = {1u, 2u, 3u, 17u, 64u};
        for (sz_size_t capacity : capacities) {
            collect_unpacked_runes_(candidate.decode, data, length, capacity, runes_candidate);
            verify(runes_candidate == runes_serial);
        }
    };
    auto check_text_ = [&](std::string const &text) { check(text.data(), (sz_size_t)text.size()); };

    // Structured length ladder around the SIMD window boundaries, every codepoint count exercised.
    sz_size_t const ladder[] = {0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 100u, 200u};
    for (sz_size_t codepoints : ladder) check_text_(random_valid_utf8_(codepoints, generator));

    // Byte-exact ladder: the codepoint ladder above lands on arbitrary byte lengths, so the
    // 16/32/64-byte vector widths and their neighbours are otherwise only hit by chance.
    sz_size_t const byte_ladder[] = {15u, 16u, 17u, 31u, 32u, 33u, 47u, 48u, 63u, 64u, 65u, 127u, 128u, 129u};
    for (sz_size_t bytes : byte_ladder) check_text_(random_valid_utf8_bytes_(bytes, generator));

    // Homogeneous runs spanning several windows: the mixed generator never emits a long
    // single-width stretch, where a width-specialized fast path runs unbroken.
    char const *const uniform_runes[] = {"x", "\xD0\x9F", "\xE4\xB8\x96", "\xF0\x9F\x98\x80"};
    sz_size_t const uniform_repeats[] = {17u, 65u, 200u};
    for (char const *encoded_rune : uniform_runes)
        for (sz_size_t repeats : uniform_repeats) check_text_(uniform_utf8_run_(encoded_rune, repeats));

    // Fuzzed inputs of random codepoint counts, each placed at every sub-cache-line offset, so
    // serial-vs-ISA agreement is checked across all alignments the SIMD kernels may hit.
    std::uniform_int_distribution<std::size_t> codepoint_distribution(0, 96);
    std::size_t const inputs = context.iterations(1000);
    for (std::size_t iteration = 0; iteration != inputs; ++iteration) {
        std::string const text = random_valid_utf8_(codepoint_distribution(generator), generator);
        for_each_cacheline_offset_(text.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
            std::memcpy(buffer, text.data(), text.size());
            check(buffer, (sz_size_t)text.size());
        });
    }
}

/**
 *  @brief Feeds malformed / invalid UTF-8 through one backend's counting and streaming-unpack
 *      kernels, asserting no crash, in-bounds output, and consumption that never escapes the input.
 *
 *  Both kernels are total on arbitrary bytes, so both face the full malformed battery. Counting
 *  must merely survive; decoding substitutes U+FFFD for ill-formed input, and each call must report
 *  no more runes than the destination holds and never consume past the input.
 */
inline void check_utf8_runes_safety_(test_context_t &context, utf8_runes_backend_t const &backend) {
    std::size_t const random_inputs = context.iterations(4000);

    std::size_t const max_input_length = utf8_unit_capacity_k;
    std::vector<sz_rune_t> rune_destination;

    auto check = [&](char const *input, std::size_t input_length) {
        // Counting must survive any bytes and never report more runes than the input holds bytes.
        sz_size_t const counted = kernel_result<sz_size_t>(backend.count, input, (sz_size_t)input_length);
        verify(counted <= input_length && "Count reported more runes than the input holds bytes");

        // Every emitted rune must be a valid Unicode scalar value (no surrogate, none beyond
        // U+10FFFF) - the precondition every binding relies on to convert runes without
        // re-validation - and only a full output may stop a call short of the end.
        if (!backend.decode) return;
        // The tiny capacities force the fill-and-resume path to restart inside malformed bytes,
        // which the whole-input capacity never does.
        std::size_t const capacities[] = {1, 3, max_input_length + 4};
        for (std::size_t rune_capacity : capacities) {
            rune_destination.assign(rune_capacity, (sz_rune_t)0);
            sz_size_t offset = 0;
            while (offset < input_length) {
                sz_size_t produced = 0;
                sz_size_t const consumed = kernel_result<sz_size_t>(
                    backend.decode, input + offset, (sz_size_t)(input_length - offset), rune_destination.data(),
                    (sz_size_t)rune_capacity, &produced);
                verify(produced <= rune_capacity && "Unpack reported more runes than the destination holds");
                verify(consumed != 0 && consumed <= input_length - offset && "Unpack must advance within the input");
                verify((produced == rune_capacity || consumed == input_length - offset) &&
                       "Unpack stopped short of the end without filling its output");
                for (sz_size_t rune_index = 0; rune_index != produced; ++rune_index) {
                    sz_rune_t const rune = rune_destination[rune_index];
                    verify(rune <= 0x10FFFFu && !(rune >= 0xD800u && rune <= 0xDFFFu) &&
                           "Unpack emitted a non-scalar value (surrogate or out of range)");
                }
                offset += consumed;
            }
        }
    };

    std::mt19937 &generator = context.generator;
    for_each_adversarial_utf8_input_(context, random_inputs, check);

    // Valid text with a few bytes overwritten: damage surrounded by long well-formed runs, which
    // neither the battery's uniform garbage nor the well-formed equivalence corpus produces.
    std::uniform_int_distribution<std::size_t> length_distribution(1, max_input_length);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    std::uniform_int_distribution<std::size_t> codepoint_distribution(1, max_input_length / 4);
    for (std::size_t iteration = 0; iteration != random_inputs / 8 + 1; ++iteration) {
        std::string text = random_valid_utf8_(codepoint_distribution(generator), generator);
        if (text.size() > max_input_length) text.resize(max_input_length);
        for (std::size_t corruption = 0; corruption != 3; ++corruption)
            text[length_distribution(generator) % text.size()] = (char)byte_distribution(generator);
        check(text.data(), text.size());
    }
}

#pragma endregion UTF8 Runes

#pragma region UTF8 Tokens

/** One expected boundary match: the byte offset where it starts and its byte length. */
struct boundary_span_t {
    sz_size_t offset;
    sz_size_t length;
};

/**
 *  @brief Drain every match a tokenizer emits over the whole input, resuming via @c bytes_consumed
 *      so an arbitrarily small @p capacity yields the identical full match list.
 *
 *  Offsets are absolute. @p matcher is any callable with the @c sz_kernel_utf8_tokenizer_t
 *  signature - a kernel pointer or a @c cpu_best dispatch point - so newlines, whitespaces and
 *  delimiters all share this one driver, and every batch is held to the resume guarantee: a full
 *  batch consumes exactly through its last match, any other batch the whole input.
 */
template <typename matcher_type_>
void drain_matches_(matcher_type_ &&matcher, sz_cptr_t text, sz_size_t length, sz_size_t capacity,
                    std::vector<sz_size_t> &offsets, std::vector<sz_size_t> &lengths) {
    offsets.clear(), lengths.clear();
    std::vector<sz_size_t> offset_batch(capacity ? capacity : 1), length_batch(capacity ? capacity : 1);
    sz_size_t position = 0;
    while (position < length) {
        sz_size_t emitted = 0, consumed = 0;
        verify(matcher(text + position, length - position, offset_batch.data(), length_batch.data(), capacity, &emitted,
                       &consumed, nullptr) == sz_success_k);
        sz_size_t const last_end = emitted ? offset_batch[emitted - 1] + length_batch[emitted - 1] : 0;
        verify(consumed == (emitted == capacity ? last_end : length - position) &&
               "A full batch must resume right after its last match, any other one at the end of the input");
        for (sz_size_t index = 0; index != emitted; ++index)
            offsets.push_back(position + offset_batch[index]), lengths.push_back(length_batch[index]);
        if (consumed == 0) break; // No forward progress: stop rather than spin.
        position += consumed;
    }
}

/**
 *  @brief Reconstruct the SEGMENTS the C++/Python/Rust split iterators would yield - the gap before
 *      each match, plus the trailing gap once the input is exhausted - advancing the suffix by
 *      @c bytes_consumed after every batch.
 *
 *  This is the consumer's view, and it sees what @ref drain_matches_ structurally cannot: a
 *  @c bytes_consumed that overshoots the end of the last emitted match. The match list is identical
 *  either way, because the skipped span holds no matches - but the caller derives the next
 *  segment's start from @c bytes_consumed, so those bytes silently vanish from their segment.
 */
template <typename matcher_type_>
void reconstruct_segments_(matcher_type_ &&matcher, sz_cptr_t text, sz_size_t length, sz_size_t capacity,
                           std::vector<sz_size_t> &offsets, std::vector<sz_size_t> &lengths) {
    offsets.clear(), lengths.clear();
    std::vector<sz_size_t> offset_batch(capacity ? capacity : 1), length_batch(capacity ? capacity : 1);
    sz_size_t suffix = 0;
    for (;;) {
        sz_size_t const region = length - suffix;
        sz_size_t emitted = 0, consumed = 0;
        verify(matcher(text + suffix, region, offset_batch.data(), length_batch.data(), capacity, &emitted, &consumed,
                       nullptr) == sz_success_k);
        verify(consumed <= region && "Segmenter consumed past the input");
        sz_size_t previous_end = 0;
        for (sz_size_t index = 0; index != emitted; ++index) {
            offsets.push_back(suffix + previous_end), lengths.push_back(offset_batch[index] - previous_end);
            previous_end = offset_batch[index] + length_batch[index];
        }
        if (consumed == region) { // Exhausted: the trailing segment runs to end-of-text.
            offsets.push_back(suffix + previous_end), lengths.push_back(region - previous_end);
            break;
        }
        if (consumed == 0) break; // No forward progress: stop rather than spin.
        suffix += consumed;
    }
}

/** Repeats @p pattern until the text is exactly @p bytes long, so a scan ends precisely on a
 *  vector-window or multistep edge rather than wherever a codepoint ladder happens to land. */
inline std::string exact_byte_length_(char const *pattern, std::size_t pattern_length, std::size_t bytes) {
    std::string text;
    text.reserve(bytes + pattern_length);
    while (text.size() < bytes) text.append(pattern, pattern_length);
    text.resize(bytes); // A truncated multi-byte tail sitting on the edge is itself worth probing
    return text;
}

/** One backend's UTF-8 count, newline and whitespace kernels: a capability's, or the dispatch
 *  points in their shape. V128 Relaxed has its own counter over the V128 tokenizers. */
struct utf8_tokens_backend_t {
    char const *name;
    sz_kernel_utf8_count_t count;
    sz_kernel_utf8_tokenizer_t newlines;
    sz_kernel_utf8_tokenizer_t whitespaces;
};

/** One backend's delimiter tokenizer: a capability's, or the dispatch point in its shape. */
struct utf8_delimiters_backend_t {
    char const *name;
    sz_kernel_utf8_tokenizer_t finder;
};

/**
 *  @brief Holds one backend's counting and boundary-finding kernels to the known-answer anchors:
 *      the codepoint count, and the exact (offset, length) newline and whitespace spans.
 *
 *  Mirrors @c check_sha256_unit_: driven once per backend, so a wrong constant shared by the
 *  serial-vs-SIMD agreement tests is still caught against these external ground-truth vectors.
 */
inline void check_utf8_tokens_unit_(utf8_tokens_backend_t const &backend) {
    // "aß中" is `a` (1 byte) + `ß` U+00DF (2 bytes) + `中` U+4E2D (3 bytes): 6 bytes, 3 codepoints.
    char const mixed[] = "a\xC3\x9F\xE4\xB8\xAD";
    verify(kernel_result<sz_size_t>(backend.count, mixed, (sz_size_t)(sizeof(mixed) - 1)) == 3u);

    auto check_boundaries_ = [](sz_kernel_utf8_tokenizer_t finder, sz_cptr_t text, sz_size_t length,
                                std::initializer_list<boundary_span_t> expected) {
        sz_size_t found_offsets[16], found_lengths[16], found = 0, consumed = 0;
        verify(finder(text, length, found_offsets, found_lengths, 16u, &found, &consumed, nullptr) == sz_success_k);
        verify(found == expected.size());
        sz_size_t index = 0;
        for (boundary_span_t const &span : expected) {
            verify(found_offsets[index] == span.offset);
            verify(found_lengths[index] == span.length);
            ++index;
        }
    };

    // In "a\nb\r\nc" the `\n` is a length-1 newline at byte 1, and the `\r\n` is a single length-2
    // newline at byte 3 (CRLF merges into one match).
    char const newline_text[] = "a\nb\r\nc";
    check_boundaries_(backend.newlines, newline_text, (sz_size_t)(sizeof(newline_text) - 1), {{1u, 1u}, {3u, 2u}});

    // The space is a length-1 match at byte 1, the tab at byte 3, and U+200A HAIR SPACE (E2 80 8A)
    // a length-3 match at byte 5 (there is no CRLF merging in the whitespace set - each codepoint
    // is its own match). The U+200B/200C/200D (ZERO WIDTH SPACE/NON-JOINER/JOINER, E2 80 8B/8C/8D)
    // that follow are Format characters with White_Space=No and must produce NO match - this pins
    // the E2 80 [80-8A] block boundary so no backend regresses to splitting on the zero-width
    // joiners (which would shatter ZWJ emoji and Indic/Arabic words).
    char const whitespace_text[] = "a b\tc\xE2\x80\x8A"                       // ... U+200A HAIR SPACE (whitespace)
                                   "d" "\xE2\x80\x8B\xE2\x80\x8C\xE2\x80\x8D" // U+200B/200C/200D (not whitespace)
                                   "e";
    check_boundaries_(backend.whitespaces, whitespace_text, (sz_size_t)(sizeof(whitespace_text) - 1),
                      {{1u, 1u}, {3u, 1u}, {5u, 3u}});
}

/**
 *  @brief Tests one backend's UTF-8 count/newline/whitespace kernels against serial.
 *
 *  Generates random strings containing:
 *  - ASCII content (1-byte)
 *  - Multi-byte UTF-8 characters (2, 3, 4-byte) - correct and broken ones
 *  - All 25 Unicode White_Space characters (including all newlines) and CRLF sequences, both
 *    correct and partial ones
 *
 *  For each generated string, compares the codepoint count and the newline and whitespace
 *  matches (position and matched length), as match lists and as reconstructed segments.
 */
inline void check_utf8_tokens_equivalence_(test_context_t &context, utf8_tokens_backend_t const &candidate) {
    std::mt19937 &generator = context.generator;
    utf8_tokens_backend_t const reference {"serial", sz_utf8_count_serial, sz_utf8_newlines_serial,
                                           sz_utf8_whitespaces_serial};

    auto check = [&](sz_cptr_t data, sz_size_t length) {
        verify(kernel_result<sz_size_t>(reference.count, data, length) ==
               kernel_result<sz_size_t>(candidate.count, data, length));

        // Sweep capacities: one huge (one-shot), the awkward 65/63 straddling the 64-byte AVX-512
        // window, the binding default 16, and tiny 3/1 - stressing the per-window / mid-window
        // capacity cut at every boundary.
        sz_size_t const capacities[] = {length + 64, 65, 63, 16, 3, 1};
        std::vector<sz_size_t> reference_offsets, reference_lengths, candidate_offsets, candidate_lengths;
        for (sz_size_t capacity : capacities) {
            drain_matches_(reference.newlines, data, length, capacity, reference_offsets, reference_lengths);
            drain_matches_(candidate.newlines, data, length, capacity, candidate_offsets, candidate_lengths);
            verify(reference_offsets == candidate_offsets && "Mismatch in newline offsets");
            verify(reference_lengths == candidate_lengths && "Mismatch in newline lengths");

            drain_matches_(reference.whitespaces, data, length, capacity, reference_offsets, reference_lengths);
            drain_matches_(candidate.whitespaces, data, length, capacity, candidate_offsets, candidate_lengths);
            verify(reference_offsets == candidate_offsets && "Mismatch in whitespace offsets");
            verify(reference_lengths == candidate_lengths && "Mismatch in whitespace lengths");

            // Iterator segments catch a `bytes_consumed` overshoot at a window-aligned fill.
            reconstruct_segments_(reference.newlines, data, length, capacity, reference_offsets, reference_lengths);
            reconstruct_segments_(candidate.newlines, data, length, capacity, candidate_offsets, candidate_lengths);
            verify(reference_offsets == candidate_offsets && reference_lengths == candidate_lengths &&
                   "Mismatch in newline segments");
            reconstruct_segments_(reference.whitespaces, data, length, capacity, reference_offsets, reference_lengths);
            reconstruct_segments_(candidate.whitespaces, data, length, capacity, candidate_offsets, candidate_lengths);
            verify(reference_offsets == candidate_offsets && reference_lengths == candidate_lengths &&
                   "Mismatch in whitespace segments");
        }
    };

    // Strings that shouldn't affect the control flow
    static char const *const utf8_content[] = {
        // Various ASCII strings
        "",
        "a",
        "hello",
        "012",
        "3456789",
        // 2-byte Cyrillic П (U+041F), Armenian Ս (U+054D), and Greek Pi π (U+03C0)
        "\xD0\x9F",
        "\xD5\xA5",
        "\xCF\x80",
        // 3-byte characters
        "\xE0\xA4\xB9", // Hindi ह (U+0939)
        "\xE1\x88\xB4", // Ethiopic ሴ (U+1234)
        "\xE2\x9C\x94", // Check mark ✔ (U+2714)
        // 4-byte emojis: U+1F600 (😀), U+1F601 (😁), U+1F602 (😂)
        "\xF0\x9F\x98\x80",
        "\xF0\x9F\x98\x81",
        "\xF0\x9F\x98\x82",
        // Characters with bytes in 0x80-0x8F range (tests unsigned comparison in SIMD)
        "\xE2\x82\x80", // U+2080 SUBSCRIPT ZERO (has 0x80 suffix, not whitespace)
        "\xE2\x84\x8A", // U+210A SCRIPT SMALL G (has 0x8A like HAIR SPACE suffix)
        "\xE2\x84\x8D", // U+210D DOUBLE-STRUCK H (has 0x8D suffix)
        // Near-miss characters (same prefix as whitespace but different suffix)
        "\xE2\x80\xB0", // U+2030 PER MILLE SIGN (E2 80 prefix like whitespace range)
        "\xE2\x80\xBB", // U+203B REFERENCE MARK (E2 80 prefix)
        "\xE2\x81\xA0", // U+2060 WORD JOINER (E2 81 prefix like MMSP)
        "\xE3\x80\x81", // U+3001 IDEOGRAPHIC COMMA (E3 80 prefix like IDEOGRAPHIC SPACE)
        "\xE3\x80\x82", // U+3002 IDEOGRAPHIC FULL STOP
        // More 4-byte sequences for boundary handling
        "\xF0\x9F\x8E\x89", // U+1F389 PARTY POPPER 🎉
        "\xF0\x9F\x92\xA9", // U+1F4A9 PILE OF POO 💩
    };

    // Special characters that will affect control flow
    static char const *const special_chars[26] = {
        "\x09",         "\x0A",         "\x0B",         "\x0C",         "\x0D",         " ", // 1-byte (6)
        "\xC2\x85",     "\xC2\xA0",     "\r\n",                                              // 2-byte (2)
        "\xE1\x9A\x80", "\xE2\x80\x80", "\xE2\x80\x81", "\xE2\x80\x82", "\xE2\x80\x83",      // 3-byte
        "\xE2\x80\x84", "\xE2\x80\x85", "\xE2\x80\x86", "\xE2\x80\x87", "\xE2\x80\x88",      // 3-byte
        "\xE2\x80\x89", "\xE2\x80\x8A", "\xE2\x80\xA8", "\xE2\x80\xA9", "\xE2\x80\xAF",      // 3-byte
        "\xE2\x81\x9F", "\xE3\x80\x80",                                                      // 3-byte
    };

    std::size_t const utf8_content_count = span_over(utf8_content).size();
    std::size_t const special_delimiter_count = span_over(special_chars).size();
    std::size_t const total_strings_to_sample = utf8_content_count + special_delimiter_count;
    std::uniform_int_distribution<std::size_t> content_dist(0, total_strings_to_sample - 1);

    // Byte-exact edges. The vector windows are 16/32/64 bytes wide, but these scanners advance 14
    // bytes per step on NEON, 30 on AVX2 and 62 on AVX-512, so multiples of both families - and
    // their neighbours - are pinned here. The repeated pattern carries a CRLF, a 3-byte U+2028 LINE
    // SEPARATOR and a 2-byte U+00A0 NO-BREAK SPACE, so a length cut mid-sequence leaves a truncated
    // delimiter sitting exactly on the edge.
    static sz_size_t const byte_lengths[] = {13, 14, 15,  16,  17,  27,  28,  29,  30,  31,  32, 33, 41,
                                             42, 43, 55,  56,  57,  59,  60,  61,  62,  63,  64, 65, 89,
                                             90, 91, 123, 124, 125, 127, 128, 129, 185, 186, 187};
    char const edge_pattern[] = "ab\r\ncd\xE2\x80\xA8" // Split, or the hex escape would swallow the `ef` after it
                                "ef\xC2\xA0gh";
    for (sz_size_t bytes : byte_lengths) {
        std::string const text = exact_byte_length_(edge_pattern, sizeof(edge_pattern) - 1, bytes);
        check(text.data(), (sz_size_t)text.size());
    }

    // Homogeneous runs several windows long: an all-match and an all-miss window drive the
    // classifier's saturated paths, which the mixed corpora below never reach.
    std::string const uniform_runs[] = {std::string(200, 'a'),      std::string(200, ' '),
                                        std::string(200, '\n'),     repeat("\r\n", 100),
                                        repeat("\xC2\xA0", 100),    repeat("\xE3\x80\x80", 70),
                                        repeat("\xE4\xB8\xAD", 70), repeat("\xF0\x9F\x98\x80", 50)};
    for (std::string const &run : uniform_runs) check(run.data(), (sz_size_t)run.size());

    // Random strings of at least 4 KB each, drained through six capacities down to 1, re-entering
    // the kernel once per match. The iteration count is this family's share of the suite budget,
    // sized against its siblings.
    std::size_t const min_text_length = 4000, iterations = context.iterations(250);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        std::string text;

        // Build up a random string of at least `min_text_length` bytes
        while (text.size() < min_text_length) {
            std::size_t random_content_index = content_dist(generator);
            if (random_content_index < utf8_content_count) { text.append(utf8_content[random_content_index]); }
            else { text.append(special_chars[random_content_index - utf8_content_count]); }
        }
        check(text.data(), text.size());

        // Replace 10% of the bytes with NUL, breaking many valid codepoints
        std::size_t num_bytes_to_corrupt = text.size() / 10;
        std::uniform_int_distribution<std::size_t> byte_index_dist(0, text.size() - 1);
        for (std::size_t i = 0; i < num_bytes_to_corrupt; ++i) {
            std::size_t byte_index = byte_index_dist(generator);
            text[byte_index] = '\0';
        }
        check(text.data(), text.size());

        // Swap 10% of bytes at random positions, creating malformed UTF-8 sequences
        for (std::size_t i = 0; i < num_bytes_to_corrupt; ++i) {
            std::size_t byte_index_1 = byte_index_dist(generator);
            std::size_t byte_index_2 = byte_index_dist(generator);
            std::swap(text[byte_index_1], text[byte_index_2]);
        }
        check(text.data(), text.size());
    }

    // Rerun count/newline/whitespace equivalence at every sub-cache-line byte offset, so the SIMD
    // load alignment, not just the content, is swept against the serial reference.
    std::string alignment_probe;
    while (alignment_probe.size() < 256) {
        std::size_t random_content_index = content_dist(generator);
        if (random_content_index < utf8_content_count) { alignment_probe.append(utf8_content[random_content_index]); }
        else { alignment_probe.append(special_chars[random_content_index - utf8_content_count]); }
    }
    for_each_cacheline_offset_(alignment_probe.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
        std::memcpy(buffer, alignment_probe.data(), alignment_probe.size());
        check(buffer, alignment_probe.size());
    });
}

/** Drives one backend's newline and whitespace finders through the malformed-input battery (named
 *  adversarial shapes, all 256 single bytes, all 65,536 byte pairs, and random garbage at every
 *  sub-cache-line alignment), asserting they survive, stay in bounds, and never report a
 *  @c bytes_consumed past the input. */
inline void check_utf8_tokens_safety_(test_context_t &context, utf8_tokens_backend_t const &backend) {
    static constexpr std::size_t max_input_length = utf8_unit_capacity_k;
    auto check = [&](char const *input, std::size_t input_length) {
        sz_size_t boundary_offsets[max_input_length + 1], boundary_lengths[max_input_length + 1];
        auto check_boundaries_ = [&](sz_kernel_utf8_tokenizer_t finder, char const *finder_name) {
            sz_size_t found = 0, bytes_consumed = 0;
            verify(finder(input, (sz_size_t)input_length, boundary_offsets, boundary_lengths,
                          (sz_size_t)(max_input_length + 1), &found, &bytes_consumed, nullptr) == sz_success_k);
            verify(bytes_consumed <= input_length && "Boundary finder consumed past the input");
            for (sz_size_t index = 0; index != found; ++index) {
                if (boundary_offsets[index] + boundary_lengths[index] <= input_length) continue;
                fmt::println(stderr, "{} {} emitted out-of-bounds boundary (offset={} len={}, input={})", backend.name,
                             finder_name, (std::size_t)boundary_offsets[index], (std::size_t)boundary_lengths[index],
                             input_length);
                print_utf8_test_bytes_("input", {input, input_length});
                verify(false && "Boundary finder emitted a span outside the input");
            }
        };
        check_boundaries_(backend.newlines, "newline finder");
        check_boundaries_(backend.whitespaces, "whitespace finder");
    };
    for_each_adversarial_utf8_input_(context, context.iterations(10000), check);
}

/** Known-answer checks of one UTF-8 delimiter segmenter on simple, hand-verifiable inputs, and on
 *  capacity-limited batches whose resume offset must land on the last emitted delimiter. */
inline void check_utf8_delimiters_unit_(utf8_delimiters_backend_t const &backend) {
    struct {
        char const *text;
        sz_size_t length, expected_offset, expected_length, expected_count;
    } const cases[] = {
        {"abc def", 7, 3, 1, 1},               // ASCII space (Zs) at byte 3
        {"hello", 5, 0, 0, 0},                 // all letters, no delimiter
        {"ab,", 3, 2, 1, 1},                   // ASCII comma (Po) at byte 2
        {"ab\xC2\xA0", 4, 2, 2, 1},            // U+00A0 NO-BREAK SPACE (Zs), 2 bytes at byte 2
        {"ab\xE2\x80\x94", 5, 2, 3, 1},        // U+2014 EM DASH (Pd), 3 bytes at byte 2
        {"ab\xF0\x9F\x98\x80", 6, 2, 4, 1},    // U+1F600 GRINNING FACE (So), 4 bytes at byte 2
        {"a\xC3\x9F\xE4\xB8\xAD", 6, 0, 0, 0}, // a + U+00DF + U+4E2D, all letters
        // U+FF0C FULLWIDTH COMMA, the last BMP block with delimiters
        {"a\xEF\xBC\x8C", 4, 1, 3, 1},
        // U+3000 IDEOGRAPHIC SPACE
        {"a\xE3\x80\x80", 4, 1, 3, 1},
        // U+FF21 FULLWIDTH LATIN CAPITAL LETTER A, next to the block of U+FF0C
        {"a\xEF\xBC\xA1", 4, 0, 0, 0},
        // empty input
        {"", 0, 0, 0, 0},
    };

    // Capacity-limited resume: when the output fills before the input is exhausted,
    // `bytes_consumed` must land exactly on the end of the last emitted delimiter. A window whose
    // delimiters fill the capacity exactly leaves no residue in the kernel's hit mask, and a driver
    // keying the resume offset off that residue instead advances to the vector window's edge,
    // swallowing every undelimited byte in between. Each probe therefore places its last emitted
    // delimiter far from that edge, so the correct and the buggy answer differ.
    struct resume_case_t {
        std::string text;
        sz_size_t capacity, expected_consumed;
    };
    std::vector<resume_case_t> resume_cases;
    {
        // One ASCII space at byte 1, then 62 letters running past the widest vector window.
        std::string text = "a ";
        text.append(62, 'b');
        text += " c";
        resume_cases.push_back({text, 1u, 2u});
    }
    {
        // Two delimiters well inside the first window; a capacity of 2 fills exactly on the second.
        std::string text = "a b,";
        text.append(60, 'c');
        text += " d";
        resume_cases.push_back({text, 2u, 4u});
    }
    {
        // A 3-byte delimiter, U+2014 EM DASH, ends at byte 5, with letters up to the window edge.
        std::string text = "ab\xE2\x80\x94";
        text.append(59, 'd');
        text += " e";
        resume_cases.push_back({text, 1u, 5u});
    }

    std::vector<sz_size_t> offsets, lengths;
    for (auto const &one : cases) {
        drain_matches_(backend.finder, one.text, one.length, one.length + 1, offsets, lengths);
        verify(offsets.size() == one.expected_count && "Delimiter count mismatch");
        if (one.expected_count) {
            verify(offsets[0] == one.expected_offset && "Delimiter offset mismatch");
            verify(lengths[0] == one.expected_length && "Delimiter length mismatch");
        }
    }
    // A multi-byte delimiter straddling every vector window edge up to 128 bytes is found whole.
    for (std::string const &delimiter : {std::string("\xE3\x80\x81"), std::string("\xF0\x9F\x98\x80")})
        for (sz_size_t prefix = 0; prefix != 128; ++prefix) {
            std::string text(prefix, 'a');
            text += delimiter + "b";
            drain_matches_(backend.finder, text.data(), (sz_size_t)text.size(), (sz_size_t)text.size() + 1, offsets,
                           lengths);
            verify(offsets.size() == 1 && offsets[0] == prefix && lengths[0] == delimiter.size() &&
                   "Delimiter straddling a window edge was missed or split");
        }
    for (resume_case_t const &one : resume_cases) {
        sz_size_t batch_offsets[4], batch_lengths[4], emitted = 0, consumed = 0;
        verify(one.capacity <= 4 && "Resume probe capacity outgrew its output buffers");
        verify(backend.finder(one.text.data(), (sz_size_t)one.text.size(), batch_offsets, batch_lengths, one.capacity,
                              &emitted, &consumed, nullptr) == sz_success_k);
        verify(emitted == one.capacity && "Capacity-limited batch should fill the output");
        verify(batch_offsets[emitted - 1] + batch_lengths[emitted - 1] == one.expected_consumed &&
               "Last emitted delimiter ends elsewhere than the probe expects");
        verify(consumed == one.expected_consumed &&
               "Resume offset must be the end of the last emitted delimiter, not the vector window's edge");
    }
}

/** Cross-checks one UTF-8 delimiter segmenter against serial on random, well-formed inputs: the
 *  full (offset, length) match list must agree, both in one shot and when a tiny capacity drains
 *  the candidate through its @c bytes_consumed resume path. */
inline void check_utf8_delimiters_equivalence_(test_context_t &context, utf8_delimiters_backend_t const &candidate) {
    std::mt19937 &generator = context.generator;
    sz_kernel_utf8_tokenizer_t const finder_serial = sz_utf8_delimiters_serial;
    std::vector<sz_size_t> serial_offsets, serial_lengths, candidate_offsets, candidate_lengths, resumed_offsets,
        resumed_lengths;

    auto check = [&](sz_cptr_t data, sz_size_t length) {
        drain_matches_(finder_serial, data, length, length + 1, serial_offsets, serial_lengths);
        drain_matches_(candidate.finder, data, length, length + 1, candidate_offsets, candidate_lengths);
        verify(candidate_offsets == serial_offsets && "Mismatch in delimiter offsets");
        verify(candidate_lengths == serial_lengths && "Mismatch in delimiter lengths");
        // Resume path: a capacity of 3 forces re-entry, and the accumulated list must still match.
        drain_matches_(candidate.finder, data, length, 3u, resumed_offsets, resumed_lengths);
        verify(resumed_offsets == serial_offsets && "Resume-path delimiter offsets diverged");
        verify(resumed_lengths == serial_lengths && "Resume-path delimiter lengths diverged");

        // Segment-level (iterator) equivalence, mirroring the newline/whitespace differential:
        // sweep the capacities that straddle the 64-byte vector window, so a batch filling exactly
        // at a window edge is caught. The delimiter lists above stay identical under such a fill -
        // only the segments shift.
        sz_size_t const capacities[] = {length + 64, 65, 64, 63, 16, 3, 2, 1};
        reconstruct_segments_(finder_serial, data, length, length + 64, serial_offsets, serial_lengths);
        for (sz_size_t capacity : capacities) {
            reconstruct_segments_(candidate.finder, data, length, capacity, candidate_offsets, candidate_lengths);
            verify(candidate_offsets == serial_offsets && "Mismatch in delimiter segment offsets");
            verify(candidate_lengths == serial_lengths && "Mismatch in delimiter segment lengths");
        }
    };

    sz_size_t const ladder[] = {0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 100u, 200u};
    for (sz_size_t codepoints : ladder) {
        std::string const text = random_valid_utf8_(codepoints, generator);
        check(text.data(), (sz_size_t)text.size());
    }

    // The ladder above counts codepoints, so a byte-exact window edge is only hit by accident.
    // These lengths land on and beside the 16/32/64-byte vector windows, with a 3-byte U+2014 EM
    // DASH in the pattern so a cut can leave a truncated delimiter on the edge itself.
    char const edge_pattern[] = "ab cd,ef\xE2\x80\x94gh";
    sz_size_t const byte_lengths[] = {15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u};
    for (sz_size_t bytes : byte_lengths) {
        std::string const text = exact_byte_length_(edge_pattern, sizeof(edge_pattern) - 1, bytes);
        check(text.data(), (sz_size_t)text.size());
    }

    // Homogeneous runs several windows long: all-letter, all-space and all-delimiter windows drive
    // the saturated classifier paths that the mixed corpora never reach.
    std::string const uniform_runs[] = {std::string(200, 'a'), std::string(200, ' '), std::string(200, ','),
                                        repeat("\xE2\x80\x94", 70), repeat("\xE4\xB8\xAD", 70)};
    for (std::string const &run : uniform_runs) check(run.data(), (sz_size_t)run.size());

    std::uniform_int_distribution<std::size_t> codepoint_distribution(0, 300);
    std::size_t const inputs = context.iterations(700);
    for (std::size_t iteration = 0; iteration != inputs; ++iteration) {
        std::string const text = random_valid_utf8_(codepoint_distribution(generator), generator);
        // The probe buffer itself is handed to the kernels: copying it back into a fresh
        // `std::string` would hand them whatever alignment the allocator picked, and the sweep
        // would test one alignment repeatedly.
        for_each_cacheline_offset_(text.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
            std::memcpy(buffer, text.data(), text.size());
            check(buffer, (sz_size_t)text.size());
        });
    }
}

/** Feeds malformed UTF-8 to a delimiter segmenter, asserting in-bounds, ascending, valid spans. */
inline void check_utf8_delimiters_safety_(test_context_t &context, utf8_delimiters_backend_t const &backend) {
    std::vector<sz_size_t> offsets, lengths;

    // Malformed bytes meet a capacity too small to hold the batch, so the resume path - not just
    // the one-shot drain - must keep emitting in-bounds, ascending, well-formed spans.
    auto check = [&](char const *input, std::size_t input_length) {
        sz_size_t const capacities[] = {(sz_size_t)input_length + 1, 3u, 2u, 1u};
        for (sz_size_t capacity : capacities) {
            drain_matches_(backend.finder, input, (sz_size_t)input_length, capacity, offsets, lengths);
            sz_size_t previous_end = 0;
            for (std::size_t index = 0; index != offsets.size(); ++index) {
                verify(lengths[index] >= 1u && lengths[index] <= 4u && "Delimiter matched an impossible byte length");
                verify(offsets[index] + lengths[index] <= input_length && "Delimiter match span outside the input");
                verify(offsets[index] >= previous_end && "Delimiter matches must be ascending and non-overlapping");
                previous_end = offsets[index] + lengths[index];
            }
        }
    };

    for_each_adversarial_utf8_input_(context, context.iterations(2500), check);
}

#pragma endregion UTF8 Tokens

#pragma region UTF8 Wordbreaks

/** Hand-checked UAX-29 word-break golden vectors: each source text and its expected words. */
inline utf8_unit_case_t const utf8_wordbreaks_unit_cases[] = {
    {""sv, {}},
    {"a"sv, {"a"sv}},
    {"don't"sv, {"don't"sv}},                                             // WB6/7: apostrophe bridges letter runs
    {"3,14"sv, {"3,14"sv}},                                               // WB11/12: comma between digits
    {"3,"sv, {"3"sv, ","sv}},                                             // digit then bare comma breaks
    {"can't_stop"sv, {"can't_stop"sv}},                                   // WB13a/b: ExtendNumLet underscore
    {"a\r\nb"sv, {"a"sv, "\r\n"sv, "b"sv}},                               // WB3: CR x LF stay together
    {"\xE4\xBD\xA0\xE5\xA5\xBD"sv, {"\xE4\xBD\xA0"sv, "\xE5\xA5\xBD"sv}}, // CJK: each ideograph is its own word
    {"Hello, world!"sv, {"Hello"sv, ","sv, " "sv, "world"sv, "!"sv}},     // letter/punct/space boundaries
};

/** Segment @p text with @p forward at @p capacity and assert the segments have exactly
 *  @p expected_lengths. */
inline void check_utf8_wordbreaks_lengths_(char const *label, sz_kernel_utf8_segmenter_t forward,
                                           std::string const &text, std::vector<sz_size_t> const &expected_lengths,
                                           sz_size_t capacity) {
    utf8_segment_cursor_t cursor = utf8_segment_cursor_make_(forward, text.data(), text.size(), capacity);
    sz_size_t start = 0, length = 0;
    for (sz_size_t const expected : expected_lengths) {
        sz_bool_t const more = utf8_segment_cursor_next_(cursor, start, length);
        verify(more && label && "deferred-mid golden emitted fewer segments than expected");
        verify(length == expected && label && "deferred-mid golden segment mismatch");
    }
    verify(!utf8_segment_cursor_next_(cursor, start, length) && label && "deferred-mid golden emitted extra segments");
}

/** WB6/WB7/WB11/WB12 deferred-mid goldens: a Mid* (`,` `:` `'` `"`) whose WB4-ignorable lookahead
 *  run crosses every SIMD window width (16/64 bytes) before the bridge completes or fails on the
 *  far side. Regression corpus for the cross-window bridge-shadow carry: a failed bridge must still
 *  emit the break at the mid, keep WB7a's Hebrew x Single_Quote join, and leave the mid as the
 *  effective left context. */
inline void check_utf8_wordbreaks_deferred_mid_(utf8_segment_backend_t const &backend) {
    static sz_size_t const run_lengths[] = {3, 8, 15, 31, 32, 33, 100};
    static char const combining_grave[] = "\xCC\x80";   // U+0300, Word_Break=Extend
    static char const hebrew_he[] = "\xD7\x94";         // U+05D4, Hebrew_Letter
    static char const emoji[] = "\xF0\x9F\x98\x80";     // U+1F600, Extended_Pictographic
    static char const zwj[] = "\xE2\x80\x8D";           // U+200D, ZWJ
    static char const math_five[] = "\xF0\x9D\x9F\x97"; // U+1D7D7, astral Numeric

    struct golden_t {
        std::string text;
        std::vector<sz_size_t> lengths;
    };
    std::vector<golden_t> goldens;
    for (sz_size_t const run : run_lengths) {
        std::string marks;
        for (sz_size_t i = 0; i != run; ++i) marks += combining_grave;
        sz_size_t const marks_size = marks.size();
        goldens.push_back({"5," + marks + "6", {3 + marks_size}});                                // WB11 completes
        goldens.push_back({"5," + marks + math_five, {6 + marks_size}});                          // WB11 astral Numeric
        goldens.push_back({"a:" + marks + "b", {3 + marks_size}});                                // WB6 completes
        goldens.push_back({hebrew_he + std::string("\"") + marks + hebrew_he, {5 + marks_size}}); // WB7b completes
        goldens.push_back({"5," + marks + "a", {1, 1 + marks_size, 1}});                          // WB11 fails
        goldens.push_back({"a:" + marks + "5", {1, 1 + marks_size, 1}});                          // WB6 fails
        goldens.push_back({"5," + marks, {1, 1 + marks_size}});                               // fails at end-of-text
        goldens.push_back({"5," + marks + ",5", {1, 1 + marks_size, 1, 1}});                  // fails into a 2nd mid
        goldens.push_back({hebrew_he + std::string("'") + marks + "x", {4 + marks_size}});    // WB7 completes
        goldens.push_back({hebrew_he + std::string("'") + marks + "5", {3 + marks_size, 1}}); // WB7a keeps the quote
        goldens.push_back({hebrew_he + std::string("\"") + marks + "x", {2, 1 + marks_size, 1}}); // WB7b fails
        goldens.push_back({"a:" + marks + zwj + emoji, {1, 8 + marks_size}}); // fails; WB3c joins the pictograph
    }
    goldens.push_back({std::string("5,") + emoji, {1, 1, 4}}); // astral lookahead right after the mid

    static sz_size_t const capacities[] = {utf8_segment_batch_k, 1};
    for (golden_t const &golden : goldens)
        for (sz_size_t const capacity : capacities)
            check_utf8_wordbreaks_lengths_(backend.name, backend.finder, golden.text, golden.lengths, capacity);
}

/** UAX-29 word-break corner motifs for the random corpus: Mid-bridges, MidNum, RI parity, etc. */
inline std::string_view const utf8_wordbreaks_motifs[] = {
    "don't"sv,                                // WB6/7: apostrophe bridges two letter runs
    "l'avion"sv,                              // WB7a/b: leading apostrophe shape
    "can't_stop"sv,                           // WB13a/b: ExtendNumLet underscore keeps the word whole
    "3,14"sv,                                 // WB11/12: comma between digits stays one word
    "1,2,3"sv,                                // chained numeric MidNum grouping
    "3,"sv,                                   // digit then bare comma: break after the number
    "Hello, world!"sv,                        // punctuation/space boundaries between letter runs
    "\xE4\xBD\xA0\xE5\xA5\xBD"sv,             // CJK: each ideograph is its own word
    "\xEC\x95\x88\xEB\x85\x95"sv,             // Hangul syllables are word chars
    "\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D"sv,     // Hebrew letter run
    "\xD7\x90\x27\xD7\x91"sv,                 // Hebrew aleph + apostrophe + bet (WB7a single-quote)
    "\xE3\x82\xAB\xE3\x82\xBF\xE3\x82\xAB"sv, // Katakana run (WB13 Katakana x Katakana)
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"sv,     // Regional-Indicator pair (WB15/16 parity)
    "\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8"sv, // RI ASCII RI: parity reset between flags
    "word\xC2\xB7word"sv,                     // U+00B7 MIDDLE DOT as MidLetter between letters
};

/** Multi-window seam regressions, each over 64 bytes: WB15/16 Regional_Indicator parity and the
 *  WB6/7/11/12 Mid-bridge carry state, pinned across the 64-byte window boundary. Stored as raw
 *  bytes for the differential driver to feed to serial-vs-ISA directly, with no inline asserts. */
inline std::string_view const utf8_wordbreaks_seam_regressions[] = {
    // ri_after_newline (65 bytes)
    "\xE3\x82\xAB\x2D\x0A\xF0\x9F\x87\xA6\x62\xC2\xAD\xF0\x9F\x87\xA6\xCC\x80\x0A\x5F\xF0\x9F\x87\xA6" //
    "\xF0\x9F\x87\xA6\xC2\xAD\x0A\xCC\x88\xF0\x9F\x87\xBA\xF0\x9F\x8F\xBB\xCC\x88\xC2\xAD\xE2\x81\xA0" //
    "\xCC\x88\xF0\x9F\x87\xA6\xF0\x9F\x87\xA6\xE4\xB8\xAD\xE2\x81\xA0\x5F"sv,
    // ri_word_joiner (82 bytes)
    "\x5F\x27\xF0\x9F\x87\xBA\xE4\xB8\xAD\xCC\x81\xF0\x9F\x87\xA6\xF0\x9F\x87\xA6\x5F\xCC\x88\x5F\xD7" //
    "\x90\x2D\xE2\x93\x82\xCC\x80\x62\xF0\x9F\x87\xA6\xE2\x81\xA0\xF0\x9F\x87\xBA\xCC\x80\xCC\x88\xF0" //
    "\x9F\x8F\xBB\xF0\x9F\x87\xBA\xCC\x81\xF0\x9F\x87\xBA\x0A\xC2\xAD\xF0\x9F\x87\xBA\xCC\x88\xE3\x82" //
    "\xAB\x5F\x27\x2E\xCC\x88\x0A\xE4\xB8\xAD"sv,
    // wb12_bridge (78 bytes)
    "\x5F\xE2\x81\xA0\x5F\x35\x2C\xF0\x9F\x98\x80\x27\xF0\x9F\x98\x80\x5F\xCC\x88\xCC\x81\xE4\xB8\xAD" //
    "\xE2\x81\xA0\x5F\x62\xCC\x80\x61\xE4\xB8\xAD\xE3\x82\xAB\x35\xE2\x81\xA0\xCC\x88\xCC\x88\xF0\x9F" //
    "\x8F\xBB\xCC\x88\xE2\x81\xA0\xCC\x80\xCC\x81\x27\x35\xCC\x80\xE4\xB8\xAD\xCC\x81\xE2\x80\x8D\xC3" //
    "\xA9\xE3\x80\x80\x35\x27"sv,
};

/** @p link_count Katakana codepoints (WB13 Katakana × Katakana), into @p out, cleared first. */
inline void utf8_wordbreaks_dense_katakana_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) out.append(encoded_rune_(0x30AB)); // カ
}

/** Numeric run with MidNum and Extend marks, @p link_count groups (WB11/12 + Extend), to @p out. */
inline void utf8_wordbreaks_dense_numeric_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append("12");
        out.append(encoded_rune_(0x0301)); // Extend combining mark inside a number
        out.append(",34 ");                // MidNum comma
    }
}

/** MidLetter (`'` and U+00B7) between letters, @p link_count dense groups (WB6/7), into @p out. */
inline void utf8_wordbreaks_dense_midletter_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append(encoded_rune_(0x0061));                         // 'a'
        out.append(encoded_rune_((index & 1u) ? 0x00B7 : 0x0027)); // MIDDLE DOT or apostrophe
        out.append(encoded_rune_(0x0062));                         // 'b'
    }
}

/** Hebrew letters bridged by single-quote, @p link_count groups (WB7a Hebrew_Letter MidLetter),
 *  into @p out. */
inline void utf8_wordbreaks_dense_hebrew_quote_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append(encoded_rune_(0x05D0)); // א
        out.append(encoded_rune_(0x0027)); // single quote
        out.append(encoded_rune_(0x05D1)); // ב
    }
}

/** Stream the word family's high-density homogeneous runs (each spans several 64-byte windows) to
 *  @p sink. */
inline void utf8_wordbreaks_dense_runs_(std::mt19937 &generator, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    std::size_t const wide_count = std::uniform_int_distribution<std::size_t>(60, 220)(generator);
    utf8_dense_regional_indicators_(scratch, generator, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_wordbreaks_dense_katakana_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_wordbreaks_dense_numeric_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_wordbreaks_dense_midletter_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_wordbreaks_dense_hebrew_quote_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
}

/** Stream the word family's long-range straddling constructions for a given @p gap to @p sink. */
inline void utf8_wordbreaks_straddles_(std::mt19937 &generator, std::size_t gap, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    utf8_dense_regional_indicators_(scratch, generator, gap);
    scratch.append("a"); // ASCII tail forces the WB15/16 parity decision after the long run
    sink(context, scratch.data(), scratch.size());
    utf8_wordbreaks_dense_midletter_(scratch, gap), sink(context, scratch.data(), scratch.size());
}

/** Word snippets: apostrophe/underscore bridges, numeric groups, CJK/Hangul/Hebrew/Katakana. */
inline char const *const utf8_wordbreaks_snippets[] = {
    "don't ",
    "can't_stop ",
    "3,14 ",
    "1,2,3 ",
    "Hello, world! ",
    "\xE4\xBD\xA0\xE5\xA5\xBD",         // CJK ideographs
    "\xEC\x95\x88\xEB\x85\x95",         // Hangul syllables
    "\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D", // Hebrew run
    "\xE3\x82\xAB\xE3\x82\xBF ",        // Katakana
    "word\xC2\xB7word ",                // MIDDLE DOT MidLetter
    "a b ",
    "_a ",
    "\xCE\xBA\xCE\xB1\xCE\xBB\xCE\xAC ",                 // Greek word carrying a tonos
    "\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0 ", // Cyrillic word
    "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7 ",         // Arabic word
    "\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7ok ",       // Arabic abutting Latin: WB5 across a bidi transition
};

/** Word alphabet, biased toward family snippets and motifs (WB6/7/11/12/13/15/16). */
inline utf8_corpus_alphabet_t const utf8_wordbreaks_alphabet = {
    span_over(utf8_wordbreaks_snippets),
    span_over(utf8_default_boundary_codepoints),
    {{40, 15, 10, 30, 5}}, // snippet, boundary, astral, motif, malformed
};

/** One motif per UAX-29 Word_Break rule, tagged with the direction it demonstrates. Rules with
 *  both senses also carry an opposite-direction motif, since the gate compares serial against
 *  each capability on every motif. */
inline utf8_rule_case_t const utf8_wordbreaks_rule_cases[] = {
    {"WB3", utf8_rule_joins_k, "\r\n"sv},                                  // CR x LF (no break)
    {"WB3a", utf8_rule_breaks_k, "\rb"sv},                                 // (Newline|CR|LF) / break after CR
    {"WB3b", utf8_rule_breaks_k, "a\n"sv},                                 // / (Newline|CR|LF) break before LF
    {"WB3c", utf8_rule_joins_k, "\xE2\x80\x8D\xF0\x9F\x98\x80"sv},         // ZWJ x Extended_Pictographic (no break)
    {"WB3d", utf8_rule_joins_k, "  "sv},                                   // WSegSpace x WSegSpace (no break)
    {"WB4", utf8_rule_joins_k, "a\xCC\x81"sv},                             // X (Extend|Format|ZWJ)* absorbed
    {"WB5", utf8_rule_joins_k, "ab"sv},                                    // AHLetter x AHLetter
    {"WB6", utf8_rule_joins_k, "a'b"sv},                                   // AHLetter x (MidLetter) AHLetter
    {"WB7", utf8_rule_joins_k, "a'b"sv},                                   // AHLetter (MidLetter) x AHLetter
    {"WB7a", utf8_rule_joins_k, "\xD7\x90'"sv},                            // Hebrew_Letter x Single_Quote
    {"WB7b", utf8_rule_joins_k, "\xD7\x90\"\xD7\x90"sv},                   // Hebrew_Letter x Double_Quote Hebrew_Letter
    {"WB7c", utf8_rule_joins_k, "\xD7\x90\"\xD7\x90"sv},                   // Hebrew_Letter Double_Quote x Hebrew_Letter
    {"WB8", utf8_rule_joins_k, "12"sv},                                    // Numeric x Numeric
    {"WB9", utf8_rule_joins_k, "a1"sv},                                    // AHLetter x Numeric
    {"WB10", utf8_rule_joins_k, "1a"sv},                                   // Numeric x AHLetter
    {"WB11", utf8_rule_joins_k, "1,2"sv},                                  // Numeric (MidNum) x Numeric
    {"WB12", utf8_rule_joins_k, "1,2"sv},                                  // Numeric x (MidNum) Numeric
    {"WB13", utf8_rule_joins_k, "\xE3\x82\xA2\xE3\x82\xA2"sv},             // Katakana x Katakana
    {"WB13a", utf8_rule_joins_k, "a_"sv},                                  // (AHLetter|Numeric|Katakana) x ExtendNumLet
    {"WB13b", utf8_rule_joins_k, "_a"sv},                                  // ExtendNumLet x (AHLetter|Numeric|Katakana)
    {"WB15", utf8_rule_joins_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"sv},     // sot (RI RI)* RI x RI (even parity)
    {"WB16", utf8_rule_joins_k, "\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8"sv}, // [^RI] (RI RI)* RI x RI (parity reset)
    {"WB999", utf8_rule_breaks_k, "a b"sv},                                // Any / Any (default break at the space)
    // Opposite-direction motifs (E1): the same rule firing the other way.
    {"WB6", utf8_rule_breaks_k, "a''b"sv},           // two MidLetters: the bridge fails → break
    {"WB6", utf8_rule_breaks_k, "a\"b"sv},           // Double_Quote is not MidNumLetQ for Latin: break twice
    {"WB11", utf8_rule_breaks_k, "1\"2"sv},          // Double_Quote never bridges numerics either
    {"WB11", utf8_rule_breaks_k, "1,,2"sv},          // two MidNum: numeric bridge fails → break
    {"WB13", utf8_rule_breaks_k, "\xE3\x82\xA2z"sv}, // Katakana then Latin: script change → break
    {"WB15", utf8_rule_breaks_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA"sv}, // 3 RI: break after the pair
};

/** Every Word_Break rule id the gate requires a motif for (spec-derived checklist). */
inline char const *const utf8_wordbreaks_required_rules[] = {
    "WB3", "WB3a", "WB3b", "WB3c", "WB3d", "WB4",  "WB5",   "WB6",   "WB7",  "WB7a", "WB7b",  "WB7c",
    "WB8", "WB9",  "WB10", "WB11", "WB12", "WB13", "WB13a", "WB13b", "WB15", "WB16", "WB999",
};

/** Known-answer word-break vectors and the deferred-mid goldens for one segmenter. */
inline void check_utf8_wordbreaks_unit_(utf8_segment_backend_t const &backend) {
    check_utf8_segment_unit_("word", backend.finder, span_over(utf8_wordbreaks_unit_cases));
    check_utf8_wordbreaks_deferred_mid_(backend);
}

/** Rule-coverage gate: every WB motif runs, and @p backend agrees with serial at window phases. */
inline void check_utf8_wordbreaks_rules_(utf8_segment_backend_t const &backend) {
    check_utf8_rule_coverage_("word", sz_utf8_wordbreaks_serial, backend.finder, span_over(utf8_wordbreaks_rule_cases),
                              span_over(utf8_wordbreaks_required_rules));
}

/** Malformed-input safety of one word segmenter. */
inline void check_utf8_wordbreaks_safety_(test_context_t &context, utf8_segment_backend_t const &backend) {
    utf8_segment_backend_t const backends[] = {backend};
    check_utf8_segment_safety_(context, "word", span_over(backends));
}

/** Word differential of @p backend against serial over the dense, long-range and seam corpora. */
inline void check_utf8_wordbreaks_equivalence_(test_context_t &context, utf8_segment_backend_t const &backend) {
    utf8_segment_corpora_t const corpora = {"word",
                                            span_over(utf8_wordbreaks_motifs),
                                            &utf8_wordbreaks_dense_runs_,
                                            &utf8_wordbreaks_straddles_,
                                            span_over(utf8_wordbreaks_seam_regressions),
                                            &utf8_wordbreaks_alphabet};
    utf8_segment_backend_t const candidates[] = {backend};
    // The iteration count is this family's share of the suite budget, sized against its siblings.
    check_utf8_segment_equivalence_(context, sz_utf8_wordbreaks_serial, span_over(candidates), corpora,
                                    context.iterations(20));
}

/** The serial word segmenter against its per-position WB1-WB16 rules, called nowhere else. */
inline void check_utf8_wordbreaks_oracle_() {
    for (std::string_view const motif : span_over(utf8_wordbreaks_motifs))
        check_utf8_segment_against_oracle_("word", sz_utf8_wordbreaks_serial, sz_utf8_is_word_boundary_serial,
                                           motif.data(), motif.size());
}

#pragma endregion UTF8 Wordbreaks

#pragma region UTF8 Graphemes

/** Hand-checked UAX-29 grapheme-cluster golden vectors: each source text and its clusters. */
inline utf8_unit_case_t const utf8_graphemes_unit_cases[] = {
    {""sv, {}},
    {"a"sv, {"a"sv}},
    {"abc"sv, {"a"sv, "b"sv, "c"sv}},
    {"a\r\nb"sv, {"a"sv, "\r\n"sv, "b"sv}}, // GB3: CR x LF stay one cluster
    {"e\xCC\x81"sv, {"e\xCC\x81"sv}},       // e + U+0301 combining acute → one cluster (GB9)
    {"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"sv, {"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"sv}}, // RI pair → one cluster (GB12/13)
    {"\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8"sv,
     {"\xF0\x9F\x87\xBA"sv, "a"sv, "\xF0\x9F\x87\xB8"sv}}, // RI ASCII RI: parity reset
    {"\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9"sv,     // ZWJ joins → one cluster (GB11)
     {"\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9"sv}},
    {"\xEA\xB0\x80"sv, {"\xEA\xB0\x80"sv}}, // Hangul LV syllable → one cluster
    {"\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"sv,
     {"\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"sv}}, // Hangul L+V+T jamo → one cluster (GB6/7/8)
    {"\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7"sv,
     {"\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7"sv}}, // Indic consonant + virama + consonant → one cluster (GB9c InCB)
};

/** UAX-29 grapheme-cluster corner motifs (sprinkled into the random corpus): emoji-ZWJ chains,
 *  VS16, skin-tone modifiers, regional-indicator runs of varying parity, Indic virama clusters,
 *  Hangul jamo combinations, and bare CR/LF shapes. All non-ASCII bytes are `\xHH` escapes. */
inline std::string_view const utf8_graphemes_motifs[] = {
    "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D" // three-link ZWJ chain
    "\xF0\x9F\x91\xA7"sv,
    "\xF0\x9F\x91\xA9\xE2\x80\x8D"sv,     // dangling trailing ZWJ
    "\xE2\x9C\x8B\xEF\xB8\x8F"sv,         // emoji + VS16 (U+FE0F)
    "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD"sv, // emoji + skin-tone modifier (U+1F3FD)
    "\xF0\x9F\x87\xBA"sv,                 // single regional indicator (run length 1)
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"sv, // regional-indicator run length 2
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"    // regional-indicator run length 3 (odd parity)
    "\xF0\x9F\x87\xAB"sv,
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8" // regional-indicator run length 5 (odd parity)
    "\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7\xF0\x9F\x87\xA9"sv,
    "\xE0\xA6\x95\xE0\xA7\x8D\xE0\xA6\x95"sv, // Bengali consonant + virama + consonant (U+09CD)
    "\xE0\xB4\x95\xE0\xB5\x8D\xE0\xB4\x95"sv, // Malayalam consonant + virama + consonant (U+0D4D)
    "\xE0\xAE\x95\xE0\xAF\x8D\xE0\xAE\x95"sv, // Tamil consonant + virama + consonant (U+0BCD)
    "\xEA\xB0\x80"sv,                         // Hangul LV syllable
    "\xEC\x95\x8A"sv,                         // Hangul LVT syllable
    "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"sv, // Hangul L+V+T jamo combination
    "\r"sv,                                   // CR alone
    "\n"sv,                                   // LF alone
    "\r\r\n"sv,                               // CR CR LF (parity of CR runs)
    "a\xDF\xBF\xE0\xA0\x80z"sv,               // ASCII + U+07FF (2-byte) + U+0800 (3-byte cold) + ASCII: every BMP gate
    "x\xF0\x9F\x98\x80y"sv,                   // ASCII + astral emoji (U+1F600) + ASCII: ASCII and astral gates together
    "\xD0\x90\xF0\x9F\x98\x80"sv,             // Cyrillic (2-byte) + astral emoji: page LUT and astral, no cold cascade
    "\xE0\xA0\x80\xF0\x9F\x98\x80"sv,         // U+0800 (cold 3-byte) + astral emoji: cold cascade and astral together
    "\xDF\xBF\xE0\xA0\x80"sv,                 // page-LUT edge: last 2-byte U+07FF abutting first 3-byte U+0800
};

/** A pictograph ZWJ chain @p link_count deep, every other link followed by VS16 (GB11), into
 *  @p out, cleared first. */
inline void utf8_graphemes_dense_zwj_pictograph_chain_(std::string &out, std::size_t link_count) {
    out.clear();
    static sz_rune_t const pictographs[] = {0x1F468, 0x1F469, 0x1F467, 0x1F466}; // man, woman, girl, boy
    out.append(encoded_rune_(pictographs[0]));
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append(encoded_rune_(0x200D));                          // ZWJ
        out.append(encoded_rune_(pictographs[(index + 1) & 0x3u])); // next pictograph
        if (index & 1u) out.append(encoded_rune_(0xFE0F));          // VS16 on alternating links
    }
}

/** A base letter and @p link_count combining marks (GB9 Extend run), into @p out, cleared first. */
inline void utf8_graphemes_dense_combining_marks_(std::string &out, std::size_t link_count) {
    out.clear();
    static sz_rune_t const marks[] = {0x0301, 0x0300, 0x0308, 0x0327, 0x0323, // acute, grave, diaeresis, cedilla, dot
                                      0x0651, 0x093C, 0x0E48, 0x1D16E};       // shadda, nukta, Thai mai ek, astral flag
    out.append(encoded_rune_(0x0061));                                        // base 'a'
    for (std::size_t index = 0; index != link_count; ++index) out.append(encoded_rune_(marks[index % 9u]));
}

/** @p link_count emoji with skin-tone modifiers from @p generator (GB9 Extend), into @p out. */
inline void utf8_graphemes_dense_skin_tone_run_(std::string &out, std::mt19937 &generator, std::size_t link_count) {
    out.clear();
    std::uniform_int_distribution<sz_rune_t> modifier(0x1F3FB, 0x1F3FF); // U+1F3FB..U+1F3FF
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append(encoded_rune_(0x1F44D)); // thumbs up
        out.append(encoded_rune_(modifier(generator)));
    }
}

/** @p link_count Indic consonant+virama links (GB9c InCB Consonant Linker chains), into @p out. */
inline void utf8_graphemes_dense_indic_conjunct_(std::string &out, std::size_t link_count) {
    out.clear();
    out.append(encoded_rune_(0x0915)); // Devanagari KA
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append(encoded_rune_(0x094D)); // virama (Linker)
        out.append(encoded_rune_(0x0915)); // KA
    }
}

/** @p link_count Hangul L/V/T jamo triples (GB6/7/8 runs), into @p out, cleared first. */
inline void utf8_graphemes_dense_hangul_jamo_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append(encoded_rune_(0x1100)); // L (Choseong Kiyeok)
        out.append(encoded_rune_(0x1161)); // V (Jungseong A)
        out.append(encoded_rune_(0x11A8)); // T (Jongseong Kiyeok)
    }
}

/** Stream the grapheme family's high-density homogeneous runs, each spanning several 64-byte
 *  windows, to @p sink. */
inline void utf8_graphemes_dense_runs_(std::mt19937 &generator, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    std::uniform_int_distribution<std::size_t> wide(60, 220);
    std::uniform_int_distribution<std::size_t> chain(20, 80);
    std::size_t const wide_count = wide(generator);
    std::size_t const chain_count = chain(generator);
    utf8_dense_regional_indicators_(scratch, generator, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_zwj_pictograph_chain_(scratch, chain_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_combining_marks_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_skin_tone_run_(scratch, generator, chain_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_indic_conjunct_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_hangul_jamo_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
}

/** Stream the grapheme family's long-range straddling constructions for @p gap to @p sink. */
inline void utf8_graphemes_straddles_(std::mt19937 &generator, std::size_t gap, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    utf8_dense_regional_indicators_(scratch, generator, gap);
    scratch.append("a"); // ASCII tail forces the GB12/13 parity decision after the long run
    sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_zwj_pictograph_chain_(scratch, gap);
    scratch.append(encoded_rune_(0x0061)); // ASCII break after the chain
    sink(context, scratch.data(), scratch.size());
    utf8_graphemes_dense_indic_conjunct_(scratch, gap);
    scratch.append(encoded_rune_(0x0061)); // ASCII break after the conjunct
    sink(context, scratch.data(), scratch.size());
    // A long RI run, then a ZWJ before a final RI: RI...RI ZWJ RI. The ZWJ does not bridge two RIs
    // (GB11 bridges only Extended_Pictographic), so this must break after the ZWJ - and the ZWJ
    // resets the GB12/13 RI parity. Straddles the 64-byte window so the parity carry and the
    // post-ZWJ break are exercised across the edge.
    utf8_dense_regional_indicators_(scratch, generator, gap);
    scratch.append(encoded_rune_(0x200D));  // ZWJ
    scratch.append(encoded_rune_(0x1F1E6)); // Regional_Indicator after the ZWJ
    scratch.append(encoded_rune_(0x0061));  // ASCII tail
    sink(context, scratch.data(), scratch.size());
}

/** Grapheme-biased snippets: ZWJ sequences, jamo, conjuncts, Arabic/Greek/Cyrillic/Thai marks. */
inline char const *const utf8_graphemes_snippets[] = {
    "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9",     // woman ZWJ woman (GB11 emoji-ZWJ sequence)
    "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD",                 // thumbs-up + skin-tone modifier (GB9 Extend)
    "e\xCC\x81",                                        // base 'e' + combining acute (GB9)
    "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8",             // Hangul L+V+T jamo (GB6/7/8)
    "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7",             // Devanagari consonant + virama + consonant (GB9c)
    "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8",                 // regional-indicator pair (GB12/13)
    "\r\n",                                             // CRLF (GB3)
    "abc",                                              // plain ASCII (GB999)
    "\xD8\xA8\xD9\x91\xD9\x8E",                         // Arabic beh + shadda + fatha (stacked Extend marks)
    "\xCE\xB1\xCC\x94\xCC\x81",                         // Greek alpha + rough breathing + acute
    "\xD0\xB8\xCC\x86",                                 // Cyrillic i + combining breve
    "\xE0\xB8\x81\xE0\xB8\xB4\xE0\xB9\x88",             // Thai ko kai + sara i + mai ek
    "\xF0\x9D\x85\x97\xF0\x9D\x85\xA5\xF0\x9D\x85\xAE", // astral notehead + stem (SpacingMark) + flag (Extend)
};

/** Grapheme family alphabet: weights bias toward astral/motif clusters (GB9/9c/11/12/13). */
inline utf8_corpus_alphabet_t const utf8_graphemes_alphabet = {
    span_over(utf8_graphemes_snippets),
    span_over(utf8_default_boundary_codepoints),
    {{35, 15, 25, 20, 5}}, // snippet, boundary, astral, motif, malformed
};

/** One motif per UAX-29 Grapheme_Cluster_Break rule (break or no-break direction). */
inline utf8_rule_case_t const utf8_graphemes_rule_cases[] = {
    {"GB3", utf8_rule_joins_k, "\r\n"sv},                                          // CR x LF (no break)
    {"GB4", utf8_rule_breaks_k, "\na"sv},                                          // (Control|CR|LF) / break after LF
    {"GB5", utf8_rule_breaks_k, "a\n"sv},                                          // / (Control|CR|LF) break before LF
    {"GB6", utf8_rule_joins_k, "\xE1\x84\x80\xE1\x85\xA1"sv},                      // L x (L|V|LV|LVT): Hangul L + V
    {"GB7", utf8_rule_joins_k, "\xEA\xB0\x80\xE1\x85\xA1"sv},                      // (LV|V) x (V|T): Hangul LV + V
    {"GB8", utf8_rule_joins_k, "\xEA\xB0\x81\xE1\x86\xA8"sv},                      // (LVT|T) x T: Hangul LVT + T
    {"GB9", utf8_rule_joins_k, "a\xCC\x81"sv},                                     // x (Extend|ZWJ)
    {"GB9a", utf8_rule_joins_k, "\xE0\xA4\x95\xE0\xA4\xBE"sv},                     // x SpacingMark: consonant + vowel
    {"GB9b", utf8_rule_joins_k, "\xD8\x80" "a"sv},                                 // Prepend x: U+0600 then letter
    {"GB9c", utf8_rule_joins_k, "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\x95"sv},         // Indic conjunct: cons virama cons
    {"GB11", utf8_rule_joins_k, "\xF0\x9F\x98\x80\xE2\x80\x8D\xF0\x9F\x98\x81"sv}, // ExtPict Extend* ZWJ x ExtPict
    {"GB12", utf8_rule_joins_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"sv},             // sot (RI RI)* RI x RI
    {"GB13", utf8_rule_joins_k, "a\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"sv},            // [^RI] (RI RI)* RI x RI
    {"GB999", utf8_rule_breaks_k, "ab"sv},                                         // Any / Any (default break)
    // Opposite-direction motifs (E1): the same rule firing the other way.
    {"GB11", utf8_rule_breaks_k, "\xF0\x9F\x98\x80z"sv}, // ExtPict then ASCII: no ZWJ bridge → break
    {"GB12", utf8_rule_breaks_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA"sv}, // 3 RI: break after the pair
    {"GB6", utf8_rule_breaks_k, "\xE1\x84\x80z"sv}, // Hangul L then ASCII: not L|V|LV|LVT → break
};

/** Every Grapheme_Cluster_Break rule id the gate requires a motif for (spec-derived checklist). */
inline char const *const utf8_graphemes_required_rules[] = {
    "GB3", "GB4", "GB5", "GB6", "GB7", "GB8", "GB9", "GB9a", "GB9b", "GB9c", "GB11", "GB12", "GB13", "GB999",
};

/** Known-answer grapheme-cluster vectors for one segmenter. */
inline void check_utf8_graphemes_unit_(utf8_segment_backend_t const &backend) {
    check_utf8_segment_unit_("grapheme", backend.finder, span_over(utf8_graphemes_unit_cases));
}

/** Rule-coverage gate: every GB motif runs, and @p backend agrees with serial at window phases. */
inline void check_utf8_graphemes_rules_(utf8_segment_backend_t const &backend) {
    check_utf8_rule_coverage_("grapheme", sz_utf8_graphemes_serial, backend.finder,
                              span_over(utf8_graphemes_rule_cases), span_over(utf8_graphemes_required_rules));
}

/** Malformed-input safety of one grapheme segmenter. */
inline void check_utf8_graphemes_safety_(test_context_t &context, utf8_segment_backend_t const &backend) {
    utf8_segment_backend_t const backends[] = {backend};
    check_utf8_segment_safety_(context, "grapheme", span_over(backends));
}

/** The grapheme differential of @p backend against serial over the dense and long-range corpora. */
inline void check_utf8_graphemes_equivalence_(test_context_t &context, utf8_segment_backend_t const &backend) {
    utf8_segment_corpora_t const corpora = {
        "grapheme",
        span_over(utf8_graphemes_motifs),
        &utf8_graphemes_dense_runs_,
        &utf8_graphemes_straddles_,
        {}, // no fixed seam regressions for graphemes
        &utf8_graphemes_alphabet,
    };
    utf8_segment_backend_t const candidates[] = {backend};
    check_utf8_segment_equivalence_(context, sz_utf8_graphemes_serial, span_over(candidates), corpora,
                                    context.iterations(8)); // This family's share of the suite budget
}

/** The serial grapheme segmenter against its per-position GB1-GB999 rules, called nowhere else. */
inline void check_utf8_graphemes_oracle_() {
    for (std::string_view const motif : span_over(utf8_graphemes_motifs))
        check_utf8_segment_against_oracle_("grapheme", sz_utf8_graphemes_serial, sz_utf8_is_grapheme_boundary_serial,
                                           motif.data(), motif.size());
}

#pragma endregion UTF8 Graphemes

#pragma region UTF8 Sentences

/** Hand-checked UAX-29 sentence-break golden vectors: each source text and its sentences. */
inline utf8_unit_case_t const utf8_sentences_unit_cases[] = {
    {""sv, {}},
    {"Hello."sv, {"Hello."sv}},
    {"Hello. World."sv, {"Hello. "sv, "World."sv}},     // terminator + space ends the sentence
    {"Mr. Smith went."sv, {"Mr. "sv, "Smith went."sv}}, // UAX-29 has no abbreviation list: '. ' before uppercase breaks
    {"etc. and so on"sv, {"etc. and so on"sv}},         // SB8: '. ' before a lowercase word does not break
    {"One.\nTwo."sv, {"One.\n"sv, "Two."sv}},           // ParaSep after terminator
    {"A! B? C."sv, {"A! "sv, "B? "sv, "C."sv}},         // multiple terminators
};

/** UAX-29 sentence-break corner motifs (sprinkled into the random corpus): ATerm vs STerm,
 *  terminator + Close + Space + case, paragraph separators after a terminator, SContinue,
 *  abbreviation-like and numeric. */
inline std::string_view const utf8_sentences_motifs[] = {
    "End. Next"sv,               // ATerm + space + uppercase (SB break)
    "End! Next"sv,               // STerm + space + uppercase (SB break)
    "End? Next"sv,               // STerm '?' + space + uppercase
    "end. next"sv,               // SB8: '. ' before lowercase does not break
    "(End.) Next"sv,             // terminator + Close ')' + space + uppercase
    "\"End.\" Next"sv,           // terminator + Close '"' + space + uppercase
    "End.\xE2\x80\xA8" "Next"sv, // ParaSep U+2028 after terminator
    "End.\xE2\x80\xA9" "Next"sv, // ParaSep U+2029 after terminator
    "End.\xC2\x85" "Next"sv,     // ParaSep U+0085 NEL after terminator
    "Item, more"sv,              // SContinue ',' continuation
    "Item: more"sv,              // SContinue ':' continuation
    "Mr. Smith"sv,               // abbreviation-like
    "Dr. House"sv,               // abbreviation-like
    "U.S.A. ok"sv,               // dotted acronym
    "3.14 value"sv,              // numeric ATerm (no break)
    // Dense-decode edge motifs for SB5 Extend/Format transparency and cross-window SB8, sprinkled
    // at window seams:
    "End.\xCC\x88 next"sv,      // ATerm + combining mark (Extend, SB5-transparent) before the space
    "End\xCC\x88. Next"sv,      // combining mark on the terminator's preceding letter
    "End.\xE2\x80\x8B Next"sv,  // ATerm + Format (U+200B ZWSP is Format here) transparency
    "A.\xCC\x80\xCC\x80 b"sv,   // ATerm + stacked Extend marks then lowercase (SB8 across transparents)
    "End. \xE2\x80\x8C Next"sv, // ATerm Sp then Format then Upper (SB8 neutral chain)
};

/** ATerm + Close* + Sp* run @p link_count wide, then a Lower (SB8: no break), into @p out. */
inline void utf8_sentences_dense_aterm_sp_lower_(std::string &out, std::size_t link_count) {
    out.clear();
    out.append(encoded_rune_(0x0055));                                                           // 'U' Upper
    out.append(encoded_rune_(0x002E));                                                           // '.' ATerm
    for (std::size_t index = 0; index != link_count; ++index) out.append(encoded_rune_(0x0020)); // Sp run
    out.append(encoded_rune_(0x0061));                                                           // 'a' Lower
}

/** Terminator-dense `A. A. A. ...` repeated @p link_count times, one break each, into @p out. */
inline void utf8_sentences_dense_terminators_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) out.append("A. ");
}

/** Ideographic full stop (U+3002) + Sp run + CJK, repeated @p link_count times (multibyte STerm),
 *  into @p out. */
inline void utf8_sentences_dense_cjk_term_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append(encoded_rune_(0x4E2D)); // 中
        out.append(encoded_rune_(0x3002)); // 。 ideographic full stop (STerm)
        out.append(encoded_rune_(0x0020)); // Sp
    }
}

/** Stream the sentence family's high-density homogeneous runs (each spans several 64-byte windows)
 *  to @p sink. */
inline void utf8_sentences_dense_runs_(std::mt19937 &generator, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    std::size_t const wide_count = std::uniform_int_distribution<std::size_t>(60, 220)(generator);
    utf8_sentences_dense_aterm_sp_lower_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_sentences_dense_terminators_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_sentences_dense_cjk_term_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
}

/** SB8: `Upper ATerm Sp{gap} Lower` — long Sp run before a lowercase keeps one sentence,
 *  into @p out. */
inline void utf8_sentences_straddle_sb8_lower_(std::string &out, std::size_t gap) {
    out.clear();
    out.append(encoded_rune_(0x0055));                                                    // 'U'
    out.append(encoded_rune_(0x002E));                                                    // '.'
    for (std::size_t index = 0; index != gap; ++index) out.append(encoded_rune_(0x0020)); // Sp run across windows
    out.append(encoded_rune_(0x0061));                                                    // 'a' Lower
}

/** SB11: `Upper ATerm Sp{gap} Upper` — same run before an uppercase must break, into @p out. */
inline void utf8_sentences_straddle_sb11_upper_(std::string &out, std::size_t gap) {
    out.clear();
    out.append(encoded_rune_(0x0055));                                                    // 'U'
    out.append(encoded_rune_(0x002E));                                                    // '.'
    for (std::size_t index = 0; index != gap; ++index) out.append(encoded_rune_(0x0020)); // Sp run across windows
    out.append(encoded_rune_(0x0042));                                                    // 'B' Upper
}

/** Stream the sentence family's long-range straddling constructions for @p gap to @p sink. */
inline void utf8_sentences_straddles_(std::mt19937 & /*generator*/, std::size_t gap, utf8_run_sink_t sink,
                                      void *context) {
    std::string scratch;
    utf8_sentences_straddle_sb8_lower_(scratch, gap), sink(context, scratch.data(), scratch.size());
    utf8_sentences_straddle_sb11_upper_(scratch, gap), sink(context, scratch.data(), scratch.size());
}

/** Sentence snippets: ATerm/STerm + space + case across scripts, numeric, CJK stop, ParaSep. */
inline char const *const utf8_sentences_snippets[] = {
    "End. Next ",               // ATerm + space + Upper (SB break)
    "end. next ",               // SB8: '. ' before Lower does not break
    "3.14 ",                    // numeric ATerm (no break)
    "Item, ",                   // SContinue ',' continuation
    "\xE4\xB8\xAD\xE3\x80\x82", // CJK ideograph + ideographic full stop (STerm)
    "\xE2\x80\xA8",             // ParaSep U+2028
    "\xCE\x9A\xCE\xB1\xCE\xBB\xCE\xAC\xCD\xBE \xCE\x9D\xCE\xB1\xCE\xB9 ", // Greek question mark U+037E + Upper
    "\xD9\x85\xD8\xB1\xDB\x94 \xD9\x85\xD8\xB1\xD8\x9F Yes. ",            // Arabic U+06D4 / U+061F, RTL then LTR
    "\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA5\xA4 \xD0\x94\xD0\xB0. ",            // Devanagari danda U+0964 + Cyrillic
};

/** Sentence alphabet, biased toward family snippets and motifs (SB6/7/8/8a/9/10/11). */
inline utf8_corpus_alphabet_t const utf8_sentences_alphabet = {
    span_over(utf8_sentences_snippets),
    span_over(utf8_default_boundary_codepoints),
    {{45, 15, 5, 30, 5}}, // snippet, boundary, astral, motif, malformed
};

/** One motif per UAX-29 Sentence_Break rule, tagged with the break or no-break it demonstrates. */
inline utf8_rule_case_t const utf8_sentences_rule_cases[] = {
    {"SB3", utf8_rule_joins_k, "\r\n"sv},      // CR x LF (no break)
    {"SB4", utf8_rule_breaks_k, "a\nb"sv},     // (Sep|CR|LF) / break after LF
    {"SB5", utf8_rule_joins_k, "a\xCC\x81"sv}, // X (Extend|Format)* absorbed (no break)
    {"SB6", utf8_rule_joins_k, "3.4"sv},       // ATerm x Numeric (no break)
    {"SB7", utf8_rule_joins_k, "A.B"sv},       // (Upper|Lower) ATerm x Upper (no break)
    {"SB8", utf8_rule_joins_k, "A. a"sv},      // ATerm Close* Sp* x (not stop)* Lower (no break)
    {"SB8a", utf8_rule_joins_k, "a.,"sv},      // (STerm|ATerm) Close* Sp* x (SContinue|STerm|ATerm) (no break)
    {"SB9", utf8_rule_joins_k, "a.)"sv},       // (STerm|ATerm) Close* x (Close|Sp|Sep|CR|LF) (no break)
    {"SB10", utf8_rule_joins_k, "a. "sv},      // (STerm|ATerm) Close* Sp* x (Sp|Sep|CR|LF) (no break)
    {"SB11", utf8_rule_breaks_k, "A. B"sv},    // (STerm|ATerm) Close* Sp* / break before the next sentence
    {"SB998", utf8_rule_joins_k, "ab"sv},      // Any x Any (no break by default)
    // Opposite-direction motifs (E1): the same rule firing the other way.
    {"SB998", utf8_rule_breaks_k, "ab cd"sv}, // default no-break interior, but the terminator-less run still varies
    {"SB5", utf8_rule_joins_k, "a\xE2\x80\x8B"sv}, // Format (U+200B ZWSP) transparency absorbed (no break)
};

/** Every Sentence_Break rule id the gate requires a motif for (spec-derived checklist). */
inline char const *const utf8_sentences_required_rules[] = {
    "SB3", "SB4", "SB5", "SB6", "SB7", "SB8", "SB8a", "SB9", "SB10", "SB11", "SB998",
};

/** Known-answer sentence-break vectors for one segmenter. */
inline void check_utf8_sentences_unit_(utf8_segment_backend_t const &backend) {
    check_utf8_segment_unit_("sentence", backend.finder, span_over(utf8_sentences_unit_cases));
}

/** Rule-coverage gate: every SB motif runs, and @p backend agrees with serial at window phases. */
inline void check_utf8_sentences_rules_(utf8_segment_backend_t const &backend) {
    check_utf8_rule_coverage_("sentence", sz_utf8_sentences_serial, backend.finder,
                              span_over(utf8_sentences_rule_cases), span_over(utf8_sentences_required_rules));
}

/** Malformed-input safety of one sentence segmenter. */
inline void check_utf8_sentences_safety_(test_context_t &context, utf8_segment_backend_t const &backend) {
    utf8_segment_backend_t const backends[] = {backend};
    check_utf8_segment_safety_(context, "sentence", span_over(backends));
}

/** The sentence differential of @p backend against serial over the dense and long-range corpora. */
inline void check_utf8_sentences_equivalence_(test_context_t &context, utf8_segment_backend_t const &backend) {
    utf8_segment_corpora_t const corpora = {
        "sentence", span_over(utf8_sentences_motifs), &utf8_sentences_dense_runs_, &utf8_sentences_straddles_,
        {},         &utf8_sentences_alphabet};
    utf8_segment_backend_t const candidates[] = {backend};
    check_utf8_segment_equivalence_(context, sz_utf8_sentences_serial, span_over(candidates), corpora,
                                    context.iterations(90)); // This family's share of the suite budget
}

#pragma endregion UTF8 Sentences

#pragma region UTF8 Linebreaks

/** Hand-checked UAX-14 line-break golden vectors: each source text and its expected lines. */
inline utf8_unit_case_t const utf8_linebreaks_unit_cases[] = {
    {""sv, {}},
    {"hello world"sv, {"hello "sv, "world"sv}},          // soft wrap after the space
    {"a\nb"sv, {"a\n"sv, "b"sv}},                        // LF hard break
    {"a\r\nb"sv, {"a\r\n"sv, "b"sv}},                    // CRLF hard break, one segment
    {"a\xE2\x80\xA8" "b"sv, {"a\xE2\x80\xA8"sv, "b"sv}}, // U+2028 LINE SEPARATOR
    {"a\xE2\x80\xA9" "b"sv, {"a\xE2\x80\xA9"sv, "b"sv}}, // U+2029 PARAGRAPH SEPARATOR
};

/** UAX-14 line-break corner motifs for the random corpus: mandatory breaks, OP/CL, HY, GL, NU. */
inline std::string_view const utf8_linebreaks_motifs[] = {
    "a\nb"sv,              // LF mandatory break
    "a\rb"sv,              // CR mandatory break
    "a\r\nb"sv,            // CRLF mandatory break (one segment)
    "a\xE2\x80\xA8" "b"sv, // U+2028 LINE SEPARATOR mandatory
    "a\xE2\x80\xA9" "b"sv, // U+2029 PARAGRAPH SEPARATOR mandatory
    "a\xC2\x85" "b"sv,     // U+0085 NEL mandatory
    "a\xE2\x80\x8D" "b"sv, // ZWJ (U+200D) adjacency
    "(word"sv,             // OP open-punctuation before
    "word)"sv,             // CL close-punctuation after
    "co-op"sv,             // HY hyphen
    "a\xC2\xA0" "b"sv,     // GL glue NBSP (U+00A0)
    "a b"sv,               // BA break-after space
    "word!"sv,             // EX exclamation
    "\"word\""sv,          // QU quotation
    "a:b"sv,               // IS infix separator
    "12345"sv,             // NU numbers
    "$50"sv,               // PR prefix
    "3,000"sv,             // numeric grouping comma
    // Regression motifs for fuzzer-found icelake-vs-serial divergences, sprinkled at window-edge
    // offsets so the cross-window carry is exercised:
    "\xD7\x90-a"sv,                   // LB21a: HL (U+05D0) HY x AL -- no break before the AL, incl. when HL carries
    "\xD7\x90\xE2\x80\x90" "a"sv,     // LB21a with HH (U+2010) instead of HY
    "\xE3\x80\xAF\xE2\x80\x98" "a"sv, // LB10/LB19: lone CM (U+302F, East-Asian) then Pi quote (U+2018) -- side bits cleared
    "\xCC\x88\xE2\x80\x9C"sv,         // lone combining diaeresis (U+0308) then QU (U+201C)
};

/** Mandatory-break-dense line run cycling CRLF/U+2028/U+2029/U+000B, @p link_count cycles (LB4/5),
 *  into @p out. */
inline void utf8_linebreaks_dense_mandatory_breaks_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) {
        out.append(encoded_rune_(0x0061)); // 'a'
        switch (index & 0x3u) {
        case 0: out.append("\r\n"); break;                 // CRLF
        case 1: out.append(encoded_rune_(0x2028)); break;  // LINE SEPARATOR
        case 2: out.append(encoded_rune_(0x2029)); break;  // PARAGRAPH SEPARATOR
        default: out.append(encoded_rune_(0x000B)); break; // vertical tab (BK)
        }
    }
}

/** OP/CL/QU/HY/BA/GL nesting cycled @p link_count times (LB13/14/15/18 adjacency), into @p out. */
inline void utf8_linebreaks_dense_nesting_(std::string &out, std::size_t link_count) {
    out.clear();
    static char const *const cycle[] = {"(", "word", ")", "\"", "-", " ", "\xC2\xA0"}; // OP CL QU HY BA SP GL(NBSP)
    for (std::size_t index = 0; index != link_count; ++index) out.append(cycle[index % 7u]);
}

/** Numeric @c NU runs interleaved with IS (`.`) and SY (`/`), @p link_count groups (LB25 numbers),
 *  into @p out. */
inline void utf8_linebreaks_dense_numeric_(std::string &out, std::size_t link_count) {
    out.clear();
    for (std::size_t index = 0; index != link_count; ++index) out.append("1.234/56 ");
}

/** Stream the linewrap family's high-density homogeneous runs (each spans several 64-byte windows)
 *  to @p sink. */
inline void utf8_linebreaks_dense_runs_(std::mt19937 &generator, utf8_run_sink_t sink, void *context) {
    std::string scratch;
    std::size_t const wide_count = std::uniform_int_distribution<std::size_t>(60, 220)(generator);
    utf8_linebreaks_dense_mandatory_breaks_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_linebreaks_dense_nesting_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
    utf8_linebreaks_dense_numeric_(scratch, wide_count), sink(context, scratch.data(), scratch.size());
}

/** Stream the linewrap family's long-range straddling constructions for @p gap to @p sink. */
inline void utf8_linebreaks_straddles_(std::mt19937 & /*generator*/, std::size_t gap, utf8_run_sink_t sink,
                                       void *context) {
    std::string scratch;
    utf8_linebreaks_dense_mandatory_breaks_(scratch, gap), sink(context, scratch.data(), scratch.size());
    utf8_linebreaks_dense_nesting_(scratch, gap), sink(context, scratch.data(), scratch.size());
}

/** Linewrap-biased snippets: mandatory breaks, separators, NBSP/GL, nesting, hyphen, numeric. */
inline char const *const utf8_linebreaks_snippets[] = {
    "a\r\nb",        // mandatory break (CRLF)
    "a\x0C" "b",     // mandatory break (form feed, BK)
    "\xE2\x80\xA8",  // U+2028 LINE SEPARATOR
    "\xE2\x80\xA9",  // U+2029 PARAGRAPH SEPARATOR
    "a\xC2\xA0" "b", // GL glue NBSP (U+00A0)
    "(a)",           // OP/CL nesting
    "\"x\"",         // QU quotes
    "a-b",           // HY hyphen
    "1,234.5 ",      // numeric grouping
    "\xD7\x90-a",    // Hebrew letter + hyphen (LB21a)
    "\xD8\xB9\xD8\xB1\xD8\xA8\xD9\x8A\xD8\x8C\xC2\xA0\xD8\xB3\xD9\x84\xD8\xA7\xD9\x85 abc", // Arabic IS+GL, RTL/LTR
    "\xE6\x97\xA5\xE6\x9C\xAC\xE3\x80\x81\xEF\xBC\x88\xE8\xAA\x9E\xEF\xBC\x89", // U+3001 CL, U+FF08/U+FF09 OP/CL
    "\xC2\xAB\xD0\x9C\xD0\xB8\xD1\x80\xC2\xBB\xCE\x91",                         // Cyrillic in guillemets (QU) + Greek
};

/** Linewrap alphabet, biased toward family snippets and motifs (LB mandatory/GL/HY/NU/nesting). */
inline utf8_corpus_alphabet_t const utf8_linebreaks_alphabet = {
    span_over(utf8_linebreaks_snippets),
    span_over(utf8_default_boundary_codepoints),
    {{45, 15, 5, 30, 5}}, // snippet, boundary, astral, motif, malformed
};

/** One motif per UAX-14 Line_Break rule, tagged with the direction it demonstrates. Rules with
 *  both senses also carry an opposite-direction motif, since the gate compares serial against
 *  each capability on every motif. */
inline utf8_rule_case_t const utf8_linebreaks_rule_cases[] = {
    {"LB4", utf8_rule_breaks_k, "a\x0C" "b"sv},                             // BK ! (mandatory break after FF)
    {"LB5", utf8_rule_breaks_k, "a\r\nb"sv},                                // CR x LF, then LF ! (CRLF, break)
    {"LB6", utf8_rule_joins_k, "a\x0C"sv},                                  // x (BK|CR|LF|NL): no break before HB
    {"LB7", utf8_rule_joins_k, "a b"sv},                                    // x SP / x ZW: no break before space
    {"LB8", utf8_rule_breaks_k, "a\xE2\x80\x8B" "b"sv},                     // ZW SP* / break after ZWSP
    {"LB8a", utf8_rule_joins_k, "a\xE2\x80\x8D\xF0\x9F\x98\x80"sv},         // ZWJ x (no break)
    {"LB9", utf8_rule_joins_k, "a\xCC\x81"sv},                              // treat X CM* as X
    {"LB10", utf8_rule_joins_k, "\xCC\x81" "a"sv},                          // lone CM → AL
    {"LB11", utf8_rule_joins_k, "a\xE2\x81\xA0" "b"sv},                     // x WJ / WJ x (Word Joiner, no break)
    {"LB12", utf8_rule_joins_k, "\xC2\xA0" "a"sv},                          // GL x (no break after NBSP)
    {"LB12a", utf8_rule_joins_k, "a\xC2\xA0"sv},                            // [^SP BA HY] x GL
    {"LB13", utf8_rule_joins_k, "a)"sv},                                    // x CL/CP/EX/SY (no break before `)`)
    {"LB14", utf8_rule_joins_k, "(a"sv},                                    // OP SP* x (no break after OP)
    {"LB15a", utf8_rule_joins_k, "\xE2\x80\x9C" "a"sv},                     // (sot|...) [QU & Pi] SP* x (no break)
    {"LB15b", utf8_rule_joins_k, "a\xE2\x80\x9D"sv},                        // x [QU & Pf] (no break before close)
    {"LB16", utf8_rule_joins_k, ")\xE3\x82\xA1"sv},                         // (CL|CP) SP* x NS (no break)
    {"LB17", utf8_rule_joins_k, "\xE2\x80\x94\xE2\x80\x94"sv},              // B2 SP* B2 (no break between dashes)
    {"LB18", utf8_rule_breaks_k, "a b"sv},                                  // SP / break after space
    {"LB19", utf8_rule_joins_k, "a\"b"sv},                                  // x QU / QU x (no break around quote)
    {"LB20", utf8_rule_breaks_k, "a\xEF\xBF\xBC" "b"sv},                    // / CB ; CB / (break around U+FFFC)
    {"LB20a", utf8_rule_joins_k, "-a"sv},                                   // (sot|...) (HY|HH) x AL (no break)
    {"LB21", utf8_rule_joins_k, "a-b"sv},                                   // x BA/HY/HH/NS, BB x (no break)
    {"LB21a", utf8_rule_joins_k, "\xD7\x90-a"sv},                           // HL (HY|HH) x [^HL] (no break)
    {"LB21b", utf8_rule_joins_k, "/\xD7\x90"sv},                            // SY x HL (no break)
    {"LB22", utf8_rule_joins_k, "a\xE2\x80\xA6"sv},                         // x IN (no break before ellipsis)
    {"LB23", utf8_rule_joins_k, "a1"sv},                                    // (AL|HL) x NU / NU x (AL|HL)
    {"LB23a", utf8_rule_joins_k, "$\xE4\xB8\xAD"sv},                        // PR x (ID|EB|EM)
    {"LB24", utf8_rule_joins_k, "$a"sv},                                    // (PR|PO) x (AL|HL) / (AL|HL) x ...
    {"LB25", utf8_rule_joins_k, "1,5"sv},                                   // numeric clusters (no break inside)
    {"LB26", utf8_rule_joins_k, "\xE1\x84\x80\xE1\x85\xA1"sv},              // Hangul L x V (no break)
    {"LB27", utf8_rule_joins_k, "\xEA\xB0\x80\xE1\x86\xA8"sv},              // Hangul (H2 x T) continuation
    {"LB28", utf8_rule_joins_k, "ab"sv},                                    // AL x AL (no break)
    {"LB28a", utf8_rule_joins_k, "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\x95"sv}, // Brahmic aksara (C virama C)
    {"LB29", utf8_rule_joins_k, ".a"sv},                                    // IS x (AL|HL) (no break)
    {"LB30", utf8_rule_joins_k, "a("sv},                                    // (AL|HL|NU) x OP[^EAW] (no break)
    {"LB30a", utf8_rule_joins_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"sv},     // RI RI (even-parity pair, no break)
    {"LB30b", utf8_rule_joins_k, "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBB"sv},     // EB x EM (base + modifier, no break)
    {"LB31", utf8_rule_breaks_k, "\xE4\xB8\xAD\xE4\xB8\xAD"sv},             // break everywhere else (ideographs)
    // Opposite-direction motifs (E1): the same rule firing the other way.
    {"LB28", utf8_rule_breaks_k, "a b"sv}, // AL SP AL → break at the space
    {"LB25", utf8_rule_breaks_k, "1 2"sv}, // numerals split by space → break
    {"LB30a", utf8_rule_breaks_k, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA\xF0\x9F\x87\xBA"sv}, // 3 RI → break
};

/** Every Line_Break rule id the gate requires a motif for (spec-derived checklist). */
inline char const *const utf8_linebreaks_required_rules[] = {
    "LB4",   "LB5",   "LB6",  "LB7",  "LB8",  "LB8a", "LB9",   "LB10",  "LB11", "LB12",  "LB12a", "LB13", "LB14",
    "LB15a", "LB15b", "LB16", "LB17", "LB18", "LB19", "LB20",  "LB20a", "LB21", "LB21a", "LB21b", "LB22", "LB23",
    "LB23a", "LB24",  "LB25", "LB26", "LB27", "LB28", "LB28a", "LB29",  "LB30", "LB30a", "LB30b", "LB31",
};

/** Known-answer line-break vectors for one segmenter. */
inline void check_utf8_linebreaks_unit_(utf8_segment_backend_t const &backend) {
    check_utf8_segment_unit_("linewrap", backend.finder, span_over(utf8_linebreaks_unit_cases));
}

/** Rule-coverage gate: every LB motif runs, and @p backend agrees with serial at window phases. */
inline void check_utf8_linebreaks_rules_(utf8_segment_backend_t const &backend) {
    check_utf8_rule_coverage_("linewrap", sz_utf8_linebreaks_serial, backend.finder,
                              span_over(utf8_linebreaks_rule_cases), span_over(utf8_linebreaks_required_rules));
}

/** Malformed-input safety of one line segmenter. */
inline void check_utf8_linebreaks_safety_(test_context_t &context, utf8_segment_backend_t const &backend) {
    utf8_segment_backend_t const backends[] = {backend};
    check_utf8_segment_safety_(context, "linewrap", span_over(backends));
}

/** The line differential of @p backend against serial over the dense and long-range corpora. */
inline void check_utf8_linebreaks_equivalence_(test_context_t &context, utf8_segment_backend_t const &backend) {
    utf8_segment_corpora_t const corpora = {
        "linewrap", span_over(utf8_linebreaks_motifs), &utf8_linebreaks_dense_runs_, &utf8_linebreaks_straddles_,
        {},         &utf8_linebreaks_alphabet};
    utf8_segment_backend_t const candidates[] = {backend};
    check_utf8_segment_equivalence_(context, sz_utf8_linebreaks_serial, span_over(candidates), corpora,
                                    context.iterations(25)); // This family's share of the suite budget
}

#pragma endregion UTF8 Linebreaks

#pragma region UTF8 Norm

/** One backend's normalizer and normalization-violation finder, taken by pointer. */
struct utf8_norm_kernels_t {
    sz_kernel_utf8_norm_t norm;
    sz_kernel_utf8_find_denormalized_t find_denormalized;
};

/**
 *  @brief Known answers for one normalization backend on a hand-verifiable input.
 *
 *  "café" with a precomposed é (U+00E9) is already NFC but breaks NFD, where é decomposes into
 *  e + U+0301: NFC normalization is a no-op, and NFD expands it to 6 bytes.
 */
inline void check_utf8_norm_unit_(utf8_norm_kernels_t kernels) {
    char const cafe_nfc[] = "caf\xC3\xA9"; // U+00E9 (precomposed é), 5 bytes
    sz_size_t const cafe_length = (sz_size_t)(sizeof(cafe_nfc) - 1);
    verify(kernel_result<sz_cptr_t>(kernels.find_denormalized, cafe_nfc, cafe_length, sz_normal_form_nfc_k) ==
           STRINGZILLA_NULL_CHAR);
    verify(kernel_result<sz_cptr_t>(kernels.find_denormalized, cafe_nfc, cafe_length, sz_normal_form_nfd_k) !=
           STRINGZILLA_NULL_CHAR);

    char norm_buffer[64];
    sz_size_t const nfc_length = kernel_result<sz_size_t>(kernels.norm, cafe_nfc, cafe_length, sz_normal_form_nfc_k,
                                                          norm_buffer);
    verify(nfc_length == cafe_length && std::memcmp(norm_buffer, cafe_nfc, cafe_length) == 0);
    sz_size_t const nfd_length = kernel_result<sz_size_t>(kernels.norm, cafe_nfc, cafe_length, sz_normal_form_nfd_k,
                                                          norm_buffer);
    verify(nfd_length == 6u); // "caf" + 'e' + U+0301 (2-byte combining acute)
}

/**
 *  @brief Holds one normalization backend to the serial one over every assigned codepoint.
 *
 *  Encodes every assigned Unicode codepoint, shuffles the order, and normalizes the full corpus
 *  under all four normal forms, so the comparison stresses canonical ordering, every
 *  decomposition/composition path, and SIMD-block straddles. Below a multiplier of 1.0 the
 *  codepoint space is strided, not truncated, so the astral planes stay reachable on a cheap run.
 *  The normalizer may stage device buffers while retaining the same callable interface.
 */
template <typename normalize_type_>
inline void check_utf8_normalize_equivalence_(test_context_t &context, normalize_type_ const &normalize) {
    std::size_t const codepoint_stride = context.sweep_stride(0x110000);
    std::vector<sz_rune_t> all_runes;
    all_runes.reserve(0x110000 / codepoint_stride);
    for (sz_rune_t codepoint = 0; codepoint <= 0x10FFFF; codepoint += (sz_rune_t)codepoint_stride) {
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) continue; // skip surrogates
        all_runes.push_back(codepoint);
    }

    std::vector<char> input_buffer(all_runes.size() * 4);
    std::vector<char> output_reference(input_buffer.size() * 4 + 64); // decomposition can expand
    std::vector<char> output_candidate(input_buffer.size() * 4 + 64);
    std::mt19937 &generator = context.generator;
    static sz_normal_form_t const norm_forms[4] = {sz_normal_form_nfd_k, sz_normal_form_nfc_k, sz_normal_form_nfkd_k,
                                                   sz_normal_form_nfkc_k};

    std::size_t const iterations = context.iterations(3);
    for (std::size_t iteration = 0; iteration != iterations; ++iteration) {
        if (iteration > 0) std::shuffle(all_runes.begin(), all_runes.end(), generator);
        char *write_cursor = input_buffer.data();
        for (sz_rune_t codepoint : all_runes) write_cursor += sz_rune_encode(codepoint, (sz_u8_t *)write_cursor);
        sz_size_t input_length = (sz_size_t)(write_cursor - input_buffer.data());

        for (sz_normal_form_t normal_form : norm_forms) {
            sz_size_t reference_length = kernel_result<sz_size_t>(sz_utf8_norm_serial, input_buffer.data(),
                                                                  input_length, normal_form, output_reference.data());
            sz_size_t candidate_length = kernel_result<sz_size_t>(normalize, input_buffer.data(), input_length,
                                                                  normal_form, output_candidate.data());
            if (reference_length != candidate_length ||
                std::memcmp(output_reference.data(), output_candidate.data(), reference_length) != 0) {
                fmt::println(stderr, "norm mismatch (form={}, iteration={}): reference_length={} candidate_length={}",
                             (int)normal_form, iteration, (size_t)reference_length, (size_t)candidate_length);
                verify(false);
            }
        }
    }
}

/** Checks normalization and quick-check results on the same generated inputs. */
template <typename kernels_type_>
inline void check_utf8_norm_equivalence_(test_context_t &context, kernels_type_ const &candidate) {
    check_utf8_normalize_equivalence_(context, [&](sz_cptr_t input, sz_size_t length, sz_normal_form_t form,
                                                   sz_ptr_t output, sz_size_t *written, sz_stream_t stream) {
        sz_cptr_t const reference = kernel_result<sz_cptr_t>(sz_utf8_find_denormalized_serial, input, length, form);
        sz_cptr_t const produced = kernel_result<sz_cptr_t>(candidate.find_denormalized, input, length, form);
        if (reference != produced) {
            fmt::println(stderr, "norm violation mismatch (form={})", (int)form);
            verify(false);
        }
        return candidate.norm(input, length, form, output, written, stream);
    });
}

/**
 *  @brief Drives a backend's normalization-violation finder and normalizer through malformed bytes.
 *
 *  The violation finder is bounds-safe on arbitrary bytes, so it faces the full malformed battery:
 *  the only requirements are that it survives and that any returned violation pointer stays inside
 *  `[text, text + length]`. The normalizer instead documents a valid-UTF-8 precondition (the
 *  decoder "performs no validity checks") and asserts internally on garbage, so it only faces the
 *  @c sz_utf8_find_malformed subset of the same adversarial shapes - its output must stay within
 *  the documented 18× bound, and @c with_guarded_buffer_ brackets the destination with canaries and
 *  asserts they survive, like @c check_utf8_uncased_safety_.
 */
inline void check_utf8_norm_safety_(test_context_t &context, utf8_norm_kernels_t kernels) {

    std::size_t const max_input_length = utf8_unit_capacity_k;
    // The normalizer's documented worst case is 18x the input for a single-codepoint compatibility
    // decomposition (see `utf8_norm.h`), and a truncated trailing sequence may add one extra rune.
    std::size_t const norm_bound = max_input_length * 18 + 18;

    static sz_normal_form_t const norm_forms[4] = {sz_normal_form_nfd_k, sz_normal_form_nfc_k, sz_normal_form_nfkd_k,
                                                   sz_normal_form_nfkc_k};

    auto check = [&](char const *input, std::size_t input_length) {
        // The bounds-safe violation finder only has to survive and keep any returned pointer
        // inside the input, on every malformed shape, not just valid UTF-8.
        for (sz_normal_form_t normal_form : norm_forms) {
            sz_cptr_t const found = kernel_result<sz_cptr_t>(kernels.find_denormalized, input, (sz_size_t)input_length,
                                                             normal_form);
            if (found == STRINGZILLA_NULL_CHAR) continue;
            if (found >= input && found <= input + input_length) continue;
            fmt::println(stderr, "norm violation returned out-of-bounds pointer (form={}, input={})", (int)normal_form,
                         input_length);
            print_utf8_test_bytes_("input", {input, input_length});
            verify(false && "Normalization-violation finder returned a pointer outside the input");
        }

        // The normalizer asserts internally on malformed bytes, so it only sees inputs that pass
        // `sz_utf8_find_malformed`, and its output must still stay within the bound.
        if (sz_utf8_find_malformed(input, (sz_size_t)input_length) == STRINGZILLA_NULL_CHAR) {
            with_guarded_buffer_(norm_bound, [&](sz_ptr_t output, std::size_t) {
                sz_size_t const normalized = kernel_result<sz_size_t>(kernels.norm, input, (sz_size_t)input_length,
                                                                      sz_normal_form_nfkd_k, output);
                if (normalized > norm_bound) {
                    fmt::println(stderr, "Norm of invalid input returned {} bytes for {} input bytes (bound {})",
                                 (std::size_t)normalized, input_length, norm_bound);
                    print_utf8_test_bytes_("input", {input, input_length});
                    verify(false && "Normalizer output must stay within the documented bound");
                }
            });
        }
    };

    for_each_adversarial_utf8_input_(context, context.iterations(10000), check);
}

#pragma endregion UTF8 Norm

#pragma region UTF8 Uncased

/** One backend's case folder, case-insensitive search and order, and cased-rune finder. */
struct utf8_uncased_kernels_t {
    sz_kernel_utf8_uncased_fold_t fold;
    sz_kernel_utf8_uncased_search_t search;
    sz_kernel_utf8_uncased_order_t order;
    sz_kernel_utf8_find_cased_t find_cased;
};

/** Prepares @p needle with the serial analysis, the one every uncased search backend consumes. */
inline sz_utf8_uncased_needle_t uncased_needle_(char const *needle, sz_size_t needle_length) {
    sz_utf8_uncased_needle_t prepared;
    verify(sz_utf8_uncased_needle_init_serial(needle, needle_length, &prepared, nullptr) == sz_success_k);
    return prepared;
}

/** Runs one uncased-find backend over a known case and asserts the match offset and length. */
inline void check_uncased_find_unit_(                                                      //
    sz_kernel_utf8_uncased_search_t find, char const *haystack, sz_size_t haystack_length, //
    char const *needle, sz_size_t needle_length,                                           //
    sz_size_t expected_offset, sz_size_t expected_length) {
    sz_utf8_uncased_needle_t const prepared = uncased_needle_(needle, needle_length);
    sz_cptr_t match = STRINGZILLA_NULL_CHAR;
    sz_size_t match_length = 0;
    verify(find(haystack, haystack_length, &prepared, &match, &match_length, nullptr) == sz_success_k);
    verify(match != STRINGZILLA_NULL_CHAR);
    verify((sz_size_t)(match - haystack) == expected_offset);
    verify(match_length == expected_length);
}

/** Runs one uncased-fold backend over @p input and asserts the folded bytes equal @p expected. */
inline void check_uncased_fold_unit_( //
    sz_kernel_utf8_uncased_fold_t fold, char const *input, sz_size_t input_length, char const *expected) {
    char produced[64];
    sz_size_t produced_length = kernel_result<sz_size_t>(fold, input, input_length, produced);
    sz_size_t const expected_length = (sz_size_t)std::strlen(expected);
    verify(produced_length == expected_length);
    verify(std::memcmp(produced, expected, expected_length) == 0);
}

/**
 *  @brief Independent ground-truth uncased search: `fold(needle)` as a run of `fold(haystack)`.
 *
 *  Folds the haystack codepoint by codepoint into fixed-size arrays, recording each folded rune's
 *  source byte span, then folds the needle and slides it over the folded stream. The earliest run
 *  wins; the reported offset and length snap to the source codepoint boundaries, so a match that
 *  begins or ends mid-expansion ("sss" inside "ssss" from "ßß") is reported in original haystack
 *  bytes. Depends on no production kernel, so it catches bugs the base-vs-SIMD differential cannot:
 *  those where every backend agrees yet all of them are wrong. Mirrors the Python reference,
 *  @c _reference_case_insensitive_find.
 */
inline sz_cptr_t reference_uncased_find_(char const *haystack, std::size_t haystack_length, //
                                         char const *needle, std::size_t needle_length, std::size_t *match_length) {

    // The adversarial harnesses below cap haystacks at 128 bytes and needles at 32; folding can
    // triple the rune count, so these fixed windows stay comfortably ahead of the longest input.
    sz_rune_t needle_folded[128];
    std::size_t needle_folded_count = 0;
    for (char const *cursor = needle, *end = needle + needle_length; cursor < end;) {
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_decode(cursor, end, &rune);
        if (rune_length == sz_rune_invalid_k) { // Malformed byte is its own 1-byte maximal subpart, copied unfolded
            verify(needle_folded_count < 128 && "reference needle buffer overflow");
            needle_folded[needle_folded_count++] = (sz_u8_t)*cursor;
            ++cursor;
            continue;
        }
        sz_rune_t folded[3];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);
        for (sz_size_t index = 0; index < folded_count; ++index) {
            verify(needle_folded_count < 128 && "reference needle buffer overflow");
            needle_folded[needle_folded_count++] = folded[index];
        }
        cursor += rune_length;
    }
    if (needle_folded_count == 0) {
        *match_length = 0;
        return haystack;
    }

    sz_rune_t haystack_folded[512];
    sz_cptr_t source_begin[512], source_end[512];
    std::size_t haystack_folded_count = 0;
    for (char const *cursor = haystack, *end = haystack + haystack_length; cursor < end;) {
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_decode(cursor, end, &rune);
        if (rune_length == sz_rune_invalid_k) { // Malformed byte is its own 1-byte maximal subpart, copied unfolded
            verify(haystack_folded_count < 512 && "reference haystack buffer overflow");
            haystack_folded[haystack_folded_count] = (sz_u8_t)*cursor;
            source_begin[haystack_folded_count] = cursor;
            source_end[haystack_folded_count] = cursor + 1;
            ++haystack_folded_count;
            ++cursor;
            continue;
        }
        sz_rune_t folded[3];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);
        for (sz_size_t index = 0; index < folded_count; ++index) {
            verify(haystack_folded_count < 512 && "reference haystack buffer overflow");
            haystack_folded[haystack_folded_count] = folded[index];
            source_begin[haystack_folded_count] = cursor;
            source_end[haystack_folded_count] = cursor + rune_length;
            ++haystack_folded_count;
        }
        cursor += rune_length;
    }

    for (std::size_t start = 0; start + needle_folded_count <= haystack_folded_count; ++start) {
        bool equal = true;
        for (std::size_t index = 0; index < needle_folded_count; ++index)
            if (haystack_folded[start + index] != needle_folded[index]) {
                equal = false;
                break;
            }
        if (!equal) continue;
        *match_length = (std::size_t)(source_end[start + needle_folded_count - 1] - source_begin[start]);
        return source_begin[start];
    }
    *match_length = 0;
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Runs one uncased find query through two backends and the reference, which must all agree.
 *
 *  Both the match pointer offset and the matched length must agree across the base backend, the
 *  SIMD backend, and the independent fold-subset reference, including the not-found case. Because
 *  the SIMD kernels delegate short needles to the serial path, a base-vs-SIMD-only check is blind
 *  to a bug both share; the reference leg closes that gap. On mismatch prints the needle and
 *  haystack as hex.
 */
inline void check_uncased_find_three_way_(                                                //
    sz_kernel_utf8_uncased_search_t find_base, sz_kernel_utf8_uncased_search_t find_simd, //
    char const *haystack, std::size_t haystack_length,                                    //
    char const *needle, std::size_t needle_length, char const *test_name) {

    sz_utf8_uncased_needle_t const prepared = uncased_needle_(needle, needle_length);
    sz_size_t base_matched = 0, simd_matched = 0;
    sz_cptr_t base_result = STRINGZILLA_NULL_CHAR, simd_result = STRINGZILLA_NULL_CHAR;
    verify(find_base(haystack, haystack_length, &prepared, &base_result, &base_matched, nullptr) == sz_success_k);
    verify(find_simd(haystack, haystack_length, &prepared, &simd_result, &simd_matched, nullptr) == sz_success_k);

    std::size_t reference_matched = 0;
    sz_cptr_t reference_result = reference_uncased_find_(haystack, haystack_length, needle, needle_length,
                                                         &reference_matched);

    bool const base_matches_simd = base_result == simd_result &&
                                   (base_result == STRINGZILLA_NULL_CHAR || base_matched == simd_matched);
    bool const base_matches_reference = base_result == reference_result &&
                                        (base_result == STRINGZILLA_NULL_CHAR || base_matched == reference_matched);
    bool const simd_matches_reference = simd_result == reference_result &&
                                        (simd_result == STRINGZILLA_NULL_CHAR || simd_matched == reference_matched);
    if (base_matches_simd && base_matches_reference && simd_matches_reference) return;

    long const base_offset = base_result ? (long)(base_result - haystack) : -1L;
    long const simd_offset = simd_result ? (long)(simd_result - haystack) : -1L;
    long const reference_offset = reference_result ? (long)(reference_result - haystack) : -1L;
    fmt::println(stderr,
                 "{} FAIL: base offset={} len={} | simd offset={} len={} kernel={} | reference offset={} len={}",
                 test_name, base_offset, (std::size_t)base_matched, simd_offset, (std::size_t)simd_matched,
                 prepared.script, reference_offset, (std::size_t)reference_matched);
    print_utf8_test_bytes_("needle  ", {needle, needle_length});
    print_utf8_test_bytes_("haystack", {haystack, haystack_length});
    verify(base_matches_reference && "Uncased find base backend disagrees with the reference");
    verify(simd_matches_reference && "Uncased find SIMD backend disagrees with the reference");
    verify(base_matches_simd && "Uncased find backends disagree with each other");
}

/**
 *  @brief Fuzz tests uncased UTF-8 substring search with controlled haystack sizes.
 *
 *  Uses two verification modes:
 *  - Exhaustive, @p max_needles_per_haystack = 0: tests all N × (N + 1) / 2 substrings of each
 *    folded haystack
 *  - Sampled, @p max_needles_per_haystack > 0: tests up to that many random substrings per haystack
 *
 *  Algorithm:
 *  1. Generate random haystack of about @p haystack_length runes from character pool
 *  2. Case-fold the haystack
 *  3. Extract needles from folded haystack (guarantees needle exists uncasedly)
 *  4. Search needle in original (unfolded) haystack with both serial and SIMD
 *  5. Both must return identical positions
 *
 *  @param[in] haystack_length Target number of bytes in each haystack.
 *  @param[in] max_needles_per_haystack Zero for exhaustive, or how many to sample per haystack.
 *  @param[in] total_queries Total needle searches to perform across all haystacks.
 */
inline void check_uncased_find_fuzz_(std::mt19937 &generator, sz_kernel_utf8_uncased_search_t find_serial,
                                     sz_kernel_utf8_uncased_search_t find_simd,
                                     sz_kernel_utf8_uncased_fold_t uncased_fold, sz_kernel_utf8_seek_t utf8_seek,
                                     sz_kernel_utf8_count_t utf8_count, std::size_t haystack_length,
                                     std::size_t max_needles_per_haystack, std::size_t total_queries) {

    // Single ASCII bytes for mixing, drawn as the first entries of the pool below
    std::string_view const ascii_pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123 .,!?";

    // Character pool with normal + weird Unicode characters from safety profiles
    char const *char_pool[] = {
        // ASCII words for realistic text patterns
        "Hello",
        "World",
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",

        // Folded forms of the multi-byte danger characters, so both directions of a fold are tested
        "ss",  // ß (U+00DF, C3 9F) and ẞ (U+1E9E, E1 BA 9E) fold to this
        "fi",  // ﬁ (U+FB01, EF AC 81) folds to this
        "fl",  // ﬂ (U+FB02, EF AC 82) folds to this
        "ff",  // ﬀ (U+FB00, EF AC 80) folds to this
        "ffi", // ﬃ (U+FB03, EF AC 83) folds to this
        "ffl", // ﬄ (U+FB04, EF AC 84) folds to this
        "st",  // ﬅ (U+FB05), ﬆ (U+FB06) fold to this
        "k",   // K (U+212A, E2 84 AA) Kelvin folds to this
        "s",   // ſ (U+017F, C5 BF) Long S folds to this

        // Latin-1/Extended (Western European)
        "\xC3\x9F", // 'ß' (U+00DF, C3 9F) - Latin Small Letter Sharp S (folds to ss)
        "\xC3\xB6", // 'ö' (U+00F6, C3 B6) - Latin Small Letter O with Diaeresis
        "\xC3\x96", // 'Ö' (U+00D6, C3 96) - Latin Capital Letter O with Diaeresis
        "\xC3\xBC", // 'ü' (U+00FC, C3 BC) - Latin Small Letter U with Diaeresis
        "\xC3\x9C", // 'Ü' (U+00DC, C3 9C) - Latin Capital Letter U with Diaeresis
        "\xC3\xA4", // 'ä' (U+00E4, C3 A4) - Latin Small Letter A with Diaeresis
        "\xC3\x84", // 'Ä' (U+00C4, C3 84) - Latin Capital Letter A with Diaeresis
        "\xC3\xA9", // 'é' (U+00E9, C3 A9) - Latin Small Letter E with Acute
        "\xC3\x89", // 'É' (U+00C9, C3 89) - Latin Capital Letter E with Acute
        "\xC3\xA0", // 'à' (U+00E0, C3 A0) - Latin Small Letter A with Grave
        "\xC3\x80", // 'À' (U+00C0, C3 80) - Latin Capital Letter A with Grave
        "\xC3\xB1", // 'ñ' (U+00F1, C3 B1) - Latin Small Letter N with Tilde
        "\xC3\x91", // 'Ñ' (U+00D1, C3 91) - Latin Capital Letter N with Tilde
        "\xC2\xAA", // 'ª' (U+00AA, C2 AA) - Feminine Ordinal Indicator (caseless)
        "\xC2\xBA", // 'º' (U+00BA, C2 BA) - Masculine Ordinal Indicator (caseless)
        "\xC2\xB5", // 'µ' (U+00B5, C2 B5) - Micro Sign (folds to Greek mu)
        "\xC3\x85", // 'Å' (U+00C5, C3 85) - Latin Capital Letter A with Ring Above
        "\xC3\xA5", // 'å' (U+00E5, C3 A5) - Latin Small Letter A with Ring Above
        "\xC5\xBF", // 'ſ' (U+017F, C5 BF) - Latin Small Letter Long S (folds to regular s)

        // Kelvin sign and Angstrom sign
        "\xE2\x84\xAA", // 'K' (U+212A, E2 84 AA) - Kelvin Sign (folds to ASCII k)
        "\xE2\x84\xAB", // 'Å' (U+212B, E2 84 AB) - Angstrom Sign (folds to Latin-1 a-ring)

        // Turkish
        "\xC4\xB0", // 'İ' (U+0130, C4 B0) - Latin Capital Letter I with Dot Above
        "\xC4\xB1", // 'ı' (U+0131, C4 B1) - Latin Small Letter Dotless I

        // Cyrillic (Russian, Ukrainian)
        "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", // "привет" (Cyrillic privet) - Russian Hello
        "\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0", // "Москва" (Cyrillic Moskva) - Moscow
        "\xD0\xB0",                                         // 'а' (U+0430, D0 B0) - Cyrillic Small Letter A
        "\xD0\x90",                                         // 'А' (U+0410, D0 90) - Cyrillic Capital Letter A
        "\xD0\xB1",                                         // 'б' (U+0431, D0 B1) - Cyrillic Small Letter Be
        "\xD0\x91",                                         // 'Б' (U+0411, D0 91) - Cyrillic Capital Letter Be
        "\xD0\xB2",                                         // 'в' (U+0432, D0 B2) - Cyrillic Small Letter Ve
        "\xD0\x92",                                         // 'В' (U+0412, D0 92) - Cyrillic Capital Letter Ve

        // Greek (including final sigma)
        "\xCE\xB1",                         // 'α' (U+03B1, CE B1) - Greek Small Letter Alpha
        "\xCE\x91",                         // 'Α' (U+0391, CE 91) - Greek Capital Letter Alpha
        "\xCE\xB2",                         // 'β' (U+03B2, CE B2) - Greek Small Letter Beta
        "\xCE\x92",                         // 'Β' (U+0392, CE 92) - Greek Capital Letter Beta
        "\xCF\x83",                         // 'σ' (U+03C3, CF 83) - Greek Small Letter Sigma
        "\xCE\xA3",                         // 'Σ' (U+03A3, CE A3) - Greek Capital Letter Sigma
        "\xCF\x82",                         // 'ς' (U+03C2, CF 82) - Greek Small Letter Final Sigma
        "\xCE\xBA\xCF\x8C\xCF\x83\xCE\xBC", // "κόσμ" (Greek kosm) - World

        // Greek symbol forms (fold to normal counterparts - danger zone chars)
        "\xCF\x90", // 'ϐ' (U+03D0, CF 90) - Greek Beta Symbol → β
        "\xCF\x91", // 'ϑ' (U+03D1, CF 91) - Greek Theta Symbol → θ
        "\xCF\x95", // 'ϕ' (U+03D5, CF 95) - Greek Phi Symbol → φ
        "\xCF\x96", // 'ϖ' (U+03D6, CF 96) - Greek Pi Symbol → π
        "\xCF\xB0", // 'ϰ' (U+03F0, CF B0) - Greek Kappa Symbol → κ
        "\xCF\xB1", // 'ϱ' (U+03F1, CF B1) - Greek Rho Symbol → ρ
        "\xCF\xB5", // 'ϵ' (U+03F5, CF B5) - Greek Lunate Epsilon Symbol → ε

        // Greek with dialytika + tonos (expand to base + combining marks)
        "\xCE\x90", // 'ΐ' (U+0390, CE 90) - Greek Small Letter Iota with Dialytika and Tonos
        "\xCE\xB0", // 'ΰ' (U+03B0, CE B0) - Greek Small Letter Upsilon with Dialytika and Tonos

        // Armenian
        "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE", // "բարև" (Armenian barev) - Hello
        "\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E", // "ԲԱՐԵՒ" (Armenian BAREV) - HELLO
        "\xD5\xA5",                                 // 'ե' (U+0565, D5 A5) - Armenian Small Letter Ech
        "\xD6\x87",                                 // 'և' (U+0587, D6 87) - Armenian Small Ligature Ech Yiwn

        // Vietnamese/Latin Extended Additional
        "\xE1\xBB\x87", // 'ệ' (U+1EC7, E1 BB 87) - Latin Small Letter E with Circumflex and Dot Below
        "\xE1\xBB\x86", // 'Ệ' (U+1EC6, E1 BB 86) - Latin Capital Letter E with Circumflex and Dot Below
        "\xE1\xBA\xA1", // 'ạ' (U+1EA1, E1 BA A1) - Latin Small Letter A with Dot Below
        "\xE1\xBA\xA0", // 'Ạ' (U+1EA0, E1 BA A0) - Latin Capital Letter A with Dot Below
        "\xC4\x90",     // 'Đ' (U+0110, C4 90) - Latin Capital Letter D with Stroke
        "\xC4\x91",     // 'đ' (U+0111, C4 91) - Latin Small Letter D with Stroke
        "\xC6\xA0",     // 'Ơ' (U+01A0, C6 A0) - Latin Capital Letter O with Horn
        "\xC6\xA1",     // 'ơ' (U+01A1, C6 A1) - Latin Small Letter O with Horn

        // Latin ligatures (one-to-many folding)
        "\xEF\xAC\x80", // 'ﬀ' (U+FB00, EF AC 80) - Latin Small Ligature ff
        "\xEF\xAC\x81", // 'ﬁ' (U+FB01, EF AC 81) - Latin Small Ligature fi
        "\xEF\xAC\x82", // 'ﬂ' (U+FB02, EF AC 82) - Latin Small Ligature fl
        "\xEF\xAC\x83", // 'ﬃ' (U+FB03, EF AC 83) - Latin Small Ligature ffi
        "\xEF\xAC\x84", // 'ﬄ' (U+FB04, EF AC 84) - Latin Small Ligature ffl
        "\xEF\xAC\x85", // 'ﬅ' (U+FB05, EF AC 85) - Latin Small Ligature Long S T
        "\xEF\xAC\x86", // 'ﬆ' (U+FB06, EF AC 86) - Latin Small Ligature St

        // Armenian ligatures (one-to-many folding)
        "\xEF\xAC\x93", // 'ﬓ' (U+FB13, EF AC 93) - Armenian Small Ligature Men Now
        "\xEF\xAC\x94", // 'ﬔ' (U+FB14, EF AC 94) - Armenian Small Ligature Men Ech
        "\xEF\xAC\x95", // 'ﬕ' (U+FB15, EF AC 95) - Armenian Small Ligature Men Ini
        "\xEF\xAC\x96", // 'ﬖ' (U+FB16, EF AC 96) - Armenian Small Ligature Vew Now
        "\xEF\xAC\x97", // 'ﬗ' (U+FB17, EF AC 97) - Armenian Small Ligature Men Xeh

        // One-to-many expansions (U+1E96-1E9A)
        "\xE1\xBA\x96", // 'ẖ' (U+1E96, E1 BA 96) - Latin Small Letter H with Line Below
        "\xE1\xBA\x97", // 'ẗ' (U+1E97, E1 BA 97) - Latin Small Letter T with Diaeresis
        "\xE1\xBA\x98", // 'ẘ' (U+1E98, E1 BA 98) - Latin Small Letter W with Ring Above
        "\xE1\xBA\x99", // 'ẙ' (U+1E99, E1 BA 99) - Latin Small Letter Y with Ring Above
        "\xE1\xBA\x9A", // 'ẚ' (U+1E9A, E1 BA 9A) - Latin Small Letter A with Right Half Ring

        // Capital Eszett (U+1E9E) - folds to ss
        "\xE1\xBA\x9E", // 'ẞ' (U+1E9E, E1 BA 9E) - Latin Capital Letter Sharp S

        // Lowercase Sharp S (U+00DF) - folds to ss (critical danger char!)
        "\xC3\x9F", // 'ß' (U+00DF, C3 9F) - Latin Small Letter Sharp S → ss

        // Long S with dot above (U+1E9B) - folds to 'ṡ'
        "\xE1\xBA\x9B", // 'ẛ' (U+1E9B, E1 BA 9B) - Latin Small Letter Long S with Dot Above

        // Ohm sign - folds to Greek omega (danger char!)
        "\xE2\x84\xA6", // 'Ω' (U+2126, E2 84 A6) - Ohm Sign → ω (U+03C9)

        // Afrikaans n-apostrophe (U+0149) → 'n
        "\xC5\x89", // 'ŉ' (U+0149, C5 89) - Latin Small Letter N Preceded by Apostrophe

        // J-caron (U+01F0) → j + combining caron
        "\xC7\xB0", // 'ǰ' (U+01F0, C7 B0) - Latin Small Letter J with Caron

        // Modifier letter apostrophe (U+02BC) - context for n
        "\xCA\xBC", // 'ʼ' (U+02BC, CA BC) - Modifier Letter Apostrophe

        // Modifier letter right half ring (U+02BE) - context for a
        "\xCA\xBE", // 'ʾ' (U+02BE, CA BE) - Modifier Letter Right Half Ring

        // Georgian (exercises the Georgian SIMD fold kernel across its three case blocks)
        "\xE1\x82\xA0", // 'Ⴀ' (U+10A0, E1 82 A0) - Georgian Capital Letter An (Asomtavruli, folds to Mkhedruli)
        "\xE1\x82\xB0", // 'Ⴐ' (U+10B0, E1 82 B0) - Georgian Capital Letter Ras (Asomtavruli)
        "\xE1\x83\x90", // 'ა' (U+10D0, E1 83 90) - Georgian Letter An (Mkhedruli, caseless lowercase)
        "\xE1\x83\x91", // 'ბ' (U+10D1, E1 83 91) - Georgian Letter Ban (Mkhedruli)
        "\xE1\xB2\x90", // 'Ა' (U+1C90, E1 B2 90) - Georgian Mtavruli Capital Letter An (folds to Mkhedruli)
        "\xE1\xB2\xBF", // 'Ჿ' (U+1CBF, E1 B2 BF) - Georgian Mtavruli Capital Letter Caucasian Cudzin

        // Caseless scripts (for coverage)
        "\xE4\xB8\xAD\xE6\x96\x87", // "中文" (CJK zhongwen) - Caseless
        "\xE3\x81\x82\xE3\x81\x84", // "あい" (Hiragana ai) - Caseless
        "\xF0\x9F\x98\x80",         // '😀' (U+1F600, F0 9F 98 80) - Grinning Face
    };
    std::size_t const pool_size = ascii_pool.size() + span_over(char_pool).size();
    std::uniform_int_distribution<std::size_t> pool_dist(0, pool_size - 1);

    std::size_t queries_remaining = total_queries;
    std::size_t haystacks_tested = 0;

    // Pre-allocate buffers - reusable between iterations
    std::string haystack;
    std::vector<char> haystack_folded;
    haystack.reserve(haystack_length);

    while (queries_remaining > 0) {
        // 1. Generate random haystack of ~haystack_length bytes
        haystack.clear();
        while (haystack.size() < haystack_length) {
            std::size_t const index = pool_dist(generator);
            if (index < ascii_pool.size()) haystack += ascii_pool[index];
            else haystack += char_pool[index - ascii_pool.size()];
        }

        // 2. Case-fold the haystack - expands up to 3x
        haystack_folded.resize(haystack.size() * 3);
        haystack_folded.resize(
            kernel_result<sz_size_t>(uncased_fold, haystack.data(), haystack.size(), haystack_folded.data()));
        if (haystack_folded.empty()) continue;

        // 3. Count runes in folded haystack
        sz_size_t runes_in_folded_haystack = kernel_result<sz_size_t>(utf8_count, haystack_folded.data(),
                                                                      haystack_folded.size());
        if (runes_in_folded_haystack < 2) continue;

        // 4. Calculate needles for this haystack
        std::size_t const needles_in_this_haystack =
            max_needles_per_haystack == 0 ? ((runes_in_folded_haystack * (runes_in_folded_haystack + 1)) / 2)
                                          : std::min(max_needles_per_haystack, queries_remaining);

        // Helper to extract needle from folded haystack and test both implementations
        auto test_needle = [&](sz_size_t start, sz_size_t rune_count) -> bool {
            sz_cptr_t needle_start = (start == 0) ? haystack_folded.data()
                                                  : kernel_result<sz_cptr_t>(utf8_seek, haystack_folded.data(),
                                                                             haystack_folded.size(), start);
            sz_cptr_t needle_end = ((start + rune_count) == runes_in_folded_haystack)
                                       ? haystack_folded.data() + haystack_folded.size()
                                       : kernel_result<sz_cptr_t>(utf8_seek, haystack_folded.data(),
                                                                  haystack_folded.size(), start + rune_count);
            if (!needle_start || !needle_end || needle_end <= needle_start) return false;

            sz_size_t needle_bytes = needle_end - needle_start;
            sz_size_t serial_matched = 0, simd_matched = 0;
            sz_cptr_t serial_result = STRINGZILLA_NULL_CHAR, simd_result = STRINGZILLA_NULL_CHAR;
            sz_utf8_uncased_needle_t const prepared = uncased_needle_(needle_start, needle_bytes);

            verify(find_serial(haystack.data(), haystack.size(), &prepared, &serial_result, &serial_matched, nullptr) ==
                   sz_success_k);
            verify(find_simd(haystack.data(), haystack.size(), &prepared, &simd_result, &simd_matched, nullptr) ==
                   sz_success_k);

            if (serial_result != simd_result || serial_matched != simd_matched) {
                sz_size_t serial_off = serial_result ? (sz_size_t)(serial_result - haystack.data())
                                                     : STRINGZILLA_SIZE_MAX;
                sz_size_t simd_off = simd_result ? (sz_size_t)(simd_result - haystack.data()) : STRINGZILLA_SIZE_MAX;
                sz_size_t print_start = (serial_off != STRINGZILLA_SIZE_MAX && serial_off > 20) ? serial_off - 20 : 0;
                sz_size_t print_end = (serial_off != STRINGZILLA_SIZE_MAX)
                                          ? std::min(serial_off + needle_bytes + 20, haystack.size())
                                          : std::min((sz_size_t)50, haystack.size());
                std::string_view const needle_shown(needle_start, std::min(needle_bytes, (sz_size_t)50));
                std::string_view const haystack_shown(haystack.data() + print_start, print_end - print_start);
                fmt::print(stderr, R"(FUZZ FAIL haystack={haystacks} start={start} rune_count={runes}
  Haystack len={haystack_length}, needle len={needle_length}
  Needle bytes: {needle:02X}
  Serial: offset={serial_offset}, len={serial_length}
  SIMD:   offset={simd_offset}, len={simd_length}
  Prepared needle: script={kernel} offset_in_unfolded={unfolded_offset}, length_in_unfolded={unfolded_length}
  Haystack bytes from offset {shown_from}:
    {haystack:02X}
)",
                           fmt::arg("haystacks", haystacks_tested), fmt::arg("start", start),
                           fmt::arg("runes", rune_count), fmt::arg("haystack_length", haystack.size()),
                           fmt::arg("needle_length", needle_bytes), fmt::arg("needle", hex_bytes(needle_shown)),
                           fmt::arg("serial_offset", (sz_ssize_t)serial_off), fmt::arg("serial_length", serial_matched),
                           fmt::arg("simd_offset", (sz_ssize_t)simd_off), fmt::arg("simd_length", simd_matched),
                           fmt::arg("kernel", prepared.script),
                           fmt::arg("unfolded_offset", prepared.offset_in_unfolded),
                           fmt::arg("unfolded_length", prepared.length_in_unfolded),
                           fmt::arg("shown_from", print_start), fmt::arg("haystack", hex_bytes(haystack_shown)));
                verify(serial_result == simd_result && "Fuzz offset mismatch");
                verify(serial_matched == simd_matched && "Fuzz length mismatch");
            }
            return true;
        };

        // 5. Generate and test needles
        if (max_needles_per_haystack == 0) {
            // Exhaustive mode: every possible (start, length) pair from folded haystack
            for (sz_size_t start = 0; start < runes_in_folded_haystack && queries_remaining > 0; ++start) {
                for (sz_size_t rune_count = 1; rune_count <= runes_in_folded_haystack - start && queries_remaining > 0;
                     ++rune_count) {
                    if (test_needle(start, rune_count)) --queries_remaining;
                }
            }
        }
        else {
            // Sampled mode: random (start, length) pairs
            std::uniform_int_distribution<sz_size_t> start_dist(0, runes_in_folded_haystack - 1);
            for (std::size_t i = 0; i < needles_in_this_haystack && queries_remaining > 0; ++i) {
                sz_size_t start = start_dist(generator);
                std::uniform_int_distribution<sz_size_t> rune_count_dist(1, runes_in_folded_haystack - start);
                if (test_needle(start, rune_count_dist(generator))) --queries_remaining;
            }
        }

        ++haystacks_tested;
    }
}

/** One codepoint whose case fold isn't the identity, in runes and in UTF-8. */
struct uncased_fold_t {
    sz_rune_t preimage;
    sz_rune_t folded_runes[3];
    sz_size_t folded_count;
    sz_u8_t preimage_utf8[4];
    std::size_t preimage_length;
    sz_u8_t folded_utf8[12];
    std::size_t folded_length;
};

/**
 *  @brief Every non-identity fold in Unicode, derived from @c sz_unicode_fold_codepoint_.
 *
 *  The whole-range scan is slow, so a battery derives it once and hands it to each enumerator.
 */
inline std::vector<uncased_fold_t> uncased_folds_() {
    std::vector<uncased_fold_t> folds;
    for (sz_rune_t preimage = 0; preimage <= 0x10FFFF; ++preimage) {
        if (preimage >= 0xD800 && preimage <= 0xDFFF) continue; // Surrogates aren't valid UTF-8
        uncased_fold_t fold;
        fold.preimage = preimage;
        fold.folded_count = sz_unicode_fold_codepoint_(preimage, fold.folded_runes);
        if (fold.folded_count == 1 && fold.folded_runes[0] == preimage) continue; // Identity folds are inert
        fold.preimage_length = (std::size_t)sz_rune_encode(preimage, fold.preimage_utf8);
        fold.folded_length = 0;
        for (sz_size_t index = 0; index < fold.folded_count; ++index)
            fold.folded_length += (std::size_t)sz_rune_encode(fold.folded_runes[index],
                                                              fold.folded_utf8 + fold.folded_length);
        folds.push_back(fold);
    }
    return folds;
}

/**
 *  @brief Differential adversarial test for uncased search over @b all fold preimages.
 *
 *  Random fuzzing rarely places a rare preimage (like 'ϴ' U+03F4 → 'θ') right where a SIMD
 *  danger-detection "alarm" crosses a chunk boundary - structured enumeration does. For every
 *  codepoint whose @c sz_unicode_fold_codepoint_ output differs from identity, the folded output
 *  becomes the needle (bare, "x"-prefixed, "x"-suffixed) and the UTF-8 @b preimage hides in
 *  'y'-padded haystacks at offsets straddling the 64-byte SIMD chunk boundary. Haystacks are
 *  built both with and without the mirroring "x" context, so the not-found path is exercised
 *  with the same adversarial shapes.
 */
inline void check_uncased_find_preimages_(test_context_t &context, std::vector<uncased_fold_t> const &folds,
                                          sz_kernel_utf8_uncased_search_t find_base,
                                          sz_kernel_utf8_uncased_search_t find_simd) {
    std::size_t const offsets[] = {0, 14, 15, 16, 17, 30, 31, 32, 33, 61, 62, 63, 64, 65};
    std::size_t const offsets_count = span_over(offsets).size();
    std::size_t const padding_length = 16;

    char needle[16];   // Longest folded form is 9 bytes, plus one ASCII context byte
    char haystack[96]; // Largest offset 65, plus context, plus a 4-byte preimage, plus padding

    std::size_t const preimage_stride = context.sweep_stride(folds.size());

    for (std::size_t fold_index = 0; fold_index < folds.size(); fold_index += preimage_stride) {
        uncased_fold_t const &fold = folds[fold_index];

        // Needle variants: 0 = bare folded form, 1 = "x"-prefixed, 2 = "x"-suffixed
        for (std::size_t variant = 0; variant < 3; ++variant) {
            std::size_t needle_length = 0;
            if (variant == 1) needle[needle_length++] = 'x';
            std::memcpy(needle + needle_length, fold.folded_utf8, fold.folded_length);
            needle_length += fold.folded_length;
            if (variant == 2) needle[needle_length++] = 'x';

            // With the mirroring "x" context the needle must match through the fold expansion;
            // without it both backends must agree on the not-found result. The bare variant has
            // no context to mirror, so it runs once.
            for (std::size_t with_context = variant == 0 ? 1 : 0; with_context < 2; ++with_context) {
                for (std::size_t offset_index = 0; offset_index < offsets_count; ++offset_index) {
                    std::size_t haystack_length = 0;
                    while (haystack_length < offsets[offset_index]) haystack[haystack_length++] = 'y';
                    if (variant == 1 && with_context) haystack[haystack_length++] = 'x';
                    std::memcpy(haystack + haystack_length, fold.preimage_utf8, fold.preimage_length);
                    haystack_length += fold.preimage_length;
                    if (variant == 2 && with_context) haystack[haystack_length++] = 'x';
                    for (std::size_t i = 0; i < padding_length; ++i) haystack[haystack_length++] = 'y';

                    check_uncased_find_three_way_(find_base, find_simd, haystack, haystack_length, needle,
                                                  needle_length, "preimage find");
                }
            }
        }
    }
}

/**
 *  @brief Differential test for matches sitting at the very tail of the haystack, where the
 *      haystack span is wider or narrower than the folded needle window.
 *
 *  Expanding preimages ('ᾳ' U+1FB3 → "αι", 'ß' → "ss", the ﬁ/ﬀ/ﬃ ligatures, 'ŉ' U+0149) and
 *  shrinking ones ('K' Kelvin U+212A → "k", 'Å' Angstrom U+212B → "å") break the byte-for-byte
 *  relation between haystack and folded needle - exactly where SIMD tail danger-windows get cut
 *  short. The set is derived generatively: every preimage whose folded UTF-8 byte length differs
 *  from its own. Each lands within the last @c needle_window bytes (windows 4..16) of haystacks
 *  whose filler also sweeps the 64-byte SIMD chunk boundary.
 */
inline void check_uncased_find_tails_(test_context_t &context, std::vector<uncased_fold_t> const &folds,
                                      sz_kernel_utf8_uncased_search_t find_base,
                                      sz_kernel_utf8_uncased_search_t find_simd) {
    std::vector<uncased_fold_t> expanding_preimages;
    for (uncased_fold_t const &fold : folds)
        if (fold.folded_length != fold.preimage_length) // Same width → not a tail-expansion shape
            expanding_preimages.push_back(fold);

    std::size_t const filler_lengths[] = {0, 1, 2, 14, 15, 16, 17, 30, 31, 32, 33, 59, 60, 61, 62, 63, 64, 65};
    std::size_t const filler_lengths_count = span_over(filler_lengths).size();
    std::size_t const needle_window_max = 16; // Tail paddings 0..15 keep the preimage within the last 4..16 bytes

    char needle[32];    // Two folded forms of up to 12 bytes, or one with an ASCII context byte
    char haystack[128]; // Largest filler 65, plus context, plus two 4-byte preimages, plus tail padding

    std::size_t const entry_stride = context.sweep_stride(expanding_preimages.size());
    for (std::size_t entry_index = 0; entry_index < expanding_preimages.size(); entry_index += entry_stride) {
        uncased_fold_t const &entry = expanding_preimages[entry_index];

        // Needle variants: 0 = folded form, 1 = "x"-suffixed, 2 = "x"-prefixed, 3 = folded twice
        for (std::size_t variant = 0; variant < 4; ++variant) {
            std::size_t needle_length = 0;
            if (variant == 2) needle[needle_length++] = 'x';
            std::memcpy(needle + needle_length, entry.folded_utf8, entry.folded_length);
            needle_length += entry.folded_length;
            if (variant == 1) needle[needle_length++] = 'x';
            if (variant == 3) {
                std::memcpy(needle + needle_length, entry.folded_utf8, entry.folded_length);
                needle_length += entry.folded_length;
            }

            for (std::size_t filler_index = 0; filler_index < filler_lengths_count; ++filler_index) {
                for (std::size_t tail_padding = 0; tail_padding < needle_window_max; ++tail_padding) {
                    std::size_t haystack_length = 0;
                    while (haystack_length < filler_lengths[filler_index]) haystack[haystack_length++] = 'o';
                    if (variant == 2) haystack[haystack_length++] = 'x';
                    std::memcpy(haystack + haystack_length, entry.preimage_utf8, entry.preimage_length);
                    haystack_length += entry.preimage_length;
                    if (variant == 3) {
                        std::memcpy(haystack + haystack_length, entry.preimage_utf8, entry.preimage_length);
                        haystack_length += entry.preimage_length;
                    }
                    if (variant == 1) haystack[haystack_length++] = 'x';
                    for (std::size_t i = 0; i < tail_padding; ++i) haystack[haystack_length++] = 'y';

                    check_uncased_find_three_way_(find_base, find_simd, haystack, haystack_length, needle,
                                                  needle_length, "expanding tail find");
                }
            }
        }
    }
}

/**
 *  @brief Differential + ground-truth test for matches whose folded runes cross the boundary
 *      between two adjacent multi-rune-folding codepoints.
 *
 *  The byte-history serial helpers are most fragile when a needle's folded runes begin mid-way
 *  through one expanding codepoint and end mid-way through the next - e.g. needle "sss" sits inside
 *  haystack "ßß" → "ssss", beginning in the first 'ß' and finishing in the second. The preimage /
 *  expanding-tail fuzzers never reproduce this: they pad expansions with ASCII filler, so a folded
 *  run never straddles two expansions. Here we place two multi-rune-folding codepoints back to
 *  back, enumerate every proper sub-run of the combined folded stream that genuinely crosses the
 *  join, and use that sub-run (as folded bytes) as the needle - swept across the 64-byte SIMD chunk
 *  boundary - validated against the fold-subset reference.
 */
inline void check_uncased_find_crossing_(test_context_t &context, std::vector<uncased_fold_t> const &folds,
                                         sz_kernel_utf8_uncased_search_t find_base,
                                         sz_kernel_utf8_uncased_search_t find_simd) {
    // Codepoints whose fold emits more than one rune, so a needle can slice through the middle of
    // their expansion - ligatures, sharp-s, decomposed accents.
    std::vector<uncased_fold_t> expanders;
    for (uncased_fold_t const &fold : folds)
        if (fold.folded_count >= 2) expanders.push_back(fold);

    std::size_t const filler_lengths[] = {0, 1, 14, 15, 16, 17, 30, 31, 32, 33, 60, 61, 62, 63, 64, 65};
    std::size_t const filler_lengths_count = span_over(filler_lengths).size();

    char haystack[128];
    char needle[32];

    std::size_t const first_stride = context.sweep_stride(expanders.size());
    for (std::size_t first_index = 0; first_index < expanders.size(); first_index += first_stride) {
        for (std::size_t second_index = 0; second_index < expanders.size(); ++second_index) {
            // Fold the adjacent pair into a flat rune stream, tagging which source codepoint (0 or
            // 1) produced each folded rune so we keep only the needles that cross the join.
            sz_rune_t folded_runes[6];
            std::array<std::size_t, 6> rune_owner;
            std::size_t folded_total = 0;
            sz_u8_t pair_utf8[8];
            std::size_t pair_length = 0;
            for (std::size_t which = 0; which < 2; ++which) {
                uncased_fold_t const &expander = expanders[which == 0 ? first_index : second_index];
                std::memcpy(pair_utf8 + pair_length, expander.preimage_utf8, expander.preimage_length);
                pair_length += expander.preimage_length;
                for (sz_size_t index = 0; index < expander.folded_count; ++index) {
                    folded_runes[folded_total] = expander.folded_runes[index];
                    rune_owner[folded_total] = which;
                    ++folded_total;
                }
            }

            // Every proper sub-run [start, start + length) that begins in the first codepoint and
            // ends in the second is a needle whose folded runes straddle the expansion boundary.
            for (std::size_t start = 0; start < folded_total; ++start) {
                for (std::size_t length = 1; start + length <= folded_total; ++length) {
                    std::size_t const last = start + length - 1;
                    if (rune_owner[start] != 0 || rune_owner[last] != 1) continue;

                    std::size_t needle_length = 0;
                    for (std::size_t index = 0; index < length; ++index)
                        needle_length += (std::size_t)sz_rune_encode(folded_runes[start + index],
                                                                     (sz_u8_t *)needle + needle_length);

                    for (std::size_t filler_index = 0; filler_index < filler_lengths_count; ++filler_index) {
                        std::size_t haystack_length = 0;
                        while (haystack_length < filler_lengths[filler_index]) haystack[haystack_length++] = 'o';
                        std::memcpy(haystack + haystack_length, pair_utf8, pair_length);
                        haystack_length += pair_length;
                        for (std::size_t tail = 0; tail < 4; ++tail) haystack[haystack_length++] = 'y';

                        check_uncased_find_three_way_(find_base, find_simd, haystack, haystack_length, needle,
                                                      needle_length, "crossing expansion find");
                    }
                }
            }
        }
    }
}

/**
 *  @brief Reference-validated coverage for the rolling-hash (Rabin-Karp) path across expansions.
 *
 *  The short-needle helpers cover needles folding to 1-3 runes; this drives the 4+-rune path with
 *  matches that begin and end mid-expansion. A haystack of repeated expanding codepoints (e.g.
 *  "ßßßß" folds to "ssssssss") is searched for every folded sub-run of 4 or more runes - which
 *  bypasses the short helpers - swept across the 64-byte chunk boundary, each result checked
 *  against the independent fold-subset reference via @c check_uncased_find_three_way_.
 */
inline void check_uncased_find_long_crossing_fuzz_(sz_kernel_utf8_uncased_search_t find_base,
                                                   sz_kernel_utf8_uncased_search_t find_simd) {
    struct expander_t {
        char const *utf8;
        char const *folded;
    };
    expander_t const expanders[] = {
        {"\xC3\x9F", "ss"},      // ß
        {"\xEF\xAC\x80", "ff"},  // ﬀ
        {"\xEF\xAC\x83", "ffi"}, // ﬃ
    };
    std::size_t const fillers[] = {0, 30, 62, 63, 64, 65};
    std::size_t const fillers_count = span_over(fillers).size();

    char haystack[256];
    char needle[64];

    for (expander_t const &expander : expanders) {
        std::size_t const utf8_length = std::strlen(expander.utf8);
        std::size_t const folded_length = std::strlen(expander.folded);
        for (std::size_t copies = 3; copies <= 6; ++copies) {
            char folded_run[64];
            std::size_t folded_total = 0;
            for (std::size_t copy = 0; copy != copies; ++copy)
                for (std::size_t byte = 0; byte != folded_length; ++byte)
                    folded_run[folded_total++] = expander.folded[byte];

            // Every folded sub-run of 4+ runes bypasses the short helpers and lands in the rolling
            // hash, which inside a haystack of `copies` expanders crosses expansion boundaries.
            for (std::size_t needle_length = 4; needle_length <= folded_total; ++needle_length) {
                for (std::size_t start = 0; start + needle_length <= folded_total; ++start) {
                    std::memcpy(needle, folded_run + start, needle_length);
                    for (std::size_t filler_index = 0; filler_index != fillers_count; ++filler_index) {
                        std::size_t haystack_length = 0;
                        while (haystack_length < fillers[filler_index]) haystack[haystack_length++] = 'o';
                        for (std::size_t copy = 0; copy != copies; ++copy) {
                            std::memcpy(haystack + haystack_length, expander.utf8, utf8_length);
                            haystack_length += utf8_length;
                        }
                        for (std::size_t tail = 0; tail != 4; ++tail) haystack[haystack_length++] = 'y';
                        check_uncased_find_three_way_(find_base, find_simd, haystack, haystack_length, needle,
                                                      needle_length, "long crossing expansion find");
                    }
                }
            }
        }
    }
}

/**
 *  @brief The full differential + ground-truth battery for one backend's uncased find.
 *
 *  Runs the fuzzers and the structured adversarial enumerators, fold preimages, expanding tails
 *  and cross-expansion needles, of @p find_simd against the serial baseline and the reference.
 */
inline void check_uncased_find_battery_(test_context_t &context, sz_kernel_utf8_uncased_search_t find_simd) {
    sz_kernel_utf8_uncased_search_t const find_serial = sz_utf8_uncased_search_serial;
    std::size_t const queries = context.iterations(8000);

    // A miss must zero the length too, as callers read it without checking the match first.
    {
        sz_utf8_uncased_needle_t const prepared = uncased_needle_("zzz", 3);
        sz_cptr_t match = "";
        sz_size_t match_length = 7;
        verify(find_simd("Hello, World!", 13, &prepared, &match, &match_length, nullptr) == sz_success_k &&
               match == STRINGZILLA_NULL_CHAR && match_length == 0);
    }

    // One prepared needle serves every haystack, as search only reads it, and an empty one matches
    // at the start.
    {
        sz_utf8_uncased_needle_t const prepared = uncased_needle_("STRASSE", 7);
        sz_cptr_t match = STRINGZILLA_NULL_CHAR;
        sz_size_t match_length = 0;
        char const first[] = "die Stra\xC3\x9F" //
                             "e";
        verify(find_simd(first, 11, &prepared, &match, &match_length, nullptr) == sz_success_k && match == first + 4 &&
               match_length == 7);
        char const second[] = "a strasse";
        verify(find_simd(second, 9, &prepared, &match, &match_length, nullptr) == sz_success_k && match == second + 2 &&
               match_length == 7);
        sz_utf8_uncased_needle_t const empty = uncased_needle_("", 0);
        verify(find_simd(second, 9, &empty, &match, &match_length, nullptr) == sz_success_k && match == second &&
               match_length == 0);
    }

    check_uncased_find_fuzz_(context.generator, find_serial, find_simd, sz_utf8_uncased_fold_serial,
                             sz_utf8_seek_serial, sz_utf8_count_serial, 16, 0, queries);
    check_uncased_find_fuzz_(context.generator, find_serial, find_simd, sz_utf8_uncased_fold_serial,
                             sz_utf8_seek_serial, sz_utf8_count_serial, 32, 0, queries);
    check_uncased_find_fuzz_(context.generator, find_serial, find_simd, sz_utf8_uncased_fold_serial,
                             sz_utf8_seek_serial, sz_utf8_count_serial, 100, 100, queries);
    check_uncased_find_fuzz_(context.generator, find_serial, find_simd, sz_utf8_uncased_fold_serial,
                             sz_utf8_seek_serial, sz_utf8_count_serial, 200, 100, queries);
    std::vector<uncased_fold_t> const folds = uncased_folds_();
    check_uncased_find_preimages_(context, folds, find_serial, find_simd);
    check_uncased_find_tails_(context, folds, find_serial, find_simd);
    check_uncased_find_crossing_(context, folds, find_serial, find_simd);
    check_uncased_find_long_crossing_fuzz_(find_serial, find_simd);

    // A long ASCII needle (well past the 32-rune ring buffer and the 3-rune short helpers) drives
    // the pure Rabin-Karp path inside a large random haystack, both where the needle was spliced in
    // (so a match exists) and where it was not (so the not-found path is exercised at scale).
    std::mt19937 &random_generator = context.generator;
    std::uniform_int_distribution<int> letter_distribution(0, 25);
    // The independent reference oracle caps its folded haystack at 512 runes, so the haystack
    // stays under that while the 48-rune needle, past the 32-rune ring buffer, drives Rabin-Karp.
    std::string needle, haystack;
    needle.reserve(48);
    haystack.reserve(400);
    for (std::size_t length = 0; length != 48; ++length)
        needle.push_back((char)('A' + letter_distribution(random_generator)));
    haystack.clear();
    for (std::size_t length = 0; length != 400; ++length)
        haystack.push_back((char)('a' + letter_distribution(random_generator)));

    // The lowercase haystack cannot hold the uppercase needle by chance: the not-found case.
    check_uncased_find_three_way_(find_serial, find_simd, haystack.data(), haystack.size(), needle.data(),
                                  needle.size(), "long ascii needle, not found");

    // Splice the needle near the middle of the haystack so the match exists.
    std::string spliced_haystack = haystack;
    std::size_t const splice_offset = spliced_haystack.size() / 2;
    spliced_haystack.replace(splice_offset, needle.size(), needle);
    check_uncased_find_three_way_(find_serial, find_simd, spliced_haystack.data(), spliced_haystack.size(),
                                  needle.data(), needle.size(), "long ascii needle, with match");
}

/** Known-answer battery for one uncased-order backend: case-insensitive equality, ASCII less and
 *  greater, length-prefix ordering, and 2-byte accented folds (ö = C3 B6, é = C3 A9). */
inline void check_uncased_order_(sz_kernel_utf8_uncased_order_t order) {
    verify(kernel_result<sz_ordering_t>(order, "Hello", 5, "HELLO", 5) == sz_equal_k);
    verify(kernel_result<sz_ordering_t>(order, "abc", 3, "abd", 3) == sz_less_k);
    verify(kernel_result<sz_ordering_t>(order, "abd", 3, "abc", 3) == sz_greater_k);
    verify(kernel_result<sz_ordering_t>(order, "ab", 2, "abc", 3) == sz_less_k);
    verify(kernel_result<sz_ordering_t>(order, "sch\xC3\xB6ner", 8, "SCH\xC3\x96NER", 8) == sz_equal_k); // 'ö' fold
    verify(kernel_result<sz_ordering_t>(order, "caf\xC3\xA9", 5, "CAF\xC3\x89", 5) == sz_equal_k);       // 'é' fold
}

/** Compares the @p reference and @p candidate folds byte-by-byte over a fixed multi-script battery,
 *  @p min_iterations random concatenations of at least @p min_text_length bytes, and the exhaustive
 *  sweep of every valid Unicode codepoint, both in order and shuffled. The candidate is a fold
 *  kernel or anything called like one, such as a device kernel behind staged buffers. */
template <typename candidate_type_>
inline void check_uncased_fold_equivalence_(test_context_t &context, sz_kernel_utf8_uncased_fold_t reference,
                                            candidate_type_ const &candidate, sz_size_t min_text_length,
                                            sz_size_t min_iterations) {

    // Output buffers (3x input for worst-case expansion)
    std::vector<char> output_base(min_text_length * 3 + 256);
    std::vector<char> output_simd(min_text_length * 3 + 256);

    auto check = [&](std::string const &text) {
        // Ensure buffers are large enough
        if (output_base.size() < text.size() * 3 + 64) {
            output_base.resize(text.size() * 3 + 64);
            output_simd.resize(text.size() * 3 + 64);
        }

        sz_size_t base_length = kernel_result<sz_size_t>(reference, text.data(), text.size(), output_base.data());
        sz_size_t simd_length = kernel_result<sz_size_t>(candidate, text.data(), text.size(), output_simd.data());

        if (base_length != simd_length) {
            fmt::println(stderr, "Case fold length mismatch: base={}, simd={}, input_length={}", //
                         base_length, simd_length, text.size());
            // Print first divergence
            for (std::size_t i = 0; i < std::min(base_length, simd_length); ++i) {
                if (output_base[i] != output_simd[i]) {
                    fmt::println(stderr, "First byte diff at output[{}]: base=0x{:02X}, simd=0x{:02X}", //
                                 i, (unsigned char)output_base[i], (unsigned char)output_simd[i]);
                    break;
                }
            }
            verify(base_length == simd_length && "Case fold length mismatch");
        }

        for (sz_size_t i = 0; i < base_length; ++i) {
            if (output_base[i] != output_simd[i]) {
                fmt::println(stderr, "Case fold content mismatch at byte {}: base=0x{:02X}, simd=0x{:02X}", //
                             i, (unsigned char)output_base[i], (unsigned char)output_simd[i]);
                // Show context around the mismatch
                std::size_t start = i > 10 ? i - 10 : 0;
                std::size_t end = std::min(i + 10, (std::size_t)base_length);
                std::string_view const base_shown(output_base.data() + start, end - start);
                std::string_view const simd_shown(output_simd.data() + start, end - start);
                fmt::println(stderr, "Base output[{0}..{1}]: {2:02X}\nSIMD output[{0}..{1}]: {3:02X}", start, end,
                             hex_bytes(base_shown), hex_bytes(simd_shown));
                verify(output_base[i] == output_simd[i] && "Case fold content mismatch");
            }
        }
    };

    // Test content - mix of scripts with case folding rules
    static char const *const utf8_content[] = {
        // ASCII
        "",
        "a",
        "A",
        "hello",
        "HELLO",
        "Hello World",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
        // German Eszett (both ß and ẞ fold to "ss")
        "\xC3\x9F", // ß (U+00DF) → ss
        "straße",
        // Latin-1 uppercase (À-Þ range, 2-byte UTF-8 starting with C3)
        "\xC3\x80", // À (U+00C0)
        "\xC3\x89", // É (U+00C9)
        "\xC3\x96", // Ö (U+00D6)
        "\xC3\x9C", // Ü (U+00DC)
        "\xC3\x9E", // Þ (U+00DE)
        // Cyrillic (2-byte UTF-8 starting with D0-D1)
        "\xD0\x90",                                         // А (U+0410)
        "\xD0\x9F",                                         // П (U+041F)
        "\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2", // ПРИВЕТ
        "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", // привет
        // Cyrillic Special
        "\xD0\x81", // Ё
        "\xD1\x91", // ё
        "\xD0\x84", // Є
        "\xD1\x94", // є
        "\xD0\x87", // Ї
        "\xD1\x97", // ї
        // Greek (2-byte UTF-8 starting with CE-CF)
        "\xCE\x91",                                         // Α (U+0391)
        "\xCE\xA9",                                         // Ω (U+03A9)
        "\xCE\x95\xCE\xBB\xCE\xBB\xCE\xAC\xCE\xB4\xCE\xB1", // Ελλάδα
        // Armenian (2-byte UTF-8 starting with D4-D5)
        "\xD4\xB1", // Ա (U+0531)
        "\xD5\x80", // Հ (U+0540)
        // Mixed content
        "Hello \xD0\x9C\xD0\xB8\xD1\x80!",      // Hello Мир!
        "Caf\xC3\xA9 \xCE\xB1\xCE\xB2\xCE\xB3", // Café αβγ
        // Georgian uppercase (3-byte UTF-8: E1 82 A0-BF, E1 83 80-85/87/8D)
        // These fold to lowercase Mkhedruli (E2 B4 XX)
        "\xE1\x82\xA0",                         // Ⴀ (U+10A0) → ა (U+2D00)
        "\xE1\x82\xB0",                         // Ⴐ (U+10B0) → ⴐ (U+2D10)
        "\xE1\x83\x80",                         // Ⴠ (U+10C0) → ⴠ (U+2D20)
        "\xE1\x83\x85",                         // Ⴥ (U+10C5) → ⴥ (U+2D25)
        "\xE1\x82\xA0\xE1\x82\xA1\xE1\x82\xA2", // ႠႡႢ → ⴀⴁⴂ
        "\xE1\x83\x90\xE1\x83\x91\xE1\x83\x92", // ა ბ გ (lowercase, no change)
        // Georgian mixed with ASCII (tests fast-path interaction)
        ("Hello \xE1\x82\xA0\xE1\x82\xA1 World"),          // Hello ႠႡ World
        ("ABC\xE1\x82\xA0\xE1\x82\xA1\xE1\x82\xA2" "DEF"), // ABCႠႡႢdef
        // Emojis (no case folding, should pass through)
        "\xF0\x9F\x98\x80",             // 😀
        "Hello \xF0\x9F\x8C\x8D World", // Hello 🌍 World
    };

    std::mt19937 &generator = context.generator;
    std::size_t const content_count = span_over(utf8_content).size();
    std::uniform_int_distribution<std::size_t> content_dist(0, content_count - 1);

    // First, test all the fixed strings
    for (std::size_t i = 0; i < content_count; ++i) { check(utf8_content[i]); }

    // Generate and test many random strings
    for (std::size_t iteration = 0; iteration < min_iterations; ++iteration) {
        std::string text;

        // Build up a random string of at least `min_text_length` bytes
        while (text.size() < min_text_length) {
            std::size_t content_index = content_dist(generator);
            text.append(utf8_content[content_index]);
        }
        check(text);
    }

    // Sweep the valid Unicode codepoints, first in order (0x0..0x10FFFF), then shuffled, so one
    // fold differential drives both the structured strings and the whole-codepoint enumeration.
    std::size_t const codepoint_stride = context.sweep_stride(0x110000);
    std::vector<sz_rune_t> all_runes;
    all_runes.reserve(0x110000 / codepoint_stride);
    for (sz_rune_t codepoint = 0; codepoint <= 0x10FFFF; codepoint += codepoint_stride) {
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) continue; // Skip surrogates
        all_runes.push_back(codepoint);
    }

    std::vector<char> input_buffer(all_runes.size() * 4); // Max UTF-8 size is 4 bytes per rune
    std::size_t const sweep_iterations = context.iterations(6);
    for (std::size_t iteration = 0; iteration < sweep_iterations; ++iteration) {
        if (iteration > 0) std::shuffle(all_runes.begin(), all_runes.end(), generator);

        char *write_cursor = input_buffer.data();
        for (sz_rune_t codepoint : all_runes) write_cursor += sz_rune_encode(codepoint, (sz_u8_t *)write_cursor);
        std::string const text(input_buffer.data(), (std::size_t)(write_cursor - input_buffer.data()));
        check(text);
    }
}

/**
 *  @brief Closure property of the case-invariant classifier over the Unicode fold table.
 *
 *  A rune may be treated as case-invariant only if no uncased match can start or hide inside it.
 *  That demands two closures over @c sz_unicode_fold_codepoint_: every preimage with a non-identity
 *  fold participates in case, and every rune @b emitted by such a fold can appear inside a folded
 *  expansion (like 'ʾ' U+02BE inside 'ẚ' → "aʾ"), so neither may be invariant.
 *  Fully generative: a Unicode table update re-derives the expected set automatically.
 */
inline void check_uncased_invariant_reference_() {
    for (uncased_fold_t const &fold : uncased_folds_()) {
        if (sz_rune_is_uncased_(fold.preimage) != sz_false_k) {
            fmt::println(stderr, "Fold preimage U+{:04X} is wrongly classified as case-invariant", fold.preimage);
            verify(false && "Fold preimages must not be case-invariant");
        }

        for (sz_size_t index = 0; index < fold.folded_count; ++index) {
            if (sz_rune_is_uncased_(fold.folded_runes[index]) != sz_false_k) {
                fmt::println(stderr,
                             "Fold output U+{:04X} (from preimage U+{:04X}) is wrongly classified as case-invariant",
                             fold.folded_runes[index], fold.preimage);
                verify(false && "Fold outputs must not be case-invariant");
            }
        }
    }
}

/**
 *  @brief Known answers for one uncased backend on hand-verifiable inputs.
 *
 *  Search, fold, order and the cased-rune finder, each against a hand-derived ground truth, so a
 *  regression that the serial-vs-SIMD agreement tests would miss - because both share a wrong
 *  constant - is still caught.
 */
inline void check_utf8_uncased_unit_(utf8_uncased_kernels_t kernels) {
    // "world" matches case-insensitively in "Hello World" at offset 6, length 5.
    char const *greeting = "Hello World";
    check_uncased_find_unit_(kernels.search, greeting, (sz_size_t)std::strlen(greeting), "world", 5, 6, 5);

    // 'ß' (U+00DF, C3 9F) folds to "ss", so the needle "SS" matches the whole 'ß' of "ßfox".
    char const *sharp_s = "\xC3\x9F" "fox"; // The split ends the hex escape before the 'f' of "fox"
    check_uncased_find_unit_(kernels.search, sharp_s, (sz_size_t)std::strlen(sharp_s), "SS", 2, 0, 2);

    // "HeLLo" folds to "hello", and 'ß' (U+00DF) to "ss".
    check_uncased_fold_unit_(kernels.fold, "HeLLo", 5, "hello");
    check_uncased_fold_unit_(kernels.fold, "\xC3\x9F", 2, "ss");

    check_uncased_order_(kernels.order);

    // "价格 123" is caseless (CJK + digits + space), so no rune participates in case, while "123Abc"
    // has its first cased codepoint 'A' at byte offset 3.
    char const *caseless = "\xE4\xBB\xB7\xE6\xA0\xBC 123"; // "价格 123"
    verify(kernel_result<sz_cptr_t>(kernels.find_cased, caseless, (sz_size_t)std::strlen(caseless)) ==
           STRINGZILLA_NULL_CHAR);
    char const *mixed = "123Abc";
    verify(kernel_result<sz_cptr_t>(kernels.find_cased, mixed, (sz_size_t)std::strlen(mixed)) == mixed + 3);

    // A cased rune hiding behind a caseless prefix longer than any SIMD front's block: sixteen
    // emojis (64 bytes) then 'a'. Regression vector for the serial-bail call that rescanned the
    // proven-caseless prefix instead of the remaining suffix and reported the string caseless.
    std::string deep_cased;
    for (std::size_t i = 0; i < 16; ++i) deep_cased += "\xF0\x9F\x98\x80"; // "😀"
    deep_cased += 'a';
    verify(kernel_result<sz_cptr_t>(kernels.find_cased, deep_cased.data(), (sz_size_t)deep_cased.size()) ==
           deep_cased.data() + 64);
}

/** Holds one uncased backend to the serial one: the fold differential and the full find battery,
 *  paired so their coverage stays in lockstep. */
inline void check_utf8_uncased_equivalence_(test_context_t &context, utf8_uncased_kernels_t kernels) {
    check_uncased_fold_equivalence_(context, sz_utf8_uncased_fold_serial, kernels.fold, 4000, context.iterations(1200));
    check_uncased_find_battery_(context, kernels.search);
}

/**
 *  @brief Feeds invalid UTF-8 through one backend's fold, search and cased-rune finder, asserting
 *      each survives and the fold writes nothing past its guarded destination.
 *
 *  Outputs are arbitrary off-contract, so only the fold length is checked, against 3 × length + 4:
 *  a truncated multi-byte tail can mis-decode into one rune of up to 4 bytes past the 3× expansion
 *  the contract documents.
 */
inline void check_utf8_uncased_safety_(test_context_t &context, utf8_uncased_kernels_t kernels) {
    // Short valid needle: the folds of 'ﬅ' and 'ﬆ' collapse onto it
    sz_utf8_uncased_needle_t const prepared = uncased_needle_("st", 2);

    auto check = [&](char const *input, std::size_t input_length) {
        // A canary-guarded fold output buffer catches any write past the documented bound, as the
        // helper asserts the flanking guards survive the call.
        std::size_t const output_bound = input_length * 3 + 4;
        with_guarded_buffer_(output_bound, [&](sz_ptr_t output, std::size_t length) {
            sz_size_t folded_length = kernel_result<sz_size_t>(kernels.fold, input, input_length, output);
            if (folded_length > length) {
                fmt::println(stderr, "Fold of invalid input returned {} bytes for {} input bytes",
                             (std::size_t)folded_length, input_length);
                print_utf8_test_bytes_("input", {input, input_length});
                verify(false && "Fold output must stay within 3x the input length plus one mis-decoded rune");
            }
        });
        // The classifier and the finder return arbitrary verdicts on garbage, and must only survive
        kernel_result<sz_cptr_t>(kernels.find_cased, input, input_length);
        sz_cptr_t match = STRINGZILLA_NULL_CHAR;
        sz_size_t match_length = 0;
        verify(kernels.search(input, input_length, &prepared, &match, &match_length, nullptr) == sz_success_k);
    };

    for_each_adversarial_utf8_input_(context, context.iterations(10000), check);
}

#pragma endregion UTF8 Uncased

} // namespace ashvardanian::stringzilla::test

#endif // STRINGZILLA_TEST_CROSS_HPP
