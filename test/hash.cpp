/**
 *  @brief  Hashing, multi-seed hashing, random-generator, and SHA256 equivalence tests.
 *  @file   scripts/test_hash.cpp
 *  @author Ash Vardanian
 *  @date June 16, 2026
 */
#undef NDEBUG // ! Enable all assertions for testing

/**
 *  The Visual C++ run-time library detects incorrect iterator use,
 *  and asserts and displays a dialog box at run time on Windows.
 */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

/**
 *  ! Overload the following with caution.
 *  ! Those parameters must never be explicitly set during releases,
 *  ! but they come handy during development, if you want to validate
 *  ! different ISA-specific implementations.

 #define SZ_USE_WESTMERE 0
 #define SZ_USE_HASWELL 0
 #define SZ_USE_GOLDMONT 0
 #define SZ_USE_SKYLAKE 0
 #define SZ_USE_ICELAKE 0
 #define SZ_USE_NEON 0
 #define SZ_USE_SVE 0
 #define SZ_USE_SVE2 0
 */
#define SZ_USE_MISALIGNED_LOADS 0
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/**
 *  Make sure to include the StringZilla headers before anything else,
 *  to intercept missing `#include` directives and other issues.
 */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h> // We use ASAN API to poison memory addresses
#endif

#include <cstdio>  // `std::printf`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::transform`
#include <iterator>      // `std::distance`
#include <map>           // `std::map`
#include <memory>        // `std::allocator`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::random_device`
#include <set>           // `std::set`
#include <sstream>       // `std::ostringstream`
#include <string>        // `std::string` baseline
#include <string_view>   // `std::string_view` baseline
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#if !SZ_IS_CPP11_
#error "This test requires C++11 or later."
#endif

#include "stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sv; // for `sz::string_view`
using sz::literals::operator""_bs; // for `sz::byteset`

#if SZ_IS_CPP17_
using namespace std::literals; // for ""sv
#endif

#pragma region Helpers

/** @brief Parses a 64-character lowercase-hex SHA256 digest into 32 bytes. */
static void sha256_digest_from_hex_(char const *hex, sz_u8_t (&digest)[SZ_SHA256_DIGEST_LENGTH]) {
    auto nibble = [](char character) -> sz_u8_t {
        if (character >= '0' && character <= '9') return (sz_u8_t)(character - '0');
        return (sz_u8_t)(character - 'a' + 10);
    };
    for (std::size_t byte_index = 0; byte_index != SZ_SHA256_DIGEST_LENGTH; ++byte_index)
        digest[byte_index] = (sz_u8_t)((nibble(hex[byte_index * 2]) << 4) | nibble(hex[byte_index * 2 + 1]));
}

/** @brief Runs one SHA256 backend (init/update/digest) over `message` and asserts the expected digest. */
static void check_sha256_unit_(                                   //
    sz_sha256_state_init_t init, sz_sha256_state_update_t update, //
    sz_sha256_state_digest_t digest, std::string const &message, char const *expected_hex) {
    sz_sha256_state_t state;
    sz_u8_t produced[SZ_SHA256_DIGEST_LENGTH], expected[SZ_SHA256_DIGEST_LENGTH];
    sha256_digest_from_hex_(expected_hex, expected);
    init(&state);
    update(&state, message.data(), (sz_size_t)message.size());
    digest(&state, produced);
    verify(std::memcmp(produced, expected, SZ_SHA256_DIGEST_LENGTH) == 0);
}

/** @brief A message paired with the digest it must produce, for known-answer testing. */
struct known_sha256_t {
    char const *message;
    char const *digest_hex;
};

/** @brief Runs one multi-state SHA256 backend over a batch of vectors and asserts every expected digest. */
static void check_sha256_multistate_unit_(                                      //
    sz_sha256_multistate_update_t update, sz_sha256_multistate_digest_t digest, //
    known_sha256_t const *vectors, std::size_t vectors_count) {
    std::vector<sz_sha256_state_t> states(vectors_count);
    std::vector<sz_u8_t> produced(vectors_count * SZ_SHA256_DIGEST_LENGTH);
    std::vector<sz_string_view_t> messages(vectors_count);
    sz_u8_t expected[SZ_SHA256_DIGEST_LENGTH];
    for (std::size_t lane_index = 0; lane_index != vectors_count; ++lane_index) {
        sz_sha256_state_init(&states[lane_index]);
        messages[lane_index].start = vectors[lane_index].message;
        messages[lane_index].length = (sz_size_t)std::strlen(vectors[lane_index].message);
    }
    sz_sequence_t texts;
    sz_sequence_from_string_views(messages.data(), vectors_count, &texts);
    update(states.data(), &texts);
    digest(states.data(), (sz_size_t)vectors_count, produced.data());
    for (std::size_t lane_index = 0; lane_index != vectors_count; ++lane_index) {
        sha256_digest_from_hex_(vectors[lane_index].digest_hex, expected);
        verify(std::memcmp(&produced[lane_index * SZ_SHA256_DIGEST_LENGTH], expected, SZ_SHA256_DIGEST_LENGTH) == 0);
    }
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer unit tests for the hashing family on simple, hand-verifiable inputs.
 *
 *  Exercises each function through the dispatched C API (automatic kernel resolution), through the
 *  natively-compiled backend kernels directly (manual propagation to a specific kernel), and through
 *  the C++ `sz::string_view` wrappers, so a regression that the serial-vs-SIMD agreement tests would
 *  miss - because both share a wrong constant - is still caught against an external ground truth.
 */
void test_hash_unit() {
    std::printf("  - testing hashing known-answer vectors...\n");

    // SHA256: the three canonical FIPS 180-4 vectors (empty, "abc", and the 56-byte two-block message).
    known_sha256_t const sha256_vectors[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},    //
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}, //
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",                 //
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    };
    for (known_sha256_t const &vector : sha256_vectors) {
        check_sha256_unit_(sz_sha256_state_init, sz_sha256_state_update, // Dispatched (automatic kernel)
                           sz_sha256_state_digest, vector.message, vector.digest_hex);
        check_sha256_unit_(sz_sha256_state_init_serial, sz_sha256_state_update_serial, // Manual: serial kernel
                           sz_sha256_state_digest_serial, vector.message, vector.digest_hex);
#if SZ_USE_GOLDMONT
        check_sha256_unit_(sz_sha256_state_init_goldmont, sz_sha256_state_update_goldmont, // Manual: goldmont kernel
                           sz_sha256_state_digest_goldmont, vector.message, vector.digest_hex);
#endif
    }

    // The same vectors as one batch, so the multi-state entry point is pinned to known digests rather than
    // only to another implementation of itself
    std::size_t const sha256_vectors_count = sizeof(sha256_vectors) / sizeof(sha256_vectors[0]);
    check_sha256_multistate_unit_(sz_sha256_multistate_update, // Dispatched (automatic kernel)
                                  sz_sha256_multistate_digest, sha256_vectors, sha256_vectors_count);
    check_sha256_multistate_unit_(sz_sha256_multistate_update_serial, // Manual: serial kernel
                                  sz_sha256_multistate_digest_serial, sha256_vectors, sha256_vectors_count);

    // An embedded-NUL message must hash past the NUL: `"abc\x00def"` is 7 bytes, not 3. Construct the
    // `std::string` with an explicit length so the interior NUL is retained.
    std::string const embedded_nul("abc\x00" "def", 7);
    verify(embedded_nul.size() == 7);
    check_sha256_unit_(sz_sha256_state_init, sz_sha256_state_update, // Dispatched (automatic kernel)
                       sz_sha256_state_digest, embedded_nul,
                       "516a5e926ce20c5f4d80f00e1a01abdf14986def6588d6abeed9fce090bc660c");
    check_sha256_unit_(sz_sha256_state_init_serial, sz_sha256_state_update_serial, // Manual: serial kernel
                       sz_sha256_state_digest_serial, embedded_nul,
                       "516a5e926ce20c5f4d80f00e1a01abdf14986def6588d6abeed9fce090bc660c");

    // `sz_bytesum` is an order-independent byte sum, so "abc" sums to 0x61 + 0x62 + 0x63 = 0x126.
    let_verify(auto bytesum_abc = sz_bytesum("abc", 3), bytesum_abc == 0x126u); // Dispatched (automatic kernel)
    verify(sz_bytesum_serial("abc", 3) == 0x126u); // Manual propagation to the serial kernel
#if SZ_USE_ICELAKE
    verify(sz_bytesum_icelake("abc", 3) == 0x126u);
#endif

    // The byte sum must include the interior NUL byte: "abc\x00def" sums to 0x255.
    let_verify(auto bytesum_nul = sz_bytesum(embedded_nul.data(), embedded_nul.size()), bytesum_nul == 0x255u);
    verify(sz_bytesum_serial(embedded_nul.data(), embedded_nul.size()) == 0x255u);

    // `sz_hash` is deterministic; the dispatched result equals the serial kernel and the C++ wrapper,
    // and a different seed must change the digest.
    char const *fox = "The quick brown fox";
    sz_size_t const fox_length = (sz_size_t)std::strlen(fox);
    let_verify(auto hash_fox = sz_hash(fox, fox_length, 0u), hash_fox == sz_hash(fox, fox_length, 0u)); // Deterministic
    verify(sz_hash(fox, fox_length, 0u) == sz_hash_serial(fox, fox_length, 0u));     // Dispatch == serial
    verify(sz_hash(fox, fox_length, 0u) != sz_hash(fox, fox_length, 1u));            // Seed changes output
    verify(sz::string_view(fox, fox_length).hash() == sz_hash(fox, fox_length, 0u)); // C++ wrapper

    // The hash must also read past an interior NUL, so truncating at the NUL changes the digest.
    let_verify(auto hash_nul = sz_hash(embedded_nul.data(), embedded_nul.size(), 0u),
               hash_nul == sz_hash_serial(embedded_nul.data(), embedded_nul.size(), 0u)); // Dispatch == serial
    verify(sz_hash(embedded_nul.data(), embedded_nul.size(), 0u) != sz_hash(embedded_nul.data(), 3u, 0u));
}

#pragma endregion // Unit

#pragma region Equivalence

/** @brief Wraps a hashing backend (one-shot + streaming) by its kernel pointers. */
template <sz_hash_t hash_, sz_hash_state_init_t init_, sz_hash_state_update_t update_, sz_hash_state_digest_t digest_>
struct hash_from_sz_ {
    sz_u64_t operator()(sz_cptr_t text, sz_size_t length, sz_u64_t seed) const noexcept {
        return hash_(text, length, seed);
    }
    void init(sz_hash_state_t *state, sz_u64_t seed) const noexcept { init_(state, seed); }
    void update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) const noexcept {
        update_(state, text, length);
    }
    sz_u64_t digest(sz_hash_state_t const *state) const noexcept { return digest_(state); }
};

/** @brief Wraps a multi-seed hashing backend (batch + single-seed) by its kernel pointers. */
template <sz_hash_multiseed_t multiseed_, sz_hash_t hash_one_>
struct hash_multiseed_from_sz_ {
    void multiseed(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds, sz_size_t seed_count,
                   sz_u64_t *output) const noexcept {
        multiseed_(text, length, seeds, seed_count, output);
    }
    sz_u64_t hash_one(sz_cptr_t text, sz_size_t length, sz_u64_t seed) const noexcept {
        return hash_one_(text, length, seed);
    }
};

/** @brief Wraps a SHA256 backend (init/update/digest) by its kernel pointers. */
template <sz_sha256_state_init_t init_, sz_sha256_state_update_t update_, sz_sha256_state_digest_t digest_>
struct sha256_from_sz_ {
    void init(sz_sha256_state_t *state) const noexcept { init_(state); }
    void update(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length) const noexcept {
        update_(state, text, length);
    }
    void digest(sz_sha256_state_t *state, sz_u8_t *output) const noexcept { digest_(state, output); }
};

/** @brief Wraps a multi-state SHA256 backend (update/digest) by its kernel pointers. */
template <sz_sha256_multistate_update_t update_, sz_sha256_multistate_digest_t digest_>
struct sha256_multistate_from_sz_ {
    void update(sz_sha256_state_t *states, sz_sequence_t const *texts) const noexcept { update_(states, texts); }
    void digest(sz_sha256_state_t const *states, sz_size_t states_count, sz_u8_t *digests) const noexcept {
        digest_(states, states_count, digests);
    }
};

/** @brief Wraps a pseudo-random fill backend by its kernel pointer. */
template <sz_fill_random_t generate_>
struct fill_random_from_sz_ {
    void operator()(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) const noexcept { generate_(text, length, nonce); }
};

/** @brief Wraps a byte-summing backend by its kernel pointer. */
template <sz_bytesum_t bytesum_>
struct bytesum_from_sz_ {
    sz_u64_t operator()(sz_cptr_t text, sz_size_t length) const noexcept { return bytesum_(text, length); }
};

/**
 *  @brief Cross-checks a byte-summing backend against a reference across lengths and alignments.
 *
 *  The wide kernels split a buffer into an unaligned head, an aligned body, and a tail, so sweeping
 *  cache-line offsets is what reaches the head and tail paths at all. The AVX-512 tiers additionally
 *  switch to non-temporal loads and bidirectional traversal past a megabyte, which only one oversized
 *  input reaches. `inputs` arrives already scaled by the caller.
 */
template <typename reference_, typename candidate_>
void test_bytesum_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    // A sum of bytes is order-independent, so a run of one repeated byte must total `length * byte` on
    // any backend. This invariant holds without consulting the reference at all.
    std::vector<std::size_t> const uniform_lengths = {0, 1, 63, 64, 65, 4096};
    for (auto length : uniform_lengths) {
        std::string const uniform(length, static_cast<char>(0xA5));
        verify(candidate(uniform.data(), static_cast<sz_size_t>(length)) == static_cast<sz_u64_t>(length) * 0xA5ull);
    }

    // The fixed ladder covers the sub-register, register, cache-line, and multi-block tiers; each length
    // is walked across cache-line offsets so the head and tail paths see every misalignment.
    std::vector<std::size_t> const lengths = {1, 11, 23, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000};
    for (auto length : lengths)
        for_each_cacheline_offset_(length, [&](sz_ptr_t pointer, std::size_t offset) {
            sz_unused_(offset);
            randomize_string(pointer, length);
            verify(reference(pointer, static_cast<sz_size_t>(length)) ==
                   candidate(pointer, static_cast<sz_size_t>(length)));
        });

    // Beyond the ladder, fuzz a contiguous run of random lengths at a single alignment.
    std::string text;
    for (sz_size_t length = 0; length != inputs; ++length) {
        text.resize(length);
        randomize_string(&text[0], length);
        verify(reference(text.data(), length) == candidate(text.data(), length));
    }

    // One oversized input, since the Skylake and Ice Lake kernels take a different branch past a megabyte.
    // The trailing bytes keep the buffer off a page boundary so the head and tail still have work to do.
    std::string huge(1024ull * 1024ull + 129ull, '\0');
    randomize_string(&huge[0], huge.size());
    verify(reference(huge.data(), (sz_size_t)huge.size()) == candidate(huge.data(), (sz_size_t)huge.size()));
}

/**
 *  @brief Hashes a string and compares the output between a reference and a candidate hashing backend.
 *
 *  The test covers increasingly long and complex strings, starting with "abcabc..." repetitions and
 *  progressing towards corner cases like empty strings, all-zero inputs, zero seeds, and so on.
 */
template <typename reference_, typename candidate_>
void test_hash_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    auto test_on_seed = [&](std::string const &text, sz_u64_t seed) {
        // Compute the entire hash at once, expecting the same output
        sz_u64_t result_base = reference(text.data(), text.size(), seed);
        sz_u64_t result_simd = candidate(text.data(), text.size(), seed);
        verify(result_base == result_simd);

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
        verify(sz_hash_state_equal(&state_base, &state_misaligned) == sz_true_k); // Same across platforms

        // Try breaking those strings into arbitrary chunks, expecting the same output in the streaming mode.
        // The length of each chunk and the number of chunks will be determined with a coin toss.
        iterate_in_random_slices(text, [&](std::string slice) {
            reference.update(&state_base, slice.data(), slice.size());
            candidate.update(&state_simd, slice.data(), slice.size());
            verify(sz_hash_state_equal(&state_base, &state_simd) == sz_true_k); // Same across platforms

            candidate.update(&state_misaligned, slice.data(), slice.size());
            verify(sz_hash_state_equal(&state_base, &state_misaligned) == sz_true_k); // Same across platforms

            result_base = reference.digest(&state_base);
            result_simd = candidate.digest(&state_simd);
            verify(result_base == result_simd);
            sz_u64_t result_misaligned = candidate.digest(&state_misaligned);
            verify(result_base == result_misaligned);
        });
    };

    // Let's try different-length strings repeating a "abc" pattern:
    std::vector<sz_u64_t> seeds = {
        0u,
        42u,                                  //
        std::numeric_limits<sz_u32_t>::max(), //
        std::numeric_limits<sz_u64_t>::max(), //
    };
    // A fixed repeat ladder: `inputs` below already carries the multiplier, and scaling both would compound.
    for (auto seed : seeds)
        for (std::size_t copies = 1; copies != 100; ++copies) //
            test_on_seed(repeat("abc", copies), seed);

    // Let's try truly random inputs of different lengths, placing each input at every sub-cache-line
    // offset so serial-vs-ISA agreement is checked across all alignments the SIMD kernels may hit.
    for (sz_size_t length = 0; length != inputs; ++length) {
        for_each_cacheline_offset_(length, [&](sz_ptr_t pointer, std::size_t offset) {
            sz_unused_(offset);
            randomize_string(pointer, length);
            std::string text(pointer, length);
            for (auto seed : seeds) test_on_seed(text, seed);
        });
    }
}

/**
 *  @brief Verifies a backend's batch multi-seed output equals a loop of its own single-seed hashes,
 *         across many lengths and seed counts (covering the 4-lane tail handling).
 *
 *  This is a single-backend self-consistency check: the candidate's `multiseed` must agree with its
 *  own `hash_one` for every seed, so a wrong shared constant in both is still caught against the
 *  per-seed reduction rather than a sibling backend.
 */
template <typename candidate_>
void test_hash_multiseed_equivalence(candidate_ candidate, sz_size_t inputs) {
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
                verify(output[index] == candidate.hash_one(text.data(), text.size(), seeds[index]));
            verify(output[seed_count] == 0xDEADBEEFDEADBEEFull); // No overwrite past `seed_count`
        }
    };

    // Cover the minimal (<= 64 byte) ladder boundaries and the wide path, well past the 64-byte tier
    // and into a few kilobytes so the shared-input load over multiple blocks is exercised too.
    for (std::size_t length = 0; length != inputs; ++length) {
        std::string text(length, '\0');
        randomize_string(&text[0], length);
        check(text);
    }
}

/**
 *  @brief Tests Pseudo-Random Number Generators (PRNGs) ensuring that the same nonce
 *         produces exactly the same output across a reference and a candidate implementation.
 */
template <typename reference_, typename candidate_>
void test_random_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    auto test_on_nonce = [&](std::size_t length, sz_u64_t nonce) {
        std::string text_base(length, '\0');
        std::string text_simd(length, '\0');
        reference(&text_base[0], static_cast<sz_size_t>(length), nonce);
        candidate(&text_simd[0], static_cast<sz_size_t>(length), nonce);
        verify(text_base == text_simd);
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

    // Beyond the structured ladder, fuzz a contiguous run of random lengths. `inputs` arrives already scaled
    // by the caller, so scaling it again here would make the dial multiplicative.
    for (sz_size_t length = 0; length != inputs; ++length)
        for (auto nonce : nonces) //
            test_on_nonce(length, nonce);
}

/**
 *  @brief Cross-checks SHA256 backends against each other (reference vs candidate) on random inputs,
 *         one-shot and incremental. The known-answer FIPS 180-4 vectors live in `test_hash_unit`.
 *         `inputs` is the maximum length fuzzed, inclusive.
 */
template <typename reference_, typename candidate_>
void test_sha256_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    // Test random inputs of various lengths
    for (sz_size_t length = 0; length <= inputs; ++length) {
        std::string random_text(length, '\0');
        randomize_string(&random_text[0], length);

        sz_sha256_state_t state_base, state_simd;
        sz_u8_t digest_base_result[SZ_SHA256_DIGEST_LENGTH], digest_simd_result[SZ_SHA256_DIGEST_LENGTH];

        // One-shot hashing
        reference.init(&state_base);
        candidate.init(&state_simd);
        reference.update(&state_base, random_text.data(), length);
        candidate.update(&state_simd, random_text.data(), length);
        reference.digest(&state_base, digest_base_result);
        candidate.digest(&state_simd, digest_simd_result);
        verify(std::memcmp(digest_base_result, digest_simd_result, SZ_SHA256_DIGEST_LENGTH) == 0);

        // Incremental hashing with random chunks
        reference.init(&state_base);
        candidate.init(&state_simd);
        iterate_in_random_slices(random_text, [&](std::string slice) {
            reference.update(&state_base, slice.data(), slice.size());
            candidate.update(&state_simd, slice.data(), slice.size());
        });
        reference.digest(&state_base, digest_base_result);
        candidate.digest(&state_simd, digest_simd_result);
        verify(std::memcmp(digest_base_result, digest_simd_result, SZ_SHA256_DIGEST_LENGTH) == 0);
    }
}

/**
 *  @brief Compares two multi-state SHA256 backends over batches of randomly-shaped messages.
 *
 *  @param reference  Reference multi-state backend, producing the expected digests.
 *  @param candidate  Candidate multi-state backend to validate against the reference.
 *  @param inputs     Maximum lane count (inclusive) and maximum message length to fuzz with.
 *
 *  Sweeps every lane count up to @p inputs, so batches that fall one lane short of a vector width take a
 *  different path through the kernel than batches that fill it. Each batch is fed twice: once in a single
 *  call, then again split into random per-lane slices, which is what carries a partial block across calls.
 *  Both digest buffers keep a guard lane past the end, since a batched kernel that miscounts lanes would
 *  otherwise corrupt the caller's memory silently.
 */
template <typename reference_, typename candidate_>
void test_sha256_multistate_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    for (sz_size_t lanes_count = 0; lanes_count <= inputs; ++lanes_count) {
        std::vector<std::string> messages;
        fuzzy_config_t config;
        config.batch_size = (std::size_t)lanes_count;
        config.min_string_length = 0;
        config.max_string_length = (std::size_t)inputs;
        randomize_strings(config, messages);

        std::vector<sz_sha256_state_t> reference_states(lanes_count ? lanes_count : 1);
        std::vector<sz_sha256_state_t> candidate_states(lanes_count ? lanes_count : 1);
        std::vector<sz_u8_t> reference_digests((lanes_count + 1) * SZ_SHA256_DIGEST_LENGTH, 0xA5);
        std::vector<sz_u8_t> candidate_digests((lanes_count + 1) * SZ_SHA256_DIGEST_LENGTH, 0xA5);
        for (std::size_t lane_index = 0; lane_index != messages.size(); ++lane_index)
            sz_sha256_state_init(&reference_states[lane_index]), sz_sha256_state_init(&candidate_states[lane_index]);

        // One-shot: every lane consumes its whole message in a single call
        sz_sequence_t const texts = sequence_from_(messages);
        reference.update(reference_states.data(), &texts);
        candidate.update(candidate_states.data(), &texts);
        reference.digest(reference_states.data(), lanes_count, reference_digests.data());
        candidate.digest(candidate_states.data(), lanes_count, candidate_digests.data());
        verify(std::memcmp(reference_digests.data(), candidate_digests.data(), lanes_count * SZ_SHA256_DIGEST_LENGTH) ==
               0);

        // Incremental: the same messages, cut into random per-lane slices across several calls
        for (std::size_t lane_index = 0; lane_index != messages.size(); ++lane_index)
            sz_sha256_state_init(&reference_states[lane_index]), sz_sha256_state_init(&candidate_states[lane_index]);
        // The slices are windows into the messages the caller already holds, not copies, so the kernels see
        // a cursor advancing through one stable buffer - the shape a real caller streams with.
        std::vector<std::size_t> offsets(messages.size(), 0);
        std::vector<sz_string_view_t> slices(messages.size());
        std::size_t remaining_lanes = messages.size();
        while (remaining_lanes != 0) {
            remaining_lanes = 0;
            for (std::size_t lane_index = 0; lane_index != messages.size(); ++lane_index) {
                std::size_t const left = messages[lane_index].size() - offsets[lane_index];
                std::uniform_int_distribution<std::size_t> slice_length_distribution(0, left);
                std::size_t const take = slice_length_distribution(global_random_generator());
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
        verify(std::memcmp(reference_digests.data(), candidate_digests.data(), lanes_count * SZ_SHA256_DIGEST_LENGTH) ==
               0);

        for (std::size_t guard_index = 0; guard_index != SZ_SHA256_DIGEST_LENGTH;
             ++guard_index) // No overwrite past the last lane
            verify(candidate_digests[lanes_count * SZ_SHA256_DIGEST_LENGTH + guard_index] == 0xA5);
    }
}

#pragma endregion // Equivalence

#pragma region Drivers

/**
 *  @brief Drives the serial-vs-SIMD hashing, random-fill, and SHA256 differential tests across
 *         every hashing backend compiled on this target. Hashing has no Haswell tier.
 */
void test_hash_all() {

    using hash_serial_t = hash_from_sz_<sz_hash_serial, sz_hash_state_init_serial, //
                                        sz_hash_state_update_serial, sz_hash_state_digest_serial>;
    hash_serial_t const hash_serial;
    sz_unused_(hash_serial); // Used only by the SIMD differential blocks below; unreferenced on no-SIMD-tier targets.

    // Number of random-length inputs to fuzz per differential test. Each sweeps lengths `0..N` and hashes a buffer
    // of that length, so the work is quadratic in the count and the baseline is scaled accordingly.
    sz_size_t const hash_inputs = (sz_size_t)scale_iterations_quadratic(200);
    sz_size_t const random_inputs = (sz_size_t)scale_iterations_quadratic(200);
    sz_size_t const sha256_inputs = (sz_size_t)scale_iterations_quadratic(256);
    sz_unused_(hash_inputs), sz_unused_(random_inputs), sz_unused_(sha256_inputs);

    // Ensure the seed affects hash results
    verify(sz_hash_serial("abc", 3, 100) != sz_hash_serial("abc", 3, 200));
    verify(sz_hash_serial("abcdefgh", 8, 0) != sz_hash_serial("abcdefgh", 8, 7));

    // Byte sums carry their own backend set - Haswell, NEON, SVE and the WASM tiers all provide one where
    // the AES-based hash does not - so they need a differential sweep of their own.
    using bytesum_serial_t = bytesum_from_sz_<sz_bytesum_serial>;
    bytesum_serial_t const bytesum_serial;
    sz_size_t const bytesum_inputs = (sz_size_t)scale_iterations_quadratic(200);
    sz_unused_(bytesum_serial), sz_unused_(bytesum_inputs);

#if SZ_USE_HASWELL
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_haswell> {}, bytesum_inputs);
#endif
#if SZ_USE_SKYLAKE
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_skylake> {}, bytesum_inputs);
#endif
#if SZ_USE_ICELAKE
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_icelake> {}, bytesum_inputs);
#endif
#if SZ_USE_NEON
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_neon> {}, bytesum_inputs);
#endif
#if SZ_USE_SVE
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_sve> {}, bytesum_inputs);
#endif
#if SZ_USE_SVE2
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_sve2> {}, bytesum_inputs);
#endif
#if SZ_USE_V128
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_v128> {}, bytesum_inputs);
#endif
#if SZ_USE_V128RELAXED
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_v128relaxed> {}, bytesum_inputs);
#endif
#if SZ_USE_RVV
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_rvv> {}, bytesum_inputs);
#endif
#if SZ_USE_LASX
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_lasx> {}, bytesum_inputs);
#endif
#if SZ_USE_POWERVSX
    test_bytesum_equivalence(bytesum_serial, bytesum_from_sz_<sz_bytesum_powervsx> {}, bytesum_inputs);
#endif

#if SZ_USE_WESTMERE
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_westmere, sz_hash_state_init_westmere, //
                                        sz_hash_state_update_westmere, sz_hash_state_digest_westmere> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_westmere> {}, random_inputs);
#endif
#if SZ_USE_SKYLAKE
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_skylake, sz_hash_state_init_skylake, //
                                        sz_hash_state_update_skylake, sz_hash_state_digest_skylake> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_skylake> {}, random_inputs);
#endif
#if SZ_USE_ICELAKE
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_icelake, sz_hash_state_init_icelake, //
                                        sz_hash_state_update_icelake, sz_hash_state_digest_icelake> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_icelake> {}, random_inputs);
#endif
#if SZ_USE_NEONAES
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_neonaes, sz_hash_state_init_neonaes, //
                                        sz_hash_state_update_neonaes, sz_hash_state_digest_neonaes> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_neonaes> {}, random_inputs);
#endif
#if SZ_USE_SVE2AES
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_sve2aes, sz_hash_state_init_sve2aes, //
                                        sz_hash_state_update_sve2aes, sz_hash_state_digest_sve2aes> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_sve2aes> {}, random_inputs);
#endif
#if SZ_USE_V128
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_v128, sz_hash_state_init_v128, //
                                        sz_hash_state_update_v128, sz_hash_state_digest_v128> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_v128> {}, random_inputs);
#endif
#if SZ_USE_V128RELAXED
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_v128relaxed, sz_hash_state_init_v128relaxed, //
                                        sz_hash_state_update_v128relaxed, sz_hash_state_digest_v128relaxed> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_v128relaxed> {}, random_inputs);
#endif
#if SZ_USE_RVV
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_rvv, sz_hash_state_init_rvv, //
                                        sz_hash_state_update_rvv, sz_hash_state_digest_rvv> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {}, fill_random_from_sz_<sz_fill_random_rvv> {},
                            random_inputs);
#endif
#if SZ_USE_LASX
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_lasx, sz_hash_state_init_lasx, //
                                        sz_hash_state_update_lasx, sz_hash_state_digest_lasx> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_lasx> {}, random_inputs);
#endif
#if SZ_USE_POWERVSX
    test_hash_equivalence(hash_serial,
                          hash_from_sz_<sz_hash_powervsx, sz_hash_state_init_powervsx, //
                                        sz_hash_state_update_powervsx, sz_hash_state_digest_powervsx> {},
                          hash_inputs);
    test_random_equivalence(fill_random_from_sz_<sz_fill_random_serial> {},
                            fill_random_from_sz_<sz_fill_random_powervsx> {}, random_inputs);
#endif

    // Test SHA256 implementations
    using sha256_serial_t =
        sha256_from_sz_<sz_sha256_state_init_serial, sz_sha256_state_update_serial, sz_sha256_state_digest_serial>;
    sha256_serial_t const sha256_serial;
    sz_unused_(sha256_serial);

#if SZ_USE_GOLDMONT
    test_sha256_equivalence(sha256_serial,
                            sha256_from_sz_<sz_sha256_state_init_goldmont, sz_sha256_state_update_goldmont,
                                            sz_sha256_state_digest_goldmont> {},
                            sha256_inputs);
#endif
#if SZ_USE_NEONSHA
    test_sha256_equivalence(sha256_serial,
                            sha256_from_sz_<sz_sha256_state_init_neonsha, sz_sha256_state_update_neonsha,
                                            sz_sha256_state_digest_neonsha> {},
                            sha256_inputs);
#endif
#if SZ_USE_V128
    test_sha256_equivalence(
        sha256_serial,
        sha256_from_sz_<sz_sha256_state_init_v128, sz_sha256_state_update_v128, sz_sha256_state_digest_v128> {},
        sha256_inputs);
#endif
#if SZ_USE_RVV
    test_sha256_equivalence(
        sha256_serial,
        sha256_from_sz_<sz_sha256_state_init_rvv, sz_sha256_state_update_rvv, sz_sha256_state_digest_rvv> {},
        sha256_inputs);
#endif
#if SZ_USE_LASX
    test_sha256_equivalence(
        sha256_serial,
        sha256_from_sz_<sz_sha256_state_init_lasx, sz_sha256_state_update_lasx, sz_sha256_state_digest_lasx> {},
        sha256_inputs);
#endif
#if SZ_USE_POWERVSX
    test_sha256_equivalence(sha256_serial,
                            sha256_from_sz_<sz_sha256_state_init_powervsx, sz_sha256_state_update_powervsx,
                                            sz_sha256_state_digest_powervsx> {},
                            sha256_inputs);
#endif

    // The multi-state kernels sweep every lane count up to the bound, and each batch carries one message per
    // lane, so the work is quadratic in the bound just like the single-message sweep above.
    sz_size_t const multistate_inputs = (sz_size_t)scale_iterations_quadratic(64);
    using multistate_serial_t =
        sha256_multistate_from_sz_<sz_sha256_multistate_update_serial, sz_sha256_multistate_digest_serial>;
    multistate_serial_t const multistate_serial;
    sz_unused_(multistate_serial), sz_unused_(multistate_inputs);

#if SZ_USE_GOLDMONT
    test_sha256_multistate_equivalence(
        multistate_serial,
        sha256_multistate_from_sz_<sz_sha256_multistate_update_goldmont, sz_sha256_multistate_digest_goldmont> {},
        multistate_inputs);
#endif
#if SZ_USE_HASWELL
    test_sha256_multistate_equivalence(
        multistate_serial,
        sha256_multistate_from_sz_<sz_sha256_multistate_update_haswell, sz_sha256_multistate_digest_haswell> {},
        multistate_inputs);
#endif
#if SZ_USE_SKYLAKE
    test_sha256_multistate_equivalence(
        multistate_serial,
        sha256_multistate_from_sz_<sz_sha256_multistate_update_skylake, sz_sha256_multistate_digest_skylake> {},
        multistate_inputs);
#endif
}

/** @brief Drives `test_hash_multiseed_equivalence` across every hashing backend compiled on this target. */
void test_hash_multiseed_all() {
    // Cover the <= 64 byte ladder, the 64-byte boundary, and into the wide path. Every length is hashed by
    // every seeded kernel on every backend, so this count is the family's whole budget.
    sz_size_t const lengths = (sz_size_t)scale_iterations_quadratic(512);

    test_hash_multiseed_equivalence(hash_multiseed_from_sz_<sz_hash_multiseed, sz_hash> {}, lengths);

    // And every backend that ships a specialized multi-seed kernel must match its own single-shot.
    test_hash_multiseed_equivalence(hash_multiseed_from_sz_<sz_hash_multiseed_serial, sz_hash_serial> {}, lengths);
#if SZ_USE_WESTMERE
    test_hash_multiseed_equivalence(hash_multiseed_from_sz_<sz_hash_multiseed_westmere, sz_hash_westmere> {}, lengths);
#endif
#if SZ_USE_ICELAKE
    test_hash_multiseed_equivalence(hash_multiseed_from_sz_<sz_hash_multiseed_icelake, sz_hash_icelake> {}, lengths);
#endif
#if SZ_USE_NEONAES
    test_hash_multiseed_equivalence(hash_multiseed_from_sz_<sz_hash_multiseed_neonaes, sz_hash_neonaes> {}, lengths);
#endif
#if SZ_USE_V128
    test_hash_multiseed_equivalence(hash_multiseed_from_sz_<sz_hash_multiseed_v128, sz_hash_v128> {}, lengths);
#endif
#if SZ_USE_V128RELAXED
    test_hash_multiseed_equivalence(hash_multiseed_from_sz_<sz_hash_multiseed_v128relaxed, sz_hash_v128relaxed> {},
                                    lengths);
#endif
}

#pragma endregion // Drivers
