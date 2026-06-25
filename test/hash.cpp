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

#include <cassert> // C-style assertions
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
static void sha256_digest_from_hex_(char const *hex, sz_u8_t (&digest)[32]) {
    auto nibble = [](char character) -> sz_u8_t {
        if (character >= '0' && character <= '9') return (sz_u8_t)(character - '0');
        return (sz_u8_t)(character - 'a' + 10);
    };
    for (std::size_t byte_index = 0; byte_index != 32; ++byte_index)
        digest[byte_index] = (sz_u8_t)((nibble(hex[byte_index * 2]) << 4) | nibble(hex[byte_index * 2 + 1]));
}

/** @brief Runs one SHA256 backend (init/update/digest) over `message` and asserts the expected digest. */
static void check_sha256_unit_(                                   //
    sz_sha256_state_init_t init, sz_sha256_state_update_t update, //
    sz_sha256_state_digest_t digest, std::string const &message, char const *expected_hex) {
    sz_sha256_state_t state;
    sz_u8_t produced[32], expected[32];
    sha256_digest_from_hex_(expected_hex, expected);
    init(&state);
    update(&state, message.data(), (sz_size_t)message.size());
    digest(&state, produced);
    assert(std::memcmp(produced, expected, 32) == 0);
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
    struct known_sha256_t {
        char const *message;
        char const *digest_hex;
    };
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
#if SZ_USE_ICELAKE
        check_sha256_unit_(sz_sha256_state_init_icelake, sz_sha256_state_update_icelake, // Manual: icelake kernel
                           sz_sha256_state_digest_icelake, vector.message, vector.digest_hex);
#endif
    }

    // An embedded-NUL message must hash past the NUL: `"abc\x00def"` is 7 bytes, not 3. Construct the
    // `std::string` with an explicit length so the interior NUL is retained.
    std::string const embedded_nul("abc\x00" "def", 7);
    assert(embedded_nul.size() == 7);
    check_sha256_unit_(sz_sha256_state_init, sz_sha256_state_update, // Dispatched (automatic kernel)
                       sz_sha256_state_digest, embedded_nul,
                       "516a5e926ce20c5f4d80f00e1a01abdf14986def6588d6abeed9fce090bc660c");
    check_sha256_unit_(sz_sha256_state_init_serial, sz_sha256_state_update_serial, // Manual: serial kernel
                       sz_sha256_state_digest_serial, embedded_nul,
                       "516a5e926ce20c5f4d80f00e1a01abdf14986def6588d6abeed9fce090bc660c");

    // `sz_bytesum` is an order-independent byte sum, so "abc" sums to 0x61 + 0x62 + 0x63 = 0x126.
    let_assert(auto bytesum_abc = sz_bytesum("abc", 3), bytesum_abc == 0x126u); // Dispatched (automatic kernel)
    assert(sz_bytesum_serial("abc", 3) == 0x126u); // Manual propagation to the serial kernel
#if SZ_USE_ICELAKE
    assert(sz_bytesum_icelake("abc", 3) == 0x126u);
#endif

    // The byte sum must include the interior NUL byte: "abc\x00def" sums to 0x255.
    let_assert(auto bytesum_nul = sz_bytesum(embedded_nul.data(), embedded_nul.size()), bytesum_nul == 0x255u);
    assert(sz_bytesum_serial(embedded_nul.data(), embedded_nul.size()) == 0x255u);

    // `sz_hash` is deterministic; the dispatched result equals the serial kernel and the C++ wrapper,
    // and a different seed must change the digest.
    char const *fox = "The quick brown fox";
    sz_size_t const fox_length = (sz_size_t)std::strlen(fox);
    let_assert(auto hash_fox = sz_hash(fox, fox_length, 0u), hash_fox == sz_hash(fox, fox_length, 0u)); // Deterministic
    assert(sz_hash(fox, fox_length, 0u) == sz_hash_serial(fox, fox_length, 0u));     // Dispatch == serial
    assert(sz_hash(fox, fox_length, 0u) != sz_hash(fox, fox_length, 1u));            // Seed changes output
    assert(sz::string_view(fox, fox_length).hash() == sz_hash(fox, fox_length, 0u)); // C++ wrapper

    // The hash must also read past an interior NUL, so truncating at the NUL changes the digest.
    let_assert(auto hash_nul = sz_hash(embedded_nul.data(), embedded_nul.size(), 0u),
               hash_nul == sz_hash_serial(embedded_nul.data(), embedded_nul.size(), 0u)); // Dispatch == serial
    assert(sz_hash(embedded_nul.data(), embedded_nul.size(), 0u) != sz_hash(embedded_nul.data(), 3u, 0u));
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

/** @brief Wraps a pseudo-random fill backend by its kernel pointer. */
template <sz_fill_random_t generate_>
struct fill_random_from_sz_ {
    void operator()(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) const noexcept { generate_(text, length, nonce); }
};

/**
 *  @brief Hashes a string and compares the output between a reference and a candidate hashing backend.
 *
 *  The test covers increasingly long and complex strings, starting with "abcabc..." repetitions and
 *  progressing towards corner cases like empty strings, all-zero inputs, zero seeds, and so on.
 *
 *  @param reference  Reference hashing backend wrapper (one-shot + streaming).
 *  @param candidate  Candidate hashing backend wrapper to validate against the reference.
 *  @param inputs     Number of random-length inputs to fuzz beyond the structured "abc" repetitions.
 */
template <typename reference_, typename candidate_>
void test_hash_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    auto test_on_seed = [&](std::string text, sz_u64_t seed) {
        // Compute the entire hash at once, expecting the same output
        sz_u64_t result_base = reference(text.data(), text.size(), seed);
        sz_u64_t result_simd = candidate(text.data(), text.size(), seed);
        assert(result_base == result_simd);

        // Compare incremental hashing across platforms
        sz_hash_state_t state_base, state_simd;
        reference.init(&state_base, seed);
        candidate.init(&state_simd, seed);
        assert(sz_hash_state_equal(&state_base, &state_base) == sz_true_k); // Self-equality
        assert(sz_hash_state_equal(&state_simd, &state_simd) == sz_true_k); // Self-equality
        assert(sz_hash_state_equal(&state_base, &state_simd) == sz_true_k); // Same across platforms

        // Let's also create an intentionally misaligned version of the state,
        // assuming some of the SIMD instructions may require alignment.
        sz_align_(64) char state_misaligned_buffer[sizeof(sz_hash_state_t) + 1];
        sz_hash_state_t &state_misaligned = *reinterpret_cast<sz_hash_state_t *>(state_misaligned_buffer + 1);
        candidate.init(&state_misaligned, seed);
        assert(sz_hash_state_equal(&state_base, &state_misaligned) == sz_true_k); // Same across platforms

        // Try breaking those strings into arbitrary chunks, expecting the same output in the streaming mode.
        // The length of each chunk and the number of chunks will be determined with a coin toss.
        iterate_in_random_slices(text, [&](std::string slice) {
            reference.update(&state_base, slice.data(), slice.size());
            candidate.update(&state_simd, slice.data(), slice.size());
            assert(sz_hash_state_equal(&state_base, &state_simd) == sz_true_k); // Same across platforms

            candidate.update(&state_misaligned, slice.data(), slice.size());
            assert(sz_hash_state_equal(&state_base, &state_misaligned) == sz_true_k); // Same across platforms

            result_base = reference.digest(&state_base);
            result_simd = candidate.digest(&state_simd);
            assert(result_base == result_simd);
            sz_u64_t result_misaligned = candidate.digest(&state_misaligned);
            assert(result_base == result_misaligned);
        });
    };

    // Let's try different-length strings repeating a "abc" pattern:
    std::vector<sz_u64_t> seeds = {
        0u,
        42u,                                  //
        std::numeric_limits<sz_u32_t>::max(), //
        std::numeric_limits<sz_u64_t>::max(), //
    };
    for (auto seed : seeds)
        for (std::size_t copies = 1; copies != scale_iterations(100); ++copies) //
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
 *
 *  @param candidate  Candidate multi-seed backend wrapper (batch + single-seed).
 *  @param inputs     Number of random-length inputs to fuzz beyond the structured length ladder.
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
                assert(output[index] == candidate.hash_one(text.data(), text.size(), seeds[index]));
            assert(output[seed_count] == 0xDEADBEEFDEADBEEFull); // No overwrite past `seed_count`
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
 *
 *  @param reference  Reference random-fill backend wrapper.
 *  @param candidate  Candidate random-fill backend wrapper to validate against the reference.
 *  @param inputs     Number of random-length inputs to fuzz beyond the structured length ladder.
 */
template <typename reference_, typename candidate_>
void test_random_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    auto test_on_nonce = [&](std::size_t length, sz_u64_t nonce) {
        std::string text_base(length, '\0');
        std::string text_simd(length, '\0');
        reference(&text_base[0], static_cast<sz_size_t>(length), nonce);
        candidate(&text_simd[0], static_cast<sz_size_t>(length), nonce);
        assert(text_base == text_simd);
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

    // Beyond the structured ladder, fuzz a contiguous run of random lengths scaled by `inputs`,
    // so `SZ_TESTS_MULTIPLIER` widens the swept range without dropping any baseline coverage.
    sz_size_t const fuzzed_lengths = (sz_size_t)scale_iterations(inputs);
    for (sz_size_t length = 0; length != fuzzed_lengths; ++length)
        for (auto nonce : nonces) //
            test_on_nonce(length, nonce);
}

/**
 *  @brief Cross-checks SHA256 backends against each other (reference vs candidate) on random inputs,
 *         one-shot and incremental. The known-answer FIPS 180-4 vectors live in `test_hash_unit`.
 *
 *  @param reference  Reference SHA256 backend wrapper (init/update/digest).
 *  @param candidate  Candidate SHA256 backend wrapper to validate against the reference.
 *  @param inputs     Maximum input length (inclusive) to fuzz with random bytes.
 */
template <typename reference_, typename candidate_>
void test_sha256_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {

    // Test random inputs of various lengths
    for (sz_size_t length = 0; length <= inputs; ++length) {
        std::string random_text(length, '\0');
        randomize_string(&random_text[0], length);

        sz_sha256_state_t state_base, state_simd;
        sz_u8_t digest_base_result[32], digest_simd_result[32];

        // One-shot hashing
        reference.init(&state_base);
        candidate.init(&state_simd);
        reference.update(&state_base, random_text.data(), length);
        candidate.update(&state_simd, random_text.data(), length);
        reference.digest(&state_base, digest_base_result);
        candidate.digest(&state_simd, digest_simd_result);
        assert(std::memcmp(digest_base_result, digest_simd_result, 32) == 0);

        // Incremental hashing with random chunks
        reference.init(&state_base);
        candidate.init(&state_simd);
        iterate_in_random_slices(random_text, [&](std::string slice) {
            reference.update(&state_base, slice.data(), slice.size());
            candidate.update(&state_simd, slice.data(), slice.size());
        });
        reference.digest(&state_base, digest_base_result);
        candidate.digest(&state_simd, digest_simd_result);
        assert(std::memcmp(digest_base_result, digest_simd_result, 32) == 0);
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

    // Number of random-length inputs to fuzz per differential test, scaled by the global multiplier.
    sz_size_t const hash_inputs = (sz_size_t)scale_iterations(200);
    sz_size_t const random_inputs = (sz_size_t)scale_iterations(200);
    sz_size_t const sha256_inputs = (sz_size_t)scale_iterations(256);

    // Ensure the seed affects hash results
    assert(sz_hash_serial("abc", 3, 100) != sz_hash_serial("abc", 3, 200));
    assert(sz_hash_serial("abcdefgh", 8, 0) != sz_hash_serial("abcdefgh", 8, 7));

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

#if SZ_USE_ICELAKE
    test_sha256_equivalence(sha256_serial,
                            sha256_from_sz_<sz_sha256_state_init_icelake, sz_sha256_state_update_icelake,
                                            sz_sha256_state_digest_icelake> {},
                            sha256_inputs);
#endif
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
}

/** @brief Drives `test_hash_multiseed_equivalence` across every hashing backend compiled on this target. */
void test_hash_multiseed_all() {
    // Cover the <= 64 byte ladder, the 64-byte boundary, and a few kilobytes of the wide path.
    sz_size_t const lengths = (sz_size_t)scale_iterations(4096);

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
