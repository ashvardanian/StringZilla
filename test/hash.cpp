/**
 *  @file test/hash.cpp
 *  @author Ash Vardanian
 *  @date February 26, 2025
 *  @brief Hashing, multi-seed hashing, random-generator, and SHA256 equivalence tests.
 */
#undef NDEBUG // ! Enable all assertions for testing

/** The Visual C++ run-time library detects incorrect iterator use, and asserts and displays a
 *  dialog box at run time on Windows. */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Include the StringZilla headers before anything else, to intercept missing @c #include
 *  directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <cstring> // `std::strlen`

#include <string> // `std::string` baseline

#include "cross.hpp"   // `hash_backend_t`, `check_hash_equivalence_`, `check_sha256_unit_`
#include "harness.hpp" // `randomize_string`, `test_context_t`

namespace ashvardanian::stringzilla::test {

#pragma region Helpers

/** The dispatched one-shot and streaming hashes, in the shape of their capability kernels. */
static hash_backend_t const hash_dispatched {cpu_best<sz_hash_best>, cpu_best<sz_hash_state_init_best>,
                                             cpu_best<sz_hash_state_update_best>, cpu_best<sz_hash_state_digest_best>};

/** The dispatched SHA256 init, update and digest. */
static sha256_backend_t const sha256_dispatched {
    cpu_best<sz_sha256_state_init_best>, cpu_best<sz_sha256_state_update_best>, cpu_best<sz_sha256_state_digest_best>};

/** The dispatched multi-state SHA256 update and digest. */
static sha256_multistate_backend_t const sha256_multistate_dispatched {cpu_best<sz_sha256_multistate_update_best>,
                                                                       cpu_best<sz_sha256_multistate_digest_best>};

#pragma endregion Helpers

#pragma region Unit

/**
 *  @brief Known-answer unit tests for the hashing family on simple, hand-verifiable inputs.
 *
 *  Exercises each function through the dispatched C API with automatic kernel resolution and
 *  through the C++ @c sz::string_view_t wrappers, so a regression that the serial-vs-SIMD agreement
 *  tests would miss - because both share a wrong constant - is still caught against an external
 *  ground truth of known answers.
 */
void test_hash_unit() {
    // SHA256: the three FIPS 180-4 vectors, alone and as one batch, and an interior-NUL message.
    check_sha256_unit_(sha256_dispatched);
    check_sha256_multistate_unit_(sha256_multistate_dispatched);

    // `sz_bytesum_best` is an order-independent sum, so "abc" sums to 0x61 + 0x62 + 0x63 = 0x126.
    check_bytesum_unit_(cpu_best<sz_bytesum_best>);

    // `sz_hash_best` is deterministic; the dispatched result equals the serial kernel and the C++
    // wrapper, and a different seed must change the digest.
    char const *fox = "The quick brown fox";
    sz_size_t const fox_length = (sz_size_t)std::strlen(fox);
    sz_u64_t const fox_hash = kernel_result<sz_u64_t>(cpu_best<sz_hash_best>, fox, fox_length, 0u);
    verify(kernel_result<sz_u64_t>(cpu_best<sz_hash_best>, fox, fox_length, 0u) == fox_hash); // Deterministic
    verify(kernel_result<sz_u64_t>(sz_hash_serial, fox, fox_length, 0u) == fox_hash);         // Dispatch == serial
    verify(kernel_result<sz_u64_t>(cpu_best<sz_hash_best>, fox, fox_length, 1u) != fox_hash); // Seed changes output
    verify(sz::string_view_t(fox, fox_length).hash() == fox_hash);                            // C++ wrapper
    check_hash_unit_(cpu_best<sz_hash_best>);

    // The hash must also read past an interior NUL, and agree with serial when it does.
    std::string const embedded_nul(interior_nul_, 7);
    verify(kernel_result<sz_u64_t>(cpu_best<sz_hash_best>, embedded_nul.data(), embedded_nul.size(), 0u) ==
           kernel_result<sz_u64_t>(sz_hash_serial, embedded_nul.data(), embedded_nul.size(), 0u));
}

#pragma endregion Unit

#pragma region Safety

/**
 *  @brief Degenerate lengths and alignments for the hashing family, asserting bounds, not digests.
 *
 *  A hash of nothing still has to be a hash: the empty input, the single byte and the streaming
 *  state fed in one-byte pieces all have to agree with the one-shot call over the same bytes, and
 *  none may write past the digest it was handed. The canary-guarded buffer is what catches the last
 *  of those, since a digest that overruns by one byte produces a perfectly plausible value.
 */
void test_hash_safety() {
    // The empty input is hashable, and its digest is stable across calls.
    verify(kernel_result<sz_u64_t>(cpu_best<sz_hash_best>, "", 0, 0) ==
           kernel_result<sz_u64_t>(cpu_best<sz_hash_best>, "", 0, 0));
    verify(kernel_result<sz_u64_t>(cpu_best<sz_bytesum_best>, "", 0) == 0);

    // Streaming in one-byte pieces must reach the same digest as one shot over the whole message.
    char const *message = "the quick brown fox jumps over the lazy dog";
    sz_size_t const message_length = (sz_size_t)std::strlen(message);
    {
        sz_hash_state_t streamed;
        verify(sz_hash_state_init_best(&streamed, 0, sz::default_capabilities(), nullptr) == sz_success_k);
        for (sz_size_t index = 0; index != message_length; ++index)
            verify(sz_hash_state_update_best(&streamed, message + index, 1, sz::default_capabilities(), nullptr) ==
                   sz_success_k);
        verify(kernel_result<sz_u64_t>(cpu_best<sz_hash_state_digest_best>, &streamed) ==
               kernel_result<sz_u64_t>(cpu_best<sz_hash_best>, message, message_length, 0));
    }

    // SHA256 into a canary-guarded destination, so a digest that writes one byte too many is caught.
    with_guarded_buffer_(STRINGZILLA_SHA256_DIGEST_LENGTH, [&](sz_ptr_t destination, std::size_t) {
        sz_sha256_state_t state;
        verify(sz_sha256_state_init_best(&state, sz::default_capabilities(), nullptr) == sz_success_k);
        verify(sz_sha256_state_update_best(&state, message, message_length, sz::default_capabilities(), nullptr) ==
               sz_success_k);
        verify(sz_sha256_state_digest_best(&state, (sz_u8_t *)destination, sz::default_capabilities(), nullptr) ==
               sz_success_k);
    });

    // Every length across the ladder, at every sub-cache-line alignment: the dispatched and serial kernels
    // must agree byte for byte, whatever the buffer's offset.
    sz_size_t const lengths[] = {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128};
    for (sz_size_t length : span_over(lengths)) {
        for_each_cacheline_offset_((std::size_t)length, [&](sz_ptr_t buffer, std::size_t) {
            for (sz_size_t index = 0; index != length; ++index) buffer[index] = (char)('a' + (index & 15));
            verify(kernel_result<sz_u64_t>(cpu_best<sz_hash_best>, buffer, length, 0) ==
                       kernel_result<sz_u64_t>(sz_hash_serial, buffer, length, 0) &&
                   "Dispatched hash disagreed with the serial kernel at this length and alignment");
            verify(kernel_result<sz_u64_t>(cpu_best<sz_bytesum_best>, buffer, length) ==
                       kernel_result<sz_u64_t>(sz_bytesum_serial, buffer, length) &&
                   "Dispatched byte sum disagreed with the serial kernel at this length and alignment");
        });
    }
}

#pragma endregion Safety

#pragma region Drivers

/** Drives the serial-versus-dispatched hashing, random-fill, byte-sum and SHA256 differentials. */
void test_hash_all(test_context_t &context) {
    check_hash_equivalence_(context, hash_dispatched);
    check_fill_random_equivalence_(context, cpu_best<sz_fill_random_best>);
    check_bytesum_equivalence_(context, cpu_best<sz_bytesum_best>);
    check_sha256_equivalence_(context, sha256_dispatched);
    check_sha256_multistate_equivalence_(context, sha256_multistate_dispatched);
}

/** Holds the dispatched multi-seed hash to its own single-seed reduction. */
void test_hash_multiseed_all(test_context_t &context) {
    check_hash_multiseed_equivalence_(context, {cpu_best<sz_hash_multiseed_best>, cpu_best<sz_hash_best>});
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
