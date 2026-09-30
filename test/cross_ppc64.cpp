/**
 *  @file test/cross_ppc64.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - Power ISA family, VSX.
 */
#include "cross.hpp"

using namespace ashvardanian::stringzilla::test;

std::size_t test_cross_ppc64(test_environment_t const &environment) {
    [[maybe_unused]] cross_section_t check(environment);

#if STRINGZILLA_TARGET_POWERVSX
    check.section("Cross Power VSX", sz_cap_powervsx_k);

    check("test_compare_unit_powervsx",
          [] { check_compare_unit_({"powervsx", sz_equal_powervsx, sz_order_powervsx}); });

    constexpr memory_backend_t memory_powervsx {"powervsx", sz_copy_powervsx, sz_move_powervsx, sz_fill_powervsx};
    check("test_memory_unit_powervsx", [&] { check_memory_unit_(memory_powervsx); });
    check("test_memory_equivalence_powervsx",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_powervsx); });
    check("test_memory_safety_powervsx", [&] { check_memory_safety_(memory_powervsx); });

    constexpr lookup_backend_t lookup_powervsx {"powervsx", sz_lookup_powervsx};
    check("test_lookup_unit_powervsx", [&] { check_lookup_unit_(lookup_powervsx); });
    check("test_lookup_equivalence_powervsx",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_powervsx); });
    check("test_lookup_safety_powervsx", [&] { check_lookup_safety_(lookup_powervsx); });

    constexpr find_backend_t find_powervsx {
        .name = "powervsx",
        .find = sz_find_powervsx,
        .rfind = sz_rfind_powervsx,
        .find_byte = sz_find_byte_powervsx,
        .rfind_byte = sz_rfind_byte_powervsx,
        .find_byteset = sz_find_byteset_powervsx,
        .rfind_byteset = sz_rfind_byteset_powervsx,
    };
    check("test_find_unit_powervsx", [&] { check_find_unit_(find_powervsx); });
    check("test_find_equivalence_powervsx",
          [&](test_context_t &context) { check_find_equivalence_(context, find_powervsx); });
    check("test_find_safety_powervsx", [&] { check_find_safety_(find_powervsx); });

    constexpr hash_backend_t hash_powervsx {
        .hash_kernel = sz_hash_powervsx,
        .init_kernel = sz_hash_state_init_powervsx,
        .update_kernel = sz_hash_state_update_powervsx,
        .digest_kernel = sz_hash_state_digest_powervsx,
    };
    constexpr sha256_backend_t sha256_powervsx {
        .init_kernel = sz_sha256_state_init_powervsx,
        .update_kernel = sz_sha256_state_update_powervsx,
        .digest_kernel = sz_sha256_state_digest_powervsx,
    };
    check("test_hash_equivalence_powervsx", [&](test_context_t &context) {
        check_bytesum_equivalence_(context, sz_bytesum_powervsx);
        check_hash_equivalence_(context, hash_powervsx);
        check_fill_random_equivalence_(context, sz_fill_random_powervsx);
        check_sha256_equivalence_(context, sha256_powervsx);
    });

    constexpr ctr_backend_t ctr_powervsx {"powervsx", sz_aes256_key_init_powervsx, sz_aes256_ctr_xor_powervsx};
    constexpr gcm_backend_t gcm_powervsx {
        .name = "powervsx",
        .key_init = sz_aes256_gcm_key_init_powervsx,
        .encrypt = sz_aes256_gcm_encrypt_powervsx,
        .decrypt = sz_aes256_gcm_decrypt_powervsx,
        .sealer_init = sz_aes256_gcm_encryptor_init_powervsx,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_powervsx,
        .sealer_update = sz_aes256_gcm_encryptor_update_powervsx,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_powervsx,
        .opener_init = sz_aes256_gcm_decryptor_init_powervsx,
        .opener_associate = sz_aes256_gcm_decryptor_associate_powervsx,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_powervsx,
        .opener_verify = sz_aes256_gcm_decryptor_verify_powervsx,
    };
    check("test_cipher_unit_powervsx", [&] { check_cipher_unit_(ctr_powervsx, gcm_powervsx); });
    check("test_cipher_equivalence_powervsx",
          [&](test_context_t &context) { check_cipher_equivalence_(context, ctr_powervsx, gcm_powervsx); });

    constexpr utf8_runes_backend_t utf8_runes_powervsx {"powervsx", sz_utf8_count_powervsx, sz_utf8_seek_powervsx,
                                                        sz_utf8_decode_powervsx};
    check("test_utf8_runes_unit_powervsx", [&] { check_utf8_runes_unit_(utf8_runes_powervsx); });
    check("test_utf8_runes_safety_powervsx",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_powervsx); });
    check("test_utf8_runes_equivalence_powervsx",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_powervsx); });

    constexpr utf8_tokens_backend_t utf8_tokens_powervsx {"powervsx", sz_utf8_count_powervsx, sz_utf8_newlines_powervsx,
                                                          sz_utf8_whitespaces_powervsx};
    check("test_utf8_tokens_unit_powervsx", [&] { check_utf8_tokens_unit_(utf8_tokens_powervsx); });
    check("test_utf8_tokens_safety_powervsx",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_powervsx); });
    check("test_utf8_tokens_equivalence_powervsx",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_powervsx); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_powervsx {"powervsx", sz_utf8_wordbreaks_powervsx};
    check("test_utf8_wordbreaks_unit_powervsx", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_powervsx); });
    check("test_utf8_wordbreaks_rules_powervsx", [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_powervsx); });
    check("test_utf8_wordbreaks_safety_powervsx",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_powervsx); });
    check("test_utf8_wordbreaks_equivalence_powervsx",
          [&](test_context_t &context) { check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_powervsx); });

    constexpr utf8_norm_kernels_t utf8_norm_powervsx {sz_utf8_norm_powervsx, sz_utf8_find_denormalized_powervsx};
    check("test_utf8_norm_unit_powervsx", [&] { check_utf8_norm_unit_(utf8_norm_powervsx); });
    check("test_utf8_norm_equivalence_powervsx",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_powervsx); });
    check("test_utf8_norm_safety_powervsx",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_powervsx); });

    constexpr utf8_uncased_kernels_t utf8_uncased_powervsx {
        sz_utf8_uncased_fold_powervsx, sz_utf8_uncased_search_powervsx, sz_utf8_uncased_order_powervsx,
        sz_utf8_find_cased_powervsx};
    check("test_utf8_uncased_unit_powervsx", [&] { check_utf8_uncased_unit_(utf8_uncased_powervsx); });
    check("test_utf8_uncased_equivalence_powervsx",
          [&](test_context_t &context) { check_utf8_uncased_equivalence_(context, utf8_uncased_powervsx); });
    check("test_utf8_uncased_safety_powervsx",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_powervsx); });
#endif // STRINGZILLA_TARGET_POWERVSX

    return check.failures;
}
