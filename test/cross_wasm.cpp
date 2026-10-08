/**
 *  @file test/cross_wasm.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel cross-checks - WebAssembly family: V128, V128 Relaxed.
 */
#include "cross.hpp"

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_wasm(environment_t const &env) {
    [[maybe_unused]] cross_section_t check(env);

#if STRINGZILLA_TARGET_V128
    check.section("Cross V128", sz_cap_v128_k);

    check("test_compare_unit_v128", [] { check_compare_unit_({"v128", sz_equal_v128, sz_order_v128}); });

    constexpr memory_backend_t memory_v128 {"v128", sz_copy_v128, sz_move_v128, sz_fill_v128};
    check("test_memory_unit_v128", [&] { check_memory_unit_(memory_v128); });
    check("test_memory_equivalence_v128",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_v128); });
    check("test_memory_safety_v128", [&] { check_memory_safety_(memory_v128); });

    constexpr lookup_backend_t lookup_v128 {"v128", sz_lookup_v128};
    check("test_lookup_unit_v128", [&] { check_lookup_unit_(lookup_v128); });
    check("test_lookup_equivalence_v128",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_v128); });
    check("test_lookup_safety_v128", [&] { check_lookup_safety_(lookup_v128); });

    constexpr find_backend_t find_v128 {
        .name = "v128",
        .find = sz_find_v128,
        .rfind = sz_rfind_v128,
        .find_byte = sz_find_byte_v128,
        .rfind_byte = sz_rfind_byte_v128,
        .find_byteset = sz_find_byteset_v128,
        .rfind_byteset = sz_rfind_byteset_v128,
    };
    check("test_find_unit_v128", [&] { check_find_unit_(find_v128); });
    check("test_find_equivalence_v128", [&](test_context_t &context) { check_find_equivalence_(context, find_v128); });
    check("test_find_safety_v128", [&] { check_find_safety_(find_v128); });

    constexpr hash_backend_t hash_v128 {
        .hash_kernel = sz_hash_v128,
        .init_kernel = sz_hash_state_init_v128,
        .update_kernel = sz_hash_state_update_v128,
        .digest_kernel = sz_hash_state_digest_v128,
    };
    constexpr sha256_backend_t sha256_v128 {
        .init_kernel = sz_sha256_state_init_v128,
        .update_kernel = sz_sha256_state_update_v128,
        .digest_kernel = sz_sha256_state_digest_v128,
    };
    check("test_hash_equivalence_v128", [&](test_context_t &context) {
        check_bytesum_equivalence_(context, sz_bytesum_v128);
        check_hash_equivalence_(context, hash_v128);
        check_fill_random_equivalence_(context, sz_fill_random_v128);
        check_sha256_equivalence_(context, sha256_v128);
    });
    check("test_hash_multiseed_equivalence_v128", [](test_context_t &context) {
        check_hash_multiseed_equivalence_(context, {sz_hash_multiseed_v128, sz_hash_v128});
    });

    constexpr ctr_backend_t ctr_v128 {"v128", sz_aes256_key_init_v128, sz_aes256_ctr_xor_v128};
    constexpr gcm_backend_t gcm_v128 {
        .name = "v128",
        .key_init = sz_aes256_gcm_key_init_v128,
        .encrypt = sz_aes256_gcm_encrypt_v128,
        .decrypt = sz_aes256_gcm_decrypt_v128,
        .sealer_init = sz_aes256_gcm_encryptor_init_v128,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_v128,
        .sealer_update = sz_aes256_gcm_encryptor_update_v128,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_v128,
        .opener_init = sz_aes256_gcm_decryptor_init_v128,
        .opener_associate = sz_aes256_gcm_decryptor_associate_v128,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_v128,
        .opener_verify = sz_aes256_gcm_decryptor_verify_v128,
    };
    check("test_cipher_unit_v128", [&] { check_cipher_unit_(ctr_v128, gcm_v128); });
    check("test_cipher_equivalence_v128",
          [&](test_context_t &context) { check_cipher_equivalence_(context, ctr_v128, gcm_v128); });

    constexpr utf8_runes_backend_t utf8_runes_v128 {"v128", sz_utf8_count_v128, sz_utf8_seek_v128, sz_utf8_decode_v128};
    check("test_utf8_runes_unit_v128", [&] { check_utf8_runes_unit_(utf8_runes_v128); });
    check("test_utf8_runes_safety_v128",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_v128); });
    check("test_utf8_runes_equivalence_v128",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_v128); });

    constexpr utf8_tokens_backend_t utf8_tokens_v128 {"v128", sz_utf8_count_v128, sz_utf8_newlines_v128,
                                                      sz_utf8_whitespaces_v128};
    check("test_utf8_tokens_unit_v128", [&] { check_utf8_tokens_unit_(utf8_tokens_v128); });
    check("test_utf8_tokens_safety_v128",
          [&](test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_v128); });
    check("test_utf8_tokens_equivalence_v128",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_v128); });

    constexpr utf8_delimiters_backend_t utf8_delimiters_v128 {"v128", sz_utf8_delimiters_v128};
    check("test_utf8_delimiters_unit_v128", [&] { check_utf8_delimiters_unit_(utf8_delimiters_v128); });
    check("test_utf8_delimiters_safety_v128",
          [&](test_context_t &context) { check_utf8_delimiters_safety_(context, utf8_delimiters_v128); });
    check("test_utf8_delimiters_equivalence_v128",
          [&](test_context_t &context) { check_utf8_delimiters_equivalence_(context, utf8_delimiters_v128); });

    constexpr utf8_segment_backend_t utf8_wordbreaks_v128 {"v128", sz_utf8_wordbreaks_v128};
    check("test_utf8_wordbreaks_unit_v128", [&] { check_utf8_wordbreaks_unit_(utf8_wordbreaks_v128); });
    check("test_utf8_wordbreaks_rules_v128", [&] { check_utf8_wordbreaks_rules_(utf8_wordbreaks_v128); });
    check("test_utf8_wordbreaks_safety_v128",
          [&](test_context_t &context) { check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_v128); });
    check("test_utf8_wordbreaks_equivalence_v128",
          [&](test_context_t &context) { check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_v128); });

    constexpr utf8_norm_kernels_t utf8_norm_v128 {sz_utf8_norm_v128, sz_utf8_find_denormalized_v128};
    check("test_utf8_norm_unit_v128", [&] { check_utf8_norm_unit_(utf8_norm_v128); });
    check("test_utf8_norm_equivalence_v128",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_v128); });
    check("test_utf8_norm_safety_v128",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_v128); });

    constexpr utf8_uncased_kernels_t utf8_uncased_v128 {sz_utf8_uncased_fold_v128, sz_utf8_uncased_search_v128,
                                                        sz_utf8_uncased_order_v128, sz_utf8_find_cased_v128};
    check("test_utf8_uncased_unit_v128", [&] { check_utf8_uncased_unit_(utf8_uncased_v128); });
    check("test_utf8_uncased_equivalence_v128",
          [&](test_context_t &context) { check_utf8_uncased_equivalence_(context, utf8_uncased_v128); });
    check("test_utf8_uncased_safety_v128",
          [&](test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_v128); });
#endif // STRINGZILLA_TARGET_V128

#if STRINGZILLA_TARGET_V128RELAXED
    check.section("Cross V128 Relaxed", sz_cap_v128relaxed_k);

    check("test_compare_unit_v128relaxed",
          [] { check_compare_unit_({"v128relaxed", sz_equal_v128relaxed, sz_order_v128relaxed}); });

    constexpr memory_backend_t memory_v128relaxed {"v128relaxed", sz_copy_v128relaxed, sz_move_v128relaxed,
                                                   sz_fill_v128relaxed};
    check("test_memory_unit_v128relaxed", [&] { check_memory_unit_(memory_v128relaxed); });
    check("test_memory_equivalence_v128relaxed",
          [&](test_context_t &context) { check_memory_equivalence_(context, memory_v128relaxed); });
    check("test_memory_safety_v128relaxed", [&] { check_memory_safety_(memory_v128relaxed); });

    constexpr lookup_backend_t lookup_v128relaxed {"v128relaxed", sz_lookup_v128relaxed};
    check("test_lookup_unit_v128relaxed", [&] { check_lookup_unit_(lookup_v128relaxed); });
    check("test_lookup_equivalence_v128relaxed",
          [&](test_context_t &context) { check_lookup_equivalence_(context, lookup_v128relaxed); });
    check("test_lookup_safety_v128relaxed", [&] { check_lookup_safety_(lookup_v128relaxed); });

    constexpr find_backend_t find_v128relaxed {
        .name = "v128relaxed",
        .find = sz_find_v128relaxed,
        .rfind = sz_rfind_v128relaxed,
        .find_byte = sz_find_byte_v128relaxed,
        .rfind_byte = sz_rfind_byte_v128relaxed,
        .find_byteset = sz_find_byteset_v128relaxed,
        .rfind_byteset = sz_rfind_byteset_v128relaxed,
    };
    check("test_find_unit_v128relaxed", [&] { check_find_unit_(find_v128relaxed); });
    check("test_find_equivalence_v128relaxed",
          [&](test_context_t &context) { check_find_equivalence_(context, find_v128relaxed); });
    check("test_find_safety_v128relaxed", [&] { check_find_safety_(find_v128relaxed); });

    constexpr hash_backend_t hash_v128relaxed {
        .hash_kernel = sz_hash_v128relaxed,
        .init_kernel = sz_hash_state_init_v128relaxed,
        .update_kernel = sz_hash_state_update_v128relaxed,
        .digest_kernel = sz_hash_state_digest_v128relaxed,
    };
    check("test_hash_equivalence_v128relaxed", [&](test_context_t &context) {
        check_bytesum_equivalence_(context, sz_bytesum_v128relaxed);
        check_hash_equivalence_(context, hash_v128relaxed);
        check_fill_random_equivalence_(context, sz_fill_random_v128relaxed);
    });
    check("test_hash_multiseed_equivalence_v128relaxed", [](test_context_t &context) {
        check_hash_multiseed_equivalence_(context, {sz_hash_multiseed_v128relaxed, sz_hash_v128relaxed});
    });

    constexpr ctr_backend_t ctr_v128relaxed {"v128relaxed", sz_aes256_key_init_v128relaxed,
                                             sz_aes256_ctr_xor_v128relaxed};
    constexpr gcm_backend_t gcm_v128relaxed {
        .name = "v128relaxed",
        .key_init = sz_aes256_gcm_key_init_v128relaxed,
        .encrypt = sz_aes256_gcm_encrypt_v128relaxed,
        .decrypt = sz_aes256_gcm_decrypt_v128relaxed,
        .sealer_init = sz_aes256_gcm_encryptor_init_v128relaxed,
        .sealer_associate = sz_aes256_gcm_encryptor_associate_v128relaxed,
        .sealer_update = sz_aes256_gcm_encryptor_update_v128relaxed,
        .sealer_digest = sz_aes256_gcm_encryptor_digest_v128relaxed,
        .opener_init = sz_aes256_gcm_decryptor_init_v128relaxed,
        .opener_associate = sz_aes256_gcm_decryptor_associate_v128relaxed,
        .opener_update = sz_aes256_gcm_decryptor_update_unverified_v128relaxed,
        .opener_verify = sz_aes256_gcm_decryptor_verify_v128relaxed,
    };
    check("test_cipher_unit_v128relaxed", [&] { check_cipher_unit_(ctr_v128relaxed, gcm_v128relaxed); });
    check("test_cipher_equivalence_v128relaxed",
          [&](test_context_t &context) { check_cipher_equivalence_(context, ctr_v128relaxed, gcm_v128relaxed); });

    // The relaxed tier has its own counter and seeker but reuses the V128 decoder and segmenters.
    constexpr utf8_runes_backend_t utf8_runes_v128relaxed {"v128relaxed", sz_utf8_count_v128relaxed,
                                                           sz_utf8_seek_v128relaxed, STRINGZILLA_NULL};
    check("test_utf8_runes_unit_v128relaxed", [&] { check_utf8_runes_unit_(utf8_runes_v128relaxed); });
    check("test_utf8_runes_safety_v128relaxed",
          [&](test_context_t &context) { check_utf8_runes_safety_(context, utf8_runes_v128relaxed); });
    check("test_utf8_runes_equivalence_v128relaxed",
          [&](test_context_t &context) { check_utf8_runes_equivalence_(context, utf8_runes_v128relaxed); });

    constexpr utf8_tokens_backend_t utf8_tokens_v128relaxed {"v128relaxed", sz_utf8_count_v128relaxed,
                                                             sz_utf8_newlines_v128, sz_utf8_whitespaces_v128};
    check("test_utf8_tokens_unit_v128relaxed", [&] { check_utf8_tokens_unit_(utf8_tokens_v128relaxed); });
    check("test_utf8_tokens_equivalence_v128relaxed",
          [&](test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_v128relaxed); });

    constexpr utf8_norm_kernels_t utf8_norm_v128relaxed {sz_utf8_norm_v128relaxed,
                                                         sz_utf8_find_denormalized_v128relaxed};
    check("test_utf8_norm_unit_v128relaxed", [&] { check_utf8_norm_unit_(utf8_norm_v128relaxed); });
    check("test_utf8_norm_equivalence_v128relaxed",
          [&](test_context_t &context) { check_utf8_norm_equivalence_(context, utf8_norm_v128relaxed); });
    check("test_utf8_norm_safety_v128relaxed",
          [&](test_context_t &context) { check_utf8_norm_safety_(context, utf8_norm_v128relaxed); });
#endif // STRINGZILLA_TARGET_V128RELAXED

    return check.failures;
}

} // namespace ashvardanian::stringzilla::test
