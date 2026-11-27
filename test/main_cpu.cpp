/**
 *  @file test/main_cpu.cpp
 *  @author Ash Vardanian
 *  @date December 21, 2023
 *  @brief CPU test entry point: registers the family suites over the dispatch points and the kernel
 *      cross-checks of every architecture.
 *
 *  Three executables share it. @b stringzilla_test links the static library and cross-checks
 *  every kernel the library defines, @b stringzilla_shared_test links the shared library with
 *  every capability off, so only the dispatch points run, and @b stringzilla_cpu_header_test
 *  compiles the kernels inline, where the dispatch points are stubs, running the cross-checks and
 *  the stubs alone. The header-only one reaches the library only through the C ABI and the
 *  containers, as the C++ SDK links the library.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#undef STRINGZILLA_DEBUG
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <map>         // `std::map`
#include <string>      // `std::string` baseline
#include <string_view> // `std::string_view` baseline
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include "stringzilla/stringzilla.h" // Primary C API
#include "stringzilla/string.hpp"    // `sz::string_view_t`, `sz::string`, the containers header-only runners keep

#include "harness.hpp" // `start_test_run`, `run_test_case`, `finish_test_run`

using namespace ashvardanian::stringzilla::test;

/*  Instantiate all the templates to make the symbols visible and also check for weird compilation
 *  errors on uncommon paths. */
template class std::basic_string_view<char>;
template class sz::string_slice<char>;
template class std::basic_string<char>;
template class sz::string<>;
template class sz::string_slice<char const>;

template class std::vector<sz::string_view_t>;
template class std::map<sz::string_view_t, int>;

namespace ashvardanian::stringzilla::test {

/** The header-only stub of @p best_, called with value-initialized operands, which it never reads,
 *  reports the missing library. */
template <auto best_>
void test_missing_library() {
    sz_status_t const status = []<typename... arguments_types_>(sz_status_t (*best)(arguments_types_...)) {
        return best(arguments_types_ {}...);
    }(best_);
    expect_equal(status, sz_missing_library_k, "a header-only dispatch point ran a kernel");
}

/** Each family's finder picks what @c sz_find_kernel_punned picks for the family's kinds and misses
 *  every other kind, while a header-only build reports the missing library from all of them. */
void test_find_kernel_unit() {
    sz_status_t (*const find_kernels[])(sz_kernel_kind_t, sz_capability_t, sz_kernel_punned_t *, sz_capability_t *) = {
        sz_compare_find_kernel,
        sz_strings_find_kernel,
        sz_find_find_kernel,
        sz_hash_find_kernel,
        sz_cipher_find_kernel,
        sz_sort_find_kernel,
        sz_intersect_find_kernel,
        sz_utf8_runes_find_kernel,
        sz_utf8_tokens_find_kernel,
        sz_utf8_wordbreaks_find_kernel,
        sz_utf8_graphemes_find_kernel,
        sz_utf8_sentences_find_kernel,
        sz_utf8_linebreaks_find_kernel,
        sz_utf8_norm_find_kernel,
        sz_utf8_uncased_fold_find_kernel,
        sz_utf8_uncased_find_kernel,
        sz_levenshtein_find_kernel,
        sz_ngrams_find_kernel,
        sz_substrings_find_kernel,
        sz_regex_find_kernel,
        sz_subdocuments_find_kernel,
    };
    sz_status_t const missing = STRINGZILLA_HEADER_ONLY ? sz_missing_library_k : sz_missing_kernel_k;
    sz_kernel_punned_t const poison = reinterpret_cast<sz_kernel_punned_t>(&test_find_kernel_unit);
    sz_capability_t enabled = 0;
    sz_capabilities_enabled_cpu(&enabled);
    for (int kind = sz_kernel_unknown_k; kind <= sz_kernel_subdocuments_find_k; ++kind) {
        sz_kernel_punned_t expected_kernel = poison;
        sz_capability_t expected_capability = sz_cap_serial_k;
        sz_status_t const expected = sz_find_kernel_punned(static_cast<sz_kernel_kind_t>(kind), enabled,
                                                           &expected_kernel, &expected_capability);
        expect_true(expected == sz_success_k || (expected == missing && !expected_kernel && !expected_capability));
        std::size_t owners = 0;
        for (auto find_kernel : find_kernels) {
            sz_kernel_punned_t kernel = poison;
            sz_capability_t capability = sz_cap_serial_k;
            sz_status_t const status = find_kernel(static_cast<sz_kernel_kind_t>(kind), enabled, &kernel, &capability);
            if (status == sz_success_k) expect_true(kernel == expected_kernel && capability == expected_capability);
            else expect_true(status == missing && !kernel && !capability);
            owners += status == sz_success_k;
        }
        expect_equal(owners, expected == sz_success_k ? 1u : 0u);
    }
}

/** The GPUs @c STRINGZILLA_DEVICES names, exiting when one is unavailable, or else the first of
 *  each kind present. */
std::vector<device_selection_t> selected_devices() {
    std::optional<std::vector<device_selection_t>> const &requested = test_run.settings.devices;
    std::vector<device_selection_t> devices;
    if constexpr (STRINGZILLA_HEADER_ONLY) {
        if (requested) {
            fmt::println(stderr, "This header-only executable accepts only CPU workloads");
            std::exit(1);
        }
        return devices;
    }
    if (requested) {
        for (device_selection_t const &selection : *requested) {
            sz_status_t const status = check_device(selection);
            if (status != sz_success_k) {
                fmt::println(stderr, "Device {}:{} is unavailable (status {})", device_name(selection.backend),
                             selection.ordinal, static_cast<int>(status));
                std::exit(1);
            }
            devices.push_back(selection);
        }
    }
    else {
        for (sz::device_kind_t kind :
             {sz::device_kind_t::cuda_k, sz::device_kind_t::rocm_k, sz::device_kind_t::metal_k})
            if (check_device({kind, 0}) == sz_success_k) devices.push_back({kind, 0});
    }
    return devices;
}

} // namespace ashvardanian::stringzilla::test

int main(int argc, char **argv) {
    start_test_run(argc, argv);
    std::vector<device_selection_t> const devices = selected_devices();

    run_test_case("test_find_kernel_unit", test_find_kernel_unit);

#if STRINGZILLA_HEADER_ONLY
    run_test_case("test_missing_library<sz_stream_synchronize_best>", test_missing_library<sz_stream_synchronize_best>);
    run_test_case("test_missing_library<sz_equal_best>", test_missing_library<sz_equal_best>);
    run_test_case("test_missing_library<sz_order_best>", test_missing_library<sz_order_best>);
    run_test_case("test_missing_library<sz_strings_to_tape_best>", test_missing_library<sz_strings_to_tape_best>);
    run_test_case("test_missing_library<sz_strings_to_rows_best>", test_missing_library<sz_strings_to_rows_best>);
    run_test_case("test_missing_library<sz_strings_escape_best>", test_missing_library<sz_strings_escape_best>);
    run_test_case("test_missing_library<sz_strings_unescape_best>", test_missing_library<sz_strings_unescape_best>);
    run_test_case("test_missing_library<sz_strings_lookup_best>", test_missing_library<sz_strings_lookup_best>);
    run_test_case("test_missing_library<sz_strings_normalize_best>", test_missing_library<sz_strings_normalize_best>);
    run_test_case("test_missing_library<sz_strings_split_best>", test_missing_library<sz_strings_split_best>);
    run_test_case("test_missing_library<sz_strings_split_lines_best>",
                  test_missing_library<sz_strings_split_lines_best>);
    run_test_case("test_missing_library<sz_strings_join_best>", test_missing_library<sz_strings_join_best>);
    run_test_case("test_missing_library<sz_allocator_init_unified_best>",
                  test_missing_library<sz_allocator_init_unified_best>);
    run_test_case("test_missing_library<sz_allocator_init_device_best>",
                  test_missing_library<sz_allocator_init_device_best>);
    run_test_case("test_missing_library<sz_allocator_init_pinned_best>",
                  test_missing_library<sz_allocator_init_pinned_best>);
    run_test_case("test_missing_library<sz_find_byte_best>", test_missing_library<sz_find_byte_best>);
    run_test_case("test_missing_library<sz_rfind_byte_best>", test_missing_library<sz_rfind_byte_best>);
    run_test_case("test_missing_library<sz_find_best>", test_missing_library<sz_find_best>);
    run_test_case("test_missing_library<sz_rfind_best>", test_missing_library<sz_rfind_best>);
    run_test_case("test_missing_library<sz_find_byteset_best>", test_missing_library<sz_find_byteset_best>);
    run_test_case("test_missing_library<sz_rfind_byteset_best>", test_missing_library<sz_rfind_byteset_best>);
    run_test_case("test_missing_library<sz_find_strings_best>", test_missing_library<sz_find_strings_best>);
    run_test_case("test_missing_library<sz_bytesum_best>", test_missing_library<sz_bytesum_best>);
    run_test_case("test_missing_library<sz_hash_best>", test_missing_library<sz_hash_best>);
    run_test_case("test_missing_library<sz_hash_multiseed_best>", test_missing_library<sz_hash_multiseed_best>);
    run_test_case("test_missing_library<sz_hash_strings_best>", test_missing_library<sz_hash_strings_best>);
    run_test_case("test_missing_library<sz_hash_state_init_best>", test_missing_library<sz_hash_state_init_best>);
    run_test_case("test_missing_library<sz_hash_state_update_best>", test_missing_library<sz_hash_state_update_best>);
    run_test_case("test_missing_library<sz_hash_state_digest_best>", test_missing_library<sz_hash_state_digest_best>);
    run_test_case("test_missing_library<sz_fill_random_best>", test_missing_library<sz_fill_random_best>);
    run_test_case("test_missing_library<sz_sha256_state_init_best>", test_missing_library<sz_sha256_state_init_best>);
    run_test_case("test_missing_library<sz_sha256_state_update_best>",
                  test_missing_library<sz_sha256_state_update_best>);
    run_test_case("test_missing_library<sz_sha256_state_digest_best>",
                  test_missing_library<sz_sha256_state_digest_best>);
    run_test_case("test_missing_library<sz_sha256_multistate_update_best>",
                  test_missing_library<sz_sha256_multistate_update_best>);
    run_test_case("test_missing_library<sz_sha256_multistate_digest_best>",
                  test_missing_library<sz_sha256_multistate_digest_best>);
    run_test_case("test_missing_library<sz_aes256_key_init_best>", test_missing_library<sz_aes256_key_init_best>);
    run_test_case("test_missing_library<sz_aes256_ctr_xor_best>", test_missing_library<sz_aes256_ctr_xor_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_key_init_best>",
                  test_missing_library<sz_aes256_gcm_key_init_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_encrypt_best>", test_missing_library<sz_aes256_gcm_encrypt_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_decrypt_best>", test_missing_library<sz_aes256_gcm_decrypt_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_encryptor_init_best>",
                  test_missing_library<sz_aes256_gcm_encryptor_init_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_encryptor_associate_best>",
                  test_missing_library<sz_aes256_gcm_encryptor_associate_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_encryptor_update_best>",
                  test_missing_library<sz_aes256_gcm_encryptor_update_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_encryptor_digest_best>",
                  test_missing_library<sz_aes256_gcm_encryptor_digest_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_decryptor_init_best>",
                  test_missing_library<sz_aes256_gcm_decryptor_init_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_decryptor_associate_best>",
                  test_missing_library<sz_aes256_gcm_decryptor_associate_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_decryptor_update_unverified_best>",
                  test_missing_library<sz_aes256_gcm_decryptor_update_unverified_best>);
    run_test_case("test_missing_library<sz_aes256_gcm_decryptor_verify_best>",
                  test_missing_library<sz_aes256_gcm_decryptor_verify_best>);
    run_test_case("test_missing_library<sz_sort_order_best>", test_missing_library<sz_sort_order_best>);
    run_test_case("test_missing_library<sz_sort_uncased_order_best>", test_missing_library<sz_sort_uncased_order_best>);
    run_test_case("test_missing_library<sz_intersect_best>", test_missing_library<sz_intersect_best>);
    run_test_case("test_missing_library<sz_utf8_count_best>", test_missing_library<sz_utf8_count_best>);
    run_test_case("test_missing_library<sz_utf8_count_strings_best>", test_missing_library<sz_utf8_count_strings_best>);
    run_test_case("test_missing_library<sz_utf8_seek_best>", test_missing_library<sz_utf8_seek_best>);
    run_test_case("test_missing_library<sz_utf8_decode_best>", test_missing_library<sz_utf8_decode_best>);
    run_test_case("test_missing_library<sz_utf8_newlines_best>", test_missing_library<sz_utf8_newlines_best>);
    run_test_case("test_missing_library<sz_utf8_whitespaces_best>", test_missing_library<sz_utf8_whitespaces_best>);
    run_test_case("test_missing_library<sz_utf8_delimiters_best>", test_missing_library<sz_utf8_delimiters_best>);
    run_test_case("test_missing_library<sz_utf8_wordbreaks_best>", test_missing_library<sz_utf8_wordbreaks_best>);
    run_test_case("test_missing_library<sz_utf8_graphemes_best>", test_missing_library<sz_utf8_graphemes_best>);
    run_test_case("test_missing_library<sz_utf8_sentences_best>", test_missing_library<sz_utf8_sentences_best>);
    run_test_case("test_missing_library<sz_utf8_linebreaks_best>", test_missing_library<sz_utf8_linebreaks_best>);
    run_test_case("test_missing_library<sz_utf8_norm_best>", test_missing_library<sz_utf8_norm_best>);
    run_test_case("test_missing_library<sz_utf8_find_denormalized_best>",
                  test_missing_library<sz_utf8_find_denormalized_best>);
    run_test_case("test_missing_library<sz_utf8_uncased_fold_best>", test_missing_library<sz_utf8_uncased_fold_best>);
    run_test_case("test_missing_library<sz_utf8_uncased_needle_init_best>",
                  test_missing_library<sz_utf8_uncased_needle_init_best>);
    run_test_case("test_missing_library<sz_utf8_uncased_search_best>",
                  test_missing_library<sz_utf8_uncased_search_best>);
    run_test_case("test_missing_library<sz_utf8_uncased_order_best>", test_missing_library<sz_utf8_uncased_order_best>);
    run_test_case("test_missing_library<sz_utf8_find_cased_best>", test_missing_library<sz_utf8_find_cased_best>);
#else
    run_test_case("test_device_unit", test_device_unit);
    run_test_case("test_arithmetic_unit", test_arithmetic_unit);
    run_test_case("test_strings_view_unit", test_strings_view_unit);
    run_test_case("test_strings_columns_unit", test_strings_columns_unit);
    run_test_case("test_allocator_unit", test_allocator_unit);
    run_test_case("test_byteset_unit", test_byteset_unit);

    run_test_case("test_hash_unit", test_hash_unit);
    run_test_case("test_hash_all", test_hash_all);
    run_test_case("test_hash_multiseed_all", test_hash_multiseed_all);
    run_test_case("test_hash_safety", test_hash_safety);

    run_test_case("test_cipher_unit", test_cipher_unit);
    run_test_case("test_cipher_safety", test_cipher_safety);
    run_test_case("test_cipher_all", test_cipher_all);

    run_test_case("test_sort_unit", test_sort_unit);
    run_test_case("test_sort_reference_equivalence", test_sort_reference_equivalence);
    run_test_case("test_sort_all", test_sort_all);
    run_test_case("test_sort_safety", test_sort_safety);
    run_test_case("test_intersect_unit", test_intersect_unit);
    run_test_case("test_intersect_equivalence", test_intersect_equivalence);

    run_test_case("test_levenshtein_unit", test_levenshtein_unit);
    run_test_case("test_levenshtein_all", test_levenshtein_all);
    run_test_case("test_levenshtein_safety", test_levenshtein_safety);
    run_test_case("test_ngrams_unit", test_ngrams_unit);
    run_test_case("test_ngrams_all", test_ngrams_all);
    run_test_case("test_ngrams_safety", test_ngrams_safety);
    run_test_case("test_substrings_unit", test_substrings_unit);
    run_test_case("test_substrings_all", test_substrings_all);
    run_test_case("test_substrings_safety", test_substrings_safety);
    run_test_case("test_regex_unit", test_regex_unit);
    run_test_case("test_regex_all", test_regex_all);
    run_test_case("test_regex_safety", test_regex_safety);
    run_test_case("test_subdocuments_unit", test_subdocuments_unit);
    run_test_case("test_subdocuments_all", test_subdocuments_all);
    run_test_case("test_subdocuments_safety", test_subdocuments_safety);

    run_test_case("test_ascii_unit<sz::string_view_t>", test_ascii_unit<sz::string_view_t>);
    run_test_case("test_strings_unit", test_strings_unit);
    run_test_case("test_strings_all", test_strings_all);
    run_test_case("test_strings_safety", test_strings_safety);
    run_test_case("test_stl_reads_unit<std::string_view>", test_stl_reads_unit<std::string_view>);
    run_test_case("test_stl_reads_unit<std::string>", test_stl_reads_unit<std::string>);
    run_test_case("test_stl_reads_unit<sz::string_view_t>", test_stl_reads_unit<sz::string_view_t>);
    run_test_case("test_stl_updates_unit", test_stl_updates_unit);
    run_test_case("test_stl_conversions_unit", test_stl_conversions_unit);
    run_test_case("test_stl_containers_unit", test_stl_containers_unit);
    run_test_case("test_extensions_reads_unit<sz::string_view_t>", test_extensions_reads_unit<sz::string_view_t>);
    run_test_case("test_extensions_updates_unit", test_extensions_updates_unit);
    run_test_case("test_extensions_ranges_unit", test_extensions_ranges_unit);
    run_test_case("test_string_constructors_unit", test_string_constructors_unit);
    run_test_case("test_string_reserve_unit", test_string_reserve_unit);
    run_test_case("test_memory_stability_equivalence(1024)", test_memory_stability_equivalence, 1024);
    run_test_case("test_memory_stability_equivalence(14)", test_memory_stability_equivalence, 14);
    run_test_case("test_string_updates_equivalence(1024)", test_string_updates_equivalence, 1024);

    run_test_case("test_compare_unit", test_compare_unit);
    run_test_case("test_find_unit", test_find_unit);
    run_test_case("test_find_all", test_find_all);
    run_test_case("test_find_safety", test_find_safety);
    run_test_case("test_lookup_equivalence(32, 16)", test_lookup_equivalence, 32, 16);
    run_test_case("test_find_misaligned_equivalence", test_find_misaligned_equivalence);

    run_test_case("test_utf8_runes_unit", test_utf8_runes_unit);
    run_test_case("test_utf8_runes_scripts_unit", test_utf8_runes_scripts_unit);
    run_test_case("test_utf8_runes_safety", test_utf8_runes_safety);
    run_test_case("test_utf8_runes_all", test_utf8_runes_all);
    run_test_case("test_utf8_tokens_unit", test_utf8_tokens_unit);
    run_test_case("test_utf8_tokens_scripts_unit", test_utf8_tokens_scripts_unit);
    run_test_case("test_utf8_tokens_safety", test_utf8_tokens_safety);
    run_test_case("test_utf8_tokens_all", test_utf8_tokens_all);
    run_test_case("test_utf8_wordbreaks_unit", test_utf8_wordbreaks_unit);
    run_test_case("test_utf8_wordbreaks_rules", test_utf8_wordbreaks_rules);
    run_test_case("test_utf8_wordbreaks_safety", test_utf8_wordbreaks_safety);
    run_test_case("test_utf8_wordbreaks_all", test_utf8_wordbreaks_all);
    run_test_case("test_utf8_graphemes_unit", test_utf8_graphemes_unit);
    run_test_case("test_utf8_graphemes_rules", test_utf8_graphemes_rules);
    run_test_case("test_utf8_graphemes_safety", test_utf8_graphemes_safety);
    run_test_case("test_utf8_graphemes_all", test_utf8_graphemes_all);
    run_test_case("test_utf8_sentences_unit", test_utf8_sentences_unit);
    run_test_case("test_utf8_sentences_rules", test_utf8_sentences_rules);
    run_test_case("test_utf8_sentences_safety", test_utf8_sentences_safety);
    run_test_case("test_utf8_sentences_all", test_utf8_sentences_all);
    run_test_case("test_utf8_linebreaks_unit", test_utf8_linebreaks_unit);
    run_test_case("test_utf8_linebreaks_rules", test_utf8_linebreaks_rules);
    run_test_case("test_utf8_linebreaks_safety", test_utf8_linebreaks_safety);
    run_test_case("test_utf8_linebreaks_all", test_utf8_linebreaks_all);
    run_test_case("test_utf8_delimiters_unit", test_utf8_delimiters_unit);
    run_test_case("test_utf8_delimiters_safety", test_utf8_delimiters_safety);
    run_test_case("test_utf8_delimiters_all", test_utf8_delimiters_all);

    run_test_case("test_utf8_norm_unit", test_utf8_norm_unit);
    run_test_case("test_utf8_norm_safety", test_utf8_norm_safety);
    run_test_case("test_utf8_norm_all", test_utf8_norm_all);
    run_test_case("test_utf8_uncased_unit", test_utf8_uncased_unit);
    run_test_case("test_utf8_uncased_scripts_unit", test_utf8_uncased_scripts_unit);
    run_test_case("test_utf8_uncased_regressions_unit", test_utf8_uncased_regressions_unit);
    run_test_case("test_utf8_uncased_all", test_utf8_uncased_all);
    run_test_case("test_utf8_uncased_safety", test_utf8_uncased_safety);
#endif // STRINGZILLA_HEADER_ONLY

    test_cross_serial();
    test_cross_x8664();
    test_cross_arm64();
    test_cross_riscv64();
    test_cross_loongarch64();
    test_cross_ppc64();
    test_cross_wasm();
    if constexpr (!STRINGZILLA_HEADER_ONLY) {
        for (device_selection_t device : devices) {
            print(device);
            switch (device.backend) {
            case sz::device_kind_t::cuda_k: test_cross_cuda(device.ordinal); break;
            case sz::device_kind_t::rocm_k: test_cross_rocm(device.ordinal); break;
            case sz::device_kind_t::metal_k: test_cross_metal(device.ordinal); break;
            case sz::device_kind_t::cpu_k: break;
            }
        }
    }

    // A check outside every case, like a GPU suite's setup, fails the run without failing a case.
    int const exit_code = finish_test_run();
    // ctest hides passing output; the blank line keeps the next binary's header out of this list
    if (char const *summary_path = std::getenv("GITHUB_STEP_SUMMARY"))
        if (std::FILE *summary = std::fopen(summary_path, "a")) {
            std::fputc('\n', summary), print(probe_machine(), summary);
            fmt::println(summary, "- Cases: {} passed, {} failed, {} skipped", test_run.passed_cases,
                         test_run.failed_cases, test_run.skipped_cases);
            std::fclose(summary);
        }
    return exit_code;
}
