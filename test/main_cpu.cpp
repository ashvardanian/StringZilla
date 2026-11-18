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
 *  the stubs alone.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#undef STRINGZILLA_DEBUG
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Include the StringZilla headers before anything else, to intercept missing @c #include
 *  directives and other issues. */
#include "stringzilla/stringzilla.h"   // Primary C API
#include "stringzilla/stringzilla.hpp" // C++ string class replacement

#include <map>           // `std::map`
#include <string>        // `std::string` baseline
#include <string_view>   // `std::string_view` baseline
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include <fmt/format.h>

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
template class std::unordered_map<sz::string_view_t, int>;

namespace ashvardanian::stringzilla::test {

#if !STRINGZILLA_HEADER_ONLY
/** The CPU device answers as the C queries do and prepares its own threads, while every kind
 *  refuses the ordinal past its last device and a GPU refuses to configure a CPU thread. */
void test_device_unit() {
    sz::device_t const cpu = sz::device_t::cpu();
    auto const [detected, detected_status] = cpu.capabilities_detected();
    auto const [enabled, enabled_status] = cpu.capabilities_enabled();
    auto const [compiled, compiled_status] = cpu.capabilities_compiled();
    expect_true(sz::succeeded(detected_status) && sz::succeeded(enabled_status) && sz::succeeded(compiled_status));
    expect_true((enabled & sz_cap_serial_k) && !(enabled & ~(detected | sz_cap_serial_k)));
    expect_true(!(enabled & ~compiled));
    expect_true(sz::succeeded(cpu.configure_thread(enabled)));
    for (sz::device_kind_t kind :
         {sz::device_kind_t::cpu_k, sz::device_kind_t::cuda_k, sz::device_kind_t::rocm_k, sz::device_kind_t::metal_k}) {
        auto const [devices, count_status] = sz::device_t::count(kind);
        if (sz::failed(count_status)) continue; // No device of this kind answers
        expect_true(kind != sz::device_kind_t::cpu_k || devices == 1);
        expect_equal(sz::device_t::make(kind, devices).status, sz::status_t::missing_gpu_k);
        if (kind == sz::device_kind_t::cpu_k) continue;
        auto const [gpu, make_status] = sz::device_t::make(kind, 0);
        if (!expect_true(sz::succeeded(make_status))) continue;
        auto const [gpu_compiled, gpu_compiled_status] = gpu.capabilities_compiled();
        expect_true(sz::succeeded(gpu_compiled_status) && !(gpu_compiled & sz_cap_cpus_k));
        expect_equal(gpu.configure_thread(sz_cap_any_k), sz::status_t::missing_kernel_k);
    }
}

#else

/** Calls @p best with every operand value-initialized, which a header-only stub never reads. */
template <typename... arguments_types_>
sz_status_t call_stub_(sz_status_t (*best)(arguments_types_...)) noexcept {
    return best(arguments_types_ {}...);
}

/** Every dispatch point reports the missing library in a header-only build. */
void test_dispatch_stubs_unit() {
    sz_status_t const statuses[] = {
        call_stub_(sz_stream_synchronize_best),
        call_stub_(sz_equal_best),
        call_stub_(sz_order_best),
        call_stub_(sz_strings_to_tape_best),
        call_stub_(sz_strings_to_rows_best),
        call_stub_(sz_strings_escape_best),
        call_stub_(sz_strings_unescape_best),
        call_stub_(sz_strings_lookup_best),
        call_stub_(sz_strings_normalize_best),
        call_stub_(sz_strings_split_best),
        call_stub_(sz_strings_split_lines_best),
        call_stub_(sz_strings_join_best),
        call_stub_(sz_allocator_init_unified_best),
        call_stub_(sz_allocator_init_device_best),
        call_stub_(sz_allocator_init_pinned_best),
        call_stub_(sz_find_byte_best),
        call_stub_(sz_rfind_byte_best),
        call_stub_(sz_find_best),
        call_stub_(sz_rfind_best),
        call_stub_(sz_find_byteset_best),
        call_stub_(sz_rfind_byteset_best),
        call_stub_(sz_find_strings_best),
        call_stub_(sz_bytesum_best),
        call_stub_(sz_hash_best),
        call_stub_(sz_hash_multiseed_best),
        call_stub_(sz_hash_strings_best),
        call_stub_(sz_hash_state_init_best),
        call_stub_(sz_hash_state_update_best),
        call_stub_(sz_hash_state_digest_best),
        call_stub_(sz_fill_random_best),
        call_stub_(sz_sha256_state_init_best),
        call_stub_(sz_sha256_state_update_best),
        call_stub_(sz_sha256_state_digest_best),
        call_stub_(sz_sha256_multistate_update_best),
        call_stub_(sz_sha256_multistate_digest_best),
        call_stub_(sz_aes256_key_init_best),
        call_stub_(sz_aes256_ctr_xor_best),
        call_stub_(sz_aes256_gcm_key_init_best),
        call_stub_(sz_aes256_gcm_encrypt_best),
        call_stub_(sz_aes256_gcm_decrypt_best),
        call_stub_(sz_aes256_gcm_encryptor_init_best),
        call_stub_(sz_aes256_gcm_encryptor_associate_best),
        call_stub_(sz_aes256_gcm_encryptor_update_best),
        call_stub_(sz_aes256_gcm_encryptor_digest_best),
        call_stub_(sz_aes256_gcm_decryptor_init_best),
        call_stub_(sz_aes256_gcm_decryptor_associate_best),
        call_stub_(sz_aes256_gcm_decryptor_update_unverified_best),
        call_stub_(sz_aes256_gcm_decryptor_verify_best),
        call_stub_(sz_sort_order_best),
        call_stub_(sz_sort_uncased_order_best),
        call_stub_(sz_intersect_best),
        call_stub_(sz_utf8_count_best),
        call_stub_(sz_utf8_count_strings_best),
        call_stub_(sz_utf8_seek_best),
        call_stub_(sz_utf8_decode_best),
        call_stub_(sz_utf8_newlines_best),
        call_stub_(sz_utf8_whitespaces_best),
        call_stub_(sz_utf8_delimiters_best),
        call_stub_(sz_utf8_wordbreaks_best),
        call_stub_(sz_utf8_graphemes_best),
        call_stub_(sz_utf8_sentences_best),
        call_stub_(sz_utf8_linebreaks_best),
        call_stub_(sz_utf8_norm_best),
        call_stub_(sz_utf8_find_denormalized_best),
        call_stub_(sz_utf8_uncased_fold_best),
        call_stub_(sz_utf8_uncased_needle_init_best),
        call_stub_(sz_utf8_uncased_search_best),
        call_stub_(sz_utf8_uncased_order_best),
        call_stub_(sz_utf8_find_cased_best),
    };
    for (sz_status_t status : statuses) expect_equal(status, sz_missing_library_k);

    // The C++ layer runs the serial kernels over the empty default mask instead.
    sz::string_view_t const text = "a quick brown fox";
    expect_equal(text.find("brown"), 8);
    expect_equal(text.utf8_count(), 17);
}

#endif // !STRINGZILLA_HEADER_ONLY

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
        sz_overlap_find_kernel,
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
std::vector<sz::device_t> selected_devices() {
    std::optional<std::vector<device_selection_t>> const &requested = test_run.settings.devices;
    std::vector<sz::device_t> devices;
    if constexpr (STRINGZILLA_HEADER_ONLY) {
        if (requested) {
            fmt::println(stderr, "This header-only executable accepts only CPU workloads");
            std::exit(1);
        }
        return devices;
    }
    if (requested) {
        for (device_selection_t const &selection : *requested) {
            auto const device = sz::device_t::make(selection.backend, selection.ordinal);
            if (!device) {
                fmt::println(stderr, "Device {}:{} is unavailable (status {})", device_name(selection.backend),
                             selection.ordinal, static_cast<int>(device.status));
                std::exit(1);
            }
            devices.push_back(device.value);
        }
    }
    else {
        for (sz::device_kind_t kind :
             {sz::device_kind_t::cuda_k, sz::device_kind_t::rocm_k, sz::device_kind_t::metal_k})
            if (auto device = sz::device_t::make(kind, 0)) devices.push_back(device.value);
    }
    return devices;
}

} // namespace ashvardanian::stringzilla::test

int main(int argc, char **argv) {
    start_test_run(argc, argv);
    std::vector<sz::device_t> const devices = selected_devices();

    run_test_case("test_find_kernel_unit", test_find_kernel_unit);

#if STRINGZILLA_HEADER_ONLY
    run_test_case("test_dispatch_stubs_unit", test_dispatch_stubs_unit);
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
    run_test_case("test_overlap_unit", test_overlap_unit);
    run_test_case("test_overlap_all", test_overlap_all);
    run_test_case("test_overlap_safety", test_overlap_safety);
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
        for (sz::device_t device : devices) {
            print(device);
            switch (device.kind()) {
            case sz::device_kind_t::cuda_k: test_cross_cuda(device.ordinal()); break;
            case sz::device_kind_t::rocm_k: test_cross_rocm(device.ordinal()); break;
            case sz::device_kind_t::metal_k: test_cross_metal(device.ordinal()); break;
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
