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
 *  directives and other issues. The C++ layer needs the library, so header-only builds skip it. */
#include <stringzilla/stringzilla.h> // Primary C API
#if !STRINGZILLA_HEADER_ONLY
#include <stringzilla/stringzilla.hpp> // C++ string class replacement
#endif

#include <map>           // `std::map`
#include <string>        // `std::string` baseline
#include <string_view>   // `std::string_view` baseline
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include <fmt/format.h>

#include "harness.hpp" // `read_settings`, `run_test`

using namespace ashvardanian::stringzilla::test;

/*  Instantiate all the templates to make the symbols visible and also check for weird compilation
 *  errors on uncommon paths. */
#if !STRINGZILLA_HEADER_ONLY

template class std::basic_string_view<char>;
template class sz::basic_string_slice<char>;
template class std::basic_string<char>;
template class sz::basic_string<>;
template class sz::basic_string_slice<char const>;

template class std::vector<sz::string_t>;
template class std::map<sz::string_t, int>;
template class std::unordered_map<sz::string_t, int>;

template class std::vector<sz::string_view_t>;
template class std::map<sz::string_view_t, int>;
template class std::unordered_map<sz::string_view_t, int>;
#endif // !STRINGZILLA_HEADER_ONLY

namespace ashvardanian::stringzilla::test {

#if !STRINGZILLA_HEADER_ONLY
/** The CPU device answers as the C queries do and prepares its own threads, while every kind
 *  refuses the ordinal past its last device and a GPU refuses to configure a CPU thread. */
void test_device_unit() {
    sz::device_t const cpu = sz::device_t::cpu();
    auto const [detected, detected_status] = cpu.capabilities_detected();
    auto const [enabled, enabled_status] = cpu.capabilities_enabled();
    verify(sz::succeeded(detected_status) && sz::succeeded(enabled_status));
    verify((enabled & sz_cap_serial_k) && !(enabled & ~(detected | sz_cap_serial_k)));
    verify(!(enabled & ~cpu.capabilities_compiled()));
    verify(sz::succeeded(cpu.configure_thread(enabled)));
    for (sz::device_kind_t kind :
         {sz::device_kind_t::cpu_k, sz::device_kind_t::cuda_k, sz::device_kind_t::rocm_k, sz::device_kind_t::metal_k}) {
        auto const [devices, count_status] = sz::device_t::count(kind);
        if (sz::failed(count_status)) continue; // No device of this kind answers
        verify(kind != sz::device_kind_t::cpu_k || devices == 1);
        verify(sz::device_t::make(kind, devices).status == sz::status_t::missing_gpu_k);
        if (kind == sz::device_kind_t::cpu_k) continue;
        auto const [gpu, make_status] = sz::device_t::make(kind, 0);
        verify(sz::succeeded(make_status) && !(gpu.capabilities_compiled() & sz_cap_cpus_k));
        verify(gpu.configure_thread(sz_cap_any_k) == sz::status_t::missing_kernel_k);
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
        call_stub_(sz_copy_best),
        call_stub_(sz_move_best),
        call_stub_(sz_fill_best),
        call_stub_(sz_lookup_best),
        call_stub_(sz_allocator_init_unified_best),
        call_stub_(sz_allocator_init_device_best),
        call_stub_(sz_allocator_init_pinned_best),
        call_stub_(sz_sequence_realloc_best),
        call_stub_(sz_find_byte_best),
        call_stub_(sz_rfind_byte_best),
        call_stub_(sz_find_best),
        call_stub_(sz_rfind_best),
        call_stub_(sz_find_byteset_best),
        call_stub_(sz_rfind_byteset_best),
        call_stub_(sz_bytesum_best),
        call_stub_(sz_hash_best),
        call_stub_(sz_hash_multiseed_best),
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
        call_stub_(sz_sequence_argsort_best),
        call_stub_(sz_sequence_argsort_uncased_best),
        call_stub_(sz_sequence_intersect_best),
        call_stub_(sz_levenshtein_distance_tiled_best),
        call_stub_(sz_utf8_count_best),
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
    for (sz_status_t status : statuses) verify(status == sz_missing_library_k);
}

#endif // !STRINGZILLA_HEADER_ONLY

/** Each family's finder picks what @c sz_find_kernel_punned picks for the family's kinds and misses
 *  every other kind, while a header-only build reports the missing library from all of them. */
void test_find_kernel_unit() {
    sz_status_t (*const find_kernels[])(sz_kernel_kind_t, sz_capability_t, sz_kernel_punned_t *, sz_capability_t *) = {
        sz_compare_find_kernel,
        sz_memory_find_kernel,
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
    };
    sz_status_t const missing = STRINGZILLA_HEADER_ONLY ? sz_missing_library_k : sz_missing_kernel_k;
    sz_kernel_punned_t const poison = reinterpret_cast<sz_kernel_punned_t>(&test_find_kernel_unit);
    sz_capability_t enabled = 0;
    sz_capabilities_enabled_cpu(&enabled);
    for (int kind = sz_kernel_unknown_k; kind <= sz_kernel_substrings_bm25_scores_k; ++kind) {
        sz_kernel_punned_t expected_kernel = poison;
        sz_capability_t expected_capability = sz_cap_serial_k;
        sz_status_t const expected = sz_find_kernel_punned(static_cast<sz_kernel_kind_t>(kind), enabled,
                                                           &expected_kernel, &expected_capability);
        verify(expected == sz_success_k || (expected == missing && !expected_kernel && !expected_capability));
        std::size_t owners = 0;
        for (auto find_kernel : find_kernels) {
            sz_kernel_punned_t kernel = poison;
            sz_capability_t capability = sz_cap_serial_k;
            sz_status_t const status = find_kernel(static_cast<sz_kernel_kind_t>(kind), enabled, &kernel, &capability);
            if (status == sz_success_k) verify(kernel == expected_kernel && capability == expected_capability);
            else verify(status == missing && !kernel && !capability);
            owners += status == sz_success_k;
        }
        verify(owners == (expected == sz_success_k));
    }
}

std::vector<sz::device_t> select_devices(std::optional<std::vector<device_selection_t>> const &requested) {
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

int main(int, char const **argv) {
    install_test_signal_handlers();
    settings_t const settings = read_settings(argv[0]);
    environment_t const env {settings, probe_machine()};
    auto const devices = select_devices(env.settings.devices);
    print(env.machine);
    print(env.settings);

    std::size_t failures = 0;
    failures += run_test(env.settings, "test_find_kernel_unit", test_find_kernel_unit);

#if STRINGZILLA_HEADER_ONLY
    failures += run_test(env.settings, "test_dispatch_stubs_unit", test_dispatch_stubs_unit);
#else
    failures += run_test(env.settings, "test_device_unit", test_device_unit);
    failures += run_test(env.settings, "test_arithmetic_unit", test_arithmetic_unit);
    failures += run_test(env.settings, "test_sequence_unit", test_sequence_unit);
    failures += run_test(env.settings, "test_tape_assign_unit", test_tape_assign_unit);
    failures += run_test(env.settings, "test_tape_overflow_unit", test_tape_overflow_unit);
    failures += run_test(env.settings, "test_allocator_unit", test_allocator_unit);
    failures += run_test(env.settings, "test_byteset_unit", test_byteset_unit);

    failures += run_test(env.settings, "test_hash_unit", test_hash_unit);
    failures += run_test(env.settings, "test_hash_all", test_hash_all);
    failures += run_test(env.settings, "test_hash_multiseed_all", test_hash_multiseed_all);
    failures += run_test(env.settings, "test_hash_safety", test_hash_safety);

    failures += run_test(env.settings, "test_cipher_unit", test_cipher_unit);
    failures += run_test(env.settings, "test_cipher_safety", test_cipher_safety);
    failures += run_test(env.settings, "test_cipher_all", test_cipher_all);

    failures += run_test(env.settings, "test_sort_unit", test_sort_unit);
    failures += run_test(env.settings, "test_sort_reference_equivalence", test_sort_reference_equivalence);
    failures += run_test(env.settings, "test_sort_all", test_sort_all);
    failures += run_test(env.settings, "test_sort_safety", test_sort_safety);
    failures += run_test(env.settings, "test_intersect_unit", test_intersect_unit);
    failures += run_test(env.settings, "test_intersect_equivalence", test_intersect_equivalence);

    failures += run_test(env.settings, "test_levenshtein_unit", test_levenshtein_unit);
    failures += run_test(env.settings, "test_levenshtein_all", test_levenshtein_all);
    failures += run_test(env.settings, "test_levenshtein_safety", test_levenshtein_safety);
    failures += run_test(env.settings, "test_overlap_unit", test_overlap_unit);
    failures += run_test(env.settings, "test_overlap_all", test_overlap_all);
    failures += run_test(env.settings, "test_overlap_safety", test_overlap_safety);
    failures += run_test(env.settings, "test_substrings_unit", test_substrings_unit);
    failures += run_test(env.settings, "test_substrings_all", test_substrings_all);
    failures += run_test(env.settings, "test_substrings_safety", test_substrings_safety);

    failures += run_test(env.settings, "test_ascii_unit<sz::string_t>", test_ascii_unit<sz::string_t>);
    failures += run_test(env.settings, "test_ascii_unit<sz::string_view_t>", test_ascii_unit<sz::string_view_t>);
    failures += run_test(env.settings, "test_memory_unit", test_memory_unit);
    failures += run_test(env.settings, "test_memory_all", test_memory_all);
    failures += run_test(env.settings, "test_memory_safety", test_memory_safety);
    failures += run_test(env.settings, "test_stl_reads_unit<std::string_view>", test_stl_reads_unit<std::string_view>);
    failures += run_test(env.settings, "test_stl_reads_unit<std::string>", test_stl_reads_unit<std::string>);
    failures += run_test(env.settings, "test_stl_reads_unit<sz::string_view_t>",
                         test_stl_reads_unit<sz::string_view_t>);
    failures += run_test(env.settings, "test_stl_reads_unit<sz::string_t>", test_stl_reads_unit<sz::string_t>);
    failures += run_test(env.settings, "test_stl_updates_unit<std::string>", test_stl_updates_unit<std::string>);
    failures += run_test(env.settings, "test_stl_updates_unit<sz::string_t>", test_stl_updates_unit<sz::string_t>);
    failures += run_test(env.settings, "test_stl_conversions_unit", test_stl_conversions_unit);
    failures += run_test(env.settings, "test_stl_containers_unit", test_stl_containers_unit);
    failures += run_test(env.settings, "test_extensions_reads_unit<sz::string_view_t>",
                         test_extensions_reads_unit<sz::string_view_t>);
    failures += run_test(env.settings, "test_extensions_reads_unit<sz::string_t>",
                         test_extensions_reads_unit<sz::string_t>);
    failures += run_test(env.settings, "test_extensions_updates_unit", test_extensions_updates_unit);
    failures += run_test(env.settings, "test_extensions_ranges_unit", test_extensions_ranges_unit);
    failures += run_test(env.settings, "test_string_constructors_unit", test_string_constructors_unit);
    failures += run_test(env.settings, "test_string_reserve_unit", test_string_reserve_unit);
    failures += run_test(env.settings, "test_memory_stability_equivalence_1024",
                         [](test_context_t &context) { test_memory_stability_equivalence(context, 1024); });
    failures += run_test(env.settings, "test_memory_stability_equivalence_14",
                         [](test_context_t &context) { test_memory_stability_equivalence(context, 14); });
    failures += run_test(env.settings, "test_string_updates_equivalence",
                         [](test_context_t &context) { test_string_updates_equivalence(context); }); // ! Defaulted

    failures += run_test(env.settings, "test_compare_unit", test_compare_unit);
    failures += run_test(env.settings, "test_find_unit", test_find_unit);
    failures += run_test(env.settings, "test_find_all", test_find_all);
    failures += run_test(env.settings, "test_find_safety", test_find_safety);
    failures += run_test(env.settings, "test_lookup_equivalence",
                         [](test_context_t &context) { test_lookup_equivalence(context); }); // ! Defaulted args
    failures += run_test(env.settings, "test_find_misaligned_equivalence", test_find_misaligned_equivalence);

    failures += run_test(env.settings, "test_utf8_runes_unit", test_utf8_runes_unit);
    failures += run_test(env.settings, "test_utf8_runes_scripts_unit", test_utf8_runes_scripts_unit);
    failures += run_test(env.settings, "test_utf8_runes_safety", test_utf8_runes_safety);
    failures += run_test(env.settings, "test_utf8_runes_all", test_utf8_runes_all);
    failures += run_test(env.settings, "test_utf8_tokens_unit", test_utf8_tokens_unit);
    failures += run_test(env.settings, "test_utf8_tokens_scripts_unit", test_utf8_tokens_scripts_unit);
    failures += run_test(env.settings, "test_utf8_tokens_safety", test_utf8_tokens_safety);
    failures += run_test(env.settings, "test_utf8_tokens_all", test_utf8_tokens_all);
    failures += run_test(env.settings, "test_utf8_wordbreaks_unit", test_utf8_wordbreaks_unit);
    failures += run_test(env.settings, "test_utf8_wordbreaks_rules", test_utf8_wordbreaks_rules);
    failures += run_test(env.settings, "test_utf8_wordbreaks_safety", test_utf8_wordbreaks_safety);
    failures += run_test(env.settings, "test_utf8_wordbreaks_all", test_utf8_wordbreaks_all);
    failures += run_test(env.settings, "test_utf8_graphemes_unit", test_utf8_graphemes_unit);
    failures += run_test(env.settings, "test_utf8_graphemes_rules", test_utf8_graphemes_rules);
    failures += run_test(env.settings, "test_utf8_graphemes_safety", test_utf8_graphemes_safety);
    failures += run_test(env.settings, "test_utf8_graphemes_all", test_utf8_graphemes_all);
    failures += run_test(env.settings, "test_utf8_sentences_unit", test_utf8_sentences_unit);
    failures += run_test(env.settings, "test_utf8_sentences_rules", test_utf8_sentences_rules);
    failures += run_test(env.settings, "test_utf8_sentences_safety", test_utf8_sentences_safety);
    failures += run_test(env.settings, "test_utf8_sentences_all", test_utf8_sentences_all);
    failures += run_test(env.settings, "test_utf8_linebreaks_unit", test_utf8_linebreaks_unit);
    failures += run_test(env.settings, "test_utf8_linebreaks_rules", test_utf8_linebreaks_rules);
    failures += run_test(env.settings, "test_utf8_linebreaks_safety", test_utf8_linebreaks_safety);
    failures += run_test(env.settings, "test_utf8_linebreaks_all", test_utf8_linebreaks_all);
    failures += run_test(env.settings, "test_utf8_delimiters_unit", test_utf8_delimiters_unit);
    failures += run_test(env.settings, "test_utf8_delimiters_safety", test_utf8_delimiters_safety);
    failures += run_test(env.settings, "test_utf8_delimiters_all", test_utf8_delimiters_all);

    failures += run_test(env.settings, "test_utf8_norm_unit", test_utf8_norm_unit);
    failures += run_test(env.settings, "test_utf8_norm_safety", test_utf8_norm_safety);
    failures += run_test(env.settings, "test_utf8_norm_all", test_utf8_norm_all);
    failures += run_test(env.settings, "test_utf8_uncased_unit", test_utf8_uncased_unit);
    failures += run_test(env.settings, "test_utf8_uncased_scripts_unit", test_utf8_uncased_scripts_unit);
    failures += run_test(env.settings, "test_utf8_uncased_regressions_unit", test_utf8_uncased_regressions_unit);
    failures += run_test(env.settings, "test_utf8_uncased_all", test_utf8_uncased_all);
    failures += run_test(env.settings, "test_utf8_uncased_safety", test_utf8_uncased_safety);
#endif // STRINGZILLA_HEADER_ONLY

    failures += test_cross_serial(env);
    failures += test_cross_x8664(env);
    failures += test_cross_arm64(env);
    failures += test_cross_riscv64(env);
    failures += test_cross_loongarch64(env);
    failures += test_cross_ppc64(env);
    failures += test_cross_wasm(env);
    if constexpr (!STRINGZILLA_HEADER_ONLY) {
        for (sz::device_t device : devices) {
            print(device);
            switch (device.kind()) {
            case sz::device_kind_t::cuda_k: failures += test_cross_cuda(env, device.ordinal()); break;
            case sz::device_kind_t::rocm_k: failures += test_cross_rocm(env, device.ordinal()); break;
            case sz::device_kind_t::metal_k: failures += test_cross_metal(env, device.ordinal()); break;
            case sz::device_kind_t::cpu_k: break;
            }
        }
    }

    if (failures != 0) {
        fmt::println(stderr, "\n{} test(s) failed.", failures);
        return 1;
    }
    fmt::println("\nAll tests passed!");
    return 0;
}
