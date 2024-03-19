/**
 *  @file   bench_search.cpp
 *  @brief  Benchmarks for bidirectional string search operations - exact and TODO: approximate.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_token.cpp` and `bench_similarity.cpp`.
 *  It accepts a file with a list of words, and benchmarks the search operations on them.
 *  Outside of present tokens also tries missing tokens.
 */
#include <cstring>    // `memmem`
#include <functional> // `std::boyer_moore_searcher`

#define SZ_USE_MISALIGNED_LOADS (1)
#include <bench.hpp>

using namespace ashvardanian::stringzilla::scripts;

tracked_binary_functions_t find_functions() {
    // ! Despite receiving string-views, following functions are assuming the strings are null-terminated.
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](std::string_view h, std::string_view n) {
            sz_cptr_t match = function(h.data(), h.size(), n.data(), n.size());
            return (match ? match - h.data() : h.size());
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.find",
         [](std::string_view h, std::string_view n) {
             auto match = n.size() == 1 ? h.find(n.front()) : h.find(n);
             return (match == std::string_view::npos ? h.size() : match);
         }},
        {"sz_find_serial", wrap_sz(sz_find_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_find_avx512", wrap_sz(sz_find_avx512), true},
#endif
#if SZ_USE_X86_AVX2
        {"sz_find_avx2", wrap_sz(sz_find_avx2), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_find_neon", wrap_sz(sz_find_neon), true},
#endif
        {"strstr/strchr",
         [](std::string_view h, std::string_view n) {
             sz_cptr_t match = n.size() == 1 ? (sz_cptr_t)strchr(h.data(), n.front()) //
                                             : (sz_cptr_t)strstr(h.data(), n.data());
             return (match ? match - h.data() : h.size());
         }},
#ifdef _GNU_SOURCE
        {"memmem/memchr", // Not supported on MSVC
         [](std::string_view h, std::string_view n) {
             sz_cptr_t match = n.size() == 1 ? (sz_cptr_t)memchr(h.data(), n.front(), h.size())
                                             : (sz_cptr_t)memmem(h.data(), h.size(), n.data(), n.size());
             return (match ? match - h.data() : h.size());
         }},
#endif
        {"std::search<>",
         [](std::string_view h, std::string_view n) {
             auto match = std::search(h.data(), h.data() + h.size(), n.data(), n.data() + n.size());
             return (match - h.data());
         }},
#if __cpp_lib_boyer_moore_searcher
        {"std::search<BM>",
         [](std::string_view h, std::string_view n) {
             auto match =
                 std::search(h.data(), h.data() + h.size(), std::boyer_moore_searcher(n.data(), n.data() + n.size()));
             return (match - h.data());
         }},
        {"std::search<BMH>",
         [](std::string_view h, std::string_view n) {
             auto match = std::search(h.data(), h.data() + h.size(),
                                      std::boyer_moore_horspool_searcher(n.data(), n.data() + n.size()));
             return (match - h.data());
         }},
#endif
    };
    return result;
}

tracked_binary_functions_t rfind_functions() {
    // ! Despite receiving string-views, following functions are assuming the strings are null-terminated.
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](std::string_view h, std::string_view n) {
            sz_cptr_t match = function(h.data(), h.size(), n.data(), n.size());
            return (match ? match - h.data() : 0);
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.rfind",
         [](std::string_view h, std::string_view n) {
             auto match = n.size() == 1 ? h.rfind(n.front()) : h.rfind(n);
             return (match == std::string_view::npos ? 0 : match);
         }},
        {"sz_rfind_serial", wrap_sz(sz_rfind_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_rfind_avx512", wrap_sz(sz_rfind_avx512), true},
#endif
#if SZ_USE_X86_AVX2
        {"sz_rfind_avx2", wrap_sz(sz_rfind_avx2), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_rfind_neon", wrap_sz(sz_rfind_neon), true},
#endif
        {"std::search<R>",
         [](std::string_view h, std::string_view n) {
             auto match = std::search(h.rbegin(), h.rend(), n.rbegin(), n.rend());
             auto offset_from_end = (sz_ssize_t)(match - h.rbegin());
             return h.size() - offset_from_end;
         }},
#if __cpp_lib_boyer_moore_searcher
        {"std::search<R, BM>",
         [](std::string_view h, std::string_view n) {
             auto match = std::search(h.rbegin(), h.rend(), std::boyer_moore_searcher(n.rbegin(), n.rend()));
             auto offset_from_end = (sz_ssize_t)(match - h.rbegin());
             return h.size() - offset_from_end;
         }},
        {"std::search<R, BMH>",
         [](std::string_view h, std::string_view n) {
             auto match = std::search(h.rbegin(), h.rend(), std::boyer_moore_horspool_searcher(n.rbegin(), n.rend()));
             auto offset_from_end = (sz_ssize_t)(match - h.rbegin());
             return h.size() - offset_from_end;
         }},
#endif
    };
    return result;
}

tracked_binary_functions_t find_charset_functions() {
    // ! Despite receiving string-views, following functions are assuming the strings are null-terminated.
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](std::string_view h, std::string_view n) {
            sz::char_set set;
            for (auto c : n) set.add(c);
            sz_cptr_t match = function(h.data(), h.size(), &set.raw());
            return (match ? match - h.data() : h.size());
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.find_first_of",
         [](std::string_view h, std::string_view n) {
             auto match = h.find_first_of(n);
             return (match == std::string_view::npos ? h.size() : match);
         }},
        {"sz_find_charset_serial", wrap_sz(sz_find_charset_serial), true},
#if SZ_USE_X86_AVX2
        {"sz_find_charset_avx2", wrap_sz(sz_find_charset_avx2), true},
#endif
#if SZ_USE_X86_AVX512
        {"sz_find_charset_avx512", wrap_sz(sz_find_charset_avx512), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_find_charset_neon", wrap_sz(sz_find_charset_neon), true},
#endif
        {"strcspn", [](std::string_view h, std::string_view n) { return strcspn(h.data(), n.data()); }},
    };
    return result;
}

tracked_binary_functions_t rfind_charset_functions() {
    // ! Despite receiving string-views, following functions are assuming the strings are null-terminated.
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](std::string_view h, std::string_view n) {
            sz::char_set set;
            for (auto c : n) set.add(c);
            sz_cptr_t match = function(h.data(), h.size(), &set.raw());
            return (match ? match - h.data() : 0);
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.find_last_of",
         [](std::string_view h, std::string_view n) {
             auto match = h.find_last_of(n);
             return (match == std::string_view::npos ? 0 : match);
         }},
        {"sz_rfind_charset_serial", wrap_sz(sz_rfind_charset_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_rfind_charset_avx512", wrap_sz(sz_rfind_charset_avx512), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_rfind_charset_neon", wrap_sz(sz_rfind_charset_neon), true},
#endif
    };
    return result;
}

/**
 *  @brief  Evaluation for search string operations: find.
 */
void bench_finds(std::string const &haystack, std::vector<std::string> const &strings,
                 tracked_binary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_tokens(strings, [&](std::string_view needle) {
                std::string_view remaining = haystack;
                while (true) {
                    auto baseline = variants[0].function(remaining, needle);
                    auto result = variant.function(remaining, needle);
                    if (result != baseline) {
                        ++variant.failed_count;
                        if (variant.failed_strings.empty()) {
                            variant.failed_strings.push_back({remaining.data(), baseline + needle.size()});
                            variant.failed_strings.push_back({needle.data(), needle.size()});
                        }
                    }

                    if (baseline == remaining.size()) break;
                    remaining = remaining.substr(baseline + 1);
                }

                return haystack.size();
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = bench_on_tokens(strings, [&](std::string_view needle) {
                std::string_view remaining = haystack;
                auto offset_from_start = variant.function(remaining, needle);
                while (offset_from_start != remaining.size()) {
                    remaining = remaining.substr(offset_from_start + 1);
                    offset_from_start = variant.function(remaining, needle);
                    do_not_optimize(offset_from_start);
                }
                return haystack.size();
            });
        }

        variant.print();
    }
}

/**
 *  @brief  Evaluation for reverse order search string operations: find.
 */
void bench_rfinds(std::string const &haystack, std::vector<std::string> const &strings,
                  tracked_binary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_tokens(strings, [&](std::string_view needle) {
                std::string_view remaining = haystack;
                while (true) {
                    auto baseline = variants[0].function(remaining, needle);
                    auto result = variant.function(remaining, needle);
                    if (result != baseline) {
                        ++variant.failed_count;
                        if (variant.failed_strings.empty()) {
                            variant.failed_strings.push_back(
                                {remaining.data() + baseline, remaining.data() + remaining.size()});
                            variant.failed_strings.push_back({needle.data(), needle.size()});
                        }
                    }

                    if (baseline == remaining.size()) break;
                    remaining = remaining.substr(0, baseline);
                }

                return haystack.size();
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = bench_on_tokens(strings, [&](std::string_view needle) {
                std::string_view remaining = haystack;
                auto offset_from_start = variant.function(remaining, needle);
                while (offset_from_start != 0) {
                    remaining = remaining.substr(0, offset_from_start - 1);
                    offset_from_start = variant.function(remaining, needle);
                    do_not_optimize(offset_from_start);
                }
                return haystack.size();
            });
        }

        variant.print();
    }
}

void bench_search(std::string const &haystack, std::vector<std::string> const &strings) {
    if (strings.size() == 0) return;

    bench_finds(haystack, strings, find_functions());
    bench_rfinds(haystack, strings, rfind_functions());
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting search benchmarks.\n");

    dataset_t dataset = prepare_benchmark_environment(argc, argv);

    // Splitting by new lines
    std::printf("Benchmarking for a newline symbol:\n");
    bench_finds(dataset.text, {"\n"}, find_functions());
    bench_rfinds(dataset.text, {"\n"}, rfind_functions());

    std::printf("Benchmarking for one whitespace:\n");
    bench_finds(dataset.text, {" "}, find_functions());
    bench_rfinds(dataset.text, {" "}, rfind_functions());

    std::printf("Benchmarking for an [\\n\\r\\v\\f] RegEx:\n");
    bench_finds(dataset.text, {"\n\r\v\f"}, find_charset_functions());
    bench_rfinds(dataset.text, {"\n\r\v\f"}, rfind_charset_functions());

    // Typical ASCII tokenization and validation benchmarks
    std::printf("Benchmarking for all whitespaces:\n");
    bench_finds(dataset.text, {{sz::whitespaces(), sizeof(sz::whitespaces())}}, find_charset_functions());
    bench_rfinds(dataset.text, {{sz::whitespaces(), sizeof(sz::whitespaces())}}, rfind_charset_functions());

    std::printf("Benchmarking for HTML tag start/end:\n");
    bench_finds(dataset.text, {"<>"}, find_charset_functions());
    bench_rfinds(dataset.text, {"<>"}, rfind_charset_functions());

    std::printf("Benchmarking for punctuation marks:\n");
    bench_finds(dataset.text, {{sz::punctuation(), sizeof(sz::punctuation())}}, find_charset_functions());
    bench_rfinds(dataset.text, {{sz::punctuation(), sizeof(sz::punctuation())}}, rfind_charset_functions());

    std::printf("Benchmarking for non-printable characters:\n");
    bench_finds(dataset.text, {{sz::ascii_controls(), sizeof(sz::ascii_controls())}}, find_charset_functions());
    bench_rfinds(dataset.text, {{sz::ascii_controls(), sizeof(sz::ascii_controls())}}, rfind_charset_functions());

    // Baseline benchmarks for present tokens, coming in all lengths
    std::printf("Benchmarking on present lines:\n");
    bench_search(dataset.text, {dataset.lines.begin(), dataset.lines.end()});
    std::printf("Benchmarking on present tokens:\n");
    bench_search(dataset.text, {dataset.tokens.begin(), dataset.tokens.end()});

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on present tokens of length %zu:\n", token_length);
        bench_search(dataset.text, filter_by_length<std::string>(dataset.tokens, token_length));
    }

    // Run bechnmarks on abstract tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking for missing tokens of length %zu:\n", token_length);
        bench_search(dataset.text, std::vector<std::string> {
                                       std::string(token_length, '\1'),
                                       std::string(token_length, '\2'),
                                       std::string(token_length, '\3'),
                                       std::string(token_length, '\4'),
                                   });
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}