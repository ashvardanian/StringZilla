/**
 *  @file   bench_search.cpp
 *  @brief  Benchmarks for bidirectional string search operations - exact and TODO: approximate.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_token.cpp` and `bench_similarity.cpp`.
 *  It accepts a file with a list of words, and benchmarks the search operations on them.
 *  Outside of present tokens also tries missing tokens.
 */
#include <bench.hpp>

using namespace ashvardanian::stringzilla::scripts;

tracked_binary_functions_t find_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](sz_string_view_t h, sz_string_view_t n) {
            sz_cptr_t match = function(h.start, h.length, n.start, n.length);
            return (sz_ssize_t)(match ? match - h.start : h.length);
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.find",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto h_view = std::string_view(h.start, h.length);
             auto n_view = std::string_view(n.start, n.length);
             auto match = h_view.find(n_view);
             return (sz_ssize_t)(match == std::string_view::npos ? h.length : match);
         }},
        {"sz_find_serial", wrap_sz(sz_find_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_find_avx512", wrap_sz(sz_find_avx512), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_find_neon", wrap_sz(sz_find_neon), true},
#endif
        {"strstr",
         [](sz_string_view_t h, sz_string_view_t n) {
             sz_cptr_t match = strstr(h.start, n.start);
             return (sz_ssize_t)(match ? match - h.start : h.length);
         }},
        {"std::search",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto match = std::search(h.start, h.start + h.length, n.start, n.start + n.length);
             return (sz_ssize_t)(match - h.start);
         }},
        {"std::search<BM>",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto match =
                 std::search(h.start, h.start + h.length, std::boyer_moore_searcher(n.start, n.start + n.length));
             return (sz_ssize_t)(match - h.start);
         }},
        {"std::search<BMH>",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto match = std::search(h.start, h.start + h.length,
                                      std::boyer_moore_horspool_searcher(n.start, n.start + n.length));
             return (sz_ssize_t)(match - h.start);
         }},
    };
    return result;
}

tracked_binary_functions_t find_last_functions() {
    // TODO: Computing throughput seems wrong
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](sz_string_view_t h, sz_string_view_t n) {
            sz_cptr_t match = function(h.start, h.length, n.start, n.length);
            return (sz_ssize_t)(match ? match - h.start : 0);
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.rfind",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto h_view = std::string_view(h.start, h.length);
             auto n_view = std::string_view(n.start, n.length);
             auto match = h_view.rfind(n_view);
             return (sz_ssize_t)(match == std::string_view::npos ? 0 : match);
         }},
        {"sz_find_last_serial", wrap_sz(sz_find_last_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_find_last_avx512", wrap_sz(sz_find_last_avx512), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_find_last_neon", wrap_sz(sz_find_last_neon), true},
#endif
        {"std::search",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto h_view = std::string_view(h.start, h.length);
             auto n_view = std::string_view(n.start, n.length);
             auto match = std::search(h_view.rbegin(), h_view.rend(), n_view.rbegin(), n_view.rend());
             auto offset_from_end = (sz_ssize_t)(match - h_view.rbegin());
             return h.length - offset_from_end;
         }},
        {"std::search<BM>",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto h_view = std::string_view(h.start, h.length);
             auto n_view = std::string_view(n.start, n.length);
             auto match =
                 std::search(h_view.rbegin(), h_view.rend(), std::boyer_moore_searcher(n_view.rbegin(), n_view.rend()));
             auto offset_from_end = (sz_ssize_t)(match - h_view.rbegin());
             return h.length - offset_from_end;
         }},
        {"std::search<BMH>",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto h_view = std::string_view(h.start, h.length);
             auto n_view = std::string_view(n.start, n.length);
             auto match = std::search(h_view.rbegin(), h_view.rend(),
                                      std::boyer_moore_horspool_searcher(n_view.rbegin(), n_view.rend()));
             auto offset_from_end = (sz_ssize_t)(match - h_view.rbegin());
             return h.length - offset_from_end;
         }},
    };
    return result;
}

/**
 *  @brief  Evaluation for search string operations: find.
 */
template <typename strings_at>
void evaluate_find_operations(std::string_view content_original, strings_at &&strings,
                              tracked_binary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_tokens(strings, [&](sz_string_view_t str_n) {
                sz_string_view_t str_h = {content_original.data(), content_original.size()};
                while (true) {
                    auto baseline = variants[0].function(str_h, str_n);
                    auto result = variant.function(str_h, str_n);
                    if (result != baseline) {
                        ++variant.failed_count;
                        if (variant.failed_strings.empty()) {
                            variant.failed_strings.push_back({str_h.start, baseline + str_n.length});
                            variant.failed_strings.push_back({str_n.start, str_n.length});
                        }
                    }

                    if (baseline == str_h.length) break;
                    str_h.start += baseline + 1;
                    str_h.length -= baseline + 1;
                }

                return content_original.size();
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = bench_on_tokens(strings, [&](sz_string_view_t str_n) {
                sz_string_view_t str_h = {content_original.data(), content_original.size()};
                auto offset_from_start = variant.function(str_h, str_n);
                while (offset_from_start != str_h.length) {
                    str_h.start += offset_from_start + 1, str_h.length -= offset_from_start + 1;
                    offset_from_start = variant.function(str_h, str_n);
                    do_not_optimize(offset_from_start);
                }
                return str_h.length;
            });
        }

        variant.print();
    }
}

/**
 *  @brief  Evaluation for reverse order search string operations: find.
 */
template <typename strings_at>
void evaluate_find_last_operations(std::string_view content_original, strings_at &&strings,
                                   tracked_binary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_tokens(strings, [&](sz_string_view_t str_n) {
                sz_string_view_t str_h = {content_original.data(), content_original.size()};
                while (true) {
                    auto baseline = variants[0].function(str_h, str_n);
                    auto result = variant.function(str_h, str_n);
                    if (result != baseline) {
                        ++variant.failed_count;
                        if (variant.failed_strings.empty()) {
                            variant.failed_strings.push_back({str_h.start + baseline, str_h.start + str_h.length});
                            variant.failed_strings.push_back({str_n.start, str_n.length});
                        }
                    }

                    if (baseline == str_h.length) break;
                    str_h.length = baseline;
                }

                return content_original.size();
            });
        }

        // Benchmarks
        if (variant.function) {
            std::size_t bytes_processed = 0;
            std::size_t mask = content_original.size() - 1;
            variant.results = bench_on_tokens(strings, [&](sz_string_view_t str_n) {
                sz_string_view_t str_h = {content_original.data(), content_original.size()};
                auto offset_from_start = variant.function(str_h, str_n);
                while (offset_from_start != 0) {
                    str_h.length = offset_from_start - 1;
                    offset_from_start = variant.function(str_h, str_n);
                    do_not_optimize(offset_from_start);
                }
                return str_h.length;
            });
        }

        variant.print();
    }
}

template <typename strings_at>
void evaluate_all(std::string_view content_original, strings_at &&strings) {
    if (strings.size() == 0) return;

    evaluate_find_operations(content_original, strings, find_functions());
    evaluate_find_last_operations(content_original, strings, find_last_functions());
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting search benchmarks.\n");

    dataset_t dataset = make_dataset(argc, argv);

    // Baseline benchmarks for real words, coming in all lengths
    std::printf("Benchmarking on real words:\n");
    evaluate_all(dataset.text, dataset.tokens);

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on real words of length %zu:\n", token_length);
        evaluate_all(dataset.text, filter_by_length(dataset.tokens, token_length));
    }

    // Run bechnmarks on abstract tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking for missing tokens of length %zu:\n", token_length);
        evaluate_all(dataset.text, std::vector<std::string> {
                                       std::string(token_length, '\1'),
                                       std::string(token_length, '\2'),
                                       std::string(token_length, '\3'),
                                       std::string(token_length, '\4'),
                                   });
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}