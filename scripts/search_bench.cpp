#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <stringzilla/stringzilla.h>

using seconds_t = double;
using unary_function_t = std::function<sz_ssize_t(sz_string_view_t)>;
using binary_function_t = std::function<sz_ssize_t(sz_string_view_t, sz_string_view_t)>;

struct loop_over_words_result_t {
    std::size_t iterations = 0;
    std::size_t bytes_passed = 0;
    seconds_t seconds = 0;
};

/**
 *  @brief  Wrapper for a single execution backend.
 */
template <typename function_at>
struct tracked_function_gt {
    std::string name {""};
    function_at function {nullptr};
    bool needs_testing {false};

    std::size_t failed_count {0};
    std::vector<std::string> failed_strings {};
    loop_over_words_result_t results {};

    void print() const {
        char const *format;
        // Now let's print in the format:
        //  - name, up to 20 characters
        //  - throughput in GB/s with up to 3 significant digits, 10 characters
        //  - call latency in ns with up to 1 significant digit, 10 characters
        //  - number of failed tests, 10 characters
        //  - first example of a failed test, up to 20 characters
        if constexpr (std::is_same<function_at, binary_function_t>())
            format = "%-20s %10.3f GB/s %10.1f ns %10zu %s %s\n";
        else
            format = "%-20s %10.3f GB/s %10.1f ns %10zu %s\n";
        std::printf(format, name.c_str(), results.bytes_passed / results.seconds / 1.e9,
                    results.seconds * 1e9 / results.iterations, failed_count,
                    failed_strings.size() ? failed_strings[0].c_str() : "",
                    failed_strings.size() ? failed_strings[1].c_str() : "");
    }
};

using tracked_unary_functions_t = std::vector<tracked_function_gt<unary_function_t>>;
using tracked_binary_functions_t = std::vector<tracked_function_gt<binary_function_t>>;

#ifdef NDEBUG // Make debugging faster
#define run_tests_m 1
#define default_seconds_m 10
#else
#define run_tests_m 1
#define default_seconds_m 10
#endif

using temporary_memory_t = std::vector<char>;

std::string content_original;
std::vector<std::string> content_words;
std::vector<sz_error_cost_t> unary_substitution_costs;
temporary_memory_t temporary_memory;

template <typename value_at>
inline void do_not_optimize(value_at &&value) {
    asm volatile("" : "+r"(value) : : "memory");
}

inline sz_string_view_t sz_string_view(std::string const &str) { return {str.data(), str.size()}; };

sz_ptr_t _sz_memory_allocate_from_vector(sz_size_t length, void *handle) {
    temporary_memory_t &vec = *reinterpret_cast<temporary_memory_t *>(handle);
    if (vec.size() < length) vec.resize(length);
    return vec.data();
}

void _sz_memory_free_from_vector(sz_ptr_t buffer, sz_size_t length, void *handle) {}

std::string read_file(std::string path) {
    std::ifstream stream(path);
    if (!stream.is_open()) { throw std::runtime_error("Failed to open file: " + path); }
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

std::vector<std::string> tokenize(std::string_view str) {
    std::vector<std::string> words;
    std::size_t start = 0;
    for (std::size_t end = 0; end <= str.length(); ++end) {
        if (end == str.length() || std::isspace(str[end])) {
            if (start < end) words.push_back({&str[start], end - start});
            start = end + 1;
        }
    }
    return words;
}

sz_string_view_t random_slice(sz_string_view_t full_text, std::size_t min_length = 2, std::size_t max_length = 8) {
    std::size_t length = std::rand() % (max_length - min_length) + min_length;
    std::size_t offset = std::rand() % (full_text.length - length);
    return {full_text.start + offset, length};
}

std::size_t round_down_to_power_of_two(std::size_t n) {
    if (n == 0) return 0;
    std::size_t most_siginificant_bit_position = 0;
    while (n > 1) n >>= 1, most_siginificant_bit_position++;
    return static_cast<std::size_t>(1) << most_siginificant_bit_position;
}

tracked_unary_functions_t hashing_functions() {
    auto wrap_sz = [](auto function) -> unary_function_t {
        return unary_function_t([function](sz_string_view_t s) { return (sz_ssize_t)function(s.start, s.length); });
    };
    return {
        {"sz_hash_serial", wrap_sz(sz_hash_serial)},
#if SZ_USE_X86_AVX512
            {"sz_hash_avx512", wrap_sz(sz_hash_avx512), true},
#endif
#if SZ_USE_ARM_NEON
            {"sz_hash_neon", wrap_sz(sz_hash_neon), true},
#endif
            {"std::hash", [](sz_string_view_t s) {
                 return (sz_ssize_t)std::hash<std::string_view> {}({s.start, s.length});
             }},
    };
}

inline tracked_binary_functions_t equality_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](sz_string_view_t a, sz_string_view_t b) {
            return (sz_ssize_t)(a.length == b.length && function(a.start, b.start, a.length));
        });
    };
    return {
        {"std::string_view.==",
         [](sz_string_view_t a, sz_string_view_t b) {
             return (sz_ssize_t)(std::string_view(a.start, a.length) == std::string_view(b.start, b.length));
         }},
            {"sz_equal_serial", wrap_sz(sz_equal_serial), true},
#if SZ_USE_X86_AVX512
            {"sz_equal_avx512", wrap_sz(sz_equal_avx512), true},
#endif
            {"memcmp", [](sz_string_view_t a, sz_string_view_t b) {
                 return (sz_ssize_t)(a.length == b.length && memcmp(a.start, b.start, a.length) == 0);
             }},
    };
}

inline tracked_binary_functions_t ordering_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](sz_string_view_t a, sz_string_view_t b) {
            return (sz_ssize_t)function(a.start, a.length, b.start, b.length);
        });
    };
    return {
        {"std::string_view.compare",
         [](sz_string_view_t a, sz_string_view_t b) {
             auto order = std::string_view(a.start, a.length).compare(std::string_view(b.start, b.length));
             return (sz_ssize_t)(order == 0 ? sz_equal_k : (order < 0 ? sz_less_k : sz_greater_k));
         }},
        {"sz_order_serial", wrap_sz(sz_order_serial), true},
        {"memcmp",
         [](sz_string_view_t a, sz_string_view_t b) {
             auto order = memcmp(a.start, b.start, a.length < b.length ? a.length : b.length);
             return order != 0 ? (a.length == b.length ? (order < 0 ? sz_less_k : sz_greater_k)
                                                       : (a.length < b.length ? sz_less_k : sz_greater_k))
                               : sz_equal_k;
         }},
    };
}

inline tracked_binary_functions_t find_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](sz_string_view_t h, sz_string_view_t n) {
            sz_cptr_t match = function(h.start, h.length, n.start, n.length);
            return (sz_ssize_t)(match ? match - h.start : h.length);
        });
    };
    return {
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
            {"std::search<BMH>", [](sz_string_view_t h, sz_string_view_t n) {
                 auto match = std::search(h.start, h.start + h.length,
                                          std::boyer_moore_horspool_searcher(n.start, n.start + n.length));
                 return (sz_ssize_t)(match - h.start);
             }},
    };
}

inline tracked_binary_functions_t find_last_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](sz_string_view_t h, sz_string_view_t n) {
            sz_cptr_t match = function(h.start, h.length, n.start, n.length);
            return (sz_ssize_t)(match ? match - h.start : h.length);
        });
    };
    return {
        {"std::string_view.rfind",
         [](sz_string_view_t h, sz_string_view_t n) {
             auto h_view = std::string_view(h.start, h.length);
             auto n_view = std::string_view(n.start, n.length);
             auto match = h_view.rfind(n_view);
             return (sz_ssize_t)(match == std::string_view::npos ? h.length : match);
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
                 auto match = std::search(h_view.rbegin(), h_view.rend(),
                                          std::boyer_moore_searcher(n_view.rbegin(), n_view.rend()));
                 auto offset_from_end = (sz_ssize_t)(match - h_view.rbegin());
                 return h.length - offset_from_end;
             }},
            {"std::search<BMH>", [](sz_string_view_t h, sz_string_view_t n) {
                 auto h_view = std::string_view(h.start, h.length);
                 auto n_view = std::string_view(n.start, n.length);
                 auto match = std::search(h_view.rbegin(), h_view.rend(),
                                          std::boyer_moore_horspool_searcher(n_view.rbegin(), n_view.rend()));
                 auto offset_from_end = (sz_ssize_t)(match - h_view.rbegin());
                 return h.length - offset_from_end;
             }},
    };
}

inline tracked_binary_functions_t distance_functions() {
    // Populate the unary substitutions matrix
    static constexpr std::size_t max_length = 256;
    unary_substitution_costs.resize(max_length * max_length);
    for (std::size_t i = 0; i != max_length; ++i)
        for (std::size_t j = 0; j != max_length; ++j) unary_substitution_costs[i * max_length + j] = (i == j ? 0 : 1);

    // Two rows of the Levenshtein matrix will occupy this much:
    temporary_memory.resize((max_length + 1) * 2 * sizeof(sz_size_t));
    sz_memory_allocator_t alloc;
    alloc.allocate = _sz_memory_allocate_from_vector;
    alloc.free = _sz_memory_free_from_vector;
    alloc.handle = &temporary_memory;

    auto wrap_sz_distance = [alloc](auto function) -> binary_function_t {
        return binary_function_t([function, alloc](sz_string_view_t a, sz_string_view_t b) {
            a.length = sz_min_of_two(a.length, max_length);
            b.length = sz_min_of_two(b.length, max_length);
            return (sz_ssize_t)function(a.start, a.length, b.start, b.length, max_length, &alloc);
        });
    };
    auto wrap_sz_scoring = [alloc](auto function) -> binary_function_t {
        return binary_function_t([function, alloc](sz_string_view_t a, sz_string_view_t b) {
            a.length = sz_min_of_two(a.length, max_length);
            b.length = sz_min_of_two(b.length, max_length);
            return (sz_ssize_t)function(a.start, a.length, b.start, b.length, 1, unary_substitution_costs.data(),
                                        &alloc);
        });
    };
    return {
        {"sz_levenshtein", wrap_sz_distance(sz_levenshtein_serial)},
            {"sz_alignment_score", wrap_sz_scoring(sz_alignment_score_serial), true},
#if SZ_USE_X86_AVX512
            {"sz_levenshtein_avx512", wrap_sz_distance(sz_levenshtein_avx512), true},
#endif
    };
}

/**
 *  @brief  Loop over all elements in a dataset in somewhat random order, benchmarking the function cost.
 *  @param  strings Strings to loop over. Length must be a power of two.
 *  @param  function Function to be applied to each `sz_string_view_t`. Must return the number of bytes processed.
 *  @return Number of seconds per iteration.
 */
template <typename strings_at, typename function_at>
loop_over_words_result_t loop_over_words(strings_at &&strings, function_at &&function,
                                         seconds_t max_time = default_seconds_m) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();
    loop_over_words_result_t result;
    std::size_t lookup_mask = round_down_to_power_of_two(strings.size()) - 1;

    while (true) {
        // Unroll a few iterations, to avoid some for-loops overhead and minimize impact of time-tracking
        {
            result.bytes_passed += function(sz_string_view(strings[(++result.iterations) & lookup_mask]));
            result.bytes_passed += function(sz_string_view(strings[(++result.iterations) & lookup_mask]));
            result.bytes_passed += function(sz_string_view(strings[(++result.iterations) & lookup_mask]));
            result.bytes_passed += function(sz_string_view(strings[(++result.iterations) & lookup_mask]));
        }

        stdcc::time_point t2 = stdcc::now();
        result.seconds = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() / 1.e9;
        if (result.seconds > max_time) break;
    }

    return result;
}

/**
 *  @brief  Evaluation for unary string operations: hashing.
 */
template <typename strings_at>
void evaluate_unary_operations(strings_at &&strings, tracked_unary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            loop_over_words(strings, [&](sz_string_view_t str) {
                auto baseline = variants[0].function(str);
                auto result = variant.function(str);
                if (result != baseline) {
                    ++variant.failed_count;
                    if (variant.failed_strings.empty()) { variant.failed_strings.push_back({str.start, str.length}); }
                }
                return str.length;
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = loop_over_words(strings, [&](sz_string_view_t str) {
                do_not_optimize(variant.function(str));
                return str.length;
            });
        }

        variant.print();
    }
}

/**
 *  @brief  Loop over all elements in a dataset, benchmarking the function cost.
 *  @param  strings Strings to loop over. Length must be a power of two.
 *  @param  function Function to be applied to pairs of `sz_string_view_t`. Must return the number of bytes
 * processed.
 *  @return Number of seconds per iteration.
 */
template <typename strings_at, typename function_at>
loop_over_words_result_t loop_over_pairs_of_words(strings_at &&strings, function_at &&function,
                                                  seconds_t max_time = default_seconds_m) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();
    loop_over_words_result_t result;
    std::size_t lookup_mask = round_down_to_power_of_two(strings.size()) - 1;

    while (true) {
        // Unroll a few iterations, to avoid some for-loops overhead and minimize impact of time-tracking
        {
            result.bytes_passed +=
                function(sz_string_view(strings[(++result.iterations) & lookup_mask]),
                         sz_string_view(strings[(result.iterations * 18446744073709551557ull) & lookup_mask]));
            result.bytes_passed +=
                function(sz_string_view(strings[(++result.iterations) & lookup_mask]),
                         sz_string_view(strings[(result.iterations * 18446744073709551557ull) & lookup_mask]));
            result.bytes_passed +=
                function(sz_string_view(strings[(++result.iterations) & lookup_mask]),
                         sz_string_view(strings[(result.iterations * 18446744073709551557ull) & lookup_mask]));
            result.bytes_passed +=
                function(sz_string_view(strings[(++result.iterations) & lookup_mask]),
                         sz_string_view(strings[(result.iterations * 18446744073709551557ull) & lookup_mask]));
        }

        stdcc::time_point t2 = stdcc::now();
        result.seconds = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() / 1.e9;
        if (result.seconds > max_time) break;
    }

    return result;
}

/**
 *  @brief  Evaluation for binary string operations: equality, ordering, prefix, suffix, distance.
 */
template <typename strings_at>
void evaluate_binary_operations(strings_at &&strings, tracked_binary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            loop_over_pairs_of_words(strings, [&](sz_string_view_t str_a, sz_string_view_t str_b) {
                auto baseline = variants[0].function(str_a, str_b);
                auto result = variant.function(str_a, str_b);
                if (result != baseline) {
                    ++variant.failed_count;
                    if (variant.failed_strings.empty()) {
                        variant.failed_strings.push_back({str_a.start, str_a.length});
                        variant.failed_strings.push_back({str_b.start, str_b.length});
                    }
                }
                return str_a.length + str_b.length;
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = loop_over_pairs_of_words(strings, [&](sz_string_view_t str_a, sz_string_view_t str_b) {
                do_not_optimize(variant.function(str_a, str_b));
                return str_a.length + str_b.length;
            });
        }

        variant.print();
    }
}

/**
 *  @brief  Evaluation for search string operations: find.
 */
template <typename strings_at>
void evaluate_find_operations(strings_at &&strings, tracked_binary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            loop_over_words(strings, [&](sz_string_view_t str_n) {
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
            variant.results = loop_over_words(strings, [&](sz_string_view_t str_n) {
                sz_string_view_t str_h = {content_original.data(), content_original.size()};
                auto result = variant.function(str_h, str_n);
                while (result != str_h.length) {
                    str_h.start += result + 1, str_h.length -= result + 1;
                    result = variant.function(str_h, str_n);
                    do_not_optimize(result);
                }
                return result;
            });
        }

        variant.print();
    }
}

/**
 *  @brief  Evaluation for reverse order search string operations: find.
 */
template <typename strings_at>
void evaluate_find_last_operations(strings_at &&strings, tracked_binary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            loop_over_words(strings, [&](sz_string_view_t str_n) {
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
            variant.results = loop_over_words(strings, [&](sz_string_view_t str_n) {
                sz_string_view_t str_h = {content_original.data(), content_original.size()};
                str_h.length -= bytes_processed & mask;
                auto result = variant.function(str_h, str_n);
                bytes_processed += (str_h.length - result) + str_n.length;
                return result;
            });
        }

        variant.print();
    }
}

template <typename strings_at>
void evaluate_all_operations(strings_at &&strings) {
    evaluate_unary_operations(strings, hashing_functions());
    evaluate_binary_operations(strings, equality_functions());
    evaluate_binary_operations(strings, ordering_functions());
    evaluate_binary_operations(strings, distance_functions());
    evaluate_find_operations(strings, find_functions());
    evaluate_find_last_operations(strings, find_last_functions());

    // evaluate_binary_operations(strings, prefix_functions());
    // evaluate_binary_operations(strings, suffix_functions());
}

int main(int, char const **) {
    std::printf("Hi Ash! ... or is it someone else?!\n");

    content_original = read_file("leipzig1M.txt");
    content_original.resize(round_down_to_power_of_two(content_original.size()));

    content_words = tokenize(content_original);
    content_words.resize(round_down_to_power_of_two(content_words.size()));

#ifdef NDEBUG // Shuffle only in release mode
    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    std::shuffle(content_words.begin(), content_words.end(), random_generator);
#endif

    // Report some basic stats about the dataset
    std::size_t mean_bytes = 0;
    for (auto const &str : content_words) mean_bytes += str.size();
    mean_bytes /= content_words.size();
    std::printf("Parsed the file with %zu words of %zu mean length!\n", content_words.size(), mean_bytes);

    // Baseline benchmarks for real words, coming in all lengths
    {
        std::printf("Benchmarking for real words:\n");
        evaluate_all_operations(content_words);
    }

    // Produce benchmarks for different word lengths, both real and impossible
    for (std::size_t word_length : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 33, 65}) {

        // Generate some impossible words of that length
        std::printf("\n\n");
        std::printf("Benchmarking for abstract tokens of length %zu:\n", word_length);
        std::vector<std::string> words = {
            std::string(word_length, '\1'),
            std::string(word_length, '\2'),
            std::string(word_length, '\3'),
            std::string(word_length, '\4'),
        };
        evaluate_all_operations(words);

        // Check for some real words of that length
        for (auto const &str : words)
            if (str.size() == word_length) words.push_back(str);
        if (!words.size()) continue;
        std::printf("Benchmarking for real words of length %zu:\n", word_length);
        evaluate_all_operations(words);
    }

    // Now lets test our functionality on longer biological sequences.
    // A single human gene is from 300 to 15,000 base pairs long.
    // Thole whole human genome is about 3 billion base pairs long.
    // The genomes of bacteria are relatively small - E. coli genome is about 4.6 million base pairs long.
    // In techniques like PCR (Polymerase Chain Reaction), short DNA sequences called primers are used.
    // These are usually 18 to 25 base pairs long.
    char aminoacids[] = "ATCG";
    for (std::size_t dna_length : {300, 2000, 15000}) {
        std::vector<std::string> dna_sequences(16);
        for (std::size_t i = 0; i != 16; ++i) {
            dna_sequences[i].resize(dna_length);
            for (std::size_t j = 0; j != dna_length; ++j) dna_sequences[i][j] = aminoacids[std::rand() % 4];
        }
        std::printf("Benchmarking for DNA-like sequences of length %zu:\n", dna_length);
        evaluate_all_operations(dna_sequences);
    }

    return 0;
}