#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <strstream>
#include <vector>

#include <stringzilla/stringzilla.h>

using seconds_t = double;

std::string content_original;
std::vector<std::string> content_tokens;
#define run_tests_m 1
#define default_seconds_m 1

std::string read_file(std::string path) {
    std::ifstream stream(path);
    if (!stream.is_open()) { throw std::runtime_error("Failed to open file: " + path); }
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

std::vector<std::string> tokenize(std::string_view str) {
    std::vector<std::string> tokens;
    std::size_t start = 0;
    for (std::size_t end = 0; end <= str.length(); ++end) {
        if (end == str.length() || std::isspace(str[end])) {
            if (start < end) tokens.push_back({&str[start], end - start});
            start = end + 1;
        }
    }
    return tokens;
}

sz_string_view_t random_slice(sz_string_view_t full_text, std::size_t min_length = 2, std::size_t max_length = 8) {
    std::size_t length = std::rand() % (max_length - min_length) + min_length;
    std::size_t offset = std::rand() % (full_text.length - length);
    return {full_text.start + offset, length};
}

std::size_t round_down_to_power_of_two(std::size_t n) {
    if (n == 0) return 0;
    std::size_t most_siginificant_bit_pisition = 0;
    while (n > 1) n >>= 1, most_siginificant_bit_pisition++;
    return static_cast<std::size_t>(1) << most_siginificant_bit_pisition;
}

template <typename value_at>
inline void do_not_optimize(value_at &&value) {
    asm volatile("" : "+r"(value) : : "memory");
}

struct loop_over_tokens_result_t {
    std::size_t iterations = 0;
    std::size_t bytes_passed = 0;
    seconds_t seconds = 0;

    void print() {
        std::printf("--- took %.2lf ns/it ~ %.2f GB/s\n", seconds * 1e9 / iterations, bytes_passed / seconds / 1.e9);
    }
};

/**
 *  @brief  Loop over all elements in a dataset in somewhat random order, benchmarking the callback cost.
 *  @param  strings Strings to loop over. Length must be a power of two.
 *  @param  callback Function to be applied to each `sz_string_view_t`. Must return the number of bytes processed.
 *  @return Number of seconds per iteration.
 */
template <typename strings_at, typename callback_at>
loop_over_tokens_result_t loop_over_tokens(strings_at &&strings, callback_at &&callback,
                                           seconds_t max_time = default_seconds_m,
                                           std::size_t repetitions_between_checks = 16) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();
    loop_over_tokens_result_t result;
    std::size_t strings_count = round_down_to_power_of_two(strings.size());

    while (true) {
        for (std::size_t i = 0; i != repetitions_between_checks; ++i, ++result.iterations) {
            std::string const &str = strings[result.iterations & (strings_count - 1)];
            result.bytes_passed += callback({str.data(), str.size()});
        }

        stdcc::time_point t2 = stdcc::now();
        result.seconds = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() / 1.e9;
        if (result.seconds > max_time) break;
    }

    return result;
}

/**
 *  @brief  Loop over all elements in a dataset, benchmarking the callback cost.
 *  @param  strings Strings to loop over. Length must be a power of two.
 *  @param  callback Function to be applied to pairs of `sz_string_view_t`. Must return the number of bytes processed.
 *  @return Number of seconds per iteration.
 */
template <typename strings_at, typename callback_at>
loop_over_tokens_result_t loop_over_pairs_of_tokens(strings_at &&strings, callback_at &&callback,
                                                    seconds_t max_time = default_seconds_m,
                                                    std::size_t repetitions_between_checks = 16) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();
    loop_over_tokens_result_t result;
    std::size_t strings_count = round_down_to_power_of_two(strings.size());

    while (true) {
        for (std::size_t i = 0; i != repetitions_between_checks; ++i, ++result.iterations) {
            std::size_t offset = result.iterations & (strings_count - 1);
            std::string const &str_a = strings[offset];
            std::string const &str_b = strings[strings_count - offset - 1];
            result.bytes_passed += callback({str_a.data(), str_a.size()}, {str_b.data(), str_b.size()});
        }

        stdcc::time_point t2 = stdcc::now();
        result.seconds = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() / 1.e9;
        if (result.seconds > max_time) break;
    }

    return result;
}

/**
 *  @brief  For an array of tokens benchmarks hashing performance.
 *          Sadly has no baselines, as LibC doesn't provide hashing capabilities out of the box.
 */
struct case_hashing_t {

    static sz_u32_t baseline_stl(sz_cptr_t text, sz_size_t length) {
        return std::hash<std::string_view> {}({text, length});
    }

    std::vector<std::pair<std::string, sz_crc32_t>> variants = {
        {"sz_crc32_serial", &sz_crc32_serial},
        // {"sz_crc32_avx512", SZ_USE_X86_AVX512 ? sz_crc32_avx512 : NULL},
        {"sz_crc32_sse42", SZ_USE_X86_SSE42 ? sz_crc32_sse42 : NULL},
        {"sz_crc32_arm", SZ_USE_ARM_CRC32 ? sz_crc32_arm : NULL},
        {"std::hash", &baseline_stl},
    };

    template <typename strings_at>
    void operator()(strings_at &&strings) {

        std::printf("- Hashing words \n");

        // First iterate over all the strings and make sure, the same hash is reported for every candidate
        if (false) {
            loop_over_tokens(strings, [&](sz_string_view_t str) {
                auto baseline = variants[0].second(str.start, str.length);
                for (auto const &[name, variant] : variants) {
                    if (!variant) continue;
                    auto result = variant(str.start, str.length);
                    if (result != baseline) throw std::runtime_error("Result mismatch!");
                }
                return str.length;
            });
            std::printf("-- tests passed! \n");
        }

        // Then iterate over all strings reporting benchmark results for each non-NULL backend.
        for (auto const &[name, variant] : variants) {
            if (!variant) continue;
            std::printf("-- %s \n", name.c_str());
            loop_over_tokens(strings, [&](sz_string_view_t str) {
                do_not_optimize(variant(str.start, str.length));
                return str.length;
            }).print();
        }
    }
};

struct case_find_t {

    std::string case_name;

    static sz_cptr_t baseline_std_search(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
        auto result = std::search(h, h + h_length, n, n + n_length);
        return result == h + h_length ? NULL : result;
    }

    static sz_cptr_t baseline_std_string(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
        auto h_view = std::string_view(h, h_length);
        auto n_view = std::string_view(n, n_length);
        auto result = h_view.find(n_view);
        return result == std::string_view::npos ? NULL : h + result;
    }

    static sz_cptr_t baseline_libc(sz_cptr_t h, sz_size_t, sz_cptr_t n, sz_size_t) { return strstr(h, n); }

    struct variant_t {
        std::string name;
        sz_find_t function = NULL;
        bool needs_testing = false;
    };

    std::vector<variant_t> variants = {
        {"std::string_view.find", &baseline_std_string, false},
        {"sz_find_serial", &sz_find_serial, true},
        {"sz_find_avx512", SZ_USE_X86_AVX512 ? sz_find_avx512 : NULL, true},
        {"sz_find_avx2", SZ_USE_X86_AVX2 ? sz_find_avx2 : NULL, true},
        // {"sz_find_neon", SZ_USE_ARM_NEON ? sz_find_neon : NULL, true},
        {"strstr", &baseline_libc},
        {"std::search", &baseline_std_search},
    };

    void scan_through_whole_dataset(sz_find_t finder, sz_string_view_t needle) {
        sz_string_view_t remaining = {content_original.data(), content_original.size()};
        while (true) {
            auto result = finder(remaining.start, remaining.length, needle.start, needle.length);
            if (!result) break;
            remaining.start = result + needle.length;
            remaining.length = content_original.data() + content_original.size() - remaining.start;
        }
    }

    void test_through_whole_dataset(sz_find_t checker, sz_find_t finder, sz_string_view_t needle) {
        sz_string_view_t remaining = {content_original.data(), content_original.size()};
        while (true) {
            auto baseline = checker(remaining.start, remaining.length, needle.start, needle.length);
            auto result = finder(remaining.start, remaining.length, needle.start, needle.length);
            if (result != baseline) throw std::runtime_error("Result mismatch!");

            if (!result) break;
            remaining.start = result + needle.length;
            remaining.length = content_original.data() + content_original.size() - remaining.start;
        }
    }

    template <typename strings_at>
    void operator()(strings_at &&strings) {

        std::printf("- Searching substrings - %s \n", case_name.c_str());
        sz_string_view_t content_view = {content_original.data(), content_original.size()};

#if run_tests_m
        // First iterate over all the strings and make sure, the same hash is reported for every candidate
        for (std::size_t variant_idx = 1; variant_idx != variants.size(); ++variant_idx) {
            variant_t const &variant = variants[variant_idx];
            if (!variant.function || !variant.needs_testing) continue;
            loop_over_tokens(strings, [&](sz_string_view_t str) {
                test_through_whole_dataset(variants[0].function, variant.function, str);
                return content_view.length;
            });
            std::printf("-- %s tests passed! \n", variant.name.c_str());
        }
#endif

        // Then iterate over all strings reporting benchmark results for each non-NULL backend.
        for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
            variant_t const &variant = variants[variant_idx];
            if (!variant.function) continue;
            std::printf("-- %s \n", variant.name.c_str());
            loop_over_tokens(strings, [&](sz_string_view_t str) {
                // Just running `variant(content_view.start, content_view.length, str.start, str.length)`
                // may not be sufficient, as different strings are represented with different frequency.
                // Enumerating the matches in the whole dataset would yield more stable numbers.
                scan_through_whole_dataset(variant.function, str);
                return content_view.length;
            }).print();
        }
    }
};

struct case_order_t {

    std::string case_name;

    static sz_ordering_t baseline_std_string(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
        auto a_view = std::string_view(a, a_length);
        auto b_view = std::string_view(b, b_length);
        auto order = a_view.compare(b_view);
        return order != 0 ? (order < 0 ? sz_less_k : sz_greater_k) : sz_equal_k;
    }

    static sz_ordering_t baseline_libc(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
        auto order = memcmp(a, b, a_length < b_length ? a_length : b_length);
        return order != 0 ? (a_length == b_length ? (order < 0 ? sz_less_k : sz_greater_k)
                                                  : (a_length < b_length ? sz_less_k : sz_greater_k))
                          : sz_equal_k;
    }

    struct variant_t {
        std::string name;
        sz_order_t function = NULL;
        bool needs_testing = false;
    };

    std::vector<variant_t> variants = {
        {"std::string.compare", &baseline_std_string},
        {"sz_order_serial", &sz_order_serial, true},
        {"sz_order_avx512", SZ_USE_X86_AVX512 ? sz_order_avx512 : NULL, true},
        {"memcmp", &baseline_libc},
    };

    template <typename strings_at>
    void operator()(strings_at &&strings) {

        std::printf("- Comparing order of strings - %s \n", case_name.c_str());

#if run_tests_m
        // First iterate over all the strings and make sure, the same hash is reported for every candidate
        for (std::size_t variant_idx = 1; variant_idx != variants.size(); ++variant_idx) {
            variant_t const &variant = variants[variant_idx];
            if (!variant.function || !variant.needs_testing) continue;
            loop_over_pairs_of_tokens(strings, [&](sz_string_view_t str_a, sz_string_view_t str_b) {
                auto baseline = variants[0].function(str_a.start, str_a.length, str_b.start, str_b.length);
                auto result = variant.function(str_a.start, str_a.length, str_b.start, str_b.length);
                if (result != baseline) throw std::runtime_error("Result mismatch!");
                return str_a.length + str_b.length;
            });
            std::printf("-- %s tests passed! \n", variant.name.c_str());
        }
#endif

        // Then iterate over all strings reporting benchmark results for each non-NULL backend.
        for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
            variant_t const &variant = variants[variant_idx];
            if (!variant.function) continue;
            std::printf("-- %s \n", variant.name.c_str());
            loop_over_pairs_of_tokens(strings, [&](sz_string_view_t str_a, sz_string_view_t str_b) {
                do_not_optimize(variant.function(str_a.start, str_a.length, str_b.start, str_b.length));
                return str_a.length + str_b.length;
            }).print();
        }
    }
};

struct case_equality_t {

    std::string case_name;

    static sz_bool_t baseline_std_string(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
        auto a_view = std::string_view(a, length);
        auto b_view = std::string_view(b, length);
        return (sz_bool_t)(a_view == b_view);
    }

    static sz_bool_t baseline_libc(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
        return (sz_bool_t)(memcmp(a, b, length) == 0);
    }

    struct variant_t {
        std::string name;
        sz_equal_t function = NULL;
        bool needs_testing = false;
    };

    std::vector<variant_t> variants = {
        {"std::string.==", &baseline_std_string},
        {"sz_equal_serial", &sz_equal_serial, true},
        {"sz_equal_avx512", SZ_USE_X86_AVX512 ? sz_equal_avx512 : NULL, true},
        {"memcmp", &baseline_libc},
    };

    template <typename strings_at>
    void operator()(strings_at &&strings) {

        std::printf("- Comparing equality of strings - %s \n", case_name.c_str());

#if run_tests_m
        // First iterate over all the strings and make sure, the same hash is reported for every candidate
        for (std::size_t variant_idx = 1; variant_idx != variants.size(); ++variant_idx) {
            variant_t const &variant = variants[variant_idx];
            if (!variant.function || !variant.needs_testing) continue;
            loop_over_pairs_of_tokens(strings, [&](sz_string_view_t str_a, sz_string_view_t str_b) {
                if (str_a.length != str_b.length) return str_a.length + str_b.length;
                auto baseline = variants[0].function(str_a.start, str_b.start, str_b.length);
                auto result = variant.function(str_a.start, str_b.start, str_b.length);
                if (result != baseline) throw std::runtime_error("Result mismatch!");
                return str_a.length + str_b.length;
            });
            std::printf("-- %s tests passed! \n", variant.name.c_str());
        }
#endif

        // Then iterate over all strings reporting benchmark results for each non-NULL backend.
        for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
            variant_t const &variant = variants[variant_idx];
            if (!variant.function) continue;
            std::printf("-- %s \n", variant.name.c_str());
            loop_over_pairs_of_tokens(strings, [&](sz_string_view_t str_a, sz_string_view_t str_b) {
                if (str_a.length != str_b.length) return str_a.length + str_b.length;
                do_not_optimize(variant.function(str_a.start, str_b.start, str_b.length));
                return str_a.length + str_b.length;
            }).print();
        }
    }
};

int main(int, char const **) {
    std::printf("Hi Ash! ... or is it someone else?!\n");

    content_original = read_file("leipzig1M.txt");
    content_tokens = tokenize(content_original);

#ifdef NDEBUG // Shuffle only in release mode
    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    std::shuffle(content_tokens.begin(), content_tokens.end(), random_generator);
#endif

    // Report some basic stats about the dataset
    std::size_t mean_bytes = 0;
    for (auto const &str : content_tokens) mean_bytes += str.size();
    mean_bytes /= content_tokens.size();
    std::printf("Parsed the file with %zu words of %zu mean length!\n", content_tokens.size(), mean_bytes);

    // Handle basic operations over exisating words
    case_find_t {"words"}(content_tokens);
    case_hashing_t {}(content_tokens);
    case_order_t {"words"}(content_tokens);
    case_equality_t {"words"}(content_tokens);

    // Produce benchmarks for different token lengths
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 33, 65}) {
        std::vector<std::string> tokens;
        for (auto const &str : content_tokens)
            if (str.size() == token_length) tokens.push_back(str);

        if (tokens.size()) {
            case_find_t {"words of length " + std::to_string(token_length)}(tokens);
            case_order_t {"words of length " + std::to_string(token_length)}(tokens);
            case_equality_t {"words of length " + std::to_string(token_length)}(tokens);
        }

        // Generate some impossible tokens of that length
        std::string impossible_token_one = std::string(token_length, '\1');
        std::string impossible_token_two = std::string(token_length, '\2');
        std::string impossible_token_three = std::string(token_length, '\3');
        std::string impossible_token_four = std::string(token_length, '\4');
        tokens = {impossible_token_one, impossible_token_two, impossible_token_three, impossible_token_four};

        case_find_t {"missing words of length " + std::to_string(token_length)}(tokens);
        case_order_t {"missing words of length " + std::to_string(token_length)}(tokens);
        case_equality_t {"missing words of length " + std::to_string(token_length)}(tokens);
    }

    return 0;
}