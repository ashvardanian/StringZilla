/**
 *  @file   bench_token.cpp
 *  @brief  Benchmarks token-level operations like hashing, equality, ordering, and copies.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_search.cpp` and `bench_similarity.cpp`.
 */
#include <bench.hpp>

using namespace ashvardanian::stringzilla::scripts;

tracked_unary_functions_t hashing_functions() {
    auto wrap_sz = [](auto function) -> unary_function_t {
        return unary_function_t([function](sz_string_view_t s) { return (sz_ssize_t)function(s.start, s.length); });
    };
    tracked_unary_functions_t result = {
        {"sz_hash_serial", wrap_sz(sz_hash_serial)},
#if SZ_USE_X86_AVX512
        {"sz_hash_avx512", wrap_sz(sz_hash_avx512), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_hash_neon", wrap_sz(sz_hash_neon), true},
#endif
        {"std::hash",
         [](sz_string_view_t s) {
             return (sz_ssize_t)std::hash<std::string_view> {}({s.start, s.length});
         }},
    };
    return result;
}

tracked_binary_functions_t equality_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](sz_string_view_t a, sz_string_view_t b) {
            return (sz_ssize_t)(a.length == b.length && function(a.start, b.start, a.length));
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.==",
         [](sz_string_view_t a, sz_string_view_t b) {
             return (sz_ssize_t)(std::string_view(a.start, a.length) == std::string_view(b.start, b.length));
         }},
        {"sz_equal_serial", wrap_sz(sz_equal_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_equal_avx512", wrap_sz(sz_equal_avx512), true},
#endif
        {"memcmp",
         [](sz_string_view_t a, sz_string_view_t b) {
             return (sz_ssize_t)(a.length == b.length && memcmp(a.start, b.start, a.length) == 0);
         }},
    };
    return result;
}

tracked_binary_functions_t ordering_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](sz_string_view_t a, sz_string_view_t b) {
            return (sz_ssize_t)function(a.start, a.length, b.start, b.length);
        });
    };
    tracked_binary_functions_t result = {
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
    return result;
}

template <typename strings_at>
void evaluate_all(strings_at &&strings) {
    if (strings.size() == 0) return;

    evaluate_unary_operations(strings, hashing_functions());
    evaluate_binary_operations(strings, equality_functions());
    evaluate_binary_operations(strings, ordering_functions());
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting token-level benchmarks.\n");

    dataset_t dataset = make_dataset(argc, argv);

    // Baseline benchmarks for real words, coming in all lengths
    std::printf("Benchmarking on real words:\n");
    evaluate_all(dataset.tokens);

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on real words of length %zu:\n", token_length);
        evaluate_all(dataset.tokens_of_length(token_length));
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}