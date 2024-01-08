/**
 *  @file   bench_token.cpp
 *  @brief  Benchmarks token-level operations like hashing, equality, ordering, and copies.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_search.cpp` and `bench_similarity.cpp`.
 */
#include <bench.hpp>
#include <test.hpp> // `random_string`

using namespace ashvardanian::stringzilla::scripts;

tracked_unary_functions_t hashing_functions() {
    auto wrap_sz = [](auto function) -> unary_function_t {
        return unary_function_t([function](std::string_view s) { return function(s.data(), s.size()); });
    };
    tracked_unary_functions_t result = {
        {"sz_hash_serial", wrap_sz(sz_hash_serial)},
#if SZ_USE_X86_AVX512
        {"sz_hash_avx512", wrap_sz(sz_hash_avx512), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_hash_neon", wrap_sz(sz_hash_neon), true},
#endif
        {"std::hash", [](std::string_view s) { return std::hash<std::string_view> {}(s); }},
    };
    return result;
}

tracked_unary_functions_t random_generation_functions(std::size_t token_length) {

    tracked_unary_functions_t result = {
        {"random std::string" + std::to_string(token_length),
         unary_function_t([token_length](std::string_view alphabet) -> std::size_t {
             return random_string(token_length, alphabet.data(), alphabet.size()).size();
         })},
        {"random sz::string" + std::to_string(token_length),
         unary_function_t([token_length](std::string_view alphabet) -> std::size_t {
             return sz::string::random(token_length, alphabet).size();
         })},
    };
    return result;
}

tracked_binary_functions_t equality_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](std::string_view a, std::string_view b) {
            return (a.size() == b.size() && function(a.data(), b.data(), a.size()));
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.==", [](std::string_view a, std::string_view b) { return (a == b); }},
        {"sz_equal_serial", wrap_sz(sz_equal_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_equal_avx512", wrap_sz(sz_equal_avx512), true},
#endif
        {"memcmp",
         [](std::string_view a, std::string_view b) {
             return (a.size() == b.size() && memcmp(a.data(), b.data(), a.size()) == 0);
         }},
    };
    return result;
}

tracked_binary_functions_t ordering_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](std::string_view a, std::string_view b) {
            return function(a.data(), a.size(), b.data(), b.size());
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.compare",
         [](std::string_view a, std::string_view b) {
             auto order = a.compare(b);
             return (order == 0 ? sz_equal_k : (order < 0 ? sz_less_k : sz_greater_k));
         }},
        {"sz_order_serial", wrap_sz(sz_order_serial), true},
        {"memcmp",
         [](std::string_view a, std::string_view b) {
             auto order = memcmp(a.data(), b.data(), a.size() < b.size() ? a.size() : b.size());
             return order != 0 ? (a.size() == b.size() ? (order < 0 ? sz_less_k : sz_greater_k)
                                                       : (a.size() < b.size() ? sz_less_k : sz_greater_k))
                               : sz_equal_k;
         }},
    };
    return result;
}

template <typename string_type>
void bench_dereferencing(std::string name, std::vector<string_type> strings) {
    auto func = unary_function_t([](std::string_view s) { return s.size(); });
    tracked_unary_functions_t converts = {{name, func}};
    bench_unary_functions(strings, converts);
}

template <typename strings_type>
void bench(strings_type &&strings) {
    if (strings.size() == 0) return;

    // Benchmark the cost of converting `std::string` and `sz::string` to `std::string_view`.
    // ! The results on a mixture of short and long strings should be similar.
    // ! If the dataset is made of exclusively short or long strings, STL will look much better
    // ! in this microbenchmark, as the correct branch of the SSO will be predicted every time.
    bench_dereferencing<std::string>("std::string -> std::string_view", {strings.begin(), strings.end()});
    bench_dereferencing<sz::string>("sz::string -> std::string_view", {strings.begin(), strings.end()});

    // Benchmark generating strings of different length using those tokens as alphabets
    bench_unary_functions(strings, random_generation_functions(5));
    bench_unary_functions(strings, random_generation_functions(20));
    bench_unary_functions(strings, random_generation_functions(100));

    // Benchmark logical operations
    bench_unary_functions(strings, hashing_functions());
    bench_binary_functions(strings, equality_functions());
    bench_binary_functions(strings, ordering_functions());
}

void bench_on_input_data(int argc, char const **argv) {
    dataset_t dataset = make_dataset(argc, argv);

    // Baseline benchmarks for real words, coming in all lengths
    std::printf("Benchmarking on real words:\n");
    bench(dataset.tokens);

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on real words of length %zu:\n", token_length);
        bench(filter_by_length(dataset.tokens, token_length));
    }
}

void bench_on_synthetic_data() {
    // Generate some random words
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting token-level benchmarks.\n");

    if (argc < 2) { bench_on_synthetic_data(); }
    else { bench_on_input_data(argc, argv); }

    std::printf("All benchmarks passed.\n");
    return 0;
}