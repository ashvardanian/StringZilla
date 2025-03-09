/**
 *  @file   bench_token.cpp
 *  @brief  Benchmarks token-level operations like hashing, equality, ordering, and copies.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_search.cpp` and `bench_similarity.cpp`.
 */
#include <numeric> // `std::accumulate`

#include <bench.hpp>
#include <test.hpp> // `random_string`

using namespace ashvardanian::stringzilla::scripts;

tracked_unary_functions_t bytesum_functions() {
    auto wrap_sz = [](auto function) -> unary_function_t {
        return unary_function_t([function](std::string_view s) { return function(s.data(), s.size()); });
    };
    tracked_unary_functions_t result = {
        {"std::accumulate",
         [](std::string_view s) {
             return std::accumulate(s.begin(), s.end(), (std::size_t)0,
                                    [](std::size_t sum, char c) { return sum + static_cast<unsigned char>(c); });
         }},
        {"sz_bytesum_serial", wrap_sz(sz_bytesum_serial), true},
#if SZ_USE_HASWELL
        {"sz_bytesum_haswell", wrap_sz(sz_bytesum_haswell), true},
#endif
#if SZ_USE_SKYLAKE
        {"sz_bytesum_skylake", wrap_sz(sz_bytesum_skylake), true},
#endif
#if SZ_USE_ICE
        {"sz_bytesum_ice", wrap_sz(sz_bytesum_ice), true},
#endif
#if SZ_USE_NEON
        {"sz_bytesum_neon", wrap_sz(sz_bytesum_neon), true},
#endif
    };
    return result;
}

tracked_unary_functions_t hash_functions() {
    auto wrap_sz = [](auto function) -> unary_function_t {
        return unary_function_t([function](std::string_view s) { return function(s.data(), s.size(), 42); });
    };
    tracked_unary_functions_t result = {
        {"sz_hash_serial", wrap_sz(sz_hash_serial)},
#if SZ_USE_HASWELL
        {"sz_hash_haswell", wrap_sz(sz_hash_haswell), true},
#endif
#if SZ_USE_SKYLAKE
        {"sz_hash_skylake", wrap_sz(sz_hash_skylake), true},
#endif
#if SZ_USE_ICE
        {"sz_hash_ice", wrap_sz(sz_hash_ice), true},
#endif
#if SZ_USE_NEON
        {"sz_hash_neon", wrap_sz(sz_hash_neon), true},
#endif
        {"std::hash", [](std::string_view s) { return std::hash<std::string_view> {}(s); }},
    };
    return result;
}

struct wrap_hash_stream {
    sz_hash_state_t state;
    sz_hash_state_stream_t stream;
    sz_hash_state_fold_t fold;

    wrap_hash_stream(sz_hash_state_stream_t s, sz_hash_state_fold_t f) : stream(s), fold(f) {
        sz_hash_state_init(&state, 42);
    }

    std::size_t operator()(std::string_view s) noexcept {
        stream(&state, s.data(), s.size());
        return fold(&state);
    }
};

tracked_unary_functions_t hash_stream_functions() {
    tracked_unary_functions_t result = {
        {"sz_hash_stream_serial", wrap_hash_stream(sz_hash_state_stream_serial, sz_hash_state_fold_serial)},
#if SZ_USE_HASWELL
        {"sz_hash_stream_haswell", wrap_hash_stream(sz_hash_state_stream_haswell, sz_hash_state_fold_haswell), true},
#endif
#if SZ_USE_SKYLAKE
        {"sz_hash_stream_skylake", wrap_hash_stream(sz_hash_state_stream_skylake, sz_hash_state_fold_skylake), true},
#endif
#if SZ_USE_ICE
        {"sz_hash_stream_ice", wrap_hash_stream(sz_hash_state_stream_ice, sz_hash_state_fold_ice), true},
#endif
#if SZ_USE_NEON
        {"sz_hash_stream_neon", wrap_hash_stream(sz_hash_state_stream_neon, sz_hash_state_fold_neon), true},
#endif
    };
    return result;
}

tracked_unary_functions_t random_generation_functions() {
    static std::vector<char> buffer;
    tracked_unary_functions_t result = {
        {"std::rand() & 0xFF", unary_function_t([](std::string_view token) -> std::size_t {
             if (buffer.size() < token.size()) buffer.resize(token.size());
             for (std::size_t i = 0; i < token.size(); ++i) buffer[i] = static_cast<char>(std::rand() & 0xFF);
             return token.size();
         })},
        {"std::uniform_int<uint8>", unary_function_t([](std::string_view token) -> std::size_t {
             if (buffer.size() < token.size()) buffer.resize(token.size());
             randomize_string(buffer.data(), token.size());
             return token.size();
         })},
        {"sz::randomize", unary_function_t([](std::string_view token) -> std::size_t {
             if (buffer.size() < token.size()) buffer.resize(token.size());
             sz::string_span span(buffer.data(), token.size());
             sz::fill_random(span);
             return token.size();
         })},
    };
    return result;
}

tracked_binary_functions_t equality_functions() {
    auto wrap_sz = [](auto function) -> binary_function_t {
        return binary_function_t([function](std::string_view a, std::string_view b) {
            return a.size() == b.size() && function(a.data(), b.data(), a.size());
        });
    };
    tracked_binary_functions_t result = {
        {"std::string_view.==", [](std::string_view a, std::string_view b) { return a == b; }},
        {"sz_equal_serial", wrap_sz(sz_equal_serial), true},
#if SZ_USE_HASWELL
        {"sz_equal_haswell", wrap_sz(sz_equal_haswell), true},
#endif
#if SZ_USE_SKYLAKE
        {"sz_equal_skylake", wrap_sz(sz_equal_skylake), true},
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
#if SZ_USE_HASWELL
        {"sz_order_haswell", wrap_sz(sz_order_haswell), true},
#endif
#if SZ_USE_SKYLAKE
        {"sz_order_skylake", wrap_sz(sz_order_skylake), true},
#endif
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

    // Benchmark logical operations
    bench_unary_functions(strings, bytesum_functions());
    bench_unary_functions(strings, hash_functions());
    bench_unary_functions(strings, hash_stream_functions());
    bench_binary_functions(strings, equality_functions());
    bench_binary_functions(strings, ordering_functions());
    bench_unary_functions(strings, random_generation_functions());

    // Benchmark the cost of converting `std::string` and `sz::string` to `std::string_view`.
    // ! The results on a mixture of short and long strings should be similar.
    // ! If the dataset is made of exclusively short or long strings, STL will look much better
    // ! in this micro-benchmark, as the correct branch of the SSO will be predicted every time.
    bench_dereferencing<std::string>("std::string -> std::string_view", {strings.begin(), strings.end()});
    bench_dereferencing<sz::string>("sz::string -> std::string_view", {strings.begin(), strings.end()});
}

void bench_on_input_data(int argc, char const **argv) {
    dataset_t dataset = prepare_benchmark_environment(argc, argv);

    // Baseline benchmarks for real words, coming in all lengths
    std::printf("Benchmarking on real words:\n");
    bench(dataset.tokens);
    std::printf("Benchmarking on real lines:\n");
    bench(dataset.lines);
    std::printf("Benchmarking on entire dataset:\n");
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, bytesum_functions());
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, hash_functions());
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, hash_stream_functions());

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
    std::printf("- Seconds per benchmark: %zu\n", seconds_per_benchmark);

    if (argc < 2) { bench_on_synthetic_data(); }
    else { bench_on_input_data(argc, argv); }

    std::printf("All benchmarks passed.\n");
    return 0;
}