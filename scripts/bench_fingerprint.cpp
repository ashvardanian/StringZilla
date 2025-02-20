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

tracked_unary_functions_t sliding_hashing_functions(std::size_t window_width, std::size_t step) {
#if _SZ_DEPRECATED_FINGERPRINTS
    auto wrap_sz = [=](auto function) -> unary_function_t {
        return unary_function_t([function, window_width, step](std::string_view s) {
            sz_size_t mixed_hash = 0;
            function(s.data(), s.size(), window_width, step, _sz_hashes_fingerprint_scalar_callback, &mixed_hash);
            return mixed_hash;
        });
    };
#endif
    std::string suffix = std::to_string(window_width) + ":step" + std::to_string(step);
    tracked_unary_functions_t result = {
#if _SZ_DEPRECATED_FINGERPRINTS
#if SZ_USE_ICE
        {"sz_hashes_ice:" + suffix, wrap_sz(sz_hashes_ice)},
#endif
#if SZ_USE_HASWELL
        {"sz_hashes_haswell:" + suffix, wrap_sz(sz_hashes_haswell)},
#endif
        {"sz_hashes_serial:" + suffix, wrap_sz(sz_hashes_serial)},
#endif
    };
    return result;
}

tracked_unary_functions_t fingerprinting_functions(std::size_t window_width = 8, std::size_t fingerprint_bytes = 4096) {
    using fingerprint_slot_t = std::uint8_t;
    static std::vector<fingerprint_slot_t> fingerprint;
    fingerprint.resize(fingerprint_bytes / sizeof(fingerprint_slot_t));
    auto wrap_sz = [](auto function) -> unary_function_t {
        return unary_function_t([function](std::string_view s) {
            sz_size_t mixed_hash = 0;
            sz_unused(s);
            return mixed_hash;
        });
    };
    tracked_unary_functions_t result = {};
    sz_unused(window_width && fingerprint_bytes);
    sz_unused(wrap_sz);
    return result;
}

tracked_unary_functions_t random_generation_functions(std::size_t token_length) {
    static std::vector<char> buffer;
    if (buffer.size() < token_length) buffer.resize(token_length);

    auto suffix = ", " + std::to_string(token_length) + " chars";
    tracked_unary_functions_t result = {
        {"std::rand % uint8" + suffix, unary_function_t([token_length](std::string_view alphabet) -> std::size_t {
             using max_alphabet_size_t = std::uint8_t;
             auto max_alphabet_size = static_cast<max_alphabet_size_t>(alphabet.size());
             for (std::size_t i = 0; i < token_length; ++i) { buffer[i] = alphabet[std::rand() % max_alphabet_size]; }
             return token_length;
         })},
        {"std::uniform_int<uint8>" + suffix, unary_function_t([token_length](std::string_view alphabet) -> std::size_t {
             randomize_string(buffer.data(), token_length, alphabet.data(), alphabet.size());
             return token_length;
         })},
        {"sz::randomize" + suffix, unary_function_t([token_length](std::string_view alphabet) -> std::size_t {
             sz::string_span span(buffer.data(), token_length);
             sz::randomize(span, global_random_generator(), alphabet);
             return token_length;
         })},
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
    bench_unary_functions(strings, hashing_functions());
    bench_binary_functions(strings, equality_functions());
    bench_binary_functions(strings, ordering_functions());

    // Benchmark the cost of converting `std::string` and `sz::string` to `std::string_view`.
    // ! The results on a mixture of short and long strings should be similar.
    // ! If the dataset is made of exclusively short or long strings, STL will look much better
    // ! in this micro-benchmark, as the correct branch of the SSO will be predicted every time.
    bench_dereferencing<std::string>("std::string -> std::string_view", {strings.begin(), strings.end()});
    bench_dereferencing<sz::string>("sz::string -> std::string_view", {strings.begin(), strings.end()});
}

void bench_on_input_data(int argc, char const **argv) {
    dataset_t dataset = prepare_benchmark_environment(argc, argv);
    std::printf("Benchmarking on the entire dataset:\n");

    // When performing fingerprinting, it's extremely important to:
    //      1. Have small output fingerprints that fit the cache.
    //      2. Have that memory in close affinity to the core, ideally on stack, to avoid cache coherency problems.
    // This introduces an additional challenge for efficient fingerprinting, as the CPU caches vary a lot.
    // On the Intel Sapphire Rapids 6455B Gold CPU they are 96 KiB x2 for L1d, 4 MiB x2 for L2.
    // Spilling into the L3 is a bad idea.
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, sliding_hashing_functions(7, 1));
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, sliding_hashing_functions(17, 4));
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, sliding_hashing_functions(33, 8));
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, sliding_hashing_functions(127, 16));

    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, fingerprinting_functions(128, 4 * 1024));
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, fingerprinting_functions(128, 64 * 1024));
    bench_unary_functions<std::vector<std::string_view>>({dataset.text}, fingerprinting_functions(128, 1024 * 1024));
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