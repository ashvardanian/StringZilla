/**
 *  @file   bench_unicode.cpp
 *  @brief  Benchmarks Unicode text processing operations like case folding.
 *          The program accepts a file path to a dataset and benchmarks the case folding operations,
 *          validating the SIMD-accelerated backends against the serial baselines.
 *
 *  Benchmarks include:
 *  - Case folding for Unicode text - @b utf8_case_fold.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=file` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams.
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_unicode_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=file build_release/stringzilla_bench_unicode_cpp20
 *  @endcode
 *
 *  This file is the sibling of `bench_token.cpp`, `bench_find.cpp`, `bench_sequence.cpp`, and `bench_memory.cpp`.
 */
#include "bench.hpp"

using namespace ashvardanian::stringzilla::scripts;

#pragma region Case Folding Functions

/** @brief Wraps a hardware-specific UTF-8 case folding backend. */
template <sz_utf8_case_fold_t func_>
struct utf8_case_fold_from_sz {

    environment_t const &env;
    mutable std::vector<char> output_buffer; // Reusable buffer to avoid repeated allocation

    utf8_case_fold_from_sz(environment_t const &env_) : env(env_) {
        // Pre-allocate worst-case buffer: 3x input size for worst-case expansion
        std::size_t max_token_size = 0;
        for (auto const &token : env.tokens) max_token_size = std::max(max_token_size, token.size());
        output_buffer.resize(max_token_size * 3 + 64); // Extra padding for safety
    }

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        // Ensure buffer is large enough
        if (output_buffer.size() < buffer.size() * 3) output_buffer.resize(buffer.size() * 3 + 64);

        sz_size_t result_length = func_(buffer.data(), buffer.size(), output_buffer.data());
        do_not_optimize(output_buffer.data());
        do_not_optimize(result_length);

        // Use sz_hash for validation checksum
        check_value_t checksum = sz_bytesum(output_buffer.data(), result_length);
        return {buffer.size(), checksum};
    }
};

void bench_utf8_case_fold(environment_t const &env) {

    auto validator = utf8_case_fold_from_sz<sz_utf8_case_fold_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_case_fold_serial", validator).log();

#if SZ_USE_ICE
    bench_unary(env, "sz_utf8_case_fold_ice", validator, utf8_case_fold_from_sz<sz_utf8_case_fold_ice> {env}).log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZilla Unicode Benchmarks!\n");

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Default to xlsum for multilingual testing
        environment_t::tokenization_t::file_k);

    std::printf("Starting Unicode benchmarks...\n");

    // Unicode operations
    bench_utf8_case_fold(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
