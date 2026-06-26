/**
 *  @file scripts/bench_utf8_norm.cpp
 *  @brief Benchmarks the @b `sz_utf8_norm_*` family — Unicode normalization and quick-check scanning.
 *         The program accepts a file path to a dataset and benchmarks the normalization operations,
 *         validating the SIMD-accelerated backends against the serial baselines.
 *
 *  Benchmarks include:
 *  - Unicode normalization for UTF-8 text - @b utf8_norm.
 *  - Normalization-form violation scanning (quick-check) - @b utf8_norm_violation.
 *
 *  Both sections normalize to @b NFC, the most common interchange form.
 *
 *  Its sibling @b `bench_utf8_uncased.cpp` covers the @b `sz_utf8_uncased_*` family (case folding and
 *  uncased substring search), and @b `bench_utf8_iterate.cpp` covers the @b `sz_utf8_*` iteration/segmentation
 *  family (codepoint counting, Nth-codepoint, newline/whitespace scanning, UAX-29 word/grapheme/sentence
 *  boundaries, UAX-14 line breaking, transcoding).
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=line` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams.
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
 *  cmake --build build_release --config Release --target stringzilla_bench_utf8_norm_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_UNIQUE=1 \
 *      build_release/stringzilla_bench_utf8_norm_cpp20
 *  @endcode
 *
 *  This file is the sibling of `bench_utf8_uncased.cpp`, `bench_utf8_iterate.cpp`, `bench_token.cpp`,
 *  `bench_find.cpp`, `bench_sequence.cpp`, and `bench_memory.cpp`.
 */
#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::scripts;

#pragma region Normalization Functions

/** @brief Wraps a hardware-specific UTF-8 normalization backend (transforms to NFC). */
template <sz_utf8_norm_t func_>
struct utf8_norm_from_sz {

    environment_t const &env;
    mutable std::vector<char> output_buffer; // Reusable buffer to avoid repeated allocation

    utf8_norm_from_sz(environment_t const &env_) : env(env_) {
        // Pre-allocate worst-case buffer: 18x input size for the worst single-codepoint
        // compatibility decomposition (see `sz_utf8_norm` buffer-sizing docs).
        std::size_t max_token_size = 0;
        for (auto const &token : env.tokens) max_token_size = std::max(max_token_size, token.size());
        output_buffer.resize(max_token_size * 18 + 64); // Extra padding for safety
    }

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        // Ensure buffer is large enough
        if (output_buffer.size() < buffer.size() * 18) output_buffer.resize(buffer.size() * 18 + 64);

        sz_size_t result_length = func_(buffer.data(), buffer.size(), sz_normal_form_nfc_k, output_buffer.data());
        do_not_optimize(output_buffer.data());
        do_not_optimize(result_length);

        // Use sz_bytesum for validation checksum
        check_value_t checksum = sz_bytesum(output_buffer.data(), result_length);
        return {buffer.size(), checksum};
    }
};

void bench_utf8_norm(environment_t const &env) {

    auto validator = utf8_norm_from_sz<sz_utf8_norm_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_norm_serial", validator).log();

#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_norm_icelake", validator, utf8_norm_from_sz<sz_utf8_norm_icelake> {env}).log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_utf8_norm_skylake", validator, utf8_norm_from_sz<sz_utf8_norm_skylake> {env}).log(base);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_norm_haswell", validator, utf8_norm_from_sz<sz_utf8_norm_haswell> {env}).log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_norm_neon", validator, utf8_norm_from_sz<sz_utf8_norm_neon> {env}).log(base);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_utf8_norm_sve2", validator, utf8_norm_from_sz<sz_utf8_norm_sve2> {env}).log(base);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_utf8_norm_sve", validator, utf8_norm_from_sz<sz_utf8_norm_sve> {env}).log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_norm_rvv", validator, utf8_norm_from_sz<sz_utf8_norm_rvv> {env}).log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_norm_lasx", validator, utf8_norm_from_sz<sz_utf8_norm_lasx> {env}).log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_norm_powervsx", validator, utf8_norm_from_sz<sz_utf8_norm_powervsx> {env}).log(base);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_utf8_norm_v128relaxed", validator, utf8_norm_from_sz<sz_utf8_norm_v128relaxed> {env})
        .log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_norm_v128", validator, utf8_norm_from_sz<sz_utf8_norm_v128> {env}).log(base);
#endif
}

#pragma endregion

#pragma region Violation (Quick-Check) Functions

/** @brief Wraps a hardware-specific UTF-8 normalization-violation backend (quick-check scan for NFC). */
template <sz_utf8_norm_violation_t func_>
struct utf8_norm_violation_from_sz {

    environment_t const &env;

    utf8_norm_violation_from_sz(environment_t const &env_) : env(env_) {}

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        sz_cptr_t violation = func_(buffer.data(), buffer.size(), sz_normal_form_nfc_k);
        do_not_optimize(violation);

        // Encode the violation offset (or "no violation") as a validation checksum.
        check_value_t offset = violation ? static_cast<check_value_t>(violation - buffer.data())
                                         : static_cast<check_value_t>(buffer.size());
        return {buffer.size(), offset};
    }
};

void bench_utf8_norm_violation(environment_t const &env) {

    auto validator = utf8_norm_violation_from_sz<sz_utf8_norm_violation_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_norm_violation_serial", validator).log();

#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_norm_violation_icelake", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_icelake> {env})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_utf8_norm_violation_skylake", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_skylake> {env})
        .log(base);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_norm_violation_haswell", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_haswell> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_norm_violation_neon", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_neon> {env})
        .log(base);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_utf8_norm_violation_sve2", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_sve2> {env})
        .log(base);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_utf8_norm_violation_sve", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_sve> {env})
        .log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_norm_violation_rvv", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_rvv> {env})
        .log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_norm_violation_lasx", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_lasx> {env})
        .log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_norm_violation_powervsx", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_powervsx> {env})
        .log(base);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_utf8_norm_violation_v128relaxed", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_v128relaxed> {env})
        .log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_norm_violation_v128", validator,
                utf8_norm_violation_from_sz<sz_utf8_norm_violation_v128> {env})
        .log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to StringZilla UTF-8 Normalization Benchmarks!\n");
    if (auto code = log_environment(); code != 0) return code;

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Default to xlsum for multilingual testing
        environment_t::tokenization_t::lines_k);

    std::printf("Starting Unicode benchmarks...\n");

    // Unicode operations
    bench_utf8_norm(env);
    bench_utf8_norm_violation(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
