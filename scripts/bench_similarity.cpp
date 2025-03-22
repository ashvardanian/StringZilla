/**
 *  @file   bench_similarity.cpp
 *  @brief  Benchmarks string similarity computations.
 *          It accepts a file with a list of words, and benchmarks the levenshtein edit-distance computations,
 *          alignment scores, and fingerprinting techniques combined with the Hamming distance.
 *
 *  Benchmarks include:
 *  - Linear-complexity basic & bounded Hamming distance computations.
 *  - Quadratic-complexity basic & bounded Levenshtein edit-distance computations.
 *  - Quadratic-complexity Needleman-Wunsch alignment scores for bioinformatics.
 *
 *  For Dynamic Programming algorithms, the number of operations per second are reported as the worst-case time
 *  complexity of the Cells Updates Per Second @b (CUPS) metric, meaning O(N*M) for a pair of strings with N and M
 *  characters, respectively.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWa.rs, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=words` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWa.rs, the following additional environment variables are supported:
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
 *  cmake --build build_release --config Release --target stringzilla_bench_similarity
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=words build_release/stringzilla_bench_similarity
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=proteins.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzilla_bench_similarity
 *  @endcode
 *
 *  Unlike the full-blown StringWa.rs, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `bench_search.cpp`, `bench_token.cpp`, `bench_sequence.cpp`, and `bench_memory.cpp`.
 */

#include "bench.hpp"
#include "test.hpp" // `levenshtein_baseline`, `unary_substitution_costs`

#include "stringzilla/similarity.hpp"

using namespace ashvardanian::stringzilla::scripts;

#pragma region Hamming Distance

/** @brief Wraps a hardware-specific Hamming-distance backend into something @b `bench_unary`-compatible . */
template <sz_hamming_distance_t hamming_distance_>
struct hamming_from_sz {

    environment_t const &env;
    sz_size_t bound = SZ_SIZE_MAX;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index], env.tokens[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_size_t result_distance;
        sz_status_t status = hamming_distance_( //
            a.data(), a.size(),                 //
            b.data(), b.size(),                 //
            bound, &result_distance);
        do_not_optimize(status);
        std::size_t bytes_passed = std::min(a.size(), b.size());
        return {bytes_passed, static_cast<check_value_t>(result_distance)};
    }
};

void bench_hamming(environment_t const &env) {
    auto base_call = hamming_from_sz<sz_hamming_distance_serial>(env);
    bench_result_t base = bench_unary(env, "sz_hamming_distance_serial", base_call).log();
    auto base_utf8_call = hamming_from_sz<sz_hamming_distance_utf8_serial>(env);
    bench_result_t base_utf8 = bench_unary(env, "sz_hamming_distance_utf8_serial", base_utf8_call).log(base);
    sz_unused(base_utf8);
}

#pragma endregion

#pragma region Levenshtein Distance and Alignment Scores

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
template <sz_levenshtein_distance_t levenshtein_distance_>
struct levenshtein_from_sz {

    environment_t const &env;
    sz_size_t bound = SZ_SIZE_MAX;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index], env.tokens[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_size_t result_distance;
        sz_status_t status = levenshtein_distance_( //
            a.data(), a.size(),                     //
            b.data(), b.size(),                     //
            bound, NULL, &result_distance);
        do_not_optimize(status);
        std::size_t bytes_passed = std::min(a.size(), b.size());
        std::size_t cells_passed = a.size() * b.size();
        return {bytes_passed, static_cast<check_value_t>(result_distance), cells_passed};
    }
};

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
template <sz_needleman_wunsch_score_t needleman_wunsch_>
struct alignment_score_from_sz {

    environment_t const &env;
    sz_size_t bound = SZ_SIZE_MAX;
    error_costs_256x256_t costs = unary_substitution_costs();

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index], env.tokens[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_ssize_t result_score;
        sz_status_t status = needleman_wunsch_( //
            a.data(), a.size(),                 //
            b.data(), b.size(),                 //
            costs.data(), (sz_error_cost_t)-1,  //
            NULL, &result_score);
        do_not_optimize(status);
        sz_size_t result_distance = (sz_size_t)(-result_score);
        std::size_t bytes_passed = std::min(a.size(), b.size());
        std::size_t cells_passed = a.size() * b.size();
        return {bytes_passed, static_cast<check_value_t>(result_distance), cells_passed};
    }
};

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
template <sz_levenshtein_distance_t levenshtein_distance_>
struct score_from_sz_cpp {

    environment_t const &env;
    sz_size_t bound = SZ_SIZE_MAX;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index], env.tokens[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept(false) {
        sz_size_t result_distance = sz::openmp::levenshtein_distance(a, b);
        do_not_optimize(result_distance);
        std::size_t bytes_passed = std::min(a.size(), b.size());
        std::size_t cells_passed = a.size() * b.size();
        return {bytes_passed, static_cast<check_value_t>(result_distance), cells_passed};
    }
};

void bench_edits(environment_t const &env) {
    auto base_call = levenshtein_from_sz<sz_levenshtein_distance_serial>(env);
    bench_result_t base = bench_unary(env, "sz_levenshtein_distance_serial", base_call).log();
    auto base_utf8_call = levenshtein_from_sz<sz_levenshtein_distance_utf8_serial>(env);
    bench_result_t base_utf8 = bench_unary(env, "sz_levenshtein_distance_utf8_serial", base_utf8_call).log(base);
    sz_unused(base_utf8);

#if SZ_USE_ICE
    auto ice_call = levenshtein_from_sz<sz_levenshtein_distance_ice>(env);
    bench_unary(env, "sz_levenshtein_distance_ice", ice_call).log(base);
#endif

    auto needleman_wunsch_call = alignment_score_from_sz<sz_needleman_wunsch_score_serial>(env);
    bench_unary(env, "sz_needleman_wunsch_score_serial", needleman_wunsch_call).log(base);
}

#pragma endregion

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZilla!\n");

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Preferred for UTF-8 content
        environment_t::tokenization_t::words_k);

    std::printf("Starting string similarity benchmarks...\n");
    bench_hamming(env);
    bench_edits(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}