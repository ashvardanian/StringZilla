/**
 *  @file   bench_similarities.cpp
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
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=words` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
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
 *  cmake --build build_release --config Release --target stringzillas_bench_similarities_cu20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=words build_release/stringzillas_bench_similarities_cu20
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=proteins.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzillas_bench_similarities_cu20
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is a sibling of `bench_fingerprints.cpp`.
 */
#include "bench_similarities.cuh"

namespace szs = ashvardanian::stringzillas;
using namespace szs::scripts;

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZillas on GPU!\n");

    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment( //
            argc, argv,                        //
            "xlsum.csv",                       // Preferred for UTF-8 content
            environment_t::tokenization_t::lines_k);

        std::printf("Starting string similarity benchmarks...\n");
        bench_levenshtein(env);
        bench_needleman_wunsch_smith_waterman(env);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks finished.\n");
    return 0;
}