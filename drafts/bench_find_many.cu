/**
 *  @file   bench_find_many.cu
 *  @brief  Benchmarks for exact multi-pattern substring search algorithms on the GPU.
 *          The program accepts a file path to a dataset, tokenizes it, and benchmarks the search operations,
 *          validating the SIMD-accelerated backends against the serial baselines.
 *
 *  Benchmarks include:
 *  - Multi-pattern substring match counting.
 *  - Multi-pattern substring search.
 *  - Multi-pattern matcher construction time.
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
 *  cmake --build build_release --config Release --target stringzillas_bench_find_many_cu20
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=words build_release/stringzillas_bench_find_many_cu20
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzillas_bench_find_many_cu20
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `bench_sequence.cpp`, `bench_token.cpp`, `bench_similarity.cpp`, and `bench_memory.cpp`.
 */
#include "bench_find_many.cuh"

namespace szs = ashvardanian::stringzillas;
using namespace szs::scripts;

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZillas on GPU!\n");

    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment( //
            argc, argv,                        //
            "xlsum.csv",                       // Preferred for UTF-8 content
            environment_t::tokenization_t::words_k);

        std::printf("Starting string multi-pattern search benchmarks...\n");
        bench_find_many(env);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks finished.\n");
    return 0;
}