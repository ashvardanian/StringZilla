/**
 *  @file bench/substrings.cpp
 *  @brief Benchmarks the multi-pattern search engine (Aho-Corasick) on CPU: serial and fork-union-parallel.
 *         Sweeps dictionary sizes and reports the dictionary properties - state count, hot/cold tier split,
 *         and the fraction of byte steps that stay on the hot path - alongside the throughput they produce,
 *         since those properties dominate the number far more than the backend choice.
 *
 *  Compute-bound: the automaton is rebuilt per sweep cell, so a 64 MiB slice exercises every hot- and cold-tier path, while a larger corpus only rebuilds the same states.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the haystack dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for
 *    N-grams) that turns the dataset into the haystacks searched. `file` reproduces a single-haystack scan,
 *    the shape the reference baselines below were measured with.
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Cross-checks the parallel backend's occurrence counts against the serial ones.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter benchmark names.
 *
 *  Needles come from the corpus itself, deterministically: whitespace-cut words regardless of how
 *  `STRINGWARS_TOKENS` shapes the haystacks, with the most frequent one percent and every term occurring
 *  once dropped. Each sweep cell draws one slice - most or least frequent, one or ten percent, or all of
 *  it - from the frequency-ordered remainder.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  g++ -std=c++17 -O3 -march=native -I include -I forkunion/include bench/substrings.cpp -o substrings_cpp \
 *      -lpthread
 *  STRINGWARS_DATASET=haystack_64mib.txt STRINGWARS_TOKENS=file ./substrings_cpp
 *  @endcode
 *
 *  The CMake target for this file forces `-O2` after `-O3` for benchmark builds, which understates every
 *  kernel here; the command above compiles directly with `-O3` for numbers worth trusting.
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is a sibling of `similarities.cpp`; its GPU counterpart is `substrings.cu`.
 */
#include "substrings.cuh"
#include "stringzilla.hpp" // `log_environment`

namespace szs = ashvardanian::stringzillas;
using namespace sz::scripts;

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to the StringZillas substrings benchmark on CPU!\n");
    if (auto code = log_environment(); code != 0) return code;

    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment( //
            argc, argv,                        //
            "xlsum.csv",                       //
            environment_t::tokenization_t::lines_k, compute_bound_slice_bytes_k);

        bench_substrings(env);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks finished.\n");
    return 0;
}
