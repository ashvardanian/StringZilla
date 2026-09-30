/**
 *  @file bench/main.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Runs every family over its dispatch points, then every capability's kernels by name.
 *
 *  The header-only build has no dispatch points to time, so it runs the kernels alone, including
 *  the private helpers only the tier headers define, like the SIMD sorts of pgrams.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_FILTER=find build_release/stringzilla_cpu_bench
 *  @endcode
 */
#include <exception> // `std::exception`

#include <fmt/format.h>

#include "harness.hpp"

using namespace ashvardanian::stringzilla::bench;

int main(int argc, char const **argv) {
    install_bench_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    log_environment();
    print_bench_environment();

    // The arms throw on a failed status, so a bad call ends the run with its message, not a crash.
    try {
        corpora_t corpora(argc, argv);
#if !STRINGZILLA_HEADER_ONLY
        bench_find(corpora);
        bench_token(corpora);
        bench_sequence(corpora);
        bench_memory(corpora);
        bench_cipher(corpora);
        bench_container(corpora);
        bench_levenshtein(corpora);
        bench_overlap(corpora);
        bench_substrings(corpora);
        bench_utf8_traverse(corpora);
        bench_utf8_scan(corpora);
        bench_utf8_segment(corpora);
        bench_utf8_norm(corpora);
        bench_utf8_uncased(corpora);
#endif
        bench_cross_serial(corpora);
        bench_cross_x8664(corpora);
        bench_cross_arm64(corpora);
        bench_cross_riscv64(corpora);
        bench_cross_loongarch64(corpora);
        bench_cross_ppc64(corpora);
        bench_cross_wasm(corpora);
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    fmt::println("All benchmarks passed.");
    return 0;
}
