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

int main() {
    install_bench_signal_handlers();
    environment_t env {read_settings(), probe_machine()};
    print(env.machine);
    print(env.settings);
    try {
#if !STRINGZILLA_HEADER_ONLY
        bench_find(env);
        bench_token(env);
        bench_sequence(env);
        bench_memory(env);
        bench_cipher(env);
        bench_container(env);
        bench_levenshtein(env);
        bench_overlap(env);
        bench_substrings(env);
        bench_utf8_traverse(env);
        bench_utf8_scan(env);
        bench_utf8_segment(env);
        bench_utf8_norm(env);
        bench_utf8_uncased(env);
#endif
        bench_cross_serial(env);
        bench_cross_x8664(env);
        bench_cross_arm64(env);
        bench_cross_riscv64(env);
        bench_cross_loongarch64(env);
        bench_cross_ppc64(env);
        bench_cross_wasm(env);
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    fmt::println("All benchmarks passed.");
    return 0;
}
