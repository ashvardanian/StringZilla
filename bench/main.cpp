/**
 *  @file bench/main.cpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Runs every family over its dispatch points, then every capability's kernels by name.
 *
 *  The header-only build runs public capability kernels inline, without dispatch points.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_FILTER=find build_release/stringzilla_bench
 *  @endcode
 */
#include <exception> // `std::exception`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "harness.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace ashvardanian::stringzilla::bench {

std::vector<device_selection_t> select_devices(std::optional<std::vector<device_selection_t>> const &requested) {
    std::vector<device_selection_t> devices;
    if constexpr (STRINGZILLA_HEADER_ONLY) {
        if (requested) {
            fmt::println(stderr, "This header-only executable accepts only CPU workloads");
            std::exit(1);
        }
        return devices;
    }
    if (requested) {
        for (device_selection_t const &selection : *requested) {
            sz_status_t const status = check_device(selection);
            if (status != sz_success_k) {
                fmt::println(stderr, "Device {}:{} is unavailable (status {})", device_name(selection.backend),
                             selection.ordinal, static_cast<int>(status));
                std::exit(1);
            }
            devices.push_back(selection);
        }
    }
    else {
        for (sz::device_kind_t kind :
             {sz::device_kind_t::cuda_k, sz::device_kind_t::rocm_k, sz::device_kind_t::metal_k})
            if (check_device({kind, 0}) == sz_success_k) devices.push_back({kind, 0});
    }
    return devices;
}

} // namespace ashvardanian::stringzilla::bench

int main(int argc, char **argv) {
    install_bench_signal_handlers();
    try {
        environment_t env {read_bench_settings(argc, argv), probe_machine()};
        auto const devices = select_devices(env.settings.devices);
        print(env.machine);
        print(env.settings);
        if constexpr (!STRINGZILLA_HEADER_ONLY) {
            bench_find(env);
            bench_token(env);
            bench_sort(env);
            bench_strings(env);
            bench_cipher(env);
            bench_container(env);
            bench_levenshtein(env);
            bench_ngrams(env);
            bench_substrings(env);
            bench_regex(env);
            bench_subdocuments(env);
            bench_utf8_traverse(env);
            bench_utf8_scan(env);
            bench_utf8_segment(env);
            bench_utf8_norm(env);
            bench_utf8_uncased(env);
        }
        bench_cross_serial(env);
        bench_cross_x8664(env);
        bench_cross_arm64(env);
        bench_cross_riscv64(env);
        bench_cross_loongarch64(env);
        bench_cross_ppc64(env);
        bench_cross_wasm(env);
        if constexpr (!STRINGZILLA_HEADER_ONLY) {
            for (device_selection_t device : devices) {
                switch (device.backend) {
                case sz::device_kind_t::cuda_k: bench_cross_cuda(env, device.ordinal); break;
                case sz::device_kind_t::rocm_k: bench_cross_rocm(env, device.ordinal); break;
                case sz::device_kind_t::metal_k: bench_cross_metal(env, device.ordinal); break;
                case sz::device_kind_t::cpu_k: break;
                }
            }
        }
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    fmt::println("All benchmarks passed.");
    return 0;
}
