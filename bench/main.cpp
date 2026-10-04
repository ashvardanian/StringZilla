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

std::vector<sz::device_t> select_devices(std::optional<std::vector<device_selection_t>> const &requested) {
    std::vector<sz::device_t> devices;
    if constexpr (STRINGZILLA_HEADER_ONLY) {
        if (requested) {
            fmt::println(stderr, "This header-only executable accepts only CPU workloads");
            std::exit(1);
        }
        return devices;
    }
    if (requested) {
        for (device_selection_t const &selection : *requested) {
            auto const device = sz::device_t::make(selection.backend, selection.ordinal);
            if (!device) {
                fmt::println(stderr, "Device {}:{} is unavailable (status {})", device_name(selection.backend),
                             selection.ordinal, static_cast<int>(device.status));
                std::exit(1);
            }
            devices.push_back(device.value);
        }
    }
    else {
        for (sz::device_kind_t kind :
             {sz::device_kind_t::cuda_k, sz::device_kind_t::rocm_k, sz::device_kind_t::metal_k})
            if (auto device = sz::device_t::make(kind, 0)) devices.push_back(device.value);
    }
    return devices;
}

} // namespace ashvardanian::stringzilla::bench

int main() {
    install_bench_signal_handlers();
    try {
        environment_t env {read_settings(), probe_machine()};
        auto const devices = select_devices(env.settings.devices);
        print(env.machine);
        print(env.settings);
        if constexpr (!STRINGZILLA_HEADER_ONLY) {
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
        }
        bench_cross_serial(env);
        bench_cross_x8664(env);
        bench_cross_arm64(env);
        bench_cross_riscv64(env);
        bench_cross_loongarch64(env);
        bench_cross_ppc64(env);
        bench_cross_wasm(env);
        if constexpr (!STRINGZILLA_HEADER_ONLY) {
            int failures = 0;
            for (sz::device_t device : devices) {
                switch (device.kind()) {
                case sz::device_kind_t::cuda_k: failures += bench_cross_cuda(env, device.ordinal()); break;
                case sz::device_kind_t::rocm_k: failures += bench_cross_rocm(env, device.ordinal()); break;
                case sz::device_kind_t::metal_k: failures += bench_cross_metal(env, device.ordinal()); break;
                case sz::device_kind_t::cpu_k: break;
                }
            }
            if (failures != 0) return 1;
        }
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    fmt::println("All benchmarks passed.");
    return 0;
}
