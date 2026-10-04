/**
 *  @file test/main_rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief ROCm test: the capability report, the ROCm kernels of `cross_rocm.hip`, and the
 *      dispatching entry points.
 *
 *  `STRINGZILLA_FILTER=<regex>` keeps only the matching tests. Without a device the test exits
 *  zero, so a build host with no GPU still passes, having compiled every kernel.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp" // `run_test`

using namespace ashvardanian::stringzilla::test;

namespace ashvardanian::stringzilla::test {

/** Every ROCm kernel this build compiled, from @c test/cross_rocm.hip. */
std::size_t test_cross_rocm(environment_t const &env);

/** The dispatching entry points, from @c test/cross_simt.cuh. */
std::size_t test_cross_dispatch(environment_t const &env);

/** The first device reports the ROCm baseline, and one past the last reports none. */
void test_rocm_capabilities_unit() {
    sz_size_t devices = 0;
    verify(sz_device_count_rocm(&devices) == sz_success_k && devices != 0);
    sz_capability_t reported = 0;
    verify(sz_capabilities_detected_rocm(0, &reported) == sz_success_k);
    verify(reported == sz_cap_rocm_k);
    verify(sz_capabilities_detected_rocm(devices, &reported) == sz_missing_gpu_k);
}

} // namespace ashvardanian::stringzilla::test

int main(int, char const **argv) {
    environment_t const env {read_settings(argv[0]), probe_machine()};
    install_test_signal_handlers(); // Backtrace on fatal signals + line-buffered stdout for crash localization.
    print(env.machine);
    print(env.settings);
    if (env.machine.device_name.empty()) return 0; // ? A build machine or CI runner need not have a device

    std::size_t failures = 0;
    failures += run_test(env.settings, "test_rocm_capabilities_unit", test_rocm_capabilities_unit);
    failures += test_cross_rocm(env);
    failures += test_cross_dispatch(env);

    if (failures != 0) {
        fmt::println(stderr, "\n{} test(s) failed.", failures);
        return 1;
    }
    fmt::println("\nAll tests passed!");
    return 0;
}
