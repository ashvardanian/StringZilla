/**
 *  @file test/main_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal test: the capability report, the Metal kernels of `cross_metal.cpp`, and the
 *      dispatching entry points.
 *
 *  Every check runs on the device @c main opens. `STRINGZILLA_FILTER=<regex>` keeps only the
 *  matching tests. Without a device the test exits zero.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp" // `log_metal_device`, `run_test`

using namespace ashvardanian::stringzilla::test;

/** Every Metal kernel this build compiled, from @c test/cross_metal.cpp. */
std::size_t test_cross_metal(test_environment_t const &environment, sz_metal_device_t &device);

/** The dispatching entry points, from @c test/cross_metal.cpp. */
std::size_t test_cross_dispatch(test_environment_t const &environment, sz_metal_device_t &device);

/** The first device reports the Metal baseline, and one past the last reports none. */
static void test_metal_capabilities_unit() {
    sz_size_t devices = 0;
    verify(sz_metal_count_devices(&devices) == sz_success_k && devices != 0);
    sz_capability_t reported = 0;
    verify(sz_metal_capabilities_detected(0, &reported) == sz_success_k);
    verify(reported == sz_cap_metal_k);
    verify(sz_metal_capabilities_detected(devices, &reported) == sz_missing_gpu_k);
}

int main(int, char const **argv) {
    test_environment_t const environment = read_test_environment(argv[0]);
    install_test_signal_handlers(); // Backtrace on fatal signals + line-buffered stdout for crash localization.
    log_environment();
    print_test_environment(environment);
    sz_metal_device_t device;
    sz_metal_device_init(0, 256u << 20, &device); // ? Left zeroed, so logged as none, without a GPU
    if (!log_metal_device(device)) return 0;

    std::size_t failures = 0;
    failures += run_test(environment, "test_metal_capabilities_unit", test_metal_capabilities_unit);
    failures += test_cross_metal(environment, device);
    failures += test_cross_dispatch(environment, device);
    sz_metal_device_free(&device);

    if (failures != 0) {
        fmt::println(stderr, "\n{} test(s) failed.", failures);
        return 1;
    }
    fmt::println("\nAll tests passed!");
    return 0;
}
