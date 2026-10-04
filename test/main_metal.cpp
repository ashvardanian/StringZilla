/**
 *  @file test/main_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal test: the capability report, the Metal kernels of `cross_metal.cpp`, and the
 *      dispatching entry points.
 *
 *  Every check runs on the queue @c main opens on the first device. `STRINGZILLA_FILTER=<regex>`
 *  keeps only the matching tests. Without a device the test exits zero.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp" // `run_test`

using namespace ashvardanian::stringzilla::test;

namespace ashvardanian::stringzilla::test {

/** Every Metal kernel this build compiled, from @c test/cross_metal.cpp. */
std::size_t test_cross_metal(environment_t const &env, sz_stream_t queue);

/** The dispatching entry points, from @c test/cross_metal.cpp. */
std::size_t test_cross_dispatch(environment_t const &env, sz_stream_t queue);

/** The first device reports the Metal baseline, and one past the last reports none. */
void test_metal_capabilities_unit() {
    sz_size_t devices = 0;
    verify(sz_device_count_metal(&devices) == sz_success_k && devices != 0);
    sz_capability_t reported = 0;
    verify(sz_capabilities_detected_metal(0, &reported) == sz_success_k);
    verify(reported == sz_cap_metal_k);
    verify(sz_capabilities_detected_metal(devices, &reported) == sz_missing_gpu_k);
}

} // namespace ashvardanian::stringzilla::test

int main(int, char const **argv) {
    environment_t const env {read_settings(argv[0]), probe_machine()};
    install_test_signal_handlers(); // Backtrace on fatal signals + line-buffered stdout for crash localization.
    print(env.machine);
    print(env.settings);
    // A queue of its own rather than the default, so every check passes a stream as a caller would.
    sz_stream_t queue = nullptr;
    if (sz_stream_init_metal(0, &queue) != sz_success_k) return 0; // ? A build machine need not have a device

    std::size_t failures = 0;
    failures += run_test(env.settings, "test_metal_capabilities_unit", test_metal_capabilities_unit);
    failures += test_cross_metal(env, queue);
    failures += test_cross_dispatch(env, queue);
    sz_stream_free_metal(queue);

    if (failures != 0) {
        fmt::println(stderr, "\n{} test(s) failed.", failures);
        return 1;
    }
    fmt::println("\nAll tests passed!");
    return 0;
}
