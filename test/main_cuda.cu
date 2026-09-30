/**
 *  @file test/main_cuda.cu
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief CUDA test: the capability report, the CUDA kernels of `cross_cuda.cu`, and the
 *      dispatching entry points.
 *
 *  `STRINGZILLA_FILTER=<regex>` keeps only the matching tests. Without a device the test exits
 *  zero, so a build host with no GPU still passes, having compiled every kernel.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp" // `run_test`, `log_cuda_device`

using namespace ashvardanian::stringzilla::test;

/** Every CUDA kernel this build compiled, from @c test/cross_cuda.cu. */
std::size_t test_cross_cuda(test_environment_t const &environment);

/** The dispatching entry points, from @c test/cross_simt.cuh. */
std::size_t test_cross_dispatch(test_environment_t const &environment);

/** The first device reports the CUDA baseline, and one past the last reports none. */
static void test_cuda_capabilities_unit() {
    sz_size_t devices = 0;
    verify(sz_cuda_count_devices(&devices) == sz_success_k && devices != 0);
    sz_capability_t reported = 0;
    verify(sz_cuda_capabilities_detected(0, &reported) == sz_success_k);
    verify(reported == sz_cap_cuda_k);
    verify(sz_cuda_capabilities_detected(devices, &reported) == sz_missing_gpu_k);
}

int main(int, char const **argv) {
    test_environment_t const environment = read_test_environment(argv[0]);
    install_test_signal_handlers(); // Backtrace on fatal signals + line-buffered stdout for crash localization.
    log_environment();
    print_test_environment(environment);
    if (!log_cuda_device()) return 0; // ? A build machine or CI runner need not have a device, so none is a skip

    std::size_t failures = 0;
    failures += run_test(environment, "test_cuda_capabilities_unit", test_cuda_capabilities_unit);
    failures += test_cross_cuda(environment);
    failures += test_cross_dispatch(environment);

    if (failures != 0) {
        fmt::println(stderr, "\n{} test(s) failed.", failures);
        return 1;
    }
    fmt::println("\nAll tests passed!");
    return 0;
}
