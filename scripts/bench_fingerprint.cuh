/**
 *  @file   bench_fingerprint.cuh
 *  @brief  Shared code for CPU and GPU batched parallel exact substring search.
 */
#include <tuple> // `std::tuple`
#include <span>  // `std::span`

#define FU_ENABLE_NUMA 0
#include <fork_union.hpp> // Fork-join scoped thread pool

#include <stringzillas/fingerprint.hpp> // C++ templates for string processing

#if SZ_USE_CUDA
#include <stringzillas/fingerprint.cuh> // Parallel string processing in CUDA
#endif

#include "bench.hpp"

namespace ashvardanian {
namespace stringzillas {
namespace scripts {

using namespace ashvardanian::stringzilla::scripts;

static constexpr std::size_t default_embedding_dims_k = 128;
static constexpr std::size_t default_window_width_k = 7;

using fingerprint_min_hashes_t = std::array<std::uint32_t, default_embedding_dims_k>;
using fingerprint_min_counts_t = std::array<std::uint32_t, default_embedding_dims_k>;
using fingerprints_min_hashes_t = unified_vector<fingerprint_min_hashes_t>;
using fingerprints_min_counts_t = unified_vector<fingerprint_min_counts_t>;

#pragma region Multi-Pattern Search

/** @brief Wraps a hardware-specific fingerprinting backend into something @b `bench_nullary`-compatible. */
template <typename engine_type_, typename... extra_args_>
struct fingerprint_callable {
    using engine_t = engine_type_;

    environment_t const &env;
    fingerprints_min_hashes_t &fingerprints_hashes;
    fingerprints_min_counts_t &fingerprints_counts;
    engine_t &engine;
    std::tuple<extra_args_...> extra_args = {};

    fingerprint_callable(environment_t const &env, fingerprints_min_hashes_t &fingerprints_hashes,
                         fingerprints_min_counts_t &fingerprints_counts, engine_t &eng, extra_args_... args)
        : env(env), fingerprints_hashes(fingerprints_hashes), fingerprints_counts(fingerprints_counts), engine(eng),
          extra_args(args...) {}

    call_result_t operator()() noexcept(false) {

        // Unpack the extra arguments from `std::tuple` into the engine call using `std::apply`
        status_t status = std::apply(
            [&](auto &&...rest) mutable {
                auto result = engine(env.tokens, fingerprints_hashes, fingerprints_counts, rest...);
                do_not_optimize(result);
                for (auto &scalar : fingerprints_hashes) do_not_optimize(scalar);
                for (auto &scalar : fingerprints_counts) do_not_optimize(scalar);
                return result;
            },
            extra_args);

        do_not_optimize(status);
        if (status != status_t::success_k) throw std::runtime_error("Failed multi-pattern search.");

        std::size_t bytes_passed = 0;
        for (std::size_t i = 0; i < env.tokens.size(); ++i) bytes_passed += env.tokens[i].size();

        call_result_t call_result;
        call_result.bytes_passed = bytes_passed;
        call_result.operations = bytes_passed * default_embedding_dims_k;
        call_result.inputs_processed = env.tokens.size();
        call_result.check_value = reinterpret_cast<check_value_t>(&fingerprints_hashes);
        return call_result;
    }
};

void bench_fingerprint(environment_t const &env) {

    // Preallocate buffers for resulting fingerprints,
    // so that we can compare baseline and accelerated results for exact matches
    using fingerprints_equality_t = arrays_equality<fingerprints_min_hashes_t>;
    fingerprints_min_hashes_t min_hashes_baseline, min_hashes_accelerated;
    fingerprints_min_counts_t min_counts_baseline, min_counts_accelerated;
    min_hashes_baseline.resize(env.tokens.size()), min_hashes_accelerated.resize(env.tokens.size());
    min_counts_baseline.resize(env.tokens.size()), min_counts_accelerated.resize(env.tokens.size());
    auto scramble_accelerated_results = [&]() {
        std::shuffle(min_hashes_accelerated.begin(), min_hashes_accelerated.end(), global_random_generator());
    };

    // Allocate all hashers on heap
    using rabin_u64_t = basic_rolling_hashers<rabin_karp_rolling_hasher<std::uint32_t, std::uint64_t>>;
    auto rabin_u64 = std::make_unique<rabin_u64_t>();
    if (rabin_u64->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Rabin Karp u64/u32 Hasher.");

    using buz_u32_t = basic_rolling_hashers<buz_rolling_hasher<std::uint32_t>>;
    auto buz_u32 = std::make_unique<buz_u32_t>();
    if (buz_u32->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Buz Hasher.");

    using multiply_u32_t = basic_rolling_hashers<multiplying_rolling_hasher<std::uint32_t>>;
    auto multiply_u32 = std::make_unique<multiply_u32_t>();
    if (multiply_u32->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Multiplying Hasher.");

    using rolling_f64_t = basic_rolling_hashers<floating_rolling_hasher<double>, std::uint32_t>;
    auto rolling_f64 = std::make_unique<rolling_f64_t>();
    if (rolling_f64->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Floating f64 Rolling Hasher.");

    using rolling_f32_t = basic_rolling_hashers<floating_rolling_hasher<float>>;
    auto rolling_f32 = std::make_unique<rolling_f32_t>();
    if (rolling_f32->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Floating f32 Rolling Hasher.");

    using rolling_serial_t =
        floating_rolling_hashers<sz_cap_serial_k, default_window_width_k, default_embedding_dims_k>;
    auto rolling_serial = std::make_unique<rolling_serial_t>();
    if (rolling_serial->try_seed() != status_t::success_k)
        throw std::runtime_error("Can't build Unrolled Floating Hasher.");

    using rolling_skylake_t =
        floating_rolling_hashers<sz_cap_skylake_k, default_window_width_k, default_embedding_dims_k>;
    auto rolling_skylake = std::make_unique<rolling_skylake_t>();
    if (rolling_skylake->try_seed() != status_t::success_k)
        throw std::runtime_error("Can't build Skylake Floating Hasher.");

    // Perform the benchmarks, passing the dictionary to the engines
    auto call_baseline =
        fingerprint_callable<rolling_f64_t>(env, min_hashes_baseline, min_counts_baseline, *rolling_f64);
    bench_result_t baseline = bench_nullary(env, "rolling_f64", call_baseline);

    // Semi-serial variants
    bench_nullary(
        env, "rolling_f32",
        fingerprint_callable<rolling_f32_t>(env, min_hashes_accelerated, min_counts_accelerated, *rolling_f32))
        .log(baseline);
    bench_nullary(env, "rabin_u64",
                  fingerprint_callable<rabin_u64_t>(env, min_hashes_accelerated, min_counts_accelerated, *rabin_u64))
        .log(baseline);
    bench_nullary(env, "buz_u32",
                  fingerprint_callable<buz_u32_t>(env, min_hashes_accelerated, min_counts_accelerated, *buz_u32)) //
        .log(baseline);
    bench_nullary(
        env, "multiply_u32",
        fingerprint_callable<multiply_u32_t>(env, min_hashes_accelerated, min_counts_accelerated, *multiply_u32))
        .log(baseline);

    // Actually unrolled hard-coded variants, including SIMD ports
    bench_result_t unrolled = bench_nullary(                            //
                                  env, "rolling_serial", call_baseline, //
                                  fingerprint_callable<rolling_serial_t>(env, min_hashes_accelerated,
                                                                         min_counts_accelerated, *rolling_serial), //
                                  callable_no_op_t {},        // preprocessing
                                  fingerprints_equality_t {}) // equality check
                                  .log(baseline);
    scramble_accelerated_results();

    bench_nullary(                             //
        env, "rolling_skylake", call_baseline, //
        fingerprint_callable<rolling_skylake_t>(env, min_hashes_accelerated, min_counts_accelerated,
                                                *rolling_skylake), //
        callable_no_op_t {},                                       // preprocessing
        fingerprints_equality_t {})                                // equality check
        .log(baseline, unrolled);
    scramble_accelerated_results();
}

#pragma endregion

} // namespace scripts
} // namespace stringzillas
} // namespace ashvardanian