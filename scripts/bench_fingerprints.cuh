/**
 *  @file   bench_fingerprints.cuh
 *  @brief  Shared code for CPU and GPU batched parallel exact substring search.
 */
#include <tuple> // `std::tuple`
#include <span>  // `std::span`

#define FU_ENABLE_NUMA 0
#include <fork_union.hpp> // Fork-join scoped thread pool

#include <stringzillas/fingerprints.hpp> // C++ templates for string processing

#if SZ_USE_CUDA
#include <stringzillas/fingerprints.cuh> // Parallel string processing in CUDA
#endif

#include "bench.hpp"

namespace ashvardanian {
namespace stringzillas {
namespace scripts {

using namespace ashvardanian::stringzilla::scripts;

static constexpr std::size_t default_embedding_dims_k = 64;
static constexpr std::size_t default_window_width_k = 7;

using fingerprint_min_hashes_t = std::array<u32_t, default_embedding_dims_k>;
using fingerprint_min_counts_t = std::array<u32_t, default_embedding_dims_k>;
using fingerprints_min_hashes_t = unified_vector<fingerprint_min_hashes_t>;
using fingerprints_min_counts_t = unified_vector<fingerprint_min_counts_t>;

#pragma region Multi-Pattern Search

/** @brief Wraps a hardware-specific fingerprinting backend into something @b `bench_nullary`-compatible. */
template <typename engine_type_, typename... extra_args_>
struct fingerprint_callable {
    using engine_t = engine_type_;

    arrow_strings_tape_t const &tape;
    fingerprints_min_hashes_t &fingerprints_hashes;
    fingerprints_min_counts_t &fingerprints_counts;
    engine_t &engine;
    std::tuple<extra_args_...> extra_args = {};

    fingerprint_callable(arrow_strings_tape_t const &tape, fingerprints_min_hashes_t &fingerprints_hashes,
                         fingerprints_min_counts_t &fingerprints_counts, engine_t &eng, extra_args_... args)
        : tape(tape), fingerprints_hashes(fingerprints_hashes), fingerprints_counts(fingerprints_counts), engine(eng),
          extra_args(args...) {}

    call_result_t operator()() noexcept(false) {

        // Unpack the extra arguments from `std::tuple` into the engine call using `std::apply`
        status_t result = status_t::success_k;
        std::apply(
            [&](auto &&...rest) mutable {
                result = engine(tape, fingerprints_hashes, fingerprints_counts, rest...);
                for (auto &scalar : fingerprints_hashes) do_not_optimize(scalar);
                for (auto &scalar : fingerprints_counts) do_not_optimize(scalar);
            },
            extra_args);
        if (static_cast<status_t>(result) != status_t::success_k) throw std::runtime_error("Failed fingerprinting.");

        std::size_t bytes_passed = 0;
        for (std::size_t i = 0; i < tape.size(); ++i) bytes_passed += tape[i].size();

        call_result_t call_result;
        call_result.bytes_passed = bytes_passed;
        call_result.operations = bytes_passed * default_embedding_dims_k;
        call_result.inputs_processed = tape.size();
        call_result.check_value = reinterpret_cast<check_value_t>(&fingerprints_hashes);
        return call_result;
    }
};

void bench_fingerprints(environment_t const &env) {

    namespace fu = fork_union;

#if SZ_USE_CUDA
    gpu_specs_t specs;
    if (gpu_specs_fetch(specs) != status_t::success_k) throw std::runtime_error("Failed to get GPU specs.");
#endif

    arrow_strings_tape_t tape;
    if (tape.try_assign(env.tokens.begin(), env.tokens.end()) != status_t::success_k)
        throw std::runtime_error("Failed to assign tokens to tape.");

    // Preallocate buffers for resulting fingerprints,
    // so that we can compare baseline and accelerated results for exact matches
    using fingerprints_equality_t = arrays_equality<fingerprint_min_hashes_t>;
    fingerprints_min_hashes_t min_hashes_baseline, min_hashes_accelerated;
    fingerprints_min_counts_t min_counts_baseline, min_counts_accelerated;
    min_hashes_baseline.resize(env.tokens.size()), min_hashes_accelerated.resize(env.tokens.size());
    min_counts_baseline.resize(env.tokens.size()), min_counts_accelerated.resize(env.tokens.size());

    // Let's reuse a thread-pool to amortize the cost of spawning threads.
    alignas(fu::default_alignment_k) fu::basic_pool_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) throw std::runtime_error("Failed to spawn thread pool.");

    auto scramble_accelerated_results = [&]() {
        std::shuffle(min_hashes_accelerated.begin(), min_hashes_accelerated.end(), global_random_generator());
    };

    // Allocate all hashers on heap
    using basic_rabin_u64_serial_t = basic_rolling_hashers<rabin_karp_rolling_hasher<u32_t, u64_t>>;
    auto basic_rabin_u64_serial = std::make_unique<basic_rabin_u64_serial_t>();
    if (basic_rabin_u64_serial->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Rabin Karp u64/u32 Hasher.");

    using basic_buz_u32_serial_t = basic_rolling_hashers<buz_rolling_hasher<u32_t>>;
    auto basic_buz_u32_serial = std::make_unique<basic_buz_u32_serial_t>();
    if (basic_buz_u32_serial->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Buz Hasher.");

    using basic_multiply_u32_serial_t = basic_rolling_hashers<multiplying_rolling_hasher<u32_t>>;
    auto basic_multiply_u32_serial = std::make_unique<basic_multiply_u32_serial_t>();
    if (basic_multiply_u32_serial->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Multiplying Hasher.");

    using basic_rolling_f64_serial_t = basic_rolling_hashers<floating_rolling_hasher<f64_t>, u32_t>;
    auto basic_rolling_f64_serial = std::make_unique<basic_rolling_f64_serial_t>();
    if (basic_rolling_f64_serial->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Floating f64 Rolling Hasher.");

    using basic_rolling_f32_serial_t = basic_rolling_hashers<floating_rolling_hasher<float>>;
    auto basic_rolling_f32_serial = std::make_unique<basic_rolling_f32_serial_t>();
    if (basic_rolling_f32_serial->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Floating f32 Rolling Hasher.");

    using floating_serial_t = floating_rolling_hashers<sz_cap_serial_k, default_embedding_dims_k>;
    auto floating_serial = std::make_unique<floating_serial_t>();
    if (floating_serial->try_seed(default_window_width_k) != status_t::success_k)
        throw std::runtime_error("Can't build Unrolled Floating Hasher.");

#if SZ_USE_CUDA
    using basic_rabin_u64_cuda_t =
        basic_rolling_hashers<rabin_karp_rolling_hasher<u32_t, u64_t>, u32_t, u32_t, unified_alloc_t, sz_cap_cuda_k>;
    auto basic_rabin_u64_cuda = std::make_unique<basic_rabin_u64_cuda_t>();
    if (basic_rabin_u64_cuda->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Rabin Karp u64/u32 CUDA Hasher.");

    using basic_rolling_f64_cuda_t =
        basic_rolling_hashers<floating_rolling_hasher<f64_t>, u32_t, u32_t, unified_alloc_t, sz_cap_cuda_k>;
    auto basic_rolling_f64_cuda = std::make_unique<basic_rolling_f64_cuda_t>();
    if (basic_rolling_f64_cuda->try_extend(default_window_width_k, default_embedding_dims_k) != status_t::success_k)
        throw std::runtime_error("Can't build Floating f64 Rolling CUDA Hasher.");
#endif // SZ_USE_CUDA

#if SZ_USE_HASWELL
    using floating_haswell_t = floating_rolling_hashers<sz_cap_haswell_k, default_embedding_dims_k>;
    auto floating_haswell = std::make_unique<floating_haswell_t>();
    if (floating_haswell->try_seed(default_window_width_k) != status_t::success_k)
        throw std::runtime_error("Can't build Haswell Floating Hasher.");
#endif // SZ_USE_HASWELL

#if SZ_USE_SKYLAKE
    using floating_skylake_t = floating_rolling_hashers<sz_cap_skylake_k, default_embedding_dims_k>;
    auto floating_skylake = std::make_unique<floating_skylake_t>();
    if (floating_skylake->try_seed(default_window_width_k) != status_t::success_k)
        throw std::runtime_error("Can't build Skylake Floating Hasher.");
#endif // SZ_USE_SKYLAKE

#if SZ_USE_CUDA
    using floating_cuda_t = floating_rolling_hashers<sz_cap_cuda_k, default_embedding_dims_k>;
    auto floating_cuda = std::make_unique<floating_cuda_t>();
    if (floating_cuda->try_seed(default_window_width_k) != status_t::success_k)
        throw std::runtime_error("Can't build CUDA Floating Hasher.");
#endif // SZ_USE_CUDA

    // Perform the benchmarks, passing the dictionary to the engines
    auto basic_rolling_f64_serial_call = fingerprint_callable<basic_rolling_f64_serial_t, fu::basic_pool_t &>(
        tape, min_hashes_baseline, min_counts_baseline, *basic_rolling_f64_serial, pool);
    bench_result_t basic_rolling_f64_serial_result =
        bench_nullary(env, "basic_rolling_f64_serial", basic_rolling_f64_serial_call).log();

    // Semi-serial variants
    bench_nullary(env, "basic_rolling_f32_serial",
                  fingerprint_callable<basic_rolling_f32_serial_t, fu::basic_pool_t &>(
                      tape, min_hashes_accelerated, min_counts_accelerated, *basic_rolling_f32_serial, pool))
        .log(basic_rolling_f64_serial_result);
    scramble_accelerated_results();

    bench_nullary(env, "basic_rabin_u64_serial",
                  fingerprint_callable<basic_rabin_u64_serial_t, fu::basic_pool_t &>(
                      tape, min_hashes_accelerated, min_counts_accelerated, *basic_rabin_u64_serial, pool))
        .log(basic_rolling_f64_serial_result);
    scramble_accelerated_results();

    bench_nullary(env, "basic_buz_u32_serial",
                  fingerprint_callable<basic_buz_u32_serial_t, fu::basic_pool_t &>(
                      tape, min_hashes_accelerated, min_counts_accelerated, *basic_buz_u32_serial, pool)) //
        .log(basic_rolling_f64_serial_result);
    scramble_accelerated_results();

    bench_nullary(env, "basic_multiply_u32_serial",
                  fingerprint_callable<basic_multiply_u32_serial_t, fu::basic_pool_t &>(
                      tape, min_hashes_accelerated, min_counts_accelerated, *basic_multiply_u32_serial, pool))
        .log(basic_rolling_f64_serial_result);
    scramble_accelerated_results();

#if SZ_USE_CUDA
    bench_nullary(                                                  //
        env, "basic_rabin_u64_cuda", basic_rolling_f64_serial_call, //
        fingerprint_callable<basic_rabin_u64_cuda_t, cuda_executor_t, gpu_specs_t>(
            tape, min_hashes_accelerated, min_counts_accelerated, *basic_rabin_u64_cuda, cuda_executor_t {}, specs), //
        callable_no_op_t {},        // preprocessing
        fingerprints_equality_t {}) // equality check
        .log(basic_rolling_f64_serial_result);
    scramble_accelerated_results();

    bench_nullary(                                                    //
        env, "basic_rolling_f64_cuda", basic_rolling_f64_serial_call, //
        fingerprint_callable<basic_rolling_f64_cuda_t, cuda_executor_t, gpu_specs_t>(
            tape, min_hashes_accelerated, min_counts_accelerated, *basic_rolling_f64_cuda, cuda_executor_t {},
            specs),                 //
        callable_no_op_t {},        // preprocessing
        fingerprints_equality_t {}) // equality check
        .log(basic_rolling_f64_serial_result);
    scramble_accelerated_results();
#endif // SZ_USE_CUDA

    // Actually unrolled hard-coded variants, including SIMD ports
    bench_result_t floating_serial_result =                        //
        bench_nullary(                                             //
            env, "floating_serial", basic_rolling_f64_serial_call, //
            fingerprint_callable<floating_serial_t, fu::basic_pool_t &>(
                tape, min_hashes_accelerated, min_counts_accelerated, *floating_serial, pool), //
            callable_no_op_t {},                                                               // preprocessing
            fingerprints_equality_t {})                                                        // equality check
            .log(basic_rolling_f64_serial_result);
    scramble_accelerated_results();

#if SZ_USE_HASWELL
    bench_nullary(                                              //
        env, "floating_haswell", basic_rolling_f64_serial_call, //
        fingerprint_callable<floating_haswell_t, fu::basic_pool_t &>(
            tape, min_hashes_accelerated, min_counts_accelerated, *floating_haswell, pool), //
        callable_no_op_t {},                                                                // preprocessing
        fingerprints_equality_t {})                                                         // equality check
        .log(basic_rolling_f64_serial_result, floating_serial_result);
    scramble_accelerated_results();
#endif // SZ_USE_HASWELL

#if SZ_USE_SKYLAKE
    bench_nullary(                                              //
        env, "floating_skylake", basic_rolling_f64_serial_call, //
        fingerprint_callable<floating_skylake_t, fu::basic_pool_t &>(
            tape, min_hashes_accelerated, min_counts_accelerated, *floating_skylake, pool), //
        callable_no_op_t {},                                                                // preprocessing
        fingerprints_equality_t {})                                                         // equality check
        .log(basic_rolling_f64_serial_result, floating_serial_result);
    scramble_accelerated_results();
#endif // SZ_USE_SKYLAKE

#if SZ_USE_CUDA
    bench_nullary(                                           //
        env, "floating_cuda", basic_rolling_f64_serial_call, //
        fingerprint_callable<floating_cuda_t, cuda_executor_t, gpu_specs_t>(
            tape, min_hashes_accelerated, min_counts_accelerated, *floating_cuda, cuda_executor_t {}, specs), //
        callable_no_op_t {},        // preprocessing
        fingerprints_equality_t {}) // equality check
        .log(basic_rolling_f64_serial_result, floating_serial_result);
    scramble_accelerated_results();
#endif // SZ_USE_CUDA
}

#pragma endregion

} // namespace scripts
} // namespace stringzillas
} // namespace ashvardanian