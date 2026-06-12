/**
 *  @file scripts/bench_similarities.cuh
 *  @brief Shared code for CPU and GPU batched string similarity kernels.
 */
#include <tuple> // `std::tuple`

#define FU_ENABLE_NUMA 0
#include <fork_union.hpp> // Fork-join scoped thread pool

#include <stringzillas/similarities.hpp> // C++ templates for string similarity measures

#if SZ_USE_CUDA
#include <stringzillas/similarities.cuh> // Parallel string processing in CUDA
#endif

#include "bench.hpp"

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

// StringZillas library symbols available on every backend:
using ashvardanian::stringzillas::affine_gap_costs_t;
using ashvardanian::stringzillas::affine_levenshtein_icelake_t;
using ashvardanian::stringzillas::affine_levenshtein_serial_t;
using ashvardanian::stringzillas::affine_needleman_wunsch_serial_t;
using ashvardanian::stringzillas::affine_smith_waterman_serial_t;
using ashvardanian::stringzillas::error_costs_32x32_t;
using ashvardanian::stringzillas::levenshtein_icelake_t;
using ashvardanian::stringzillas::levenshtein_serial_t;
using ashvardanian::stringzillas::levenshtein_utf8_icelake_t;
using ashvardanian::stringzillas::levenshtein_utf8_serial_t;
using ashvardanian::stringzillas::linear_gap_costs_t;
using ashvardanian::stringzillas::affine_needleman_wunsch_icelake_t;
using ashvardanian::stringzillas::affine_smith_waterman_icelake_t;
using ashvardanian::stringzillas::needleman_wunsch_icelake_t;
using ashvardanian::stringzillas::needleman_wunsch_serial_t;
using ashvardanian::stringzillas::smith_waterman_icelake_t;
using ashvardanian::stringzillas::smith_waterman_serial_t;
using ashvardanian::stringzillas::uniform_substitution_costs_t;
#if SZ_USE_NEON
using ashvardanian::stringzillas::affine_levenshtein_neon_t;
using ashvardanian::stringzillas::levenshtein_neon_t;
using ashvardanian::stringzillas::levenshtein_utf8_neon_t;
#endif

// StringZillas library symbols provided only by the CUDA backend:
#if SZ_USE_CUDA
using ashvardanian::stringzillas::affine_levenshtein_cuda_t;
using ashvardanian::stringzillas::affine_levenshtein_hopper_t;
using ashvardanian::stringzillas::affine_levenshtein_kepler_t;
using ashvardanian::stringzillas::affine_needleman_wunsch_cuda_t;
using ashvardanian::stringzillas::affine_needleman_wunsch_hopper_t;
using ashvardanian::stringzillas::affine_smith_waterman_cuda_t;
using ashvardanian::stringzillas::affine_smith_waterman_hopper_t;
using ashvardanian::stringzillas::cuda_executor_t;
using ashvardanian::stringzillas::gpu_specs_fetch;
using ashvardanian::stringzillas::levenshtein_cuda_t;
using ashvardanian::stringzillas::levenshtein_hopper_t;
using ashvardanian::stringzillas::levenshtein_kepler_t;
using ashvardanian::stringzillas::needleman_wunsch_cuda_t;
using ashvardanian::stringzillas::needleman_wunsch_hopper_t;
using ashvardanian::stringzillas::smith_waterman_cuda_t;
using ashvardanian::stringzillas::smith_waterman_hopper_t;
#endif

using namespace ashvardanian::stringzilla::scripts;

using similarities_t = unified_vector<sz_ssize_t>;

/**
 *  @brief Reads the `status_t` out of an engine's return value through opaque memory, defeating NVCC's
 *         host-codegen folding of the read.
 *
 *  @note NVCC 12.x miscompiles the 12-byte `cuda_status_t` return-by-value of the @b affine alignment
 *        engine instantiations inside this large translation unit at @b -O2 : the `float elapsed_milliseconds`
 *        field survives, but the two leading enum fields (`status`, `cuda_error`) come back as garbage, so a
 *        direct `static_cast<status_t>(result)` reports a bogus `unrecognized` status. The StringZillas
 *        library itself is correct - a direct call to the same engine returns `success_k` - this only bites
 *        the affine instantiations folded into this benchmark. Bouncing the struct through `std::memcpy` in a
 *        `[[gnu::noinline]]` helper forces the compiler to materialize the full object before reading it.
 */
template <typename status_type_>
[[gnu::noinline]] status_t read_engine_status_(status_type_ const &engine_result) noexcept {
    status_type_ materialized;
    std::memcpy((void *)&materialized, (void const *)&engine_result, sizeof(status_type_));
    return static_cast<status_t>(materialized);
}

#pragma region Levenshtein Distance and Alignment Scores

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
template <typename engine_type_, typename... extra_args_>
struct similarities_callable {
    using engine_t = engine_type_;

    environment_t const &env;
    similarities_t &results;
    engine_t engine = {};
    std::tuple<extra_args_...> extra_args = {};

    similarities_callable(environment_t const &env, similarities_t &res, engine_t eng = {}, extra_args_... args)
        : env(env), results(res), engine(std::move(eng)), extra_args(std::forward<extra_args_>(args)...) {
        if (env.tokens.size() <= results.size()) throw std::runtime_error("Batch size is too large.");
    }

    call_result_t operator()(std::size_t batch_index) noexcept(false) {
        std::size_t const batch_size = results.size();
        std::size_t const forward_token_index = (batch_index * batch_size) % (env.tokens.size() - batch_size);
        std::size_t const backward_token_index = env.tokens.size() - forward_token_index - batch_size;

        return operator()({env.tokens.data() + forward_token_index, batch_size},
                          {env.tokens.data() + backward_token_index, batch_size});
    }

    call_result_t operator()(std::span<token_view_t const> a, std::span<token_view_t const> b) noexcept(false) {
        // Unpack the extra arguments from `std::tuple` into the engine call using `std::apply`
        auto status = std::apply([&](auto &&...rest) { return engine(a, b, results, rest...); }, extra_args);
        do_not_optimize(status);

        // ! Read the status through `read_engine_status_` to work around an NVCC -O2 host-codegen artifact
        // ! that corrupts the integer fields of the affine engines' returned `cuda_status_t` in this TU.
        status_t const status_code = read_engine_status_(status);
        if (status_code != status_t::success_k)
            throw std::runtime_error(std::string("Failed to compute Levenshtein distance: ") +
                                     status_name(status_code));
        do_not_optimize(results);
        std::size_t bytes_passed = 0, cells_passed = 0;
        for (std::size_t i = 0; i < results.size(); ++i) {
            bytes_passed += a[i].size() + b[i].size();
            cells_passed += a[i].size() * b[i].size();
        }
        call_result_t call_result;
        call_result.bytes_passed = bytes_passed;
        call_result.operations = cells_passed;
        call_result.inputs_processed = results.size();
        call_result.check_value = reinterpret_cast<check_value_t>(&results);
        return call_result;
    }
};

struct similarities_equality_t {
    bool operator()(check_value_t const &a, check_value_t const &b) const noexcept {
        similarities_t const &a_ = *reinterpret_cast<similarities_t const *>(a);
        similarities_t const &b_ = *reinterpret_cast<similarities_t const *>(b);
        if (a_.size() != b_.size()) return false;
        for (std::size_t i = 0; i < a_.size(); ++i)
            if (a_[i] != b_[i]) {
                std::printf("Mismatch at index %zu: %zd != %zd\n", i, a_[i], b_[i]);
                return false;
            }
        return true;
    }
};

void bench_levenshtein(environment_t const &env) {

    using namespace std::string_literals; // for "s" suffix
    namespace fu = fork_union;

#if SZ_USE_CUDA
    gpu_specs_t specs;
    if (gpu_specs_fetch(specs) != status_t::success_k) throw std::runtime_error("Failed to fetch GPU specs.");
#endif
    std::vector<std::size_t> batch_sizes = {1, 64, 1024, 32 * 1024};
#if SZ_DEBUG
    batch_sizes = {1, 2, 64};
#endif
    if (!env.batch_sizes_override.empty()) batch_sizes = env.batch_sizes_override;
    similarities_t results_linear_baseline, results_linear_accelerated;
    similarities_t results_affine_baseline, results_affine_accelerated;
    similarities_t results_utf8_baseline, results_utf8_accelerated;

    // Let's reuse a thread-pool to amortize the cost of spawning threads.
    alignas(fu::default_alignment_k) fu::basic_pool_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) throw std::runtime_error("Failed to spawn thread pool.");

    auto scramble_accelerated_results = [&](similarities_t &results_accelerated) {
        std::shuffle(results_accelerated.begin(), results_accelerated.end(), global_random_generator());
    };
    sz_unused_(scramble_accelerated_results);

    // Two cost paths: "unit" costs (match 0, mismatch 1, gap 1) take the bit-parallel Myers fast path on
    // linear byte inputs, while the "weird" non-unit costs force the general anti-diagonal scorers. Both
    // are benchmarked, with the scheme tag spliced into each benchmark name (e.g. `levenshtein_neon_unit`).
    struct cost_scheme_t {
        uniform_substitution_costs_t uniform;
        linear_gap_costs_t linear;
        affine_gap_costs_t affine;
        char const *tag;
    };
    cost_scheme_t const cost_schemes[] = {
        {{0, 1}, {1}, {1, 1}, "unit"},
        {{1, 3}, {3}, {4, 2}, "weird"},
    };

    for (auto const &scheme : cost_schemes)
        for (std::size_t batch_size : batch_sizes) {
        results_linear_baseline.resize(batch_size), results_linear_accelerated.resize(batch_size);
        results_affine_baseline.resize(batch_size), results_affine_accelerated.resize(batch_size);
        results_utf8_baseline.resize(batch_size), results_utf8_accelerated.resize(batch_size);

        auto call_linear_baseline = similarities_callable<levenshtein_serial_t, fu::basic_pool_t &>(
            env, results_linear_baseline, levenshtein_serial_t {scheme.uniform, scheme.linear}, pool);
        auto name_linear_baseline = "levenshtein_serial_"s + scheme.tag + ":batch" + std::to_string(batch_size);
        bench_result_t linear_baseline = bench_unary(env, name_linear_baseline, call_linear_baseline).log();

        auto call_utf8_baseline = similarities_callable<levenshtein_utf8_serial_t>(
            env, results_utf8_baseline, levenshtein_utf8_serial_t {scheme.uniform, scheme.linear});
        auto name_utf8_baseline = "levenshtein_utf8_serial_"s + scheme.tag + ":batch" + std::to_string(batch_size);
        bench_result_t utf8_baseline = bench_unary(env, name_utf8_baseline, call_utf8_baseline).log();

        auto call_affine_baseline = similarities_callable<affine_levenshtein_serial_t, fu::basic_pool_t &>(
            env, results_affine_baseline, affine_levenshtein_serial_t {scheme.uniform, scheme.affine}, pool);
        auto name_affine_baseline = "affine_levenshtein_serial_"s + scheme.tag + ":batch" + std::to_string(batch_size);
        bench_result_t affine_baseline =
            bench_unary(env, name_affine_baseline, call_affine_baseline).log(linear_baseline);
        sz_unused_(affine_baseline);

#if SZ_USE_ICELAKE
        bench_unary(env, "levenshtein_icelake_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_linear_baseline,
                    similarities_callable<levenshtein_icelake_t, fu::basic_pool_t &>(
                        env, results_linear_accelerated, levenshtein_icelake_t {scheme.uniform, scheme.linear}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_baseline);
        scramble_accelerated_results(results_linear_accelerated);

        bench_unary(
            env, "affine_levenshtein_icelake_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_affine_baseline,
            similarities_callable<affine_levenshtein_icelake_t, fu::basic_pool_t &>(
                env, results_affine_accelerated, affine_levenshtein_icelake_t {scheme.uniform, scheme.affine}, pool),
            callable_no_op_t {},        // preprocessing
            similarities_equality_t {}) // equality check
            .log(linear_baseline, affine_baseline);
        scramble_accelerated_results(results_affine_accelerated);

        bench_unary(env, "levenshtein_utf8_icelake_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_utf8_baseline,
                    similarities_callable<levenshtein_utf8_icelake_t>(
                        env, results_utf8_accelerated, levenshtein_utf8_icelake_t {scheme.uniform, scheme.linear}),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(utf8_baseline);
        scramble_accelerated_results(results_utf8_accelerated);
#endif

#if SZ_USE_NEON
        bench_unary(env, "levenshtein_neon_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_linear_baseline,
                    similarities_callable<levenshtein_neon_t, fu::basic_pool_t &>(
                        env, results_linear_accelerated, levenshtein_neon_t {scheme.uniform, scheme.linear}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_baseline);
        scramble_accelerated_results(results_linear_accelerated);

        bench_unary(
            env, "affine_levenshtein_neon_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_affine_baseline,
            similarities_callable<affine_levenshtein_neon_t, fu::basic_pool_t &>(
                env, results_affine_accelerated, affine_levenshtein_neon_t {scheme.uniform, scheme.affine}, pool),
            callable_no_op_t {},        // preprocessing
            similarities_equality_t {}) // equality check
            .log(linear_baseline, affine_baseline);
        scramble_accelerated_results(results_affine_accelerated);

        bench_unary(env, "levenshtein_utf8_neon_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_utf8_baseline,
                    similarities_callable<levenshtein_utf8_neon_t>(
                        env, results_utf8_accelerated, levenshtein_utf8_neon_t {scheme.uniform, scheme.linear}),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(utf8_baseline);
        scramble_accelerated_results(results_utf8_accelerated);
#endif

#if SZ_USE_CUDA
        bench_unary(env, "levenshtein_cuda_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_linear_baseline,
                    similarities_callable<levenshtein_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_accelerated, levenshtein_cuda_t {scheme.uniform, scheme.linear},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_baseline);
        scramble_accelerated_results(results_linear_accelerated);

        bench_unary(env, "affine_levenshtein_cuda_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_affine_baseline,
                    similarities_callable<affine_levenshtein_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_accelerated, affine_levenshtein_cuda_t {scheme.uniform, scheme.affine},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_baseline, affine_baseline);
        scramble_accelerated_results(results_affine_accelerated);
#endif

#if SZ_USE_KEPLER
        bench_unary(env, "levenshtein_kepler_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_linear_baseline,
                    similarities_callable<levenshtein_kepler_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_accelerated, levenshtein_kepler_t {scheme.uniform, scheme.linear},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_baseline);
        scramble_accelerated_results(results_linear_accelerated);

        bench_unary(env, "affine_levenshtein_kepler_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_affine_baseline,
                    similarities_callable<affine_levenshtein_kepler_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_accelerated, affine_levenshtein_kepler_t {scheme.uniform, scheme.affine},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_baseline, affine_baseline);
        scramble_accelerated_results(results_affine_accelerated);
#endif

#if SZ_USE_HOPPER
        bench_unary(env, "levenshtein_hopper_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_linear_baseline,
                    similarities_callable<levenshtein_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_accelerated, levenshtein_hopper_t {scheme.uniform, scheme.linear},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_baseline);
        scramble_accelerated_results(results_linear_accelerated);

        bench_unary(env, "affine_levenshtein_hopper_"s + scheme.tag + ":batch" + std::to_string(batch_size), call_affine_baseline,
                    similarities_callable<affine_levenshtein_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_accelerated, affine_levenshtein_hopper_t {scheme.uniform, scheme.affine},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_baseline, affine_baseline);
        scramble_accelerated_results(results_affine_accelerated);
#endif
    }
}

void bench_needleman_wunsch_smith_waterman(environment_t const &env) {

    using namespace std::string_literals; // for "s" suffix
    namespace fu = fork_union;

    constexpr linear_gap_costs_t blosum62_linear_cost {-4};
    constexpr affine_gap_costs_t blosum62_affine_cost {-4, -1};
    auto blosum62_matrix32 = error_costs_32x32_t::blosum62();

#if SZ_USE_CUDA
    gpu_specs_t specs;
    if (gpu_specs_fetch(specs) != status_t::success_k) throw std::runtime_error("Failed to fetch GPU specs.");
#endif
    std::vector<std::size_t> batch_sizes = {1, 64, 1024, 32 * 1024};
#if SZ_DEBUG
    batch_sizes = {1, 2, 64};
#endif
    if (!env.batch_sizes_override.empty()) batch_sizes = env.batch_sizes_override;
    similarities_t results_linear_global_baseline, results_linear_global_accelerated;
    similarities_t results_affine_global_baseline, results_affine_global_accelerated;
    similarities_t results_linear_local_baseline, results_linear_local_accelerated;
    similarities_t results_affine_local_baseline, results_affine_local_accelerated;

    // Let's reuse a thread-pool to amortize the cost of spawning threads.
    alignas(fu::default_alignment_k) fu::basic_pool_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) throw std::runtime_error("Failed to spawn thread pool.");

    auto scramble_accelerated_results = [&](similarities_t &results_accelerated) {
        std::shuffle(results_accelerated.begin(), results_accelerated.end(), global_random_generator());
    };
    sz_unused_(scramble_accelerated_results);

    for (std::size_t batch_size : batch_sizes) {
        results_linear_global_baseline.resize(batch_size), results_linear_global_accelerated.resize(batch_size);
        results_affine_global_baseline.resize(batch_size), results_affine_global_accelerated.resize(batch_size);
        results_linear_local_baseline.resize(batch_size), results_linear_local_accelerated.resize(batch_size);
        results_affine_local_baseline.resize(batch_size), results_affine_local_accelerated.resize(batch_size);

        auto call_linear_global_baseline = similarities_callable<needleman_wunsch_serial_t, fu::basic_pool_t &>(
            env, results_linear_global_baseline, {blosum62_matrix32, blosum62_linear_cost}, pool);
        auto name_linear_global_baseline = "needleman_wunsch_serial:batch"s + std::to_string(batch_size);
        bench_result_t linear_global_baseline =
            bench_unary(env, name_linear_global_baseline, call_linear_global_baseline).log();

        auto call_linear_local_baseline = similarities_callable<smith_waterman_serial_t, fu::basic_pool_t &>(
            env, results_linear_local_baseline, {blosum62_matrix32, blosum62_linear_cost}, pool);
        auto name_linear_local_baseline = "smith_waterman_serial:batch"s + std::to_string(batch_size);
        bench_result_t linear_local_baseline =
            bench_unary(env, name_linear_local_baseline, call_linear_local_baseline).log();

        auto call_affine_global_baseline = similarities_callable<affine_needleman_wunsch_serial_t, fu::basic_pool_t &>(
            env, results_affine_global_baseline, {blosum62_matrix32, blosum62_affine_cost}, pool);
        auto name_affine_global_baseline = "affine_needleman_wunsch_serial:batch"s + std::to_string(batch_size);
        bench_result_t affine_global_baseline =
            bench_unary(env, name_affine_global_baseline, call_affine_global_baseline).log();

        auto call_affine_local_baseline = similarities_callable<affine_smith_waterman_serial_t, fu::basic_pool_t &>(
            env, results_affine_local_baseline, {blosum62_matrix32, blosum62_affine_cost}, pool);
        auto name_affine_local_baseline = "affine_smith_waterman_serial:batch"s + std::to_string(batch_size);
        bench_result_t affine_local_baseline =
            bench_unary(env, name_affine_local_baseline, call_affine_local_baseline).log();

#if SZ_USE_ICELAKE
        bench_unary(env, "needleman_wunsch_icelake:batch"s + std::to_string(batch_size), call_linear_global_baseline,
                    similarities_callable<needleman_wunsch_icelake_t, fu::basic_pool_t &>(
                        env, results_linear_global_accelerated, {blosum62_matrix32, blosum62_linear_cost}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_global_baseline);
        scramble_accelerated_results(results_linear_global_accelerated);

        bench_unary(env, "smith_waterman_icelake:batch"s + std::to_string(batch_size), call_linear_local_baseline,
                    similarities_callable<smith_waterman_icelake_t, fu::basic_pool_t &>(
                        env, results_linear_local_accelerated, {blosum62_matrix32, blosum62_linear_cost}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_local_baseline);
        scramble_accelerated_results(results_linear_local_accelerated);

        bench_unary(env, "affine_needleman_wunsch_icelake:batch"s + std::to_string(batch_size),
                    call_affine_global_baseline,
                    similarities_callable<affine_needleman_wunsch_icelake_t, fu::basic_pool_t &>(
                        env, results_affine_global_accelerated, {blosum62_matrix32, blosum62_affine_cost}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_global_baseline);
        scramble_accelerated_results(results_affine_global_accelerated);

        bench_unary(env, "affine_smith_waterman_icelake:batch"s + std::to_string(batch_size),
                    call_affine_local_baseline,
                    similarities_callable<affine_smith_waterman_icelake_t, fu::basic_pool_t &>(
                        env, results_affine_local_accelerated, {blosum62_matrix32, blosum62_affine_cost}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_local_baseline);
        scramble_accelerated_results(results_affine_local_accelerated);
#endif

#if SZ_USE_CUDA
        bench_unary(env, "needleman_wunsch_cuda:batch"s + std::to_string(batch_size), call_linear_global_baseline,
                    similarities_callable<needleman_wunsch_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_global_accelerated, {blosum62_matrix32, blosum62_linear_cost},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_global_baseline);
        scramble_accelerated_results(results_linear_global_accelerated);

        bench_unary(env, "smith_waterman_cuda:batch"s + std::to_string(batch_size), call_linear_local_baseline,
                    similarities_callable<smith_waterman_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_local_accelerated, {blosum62_matrix32, blosum62_linear_cost},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_local_baseline);
        scramble_accelerated_results(results_linear_local_accelerated);

        bench_unary(env, "affine_needleman_wunsch_cuda:batch"s + std::to_string(batch_size),
                    call_affine_global_baseline,
                    similarities_callable<affine_needleman_wunsch_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_global_accelerated, {blosum62_matrix32, blosum62_affine_cost},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_global_baseline);
        scramble_accelerated_results(results_affine_global_accelerated);

        bench_unary(env, "affine_smith_waterman_cuda:batch"s + std::to_string(batch_size), call_affine_local_baseline,
                    similarities_callable<affine_smith_waterman_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_local_accelerated, {blosum62_matrix32, blosum62_affine_cost},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_local_baseline);
        scramble_accelerated_results(results_affine_local_accelerated);
#endif

#if SZ_USE_HOPPER
        bench_unary(env, "needleman_wunsch_hopper:batch"s + std::to_string(batch_size), call_linear_global_baseline,
                    similarities_callable<needleman_wunsch_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_global_accelerated, {blosum62_matrix32, blosum62_linear_cost},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_global_baseline);
        scramble_accelerated_results(results_linear_global_accelerated);

        bench_unary(env, "smith_waterman_hopper:batch"s + std::to_string(batch_size), call_linear_local_baseline,
                    similarities_callable<smith_waterman_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_local_accelerated, {blosum62_matrix32, blosum62_linear_cost},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_local_baseline);
        scramble_accelerated_results(results_linear_local_accelerated);

        bench_unary(env, "affine_needleman_wunsch_hopper:batch"s + std::to_string(batch_size),
                    call_affine_global_baseline,
                    similarities_callable<affine_needleman_wunsch_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_global_accelerated, {blosum62_matrix32, blosum62_affine_cost},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_global_baseline);
        scramble_accelerated_results(results_affine_global_accelerated);

        bench_unary(env, "affine_smith_waterman_hopper:batch"s + std::to_string(batch_size), call_affine_local_baseline,
                    similarities_callable<affine_smith_waterman_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_local_accelerated, {blosum62_matrix32, blosum62_affine_cost},
                        cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_local_baseline);
        scramble_accelerated_results(results_affine_local_accelerated);
#endif
    }
}

#pragma endregion

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
