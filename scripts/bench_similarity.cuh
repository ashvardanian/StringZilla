/**
 *  @file   bench_similarity.cuh
 *  @brief  Shared code for CPU and GPU batched string similarity kernels.
 */
#include <tuple> // `std::tuple`

#include <fork_union.hpp> // Fork-join scoped thread pool

#include <stringcuzilla/similarity.hpp> // C++ templates for string similarity measures

#if SZ_USE_CUDA
#include <stringcuzilla/similarity.cuh> // Parallel string processing in CUDA
#endif

#include "bench.hpp"

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

using similarities_t = unified_vector<sz_ssize_t>;

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
        : env(env), results(res), engine(eng), extra_args(args...) {
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
        sz::status_t status =
            std::apply([&](auto &&...rest) { return engine(a, b, results.data(), rest...); }, extra_args);
        do_not_optimize(status);

        if (status != sz::status_t::success_k) throw std::runtime_error("Failed to compute Levenshtein distance.");
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

#if SZ_USE_CUDA
    sz::gpu_specs_t specs = *sz::gpu_specs();
#endif
    std::vector<std::size_t> batch_sizes = {1, 32, 1024, 32 * 1024};
#if SZ_DEBUG
    batch_sizes = {1, 2, 32};
#endif
    similarities_t results_baseline, results_accelerated;
    fork_union_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) throw std::runtime_error("Failed to spawn thread pool.");
    static_assert(executor_like<fork_union_t>);

    auto scramble_accelerated_results = [&] {
        std::shuffle(results_accelerated.begin(), results_accelerated.end(), global_random_generator());
    };

    // Let's define some weird scoring schemes for Levenshtein-like distance, that are not unary:
    constexpr linear_gap_costs_t weird_linear {3};
    constexpr affine_gap_costs_t weird_affine {4, 2};
    constexpr uniform_substitution_costs_t weird_uniform {1, 3};

    for (std::size_t batch_size : batch_sizes) {
        results_baseline.resize(batch_size);
        results_accelerated.resize(batch_size);

        auto call_baseline = similarities_callable<levenshtein_serial_t, fork_union_t &>(
            env, results_baseline, levenshtein_serial_t {weird_uniform, weird_linear}, pool);
        auto name_baseline = "levenshtein_serial:batch"s + std::to_string(batch_size);
        bench_result_t baseline = bench_unary(env, name_baseline, call_baseline).log();

        auto call_affine_baseline = similarities_callable<affine_levenshtein_serial_t, fork_union_t &>(
            env, results_baseline, affine_levenshtein_serial_t {weird_uniform, weird_affine}, pool);
        auto name_affine_baseline = "affine_levenshtein_serial:batch"s + std::to_string(batch_size);
        bench_result_t affine_baseline = bench_unary(env, name_affine_baseline, call_affine_baseline).log(baseline);
        sz_unused(affine_baseline);

#if SZ_USE_ICE
        bench_unary(env, "levenshtein_ice:batch"s + std::to_string(batch_size), call_baseline,
                    similarities_callable<levenshtein_ice_t, fork_union_t &>(
                        env, results_accelerated, levenshtein_ice_t {weird_uniform, weird_linear}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(baseline);
        scramble_accelerated_results();
#endif

#if SZ_USE_CUDA
        bench_unary(env, "levenshtein_cuda:batch"s + std::to_string(batch_size), call_baseline,
                    similarities_callable<levenshtein_cuda_t, sz::gpu_specs_t>(
                        env, results_accelerated, levenshtein_cuda_t {weird_uniform, weird_linear}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(baseline);
        scramble_accelerated_results();

        bench_unary(env, "affine_levenshtein_cuda:batch"s + std::to_string(batch_size), call_baseline,
                    similarities_callable<affine_levenshtein_cuda_t, sz::gpu_specs_t>(
                        env, results_accelerated, affine_levenshtein_cuda_t {weird_uniform, weird_affine}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(baseline);
        scramble_accelerated_results();
#endif

#if SZ_USE_KEPLER
        bench_unary(env, "levenshtein_kepler:batch"s + std::to_string(batch_size), call_baseline,
                    similarities_callable<levenshtein_kepler_t, sz::gpu_specs_t>(
                        env, results_accelerated, levenshtein_kepler_t {weird_uniform, weird_linear}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(baseline);
        scramble_accelerated_results();
#endif

        auto call_utf8_baseline = similarities_callable<levenshtein_utf8_serial_t>(
            env, results_baseline, levenshtein_utf8_serial_t {weird_uniform, weird_linear});
        auto name_utf8_baseline = "levenshtein_utf8_serial:batch"s + std::to_string(batch_size);
        bench_result_t utf8_baseline = bench_unary(env, name_utf8_baseline, call_utf8_baseline).log();

#if SZ_USE_ICE
        bench_unary(env, "levenshtein_utf8_ice:batch"s + std::to_string(batch_size), call_utf8_baseline,
                    similarities_callable<levenshtein_utf8_ice_t>(env, results_accelerated,
                                                                  levenshtein_utf8_ice_t {weird_uniform, weird_linear}),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(utf8_baseline);
        scramble_accelerated_results();
#endif
    }
}

void bench_needleman_wunsch(environment_t const &env) {

    using namespace std::string_literals; // for "s" suffix

    constexpr linear_gap_costs_t blosum62_linear_costs {-4};
    auto blosum62_mat = error_costs_26x26ascii_t::blosum62();
    auto blosum62_matrix = blosum62_mat.decompressed();

#if SZ_USE_CUDA
    sz::gpu_specs_t specs = *sz::gpu_specs();
#endif
    std::vector<std::size_t> batch_sizes = {1024 / 32, 1024, 1024 * 32};
#if SZ_DEBUG
    batch_sizes = {1, 2, 32};
#endif
    similarities_t results_baseline, results_accelerated;

    auto scramble_accelerated_results = [&] {
        std::shuffle(results_accelerated.begin(), results_accelerated.end(), global_random_generator());
    };

    for (std::size_t batch_size : batch_sizes) {
        results_baseline.resize(batch_size);
        results_accelerated.resize(batch_size);

        auto call_baseline = similarities_callable<needleman_wunsch_serial_t>(env, results_baseline,
                                                                              {blosum62_matrix, blosum62_linear_costs});
        auto name_baseline = "needleman_wunsch_serial:batch"s + std::to_string(batch_size);
        bench_result_t baseline = bench_unary(env, name_baseline, call_baseline).log();

#if SZ_USE_ICE
        bench_unary(env, "needleman_wunsch_ice:batch"s + std::to_string(batch_size), call_baseline,
                    similarities_callable<needleman_wunsch_ice_t>(env, results_accelerated,
                                                                  {blosum62_matrix, blosum62_linear_costs}),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(baseline);
        scramble_accelerated_results();
#endif

#if SZ_USE_CUDA && 0
        bench_unary(env, "needleman_wunsch_cuda:batch"s + std::to_string(batch_size), call_baseline,
                    similarities_callable<needleman_wunsch_cuda_t, sz::gpu_specs_t>(
                        env, results_accelerated, {blosum62_matrix, blosum62_linear_costs}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(baseline);
        scramble_accelerated_results();
#endif
    }
}

#pragma endregion

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian