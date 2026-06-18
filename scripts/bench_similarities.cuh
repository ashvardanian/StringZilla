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
using ashvardanian::stringzillas::affine_levenshtein_neon_t;
using ashvardanian::stringzillas::levenshtein_neon_t;
using ashvardanian::stringzillas::levenshtein_utf8_neon_t;

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

using ashvardanian::stringzillas::strided_rows;

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

/**
 *  @brief Calls an engine and reads its status entirely inside one @b `[[gnu::noinline]]` frame.
 *
 *  @note `read_engine_status_` alone is not enough: when the optimizer folds the engine's return-by-value
 *        into the caller it can corrupt the status @b before the read runs (seen with g++ trunk on the Ice
 *        Lake instantiations in this large TU, and with NVCC 12.x on the affine `cuda_status_t`). Performing
 *        the call and the read together in a non-inlined frame reproduces the same code path as a separate-TU
 *        call, which the library validates as correct in isolation.
 */
template <typename engine_type_, typename queries_type_, typename candidates_type_, typename... rest_types_>
[[gnu::noinline]] status_t invoke_engine_status_(engine_type_ &engine, queries_type_ const &queries,
                                                 candidates_type_ const &candidates, strided_rows<sz_ssize_t> results,
                                                 rest_types_ &...rest) noexcept {
    auto status = engine(queries, candidates, results, rest...);
    do_not_optimize(status);
    return read_engine_status_(status);
}

#pragma region Levenshtein Distance and Alignment Scores

/**
 *  @brief Describes one benchmark workload: how many queries, how many candidates, and the pairing mode.
 *
 *  @b all_pairs : score every query against every candidate, a full `queries x candidates` cross-product tile.
 *  This is the path that exercises the inter-/intra-sequence tiling and lane packing of the SIMD/GPU backends,
 *  so the two dimensions are scaled @b independently (the candidate axis is no longer capped at a constant).
 *
 *  @b pairwise : score only the `min(queries, candidates)` diagonal pairs `(query_i, candidate_i)`, one engine
 *  call per pair. This mirrors how a per-pair library is driven, making the GCUPS directly comparable.
 */
struct shape_t {
    std::size_t queries = 0;
    std::size_t candidates = 0;
    bool all_pairs = true;

    /** @brief Number of scored pairs: the full tile for all-pairs, the diagonal for pairwise. */
    std::size_t scored_pairs() const noexcept {
        return all_pairs ? queries * candidates : (std::min)(queries, candidates);
    }

    /** @brief A compact label fragment such as `q1024xc256:allpairs` for the benchmark name. */
    std::string label() const {
        return "q" + std::to_string(queries) + "xc" + std::to_string(candidates) +
               (all_pairs ? ":allpairs" : ":pairwise");
    }
};

/**
 *  @brief Builds the list of benchmark shapes, scaling @b both dimensions under a cell budget.
 *
 *  We deliberately scale the candidate axis alongside the query axis (instead of pinning it to a constant) so the
 *  cross-product tiling, lane packing and tier routing are all exercised. The all-pairs shapes grow squarely up
 *  to a per-tile pair budget; a separate pairwise (diagonal) batch gives an apples-to-apples per-pair baseline.
 *
 *  @sa `STRINGWARS_BATCH` (`env.batch_sizes_override`): when set, each override `N` is mapped onto a square-ish
 *      all-pairs shape `q(N) x c(min(N, budget/N))` plus a `q(N) x c(N)` pairwise diagonal, honoring the dataset
 *      size and the cell budget.
 */
inline std::vector<shape_t> make_similarity_shapes(environment_t const &env) {
    // Cap the total scored pairs per tile so the largest shapes stay tractable; keep every block strictly
    // smaller than the dataset (the callable needs `queries + candidates < tokens`).
    std::size_t const pair_budget = SZ_DEBUG ? 256u : 256u * 1024u;
    std::size_t const max_block = env.tokens.size() > 1 ? env.tokens.size() / 2 : 1;

    auto clamp_block = [&](std::size_t value) -> std::size_t {
        return (std::max<std::size_t>)(1, (std::min)(value, max_block));
    };
    // Build a tile whose two blocks together fit strictly inside the dataset (the callable guard requires
    // `queries + candidates < tokens`), shrinking the candidate axis first since the query axis carries the label.
    auto fit_all_pairs = [&](std::size_t queries_request) -> shape_t {
        std::size_t const queries = clamp_block(queries_request);
        std::size_t candidates = clamp_block((std::max<std::size_t>)(1, pair_budget / (std::max<std::size_t>)(1, queries)));
        if (env.tokens.size() > queries + 1)
            candidates = (std::min)(candidates, env.tokens.size() - queries - 1);
        return shape_t {queries, (std::max<std::size_t>)(1, candidates), true};
    };
    // A diagonal of `queries` independent `1 x 1` pairs; both blocks are `queries` long, so cap at just under half.
    auto fit_pairwise = [&](std::size_t queries_request) -> shape_t {
        std::size_t const half = env.tokens.size() > 2 ? (env.tokens.size() - 1) / 2 : 1;
        std::size_t const queries = (std::max<std::size_t>)(1, (std::min)(clamp_block(queries_request), half));
        return shape_t {queries, queries, false};
    };

    std::vector<shape_t> shapes;
    if (!env.batch_sizes_override.empty()) {
        for (std::size_t batch : env.batch_sizes_override) {
            shapes.push_back(fit_all_pairs(batch));
            shapes.push_back(fit_pairwise(batch));
        }
        return shapes;
    }

#if SZ_DEBUG
    std::size_t const query_sizes[] = {1, 2, 16};
#else
    std::size_t const query_sizes[] = {1, 64, 1024, 16 * 1024};
#endif
    for (std::size_t queries_request : query_sizes) {
        shapes.push_back(fit_all_pairs(queries_request)); // all-pairs, both dims scaled under the cell budget
        shapes.push_back(fit_pairwise(queries_request));  // pairwise diagonal, apples-to-apples per-pair baseline
    }
    return shapes;
}

/**
 *  @brief Wraps a hardware-specific similarity backend into something @b `bench_unary`-compatible.
 *
 *  Each invocation draws @b two independent, randomly-sampled blocks out of `env.tokens` - a block of
 *  `shape.queries` query tokens and a block of `shape.candidates` candidate tokens - so consecutive batches and
 *  consecutive engines see varied, non-degenerate inputs instead of one fixed candidate tail.
 *
 *  In @b all_pairs mode the engine scores the full `queries x candidates` matrix in one call. In @b pairwise
 *  mode the callable issues `min(queries, candidates)` independent `1 x 1` engine calls along the diagonal,
 *  matching a per-pair library. Either way the reported `operations` count is the exact Cell-Updates total - the
 *  aggregate `sum over every scored (query, candidate) pair of len(query) * len(candidate)` - so
 *  GCUPS = aggregate_cells / elapsed_seconds.
 *
 *  Sampling is a pure function of `(global_random_seed(), shape, batch_index)`; it never touches the shared
 *  stateful `global_random_generator()`. This is load-bearing: `bench_unary` drives the baseline and the
 *  accelerated callable with the @b same `batch_index` and then compares their result buffers, so both must
 *  observe byte-identical tiles for a given index.
 */
template <typename engine_type_, typename... extra_args_>
struct similarities_callable {
    using engine_t = engine_type_;

    environment_t const &env;
    similarities_t &results;
    shape_t shape = {};
    engine_t engine = {};
    std::tuple<extra_args_...> extra_args = {};

    similarities_callable(environment_t const &env, similarities_t &res, shape_t shape, engine_t eng = {},
                          extra_args_... args)
        : env(env), results(res), shape(shape), engine(std::move(eng)),
          extra_args(std::forward<extra_args_>(args)...) {
        if (env.tokens.size() <= shape.queries + shape.candidates)
            throw std::runtime_error("Cross-product tile is too large for the dataset.");
        if (results.size() < shape.scored_pairs())
            throw std::runtime_error("Results buffer is smaller than the number of scored pairs.");
    }

    /** @brief Deterministically picks a contiguous block start in `[0, count_total - block_size]`. */
    std::size_t sample_block_start_(std::size_t block_size, std::uint64_t salt) const noexcept {
        std::size_t const last_start = env.tokens.size() - block_size;
        std::mt19937_64 generator(static_cast<std::uint64_t>(global_random_seed()) ^ salt);
        return std::uniform_int_distribution<std::size_t>(0, last_start)(generator);
    }

    call_result_t operator()(std::size_t batch_index) noexcept(false) {
        // Draw the query block and the candidate block from independent pseudo-random offsets, salted by the
        // batch index so successive batches see different tiles, yet reproducibly for a fixed seed and index.
        std::uint64_t const batch_salt = static_cast<std::uint64_t>(batch_index) * 0x9E3779B97F4A7C15ull;
        std::size_t const queries_start = sample_block_start_(shape.queries, batch_salt ^ 0x1111111111111111ull);
        std::size_t candidates_start = sample_block_start_(shape.candidates, batch_salt ^ 0x2222222222222222ull);

        // Nudge the candidate block off the query block when the dataset is large enough to keep them disjoint.
        bool const overlaps = candidates_start + shape.candidates > queries_start &&
                              queries_start + shape.queries > candidates_start;
        if (overlaps && env.tokens.size() > shape.queries + shape.candidates) {
            std::size_t const after_queries = queries_start + shape.queries;
            candidates_start = after_queries + shape.candidates <= env.tokens.size()
                                   ? after_queries
                                   : queries_start >= shape.candidates ? queries_start - shape.candidates : 0;
        }

        return operator()({env.tokens.data() + queries_start, shape.queries},
                          {env.tokens.data() + candidates_start, shape.candidates});
    }

    call_result_t operator()(std::span<token_view_t const> queries_block,
                             std::span<token_view_t const> candidates_block) noexcept(false) {
        if (shape.all_pairs) score_all_pairs_(queries_block, candidates_block);
        else score_pairwise_(queries_block, candidates_block);
        do_not_optimize(results);

        // Aggregate the exact Cell-Updates over only the scored pairs: the whole tile for all-pairs, the
        // diagonal for pairwise. The denominator is the sum of `len(query) * len(candidate)` products.
        std::size_t bytes_passed = 0, cells_passed = 0;
        if (shape.all_pairs) {
            for (std::size_t query_index = 0; query_index < shape.queries; ++query_index)
                for (std::size_t candidate_index = 0; candidate_index < shape.candidates; ++candidate_index) {
                    bytes_passed += queries_block[query_index].size() + candidates_block[candidate_index].size();
                    cells_passed += queries_block[query_index].size() * candidates_block[candidate_index].size();
                }
        }
        else {
            std::size_t const pairs = shape.scored_pairs();
            for (std::size_t pair_index = 0; pair_index < pairs; ++pair_index) {
                bytes_passed += queries_block[pair_index].size() + candidates_block[pair_index].size();
                cells_passed += queries_block[pair_index].size() * candidates_block[pair_index].size();
            }
        }
        call_result_t call_result;
        call_result.bytes_passed = bytes_passed;
        call_result.operations = cells_passed;
        call_result.inputs_processed = shape.scored_pairs();
        call_result.check_value = reinterpret_cast<check_value_t>(&results);
        return call_result;
    }

  private:
    /** @brief Runs the engine call + status read behind the non-inlined `invoke_engine_status_` boundary. */
    void run_engine_(std::span<token_view_t const> queries_block, std::span<token_view_t const> candidates_block,
                     strided_rows<sz_ssize_t> results_matrix) noexcept(false) {
        status_t const status_code = std::apply(
            [&](auto &&...rest) {
                return invoke_engine_status_(engine, queries_block, candidates_block, results_matrix, rest...);
            },
            extra_args);
        if (status_code != status_t::success_k)
            throw std::runtime_error(std::string("Failed to compute similarity matrix: ") + status_name(status_code));
    }

    /** @brief Full `queries x candidates` cross-product in one engine call. */
    void score_all_pairs_(std::span<token_view_t const> queries_block,
                          std::span<token_view_t const> candidates_block) noexcept(false) {
        strided_rows<sz_ssize_t> const results_matrix {results.data(), shape.queries, shape.candidates,
                                                       shape.candidates};
        run_engine_(queries_block, candidates_block, results_matrix);
    }

    /** @brief `min(queries, candidates)` independent `1 x 1` diagonal pairs, one engine call each. */
    void score_pairwise_(std::span<token_view_t const> queries_block,
                         std::span<token_view_t const> candidates_block) noexcept(false) {
        std::size_t const pairs = shape.scored_pairs();
        for (std::size_t pair_index = 0; pair_index < pairs; ++pair_index) {
            strided_rows<sz_ssize_t> const results_cell {results.data() + pair_index, 1, 1, 1};
            run_engine_(queries_block.subspan(pair_index, 1), candidates_block.subspan(pair_index, 1), results_cell);
        }
    }
};

struct similarities_equality_t {
    bool operator()(check_value_t const &left, check_value_t const &right) const noexcept {
        similarities_t const &left_matrix = *reinterpret_cast<similarities_t const *>(left);
        similarities_t const &right_matrix = *reinterpret_cast<similarities_t const *>(right);
        if (left_matrix.size() != right_matrix.size()) return false;
        for (std::size_t cell_index = 0; cell_index < left_matrix.size(); ++cell_index)
            if (left_matrix[cell_index] != right_matrix[cell_index]) {
                std::printf("Mismatch at cell %zu: %zd != %zd\n", cell_index, left_matrix[cell_index],
                            right_matrix[cell_index]);
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
    std::vector<shape_t> shapes = make_similarity_shapes(env);
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
        for (shape_t const &shape : shapes) {
            std::string const shape_label = shape.label();
            std::size_t const matrix_size = shape.all_pairs ? shape.queries * shape.candidates : shape.scored_pairs();
            results_linear_baseline.resize(matrix_size), results_linear_accelerated.resize(matrix_size);
            results_affine_baseline.resize(matrix_size), results_affine_accelerated.resize(matrix_size);
            results_utf8_baseline.resize(matrix_size), results_utf8_accelerated.resize(matrix_size);

            auto call_linear_baseline = similarities_callable<levenshtein_serial_t, fu::basic_pool_t &>(
                env, results_linear_baseline, shape,
                levenshtein_serial_t {scheme.uniform, scheme.linear}, pool);
            auto name_linear_baseline = "levenshtein_serial_"s + scheme.tag + ":" + shape_label;
            bench_result_t linear_baseline = bench_unary(env, name_linear_baseline, call_linear_baseline).log();

            auto call_utf8_baseline = similarities_callable<levenshtein_utf8_serial_t>(
                env, results_utf8_baseline, shape,
                levenshtein_utf8_serial_t {scheme.uniform, scheme.linear});
            auto name_utf8_baseline = "levenshtein_utf8_serial_"s + scheme.tag + ":" + shape_label;
            bench_result_t utf8_baseline = bench_unary(env, name_utf8_baseline, call_utf8_baseline).log();

            auto call_affine_baseline = similarities_callable<affine_levenshtein_serial_t, fu::basic_pool_t &>(
                env, results_affine_baseline, shape,
                affine_levenshtein_serial_t {scheme.uniform, scheme.affine}, pool);
            auto name_affine_baseline = "affine_levenshtein_serial_"s + scheme.tag + ":" + shape_label;
            bench_result_t affine_baseline =
                bench_unary(env, name_affine_baseline, call_affine_baseline).log(linear_baseline);
            sz_unused_(affine_baseline);

#if SZ_USE_ICELAKE
            bench_unary(env, "levenshtein_icelake_"s + scheme.tag + ":" + shape_label,
                        call_linear_baseline,
                        similarities_callable<levenshtein_icelake_t, fu::basic_pool_t &>(
                            env, results_linear_accelerated, shape,
                            levenshtein_icelake_t {scheme.uniform, scheme.linear}, pool),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline);
            scramble_accelerated_results(results_linear_accelerated);

            bench_unary(env, "affine_levenshtein_icelake_"s + scheme.tag + ":" + shape_label,
                        call_affine_baseline,
                        similarities_callable<affine_levenshtein_icelake_t, fu::basic_pool_t &>(
                            env, results_affine_accelerated, shape,
                            affine_levenshtein_icelake_t {scheme.uniform, scheme.affine}, pool),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline, affine_baseline);
            scramble_accelerated_results(results_affine_accelerated);

            bench_unary(env, "levenshtein_utf8_icelake_"s + scheme.tag + ":" + shape_label,
                        call_utf8_baseline,
                        similarities_callable<levenshtein_utf8_icelake_t>(
                            env, results_utf8_accelerated, shape,
                            levenshtein_utf8_icelake_t {scheme.uniform, scheme.linear}),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(utf8_baseline);
            scramble_accelerated_results(results_utf8_accelerated);
#endif

#if SZ_USE_NEON
            bench_unary(env, "levenshtein_neon_"s + scheme.tag + ":" + shape_label,
                        call_linear_baseline,
                        similarities_callable<levenshtein_neon_t, fu::basic_pool_t &>(
                            env, results_linear_accelerated, shape,
                            levenshtein_neon_t {scheme.uniform, scheme.linear}, pool),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline);
            scramble_accelerated_results(results_linear_accelerated);

            bench_unary(env, "affine_levenshtein_neon_"s + scheme.tag + ":" + shape_label,
                        call_affine_baseline,
                        similarities_callable<affine_levenshtein_neon_t, fu::basic_pool_t &>(
                            env, results_affine_accelerated, shape,
                            affine_levenshtein_neon_t {scheme.uniform, scheme.affine}, pool),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline, affine_baseline);
            scramble_accelerated_results(results_affine_accelerated);

            bench_unary(env, "levenshtein_utf8_neon_"s + scheme.tag + ":" + shape_label,
                        call_utf8_baseline,
                        similarities_callable<levenshtein_utf8_neon_t>(
                            env, results_utf8_accelerated, shape,
                            levenshtein_utf8_neon_t {scheme.uniform, scheme.linear}),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(utf8_baseline);
            scramble_accelerated_results(results_utf8_accelerated);
#endif

#if SZ_USE_CUDA
            bench_unary(env, "levenshtein_cuda_"s + scheme.tag + ":" + shape_label,
                        call_linear_baseline,
                        similarities_callable<levenshtein_cuda_t, cuda_executor_t, gpu_specs_t>(
                            env, results_linear_accelerated, shape,
                            levenshtein_cuda_t {scheme.uniform, scheme.linear}, cuda_executor_t {}, specs),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline);
            scramble_accelerated_results(results_linear_accelerated);

            bench_unary(env, "affine_levenshtein_cuda_"s + scheme.tag + ":" + shape_label,
                        call_affine_baseline,
                        similarities_callable<affine_levenshtein_cuda_t, cuda_executor_t, gpu_specs_t>(
                            env, results_affine_accelerated, shape,
                            affine_levenshtein_cuda_t {scheme.uniform, scheme.affine}, cuda_executor_t {}, specs),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline, affine_baseline);
            scramble_accelerated_results(results_affine_accelerated);
#endif

#if SZ_USE_KEPLER
            bench_unary(env, "levenshtein_kepler_"s + scheme.tag + ":" + shape_label,
                        call_linear_baseline,
                        similarities_callable<levenshtein_kepler_t, cuda_executor_t, gpu_specs_t>(
                            env, results_linear_accelerated, shape,
                            levenshtein_kepler_t {scheme.uniform, scheme.linear}, cuda_executor_t {}, specs),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline);
            scramble_accelerated_results(results_linear_accelerated);

            bench_unary(env, "affine_levenshtein_kepler_"s + scheme.tag + ":" + shape_label,
                        call_affine_baseline,
                        similarities_callable<affine_levenshtein_kepler_t, cuda_executor_t, gpu_specs_t>(
                            env, results_affine_accelerated, shape,
                            affine_levenshtein_kepler_t {scheme.uniform, scheme.affine}, cuda_executor_t {}, specs),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline, affine_baseline);
            scramble_accelerated_results(results_affine_accelerated);
#endif

#if SZ_USE_HOPPER
            bench_unary(env, "levenshtein_hopper_"s + scheme.tag + ":" + shape_label,
                        call_linear_baseline,
                        similarities_callable<levenshtein_hopper_t, cuda_executor_t, gpu_specs_t>(
                            env, results_linear_accelerated, shape,
                            levenshtein_hopper_t {scheme.uniform, scheme.linear}, cuda_executor_t {}, specs),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(linear_baseline);
            scramble_accelerated_results(results_linear_accelerated);

            bench_unary(env, "affine_levenshtein_hopper_"s + scheme.tag + ":" + shape_label,
                        call_affine_baseline,
                        similarities_callable<affine_levenshtein_hopper_t, cuda_executor_t, gpu_specs_t>(
                            env, results_affine_accelerated, shape,
                            affine_levenshtein_hopper_t {scheme.uniform, scheme.affine}, cuda_executor_t {}, specs),
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
    std::vector<shape_t> shapes = make_similarity_shapes(env);
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

    for (shape_t const &shape : shapes) {
        std::string const shape_label = shape.label();
        std::size_t const matrix_size = shape.all_pairs ? shape.queries * shape.candidates : shape.scored_pairs();
        results_linear_global_baseline.resize(matrix_size), results_linear_global_accelerated.resize(matrix_size);
        results_affine_global_baseline.resize(matrix_size), results_affine_global_accelerated.resize(matrix_size);
        results_linear_local_baseline.resize(matrix_size), results_linear_local_accelerated.resize(matrix_size);
        results_affine_local_baseline.resize(matrix_size), results_affine_local_accelerated.resize(matrix_size);

        auto call_linear_global_baseline = similarities_callable<needleman_wunsch_serial_t, fu::basic_pool_t &>(
            env, results_linear_global_baseline, shape,
            {blosum62_matrix32, blosum62_linear_cost}, pool);
        auto name_linear_global_baseline = "needleman_wunsch_serial:"s + shape_label;
        bench_result_t linear_global_baseline =
            bench_unary(env, name_linear_global_baseline, call_linear_global_baseline).log();

        auto call_linear_local_baseline = similarities_callable<smith_waterman_serial_t, fu::basic_pool_t &>(
            env, results_linear_local_baseline, shape,
            {blosum62_matrix32, blosum62_linear_cost}, pool);
        auto name_linear_local_baseline = "smith_waterman_serial:"s + shape_label;
        bench_result_t linear_local_baseline =
            bench_unary(env, name_linear_local_baseline, call_linear_local_baseline).log();

        auto call_affine_global_baseline = similarities_callable<affine_needleman_wunsch_serial_t, fu::basic_pool_t &>(
            env, results_affine_global_baseline, shape,
            {blosum62_matrix32, blosum62_affine_cost}, pool);
        auto name_affine_global_baseline = "affine_needleman_wunsch_serial:"s + shape_label;
        bench_result_t affine_global_baseline =
            bench_unary(env, name_affine_global_baseline, call_affine_global_baseline).log();

        auto call_affine_local_baseline = similarities_callable<affine_smith_waterman_serial_t, fu::basic_pool_t &>(
            env, results_affine_local_baseline, shape,
            {blosum62_matrix32, blosum62_affine_cost}, pool);
        auto name_affine_local_baseline = "affine_smith_waterman_serial:"s + shape_label;
        bench_result_t affine_local_baseline =
            bench_unary(env, name_affine_local_baseline, call_affine_local_baseline).log();

#if SZ_USE_ICELAKE
        bench_unary(env, "needleman_wunsch_icelake:"s + shape_label, call_linear_global_baseline,
                    similarities_callable<needleman_wunsch_icelake_t, fu::basic_pool_t &>(
                        env, results_linear_global_accelerated, shape,
                        {blosum62_matrix32, blosum62_linear_cost}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_global_baseline);
        scramble_accelerated_results(results_linear_global_accelerated);

        bench_unary(env, "smith_waterman_icelake:"s + shape_label, call_linear_local_baseline,
                    similarities_callable<smith_waterman_icelake_t, fu::basic_pool_t &>(
                        env, results_linear_local_accelerated, shape,
                        {blosum62_matrix32, blosum62_linear_cost}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_local_baseline);
        scramble_accelerated_results(results_linear_local_accelerated);

        bench_unary(env, "affine_needleman_wunsch_icelake:"s + shape_label,
                    call_affine_global_baseline,
                    similarities_callable<affine_needleman_wunsch_icelake_t, fu::basic_pool_t &>(
                        env, results_affine_global_accelerated, shape,
                        {blosum62_matrix32, blosum62_affine_cost}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_global_baseline);
        scramble_accelerated_results(results_affine_global_accelerated);

        bench_unary(env, "affine_smith_waterman_icelake:"s + shape_label,
                    call_affine_local_baseline,
                    similarities_callable<affine_smith_waterman_icelake_t, fu::basic_pool_t &>(
                        env, results_affine_local_accelerated, shape,
                        {blosum62_matrix32, blosum62_affine_cost}, pool),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_local_baseline);
        scramble_accelerated_results(results_affine_local_accelerated);
#endif

#if SZ_USE_CUDA
        bench_unary(env, "needleman_wunsch_cuda:"s + shape_label, call_linear_global_baseline,
                    similarities_callable<needleman_wunsch_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_global_accelerated, shape,
                        {blosum62_matrix32, blosum62_linear_cost}, cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_global_baseline);
        scramble_accelerated_results(results_linear_global_accelerated);

        bench_unary(env, "smith_waterman_cuda:"s + shape_label, call_linear_local_baseline,
                    similarities_callable<smith_waterman_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_local_accelerated, shape,
                        {blosum62_matrix32, blosum62_linear_cost}, cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_local_baseline);
        scramble_accelerated_results(results_linear_local_accelerated);

        bench_unary(env, "affine_needleman_wunsch_cuda:"s + shape_label,
                    call_affine_global_baseline,
                    similarities_callable<affine_needleman_wunsch_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_global_accelerated, shape,
                        {blosum62_matrix32, blosum62_affine_cost}, cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_global_baseline);
        scramble_accelerated_results(results_affine_global_accelerated);

        bench_unary(env, "affine_smith_waterman_cuda:"s + shape_label, call_affine_local_baseline,
                    similarities_callable<affine_smith_waterman_cuda_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_local_accelerated, shape,
                        {blosum62_matrix32, blosum62_affine_cost}, cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_local_baseline);
        scramble_accelerated_results(results_affine_local_accelerated);
#endif

#if SZ_USE_HOPPER
        bench_unary(env, "needleman_wunsch_hopper:"s + shape_label, call_linear_global_baseline,
                    similarities_callable<needleman_wunsch_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_global_accelerated, shape,
                        {blosum62_matrix32, blosum62_linear_cost}, cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_global_baseline);
        scramble_accelerated_results(results_linear_global_accelerated);

        bench_unary(env, "smith_waterman_hopper:"s + shape_label, call_linear_local_baseline,
                    similarities_callable<smith_waterman_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_linear_local_accelerated, shape,
                        {blosum62_matrix32, blosum62_linear_cost}, cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(linear_local_baseline);
        scramble_accelerated_results(results_linear_local_accelerated);

        bench_unary(env, "affine_needleman_wunsch_hopper:"s + shape_label,
                    call_affine_global_baseline,
                    similarities_callable<affine_needleman_wunsch_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_global_accelerated, shape,
                        {blosum62_matrix32, blosum62_affine_cost}, cuda_executor_t {}, specs),
                    callable_no_op_t {},        // preprocessing
                    similarities_equality_t {}) // equality check
            .log(affine_global_baseline);
        scramble_accelerated_results(results_affine_global_accelerated);

        bench_unary(env, "affine_smith_waterman_hopper:"s + shape_label, call_affine_local_baseline,
                    similarities_callable<affine_smith_waterman_hopper_t, cuda_executor_t, gpu_specs_t>(
                        env, results_affine_local_accelerated, shape,
                        {blosum62_matrix32, blosum62_affine_cost}, cuda_executor_t {}, specs),
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
