/**
 *  @file bench/container.cpp
 *  @author Ash Vardanian
 *  @date January 4, 2024
 *  @brief Benchmarks STL associative containers with @c std::string_view-compatible keys.
 *
 *  Times lookups of the English words in STL containers ordered, hashed, and compared through the
 *  dispatch points, against the STL defaults and other key classes. The containers built on every
 *  capability's kernels are timed against the serial ones by the `cross_<arch>.cpp` files, through
 *  the adapters in `cross.hpp`.
 *
 *  Memory-bound: associative build and probe are latency-limited over the whole key set, so it
 *  reads the whole file by default.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment
 *  variables are used:
 *  - `STRINGWARS_DATASET=path` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=0` : Reads at most this many dataset bytes; `0` reads the whole
 *    file.
 *  - `STRINGWARS_TOKENS=words` : Tokenization model ("file", "lines", "words", or positive integer
 *    [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_MAX_SECONDS=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=words STRINGWARS_FILTER=map build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size
 *  inputs, like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes
 *  (exactly 1 cache line), your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or
 *  Google Benchmark. Its siblings are `sequence.cpp`, `find.cpp`, `token.cpp`, and `memory.cpp`.
 */
#include <map>           // `std::map`
#include <unordered_map> // `std::unordered_map`

#include <fmt/format.h>

#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

/** Times lookups through the dispatch points against the default STL comparison and hashes. The
 *  containers fill lazily inside each run, so none of them can check another's lookups. */
void bench_associative_lookups_with_different_simd_backends(environment_t const &env) {
    using map_best_t = std::map<std::string_view, unsigned, less_from_sz<cpu_best<sz_order_best>>>;
    using umap_best_t = std::unordered_map<std::string_view, unsigned, hasher_from_sz<cpu_best<sz_hash_best>>,
                                           equal_to_from_sz<cpu_best<sz_equal_best>>>;
    bench_result_t base_map, base_umap;
    {
        auto callable_map = callable_for_associative_lookups<map_best_t>(env);
        base_map = bench_unary(env, "map<sz_order_best>::find", callable_no_op_t(), callable_map,
                               callable_map.preprocessor())
                       .log();
        auto callable_umap = callable_for_associative_lookups<umap_best_t>(env);
        base_umap = bench_unary(env, "unordered_map<sz_hash_best, sz_equal_best>::find", callable_no_op_t(),
                                callable_umap, callable_umap.preprocessor())
                        .log();
    }

    {
        auto callable_map = callable_for_associative_lookups<std::map<std::string_view, unsigned>>(env);
        bench_unary(env, "map::find", callable_no_op_t(), callable_map, callable_map.preprocessor()).log(base_map);
        auto callable_umap = callable_for_associative_lookups<std::unordered_map<std::string_view, unsigned>>(env);
        bench_unary(env, "unordered_map::find", callable_no_op_t(), callable_umap, callable_umap.preprocessor())
            .log(base_umap);
    }
}

struct less_through_std_t {
    using is_transparent = void;
    template <typename first_type_, typename second_type_>
    inline bool operator()(first_type_ const &a, second_type_ const &b) const noexcept {
        return std::less<std::string_view> {}(string_cast<std::string_view>(a), string_cast<std::string_view>(b));
    }
};

struct hash_through_std_t {
    using is_transparent = void;
    template <typename string_like_>
    inline std::size_t operator()(string_like_ const &str) const noexcept {
        return std::hash<std::string_view> {}(string_cast<std::string_view>(str));
    }
};

struct equal_to_through_std_t {
    using is_transparent = void;
    template <typename first_type_, typename second_type_>
    inline bool operator()(first_type_ const &a, second_type_ const &b) const noexcept {
        return std::equal_to<std::string_view> {}(string_cast<std::string_view>(a), string_cast<std::string_view>(b));
    }
};

void bench_associative_lookups_with_different_key_classes(environment_t const &env) {

    // First, benchmark the default STL equality comparison and hashes for `std::string_view` keys
    bench_result_t base_map, base_umap;
    {
        auto callable_map = callable_for_associative_lookups<std::map<std::string_view, unsigned>>(env);
        base_map = bench_unary(env, "map<std::string_view>::find", callable_no_op_t(), callable_map,
                               callable_map.preprocessor())
                       .log();
        auto callable_umap = callable_for_associative_lookups<std::unordered_map<std::string_view, unsigned>>(env);
        base_umap = bench_unary(env, "unordered_map<std::string_view>::find", callable_no_op_t(), callable_umap,
                                callable_umap.preprocessor())
                        .log();
    }

    // Compare that to using `std::string` for keys
    {
        auto callable_map = callable_for_associative_lookups<std::map<std::string, unsigned, less_through_std_t>>(env);
        bench_unary(env, "map<std::string>::find", callable_no_op_t(), callable_map, callable_map.preprocessor())
            .log(base_map);
        auto callable_umap = callable_for_associative_lookups<
            std::unordered_map<std::string, unsigned, hash_through_std_t, equal_to_through_std_t>>(env);
        bench_unary(env, "unordered_map<std::string>::find", callable_no_op_t(), callable_umap,
                    callable_umap.preprocessor())
            .log(base_umap);
    }

    // Try using StringZilla's `sz::string_view_t` for keys
    {
        auto callable_map = callable_for_associative_lookups<std::map<sz::string_view_t, unsigned, less_through_std_t>>(
            env);
        bench_unary(env, "map<sz::string_view_t>::find", callable_no_op_t(), callable_map, callable_map.preprocessor())
            .log(base_map);
        auto callable_umap = callable_for_associative_lookups<
            std::unordered_map<sz::string_view_t, unsigned, hash_through_std_t, equal_to_through_std_t>>(env);
        bench_unary(env, "unordered_map<sz::string_view_t>::find", callable_no_op_t(), callable_umap,
                    callable_umap.preprocessor())
            .log(base_umap);
    }

    // Try StringZilla's "Small String Optimization" class - `sz::string_t`
    {
        auto callable_map = callable_for_associative_lookups<std::map<sz::string_t, unsigned, less_through_std_t>>(env);
        bench_unary(env, "map<sz::string_t>::find", callable_no_op_t(), callable_map, callable_map.preprocessor())
            .log(base_map);
        auto callable_umap = callable_for_associative_lookups<
            std::unordered_map<sz::string_t, unsigned, hash_through_std_t, equal_to_through_std_t>>(env);
        bench_unary(env, "unordered_map<sz::string_t>::find", callable_no_op_t(), callable_umap,
                    callable_umap.preprocessor())
            .log(base_umap);
    }
}

} // namespace

void bench_container(corpora_t &corpora) {
    environment_t const &env = corpora.words();
    fmt::println("Starting associative STL container benchmarks...");
    bench_associative_lookups_with_different_simd_backends(env);
    bench_associative_lookups_with_different_key_classes(env);
}
