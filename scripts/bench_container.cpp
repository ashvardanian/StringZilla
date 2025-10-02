/**
 *  @file   bench_container.cpp
 *  @brief  Benchmarks STL associative containers with @b `std::string_view`-compatible keys.
 *          The program accepts a file path to a dataset, tokenizes it, and benchmarks the lookup operations.
 *
 *  This file is the sibling of `bench_sequence.cpp`, `bench_find.cpp` and `bench_token.cpp`.
 *  It accepts a file with a list of words, constructs associative containers with string keys,
 *  using `std::string`, `std::string_view`, `sz::string_view`, and `sz::string`, and then
 *  evaluates the latency of lookups.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWa.rs, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=words` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWa.rs, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_container_cpp20
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines build_release/stringzilla_bench_container_cpp20
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  build_release/stringzilla_bench_container_cpp20
 *  @endcode
 *
 *  Unlike the full-blown StringWa.rs, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `bench_sequence.cpp`, `bench_token.cpp`, and `bench_memory.cpp`.
 */
#include <map>           // `std::map`
#include <unordered_map> // `std::unordered_map`

#define SZ_USE_MISALIGNED_LOADS (1)
#include "bench.hpp"

using namespace ashvardanian::stringzilla::scripts;

/**
 *  @brief Helper function-like object to order string-view convertible objects with StringZilla.
 *  @see Similar to `std::less<std::string_view>`: https://en.cppreference.com/w/cpp/utility/functional/less
 *  @note Unlike the `sz::less`, the structure below supports different hardware backends.
 */
template <sz_order_t order_>
struct less_from_sz {
    inline bool operator()(std::string_view a, std::string_view b) const noexcept {
        return order_(a.data(), a.size(), b.data(), b.size()) < 0;
    }
};

/**
 *  @brief Helper function-like object to check equality between string-view convertible objects with StringZilla.
 *  @see Similar to `std::equal_to<std::string_view>`: https://en.cppreference.com/w/cpp/utility/functional/equal_to
 *  @note Unlike the `sz::equal_to`, the structure below supports different hardware backends.
 */
template <sz_equal_t equal_>
struct equal_to_from_sz {
    inline bool operator()(std::string_view a, std::string_view b) const noexcept {
        return a.size() == b.size() && equal_(a.data(), b.data(), b.size());
    }
};

/**
 *  @brief Helper function-like object to hash string-view convertible objects with StringZilla.
 *  @see Similar to `hash_through_std_t`: https://en.cppreference.com/w/cpp/utility/functional/hash
 *  @note Unlike the `sz::hash`, the structure below supports different hardware backends.
 */
template <sz_hash_t hash_>
struct hash_from_sz {
    inline std::size_t operator()(std::string_view str) const noexcept { return hash_(str.data(), str.size(), 0); }
};

template <typename container_type_>
struct callable_for_associative_lookups {

    container_type_ container;
    environment_t const &env;

    inline callable_for_associative_lookups(environment_t const &env) noexcept : env(env) {}
    void preprocess() {
        using key_type = typename container_type_::key_type;
        for (std::string_view const &key : env.tokens) container[to_str<key_type>(key)]++;
    }

    /** @brief Helper API to produce a delayed construction lambda. */
    inline auto preprocessor() {
        return [this] { preprocess(); };
    }

    /** @brief The actual lookup operation to be benchmarked. */
    call_result_t operator()(std::size_t token_index) const {
        std::string_view key = env.tokens[token_index];
        auto counter = container.find(key)->second;
        return {key.size(), static_cast<std::size_t>(counter)};
    }
};

/**
 *  @brief Find all inclusions of each given token in the dataset, using various search backends.
 */
void bench_associative_lookups_with_different_simd_backends(environment_t const &env) {

    // First, benchmark the default STL equality comparison and hashes
    bench_result_t base_map, base_umap;
    {
        auto callable_map = callable_for_associative_lookups<std::map<std::string_view, unsigned>>(env);
        base_map = bench_unary(env, "map::find", callable_no_op_t(), callable_map, callable_map.preprocessor()).log();
        auto callable_umap = callable_for_associative_lookups<std::unordered_map<std::string_view, unsigned>>(env);
        base_umap =
            bench_unary(env, "unordered_map::find", callable_no_op_t(), callable_umap, callable_umap.preprocessor())
                .log();
    }

    // Conditionally include SIMD-accelerated backends
#if SZ_USE_SKYLAKE
    {
        auto callable_map =
            callable_for_associative_lookups<std::map<std::string_view, unsigned, less_from_sz<sz_order_skylake>>>(env);
        bench_unary(env, "map<sz_order_skylake>::find", callable_no_op_t(), callable_map, callable_map.preprocessor())
            .log(base_map);
        auto callable_umap = callable_for_associative_lookups<std::unordered_map<
            std::string_view, unsigned, hash_from_sz<sz_hash_skylake>, equal_to_from_sz<sz_equal_skylake>>>(env);
        bench_unary(env, "unordered_map<sz_hash_skylake, sz_equal_skylake>::find", callable_no_op_t(), callable_umap,
                    callable_umap.preprocessor())
            .log(base_umap);
    }

#endif
#if SZ_USE_HASWELL
    {
        auto callable_map =
            callable_for_associative_lookups<std::map<std::string_view, unsigned, less_from_sz<sz_order_haswell>>>(env);
        bench_unary(env, "map<sz_order_haswell>::find", callable_no_op_t(), callable_map, callable_map.preprocessor())
            .log(base_map);
        auto callable_umap = callable_for_associative_lookups<std::unordered_map<
            std::string_view, unsigned, hash_from_sz<sz_hash_westmere>, equal_to_from_sz<sz_equal_haswell>>>(env);
        bench_unary(env, "unordered_map<sz_hash_westmere, sz_equal_haswell>::find", callable_no_op_t(), callable_umap,
                    callable_umap.preprocessor())
            .log(base_umap);
    }
#endif
#if SZ_USE_NEON_AES
    {
        auto callable_map =
            callable_for_associative_lookups<std::map<std::string_view, unsigned, less_from_sz<sz_order_neon>>>(env);
        bench_unary(env, "map<sz_order_neon>::find", callable_no_op_t(), callable_map, callable_map.preprocessor())
            .log(base_map);
        auto callable_umap =
            callable_for_associative_lookups<std::unordered_map<std::string_view, unsigned, hash_from_sz<sz_hash_neon>,
                                                                equal_to_from_sz<sz_equal_neon>>>(env);
        bench_unary(env, "unordered_map<sz_hash_neon, sz_equal_neon>::find", callable_no_op_t(), callable_umap,
                    callable_umap.preprocessor())
            .log(base_umap);
    }
#endif
}

struct less_through_std_t {
    using is_transparent = void;
    template <typename first_type_, typename second_type_>
    inline bool operator()(first_type_ const &a, second_type_ const &b) const noexcept {
        return std::less<std::string_view> {}(to_str<std::string_view>(a), to_str<std::string_view>(b));
    }
};

struct hash_through_std_t {
    using is_transparent = void;
    template <typename string_like_>
    inline std::size_t operator()(string_like_ const &str) const noexcept {
        return std::hash<std::string_view> {}(to_str<std::string_view>(str));
    }
};

struct equal_to_through_std_t {
    using is_transparent = void;
    template <typename first_type_, typename second_type_>
    inline bool operator()(first_type_ const &a, second_type_ const &b) const noexcept {
        return std::equal_to<std::string_view> {}(to_str<std::string_view>(a), to_str<std::string_view>(b));
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

    // Try using StringZilla's `sz::string_view` for keys
    {
        auto callable_map =
            callable_for_associative_lookups<std::map<sz::string_view, unsigned, less_through_std_t>>(env);
        bench_unary(env, "map<sz::string_view>::find", callable_no_op_t(), callable_map, callable_map.preprocessor())
            .log(base_map);
        auto callable_umap = callable_for_associative_lookups<
            std::unordered_map<sz::string_view, unsigned, hash_through_std_t, equal_to_through_std_t>>(env);
        bench_unary(env, "unordered_map<sz::string_view>::find", callable_no_op_t(), callable_umap,
                    callable_umap.preprocessor())
            .log(base_umap);
    }

    // Try StringZilla's "Small String Optimization" class - `sz::string`
    {
        auto callable_map = callable_for_associative_lookups<std::map<sz::string, unsigned, less_through_std_t>>(env);
        bench_unary(env, "map<sz::string>::find", callable_no_op_t(), callable_map, callable_map.preprocessor())
            .log(base_map);
        auto callable_umap = callable_for_associative_lookups<
            std::unordered_map<sz::string, unsigned, hash_through_std_t, equal_to_through_std_t>>(env);
        bench_unary(env, "unordered_map<sz::string>::find", callable_no_op_t(), callable_umap,
                    callable_umap.preprocessor())
            .log(base_umap);
    }
}

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZilla!\n");

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "leipzig1M.txt",                   //
        environment_t::tokenization_t::words_k);

    std::printf("Starting associative STL container benchmarks...\n");
    bench_associative_lookups_with_different_simd_backends(env);
    bench_associative_lookups_with_different_key_classes(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}