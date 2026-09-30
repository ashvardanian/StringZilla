/**
 *  @file bench/memory.cpp
 *  @author Ash Vardanian
 *  @date September 28, 2024
 *  @brief Benchmarks memory operations: copying, moving, resetting, and lookup-table conversion.
 *
 *  Times the memory dispatch points over the English lines, against their LibC and STL analogs,
 *  using the tokens only for size references to mimic real-world scenarios dealing with individual
 *  strings of different lengths. Every capability's kernels are timed against the serial ones by
 *  the `cross_<arch>.cpp` files.
 *
 *  Memory-bound: the copy, move, and fill primitives are pure bandwidth, so it reads the whole file
 *  by default and a larger buffer measures throughput truer.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment
 *  variables are used:
 *  - `STRINGWARS_DATASET=path` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=0` : Reads at most this many dataset bytes; `0` reads the whole
 *    file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer
 *    [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_MAX_SECONDS=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines STRINGWARS_FILTER='copy|move|fill|lookup' \
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size
 *  inputs, like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes
 *  (exactly 1 cache line), your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or
 *  Google Benchmark. This file is the sibling of `find.cpp`, `token.cpp`, and `sequence.cpp`.
 */
#include <algorithm> // `std::transform`, `std::generate`
#include <random>    // `std::minstd_rand`

#include <fmt/format.h>

#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

#pragma region MemCpy

sz_status_t memcpy_like_sz(sz_ptr_t output, sz_cptr_t input, sz_size_t length, void *) {
    std::memcpy(output, input, length);
    return sz_success_k;
}

/**
 *  @brief Benchmarks @c memcpy -like operations into @b aligned and @b shifted misaligned output.
 *
 *  In the aligned case we copy a random part of the input string into the start of a matching cache
 *  line in the output. In the unaligned case we also locate a matching cache line in the output,
 *  but shift by one to guarantee unaligned writes.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or
 *  output. So the dispatch point can be compared against the baseline @c memcpy function.
 */
void bench_copy(environment_t const &env) {
    dataset_copy_t output(env);
    sz_ptr_t o = output.data();

    bench_result_t align = bench_unary(env, "sz_copy_best(align)", copy_from_sz<cpu_best<sz_copy_best>> {env, o}).log();
    bench_result_t shift =
        bench_unary(env, "sz_copy_best(shift)", copy_from_sz<cpu_best<sz_copy_best>, 1> {env, o}).log(align);

    bench_unary(env, "std::memcpy(align)", copy_from_sz<memcpy_like_sz> {env, o}).log(align);
    bench_unary(env, "std::memcpy(shift)", copy_from_sz<memcpy_like_sz, 1> {env, o}).log(align, shift);
}

#pragma endregion MemCpy

#pragma region MemMove

sz_status_t memmove_like_sz(sz_ptr_t output, sz_cptr_t input, sz_size_t length, void *) {
    std::memmove(output, input, length);
    return sz_success_k;
}

/**
 *  @brief Benchmarks @c memmove -like operations shuffling regions of output memory back and forth.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or
 *  output. This is achieved by performing a combination of a forward and a backward move. So the
 *  dispatch point can be compared against the baseline @c memmove function.
 */
void bench_move(environment_t const &env) {
    dataset_copy_t output(env);
    sz_ptr_t o = output.data();

    // Shift forward by a single byte or a single cache line
    bench_result_t byte = bench_unary(env, "sz_move_best(by1)", move_from_sz<cpu_best<sz_move_best>, 1> {env, o}).log();
    bench_result_t page =
        bench_unary(env, "sz_move_best(by64)", move_from_sz<cpu_best<sz_move_best>, 64> {env, o}).log(byte);

    bench_unary(env, "std::memmove(by1)", move_from_sz<memmove_like_sz, 1> {env, o}).log(byte);
    bench_unary(env, "std::memmove(by64)", move_from_sz<memmove_like_sz, 64> {env, o}).log(byte, page);
}

#pragma endregion MemMove

#pragma region Broadcasting Constants with MemSet

sz_status_t memset_like_sz(sz_ptr_t output, sz_size_t length, sz_u8_t value, void *) {
    std::memset(output, value, length);
    return sz_success_k;
}

/**
 *  @brief The `std::` baseline for @c sz_fill_random_best, measuring generator throughput alone.
 *
 *  Like the kernels behind it, the same nonce replays the same bytes: each call seeds a
 *  @c std::minstd_rand from it, whose one word of state keeps the reseed out of the measurement
 *  where a Mersenne-Twister state fill would not.
 */
sz_status_t generate_like_sz(sz_ptr_t output, sz_size_t length, sz_u64_t nonce, void *) {
    std::minstd_rand generator(static_cast<std::minstd_rand::result_type>(nonce));
    std::uniform_int_distribution<std::uint32_t> distribution(1, 255);
    std::generate(output, output + length, [&]() -> char { return static_cast<char>(distribution(generator)); });
    return sz_success_k;
}

/**
 *  @brief Benchmarks @c memset -like operations overwriting regions of output memory filling them
 *      with the first byte of the input regions or with random @b (reproducible) byte streams.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or
 *  output. So the dispatch points can be compared against the baseline @c memset function.
 */
void bench_fill(environment_t const &env) {
    dataset_copy_t output(env);
    sz_ptr_t o = output.data();

    bench_result_t zeros = bench_unary(env, "sz_fill_best", fill_from_sz<cpu_best<sz_fill_best>> {env, o}).log();
    bench_result_t random =
        bench_unary(env, "sz_fill_random_best", fill_random_from_sz<cpu_best<sz_fill_random_best>> {env, o}).log(zeros);

    // The generators differ, so the random baseline is timed but never validated.
    bench_unary(env, "fill<std::memset>", fill_from_sz<memset_like_sz> {env, o}).log(zeros);
    bench_unary(env, "fill<std::random_device>", fill_random_from_sz<generate_like_sz> {env, o}).log(zeros, random);
}

#pragma endregion Broadcasting Constants with MemSet

#pragma region Lookup Transformations

sz_status_t transform_like_sz(sz_ptr_t output, sz_cptr_t input, sz_size_t length, sz_cptr_t lookup_table, void *) {
    std::transform(input, input + length, output, [=](char c) { return (char)lookup_table[(unsigned char)c]; });
    return sz_success_k;
}

/**
 *  @brief Benchmarks look-up transformations on the provided slices, updating them inplace.
 *
 *  Performs a simple cyclical rotation of the alphabet, to test the performance of the different
 *  "look-up table"-based transformations.
 */
void bench_lookup(environment_t const &env) {
    dataset_copy_t output(env);
    sz_ptr_t o = output.data();
    sz_cptr_t lut = rotated_alphabet();

    bench_result_t zeros =
        bench_unary(env, "sz_lookup_best", lookup_from_sz<cpu_best<sz_lookup_best>> {env, o, lut}).log();
    bench_unary(env, "lookup<std::transform>", lookup_from_sz<transform_like_sz> {env, o, lut}).log(zeros);
}

#pragma endregion Lookup Transformations

} // namespace

void bench_memory(corpora_t &corpora) {
    environment_t const &env = corpora.lines();
    fmt::println("Starting low-level memory-operation benchmarks...");
    bench_copy(env);
    bench_move(env);
    bench_fill(env);
    bench_lookup(env);
}
