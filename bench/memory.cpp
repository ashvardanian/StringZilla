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
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines STRINGWARS_FILTER='copy|move|fill|lookup' \
 *  build_release/stringzilla_bench
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size
 *  inputs, like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes
 *  (exactly 1 cache line), your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGZILLA_STRESS=1 STRINGZILLA_STRESS_TIME_LIMIT=120s STRINGZILLA_STRESS_DIR=logs
 *  build_release/stringzilla_bench
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or
 *  Google Benchmark. This file is the sibling of `find.cpp`, `token.cpp`, and `sequence.cpp`.
 */
#include <algorithm> // `std::transform`, `std::generate`
#include <random>    // `std::minstd_rand`

#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

#pragma region MemCpy

sz_status_t memcpy_like_sz(sz_ptr_t output, sz_cptr_t input, sz_size_t length, sz_stream_t) {
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
void bench_copy(environment_t const &env, corpus_t const &corpus) {
    dataset_copy_t output(corpus);
    sz_ptr_t o = output.data();

    std::optional<row_t> const align = bench_unary(env, corpus, "sz_copy_best(align)",
                                                   copy_from_sz<cpu_best<sz_copy_best>> {corpus, o});
    print(align);
    std::optional<row_t> const shift = bench_unary(env, corpus, "sz_copy_best(shift)",
                                                   copy_from_sz<cpu_best<sz_copy_best>, 1> {corpus, o});
    print(shift, baseline_of(align));

    print(bench_unary(env, corpus, "std::memcpy(align)", copy_from_sz<memcpy_like_sz> {corpus, o}), baseline_of(align));
    print(bench_unary(env, corpus, "std::memcpy(shift)", copy_from_sz<memcpy_like_sz, 1> {corpus, o}),
          baseline_of(shift));
}

#pragma endregion MemCpy

#pragma region MemMove

sz_status_t memmove_like_sz(sz_ptr_t output, sz_cptr_t input, sz_size_t length, sz_stream_t) {
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
void bench_move(environment_t const &env, corpus_t const &corpus) {
    dataset_copy_t output(corpus);
    sz_ptr_t o = output.data();

    // Shift forward by a single byte or a single cache line
    std::optional<row_t> const byte = bench_unary(env, corpus, "sz_move_best(by1)",
                                                  move_from_sz<cpu_best<sz_move_best>, 1> {corpus, o});
    print(byte);
    std::optional<row_t> const page = bench_unary(env, corpus, "sz_move_best(by64)",
                                                  move_from_sz<cpu_best<sz_move_best>, 64> {corpus, o});
    print(page, baseline_of(byte));

    print(bench_unary(env, corpus, "std::memmove(by1)", move_from_sz<memmove_like_sz, 1> {corpus, o}),
          baseline_of(byte));
    print(bench_unary(env, corpus, "std::memmove(by64)", move_from_sz<memmove_like_sz, 64> {corpus, o}),
          baseline_of(page));
}

#pragma endregion MemMove

#pragma region Broadcasting Constants with MemSet

sz_status_t memset_like_sz(sz_ptr_t output, sz_size_t length, sz_u8_t value, sz_stream_t) {
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
sz_status_t generate_like_sz(sz_ptr_t output, sz_size_t length, sz_u64_t nonce, sz_stream_t) {
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
void bench_fill(environment_t const &env, corpus_t const &corpus) {
    dataset_copy_t output(corpus);
    sz_ptr_t o = output.data();

    std::optional<row_t> const zeros = bench_unary(env, corpus, "sz_fill_best",
                                                   fill_from_sz<cpu_best<sz_fill_best>> {corpus, o});
    print(zeros);
    std::optional<row_t> const random = bench_unary(env, corpus, "sz_fill_random_best",
                                                    fill_random_from_sz<cpu_best<sz_fill_random_best>> {corpus, o});
    print(random, baseline_of(zeros));

    // The generators differ, so the random baseline is timed but never validated.
    print(bench_unary(env, corpus, "fill<std::memset>", fill_from_sz<memset_like_sz> {corpus, o}), baseline_of(zeros));
    print(bench_unary(env, corpus, "fill<std::random_device>", fill_random_from_sz<generate_like_sz> {corpus, o}),
          baseline_of(random));
}

#pragma endregion Broadcasting Constants with MemSet

#pragma region Lookup Transformations

sz_status_t transform_like_sz(sz_ptr_t output, sz_cptr_t input, sz_size_t length, sz_cptr_t lookup_table, sz_stream_t) {
    std::transform(input, input + length, output, [=](char c) { return (char)lookup_table[(unsigned char)c]; });
    return sz_success_k;
}

/**
 *  @brief Benchmarks look-up transformations on the provided slices, updating them inplace.
 *
 *  Performs a simple cyclical rotation of the alphabet, to test the performance of the different
 *  "look-up table"-based transformations.
 */
void bench_lookup(environment_t const &env, corpus_t const &corpus) {
    dataset_copy_t output(corpus);
    sz_ptr_t o = output.data();
    std::array<unsigned char, 256> const alphabet = rotated_alphabet();
    sz_cptr_t lut = reinterpret_cast<sz_cptr_t>(alphabet.data());

    std::optional<row_t> const zeros = bench_unary(env, corpus, "sz_lookup_best",
                                                   lookup_from_sz<cpu_best<sz_lookup_best>> {corpus, o, lut});
    print(zeros);
    print(bench_unary(env, corpus, "lookup<std::transform>", lookup_from_sz<transform_like_sz> {corpus, o, lut}),
          baseline_of(zeros));
}

#pragma endregion Lookup Transformations

void bench_memory(environment_t &env) {
    corpus_t const &corpus = env.corpora.lines();
    fmt::println("Starting low-level memory-operation benchmarks...");
    bench_copy(env, corpus);
    bench_move(env, corpus);
    bench_fill(env, corpus);
    bench_lookup(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
