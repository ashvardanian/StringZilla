/**
 *  @file   bench_memory.cpp
 *  @brief  Benchmarks for memory operations like copying, moving, resetting, and converting with lookup tables.
 *          The program accepts a file path to a dataset, tokenizes it, and uses those tokens only for size
 *          references to mimic real-world scenarios dealing with individual strings of different lengths.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWa.rs, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=words` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWa.rs, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_memory_cpp20
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines build_release/stringzilla_bench_memory_cpp20
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzilla_bench_memory_cpp20
 *  @endcode
 *
 *  Unlike the full-blown StringWa.rs, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `bench_find.cpp`, `bench_token.cpp`, and `bench_sequence.cpp`.
 */
#include <cstring> // `memmem`
#include <memory>  // `std::unique_ptr`
#include <numeric> // `std::iota`
#include <string>  // `std::string`

#ifdef _WIN32
#include <malloc.h> // `_aligned_malloc`
#else
#include <cstdlib> // `std::aligned_alloc`
#endif

#define SZ_USE_MISALIGNED_LOADS (1)
#include "bench.hpp"

using namespace ashvardanian::stringzilla::scripts;
constexpr std::size_t max_shift_length = 299;

/**
 *  @brief  Wraps platform-specific @b aligned memory allocation and deallocation functions.
 *          Compatible with `std::unique_ptr` as the second template argument, to free the memory.
 */
struct page_alloc_and_free_t {
#ifdef _WIN32
    inline char *operator()(std::size_t alignment, std::size_t size) const noexcept {
        return reinterpret_cast<char *>(_aligned_malloc(size, alignment));
    }
    inline void operator()(char *ptr) const noexcept { _aligned_free(ptr); }
#else
    inline char *operator()(std::size_t alignment, std::size_t size) const noexcept {
        return reinterpret_cast<char *>(std::aligned_alloc(alignment, size));
    }
    inline void operator()(char *ptr) const noexcept { std::free(ptr); }
#endif
};

#pragma region MemCpy

/** @brief Wraps a hardware-specific @b `memcpy`-like backend into something compatible with @b `bench_unary`. */
template <sz_copy_t copy_func_, int page_misalignment_ = 0>
struct copy_from_sz {

    environment_t const &env;
    sz_ptr_t output;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - env.dataset.data();
        // Round down to the nearest multiple of a cache line width for aligned writes
        output_offset = round_up_to_multiple<SZ_CACHE_LINE_WIDTH>(output_offset) - SZ_CACHE_LINE_WIDTH;
        // Ensure unaligned exports if needed
        output_offset += page_misalignment_;
        copy_func_(output + output_offset, slice.data(), slice.size());
        return {slice.size()};
    }
};

void memcpy_like_sz(sz_ptr_t output, sz_cptr_t input, std::size_t length) { std::memcpy(output, input, length); }

/**
 *  @brief Benchmarks `memcpy`-like operations in 2 modes: @b aligned output buffer and @b shifted misaligned.
 *
 *  In the aligned case we copy a random part of the input string into the start of a matching cache line in the output.
 *  In the unaligned case we also locate a matching cache line in the output, but shift by one to guarantee unaligned
 *  writes.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or output.
 *  So the kernels can be compared against the baseline `memcpy` function.
 */
void bench_copy(environment_t const &env) {

    // Create an aligned buffer for the output
    std::unique_ptr<char, page_alloc_and_free_t> output_buffer;
    // Add space for at least one cache line to simplify unaligned exports
    std::size_t const output_length = round_up_to_multiple<4096>(env.dataset.size() + max_shift_length);
    output_buffer.reset(page_alloc_and_free_t {}(4096, output_length));
    sz_ptr_t o = output_buffer.get();

    // Provide a baseline
    bench_result_t align = bench_unary(env, "sz_copy_serial(align)", copy_from_sz<sz_copy_serial>(env, o)).log();
    bench_result_t shift = bench_unary(env, "sz_copy_serial(shift)", copy_from_sz<sz_copy_serial, 1>(env, o)) //
                               .log(align);

#if SZ_USE_HASWELL
    bench_unary(env, "sz_copy_haswell(align)", copy_from_sz<sz_copy_haswell>(env, o)).log(align);
    bench_unary(env, "sz_copy_haswell(shift)", copy_from_sz<sz_copy_haswell, 1>(env, o)).log(align, shift);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_copy_skylake(align)", copy_from_sz<sz_copy_skylake>(env, o)).log(align);
    bench_unary(env, "sz_copy_skylake(shift)", copy_from_sz<sz_copy_skylake, 1>(env, o)).log(align, shift);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_copy_neon(align)", copy_from_sz<sz_copy_neon>(env, o)).log(align);
    bench_unary(env, "sz_copy_neon(shift)", copy_from_sz<sz_copy_neon, 1>(env, o)).log(align, shift);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_copy_sve(align)", copy_from_sz<sz_copy_sve>(env, o)).log(align);
    bench_unary(env, "sz_copy_sve(shift)", copy_from_sz<sz_copy_sve, 1>(env, o)).log(align, shift);
#endif

    bench_unary(env, "std::memcpy(align)", copy_from_sz<memcpy_like_sz>(env, o)).log(align);
    bench_unary(env, "std::memcpy(shift)", copy_from_sz<memcpy_like_sz, 1>(env, o)).log(align, shift);
}

#pragma endregion // MemCpy

#pragma region MemMove

/** @brief Wraps a hardware-specific @b `memmove`-like backend into something compatible with @b `bench_unary`. */
template <sz_move_t move_func_, int shift_ = 0>
struct move_from_sz {

    environment_t const &env;
    sz_ptr_t output;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - env.dataset.data();
        // Shift forward
        move_func_(output + output_offset + shift_, output + output_offset, slice.size());
        // Shift backward to revert the changes
        move_func_(output + output_offset, output + output_offset + shift_, slice.size());
        return {slice.size() * 2};
    }
};

void memmove_like_sz(sz_ptr_t output, sz_cptr_t input, std::size_t length) { std::memmove(output, input, length); }

/**
 *  @brief Benchmarks @b `memmove`-like operations shuffling back and forth the regions of output memory.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or output.
 *  This is achieved by performing a combination of a forward and a backward move.
 *  So the kernels can be compared against the baseline `memmove` function.
 */
void bench_move(environment_t const &env) {

    // Create an aligned buffer for the output
    std::unique_ptr<char, page_alloc_and_free_t> output_buffer;
    // Add space for at least one cache line to simplify unaligned exports
    std::size_t const output_length = round_up_to_multiple<4096>(env.dataset.size() + max_shift_length);
    output_buffer.reset(page_alloc_and_free_t {}(4096, output_length));
    sz_ptr_t o = output_buffer.get();

    // Copy the dataset to the output buffer
    std::memcpy(o, env.dataset.data(), env.dataset.size());

    // Provide a baseline for shifting forward by a single byte or a single cache line
    bench_result_t byte = bench_unary(env, "sz_move_serial(by1)", move_from_sz<sz_move_serial, 1>(env, o)).log();
    bench_result_t page = bench_unary(env, "sz_move_serial(by64)", move_from_sz<sz_move_serial, 64>(env, o)).log(byte);

#if SZ_USE_HASWELL
    bench_unary(env, "sz_move_haswell(by1)", move_from_sz<sz_move_haswell, 1>(env, o)).log(byte);
    bench_unary(env, "sz_move_haswell(by64)", move_from_sz<sz_move_haswell, 64>(env, o)).log(byte, page);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_move_skylake(by1)", move_from_sz<sz_move_skylake, 1>(env, o)).log(byte);
    bench_unary(env, "sz_move_skylake(by64)", move_from_sz<sz_move_skylake, 64>(env, o)).log(byte, page);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_move_neon(by1)", move_from_sz<sz_move_neon, 1>(env, o)).log(byte);
    bench_unary(env, "sz_move_neon(by64)", move_from_sz<sz_move_neon, 64>(env, o)).log(byte, page);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_move_sve(by1)", move_from_sz<sz_move_sve, 1>(env, o)).log(byte);
    bench_unary(env, "sz_move_sve(by64)", move_from_sz<sz_move_sve, 64>(env, o)).log(byte, page);
#endif

    bench_unary(env, "std::memmove(by1)", move_from_sz<memmove_like_sz, 1>(env, o)).log(byte);
    bench_unary(env, "std::memmove(by64)", move_from_sz<memmove_like_sz, 64>(env, o)).log(byte, page);
}

#pragma endregion // MemMove

#pragma region Broadcasting Constants with MemSet

/** @brief Wraps a hardware-specific @b `memset`-like backend into something compatible with @b `bench_unary`. */
template <sz_fill_t fill_func_>
struct fill_from_sz {

    environment_t const &env;
    sz_ptr_t output;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - env.dataset.data();
        fill_func_(output + output_offset, slice.size(), slice.front());
        return {slice.size(), static_cast<check_value_t>(slice.front())};
    }
};

/** @brief Wraps a hardware-specific @b `std::generate`-like backend into something compatible with @b `bench_unary`. */
template <sz_fill_random_t fill_func_>
struct fill_random_from_sz {

    environment_t const &env;
    sz_ptr_t output;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - env.dataset.data();
        fill_func_(output + output_offset, slice.size(), slice.front());
        char last_random_byte = output[output_offset + slice.size() - 1];
        do_not_optimize(last_random_byte);
        return {slice.size(), static_cast<check_value_t>(last_random_byte)};
    }
};

void memset_like_sz(sz_ptr_t output, sz_size_t length, sz_u8_t value) { std::memset(output, value, length); }

void generate_like_sz(sz_ptr_t output, sz_size_t length, sz_u64_t nonce) {
    uniform_u8_distribution_t distribution;
    std::generate(output, output + length, [&]() -> char { return distribution(global_random_generator()); });
    sz_unused_(nonce);
}

/**
 *  @brief  Benchmarks `memset`-like operations overwriting regions of output memory filling
 *          them with the first byte of the input regions or with random @b (reproducible) byte streams.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or output.
 *  So the kernels can be compared against the baseline `memset` function.
 */
void bench_fill(environment_t const &env) {

    // Create an aligned buffer for the output
    std::unique_ptr<char, page_alloc_and_free_t> output_buffer;
    // Add space for at least one cache line to simplify unaligned exports
    std::size_t const output_length = round_up_to_multiple<4096>(env.dataset.size() + max_shift_length);
    output_buffer.reset(page_alloc_and_free_t {}(4096, output_length));
    sz_ptr_t o = output_buffer.get();

    // Copy the dataset to the output buffer
    std::memcpy(o, env.dataset.data(), env.dataset.size());

    // Provide a baseline for overwriting the `output_buffer` memory
    bench_result_t zeros = bench_unary(env, "sz_fill_serial", fill_from_sz<sz_fill_serial>(env, o)).log();
    auto random_call = fill_random_from_sz<sz_fill_random_serial>(env, o);
    bench_result_t random = bench_unary(env, "sz_fill_random_serial", random_call).log(zeros);

#if SZ_USE_HASWELL
    bench_unary(env, "sz_fill_haswell", fill_from_sz<sz_fill_haswell>(env, o)).log(zeros);
    bench_unary(env, "sz_fill_random_haswell", random_call, fill_random_from_sz<sz_fill_random_haswell>(env, o))
        .log(zeros, random);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_fill_skylake", fill_from_sz<sz_fill_skylake>(env, o)).log(zeros);
    bench_unary(env, "sz_fill_random_skylake", random_call, fill_random_from_sz<sz_fill_random_skylake>(env, o))
        .log(zeros, random);
#endif
#if SZ_USE_ICE
    bench_unary(env, "sz_fill_random_ice", random_call, fill_random_from_sz<sz_fill_random_ice>(env, o))
        .log(zeros, random);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_fill_neon", fill_from_sz<sz_fill_neon>(env, o)).log(zeros);
    bench_unary(env, "sz_fill_random_neon", random_call, fill_random_from_sz<sz_fill_random_neon>(env, o))
        .log(zeros, random);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_fill_sve", fill_from_sz<sz_fill_sve>(env, o)).log(zeros);
#endif
    bench_unary(env, "fill<std::memset>", fill_from_sz<memset_like_sz>(env, o)).log(zeros);
    bench_unary(env, "fill<std::random_device>", fill_random_from_sz<generate_like_sz>(env, o)).log(zeros, random);
}

#pragma endregion // Broadcasting Constants with MemSet

#pragma region Lookup Transformations

/** @brief Wraps a hardware-specific @b `memset`-like backend into something compatible with @b `bench_unary`. */
template <sz_lookup_t lookup_func_>
struct lookup_from_sz {

    environment_t const &env;
    sz_ptr_t output;
    sz_cptr_t lookup_table;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - env.dataset.data();
        lookup_func_(output + output_offset, slice.size(), slice.data(), lookup_table);
        return {slice.size(), static_cast<check_value_t>(slice.front())};
    }
};

void transform_like_sz(sz_ptr_t output, sz_size_t length, sz_cptr_t input, sz_cptr_t lookup_table) {
    std::transform(input, input + length, output, [=](char c) { return (char)lookup_table[(unsigned char)c]; });
}

/**
 *  @brief  Benchmarks look-up transformations on the provided slices, updating them inplace.
 *
 *  Performs a simple cyclical rotation of the alphabet, to test the performance of the different
 *  "look-up table"-based transformations.
 */
void bench_lookup(environment_t const &env) {

    // Create an aligned buffer for the output
    std::unique_ptr<char, page_alloc_and_free_t> output_buffer;
    // Add space for at least one cache line to simplify unaligned exports
    std::size_t const output_length = round_up_to_multiple<4096>(env.dataset.size() + max_shift_length);
    output_buffer.reset(page_alloc_and_free_t {}(4096, output_length));
    sz_ptr_t o = output_buffer.get();

    // Copy the dataset to the output buffer
    std::memcpy(o, env.dataset.data(), env.dataset.size());

    // Prepare cyclic rotation of the alphabet
    static unsigned char lookup_table[256];
    std::iota(std::begin(lookup_table), std::end(lookup_table), 0);
    std::rotate(std::begin(lookup_table), std::begin(lookup_table) + 1, std::end(lookup_table));

    // Provide a baseline for overwriting the `output_buffer` memory
    sz_cptr_t lut = reinterpret_cast<sz_cptr_t>(lookup_table);
    bench_result_t zeros = bench_unary(env, "sz_lookup_serial", lookup_from_sz<sz_lookup_serial>(env, o, lut)).log();

#if SZ_USE_HASWELL
    bench_unary(env, "sz_lookup_haswell", lookup_from_sz<sz_lookup_haswell>(env, o, lut)).log(zeros);
#endif
#if SZ_USE_ICE
    bench_unary(env, "sz_lookup_ice", lookup_from_sz<sz_lookup_ice>(env, o, lut)).log(zeros);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_lookup_neon", lookup_from_sz<sz_lookup_neon>(env, o, lut)).log(zeros);
#endif
    bench_unary(env, "lookup<std::transform>", lookup_from_sz<transform_like_sz>(env, o, lut)).log(zeros);
}

#pragma endregion // Lookup Transformations

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZilla!\n");

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "leipzig1M.txt",                   //
        environment_t::tokenization_t::lines_k);

    std::printf("Starting low-level memory-operation benchmarks...\n");
    bench_copy(env);
    bench_move(env);
    bench_fill(env);
    bench_lookup(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}