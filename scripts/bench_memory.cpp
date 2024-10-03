/**
 *  @file   bench_memory.cpp
 *  @brief  Benchmarks for memory operations like copying, moving, and comparing.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_token.cpp` and `bench_similarity.cpp`.
 *  It accepts a file with a list of words, and benchmarks the memory operations on them.
 */
#include <cstdlib> // `std::aligned_alloc`
#include <cstring> // `memmem`
#include <memory>  // `std::unique_ptr`
#include <string>  // `std::string`

#define SZ_USE_MISALIGNED_LOADS (1)
#include <bench.hpp>

using namespace ashvardanian::stringzilla::scripts;

template <bool aligned_output>
tracked_unary_functions_t copy_functions(sz_cptr_t input_start_ptr, sz_ptr_t output_buffer_ptr) {
    std::string suffix = aligned_output ? "<aligned>" : "<unaligned>";
    auto wrap_sz = [input_start_ptr, output_buffer_ptr](auto function) -> unary_function_t {
        return unary_function_t([function, input_start_ptr, output_buffer_ptr](std::string_view slice) {
            std::size_t output_offset = slice.data() - input_start_ptr;
            // Round down to the nearest multiple of 64 for aligned writes
            output_offset = divide_round_up<64>(output_offset) - 64;
            // Ensure unaligned exports if needed
            if constexpr (!aligned_output) output_offset += 1;
            function(output_buffer_ptr + output_offset, slice.data(), slice.size());
            return slice.size();
        });
    };
    tracked_unary_functions_t result = {
        {"memcpy" + suffix, wrap_sz(memcpy)},
        {"sz_copy_serial" + suffix, wrap_sz(sz_copy_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_copy_avx512" + suffix, wrap_sz(sz_copy_avx512), true},
#endif
#if SZ_USE_X86_AVX2
        {"sz_copy_avx2" + suffix, wrap_sz(sz_copy_avx2), true},
#endif
#if SZ_USE_ARM_SVE
        {"sz_copy_sve" + suffix, wrap_sz(sz_copy_sve), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_copy_neon" + suffix, wrap_sz(sz_copy_neon), true},
#endif
    };
    return result;
}

tracked_unary_functions_t fill_functions(sz_cptr_t input_start_ptr, sz_ptr_t output_buffer_ptr) {
    auto wrap_sz = [input_start_ptr, output_buffer_ptr](auto function) -> unary_function_t {
        return unary_function_t([function, input_start_ptr, output_buffer_ptr](std::string_view slice) {
            std::size_t output_offset = (std::size_t)(slice.data() - input_start_ptr);
            function(output_buffer_ptr + output_offset, slice.size(), slice.front());
            return slice.size();
        });
    };
    tracked_unary_functions_t result = {
        {"memset", unary_function_t([input_start_ptr, output_buffer_ptr](std::string_view slice) {
             std::size_t output_offset = (std::size_t)(slice.data() - input_start_ptr);
             memset(output_buffer_ptr + output_offset, slice.front(), slice.size());
             return slice.size();
         })},
        {"sz_fill_serial", wrap_sz(sz_fill_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_fill_avx512", wrap_sz(sz_fill_avx512), true},
#endif
#if SZ_USE_X86_AVX2
        {"sz_fill_avx2", wrap_sz(sz_fill_avx2), true},
#endif
#if SZ_USE_ARM_SVE
        {"sz_fill_sve", wrap_sz(sz_fill_sve), true},
#endif
#if SZ_USE_ARM_NEON
        {"sz_fill_neon", wrap_sz(sz_fill_neon), true},
#endif
    };
    return result;
}
/**
 *  @brief  Evaluation for search string operations: find.
 */
void bench_memory(std::vector<std::string_view> const &slices, tracked_unary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_tokens(slices, [&](std::string_view slice) {
                // TODO: Obfurscate the output with random contents, and later check if the contents
                // of the output are identical for the baseline and the variant.
                return slice.size();
            });
        }

        // Benchmarks
        if (variant.function) variant.results = bench_on_tokens(slices, variant.function);

        variant.print();
    }
}

void bench_memory(std::vector<std::string_view> const &slices, sz_cptr_t input_start_ptr, sz_ptr_t output_buffer_ptr) {
    if (slices.size() == 0) return;

    bench_memory(slices, copy_functions<true>(input_start_ptr, output_buffer_ptr));
    bench_memory(slices, copy_functions<false>(input_start_ptr, output_buffer_ptr));
    bench_memory(slices, fill_functions(input_start_ptr, output_buffer_ptr));
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting memory benchmarks.\n");

    dataset_t dataset = prepare_benchmark_environment(argc, argv);
    sz_cptr_t const input_start_ptr = dataset.text.data();

    // These benchmarks should be heavier than substring search and other less critical operations.
    seconds_per_benchmark *= 5;

    // Create an aligned buffer for the output
    struct aligned_free_t {
        inline void operator()(char *ptr) const noexcept { std::free(ptr); }
    };
    std::unique_ptr<char, aligned_free_t> output_buffer;
    // Add space for at least one cache line to simplify unaligned exports
    std::size_t const output_length = divide_round_up<4096>(dataset.text.size() + 64);
    output_buffer.reset(reinterpret_cast<char *>(std::aligned_alloc(4096, output_length)));

    // Baseline benchmarks for present tokens, coming in all lengths
    std::printf("Benchmarking on lines:\n");
    bench_memory({dataset.lines.begin(), dataset.lines.end()}, input_start_ptr, output_buffer.get());
    std::printf("Benchmarking on tokens:\n");
    bench_memory({dataset.tokens.begin(), dataset.tokens.end()}, input_start_ptr, output_buffer.get());
    std::printf("Benchmarking on entire dataset:\n");
    bench_memory({dataset.text}, input_start_ptr, output_buffer.get());

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on tokens of length %zu:\n", token_length);
        bench_memory(filter_by_length<std::string_view>(dataset.tokens, token_length), input_start_ptr,
                     output_buffer.get());
    }
    std::printf("All benchmarks passed.\n");
    return 0;
}