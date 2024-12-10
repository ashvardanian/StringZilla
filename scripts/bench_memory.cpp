/**
 *  @file   bench_memory.cpp
 *  @brief  Benchmarks for memory operations like copying, moving, and comparing.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_token.cpp` and `bench_similarity.cpp`.
 *  It accepts a file with a list of words, and benchmarks the memory operations on them.
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
#include <bench.hpp>

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

/**
 *  @brief  Benchmarks `memcpy`-like operations in 2 modes: aligned @b output buffer and unaligned.
 *
 *  In the aligned case we copy a random part of the input string into the start of a matching cache line in the output.
 *  In the unaligned case we also locate a matching cache line in the output, but shift by one to guarantee unaligned
 *  writes.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or output.
 *  So the kernels can be compared against the baseline `memcpy` function.
 *
 *  @param  output_buffer_ptr Aligned output buffer.
 */
template <bool aligned_output>
tracked_unary_functions_t copy_functions(sz_cptr_t dataset_start_ptr, sz_ptr_t output_buffer_ptr) {
    std::string suffix = aligned_output ? "<aligned>" : "<unaligned>";
    auto wrap_sz = [dataset_start_ptr, output_buffer_ptr](auto function) -> unary_function_t {
        return unary_function_t([function, dataset_start_ptr, output_buffer_ptr](std::string_view slice) {
            std::size_t output_offset = slice.data() - dataset_start_ptr;
            // Round down to the nearest multiple of a cache line width for aligned writes
            output_offset = round_up_to_multiple<SZ_CACHE_LINE_WIDTH>(output_offset) - SZ_CACHE_LINE_WIDTH;
            // Ensure unaligned exports if needed
            if constexpr (!aligned_output) output_offset += 1;
            function(output_buffer_ptr + output_offset, slice.data(), slice.size());
            return slice.size();
        });
    };
    tracked_unary_functions_t result = {
        {"memcpy" + suffix, wrap_sz(memcpy)},
        {"sz_copy_serial" + suffix, wrap_sz(sz_copy_serial)},
#if SZ_USE_SKYLAKE
        {"sz_copy_skylake" + suffix, wrap_sz(sz_copy_skylake)},
#endif
#if SZ_USE_HASWELL
        {"sz_copy_haswell" + suffix, wrap_sz(sz_copy_haswell)},
#endif
#if SZ_USE_SVE
        {"sz_copy_sve" + suffix, wrap_sz(sz_copy_sve)},
#endif
#if SZ_USE_NEON
        {"sz_copy_neon" + suffix, wrap_sz(sz_copy_neon)},
#endif
    };
    return result;
}

/**
 *  @brief  Benchmarks `memset`-like operations overwriting regions of output memory filling
 *          them with the first byte of the input regions.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or output.
 *  So the kernels can be compared against the baseline `memset` function.
 *
 *  @param  output_buffer_ptr Aligned output buffer.
 */
tracked_unary_functions_t fill_functions(sz_cptr_t dataset_start_ptr, sz_ptr_t output_buffer_ptr) {
    auto wrap_sz = [dataset_start_ptr, output_buffer_ptr](auto function) -> unary_function_t {
        return unary_function_t([function, dataset_start_ptr, output_buffer_ptr](std::string_view slice) {
            std::size_t output_offset = (std::size_t)(slice.data() - dataset_start_ptr);
            function(output_buffer_ptr + output_offset, slice.size(), slice.front());
            return slice.size();
        });
    };
    tracked_unary_functions_t result = {
        {"memset", unary_function_t([dataset_start_ptr, output_buffer_ptr](std::string_view slice) {
             std::size_t output_offset = (std::size_t)(slice.data() - dataset_start_ptr);
             memset(output_buffer_ptr + output_offset, slice.front(), slice.size());
             return slice.size();
         })},
        {"sz_fill_serial", wrap_sz(sz_fill_serial)},
#if SZ_USE_SKYLAKE
        {"sz_fill_avx512", wrap_sz(sz_fill_skylake)},
#endif
#if SZ_USE_HASWELL
        {"sz_fill_haswell", wrap_sz(sz_fill_haswell)},
#endif
#if SZ_USE_SVE
        {"sz_fill_sve", wrap_sz(sz_fill_sve)},
#endif
#if SZ_USE_NEON
        {"sz_fill_neon", wrap_sz(sz_fill_neon)},
#endif
    };
    return result;
}

/**
 *  @brief  Benchmarks `memmove`-like operations shuffling back and forth the regions of output memory.
 *
 *  Multiple calls to the provided functions even with the same arguments won't change the input or output.
 *  This is achieved by performing a combination of a forward and a backward move.
 *  So the kernels can be compared against the baseline `memmove` function.
 *
 *  @param  output_buffer_ptr Aligned output buffer, that ahs at least `shift` bytes of space at the end.
 */
tracked_unary_functions_t move_functions(sz_cptr_t dataset_start_ptr, sz_ptr_t output_buffer_ptr, std::size_t shift) {
    std::string suffix = "<shift" + std::to_string(shift) + ">";
    auto wrap_sz = [dataset_start_ptr, output_buffer_ptr, shift](auto function) -> unary_function_t {
        return unary_function_t([function, dataset_start_ptr, output_buffer_ptr, shift](std::string_view slice) {
            std::size_t output_offset = slice.data() - dataset_start_ptr;
            // Shift forward
            function(output_buffer_ptr + output_offset + shift, output_buffer_ptr + output_offset, slice.size());
            // Shift backward to revert the changes
            function(output_buffer_ptr + output_offset, output_buffer_ptr + output_offset + shift, slice.size());
            return slice.size() * 2;
        });
    };
    tracked_unary_functions_t result = {
        {"memmove" + suffix, wrap_sz(memmove)},
        {"sz_move_serial" + suffix, wrap_sz(sz_move_serial)},
#if SZ_USE_SKYLAKE
        {"sz_move_skylake" + suffix, wrap_sz(sz_move_skylake)},
#endif
#if SZ_USE_HASWELL
        {"sz_move_haswell" + suffix, wrap_sz(sz_move_haswell)},
#endif
#if SZ_USE_NEON
        {"sz_move_neon" + suffix, wrap_sz(sz_move_neon)},
#endif
    };
    return result;
}

/**
 *  @brief  Benchmarks look-up transformations on the provided slices, updating them inplace.
 *
 *  Performs a simple cyclical rotation of the alphabet, to test the performance of the different
 * "look-up table"-based transformations.
 */
tracked_unary_functions_t transform_functions() {
    static unsigned char look_up_table[256];
    std::iota(std::begin(look_up_table), std::end(look_up_table), 0);
    std::rotate(std::begin(look_up_table), std::begin(look_up_table) + 1, std::end(look_up_table));

    auto wrap_sz = [](auto function) -> unary_function_t {
        return unary_function_t([function](std::string_view slice) {
            char *output = const_cast<char *>(slice.data());
            function((sz_cptr_t)output, (sz_size_t)slice.size(), (sz_cptr_t)look_up_table, (sz_ptr_t)output);
            return slice.size();
        });
    };
    tracked_unary_functions_t result = {
        {"str::transform<lookup>", unary_function_t([](std::string_view slice) {
             char *output = const_cast<char *>(slice.data());
             std::transform(slice.begin(), slice.end(), output, [](char c) { return look_up_table[(unsigned char)c]; });
             return slice.size();
         })},
        {"str::transform<increment>", unary_function_t([](std::string_view slice) {
             char *output = const_cast<char *>(slice.data());
             std::transform(slice.begin(), slice.end(), output, [](char c) { return c + 1; });
             return slice.size();
         })},
        {"sz_look_up_transform_serial", wrap_sz(sz_look_up_transform_serial)},
#if SZ_USE_ICE
        {"sz_look_up_transform_ice", wrap_sz(sz_look_up_transform_ice)},
#endif
#if SZ_USE_HASWELL
        {"sz_look_up_transform_haswell", wrap_sz(sz_look_up_transform_haswell)},
#endif
#if SZ_USE_NEON
        {"sz_look_up_transform_neon", wrap_sz(sz_look_up_transform_neon)},
#endif
    };
    return result;
}

void bench_memory(std::vector<std::string_view> const &slices, tracked_unary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            std::fprintf(stderr, "Testing is not currently implemented.\n");
            exit(1);
        }

        // Benchmarks
        if (variant.function) variant.results = bench_on_tokens(slices, variant.function);
        variant.print();
    }
}

void bench_memory(std::vector<std::string_view> const &slices, sz_cptr_t dataset_start_ptr,
                  sz_ptr_t output_buffer_ptr) {

    if (slices.size() == 0) return;
    (void)dataset_start_ptr;
    (void)output_buffer_ptr;

    bench_memory(slices, copy_functions<true>(dataset_start_ptr, output_buffer_ptr));
    bench_memory(slices, copy_functions<false>(dataset_start_ptr, output_buffer_ptr));
    bench_memory(slices, fill_functions(dataset_start_ptr, output_buffer_ptr));
    bench_memory(slices, move_functions(dataset_start_ptr, output_buffer_ptr, 1));
    bench_memory(slices, move_functions(dataset_start_ptr, output_buffer_ptr, 8));
    bench_memory(slices, move_functions(dataset_start_ptr, output_buffer_ptr, SZ_CACHE_LINE_WIDTH));
    bench_memory(slices, move_functions(dataset_start_ptr, output_buffer_ptr, max_shift_length));
    bench_memory(slices, transform_functions());
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting memory benchmarks.\n");

    dataset_t dataset = prepare_benchmark_environment(argc, argv);
    sz_cptr_t const dataset_start_ptr = dataset.text.data();

    // These benchmarks should be heavier than substring search and other less critical operations.
    if (!SZ_DEBUG) seconds_per_benchmark *= 5;

    // Create an aligned buffer for the output
    std::unique_ptr<char, page_alloc_and_free_t> output_buffer;
    // Add space for at least one cache line to simplify unaligned exports
    std::size_t const output_length = round_up_to_multiple<4096>(dataset.text.size() + max_shift_length);
    output_buffer.reset(page_alloc_and_free_t {}(4096, output_length));
    if (!output_buffer) {
        std::fprintf(stderr, "Failed to allocate an output buffer of %zu bytes.\n", output_length);
        return 1;
    }
    std::memcpy(output_buffer.get(), dataset.text.data(), dataset.text.size());

    // Baseline benchmarks for present tokens, coming in all lengths
    std::printf("Benchmarking on entire dataset:\n");
    bench_memory({dataset.text}, dataset_start_ptr, output_buffer.get());
    std::printf("Benchmarking on lines:\n");
    bench_memory(dataset.lines, dataset_start_ptr, output_buffer.get());
    std::printf("Benchmarking on tokens:\n");
    bench_memory(dataset.tokens, dataset_start_ptr, output_buffer.get());

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on tokens of length %zu:\n", token_length);
        bench_memory(filter_by_length<std::string_view>(dataset.tokens, token_length), dataset_start_ptr,
                     output_buffer.get());
    }
    std::printf("All benchmarks passed.\n");
    return 0;
}