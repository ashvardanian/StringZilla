/**
 *  @file   bench_similarity.cpp
 *  @brief  Benchmarks string similarity computations.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_search.cpp` and `bench_token.cpp`.
 *  It accepts a file with a list of words, and benchmarks the levenshtein edit-distance computations,
 *  alignment scores, and fingerprinting techniques combined with the Hamming distance.
 */
#include <bench.hpp>

using namespace ashvardanian::stringzilla::scripts;

using temporary_memory_t = std::vector<char>;
temporary_memory_t temporary_memory;

static sz_ptr_t allocate_from_vector(sz_size_t length, void *handle) {
    temporary_memory_t &vec = *reinterpret_cast<temporary_memory_t *>(handle);
    if (vec.size() < length) vec.resize(length);
    return vec.data();
}

static void free_from_vector(sz_ptr_t buffer, sz_size_t length, void *handle) {}

tracked_binary_functions_t distance_functions() {
    // Populate the unary substitutions matrix
    static constexpr std::size_t max_length = 256;
    static std::vector<sz_error_cost_t> unary_substitution_costs;
    unary_substitution_costs.resize(max_length * max_length);
    for (std::size_t i = 0; i != max_length; ++i)
        for (std::size_t j = 0; j != max_length; ++j) unary_substitution_costs[i * max_length + j] = (i == j ? 0 : 1);

    // Two rows of the Levenshtein matrix will occupy this much:
    temporary_memory.resize((max_length + 1) * 2 * sizeof(sz_size_t));
    sz_memory_allocator_t alloc;
    alloc.allocate = &allocate_from_vector;
    alloc.free = &free_from_vector;
    alloc.handle = &temporary_memory;

    auto wrap_sz_distance = [alloc](auto function) -> binary_function_t {
        return binary_function_t([function, alloc](std::string_view a_str, std::string_view b_str) {
            sz_string_view_t a = to_c(a_str);
            sz_string_view_t b = to_c(b_str);
            a.length = sz_min_of_two(a.length, max_length);
            b.length = sz_min_of_two(b.length, max_length);
            return function(a.start, a.length, b.start, b.length, max_length, &alloc);
        });
    };
    auto wrap_sz_scoring = [alloc](auto function) -> binary_function_t {
        return binary_function_t([function, alloc](std::string_view a_str, std::string_view b_str) {
            sz_string_view_t a = to_c(a_str);
            sz_string_view_t b = to_c(b_str);
            a.length = sz_min_of_two(a.length, max_length);
            b.length = sz_min_of_two(b.length, max_length);
            return function(a.start, a.length, b.start, b.length, 1, unary_substitution_costs.data(),
                                        &alloc);
        });
    };
    tracked_binary_functions_t result = {
        {"sz_edit_distance", wrap_sz_distance(sz_edit_distance_serial)},
        {"sz_alignment_score", wrap_sz_scoring(sz_alignment_score_serial), true},
#if SZ_USE_X86_AVX512
        {"sz_edit_distance_avx512", wrap_sz_distance(sz_edit_distance_avx512), true},
#endif
    };
    return result;
}

template <typename strings_at>
void bench_similarity(strings_at &&strings) {
    if (strings.size() == 0) return;
    bench_binary_functions(strings, distance_functions());
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting similarity benchmarks.\n");

    dataset_t dataset = make_dataset(argc, argv);

    // Baseline benchmarks for real words, coming in all lengths
    std::printf("Benchmarking on real words:\n");
    bench_similarity(dataset.tokens);

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on real words of length %zu:\n", token_length);
        bench_similarity(filter_by_length(dataset.tokens, token_length));
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}