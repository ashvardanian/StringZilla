/**
 *  @file   bench_container.cpp
 *  @brief  Benchmarks STL associative containers with string keys.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_search.cpp` and `bench_token.cpp`.
 *  It accepts a file with a list of words, constructs associative containers with string keys,
 *  using `std::string`, `std::string_view`, `sz::string_view`, and `sz::string`, and then
 *  evaluates the latency of lookups.
 */
#include <map>
#include <unordered_map>

#include <bench.hpp>

using namespace ashvardanian::stringzilla::scripts;

/**
 *  @brief  Evaluation for search string operations: find.
 */
template <typename container_at, typename strings_at>
void bench(strings_at &&strings) {

    // Build up the container
    container_at container;
    for (auto &&str : strings) { container[str] = 0; }

    tracked_function_gt<unary_function_t> variant;

    variant.results = bench_on_tokens(strings, [&](sz_string_view_t str_n) {
        sz_string_view_t str_h = {content_original.data(), content_original.size()};
        auto offset_from_start = variant.function(str_h, str_n);
        while (offset_from_start != str_h.length) {
            str_h.start += offset_from_start + 1, str_h.length -= offset_from_start + 1;
            offset_from_start = variant.function(str_h, str_n);
            do_not_optimize(offset_from_start);
        }
        return str_h.length;
    });

    variant.print();
}

template <typename strings_at>
void bench_tokens(strings_at &&strings) {
    if (strings.size() == 0) return;

    // Pure STL
    bench<std::map<std::string, int>>(strings);
    bench<std::map<std::string_view, int>>(strings);
    bench<std::unordered_map<std::string, int>>(strings);
    bench<std::unordered_map<std::string_view, int>>(strings);

    // StringZilla structures
    bench<std::map<sz::string, int>>(strings);
    bench<std::map<sz::string_view, int>>(strings);
    bench<std::unordered_map<sz::string, int>>(strings);
    bench<std::unordered_map<sz::string_view, int>>(strings);

    // STL structures with StringZilla operations
    bench<std::map<std::string, int, sz::less>>(strings);
    bench<std::map<std::string_view, int, sz::less>>(strings);
    bench<std::unordered_map<std::string, int, sz::hash, sz::equal_to>>(strings);
    bench<std::unordered_map<std::string_view, int, sz::hash, sz::equal_to>>(strings);
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting search benchmarks.\n");

    dataset_t dataset = make_dataset(argc, argv);

    // Baseline benchmarks for real words, coming in all lengths
    std::printf("Benchmarking on real words:\n");
    bench_tokens(dataset.tokens);

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on real words of length %zu:\n", token_length);
        bench_tokens(filter_by_length(dataset.tokens, token_length));
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}