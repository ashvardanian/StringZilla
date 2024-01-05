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
template <typename container_at>
void bench(std::vector<typename container_at::key_type> const &strings) {

    using key_type = typename container_at::key_type;

    // Build up the container
    container_at container;
    for (key_type const &key : strings) container[key] = 0;

    tracked_function_gt<unary_function_t> variant;
    variant.results = bench_on_tokens(strings, [&](key_type const &key) {
        container[key]++;
        return 1;
    });

    variant.print();
}

template <typename string_type_to, typename string_type_from>
std::vector<string_type_to> to(std::vector<string_type_from> const &strings) {
    std::vector<string_type_to> result;
    result.reserve(strings.size());
    for (string_type_from const &string : strings) result.push_back({string.data(), string.size()});
    return result;
}

template <typename strings_type>
void bench_tokens(strings_type const &strings) {
    if (strings.size() == 0) return;

    // Pure STL
    bench<std::map<std::string, int>>(to<std::string>(strings));
    bench<std::map<std::string_view, int>>(to<std::string_view>(strings));
    bench<std::unordered_map<std::string, int>>(to<std::string>(strings));
    bench<std::unordered_map<std::string_view, int>>(to<std::string_view>(strings));

    // StringZilla structures
    bench<std::map<sz::string, int>>(to<sz::string>(strings));
    bench<std::map<sz::string_view, int>>(to<sz::string_view>(strings));
    bench<std::unordered_map<sz::string, int>>(to<sz::string>(strings));
    bench<std::unordered_map<sz::string_view, int>>(to<sz::string_view>(strings));

    // STL structures with StringZilla operations
    // bench<std::map<std::string, int, sz::less>>(to<std::string>(strings));
    // bench<std::map<std::string_view, int, sz::less>>(to<std::string_view>(strings));
    // bench<std::unordered_map<std::string, int, sz::hash, sz::equal_to>>(to<std::string>(strings));
    // bench<std::unordered_map<std::string_view, int, sz::hash, sz::equal_to>>(to<std::string_view>(strings));
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