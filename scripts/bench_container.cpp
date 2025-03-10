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

template <typename string_to_type_, typename string_from_type_>
std::vector<string_to_type_> to(std::vector<string_from_type_> const &strings) {
    std::vector<string_to_type_> result;
    result.reserve(strings.size());
    for (string_from_type_ const &string : strings) result.push_back({string.data(), string.size()});
    return result;
}

/**
 *  @brief  Evaluation for search string operations: find.
 */
template <typename container_type_>
void bench(std::string name, std::vector<std::string_view> const &strings) {

    using key_type = typename container_type_::key_type;
    std::vector<key_type> keys = to<key_type>(strings);

    // Build up the container
    container_type_ container;
    for (key_type const &key : keys) container[key] = 0;

    tracked_function_gt<unary_function_t> variant;
    variant.name = name;
    variant.results = bench_on_tokens(keys, [&](key_type const &key) {
        container.find(key)->second++;
        return key.size();
    });

    variant.print();
}

template <typename strings_type_>
void bench_tokens(strings_type_ const &s) {
    if (s.size() == 0) return;

    // STL containers with StringZilla strings and views
    bench<std::map<sz::string, int>>("std::map<sz::string>::find", s);
    bench<std::map<sz::string_view, int>>("std::map<sz::string_view>::find", s);
    bench<std::unordered_map<sz::string, int>>("std::umap<sz::string>::find", s);
    bench<std::unordered_map<sz::string_view, int>>("std::umap<sz::string_view>::find", s);

    // STL containers with STL strings and views
    bench<std::map<std::string, int>>("std::map<std::string>::find", s);
    bench<std::map<std::string_view, int>>("std::map<std::string_view>::find", s);
    bench<std::unordered_map<std::string, int>>("std::umap<std::string>::find", s);
    bench<std::unordered_map<std::string_view, int>>("std::umap<std::string_view>::find", s);

    // STL structures with StringZilla operations
    bench<std::map<std::string, int, sz::less>>("std::map<std::string, sz::less>::find", s);
    bench<std::map<std::string_view, int, sz::less>>("std::map<std::string_view, sz::less>::find", s);
    bench<std::unordered_map<std::string, int, sz::hash, sz::equal_to>>("std::umap<std::string, sz::hash>::find", s);
    bench<std::unordered_map<std::string_view, int, sz::hash, sz::equal_to>>(
        "std::umap<std::string_view, sz::hash>::find", s);
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting container benchmarks.\n");

    dataset_t dataset = prepare_benchmark_environment(argc, argv);

    // Baseline benchmarks for real words, coming in all lengths
    std::printf("Benchmarking on real words:\n");
    bench_tokens(dataset.tokens);
    std::printf("Benchmarking on real lines:\n");
    bench_tokens(dataset.lines);

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
        std::printf("Benchmarking on real words of length %zu:\n", token_length);
        bench_tokens(filter_by_length(dataset.tokens, token_length));
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}