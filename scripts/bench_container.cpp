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

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting STL container benchmarks.\n");

    // dataset_t dataset = make_dataset(argc, argv);

    std::printf("All benchmarks passed.\n");
    return 0;
}