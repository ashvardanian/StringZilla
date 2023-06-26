#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <strstream>

#include <stringzilla.h>

using strings_t = std::vector<std::string>;
using permute_t = std::vector<std::size_t>;

static char const *get_start(void const *array_c, size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(array_c);
    return array[i].c_str();
}

static size_t get_length(void const *array_c, size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(array_c);
    return array[i].size();
}

static bool has_under_four_chars(char const *, size_t len) { return len < 4; }

void populate_from_file( //
    char const *path,
    strings_t &strings,
    std::size_t limit = std::numeric_limits<std::size_t>::max()) {

    std::ifstream f(path, std::ios::in);
    std::string s;
    while (strings.size() < limit && std::getline(f, s, ' '))
        strings.push_back(s);
}

void populate_with_test(strings_t &strings) {
    strings.push_back("bbbb");
    strings.push_back("bbbbbb");
    strings.push_back("aac");
    strings.push_back("aa");
    strings.push_back("bb");
    strings.push_back("ab");
    strings.push_back("a");
    strings.push_back("");
    strings.push_back("cccc");
    strings.push_back("ccccccc");
}

template <bool use_stl_ak>
void benchmark(strings_t &strings, permute_t &permute, std::size_t iterations) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();

    // Run multiple iterations
    for (std::size_t i = 0; i != iterations; ++i) {
        std::iota(permute.begin(), permute.end(), 0);
        if constexpr (use_stl_ak)
            std::sort(permute.begin(), permute.end(), [&](std::size_t i, std::size_t j) {
                return strings[i] < strings[j];
            });
        else
            strzl_sort(&strings, strings.size(), &get_start, &get_length, permute.data(), false);
    }

    // Measure elapsed time
    stdcc::time_point t2 = stdcc::now();
    double dif = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count();
    std::printf("Elasped time is %lf nanoseconds/iteration.\n", dif / iterations);

    // Make sure it is properly sorted:
    if (!std::is_sorted(permute.begin(), permute.end(), [&](std::size_t i, std::size_t j) {
            return strings[i] < strings[j];
        }))
        std::printf("Failed to sort!\n");
}

int main(int, char const **) {
    std::printf("Hey, Ash!\n");

    strings_t strings;
    permute_t permute;
    populate_from_file("leipzig1M.txt", strings, 100000);
    std::printf("Parsed the file with %zu words!\n", strings.size());
    permute.resize(strings.size());

    // Identical partitioning ops:
    // std::partition(permute.begin(), permute.end(), [&](size_t i) { return strings[i].size() < 4; });
    // strzl_partition(&strings, strings.size(), &get_start, &get_length, &has_under_four_chars, permute.data());
    //
    // Identical sorting ops:
    // std::sort(permute.begin(), permute.end(), [&](size_t i, size_t j) { return strings[i] < strings[j]; });
    // strzl_sort(&strings, strings.size(), &get_start, &get_length, permute.data(), false);

    benchmark<true>(strings, permute, 100);
    benchmark<false>(strings, permute, 100);
    return 0;
}