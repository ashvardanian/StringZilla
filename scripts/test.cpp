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
using idx_t = std::size_t;
using permute_t = std::vector<idx_t>;

#pragma region - C callbacks

static char const *get_begin(void const *array_c, size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(array_c);
    return array[i].c_str();
}

static size_t get_length(void const *array_c, size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(array_c);
    return array[i].size();
}

static bool has_under_four_chars(char const *, size_t len) { return len < 4; }

#pragma endregion

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

constexpr size_t offset_in_word = 0;

inline static idx_t hybrid_sort(strings_t const &strings, idx_t *order) {

    // What if we take up-to 4 first characters and the index
    for (size_t i = 0; i != strings.size(); ++i)
        std::memcpy((char *)&order[i] + offset_in_word,
                    strings[order[i]].c_str(),
                    std::min(strings[order[i]].size(), 4ul));

    std::sort(order, order + strings.size(), [&](idx_t i, idx_t j) {
        char *i_bytes = (char *)&i;
        char *j_bytes = (char *)&j;
        return *(uint32_t *)(i_bytes + offset_in_word) < *(uint32_t *)(j_bytes + offset_in_word);
    });

    for (size_t i = 0; i != strings.size(); ++i)
        std::memset((char *)&order[i] + offset_in_word, 0, 4ul);

    std::sort(order, order + strings.size(), [&](idx_t i, idx_t j) { return strings[i] < strings[j]; });

    return strings.size();
}

inline static idx_t hybrid_stable_sort(strings_t const &strings, idx_t *order) {

    // What if we take up-to 4 first characters and the index
    for (size_t i = 0; i != strings.size(); ++i)
        std::memcpy((char *)&order[i] + offset_in_word,
                    strings[order[i]].c_str(),
                    std::min(strings[order[i]].size(), 4ul));

    std::stable_sort(order, order + strings.size(), [&](idx_t i, idx_t j) {
        char *i_bytes = (char *)&i;
        char *j_bytes = (char *)&j;
        return *(uint32_t *)(i_bytes + offset_in_word) < *(uint32_t *)(j_bytes + offset_in_word);
    });

    for (size_t i = 0; i != strings.size(); ++i)
        std::memset((char *)&order[i] + offset_in_word, 0, 4ul);

    std::stable_sort(order, order + strings.size(), [&](idx_t i, idx_t j) { return strings[i] < strings[j]; });

    return strings.size();
}

void expect_partitioned_by_length(strings_t const &strings, permute_t const &permute) {
    if (!std::is_partitioned(permute.begin(), permute.end(), [&](size_t i) { return strings[i].size() < 4; }))
        throw std::runtime_error("Partitioning failed!");
}

void expect_sorted(strings_t const &strings, permute_t const &permute) {
    if (!std::is_sorted(permute.begin(), permute.end(), [&](std::size_t i, std::size_t j) {
            return strings[i] < strings[j];
        }))
        throw std::runtime_error("Sorting failed!");
}

void expect_same(permute_t const &permute_base, permute_t const &permute_new) {
    if (!std::equal(permute_base.begin(), permute_base.end(), permute_new.begin()))
        throw std::runtime_error("Permutations differ!");
}

template <typename algo_at>
void bench_permute(char const *name, strings_t &strings, permute_t &permute, algo_at &&algo) {
    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    constexpr std::size_t iterations = 3;
    stdcc::time_point t1 = stdcc::now();

    // Run multiple iterations
    for (std::size_t i = 0; i != iterations; ++i) {
        std::iota(permute.begin(), permute.end(), 0);
        algo(strings, permute);
    }

    // Measure elapsed time
    stdcc::time_point t2 = stdcc::now();
    double dif = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count();
    double milisecs = dif / (iterations * 1e6);
    std::printf("Elapsed time is %.2lf miliseconds/iteration for %s.\n", milisecs, name);
}

template <typename algo_at>
void bench_search(char const *name, std::string_view full_text, algo_at &&algo) {
    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    constexpr std::size_t iterations = 3000;
    stdcc::time_point t1 = stdcc::now();

    // Run multiple iterations
    for (std::size_t i = 0; i != iterations; ++i) {
        std::size_t length = std::rand() % 6 + 2;
        std::size_t offset = std::rand() % (full_text.size() - length);
        algo(full_text, full_text.substr(offset, length));
    }

    // Measure elapsed time
    stdcc::time_point t2 = stdcc::now();
    double dif = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count();
    double milisecs = dif / (iterations * 1e6);
    std::printf("Elapsed time is %.2lf miliseconds/iteration for %s.\n", milisecs, name);
}

int main(int, char const **) {
    std::printf("Hey, Ash!\n");

    strings_t strings;
    populate_from_file("leipzig1M.txt", strings, 1000000);
    std::size_t mean_bytes = 0;
    for (std::string const &str : strings)
        mean_bytes += str.size();
    mean_bytes /= strings.size();
    std::printf("Parsed the file with %zu words of %zu mean length!\n", strings.size(), mean_bytes);

    std::string full_text;
    full_text.reserve(mean_bytes + strings.size() * 2);
    for (std::string const &str : strings)
        full_text.append(str), full_text.push_back(' ');

    // Search substring
    bench_search("strzl_naive_find_substr", full_text, [](std::string_view haystack, std::string_view needle) {
        strzl_haystack_t h {};
        strzl_needle_t n {};
        h.ptr = haystack.data();
        h.len = haystack.size();
        n.ptr = needle.data();
        n.len = needle.size();
        strzl_naive_find_substr(h, n);
    });
    bench_search("std::search", full_text, [](std::string_view haystack, std::string_view needle) {
        return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    });

    // Partitioning
    permute_t permute_base, permute_new;
    permute_base.resize(strings.size());
    permute_new.resize(strings.size());
    bench_permute("std::partition", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
        std::partition(permute.begin(), permute.end(), [&](size_t i) { return strings[i].size() < 4; });
    });
    expect_partitioned_by_length(strings, permute_base);

    bench_permute("std::stable_partition", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
        std::stable_partition(permute.begin(), permute.end(), [&](size_t i) { return strings[i].size() < 4; });
    });
    expect_partitioned_by_length(strings, permute_base);

    bench_permute("strzl_partition", strings, permute_new, [](strings_t const &strings, permute_t &permute) {
        strzl_partition(&strings, strings.size(), &get_begin, &get_length, &has_under_four_chars, permute.data());
    });
    expect_partitioned_by_length(strings, permute_new);
    // TODO: expect_same(permute_base, permute_new);

    // Sorting
    // bench_permute("strzl_qsort", strings, permute_new, [](strings_t const &strings, permute_t &permute) {
    //     strzl_qsort(&strings,
    //                 strings.size(),
    //                 get_begin,
    //                 get_length,
    //                 strzl_less_entire,
    //                 permute.data(),
    //                 0,
    //                 strings.size());
    // });
    // expect_sorted(strings, permute_new);

    bench_permute("strzl_sort", strings, permute_new, [](strings_t const &strings, permute_t &permute) {
        strzl_sort(&strings, strings.size(), get_begin, get_length, permute.data());
    });
    expect_sorted(strings, permute_new);

    bench_permute("std::sort", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
        std::sort(permute.begin(), permute.end(), [&](idx_t i, idx_t j) { return strings[i] < strings[j]; });
    });
    expect_sorted(strings, permute_base);

    bench_permute("hybrid_sort", strings, permute_new, [](strings_t const &strings, permute_t &permute) {
        hybrid_sort(strings, permute.data());
    });
    expect_sorted(strings, permute_new);

    // Stable-sorting
    bench_permute("std::stable_sort", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
        std::stable_sort(permute.begin(), permute.end(), [&](idx_t i, idx_t j) { return strings[i] < strings[j]; });
    });
    expect_sorted(strings, permute_base);

    bench_permute("hybrid_stable_sort", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
        hybrid_stable_sort(strings, permute.data());
    });
    expect_sorted(strings, permute_new);
    expect_same(permute_base, permute_new);

    return 0;
}