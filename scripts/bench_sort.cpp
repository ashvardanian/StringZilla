
/**
 *  @file   bench_sort.cpp
 *  @brief  Benchmarks sorting, partitioning, and merging operations on string sequences.
 *
 *  This file is the sibling of `bench_similarity.cpp`, `bench_search.cpp` and `bench_token.cpp`.
 *  It accepts a file with a list of words, and benchmarks the sorting operations on them.
 */
#include <memory>  // `std::memcpy`
#include <numeric> // `std::iota`

#if __linux__ && defined(_GNU_SOURCE)
#include <stdlib.h> // `qsort_r`
#endif

#include <bench.hpp>

using namespace ashvardanian::stringzilla::scripts;

using strings_t = std::vector<std::string>;
using idx_t = sz_size_t;
using permute_t = std::vector<sz_u64_t>;

#pragma region - C callbacks

static char const *get_start(sz_sequence_t const *array_c, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(array_c->handle);
    return array[i].c_str();
}

static sz_size_t get_length(sz_sequence_t const *array_c, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(array_c->handle);
    return array[i].size();
}

static sz_bool_t has_under_four_chars(sz_sequence_t const *array_c, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(array_c->handle);
    return (sz_bool_t)(array[i].size() < 4);
}

#if defined(_MSC_VER)
static int _get_qsort_order(void *arg, const void *a, const void *b) {
#else
static int _get_qsort_order(const void *a, const void *b, void *arg) {
#endif
    sz_sequence_t *sequence = (sz_sequence_t *)arg;
    sz_size_t idx_a = *(sz_size_t *)a;
    sz_size_t idx_b = *(sz_size_t *)b;

    const char *str_a = sequence->get_start(sequence, idx_a);
    const char *str_b = sequence->get_start(sequence, idx_b);
    sz_size_t len_a = sequence->get_length(sequence, idx_a);
    sz_size_t len_b = sequence->get_length(sequence, idx_b);

    int res = strncmp(str_a, str_b, len_a < len_b ? len_a : len_b);
    return res ? res : (int)(len_a - len_b);
}

#pragma endregion

void populate_from_file(std::string path, strings_t &strings,
                        std::size_t limit = std::numeric_limits<std::size_t>::max()) {

    std::ifstream f(path, std::ios::in);
    std::string s;
    while (strings.size() < limit && std::getline(f, s, ' ')) strings.push_back(s);
}

constexpr size_t offset_in_word = 0;

static idx_t hybrid_sort_cpp(strings_t const &strings, sz_u64_t *order) {

    // What if we take up-to 4 first characters and the index
    for (size_t i = 0; i != strings.size(); ++i)
        std::memcpy((char *)&order[i] + offset_in_word, strings[order[i]].c_str(),
                    std::min<std::size_t>(strings[order[i]].size(), 4ul));

    std::sort(order, order + strings.size(), [&](sz_u64_t i, sz_u64_t j) {
        char *i_bytes = (char *)&i;
        char *j_bytes = (char *)&j;
        return *(uint32_t *)(i_bytes + offset_in_word) < *(uint32_t *)(j_bytes + offset_in_word);
    });

    for (size_t i = 0; i != strings.size(); ++i) std::memset((char *)&order[i] + offset_in_word, 0, 4ul);

    std::sort(order, order + strings.size(), [&](sz_u64_t i, sz_u64_t j) { return strings[i] < strings[j]; });

    return strings.size();
}

static idx_t hybrid_stable_sort_cpp(strings_t const &strings, sz_u64_t *order) {

    // What if we take up-to 4 first characters and the index
    for (size_t i = 0; i != strings.size(); ++i)
        std::memcpy((char *)&order[i] + offset_in_word, strings[order[i]].c_str(),
                    std::min<std::size_t>(strings[order[i]].size(), 4ull));

    std::stable_sort(order, order + strings.size(), [&](sz_u64_t i, sz_u64_t j) {
        char *i_bytes = (char *)&i;
        char *j_bytes = (char *)&j;
        return *(uint32_t *)(i_bytes + offset_in_word) < *(uint32_t *)(j_bytes + offset_in_word);
    });

    for (size_t i = 0; i != strings.size(); ++i) std::memset((char *)&order[i] + offset_in_word, 0, 4ul);

    std::stable_sort(order, order + strings.size(), [&](sz_u64_t i, sz_u64_t j) { return strings[i] < strings[j]; });

    return strings.size();
}

void expect_partitioned_by_length(strings_t const &strings, permute_t const &permute) {
    if (!std::is_partitioned(permute.begin(), permute.end(), [&](size_t i) { return strings[i].size() < 4; }))
        throw std::runtime_error("Partitioning failed!");
}

void expect_sorted(strings_t const &strings, permute_t const &permute) {
    if (!std::is_sorted(permute.begin(), permute.end(),
                        [&](std::size_t i, std::size_t j) { return strings[i] < strings[j]; }))
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
    double dif = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() * 1.0;
    double milisecs = dif / (iterations * 1e6);
    std::printf("Elapsed time is %.2lf miliseconds/iteration for %s.\n", milisecs, name);
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting sorting benchmarks.\n");
    dataset_t dataset = prepare_benchmark_environment(argc, argv);
    strings_t strings {dataset.tokens.begin(), dataset.tokens.end()};

    permute_t permute_base, permute_new;
    permute_base.resize(strings.size());
    permute_new.resize(strings.size());

    // Partitioning
    {
        std::printf("---- Partitioning:\n");
        bench_permute("std::partition", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
            std::partition(permute.begin(), permute.end(), [&](size_t i) { return strings[i].size() < 4; });
        });
        expect_partitioned_by_length(strings, permute_base);

        bench_permute("std::stable_partition", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
            std::stable_partition(permute.begin(), permute.end(), [&](size_t i) { return strings[i].size() < 4; });
        });
        expect_partitioned_by_length(strings, permute_base);

        bench_permute("sz_partition", strings, permute_new, [](strings_t const &strings, permute_t &permute) {
            sz_sequence_t array;
            array.order = permute.data();
            array.count = strings.size();
            array.handle = &strings;
            sz_partition(&array, &has_under_four_chars);
        });
        expect_partitioned_by_length(strings, permute_new);
        // TODO: expect_same(permute_base, permute_new);
    }

    // Sorting
    {
        std::printf("---- Sorting:\n");
        bench_permute("std::sort", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
            std::sort(permute.begin(), permute.end(), [&](idx_t i, idx_t j) { return strings[i] < strings[j]; });
        });
        expect_sorted(strings, permute_base);

        bench_permute("sz_sort", strings, permute_new, [](strings_t const &strings, permute_t &permute) {
            sz_sequence_t array;
            array.order = permute.data();
            array.count = strings.size();
            array.handle = &strings;
            array.get_start = get_start;
            array.get_length = get_length;
            sz_sort(&array);
        });
        expect_sorted(strings, permute_new);

#if __linux__ && defined(_GNU_SOURCE)
        bench_permute("qsort_r", strings, permute_new, [](strings_t const &strings, permute_t &permute) {
            sz_sequence_t array;
            array.order = permute.data();
            array.count = strings.size();
            array.handle = &strings;
            array.get_start = get_start;
            array.get_length = get_length;
            qsort_r(array.order, array.count, sizeof(sz_u64_t), _get_qsort_order, &array);
        });
        expect_sorted(strings, permute_new);
#elif defined(_MSC_VER)
        bench_permute("qsort_s", strings, permute_new, [](strings_t const &strings, permute_t &permute) {
            sz_sequence_t array;
            array.order = permute.data();
            array.count = strings.size();
            array.handle = &strings;
            array.get_start = get_start;
            array.get_length = get_length;
            qsort_s(array.order, array.count, sizeof(sz_u64_t), _get_qsort_order, &array);
        });
        expect_sorted(strings, permute_new);
#else
        sz_unused(_get_qsort_order);
#endif

        bench_permute("hybrid_sort_cpp", strings, permute_new,
                      [](strings_t const &strings, permute_t &permute) { hybrid_sort_cpp(strings, permute.data()); });
        expect_sorted(strings, permute_new);

        std::printf("---- Stable Sorting:\n");
        bench_permute("std::stable_sort", strings, permute_base, [](strings_t const &strings, permute_t &permute) {
            std::stable_sort(permute.begin(), permute.end(), [&](idx_t i, idx_t j) { return strings[i] < strings[j]; });
        });
        expect_sorted(strings, permute_base);

        bench_permute(
            "hybrid_stable_sort_cpp", strings, permute_base,
            [](strings_t const &strings, permute_t &permute) { hybrid_stable_sort_cpp(strings, permute.data()); });
        expect_sorted(strings, permute_new);
        expect_same(permute_base, permute_new);
    }

    return 0;
}