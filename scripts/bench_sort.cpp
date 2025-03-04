
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
namespace sz = ashvardanian::stringzilla;

using strings_t = std::vector<std::string>;
using permute_t = std::vector<sz_sorted_idx_t>;

#pragma region C callbacks

static sz_cptr_t get_start(void const *handle, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(handle);
    return array[i].c_str();
}

static sz_size_t get_length(void const *handle, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(handle);
    return array[i].size();
}

#if defined(_MSC_VER)
static int _get_qsort_order(void *arg, const void *a, const void *b) {
#else
static int _get_qsort_order(const void *a, const void *b, void *arg) {
#endif
    sz_sequence_t *sequence = (sz_sequence_t *)arg;
    sz_size_t idx_a = *(sz_size_t *)a;
    sz_size_t idx_b = *(sz_size_t *)b;

    char const *str_a = sequence->get_start(sequence->handle, idx_a);
    char const *str_b = sequence->get_start(sequence->handle, idx_b);
    sz_size_t len_a = sequence->get_length(sequence->handle, idx_a);
    sz_size_t len_b = sequence->get_length(sequence->handle, idx_b);

    int res = strncmp(str_a, str_b, len_a < len_b ? len_a : len_b);
    return res ? res : (int)(len_a - len_b);
}

#pragma endregion

template <typename strings_type_>
void expect_sorted(strings_type_ const &strings, permute_t const &permute) {
    if (!std::is_sorted(permute.begin(), permute.end(),
                        [&](std::size_t i, std::size_t j) { return strings[i] < strings[j]; }))
        throw std::runtime_error("Sorting failed!");
}

template <typename callback_type_>
void bench_permute(char const *name, callback_type_ &&callback) {

    // Run multiple iterations
    std::size_t iterations = 0;
    seconds_t duration = repeat_until_limit([&]() {
        callback();
        iterations++;
    });

    // Measure elapsed time
    duration /= iterations;
    if (duration >= 0.1) { std::printf("Elapsed time is %.2lf seconds for %s.\n", duration, name); }
    else if (duration >= 0.001) { std::printf("Elapsed time is %.2lf milliseconds for %s.\n", duration * 1e3, name); }
    else { std::printf("Elapsed time is %.2lf microseconds for %s.\n", duration * 1e6, name); }
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting sorting benchmarks.\n");
    dataset_t const dataset = prepare_benchmark_environment(argc, argv);
    strings_t const strings {dataset.tokens.begin(), dataset.tokens.end()};
    permute_t permute(strings.size());
    using allocator_t = std::allocator<char>;

    // Before sorting the strings themselves, which is a heavy operation, let's sort some prefixes
    // to understand how the sorting algorithm behaves.
    std::vector<sz_pgram_t> pgrams(strings.size());
    std::transform(strings.begin(), strings.end(), pgrams.begin(), [](std::string const &str) {
        sz_pgram_t pgram = 0;
        std::memcpy(&pgram, str.c_str(), (std::min)(sizeof(pgram), str.size()));
        return pgram;
    });

    // Sorting P-grams
    bench_permute("std::sort(pgrams)", [&]() {
        std::iota(permute.begin(), permute.end(), 0);
        std::sort(permute.begin(), permute.end(),
                  [&](sz_sorted_idx_t i, sz_sorted_idx_t j) { return pgrams[i] < pgrams[j]; });
    });
    expect_sorted(pgrams, permute);

    // Unlike the `std::sort` adaptation above, the `sz_pgrams_sort_serial` also sorts the input array inplace
    std::vector<sz_pgram_t> pgrams_sorted(strings.size());
    bench_permute("sz_pgrams_sort_serial", [&]() {
        std::copy(pgrams.begin(), pgrams.end(), pgrams_sorted.begin());
        std::iota(permute.begin(), permute.end(), 0);
        sz::_with_alloc<allocator_t>([&](sz_memory_allocator_t &alloc) {
            return sz_pgrams_sort_serial(pgrams_sorted.data(), pgrams_sorted.size(), &alloc, permute.data());
        });
    });
    expect_sorted(pgrams, permute);

    bench_permute("sz_pgrams_sort_skylake", [&]() {
        std::copy(pgrams.begin(), pgrams.end(), pgrams_sorted.begin());
        std::iota(permute.begin(), permute.end(), 0);
        sz::_with_alloc<allocator_t>([&](sz_memory_allocator_t &alloc) {
            return sz_pgrams_sort_skylake(pgrams_sorted.data(), pgrams_sorted.size(), &alloc, permute.data());
        });
    });
    expect_sorted(pgrams, permute);

    // Sorting strings
    bench_permute("std::sort(positions)", [&]() {
        std::iota(permute.begin(), permute.end(), 0);
        std::sort(permute.begin(), permute.end(),
                  [&](sz_sorted_idx_t i, sz_sorted_idx_t j) { return strings[i] < strings[j]; });
    });
    expect_sorted(strings, permute);

    bench_permute("sz_sequence_argsort_serial", [&]() {
        std::iota(permute.begin(), permute.end(), 0);
        sz_sequence_t array;
        array.count = strings.size();
        array.handle = &strings;
        array.get_start = get_start;
        array.get_length = get_length;
        sz::_with_alloc<allocator_t>(
            [&](sz_memory_allocator_t &alloc) { return sz_sequence_argsort_serial(&array, &alloc, permute.data()); });
    });
    expect_sorted(strings, permute);

    bench_permute("sz_sequence_argsort_skylake", [&]() {
        std::iota(permute.begin(), permute.end(), 0);
        sz_sequence_t array;
        array.count = strings.size();
        array.handle = &strings;
        array.get_start = get_start;
        array.get_length = get_length;
        sz::_with_alloc<allocator_t>(
            [&](sz_memory_allocator_t &alloc) { return sz_sequence_argsort_skylake(&array, &alloc, permute.data()); });
    });
    expect_sorted(strings, permute);

#if __linux__ && defined(_GNU_SOURCE) && !defined(__BIONIC__)
    bench_permute("qsort_r", [&]() {
        std::iota(permute.begin(), permute.end(), 0);
        sz_sequence_t array;
        array.count = strings.size();
        array.handle = &strings;
        array.get_start = get_start;
        array.get_length = get_length;
        qsort_r(permute.data(), array.count, sizeof(sz_sorted_idx_t), _get_qsort_order, &array);
    });
    expect_sorted(strings, permute);
#elif defined(_MSC_VER)
    bench_permute("qsort_s", [&]() {
        std::iota(permute.begin(), permute.end(), 0);
        sz_sequence_t array;
        array.count = strings.size();
        array.handle = &strings;
        array.get_start = get_start;
        array.get_length = get_length;
        qsort_s(permute.data(), array.count, sizeof(sz_sorted_idx_t), _get_qsort_order, &array);
    });
    expect_sorted(strings, permute);
#else
    sz_unused(_get_qsort_order);
#endif

    std::printf("---- Stable Sorting:\n");
    bench_permute("std::stable_sort", [&]() {
        std::iota(permute.begin(), permute.end(), 0);
        std::stable_sort(permute.begin(), permute.end(),
                         [&](sz_sorted_idx_t i, sz_sorted_idx_t j) { return strings[i] < strings[j]; });
    });
    expect_sorted(strings, permute);

    return 0;
}
