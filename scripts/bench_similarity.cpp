/**
 *  @file   bench_similarity.cpp
 *  @brief  Benchmarks string similarity computations.
 *
 *  This file is the sibling of `bench_sort.cpp`, `bench_search.cpp` and `bench_token.cpp`.
 *  It accepts a file with a list of words, and benchmarks the levenshtein edit-distance computations,
 *  alignment scores, and fingerprinting techniques combined with the Hamming distance.
 */
#include <bench.hpp>
#include <test.hpp> // `levenshtein_baseline`, `unary_substitution_costs`

using namespace ashvardanian::stringzilla::scripts;

using temporary_memory_t = std::vector<char>;
temporary_memory_t temporary_memory;

static void *allocate_from_vector(sz_size_t length, void *handle) {
    temporary_memory_t &vec = *reinterpret_cast<temporary_memory_t *>(handle);
    if (vec.size() < length) vec.resize(length);
    return vec.data();
}

static void free_from_vector(void *buffer, sz_size_t length, void *handle) { sz_unused(buffer && length && handle); }

tracked_binary_functions_t distance_functions() {
    // Populate the unary substitutions matrix
    static std::vector<std::int8_t> costs = unary_substitution_costs();

    // Two rows of the Levenshtein matrix will occupy this much:
    sz_memory_allocator_t alloc;
    alloc.allocate = &allocate_from_vector;
    alloc.free = &free_from_vector;
    alloc.handle = &temporary_memory;

    auto wrap_sz_distance = [alloc](auto function) mutable -> binary_function_t {
        return binary_function_t([function, alloc](std::string_view a_str, std::string_view b_str) mutable {
            sz_string_view_t a = to_c(a_str);
            sz_string_view_t b = to_c(b_str);
            return function(a.start, a.length, b.start, b.length, 0, &alloc);
        });
    };
    auto wrap_sz_scoring = [alloc](auto function) mutable -> binary_function_t {
        return binary_function_t([function, alloc](std::string_view a_str, std::string_view b_str) mutable {
            sz_string_view_t a = to_c(a_str);
            sz_string_view_t b = to_c(b_str);
            return function(a.start, a.length, b.start, b.length, 1, costs.data(), &alloc);
        });
    };
    tracked_binary_functions_t result = {
        {"naive", &levenshtein_baseline},
        {"sz_edit_distance", wrap_sz_distance(sz_edit_distance_serial), true},
        {"sz_alignment_score", wrap_sz_scoring(sz_alignment_score_serial), true},
    };
    return result;
}

template <typename strings_at>
void bench_similarity(strings_at &&strings) {
    if (strings.size() == 0) return;
    bench_binary_functions(strings, distance_functions());
}

void bench_similarity_on_bio_data() {
    std::vector<std::string> proteins;

    // A typical protein is 100-1000 amino acids long.
    // The alphabet is generally 20 amino acids, but that won't affect the throughput.
    char alphabet[2] = {'a', 'b'};
    constexpr std::size_t bio_samples = 128;
    struct {
        std::size_t length_lower_bound;
        std::size_t length_upper_bound;
        char const *name;
    } bio_cases[] = {
        {60, 60, "60 aminoacids"},       {100, 100, "100 aminoacids"},       {300, 300, "300 aminoacids"},
        {1000, 1000, "1000 aminoacids"}, {100, 1000, "100-1000 aminoacids"}, {1000, 10000, "1000-10000 aminoacids"},
    };
    std::random_device random_device;
    std::mt19937 generator(random_device());
    for (auto bio_case : bio_cases) {
        std::uniform_int_distribution<std::size_t> length_distribution(bio_case.length_lower_bound,
                                                                       bio_case.length_upper_bound);
        for (std::size_t i = 0; i != bio_samples; ++i) {
            std::size_t length = length_distribution(generator);
            std::string protein(length, 'a');
            std::generate(protein.begin(), protein.end(), [&]() { return alphabet[generator() % 2]; });
            proteins.push_back(protein);
        }

        std::printf("Benchmarking on protein-like sequences with %s:\n", bio_case.name);
        bench_similarity(proteins);
        proteins.clear();
    }
}

void bench_similarity_on_input_data(int argc, char const **argv) {

    dataset_t dataset = make_dataset(argc, argv);

    // Baseline benchmarks for real words, coming in all lengths
    std::printf("Benchmarking on real words:\n");
    bench_similarity(dataset.tokens);

    // Run benchmarks on tokens of different length
    for (std::size_t token_length : {20}) {
        std::printf("Benchmarking on real words of length %zu and longer:\n", token_length);
        bench_similarity(filter_by_length(dataset.tokens, token_length, std::greater_equal<std::size_t> {}));
    }
}

int main(int argc, char const **argv) {
    std::printf("StringZilla. Starting similarity benchmarks.\n");

    if (argc < 2) { bench_similarity_on_bio_data(); }
    else { bench_similarity_on_input_data(argc, argv); }

    std::printf("All benchmarks passed.\n");
    return 0;
}
