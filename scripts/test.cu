/**
 *  @brief   Extensive @b stress-testing suite for StringCuZilla parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test.cu
 *  @author  Ash Vardanian
 */
#undef NDEBUG // ! Enable all assertions for testing

/**
 *  ! Overload the following with caution.
 *  ! Those parameters must never be explicitly set during releases,
 *  ! but they come handy during development, if you want to validate
 *  ! different ISA-specific implementations.

#define SZ_USE_HASWELL 0
#define SZ_USE_SKYLAKE 0
#define SZ_USE_ICE 0
#define SZ_USE_NEON 0
#define SZ_USE_SVE 0
 */
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // Enforce aggressive logging for this unit.

/**
 *  ! Overload the following with caution to enable parallelism.
 *  ! They control the OpenMP CPU backend as well as the CUDA GPU backend.
 */
#include "stringcuzilla/similarity.hpp"

#if SZ_USE_CUDA
#include "stringcuzilla/similarity.cuh"
#endif

#if !_SZ_IS_CPP17
#error "This test requires C++17 or later."
#endif

#include "test.hpp" // `levenshtein_baseline`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using namespace std::literals; // for ""sv

struct levenshtein_baselines_t {
    template <typename results_type_>
    sz::status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, results_type_ *results) const {
        _sz_assert(first.size() == second.size());
#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = sz::scripts::levenshtein_baseline(first[i].data(), first[i].size(), //
                                                           second[i].data(), second[i].size());
        return sz::status_t::success_k;
    }
};

struct needleman_wunsch_baselines_t {

    sz::error_costs_256x256_t substitution_costs = sz::error_costs_256x256_t::diagonal();
    sz::error_cost_t gap_cost = -1;

    sz::status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, sz_ssize_t *results) const {
        _sz_assert(first.size() == second.size());

#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = sz::scripts::needleman_wunsch_baseline(first[i].data(), first[i].size(),   //
                                                                second[i].data(), second[i].size(), //
                                                                substitution_costs, gap_cost);
        return sz::status_t::success_k;
    }
};

struct smith_waterman_baselines_t {

    sz::error_costs_256x256_t substitution_costs = sz::error_costs_256x256_t::diagonal();
    sz::error_cost_t gap_cost = -1;

    sz::status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, sz_ssize_t *results) const {
        _sz_assert(first.size() == second.size());

#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = sz::scripts::smith_waterman_baseline(first[i].data(), first[i].size(),   //
                                                              second[i].data(), second[i].size(), //
                                                              substitution_costs, gap_cost);
        return sz::status_t::success_k;
    }
};

using levenshtein_serial_t = sz::levenshtein_distances<sz_cap_parallel_k, char, std::allocator<char>>;
using levenshtein_utf8_serial_t = sz::levenshtein_distances_utf8<sz_cap_parallel_k, char, std::allocator<char>>;
using needleman_wunsch_serial_t = sz::needleman_wunsch_scores<sz_cap_parallel_k, char, std::allocator<char>>;
using smith_waterman_serial_t = sz::smith_waterman_scores<sz_cap_parallel_k, char, std::allocator<char>>;

/**
 *  In @b AVX-512:
 *  - for Global Alignments, we can vectorize the min-max calculation for diagonal "walkers"
 *  - for Local Alignments, we can vectorize the character substitution lookups for horizontal "walkers"
 */
using levenshtein_ice_t = sz::levenshtein_distances<sz_cap_ice_k, char, std::allocator<char>>;
using levenshtein_utf8_ice_t = sz::levenshtein_distances_utf8<sz_cap_ice_k, char, std::allocator<char>>;
using needleman_wunsch_ice_t = sz::needleman_wunsch_scores<sz_cap_ice_k, char, std::allocator<char>>;
using smith_waterman_ice_t = sz::smith_waterman_scores<sz_cap_ice_k, char, std::allocator<char>>;

/**
 *  In @b CUDA:
 *  - for GPUs before Hopper, we can use the @b SIMT model for warp-level parallelism using diagonal "walkers"
 *  - for GPUs after Hopper, we compound that with thread-level @b SIMD via @b DPX instructions for min-max
 */
using levenshtein_cuda_t = sz::levenshtein_distances<sz_cap_cuda_k, char>;
using levenshtein_utf8_cuda_t = sz::levenshtein_distances_utf8<sz_cap_cuda_k, char>;
using needleman_wunsch_cuda_t = sz::needleman_wunsch_scores<sz_cap_cuda_k, char>;
using smith_waterman_cuda_t = sz::smith_waterman_scores<sz_cap_cuda_k, char>;

using levenshtein_hopper_t = sz::levenshtein_distances<sz_cap_hopper_k, char>;
using levenshtein_utf8_hopper_t = sz::levenshtein_distances_utf8<sz_cap_hopper_k, char>;
using needleman_wunsch_hopper_t = sz::needleman_wunsch_scores<sz_cap_hopper_k, char>;
using smith_waterman_hopper_t = sz::smith_waterman_scores<sz_cap_hopper_k, char>;

template <typename score_type_>
void edit_distance_log_mismatch(std::string const &first, std::string const &second, //
                                score_type_ result_base, score_type_ result_simd) {
    char const *ellipsis = first.length() > 22 || second.length() > 22 ? "..." : "";
    char const *format_string;
    constexpr bool is_signed = std::is_signed<score_type_>();
    if constexpr (is_signed) {
        format_string = "Edit Distance error (got %zd, expected %zd): \"%.22s%s\" ⇔ \"%.22s%s\" \n";
    }
    else { format_string = "Edit Distance error (got %zu, expected %zu): \"%.22s%s\" ⇔ \"%.22s%s\" \n"; }
    std::printf(format_string, result_simd, result_base, first.c_str(), ellipsis, second.c_str(), ellipsis);
}

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads
 *          on a @b fixed set of different representative ASCII and UTF-8 strings.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_>
static void edit_distances_fixed(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                 std::string_view allowed_chars = {}) {

    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"ABC", "ABC"},                  // same string; distance ~ 0
        {"LISTEN", "SILENT"},            // distance ~ 4
        {"ATCA", "CTACTCACCC"},          // distance ~ 6
        {"A", "="},                      // distance ~ 1
        {"A", "A"},                      // distance ~ 0
        {"", ""},                        // distance ~ 0
        {"", "ABC"},                     // distance ~ 3
        {"ABC", ""},                     // distance ~ 3
        {"ABC", "AC"},                   // one deletion; distance ~ 1
        {"ABC", "A_BC"},                 // one insertion; distance ~ 1
        {"ggbuzgjux{}l", "gbuzgjux{}l"}, // one (prepended) insertion; distance ~ 1
        {"ABC", "ADC"},                  // one substitution; distance ~ 1
        {"APPLE", "APLE"},               // distance ~ 1
        //
        // Unicode:
        {"αβγδ", "αγδ"},                      // Each Greek symbol is 2 bytes in size; 2 bytes, 1 runes diff.
        {"مرحبا بالعالم", "مرحبا يا عالم"},   // "Hello World" vs "Welcome to the World" ?; 3 bytes, 2 runes diff.
        {"école", "école"},                   // letter "é" as a single character vs "e" + "´"; 3 bytes, 2 runes diff.
        {"Schön", "Scho\u0308n"},             // "ö" represented as "o" + "¨"; 3 bytes, 2 runes diff.
        {"💖", "💗"},                         // 4-byte emojis: Different hearts; 1 bytes, 1 runes diff.
        {"𠜎 𠜱 𠝹 𠱓", "𠜎𠜱𠝹𠱓"},          // Ancient Chinese characters, no spaces vs spaces; 3 bytes, 3 runes
        {"München", "Muenchen"},              // German name with umlaut vs. its transcription; 2 bytes, 2 runes
        {"façade", "facade"},                 // "ç" represented as "c" with cedilla vs. plain "c"; 2 bytes, 1 runes
        {"こんにちは世界", "こんばんは世界"}, // "Good morning world" vs "Good evening world"; 3 bytes, 2 runes
        {"👩‍👩‍👧‍👦", "👨‍👩‍👧‍👦"}, // Different family emojis; 1 bytes, 1 runes
        {"Data科学123", "Data科學321"},                             // 3 bytes, 3 runes
        {"🙂🌍🚀", "🙂🌎✨"},                                       // 5 bytes, 2 runes
    };

    // First check with a batch-size of 1
    using score_t = score_type_;
    unified_vector<score_t> results_base(1), results_simd(1);
    arrow_strings_tape_t first_tape, second_tape;
    bool contains_missing_in_any_case = false;
    for (auto [first, second] : test_cases) {

        // Check if the input strings fit into our allowed characters set
        if (!allowed_chars.empty()) {
            bool contains_missing = false;
            for (auto c : first) contains_missing |= allowed_chars.find(c) == std::string_view::npos;
            for (auto c : second) contains_missing |= allowed_chars.find(c) == std::string_view::npos;
            contains_missing_in_any_case |= contains_missing;
            if (contains_missing) continue;
        }

        // Reset the tapes and results
        results_base[0] = 0, results_simd[0] = 0;
        first_tape.try_assign(&first, &first + 1);
        second_tape.try_assign(&second, &second + 1);

        // Compute with both backends
        sz::status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        sz::status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data());
        _sz_assert(status_base == sz::status_t::success_k);
        _sz_assert(status_simd == sz::status_t::success_k);
        if (results_base[0] != results_simd[0])
            edit_distance_log_mismatch(first, second, results_base[0], results_simd[0]);
    }

    // Unzip the test cases into two separate tapes and perform batch processing
    if (!contains_missing_in_any_case) {
        results_base.resize(test_cases.size());
        results_simd.resize(test_cases.size());
        first_tape.reset();
        second_tape.reset();
        for (auto [first, second] : test_cases) {
            _sz_assert(first_tape.try_append({first.data(), first.size()}) == sz::status_t::success_k);
            _sz_assert(second_tape.try_append({second.data(), second.size()}) == sz::status_t::success_k);
        }

        // Compute with both backends
        sz::status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        sz::status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data());
        _sz_assert(status_base == sz::status_t::success_k);
        _sz_assert(status_simd == sz::status_t::success_k);

        // Individually log the failed results
        for (std::size_t i = 0; i != test_cases.size(); ++i) {
            if (results_base[i] == results_simd[i]) continue;
            edit_distance_log_mismatch(test_cases[i].first, test_cases[i].second, results_base[i], results_simd[i]);
        }
    }
}

struct fuzzy_config_t {
    std::string_view alphabet = "ABC";
    std::size_t batch_size = 1024 * 16;
    std::size_t min_string_length = 1;
    std::size_t max_string_length = 512;
    std::size_t iterations = 10;
};

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads
 *          on a synthetic @b randomly-generated set of strings from a given @p alphabet.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_>
static void edit_distances_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                 fuzzy_config_t config = {}) {

    using score_t = score_type_;
    unified_vector<score_t> results_base(config.batch_size), results_simd(config.batch_size);
    std::vector<std::string> first_array(config.batch_size), second_array(config.batch_size);
    arrow_strings_tape_t first_tape, second_tape;
    std::uniform_int_distribution<std::size_t> length_distribution(config.min_string_length, config.max_string_length);

    // Generate some random strings, using a small alphabet
    for (std::size_t iteration_idx = 0; iteration_idx < config.iterations; ++iteration_idx) {
        for (std::size_t i = 0; i != config.batch_size; ++i) {
            std::size_t first_length = length_distribution(global_random_generator());
            std::size_t second_length = length_distribution(global_random_generator());
            first_array[i] = random_string(first_length, config.alphabet.data(), config.alphabet.size());
            second_array[i] = random_string(second_length, config.alphabet.data(), config.alphabet.size());
        }

        // Convert to a GPU-friendly layout
        first_tape.try_assign(first_array.data(), first_array.data() + config.batch_size);
        second_tape.try_assign(second_array.data(), second_array.data() + config.batch_size);

        // Compute with both backends
        sz::status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        sz::status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data());
        _sz_assert(status_base == sz::status_t::success_k);
        _sz_assert(status_simd == sz::status_t::success_k);

        // Individually log the failed results
        for (std::size_t i = 0; i != config.batch_size; ++i) {
            if (results_base[i] == results_simd[i]) continue;
            edit_distance_log_mismatch(first_array[i], second_array[i], results_base[i], results_simd[i]);
        }
    }
}

template <typename score_type_, typename base_operator_, typename simd_operator_>
static void edit_distances_fixed_and_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                           std::string_view allowed_chars = {}, fuzzy_config_t config = {}) {
    edit_distances_fixed<score_type_>(base_operator, simd_operator, allowed_chars);
    edit_distances_fuzzy<score_type_>(base_operator, simd_operator, config);
}

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance, NW & SW score computation,
 *          comparing the results to some baseline implementation for predefined and random inputs.
 */
static void test_equivalence() {

    using error_t = sz::error_cost_t;
    using error_matrix_t = sz::error_costs_256x256_t; // ? Full matrix for all 256 ASCII characters
    using error_mat_t = sz::error_costs_26x26ascii_t; // ? Smaller compact form for 26 capital ASCII characters

    // Our logic of computing NW and SW alignment similarity scores differs in sign from most implementations.
    // It's similar to how the "cosine distance" is the inverse of the "cosine similarity".
    // In our case we compute the "distance" and by negating the sign, we can compute the "similarity".
    constexpr error_t unary_match_score = 1;
    constexpr error_t unary_mismatch_score = 0;
    constexpr error_t unary_gap_score = 0;
    error_matrix_t substituter_unary = error_matrix_t::diagonal(unary_match_score, unary_mismatch_score);
    {
        auto distance_l = levenshtein_baseline("abcdefg", 7, "abc_efg", 7);
        auto similarity_nw = needleman_wunsch_baseline("abcdefg", 7, "abc_efg", 7, substituter_unary, unary_gap_score);
        auto similarity_sw = smith_waterman_baseline("abcdefg", 7, "abc_efg", 7, substituter_unary, unary_gap_score);
        // Distance can be computed from the similarity, by inverting the sign around the length of the longest string:
        auto distance_nw = std::max(7, 7) - similarity_nw;
        auto distance_sw = std::max(7, 7) - similarity_sw;
        _sz_assert(distance_l == 1);
        _sz_assert(distance_nw == 1);
        _sz_assert(distance_sw == 1);
    }

    // Now systematically compare the results of the baseline and SIMD implementations
    constexpr sz_capability_t serial_k = sz_cap_serial_k;
    constexpr sz_capability_t parallel_k = sz_cap_parallel_k;
    constexpr sz_capability_t cuda_k = sz_cap_cuda_k;
    constexpr sz_capability_t hopper_k = sz_cap_hopper_k;

    // Single-threaded serial Levenshtein distance implementation
    edit_distances_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_baselines_t {},            //
        sz::levenshtein_distances<serial_k, char, std::allocator<char>> {});

    // Multi-threaded parallel Levenshtein distance implementation
    edit_distances_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_baselines_t {},            //
        sz::levenshtein_distances<parallel_k, char, std::allocator<char>> {});

    // Now let's take non-unary substitution costs, like BLOSUM62
    constexpr error_t blosum62_gap_extension_cost = -4;
    error_mat_t blosum62_mat = sz::error_costs_26x26ascii_t::blosum62();
    error_matrix_t blosum62_matrix = blosum62_mat.decompressed();

    // Single-threaded serial NW implementation
    edit_distances_fixed_and_fuzzy<sz_ssize_t>(                                      //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        sz::needleman_wunsch_scores<serial_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost});

    // Multi-threaded parallel NW implementation
    edit_distances_fixed_and_fuzzy<sz_ssize_t>(                                      //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        sz::needleman_wunsch_scores<parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost});

    // Single-threaded serial SW implementation
    edit_distances_fixed_and_fuzzy<sz_ssize_t>(                                    //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        sz::smith_waterman_scores<serial_k, char, error_matrix_t, std::allocator<char>> {blosum62_matrix,
                                                                                         blosum62_gap_extension_cost});

    // Multi-threaded parallel SW implementation
    edit_distances_fixed_and_fuzzy<sz_ssize_t>(                                    //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        sz::smith_waterman_scores<parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost});

    // Switch to the GPU, using an identical matrix, but move it into unified memory
    unified_vector<error_mat_t> blosum62_unified(1);
    blosum62_unified[0] = blosum62_mat;

    // CUDA Levenshtein distance against Multi-threaded on CPU
    edit_distances_fixed_and_fuzzy<sz_size_t>(                                //
        sz::levenshtein_distances<parallel_k, char, std::allocator<char>> {}, //
        sz::levenshtein_distances<cuda_k, char> {});

#if SZ_USE_HOPPER
    // CUDA Levenshtein distance on Hopper against Multi-threaded on CPU
    edit_distances_fixed_and_fuzzy<sz_size_t>(                                //
        sz::levenshtein_distances<parallel_k, char, std::allocator<char>> {}, //
        sz::levenshtein_distances<hopper_k, char> {});
#endif

    // CUDA Needleman-Wunsch distance against Multi-threaded on CPU,
    // using a compressed smaller matrix to fit into GPU shared memory
    std::string_view ascii_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    edit_distances_fixed_and_fuzzy<sz_ssize_t>( //
        sz::needleman_wunsch_scores<parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost}, //
        sz::needleman_wunsch_scores<cuda_k, char, error_mat_t *> {blosum62_unified.data(), blosum62_gap_extension_cost},
        ascii_alphabet);
}

/**
 *  @brief  Many GPU algorithms depend on effective use of shared memory and scheduling its allocation for
 *          long inputs or very large batches isn't trivial.
 */
void test_growing_memory_usage() {

    // Now systematically compare the results of the baseline and SIMD implementations
    constexpr sz_capability_t serial_k = sz_cap_serial_k;
    constexpr sz_capability_t parallel_k = sz_cap_parallel_k;
    constexpr sz_capability_t cuda_k = sz_cap_cuda_k;
    constexpr sz_capability_t hopper_k = sz_cap_hopper_k;

    std::vector<fuzzy_config_t> experiments = {
        // Single string pair of same length:
        {.batch_size = 1, .min_string_length = 512, .max_string_length = 512, .iterations = 1},
        {.batch_size = 1, .min_string_length = 2048, .max_string_length = 2048, .iterations = 1},
        {.batch_size = 1, .min_string_length = 8192, .max_string_length = 8192, .iterations = 1},
        {.batch_size = 1, .min_string_length = 32768, .max_string_length = 32768, .iterations = 1},
        {.batch_size = 1, .min_string_length = 131072, .max_string_length = 131072, .iterations = 1},
        // Two strings of a same length:
        {.batch_size = 2, .min_string_length = 512, .max_string_length = 512, .iterations = 1},
        {.batch_size = 2, .min_string_length = 2048, .max_string_length = 2048, .iterations = 1},
        {.batch_size = 2, .min_string_length = 8192, .max_string_length = 8192, .iterations = 1},
        {.batch_size = 2, .min_string_length = 32768, .max_string_length = 32768, .iterations = 1},
        {.batch_size = 2, .min_string_length = 131072, .max_string_length = 131072, .iterations = 1},
        // Ten strings of random lengths:
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 512, .iterations = 1},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 2048, .iterations = 1},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 8192, .iterations = 1},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 32768, .iterations = 1},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 131072, .iterations = 1},
    };

    // Progress until something fails
    for (fuzzy_config_t const &experiment : experiments) {
        std::printf("Testing with batch size %zu, min length %zu, max length %zu, iterations %zu\n",
                    experiment.batch_size, experiment.min_string_length, experiment.max_string_length,
                    experiment.iterations);

        // Single-threaded serial Levenshtein distance implementation
        edit_distances_fuzzy<sz_size_t>( //
            levenshtein_baselines_t {},  //
            sz::levenshtein_distances<serial_k, char, std::allocator<char>> {}, experiment);

        // Multi-threaded parallel Levenshtein distance implementation
        edit_distances_fuzzy<sz_size_t>( //
            levenshtein_baselines_t {},  //
            sz::levenshtein_distances<parallel_k, char, std::allocator<char>> {}, experiment);
    }
}

int main(int argc, char const **argv) {

    // Let's greet the user nicely
    sz_unused(argc && argv);
    std::printf("Hi, dear tester! You look nice today!\n");
    std::printf("- Uses Haswell: %s \n", SZ_USE_HASWELL ? "yes" : "no");
    std::printf("- Uses Skylake: %s \n", SZ_USE_SKYLAKE ? "yes" : "no");
    std::printf("- Uses Ice Lake: %s \n", SZ_USE_ICE ? "yes" : "no");
    std::printf("- Uses NEON: %s \n", SZ_USE_NEON ? "yes" : "no");
    std::printf("- Uses SVE: %s \n", SZ_USE_SVE ? "yes" : "no");
    std::printf("- Uses SVE2: %s \n", SZ_USE_SVE2 ? "yes" : "no");
    std::printf("- Uses OpenMP: %s \n", SZ_USE_OPENMP ? "yes" : "no");
    std::printf("- Uses CUDA: %s \n", SZ_USE_CUDA ? "yes" : "no");

#if SZ_USE_CUDA
    cudaError_t cuda_error = cudaFree(0); // Force context initialization
    if (cuda_error != cudaSuccess) {
        std::printf("CUDA initialization error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    int device_count = 0;
    cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    std::printf("CUDA device count: %d\n", device_count);
    if (device_count == 0) {
        std::printf("No CUDA devices found.\n");
        return 1;
    }
    std::printf("- CUDA devices:\n");
    cudaDeviceProp prop;
    for (int i = 0; i < device_count; ++i) {
        cuda_error = cudaGetDeviceProperties(&prop, i);
        std::printf("  - %s\n", prop.name);
    }
    std::printf("- CUDA managed memory support: %s\n", prop.managedMemory == 1 ? "yes" : "no");
    std::printf("- CUDA unified memory support: %s\n", prop.unifiedAddressing == 1 ? "yes" : "no");
#endif

    test_equivalence();
    test_growing_memory_usage();

    std::printf("All tests passed... Unbelievable!\n");
    return 0;
}
