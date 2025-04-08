/**
 *  @brief   Extensive @b stress-testing suite for StringCuZilla parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test_stringcuzilla.cuh
 *  @author  Ash Vardanian
 */
#include "stringcuzilla/similarity.hpp"

#if SZ_USE_CUDA
#include "stringcuzilla/similarity.cuh"
#endif

#if !_SZ_IS_CPP17
#error "This test requires C++17 or later."
#endif

#include "test_stringzilla.hpp" // `arrow_strings_view_t`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

inline gpu_specs_t gpu_specs(int device = 0) noexcept(false) {
    gpu_specs_t specs;
#if SZ_USE_CUDA
    cudaDeviceProp prop;
    cudaError_t cuda_error = cudaGetDeviceProperties(&prop, device);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error(std::string("Error retrieving device properties: ") + cudaGetErrorString(cuda_error));

    // Set the GPU specs
    specs.streaming_multiprocessors = prop.multiProcessorCount;
    specs.constant_memory_bytes = prop.totalConstMem;
    specs.vram_bytes = prop.totalGlobalMem;

    // Infer other global settings, that CUDA doesn't expose directly
    specs.shared_memory_bytes = prop.sharedMemPerMultiprocessor * prop.multiProcessorCount;
    specs.cuda_cores = gpu_specs_t::cores_per_multiprocessor(prop.major, prop.minor) * specs.streaming_multiprocessors;

    // Scheduling-related constants
    specs.max_blocks_per_multiprocessor = prop.maxBlocksPerMultiProcessor;
    specs.reserved_memory_per_block = prop.reservedSharedMemPerBlock;
#endif
    return specs;
}

int log_environment() {
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
        if (cuda_error != cudaSuccess) {
            std::printf("Error retrieving properties for device %d: %s\n", i, cudaGetErrorString(cuda_error));
            continue;
        }
        int warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        int shared_memory_per_warp = (warps_per_sm > 0) ? (prop.sharedMemPerMultiprocessor / warps_per_sm) : 0;
        std::printf("  - %s\n", prop.name);
        std::printf("    Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        std::printf("    Maximum Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        std::printf("    Warp Size: %d threads\n", prop.warpSize);
        std::printf("    Max Warps per SM: %d warps\n", warps_per_sm);
        std::printf("    Shared Memory per Warp: %d bytes\n", shared_memory_per_warp);
    }
    std::printf("- CUDA managed memory support: %s\n", prop.managedMemory == 1 ? "yes" : "no");
    std::printf("- CUDA unified memory support: %s\n", prop.unifiedAddressing == 1 ? "yes" : "no");
#endif
    return 0;
}

/**
 *  @brief Inefficient baseline Levenshtein distance computation, as implemented in most codebases.
 *  @warning Allocates a new matrix on every call, with rows potentially scattered around memory.
 */
inline std::size_t levenshtein_baseline(char const *s1, std::size_t len1, char const *s2,
                                        std::size_t len2) noexcept(false) {
    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::size_t> matrix_buffer(rows * cols);

    // Initialize the borders of the matrix.
    for (std::size_t i = 0; i < rows; ++i) matrix_buffer[i * cols + 0] /* [i][0] in 2D */ = i;
    for (std::size_t j = 0; j < cols; ++j) matrix_buffer[0 * cols + j] /* [0][j] in 2D */ = j;

    for (std::size_t i = 1; i < rows; ++i) {
        std::size_t const *last_row = &matrix_buffer[(i - 1) * cols];
        std::size_t *row = &matrix_buffer[i * cols];
        for (std::size_t j = 1; j < cols; ++j) {
            std::size_t substitution_cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            std::size_t if_deletion_or_insertion = std::min(last_row[j], row[j - 1]) + 1;
            row[j] = std::min(if_deletion_or_insertion, last_row[j - 1] + substitution_cost);
        }
    }

    return matrix_buffer.back();
}

/**
 *  @brief Inefficient baseline Needleman-Wunsch alignment score computation, as implemented in most codebases.
 *  @warning Allocates a new matrix on every call, with rows potentially scattered around memory.
 */
inline std::ptrdiff_t needleman_wunsch_baseline(char const *s1, std::size_t len1, char const *s2, std::size_t len2,
                                                std::function<error_cost_t(char, char)> substitution_cost_for,
                                                error_cost_t gap_cost) noexcept(false) {
    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::ptrdiff_t> matrix_buffer(rows * cols);

    // Initialize the borders of the matrix.
    for (std::size_t i = 0; i < rows; ++i) matrix_buffer[i * cols + 0] /* [i][0] in 2D */ = i * gap_cost;
    for (std::size_t j = 0; j < cols; ++j) matrix_buffer[0 * cols + j] /* [0][j] in 2D */ = j * gap_cost;

    // Fill in the rest of the matrix.
    for (std::size_t i = 1; i < rows; ++i) {
        std::ptrdiff_t const *last_row = &matrix_buffer[(i - 1) * cols];
        std::ptrdiff_t *row = &matrix_buffer[i * cols];
        for (std::size_t j = 1; j < cols; ++j) {
            std::ptrdiff_t substitution_cost = substitution_cost_for(s1[i - 1], s2[j - 1]);
            std::ptrdiff_t if_substitution = last_row[j - 1] + substitution_cost;
            std::ptrdiff_t if_deletion_or_insertion = std::max(last_row[j], row[j - 1]) + gap_cost;
            row[j] = std::max(if_deletion_or_insertion, if_substitution);
        }
    }

    return matrix_buffer.back();
}

/**
 *  @brief Inefficient baseline Smith-Waterman local alignment score computation, as implemented in most codebases.
 *  @warning Allocates a new matrix on every call, with rows potentially scattered around memory.
 */
inline std::ptrdiff_t smith_waterman_baseline(char const *s1, std::size_t len1, char const *s2, std::size_t len2,
                                              std::function<error_cost_t(char, char)> substitution_cost_for,
                                              error_cost_t gap_cost) noexcept(false) {
    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::ptrdiff_t> matrix_buffer(rows * cols);

    // Unlike the global alignment we need to track the largest score in the matrix.
    std::ptrdiff_t best_score = 0;

    // Initialize the borders of the matrix to 0.
    for (std::size_t i = 0; i < rows; ++i) matrix_buffer[i * cols + 0] /* [i][0] in 2D */ = 0;
    for (std::size_t j = 0; j < cols; ++j) matrix_buffer[0 * cols + j] /* [0][j] in 2D */ = 0;

    // Fill in the rest of the matrix.
    for (std::size_t i = 1; i < rows; ++i) {
        std::ptrdiff_t const *last_row = &matrix_buffer[(i - 1) * cols];
        std::ptrdiff_t *row = &matrix_buffer[i * cols];
        for (std::size_t j = 1; j < cols; ++j) {
            std::ptrdiff_t substitution_cost = substitution_cost_for(s1[i - 1], s2[j - 1]);
            std::ptrdiff_t if_substitution = last_row[j - 1] + substitution_cost;
            std::ptrdiff_t if_deletion_or_insertion = std::max(row[j - 1], last_row[j]) + gap_cost;
            std::ptrdiff_t if_substitution_or_reset = std::max<std::ptrdiff_t>(if_substitution, 0);
            std::ptrdiff_t score = std::max(if_deletion_or_insertion, if_substitution_or_reset);
            row[j] = score;
            best_score = std::max(best_score, score);
        }
    }

    return best_score;
}

struct levenshtein_baselines_t {
    template <typename results_type_>
    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, results_type_ *results) const {
        _sz_assert(first.size() == second.size());
#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = levenshtein_baseline(first[i].data(), first[i].size(), //
                                              second[i].data(), second[i].size());
        return status_t::success_k;
    }
};

struct needleman_wunsch_baselines_t {

    error_costs_256x256_t substitution_costs = error_costs_256x256_t::diagonal();
    error_cost_t gap_cost = -1;

    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, sz_ssize_t *results) const {
        _sz_assert(first.size() == second.size());

#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = needleman_wunsch_baseline(first[i].data(), first[i].size(),   //
                                                   second[i].data(), second[i].size(), //
                                                   substitution_costs, gap_cost);
        return status_t::success_k;
    }
};

struct smith_waterman_baselines_t {

    error_costs_256x256_t substitution_costs = error_costs_256x256_t::diagonal();
    error_cost_t gap_cost = -1;

    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, sz_ssize_t *results) const {
        _sz_assert(first.size() == second.size());

#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = smith_waterman_baseline(first[i].data(), first[i].size(),   //
                                                 second[i].data(), second[i].size(), //
                                                 substitution_costs, gap_cost);
        return status_t::success_k;
    }
};

using levenshtein_serial_t = levenshtein_distances<sz_cap_parallel_k, char, std::allocator<char>>;
using levenshtein_utf8_serial_t = levenshtein_distances_utf8<sz_cap_parallel_k, char, std::allocator<char>>;
using needleman_wunsch_serial_t = needleman_wunsch_scores<sz_cap_parallel_k, char, std::allocator<char>>;
using smith_waterman_serial_t = smith_waterman_scores<sz_cap_parallel_k, char, std::allocator<char>>;

/**
 *  In @b AVX-512:
 *  - for Global Alignments, we can vectorize the min-max calculation for diagonal "walkers"
 *  - for Local Alignments, we can vectorize the character substitution lookups for horizontal "walkers"
 */
using levenshtein_ice_t = levenshtein_distances<sz_cap_ice_k, char, std::allocator<char>>;
using levenshtein_utf8_ice_t = levenshtein_distances_utf8<sz_cap_ice_k, char, std::allocator<char>>;
using needleman_wunsch_ice_t = needleman_wunsch_scores<sz_cap_ice_k, char, std::allocator<char>>;
using smith_waterman_ice_t = smith_waterman_scores<sz_cap_ice_k, char, std::allocator<char>>;

/**
 *  In @b CUDA:
 *  - for GPUs before Hopper, we can use the @b SIMT model for warp-level parallelism using diagonal "walkers"
 *  - for GPUs after Hopper, we compound that with thread-level @b SIMD via @b DPX instructions for min-max
 */
using levenshtein_cuda_t = levenshtein_distances<sz_cap_cuda_k, char>;
using levenshtein_utf8_cuda_t = levenshtein_distances_utf8<sz_cap_cuda_k, char>;
using needleman_wunsch_cuda_t = needleman_wunsch_scores<sz_cap_cuda_k, char>;
using smith_waterman_cuda_t = smith_waterman_scores<sz_cap_cuda_k, char>;

using levenshtein_hopper_t = levenshtein_distances<sz_cap_hopper_k, char>;
using levenshtein_utf8_hopper_t = levenshtein_distances_utf8<sz_cap_hopper_k, char>;
using needleman_wunsch_hopper_t = needleman_wunsch_scores<sz_cap_hopper_k, char>;
using smith_waterman_hopper_t = smith_waterman_scores<sz_cap_hopper_k, char>;

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
void test_similarity_scores_fixed(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
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
        status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data());
        _sz_assert(status_base == status_t::success_k);
        _sz_assert(status_simd == status_t::success_k);
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
            _sz_assert(first_tape.try_append({first.data(), first.size()}) == status_t::success_k);
            _sz_assert(second_tape.try_append({second.data(), second.size()}) == status_t::success_k);
        }

        // Compute with both backends
        status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data());
        _sz_assert(status_base == status_t::success_k);
        _sz_assert(status_simd == status_t::success_k);

        // Individually log the failed results
        for (std::size_t i = 0; i != test_cases.size(); ++i) {
            if (results_base[i] == results_simd[i]) continue;
            edit_distance_log_mismatch(test_cases[i].first, test_cases[i].second, results_base[i], results_simd[i]);
        }
    }
}

struct fuzzy_config_t {
    std::string_view alphabet = "ABC";
    std::size_t batch_size = 16;
    std::size_t min_string_length = 1;
    std::size_t max_string_length = 200;
    std::size_t iterations = 10;
};

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads
 *          on a synthetic @b randomly-generated set of strings from a given @p alphabet.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_>
void test_similarity_scores_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
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
        status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data());
        _sz_assert(status_base == status_t::success_k);
        _sz_assert(status_simd == status_t::success_k);

        // Individually log the failed results
        for (std::size_t i = 0; i != config.batch_size; ++i) {
            if (results_base[i] == results_simd[i]) continue;
            edit_distance_log_mismatch(first_array[i], second_array[i], results_base[i], results_simd[i]);
        }
    }
}

template <typename score_type_, typename base_operator_, typename simd_operator_>
void test_similarity_scores_fixed_and_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                            std::string_view allowed_chars = {}, fuzzy_config_t config = {}) {
    test_similarity_scores_fixed<score_type_>(base_operator, simd_operator, allowed_chars);
    test_similarity_scores_fuzzy<score_type_>(base_operator, simd_operator, config);
}

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance, NW & SW score computation,
 *          comparing the results to some baseline implementation for predefined and random inputs.
 */
void test_similarity_scores_equivalence() {

    using error_t = error_cost_t;
    using error_matrix_t = error_costs_256x256_t; // ? Full matrix for all 256 ASCII characters
    using error_mat_t = error_costs_26x26ascii_t; // ? Smaller compact form for 26 capital ASCII characters

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

    // Single-threaded serial Levenshtein distance implementation
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_baselines_t {},                    //
        levenshtein_distances<sz_cap_serial_k, char, std::allocator<char>> {});

    // Multi-threaded parallel Levenshtein distance implementation
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_baselines_t {},                    //
        levenshtein_distances<sz_cap_parallel_k, char, std::allocator<char>> {});

    // Now let's take non-unary substitution costs, like BLOSUM62
    constexpr error_t blosum62_gap_extension_cost = -4;
    error_mat_t blosum62_mat = error_costs_26x26ascii_t::blosum62();
    error_matrix_t blosum62_matrix = blosum62_mat.decompressed();

    // Single-threaded serial NW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                              //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        needleman_wunsch_scores<sz_cap_serial_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost});

    // Multi-threaded parallel NW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                              //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        needleman_wunsch_scores<sz_cap_parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost});

    // Single-threaded serial SW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                            //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        smith_waterman_scores<sz_cap_serial_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost});

    // Multi-threaded parallel SW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                            //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        smith_waterman_scores<sz_cap_parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost});

    // Switch to the GPU, using an identical matrix, but move it into unified memory
    unified_vector<error_mat_t> blosum62_unified(1);
    blosum62_unified[0] = blosum62_mat;

#if SZ_USE_CUDA
    // CUDA Levenshtein distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                           //
        levenshtein_distances<sz_cap_parallel_k, char, std::allocator<char>> {}, //
        levenshtein_distances<sz_cap_cuda_k, char> {});
#endif

#if SZ_USE_HOPPER && 0
    // CUDA Levenshtein distance on Hopper against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                           //
        levenshtein_distances<sz_cap_parallel_k, char, std::allocator<char>> {}, //
        levenshtein_distances<sz_cap_hopper_k, char> {});
#endif

#if SZ_USE_CUDA
    // CUDA Needleman-Wunsch distance against Multi-threaded on CPU,
    // using a compressed smaller matrix to fit into GPU shared memory
    std::string_view ascii_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<sz_cap_parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62_matrix, blosum62_gap_extension_cost}, //
        needleman_wunsch_scores<sz_cap_cuda_k, char, error_mat_t *> {blosum62_unified.data(),
                                                                     blosum62_gap_extension_cost},
        ascii_alphabet);
#endif
}

/**
 *  @brief  Many GPU algorithms depend on effective use of shared memory and scheduling its allocation for
 *          long inputs or very large batches isn't trivial.
 */
void test_similarity_scores_memory_usage() {

    std::vector<fuzzy_config_t> experiments = {
        // Single string pair of same length:
        {.batch_size = 1, .min_string_length = 128, .max_string_length = 128, .iterations = 1},
        {.batch_size = 1, .min_string_length = 512, .max_string_length = 512, .iterations = 1},
        {.batch_size = 1, .min_string_length = 2048, .max_string_length = 2048, .iterations = 1},
        {.batch_size = 1, .min_string_length = 8192, .max_string_length = 8192, .iterations = 1},
        {.batch_size = 1, .min_string_length = 32768, .max_string_length = 32768, .iterations = 1},
        {.batch_size = 1, .min_string_length = 131072, .max_string_length = 131072, .iterations = 1},
        // Two strings of a same length:
        {.batch_size = 2, .min_string_length = 128, .max_string_length = 128, .iterations = 1},
        {.batch_size = 2, .min_string_length = 512, .max_string_length = 512, .iterations = 1},
        {.batch_size = 2, .min_string_length = 2048, .max_string_length = 2048, .iterations = 1},
        {.batch_size = 2, .min_string_length = 8192, .max_string_length = 8192, .iterations = 1},
        {.batch_size = 2, .min_string_length = 32768, .max_string_length = 32768, .iterations = 1},
        {.batch_size = 2, .min_string_length = 131072, .max_string_length = 131072, .iterations = 1},
        // Ten strings of random lengths:
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 128, .iterations = 1},
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

        // Multi-threaded serial Levenshtein distance implementation
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_baselines_t {},          //
            levenshtein_distances<sz_cap_parallel_k, char, std::allocator<char>> {}, experiment);

        // CUDA Levenshtein distance against Multi-threaded on CPU
        test_similarity_scores_fuzzy<sz_size_t>(                                     //
            levenshtein_distances<sz_cap_parallel_k, char, std::allocator<char>> {}, //
            levenshtein_distances<sz_cap_cuda_k, char, std::allocator<char>> {}, experiment);
    }
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
