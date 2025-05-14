/**
 *  @brief   Extensive @b stress-testing suite for StringCuZilla parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test_stringcuzilla.cuh
 *  @author  Ash Vardanian
 */
#include <cstring> // `std::memcmp`
#include <thread>  // `std::thread::hardware_concurrency`

#include <fork_union.hpp> // Fork-join scoped thread pool

#include "stringcuzilla/find_many.hpp"
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
inline std::size_t levenshtein_baseline(                                //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    error_cost_t match_cost = 0, error_cost_t mismatch_cost = 1, error_cost_t gap_cost = 1) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::size_t> matrix_buffer(rows * cols);

    // Initialize the borders of the matrix.
    for (std::size_t i = 0; i < rows; ++i) matrix_buffer[i * cols + 0] /* [i][0] in 2D */ = i * gap_cost;
    for (std::size_t j = 0; j < cols; ++j) matrix_buffer[0 * cols + j] /* [0][j] in 2D */ = j * gap_cost;

    for (std::size_t i = 1; i < rows; ++i) {
        std::size_t const *last_row = &matrix_buffer[(i - 1) * cols];
        std::size_t *row = &matrix_buffer[i * cols];
        for (std::size_t j = 1; j < cols; ++j) {
            std::size_t substitution_cost = (s1[i - 1] == s2[j - 1]) ? match_cost : mismatch_cost;
            std::size_t if_deletion_or_insertion = std::min(last_row[j], row[j - 1]) + gap_cost;
            row[j] = std::min(if_deletion_or_insertion, last_row[j - 1] + substitution_cost);
        }
    }

    return matrix_buffer.back();
}

/**
 *  @brief Inefficient baseline Needleman-Wunsch alignment score computation, as implemented in most codebases.
 *  @warning Allocates a new matrix on every call, with rows potentially scattered around memory.
 */
inline std::ptrdiff_t needleman_wunsch_baseline(                        //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    std::function<error_cost_t(char, char)> substitution_cost_for, error_cost_t gap_cost) noexcept(false) {

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

/**
 *  @brief Inefficient baseline Levenshtein-Gotoh distance computation, as implemented in most codebases.
 *  @warning Allocates a new matrix on every call, with rows potentially scattered around memory.
 */
inline std::size_t levenshtein_gotoh_baseline(                          //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    error_cost_t match_cost, error_cost_t mismatch_cost,                //
    error_cost_t gap_opening_cost, error_cost_t gap_extension_cost) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::size_t> matrix_scores(rows * cols);
    std::vector<std::size_t> matrix_inserts(rows * cols);
    std::vector<std::size_t> matrix_deletes(rows * cols);

    // Initialize the borders of the matrix.
    // The supplementary matrices are initialized with values of higher magnitude,
    // which is equivalent to discarding them. That's better than using `SIZE_MAX`
    // as subsequent additions won't overflow.
    matrix_scores[0] = 0;
    for (std::size_t j = 1; j < cols; ++j) {
        matrix_scores[0 * cols + j] = gap_opening_cost + (j - 1) * gap_extension_cost;
        matrix_deletes[0 * cols + j] = matrix_scores[0 * cols + j] + gap_opening_cost + gap_extension_cost;
    }
    for (std::size_t i = 1; i < rows; ++i) {
        matrix_scores[i * cols + 0] = gap_opening_cost + (i - 1) * gap_extension_cost;
        matrix_inserts[i * cols + 0] = matrix_scores[i * cols + 0] + gap_opening_cost + gap_extension_cost;
    }

    // Fill in the rest of the matrix.
    for (std::size_t i = 1; i < rows; ++i) {
        std::size_t const *last_row = &matrix_scores[(i - 1) * cols];
        std::size_t *row = &matrix_scores[i * cols];
        std::size_t *row_inserts = &matrix_inserts[i * cols];
        std::size_t const *last_deletes_row = &matrix_deletes[(i - 1) * cols];
        std::size_t *row_deletes = &matrix_deletes[i * cols];
        for (std::size_t j = 1; j < cols; ++j) {
            std::size_t substitution_cost = (s1[i - 1] == s2[j - 1]) ? match_cost : mismatch_cost;
            std::size_t if_substitution = last_row[j - 1] + substitution_cost;
            std::size_t if_insertion =
                std::min<std::size_t>(row[j - 1] + gap_opening_cost, row_inserts[j - 1] + gap_extension_cost);
            std::size_t if_deletion =
                std::min<std::size_t>(last_row[j] + gap_opening_cost, last_deletes_row[j] + gap_extension_cost);
            std::size_t if_deletion_or_insertion = std::min(if_deletion, if_insertion);
            row[j] = std::min(if_deletion_or_insertion, if_substitution);
            row_inserts[j] = if_insertion;
            row_deletes[j] = if_deletion;
        }
    }

    return matrix_scores.back();
}

/**
 *  @brief Inefficient baseline Needleman-Wunsch-Gotoh alignment score computation, as implemented in most codebases.
 *  @warning Allocates a new matrix on every call, with rows potentially scattered around memory.
 *  @see https://github.com/gata-bio/affine-gaps
 */
inline std::ptrdiff_t needleman_wunsch_gotoh_baseline(                  //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    std::function<error_cost_t(char, char)> substitution_cost_for,      //
    error_cost_t gap_opening_cost, error_cost_t gap_extension_cost) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::ptrdiff_t> matrix_scores(rows * cols);
    std::vector<std::ptrdiff_t> matrix_inserts(rows * cols);
    std::vector<std::ptrdiff_t> matrix_deletes(rows * cols);

    // Initialize the borders of the matrix.
    matrix_scores[0] = 0;
    for (std::size_t i = 1; i < rows; ++i) {
        matrix_scores[i * cols + 0] = gap_opening_cost + (i - 1) * gap_extension_cost;
        matrix_inserts[i * cols + 0] = matrix_scores[i * cols + 0] + gap_opening_cost + gap_extension_cost;
    }
    for (std::size_t j = 1; j < cols; ++j) {
        matrix_scores[0 * cols + j] = gap_opening_cost + (j - 1) * gap_extension_cost;
        matrix_deletes[0 * cols + j] = matrix_scores[0 * cols + j] + gap_opening_cost + gap_extension_cost;
    }

    // Fill in the rest of the matrix.
    for (std::size_t i = 1; i < rows; ++i) {
        std::ptrdiff_t const *last_row = &matrix_scores[(i - 1) * cols];
        std::ptrdiff_t *row = &matrix_scores[i * cols];
        std::ptrdiff_t *row_inserts = &matrix_inserts[i * cols];
        std::ptrdiff_t const *last_deletes_row = &matrix_deletes[(i - 1) * cols];
        std::ptrdiff_t *row_deletes = &matrix_deletes[i * cols];
        for (std::size_t j = 1; j < cols; ++j) {
            std::ptrdiff_t substitution_cost = substitution_cost_for(s1[i - 1], s2[j - 1]);
            std::ptrdiff_t if_substitution = last_row[j - 1] + substitution_cost;
            std::ptrdiff_t if_insertion =
                std::max(row[j - 1] + gap_opening_cost, row_inserts[j - 1] + gap_extension_cost);
            std::ptrdiff_t if_deletion =
                std::max(last_row[j] + gap_opening_cost, last_deletes_row[j] + gap_extension_cost);
            std::ptrdiff_t if_deletion_or_insertion = std::max(if_deletion, if_insertion);
            row[j] = std::max(if_deletion_or_insertion, if_substitution);
            row_inserts[j] = if_insertion;
            row_deletes[j] = if_deletion;
        }
    }

    return matrix_scores.back();
}

/**
 *  @brief Inefficient baseline Smith-Waterman-Gotoh alignment score computation, as implemented in most codebases.
 *  @warning Allocates a new matrix on every call, with rows potentially scattered around memory.
 *  @see https://github.com/gata-bio/affine-gaps
 */
inline std::ptrdiff_t smith_waterman_gotoh_baseline(                    //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    std::function<error_cost_t(char, char)> substitution_cost_for,      //
    error_cost_t gap_opening_cost, error_cost_t gap_extension_cost) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::ptrdiff_t> matrix_scores(rows * cols);
    std::vector<std::ptrdiff_t> matrix_inserts(rows * cols);
    std::vector<std::ptrdiff_t> matrix_deletes(rows * cols);

    // Unlike the global alignment we need to track the largest score in the matrix.
    std::ptrdiff_t best_score = 0;

    // Initialize the borders of the matrix.
    matrix_scores[0] = 0;
    for (std::size_t i = 1; i < rows; ++i) {
        matrix_scores[i * cols + 0] = 0;
        matrix_inserts[i * cols + 0] = gap_opening_cost + gap_extension_cost;
    }
    for (std::size_t j = 1; j < cols; ++j) {
        matrix_scores[0 * cols + j] = 0;
        matrix_deletes[0 * cols + j] = gap_opening_cost + gap_extension_cost;
    }

    // Fill in the rest of the matrix.
    for (std::size_t i = 1; i < rows; ++i) {
        std::ptrdiff_t const *last_row = &matrix_scores[(i - 1) * cols];
        std::ptrdiff_t *row = &matrix_scores[i * cols];
        std::ptrdiff_t *row_inserts = &matrix_inserts[i * cols];
        std::ptrdiff_t const *last_deletes_row = &matrix_deletes[(i - 1) * cols];
        std::ptrdiff_t *row_deletes = &matrix_deletes[i * cols];
        for (std::size_t j = 1; j < cols; ++j) {
            std::ptrdiff_t substitution_cost = substitution_cost_for(s1[i - 1], s2[j - 1]);
            std::ptrdiff_t if_substitution = last_row[j - 1] + substitution_cost;
            std::ptrdiff_t if_insertion =
                std::max(row[j - 1] + gap_opening_cost, row_inserts[j - 1] + gap_extension_cost);
            std::ptrdiff_t if_deletion =
                std::max(last_row[j] + gap_opening_cost, last_deletes_row[j] + gap_extension_cost);
            std::ptrdiff_t if_deletion_or_insertion = std::max(if_deletion, if_insertion);
            std::ptrdiff_t if_substitution_or_reset = std::max<std::ptrdiff_t>(if_substitution, 0);
            std::ptrdiff_t score = std::max(if_deletion_or_insertion, if_substitution_or_reset);
            row[j] = score;
            row_inserts[j] = if_insertion;
            row_deletes[j] = if_deletion;
            best_score = std::max(best_score, score);
        }
    }

    return best_score;
}

struct levenshtein_baselines_t {

    uniform_substitution_costs_t substitution_costs = {0, 1};
    error_cost_t gap_opening_cost = {1};
    error_cost_t gap_extension_cost = {1};

    levenshtein_baselines_t() = default;
    levenshtein_baselines_t(uniform_substitution_costs_t subs, linear_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open_or_extend), gap_extension_cost(gap.open_or_extend) {}
    levenshtein_baselines_t(uniform_substitution_costs_t subs, affine_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open), gap_extension_cost(gap.extend) {}

    template <typename results_type_>
    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, results_type_ *results) const {
        _sz_assert(first.size() == second.size());
#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] =
                gap_opening_cost == gap_extension_cost
                    ? levenshtein_baseline(first[i].data(), first[i].size(),   //
                                           second[i].data(), second[i].size(), //
                                           substitution_costs.match, substitution_costs.mismatch, gap_opening_cost)
                    : levenshtein_gotoh_baseline(first[i].data(), first[i].size(),   //
                                                 second[i].data(), second[i].size(), //
                                                 substitution_costs.match, substitution_costs.mismatch,
                                                 gap_opening_cost, gap_extension_cost);
        return status_t::success_k;
    }
};

struct needleman_wunsch_baselines_t {

    error_costs_256x256_t substitution_costs = error_costs_256x256_t::diagonal();
    error_cost_t gap_opening_cost = -1;
    error_cost_t gap_extension_cost = -1;

    needleman_wunsch_baselines_t() = default;
    needleman_wunsch_baselines_t(error_costs_256x256_t subs, linear_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open_or_extend), gap_extension_cost(gap.open_or_extend) {}
    needleman_wunsch_baselines_t(error_costs_256x256_t subs, affine_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open), gap_extension_cost(gap.extend) {}

    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, sz_ssize_t *results) const {
        _sz_assert(first.size() == second.size());

#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] =
                gap_opening_cost == gap_extension_cost
                    ? needleman_wunsch_baseline(first[i].data(), first[i].size(),   //
                                                second[i].data(), second[i].size(), //
                                                substitution_costs, gap_opening_cost)
                    : needleman_wunsch_gotoh_baseline(first[i].data(), first[i].size(),   //
                                                      second[i].data(), second[i].size(), //
                                                      substitution_costs, gap_opening_cost, gap_extension_cost);
        return status_t::success_k;
    }
};

struct smith_waterman_baselines_t {

    error_costs_256x256_t substitution_costs = error_costs_256x256_t::diagonal();
    error_cost_t gap_opening_cost = -1;
    error_cost_t gap_extension_cost = -1;

    smith_waterman_baselines_t() = default;
    smith_waterman_baselines_t(error_costs_256x256_t subs, linear_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open_or_extend), gap_extension_cost(gap.open_or_extend) {}
    smith_waterman_baselines_t(error_costs_256x256_t subs, affine_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open), gap_extension_cost(gap.extend) {}

    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, sz_ssize_t *results) const {
        _sz_assert(first.size() == second.size());

#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = gap_opening_cost == gap_extension_cost
                             ? smith_waterman_baseline(first[i].data(), first[i].size(),   //
                                                       second[i].data(), second[i].size(), //
                                                       substitution_costs, gap_opening_cost)
                             : smith_waterman_gotoh_baseline(first[i].data(), first[i].size(),   //
                                                             second[i].data(), second[i].size(), //
                                                             substitution_costs, gap_opening_cost, gap_extension_cost);
        return status_t::success_k;
    }
};

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
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_similarity_scores_fixed(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                  std::string_view allowed_chars = {}, extra_args_ &&...extra_args) {

    std::vector<std::pair<std::string, std::string>> test_cases;
    auto append = [&test_cases](std::string const &first, std::string const &second) {
        test_cases.emplace_back(first, second);
    };

    // Some vary basic variants:
    append("ggbuzgjux{}l", "gbuzgjux{}l"); // one (prepended) insertion; distance ~ 1
    append("A", "A");                      // distance ~ 0
    append("A", "=");                      // distance ~ 1
    append("", "");                        // distance ~ 0
    append("ABC", "ABC");                  // same string; distance ~ 0
    append("ABC", "AABC");                 // distance ~ 1, prepended
    append("ABC", "ABCC");                 // distance ~ 1, appended
    append("", "ABC");                     // distance ~ 3
    append("ABC", "");                     // distance ~ 3
    append("ABC", "AC");                   // one deletion; distance ~ 1
    append("ABC", "AXBC");                 // one X insertion; distance ~ 1
    append("ABC", "AXC");                  // one X substitution; distance ~ 1
    append("ABCDEFG", "ABCXEFG");          // one X substitution; distance ~ 1
    append("LISTEN", "SILENT");            // distance ~ 4
    append("ATCA", "CTACTCACCC");          // distance ~ 6
    append("APPLE", "APLE");               // distance ~ 1

    // Longer strings made of simple characters:
    append("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"); // same string; distance ~ 0
    append("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ABCD_FGHI_KLMNOP_RSTU_WXYZ"); // same length; 4 substitutions; distance ~ 4

    // Short Unicode samples that we also use on the Python side:
    append("αβγδ", "αγδ");                      // Each Greek symbol is 2 bytes in size; 2 bytes, 1 runes diff.
    append("école", "école");                   // letter "é" as a single character vs "e" + "´"; 3 bytes, 2 runes diff.
    append("Schön", "Scho\u0308n");             // "ö" represented as "o" + "¨"; 3 bytes, 2 runes diff.
    append("Data科学123", "Data科學321");       // 3 bytes, 3 runes
    append("🙂🌍🚀", "🙂🌎✨");                 // 5 bytes, 2 runes
    append("💖", "💗");                         // 4-byte emojis: Different hearts; 1 bytes, 1 runes diff.
    append("مرحبا بالعالم", "مرحبا يا عالم");   // "Hello World" vs "Welcome to the World" ?; 3 bytes, 2 runes diff.
    append("𠜎 𠜱 𠝹 𠱓", "𠜎𠜱𠝹𠱓");          // Ancient Chinese characters, no spaces vs spaces; 3 bytes, 3 runes
    append("München", "Muenchen");              // German name with umlaut vs. its transcription; 2 bytes, 2 runes
    append("façade", "facade");                 // "ç" represented as "c" with cedilla vs. plain "c"; 2 bytes, 1 runes
    append("こんにちは世界", "こんばんは世界"); // "Good morning world" vs "Good evening world"; 3 bytes, 2 runes
    append("👩‍👩‍👧‍👦", "👨‍👩‍👧‍👦"); // Different family emojis; 1 bytes, 1 runes

    // ~20 characters; two similar integral expressions that differ in the upper limit.
    append("∫₀¹ x² dx = 1/3", "∫₀² x² dx = 8/3");

    // ~50 characters; typography test with box-drawing, quote style, currency symbol, dash type, and case differences.
    append("╔══╦══╗ • ‘single’ and “double” quotes, € 14.95 — OK",
           "╔══╦══╗ • ‘single’ and «double» quotes, $ 14.95 – ok");

    // ~100 characters in one string combining Armenian, Georgian, and Greek:
    append("Երևան, თბილისი, και Αθήνα – 3 մայրքաղաքներ: Բարի գալուստ, მოგესალმებით, και Καλώς ορίσατε!",
           "Երևան, თბილისი, και Αθήνα – երեք մայրքաղաքներ: բարև, სტუმრები, και Καλώς ήρθατε!");

    // ~200 characters in ASCII English, Traditional Chinese, and Russian, describing their capitals.
    append("London, the iconic capital of the United Kingdom, seamlessly blends centuries-old traditions with bold "
           "modernity;"
           "倫敦作為英國的標誌性首都，其歷史沉澱與當代創新彼此交融，展現獨特風範;"
           "Лондон, столица Великобритании, объединяет древние традиции с динамичной современностью, "
           "offering a rich tapestry of cultural heritage and visionary progress.", // First string ends here ;)
           "London, the renowned capital of the UK, fuses its rich historical legacy with a spirit of modern "
           "innovation;"
           "倫敦，作為英國的著名首都，以悠久歷史與現代創意相互融合，呈現獨特都市風貌;"
           "Лондон – известная столица Великобритании, где древность встречается с современной энергией, "
           "creating an inspiring environment for cultural exploration and future development.");

    // ~300 characters; a complex variant with translations and visible regions of Korean, Japanese, Chinese
    // (traditional and simplified), German, French, Spanish.
    append("An epic voyage through multicultural realms: "
           "In a city where ancient traditions fuse with modern innovation, dynamic energy permeates every street. "
           "서울의 번화한 거리에선 전통과 현대가 어우러져 감동을 주며, 東京では伝統美と未来の夢が共鳴する。在這裡, "
           "傳統文化與現代科技和諧並存, 而这里, 传统文化与现代科技交织创新; "
           "Deutschland zeigt eine reiche Geschichte, "
           "la France révèle une élégance subtile, "
           "y España irradia pasión y color.", // First string ends here ;)
           "An epic journey through diverse cultures: "
           "In a town where old traditions fuse with innovation, energy permeates every historic street. "
           "서울의 번화한 거리는 전통과 현대가 어울려 독특한 풍경을 이루며, "
           "東京では伝統美と未来への展望が響き合う。在這裡, 傳統與現代科技融合無間, 而这里, 传统与现代科技紧密相连; "
           "Deutschland offenbart eine stolze Geschichte, "
           "la France incarne une élégance fine, "
           "y España resplandece con pasión y vivacidad.");

    // First check with a batch-size of 1
    using score_t = score_type_;
    unified_vector<score_t> results_base(1), results_simd(1);
    arrow_strings_tape_t first_tape, second_tape;
    bool contains_missing_in_any_case = false;
    constexpr score_t signaling_score = std::numeric_limits<score_t>::max();

    // Old C-style for-loops are much more debuggable than range-based loops!
    for (std::size_t pair_idx = 0; pair_idx != test_cases.size(); ++pair_idx) {
        auto const &first = test_cases[pair_idx].first;
        auto const &second = test_cases[pair_idx].second;

        // Check if the input strings fit into our allowed characters set
        if (!allowed_chars.empty()) {
            bool contains_missing = false;
            for (auto c : first) contains_missing |= allowed_chars.find(c) == std::string_view::npos;
            for (auto c : second) contains_missing |= allowed_chars.find(c) == std::string_view::npos;
            contains_missing_in_any_case |= contains_missing;
            if (contains_missing) continue;
        }

        // Reset the tapes and results
        results_base[0] = signaling_score, results_simd[0] = signaling_score;
        first_tape.try_assign(&first, &first + 1);
        second_tape.try_assign(&second, &second + 1);

        // Compute with both backends
        arrow_strings_view_t first_view = first_tape.view();
        arrow_strings_view_t second_view = second_tape.view();
        score_t *results_base_ptr = results_base.data();
        score_t *results_simd_ptr = results_simd.data();
        status_t status_base = base_operator(first_view, second_view, results_base_ptr);
        status_t status_simd = simd_operator(first_view, second_view, results_simd_ptr, extra_args...);
        _sz_assert(status_base == status_t::success_k);
        _sz_assert(status_simd == status_t::success_k);
        if (results_base[0] != results_simd[0])
            edit_distance_log_mismatch(first, second, results_base[0], results_simd[0]);
    }

    // Unzip the test cases into two separate tapes and perform batch processing
    if (!contains_missing_in_any_case) {
        results_base.resize(test_cases.size(), signaling_score);
        results_simd.resize(test_cases.size(), signaling_score);
        first_tape.reset();
        second_tape.reset();
        for (auto [first, second] : test_cases) {
            _sz_assert(first_tape.try_append({first.data(), first.size()}) == status_t::success_k);
            _sz_assert(second_tape.try_append({second.data(), second.size()}) == status_t::success_k);
        }

        // Compute with both backends
        status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data(), extra_args...);
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
};

void randomize_strings(fuzzy_config_t config, std::vector<std::string> &array, arrow_strings_tape_t &tape,
                       bool unique = false) {
    array.resize(config.batch_size);

    std::uniform_int_distribution<std::size_t> length_distribution(config.min_string_length, config.max_string_length);
    for (std::size_t i = 0; i != config.batch_size; ++i) {
        std::size_t length = length_distribution(global_random_generator());
        array[i] = random_string(length, config.alphabet.data(), config.alphabet.size());
    }

    if (unique) {
        std::sort(array.begin(), array.end());
        auto last = std::unique(array.begin(), array.end());
        array.erase(last, array.end());
    }

    // Convert to a GPU-friendly layout
    status_t status = tape.try_assign(array.data(), array.data() + array.size());
    _sz_assert(status == status_t::success_k);
}

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads
 *          on a synthetic @b randomly-generated set of strings from a given @p alphabet.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_similarity_scores_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                  fuzzy_config_t config = {}, std::size_t iterations = 10,
                                  extra_args_ &&...extra_args) {

    unified_vector<score_type_> results_base(config.batch_size), results_simd(config.batch_size);
    std::vector<std::string> first_array, second_array;
    arrow_strings_tape_t first_tape, second_tape;

    // Generate some random strings, using a small alphabet
    for (std::size_t iteration_idx = 0; iteration_idx < iterations; ++iteration_idx) {
        randomize_strings(config, first_array, first_tape);
        randomize_strings(config, second_array, second_tape);

        // Compute with both backends
        status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data(), extra_args...);
        _sz_assert(status_base == status_t::success_k);
        _sz_assert(status_simd == status_t::success_k);

        // Individually log the failed results
        for (std::size_t i = 0; i != config.batch_size; ++i) {
            if (results_base[i] == results_simd[i]) continue;
            edit_distance_log_mismatch(first_array[i], second_array[i], results_base[i], results_simd[i]);
        }
    }
}

template <typename score_type_, typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_similarity_scores_fixed_and_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                            std::string_view allowed_chars = {}, fuzzy_config_t config = {},
                                            extra_args_ &&...extra_args) {
    test_similarity_scores_fixed<score_type_>(base_operator, simd_operator, allowed_chars, extra_args...);
    test_similarity_scores_fuzzy<score_type_>(base_operator, simd_operator, config, 1, extra_args...);
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
    {
        constexpr error_t unary_match_score = 1;
        constexpr error_t unary_mismatch_score = 0;
        constexpr error_t unary_gap_score = 0;
        error_matrix_t substituter_unary = error_matrix_t::diagonal(unary_match_score, unary_mismatch_score);
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

    // Let's define some weird scoring schemes for Levenshtein-like distance, that are not unary:
    constexpr linear_gap_costs_t weird_linear {3};
    constexpr affine_gap_costs_t weird_affine {4, 2};
    constexpr uniform_substitution_costs_t weird_uniform {1, 3};

    // Single-threaded serial Levenshtein distance implementation
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_baselines_t {},                    //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {});

    // Multi-threaded parallel Levenshtein distance implementation
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_baselines_t {},                    //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {});

    // Single-threaded serial Levenshtein distance implementation with weird linear costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(         //
        levenshtein_baselines_t {weird_uniform, weird_linear}, //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear});

    // Multi-threaded parallel Levenshtein distance implementation with weird linear costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(         //
        levenshtein_baselines_t {weird_uniform, weird_linear}, //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear});

    // Single-threaded serial Levenshtein distance implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(         //
        levenshtein_baselines_t {weird_uniform, weird_affine}, //
        levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine});

    // Multi-threaded parallel Levenshtein distance implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(         //
        levenshtein_baselines_t {weird_uniform, weird_affine}, //
        levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine});

    // Now let's take non-unary substitution costs, like BLOSUM62
    constexpr linear_gap_costs_t blosum62_linear_cost {-4};
    constexpr affine_gap_costs_t blosum62_affine_cost {-4, -1};
    error_mat_t blosum62_mat = error_costs_26x26ascii_t::blosum62();
    error_matrix_t blosum62_matrix = blosum62_mat.decompressed();

    // Single-threaded serial NW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Multi-threaded parallel NW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Single-threaded serial SW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Multi-threaded parallel SW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Single-threaded serial NW implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost});

    // Multi-threaded parallel NW implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost});

    // Single-threaded serial SW implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        smith_waterman_scores<char, error_matrix_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost});

    // Multi-threaded parallel SW implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        smith_waterman_scores<char, error_matrix_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost});

#if SZ_USE_ICE
    // Ice Lake Levenshtein distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                 //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_si_k> {});

    // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird linear costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}, //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_si_k> {weird_uniform, weird_linear});

    // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}, //
        levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_caps_si_k> {weird_uniform, weird_affine});

    // Ice Lake Levenshtein UTF8 distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                      //
        levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, //
        levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_caps_si_k> {});

    // Ice Lake Levenshtein UTF8 distance against Multi-threaded on CPU with weird linear costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear},
        levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_caps_si_k> {weird_uniform, weird_linear});

    // Ice Lake Needleman-Wunsch distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_caps_si_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Ice Lake Smith-Waterman distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_caps_si_k> {blosum62_matrix,
                                                                                                 blosum62_linear_cost});

#endif

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs = *gpu_specs();
#endif

#if SZ_USE_CUDA
    // CUDA Levenshtein distance against Multi-threaded on CPU with weird linear costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}, //
        levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {weird_uniform, weird_linear}, {}, {},
        first_gpu_specs);

    // CUDA Levenshtein distance against Multi-threaded on CPU with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}, //
        levenshtein_distances<char, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {weird_uniform, weird_affine}, {}, {},
        first_gpu_specs);
#endif

#if SZ_USE_KEPLER
    // CUDA Levenshtein distance on Kepler against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}, //
        levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_caps_ck_k> {weird_uniform, weird_linear}, {}, {},
        first_gpu_specs);
#endif

#if SZ_USE_CUDA
    // CUDA Needleman-Wunsch score against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_linear_cost},
        {}, {}, first_gpu_specs);

    // CUDA Needleman-Wunsch score against Multi-threaded on CPU with affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_affine_cost},
        {}, {}, first_gpu_specs);

    // CUDA Smith-Waterman score against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        smith_waterman_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {blosum62_matrix,
                                                                                                  blosum62_linear_cost},
        {}, {}, first_gpu_specs);

    // CUDA Smith-Waterman score against Multi-threaded on CPU with affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        smith_waterman_scores<char, error_matrix_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}, //
        smith_waterman_scores<char, error_matrix_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {blosum62_matrix,
                                                                                                  blosum62_affine_cost},
        {}, {}, first_gpu_specs);
#endif

#if SZ_USE_HOPPER
    // CUDA Needleman-Wunsch score on Hopper against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_linear_cost},
        {}, {}, first_gpu_specs);

    // CUDA Needleman-Wunsch score on Hopper against Multi-threaded on CPU with affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_affine_cost},
        {}, {}, first_gpu_specs);

    // CUDA Smith-Waterman score on Hopper against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        smith_waterman_scores<char, error_matrix_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k> {blosum62_matrix,
                                                                                                  blosum62_linear_cost},
        {}, {}, first_gpu_specs);

    // CUDA Smith-Waterman score on Hopper against Multi-threaded on CPU with affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        smith_waterman_scores<char, error_matrix_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}, //
        smith_waterman_scores<char, error_matrix_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k> {blosum62_matrix,
                                                                                                  blosum62_affine_cost},
        {}, {}, first_gpu_specs);

#endif
}

/**
 *  @brief  Many GPU algorithms depend on effective use of shared memory and scheduling its allocation for
 *          long inputs or very large batches isn't trivial.
 */
void test_similarity_scores_memory_usage() {

    std::vector<fuzzy_config_t> experiments = {
        // Single string pair of same length:
        {.batch_size = 1, .min_string_length = 128, .max_string_length = 128},
        {.batch_size = 1, .min_string_length = 512, .max_string_length = 512},
        {.batch_size = 1, .min_string_length = 2048, .max_string_length = 2048},
        {.batch_size = 1, .min_string_length = 8192, .max_string_length = 8192},
        {.batch_size = 1, .min_string_length = 32768, .max_string_length = 32768},
        {.batch_size = 1, .min_string_length = 131072, .max_string_length = 131072},
        // Two strings of a same length:
        {.batch_size = 2, .min_string_length = 128, .max_string_length = 128},
        {.batch_size = 2, .min_string_length = 512, .max_string_length = 512},
        {.batch_size = 2, .min_string_length = 2048, .max_string_length = 2048},
        {.batch_size = 2, .min_string_length = 8192, .max_string_length = 8192},
        {.batch_size = 2, .min_string_length = 32768, .max_string_length = 32768},
        {.batch_size = 2, .min_string_length = 131072, .max_string_length = 131072},
        // Ten strings of random lengths:
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 128},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 512},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 2048},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 8192},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 32768},
        {.batch_size = 10, .min_string_length = 1, .max_string_length = 131072},
    };

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs = *gpu_specs();
#endif

    // Let's define some weird scoring schemes for Levenshtein-like distance, that are not unary:
    constexpr linear_gap_costs_t weird_linear {3};
    constexpr affine_gap_costs_t weird_affine {4, 2};
    constexpr uniform_substitution_costs_t weird_uniform {1, 3};

    // Progress until something fails
    for (fuzzy_config_t const &experiment : experiments) {
        std::printf("Testing with batch size %zu, min length %zu, max length %zu\n", experiment.batch_size,
                    experiment.min_string_length, experiment.max_string_length);

        // Multi-threaded serial Levenshtein distance implementation
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_baselines_t {},          //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, experiment, 1);

        // Multi-threaded serial Levenshtein distance implementation with weird linear costs
        test_similarity_scores_fuzzy<sz_size_t>(                   //
            levenshtein_baselines_t {weird_uniform, weird_linear}, //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear},
            experiment, 1);

        // Multi-threaded serial Levenshtein distance implementation with weird affine costs
        test_similarity_scores_fuzzy<sz_size_t>(                   //
            levenshtein_baselines_t {weird_uniform, weird_affine}, //
            levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine},
            experiment, 1);

#if SZ_USE_ICE
        // Ice Lake Levenshtein distance against Multi-threaded on CPU
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {},
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_si_k> {}, experiment, 1);

        // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird linear costs
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear},
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_si_k> {weird_uniform, weird_linear},
            experiment, 1);

        // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird affine costs
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine},
            levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_caps_si_k> {weird_uniform, weird_affine},
            experiment, 1);
#endif

#if SZ_USE_CUDA
        // CUDA Levenshtein distance against Multi-threaded on CPU
        test_similarity_scores_fuzzy<sz_size_t>(                                           //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, //
            levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {}, experiment, 10,
            first_gpu_specs);
#endif

#if SZ_USE_KEPLER
        // CUDA Levenshtein distance on Kepler against Multi-threaded on CPU
        test_similarity_scores_fuzzy<sz_size_t>(                                           //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, //
            levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_caps_ck_k> {}, experiment, 10,
            first_gpu_specs);
#endif
    }
}

struct find_many_baselines_t {
    using match_t = find_many_match_t;

    arrow_strings_tape_t needles_;

    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles) noexcept {
        return needles_.try_assign(needles.begin(), needles.end());
    }

    void reset() noexcept { needles_.reset(); }

    template <typename haystack_type_, typename needles_type_, typename match_callback_type_>
    bool one_haystack(haystack_type_ const &haystack, needles_type_ const &needles,
                      match_callback_type_ &&callback) const noexcept {

        // A wise man once said, `omp parallel for collapse(2) schedule(dynamic, 1)`...
        // But the compiler wasn't listening, and won't compile the cancellation point!
        // So we resort to a much less intricate solution:
        // - Manually slice the data per thread,
        // - Keep one atomic variable to signal cancellation,
        // - Use absolutely minimal OpenMP functionality just to assign N slices to N threads.
        std::atomic<bool> aborted {false};
        std::size_t const haystack_size = haystack.size();
        std::size_t const threads_count = std::thread::hardware_concurrency();
        std::size_t const start_offsets_per_thread = divide_round_up(haystack_size, threads_count);

#pragma omp parallel for schedule(static, 1)
        for (std::size_t thread_index = 0; thread_index != threads_count; ++thread_index) {
            std::size_t const start_offset = std::min(thread_index * start_offsets_per_thread, haystack_size);
            std::size_t const end_offset = std::min(start_offset + start_offsets_per_thread, haystack_size);

            // Check for matches in the current slice
            for (std::size_t match_offset = start_offset;
                 match_offset != end_offset && !aborted.load(std::memory_order_relaxed); ++match_offset) {
                for (std::size_t needle_index = 0; needle_index != needles.size(); ++needle_index) {
                    auto const &needle = needles[needle_index];
                    if (match_offset + needle.size() > haystack_size) continue;
                    auto const same = std::memcmp(haystack.data() + match_offset, needle.data(), needle.size()) == 0;
                    if (!same) continue;

                    // Create a match object
                    match_t match;
                    match.haystack_index = 0;
                    match.needle_index = needle_index;
                    match.haystack = {reinterpret_cast<byte_t const *>(haystack.data()), haystack.size()};
                    match.needle = {reinterpret_cast<byte_t const *>(haystack.data() + match_offset), needle.size()};
                    if (!callback(match)) {
                        aborted.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            }
        }

        return !aborted.load(std::memory_order_relaxed);
    }

    template <typename haystacks_type_, typename needles_type_, typename match_callback_type_>
    void all_pairs(haystacks_type_ &&haystacks, needles_type_ &&needles,
                   match_callback_type_ &&callback) const noexcept {

        for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
            auto const &haystack = haystacks[haystack_index];
            if (!one_haystack(haystack, needles, [&](match_t match) noexcept {
                    match.haystack_index = haystack_index;
                    return callback(match);
                }))
                return;
        }
    }

    template <typename haystacks_type_>
    status_t try_count(haystacks_type_ &&haystacks, span<size_t> counts) const noexcept {
        for (size_t &count : counts) count = 0;
        all_pairs(haystacks, needles_, [&](match_t const &match) noexcept {
            std::atomic_ref<size_t> count(counts[match.haystack_index]);
            count.fetch_add(1, std::memory_order_relaxed);
            return true;
        });
        return status_t::success_k;
    }

    template <typename haystacks_type_, typename output_matches_type_>
    status_t try_find(haystacks_type_ &&haystacks, span<size_t const> counts,
                      output_matches_type_ &&matches) const noexcept {

        sz_unused(counts);
        std::atomic<size_t> count_found {0};
        std::size_t const count_allowed {matches.size()};
        all_pairs(haystacks, needles_, [&](match_t const &match) noexcept {
            size_t match_index = count_found.fetch_add(1, std::memory_order_relaxed);
            matches[match_index] = match;
            return match_index < count_allowed;
        });
        return status_t::success_k;
    }
};

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads
 *          on a @b fixed set of different representative ASCII and UTF-8 strings.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many_fixed(base_operator_ &&base_operator, simd_operator_ &&simd_operator, extra_args_ &&...extra_args) {

    std::vector<std::string> haystacks, needles;

    // Some vary basic variants:
    needles.emplace_back("his");
    needles.emplace_back("is");
    needles.emplace_back("she");
    needles.emplace_back("her");

    needles.emplace_back("école"), needles.emplace_back("école");                   // decomposed
    needles.emplace_back("Schön"), needles.emplace_back("Scho\u0308n");             // combining diaeresis
    needles.emplace_back("naïve"), needles.emplace_back("naive");                   // stripped diaeresis
    needles.emplace_back("façade"), needles.emplace_back("facade");                 // no cedilla
    needles.emplace_back("office"), needles.emplace_back("ofﬁce");                  // “fi” ligature
    needles.emplace_back("Straße"), needles.emplace_back("Strasse");                // ß vs ss
    needles.emplace_back("ABBA"), needles.emplace_back("\u0410\u0412\u0412\u0410"); // Latin vs Cyrillic
    needles.emplace_back("中国"), needles.emplace_back("中國");                     // simplified vs traditional
    needles.emplace_back("🙂"), needles.emplace_back("☺️");                          // emoji variants
    needles.emplace_back("€100"), needles.emplace_back("EUR 100");                  // currency symbol vs abbreviation

    // Haystacks should contain arbitrary strings including those needles
    // in different positions, potentially interleaving
    haystacks.emplace_back("That is a test string"); // ? "only "is"
    haystacks.emplace_back("This is a test string"); // ? "his", 2x "is"
    haystacks.emplace_back("ahishers");              // textbook example
    haystacks.emplace_back("hishishersherishis");    // heavy overlap, prefix & suffix collisions
    haystacks.emplace_back("si siht si a tset gnirts; reh ton si ehs, tub sih ti si."); // no real matches
    haystacks.emplace_back("his\0is\r\nshe\0her");                                      // null-included

    // ~260 chars – dense English with overlapping words (“his”, “is”, “she”, “her”)
    haystacks.emplace_back(R"(
    In this historic thesis, the historian highlights his findings: this is the synthesis of data.
    She examined the theory, he shared her methodology. In this chapter, he lists his equipment:
    microscope, test kit, sensor. It is here that she erred: misalignment arises.
    )");

    // ~320 chars – multilingual snippet with needles in Latin, Arabic, Chinese, English
    haystacks.emplace_back(R"(
    The conference in 北京 attracted researchers from across the globe. His presentation “AI in Healthcare”
    was a hit—she received awards. الباحثون استعرضوا الأبحاث، واستشارت her colleagues. 这是一次重要的会议。
    She said: “This is only the beginning.” In her report, his name appears seventeen times.
    )");
    using match_t = find_many_match_t;

    // First check with a batch-size of 1
    unified_vector<size_t> counts_base(1), counts_simd(1);
    unified_vector<match_t> matches_base(1), matches_simd(1);
    arrow_strings_tape_t haystacks_tape, needles_tape;
    needles_tape.try_assign(needles.data(), needles.data() + needles.size());

    // Construct the matchers
    status_t status_base = base_operator.try_build(needles_tape.view());
    status_t status_simd = simd_operator.try_build(needles_tape.view());
    _sz_assert(status_base == status_t::success_k);
    _sz_assert(status_simd == status_t::success_k);

    // Old C-style for-loops are much more debuggable than range-based loops!
    for (std::size_t haystack_idx = 0; haystack_idx != haystacks.size(); ++haystack_idx) {
        auto const &haystack = haystacks[haystack_idx];

        // Reset the tapes and results
        counts_base[0] = 0, counts_simd[0] = 0;
        matches_base.clear(), matches_simd.clear();
        haystacks_tape.try_assign(&haystack, &haystack + 1);

        // Count with both backends
        span<size_t> counts_base_span {counts_base.data(), counts_base.size()};
        span<size_t> counts_simd_span {counts_simd.data(), counts_simd.size()};
        status_t status_count_base = base_operator.try_count(haystacks_tape, counts_base_span);
        status_t status_count_simd = simd_operator.try_count(haystacks_tape, counts_simd_span, extra_args...);
        _sz_assert(status_count_base == status_t::success_k);
        _sz_assert(status_count_simd == status_t::success_k);
        _sz_assert(counts_base[0] == counts_simd[0]);

        // Check the matches themselves
        matches_base.resize(std::accumulate(counts_base.begin(), counts_base.end(), 0));
        matches_simd.resize(std::accumulate(counts_simd.begin(), counts_simd.end(), 0));
        status_t status_matched_base = base_operator.try_find(haystacks_tape, counts_base_span, matches_base);
        status_t status_matched_simd =
            simd_operator.try_find(haystacks_tape, counts_simd_span, matches_simd, extra_args...);
        _sz_assert(status_matched_base == status_t::success_k);
        _sz_assert(status_matched_simd == status_t::success_k);

        // Check the contents and order of the matches
        std::sort(matches_base.begin(), matches_base.end(), match_t::less_globally);
        std::sort(matches_simd.begin(), matches_simd.end(), match_t::less_globally);
        for (std::size_t i = 0; i != matches_base.size(); ++i) {
            _sz_assert(matches_base[i].haystack.data() == matches_simd[i].haystack.data());
            _sz_assert(matches_base[i].needle.data() == matches_simd[i].needle.data());
            _sz_assert(matches_base[i].needle_index == matches_simd[i].needle_index);
        }
    }

    // Now test all the haystacks simultaneously
    {
        haystacks_tape.try_assign(haystacks.data(), haystacks.data() + haystacks.size());
        counts_base.resize(haystacks.size());
        counts_simd.resize(haystacks.size());

        // Count with both backends and compare all of the bounds
        span<size_t> counts_base_span {counts_base.data(), counts_base.size()};
        span<size_t> counts_simd_span {counts_simd.data(), counts_simd.size()};
        status_t status_count_base = base_operator.try_count(haystacks_tape, counts_base_span);
        status_t status_count_simd = simd_operator.try_count(haystacks_tape, counts_simd_span, extra_args...);
        _sz_assert(status_count_base == status_t::success_k);
        _sz_assert(status_count_simd == status_t::success_k);
        _sz_assert(std::equal(counts_base.begin(), counts_base.end(), counts_simd.begin()));

        // Check the matches themselves
        matches_base.resize(std::accumulate(counts_base.begin(), counts_base.end(), 0));
        matches_simd.resize(std::accumulate(counts_simd.begin(), counts_simd.end(), 0));
        status_t status_matched_base = base_operator.try_find(haystacks_tape, counts_base_span, matches_base);
        status_t status_matched_simd =
            simd_operator.try_find(haystacks_tape, counts_simd_span, matches_simd, extra_args...);
        _sz_assert(status_matched_base == status_t::success_k);
        _sz_assert(status_matched_simd == status_t::success_k);

        // Check the contents and order of the matches
        std::sort(matches_base.begin(), matches_base.end(), match_t::less_globally);
        std::sort(matches_simd.begin(), matches_simd.end(), match_t::less_globally);
        for (std::size_t i = 0; i != matches_base.size(); ++i) {
            _sz_assert(matches_base[i].haystack.data() == matches_simd[i].haystack.data());
            _sz_assert(matches_base[i].needle.data() == matches_simd[i].needle.data());
            _sz_assert(matches_base[i].needle_index == matches_simd[i].needle_index);
        }
    }
}

/**
 *  @brief Fuzzy test for multi-pattern exact search algorithms using randomly-generated haystacks and needles.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                    arrow_strings_tape_t const &haystacks_tape, arrow_strings_tape_t const &needles_tape,
                    extra_args_ &&...extra_args) {

    using match_t = find_many_match_t;
    unified_vector<match_t> results_base, results_simd;
    unified_vector<size_t> counts_base, counts_simd;

    counts_base.resize(haystacks_tape.size());
    counts_simd.resize(haystacks_tape.size());

    // Build the matchers
    _sz_assert(base_operator.try_build(needles_tape.view()) == status_t::success_k);
    _sz_assert(simd_operator.try_build(needles_tape.view()) == status_t::success_k);

    // Count the number of matches with both backends
    span<size_t> counts_base_span {counts_base.data(), counts_base.size()};
    span<size_t> counts_simd_span {counts_simd.data(), counts_simd.size()};
    status_t status_count_base = base_operator.try_count(haystacks_tape.view(), counts_base_span);
    status_t status_count_simd = simd_operator.try_count(haystacks_tape.view(), counts_simd_span, extra_args...);
    _sz_assert(status_count_base == status_t::success_k);
    _sz_assert(status_count_simd == status_t::success_k);
    size_t total_count_base = std::accumulate(counts_base.begin(), counts_base.end(), 0);
    size_t total_count_simd = std::accumulate(counts_simd.begin(), counts_simd.end(), 0);
    _sz_assert(total_count_base == total_count_simd);
    _sz_assert(std::equal(counts_base.begin(), counts_base.end(), counts_simd.begin()));

    // Compute with both backends
    results_base.resize(total_count_base);
    results_simd.resize(total_count_simd);
    size_t count_base = 0, count_simd = 0;
    status_t status_base = base_operator.try_find(haystacks_tape.view(), counts_base_span, results_base);
    status_t status_simd = simd_operator.try_find(haystacks_tape.view(), counts_simd_span, results_simd, extra_args...);
    _sz_assert(status_base == status_t::success_k);
    _sz_assert(status_simd == status_t::success_k);
    _sz_assert(count_base == count_simd);

    // Individually log the failed results
    std::sort(results_base.begin(), results_base.end(), match_t::less_globally);
    std::sort(results_simd.begin(), results_simd.end(), match_t::less_globally);
    for (std::size_t i = 0; i != results_base.size(); ++i) {
        _sz_assert(results_base[i].haystack_index == results_simd[i].haystack_index);
        _sz_assert(results_base[i].needle_index == results_simd[i].needle_index);
        _sz_assert(results_base[i].needle.data() == results_simd[i].needle.data());
    }

    base_operator.reset();
    simd_operator.reset();
}

/**
 *  @brief Fuzzy test for multi-pattern exact search algorithms using randomly-generated haystacks and needles.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                          fuzzy_config_t needles_config = {}, fuzzy_config_t haystacks_config = {},
                          std::size_t iterations = 10, extra_args_ &&...extra_args) {

    std::vector<std::string> haystacks_array, needles_array;
    arrow_strings_tape_t haystacks_tape, needles_tape;

    // Generate some random strings, using a small alphabet
    for (std::size_t iteration_idx = 0; iteration_idx < iterations; ++iteration_idx) {
        randomize_strings(haystacks_config, haystacks_array, haystacks_tape);
        randomize_strings(needles_config, needles_array, needles_tape, true);
        test_find_many(base_operator, simd_operator, haystacks_tape, needles_tape, extra_args...);
    }
}

/**
 *  @brief  Fuzzy test for multi-pattern exact search algorithms using randomly-generated haystacks,
 *          and using incrementally longer potentially-overlapping substrings as needles.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many_prefixes(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                             fuzzy_config_t haystacks_config, std::size_t needle_length_limit,
                             std::size_t iterations = 10, extra_args_ &&...extra_args) {

    std::vector<std::string> haystacks_array;
    std::vector<std::string_view> needles_array;
    arrow_strings_tape_t haystacks_tape, needles_tape;

    for (std::size_t iteration_idx = 0; iteration_idx < iterations; ++iteration_idx) {
        randomize_strings(haystacks_config, haystacks_array, haystacks_tape);

        // Pick various substrings as needles from the first haystack
        needles_array.resize(std::min(haystacks_array[0].size(), needle_length_limit));
        for (std::size_t i = 0; i != needles_array.size(); ++i)
            needles_array[i] = std::string_view(haystacks_array[0]).substr(0, i + 1);
        needles_tape.try_assign(needles_array.data(), needles_array.data() + needles_array.size());

        test_find_many(base_operator, simd_operator, haystacks_tape, needles_tape, extra_args...);
    }
}

/**
 *  @brief  Tests the multi-pattern exact substring search algorithm
 *          against a baseline implementation for predefined and random inputs.
 */
void test_find_many_equivalence() {

    cpu_specs_t default_cpu_specs;
    fuzzy_config_t needles_short_config, needles_long_config, haystacks_config;
    haystacks_config.batch_size = default_cpu_specs.cores_total() * 4;
    haystacks_config.max_string_length = default_cpu_specs.l3_bytes;

    needles_short_config.min_string_length = 1;
    needles_short_config.max_string_length = 4;
    needles_short_config.batch_size =
        std::pow(needles_short_config.alphabet.size(), needles_short_config.max_string_length);

    needles_long_config.min_string_length = 3;
    needles_long_config.max_string_length = 6;
    needles_long_config.batch_size =
        std::pow(needles_long_config.alphabet.size(), needles_long_config.max_string_length);

    // Let's reuse a thread-pool to amortize the cost of spawning threads.
    fork_union_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) throw std::runtime_error("Failed to spawn thread pool.");
    static_assert(executor_like<fork_union_t>);

    // Single-threaded serial Aho-Corasick implementation
    test_find_many_fixed(find_many_baselines_t {}, find_many_u32_serial_t {});

    // Multi-threaded parallel Aho-Corasick implementation
    test_find_many_fixed(find_many_baselines_t {}, find_many_u32_parallel_t {}, pool);

    // Fuzzy tests with random inputs
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_serial_t {}, needles_short_config, haystacks_config,
                         1);
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_serial_t {}, needles_long_config, haystacks_config, 1);
    test_find_many_prefixes(find_many_baselines_t {}, find_many_u32_serial_t {}, haystacks_config, 1024, 1);

    // Fuzzy tests with random inputs for multi-threaded CPU backend
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_parallel_t {}, needles_short_config, haystacks_config,
                         10, pool);
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_parallel_t {}, needles_long_config, haystacks_config,
                         10, pool);
    test_find_many_prefixes(find_many_baselines_t {}, find_many_u32_parallel_t {}, haystacks_config, 1024, 10, pool);
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
