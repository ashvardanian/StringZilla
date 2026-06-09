/**
 *  @brief Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file scripts/test_similarities.cuh
 *  @author Ash Vardanian
 */
#include "stringzillas/similarities.hpp"

#if SZ_USE_CUDA
#include "stringzillas/similarities.cuh"
#endif

#if !SZ_IS_CPP17_
#error "This test requires C++17 or later."
#endif

#include "test_stringzilla.hpp" // `arrow_strings_view_t`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

// StringZillas library symbols available on every backend:
using ashvardanian::stringzillas::affine_gap_costs_t;
using ashvardanian::stringzillas::error_costs_32x32_t;
using ashvardanian::stringzillas::levenshtein_distances;
using ashvardanian::stringzillas::levenshtein_distances_utf8;
using ashvardanian::stringzillas::linear_gap_costs_t;
using ashvardanian::stringzillas::malloc_t;
using ashvardanian::stringzillas::needleman_wunsch_scores;
using ashvardanian::stringzillas::smith_waterman_scores;
using ashvardanian::stringzillas::uniform_substitution_costs_t;

// StringZillas library symbols provided only by the CUDA backend:
#if SZ_USE_CUDA
using ashvardanian::stringzillas::cuda_executor_t;
using ashvardanian::stringzillas::gpu_specs_fetch;
using ashvardanian::stringzillas::ualloc_t;
#endif

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
template <typename substituter_type_>
inline std::ptrdiff_t needleman_wunsch_baseline(                        //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    substituter_type_ substitution_cost_for, error_cost_t gap_cost) noexcept(false) {

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
template <typename substituter_type_>
inline std::ptrdiff_t smith_waterman_baseline(char const *s1, std::size_t len1, char const *s2, std::size_t len2,
                                              substituter_type_ substitution_cost_for,
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
            std::size_t if_insertion = std::min<std::size_t>(row[j - 1] + gap_opening_cost,
                                                             row_inserts[j - 1] + gap_extension_cost);
            std::size_t if_deletion = std::min<std::size_t>(last_row[j] + gap_opening_cost,
                                                            last_deletes_row[j] + gap_extension_cost);
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
template <typename substituter_type_>
inline std::ptrdiff_t needleman_wunsch_gotoh_baseline(                  //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    substituter_type_ substitution_cost_for,                            //
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
            std::ptrdiff_t if_insertion = std::max(row[j - 1] + gap_opening_cost,
                                                   row_inserts[j - 1] + gap_extension_cost);
            std::ptrdiff_t if_deletion = std::max(last_row[j] + gap_opening_cost,
                                                  last_deletes_row[j] + gap_extension_cost);
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
template <typename substituter_type_>
inline std::ptrdiff_t smith_waterman_gotoh_baseline(                    //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    substituter_type_ substitution_cost_for,                            //
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
            std::ptrdiff_t if_insertion = std::max(row[j - 1] + gap_opening_cost,
                                                   row_inserts[j - 1] + gap_extension_cost);
            std::ptrdiff_t if_deletion = std::max(last_row[j] + gap_opening_cost,
                                                  last_deletes_row[j] + gap_extension_cost);
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
        sz_assert_(first.size() == second.size());
#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = gap_opening_cost == gap_extension_cost
                             ? levenshtein_baseline(first[i].data(), first[i].size(),   //
                                                    second[i].data(), second[i].size(), //
                                                    substitution_costs.match, substitution_costs.mismatch,
                                                    gap_opening_cost)
                             : levenshtein_gotoh_baseline(first[i].data(), first[i].size(),   //
                                                          second[i].data(), second[i].size(), //
                                                          substitution_costs.match, substitution_costs.mismatch,
                                                          gap_opening_cost, gap_extension_cost);
        return status_t::success_k;
    }
};

struct needleman_wunsch_baselines_t {

    error_costs_32x32_t substitution_costs {};
    error_cost_t gap_opening_cost = -1;
    error_cost_t gap_extension_cost = -1;

    needleman_wunsch_baselines_t() = default;
    needleman_wunsch_baselines_t(error_costs_32x32_t subs, linear_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open_or_extend), gap_extension_cost(gap.open_or_extend) {}
    needleman_wunsch_baselines_t(error_costs_32x32_t subs, affine_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open), gap_extension_cost(gap.extend) {}

    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, sz_ssize_t *results) const {
        sz_assert_(first.size() == second.size());

#pragma omp parallel for
        for (std::size_t i = 0; i != first.size(); ++i)
            results[i] = gap_opening_cost == gap_extension_cost
                             ? needleman_wunsch_baseline(first[i].data(), first[i].size(),   //
                                                         second[i].data(), second[i].size(), //
                                                         substitution_costs, gap_opening_cost)
                             : needleman_wunsch_gotoh_baseline(first[i].data(), first[i].size(),   //
                                                               second[i].data(), second[i].size(), //
                                                               substitution_costs, gap_opening_cost,
                                                               gap_extension_cost);
        return status_t::success_k;
    }
};

struct smith_waterman_baselines_t {

    error_costs_32x32_t substitution_costs {};
    error_cost_t gap_opening_cost = -1;
    error_cost_t gap_extension_cost = -1;

    smith_waterman_baselines_t() = default;
    smith_waterman_baselines_t(error_costs_32x32_t subs, linear_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open_or_extend), gap_extension_cost(gap.open_or_extend) {}
    smith_waterman_baselines_t(error_costs_32x32_t subs, affine_gap_costs_t gap)
        : substitution_costs(subs), gap_opening_cost(gap.open), gap_extension_cost(gap.extend) {}

    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, sz_ssize_t *results) const {
        sz_assert_(first.size() == second.size());

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
 *  @brief Tests the correctness of the string class Levenshtein distance computation,
 *         as well as the similarity scoring functions for bioinformatics-like workloads
 *         on a @b fixed set of different representative ASCII and UTF-8 strings.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... simd_extra_args_>
void test_similarity_scores_fixed(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                  std::string_view allowed_chars = {}, simd_extra_args_ &&...simd_extra_args) {

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
    append("London, the iconic capital of the United Kingdom, seamlessly blends centuries-old traditions with bold " //
           "modernity;"                                                                                              //
           "倫敦作為英國的標誌性首都，其歷史沉澱與當代創新彼此交融，展現獨特風範;"                                   //
           "Лондон, столица Великобритании, объединяет древние традиции с динамичной современностью, "               //
           "offering a rich tapestry of cultural heritage and visionary progress.", // First string ends here ;)
           "London, the renowned capital of the UK, fuses its rich historical legacy with a spirit of modern " //
           "innovation;"                                                                                       //
           "倫敦，作為英國的著名首都，以悠久歷史與現代創意相互融合，呈現獨特都市風貌;"                         //
           "Лондон – известная столица Великобритании, где древность встречается с современной энергией, "     //
           "creating an inspiring environment for cultural exploration and future development.");

    // ~300 characters; a complex variant with translations and visible regions of Korean, Japanese, Chinese
    // (traditional and simplified), German, French, Spanish.
    append("An epic voyage through multicultural realms: "                                                           //
           "In a city where ancient traditions fuse with modern innovation, dynamic energy permeates every street. " //
           "서울의 번화한 거리에선 전통과 현대가 어우러져 감동을 주며, 東京では伝統美と未来の夢が共鳴する。在這裡, " //
           "傳統文化與現代科技和諧並存, 而这里, 传统文化与现代科技交织创新; "                                        //
           "Deutschland zeigt eine reiche Geschichte, "                                                              //
           "la France révèle une élégance subtile, "                                                                 //
           "y España irradia pasión y color.",          // First string ends here ;)
           "An epic journey through diverse cultures: " //
           "In a town where old traditions fuse with innovation, energy permeates every historic street. "            //
           "서울의 번화한 거리는 전통과 현대가 어울려 독특한 풍경을 이루며, "                                         //
           "東京では伝統美と未来への展望が響き合う。在這裡, 傳統與現代科技融合無間, 而这里, 传统与现代科技紧密相连; " //
           "Deutschland offenbart eine stolze Geschichte, "                                                           //
           "la France incarne une élégance fine, "                                                                    //
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
        status_t status_simd = simd_operator(first_view, second_view, results_simd_ptr, simd_extra_args...);
        sz_assert_(status_base == status_t::success_k);
        sz_assert_(status_simd == status_t::success_k);
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
            sz_assert_(first_tape.try_append({first.data(), first.size()}) == status_t::success_k);
            sz_assert_(second_tape.try_append({second.data(), second.size()}) == status_t::success_k);
        }

        // Compute with both backends
        status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data(),
                                             simd_extra_args...);
        sz_assert_(status_base == status_t::success_k);
        sz_assert_(status_simd == status_t::success_k);

        // Individually log the failed results
        for (std::size_t i = 0; i != test_cases.size(); ++i) {
            if (results_base[i] == results_simd[i]) continue;
            edit_distance_log_mismatch(test_cases[i].first, test_cases[i].second, results_base[i], results_simd[i]);
        }
    }
}

/**
 *  @brief Tests the correctness of the string class Levenshtein distance computation,
 *         as well as the similarity scoring functions for bioinformatics-like workloads
 *         on a synthetic @b randomly-generated set of strings from a given @p alphabet.
 *
 *  @note The default iteration count (100) is scaled by `SZ_TEST_ITERATIONS_MULTIPLIER`.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... simd_extra_args_>
void test_similarity_scores_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                  fuzzy_config_t config = {}, std::size_t iterations = scale_iterations(100),
                                  simd_extra_args_ &&...simd_extra_args) {

    unified_vector<score_type_> results_base(config.batch_size), results_simd(config.batch_size);
    std::vector<std::string> first_array, second_array;
    arrow_strings_tape_t first_tape, second_tape;

    // Generate some random strings, using a small alphabet
    for (std::size_t iteration_idx = 0; iteration_idx < iterations; ++iteration_idx) {
        randomize_strings(config, first_array, first_tape);
        randomize_strings(config, second_array, second_tape);

        // Compute with both backends
        status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data(),
                                             simd_extra_args...);
        sz_assert_(status_base == status_t::success_k);
        sz_assert_(status_simd == status_t::success_k);

        // Individually log the failed results
        for (std::size_t i = 0; i != config.batch_size; ++i) {
            if (results_base[i] == results_simd[i]) continue;
            edit_distance_log_mismatch(first_array[i], second_array[i], results_base[i], results_simd[i]);
        }
    }
}

template <typename score_type_, typename base_operator_, typename simd_operator_, typename... simd_extra_args_>
void test_similarity_scores_fixed_and_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                            std::string_view allowed_chars = {}, fuzzy_config_t config = {},
                                            simd_extra_args_ &&...simd_extra_args) {
    test_similarity_scores_fixed<score_type_>(base_operator, simd_operator, allowed_chars, simd_extra_args...);
    test_similarity_scores_fuzzy<score_type_>(base_operator, simd_operator, config, 1, simd_extra_args...);
}

/**
 *  @brief Tests the correctness of the string class Levenshtein distance, NW & SW score computation,
 *         comparing the results to some baseline implementation for predefined and random inputs.
 */
void test_similarity_scores_equivalence() {

    using error_t = error_cost_t;
    using error_matrix32_t = error_costs_32x32_t;

    // Our logic of computing NW and SW alignment similarity scores differs in sign from most implementations.
    // It's similar to how the "cosine distance" is the inverse of the "cosine similarity".
    // In our case we compute the "distance" and by negating the sign, we can compute the "similarity".
    {
        constexpr error_t unary_match_score = 1;
        constexpr error_t unary_mismatch_score = 0;
        constexpr error_t unary_gap_score = 0;
        uniform_substitution_costs_t substituter_unary {unary_match_score, unary_mismatch_score};
        auto distance_l = levenshtein_baseline("abcdefg", 7, "abc_efg", 7);
        auto similarity_nw = needleman_wunsch_baseline("abcdefg", 7, "abc_efg", 7, substituter_unary, unary_gap_score);
        auto similarity_sw = smith_waterman_baseline("abcdefg", 7, "abc_efg", 7, substituter_unary, unary_gap_score);
        // Distance can be computed from the similarity, by inverting the sign around the length of the longest string:
        auto distance_nw = std::max(7, 7) - similarity_nw;
        auto distance_sw = std::max(7, 7) - similarity_sw;
        sz_assert_(distance_l == 1);
        sz_assert_(distance_nw == 1);
        sz_assert_(distance_sw == 1);
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
    error_matrix32_t blosum62_matrix = error_costs_32x32_t::blosum62();

    // Single-threaded serial NW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Multi-threaded parallel NW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Single-threaded serial SW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Multi-threaded parallel SW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Single-threaded serial NW implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost});

    // Multi-threaded parallel NW implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost});

    // Single-threaded serial SW implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        smith_waterman_scores<char, error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost});

    // Multi-threaded parallel SW implementation with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        smith_waterman_scores<char, error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost});

#if SZ_USE_ICELAKE
    // Ice Lake Levenshtein distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                 //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {});

    // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird linear costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}, //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_linear});

    // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}, //
        levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_affine});

    // Ice Lake Levenshtein UTF8 distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                      //
        levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, //
        levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {});

    // Ice Lake Levenshtein UTF8 distance against Multi-threaded on CPU with weird linear costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear},
        levenshtein_distances_utf8<char, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_linear});

    // Ice Lake Needleman-Wunsch distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_linear_cost});

    // Ice Lake Smith-Waterman distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_linear_cost});

#endif

#if SZ_USE_HASWELL
    // Haswell Needleman-Wunsch distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                       //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t,
                                (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {blosum62_matrix,
                                                                                        blosum62_linear_cost});

    // Haswell Smith-Waterman distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                     //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t,
                              (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {blosum62_matrix,
                                                                                      blosum62_linear_cost});

#endif

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs;
    sz_assert_(gpu_specs_fetch(first_gpu_specs) == status_t::success_k);
#endif

#if SZ_USE_CUDA
    // CUDA Levenshtein distance against Multi-threaded on CPU with weird linear costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}, //
        levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {weird_uniform, weird_linear}, {}, {},
        cuda_executor_t {}, first_gpu_specs);

    // CUDA Levenshtein distance against Multi-threaded on CPU with weird affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}, //
        levenshtein_distances<char, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {weird_uniform, weird_affine}, {}, {},
        cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_KEPLER
    // CUDA Levenshtein distance on Kepler against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(                                                            //
        levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}, //
        levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_caps_ck_k> {weird_uniform, weird_linear}, {}, {},
        cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_CUDA
    // CUDA Needleman-Wunsch score against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_linear_cost},
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Needleman-Wunsch score against Multi-threaded on CPU with affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_affine_cost},
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Smith-Waterman score against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        smith_waterman_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_linear_cost},
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Smith-Waterman score against Multi-threaded on CPU with affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        smith_waterman_scores<char, error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}, //
        smith_waterman_scores<char, error_matrix32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_affine_cost},
        {}, {}, cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_HOPPER
    // CUDA Needleman-Wunsch score on Hopper against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_linear_cost},
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Needleman-Wunsch score on Hopper against Multi-threaded on CPU with affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}, //
        needleman_wunsch_scores<char, error_matrix32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_affine_cost},
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Smith-Waterman score on Hopper against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        smith_waterman_scores<char, error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}, //
        smith_waterman_scores<char, error_matrix32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_linear_cost},
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Smith-Waterman score on Hopper against Multi-threaded on CPU with affine costs
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        smith_waterman_scores<char, error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}, //
        smith_waterman_scores<char, error_matrix32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_affine_cost},
        {}, {}, cuda_executor_t {}, first_gpu_specs);

#endif
}

/**
 *  @brief Many GPU algorithms depend on effective use of shared memory and scheduling its allocation for
 *         long inputs or very large batches isn't trivial.
 */
void test_similarity_scores_memory_usage() {

    std::vector<fuzzy_config_t> experiments {
        // Single string pair of same length:
        {"ABC", /* batch_size */ 1, /* min_string_length */ 128, /* max_string_length */ 128},
        {"ABC", /* batch_size */ 1, /* min_string_length */ 512, /* max_string_length */ 512},
        {"ABC", /* batch_size */ 1, /* min_string_length */ 2048, /* max_string_length */ 2048},
        {"ABC", /* batch_size */ 1, /* min_string_length */ 8192, /* max_string_length */ 8192},
        {"ABC", /* batch_size */ 1, /* min_string_length */ 32768, /* max_string_length */ 32768},
        {"ABC", /* batch_size */ 1, /* min_string_length */ 131072, /* max_string_length */ 131072},
        // Two strings of a same length:
        {"ABC", /* batch_size */ 2, /* min_string_length */ 128, /* max_string_length */ 128},
        {"ABC", /* batch_size */ 2, /* min_string_length */ 512, /* max_string_length */ 512},
        {"ABC", /* batch_size */ 2, /* min_string_length */ 2048, /* max_string_length */ 2048},
        {"ABC", /* batch_size */ 2, /* min_string_length */ 8192, /* max_string_length */ 8192},
        {"ABC", /* batch_size */ 2, /* min_string_length */ 32768, /* max_string_length */ 32768},
        {"ABC", /* batch_size */ 2, /* min_string_length */ 131072, /* max_string_length */ 131072},
        // Ten strings of random lengths:
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 128},
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 512},
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 2048},
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 8192},
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 32768},
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 131072},
    };

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs;
    sz_assert_(gpu_specs_fetch(first_gpu_specs) == status_t::success_k);
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

#if SZ_USE_ICELAKE
        // Ice Lake Levenshtein distance against Multi-threaded on CPU
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {},
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {}, experiment, 1);

        // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird linear costs
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear},
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_linear},
            experiment, 1);

        // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird affine costs
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine},
            levenshtein_distances<char, affine_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_affine},
            experiment, 1);
#endif

#if SZ_USE_CUDA
        // CUDA Levenshtein distance against Multi-threaded on CPU
        test_similarity_scores_fuzzy<sz_size_t>(                                           //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, //
            levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {}, experiment, 10,
            cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_KEPLER
        // CUDA Levenshtein distance on Kepler against Multi-threaded on CPU
        test_similarity_scores_fuzzy<sz_size_t>(                                           //
            levenshtein_distances<char, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}, //
            levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_caps_ck_k> {}, experiment, 10,
            cuda_executor_t {}, first_gpu_specs);
#endif
    }
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
