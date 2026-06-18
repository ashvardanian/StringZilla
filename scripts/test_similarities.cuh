/**
 *  @brief Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file scripts/test_similarities.cuh
 *  @author Ash Vardanian
 *  @date June 16, 2026
 */
#include "stringzillas/similarities.hpp"

#if SZ_USE_CUDA
#include "stringzillas/similarities.cuh"
#endif

#if !SZ_IS_CPP17_
#error "This test requires C++17 or later."
#endif

#include "test_stringzilla.hpp" // `arrow_strings_view_t`

#include <sys/resource.h> // `getrusage`, `RUSAGE_SELF` for the coarse CPU peak-RSS backstop

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
using ashvardanian::stringzillas::strided_rows;
using ashvardanian::stringzillas::uniform_substitution_costs_t;

// StringZillas library symbols provided only by the CUDA backend:
#if SZ_USE_CUDA
using ashvardanian::stringzillas::cuda_executor_t;
using ashvardanian::stringzillas::gpu_specs_fetch;
using ashvardanian::stringzillas::ualloc_t;
#endif

#pragma region Helpers

/** @brief Dual-row O(n)-memory reference Levenshtein distance, keeping only the previous and current rows. */
inline std::size_t levenshtein_baseline(                                //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    error_cost_t match_cost = 0, error_cost_t mismatch_cost = 1, error_cost_t gap_cost = 1) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::size_t> previous_row(cols), current_row(cols);

    // Initialize the first row's border.
    for (std::size_t j = 0; j < cols; ++j) previous_row[j] /* [0][j] in 2D */ = j * gap_cost;

    for (std::size_t i = 1; i < rows; ++i) {
        current_row[0] /* [i][0] in 2D */ = i * gap_cost;
        for (std::size_t j = 1; j < cols; ++j) {
            std::size_t substitution_cost = (s1[i - 1] == s2[j - 1]) ? match_cost : mismatch_cost;
            std::size_t if_deletion_or_insertion = std::min(previous_row[j], current_row[j - 1]) + gap_cost;
            current_row[j] = std::min(if_deletion_or_insertion, previous_row[j - 1] + substitution_cost);
        }
        previous_row.swap(current_row);
    }

    // After the last swap, the bottom-right cell sits at the end of `previous_row`.
    return previous_row.back();
}

/** @brief Dual-row O(n)-memory reference Needleman-Wunsch alignment score, keeping only two rows. */
template <typename substituter_type_>
inline std::ptrdiff_t needleman_wunsch_baseline(                        //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    substituter_type_ substitution_cost_for, error_cost_t gap_cost) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::ptrdiff_t> previous_row(cols), current_row(cols);

    // Initialize the first row's border.
    for (std::size_t j = 0; j < cols; ++j) previous_row[j] /* [0][j] in 2D */ = j * gap_cost;

    // Fill in the rest of the matrix, one row at a time.
    for (std::size_t i = 1; i < rows; ++i) {
        current_row[0] /* [i][0] in 2D */ = i * gap_cost;
        for (std::size_t j = 1; j < cols; ++j) {
            std::ptrdiff_t substitution_cost = substitution_cost_for(s1[i - 1], s2[j - 1]);
            std::ptrdiff_t if_substitution = previous_row[j - 1] + substitution_cost;
            std::ptrdiff_t if_deletion_or_insertion = std::max(previous_row[j], current_row[j - 1]) + gap_cost;
            current_row[j] = std::max(if_deletion_or_insertion, if_substitution);
        }
        previous_row.swap(current_row);
    }

    // After the last swap, the bottom-right cell sits at the end of `previous_row`.
    return previous_row.back();
}

/** @brief Dual-row O(n)-memory reference Smith-Waterman local alignment score, keeping only two rows. */
template <typename substituter_type_>
inline std::ptrdiff_t smith_waterman_baseline(char const *s1, std::size_t len1, char const *s2, std::size_t len2,
                                              substituter_type_ substitution_cost_for,
                                              error_cost_t gap_cost) noexcept(false) {
    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::ptrdiff_t> previous_row(cols), current_row(cols);

    // Unlike the global alignment we need to track the largest score across all cells.
    std::ptrdiff_t best_score = 0;

    // Initialize the first row's border to 0.
    for (std::size_t j = 0; j < cols; ++j) previous_row[j] /* [0][j] in 2D */ = 0;

    // Fill in the rest of the matrix, one row at a time.
    for (std::size_t i = 1; i < rows; ++i) {
        current_row[0] /* [i][0] in 2D */ = 0;
        for (std::size_t j = 1; j < cols; ++j) {
            std::ptrdiff_t substitution_cost = substitution_cost_for(s1[i - 1], s2[j - 1]);
            std::ptrdiff_t if_substitution = previous_row[j - 1] + substitution_cost;
            std::ptrdiff_t if_deletion_or_insertion = std::max(current_row[j - 1], previous_row[j]) + gap_cost;
            std::ptrdiff_t if_substitution_or_reset = std::max<std::ptrdiff_t>(if_substitution, 0);
            std::ptrdiff_t score = std::max(if_deletion_or_insertion, if_substitution_or_reset);
            current_row[j] = score;
            best_score = std::max(best_score, score);
        }
        previous_row.swap(current_row);
    }

    return best_score;
}

/** @brief Dual-row O(n)-memory reference Levenshtein-Gotoh (affine-gap) distance, keeping two rows per matrix. */
inline std::size_t levenshtein_gotoh_baseline(                          //                      //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    error_cost_t match_cost, error_cost_t mismatch_cost,                //
    error_cost_t gap_opening_cost, error_cost_t gap_extension_cost) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::size_t> previous_scores(cols), current_scores(cols);
    std::vector<std::size_t> previous_inserts(cols), current_inserts(cols);
    std::vector<std::size_t> previous_deletes(cols), current_deletes(cols);

    // Initialize the first row's border.
    // The supplementary matrices are initialized with values of higher magnitude,
    // which is equivalent to discarding them. That's better than using `SIZE_MAX`
    // as subsequent additions won't overflow.
    previous_scores[0] = 0;
    for (std::size_t j = 1; j < cols; ++j) {
        previous_scores[j] = gap_opening_cost + (j - 1) * gap_extension_cost;
        previous_deletes[j] = previous_scores[j] + gap_opening_cost + gap_extension_cost;
    }

    // Fill in the rest of the matrix, one row at a time.
    for (std::size_t i = 1; i < rows; ++i) {
        current_scores[0] /* [i][0] in 2D */ = gap_opening_cost + (i - 1) * gap_extension_cost;
        current_inserts[0] /* [i][0] in 2D */ = current_scores[0] + gap_opening_cost + gap_extension_cost;
        for (std::size_t j = 1; j < cols; ++j) {
            std::size_t substitution_cost = (s1[i - 1] == s2[j - 1]) ? match_cost : mismatch_cost;
            std::size_t if_substitution = previous_scores[j - 1] + substitution_cost;
            std::size_t if_insertion = std::min<std::size_t>(current_scores[j - 1] + gap_opening_cost,
                                                             current_inserts[j - 1] + gap_extension_cost);
            std::size_t if_deletion = std::min<std::size_t>(previous_scores[j] + gap_opening_cost,
                                                            previous_deletes[j] + gap_extension_cost);
            std::size_t if_deletion_or_insertion = std::min(if_deletion, if_insertion);
            current_scores[j] = std::min(if_deletion_or_insertion, if_substitution);
            current_inserts[j] = if_insertion;
            current_deletes[j] = if_deletion;
        }
        previous_scores.swap(current_scores);
        previous_inserts.swap(current_inserts);
        previous_deletes.swap(current_deletes);
    }

    // After the last swap, the bottom-right cell sits at the end of `previous_scores`.
    return previous_scores.back();
}

/**
 *  @brief Dual-row O(n)-memory reference Needleman-Wunsch-Gotoh (affine-gap) score, keeping two rows per matrix.
 *  @see https://github.com/gata-bio/affine-gaps
 */
template <typename substituter_type_>
inline std::ptrdiff_t needleman_wunsch_gotoh_baseline(                  //              //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    substituter_type_ substitution_cost_for,                            //
    error_cost_t gap_opening_cost, error_cost_t gap_extension_cost) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::ptrdiff_t> previous_scores(cols), current_scores(cols);
    std::vector<std::ptrdiff_t> previous_inserts(cols), current_inserts(cols);
    std::vector<std::ptrdiff_t> previous_deletes(cols), current_deletes(cols);

    // Initialize the first row's border.
    previous_scores[0] = 0;
    for (std::size_t j = 1; j < cols; ++j) {
        previous_scores[j] = gap_opening_cost + (j - 1) * gap_extension_cost;
        previous_deletes[j] = previous_scores[j] + gap_opening_cost + gap_extension_cost;
    }

    // Fill in the rest of the matrix, one row at a time.
    for (std::size_t i = 1; i < rows; ++i) {
        current_scores[0] /* [i][0] in 2D */ = gap_opening_cost + (i - 1) * gap_extension_cost;
        current_inserts[0] /* [i][0] in 2D */ = current_scores[0] + gap_opening_cost + gap_extension_cost;
        for (std::size_t j = 1; j < cols; ++j) {
            std::ptrdiff_t substitution_cost = substitution_cost_for(s1[i - 1], s2[j - 1]);
            std::ptrdiff_t if_substitution = previous_scores[j - 1] + substitution_cost;
            std::ptrdiff_t if_insertion = std::max(current_scores[j - 1] + gap_opening_cost,
                                                   current_inserts[j - 1] + gap_extension_cost);
            std::ptrdiff_t if_deletion = std::max(previous_scores[j] + gap_opening_cost,
                                                  previous_deletes[j] + gap_extension_cost);
            std::ptrdiff_t if_deletion_or_insertion = std::max(if_deletion, if_insertion);
            current_scores[j] = std::max(if_deletion_or_insertion, if_substitution);
            current_inserts[j] = if_insertion;
            current_deletes[j] = if_deletion;
        }
        previous_scores.swap(current_scores);
        previous_inserts.swap(current_inserts);
        previous_deletes.swap(current_deletes);
    }

    // After the last swap, the bottom-right cell sits at the end of `previous_scores`.
    return previous_scores.back();
}

/**
 *  @brief Dual-row O(n)-memory reference Smith-Waterman-Gotoh (affine-gap) local score, keeping two rows per matrix.
 *  @see https://github.com/gata-bio/affine-gaps
 */
template <typename substituter_type_>
inline std::ptrdiff_t smith_waterman_gotoh_baseline(                    //                //
    char const *s1, std::size_t len1, char const *s2, std::size_t len2, //
    substituter_type_ substitution_cost_for,                            //
    error_cost_t gap_opening_cost, error_cost_t gap_extension_cost) noexcept(false) {

    std::size_t const rows = len1 + 1;
    std::size_t const cols = len2 + 1;
    std::vector<std::ptrdiff_t> previous_scores(cols), current_scores(cols);
    std::vector<std::ptrdiff_t> previous_inserts(cols), current_inserts(cols);
    std::vector<std::ptrdiff_t> previous_deletes(cols), current_deletes(cols);

    // Unlike the global alignment we need to track the largest score across all cells.
    std::ptrdiff_t best_score = 0;

    // Initialize the first row's border.
    previous_scores[0] = 0;
    for (std::size_t j = 1; j < cols; ++j) {
        previous_scores[j] = 0;
        previous_deletes[j] = gap_opening_cost + gap_extension_cost;
    }

    // Fill in the rest of the matrix, one row at a time.
    for (std::size_t i = 1; i < rows; ++i) {
        current_scores[0] /* [i][0] in 2D */ = 0;
        current_inserts[0] /* [i][0] in 2D */ = gap_opening_cost + gap_extension_cost;
        for (std::size_t j = 1; j < cols; ++j) {
            std::ptrdiff_t substitution_cost = substitution_cost_for(s1[i - 1], s2[j - 1]);
            std::ptrdiff_t if_substitution = previous_scores[j - 1] + substitution_cost;
            std::ptrdiff_t if_insertion = std::max(current_scores[j - 1] + gap_opening_cost,
                                                   current_inserts[j - 1] + gap_extension_cost);
            std::ptrdiff_t if_deletion = std::max(previous_scores[j] + gap_opening_cost,
                                                  previous_deletes[j] + gap_extension_cost);
            std::ptrdiff_t if_deletion_or_insertion = std::max(if_deletion, if_insertion);
            std::ptrdiff_t if_substitution_or_reset = std::max<std::ptrdiff_t>(if_substitution, 0);
            std::ptrdiff_t score = std::max(if_deletion_or_insertion, if_substitution_or_reset);
            current_scores[j] = score;
            current_inserts[j] = if_insertion;
            current_deletes[j] = if_deletion;
            best_score = std::max(best_score, score);
        }
        previous_scores.swap(current_scores);
        previous_inserts.swap(current_inserts);
        previous_deletes.swap(current_deletes);
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

/** @brief Logs a single base-vs-SIMD edit-distance mismatch with truncated string previews. */
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
 *  @brief Adapts a cross-product similarity engine into the @b pairwise calling convention the agreement suites use.
 *
 *  The agreement suites (`check_similarities_*`) drive each engine over @b paired views (`first[i]` against
 *  `second[i]`), writing a flat `results[i]`. The cross-product engines instead score a `Q × C` matrix into a
 *  `strided_rows`, so this adapter forwards each diagonal pair as its own 1x1 tile and threads any trailing
 *  executor / spec arguments straight through.
 */
template <typename engine_type_>
struct pairwise_via_cross_t {
    using engine_t = engine_type_;
    engine_t engine = {};

    template <typename score_type_, typename... extra_args_>
    status_t operator()(arrow_strings_view_t first, arrow_strings_view_t second, score_type_ *results,
                        extra_args_ &&...extra_args) {
        sz_assert_(first.size() == second.size());
        std::size_t const pairs_count = first.size();
        for (std::size_t pair_index = 0; pair_index != pairs_count; ++pair_index) {
            arrow_strings_view_t const first_cell {first.buffer_, first.offsets_.subspan(pair_index, 2)};
            arrow_strings_view_t const second_cell {second.buffer_, second.offsets_.subspan(pair_index, 2)};
            strided_rows<score_type_> const single_cell {&results[pair_index], 1, 1, 1};
            status_t const status = engine(first_cell, second_cell, single_cell, extra_args...);
            if (status != status_t::success_k) return status;
        }
        return status_t::success_k;
    }
};

/** @brief Deduces the engine type so call sites stay terse: `make_pairwise(engine)`. */
template <typename engine_type_>
inline pairwise_via_cross_t<engine_type_> make_pairwise(engine_type_ engine) noexcept {
    return pairwise_via_cross_t<engine_type_> {std::move(engine)};
}

#pragma endregion // Helpers

#pragma region Unit

/** @brief Checks base-vs-SIMD agreement on a @b fixed set of representative ASCII and UTF-8 strings. */
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... simd_extra_args_>
static void check_similarities_fixed_(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
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

/** @brief Runs one string pair through a similarity operator and asserts the @b known expected score. */
template <typename score_type_, typename operator_type_, typename... extra_args_>
static void check_similarities_known_(operator_type_ &&similarity_operator, std::string const &first,
                                      std::string const &second, score_type_ expected, extra_args_ &&...extra_args) {

    arrow_strings_tape_t first_tape, second_tape;
    sz_assert_(first_tape.try_append({first.data(), first.size()}) == status_t::success_k);
    sz_assert_(second_tape.try_append({second.data(), second.size()}) == status_t::success_k);

    unified_vector<score_type_> result(1);
    result[0] = std::numeric_limits<score_type_>::max();
    status_t status = similarity_operator(first_tape.view(), second_tape.view(), result.data(), extra_args...);
    sz_assert_(status == status_t::success_k);
    if (result[0] != expected) edit_distance_log_mismatch(first, second, expected, result[0]);
    sz_assert_(result[0] == expected);
}

/**
 *  @brief Known-answer unit tests for the similarity family on simple, hand-verifiable string pairs.
 *
 *  Unlike `test_similarities_equivalence`, this needs no second (SIMD/GPU) backend: it pins the
 *  hand-computed unit-cost Levenshtein distance for each pair and drives it through both the dispatched
 *  serial engine and the dual-row baseline. A regression that the serial-vs-SIMD agreement tests would
 *  miss - because both share a wrong constant - is still caught here against the external ground truth.
 */
void test_similarities_unit() {
    std::printf("  - testing similarity known-answer vectors...\n");

    // Hand-verifiable unit-cost (match 0, mismatch 1, gap 1) Levenshtein distances on ASCII pairs.
    struct known_levenshtein_t {
        char const *first;
        char const *second;
        sz_size_t distance;
    };
    known_levenshtein_t const levenshtein_vectors[] = {
        {"", "", 0},                        // both empty
        {"ABC", "ABC", 0},                  // identical
        {"A", "=", 1},                      // single substitution
        {"", "ABC", 3},                     // pure insertion
        {"ABC", "", 3},                     // pure deletion
        {"ABC", "AABC", 1},                 // one prepended insertion
        {"ABC", "ABCC", 1},                 // one appended insertion
        {"ABC", "AC", 1},                   // one deletion
        {"ABC", "AXBC", 1},                 // one insertion
        {"ABC", "AXC", 1},                  // one substitution
        {"ABCDEFG", "ABCXEFG", 1},          // one substitution
        {"ggbuzgjux{}l", "gbuzgjux{}l", 1}, // one prepended insertion
        {"APPLE", "APLE", 1},               // one deletion
        {"LISTEN", "SILENT", 4},            // classic anagram-ish pair
        {"ATCA", "CTACTCACCC", 6},          // DNA-like pair
    };

    constexpr uniform_substitution_costs_t unit_uniform {0, 1};
    constexpr linear_gap_costs_t unit_linear {1};

    for (known_levenshtein_t const &vector : levenshtein_vectors) {
        std::string const first {vector.first};
        std::string const second {vector.second};

        // Dispatched serial engine (automatic kernel resolution within the serial capability).
        check_similarities_known_<sz_size_t>(
            make_pairwise(
                levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {unit_uniform, unit_linear}), //
            first, second, vector.distance);

        // The dual-row reference baseline, sharing none of the engine's machinery.
        check_similarities_known_<sz_size_t>(levenshtein_baselines_t {unit_uniform, unit_linear}, //
                                             first, second, vector.distance);
    }
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief Checks base-vs-SIMD agreement on synthetic @b randomly-generated strings from a given @p alphabet.
 *  @note The default iteration count (100) is scaled by `SZ_TEST_ITERATIONS_MULTIPLIER`.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... simd_extra_args_>
static void check_similarities_fuzzy_(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
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

/** @brief Runs both the fixed and a single fuzzy base-vs-SIMD agreement pass for one backend pairing. */
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... simd_extra_args_>
static void check_similarities_fixed_and_fuzzy_(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                                std::string_view allowed_chars = {}, fuzzy_config_t config = {},
                                                simd_extra_args_ &&...simd_extra_args) {
    check_similarities_fixed_<score_type_>(base_operator, simd_operator, allowed_chars, simd_extra_args...);
    check_similarities_fuzzy_<score_type_>(base_operator, simd_operator, config, 1, simd_extra_args...);
}

/**
 *  @brief Validates Levenshtein, Needleman-Wunsch & Smith-Waterman scores across every backend against
 *         the dual-row baselines, over fixed representative strings and random fuzzed inputs.
 */
void test_similarities_equivalence() {

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
    check_similarities_fixed_and_fuzzy_<sz_size_t>( //
        levenshtein_baselines_t {},                 //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}));

    // Multi-threaded parallel Levenshtein distance implementation
    check_similarities_fixed_and_fuzzy_<sz_size_t>( //
        levenshtein_baselines_t {},                 //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}));

    // Single-threaded serial Levenshtein distance implementation with weird linear costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>(            //
        levenshtein_baselines_t {weird_uniform, weird_linear}, //
        make_pairwise(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}));

    // Multi-threaded parallel Levenshtein distance implementation with weird linear costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>(            //
        levenshtein_baselines_t {weird_uniform, weird_linear}, //
        make_pairwise(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}));

    // Single-threaded serial Levenshtein distance implementation with weird affine costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>(            //
        levenshtein_baselines_t {weird_uniform, weird_affine}, //
        make_pairwise(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}));

    // Multi-threaded parallel Levenshtein distance implementation with weird affine costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>(            //
        levenshtein_baselines_t {weird_uniform, weird_affine}, //
        make_pairwise(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}));

    // Now let's take non-unary substitution costs, like BLOSUM62
    constexpr linear_gap_costs_t blosum62_linear_cost {-4};
    constexpr affine_gap_costs_t blosum62_affine_cost {-4, -1};
    error_matrix32_t blosum62_matrix = error_costs_32x32_t::blosum62();

    // Single-threaded serial NW implementation
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                          //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}));

    // Multi-threaded parallel NW implementation
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                          //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}));

    // Single-threaded serial SW implementation
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                        //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        make_pairwise(smith_waterman_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}));

    // Multi-threaded parallel SW implementation
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                        //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        make_pairwise(smith_waterman_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}));

    // Single-threaded serial NW implementation with weird affine costs
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                          //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}));

    // Multi-threaded parallel NW implementation with weird affine costs
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                          //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}));

    // Single-threaded serial SW implementation with weird affine costs
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                        //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        make_pairwise(smith_waterman_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}));

    // Multi-threaded parallel SW implementation with weird affine costs
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                        //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        make_pairwise(smith_waterman_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}));

#if SZ_USE_ICELAKE
    // Ice Lake Levenshtein distance against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_size_t>(                                             //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}), //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {}));

    // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird linear costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>( //
        make_pairwise(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}), //
        make_pairwise(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_linear}));

    // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird affine costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>( //
        make_pairwise(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}), //
        make_pairwise(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_affine}));

    // Ice Lake Levenshtein UTF8 distance against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_size_t>(                                                  //
        make_pairwise(levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}), //
        make_pairwise(levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {}));

    // Ice Lake Levenshtein UTF8 distance against Multi-threaded on CPU with weird linear costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>( //
        make_pairwise(
            levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}),
        make_pairwise(
            levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_linear}));

    // Ice Lake Needleman-Wunsch distance against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                          //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_linear_cost}));

    // Ice Lake Smith-Waterman distance against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                        //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        make_pairwise(smith_waterman_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_linear_cost}));

    // Ice Lake AFFINE Needleman-Wunsch / Smith-Waterman against Multi-threaded on CPU (new affine candidate-lane).
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                          //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_affine_cost}));
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                        //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        make_pairwise(smith_waterman_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_affine_cost}));

#endif

#if SZ_USE_HASWELL
    // Haswell Needleman-Wunsch distance against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                          //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, linear_gap_costs_t, malloc_t,
                                              (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {
            blosum62_matrix, blosum62_linear_cost}));

    // Haswell Smith-Waterman distance against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                        //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, //
        make_pairwise(smith_waterman_scores<error_matrix32_t, linear_gap_costs_t, malloc_t,
                                            (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {
            blosum62_matrix, blosum62_linear_cost}));

    // Haswell AFFINE Needleman-Wunsch / Smith-Waterman against Multi-threaded on CPU (new affine candidate-lane).
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                          //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, affine_gap_costs_t, malloc_t,
                                              (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {
            blosum62_matrix, blosum62_affine_cost}));
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>(                        //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, //
        make_pairwise(smith_waterman_scores<error_matrix32_t, affine_gap_costs_t, malloc_t,
                                            (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {
            blosum62_matrix, blosum62_affine_cost}));

#endif

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs;
    sz_assert_(gpu_specs_fetch(first_gpu_specs) == status_t::success_k);
#endif

#if SZ_USE_CUDA
    // CUDA Levenshtein distance against Multi-threaded on CPU with weird linear costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>( //
        make_pairwise(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}), //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {weird_uniform, weird_linear}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Levenshtein distance against Multi-threaded on CPU with weird affine costs
    check_similarities_fixed_and_fuzzy_<sz_size_t>( //
        make_pairwise(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}), //
        make_pairwise(levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {weird_uniform, weird_affine}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_KEPLER
    // CUDA Levenshtein distance on Kepler against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_size_t>( //
        make_pairwise(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}), //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_caps_ck_k> {weird_uniform, weird_linear}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_CUDA
    // CUDA Needleman-Wunsch score against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>( //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}), //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_linear_cost}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Needleman-Wunsch score against Multi-threaded on CPU with affine costs
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>( //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}), //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_affine_cost}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Smith-Waterman score against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>( //
        make_pairwise(smith_waterman_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}), //
        make_pairwise(smith_waterman_scores<error_matrix32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_linear_cost}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Smith-Waterman score against Multi-threaded on CPU with affine costs
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>( //
        make_pairwise(smith_waterman_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}), //
        make_pairwise(smith_waterman_scores<error_matrix32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k> {
            blosum62_matrix, blosum62_affine_cost}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_HOPPER
    // CUDA Needleman-Wunsch score on Hopper against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>( //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}), //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_linear_cost}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Needleman-Wunsch score on Hopper against Multi-threaded on CPU with affine costs
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>( //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}), //
        make_pairwise(needleman_wunsch_scores<error_matrix32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_affine_cost}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Smith-Waterman score on Hopper against Multi-threaded on CPU
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>( //
        make_pairwise(smith_waterman_scores<error_matrix32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost}), //
        make_pairwise(smith_waterman_scores<error_matrix32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_linear_cost}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);

    // CUDA Smith-Waterman score on Hopper against Multi-threaded on CPU with affine costs
    check_similarities_fixed_and_fuzzy_<sz_ssize_t>( //
        make_pairwise(smith_waterman_scores<error_matrix32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_affine_cost}), //
        make_pairwise(smith_waterman_scores<error_matrix32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k> {
            blosum62_matrix, blosum62_affine_cost}),
        {}, {}, cuda_executor_t {}, first_gpu_specs);

#endif
}

#pragma endregion // Equivalence

#pragma region Safety

/**
 *  @brief Runs one base-vs-SIMD agreement pass over @b degenerate strings, then asserts the closed-form
 *         Levenshtein identities `distance(x, "") == |x| * gap` and `distance(x, x) == 0` directly.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... simd_extra_args_>
static void check_similarities_degenerate_(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                           error_cost_t gap_cost, simd_extra_args_ &&...simd_extra_args) {

    // The degenerate corpus: empty/empty, empty/non-empty, single-char, identical, and a near-identical
    // one-edit pair. They live in one batch so the engines also face a mixed-length, mostly-tiny input.
    std::vector<std::pair<std::string, std::string>> degenerate_cases {
        {"", ""},             // both empty; distance 0
        {"", "ABC"},          // empty vs non-empty; pure insertion
        {"ABC", ""},          // non-empty vs empty; pure deletion
        {"A", "A"},           // single-char identical; distance 0
        {"A", "B"},           // single-char substitution
        {"ABCABC", "ABCABC"}, // identical longer string; distance 0
        {"ABCABC", "ABCXBC"}, // one planted substitution; unit-cost distance 1
    };

    unified_vector<score_type_> results_base(degenerate_cases.size());
    unified_vector<score_type_> results_simd(degenerate_cases.size());
    arrow_strings_tape_t first_tape, second_tape;
    for (auto const &pair : degenerate_cases) {
        sz_assert_(first_tape.try_append({pair.first.data(), pair.first.size()}) == status_t::success_k);
        sz_assert_(second_tape.try_append({pair.second.data(), pair.second.size()}) == status_t::success_k);
    }

    status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
    status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data(),
                                         simd_extra_args...);
    sz_assert_(status_base == status_t::success_k);
    sz_assert_(status_simd == status_t::success_k);
    for (std::size_t pair_idx = 0; pair_idx != degenerate_cases.size(); ++pair_idx) {
        if (results_base[pair_idx] == results_simd[pair_idx]) continue;
        edit_distance_log_mismatch(degenerate_cases[pair_idx].first, degenerate_cases[pair_idx].second,
                                   results_base[pair_idx], results_simd[pair_idx]);
        sz_assert_(results_base[pair_idx] == results_simd[pair_idx]);
    }

    // Closed-form identities, independent of any O(n²) reference. With a single uniform-cost string the only
    // alignment against an empty string is all-gaps, so the linear-gap Levenshtein distance is exactly `|x| * gap`.
    std::string const sample = "ADVERSARIAL";
    std::string const empty;
    arrow_strings_tape_t sample_tape, empty_tape, copy_tape;
    sz_assert_(sample_tape.try_append({sample.data(), sample.size()}) == status_t::success_k);
    sz_assert_(empty_tape.try_append({empty.data(), empty.size()}) == status_t::success_k);
    sz_assert_(copy_tape.try_append({sample.data(), sample.size()}) == status_t::success_k);

    unified_vector<score_type_> closed_form(1);
    score_type_ const expected_all_gaps = static_cast<score_type_>(sample.size() * static_cast<std::size_t>(gap_cost));
    sz_assert_(simd_operator(sample_tape.view(), empty_tape.view(), closed_form.data(), simd_extra_args...) ==
               status_t::success_k);
    sz_assert_(closed_form[0] == expected_all_gaps);
    sz_assert_(simd_operator(sample_tape.view(), copy_tape.view(), closed_form.data(), simd_extra_args...) ==
               status_t::success_k);
    sz_assert_(closed_form[0] == static_cast<score_type_>(0));
}

/**
 *  @brief Feeds @b degenerate and adversarial inputs through the distance/alignment engines on every
 *         available backend (serial, each `#if SZ_USE_*`, and CUDA where guarded), asserting no crash and
 *         agreement with the dual-row baselines and the closed-form Levenshtein identities.
 */
void test_similarities_safety() {
    std::printf("  - testing degenerate-input safety of similarity engines...\n");

    // A tiny, empty-inclusive fuzz pass shares the agreement machinery with the equivalence suite, but
    // pins the minimum length to zero so empty strings reach every kernel's prologue and tail handling.
    fuzzy_config_t degenerate_config {"ABC", /* batch_size */ 8, /* min_string_length */ 0, /* max_string_length */ 6};
    std::size_t const iterations = scale_iterations(8);

    constexpr linear_gap_costs_t unit_linear {1};
    constexpr uniform_substitution_costs_t unit_uniform {0, 1};

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs;
    sz_assert_(gpu_specs_fetch(first_gpu_specs) == status_t::success_k);
#endif

    // Serial Levenshtein distance against the dual-row baseline on degenerate inputs.
    check_similarities_fuzzy_<sz_size_t>(                    //
        levenshtein_baselines_t {unit_uniform, unit_linear}, //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {unit_uniform, unit_linear}),
        degenerate_config, iterations);
    check_similarities_degenerate_<sz_size_t>(               //
        levenshtein_baselines_t {unit_uniform, unit_linear}, //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {unit_uniform, unit_linear}),
        1);

#if SZ_USE_ICELAKE
    // Ice Lake Levenshtein distance against the serial CPU engine on degenerate inputs.
    check_similarities_fuzzy_<sz_size_t>( //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {unit_uniform, unit_linear}),
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {unit_uniform, unit_linear}),
        degenerate_config, iterations);
    check_similarities_degenerate_<sz_size_t>( //
        levenshtein_baselines_t {unit_uniform, unit_linear},
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {unit_uniform, unit_linear}),
        1);
#endif

#if SZ_USE_CUDA
    // CUDA Levenshtein distance against the serial CPU engine on degenerate inputs.
    check_similarities_fuzzy_<sz_size_t>( //
        make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {unit_uniform, unit_linear}),
        make_pairwise(levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {unit_uniform, unit_linear}),
        degenerate_config, iterations, cuda_executor_t {}, first_gpu_specs);
    check_similarities_degenerate_<sz_size_t>( //
        levenshtein_baselines_t {unit_uniform, unit_linear},
        make_pairwise(levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {unit_uniform, unit_linear}),
        1, cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_KEPLER
    // CUDA Levenshtein distance on Kepler against the serial CPU engine on degenerate inputs.
    check_similarities_degenerate_<sz_size_t>( //
        levenshtein_baselines_t {unit_uniform, unit_linear},
        make_pairwise(levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_caps_ck_k> {unit_uniform, unit_linear}), 1,
        cuda_executor_t {}, first_gpu_specs);
#endif
}

#pragma endregion // Safety

#pragma region Cross Product

/**
 *  @brief Fills the `queries × candidates` @p reference_matrix cell-by-cell from a pairwise @p baseline operator.
 *
 *  The baseline operators score @b paired views (`first[i]` against `second[i]`), so each grid cell is computed by
 *  handing the baseline a one-string sub-view of the query tape and a one-string sub-view of the candidate tape.
 */
template <typename score_type_, typename baseline_operator_>
static void fill_reference_matrix_(baseline_operator_ const &baseline, arrow_strings_view_t queries_view,
                                   arrow_strings_view_t candidates_view, unified_vector<score_type_> &reference_matrix,
                                   std::size_t row_stride) {
    std::size_t const queries_count = queries_view.size();
    std::size_t const candidates_count = candidates_view.size();
    for (std::size_t query_index = 0; query_index != queries_count; ++query_index)
        for (std::size_t candidate_index = 0; candidate_index != candidates_count; ++candidate_index) {
            arrow_strings_view_t const query_cell {queries_view.buffer_, queries_view.offsets_.subspan(query_index, 2)};
            arrow_strings_view_t const candidate_cell {candidates_view.buffer_,
                                                       candidates_view.offsets_.subspan(candidate_index, 2)};
            score_type_ cell_score = 0;
            sz_assert_(baseline(query_cell, candidate_cell, &cell_score) == status_t::success_k);
            reference_matrix[query_index * row_stride + candidate_index] = cell_score;
        }
}

/**
 *  @brief Drives a cross-product engine over @p queries_config × @p candidates_config and asserts every cell of the
 *         `Q × C` matrix against the dual-row @p baseline, across the full set of shapes (1xN, Nx1, square, ragged,
 *         rectangular, empty).
 */
template <typename score_type_, typename engine_type_, typename baseline_operator_, typename... trailing_args_>
static void check_cross_product_cell_exact_(engine_type_ &&engine, baseline_operator_ const &baseline,
                                            fuzzy_config_t queries_config, fuzzy_config_t candidates_config,
                                            trailing_args_ &&...trailing_args) {

    // The Arrow tape cannot represent a zero-string set, so a `batch_size == 0` config is materialized as an
    // empty sub-view sliced (one offset, zero strings) off a one-string fallback tape.
    std::string const empty_fallback_string;
    arrow_strings_tape_t empty_fallback_tape;
    sz_assert_(empty_fallback_tape.try_append({empty_fallback_string.data(), empty_fallback_string.size()}) ==
               status_t::success_k);
    auto build_view = [&](fuzzy_config_t config, std::vector<std::string> &array,
                          arrow_strings_tape_t &tape) -> arrow_strings_view_t {
        if (config.batch_size == 0)
            return arrow_strings_view_t {empty_fallback_tape.view().buffer_,
                                         empty_fallback_tape.view().offsets_.subspan(0, 1)};
        randomize_strings(config, array, tape);
        return tape.view();
    };

    std::vector<std::string> queries_array, candidates_array;
    arrow_strings_tape_t queries_tape, candidates_tape;
    arrow_strings_view_t const queries_view = build_view(queries_config, queries_array, queries_tape);
    arrow_strings_view_t const candidates_view = build_view(candidates_config, candidates_array, candidates_tape);
    std::size_t const queries_count = queries_view.size();
    std::size_t const candidates_count = candidates_view.size();
    std::size_t const row_stride = candidates_count;

    unified_vector<score_type_> engine_matrix(queries_count * candidates_count);
    unified_vector<score_type_> reference_matrix(queries_count * candidates_count);

    strided_rows<score_type_> const results {engine_matrix.data(), queries_count, candidates_count, row_stride};
    status_t const status = engine(queries_view, candidates_view, results, trailing_args...);
    sz_assert_(status == status_t::success_k);

    // The empty shape is fully validated by the success status above - there are no cells to compare.
    if (queries_count == 0 || candidates_count == 0) return;

    fill_reference_matrix_<score_type_>(baseline, queries_view, candidates_view, reference_matrix, row_stride);
    for (std::size_t query_index = 0; query_index != queries_count; ++query_index)
        for (std::size_t candidate_index = 0; candidate_index != candidates_count; ++candidate_index) {
            std::size_t const cell_offset = query_index * row_stride + candidate_index;
            if (engine_matrix[cell_offset] == reference_matrix[cell_offset]) continue;
            edit_distance_log_mismatch(queries_array[query_index], candidates_array[candidate_index],
                                       reference_matrix[cell_offset], engine_matrix[cell_offset]);
            sz_assert_(engine_matrix[cell_offset] == reference_matrix[cell_offset]);
        }
}

/**
 *  @brief Drives the @b symmetric self-similarity engine overload over one set and asserts the matrix is
 *         symmetric with a zero diagonal (the Levenshtein distance of a string to itself is zero).
 */
template <typename engine_type_, typename... trailing_args_>
static void check_symmetric_cell_exact_(engine_type_ &&engine, fuzzy_config_t sequences_config,
                                        trailing_args_ &&...trailing_args) {

    std::vector<std::string> sequences_array;
    arrow_strings_tape_t sequences_tape;
    randomize_strings(sequences_config, sequences_array, sequences_tape);

    arrow_strings_view_t const sequences_view = sequences_tape.view();
    std::size_t const sequences_count = sequences_view.size();
    std::size_t const row_stride = sequences_count;

    unified_vector<sz_size_t> symmetric_matrix(sequences_count * sequences_count);
    strided_rows<sz_size_t> const results {symmetric_matrix.data(), sequences_count, sequences_count, row_stride};
    status_t const status = engine(sequences_view, results, trailing_args...);
    sz_assert_(status == status_t::success_k);

    // The diagonal-is-zero identity holds only for a zero match cost (a string aligned to itself pays nothing).
    // The symmetry identity holds for any cost scheme, so it is always asserted.
    for (std::size_t row_index = 0; row_index != sequences_count; ++row_index) {
        sz_assert_(symmetric_matrix[row_index * row_stride + row_index] == static_cast<sz_size_t>(0));
        for (std::size_t column_index = 0; column_index != sequences_count; ++column_index)
            sz_assert_(symmetric_matrix[row_index * row_stride + column_index] ==
                       symmetric_matrix[column_index * row_stride + row_index]);
    }
}

/**
 *  @brief Validates the cross-product and symmetric engine overloads cell-by-cell across every backend.
 *
 *  Complements the diagonal-only agreement suites: those drive the engines pairwise, whereas this pins the full
 *  `Q x C` matrix and the symmetric one-set matrix directly produced by the new API, asserting each cell against
 *  the dual-row baseline (cross) or the symmetry/zero-diagonal identities (symmetric).
 */
void test_similarities_cross_product() {
    std::printf("  - testing cross-product and symmetric similarity matrices...\n");

    constexpr uniform_substitution_costs_t unit_uniform {0, 1};
    constexpr linear_gap_costs_t unit_linear {1};

    // Non-unit uniform costs (mismatch != 1, gap != 1) skip the Myers fast path and exercise the `u16` candidate-lane
    // batch; lengths stay small so the worst-case distance fits the `u16` reach bound and the long tail is not taken.
    [[maybe_unused]] constexpr uniform_substitution_costs_t nonunit_uniform {0, 2};
    [[maybe_unused]] constexpr linear_gap_costs_t nonunit_linear {3};
    [[maybe_unused]] constexpr affine_gap_costs_t nonunit_affine {3, 1}; // open 3, extend 1

    // High-cost Levenshtein: a short-string reach `(query + candidate) * max(mismatch, gap)` that overflows the `u16`
    // narrow tier (limit 60000) so cells route to the new `u32` WIDE candidate-lane kernel; values still fit `u32`.
    [[maybe_unused]] constexpr uniform_substitution_costs_t wide_uniform {0, 100};
    [[maybe_unused]] constexpr linear_gap_costs_t wide_linear {100};
    [[maybe_unused]] constexpr affine_gap_costs_t wide_affine {100, 50};

    constexpr linear_gap_costs_t blosum62_linear_cost {-4};
    [[maybe_unused]] constexpr affine_gap_costs_t blosum62_affine_cost {-4, -1}; // used only in SIMD #if blocks
    error_costs_32x32_t const blosum62_matrix = error_costs_32x32_t::blosum62();

    // Mixed-length random strings keep the alphabet small so collisions and zero-distance cells appear naturally.
    fuzzy_config_t const single_query {"ABC", /* batch_size */ 1, /* min_string_length */ 1, /* max */ 24};
    fuzzy_config_t const single_candidate {"ABC", /* batch_size */ 1, /* min_string_length */ 1, /* max */ 24};
    fuzzy_config_t const many_queries {"ABC", /* batch_size */ 7, /* min_string_length */ 1, /* max */ 24};
    fuzzy_config_t const many_candidates {"ABC", /* batch_size */ 5, /* min_string_length */ 1, /* max */ 24};
    fuzzy_config_t const square_set {"ABC", /* batch_size */ 6, /* min_string_length */ 0, /* max */ 24};
    fuzzy_config_t const empty_set {"ABC", /* batch_size */ 0, /* min_string_length */ 1, /* max */ 24};

    // Serial Levenshtein cross-product over the full set of shapes: 1xN, Nx1, square, rectangular, ragged, empty.
    auto serial_levenshtein = [&]() {
        return levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {unit_uniform, unit_linear};
    };
    check_cross_product_cell_exact_<sz_size_t>(serial_levenshtein(),
                                               levenshtein_baselines_t {unit_uniform, unit_linear}, single_query,
                                               many_candidates); // 1xN
    check_cross_product_cell_exact_<sz_size_t>(serial_levenshtein(),
                                               levenshtein_baselines_t {unit_uniform, unit_linear}, many_queries,
                                               single_candidate); // Nx1
    check_cross_product_cell_exact_<sz_size_t>(serial_levenshtein(),
                                               levenshtein_baselines_t {unit_uniform, unit_linear}, square_set,
                                               square_set); // square + ragged (min length 0)
    check_cross_product_cell_exact_<sz_size_t>(serial_levenshtein(),
                                               levenshtein_baselines_t {unit_uniform, unit_linear}, many_queries,
                                               many_candidates); // rectangular
    check_cross_product_cell_exact_<sz_size_t>(serial_levenshtein(),
                                               levenshtein_baselines_t {unit_uniform, unit_linear}, empty_set,
                                               many_candidates); // empty queries
    check_cross_product_cell_exact_<sz_size_t>(serial_levenshtein(),
                                               levenshtein_baselines_t {unit_uniform, unit_linear}, many_queries,
                                               empty_set); // empty candidates

    // Serial Needleman-Wunsch and Smith-Waterman cross-products, rectangular shape.
    check_cross_product_cell_exact_<sz_ssize_t>(
        needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost},
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, many_queries, many_candidates);
    check_cross_product_cell_exact_<sz_ssize_t>(
        smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k> {
            blosum62_matrix, blosum62_linear_cost},
        smith_waterman_baselines_t {blosum62_matrix, blosum62_linear_cost}, many_queries, many_candidates);

    // Serial symmetric self-similarity. The diagonal-is-zero identity needs a zero match cost, so both the
    // linear and affine variants keep `unit_uniform` (match 0) while still exercising the affine gap path.
    constexpr affine_gap_costs_t unit_affine {1, 1};
    check_symmetric_cell_exact_(serial_levenshtein(), square_set);
    check_symmetric_cell_exact_(
        levenshtein_distances<affine_gap_costs_t, malloc_t, sz_cap_serial_k> {unit_uniform, unit_affine}, square_set);

#if SZ_USE_ICELAKE
    check_cross_product_cell_exact_<sz_size_t>(
        levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {unit_uniform, unit_linear},
        levenshtein_baselines_t {unit_uniform, unit_linear}, many_queries, many_candidates);
    check_symmetric_cell_exact_(
        levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {unit_uniform, unit_linear}, square_set);

    // Ice Lake NON-UNIT byte Levenshtein: the new `u16` candidate-lane batch (32 lanes). make_pairwise non-unit is
    // 1x1, so it never fills the lanes; these many-candidate rows pin the cost-honoring recurrence cell-by-cell
    // against the serial oracle, and the mixed-length row exercises ragged lane padding within a block.
    {
        fuzzy_config_t const nonunit_queries {"ABC", /* batch */ 4, /* min */ 1, /* max */ 200};
        fuzzy_config_t const nonunit_candidates {"ABC", /* batch */ 40, /* min */ 1, /* max */ 200};
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {nonunit_uniform, nonunit_linear},
            levenshtein_baselines_t {nonunit_uniform, nonunit_linear}, nonunit_queries, nonunit_candidates);
        check_symmetric_cell_exact_(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {nonunit_uniform, nonunit_linear},
            square_set);

        // Ice Lake AFFINE byte Levenshtein: the new `u16` Gotoh E/F candidate-lane batch (32 lanes), cell-by-cell vs
        // the serial affine oracle. Fills the lanes; the symmetric case mirrors the lower triangle.
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sil_k> {nonunit_uniform, nonunit_affine},
            levenshtein_baselines_t {nonunit_uniform, nonunit_affine}, nonunit_queries, nonunit_candidates);
        check_symmetric_cell_exact_(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sil_k> {nonunit_uniform, nonunit_affine},
            square_set);
    }

    check_cross_product_cell_exact_<sz_ssize_t>(
        needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_linear_cost},
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, many_queries, many_candidates);

    // NEW affine candidate-lane (Gotoh E/F) for icelake NW/SW: the equivalence suite is 1x1 (single lane); these
    // multi-cell rows with many candidates per query FILL the 32 i16 lanes, so they pin the affine batch kernel
    // cell-by-cell against the Gotoh baseline. Lengths fit i16 so the INTER candidate-lane path is taken.
    fuzzy_config_t const affine_queries {"ABC", /* batch_size */ 4, /* min */ 1, /* max */ 48};
    fuzzy_config_t const affine_candidates {"ABC", /* batch_size */ 40, /* min */ 1, /* max */ 48};
    check_cross_product_cell_exact_<sz_ssize_t>(
        needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_affine_cost},
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, affine_queries, affine_candidates);
    check_cross_product_cell_exact_<sz_ssize_t>(
        smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k> {
            blosum62_matrix, blosum62_affine_cost},
        smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, affine_queries, affine_candidates);

    // Ice Lake WIDE tier (i32 / u32): long weighted pairs whose `(query+candidate)*magnitude` overflows the `i16`
    // narrow tier route to the new 16-lane i32 NW/SW kernel; high-cost Levenshtein overflows `u16` into the u32
    // kernel. Lengths/costs fill the 16 wide lanes and pin each new kernel cell-by-cell against the serial oracle.
    {
        fuzzy_config_t const wide_nw_queries {"ABC", /* batch */ 2, /* min */ 1500, /* max */ 1500};
        fuzzy_config_t const wide_nw_candidates {"ABC", /* batch */ 18, /* min */ 1500, /* max */ 1500};
        check_cross_product_cell_exact_<sz_ssize_t>(
            needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k> {
                blosum62_matrix, blosum62_linear_cost},
            needleman_wunsch_baselines_t {blosum62_matrix, blosum62_linear_cost}, wide_nw_queries, wide_nw_candidates);
        check_cross_product_cell_exact_<sz_ssize_t>(
            needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k> {
                blosum62_matrix, blosum62_affine_cost},
            needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, wide_nw_queries, wide_nw_candidates);
        check_cross_product_cell_exact_<sz_ssize_t>(
            smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k> {
                blosum62_matrix, blosum62_affine_cost},
            smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, wide_nw_queries, wide_nw_candidates);

        fuzzy_config_t const wide_lev_queries {"ABC", /* batch */ 2, /* min */ 350, /* max */ 350};
        fuzzy_config_t const wide_lev_candidates {"ABC", /* batch */ 20, /* min */ 350, /* max */ 350};
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {wide_uniform, wide_linear},
            levenshtein_baselines_t {wide_uniform, wide_linear}, wide_lev_queries, wide_lev_candidates);
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sil_k> {wide_uniform, wide_affine},
            levenshtein_baselines_t {wide_uniform, wide_affine}, wide_lev_queries, wide_lev_candidates);
    }

    // UTF-8 rune candidate-lane (multi-cell, FILLS the 32 rune lanes) vs the serial UTF-8 engine as oracle. The
    // make_pairwise utf8 suite is 1x1 (one lane), so it never exercises the u32-key compare + lane-mask recombine;
    // these many-candidate rows do. (ASCII alphabet = valid UTF-8, one rune per byte.)
    {
        levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_cap_serial_k> utf8_oracle {};
        auto const utf8_baseline = [&utf8_oracle](arrow_strings_view_t q, arrow_strings_view_t c, sz_size_t *out) {
            strided_rows<sz_size_t> const cell {out, 1, 1, 1};
            return utf8_oracle(q, c, cell);
        };
        fuzzy_config_t const utf8_queries {"ABCD", /* batch_size */ 4, /* min */ 1, /* max */ 48};
        fuzzy_config_t const utf8_candidates {"ABCD", /* batch_size */ 40, /* min */ 1, /* max */ 48};
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {}, utf8_baseline, utf8_queries,
            utf8_candidates);
    }

    // Drive the Ice Lake `distances_8xN_` multi-word Myers (shorter side > 512). The diagonal-only and small-length
    // cross suites never reach it: every pairwise call is a 1x1 grid (one cell, always the lone-cell DP path), and
    // the cross cases above cap lengths at 24. Multi-cell rows of uniform > 512 length make consecutive same-bucket
    // cells group into full 8-lane batches, so these pin the new kernel cell-by-cell against the dual-row baseline.
    fuzzy_config_t const long_uniform_700 {"ABC", /* batch_size */ 8, /* min */ 700, /* max */ 700};
    fuzzy_config_t const long_uniform_2048 {"ABC", /* batch_size */ 8, /* min */ 2048, /* max */ 2048};
    fuzzy_config_t const long_mixed {"ABC", /* batch_size */ 10, /* min */ 1, /* max */ 1100};
    fuzzy_config_t const long_uniform_8192 {"ABC", /* batch_size */ 3, /* min */ 8192, /* max */ 8192};
    auto icelake_levenshtein = [&]() {
        return levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {unit_uniform, unit_linear};
    };
    // Full 8-lane groups at a just-above-512 bucket (words_count == 11).
    check_cross_product_cell_exact_<sz_size_t>(icelake_levenshtein(), levenshtein_baselines_t {unit_uniform, unit_linear},
                                               long_uniform_700, long_uniform_700);
    // Wider multi-word integers (words_count == 32), groups of 8 + a trailing partial group.
    check_cross_product_cell_exact_<sz_size_t>(icelake_levenshtein(), levenshtein_baselines_t {unit_uniform, unit_linear},
                                               long_uniform_2048, long_uniform_2048);
    // Mixed lengths interleave the <= 512 tiers, the > 512 8xN groups, bucket breaks, and the lone-cell DP path.
    check_cross_product_cell_exact_<sz_size_t>(icelake_levenshtein(), levenshtein_baselines_t {unit_uniform, unit_linear},
                                               long_mixed, long_mixed);
    // Beyond the on-stack word cap (words_count == 128 > 64): the kernel returns `bad_alloc_k` and the engine falls
    // back to the anti-diagonal DP, so this pins that defensive path too.
    check_cross_product_cell_exact_<sz_size_t>(icelake_levenshtein(), levenshtein_baselines_t {unit_uniform, unit_linear},
                                               long_uniform_8192, long_uniform_8192);

    // The <= 512 lockstep tiers (`distances_4x128_` / `distances_2x256_`, multiple lanes per pair with cross-lane
    // carry/shift logic) are otherwise only ever reached with a single pair: the pairwise harness drives the engine
    // 1x1, and the small cross cases above cap lengths at 24 (the single-word `distances_8x64_` tier). Uniform rows
    // of mid-tier length make consecutive cells group into multi-pair batches (4 pairs at 128, 2 pairs at 256), so
    // these pin that cross-lane grouping cell-by-cell against the dual-row baseline.
    fuzzy_config_t const mid_uniform_100 {"ABC", /* batch_size */ 8, /* min */ 100, /* max */ 100}; // 4x128, 4-pair
    fuzzy_config_t const mid_uniform_200 {"ABC", /* batch_size */ 8, /* min */ 200, /* max */ 200}; // 2x256, 2-pair
    check_cross_product_cell_exact_<sz_size_t>(icelake_levenshtein(), levenshtein_baselines_t {unit_uniform, unit_linear},
                                               mid_uniform_100, mid_uniform_100);
    check_cross_product_cell_exact_<sz_size_t>(icelake_levenshtein(), levenshtein_baselines_t {unit_uniform, unit_linear},
                                               mid_uniform_200, mid_uniform_200);
    // Mixed exact lengths that share one word bucket: [513, 576] all map to `ceil(shorter / 64) == 9`, so the >512
    // 8xN kernel groups them with a common `words_count`/`last_word` but distinct per-lane `top_bits` - the case the
    // uniform configs above (and the original prototype) never hit. Pins the per-lane top-bit probe under a batched
    // group of unequal lengths.
    fuzzy_config_t const mixed_bucket9 {"ABC", /* batch_size */ 8, /* min */ 513, /* max */ 576};
    check_cross_product_cell_exact_<sz_size_t>(icelake_levenshtein(), levenshtein_baselines_t {unit_uniform, unit_linear},
                                               mixed_bucket9, mixed_bucket9);
#endif

#if SZ_USE_HASWELL
    // Haswell byte Levenshtein 4-lane Myers (multi-cell fills the 4 pair-lanes; the byte-Lev cross cases above use
    // icelake/serial only, and make_pairwise is 1x1). Uniform >512 also exercises distances_4xN_large_.
    {
        auto haswell_levenshtein = [&]() {
            return levenshtein_distances<linear_gap_costs_t, malloc_t,
                                         (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {unit_uniform,
                                                                                                unit_linear};
        };
        levenshtein_baselines_t const lev_base {unit_uniform, unit_linear};
        fuzzy_config_t const hb_small_q {"ABC", /* batch */ 6, /* min */ 1, /* max */ 200};
        fuzzy_config_t const hb_small_c {"ABC", /* batch */ 40, /* min */ 1, /* max */ 200};
        fuzzy_config_t const hb_long {"ABC", /* batch */ 8, /* min */ 700, /* max */ 700};
        check_cross_product_cell_exact_<sz_size_t>(haswell_levenshtein(), lev_base, hb_small_q, hb_small_c);
        check_cross_product_cell_exact_<sz_size_t>(haswell_levenshtein(), lev_base, hb_long, hb_long);

        // Haswell NON-UNIT byte Levenshtein: the new `u16` candidate-lane batch (16 lanes), cell-by-cell vs the
        // serial oracle. Many candidates per query fill the lanes; the symmetric case mirrors the lower triangle.
        auto haswell_nonunit_levenshtein = [&]() {
            return levenshtein_distances<linear_gap_costs_t, malloc_t,
                                         (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {nonunit_uniform,
                                                                                                nonunit_linear};
        };
        levenshtein_baselines_t const nonunit_base {nonunit_uniform, nonunit_linear};
        check_cross_product_cell_exact_<sz_size_t>(haswell_nonunit_levenshtein(), nonunit_base, hb_small_q, hb_small_c);
        check_symmetric_cell_exact_(haswell_nonunit_levenshtein(), square_set);

        // Haswell AFFINE byte Levenshtein: the new `u16` Gotoh E/F candidate-lane batch (16 lanes), cell-by-cell vs
        // the serial affine oracle. Many candidates per query fill the lanes; the symmetric case mirrors the triangle.
        auto haswell_affine_levenshtein = [&]() {
            return levenshtein_distances<affine_gap_costs_t, malloc_t,
                                         (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {nonunit_uniform,
                                                                                                nonunit_affine};
        };
        levenshtein_baselines_t const affine_base {nonunit_uniform, nonunit_affine};
        check_cross_product_cell_exact_<sz_size_t>(haswell_affine_levenshtein(), affine_base, hb_small_q, hb_small_c);
        check_symmetric_cell_exact_(haswell_affine_levenshtein(), square_set);
    }

    // Haswell affine NW/SW candidate-lane at full 16-lane fill (many candidates per query), vs the Gotoh baseline.
    constexpr affine_gap_costs_t haswell_affine_cost {-4, -1};
    fuzzy_config_t const haswell_affine_queries {"ABC", /* batch_size */ 4, /* min */ 1, /* max */ 48};
    fuzzy_config_t const haswell_affine_candidates {"ABC", /* batch_size */ 40, /* min */ 1, /* max */ 48};
    check_cross_product_cell_exact_<sz_ssize_t>(
        needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t,
                                (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {blosum62_matrix,
                                                                                       haswell_affine_cost},
        needleman_wunsch_baselines_t {blosum62_matrix, haswell_affine_cost}, haswell_affine_queries,
        haswell_affine_candidates);
    check_cross_product_cell_exact_<sz_ssize_t>(
        smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t,
                              (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {blosum62_matrix,
                                                                                     haswell_affine_cost},
        smith_waterman_baselines_t {blosum62_matrix, haswell_affine_cost}, haswell_affine_queries,
        haswell_affine_candidates);
    // Haswell UTF-8 rune candidate-lane (multi-cell, fills the 16 rune lanes) vs the serial UTF-8 oracle.
    {
        levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_cap_serial_k> utf8_oracle {};
        auto const utf8_baseline = [&utf8_oracle](arrow_strings_view_t q, arrow_strings_view_t c, sz_size_t *out) {
            strided_rows<sz_size_t> const cell {out, 1, 1, 1};
            return utf8_oracle(q, c, cell);
        };
        fuzzy_config_t const utf8_queries {"ABCD", /* batch_size */ 4, /* min */ 1, /* max */ 48};
        fuzzy_config_t const utf8_candidates {"ABCD", /* batch_size */ 40, /* min */ 1, /* max */ 48};
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances_utf8<linear_gap_costs_t, malloc_t,
                                       (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k)> {},
            utf8_baseline, utf8_queries, utf8_candidates);
    }

    // Haswell WIDE tier (8-lane i32 / u32): overflow the i16/u16 narrow tier so cells route to the new wide kernels.
    {
        constexpr sz_capability_t hcap = (sz_capability_t)(sz_cap_serial_k | sz_cap_haswell_k);
        fuzzy_config_t const wide_nw_q {"ABC", /* batch */ 2, /* min */ 1500, /* max */ 1500};
        fuzzy_config_t const wide_nw_c {"ABC", /* batch */ 12, /* min */ 1500, /* max */ 1500};
        check_cross_product_cell_exact_<sz_ssize_t>(
            needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, hcap> {blosum62_matrix,
                                                                                             blosum62_affine_cost},
            needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, wide_nw_q, wide_nw_c);
        check_cross_product_cell_exact_<sz_ssize_t>(
            smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, hcap> {blosum62_matrix,
                                                                                           blosum62_affine_cost},
            smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, wide_nw_q, wide_nw_c);
        fuzzy_config_t const wide_lev_q {"ABC", /* batch */ 2, /* min */ 350, /* max */ 350};
        fuzzy_config_t const wide_lev_c {"ABC", /* batch */ 16, /* min */ 350, /* max */ 350};
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances<affine_gap_costs_t, malloc_t, hcap> {wide_uniform, wide_affine},
            levenshtein_baselines_t {wide_uniform, wide_affine}, wide_lev_q, wide_lev_c);
    }
#endif

#if SZ_USE_NEON
    // NEON byte Levenshtein 2-lane Myers (multi-cell fills the 2 pair-lanes; uniform >512 hits distances_2xN_large_).
    {
        auto neon_levenshtein = [&]() {
            return levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sn_k> {unit_uniform, unit_linear};
        };
        levenshtein_baselines_t const lev_base {unit_uniform, unit_linear};
        fuzzy_config_t const nb_small_q {"ABC", /* batch */ 6, /* min */ 1, /* max */ 200};
        fuzzy_config_t const nb_small_c {"ABC", /* batch */ 40, /* min */ 1, /* max */ 200};
        fuzzy_config_t const nb_long {"ABC", /* batch */ 4, /* min */ 700, /* max */ 700};
        check_cross_product_cell_exact_<sz_size_t>(neon_levenshtein(), lev_base, nb_small_q, nb_small_c);
        check_cross_product_cell_exact_<sz_size_t>(neon_levenshtein(), lev_base, nb_long, nb_long);

        // NEON NON-UNIT byte Levenshtein: the new `u16` candidate-lane batch (8 lanes), cell-by-cell vs the serial
        // oracle. Many candidates per query fill the 8 lanes; the symmetric case mirrors the lower triangle.
        auto neon_nonunit_levenshtein = [&]() {
            return levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sn_k> {nonunit_uniform, nonunit_linear};
        };
        levenshtein_baselines_t const nonunit_base {nonunit_uniform, nonunit_linear};
        check_cross_product_cell_exact_<sz_size_t>(neon_nonunit_levenshtein(), nonunit_base, nb_small_q, nb_small_c);
        check_symmetric_cell_exact_(neon_nonunit_levenshtein(), square_set);

        // NEON AFFINE byte Levenshtein: the new `u16` Gotoh E/F candidate-lane batch (8 lanes), cell-by-cell vs the
        // serial affine oracle. Fills the 8 lanes; the symmetric case mirrors the lower triangle.
        auto neon_affine_levenshtein = [&]() {
            return levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sn_k> {nonunit_uniform, nonunit_affine};
        };
        levenshtein_baselines_t const affine_base {nonunit_uniform, nonunit_affine};
        check_cross_product_cell_exact_<sz_size_t>(neon_affine_levenshtein(), affine_base, nb_small_q, nb_small_c);
        check_symmetric_cell_exact_(neon_affine_levenshtein(), square_set);
    }

    // NEON affine NW/SW candidate-lane at full 8-lane fill, vs the Gotoh baseline (validated on aarch64 under qemu).
    constexpr affine_gap_costs_t neon_affine_cost {-4, -1};
    fuzzy_config_t const neon_affine_queries {"ABC", /* batch_size */ 4, /* min */ 1, /* max */ 48};
    fuzzy_config_t const neon_affine_candidates {"ABC", /* batch_size */ 40, /* min */ 1, /* max */ 48};
    check_cross_product_cell_exact_<sz_ssize_t>(
        needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sn_k> {blosum62_matrix,
                                                                                                 neon_affine_cost},
        needleman_wunsch_baselines_t {blosum62_matrix, neon_affine_cost}, neon_affine_queries, neon_affine_candidates);
    check_cross_product_cell_exact_<sz_ssize_t>(
        smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sn_k> {blosum62_matrix,
                                                                                               neon_affine_cost},
        smith_waterman_baselines_t {blosum62_matrix, neon_affine_cost}, neon_affine_queries, neon_affine_candidates);
    // NEON UTF-8 rune candidate-lane (multi-cell, fills the 8 rune lanes) vs the serial UTF-8 oracle.
    {
        levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_cap_serial_k> utf8_oracle {};
        auto const utf8_baseline = [&utf8_oracle](arrow_strings_view_t q, arrow_strings_view_t c, sz_size_t *out) {
            strided_rows<sz_size_t> const cell {out, 1, 1, 1};
            return utf8_oracle(q, c, cell);
        };
        fuzzy_config_t const utf8_queries {"ABCD", /* batch_size */ 4, /* min */ 1, /* max */ 48};
        fuzzy_config_t const utf8_candidates {"ABCD", /* batch_size */ 40, /* min */ 1, /* max */ 48};
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_caps_sn_k> {}, utf8_baseline, utf8_queries,
            utf8_candidates);
    }

    // NEON WIDE tier (4-lane i32 / u32): overflow the i16/u16 narrow tier so cells route to the new wide kernels.
    {
        fuzzy_config_t const wide_nw_q {"ABC", /* batch */ 2, /* min */ 1500, /* max */ 1500};
        fuzzy_config_t const wide_nw_c {"ABC", /* batch */ 8, /* min */ 1500, /* max */ 1500};
        check_cross_product_cell_exact_<sz_ssize_t>(
            needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sn_k> {
                blosum62_matrix, blosum62_affine_cost},
            needleman_wunsch_baselines_t {blosum62_matrix, blosum62_affine_cost}, wide_nw_q, wide_nw_c);
        check_cross_product_cell_exact_<sz_ssize_t>(
            smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sn_k> {
                blosum62_matrix, blosum62_affine_cost},
            smith_waterman_baselines_t {blosum62_matrix, blosum62_affine_cost}, wide_nw_q, wide_nw_c);
        fuzzy_config_t const wide_lev_q {"ABC", /* batch */ 2, /* min */ 350, /* max */ 350};
        fuzzy_config_t const wide_lev_c {"ABC", /* batch */ 8, /* min */ 350, /* max */ 350};
        check_cross_product_cell_exact_<sz_size_t>(
            levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sn_k> {wide_uniform, wide_affine},
            levenshtein_baselines_t {wide_uniform, wide_affine}, wide_lev_q, wide_lev_c);
    }
#endif

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs;
    sz_assert_(gpu_specs_fetch(first_gpu_specs) == status_t::success_k);

    // CUDA cross-product and symmetric matrices stay small so device memory remains bounded.
    check_cross_product_cell_exact_<sz_size_t>(
        levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {unit_uniform, unit_linear},
        levenshtein_baselines_t {unit_uniform, unit_linear}, many_queries, many_candidates, cuda_executor_t {},
        first_gpu_specs);
    check_symmetric_cell_exact_(
        levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {unit_uniform, unit_linear}, square_set,
        cuda_executor_t {}, first_gpu_specs);

    // Exercise the per-query Myers `Peq`-reuse fast path: single-word queries (<= 64) against a candidate row
    // wider than a warp (>= 32), non-symmetric - the only shape that triggers the reuse kernel. Candidates may be
    // multi-word (max 96) since only the query bounds the single-word table; min length 0 covers the empty edges.
    fuzzy_config_t const reuse_queries {"ABC", /* batch_size */ 4, /* min_string_length */ 0, /* max */ 24};
    fuzzy_config_t const reuse_candidates {"ABC", /* batch_size */ 40, /* min_string_length */ 0, /* max */ 96};
    check_cross_product_cell_exact_<sz_size_t>(
        levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {unit_uniform, unit_linear},
        levenshtein_baselines_t {unit_uniform, unit_linear}, reuse_queries, reuse_candidates, cuda_executor_t {},
        first_gpu_specs);
#endif
}

#pragma endregion // Cross Product

#pragma region Drivers

/**
 *  @brief @b GPU-oriented memory-scale stress test for the StringZillas similarity engines: closed-form
 *         answers plus an RSS/scratch bound, checking @b memory-usage regressions and scale.
 *
 *  The test is split into two phases following two principles - correctness is verified only where it is
 *  cheap, and scale is verified only against closed-form answers that never touch an O(n²) reference:
 *
 *  1. @b Correctness @b sweep (small/medium, max length <= 8192). Exercises every backend against the
 *     dual-row baselines through the existing `check_similarities_fuzzy_` machinery, exactly as before.
 *     The O(n²)-time reference is affordable at these lengths.
 *
 *  2. @b Scale/memory @b probe (32768 and 131072). Never calls the O(n²) baseline. It verifies the engine
 *     output against closed-form answers computable in O(n) and, for the GPU, asserts the device scratch
 *     stays within a generous @b linear bound observed through a counting allocator. The CPU side has no
 *     allocator hook (its DP scratch lives in per-worker arenas inside `score_in_parallel_`), so it uses a
 *     coarse process-wide peak-RSS backstop instead.
 *
 *  Closed-form cases (simple unary cost scheme, so the answer is provably exact):
 *  - @b string-vs-empty: the only alignment is all-gaps, so the Levenshtein distance is `n * gap` (linear)
 *    or `gap_open + (n-1) * gap_extend` (affine). This is the rock-solid case.
 *  - @b identical-strings: the Levenshtein distance is `0`.
 *  - @b one-planted-edit: copy a string and flip one character; unit-cost Levenshtein distance is `1`.
 *  Each formula is validated at n=512 against the dual-row baseline (`levenshtein_baseline` /
 *  `levenshtein_gotoh_baseline`) before being trusted at 32768/131072 without paying O(n²) at scale.
 */
void test_similarities_memory_usage() {

    std::vector<fuzzy_config_t> correctness_experiments {
        // Single string pair of same length:
        {"ABC", /* batch_size */ 1, /* min_string_length */ 128, /* max_string_length */ 128},
        {"ABC", /* batch_size */ 1, /* min_string_length */ 512, /* max_string_length */ 512},
        {"ABC", /* batch_size */ 1, /* min_string_length */ 2048, /* max_string_length */ 2048},
        {"ABC", /* batch_size */ 1, /* min_string_length */ 8192, /* max_string_length */ 8192},
        // Two strings of a same length:
        {"ABC", /* batch_size */ 2, /* min_string_length */ 128, /* max_string_length */ 128},
        {"ABC", /* batch_size */ 2, /* min_string_length */ 512, /* max_string_length */ 512},
        {"ABC", /* batch_size */ 2, /* min_string_length */ 2048, /* max_string_length */ 2048},
        {"ABC", /* batch_size */ 2, /* min_string_length */ 8192, /* max_string_length */ 8192},
        // Ten strings of random lengths (mixed-length, GPU-safe range):
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 128},
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 512},
        {"ABC", /* batch_size */ 10, /* min_string_length */ 1, /* max_string_length */ 2048},
        // Ten strings of a same larger length, to reach 8192 across every backend including the GPU:
        {"ABC", /* batch_size */ 10, /* min_string_length */ 8192, /* max_string_length */ 8192},
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
    for (fuzzy_config_t const &experiment : correctness_experiments) {
        std::printf("Correctness sweep: batch size %zu, min length %zu, max length %zu\n", experiment.batch_size,
                    experiment.min_string_length, experiment.max_string_length);

        // Multi-threaded serial Levenshtein distance implementation
        check_similarities_fuzzy_<sz_size_t>( //
            levenshtein_baselines_t {},       //
            make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}), experiment, 1);

        // Multi-threaded serial Levenshtein distance implementation with weird linear costs
        check_similarities_fuzzy_<sz_size_t>(                      //
            levenshtein_baselines_t {weird_uniform, weird_linear}, //
            make_pairwise(
                levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}),
            experiment, 1);

        // Multi-threaded serial Levenshtein distance implementation with weird affine costs
        check_similarities_fuzzy_<sz_size_t>(                      //
            levenshtein_baselines_t {weird_uniform, weird_affine}, //
            make_pairwise(
                levenshtein_distances<affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}),
            experiment, 1);

#if SZ_USE_ICELAKE
        // Ice Lake Levenshtein distance against Multi-threaded on CPU
        check_similarities_fuzzy_<sz_size_t>( //
            make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}),
            make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {}), experiment, 1);

        // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird linear costs
        check_similarities_fuzzy_<sz_size_t>( //
            make_pairwise(
                levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_linear}),
            make_pairwise(
                levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_linear}),
            experiment, 1);

        // Ice Lake Levenshtein distance against Multi-threaded on CPU with weird affine costs
        check_similarities_fuzzy_<sz_size_t>( //
            make_pairwise(
                levenshtein_distances<affine_gap_costs_t, malloc_t, sz_cap_serial_k> {weird_uniform, weird_affine}),
            make_pairwise(
                levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sil_k> {weird_uniform, weird_affine}),
            experiment, 1);
#endif

#if SZ_USE_CUDA
        // CUDA Levenshtein distance against Multi-threaded on CPU
        check_similarities_fuzzy_<sz_size_t>(                                                       //
            make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}), //
            make_pairwise(levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k> {}), experiment, 10,
            cuda_executor_t {}, first_gpu_specs);
#endif

#if SZ_USE_KEPLER
        // CUDA Levenshtein distance on Kepler against Multi-threaded on CPU
        check_similarities_fuzzy_<sz_size_t>(                                                       //
            make_pairwise(levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k> {}), //
            make_pairwise(levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_caps_ck_k> {}), experiment, 10,
            cuda_executor_t {}, first_gpu_specs);
#endif
    }
}

#pragma endregion // Drivers

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
