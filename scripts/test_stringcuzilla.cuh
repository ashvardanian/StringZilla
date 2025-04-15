/**
 *  @brief   Extensive @b stress-testing suite for StringCuZilla parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test_stringcuzilla.cuh
 *  @author  Ash Vardanian
 */
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

using malloc_t = std::allocator<char>;

/**
 *  In non-SIMD backends we still leverage OpenMP for parallelism.
 */
using levenshtein_serial_t = levenshtein_distances<char, malloc_t, sz_caps_sp_k>;
using levenshtein_utf8_serial_t = levenshtein_distances_utf8<char, malloc_t, sz_caps_sp_k>;
using needleman_wunsch_serial_t = needleman_wunsch_scores<char, error_costs_256x256_t, malloc_t, sz_caps_sp_k>;
using smith_waterman_serial_t = smith_waterman_scores<char, error_costs_256x256_t, malloc_t, sz_caps_sp_k>;

/**
 *  In @b AVX-512:
 *  - for Global Alignments, we can vectorize the min-max calculation for diagonal "walkers"
 *  - for Local Alignments, we can vectorize the character substitution lookups for horizontal "walkers"
 */
using levenshtein_ice_t = levenshtein_distances<char, malloc_t, sz_caps_spi_k>;
using levenshtein_utf8_ice_t = levenshtein_distances_utf8<char, malloc_t, sz_caps_spi_k>;
using needleman_wunsch_ice_t = needleman_wunsch_scores<char, error_costs_256x256_t, malloc_t, sz_caps_spi_k>;
using smith_waterman_ice_t = smith_waterman_scores<char, error_costs_256x256_t, malloc_t, sz_caps_spi_k>;

/**
 *  In @b CUDA:
 *  - for GPUs before Hopper, we can use the @b SIMT model for warp-level parallelism using diagonal "walkers"
 *  - for GPUs after Hopper, we compound that with thread-level @b SIMD via @b DPX instructions for min-max
 */
using levenshtein_cuda_t = levenshtein_distances<char, dummy_alloc_t, sz_cap_cuda_k>;
using levenshtein_utf8_cuda_t = levenshtein_distances_utf8<char, dummy_alloc_t, sz_cap_cuda_k>;
using needleman_wunsch_cuda_t = needleman_wunsch_scores<char, error_costs_256x256_t, dummy_alloc_t, sz_cap_cuda_k>;
using smith_waterman_cuda_t = smith_waterman_scores<char, error_costs_256x256_t, dummy_alloc_t, sz_cap_cuda_k>;

using levenshtein_hopper_t = levenshtein_distances<char, dummy_alloc_t, sz_caps_ckh_k>;
using levenshtein_utf8_hopper_t = levenshtein_distances_utf8<char, dummy_alloc_t, sz_caps_ckh_k>;
using needleman_wunsch_hopper_t = needleman_wunsch_scores<char, error_costs_256x256_t, dummy_alloc_t, sz_caps_ckh_k>;
using smith_waterman_hopper_t = smith_waterman_scores<char, error_costs_256x256_t, dummy_alloc_t, sz_caps_ckh_k>;

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
    append("ABC", "ABC");                  // same string; distance ~ 0
    append("LISTEN", "SILENT");            // distance ~ 4
    append("ATCA", "CTACTCACCC");          // distance ~ 6
    append("A", "=");                      // distance ~ 1
    append("A", "A");                      // distance ~ 0
    append("", "");                        // distance ~ 0
    append("", "ABC");                     // distance ~ 3
    append("ABC", "");                     // distance ~ 3
    append("ABC", "AC");                   // one deletion; distance ~ 1
    append("ABC", "A_BC");                 // one insertion; distance ~ 1
    append("ggbuzgjux{}l", "gbuzgjux{}l"); // one (prepended) insertion; distance ~ 1
    append("ABC", "ADC");                  // one substitution; distance ~ 1
    append("APPLE", "APLE");               // distance ~ 1

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
        results_base[0] = 0, results_simd[0] = 0;
        first_tape.try_assign(&first, &first + 1);
        second_tape.try_assign(&second, &second + 1);

        // Compute with both backends
        status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data(), extra_args...);
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
    std::size_t iterations = 10;
};

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads
 *          on a synthetic @b randomly-generated set of strings from a given @p alphabet.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_similarity_scores_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                  fuzzy_config_t config = {}, extra_args_ &&...extra_args) {

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
    test_similarity_scores_fuzzy<score_type_>(base_operator, simd_operator, config, extra_args...);
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
        levenshtein_distances<char, malloc_t, sz_cap_serial_k> {});

    // Multi-threaded parallel Levenshtein distance implementation
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>( //
        levenshtein_baselines_t {},                    //
        levenshtein_distances<char, malloc_t, sz_caps_sp_k> {});

    // Now let's take non-unary substitution costs, like BLOSUM62
    constexpr error_t blosum62_gap_extension_cost = -4;
    error_mat_t blosum62_mat = error_costs_26x26ascii_t::blosum62();
    error_matrix_t blosum62_matrix = blosum62_mat.decompressed();

    // Single-threaded serial NW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                              //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, malloc_t, sz_cap_serial_k> {blosum62_matrix,
                                                                                  blosum62_gap_extension_cost});

    // Multi-threaded parallel NW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                              //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, malloc_t, sz_caps_sp_k> {blosum62_matrix,
                                                                               blosum62_gap_extension_cost});

    // Single-threaded serial SW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                            //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        smith_waterman_scores<char, error_matrix_t, malloc_t, sz_cap_serial_k> {blosum62_matrix,
                                                                                blosum62_gap_extension_cost});

    // Multi-threaded parallel SW implementation
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                            //
        smith_waterman_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        smith_waterman_scores<char, error_matrix_t, malloc_t, sz_caps_sp_k> {blosum62_matrix,
                                                                             blosum62_gap_extension_cost});

#if SZ_USE_ICE
    // Ice Lake Levenshtein distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(          //
        levenshtein_distances<char, malloc_t, sz_caps_sp_k> {}, //
        levenshtein_distances<char, malloc_t, sz_caps_spi_k> {});

    // Ice Lake Levenshtein distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(               //
        levenshtein_distances_utf8<char, malloc_t, sz_caps_sp_k> {}, //
        levenshtein_distances_utf8<char, malloc_t, sz_caps_spi_k> {});

    // Ice Lake Needleman-Wunsch distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>(                              //
        needleman_wunsch_baselines_t {blosum62_matrix, blosum62_gap_extension_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, malloc_t, sz_caps_spi_k> {blosum62_matrix,
                                                                                blosum62_gap_extension_cost});

#endif

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs = *gpu_specs();
#endif

#if SZ_USE_CUDA
    // CUDA Levenshtein distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(          //
        levenshtein_distances<char, malloc_t, sz_caps_sp_k> {}, //
        levenshtein_distances<char, dummy_alloc_t, sz_cap_cuda_k> {}, {}, {}, first_gpu_specs);
#endif

#if SZ_USE_HOPPER && 0
    // CUDA Levenshtein distance on Hopper against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_size_t>(          //
        levenshtein_distances<char, malloc_t, sz_caps_sp_k> {}, //
        levenshtein_distances<char, dummy_alloc_t, sz_cap_hopper_k> {});
#endif

#if SZ_USE_CUDA
    // CUDA Needleman-Wunsch distance against Multi-threaded on CPU
    test_similarity_scores_fixed_and_fuzzy<sz_ssize_t>( //
        needleman_wunsch_scores<char, error_matrix_t, malloc_t, sz_caps_sp_k> {blosum62_matrix,
                                                                               blosum62_gap_extension_cost}, //
        needleman_wunsch_scores<char, error_matrix_t, dummy_alloc_t, sz_cap_cuda_k> {blosum62_matrix,
                                                                                     blosum62_gap_extension_cost},
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

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs = *gpu_specs();
#endif

    // Progress until something fails
    for (fuzzy_config_t const &experiment : experiments) {
        std::printf("Testing with batch size %zu, min length %zu, max length %zu, iterations %zu\n",
                    experiment.batch_size, experiment.min_string_length, experiment.max_string_length,
                    experiment.iterations);

        // Multi-threaded serial Levenshtein distance implementation
        test_similarity_scores_fuzzy<sz_size_t>( //
            levenshtein_baselines_t {},          //
            levenshtein_distances<char, malloc_t, sz_caps_sp_k> {}, experiment);

#if SZ_USE_CUDA
        // CUDA Levenshtein distance against Multi-threaded on CPU
        test_similarity_scores_fuzzy<sz_size_t>(                    //
            levenshtein_distances<char, malloc_t, sz_caps_sp_k> {}, //
            levenshtein_distances<char, dummy_alloc_t, sz_cap_cuda_k> {}, experiment, first_gpu_specs);
#endif
    }
}

struct find_many_baselines_t {
    using state_id_t = sz_u32_t;
    using match_t = find_many_match_t;

    arrow_strings_tape_t needles_;

    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles) noexcept {
        return needles_.try_assign(needles.begin(), needles.end());
    }

    void reset() noexcept { needles_.reset(); }

    template <typename haystacks_type_, typename needles_type_, typename match_callback_type_>
    void iterate_through_unsorted_matches(haystacks_type_ &&haystacks, needles_type_ &&needles,
                                          match_callback_type_ &&callback) const noexcept {
        for (std::size_t i = 0; i != haystacks.size(); ++i) {
            auto const &haystack = haystacks[i];
            for (std::size_t j = 0; j != needles.size(); ++j) {
                auto const &needle = needles[j];
                // Define iterators for the current haystack and the needle.
                auto haystack_begin = haystack.begin();
                auto haystack_end = haystack.end();
                auto needle_begin = needle.begin();
                auto needle_end = needle.end();

                // Use `std::search` to find all occurrences of needle in haystack.
                while (true) {
                    auto it = std::search(haystack_begin, haystack_end, needle_begin, needle_end);
                    if (it == haystack_end) break;

                    // Compute the starting index of the found occurrence.
                    std::size_t found = static_cast<std::size_t>(std::distance(haystack.begin(), it));

                    // Construct a match record.
                    match_t match;
                    match.haystack_index = i;
                    match.needle_index = j;
                    match.haystack = {haystack.data(), haystack.size()};
                    match.needle = {haystack.data() + found, needle.size()};

                    // Invoke the callback. If it returns false, abort all further processing.
                    if (!callback(match)) return;

                    // Advance the starting iterator for the next search.
                    haystack_begin = it + 1;
                }
            }
        }
    }

    template <typename haystacks_type_>
    size_t count(haystacks_type_ &&haystacks, span<size_t> counts) const noexcept {
        size_t count_total = 0;
        for (size_t &count : counts) count = 0;
        iterate_through_unsorted_matches(haystacks, needles_, [&](match_t const &match) {
            counts[match.haystack_index] += 1;
            count_total += 1;
            return true;
        });
        return count_total;
    }

    template <typename haystacks_type_, typename output_matches_type_>
    size_t find(haystacks_type_ &&haystacks, output_matches_type_ &&matches) const noexcept {
        size_t count_found = 0, count_allowed = matches.size();
        iterate_through_unsorted_matches(haystacks, needles_, [&](match_t const &match) {
            matches[count_found] = match;
            count_found += 1;
            return count_found < count_allowed;
        });
        return count_found;
    }
};

using find_many_serial_t = find_many<sz_u32_t, malloc_t, sz_cap_serial_k>;
using find_many_parallel_t = find_many<sz_u32_t, malloc_t, sz_caps_sp_k>;

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

    // Haystacks should contain arbitrary strings including those needles
    // in different positions, potentially interleaving
    haystacks.emplace_back("That is a test string"); // ? "only "is"
    haystacks.emplace_back("This is a test string"); // ? "his", 2x "is"

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
        size_t total_found_base = base_operator.count(haystacks_tape, counts_base_span);
        size_t total_found_simd = simd_operator.count(haystacks_tape, counts_simd_span, extra_args...);
        _sz_assert(total_found_base == total_found_simd);
        _sz_assert(counts_base[0] == counts_simd[0]);

        // Check the matches themselves
        matches_base.resize(total_found_base);
        matches_simd.resize(total_found_simd);
        size_t total_matched_base = base_operator.find(haystacks_tape, matches_base);
        size_t total_matched_simd = simd_operator.find(haystacks_tape, matches_simd, extra_args...);
        _sz_assert(total_matched_base == total_matched_simd);

        // Check the contents and order of the matches
        for (std::size_t i = 0; i != total_matched_base; ++i) {
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

        // Count with both backends
        span<size_t> counts_base_span {counts_base.data(), counts_base.size()};
        span<size_t> counts_simd_span {counts_simd.data(), counts_simd.size()};
        size_t total_found_base = base_operator.count(haystacks_tape, counts_base_span);
        size_t total_found_simd = simd_operator.count(haystacks_tape, counts_simd_span, extra_args...);
        _sz_assert(total_found_base == total_found_simd);

        // Check the matches themselves
        matches_base.resize(total_found_base);
        matches_simd.resize(total_found_simd);
        size_t total_matched_base = base_operator.find(haystacks_tape, matches_base);
        size_t total_matched_simd = simd_operator.find(haystacks_tape, matches_simd, extra_args...);
        _sz_assert(total_matched_base == total_matched_simd);

        // Check the contents and order of the matches
        for (std::size_t i = 0; i != total_matched_base; ++i) {
            _sz_assert(matches_base[i].haystack.data() == matches_simd[i].haystack.data());
            _sz_assert(matches_base[i].needle.data() == matches_simd[i].needle.data());
            _sz_assert(matches_base[i].needle_index == matches_simd[i].needle_index);
        }
    }
}

/**
 *  @brief  Tests the multi-pattern exact substring search algorithm
 *          against a baseline implementation for predefined and random inputs.
 */
void test_find_many_equivalence() {

    // Single-threaded serial Levenshtein distance implementation
    test_find_many_fixed(find_many_baselines_t {}, find_many_serial_t {});

    // Multi-threaded parallel Levenshtein distance implementation
    test_find_many_fixed(find_many_baselines_t {}, find_many_parallel_t {});
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
