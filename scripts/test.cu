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
#include <stringcuzilla/similarity.hpp>

#if SZ_USE_CUDA
#include <stringcuzilla/similarity.cuh>
#endif

#if !_SZ_IS_CPP17
#error "This test requires C++17 or later."
#endif

#include "test.hpp" // `levenshtein_baseline`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using namespace std::literals; // for ""sv

using arrow_strings_view_t = sz::arrow_strings_view<char, sz_size_t>;

#if !SZ_USE_CUDA
using arrow_strings_tape_t = sz::arrow_strings_tape<char, sz_size_t, std::allocator<char>>;
template <typename value_type_>
using unified_vector = std::vector<value_type_, std::allocator<value_type_>>;
#else
using arrow_strings_tape_t = sz::arrow_strings_tape<char, sz_size_t, sz::unified_alloc<char>>;
template <typename value_type_>
using unified_vector = std::vector<value_type_, sz::unified_alloc<value_type_>>;
#endif

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
 *          as well as the similarity scoring functions for bioinformatics-like workloads.
 */
template <typename score_type_, typename base_operator_, typename simd_operator_>
static void edit_distances_compare(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                                   std::size_t batch_size = 1024 * 16, std::size_t max_string_length = 512) {

    using score_t = score_type_;

    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"ABC", "ABC"},                  // same string; distance ~ 0
        {"listen", "silent"},            // distance ~ 4
        {"atca", "ctactcaccc"},          // distance ~ 6
        {"A", "="},                      // distance ~ 1
        {"a", "a"},                      // distance ~ 0
        {"", ""},                        // distance ~ 0
        {"", "abc"},                     // distance ~ 3
        {"abc", ""},                     // distance ~ 3
        {"abc", "ac"},                   // one deletion; distance ~ 1
        {"abc", "a_bc"},                 // one insertion; distance ~ 1
        {"ggbuzgjux{}l", "gbuzgjux{}l"}, // one (prepended) insertion; distance ~ 1
        {"abc", "adc"},                  // one substitution; distance ~ 1
        {"apple", "aple"},               // distance ~ 1
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
    unified_vector<score_t> results_base(1), results_simd(1);
    arrow_strings_tape_t first_tape, second_tape;
    for (auto [first, second] : test_cases) {

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

    // Generate some random strings, using a small alphabet
    for (std::size_t iteration_idx = 0; iteration_idx < 10; ++iteration_idx) {
        std::vector<std::string> first_array(batch_size), second_array(batch_size);
        for (std::size_t i = 0; i != batch_size; ++i) {
            std::size_t first_length = 1u + std::rand() % max_string_length;
            std::size_t second_length = 1u + std::rand() % max_string_length;
            first_array[i] = random_string(first_length, "abc", 3);
            second_array[i] = random_string(second_length, "abc", 3);
        }

        // Convert to a GPU-friendly layout
        first_tape.try_assign(first_array.data(), first_array.data() + batch_size);
        second_tape.try_assign(second_array.data(), second_array.data() + batch_size);
        results_base.resize(batch_size);
        results_simd.resize(batch_size);

        // Compute with both backends
        sz::status_t status_base = base_operator(first_tape.view(), second_tape.view(), results_base.data());
        sz::status_t status_simd = simd_operator(first_tape.view(), second_tape.view(), results_simd.data());
        _sz_assert(status_base == sz::status_t::success_k);
        _sz_assert(status_simd == sz::status_t::success_k);

        // Individually log the failed results
        for (std::size_t i = 0; i != test_cases.size(); ++i) {
            if (results_base[i] == results_simd[i]) continue;
            edit_distance_log_mismatch(first_array[i], second_array[i], results_base[i], results_simd[i]);
        }
    }
}

static void test_equivalence(std::size_t batch_size = 1024, std::size_t max_string_length = 100) {

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

    // Single-threaded serial Levenshtein distance implementation
    edit_distances_compare<sz_size_t>(                                      //
        levenshtein_baselines_t {},                                         //
        sz::levenshtein_distances<serial_k, char, std::allocator<char>> {}, //
        batch_size, max_string_length);

    // Multi-threaded parallel Levenshtein distance implementation
    edit_distances_compare<sz_size_t>(                                        //
        levenshtein_baselines_t {},                                           //
        sz::levenshtein_distances<parallel_k, char, std::allocator<char>> {}, //
        batch_size, max_string_length);

    // Now let's take non-unary substitution costs, like BLOSUM62
    constexpr error_t blosum62_gap_extension_cost = 4; // ? The inverted typical (-4) value
    error_matrix_t blosum62 = sz::error_costs_26x26ascii_t::blosum62().decompressed();

    // Single-threaded serial NW implementation
    edit_distances_compare<sz_ssize_t>(                                       //
        needleman_wunsch_baselines_t {blosum62, blosum62_gap_extension_cost}, //
        sz::needleman_wunsch_scores<serial_k, char, error_matrix_t, std::allocator<char>> {
            blosum62, blosum62_gap_extension_cost}, //
        batch_size, max_string_length);

    // Multi-threaded parallel NW implementation
    edit_distances_compare<sz_ssize_t>(                                       //
        needleman_wunsch_baselines_t {blosum62, blosum62_gap_extension_cost}, //
        sz::needleman_wunsch_scores<parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62, blosum62_gap_extension_cost}, //
        batch_size, max_string_length);

    // Single-threaded serial SW implementation
    edit_distances_compare<sz_ssize_t>(                                     //
        smith_waterman_baselines_t {blosum62, blosum62_gap_extension_cost}, //
        sz::smith_waterman_scores<serial_k, char, error_matrix_t, std::allocator<char>> {
            blosum62, blosum62_gap_extension_cost}, //
        batch_size, max_string_length);

    // Multi-threaded parallel SW implementation
    edit_distances_compare<sz_ssize_t>(                                     //
        smith_waterman_baselines_t {blosum62, blosum62_gap_extension_cost}, //
        sz::smith_waterman_scores<parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62, blosum62_gap_extension_cost}, //
        batch_size, max_string_length);

    // Switch to the GPU, using an identical matrix, but move it into unified memory
    unified_vector<error_matrix_t> blosum62_unified(1);
    blosum62_unified[0] = blosum62;

    // CUDA Levenshtein distance against Multi-threaded on CPU
    edit_distances_compare<sz_size_t>(                                        //
        sz::levenshtein_distances<parallel_k, char, std::allocator<char>> {}, //
        sz::levenshtein_distances<cuda_k, char> {},                           //
        batch_size, max_string_length);

    // CUDA Needleman-Wunsch distance against Multi-threaded on CPU
    edit_distances_compare<sz_ssize_t>( //
        sz::needleman_wunsch_scores<parallel_k, char, error_matrix_t, std::allocator<char>> {
            blosum62, blosum62_gap_extension_cost}, //
        sz::needleman_wunsch_scores<cuda_k, char, error_matrix_t *> {blosum62_unified.data(),
                                                                     blosum62_gap_extension_cost},
        batch_size, max_string_length);
};

#if 0
/**
 *  @brief  Invokes different C++ member methods of immutable strings to cover
 *          extensions beyond the STL API.
 */
template <typename string_type>
static void test_non_stl_extensions_for_reads() {
    using str = string_type;

    // Computing edit-distances.
    _sz_assert(sz::hamming_distance(str("hello"), str("hello")) == 0);
    _sz_assert(sz::hamming_distance(str("hello"), str("hell")) == 1);
    _sz_assert(sz::hamming_distance(str("abc"), str("adc")) == 1);                // one substitution
    _sz_assert(sz::hamming_distance(str("αβγδ"), str("αxxγδ")) == 2);             // replace Beta UTF8 codepoint
    _sz_assert(sz::hamming_distance_utf8(str("abcdefgh"), str("_bcdefg_")) == 2); // replace ASCI prefix and suffix
    _sz_assert(sz::hamming_distance_utf8(str("αβγδ"), str("αγγδ")) == 1);         // replace Beta UTF8 codepoint

    _sz_assert(sz::levenshtein_distance(str("hello"), str("hello")) == 0);
    _sz_assert(sz::levenshtein_distance(str("hello"), str("hell")) == 1);
    _sz_assert(sz::levenshtein_distance(str(""), str("")) == 0);
    _sz_assert(sz::levenshtein_distance(str(""), str("abc")) == 3);
    _sz_assert(sz::levenshtein_distance(str("abc"), str("")) == 3);
    _sz_assert(sz::levenshtein_distance(str("abc"), str("ac")) == 1);                   // one deletion
    _sz_assert(sz::levenshtein_distance(str("abc"), str("a_bc")) == 1);                 // one insertion
    _sz_assert(sz::levenshtein_distance(str("abc"), str("adc")) == 1);                  // one substitution
    _sz_assert(sz::levenshtein_distance(str("ggbuzgjux{}l"), str("gbuzgjux{}l")) == 1); // one insertion (prepended)
    _sz_assert(sz::levenshtein_distance(str("abcdefgABCDEFG"), str("ABCDEFGabcdefg")) == 14);

    _sz_assert(sz::levenshtein_distance_utf8(str("hello"), str("hell")) == 1);           // no unicode symbols, just ASCII
    _sz_assert(sz::levenshtein_distance_utf8(str("𠜎 𠜱 𠝹 𠱓"), str("𠜎𠜱𠝹𠱓")) == 3); // add 3 whitespaces in Chinese
    _sz_assert(sz::levenshtein_distance_utf8(str("💖"), str("💗")) == 1);

    _sz_assert(sz::levenshtein_distance_utf8(str("αβγδ"), str("αγδ")) == 1); // insert Beta
    _sz_assert(sz::levenshtein_distance_utf8(str("école"), str("école")) ==
           2); // etter "é" as a single character vs "e" + "´"
    _sz_assert(sz::levenshtein_distance_utf8(str("façade"), str("facade")) == 1);     // "ç" with cedilla vs. plain
    _sz_assert(sz::levenshtein_distance_utf8(str("Schön"), str("Scho\u0308n")) == 2); // "ö" represented as "o" + "¨"
    _sz_assert(sz::levenshtein_distance_utf8(str("München"), str("Muenchen")) == 2); // German with umlaut vs. transcription
    _sz_assert(sz::levenshtein_distance_utf8(str("こんにちは世界"), str("こんばんは世界")) == 2);

    // Computing alignment scores.
    using matrix_t = std::int8_t[256][256];
    sz::error_costs_256x256_t substitution_costs = error_costs_256x256_diagonal();
    matrix_t &costs = *reinterpret_cast<matrix_t *>(substitution_costs.data());

    _sz_assert(sz::alignment_score(str("listen"), str("silent"), costs, -1) == -4);
    _sz_assert(sz::alignment_score(str("abcdefgABCDEFG"), str("ABCDEFGabcdefg"), costs, -1) == -14);
    _sz_assert(sz::alignment_score(str("hello"), str("hello"), costs, -1) == 0);
    _sz_assert(sz::alignment_score(str("hello"), str("hell"), costs, -1) == -1);

    // Computing rolling fingerprints.
    _sz_assert(sz::hashes_fingerprint<512>(str("aaaa"), 3).count() == 1);
    _sz_assert(sz::hashes_fingerprint<512>(str("hello"), 4).count() == 2);
    _sz_assert(sz::hashes_fingerprint<512>(str("hello"), 3).count() == 3);

    // No matter how many times one repeats a character, the hash should only contain at most one set bit.
    _sz_assert(sz::hashes_fingerprint<512>(str("a"), 3).count() == 0);
    _sz_assert(sz::hashes_fingerprint<512>(str("aa"), 3).count() == 0);
    _sz_assert(sz::hashes_fingerprint<512>(str("aaa"), 3).count() == 1);
    _sz_assert(sz::hashes_fingerprint<512>(str("aaaa"), 3).count() == 1);
    _sz_assert(sz::hashes_fingerprint<512>(str("aaaaa"), 3).count() == 1);

    // Computing fuzzy search results.
}
#endif

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

    std::printf("All tests passed... Unbelievable!\n");
    return 0;
}
