/**
 *  @brief  Helper structures and functions for C++ unit- and stress-tests.
 *  @file   test.hpp
 *  @author Ash Vardanian
 */
#pragma once
#include <fstream>    // `std::ifstream`
#include <iostream>   // `std::cout`, `std::endl`
#include <random>     // `std::random_device`
#include <string>     // `std::string`
#include <vector>     // `std::vector`
#include <array>      // `std::array`
#include <functional> // `std::function`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

inline std::string read_file(std::string path) noexcept(false) {
    std::ifstream stream(path);
    if (!stream.is_open()) throw std::runtime_error("Failed to open file: " + path);
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

inline void write_file(std::string path, std::string content) noexcept(false) {
    std::ofstream stream(path);
    if (!stream.is_open()) throw std::runtime_error("Failed to open file: " + path);
    stream << content;
    stream.close();
}

inline std::mt19937 &global_random_generator() noexcept {
    static std::random_device seed_source; // Too expensive to construct every time
    static std::mt19937 generator(seed_source());
    return generator;
}

/**
 *  @brief  A uniform distribution of characters, with a given alphabet size.
 *          The alphabet size is the number of distinct characters in the distribution.
 *
 *  We can't use `std::uniform_int_distribution<char>` because `char` overload is not supported by some platforms.
 *  MSVC, for example, requires one of `short`, `int`, `long`, `long long`, `unsigned short`, `unsigned int`,
 *  `unsigned long`, or `unsigned long long`.
 */
struct uniform_u8_distribution_t {
    std::uniform_int_distribution<std::uint32_t> distribution;

    inline uniform_u8_distribution_t(std::size_t alphabet_size = 255)
        : distribution(1, static_cast<std::uint32_t>(alphabet_size)) {}
    inline uniform_u8_distribution_t(char from, char to)
        : distribution(static_cast<std::uint32_t>(from), static_cast<std::uint32_t>(to)) {}

    template <typename generator_type_>
    std::uint8_t operator()(generator_type_ &&generator) noexcept {
        return static_cast<std::uint8_t>(distribution(generator));
    }
};

inline void randomize_string(char *string, std::size_t length, char const *alphabet, std::size_t cardinality) noexcept {
    uniform_u8_distribution_t distribution(0, cardinality - 1);
    std::generate(string, string + length, [&]() -> char { return alphabet[distribution(global_random_generator())]; });
}

inline void randomize_string(char *string, std::size_t length) noexcept {
    uniform_u8_distribution_t distribution;
    std::generate(string, string + length, [&]() -> char { return distribution(global_random_generator()); });
}

inline std::string random_string(std::size_t length, char const *alphabet, std::size_t cardinality) noexcept(false) {
    std::string result(length, '\0');
    randomize_string(&result[0], length, alphabet, cardinality);
    return result;
}

inline std::string repeat(std::string const &patten, std::size_t count) noexcept(false) {
    std::string result(patten.size() * count, '\0');
    for (std::size_t i = 0; i < count; ++i) std::copy(patten.begin(), patten.end(), result.begin() + i * patten.size());
    return result;
}

/**
 *  @brief Randomly slices a string into consecutive parts and passes those to @p slice_callback.
 *  @warning Is @b single-threaded in nature, as it depends on the `global_random_generator`.
 */
template <typename slice_callback_type_>
inline void iterate_in_random_slices(std::string const &text, slice_callback_type_ &&slice_callback) noexcept {
    std::size_t remaining = text.size();
    while (remaining > 0) {
        std::uniform_int_distribution<std::size_t> slice_length_distribution(1, remaining);
        std::size_t slice_length = slice_length_distribution(global_random_generator());
        slice_callback({text.data() + text.size() - remaining, slice_length});
        remaining -= slice_length;
    }
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
            std::ptrdiff_t if_deletion_or_insertion = std::min(last_row[j], row[j - 1]) + gap_cost;
            row[j] = std::min(if_deletion_or_insertion, last_row[j - 1] + substitution_cost);
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
    std::ptrdiff_t max_score = 0;

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
            std::ptrdiff_t if_deletion = last_row[j] + gap_cost;
            std::ptrdiff_t if_insertion = row[j - 1] + gap_cost;
            std::ptrdiff_t score = std::max({std::ptrdiff_t(0), if_substitution, if_deletion, if_insertion});
            row[j] = score;
            max_score = std::max(max_score, score);
        }
    }

    return max_score;
}

using error_costs_256x256_t = std::array<error_cost_t, 256 * 256>;

/**
 *  @brief  Produces a substitution cost matrix for the Needleman-Wunsch alignment score,
 *          that would yield the same result as the negative Levenshtein distance.
 */
inline error_costs_256x256_t error_costs_256x256_unary() noexcept {
    error_costs_256x256_t result;
    for (std::size_t i = 0; i != 256; ++i)
        for (std::size_t j = 0; j != 256; ++j) //
            result[i * 256 + j] = i == j ? 0 : -1;
    return result;
}

/**
 *  @brief  Produces a substitution cost matrix using Blosum62 weights.
 *
 *  For characters corresponding to the 20 standard amino acids in Blosum62,
 *  the matrix is initialized with the respective scores. For any other character,
 *  a default penalty (e.g. -4) is used.
 */
inline error_costs_256x256_t error_costs_256x256_blosum62() noexcept {
    error_costs_256x256_t result;

    constexpr char amino_acids[] = "ARNDCQEGHILKMFPSTWYV";
    constexpr int num_amino = 20;

    // BLOSUM62 substitution matrix for the 20 amino acids.
    constexpr int blosum62[num_amino][num_amino] = {
        {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0},
        {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3},
        {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3},
        {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3},
        {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1},
        {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2},
        {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2},
        {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3},
        {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3},
        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3},
        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1},
        {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2},
        {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1},
        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1},
        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2},
        {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2},
        {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0},
        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3},
        {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1},
        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4}};

    // Build a lookup table to map any character (0–255) to an index in amino_acids,
    // or -1 if the character is not one of the valid amino acids.
    std::array<int, 256> amino_index;
    amino_index.fill(-1);
    for (int k = 0; k < num_amino; ++k) amino_index[static_cast<unsigned char>(amino_acids[k])] = k;

    // Define a default penalty for characters not in Blosum62 (e.g., a typical gap or mismatch penalty)
    constexpr error_cost_t default_penalty = -4;

    // Initialize the substitution matrix.
    for (std::size_t i = 0; i < 256; ++i) {
        for (std::size_t j = 0; j < 256; ++j) {
            int idx1 = amino_index[i];
            int idx2 = amino_index[j];
            if (idx1 != -1 && idx2 != -1) { result[i * 256 + j] = blosum62[idx1][idx2]; }
            else { result[i * 256 + j] = default_penalty; }
        }
    }
    return result;
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian