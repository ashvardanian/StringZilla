/**
 *  @brief  Helper structures and functions for C++ unit- and stress-tests.
 */
#pragma once
#include <fstream>  // `std::ifstream`
#include <iostream> // `std::cout`, `std::endl`
#include <random>   // `std::random_device`
#include <string>   // `std::string`
#include <vector>   // `std::vector`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

inline std::string read_file(std::string path) {
    std::ifstream stream(path);
    if (!stream.is_open()) { throw std::runtime_error("Failed to open file: " + path); }
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

inline void write_file(std::string path, std::string content) {
    std::ofstream stream(path);
    if (!stream.is_open()) { throw std::runtime_error("Failed to open file: " + path); }
    stream << content;
    stream.close();
}

inline std::mt19937 &global_random_generator() {
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
struct uniform_uint8_distribution_t {
    std::uniform_int_distribution<std::uint32_t> distribution;
    inline uniform_uint8_distribution_t(std::size_t alphabet_size = 255)
        : distribution(1, static_cast<std::uint32_t>(alphabet_size)) {}
    inline uniform_uint8_distribution_t(char from, char to)
        : distribution(static_cast<std::uint32_t>(from), static_cast<std::uint32_t>(to)) {}
    template <typename generator_type>
    std::uint8_t operator()(generator_type &&generator) {
        return static_cast<std::uint8_t>(distribution(generator));
    }
};

inline void randomize_string(char *string, std::size_t length, char const *alphabet, std::size_t cardinality) {
    uniform_uint8_distribution_t distribution(0, cardinality - 1);
    std::generate(string, string + length, [&]() -> char { return alphabet[distribution(global_random_generator())]; });
}

inline void randomize_string(char *string, std::size_t length) {
    uniform_uint8_distribution_t distribution;
    std::generate(string, string + length, [&]() -> char { return distribution(global_random_generator()); });
}

inline std::string random_string(std::size_t length, char const *alphabet, std::size_t cardinality) {
    std::string result(length, '\0');
    randomize_string(&result[0], length, alphabet, cardinality);
    return result;
}

inline std::string repeat(std::string const &patten, std::size_t count) {
    std::string result(patten.size() * count, '\0');
    for (std::size_t i = 0; i < count; ++i) std::copy(patten.begin(), patten.end(), result.begin() + i * patten.size());
    return result;
}

/**
 *  @brief  A callback type for iterating over consecutive random-length slices of a string.
 */
template <typename slice_callback_type_>
inline void iterate_in_random_slices(std::string const &text, slice_callback_type_ &&slice_callback) {
    std::size_t remaining = text.size();
    while (remaining > 0) {
        std::size_t slice_length = std::uniform_int_distribution<std::size_t>(1, remaining)(global_random_generator());
        slice_callback({text.data() + text.size() - remaining, slice_length});
        remaining -= slice_length;
    }
}

/**
 *  @brief  Inefficient baseline Levenshtein distance computation, as implemented in most codebases.
 *          Allocates a new matrix on every call, with rows potentially scattered around memory.
 */
inline std::size_t levenshtein_baseline(char const *s1, std::size_t len1, char const *s2, std::size_t len2) {
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
            std::size_t cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            std::size_t deletion_or_insertion = std::min(last_row[j], row[j - 1]) + 1;
            row[j] = std::min(deletion_or_insertion, last_row[j - 1] + cost);
        }
    }

    return matrix_buffer.back();
}

using error_costs_256x256_t = std::array<sz_error_cost_t, 256 * 256>;

/**
 *  @brief  Produces a substitution cost matrix for the Needleman-Wunsch alignment score,
 *          that would yield the same result as the negative Levenshtein distance.
 */
inline error_costs_256x256_t unary_substitution_costs() {
    error_costs_256x256_t result;
    for (std::size_t i = 0; i != 256; ++i)
        for (std::size_t j = 0; j != 256; ++j) result[i * 256 + j] = (i == j ? 0 : -1);
    return result;
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian