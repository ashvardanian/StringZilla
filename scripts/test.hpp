/**
 *  @brief  Helper structures and functions for C++ tests.
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
 *  MSVC, for example, requires one of short, int, long, long long, unsigned short, unsigned int, unsigned long,
 *  or unsigned long long
 */
struct uniform_uint8_distribution_t {
    std::uniform_int_distribution<std::uint32_t> distribution;
    inline uniform_uint8_distribution_t(std::size_t alphabet_size = 255)
        : distribution(1, static_cast<std::uint32_t>(alphabet_size)) {}
    template <typename generator_type>
    std::uint8_t operator()(generator_type &&generator) {
        return static_cast<std::uint8_t>(distribution(generator));
    }
};

inline void randomize_string(char *string, std::size_t length, char const *alphabet, std::size_t cardinality) {
    uniform_uint8_distribution_t distribution(cardinality);
    std::generate(string, string + length, [&]() -> char { return alphabet[distribution(global_random_generator())]; });
}

inline std::string random_string(std::size_t length, char const *alphabet, std::size_t cardinality) {
    std::string result(length, '\0');
    randomize_string(&result[0], length, alphabet, cardinality);
    return result;
}

/**
 *  @brief  Inefficient baseline Levenshtein distance computation, as implemented in most codebases.
 *          Allocates a new matrix on every call, with rows potentially scattered around memory.
 */
inline std::size_t levenshtein_baseline(char const *s1, std::size_t len1, char const *s2, std::size_t len2) {
    std::vector<std::vector<std::size_t>> dp(len1 + 1, std::vector<std::size_t>(len2 + 1));

    // Initialize the borders of the matrix.
    for (std::size_t i = 0; i <= len1; ++i) dp[i][0] = i;
    for (std::size_t j = 0; j <= len2; ++j) dp[0][j] = j;

    for (std::size_t i = 1; i <= len1; ++i) {
        for (std::size_t j = 1; j <= len2; ++j) {
            std::size_t cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            // dp[i][j] is the minimum of deletion, insertion, or substitution
            dp[i][j] = std::min({
                dp[i - 1][j] + 1,       // Deletion
                dp[i][j - 1] + 1,       // Insertion
                dp[i - 1][j - 1] + cost // Substitution
            });
        }
    }

    return dp[len1][len2];
}

/**
 *  @brief  Produces a substitution cost matrix for the Needlemann-Wunsch alignment score,
 *          that would yield the same result as the negative Levenshtein distance.
 */
inline std::vector<std::int8_t> unary_substitution_costs() {
    std::vector<std::int8_t> result(256 * 256);
    for (std::size_t i = 0; i != 256; ++i)
        for (std::size_t j = 0; j != 256; ++j) result[i * 256 + j] = (i == j ? 0 : -1);
    return result;
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian