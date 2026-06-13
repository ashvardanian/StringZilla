/**
 *  @brief ISA-agnostic parallel string-similarity template core (serial backend).
 *  @file include/stringzillas/similarities/serial.hpp
 *  @author Ash Vardanian
 *
 *  Contains the shared template core for all parallel similarity backends:
 *  Algorithm Building Blocks, the base `tile_scorer`, `diagonal_walker` & `horizontal_walker`,
 *  cost types, and substitution matrices, plus the serial backend aliases.
 *
 *  Every backend specialization header (`icelake.hpp`, `cuda.cuh`, ...) must include this
 *  file first, so that the primary templates are visible before any specialization.
 * 
 *  This file is designed around several guiding principles:
 * 
 *  - Avoid larger integral types, where smaller ones are enough.
 *  - Larger kernels are assembled from smaller templates, so to keep binary size and compilation time
 *    sane, type-invariant pieces are shielded from generic interfaces via "trampolines".
 *  - No single-pair scorer should be reposible for memory allocations, even if the whole batch
 *    contains just one pair of entries.
 */
#ifndef STRINGZILLAS_SIMILARITIES_SERIAL_HPP_
#define STRINGZILLAS_SIMILARITIES_SERIAL_HPP_

#include "stringzilla/types.hpp"  // `sz::error_cost_t`
#include "stringzilla/memory.h"   // `sz_move_serial`
#include "stringzillas/types.hpp" // `sz::executor_like`

#include <atomic>      // `std::atomic` to synchronize threads
#include <type_traits> // `std::enable_if_t` for meta-programming
#include <limits>      // `std::numeric_limits` for numeric types
#include <iterator>    // `std::iterator_traits` for iterators

namespace ashvardanian {
namespace stringzillas {

struct error_costs_32x32_t;

constexpr sz_capability_t serialize_capability(sz_capability_t capability) noexcept {
    sz_capability_t without_parallel = static_cast<sz_capability_t>(capability & ~sz_cap_parallel_k);
    sz_capability_t without_serial = static_cast<sz_capability_t>(without_parallel & ~sz_cap_serial_k);
    return without_serial != 0 ? without_serial : without_parallel;
}

template <sz_similarity_objective_t objective_, typename score_type_>
constexpr score_type_ min_or_max(score_type_ a, score_type_ b) noexcept {
    if constexpr (objective_ == sz_minimize_distance_k) { return sz_min_of_two(a, b); }
    else { return sz_max_of_two(a, b); }
}

template <typename value_type_>
constexpr void rotate_three(value_type_ &a, value_type_ &b, value_type_ &c) noexcept {
    value_type_ tmp = a;
    a = b;
    b = c;
    c = tmp;
}

/**
 *  @brief A trivial `error_cost_abs` analog, that's `constexpr`-friendly.
 */
constexpr error_cost_magnitude_t error_cost_abs(error_cost_t x) noexcept {
    return static_cast<error_cost_magnitude_t>(x < 0 ? -(i16_t)x : (i16_t)x);
}

/**
 *  @brief A trivial function object for linear and affine gap costs in Levenshtein-like similarity algorithms.
 *  @sa affine_gap_costs_t
 */
struct linear_gap_costs_t {
    error_cost_t open_or_extend = 1;

    constexpr error_cost_magnitude_t magnitude() const noexcept { return error_cost_abs(open_or_extend); }
};

/**
 *  @brief A trivial function object for affine gap costs in Levenshtein-like similarity algorithms.
 *  @sa linear_gap_costs_t
 */
struct affine_gap_costs_t {
    error_cost_t open = 1;
    error_cost_t extend = 1;

    constexpr error_cost_magnitude_t magnitude() const noexcept {
        return std::max(error_cost_abs(open), error_cost_abs(extend));
    }
};

template <typename gap_costs_type_>
constexpr sz_similarity_gaps_t gap_type() {
    constexpr bool is_linear_k = is_same_type<gap_costs_type_, linear_gap_costs_t>::value;
    constexpr bool is_affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    static_assert(is_linear_k || is_affine_k, "Invalid gap costs type");
    if constexpr (is_linear_k) { return sz_gaps_linear_k; }
    else { return sz_gaps_affine_k; }
}

/**
 *  @brief A trivial function object for uniform character substitution costs in Levenshtein-like similarity algorithms.
 *  @sa error_costs_32x32_t, unary_substitution_costs_t
 */
struct uniform_substitution_costs_t {
    error_cost_t match = 0;
    error_cost_t mismatch = 1;

    constexpr error_cost_t operator()(char a, char b) const noexcept { return a == b ? match : mismatch; }
    constexpr error_cost_t operator()(rune_t a, rune_t b) const noexcept { return a == b ? match : mismatch; }
    constexpr error_cost_magnitude_t magnitude() const noexcept {
        return std::max(error_cost_abs(match), error_cost_abs(mismatch));
    }
};

/**
 *  @brief The number of distinct character classes in a @b `error_costs_32x32_t` table.
 *      Fixed at 32 to cover the largest practical alphabets - ~4 DNA bases, ~20 amino-acids,
 *      or keyboard-distance buckets - while keeping the table tiny.
 */
static constexpr size_t error_costs_classes_count_k = 32;

/**
 *  @brief A @b compact error costs matrix for byte-level similarity scoring.
 *
 *  Instead of a dense (256 x 256) ~ 65'536 byte table, every input byte is first mapped to one of
 *  @b `error_costs_classes_count_k` (32) classes through `byte_to_class`, and the substitution cost
 *  between two bytes is looked up in the (32 x 32) `class_substitution_costs` table:
 *
 *      cost(a, b) = class_substitution_costs[byte_to_class[a]][byte_to_class[b]]
 *
 *  The 1 KB cost table plus the 256 byte class map fit trivially into shared memory or registers,
 *  matching bioinformatics alphabets (≤4 DNA / ≤20 amino-acid classes) and keyboard-distance pricing,
 *  while avoiding the divergent serialization of a full 256 x 256 table on the GPU.
 * 
 *  @section Biological Data
 *
 *  For proteins, a (32 x 32) matrix takes 1024 bytes, which is a steep 2.56x increase from (20 x 20) ~ 400 bytes.
 *  Still, its an acceptable tradeoff given the convenience of using ASCII arithmetic for lookups, and occasional
 *  use of special "ambiguous" characters. The 20 standard amino-acids are @b ARNDCQEGHILKMFPSTWYV. Others include:
 *  - @b U: Selenocysteine, sometimes called the 21st amino acid.
 *  - @b O: Pyrrolysine, occasionally referred to as the 22nd amino acid.
 *  - @b B: An ambiguous code representing either Aspartic acid (D) or Asparagine (N).
 *  - @b Z: An ambiguous code representing either Glutamic acid (E) or Glutamine (Q).
 *  - @b X: Used when the identity of an amino acid is unknown or unspecified.
 *  - @b *: Denotes a stop codon, signaling the end of the protein sequence during translation.
 *  This leaves @b J as the only ASCII letter not used in protein sequences and @b (*) asterisk as the the only
 *  non-letter character used.
 *
 *  For DNA and RNA sequences, often a (4 x 4) matrix can be enough, but in the general case, additional characters
 *  are used to mark ambiguous reads. For nucleic acids the standard alphabets are @b ACGT for @b DNA and @b ACGU
 *  for @b RNA. There are a lot more ambiguity codes though, where each row lists which of A, C, G, and T/U a
 *  given code can stand for:
 *
 *       Code | Can be A | Can be C | Can be G | Can be T/U
 *       A    |    X     |          |          |
 *       C    |          |    X     |          |
 *       G    |          |          |    X     |
 *       T    |          |          |          |     X
 *       R    |    X     |          |    X     |
 *       Y    |          |    X     |          |     X
 *       S    |          |    X     |    X     |
 *       W    |    X     |          |          |     X
 *       K    |          |          |    X     |     X
 *       M    |    X     |    X     |          |
 *       B    |          |    X     |    X     |     X
 *       D    |    X     |          |    X     |     X
 *       H    |    X     |    X     |          |     X
 *       V    |    X     |    X     |    X     |
 *       N    |    X     |    X     |    X     |     X
 *
 *  If the BLOSUM62 matrix is often used for proteins, the IUB or NUC.4.4 are often used for nucleic acids.
 *  Both can be easily extracted from BioPython and converted to our ASCII order:
 *
 *  @code{.py}
 *  import string
 *  from Bio.Align import substitution_matrices
 *
 *  def map_to_new_alphabet(matrix, new_alphabet: str, default_value: int = -128):
 *      old_alphabet = str(matrix.alphabet)
 *      indices = {ch: old_alphabet.find(ch) for ch in new_alphabet}
 *      return [
 *          [matrix[indices[r], indices[c]] if indices[r] != -1 and indices[c] != -1 else default_value
 *          for c in new_alphabet]
 *          for r in new_alphabet
 *      ]
 *
 *  matrix = substitution_matrices.load("BLOSUM62").astype(int) # Or "NUC.4.4"
 *  print(map_to_new_alphabet(matrix, string.ascii_uppercase))
 *  @endcode
 */
struct error_costs_32x32_t {
    static constexpr size_t classes_count_k = error_costs_classes_count_k;

    u8_t byte_to_class[256] = {0};
    error_cost_t class_substitution_costs[classes_count_k][classes_count_k] = {{0}};

    constexpr error_cost_t operator()(char a, char b) const noexcept {
        return class_substitution_costs[byte_to_class[(u8_t)a]][byte_to_class[(u8_t)b]];
    }
    constexpr error_cost_t operator()(u8_t a, u8_t b) const noexcept {
        return class_substitution_costs[byte_to_class[a]][byte_to_class[b]];
    }

    constexpr error_cost_magnitude_t magnitude() const noexcept {
        error_cost_magnitude_t max_magnitude = 0;
        for (size_t i = 0; i != classes_count_k; ++i)
            for (size_t j = 0; j != classes_count_k; ++j) //
                max_magnitude = (std::max)(max_magnitude, error_cost_abs(class_substitution_costs[i][j]));
        return max_magnitude;
    }

    /**
     *  @brief BLOSUM62 substitution matrix for protein analysis in bioinformatics, folded into the compact form.
     *  @see https://en.wikipedia.org/wiki/BLOSUM
     */
    static constexpr error_costs_32x32_t blosum62() noexcept {
        constexpr error_cost_t na = -128; // Placeholder for unused residues
        constexpr error_cost_t cells[26][26] = {
            {4, -2, 0, -2, -1, -2, 0, -2, -1, na, -1, -1, -1, -2, na, -1, -1, -1, 1, 0, na, 0, -3, 0, -2, -1},
            {-2, 4, -3, 4, 1, -3, -1, 0, -3, na, 0, -4, -3, 3, na, -2, 0, -1, 0, -1, na, -3, -4, -1, -3, 1},
            {0, -3, 9, -3, -4, -2, -3, -3, -1, na, -3, -1, -1, -3, na, -3, -3, -3, -1, -1, na, -1, -2, -2, -2, -3},
            {-2, 4, -3, 6, 2, -3, -1, -1, -3, na, -1, -4, -3, 1, na, -1, 0, -2, 0, -1, na, -3, -4, -1, -3, 1},
            {-1, 1, -4, 2, 5, -3, -2, 0, -3, na, 1, -3, -2, 0, na, -1, 2, 0, 0, -1, na, -2, -3, -1, -2, 4},
            {-2, -3, -2, -3, -3, 6, -3, -1, 0, na, -3, 0, 0, -3, na, -4, -3, -3, -2, -2, na, -1, 1, -1, 3, -3},
            {0, -1, -3, -1, -2, -3, 6, -2, -4, na, -2, -4, -3, 0, na, -2, -2, -2, 0, -2, na, -3, -2, -1, -3, -2},
            {-2, 0, -3, -1, 0, -1, -2, 8, -3, na, -1, -3, -2, 1, na, -2, 0, 0, -1, -2, na, -3, -2, -1, 2, 0},
            {-1, -3, -1, -3, -3, 0, -4, -3, 4, na, -3, 2, 1, -3, na, -3, -3, -3, -2, -1, na, 3, -3, -1, -1, -3},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {-1, 0, -3, -1, 1, -3, -2, -1, -3, na, 5, -2, -1, 0, na, -1, 1, 2, 0, -1, na, -2, -3, -1, -2, 1},
            {-1, -4, -1, -4, -3, 0, -4, -3, 2, na, -2, 4, 2, -3, na, -3, -2, -2, -2, -1, na, 1, -2, -1, -1, -3},
            {-1, -3, -1, -3, -2, 0, -3, -2, 1, na, -1, 2, 5, -2, na, -2, 0, -1, -1, -1, na, 1, -1, -1, -1, -1},
            {-2, 3, -3, 1, 0, -3, 0, 1, -3, na, 0, -3, -2, 6, na, -2, 0, 0, 1, 0, na, -3, -4, -1, -2, 0},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {-1, -2, -3, -1, -1, -4, -2, -2, -3, na, -1, -3, -2, -2, na, 7, -1, -2, -1, -1, na, -2, -4, -2, -3, -1},
            {-1, 0, -3, 0, 2, -3, -2, 0, -3, na, 1, -2, 0, 0, na, -1, 5, 1, 0, -1, na, -2, -2, -1, -1, 3},
            {-1, -1, -3, -2, 0, -3, -2, 0, -3, na, 2, -2, -1, 0, na, -2, 1, 5, -1, -1, na, -3, -3, -1, -2, 0},
            {1, 0, -1, 0, 0, -2, 0, -1, -2, na, 0, -2, -1, 1, na, -1, 0, -1, 4, 1, na, -2, -3, 0, -2, 0},
            {0, -1, -1, -1, -1, -2, -2, -2, -1, na, -1, -1, -1, 0, na, -1, -1, -1, 1, 5, na, 0, -2, 0, -2, -1},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {0, -3, -1, -3, -2, -1, -3, -3, 3, na, -2, 1, 1, -3, na, -2, -2, -3, -2, 0, na, 4, -3, -1, -1, -2},
            {-3, -4, -2, -4, -3, 1, -2, -2, -3, na, -3, -2, -1, -4, na, -4, -2, -3, -3, -2, na, -3, 11, -2, 2, -3},
            {0, -1, -2, -1, -1, -1, -1, -1, -1, na, -1, -1, -1, -1, na, -2, -1, -1, 0, 0, na, -1, -2, -1, -1, -1},
            {-2, -3, -2, -3, -2, 3, -3, 2, -1, na, -2, -1, -1, -2, na, -3, -1, -2, -2, -2, na, -1, 2, -1, 7, -2},
            {-1, 1, -3, 1, 4, -3, -2, 0, -3, na, 1, -3, -1, 0, na, -1, 3, 0, 0, -1, na, -2, -3, -1, -2, 4}};
        return from_ascii_26x26_(cells);
    }

    /**
     *  @brief NUC.4.4 substitution matrix for DNA analysis in bioinformatics, folded into the compact form.
     *  @see https://www.biostars.org/p/73028/#93435
     */
    static constexpr error_costs_32x32_t nuc44() noexcept {
        constexpr error_cost_t na = -128; // Placeholder for unused residues
        constexpr error_cost_t cells[26][26] = {
            {5, -4, -4, -1, na, na, -4, -1, na, na, -4, na, 1, -2, na, na, na, 1, -4, -4, na, -1, 1, na, -4, na},
            {-4, -1, -1, -2, na, na, -1, -2, na, na, -1, na, -3, -1, na, na, na, -3, -1, -1, na, -2, -3, na, -1, na},
            {-4, -1, 5, -4, na, na, -4, -1, na, na, -4, na, 1, -2, na, na, na, -4, 1, -4, na, -1, -4, na, 1, na},
            {-1, -2, -4, -1, na, na, -1, -2, na, na, -1, na, -3, -1, na, na, na, -1, -3, -1, na, -2, -1, na, -3, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {-4, -1, -4, -1, na, na, 5, -4, na, na, 1, na, -4, -2, na, na, na, 1, 1, -4, na, -1, -4, na, -4, na},
            {-1, -2, -1, -2, na, na, -4, -1, na, na, -3, na, -1, -1, na, na, na, -3, -3, -1, na, -2, -1, na, -1, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {-4, -1, -4, -1, na, na, 1, -3, na, na, -1, na, -4, -1, na, na, na, -2, -2, 1, na, -3, -2, na, -2, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {1, -3, 1, -3, na, na, -4, -1, na, na, -4, na, -1, -1, na, na, na, -2, -2, -4, na, -1, -2, na, -2, na},
            {-2, -1, -2, -1, na, na, -2, -1, na, na, -1, na, -1, -1, na, na, na, -1, -1, -2, na, -1, -1, na, -1, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {1, -3, -4, -1, na, na, 1, -3, na, na, -2, na, -2, -1, na, na, na, -1, -2, -4, na, -1, -2, na, -4, na},
            {-4, -1, 1, -3, na, na, 1, -3, na, na, -2, na, -2, -1, na, na, na, -2, -1, -4, na, -1, -4, na, -2, na},
            {-4, -1, -4, -1, na, na, -4, -1, na, na, 1, na, -4, -2, na, na, na, -4, -4, 5, na, -4, 1, na, 1, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {-1, -2, -1, -2, na, na, -1, -2, na, na, -3, na, -1, -1, na, na, na, -1, -1, -4, na, -1, -3, na, -3, na},
            {1, -3, -4, -1, na, na, -4, -1, na, na, -2, na, -2, -1, na, na, na, -2, -4, 1, na, -3, -1, na, -2, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na},
            {-4, -1, 1, -3, na, na, -4, -1, na, na, -2, na, -2, -1, na, na, na, -4, -2, 1, na, -3, -2, na, -1, na},
            {na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na, na}};
        return from_ascii_26x26_(cells);
    }

  private:
    /**
     *  @brief Folds a dense (26 x 26) ASCII substitution matrix into this compact class-based table,
     *      assigning one class per @b used residue (those with a non-placeholder diagonal cost).
     *
     *  Class 0 is the catch-all bucket for every byte that is not a used uppercase residue and costs
     *  zero against everything. Each used residue receives a distinct class index, so BLOSUM62 (≤20
     *  amino-acids) and NUC.4.4 (≤5 nucleotides) both fit comfortably within the 32 available classes.
     */
    static constexpr error_costs_32x32_t from_ascii_26x26_(error_cost_t const (&cells)[26][26]) noexcept {
        constexpr error_cost_t na = -128; // Placeholder for unused residues
        error_costs_32x32_t result;

        // Assign a fresh class to every residue whose diagonal cost is not a placeholder.
        u8_t residue_to_class[26] = {0};
        u8_t next_class = 1;
        for (size_t i = 0; i != 26u; ++i)
            if (cells[i][i] != na) {
                residue_to_class[i] = next_class;
                result.byte_to_class[i + 65u] = next_class;
                ++next_class;
            }

        // Populate the cost between every pair of used residues.
        for (size_t i = 0; i != 26u; ++i)
            for (size_t j = 0; j != 26u; ++j)
                if (residue_to_class[i] != 0 && residue_to_class[j] != 0)
                    result.class_substitution_costs[residue_to_class[i]][residue_to_class[j]] = cells[i][j];
        return result;
    }
};

#pragma region - Algorithm Building Blocks

/**
 *  @brief Helper object to guess the amount of SRAM we want to effectively process the input
 *      without fetching from RAM/VRAM all the time, including the space for 3 diagonals
 *      and the strings themselves.
 *
 *  @tparam score_type_ The arithmetics will differ for .
 *  @tparam is_signed_ Whether the similarity scores can be negative or not.
 */
template <typename score_type_>
struct diagonal_memory_requirements {
    using score_t = score_type_;
    static constexpr bool is_signed_k = std::is_signed_v<score_t>;

    size_t max_diagonal_length = 0;
    bytes_per_cell_t bytes_per_cell = zero_bytes_per_cell_k;
    size_t bytes_per_diagonal = 0;
    size_t bytes_for_diagonals = 0; // ? Shared-memory bytes for just the DP diagonals (excludes the input strings).
    size_t total = 0;

    /**
     *  @param[in] first_length,second_length The lengths of strings in characters/codepoints/runes.
     *  @param[in] substitute_magnitude,gap_magnitude The absolute value of the maximum change in nearby cells.
     *  @param[in] bytes_per_char The number of bytes per character, 4 for UTF-32, 1 for ASCII.
     *  @param[in] register_width The alignment of the data in bytes, 4 for CUDA, 64 for AVX-512.
     *  @param[in] min_bytes_per_cell The minimum number of bytes per cell, if kernels for some types aren't available.
     *
     *  To understand the @p substitute_magnitude,gap_magnitude parameters, consider the following example:
     *  - substitution costs ranging from -16 to +15
     *  - gap costs equal to -10
     *  In that case, the biggest change will be `abs(-16) = 16`, so the passed argument should be 16.
     *  In case of default Levenshtein distance, the maximum change is 1, so the passed argument should be 1.
     */
    constexpr diagonal_memory_requirements(                                                //
        size_t first_length, size_t second_length,                                         //
        sz_similarity_gaps_t gap_type,                                                     //
        error_cost_magnitude_t substitute_magnitude, error_cost_magnitude_t gap_magnitude, //
        size_t bytes_per_char,                                                             //
        size_t register_width,                                                             //
        bytes_per_cell_t min_bytes_per_cell = one_byte_per_cell_k) noexcept {

        // If any of the strings is empty, we don't need any memory to perform the similarity scoring.
        size_t shorter_length = sz_min_of_two(first_length, second_length);
        if (shorter_length == 0) {
            this->max_diagonal_length = 0;
            this->bytes_per_cell = zero_bytes_per_cell_k;
            this->bytes_per_diagonal = 0;
            this->bytes_for_diagonals = 0;
            this->total = 0;
            return;
        }

        // Each diagonal in the DP matrix is only by 1 longer than the shorter string.
        size_t longer_length = sz_max_of_two(first_length, second_length);
        this->max_diagonal_length = shorter_length + 1;

        // The amount of memory we need per diagonal, depends on the maximum number of the differences
        // between 2 strings and the maximum cost of each change.
        error_cost_magnitude_t magnitude = sz_max_of_two(substitute_magnitude, gap_magnitude);
        size_t max_cell_value = (longer_length + 1) * magnitude;
        if constexpr (!is_signed_k)
            this->bytes_per_cell = //
                max_cell_value < 256          ? one_byte_per_cell_k
                : max_cell_value < 65536      ? two_bytes_per_cell_k
                : max_cell_value < 4294967296 ? four_bytes_per_cell_k
                                              : eight_bytes_per_cell_k;
        else
            this->bytes_per_cell = //
                max_cell_value < 127          ? one_byte_per_cell_k
                : max_cell_value < 32767      ? two_bytes_per_cell_k
                : max_cell_value < 2147483647 ? four_bytes_per_cell_k
                                              : eight_bytes_per_cell_k;
        if (this->bytes_per_cell < min_bytes_per_cell) this->bytes_per_cell = min_bytes_per_cell;

        // For each string we need to copy its contents, and allocate 3 bands proportional to the length
        // of the shorter string with each cell being big enough to hold the length of the longer one.
        // The diagonals should be aligned to `register_width` bytes to allow for SIMD operations.
        this->bytes_per_diagonal = round_up_to_multiple<size_t>(max_diagonal_length * bytes_per_cell, register_width);

        // When dealing with linear gaps, we need 3x diagonals of 1 matrix.
        // When dealing with affine gaps, we need 3x diagonals of 1 matrix and 2x diagonals of 2 matrices
        // = 7x diagonals total.
        size_t diagonals_count = gap_type == sz_gaps_linear_k ? 3 : 7;
        size_t first_length_bytes = round_up_to_multiple<size_t>(first_length * bytes_per_char, register_width);
        size_t second_length_bytes = round_up_to_multiple<size_t>(second_length * bytes_per_char, register_width);
        this->bytes_for_diagonals = diagonals_count * bytes_per_diagonal;
        this->total = this->bytes_for_diagonals + first_length_bytes + second_length_bytes;
    }
};

using scratch_space_t = span<std::byte>;

/**
 *  @brief A running, cache-line-padded scratch byte amount, used to lay out a walker's sub-buffers.
 *
 *  Each walker partitions its `scratch_space_t` into a handful of sub-buffers (score diagonals, a reversed
 *  copy of the shorter string, a Myers `match_masks` table, ...). Growing this amount once per sub-buffer keeps
 *  every offset cache-line aligned and yields the total scratch the walker needs - a single source of truth
 *  shared by the walker's `layout()` and its `operator()`. Cache-line width is `>=` any CPU register width, so
 *  the padding also keeps full-register SIMD over-reads near a buffer's end in bounds.
 */
struct scratch_amount_t {
    // ? Deliberately a poison default (not `SZ_CACHE_LINE_WIDTH`): an instance built without an explicit
    // ? `cpu_specs_t::cache_line_width` should produce an obviously-broken `total` (huge -> `bad_alloc`/ASan),
    // ? surfacing any place that forgot to propagate the alignment rather than silently assuming 64 bytes.
    size_t alignment = std::numeric_limits<size_t>::max();
    size_t total = 0; // ? The accumulated, padded byte count == the next buffer's offset.

    /** @brief Reads the current end of the scratch, i.e. the offset where the next sub-buffer would start. */
    constexpr operator size_t() const noexcept { return total; }

    /** @brief Reserves @p bytes for the next sub-buffer, padded so the following offset stays aligned. */
    constexpr scratch_amount_t &operator+=(size_t bytes) noexcept {
        total += round_up_to_multiple<size_t>(bytes, alignment);
        return *this;
    }
};

#pragma region - Core Templates

#if SZ_HAS_CONCEPTS_

template <typename iterator_type_>
concept pointer_like = requires(iterator_type_ iterator, size_t idx) {
    { ++iterator } -> std::same_as<iterator_type_ &>; // pre-increment
    { *iterator };                                    // dereference
    { iterator[idx] };                                // random access
};

template <typename value_type_>
concept score_like = std::integral<value_type_> && std::is_trivial_v<value_type_>;

template <typename substituter_type_>
concept substituter_like = requires(substituter_type_ costs) {
    { costs.magnitude() } -> std::convertible_to<error_cost_magnitude_t>;      // retrieving the magnitude
    { costs.operator()(char(), char()) } -> std::convertible_to<error_cost_t>; // cost of substitution
};

template <typename gap_costs_type_>
concept gap_costs_like = requires(gap_costs_type_ costs) {
    { costs.magnitude() } -> std::convertible_to<error_cost_magnitude_t>; // retrieving the magnitude
};

#endif

/**
 *  @brief An operator to be applied to be applied to all @b 2x2 tiles of the DP matrix to produce
 *      the bottom-right value from the 3x others when populating the Dynamic Programming matrix.
 *
 *  @tparam first_iterator_type_ Typically `char*`, `rune_t*`, or a `constant_iterator`.
 *  @tparam second_iterator_type_ Typically `char*` or `rune_t*`.
 *  @tparam score_type_ The type of the score, typically `size_t` or `ssize_t`.
 *  @tparam substituter_type_ Typically `uniform_substitution_costs_t` or a lookup table.
 *  @tparam gap_costs_type_ Either `linear_gap_costs_t` or `sz_gaps_affine_k`.
 *  @tparam objective_ Either `sz_minimize_distance_k` or `sz_maximize_score_k`.
 *  @tparam locality_ Either `sz_similarity_global_k` or `sz_similarity_local_k`.
 *  @tparam capability_ The SIMD capabilities of the target architecture.
 *  @tparam enable_ Used to enable/disable the specialization.
 */
template <                                                       //
    typename first_iterator_type_ = char const *,                //
    typename second_iterator_type_ = char const *,               //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    typename gap_costs_type_ = linear_gap_costs_t,               //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_serial_k,               //
    typename enable_ = void                                      //
    >
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_> && gap_costs_like<gap_costs_type_>
#endif
struct tile_scorer;

/**
 *  @brief Alignment Score and Edit Distance algorithm evaluating the Dynamic Programming matrix
 *      @b (anti)diagonal-by-(anti)diagonal on a CPU.
 *
 *  Can be used for both global and local alignment, like Needleman-Wunsch and Smith-Waterman.
 *  Can be used for both linear and affine gap penalties.
 *
 *  ? There are smarter algorithms for computing the Levenshtein distance, mostly based on bit-level operations.
 *  ? Those, however, don't generalize well to arbitrary length inputs or non-uniform substitution costs.
 *  ? This algorithm provides a more flexible baseline implementation for future SIMD and GPGPU optimizations.
 *
 *  @tparam char_or_rune_type_ The type of the characters in the strings, generally `char` or @b `rune_t` for UTF-8.
 *  @tparam score_type_ The smallest type that can hold the distance, ideally `i8_t` or `u8_t`.
 *  @tparam substituter_type_ A callable type that takes two characters and returns the substitution cost.
 *  @tparam gap_costs_type_ Whether to use linear or affine gap penalties.
 *  @tparam objective_ Whether to minimize the distance or maximize the score.
 *  @tparam locality_ Whether to use the global alignment algorithm or the local one.
 *  @tparam capability_ Whether to use @b multi-threading or some form of @b SIMD vectorization, or both.
 *  @tparam enable_ Used to enable/disable the specialization.
 */
template <                                                       //
    typename char_or_rune_type_ = char,                          //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    typename gap_costs_type_ = linear_gap_costs_t,               //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_serial_k,               //
    typename enable_ = void                                      //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_> && gap_costs_like<gap_costs_type_>
#endif
struct diagonal_walker;

/**
 *  @brief Alignment Score and Edit Distance algorithm evaluating the Dynamic Programming matrix
 *      @b row-by-row on a CPU, using the conventional Wagner-Fischer algorithm.
 *
 *  Can be used for both global and local alignment, like Needleman-Wunsch and Smith-Waterman.
 *  Can be used for both linear and affine gap penalties.
 *
 *  @tparam char_or_rune_type_ The type of the characters in the strings, generally `char` or @b `rune_t` for UTF-8.
 *  @tparam score_type_ The smallest type that can hold the distance, ideally `i8_t` or `u8_t`.
 *  @tparam substituter_type_ A callable type that takes two characters and returns the substitution cost.
 *  @tparam gap_costs_type_ Whether to use linear or affine gap penalties.
 *  @tparam allocator_type_ A default-constructible allocator type for the internal buffers.
 *  @tparam objective_ Whether to minimize the distance or maximize the score.
 *  @tparam locality_ Whether to use the global alignment algorithm or the local one.
 *  @tparam capability_ Whether to use @b multi-threading or some form of @b SIMD vectorization, or both.
 *  @tparam enable_ Used to enable/disable the specialization.
 *
 *  @note The API of this algorithm is a bit weird, but it's designed to minimize the reliance on the definitions
 *        in the `stringzilla.hpp` header, making compilation times shorter for the end-user.
 *  @sa For lower-level API, check `szs_levenshtein_distance[_utf8]` and `szs_needleman_wunsch_score`.
 *  @sa For simplicity, use the `sz::levenshtein_distance[_utf8]` and `sz::needleman_wunsch_score`.
 *  @sa For bulk API, use `sz::levenshtein_distances[_utf8]`.
 */
template <                                                       //
    typename char_or_rune_type_ = char,                          //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    typename gap_costs_type_ = linear_gap_costs_t,               //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_serial_k,               //
    typename enable_ = void                                      //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_> && gap_costs_like<gap_costs_type_>
#endif
struct horizontal_walker;

/**
 *  @brief Specialized Myers algorithm implementation for Levenshtein one-to-one distance.
 *      Doesn't support non-unary substitution costs or affine gaps.
 */
template <                                         //
    typename char_or_rune_type_ = char,            //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename enable_ = void                        //
    >
struct levenshtein_distance_myers;

/**
 *  @brief Computes one or many pairwise Levenshtein distances in parallel using the CPU backend.
 *      For pairs of very large strings, all cores cooperate to compute one distance maximizing
 *      cache hits. For smaller strings, each core computes its own distance.
 */
template <                                         //
    typename gap_costs_type_ = linear_gap_costs_t, //
    typename allocator_type_ = dummy_alloc_t,      //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename enable_ = void                        //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distances;

template <                                         //
    typename gap_costs_type_ = linear_gap_costs_t, //
    typename allocator_type_ = dummy_alloc_t,      //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename enable_ = void                        //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distances_utf8;

template <                                            //
    typename substituter_type_ = error_costs_32x32_t, //
    typename gap_costs_type_ = linear_gap_costs_t,    //
    typename allocator_type_ = dummy_alloc_t,         //
    sz_capability_t capability_ = sz_cap_serial_k,    //
    typename enable_ = void                           //
    >
#if SZ_HAS_CONCEPTS_
    requires substituter_like<substituter_type_> && gap_costs_like<gap_costs_type_>
#endif
struct needleman_wunsch_scores;

template <                                            //
    typename substituter_type_ = error_costs_32x32_t, //
    typename gap_costs_type_ = linear_gap_costs_t,    //
    typename allocator_type_ = dummy_alloc_t,         //
    sz_capability_t capability_ = sz_cap_serial_k,    //
    typename enable_ = void                           //
    >
#if SZ_HAS_CONCEPTS_
    requires substituter_like<substituter_type_> && gap_costs_like<gap_costs_type_>
#endif
struct smith_waterman_scores;

#pragma endregion - Core Templates

#pragma region - Common Aliases

using malloc_t = std::allocator<char>;

/**
 *  In non-SIMD backends we still leverage multi-threading for parallelism.
 */
using levenshtein_serial_t = levenshtein_distances<linear_gap_costs_t, malloc_t, sz_cap_serial_k>;
using levenshtein_utf8_serial_t = levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_cap_serial_k>;
using needleman_wunsch_serial_t =
    needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k>;
using smith_waterman_serial_t =
    smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_cap_serial_k>;

using affine_levenshtein_serial_t = levenshtein_distances<affine_gap_costs_t, malloc_t, sz_cap_serial_k>;
using affine_levenshtein_utf8_serial_t = levenshtein_distances_utf8<affine_gap_costs_t, malloc_t, sz_cap_serial_k>;
using affine_needleman_wunsch_serial_t =
    needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k>;
using affine_smith_waterman_serial_t =
    smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_cap_serial_k>;

/**
 *  In @b AVX-512:
 *  - for Global Alignments, we can vectorize the min-max calculation for diagonal "walkers"
 *  - for Local Alignments, we can vectorize the character substitution lookups for horizontal "walkers"
 */
using levenshtein_icelake_t = levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sil_k>;
using levenshtein_utf8_icelake_t = levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_caps_sil_k>;
using needleman_wunsch_icelake_t =
    needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k>;
using smith_waterman_icelake_t =
    smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sil_k>;

using affine_levenshtein_icelake_t = levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sil_k>;
using affine_levenshtein_utf8_icelake_t = levenshtein_distances_utf8<affine_gap_costs_t, malloc_t, sz_caps_sil_k>;

using affine_needleman_wunsch_icelake_t =
    needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k>;
using affine_smith_waterman_icelake_t =
    smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sil_k>;

/**
 *  In @b AVX2 (Haswell) we vectorize the per-character substitution lookups for horizontal "walkers",
 *  emulating the Ice Lake `VPERMB` class lookup with high-nibble-selected `VPSHUFB` blends. The aliases are
 *  always declared (the composite capability is a plain constant); only their instantiation is `SZ_USE_HASWELL`-gated.
 */
using needleman_wunsch_haswell_t =
    needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sh_k>;
using smith_waterman_haswell_t = smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sh_k>;
using affine_needleman_wunsch_haswell_t =
    needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sh_k>;
using affine_smith_waterman_haswell_t =
    smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sh_k>;

/**
 *  In @b ARM NEON (AArch64) the per-character class lookups use the native `vqtbl4q_u8` byte-gather, feeding
 *  the anti-diagonal scorers; uniform-cost Levenshtein swaps that lookup for a `vceqq`/`vbslq` match-vs-mismatch
 *  select and minimizes with `vminq`. As with Ice Lake, the aliases are unconditional; only their instantiation
 *  is `SZ_USE_NEON`-gated.
 */
using levenshtein_neon_t = levenshtein_distances<linear_gap_costs_t, malloc_t, sz_caps_sn_k>;
using levenshtein_utf8_neon_t = levenshtein_distances_utf8<linear_gap_costs_t, malloc_t, sz_caps_sn_k>;
using needleman_wunsch_neon_t =
    needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sn_k>;
using smith_waterman_neon_t = smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, malloc_t, sz_caps_sn_k>;

using affine_levenshtein_neon_t = levenshtein_distances<affine_gap_costs_t, malloc_t, sz_caps_sn_k>;
using affine_needleman_wunsch_neon_t =
    needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sn_k>;
using affine_smith_waterman_neon_t =
    smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, malloc_t, sz_caps_sn_k>;

#pragma endregion - Common Aliases

#pragma region - Autovectorized Tile Scorer

/**
 *  This overload handles:
 *  - Only @b Global alignment, not Local!
 *  - Only @b Linear gaps, not Affine!
 *  - Both auto-vectorized @b Serial and @b Parallel execution, but not hand-rolled SIMD!
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_>
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_global_k, sz_cap_serial_k, void> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = remove_cvref<first_char_t>;

    using tile_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    score_t last_score_ {0};
    bool transpose_ {false};

  public:
    tile_scorer() = default;
    tile_scorer(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Selects the operand order for the substitution lookup.
     *  @param transpose When set, the two class operands are swapped, compensating for a walker that
     *      exchanged the shorter and longer strings around an @b asymmetric cost matrix.
     */
    void prepare(bool transpose) noexcept { transpose_ = transpose; }

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    void init_score(score_t &cell, size_t diagonal_index) const noexcept {
        cell = gap_costs_.open_or_extend * diagonal_index;
    }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    score_t score() const noexcept { return last_score_; }

    /** @brief Executor-independent trampoline, computing one diagonal fo the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                        //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice,       //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, //
        score_t const *scores_pre_deletion, score_t *scores_new, size_t from, size_t to) noexcept {

        error_cost_t const gap_cost = gap_costs_.open_or_extend;
        for (size_t i = from; i < to; ++i) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = transpose_ ? substituter_(second_slice[i], first_reversed_slice[i])
                                                           : substituter_(first_reversed_slice[i], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = min_or_max<objective_k>(pre_deletion, pre_insertion) + gap_cost;
            score_t cell_score = min_or_max<objective_k>(if_deletion_or_insertion, if_substitution);
            scores_new[i] = cell_score;
        }
    }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_reversed_slice The first string, @b reversed.
     *  @param second_slice The second string.
     *  @param n The length of the diagonal to evaluate and the number of characters to compare from each string.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, size_t n, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, score_t const *scores_pre_deletion,
        score_t *scores_new, executor_type_ &&executor = {}) noexcept {

        executor.for_slices(n, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, from, to);
        });

        // The last element of the last chunk is the result of the global alignment.
        last_score_ = scores_new[n - 1];
    }
};

/**
 *  This overload handles:
 *  - Only @b Local alignment, not Global!
 *  - Only @b Linear gaps, not Affine!
 *  - Both auto-vectorized @b Serial and @b Parallel execution, but not hand-rolled SIMD!
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_>
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_local_k, sz_cap_serial_k, void> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = first_char_t;

    using tile_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    score_t best_score_ {0};
    bool transpose_ {false};

  public:
    tile_scorer() = default;
    tile_scorer(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Selects the operand order for the substitution lookup.
     *  @param transpose When set, the two class operands are swapped, compensating for a walker that
     *      exchanged the shorter and longer strings around an @b asymmetric cost matrix.
     */
    void prepare(bool transpose) noexcept { transpose_ = transpose; }

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    void init_score(score_t &cell, size_t /* diagonal_index */) const noexcept { cell = 0; }

    /**
     *  @brief Extract the final result of the scoring operation which will be maximum encountered value.
     */
    score_t score() const noexcept { return best_score_; }

    /** @brief Executor-independent trampoline, computing one diagonal fo the DP matrix. */
    SZ_NOINLINE score_t score_slice_trampoline_(                                     //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice,       //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, //
        score_t const *scores_pre_deletion, score_t *scores_new, size_t from, size_t to,
        score_t running_best) noexcept {

        error_cost_t const gap_cost = gap_costs_.open_or_extend;
        for (size_t i = from; i < to; ++i) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = transpose_ ? substituter_(second_slice[i], first_reversed_slice[i])
                                                           : substituter_(first_reversed_slice[i], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = min_or_max<objective_k>(pre_deletion, pre_insertion) + gap_cost;
            // ! This is the main difference with global alignment:
            score_t if_substitution_or_reset = min_or_max<objective_k, score_t>(if_substitution, 0);
            score_t cell_score = min_or_max<objective_k>(if_deletion_or_insertion, if_substitution_or_reset);
            scores_new[i] = cell_score;

            // ! Update the global maximum score if this cell beats it - this is the costliest operation:
            running_best = min_or_max<objective_k>(running_best, cell_score);
        }
        return running_best;
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                           //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, size_t const n, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, score_t const *scores_pre_deletion,
        score_t *scores_new, executor_type_ &&executor = {}) noexcept {

        std::atomic<score_t> atomic_best_score {best_score_};
        executor.for_slices(n, [&](size_t i_start, size_t i_end) noexcept {
            score_t local_best_score = score_slice_trampoline_(
                first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion, scores_pre_deletion,
                scores_new, i_start, i_end, atomic_best_score);
            atomic_best_score = min_or_max<objective_k, score_t>(atomic_best_score, local_best_score);
        });
        best_score_ = min_or_max<objective_k, score_t>(best_score_, atomic_best_score);
    }
};

/**
 *  This overload handles:
 *  - Only @b Global alignment, not Local!
 *  - Only @b Affine gaps, not Linear!
 *  - Both auto-vectorized @b Serial and @b Parallel execution, but not hand-rolled SIMD!
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_>
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_global_k, sz_cap_serial_k, void> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = remove_cvref<first_char_t>;

    using tile_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    score_t last_score_ {0};
    bool transpose_ {false};

  public:
    tile_scorer() = default;
    tile_scorer(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Selects the operand order for the substitution lookup.
     *  @param transpose When set, the two class operands are swapped, compensating for a walker that
     *      exchanged the shorter and longer strings around an @b asymmetric cost matrix.
     */
    void prepare(bool transpose) noexcept { transpose_ = transpose; }

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    void init_score(score_t &cell, size_t diagonal_index) const noexcept {
        cell = diagonal_index ? gap_costs_.open + gap_costs_.extend * (diagonal_index - 1) : 0;
    }

    void init_gap(score_t &cell, size_t diagonal_index) const noexcept {
        // Make sure the initial value of the gap is not smaller in magnitude than the primary.
        // The supplementary matrices are initialized with values of higher magnitude,
        // which is equivalent to discarding them. That's better than using `SIZE_MAX`
        // as subsequent additions won't overflow.
        cell = (gap_costs_.open + gap_costs_.extend) +
               (diagonal_index ? gap_costs_.open + gap_costs_.extend * (diagonal_index - 1) : 0);
    }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    score_t score() const noexcept { return last_score_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_reversed_slice The first string, @b reversed.
     *  @param second_slice The second string.
     *  @param n The length of the diagonal to evaluate and the number of characters to compare from each string.
     */
    SZ_NOINLINE void score_slice_trampoline_(                                  //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, //
        score_t const *scores_pre_substitution,                                //
        score_t const *scores_pre_insertion,                                   //
        score_t const *scores_pre_deletion,                                    //
        score_t const *scores_running_insertions,                              //
        score_t const *scores_running_deletions,                               //
        score_t *scores_new,                                                   //
        score_t *scores_new_insertions,                                        //
        score_t *scores_new_deletions, size_t from, size_t to) noexcept {

        for (size_t i = from; i < to; ++i) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion_opening = scores_pre_insertion[i];
            score_t pre_deletion_opening = scores_pre_deletion[i];
            score_t pre_insertion_expansion = scores_running_insertions[i];
            score_t pre_deletion_expansion = scores_running_deletions[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = transpose_ ? substituter_(second_slice[i], first_reversed_slice[i])
                                                           : substituter_(first_reversed_slice[i], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_insertion = min_or_max<objective_k>(pre_insertion_opening + gap_costs_.open,
                                                           pre_insertion_expansion + gap_costs_.extend);
            score_t if_deletion = min_or_max<objective_k>(pre_deletion_opening + gap_costs_.open,
                                                          pre_deletion_expansion + gap_costs_.extend);
            score_t if_deletion_or_insertion = min_or_max<objective_k>(if_deletion, if_insertion);
            score_t cell_score = min_or_max<objective_k>(if_deletion_or_insertion, if_substitution);

            // Export results.
            scores_new[i] = cell_score;
            scores_new_insertions[i] = if_insertion;
            scores_new_deletions[i] = if_deletion;
        }
    }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_reversed_slice The first string, @b reversed.
     *  @param second_slice The second string.
     *  @param n The length of the diagonal to evaluate and the number of characters to compare from each string.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, size_t n, //
        score_t const *scores_pre_substitution,                                          //
        score_t const *scores_pre_insertion,                                             //
        score_t const *scores_pre_deletion,                                              //
        score_t const *scores_running_insertions,                                        //
        score_t const *scores_running_deletions,                                         //
        score_t *scores_new,                                                             //
        score_t *scores_new_insertions,                                                  //
        score_t *scores_new_deletions,                                                   //
        executor_type_ &&executor = {}) noexcept {

        executor.for_slices(n, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_running_insertions, scores_running_deletions,
                                    scores_new, scores_new_insertions, scores_new_deletions, from, to);
        });

        // The last element of the last chunk is the result of the global alignment.
        last_score_ = scores_new[n - 1];
    }
};

/**
 *  This overload handles:
 *  - Only @b Local alignment, not Global!
 *  - Only @b Affine gaps, not Linear!
 *  - Both auto-vectorized @b Serial and @b Parallel execution, but not hand-rolled SIMD!
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_>
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_local_k, sz_cap_serial_k, void> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = first_char_t;

    using tile_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    score_t best_score_ {0};
    bool transpose_ {false};

  public:
    tile_scorer() = default;
    tile_scorer(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Selects the operand order for the substitution lookup.
     *  @param transpose When set, the two class operands are swapped, compensating for a walker that
     *      exchanged the shorter and longer strings around an @b asymmetric cost matrix.
     */
    void prepare(bool transpose) noexcept { transpose_ = transpose; }

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    void init_score(score_t &cell, size_t /* diagonal_index */) const noexcept { cell = 0; }
    void init_gap(score_t &cell, size_t /* diagonal_index */) const noexcept {
        // Make sure the initial value of the gap is not smaller in magnitude than the primary.
        // The supplementary matrices are initialized with values of higher magnitude,
        // which is equivalent to discarding them. That's better than using `SIZE_MAX`
        // as subsequent additions won't overflow.
        cell = gap_costs_.open + gap_costs_.extend;
    }

    /**
     *  @brief Extract the final result of the scoring operation which will be maximum encountered value.
     */
    score_t score() const noexcept { return best_score_; }

    /** @brief Executor-independent trampoline, computing one diagonal fo the DP matrix. */
    SZ_NOINLINE score_t score_slice_trampoline_(                               //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, //
        score_t const *scores_pre_substitution,                                //
        score_t const *scores_pre_insertion,                                   //
        score_t const *scores_pre_deletion,                                    //
        score_t const *scores_running_insertions,                              //
        score_t const *scores_running_deletions,                               //
        score_t *scores_new,                                                   //
        score_t *scores_new_insertions,                                        //
        score_t *scores_new_deletions, size_t from, size_t to, score_t running_best) noexcept {

        for (size_t i = from; i < to; ++i) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion_opening = scores_pre_insertion[i];
            score_t pre_deletion_opening = scores_pre_deletion[i];
            score_t pre_insertion_expansion = scores_running_insertions[i];
            score_t pre_deletion_expansion = scores_running_deletions[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = transpose_ ? substituter_(second_slice[i], first_reversed_slice[i])
                                                           : substituter_(first_reversed_slice[i], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion = min_or_max<objective_k>(pre_deletion_opening + gap_costs_.open,
                                                          pre_deletion_expansion + gap_costs_.extend);
            score_t if_insertion = min_or_max<objective_k>(pre_insertion_opening + gap_costs_.open,
                                                           pre_insertion_expansion + gap_costs_.extend);
            score_t if_deletion_or_insertion = min_or_max<objective_k>(if_deletion, if_insertion);
            // ! This is the main difference with global alignment:
            score_t if_substitution_or_reset = min_or_max<objective_k, score_t>(if_substitution, 0);
            score_t cell_score = min_or_max<objective_k>(if_deletion_or_insertion, if_substitution_or_reset);

            // Export results.
            scores_new[i] = cell_score;
            scores_new_deletions[i] = if_deletion;
            scores_new_insertions[i] = if_insertion;

            // ! Update the global maximum score if this cell beats it - this is the costliest operation:
            running_best = min_or_max<objective_k>(running_best, cell_score);
        }
        return running_best;
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                           //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, size_t const n, //
        score_t const *scores_pre_substitution,                                                //
        score_t const *scores_pre_insertion,                                                   //
        score_t const *scores_pre_deletion,                                                    //
        score_t const *scores_running_insertions,                                              //
        score_t const *scores_running_deletions,                                               //
        score_t *scores_new,                                                                   //
        score_t *scores_new_insertions,                                                        //
        score_t *scores_new_deletions,                                                         //
        executor_type_ &&executor = {}) noexcept {

        std::atomic<score_t> atomic_best_score {best_score_};
        executor.for_slices(n, [&](size_t i_start, size_t i_end) noexcept {
            score_t local_best_score = score_slice_trampoline_(
                first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion, scores_pre_deletion,
                scores_running_insertions, scores_running_deletions, scores_new, scores_new_insertions,
                scores_new_deletions, i_start, i_end, atomic_best_score);
            atomic_best_score = min_or_max<objective_k, score_t>(atomic_best_score, local_best_score);
        });
        best_score_ = min_or_max<objective_k, score_t>(best_score_, atomic_best_score);
    }
};

#pragma endregion - Autovectorized Tile Scorer

#pragma region - Diagonal Walker

/**
 *  This overload handles:
 *  - Both @b Global and @b Local alignment!
 *  - Only @b Linear gaps, not Affine!
 *  - All @b CPU capability levels, but not used on the GPU side.
 *
 *  Allocates 3x diagonals of the DP matrix.
 */
template <typename char_or_rune_type_, typename score_type_, typename substituter_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename enable_>
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_>
#endif
struct diagonal_walker<char_or_rune_type_, score_type_, substituter_type_, linear_gap_costs_t, objective_, locality_,
                       capability_, enable_> {

    using char_t = char_or_rune_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = capability_;

    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    diagonal_walker() noexcept {}

    /**
     *  @param[in] subs A commutative function returning the cost of substituting one char with another.
     *  @param[in] gaps The uniform cost of a gap (insertion or deletion).
     *  @param[in] scratch_space A buffer to be used for running score values
     */
    diagonal_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Byte offsets of this walker's scratch sub-buffers within its `scratch_space`. */
    struct layout_t {
        size_t previous_scores = 0;  // ? First of the 3 rotating score diagonals.
        size_t current_scores = 0;   // ? Second of the 3 rotating score diagonals.
        size_t next_scores = 0;      // ? Third of the 3 rotating score diagonals.
        size_t shorter_reversed = 0; // ? Reversed copy of the shorter string, right after the diagonals.
        size_t total = 0;            // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /**
     *  @brief The single source of truth for this walker's scratch size and sub-buffer offsets.
     *  @return Cache-line-padded byte offsets of every sub-buffer, and their `total`.
     */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        size_t const diagonal_bytes = sizeof(score_t) * (shorter_length + 1); // one anti-diagonal, unpadded
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.previous_scores = amount, amount += diagonal_bytes;
        at.current_scores = amount, amount += diagonal_bytes;
        at.next_scores = amount, amount += diagonal_bytes;
        at.shorter_reversed = amount, amount += shorter_length * sizeof(char_t);
        at.total = amount;
        return at;
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) { result_ref = gap_costs_.open_or_extend * first.size(); }
                else if (first.empty() && !second.empty()) { result_ref = gap_costs_.open_or_extend * second.size(); }
            }
            return status_t::success_k;
        }

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
        size_t shorter_length = first.size(), longer_length = second.size();
        if (shorter_length > longer_length) {
            trivial_swap(shorter, longer);
            trivial_swap(shorter_length, longer_length);
        }

        // We are going to store 3 diagonals of the matrix.
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;

        // Let's say we are dealing with 3 and 5 letter words.
        // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
        // It will have:
        // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
        // - 2 diagonals of fixed length, at positions: 4, 5.
        // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
        size_t const diagonals_count = shorter_dim + longer_dim - 1;
        size_t const max_diagonal_length = shorter_length + 1;

        // One `layout()` describes every sub-buffer, so the sizing in `scratch_space_needed` and the pointers
        // here can never disagree. We validate the walker's own footprint; callers may hand us a larger span and
        // keep the surplus for their own bookkeeping (e.g. the UTF-8 backend's transcode region).
        layout_t const at = layout(first, second, specs);
        if (scratch_space.size() < at.total) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around. The reversed shorter string sits right after the diagonals,
        // so we can avoid reverse-order iteration over the shorter string.
        score_t *previous_scores = (score_t *)(scratch_space.data() + at.previous_scores);
        score_t *current_scores = (score_t *)(scratch_space.data() + at.current_scores);
        score_t *next_scores = (score_t *)(scratch_space.data() + at.next_scores);
        char_t *const shorter_reversed = (char_t *)(scratch_space.data() + at.shorter_reversed);

        // Export the reversed string into the buffer.
        for (size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

        // Initialize the first two diagonals:
        tile_scorer_t scorer {substituter_, gap_costs_};
        // The walker exchanges the strings so the shorter one is reversed into the `first` operand; an
        // asymmetric cost matrix must then be looked up transposed when that exchange happened.
        scorer.prepare(first.size() > second.size());
        scorer.init_score(previous_scores[0], 0);
        scorer.init_score(current_scores[0], 1);
        scorer.init_score(current_scores[1], 1);

        // We skip diagonals 0 and 1, as they are trivial.
        // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
        // so we are effectively computing just one value, as will be marked by a single set bit in
        // the `next_diagonal_mask` on the very first iteration.
        size_t next_diagonal_index = 2;

        // Progress through the upper-left triangle of the Levenshtein matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = next_diagonal_index + 1;
            scorer(                                                          //
                shorter_reversed + shorter_length - next_diagonal_index + 1, // first sequence of characters
                longer,                                                      // second sequence of characters
                next_diagonal_length - 2,           // number of elements to compute with the `scorer`
                previous_scores,                    // costs pre substitution
                current_scores, current_scores + 1, // costs pre insertion/deletion
                next_scores + 1,                    // new scores for the next diagonal
                executor);                          // parallel execution within the diagonal

            // Don't forget to populate the first row and the first column of the Levenshtein matrix.
            scorer.init_score(next_scores[0], next_diagonal_index);
            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);

            // Perform a circular rotation of those buffers, to reuse the memory.
            rotate_three(previous_scores, current_scores, next_scores);
        }

        // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = shorter_dim;
            scorer(                                                  //
                shorter_reversed + shorter_length - shorter_dim + 1, // first sequence of characters
                longer + next_diagonal_index - shorter_dim,          // second sequence of characters
                next_diagonal_length - 1,                            // number of elements to compute with the `scorer`
                previous_scores,                                     // costs pre substitution
                current_scores, current_scores + 1,                  // costs pre insertion/deletion
                next_scores,                                         // new scores for the next diagonal
                executor);                                           // parallel execution within the diagonal

            // Don't forget to populate the first row of the Levenshtein matrix.
            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);

            // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
            // dropping the first element in the current array.
            rotate_three(previous_scores, current_scores, next_scores);

            // ! Drop the first entry among the current scores.
            sz_move_serial((ptr_t)(previous_scores), (ptr_t)(previous_scores + 1),
                           (max_diagonal_length - 1) * sizeof(score_t));
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
            scorer(                                                  //
                shorter_reversed + shorter_length - shorter_dim + 1, // first sequence of characters
                longer + next_diagonal_index - shorter_dim,          // second sequence of characters
                next_diagonal_length,                                // number of elements to compute with the `scorer`
                previous_scores,                                     // costs pre substitution
                current_scores, current_scores + 1,                  // costs pre insertion/deletion
                next_scores,                                         // new scores for the next diagonal
                executor);                                           // parallel execution within the diagonal

            // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
            // dropping the first element in the current array.
            rotate_three(previous_scores, current_scores, next_scores);

            // ! Drop the first entry among the current scores.
            // ! Assuming every next diagonal is shorter by one element,
            // ! we don't need a full-blown `sz_move_serial` to shift the array by one element.
            previous_scores++;
        }

        // Export the scalar before `free` call.
        result_ref = scorer.score();
        return status_t::success_k;
    }
};

/**
 *  This overload handles:
 *  - Both @b Global and @b Local alignment!
 *  - Only @b Affine gaps, not Linear!
 *  - All @b CPU capability levels, but not used on the GPU side.
 *
 *  Allocates 3x diagonals of the DP matrix and 2x diagonals of 2x affine gaps matrices.
 */
template <typename char_or_rune_type_, typename score_type_, typename substituter_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename enable_>
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_>
#endif
struct diagonal_walker<char_or_rune_type_, score_type_, substituter_type_, affine_gap_costs_t, objective_, locality_,
                       capability_, enable_> {

    using char_t = char_or_rune_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_gaps_t gaps_k = sz_gaps_affine_k;
    static constexpr sz_capability_t capability_k = capability_;

    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    diagonal_walker() noexcept {}

    /**
     *  @param[in] subs A commutative function returning the cost of substituting one char with another.
     *  @param[in] gaps The affine costs of opening and extending a gap.
     */
    diagonal_walker(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Byte offsets of this walker's scratch sub-buffers within its `scratch_space`. */
    struct layout_t {
        size_t previous_scores = 0; // ? The 3 rotating score diagonals.
        size_t current_scores = 0;
        size_t next_scores = 0;
        size_t current_inserts = 0; // ? The 2 rotating insertion-gap diagonals.
        size_t next_inserts = 0;
        size_t current_deletes = 0; // ? The 2 rotating deletion-gap diagonals.
        size_t next_deletes = 0;
        size_t shorter_reversed = 0; // ? Reversed copy of the shorter string, right after the diagonals.
        size_t total = 0;            // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /**
     *  @brief The single source of truth for this walker's scratch size and sub-buffer offsets.
     *  @return Cache-line-padded byte offsets of every sub-buffer, and their `total`.
     */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        size_t const diagonal_bytes = sizeof(score_t) * (shorter_length + 1); // one anti-diagonal, unpadded
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.previous_scores = amount, amount += diagonal_bytes;
        at.current_scores = amount, amount += diagonal_bytes;
        at.next_scores = amount, amount += diagonal_bytes;
        at.current_inserts = amount, amount += diagonal_bytes;
        at.next_inserts = amount, amount += diagonal_bytes;
        at.current_deletes = amount, amount += diagonal_bytes;
        at.next_deletes = amount, amount += diagonal_bytes;
        at.shorter_reversed = amount, amount += shorter_length * sizeof(char_t);
        at.total = amount;
        return at;
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     *  @param[in] scratch_space A buffer to be used for running score values
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) {
                    result_ref = gap_costs_.open + gap_costs_.extend * (first.size() - 1);
                }
                else if (first.empty() && !second.empty()) {
                    result_ref = gap_costs_.open + gap_costs_.extend * (second.size() - 1);
                }
            }
            return status_t::success_k;
        }

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
        size_t shorter_length = first.size(), longer_length = second.size();
        if (shorter_length > longer_length) {
            trivial_swap(shorter, longer);
            trivial_swap(shorter_length, longer_length);
        }

        // We are going to store 7 diagonals of the matrix.
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;

        // Let's say we are dealing with 3 and 5 letter words.
        // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
        // It will have:
        // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
        // - 2 diagonals of fixed length, at positions: 4, 5.
        // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
        size_t const diagonals_count = shorter_dim + longer_dim - 1;
        size_t const max_diagonal_length = shorter_length + 1;

        // One `layout()` describes every sub-buffer (7 diagonals + the reversed shorter string), so the sizing
        // in `scratch_space_needed` and the pointers here can never disagree. We validate the walker's own
        // footprint; callers may hand us a larger span and keep the surplus for their own bookkeeping.
        layout_t const at = layout(first, second, specs);
        if (scratch_space.size() < at.total) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around. The reversed shorter string sits right after the diagonals,
        // so we can avoid reverse-order iteration over the shorter string.
        score_t *previous_scores = (score_t *)(scratch_space.data() + at.previous_scores);
        score_t *current_scores = (score_t *)(scratch_space.data() + at.current_scores);
        score_t *next_scores = (score_t *)(scratch_space.data() + at.next_scores);
        score_t *current_inserts = (score_t *)(scratch_space.data() + at.current_inserts);
        score_t *next_inserts = (score_t *)(scratch_space.data() + at.next_inserts);
        score_t *current_deletes = (score_t *)(scratch_space.data() + at.current_deletes);
        score_t *next_deletes = (score_t *)(scratch_space.data() + at.next_deletes);
        char_t *const shorter_reversed = (char_t *)(scratch_space.data() + at.shorter_reversed);

        // Export the reversed string into the buffer.
        for (size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

        // Initialize the first two diagonals:
        tile_scorer_t scorer {substituter_, gap_costs_};
        // The walker exchanges the strings so the shorter one is reversed into the `first` operand; an
        // asymmetric cost matrix must then be looked up transposed when that exchange happened.
        scorer.prepare(first.size() > second.size());
        scorer.init_score(previous_scores[0], 0);
        scorer.init_score(current_scores[0], 1);
        scorer.init_score(current_scores[1], 1);
        scorer.init_gap(current_inserts[0], 1);
        scorer.init_gap(current_deletes[1], 1);

        // We skip diagonals 0 and 1, as they are trivial.
        // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
        // so we are effectively computing just one value, as will be marked by a single set bit in
        // the `next_diagonal_mask` on the very first iteration.
        size_t next_diagonal_index = 2;

        // Progress through the upper-left triangle of the Levenshtein matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = next_diagonal_index + 1;
            scorer(                                                          //
                shorter_reversed + shorter_length - next_diagonal_index + 1, // first sequence of characters
                longer,                                                      // second sequence of characters
                next_diagonal_length - 2,             // number of elements to compute with the `scorer`
                previous_scores,                      // costs pre substitution
                current_scores, current_scores + 1,   // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1, // costs pre insertion/deletion extension
                next_scores + 1,                      // updated similarity scores
                next_inserts + 1, next_deletes + 1,   // updated insertion/deletion extensions
                executor                              // parallel execution within the diagonal
            );

            // Don't forget to populate the first row and the first column of the Levenshtein matrix.
            scorer.init_score(next_scores[0], next_diagonal_index);
            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            scorer.init_gap(next_inserts[0], next_diagonal_index);
            scorer.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);

            // Perform a circular rotation of those buffers, to reuse the memory.
            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);
        }

        // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = shorter_dim;
            scorer(                                                  //
                shorter_reversed + shorter_length - shorter_dim + 1, // first sequence of characters
                longer + next_diagonal_index - shorter_dim,          // second sequence of characters
                next_diagonal_length - 1,                            // number of elements to compute with the `scorer`
                previous_scores,                                     // costs pre substitution
                current_scores, current_scores + 1,                  // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,                // costs pre insertion/deletion extension
                next_scores,                                         // updated similarity scores
                next_inserts, next_deletes,                          // updated insertion/deletion extensions
                executor                                             // parallel execution within the diagonal
            );

            // Don't forget to populate the first row of the Levenshtein matrix.
            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            scorer.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);

            // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
            // dropping the first element in the current array.
            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);

            // ! Drop the first entry among the current scores.
            sz_move_serial((ptr_t)(previous_scores), (ptr_t)(previous_scores + 1),
                           (max_diagonal_length - 1) * sizeof(score_t));
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
            scorer(                                                  //
                shorter_reversed + shorter_length - shorter_dim + 1, // first sequence of characters
                longer + next_diagonal_index - shorter_dim,          // second sequence of characters
                next_diagonal_length,                                // number of elements to compute with the `scorer`
                previous_scores,                                     // costs pre substitution
                current_scores, current_scores + 1,                  // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,                // costs pre insertion/deletion extension
                next_scores,                                         // updated similarity scores
                next_inserts, next_deletes,                          // updated insertion/deletion extensions
                executor                                             // parallel execution within the diagonal
            );

            // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
            // dropping the first element in the current array.
            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);

            // ! Drop the first entry among the current scores.
            // ! Assuming every next diagonal is shorter by one element,
            // ! we don't need a full-blown `sz_move_serial` to shift the array by one element.
            previous_scores++;
        }

        // Export the scalar before `free` call.
        result_ref = scorer.score();
        return status_t::success_k;
    }
};

#pragma endregion - Diagonal Walker
#pragma region - Horizontal Walker

/**
 *  This overload handles:
 *  - Both @b Global and @b Local alignment!
 *  - Only @b Linear gaps, not Affine!
 *  - All @b CPU capability levels, but not used on the GPU side.
 *
 *  Allocates 2x rows of the DP matrix.
 */
template <typename char_or_rune_type_, typename score_type_, typename substituter_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_>
#endif
struct horizontal_walker<char_or_rune_type_, score_type_, substituter_type_, linear_gap_costs_t, objective_, locality_,
                         sz_cap_serial_k, void> {

    using char_t = char_or_rune_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    using walker_t =
        horizontal_walker<char_t, score_t, substituter_t, gap_costs_t, objective_k, locality_k, capability_k, void>;
    using tile_scorer_t = tile_scorer<constant_iterator<char_t>, char_t const *, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    horizontal_walker() noexcept {}

    /**
     *  @param[in] subs A commutative function returning the cost of substituting one char with another.
     *  @param[in] gaps The uniform cost of a gap (insertion or deletion).
     *
     */
    horizontal_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Byte offsets of the two cache-line-padded rows this walker rolls over. */
    struct layout_t {
        size_t previous_row = 0; // ? The 2 rotating score rows.
        size_t current_row = 0;
        size_t total = 0; // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = sizeof(score_t) * (sz_min_of_two(first.size(), second.size()) + 1);
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.previous_row = amount, amount += row_bytes;
        at.current_row = amount, amount += row_bytes;
        at.total = amount;
        return at;
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     *  @param[in] scratch_space A buffer to be used for running score values
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) { result_ref = gap_costs_.open_or_extend * first.size(); }
                else if (first.empty() && !second.empty()) { result_ref = gap_costs_.open_or_extend * second.size(); }
            }
            return status_t::success_k;
        }

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
        size_t shorter_length = first.size(), longer_length = second.size();
        if (shorter_length > longer_length) {
            trivial_swap(shorter, longer);
            trivial_swap(shorter_length, longer_length);
        }

        // We are going to store 2 rows of the matrix. It will be either 2 rows of length `shorter_length + 1`
        // or 2 rows of length `longer_length + 1`, depending on our preference - either minimizing the memory
        // consumption or the inner loop performance.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;

        // One `layout()` describes both rows; validate the walker's own footprint and place the pointers from it.
        layout_t const at = layout(first, second, specs);
        if (scratch_space.size() < at.total) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = (score_t *)(scratch_space.data() + at.previous_row);
        score_t *current_scores = (score_t *)(scratch_space.data() + at.current_row);

        // Initialize the first row:
        tile_scorer_t scorer {substituter_, gap_costs_};
        // The horizontal walker broadcasts characters of the @b longer string into the `first` operand, so the
        // transpose condition is inverted relative to the diagonal walker: compensate when no exchange happened.
        scorer.prepare(first.size() <= second.size());
        for (size_t col_idx = 0; col_idx < shorter_dim; ++col_idx) scorer.init_score(previous_scores[col_idx], col_idx);

        // Progress through the matrix row-by-row:
        for (size_t row_idx = 1; row_idx < longer_dim; ++row_idx) {

            // Don't forget to populate the first column of each row:
            scorer.init_score(current_scores[0], row_idx);

            scorer(                                              //
                constant_iterator<char_t> {longer[row_idx - 1]}, // first sequence of characters
                shorter,                                         // second sequence of characters
                shorter_dim - 1,                                 // number of elements to compute with the `scorer`
                previous_scores,                                 // costs pre substitution
                previous_scores + 1,                             // costs pre insertion
                current_scores,                                  // costs pre deletion
                current_scores + 1,                              // new scores
                executor                                         // ! note, most horizontal scorers are not parallel
            );

            // Reuse the memory.
            trivial_swap(previous_scores, current_scores);
        }

        // Export the scalar before `free` call.
        result_ref = scorer.score();
        return status_t::success_k;
    }
};

/**
 *  This overload handles:
 *  - Both @b Global and @b Local alignment!
 *  - Only @b Affine gaps, not Linear!
 *  - All @b CPU capability levels, but not used on the GPU side.
 *
 *  Allocates 2x rows of the DP matrix and 2x rows of 2x affine gaps matrices.
 */
template <typename char_or_rune_type_, typename score_type_, typename substituter_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_>
#endif
struct horizontal_walker<char_or_rune_type_, score_type_, substituter_type_, affine_gap_costs_t, objective_, locality_,
                         sz_cap_serial_k, void> {

    using char_t = char_or_rune_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    using walker_t =
        horizontal_walker<char_t, score_t, substituter_t, gap_costs_t, objective_k, locality_k, capability_k, void>;
    using tile_scorer_t = tile_scorer<constant_iterator<char_t>, char_t const *, score_t, substituter_t, gap_costs_t,
                                      objective_k, locality_k, capability_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    horizontal_walker() noexcept {}

    /**
     *  @param[in] subs A commutative function returning the cost of substituting one char with another.
     *  @param[in] gap_opening_cost The cost of opening a gap (insertion or deletion).
     *  @param[in] gap_extension_cost The cost of extending a gap (insertion or deletion).
     */
    horizontal_walker(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Byte offsets of the two cache-line-padded rows of each of the 3 affine matrices. */
    struct layout_t {
        size_t previous_scores = 0; // ? The 2 rotating score rows.
        size_t current_scores = 0;
        size_t previous_inserts = 0; // ? The 2 rotating insertion-gap rows.
        size_t current_inserts = 0;
        size_t previous_deletes = 0; // ? The 2 rotating deletion-gap rows.
        size_t current_deletes = 0;
        size_t total = 0; // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = sizeof(score_t) * (sz_min_of_two(first.size(), second.size()) + 1);
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.previous_scores = amount, amount += row_bytes;
        at.current_scores = amount, amount += row_bytes;
        at.previous_inserts = amount, amount += row_bytes;
        at.current_inserts = amount, amount += row_bytes;
        at.previous_deletes = amount, amount += row_bytes;
        at.current_deletes = amount, amount += row_bytes;
        at.total = amount;
        return at;
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     *  @param[in] scratch_space A buffer to be used for running score values
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) {
                    result_ref = static_cast<score_t>(gap_costs_.open + gap_costs_.extend * (first.size() - 1));
                }
                else if (first.empty() && !second.empty()) {
                    result_ref = static_cast<score_t>(gap_costs_.open + gap_costs_.extend * (second.size() - 1));
                }
            }
            return status_t::success_k;
        }

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
        size_t shorter_length = first.size(), longer_length = second.size();
        if (shorter_length > longer_length) {
            trivial_swap(shorter, longer);
            trivial_swap(shorter_length, longer_length);
        }

        // We are going to store 2 rows of the matrix. It will be either 2 rows of length `shorter_length + 1`
        // or 2 rows of length `longer_length + 1`, depending on our preference - either minimizing the memory
        // consumption or the inner loop performance.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;

        // One `layout()` describes the 2 rows of each of the 3 affine matrices; validate the walker's own
        // footprint and place the pointers from it.
        layout_t const at = layout(first, second, specs);
        if (scratch_space.size() < at.total) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = (score_t *)(scratch_space.data() + at.previous_scores);
        score_t *current_scores = (score_t *)(scratch_space.data() + at.current_scores);
        score_t *previous_inserts = (score_t *)(scratch_space.data() + at.previous_inserts);
        score_t *current_inserts = (score_t *)(scratch_space.data() + at.current_inserts);
        score_t *previous_deletes = (score_t *)(scratch_space.data() + at.previous_deletes);
        score_t *current_deletes = (score_t *)(scratch_space.data() + at.current_deletes);

        // Initialize the first row:
        tile_scorer_t scorer {substituter_, gap_costs_};
        // The horizontal walker broadcasts characters of the @b longer string into the `first` operand, so the
        // transpose condition is inverted relative to the diagonal walker: compensate when no exchange happened.
        scorer.prepare(first.size() <= second.size());
        previous_scores[0] = 0;
        for (size_t col_idx = 1; col_idx < shorter_dim; ++col_idx) {
            scorer.init_score(previous_scores[col_idx], col_idx);
            scorer.init_gap(previous_deletes[col_idx], col_idx);
        }

        // Progress through the matrix row-by-row:
        for (size_t row_idx = 1; row_idx < longer_dim; ++row_idx) {

            // Don't forget to populate the first column of each row:
            scorer.init_score(current_scores[0], row_idx);
            scorer.init_gap(current_inserts[0], row_idx);

            scorer(                                              //
                constant_iterator<char_t> {longer[row_idx - 1]}, // first sequence of characters
                shorter,                                         // second sequence of characters
                shorter_dim - 1,                                 // number of elements to compute with the `scorer`
                previous_scores,                                 // costs pre substitution
                current_scores, previous_scores + 1,             // costs pre insertion/deletion opening
                current_inserts, previous_deletes + 1,           // costs pre insertion/deletion extension
                current_scores + 1,                              // updated similarity scores
                current_inserts + 1, current_deletes + 1,        // updated insertion/deletion extensions
                executor                                         // ! note, most horizontal scorers are not parallel
            );

            // Reuse the memory.
            trivial_swap(previous_scores, current_scores);
            trivial_swap(previous_inserts, current_inserts);
            trivial_swap(previous_deletes, current_deletes);
        }

        // Export the scalar before `free` call.
        result_ref = scorer.score();
        return status_t::success_k;
    }
};

#pragma endregion - Horizontal Walker

#pragma endregion - Algorithm Building Blocks

#pragma region - Pairwise Algorithms on CPU

template <>
struct levenshtein_distance_myers<char, sz_cap_serial_k> {

    using char_t = char;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    levenshtein_distance_myers() noexcept {}

    /** @brief Byte offset of the bit-parallel `match_masks[words_count][256]` table. */
    struct layout_t {
        size_t match_masks = 0; // ? The per-word `Peq` tables (256 entries each).
        size_t total = 0;       // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        // Must match the tier the dispatch in `operator()` picks (1/2/4/8 words -> shorter <= 64/128/256/512).
        size_t const words_count = shorter_length <= 64 ? 1 : shorter_length <= 128 ? 2 : shorter_length <= 256 ? 4 : 8;
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.match_masks = amount, amount += sizeof(u64_t) * words_count * 256;
        at.total = amount;
        return at;
    }

    /**
     *  @brief Bit-parallel Myers/Hyyrö Levenshtein for one pair whose @b shorter side is at most 64 runes.
     *      Exact @b unit-cost edit distance in O(longer) word-operations, 64 DP cells per machine word,
     *      no DP matrix and no allocation.
     */
    using index_t = u32_t;

    /**
     *  @brief Bit-parallel Myers/Hyyrö unit-cost Levenshtein for one pair whose @b shorter side is at most
     *      `words_count_ * 64` runes, carrying the horizontal deltas across the `words_count_` 64-bit blocks.
     *      Exact edit distance in O(longer * words_count_) word-operations, 64 DP cells per machine word, no DP
     *      matrix and (beyond the scratch `match_masks` table) no allocation. `words_count_` is a power of two so
     *      it lines up with the Ice Lake lockstep family; trailing empty blocks stay at the boundary harmlessly.
     */
    template <index_t words_count_> // 1, 2, 4, 8  ->  shorter <= 64, 128, 256, 512
    inline status_t unrolled_(span<char const> shorter, span<char const> longer, size_t &result_ref,
                              scratch_space_t scratch_space) const noexcept {
        index_t const shorter_length = (index_t)shorter.size();
        size_t const longer_length = longer.size();
        if (shorter_length > words_count_ * 64) return status_t::unexpected_dimensions_k;
        // Empty pattern: the distance is the text length, and the top-bit read below would underflow.
        if (shorter_length == 0) {
            result_ref = longer_length;
            return status_t::success_k;
        }
        if (scratch_space.size() < sizeof(u64_t) * words_count_ * 256) return status_t::bad_alloc_k;

        using match_masks_t = u64_t[words_count_][256];
        match_masks_t &match_masks = *reinterpret_cast<match_masks_t *>(scratch_space.data());
        // Every block of every touched character is read, so zero them all before building the `Peq` table: the
        // scratch is uninitialized (and may hold a previous walker's bytes).
        for (index_t position = 0; position != shorter_length; ++position)
            for (index_t word = 0; word != words_count_; ++word) match_masks[word][(u8_t)shorter[position]] = 0;
        for (size_t position = 0; position != longer_length; ++position)
            for (index_t word = 0; word != words_count_; ++word) match_masks[word][(u8_t)longer[position]] = 0;
        for (index_t position = 0; position != shorter_length; ++position)
            match_masks[position >> 6][(u8_t)shorter[position]] |= (u64_t)1 << (position & 63);

        u64_t vertical_positives[words_count_], vertical_negatives[words_count_]; // Myers' VP / VN, per block
        for (index_t word = 0; word != words_count_; ++word)
            vertical_positives[word] = ~(u64_t)0, vertical_negatives[word] = 0;
        index_t const last_word = (shorter_length - 1) >> 6, last_bit = (shorter_length - 1) & 63;
        size_t distance = shorter_length;
        for (size_t longer_position = 0; longer_position != longer_length; ++longer_position) {
            u8_t const symbol = (u8_t)longer[longer_position];
            u64_t horizontal_positive_carry = 1, horizontal_negative_carry = 0; // Top-row boundary into block 0 is +1.
            for (index_t word = 0; word != words_count_; ++word) {
                u64_t const pattern_matches = match_masks[word][symbol];
                u64_t const vertical_carry = pattern_matches | vertical_negatives[word];
                u64_t const matched_with_carry = pattern_matches | horizontal_negative_carry;
                u64_t const diagonal_zero =
                    (((matched_with_carry & vertical_positives[word]) + vertical_positives[word]) ^
                     vertical_positives[word]) |
                    matched_with_carry;
                u64_t horizontal_positive = vertical_negatives[word] | ~(diagonal_zero | vertical_positives[word]);
                u64_t horizontal_negative = vertical_positives[word] & diagonal_zero;
                if (word == last_word) {
                    distance += (horizontal_positive >> last_bit) & 1;
                    distance -= (horizontal_negative >> last_bit) & 1;
                }
                u64_t const horizontal_positive_carry_next = horizontal_positive >> 63;
                u64_t const horizontal_negative_carry_next = horizontal_negative >> 63;
                horizontal_positive = (horizontal_positive << 1) | horizontal_positive_carry;
                horizontal_negative = (horizontal_negative << 1) | horizontal_negative_carry;
                horizontal_positive_carry = horizontal_positive_carry_next;
                horizontal_negative_carry = horizontal_negative_carry_next;
                vertical_positives[word] = horizontal_negative | ~(vertical_carry | horizontal_positive);
                vertical_negatives[word] = horizontal_positive & vertical_carry;
            }
        }

        for (index_t position = 0; position != shorter_length; ++position)
            match_masks[position >> 6][(u8_t)shorter[position]] = 0;
        result_ref = distance;
        return status_t::success_k;
    }

    inline status_t operator()(span<char const> first, span<char const> second, size_t &result_ref,
                               scratch_space_t scratch_space) noexcept {
        bool const first_is_shorter = first.size() <= second.size();
        span<char const> shorter = first_is_shorter ? first : second;
        span<char const> longer = first_is_shorter ? second : first;
        size_t const shorter_length = shorter.size();
        if (shorter_length <= 64) return unrolled_<1>(shorter, longer, result_ref, scratch_space);
        if (shorter_length <= 128) return unrolled_<2>(shorter, longer, result_ref, scratch_space);
        if (shorter_length <= 256) return unrolled_<4>(shorter, longer, result_ref, scratch_space);
        return unrolled_<8>(shorter, longer, result_ref, scratch_space); // shorter <= 512
    }
};

/**
 *  @brief Computes the @b byte-level Levenshtein distance between two strings using the CPU backend.
 *  @sa `levenshtein_distance_utf8` for UTF-8 strings.
 *
 *  @tparam char_or_rune_type_ Can be any POD integer type, but @b `char` and @b `rune_t` are preferred.
 *  @tparam gap_costs_type_ Can be either `linear_gap_costs_t` or `affine_gap_costs_t`.
 *  @tparam capability_ Can be either `sz_cap_serial_k`, `sz_caps_sil_k`, `sz_cap_cuda_k`.
 */
template <                                         //
    typename char_or_rune_type_ = char,            //
    typename gap_costs_type_ = linear_gap_costs_t, //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename enable_ = void                        //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distance {

    using char_t = char_or_rune_type_;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = serialize_capability(capability_k);

    using myers_t = levenshtein_distance_myers<char_t, capability_serialized_k>;
    using horizontal_u8_t =                                                        //
        horizontal_walker<char_t, u8_t, uniform_substitution_costs_t, gap_costs_t, //
                          sz_minimize_distance_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_u8_t =                                                        //
        diagonal_walker<char_t, u8_t, uniform_substitution_costs_t, gap_costs_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_u16_t =                                                        //
        diagonal_walker<char_t, u16_t, uniform_substitution_costs_t, gap_costs_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u32_t =                                                        //
        diagonal_walker<char_t, u32_t, uniform_substitution_costs_t, gap_costs_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u64_t =                                                        //
        diagonal_walker<char_t, u64_t, uniform_substitution_costs_t, gap_costs_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;

    using linearized_fallback_t = levenshtein_distance<char_t, linear_gap_costs_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};

    levenshtein_distance() noexcept {}
    levenshtein_distance(uniform_substitution_costs_t subs, gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {

        // Redirect to linear costs
        if constexpr (is_same_type<gap_costs_t, affine_gap_costs_t>::value)
            if (gap_costs_.open == gap_costs_.extend) {
                linear_gap_costs_t linear_gap {gap_costs_.open};
                linearized_fallback_t linear_backend(substituter_, linear_gap);
                return linear_backend.scratch_space_needed(first, second, specs);
            }

        // Myers fast path, only when the shorter side fits its 512-rune limit (matches the guard in `operator()`).
        if constexpr (is_same_type<gap_costs_t, linear_gap_costs_t>::value && sizeof(char_t) == 1)
            if (substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1 &&
                (std::min)(first.size(), second.size()) <= 512) {
                return myers_t {}.layout(first, second, specs);
            }

        // Estimate the maximum dimension of the DP matrix and choose the best type for it. The dispatch below must
        // mirror `operator()` so the chosen walker's `layout()` is the single authority for both sizing and running.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<size_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        if (requirements.bytes_per_cell <= 1 && requirements.max_diagonal_length < 16)
            return horizontal_u8_t {substituter_, gap_costs_}.layout(first, second, specs);
        if (requirements.bytes_per_cell <= 1)
            return diagonal_u8_t {substituter_, gap_costs_}.layout(first, second, specs);
        if (requirements.bytes_per_cell == 2)
            return diagonal_u16_t {substituter_, gap_costs_}.layout(first, second, specs);
        if (requirements.bytes_per_cell == 4)
            return diagonal_u32_t {substituter_, gap_costs_}.layout(first, second, specs);
        return diagonal_u64_t {substituter_, gap_costs_}.layout(first, second, specs);
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        // If the cost of gap opening and extension is the same and we've mistakenly instantiated
        // the more memory-intensive `affine_gap_costs_t`, we can fall-back to the linearized version.
        if constexpr (is_same_type<gap_costs_t, affine_gap_costs_t>::value)
            if (gap_costs_.open == gap_costs_.extend) {
                linear_gap_costs_t linear_gap {gap_costs_.open};
                linearized_fallback_t linear_backend(substituter_, linear_gap);
                return linear_backend(first, second, result_ref, scratch_space, executor, specs);
            }

        // Bit-parallel Myers fast path (~5x DP on short unit-cost pairs); only when the shorter side fits Myers'
        // 512-rune limit, otherwise fall through to the anti-diagonal DP below.
        if constexpr (is_same_type<gap_costs_t, linear_gap_costs_t>::value && sizeof(char_t) == 1)
            if (substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1 &&
                (std::min)(first.size(), second.size()) <= 512)
                return myers_t {}(first, second, result_ref, scratch_space);

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<size_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        if (requirements.bytes_per_cell <= 1 && requirements.max_diagonal_length < 16) {
            u8_t result_u8 = std::numeric_limits<u8_t>::max();
            status_t status = horizontal_u8_t {substituter_, gap_costs_}(first, second, result_u8, scratch_space,
                                                                         executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        else if (requirements.bytes_per_cell <= 1) {
            u8_t result_u8 = std::numeric_limits<u8_t>::max();
            status_t status = diagonal_u8_t {substituter_, gap_costs_}(first, second, result_u8, scratch_space,
                                                                       executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }
        else if (requirements.bytes_per_cell == 2) {
            u16_t result_u16 = std::numeric_limits<u16_t>::max();
            status_t status = diagonal_u16_t {substituter_, gap_costs_}(first, second, result_u16, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u16;
        }
        else if (requirements.bytes_per_cell == 4) {
            u32_t result_u32 = std::numeric_limits<u32_t>::max();
            status_t status = diagonal_u32_t {substituter_, gap_costs_}(first, second, result_u32, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u32;
        }
        else if (requirements.bytes_per_cell == 8) {
            u64_t result_u64 = std::numeric_limits<u64_t>::max();
            status_t status = diagonal_u64_t {substituter_, gap_costs_}(first, second, result_u64, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Computes the @b rune-level Levenshtein distance between two UTF-8 strings using the CPU backend.
 *  @sa `levenshtein_distance` for binary strings.
 */
template <                                         //
    typename gap_costs_type_ = linear_gap_costs_t, //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename enable_ = void                        //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distance_utf8 {

    using gap_costs_t = gap_costs_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = serialize_capability(capability_k);

    using horizontal_u8_t = horizontal_walker<rune_t, u8_t, uniform_substitution_costs_t, gap_costs_t, //
                                              sz_minimize_distance_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_u8_t = diagonal_walker<rune_t, u8_t, uniform_substitution_costs_t, gap_costs_t, //
                                          sz_minimize_distance_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_u16_t = diagonal_walker<rune_t, u16_t, uniform_substitution_costs_t, gap_costs_t, //
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u32_t = diagonal_walker<rune_t, u32_t, uniform_substitution_costs_t, gap_costs_t, //
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u64_t = diagonal_walker<rune_t, u64_t, uniform_substitution_costs_t, gap_costs_t, //
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_k>;

    using linearized_fallback_t = levenshtein_distance<char, linear_gap_costs_t, capability_k>;
    using ascii_fallback_t = levenshtein_distance<char, gap_costs_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};

    levenshtein_distance_utf8() noexcept {}
    levenshtein_distance_utf8(uniform_substitution_costs_t subs, gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char const> first, span<char const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const first_unpacking_ceiling = round_up_to_multiple(sizeof(rune_t) * first.size(),
                                                                    specs.cache_line_width);
        size_t const second_unpacking_ceiling = round_up_to_multiple(sizeof(rune_t) * second.size(),
                                                                     specs.cache_line_width);
        size_t const transcode_bytes = first_unpacking_ceiling + second_unpacking_ceiling;

        // The UTF-8 path transcodes both strings into the front of scratch, then runs a @b rune diagonal walker on
        // the remainder. The walker sees at most `first.size()`/`second.size()` runes (one rune per byte in the
        // worst case), and its reversed-rune buffer costs `runes * sizeof(rune_t)` - so we must size the walker
        // region from rune-width requirements, not the byte-width `ascii_fallback` (which would under-reserve here).
        diagonal_memory_requirements<size_t> rune_requirements(                        //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(rune_t), specs.cache_line_width);
        size_t const utf8_path = transcode_bytes + rune_requirements.total;

        // The pure-ASCII shortcut bypasses transcoding and runs the char fallback over the whole buffer instead.
        size_t const ascii_path = ascii_fallback_t {substituter_, gap_costs_}.scratch_space_needed(first, second,
                                                                                                   specs);
        return sz_max_of_two(utf8_path, ascii_path);
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char const> first, span<char const> second, size_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        // If the cost of gap opening and extension is the same and we've mistakenly instantiated
        // the more memory-intensive `affine_gap_costs_t`, we can fall-back to the linearized version.
        if constexpr (is_same_type<gap_costs_t, affine_gap_costs_t>::value)
            if (gap_costs_.open == gap_costs_.extend) {
                linear_gap_costs_t linear_gap {gap_costs_.open};
                linearized_fallback_t linear_backend(substituter_, linear_gap);
                return linear_backend(first, second, result_ref, scratch_space, executor, specs);
            }

        // Check if the strings are entirely composed of ASCII characters,
        // and default to a simpler algorithm in that case.
        if (sz_isascii(first.data(), first.size()) && sz_isascii(second.data(), second.size()))
            return ascii_fallback_t {substituter_, gap_costs_}(first, second, result_ref, scratch_space, executor,
                                                               specs);

        // Carve the transcode region off the front of scratch, then pass the remainder to walkers.
        size_t const first_unpacking_ceiling = round_up_to_multiple(sizeof(rune_t) * first.size(),
                                                                    specs.cache_line_width);
        size_t const second_unpacking_ceiling = round_up_to_multiple(sizeof(rune_t) * second.size(),
                                                                     specs.cache_line_width);
        size_t const transcode_bytes = first_unpacking_ceiling + second_unpacking_ceiling;
        if (scratch_space.size() < transcode_bytes) return status_t::bad_alloc_k;
        rune_t *const first_data_utf32 = reinterpret_cast<rune_t *>(scratch_space.data());
        rune_t *const second_data_utf32 = reinterpret_cast<rune_t *>(scratch_space.data() + first_unpacking_ceiling);
        scratch_space_t const walker_scratch = scratch_space.subspan(transcode_bytes,
                                                                     scratch_space.size() - transcode_bytes);

        // Export into UTF-32 buffer.
        rune_length_t rune_length;
        size_t first_length_utf32 = 0, second_length_utf32 = 0;
        for (size_t progress_utf8 = 0; progress_utf8 < first.size();
             progress_utf8 += rune_length, ++first_length_utf32) {
            sz_rune_parse_unchecked(first.data() + progress_utf8, first_data_utf32 + first_length_utf32, &rune_length);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }
        for (size_t progress_utf8 = 0; progress_utf8 < second.size();
             progress_utf8 += rune_length, ++second_length_utf32) {
            sz_rune_parse_unchecked(second.data() + progress_utf8, second_data_utf32 + second_length_utf32, &rune_length);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<size_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first_length_utf32, second_length_utf32,                                   //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(rune_t), specs.cache_line_width);

        span<rune_t const> const first_utf32 {first_data_utf32, first_length_utf32};
        span<rune_t const> const second_utf32 {second_data_utf32, second_length_utf32};

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        if (requirements.bytes_per_cell <= 1 && requirements.max_diagonal_length < 16) {
            u8_t result_u8 = std::numeric_limits<u8_t>::max();
            status_t status = horizontal_u8_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u8,
                                                                         walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        else if (requirements.bytes_per_cell <= 1) {
            u8_t result_u8 = std::numeric_limits<u8_t>::max();
            status_t status = diagonal_u8_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u8,
                                                                       walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }
        else if (requirements.bytes_per_cell == 2) {
            u16_t result_u16 = std::numeric_limits<u16_t>::max();
            status_t status = diagonal_u16_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u16,
                                                                        walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u16;
        }
        else if (requirements.bytes_per_cell == 4) {
            u32_t result_u32 = std::numeric_limits<u32_t>::max();
            status_t status = diagonal_u32_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u32,
                                                                        walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u32;
        }
        else if (requirements.bytes_per_cell == 8) {
            u64_t result_u64 = std::numeric_limits<u64_t>::max();
            status_t status = diagonal_u64_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u64,
                                                                        walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score between two strings using the CPU backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <                                            //
    typename char_or_rune_type_ = char,               //
    typename substituter_type_ = error_costs_32x32_t, //
    typename gap_costs_type_ = linear_gap_costs_t,    //
    sz_capability_t capability_ = sz_cap_serial_k,    //
    typename enable_ = void                           //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct needleman_wunsch_score {

    using char_t = char_or_rune_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = serialize_capability(capability_k);

    using horizontal_i16_t =                                         //
        horizontal_walker<char_t, i16_t, substituter_t, gap_costs_t, //
                          sz_maximize_score_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_i16_t =                                         //
        diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, //
                        sz_maximize_score_k, sz_similarity_global_k, capability_serialized_k>;
    using diagonal_i32_t =                                         //
        diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, //
                        sz_maximize_score_k, sz_similarity_global_k, capability_k>;
    using diagonal_i64_t =                                         //
        diagonal_walker<char_t, i64_t, substituter_t, gap_costs_t, //
                        sz_maximize_score_k, sz_similarity_global_k, capability_k>;

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};

    needleman_wunsch_score() noexcept {}
    needleman_wunsch_score(substituter_t subs, gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        // The dispatch must mirror `operator()` so the chosen walker's `layout()` is the single sizing authority.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);
        if (requirements.bytes_per_cell <= 2 && requirements.max_diagonal_length < 16)
            return horizontal_i16_t {substituter_, gap_costs_}.layout(first, second, specs);
        if (requirements.bytes_per_cell <= 2)
            return diagonal_i16_t {substituter_, gap_costs_}.layout(first, second, specs);
        if (requirements.bytes_per_cell == 4)
            return diagonal_i32_t {substituter_, gap_costs_}.layout(first, second, specs);
        return diagonal_i64_t {substituter_, gap_costs_}.layout(first, second, specs);
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2 && requirements.max_diagonal_length < 16) {
            i16_t result_i16 = std::numeric_limits<i16_t>::min();
            status = horizontal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                                 specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        else if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16 = std::numeric_limits<i16_t>::min();
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32 = std::numeric_limits<i32_t>::min();
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64 = std::numeric_limits<i64_t>::min();
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score between two strings using the CPU backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <                                            //
    typename char_or_rune_type_ = char,               //
    typename substituter_type_ = error_costs_32x32_t, //
    typename gap_costs_type_ = linear_gap_costs_t,    //
    sz_capability_t capability_ = sz_cap_serial_k,    //
    typename enable_ = void                           //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct smith_waterman_score {

    using char_t = char_or_rune_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = serialize_capability(capability_k);

    using horizontal_i16_t =                                         //
        horizontal_walker<char_t, i16_t, substituter_t, gap_costs_t, //
                          sz_maximize_score_k, sz_similarity_local_k, capability_serialized_k>;
    using diagonal_i16_t =                                         //
        diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, //
                        sz_maximize_score_k, sz_similarity_local_k, capability_serialized_k>;
    using diagonal_i32_t =                                         //
        diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, //
                        sz_maximize_score_k, sz_similarity_local_k, capability_k>;
    using diagonal_i64_t =                                         //
        diagonal_walker<char_t, i64_t, substituter_t, gap_costs_t, //
                        sz_maximize_score_k, sz_similarity_local_k, capability_k>;

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};

    smith_waterman_score() noexcept {}
    smith_waterman_score(substituter_t subs, gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        // The dispatch must mirror `operator()` so the chosen walker's `layout()` is the single sizing authority.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);
        if (requirements.bytes_per_cell <= 2 && requirements.max_diagonal_length < 16)
            return horizontal_i16_t {substituter_, gap_costs_}.layout(first, second, specs);
        if (requirements.bytes_per_cell <= 2)
            return diagonal_i16_t {substituter_, gap_costs_}.layout(first, second, specs);
        if (requirements.bytes_per_cell == 4)
            return diagonal_i32_t {substituter_, gap_costs_}.layout(first, second, specs);
        return diagonal_i64_t {substituter_, gap_costs_}.layout(first, second, specs);
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        if (first.empty() || second.empty()) {
            result_ref = 0;
            return status_t::success_k;
        }

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        if (requirements.bytes_per_cell <= 2 && requirements.max_diagonal_length < 16) {
            i16_t result_i16 = std::numeric_limits<i16_t>::min();
            status_t status = horizontal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space,
                                                                          executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        else if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16 = std::numeric_limits<i16_t>::min();
            status_t status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32 = std::numeric_limits<i32_t>::min();
            status_t status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64 = std::numeric_limits<i64_t>::min();
            status_t status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_i64;
        }

        return status_t::success_k;
    }
};

#pragma endregion

#pragma region - Parallel Batch Algorithms

/**
 *  @brief Helper method, applying the desired pairwise scoring kernel to all input pairs,
 *      reusing the sae dynamic memory buffer across pairs.
 */
template <                                   //
    typename score_type_,                    //
    typename scoring_engine_type_,           //
    typename first_strings_type_,            //
    typename second_strings_type_,           //
    typename results_type_,                  //
    typename allocator_type_ = dummy_alloc_t //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && indexed_results_like<results_type_>
#endif
status_t score_sequentially_( //
    scoring_engine_type_ &&scoring, first_strings_type_ const &first_strings,
    second_strings_type_ const &second_strings, //
    results_type_ &&results, allocator_type_ &alloc, cpu_specs_t const &specs) noexcept {

    using score_t = score_type_;
    using allocator_t = allocator_type_;

    auto first_size = first_strings.size();
    auto second_size = second_strings.size();
    sz_assert_(first_size == second_size && "Expect equal number of strings");

    // Allocate one shared buffer for all pairs
    size_t max_memory_requirement = 0;
    for (size_t i = 0; i < first_size; ++i) {
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];
        max_memory_requirement = std::max(max_memory_requirement,
                                          scoring.scratch_space_needed(to_view(first), to_view(second), specs));
    }
    safe_vector<std::byte, allocator_t> scratch_buffer(alloc);
    if (status_t status = scratch_buffer.try_resize(max_memory_requirement); status != status_t::success_k)
        return status;

    // Loop through all pairs again, but now reusing the allocated scratch space.
    for (size_t i = 0; i < first_size; ++i) {
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];
        // Pass the dummy by lvalue so the single-pair `operator()` instantiates on the bare executor type.
        dummy_executor_t dummy_executor;
        status_t status = scoring(to_view(first), to_view(second), result, scratch_space_t(scratch_buffer),
                                  dummy_executor, specs);
        if (status == status_t::success_k) results[i] = result;
        else return status;
    }
    return status_t::success_k;
}

/**
 *  @brief Helper method, applying the desired pairwise scoring kernel to all input pairs,
 *      differentiating multi-threaded and single-threaded cases.
 *      For pairs of very large strings, all cores cooperate to compute one distance maximizing
 *      cache hits. For smaller strings, each core computes its own distance.
 */
template <                                     //
    typename score_type_,                      //
    typename scoring_engine_type_,             //
    typename first_strings_type_,              //
    typename second_strings_type_,             //
    typename results_type_,                    //
    typename allocator_type_ = dummy_alloc_t,  //
    typename executor_type_ = dummy_executor_t //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
status_t score_in_parallel_(                           //
    scoring_engine_type_ &&scoring,                    //
    first_strings_type_ const &first_strings,          //
    second_strings_type_ const &second_strings,        //
    results_type_ &&results,                           //
    allocator_type_ &alloc, executor_type_ &&executor, //
    cpu_specs_t const &specs) noexcept {

    using score_t = score_type_;
    using allocator_t = allocator_type_;
    using executor_t = typename std::decay<executor_type_>::type;
    using prong_t = typename executor_t::prong_t;

    auto first_size = first_strings.size();
    auto second_size = second_strings.size();
    sz_assert_(first_size == second_size && "Expect equal number of strings");

    // Estimate our memory requirements for both single-threaded processing of small pairs on
    // all CPU cores and multi-threaded processing of larger pairs.
    size_t max_memory_per_small = 0, max_memory_for_large = 0;
    auto const is_small = [&specs](size_t first_length, size_t second_length) noexcept {
        return std::min(first_length, second_length) <= specs.l1_bytes;
    };
    for (size_t i = 0; i < first_size; ++i) {
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];
        size_t const needed = scoring.scratch_space_needed(to_view(first), to_view(second), specs);
        if (is_small(first.size(), second.size())) max_memory_per_small = std::max(max_memory_per_small, needed);
        else max_memory_for_large = std::max(max_memory_for_large, needed);
    }
    size_t const threads_count = executor.threads_count();
    size_t const max_memory_requirement = std::max(max_memory_per_small * threads_count, max_memory_for_large);
    safe_vector<std::byte, allocator_t> scratch_buffer(alloc);
    if (status_t status = scratch_buffer.try_resize(max_memory_requirement); status != status_t::success_k)
        return status;

    // Use an atomic to store any error encountered.
    std::atomic<status_t> error {status_t::success_k};

    // There may be a huge variance in the lengths of the strings,
    // so we need to use a dynamic schedule.
    executor.for_n_dynamic(first_size, [&](prong_t prong) noexcept {
        if (error.load() != status_t::success_k) return;
        size_t const i = prong.task;
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];

        if (!is_small(first.size(), second.size())) return;
        scratch_space_t worker_scratch = scratch_space_t(scratch_buffer).part_i_of_n(prong.thread, threads_count);
        dummy_executor_t dummy_executor; // Lvalue, so the scorer pins the bare executor type.
        status_t status = scoring(to_view(first), to_view(second), result, worker_scratch, dummy_executor, specs);
        if (status == status_t::success_k) results[i] = result;
        else error.store(status);
    });

    // Now handle the longer strings - all cores cooperate on a single pair, so only this thread allocates.
    for (size_t i = 0; i < first_size && error.load() == status_t::success_k; ++i) {
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];

        if (is_small(first.size(), second.size())) continue;
        status_t status = scoring(to_view(first), to_view(second), result, scratch_space_t(scratch_buffer), executor,
                                  specs);
        if (status == status_t::success_k) results[i] = result;
        else error.store(status);
    }
    return error.load();
}

template <                       //
    typename gap_costs_type_,    //
    typename allocator_type_,    //
    sz_capability_t capability_, //
    typename enable_             //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distances {

    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = levenshtein_distance<char, gap_costs_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    levenshtein_distances(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances(uniform_substitution_costs_t subs, gap_costs_t gaps,
                          allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, cpu_specs_t const &specs = {}) noexcept {
        return score_sequentially_<size_t>(       //
            scoring_t {substituter_, gap_costs_}, //
            first_strings, second_strings, std::forward<results_type_>(results), alloc_, specs);
    }

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_,
              typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_in_parallel_<size_t>(                                       //
            scoring_t {substituter_, gap_costs_},                                //
            first_strings, second_strings, std::forward<results_type_>(results), //
            alloc_, executor, specs);
    }
};

template <                       //
    typename gap_costs_type_,    //
    typename allocator_type_,    //
    sz_capability_t capability_, //
    typename enable_             //
    >
#if SZ_HAS_CONCEPTS_
    requires gap_costs_like<gap_costs_type_>
#endif
struct levenshtein_distances_utf8 {

    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = levenshtein_distance_utf8<gap_costs_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    levenshtein_distances_utf8(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances_utf8(uniform_substitution_costs_t subs, gap_costs_t gaps,
                               allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, cpu_specs_t const &specs = {}) noexcept {
        return score_sequentially_<size_t>(       //
            scoring_t {substituter_, gap_costs_}, //
            first_strings, second_strings, std::forward<results_type_>(results), alloc_, specs);
    }

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_,
              typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_in_parallel_<size_t>(                                       //
            scoring_t {substituter_, gap_costs_},                                //
            first_strings, second_strings, std::forward<results_type_>(results), //
            alloc_, executor, specs);
    }
};

template <                       //
    typename substituter_type_,  //
    typename gap_costs_type_,    //
    typename allocator_type_,    //
    sz_capability_t capability_, //
    typename enable_             //
    >
#if SZ_HAS_CONCEPTS_
    requires substituter_like<substituter_type_> && gap_costs_like<gap_costs_type_>
#endif
struct needleman_wunsch_scores {

    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, capability_k>;

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    needleman_wunsch_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_scores(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, cpu_specs_t const &specs = {}) noexcept {
        return score_sequentially_<ssize_t>(      //
            scoring_t {substituter_, gap_costs_}, //
            first_strings, second_strings, std::forward<results_type_>(results), alloc_, specs);
    }

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_,
              typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_in_parallel_<ssize_t>(                                      //
            scoring_t {substituter_, gap_costs_},                                //
            first_strings, second_strings, std::forward<results_type_>(results), //
            alloc_, executor, specs);
    }
};

template <                       //
    typename substituter_type_,  //
    typename gap_costs_type_,    //
    typename allocator_type_,    //
    sz_capability_t capability_, //
    typename enable_             //
    >
#if SZ_HAS_CONCEPTS_
    requires substituter_like<substituter_type_> && gap_costs_like<gap_costs_type_>
#endif
struct smith_waterman_scores {

    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, capability_k>;

    substituter_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    smith_waterman_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    smith_waterman_scores(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, cpu_specs_t const &specs = {}) noexcept {
        return score_sequentially_<ssize_t>(      //
            scoring_t {substituter_, gap_costs_}, //
            first_strings, second_strings, std::forward<results_type_>(results), alloc_, specs);
    }

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_,
              typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_> && indexed_results_like<results_type_>
#endif
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_in_parallel_<ssize_t>(                                      //
            scoring_t {substituter_, gap_costs_},                                //
            first_strings, second_strings, std::forward<results_type_>(results), //
            alloc_, executor, specs);
    }
};

#pragma endregion

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_SERIAL_HPP_
