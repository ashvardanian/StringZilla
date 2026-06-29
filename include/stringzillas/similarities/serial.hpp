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

#include "stringzilla/types.hpp"           // `sz::error_cost_t`
#include "stringzilla/memory.h"            // `sz_move_serial`
#include "stringzilla/utf8_runes/serial.h" // `sz_rune_decode_unchecked`
#include "stringzillas/types.hpp"          // `sz::executor_like`

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

#pragma region Algorithm Building Blocks

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
        // The DP itself only ever indexes a band in `[0, max_diagonal_length)`, so the bands fit in exactly
        // `diagonals_count * bytes_per_diagonal`. The hardware, however, services a read in whole transaction words
        // of `register_width` bytes: a cell narrower than that word (`u8`/`u16`) is fetched by reading the word that
        // encloses it, so reading the last cell of the final band reaches up to `register_width - bytes_per_cell`
        // bytes beyond it. Between bands that overhang lands in the next band; past the final band it would read
        // off the end of the (tightly packed, per-warp) GPU allocation. Reserve one transaction word so the widened
        // read of the last cell is always backed by allocated memory.
        size_t const widened_read_overhang = register_width > bytes_per_cell ? register_width - bytes_per_cell : 0;
        this->bytes_for_diagonals = diagonals_count * bytes_per_diagonal + widened_read_overhang;
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

/**
 *  @brief Routes a runtime word-count @p bucket in `[current_k, high_k]` to the matching @b compile-time @p fixed
 *         callback (invoked with `std::integral_constant<size_t, bucket>` so it can pick the right
 *         `distances_*_multiword_<bucket>` instantiation); a bucket past @p high_k invokes @p overflow. One
 *         compile-time-unrolled dispatch replacing the per word count `switch` ladders in the SIMD Myers cross drivers.
 */
template <size_t current_k, size_t high_k, typename fixed_type_, typename overflow_type_>
status_t dispatch_word_bucket_(size_t bucket, fixed_type_ &&fixed, overflow_type_ &&overflow) noexcept {
    if constexpr (current_k > high_k) return sz_unused_(bucket), overflow();
    else if (bucket == current_k) return fixed(std::integral_constant<size_t, current_k> {});
    else return dispatch_word_bucket_<current_k + 1, high_k>(bucket, fixed, overflow);
}

#pragma region Core Templates

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
 *  @brief @b Inter-sequence walker: scores a block of candidates against @b one shared query, one candidate per
 *      SIMD lane, advancing the Dynamic Programming matrix @b row-by-row.
 *
 *  This is the cross-product workhorse. Unlike `diagonal_walker` (which vectorizes the anti-diagonal of a single
 *  pair and therefore starves its lanes for short strings), here every lane carries an independent candidate, so
 *  there is no intra-pair left-dependency to break — a plain row walk keeps all lanes busy regardless of length.
 *  The `sz_cap_serial_k` specialization below is the scalar per-lane @b reference oracle that every SIMD/GPU
 *  candidate-lane kernel is validated against.
 *
 *  @tparam candidate_lanes_ Number of candidates packed side-by-side (64 for 8-bit cells, 32 for 16-bit).
 *  @sa `candidate_lanes_block`, `sz_packing_candidates_across_lanes_k`.
 */
template <                                                         //
    typename char_or_rune_type_ = char,                            //
    typename score_type_ = size_t,                                 //
    typename substituter_type_ = uniform_substitution_costs_t,     //
    typename gap_costs_type_ = linear_gap_costs_t,                 //
    sz_similarity_objective_t objective_ = sz_minimize_distance_k, //
    sz_similarity_locality_t locality_ = sz_similarity_global_k,   //
    sz_capability_t capability_ = sz_cap_serial_k,                 //
    size_t candidate_lanes_ = 64,                                  //
    typename enable_ = void                                        //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && substituter_like<substituter_type_> && gap_costs_like<gap_costs_type_>
#endif
struct candidate_lane_walker;

/**
 *  @brief Serial reference: scalar per-lane row Dynamic Programming. Differential oracle for the SIMD/GPU
 *      candidate-lane kernels. Covers @b global alignment with @b linear gaps (Levenshtein when the substituter
 *      is uniform, Needleman-Wunsch when it is a class-cost matrix); local alignment is a separate specialization.
 */
template <typename char_or_rune_type_, typename score_type_, typename substituter_type_,
          sz_similarity_objective_t objective_, size_t candidate_lanes_>
struct candidate_lane_walker<char_or_rune_type_, score_type_, substituter_type_, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_serial_k, candidate_lanes_, void> {

    using char_t = char_or_rune_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    static constexpr size_t candidate_lanes_k = candidate_lanes_;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` cells each. */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = sizeof(score_t) * (longest_candidate + 1);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to `candidate_lanes_` candidates (see `candidate_lanes_block`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        error_cost_t const gap = gap_costs_.open_or_extend;
        size_t const query_length = query.size();
        size_t const row_cells = candidates.longest_candidate + 1;

        // Two rows carved from the scratch byte span (the allocation is over-aligned for `score_t`).
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_cells;

        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
            size_t const candidate_length = candidates.lengths[lane_index];

            // Row 0: the empty query prefix against every candidate prefix is a run of gaps.
            for (size_t column = 0; column <= candidate_length; ++column)
                previous_row[column] = static_cast<score_t>(gap * column);

            for (size_t query_position = 1; query_position <= query_length; ++query_position) {
                current_row[0] = static_cast<score_t>(gap * query_position);
                char_t const query_char = query[query_position - 1];
                for (size_t column = 1; column <= candidate_length; ++column) {
                    char_t const candidate_char = candidates.character_of_lane(lane_index, column - 1);
                    error_cost_t const cost_of_substitution = substituter_(query_char, candidate_char);
                    score_t const if_substitution = previous_row[column - 1] + cost_of_substitution;
                    score_t const if_gap = min_or_max<objective_k>(previous_row[column], current_row[column - 1]) + gap;
                    current_row[column] = min_or_max<objective_k>(if_substitution, if_gap);
                }
                trivial_swap(previous_row, current_row);
            }

            result_lanes[lane_index] = previous_row[candidate_length];
        }
        return status_t::success_k;
    }
};

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

#pragma endregion Core Templates

#pragma region Common Aliases

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

#pragma endregion Common Aliases

#pragma region Autovectorized Tile Scorer

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

#pragma endregion Autovectorized Tile Scorer

#pragma region Diagonal Walker

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

#pragma endregion Diagonal Walker
#pragma region Horizontal Walker

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

#pragma endregion Horizontal Walker

#pragma endregion Algorithm Building Blocks

#pragma region Pairwise Algorithms on CPU

template <>
struct levenshtein_distance_myers<char, sz_cap_serial_k> {

    using char_t = char;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    levenshtein_distance_myers() noexcept {}

    /**
     *  @brief Largest @b shorter-side length whose per-word `vertical_positives` / `vertical_negatives` state fits
     *      the on-stack arrays. Above this the generic path parks that state in scratch instead. Eight 64-bit words
     *      mirrors the widest unrolled tier (512 runes) and the Ice Lake lockstep family.
     */
    static constexpr size_t stack_words_capacity_k = 8;

    /** @brief Number of 64-bit words spanning a @p shorter_length-rune pattern, i.e. `ceil(shorter_length / 64)`. */
    static constexpr size_t words_count_for(size_t shorter_length) noexcept { return (shorter_length + 63) / 64; }

    /**
     *  @brief Number of 64-bit words the dispatch in `operator()` actually allocates for a @p shorter_length-rune
     *      pattern. The unrolled tiers round up to the next power of two (1/2/4/8 words for shorter <= 64/128/256/512)
     *      so their templates line up with the Ice Lake lockstep family; the generic tier above 512 uses the exact
     *      `ceil(shorter / 64)`. `layout()` must size the `match_masks` table to this so the scratch never falls short.
     */
    static constexpr size_t dispatch_words_count_for(size_t shorter_length) noexcept {
        return shorter_length <= 64    ? 1
               : shorter_length <= 128 ? 2
               : shorter_length <= 256 ? 4
               : shorter_length <= 512 ? 8
                                       : words_count_for(shorter_length);
    }

    /** @brief Byte offsets of this walker's scratch sub-buffers. */
    struct layout_t {
        /** @brief The per-word `match_masks` tables (256 entries each). */
        size_t match_masks = 0;
        /** @brief The generic path's per-word `vertical_positives` state (zero for the on-stack tiers). */
        size_t vertical_positives = 0;
        /** @brief The generic path's per-word `vertical_negatives` state (zero for the on-stack tiers). */
        size_t vertical_negatives = 0;
        /** @brief Bytes this walker touches; doubles as its scratch-size estimate. */
        size_t total = 0;
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        // The small tiers pick 1/2/4/8 words (shorter <= 64/128/256/512); above that the generic path spans
        // `ceil(shorter / 64)` words. Either way the `match_masks` table needs one 256-entry row per word.
        size_t const words_count = dispatch_words_count_for(shorter_length);
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.match_masks = amount, amount += sizeof(u64_t) * words_count * 256;
        // Only the generic path (above the on-stack capacity) parks its vertical state in scratch.
        if (words_count > stack_words_capacity_k) {
            at.vertical_positives = amount, amount += sizeof(u64_t) * words_count;
            at.vertical_negatives = amount, amount += sizeof(u64_t) * words_count;
        }
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
    status_t unrolled_(span<char const> shorter, span<char const> longer, size_t &result_ref,
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
        // Every block of every touched character is read, so zero them all before building the `match_masks` table: the
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

    /**
     *  @brief Bit-parallel Myers/Hyyrö unit-cost Levenshtein for one pair of @b any @p shorter size, carrying the
     *      horizontal deltas across a run-time `words_count = ceil(shorter / 64)` of 64-bit blocks. Exact edit
     *      distance in O(longer * words_count) word-operations, 64 DP cells per machine word, no DP matrix.
     *
     *  Uses the same recurrence as `unrolled_` but with a run-time block count. The `match_masks` `match_masks` table lives
     *  in the supplied scratch (`256 * words_count` u64 entries - the dominant memory cost, ~2KiB per block). The
     *  per-block `vertical_positives` / `vertical_negatives` state stays on the stack up to `stack_words_capacity_k`
     *  blocks and otherwise spills into scratch, so the hot loop never touches the heap.
     */
    status_t generic_(span<char const> shorter, span<char const> longer, size_t &result_ref,
                      scratch_space_t scratch_space) const noexcept {
        size_t const shorter_length = shorter.size();
        size_t const longer_length = longer.size();
        // Empty pattern: the distance is the text length, and the top-bit read below would underflow.
        if (shorter_length == 0) {
            result_ref = longer_length;
            return status_t::success_k;
        }
        size_t const words_count = words_count_for(shorter_length);
        size_t const match_masks_bytes = sizeof(u64_t) * words_count * 256;
        size_t const vertical_bytes = words_count > stack_words_capacity_k ? sizeof(u64_t) * words_count : 0;
        if (scratch_space.size() < match_masks_bytes + 2 * vertical_bytes) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data());
        // The vertical state lives on the stack for the common case and in scratch for very long patterns, so the
        // inner loop is allocation-free either way. The `match_masks` table starts at the (cache-line-aligned) front; the
        // vertical state is parked at the very end of scratch, sidestepping `layout()`'s internal alignment padding
        // between sub-buffers so the offsets here cannot disagree with the sizing in `scratch_space_needed`.
        u64_t stack_vertical_positives[stack_words_capacity_k], stack_vertical_negatives[stack_words_capacity_k];
        u64_t *vertical_positives = stack_vertical_positives;
        u64_t *vertical_negatives = stack_vertical_negatives;
        if (words_count > stack_words_capacity_k) {
            std::byte *const scratch_end = scratch_space.data() + scratch_space.size();
            vertical_negatives = reinterpret_cast<u64_t *>(scratch_end - vertical_bytes);
            vertical_positives = reinterpret_cast<u64_t *>(scratch_end - 2 * vertical_bytes);
        }

        // Build the `match_masks` table: clear every touched character's blocks, then set the shorter side's match bits. The
        // scratch is uninitialized (and may hold a previous walker's bytes), so each row is `match_masks[char][word]`
        // laid out as `match_masks[char * words_count + word]`.
        for (size_t position = 0; position != shorter_length; ++position)
            for (size_t word = 0; word != words_count; ++word)
                match_masks[(size_t)(u8_t)shorter[position] * words_count + word] = 0;
        for (size_t position = 0; position != longer_length; ++position)
            for (size_t word = 0; word != words_count; ++word)
                match_masks[(size_t)(u8_t)longer[position] * words_count + word] = 0;
        for (size_t position = 0; position != shorter_length; ++position)
            match_masks[(size_t)(u8_t)shorter[position] * words_count + (position >> 6)] |= (u64_t)1 << (position & 63);

        for (size_t word = 0; word != words_count; ++word)
            vertical_positives[word] = ~(u64_t)0, vertical_negatives[word] = 0;
        size_t const last_word = (shorter_length - 1) >> 6, last_bit = (shorter_length - 1) & 63;
        size_t distance = shorter_length;
        for (size_t longer_position = 0; longer_position != longer_length; ++longer_position) {
            u8_t const symbol = (u8_t)longer[longer_position];
            u64_t const *const match_row = &match_masks[(size_t)symbol * words_count];
            u64_t horizontal_positive_carry = 1, horizontal_negative_carry = 0; // Top-row boundary into block 0 is +1.
            for (size_t word = 0; word != words_count; ++word) {
                u64_t const pattern_matches = match_row[word];
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

        for (size_t position = 0; position != shorter_length; ++position)
            match_masks[(size_t)(u8_t)shorter[position] * words_count + (position >> 6)] = 0;
        result_ref = distance;
        return status_t::success_k;
    }

    status_t operator()(span<char const> first, span<char const> second, size_t &result_ref,
                        scratch_space_t scratch_space) noexcept {
        bool const first_is_shorter = first.size() <= second.size();
        span<char const> shorter = first_is_shorter ? first : second;
        span<char const> longer = first_is_shorter ? second : first;
        size_t const shorter_length = shorter.size();
        if (shorter_length <= 64) return unrolled_<1>(shorter, longer, result_ref, scratch_space);
        if (shorter_length <= 128) return unrolled_<2>(shorter, longer, result_ref, scratch_space);
        if (shorter_length <= 256) return unrolled_<4>(shorter, longer, result_ref, scratch_space);
        if (shorter_length <= 512) return unrolled_<8>(shorter, longer, result_ref, scratch_space);
        return generic_(shorter, longer, result_ref, scratch_space); // any longer shorter side
    }
};

/**
 *  @brief Bit-parallel Myers/Hyyrö unit-cost Levenshtein for two @b UTF-32 (rune) strings on the serial backend.
 *
 *  The Myers scan (`vertical_positives` / `vertical_negatives`, the horizontal `+1`/`-1` carries, the multi-word
 *  low->high ripple and the distance probe) operates @b only on the per-symbol `match_masks` bitmask words and is therefore
 *  independent of the key type. The byte specialization (`levenshtein_distance_myers<char, sz_cap_serial_k>`, the
 *  byte oracle) indexes a dense 256-entry `match_masks` table by the symbol byte; a 4-byte `rune_t` key cannot index a dense
 *  table, so this specialization swaps the dense table for an @b open-addressing hash (rune -> `words_count` bitmask
 *  words). The recurrence below is the same one the byte oracle runs, only the per-symbol `match_row` lookup changes:
 *  a text rune absent from the pattern hashes to an all-zero row, which is exactly correct (it matches nothing).
 */
template <>
struct levenshtein_distance_myers<rune_t, sz_cap_serial_k> {

    using char_t = rune_t;
    using index_t = u32_t;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    levenshtein_distance_myers() noexcept {}

    /**
     *  @brief Largest @b shorter-side length whose per-word `vertical_positives` / `vertical_negatives` state fits the
     *      on-stack arrays. Above this the state spills into scratch, mirroring the byte walker's policy.
     */
    static constexpr size_t stack_words_capacity_k = 8;

    /** @brief Number of 64-bit words spanning a @p shorter_length-rune pattern, i.e. `ceil(shorter_length / 64)`. */
    static constexpr size_t words_count_for(size_t shorter_length) noexcept {
        return divide_round_up<size_t>(shorter_length, 64);
    }

    /**
     *  @brief Open-addressing capacity (a power of two) for a pattern of @p distinct_upper_bound distinct runes. The
     *      pattern has at most `shorter_length` distinct runes, so passing `shorter_length` is a safe upper bound. The
     *      `2 *` keeps the load factor <= 0.5 for cheap linear probing; `sz_size_bit_ceil` rounds to a power of two so
     *      the multiply-shift hash maps cleanly onto the slot range. Always at least one slot.
     */
    static index_t hash_capacity_for(size_t distinct_upper_bound) noexcept {
        size_t const slots_wanted = sz_max_of_two(2 * distinct_upper_bound, (size_t)1);
        return static_cast<index_t>(sz_size_bit_ceil(slots_wanted));
    }

    /** @brief Sentinel rune marking an empty hash slot (`rune_t` never reaches `0xFFFFFFFF`; valid <= 0x10FFFF). */
    static constexpr rune_t empty_slot_k = static_cast<rune_t>(0xFFFFFFFFu);

    /**
     *  @brief Multiply-shift hash of a rune into `[0, capacity)`; @p capacity must be a power of two. The 64-bit
     *      Fibonacci multiply spreads the rune's bits into the high word, then a single mask selects the slot - no
     *      `>> 64` undefined shift at the `capacity == 1` boundary.
     */
    static index_t hash_rune(rune_t rune, index_t capacity) noexcept {
        u64_t const mixed = static_cast<u64_t>(static_cast<u32_t>(rune)) * 0x9E3779B97F4A7C15ull;
        return static_cast<index_t>((mixed >> 32) & static_cast<u64_t>(capacity - 1));
    }

    /** @brief Byte offsets of this walker's scratch sub-buffers. */
    struct layout_t {
        /** @brief The open-addressing slot keys (`hash_capacity` runes). */
        size_t slot_keys = 0;
        /** @brief The per-slot `match_masks` bitmask words (`hash_capacity * words_count` u64 entries). */
        size_t slot_masks = 0;
        /** @brief A permanently-zero bitmask row (`words_count` u64) returned for runes absent from the pattern. */
        size_t absent_row = 0;
        /** @brief The per-word `vertical_positives` state when it spills past the on-stack capacity. */
        size_t vertical_positives = 0;
        /** @brief The per-word `vertical_negatives` state when it spills past the on-stack capacity. */
        size_t vertical_negatives = 0;
        /** @brief Bytes this walker touches; doubles as its scratch-size estimate. */
        size_t total = 0;
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        size_t const words_count = words_count_for(shorter_length);
        index_t const capacity = hash_capacity_for(shorter_length);
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.slot_keys = amount, amount += sizeof(rune_t) * capacity;
        at.slot_masks = amount, amount += sizeof(u64_t) * static_cast<size_t>(capacity) * words_count;
        at.absent_row = amount, amount += sizeof(u64_t) * words_count;
        if (words_count > stack_words_capacity_k) {
            at.vertical_positives = amount, amount += sizeof(u64_t) * words_count;
            at.vertical_negatives = amount, amount += sizeof(u64_t) * words_count;
        }
        at.total = amount;
        return at;
    }

    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        scratch_space_t scratch_space) const noexcept {
        bool const first_is_shorter = first.size() <= second.size();
        span<char_t const> shorter = first_is_shorter ? first : second;
        span<char_t const> longer = first_is_shorter ? second : first;
        size_t const shorter_length = shorter.size();
        size_t const longer_length = longer.size();

        // Empty pattern: the distance is the text length, and the top-bit read below would underflow.
        if (shorter_length == 0) {
            result_ref = longer_length;
            return status_t::success_k;
        }

        size_t const words_count = words_count_for(shorter_length);
        index_t const capacity = hash_capacity_for(shorter_length);
        // Carve the hash from the (caller-aligned) front of scratch and park the optional vertical-state spill at the
        // very end - exactly the byte `generic_` policy. This sidesteps `layout()`'s inter-buffer cache-line padding,
        // so the offsets here can never disagree with the (padded, hence larger) size from `scratch_space_needed`.
        size_t const slot_keys_bytes = sizeof(rune_t) * capacity;
        size_t const slot_masks_bytes = sizeof(u64_t) * static_cast<size_t>(capacity) * words_count;
        size_t const absent_row_bytes = sizeof(u64_t) * words_count;
        size_t const vertical_bytes = words_count > stack_words_capacity_k ? sizeof(u64_t) * words_count : 0;
        if (scratch_space.size() < slot_keys_bytes + slot_masks_bytes + absent_row_bytes + 2 * vertical_bytes)
            return status_t::bad_alloc_k;

        rune_t *const slot_keys = reinterpret_cast<rune_t *>(scratch_space.data());
        u64_t *const slot_masks = reinterpret_cast<u64_t *>(scratch_space.data() + slot_keys_bytes);
        u64_t *const absent_row = reinterpret_cast<u64_t *>(scratch_space.data() + slot_keys_bytes + slot_masks_bytes);

        // The vertical state lives on the stack for the common case and in scratch for very long patterns, so the
        // inner loop is allocation-free either way.
        u64_t stack_vertical_positives[stack_words_capacity_k], stack_vertical_negatives[stack_words_capacity_k];
        u64_t *vertical_positives = stack_vertical_positives;
        u64_t *vertical_negatives = stack_vertical_negatives;
        if (words_count > stack_words_capacity_k) {
            std::byte *const scratch_end = scratch_space.data() + scratch_space.size();
            vertical_negatives = reinterpret_cast<u64_t *>(scratch_end - vertical_bytes);
            vertical_positives = reinterpret_cast<u64_t *>(scratch_end - 2 * vertical_bytes);
        }

        // Build the open-addressing `match_masks` hash: clear keys to the empty sentinel and the absent row to zero, then
        // insert each pattern rune and set its match bit in the slot's `words_count` bitmask words.
        for (index_t slot = 0; slot != capacity; ++slot) slot_keys[slot] = empty_slot_k;
        for (size_t word = 0; word != words_count; ++word) absent_row[word] = 0;
        for (size_t position = 0; position != shorter_length; ++position) {
            rune_t const rune = shorter[position];
            index_t slot = hash_rune(rune, capacity);
            for (;; slot = (slot + 1) & (capacity - 1)) {
                if (slot_keys[slot] == rune) break;
                if (slot_keys[slot] == empty_slot_k) {
                    slot_keys[slot] = rune;
                    for (size_t word = 0; word != words_count; ++word)
                        slot_masks[static_cast<size_t>(slot) * words_count + word] = 0;
                    break;
                }
            }
            slot_masks[static_cast<size_t>(slot) * words_count + (position >> 6)] |= (u64_t)1 << (position & 63);
        }

        for (size_t word = 0; word != words_count; ++word)
            vertical_positives[word] = ~(u64_t)0, vertical_negatives[word] = 0;
        size_t const last_word = (shorter_length - 1) >> 6, last_bit = (shorter_length - 1) & 63;
        size_t distance = shorter_length;
        for (size_t longer_position = 0; longer_position != longer_length; ++longer_position) {
            rune_t const symbol = longer[longer_position];
            // Look up the rune's bitmask row: probe to its slot, or the permanently-zero `absent_row` if the rune is
            // not in the pattern (a text rune absent from the pattern matches nothing -> all-zero `match_masks`, correct).
            u64_t const *match_row = absent_row;
            for (index_t slot = hash_rune(symbol, capacity);; slot = (slot + 1) & (capacity - 1)) {
                rune_t const key = slot_keys[slot];
                if (key == symbol) {
                    match_row = &slot_masks[static_cast<size_t>(slot) * words_count];
                    break;
                }
                if (key == empty_slot_k) break;
            }
            u64_t horizontal_positive_carry = 1, horizontal_negative_carry = 0; // Top-row boundary into block 0 is +1.
            for (size_t word = 0; word != words_count; ++word) {
                u64_t const pattern_matches = match_row[word];
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

        result_ref = distance;
        return status_t::success_k;
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
    /**
     *  @brief Whether the resolved `myers_t` covers @b any shorter-side length. Only the scalar serial walker has
     *      the generic multi-word path; the SIMD lockstep families top out at their widest 512-rune tier, so for
     *      them the Myers fast path stays bounded and longer shorter sides fall through to the anti-diagonal DP.
     */
    static constexpr bool myers_handles_any_length_k = capability_serialized_k == sz_cap_serial_k;
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

        // Bit-parallel Myers path (matches the guard in `operator()`). The scalar serial walker covers any
        // shorter-side length - its generic tier above 512 runes still beats the anti-diagonal DP - while the
        // SIMD lockstep families stay bounded to their widest 512-rune tier. Its `layout()` sizes both the `match_masks`
        // table and, for very long patterns, the spilled vertical state.
        if constexpr (is_same_type<gap_costs_t, linear_gap_costs_t>::value && sizeof(char_t) == 1)
            if (substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1 &&
                (myers_handles_any_length_k || (std::min)(first.size(), second.size()) <= 512)) {
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

        // Bit-parallel Myers fast path (~5x DP on short unit-cost pairs, and still ~25x the anti-diagonal DP on the
        // scalar generic tier above 512 runes). The serial walker covers any shorter-side length; the SIMD lockstep
        // families stay bounded to 512 and fall through to the anti-diagonal DP below for longer shorter sides.
        if constexpr (is_same_type<gap_costs_t, linear_gap_costs_t>::value && sizeof(char_t) == 1)
            if (substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1 &&
                (myers_handles_any_length_k || (std::min)(first.size(), second.size()) <= 512))
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

    /** @brief Bit-parallel rune Myers fast path for unit-cost linear UTF-8 Levenshtein (rune-keyed `match_masks`, R8). */
    using rune_myers_t = levenshtein_distance_myers<rune_t, sz_cap_serial_k>;
    /**
     *  @brief Whether the rune Myers fast path is reachable for this backend. Only the serial rune Myers exists
     *      (its `match_masks` is an open-addressing hash, ISA-independent); SIMD UTF-8 backends keep the rune diagonal walker
     *      until they grow their own rune-keyed Myers, so they stay on the diagonal path below.
     */
    static constexpr bool rune_myers_available_k = capability_serialized_k == sz_cap_serial_k;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};

    levenshtein_distance_utf8() noexcept {}
    levenshtein_distance_utf8(uniform_substitution_costs_t subs, gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Cache-line-padded byte offsets of the two transcoded UTF-32 (`rune_t`) buffers carved off the front of
     *         scratch: `first` at 0, `second` at @ref first_ceiling, the whole carve being @ref total. One source of
     *         truth shared by `scratch_space_needed` (sizing) and `operator()` (the actual carve) so they cannot drift. */
    struct transcode_layout_t {
        size_t first_ceiling, second_ceiling, total;
    };
    SZ_INLINE transcode_layout_t transcode_layout_(span<char const> first, span<char const> second,
                                                   cpu_specs_t const &specs) const noexcept {
        size_t const first_ceiling = round_up_to_multiple(sizeof(rune_t) * first.size(), specs.cache_line_width);
        size_t const second_ceiling = round_up_to_multiple(sizeof(rune_t) * second.size(), specs.cache_line_width);
        return {first_ceiling, second_ceiling, first_ceiling + second_ceiling};
    }

    size_t scratch_space_needed(span<char const> first, span<char const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const transcode_bytes = transcode_layout_(first, second, specs).total;

        // The UTF-8 path transcodes both strings into the front of scratch, then runs a @b rune diagonal walker on
        // the remainder. The walker sees at most `first.size()`/`second.size()` runes (one rune per byte in the
        // worst case), and its reversed-rune buffer costs `runes * sizeof(rune_t)` - so we must size the walker
        // region from rune-width requirements, not the byte-width `ascii_fallback` (which would under-reserve here).
        diagonal_memory_requirements<size_t> rune_requirements(                        //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(rune_t), specs.cache_line_width);
        size_t utf8_path = transcode_bytes + rune_requirements.total;

        // The unit-cost-linear UTF-8 path scores the transcoded runes with the rune Myers walker instead of the rune
        // diagonal. Its `match_masks` hash + vertical state can outsize the diagonal's two rune rows, so widen the reserve.
        // The Myers `layout()` reads only the rune-count via `.size()`; the worst case is one rune per UTF-8 byte.
        if constexpr (rune_myers_available_k && is_same_type<gap_costs_t, linear_gap_costs_t>::value)
            if (substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1) {
                span<rune_t const> const first_runes_upper_bound {nullptr, first.size()};
                span<rune_t const> const second_runes_upper_bound {nullptr, second.size()};
                size_t const myers_path =
                    transcode_bytes +
                    rune_myers_t {}.layout(first_runes_upper_bound, second_runes_upper_bound, specs).total;
                utf8_path = sz_max_of_two(utf8_path, myers_path);
            }

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
        transcode_layout_t const layout = transcode_layout_(first, second, specs);
        size_t const transcode_bytes = layout.total;
        if (scratch_space.size() < transcode_bytes) return status_t::bad_alloc_k;
        rune_t *const first_data_utf32 = reinterpret_cast<rune_t *>(scratch_space.data());
        rune_t *const second_data_utf32 = reinterpret_cast<rune_t *>(scratch_space.data() + layout.first_ceiling);
        scratch_space_t const walker_scratch = scratch_space.subspan(transcode_bytes,
                                                                     scratch_space.size() - transcode_bytes);

        // Export into UTF-32 buffer.
        rune_length_t rune_length;
        size_t first_length_utf32 = 0, second_length_utf32 = 0;
        for (size_t progress_utf8 = 0; progress_utf8 < first.size();
             progress_utf8 += rune_length, ++first_length_utf32) {
            rune_length = sz_rune_decode_unchecked(first.data() + progress_utf8, first_data_utf32 + first_length_utf32);
            if (rune_length == sz_rune_invalid_k) return status_t::invalid_utf8_k;
        }
        for (size_t progress_utf8 = 0; progress_utf8 < second.size();
             progress_utf8 += rune_length, ++second_length_utf32) {
            rune_length = sz_rune_decode_unchecked(second.data() + progress_utf8,
                                                   second_data_utf32 + second_length_utf32);
            if (rune_length == sz_rune_invalid_k) return status_t::invalid_utf8_k;
        }

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<size_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first_length_utf32, second_length_utf32,                                   //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(rune_t), specs.cache_line_width);

        span<rune_t const> const first_utf32 {first_data_utf32, first_length_utf32};
        span<rune_t const> const second_utf32 {second_data_utf32, second_length_utf32};

        // Bit-parallel rune Myers fast path: ~20-30x the rune anti-diagonal DP at L >= 256 (the rune-keyed `match_masks`
        // hash lookup does not erode Myers' advantage). Unit-cost linear only (match 0, mismatch 1, gap 1); the rune
        // diagonal walker below remains the oracle for non-unit / affine costs. Bit-exact with that diagonal.
        if constexpr (rune_myers_available_k && is_same_type<gap_costs_t, linear_gap_costs_t>::value)
            if (substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1)
                return rune_myers_t {}(first_utf32, second_utf32, result_ref, walker_scratch);

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

#pragma region Parallel Batch Algorithms

/**
 *  @brief Helper applying a single-pair scoring kernel across the @b cross-product of @p queries and
 *      @p candidates - every query against every candidate - reusing one dynamic scratch buffer. When
 *      @p is_symmetric, only the lower triangle (including the diagonal) is computed and mirrored.
 */
template <                         //
    typename score_type_,          //
    typename scoring_engine_type_, //
    typename queries_type_,        //
    typename candidates_type_,     //
    typename results_type_,        //
    typename scratch_buffer_type_  //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_>
#endif
status_t cross_sequentially_( //
    scoring_engine_type_ &&scoring, queries_type_ const &queries, candidates_type_ const &candidates,
    results_type_ &&results, cross_similarities_t cross_kind, scratch_buffer_type_ &scratch_buffer,
    cpu_specs_t const &specs) noexcept {

    using score_t = score_type_;
    bool const is_symmetric = cross_kind == cross_similarities_t::symmetric_k;

    size_t const queries_count = queries.size();
    size_t const candidates_count = candidates.size();

    // One shared scratch buffer sized to the largest (query, candidate) pair. The DP scratch is monotonic
    // non-decreasing in both lengths, so the longest query against the longest candidate maximizes it.
    size_t max_memory_requirement = 0;
    if (queries_count != 0 && candidates_count != 0) {
        size_t longest_query_index = 0, longest_candidate_index = 0;
        for (size_t query_index = 1; query_index < queries_count; ++query_index)
            if (queries[query_index].size() > queries[longest_query_index].size()) longest_query_index = query_index;
        for (size_t candidate_index = 1; candidate_index < candidates_count; ++candidate_index)
            if (candidates[candidate_index].size() > candidates[longest_candidate_index].size())
                longest_candidate_index = candidate_index;
        max_memory_requirement = scoring.scratch_space_needed(to_view(queries[longest_query_index]),
                                                              to_view(candidates[longest_candidate_index]), specs);
    }
    // The caller owns `scratch_buffer` (a grow-only engine member reused across calls); size it here.
    if (status_t status = scratch_buffer.try_resize(max_memory_requirement); status != status_t::success_k)
        return status;

    // Walk the grid, reusing the scratch space; mirror across the diagonal in the symmetric case.
    for (size_t query_index = 0; query_index < queries_count; ++query_index) {
        size_t const candidate_end = is_symmetric ? query_index + 1 : candidates_count;
        for (size_t candidate_index = 0; candidate_index < candidate_end; ++candidate_index) {
            score_t result_score = 0;
            // Pass the dummy by lvalue so the single-pair `operator()` instantiates on the bare executor type.
            dummy_executor_t dummy_executor;
            status_t status = scoring(to_view(queries[query_index]), to_view(candidates[candidate_index]), result_score,
                                      scratch_space_t(scratch_buffer), dummy_executor, specs);
            if (status != status_t::success_k) return status;
            results.data[query_index * results.row_stride + candidate_index] = result_score;
            if (is_symmetric && candidate_index != query_index)
                results.data[candidate_index * results.row_stride + query_index] = result_score;
        }
    }
    return status_t::success_k;
}

/**
 *  @brief Helper applying a single-pair scoring kernel across the @b cross-product of @p queries and
 *      @p candidates, differentiating multi-threaded and single-threaded cases. For very large pairs, all
 *      cores cooperate on one cell maximizing cache hits; for smaller pairs, each core computes its own cell.
 *      When @p is_symmetric, only the lower triangle (including the diagonal) is computed and mirrored.
 */
template <                                     //
    typename score_type_,                      //
    typename scoring_engine_type_,             //
    typename queries_type_,                    //
    typename candidates_type_,                 //
    typename results_type_,                    //
    typename scratch_buffer_type_,             //
    typename executor_type_ = dummy_executor_t //
    >
#if SZ_HAS_CONCEPTS_
    requires score_like<score_type_> && executor_like<executor_type_>
#endif
status_t cross_in_parallel_(                                         //
    scoring_engine_type_ &&scoring,                                  //
    queries_type_ const &queries,                                    //
    candidates_type_ const &candidates,                              //
    results_type_ &&results, cross_similarities_t cross_kind,        //
    scratch_buffer_type_ &scratch_buffer, executor_type_ &&executor, //
    cpu_specs_t const &specs) noexcept {

    using score_t = score_type_;
    using executor_t = typename std::decay<executor_type_>::type;
    using prong_t = typename executor_t::prong_t;
    bool const is_symmetric = cross_kind == cross_similarities_t::symmetric_k;

    size_t const queries_count = queries.size();
    size_t const candidates_count = candidates.size();
    if (queries_count == 0 || candidates_count == 0) return status_t::success_k;

    // Estimate memory for both single-threaded processing of small cells across all cores and cooperative
    // processing of large cells. Scan only the live cells (the lower triangle when symmetric).
    size_t max_memory_per_small = 0, max_memory_for_large = 0;
    auto const is_small = [&specs](size_t query_length, size_t candidate_length) noexcept {
        return std::min(query_length, candidate_length) <= specs.l1_bytes;
    };
    for (size_t query_index = 0; query_index < queries_count; ++query_index) {
        size_t const candidate_end = is_symmetric ? query_index + 1 : candidates_count;
        for (size_t candidate_index = 0; candidate_index < candidate_end; ++candidate_index) {
            size_t const needed = scoring.scratch_space_needed(to_view(queries[query_index]),
                                                               to_view(candidates[candidate_index]), specs);
            if (is_small(queries[query_index].size(), candidates[candidate_index].size()))
                max_memory_per_small = std::max(max_memory_per_small, needed);
            else max_memory_for_large = std::max(max_memory_for_large, needed);
        }
    }
    size_t const threads_count = executor.threads_count();
    size_t const max_memory_requirement = std::max(max_memory_per_small * threads_count, max_memory_for_large);
    // The caller owns `scratch_buffer` (a grow-only engine member reused across calls); size it here.
    if (status_t status = scratch_buffer.try_resize(max_memory_requirement); status != status_t::success_k)
        return status;

    std::atomic<status_t> error {status_t::success_k};

    auto const write_cell = [&](size_t query_index, size_t candidate_index, score_t result_score) noexcept {
        results.data[query_index * results.row_stride + candidate_index] = result_score;
        if (is_symmetric && candidate_index != query_index)
            results.data[candidate_index * results.row_stride + query_index] = result_score;
    };

    // Small cells: one per thread, dynamically scheduled over the flattened rectangle for load balance. The
    // symmetric upper triangle is skipped here (it is filled by mirroring the lower triangle).
    size_t const flattened_cells = queries_count * candidates_count;
    executor.for_n_dynamic(flattened_cells, [&](prong_t prong) noexcept {
        if (error.load() != status_t::success_k) return;
        size_t const query_index = prong.task / candidates_count;
        size_t const candidate_index = prong.task % candidates_count;
        if (is_symmetric && candidate_index > query_index) return;
        if (!is_small(queries[query_index].size(), candidates[candidate_index].size())) return;
        score_t result_score = 0;
        scratch_space_t worker_scratch = scratch_space_t(scratch_buffer).part_i_of_n(prong.thread, threads_count);
        dummy_executor_t dummy_executor; // Lvalue, so the scorer pins the bare executor type.
        status_t status = scoring(to_view(queries[query_index]), to_view(candidates[candidate_index]), result_score,
                                  worker_scratch, dummy_executor, specs);
        if (status == status_t::success_k) write_cell(query_index, candidate_index, result_score);
        else error.store(status);
    });

    // Large cells: all cores cooperate on one cell at a time, so only this thread allocates.
    for (size_t query_index = 0; query_index < queries_count && error.load() == status_t::success_k; ++query_index) {
        size_t const candidate_end = is_symmetric ? query_index + 1 : candidates_count;
        for (size_t candidate_index = 0; candidate_index < candidate_end; ++candidate_index) {
            if (is_small(queries[query_index].size(), candidates[candidate_index].size())) continue;
            score_t result_score = 0;
            status_t status = scoring(to_view(queries[query_index]), to_view(candidates[candidate_index]), result_score,
                                      scratch_space_t(scratch_buffer), executor, specs);
            if (status != status_t::success_k) {
                error.store(status);
                break;
            }
            write_cell(query_index, candidate_index, result_score);
        }
    }
    return error.load();
}

#pragma region Shared Candidate Lane Cross Product Driver

/**
 *  @brief A destination for one scored cell: the primary matrix slot, plus an optional mirror slot for the symmetric
 *      self-similarity case (lower triangle scored once, written to both `[i][j]` and `[j][i]`).
 */
template <typename value_type_>
struct cross_cell_destination_t {
    value_type_ *primary = nullptr;
    value_type_ *mirror = nullptr;
};

/** @brief Triangular number `T(n) = n*(n+1)/2`: the count of lower-triangle cells (including the diagonal) across the
 *         first @p rows rows of a symmetric self-similarity grid. */
SZ_INLINE size_t triangular_number_(size_t rows) noexcept { return rows * (rows + 1) / 2; }

/** @brief The number of live cells: the full rectangle, or the lower triangle (incl. diagonal) when symmetric. */
SZ_INLINE size_t cross_live_cells_count_(size_t queries_count, size_t candidates_count,
                                         cross_similarities_t cross_kind) noexcept {
    if (cross_kind == cross_similarities_t::symmetric_k) return triangular_number_(queries_count);
    return queries_count * candidates_count;
}

/** @brief Decodes a flat live-cell index into its `(query_index, candidate_index)` grid coordinates. */
SZ_INLINE void cross_cell_to_indices_(size_t cell_index, size_t candidates_count, cross_similarities_t cross_kind,
                                      size_t &query_index, size_t &candidate_index) noexcept {
    if (cross_kind == cross_similarities_t::symmetric_k) {
        // The row containing the flat cell is the largest `row` with `T(row) <= cell_index`; the column is the
        // remainder past that row's triangular base.
        size_t row = 0;
        while (triangular_number_(row + 1) <= cell_index) ++row;
        query_index = row;
        candidate_index = cell_index - triangular_number_(row);
    }
    else {
        query_index = cell_index / candidates_count;
        candidate_index = cell_index % candidates_count;
    }
}

/**
 *  @brief Dyadic length bucket of a candidate: `bit_width(length - 1)`. Two candidates in one bucket differ in length
 *      by less than 2x, so packing a lane block from a single bucket bounds the transpose zero-padding waste - the
 *      lane-fill tiling that turns ragged rows into dense kernels (R6).
 */
SZ_INLINE int candidate_length_bucket_(size_t length) noexcept {
    return length <= 1 ? 0 : (int)(64 - sz_u64_clz((sz_u64_t)(length - 1)));
}

/**
 *  @brief Host-side orchestration loop shared by every @b byte candidate-lane engine (Needleman-Wunsch,
 *      Smith-Waterman, and non-unit Levenshtein, across haswell / icelake / neon). Contains @b no SIMD and @b no
 *      kernel code: it walks the live cells `[cell_begin, cell_end)` one query-row at a time, groups each row's
 *      candidates into @b dyadic length buckets so a block's lanes carry similar lengths (minimal zero-pad),
 *      transposes each block into column-major lane order, dispatches it to the per-backend @p kernel, and scatters
 *      the lane results into the strided matrix. Empty cells and cells whose worst-case score escapes the kernel's
 *      range are scored individually through the per-pair @p fallback. The @p kernel and @p fallback are passed by
 *      reference; nothing is shared at the kernel level - per the design, only the host driver is consolidated.
 *
 *  @param fits `(query_length, candidate_length) -> bool`: whether a cell's worst-case score fits the kernel's range.
 *  @param empty_cell `(query_length, candidate_length) -> score`: the all-gap score for a degenerate (empty) cell.
 */
template <typename narrow_kernel_type_, typename wide_kernel_type_, typename fallback_type_, typename queries_type_,
          typename candidates_type_, typename results_type_, typename fits_narrow_type_, typename fits_wide_type_,
          typename empty_cell_type_>
status_t cross_product_candidate_lanes_range_( //
    narrow_kernel_type_ &narrow_kernel, wide_kernel_type_ &wide_kernel, fallback_type_ &fallback,
    queries_type_ const &queries, candidates_type_ const &candidates, results_type_ &&results,
    cross_similarities_t cross_kind, size_t cell_begin, size_t cell_end, fits_narrow_type_ &&fits_narrow,
    fits_wide_type_ &&fits_wide, empty_cell_type_ &&empty_cell, scratch_space_t scratch,
    cpu_specs_t const &specs) noexcept {

    using narrow_t = remove_cvref<narrow_kernel_type_>;
    using wide_t = remove_cvref<wide_kernel_type_>;
    using element_t = typename narrow_t::char_t; // ? `char` for byte engines, `rune_t` for the UTF-8 engines.
    using value_t = remove_cvref<decltype(results.data[0])>;
    // The narrow kernel has the most lanes (narrower cells), so its lane count bounds the transpose stride and the
    // per-block staging arrays; the wide kernel handles cells whose score escapes the narrow range but fits the wide.
    constexpr size_t narrow_lanes_k = narrow_t::candidate_lanes_k;
    // The per-pair fallback writes its native score type (unsigned distance for minimization, signed score for
    // maximization), which may differ from the result matrix's `value_t`; score into the fallback's own type.
    using fallback_score_t =
        typename std::conditional<narrow_t::objective_k == sz_minimize_distance_k, size_t, ssize_t>::type;

    bool const is_symmetric = cross_kind == cross_similarities_t::symmetric_k;
    size_t const candidates_count = candidates.size();

    // Size the transpose buffer to the widest lane block (narrow kernel) and the walker arena to the larger of the
    // two kernels' needs, over the longest candidate any batchable cell can present.
    size_t longest_candidate = 0;
    for (size_t cell_index = cell_begin; cell_index != cell_end; ++cell_index) {
        size_t query_index = 0, candidate_index = 0;
        cross_cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
        size_t const query_length = to_view(queries[query_index]).size();
        size_t const candidate_length = to_view(candidates[candidate_index]).size();
        if (query_length != 0 && candidate_length != 0 && fits_wide(query_length, candidate_length))
            longest_candidate = sz_max_of_two(longest_candidate, candidate_length);
    }
    size_t const transpose_bytes = narrow_lanes_k * longest_candidate * sizeof(element_t);
    size_t const walker_scratch = longest_candidate
                                      ? sz_max_of_two(narrow_kernel.scratch_space_needed(longest_candidate, specs),
                                                      wide_kernel.scratch_space_needed(longest_candidate, specs))
                                      : 0;
    element_t *transposed = reinterpret_cast<element_t *>(scratch.data());
    scratch_space_t walker_scratch_space = scratch.subspan(transpose_bytes, walker_scratch);
    scratch_space_t fallback_scratch_space = scratch;

    auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
        cross_cell_destination_t<value_t> destination;
        destination.primary = results.data + query_index * results.row_stride + candidate_index;
        if (is_symmetric && candidate_index != query_index)
            destination.mirror = results.data + candidate_index * results.row_stride + query_index;
        return destination;
    };
    auto const scatter = [&](cross_cell_destination_t<value_t> const &destination, value_t score) noexcept {
        *destination.primary = score;
        if (destination.mirror) *destination.mirror = score;
    };

    dummy_executor_t dummy;
    size_t lengths[narrow_lanes_k];
    size_t block_candidates[narrow_lanes_k];
    cross_cell_destination_t<value_t> destinations[narrow_lanes_k];

    // Row state, refreshed per query-row and read by the emit/tier closures below.
    span<element_t const> query;
    size_t query_length = 0, seed_query_index = 0, row_base = 0, cell_index = cell_begin, row_end = cell_begin;

    // Transpose, dispatch, and scatter one full or partial lane block through the given kernel (narrow or wide).
    auto const emit_block = [&](auto &chosen_kernel, size_t lanes_count, size_t block_longest) noexcept {
        using chosen_t = remove_cvref<decltype(chosen_kernel)>;
        constexpr size_t lane_capacity = chosen_t::candidate_lanes_k;
        typename chosen_t::score_t result_lanes[narrow_t::candidate_lanes_k];
        for (size_t position = 0; position != lane_capacity * block_longest; ++position) transposed[position] = 0;
        for (size_t lane_index = 0; lane_index != lanes_count; ++lane_index) {
            auto const lane_candidate = to_view(candidates[block_candidates[lane_index]]);
            for (size_t position = 0; position < lane_candidate.size(); ++position)
                transposed[position * lane_capacity + lane_index] = lane_candidate[position];
        }
        candidate_lanes_block<element_t> block;
        block.transposed = transposed;
        block.lane_capacity = lane_capacity;
        block.lanes_count = lanes_count;
        block.lengths = lengths;
        block.longest_candidate = block_longest;
        status_t status = chosen_kernel(query, block, result_lanes, walker_scratch_space, specs);
        if (status != status_t::success_k) return status;
        for (size_t lane_index = 0; lane_index != lanes_count; ++lane_index)
            scatter(destinations[lane_index], static_cast<value_t>(result_lanes[lane_index]));
        return status_t::success_k;
    };

    // Score every cell of the current row that belongs to one width tier (`in_tier(candidate_length)`), grouping by
    // dyadic length bucket so each block's lanes carry similar lengths, and dispatching through `chosen_kernel`.
    auto const run_tier = [&](auto &chosen_kernel, auto &&in_tier, size_t lane_capacity) noexcept {
        int max_bucket = -1;
        for (size_t r = cell_index; r != row_end; ++r) {
            size_t const candidate_length = to_view(candidates[r - row_base]).size();
            if (candidate_length == 0 || !in_tier(candidate_length)) continue;
            int const bucket = candidate_length_bucket_(candidate_length);
            if (bucket > max_bucket) max_bucket = bucket;
        }
        for (int bucket = 0; bucket <= max_bucket; ++bucket) {
            size_t lanes_count = 0, block_longest = 0;
            for (size_t r = cell_index; r != row_end; ++r) {
                size_t const candidate_index = r - row_base;
                size_t const candidate_length = to_view(candidates[candidate_index]).size();
                if (candidate_length == 0 || !in_tier(candidate_length)) continue;
                if (candidate_length_bucket_(candidate_length) != bucket) continue;
                block_candidates[lanes_count] = candidate_index;
                lengths[lanes_count] = candidate_length;
                destinations[lanes_count] = destination_for(seed_query_index, candidate_index);
                block_longest = sz_max_of_two(block_longest, candidate_length);
                ++lanes_count;
                if (lanes_count == lane_capacity) {
                    if (status_t status = emit_block(chosen_kernel, lanes_count, block_longest);
                        status != status_t::success_k)
                        return status;
                    lanes_count = 0, block_longest = 0;
                }
            }
            if (lanes_count)
                if (status_t status = emit_block(chosen_kernel, lanes_count, block_longest);
                    status != status_t::success_k)
                    return status;
        }
        return status_t::success_k;
    };

    while (cell_index != cell_end) {
        size_t seed_candidate_index = 0;
        cross_cell_to_indices_(cell_index, candidates_count, cross_kind, seed_query_index, seed_candidate_index);
        query = to_view(queries[seed_query_index]);
        query_length = query.size();
        // Cells of one query-row are contiguous in flat index for both layouts; derive the row base to map back.
        row_base = is_symmetric ? triangular_number_(seed_query_index) : seed_query_index * candidates_count;
        size_t const row_full_end = is_symmetric ? row_base + seed_query_index + 1 : row_base + candidates_count;
        row_end = sz_min_of_two(row_full_end, cell_end);

        // Degenerate (empty) cells and cells whose score escapes even the wide range are scored individually.
        for (size_t r = cell_index; r != row_end; ++r) {
            size_t const candidate_index = r - row_base;
            size_t const candidate_length = to_view(candidates[candidate_index]).size();
            if (query_length == 0 || candidate_length == 0) {
                scatter(destination_for(seed_query_index, candidate_index),
                        static_cast<value_t>(empty_cell(query_length, candidate_length)));
                continue;
            }
            if (!fits_wide(query_length, candidate_length)) {
                fallback_score_t result_score = 0;
                if (status_t status = fallback(query, to_view(candidates[candidate_index]), result_score,
                                               fallback_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(seed_query_index, candidate_index), static_cast<value_t>(result_score));
            }
        }

        // Narrow tier (fits the narrow range), then the wide tier (escapes narrow but fits wide).
        auto const fits_narrow_tier = [&](size_t candidate_length) noexcept {
            return fits_narrow(query_length, candidate_length);
        };
        auto const fits_wide_tier = [&](size_t candidate_length) noexcept {
            return !fits_narrow(query_length, candidate_length) && fits_wide(query_length, candidate_length);
        };
        if (status_t status = run_tier(narrow_kernel, fits_narrow_tier, narrow_t::candidate_lanes_k);
            status != status_t::success_k)
            return status;
        if (status_t status = run_tier(wide_kernel, fits_wide_tier, wide_t::candidate_lanes_k);
            status != status_t::success_k)
            return status;
        cell_index = row_end;
    }
    return status_t::success_k;
}

/**
 *  @brief Worst-case single-worker scratch for the shared candidate-lane driver, in O(Q+C): the transposed block plus
 *      the lane-walker arena for the longest candidate, or the per-pair fallback arena for the longest cell.
 */
template <typename narrow_kernel_type_, typename wide_kernel_type_, typename fallback_type_, typename queries_type_,
          typename candidates_type_, typename fits_wide_type_>
size_t cross_product_candidate_lanes_scratch_( //
    narrow_kernel_type_ &narrow_kernel, wide_kernel_type_ &wide_kernel, fallback_type_ &fallback,
    queries_type_ const &queries, candidates_type_ const &candidates, fits_wide_type_ &&fits_wide,
    cpu_specs_t const &specs) noexcept {

    constexpr size_t narrow_lanes_k = remove_cvref<narrow_kernel_type_>::candidate_lanes_k;
    using element_t = typename remove_cvref<narrow_kernel_type_>::char_t;

    size_t longest_query = 0, longest_query_index = 0, longest_candidate = 0, longest_candidate_index = 0;
    for (size_t index = 0; index < queries.size(); ++index)
        if (to_view(queries[index]).size() > longest_query)
            longest_query = to_view(queries[index]).size(), longest_query_index = index;
    for (size_t index = 0; index < candidates.size(); ++index)
        if (to_view(candidates[index]).size() > longest_candidate)
            longest_candidate = to_view(candidates[index]).size(), longest_candidate_index = index;
    size_t const transpose_bytes = narrow_lanes_k * longest_candidate * sizeof(element_t);
    size_t const walker_scratch = longest_candidate
                                      ? sz_max_of_two(narrow_kernel.scratch_space_needed(longest_candidate, specs),
                                                      wide_kernel.scratch_space_needed(longest_candidate, specs))
                                      : 0;
    size_t fallback_scratch = 0;
    if (queries.size() && candidates.size() && !fits_wide(longest_query, longest_candidate))
        fallback_scratch = fallback.scratch_space_needed(to_view(queries[longest_query_index]),
                                                         to_view(candidates[longest_candidate_index]), specs);
    return sz_max_of_two(transpose_bytes + walker_scratch, fallback_scratch);
}

/**
 *  @brief Parallel wrapper for the shared candidate-lane driver: sizes the engine-owned @p scratch_buffer for all
 *      workers, then hands each worker a contiguous live-cell slice and its own scratch partition.
 */
template <typename narrow_kernel_type_, typename wide_kernel_type_, typename fallback_type_, typename queries_type_,
          typename candidates_type_, typename results_type_, typename scratch_buffer_type_, typename executor_type_,
          typename fits_narrow_type_, typename fits_wide_type_, typename empty_cell_type_>
status_t cross_product_candidate_lanes_parallel_( //
    narrow_kernel_type_ &narrow_kernel, wide_kernel_type_ &wide_kernel, fallback_type_ &fallback,
    queries_type_ const &queries, candidates_type_ const &candidates, results_type_ &&results,
    cross_similarities_t cross_kind, scratch_buffer_type_ &scratch_buffer, executor_type_ &&executor,
    fits_narrow_type_ &&fits_narrow, fits_wide_type_ &&fits_wide, empty_cell_type_ &&empty_cell,
    cpu_specs_t const &specs) noexcept {

    size_t const cells_count = cross_live_cells_count_(queries.size(), candidates.size(), cross_kind);
    size_t const worker_scratch = cross_product_candidate_lanes_scratch_(narrow_kernel, wide_kernel, fallback, queries,
                                                                         candidates, fits_wide, specs);
    size_t const workers = sz_max_of_two(sz_min_of_two(executor.threads_count(), cells_count), (size_t)1);
    if (status_t status = scratch_buffer.try_resize(worker_scratch * workers); status != status_t::success_k)
        return status;
    std::atomic<size_t> next_worker {0};
    std::atomic<status_t> error {status_t::success_k};
    executor.for_slices(cells_count, [&](size_t cell_begin, size_t length) noexcept {
        if (length == 0) return; // empty slice: no work, and it must not consume a scratch partition
        size_t const worker = next_worker.fetch_add(1, std::memory_order_relaxed);
        scratch_space_t slice = scratch_space_t(scratch_buffer).subspan(worker * worker_scratch, worker_scratch);
        status_t status = cross_product_candidate_lanes_range_(
            narrow_kernel, wide_kernel, fallback, queries, candidates, results, cross_kind, cell_begin,
            cell_begin + length, fits_narrow, fits_wide, empty_cell, slice, specs);
        if (status != status_t::success_k) error.store(status);
    });
    return error.load();
}

#pragma endregion Shared Candidate Lane Cross Product Driver

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

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    safe_vector<std::byte, scratch_allocator_t> scratch_ {alloc_}; // grow-only cross-product scratch, reused

    levenshtein_distances(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances(uniform_substitution_costs_t subs, gap_costs_t gaps,
                          allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    // The concrete `strided_rows<value_type_>` parameter disambiguates the two-set and symmetric overloads by type
    // alone, since `SZ_HAS_CONCEPTS_` is off repo-wide for the GCC concepts bug.
    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                           cross_similarities_t::all_pairs_k, scratch_, specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                          cross_similarities_t::all_pairs_k, scratch_, executor, specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                           cross_similarities_t::symmetric_k, scratch_, specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                          cross_similarities_t::symmetric_k, scratch_, executor, specs);
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

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    safe_vector<std::byte, scratch_allocator_t> scratch_ {alloc_}; // grow-only cross-product scratch, reused

    levenshtein_distances_utf8(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances_utf8(uniform_substitution_costs_t subs, gap_costs_t gaps,
                               allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                           cross_similarities_t::all_pairs_k, scratch_, specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                          cross_similarities_t::all_pairs_k, scratch_, executor, specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                           cross_similarities_t::symmetric_k, scratch_, specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                          cross_similarities_t::symmetric_k, scratch_, executor, specs);
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

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    safe_vector<std::byte, scratch_allocator_t> scratch_ {alloc_}; // grow-only cross-product scratch, reused

    needleman_wunsch_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_scores(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        return cross_sequentially_<ssize_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                            cross_similarities_t::all_pairs_k, scratch_, specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_in_parallel_<ssize_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                           cross_similarities_t::all_pairs_k, scratch_, executor, specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_sequentially_<ssize_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                            cross_similarities_t::symmetric_k, scratch_, specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return cross_in_parallel_<ssize_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                           cross_similarities_t::symmetric_k, scratch_, executor, specs);
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

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    safe_vector<std::byte, scratch_allocator_t> scratch_ {alloc_}; // grow-only cross-product scratch, reused

    smith_waterman_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    smith_waterman_scores(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        return cross_sequentially_<ssize_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                            cross_similarities_t::all_pairs_k, scratch_, specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                                 strided_rows<value_type_> results, executor_type_ &&executor,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_in_parallel_<ssize_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                           cross_similarities_t::all_pairs_k, scratch_, executor, specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 cpu_specs_t const &specs = {}) noexcept {
        return cross_sequentially_<ssize_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                            cross_similarities_t::symmetric_k, scratch_, specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    SZ_NOIPA status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                                 executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return cross_in_parallel_<ssize_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                           cross_similarities_t::symmetric_k, scratch_, executor, specs);
    }
};

#pragma endregion

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_SERIAL_HPP_
